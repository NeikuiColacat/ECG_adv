"""CPU checks of paired inference packing; GPU generation needs live admission."""
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch

from util.evaluation.pulse_adapters import decode_generated, pack_shared_prompt_features


@pytest.mark.parametrize("fail", [False, True])
def test_torch_profile_stages_preserve_outputs_and_remove_hooks(monkeypatch, tmp_path, fail):
    from util.evaluation.pulse_profile import profile_generation

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_projector = torch.nn.Linear(4, 4)

        def forward(self, *, inputs_embeds):
            return inputs_embeds + 1

    backbone = Backbone()
    vision, head = torch.nn.Linear(4, 4), torch.nn.Linear(4, 7)
    base = SimpleNamespace(get_model=lambda: backbone, get_vision_tower=lambda: vision, lm_head=head)
    backend = SimpleNamespace(base=base)

    def generate(views):
        x = backbone.mm_projector(vision(views[0]))
        head(backbone(inputs_embeds=x))
        ids = head(backbone(inputs_embeds=x[:, :1])).argmax(-1).tolist()
        if fail:
            raise RuntimeError("probe failure")
        backend.last_view_token_ids = ids
        return ids

    backend.generate_views = generate
    original_profile = torch.profiler.profile
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs:
        original_profile(activities=[torch.profiler.ProfilerActivity.CPU]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    views = [torch.ones(1, 3, 4)]
    if fail:
        with pytest.raises(RuntimeError, match="probe failure"):
            profile_generation(backend, views, tmp_path)
    else:
        expected = generate(views)
        answers, tokens, summary = profile_generation(backend, views, tmp_path)
        assert answers == tokens == expected
        assert {e["name"] for e in summary["stages"]} == {
            "pulse/vision", "pulse/projector", "pulse/llm_prefill", "pulse/llm_decode", "pulse/lm_head"}
        assert not summary["cuda_trace_available"]
        assert (tmp_path / "torch_trace.json").is_file()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in (backbone, vision, head, backbone.mm_projector))


@pytest.mark.parametrize("fail", [False, True])
@torch.inference_mode()
def test_fp16_inference_storage_restores_exact_reference_after_arm_writes(fail):
    from util.evaluation.pulse_adapters import HybridPulseBackend
    backend = HybridPulseBackend.__new__(HybridPulseBackend)
    backend.fp16_adapters, backend.bypass_original_lora = True, False
    parameter = torch.nn.Parameter(torch.tensor([1.125, -2.75]), requires_grad=False)
    backend.parameters = {"layer.lora_A.default.weight": parameter}
    reference, pointer = parameter.clone(), parameter.data_ptr()
    states = [torch.tensor([0.125, 0.3]), torch.tensor([-1.3, 2.5])]

    def generate(views, *, bypass_original_lora):
        assert parameter.dtype == torch.float16 and not torch.is_grad_enabled()
        result = []
        for state in states:
            torch._foreach_copy_([parameter], [state])
            result.append(parameter.clone())
        if fail:
            raise RuntimeError("probe failure")
        return result

    backend._generate_views = generate
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="probe failure"):
                backend.generate_views([None])
        else:
            assert all(torch.equal(a, b.half()) for a, b in zip(backend.generate_views([None]), states))
        assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer
        assert torch.equal(parameter, reference)


@pytest.mark.parametrize("value", [None, 0.1, float("nan")])
def test_original_lora_bypass_rejects_missing_or_nonzero_factors(value):
    from util.evaluation.pulse_adapters import HybridPulseBackend
    backend = HybridPulseBackend.__new__(HybridPulseBackend)
    backend.states = {"original": {} if value is None else {"layer.lora_B.default.weight": torch.tensor([value])}}
    with pytest.raises(ValueError, match="exactly zero"):
        backend._validate_original_lora()
    backend.states["original"] = {"layer.lora_B.default.weight": torch.zeros(2, 3)}
    backend._validate_original_lora()


@pytest.mark.parametrize("fail", [False, True])
@torch.inference_mode()
def test_original_lora_bypass_skips_only_zero_arm_and_restores_frozen_state(fail):
    peft = pytest.importorskip("peft")
    from util.evaluation.pulse_adapters import HybridPulseBackend
    torch.manual_seed(17)
    backend = HybridPulseBackend.__new__(HybridPulseBackend)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7))
    backend.model = peft.get_peft_model(model, peft.LoraConfig(r=2, target_modules=["0"], lora_alpha=4)).eval()
    backend.model.requires_grad_(False)
    backend.states = {"original": {n: p.clone() for n, p in backend.model.named_parameters()}}
    backend._validate_original_lora()
    before = {n: p.clone() for n, p in backend.model.named_parameters()}
    layer = backend.model.base_model.model[0]
    x = torch.randn(2, 11, 5)
    expected = backend.model(x)
    calls = []
    hook = layer.lora_A["default"].register_forward_hook(lambda *args: calls.append(True))
    def run():
        with backend._without_original_lora():
            assert layer.disable_adapters
            assert torch.equal(backend.model(x), expected)
            assert not calls
            if fail:
                raise RuntimeError("probe failure")
    if fail:
        with pytest.raises(RuntimeError, match="probe failure"):
            run()
    else:
        run()
    assert not layer.disable_adapters and not layer.merged
    assert all(not p.requires_grad and torch.equal(p, before[n]) for n, p in backend.model.named_parameters())
    # Simulate selecting a trained arm after original; its update must execute.
    layer.lora_B["default"].weight.fill_(0.1)
    assert not torch.equal(backend.model(x), expected)
    assert calls == [True]
    hook.remove()
    with backend.model.disable_adapter(), pytest.raises(ValueError, match="enabled, unmerged"):
        with backend._without_original_lora():
            pass


def test_profile_timing_restores_methods_and_existing_overrides():
    from util.evaluation.pulse_profile import timed_calls
    class Target:
        def method(self, value):
            return value + 1
    target = Target()
    with timed_calls([(target, "method", "method")], lambda: None) as timings:
        assert target.method(3) == 4
    assert "method" not in target.__dict__ and len(timings["method"]) == 1
    target.method = lambda value: value * 2
    original = target.method
    with pytest.raises(RuntimeError):
        with timed_calls([(target, "method", "method")], lambda: None):
            assert target.method(3) == 6
            raise RuntimeError("probe failure")
    assert target.method is original


@torch.inference_mode()
def test_original_bypass_accepts_transformers_adapter_mixin_and_preserves_tokens():
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from util.evaluation.pulse_adapters import HybridPulseBackend
    config = transformers.LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=None)
    model = peft.get_peft_model(transformers.LlamaForCausalLM(config),
        peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    backend = HybridPulseBackend.__new__(HybridPulseBackend)
    backend.model = model.eval().requires_grad_(False)
    ids = torch.tensor([[1, 3, 9, 17]])
    kwargs = dict(do_sample=False, max_new_tokens=4, pad_token_id=0)
    expected = model.generate(ids, **kwargs)
    with backend._without_original_lora():
        assert torch.equal(model.generate(ids, **kwargs), expected)
    assert torch.equal(model.generate(ids, **kwargs), expected)
    assert not any(p.requires_grad for p in model.parameters())


def test_engineering_profile_config_has_managed_result(tmp_path):
    from pathlib import Path
    from boot_scripts.run_experiment import load_experiment_plan
    repo = Path(__file__).resolve().parents[2]
    plan = load_experiment_plan(repo / "configs/experiments/pulse_adapter_profile.yaml", run_dir=tmp_path / "profile")
    assert plan.entrypoint_name == "profile_pulse_adapters"
    assert plan.expected_result_type == "pulse_profile_result"


def test_logit_distance_formulas_and_softmax_shift_invariance():
    from util.evaluation.pulse_profile import logit_distances
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    zero = logit_distances(a, a.clone())
    assert zero["logit_l2"] == [0., 0.]
    assert zero["probability_l2"] == [0., 0.]
    assert zero["kl_reference_candidate"] == [0., 0.]
    assert zero["logits_bitwise_equal"] == [True, True]
    shifted = logit_distances(a, a + 5)
    assert shifted["logit_l2"] == pytest.approx([75 ** .5] * 2)
    assert shifted["logit_rmse"] == pytest.approx([5., 5.])
    assert shifted["relative_logit_l2"] == pytest.approx([75 ** .5 / 14 ** .5, 75 ** .5 / 77 ** .5])
    assert shifted["centered_logit_l2"] == pytest.approx([0., 0.])
    assert max(shifted["probability_total_variation"]) < 1e-14
    assert shifted["argmax_equal"] == [True, True]
    changed = logit_distances(a, a.flip(-1))
    assert changed["argmax_equal"] == [False, False]
    assert min(changed["kl_reference_candidate"]) > 0


def test_logit_distance_rejects_bad_inputs_and_launcher_rejects_unknown_suite():
    from util.evaluation.pulse_profile import logit_distances
    from boot_scripts.run_experiment import _validated_arguments
    with pytest.raises(ValueError, match="shapes"):
        logit_distances(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="nonfinite"):
        logit_distances(torch.zeros(2, 3), torch.full((2, 3), float("nan")))
    assert _validated_arguments("profile_pulse_adapters", ["--suite", "logits"]) == ("--suite", "logits")
    assert _validated_arguments("profile_pulse_adapters", ["--suite", "deployment"]) == ("--suite", "deployment")
    assert _validated_arguments("profile_pulse_adapters", ["--suite", "deployment_cache"]) == ("--suite", "deployment_cache")
    with pytest.raises(ValueError, match="diagnostic suite"):
        _validated_arguments("profile_pulse_adapters", ["--suite", "unknown"])


def test_deployment_answer_tokens_exclude_bos_and_post_eos_padding():
    from util.evaluation.pulse_deployment import answer_tokens
    assert answer_tokens(torch.tensor([[1, 5, 2, 2], [1, 6, 7, 8]]), 1, 2) == [[5, 2], [6, 7, 8]]
    with pytest.raises(ValueError, match="BOS"):
        answer_tokens(torch.tensor([[3, 4]]), 1, 2)


def test_deployment_admission_gate_rejects_incomplete_or_different_outputs():
    from copy import deepcopy
    from util.evaluation.pulse_deployment import admission_verdict
    conditions = [str(i) for i in range(21)]
    rows = [{"arm": arm, "condition_id": condition, "mode": mode, "batch_size": batch,
        "cache_variant": cache, "status": "complete", "records": 8,
        "answer_tokens_equal": [8, 8], "repeat_tokens_stable": True,
        "parsed_answers_equal": 8, "eos_and_length_equal": 8,
        "skipped_divergent_prefix_positions": 0, "compared_identical_prefix_positions": 8,
        "author_reference_verified": mode == "unmerged_reference"}
        for arm in ("w1", "w2") for condition in conditions
        for mode, batch, cache in (("unmerged_reference", 2, "legacy"), ("merged_unloaded", 3, "mutable"))]
    assert admission_verdict(rows, conditions)["status"] == "passed"
    for field, value in [("status", "out_of_memory"), ("answer_tokens_equal", [8, 7]),
                         ("parsed_answers_equal", 7), ("eos_and_length_equal", 7),
                         ("author_reference_verified", False), ("repeat_tokens_stable", False),
                         ("skipped_divergent_prefix_positions", 1)]:
        changed = deepcopy(rows); changed[0][field] = value
        assert admission_verdict(changed, conditions)["status"] == "failed"
    assert admission_verdict(rows[:-1], conditions)["status"] == "failed"
    assert admission_verdict(rows + [rows[0]], conditions)["status"] == "failed"


def test_deployment_admission_launcher_requires_exact_center_selector(tmp_path):
    from pathlib import Path
    from boot_scripts.run_experiment import _validated_arguments, load_experiment_plan
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        args = ["--suite", "deployment_admission", "--center", center]
        assert _validated_arguments("profile_pulse_adapters", args) == tuple(args)
        plan = load_experiment_plan(Path(__file__).resolve().parents[2] /
            f"configs/experiments/pulse_deployment_admission_{center}.yaml", run_dir=tmp_path / center)
        assert plan.expected_result_type == "pulse_profile_result"
    for args in (["--suite", "deployment_admission"], ["--suite", "deployment", "--center", "ningbo"],
                 ["--suite", "deployment_admission", "--center", "bad"]):
        with pytest.raises(ValueError):
            _validated_arguments("profile_pulse_adapters", args)


def fake_base():
    model = SimpleNamespace(embed_tokens=torch.nn.Embedding(50, 8), mm_projector=torch.nn.Linear(1024, 8))
    return SimpleNamespace(get_model=lambda: model,
                           config=SimpleNamespace(mm_patch_merge_type="flat", tokenizer_model_max_length=4096))


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_shared_prompt_packing_matches_author_flat_batch_layout(batch):
    torch.manual_seed(61)
    base = fake_base()
    ids = torch.tensor([[1, 7, -200, 9, 6]]).expand(batch, -1)
    features = torch.randn(batch * 5, 576, 1024)
    projected = base.get_model().mm_projector(features).split(5)
    expected = []
    for row, visual in zip(ids, projected, strict=True):
        text = base.get_model().embed_tokens(torch.cat((row[:2], row[3:])))
        expected.append(torch.cat((text[:2], visual.flatten(0, 1), text[2:])))
    actual = pack_shared_prompt_features(base, ids, features)
    assert actual.shape == (batch, 2884, 8)
    assert torch.equal(actual, torch.stack(expected))


def test_different_prompts_and_wrong_tile_count_are_rejected():
    base = fake_base()
    with pytest.raises(ValueError, match="identical"):
        pack_shared_prompt_features(base, torch.tensor([[1, -200, 3], [1, -200, 4]]), torch.zeros(10, 576, 1024))
    with pytest.raises(ValueError, match="five"):
        pack_shared_prompt_features(base, torch.tensor([[1, -200, 3]]), torch.zeros(4, 576, 1024))


def test_output_lengths_exclude_dummy_bos_and_padding():
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, decode=lambda t, **kw: str(t.tolist()))
    rows = decode_generated(tokenizer, torch.tensor([[1, 8, 2, 2], [1, 4, 5, 6]]), max_new_tokens=3)
    assert rows[0]["generated_tokens"] == 2 and rows[0]["hit_max_new_tokens"] is False
    assert rows[1]["generated_tokens"] == 3 and rows[1]["hit_max_new_tokens"] is True


def test_adapter_switch_ends_previous_amp_cache_and_shares_vision(monkeypatch):
    from util.evaluation import pulse_adapters as module
    active, contexts, vision_calls = [], [], []
    @contextmanager
    def autocast(*args, **kwargs):
        assert not active
        contexts.append(len(contexts))
        active.append(contexts[-1])
        try:
            yield
        finally:
            active.pop()
    class Pixels:
        ndim, shape = 5, (1, 5, 3, 336, 336)
        def __len__(self): return 1
        def to(self, **kwargs): return self
        def flatten(self, *args): return self
    def vision(pixels):
        vision_calls.append(True)
        return torch.zeros(1)
    backend = object.__new__(module.PairedPulseBackend)
    backend.base = SimpleNamespace(get_vision_tower=lambda: vision)
    backend.prompt_ids = torch.tensor([[1, -200, 5]])
    backend.tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, decode=lambda t, **kw: str(t.tolist()))
    backend.max_new_tokens, backend.arm = 32, None
    def select(arm):
        assert not active, "adapter overwrite occurred within cached AMP context"
        backend.arm = arm
    backend._select = select
    backend._generate = lambda packed: torch.tensor([[1, 8 if backend.arm == "w1" else 9, 2]])
    monkeypatch.setattr(module.torch, "autocast", autocast)
    monkeypatch.setattr(module, "pack_shared_prompt_features", lambda *args: torch.zeros(1))
    outputs = backend.generate_pair(Pixels())
    assert len(vision_calls) == 1 and len(contexts) == 3
    assert outputs["w1"] != outputs["w2"]


def test_visual_lora_recomputes_vision_features_after_each_arm(monkeypatch):
    from util.evaluation import pulse_adapters as module
    from contextlib import nullcontext
    calls, active = [], []

    class Pixels:
        ndim, shape = 5, (1, 5, 3, 336, 336)
        def __len__(self): return 1
        def to(self, **kwargs): return self
        def flatten(self, *args): return self

    def vision(_pixels):
        calls.append(backend.arm)
        return torch.zeros(1)

    backend = object.__new__(module.PairedPulseBackend)
    backend.base = SimpleNamespace(get_vision_tower=lambda: vision)
    backend.prompt_ids = torch.tensor([[1, -200, 5]])
    backend.tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, decode=lambda t, **kw: str(t.tolist()))
    backend.max_new_tokens, backend.arm, backend.visual_lora = 32, None, True
    def select(arm): backend.arm = arm
    backend._select = select
    backend._generate = lambda packed: torch.tensor([[1, 8 if backend.arm == "w1" else 9, 2]])
    monkeypatch.setattr(module.torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(module, "pack_shared_prompt_features", lambda *args: torch.zeros(1))
    outputs = backend.generate_pair(Pixels())
    assert calls == ["w1", "w2"]
    assert outputs["w1"] != outputs["w2"]
