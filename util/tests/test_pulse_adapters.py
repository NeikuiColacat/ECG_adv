"""CPU checks of paired inference packing; GPU generation needs live admission."""
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch

from util.evaluation.pulse_adapters import decode_generated, pack_shared_prompt_features


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
