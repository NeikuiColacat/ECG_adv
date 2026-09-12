"""Small CPU causal-model checks, explicitly not PULSE GPU admission."""
import pytest
import torch

pytest.importorskip("transformers", reason="requires the dedicated PULSE environment")
from transformers import LlamaConfig, LlamaForCausalLM

from util.evaluation.pulse_generation_projection import last_token_projection, generation_cache


def tiny_model():
    torch.manual_seed(51)
    config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=None)
    model = LlamaForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("length", [7, 31])
@torch.inference_mode()
def test_cached_greedy_tokens_and_last_logits_match_cpu_reference(batch, length):
    model = tiny_model()
    ids = torch.randint(3, 97, (batch, length))
    original = model.lm_head.forward
    weights = model.lm_head.weight.clone()
    keys = list(model.state_dict())
    kwargs = {"do_sample": False, "max_new_tokens": 8, "use_cache": True, "pad_token_id": 0}
    expected = model.generate(ids, **kwargs)
    logits = model(ids).logits[:, -1:]
    with last_token_projection(model):
        reduced = model(ids).logits
        assert reduced.shape == (batch, 1, 97)
        torch.testing.assert_close(reduced, logits, atol=1e-6, rtol=1e-5)
    with last_token_projection(model) as rows:
        actual = model.generate(ids, **kwargs)
    assert torch.equal(expected, actual)
    assert rows == {"calls": 8, "input_rows": batch * (length + 7), "projected_rows": batch * 8}
    assert model.lm_head.forward == original and "forward" not in model.lm_head.__dict__
    assert list(model.state_dict()) == keys and torch.equal(model.lm_head.weight, weights)
    assert not model._forward_pre_hooks


@torch.inference_mode()
def test_pulse_model_type_is_supported():
    model = tiny_model()
    model.config.model_type = "llava_llama"
    ids = torch.ones((2, 7), dtype=torch.long)
    expected = model(ids).logits[:, -1:]
    with last_token_projection(model):
        torch.testing.assert_close(model(ids).logits, expected, atol=1e-6, rtol=1e-5)


def test_training_or_grad_enabled_usage_is_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="eval/no-grad"):
        with last_token_projection(model):
            pass
    with torch.inference_mode(), pytest.raises(ValueError, match="eval/no-grad"):
        with last_token_projection(model.train()):
            pass


@torch.inference_mode()
@pytest.mark.parametrize("positional", [False, True])
def test_labels_are_rejected_and_exception_restores_original_head(positional):
    model = tiny_model()
    ids = torch.ones((1, 7), dtype=torch.long)
    original = model.lm_head.forward
    with pytest.raises(ValueError, match="supervised labels"):
        with last_token_projection(model):
            if positional:
                model(ids, None, None, None, None, ids)
            else:
                model(ids, labels=ids)
    assert model.lm_head.forward == original and "forward" not in model.lm_head.__dict__
    assert not model._forward_pre_hooks
    assert model(ids, labels=ids).loss.isfinite()


@torch.inference_mode()
def test_nested_context_and_tensor_parallel_projection_are_rejected():
    model = tiny_model()
    with last_token_projection(model):
        with pytest.raises(ValueError, match="unmodified"):
            with last_token_projection(model):
                pass
    model.config.pretraining_tp = 2
    with pytest.raises(ValueError, match="unsupported"):
        with last_token_projection(model):
            pass


@torch.inference_mode()
def test_merge_probe_restores_exact_original_weights_on_exception():
    peft = pytest.importorskip("peft")
    from util.evaluation.pulse_profile import merged_lora_probe
    model = peft.get_peft_model(tiny_model(), peft.LoraConfig(r=4,
        target_modules=["q_proj", "v_proj"], lora_alpha=8)).eval()
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.normal_()
    original = {n: p.clone() for n, p in model.named_parameters()}
    ids = torch.ones((2, 7), dtype=torch.long)
    expected = model(ids).logits
    with pytest.raises(RuntimeError, match="probe failure"):
        with merged_lora_probe(model) as count:
            assert count == 4
            torch.testing.assert_close(model(ids).logits, expected, atol=2e-5, rtol=2e-5)
            raise RuntimeError("probe failure")
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])
    assert all(not m.merged for m in model.modules() if hasattr(m, "merged"))


@torch.inference_mode()
def test_dtype_probe_restores_exact_originals_on_exception():
    from util.evaluation.pulse_profile import half_trainables_probe
    parameter = torch.nn.Parameter(torch.randn(3, 4), requires_grad=False)
    expected = parameter.clone()
    with pytest.raises(RuntimeError, match="probe failure"):
        with half_trainables_probe({"probe": parameter}):
            assert parameter.dtype == torch.float16
            raise RuntimeError("probe failure")
    assert parameter.dtype == torch.float32 and torch.equal(parameter, expected)


@torch.inference_mode()
def test_raw_logit_capture_matches_greedy_tokens_and_restores_hook():
    from util.evaluation.pulse_profile import capture_generation_logits
    model = tiny_model()
    packed = model.get_input_embeddings()(torch.ones((2, 7), dtype=torch.long))
    ids, scores = capture_generation_logits(model, lambda: model.generate(inputs_embeds=packed,
        max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=0))
    assert len(scores) == 4
    assert all(score.shape == (2, 97) for score in scores)
    assert torch.equal(torch.stack(scores, dim=1).argmax(-1), ids[:, 1:])
    assert not model._forward_hooks
    with pytest.raises(RuntimeError, match="capture failure"):
        capture_generation_logits(model, lambda: (_ for _ in ()).throw(RuntimeError("capture failure")))
    assert not model._forward_hooks


@torch.inference_mode()
def test_single_deployment_merge_removes_branches_and_adapter_references():
    from types import SimpleNamespace
    peft = pytest.importorskip("peft")
    from util.evaluation.pulse_deployment import merge_single_backend
    model = peft.get_peft_model(tiny_model(), peft.LoraConfig(r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"])).eval()
    model.requires_grad_(False)
    for name, value in model.named_parameters():
        if "lora_B" in name:
            value.normal_(std=.01)
    ids = torch.ones((2, 7), dtype=torch.long)
    expected = model(ids).logits.clone()
    backend = SimpleNamespace(model=model, base=model.get_base_model(),
        parameters=dict(model.named_parameters()), states={"w1": {"dummy": torch.ones(2)}})
    assert merge_single_backend(backend) == 4
    assert backend.model is backend.base
    assert not backend.parameters and not backend.states
    assert not any("lora_" in n for n, _ in backend.model.named_parameters())
    torch.testing.assert_close(backend.model(ids).logits, expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="unmerged"):
        merge_single_backend(backend)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("chunk_size", [0, 7, 64])
@torch.inference_mode()
def test_mutable_cache_embedded_prompt_tokens_logits_and_full_cache(batch, chunk_size):
    from util.evaluation.pulse_profile import capture_generation_logits
    model = tiny_model()
    packed = model.get_input_embeddings()(torch.randint(3, 97, (batch, 31)))
    mask = torch.ones((batch, 31), dtype=torch.long)
    def generate():
        return model.generate(inputs_embeds=packed, attention_mask=mask,
            max_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=0)
    with last_token_projection(model):
        expected, reference = capture_generation_logits(model, generate)
        with generation_cache(model, chunk_size=chunk_size) as counts:
            actual, scores = capture_generation_logits(model, generate)
    assert torch.equal(expected, actual)
    torch.testing.assert_close(torch.stack(scores), torch.stack(reference), atol=1e-6, rtol=1e-5)
    if not chunk_size:
        assert torch.equal(torch.stack(scores), torch.stack(reference))
    assert counts == {"prefills": 1, "prefill_chunks": (31 + (chunk_size or 31) - 1) // (chunk_size or 31),
                      "prompt_tokens": 31}
    assert "forward" not in model.model.__dict__
    assert not model._forward_pre_hooks and not model._forward_hooks


@torch.inference_mode()
def test_mutable_cache_rejections_restore_decoder():
    model = tiny_model()
    original = model.model.forward
    ids = torch.ones((1, 7), dtype=torch.long)
    for inputs, message in [({"input_ids": ids}, "embedded prompt"),
                            ({"input_ids": ids, "labels": ids}, "supervised labels"),
                            ({"inputs_embeds": model.get_input_embeddings()(ids), "use_cache": False}, "use_cache")]:
        with pytest.raises(ValueError, match=message):
            with generation_cache(model):
                model(**inputs)
        assert model.model.forward == original and not model._forward_pre_hooks
    with generation_cache(model):
        with pytest.raises(ValueError, match="unmodified"):
            with generation_cache(model):
                pass
    with pytest.raises(RuntimeError, match="probe failure"):
        with generation_cache(model):
            raise RuntimeError("probe failure")
    assert model.model.forward == original
