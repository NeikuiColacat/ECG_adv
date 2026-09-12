"""Opt-in generation-only projection prototype, not enabled by the benchmark.

Greedy decoding consumes only the last position's vocabulary logits. Preserve
the original trained linear layer and weights while projecting that position.
Real PULSE token equivalence and throughput require a separate GPU admission.
"""
from contextlib import contextmanager
from functools import wraps
import inspect

import torch


@contextmanager
def last_token_projection(model):
    if model.training or torch.is_grad_enabled():
        raise ValueError("last-token projection is restricted to eval/no-grad inference")
    if model.config.model_type not in {"llama", "llava", "llava_llama"} or model.config.pretraining_tp != 1:
        raise ValueError("unsupported causal-Llama projection contract")
    head = model.lm_head
    if type(head) is not torch.nn.Linear or "forward" in head.__dict__:
        raise ValueError("expected an unmodified original linear vocabulary layer")
    original = head.forward
    counter = {"calls": 0, "input_rows": 0, "projected_rows": 0}

    def project(hidden):
        if hidden.ndim != 3 or not hidden.shape[1]:
            raise ValueError("expected nonempty batch/sequence/hidden input")
        counter["calls"] += 1
        counter["input_rows"] += hidden.shape[0] * hidden.shape[1]
        counter["projected_rows"] += hidden.shape[0]
        return original(hidden[:, -1:, :])

    def reject_supervision(module, args, kwargs):
        if kwargs.get("labels") is not None or (len(args) > 5 and args[5] is not None):
            raise ValueError("last-token projection must not be used with supervised labels")

    guard = model.register_forward_pre_hook(reject_supervision, with_kwargs=True)
    head.forward = project
    try:
        yield counter
    finally:
        del head.forward
        guard.remove()


@contextmanager
def generation_cache(model, *, chunk_size=0):
    """Opt-in mutable HF 4.37 cache, optionally chunking embedded-prompt prefill.

    Inject the empty cache INSIDE the decoder, after generation has selected
    inputs_embeds. Passing it to generate itself would discard the image prompt.
    Only the bounded deployment probe uses this; no global monkey patch.
    """
    from transformers.cache_utils import DynamicCache
    if model.training or torch.is_grad_enabled():
        raise ValueError("generation cache requires eval/no-grad")
    if model.config.model_type not in {"llama", "llava", "llava_llama"}:
        raise ValueError("unsupported generation cache model")
    if not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("chunk size must be a nonnegative integer")
    decoder = model.model
    if "forward" in decoder.__dict__:
        raise ValueError("expected an unmodified decoder")
    original = decoder.forward
    signature = inspect.signature(original)
    counter = {"prefills": 0, "prefill_chunks": 0, "prompt_tokens": 0}

    @wraps(original)
    def forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = dict(bound.arguments)
        use_cache = values.get("use_cache")
        if not (model.config.use_cache if use_cache is None else use_cache):
            raise ValueError("generation cache requires use_cache")
        if values.get("output_attentions") or values.get("output_hidden_states") or values.get("return_dict") is False:
            raise ValueError("unsupported diagnostic output contract")
        past = values.get("past_key_values")
        if past is not None:
            if not isinstance(past, DynamicCache):
                raise ValueError("legacy cache supplied inside mutable-cache context")
            return original(**values)
        embeds = values.get("inputs_embeds")
        if embeds is None or values.get("input_ids") is not None or embeds.ndim != 3:
            raise ValueError("initial generation cache call requires embedded prompt")
        length = embeds.shape[1]
        if not length:
            raise ValueError("empty embedded prompt")
        mask, positions = values.get("attention_mask"), values.get("position_ids")
        if mask is not None and (mask.ndim != 2 or mask.shape[-1] != length):
            raise ValueError("expected a full two-dimensional prompt mask")
        values["past_key_values"] = DynamicCache()
        counter["prefills"] += 1
        counter["prompt_tokens"] += length
        stride = chunk_size or length
        for start in range(0, length, stride):
            end = min(start + stride, length)
            piece = dict(values, inputs_embeds=embeds[:, start:end])
            if mask is not None:
                piece["attention_mask"] = mask[:, :end]
            if positions is not None:
                piece["position_ids"] = positions[:, start:end]
            output = original(**piece)
            if output.past_key_values is not values["past_key_values"]:
                raise ValueError("decoder did not preserve the mutable cache")
            counter["prefill_chunks"] += 1
        if values["past_key_values"].get_seq_length() != length:
            raise ValueError("prefill lost prompt tokens")
        return output

    def reject_supervision(module, args, kwargs):
        if kwargs.get("labels") is not None or (len(args) > 5 and args[5] is not None):
            raise ValueError("generation cache must not be used with supervised labels")

    guard = model.register_forward_pre_hook(reject_supervision, with_kwargs=True)
    decoder.forward = forward
    try:
        yield counter
    finally:
        del decoder.forward
        guard.remove()
