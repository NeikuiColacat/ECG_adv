"""Waveform corruption followed by image-branch AugMix; two views for JSD."""
from __future__ import annotations

import hashlib

import torch

from core.augmix import _dirichlet
from core.corruption import CANONICAL_OPERATORS
from core.image_corruption import GPU_IMAGE_IMPLEMENTATIONS, image_operator, validate_image_config
from util.augmentations.torch_operators import apply_operator_batch_prevalidated


def hybrid_jsd_views(clean, *, width, profile, seed, identity, renderer, image_config):
    validate_image_config(image_config, for_execution=True)
    if tuple(clean.shape) != (1, 5000, 12) or width not in (0, 1, 3):
        raise ValueError("hybrid AugMix requires one native500 ECG and width 0/1/3")
    if not clean.is_floating_point() or not bool(torch.isfinite(clean).all()):
        raise ValueError("expected finite physical mV")
    def _keyed_seed(tag):
        raw = f"pulse-hybrid-v1|{seed}|{identity}|{tag}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
    def keyed(tag):
        value = _keyed_seed(tag)
        return torch.Generator(device=clean.device).manual_seed(value)
    def keyed_cpu(tag):
        return torch.Generator(device="cpu").manual_seed(_keyed_seed(tag))
    def choice(count, rng):
        return int(torch.randint(count, (), device="cpu", generator=rng))

    image_implementation = image_config.get("implementation")
    apply_image = image_operator(image_config)
    gpu_image_only = image_implementation in GPU_IMAGE_IMPLEMENTATIONS
    original = renderer.render(clean)
    if gpu_image_only and width:
        from core.image_augmix_gpu import validate_gpu_image
        validate_gpu_image(original)
    views, traces, mix_parameters = [original], [], []
    if width:
        for view in range(2):
            if gpu_image_only:
                # Waveform corruption is disabled for this throughput protocol.
                anchor = original
                wave_ops = []
            else:
                wave = clean.clone()
                rng = keyed(f"view{view}/wave")
                choice_rng = keyed_cpu(f"view{view}/wave_choices")
                wave_ops = []
                for _ in range(1 + choice(3, choice_rng)):
                    op = CANONICAL_OPERATORS[choice(len(CANONICAL_OPERATORS), choice_rng)]
                    params = dict(profile.parameters_for(op))
                    for key in ("min_amplitude", "max_amplitude", "mask_leads_prob"):
                        if key in params:
                            params[key] *= image_config["waveform_strength"]
                    if image_config["waveform_strength"] > 0:
                        wave = apply_operator_batch_prevalidated(op, wave, params=params, sampling_rate_hz=500, rng=rng)
                    wave_ops.append(op)
                # All image branches see exactly the same wave, avoiding ghost traces.
                anchor = renderer.render(wave)
            weights = _dirichlet(1, width, alpha=1, device=clean.device, generator=keyed(f"view{view}/weights"))[0]
            m = torch.rand((), device=clean.device, generator=keyed(f"view{view}/beta"))
            mixed = torch.zeros_like(anchor)
            chains = []
            for chain in range(width):
                rng = keyed(f"view{view}/image{chain}")
                choice_rng = keyed_cpu(f"view{view}/image{chain}_choices")
                value, ops = anchor, []
                for _ in range(1 + choice(3, choice_rng)):
                    op = image_config["operators"][choice(len(image_config["operators"]), choice_rng)]
                    value = apply_image(value, op, rng=rng)
                    ops.append(op)
                # Keep the mix coefficient on-device; converting a CUDA scalar
                # to Python here inserts a host synchronization per chain.
                mixed.add_(value * weights[chain].to(dtype=value.dtype))
                chains.append(ops)
            views.append(((1 - m) * anchor + m * mixed).clamp(0, 1))
            traces.append({"waveform_operators": wave_ops, "image_chains": chains})
            mix_parameters.append(torch.cat((weights, m.reshape(1))))
        # One small metadata transfer after both views; never copy image tensors.
        for trace, values in zip(traces, torch.stack(mix_parameters).tolist()):
            trace.update(weights=values[:-1], m=values[-1])
    return views, {"width": width, "views": traces, "mixing_domain": "rendered_rgb",
                   "topology": "image_only_gpu_branches_v1" if gpu_image_only else "shared_waveform_image_branches_v1",
                   "residual": "clean_render" if gpu_image_only else "corrupted_waveform_render"}
