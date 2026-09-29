"""Mild, geometry-preserving paper/scan effects on float RGB ECG tensors.

No rotations, lead rearrangements, text insertion or cross-record mixing.
All randomness is supplied by the caller; input pixels remain unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGE_OPERATORS = ("paper_texture", "grid_fade", "tone", "shadow")


def validate_image_config(config, *, for_execution=False):
    operators = tuple(config.get("operators", ()))
    implementation = config.get("implementation")
    reference = False
    if "implementation" in config or operators != IMAGE_OPERATORS:
        from core.image_augmix_c import AUGMIX_IMPLEMENTATION, AUGMIX_TRAIN_OPERATORS
        from core.image_augmix_gpu import GPU_AUGMIX_IMPLEMENTATION
        reference = implementation in {AUGMIX_IMPLEMENTATION, GPU_AUGMIX_IMPLEMENTATION}
    expected = {"operators", "waveform_strength", "jsd_weight"}
    expected |= {"implementation", "severity"} if reference else {"strength"}
    if set(config) != expected:
        raise ValueError("invalid hybrid image configuration")
    if reference:
        if operators != AUGMIX_TRAIN_OPERATORS:
            raise ValueError("reference AugMix requires its nine training operators")
        if implementation == "augmix_torch_gpu_v2" and config.get("waveform_strength") != 0:
            raise ValueError("GPU v2 AugMix is image-only and requires waveform_strength=0")
        level = config["severity"]
        if type(level) not in (int, float) or not math.isfinite(level) or not 0.1 <= level <= 10:
            raise ValueError("invalid reference AugMix severity")
    elif operators != IMAGE_OPERATORS:
        if operators != AUGMIX_TRAIN_OPERATORS:
            raise ValueError("hybrid image operator pool changed")
        if for_execution:
            raise ValueError("2026-09-18 approximate AugMix is historical only; use the reference config")
    bounds = [("waveform_strength", 1), ("jsd_weight", 24)]
    if not reference:
        bounds.append(("strength", 1))
    for key, maximum in bounds:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"invalid image augmentation {key}")


def apply_image_operator(image, operator, *, strength, rng):
    """BCHW RGB in [0,1]; strength 0 is bitwise identity, 1 is the mild cap."""
    if operator not in IMAGE_OPERATORS:
        raise ValueError("invalid image operator or strength")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("invalid image operator or strength")
    if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
        raise ValueError("expected floating BCHW RGB")
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any() | (image > 1).any()):
        raise ValueError("expected finite RGB in [0,1]")
    if strength == 0:
        return image.clone()
    def rand(*shape):
        return torch.rand(shape, device=image.device, dtype=image.dtype, generator=rng)
    batch, _, height, width = image.shape
    amount = strength * rand(batch, 1, 1, 1)
    if operator == "paper_texture":
        texture = F.interpolate(rand(batch, 1, 24, 24), size=(height, width), mode="bilinear", align_corners=False)
        tint = image.new_tensor([0.015, 0.04, 0.10]).view(1, 3, 1, 1)
        # Multiplication preserves dark traces; no high-frequency fake waveforms.
        result = image * (1 - amount * (tint + 0.04 * texture))
    elif operator == "grid_fade":
        # Only fade red/pink grid pixels. Achromatic ECG traces/text are untouched.
        red = image[:, :1]
        mask = ((red - image[:, 1:2] > 0.015) & (red - image[:, 2:3] > 0.015)).to(image.dtype)
        result = image + 0.8 * amount * mask * (1 - image)
    elif operator == "tone":
        gamma = 1 + (2 * rand(batch, 1, 1, 1) - 1) * 0.15 * strength
        result = image.pow(gamma) * (1 - 0.10 * amount)
    else:
        x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)[None, None, None, :]
        y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)[None, None, :, None]
        cx, cy = 2 * rand(batch, 1, 1, 1) - 1, 2 * rand(batch, 1, 1, 1) - 1
        shade = torch.exp(-((x - cx).square() + (y - cy).square()) / 0.8)
        result = image * (1 - 0.22 * amount * shade)
    return result.clamp(0, 1)
