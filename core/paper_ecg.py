"""Versioned paper-ECG appearance operators, implemented entirely in PyTorch.

BCHW RGB floats in [0, 1], before processor normalization. Parameters and
pixels stay on the input device; CPU execution exists for small contract tests.
These are procedural approximations, not ECG-Image-Kit pixel reproductions.
See docs/pulse_paper_operators.md for the research and severity contract.
"""
from functools import lru_cache
import math

import torch
import torch.nn.functional as F

PAPER_IMPLEMENTATION = "paper_ecg_torch_v1"
PAPER_TRAIN_OPERATORS = ("yellowing", "exposure", "shadow", "crease", "wrinkle",
                         "ink_fade", "defocus", "sensor_noise", "low_resolution", "grid_fade")
PAPER_HELDOUT_OPERATORS = ("rotate", "perspective", "stain", "glare", "occlusion", "jpeg_compression")
PAPER_EVAL_OPERATORS = PAPER_TRAIN_OPERATORS + PAPER_HELDOUT_OPERATORS


def validate_paper_image(image):
    if (image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3
            or min(image.shape[-2:]) < 2 or not image.is_floating_point()):
        raise ValueError("expected nonempty floating BCHW RGB")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("expected finite RGB in [0, 1]")


@lru_cache(maxsize=8)
def _axes(device, height, width):
    # O(H+W) cached coordinates, not a cache of full images or random outcomes.
    x = ((torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2 / width) - 1)[None, None, None, :]
    y = ((torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2 / height) - 1)[None, None, :, None]
    return x, y


def _blur(image, sigma):
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    kernel = (-0.5 * (x / sigma).square()).exp()
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    value = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=3)
    return F.conv2d(F.pad(value, (0, 0, radius, radius), mode="replicate"), vertical, groups=3)


def _warp(image, gx, gy):
    batch, _, height, width = image.shape
    grid = torch.stack((gx.expand(batch, 1, height, width),
                        gy.expand(batch, 1, height, width)), -1).squeeze(1)
    # Sampling image - white makes out-of-frame pixels paper colored.
    return F.grid_sample(image - 1, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False) + 1


def apply_paper_operator(image, operator, *, severity, rng, validate=True):
    """Apply one seeded operator; severity 0 is exact identity, 1..5 are stress levels.

    A trusted chain can validate its RGB anchor once and pass validate=False.
    There is no per-image Python loop, image transfer, or RNG scalar readback.
    Same-device replay is exact; cross-device bitwise identity is not promised.
    """
    if (operator not in PAPER_EVAL_OPERATORS or type(severity) not in (int, float)
            or not math.isfinite(severity) or not 0 <= severity <= 5):
        raise ValueError("invalid paper operator or severity")
    if rng is None or torch.device(rng.device) != image.device:
        raise ValueError("paper operators require an explicit generator on the image device")
    if validate:
        validate_paper_image(image)
    if severity == 0:
        return image.clone()
    value = image.float()
    batch, _, height, width = value.shape
    strength = float(severity) / 5

    def rand(*shape):
        return torch.rand(shape or (batch, 1, 1, 1), device=image.device, dtype=torch.float32, generator=rng)

    amount = strength * (0.75 + 0.25 * rand())
    x, y = _axes(str(image.device), height, width)
    if operator == "yellowing":
        gain = torch.cat((1 - 0.04 * amount, 1 - 0.18 * amount, 1 - 0.48 * amount), 1)
        result = value * gain
    elif operator == "exposure":
        gamma = 1 + (2 * rand() - 1) * 0.65 * amount
        result = value.clamp_min(1e-6).pow(gamma) * (1 - 0.3 * amount)
    elif operator in ("shadow", "stain", "glare"):
        cx, cy = 1.6 * rand() - 0.8, 1.6 * rand() - 0.8
        sx, sy = 0.12 + 0.38 * rand(), 0.12 + 0.38 * rand()
        field = (-0.5 * (((x - cx) / sx).square() + ((y - cy) / sy).square())).exp()
        if operator == "shadow":
            result = value * (1 - 0.7 * amount * field)
        elif operator == "glare":
            result = value + 0.95 * amount * field * (1 - value)
        else:
            # Translucent brown blotch: an appearance proxy for a paper stain.
            opacity = 0.85 * amount * field
            tint = torch.cat((0.62 + 0.12 * rand(), 0.33 + 0.10 * rand(), 0.12 + 0.08 * rand()), 1)
            result = value * (1 - opacity) + tint * opacity
    elif operator == "crease":
        angle, offset = math.pi * rand(), rand() - 0.5
        distance = x * angle.cos() + y * angle.sin() - offset
        band = 0.003 + 0.007 * rand()
        shade = (-0.5 * (distance / band).square()).exp()
        highlight = (-0.5 * ((distance - 1.8 * band) / band).square()).exp()
        result = value * (1 - 0.55 * amount * shade) + 0.25 * amount * highlight * (1 - value)
    elif operator == "wrinkle":
        # Smooth random shading plus directional ridges, without bending traces.
        texture = F.interpolate(rand(batch, 1, 24, 24), (height, width), mode="bilinear", align_corners=False)
        angle, frequency, phase = math.pi * rand(), 12 + 20 * rand(), math.tau * rand()
        ridges = (frequency * (x * angle.cos() + y * angle.sin()) + phase).sin().square()
        field = 0.5 * texture + 0.5 * ridges
        result = value * (1 - 0.35 * amount * field)
    elif operator in ("ink_fade", "grid_fade"):
        if operator == "ink_fade":
            high, low = value.amax(1, keepdim=True), value.amin(1, keepdim=True)
            mask = ((high - low < 0.08) & (high < 0.65)).to(value.dtype)
        else:
            mask = ((value[:, :1] - value[:, 1:2] > 0.015)
                    & (value[:, :1] - value[:, 2:3] > 0.015)).to(value.dtype)
        result = value + 0.85 * amount * mask * (1 - value)
    elif operator == "defocus":
        result = _blur(value, max(0.3, (0.4 + 0.8 * severity) * min(height, width) / 1700))
    elif operator == "sensor_noise":
        noise = torch.randn(value.shape, device=image.device, dtype=torch.float32, generator=rng)
        result = value + 0.10 * amount * noise
    elif operator == "low_resolution":
        size = (max(2, round(height * (1 - 0.75 * strength))),
                max(2, round(width * (1 - 0.75 * strength))))
        reduced = F.interpolate(value, size, mode="bilinear", align_corners=False, antialias=True)
        result = F.interpolate(reduced, (height, width), mode="bilinear", align_corners=False)
    elif operator == "rotate":
        angle = (2 * rand() - 1) * amount * math.pi / 18
        cosine, sine = angle.cos(), angle.sin()
        fit = torch.maximum(cosine + sine.abs() * height / width,
                            cosine + sine.abs() * width / height)
        result = _warp(value, fit * (x * cosine + y * sine * height / width),
                        fit * (y * cosine - x * sine * width / height))
    elif operator == "perspective":
        px, py = (2 * rand() - 1) * 0.15 * amount, (2 * rand() - 1) * 0.15 * amount
        denominator = 1 + px * x + py * y
        result = _warp(value, x / denominator, y / denominator)
    elif operator == "occlusion":
        cx, cy = 1.6 * rand() - 0.8, 1.6 * rand() - 0.8
        mask = ((x - cx).abs() < 0.25 * amount) & ((y - cy).abs() < 0.16 * amount)
        result = torch.where(mask, torch.ones_like(value), value)
    else:
        from core.image_augmix_gpu import _jpeg_compression
        result = _jpeg_compression(value, torch.full((batch,), 9 * strength, device=image.device))
    return result.clamp(0, 1).to(image.dtype)


def paper_conditions(severities=(1, 2, 3, 4, 5)):
    """Fixed per-operator tests; never mix seen and held-out family averages."""
    if not severities or any(type(s) is not int or not 1 <= s <= 5 for s in severities) or len(set(severities)) != len(severities):
        raise ValueError("paper stress severities must be unique integers from one to five")
    return [{"condition_id": f"paper/{op}/s{severity}", "family": "image",
             "image_operator": op, "image_severity": severity,
             "stress_group": "seen_family" if op in PAPER_TRAIN_OPERATORS else "held_out_family",
             "implementation": PAPER_IMPLEMENTATION}
            for op in PAPER_EVAL_OPERATORS for severity in severities]
