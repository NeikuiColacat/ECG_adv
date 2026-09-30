"""Torch adaptations of selected Augraphy ink/paper algorithms.

Derived algorithm references: sparkfish/augraphy, MIT, commit below.
See docs/licenses/augraphy-MIT.txt and docs/paper_ecg_open_source.md.
RGB BCHW floats; quantized kernels follow the upstream uint8 image domain.
RNG, severity scaling and low-ink sampling are versioned project choices.
"""
from functools import lru_cache
import math

import torch
import torch.nn.functional as F

from core.paper_ecg import (PAPER_HELDOUT_OPERATORS, apply_paper_operator as apply_v1,
                            validate_paper_image)

AUGRAPHY_COMMIT = 'ed4dcbdaf7b1da6ef59ac60816f96ea253da4c9b'
PAPER_IMPLEMENTATION = 'paper_ecg_torch_v2'
PAPER_TRAIN_OPERATORS = ('color_paper', 'ink_bleed', 'low_ink_lines', 'exposure',
    'shadow', 'crease', 'wrinkle', 'defocus', 'sensor_noise', 'low_resolution')
PAPER_EVAL_OPERATORS = PAPER_TRAIN_OPERATORS + PAPER_HELDOUT_OPERATORS
PORTED_OPERATORS = ('color_paper', 'ink_bleed', 'low_ink_lines')


@lru_cache(maxsize=8)
def _constants(device):
    dx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], device=device, dtype=torch.float32)
    gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=device, dtype=torch.float32) / 16
    luma = torch.tensor([.299, .587, .114], device=device, dtype=torch.float32).reshape(1, 3, 1, 1)
    offsets = torch.tensor([5, 3, 1], device=device, dtype=torch.float32).reshape(1, 3, 1, 1)
    return dx.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1), gaussian.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1), luma, offsets


def _byte_rgb(image):
    return image.float().mul(255).round().clamp(0, 255)


def _color_paper_from_hs(image, hue, saturation):
    """Fixed-parameter ColorPaper; H/S use OpenCV's byte-domain conventions."""
    value = _byte_rgb(image).amax(1, keepdim=True)
    offsets = _constants(str(image.device))[3]
    k = (hue / 30 + offsets).remainder(6)
    chroma = torch.minimum(k, 4-k).clamp(0, 1)
    result = value * (1 - saturation / 255 * chroma)
    return result.round().clamp(0, 255) / 255


def _ink_bleed_from_mask(image, selection, intensity):
    """Fixed-mask InkBleed: Scharr difference, dilation, erosion, blur, blend."""
    value = _byte_rgb(image)
    dx, gaussian, luma, _ = _constants(str(image.device))
    padded = F.pad(value, (1, 1, 1, 1), mode='reflect')
    # Upstream calls this Sobel, but ksize=-1 selects Scharr, and uses dx-dy.
    edge = (F.conv2d(padded, dx, groups=3) - F.conv2d(padded, dx.transpose(-1, -2), groups=3)).abs().round().clamp(0, 255)
    edge = (F.max_pool2d(edge, 5, 1, 2) * luma).sum(1, keepdim=True).round() > 0
    eroded = -F.max_pool2d(-value, 5, 1, 2)
    updated = torch.where(edge & selection, eroded, value)
    blurred = F.conv2d(F.pad(updated, (1, 1, 1, 1), mode='reflect'), gaussian, groups=3).round()
    return (intensity * blurred + (1-intensity) * value).round().clamp(0, 255) / 255


def _low_ink_from_rows(image, period, offset, alpha):
    """A documented variant: only lighten selected rows; no neighbor darkening."""
    y = torch.arange(image.shape[-2], device=image.device).reshape(1, 1, -1, 1)
    rows = (y-offset).remainder(period) == 0
    return torch.where(rows, image.float().clamp_min(alpha), image.float())


def apply_paper_operator(image, operator, *, severity, rng, validate=True):
    """Device-local, seeded version two; zero strength preserves pixels and RNG.

    This supports the project's RGB subset, not Augraphy's grayscale/alpha API.
    Fixed-parameter rounding tolerances and sampling changes are documented.
    """
    if (operator not in PAPER_EVAL_OPERATORS or type(severity) not in (int, float)
            or not math.isfinite(severity) or not 0 <= severity <= 5):
        raise ValueError('invalid upstream paper operator or severity')
    if rng is None or torch.device(rng.device) != image.device:
        raise ValueError('paper operators require an explicit generator on the image device')
    if validate:
        validate_paper_image(image)
    if severity == 0:
        return image.clone()
    with torch.autocast(image.device.type, enabled=False):
        if operator not in PORTED_OPERATORS:
            return apply_v1(image, operator, severity=severity, rng=rng, validate=False)
        batch, _, height, width = image.shape
        shape = (batch, 1, 1, 1)
        s = float(severity) / 5
        def rand(size=shape):
            return torch.rand(size, device=image.device, dtype=torch.float32, generator=rng)
        if operator == 'color_paper':
            hue = (28 + 18*rand()).floor() + (10*rand((batch, 1, height, width))).floor() - 5
            saturation = ((10 + 31*rand()).floor()*s + (10*rand((batch, 1, height, width))-5)*s).floor().clamp(0, 255)
            result = _color_paper_from_hs(image, hue, saturation)
        elif operator == 'ink_bleed':
            probability = (.3 + .1*rand())*s
            selection = rand((batch, 1, height, width)) < probability
            result = _ink_bleed_from_mask(image, selection, (.4 + .3*rand())*s)
        else:
            count = (2 + max(1, round(18*s))*rand()).floor().clamp_min(1)
            period = (height/count).floor().clamp_min(1)
            result = _low_ink_from_rows(image, period, (rand()*period).floor(), (.3+.6*s)*(.75+.25*rand()))
        return result.clamp(0, 1).to(image.dtype)


def paper_conditions(severities=(1, 2, 3, 4, 5)):
    if not severities or any(type(s) is not int or not 1 <= s <= 5 for s in severities) or len(set(severities)) != len(severities):
        raise ValueError('paper stress severities must be unique integers from one to five')
    return [{'condition_id': f'paper_v2/{op}/s{severity}', 'family': 'image',
        'image_operator': op, 'image_severity': severity,
        'stress_group': 'seen_family' if op in PAPER_TRAIN_OPERATORS else 'held_out_family',
        'implementation': PAPER_IMPLEMENTATION} for op in PAPER_EVAL_OPERATORS for severity in severities]
