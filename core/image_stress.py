"""Frozen image-v2 acquisition probes, separate from the training operators.

These are synthetic development stresses, not validated hospital simulators.
No lead reorder, amplitude scaling, text insertion, or random waveform cutout.
"""
import math

import torch
import torch.nn.functional as F

IMAGE_PROFILES = ("perspective", "low_resolution", "illumination")


def apply_image_stress(image, profile, *, rng):
    """BCHW float RGB [0,1] -> same canvas; explicit local RNG only."""
    if profile not in IMAGE_PROFILES or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("invalid image-v2 profile or RGB shape")
    batch, _, height, width = image.shape
    def rand(*shape):
        return torch.rand(shape, device=image.device, dtype=image.dtype, generator=rng)
    if profile == "perspective":
        # Inverse planar homography, with enough margin to retain all corners.
        angle = (2 * rand(batch, 1, 1) - 1) * math.radians(4)
        px, py = [(2 * rand(batch, 1, 1) - 1) * 0.035 for _ in range(2)]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
                               torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype), indexing="ij")
        denominator = 1 + px * xx + py * yy
        gx = (angle.cos() * xx + angle.sin() * yy) / (0.84 * denominator)
        gy = (-angle.sin() * xx + angle.cos() * yy) / (0.84 * denominator)
        grid = torch.stack((gx, gy), -1)
        # Sample darkness on a white canvas, not black padding around the paper.
        result = 1 - F.grid_sample(1 - image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    elif profile == "low_resolution":
        # Fixed 2.5x down/up acquisition loss plus a small box-blur. No fake JPEG.
        result = F.interpolate(image, size=(max(1, round(height * 0.4)), max(1, round(width * 0.4))), mode="area")
        result = F.interpolate(result, size=(height, width), mode="bilinear", align_corners=False)
        result = F.avg_pool2d(F.pad(result, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    else:
        x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)[None, None, None, :]
        y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)[None, None, :, None]
        cx, cy = [2 * rand(batch, 1, 1, 1) - 1 for _ in range(2)]
        shade = torch.exp(-((x - cx).square() + (y - cy).square()) / 0.65)
        depth = 0.25 + 0.15 * rand(batch, 1, 1, 1)
        gamma = 0.8 + 0.4 * rand(batch, 1, 1, 1)
        result = image.pow(gamma) * (1 - depth * shade)
    return result.clamp(0, 1)
