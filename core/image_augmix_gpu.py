"""CUDA resident image operators for the PULSE visual approximation protocol.

All public functions consume floating BCHW RGB tensors in [0, 1].  Image
tensors and random tensors stay on the input device.  The nine AugMix
operators and the five C5 evaluation operators are deliberately visual
approximations; they are not pixel-equivalent to Pillow or imagecorruptions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


GPU_AUGMIX_IMPLEMENTATION = "augmix_torch_gpu_v2"
GPU_C5_IMPLEMENTATION = "image_c5_torch_gpu_v1"
AUGMIX_GPU_OPERATORS = (
    "autocontrast", "equalize", "posterize", "rotate", "solarize",
    "shear_x", "shear_y", "translate_x", "translate_y",
)
C5_GPU_OPERATORS = (
    "gaussian_noise", "motion_blur", "brightness", "elastic_transform",
    "jpeg_compression",
)


def _check(image: torch.Tensor) -> None:
    if (image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3
            or min(image.shape[-2:]) < 2 or not image.is_floating_point()):
        raise ValueError("expected nonempty floating BCHW RGB")
    if image.device.type != "cuda":
        raise ValueError("GPU image operators require a CUDA image tensor")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("expected finite RGB")
    if bool(((image < 0) | (image > 1)).any()):
        raise ValueError("expected RGB values in [0, 1]")


def validate_gpu_image(image: torch.Tensor) -> None:
    """Validate one RGB anchor before a trusted multi-op chain."""
    _check(image)


def _rng(image: torch.Tensor, generator: torch.Generator | None) -> torch.Generator:
    if generator is None:
        raise ValueError("GPU image operators require an explicit device-local generator")
    gdev, idev = torch.device(generator.device), image.device
    if gdev.type != idev.type or (gdev.type == "cuda" and gdev.index not in (None, idev.index)):
        raise ValueError(f"generator device {gdev} does not match image device {idev}")
    return generator


def _rand(image: torch.Tensor, generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.rand(shape, device=image.device, dtype=torch.float32, generator=generator)


def _sample_level(image: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    return _rand(image, generator, int(image.shape[0])) * (severity - 0.1) + 0.1


def _quantize(image: torch.Tensor) -> torch.Tensor:
    return image.float().mul(255).round().clamp(0, 255).div(255).to(image.dtype)


def _autocontrast(image: torch.Tensor) -> torch.Tensor:
    value = image.float()
    low, high = value.amin((-2, -1), True), value.amax((-2, -1), True)
    scaled = (value - low) / (high - low).clamp_min(1e-6)
    # PIL preserves channels with no dynamic range.
    return torch.where(high > low, scaled, value).clamp(0, 1).to(image.dtype)


def _equalize(image: torch.Tensor) -> torch.Tensor:
    """Per-image/per-channel 256-bin histogram equalization on the device."""
    value = image.float()
    b, c, h, w = value.shape
    bins = value.mul(255).round().clamp(0, 255).to(torch.long).flatten(2)
    hist = torch.zeros((b, c, 256), device=image.device, dtype=torch.int32)
    hist.scatter_add_(2, bins, torch.ones_like(bins, dtype=torch.int32))
    cdf = hist.cumsum(2).to(torch.float32)
    present = hist > 0
    levels = torch.arange(256, device=image.device)
    first = torch.where(present, levels, torch.full((256,), 255, device=image.device)).amin(2, keepdim=True)
    cdf_first = cdf.gather(2, first)
    denom = float(h * w) - cdf_first
    mapped = (cdf.gather(2, bins) - cdf_first).div(denom.clamp_min(1))
    dynamic = present.sum(2, keepdim=True) > 1
    out = torch.where(dynamic, mapped, value.flatten(2)).reshape(b, c, h, w)
    return out.clamp(0, 1).to(image.dtype)


def _posterize(image: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    bits = (4 - (level * 4 / 10).floor()).clamp(1, 8).to(torch.int64)
    shift = (8 - bits).view(-1, 1, 1, 1)
    mask = torch.bitwise_left_shift(torch.full_like(shift, 255), shift).bitwise_and(255)
    value = image.float().mul(255).round().to(torch.int64)
    return value.bitwise_and(mask).float().div(255).to(image.dtype)


def _solarize(image: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    threshold = (256 - (level * 256 / 10).floor()).to(torch.int64)
    value = image.float().mul(255).round().to(torch.int64)
    return torch.where(value >= threshold.view(-1, 1, 1, 1), 255 - value, value).float().div(255).to(image.dtype)


def _affine(image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False).to(image.dtype)


def _geometric(image: torch.Tensor, operator: str, level: torch.Tensor,
               generator: torch.Generator) -> torch.Tensor:
    b, _, h, w = image.shape
    sign = torch.where(_rand(image, generator, b) > 0.5, -1.0, 1.0)
    theta = torch.zeros((b, 2, 3), device=image.device, dtype=torch.float32)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    x_to_y, y_to_x = float(h) / float(w), float(w) / float(h)
    if operator == "rotate":
        angle = sign * (level * 3).floor() * math.pi / 180
        cosine, sine = angle.cos(), angle.sin()
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine * x_to_y
        theta[:, 1, 0], theta[:, 1, 1] = sine * y_to_x, cosine
    elif operator == "shear_x":
        theta[:, 0, 1] = sign * level * 0.3 / 10 * x_to_y
    elif operator == "shear_y":
        theta[:, 1, 0] = sign * level * 0.3 / 10 * y_to_x
    elif operator == "translate_x":
        theta[:, 0, 2] = sign * level * (2 / 3) / 10
    elif operator == "translate_y":
        theta[:, 1, 2] = sign * level * (2 / 3) / 10
    else:
        raise ValueError(f"unknown geometric operator: {operator}")
    return _quantize(_affine(_quantize(image), theta))


def _motion_blur(image: torch.Tensor, level: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
    """Variable-angle line samples without a dense 41x41 convolution."""
    b, _, h, w = image.shape
    radius = (level * 2).clamp(1, 20).view(b, 1, 1)
    sigma = radius * 0.35 + 0.5
    angle = _rand(image, generator, b).mul(math.pi).view(b, 1, 1)
    yy = torch.linspace(-1, 1, h, device=image.device, dtype=torch.float32)
    xx = torch.linspace(-1, 1, w, device=image.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    base = torch.stack((gx.expand(b, -1, -1), gy.expand(b, -1, -1)), -1)
    result = torch.zeros_like(image.float())
    normalizer = torch.zeros((b, 1, 1, 1), device=image.device, dtype=torch.float32)
    # Scale all 21 taps to the sampled radius. Fixed two-pixel spacing would
    # reduce every radius below two to an identity-only central tap.
    for tap in range(-10, 11):
        distance = radius * (tap / 10)
        weight = torch.exp(-0.5 * (distance / sigma).square())
        weight = weight.view(b, 1, 1, 1)
        grid = base.clone()
        grid[..., 0] += distance * angle.cos() * (2 / max(w - 1, 1))
        grid[..., 1] += distance * angle.sin() * (2 / max(h - 1, 1))
        result += F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="border",
                                align_corners=True) * weight
        normalizer += weight
    return (result / normalizer.clamp_min(1e-6)).clamp(0, 1).to(image.dtype)


def _brightness(image: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    value = image.float()
    vmax = value.amax(1, keepdim=True)
    new_v = (vmax + level.view(-1, 1, 1, 1) * 0.1).clamp(0, 1)
    scaled = value * (new_v / vmax.clamp_min(1e-6))
    return torch.where(vmax > 1e-6, scaled, new_v.expand_as(value)).clamp(0, 1).to(image.dtype)


def _elastic(image: torch.Tensor, level: torch.Tensor,
             generator: torch.Generator) -> torch.Tensor:
    b, _, h, w = image.shape
    gh, gw = max(4, h // 16), max(4, w // 16)
    field = _rand(image, generator, b, 2, gh, gw).mul(2).sub(1)
    field = F.avg_pool2d(field, 3, stride=1, padding=1)
    field = F.interpolate(field, (h, w), mode="bilinear", align_corners=True)
    amplitude = (1 + 1.5 * level).view(b, 1, 1)
    yy = torch.linspace(-1, 1, h, device=image.device, dtype=torch.float32)
    xx = torch.linspace(-1, 1, w, device=image.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((gx.expand(b, -1, -1), gy.expand(b, -1, -1)), -1)
    grid[..., 0] += field[:, 0] * amplitude * (2 / max(w - 1, 1))
    grid[..., 1] += field[:, 1] * amplitude * (2 / max(h - 1, 1))
    return F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="reflection",
                         align_corners=True).clamp(0, 1).to(image.dtype)


def _dct_matrix(device: torch.device) -> torch.Tensor:
    n = torch.arange(8, device=device, dtype=torch.float32)
    k = n[:, None]
    mat = torch.cos(math.pi / 8 * (n[None, :] + 0.5) * k)
    mat[0] *= 1 / math.sqrt(8)
    mat[1:] *= math.sqrt(2 / 8)
    return mat


_QY = (16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,
       14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
       49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99)
_QC = (17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,
       47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
       99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99)


def _jpeg_compression(image: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """JPEG-style RGB->YCbCr block DCT, quantization, and inverse transform."""
    b, _, h, w = image.shape
    hp, wp = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    rgb = F.pad(image.float(), (0, wp - w, 0, hp - h), mode="replicate")
    r, g, bl = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    ycc = torch.cat((0.299*r + 0.587*g + 0.114*bl,
                     -0.168736*r - 0.331264*g + 0.5*bl + 0.5,
                     0.5*r - 0.418688*g - 0.081312*bl + 0.5), 1).mul(255).sub(128)
    blocks = ycc.reshape(b, 3, hp // 8, 8, wp // 8, 8).permute(0, 1, 2, 4, 3, 5)
    dct = _dct_matrix(image.device)
    coeff = dct @ blocks @ dct.t()
    qy = torch.tensor(_QY, device=image.device, dtype=torch.float32).reshape(1,1,1,1,8,8)
    qc = torch.tensor(_QC, device=image.device, dtype=torch.float32).reshape(1,1,1,1,8,8)
    quality = (100 - level * 8).clamp(10, 90).view(b,1,1,1,1,1)
    scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality) / 100
    quant = torch.cat((qy.expand(b,1,1,1,-1,-1), qc.expand(b,2,1,1,-1,-1)), 1) * scale
    decoded = dct.t() @ ((coeff / quant.clamp_min(1)).round() * quant) @ dct
    decoded = decoded.permute(0,1,2,4,3,5).reshape(b,3,hp,wp)[:, :, :h, :w].add(128).div(255)
    y, cb, cr = decoded[:,0:1], decoded[:,1:2] - 0.5, decoded[:,2:3] - 0.5
    return torch.cat((y + 1.402*cr, y - 0.344136*cb - 0.714136*cr, y + 1.772*cb), 1).clamp(0,1).to(image.dtype)


def _apply_gpu_augmix_image_operator(image: torch.Tensor, operator: str, *, severity: float,
                                     rng: torch.Generator | None = None) -> torch.Tensor:
    if image.device.type != "cuda":
        raise ValueError("GPU image operators require a CUDA image tensor")
    if (operator not in AUGMIX_GPU_OPERATORS or type(severity) not in (int, float)
            or not math.isfinite(severity) or not 0.1 <= severity <= 10):
        raise ValueError("invalid GPU AugMix operator or severity")
    generator = _rng(image, rng)
    if operator == "autocontrast":
        return _autocontrast(image)
    if operator == "equalize":
        return _equalize(image)
    level = _sample_level(image, float(severity), generator)
    if operator == "posterize":
        out = _posterize(image, level)
    elif operator == "solarize":
        out = _solarize(image, level)
    else:
        out = _geometric(image, operator, level, generator)
    return out.clamp(0, 1).to(image.dtype)


def apply_gpu_augmix_image_operator(image: torch.Tensor, operator: str, *, severity: float,
                                    rng: torch.Generator | None = None) -> torch.Tensor:
    _check(image)
    return _apply_gpu_augmix_image_operator(image, operator, severity=severity, rng=rng)


def apply_gpu_augmix_image_operator_prevalidated(
    image: torch.Tensor, operator: str, *, severity: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Trusted chain path: caller has validated the BCHW RGB anchor once."""
    return _apply_gpu_augmix_image_operator(image, operator, severity=severity, rng=rng)


def _apply_gpu_c5_image_operator(image: torch.Tensor, operator: str, *, severity: int = 5,
                                 rng: torch.Generator | None = None) -> torch.Tensor:
    if image.device.type != "cuda":
        raise ValueError("GPU image operators require a CUDA image tensor")
    if operator not in C5_GPU_OPERATORS or type(severity) is not int or not 1 <= severity <= 5:
        raise ValueError("invalid GPU C5 operator or severity")
    generator = _rng(image, rng)
    level = _sample_level(image, float(severity * 2), generator)
    if operator == "gaussian_noise":
        std = (0.06 + 0.055 * level).view(-1,1,1,1)
        out = image.float() + torch.randn(image.shape, device=image.device, dtype=torch.float32,
                                          generator=generator) * std
    elif operator == "motion_blur":
        out = _motion_blur(image, level, generator)
    elif operator == "brightness":
        out = _brightness(image, level)
    elif operator == "elastic_transform":
        out = _elastic(image, level, generator)
    else:
        out = _jpeg_compression(image, level)
    return out.clamp(0, 1).to(image.dtype)


def apply_gpu_c5_image_operator(image: torch.Tensor, operator: str, *, severity: int = 5,
                                rng: torch.Generator | None = None) -> torch.Tensor:
    _check(image)
    return _apply_gpu_c5_image_operator(image, operator, severity=severity, rng=rng)


def apply_gpu_c5_image_operator_prevalidated(
    image: torch.Tensor, operator: str, *, severity: int = 5,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Trusted evaluation path after the clean RGB batch was checked once."""
    return _apply_gpu_c5_image_operator(image, operator, severity=severity, rng=rng)


__all__ = ["AUGMIX_GPU_OPERATORS", "C5_GPU_OPERATORS", "GPU_AUGMIX_IMPLEMENTATION",
           "GPU_C5_IMPLEMENTATION", "validate_gpu_image", "apply_gpu_augmix_image_operator",
           "apply_gpu_augmix_image_operator_prevalidated", "apply_gpu_c5_image_operator",
           "apply_gpu_c5_image_operator_prevalidated"]
