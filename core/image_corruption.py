"""Configuration, dispatch and recorded identities for PULSE image augmentation.

The retained mild operators below preserve geometry. Versioned AugMix and
paper-image implementations have their own explicit pools and source owners.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F

IMAGE_OPERATORS = ("paper_texture", "grid_fade", "tone", "shadow")
GPU_IMAGE_IMPLEMENTATIONS = ("augmix_torch_gpu_v2", "paper_ecg_torch_v1")
IMAGE_IMPLEMENTATION_SOURCES = {
    "augmix_pil_reference_v1": ("core/image_augmix_c.py",),
    "augmix_torch_gpu_v2": ("core/image_augmix_gpu.py",),
    "paper_ecg_torch_v1": ("core/paper_ecg.py", "core/image_augmix_gpu.py"),
}


def validate_image_config(config, *, for_execution=False):
    operators = tuple(config.get("operators", ()))
    implementation = config.get("implementation")
    reference = implementation in IMAGE_IMPLEMENTATION_SOURCES
    if "implementation" in config or operators != IMAGE_OPERATORS:
        from core.image_augmix_c import AUGMIX_TRAIN_OPERATORS
    expected = {"operators", "waveform_strength", "jsd_weight"}
    expected |= {"implementation", "severity"} if reference else {"strength"}
    if set(config) != expected:
        raise ValueError("invalid hybrid image configuration")
    if reference:
        from core.paper_ecg import PAPER_TRAIN_OPERATORS
        paper = implementation == "paper_ecg_torch_v1"
        if operators != (PAPER_TRAIN_OPERATORS if paper else AUGMIX_TRAIN_OPERATORS):
            raise ValueError("paper ECG requires its ten training operators" if paper else
                             "reference AugMix requires its nine training operators")
        if implementation in GPU_IMAGE_IMPLEMENTATIONS and config.get("waveform_strength") != 0:
            raise ValueError("GPU image augmentation is image-only and requires waveform_strength=0")
        level = config["severity"]
        if type(level) not in (int, float) or not math.isfinite(level) or not 0.1 <= level <= (5 if paper else 10):
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


def image_operator(config):
    """Resolve a code-owned implementation after configuration validation."""
    implementation = config.get("implementation")
    if implementation is None:
        return partial(apply_image_operator, strength=config["strength"])
    if implementation == "augmix_pil_reference_v1":
        from core.image_augmix_c import apply_augmix_image_operator as apply
    elif implementation == "augmix_torch_gpu_v2":
        from core.image_augmix_gpu import apply_gpu_augmix_image_operator_prevalidated as apply
    elif implementation == "paper_ecg_torch_v1":
        from core.paper_ecg import apply_paper_operator
        return partial(apply_paper_operator, severity=config["severity"], validate=False)
    else:
        raise ValueError(f"unknown image implementation: {implementation}")
    return partial(apply, severity=config["severity"])


def gpu_image_identity(implementation):
    if implementation not in GPU_IMAGE_IMPLEMENTATIONS:
        raise ValueError("unknown GPU image implementation")
    paper = implementation == "paper_ecg_torch_v1"
    return {"implementation": implementation, "device_policy": "same_device_as_rendered_rgb",
            "randomness": "caller_owned_torch_generator_on_input_device",
            "parity": "procedural_appearance_not_author_pixel_equivalence" if paper else "visual_approximation_not_reference_pixel_equivalence",
            "host_tensor_transfer": "none_inside_prevalidated_operator" if paper else "none_inside_operator",
            "waveform_corruption": "disabled"}


def image_augmentation_protocol(config, width):
    """One producer for the image recipe fields preserved in training artifacts."""
    implementation = config.get("implementation")
    gpu = implementation in GPU_IMAGE_IMPLEMENTATIONS
    protocol = {"image_augmentation": config,
        "augmentation_topology": "image_only_gpu_branches_v1" if gpu else "shared_waveform_image_branches_v1",
        "mix_residual": "clean_render" if gpu else "corrupted_waveform_render",
        "jsd_weight": config["jsd_weight"] if width else 0.0}
    if gpu:
        protocol["image_gpu"] = gpu_image_identity(implementation)
    elif implementation == "augmix_pil_reference_v1":
        import importlib.metadata
        from core.image_augmix_c import AUGMIX_UPSTREAM_COMMIT
        protocol["image_reference"] = {"implementation": implementation, "upstream_commit": AUGMIX_UPSTREAM_COMMIT,
            "pillow_version": importlib.metadata.version("Pillow"), "input_quantization": "round_rgb_uint8", "size": "native_canvas"}
    elif implementation is not None:
        raise ValueError(f"unsupported image implementation: {implementation}")
    return protocol


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
