# AugMix operator formulas adapted from Copyright 2019 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference operators on native-size ECG paper, before PULSE processing.

AugMix formulas: google-research/augmix, commit 9b9824c7 (Apache-2.0).
C15: imagecorruptions 1.1.2, the rectangular-image extension of ImageNet-C.
These are not the 2026-09-18 GPU approximations, nor CIFAR-C severity tables.
The ECG canvas and teacher-forced PULSE objective still differ from AugMix's
natural-image classification experiment. No claim of clinical label preservation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps
import torch


AUGMIX_IMPLEMENTATION = "augmix_pil_reference_v1"
C_IMPLEMENTATION = "imagecorruptions_1.1.2_v1"
AUGMIX_UPSTREAM_COMMIT = "9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4"
AUGMIX_TRAIN_OPERATORS = (
    "autocontrast", "equalize", "posterize", "rotate", "solarize",
    "shear_x", "shear_y", "translate_x", "translate_y",
)

C_IMAGE_OPERATORS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
_REFERENCE_LOCK = threading.Lock()


def _check(image: torch.Tensor) -> None:
    if (image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1
            or not image.is_floating_point()):
        raise ValueError("expected nonempty floating BCHW RGB")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("expected finite RGB in [0,1]")


def _to_bytes(image):
    return (image.detach().float().cpu().permute(0, 2, 3, 1) * 255).round().byte().numpy()


def _from_bytes(images, like):
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(
        device=like.device, dtype=like.dtype).div(255)


def _augmix_pil(image, operator, severity, rng):
    """Google's PIL formulas; only the RNG and rectangular output size differ."""
    if operator == "autocontrast":
        return ImageOps.autocontrast(image)
    if operator == "equalize":
        return ImageOps.equalize(image)
    level = float(torch.rand((), generator=rng, device=rng.device)) * (severity - 0.1) + 0.1
    if operator == "posterize":
        return ImageOps.posterize(image, 4 - int(level * 4 / 10))
    if operator == "solarize":
        return ImageOps.solarize(image, 256 - int(level * 256 / 10))
    sign = -1 if float(torch.rand((), generator=rng, device=rng.device)) > 0.5 else 1
    if operator == "rotate":
        return image.rotate(sign * int(level * 30 / 10), resample=Image.Resampling.BILINEAR)
    value = sign * level * 0.3 / 10
    if operator == "shear_x":
        matrix = (1, value, 0, 0, 1, 0)
    elif operator == "shear_y":
        matrix = (1, 0, 0, value, 1, 0)
    elif operator == "translate_x":
        matrix = (1, 0, sign * int(level * (image.width / 3) / 10), 0, 1, 0)
    else:
        matrix = (1, 0, 0, 0, 1, sign * int(level * (image.height / 3) / 10))
    return image.transform(image.size, Image.Transform.AFFINE, matrix,
                           resample=Image.Resampling.BILINEAR)


def apply_augmix_image_operator(image: torch.Tensor, operator: str, *, severity: float,
                                rng: torch.Generator) -> torch.Tensor:
    """Original nine-op pool, severity 0.1--10; no extra per-op residual mixing."""
    _check(image)
    if (operator not in AUGMIX_TRAIN_OPERATORS or type(severity) not in (int, float)
            or not math.isfinite(severity) or not 0.1 <= severity <= 10):
        raise ValueError("invalid reference AugMix operator or severity")
    results = [np.array(_augmix_pil(Image.fromarray(rgb), operator, severity, rng), copy=True)
               for rgb in _to_bytes(image)]
    return _from_bytes(results, image)


@lru_cache(maxsize=1)
def c_reference_identity():
    """Fail before model loading if the separately pinned CPU backend is absent."""
    from imagecorruptions import corruptions
    import cv2

    versions = {name: importlib.metadata.version(name) for name in (
        "imagecorruptions", "scikit-image", "opencv-python-headless", "numpy", "scipy", "Pillow",
        "numba", "llvmlite")}
    if versions["imagecorruptions"] != "1.1.2" or versions["scikit-image"] != "0.22.0":
        raise ValueError("C15 reference requires environments/pulse-image-requirements.txt")
    cv2.setNumThreads(1)
    source = Path(corruptions.__file__)
    members = [source, *sorted((source.parent / "frost").glob("*"))]
    if len(members) != 7:
        raise ValueError("reference frost assets are missing")
    return {"implementation": C_IMPLEMENTATION, "versions": versions,
            "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in members},
            "compatibility": "gaussian_channel_axis_explicit_impulse_rng_exact_glass_loop_v1",
            "input_quantization": "round_rgb_uint8", "size": "native_canvas"}


def _reference_gaussian(image, sigma=1, *, multichannel=False, **kwargs):
    from skimage.filters import gaussian
    return gaussian(image, sigma=sigma, channel_axis=-1 if multichannel else None, **kwargs)


@lru_cache(maxsize=1)
def _glass_loop():
    from numba import njit

    @njit
    def apply(image, offsets, delta):
        for row in range(offsets.shape[0]):
            h = image.shape[0] - delta - row
            for column in range(offsets.shape[1]):
                w = image.shape[1] - delta - column
                dx, dy = offsets[row, column]
                # Preserve the reference's NumPy view-assignment semantics,
                # including its aliasing, rather than replacing it with a swap.
                for channel in range(3):
                    image[h, w, channel] = image[h + dy, w + dx, channel]
        return image
    return apply


def _glass_blur_exact(image, severity):
    sigma, delta, iterations = ((0.7, 1, 2), (0.9, 2, 1), (1., 2, 3),
                                 (1.1, 3, 2), (1.5, 4, 2))[severity - 1]
    value = np.uint8(_reference_gaussian(np.array(image) / 255., sigma, multichannel=True) * 255)
    for _ in range(iterations):
        # Bulk draws consume the same RandomState stream as per-pixel size=2 draws.
        offsets = np.random.randint(-delta, delta, size=(value.shape[0] - 2 * delta,
                                                        value.shape[1] - 2 * delta, 2))
        _glass_loop()(value, offsets, delta)
    return np.clip(_reference_gaussian(value / 255., sigma, multichannel=True), 0, 1) * 255


def apply_c_image_operator(image: torch.Tensor, operator: str, *, severity: int,
                           rng: torch.Generator) -> torch.Tensor:
    """Real C15 mechanisms and ImageNet-C tables, with caller-owned randomness.

    imagecorruptions uses NumPy's legacy global RNG. Calls here are serialized
    and restore it even on failure. The evaluator calls this on its main thread;
    do not concurrently invoke unrelated NumPy-random work in that process.
    """
    _check(image)
    if operator not in C_IMAGE_OPERATORS or type(severity) is not int or not 1 <= severity <= 5:
        raise ValueError("invalid C operator or severity")
    if min(image.shape[-2:]) < 32:
        raise ValueError("C15 reference requires image dimensions >= 32")
    c_reference_identity()
    from imagecorruptions import corruptions
    from skimage.util import random_noise

    results = []
    for rgb in _to_bytes(image):
        seed = int(torch.randint(2**32, (), generator=rng, device=rng.device, dtype=torch.int64))
        with _REFERENCE_LOCK:
            state, gaussian = np.random.get_state(), corruptions.gaussian
            try:
                np.random.seed(seed)
                corruptions.gaussian = _reference_gaussian
                if operator == "impulse_noise":
                    # Modern skimage has a private default RNG; seeding NumPy is insufficient.
                    amount = (0.03, 0.06, 0.09, 0.17, 0.27)[severity - 1]
                    result = random_noise(rgb / 255., mode="s&p", amount=amount,
                                          rng=np.random.default_rng(seed)) * 255
                elif operator == "glass_blur":
                    result = _glass_blur_exact(Image.fromarray(rgb), severity)
                else:
                    result = getattr(corruptions, operator)(Image.fromarray(rgb), severity=severity)
                result = np.array(result, copy=True)
            finally:
                corruptions.gaussian = gaussian
                np.random.set_state(state)
        if result.shape != rgb.shape or not np.isfinite(result).all():
            raise ValueError(f"invalid C15 reference output: {operator}")
        results.append(np.clip(result, 0, 255).astype(np.uint8))
    return _from_bytes(results, image)
