"""Tensor-only equivalents of the frozen R1 uint8 image preprocessing.

The CPU Torch uint8 bicubic antialias path is separable fixed-point filtering,
with uint8 rounding/saturation after EACH axis. CUDA float interpolation is not
equivalent. Coefficient semantics follow PyTorch v2.6.0 UpSampleKernel.cpp and
UpSampleKernelAVXAntialias.h (https://github.com/pytorch/pytorch/tree/v2.6.0/aten/src/ATen/native/cpu).
Only tiny geometry-dependent coefficients are built on CPU; pixels stay on
their input device. No CUDA extension, driver change, or image cache is needed.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch


def _cubic(value: float) -> float:
    value = abs(value)
    if value < 1:
        return (1.5 * value - 2.5) * value * value + 1.0
    if value < 2:
        return ((-0.5 * value + 2.5) * value - 4.0) * value + 2.0
    return 0.0


@lru_cache(maxsize=16)
def _axis_coefficients(source: int, target: int, device: str):
    scale = source / target
    support = 2.0 * max(scale, 1.0)
    inverse_scale = 1.0 / scale if scale >= 1.0 else 1.0
    width = math.ceil(support) * 2 + 1
    indices, weights = [], []
    maximum = 0.0
    for output in range(target):
        center = scale * (output + 0.5)
        start = max(int(center - support + 0.5), 0)
        count = min(max(min(int(center + support + 0.5), source) - start, 0), width)
        row = [_cubic((j + start - center + 0.5) * inverse_scale) for j in range(count)]
        total = sum(row)
        row = [value / total for value in row]
        maximum = max(maximum, max(row))
        indices.append([min(start + j, source - 1) for j in range(width)])
        weights.append(row + [0.0] * (width - count))
    precision = 0
    while precision < 22 and int(0.5 + maximum * (1 << (precision + 1))) < (1 << 15):
        precision += 1
    fixed = [[int(value * (1 << precision) + (0.5 if value >= 0 else -0.5)) for value in row]
             for row in weights]
    return (torch.tensor(indices, dtype=torch.int64, device=device),
            torch.tensor(fixed, dtype=torch.int32, device=device), precision)


@torch.inference_mode()
def resize_uint8_cpu_equivalent(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """BCHW uint8 bicubic antialias resize with CPU integer rounding semantics."""
    if images.ndim != 4 or images.dtype != torch.uint8 or len(size) != 2 or min(size) < 1:
        raise ValueError("expected BCHW uint8 images and positive (height, width)")
    result = images
    for axis, target in ((3, size[1]), (2, size[0])):
        if result.shape[axis] == target:
            continue
        indices, weights, precision = _axis_coefficients(result.shape[axis], target, str(images.device))
        shape = list(result.shape)
        shape[axis] = target
        accumulator = torch.full(shape, 1 << (precision - 1), dtype=torch.int32, device=images.device)
        weight_shape = [1, 1, 1, 1]
        weight_shape[axis] = target
        for tap in range(indices.shape[1]):
            values = result.index_select(axis, indices[:, tap]).to(torch.int32)
            accumulator.add_(values * weights[:, tap].reshape(weight_shape))
        result = accumulator.bitwise_right_shift_(precision).clamp_(0, 255).to(torch.uint8)
    return result
