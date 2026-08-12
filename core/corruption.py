"""Canonical batched online ECG corruption kernel.

The training cache contract is always raw-mV ``(B,1000,12)`` at 100 Hz.  This
module performs both linear resampling steps on the input device, applies the
five configured Torch operators only in the 500 Hz domain, and returns raw-mV
``(B,1000,12)``.  Per-sample z-score remains a model/trainer responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from util.augmentations.torch_operators import apply_operator_batch_prevalidated


INPUT_SAMPLING_RATE_HZ = 100
CORRUPTION_DOMAIN_SAMPLING_RATE_HZ = 500
OUTPUT_SAMPLING_RATE_HZ = 100
DURATION_SECONDS = 10
INPUT_POINTS = INPUT_SAMPLING_RATE_HZ * DURATION_SECONDS
CORRUPTION_DOMAIN_POINTS = CORRUPTION_DOMAIN_SAMPLING_RATE_HZ * DURATION_SECONDS
OUTPUT_POINTS = OUTPUT_SAMPLING_RATE_HZ * DURATION_SECONDS
INTERPOLATION_MODE = "linear"
INTERPOLATION_ALIGN_CORNERS = True
CANONICAL_OPERATORS = (
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
)
COMPOSITIONS = tuple(combinations(CANONICAL_OPERATORS, 2)) + tuple(
    combinations(CANONICAL_OPERATORS, 3)
)
_COMPOSITION_MASK_ROWS = tuple(
    tuple(operator in composition for operator in CANONICAL_OPERATORS)
    for composition in COMPOSITIONS
)
_DEPTHS = tuple(len(composition) for composition in COMPOSITIONS)


@dataclass(frozen=True)
class CorruptionDiagnostics:
    """Device-resident per-sample corruption provenance."""

    composition_index: torch.Tensor
    depth: torch.Tensor
    operator_mask: torch.Tensor
    output_nonfinite_count: torch.Tensor


@dataclass(frozen=True)
class CorruptionBatch:
    waveform_raw_100hz: torch.Tensor
    diagnostics: CorruptionDiagnostics


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("corruption generator device must match waveform device")
    if device.type == "cuda":
        current_index = torch.cuda.current_device()
        waveform_index = current_index if device.index is None else device.index
        generator_index = (
            current_index if generator_device.index is None else generator_device.index
        )
        if waveform_index != generator_index:
            raise ValueError(
                "corruption generator CUDA index must match waveform CUDA index"
            )


def _validate_clean_raw_100hz(signal: torch.Tensor) -> torch.Tensor:
    if not isinstance(signal, torch.Tensor):
        raise TypeError("clean_raw_100hz must be a torch.Tensor")
    if signal.ndim != 3 or signal.shape[0] < 1 or tuple(signal.shape[1:]) != (1000, 12):
        raise ValueError(
            "clean_raw_100hz must have shape (B,1000,12) in PTB-XL lead order"
        )
    if not signal.is_floating_point():
        raise TypeError("clean_raw_100hz must use a floating-point dtype")
    # This is the sole tensor-wide finite validation/synchronization in the
    # operator pipeline.  Internal operators use their prevalidated batch hook.
    if not bool(torch.isfinite(signal).all().item()):
        raise ValueError("clean_raw_100hz contains NaN or infinity")
    return signal


def _validate_operator_params(
    operator_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = [name for name in CANONICAL_OPERATORS if name not in operator_params]
    extra = sorted(set(operator_params) - set(CANONICAL_OPERATORS))
    if missing or extra:
        raise ValueError(
            "operator_params must contain exactly the canonical five operators; "
            f"missing={missing}, extra={extra}"
        )
    output: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_OPERATORS:
        params = operator_params[name]
        if not isinstance(params, Mapping):
            raise TypeError(f"operator_params[{name!r}] must be a mapping")
        output[name] = dict(params)
    return output


def _resample_btc(signal: torch.Tensor, *, points: int) -> torch.Tensor:
    return F.interpolate(
        signal.transpose(1, 2),
        size=int(points),
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()


def _resolve_composition_indices(
    composition_indices: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if composition_indices is None:
        return torch.randint(
            0,
            len(COMPOSITIONS),
            (batch_size,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
    if not isinstance(composition_indices, torch.Tensor):
        raise TypeError("composition_indices must be a torch.Tensor or None")
    if composition_indices.device != device:
        raise ValueError("composition_indices device must match waveform device")
    if composition_indices.ndim != 1 or composition_indices.shape[0] != batch_size:
        raise ValueError("composition_indices must have shape (B,)")
    if composition_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("composition_indices must use an integer dtype")
    indices = composition_indices.to(dtype=torch.int64)
    if bool(((indices < 0) | (indices >= len(COMPOSITIONS))).any().item()):
        raise ValueError(f"composition_indices must be in [0, {len(COMPOSITIONS) - 1}]")
    return indices


def generate_canonical_corruption(
    clean_raw_100hz: torch.Tensor,
    *,
    operator_params: Mapping[str, Mapping[str, Any]],
    generator: torch.Generator,
    composition_indices: torch.Tensor | None = None,
    _input_prevalidated: bool = False,
) -> CorruptionBatch:
    """Generate one depth-2/3 corruption view in the canonical 500 Hz domain.

    Every operator is invoked once for the whole batch in canonical order.  A
    device-side composition mask retains only the operators selected for each
    sample, avoiding up to twenty per-composition Python branches and their
    repeated CUDA finite checks.
    """

    clean = (
        clean_raw_100hz
        if _input_prevalidated
        else _validate_clean_raw_100hz(clean_raw_100hz)
    )
    _validate_generator(generator, clean.device)
    params = _validate_operator_params(operator_params)
    indices = _resolve_composition_indices(
        composition_indices,
        batch_size=int(clean.shape[0]),
        device=clean.device,
        generator=generator,
    )
    mask_table = torch.tensor(
        _COMPOSITION_MASK_ROWS,
        device=clean.device,
        dtype=torch.bool,
    )
    depth_table = torch.tensor(_DEPTHS, device=clean.device, dtype=torch.int64)
    operator_mask = mask_table[indices]

    waveform = _resample_btc(clean.to(dtype=torch.float32), points=CORRUPTION_DOMAIN_POINTS)
    for operator_index, operator in enumerate(CANONICAL_OPERATORS):
        candidate = apply_operator_batch_prevalidated(
            operator,
            waveform,
            params=params[operator],
            sampling_rate_hz=CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
            rng=generator,
        )
        selected = operator_mask[:, operator_index].view(-1, 1, 1)
        waveform = torch.where(selected, candidate, waveform)

    output = _resample_btc(waveform, points=OUTPUT_POINTS)
    nonfinite_count = (~torch.isfinite(output)).sum(dim=(1, 2), dtype=torch.int64)
    # Preserve a finite downstream contract without a second host synchronization;
    # diagnostics retain the exact per-sample repair count for logging/gating.
    output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    return CorruptionBatch(
        waveform_raw_100hz=output,
        diagnostics=CorruptionDiagnostics(
            composition_index=indices.contiguous(),
            depth=depth_table[indices].contiguous(),
            operator_mask=operator_mask.contiguous(),
            output_nonfinite_count=nonfinite_count.contiguous(),
        ),
    )


__all__ = [
    "CANONICAL_OPERATORS",
    "COMPOSITIONS",
    "CORRUPTION_DOMAIN_SAMPLING_RATE_HZ",
    "CorruptionBatch",
    "CorruptionDiagnostics",
    "INPUT_SAMPLING_RATE_HZ",
    "INTERPOLATION_ALIGN_CORNERS",
    "INTERPOLATION_MODE",
    "OUTPUT_SAMPLING_RATE_HZ",
    "generate_canonical_corruption",
]
