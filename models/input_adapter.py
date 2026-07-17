"""Canonical raw-ECG to managed-model input adaptation.

Every PN2021 method produces the same raw physical-mV bottleneck:
``(B,1000,12)`` at 100 Hz in PTB-XL lead order.  This module owns the only
model-facing conversion from that bottleneck.  Resampling deliberately happens
on the input tensor's current device before per-sample normalization, so online
training and evaluation can share exactly the same operation order.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.contracts import (
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
    validate_model_input,
)


CANONICAL_SAMPLING_RATE_HZ = 100
CANONICAL_POINTS = 1000
CANONICAL_CHANNELS = 12
ECGFOUNDER_TARGET_POINTS = 5000
INTERPOLATION_MODE = "linear"
INTERPOLATION_ALIGN_CORNERS = True


def _normalization_epsilon(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("normalization epsilon must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("normalization epsilon must be finite and positive")
    return resolved


def prepare_canonical_model_input(
    raw_100hz_btc: torch.Tensor,
    spec: ModelSpec,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Convert canonical raw 100 Hz BTC waveforms to one managed model input.

    The fixed order is:

    ``raw mV BTC -> sanitize -> model-domain linear adaptation -> global
    per-sample z-score -> contiguous BCT``.

    EfficientNet keeps the 100 Hz waveform unchanged before normalization.
    ECGFounder linearly upsamples to 5000 points on the tensor's current device
    with ``align_corners=True`` before normalization.  The input is never
    modified in place and the result is always finite float32.
    """

    if not isinstance(raw_100hz_btc, torch.Tensor):
        raise TypeError("canonical raw waveform must be a torch.Tensor")
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError(
            "canonical input adapter supports only managed EfficientNet1DV2 "
            "and ECGFounder ModelSpecs"
        )
    if (
        raw_100hz_btc.ndim != 3
        or raw_100hz_btc.shape[0] < 1
        or tuple(raw_100hz_btc.shape[1:])
        != (CANONICAL_POINTS, CANONICAL_CHANNELS)
    ):
        raise ValueError(
            "canonical raw waveform must have shape (B,1000,12) in "
            "time-channel layout"
        )
    if not raw_100hz_btc.is_floating_point():
        raise TypeError("canonical raw waveform must use a floating dtype")
    resolved_epsilon = _normalization_epsilon(epsilon)

    # Explicit zero repair matches the project's sanitization contract while
    # remaining device-local and non-mutating.
    model_domain_btc = torch.nan_to_num(
        raw_100hz_btc.to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if spec == ECGFOUNDER_SPEC:
        model_domain_btc = F.interpolate(
            model_domain_btc.transpose(1, 2),
            size=ECGFOUNDER_TARGET_POINTS,
            mode=INTERPOLATION_MODE,
            align_corners=INTERPOLATION_ALIGN_CORNERS,
        ).transpose(1, 2).contiguous()

    mean = model_domain_btc.mean(dim=(1, 2), keepdim=True)
    scale = model_domain_btc.std(
        dim=(1, 2), correction=0, keepdim=True
    ).clamp_min(resolved_epsilon)
    normalized_bct = ((model_domain_btc - mean) / scale).transpose(1, 2)
    # Extremely large but finite raw values can overflow a reduction. Repair
    # device-side so this hot path does not need a host-synchronizing finite
    # assertion for every fixed20 exposure.
    normalized_bct = torch.nan_to_num(
        normalized_bct,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()
    return validate_model_input(normalized_bct, spec, check_finite=False)


__all__ = [
    "CANONICAL_CHANNELS",
    "CANONICAL_POINTS",
    "CANONICAL_SAMPLING_RATE_HZ",
    "ECGFOUNDER_TARGET_POINTS",
    "INTERPOLATION_ALIGN_CORNERS",
    "INTERPOLATION_MODE",
    "prepare_canonical_model_input",
]
