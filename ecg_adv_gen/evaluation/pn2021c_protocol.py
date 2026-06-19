"""Locked PN2021-C protocol identifiers and metadata helpers."""

from __future__ import annotations

from typing import Any


LOCKED_MAIN_INPUT_ORDER_ID = "waveform_bottleneck_then_corrupt_then_zscore"
LOCKED_EFFNET_CORRUPTION_INPUT = "raw_first"
LOCKED_ECGFOUNDER_CORRUPTION_INPUT = "bottleneck5000"

_COMMON = {
    "input_order_id": LOCKED_MAIN_INPUT_ORDER_ID,
    "zscore_timing": "after_corruption_before_model",
    "corrupts_pre_zscore_waveform": True,
    "diagnostic_stabilizer_allowed": False,
}

_MODEL_PROTOCOLS = {
    "efficientnet1dv2": {
        **_COMMON,
        "model_family": "efficientnet1dv2",
        "corruption_input": LOCKED_EFFNET_CORRUPTION_INPUT,
        "waveform_order": [
            "raw_ecg",
            "canonical_lead_order",
            "resample_pad_or_truncate_to_100hz_1000_no_zscore",
            "corruption",
            "per_sample_global_zscore",
            "model",
        ],
    },
    "ecgfounder": {
        **_COMMON,
        "model_family": "ecgfounder",
        "corruption_input": LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
        "waveform_order": [
            "raw_ecg",
            "canonical_lead_order",
            "resample_pad_or_truncate_to_100hz_1000_no_zscore",
            "interpolate_100hz_1000_to_500hz_5000_no_zscore",
            "corruption",
            "per_sample_global_zscore",
            "model",
        ],
    },
}


def locked_protocol_metadata(model_family: str) -> dict[str, Any]:
    """Return a copy of the locked PN2021-C protocol metadata."""

    key = str(model_family).strip().lower()
    if key not in _MODEL_PROTOCOLS:
        raise ValueError(f"unknown PN2021-C locked protocol model family: {model_family!r}")
    return dict(_MODEL_PROTOCOLS[key])
