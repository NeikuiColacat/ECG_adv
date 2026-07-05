"""Locked PN2021-C protocol identifiers and metadata helpers."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any


LOCKED_MAIN_INPUT_ORDER_ID = "waveform_bottleneck_then_corrupt_then_zscore"
LOCKED_EFFNET_CORRUPTION_INPUT = "raw_first"
LOCKED_ECGFOUNDER_CORRUPTION_INPUT = "bottleneck5000"
OFFICIAL_S5_COMPOSITE_CORRUPTION_SET = "official_s5_depth23_composite"
OFFICIAL_S5_PUBLIC_SEVERITY = 5
OFFICIAL_S5_SEVERITY_PROFILE = "standard"
OFFICIAL_S5_COMPOSITE_OPS = (
    "baseline_shift",
    "baseline_wander",
    "emg_noise",
    "powerline_noise",
    "random_leads_masking",
)
OFFICIAL_S5_COMPOSITE_DEPTHS = (2, 3)

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


def ecgfounder_pn2021c_corruption_order(corruption_input: str) -> list[str]:
    """Return the report-only waveform order for ECGFounder PN2021-C eval."""

    if corruption_input == "native_raw_first":
        return [
            "wfdb_read_native_fs_native_length",
            "lead_reorder_nan_guard_no_resample_no_zscore",
            "corruption_with_native_sample_rate",
            "resample_pad_or_truncate_to_100hz_1000",
            "per_sample_global_zscore",
            "center_crop",
            "ecgfounder_resample_1000_to_5000_no_second_zscore",
            "model",
        ]
    if corruption_input == "raw_first":
        return [
            "wfdb_read",
            "lead_reorder_nan_guard_resample_pad_no_zscore",
            "corruption",
            "per_sample_global_zscore",
            "center_crop",
            "ecgfounder_resample_1000_to_5000_no_second_zscore",
            "model",
        ]
    if corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT:
        return list(locked_protocol_metadata("ecgfounder")["waveform_order"])
    if corruption_input == "preprocessed_cache":
        return [
            "load_100hz1000_per_sample_global_zscore_cache",
            "center_crop_or_full_signal",
            "corruption",
            "ecgfounder_resample_1000_to_5000_and_global_zscore",
            "model",
        ]
    raise ValueError(f"unknown ECGFounder PN2021-C corruption_input: {corruption_input!r}")


def official_s5_composite_names(depths: Sequence[int] | None = None) -> list[str]:
    """Return official severity-5 composite corruption names in stable order."""

    selected_depths = tuple(OFFICIAL_S5_COMPOSITE_DEPTHS if depths is None else depths)
    names: list[str] = []
    for depth in selected_depths:
        d = int(depth)
        if d < 1 or d > len(OFFICIAL_S5_COMPOSITE_OPS):
            raise ValueError(f"invalid official S5 composite depth: {depth!r}")
        names.extend("+".join(combo) for combo in itertools.combinations(OFFICIAL_S5_COMPOSITE_OPS, d))
    return names


def official_s5_depth23_composites() -> list[str]:
    """Return the 10 pair plus 10 triple official severity-5 composite names."""

    return official_s5_composite_names(OFFICIAL_S5_COMPOSITE_DEPTHS)
