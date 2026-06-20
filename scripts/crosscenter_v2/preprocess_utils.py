"""Compatibility wrapper for classifier ECG preprocessing helpers.

The implementation now lives in :mod:`ecg_adv_gen.preprocessing.classifier`.
Keep this module so legacy scripts can continue importing the historical path
while package code uses ``ecg_adv_gen.preprocessing`` directly.
"""

from ecg_adv_gen.preprocessing.classifier import (
    EXPECTED_LEADS,
    EXPECTED_LEADS_UPPER,
    _resolve_preprocess_flags,
    crop_signal_tc,
    filter_bandpass_safe,
    pad_or_truncate_tc,
    per_sample_zscore,
    reorder_leads_tc,
    resample_tc,
    unified_preprocess_to_1000,
)

__all__ = [
    "EXPECTED_LEADS",
    "EXPECTED_LEADS_UPPER",
    "_resolve_preprocess_flags",
    "crop_signal_tc",
    "filter_bandpass_safe",
    "pad_or_truncate_tc",
    "per_sample_zscore",
    "reorder_leads_tc",
    "resample_tc",
    "unified_preprocess_to_1000",
]
