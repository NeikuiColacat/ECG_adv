"""Classifier ECG preprocessing helpers.

This module owns shared classifier preprocessing behavior so package code can
depend on ``ecg_adv_gen.preprocessing``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample


EXPECTED_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
EXPECTED_LEADS_UPPER = [lead.upper() for lead in EXPECTED_LEADS]


def filter_bandpass_safe(signal: np.ndarray, fs: int | float, skip_notch_threshold: int | float = 150) -> np.ndarray:
    """Bandpass + baseline removal, skipping notch when Nyquist is unstable."""

    filtered = signal.astype(np.float64, copy=True)

    if fs > skip_notch_threshold:
        b, a = iirnotch(50.0, 30.0, fs)
        for channel in range(filtered.shape[0]):
            filtered[channel] = filtfilt(b, a, filtered[channel])

    high_cut = min(40.0, 0.95 * fs / 2)
    low_cut = 0.67
    b, a = butter(N=4, Wn=[low_cut, high_cut], btype="bandpass", fs=fs)
    for channel in range(filtered.shape[0]):
        filtered[channel] = filtfilt(b, a, filtered[channel])

    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= filtered.shape[1]:
        kernel_size = filtered.shape[1] - 1
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            return filtered.astype(np.float32)

    baseline = np.zeros_like(filtered)
    for channel in range(filtered.shape[0]):
        baseline[channel] = medfilt(filtered[channel], kernel_size=kernel_size)

    return (filtered - baseline).astype(np.float32)


def reorder_leads_tc(signal_tc: np.ndarray, source_leads: list[str]) -> np.ndarray | None:
    """Reorder ``(time, channels)`` signal into canonical PTB-XL lead order."""

    source_upper = [lead.strip().upper() for lead in source_leads]
    result = np.zeros((signal_tc.shape[0], 12), dtype=signal_tc.dtype)
    for index, lead in enumerate(EXPECTED_LEADS_UPPER):
        if lead in source_upper:
            result[:, index] = signal_tc[:, source_upper.index(lead)]
        else:
            return None
    return result


def per_sample_zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-sample global z-score over the full record."""

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return ((signal - mean) / (std + eps)).astype(np.float32)


def resample_tc(signal_tc: np.ndarray, source_fs: int | float, target_fs: int | float) -> np.ndarray:
    """Resample ``(time, channels)`` along the time axis."""

    if int(source_fs) == int(target_fs):
        return signal_tc.astype(np.float32)
    n_target = int(round(signal_tc.shape[0] * target_fs / source_fs))
    return resample(signal_tc, n_target, axis=0).astype(np.float32)


def pad_or_truncate_tc(signal_tc: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate ``(time, channels)`` to ``target_len``."""

    n = signal_tc.shape[0]
    if n < target_len:
        pad = np.zeros((target_len - n, signal_tc.shape[1]), dtype=signal_tc.dtype)
        return np.concatenate([signal_tc, pad], axis=0)
    if n > target_len:
        return signal_tc[:target_len]
    return signal_tc


def _resolve_preprocess_flags(
    apply_filter: bool,
    apply_zscore: bool,
    preprocess_mode: str | None = None,
    norm_mode: str | None = None,
) -> tuple[bool, bool]:
    """Map named preprocessing modes to legacy filter/z-score flags."""

    if preprocess_mode is not None:
        if preprocess_mode == "minimal_resample":
            apply_filter = False
        elif preprocess_mode == "legacy_ecgfounder_filter":
            apply_filter = True
        elif preprocess_mode == "raw_for_generation_or_digital":
            apply_filter = False
        else:
            raise ValueError(f"unknown preprocess_mode={preprocess_mode!r}")

    if norm_mode is not None:
        if norm_mode == "per_sample_global":
            apply_zscore = True
        elif norm_mode == "none":
            apply_zscore = False
        else:
            raise ValueError(f"unknown norm_mode={norm_mode!r}")

    return bool(apply_filter), bool(apply_zscore)


def unified_preprocess_to_1000(
    signal_tc: np.ndarray,
    fs: int | float,
    source_leads: list[str] | None = None,
    target_fs: int = 100,
    target_len: int = 1000,
    apply_filter: bool = True,
    apply_zscore: bool = True,
    preprocess_mode: str | None = None,
    norm_mode: str | None = None,
) -> np.ndarray | None:
    """Full classifier preprocessing up to ``(target_len, 12)``.

    Returns ``None`` on malformed records or filtering failures, preserving the
    legacy evaluation-loader contract.
    """

    apply_filter, apply_zscore = _resolve_preprocess_flags(
        apply_filter=apply_filter,
        apply_zscore=apply_zscore,
        preprocess_mode=preprocess_mode,
        norm_mode=norm_mode,
    )
    signal_tc = np.asarray(signal_tc, dtype=np.float32)
    if signal_tc.ndim != 2:
        return None
    if not np.isfinite(signal_tc).all():
        signal_tc = np.nan_to_num(signal_tc, nan=0.0, posinf=0.0, neginf=0.0)

    if source_leads is not None:
        reordered = reorder_leads_tc(signal_tc, source_leads)
        if reordered is None:
            return None
        signal_tc = reordered

    if signal_tc.shape[1] != 12:
        return None

    if apply_filter:
        signal_ct = signal_tc.T
        try:
            signal_ct = filter_bandpass_safe(signal_ct, fs)
        except Exception:
            return None
        signal_tc = signal_ct.T

    signal_tc = resample_tc(signal_tc, fs, target_fs)
    signal_tc = pad_or_truncate_tc(signal_tc, target_len)

    if apply_zscore:
        signal_tc = per_sample_zscore(signal_tc)

    if not np.isfinite(signal_tc).all():
        return None

    return signal_tc.astype(np.float32)


def crop_signal_tc(signal_tc: np.ndarray, crop_len: int, mode: str = "center") -> np.ndarray:
    """Crop ``(time, channels)`` to ``(crop_len, channels)``."""

    time_len = signal_tc.shape[0]
    if time_len < crop_len:
        pad = np.zeros((crop_len - time_len, signal_tc.shape[1]), dtype=signal_tc.dtype)
        return np.concatenate([signal_tc, pad], axis=0)
    max_start = time_len - crop_len
    if mode == "random" and max_start > 0:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = max_start // 2
    return signal_tc[start : start + crop_len, :]


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
