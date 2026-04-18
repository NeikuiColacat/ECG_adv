"""
Unified ECG preprocessing pipeline for PTBXL (training) and PN2021/MIMIC (OOD eval).

Applied identically to all datasets so any remaining cross-center gap reflects
genuine domain shift rather than preprocessing artifacts.

Pipeline (per record):
  (1) handle NaN (nan_to_num)
  (2) lead reorder to [I, II, III, aVR, aVL, aVF, V1..V6]   if source_leads given
  (3) filter_bandpass at NATIVE fs (skip 50Hz notch when fs <= 150 because Nyquist is unstable)
  (4) resample to target_fs (100 Hz)
  (5) pad/truncate to target_len (1000 samples = 10s)
  (6) per-sample GLOBAL z-score  (mean/std over 12 x time flattened)
  (7) crop to crop_len (250 samples = 2.5s)  — random for train, center for eval

Output tensor layout: (12, crop_len) float32
"""

import numpy as np
from scipy.signal import resample, medfilt, iirnotch, filtfilt, butter


EXPECTED_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
EXPECTED_LEADS_UPPER = [l.upper() for l in EXPECTED_LEADS]


def filter_bandpass_safe(signal, fs, skip_notch_threshold=150):
    """Bandpass + baseline removal, notch only when fs is high enough.

    Args:
        signal: (channels, time) float
        fs: native sampling rate in Hz
        skip_notch_threshold: skip the 50 Hz notch when fs <= threshold
            (at fs=100, Nyquist=50 and iirnotch is ill-conditioned).

    Returns: (channels, time) float32
    """
    filtered = signal.astype(np.float64, copy=True)

    if fs > skip_notch_threshold:
        b, a = iirnotch(50.0, 30.0, fs)
        for c in range(filtered.shape[0]):
            filtered[c] = filtfilt(b, a, filtered[c])

    # Bandpass 0.67 - 40 Hz (Butterworth N=4, zero-phase)
    # Clamp high-cut to < Nyquist just in case (40Hz<50Hz at fs=100 is fine).
    high_cut = min(40.0, 0.95 * fs / 2)
    low_cut = 0.67
    b, a = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=fs)
    for c in range(filtered.shape[0]):
        filtered[c] = filtfilt(b, a, filtered[c])

    # Baseline wander removal via median filter (kernel ~ 0.4s)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Guard against kernel_size > signal length
    if kernel_size >= filtered.shape[1]:
        kernel_size = filtered.shape[1] - 1
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            # Too short to baseline-correct, skip
            return filtered.astype(np.float32)

    baseline = np.zeros_like(filtered)
    for c in range(filtered.shape[0]):
        baseline[c] = medfilt(filtered[c], kernel_size=kernel_size)

    return (filtered - baseline).astype(np.float32)


def reorder_leads_tc(signal_tc, source_leads):
    """Reorder (time, channels) signal so channels match EXPECTED_LEADS.

    Case-insensitive match. Returns None if any expected lead is missing.
    """
    source_upper = [s.strip().upper() for s in source_leads]
    result = np.zeros((signal_tc.shape[0], 12), dtype=signal_tc.dtype)
    for i, lead in enumerate(EXPECTED_LEADS_UPPER):
        if lead in source_upper:
            result[:, i] = signal_tc[:, source_upper.index(lead)]
        else:
            return None
    return result


def per_sample_zscore(signal, eps=1e-8):
    """Per-sample global z-score (flatten 12 x time, single mean/std per record)."""
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return ((signal - mean) / (std + eps)).astype(np.float32)


def resample_tc(signal_tc, source_fs, target_fs):
    """Resample (time, channels) along time."""
    if int(source_fs) == int(target_fs):
        return signal_tc.astype(np.float32)
    n_target = int(round(signal_tc.shape[0] * target_fs / source_fs))
    return resample(signal_tc, n_target, axis=0).astype(np.float32)


def pad_or_truncate_tc(signal_tc, target_len):
    """Pad/truncate (time, channels) to target_len along time."""
    n = signal_tc.shape[0]
    if n < target_len:
        pad = np.zeros((target_len - n, signal_tc.shape[1]), dtype=signal_tc.dtype)
        return np.concatenate([signal_tc, pad], axis=0)
    if n > target_len:
        return signal_tc[:target_len]
    return signal_tc


def unified_preprocess_to_1000(
    signal_tc,
    fs,
    source_leads=None,
    target_fs=100,
    target_len=1000,
    apply_filter=True,
    apply_zscore=True,
):
    """Full pipeline up to (target_len, 12). Returns None on failure.

    Args:
        signal_tc: (time, channels) float array
        fs: native sampling rate
        source_leads: list of lead names for reorder. None => already canonical.
        target_fs: unified rate (default 100)
        target_len: unified length (default 1000 = 10s)
        apply_filter: toggle bandpass+baseline (ablation)
        apply_zscore: toggle per-sample z-score (ablation)

    Returns: (target_len, 12) float32 or None
    """
    signal_tc = np.asarray(signal_tc, dtype=np.float32)
    if signal_tc.ndim != 2:
        return None
    if np.isnan(signal_tc).any():
        signal_tc = np.nan_to_num(signal_tc, nan=0.0)

    # Reorder leads if source order provided
    if source_leads is not None:
        reordered = reorder_leads_tc(signal_tc, source_leads)
        if reordered is None:
            return None
        signal_tc = reordered

    if signal_tc.shape[1] != 12:
        return None

    # Filter at native fs (operates on (channels, time))
    if apply_filter:
        signal_ct = signal_tc.T  # (12, time)
        try:
            signal_ct = filter_bandpass_safe(signal_ct, fs)
        except Exception:
            return None
        signal_tc = signal_ct.T  # back to (time, 12)

    # Resample to target_fs
    signal_tc = resample_tc(signal_tc, fs, target_fs)

    # Pad/truncate
    signal_tc = pad_or_truncate_tc(signal_tc, target_len)

    # Per-sample z-score
    if apply_zscore:
        signal_tc = per_sample_zscore(signal_tc)

    # Final NaN guard (filtfilt sometimes produces NaN on pathological inputs)
    if not np.isfinite(signal_tc).all():
        return None

    return signal_tc.astype(np.float32)


def crop_signal_tc(signal_tc, crop_len, mode='center'):
    """Crop (time, channels) to (crop_len, channels).

    mode: 'center' or 'random'
    """
    time_len = signal_tc.shape[0]
    if time_len < crop_len:
        # pad instead of fail
        pad = np.zeros((crop_len - time_len, signal_tc.shape[1]), dtype=signal_tc.dtype)
        return np.concatenate([signal_tc, pad], axis=0)
    max_start = time_len - crop_len
    if mode == 'random' and max_start > 0:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = max_start // 2
    return signal_tc[start:start + crop_len, :]
