"""Small ECG waveform preprocessing helpers.

These functions cover shape/axis normalization, ECGTwin -> PTB-XL lead order,
and deterministic linear resampling. Dataset-specific IO remains in legacy
scripts until the data loaders are extracted.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ecg_adv_gen.data import (
    CLASSIFIER_LEN,
    ECGTWIN_DECODE_OUTPUT_LEN,
    ECGTWIN_TO_PTBXL_INDICES,
)


class PreprocessingError(ValueError):
    """Raised when waveform axes or shapes are ambiguous."""


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise PreprocessingError(f"axis={axis} is invalid for ndim={ndim}")
    return axis


def infer_lead_axis(signal: np.ndarray, *, n_leads: int = 12) -> int:
    arr = np.asarray(signal)
    matches = [idx for idx, size in enumerate(arr.shape) if size == n_leads]
    if len(matches) != 1:
        raise PreprocessingError(
            f"Expected exactly one lead axis of size {n_leads}, got shape={arr.shape}"
        )
    return matches[0]


def infer_time_axis(signal: np.ndarray, *, lead_axis: int | None = None) -> int:
    arr = np.asarray(signal)
    if arr.ndim < 2:
        raise PreprocessingError(f"ECG signal must have at least 2 dimensions, got {arr.shape}")
    lead_axis = infer_lead_axis(arr) if lead_axis is None else _normalize_axis(lead_axis, arr.ndim)
    candidates = [axis for axis in range(arr.ndim) if axis != lead_axis]
    if not candidates:
        raise PreprocessingError(f"Could not infer time axis for shape={arr.shape}")
    return candidates[-1]


def reorder_leads(
    signal: np.ndarray,
    indices: Iterable[int] = ECGTWIN_TO_PTBXL_INDICES,
    *,
    lead_axis: int | None = None,
) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim < 2:
        raise PreprocessingError(f"ECG signal must have at least 2 dimensions, got {arr.shape}")
    lead_axis = infer_lead_axis(arr) if lead_axis is None else _normalize_axis(lead_axis, arr.ndim)
    return np.take(arr, list(indices), axis=lead_axis)


def ecgtwin_to_ptbxl(signal: np.ndarray, *, lead_axis: int | None = None) -> np.ndarray:
    return reorder_leads(signal, ECGTWIN_TO_PTBXL_INDICES, lead_axis=lead_axis)


def linear_resample(
    signal: np.ndarray,
    target_length: int = CLASSIFIER_LEN,
    *,
    time_axis: int | None = None,
) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim < 1:
        raise PreprocessingError("Cannot resample a scalar")
    time_axis = arr.ndim - 1 if time_axis is None else _normalize_axis(time_axis, arr.ndim)
    source_length = arr.shape[time_axis]
    if source_length <= 0:
        raise PreprocessingError(f"Cannot resample empty time axis for shape={arr.shape}")
    if source_length == target_length:
        return np.array(arr, copy=True)

    moved = np.moveaxis(arr, time_axis, -1)
    flat = moved.reshape(-1, source_length)
    old_x = np.linspace(0.0, 1.0, source_length, dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, int(target_length), dtype=np.float64)
    out = np.empty((flat.shape[0], int(target_length)), dtype=np.float64)
    for idx, row in enumerate(flat):
        out[idx] = np.interp(new_x, old_x, row)
    out = out.reshape(*moved.shape[:-1], int(target_length))
    out = np.moveaxis(out, -1, time_axis)
    if np.issubdtype(arr.dtype, np.floating):
        out = out.astype(arr.dtype, copy=False)
    return out


def move_to_channel_last(
    signal: np.ndarray,
    *,
    lead_axis: int | None = None,
    time_axis: int | None = None,
) -> np.ndarray:
    arr = np.asarray(signal)
    lead_axis = infer_lead_axis(arr) if lead_axis is None else _normalize_axis(lead_axis, arr.ndim)
    time_axis = infer_time_axis(arr, lead_axis=lead_axis) if time_axis is None else _normalize_axis(time_axis, arr.ndim)
    if lead_axis == time_axis:
        raise PreprocessingError(f"lead_axis and time_axis must differ for shape={arr.shape}")
    return np.moveaxis(arr, [time_axis, lead_axis], [-2, -1])


def prepare_ecgtwin_decoded_for_classifier(
    signal: np.ndarray,
    *,
    target_length: int = ECGTWIN_DECODE_OUTPUT_LEN,
    lead_axis: int | None = None,
    time_axis: int | None = None,
) -> np.ndarray:
    """Return PTB-XL-order ECG in channel-last classifier format.

    Accepts ECGTwin decoded arrays such as ``(B, 1024, 12)`` or
    ``(B, 12, 1024)`` and returns ``(B, 1000, 12)`` by default.
    """
    arr = np.asarray(signal)
    lead_axis = infer_lead_axis(arr) if lead_axis is None else _normalize_axis(lead_axis, arr.ndim)
    time_axis = infer_time_axis(arr, lead_axis=lead_axis) if time_axis is None else _normalize_axis(time_axis, arr.ndim)
    reordered = ecgtwin_to_ptbxl(arr, lead_axis=lead_axis)
    resampled = linear_resample(reordered, target_length=target_length, time_axis=time_axis)
    return move_to_channel_last(resampled, lead_axis=lead_axis, time_axis=time_axis)
