from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_SAMPLE_RATE = 100.0


def to_signal_ct(signal: np.ndarray) -> np.ndarray:
    """Return ECG as (12, 1000-ish) float32 channels-first."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"batched ECG must have batch=1 for demo, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"ECG must be 2D, got shape={arr.shape}")
    if arr.shape[0] == 12:
        out = arr
    elif arr.shape[1] == 12:
        out = arr.T
    else:
        raise ValueError(f"cannot infer lead axis from shape={arr.shape}")
    return np.ascontiguousarray(out, dtype=np.float32)


def crop_or_pad_ct(signal_ct: np.ndarray, target_len: int = 1000) -> np.ndarray:
    arr = to_signal_ct(signal_ct)
    length = arr.shape[1]
    if length == target_len:
        return arr
    if length > target_len:
        start = (length - target_len) // 2
        return np.ascontiguousarray(arr[:, start:start + target_len], dtype=np.float32)
    pad = np.zeros((12, target_len - length), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


def classifier_input(signal: np.ndarray, target_len: int = 1000) -> np.ndarray:
    """Return (1, 12, target_len) classifier input."""
    arr = crop_or_pad_ct(signal, target_len=target_len)
    return arr[None].astype(np.float32, copy=False)


def global_zscore_ct(signal_ct: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = to_signal_ct(signal_ct)
    mean = float(arr.mean())
    std = float(arr.std())
    return ((arr - mean) / (std + eps)).astype(np.float32)


def load_npz_samples(path: str | Path, max_items: int = 64) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    if "signals" not in data:
        raise ValueError(f"{path} does not contain key 'signals'")
    signals = data["signals"]
    labels = data["labels"] if "labels" in data else None
    class_names = [str(x) for x in data["class_names"]] if "class_names" in data else CLASS_NAMES
    center_name = str(data["center_name"]) if "center_name" in data else Path(path).parent.name
    out = []
    for i in range(min(int(max_items), len(signals))):
        label = labels[i].astype(np.float32).tolist() if labels is not None else None
        label_names = []
        if label is not None:
            label_names = [class_names[j] for j, v in enumerate(label) if float(v) > 0.5]
        out.append({
            "id": f"{Path(path).stem}:{i}",
            "source_path": str(path),
            "center": center_name,
            "signal": to_signal_ct(signals[i]),
            "label": label,
            "label_names": label_names,
        })
    return out


def load_demo_samples(paths: Iterable[str | Path], max_items_per_file: int = 32) -> list[dict]:
    samples: list[dict] = []
    for path in paths:
        p = Path(path)
        if p.exists():
            samples.extend(load_npz_samples(p, max_items=max_items_per_file))
    return samples


def apply_corruption(signal_ct: np.ndarray, name: str, severity: int, seed: int = 42) -> np.ndarray:
    """Lightweight PN2021-C-style corruptions for the Streamlit demo."""
    rng = np.random.default_rng(seed)
    arr = to_signal_ct(signal_ct).copy()
    sev = max(1, min(int(severity), 5))
    t = np.linspace(0.0, arr.shape[1] / DEFAULT_SAMPLE_RATE, arr.shape[1], endpoint=False)

    if name == "powerline_noise":
        amp = 0.02 * sev
        arr += amp * np.sin(2 * np.pi * 50.0 * t)[None, :]
    elif name == "emg_noise":
        arr += rng.normal(0.0, 0.03 * sev, size=arr.shape).astype(np.float32)
    elif name == "baseline_wander":
        arr += (0.05 * sev * np.sin(2 * np.pi * 0.33 * t))[None, :]
    elif name == "baseline_shift":
        arr += np.float32(0.05 * sev)
    elif name == "random_leads_masking":
        n_mask = min(12, sev)
        idx = rng.choice(np.arange(12), size=n_mask, replace=False)
        arr[idx] = 0.0
    elif name in {"none", ""}:
        pass
    else:
        raise ValueError(f"unknown corruption={name!r}")
    return arr.astype(np.float32)

