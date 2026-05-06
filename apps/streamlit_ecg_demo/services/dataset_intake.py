from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from apps.streamlit_ecg_demo.services.preprocessing import (
    CLASS_NAMES,
    DEFAULT_SAMPLE_RATE,
    crop_or_pad_ct,
    to_signal_ct,
)


def _read_signal_upload(upload: BinaryIO, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    if name.endswith(".npy"):
        signals = np.load(upload)
        return signals, None
    if name.endswith(".npz"):
        data = np.load(upload, allow_pickle=True)
        signal_key = "signals" if "signals" in data.files else data.files[0]
        label_key = "labels" if "labels" in data.files else None
        labels = data[label_key] if label_key else None
        return data[signal_key], labels
    raise ValueError("ECG upload must be .npy or .npz")


def _read_label_upload(upload: BinaryIO | None) -> np.ndarray | None:
    if upload is None:
        return None
    df = pd.read_csv(upload)
    missing = [c for c in CLASS_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"label CSV is missing columns: {missing}")
    return df[CLASS_NAMES].to_numpy(dtype=np.float32)


def canonicalize_signals(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"signals must be 2D or 3D, got shape={arr.shape}")
    out = []
    for sample in arr:
        out.append(crop_or_pad_ct(to_signal_ct(sample), target_len=1000))
    return np.stack(out, axis=0).astype(np.float32)


def audit_dataset(signals_ct: np.ndarray, labels: np.ndarray) -> dict:
    if labels.ndim != 2 or labels.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"labels must have shape (N, {len(CLASS_NAMES)}), got {labels.shape}")
    if signals_ct.shape[0] != labels.shape[0]:
        raise ValueError(f"signals N={signals_ct.shape[0]} but labels N={labels.shape[0]}")
    finite = np.isfinite(signals_ct).all(axis=(1, 2))
    flatline = np.std(signals_ct, axis=2).max(axis=1) < 1e-6
    class_counts = labels.sum(axis=0).astype(int).tolist()
    primary = []
    priority = ["MI", "HYP", "CD", "STTC", "NORM"]
    for row in labels:
        chosen = "NONE"
        for cls in priority:
            if row[CLASS_NAMES.index(cls)] > 0.5:
                chosen = cls
                break
        primary.append(chosen)
    primary_counts = {k: int(primary.count(k)) for k in ["CD", "HYP", "MI", "NORM", "STTC", "NONE"]}
    return {
        "n_samples": int(signals_ct.shape[0]),
        "signal_shape": list(signals_ct.shape),
        "lead_count": int(signals_ct.shape[1]),
        "sample_rate_hz": float(DEFAULT_SAMPLE_RATE),
        "class_names": list(CLASS_NAMES),
        "class_counts": {c: int(v) for c, v in zip(CLASS_NAMES, class_counts)},
        "primary_counts": primary_counts,
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "flatline_count": int(flatline.sum()),
        "recommended_minimum_note": "Center-token training is more stable with tens to hundreds of labeled target-center ECGs.",
    }


def save_uploaded_dataset(
    project_root: Path,
    signal_upload: BinaryIO,
    signal_name: str,
    label_upload: BinaryIO | None,
    label_name: str | None,
) -> tuple[Path, Path, dict]:
    raw_signals, labels_from_npz = _read_signal_upload(signal_upload, signal_name)
    labels_from_csv = _read_label_upload(label_upload)
    labels = labels_from_csv if labels_from_csv is not None else labels_from_npz
    if labels is None:
        raise ValueError("labels are required: provide labels in NPZ or upload a CSV with CD/HYP/MI/NORM/STTC columns")
    signals_ct = canonicalize_signals(raw_signals)
    labels = np.asarray(labels, dtype=np.float32)
    audit = audit_dataset(signals_ct, labels)

    dataset_path = project_root / "preprocessed" / "hospital_dataset.npz"
    audit_path = project_root / "reports" / "dataset_audit.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        signals=signals_ct,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        source_signal_file=str(signal_name),
        source_label_file=str(label_name or ""),
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return dataset_path, audit_path, audit


def load_audit(audit_path: Path) -> dict | None:
    if not audit_path.exists():
        return None
    return json.loads(audit_path.read_text(encoding="utf-8"))
