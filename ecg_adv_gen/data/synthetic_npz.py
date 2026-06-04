"""Load and validate synthetic classifier NPZ artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SyntheticNPZArrays:
    """Loaded synthetic ECG arrays in classifier format."""

    signals: np.ndarray
    labels: np.ndarray
    source_count: int
    paths: tuple[Path, ...]


def _normalize_npz_paths(npz_paths: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(npz_paths, str):
        return tuple(Path(path) for path in npz_paths.split(",") if path)
    if isinstance(npz_paths, Path):
        return (npz_paths,)
    return tuple(Path(path) for path in npz_paths)


def normalize_synthetic_signals(signals: np.ndarray) -> np.ndarray:
    """Return synthetic signals as ``(N, 1000, 12)`` float32 arrays."""
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected synth signals ndim=3, got {arr.shape}")
    if arr.shape[1:] == (12, 1000):
        arr = arr.transpose(0, 2, 1)
    if arr.shape[1:] != (1000, 12):
        raise ValueError(
            "Expected synth signals as (N,1000,12) or (N,12,1000), "
            f"got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def _flat_label_key(data: np.lib.npyio.NpzFile, path: Path) -> str:
    if "labels" in data.files:
        return "labels"
    if "labels5" in data.files:
        return "labels5"
    raise ValueError(f"{path} missing labels or labels5 for flat synthetic signals")


def _append_arrays(
    *,
    signals_all: list[np.ndarray],
    labels_all: list[np.ndarray],
    signals: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    normalized = normalize_synthetic_signals(signals)
    label_arr = np.asarray(labels, dtype=np.float32)
    if normalized.shape[0] != label_arr.shape[0]:
        raise ValueError(
            f"synth signals/labels length mismatch in {path}: "
            f"{normalized.shape} vs {label_arr.shape}"
        )
    signals_all.append(normalized)
    labels_all.append(label_arr)


def load_synthetic_npz_arrays(
    npz_paths: str | Path | Sequence[str | Path],
) -> SyntheticNPZArrays:
    """Load synthetic classifier arrays from flat or per-center NPZ files.

    Supports flat keys ``signals`` plus ``labels`` or ``labels5``, and
    per-center cached keys ``<center>__signals`` plus ``<center>__labels5``.
    """
    paths = _normalize_npz_paths(npz_paths)
    signals_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for path in paths:
        with np.load(str(path)) as data:
            if "signals" in data.files:
                label_key = _flat_label_key(data, path)
                _append_arrays(
                    signals_all=signals_all,
                    labels_all=labels_all,
                    signals=data["signals"],
                    labels=data[label_key],
                    path=path,
                )
                continue

            signal_keys = sorted(key for key in data.files if key.endswith("__signals"))
            for sig_key in signal_keys:
                prefix = sig_key[: -len("__signals")]
                label_key = f"{prefix}__labels5"
                if label_key not in data.files:
                    continue
                _append_arrays(
                    signals_all=signals_all,
                    labels_all=labels_all,
                    signals=data[sig_key],
                    labels=data[label_key],
                    path=path,
                )

    if not signals_all:
        raise ValueError(f"No synthetic signals found in {list(paths)}")

    return SyntheticNPZArrays(
        signals=np.concatenate(signals_all, axis=0).astype(np.float32, copy=False),
        labels=np.concatenate(labels_all, axis=0).astype(np.float32, copy=False),
        source_count=len(signals_all),
        paths=paths,
    )
