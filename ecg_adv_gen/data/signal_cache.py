"""Signal cache metadata helpers shared by ECGFounder full-FT scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SIGNAL_CACHE_METADATA_KEYS = (
    "labels",
    "centers",
    "record_ids",
    "folds",
    "preprocess_policy",
    "input_shape",
    "lead_order",
)


def signal_cache_shape(n_items: int, input_shape: Sequence[int]) -> tuple[int, ...]:
    """Return the memmap shape for a signal cache."""

    return (int(n_items), *tuple(int(v) for v in input_shape))


def build_signal_cache_metadata(
    items: Sequence[Mapping[str, Any]],
    *,
    preprocess_policy: str,
    input_shape: Sequence[int],
    lead_order: Sequence[str],
    num_classes: int,
) -> dict[str, np.ndarray]:
    """Build sidecar metadata for a signal memmap cache.

    Waveform loading stays in the caller so scripts can keep their existing
    preprocessing policy, while metadata schema and dtype handling are shared.
    """

    labels = np.empty((len(items), int(num_classes)), dtype=np.float32)
    centers: list[str] = []
    record_ids: list[str] = []
    folds: list[int] = []

    for i, item in enumerate(items):
        label = np.asarray(item["label"], dtype=np.float32)
        expected_shape = (int(num_classes),)
        if label.shape != expected_shape:
            raise ValueError(f"item {i} label shape {label.shape} != {expected_shape}")
        labels[i] = label
        centers.append(str(item.get("center", "")))
        record_ids.append(str(item.get("record_id", "")))
        folds.append(int(item.get("strat_fold", -1)))

    return {
        "labels": labels,
        "centers": np.asarray(centers, dtype=str),
        "record_ids": np.asarray(record_ids, dtype=str),
        "folds": np.asarray(folds, dtype=np.int64),
        "preprocess_policy": np.asarray([preprocess_policy], dtype=str),
        "input_shape": np.asarray(tuple(int(v) for v in input_shape), dtype=np.int64),
        "lead_order": np.asarray(tuple(str(v) for v in lead_order), dtype=str),
    }


def write_signal_cache_metadata(meta_path: Path, metadata: Mapping[str, np.ndarray]) -> None:
    """Write signal cache sidecar metadata using the project schema."""

    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: metadata[key] for key in SIGNAL_CACHE_METADATA_KEYS}
    np.savez_compressed(meta_path, **payload)


def load_signal_cache(
    signal_path: Path,
    meta_path: Path,
    *,
    mmap_mode: str | None = "r",
) -> dict[str, np.ndarray]:
    """Load a signal memmap and its metadata sidecar."""

    with np.load(meta_path, allow_pickle=True) as meta:
        metadata = {key: meta[key] for key in meta.files}
    return {
        "signals": np.load(signal_path, mmap_mode=mmap_mode),
        **metadata,
    }
