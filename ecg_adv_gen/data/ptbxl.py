"""PTB-XL data helpers for Super5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ecg_adv_gen.labels.super5_mapping import CLASS_NAMES_SUPER5

SUPER5_ORDER = CLASS_NAMES_SUPER5


@dataclass(frozen=True)
class FoldSplit:
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]


def _ids_for_folds(frame: Any, folds: set[int]) -> list[int]:
    rows = frame[frame["strat_fold"].isin(folds)]
    return [int(value) for value in rows["ecg_id"].tolist()]


def fold_split(
    frame: Any,
    train_folds: set[int],
    val_folds: set[int],
    test_folds: set[int],
) -> FoldSplit:
    if "ecg_id" not in frame or "strat_fold" not in frame:
        raise ValueError("PTB-XL frame must contain ecg_id and strat_fold")
    overlap = (train_folds & val_folds) | (train_folds & test_folds) | (val_folds & test_folds)
    if overlap:
        raise ValueError(f"PTB-XL fold sets overlap: {sorted(overlap)}")
    return FoldSplit(
        train_ids=_ids_for_folds(frame, train_folds),
        val_ids=_ids_for_folds(frame, val_folds),
        test_ids=_ids_for_folds(frame, test_folds),
    )


def normalize_super5_labels(
    rows: Sequence[Mapping[str, int | float | bool]],
) -> np.ndarray:
    labels = []
    for idx, row in enumerate(rows):
        missing = [name for name in SUPER5_ORDER if name not in row]
        if missing:
            raise ValueError(f"row {idx} missing Super5 labels: {missing}")
        labels.append([float(row[name]) for name in SUPER5_ORDER])
    return np.asarray(labels, dtype=np.float32)


def get_ptbxl_labels_for_scheme(csv_path: str | Path, scheme: Mapping[str, Any], label_cache_path: str | Path, folds=None):
    """Load PTB-XL metadata and scheme-specific labels."""
    import pandas as pd

    df_full = pd.read_csv(csv_path)
    if folds is not None:
        mask = df_full.strat_fold.isin(folds)
        indices = np.where(mask)[0].tolist()
        df = df_full[mask].reset_index(drop=True)
    else:
        indices = list(range(len(df_full)))
        df = df_full.copy()

    cache_key = f"{label_cache_path}.C{scheme['num_classes']}.all.npy"
    if Path(cache_key).exists():
        all_labels = np.load(cache_key)
        assert all_labels.shape[1] == scheme["num_classes"], (
            f"label cache {cache_key} has shape {all_labels.shape}, "
            f"expected C={scheme['num_classes']}"
        )
    else:
        df_for_labels = df_full if folds is not None else df
        all_labels = np.stack([scheme["ptbxl_fn"](row) for row in df_for_labels.scp_codes])
        Path(cache_key).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_key, all_labels)
    labels = all_labels[indices] if folds is not None else all_labels
    return indices, labels.astype(np.float32), df


def preprocess_ptbxl_all(
    raw_path: str | Path,
    cache_path: str | Path,
    target_fs: int = 100,
    target_len: int = 1000,
    preprocess_mode: str = "legacy_ecgfounder_filter",
    norm_mode: str = "per_sample_global",
):
    """Preprocess the full PTB-XL raw array into cached classifier layout."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[preprocess] cache hit: {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    from ecg_adv_gen.preprocessing import unified_preprocess_to_1000

    print(
        f"[preprocess] preprocessing all PTBXL -> {cache_path} (first run) "
        f"mode={preprocess_mode} norm={norm_mode}"
    )
    raw = np.load(raw_path, allow_pickle=True).astype(np.float32)
    out = np.zeros((raw.shape[0], target_len, 12), dtype=np.float32)
    fails = []
    for idx in range(raw.shape[0]):
        proc = unified_preprocess_to_1000(
            raw[idx],
            fs=100,
            source_leads=None,
            target_fs=target_fs,
            target_len=target_len,
            preprocess_mode=preprocess_mode,
            norm_mode=norm_mode,
        )
        if proc is None:
            fails.append(idx)
            sig = np.nan_to_num(raw[idx], nan=0.0, posinf=0.0, neginf=0.0)
            if norm_mode == "per_sample_global":
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            out[idx] = sig.astype(np.float32)
        else:
            out[idx] = proc
        if (idx + 1) % 5000 == 0:
            print(f"  ... {idx + 1}/{raw.shape[0]}")
    if fails:
        print(f"[preprocess] WARN: {len(fails)} records fell back (filter failed)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    print(f"[preprocess] saved cache: {cache_path}")
    return out
