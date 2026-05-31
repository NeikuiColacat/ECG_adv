#!/usr/bin/env python
"""Build PTB-XL records500 caches for the repo-owned 500 Hz VAE.

This script intentionally uses only PTB-XL `filename_hr` records. It creates
fold-isolated train/val/audit arrays for VAE training and reconstruction audit:

* train: strat_fold 1-8
* val: strat_fold 9
* audit: strat_fold 10

Signals are saved as channels-last `(N, 5000, 12)` arrays in PTB-XL canonical
lead order. Raw physical mV and per-sample-global z-score views are both saved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.labels.super5 import get_super5_metadata
from scripts.triple_labels.label_schemes import get_scheme


CANONICAL_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
SPLIT_FOLDS = {
    "train": tuple(range(1, 9)),
    "val": (9,),
    "audit": (10,),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ensure_length(signal: np.ndarray, target_len: int) -> tuple[np.ndarray, str]:
    if signal.shape[0] == target_len:
        return signal, "exact"
    if signal.shape[0] > target_len:
        return signal[:target_len], "truncated_right"
    out = np.zeros((target_len, signal.shape[1]), dtype=signal.dtype)
    out[: signal.shape[0]] = signal
    return out, "padded_right"


def _reorder_to_canonical(signal: np.ndarray, sig_names: list[str] | None) -> tuple[np.ndarray, bool]:
    if sig_names is None:
        return signal, False
    upper = [str(name).upper() for name in sig_names]
    if not all(lead in upper for lead in CANONICAL_LEADS):
        return signal, False
    indices = [upper.index(lead) for lead in CANONICAL_LEADS]
    return signal[:, indices], True


def _per_sample_global(signal: np.ndarray) -> np.ndarray:
    mean = float(signal.mean())
    std = float(signal.std())
    return ((signal - mean) / (std + 1e-8)).astype(np.float32, copy=False)


def _read_record(record_base: Path, target_len: int, expected_fs: int) -> tuple[np.ndarray, dict[str, Any]]:
    record = wfdb.rdrecord(str(record_base))
    signal = np.asarray(record.p_signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"expected 2D p_signal for {record_base}, got {signal.shape}")
    if signal.shape[1] != 12 and signal.shape[0] == 12:
        signal = signal.T
    if signal.shape[1] != 12:
        raise ValueError(f"expected 12 leads for {record_base}, got {signal.shape}")
    signal, reordered = _reorder_to_canonical(signal, getattr(record, "sig_name", None))
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    signal, length_action = _ensure_length(signal, target_len)
    meta = {
        "record_base": str(record_base),
        "source_fs": int(round(float(getattr(record, "fs", -1)))),
        "expected_fs_match": int(round(float(getattr(record, "fs", -1)))) == int(expected_fs),
        "source_shape": list(np.asarray(record.p_signal).shape),
        "length_action": length_action,
        "lead_reordered": reordered,
        "sig_name": list(getattr(record, "sig_name", []) or []),
    }
    return signal, meta


def _split_frame(df: pd.DataFrame, folds: tuple[int, ...], limit_per_fold: int | None) -> pd.DataFrame:
    split = df[df["strat_fold"].astype(int).isin(folds)].copy()
    if limit_per_fold is not None and limit_per_fold > 0:
        split = (
            split.sort_values(["strat_fold", "ecg_id"])
            .groupby("strat_fold", group_keys=False)
            .head(int(limit_per_fold))
            .copy()
        )
    return split.reset_index(drop=True)


def _write_split(
    *,
    split_name: str,
    df_split: pd.DataFrame,
    ptbxl_root: Path,
    output_dir: Path,
    target_len: int,
    expected_fs: int,
    labels: np.ndarray,
    save_raw: bool,
) -> dict[str, Any]:
    n = int(len(df_split))
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    norm_path = split_dir / "signals_norm.npy"
    raw_path = split_dir / "signals_raw_mV.npy"
    labels_path = split_dir / "labels.npy"
    meta_path = split_dir / "records.jsonl"

    norm = np.lib.format.open_memmap(norm_path, mode="w+", dtype=np.float32, shape=(n, target_len, 12))
    raw = None
    if save_raw:
        raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(n, target_len, 12))

    record_meta: list[dict[str, Any]] = []
    n_bad_fs = 0
    n_length_adjusted = 0
    n_reordered = 0
    for i, row in enumerate(df_split.itertuples(index=False)):
        record_rel = getattr(row, "filename_hr")
        record_base = ptbxl_root / str(record_rel)
        signal_raw, meta = _read_record(record_base, target_len=target_len, expected_fs=expected_fs)
        norm[i] = _per_sample_global(signal_raw)
        if raw is not None:
            raw[i] = signal_raw
        if not meta["expected_fs_match"]:
            n_bad_fs += 1
        if meta["length_action"] != "exact":
            n_length_adjusted += 1
        if meta["lead_reordered"]:
            n_reordered += 1
        item = {
            "row_index": int(i),
            "ecg_id": int(getattr(row, "ecg_id")),
            "patient_id": int(getattr(row, "patient_id")),
            "strat_fold": int(getattr(row, "strat_fold")),
            "filename_hr": str(record_rel),
            "label": labels[i].astype(float).tolist(),
            **meta,
        }
        record_meta.append(item)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"[{split_name}] {i + 1}/{n}", flush=True)

    norm.flush()
    if raw is not None:
        raw.flush()
    np.save(labels_path, labels.astype(np.float32, copy=False))
    with meta_path.open("w", encoding="utf-8") as f:
        for item in record_meta:
            f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")

    return {
        "split": split_name,
        "folds": list(SPLIT_FOLDS[split_name]),
        "n_records": n,
        "signals_norm": str(norm_path),
        "signals_raw_mV": str(raw_path) if save_raw else None,
        "labels": str(labels_path),
        "records_jsonl": str(meta_path),
        "shape": [n, target_len, 12],
        "bad_source_fs_count": n_bad_fs,
        "length_adjusted_count": n_length_adjusted,
        "lead_reordered_count": n_reordered,
        "positive_counts": labels.sum(axis=0).astype(int).tolist(),
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    ptbxl_root = Path(args.ptbxl_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve() if args.csv_path else ptbxl_root / "ptbxl_database.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme = get_scheme(args.scheme)
    super5_meta = get_super5_metadata() if args.scheme == "super5" else None
    df = pd.read_csv(csv_path)
    required_cols = {"ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    split_summaries = []
    for split_name, folds in SPLIT_FOLDS.items():
        df_split = _split_frame(df, folds, args.limit_per_fold)
        labels = np.stack([scheme["ptbxl_fn"](scp) for scp in df_split["scp_codes"]]).astype(np.float32)
        split_summaries.append(
            _write_split(
                split_name=split_name,
                df_split=df_split,
                ptbxl_root=ptbxl_root,
                output_dir=output_dir,
                target_len=args.length,
                expected_fs=args.sampling_rate,
                labels=labels,
                save_raw=not args.no_save_raw,
            )
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "PTB-XL records500 cache for repo-owned 500Hz VAE",
        "ptbxl_root": str(ptbxl_root),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "source_record_column": "filename_hr",
        "source_sampling_rate": args.sampling_rate,
        "target_sampling_rate": args.sampling_rate,
        "target_length": args.length,
        "unit": "physical_mV",
        "lead_order": list(CANONICAL_LEADS),
        "signal_axis_order": "N,T,C",
        "normalization": {
            "signals_norm": "per_sample_global_zscore over full 12xT sample",
            "signals_raw_mV": "NaN/Inf guarded physical mV",
        },
        "scheme": args.scheme,
        "class_names": list(scheme["class_names"]),
        "super5_mapping": (
            {
                "mapping_version": super5_meta.mapping_version,
                "mapping_hash": super5_meta.mapping_hash,
            }
            if super5_meta is not None
            else None
        ),
        "fold_contract": {
            "train": "PTB-XL strat_fold 1-8",
            "val": "PTB-XL strat_fold 9",
            "audit": "PTB-XL strat_fold 10",
        },
        "limit_per_fold": args.limit_per_fold,
        "splits": split_summaries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[done] wrote {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl_root", default="/root/autodl-tmp/ptbxl")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scheme", default="super5", choices=["super5", "sub23", "pn26"])
    parser.add_argument("--sampling_rate", type=int, default=500)
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--limit_per_fold", type=int, default=None)
    parser.add_argument("--no_save_raw", action="store_true", help="Only save normalized signals; manifest still records raw policy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling_rate != 500:
        print(f"[warn] expected records500 workflow, got sampling_rate={args.sampling_rate}", flush=True)
    if args.length != 5000:
        print(f"[warn] expected 10s records500 length 5000, got length={args.length}", flush=True)
    build_cache(args)


if __name__ == "__main__":
    main()
