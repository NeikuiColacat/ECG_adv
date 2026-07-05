#!/usr/bin/env python3
"""Build K500 subsets with official severity-5 composite corruptions.

This creates a supervised stress-training baseline without changing the direct
K500 fine-tune runner: labels and reference metadata are copied from the clean
K500 subset, while ``signals.npz`` is replaced by clean plus pair/triple
official severity-5 corruption views.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.evaluation.pn2021c_protocol import (  # noqa: E402
    OFFICIAL_S5_COMPOSITE_OPS,
    official_s5_composite_names,
)
from methods.augmix.augmix import _apply_op  # noqa: E402


def _load_clean_subset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        record_ids = np.asarray(data["record_ids"]).astype(str)
    if signals.ndim != 3 or signals.shape[1:] != (1000, 12):
        raise ValueError(f"{path} has signals shape {signals.shape}, expected (N,1000,12)")
    return signals, labels, record_ids


def _apply_combo(signal_tl: np.ndarray, combo: tuple[str, ...], severity: int) -> np.ndarray:
    sig_ct = torch.from_numpy(np.ascontiguousarray(signal_tl.T)).float()
    for op_name in combo:
        sig_ct = _apply_op(sig_ct, op_name, severity)
    return sig_ct.cpu().numpy().astype(np.float32, copy=False).T


def build_center_subset(
    *,
    center: str,
    clean_subset_root: Path,
    output_subset_root: Path,
    k: int,
    seed: int,
    severity: int,
    include_clean: bool,
    depths: tuple[int, ...],
) -> Path:
    center_seed = int(seed) + sum((i + 1) * ord(ch) for i, ch in enumerate(center))
    np.random.seed(center_seed)
    random.seed(center_seed)
    torch.manual_seed(center_seed)

    clean_dir = clean_subset_root / center / f"k{k}_seed{seed}"
    clean_base = clean_dir / f"{center}_real_k{k}_seed{seed}"
    clean_signals_path = clean_base.with_suffix(".signals.npz")
    clean_meta_path = clean_base.with_suffix(".ref_meta.json")
    if not clean_signals_path.exists():
        raise FileNotFoundError(clean_signals_path)
    if not clean_meta_path.exists():
        raise FileNotFoundError(clean_meta_path)

    signals, labels, record_ids = _load_clean_subset(clean_signals_path)
    combos = [tuple(name.split("+")) for name in official_s5_composite_names(depths)]

    out_signals: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    out_record_ids: list[np.ndarray] = []
    view_names: list[str] = []

    if include_clean:
        out_signals.append(signals.astype(np.float32, copy=False))
        out_labels.append(labels.astype(np.float32, copy=False))
        out_record_ids.append(record_ids.astype(str))
        view_names.append("clean")

    for combo in combos:
        combo_name = "+".join(combo)
        corrupted = np.empty_like(signals, dtype=np.float32)
        for i in range(signals.shape[0]):
            corrupted[i] = _apply_combo(signals[i], combo, severity)
        out_signals.append(corrupted)
        out_labels.append(labels.astype(np.float32, copy=False))
        out_record_ids.append(np.asarray([f"{rid}|{combo_name}" for rid in record_ids], dtype=str))
        view_names.append(combo_name)

    merged_signals = np.concatenate(out_signals, axis=0).astype(np.float32, copy=False)
    merged_labels = np.concatenate(out_labels, axis=0).astype(np.float32, copy=False)
    merged_record_ids = np.concatenate(out_record_ids, axis=0).astype(str)

    out_dir = output_subset_root / center / f"k{k}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / f"{center}_real_k{k}_seed{seed}"
    out_signals_path = out_base.with_suffix(".signals.npz")
    out_meta_path = out_base.with_suffix(".ref_meta.json")
    np.savez_compressed(
        out_signals_path,
        signals=merged_signals,
        labels=merged_labels,
        record_ids=merged_record_ids,
        metadata=np.asarray(
            [
                {
                    "purpose": "direct K500 supervised fine-tune on official severity-5 composite corrupted views",
                    "center": center,
                    "source_signals": str(clean_signals_path),
                    "source_ref_meta": str(clean_meta_path),
                    "k": int(k),
                    "seed": int(seed),
                    "severity": int(severity),
                    "severity_profile": "standard",
                    "ops": list(OFFICIAL_S5_COMPOSITE_OPS),
                    "depths": list(depths),
                    "combos": view_names,
                    "include_clean": bool(include_clean),
                    "n_source_records": int(signals.shape[0]),
                    "n_views_per_source": int(len(view_names)),
                    "n_total_records": int(merged_signals.shape[0]),
                    "data_flow": [
                        "clean K500 raw1000 signal",
                        "official severity-5 composite corruption on raw1000 signal",
                        "direct fine-tune runner per-sample global z-score",
                        "EfficientNet1DV2",
                    ],
                }
            ],
            dtype=object,
        ),
    )
    shutil.copy2(clean_meta_path, out_meta_path)
    summary_path = out_dir / "composite_subset_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "center": center,
                "signals_npz": str(out_signals_path),
                "ref_meta_json": str(out_meta_path),
                "n_total_records": int(merged_signals.shape[0]),
                "n_source_records": int(signals.shape[0]),
                "n_views_per_source": int(len(view_names)),
                "views": view_names,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(summary_path)
    return out_signals_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", nargs="+", required=True)
    parser.add_argument("--clean-subset-root", required=True)
    parser.add_argument("--output-subset-root", required=True)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    if not (1 <= int(args.severity) <= 10):
        raise SystemExit("--severity must be in [1,10]")
    depths = tuple(sorted(set(int(d) for d in args.depths)))
    bad_depths = [d for d in depths if d < 1 or d > len(OFFICIAL_S5_COMPOSITE_OPS)]
    if bad_depths:
        raise SystemExit(f"invalid depths: {bad_depths}")

    for center in args.centers:
        build_center_subset(
            center=center,
            clean_subset_root=Path(args.clean_subset_root),
            output_subset_root=Path(args.output_subset_root),
            k=args.k,
            seed=args.seed,
            severity=args.severity,
            include_clean=not bool(args.no_clean),
            depths=depths,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
