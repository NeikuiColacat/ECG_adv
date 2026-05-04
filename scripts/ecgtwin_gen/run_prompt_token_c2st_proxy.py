"""Proxy C2ST for prompt-token synthetic ECGs.

This compares held-out real PN2021 center ECGs against each synthetic arm using
a shallow logistic classifier on downsampled waveforms. It is a fast artifact
probe, not the final no-leak same-label C2ST.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _load_center_names(npz) -> List[str]:
    names = npz["center_names"]
    return [str(x) for x in names.tolist()]


def _features(signals_ct: np.ndarray, stride: int) -> np.ndarray:
    """Input `(N, 12, 1000)` -> flattened downsampled features."""
    x = signals_ct[:, :, ::stride]
    return x.reshape(x.shape[0], -1).astype(np.float32)


def _load_synth(input_dir: Path, npz_name: str) -> tuple[np.ndarray, str, str]:
    npz_path = input_dir / npz_name
    with np.load(npz_path, allow_pickle=True) as z:
        signals = z["signals"].astype(np.float32)
        center = str(z["center_name"]) if "center_name" in z.files else input_dir.name
    summary_path = input_dir / "summary.json"
    arm = input_dir.parent.name
    if summary_path.exists():
        records = json.loads(summary_path.read_text()).get("records", [])
        if records and records[0].get("arm"):
            arm = str(records[0]["arm"])
    elif input_dir.name == "gated" and (input_dir.parent / "summary.json").exists():
        records = json.loads((input_dir.parent / "summary.json").read_text()).get("records", [])
        if records and records[0].get("arm"):
            arm = str(records[0]["arm"])
    return signals, center, arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_cache", default="/root/autodl-tmp/per_center_style_classifier_full10s_max3000/cache_max3000.npz")
    ap.add_argument("--input_dirs", nargs="+", required=True)
    ap.add_argument("--npz_name", default="samples.npz")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--real_multiplier", type=int, default=5)
    ap.add_argument("--max_real", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    real_npz = np.load(args.real_cache, allow_pickle=True)
    real_signals_tc = real_npz["signals"].astype(np.float32)  # (N,1000,12)
    real_labels = real_npz["labels"].astype(np.int64)
    center_names = _load_center_names(real_npz)
    real_by_center: Dict[str, np.ndarray] = {}
    for ci, center in enumerate(center_names):
        idx = np.where(real_labels == ci)[0]
        real_by_center[center] = real_signals_tc[idx].transpose(0, 2, 1)

    rows = []
    rng = np.random.default_rng(args.seed)
    for raw_dir in args.input_dirs:
        input_dir = Path(raw_dir)
        if not (input_dir / args.npz_name).exists():
            continue
        synth, center, arm = _load_synth(input_dir, args.npz_name)
        if center not in real_by_center:
            continue
        real_pool = real_by_center[center]
        n_real = min(len(real_pool), max(len(synth) * args.real_multiplier, len(synth)), args.max_real)
        if n_real < 4 or len(synth) < 4:
            continue
        repeat_rows = []
        for rep in range(args.repeats):
            real_idx = rng.choice(len(real_pool), size=n_real, replace=False)
            real = real_pool[real_idx]
            X = np.concatenate([_features(real, args.stride), _features(synth, args.stride)], axis=0)
            y = np.concatenate([np.zeros(len(real), dtype=np.int64), np.ones(len(synth), dtype=np.int64)])
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.35, random_state=args.seed + rep, stratify=y
            )
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=args.seed + rep,
            )
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(np.int64)
            repeat_rows.append({
                "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
                "auroc": float(roc_auc_score(y_te, prob)),
            })
        rows.append({
            "input_dir": input_dir.as_posix(),
            "center": center,
            "arm": arm,
            "npz_name": args.npz_name,
            "n_synth": int(len(synth)),
            "n_real": int(n_real),
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in repeat_rows])),
            "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in repeat_rows])),
            "auroc_mean": float(np.mean([r["auroc"] for r in repeat_rows])),
            "auroc_std": float(np.std([r["auroc"] for r in repeat_rows])),
            "repeats": int(args.repeats),
            "feature_stride": int(args.stride),
        })
        print(
            f"[c2st] {center} {arm} n={len(synth)} "
            f"bacc={rows[-1]['balanced_accuracy_mean']:.3f} auroc={rows[-1]['auroc_mean']:.3f}"
        )

    if not rows:
        raise RuntimeError("no C2ST rows")
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"[done] wrote {out}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
