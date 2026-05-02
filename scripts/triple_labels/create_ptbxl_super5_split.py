"""Create a random PTB-XL super5 split for the graduate-project experiment.

Default policy follows docs/pipelines/graduate_project.md:

  train_real = 2000 random PTB-XL records
  val_real   = 2000 random records from the remaining records
  test_real  = all remaining records

The split is record-level because the user request explicitly asks for random
records. The JSON records label/combination counts so leakage/imbalance can be
audited before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, get_scheme  # noqa: E402


def _combo_name(row: np.ndarray) -> str:
    names = [CLASS_NAMES_SUPER5[i] for i, v in enumerate(row) if float(v) == 1.0]
    return "+".join(names) if names else "NONE"


def _counts(labels: np.ndarray) -> dict:
    class_counts = {
        cls: int((labels[:, i] == 1.0).sum())
        for i, cls in enumerate(CLASS_NAMES_SUPER5)
    }
    card = labels.sum(axis=1).astype(int)
    cardinality_counts = {str(k): int((card == k).sum()) for k in sorted(set(card.tolist()))}
    combos = Counter(_combo_name(row) for row in labels)
    return {
        "n": int(labels.shape[0]),
        "class_counts": class_counts,
        "cardinality_counts": cardinality_counts,
        "combo_counts": dict(combos.most_common()),
    }


def _patient_overlap(df: pd.DataFrame, a: list[int], b: list[int]) -> int:
    if "patient_id" not in df.columns:
        return 0
    pa = set(str(x) for x in df.iloc[a].patient_id.tolist())
    pb = set(str(x) for x in df.iloc[b].patient_id.tolist())
    return len(pa & pb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="/root/autodl-tmp/ptbxl/ptbxl_database.csv")
    ap.add_argument("--out_path", default=(
        "/root/autodl-tmp/graduate_project/splits/"
        "ptbxl_super5_seed42_train2000_val2000.json"
    ))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out_path)
    if out_path.exists() and not args.overwrite:
        print(f"[split] exists, use --overwrite to replace: {out_path}")
        return

    df = pd.read_csv(args.csv_path)
    scheme = get_scheme("super5")
    labels = np.stack([scheme["ptbxl_fn"](scp) for scp in df.scp_codes]).astype(np.float32)
    n = int(labels.shape[0])
    if args.train_size + args.val_size >= n:
        raise ValueError(
            f"train_size + val_size must be < N, got "
            f"{args.train_size}+{args.val_size} >= {n}"
        )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n).astype(int).tolist()
    train_indices = perm[: args.train_size]
    val_indices = perm[args.train_size: args.train_size + args.val_size]
    test_indices = perm[args.train_size + args.val_size:]

    split = {
        "version": "ptbxl_super5_random_record_split_v1",
        "seed": int(args.seed),
        "scheme": "super5",
        "class_names": list(CLASS_NAMES_SUPER5),
        "policy": "random_record_without_replacement",
        "csv_path": args.csv_path,
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(len(test_indices)),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "all_counts": _counts(labels),
        "train_counts": _counts(labels[train_indices]),
        "val_counts": _counts(labels[val_indices]),
        "test_counts": _counts(labels[test_indices]),
        "patient_overlap_counts": {
            "train_val": _patient_overlap(df, train_indices, val_indices),
            "train_test": _patient_overlap(df, train_indices, test_indices),
            "val_test": _patient_overlap(df, val_indices, test_indices),
        },
        "note": (
            "Record-level random split. Patient overlap may be nonzero because "
            "the experiment request specified random PTB-XL records."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(split, f, indent=2)
    print(f"[split] wrote {out_path}")
    print(f"[split] train counts: {split['train_counts']['class_counts']}")
    print(f"[split] val counts:   {split['val_counts']['class_counts']}")
    print(f"[split] test counts:  {split['test_counts']['class_counts']}")
    print(f"[split] patient overlap: {split['patient_overlap_counts']}")


if __name__ == "__main__":
    main()
