"""Prepare a balanced PTB-XL super5 ECGTwin reference subset.

This script selects ECGTwin-compatible PTB-XL latent records from folds 1-8 by
default, preserving the list-of-dict schema consumed by the prompt-token
generation scripts such as generate_center_prompt_token_synth.py.

Output entries keep:
  data:  (4,128) ECGTwin VAE latent
  label: text/text_embed/hr/age/sex/diagnostic_class/strat_fold plus
         super5_multi_hot and source metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
PTBXL_ROOT = Path(os.environ.get("ECG_ADV_PTBXL_ROOT", DATA_ROOT / "ptbxl")).expanduser()

from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_scheme,
)


def _load_ptbxl_super5_labels(csv_path: str) -> np.ndarray:
    scheme = get_scheme("super5")
    df = pd.read_csv(csv_path)
    return np.stack([scheme["ptbxl_fn"](scp) for scp in df.scp_codes]).astype(np.float32)


def _sample_by_primary_class(
    dataset: list[dict],
    labels: np.ndarray,
    classes: list[str],
    folds: set[int],
    per_class: int,
    seed: int,
    allow_replace: bool,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    selected_source_indices: list[int] = []
    counts: dict[str, int] = {}

    for cls in classes:
        idxs = []
        for i, sample in enumerate(dataset):
            label = sample.get("label", {})
            if int(label.get("strat_fold", -1)) not in folds:
                continue
            if label.get("diagnostic_class") != cls:
                continue
            idxs.append(i)

        if not idxs:
            raise RuntimeError(f"No PTB-XL refs found for class {cls!r} in folds={sorted(folds)}")
        replace = allow_replace and len(idxs) < per_class
        if len(idxs) < per_class and not replace:
            raise RuntimeError(
                f"Class {cls!r} has only {len(idxs)} refs, need {per_class}. "
                "Pass --allow_replace to sample with replacement."
            )
        chosen = rng.choice(np.asarray(idxs, dtype=np.int64), size=per_class, replace=replace)
        counts[cls] = int(len(chosen))

        cls_idx = CLASS_NAMES_SUPER5.index(cls)
        for src_i in chosen.tolist():
            src = dataset[int(src_i)]
            item = {
                "data": src["data"].clone() if torch.is_tensor(src["data"]) else torch.as_tensor(src["data"]).float(),
                "label": dict(src["label"]),
            }
            item["label"]["diagnostic_class"] = cls
            item["label"]["generation_class"] = cls
            item["label"]["generation_class_idx"] = cls_idx
            item["label"]["source_index"] = int(src_i)
            item["label"]["source_fold"] = int(src["label"].get("strat_fold", -1))
            item["label"]["record_id"] = str(src["label"].get("ecg_id", src_i))
            item["label"]["super5_multi_hot"] = labels[int(src_i)].astype(np.float32)
            selected.append(item)
            selected_source_indices.append(int(src_i))

    meta = {
        "classes": classes,
        "folds": sorted(folds),
        "per_class": int(per_class),
        "n_total": int(len(selected)),
        "counts": counts,
        "seed": int(seed),
        "allow_replace": bool(allow_replace),
        "source_indices": selected_source_indices,
        "source_index_unique_count": int(len(set(selected_source_indices))),
        "source_primary_counts": dict(Counter(item["label"]["diagnostic_class"] for item in selected)),
        "class_order": list(CLASS_NAMES_SUPER5),
        "selection_policy": "balanced_by_label.diagnostic_class_from_ptbxl_ecgtwin_cache",
        "leakage_rule": "Only selected from requested folds; default is PTB-XL folds 1-8.",
    }
    return selected, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl_cache", default=str(PTBXL_ROOT / "PTBXL_vae_multi_nomic.pt"))
    ap.add_argument("--ptbxl_csv", default=str(PTBXL_ROOT / "ptbxl_database.csv"))
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--classes", nargs="+", default=list(CLASS_NAMES_SUPER5))
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--total_refs", type=int, default=0,
                    help="Optional sanity check; 0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_replace", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    classes = list(args.classes)
    unknown = [c for c in classes if c not in CLASS_NAMES_SUPER5]
    if unknown:
        raise SystemExit(f"unknown super5 classes: {unknown}; valid={CLASS_NAMES_SUPER5}")

    print(f"[prep-ptbxl-ref] loading cache: {args.ptbxl_cache}", flush=True)
    dataset = torch.load(args.ptbxl_cache, map_location="cpu", weights_only=False)
    print(f"[prep-ptbxl-ref] loaded {len(dataset)} records")

    print(f"[prep-ptbxl-ref] loading PTB-XL labels: {args.ptbxl_csv}", flush=True)
    labels = _load_ptbxl_super5_labels(args.ptbxl_csv)
    if labels.shape[0] != len(dataset):
        raise RuntimeError(f"label/cache length mismatch: {labels.shape[0]} vs {len(dataset)}")

    selected, meta = _sample_by_primary_class(
        dataset=dataset,
        labels=labels,
        classes=classes,
        folds=set(args.folds),
        per_class=args.per_class,
        seed=args.seed,
        allow_replace=args.allow_replace,
    )
    if args.total_refs and len(selected) != args.total_refs:
        raise RuntimeError(f"selected {len(selected)} refs, expected --total_refs {args.total_refs}")

    out_pt = Path(args.out_pt)
    out_meta = Path(args.out_meta)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, out_pt)
    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[prep-ptbxl-ref] counts={meta['counts']}")
    print(f"[prep-ptbxl-ref] wrote {len(selected)} refs -> {out_pt}")
    print(f"[prep-ptbxl-ref] wrote meta -> {out_meta}")


if __name__ == "__main__":
    main()
