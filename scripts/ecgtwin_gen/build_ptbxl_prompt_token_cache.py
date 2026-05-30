"""Build a PTB-XL source-style cache for ECGTwin prompt-token training.

The existing center prompt-token trainer consumes:

  cache_root/center_full_latents/<center>.pt
  cache_root/ref_selection/<center>_k<K>_seed<seed>.json

This script adapts the existing PTB-XL ECGTwin latent cache into that schema so
we can train a source-domain token bank such as:

  <ptbxl_source_CD>, <ptbxl_source_HYP>, ...

Only PTB-XL train folds are selected by default. Fold 9/10 must stay out of
token training and reference selection.
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
    SUPER5_TO_IDX,
    get_scheme,
)


VERSION = "ptbxl_source_prompt_token_cache_v1"


def _load_super5_labels(csv_path: str) -> np.ndarray:
    scheme = get_scheme("super5")
    df = pd.read_csv(csv_path)
    return np.stack([scheme["ptbxl_fn"](scp) for scp in df.scp_codes]).astype(np.float32)


def _sex_to_str(value) -> str:
    text = str(value).upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("F"):
        return "F"
    return "U"


def _float_or_default(value, default: float) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _build_full_cache(
    dataset: list[dict],
    labels: np.ndarray,
    folds: set[int],
    center: str,
    allowed_source_indices: set[int] | None = None,
) -> dict:
    kept = []
    kept_indices = []
    for i, sample in enumerate(dataset):
        label = sample.get("label", {})
        fold = int(label.get("strat_fold", -1))
        primary_class = label.get("diagnostic_class")
        if allowed_source_indices is not None and i not in allowed_source_indices:
            continue
        if fold not in folds:
            continue
        if primary_class not in CLASS_NAMES_SUPER5:
            continue
        kept.append(sample)
        kept_indices.append(i)

    if not kept:
        raise RuntimeError(f"No PTB-XL records found for folds={sorted(folds)}")

    latents = torch.stack([
        sample["data"].detach().float()
        if torch.is_tensor(sample["data"]) else torch.as_tensor(sample["data"]).float()
        for sample in kept
    ], dim=0)
    primary_class = [sample["label"]["diagnostic_class"] for sample in kept]
    primary_class_idx = torch.tensor([SUPER5_TO_IDX[c] for c in primary_class], dtype=torch.long)
    age = torch.tensor([
        _float_or_default(sample["label"].get("age"), 60.0)
        for sample in kept
    ], dtype=torch.float32)
    hr = torch.tensor([
        _float_or_default(sample["label"].get("hr"), 75.0)
        for sample in kept
    ], dtype=torch.float32)
    record_ids = [str(sample["label"].get("ecg_id", kept_indices[j])) for j, sample in enumerate(kept)]
    patient_ids = [str(sample["label"].get("patient_id", "")) for sample in kept]
    text = [str(sample["label"].get("text", "")) for sample in kept]
    text_embed = [
        sample["label"].get("text_embed").detach().cpu().float()
        if torch.is_tensor(sample["label"].get("text_embed")) else None
        for sample in kept
    ]

    return {
        "version": VERSION,
        "center": center,
        "source": "PTB-XL ECGTwin latent cache",
        "folds": sorted(folds),
        "preprocessing": {
            "latent_source": "datasets/PTBXL/PTBXL_vae_multi_nomic.pt",
            "note": "PTB-XL ECGTwin VAE latent cache; no fold9/fold10 selected by default.",
        },
        "record_ids": record_ids,
        "patient_ids": patient_ids,
        "source_indices": kept_indices,
        "latents": latents,
        "age": age,
        "hr": hr,
        "sex": [_sex_to_str(sample["label"].get("sex", "U")) for sample in kept],
        "text": text,
        "text_embed": text_embed,
        "snomed_codes": [[] for _ in kept],
        "primary_snomed": [None for _ in kept],
        "primary_class": primary_class,
        "primary_class_idx": primary_class_idx,
        "super5_multi_hot": torch.from_numpy(labels[np.asarray(kept_indices, dtype=np.int64)]).float(),
        "strat_fold": torch.tensor([
            int(sample["label"].get("strat_fold", -1)) for sample in kept
        ], dtype=torch.long),
    }


def _select_balanced(cache: dict, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in CLASS_NAMES_SUPER5:
        pool = np.asarray([
            i for i, c in enumerate(cache["primary_class"]) if c == cls
        ], dtype=np.int64)
        if pool.size == 0:
            raise RuntimeError(f"PTB-XL cache has no records for class {cls}")
        replace = pool.size < per_class
        chosen = rng.choice(pool, size=per_class, replace=replace)
        selected.extend(int(x) for x in chosen.tolist())
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl_cache", default=str(PTBXL_ROOT / "PTBXL_vae_multi_nomic.pt"))
    ap.add_argument("--ptbxl_csv", default=str(PTBXL_ROOT / "ptbxl_database.csv"))
    ap.add_argument("--out_root", default=str(DATA_ROOT / "ecgtwin_prompt_token_super5/ptbxl_source_cache_v1"))
    ap.add_argument("--center", default="ptbxl_source")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--split_json", default=None,
                    help="Optional split JSON; when set, use split_key indices as the only source records.")
    ap.add_argument("--split_key", default="train_indices",
                    help="Key inside split_json to use, default train_indices.")
    ap.add_argument("--selection_policy", choices=["balanced_primary_class", "all_split_selected"],
                    default="balanced_primary_class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    full_dir = out_root / "center_full_latents"
    selection_dir = out_root / "ref_selection"
    full_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"{args.center}.pt"
    if args.selection_policy == "all_split_selected" and args.split_json:
        with open(args.split_json) as f:
            split = json.load(f)
        if args.split_key not in split:
            raise KeyError(f"split_json missing {args.split_key}: {args.split_json}")
        allowed_source_indices = {int(i) for i in split[args.split_key]}
        K = len(allowed_source_indices)
    else:
        split = None
        allowed_source_indices = None
        K = int(args.per_class) * len(CLASS_NAMES_SUPER5)
    selection_path = selection_dir / f"{args.center}_k{K}_seed{args.seed}.json"

    if full_path.exists() and not args.overwrite:
        print(f"[cache] using existing full cache: {full_path}")
        cache = torch.load(full_path, map_location="cpu", weights_only=False)
    else:
        print(f"[load] PTB-XL ECGTwin cache: {args.ptbxl_cache}", flush=True)
        dataset = torch.load(args.ptbxl_cache, map_location="cpu", weights_only=False)
        print(f"[load] PTB-XL labels: {args.ptbxl_csv}", flush=True)
        labels = _load_super5_labels(args.ptbxl_csv)
        if labels.shape[0] != len(dataset):
            raise RuntimeError(f"label/cache length mismatch: {labels.shape[0]} vs {len(dataset)}")
        cache = _build_full_cache(
            dataset,
            labels,
            set(args.folds),
            args.center,
            allowed_source_indices=allowed_source_indices,
        )
        torch.save(cache, full_path)
        print(f"[write] {len(cache['record_ids'])} fold-train latents -> {full_path}")

    if args.selection_policy == "all_split_selected" and args.split_json:
        selected_idx = list(range(len(cache["record_ids"])))
    else:
        selected_idx = _select_balanced(cache, args.per_class, args.seed)
    counts = Counter(cache["primary_class"][i] for i in selected_idx)
    selected_ids = [cache["record_ids"][i] for i in selected_idx]
    selection = {
        "version": VERSION,
        "center": args.center,
        "K_request": K,
        "K": len(selected_idx),
        "seed": int(args.seed),
        "folds": sorted(int(f) for f in args.folds),
        "sampling_policy": args.selection_policy,
        "per_class": int(args.per_class),
        "selected_indices_in_full_cache": selected_idx,
        "selected_record_ids": selected_ids,
        "ref_record_ids": selected_ids,
        "primary_class_counts": dict(counts),
        "full_cache": str(full_path),
        "split_json": str(args.split_json) if args.split_json else None,
        "split_key": args.split_key if args.split_json else None,
        "source_indices_filter_size": len(allowed_source_indices) if allowed_source_indices is not None else None,
        "leakage_rule": (
            "When split_json is provided, only the requested split indices are used. "
            "Otherwise only PTB-XL folds 1-8 are used by default."
        ),
    }
    with selection_path.open("w") as f:
        json.dump(selection, f, indent=2)
    print(f"[write] selection {dict(counts)} -> {selection_path}")


if __name__ == "__main__":
    main()
