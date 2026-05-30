"""Export PTB-XL real ECGTwin latents as a Super5 latent pool.

The latent-hull AT scripts consume ``*.latent.npz`` files with ``latents`` and
``labels`` keys. This adapter converts the PTB-XL prompt-token cache into that
format while restricting records to the custom graduation-project split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_pt", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    blob = torch.load(args.cache_pt, map_location="cpu")
    with open(args.split_json, "r") as f:
        split = json.load(f)
    split_key = f"{args.split}_indices"
    wanted = {int(i) for i in split[split_key]}

    source_indices = np.asarray([int(i) for i in blob["source_indices"]], dtype=np.int64)
    keep = np.asarray([int(i) in wanted for i in source_indices], dtype=bool)
    if not keep.any():
        raise SystemExit(f"No cache rows match {split_key} from {args.split_json}")

    latents = blob["latents"].detach().cpu().numpy().astype(np.float32)[keep]
    labels = blob["super5_multi_hot"].detach().cpu().numpy().astype(np.float32)[keep]
    source_indices_keep = source_indices[keep]
    record_ids = np.asarray(blob.get("record_ids", []), dtype=str)[keep]
    primary_class = np.asarray(blob.get("primary_class", []), dtype=str)[keep]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        latents=latents,
        labels=labels,
        center_name=np.asarray("ptbxl_real_anchor"),
        class_names=np.asarray(["CD", "HYP", "MI", "NORM", "STTC"]),
        source_indices=source_indices_keep,
        record_ids=record_ids,
        primary_class=primary_class,
        source_ids=np.zeros((latents.shape[0],), dtype=np.int64),
        source_names=np.asarray(["real_anchor"]),
    )
    counts = {
        c: int(labels[:, j].sum())
        for j, c in enumerate(["CD", "HYP", "MI", "NORM", "STTC"])
    }
    meta = {
        "cache_pt": args.cache_pt,
        "split_json": args.split_json,
        "split": args.split,
        "n": int(latents.shape[0]),
        "class_counts": counts,
        "out": str(out),
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
