#!/usr/bin/env python3
"""Summarize ECGFounder full-FT target-val pilot result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOT = Path("/root/autodl-tmp/paper_ecgfounder_fullft_super5_20260523/runs")


def arm_from_config(result: dict) -> str:
    cfg = result.get("config", {})
    if not result.get("vae_stream_enabled", False):
        return "no_vae"
    aw = cfg.get("adv_weight", "unknown")
    return f"vae_aw{aw:g}" if isinstance(aw, (float, int)) else f"vae_aw{aw}"


def metric_pair(metrics: dict) -> str:
    return f"{metrics['macro_auroc']:.4f}/{metrics['macro_auprc']:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--pattern", default="*/eval_result.json")
    ap.add_argument("--csv_out", default="")
    ap.add_argument("--md_out", default="")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.runs_root).glob(args.pattern)):
        with path.open() as f:
            result = json.load(f)
        cfg = result.get("config", {})
        if int(result.get("target_val_K") or 0) <= 0:
            continue
        target = result["target_excluding_ref"]
        drop = result["target_drop_all_zero_excluding_ref"]
        ptbxl = result["ptbxl_fold10"]
        row = {
            "center": result["center"],
            "arm": arm_from_config(result),
            "K": result["K"],
            "target_train_K": result["target_train_K"],
            "target_val_K": result["target_val_K"],
            "target_val_seed": cfg.get("target_val_seed"),
            "split_mode": result.get("target_val_split_mode"),
            "selection_metric": result.get("selection_metric"),
            "best_epoch": result.get("best_epoch"),
            "ptbxl_auroc": ptbxl["macro_auroc"],
            "ptbxl_auprc": ptbxl["macro_auprc"],
            "target_auroc": target["macro_auroc"],
            "target_auprc": target["macro_auprc"],
            "drop_auroc": drop["macro_auroc"],
            "drop_auprc": drop["macro_auprc"],
            "n_target_eval": result.get("n_target_eval"),
            "n_drop_eval": result.get("n_target_drop_all_zero_eval"),
            "path": str(path),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["center"], str(r["target_val_seed"]), r["arm"]))

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "| center | split seed | arm | best | PTB-XL | target | drop-all-zero |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {center} | {target_val_seed} | {arm} | {best_epoch} | "
            "{ptbxl_auroc:.4f}/{ptbxl_auprc:.4f} | "
            "{target_auroc:.4f}/{target_auprc:.4f} | "
            "{drop_auroc:.4f}/{drop_auprc:.4f} |".format(**r)
        )
    md = "\n".join(lines) + "\n"
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
