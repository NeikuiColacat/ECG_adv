#!/usr/bin/env python3
"""Evaluate fixed ECGFounder score blends for K-shot and VAE-online heads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    compute_metrics,
    evaluate_pn2021_views,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


DEFAULT_LINEAR_PROBE_DIR = (
    DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_linear_probe_v5_from_legacy_cache"
)
DEFAULT_K500_RUN = (
    DATA_ROOT
    / "paper_foundation_baselines_20260524/ecgfounder_kshot_head_ft_v5_from_legacy_cache"
    / "runs/georgia_K500_fromK500_headft_ep50_seed20260531"
)
DEFAULT_VAE_RUN = (
    DATA_ROOT
    / "paper_ecgfounder_vae_only_lhat_linear_georgia_stage27_compatible_soft_fullcap_lam045_v5_20260524"
    / "runs/georgia_K500_M20_lam0p45_ep15_seed20260531"
)
DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_score_blend_20260524"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def linear_scores(head_path: Path, features: np.ndarray) -> np.ndarray:
    state = torch.load(head_path, map_location="cpu")
    weight = state["weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["bias"].detach().cpu().numpy().astype(np.float32)
    return sigmoid(features.astype(np.float32) @ weight.T + bias)


def load_ref_ids(k500_eval_path: Path, center: str, all_centers: np.ndarray) -> dict[str, set[str]]:
    with k500_eval_path.open() as f:
        payload = json.load(f)
    ref_ids = {str(c): set() for c in np.unique(all_centers.astype(str))}
    by_center = payload.get("selected_ref_record_ids_by_center")
    if isinstance(by_center, dict) and center in by_center:
        selected = by_center[center]
    else:
        selected = payload["selected_ref_record_ids"]
    ref_ids[center] = set(str(x) for x in selected)
    return ref_ids


def summarize_target(view: dict[str, Any], center: str) -> dict[str, Any]:
    row = view[center]["per_center"][center]
    out = {
        "target_center": center,
        "macro_auroc": row["macro_auroc"],
        "macro_auprc": row["macro_auprc"],
        "drop_all_zero_macro_auroc": row.get("drop_all_zero_macro_auroc"),
        "drop_all_zero_macro_auprc": row.get("drop_all_zero_macro_auprc"),
        "per_class": row["per_class"],
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--center", default="georgia")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k500_head_path", default=str(DEFAULT_K500_RUN / "best_head.pt"))
    p.add_argument("--k500_eval_path", default=str(DEFAULT_K500_RUN / "eval_result.json"))
    p.add_argument("--vae_head_path", default=str(DEFAULT_VAE_RUN / "best_head.pt"))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    linear_dir = Path(args.linear_probe_dir)
    ptbxl_path = linear_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz"
    pn_path = linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz"
    ptbxl = np.load(ptbxl_path, allow_pickle=True)
    pn = np.load(pn_path, allow_pickle=True)

    k500_pn = linear_scores(Path(args.k500_head_path), pn["features"])
    vae_pn = linear_scores(Path(args.vae_head_path), pn["features"])
    blend_pn = (1.0 - args.alpha) * k500_pn + args.alpha * vae_pn

    ref_ids = load_ref_ids(Path(args.k500_eval_path), args.center, pn["centers"])
    pn_labels = pn["labels"].astype(np.float32)
    pn_centers = pn["centers"].astype(str)
    pn_record_ids = pn["record_ids"].astype(str)
    k500_views = evaluate_pn2021_views(
        pn_labels,
        k500_pn,
        pn_centers,
        pn_record_ids,
        ref_ids,
        report_drop_all_zero=True,
    )
    vae_views = evaluate_pn2021_views(
        pn_labels,
        vae_pn,
        pn_centers,
        pn_record_ids,
        ref_ids,
        report_drop_all_zero=True,
    )
    blend_views = evaluate_pn2021_views(
        pn_labels,
        blend_pn,
        pn_centers,
        pn_record_ids,
        ref_ids,
        report_drop_all_zero=True,
    )

    k500_ptbxl = linear_scores(Path(args.k500_head_path), ptbxl["features"])
    vae_ptbxl = linear_scores(Path(args.vae_head_path), ptbxl["features"])
    blend_ptbxl = (1.0 - args.alpha) * k500_ptbxl + args.alpha * vae_ptbxl
    fold10 = ptbxl["folds"].astype(np.int64) == 10
    ptbxl_labels = ptbxl["labels"].astype(np.float32)
    k500_source = compute_metrics(ptbxl_labels[fold10], k500_ptbxl[fold10], min_pos=1)
    vae_source = compute_metrics(ptbxl_labels[fold10], vae_ptbxl[fold10], min_pos=1)
    blend_source = compute_metrics(ptbxl_labels[fold10], blend_ptbxl[fold10], min_pos=1)

    k500_target = summarize_target(k500_views, args.center)
    vae_target = summarize_target(vae_views, args.center)
    blend_target = summarize_target(blend_views, args.center)

    payload = {
        "method": "ECGFounder fixed score blend: (1-alpha) K500 direct + alpha VAE-online AT",
        "class_names": list(CLASS_NAMES_SUPER5),
        "center": args.center,
        "alpha": args.alpha,
        "linear_probe_dir": str(linear_dir),
        "k500_head_path": str(Path(args.k500_head_path)),
        "k500_eval_path": str(Path(args.k500_eval_path)),
        "vae_head_path": str(Path(args.vae_head_path)),
        "target": {
            "k500": k500_target,
            "vae": vae_target,
            "blend": blend_target,
            "blend_minus_k500": {
                "macro_auroc": blend_target["macro_auroc"] - k500_target["macro_auroc"],
                "macro_auprc": blend_target["macro_auprc"] - k500_target["macro_auprc"],
            },
        },
        "ptbxl_fold10": {
            "k500": k500_source,
            "vae": vae_source,
            "blend": blend_source,
        },
        "pn2021_views": {
            "k500": k500_views,
            "vae": vae_views,
            "blend": blend_views,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = str(args.alpha).replace(".", "p")
    json_path = out_dir / f"{args.center}_k500_vae_score_blend_alpha{alpha_tag}.json"
    csv_path = out_dir / f"{args.center}_k500_vae_score_blend_alpha{alpha_tag}.csv"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center",
                "alpha",
                "k500_target_auroc",
                "k500_target_auprc",
                "vae_target_auroc",
                "vae_target_auprc",
                "blend_target_auroc",
                "blend_target_auprc",
                "delta_blend_vs_k500_auroc",
                "delta_blend_vs_k500_auprc",
                "blend_ptbxl_auroc",
                "blend_ptbxl_auprc",
                "json_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "center": args.center,
                "alpha": args.alpha,
                "k500_target_auroc": k500_target["macro_auroc"],
                "k500_target_auprc": k500_target["macro_auprc"],
                "vae_target_auroc": vae_target["macro_auroc"],
                "vae_target_auprc": vae_target["macro_auprc"],
                "blend_target_auroc": blend_target["macro_auroc"],
                "blend_target_auprc": blend_target["macro_auprc"],
                "delta_blend_vs_k500_auroc": (
                    blend_target["macro_auroc"] - k500_target["macro_auroc"]
                ),
                "delta_blend_vs_k500_auprc": (
                    blend_target["macro_auprc"] - k500_target["macro_auprc"]
                ),
                "blend_ptbxl_auroc": blend_source["macro_auroc"],
                "blend_ptbxl_auprc": blend_source["macro_auprc"],
                "json_path": str(json_path),
            }
        )
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")
    print(
        "[target] "
        f"k500={k500_target['macro_auroc']:.6f}/{k500_target['macro_auprc']:.6f} "
        f"vae={vae_target['macro_auroc']:.6f}/{vae_target['macro_auprc']:.6f} "
        f"blend={blend_target['macro_auroc']:.6f}/{blend_target['macro_auprc']:.6f}"
    )


if __name__ == "__main__":
    main()
