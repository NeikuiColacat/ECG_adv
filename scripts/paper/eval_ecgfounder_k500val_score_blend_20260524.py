#!/usr/bin/env python3
"""K500-internal ECGFounder K-shot/VAE score-blend selection.

The older ECGFounder blend grid is useful for diagnosis but chooses alpha from
PN2021 held-out target metrics. This script chooses fixed/global and classwise
blend weights only from the known K=500 target ECGs, then reports the held-out
target-center metrics after ref exclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.eval_ecgfounder_score_blend_20260524 import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_ALPHAS,
    DEFAULT_CENTERS,
    DEFAULT_K500_RUN_ROOT,
    DEFAULT_LINEAR_PROBE_DIR,
    blend_arrays,
    build_ref_ids,
    center_k500_run,
    center_vae_run,
    head_logits,
    sigmoid,
    summarize_target,
)
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    compute_metrics,
    evaluate_pn2021_views,
)
from scripts.pgd_cross_center.synth_online_at_super5 import build_k500_internal_val_mask  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


DEFAULT_SAME_ANCHOR_VAE_ROOT = (
    DATA_ROOT / "paper_ecgfounder_vae_lhat_seed20260531anchors_sourcebase_v6_20260524/runs"
)


def metric_for_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return compute_metrics(y_true.astype(np.float32), scores.astype(np.float32), min_pos=1)


def choose_global_alpha(
    y_true: np.ndarray,
    k500_logits: np.ndarray,
    vae_logits: np.ndarray,
    *,
    alphas: list[float],
    blend_space: str,
) -> dict[str, Any]:
    rows = []
    for alpha in alphas:
        scores = blend_arrays(k500_logits, vae_logits, alpha=alpha, blend_space=blend_space)
        metrics = metric_for_scores(y_true, scores)
        rows.append({"alpha": float(alpha), "metrics": metrics})
    best = max(
        rows,
        key=lambda r: (
            float(r["metrics"]["macro_auprc"]),
            float(r["metrics"]["macro_auroc"]),
        ),
    )
    return {"best_alpha": float(best["alpha"]), "grid": rows}


def choose_classwise_alphas(
    y_true: np.ndarray,
    k500_logits: np.ndarray,
    vae_logits: np.ndarray,
    *,
    alphas: list[float],
    blend_space: str,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}
    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        y = y_true[:, idx : idx + 1].astype(np.float32)
        n_pos = int((y[:, 0] > 0.5).sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 1 or n_neg < 1:
            choices[cls] = {
                "alpha": 0.0,
                "n_pos": n_pos,
                "selection": "direct_fallback_insufficient_val_positives",
            }
            continue
        best: dict[str, Any] | None = None
        for alpha in alphas:
            scores = blend_arrays(
                k500_logits[:, idx : idx + 1],
                vae_logits[:, idx : idx + 1],
                alpha=alpha,
                blend_space=blend_space,
            )[:, 0]
            row = {
                "alpha": float(alpha),
                "auroc": float(roc_auc_score(y[:, 0], scores)),
                "auprc": float(average_precision_score(y[:, 0], scores)),
                "n_pos": n_pos,
            }
            if best is None or (
                float(row["auprc"]) > float(best["auprc"])
                or (
                    float(row["auprc"]) == float(best["auprc"])
                    and float(row["auroc"]) > float(best["auroc"])
                )
            ):
                best = row
        assert best is not None
        choices[cls] = best
    return choices


def apply_classwise_blend(
    k500_logits: np.ndarray,
    vae_logits: np.ndarray,
    choices: dict[str, Any],
    *,
    blend_space: str,
) -> np.ndarray:
    if blend_space == "logit":
        out_logits = k500_logits.copy()
        for idx, cls in enumerate(CLASS_NAMES_SUPER5):
            alpha = float(choices[cls]["alpha"])
            out_logits[:, idx] = (1.0 - alpha) * k500_logits[:, idx] + alpha * vae_logits[:, idx]
        return sigmoid(out_logits)
    k500_scores = sigmoid(k500_logits)
    vae_scores = sigmoid(vae_logits)
    out = k500_scores.copy()
    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        alpha = float(choices[cls]["alpha"])
        out[:, idx] = (1.0 - alpha) * k500_scores[:, idx] + alpha * vae_scores[:, idx]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k500_run_root", default=str(DEFAULT_K500_RUN_ROOT))
    p.add_argument("--vae_run_root", default=str(DEFAULT_SAME_ANCHOR_VAE_ROOT))
    p.add_argument("--vae_adapter_scale", type=float, default=1.0)
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument(
        "--out_dir",
        default=str(DATA_ROOT / "paper_ecgfounder_k500val_score_blend_v6_20260524"),
    )
    args = p.parse_args()

    linear_dir = Path(args.linear_probe_dir)
    ptbxl = np.load(
        linear_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz",
        allow_pickle=True,
    )
    pn = np.load(
        linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz",
        allow_pickle=True,
    )
    pn_features = pn["features"].astype(np.float32)
    pn_labels = pn["labels"].astype(np.float32)
    pn_centers = pn["centers"].astype(str)
    pn_record_ids = pn["record_ids"].astype(str)
    ptbxl_features = ptbxl["features"].astype(np.float32)
    ptbxl_labels = ptbxl["labels"].astype(np.float32)
    fold10 = ptbxl["folds"].astype(np.int64) == 10

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "method": "ECGFounder K500/VAE score fusion selected on K500-internal validation",
        "class_names": list(CLASS_NAMES_SUPER5),
        "blend_space": args.blend_space,
        "alphas": args.alphas,
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "centers": {},
    }

    for center in args.centers:
        print(f"[center] {center}")
        k500_run = center_k500_run(Path(args.k500_run_root), center)
        vae_run = center_vae_run(Path(args.vae_run_root), center)
        k500_head_path = k500_run / "best_head.pt"
        vae_head_path = vae_run / "best_head.pt"
        k500_eval_path = k500_run / "eval_result.json"
        print(f"  k500={k500_head_path}")
        print(f"  vae={vae_head_path}")

        k500_pn_logits = head_logits(k500_head_path, pn_features)
        vae_pn_logits = head_logits(vae_head_path, pn_features, adapter_scale=args.vae_adapter_scale)
        k500_ptbxl_logits = head_logits(k500_head_path, ptbxl_features)
        vae_ptbxl_logits = head_logits(vae_head_path, ptbxl_features, adapter_scale=args.vae_adapter_scale)

        ref_ids = build_ref_ids(k500_eval_path, center, pn_centers)
        selected_ids = ref_ids.get(center, set())
        k500_mask = (pn_centers == center) & np.asarray([rid in selected_ids for rid in pn_record_ids])
        if int(k500_mask.sum()) == 0:
            raise RuntimeError(f"no K500 records found in feature cache for {center}")
        val_mask_inner = build_k500_internal_val_mask(
            pn_labels[k500_mask],
            val_fraction=args.target_real_val_fraction,
            seed=args.target_real_val_seed,
        )
        k500_indices = np.where(k500_mask)[0]
        val_indices = k500_indices[val_mask_inner]
        y_val = pn_labels[val_indices]
        global_choice = choose_global_alpha(
            y_val,
            k500_pn_logits[val_indices],
            vae_pn_logits[val_indices],
            alphas=args.alphas,
            blend_space=args.blend_space,
        )
        class_choices = choose_classwise_alphas(
            y_val,
            k500_pn_logits[val_indices],
            vae_pn_logits[val_indices],
            alphas=args.alphas,
            blend_space=args.blend_space,
        )

        k500_pn_scores = sigmoid(k500_pn_logits)
        vae_pn_scores = sigmoid(vae_pn_logits)
        global_alpha = float(global_choice["best_alpha"])
        global_pn_scores = blend_arrays(
            k500_pn_logits,
            vae_pn_logits,
            alpha=global_alpha,
            blend_space=args.blend_space,
        )
        classwise_pn_scores = apply_classwise_blend(
            k500_pn_logits,
            vae_pn_logits,
            class_choices,
            blend_space=args.blend_space,
        )

        methods = {
            "k500": k500_pn_scores,
            "vae": vae_pn_scores,
            "global_k500val_blend": global_pn_scores,
            "classwise_k500val_blend": classwise_pn_scores,
        }
        method_targets = {}
        for method_name, scores in methods.items():
            views = evaluate_pn2021_views(
                pn_labels,
                scores,
                pn_centers,
                pn_record_ids,
                ref_ids,
                report_drop_all_zero=True,
            )
            method_targets[method_name] = summarize_target(views, center)
        k500_target = method_targets["k500"]

        # Source fold10 is only a guardrail report; alpha selection never uses it.
        global_ptbxl = blend_arrays(
            k500_ptbxl_logits,
            vae_ptbxl_logits,
            alpha=global_alpha,
            blend_space=args.blend_space,
        )
        source_metrics = {
            "k500": metric_for_scores(ptbxl_labels[fold10], sigmoid(k500_ptbxl_logits)[fold10]),
            "vae": metric_for_scores(ptbxl_labels[fold10], sigmoid(vae_ptbxl_logits)[fold10]),
            "global_k500val_blend": metric_for_scores(ptbxl_labels[fold10], global_ptbxl[fold10]),
        }

        for method_name, target in method_targets.items():
            rows.append(
                {
                    "center": center,
                    "method": method_name,
                    "target_auroc": target["macro_auroc"],
                    "target_auprc": target["macro_auprc"],
                    "delta_vs_k500_auroc": target["macro_auroc"] - k500_target["macro_auroc"],
                    "delta_vs_k500_auprc": target["macro_auprc"] - k500_target["macro_auprc"],
                    "global_alpha_selected": global_alpha,
                    "classwise_alphas": json.dumps({k: v["alpha"] for k, v in class_choices.items()}, sort_keys=True),
                    "k500_head_path": str(k500_head_path),
                    "vae_head_path": str(vae_head_path),
                }
            )

        payload["centers"][center] = {
            "paths": {
                "k500_head_path": str(k500_head_path),
                "vae_head_path": str(vae_head_path),
                "k500_eval_path": str(k500_eval_path),
            },
            "k500_val": {
                "n": int(len(val_indices)),
                "positive_counts": pn_labels[val_indices].sum(axis=0).astype(int).tolist(),
                "record_ids": pn_record_ids[val_indices].tolist(),
            },
            "global_alpha_selection": global_choice,
            "classwise_alpha_selection": class_choices,
            "target": method_targets,
            "ptbxl_fold10": source_metrics,
        }
        best_blend = max(
            [r for r in rows if r["center"] == center and r["method"].endswith("blend")],
            key=lambda r: (float(r["target_auprc"]), float(r["target_auroc"])),
        )
        print(
            f"  selected global alpha={global_alpha:.2f}; "
            f"best reported {best_blend['method']} "
            f"{float(best_blend['target_auroc']):.6f}/{float(best_blend['target_auprc']):.6f} "
            f"delta={float(best_blend['delta_vs_k500_auroc']):+.6f}/"
            f"{float(best_blend['delta_vs_k500_auprc']):+.6f}"
        )

    csv_path = out_dir / f"ecgfounder_k500val_selected_{args.blend_space}_blends.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"ecgfounder_k500val_selected_{args.blend_space}_blends.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
