#!/usr/bin/env python3
"""Paper-safe ECGFounder tri-head blend.

Blend three existing ECGFounder heads:

1. direct K=500 head fine-tuned on the target center;
2. sourcebase VAE-online head;
3. k500base VAE-online head.

The convex weights are selected only on the known K=500 target ECG internal
validation split. PN2021 held-out target-center data is used only for reporting
after weight selection.
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
    DEFAULT_CENTERS,
    DEFAULT_K500_RUN_ROOT,
    DEFAULT_LINEAR_PROBE_DIR,
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


DEFAULT_SOURCEBASE_VAE_ROOT = (
    DATA_ROOT / "paper_ecgfounder_vae_lhat_seed20260531anchors_sourcebase_v6_20260524/runs"
)
DEFAULT_K500BASE_VAE_ROOT = (
    DATA_ROOT / "paper_ecgfounder_vae_lhat_seed20260531anchors_k500base_v6_20260524/runs"
)


def convex_weight_grid(step: float) -> list[tuple[float, float, float]]:
    if not 0.0 < step <= 1.0:
        raise ValueError(f"bad grid step {step}")
    n = int(round(1.0 / step))
    weights: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            weights.append((round(i / n, 6), round(j / n, 6), round(k / n, 6)))
    return weights


def blend_three(
    k500_logits: np.ndarray,
    sourcebase_logits: np.ndarray,
    k500base_logits: np.ndarray,
    weights: tuple[float, float, float],
    *,
    blend_space: str,
) -> np.ndarray:
    w0, w1, w2 = weights
    if blend_space == "logit":
        return sigmoid(w0 * k500_logits + w1 * sourcebase_logits + w2 * k500base_logits)
    return (
        w0 * sigmoid(k500_logits)
        + w1 * sigmoid(sourcebase_logits)
        + w2 * sigmoid(k500base_logits)
    )


def single_class_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def choose_global_weights(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    weight_grid: list[tuple[float, float, float]],
    *,
    blend_space: str,
) -> dict[str, Any]:
    rows = []
    for weights in weight_grid:
        scores = blend_three(
            logits["k500"],
            logits["sourcebase"],
            logits["k500base"],
            weights,
            blend_space=blend_space,
        )
        metrics = compute_metrics(y_true.astype(np.float32), scores.astype(np.float32), min_pos=1)
        row = {"weights": list(weights), "metrics": metrics}
        rows.append(row)
    best = max(
        rows,
        key=lambda r: (
            float(r["metrics"]["macro_auprc"]),
            float(r["metrics"]["macro_auroc"]),
        ),
    )
    return {"best_weights": best["weights"], "grid": rows}


def choose_classwise_weights(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    weight_grid: list[tuple[float, float, float]],
    *,
    blend_space: str,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}
    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        y = y_true[:, idx].astype(np.float32)
        n_pos = int((y > 0.5).sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 1 or n_neg < 1:
            choices[cls] = {
                "weights": [1.0, 0.0, 0.0],
                "n_pos": n_pos,
                "selection": "direct_fallback_insufficient_val_positives",
            }
            continue
        best: dict[str, Any] | None = None
        for weights in weight_grid:
            score = blend_three(
                logits["k500"][:, idx : idx + 1],
                logits["sourcebase"][:, idx : idx + 1],
                logits["k500base"][:, idx : idx + 1],
                weights,
                blend_space=blend_space,
            )[:, 0]
            metrics = single_class_metrics(y, score)
            row = {
                "weights": list(weights),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "n_pos": n_pos,
            }
            if best is None or (
                row["auprc"] > best["auprc"]
                or (row["auprc"] == best["auprc"] and row["auroc"] > best["auroc"])
            ):
                best = row
        assert best is not None
        choices[cls] = best
    return choices


def apply_classwise_weights(
    logits: dict[str, np.ndarray],
    choices: dict[str, Any],
    *,
    blend_space: str,
) -> np.ndarray:
    out = np.zeros_like(logits["k500"], dtype=np.float32)
    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        weights = tuple(float(x) for x in choices[cls]["weights"])
        out[:, idx] = blend_three(
            logits["k500"][:, idx : idx + 1],
            logits["sourcebase"][:, idx : idx + 1],
            logits["k500base"][:, idx : idx + 1],
            weights,  # type: ignore[arg-type]
            blend_space=blend_space,
        )[:, 0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    p.add_argument("--weight_step", type=float, default=0.05)
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k500_run_root", default=str(DEFAULT_K500_RUN_ROOT))
    p.add_argument("--sourcebase_vae_run_root", default=str(DEFAULT_SOURCEBASE_VAE_ROOT))
    p.add_argument("--k500base_vae_run_root", default=str(DEFAULT_K500BASE_VAE_ROOT))
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument(
        "--out_dir",
        default=str(DATA_ROOT / "paper_ecgfounder_k500val_trihead_blend_v6_20260524"),
    )
    args = p.parse_args()

    weight_grid = convex_weight_grid(args.weight_step)
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
        "method": "ECGFounder tri-head convex blend selected on K500-internal validation",
        "class_names": list(CLASS_NAMES_SUPER5),
        "blend_space": args.blend_space,
        "weight_step": args.weight_step,
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "centers": {},
    }

    for center in args.centers:
        print(f"[center] {center}")
        k500_run = center_k500_run(Path(args.k500_run_root), center)
        sourcebase_run = center_vae_run(Path(args.sourcebase_vae_run_root), center)
        k500base_run = center_vae_run(Path(args.k500base_vae_run_root), center)
        head_paths = {
            "k500": k500_run / "best_head.pt",
            "sourcebase": sourcebase_run / "best_head.pt",
            "k500base": k500base_run / "best_head.pt",
        }
        for name, path in head_paths.items():
            print(f"  {name}={path}")

        pn_logits = {
            name: head_logits(path, pn_features)
            for name, path in head_paths.items()
        }
        ptbxl_logits = {
            name: head_logits(path, ptbxl_features)
            for name, path in head_paths.items()
        }
        k500_eval_path = k500_run / "eval_result.json"
        ref_ids = build_ref_ids(k500_eval_path, center, pn_centers)
        selected_ids = ref_ids.get(center, set())
        k500_mask = (pn_centers == center) & np.asarray([rid in selected_ids for rid in pn_record_ids])
        if int(k500_mask.sum()) == 0:
            raise RuntimeError(f"no K500 records found in feature cache for {center}")
        inner_val_mask = build_k500_internal_val_mask(
            pn_labels[k500_mask],
            val_fraction=args.target_real_val_fraction,
            seed=args.target_real_val_seed,
        )
        k500_indices = np.where(k500_mask)[0]
        val_indices = k500_indices[inner_val_mask]
        val_logits = {name: arr[val_indices] for name, arr in pn_logits.items()}
        y_val = pn_labels[val_indices]

        global_choice = choose_global_weights(
            y_val, val_logits, weight_grid, blend_space=args.blend_space
        )
        class_choices = choose_classwise_weights(
            y_val, val_logits, weight_grid, blend_space=args.blend_space
        )
        global_weights = tuple(float(x) for x in global_choice["best_weights"])

        method_scores = {
            "k500": sigmoid(pn_logits["k500"]),
            "sourcebase_vae": sigmoid(pn_logits["sourcebase"]),
            "k500base_vae": sigmoid(pn_logits["k500base"]),
            "global_k500val_trihead": blend_three(
                pn_logits["k500"],
                pn_logits["sourcebase"],
                pn_logits["k500base"],
                global_weights,  # type: ignore[arg-type]
                blend_space=args.blend_space,
            ),
            "classwise_k500val_trihead": apply_classwise_weights(
                pn_logits,
                class_choices,
                blend_space=args.blend_space,
            ),
        }
        target_metrics = {}
        for method_name, scores in method_scores.items():
            views = evaluate_pn2021_views(
                pn_labels,
                scores,
                pn_centers,
                pn_record_ids,
                ref_ids,
                report_drop_all_zero=True,
            )
            target_metrics[method_name] = summarize_target(views, center)
        k500_target = target_metrics["k500"]

        ptbxl_scores = {
            "k500": sigmoid(ptbxl_logits["k500"]),
            "sourcebase_vae": sigmoid(ptbxl_logits["sourcebase"]),
            "k500base_vae": sigmoid(ptbxl_logits["k500base"]),
            "global_k500val_trihead": blend_three(
                ptbxl_logits["k500"],
                ptbxl_logits["sourcebase"],
                ptbxl_logits["k500base"],
                global_weights,  # type: ignore[arg-type]
                blend_space=args.blend_space,
            ),
        }
        source_metrics = {
            name: compute_metrics(ptbxl_labels[fold10], scores[fold10], min_pos=1)
            for name, scores in ptbxl_scores.items()
        }

        for method_name, target in target_metrics.items():
            rows.append(
                {
                    "center": center,
                    "method": method_name,
                    "target_auroc": target["macro_auroc"],
                    "target_auprc": target["macro_auprc"],
                    "delta_vs_k500_auroc": target["macro_auroc"] - k500_target["macro_auroc"],
                    "delta_vs_k500_auprc": target["macro_auprc"] - k500_target["macro_auprc"],
                    "global_weights": json.dumps(list(global_weights)),
                    "classwise_weights": json.dumps(
                        {k: v["weights"] for k, v in class_choices.items()},
                        sort_keys=True,
                    ),
                }
            )
        payload["centers"][center] = {
            "paths": {k: str(v) for k, v in head_paths.items()},
            "k500_eval_path": str(k500_eval_path),
            "k500_val": {
                "n": int(len(val_indices)),
                "positive_counts": pn_labels[val_indices].sum(axis=0).astype(int).tolist(),
                "record_ids": pn_record_ids[val_indices].tolist(),
            },
            "global_weight_selection": global_choice,
            "classwise_weight_selection": class_choices,
            "target": target_metrics,
            "ptbxl_fold10": source_metrics,
        }
        best_blend = max(
            [r for r in rows if r["center"] == center and r["method"].endswith("trihead")],
            key=lambda r: (float(r["target_auprc"]), float(r["target_auroc"])),
        )
        print(
            f"  global weights={list(global_weights)}; "
            f"best reported {best_blend['method']} "
            f"{float(best_blend['target_auroc']):.6f}/"
            f"{float(best_blend['target_auprc']):.6f} "
            f"delta={float(best_blend['delta_vs_k500_auroc']):+.6f}/"
            f"{float(best_blend['delta_vs_k500_auprc']):+.6f}"
        )

    csv_path = out_dir / f"ecgfounder_k500val_trihead_{args.blend_space}_step{args.weight_step:g}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"ecgfounder_k500val_trihead_{args.blend_space}_step{args.weight_step:g}.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
