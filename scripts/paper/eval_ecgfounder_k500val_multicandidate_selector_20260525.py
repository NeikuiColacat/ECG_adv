#!/usr/bin/env python3
"""Paper-safe ECGFounder multi-candidate selector.

This script reuses cached ECGFounder features and saved heads from direct K500
fine-tuning plus multiple VAE-online-AT variants. Candidate selection is based
only on the known K=500 target-center ECG internal validation split. The
ref-excluded PN2021 target-center data is used only after selection for report.
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
    blend_arrays,
    build_ref_ids,
    center_k500_run,
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


DEFAULT_CANDIDATE_SPECS = [
    "sourcebase_seedanchors="
    + str(DATA_ROOT / "paper_ecgfounder_vae_lhat_seed20260531anchors_sourcebase_v6_20260524/runs"),
    "k500base_seedanchors="
    + str(DATA_ROOT / "paper_ecgfounder_vae_lhat_seed20260531anchors_k500base_v6_20260524/runs"),
    "warm20_lam005="
    + str(DATA_ROOT / "paper_ecgfounder_hardbce_inc_warm20_lam005_ep20_v6_20260525"),
    "rankrare="
    + str(DATA_ROOT / "paper_ecgfounder_rankrare_warm20_lam005_ep20_v6_20260525"),
    "basehard_pow3="
    + str(DATA_ROOT / "paper_ecgfounder_basehard_pow3_warm20_lam005_ep20_v6_20260525"),
    "tanchor01="
    + str(DATA_ROOT / "paper_ecgfounder_hardbce_inc_warm20_lam005_tanchor01_ep20_v6_20260525"),
    "tanchor05="
    + str(DATA_ROOT / "paper_ecgfounder_hardbce_inc_warm20_lam005_tanchor05_ep20_v6_20260525"),
    "linear_l2="
    + str(DATA_ROOT / "paper_ecgfounder_linear_l2_warm20_lam005_ep20_v6_20260525"),
]


def parse_candidate_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"candidate must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty candidate name in {spec!r}")
        if name == "direct":
            raise ValueError("'direct' is reserved for the K500 baseline")
        if name in out:
            raise ValueError(f"duplicate candidate name {name!r}")
        out[name] = Path(path).expanduser()
    return out


def single_match(paths: list[Path], *, label: str) -> Path:
    uniq = sorted({p for p in paths if p.exists()})
    if len(uniq) != 1:
        found = "\n".join(str(p) for p in uniq[:20])
        raise FileNotFoundError(f"expected one {label}, found {len(uniq)}\n{found}")
    return uniq[0]


def find_candidate_head(root: Path, center: str) -> Path:
    candidates: list[Path] = []
    candidates.extend(root.glob(f"{center}_K500_*/best_head.pt"))
    candidates.extend((root / "runs").glob(f"{center}_K500_*/best_head.pt"))
    candidates.extend((root / center).glob("runs/*/best_head.pt"))
    candidates.extend((root / center / "runs").glob("*/best_head.pt"))
    return single_match(candidates, label=f"head for {center} under {root}")


def metric_for_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return compute_metrics(y_true.astype(np.float32), scores.astype(np.float32), min_pos=1)


def per_class_metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def make_blended_scores(
    direct_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    alpha: float,
    blend_space: str,
) -> np.ndarray:
    if alpha <= 0.0:
        return sigmoid(direct_logits)
    return blend_arrays(direct_logits, candidate_logits, alpha=alpha, blend_space=blend_space)


def choose_global_option(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    *,
    candidate_names: list[str],
    alphas: list[float],
    blend_space: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in candidate_names:
        for alpha in alphas:
            scores = make_blended_scores(
                logits["direct"],
                logits[name],
                alpha=alpha,
                blend_space=blend_space,
            )
            metrics = metric_for_scores(y_true, scores)
            rows.append({"candidate": name, "alpha": float(alpha), "metrics": metrics})
    best = max(
        rows,
        key=lambda r: (
            float(r["metrics"]["macro_auprc"]),
            float(r["metrics"]["macro_auroc"]),
            -float(r["alpha"]),
        ),
    )
    return {"best": {"candidate": best["candidate"], "alpha": best["alpha"]}, "grid": rows}


def choose_classwise_options(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    *,
    source_y_true: np.ndarray | None,
    source_logits: dict[str, np.ndarray] | None,
    candidate_names: list[str],
    alphas: list[float],
    blend_space: str,
    class_selection_metric: str,
    source_selection_weight: float,
    auroc_selection_weight: float,
    source_auroc_selection_weight: float,
    min_val_auprc_gain: float,
    min_val_auroc_gain: float,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}

    def score_row(row: dict[str, Any]) -> tuple[float, float, float]:
        val_auprc = float(row["auprc"])
        val_auroc = float(row["auroc"])
        src_auprc = float(row.get("source_auprc", 0.0) or 0.0)
        src_auroc = float(row.get("source_auroc", 0.0) or 0.0)
        if class_selection_metric == "val_plus_source_auprc":
            primary = val_auprc + float(source_selection_weight) * src_auprc
        elif class_selection_metric == "val_source_hmean_auprc":
            primary = 0.0 if (val_auprc + src_auprc) <= 0.0 else (
                2.0 * val_auprc * src_auprc / (val_auprc + src_auprc)
            )
        else:
            primary = val_auprc
        primary += float(auroc_selection_weight) * val_auroc
        primary += float(source_auroc_selection_weight) * src_auroc
        return (primary, val_auprc, val_auroc + 0.01 * src_auroc)

    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        y = y_true[:, idx].astype(np.float32)
        n_pos = int((y > 0.5).sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 1 or n_neg < 1:
            choices[cls] = {
                "candidate": "direct",
                "alpha": 0.0,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "selection": "direct_fallback_insufficient_val_class_balance",
            }
            continue

        direct_scores = sigmoid(logits["direct"][:, idx : idx + 1])[:, 0]
        direct_metrics = per_class_metric(y, direct_scores)
        direct_source_metrics: dict[str, float] = {"auroc": 0.0, "auprc": 0.0}
        if source_y_true is not None and source_logits is not None:
            source_y = source_y_true[:, idx].astype(np.float32)
            if int((source_y > 0.5).sum()) > 0 and int((source_y <= 0.5).sum()) > 0:
                source_scores = sigmoid(source_logits["direct"][:, idx : idx + 1])[:, 0]
                direct_source_metrics = per_class_metric(source_y, source_scores)
        best: dict[str, Any] = {
            "candidate": "direct",
            "alpha": 0.0,
            "auroc": direct_metrics["auroc"],
            "auprc": direct_metrics["auprc"],
            "source_auroc": direct_source_metrics["auroc"],
            "source_auprc": direct_source_metrics["auprc"],
            "direct_auroc": direct_metrics["auroc"],
            "direct_auprc": direct_metrics["auprc"],
            "n_pos": n_pos,
            "n_neg": n_neg,
            "selection": "direct",
        }
        for name in candidate_names:
            for alpha in alphas:
                scores = make_blended_scores(
                    logits["direct"][:, idx : idx + 1],
                    logits[name][:, idx : idx + 1],
                    alpha=alpha,
                    blend_space=blend_space,
                )[:, 0]
                metrics = per_class_metric(y, scores)
                source_metrics: dict[str, float] = {"auroc": 0.0, "auprc": 0.0}
                if source_y_true is not None and source_logits is not None:
                    source_y = source_y_true[:, idx].astype(np.float32)
                    if int((source_y > 0.5).sum()) > 0 and int((source_y <= 0.5).sum()) > 0:
                        source_scores = make_blended_scores(
                            source_logits["direct"][:, idx : idx + 1],
                            source_logits[name][:, idx : idx + 1],
                            alpha=alpha,
                            blend_space=blend_space,
                        )[:, 0]
                        source_metrics = per_class_metric(source_y, source_scores)
                row = {
                    "candidate": name,
                    "alpha": float(alpha),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "source_auroc": source_metrics["auroc"],
                    "source_auprc": source_metrics["auprc"],
                    "direct_auroc": direct_metrics["auroc"],
                    "direct_auprc": direct_metrics["auprc"],
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "selection": "candidate",
                }
                if score_row(row) > score_row(best) or (
                    score_row(row) == score_row(best) and row["alpha"] < best["alpha"]
                ):
                    best = row

        if best["candidate"] != "direct":
            auprc_gain = float(best["auprc"]) - float(direct_metrics["auprc"])
            auroc_gain = float(best["auroc"]) - float(direct_metrics["auroc"])
            if auprc_gain < min_val_auprc_gain and auroc_gain < min_val_auroc_gain:
                best = {
                    "candidate": "direct",
                    "alpha": 0.0,
                    "auroc": direct_metrics["auroc"],
                    "auprc": direct_metrics["auprc"],
                    "source_auroc": direct_source_metrics["auroc"],
                    "source_auprc": direct_source_metrics["auprc"],
                    "direct_auroc": direct_metrics["auroc"],
                    "direct_auprc": direct_metrics["auprc"],
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "selection": "direct_conservative_fallback",
                    "rejected_best": best,
                }
        choices[cls] = best
    return choices


def apply_global_option(
    logits: dict[str, np.ndarray],
    choice: dict[str, Any],
    *,
    blend_space: str,
) -> np.ndarray:
    best = choice["best"]
    return make_blended_scores(
        logits["direct"],
        logits[str(best["candidate"])],
        alpha=float(best["alpha"]),
        blend_space=blend_space,
    )


def apply_classwise_options(
    logits: dict[str, np.ndarray],
    choices: dict[str, Any],
    *,
    blend_space: str,
) -> np.ndarray:
    out = sigmoid(logits["direct"]).astype(np.float32)
    for idx, cls in enumerate(CLASS_NAMES_SUPER5):
        choice = choices[cls]
        name = str(choice["candidate"])
        alpha = float(choice["alpha"])
        if name == "direct" or alpha <= 0.0:
            continue
        out[:, idx] = make_blended_scores(
            logits["direct"][:, idx : idx + 1],
            logits[name][:, idx : idx + 1],
            alpha=alpha,
            blend_space=blend_space,
        )[:, 0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--candidate", action="append", dest="candidates", default=[])
    p.add_argument("--use_default_candidates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k500_run_root", default=str(DEFAULT_K500_RUN_ROOT))
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument(
        "--class_selection_metric",
        choices=["val_auprc", "val_plus_source_auprc", "val_source_hmean_auprc"],
        default="val_auprc",
    )
    p.add_argument("--source_selection_weight", type=float, default=0.2)
    p.add_argument(
        "--auroc_selection_weight",
        type=float,
        default=0.0,
        help="Optional classwise selector weight for K500-val AUROC in the primary score.",
    )
    p.add_argument(
        "--source_auroc_selection_weight",
        type=float,
        default=0.0,
        help="Optional classwise selector weight for PTB-XL fold10 AUROC in the primary score.",
    )
    p.add_argument("--min_val_auprc_gain", type=float, default=0.0)
    p.add_argument("--min_val_auroc_gain", type=float, default=0.0)
    p.add_argument(
        "--out_dir",
        default=str(DATA_ROOT / "paper_ecgfounder_k500val_multicandidate_selector_v6_20260525"),
    )
    args = p.parse_args()

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("--alphas values must be in [0, 1]")

    candidate_specs = list(DEFAULT_CANDIDATE_SPECS if args.use_default_candidates else [])
    candidate_specs.extend(args.candidates)
    candidate_roots = parse_candidate_specs(candidate_specs)
    candidate_names = list(candidate_roots.keys())
    if not candidate_names:
        raise ValueError("no candidates configured")

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
        "method": "ECGFounder multi-candidate selector selected on K500-internal validation",
        "class_names": list(CLASS_NAMES_SUPER5),
        "blend_space": args.blend_space,
        "alphas": args.alphas,
        "candidate_roots": {k: str(v) for k, v in candidate_roots.items()},
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "class_selection_metric": args.class_selection_metric,
        "source_selection_weight": args.source_selection_weight,
        "auroc_selection_weight": args.auroc_selection_weight,
        "source_auroc_selection_weight": args.source_auroc_selection_weight,
        "min_val_auprc_gain": args.min_val_auprc_gain,
        "min_val_auroc_gain": args.min_val_auroc_gain,
        "centers": {},
    }

    for center in args.centers:
        print(f"[center] {center}")
        k500_run = center_k500_run(Path(args.k500_run_root), center)
        head_paths = {"direct": k500_run / "best_head.pt"}
        for name, root in candidate_roots.items():
            head_paths[name] = find_candidate_head(root, center)
        k500_eval_path = k500_run / "eval_result.json"
        for name, path in head_paths.items():
            print(f"  {name}={path}")

        pn_logits = {name: head_logits(path, pn_features) for name, path in head_paths.items()}
        ptbxl_logits = {name: head_logits(path, ptbxl_features) for name, path in head_paths.items()}

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

        global_choice = choose_global_option(
            y_val,
            val_logits,
            candidate_names=candidate_names,
            alphas=args.alphas,
            blend_space=args.blend_space,
        )
        class_choices = choose_classwise_options(
            y_val,
            val_logits,
            source_y_true=ptbxl_labels[fold10],
            source_logits={name: arr[fold10] for name, arr in ptbxl_logits.items()},
            candidate_names=candidate_names,
            alphas=args.alphas,
            blend_space=args.blend_space,
            class_selection_metric=args.class_selection_metric,
            source_selection_weight=args.source_selection_weight,
            auroc_selection_weight=args.auroc_selection_weight,
            source_auroc_selection_weight=args.source_auroc_selection_weight,
            min_val_auprc_gain=args.min_val_auprc_gain,
            min_val_auroc_gain=args.min_val_auroc_gain,
        )

        method_scores = {"direct": sigmoid(pn_logits["direct"])}
        for name in candidate_names:
            method_scores[name] = sigmoid(pn_logits[name])
        method_scores["global_k500val_selector"] = apply_global_option(
            pn_logits, global_choice, blend_space=args.blend_space
        )
        method_scores["classwise_k500val_selector"] = apply_classwise_options(
            pn_logits, class_choices, blend_space=args.blend_space
        )

        target_metrics: dict[str, Any] = {}
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
        direct_target = target_metrics["direct"]

        ptbxl_scores = {"direct": sigmoid(ptbxl_logits["direct"])}
        for name in candidate_names:
            ptbxl_scores[name] = sigmoid(ptbxl_logits[name])
        ptbxl_scores["global_k500val_selector"] = apply_global_option(
            ptbxl_logits, global_choice, blend_space=args.blend_space
        )
        ptbxl_scores["classwise_k500val_selector"] = apply_classwise_options(
            ptbxl_logits, class_choices, blend_space=args.blend_space
        )
        source_metrics = {
            name: metric_for_scores(ptbxl_labels[fold10], scores[fold10])
            for name, scores in ptbxl_scores.items()
        }

        for method_name, target in target_metrics.items():
            source = source_metrics.get(method_name, {})
            rows.append(
                {
                    "center": center,
                    "method": method_name,
                    "target_auroc": target["macro_auroc"],
                    "target_auprc": target["macro_auprc"],
                    "delta_vs_direct_auroc": target["macro_auroc"] - direct_target["macro_auroc"],
                    "delta_vs_direct_auprc": target["macro_auprc"] - direct_target["macro_auprc"],
                    "ptbxl_fold10_auroc": source.get("macro_auroc"),
                    "ptbxl_fold10_auprc": source.get("macro_auprc"),
                    "global_choice": json.dumps(global_choice["best"], sort_keys=True),
                    "classwise_choices": json.dumps(
                        {
                            k: {"candidate": v["candidate"], "alpha": v["alpha"]}
                            for k, v in class_choices.items()
                        },
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
            "global_selection": global_choice,
            "classwise_selection": class_choices,
            "target": target_metrics,
            "ptbxl_fold10": source_metrics,
        }
        reported = target_metrics["classwise_k500val_selector"]
        print(
            "  classwise="
            f"{float(reported['macro_auroc']):.6f}/{float(reported['macro_auprc']):.6f} "
            "delta="
            f"{float(reported['macro_auroc'] - direct_target['macro_auroc']):+.6f}/"
            f"{float(reported['macro_auprc'] - direct_target['macro_auprc']):+.6f}"
        )

    mean_rows: list[dict[str, Any]] = []
    for method_name in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method_name]
        mean_rows.append(
            {
                "center": "MEAN",
                "method": method_name,
                "target_auroc": float(np.mean([float(r["target_auroc"]) for r in subset])),
                "target_auprc": float(np.mean([float(r["target_auprc"]) for r in subset])),
                "delta_vs_direct_auroc": float(
                    np.mean([float(r["delta_vs_direct_auroc"]) for r in subset])
                ),
                "delta_vs_direct_auprc": float(
                    np.mean([float(r["delta_vs_direct_auprc"]) for r in subset])
                ),
                "ptbxl_fold10_auroc": float(
                    np.mean([float(r["ptbxl_fold10_auroc"]) for r in subset])
                ),
                "ptbxl_fold10_auprc": float(
                    np.mean([float(r["ptbxl_fold10_auprc"]) for r in subset])
                ),
                "global_choice": "",
                "classwise_choices": "",
            }
        )
    rows.extend(mean_rows)

    csv_path = out_dir / f"ecgfounder_k500val_multicandidate_{args.blend_space}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"ecgfounder_k500val_multicandidate_{args.blend_space}.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
