#!/usr/bin/env python3
"""Evaluate fixed ECGFounder score blends for K-shot and VAE-online heads.

This is intentionally a lightweight post-hoc diagnostic: it reuses cached
ECGFounder features and saved heads, then evaluates whether score/logit blending
can recover target-center AUROC without launching another training job.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

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
    DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_linear_probe_v6_from_legacy_cache"
)
DEFAULT_K500_RUN_ROOT = (
    DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_kshot_head_ft_v6_from_legacy_cache/runs"
)
DEFAULT_VAE_RUN_ROOT = (
    DATA_ROOT / "paper_ecgfounder_vae_only_lhat_resadapter_targetheavy_v6_20260524/runs"
)
DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_score_blend_v6_20260524"
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_ALPHAS = [round(x * 0.05, 2) for x in range(0, 21)]


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def linear_logits_from_state(state: dict[str, torch.Tensor], features: np.ndarray) -> np.ndarray:
    weight = state["weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["bias"].detach().cpu().numpy().astype(np.float32)
    return features.astype(np.float32) @ weight.T + bias


def residual_adapter_logits_from_state(
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    *,
    scale: float = 1.0,
    batch_size: int = 8192,
) -> np.ndarray:
    """CPU forward for ResidualAdapterHead saved by the VAE-LHAT runner."""
    base_w = state["base_head.weight"].detach().cpu()
    base_b = state["base_head.bias"].detach().cpu()
    ln_w = state["adapter.0.weight"].detach().cpu()
    ln_b = state["adapter.0.bias"].detach().cpu()
    h_w = state["adapter.1.weight"].detach().cpu()
    h_b = state["adapter.1.bias"].detach().cpu()
    out_w = state["adapter.4.weight"].detach().cpu()
    out_b = state["adapter.4.bias"].detach().cpu()

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            x = torch.from_numpy(features[start : start + int(batch_size)].astype(np.float32))
            base = torch.nn.functional.linear(x, base_w, base_b)
            z = torch.nn.functional.layer_norm(x, (x.shape[1],), ln_w, ln_b, eps=1e-5)
            z = torch.nn.functional.linear(z, h_w, h_b)
            z = torch.nn.functional.gelu(z)
            z = torch.nn.functional.linear(z, out_w, out_b)
            logits.append((base + float(scale) * z).cpu().numpy().astype(np.float32))
    return np.concatenate(logits, axis=0)


def feature_adapter_logits_from_state(
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    *,
    scale: float = 1.0,
    batch_size: int = 8192,
) -> np.ndarray:
    """CPU forward for FeatureAdapterHead saved by the VAE-LHAT runner."""
    base_w = state["base_head.weight"].detach().cpu()
    base_b = state["base_head.bias"].detach().cpu()
    ln_w = state["adapter.0.weight"].detach().cpu()
    ln_b = state["adapter.0.bias"].detach().cpu()
    h_w = state["adapter.1.weight"].detach().cpu()
    h_b = state["adapter.1.bias"].detach().cpu()
    out_w = state["adapter.4.weight"].detach().cpu()
    out_b = state["adapter.4.bias"].detach().cpu()

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            x = torch.from_numpy(features[start : start + int(batch_size)].astype(np.float32))
            z = torch.nn.functional.layer_norm(x, (x.shape[1],), ln_w, ln_b, eps=1e-5)
            z = torch.nn.functional.linear(z, h_w, h_b)
            z = torch.nn.functional.gelu(z)
            z = torch.nn.functional.linear(z, out_w, out_b)
            x_adapted = x + float(scale) * z
            logits.append(torch.nn.functional.linear(x_adapted, base_w, base_b).cpu().numpy().astype(np.float32))
    return np.concatenate(logits, axis=0)


def head_logits(head_path: Path, features: np.ndarray, *, adapter_scale: float = 1.0) -> np.ndarray:
    state = torch.load(head_path, map_location="cpu")
    if "weight" in state and "bias" in state:
        return linear_logits_from_state(state, features)
    if "base_head.weight" in state and "adapter.4.bias" in state:
        if int(state["adapter.4.bias"].numel()) != len(CLASS_NAMES_SUPER5):
            return feature_adapter_logits_from_state(state, features, scale=adapter_scale)
        return residual_adapter_logits_from_state(state, features, scale=adapter_scale)
    keys = ", ".join(sorted(state.keys())[:12])
    raise ValueError(f"unsupported head state in {head_path}; first keys: {keys}")


def head_scores(head_path: Path, features: np.ndarray, *, adapter_scale: float = 1.0) -> np.ndarray:
    return sigmoid(head_logits(head_path, features, adapter_scale=adapter_scale))


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
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--blend_space", choices=["prob", "logit"], default="logit")
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k500_run_root", default=str(DEFAULT_K500_RUN_ROOT))
    p.add_argument("--vae_run_root", default=str(DEFAULT_VAE_RUN_ROOT))
    p.add_argument("--k500_head_path", default="")
    p.add_argument("--k500_eval_path", default="")
    p.add_argument("--vae_head_path", default="")
    p.add_argument("--vae_adapter_scale", type=float, default=1.0)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args()


def center_k500_run(run_root: Path, center: str) -> Path:
    return run_root / f"{center}_K500_fromK500_headft_ep50_seed20260531"


def center_vae_run(run_root: Path, center: str) -> Path:
    matches = sorted(run_root.glob(f"{center}_K500_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one VAE run for {center} under {run_root}, found {len(matches)}")
    return matches[0]


def build_ref_ids(k500_eval_path: Path, center: str, all_centers: np.ndarray) -> dict[str, set[str]]:
    return load_ref_ids(k500_eval_path, center, all_centers)


def blend_arrays(
    k500_logits: np.ndarray,
    vae_logits: np.ndarray,
    *,
    alpha: float,
    blend_space: str,
) -> np.ndarray:
    if blend_space == "logit":
        return sigmoid((1.0 - alpha) * k500_logits + alpha * vae_logits)
    k500_scores = sigmoid(k500_logits)
    vae_scores = sigmoid(vae_logits)
    return (1.0 - alpha) * k500_scores + alpha * vae_scores


def mean_row(rows: Iterable[dict[str, Any]], prefix: str) -> dict[str, Any]:
    rows = list(rows)
    return {
        f"{prefix}_mean_auroc": float(np.mean([float(r[f"{prefix}_target_auroc"]) for r in rows])),
        f"{prefix}_mean_auprc": float(np.mean([float(r[f"{prefix}_target_auprc"]) for r in rows])),
    }


def main() -> None:
    args = parse_args()
    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("--alphas values must be in [0, 1]")

    linear_dir = Path(args.linear_probe_dir)
    ptbxl_path = linear_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz"
    pn_path = linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz"
    ptbxl = np.load(ptbxl_path, allow_pickle=True)
    pn = np.load(pn_path, allow_pickle=True)

    pn_labels = pn["labels"].astype(np.float32)
    pn_centers = pn["centers"].astype(str)
    pn_record_ids = pn["record_ids"].astype(str)
    fold10 = ptbxl["folds"].astype(np.int64) == 10
    ptbxl_labels = ptbxl["labels"].astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for center in args.centers:
        k500_run = center_k500_run(Path(args.k500_run_root), center)
        vae_run = center_vae_run(Path(args.vae_run_root), center)
        k500_head_path = Path(args.k500_head_path) if args.k500_head_path else k500_run / "best_head.pt"
        k500_eval_path = Path(args.k500_eval_path) if args.k500_eval_path else k500_run / "eval_result.json"
        vae_head_path = Path(args.vae_head_path) if args.vae_head_path else vae_run / "best_head.pt"

        k500_pn_logits = head_logits(k500_head_path, pn["features"])
        vae_pn_logits = head_logits(vae_head_path, pn["features"], adapter_scale=args.vae_adapter_scale)
        k500_ptbxl_logits = head_logits(k500_head_path, ptbxl["features"])
        vae_ptbxl_logits = head_logits(
            vae_head_path, ptbxl["features"], adapter_scale=args.vae_adapter_scale
        )
        k500_pn = sigmoid(k500_pn_logits)
        vae_pn = sigmoid(vae_pn_logits)
        k500_ptbxl = sigmoid(k500_ptbxl_logits)
        vae_ptbxl = sigmoid(vae_ptbxl_logits)

        ref_ids = build_ref_ids(k500_eval_path, center, pn["centers"])
        k500_views = evaluate_pn2021_views(
            pn_labels, k500_pn, pn_centers, pn_record_ids, ref_ids, report_drop_all_zero=True
        )
        vae_views = evaluate_pn2021_views(
            pn_labels, vae_pn, pn_centers, pn_record_ids, ref_ids, report_drop_all_zero=True
        )
        k500_source = compute_metrics(ptbxl_labels[fold10], k500_ptbxl[fold10], min_pos=1)
        vae_source = compute_metrics(ptbxl_labels[fold10], vae_ptbxl[fold10], min_pos=1)
        k500_target = summarize_target(k500_views, center)
        vae_target = summarize_target(vae_views, center)

        for alpha in args.alphas:
            blend_pn = blend_arrays(
                k500_pn_logits, vae_pn_logits, alpha=alpha, blend_space=args.blend_space
            )
            blend_ptbxl = blend_arrays(
                k500_ptbxl_logits, vae_ptbxl_logits, alpha=alpha, blend_space=args.blend_space
            )
            blend_views = evaluate_pn2021_views(
                pn_labels,
                blend_pn,
                pn_centers,
                pn_record_ids,
                ref_ids,
                report_drop_all_zero=True,
            )
            blend_source = compute_metrics(ptbxl_labels[fold10], blend_ptbxl[fold10], min_pos=1)
            blend_target = summarize_target(blend_views, center)
            row = {
                "center": center,
                "alpha": alpha,
                "blend_space": args.blend_space,
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
                "k500_head_path": str(k500_head_path),
                "vae_head_path": str(vae_head_path),
                "k500_eval_path": str(k500_eval_path),
            }
            rows.append(row)
            payloads.append(
                {
                    "method": "ECGFounder fixed score blend: (1-alpha) K500 direct + alpha VAE-online AT",
                    "class_names": list(CLASS_NAMES_SUPER5),
                    "center": center,
                    "alpha": alpha,
                    "blend_space": args.blend_space,
                    "linear_probe_dir": str(linear_dir),
                    "target": {
                        "k500": k500_target,
                        "vae": vae_target,
                        "blend": blend_target,
                        "blend_minus_k500": {
                            "macro_auroc": row["delta_blend_vs_k500_auroc"],
                            "macro_auprc": row["delta_blend_vs_k500_auprc"],
                        },
                    },
                    "ptbxl_fold10": {
                        "k500": k500_source,
                        "vae": vae_source,
                        "blend": blend_source,
                    },
                }
            )

    csv_path = out_dir / f"ecgfounder_k500_vae_score_blend_{args.blend_space}_grid.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center",
                "alpha",
                "blend_space",
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
                "k500_head_path",
                "vae_head_path",
                "k500_eval_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"ecgfounder_k500_vae_score_blend_{args.blend_space}_grid.json"
    print(f"[summary] wrote {csv_path}")
    with json_path.open("w") as f:
        json.dump(
            {
                "class_names": list(CLASS_NAMES_SUPER5),
                "linear_probe_dir": str(linear_dir),
                "blend_space": args.blend_space,
                "centers": args.centers,
                "alphas": args.alphas,
                "payloads": payloads,
            },
            f,
            indent=2,
        )
    print(f"[summary] wrote {json_path}")
    best = max(rows, key=lambda r: (float(r["blend_target_auroc"]), float(r["blend_target_auprc"])))
    print(
        "[best-center-alpha] "
        f"{best['center']} alpha={best['alpha']} "
        f"blend={float(best['blend_target_auroc']):.6f}/{float(best['blend_target_auprc']):.6f} "
        f"delta={float(best['delta_blend_vs_k500_auroc']):+.6f}/"
        f"{float(best['delta_blend_vs_k500_auprc']):+.6f}"
    )


if __name__ == "__main__":
    main()
