#!/usr/bin/env python3
"""K500-internal selection for EfficientNet direct/VAE score blends.

This is the paper-safe counterpart to held-out alpha sweeps: alpha selection is
performed only on the known K=500 target samples via the same deterministic
internal validation split used by the Stage-3 VAE online-AT pilot. PN2021
held-out center data is used only once after selection for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.paper.eval_effnet_score_blend_20260524 import (  # noqa: E402
    DATA_ROOT,
    ALPHAS,
    CENTERS,
    infer_pair_logits,
    load_center_dataset,
    load_model,
    metric_view,
    sigmoid,
    target_paths,
)
from scripts.pgd_cross_center.synth_online_at_super5 import (  # noqa: E402
    build_k500_internal_val_mask,
)
from scripts.triple_labels.eval_crosscenter import PN2021CachedCenterDataset  # noqa: E402
from scripts.triple_labels.label_schemes import get_scheme, get_super5_pn2021_mapping_metadata  # noqa: E402
from scripts.triple_labels.model_zoo import normalize_model_name  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_macro_auroc_auprc  # noqa: E402


K500VAL_VAE_RUNS = {
    "ningbo": (
        "paper_effnet_k500val_latent_augmix_stage3_v6_20260524/"
        "ningbo_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdhypminormsttc_k500val_papersafe_ep30_seed20260524"
    ),
    "cpsc_2018": (
        "paper_effnet_k500val_latent_augmix_stage3_v6_20260524/"
        "cpsc_2018_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdnormsttc_k500val_papersafe_ep30_seed20260524"
    ),
    "chapman_shaoxing": (
        "paper_effnet_k500val_latent_augmix_stage3_v6_20260524/"
        "chapman_shaoxing_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdhypminormsttc_k500val_papersafe_autoM20_ep30_seed20260524"
    ),
    "georgia": (
        "paper_effnet_k500val_latent_augmix_stage3_v6_20260524/"
        "georgia_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdhypnormsttc_k500val_papersafe_autoM20_ep30_seed20260524"
    ),
}


def parse_center_path_map(items: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected CENTER=/path, got {item!r}")
        center, path = item.split("=", 1)
        out[center.strip()] = Path(path.strip())
    return out


def ref_meta_to_signals_npz(ref_meta: Path) -> Path:
    raw = str(ref_meta)
    if not raw.endswith(".ref_meta.json"):
        raise ValueError(f"unexpected ref_meta path: {ref_meta}")
    return Path(raw.removesuffix(".ref_meta.json") + ".signals.npz")


def build_k500_val_dataset(center: str, args: argparse.Namespace) -> tuple[PN2021CachedCenterDataset, dict[str, Any]]:
    ref_meta = target_paths(args.data_root, center)["ref_meta"]
    signal_npz = ref_meta_to_signals_npz(ref_meta)
    with np.load(signal_npz, allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        record_ids = data["record_ids"].astype(str)
    if signals.shape[1:] == (12, 1000):
        signals = signals.transpose(0, 2, 1)
    if signals.shape[1:] != (1000, 12):
        raise ValueError(f"expected K500 signals (N,1000,12), got {signals.shape}")
    val_mask = build_k500_internal_val_mask(
        labels,
        val_fraction=args.target_real_val_fraction,
        seed=args.target_real_val_seed,
    )
    ds = PN2021CachedCenterDataset(signals[val_mask], labels[val_mask], crop_len=args.crop_len)
    return ds, {
        "signal_npz": str(signal_npz),
        "ref_meta": str(ref_meta),
        "n_k500": int(len(labels)),
        "n_val": int(val_mask.sum()),
        "val_record_ids": record_ids[val_mask].tolist(),
        "val_positive_counts": labels[val_mask].sum(axis=0).astype(int).tolist(),
    }


def choose_global_alpha(
    y_true: np.ndarray,
    direct_scores: np.ndarray,
    vae_scores: np.ndarray,
    class_names: list[str],
    alphas: list[float],
    min_pos: int,
) -> dict[str, Any]:
    rows = []
    for alpha in alphas:
        scores = (1.0 - alpha) * direct_scores + alpha * vae_scores
        metrics = metric_view(y_true, scores, class_names, min_pos)
        rows.append({"alpha": alpha, "metrics": metrics})
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
    direct_scores: np.ndarray,
    vae_scores: np.ndarray,
    class_names: list[str],
    alphas: list[float],
    min_pos: int,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}
    for idx, cls in enumerate(class_names):
        y = y_true[:, idx : idx + 1]
        n_pos = int((y[:, 0] > 0.5).sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < min_pos or n_neg < 1:
            choices[cls] = {
                "alpha": 0.0,
                "n_pos": n_pos,
                "selection": "direct_fallback_insufficient_val_positives",
            }
            continue
        best: dict[str, Any] | None = None
        for alpha in alphas:
            scores = (1.0 - alpha) * direct_scores[:, idx : idx + 1] + alpha * vae_scores[:, idx : idx + 1]
            metrics = compute_macro_auroc_auprc(y, scores, [cls], min_pos=min_pos)
            row = {
                "alpha": float(alpha),
                "auroc": metrics["per_class"][cls]["auroc"],
                "auprc": metrics["per_class"][cls]["auprc"],
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


def apply_classwise_alphas(
    direct_scores: np.ndarray,
    vae_scores: np.ndarray,
    class_names: list[str],
    choices: dict[str, Any],
) -> np.ndarray:
    out = direct_scores.copy()
    for idx, cls in enumerate(class_names):
        alpha = float(choices[cls]["alpha"])
        out[:, idx] = (1.0 - alpha) * direct_scores[:, idx] + alpha * vae_scores[:, idx]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=CENTERS)
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--data_root", type=Path, default=DATA_ROOT)
    p.add_argument("--out_dir", type=Path, default=DATA_ROOT / "paper_effnet_k500val_classwise_blend_v6_20260524")
    p.add_argument("--vae_run_dirs", nargs="*", default=[])
    p.add_argument("--model_name", default="efficientnet1dv2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--preprocess_mode", default="minimal_resample")
    p.add_argument("--norm_mode", default="per_sample_global")
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    args = p.parse_args()

    args.model_name = normalize_model_name(args.model_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    scheme = get_scheme("super5")
    class_names = list(scheme["class_names"])
    override_vae_dirs = parse_center_path_map(args.vae_run_dirs)

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "method": "EfficientNet direct/VAE score fusion selected on K500-internal validation",
        "class_names": class_names,
        "alphas": args.alphas,
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "label_mapping": get_super5_pn2021_mapping_metadata(),
        "centers": {},
    }
    for center in args.centers:
        paths = target_paths(args.data_root, center)
        vae_dir = override_vae_dirs.get(center, args.data_root / K500VAL_VAE_RUNS[center])
        if not (vae_dir / "best_model.pt").exists():
            print(f"[skip] {center}: missing VAE best_model.pt under {vae_dir}")
            continue
        print(f"[center] {center}")
        print(f"  direct={paths['direct_dir']}")
        print(f"  vae={vae_dir}")

        direct = load_model(paths["direct_dir"], args.model_name, scheme["num_classes"], device)
        vae = load_model(vae_dir, args.model_name, scheme["num_classes"], device)

        val_ds, val_meta = build_k500_val_dataset(center, args)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        y_val, direct_val_logits, vae_val_logits = infer_pair_logits(direct, vae, val_loader, device)
        direct_val = sigmoid(direct_val_logits)
        vae_val = sigmoid(vae_val_logits)

        global_choice = choose_global_alpha(y_val, direct_val, vae_val, class_names, args.alphas, args.min_pos)
        class_choices = choose_classwise_alphas(y_val, direct_val, vae_val, class_names, args.alphas, args.min_pos)

        heldout_ds, heldout_meta = load_center_dataset(center, scheme, args)
        heldout_loader = DataLoader(
            heldout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        y_target, direct_logits, vae_logits = infer_pair_logits(direct, vae, heldout_loader, device)
        direct_scores = sigmoid(direct_logits)
        vae_scores = sigmoid(vae_logits)
        global_alpha = float(global_choice["best_alpha"])
        global_scores = (1.0 - global_alpha) * direct_scores + global_alpha * vae_scores
        classwise_scores = apply_classwise_alphas(direct_scores, vae_scores, class_names, class_choices)

        metrics = {
            "direct": metric_view(y_target, direct_scores, class_names, args.min_pos),
            "vae": metric_view(y_target, vae_scores, class_names, args.min_pos),
            "global_k500val_blend": metric_view(y_target, global_scores, class_names, args.min_pos),
            "classwise_k500val_blend": metric_view(y_target, classwise_scores, class_names, args.min_pos),
        }
        direct_m = metrics["direct"]
        for method_name, method_metrics in metrics.items():
            rows.append(
                {
                    "center": center,
                    "method": method_name,
                    "target_auroc": method_metrics["macro_auroc"],
                    "target_auprc": method_metrics["macro_auprc"],
                    "delta_vs_direct_auroc": method_metrics["macro_auroc"] - direct_m["macro_auroc"],
                    "delta_vs_direct_auprc": method_metrics["macro_auprc"] - direct_m["macro_auprc"],
                    "global_alpha_selected": global_alpha,
                    "classwise_alphas": json.dumps({k: v["alpha"] for k, v in class_choices.items()}, sort_keys=True),
                    "direct_model_dir": str(paths["direct_dir"]),
                    "vae_model_dir": str(vae_dir),
                }
            )
        payload["centers"][center] = {
            "paths": {
                "direct_dir": str(paths["direct_dir"]),
                "vae_dir": str(vae_dir),
                "ref_meta": str(paths["ref_meta"]),
            },
            "k500_val": val_meta,
            "heldout": heldout_meta,
            "global_alpha_selection": global_choice,
            "classwise_alpha_selection": class_choices,
            "heldout_metrics": metrics,
        }
        best_row = max(
            [r for r in rows if r["center"] == center and r["method"].endswith("blend")],
            key=lambda r: (float(r["target_auprc"]), float(r["target_auroc"])),
        )
        print(
            f"  selected global alpha={global_alpha:.2f}; "
            f"best reported {best_row['method']} "
            f"{float(best_row['target_auroc']):.6f}/{float(best_row['target_auprc']):.6f} "
            f"delta={float(best_row['delta_vs_direct_auroc']):+.6f}/"
            f"{float(best_row['delta_vs_direct_auprc']):+.6f}"
        )

    if not rows:
        raise RuntimeError("no centers evaluated")
    csv_path = args.out_dir / "effnet_k500val_selected_blends.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / "effnet_k500val_selected_blends.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
