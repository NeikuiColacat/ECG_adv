#!/usr/bin/env python3
"""Evaluate EfficientNet direct-FT and VAE-LHAT score blends on PN2021 targets."""

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

from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021CachedCenterDataset,
    _load_excluded_ref_ids,
    _load_or_build_pn2021_center,
)
from scripts.triple_labels.label_schemes import get_scheme, get_super5_pn2021_mapping_metadata  # noqa: E402
from scripts.triple_labels.model_zoo import build_super5_model, normalize_model_name  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_macro_auroc_auprc  # noqa: E402


DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp",
    )
)
CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
ALPHAS = [round(i * 0.05, 2) for i in range(21)]


VAE_RUNS = {
    "ningbo": (
        "paper_effnet_latent_augmix_stage3_seed20260531_v6_20260524/"
        "ningbo_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdminormsttc_ep30_seed20260524"
    ),
    "chapman_shaoxing": (
        "paper_effnet_latent_augmix_stage3_seed20260531_v6_20260524/"
        "chapman_shaoxing_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdminormsttc_ep30_seed20260524"
    ),
    "cpsc_2018": (
        "paper_effnet_latent_augmix_stage3_seed20260531_v6_20260524/"
        "cpsc_2018_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdnormsttc_ep30_seed20260524"
    ),
    "georgia": (
        "paper_effnet_latent_augmix_stage3_seed20260531_v6_20260524/"
        "georgia_realall_targetheavy_M20_lam015_augmix_s2_wlat030_"
        "hs3_target_macro_auprc_cdhypnormsttc_ep30_seed20260524"
    ),
}


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def load_model(model_dir: Path, model_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_super5_model(model_name, num_classes=num_classes).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_pair_logits(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, logits_a, logits_b = [], [], []
    amp = "cuda" in str(device)
    for signals, y in loader:
        signals = signals.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out_a = model_a(signals)
            out_b = model_b(signals)
        labels.append(y.numpy())
        logits_a.append(out_a.float().cpu().numpy())
        logits_b.append(out_b.float().cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits_a), np.concatenate(logits_b)


def target_paths(data_root: Path, center: str) -> dict[str, Path]:
    direct = (
        data_root
        / "paper_direct_finetune_k500_20260516"
        / "runs"
        / f"{center}_K500_direct_ft_ep30_seed20260531_val0.2"
    )
    if center == "ningbo" and not direct.exists():
        direct = (
            data_root
            / "paper_direct_finetune_k500_20260516"
            / "runs"
            / f"{center}_K500_direct_ft_ep1_seed20260531_val0.2"
        )
    anchor = (
        data_root
        / "paper_vae_only_lhat_kcurve_20260518"
        / "subsets"
        / center
        / "k500_seed20260531"
        / f"{center}_real_k500_seed20260531.ref_meta.json"
    )
    return {
        "direct_dir": direct,
        "vae_dir": data_root / VAE_RUNS[center],
        "ref_meta": anchor,
    }


def load_center_dataset(center: str, scheme: dict[str, Any], args: argparse.Namespace):
    ref_ids = _load_excluded_ref_ids([str(target_paths(args.data_root, center)["ref_meta"])])
    eval_args = SimpleNamespace(
        scheme="super5",
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
        crop_len=args.crop_len,
        pn2021_root=str(args.data_root / "physionet2021"),
        pn2021_cache_dir=str(args.data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        pn2021_mmap_cache_dir=str(args.data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        pn2021_limit=None,
        min_pos=args.min_pos,
        _excluded_by_center=ref_ids,
        _included_by_center={},
    )
    center_dir = args.data_root / "physionet2021" / "training" / center
    signals, labels, record_ids, fail, load_time, cache_hit, cache_kind = _load_or_build_pn2021_center(
        center, str(center_dir), scheme, eval_args
    )
    excluded = ref_ids.get(center, set())
    if excluded:
        keep = np.asarray([str(rid) not in excluded for rid in record_ids.astype(str)], dtype=bool)
        signals = signals[keep]
        labels = labels[keep]
        record_ids = record_ids[keep]
    ds = PN2021CachedCenterDataset(signals, labels, crop_len=args.crop_len)
    return ds, {
        "n_records": int(len(ds)),
        "n_scanned": int(len(record_ids) + fail),
        "n_excluded_ref": int(len(excluded)),
        "cache_hit": bool(cache_hit),
        "cache_kind": cache_kind,
        "load_time_s": float(load_time),
    }


def metric_view(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    min_pos: int,
) -> dict[str, Any]:
    m = compute_macro_auroc_auprc(y_true, y_score, class_names, min_pos=min_pos)
    nonzero = (y_true == 1.0).sum(axis=1) > 0
    if int(nonzero.sum()) > 0:
        drop = compute_macro_auroc_auprc(y_true[nonzero], y_score[nonzero], class_names, min_pos=min_pos)
    else:
        drop = {"macro_auroc": float("nan"), "macro_auprc": float("nan"), "per_class": {}}
    return {
        "macro_auroc": m["macro_auroc"],
        "macro_auprc": m["macro_auprc"],
        "n_classes_used": m["n_classes_used"],
        "per_class": m["per_class"],
        "drop_all_zero_macro_auroc": drop["macro_auroc"],
        "drop_all_zero_macro_auprc": drop["macro_auprc"],
        "drop_all_zero_per_class": drop["per_class"],
        "drop_all_zero_n_records": int(nonzero.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=CENTERS)
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    p.add_argument("--data_root", type=Path, default=DATA_ROOT)
    p.add_argument("--out_dir", type=Path, default=DATA_ROOT / "paper_effnet_score_blend_v6_20260524")
    p.add_argument("--model_name", default="efficientnet1dv2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--preprocess_mode", default="minimal_resample")
    p.add_argument("--norm_mode", default="per_sample_global")
    args = p.parse_args()

    args.model_name = normalize_model_name(args.model_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    scheme = get_scheme("super5")
    class_names = list(scheme["class_names"])

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "class_names": class_names,
        "blend_space": args.blend_space,
        "alphas": args.alphas,
        "label_mapping": get_super5_pn2021_mapping_metadata(),
        "centers": {},
    }
    for center in args.centers:
        paths = target_paths(args.data_root, center)
        print(f"[center] {center}")
        print(f"  direct={paths['direct_dir']}")
        print(f"  vae={paths['vae_dir']}")
        ds, meta = load_center_dataset(center, scheme, args)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        direct = load_model(paths["direct_dir"], args.model_name, scheme["num_classes"], device)
        vae = load_model(paths["vae_dir"], args.model_name, scheme["num_classes"], device)
        y_true, direct_logits, vae_logits = infer_pair_logits(direct, vae, loader, device)
        direct_scores = sigmoid(direct_logits)
        vae_scores = sigmoid(vae_logits)
        direct_m = metric_view(y_true, direct_scores, class_names, args.min_pos)
        vae_m = metric_view(y_true, vae_scores, class_names, args.min_pos)
        center_payload = {
            "meta": meta,
            "paths": {k: str(v) for k, v in paths.items()},
            "direct": direct_m,
            "vae": vae_m,
            "blend": [],
        }
        for alpha in args.alphas:
            if args.blend_space == "logit":
                scores = sigmoid((1.0 - alpha) * direct_logits + alpha * vae_logits)
            else:
                scores = (1.0 - alpha) * direct_scores + alpha * vae_scores
            blend_m = metric_view(y_true, scores, class_names, args.min_pos)
            row = {
                "center": center,
                "alpha": alpha,
                "blend_space": args.blend_space,
                "direct_target_auroc": direct_m["macro_auroc"],
                "direct_target_auprc": direct_m["macro_auprc"],
                "vae_target_auroc": vae_m["macro_auroc"],
                "vae_target_auprc": vae_m["macro_auprc"],
                "blend_target_auroc": blend_m["macro_auroc"],
                "blend_target_auprc": blend_m["macro_auprc"],
                "delta_blend_vs_direct_auroc": blend_m["macro_auroc"] - direct_m["macro_auroc"],
                "delta_blend_vs_direct_auprc": blend_m["macro_auprc"] - direct_m["macro_auprc"],
                "direct_model_dir": str(paths["direct_dir"]),
                "vae_model_dir": str(paths["vae_dir"]),
                "ref_meta": str(paths["ref_meta"]),
            }
            rows.append(row)
            center_payload["blend"].append({"alpha": alpha, "metrics": blend_m})
        payload["centers"][center] = center_payload
        best = max(
            [r for r in rows if r["center"] == center],
            key=lambda r: (float(r["blend_target_auroc"]), float(r["blend_target_auprc"])),
        )
        print(
            f"  best alpha={best['alpha']} "
            f"{best['blend_target_auroc']:.6f}/{best['blend_target_auprc']:.6f} "
            f"delta={best['delta_blend_vs_direct_auroc']:+.6f}/"
            f"{best['delta_blend_vs_direct_auprc']:+.6f}"
        )

    csv_path = args.out_dir / f"effnet_direct_vae_score_blend_{args.blend_space}_grid.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / f"effnet_direct_vae_score_blend_{args.blend_space}_grid.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
