#!/usr/bin/env python3
"""Run a lightweight TENT-style target-unlabeled baseline for Super5 PN2021.

Reference implementation cloned at /root/autodl-tmp/external_repos/tent.
This script adapts the idea to this repo's 1D multi-label ECG classifier:

* source model: PTB-XL EfficientNet1DV2 Super5 checkpoint
* update params: BatchNorm1d affine weights/biases only
* objective: Bernoulli entropy of sigmoid logits, no target labels
* protocol: adapt on one target center held-out stream after excluding K=500 refs
* metrics: same PN2021 v3 Super5 labels and macro AUROC/AUPRC as eval_crosscenter
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021CachedCenterDataset,
    compute_macro_auroc_auprc,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


PN2021_CENTERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
]
TARGET_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]


def load_model(ckpt: str, device: torch.device) -> nn.Module:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def configure_tent(model: nn.Module) -> list[nn.Parameter]:
    """Configure model for TENT on 1D ECG: update BN affine params only."""
    model.train()
    model.requires_grad_(False)
    params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    if not params:
        raise RuntimeError("TENT requires BatchNorm1d affine parameters, found none")
    return params


def bernoulli_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    eps = 1e-6
    ent = -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))
    return ent.mean()


def cache_dir_for(center: str, mmap_root: Path) -> Path:
    return mmap_root / f"super5_{center}_100hz1000_v3_super5_normsuppress"


def load_center(center: str, mmap_root: Path):
    root = cache_dir_for(center, mmap_root)
    signals = np.load(root / "signals.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    record_ids = np.load(root / "record_ids.npy", allow_pickle=True).astype(str)
    return signals, labels, record_ids


def load_ref_ids(ref_root: Path, centers: list[str]) -> dict[str, set[str]]:
    out = {}
    for center in centers:
        p = ref_root / center / "k500_seed20260531" / f"{center}_real_k500_seed20260531.ref_meta.json"
        with open(p) as f:
            out[center] = set(json.load(f)["ref_record_ids"])
    return out


def make_loader(signals, labels, batch_size: int, crop_len: int, num_workers: int, shuffle: bool = False):
    ds = PN2021CachedCenterDataset(signals, labels, crop_len=crop_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


@torch.enable_grad()
def adapt_tent(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, amp: bool):
    all_labels, all_logits = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(signals)
            loss = bernoulli_entropy_from_logits(logits)
        loss.backward()
        optimizer.step()
        all_labels.append(labels.numpy())
        all_logits.append(logits.detach().float().cpu().numpy())
    return np.concatenate(all_labels), sigmoid_np(np.concatenate(all_logits))


@torch.no_grad()
def infer_no_adapt(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    all_labels, all_logits = [], []
    model.eval()
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(signals)
        all_labels.append(labels.numpy())
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_labels), sigmoid_np(np.concatenate(all_logits))


def filter_refs(signals, labels, record_ids, ref_ids: set[str]):
    keep = np.asarray([rid not in ref_ids for rid in record_ids], dtype=bool)
    return signals[keep], labels[keep], record_ids[keep], int((~keep).sum())


def run_one_target(args, target: str, ref_ids_by_center: dict[str, set[str]], device: torch.device):
    model = load_model(args.ckpt, device)
    params = configure_tent(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    signals, labels, record_ids = load_center(target, Path(args.mmap_root))
    signals, labels, record_ids, n_excluded = filter_refs(signals, labels, record_ids, ref_ids_by_center[target])
    adapt_loader = make_loader(
        signals, labels, args.batch_size, args.crop_len, args.num_workers,
        shuffle=args.shuffle,
    )
    y_true_target, y_score_target = adapt_tent(
        model, adapt_loader, optimizer, device, amp=device.type == "cuda" and not args.no_amp,
    )
    target_metric = compute_macro_auroc_auprc(
        y_true_target, y_score_target, CLASS_NAMES_SUPER5, min_pos=args.min_pos
    )

    per_center = {
        target: {
            "n_records": int(len(labels)),
            "n_excluded_ref": n_excluded,
            "effective_n": int(len(labels)),
            "macro_auroc": target_metric["macro_auroc"],
            "macro_auprc": target_metric["macro_auprc"],
            "n_classes_used": target_metric["n_classes_used"],
            "per_class": target_metric["per_class"],
            "protocol": "online_tent_predictions_during_adaptation",
        }
    }

    avg_aurocs = [target_metric["macro_auroc"]]
    avg_auprcs = [target_metric["macro_auprc"]]
    for center in PN2021_CENTERS:
        if center == target:
            continue
        sig_c, lab_c, rid_c = load_center(center, Path(args.mmap_root))
        loader = make_loader(sig_c, lab_c, args.eval_batch_size, args.crop_len, args.num_workers, shuffle=False)
        y_true, y_score = infer_no_adapt(model, loader, device, amp=device.type == "cuda" and not args.no_amp)
        metric = compute_macro_auroc_auprc(y_true, y_score, CLASS_NAMES_SUPER5, min_pos=args.min_pos)
        per_center[center] = {
            "n_records": int(len(lab_c)),
            "n_excluded_ref": 0,
            "effective_n": int(len(lab_c)),
            "macro_auroc": metric["macro_auroc"],
            "macro_auprc": metric["macro_auprc"],
            "n_classes_used": metric["n_classes_used"],
            "per_class": metric["per_class"],
            "protocol": "post_tent_no_further_adaptation",
        }
        avg_aurocs.append(metric["macro_auroc"])
        avg_auprcs.append(metric["macro_auprc"])

    return {
        "target_center": target,
        "tent_lr": args.lr,
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle),
        "avg_macro_auroc": float(np.mean(avg_aurocs)),
        "avg_macro_auprc": float(np.mean(avg_auprcs)),
        "per_center": per_center,
    }


def write_outputs(out_dir: Path, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tent_super5_ref_excluded.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    csv_path = out_dir / "tent_super5_ref_excluded.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_center_view",
                "center",
                "macro_auroc",
                "macro_auprc",
                "n_records",
                "n_excluded_ref",
                "protocol",
            ],
        )
        writer.writeheader()
        for target, view in result["views"].items():
            for center, row in view["per_center"].items():
                writer.writerow({
                    "target_center_view": target,
                    "center": center,
                    "macro_auroc": row["macro_auroc"],
                    "macro_auprc": row["macro_auprc"],
                    "n_records": row["n_records"],
                    "n_excluded_ref": row["n_excluded_ref"],
                    "protocol": row["protocol"],
                })
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt")
    ap.add_argument("--mmap_root", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal")
    ap.add_argument("--ref_root", default="/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/paper_tta_baselines_20260517/tent_lr1e-4")
    ap.add_argument("--centers", nargs="+", default=TARGET_CENTERS)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--crop_len", type=int, default=1000)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--shuffle", action="store_true", help="shuffle target stream before online adaptation")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ref_ids = load_ref_ids(Path(args.ref_root), TARGET_CENTERS)
    result = {
        "method": "TENT_BN1d_BernoulliEntropy",
        "reference_repo": "/root/autodl-tmp/external_repos/tent",
        "source_checkpoint": args.ckpt,
        "views": {},
    }
    for target in args.centers:
        print(f"[target] {target}")
        view = run_one_target(args, target, ref_ids, device)
        result["views"][target] = view
        row = view["per_center"][target]
        print(
            f"  target {target}: AUROC={row['macro_auroc']:.4f} "
            f"AUPRC={row['macro_auprc']:.4f} avg={view['avg_macro_auroc']:.4f}/{view['avg_macro_auprc']:.4f}"
        )
    write_outputs(Path(args.out_dir), result)


if __name__ == "__main__":
    main()
