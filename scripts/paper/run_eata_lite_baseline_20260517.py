#!/usr/bin/env python3
"""EATA-lite target test-time adaptation baseline for Super5 PN2021.

The official EATA implementation is cloned at
/root/autodl-tmp/external_repos/EATA.  The original code is written for
softmax multi-class image classifiers with BatchNorm2d, so this script adapts
the core idea to this repo's multi-label 1D ECG setting:

* update BatchNorm1d affine weights/biases only;
* minimize Bernoulli entropy on reliable low-entropy target samples;
* keep a light anti-forgetting L2 anchor to the source BN parameters;
* predict each target batch before applying the online update, as in TTA.

This is a deployment-style baseline, not a K-shot labeled adaptation method.
The target center stream excludes the K=500 ref ids for comparability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.run_tent_baseline_20260517 import (  # noqa: E402
    PN2021_CENTERS,
    TARGET_CENTERS,
    configure_tent,
    filter_refs,
    infer_no_adapt,
    load_center,
    load_model,
    load_ref_ids,
    make_loader,
    sigmoid_np,
)
from scripts.triple_labels.eval_crosscenter import compute_macro_auroc_auprc  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def bernoulli_entropy_per_sample(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    eps = 1e-6
    ent = -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))
    return ent.mean(dim=1)


def collect_param_anchors(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    anchors = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            anchors.append((name, param, param.detach().clone()))
    if not anchors:
        raise RuntimeError("EATA-lite found no trainable parameters")
    return anchors


@torch.enable_grad()
def adapt_eata_lite(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    anchors: list[tuple[str, torch.nn.Parameter, torch.Tensor]],
    device: torch.device,
    amp: bool,
    e_margin: float,
    d_margin: float,
    anchor_l2_alpha: float,
    grad_clip: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    all_labels, all_logits = [], []
    current_probs = None
    n_seen = 0
    n_reliable = 0
    n_updated = 0
    loss_values = []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(signals)
            entropy = bernoulli_entropy_per_sample(logits)

        probs = torch.sigmoid(logits.detach())
        reliable_mask = entropy.detach() < e_margin
        selected_mask = reliable_mask.clone()
        if current_probs is not None and bool(reliable_mask.any()):
            sims = F.cosine_similarity(probs[reliable_mask], current_probs.unsqueeze(0), dim=1)
            selected_mask[reliable_mask.clone()] = sims < d_margin

        selected_idx = torch.where(selected_mask)[0]
        n_seen += int(signals.shape[0])
        n_reliable += int(reliable_mask.sum().item())
        n_updated += int(selected_idx.numel())

        if selected_idx.numel() > 0:
            selected_entropy = entropy[selected_idx]
            coeff = 1.0 / torch.exp(selected_entropy.detach() - e_margin)
            loss = (selected_entropy * coeff).mean()
            if anchor_l2_alpha > 0:
                l2 = torch.zeros((), dtype=torch.float32, device=device)
                for _, param, anchor in anchors:
                    l2 = l2 + (param.float() - anchor.float()).pow(2).sum()
                loss = loss + float(anchor_l2_alpha) * l2
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for _, p, _ in anchors], grad_clip)
            optimizer.step()
            loss_values.append(float(loss.detach().item()))

        with torch.no_grad():
            if selected_idx.numel() > 0:
                selected_probs = probs[selected_idx].mean(dim=0)
                current_probs = selected_probs if current_probs is None else 0.9 * current_probs + 0.1 * selected_probs

        all_labels.append(labels.numpy())
        all_logits.append(logits.detach().float().cpu().numpy())

    stats = {
        "n_seen": n_seen,
        "n_reliable_entropy": n_reliable,
        "n_updated_after_diversity": n_updated,
        "update_fraction": float(n_updated / max(n_seen, 1)),
        "mean_update_loss": float(np.mean(loss_values)) if loss_values else float("nan"),
    }
    return np.concatenate(all_labels), sigmoid_np(np.concatenate(all_logits)), stats


def run_one_target(args, target: str, ref_ids_by_center: dict[str, set[str]], device: torch.device):
    model = load_model(args.ckpt, device)
    params = configure_tent(model)
    anchors = collect_param_anchors(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    signals, labels, record_ids = load_center(target, Path(args.mmap_root))
    signals, labels, record_ids, n_excluded = filter_refs(signals, labels, record_ids, ref_ids_by_center[target])
    adapt_loader = make_loader(
        signals,
        labels,
        args.batch_size,
        args.crop_len,
        args.num_workers,
        shuffle=args.shuffle,
    )
    y_true_target, y_score_target, stats = adapt_eata_lite(
        model,
        adapt_loader,
        optimizer,
        anchors,
        device,
        amp=device.type == "cuda" and not args.no_amp,
        e_margin=args.e_margin,
        d_margin=args.d_margin,
        anchor_l2_alpha=args.anchor_l2_alpha,
        grad_clip=args.grad_clip,
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
            "protocol": "online_eata_lite_predictions_during_adaptation",
            "adaptation_stats": stats,
        }
    }

    avg_aurocs = [target_metric["macro_auroc"]]
    avg_auprcs = [target_metric["macro_auprc"]]
    for center in PN2021_CENTERS:
        if center == target:
            continue
        sig_c, lab_c, _ = load_center(center, Path(args.mmap_root))
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
            "protocol": "post_eata_lite_no_further_adaptation",
        }
        avg_aurocs.append(metric["macro_auroc"])
        avg_auprcs.append(metric["macro_auprc"])

    return {
        "target_center": target,
        "lr": args.lr,
        "e_margin": args.e_margin,
        "d_margin": args.d_margin,
        "anchor_l2_alpha": args.anchor_l2_alpha,
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle),
        "avg_macro_auroc": float(np.mean(avg_aurocs)),
        "avg_macro_auprc": float(np.mean(avg_auprcs)),
        "per_center": per_center,
    }


def write_outputs(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eata_lite_super5_ref_excluded.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    csv_path = out_dir / "eata_lite_super5_ref_excluded.csv"
    with csv_path.open("w", newline="") as f:
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
                "n_seen",
                "n_reliable_entropy",
                "n_updated_after_diversity",
                "update_fraction",
            ],
        )
        writer.writeheader()
        for target, view in result["views"].items():
            for center, row in view["per_center"].items():
                stats = row.get("adaptation_stats", {})
                writer.writerow({
                    "target_center_view": target,
                    "center": center,
                    "macro_auroc": row["macro_auroc"],
                    "macro_auprc": row["macro_auprc"],
                    "n_records": row["n_records"],
                    "n_excluded_ref": row["n_excluded_ref"],
                    "protocol": row["protocol"],
                    "n_seen": stats.get("n_seen", ""),
                    "n_reliable_entropy": stats.get("n_reliable_entropy", ""),
                    "n_updated_after_diversity": stats.get("n_updated_after_diversity", ""),
                    "update_fraction": stats.get("update_fraction", ""),
                })
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt")
    ap.add_argument("--mmap_root", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal")
    ap.add_argument("--ref_root", default="/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/paper_tta_baselines_20260517/eata_lite_lr1e-5")
    ap.add_argument("--centers", nargs="+", default=TARGET_CENTERS)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--e_margin", type=float, default=0.55)
    ap.add_argument("--d_margin", type=float, default=0.995)
    ap.add_argument("--anchor_l2_alpha", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--crop_len", type=int, default=1000)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ref_ids = load_ref_ids(Path(args.ref_root), TARGET_CENTERS)
    result = {
        "method": "EATA_lite_BN1d_BernoulliEntropy",
        "reference_repo": "/root/autodl-tmp/external_repos/EATA",
        "source_checkpoint": args.ckpt,
        "views": {},
    }
    for target in args.centers:
        print(f"[target] {target}")
        view = run_one_target(args, target, ref_ids, device)
        result["views"][target] = view
        row = view["per_center"][target]
        stats = row.get("adaptation_stats", {})
        print(
            f"  target {target}: AUROC={row['macro_auroc']:.4f} "
            f"AUPRC={row['macro_auprc']:.4f} avg={view['avg_macro_auroc']:.4f}/{view['avg_macro_auprc']:.4f} "
            f"updated={stats.get('update_fraction', float('nan')):.3f}"
        )
    write_outputs(Path(args.out_dir), result)


if __name__ == "__main__":
    main()
