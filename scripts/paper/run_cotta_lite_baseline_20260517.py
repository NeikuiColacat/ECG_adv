#!/usr/bin/env python3
"""CoTTA-lite target test-time adaptation baseline for Super5 PN2021.

The official CoTTA implementation is cloned at
/root/autodl-tmp/external_repos/cotta.  The original code targets image
classifiers, so this script keeps only the core CoTTA mechanics and adapts them
to this repo's 1D multi-label ECG setting:

* source model: PTB-XL EfficientNet1DV2 Super5 checkpoint;
* update params: BatchNorm1d affine weights/biases only, matching TENT/EATA;
* teacher: EMA copy of the adapting student;
* objective: multi-label consistency between student logits and teacher
  sigmoid probabilities;
* augmentation: light ECG amplitude/noise test-time augmentation for teacher
  averaging;
* stochastic restore: randomly restore trainable parameters to source values.

This is a deployment-style TTA baseline, not a target-labeled K-shot method.
The target stream excludes the same K=500 ref ids for comparability.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
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


def clone_teacher(model: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.detach_()
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, alpha: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(t_buf):
            t_buf.data.mul_(alpha).add_(s_buf.data, alpha=1.0 - alpha)
        else:
            t_buf.data.copy_(s_buf.data)


def augment_ecg(x: torch.Tensor, noise_std: float, scale_std: float) -> torch.Tensor:
    out = x
    if scale_std > 0:
        scale = torch.randn(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * scale_std + 1.0
        out = out * scale
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def teacher_logits(
    teacher: torch.nn.Module,
    anchor: torch.nn.Module,
    x: torch.Tensor,
    args: argparse.Namespace,
    amp: bool,
) -> tuple[torch.Tensor, dict]:
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        anchor_logits = anchor(x)
        anchor_prob = torch.sigmoid(anchor_logits)
        anchor_conf = torch.maximum(anchor_prob, 1.0 - anchor_prob).mean()
        standard = teacher(x)
        use_aug = bool(anchor_conf.item() < args.aug_conf_threshold and args.aug_repeats > 0)
        if not use_aug:
            return standard.detach(), {"anchor_conf": float(anchor_conf.item()), "used_aug": 0}
        outs = [standard.detach()]
        for _ in range(args.aug_repeats):
            outs.append(teacher(augment_ecg(x, args.noise_std, args.scale_std)).detach())
        return torch.stack(outs, dim=0).mean(dim=0), {
            "anchor_conf": float(anchor_conf.item()),
            "used_aug": len(outs),
        }


def consistency_loss(student_logits: torch.Tensor, teacher_logits_detached: torch.Tensor) -> torch.Tensor:
    target_probs = torch.sigmoid(teacher_logits_detached).detach()
    return F.binary_cross_entropy_with_logits(student_logits, target_probs)


@torch.no_grad()
def stochastic_restore(
    model: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
    restore_prob: float,
) -> int:
    if restore_prob <= 0:
        return 0
    n_restored = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in source_state:
            continue
        mask = torch.rand_like(param, dtype=torch.float32) < restore_prob
        if bool(mask.any()):
            src = source_state[name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(torch.where(mask, src, param.data))
            n_restored += int(mask.sum().item())
    return n_restored


@torch.enable_grad()
def adapt_cotta_lite(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    anchor: torch.nn.Module,
    source_state: dict[str, torch.Tensor],
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    all_labels, all_logits = [], []
    losses, anchor_confs = [], []
    n_aug_batches = 0
    n_restored_total = 0
    amp = device.type == "cuda" and not args.no_amp
    model.train()
    teacher.train()
    anchor.eval()
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        t_logits, t_stats = teacher_logits(teacher, anchor, signals, args, amp)
        with torch.cuda.amp.autocast(enabled=amp):
            student_logits = model(signals)
            loss = consistency_loss(student_logits, t_logits)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        update_ema(teacher, model, args.teacher_alpha)
        n_restored_total += stochastic_restore(model, source_state, args.restore_prob)

        losses.append(float(loss.detach().item()))
        anchor_confs.append(float(t_stats["anchor_conf"]))
        n_aug_batches += int(t_stats["used_aug"] > 0)
        all_labels.append(labels.numpy())
        all_logits.append(t_logits.detach().float().cpu().numpy())
    stats = {
        "n_seen": int(sum(len(x) for x in all_labels)),
        "mean_consistency_loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_anchor_conf": float(np.mean(anchor_confs)) if anchor_confs else float("nan"),
        "n_aug_batches": int(n_aug_batches),
        "n_restored_params": int(n_restored_total),
    }
    return np.concatenate(all_labels), sigmoid_np(np.concatenate(all_logits)), stats


def run_one_target(args: argparse.Namespace, target: str, ref_ids_by_center: dict[str, set[str]], device: torch.device):
    model = load_model(args.ckpt, device)
    params = configure_tent(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    teacher = clone_teacher(model).to(device)
    anchor = clone_teacher(model).to(device)
    source_state = {name: p.detach().clone().cpu() for name, p in model.named_parameters() if p.requires_grad}

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
    y_true_target, y_score_target, stats = adapt_cotta_lite(
        model, teacher, anchor, source_state, adapt_loader, optimizer, device, args
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
            "protocol": "online_cotta_lite_teacher_predictions_during_adaptation",
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
            "protocol": "post_cotta_lite_no_further_adaptation",
        }
        avg_aurocs.append(metric["macro_auroc"])
        avg_auprcs.append(metric["macro_auprc"])

    return {
        "target_center": target,
        "lr": args.lr,
        "teacher_alpha": args.teacher_alpha,
        "restore_prob": args.restore_prob,
        "aug_repeats": args.aug_repeats,
        "aug_conf_threshold": args.aug_conf_threshold,
        "noise_std": args.noise_std,
        "scale_std": args.scale_std,
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle),
        "avg_macro_auroc": float(np.mean(avg_aurocs)),
        "avg_macro_auprc": float(np.mean(avg_auprcs)),
        "per_center": per_center,
    }


def write_outputs(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cotta_lite_super5_ref_excluded.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    csv_path = out_dir / "cotta_lite_super5_ref_excluded.csv"
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
                "mean_consistency_loss",
                "mean_anchor_conf",
                "n_aug_batches",
                "n_restored_params",
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
                    "mean_consistency_loss": stats.get("mean_consistency_loss", ""),
                    "mean_anchor_conf": stats.get("mean_anchor_conf", ""),
                    "n_aug_batches": stats.get("n_aug_batches", ""),
                    "n_restored_params": stats.get("n_restored_params", ""),
                })
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt")
    ap.add_argument("--mmap_root", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal")
    ap.add_argument("--ref_root", default="/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/paper_tta_baselines_20260517/cotta_lite_lr1e-5")
    ap.add_argument("--centers", nargs="+", default=TARGET_CENTERS)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--teacher_alpha", type=float, default=0.999)
    ap.add_argument("--restore_prob", type=float, default=0.001)
    ap.add_argument("--aug_repeats", type=int, default=4)
    ap.add_argument("--aug_conf_threshold", type=float, default=0.72)
    ap.add_argument("--noise_std", type=float, default=0.01)
    ap.add_argument("--scale_std", type=float, default=0.03)
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
        "method": "CoTTA_lite_BN1d_multilabel_consistency",
        "reference_repo": "/root/autodl-tmp/external_repos/cotta",
        "source_checkpoint": args.ckpt,
        "views": {},
    }
    for target in args.centers:
        print(f"[target] {target}", flush=True)
        view = run_one_target(args, target, ref_ids, device)
        result["views"][target] = view
        row = view["per_center"][target]
        stats = row.get("adaptation_stats", {})
        print(
            f"  target {target}: AUROC={row['macro_auroc']:.4f} "
            f"AUPRC={row['macro_auprc']:.4f} avg={view['avg_macro_auroc']:.4f}/{view['avg_macro_auprc']:.4f} "
            f"aug_batches={stats.get('n_aug_batches', 0)}",
            flush=True,
        )
    write_outputs(Path(args.out_dir), result)


if __name__ == "__main__":
    main()
