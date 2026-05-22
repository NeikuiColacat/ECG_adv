#!/usr/bin/env python3
"""GroupDRO source-only proxy baseline for PTB-XL Super5 -> PN2021.

Official reference repo:
  /root/autodl-tmp/external_repos/group_DRO

PTB-XL is a single-source dataset in this project, so this is a proxy baseline:
we treat PTB-XL stratification folds 1-8 as source groups and train with the
GroupDRO exponential group-weight update.  No PN2021 target data is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PYTHON,
)
from scripts.triple_labels.label_schemes import get_scheme  # noqa: E402
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    PTBXLDatasetScheme,
    compute_macro_auroc_auprc,
    compute_pos_weight,
    evaluate,
    get_ptbxl_labels_for_scheme,
    init_weights,
    masked_bce_with_logits,
    preprocess_ptbxl_all,
)


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_source_dg_baselines_20260517/groupdro_super5")


class PTBXLDatasetWithGroup(Dataset):
    def __init__(self, base: PTBXLDatasetScheme, groups: np.ndarray) -> None:
        self.base = base
        self.groups = np.asarray(groups, dtype=np.int64)
        if len(self.base) != len(self.groups):
            raise ValueError(f"base/groups length mismatch: {len(self.base)} vs {len(self.groups)}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, torch.tensor(int(self.groups[idx]), dtype=torch.long)


def masked_bce_per_sample(logits: torch.Tensor, labels_mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    mask = (labels_mask >= 0).float()
    labels_safe = torch.where(mask.bool(), labels_mask, torch.zeros_like(labels_mask))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels_safe,
        pos_weight=pos_weight,
        reduction="none",
    )
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (bce * mask).sum(dim=1) / denom


class GroupDROLoss:
    def __init__(self, n_groups: int, step_size: float, device: torch.device) -> None:
        self.n_groups = int(n_groups)
        self.step_size = float(step_size)
        self.adv_probs = torch.ones(self.n_groups, device=device) / self.n_groups

    def __call__(self, per_sample_loss: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        group_losses = []
        for g in range(self.n_groups):
            mask = groups == g
            if bool(mask.any()):
                group_losses.append(per_sample_loss[mask].mean())
            else:
                group_losses.append(torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device))
        group_loss = torch.stack(group_losses)
        with torch.no_grad():
            present = torch.tensor([(groups == g).any() for g in range(self.n_groups)], device=groups.device)
            adjusted = torch.where(present, group_loss.detach(), torch.zeros_like(group_loss))
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted)
            self.adv_probs = self.adv_probs / self.adv_probs.sum().clamp_min(1e-12)
        return group_loss @ self.adv_probs.detach(), group_loss.detach()


def run_cmd(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def train(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] GroupDRO already evaluated: {eval_path}")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    scheme = get_scheme("super5")
    class_names = scheme["class_names"]
    label_cache = str(out_dir / "ptbxl_labels")
    train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(PTBXL_CSV, scheme, label_cache, folds=list(range(1, 9)))
    val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(PTBXL_CSV, scheme, label_cache, folds=[9])
    test_idx, test_labels, _ = get_ptbxl_labels_for_scheme(PTBXL_CSV, scheme, label_cache, folds=[10])
    df = pd.read_csv(PTBXL_CSV)
    train_groups = df.iloc[train_idx].strat_fold.astype(int).to_numpy() - 1
    if train_groups.min() < 0 or train_groups.max() > 7:
        raise ValueError(f"unexpected train group ids from strat_fold: {train_groups.min()}..{train_groups.max()}")

    all_sig = preprocess_ptbxl_all(
        args.data_path,
        args.cache_path,
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
    )
    train_base = PTBXLDatasetScheme(np.asarray(all_sig[train_idx]), train_labels, crop_len=args.crop_len, mode="train")
    train_ds = PTBXLDatasetWithGroup(train_base, train_groups)
    val_ds = PTBXLDatasetScheme(np.asarray(all_sig[val_idx]), val_labels, crop_len=args.crop_len, mode="eval")
    test_ds = PTBXLDatasetScheme(np.asarray(all_sig[test_idx]), test_labels, crop_len=args.crop_len, mode="eval")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

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
    model.apply(init_weights)
    pos_weight = torch.tensor(compute_pos_weight(train_labels, 5, clip_max=args.pos_weight_clip_max), dtype=torch.float32, device=device)

    def criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    gdro = GroupDROLoss(n_groups=8, step_size=args.groupdro_step_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    cfg = {
        **vars(args),
        "method": "groupdro_source_only_proxy",
        "reference_repo": "/root/autodl-tmp/external_repos/group_DRO",
        "protocol": "PTB-XL folds 1-8 only; source groups are PTB-XL strat_fold IDs 1-8; no PN2021 target data",
        "group_counts": np.bincount(train_groups, minlength=8).astype(int).tolist(),
        "class_names": class_names,
        "pos_weight": pos_weight.detach().cpu().tolist(),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    best_score = -float("inf")
    best_test = None
    patience = 0
    log_rows = []
    t0_all = time.time()
    print(f"[train] GroupDRO proxy groups={cfg['group_counts']} train={len(train_ds)} val={len(val_ds)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, group_loss_rows = [], []
        t0 = time.time()
        for x, y, g in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                per_sample_loss = masked_bce_per_sample(logits, y, pos_weight)
                loss, group_loss = gdro(per_sample_loss, g)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            group_loss_rows.append(group_loss.cpu().numpy())
        scheduler.step()
        val_loss, vy, vp = evaluate(model, val_loader, criterion, device)
        val_metric = compute_macro_auroc_auprc(vy, vp, class_names)
        score = float(val_metric["macro_auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_loss),
            "val_macro_auroc": float(val_metric["macro_auroc"]),
            "val_macro_auprc": float(val_metric["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "adv_probs": gdro.adv_probs.detach().cpu().tolist(),
            "mean_batch_group_loss": np.nan_to_num(np.stack(group_loss_rows), nan=0.0).mean(axis=0).tolist(),
            "time": round(time.time() - t0, 1),
        }
        log_rows.append(row)
        print(
            f"Ep {epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} val={row['val_loss']:.4f} "
            f"auroc={row['val_macro_auroc']:.4f} auprc={row['val_macro_auprc']:.4f}",
            flush=True,
        )
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(log_rows, f, indent=2)
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            test_loss, ty, tp = evaluate(model, test_loader, criterion, device)
            test_metric = compute_macro_auroc_auprc(ty, tp, class_names)
            best_test = {"test_loss": float(test_loss), **test_metric}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early-stop] patience={args.patience}", flush=True)
                break

    with (out_dir / "train_result.json").open("w") as f:
        json.dump(
            {
                "method": "groupdro_source_only_proxy",
                "best_val_macro_auprc": best_score,
                "best_test": best_test,
                "epochs_trained": len(log_rows),
                "elapsed_sec": round(time.time() - t0_all, 1),
                "config": cfg,
            },
            f,
            indent=2,
        )

    eval_cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(out_dir),
        "--device",
        args.device,
        "--crop_len",
        str(args.crop_len),
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_cache",
        args.cache_path,
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir",
        PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--output_path",
        str(eval_path),
    ]
    run_cmd(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def parse_eval(args: argparse.Namespace, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    row = {
        "method": "groupdro_source_only_proxy",
        "tag": Path(args.output_dir).name,
        "ptbxl_auroc": float(data["ptbxl_test"]["macro_auroc"]),
        "ptbxl_auprc": float(data["ptbxl_test"]["macro_auprc"]),
        "pn2021_avg_auroc": float(data["pn2021"]["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(data["pn2021"]["avg_macro_auprc"]),
        "eval_path": str(eval_path),
    }
    for c in ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]:
        pc = data["pn2021"]["per_center"][c]
        row[f"{c}_auroc"] = float(pc["macro_auroc"])
        row[f"{c}_auprc"] = float(pc["macro_auprc"])
    return row


def write_summary(row: dict, out_root: Path) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "groupdro_source_only_proxy.csv"
    fields = list(row.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(f"[summary] wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(DEFAULT_OUT_ROOT / "groupdro_folds_step0p01_seed42"))
    p.add_argument("--data_path", default="/root/autodl-tmp/ptbxl/raw100.npy")
    p.add_argument("--cache_path", default=PTBXL_PREP)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--cosine_tmax", type=int, default=30)
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--groupdro_step_size", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = train(args)
    if not args.dry_run:
        write_summary(parse_eval(args, eval_path), Path(args.output_dir).parents[0])


if __name__ == "__main__":
    main()
