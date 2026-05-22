#!/usr/bin/env python3
"""Source-only MixStyle baseline for PTB-XL Super5 -> PN2021.

Official reference repo:
  /root/autodl-tmp/external_repos/mixstyle-release

The official MixStyle implementation is 2D CNN oriented.  This runner adapts the
same feature-statistic mixing idea to 1D ECG features and trains the existing
EfficientNet1DV2 source model from PTB-XL folds 1-8.  No target-center data is
used during training.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_source_dg_baselines_20260517/mixstyle_super5")


class MixStyle1D(nn.Module):
    """1D variant of Zhou et al. MixStyle feature-statistic mixing."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random") -> None:
        super().__init__()
        self.p = float(p)
        self.beta = torch.distributions.Beta(float(alpha), float(alpha))
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.mix = str(mix)
        self._activated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if torch.rand(()) > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = self.beta.sample((b, 1, 1)).to(x.device)
        if self.mix == "random":
            perm = torch.randperm(b, device=x.device)
        elif self.mix == "crossdomain":
            perm = torch.arange(b - 1, -1, -1, device=x.device)
        else:
            raise NotImplementedError(f"unknown MixStyle mix mode: {self.mix}")
        mu2, sig2 = mu[perm], sig[perm]
        return x_normed * (sig * lmda + sig2 * (1 - lmda)) + (mu * lmda + mu2 * (1 - lmda))


class MixStyleEfficientNet1D(nn.Module):
    def __init__(self, backbone: EfficientNet1DV2, layers: list[int], p: float, alpha: float, mix: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = set(int(i) for i in layers)
        self.mixstyle = MixStyle1D(p=p, alpha=alpha, mix=mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.initial_conv(x)
        if -1 in self.layers:
            x = self.mixstyle(x)
        for i, block in enumerate(b.features):
            x = block(x)
            if i in self.layers:
                x = self.mixstyle(x)
        x = b.final_conv(x)
        x = b.final_norm(x)
        return b.classifier(x)


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
        print(f"[skip] MixStyle already evaluated: {eval_path}")
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
    all_sig = preprocess_ptbxl_all(
        args.data_path,
        args.cache_path,
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
    )
    train_ds = PTBXLDatasetScheme(np.asarray(all_sig[train_idx]), train_labels, crop_len=args.crop_len, mode="train")
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

    backbone = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    backbone.apply(init_weights)
    model = MixStyleEfficientNet1D(backbone, args.mix_layers, args.mixstyle_p, args.mixstyle_alpha, args.mixstyle_mode).to(device)

    pos_weight = torch.tensor(compute_pos_weight(train_labels, 5, clip_max=args.pos_weight_clip_max), dtype=torch.float32, device=device)

    def criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    cfg = {
        **vars(args),
        "method": "mixstyle_source_only",
        "reference_repo": "/root/autodl-tmp/external_repos/mixstyle-release",
        "protocol": "PTB-XL folds 1-8 only; no PN2021 target data",
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
    print(
        f"[train] MixStyle layers={args.mix_layers} p={args.mixstyle_p} alpha={args.mixstyle_alpha} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
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
            torch.save(model.backbone.state_dict(), out_dir / "best_model.pt")
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
                "method": "mixstyle_source_only",
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
        "method": "mixstyle_source_only",
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
    csv_path = summary_dir / "mixstyle_source_only.csv"
    fields = list(row.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(f"[summary] wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(DEFAULT_OUT_ROOT / "mixstyle_p0p5_a0p1_layers-1_3_seed42"))
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
    p.add_argument("--mixstyle_p", type=float, default=0.5)
    p.add_argument("--mixstyle_alpha", type=float, default=0.1)
    p.add_argument("--mixstyle_mode", default="random", choices=["random", "crossdomain"])
    p.add_argument("--mix_layers", nargs="+", type=int, default=[-1, 3])
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
