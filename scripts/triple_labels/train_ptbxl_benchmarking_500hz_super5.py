#!/usr/bin/env python3
"""Train selected ecg_ptbxl_benchmarking 1D backbones on PTB-XL Super5 500Hz.

This wrapper keeps the external benchmark repo read-only and uses this
project's PTB-XL records500 cache. It intentionally follows the benchmark
normalization style: fit one global StandardScaler-like mean/std on the PTB-XL
train split, then apply it to train/val/test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_CODE = REPO_ROOT / "model" / "ecg_ptbxl_benchmarking" / "code"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCH_CODE))

from ecg_adv_gen.training import compute_pos_weight  # noqa: E402
from models.basic_conv1d import fcn_wang, schirrmeister  # noqa: E402
from models.inception1d import inception1d  # noqa: E402
from models.resnet1d import resnet1d_wang  # noqa: E402
from models.xresnet1d import xresnet1d50  # noqa: E402

CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(args: argparse.Namespace) -> None:
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def model_builder(name: str) -> Callable[[], nn.Module]:
    common = {
        "num_classes": len(CLASS_NAMES),
        "input_channels": 12,
        "lin_ftrs_head": [128],
        "ps_head": 0.5,
    }
    builders = {
        "fastai_inception1d": lambda: inception1d(**common),
        "fastai_resnet1d_wang": lambda: resnet1d_wang(
            **common,
            kernel_size=5,
        ),
        "fastai_xresnet1d50": lambda: xresnet1d50(**common),
        "fastai_fcn_wang": lambda: fcn_wang(**common),
        "fastai_schirrmeister": lambda: schirrmeister(**common),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; valid={sorted(builders)}")
    return builders[name]


def load_split(cache_root: Path, split: str, *, mmap: bool, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mode = "r" if mmap else None
    x = np.load(cache_root / split / "signals_raw_mV.npy", mmap_mode=mode)
    y = np.load(cache_root / split / "labels.npy", mmap_mode=mode)
    if limit and limit > 0:
        x = x[:limit]
        y = y[:limit]
    return x, y.astype(np.float32, copy=False)


def as_memory(x: np.ndarray) -> np.ndarray:
    if isinstance(x, np.memmap):
        return np.asarray(x, dtype=np.float32)
    return x.astype(np.float32, copy=False)


def global_mean_std(x: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    mean = float(arr.mean(dtype=np.float64))
    std = float(arr.std(dtype=np.float64))
    return mean, max(std, 1e-8)


class ECGArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mean: float, std: float, train: bool) -> None:
        self.x = x
        self.y = y.astype(np.float32, copy=False)
        self.mean = float(mean)
        self.std = float(std)
        self.train = bool(train)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.x[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        x = np.transpose(x, (1, 0)).copy()
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


def loader_kwargs(args: argparse.Namespace) -> dict:
    out = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        out["persistent_workers"] = args.persistent_workers
        out["prefetch_factor"] = args.prefetch_factor
    return out


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    per_class = {}
    aurocs, auprcs = [], []
    for i, name in enumerate(CLASS_NAMES):
        yt = y_true[:, i]
        ys = y_score[:, i]
        n_pos = int((yt == 1).sum())
        if n_pos < 1 or n_pos == len(yt):
            per_class[name] = {"auroc": None, "auprc": None, "n_pos": n_pos}
            continue
        au = float(roc_auc_score(yt, ys))
        ap = float(average_precision_score(yt, ys))
        per_class[name] = {"auroc": au, "auprc": ap, "n_pos": n_pos}
        aurocs.append(au)
        auprcs.append(ap)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_classes_used": int(len(aurocs)),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, probs, labels = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(float(loss.item()))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(y.float().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(probs)


def train_one(model_name: str, args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mmap = not args.load_in_memory
    x_train, y_train = load_split(Path(args.cache_root), "train", mmap=mmap, limit=args.limit_train)
    x_val, y_val = load_split(Path(args.cache_root), "val", mmap=mmap, limit=args.limit_val)
    x_test, y_test = load_split(Path(args.cache_root), "audit", mmap=mmap, limit=args.limit_test)
    if args.load_in_memory:
        x_train = as_memory(x_train)
        x_val = as_memory(x_val)
        x_test = as_memory(x_test)
    mean, std = global_mean_std(x_train)

    train_ds = ECGArrayDataset(x_train, y_train, mean, std, train=True)
    val_ds = ECGArrayDataset(x_val, y_val, mean, std, train=False)
    test_ds = ECGArrayDataset(x_test, y_test, mean, std, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        **loader_kwargs(args),
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs(args))
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs(args))

    model = model_builder(model_name)().to(device)
    pos_weight = torch.tensor(
        compute_pos_weight(y_train, len(CLASS_NAMES), clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw and device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")
    best_score = -float("inf")
    logs = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val_loss, yv, pv = evaluate(model, val_loader, device, amp_dtype)
        val_metrics = compute_metrics(yv, pv)
        score = val_metrics["macro_auprc"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(row)
        print(
            f"[{model_name}] ep={epoch:03d}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val={row['val_macro_auroc']:.4f}/{row['val_macro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score or not (out_dir / "best_model.pt").exists():
            best_score = score
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    _, yt, pt = evaluate(model, test_loader, device, amp_dtype)
    test_metrics = compute_metrics(yt, pt)
    result = {
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "config": vars(args),
        "cache_root": str(args.cache_root),
        "normalization": {
            "policy": "dataset_global_standard_scaler",
            "train_mean": mean,
            "train_std": std,
        },
        "best_val_macro_auprc": best_score,
        "ptbxl_fold10": test_metrics,
        "runtime_sec": time.time() - t0,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    with (out_dir / "train_result.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[{model_name}] fold10={test_metrics['macro_auroc']:.4f}/"
        f"{test_metrics['macro_auprc']:.4f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default="/root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", required=True, help="Comma-separated model names.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_val", type=int, default=0)
    p.add_argument("--limit_test", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--load_in_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    p.add_argument("--cudnn_benchmark", action="store_true")
    p.add_argument("--fused_adamw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torch(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    results = []
    for model in [x.strip() for x in args.models.split(",") if x.strip()]:
        results.append(train_one(model, args))
    with (Path(args.output_dir) / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
