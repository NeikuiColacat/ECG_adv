#!/usr/bin/env python
"""Train the repo-owned 500 Hz PTB-XL VAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.vae import build_vae500, vae500_loss


class VAE500ArrayDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        *,
        limit: int | None = None,
        cache_in_memory: bool = False,
    ):
        self.path = cache_dir / split / "signals_norm.npy"
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if cache_in_memory:
            self.signals = np.asarray(np.load(self.path, mmap_mode=None), dtype=np.float32)
        else:
            self.signals = np.load(self.path, mmap_mode="r")
        self.cache_in_memory = bool(cache_in_memory)
        self.limit = min(int(limit), len(self.signals)) if limit else len(self.signals)

    def __len__(self) -> int:
        return int(self.limit)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache_in_memory:
            signal = np.asarray(self.signals[idx], dtype=np.float32)
        else:
            signal = np.array(self.signals[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(np.ascontiguousarray(signal))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")


def _loader(dataset: Dataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0 and args.persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=train and args.drop_last,
    )


def _amp_context(args: argparse.Namespace):
    enabled = bool(args.amp and args.device.startswith("cuda"))
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _kl_beta(args: argparse.Namespace, epoch: int) -> float:
    if args.kl_warmup_epochs <= 0:
        return float(args.beta)
    scale = min(1.0, max(0.0, epoch / float(args.kl_warmup_epochs)))
    return float(args.beta) * scale


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {
        "loss": 0.0,
        "recon_huber": 0.0,
        "kl": 0.0,
        "first_diff_huber": 0.0,
        "lead_consistency_huber": 0.0,
    }
    n_samples = 0
    beta = _kl_beta(args, epoch)
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with _amp_context(args):
                out = model(batch)
                loss_out = vae500_loss(
                    out.recon,
                    batch,
                    out.mu,
                    out.log_var,
                    beta=beta,
                    first_diff_weight=args.first_diff_weight,
                    lead_consistency_weight=args.lead_consistency_weight,
                )
            if train:
                loss_out.loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
        bsz = int(batch.shape[0])
        n_samples += bsz
        totals["loss"] += float(loss_out.loss.detach().cpu()) * bsz
        totals["recon_huber"] += float(loss_out.recon_huber.cpu()) * bsz
        totals["kl"] += float(loss_out.kl.cpu()) * bsz
        totals["first_diff_huber"] += float(loss_out.first_diff_huber.cpu()) * bsz
        totals["lead_consistency_huber"] += float(loss_out.lead_consistency_huber.cpu()) * bsz
    if n_samples <= 0:
        raise RuntimeError("empty dataloader")
    return {key: value / n_samples for key, value in totals.items()} | {"kl_beta": beta, "n": float(n_samples)}


@torch.no_grad()
def _save_reconstruction_sample(model: torch.nn.Module, dataset: VAE500ArrayDataset, device: torch.device, path: Path, n: int) -> None:
    model.eval()
    n = min(int(n), len(dataset))
    if n <= 0:
        return
    batch = torch.stack([dataset[i] for i in range(n)], dim=0).to(device)
    out = model(batch, noise=torch.zeros((n, model.latent_channels, model.latent_length), device=device))
    np.savez_compressed(
        path,
        input=batch.detach().cpu().numpy().astype(np.float32),
        recon=out.recon.detach().cpu().numpy().astype(np.float32),
        z=out.z.detach().cpu().numpy().astype(np.float32),
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "model_metadata": {
                "variant": args.model_variant,
                "input_length": model.input_length,
                "in_channels": model.in_channels,
                "latent_channels": model.latent_channels,
                "latent_length": model.latent_length,
                "latent_scale": model.latent_scale,
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=_json_default), encoding="utf-8")

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        if args.matmul_precision:
            torch.set_float32_matmul_precision(args.matmul_precision)

    train_ds = VAE500ArrayDataset(
        cache_dir,
        "train",
        limit=args.limit_train,
        cache_in_memory=args.cache_in_memory,
    )
    val_ds = VAE500ArrayDataset(
        cache_dir,
        "val",
        limit=args.limit_val,
        cache_in_memory=args.cache_in_memory,
    )
    if args.cache_in_memory:
        train_gib = float(train_ds.signals.nbytes) / (1024**3)
        val_gib = float(val_ds.signals.nbytes) / (1024**3)
        print(f"[data] cache_in_memory train={train_gib:.2f}GiB val={val_gib:.2f}GiB", flush=True)
    train_loader = _loader(train_ds, args, train=True)
    val_loader = _loader(val_ds, args, train=False)

    device = torch.device(args.device)
    model = build_vae500(
        args.model_variant,
        base_channels=args.base_channels,
        use_attention=not args.no_attention,
    ).to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[model] initialized from {args.init_ckpt}", flush=True)
    if args.compile:
        print(f"[model] torch.compile mode={args.compile_mode}", flush=True)
        model = torch.compile(model, mode=args.compile_mode)
    optim_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.fused_adamw and args.device.startswith("cuda"):
        optim_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    except TypeError:
        optim_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, args=args, epoch=epoch)
        val_metrics = _run_epoch(model=model, loader=val_loader, optimizer=None, device=device, args=args, epoch=epoch)
        row = {
            "epoch": epoch,
            "time_sec": time.time() - start,
            "train": train_metrics,
            "val": val_metrics,
        }
        _append_jsonl(metrics_path, row)
        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} beta={train_metrics['kl_beta']:.6g}",
            flush=True,
        )
        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        _save_checkpoint(checkpoint_dir / "latest.pt", model=target_model, optimizer=optimizer, epoch=epoch, args=args, metrics=row)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            _save_checkpoint(checkpoint_dir / "best.pt", model=target_model, optimizer=optimizer, epoch=epoch, args=args, metrics=row)
            _save_reconstruction_sample(target_model, val_ds, device, sample_dir / "val_best_recon.npz", args.sample_count)

    summary = {
        "status": "finished",
        "best_val_loss": best_val,
        "epochs": args.epochs,
        "output_dir": str(output_dir),
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_variant", default="diffusets500_v1_dynamic", choices=["diffusets500_v1_dynamic", "diffusets500_v2_compact"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument("--first_diff_weight", type=float, default=0.05)
    parser.add_argument("--lead_consistency_weight", type=float, default=0.02)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--init_ckpt", default=None,
                        help="Optional VAE500 checkpoint used to initialize model weights.")
    parser.add_argument("--cache_in_memory", action="store_true",
                        help="Load train/val normalized arrays into RAM instead of mmap reads.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Enable TF32 matmul/cuDNN paths on Ampere/Ada GPUs.")
    parser.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium", ""],
                        help="torch.set_float32_matmul_precision value; empty string disables the call.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for the VAE model.")
    parser.add_argument("--compile_mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--fused_adamw", action="store_true",
                        help="Use fused AdamW when available on CUDA.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
