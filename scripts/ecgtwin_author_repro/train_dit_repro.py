#!/usr/bin/env python
"""Reproduce ECGTwin stage-2 DiT latent diffusion with validation curves."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
ECGTWIN_ROOT = REPO_ROOT / "model" / "ECGTwin"
if str(ECGTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGTWIN_ROOT))

from module.IBExtractor import IBExtractor  # noqa: E402
from utils.data_utils import PairedECGDataset, paired_ecg_collate_fn, process_pat_info  # noqa: E402
from utils.model_utils import build_noise_predictor  # noqa: E402


def amp_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"train_dit_repro_{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def yaml_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [yaml_safe(v) for v in obj]
    return obj


def save_curves(records: list[dict[str, Any]], output_dir: Path) -> None:
    if not records:
        return
    keys = ["epoch", "train_loss", "val_loss", "lr", "epoch_time_sec", "total_time_sec"]
    with (output_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in keys})
    epochs = [r["epoch"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, [r["train_loss"] for r in records], marker="o", label="train loss")
    ax.plot(epochs, [r["val_loss"] for r in records], marker="s", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("diffusion noise MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def build_hparams() -> dict[str, Any]:
    return {
        "lr": 1e-4,
        "ddpm": {"num_train_steps": 1000, "beta_start": 0.00085, "beta_end": 0.0120},
        "DiT": {"hidden_size": 256, "depth": 7, "num_heads": 8, "patient_info_size": 3},
        "ibe": {
            "embed_dim": 256,
            "num_heads": 8,
            "ff_hidden_size": 1024,
            "num_layers": 3,
            "text_embed_dim": 768,
            "patient_info_size": 3,
        },
    }


def prepare_conditions(
    ecg_ref: dict[str, Any],
    ecg_tar: dict[str, Any],
    ibe_model: IBExtractor,
    device: torch.device,
    mix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    pat_info_ref = process_pat_info(
        hr=ecg_ref["label"]["hr"],
        age=ecg_ref["label"]["age"],
        sex=ecg_ref["label"]["sex"],
    ).to(device)
    text_embed_ref = ecg_ref["label"]["text_embed"].to(device)
    text_embed_mask_ref = ecg_ref["label"]["text_embed_mask"].to(device)
    latent_ref = ecg_ref["data"].transpose(2, 1).to(device)
    with torch.no_grad():
        base_vector = ibe_model.extract_features(
            latent_ref, text_embed_ref, text_embed_mask_ref, pat_info_ref, reduce=True
        )
    base_vector_mask = (torch.rand(1, base_vector.shape[1], device=device) > 0.15).float()
    base_vector = base_vector * base_vector_mask

    if mix:
        text_embed_tar = ecg_tar["label"]["text_embed_whole"].unsqueeze(1).to(device, dtype=torch.float32)
        text_embed_mask_tar = None
    else:
        text_embed_tar = ecg_tar["label"]["text_embed"].to(device)
        text_embed_mask_tar = ecg_tar["label"]["text_embed_mask"].to(device)
    pat_info_tar = process_pat_info(
        normalize=True,
        add_token=False,
        hr=ecg_tar["label"]["hr"],
        age=ecg_tar["label"]["age"],
        sex=ecg_tar["label"]["sex"],
    ).to(device)
    latent_tar = ecg_tar["data"].to(device)
    return latent_tar, text_embed_tar, text_embed_mask_tar, pat_info_tar, base_vector


def diffusion_loss(
    ecg_ref: dict[str, Any],
    ecg_tar: dict[str, Any],
    noise_predictor: torch.nn.Module,
    diffused_model: DDPMScheduler,
    ibe_model: IBExtractor,
    device: torch.device,
    mix: bool,
    amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)
        if device.type == "cuda"
        else nullcontext()
    ):
        latent_tar, text_embed_tar, text_embed_mask_tar, pat_info_tar, base_vector = prepare_conditions(
            ecg_ref, ecg_tar, ibe_model, device, mix=mix
        )
    noise = torch.randn_like(latent_tar)
    t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (latent_tar.shape[0],), device=device)
    xt = diffused_model.add_noise(latent_tar, noise, t)
    with (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)
        if device.type == "cuda"
        else nullcontext()
    ):
        noise_estim = noise_predictor(xt, t, text_embed_tar, text_embed_mask_tar, pat_info_tar, base_vector)
        loss = F.mse_loss(noise_estim.float(), noise.float(), reduction="sum").div(noise.size(0))
    return loss


def train_epoch(
    dataloader: DataLoader,
    noise_predictor: torch.nn.Module,
    diffused_model: DDPMScheduler,
    ibe_model: IBExtractor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    mix: bool,
    max_batches: int | None,
    logger: logging.Logger,
    log_every: int,
    amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    noise_predictor.train()
    total_loss = 0.0
    count = 0
    for step, (ecg_ref, ecg_tar) in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break
        loss = diffusion_loss(ecg_ref, ecg_tar, noise_predictor, diffused_model, ibe_model, device, mix, amp, amp_dtype)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        bs = ecg_tar["data"].shape[0]
        total_loss += loss.item() * bs
        count += bs
        if step % log_every == 0:
            logger.info("step %s/%s train_loss %.6f lr %.8f", step, len(dataloader), loss.item(), scheduler.get_last_lr()[0])
    return total_loss / max(count, 1)


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    noise_predictor: torch.nn.Module,
    diffused_model: DDPMScheduler,
    ibe_model: IBExtractor,
    device: torch.device,
    mix: bool,
    max_batches: int | None,
    amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    noise_predictor.eval()
    total_loss = 0.0
    count = 0
    for step, (ecg_ref, ecg_tar) in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break
        loss = diffusion_loss(ecg_ref, ecg_tar, noise_predictor, diffused_model, ibe_model, device, mix, amp, amp_dtype)
        bs = ecg_tar["data"].shape[0]
        total_loss += loss.item() * bs
        count += bs
    return total_loss / max(count, 1)


def load_ibe(path: Path, hparams: dict[str, Any], device: torch.device) -> IBExtractor:
    ibe = IBExtractor(
        embed_dim=hparams["ibe"]["embed_dim"],
        num_heads=hparams["ibe"]["num_heads"],
        ff_hidden_size=hparams["ibe"]["ff_hidden_size"],
        num_layers=hparams["ibe"]["num_layers"],
        text_embed_dim=hparams["ibe"]["text_embed_dim"],
        patient_info_size=hparams["ibe"]["patient_info_size"],
    )
    obj = torch.load(path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    ibe.load_state_dict(state)
    ibe.to(device)
    ibe.eval()
    return ibe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("/root/autodl-tmp/ecgtwin_author_repro/dit_stage2"))
    parser.add_argument("--train_path", type=Path, default=Path("/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt"))
    parser.add_argument("--val_path", type=Path, default=Path("/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt"))
    parser.add_argument("--ibe_path", type=Path, default=Path("/root/autodl-tmp/ecgtwin_author_repro/ibe_stage1/checkpoints/IBE_best.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--use_pretrained_author_ibe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    logger = setup_logger(args.output_dir)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(args.matmul_precision)
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    hparams = build_hparams()
    if args.use_pretrained_author_ibe:
        args.ibe_path = ECGTWIN_ROOT / "checkpoints" / "ibe_model.pth"

    config = vars(args).copy()
    config.update(
        {
            "meta": {"model_type": "DiT_ECGTwin", "vae_latent": True, "mix": False},
            "hyper_para": hparams,
            "paper_reference": "ECGTwin C.2: DiT latent diffusion, T=1000, beta=[8.5e-4,1.2e-2], base-vector dropout 0.15.",
        }
    )
    config = yaml_safe(config)
    with (args.output_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("config: %s", config)

    train_dataset = PairedECGDataset(str(args.train_path))
    val_dataset = PairedECGDataset(str(args.val_path))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=paired_ecg_collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=paired_ecg_collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    logger.info("train_size=%s val_size=%s", len(train_dataset), len(val_dataset))

    noise_predictor = build_noise_predictor("DiT_ECGTwin", 4, hparams).to(device)
    diffused_model = DDPMScheduler(
        num_train_timesteps=hparams["ddpm"]["num_train_steps"],
        beta_start=hparams["ddpm"]["beta_start"],
        beta_end=hparams["ddpm"]["beta_end"],
    )
    ibe_model = load_ibe(args.ibe_path, hparams, device)
    optimizer = torch.optim.AdamW(noise_predictor.parameters(), lr=hparams["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs * (args.max_train_batches or len(train_loader)),
        eta_min=0.1 * hparams["lr"],
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_train = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            train_loader,
            noise_predictor,
            diffused_model,
            ibe_model,
            optimizer,
            scheduler,
            device,
            mix=False,
            max_batches=args.max_train_batches,
            logger=logger,
            log_every=args.log_every,
            amp=args.amp and device.type == "cuda",
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss = validate(
            val_loader,
            noise_predictor,
            diffused_model,
            ibe_model,
            device,
            mix=False,
            max_batches=args.max_val_batches,
            amp=args.amp and device.type == "cuda",
            amp_dtype=amp_dtype,
        )
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": elapsed,
            "total_time_sec": time.time() - start,
        }
        records.append(record)
        write_jsonl(metrics_path, record)
        save_curves(records, args.output_dir)

        state = {
            "model": noise_predictor.state_dict(),
            "epoch": epoch,
            "config": config,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, ckpt_dir / "latest.pt")
        torch.save(noise_predictor.state_dict(), ckpt_dir / "DiT_ECGTwin_latest.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, ckpt_dir / "best_val.pt")
            torch.save(noise_predictor.state_dict(), ckpt_dir / "DiT_ECGTwin_best_val.pth")
        if train_loss < best_train:
            best_train = train_loss
            torch.save(state, ckpt_dir / "best_train.pt")
            torch.save(noise_predictor.state_dict(), ckpt_dir / "DiT_ECGTwin_best_train.pth")
        logger.info("epoch=%s train_loss=%.6f val_loss=%.6f time=%.1fs", epoch, train_loss, val_loss, elapsed)

    logger.info("done best_val=%.6f best_train=%.6f", best_val, best_train)


if __name__ == "__main__":
    main()
