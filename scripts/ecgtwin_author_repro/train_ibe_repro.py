#!/usr/bin/env python
"""Reproduce ECGTwin stage-1 IBExtractor training with thesis-grade logs."""

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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
ECGTWIN_ROOT = REPO_ROOT / "model" / "ECGTwin"
if str(ECGTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGTWIN_ROOT))
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()

from module.IBExtractor import IBExtractor  # noqa: E402
from utils.data_utils import PairedECGDataset, paired_ecg_collate_fn, process_pat_info  # noqa: E402


def amp_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"train_ibe_repro_{output_dir.name}")
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
    csv_path = output_dir / "loss_curve.csv"
    keys = ["epoch", "train_loss", "eval_score", "lr", "epoch_time_sec", "total_time_sec"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in keys})

    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    eval_score = [r["eval_score"] for r in records]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(epochs, train_loss, marker="o", label="train loss", color="#1f77b4")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, eval_score, marker="s", label="eval score", color="#d62728")
    ax2.set_ylabel("eval score")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def move_batch_to_device(ecg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ecg = {"data": ecg["data"], "label": dict(ecg["label"])}
    ecg["data"] = ecg["data"].transpose(2, 1).to(device)
    ecg["label"]["text_embed"] = ecg["label"]["text_embed"].to(device)
    ecg["label"]["text_embed_mask"] = ecg["label"]["text_embed_mask"].to(device)
    ecg["label"]["pat_info"] = process_pat_info(
        hr=ecg["label"]["hr"],
        age=ecg["label"]["age"],
        sex=ecg["label"]["sex"],
    ).to(device)
    return ecg


def train_epoch(
    dataloader: DataLoader,
    model: IBExtractor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mini_batch_size: int,
    accumulation_steps: int,
    logger: logging.Logger,
    log_every: int,
    max_batches: int | None,
    amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    accum_loss_sum = 0.0
    accum_count = 0
    global_steps_per_epoch = len(dataloader) // accumulation_steps
    for micro_step, (ecg_1_raw, ecg_2_raw) in enumerate(dataloader, start=1):
        batch_size = ecg_1_raw["data"].shape[0]
        if batch_size != mini_batch_size:
            logger.warning("Expected micro batch %s, got %s.", mini_batch_size, batch_size)

        pat_info_1 = process_pat_info(
            hr=ecg_1_raw["label"]["hr"],
            age=ecg_1_raw["label"]["age"],
            sex=ecg_1_raw["label"]["sex"],
        ).to(device, non_blocking=True)
        pat_info_2 = process_pat_info(
            hr=ecg_2_raw["label"]["hr"],
            age=ecg_2_raw["label"]["age"],
            sex=ecg_2_raw["label"]["sex"],
        ).to(device, non_blocking=True)

        x_1 = ecg_1_raw["data"].transpose(2, 1).to(device, non_blocking=True)
        x_2 = ecg_2_raw["data"].transpose(2, 1).to(device, non_blocking=True)
        text_1 = ecg_1_raw["label"]["text_embed"].to(device, non_blocking=True)
        text_2 = ecg_2_raw["label"]["text_embed"].to(device, non_blocking=True)
        if torch.rand(1).item() <= 0.15:
            text_1 = None
            text_2 = None
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            logits_1, logits_2 = model(
                x_1,
                text_1,
                ecg_1_raw["label"]["text_embed_mask"].to(device, non_blocking=True),
                pat_info_1,
                x_2,
                text_2,
                ecg_2_raw["label"]["text_embed_mask"].to(device, non_blocking=True),
                pat_info_2,
            )
            labels = torch.arange(logits_1.shape[0], device=device)
            raw_loss = (criterion(logits_1.float(), labels) + criterion(logits_2.float(), labels)) * 0.5
        if scaler.is_enabled():
            scaler.scale(raw_loss / accumulation_steps).backward()
        else:
            (raw_loss / accumulation_steps).backward()
        accum_loss_sum += raw_loss.item()
        accum_count += batch_size

        if micro_step % accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            global_loss = accum_loss_sum / accumulation_steps
            total_loss += global_loss * accum_count
            total_count += accum_count
            if optimizer_step % log_every == 0:
                logger.info(
                    "step %s/%s train_loss %.6f micro_step %s samples_seen %s",
                    optimizer_step,
                    global_steps_per_epoch,
                    global_loss,
                    micro_step,
                    total_count,
                )
            accum_loss_sum = 0.0
            accum_count = 0
            if max_batches is not None and optimizer_step >= max_batches:
                break

    return total_loss / max(total_count, 1)


@torch.no_grad()
def eval_score(
    dataloader: DataLoader,
    model: IBExtractor,
    device: torch.device,
    max_batches: int | None,
    amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_score = 0.0
    total_count = 0
    for step, (ecg_1_raw, ecg_2_raw) in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break
        ecg_1 = move_batch_to_device(ecg_1_raw, device)
        ecg_2 = move_batch_to_device(ecg_2_raw, device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            f1 = model.extract_features(
                ecg_1["data"],
                ecg_1["label"]["text_embed"],
                ecg_1["label"]["text_embed_mask"],
                ecg_1["label"]["pat_info"],
                reduce=True,
            )
            f2 = model.extract_features(
                ecg_2["data"],
                ecg_2["label"]["text_embed"],
                ecg_2["label"]["text_embed_mask"],
                ecg_2["label"]["pat_info"],
                reduce=True,
            )
        f1 = f1.float()
        f2 = f2.float()
        f1 = f1 / f1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        f2 = f2 / f2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        total_score += torch.trace(f1 @ f2.t()).item()
        total_count += f1.shape[0]
    return total_score / max(total_count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DATA_ROOT / "ecgtwin_author_repro/ibe_stage1")
    parser.add_argument("--train_path", type=Path, default=DATA_ROOT / "ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt")
    parser.add_argument("--val_path", type=Path, default=DATA_ROOT / "ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
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

    config = vars(args).copy()
    config.update(
        {
            "model": {
                "embed_dim": 256,
                "num_heads": 8,
                "ff_hidden_size": 1024,
                "num_layers": 3,
                "text_embed_dim": 768,
                "patient_info_size": 3,
            },
            "effective_batch_size": args.batch_size,
            "accumulation_steps": args.batch_size // args.mini_batch_size,
            "paper_reference": "ECGTwin C.2: global batch 65536, mini batch 512, lr 1e-3, epochs 40.",
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
        batch_size=args.mini_batch_size,
        shuffle=True,
        collate_fn=paired_ecg_collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=args.drop_last,
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

    model = IBExtractor(
        embed_dim=256,
        num_heads=8,
        ff_hidden_size=1024,
        num_layers=3,
        text_embed_dim=768,
        patient_info_size=3,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accumulation_steps = args.batch_size // args.mini_batch_size
    if accumulation_steps * args.mini_batch_size != args.batch_size:
        raise ValueError("batch_size must be divisible by mini_batch_size")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16" and device.type == "cuda")

    metrics_path = args.output_dir / "metrics.jsonl"
    records: list[dict[str, Any]] = []
    best_score = -float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            criterion,
            device,
            args.mini_batch_size,
            accumulation_steps,
            logger,
            args.log_every,
            args.max_train_batches,
            args.amp and device.type == "cuda",
            amp_dtype,
            scaler,
        )
        score = eval_score(val_loader, model, device, args.max_val_batches, args.amp and device.type == "cuda", amp_dtype)
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": elapsed,
            "total_time_sec": time.time() - start,
        }
        records.append(record)
        write_jsonl(metrics_path, record)
        save_curves(records, args.output_dir)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "latest.pt")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "best.pt")
            torch.save(model.state_dict(), ckpt_dir / "IBE_best.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"IBE_model_ep{epoch}.pth")
        logger.info("epoch=%s train_loss=%.6f eval_score=%.6f time=%.1fs", epoch, train_loss, score, elapsed)

    logger.info("done best_eval_score=%.6f", best_score)


if __name__ == "__main__":
    main()
