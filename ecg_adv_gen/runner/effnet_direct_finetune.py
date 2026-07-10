#!/usr/bin/env python3
"""Historical unmatched Direct K-shot target-center fine-tuning runner.

This legacy runner has an independent optimizer/selection path and therefore
must not be reported as matched A0. New matched A0/A5 runs use the shared
``synth_online_at_super5.py`` trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from ecg_adv_gen.preprocessing import crop_signal_tc  # noqa: E402
from ecg_adv_gen.runner.effnet_direct import (  # noqa: E402
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PROJECT_ROOT,
    PYTHON,
    eval_metrics,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, NUM_SUPER5  # noqa: E402
from ecg_adv_gen.run_naming import build_effnet_direct_run_leaf  # noqa: E402
from ecg_adv_gen.evaluation.ref_exclusion import append_target_ref_exclusion_args  # noqa: E402
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc  # noqa: E402
from ecg_adv_gen.runner.process import build_process_env, run_stream  # noqa: E402
from ecg_adv_gen.training import compute_pos_weight, masked_bce_with_logits, random_split_indices  # noqa: E402


_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
OUT_ROOT = DATA_ROOT / "paper_direct_finetune_k500_20260516"
SUBSET_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516/subsets"
COMPARISON_STATUS = "historical_unmatched"


class NPZRealDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        crop_len: int,
        mode: str,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = int(crop_len)
        self.mode = mode

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        crop = crop_signal_tc(
            sig_tc,
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[real_idx]).float(),
        )


def subset_paths(center: str, k: int, seed: int, subset_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(subset_root) if subset_root else SUBSET_ROOT
    base_dir = root / center / f"k{k}_seed{seed}"
    base = base_dir / f"{center}_real_k{k}_seed{seed}"
    return {
        "signals": base.with_suffix(".signals.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
    }


def load_model(device: torch.device) -> nn.Module:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=NUM_SUPER5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    state = torch.load(BASELINE_CKPT, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return random_split_indices(n, val_fraction, seed, zero_val_policy="identity")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> dict:
    model.eval()
    losses, labels_all, logits_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(float(loss.item()))
        labels_all.append(y.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    labels = np.concatenate(labels_all, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    metrics = compute_macro_auroc_auprc(labels, probs, CLASS_NAMES_SUPER5, min_pos=1)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed, args.subset_root)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    out_root = Path(args.out_root) if args.out_root else OUT_ROOT
    out_dir = out_root / "runs" / build_effnet_direct_run_leaf(
        {
            "center": center,
            "k": args.k,
            "epochs": args.epochs,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        }
    )
    eval_path = out_dir / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} direct fine-tune already evaluated")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(paths["signals"], allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        record_ids = data["record_ids"].astype(str)
    if signals.shape[1:] != (1000, 12):
        raise ValueError(f"expected signals (N,1000,12), got {signals.shape}")
    if labels.shape[1] != NUM_SUPER5:
        raise ValueError(f"expected labels (N,{NUM_SUPER5}), got {labels.shape}")

    train_idx, val_idx = split_indices(len(labels), args.val_fraction, args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = NPZRealDataset(signals, labels, train_idx, crop_len=args.crop_len, mode="train")
    val_ds = NPZRealDataset(signals, labels, val_idx, crop_len=args.crop_len, mode="eval")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    pos_weight = torch.tensor(
        compute_pos_weight(labels[train_idx], NUM_SUPER5, clip_max=args.pos_weight_clip_max),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits, y):
        return masked_bce_with_logits(logits, y, pos_weight)

    model = load_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "center": center,
            "subset_signals": str(paths["signals"]),
            "ref_meta": str(paths["meta"]),
            "init_checkpoint": BASELINE_CKPT,
            "class_names": list(CLASS_NAMES_SUPER5),
            "train_record_ids": record_ids[train_idx].tolist(),
            "val_record_ids": record_ids[val_idx].tolist(),
            "selection_policy": "direct target-center real fine-tune; no latent hull, no PGD, no synthetic ECG",
            "comparison_status": COMPARISON_STATUS,
        }
    )
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    best_score = -float("inf")
    log_rows = []
    t0 = time.time()
    print(
        f"[data] center={center} train={len(train_ds)} val={len(val_ds)} "
        f"signals={paths['signals']}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
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
            losses.append(float(loss.item()))
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        score = float(val_metrics["macro_auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_loss": val_metrics["loss"],
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
            "val_macro_auprc": float(val_metrics["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log_rows.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} auroc={row['val_macro_auroc']:.4f} "
            f"auprc={row['val_macro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(log_rows, f, indent=2)

    with (out_dir / "train_result.json").open("w") as f:
        json.dump(
            {
                "center": center,
                "best_val_macro_auprc": best_score,
                "epochs_trained": args.epochs,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "pos_weight": pos_weight.detach().cpu().tolist(),
                "comparison_status": COMPARISON_STATUS,
            },
            f,
            indent=2,
        )

    eval_cmd = [
        PYTHON,
        "-u",
        "ecg_adv_gen/runner/pn2021_clean_eval.py",
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
        "--min_pos",
        str(args.eval_min_pos),
        "--ptbxl_csv",
        str(DATA_ROOT / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache",
        str(DATA_ROOT / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir",
        PN2021_MMAP_CACHE_DIR,
        "--pn2021_root",
        str(DATA_ROOT / "physionet2021"),
        "--skip_mimic",
    ]
    append_target_ref_exclusion_args(
        eval_cmd,
        args.subset_root or SUBSET_ROOT,
        k=args.k,
        seed=args.subset_seed,
    )
    eval_cmd.extend([
        "--report_drop_all_zero_pn2021",
        "--output_path",
        str(eval_path),
    ])
    if args.eval_pn2021_limit:
        eval_cmd.extend(["--pn2021_limit", str(args.eval_pn2021_limit)])
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))
    env.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / "cache"))
    run_stream(
        eval_cmd,
        log_path=out_dir / "eval_full.log",
        cwd=PROJECT_ROOT,
        env=build_process_env(base=env),
    )
    return eval_path


def write_summary(rows: list[dict], out_root: Path = OUT_ROOT) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "direct_finetune_k500.csv"
    fields = [
        "center",
        "K",
        "epochs",
        "target_auroc",
        "target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    md_path = summary_dir / "direct_finetune_k500.md"
    with md_path.open("w") as f:
        f.write("# Direct K=500 Target-Center Fine-Tune\n\n")
        f.write("| center | K | epochs | target AUROC/AUPRC | PN2021 avg | PTB-XL |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['center']} | {row['K']} | {row['epochs']} | "
                f"{row['target_auroc']:.4f} / {row['target_auprc']:.4f} | "
                f"{row['pn2021_avg_auroc']:.4f} / {row['pn2021_avg_auprc']:.4f} | "
                f"{row['ptbxl_auroc']:.4f} / {row['ptbxl_auprc']:.4f} |\n"
            )
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def parse_eval(center: str, k: int, epochs: int, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    return {
        "center": center,
        "K": k,
        "epochs": epochs,
        "target_auroc": float(data["pn2021"]["per_center"][center]["macro_auroc"]),
        "target_auprc": float(data["pn2021"]["per_center"][center]["macro_auprc"]),
        "pn2021_avg_auroc": float(data["pn2021"]["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(data["pn2021"]["avg_macro_auprc"]),
        "ptbxl_auroc": float(data["ptbxl_test"]["macro_auroc"]),
        "ptbxl_auprc": float(data["ptbxl_test"]["macro_auprc"]),
        "eval_path": str(eval_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["ningbo", "cpsc_2018"])
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--subset_seed", type=int, default=20260531)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--eval_min_pos", type=int, default=10)
    p.add_argument(
        "--eval_pn2021_limit",
        type=int,
        default=0,
        help="Optional PN2021 per-center record cap for engineering smoke runs only.",
    )
    p.add_argument("--out_root", default="")
    p.add_argument(
        "--subset_root",
        default="",
        help=(
            "Optional K-shot subset root. Defaults to the historical "
            "paper_vae_only_latenthull_sweep_20260516/subsets root."
        ),
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("[warning] historical_unmatched Direct runner; not the matched A0 contract", flush=True)
    out_root = Path(args.out_root) if args.out_root else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for center in args.centers:
        eval_path = train_one(center, args)
        rows.append(parse_eval(center, args.k, args.epochs, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
