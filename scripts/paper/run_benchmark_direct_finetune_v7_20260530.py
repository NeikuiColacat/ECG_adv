#!/usr/bin/env python3
"""Generic direct K-shot fine-tuning baseline for benchmark 1D ECG backbones.

This is the architecture-general counterpart to
``run_direct_finetune_k500_20260516.py``.  It keeps the same target-center
protocol, ref-excluded PN2021 v7 evaluation, and simple supervised baseline,
but initializes any model registered in ``scripts.triple_labels.model_zoo``.
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
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.training import compute_pos_weight, masked_bce_with_logits, random_split_indices  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, NUM_SUPER5  # noqa: E402
from scripts.triple_labels.model_zoo import available_model_names, build_super5_model, normalize_model_name  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_macro_auroc_auprc  # noqa: E402


DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp",
    )
)
PYTHON = os.environ.get("ECGTWIN_PYTHON", "/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
SUBSET_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"


def model_slug(model_name: str) -> str:
    return normalize_model_name(model_name).replace("/", "_").replace(" ", "_")


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


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    name = normalize_model_name(args.model_name)
    model = build_super5_model(name, num_classes=NUM_SUPER5).to(device)
    if not args.init_ckpt:
        raise ValueError("--init_ckpt is required for benchmark direct fine-tuning")
    state = torch.load(args.init_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
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


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(Path.home() / "tmp_ecg"))
    env.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / "cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def run_leaf(args: argparse.Namespace, center: str) -> str:
    val_tag = str(args.val_fraction).replace(".", "p")
    return (
        f"{center}_K{args.k}_direct_ft_{model_slug(args.model_name)}"
        f"_ep{args.epochs}_seed{args.seed}_val{val_tag}"
    )


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed, args.subset_root)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    out_root = Path(args.out_root)
    out_dir = out_root / "runs" / run_leaf(args, center)
    eval_path = out_dir / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} {model_slug(args.model_name)} direct fine-tune already evaluated")
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

    model = build_model(args, device)
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
            "init_checkpoint": str(args.init_ckpt),
            "model_name": normalize_model_name(args.model_name),
            "class_names": list(CLASS_NAMES_SUPER5),
            "train_record_ids": record_ids[train_idx].tolist(),
            "val_record_ids": record_ids[val_idx].tolist(),
            "selection_policy": "direct target-center real fine-tune; no latent hull, no PGD, no synthetic ECG",
        }
    )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    best_score = -float("inf")
    log_rows = []
    t0 = time.time()
    print(
        f"[data] model={model_slug(args.model_name)} center={center} "
        f"train={len(train_ds)} val={len(val_ds)} signals={paths['signals']}"
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
        with (out_dir / "training_log.json").open("w", encoding="utf-8") as f:
            json.dump(log_rows, f, indent=2)

    with (out_dir / "train_result.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "center": center,
                "model_name": normalize_model_name(args.model_name),
                "best_val_macro_auprc": best_score,
                "epochs_trained": args.epochs,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "pos_weight": pos_weight.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )

    eval_cmd = [
        args.python,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(out_dir),
        "--model_name",
        normalize_model_name(args.model_name),
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
        str(args.data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache",
        str(args.data_root / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--pn2021_mmap_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--pn2021_root",
        str(args.data_root / "physionet2021"),
        "--skip_mimic",
        "--exclude_ref_ids",
        str(paths["meta"]),
        "--report_drop_all_zero_pn2021",
        "--output_path",
        str(eval_path),
    ]
    if args.eval_pn2021_limit:
        eval_cmd.extend(["--pn2021_limit", str(args.eval_pn2021_limit)])
    run_cmd(eval_cmd, out_dir / "eval_full.log")
    return eval_path


def parse_eval(center: str, k: int, epochs: int, eval_path: Path) -> dict:
    with eval_path.open(encoding="utf-8") as f:
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


def write_summary(rows: list[dict], out_root: Path) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "direct_finetune_v7.csv"
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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[summary] wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--subset_seed", type=int, default=20260601)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--model_name", default="benchmark_resnet1d_wang", choices=available_model_names())
    p.add_argument("--init_ckpt", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--eval_min_pos", type=int, default=10)
    p.add_argument("--eval_pn2021_limit", type=int, default=0)
    p.add_argument("--data_root", type=Path, default=DATA_ROOT)
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--subset_root", type=Path, default=SUBSET_ROOT)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for center in args.centers:
        eval_path = train_one(center, args)
        rows.append(parse_eval(center, args.k, args.epochs, eval_path))
        write_summary(rows, args.out_root)
    write_summary(rows, args.out_root)


if __name__ == "__main__":
    main()
