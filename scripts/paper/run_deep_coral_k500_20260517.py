#!/usr/bin/env python3
"""Deep CORAL K=500 target-center adaptation baseline for Super5 PN2021.

This runner reuses the official THUML Transfer-Learning-Library CORAL loss
cloned under /root/autodl-tmp/external_repos/Transfer-Learning-Library and the
project's existing EfficientNet1DV2 / PTB-XL / PN2021 evaluation stack.

Protocol:
  * initialize from the PTB-XL Super5 EfficientNet1DV2 checkpoint;
  * source batches are labeled PTB-XL folds 1-8;
  * target batches are the same K=500 center records used by LH-AT, labels
    ignored;
  * optimize source masked BCE + lambda_coral * CORAL(penultimate features);
  * select checkpoint by PTB-XL fold9 AUPRC, then evaluate PN2021 with the
    K=500 target ref ids excluded from the corresponding target center.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc  # noqa: E402
from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PYTHON,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, NUM_SUPER5, get_scheme  # noqa: E402
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    compute_macro_auroc_auprc,
    compute_pos_weight,
    get_ptbxl_labels_for_scheme,
    masked_bce_with_logits,
)


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_uda_baselines_20260517/deep_coral_k500")
SUBSET_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
CORAL_FILE = Path("/root/autodl-tmp/external_repos/Transfer-Learning-Library/tllib/alignment/coral.py")


def load_official_coral_loss_class():
    """Load only THUML's standalone coral.py to avoid optional package deps."""
    spec = importlib.util.spec_from_file_location("thuml_coral_loss", CORAL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official CORAL file: {CORAL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CorrelationAlignmentLoss


class IndexedPTBXLDataset(Dataset):
    def __init__(self, signals_mmap, indices, labels, crop_len: int, mode: str) -> None:
        self.signals = signals_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)
        self.mode = mode

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        sig_tc = np.asarray(self.signals[int(self.indices[idx])], dtype=np.float32)
        crop = crop_signal_tc(
            sig_tc,
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


class TargetNPZDataset(Dataset):
    def __init__(self, signals: np.ndarray, crop_len: int, mode: str) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)
        self.mode = mode

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int):
        crop = crop_signal_tc(
            self.signals[idx],
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        return torch.from_numpy(np.ascontiguousarray(crop.T)).float()


def subset_paths(center: str, k: int, seed: int) -> dict[str, Path]:
    base_dir = SUBSET_ROOT / center / f"k{k}_seed{seed}"
    base = base_dir / f"{center}_real_k{k}_seed{seed}"
    return {
        "signals": base.with_suffix(".signals.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
    }


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
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def forward_feature_map(model: EfficientNet1DV2, x: torch.Tensor) -> torch.Tensor:
    x = model.initial_conv(x)
    x = model.features(x)
    x = model.final_conv(x)
    x = model.final_norm(x)
    return x


def pooled_features(feat_map: torch.Tensor) -> torch.Tensor:
    return torch.flatten(F.adaptive_avg_pool1d(feat_map, 1), 1)


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
    metrics = compute_macro_auroc_auprc(labels, probs, CLASS_NAMES_SUPER5, min_pos=10)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def make_loaders(args: argparse.Namespace):
    scheme = get_scheme("super5")
    label_cache_path = str(Path(args.out_root) / "_shared_ptbxl_labels")
    train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
        PTBXL_CSV, scheme, label_cache_path, folds=list(range(1, 9))
    )
    val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
        PTBXL_CSV, scheme, label_cache_path, folds=[9]
    )
    signals_mmap = np.load(PTBXL_PREP, mmap_mode="r")
    train_ds = IndexedPTBXLDataset(signals_mmap, train_idx, train_labels, args.crop_len, "train")
    val_ds = IndexedPTBXLDataset(signals_mmap, val_idx, val_labels, args.crop_len, "eval")
    return train_ds, val_ds, train_labels


def method_tag(args: argparse.Namespace) -> str:
    return f"deep_coral_lam{args.coral_lambda:g}".replace(".", "p")


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    tag = method_tag(args)
    out_dir = Path(args.out_root) / "runs" / (
        f"{center}_K{args.k}_{tag}_ep{args.epochs}_seed{args.seed}"
    )
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} {tag} already evaluated")
        return eval_path
    if args.dry_run:
        print(f"[dry-run] would train {center} {tag} into {out_dir}")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_train_ds, source_val_ds, train_labels = make_loaders(args)
    with np.load(paths["signals"], allow_pickle=True) as data:
        target_signals = data["signals"].astype(np.float32, copy=False)
    target_ds = TargetNPZDataset(target_signals, crop_len=args.crop_len, mode="train")

    source_loader = DataLoader(
        source_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        source_val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5, clip_max=args.pos_weight_clip_max),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits, y):
        return masked_bce_with_logits(logits, y, pos_weight)

    model = load_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    coral_loss = load_official_coral_loss_class()().to(device)

    with (out_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                **vars(args),
                "center": center,
                "method": "deep_coral",
                "method_tag": tag,
                "source_protocol": "PTB-XL folds 1-8 labeled",
                "target_protocol": "K=500 target-center refs, labels ignored",
                "checkpoint_selection": "PTB-XL fold9 macro AUPRC",
                "init_checkpoint": BASELINE_CKPT,
                "coral_loss_source": "/root/autodl-tmp/external_repos/Transfer-Learning-Library/tllib/alignment/coral.py",
                "subset_signals": str(paths["signals"]),
                "ref_meta": str(paths["meta"]),
                "class_names": list(CLASS_NAMES_SUPER5),
                "pos_weight": pos_weight.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )

    best_score = -float("inf")
    rows = []
    t0 = time.time()
    print(
        f"[data] center={center} source_train={len(source_train_ds)} "
        f"source_val={len(source_val_ds)} target={len(target_ds)} lambda={args.coral_lambda}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iter = itertools.cycle(target_loader)
        losses_cls, losses_coral, losses_total = [], [], []
        for x_s, y_s in source_loader:
            x_t = next(target_iter)
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feat_s_map = forward_feature_map(model, x_s)
                feat_t_map = forward_feature_map(model, x_t)
                logits_s = model.classifier(feat_s_map)
                loss_cls = criterion(logits_s, y_s)
            feat_s = pooled_features(feat_s_map).float()
            feat_t = pooled_features(feat_t_map).float()
            loss_coral = coral_loss(feat_s, feat_t)
            loss = loss_cls + float(args.coral_lambda) * loss_coral
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            losses_cls.append(float(loss_cls.detach().item()))
            losses_coral.append(float(loss_coral.detach().item()))
            losses_total.append(float(loss.detach().item()))

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        score = float(val_metrics["macro_auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses_total)),
            "train_cls_loss": float(np.mean(losses_cls)),
            "train_coral_loss": float(np.mean(losses_coral)),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
            "val_macro_auprc": float(val_metrics["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} cls={row['train_cls_loss']:.4f} "
            f"coral={row['train_coral_loss']:.5f} val={row['val_loss']:.4f} "
            f"auroc={row['val_macro_auroc']:.4f} auprc={row['val_macro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(rows, f, indent=2)

    with (out_dir / "train_result.json").open("w") as f:
        json.dump(
            {
                "center": center,
                "method": "deep_coral",
                "method_tag": tag,
                "best_val_macro_auprc": best_score,
                "epochs_trained": args.epochs,
                "n_source_train": int(len(source_train_ds)),
                "n_source_val": int(len(source_val_ds)),
                "n_target_unlabeled": int(len(target_ds)),
                "pos_weight": pos_weight.detach().cpu().tolist(),
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
        PTBXL_PREP,
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir",
        PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--exclude_ref_ids",
        str(paths["meta"]),
        "--output_path",
        str(eval_path),
    ]
    run_cmd(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def parse_eval(center: str, args: argparse.Namespace, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    return {
        "method": "deep_coral",
        "tag": method_tag(args),
        "center": center,
        "K": int(args.k),
        "epochs": int(args.epochs),
        "coral_lambda": float(args.coral_lambda),
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
    rows_sorted = sorted(rows, key=lambda r: (r["center"], r["tag"]))
    csv_path = summary_dir / "deep_coral_k500.csv"
    fields = [
        "method",
        "tag",
        "center",
        "K",
        "epochs",
        "coral_lambda",
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
        for row in rows_sorted:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[summary] wrote {csv_path}", flush=True)


def load_existing_rows(out_root: Path) -> list[dict]:
    rows = []
    for eval_path in sorted((out_root / "runs").glob("*/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json")):
        cfg_path = eval_path.parent / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            args_obj = argparse.Namespace(**cfg)
            rows.append(parse_eval(str(cfg["center"]), args_obj, eval_path))
        except Exception as exc:
            print(f"[warn] failed to parse {eval_path}: {exc}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--subset_seed", type=int, default=20260531)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--coral_lambda", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_root)
    for center in args.centers:
        eval_path = train_one(center, args)
        if not args.dry_run:
            rows = [r for r in rows if not (r["center"] == center and r["tag"] == method_tag(args))]
            rows.append(parse_eval(center, args, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
