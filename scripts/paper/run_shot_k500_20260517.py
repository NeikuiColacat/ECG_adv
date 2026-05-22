#!/usr/bin/env python3
"""SHOT-style source-free K=500 adaptation baseline for Super5 PN2021.

Official reference repo:
  /root/autodl-tmp/external_repos/SHOT-plus

The official SHOT++ code targets softmax single-label image classification.  For
this repo's five-class multi-label ECG task, this runner keeps the comparable
parts of SHOT:

  * source-free target adaptation from a frozen source model;
  * freeze the source classifier head and adapt feature extractor parameters;
  * refresh target pseudo labels from the current model;
  * optimize pseudo-label BCE plus information-maximization loss.

Only the same K=500 target-center ref ECGs are used for adaptation; their labels
are ignored.  Evaluation excludes those K=500 ref ids from the target center.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc  # noqa: E402
from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PROJECT_ROOT,
    PTBXL_PREP,
    PYTHON,
)
from scripts.paper.run_deep_coral_k500_20260517 import load_model, subset_paths  # noqa: E402


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_source_free_baselines_20260517/shot_k500")


class IndexedTargetNPZDataset(Dataset):
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
        return torch.from_numpy(np.ascontiguousarray(crop.T)).float(), int(idx)


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


def configure_shot(model: nn.Module, update_bn_only: bool) -> list[nn.Parameter]:
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    params: list[nn.Parameter] = []
    if update_bn_only:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.requires_grad_(True)
                # Keep running stats materialized so the saved checkpoint remains
                # loadable by eval_crosscenter.py.  Train mode still uses batch
                # statistics and updates the running estimates during SHOT.
                module.track_running_stats = True
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        params.append(p)
    else:
        for name, p in model.named_parameters():
            if not name.startswith("classifier."):
                p.requires_grad_(True)
                params.append(p)
    if not params:
        raise RuntimeError("SHOT adaptation found no trainable parameters")
    return params


def bernoulli_entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return -(p * torch.log(p.clamp_min(eps)) + (1 - p) * torch.log((1 - p).clamp_min(eps)))


@torch.no_grad()
def refresh_pseudo_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    min_confidence: float,
) -> tuple[torch.Tensor, dict]:
    model.eval()
    probs_all = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        probs_all.append(torch.sigmoid(logits).float().cpu())
    probs = torch.cat(probs_all, dim=0)
    pseudo = (probs >= threshold).float()
    confidence = torch.maximum(probs, 1.0 - probs).mean(dim=1)
    reliable = confidence >= min_confidence
    stats = {
        "n_samples": int(probs.shape[0]),
        "n_reliable": int(reliable.sum().item()),
        "reliable_fraction": float(reliable.float().mean().item()),
        "positive_rate_per_class": pseudo.mean(dim=0).tolist(),
        "mean_prob_per_class": probs.mean(dim=0).tolist(),
    }
    return pseudo, reliable, stats


def shot_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    reliable: torch.Tensor,
    cls_weight: float,
    ent_weight: float,
    div_weight: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    loss = torch.zeros((), dtype=torch.float32, device=logits.device)
    if cls_weight > 0 and bool(reliable.any()):
        bce = F.binary_cross_entropy_with_logits(logits[reliable], pseudo[reliable], reduction="none")
        loss = loss + float(cls_weight) * bce.mean()
    if ent_weight > 0:
        ent = bernoulli_entropy_from_probs(probs).mean()
        loss = loss + float(ent_weight) * ent
    if div_weight > 0:
        marginal = probs.mean(dim=0)
        div_ent = bernoulli_entropy_from_probs(marginal).mean()
        loss = loss - float(div_weight) * div_ent
    return loss


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    tag = "shot_bn" if args.update_bn_only else "shot_feature"
    tag += f"_thr{args.threshold:g}_conf{args.min_confidence:g}".replace(".", "p")
    out_dir = Path(args.out_root) / "runs" / f"{center}_K{args.k}_{tag}_ep{args.epochs}_seed{args.seed}"
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} {tag} already evaluated")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    with np.load(paths["signals"], allow_pickle=True) as data:
        target_signals = data["signals"].astype(np.float32, copy=False)
    train_ds = IndexedTargetNPZDataset(target_signals, args.crop_len, "train")
    eval_ds = IndexedTargetNPZDataset(target_signals, args.crop_len, "eval")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = load_model(device)
    params = configure_shot(model, args.update_bn_only)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    cfg = {
        **vars(args),
        "center": center,
        "method": "shot_multilabel_k500",
        "method_tag": tag,
        "reference_repo": "/root/autodl-tmp/external_repos/SHOT-plus",
        "source_checkpoint": str(BASELINE_CKPT),
        "target_protocol": "K=500 target-center refs; labels ignored; evaluation excludes refs",
        "subset_signals": str(paths["signals"]),
        "ref_meta": str(paths["meta"]),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        pseudo, reliable, pseudo_stats = refresh_pseudo_labels(
            model, eval_loader, device, args.threshold, args.min_confidence
        )
        pseudo = pseudo.to(device)
        reliable = reliable.to(device)
        model.train()
        losses = []
        for x, idx in train_loader:
            x = x.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = shot_loss(
                    logits,
                    pseudo[idx],
                    reliable[idx],
                    cls_weight=args.cls_weight,
                    ent_weight=args.ent_weight,
                    div_weight=args.div_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "elapsed_sec": round(time.time() - t0, 1),
            **pseudo_stats,
        }
        rows.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} loss={row['train_loss']:.4f} "
            f"reliable={row['reliable_fraction']:.3f}",
            flush=True,
        )
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(rows, f, indent=2)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    with (out_dir / "train_result.json").open("w") as f:
        json.dump(
            {
                "center": center,
                "method": "shot_multilabel_k500",
                "method_tag": tag,
                "epochs_trained": int(args.epochs),
                "n_target_unlabeled": int(len(train_ds)),
                "last_epoch": rows[-1] if rows else {},
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
        "method": "shot_multilabel",
        "tag": Path(eval_path).parent.name,
        "center": center,
        "K": int(args.k),
        "epochs": int(args.epochs),
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
    csv_path = summary_dir / "shot_k500.csv"
    fields = [
        "method",
        "tag",
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
        for row in sorted(rows, key=lambda r: (r["center"], r["tag"])):
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
            rows.append(parse_eval(str(cfg["center"]), argparse.Namespace(**cfg), eval_path))
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
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_confidence", type=float, default=0.65)
    p.add_argument("--cls_weight", type=float, default=0.3)
    p.add_argument("--ent_weight", type=float, default=1.0)
    p.add_argument("--div_weight", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--update_bn_only", action="store_true")
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
            rows = [r for r in rows if not (r["center"] == center and Path(r["eval_path"]).parent.name == Path(eval_path).parent.name)]
            rows.append(parse_eval(center, args, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
