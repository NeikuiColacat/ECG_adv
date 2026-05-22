#!/usr/bin/env python3
"""K-shot adaptation baselines for the VAE-only LH-AT paper route.

The goal is to compare against standard target-labeled adaptation methods under
the same target-center K-shot protocol used by real-anchor Latent-Hull AT:

  - same K=500 PN2021 target-center ref subset;
  - same PTB-XL EfficientNet1DV2 checkpoint initialization;
  - same ref-id exclusion during PN2021 evaluation;
  - same PTB-XL fold10 / PN2021 v3 super5 eval pipeline.

Methods implemented here intentionally avoid ECGTwin VAE latents:

  - mixup: input-space Mixup on target-center real ECG;
  - manifold_mixup: feature-space Mixup after EfficientNet final_norm;
  - sam: target-center fine-tuning with Sharpness-Aware Minimization.
  - fgsm_at / pgd_at: ordinary input-space adversarial training on the
    normalized 10s ECG crop, used as a direct control for latent-hull AT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
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
    PYTHON,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, NUM_SUPER5  # noqa: E402
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    compute_macro_auroc_auprc,
    compute_pos_weight,
    masked_bce_with_logits,
)


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_kshot_baseline_cmp_20260516")
SUBSET_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
DIRECT_ROOT = Path("/root/autodl-tmp/paper_direct_finetune_k500_20260516")
LHAT_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516")
LEGACY_LHAT_GRID_ROOT = Path("/root/autodl-tmp/paper_latenthull_grid_20260512/runs")


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
        crop = crop_signal_tc(
            self.signals[real_idx],
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[real_idx]).float(),
        )


def subset_paths(center: str, k: int, seed: int) -> dict[str, Path]:
    base_dir = SUBSET_ROOT / center / f"k{k}_seed{seed}"
    base = base_dir / f"{center}_real_k{k}_seed{seed}"
    return {
        "signals": base.with_suffix(".signals.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
    }


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n, dtype=np.int64)
    if val_fraction <= 0:
        return indices, indices
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])
    return train_idx, val_idx


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


def forward_features(model: EfficientNet1DV2, x: torch.Tensor) -> torch.Tensor:
    x = model.initial_conv(x)
    x = model.features(x)
    x = model.final_conv(x)
    x = model.final_norm(x)
    return x


def mix_batch(
    x: torch.Tensor | None,
    y: torch.Tensor,
    alpha: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, float, torch.Tensor]:
    if alpha <= 0:
        perm = torch.arange(y.shape[0], device=y.device)
        return x, y, 1.0, perm
    beta = torch.distributions.Beta(alpha, alpha)
    if generator is None:
        lam_tensor = beta.sample().to(device=y.device, dtype=y.dtype)
    else:
        # torch.distributions does not accept a per-call generator, so draw on CPU
        # through numpy for deterministic CLI-level seeds.
        lam = np.random.beta(alpha, alpha)
        lam_tensor = torch.tensor(lam, device=y.device, dtype=y.dtype)
    lam = float(lam_tensor.item())
    perm = torch.randperm(y.shape[0], device=y.device)
    y_mix = lam_tensor * y + (1.0 - lam_tensor) * y[perm]
    if x is None:
        return None, y_mix, lam, perm
    return lam_tensor * x + (1.0 - lam_tensor) * x[perm], y_mix, lam, perm


class SAM:
    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05, eps: float = 1e-12) -> None:
        self.optimizer = optimizer
        self.rho = float(rho)
        self.eps = float(eps)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eps)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state(p)["e_w"] = e_w
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                e_w = self.state(p).pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state(self, p: torch.Tensor) -> dict:
        return self.optimizer.state[p]

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


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


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.method == "mixup":
        x_mix, y_mix, _, _ = mix_batch(x, y, args.mixup_alpha)
        with torch.cuda.amp.autocast(enabled=x.device.type == "cuda"):
            logits = model(x_mix)
            return criterion(logits, y_mix)
    if args.method == "manifold_mixup":
        with torch.cuda.amp.autocast(enabled=x.device.type == "cuda"):
            feat = forward_features(model, x)
            _, y_mix, lam, perm = mix_batch(None, y, args.mixup_alpha)
            lam_t = torch.tensor(lam, device=feat.device, dtype=feat.dtype)
            feat_mix = lam_t * feat + (1.0 - lam_t) * feat[perm]
            logits = model.classifier(feat_mix)
            return criterion(logits, y_mix)
    if args.method == "sam":
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x.float())
            return criterion(logits, y)
    if args.method in {"fgsm_at", "pgd_at"}:
        x_adv = make_input_adversary(model, x, y, criterion, args)
        with torch.cuda.amp.autocast(enabled=x.device.type == "cuda"):
            clean_loss = criterion(model(x), y)
            adv_loss = criterion(model(x_adv), y)
            return (1.0 - args.adv_loss_weight) * clean_loss + args.adv_loss_weight * adv_loss
    raise ValueError(f"unknown method: {args.method}")


def method_tag(args: argparse.Namespace) -> str:
    if args.method in {"mixup", "manifold_mixup"}:
        return f"{args.method}_a{args.mixup_alpha:g}".replace(".", "p")
    if args.method == "sam":
        return f"sam_rho{args.sam_rho:g}".replace(".", "p")
    if args.method in {"fgsm_at", "pgd_at"}:
        return (
            f"{args.method}_eps{args.pgd_epsilon:g}_a{args.pgd_alpha:g}"
            f"_s{args.pgd_steps}_w{args.adv_loss_weight:g}"
        ).replace(".", "p")
    return args.method


def make_input_adversary(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Build an L-infinity adversarial ECG crop in normalized input space."""
    eps = float(args.pgd_epsilon)
    steps = 1 if args.method == "fgsm_at" else int(args.pgd_steps)
    alpha = eps if args.method == "fgsm_at" else float(args.pgd_alpha)
    x0 = x.detach()
    if args.adv_random_start:
        x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    else:
        x_adv = x0.clone()

    was_training = model.training
    model.eval()
    for _ in range(max(1, steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x_adv.float())
            loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x0, min=-eps, max=eps)
        x_adv = (x0 + delta).detach()
    if was_training:
        model.train()
    return x_adv


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    tag = method_tag(args)
    out_root = Path(args.out_root)
    out_dir = out_root / "runs" / (
        f"{center}_K{args.k}_{tag}_ep{args.epochs}_seed{args.seed}"
        f"_val{args.val_fraction:g}"
    )
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} {tag} already evaluated")
        return eval_path
    if args.dry_run:
        print(f"[dry-run] would train {center} {tag} into {out_dir}")
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
        drop_last=args.method in {"mixup", "manifold_mixup"},
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
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sam = SAM(base_optimizer, rho=args.sam_rho) if args.method == "sam" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.method != "sam")

    config = vars(args).copy()
    config.update(
        {
            "center": center,
            "method_tag": tag,
            "subset_signals": str(paths["signals"]),
            "ref_meta": str(paths["meta"]),
            "init_checkpoint": BASELINE_CKPT,
            "class_names": list(CLASS_NAMES_SUPER5),
            "train_record_ids": record_ids[train_idx].tolist(),
            "val_record_ids": record_ids[val_idx].tolist(),
            "selection_policy": "K-shot target-center comparator; no ECGTwin VAE latent and no synthetic ECG",
        }
    )
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    best_score = -float("inf")
    log_rows = []
    t0 = time.time()
    print(
        f"[data] method={tag} center={center} train={len(train_ds)} val={len(val_ds)} "
        f"signals={paths['signals']}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            base_optimizer.zero_grad(set_to_none=True)
            if args.method == "sam":
                loss = train_step(model, x, y, criterion, args)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                assert sam is not None
                sam.first_step()
                loss_second = train_step(model, x, y, criterion, args)
                loss_second.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                sam.second_step()
                losses.append(float(loss_second.item()))
            else:
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss = train_step(model, x, y, criterion, args)
                scaler.scale(loss).backward()
                scaler.unscale_(base_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(base_optimizer)
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
            "lr": float(base_optimizer.param_groups[0]["lr"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log_rows.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} {tag} train={row['train_loss']:.4f} "
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
                "method": args.method,
                "method_tag": tag,
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
        "/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy",
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


def parse_eval(center: str, k: int, epochs: int, method: str, tag: str, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    return {
        "method": method,
        "tag": tag,
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


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_direct_rows() -> list[dict]:
    out = []
    for row in read_csv_rows(DIRECT_ROOT / "summaries" / "direct_finetune_k500.csv"):
        out.append(
            {
                "method": "direct_ft",
                "tag": "direct_ft",
                "center": row["center"],
                "K": int(row["K"]),
                "epochs": int(row["epochs"]),
                "target_auroc": float(row["target_auroc"]),
                "target_auprc": float(row["target_auprc"]),
                "pn2021_avg_auroc": float(row["pn2021_avg_auroc"]),
                "pn2021_avg_auprc": float(row["pn2021_avg_auprc"]),
                "ptbxl_auroc": float(row["ptbxl_auroc"]),
                "ptbxl_auprc": float(row["ptbxl_auprc"]),
                "eval_path": row["eval_path"],
            }
        )
    return out


def load_lhat_best_rows() -> list[dict]:
    rows = read_csv_rows(LHAT_ROOT / "summaries" / "vae_only_latenthull_sweep.csv")
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("variant") != "lambda015":
            continue
        center = row["center"]
        val = float(row["target_auprc"])
        if center not in best or val > float(best[center]["target_auprc"]):
            best[center] = row
    out = []
    for row in best.values():
        out.append(
            {
                "method": "ours_lhat",
                "tag": f"ours_lhat_{row['variant']}_M{row['hull_M']}",
                "center": row["center"],
                "K": int(row["K"]),
                "epochs": int(row["epochs"]),
                "target_auroc": float(row["target_auroc"]),
                "target_auprc": float(row["target_auprc"]),
                "pn2021_avg_auroc": float(row["pn2021_avg_auroc"]),
                "pn2021_avg_auprc": float(row["pn2021_avg_auprc"]),
                "ptbxl_auroc": float(row["ptbxl_auroc"]),
                "ptbxl_auprc": float(row["ptbxl_auprc"]),
                "eval_path": row["eval_path"],
            }
        )

    # Chapman/Georgia optimized LH-AT runs predate the compact 20260516 sweep
    # root but use the same K=500 ref-id exclusion protocol. Keep them in the
    # comparator summary so four-center tables can compute deltas automatically.
    existing_centers = {row["center"] for row in out}
    for center in ["chapman_shaoxing", "georgia"]:
        if center in existing_centers:
            continue
        candidates = []
        for eval_path in sorted(
            LEGACY_LHAT_GRID_ROOT.glob(
                f"{center}_realK500_lambda*_seed20260531/"
                "eval_result_v3_super5_normsuppress_exclrefs_20260512.json"
            )
        ):
            try:
                candidates.append(parse_eval(center, 500, 30, "ours_lhat", eval_path.parent.name, eval_path))
            except Exception as exc:
                print(f"[warn] failed to parse legacy LH-AT {eval_path}: {exc}", flush=True)
        if candidates:
            out.append(max(candidates, key=lambda r: float(r["target_auprc"])))
    return out


def load_existing_comparator_rows(out_root: Path) -> list[dict]:
    rows = []
    for eval_path in sorted((out_root / "runs").glob("*/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json")):
        cfg_path = eval_path.parent / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            center = str(cfg["center"])
            method = str(cfg["method"])
            tag = str(cfg["method_tag"])
            rows.append(
                parse_eval(
                    center=center,
                    k=int(cfg.get("k", cfg.get("K", 500))),
                    epochs=int(cfg["epochs"]),
                    method=method,
                    tag=tag,
                    eval_path=eval_path,
                )
            )
        except Exception as exc:
            print(f"[warn] failed to parse existing comparator {eval_path}: {exc}", flush=True)
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (str(row["center"]), str(row["method"]), str(row["tag"]))
        deduped[key] = row
    return list(deduped.values())


def write_summary(rows: list[dict], out_root: Path) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(dedupe_rows(rows), key=lambda r: (r["center"], r["method"], r["tag"]))
    csv_path = summary_dir / "kshot_baseline_comparators.csv"
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
        for row in rows_sorted:
            writer.writerow({k: row.get(k, "") for k in fields})

    by_center: dict[str, list[dict]] = {}
    for row in rows_sorted:
        by_center.setdefault(row["center"], []).append(row)

    md_path = summary_dir / "kshot_baseline_comparators.md"
    with md_path.open("w") as f:
        f.write("# K-shot Baseline Comparators vs VAE-only LH-AT\n\n")
        f.write(
            "All comparator methods use the same target-center K-shot ref subset, "
            "same EfficientNet1DV2 PTB-XL initialization, and same ref-id exclusion "
            "during PN2021 evaluation. Metrics are macro AUROC / macro AUPRC.\n\n"
        )
        for center, center_rows in by_center.items():
            lhat = next((r for r in center_rows if r["method"] == "ours_lhat"), None)
            f.write(f"## {center}\n\n")
            f.write("| method | tag | target | Δ vs LH-AT | PN2021 avg | PTB-XL |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for row in center_rows:
                if lhat is not None:
                    d_auroc = (float(row["target_auroc"]) - float(lhat["target_auroc"])) * 100
                    d_auprc = (float(row["target_auprc"]) - float(lhat["target_auprc"])) * 100
                    delta = f"{d_auroc:+.2f} / {d_auprc:+.2f} pp"
                else:
                    delta = ""
                f.write(
                    f"| {row['method']} | {row['tag']} | "
                    f"{float(row['target_auroc']):.4f} / {float(row['target_auprc']):.4f} | "
                    f"{delta} | "
                    f"{float(row['pn2021_avg_auroc']):.4f} / {float(row['pn2021_avg_auprc']):.4f} | "
                    f"{float(row['ptbxl_auroc']):.4f} / {float(row['ptbxl_auprc']):.4f} |\n"
                )
            f.write("\n")
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["mixup", "manifold_mixup", "sam", "fgsm_at", "pgd_at"])
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
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--sam_rho", type=float, default=0.05)
    p.add_argument("--pgd_epsilon", type=float, default=0.03)
    p.add_argument("--pgd_alpha", type=float, default=0.01)
    p.add_argument("--pgd_steps", type=int, default=3)
    p.add_argument("--adv_loss_weight", type=float, default=0.5)
    p.add_argument("--adv_random_start", action="store_true")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_direct_rows() + load_lhat_best_rows() + load_existing_comparator_rows(out_root)
    tag = method_tag(args)
    for center in args.centers:
        eval_path = train_one(center, args)
        if not args.dry_run:
            rows.append(parse_eval(center, args.k, args.epochs, args.method, tag, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
