"""Self-distillation trainer for PTB-XL Super5 EfficientNet1DV2.

This keeps the graduate-project split/preprocessing protocol from
``ptbxl_source_train.py`` but adds a same-architecture teacher. Real PTB-XL samples use
hard labels plus teacher soft labels. Optional synthetic samples use teacher
soft labels only, which avoids trusting noisy synthetic hard labels directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model" / "DeepECG" / "notebooks"))
sys.path.insert(0, "/root/autodl-tmp/models/DeepECG/notebooks")

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from ecg_adv_gen.data.ptbxl import get_ptbxl_labels_for_scheme, preprocess_ptbxl_all  # noqa: E402
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc  # noqa: E402
from ecg_adv_gen.labels import get_super5_scheme  # noqa: E402
from ecg_adv_gen.preprocessing import crop_signal_tc  # noqa: E402
from ecg_adv_gen.training import compute_pos_weight, evaluate, init_weights, masked_bce_with_logits  # noqa: E402


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_model(device: str) -> EfficientNet1DV2:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    return model


def load_state_dict_into(model: nn.Module, ckpt: str, device: str) -> None:
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def normalize_synth_signals(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected signals ndim=3, got {arr.shape}")
    if arr.shape[1:] == (12, 1000):
        arr = arr.transpose(0, 2, 1)
    if arr.shape[1:] != (1000, 12):
        raise ValueError(f"Expected (N,1000,12) or (N,12,1000), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


class DistillDataset(Dataset):
    def __init__(
        self,
        signals_tc: np.ndarray,
        labels: np.ndarray,
        teacher_values: np.ndarray,
        crop_len: int,
        mode: str,
        has_hard: bool,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.signals = signals_tc.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.teacher_values = teacher_values.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)
        self.mode = mode
        self.has_hard = bool(has_hard)
        if sample_weights is None:
            sample_weights = np.ones((self.signals.shape[0],), dtype=np.float32)
        self.sample_weights = sample_weights.astype(np.float32, copy=False)
        if self.signals.shape[0] != self.labels.shape[0]:
            raise ValueError("signals/labels length mismatch")
        if self.teacher_values.shape != self.labels.shape:
            raise ValueError(f"teacher/target shape mismatch: {self.teacher_values.shape} vs {self.labels.shape}")
        if self.sample_weights.shape[0] != self.signals.shape[0]:
            raise ValueError("sample_weights/signals length mismatch")

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode="random" if self.mode == "train" else "center")
        sig_ct = np.ascontiguousarray(crop.T)
        has_hard = 1.0 if self.has_hard else 0.0
        return (
            torch.from_numpy(sig_ct).float(),
            torch.from_numpy(self.labels[idx]).float(),
            torch.from_numpy(self.teacher_values[idx]).float(),
            torch.tensor(has_hard, dtype=torch.float32),
            torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
        )


class EvalDataset(Dataset):
    def __init__(self, signals_tc: np.ndarray, labels: np.ndarray, crop_len: int):
        self.signals = signals_tc.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int):
        crop = crop_signal_tc(self.signals[idx], self.crop_len, mode="center")
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


@torch.no_grad()
def compute_teacher_logits(
    teacher: nn.Module,
    signals_tc: np.ndarray,
    crop_len: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    labels_dummy = np.zeros((signals_tc.shape[0], 5), dtype=np.float32)
    ds = EvalDataset(signals_tc, labels_dummy, crop_len=crop_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        persistent_workers=num_workers > 0,
    )
    teacher.eval()
    out: List[np.ndarray] = []
    for signals, _ in loader:
        signals = signals.to(device, non_blocking=True)
        logits = teacher(signals)
        out.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def soft_distill_loss(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    temperature: float,
    sample_weights: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "bce_soft":
        targets = teacher_values.clamp(0.0, 1.0)
        per_elem = F.binary_cross_entropy_with_logits(
            student_logits,
            targets,
            reduction="none",
        )
    elif mode == "temperature":
        t = float(temperature)
        teacher_prob = torch.sigmoid(teacher_values / t)
        per_elem = F.binary_cross_entropy_with_logits(
            student_logits / t,
            teacher_prob,
            reduction="none",
        ) * (t * t)
    else:
        raise ValueError(f"Unknown soft loss mode: {mode}")
    per_sample = per_elem.mean(dim=1)
    weights = sample_weights.float()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def hard_loss_real_only(
    logits: torch.Tensor,
    labels: torch.Tensor,
    has_hard: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    mask_rows = has_hard > 0.5
    if not bool(mask_rows.any()):
        return logits.sum() * 0.0
    return masked_bce_with_logits(logits[mask_rows], labels[mask_rows], pos_weight)


def load_split(path: str) -> Dict[str, List[int]]:
    with open(path, "r") as f:
        split = json.load(f)
    for key in ["train_indices", "val_indices", "test_indices"]:
        if key not in split:
            raise ValueError(f"split_json missing {key}")
    return split


def _logit_np(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32, copy=False)


def load_synth_npz(
    path: str,
    temperature: float,
    soft_loss_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if "signals" not in data:
        raise ValueError(f"{path} missing signals")
    label_key = "labels" if "labels" in data else "labels5"
    signals = normalize_synth_signals(data["signals"])
    labels = np.asarray(data[label_key], dtype=np.float32)
    if labels.shape[1] != 5:
        raise ValueError(f"Expected synth labels C=5, got {labels.shape}")
    sample_weights = None
    if "sample_weights" in data:
        sample_weights = np.asarray(data["sample_weights"], dtype=np.float32)

    teacher_values = None
    if soft_loss_mode == "bce_soft":
        if "soft_labels" in data:
            teacher_values = np.asarray(data["soft_labels"], dtype=np.float32)
        elif "teacher_probs" in data:
            teacher_values = np.asarray(data["teacher_probs"], dtype=np.float32)
        elif "soft_logits" in data:
            teacher_values = 1.0 / (1.0 + np.exp(-np.asarray(data["soft_logits"], dtype=np.float32)))
        elif "teacher_logits" in data:
            teacher_values = 1.0 / (1.0 + np.exp(-np.asarray(data["teacher_logits"], dtype=np.float32)))
    else:
        if "soft_logits" in data:
            teacher_values = np.asarray(data["soft_logits"], dtype=np.float32)
        elif "teacher_logits" in data:
            teacher_values = np.asarray(data["teacher_logits"], dtype=np.float32)
        elif "soft_labels" in data:
            teacher_values = _logit_np(np.asarray(data["soft_labels"], dtype=np.float32)) * float(temperature)
        elif "teacher_probs" in data:
            teacher_values = _logit_np(np.asarray(data["teacher_probs"], dtype=np.float32)) * float(temperature)
    if teacher_values is not None and teacher_values.shape != labels.shape:
        raise ValueError(f"synth teacher/soft target shape mismatch: {teacher_values.shape} vs {labels.shape}")
    return signals, labels, teacher_values, sample_weights


def build_loader(train_real: DistillDataset, train_synth: Optional[DistillDataset], args) -> DataLoader:
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=True,
    )
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    if train_synth is None:
        return DataLoader(train_real, shuffle=True, **kwargs)
    combo = ConcatDataset([train_real, train_synth])
    real_weight = np.ones(len(train_real), dtype=np.float64) / max(len(train_real), 1)
    synth_weight = (
        np.ones(len(train_synth), dtype=np.float64)
        * float(args.synth_ratio)
        / max(len(train_synth), 1)
    )
    sampler = WeightedRandomSampler(
        weights=np.concatenate([real_weight, synth_weight]),
        num_samples=len(train_real),
        replacement=True,
    )
    return DataLoader(combo, sampler=sampler, **kwargs)


def train(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    scheme = get_super5_scheme()
    label_cache = os.path.join(args.output_dir, "ptbxl_labels")
    all_idx, all_labels, _ = get_ptbxl_labels_for_scheme(args.csv_path, scheme, label_cache, folds=None)
    split = load_split(args.split_json)
    train_idx = [int(i) for i in split["train_indices"]]
    val_idx = [int(i) for i in split["val_indices"]]
    test_idx = [int(i) for i in split["test_indices"]]
    max_idx = len(all_idx) - 1
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if any(i < 0 or i > max_idx for i in idxs):
            raise ValueError(f"{name} split has index outside [0,{max_idx}]")
    shutil.copyfile(args.split_json, os.path.join(args.output_dir, "split.json"))

    all_sig = preprocess_ptbxl_all(
        args.data_path,
        args.cache_path,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )
    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])
    test_signals = np.asarray(all_sig[test_idx])
    train_labels = all_labels[train_idx].astype(np.float32)
    val_labels = all_labels[val_idx].astype(np.float32)
    test_labels = all_labels[test_idx].astype(np.float32)

    print(f"[data] real train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    print("[data] train positives: " + ", ".join(
        f"{c}={int((train_labels[:, j] == 1).sum())}" for j, c in enumerate(CLASS_NAMES)
    ), flush=True)

    teacher = build_model(args.device)
    load_state_dict_into(teacher, args.teacher_ckpt, args.device)
    teacher.eval()
    print(f"[teacher] loaded {args.teacher_ckpt}", flush=True)

    print("[teacher] computing real train targets", flush=True)
    train_teacher_values = compute_teacher_logits(
        teacher, train_signals, args.crop_len, args.device, args.eval_batch_size, args.num_workers
    )
    if args.soft_loss_mode == "bce_soft":
        train_teacher_values = 1.0 / (1.0 + np.exp(-np.clip(train_teacher_values, -50, 50)))

    synth_ds = None
    synth_count = 0
    if args.synth_npz:
        synth_signals, synth_labels, synth_teacher_values, synth_sample_weights = load_synth_npz(
            args.synth_npz,
            temperature=args.temperature,
            soft_loss_mode=args.soft_loss_mode,
        )
        if args.max_synth > 0 and synth_signals.shape[0] > args.max_synth:
            rng = np.random.default_rng(args.seed)
            pick = rng.choice(np.arange(synth_signals.shape[0]), size=args.max_synth, replace=False)
            pick.sort()
            synth_signals = synth_signals[pick]
            synth_labels = synth_labels[pick]
            if synth_teacher_values is not None:
                synth_teacher_values = synth_teacher_values[pick]
            if synth_sample_weights is not None:
                synth_sample_weights = synth_sample_weights[pick]
        synth_count = int(synth_signals.shape[0])
        if synth_teacher_values is None:
            print(f"[teacher] computing synthetic targets for {synth_count} samples", flush=True)
            synth_teacher_values = compute_teacher_logits(
                teacher, synth_signals, args.crop_len, args.device, args.eval_batch_size, args.num_workers
            )
            if args.soft_loss_mode == "bce_soft":
                synth_teacher_values = 1.0 / (1.0 + np.exp(-np.clip(synth_teacher_values, -50, 50)))
        else:
            print(f"[teacher] using precomputed synthetic targets for {synth_count} samples", flush=True)
        # Synthetic hard labels are deliberately not trusted by default.
        synth_hard_labels = synth_labels if args.use_synth_hard_labels else np.full_like(synth_labels, -1.0)
        synth_ds = DistillDataset(
            synth_signals,
            synth_hard_labels,
            synth_teacher_values,
            crop_len=args.crop_len,
            mode="train",
            has_hard=args.use_synth_hard_labels,
            sample_weights=synth_sample_weights,
        )

    train_ds = DistillDataset(
        train_signals,
        train_labels,
        train_teacher_values,
        crop_len=args.crop_len,
        mode="train",
        has_hard=True,
    )
    val_loader = DataLoader(
        EvalDataset(val_signals, val_labels, crop_len=args.crop_len),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        EvalDataset(test_signals, test_labels, crop_len=args.crop_len),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.device)
    if args.init_ckpt:
        load_state_dict_into(model, args.init_ckpt, args.device)
        print(f"[student] initialized from {args.init_ckpt}", flush=True)
    else:
        model.apply(init_weights)
        print("[student] initialized from scratch", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[student] params={n_params:,}", flush=True)

    pos_weight_np = compute_pos_weight(train_labels, 5, clip_max=args.pos_weight_clip_max)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=args.device)
    print("[loss] pos_weight: " + ", ".join(f"{c}={w:.2f}" for c, w in zip(CLASS_NAMES, pos_weight_np)), flush=True)

    def eval_criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler() if "cuda" in args.device else None

    best_val_auroc = -1.0
    best_val_auprc = -1.0
    best_selected = -1.0
    patience = 0
    log = []

    print(
        f"[train] epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"alpha={args.real_distill_alpha} T={args.temperature} "
        f"soft_mode={args.soft_loss_mode} synth={synth_count}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        loader = build_loader(train_ds, synth_ds, args)
        model.train()
        t0 = time.time()
        train_loss = 0.0
        hard_meter = 0.0
        soft_meter = 0.0
        n_batches = 0
        for signals, labels, teacher_values, has_hard, sample_weight in loader:
            signals = signals.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            teacher_values = teacher_values.to(args.device, non_blocking=True)
            has_hard = has_hard.to(args.device, non_blocking=True)
            sample_weight = sample_weight.to(args.device, non_blocking=True)
            base_distill_weights = torch.where(
                has_hard > 0.5,
                torch.full_like(has_hard, float(args.real_distill_alpha)),
                torch.full_like(has_hard, float(args.synth_distill_weight)),
            )
            distill_weights = base_distill_weights * sample_weight
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(signals)
                    hard = hard_loss_real_only(logits, labels, has_hard, pos_weight)
                    soft = soft_distill_loss(
                        logits,
                        teacher_values,
                        args.temperature,
                        distill_weights,
                        args.soft_loss_mode,
                    )
                    loss = float(args.hard_weight) * hard + soft
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(signals)
                hard = hard_loss_real_only(logits, labels, has_hard, pos_weight)
                soft = soft_distill_loss(
                    logits,
                    teacher_values,
                    args.temperature,
                    distill_weights,
                    args.soft_loss_mode,
                )
                loss = float(args.hard_weight) * hard + soft
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            train_loss += float(loss.item())
            hard_meter += float(hard.item())
            soft_meter += float(soft.item())
            n_batches += 1
        scheduler.step()

        val_loss, vy_true, vy_score = evaluate(model, val_loader, eval_criterion, args.device)
        val_metrics = compute_macro_auroc_auprc(vy_true, vy_score, CLASS_NAMES)
        selected = val_metrics[f"macro_{args.checkpoint_metric}"]
        improved_auroc = val_metrics["macro_auroc"] > best_val_auroc
        improved_auprc = val_metrics["macro_auprc"] > best_val_auprc
        improved_selected = selected > best_selected
        sd = None
        if improved_auroc or improved_auprc or improved_selected:
            sd = model.state_dict()
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        if improved_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            torch.save(sd, os.path.join(args.output_dir, "best_model_auroc.pt"))
        if improved_auprc:
            best_val_auprc = val_metrics["macro_auprc"]
            torch.save(sd, os.path.join(args.output_dir, "best_model_auprc.pt"))
        if improved_selected:
            best_selected = selected
            patience = 0
            torch.save(sd, os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience += 1

        entry = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "hard_loss": hard_meter / max(n_batches, 1),
            "soft_loss": soft_meter / max(n_batches, 1),
            "val_loss": float(val_loss),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
            "val_macro_auprc": float(val_metrics["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": round(time.time() - t0, 1),
        }
        log.append(entry)
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2, default=_json_default)
        marker = "*" if improved_selected else ""
        print(
            f"[train] ep {epoch:03d}/{args.epochs} loss={entry['train_loss']:.5f} "
            f"hard={entry['hard_loss']:.5f} soft={entry['soft_loss']:.5f} "
            f"val_auroc={entry['val_macro_auroc']:.4f} "
            f"val_auprc={entry['val_macro_auprc']:.4f} "
            f"lr={entry['lr']:.6g} {entry['time_sec']:.0f}s{marker}",
            flush=True,
        )
        if patience >= args.patience:
            print(f"[train] early stop at epoch {epoch} patience={args.patience}", flush=True)
            break

    ckpt = os.path.join(args.output_dir, f"best_model_{args.checkpoint_metric}.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.output_dir, "best_model.pt")
    load_state_dict_into(model, ckpt, args.device)
    test_loss, ty_true, ty_score = evaluate(model, test_loader, eval_criterion, args.device)
    test_metrics = compute_macro_auroc_auprc(ty_true, ty_score, CLASS_NAMES)
    result = {
        "scheme": "super5",
        "class_names": CLASS_NAMES,
        "test_loss": float(test_loss),
        "test_macro_auroc": float(test_metrics["macro_auroc"]),
        "test_macro_auprc": float(test_metrics["macro_auprc"]),
        "test_per_class": test_metrics["per_class"],
        "best_val_macro_auroc": float(best_val_auroc),
        "best_val_macro_auprc": float(best_val_auprc),
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_path": ckpt,
        "epochs_trained": len(log),
        "pos_weight": pos_weight_np.tolist(),
        "config": vars(args),
        "split": split,
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(
        f"[done] test AUROC={result['test_macro_auroc']:.4f} "
        f"AUPRC={result['test_macro_auprc']:.4f} saved={args.output_dir}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--init_ckpt", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_path", default="/root/autodl-tmp/ptbxl/raw100.npy")
    p.add_argument("--csv_path", default="/root/autodl-tmp/ptbxl/ptbxl_database.csv")
    p.add_argument("--cache_path", default="/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy")
    p.add_argument("--split_json", required=True)
    p.add_argument("--preprocess_mode", default="minimal_resample",
                   choices=["minimal_resample", "legacy_ecgfounder_filter", "raw_for_generation_or_digital"])
    p.add_argument("--norm_mode", default="per_sample_global", choices=["per_sample_global", "none"])
    p.add_argument("--synth_npz", default="")
    p.add_argument("--synth_ratio", type=float, default=0.5)
    p.add_argument("--max_synth", type=int, default=0)
    p.add_argument("--use_synth_hard_labels", action="store_true")
    p.add_argument("--hard_weight", type=float, default=1.0)
    p.add_argument("--real_distill_alpha", type=float, default=0.3)
    p.add_argument("--synth_distill_weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--soft_loss_mode", choices=["temperature", "bce_soft"], default="temperature")
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--cosine_tmax", type=int, default=15)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--checkpoint_metric", choices=["auroc", "auprc"], default="auprc")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7042)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
