"""Online AT on Super5 victim with ECGTwin VAE latent-hull anchors.

Each epoch:
  A) Sample K_anchor latents (stratified by class) from a frozen Stage-1 synth
     pool (.latent.npz).
  B) Search same-label latent-hull hard samples with the current victim.
  C) Run gates per epoch (Plan Rev 6 Issue #38):
       - ASR gate     : compute_asr → require asr_overall ≥ 0.30 over a sliding
                         3-epoch window (else raise — PGD is broken)
       - Semantic gate: compute_semantic_gate → if Einthoven p95 / HR / QRS
                         ratio fail, do not push the (otherwise high-confidence)
                         garbage into the buffer.
  D) On gate pass, push (signal_ct_250, target_label_5_with_-1_sentinel, score)
     into a QualityAwareBuffer (FIFO + informativeness eviction).
  E) Build a mixed DataLoader with three streams:
       PTBXL real  weight 1.0
      adv buffer  weight 2.0    (cold-start guard: weight=0 in epoch 0/empty)
  F) train_one_epoch with masked BCE on the -1 sentinel + EWA anchor regularizer
     (Plan Issue #21 Q4 — ADR ICLR 2024 EMA self-distill).
  G) save the last checkpoint for the managed ref-excluded PN2021/PN2021-C
     evaluation jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import (
    DataLoader, ConcatDataset, Dataset, TensorDataset, WeightedRandomSampler,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEEPECG_NOTEBOOKS = Path(
    os.environ.get(
        "DEEPECG_NOTEBOOKS",
        str(PROJECT_ROOT / "model" / "DeepECG" / "notebooks"),
    )
)
sys.path.insert(0, str(Path("/root/autodl-tmp/models/DeepECG/notebooks")))
sys.path.insert(0, str(DEEPECG_NOTEBOOKS))

from adversarial.adv_validation import compute_asr, compute_semantic_gate  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, TIERM_INPUT_LENGTH,
)
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from methods.augmix.jsd_loss import jsd_multilabel  # noqa: E402
from methods.augmix.severity import AVAILABLE_OPS, build_op  # noqa: E402
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    STRESS_PROFILE_CHOICES as PN2021C_STRESS_PROFILE_CHOICES,
    apply_corruption_sequence,
    build_corruption_op as _build_pn2021c_corruption_op,
    file_sha256,
)
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc  # noqa: E402
from ecg_adv_gen.evaluation.selection import (  # noqa: E402
    build_f004_training_record,
    build_matched_training_record,
    update_matched_checkpoint_selection,
)
from ecg_adv_gen.data.kshot import filter_latent_candidates, matched_k500_split  # noqa: E402

from ecg_adv_gen.training.online_buffer import (  # noqa: E402
    QualityAwareBuffer,
    center_crop_ct,
    train_one_epoch_grouped_target_bce,
    train_one_epoch_masked_bce,
)
from ecg_adv_gen.training.resume_contract import (  # noqa: E402
    LOCKED_LATENT_AUGMIX_SIGNAL_SPACE,
)
from ecg_adv_gen.matched_effnet import (  # noqa: E402
    MATCHED_EFFNET_ARMS,
    MATCHED_EFFNET_CONTRACT_VERSION,
    MATCHED_EFFNET_THIRD_CHAIN_ROUTES,
    f004_identity,
    is_f004_rho_sweep,
    is_matched_effnet_arm,
    is_paper_matched_effnet_run,
    matched_effnet_arm,
    validate_f004_runtime,
    validate_matched_effnet_runtime,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, NUM_SUPER5, get_super5_scheme  # noqa: E402
from ecg_adv_gen.labels.super5_mapping import SUPER5_TO_IDX  # noqa: E402
from ecg_adv_gen.models.super5_model_zoo import available_model_names  # noqa: E402
from ecg_adv_gen.training import (  # noqa: E402
    attack_success_stats,
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    compute_pos_weight,
    masked_bce_with_logits,
    quality_buffer_state,
    resolve_resume_path,
    restore_quality_buffer_state,
    restore_rng_state,
    validate_resume_contract,
    PTBXLDatasetScheme,
    evaluate,
)
from ecg_adv_gen.data.latent_pools import (  # noqa: E402
    LatentPoolError,
    load_synth_pool as _load_synth_pool,
)
from ecg_adv_gen.data.ptbxl import get_ptbxl_labels_for_scheme, preprocess_ptbxl_all  # noqa: E402
from ecg_adv_gen.preprocessing import crop_signal_tc  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    agent_attack_decision,
    build_anchor_preserving_soft_labels,
    build_adv_buffer_label,
    build_three_chain_vae_lhat_augmix_views as _build_three_chain_vae_lhat_augmix_views_core,
    derive_kshot_anchor_class_weights,
    decoded_signal_invalid_stats,
    parse_class_weight_map,
    weighted_anchor_quotas,
)

DEFAULT_PTBXL_RAW = "/root/autodl-tmp/ptbxl/raw100.npy"
DEFAULT_PTBXL_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_PTBXL_PREP = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"


# Plan Rev 11/13: synth scope narrowed to 3 classes — HYP/CD synth disabled
# because their digital-GT validation fails 0/3 best-cell.
SUPER5_GEN_SUBSET = {"NORM", "MI", "STTC"}

# Indices of in-scope generation classes (NORM/MI/STTC) in the 5-class scheme.
SUPER5_GEN_SUBSET_IDX = sorted(SUPER5_TO_IDX[c] for c in SUPER5_GEN_SUBSET)


def _per_sample_global_zscore_np(signal_tc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    signal_tc = np.asarray(signal_tc, dtype=np.float32)
    mean = float(np.mean(signal_tc))
    std = float(np.std(signal_tc))
    return ((signal_tc - mean) / (std + eps)).astype(np.float32, copy=False)


def _crop_signal_pair_tc(
    clean_tc: np.ndarray,
    raw_tc: np.ndarray,
    crop_len: int,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if clean_tc.shape != raw_tc.shape:
        raise ValueError(f"paired clean/raw crop shape mismatch: {clean_tc.shape} vs {raw_tc.shape}")
    time_len = int(clean_tc.shape[0])
    crop_len = int(crop_len)
    if time_len < crop_len:
        pad_shape = (crop_len - time_len, clean_tc.shape[1])
        clean_pad = np.zeros(pad_shape, dtype=clean_tc.dtype)
        raw_pad = np.zeros(pad_shape, dtype=raw_tc.dtype)
        return (
            np.concatenate([clean_tc, clean_pad], axis=0),
            np.concatenate([raw_tc, raw_pad], axis=0),
        )
    max_start = time_len - crop_len
    if mode == "random" and max_start > 0:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = max_start // 2
    return clean_tc[start:start + crop_len, :], raw_tc[start:start + crop_len, :]


class TargetRealWaveformDataset(Dataset):
    """Target-center supervised ECG stream with explicit normalization contract."""

    def __init__(
        self,
        signals_1000: np.ndarray,
        labels: np.ndarray,
        *,
        crop_len: int = TIERM_INPUT_LENGTH,
        mode: str = "train",
        norm_mode: str = "pre_zscored",
    ) -> None:
        signals = np.asarray(signals_1000, dtype=np.float32)
        if signals.ndim != 3 or signals.shape[1:] != (1000, 12):
            raise ValueError(f"signals_1000 must be shaped (N,1000,12), got {signals.shape}")
        if norm_mode not in {"pre_zscored", "per_sample_global"}:
            raise ValueError(f"unknown target-real norm_mode={norm_mode!r}")
        self.signals = signals
        self.labels = np.asarray(labels, dtype=np.float32)
        self.crop_len = int(crop_len)
        self.mode = str(mode)
        self.norm_mode = str(norm_mode)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig_tc = self.signals[idx]
        if self.norm_mode == "per_sample_global":
            sig_tc = _per_sample_global_zscore_np(sig_tc)
        crop = crop_signal_tc(
            sig_tc,
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        sig_ct = crop.T
        return (
            torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


def _build_primary_adv_dataset(
    signals_ct: np.ndarray,
    labels: np.ndarray,
    *,
    crop_len: int,
) -> TensorDataset:
    """Build the primary adversarial stream at the classifier input length."""
    signals = np.asarray(signals_ct, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    crop_len = int(crop_len)
    if signals.ndim != 3 or signals.shape[1] != 12:
        raise ValueError(f"primary adversarial signals must be (N,12,T), got {signals.shape}")
    if labels_arr.shape[0] != signals.shape[0]:
        raise ValueError(
            "primary adversarial labels mismatch: "
            f"signals={signals.shape} labels={labels_arr.shape}"
        )
    if crop_len <= 0:
        raise ValueError("primary adversarial crop_len must be positive")
    cropped = np.stack(
        [center_crop_ct(signal, crop_len) for signal in signals],
        axis=0,
    ).astype(np.float32, copy=False)
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(cropped)).float(),
        torch.from_numpy(np.ascontiguousarray(labels_arr)).float(),
    )


class TargetRealRawFirstCorruptionDataset(Dataset):
    """Target K500 stream: clean model input plus raw-pre-zscore corruption base."""

    def __init__(
        self,
        signals_1000: np.ndarray,
        labels: np.ndarray,
        *,
        crop_len: int = TIERM_INPUT_LENGTH,
        mode: str = "train",
        clean_norm_mode: str = "per_sample_global",
    ) -> None:
        signals = np.asarray(signals_1000, dtype=np.float32)
        if signals.ndim != 3 or signals.shape[1:] != (1000, 12):
            raise ValueError(f"signals_1000 must be shaped (N,1000,12), got {signals.shape}")
        if clean_norm_mode not in {"pre_zscored", "per_sample_global"}:
            raise ValueError(f"unknown clean_norm_mode={clean_norm_mode!r}")
        self.signals = signals
        self.labels = np.asarray(labels, dtype=np.float32)
        self.crop_len = int(crop_len)
        self.mode = str(mode)
        self.clean_norm_mode = str(clean_norm_mode)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_tc = self.signals[idx]
        clean_crop, raw_crop = _crop_signal_pair_tc(
            raw_tc,
            raw_tc,
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        if self.clean_norm_mode == "per_sample_global":
            clean_crop = _per_sample_global_zscore_np(clean_crop)
        return (
            torch.from_numpy(np.ascontiguousarray(clean_crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
            torch.from_numpy(np.ascontiguousarray(raw_crop.T)).float(),
        )


def _adjust_channel_dependency_np(ecg: np.ndarray) -> np.ndarray:
    ecg[2] = ecg[1] - ecg[0]
    ecg[3] = -(ecg[1] + ecg[0]) / 2
    ecg[4] = ecg[0] - ecg[1] / 2
    ecg[5] = ecg[1] - ecg[0] / 2
    return ecg


def _standard_augmix_op_np(sig_ct: np.ndarray, op_name: str, severity: int) -> np.ndarray:
    """NumPy equivalent of methods.augmix severity ops for standard profiles."""

    op = build_op(op_name, int(severity))
    out = np.asarray(sig_ct, dtype=np.float32).copy()
    if getattr(op, "p", 1.0) <= np.random.uniform(0, 1):
        return out
    csz, tsz = out.shape
    if op_name == "powerline_noise":
        amp = np.random.uniform(op.min_amplitude, op.max_amplitude, size=(1, 1))
        f = 50 if np.random.uniform(0, 1) > 0.5 else 60
        t = np.linspace(0, tsz - 1, tsz)
        phase = np.random.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * f * (t / op.freq) + phase)
        out = out + noise * amp
        if getattr(op, "dependency", False):
            out = _adjust_channel_dependency_np(out)
    elif op_name == "emg_noise":
        amp = np.random.uniform(op.min_amplitude, op.max_amplitude, size=(csz, 1))
        noise = np.random.normal(0, 1, [csz, tsz])
        out = out + noise * amp
        if getattr(op, "dependency", False):
            out = _adjust_channel_dependency_np(out)
    elif op_name == "baseline_shift":
        shift_length = tsz * op.shift_ratio
        amp_channel = np.random.choice([1, -1], size=(csz, 1))
        amp_general = np.random.uniform(op.min_amplitude, op.max_amplitude, size=(1, 1))
        amp = amp_channel - amp_general
        noise = np.zeros(shape=(csz, tsz), dtype=np.float32)
        for _ in range(op.num_segment):
            segment_len = np.random.normal(shift_length, shift_length * 0.2)
            t0 = int(np.random.uniform(0, tsz - segment_len))
            t = int(t0 + segment_len)
            noise[np.arange(csz), t0:t] = 1
        out = out + noise * amp
        if getattr(op, "dependency", False):
            out = _adjust_channel_dependency_np(out)
    elif op_name == "baseline_wander":
        amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
        amp_general = np.random.uniform(op.min_amplitude, op.max_amplitude, size=op.k)
        noise = np.zeros(shape=(1, tsz), dtype=np.float32)
        t = np.linspace(0, tsz - 1, tsz)
        for k in range(op.k):
            f = np.random.uniform(op.min_freq, op.max_freq)
            phase = np.random.uniform(0, 2 * np.pi)
            noise += np.cos(2 * np.pi * f * (t / op.freq) + phase) * amp_general[k]
        out = out + (noise * amp_channel).astype(np.float32)
        if getattr(op, "dependency", False):
            out = _adjust_channel_dependency_np(out)
    elif op_name == "random_leads_masking":
        new_sample = np.zeros_like(out)
        if op.mask_leads_selection == "random":
            if op.max_masked_leads is None:
                survivors = np.random.uniform(0, 1, size=12) >= op.mask_leads_prob
            else:
                max_masked = min(12, max(0, int(op.max_masked_leads)))
                min_masked = min(max_masked, max(0, int(op.min_masked_leads)))
                survivors = np.ones(12, dtype=bool)
                if max_masked > 0:
                    n_masked = int(np.random.randint(min_masked, max_masked + 1))
                    if n_masked > 0:
                        masked = np.random.choice(np.arange(12), size=n_masked, replace=False)
                        survivors[masked] = False
            new_sample[survivors] = out[survivors]
            out = new_sample
        else:
            raise ValueError("standard NumPy fast path only supports random lead masking")
    else:
        raise ValueError(f"unknown op: {op_name}")
    return out.astype(np.float32, copy=False)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_float_sequence(value: str | None) -> List[float] | None:
    if value is None or value == "":
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def evaluate_loader_macro(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    def masked_criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return masked_bce_with_logits(logits, labels, criterion.pos_weight)

    loss, labels_np, probs = evaluate(model, loader, masked_criterion, device)
    metrics = compute_macro_auroc_auprc(labels_np, probs, CLASS_NAMES_SUPER5, min_pos=1)
    return {
        "loss": float(loss),
        "macro_auroc": float(metrics["macro_auroc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
    }


def update_best_selection_state(
    best_metric: float,
    best_epoch: int,
    best_source_floor_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
    *,
    epoch: int,
) -> tuple[float, int, Dict[str, Any]]:
    return (
        (float(candidate_result["candidate_metric"]), int(epoch), dict(candidate_result))
        if candidate_result["selected"]
        else (best_metric, best_epoch, best_source_floor_result)
    )


def restore_best_selection_state(
    checkpoint: Dict[str, Any],
    fallback: tuple[float, int, Dict[str, Any]],
) -> tuple[float, int, Dict[str, Any]]:
    metric, epoch, result = fallback
    return (float(checkpoint.get("best_metric", metric)), int(checkpoint.get("best_epoch", epoch)),
            dict(checkpoint.get("best_source_floor_result", result)))


def run_zero_effect_auxiliary_optimizer_control(
    optimizer: torch.optim.Optimizer,
    *,
    n_samples: int,
    batch_size: int,
    max_batches: int = 0,
) -> Dict[str, Any]:
    """Consume a matched auxiliary step budget without model or parameter updates."""

    n_samples = max(0, int(n_samples))
    batch_size = max(1, int(batch_size))
    n_batches = math.ceil(n_samples / batch_size) if n_samples else 0
    if int(max_batches) > 0:
        n_batches = min(n_batches, int(max_batches))
    for _ in range(n_batches):
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
    return {
        "enabled": True,
        "control": "zero_effect_optimizer_control",
        "loss": 0.0,
        "bce_loss": 0.0,
        "consistency_loss": 0.0,
        "n_batches": int(n_batches),
    }


def should_run_zero_effect_auxiliary_control(
    *, matched_comparison: bool, consistency_weight: float, bce_weight: float
) -> bool:
    return bool(matched_comparison) and consistency_weight <= 0.0 and bce_weight <= 0.0


def train_latent_augmix_consistency_epoch(
    model: nn.Module,
    clean_signals_ct: np.ndarray,
    augmix_signals_ct: np.ndarray,
    labels_np: np.ndarray,
    optimizer: AdamW,
    criterion: nn.Module,
    device: str,
    *,
    copies: int,
    consistency_weight: float,
    bce_weight: float,
    consistency_loss: str,
    batch_size: int,
    crop_len: int,
    grad_clip: float,
    trainable_params: List[nn.Parameter],
    max_batches: int = 0,
) -> Dict[str, Any]:
    """Train directly on latent-AugMix views generated for the current epoch."""
    if copies <= 0 or (consistency_weight <= 0 and bce_weight <= 0):
        return {
            "enabled": False,
            "reason": "disabled_or_zero_weight",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "n_batches": 0,
            "n_generated": 0,
        }
    consistency_loss = str(consistency_loss)
    if consistency_loss not in {"soft_bce", "jsd"}:
        raise ValueError(f"unknown latent AugMix consistency loss: {consistency_loss}")
    if consistency_loss == "jsd" and int(copies) < 2:
        raise ValueError("--latent_augmix_consistency_loss jsd requires --latent_augmix_copies >= 2")

    clean_np = np.asarray(clean_signals_ct, dtype=np.float32)
    aug_np = np.asarray(augmix_signals_ct, dtype=np.float32)
    labels_arr = np.asarray(labels_np, dtype=np.float32)
    n = int(clean_np.shape[0])
    if n == 0 or aug_np.shape[0] == 0:
        return {
            "enabled": True,
            "reason": "empty_views",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "n_batches": 0,
            "n_generated": int(aug_np.shape[0]),
            "copies": int(copies),
            "consistency_weight": float(consistency_weight),
            "consistency_objective": consistency_loss,
            "bce_weight": float(bce_weight),
        }
    expected = int(copies) * n
    if aug_np.shape[0] != expected:
        raise ValueError(f"latent AugMix view count mismatch: got {aug_np.shape[0]}, expected {expected}")
    if labels_arr.shape[0] != n:
        raise ValueError(f"label count mismatch: got {labels_arr.shape[0]}, expected {n}")

    start = max(0, (clean_np.shape[-1] - int(crop_len)) // 2)
    stop = start + int(crop_len)
    clean_np = clean_np[..., start:stop]
    aug_views_np = aug_np.reshape(int(copies), n, *aug_np.shape[1:])[..., start:stop]
    labels_t = torch.from_numpy(labels_arr).float()

    losses: List[float] = []
    bce_losses: List[float] = []
    consistency_losses: List[float] = []
    batch_size = max(1, int(batch_size))
    order = np.arange(n)

    for batch_i, lo in enumerate(range(0, n, batch_size), start=1):
        idx = order[lo:lo + batch_size]
        clean = torch.from_numpy(clean_np[idx]).float().to(device, non_blocking=True)
        labels = labels_t[idx].to(device, non_blocking=True)
        views = torch.from_numpy(aug_views_np[:, idx].reshape(-1, *aug_views_np.shape[2:])).float().to(
            device,
            non_blocking=True,
        )
        labels_rep = labels.repeat((int(copies), 1))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if consistency_loss == "soft_bce":
            with torch.no_grad():
                soft_targets = torch.sigmoid(model(clean)).detach()
            logits = model(views)
            soft_rep = soft_targets.repeat((int(copies), 1))
            mask = (labels_rep >= 0).float()
            denom = mask.sum().clamp(min=1.0)
            hard_bce = masked_bce_with_logits(logits, labels_rep, criterion.pos_weight)
            direct_consistency = (
                F.binary_cross_entropy_with_logits(logits, soft_rep, reduction="none") * mask
            ).sum() / denom
        else:
            clean_logits = model(clean)
            logits = model(views)
            logits_views = logits.view(int(copies), clean.shape[0], -1)
            aug_hard_bce = masked_bce_with_logits(logits, labels_rep, criterion.pos_weight)
            hard_bce = aug_hard_bce
            direct_consistency = torch.stack(
                [
                    jsd_multilabel(
                        clean_logits,
                        logits_views[copy_i],
                        logits_views[(copy_i + 1) % int(copies)],
                    )
                    for copy_i in range(int(copies))
                ]
            ).mean()
        loss = float(bce_weight) * hard_bce + float(consistency_weight) * direct_consistency
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        bce_losses.append(float(hard_bce.item()))
        consistency_losses.append(float(direct_consistency.item()))
        if max_batches > 0 and batch_i >= max_batches:
            break

    return {
        "enabled": True,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else float("nan"),
        "n_batches": len(losses),
        "n_generated": int(aug_np.shape[0]),
        "copies": int(copies),
        "consistency_weight": float(consistency_weight),
        "consistency_objective": consistency_loss,
        "bce_weight": float(bce_weight),
        "max_batches": int(max_batches),
    }


def _walker_state(walker: StratifiedPoolWalker) -> Dict[str, Any]:
    return {
        "cls_pools": {k: v.copy() for k, v in walker.cls_pools.items()},
        "cursors": dict(walker.cursors),
        "epochs_completed": dict(walker.epochs_completed),
        "rng_state": walker.rng.bit_generator.state,
        "source_cls_pools": {f"{k[0]}::{k[1]}": v.copy() for k, v in walker.source_cls_pools.items()},
        "source_cursors": {f"{k[0]}::{k[1]}": v for k, v in walker.source_cursors.items()},
        "source_epochs_completed": {
            f"{k[0]}::{k[1]}": v for k, v in walker.source_epochs_completed.items()
        },
        "last_source_counts": dict(walker.last_source_counts),
        "last_class_source_counts": dict(walker.last_class_source_counts),
    }


def _restore_walker_state(walker: StratifiedPoolWalker, state: Dict[str, Any]) -> None:
    if not state:
        return
    walker.cls_pools = {k: np.asarray(v, dtype=np.int64) for k, v in state.get("cls_pools", {}).items()}
    walker.cursors = {k: int(v) for k, v in state.get("cursors", {}).items()}
    walker.epochs_completed = {k: int(v) for k, v in state.get("epochs_completed", {}).items()}
    if "rng_state" in state:
        walker.rng.bit_generator.state = state["rng_state"]

    def split_key(raw: str) -> Tuple[str, str]:
        left, right = raw.split("::", 1)
        return left, right

    walker.source_cls_pools = {
        split_key(k): np.asarray(v, dtype=np.int64)
        for k, v in state.get("source_cls_pools", {}).items()
    }
    walker.source_cursors = {
        split_key(k): int(v)
        for k, v in state.get("source_cursors", {}).items()
    }
    walker.source_epochs_completed = {
        split_key(k): int(v)
        for k, v in state.get("source_epochs_completed", {}).items()
    }
    walker.last_source_counts = dict(state.get("last_source_counts", {}))
    walker.last_class_source_counts = dict(state.get("last_class_source_counts", {}))


@torch.no_grad()
def attack_bce_diagnostics(
    model: nn.Module,
    clean_signals_ct: np.ndarray,
    initial_signals_ct: Optional[np.ndarray],
    adv_signals_ct: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    crop_len: int,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if adv_signals_ct.size == 0:
        return {}

    def logits_for(signals: np.ndarray) -> torch.Tensor:
        start = max(0, (signals.shape[-1] - crop_len) // 2)
        cropped = signals[..., start:start + crop_len]
        chunks: List[torch.Tensor] = []
        for i in range(0, cropped.shape[0], batch_size):
            x = torch.from_numpy(cropped[i:i + batch_size]).float().to(device)
            chunks.append(model(x).detach().cpu())
        return torch.cat(chunks, dim=0)

    model.eval()
    clean_logits = logits_for(clean_signals_ct)
    initial_logits = logits_for(initial_signals_ct) if initial_signals_ct is not None else None
    adv_logits = logits_for(adv_signals_ct)
    labels_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    attack_vs_anchor = attack_success_stats(clean_logits, adv_logits, labels_t, margin=0.0)
    attack_vs_init = (
        attack_success_stats(initial_logits, adv_logits, labels_t, margin=0.0)
        if initial_logits is not None
        else None
    )
    return {
        "atk_anchor": attack_vs_anchor["success_rate"],
        "atk_init": None if attack_vs_init is None else attack_vs_init["success_rate"],
        "attack_vs_anchor": attack_vs_anchor,
        "attack_vs_init": attack_vs_init,
        "clean_bce": attack_vs_anchor["clean_loss_mean"],
        "initial_bce": None if attack_vs_init is None else attack_vs_init["clean_loss_mean"],
        "adv_bce": attack_vs_anchor["adv_loss_mean"],
        "loss_gain": attack_vs_anchor["loss_gain_mean"],
    }


# ────────────────────────────────────────────────────────────────────────────
# Latent-hull epoch step
# ────────────────────────────────────────────────────────────────────────────


_LATENT_HULL_GEOMETRY_FIELDS = (
    "pre_projection_norm", "post_projection_norm", "projection_scale",
    "effective_lambda", "candidate_anchor_weight", "effective_original_share",
)
_LATENT_HULL_ARRAY_FIELDS = (
    "anchor_pool_indices",
    "candidate_pool_indices",
    "candidate_is_anchor",
    "initial_weights",
    "final_weights",
    "initial_top1_candidate_pool_indices",
    "final_top1_candidate_pool_indices",
    "initial_top1_weights",
    "final_top1_weights",
) + tuple(
    f"{stage}_{field}"
    for stage in ("initial", "final")
    for field in _LATENT_HULL_GEOMETRY_FIELDS
)


def merge_latent_hull_batch_diagnostics(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"schema_version": 1, "n": 0}
    merged.update({field: [] for field in _LATENT_HULL_ARRAY_FIELDS})
    for batch_index, batch in enumerate(batches):
        anchors = batch.get("anchor_pool_indices")
        if anchors is None:
            raise RuntimeError("latent-hull generator omitted diagnostic field 'anchor_pool_indices'")
        try:
            anchor_rows = len(anchors)
        except TypeError:
            raise RuntimeError(
                f"latent-hull diagnostic batch {batch_index} anchor_pool_indices has no row axis"
            ) from None
        for field in _LATENT_HULL_ARRAY_FIELDS:
            values = batch.get(field)
            if values is None:
                raise RuntimeError(f"latent-hull generator omitted diagnostic field {field!r}")
            try:
                rows = len(values)
            except TypeError:
                raise RuntimeError(
                    f"latent-hull diagnostic batch {batch_index} field {field!r} has no row axis; "
                    f"expected {anchor_rows} anchor rows"
                ) from None
            if rows != anchor_rows:
                raise RuntimeError(
                    f"latent-hull diagnostic batch {batch_index} field {field!r} has {rows} rows; "
                    f"expected {anchor_rows} anchor rows"
                )
            merged[field].extend(values)
    merged["n"] = len(merged["anchor_pool_indices"])
    for field in _LATENT_HULL_ARRAY_FIELDS:
        if len(merged[field]) != merged["n"]:
            raise RuntimeError(
                f"merged latent-hull diagnostic field {field!r} has {len(merged[field])} rows; "
                f"expected {merged['n']} anchor rows"
            )
    return merged


def summarize_latent_hull_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"schema_version": 1, "n": int(diagnostics.get("n", 0))}
    for field in (*_LATENT_HULL_GEOMETRY_FIELDS, "top1_weights"):
        for stage in ("initial", "final"):
            key = f"{stage}_{field}"
            values = np.asarray(diagnostics.get(key, []), dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                continue
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                raise RuntimeError(
                    f"latent-hull diagnostic field {key!r} is non-finite at sample {int(bad[0])}"
                )
            field_summary = {}
            with np.errstate(over="ignore", invalid="ignore"):
                for name, fn in (("mean", np.mean), ("p50", lambda x: np.percentile(x, 50)),
                                 ("p90", lambda x: np.percentile(x, 90))):
                    value = float(fn(values))
                    if not np.isfinite(value):
                        raise RuntimeError(
                            f"latent-hull diagnostic summary field {key!r} statistic {name!r} "
                            "is non-finite"
                        )
                    field_summary[name] = value
            summary[key] = field_summary
    return summary


def run_pgd_on_synth_pool(
    pgd_gen: LatentHullPGDGenerator,
    synth_latents: np.ndarray,    # (N, 4, 128)
    synth_labels: np.ndarray,     # (N, C) one-hot
    pgd_batch: int,
    device: str,
    latent_hull_index: SameLabelLatentIndex,
    picked_indices: Optional[np.ndarray] = None,
    hull_M: int = 10,
    hull_mix_label_mode: str = "anchor",
    hull_label_lambda_y: float = 0.5,
    hull_label_positive: float = 0.95,
    hull_label_negative_floor: float = 0.0,
    hull_label_new_class_cap: float = 0.5,
    store_raw_decoded: bool = False,
    pool_record_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sample K anchors stratified by class, run latent-hull PGD in batches.

    Returns:
      adv_signals_ct:   (K, 12, 1000)  float32
      anchor_signals_ct:(K, 12, 1000)  float32  (clean reference for sem-gate)
      labels_one_hot:   (K, C)         float32
      delta_stats:      mean / max L2 norm plus optional latent-hull weight stats
    """
    if picked_indices is None:
        raise ValueError("picked_indices is required; latest mainline uses StratifiedPoolWalker")
    pick = picked_indices
    if pool_record_ids is not None and len(pool_record_ids) != len(synth_latents):
        raise ValueError("pool_record_ids must align with synth_latents")

    if len(pick) == 0:
        return (np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, synth_labels.shape[1]), dtype=np.float32),
                {"mean_delta_norm": float('nan'), "max_delta_norm": float('nan')})

    y_anchor_np = synth_labels[pick].astype(np.float32, copy=False)
    z_anchors = torch.from_numpy(synth_latents[pick]).float()        # (K, 4, 128)
    y_anchors = torch.from_numpy(y_anchor_np).float()                # (K, C)

    adv_chunks, anc_chunks, init_chunks, label_chunks, delta_norms = [], [], [], [], []
    raw_adv_chunks, raw_anc_chunks = [], []
    diagnostic_batches: List[Dict[str, Any]] = []
    for i in range(0, z_anchors.shape[0], pgd_batch):
        z_b = z_anchors[i:i + pgd_batch].to(device)
        y_b = y_anchors[i:i + pgd_batch].to(device)
        batch_pick = pick[i:i + pgd_batch]
        cand_np = latent_hull_index.candidates_for(batch_pick, hull_M)
        candidate_indices = np.asarray(
            getattr(latent_hull_index, "last_candidate_indices", None), dtype=np.int64
        ).copy()
        if candidate_indices.shape != (len(batch_pick), hull_M):
            raise RuntimeError("latent-hull index did not expose aligned candidate identities")
        candidate_is_anchor = candidate_indices == batch_pick[:, None]
        cand_b = torch.from_numpy(cand_np).float().to(device)
        x_adv, delta = pgd_gen.attack_from_latent(
            z_b,
            y_b,
            candidate_latents=cand_b,
            anchor_pool_indices=torch.from_numpy(batch_pick),
            candidate_pool_indices=torch.from_numpy(candidate_indices),
            candidate_is_anchor=torch.from_numpy(candidate_is_anchor),
        )
        if hull_mix_label_mode == "anchor":
            label_chunks.append(y_anchor_np[i:i + pgd_batch])
        elif hull_mix_label_mode == "anchor_soft":
            weights_t = getattr(pgd_gen, "last_weights", None)
            if weights_t is None:
                raise RuntimeError(
                    "latent-hull soft labels require candidate indices and weights"
                )
            weights_np = weights_t.numpy().astype(np.float32, copy=False)
            cand_labels = synth_labels[candidate_indices]
            label_chunks.append(build_anchor_preserving_soft_labels(
                y_anchor_np[i:i + pgd_batch],
                cand_labels,
                weights_np,
                lambda_y=hull_label_lambda_y,
                positive_value=hull_label_positive,
                negative_floor=hull_label_negative_floor,
                new_class_cap=hull_label_new_class_cap,
            ))
        else:
            raise ValueError(
                "hull_mix_label_mode must be anchor|anchor_soft, "
                f"got {hull_mix_label_mode!r}"
            )
        hull_info = getattr(pgd_gen, "last_info", {})
        diagnostic_batches.append(hull_info)
        initial_signals = getattr(pgd_gen, "last_initial_signals", None)
        if initial_signals is None:
            raise RuntimeError("latent-hull generator did not retain the true initial decoded signals")
        init_chunks.append(initial_signals.numpy().astype(np.float32, copy=False))
        adv_chunks.append(x_adv.detach().cpu().numpy().astype(np.float32))
        # Clean anchor reference (z_b alone, no delta)
        with torch.no_grad():
            anc_x = pgd_gen._decode_to_ptbxl_1000(z_b)
            if store_raw_decoded:
                raw_anc_x = pgd_gen._decode_to_ptbxl_1000_raw(z_b)
                raw_adv_x = pgd_gen._decode_to_ptbxl_1000_raw(z_b + delta.detach())
        anc_chunks.append(anc_x.detach().cpu().numpy().astype(np.float32))
        if store_raw_decoded:
            raw_anc_chunks.append(raw_anc_x.detach().cpu().numpy().astype(np.float32))
            raw_adv_chunks.append(raw_adv_x.detach().cpu().numpy().astype(np.float32))
        delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())

    adv_signals = np.concatenate(adv_chunks, axis=0)                  # (K, 12, 1000)
    anc_signals = np.concatenate(anc_chunks, axis=0)
    pgd_gen.last_initial_signals_ptbxl_1000 = np.concatenate(init_chunks, axis=0)
    if store_raw_decoded:
        pgd_gen.last_anchor_raw_ptbxl_1000 = np.concatenate(raw_anc_chunks, axis=0)
        pgd_gen.last_adv_raw_ptbxl_1000 = np.concatenate(raw_adv_chunks, axis=0)
    else:
        pgd_gen.last_anchor_raw_ptbxl_1000 = None
        pgd_gen.last_adv_raw_ptbxl_1000 = None
    stats = {
        "mean_delta_norm": float(np.mean(delta_norms)),
        "max_delta_norm":  float(np.max(delta_norms)),
    }
    diagnostics = merge_latent_hull_batch_diagnostics(diagnostic_batches)
    if pool_record_ids is not None:
        record_ids = np.asarray(pool_record_ids).astype(str)
        candidate_ids = np.asarray(diagnostics["candidate_pool_indices"], dtype=np.int64)
        diagnostics.update({
            "anchor_record_ids": record_ids[np.asarray(diagnostics["anchor_pool_indices"], dtype=np.int64)].tolist(),
            "candidate_record_ids": record_ids[candidate_ids].tolist(),
            "initial_top1_candidate_record_ids": record_ids[
                np.asarray(diagnostics["initial_top1_candidate_pool_indices"], dtype=np.int64)
            ].tolist(),
            "final_top1_candidate_record_ids": record_ids[
                np.asarray(diagnostics["final_top1_candidate_pool_indices"], dtype=np.int64)
            ].tolist(),
        })
    final_weights = np.asarray(diagnostics["final_weights"], dtype=np.float64)
    entropy = -(final_weights * np.log(np.clip(final_weights, 1e-12, None))).sum(axis=1)
    stats.update({
        "hull_weight_entropy_mean": float(np.mean(entropy)),
        "hull_weight_top1_mean": float(np.mean(final_weights.max(axis=1))),
        "latent_hull_diagnostics": diagnostics,
        "latent_hull_diagnostics_summary": summarize_latent_hull_diagnostics(diagnostics),
    })
    out_labels = np.concatenate(label_chunks, axis=0).astype(np.float32)
    return adv_signals, anc_signals, out_labels, stats


def build_three_chain_vae_lhat_augmix_views(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    severity_profile: str = "standard",
    width: int,
    depth: int,
    alpha: float,
    ops: List[str],
    rng: np.random.Generator,
    third_chain_role: str = "vae_lhat_adversarial_waveform",
    chain_base_mode: str = "clean_clean_third",
    chain_weights: List[float] | None = None,
    adv_base_mix: float = 1.0,
    locked_raw_chain: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Locked wrapper: two ECG corruption chains plus one uncorrupted VAE-LHAT chain."""

    def _apply_augmix_op_np(
        sig_ct: np.ndarray,
        op_name: str,
        op_severity: int,
        op_severity_profile: str,
    ) -> np.ndarray:
        if locked_raw_chain:
            sig_t = torch.from_numpy(sig_ct.copy()).float()
            return apply_corruption_sequence(
                sig_t,
                op_name,
                int(op_severity),
                op_severity_profile,
                base_seed=int(rng.integers(0, 2**31)),
                seed_parts=("effnet_locked_raw_augmix", op_name),
                sample_rate_hz=100.0,
            ).cpu().numpy().astype(np.float32, copy=False)
        if op_severity_profile == "standard":
            return _standard_augmix_op_np(sig_ct, op_name, int(op_severity))
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        op = _build_pn2021c_corruption_op(
            op_name,
            int(op_severity),
            op_severity_profile,
            severity_profile_params=None,
        )
        return op(sig_t).cpu().numpy().astype(np.float32, copy=False)

    return _build_three_chain_vae_lhat_augmix_views_core(
        anchor_signals_ct,
        adv_signals_ct,
        copies=copies,
        severity=severity,
        severity_profile=severity_profile,
        width=width,
        depth=depth,
        alpha=alpha,
        mixture_mode="beta",
        mixture_prob=0.5,
        mixture_beta_a=None,
        mixture_beta_b=None,
        op_schedule="random",
        chain_weights=chain_weights,
        ops=ops,
        rng=rng,
        op_apply_fn=_apply_augmix_op_np,
        available_ops=AVAILABLE_OPS,
        renorm=True,
        clip_abs=6.0,
        third_chain_role=third_chain_role,
        chain_base_mode=chain_base_mode,
        adv_base_mix=adv_base_mix,
    )


def _resolve_latent_augmix_epoch_route(
    pgd_gen: LatentHullPGDGenerator,
    normalized_anchor_signals: np.ndarray,
    normalized_adv_signals: np.ndarray,
    signal_space: str,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return AugMix inputs and whether the independent VAE stream is enabled."""
    if signal_space != LOCKED_LATENT_AUGMIX_SIGNAL_SPACE:
        return normalized_anchor_signals, normalized_adv_signals, True
    raw_anchor = getattr(pgd_gen, "last_anchor_raw_ptbxl_1000", None)
    raw_adv = getattr(pgd_gen, "last_adv_raw_ptbxl_1000", None)
    if raw_anchor is None or raw_adv is None:
        raise RuntimeError("locked raw-first AugMix requires stored raw VAE decodes")
    return raw_anchor, raw_adv, False


def build_clean_anchor_augmix_epoch(
    target_real_ds: TargetRealWaveformDataset | TargetRealRawFirstCorruptionDataset,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, Any]:
    """Build no-VAE three-chain AugMix views from K500 clean anchors."""

    n = len(target_real_ds)
    if n <= 0:
        raise RuntimeError("clean-anchor AugMix requires a non-empty target_real_npz")
    k_anchor = max(1, int(args.K_anchor))
    rng = np.random.default_rng(int(args.seed) + int(epoch) * 100003)
    picks = rng.choice(n, size=k_anchor, replace=k_anchor > n)
    clean: List[np.ndarray] = []
    raw: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for idx in picks:
        item = target_real_ds[int(idx)]
        if len(item) == 3:
            sig_t, label_t, raw_t = item
            raw.append(raw_t.detach().cpu().numpy().astype(np.float32, copy=False))
        else:
            sig_t, label_t = item
        clean.append(sig_t.detach().cpu().numpy().astype(np.float32, copy=False))
        labels.append(label_t.detach().cpu().numpy().astype(np.float32, copy=False))
    clean_np = np.stack(clean, axis=0).astype(np.float32, copy=False)
    raw_np = np.stack(raw, axis=0).astype(np.float32, copy=False) if raw else None
    labels_np = np.stack(labels, axis=0).astype(np.float32, copy=False)
    if bool(args.enable_raw_augmix):
        locked_raw_chain = (
            args.latent_augmix_signal_space == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
        )
        if locked_raw_chain and raw_np is None:
            raise RuntimeError(
                "locked target-real AugMix requires paired raw-pre-zscore crops"
            )
        corruption_base_np = raw_np if locked_raw_chain else clean_np
        views_np, stats = build_three_chain_vae_lhat_augmix_views(
            anchor_signals_ct=corruption_base_np,
            adv_signals_ct=corruption_base_np,
            copies=args.latent_augmix_copies,
            severity=args.latent_augmix_severity,
            severity_profile=args.latent_augmix_severity_profile,
            width=args.latent_augmix_width,
            depth=args.latent_augmix_depth,
            alpha=args.latent_augmix_alpha,
            ops=list(args.latent_augmix_ops),
            rng=rng,
            third_chain_role="clean_anchor_control",
            chain_base_mode=args.latent_augmix_chain_base_mode,
            chain_weights=parse_float_sequence(args.latent_augmix_chain_weights),
            locked_raw_chain=locked_raw_chain,
        )
    else:
        views_np = np.tile(clean_np, (max(1, int(args.latent_augmix_copies)), 1, 1))
        stats = {
            "enabled": True,
            "control": "clean_budget_control",
            "raw_augmix": False,
            "n_generated": int(views_np.shape[0]),
        }
    labels_rep = np.tile(
        labels_np,
        (max(1, int(args.latent_augmix_copies)), 1),
    )[: views_np.shape[0]]
    clean_rep = np.tile(
        clean_np,
        (max(1, int(args.latent_augmix_copies)), 1, 1),
    )[: views_np.shape[0]]
    stats["corruption_source"] = (
        "target_real_npz.signals_raw1000"
        if raw_np is not None and bool(args.enable_raw_augmix)
        else "target_real_clean_anchor"
    )
    stats["signal_space"] = str(
        getattr(args, "latent_augmix_signal_space", "model_zscore")
    )
    stats["anchor_sample_mode"] = "clean_anchor_control"
    stats["anchor_count"] = int(clean_np.shape[0])
    return {
        "clean": clean_np,
        "raw": raw_np,
        "clean_rep": clean_rep.astype(np.float32, copy=False),
        "views": views_np.astype(np.float32, copy=False),
        "labels": labels_np.astype(np.float32, copy=False),
        "labels_rep": labels_rep.astype(np.float32, copy=False),
        "stats": stats,
    }


def build_no_raw_augmix_epoch(
    target_real_ds: TargetRealWaveformDataset,
    args: argparse.Namespace,
    epoch: int,
    *,
    matched_comparison: bool,
) -> Dict[str, Any] | None:
    return None if matched_comparison else build_clean_anchor_augmix_epoch(target_real_ds, args, epoch)


def raw_augmix_disabled_stats(signal_space: str) -> Dict[str, Any]:
    return {"enabled": False, "reason": "raw_augmix_disabled", "raw_augmix": False,
            "n_generated": 0, "signal_space": str(signal_space)}


# ────────────────────────────────────────────────────────────────────────────
# Adv buffer push with masked-BCE label semantics
# ────────────────────────────────────────────────────────────────────────────

def push_adv_to_buffer(
    buffer: QualityAwareBuffer,
    adv_signals_ct: np.ndarray,    # (K, 12, 1000)
    target_one_hot: np.ndarray,    # (K, C)
    victim_logits: np.ndarray,     # (K, C)
    crop_len: int,
    class_trust: Optional[Dict[str, float]] = None,
    boundary_prob_min: float = 0.0,
    boundary_prob_max: float = 1.0,
    teacher_probs: Optional[np.ndarray] = None,
    label_mode: str = "hard",
    teacher_mix: float = 0.7,
    soft_target_floor: float = 0.0,
    sample_weight_scale: float = 1.0,
    enabled: bool = True,
) -> Dict[str, int]:
    """Push gates-passed adv signals into the buffer with -1 sentinel labels.

    Label scheme: target_only (Plan Issue #28 — CheXpert U-Ignore + SPML).
    Non-target dims set to -1.0 → masked out by masked_bce_with_logits.

    Plan Rev 13.2: per-sample buffer push score is multiplied by class_trust
    (HYP/CD trust=0 → effectively dropped even if Stage 1 produced any). With
    the 3-class generation scope (NORM/MI/STTC) HYP/CD synth is already absent
    from the synth pool; this is a defense-in-depth check.
    """
    if not enabled:
        return {
            "n_pushed": 0,
            "n_dropped_by_trust": 0,
            "n_dropped_by_boundary": 0,
            "label_mode": label_mode,
            "sample_weight_scale": float(sample_weight_scale),
            "reason": "disabled",
        }
    n_pushed = 0
    n_dropped_by_trust = 0
    n_dropped_by_boundary = 0
    for i in range(adv_signals_ct.shape[0]):
        sig_250 = center_crop_ct(adv_signals_ct[i], crop_len)          # (12, 250)
        target_idx = int(target_one_hot[i].argmax())
        target_class = CLASS_NAMES_SUPER5[target_idx] if target_idx < len(CLASS_NAMES_SUPER5) else None
        trust = float(class_trust.get(target_class, 1.0)) if class_trust else 1.0
        if trust <= 0.0:
            n_dropped_by_trust += 1
            continue
        prob_t = float(1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, victim_logits[i, target_idx])))))
        if prob_t < boundary_prob_min or prob_t > boundary_prob_max:
            n_dropped_by_boundary += 1
            continue
        teacher_i = teacher_probs[i] if teacher_probs is not None else None
        lbl = torch.from_numpy(
            build_adv_buffer_label(
                target_one_hot[i],
                label_mode=label_mode,
                teacher_probs=teacher_i,
                teacher_mix=teacher_mix,
                soft_target_floor=soft_target_floor,
            )
        )
        score = (1.0 - 2.0 * abs(prob_t - 0.5)) * trust  # ∈ [0, trust]
        buffer.add_one(
            torch.from_numpy(np.ascontiguousarray(sig_250)).float(),
            lbl,
            score,
            sample_weight=float(sample_weight_scale),
        )
        n_pushed += 1
    return {
        "n_pushed": n_pushed,
        "n_dropped_by_trust": n_dropped_by_trust,
        "n_dropped_by_boundary": n_dropped_by_boundary,
        "label_mode": label_mode,
        "sample_weight_scale": float(sample_weight_scale),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--center_name", required=True, help="cpsc_2018_extra | ningbo")
    p.add_argument(
        "--comparison_arm",
        choices=["historical_unmatched", *MATCHED_EFFNET_ARMS],
        default="historical_unmatched",
    )
    p.add_argument("--comparison_protocol", default="")
    p.add_argument("--comparison_variant", default="")
    p.add_argument("--comparison_topology_version", default="")
    p.add_argument("--ref_meta_json",
                   help="path to {tag}_k200.meta.json (for record_id exclusion in eval)")
    p.add_argument("--synth_npz", required=True,
                   help="Stage 1 latent npz: {latents (N,4,128), labels (N,5)}")
    p.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
    p.add_argument("--init_checkpoint_sha256", default="")
    p.add_argument("--init_lineage_stage", default="")
    p.add_argument("--model_name", default="efficientnet1dv2",
                   choices=available_model_names())
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_real_npz", default="",
                   help="Optional selected target-center real ECG npz with signals (N,1000,12) and labels.")
    p.add_argument(
        "--target_real_norm_mode",
        choices=["pre_zscored", "per_sample_global"],
        default="pre_zscored",
        help=(
            "Normalization contract for --target_real_npz. Use pre_zscored "
            "for legacy selected .signals.npz and per_sample_global for "
            "raw1000 K-shot artifacts."
        ),
    )

    # PTBXL paths
    p.add_argument("--ptbxl_raw", default=DEFAULT_PTBXL_RAW)
    p.add_argument("--ptbxl_csv", default=DEFAULT_PTBXL_CSV)
    p.add_argument("--ptbxl_prep", default=DEFAULT_PTBXL_PREP)

    # Latent-hull hard-sample search.
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_batch", type=int, default=32)        # Issue #45
    p.add_argument("--K_anchor", type=int, default=300)
    p.add_argument("--hull_M", type=int, default=10,
                   help="Number of same-label candidate latents for latent_hull")
    p.add_argument("--hull_lambda", type=float, default=0.25)
    p.add_argument("--hull_steps", type=int, default=5)
    p.add_argument("--hull_lr", type=float, default=0.3)
    p.add_argument("--hull_init_logit_gap", type=float, default=4.0)
    p.add_argument(
        "--hull_weight_mode",
        choices=["optimized", "one_hot", "uniform", "dirichlet"],
        default="optimized",
        help="Latent-Hull coefficient policy: optimized=C3 main, or fixed C0/C1/C2 ablations.",
    )
    p.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    p.add_argument(
        "--hull_label_mode",
        choices=["primary", "exact", "compatible"],
        default="primary",
    )
    p.add_argument(
        "--hull_mix_label_mode",
        choices=["anchor", "anchor_soft"],
        default="anchor",
        help="anchor keeps the anchor multi-hot label; anchor_soft builds "
             "anchor-preserving fractional labels from latent-hull weights.",
    )
    p.add_argument("--hull_label_lambda_y", type=float, default=0.5)
    p.add_argument("--hull_label_positive", type=float, default=0.95)
    p.add_argument("--hull_label_negative_floor", type=float, default=0.0)
    p.add_argument("--hull_label_new_class_cap", type=float, default=0.5)
    p.add_argument(
        "--hull_include_anchor",
        action="store_true",
        help="Include the anchor latent itself as candidate 0 in same-label hull. "
             "Useful for no-lambda convex-hull ablations with --hull_lambda 1.0.",
    )
    p.add_argument(
        "--hull_neighbor_distance_space",
        choices=["raw", "standardized"],
        default="raw",
        help=(
            "Latent distance space used when selecting same-label hull partners. "
            "standardized uses per-dimension z-score distances over the latent pool."
        ),
    )
    p.add_argument(
        "--hull_neighbor_mode",
        choices=["nearest", "local_random", "random"],
        default="nearest",
        help=(
            "Partner selection inside the same-label pool. local_random samples "
            "from a nearby kNN pool, which is useful for locality-aware mixup."
        ),
    )
    p.add_argument("--hull_neighbor_pool_size", type=int, default=0)
    p.add_argument("--hull_neighbor_pool_multiplier", type=int, default=4)
    p.add_argument(
        "--anchor_class_weights",
        default=None,
        help=(
            "Comma map CLASS=weight for per-epoch latent anchor quotas, e.g. "
            "HYP=3,MI=3,CD=2,NORM=1,STTC=1. Uses only classes present in "
            "the K-shot latent pool."
        ),
    )
    p.add_argument(
        "--anchor_class_weight_mode",
        choices=["manual", "inv_freq_kshot"],
        default="manual",
        help=(
            "manual uses --anchor_class_weights. inv_freq_kshot derives a "
            "single global formula from the K500-train latent labels after the "
            "internal validation split is removed; no held-out target labels "
            "or target-center distribution are used."
        ),
    )
    p.add_argument(
        "--anchor_class_weight_reference_source",
        default="real_anchor",
        help=(
            "Source label used for inv_freq_kshot counts when source metadata "
            "exists; use 'all' to count the full latent pool."
        ),
    )
    p.add_argument("--anchor_class_weight_gamma", type=float, default=0.5)
    p.add_argument("--anchor_class_weight_min", type=float, default=0.35)
    p.add_argument("--anchor_class_weight_cap", type=float, default=4.0)
    p.add_argument(
        "--anchor_class_missing_weight",
        type=float,
        default=0.35,
        help="Weight assigned to classes absent from the K500-train reference subset.",
    )
    p.add_argument("--classes_in_scope", nargs="+", default=sorted(SUPER5_GEN_SUBSET),
                   help="Super5 classes sampled as adversarial anchors. Default keeps historical NORM/MI/STTC.")
    p.add_argument("--boundary_prob_min", type=float, default=0.0,
                   help="Only push adv samples whose target sigmoid probability is >= this value.")
    p.add_argument("--boundary_prob_max", type=float, default=1.0,
                   help="Only push adv samples whose target sigmoid probability is <= this value.")
    p.add_argument("--adv_label_mode",
                   choices=[
                       "hard", "multi_hot_hard", "latent_soft",
                       "mixed_soft", "teacher_soft", "latent_mixed_teacher",
                   ],
                   default="hard",
                   help="Label policy for generated adversarial ECGs: hard keeps historical target-only labels; "
                        "multi_hot_hard keeps the full anchor multi-hot label; "
                        "latent_soft uses latent-hull fractional labels; "
                        "mixed_soft blends frozen-teacher probabilities with a hard target; "
                        "latent_mixed_teacher blends frozen-teacher probabilities with latent_soft labels; "
                        "teacher_soft uses the frozen initial teacher probabilities directly.")
    p.add_argument("--adv_teacher_mix", type=float, default=0.7,
                   help="For --adv_label_mode mixed_soft, weight on frozen-teacher probabilities.")
    p.add_argument("--adv_soft_target_floor", type=float, default=0.0,
                   help="For soft adv labels, clamp the intended target class label to at least this value.")

    # Mix loader (Plan Rev 13.2: real-dominated mix, adv_w=0.5 vs Wang 2023 0.7 reverse)
    p.add_argument("--ptbxl_weight", type=float, default=1.0)
    p.add_argument("--target_real_weight", type=float, default=0.0,
                   help="Sampling weight for --target_real_npz supervised stream.")
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument("--selection_metric", choices=["macro_auroc", "macro_auprc"], default="macro_auprc")
    p.add_argument("--source_floor_max_drop", type=float, default=0.02)
    p.add_argument(
        "--target_adv_fraction",
        type=float,
        default=None,
        help="Target-objective coefficient for paired clean and pure VAE-LHAT losses.",
    )
    p.add_argument("--adv_weight", type=float, default=0.5)
    p.add_argument(
        "--vae_adv_stream_sample_scale",
        type=float,
        default=1.0,
        help="Quality-buffer score multiplier for VAE hard samples when clean AugMix views share the augment stream.",
    )
    p.add_argument(
        "--adv_weight_warmup_epochs",
        type=int,
        default=0,
        help=(
            "If >0, linearly ramp the adversarial buffer sampling weight from "
            "a small value to --adv_weight over this many epochs."
        ),
    )
    p.add_argument("--latent_augmix_copies", type=int, default=1,
                   help="Number of latent-branch AugMix samples to create per x_adv.")
    p.add_argument("--latent_augmix_width", type=int, default=3,
                   help="Total AugMix branches; branch 0 is x_adv, remaining branches are ECG op chains.")
    p.add_argument("--latent_augmix_depth", type=int, default=-1,
                   help="Depth per ECG op chain; -1 samples uniformly from {1,2,3}.")
    p.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    p.add_argument("--latent_augmix_severity", type=int, default=2)
    p.add_argument(
        "--latent_augmix_severity_profile",
        choices=[name for name in PN2021C_STRESS_PROFILE_CHOICES if name != "custom"],
        default="standard",
        help=(
            "Parameter profile used by non-latent latent-AugMix ECG op chains. "
            "'standard' preserves the original mild AugMix table; "
            "'calibrated_10to20pp' matches the strong PN2021-C evaluation profile."
        ),
    )
    p.add_argument(
        "--latent_augmix_third_chain_role",
        choices=list(MATCHED_EFFNET_THIRD_CHAIN_ROUTES),
        default="vae_lhat_adversarial_waveform",
        help=(
            "Role of the third locked AugMix chain. clean_anchor_control "
            "skips VAE-LHAT adversarial generation and uses clean K500 anchors."
        ),
    )
    p.add_argument(
        "--latent_augmix_chain_base_mode",
        choices=["clean_clean_third", "all_clean", "one_adv", "all_adv", "all_clean_plus_vae_adv"],
        default="clean_clean_third",
        help=(
            "Base waveform for locked AugMix chains. Default keeps c117: two "
            "clean-anchor corruption chains plus the third chain role. "
            "all_clean/all_adv corrupt all three chains from clean anchors or "
            "VAE-LHAT adversarial waveforms; one_adv corrupts one chain from "
            "the VAE-LHAT adversarial waveform. all_clean_plus_vae_adv keeps "
            "clean-anchor AugMix chains while adding a separate VAE-LHAT "
            "supervised hard-sample stream."
        ),
    )
    p.add_argument(
        "--latent_augmix_adv_base_mix",
        type=float,
        default=1.0,
        help="For one_adv/all_adv, use (1-rho)*clean + rho*VAE-adv as the corruption base.",
    )
    p.add_argument("--latent_augmix_chain_weights", default="")
    p.add_argument(
        "--latent_augmix_ops",
        nargs="+",
        default=["powerline_noise", "emg_noise", "baseline_wander", "baseline_shift"],
        choices=AVAILABLE_OPS,
        help="ECG corruption ops for non-latent AugMix branches. Random lead masking is excluded by default.",
    )
    p.add_argument("--latent_augmix_consistency_weight", type=float, default=2.0)
    p.add_argument(
        "--latent_augmix_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    p.add_argument("--latent_augmix_bce_weight", type=float, default=1.0)
    p.add_argument("--latent_augmix_consistency_max_batches", type=int, default=0)
    consistency = p.add_mutually_exclusive_group()
    consistency.add_argument(
        "--enable_latent_augmix_consistency",
        dest="enable_latent_augmix_consistency",
        action="store_true",
    )
    consistency.add_argument(
        "--disable_latent_augmix_consistency",
        dest="enable_latent_augmix_consistency",
        action="store_false",
    )
    p.set_defaults(enable_latent_augmix_consistency=True)
    vae_lhat = p.add_mutually_exclusive_group()
    vae_lhat.add_argument("--enable_vae_lhat", dest="enable_vae_lhat", action="store_true")
    vae_lhat.add_argument("--disable_vae_lhat", dest="enable_vae_lhat", action="store_false")
    raw_augmix = p.add_mutually_exclusive_group()
    raw_augmix.add_argument("--enable_raw_augmix", dest="enable_raw_augmix", action="store_true")
    raw_augmix.add_argument("--disable_raw_augmix", dest="enable_raw_augmix", action="store_false")
    p.set_defaults(enable_vae_lhat=True, enable_raw_augmix=True)
    p.add_argument(
        "--vae_adv_consistency_weight",
        type=float,
        default=0.0,
        help="Extra soft-BCE consistency from clean VAE anchor to VAE-LHAT adversarial waveform.",
    )
    p.add_argument("--qab_size", type=int, default=2048)
    p.add_argument("--class_trust", default=None,
                   help="Path to the real-all-present class_trust.json written by the managed wrapper.")

    # Optim
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=12)      # Issue #46
    p.add_argument("--anchor_lambda", type=float, default=0.05)
    p.add_argument("--ewa_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--rescore_interval", type=int, default=3)

    # Gates (Plan Issue #38)
    p.add_argument("--asr_consec_low_max", type=int, default=3,
                   help="Halt with RuntimeError after this many consecutive low-ASR epochs")
    p.add_argument("--asr_low_threshold", type=float, default=0.30)
    p.add_argument("--asr_high_threshold", type=float, default=0.70)
    p.add_argument("--einthoven_p95_max", type=float, default=0.5)
    p.add_argument(
        "--disable_quality_gate",
        action="store_true",
        help="Do not skip buffer push when semantic/quality gate fails. "
             "Still compute and log gate metrics for ablation.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop_len", type=int, default=TIERM_INPUT_LENGTH)
    p.add_argument(
        "--resume",
        default="",
        help=(
            "Resume from an epoch-boundary training checkpoint. Use 'latest' "
            "for output_dir/checkpoints/checkpoint_latest.pt, or pass a path."
        ),
    )
    p.add_argument(
        "--allow_resume_config_drift",
        action="store_true",
        help="Allow critical args in a resume checkpoint to differ from the current command.",
    )
    p.add_argument(
        "--final_checkpoint_only",
        action="store_true",
        help="Skip epoch-boundary resume checkpoints and write last_model.pt once at the end.",
    )
    args = p.parse_args(argv)
    paper_matched = is_paper_matched_effnet_run(
        args.comparison_arm, args.comparison_protocol
    )
    if paper_matched:
        if args.init_lineage_stage != "ptbxl_source" or len(args.init_checkpoint_sha256) != 64:
            p.error("matched comparison requires verified ptbxl_source initialization")
        if not 0.0 < args.target_real_val_fraction < 0.5:
            p.error("matched comparison requires --target_real_val_fraction in (0, 0.5)")
    if args.target_adv_fraction is not None and not 0.0 <= args.target_adv_fraction <= 1.0:
        p.error("--target_adv_fraction must be in [0, 1]")
    if is_matched_effnet_arm(args.comparison_arm):
        try:
            validate_matched_effnet_runtime(
                args.comparison_arm,
                enable_vae_lhat=args.enable_vae_lhat,
                enable_raw_augmix=args.enable_raw_augmix,
                enable_auxiliary_steps=args.enable_latent_augmix_consistency,
                bce_weight=args.latent_augmix_bce_weight,
                jsd_weight=args.latent_augmix_consistency_weight,
                third_chain_route=args.latent_augmix_third_chain_role,
                target_adv_fraction=args.target_adv_fraction,
            )
        except ValueError as exc:
            p.error(str(exc))
    if float(args.vae_adv_stream_sample_scale) < 0.0:
        p.error("--vae_adv_stream_sample_scale must be non-negative")
    if not (0.0 <= float(args.latent_augmix_adv_base_mix) <= 1.0):
        p.error("--latent_augmix_adv_base_mix must be in [0, 1]")
    if float(args.vae_adv_consistency_weight) < 0.0:
        p.error("--vae_adv_consistency_weight must be non-negative")
    if int(args.latent_augmix_copies) <= 0:
        p.error("--latent_augmix_copies must be > 0 for locked latent AugMix")
    if (
        args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
        and args.latent_augmix_third_chain_role == "clean_anchor_control"
    ):
        p.error(
            f"--latent_augmix_chain_base_mode {args.latent_augmix_chain_base_mode} "
            "cannot use clean_anchor_control"
        )
    if (
        (
            args.latent_augmix_third_chain_role == "clean_anchor_control"
            or args.latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
        )
        and not args.target_real_npz
    ):
        p.error("clean-anchor latent AugMix requires --target_real_npz")
    if (
        args.latent_augmix_third_chain_role == "clean_anchor_control"
        or args.latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
    ) and float(args.adv_weight) <= 0.0:
        p.error("clean-anchor latent AugMix requires --adv_weight > 0; it samples the augment stream")
    raw_signal_space = args.enable_raw_augmix if paper_matched else (
        args.latent_augmix_chain_base_mode == "clean_clean_third"
        and args.latent_augmix_third_chain_role == "vae_lhat_adversarial_waveform"
    )
    args.latent_augmix_signal_space = (
        LOCKED_LATENT_AUGMIX_SIGNAL_SPACE if raw_signal_space else "model_zscore"
    )
    if args.comparison_protocol or args.comparison_variant or args.comparison_topology_version:
        if not is_f004_rho_sweep(args.comparison_protocol):
            p.error(f"unsupported --comparison_protocol {args.comparison_protocol!r}")
        if args.comparison_topology_version != MATCHED_EFFNET_CONTRACT_VERSION:
            p.error("F-004 comparison topology version mismatch")
        try:
            validate_f004_runtime(
                comparison_protocol=args.comparison_protocol,
                comparison_arm=args.comparison_arm,
                comparison_variant=args.comparison_variant,
                target_adv_fraction=args.target_adv_fraction,
                enable_vae_lhat=args.enable_vae_lhat,
                enable_raw_augmix=args.enable_raw_augmix,
                enable_auxiliary_steps=args.enable_latent_augmix_consistency,
                bce_weight=args.latent_augmix_bce_weight,
                jsd_weight=args.latent_augmix_consistency_weight,
                third_chain_route=args.latent_augmix_third_chain_role,
                latent_augmix_signal_space=args.latent_augmix_signal_space,
            )
        except (TypeError, ValueError) as exc:
            p.error(str(exc))
    return args


def load_synth_pool(synth_npz_path: str) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Load Stage 1 frozen latent pool. Accepts either:

      - {basename}.npz       (signals + labels): auto-finds {basename}.latent.npz
      - {basename}.latent.npz (latents + labels): used directly

    Returns (latents (N,4,128), labels (N,C), center_name, source_meta).
    """
    try:
        return _load_synth_pool(synth_npz_path)
    except LatentPoolError as exc:
        raise SystemExit(str(exc)) from exc


def load_optional_vae_assets(
    args: argparse.Namespace,
    *,
    ecgtwin_factory=ECGTwinWrapper,
    latent_pool_loader=load_synth_pool,
):
    """Load VAE-only assets only for arms whose canonical contract enables them."""
    if not bool(args.enable_vae_lhat):
        return None, None
    return (
        ecgtwin_factory(device=args.device, load_encoder=True, load_text_model=False),
        latent_pool_loader(args.synth_npz),
    )


def load_optional_vae_component(enabled: bool, factory):
    return factory() if enabled else None


def main():
    args = parse_args()
    matched_comparison = is_paper_matched_effnet_run(
        args.comparison_arm, args.comparison_protocol
    )
    arm_components = (
        matched_effnet_arm("a5")
        if is_f004_rho_sweep(args.comparison_protocol)
        else matched_effnet_arm(args.comparison_arm) if matched_comparison else None
    )
    method_updates_enabled = bool(
        arm_components.vae_lhat or arm_components.raw_augmix
    ) if arm_components else True
    source_checkpoint_sha256 = file_sha256(args.init_ckpt)
    if matched_comparison and source_checkpoint_sha256 != args.init_checkpoint_sha256:
        raise RuntimeError("matched comparison source checkpoint sha256 mismatch")
    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Synth-anchored online AT (Super5) — Plan Rev 13.2")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 72)

    ecgtwin, latent_pool = load_optional_vae_assets(args)
    print(f"[setup] Loading Super5 victim from {args.init_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=ecgtwin,
        num_classes=NUM_SUPER5,
        crop_len=args.crop_len,
        model_name=args.model_name,
    )

    if not args.class_trust or not os.path.exists(args.class_trust):
        raise SystemExit(f"--class_trust required for training (got {args.class_trust!r}). "
                          f"Launch through effnet_vae_lhat_augmix.py so the managed wrapper writes it.")
    with open(args.class_trust) as f:
        trust_blob = json.load(f)
    class_trust: Dict[str, float] = dict(trust_blob["class_trust"])
    print(f"[setup] class_trust loaded: {class_trust}")
    classes_in_scope = []
    for cls in args.classes_in_scope:
        cls = cls.upper()
        if cls not in SUPER5_TO_IDX:
            raise SystemExit(f"unknown --classes_in_scope class {cls!r}; valid={CLASS_NAMES_SUPER5}")
        classes_in_scope.append(cls)
    if not classes_in_scope:
        raise SystemExit("--classes_in_scope must contain at least one class")
    print(f"[setup] classes_in_scope={classes_in_scope}")
    print(f"[setup] boundary target probability window=[{args.boundary_prob_min}, {args.boundary_prob_max}]")
    print(f"[setup] adv label mode={args.adv_label_mode} "
          f"teacher_mix={args.adv_teacher_mix} target_floor={args.adv_soft_target_floor}")
    print(f"[setup] hull mix label mode={args.hull_mix_label_mode} "
          f"lambda_y={args.hull_label_lambda_y} pos={args.hull_label_positive} "
          f"neg_floor={args.hull_label_negative_floor} new_cap={args.hull_label_new_class_cap}")
    print(
        "[setup] latent-branch AugMix enabled: "
        "topology=locked_three_chain "
        f"copies={args.latent_augmix_copies} width={args.latent_augmix_width} "
        f"depth={args.latent_augmix_depth} severity={args.latent_augmix_severity} "
        f"profile={args.latent_augmix_severity_profile} "
        "mixture=beta:0.5 op_schedule=random chain_weights=dirichlet "
        f"signal_space={args.latent_augmix_signal_space} corruption_source=vae_decode "
        f"third_chain_role={args.latent_augmix_third_chain_role} "
        f"chain_base_mode={args.latent_augmix_chain_base_mode} "
        f"adv_base_mix={args.latent_augmix_adv_base_mix} "
        f"chain_weights={args.latent_augmix_chain_weights or 'dirichlet'} "
        f"ops={args.latent_augmix_ops}",
        flush=True,
    )
    print(
        f"[setup] latent-branch AugMix direct consistency enabled={args.enable_latent_augmix_consistency}: "
        f"loss={args.latent_augmix_consistency_loss} "
        f"weights=(consistency={args.latent_augmix_consistency_weight}, "
        f"bce={args.latent_augmix_bce_weight}) "
        f"max_batches={args.latent_augmix_consistency_max_batches}",
        flush=True,
    )
    if int(args.latent_augmix_width) != 3:
        raise ValueError("locked three-chain latent AugMix requires --latent_augmix_width 3")
    chain_weights = parse_float_sequence(args.latent_augmix_chain_weights)
    if chain_weights is not None and len(chain_weights) != 3:
        raise ValueError("--latent_augmix_chain_weights must contain exactly three comma-separated values")
    if latent_pool is not None:
        synth_latents, synth_labels, synth_center, source_meta = latent_pool
    else:
        synth_latents = synth_labels = None
        synth_center, source_meta = args.center_name, {
            "source_labels": [], "source_names": [], "has_source_metadata": False,
        }
    print(f"[setup] synth pool: {synth_latents.shape} labels={synth_labels.shape} "
          f"center={synth_center}" if synth_latents is not None else "[setup] VAE pool bypassed")
    cls_dist = synth_labels.argmax(1) if synth_labels is not None else np.empty(0, dtype=np.int64)
    from collections import Counter
    pool_class_counts = Counter(int(c) for c in cls_dist)
    print(f"[setup] synth class counts (idx): {dict(pool_class_counts)}")
    source_counts = Counter(str(s) for s in source_meta["source_labels"])
    print(f"[setup] synth source counts: {dict(source_counts)} "
          f"has_metadata={source_meta['has_source_metadata']}")
    scheme = get_super5_scheme()

    label_cache = os.path.join(args.output_dir, "ptbxl_labels")
    train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=list(range(1, 9))
    )
    val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=[9]
    )

    cache_path = args.ptbxl_prep if os.path.exists(args.ptbxl_prep) else \
        os.path.join(args.output_dir, "ptbxl_preprocessed.npy")
    print(f"[setup] PTBXL preprocessed cache → {cache_path}")
    all_sig = preprocess_ptbxl_all(args.ptbxl_raw, cache_path)
    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])

    train_ds = PTBXLDatasetScheme(train_signals, train_labels,
                                  crop_len=args.crop_len, mode='train')
    val_ds = PTBXLDatasetScheme(val_signals, val_labels,
                                crop_len=args.crop_len, mode='eval')
    target_real_ds = None
    target_real_raw_augmix_ds = None
    target_val_ds = None
    matched_split: Dict[str, Any] | None = None
    if args.target_real_npz:
        with np.load(args.target_real_npz, allow_pickle=True) as real_data:
            real_signals = np.asarray(real_data["signals"], dtype=np.float32)
            real_labels = np.asarray(real_data["labels"], dtype=np.float32)
            real_record_ids = (
                np.asarray(real_data["record_ids"]).astype(str)
                if "record_ids" in real_data.files
                else None
            )
        if real_signals.ndim != 3:
            raise ValueError(f"target_real_npz signals must be 3D, got {real_signals.shape}")
        if real_signals.shape[1:] == (12, 1000):
            real_signals = real_signals.transpose(0, 2, 1)
        if real_signals.shape[1:] != (1000, 12):
            raise ValueError(f"target_real_npz signals must be (N,1000,12) or (N,12,1000), got {real_signals.shape}")
        if real_labels.shape[0] != real_signals.shape[0] or real_labels.shape[1] != NUM_SUPER5:
            raise ValueError(f"target_real_npz labels mismatch: signals={real_signals.shape} labels={real_labels.shape}")
        if matched_comparison:
            if real_record_ids is None:
                raise ValueError("matched comparison target_real_npz requires record_ids")
            matched_split = matched_k500_split(
                real_record_ids,
                real_labels,
                val_fraction=args.target_real_val_fraction,
                seed=args.target_real_val_seed,
            )
            train_indices = np.asarray(matched_split["train_indices"], dtype=np.int64)
            val_indices = np.asarray(matched_split["val_indices"], dtype=np.int64)
            target_val_ds = TargetRealWaveformDataset(
                real_signals[val_indices],
                real_labels[val_indices],
                crop_len=args.crop_len,
                mode="eval",
                norm_mode=args.target_real_norm_mode,
            )
            real_signals = real_signals[train_indices]
            real_labels = real_labels[train_indices]
            if args.enable_vae_lhat:
                synth_latents, synth_labels, source_meta = filter_latent_candidates(
                    synth_latents, synth_labels, source_meta,
                    train_record_ids=matched_split["train_record_ids"],
                    validation_record_ids=matched_split["val_record_ids"],
                )
                cls_dist = synth_labels.argmax(1)
                pool_class_counts = Counter(int(c) for c in cls_dist)
                source_counts = Counter(str(s) for s in source_meta["source_labels"])
            print(
                f"[setup] matched K500 split: train={len(train_indices)} val={len(val_indices)} "
                f"train_hash={matched_split['train_record_ids_sha256']} "
                f"val_hash={matched_split['val_record_ids_sha256']}",
                flush=True,
            )
        target_real_ds = TargetRealWaveformDataset(
            real_signals,
            real_labels,
            crop_len=args.crop_len,
            mode='train',
            norm_mode=args.target_real_norm_mode,
        )
        if (
            args.enable_raw_augmix
            and args.latent_augmix_signal_space == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
        ):
            target_real_raw_augmix_ds = TargetRealRawFirstCorruptionDataset(
                real_signals,
                real_labels,
                crop_len=args.crop_len,
                mode="train",
                clean_norm_mode=args.target_real_norm_mode,
            )
        print(
            f"[setup] target-real supervised stream: n={len(target_real_ds)} "
            f"weight={args.target_real_weight} norm_mode={args.target_real_norm_mode} "
            f"path={args.target_real_npz}",
            flush=True,
        )
    elif matched_comparison:
        raise ValueError("matched comparison requires --target_real_npz")

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5),
        dtype=torch.float32, device=args.device)
    print(f"[loss] pos_weight: {pos_weight.cpu().tolist()}")

    # reduction='none' for mask × bce (-1 sentinel handling)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    target_val_loader = (
        DataLoader(
            target_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if target_val_ds is not None
        else None
    )
    best_metric = -float("inf")
    best_epoch = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    source_baseline_metrics: Dict[str, float] = {}
    source_floor_result: Dict[str, Any] = {}
    best_source_floor_result: Dict[str, Any] = {}
    realized_optimizer_steps = 0
    scheduler_steps = 0
    optimizer_steps_per_epoch = 0
    if matched_comparison and target_real_ds is not None:
        auxiliary_steps = math.ceil(int(args.K_anchor) / int(args.batch_size))
        if args.latent_augmix_consistency_max_batches > 0:
            auxiliary_steps = min(auxiliary_steps, args.latent_augmix_consistency_max_batches)
        optimizer_steps_per_epoch = len(target_real_ds) // int(args.batch_size) + auxiliary_steps
    if matched_comparison:
        assert target_val_loader is not None and matched_split is not None
        source_baseline_metrics = evaluate_loader_macro(
            victim.model, val_loader, criterion, args.device
        )
        best_metric = float(evaluate_loader_macro(
            victim.model, target_val_loader, criterion, args.device
        )[args.selection_metric])
        source_floor_result = update_matched_checkpoint_selection(
            best_metric=-float("inf"),
            candidate_metric=best_metric,
            source_metric=float(source_baseline_metrics[args.selection_metric]),
            source_baseline_metric=float(source_baseline_metrics[args.selection_metric]),
            source_max_drop=args.source_floor_max_drop,
        )
        best_source_floor_result = dict(source_floor_result)
        if not args.resume:
            torch.save(victim.model.state_dict(), best_ckpt_path)

    contract_base = dict(
        source_checkpoint_path=args.init_ckpt,
        source_checkpoint_sha256=source_checkpoint_sha256, split=matched_split,
        selection_metric=args.selection_metric,
        source_floor_max_drop=args.source_floor_max_drop, epochs=args.n_epochs,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )

    def comparison_contract(realized_steps: int = 0, scheduler_count: int = 0) -> Dict[str, Any]:
        if not matched_comparison or matched_split is None:
            return {"contract": "historical_unmatched"}
        if is_f004_rho_sweep(args.comparison_protocol):
            return build_f004_training_record(
                target_adv_fraction=args.target_adv_fraction,
                **contract_base,
                realized_optimizer_steps=realized_steps,
                scheduler_steps=scheduler_count,
                source_floor_result=best_source_floor_result,
            )
        return build_matched_training_record(
            comparison_arm=args.comparison_arm,
            runtime_target_adv_fraction=args.target_adv_fraction,
            **contract_base, realized_optimizer_steps=realized_steps,
            scheduler_steps=scheduler_count, source_floor_result=best_source_floor_result,
        )

    pgd_gen = load_optional_vae_component(args.enable_vae_lhat, lambda: LatentHullPGDGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, epsilon=args.pgd_eps,
        hull_lambda=args.hull_lambda, hull_steps=args.hull_steps, hull_lr=args.hull_lr,
        init_logit_gap=args.hull_init_logit_gap, weight_mode=args.hull_weight_mode,
        dirichlet_alpha=args.hull_dirichlet_alpha, device=args.device,
    ))
    latent_hull_index = load_optional_vae_component(args.enable_vae_lhat, lambda: SameLabelLatentIndex(
        synth_latents, synth_labels, label_mode=args.hull_label_mode, seed=args.seed,
        include_self=args.hull_include_anchor, distance_space=args.hull_neighbor_distance_space,
        neighbor_mode=args.hull_neighbor_mode, neighbor_pool_size=args.hull_neighbor_pool_size,
        neighbor_pool_multiplier=args.hull_neighbor_pool_multiplier,
    ))
    for p in victim.model.parameters():
        p.requires_grad_(True)
    trainable_params = [p for p in victim.model.parameters() if p.requires_grad]
    teacher_model = None
    teacher_label_modes = {"mixed_soft", "teacher_soft", "latent_mixed_teacher"}
    if args.adv_label_mode in teacher_label_modes:
        teacher_model = copy.deepcopy(victim.model).to(args.device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print("[setup] frozen initial teacher enabled for soft adv labels")
    buffer = QualityAwareBuffer(max_size=args.qab_size)

    # EWA anchor snapshot — only trainable params (after PGD-freeze override)
    ewa_params = [p.data.clone().detach() for p in trainable_params]
    print(f"[setup] EWA anchor: {len(ewa_params)} param tensors snapshotted "
          f"({sum(p.numel() for p in ewa_params):,} elements)")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs,
                                  eta_min=args.lr * 0.01)

    if args.enable_vae_lhat:
        anchor_class_weight_map, anchor_class_weight_info = derive_kshot_anchor_class_weights(
            synth_labels, classes_in_scope, SUPER5_TO_IDX, mode=args.anchor_class_weight_mode,
            source_labels=source_meta.get("source_labels"),
            reference_source=args.anchor_class_weight_reference_source,
            gamma=args.anchor_class_weight_gamma, min_weight=args.anchor_class_weight_min,
            max_weight=args.anchor_class_weight_cap, missing_weight=args.anchor_class_missing_weight,
            manual_prior=parse_class_weight_map(args.anchor_class_weights),
        )
    else:
        anchor_class_weight_map, anchor_class_weight_info = {}, {"mode": "disabled_no_vae"}
    print(f"[setup] anchor class weight policy: {anchor_class_weight_info}", flush=True)

    log: Dict[str, Any] = {
        "args": vars(args),
        "comparison_contract": comparison_contract(),
        "class_trust": class_trust,
        "adaptation": {
            "n_trainable_tensors": len(trainable_params),
            "n_trainable_params": int(sum(p.numel() for p in trainable_params)),
        },
        "source_meta": {
            "source_names": source_meta["source_names"],
            "source_counts": dict(source_counts),
            "has_source_metadata": source_meta["has_source_metadata"],
        },
        "adv_label_policy": {
            "mode": args.adv_label_mode,
            "teacher_mix": args.adv_teacher_mix,
            "soft_target_floor": args.adv_soft_target_floor,
            "hull_mix_label_mode": args.hull_mix_label_mode,
            "hull_label_lambda_y": args.hull_label_lambda_y,
            "hull_label_positive": args.hull_label_positive,
            "hull_label_negative_floor": args.hull_label_negative_floor,
            "hull_label_new_class_cap": args.hull_label_new_class_cap,
        },
        "latent_hull_partner_selection": {
            "include_anchor": bool(args.hull_include_anchor),
            "distance_space": args.hull_neighbor_distance_space,
            "neighbor_mode": args.hull_neighbor_mode,
            "neighbor_pool_size": int(args.hull_neighbor_pool_size),
            "neighbor_pool_multiplier": int(args.hull_neighbor_pool_multiplier),
        },
        "anchor_sampling": {
            "classes_in_scope": list(classes_in_scope),
            "anchor_class_weights": anchor_class_weight_map,
            "anchor_class_weight_policy": anchor_class_weight_info,
            "K_anchor": int(args.K_anchor),
        },
        "latent_augmix_branch": {
            "enabled": True,
            "topology": "locked_three_chain",
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "mixture_mode": "beta",
            "mixture_prob": 0.5,
            "mixture_beta_a": None,
            "mixture_beta_b": None,
            "op_schedule": "random",
            "chain_weights": parse_float_sequence(args.latent_augmix_chain_weights),
            "signal_space": str(args.latent_augmix_signal_space),
            "corruption_source": (
                "vae_decode"
                if args.latent_augmix_third_chain_role == "vae_lhat_adversarial_waveform"
                else "target_real_clean_anchor"
            ),
            "third_chain_role": str(args.latent_augmix_third_chain_role),
            "chain_base_mode": str(args.latent_augmix_chain_base_mode),
            "adv_base_mix": float(args.latent_augmix_adv_base_mix),
            "severity": int(args.latent_augmix_severity),
            "severity_profile": str(args.latent_augmix_severity_profile),
            "severity_params_file": "",
            "severity_params_name": "",
            "ops": list(args.latent_augmix_ops),
            "renorm": True,
            "clip_abs": 6.0,
            "direct_consistency": {
                "enabled": bool(args.enable_latent_augmix_consistency),
                "consistency_weight": float(args.latent_augmix_consistency_weight),
                "consistency_loss": str(args.latent_augmix_consistency_loss),
                "bce_weight": float(args.latent_augmix_bce_weight),
                "max_batches": int(args.latent_augmix_consistency_max_batches),
            },
        },
        "epochs": [],
    }
    last_ckpt_path = os.path.join(args.output_dir, "last_model.pt")
    log_path = os.path.join(args.output_dir, "training_log.json")
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_latest_path = checkpoint_dir / "checkpoint_latest.pt"
    checkpoint_index_path = checkpoint_dir / "checkpoint_index.jsonl"
    diagnostics_epoch_path = Path(args.output_dir) / "diagnostics_epoch.jsonl"
    agent_decision_path = Path(args.output_dir) / "agent_decision.json"
    resume_path = resolve_resume_path(args.resume, args.output_dir)

    walker = load_optional_vae_component(args.enable_vae_lhat, lambda: StratifiedPoolWalker(
        labels_one_hot=synth_labels, classes_in_scope=classes_in_scope,
        class_to_idx=SUPER5_TO_IDX, seed=args.seed,
    ))
    walker_class_sizes = walker.class_sizes() if walker is not None else {}
    print(f"[setup] walker class sizes: {walker_class_sizes}")
    print(f"[setup] anchor class weights: {anchor_class_weight_map or {'<default>': 1.0}}")

    rng = np.random.default_rng(args.seed)
    consecutive_low_asr = 0
    start_epoch = 1
    if resume_path is None:
        for reset_path in (diagnostics_epoch_path, checkpoint_index_path):
            if reset_path.exists():
                reset_path.unlink()
    else:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        print(f"[resume] loading training checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=args.device)
        resume_mismatches = validate_resume_contract(
            ckpt.get("args"),
            vars(args),
            allow_drift=bool(args.allow_resume_config_drift),
        )
        if resume_mismatches:
            print(f"[resume-warning] allowing resume config drift: {resume_mismatches}", flush=True)
        victim.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "ewa_params" in ckpt:
            ewa_params = [p.to(args.device) for p in ckpt["ewa_params"]]
        restore_quality_buffer_state(buffer, ckpt.get("buffer_state", {}))
        if walker is not None:
            _restore_walker_state(walker, ckpt.get("walker_state", {}))
        restore_rng_state(ckpt.get("rng_state", {}), rng)
        log = ckpt.get("training_log", log)
        best_metric, best_epoch, best_source_floor_result = restore_best_selection_state(
            ckpt, (best_metric, best_epoch, best_source_floor_result)
        )
        realized_optimizer_steps = int(
            ckpt.get("realized_optimizer_steps", realized_optimizer_steps)
        )
        scheduler_steps = int(ckpt.get("scheduler_steps", scheduler_steps))
        consecutive_low_asr = int(ckpt.get("consecutive_low_asr", consecutive_low_asr))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"[resume] start_epoch={start_epoch} buffer={len(buffer)}",
            flush=True,
        )

    # ── Main loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):
        epoch_t0 = time.time()
        latent_augmix_direct_clean: np.ndarray | None = None
        latent_augmix_direct_views: np.ndarray | None = None
        latent_augmix_direct_labels: np.ndarray | None = None
        locked_mixed_view_mode = (
            args.latent_augmix_signal_space == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
        )
        decoupled_clean_augmix_mode = (
            args.latent_augmix_chain_base_mode == "all_clean_plus_vae_adv"
        )
        clean_anchor_mode = (
            not decoupled_clean_augmix_mode
            and (
                not args.enable_vae_lhat
                or
                args.latent_augmix_third_chain_role == "clean_anchor_control"
                or args.latent_augmix_chain_base_mode == "all_clean"
            )
        )
        clean_anchor_bundle: Dict[str, Any] | None = None

        k_per_cls: Dict[str, int] = {}
        victim.model.eval()
        if clean_anchor_mode:
            if target_real_ds is None:
                raise RuntimeError("clean-anchor AugMix requires target_real_ds")
            clean_anchor_source_ds = target_real_raw_augmix_ds or target_real_ds
            clean_anchor_bundle = build_clean_anchor_augmix_epoch(
                clean_anchor_source_ds, args, epoch
            )
            n_clean = int(clean_anchor_bundle["clean"].shape[0])
            anc_signals = clean_anchor_bundle["clean"]
            adv_signals = clean_anchor_bundle["views"][:n_clean]
            target_oh = clean_anchor_bundle["labels"]
            if clean_anchor_bundle["views"].shape[0] == 0:
                print(f"[ep{epoch:02d}] empty clean-anchor AugMix view set — skip epoch", flush=True)
                continue
            diff = adv_signals.reshape(adv_signals.shape[0], -1) - anc_signals.reshape(anc_signals.shape[0], -1)
            delta_norms = np.linalg.norm(diff, axis=1)
            delta_stats = {
                "mean_delta_norm": float(np.mean(delta_norms)),
                "max_delta_norm": float(np.max(delta_norms)),
            }
            k_per_cls = {"clean_anchor_control": n_clean}
            print(
                f"[ep{epoch:02d}] clean-anchor AugMix anchors: "
                f"k={k_per_cls['clean_anchor_control']} generated={clean_anchor_bundle['views'].shape[0]}",
                flush=True,
            )
        else:
            # Phase A: PGD on synth pool with the *current* victim.
            k_per_cls = weighted_anchor_quotas(
                classes_in_scope,
                walker_class_sizes,
                args.K_anchor,
                anchor_class_weight_map,
            )
            drawn = walker.sample(k_per_cls)
            print(f"[ep{epoch:02d}] anchor class quotas: {k_per_cls}", flush=True)
            all_picks = np.concatenate(
                [drawn[c] for c in classes_in_scope if drawn[c].size > 0]
            ) if any(drawn[c].size > 0 for c in classes_in_scope) else np.empty(0, dtype=np.int64)
            adv_signals, anc_signals, target_oh, delta_stats = run_pgd_on_synth_pool(
                pgd_gen=pgd_gen,
                synth_latents=synth_latents,
                synth_labels=synth_labels,
                pgd_batch=args.pgd_batch,
                device=args.device,
                picked_indices=all_picks,
                latent_hull_index=latent_hull_index,
                hull_M=args.hull_M,
                hull_mix_label_mode=args.hull_mix_label_mode,
                hull_label_lambda_y=args.hull_label_lambda_y,
                hull_label_positive=args.hull_label_positive,
                hull_label_negative_floor=args.hull_label_negative_floor,
                hull_label_new_class_cap=args.hull_label_new_class_cap,
                store_raw_decoded=locked_mixed_view_mode,
                pool_record_ids=source_meta.get("record_ids"),
            )
            if adv_signals.shape[0] == 0:
                print(f"[ep{epoch:02d}] empty walker pick — skip epoch", flush=True)
                continue

        latent_augmix_anchor_signals, latent_augmix_adv_signals, push_independent_vae_stream = (
            _resolve_latent_augmix_epoch_route(
                pgd_gen,
                anc_signals,
                adv_signals,
                args.latent_augmix_signal_space,
            )
            if args.enable_vae_lhat
            else (anc_signals, adv_signals, False)
        )

        # Phase B: gates and diagnostics.
        asr_info = compute_asr(victim, adv_signals, target_oh,
                               device=args.device, batch_size=128)
        decode_invalid_stats = decoded_signal_invalid_stats(adv_signals)
        attack_diagnostics = attack_bce_diagnostics(
            victim.model,
            anc_signals,
            None if clean_anchor_mode else pgd_gen.last_initial_signals_ptbxl_1000,
            adv_signals,
            target_oh,
            device=args.device,
            crop_len=args.crop_len,
        ) if args.enable_vae_lhat else {}
        sem_info = compute_semantic_gate(
            adv_signals, anc_signals,
            einthoven_p95_max=args.einthoven_p95_max,
        )

        gate_skipped = False
        latent_augmix_stats = {
            "enabled": True,
            "n_generated": 0,
        }
        if matched_comparison and args.enable_vae_lhat and not args.enable_raw_augmix:
            latent_augmix_stats = raw_augmix_disabled_stats(args.latent_augmix_signal_space)
        latent_augmix_push_stats = {}
        if (not args.disable_quality_gate) and (not sem_info.get("PASS", False)):
            gate_skipped = True
            print(f"[ep{epoch:02d}] medical gate FAIL: {sem_info.get('fail_reasons')} "
                  f"— skip buffer push this epoch", flush=True)
        else:
            if args.disable_quality_gate and not sem_info.get("PASS", False):
                print(f"[ep{epoch:02d}] medical gate FAIL ignored: "
                      f"{sem_info.get('fail_reasons')}", flush=True)
            push_stats = {}
            if clean_anchor_mode:
                latent_augmix_signals = clean_anchor_bundle["views"]
                latent_augmix_stats = clean_anchor_bundle["stats"]
                labels_rep = clean_anchor_bundle["labels_rep"]
                latent_augmix_clean_for_consistency = clean_anchor_bundle["clean"]
                latent_augmix_labels_for_consistency = clean_anchor_bundle["labels"]
            else:
                start = (adv_signals.shape[-1] - args.crop_len) // 2
                adv_ct_crop = adv_signals[..., start:start + args.crop_len]
                with torch.no_grad():
                    lg_chunks = []
                    teacher_prob_chunks = []
                    for i in range(0, adv_ct_crop.shape[0], 128):
                        x_t = torch.from_numpy(adv_ct_crop[i:i + 128]).float().to(args.device)
                        lg_chunks.append(victim.model(x_t).cpu().numpy())
                        if teacher_model is not None:
                            teacher_prob_chunks.append(torch.sigmoid(teacher_model(x_t)).cpu().numpy())
                    logits_arr = np.concatenate(lg_chunks)
                    teacher_probs_arr = (
                        np.concatenate(teacher_prob_chunks)
                        if teacher_prob_chunks else None
                    )
                push_stats = push_adv_to_buffer(
                    buffer=buffer, adv_signals_ct=adv_signals,
                    target_one_hot=target_oh, victim_logits=logits_arr,
                    crop_len=args.crop_len, class_trust=class_trust,
                    boundary_prob_min=args.boundary_prob_min,
                    boundary_prob_max=args.boundary_prob_max,
                    teacher_probs=teacher_probs_arr,
                    label_mode=args.adv_label_mode,
                    teacher_mix=args.adv_teacher_mix,
                    soft_target_floor=args.adv_soft_target_floor,
                    sample_weight_scale=float(args.vae_adv_stream_sample_scale),
                    enabled=push_independent_vae_stream and method_updates_enabled,
                )
                if not args.enable_raw_augmix:
                    if target_real_ds is None:
                        raise RuntimeError("clean budget control requires target_real_ds")
                    clean_anchor_bundle = build_no_raw_augmix_epoch(
                        target_real_ds,
                        args,
                        epoch,
                        matched_comparison=matched_comparison,
                    )
                    if clean_anchor_bundle is None:
                        latent_augmix_signals = np.empty(
                            (0,) + tuple(anc_signals.shape[1:]), dtype=np.float32
                        )
                    else:
                        latent_augmix_signals = clean_anchor_bundle["views"]
                        latent_augmix_stats = clean_anchor_bundle["stats"]
                        labels_rep = clean_anchor_bundle["labels_rep"]
                        latent_augmix_clean_for_consistency = clean_anchor_bundle["clean"]
                        latent_augmix_labels_for_consistency = clean_anchor_bundle["labels"]
                elif decoupled_clean_augmix_mode:
                    if target_real_ds is None:
                        raise RuntimeError("all_clean_plus_vae_adv requires target_real_ds")
                    clean_anchor_source_ds = target_real_raw_augmix_ds or target_real_ds
                    clean_anchor_bundle = build_clean_anchor_augmix_epoch(
                        clean_anchor_source_ds, args, epoch
                    )
                    latent_augmix_signals = clean_anchor_bundle["views"]
                    latent_augmix_stats = clean_anchor_bundle["stats"]
                    latent_augmix_stats["decoupled_vae_adv_stream"] = True
                    latent_augmix_stats["vae_adv_pushed"] = int(push_stats.get("n_pushed", 0))
                    labels_rep = clean_anchor_bundle["labels_rep"]
                    latent_augmix_clean_for_consistency = clean_anchor_bundle["clean"]
                    latent_augmix_labels_for_consistency = clean_anchor_bundle["labels"]
                else:
                    latent_augmix_clean_for_consistency = anc_signals
                    latent_augmix_labels_for_consistency = target_oh
                    latent_augmix_signals, latent_augmix_stats = build_three_chain_vae_lhat_augmix_views(
                        anchor_signals_ct=latent_augmix_anchor_signals,
                        adv_signals_ct=latent_augmix_adv_signals,
                        copies=args.latent_augmix_copies,
                        severity=args.latent_augmix_severity,
                        severity_profile=args.latent_augmix_severity_profile,
                        width=args.latent_augmix_width,
                        depth=args.latent_augmix_depth,
                        alpha=args.latent_augmix_alpha,
                        ops=list(args.latent_augmix_ops),
                        rng=rng,
                        third_chain_role=args.latent_augmix_third_chain_role,
                        chain_base_mode=args.latent_augmix_chain_base_mode,
                        chain_weights=parse_float_sequence(args.latent_augmix_chain_weights),
                        adv_base_mix=float(args.latent_augmix_adv_base_mix),
                        locked_raw_chain=locked_mixed_view_mode,
                    )
                    latent_augmix_stats["corruption_source"] = (
                        "vae_decode_raw" if locked_mixed_view_mode else "vae_decode"
                    )
                    latent_augmix_stats["signal_space"] = str(args.latent_augmix_signal_space)
            if latent_augmix_signals.shape[0] > 0:
                start = (latent_augmix_signals.shape[-1] - args.crop_len) // 2
                latent_augmix_ct_crop = latent_augmix_signals[..., start:start + args.crop_len]
                if decoupled_clean_augmix_mode:
                    labels_rep = clean_anchor_bundle["labels_rep"]
                elif not clean_anchor_mode:
                    labels_rep = np.tile(
                        target_oh,
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[:latent_augmix_signals.shape[0]]
                with torch.no_grad():
                    lg_chunks, teacher_prob_chunks = [], []
                    for i in range(0, latent_augmix_ct_crop.shape[0], 128):
                        x_t = torch.from_numpy(latent_augmix_ct_crop[i:i + 128]).float().to(args.device)
                        lg_chunks.append(victim.model(x_t).cpu().numpy())
                        if teacher_model is not None:
                            teacher_prob_chunks.append(torch.sigmoid(teacher_model(x_t)).cpu().numpy())
                    latent_augmix_logits_arr = np.concatenate(lg_chunks)
                    latent_augmix_teacher_probs_arr = (
                        np.concatenate(teacher_prob_chunks) if teacher_prob_chunks else None
                    )
                latent_augmix_push_stats = push_adv_to_buffer(
                    buffer=buffer,
                    adv_signals_ct=latent_augmix_signals,
                    target_one_hot=labels_rep,
                    victim_logits=latent_augmix_logits_arr,
                    crop_len=args.crop_len,
                    class_trust=class_trust,
                    boundary_prob_min=args.boundary_prob_min,
                    boundary_prob_max=args.boundary_prob_max,
                    teacher_probs=latent_augmix_teacher_probs_arr,
                    label_mode=args.adv_label_mode,
                    teacher_mix=args.adv_teacher_mix,
                    soft_target_floor=args.adv_soft_target_floor,
                    enabled=method_updates_enabled and args.enable_raw_augmix,
                )
                latent_augmix_direct_clean = latent_augmix_clean_for_consistency.astype(np.float32, copy=False)
                latent_augmix_direct_views = latent_augmix_signals.astype(np.float32, copy=False)
                latent_augmix_direct_labels = latent_augmix_labels_for_consistency.astype(np.float32, copy=False)
            print(
                f"[ep{epoch:02d}] latent-branch AugMix: "
                f"generated={latent_augmix_stats.get('n_generated', 0)} "
                f"pushed={latent_augmix_push_stats.get('n_pushed', 0)} "
                f"w_lat_mean={latent_augmix_stats.get('latent_weight_mean', latent_augmix_stats.get('adv_weight_mean', float('nan'))):.3f} "
                f"m_mean={latent_augmix_stats.get('beta_m_mean', float('nan')):.3f}",
                flush=True,
            )
        # Track consecutive low ASR
        if args.enable_vae_lhat and asr_info["asr_overall"] < args.asr_low_threshold:
            consecutive_low_asr += 1
        else:
            consecutive_low_asr = 0
        if consecutive_low_asr >= args.asr_consec_low_max:
            raise RuntimeError(
                f"PGD broken: ASR < {args.asr_low_threshold} for "
                f"{args.asr_consec_low_max} consecutive epochs — abort training.")

        if epoch > 1 and (epoch - 1) % args.rescore_interval == 0 and len(buffer) > 0:
            buffer.rescore(victim.model, args.device)

        # Phase C/D: train matched groups explicitly; keep the legacy sampler for unmatched runs.
        buf_ds = buffer.to_dataset()
        epoch_adv_weight = 0.0
        if buf_ds is not None and len(buf_ds) > 0:
            if args.adv_weight_warmup_epochs > 0:
                adv_scale = min(1.0, epoch / float(args.adv_weight_warmup_epochs))
            else:
                adv_scale = 1.0
            epoch_adv_weight = float(args.adv_weight) * adv_scale
        primary_objective_stats: Dict[str, Any] = {
            "enabled": False,
            "reason": "legacy_unmatched_sampler",
        }
        if matched_comparison:
            assert target_real_ds is not None and args.target_adv_fraction is not None
            target_clean_loader = DataLoader(
                target_real_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=args.num_workers > 0,
                generator=torch.Generator().manual_seed(int(args.seed) + int(epoch)),
            )
            primary_adv_signals = adv_signals if args.enable_vae_lhat else anc_signals
            target_adv_loader = DataLoader(
                _build_primary_adv_dataset(
                    primary_adv_signals,
                    target_oh,
                    crop_len=args.crop_len,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(int(args.seed) + int(epoch) + 100_000),
            )
            source_loader = (
                DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=args.num_workers > 0,
                    generator=torch.Generator().manual_seed(
                        int(args.seed) + int(epoch) + 200_000
                    ),
                )
                if args.ptbxl_weight > 0
                else None
            )
            primary_objective_stats = train_one_epoch_grouped_target_bce(
                model=victim.model,
                target_clean_loader=target_clean_loader,
                target_adv_loader=target_adv_loader,
                source_loader=source_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=args.device,
                target_adv_fraction=args.target_adv_fraction,
                source_coefficient=args.ptbxl_weight,
                target_coefficient=args.target_real_weight,
                grad_clip=args.grad_clip,
                ewa_params=ewa_params,
                anchor_lambda=args.anchor_lambda,
                ewa_decay=args.ewa_decay,
            )
            primary_objective_stats["enabled"] = True
            train_loss = float(primary_objective_stats["loss"])
            realized_epoch_optimizer_steps = int(primary_objective_stats["n_batches"])
        else:
            streams = []
            if args.ptbxl_weight > 0:
                streams.append((train_ds, args.ptbxl_weight, None))
            if target_real_ds is not None and args.target_real_weight > 0:
                streams.append((target_real_ds, args.target_real_weight, None))
            if buf_ds is not None and epoch_adv_weight > 0:
                streams.append((buf_ds, epoch_adv_weight, buffer.get_sampling_weights()))
            if len(streams) == 0:
                raise RuntimeError(
                    "No training streams are active. Check ptbxl_weight, "
                    "target_real_weight, and adv buffer gates."
                )
            if len(streams) == 1:
                only_ds = streams[0][0]
                train_loader = DataLoader(
                    only_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=args.num_workers > 0,
                    generator=torch.Generator().manual_seed(int(args.seed) + int(epoch)),
                )
            else:
                weights = []
                total_n = 0
                for ds_i, base_w, per_sample_w in streams:
                    n = len(ds_i)
                    if per_sample_w is None:
                        weights.extend([base_w] * n)
                    else:
                        weights.extend([base_w * max(w, 0.05) for w in per_sample_w])
                    total_n += n
                sampler = WeightedRandomSampler(
                    weights,
                    num_samples=total_n,
                    replacement=True,
                    generator=torch.Generator().manual_seed(int(args.seed) + int(epoch)),
                )
                combined = ConcatDataset([s[0] for s in streams])
                train_loader = DataLoader(
                    combined,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=args.num_workers > 0,
                )
            train_loss = train_one_epoch_masked_bce(
                victim.model, train_loader, optimizer, criterion, args.device,
                grad_clip=args.grad_clip, ewa_params=ewa_params,
                anchor_lambda=args.anchor_lambda, ewa_decay=args.ewa_decay,
            )
            realized_epoch_optimizer_steps = len(train_loader)
        zero_effect_aux_control = should_run_zero_effect_auxiliary_control(
            matched_comparison=matched_comparison,
            consistency_weight=float(args.latent_augmix_consistency_weight),
            bce_weight=float(args.latent_augmix_bce_weight),
        )
        if not args.enable_latent_augmix_consistency or (
            not zero_effect_aux_control and latent_augmix_direct_clean is None
        ):
            latent_augmix_consistency_stats = {
                "enabled": False,
                "reason": "no_latent_augmix_views_this_epoch",
                "loss": float("nan"),
                "bce_loss": float("nan"),
                "consistency_loss": float("nan"),
                "n_batches": 0,
                "n_generated": 0,
            }
        else:
            if zero_effect_aux_control:
                latent_augmix_consistency_stats = run_zero_effect_auxiliary_optimizer_control(
                    optimizer,
                    n_samples=int(args.K_anchor),
                    batch_size=args.batch_size,
                    max_batches=args.latent_augmix_consistency_max_batches,
                )
                latent_augmix_consistency_stats.update({
                    "n_generated": int(args.K_anchor) * int(args.latent_augmix_copies),
                    "copies": int(args.latent_augmix_copies),
                    "consistency_weight": 0.0,
                    "consistency_objective": str(args.latent_augmix_consistency_loss),
                    "bce_weight": 0.0,
                })
            else:
                assert latent_augmix_direct_clean is not None
                assert latent_augmix_direct_views is not None
                assert latent_augmix_direct_labels is not None
                latent_augmix_consistency_stats = train_latent_augmix_consistency_epoch(
                    model=victim.model,
                    clean_signals_ct=latent_augmix_direct_clean,
                    augmix_signals_ct=latent_augmix_direct_views,
                    labels_np=latent_augmix_direct_labels,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=args.device,
                    copies=args.latent_augmix_copies,
                    consistency_weight=args.latent_augmix_consistency_weight,
                    bce_weight=args.latent_augmix_bce_weight,
                    consistency_loss=args.latent_augmix_consistency_loss,
                    batch_size=args.batch_size,
                    crop_len=args.crop_len,
                    grad_clip=args.grad_clip,
                    trainable_params=trainable_params,
                    max_batches=args.latent_augmix_consistency_max_batches,
                )
                latent_augmix_consistency_stats["control"] = (
                    "vae_lhat_augmix_jsd" if method_updates_enabled else "clean_control_steps"
                )
            realized_epoch_optimizer_steps += int(latent_augmix_consistency_stats["n_batches"])
        vae_adv_consistency_stats = {
            "enabled": False,
            "reason": "disabled_or_no_vae_adv",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "n_batches": 0,
            "n_generated": 0,
        }
        if (
            float(args.vae_adv_consistency_weight) > 0.0
            and (not clean_anchor_mode)
            and adv_signals.shape[0] > 0
            and anc_signals.shape[0] == adv_signals.shape[0]
        ):
            vae_adv_consistency_stats = train_latent_augmix_consistency_epoch(
                model=victim.model,
                clean_signals_ct=anc_signals,
                augmix_signals_ct=adv_signals,
                labels_np=target_oh,
                optimizer=optimizer,
                criterion=criterion,
                device=args.device,
                copies=1,
                consistency_weight=args.vae_adv_consistency_weight,
                bce_weight=0.0,
                consistency_loss="soft_bce",
                batch_size=args.batch_size,
                crop_len=args.crop_len,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                max_batches=args.latent_augmix_consistency_max_batches,
            )
            realized_epoch_optimizer_steps += int(vae_adv_consistency_stats["n_batches"])
        if matched_comparison:
            realized_optimizer_steps += realized_epoch_optimizer_steps
        scheduler.step()
        if matched_comparison:
            scheduler_steps += 1

        # Phase E: PTBXL val loss
        source_val_metrics = evaluate_loader_macro(victim.model, val_loader, criterion, args.device)
        val_loss = float(source_val_metrics["loss"])
        target_val_metrics: Dict[str, float] = {}
        if matched_comparison:
            assert target_val_loader is not None
            target_val_metrics = evaluate_loader_macro(
                victim.model, target_val_loader, criterion, args.device
            )
            candidate_metric = float(target_val_metrics[args.selection_metric])
            source_floor_result = update_matched_checkpoint_selection(
                best_metric=best_metric,
                candidate_metric=candidate_metric,
                source_metric=float(source_val_metrics[args.selection_metric]),
                source_baseline_metric=float(source_baseline_metrics[args.selection_metric]),
                source_max_drop=args.source_floor_max_drop,
            )
            best_metric, best_epoch, best_source_floor_result = update_best_selection_state(
                best_metric, best_epoch, best_source_floor_result,
                source_floor_result, epoch=epoch,
            )
            if source_floor_result["selected"]:
                torch.save(victim.model.state_dict(), best_ckpt_path)

        elapsed = time.time() - epoch_t0
        latent_hull_diagnostics = delta_stats.get("latent_hull_diagnostics")
        latent_hull_diagnostics_summary = delta_stats.get("latent_hull_diagnostics_summary")
        entry = {
            "epoch": epoch,
            "comparison_arm": args.comparison_arm,
            "comparison_protocol": args.comparison_protocol or None,
            "comparison_variant": args.comparison_variant or None,
            "attack_family": "latent_hull" if args.enable_vae_lhat else None,
            "train_loss": round(train_loss, 4),
            "target_val_macro_auroc": target_val_metrics.get("macro_auroc"),
            "target_val_macro_auprc": target_val_metrics.get("macro_auprc"),
            "source_val_macro_auroc": source_val_metrics.get("macro_auroc"),
            "source_val_macro_auprc": source_val_metrics.get("macro_auprc"),
            "source_floor_passed": source_floor_result.get("source_floor_passed")
            if matched_comparison
            else None,
            "source_floor_result": source_floor_result if matched_comparison else {},
            "realized_optimizer_steps": realized_epoch_optimizer_steps
            if matched_comparison
            else None,
            "realized_optimizer_steps_total": realized_optimizer_steps
            if matched_comparison
            else None,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch
            if matched_comparison
            else None,
            "scheduler_steps": scheduler_steps if matched_comparison else None,
            "latent_augmix_consistency_loss": round(
                float(latent_augmix_consistency_stats.get("loss", float("nan"))), 6
            )
            if latent_augmix_consistency_stats.get("loss", float("nan"))
            == latent_augmix_consistency_stats.get("loss", float("nan"))
            else None,
            "latent_augmix_consistency_bce_loss": round(
                float(latent_augmix_consistency_stats.get("bce_loss", float("nan"))), 6
            )
            if latent_augmix_consistency_stats.get("bce_loss", float("nan"))
            == latent_augmix_consistency_stats.get("bce_loss", float("nan"))
            else None,
            "latent_augmix_consistency_objective_loss": round(
                float(latent_augmix_consistency_stats.get("consistency_loss", float("nan"))), 6
            )
            if latent_augmix_consistency_stats.get("consistency_loss", float("nan"))
            == latent_augmix_consistency_stats.get("consistency_loss", float("nan"))
            else None,
            "val_loss":   round(val_loss, 4),
            "asr_overall": round(float(asr_info["asr_overall"]), 4),
            "asr_per_class": {k: round(float(v), 4) for k, v in asr_info["per_class_asr"].items()},
            "multilabel_positive_label_asr": round(
                float(asr_info.get("multilabel_positive_label_asr", float("nan"))), 4
            ),
            "sample_any_positive_below_0p5_asr": round(
                float(asr_info.get("sample_any_positive_below_0p5_asr", float("nan"))), 4
            ),
            "sample_all_positive_below_0p5_asr": round(
                float(asr_info.get("sample_all_positive_below_0p5_asr", float("nan"))), 4
            ),
            "sample_all_positive_recognized_rate": round(
                float(asr_info.get("sample_all_positive_recognized_rate", float("nan"))), 4
            ),
            "atk_init": attack_diagnostics.get("atk_init"),
            "atk_anchor": attack_diagnostics.get("atk_anchor"),
            "attack_vs_init": attack_diagnostics.get("attack_vs_init"),
            "attack_vs_anchor": attack_diagnostics.get("attack_vs_anchor"),
            "clean_bce": attack_diagnostics.get("clean_bce"),
            "initial_bce": attack_diagnostics.get("initial_bce"),
            "adv_bce": attack_diagnostics.get("adv_bce"),
            "loss_gain": attack_diagnostics.get("loss_gain"),
            "decoded_invalid_rate": round(float(decode_invalid_stats["decoded_invalid_rate"]), 6)
            if decode_invalid_stats["decoded_invalid_rate"] == decode_invalid_stats["decoded_invalid_rate"]
            else None,
            "decoded_nan_rate": round(float(decode_invalid_stats["nan_rate"]), 6)
            if decode_invalid_stats["nan_rate"] == decode_invalid_stats["nan_rate"]
            else None,
            "decoded_flatline_rate": round(float(decode_invalid_stats["flatline_rate"]), 6)
            if decode_invalid_stats["flatline_rate"] == decode_invalid_stats["flatline_rate"]
            else None,
            "per_class_positive_label_asr": {
                k: round(float(v), 4)
                for k, v in asr_info.get("per_class_positive_label_asr", {}).items()
            },
            "einthoven_p95":  round(float(sem_info.get("einthoven_mean_p95", float('nan'))), 4),
            "hr_mean_delta": round(float(sem_info.get("hr_mean_delta", float('nan'))), 4),
            "qrs_amp_ratio": round(float(sem_info.get("qrs_amp_ratio", float('nan'))), 4)
                if sem_info.get("qrs_amp_ratio", None) == sem_info.get("qrs_amp_ratio", None)
                else None,
            "buffer_skipped": gate_skipped,
            "buffer_size":   len(buffer),
            "push_stats": push_stats if not gate_skipped else {},
            "latent_augmix_stats": latent_augmix_stats,
            "latent_augmix_push_stats": latent_augmix_push_stats,
            "primary_objective_stats": primary_objective_stats,
            "latent_augmix_consistency_stats": latent_augmix_consistency_stats,
            "target_adv_fraction_effective": args.target_adv_fraction,
            "vae_adv_consistency_stats": vae_adv_consistency_stats,
            "adv_weight_effective": round(float(epoch_adv_weight), 6),
            "adv_weight_warmup_epochs": int(args.adv_weight_warmup_epochs),
            "anchor_class_quotas": dict(k_per_cls),
            "delta_mean":    round(delta_stats["mean_delta_norm"], 4),
            "delta_max":     round(delta_stats["max_delta_norm"], 4),
            "latent_hull_diagnostics": latent_hull_diagnostics,
            "latent_hull_diagnostics_summary": latent_hull_diagnostics_summary,
            "latent_hull_final_effective_original_share_p50": (
                latent_hull_diagnostics_summary.get("final_effective_original_share", {}).get("p50")
                if latent_hull_diagnostics_summary else None
            ),
            "latent_hull_final_projection_scale_p50": (
                latent_hull_diagnostics_summary.get("final_projection_scale", {}).get("p50")
                if latent_hull_diagnostics_summary else None
            ),
            "lr":            round(optimizer.param_groups[0]["lr"], 6),
            "time_s":        round(elapsed, 1),
        }
        entry.update({
            "hull_M": args.hull_M,
            "hull_lambda": args.hull_lambda,
            "hull_steps": args.hull_steps,
            "hull_label_mode": args.hull_label_mode,
            "hull_mix_label_mode": args.hull_mix_label_mode,
            "hull_label_lambda_y": args.hull_label_lambda_y,
            "hull_label_positive": args.hull_label_positive,
            "hull_label_negative_floor": args.hull_label_negative_floor,
            "hull_label_new_class_cap": args.hull_label_new_class_cap,
            "hull_include_anchor": args.hull_include_anchor,
            "hull_neighbor_distance_space": args.hull_neighbor_distance_space,
            "hull_neighbor_mode": args.hull_neighbor_mode,
            "hull_neighbor_pool_size": args.hull_neighbor_pool_size,
            "hull_neighbor_pool_multiplier": args.hull_neighbor_pool_multiplier,
            "hull_weight_entropy_mean": round(
                float(delta_stats.get("hull_weight_entropy_mean", float('nan'))), 4
            ) if args.enable_vae_lhat else None,
            "hull_weight_top1_mean": round(
                float(delta_stats.get("hull_weight_top1_mean", float('nan'))), 4
            ) if args.enable_vae_lhat else None,
        })
        print(f"Ep {epoch:2d}/{args.n_epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"asr={asr_info['asr_overall']:.2f} "
              f"ml_any={asr_info.get('sample_any_positive_below_0p5_asr', float('nan')):.2f} "
              f"ml_pos={asr_info.get('multilabel_positive_label_asr', float('nan')):.2f} "
              f"eint_p95={sem_info.get('einthoven_mean_p95', float('nan')):.3f} "
              f"buf={len(buffer)} skip={gate_skipped} attack=latent_hull | {elapsed:.0f}s")

        if not args.final_checkpoint_only:
            torch.save(victim.model.state_dict(), last_ckpt_path)
        log["epochs"].append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

        decision = agent_attack_decision(
            entry,
            asr_low_threshold=args.asr_low_threshold,
            consecutive_low_asr=consecutive_low_asr,
        )
        decision_payload = {
            "schema_version": 1,
            "epoch": epoch,
            "run_dir": args.output_dir,
            **decision,
        }
        with agent_decision_path.open("w", encoding="utf-8") as f:
            json.dump(decision_payload, f, indent=2, sort_keys=True, ensure_ascii=True, default=str)
        if not args.final_checkpoint_only:
            diagnostics_payload = {
                "schema_version": 1,
                "epoch": epoch,
                "run_dir": args.output_dir,
                "status": "epoch_complete",
                "train_loss": entry.get("train_loss"),
                "val_loss": entry.get("val_loss"),
                "asr": entry.get("asr_overall"),
                "atk_init": entry.get("atk_init"),
                "atk_anchor": entry.get("atk_anchor"),
                "loss_gain": entry.get("loss_gain"),
                "clean_bce": entry.get("clean_bce"),
                "initial_bce": entry.get("initial_bce"),
                "adv_bce": entry.get("adv_bce"),
                "attack_vs_init": entry.get("attack_vs_init"),
                "attack_vs_anchor": entry.get("attack_vs_anchor"),
                "latent_hull_diagnostics_summary": entry.get("latent_hull_diagnostics_summary"),
                "latent_hull_final_effective_original_share_p50": entry.get(
                    "latent_hull_final_effective_original_share_p50"
                ),
                "latent_hull_final_projection_scale_p50": entry.get(
                    "latent_hull_final_projection_scale_p50"
                ),
                "decoded_invalid_rate": entry.get("decoded_invalid_rate"),
                "adv_weight_effective": entry.get("adv_weight_effective"),
                "buffer_size": entry.get("buffer_size"),
                "agent_decision": decision,
                "checkpoint_latest": str(checkpoint_latest_path),
            }
            append_jsonl(diagnostics_epoch_path, diagnostics_payload)

            ckpt_payload = {
                "schema_version": 1,
                "epoch": epoch,
                "global_step": epoch,
                "model_state_dict": victim.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ewa_params": [p.detach().cpu() for p in ewa_params],
                "buffer_state": quality_buffer_state(buffer),
                "walker_state": _walker_state(walker) if walker is not None else {},
                "rng_state": capture_rng_state(rng),
                "consecutive_low_asr": consecutive_low_asr,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_source_floor_result": best_source_floor_result,
                "realized_optimizer_steps": realized_optimizer_steps,
                "scheduler_steps": scheduler_steps,
                "training_log": log,
                "args": vars(args),
                "comparison_identity": (
                    f004_identity(args.target_adv_fraction)
                    if is_f004_rho_sweep(args.comparison_protocol)
                    else None
                ),
                "diagnostics_epoch_jsonl": str(diagnostics_epoch_path),
                "agent_decision_json": str(agent_decision_path),
            }
            atomic_torch_save(ckpt_payload, checkpoint_latest_path)
            latest_index = {
                "epoch": epoch,
                "path": str(checkpoint_latest_path),
                "kind": "latest",
                "agent_decision": decision["attack_state"],
            }
            append_jsonl(checkpoint_index_path, latest_index)

    # Final result
    if args.final_checkpoint_only:
        torch.save(victim.model.state_dict(), last_ckpt_path)
    final_contract = comparison_contract(realized_optimizer_steps, scheduler_steps)
    final = {
        "args":               vars(args),
        "comparison_contract": final_contract,
        "selected_checkpoint": best_ckpt_path if matched_comparison else last_ckpt_path,
        "best_model_path": best_ckpt_path if matched_comparison else None,
        "best_epoch": best_epoch if matched_comparison else None,
        "best_metric": best_metric if matched_comparison else None,
        "last_model_path":     last_ckpt_path,
        "n_epochs_run":       len(log["epochs"]),
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    if matched_comparison and matched_split is not None:
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as handle:
            json.dump({"args": vars(args), "comparison_contract": final_contract}, handle, indent=2)

    print("\n" + "=" * 72)
    print(f"Training done. last checkpoint → {last_ckpt_path}")


if __name__ == "__main__":
    main()
