"""Online AT on Super5 victim with synth-anchored PGD (Plan Rev 8).

Each epoch:
  A) Sample K_anchor latents (stratified by class) from a frozen Stage-1 synth
     pool (.latent.npz).
  B) Run K_pgd-step PGD on the current victim → (K, 12, 1000) adv signals +
     one-hot target labels.
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

NOT done in this fork (per Plan Rev 8 explicit non-goals):
  - AugMix latent injection (Issue #25 (b) — off in pilot).
  - K-sensitivity grid (Issue #25 (a) — K=200 only).
  - BoundaryAdvDiff (already known to fail -0.91pp; replaced with PGDAdvDiff).
  - CenterToken hook injection at training time (Stage 1 produced the synth
    latents already; the trainer only sees decoded signals).
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
    DataLoader, ConcatDataset, Dataset, WeightedRandomSampler,
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
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402
from methods.augmix.augmix import _apply_op  # noqa: E402
from methods.augmix.jsd_loss import jsd_multilabel  # noqa: E402
from methods.augmix.severity import AVAILABLE_OPS  # noqa: E402
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    STRESS_PROFILE_CHOICES as PN2021C_STRESS_PROFILE_CHOICES,
    build_corruption_op as _build_pn2021c_corruption_op,
    load_custom_severity_profile as _load_custom_severity_profile,
)

from ecg_adv_gen.training.online_buffer import (  # noqa: E402
    QualityAwareBuffer,
    center_crop_ct,
    train_one_epoch_masked_bce,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, NUM_SUPER5, get_super5_scheme  # noqa: E402
from ecg_adv_gen.labels.super5_mapping import SUPER5_TO_IDX  # noqa: E402
from ecg_adv_gen.models.super5_model_zoo import available_model_names  # noqa: E402
from ecg_adv_gen.training import (  # noqa: E402
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    compute_pos_weight,
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
    parse_class_source_weight_map,
    parse_class_weight_map,
    parse_source_weight_map,
    weighted_anchor_quotas,
)

DEFAULT_PTBXL_RAW = "/root/autodl-tmp/ptbxl/raw100.npy"
DEFAULT_PTBXL_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_PTBXL_PREP = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"


def _resolve_optional_custom_severity_profile(
    *,
    enabled: bool,
    severity_profile: str,
    params_file: str,
    params_name: str,
    flag_prefix: str,
) -> Optional[Dict[str, Dict[int, Dict[str, Any]]]]:
    """Load custom PN2021-C operator parameters only when that branch is active."""
    if not enabled:
        return None
    severity_profile = str(severity_profile)
    params_file = str(params_file or "")
    params_name = str(params_name or "")
    if severity_profile != "custom":
        if params_file or params_name:
            raise ValueError(
                f"{flag_prefix}_severity_params_file/name are only valid with "
                f"{flag_prefix}_severity_profile custom"
            )
        return None
    return _load_custom_severity_profile(params_file, params_name)


# Plan Rev 11/13: synth scope narrowed to 3 classes — HYP/CD synth disabled
# because their digital-GT validation fails 0/3 best-cell.
SUPER5_GEN_SUBSET = {"NORM", "MI", "STTC"}

def _parse_optional_float_list(raw: str | List[float] | Tuple[float, ...] | None) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    return [float(item) for item in raw]


# Indices of in-scope generation classes (NORM/MI/STTC) in the 5-class scheme.
SUPER5_GEN_SUBSET_IDX = sorted(SUPER5_TO_IDX[c] for c in SUPER5_GEN_SUBSET)


def _per_sample_global_zscore_np(signal_tc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    signal_tc = np.asarray(signal_tc, dtype=np.float32)
    mean = float(np.mean(signal_tc))
    std = float(np.std(signal_tc))
    return ((signal_tc - mean) / (std + eps)).astype(np.float32, copy=False)


def _normalize_target_real_signals(signals_tc: np.ndarray, norm_mode: str) -> np.ndarray:
    signals_tc = np.asarray(signals_tc, dtype=np.float32)
    if norm_mode == "pre_zscored":
        return signals_tc.astype(np.float32, copy=False)
    if norm_mode == "per_sample_global":
        if signals_tc.ndim != 3:
            raise ValueError(f"target-real signals must be 3D for z-score, got {signals_tc.shape}")
        mean = signals_tc.mean(axis=(1, 2), keepdims=True)
        std = signals_tc.std(axis=(1, 2), keepdims=True)
        return ((signals_tc - mean) / (std + 1e-8)).astype(np.float32, copy=False)
    raise ValueError(f"unknown target_real_norm_mode={norm_mode!r}")


def select_target_real_augmix_anchors_ct(
    target_real_signals_tc: np.ndarray,
    *,
    picked_indices: np.ndarray,
    expected_count: int,
) -> np.ndarray:
    """Select real K500 raw anchors for locked AugMix corruption chains."""
    signals = np.asarray(target_real_signals_tc, dtype=np.float32)
    if signals.ndim != 3 or signals.shape[1:] != (1000, 12):
        raise ValueError(f"target_real_signals_tc must be shaped (N,1000,12), got {signals.shape}")
    picks = np.asarray(picked_indices, dtype=np.int64)
    if picks.ndim != 1:
        raise ValueError(f"picked_indices must be 1D, got {picks.shape}")
    if int(expected_count) != int(picks.shape[0]):
        raise ValueError(f"expected_count={expected_count} does not match picked_indices={picks.shape[0]}")
    if picks.size == 0:
        return np.empty((0, 12, 1000), dtype=np.float32)
    if int(picks.min()) < 0 or int(picks.max()) >= signals.shape[0]:
        raise IndexError(
            f"picked_indices out of range for target_real_signals_tc length {signals.shape[0]}"
        )
    return np.ascontiguousarray(signals[picks].transpose(0, 2, 1)).astype(np.float32, copy=False)


def _zscore_ct_batch(signals_ct: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    signals = np.asarray(signals_ct, dtype=np.float32)
    if signals.ndim != 3:
        raise ValueError(f"signals_ct must be 3D, got {signals.shape}")
    mean = signals.mean(axis=(1, 2), keepdims=True)
    std = signals.std(axis=(1, 2), keepdims=True)
    return ((signals - mean) / (std + eps)).astype(np.float32, copy=False)


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
        clean_tc = _per_sample_global_zscore_np(raw_tc) if self.clean_norm_mode == "per_sample_global" else raw_tc
        clean_crop, raw_crop = _crop_signal_pair_tc(
            clean_tc,
            raw_tc,
            self.crop_len,
            mode="random" if self.mode == "train" else "center",
        )
        return (
            torch.from_numpy(np.ascontiguousarray(clean_crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
            torch.from_numpy(np.ascontiguousarray(raw_crop.T)).float(),
        )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
                clean_logits = model(clean)
                soft_targets = torch.sigmoid(clean_logits).detach()
            logits = model(views)
            soft_rep = soft_targets.repeat((int(copies), 1))

            mask = (labels_rep >= 0).float()
            labels_clamp = labels_rep.clamp(min=0.0)
            denom = mask.sum().clamp(min=1.0)
            hard_bce = (criterion(logits, labels_clamp) * mask).sum() / denom
            direct_consistency = (
                F.binary_cross_entropy_with_logits(logits, soft_rep, reduction="none") * mask
            ).sum() / denom
        else:
            clean_logits = model(clean)
            logits = model(views)
            logits_views = logits.view(int(copies), clean.shape[0], -1)

            mask_clean = (labels >= 0).float()
            denom_clean = mask_clean.sum().clamp(min=1.0)
            clean_hard_bce = (
                criterion(clean_logits, labels.clamp(min=0.0)) * mask_clean
            ).sum() / denom_clean

            mask_rep = (labels_rep >= 0).float()
            labels_rep_clamp = labels_rep.clamp(min=0.0)
            denom_rep = mask_rep.sum().clamp(min=1.0)
            aug_hard_bce = (criterion(logits, labels_rep_clamp) * mask_rep).sum() / denom_rep
            hard_bce = 0.5 * (clean_hard_bce + aug_hard_bce)

            jsd_terms = []
            for copy_i in range(int(copies)):
                copy_j = (copy_i + 1) % int(copies)
                jsd_terms.append(jsd_multilabel(clean_logits, logits_views[copy_i], logits_views[copy_j]))
            direct_consistency = torch.stack(jsd_terms).mean()

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


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    atomic_torch_save(payload, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    append_jsonl(path, payload)


def _resolve_resume_path(resume: str, output_dir: str) -> Optional[Path]:
    return resolve_resume_path(resume, output_dir)


def _buffer_state(buffer: QualityAwareBuffer) -> Dict[str, Any]:
    return quality_buffer_state(buffer)


def _restore_buffer_state(buffer: QualityAwareBuffer, state: Dict[str, Any]) -> None:
    restore_quality_buffer_state(buffer, state)


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


def _rng_state(epoch_rng: np.random.Generator) -> Dict[str, Any]:
    return capture_rng_state(epoch_rng)


def _restore_rng_state(state: Dict[str, Any], epoch_rng: np.random.Generator) -> None:
    restore_rng_state(state, epoch_rng)


@torch.no_grad()
def attack_bce_diagnostics(
    model: nn.Module,
    clean_signals_ct: np.ndarray,
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
    adv_logits = logits_for(adv_signals_ct)
    labels_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, labels_t, reduction="none").mean(dim=1)
    adv_bce = F.binary_cross_entropy_with_logits(adv_logits, labels_t, reduction="none").mean(dim=1)
    gain = adv_bce - clean_bce
    success = gain > 0.0
    return {
        "n": int(labels_t.shape[0]),
        "success_rate": float(success.float().mean().item()),
        "clean_bce_mean": float(clean_bce.mean().item()),
        "adv_bce_mean": float(adv_bce.mean().item()),
        "loss_gain_mean": float(gain.mean().item()),
        "loss_gain_p50": float(torch.quantile(gain, 0.50).item()),
        "loss_gain_p90": float(torch.quantile(gain, 0.90).item()),
    }


# ────────────────────────────────────────────────────────────────────────────
# PGD-on-frozen-synth-pool epoch step
# ────────────────────────────────────────────────────────────────────────────

def stratified_sample_synth(
    labels_npz: np.ndarray, K_anchor: int, num_classes: int, rng: np.random.Generator,
) -> np.ndarray:
    """LEGACY (Plan Rev 8): with-replacement stratified sample over all classes.

    Kept for backward compat / smoke. Plan Rev 13.2 onward uses StratifiedPoolWalker
    instead — that gives no-revisit-per-epoch + restricts to SUPER5_GEN_SUBSET.
    """
    cls_ids = labels_npz.argmax(axis=1)
    per_cls = max(1, K_anchor // num_classes)
    picked = []
    for c in range(num_classes):
        pool = np.where(cls_ids == c)[0]
        if len(pool) == 0:
            continue
        n_take = min(per_cls, len(pool))
        picked.extend(rng.choice(pool, size=n_take, replace=False).tolist())
    picked = list(set(picked))
    deficit = K_anchor - len(picked)
    if deficit > 0:
        rest = np.array([i for i in range(len(labels_npz)) if i not in set(picked)])
        if len(rest) > 0:
            picked.extend(rng.choice(rest, size=min(deficit, len(rest)), replace=False).tolist())
    return np.array(sorted(set(picked))[:K_anchor])


# Plan Rev 13.2 helper classes are imported from ecg_adv_gen.adaptation. They
# remain in this module's namespace for older scripts that import from here.


def run_pgd_on_synth_pool(
    pgd_gen: PGDAdvDiffGenerator,
    synth_latents: np.ndarray,    # (N, 4, 128)
    synth_labels: np.ndarray,     # (N, C) one-hot
    K_anchor: int,
    pgd_batch: int,
    rng: np.random.Generator,
    device: str,
    picked_indices: Optional[np.ndarray] = None,
    attack_mode: str = "pgd",
    latent_hull_index: Optional[SameLabelLatentIndex] = None,
    hull_M: int = 10,
    hull_mix_label_mode: str = "anchor",
    hull_label_lambda_y: float = 0.5,
    hull_label_positive: float = 0.95,
    hull_label_negative_floor: float = 0.0,
    hull_label_new_class_cap: float = 0.5,
    store_raw_decoded: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Sample K anchors stratified by class, run PGD in batches of pgd_batch.

    If `picked_indices` is given (e.g. from StratifiedPoolWalker), use it and
    skip the with-replacement stratified sampler. This is the Plan Rev 13.2
    path; the legacy path (no picked_indices) is kept for backward compat
    with smoke / Plan Rev 8.

    Returns:
      adv_signals_ct:   (K, 12, 1000)  float32
      anchor_signals_ct:(K, 12, 1000)  float32  (clean reference for sem-gate)
      labels_one_hot:   (K, C)         float32
      delta_stats:      mean / max L2 norm plus optional latent-hull weight stats
    """
    if picked_indices is None:
        num_classes = synth_labels.shape[1]
        pick = stratified_sample_synth(synth_labels, K_anchor, num_classes, rng)
    else:
        pick = picked_indices

    if len(pick) == 0:
        return (np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, synth_labels.shape[1]), dtype=np.float32),
                {"mean_delta_norm": float('nan'), "max_delta_norm": float('nan')})

    y_anchor_np = synth_labels[pick].astype(np.float32, copy=False)
    z_anchors = torch.from_numpy(synth_latents[pick]).float()        # (K, 4, 128)
    y_anchors = torch.from_numpy(y_anchor_np).float()                # (K, C)

    adv_chunks, anc_chunks, label_chunks, delta_norms = [], [], [], []
    raw_adv_chunks, raw_anc_chunks = [], []
    hull_entropies, hull_top1 = [], []
    for i in range(0, z_anchors.shape[0], pgd_batch):
        z_b = z_anchors[i:i + pgd_batch].to(device)
        y_b = y_anchors[i:i + pgd_batch].to(device)
        if attack_mode == "latent_hull":
            if latent_hull_index is None:
                raise ValueError("latent_hull_index is required when attack_mode=latent_hull")
            batch_pick = pick[i:i + pgd_batch]
            cand_np = latent_hull_index.candidates_for(batch_pick, hull_M)
            cand_b = torch.from_numpy(cand_np).float().to(device)
            x_adv, delta = pgd_gen.attack_from_latent(
                z_b, y_b, candidate_latents=cand_b
            )
            if hull_mix_label_mode == "anchor":
                label_chunks.append(y_anchor_np[i:i + pgd_batch])
            elif hull_mix_label_mode == "anchor_soft":
                cand_idx = getattr(latent_hull_index, "last_candidate_indices", None)
                weights_t = getattr(pgd_gen, "last_weights", None)
                if cand_idx is None or weights_t is None:
                    raise RuntimeError(
                        "latent-hull soft labels require candidate indices and weights"
                    )
                weights_np = weights_t.numpy().astype(np.float32, copy=False)
                cand_labels = synth_labels[cand_idx]
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
            if "hull_weight_entropy_mean" in hull_info:
                hull_entropies.append(float(hull_info["hull_weight_entropy_mean"]))
            if "hull_weight_top1_mean" in hull_info:
                hull_top1.append(float(hull_info["hull_weight_top1_mean"]))
        else:
            # Random init delta — Plan Issue #41 clean-anchor restart each epoch
            delta_init = torch.randn_like(z_b) * pgd_gen.delta_init_scale
            x_adv, delta = pgd_gen.attack_from_latent(z_b, y_b, delta_init=delta_init)
            label_chunks.append(y_anchor_np[i:i + pgd_batch])
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
    if hull_entropies:
        stats.update({
            "hull_weight_entropy_mean": float(np.mean(hull_entropies)),
            "hull_weight_top1_mean": float(np.mean(hull_top1)) if hull_top1 else float('nan'),
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
    mixture_mode: str = "beta",
    mixture_prob: float = 0.5,
    mixture_beta_a: Optional[float] = None,
    mixture_beta_b: Optional[float] = None,
    op_schedule: str = "random",
    chain_weights: Optional[List[float]] = None,
    ops: List[str],
    rng: np.random.Generator,
    renorm: bool = False,
    clip_abs: float = 6.0,
    severity_profile_params: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Locked wrapper: two ECG corruption chains plus one uncorrupted VAE-LHAT chain."""

    def _apply_augmix_op_np(
        sig_ct: np.ndarray,
        op_name: str,
        op_severity: int,
        op_severity_profile: str,
    ) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        if op_severity_profile == "standard":
            return _apply_op(sig_t, op_name, int(op_severity)).cpu().numpy().astype(np.float32, copy=False)
        op = _build_pn2021c_corruption_op(
            op_name,
            int(op_severity),
            op_severity_profile,
            severity_profile_params=severity_profile_params,
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
        mixture_mode=mixture_mode,
        mixture_prob=mixture_prob,
        mixture_beta_a=mixture_beta_a,
        mixture_beta_b=mixture_beta_b,
        op_schedule=op_schedule,
        chain_weights=chain_weights,
        ops=ops,
        rng=rng,
        op_apply_fn=_apply_augmix_op_np,
        available_ops=AVAILABLE_OPS,
        renorm=renorm,
        clip_abs=clip_abs,
    )


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
) -> Dict[str, int]:
    """Push gates-passed adv signals into the buffer with -1 sentinel labels.

    Label scheme: target_only (Plan Issue #28 — CheXpert U-Ignore + SPML).
    Non-target dims set to -1.0 → masked out by masked_bce_with_logits.

    Plan Rev 13.2: per-sample buffer push score is multiplied by class_trust
    (HYP/CD trust=0 → effectively dropped even if Stage 1 produced any). With
    the 3-class generation scope (NORM/MI/STTC) HYP/CD synth is already absent
    from the synth pool; this is a defense-in-depth check.
    """
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
        score = (1.0 - 2.0 * abs(prob_t - 0.5)) * trust                # ∈ [0, trust]
        buffer.add_one(
            torch.from_numpy(np.ascontiguousarray(sig_250)).float(),
            lbl,
            score,
        )
        n_pushed += 1
    return {
        "n_pushed": n_pushed,
        "n_dropped_by_trust": n_dropped_by_trust,
        "n_dropped_by_boundary": n_dropped_by_boundary,
        "label_mode": label_mode,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--center_name", required=True, help="cpsc_2018_extra | ningbo")
    p.add_argument("--ref_meta_json",
                   help="path to {tag}_k200.meta.json (for record_id exclusion in eval)")
    p.add_argument("--synth_npz", required=True,
                   help="Stage 1 latent npz: {latents (N,4,128), labels (N,5)}")
    p.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
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

    # PGD (Plan Rev 13: K_pgd=10 + ε=2.0 + K_anchor=300)
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_K", type=int, default=10)
    p.add_argument("--pgd_batch", type=int, default=32)        # Issue #45
    p.add_argument("--K_anchor", type=int, default=300)
    p.add_argument("--pgd_alpha", type=float, default=None)
    p.add_argument("--delta_init_scale", type=float, default=0.1)
    p.add_argument("--attack_mode", choices=["pgd", "latent_hull"], default="pgd",
                   help="pgd = free z0+delta PGD; latent_hull = same-label convex hull")
    p.add_argument("--hull_M", type=int, default=10,
                   help="Number of same-label candidate latents for latent_hull")
    p.add_argument("--hull_lambda", type=float, default=0.25)
    p.add_argument("--hull_steps", type=int, default=5)
    p.add_argument("--hull_lr", type=float, default=0.3)
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
    p.add_argument("--source_sampling_strategy",
                   choices=["class_balanced", "source_weighted"],
                   default="class_balanced",
                   help="class_balanced ignores source metadata; source_weighted "
                        "allocates each class quota by source weights when the "
                        "latent pool has source_ids/source_names.")
    p.add_argument("--source_weights", default=None,
                   help="Comma map for source_weighted, e.g. real_anchor=1.0,prompt_token=0.35")
    p.add_argument("--source_class_weights", default=None,
                   help="Comma overrides, e.g. MI:prompt_token=1.0,STTC:prompt_token=0.5")
    p.add_argument("--source_floor_per_class", type=int, default=0,
                   help="Minimum anchors per positive-weight source within each class quota.")
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
    p.add_argument("--adv_weight", type=float, default=0.5)
    p.add_argument(
        "--adv_weight_warmup_epochs",
        type=int,
        default=0,
        help=(
            "If >0, linearly ramp the adversarial buffer sampling weight from "
            "a small value to --adv_weight over this many epochs."
        ),
    )
    p.add_argument(
        "--enable_latent_augmix_branch",
        action="store_true",
        help=(
            "Stage-3 experiment: after each latent-hull adversarial decode, "
            "treat x_adv as one AugMix branch and mix it with ECG corruption "
            "chains from the clean anchor before pushing extra samples into "
            "the adversarial buffer."
        ),
    )
    p.add_argument(
        "--latent_augmix_topology",
        choices=["locked_three_chain"],
        default="locked_three_chain",
        help=(
            "locked_three_chain enforces the PN2021-C protocol topology: "
            "two raw ECG corruption chains plus one uncorrupted VAE-LHAT "
            "adversarial waveform chain."
        ),
    )
    p.add_argument("--latent_augmix_copies", type=int, default=1,
                   help="Number of latent-branch AugMix samples to create per x_adv.")
    p.add_argument("--latent_augmix_width", type=int, default=3,
                   help="Total AugMix branches; branch 0 is x_adv, remaining branches are ECG op chains.")
    p.add_argument("--latent_augmix_depth", type=int, default=-1,
                   help="Depth per ECG op chain; -1 samples uniformly from {1,2,3}.")
    p.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    p.add_argument("--latent_augmix_mixture_mode", choices=["beta", "fixed"], default="beta")
    p.add_argument("--latent_augmix_mixture_prob", type=float, default=0.5)
    p.add_argument("--latent_augmix_mixture_beta_a", type=float, default=0.0)
    p.add_argument("--latent_augmix_mixture_beta_b", type=float, default=0.0)
    p.add_argument(
        "--latent_augmix_op_schedule",
        choices=["random", "cycle", "per_op", "official_s5_depth23_composite_cycle"],
        default="random",
        help=(
            "Operator schedule for locked_three_chain raw corruption chains. "
            "random preserves historical AugMix behavior; per_op maps copies "
            "onto ops so every target anchor receives explicit single-operator "
            "views inside the three-chain graph."
        ),
    )
    p.add_argument(
        "--latent_augmix_chain_weights",
        default="",
        help=(
            "Optional comma-separated weights for the two raw corruption "
            "chains and the VAE-LHAT adversarial chain, e.g. 0.45,0.45,0.10. "
            "Empty keeps Dirichlet AugMix weights."
        ),
    )
    p.add_argument(
        "--latent_augmix_signal_space",
        choices=["model_zscore", "raw_pre_zscore"],
        default="model_zscore",
        help=(
            "Signal space used by locked_three_chain raw corruption chains. "
            "model_zscore preserves historical behavior. raw_pre_zscore applies "
            "corruption to VAE-decoded raw 100 Hz waveforms, then relies on the "
            "final AugMix renorm before model input."
        ),
    )
    p.add_argument(
        "--latent_augmix_corruption_source",
        choices=["vae_decode", "target_real"],
        default="vae_decode",
        help=(
            "Clean anchor source for locked_three_chain raw corruption chains. "
            "vae_decode preserves historical VAE-decoded anchors. target_real "
            "uses the same picked K500 raw ECG anchors for the two corruption "
            "chains while keeping the third chain as the VAE-LH adversarial waveform."
        ),
    )
    p.add_argument("--latent_augmix_severity", type=int, default=2)
    p.add_argument(
        "--latent_augmix_severity_profile",
        choices=PN2021C_STRESS_PROFILE_CHOICES,
        default="standard",
        help=(
            "Parameter profile used by non-latent latent-AugMix ECG op chains. "
            "'standard' preserves the original mild AugMix table; "
            "'calibrated_10to20pp' matches the strong PN2021-C evaluation profile."
        ),
    )
    p.add_argument(
        "--latent_augmix_severity_params_file",
        default="",
        help="YAML/JSON operator profile file used only with --latent_augmix_severity_profile custom.",
    )
    p.add_argument(
        "--latent_augmix_severity_params_name",
        default="",
        help="Profile name under top-level profiles used only with --latent_augmix_severity_profile custom.",
    )
    p.add_argument("--latent_augmix_latent_weight_cap", type=float, default=0.30,
                   help="Maximum Dirichlet weight assigned to the x_adv branch.")
    p.add_argument(
        "--latent_augmix_ops",
        nargs="+",
        default=["powerline_noise", "emg_noise", "baseline_wander", "baseline_shift"],
        choices=AVAILABLE_OPS,
        help="ECG corruption ops for non-latent AugMix branches. Random lead masking is excluded by default.",
    )
    p.add_argument("--no_latent_augmix_renorm", action="store_true",
                   help="Do not global-zscore the final latent-branch AugMix waveform before buffering.")
    p.add_argument("--latent_augmix_clip_abs", type=float, default=6.0,
                   help="Clip final latent-branch AugMix waveform after optional zscore; <=0 disables clipping.")
    p.add_argument(
        "--enable_latent_augmix_consistency",
        action="store_true",
        help=(
            "After each normal mixed epoch, directly train on the latent-AugMix "
            "views generated for that epoch with hard BCE plus soft-BCE/JSD."
        ),
    )
    p.add_argument("--latent_augmix_consistency_weight", type=float, default=2.0)
    p.add_argument(
        "--latent_augmix_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    p.add_argument("--latent_augmix_bce_weight", type=float, default=1.0)
    p.add_argument("--latent_augmix_consistency_max_batches", type=int, default=0)
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
    p.add_argument("--smoke", action="store_true",
                   help="Run only --n_epochs but with tiny subsets for sanity")
    return p.parse_args()


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


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Synth-anchored online AT (Super5) — Plan Rev 13.2")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 72)

    # ── Build Super5 victim early (needed for both sanity + training) ──────
    print("[setup] Loading ECGTwin (encoder + decoder, no text model)...")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
    print(f"[setup] Loading Super5 victim from {args.init_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=ecgtwin,
        num_classes=NUM_SUPER5,
        crop_len=args.crop_len,
        model_name=args.model_name,
    )

    # ── Load class_trust (required for training) ───────────────────────────
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
    latent_augmix_severity_profile_params = _resolve_optional_custom_severity_profile(
        enabled=bool(args.enable_latent_augmix_branch),
        severity_profile=args.latent_augmix_severity_profile,
        params_file=args.latent_augmix_severity_params_file,
        params_name=args.latent_augmix_severity_params_name,
        flag_prefix="--latent_augmix",
    )
    print(f"[setup] boundary target probability window=[{args.boundary_prob_min}, {args.boundary_prob_max}]")
    print(f"[setup] adv label mode={args.adv_label_mode} "
          f"teacher_mix={args.adv_teacher_mix} target_floor={args.adv_soft_target_floor}")
    print(f"[setup] hull mix label mode={args.hull_mix_label_mode} "
          f"lambda_y={args.hull_label_lambda_y} pos={args.hull_label_positive} "
          f"neg_floor={args.hull_label_negative_floor} new_cap={args.hull_label_new_class_cap}")
    if args.enable_latent_augmix_branch:
        print(
            "[setup] latent-branch AugMix enabled: "
            f"topology={args.latent_augmix_topology} "
            f"copies={args.latent_augmix_copies} width={args.latent_augmix_width} "
            f"depth={args.latent_augmix_depth} severity={args.latent_augmix_severity} "
            f"profile={args.latent_augmix_severity_profile} "
            f"mixture={args.latent_augmix_mixture_mode}:{args.latent_augmix_mixture_prob} "
            f"op_schedule={args.latent_augmix_op_schedule} "
            f"chain_weights={args.latent_augmix_chain_weights or 'dirichlet'} "
            f"signal_space={args.latent_augmix_signal_space} "
            f"corruption_source={args.latent_augmix_corruption_source} "
            f"w_lat_cap={args.latent_augmix_latent_weight_cap} "
            f"ops={args.latent_augmix_ops}",
            flush=True,
        )
        if args.enable_latent_augmix_consistency:
            print(
                "[setup] latent-branch AugMix direct consistency enabled: "
                f"loss={args.latent_augmix_consistency_loss} "
                f"weights=(consistency={args.latent_augmix_consistency_weight}, "
                f"bce={args.latent_augmix_bce_weight}) "
                f"max_batches={args.latent_augmix_consistency_max_batches}",
                flush=True,
            )
    elif args.enable_latent_augmix_consistency:
        raise ValueError("--enable_latent_augmix_consistency requires --enable_latent_augmix_branch")
    if args.enable_latent_augmix_branch and args.latent_augmix_topology == "locked_three_chain":
        if int(args.latent_augmix_width) != 3:
            raise ValueError("--latent_augmix_topology locked_three_chain requires --latent_augmix_width 3")
        if args.latent_augmix_signal_space == "raw_pre_zscore" and args.no_latent_augmix_renorm:
            raise ValueError(
                "--latent_augmix_signal_space raw_pre_zscore requires final latent-AugMix renorm "
                "so generated views match classifier model-input normalization"
            )
        if args.latent_augmix_corruption_source == "target_real":
            if args.latent_augmix_signal_space != "raw_pre_zscore":
                raise ValueError(
                    "--latent_augmix_corruption_source target_real requires "
                    "--latent_augmix_signal_space raw_pre_zscore"
                )
            if not args.target_real_npz:
                raise ValueError("--latent_augmix_corruption_source target_real requires --target_real_npz")
        if args.latent_augmix_op_schedule == "per_op" and int(args.latent_augmix_copies) < len(args.latent_augmix_ops):
            print(
                "[setup] warning: latent_augmix_op_schedule=per_op has fewer copies "
                "than ops; only the first scheduled operators will appear each epoch.",
                flush=True,
            )
    # ── Load synth pool (Stage 1 frozen) for training ──────────────────────
    synth_latents, synth_labels, synth_center, source_meta = load_synth_pool(args.synth_npz)
    print(f"[setup] synth pool: {synth_latents.shape} labels={synth_labels.shape} "
          f"center={synth_center}")
    cls_dist = synth_labels.argmax(1)
    from collections import Counter
    pool_class_counts = Counter(int(c) for c in cls_dist)
    print(f"[setup] synth class counts (idx): {dict(pool_class_counts)}")
    source_counts = Counter(str(s) for s in source_meta["source_labels"])
    print(f"[setup] synth source counts: {dict(source_counts)} "
          f"has_metadata={source_meta['has_source_metadata']}")
    if args.source_sampling_strategy == "source_weighted" and not source_meta["has_source_metadata"]:
        print("[setup] WARNING: source_weighted requested but pool lacks source metadata; "
              "all samples use source='unknown'.")

    # ── PTBXL super5 train / val ────────────────────────────────────────────
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

    if args.smoke:
        train_signals = train_signals[:512]
        train_labels = train_labels[:512]
        val_signals = val_signals[:128]
        val_labels = val_labels[:128]
        print("[smoke] truncated PTBXL train/val")

    train_ds = PTBXLDatasetScheme(train_signals, train_labels,
                                  crop_len=args.crop_len, mode='train')
    val_ds = PTBXLDatasetScheme(val_signals, val_labels,
                                crop_len=args.crop_len, mode='eval')
    target_real_ds = None
    target_real_augmix_signals_tc: np.ndarray | None = None
    if args.target_real_npz:
        with np.load(args.target_real_npz, allow_pickle=True) as real_data:
            real_signals = np.asarray(real_data["signals"], dtype=np.float32)
            real_labels = np.asarray(real_data["labels"], dtype=np.float32)
        if real_signals.ndim != 3:
            raise ValueError(f"target_real_npz signals must be 3D, got {real_signals.shape}")
        if real_signals.shape[1:] == (12, 1000):
            real_signals = real_signals.transpose(0, 2, 1)
        if real_signals.shape[1:] != (1000, 12):
            raise ValueError(f"target_real_npz signals must be (N,1000,12) or (N,12,1000), got {real_signals.shape}")
        if real_labels.shape[0] != real_signals.shape[0] or real_labels.shape[1] != NUM_SUPER5:
            raise ValueError(f"target_real_npz labels mismatch: signals={real_signals.shape} labels={real_labels.shape}")
        target_real_augmix_signals_tc = real_signals.astype(np.float32, copy=True)
        target_real_ds = TargetRealWaveformDataset(
            real_signals,
            real_labels,
            crop_len=args.crop_len,
            mode='train',
            norm_mode=args.target_real_norm_mode,
        )
        print(
            f"[setup] target-real supervised stream: n={len(target_real_ds)} "
            f"weight={args.target_real_weight} norm_mode={args.target_real_norm_mode} "
            f"path={args.target_real_npz}",
            flush=True,
        )
    if args.latent_augmix_corruption_source == "target_real":
        if target_real_augmix_signals_tc is None:
            raise ValueError("--latent_augmix_corruption_source target_real requires loaded target-real signals")
        if int(target_real_augmix_signals_tc.shape[0]) != int(synth_latents.shape[0]):
            raise ValueError(
                "--latent_augmix_corruption_source target_real requires target-real raw signals "
                f"to align 1:1 with latent anchors, got raw={target_real_augmix_signals_tc.shape[0]} "
                f"latents={synth_latents.shape[0]}"
            )

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5),
        dtype=torch.float32, device=args.device)
    print(f"[loss] pos_weight: {pos_weight.cpu().tolist()}")

    # reduction='none' for mask × bce (-1 sentinel handling)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── PGD / Latent-Hull generator + buffer ────────────────────────────────
    # Note: generator __init__ calls victim.parameters().requires_grad_(False)
    # which would prevent us from training the victim afterwards. We re-enable
    # requires_grad on all params right after, then snapshot EWA + build optimizer.
    latent_hull_index = None
    if args.attack_mode == "latent_hull":
        pgd_gen = LatentHullPGDGenerator(
            ecgtwin_wrapper=ecgtwin, victim=victim,
            epsilon=args.pgd_eps,
            hull_lambda=args.hull_lambda,
            hull_steps=args.hull_steps,
            hull_lr=args.hull_lr,
            weight_mode=args.hull_weight_mode,
            dirichlet_alpha=args.hull_dirichlet_alpha,
            device=args.device,
        )
        latent_hull_index = SameLabelLatentIndex(
            synth_latents, synth_labels,
            label_mode=args.hull_label_mode,
            seed=args.seed,
            include_self=args.hull_include_anchor,
            distance_space=args.hull_neighbor_distance_space,
            neighbor_mode=args.hull_neighbor_mode,
            neighbor_pool_size=args.hull_neighbor_pool_size,
            neighbor_pool_multiplier=args.hull_neighbor_pool_multiplier,
        )
        print(f"[setup] latent-hull index mode={args.hull_label_mode} "
              f"include_anchor={args.hull_include_anchor} "
              f"distance_space={args.hull_neighbor_distance_space} "
              f"neighbor_mode={args.hull_neighbor_mode} "
              f"neighbor_pool_size={args.hull_neighbor_pool_size} "
              f"neighbor_pool_multiplier={args.hull_neighbor_pool_multiplier} "
              f"sizes={latent_hull_index.class_sizes()}")
    else:
        pgd_gen = PGDAdvDiffGenerator(
            ecgtwin_wrapper=ecgtwin, victim=victim,
            epsilon=args.pgd_eps, K_pgd=args.pgd_K,
            alpha=args.pgd_alpha, delta_init_scale=args.delta_init_scale,
            device=args.device,
        )
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

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs,
                                  eta_min=args.lr * 0.01)

    source_weight_map = parse_source_weight_map(args.source_weights)
    source_class_weight_map = parse_class_source_weight_map(args.source_class_weights)
    manual_anchor_class_weight_map = parse_class_weight_map(args.anchor_class_weights)
    anchor_class_weight_map, anchor_class_weight_info = derive_kshot_anchor_class_weights(
        synth_labels,
        classes_in_scope,
        SUPER5_TO_IDX,
        mode=args.anchor_class_weight_mode,
        source_labels=source_meta.get("source_labels"),
        reference_source=args.anchor_class_weight_reference_source,
        gamma=args.anchor_class_weight_gamma,
        min_weight=args.anchor_class_weight_min,
        max_weight=args.anchor_class_weight_cap,
        missing_weight=args.anchor_class_missing_weight,
        manual_prior=manual_anchor_class_weight_map,
    )
    print(f"[setup] anchor class weight policy: {anchor_class_weight_info}", flush=True)

    log: Dict[str, Any] = {
        "args": vars(args),
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
            "enabled": bool(args.enable_latent_augmix_branch),
            "topology": str(args.latent_augmix_topology),
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "mixture_mode": str(args.latent_augmix_mixture_mode),
            "mixture_prob": float(args.latent_augmix_mixture_prob),
            "mixture_beta_a": float(args.latent_augmix_mixture_beta_a),
            "mixture_beta_b": float(args.latent_augmix_mixture_beta_b),
            "op_schedule": str(args.latent_augmix_op_schedule),
            "chain_weights": str(args.latent_augmix_chain_weights),
            "signal_space": str(args.latent_augmix_signal_space),
            "corruption_source": str(args.latent_augmix_corruption_source),
            "severity": int(args.latent_augmix_severity),
            "severity_profile": str(args.latent_augmix_severity_profile),
            "severity_params_file": str(args.latent_augmix_severity_params_file),
            "severity_params_name": str(args.latent_augmix_severity_params_name),
            "latent_weight_cap": float(args.latent_augmix_latent_weight_cap),
            "ops": list(args.latent_augmix_ops),
            "renorm": not bool(args.no_latent_augmix_renorm),
            "clip_abs": float(args.latent_augmix_clip_abs),
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
    resume_path = _resolve_resume_path(args.resume, args.output_dir)

    # Plan Rev 13.2: stratified pool walker over NORM/MI/STTC scope only
    walker = StratifiedPoolWalker(
        labels_one_hot=synth_labels,
        classes_in_scope=classes_in_scope,
        class_to_idx=SUPER5_TO_IDX, seed=args.seed,
        source_labels=source_meta["source_labels"],
        source_sampling_strategy=args.source_sampling_strategy,
        source_weights=source_weight_map,
        source_class_weights=source_class_weight_map,
        source_floor_per_class=args.source_floor_per_class,
    )
    walker_class_sizes = walker.class_sizes()
    print(f"[setup] walker class sizes: {walker_class_sizes}")
    print(f"[setup] anchor class weights: {anchor_class_weight_map or {'<default>': 1.0}}")
    if args.source_sampling_strategy == "source_weighted":
        print(f"[setup] walker source-class sizes: {walker.source_class_sizes()}")
        print(f"[setup] source weights: global={source_weight_map or {'<default>': 1.0}} "
              f"class_overrides={source_class_weight_map or {}} "
              f"floor_per_class={args.source_floor_per_class}")

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
        _restore_buffer_state(buffer, ckpt.get("buffer_state", {}))
        _restore_walker_state(walker, ckpt.get("walker_state", {}))
        _restore_rng_state(ckpt.get("rng_state", {}), rng)
        log = ckpt.get("training_log", log)
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

        # Phase A: PGD on synth pool with the *current* victim
        # Plan Rev 13.2: StratifiedPoolWalker draws no-revisit-per-epoch,
        # restricted to NORM/MI/STTC scope.
        k_per_cls: Dict[str, int] = {}
        victim.model.eval()
        k_per_cls = weighted_anchor_quotas(
            classes_in_scope,
            walker_class_sizes,
            args.K_anchor,
            anchor_class_weight_map,
        )
        drawn = walker.sample(k_per_cls)
        print(f"[ep{epoch:02d}] anchor class quotas: {k_per_cls}", flush=True)
        if args.source_sampling_strategy == "source_weighted":
            print(f"[ep{epoch:02d}] anchor source counts: {walker.last_source_counts} "
                  f"class_source={walker.last_class_source_counts}", flush=True)
        all_picks = np.concatenate(
            [drawn[c] for c in classes_in_scope if drawn[c].size > 0]
        ) if any(drawn[c].size > 0 for c in classes_in_scope) else np.empty(0, dtype=np.int64)
        adv_signals, anc_signals, target_oh, delta_stats = run_pgd_on_synth_pool(
            pgd_gen=pgd_gen,
            synth_latents=synth_latents,
            synth_labels=synth_labels,
            K_anchor=args.K_anchor,
            pgd_batch=args.pgd_batch,
            rng=rng, device=args.device,
            picked_indices=all_picks,
            attack_mode=args.attack_mode,
            latent_hull_index=latent_hull_index,
            hull_M=args.hull_M,
            hull_mix_label_mode=args.hull_mix_label_mode,
            hull_label_lambda_y=args.hull_label_lambda_y,
            hull_label_positive=args.hull_label_positive,
            hull_label_negative_floor=args.hull_label_negative_floor,
            hull_label_new_class_cap=args.hull_label_new_class_cap,
            store_raw_decoded=(
                bool(args.enable_latent_augmix_branch)
                and args.latent_augmix_topology == "locked_three_chain"
                and args.latent_augmix_signal_space == "raw_pre_zscore"
            ),
        )
        if adv_signals.shape[0] == 0:
            print(f"[ep{epoch:02d}] empty walker pick — skip epoch", flush=True)
            continue

        # Phase B: gates
        # ASR (signals → victim)
        asr_info = compute_asr(victim, adv_signals, target_oh,
                               device=args.device, batch_size=128)
        decode_invalid_stats = decoded_signal_invalid_stats(adv_signals)
        attack_vs_anchor_stats = attack_bce_diagnostics(
            victim.model,
            anc_signals,
            adv_signals,
            target_oh,
            device=args.device,
            crop_len=args.crop_len,
        )
        # Semantic (Einthoven, HR, QRS)
        sem_info = compute_semantic_gate(
            adv_signals, anc_signals,
            einthoven_p95_max=args.einthoven_p95_max,
        )

        gate_skipped = False
        latent_augmix_stats = {
            "enabled": bool(args.enable_latent_augmix_branch),
            "n_generated": 0,
        }
        latent_augmix_push_stats = {}
        if (not args.disable_quality_gate) and (not sem_info.get("PASS", False)):
            gate_skipped = True
            print(f"[ep{epoch:02d}] medical gate FAIL: {sem_info.get('fail_reasons')} "
                  f"— skip buffer push this epoch", flush=True)
        else:
            if args.disable_quality_gate and not sem_info.get("PASS", False):
                print(f"[ep{epoch:02d}] medical gate FAIL ignored: "
                      f"{sem_info.get('fail_reasons')}", flush=True)
            # Center-crop adv (B, 12, 1000) → (B, 12, crop_len) on the time axis
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
            )
            if args.enable_latent_augmix_branch:
                augmix_anchor_signals = anc_signals
                augmix_adv_signals = adv_signals
                latent_augmix_clean_for_consistency = anc_signals
                if (
                    args.latent_augmix_topology == "locked_three_chain"
                    and args.latent_augmix_signal_space == "raw_pre_zscore"
                ):
                    augmix_anchor_signals = getattr(pgd_gen, "last_anchor_raw_ptbxl_1000", None)
                    augmix_adv_signals = getattr(pgd_gen, "last_adv_raw_ptbxl_1000", None)
                    if augmix_anchor_signals is None or augmix_adv_signals is None:
                        raise RuntimeError(
                            "raw_pre_zscore latent AugMix requested but raw decoded PGD signals were not cached"
                        )
                    if args.latent_augmix_corruption_source == "target_real":
                        if target_real_augmix_signals_tc is None:
                            raise RuntimeError(
                                "target_real latent AugMix corruption source requested but no target-real raw signals are loaded"
                            )
                        augmix_anchor_signals = select_target_real_augmix_anchors_ct(
                            target_real_augmix_signals_tc,
                            picked_indices=all_picks,
                            expected_count=adv_signals.shape[0],
                        )
                        latent_augmix_clean_for_consistency = _zscore_ct_batch(augmix_anchor_signals)
                latent_augmix_signals, latent_augmix_stats = build_three_chain_vae_lhat_augmix_views(
                    anchor_signals_ct=augmix_anchor_signals,
                    adv_signals_ct=augmix_adv_signals,
                    copies=args.latent_augmix_copies,
                    severity=args.latent_augmix_severity,
                    severity_profile=args.latent_augmix_severity_profile,
                    severity_profile_params=latent_augmix_severity_profile_params,
                    width=args.latent_augmix_width,
                    depth=args.latent_augmix_depth,
                    alpha=args.latent_augmix_alpha,
                    mixture_mode=args.latent_augmix_mixture_mode,
                    mixture_prob=args.latent_augmix_mixture_prob,
                    mixture_beta_a=None
                    if args.latent_augmix_mixture_beta_a <= 0
                    else args.latent_augmix_mixture_beta_a,
                    mixture_beta_b=None
                    if args.latent_augmix_mixture_beta_b <= 0
                    else args.latent_augmix_mixture_beta_b,
                    op_schedule=args.latent_augmix_op_schedule,
                    chain_weights=_parse_optional_float_list(args.latent_augmix_chain_weights),
                    ops=list(args.latent_augmix_ops),
                    rng=rng,
                    renorm=not args.no_latent_augmix_renorm,
                    clip_abs=args.latent_augmix_clip_abs,
                )
                latent_augmix_stats["corruption_source"] = str(args.latent_augmix_corruption_source)
                if latent_augmix_signals.shape[0] > 0:
                    start = (latent_augmix_signals.shape[-1] - args.crop_len) // 2
                    latent_augmix_ct_crop = latent_augmix_signals[..., start:start + args.crop_len]
                    labels_rep = np.tile(
                        target_oh,
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[:latent_augmix_signals.shape[0]]
                    with torch.no_grad():
                        lg_chunks = []
                        teacher_prob_chunks = []
                        for i in range(0, latent_augmix_ct_crop.shape[0], 128):
                            x_t = torch.from_numpy(
                                latent_augmix_ct_crop[i:i + 128]
                            ).float().to(args.device)
                            lg_chunks.append(victim.model(x_t).cpu().numpy())
                            if teacher_model is not None:
                                teacher_prob_chunks.append(
                                    torch.sigmoid(teacher_model(x_t)).cpu().numpy()
                                )
                        latent_augmix_logits_arr = np.concatenate(lg_chunks)
                        latent_augmix_teacher_probs_arr = (
                            np.concatenate(teacher_prob_chunks)
                            if teacher_prob_chunks else None
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
                    )
                    if args.enable_latent_augmix_consistency:
                        latent_augmix_direct_clean = latent_augmix_clean_for_consistency.astype(np.float32, copy=False)
                        latent_augmix_direct_views = latent_augmix_signals.astype(np.float32, copy=False)
                        latent_augmix_direct_labels = target_oh.astype(np.float32, copy=False)
                print(
                    f"[ep{epoch:02d}] latent-branch AugMix: "
                    f"generated={latent_augmix_stats.get('n_generated', 0)} "
                    f"pushed={latent_augmix_push_stats.get('n_pushed', 0)} "
                    f"w_lat_mean={latent_augmix_stats.get('latent_weight_mean', latent_augmix_stats.get('adv_weight_mean', float('nan'))):.3f} "
                    f"m_mean={latent_augmix_stats.get('beta_m_mean', float('nan')):.3f}",
                    flush=True,
                )
        # Track consecutive low ASR
        if asr_info["asr_overall"] < args.asr_low_threshold:
            consecutive_low_asr += 1
        else:
            consecutive_low_asr = 0
        if consecutive_low_asr >= args.asr_consec_low_max:
            raise RuntimeError(
                f"PGD broken: ASR < {args.asr_low_threshold} for "
                f"{args.asr_consec_low_max} consecutive epochs — abort training.")

        if epoch > 1 and (epoch - 1) % args.rescore_interval == 0 and len(buffer) > 0:
            buffer.rescore(victim.model, args.device)

        # Phase C: build mixed loader (cold-start guard for empty buffer)
        buf_ds = buffer.to_dataset()
        streams = []
        if args.ptbxl_weight > 0:
            streams.append((train_ds, args.ptbxl_weight, None))
        if target_real_ds is not None and args.target_real_weight > 0:
            streams.append((target_real_ds, args.target_real_weight, None))
        epoch_adv_weight = 0.0
        if buf_ds is not None and len(buf_ds) > 0:
            if args.adv_weight_warmup_epochs > 0:
                adv_scale = min(1.0, epoch / float(args.adv_weight_warmup_epochs))
            else:
                adv_scale = 1.0
            epoch_adv_weight = float(args.adv_weight) * adv_scale
            if epoch_adv_weight > 0:
                streams.append((buf_ds, epoch_adv_weight, buffer.get_sampling_weights()))

        if len(streams) == 0:
            raise RuntimeError(
                "No training streams are active. Check ptbxl_weight, "
                "target_real_weight, and adv buffer gates."
            )
        if len(streams) == 1:
            only_ds = streams[0][0]
            train_loader = DataLoader(only_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=args.num_workers > 0)
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
            sampler = WeightedRandomSampler(weights, num_samples=total_n, replacement=True)
            combined = ConcatDataset([s[0] for s in streams])
            train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=args.num_workers > 0)

        # Phase D: train
        train_loss = train_one_epoch_masked_bce(
            victim.model, train_loader, optimizer, criterion, args.device,
            grad_clip=args.grad_clip, ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda, ewa_decay=args.ewa_decay,
        )
        latent_augmix_consistency_stats = {"enabled": False, "reason": "disabled"}
        if args.enable_latent_augmix_consistency:
            if (
                latent_augmix_direct_clean is None
                or latent_augmix_direct_views is None
                or latent_augmix_direct_labels is None
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
        scheduler.step()

        # Phase E: PTBXL val loss
        victim.model.eval()
        val_losses = []
        with torch.no_grad():
            for sigs, labels in val_loader:
                sigs = sigs.to(args.device)
                labels = labels.to(args.device)
                logits = victim.model(sigs)
                mask = (labels >= 0).float()
                labels_safe = torch.where(mask.bool(), labels, torch.zeros_like(labels))
                per_elem = criterion(logits, labels_safe)
                denom = mask.sum().clamp(min=1.0)
                vl = (per_elem * mask).sum() / denom
                val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        elapsed = time.time() - epoch_t0
        entry = {
            "epoch": epoch,
            "attack_mode": args.attack_mode,
            "train_loss": round(train_loss, 4),
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
            "atk_init": None,
            "atk_init_reason": "not_available_for_current_latent_hull_generator",
            "atk_anchor": attack_vs_anchor_stats.get("success_rate"),
            "attack_vs_anchor": attack_vs_anchor_stats,
            "clean_bce": attack_vs_anchor_stats.get("clean_bce_mean"),
            "adv_bce": attack_vs_anchor_stats.get("adv_bce_mean"),
            "loss_gain": attack_vs_anchor_stats.get("loss_gain_mean"),
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
            "latent_augmix_consistency_stats": latent_augmix_consistency_stats,
            "adv_weight_effective": round(float(epoch_adv_weight), 6),
            "adv_weight_warmup_epochs": int(args.adv_weight_warmup_epochs),
            "anchor_class_quotas": dict(k_per_cls),
            "delta_mean":    round(delta_stats["mean_delta_norm"], 4),
            "delta_max":     round(delta_stats["max_delta_norm"], 4),
            "lr":            round(optimizer.param_groups[0]["lr"], 6),
            "time_s":        round(elapsed, 1),
        }
        if args.source_sampling_strategy == "source_weighted":
            entry.update({
                "anchor_source_counts": dict(walker.last_source_counts),
                "anchor_class_source_counts": walker.last_class_source_counts,
            })
        if args.attack_mode == "latent_hull":
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
                ),
                "hull_weight_top1_mean": round(
                    float(delta_stats.get("hull_weight_top1_mean", float('nan'))), 4
                ),
            })
        print(f"Ep {epoch:2d}/{args.n_epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"asr={asr_info['asr_overall']:.2f} "
              f"ml_any={asr_info.get('sample_any_positive_below_0p5_asr', float('nan')):.2f} "
              f"ml_pos={asr_info.get('multilabel_positive_label_asr', float('nan')):.2f} "
              f"eint_p95={sem_info.get('einthoven_mean_p95', float('nan')):.3f} "
              f"buf={len(buffer)} skip={gate_skipped} attack={args.attack_mode} | {elapsed:.0f}s")

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
            "adv_bce": entry.get("adv_bce"),
            "decoded_invalid_rate": entry.get("decoded_invalid_rate"),
            "adv_weight_effective": entry.get("adv_weight_effective"),
            "buffer_size": entry.get("buffer_size"),
            "agent_decision": decision,
            "checkpoint_latest": str(checkpoint_latest_path),
        }
        _append_jsonl(diagnostics_epoch_path, diagnostics_payload)

        ckpt_payload = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": epoch,
            "model_state_dict": victim.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ewa_params": [p.detach().cpu() for p in ewa_params],
            "buffer_state": _buffer_state(buffer),
            "walker_state": _walker_state(walker),
            "rng_state": _rng_state(rng),
            "consecutive_low_asr": consecutive_low_asr,
            "training_log": log,
            "args": vars(args),
            "diagnostics_epoch_jsonl": str(diagnostics_epoch_path),
            "agent_decision_json": str(agent_decision_path),
        }
        _atomic_torch_save(ckpt_payload, checkpoint_latest_path)
        latest_index = {
            "epoch": epoch,
            "path": str(checkpoint_latest_path),
            "kind": "latest",
            "agent_decision": decision["attack_state"],
        }
        _append_jsonl(checkpoint_index_path, latest_index)

    # Final result
    final = {
        "args":               vars(args),
        "selected_checkpoint": last_ckpt_path,
        "last_model_path":     last_ckpt_path,
        "n_epochs_run":       len(log["epochs"]),
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"Training done. last checkpoint → {last_ckpt_path}")


if __name__ == "__main__":
    main()
