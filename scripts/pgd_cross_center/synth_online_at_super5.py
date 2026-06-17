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
       roundtrip   weight 0.5    (Plan Issue #29 — keep against AugMix-off)
       adv buffer  weight 2.0    (cold-start guard: weight=0 in epoch 0/empty)
  F) train_one_epoch with masked BCE on the -1 sentinel + EWA anchor regularizer
     (Plan Issue #21 Q4 — ADR ICLR 2024 EMA self-distill).
  G) every eval_every epoch: PTB-XL fold9 val + PN2021 quick subset macro AUROC,
     update best ckpt.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import (
    DataLoader, ConcatDataset, TensorDataset, WeightedRandomSampler,
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
from scripts.triple_labels.eval_pn2021_corruptions import (  # noqa: E402
    STRESS_PROFILE_CHOICES as PN2021C_STRESS_PROFILE_CHOICES,
    _build_corruption_op as _build_pn2021c_corruption_op,
)

from scripts.crosscenter_tierM.online_adv_train_tierM import (  # noqa: E402
    _center_crop_ct,
    QualityAwareBufferTierM as QualityAwareBuffer,
    build_roundtrip_anchor_dataset,
    train_one_epoch_tierM as train_one_epoch_masked_bce,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5, NUM_SUPER5, snomed_list_to_super5,
)
from scripts.triple_labels.model_zoo import available_model_names  # noqa: E402
from ecg_adv_gen.training import (  # noqa: E402
    append_jsonl,
    atomic_torch_save,
    build_checkpoint_selection_record,
    capture_rng_state,
    compute_pos_weight,
    quality_buffer_state,
    resolve_quick_eval_plan,
    resolve_resume_path,
    restore_quality_buffer_state,
    restore_rng_state,
    should_save_initial_best_model,
    validate_resume_contract,
)
from ecg_adv_gen.models.ecgfounder_torch import (  # noqa: E402
    global_zscore_torch,
    stabilize_ecg_torch,
)
from ecg_adv_gen.data.latent_pools import (  # noqa: E402
    LatentPoolError,
    load_synth_pool as _load_synth_pool,
)
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    PTBXLDatasetScheme, compute_macro_auroc_auprc,
    evaluate, get_ptbxl_labels_for_scheme, preprocess_ptbxl_all,
)
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    agent_attack_decision,
    auroc_to_trust,
    build_anchor_preserving_soft_labels,
    build_adv_buffer_label,
    build_k500_internal_val_mask,
    build_latent_augmix_branch_signals as _build_latent_augmix_branch_signals_core,
    build_raw_augmix_views as _build_raw_augmix_views_core,
    build_raw_corruption_views as _build_raw_corruption_views_core,
    derive_class_trust,
    derive_kshot_anchor_class_weights,
    decoded_signal_invalid_stats,
    parse_class_source_weight_map,
    parse_class_weight_map,
    parse_source_weight_map,
    weighted_anchor_quotas,
)

DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"
DEFAULT_PTBXL_RAW = "/root/autodl-tmp/ptbxl/raw100.npy"
DEFAULT_PTBXL_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_PTBXL_PREP = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"

PN2021_FORBIDDEN = {"ptb-xl", "ptbxl"}     # never eval against this shard

# Plan Rev 11/13: synth scope narrowed to 3 classes — HYP/CD synth disabled
# because their digital-GT validation fails 0/3 best-cell.
SUPER5_GEN_SUBSET = {"NORM", "MI", "STTC"}

# Plan Rev 11/13: hardcoded distrust regardless of Stage 0.4 sanity output
# (HYP synth fails Sokolow voltage; CD synth fails QRS broadening).
DEFAULT_TRUST_HARDCODE = {"HYP": 0.0, "CD": 0.0}


def _has_raw_input_stabilizer(config: Optional[Dict[str, Any]]) -> bool:
    config = dict(config or {})
    return bool(
        config.get("bandpass_low_hz") is not None
        or config.get("bandpass_high_hz") is not None
        or config.get("repair_flat_leads")
        or config.get("clip_abs") is not None
        or config.get("renorm_after_stabilizer")
    )


def _raw_input_stabilizer_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "bandpass_low_hz": args.raw_input_bandpass_low_hz,
        "bandpass_high_hz": args.raw_input_bandpass_high_hz,
        "repair_flat_leads": bool(args.raw_input_repair_flat_leads),
        "clip_abs": args.raw_input_clip_abs,
        "renorm_after_stabilizer": bool(args.raw_input_renorm_after_stabilizer),
        "sample_rate_hz": float(args.raw_input_sample_rate_hz),
    }


def _apply_raw_input_stabilizer_np(sig_ct: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    if not _has_raw_input_stabilizer(config):
        return np.asarray(sig_ct, dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(sig_ct)).float().unsqueeze(0)
    y = stabilize_ecg_torch(
        x,
        sample_rate_hz=float(config.get("sample_rate_hz", 100.0)),
        bandpass_low_hz=config.get("bandpass_low_hz"),
        bandpass_high_hz=config.get("bandpass_high_hz"),
        repair_flat_leads=bool(config.get("repair_flat_leads", False)),
        clip_abs=config.get("clip_abs"),
    )
    if bool(config.get("renorm_after_stabilizer", False)):
        y = global_zscore_torch(y)
    return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _raw_input_stabilizer_postprocess(
    config: Optional[Dict[str, Any]],
) -> tuple[Optional[Callable[[np.ndarray], np.ndarray]], Optional[str]]:
    config = dict(config or {})
    if not _has_raw_input_stabilizer(config):
        return None, None

    def _postprocess(sig_ct: np.ndarray) -> np.ndarray:
        return _apply_raw_input_stabilizer_np(sig_ct, config)

    parts: list[str] = []
    if config.get("bandpass_low_hz") is not None or config.get("bandpass_high_hz") is not None:
        parts.append(f"bandpass{config.get('bandpass_low_hz')}-{config.get('bandpass_high_hz')}")
    if config.get("repair_flat_leads"):
        parts.append("repairflat")
    if config.get("renorm_after_stabilizer"):
        parts.append("renorm")
    return _postprocess, "+".join(parts) or "raw_input_stabilizer"

from scripts.triple_labels.label_schemes import SUPER5_TO_IDX  # noqa: E402

# Indices of in-scope generation classes (NORM/MI/STTC) in the 5-class scheme.
SUPER5_GEN_SUBSET_IDX = sorted(SUPER5_TO_IDX[c] for c in SUPER5_GEN_SUBSET)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_source_logit_anchor_epoch(
    model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
    weight: float,
    max_batches: int = 0,
    grad_clip: float = 0.0,
    trainable_params: Optional[List[nn.Parameter]] = None,
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
) -> float:
    """One lightweight source-consistency pass against the frozen PTB-XL teacher.

    This is used after the mixed target/adv epoch to reduce PTB-XL source
    forgetting. It does not change labels; it only constrains source logits.
    """
    if weight <= 0:
        return float("nan")
    model.train()
    if freeze_backbone_eval_fn is not None:
        freeze_backbone_eval_fn()
    teacher_model.eval()
    grad_params = trainable_params if trainable_params is not None else list(model.parameters())
    losses: List[float] = []
    for batch_i, batch in enumerate(loader, start=1):
        signals = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        with torch.no_grad():
            teacher_logits = teacher_model(signals)
        loss = F.mse_loss(logits, teacher_logits) * float(weight)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(grad_params, grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        if max_batches > 0 and batch_i >= max_batches:
            break
    return float(np.mean(losses)) if losses else float("nan")


def train_one_epoch_masked_bce_freeze_aware(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
    trainable_params: List[nn.Parameter],
    ewa_params: Optional[List[torch.Tensor]],
    anchor_lambda: float,
    ewa_decay: float,
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
) -> float:
    """Masked BCE epoch for frozen-backbone adaptation.

    The shared Tier-M helper calls ``model.train()`` internally and anchors by
    zipping over all model parameters. That is correct for full-model training,
    but wrong for classifier-only adaptation: frozen BatchNorm modules would
    update running statistics, and the EWA anchor list would no longer align
    with trainable parameters. This local variant keeps the backbone in eval
    mode and applies anchor/grad clipping only to the trainable head.
    """
    model.train()
    if freeze_backbone_eval_fn is not None:
        freeze_backbone_eval_fn()
    losses: List[float] = []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        mask = (labels >= 0).float()
        labels_clamp = labels.clamp(min=0.0)
        per_elem = criterion(logits, labels_clamp)
        denom = mask.sum().clamp(min=1.0)
        bce = (per_elem * mask).sum() / denom
        if ewa_params is not None and anchor_lambda > 0:
            anchor = sum(
                (p - p_anchor.detach()).pow(2).sum()
                for p, p_anchor in zip(trainable_params, ewa_params)
            )
            loss = bce + anchor_lambda * anchor
        else:
            loss = bce
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        if ewa_params is not None and ewa_decay > 0 and ewa_decay < 1.0:
            with torch.no_grad():
                for p, p_anchor in zip(trainable_params, ewa_params):
                    p_anchor.mul_(ewa_decay).add_(p.data, alpha=1 - ewa_decay)
        losses.append(float(bce.item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_raw_corruption_consistency_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: str,
    *,
    copies: int,
    severity: int,
    severity_profile: str,
    ops: List[str],
    prob: float,
    consistency_weight: float,
    bce_weight: float,
    consistency_loss: str,
    rng: np.random.Generator,
    grad_clip: float,
    trainable_params: List[nn.Parameter],
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
    max_batches: int = 0,
    renorm: bool = False,
    clip_abs: float = 6.0,
    view_mode: str = "single_op",
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    augmix_mixture_mode: str = "beta",
    augmix_mixture_prob: float = 0.5,
    augmix_mixture_beta_a: float = 0.0,
    augmix_mixture_beta_b: float = 0.0,
    input_stabilizer_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train on raw ECG corruptions with clean-model consistency targets."""
    if copies <= 0 or (consistency_weight <= 0 and bce_weight <= 0):
        return {
            "enabled": False,
            "reason": "disabled_or_zero_weight",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "n_batches": 0,
            "n_generated": 0,
            "n_corrupted": 0,
            "op_counts": {},
        }
    consistency_loss = str(consistency_loss)
    if consistency_loss not in {"soft_bce", "jsd"}:
        raise ValueError(f"unknown raw corruption consistency loss: {consistency_loss}")
    if consistency_loss == "jsd" and copies < 2:
        raise ValueError("--raw_corrupt_consistency_loss jsd requires --raw_corrupt_copies >= 2")
    severity_profile = str(severity_profile)
    if severity_profile not in PN2021C_STRESS_PROFILE_CHOICES:
        raise ValueError(f"unknown raw corruption severity profile: {severity_profile}")
    view_mode = str(view_mode)
    if view_mode not in {"single_op", "augmix"}:
        raise ValueError(f"unknown raw corruption view mode: {view_mode}")

    losses: List[float] = []
    bce_losses: List[float] = []
    consistency_losses: List[float] = []
    n_generated = 0
    n_corrupted = 0
    op_counts: Dict[str, int] = {}
    postprocess_names: set[str] = set()

    for batch_i, (signals, labels) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_np = signals.detach().cpu().numpy().astype(np.float32, copy=False)
        if view_mode == "augmix":
            corrupt_np, stats = build_raw_augmix_views(
                clean_np,
                copies=copies,
                severity=severity,
                severity_profile=severity_profile,
                width=augmix_width,
                depth=augmix_depth,
                alpha=augmix_alpha,
                mixture_mode=augmix_mixture_mode,
                mixture_prob=augmix_mixture_prob,
                mixture_beta_a=None if augmix_mixture_beta_a <= 0 else augmix_mixture_beta_a,
                mixture_beta_b=None if augmix_mixture_beta_b <= 0 else augmix_mixture_beta_b,
                ops=ops,
                rng=rng,
                renorm=renorm,
                clip_abs=clip_abs,
                input_stabilizer_config=input_stabilizer_config,
            )
        else:
            corrupt_np, stats = build_raw_corruption_views(
                clean_np,
                copies=copies,
                severity=severity,
                severity_profile=severity_profile,
                ops=ops,
                prob=prob,
                rng=rng,
                renorm=renorm,
                clip_abs=clip_abs,
                input_stabilizer_config=input_stabilizer_config,
            )
        if corrupt_np.shape[0] == 0:
            continue

        labels_rep = labels.repeat((int(copies), 1))
        corrupt = torch.from_numpy(corrupt_np).float().to(device, non_blocking=True)

        model.train()
        if freeze_backbone_eval_fn is not None:
            freeze_backbone_eval_fn()
        optimizer.zero_grad(set_to_none=True)
        if consistency_loss == "soft_bce":
            with torch.no_grad():
                clean_logits = model(signals)
                soft_targets = torch.sigmoid(clean_logits).detach()
            logits = model(corrupt)
            soft_rep = soft_targets.repeat((int(copies), 1))

            mask = (labels_rep >= 0).float()
            labels_clamp = labels_rep.clamp(min=0.0)
            denom = mask.sum().clamp(min=1.0)
            hard_bce = (criterion(logits, labels_clamp) * mask).sum() / denom
            raw_consistency = (
                F.binary_cross_entropy_with_logits(logits, soft_rep, reduction="none") * mask
            ).sum() / denom
        else:
            clean_logits = model(signals)
            logits = model(corrupt)
            logits_views = logits.view(int(copies), signals.shape[0], -1)

            mask_clean = (labels >= 0).float()
            denom_clean = mask_clean.sum().clamp(min=1.0)
            clean_hard_bce = (
                criterion(clean_logits, labels.clamp(min=0.0)) * mask_clean
            ).sum() / denom_clean

            mask_rep = (labels_rep >= 0).float()
            labels_rep_clamp = labels_rep.clamp(min=0.0)
            denom_rep = mask_rep.sum().clamp(min=1.0)
            corrupt_hard_bce = (criterion(logits, labels_rep_clamp) * mask_rep).sum() / denom_rep
            hard_bce = 0.5 * (clean_hard_bce + corrupt_hard_bce)

            jsd_terms = []
            for copy_i in range(int(copies)):
                copy_j = (copy_i + 1) % int(copies)
                jsd_terms.append(jsd_multilabel(clean_logits, logits_views[copy_i], logits_views[copy_j]))
            raw_consistency = torch.stack(jsd_terms).mean()

        loss = float(bce_weight) * hard_bce + float(consistency_weight) * raw_consistency
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        bce_losses.append(float(hard_bce.item()))
        consistency_losses.append(float(raw_consistency.item()))
        n_generated += int(stats.get("n_generated", 0))
        n_corrupted += int(stats.get("n_corrupted", 0))
        if stats.get("postprocess"):
            postprocess_names.add(str(stats["postprocess"]))
        for op_name, count in dict(stats.get("op_counts", {})).items():
            op_counts[str(op_name)] = op_counts.get(str(op_name), 0) + int(count)
        if max_batches > 0 and batch_i >= max_batches:
            break
    postprocess_name = None
    if postprocess_names:
        postprocess_name = next(iter(postprocess_names)) if len(postprocess_names) == 1 else "+".join(sorted(postprocess_names))
    elif _has_raw_input_stabilizer(input_stabilizer_config):
        _unused_postprocess_fn, postprocess_name = _raw_input_stabilizer_postprocess(input_stabilizer_config)

    result = {
        "enabled": True,
        "view_mode": view_mode,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else float("nan"),
        "n_batches": len(losses),
        "n_generated": int(n_generated),
        "n_corrupted": int(n_corrupted),
        "corrupt_fraction": float(n_corrupted / max(1, n_generated)),
        "copies": int(copies),
        "severity": int(severity),
        "severity_profile": str(severity_profile),
        "prob": float(prob),
        "consistency_weight": float(consistency_weight),
        "consistency_objective": str(consistency_loss),
        "bce_weight": float(bce_weight),
        "ops": list(ops),
        "op_counts": op_counts,
        "renorm": bool(renorm),
        "clip_abs": float(clip_abs),
        "augmix_width": int(augmix_width) if view_mode == "augmix" else None,
        "augmix_depth": int(augmix_depth) if view_mode == "augmix" else None,
        "augmix_alpha": float(augmix_alpha) if view_mode == "augmix" else None,
        "augmix_mixture_mode": str(augmix_mixture_mode) if view_mode == "augmix" else None,
        "augmix_mixture_prob": float(augmix_mixture_prob) if view_mode == "augmix" else None,
        "augmix_mixture_beta_a": float(augmix_mixture_beta_a) if view_mode == "augmix" else None,
        "augmix_mixture_beta_b": float(augmix_mixture_beta_b) if view_mode == "augmix" else None,
    }
    if _has_raw_input_stabilizer(input_stabilizer_config):
        result["input_stabilizer"] = dict(input_stabilizer_config or {})
        result["postprocess"] = postprocess_name or "raw_input_stabilizer"
    return result


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
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
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
        if freeze_backbone_eval_fn is not None:
            freeze_backbone_eval_fn()
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


def train_mask_shift_consistency_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: str,
    *,
    copies: int,
    mask_severity: int,
    shift_severity: int,
    consistency_weight: float,
    bce_weight: float,
    consistency_loss: str,
    rng: np.random.Generator,
    grad_clip: float,
    trainable_params: List[nn.Parameter],
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
    max_batches: int = 0,
    renorm: bool = False,
    clip_abs: float = 6.0,
) -> Dict[str, Any]:
    """Train deterministic lead-mask and baseline-shift invariance views."""
    if copies <= 0 or (consistency_weight <= 0 and bce_weight <= 0):
        return {
            "enabled": False,
            "reason": "disabled_or_zero_weight",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "n_batches": 0,
            "n_generated": 0,
            "n_mask_generated": 0,
            "n_shift_generated": 0,
        }
    consistency_loss = str(consistency_loss)
    if consistency_loss not in {"soft_bce", "jsd"}:
        raise ValueError(f"unknown mask-shift consistency loss: {consistency_loss}")

    losses: List[float] = []
    bce_losses: List[float] = []
    consistency_losses: List[float] = []
    n_mask_generated = 0
    n_shift_generated = 0
    n_mask_corrupted = 0
    n_shift_corrupted = 0

    for batch_i, (signals, labels) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_np = signals.detach().cpu().numpy().astype(np.float32, copy=False)
        mask_np, mask_stats = build_raw_corruption_views(
            clean_np,
            copies=copies,
            severity=mask_severity,
            ops=["random_leads_masking"],
            prob=1.0,
            rng=rng,
            renorm=renorm,
            clip_abs=clip_abs,
        )
        shift_np, shift_stats = build_raw_corruption_views(
            clean_np,
            copies=copies,
            severity=shift_severity,
            ops=["baseline_shift"],
            prob=1.0,
            rng=rng,
            renorm=renorm,
            clip_abs=clip_abs,
        )
        corrupt_np = np.concatenate([mask_np, shift_np], axis=0)
        total_views = int(corrupt_np.shape[0] // max(1, signals.shape[0]))
        if corrupt_np.shape[0] == 0 or total_views <= 0:
            continue

        labels_rep = labels.repeat((total_views, 1))
        corrupt = torch.from_numpy(corrupt_np).float().to(device, non_blocking=True)

        model.train()
        if freeze_backbone_eval_fn is not None:
            freeze_backbone_eval_fn()
        optimizer.zero_grad(set_to_none=True)
        if consistency_loss == "soft_bce":
            with torch.no_grad():
                clean_logits = model(signals)
                soft_targets = torch.sigmoid(clean_logits).detach()
            logits = model(corrupt)
            soft_rep = soft_targets.repeat((total_views, 1))

            mask = (labels_rep >= 0).float()
            labels_clamp = labels_rep.clamp(min=0.0)
            denom = mask.sum().clamp(min=1.0)
            hard_bce = (criterion(logits, labels_clamp) * mask).sum() / denom
            mask_shift_consistency = (
                F.binary_cross_entropy_with_logits(logits, soft_rep, reduction="none") * mask
            ).sum() / denom
        else:
            clean_logits = model(signals)
            logits = model(corrupt)
            logits_views = logits.view(total_views, signals.shape[0], -1)

            mask_clean = (labels >= 0).float()
            denom_clean = mask_clean.sum().clamp(min=1.0)
            clean_hard_bce = (
                criterion(clean_logits, labels.clamp(min=0.0)) * mask_clean
            ).sum() / denom_clean

            mask_rep = (labels_rep >= 0).float()
            labels_rep_clamp = labels_rep.clamp(min=0.0)
            denom_rep = mask_rep.sum().clamp(min=1.0)
            corrupt_hard_bce = (criterion(logits, labels_rep_clamp) * mask_rep).sum() / denom_rep
            hard_bce = 0.5 * (clean_hard_bce + corrupt_hard_bce)

            mask_logits = logits_views[: int(copies)]
            shift_logits = logits_views[int(copies): int(copies) * 2]
            jsd_terms = [
                jsd_multilabel(clean_logits, mask_logits[copy_i], shift_logits[copy_i])
                for copy_i in range(int(copies))
            ]
            mask_shift_consistency = torch.stack(jsd_terms).mean()

        loss = float(bce_weight) * hard_bce + float(consistency_weight) * mask_shift_consistency
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        bce_losses.append(float(hard_bce.item()))
        consistency_losses.append(float(mask_shift_consistency.item()))
        n_mask_generated += int(mask_stats.get("n_generated", 0))
        n_shift_generated += int(shift_stats.get("n_generated", 0))
        n_mask_corrupted += int(mask_stats.get("n_corrupted", 0))
        n_shift_corrupted += int(shift_stats.get("n_corrupted", 0))
        if max_batches > 0 and batch_i >= max_batches:
            break

    n_generated = n_mask_generated + n_shift_generated
    n_corrupted = n_mask_corrupted + n_shift_corrupted
    return {
        "enabled": True,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else float("nan"),
        "n_batches": len(losses),
        "n_generated": int(n_generated),
        "n_corrupted": int(n_corrupted),
        "corrupt_fraction": float(n_corrupted / max(1, n_generated)),
        "n_mask_generated": int(n_mask_generated),
        "n_shift_generated": int(n_shift_generated),
        "n_mask_corrupted": int(n_mask_corrupted),
        "n_shift_corrupted": int(n_shift_corrupted),
        "copies_per_op": int(copies),
        "mask_severity": int(mask_severity),
        "shift_severity": int(shift_severity),
        "consistency_weight": float(consistency_weight),
        "consistency_objective": str(consistency_loss),
        "bce_weight": float(bce_weight),
        "ops": ["random_leads_masking", "baseline_shift"],
        "op_counts": {
            "random_leads_masking": int(n_mask_corrupted),
            "baseline_shift": int(n_shift_corrupted),
        },
        "renorm": bool(renorm),
        "clip_abs": float(clip_abs),
    }


def configure_classifier_only_adaptation(
    model: nn.Module,
    train_final_norm: bool = False,
    adapter_type: str = "linear",
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
) -> Tuple[List[nn.Parameter], Callable[[], None]]:
    """Freeze EfficientNet1DV2 backbone and train only the classifier head.

    `adapter_type=linear` trains the existing classifier. `adapter_type=lora`
    trains a low-rank residual on top of the final classifier Linear; checkpoints
    are later folded back to the normal EfficientNet state_dict.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    if not hasattr(model, "classifier"):
        raise ValueError("classifier-only adaptation requires model.classifier")
    if adapter_type == "linear":
        for p in model.classifier.parameters():
            p.requires_grad_(True)
    elif adapter_type == "lora":
        attach_foldable_lora_classifier(model, rank=lora_rank, alpha=lora_alpha)
        for module in model.modules():
            if isinstance(module, FoldableLowRankLinear):
                for p in module.down.parameters():
                    p.requires_grad_(True)
                for p in module.up.parameters():
                    p.requires_grad_(True)
    else:
        raise ValueError(f"unsupported classifier adapter_type={adapter_type!r}")
    if train_final_norm and hasattr(model, "final_norm"):
        for p in model.final_norm.parameters():
            p.requires_grad_(True)

    def freeze_backbone_eval() -> None:
        for name in ("initial_conv", "features", "final_conv", "final_norm"):
            module = getattr(model, name, None)
            if module is not None:
                module.eval()

    freeze_backbone_eval()
    return [p for p in model.parameters() if p.requires_grad], freeze_backbone_eval


def configure_last_blocks_adaptation(
    model: nn.Module,
    last_n_features: int,
    train_final_norm: bool = True,
) -> Tuple[List[nn.Parameter], Callable[[], None]]:
    """Train classifier plus the last N EfficientNet feature blocks.

    This is a conservative middle ground between classifier-only adaptation and
    full-model fine-tuning. BatchNorm running statistics are kept frozen by the
    returned eval callback; trainable convolution/norm affine parameters still
    receive gradients.
    """
    if last_n_features <= 0:
        raise ValueError(f"last_n_features must be positive, got {last_n_features}")
    if not hasattr(model, "features") or not hasattr(model.features, "__len__"):
        raise ValueError("last-block adaptation requires model.features sequence")
    for p in model.parameters():
        p.requires_grad_(False)

    n_features = len(model.features)
    start = max(0, n_features - int(last_n_features))
    for module in model.features[start:]:
        for p in module.parameters():
            p.requires_grad_(True)

    if hasattr(model, "final_conv"):
        for p in model.final_conv.parameters():
            p.requires_grad_(True)
    if train_final_norm and hasattr(model, "final_norm"):
        for p in model.final_norm.parameters():
            p.requires_grad_(True)
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad_(True)

    def freeze_backbone_eval() -> None:
        for name in ("initial_conv", "features", "final_conv", "final_norm"):
            module = getattr(model, name, None)
            if module is not None:
                module.eval()

    freeze_backbone_eval()
    return [p for p in model.parameters() if p.requires_grad], freeze_backbone_eval


class FoldableLowRankLinear(nn.Module):
    """A foldable low-rank residual adapter for a Linear layer.

    Forward uses `base(x) + alpha/rank * up(down(x))`. The base linear layer is
    frozen. `folded_weight_bias()` returns a normal Linear weight/bias pair, so
    saved checkpoints stay compatible with vanilla EfficientNet evaluation.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.down = nn.Linear(base.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, base.out_features, bias=False)
        self.down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.up.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))

    def folded_weight_bias(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = self.scale * (self.up.weight @ self.down.weight)
        return self.base.weight.detach() + delta.detach(), (
            None if self.base.bias is None else self.base.bias.detach()
        )


def _set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def attach_foldable_lora_classifier(model: nn.Module, rank: int, alpha: float) -> None:
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError("model has no classifier")
    linear_name = None
    linear_module = None
    for name, module in reversed(list(classifier.named_children())):
        if isinstance(module, FoldableLowRankLinear):
            return
        if isinstance(module, nn.Linear):
            linear_name = name
            linear_module = module
            break
    if linear_name is None or linear_module is None:
        raise ValueError("could not find final Linear inside model.classifier")
    _set_child_module(
        classifier,
        linear_name,
        FoldableLowRankLinear(linear_module, rank=rank, alpha=alpha),
    )


def compatible_state_dict_for_save(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state_dict compatible with vanilla EfficientNet1DV2.

    FoldableLowRankLinear modules are materialized into their corresponding
    `.weight` and `.bias` keys and their adapter internals are omitted.
    """
    state = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    folded_prefixes: Dict[str, FoldableLowRankLinear] = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, FoldableLowRankLinear)
    }
    for key, value in state.items():
        skip = False
        for prefix in folded_prefixes:
            if key.startswith(prefix + "."):
                skip = True
                break
        if not skip:
            out[key] = value
    for prefix, module in folded_prefixes.items():
        weight, bias = module.folded_weight_bias()
        out[f"{prefix}.weight"] = weight.detach().cpu()
        if bias is not None:
            out[f"{prefix}.bias"] = bias.detach().cpu()
    return out


def save_compatible_model_state(model: nn.Module, path: str) -> None:
    torch.save(compatible_state_dict_for_save(model), path)


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
# Quick eval (Super5 PN2021 multi-center stratified subset)
# ────────────────────────────────────────────────────────────────────────────

def build_quick_eval_subset_super5(
    centers: List[str],
    data_dir: str,
    n_per_center: int,
    cache_path: str,
    seed: int = 0,
    exclude_record_ids: Optional[set] = None,    # Issue #39
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Stratified-by-class-presence subsample of PN2021 records, super5 labels.

    `exclude_record_ids` lets us drop the ref pool's records from the same
    center's eval split (Issue #39 patient-level isolation).
    """
    cache_key = f"{cache_path}.super5.npz"
    if os.path.exists(cache_key):
        data = np.load(cache_key, allow_pickle=True)
        out = {}
        for c in centers:
            if f"{c}__signals" in data.files:
                out[c] = {
                    "signals_tc": data[f"{c}__signals"],
                    "labels_5":   data[f"{c}__labels5"],
                }
        if len(out) == len(centers):
            if verbose:
                print(f"[quick_eval] cache hit: {cache_key}")
            return out

    import wfdb
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for center in centers:
        if center.lower() in PN2021_FORBIDDEN:
            print(f"[quick_eval] SKIP forbidden shard: {center}")
            continue
        center_dir = os.path.join(data_dir, center)
        if not os.path.isdir(center_dir):
            continue
        t0 = time.time()
        hea_paths = []
        for root, _, files in os.walk(center_dir):
            for f in files:
                if f.endswith('.hea'):
                    hea_paths.append(os.path.join(root, f))
        if not hea_paths:
            continue

        # Parse SNOMED → super5 multi-hot for all
        labels_5 = []
        usable_idx = []
        for i, hea in enumerate(hea_paths):
            rec_id = os.path.basename(hea)[:-4]
            if exclude_record_ids and rec_id in exclude_record_ids:
                continue
            codes = parse_header_snomed(hea)
            labels_5.append(snomed_list_to_super5(codes))
            usable_idx.append(i)
        if not usable_idx:
            continue
        labels_5 = np.stack(labels_5).astype(np.float32)

        # Stratified pick
        if len(usable_idx) <= n_per_center:
            chosen_local = list(range(len(usable_idx)))
        else:
            per_class_quota = max(10, n_per_center // 10)
            picked = set()
            for cls_i in range(NUM_SUPER5):
                pos = np.where(labels_5[:, cls_i] == 1.0)[0]
                pos = np.array([p for p in pos if int(p) not in picked])
                if len(pos) == 0:
                    continue
                n_take = min(per_class_quota, len(pos))
                chosen = rng.choice(pos, size=n_take, replace=False)
                picked.update(int(c) for c in chosen)
            deficit = n_per_center - len(picked)
            if deficit > 0:
                rest = np.array([i for i in range(len(usable_idx)) if int(i) not in picked])
                if len(rest) > 0:
                    chosen = rng.choice(rest, size=min(deficit, len(rest)), replace=False)
                    picked.update(int(c) for c in chosen)
            chosen_local = sorted(picked)

        signals = []
        kept_labels = []
        for ci in chosen_local:
            i = usable_idx[ci]
            try:
                rec = wfdb.rdrecord(hea_paths[i][:-4])
            except Exception:
                continue
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                continue
            sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
            proc = unified_preprocess_to_1000(
                sig.astype(np.float32), fs=rec.fs, source_leads=sig_names,
                target_fs=100, target_len=1000,
                apply_filter=True, apply_zscore=True,
            )
            if proc is None:
                continue
            signals.append(proc)
            kept_labels.append(labels_5[ci])
        if not signals:
            continue
        out[center] = {
            "signals_tc": np.stack(signals).astype(np.float32),
            "labels_5":   np.stack(kept_labels).astype(np.float32),
        }
        if verbose:
            print(f"  [quick_eval] {center}: {out[center]['signals_tc'].shape[0]} records "
                  f"({time.time() - t0:.0f}s)"
                  + (f"; excluded {len(hea_paths) - len(usable_idx)} ref ids" if exclude_record_ids else ""))

    os.makedirs(os.path.dirname(cache_key) or ".", exist_ok=True)
    dump = {}
    for c, d in out.items():
        dump[f"{c}__signals"] = d["signals_tc"]
        dump[f"{c}__labels5"] = d["labels_5"]
    np.savez_compressed(cache_key, **dump)
    if verbose:
        print(f"[quick_eval] cached → {cache_key}")
    return out


@torch.no_grad()
def quick_eval_super5(
    model: nn.Module, quick_subset: Dict[str, Dict], device: str,
    crop_len: int = TIERM_INPUT_LENGTH, min_pos: int = 10,
) -> Dict[str, Any]:
    model.eval()
    result: Dict[str, Any] = {"per_center": {}}
    macro_aurocs, macro_auprcs = [], []
    for center, data in quick_subset.items():
        signals_tc = data["signals_tc"]   # (N, 1000, 12)
        labels_5 = data["labels_5"]       # (N, 5)
        N = signals_tc.shape[0]

        x_ct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)  # (N, 12, 1000)
        start = (x_ct.shape[-1] - crop_len) // 2
        x_ct = x_ct[..., start:start + crop_len]
        x_t = torch.from_numpy(x_ct).to(device)

        all_logits = []
        for i in range(0, N, 128):
            lg = model(x_t[i:i + 128])
            all_logits.append(lg.cpu().numpy())
        logits = np.concatenate(all_logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        m = compute_macro_auroc_auprc(labels_5, probs, CLASS_NAMES_SUPER5,
                                      min_pos=min_pos)
        result["per_center"][center] = {
            "n":           N,
            "macro_auroc": round(m["macro_auroc"], 4) if not math.isnan(m["macro_auroc"]) else None,
            "macro_auprc": round(m["macro_auprc"], 4) if not math.isnan(m["macro_auprc"]) else None,
            "n_classes_used": m["n_classes_used"],
            "per_class":   m["per_class"],
        }
        if not math.isnan(m["macro_auroc"]):
            macro_aurocs.append(m["macro_auroc"])
            macro_auprcs.append(m["macro_auprc"])

    result["avg_macro_auroc"] = round(float(np.mean(macro_aurocs)), 4) if macro_aurocs else float('nan')
    result["avg_macro_auprc"] = round(float(np.mean(macro_auprcs)), 4) if macro_auprcs else float('nan')
    return result


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


# ─────────────────────────────────────────────────────────────────────────
# Plan Rev 13 Stage 0.4: synth pool sanity → class_trust map
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_synth_sanity_auroc(
    synth_pool_signals: np.ndarray,        # (N, 12, 1000) z-scored
    labels_one_hot: np.ndarray,            # (N, 5)
    victim: nn.Module,
    device: str,
    crop_len: int = TIERM_INPUT_LENGTH,
    batch_size: int = 128,
) -> Dict[str, Optional[float]]:
    """Forward synth pool through Super5 victim, compute per-class AUROC.

    A class's synth is "trustworthy" iff victim AUROC > 0.55 (separable from
    the other 4 classes' decision regions).
    """
    N = synth_pool_signals.shape[0]
    crops = []
    for i in range(N):
        sig_tc = synth_pool_signals[i].T  # (1000, 12)
        start = (sig_tc.shape[0] - crop_len) // 2
        crop = sig_tc[start:start + crop_len, :]
        crops.append(crop.T.astype(np.float32))
    crops_arr = np.stack(crops, axis=0)
    victim.eval() if hasattr(victim, 'eval') else None
    all_logits = []
    for i in range(0, N, batch_size):
        x = torch.from_numpy(crops_arr[i:i + batch_size]).float().to(device)
        if hasattr(victim, 'compute_logits_from_ecg'):
            lg = victim.compute_logits_from_ecg(x)
        else:
            lg = victim(x)
        all_logits.append(lg.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    per_class: Dict[str, Optional[float]] = {}
    for j, c in enumerate(CLASS_NAMES_SUPER5):
        col = labels_one_hot[:, j]
        n_pos = int((col == 1.0).sum())
        if n_pos == 0 or n_pos == N:
            per_class[c] = None
            continue
        try:
            per_class[c] = float(roc_auc_score(col, probs[:, j]))
        except Exception:
            per_class[c] = None
    return per_class


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
        anc_chunks.append(anc_x.detach().cpu().numpy().astype(np.float32))
        delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())

    adv_signals = np.concatenate(adv_chunks, axis=0)                  # (K, 12, 1000)
    anc_signals = np.concatenate(anc_chunks, axis=0)
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


def build_latent_augmix_branch_signals(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    severity_profile: str = "standard",
    width: int,
    depth: int,
    alpha: float,
    latent_weight_cap: float,
    ops: List[str],
    rng: np.random.Generator,
    renorm: bool = True,
    clip_abs: float = 6.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Legacy wrapper around the package-level latent AugMix core."""

    def _apply_augmix_op_np(
        sig_ct: np.ndarray,
        op_name: str,
        op_severity: int,
        op_severity_profile: str,
    ) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        if op_severity_profile == "standard":
            return _apply_op(sig_t, op_name, int(op_severity)).cpu().numpy().astype(np.float32, copy=False)
        op = _build_pn2021c_corruption_op(op_name, int(op_severity), op_severity_profile)
        return op(sig_t).cpu().numpy().astype(np.float32, copy=False)

    return _build_latent_augmix_branch_signals_core(
        anchor_signals_ct,
        adv_signals_ct,
        copies=copies,
        severity=severity,
        severity_profile=severity_profile,
        width=width,
        depth=depth,
        alpha=alpha,
        latent_weight_cap=latent_weight_cap,
        ops=ops,
        rng=rng,
        op_apply_fn=_apply_augmix_op_np,
        available_ops=AVAILABLE_OPS,
        renorm=renorm,
        clip_abs=clip_abs,
    )


def build_raw_corruption_views(
    signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    severity_profile: str = "standard",
    ops: List[str],
    prob: float,
    rng: np.random.Generator,
    renorm: bool = False,
    clip_abs: float = 6.0,
    input_stabilizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Legacy wrapper around the package-level raw corruption view core."""

    def _apply_augmix_op_np(sig_ct: np.ndarray, op_name: str, op_severity: int) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        if severity_profile == "standard":
            return _apply_op(sig_t, op_name, int(op_severity)).cpu().numpy().astype(np.float32, copy=False)
        op = _build_pn2021c_corruption_op(op_name, int(op_severity), severity_profile)
        return op(sig_t).cpu().numpy().astype(np.float32, copy=False)

    postprocess_fn, postprocess_name = _raw_input_stabilizer_postprocess(input_stabilizer_config)
    arr, stats = _build_raw_corruption_views_core(
        signals_ct,
        copies=copies,
        severity=severity,
        ops=ops,
        prob=prob,
        rng=rng,
        op_apply_fn=_apply_augmix_op_np,
        available_ops=AVAILABLE_OPS,
        renorm=renorm,
        clip_abs=clip_abs,
        postprocess_fn=postprocess_fn,
        postprocess_name=postprocess_name,
    )
    stats["severity_profile"] = str(severity_profile)
    if input_stabilizer_config:
        stats["input_stabilizer"] = dict(input_stabilizer_config)
    return arr, stats


def build_raw_augmix_views(
    signals_ct: np.ndarray,
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
    ops: List[str],
    rng: np.random.Generator,
    renorm: bool = False,
    clip_abs: float = 6.0,
    input_stabilizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Legacy wrapper around the package-level full-pool raw AugMix view core."""

    def _apply_augmix_op_np(
        sig_ct: np.ndarray,
        op_name: str,
        op_severity: int,
        op_severity_profile: str,
    ) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        if op_severity_profile == "standard":
            return _apply_op(sig_t, op_name, int(op_severity)).cpu().numpy().astype(np.float32, copy=False)
        op = _build_pn2021c_corruption_op(op_name, int(op_severity), op_severity_profile)
        return op(sig_t).cpu().numpy().astype(np.float32, copy=False)

    postprocess_fn, postprocess_name = _raw_input_stabilizer_postprocess(input_stabilizer_config)
    arr, stats = _build_raw_augmix_views_core(
        signals_ct,
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
        ops=ops,
        rng=rng,
        op_apply_fn=_apply_augmix_op_np,
        available_ops=AVAILABLE_OPS,
        renorm=renorm,
        clip_abs=clip_abs,
        postprocess_fn=postprocess_fn,
        postprocess_name=postprocess_name,
    )
    if input_stabilizer_config:
        stats["input_stabilizer"] = dict(input_stabilizer_config)
    return arr, stats


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
        sig_250 = _center_crop_ct(adv_signals_ct[i], crop_len)         # (12, 250)
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

    # PN2021 quick eval
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--quick_eval_centers", nargs='+',
                   default=["chapman_shaoxing", "cpsc_2018_extra", "georgia",
                            "ningbo"])
    p.add_argument("--quick_eval_n_per_center", type=int, default=1000)
    p.add_argument(
        "--quick_eval_source",
        choices=["pn2021", "target_real_val"],
        default="pn2021",
        help=(
            "pn2021 uses the historical ref-excluded PN2021 quick subset. "
            "target_real_val selects checkpoints on a validation split held "
            "out from --target_real_npz, avoiding target-center test leakage."
        ),
    )
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)

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
    p.add_argument("--allow_hyp_cd_trust", action="store_true",
                   help="Do not hard-force HYP/CD class_trust to zero.")
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
    p.add_argument("--roundtrip_weight", type=float, default=0.5)
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
        "--disable_adv_stream",
        action="store_true",
        help=(
            "Skip latent PGD generation and do not add adversarial samples to "
            "the training stream. Use this as the direct target-real adaptation "
            "control under the same data/optimizer protocol."
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
        choices=PN2021C_STRESS_PROFILE_CHOICES,
        default="standard",
        help=(
            "Parameter profile used by non-latent latent-AugMix ECG op chains. "
            "'standard' preserves the original mild AugMix table; "
            "'calibrated_10to20pp' matches the strong PN2021-C evaluation profile."
        ),
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
    p.add_argument(
        "--enable_raw_corrupt_consistency",
        action="store_true",
        help=(
            "After the normal mixed epoch, train a small target-real corruption "
            "consistency phase using raw ECG AugMix-style ops. This directly "
            "matches PN2021-C corruption types and is disabled by default."
        ),
    )
    p.add_argument("--raw_corrupt_copies", type=int, default=1)
    p.add_argument("--raw_corrupt_prob", type=float, default=0.5)
    p.add_argument("--raw_corrupt_severity", type=int, default=4)
    p.add_argument(
        "--raw_corrupt_severity_profile",
        choices=PN2021C_STRESS_PROFILE_CHOICES,
        default="standard",
        help=(
            "Parameter profile used by the raw ECG corruption consistency branch. "
            "'standard' preserves the training-time AugMix severity table; "
            "'calibrated_10to20pp' matches the strong PN2021-C evaluation profile."
        ),
    )
    p.add_argument(
        "--raw_corrupt_ops",
        nargs="+",
        default=[
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
        choices=AVAILABLE_OPS,
        help="Raw ECG corruption ops for the consistency branch.",
    )
    p.add_argument("--raw_corrupt_consistency_weight", type=float, default=0.5)
    p.add_argument(
        "--raw_corrupt_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="soft_bce",
        help=(
            "Consistency objective for raw ECG corruptions. 'soft_bce' preserves "
            "the previous clean-teacher BCE behavior; 'jsd' uses multi-label "
            "AugMix-style JSD over clean and two corrupted views."
        ),
    )
    p.add_argument("--raw_corrupt_bce_weight", type=float, default=0.1)
    p.add_argument("--raw_corrupt_max_batches", type=int, default=0)
    p.add_argument(
        "--raw_corrupt_scope",
        choices=["target", "source", "source_target"],
        default="target",
        help=(
            "Dataset used by the raw ECG corruption consistency phase. "
            "'target' preserves the original K-shot-only behavior; 'source' "
            "uses PTB-XL train samples; 'source_target' concatenates both."
        ),
    )
    p.add_argument("--raw_corrupt_no_renorm", action="store_true")
    p.add_argument("--raw_corrupt_clip_abs", type=float, default=6.0)
    p.add_argument(
        "--raw_corrupt_view_mode",
        choices=["single_op", "augmix"],
        default="single_op",
        help="Use single-op raw corruption views or full AugMix multi-chain raw views.",
    )
    p.add_argument("--raw_augmix_width", type=int, default=3)
    p.add_argument("--raw_augmix_depth", type=int, default=-1)
    p.add_argument("--raw_augmix_alpha", type=float, default=1.0)
    p.add_argument("--raw_augmix_mixture_mode", choices=["beta", "fixed"], default="beta")
    p.add_argument("--raw_augmix_mixture_prob", type=float, default=0.5)
    p.add_argument("--raw_augmix_mixture_beta_a", type=float, default=0.0)
    p.add_argument("--raw_augmix_mixture_beta_b", type=float, default=0.0)
    p.add_argument("--raw_input_bandpass_low_hz", type=float, default=None)
    p.add_argument("--raw_input_bandpass_high_hz", type=float, default=None)
    p.add_argument("--raw_input_repair_flat_leads", action="store_true")
    p.add_argument("--raw_input_clip_abs", type=float, default=None)
    p.add_argument("--raw_input_renorm_after_stabilizer", action="store_true")
    p.add_argument("--raw_input_sample_rate_hz", type=float, default=100.0)
    p.add_argument(
        "--enable_mask_shift_consistency",
        action="store_true",
        help=(
            "After the normal mixed epoch, train deterministic target-real "
            "views for random_leads_masking and baseline_shift. This is a "
            "targeted lead-invariance branch for PN2021-C masking/shift "
            "stressors, separate from random raw-corruption sampling."
        ),
    )
    p.add_argument("--mask_shift_copies", type=int, default=1)
    p.add_argument("--mask_shift_mask_severity", type=int, default=6)
    p.add_argument("--mask_shift_shift_severity", type=int, default=6)
    p.add_argument("--mask_shift_consistency_weight", type=float, default=1.0)
    p.add_argument(
        "--mask_shift_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    p.add_argument("--mask_shift_bce_weight", type=float, default=0.05)
    p.add_argument("--mask_shift_max_batches", type=int, default=0)
    p.add_argument(
        "--mask_shift_scope",
        choices=["target", "source", "source_target"],
        default="target",
    )
    p.add_argument("--mask_shift_no_renorm", action="store_true")
    p.add_argument("--mask_shift_clip_abs", type=float, default=6.0)
    p.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    p.add_argument("--qab_size", type=int, default=2048)
    p.add_argument(
        "--source_logit_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "Optional source-consistency distillation weight. After each mixed "
            "training epoch, run a PTB-XL source pass and penalize MSE between "
            "current logits and the frozen initial source model logits."
        ),
    )
    p.add_argument(
        "--source_logit_anchor_batches",
        type=int,
        default=0,
        help="Max PTB-XL source batches per source-logit anchor pass; 0 uses the full source loader.",
    )
    p.add_argument(
        "--freeze_backbone_classifier_only",
        action="store_true",
        help=(
            "Freeze the EfficientNet backbone and train only model.classifier. "
            "Backbone BatchNorm modules are forced to eval during adaptation. "
            "This keeps the checkpoint compatible with the normal model while "
            "testing whether VAE latent-hull samples add value beyond head fitting."
        ),
    )
    p.add_argument(
        "--classifier_only_train_final_norm",
        action="store_true",
        help=(
            "With --freeze_backbone_classifier_only, also train final_norm affine "
            "parameters while keeping its BatchNorm running statistics frozen. "
            "This gives a small domain-calibration adapter without full backbone FT."
        ),
    )
    p.add_argument(
        "--classifier_adapter_type",
        choices=["linear", "lora"],
        default="linear",
        help=(
            "Adapter used with --freeze_backbone_classifier_only. linear trains "
            "the existing classifier; lora trains a foldable low-rank residual "
            "on the final Linear and saves a vanilla-compatible checkpoint."
        ),
    )
    p.add_argument("--classifier_lora_rank", type=int, default=16)
    p.add_argument("--classifier_lora_alpha", type=float, default=16.0)
    p.add_argument(
        "--unfreeze_last_n_features",
        type=int,
        default=0,
        help=(
            "Conservative EfficientNet adaptation: train classifier, "
            "final_conv/final_norm, and the last N feature blocks while "
            "keeping BatchNorm running statistics frozen. Mutually exclusive "
            "with --freeze_backbone_classifier_only."
        ),
    )

    # Class trust (Plan Rev 13 H4 gate)
    p.add_argument("--class_trust", default=None,
                   help="Path to class_trust.json (Stage 0.4 sanity output). Required unless --build_class_trust.")
    p.add_argument("--build_class_trust", action="store_true",
                   help="Run sanity AUROC on synth pool, write class_trust.json next to synth pool, then EXIT.")

    # Optim (Plan Rev 13.1: 100 ep + early-stop patience=20 on val_macro_auroc)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stop after this many quick_eval rounds without improvement")
    p.add_argument("--es_metric",
                   choices=[
                       "val_macro_auroc",
                       "val_macro_auprc",
                       "target_macro_auroc",
                       "target_macro_auprc",
                   ],
                   default="val_macro_auroc")
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

    # ── Plan Rev 13 Stage 0.4 sanity-build mode (early exit) ────────────────
    if args.build_class_trust:
        # Need synth_pool's signals (decoded) — load from .npz (signals key) or
        # re-derive from latents via VAE decoder.
        signal_npz = args.synth_npz
        if signal_npz.endswith(".latent.npz"):
            signal_npz = signal_npz.replace(".latent.npz", ".npz")
        if not os.path.exists(signal_npz):
            raise SystemExit(f"signal pool .npz not found at {signal_npz}; "
                              "regen synth with --save_latent (this stores both signals + latents).")
        sig_data = np.load(signal_npz, allow_pickle=True)
        if "signals" not in sig_data.files:
            raise SystemExit(f"{signal_npz} missing 'signals' key — re-run "
                              "generate_center_synth.py with the current head.")
        signals = sig_data["signals"].astype(np.float32)
        labels_oh = sig_data["labels"].astype(np.float32)
        print(f"[sanity] forwarding {signals.shape} synth signals through "
              f"victim for per-class AUROC ...")
        per_class = compute_synth_sanity_auroc(
            signals, labels_oh, victim, args.device,
            crop_len=args.crop_len, batch_size=128,
        )
        trust = derive_class_trust(per_class)
        ct_path = args.class_trust or str(
            Path(args.synth_npz).with_suffix("").with_suffix(".class_trust.json"))
        os.makedirs(os.path.dirname(ct_path) or ".", exist_ok=True)
        with open(ct_path, "w") as f:
            json.dump({
                "tag": args.center_name,
                "synth_pool": args.synth_npz,
                "per_class_auroc": per_class,
                "class_trust": trust,
                "policy": "AUROC>0.7→1.0; >0.55→0.5; else 0.0; HYP/CD hardcoded 0.0",
            }, f, indent=2)
        print(f"[sanity] per-class AUROC: {per_class}")
        print(f"[sanity] class_trust: {trust}")
        print(f"[sanity] wrote → {ct_path}")
        print("[sanity] DONE — exiting (build_class_trust mode)")
        return

    # ── Load class_trust (required for training) ───────────────────────────
    if not args.class_trust or not os.path.exists(args.class_trust):
        raise SystemExit(f"--class_trust required for training (got {args.class_trust!r}). "
                          f"Run with --build_class_trust first to derive it.")
    with open(args.class_trust) as f:
        trust_blob = json.load(f)
    class_trust: Dict[str, float] = dict(trust_blob["class_trust"])
    if not args.allow_hyp_cd_trust:
        class_trust.update(DEFAULT_TRUST_HARDCODE)   # Plan Rev 11 hard-enforce
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
    if args.enable_latent_augmix_branch:
        print(
            "[setup] latent-branch AugMix enabled: "
            f"copies={args.latent_augmix_copies} width={args.latent_augmix_width} "
            f"depth={args.latent_augmix_depth} severity={args.latent_augmix_severity} "
            f"profile={args.latent_augmix_severity_profile} "
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
    from scripts.triple_labels.label_schemes import get_scheme
    scheme = get_scheme("super5")

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
        args.roundtrip_anchor_n = 64
        print("[smoke] truncated PTBXL train/val + roundtrip_anchor_n=64")

    train_ds = PTBXLDatasetScheme(train_signals, train_labels,
                                  crop_len=args.crop_len, mode='train')
    val_ds = PTBXLDatasetScheme(val_signals, val_labels,
                                crop_len=args.crop_len, mode='eval')
    source_logit_anchor_loader = None
    if args.source_logit_anchor_weight > 0:
        source_logit_anchor_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        print(
            f"[setup] source-logit anchor enabled: weight={args.source_logit_anchor_weight} "
            f"max_batches={args.source_logit_anchor_batches or 'full'}",
            flush=True,
        )
    target_real_ds = None
    target_val_quick_subset = None
    target_val_record_ids: set[str] = set()
    if args.target_real_npz:
        with np.load(args.target_real_npz, allow_pickle=True) as real_data:
            real_signals = np.asarray(real_data["signals"], dtype=np.float32)
            real_labels = np.asarray(real_data["labels"], dtype=np.float32)
            real_record_ids = (
                real_data["record_ids"].astype(str)
                if "record_ids" in real_data.files
                else np.asarray([str(i) for i in range(real_labels.shape[0])])
            )
        if real_signals.ndim != 3:
            raise ValueError(f"target_real_npz signals must be 3D, got {real_signals.shape}")
        if real_signals.shape[1:] == (12, 1000):
            real_signals = real_signals.transpose(0, 2, 1)
        if real_signals.shape[1:] != (1000, 12):
            raise ValueError(f"target_real_npz signals must be (N,1000,12) or (N,12,1000), got {real_signals.shape}")
        if real_labels.shape[0] != real_signals.shape[0] or real_labels.shape[1] != NUM_SUPER5:
            raise ValueError(f"target_real_npz labels mismatch: signals={real_signals.shape} labels={real_labels.shape}")
        if args.quick_eval_source == "target_real_val":
            val_mask = build_k500_internal_val_mask(
                real_labels,
                val_fraction=args.target_real_val_fraction,
                seed=args.target_real_val_seed,
            )
            train_mask = ~val_mask
            target_val_record_ids = set(str(x) for x in real_record_ids[val_mask])
            target_val_quick_subset = {
                args.center_name: {
                    "signals_tc": real_signals[val_mask].astype(np.float32),
                    "labels_5": real_labels[val_mask].astype(np.float32),
                }
            }
            print(
                f"[setup] target-real internal val split: "
                f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
                f"fraction={args.target_real_val_fraction} seed={args.target_real_val_seed}",
                flush=True,
            )
            real_signals = real_signals[train_mask]
            real_labels = real_labels[train_mask]
        target_real_ds = PTBXLDatasetScheme(
            real_signals,
            real_labels,
            crop_len=args.crop_len,
            mode='train',
        )
        print(
            f"[setup] target-real supervised stream: n={len(target_real_ds)} "
            f"weight={args.target_real_weight} path={args.target_real_npz}",
            flush=True,
        )
    elif args.quick_eval_source == "target_real_val":
        raise ValueError("--quick_eval_source target_real_val requires --target_real_npz")

    raw_corrupt_consistency_loader = None
    raw_input_stabilizer_config = _raw_input_stabilizer_config(args)
    if args.enable_raw_corrupt_consistency:
        if args.raw_corrupt_scope == "target":
            if target_real_ds is None:
                raise ValueError("--enable_raw_corrupt_consistency with --raw_corrupt_scope target requires --target_real_npz")
            raw_corrupt_ds = target_real_ds
        elif args.raw_corrupt_scope == "source":
            raw_corrupt_ds = train_ds
        else:
            if target_real_ds is None:
                raise ValueError("--enable_raw_corrupt_consistency with --raw_corrupt_scope source_target requires --target_real_npz")
            raw_corrupt_ds = ConcatDataset([train_ds, target_real_ds])
        raw_corrupt_consistency_loader = DataLoader(
            raw_corrupt_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        print(
            "[setup] raw corruption consistency enabled: "
            f"copies={args.raw_corrupt_copies} severity={args.raw_corrupt_severity} "
            f"profile={args.raw_corrupt_severity_profile} "
            f"prob={args.raw_corrupt_prob} "
            f"weights=(consistency={args.raw_corrupt_consistency_weight}, "
            f"bce={args.raw_corrupt_bce_weight}) "
            f"scope={args.raw_corrupt_scope} n={len(raw_corrupt_ds)} "
            f"view_mode={args.raw_corrupt_view_mode} "
            f"augmix=(width={args.raw_augmix_width},depth={args.raw_augmix_depth},"
            f"alpha={args.raw_augmix_alpha},mixture={args.raw_augmix_mixture_mode}:"
            f"{args.raw_augmix_mixture_prob},beta=({args.raw_augmix_mixture_beta_a},"
            f"{args.raw_augmix_mixture_beta_b})) "
            f"input_stabilizer={raw_input_stabilizer_config if _has_raw_input_stabilizer(raw_input_stabilizer_config) else 'disabled'} "
            f"ops={args.raw_corrupt_ops} "
            f"max_batches={args.raw_corrupt_max_batches or 'full'}",
            flush=True,
        )

    mask_shift_consistency_loader = None
    if args.enable_mask_shift_consistency:
        if args.mask_shift_scope == "target":
            if target_real_ds is None:
                raise ValueError("--enable_mask_shift_consistency with --mask_shift_scope target requires --target_real_npz")
            mask_shift_ds = target_real_ds
        elif args.mask_shift_scope == "source":
            mask_shift_ds = train_ds
        else:
            if target_real_ds is None:
                raise ValueError("--enable_mask_shift_consistency with --mask_shift_scope source_target requires --target_real_npz")
            mask_shift_ds = ConcatDataset([train_ds, target_real_ds])
        mask_shift_consistency_loader = DataLoader(
            mask_shift_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        print(
            "[setup] mask/shift consistency enabled: "
            f"copies_per_op={args.mask_shift_copies} "
            f"mask_severity={args.mask_shift_mask_severity} "
            f"shift_severity={args.mask_shift_shift_severity} "
            f"weights=(consistency={args.mask_shift_consistency_weight}, "
            f"bce={args.mask_shift_bce_weight}) "
            f"loss={args.mask_shift_consistency_loss} "
            f"scope={args.mask_shift_scope} n={len(mask_shift_ds)} "
            f"max_batches={args.mask_shift_max_batches or 'full'}",
            flush=True,
        )

    if target_val_record_ids and "record_ids" in source_meta:
        keep_mask = np.asarray(
            [str(rid) not in target_val_record_ids for rid in source_meta["record_ids"]],
            dtype=bool,
        )
        n_drop = int((~keep_mask).sum())
        if n_drop > 0:
            synth_latents = synth_latents[keep_mask]
            synth_labels = synth_labels[keep_mask]
            for key in ("source_ids", "source_labels", "record_ids"):
                if key in source_meta:
                    source_meta[key] = source_meta[key][keep_mask]
            print(
                f"[setup] removed {n_drop} K500-val records from latent anchor pool; "
                f"train_latents={len(synth_latents)}",
                flush=True,
            )

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5),
        dtype=torch.float32, device=args.device)
    print(f"[loss] pos_weight: {pos_weight.cpu().tolist()}")

    # reduction='none' for mask × bce (-1 sentinel handling)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Roundtrip anchor (cached) ────────────────────────────────────────────
    rt_cache = os.path.join(args.output_dir, f"roundtrip_anchor_n{args.roundtrip_anchor_n}.npz")
    roundtrip_ds: Optional[TensorDataset] = build_roundtrip_anchor_dataset(
        train_ds=train_ds, ecgtwin=ecgtwin, n_samples=args.roundtrip_anchor_n,
        device=args.device, crop_len=args.crop_len,
        seed=args.seed, cache_path=rt_cache,
    )
    if roundtrip_ds is not None:
        print(f"[setup] roundtrip-anchor: n={len(roundtrip_ds)}, weight={args.roundtrip_weight}")

    # ── Quick eval subset ───────────────────────────────────────────────────
    target_val_n = (
        int(target_val_quick_subset[args.center_name]["signals_tc"].shape[0])
        if target_val_quick_subset is not None
        else 0
    )
    quick_eval_plan = resolve_quick_eval_plan(
        quick_eval_source=args.quick_eval_source,
        center_name=args.center_name,
        quick_eval_centers=args.quick_eval_centers,
        output_dir=args.output_dir,
        quick_eval_n_per_center=args.quick_eval_n_per_center,
        target_val_available=target_val_quick_subset is not None,
        target_val_n=target_val_n,
    )
    if quick_eval_plan.source == "target_real_val":
        assert target_val_quick_subset is not None
        quick_subset = target_val_quick_subset
        print(f"[setup] {quick_eval_plan.message}", flush=True)
    else:
        # Historical path: ref-excluded PN2021 target-center quick subset.
        # This is useful for exploration, but final paper-safe model selection
        # should use --quick_eval_source target_real_val.
        excluded = None
        if args.ref_meta_json and os.path.exists(args.ref_meta_json):
            with open(args.ref_meta_json) as f:
                meta = json.load(f)
            excluded = set(meta.get("ref_record_ids", []))
            print(f"[setup] excluding {len(excluded)} ref_record_ids from "
                  f"quick_eval (center={args.center_name})")

        assert quick_eval_plan.cache_path is not None
        qe_cache = str(quick_eval_plan.cache_path)
        quick_subset = build_quick_eval_subset_super5(
            centers=list(quick_eval_plan.centers), data_dir=args.data_dir,
            n_per_center=args.quick_eval_n_per_center, cache_path=qe_cache,
            seed=args.seed, exclude_record_ids=excluded,
        )

    print("[baseline] Computing baseline quick-eval ...")
    baseline_qe = quick_eval_super5(victim.model, quick_subset, args.device,
                                    crop_len=args.crop_len)
    print(f"[baseline] avg macro AUROC={baseline_qe['avg_macro_auroc']}, "
          f"AUPRC={baseline_qe['avg_macro_auprc']}")
    for c, info in baseline_qe["per_center"].items():
        print(f"    {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}  "
              f"(n={info['n']}  classes_used={info['n_classes_used']})")

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
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None
    if args.freeze_backbone_classifier_only and args.unfreeze_last_n_features > 0:
        raise ValueError(
            "--freeze_backbone_classifier_only and --unfreeze_last_n_features "
            "are mutually exclusive"
        )
    if args.freeze_backbone_classifier_only:
        trainable_params, freeze_backbone_eval_fn = configure_classifier_only_adaptation(
            victim.model,
            train_final_norm=args.classifier_only_train_final_norm,
            adapter_type=args.classifier_adapter_type,
            lora_rank=args.classifier_lora_rank,
            lora_alpha=args.classifier_lora_alpha,
        )
        print(
            "[setup] classifier-only adaptation enabled: "
            f"{sum(p.numel() for p in trainable_params):,} trainable params "
            f"(train_final_norm={args.classifier_only_train_final_norm}, "
            f"adapter={args.classifier_adapter_type})",
            flush=True,
        )
    elif args.unfreeze_last_n_features > 0:
        trainable_params, freeze_backbone_eval_fn = configure_last_blocks_adaptation(
            victim.model,
            last_n_features=args.unfreeze_last_n_features,
            train_final_norm=True,
        )
        print(
            "[setup] last-block adaptation enabled: "
            f"{sum(p.numel() for p in trainable_params):,} trainable params "
            f"(last_n_features={args.unfreeze_last_n_features}, "
            "final_conv=True, final_norm=True, classifier=True)",
            flush=True,
        )
    else:
        for p in victim.model.parameters():
            p.requires_grad_(True)
        trainable_params = [p for p in victim.model.parameters() if p.requires_grad]
    source_logit_teacher_model = None
    if args.source_logit_anchor_weight > 0:
        source_logit_teacher_model = copy.deepcopy(victim.model).to(args.device)
        source_logit_teacher_model.eval()
        for p in source_logit_teacher_model.parameters():
            p.requires_grad_(False)
        print("[setup] frozen source-logit teacher enabled")
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
        "baseline_quick_eval": baseline_qe,
        "class_trust": class_trust,
        "adaptation": {
            "freeze_backbone_classifier_only": bool(args.freeze_backbone_classifier_only),
            "classifier_only_train_final_norm": bool(args.classifier_only_train_final_norm),
            "classifier_adapter_type": args.classifier_adapter_type,
            "classifier_lora_rank": int(args.classifier_lora_rank),
            "classifier_lora_alpha": float(args.classifier_lora_alpha),
            "unfreeze_last_n_features": int(args.unfreeze_last_n_features),
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
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "severity": int(args.latent_augmix_severity),
            "severity_profile": str(args.latent_augmix_severity_profile),
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
        "raw_corrupt_consistency": {
            "enabled": bool(args.enable_raw_corrupt_consistency),
            "copies": int(args.raw_corrupt_copies),
            "prob": float(args.raw_corrupt_prob),
            "severity": int(args.raw_corrupt_severity),
            "severity_profile": str(args.raw_corrupt_severity_profile),
            "ops": list(args.raw_corrupt_ops),
            "consistency_weight": float(args.raw_corrupt_consistency_weight),
            "consistency_loss": str(args.raw_corrupt_consistency_loss),
            "bce_weight": float(args.raw_corrupt_bce_weight),
            "max_batches": int(args.raw_corrupt_max_batches),
            "scope": str(args.raw_corrupt_scope),
            "renorm": not bool(args.raw_corrupt_no_renorm),
            "clip_abs": float(args.raw_corrupt_clip_abs),
            "view_mode": str(args.raw_corrupt_view_mode),
            "augmix_width": int(args.raw_augmix_width),
            "augmix_depth": int(args.raw_augmix_depth),
            "augmix_alpha": float(args.raw_augmix_alpha),
            "augmix_mixture_mode": str(args.raw_augmix_mixture_mode),
            "augmix_mixture_prob": float(args.raw_augmix_mixture_prob),
            "augmix_mixture_beta_a": float(args.raw_augmix_mixture_beta_a),
            "augmix_mixture_beta_b": float(args.raw_augmix_mixture_beta_b),
            "input_stabilizer": dict(raw_input_stabilizer_config),
        },
        "mask_shift_consistency": {
            "enabled": bool(args.enable_mask_shift_consistency),
            "copies_per_op": int(args.mask_shift_copies),
            "mask_severity": int(args.mask_shift_mask_severity),
            "shift_severity": int(args.mask_shift_shift_severity),
            "consistency_weight": float(args.mask_shift_consistency_weight),
            "consistency_loss": str(args.mask_shift_consistency_loss),
            "bce_weight": float(args.mask_shift_bce_weight),
            "max_batches": int(args.mask_shift_max_batches),
            "scope": str(args.mask_shift_scope),
            "renorm": not bool(args.mask_shift_no_renorm),
            "clip_abs": float(args.mask_shift_clip_abs),
        },
        "epochs": [],
    }
    def selected_es_metric(qe: Dict[str, Any]) -> float:
        if args.es_metric == "val_macro_auroc":
            return float(qe.get("avg_macro_auroc", float("nan")))
        if args.es_metric == "val_macro_auprc":
            return float(qe.get("avg_macro_auprc", float("nan")))
        center_info = qe.get("per_center", {}).get(args.center_name, {})
        if args.es_metric == "target_macro_auroc":
            return float(center_info.get("macro_auroc", float("nan")))
        if args.es_metric == "target_macro_auprc":
            return float(center_info.get("macro_auprc", float("nan")))
        raise ValueError(f"unsupported es_metric={args.es_metric}")

    best_metric = selected_es_metric(baseline_qe)
    if best_metric != best_metric:
        best_metric = -1.0  # NaN-safe
    best_epoch = 0
    epochs_since_best = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    log_path = os.path.join(args.output_dir, "training_log.json")
    es_path = os.path.join(args.output_dir, "early_stop_info.json")
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_latest_path = checkpoint_dir / "checkpoint_latest.pt"
    checkpoint_best_path = checkpoint_dir / "checkpoint_best.pt"
    checkpoint_index_path = checkpoint_dir / "checkpoint_index.jsonl"
    diagnostics_epoch_path = Path(args.output_dir) / "diagnostics_epoch.jsonl"
    agent_decision_path = Path(args.output_dir) / "agent_decision.json"
    resume_path = _resolve_resume_path(args.resume, args.output_dir)
    if should_save_initial_best_model(resume_path):
        save_compatible_model_state(victim.model, best_ckpt_path)
    elif not Path(best_ckpt_path).exists():
        print(
            f"[resume-warning] best_model.pt is missing before resume: {best_ckpt_path}. "
            "It will not be recreated unless a later epoch improves.",
            flush=True,
        )

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
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        epochs_since_best = int(ckpt.get("epochs_since_best", epochs_since_best))
        consecutive_low_asr = int(ckpt.get("consecutive_low_asr", consecutive_low_asr))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"[resume] start_epoch={start_epoch} best_epoch={best_epoch} "
            f"best_metric={best_metric} buffer={len(buffer)}",
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
        if args.disable_adv_stream:
            asr_info = {
                "asr_overall": float("nan"),
                "per_class_asr": {},
                "multilabel_positive_label_asr": float("nan"),
                "sample_any_positive_below_0p5_asr": float("nan"),
                "sample_all_positive_below_0p5_asr": float("nan"),
                "sample_all_positive_recognized_rate": float("nan"),
                "per_class_positive_label_asr": {},
            }
            sem_info = {"PASS": True}
            push_stats = {"n_pushed": 0, "label_mode": "adv_stream_disabled"}
            latent_augmix_stats = {"enabled": False, "reason": "adv_stream_disabled", "n_generated": 0}
            latent_augmix_push_stats = {}
            delta_stats = {"mean_delta_norm": float("nan"), "max_delta_norm": float("nan")}
            decode_invalid_stats = {
                "decoded_invalid_rate": float("nan"),
                "nan_rate": float("nan"),
                "flatline_rate": float("nan"),
            }
            attack_vs_anchor_stats: Dict[str, Any] = {}
            gate_skipped = True
        else:
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
                    latent_augmix_signals, latent_augmix_stats = build_latent_augmix_branch_signals(
                        anchor_signals_ct=anc_signals,
                        adv_signals_ct=adv_signals,
                        copies=args.latent_augmix_copies,
                        severity=args.latent_augmix_severity,
                        severity_profile=args.latent_augmix_severity_profile,
                        width=args.latent_augmix_width,
                        depth=args.latent_augmix_depth,
                        alpha=args.latent_augmix_alpha,
                        latent_weight_cap=args.latent_augmix_latent_weight_cap,
                        ops=list(args.latent_augmix_ops),
                        rng=rng,
                        renorm=not args.no_latent_augmix_renorm,
                        clip_abs=args.latent_augmix_clip_abs,
                    )
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
                            latent_augmix_direct_clean = anc_signals.astype(np.float32, copy=False)
                            latent_augmix_direct_views = latent_augmix_signals.astype(np.float32, copy=False)
                            latent_augmix_direct_labels = target_oh.astype(np.float32, copy=False)
                    print(
                        f"[ep{epoch:02d}] latent-branch AugMix: "
                        f"generated={latent_augmix_stats.get('n_generated', 0)} "
                        f"pushed={latent_augmix_push_stats.get('n_pushed', 0)} "
                        f"w_lat_mean={latent_augmix_stats.get('latent_weight_mean', float('nan')):.3f} "
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
        if roundtrip_ds is not None and args.roundtrip_weight > 0:
            streams.append((roundtrip_ds, args.roundtrip_weight, None))
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
                "target_real_weight, roundtrip_weight, and adv buffer gates."
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
        if freeze_backbone_eval_fn is not None:
            train_loss = train_one_epoch_masked_bce_freeze_aware(
                victim.model, train_loader, optimizer, criterion, args.device,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                ewa_params=ewa_params,
                anchor_lambda=args.anchor_lambda,
                ewa_decay=args.ewa_decay,
                freeze_backbone_eval_fn=freeze_backbone_eval_fn,
            )
        else:
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
                    freeze_backbone_eval_fn=freeze_backbone_eval_fn,
                    max_batches=args.latent_augmix_consistency_max_batches,
                )
        raw_corrupt_stats = {"enabled": False, "reason": "disabled"}
        if args.enable_raw_corrupt_consistency and raw_corrupt_consistency_loader is not None:
            raw_corrupt_stats = train_raw_corruption_consistency_epoch(
                model=victim.model,
                loader=raw_corrupt_consistency_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=args.device,
                copies=args.raw_corrupt_copies,
                severity=args.raw_corrupt_severity,
                severity_profile=args.raw_corrupt_severity_profile,
                ops=list(args.raw_corrupt_ops),
                prob=args.raw_corrupt_prob,
                consistency_weight=args.raw_corrupt_consistency_weight,
                bce_weight=args.raw_corrupt_bce_weight,
                consistency_loss=args.raw_corrupt_consistency_loss,
                rng=rng,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                freeze_backbone_eval_fn=freeze_backbone_eval_fn,
                max_batches=args.raw_corrupt_max_batches,
                renorm=not args.raw_corrupt_no_renorm,
                clip_abs=args.raw_corrupt_clip_abs,
                view_mode=args.raw_corrupt_view_mode,
                augmix_width=args.raw_augmix_width,
                augmix_depth=args.raw_augmix_depth,
                augmix_alpha=args.raw_augmix_alpha,
                augmix_mixture_mode=args.raw_augmix_mixture_mode,
                augmix_mixture_prob=args.raw_augmix_mixture_prob,
                augmix_mixture_beta_a=args.raw_augmix_mixture_beta_a,
                augmix_mixture_beta_b=args.raw_augmix_mixture_beta_b,
                input_stabilizer_config=raw_input_stabilizer_config,
            )
        mask_shift_stats = {"enabled": False, "reason": "disabled"}
        if args.enable_mask_shift_consistency and mask_shift_consistency_loader is not None:
            mask_shift_stats = train_mask_shift_consistency_epoch(
                model=victim.model,
                loader=mask_shift_consistency_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=args.device,
                copies=args.mask_shift_copies,
                mask_severity=args.mask_shift_mask_severity,
                shift_severity=args.mask_shift_shift_severity,
                consistency_weight=args.mask_shift_consistency_weight,
                bce_weight=args.mask_shift_bce_weight,
                consistency_loss=args.mask_shift_consistency_loss,
                rng=rng,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                freeze_backbone_eval_fn=freeze_backbone_eval_fn,
                max_batches=args.mask_shift_max_batches,
                renorm=not args.mask_shift_no_renorm,
                clip_abs=args.mask_shift_clip_abs,
            )
        source_logit_anchor_loss = float("nan")
        if (
            args.source_logit_anchor_weight > 0
            and source_logit_teacher_model is not None
            and source_logit_anchor_loader is not None
        ):
            source_logit_anchor_loss = train_source_logit_anchor_epoch(
                model=victim.model,
                teacher_model=source_logit_teacher_model,
                loader=source_logit_anchor_loader,
                optimizer=optimizer,
                device=args.device,
                weight=args.source_logit_anchor_weight,
                max_batches=args.source_logit_anchor_batches,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                freeze_backbone_eval_fn=freeze_backbone_eval_fn,
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
            "raw_corrupt_loss": round(float(raw_corrupt_stats.get("loss", float("nan"))), 6)
            if raw_corrupt_stats.get("loss", float("nan")) == raw_corrupt_stats.get("loss", float("nan"))
            else None,
            "raw_corrupt_bce_loss": round(float(raw_corrupt_stats.get("bce_loss", float("nan"))), 6)
            if raw_corrupt_stats.get("bce_loss", float("nan")) == raw_corrupt_stats.get("bce_loss", float("nan"))
            else None,
            "raw_corrupt_consistency_loss": round(
                float(raw_corrupt_stats.get("consistency_loss", float("nan"))), 6
            )
            if raw_corrupt_stats.get("consistency_loss", float("nan"))
            == raw_corrupt_stats.get("consistency_loss", float("nan"))
            else None,
            "mask_shift_loss": round(float(mask_shift_stats.get("loss", float("nan"))), 6)
            if mask_shift_stats.get("loss", float("nan")) == mask_shift_stats.get("loss", float("nan"))
            else None,
            "mask_shift_bce_loss": round(float(mask_shift_stats.get("bce_loss", float("nan"))), 6)
            if mask_shift_stats.get("bce_loss", float("nan")) == mask_shift_stats.get("bce_loss", float("nan"))
            else None,
            "mask_shift_consistency_loss": round(
                float(mask_shift_stats.get("consistency_loss", float("nan"))), 6
            )
            if mask_shift_stats.get("consistency_loss", float("nan"))
            == mask_shift_stats.get("consistency_loss", float("nan"))
            else None,
            "source_logit_anchor_loss": round(source_logit_anchor_loss, 6)
            if source_logit_anchor_loss == source_logit_anchor_loss else None,
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
            "raw_corrupt_stats": raw_corrupt_stats,
            "mask_shift_stats": mask_shift_stats,
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

        # Phase F: quick eval (every eval_every; also last epoch)
        if (epoch % args.eval_every == 0) or (epoch == args.n_epochs):
            qe = quick_eval_super5(victim.model, quick_subset, args.device,
                                   crop_len=args.crop_len)
            entry["quick_eval"] = qe
            print(f"   quick eval: avg AUROC={qe['avg_macro_auroc']}  "
                  f"AUPRC={qe['avg_macro_auprc']}")
            for c, info in qe["per_center"].items():
                print(f"      {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}")
            cur = selected_es_metric(qe)
            improved = (cur == cur) and (cur > best_metric + 1e-6)   # NaN-safe
            if improved:
                best_metric = cur
                best_epoch = epoch
                epochs_since_best = 0
                save_compatible_model_state(victim.model, best_ckpt_path)
                print(f"   ** saved best @ ep{epoch}: {args.es_metric} {best_metric}")
                entry["best_update"] = True
            else:
                epochs_since_best += args.eval_every
                entry["best_update"] = False

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
            "quick_eval": entry.get("quick_eval"),
            "agent_decision": decision,
            "checkpoint_latest": str(checkpoint_latest_path),
            "checkpoint_best": str(checkpoint_best_path if best_epoch == epoch else ""),
        }
        _append_jsonl(diagnostics_epoch_path, diagnostics_payload)

        selection_record = build_checkpoint_selection_record(
            center=args.center_name,
            best_epoch=best_epoch,
            metric_name=args.es_metric,
            metric_value=float(best_metric),
            selection_source=args.quick_eval_source,
            heldout_target_labels_used=bool(quick_eval_plan.uses_heldout_selection),
        )
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
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "checkpoint_selection": selection_record,
            "best_model_path": best_ckpt_path,
            "es_metric": args.es_metric,
            "epochs_since_best": epochs_since_best,
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
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "checkpoint_selection": selection_record,
            "agent_decision": decision["attack_state"],
        }
        _append_jsonl(checkpoint_index_path, latest_index)
        if entry.get("best_update"):
            _atomic_torch_save(ckpt_payload, checkpoint_best_path)
            _append_jsonl(
                checkpoint_index_path,
                {
                    **latest_index,
                    "path": str(checkpoint_best_path),
                    "kind": "best",
                    "reason": f"{args.es_metric} improved",
                },
            )

        # Plan Rev 13.1: early-stop on val_macro_auroc plateau
        if (epoch % args.eval_every == 0) and epochs_since_best >= args.patience:
            print(f"\n[early-stop] patience {args.patience} hit at ep{epoch}; "
                  f"best @ ep{best_epoch} ({args.es_metric}={best_metric})")
            with open(es_path, "w") as f:
                json.dump({
                    "stopped_epoch": epoch, "best_epoch": best_epoch,
                    "best_metric": best_metric, "es_metric": args.es_metric, "patience": args.patience,
                    "n_epochs_run": epoch, "early_stopped": True,
                }, f, indent=2)
            break
    else:
        # Loop completed without early-stop
        with open(es_path, "w") as f:
            json.dump({
                "stopped_epoch": args.n_epochs, "best_epoch": best_epoch,
                "best_metric": best_metric, "es_metric": args.es_metric, "patience": args.patience,
                "n_epochs_run": args.n_epochs, "early_stopped": False,
            }, f, indent=2)

    # Final result
    final = {
        "args":               vars(args),
        "baseline_quick_eval": baseline_qe,
        "best_metric":         best_metric,
        "es_metric":           args.es_metric,
        "n_epochs_run":       len(log["epochs"]),
        "last_quick_eval":    log["epochs"][-1].get("quick_eval") if log["epochs"] else None,
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"Training done. best {args.es_metric}={best_metric} → {best_ckpt_path}")


if __name__ == "__main__":
    main()
