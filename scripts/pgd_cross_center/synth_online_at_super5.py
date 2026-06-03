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
from methods.augmix.severity import AVAILABLE_OPS  # noqa: E402

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
    compute_pos_weight,
    should_save_initial_best_model,
    validate_resume_contract,
)
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    PTBXLDatasetScheme, compute_macro_auroc_auprc,
    evaluate, get_ptbxl_labels_for_scheme, preprocess_ptbxl_all,
)
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from ecg_adv_gen.vae import VAE500RuntimeWrapper  # noqa: E402
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    auroc_to_trust,
    build_anchor_preserving_soft_labels,
    build_k500_internal_val_mask,
    build_latent_augmix_branch_signals as _build_latent_augmix_branch_signals_core,
    derive_class_trust,
    derive_kshot_anchor_class_weights,
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


def configure_torch_acceleration(args: argparse.Namespace) -> None:
    """Apply opt-in PyTorch runtime speed knobs for long GPU runs."""
    if getattr(args, "allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(args, "matmul_precision", ""):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if getattr(args, "cudnn_benchmark", False):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def dataloader_perf_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor and args.prefetch_factor > 0:
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


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


def multilabel_pairwise_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 1.0,
    positive_class_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Pairwise multilabel ranking loss over valid positive-vs-negative labels.

    For each sample, every selected positive class should have a logit at least
    `margin` above every valid negative class. Unknown labels (-1) are ignored.
    With only five Super5 classes, a simple per-row loop is clearer and cheap.
    """
    losses: List[torch.Tensor] = []
    class_filter = None
    if positive_class_indices:
        class_filter = torch.zeros(labels.shape[1], dtype=torch.bool, device=labels.device)
        valid_indices = [idx for idx in positive_class_indices if 0 <= idx < labels.shape[1]]
        if valid_indices:
            class_filter[valid_indices] = True
    for row_logits, row_labels in zip(logits, labels, strict=True):
        valid = row_labels >= 0
        pos = valid & (row_labels > 0.5)
        neg = valid & (row_labels <= 0.5)
        if class_filter is not None:
            pos = pos & class_filter
        if int(pos.sum().item()) == 0 or int(neg.sum().item()) == 0:
            continue
        pair_margin = float(margin) - row_logits[pos].unsqueeze(1) + row_logits[neg].unsqueeze(0)
        losses.append(F.softplus(pair_margin).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_one_epoch_masked_bce_rank_aware(
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
    *,
    rank_loss_weight: float = 0.0,
    rank_loss_margin: float = 1.0,
    rank_loss_positive_indices: Optional[List[int]] = None,
    freeze_backbone_eval_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, float]:
    """Masked BCE plus optional pairwise multilabel rank loss.

    This keeps the historical masked-BCE behavior when rank_loss_weight <= 0.
    The helper is local to this script so older Tier-M entrypoints are not
    affected.
    """
    model.train()
    if freeze_backbone_eval_fn is not None:
        freeze_backbone_eval_fn()
    bce_losses: List[float] = []
    rank_losses: List[float] = []
    total_losses: List[float] = []
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
        if rank_loss_weight > 0:
            rank_loss = multilabel_pairwise_rank_loss(
                logits,
                labels,
                margin=rank_loss_margin,
                positive_class_indices=rank_loss_positive_indices,
            )
        else:
            rank_loss = logits.sum() * 0.0
        loss = bce + float(rank_loss_weight) * rank_loss
        if ewa_params is not None and anchor_lambda > 0:
            anchor = sum(
                (p - p_anchor.detach()).pow(2).sum()
                for p, p_anchor in zip(trainable_params, ewa_params)
            )
            loss = loss + anchor_lambda * anchor
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        if ewa_params is not None and ewa_decay > 0 and ewa_decay < 1.0:
            with torch.no_grad():
                for p, p_anchor in zip(trainable_params, ewa_params):
                    p_anchor.mul_(ewa_decay).add_(p.data, alpha=1 - ewa_decay)
        bce_losses.append(float(bce.item()))
        rank_losses.append(float(rank_loss.item()))
        total_losses.append(float(loss.item()))
    return {
        "bce": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "rank": float(np.mean(rank_losses)) if rank_losses else float("nan"),
        "total": float(np.mean(total_losses)) if total_losses else float("nan"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def _resolve_resume_path(resume: str, output_dir: str) -> Optional[Path]:
    if not resume:
        return None
    if resume == "latest":
        return Path(output_dir) / "checkpoints" / "checkpoint_latest.pt"
    return Path(resume).expanduser()


def _buffer_state(buffer: QualityAwareBuffer) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "max_size": int(buffer.max_size),
        "size": int(len(buffer)),
        "score_list": list(buffer.score_list),
    }
    if len(buffer) > 0:
        state["ecg_tensor"] = torch.stack(buffer.ecg_list).cpu()
        state["label_tensor"] = torch.stack(buffer.label_list).cpu()
    return state


def _restore_buffer_state(buffer: QualityAwareBuffer, state: Dict[str, Any]) -> None:
    buffer.max_size = int(state.get("max_size", buffer.max_size))
    ecg_tensor = state.get("ecg_tensor")
    label_tensor = state.get("label_tensor")
    scores = list(state.get("score_list") or [])
    if ecg_tensor is None or label_tensor is None:
        buffer.ecg_list = []
        buffer.label_list = []
        buffer.score_list = []
        return
    buffer.ecg_list = [row.detach().cpu() for row in ecg_tensor]
    buffer.label_list = [row.detach().cpu() for row in label_tensor]
    buffer.score_list = [float(x) for x in scores[: len(buffer.ecg_list)]]


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
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_epoch_generator": epoch_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any], epoch_rng: np.random.Generator) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if "numpy_epoch_generator" in state:
        epoch_rng.bit_generator.state = state["numpy_epoch_generator"]
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def decoded_signal_invalid_stats(signals: np.ndarray) -> Dict[str, float]:
    if signals.size == 0:
        return {"decoded_invalid_rate": float("nan"), "nan_rate": float("nan"), "flatline_rate": float("nan")}
    finite = np.isfinite(signals).all(axis=tuple(range(1, signals.ndim)))
    p2p = np.ptp(np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0), axis=-1).max(axis=1)
    flatline = p2p < 1e-6
    invalid = (~finite) | flatline
    return {
        "decoded_invalid_rate": float(np.mean(invalid)),
        "nan_rate": float(np.mean(~finite)),
        "flatline_rate": float(np.mean(flatline)),
    }


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
    attack_loss_mode: str = "bce",
    attack_pos_hide_weight: float = 1.0,
    attack_neg_add_weight: float = 0.25,
    attack_negative_exclude_indices: Optional[List[int]] = None,
    attack_neg_topk: int = 0,
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
    attack_negative_exclude_indices = attack_negative_exclude_indices or []

    def targeted_loss_per_sample(logits: torch.Tensor) -> torch.Tensor:
        if attack_loss_mode == "bce":
            return F.binary_cross_entropy_with_logits(logits, labels_t, reduction="none").mean(dim=1)
        pos_mask = labels_t > 0.5
        neg_mask = labels_t < 0.5
        valid_exclude = [idx for idx in attack_negative_exclude_indices if 0 <= idx < labels_t.shape[1]]
        if valid_exclude:
            neg_mask = neg_mask.clone()
            neg_mask[:, valid_exclude] = False
        pos_terms = F.softplus(-logits)
        neg_terms = F.softplus(logits)
        pos_denom = pos_mask.sum(dim=1).clamp(min=1).to(dtype=logits.dtype)
        pos_loss = (pos_terms * pos_mask.to(dtype=logits.dtype)).sum(dim=1) / pos_denom
        if int(attack_neg_topk) > 0:
            neg_losses = []
            for row_terms, row_mask in zip(neg_terms, neg_mask, strict=True):
                selected = row_terms[row_mask]
                if selected.numel() == 0:
                    neg_losses.append(row_terms.sum() * 0.0)
                else:
                    kk = min(int(attack_neg_topk), selected.numel())
                    neg_losses.append(selected.topk(kk).values.mean())
            neg_loss = torch.stack(neg_losses)
        else:
            neg_denom = neg_mask.sum(dim=1).clamp(min=1).to(dtype=logits.dtype)
            neg_loss = (neg_terms * neg_mask.to(dtype=logits.dtype)).sum(dim=1) / neg_denom
        if attack_loss_mode == "positive_hide":
            return float(attack_pos_hide_weight) * pos_loss
        if attack_loss_mode == "negative_add":
            return float(attack_neg_add_weight) * neg_loss
        if attack_loss_mode == "pos_hide_neg_add":
            return float(attack_pos_hide_weight) * pos_loss + float(attack_neg_add_weight) * neg_loss
        raise ValueError(f"unknown attack_loss_mode={attack_loss_mode!r}")

    clean_target = targeted_loss_per_sample(clean_logits)
    adv_target = targeted_loss_per_sample(adv_logits)
    target_gain = adv_target - clean_target
    return {
        "n": int(labels_t.shape[0]),
        "success_rate": float(success.float().mean().item()),
        "clean_bce_mean": float(clean_bce.mean().item()),
        "adv_bce_mean": float(adv_bce.mean().item()),
        "loss_gain_mean": float(gain.mean().item()),
        "loss_gain_p50": float(torch.quantile(gain, 0.50).item()),
        "loss_gain_p90": float(torch.quantile(gain, 0.90).item()),
        "attack_loss_mode": attack_loss_mode,
        "attack_target_clean_loss_mean": float(clean_target.mean().item()),
        "attack_target_adv_loss_mean": float(adv_target.mean().item()),
        "attack_target_loss_gain_mean": float(target_gain.mean().item()),
        "attack_target_loss_gain_p50": float(torch.quantile(target_gain, 0.50).item()),
        "attack_target_loss_gain_p90": float(torch.quantile(target_gain, 0.90).item()),
        "_loss_gain_per_sample": gain.detach().cpu().numpy().astype(np.float32).tolist(),
        "_attack_target_loss_gain_per_sample": target_gain.detach().cpu().numpy().astype(np.float32).tolist(),
    }


def agent_attack_decision(
    entry: Dict[str, Any],
    *,
    asr_low_threshold: float,
    consecutive_low_asr: int,
) -> Dict[str, Any]:
    def _float_or_nan(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    asr = _float_or_nan(entry.get("asr_overall", float("nan")))
    invalid = _float_or_nan(entry.get("decoded_invalid_rate", float("nan")))
    attack_vs_anchor = entry.get("attack_vs_anchor", {}) or {}
    loss_gain = _float_or_nan(attack_vs_anchor.get("loss_gain_mean"))
    if asr != asr:
        state = "no_attack_or_disabled"
        action = "continue_if_this_is_an_ablation"
    elif invalid == invalid and invalid > 0.05:
        state = "attack_too_strong_or_decode_invalid"
        action = "lower_hull_lambda_or_attack_strength_before_paper_run"
    elif asr < asr_low_threshold:
        state = "attack_too_weak"
        action = "increase_attack_strength_only_if_repeated_and_source_floor_is_safe"
    elif asr > 0.85 and (loss_gain != loss_gain or loss_gain > 0.05):
        state = "attack_too_strong"
        action = "lower_adv_weight_or_attack_strength_if_target/source_metrics_drop"
    elif consecutive_low_asr > 0:
        state = "watch_low_asr"
        action = "continue_but_watch_next_epoch"
    else:
        state = "healthy"
        action = "continue"
    return {
        "attack_state": state,
        "action": action,
        "stop_or_continue": "continue" if state not in {"attack_too_strong_or_decode_invalid"} else "review_before_continue",
        "asr_low_threshold": float(asr_low_threshold),
        "consecutive_low_asr": int(consecutive_low_asr),
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
    synth_latents: np.ndarray,    # (N, C_lat, L_lat)
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
      adv_signals_ct:   (K, 12, L)  float32
      anchor_signals_ct:(K, 12, L)  float32  (clean reference for sem-gate)
      labels_one_hot:   (K, C)         float32
      delta_stats:      mean / max L2 norm plus optional latent-hull weight stats
    """
    if picked_indices is None:
        num_classes = synth_labels.shape[1]
        pick = stratified_sample_synth(synth_labels, K_anchor, num_classes, rng)
    else:
        pick = picked_indices

    if len(pick) == 0:
        signal_len = int(getattr(pgd_gen.victim, "latent_preproc_length", pgd_gen.victim.crop_len))
        return (np.empty((0, 12, signal_len), dtype=np.float32),
                np.empty((0, 12, signal_len), dtype=np.float32),
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

    def _apply_augmix_op_np(sig_ct: np.ndarray, op_name: str, op_severity: int) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        return _apply_op(sig_t, op_name, int(op_severity)).cpu().numpy().astype(np.float32, copy=False)

    return _build_latent_augmix_branch_signals_core(
        anchor_signals_ct,
        adv_signals_ct,
        copies=copies,
        severity=severity,
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
    loss_gain_per_sample: Optional[np.ndarray] = None,
    min_loss_gain: Optional[float] = None,
    loss_gain_score_scale: float = 0.0,
    loss_gain_score_strength: float = 0.0,
) -> Dict[str, Any]:
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
    n_dropped_by_loss_gain = 0
    accepted_indices: List[int] = []
    loss_gain_score_multipliers: List[float] = []
    if loss_gain_per_sample is not None:
        loss_gain_per_sample = np.asarray(loss_gain_per_sample, dtype=np.float32)
        if loss_gain_per_sample.shape[0] != adv_signals_ct.shape[0]:
            raise ValueError(
                "loss_gain_per_sample length mismatch: "
                f"{loss_gain_per_sample.shape[0]} vs {adv_signals_ct.shape[0]}"
            )
    for i in range(adv_signals_ct.shape[0]):
        sig_250 = _center_crop_ct(adv_signals_ct[i], crop_len)         # (12, 250)
        target_idx = int(target_one_hot[i].argmax())
        target_class = CLASS_NAMES_SUPER5[target_idx] if target_idx < len(CLASS_NAMES_SUPER5) else None
        trust = float(class_trust.get(target_class, 1.0)) if class_trust else 1.0
        if trust <= 0.0:
            n_dropped_by_trust += 1
            continue
        if min_loss_gain is not None and loss_gain_per_sample is not None:
            if float(loss_gain_per_sample[i]) < float(min_loss_gain):
                n_dropped_by_loss_gain += 1
                continue
        prob_t = float(1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, victim_logits[i, target_idx])))))
        if prob_t < boundary_prob_min or prob_t > boundary_prob_max:
            n_dropped_by_boundary += 1
            continue
        if label_mode == "hard":
            lbl = torch.full((target_one_hot.shape[1],), -1.0)
            lbl[target_idx] = 1.0
        elif label_mode == "multi_hot_hard":
            lbl = torch.from_numpy((target_one_hot[i] > 0.5).astype(np.float32))
        elif label_mode == "latent_soft":
            lbl = torch.from_numpy(np.clip(target_one_hot[i].astype(np.float32), 0.0, 1.0))
        else:
            if teacher_probs is None:
                raise ValueError(f"teacher_probs required for label_mode={label_mode}")
            soft = torch.from_numpy(
                np.clip(teacher_probs[i].astype(np.float32), 0.0, 1.0)
            )
            if label_mode == "teacher_soft":
                lbl = soft
            elif label_mode == "mixed_soft":
                hard_full = torch.zeros((target_one_hot.shape[1],), dtype=torch.float32)
                hard_full[target_idx] = 1.0
                mix = max(0.0, min(1.0, float(teacher_mix)))
                lbl = mix * soft + (1.0 - mix) * hard_full
            elif label_mode == "latent_mixed_teacher":
                latent_soft = torch.from_numpy(
                    np.clip(target_one_hot[i].astype(np.float32), 0.0, 1.0)
                )
                mix = max(0.0, min(1.0, float(teacher_mix)))
                lbl = mix * soft + (1.0 - mix) * latent_soft
            else:
                raise ValueError(f"unsupported adv label_mode={label_mode}")
            if soft_target_floor > 0.0:
                lbl[target_idx] = torch.clamp(lbl[target_idx], min=float(soft_target_floor))
        score = (1.0 - 2.0 * abs(prob_t - 0.5)) * trust                # ∈ [0, trust]
        if (
            loss_gain_per_sample is not None
            and loss_gain_score_scale > 0.0
            and loss_gain_score_strength > 0.0
        ):
            gain = float(loss_gain_per_sample[i])
            multiplier = 1.0 + float(loss_gain_score_strength) * math.tanh(
                gain / float(loss_gain_score_scale)
            )
            multiplier = max(0.05, float(multiplier))
            score *= multiplier
            loss_gain_score_multipliers.append(multiplier)
        buffer.add_one(
            torch.from_numpy(np.ascontiguousarray(sig_250)).float(),
            lbl,
            score,
        )
        accepted_indices.append(int(i))
        n_pushed += 1
    return {
        "n_pushed": n_pushed,
        "n_dropped_by_trust": n_dropped_by_trust,
        "n_dropped_by_boundary": n_dropped_by_boundary,
        "n_dropped_by_loss_gain": n_dropped_by_loss_gain,
        "accepted_indices": accepted_indices,
        "label_mode": label_mode,
        "loss_gain_score_mult_mean": (
            float(np.mean(loss_gain_score_multipliers))
            if loss_gain_score_multipliers else float("nan")
        ),
        "loss_gain_score_mult_min": (
            float(np.min(loss_gain_score_multipliers))
            if loss_gain_score_multipliers else float("nan")
        ),
        "loss_gain_score_mult_max": (
            float(np.max(loss_gain_score_multipliers))
            if loss_gain_score_multipliers else float("nan")
        ),
    }


@torch.no_grad()
def score_target_real_anchors(
    weight_path: str,
    model_name: str,
    signals_tc: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    *,
    device: str,
    crop_len: int,
    mode: str,
    low_margin_weight: float,
    batch_size: int,
    ecgtwin_wrapper: Any,
    vae_backend: str,
    latent_preproc_length: int,
    latent_amp_clamp: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Score K-shot real anchors using a frozen checkpoint.

    Scores are used only to order/select latent anchors inside the known K-shot
    pool. They must be computed from the K-shot split, not from held-out PN2021
    target-center labels.
    """
    if signals_tc.size == 0:
        return {}, {"enabled": False, "reason": "empty target-real train split"}
    labels = np.asarray(labels, dtype=np.float32)
    record_ids = np.asarray(record_ids).astype(str)
    if labels.shape[0] != signals_tc.shape[0] or record_ids.shape[0] != signals_tc.shape[0]:
        raise ValueError(
            "anchor scoring target-real arrays mismatch: "
            f"signals={signals_tc.shape} labels={labels.shape} record_ids={record_ids.shape}"
        )
    scorer = EfficientNetVictimTierM(
        weight_path=weight_path,
        device=device,
        ecgtwin_wrapper=ecgtwin_wrapper,
        num_classes=NUM_SUPER5,
        crop_len=crop_len,
        model_name=model_name,
        latent_backend=vae_backend,
        latent_preproc_length=latent_preproc_length,
        latent_amp_clamp=latent_amp_clamp,
    )
    scorer.eval()
    logits_chunks: List[torch.Tensor] = []
    for i in range(0, signals_tc.shape[0], batch_size):
        chunk = signals_tc[i:i + batch_size]
        if chunk.shape[1:] == (crop_len, 12):
            x = torch.from_numpy(chunk).float().permute(0, 2, 1).contiguous().to(device)
        elif chunk.shape[1:] == (12, crop_len):
            x = torch.from_numpy(chunk).float().contiguous().to(device)
        else:
            raise ValueError(f"unexpected target-real signal shape for scoring: {chunk.shape}")
        logits_chunks.append(scorer.compute_logits_from_ecg(x).detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    labels_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    bce = F.binary_cross_entropy_with_logits(logits, labels_t, reduction="none")
    probs = torch.sigmoid(logits)
    pos_mask = labels_t > 0.5
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    pos_bce = (bce * pos_mask.float()).sum(dim=1) / pos_count
    full_bce = bce.mean(dim=1)
    pos_uncertainty = ((1.0 - 2.0 * torch.abs(probs - 0.5)).clamp(min=0.0) * pos_mask.float()).sum(dim=1) / pos_count

    mode = str(mode)
    if mode == "hard_bce":
        score = full_bce
    elif mode == "positive_hard_bce":
        score = pos_bce
    elif mode == "low_margin":
        score = pos_uncertainty
    elif mode == "hard_bce_plus_low_margin":
        score = pos_bce + float(low_margin_weight) * pos_uncertainty
    else:
        raise ValueError(f"unsupported anchor_score_mode={mode!r}")
    score_np = score.detach().cpu().numpy().astype(np.float64)
    score_map = {str(rid): float(s) for rid, s in zip(record_ids, score_np)}
    info = {
        "enabled": True,
        "weight_path": weight_path,
        "mode": mode,
        "low_margin_weight": float(low_margin_weight),
        "n_scored": int(score_np.shape[0]),
        "score_mean": float(np.mean(score_np)) if score_np.size else float("nan"),
        "score_p50": float(np.quantile(score_np, 0.50)) if score_np.size else float("nan"),
        "score_p90": float(np.quantile(score_np, 0.90)) if score_np.size else float("nan"),
        "score_max": float(np.max(score_np)) if score_np.size else float("nan"),
    }
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_map, info


def build_anchor_score_vector(
    score_map: Dict[str, float],
    source_meta: Dict[str, Any],
    n_pool: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if "record_ids" not in source_meta:
        raise ValueError("anchor scoring requires record_ids in the latent pool")
    record_ids = np.asarray(source_meta["record_ids"]).astype(str)
    if record_ids.shape[0] != n_pool:
        raise ValueError(
            f"latent pool record_ids length mismatch: {record_ids.shape[0]} vs {n_pool}"
        )
    scores = np.full((n_pool,), np.nan, dtype=np.float64)
    n_matched = 0
    for i, rid in enumerate(record_ids):
        if str(rid) in score_map:
            scores[i] = float(score_map[str(rid)])
            n_matched += 1
    return scores, {
        "n_pool": int(n_pool),
        "n_matched_record_ids": int(n_matched),
        "n_missing_record_ids": int(n_pool - n_matched),
    }


def apply_anchor_scores_to_walker(
    walker: StratifiedPoolWalker,
    scores: np.ndarray,
    *,
    top_frac: float,
    min_per_class: int,
) -> Dict[str, Any]:
    """Order and optionally trim walker pools by precomputed anchor scores."""
    top_frac = float(top_frac)
    min_per_class = max(1, int(min_per_class))
    summary: Dict[str, Any] = {
        "enabled": True,
        "top_frac": top_frac,
        "min_per_class": min_per_class,
        "per_class": {},
        "per_source_class": {},
    }

    def order_pool(pool: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        pool = np.asarray(pool, dtype=np.int64)
        if pool.size == 0:
            return pool, {"before": 0, "after": 0}
        vals = scores[pool]
        finite_vals = vals[np.isfinite(vals)]
        sortable = np.nan_to_num(vals, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ordered = pool[np.argsort(-sortable)]
        before = int(pool.size)
        keep_n = before
        if 0.0 < top_frac < 1.0:
            keep_n = min(before, max(min_per_class, int(math.ceil(before * top_frac))))
            ordered = ordered[:keep_n]
        info = {
            "before": before,
            "after": int(ordered.size),
            "finite_scores": int(finite_vals.size),
            "score_p50": float(np.quantile(finite_vals, 0.50)) if finite_vals.size else float("nan"),
            "score_p90": float(np.quantile(finite_vals, 0.90)) if finite_vals.size else float("nan"),
            "score_max": float(np.max(finite_vals)) if finite_vals.size else float("nan"),
        }
        return ordered.astype(np.int64, copy=False), info

    for cls in walker.classes:
        ordered, info = order_pool(walker.cls_pools[cls])
        walker.cls_pools[cls] = ordered
        walker.cursors[cls] = 0
        walker.epochs_completed[cls] = 0
        summary["per_class"][cls] = info

    for key in list(walker.source_cls_pools):
        ordered, info = order_pool(walker.source_cls_pools[key])
        walker.source_cls_pools[key] = ordered
        walker.source_cursors[key] = 0
        walker.source_epochs_completed[key] = 0
        cls, source = key
        summary["per_source_class"].setdefault(cls, {})[source] = info
    return summary


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--center_name", required=True, help="cpsc_2018_extra | ningbo")
    p.add_argument("--ref_meta_json",
                   help="path to {tag}_k200.meta.json (for record_id exclusion in eval)")
    p.add_argument("--synth_npz", required=True,
                   help="Latent npz: {latents (N,C,L), labels (N,5)}")
    p.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
    p.add_argument("--model_name", default="efficientnet1dv2",
                   choices=available_model_names())
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_real_npz", default="",
                   help="Optional selected target-center real ECG npz with signals (N,L,12) and labels.")

    # PN2021 quick eval
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--quick_eval_centers", nargs='+',
                   default=["chapman_shaoxing", "cpsc_2018_extra", "georgia",
                            "ningbo"])
    p.add_argument("--quick_eval_n_per_center", type=int, default=1000)
    p.add_argument(
        "--quick_eval_min_pos",
        type=int,
        default=10,
        help=(
            "Minimum positive labels required for a class in quick_eval metrics. "
            "Keep the default 10 for K500/K1000 runs; lower it only for explicit "
            "low-K screening where the internal target validation split is small."
        ),
    )
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
    p.add_argument("--sampling_rate", type=int, default=100)
    p.add_argument("--input_len", type=int, default=1000)
    p.add_argument("--preprocess_mode", default="legacy_ecgfounder_filter",
                   choices=["minimal_resample", "legacy_ecgfounder_filter",
                            "raw_for_generation_or_digital"])
    p.add_argument("--norm_mode", default="per_sample_global",
                   choices=["per_sample_global", "none"])
    p.add_argument("--vae_backend", default="ecgtwin1024",
                   choices=["ecgtwin1024", "diffusets500_v1"],
                   help="Latent decoder backend used for online AT.")
    p.add_argument("--vae500_ckpt", default="",
                   help="Required when --vae_backend diffusets500_v1.")
    p.add_argument("--vae500_variant", default=None)
    p.add_argument("--latent_amp_clamp", type=float, default=3.0,
                   help="Clamp decoded latent ECG before z-score; <=0 disables.")

    # PGD (Plan Rev 13: K_pgd=10 + ε=2.0 + K_anchor=300)
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_K", type=int, default=10)
    p.add_argument("--pgd_batch", type=int, default=32)        # Issue #45
    p.add_argument("--K_anchor", type=int, default=300)
    p.add_argument("--pgd_alpha", type=float, default=None)
    p.add_argument("--delta_init_scale", type=float, default=0.1)
    p.add_argument(
        "--attack_loss_mode",
        choices=["bce", "positive_hide", "negative_add", "pos_hide_neg_add"],
        default="bce",
        help="Latent attack objective. bce preserves historical full-label BCE behavior.",
    )
    p.add_argument("--attack_pos_hide_weight", type=float, default=1.0)
    p.add_argument("--attack_neg_add_weight", type=float, default=0.25)
    p.add_argument(
        "--attack_negative_exclude_classes",
        nargs="*",
        default=[],
        help="Super5 classes excluded from negative-add terms, e.g. NORM.",
    )
    p.add_argument(
        "--attack_neg_topk",
        type=int,
        default=0,
        help="If >0, negative-add uses top-k negative class losses per sample.",
    )
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
    p.add_argument(
        "--anchor_score_mode",
        choices=[
            "none",
            "hard_bce",
            "positive_hard_bce",
            "low_margin",
            "hard_bce_plus_low_margin",
        ],
        default="none",
        help=(
            "Optional boundary-targeted anchor ordering. Scores are computed "
            "only on the K-shot target-real train split, then used to order or "
            "trim the latent anchor pools before sampling."
        ),
    )
    p.add_argument(
        "--anchor_score_ckpt",
        default="",
        help=(
            "Checkpoint used to score boundary anchors. If empty and "
            "--anchor_score_mode is not none, the current --init_ckpt is used."
        ),
    )
    p.add_argument(
        "--anchor_score_low_margin_weight",
        type=float,
        default=0.5,
        help="Weight for the low-margin term in hard_bce_plus_low_margin mode.",
    )
    p.add_argument(
        "--anchor_score_top_frac",
        type=float,
        default=1.0,
        help=(
            "If in (0,1), keep only the top fraction of scored anchors inside "
            "each class pool. 1.0 keeps all anchors but samples hardest first."
        ),
    )
    p.add_argument(
        "--anchor_score_min_per_class",
        type=int,
        default=8,
        help="Minimum anchors retained per class when --anchor_score_top_frac < 1.",
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
    p.add_argument("--adv_accept_min_loss_gain", type=float, default=None,
                   help="If set, only push adversarial samples with adv_bce - clean_bce >= this value.")
    p.add_argument(
        "--adv_accept_gain_mode",
        choices=["bce", "target"],
        default="bce",
        help=(
            "Loss-gain vector used by --adv_accept_min_loss_gain and soft score scaling. "
            "bce is legacy full BCE; target follows --attack_loss_mode."
        ),
    )
    p.add_argument("--adv_loss_gain_score_scale", type=float, default=0.0,
                   help="If >0 with --adv_loss_gain_score_strength >0, softly reweight buffer scores by loss_gain / scale.")
    p.add_argument("--adv_loss_gain_score_strength", type=float, default=0.0,
                   help="Soft loss-gain score multiplier strength. Multiplier is 1 + strength * tanh(loss_gain / scale).")
    p.add_argument(
        "--rank_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Optional pairwise multilabel ranking loss weight. Default 0 keeps "
            "historical masked-BCE training. Use with "
            "--rank_loss_positive_classes to target rare abnormal positives."
        ),
    )
    p.add_argument("--rank_loss_margin", type=float, default=1.0)
    p.add_argument(
        "--rank_loss_positive_classes",
        nargs="*",
        default=[],
        help=(
            "Positive classes included in the ranking term, e.g. CD HYP MI STTC. "
            "Empty means all positive classes."
        ),
    )

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
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    p.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable CuDNN benchmark for fixed-shape long runs; disables deterministic CuDNN.",
    )
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

    Returns (latents (N,C_lat,L_lat), labels (N,C), center_name, source_meta).
    """
    p = Path(synth_npz_path)
    if p.name.endswith(".latent.npz"):
        latent_path = p
    else:
        # Convert e.g. extra_pool300.npz → extra_pool300.latent.npz
        latent_path = p.with_name(p.stem + ".latent.npz")
        if not latent_path.exists():
            raise SystemExit(
                f"latent pool .npz not found: tried {latent_path}. "
                f"Re-run generate_center_synth.py with --save_latent."
            )
    d = np.load(str(latent_path), allow_pickle=True)
    latents = d["latents"]
    labels = d["labels"]
    center = str(d["center_name"]) if "center_name" in d.files else "?"
    assert latents.ndim == 3, f"bad synth latent shape: {latents.shape}"
    assert labels.shape[0] == latents.shape[0]

    source_ids = None
    source_names = None
    source_labels = None
    if "source_ids" in d.files and "source_names" in d.files:
        source_ids = d["source_ids"].astype(np.int64)
        source_names = [str(x) for x in d["source_names"].tolist()]
        if source_ids.shape[0] != latents.shape[0]:
            raise ValueError("source_ids length does not match latents")
        source_labels = np.asarray([
            source_names[int(i)] if 0 <= int(i) < len(source_names) else f"source_{int(i)}"
            for i in source_ids
        ])
    else:
        source_ids = np.zeros((latents.shape[0],), dtype=np.int64)
        source_names = ["unknown"]
        source_labels = np.asarray(["unknown"] * latents.shape[0])

    source_meta = {
        "source_ids": source_ids,
        "source_names": source_names,
        "source_labels": source_labels,
        "has_source_metadata": "source_ids" in d.files and "source_names" in d.files,
    }
    if "record_ids" in d.files:
        source_meta["record_ids"] = d["record_ids"].astype(str)
    return latents.astype(np.float32), labels.astype(np.float32), center, source_meta


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    configure_torch_acceleration(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Synth-anchored online AT (Super5) — Plan Rev 13.2")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 72)

    # ── Build Super5 victim early (needed for both sanity + training) ──────
    if args.vae_backend == "ecgtwin1024":
        print("[setup] Loading ECGTwin (encoder + decoder, no text model)...")
        ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
        # Keep old 100 Hz behavior when input_len=1000, but allow the original
        # ECGTwin 1024-point VAE latent manifold to feed 500 Hz classifiers by
        # decoding to 1024 and differentiably interpolating to input_len.
        latent_preproc_length = int(args.input_len)
    elif args.vae_backend == "diffusets500_v1":
        if not args.vae500_ckpt:
            raise ValueError("--vae500_ckpt is required when --vae_backend diffusets500_v1")
        print(f"[setup] Loading VAE500 backend from {args.vae500_ckpt}")
        ecgtwin = VAE500RuntimeWrapper(
            args.vae500_ckpt,
            variant=args.vae500_variant,
            device=args.device,
        )
        latent_preproc_length = int(args.input_len)
        if args.roundtrip_anchor_n > 0:
            print("[setup] disabling roundtrip_anchor_n for VAE500 backend")
            args.roundtrip_anchor_n = 0
    else:
        raise ValueError(f"unsupported --vae_backend {args.vae_backend!r}")
    print(f"[setup] Loading Super5 victim from {args.init_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=ecgtwin,
        num_classes=NUM_SUPER5,
        crop_len=args.crop_len,
        model_name=args.model_name,
        latent_backend=args.vae_backend,
        latent_preproc_length=latent_preproc_length,
        latent_amp_clamp=args.latent_amp_clamp,
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
    attack_negative_exclude_indices: List[int] = []
    attack_negative_exclude_classes: List[str] = []
    for cls in args.attack_negative_exclude_classes:
        cls = cls.upper()
        if cls not in SUPER5_TO_IDX:
            raise SystemExit(
                f"unknown --attack_negative_exclude_classes class {cls!r}; "
                f"valid={CLASS_NAMES_SUPER5}"
            )
        attack_negative_exclude_classes.append(cls)
        attack_negative_exclude_indices.append(int(SUPER5_TO_IDX[cls]))
    rank_loss_positive_classes: List[str] = []
    rank_loss_positive_indices: List[int] = []
    for cls in args.rank_loss_positive_classes:
        cls = cls.upper()
        if cls not in SUPER5_TO_IDX:
            raise SystemExit(
                f"unknown --rank_loss_positive_classes class {cls!r}; "
                f"valid={CLASS_NAMES_SUPER5}"
            )
        rank_loss_positive_classes.append(cls)
        rank_loss_positive_indices.append(int(SUPER5_TO_IDX[cls]))
    print(
        "[setup] attack objective: "
        f"mode={args.attack_loss_mode} pos_w={args.attack_pos_hide_weight} "
        f"neg_w={args.attack_neg_add_weight} neg_exclude={attack_negative_exclude_classes} "
        f"neg_topk={args.attack_neg_topk} accept_gain={args.adv_accept_gain_mode}"
    )
    print(
        "[setup] rank-aware loss: "
        f"weight={args.rank_loss_weight} margin={args.rank_loss_margin} "
        f"positive_classes={rank_loss_positive_classes or '<all positives>'}"
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
            f"copies={args.latent_augmix_copies} width={args.latent_augmix_width} "
            f"depth={args.latent_augmix_depth} severity={args.latent_augmix_severity} "
            f"w_lat_cap={args.latent_augmix_latent_weight_cap} "
            f"ops={args.latent_augmix_ops}",
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
    from scripts.triple_labels.label_schemes import get_scheme
    scheme = get_scheme("super5")

    label_cache = os.path.join(args.output_dir, "ptbxl_labels")
    train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=list(range(1, 9))
    )
    val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=[9]
    )

    use_default_100hz_cache = (
        args.ptbxl_prep == DEFAULT_PTBXL_PREP
        and int(args.sampling_rate) == 100
        and int(args.input_len) == 1000
    )
    if os.path.exists(args.ptbxl_prep) and (args.ptbxl_prep != DEFAULT_PTBXL_PREP or use_default_100hz_cache):
        cache_path = args.ptbxl_prep
    else:
        cache_path = os.path.join(args.output_dir, "ptbxl_preprocessed.npy")
    print(f"[setup] PTBXL preprocessed cache → {cache_path}")
    all_sig = preprocess_ptbxl_all(
        args.ptbxl_raw,
        cache_path,
        csv_path=args.ptbxl_csv,
        target_fs=args.sampling_rate,
        target_len=args.input_len,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )
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
            drop_last=True,
            **dataloader_perf_kwargs(args),
        )
        print(
            f"[setup] source-logit anchor enabled: weight={args.source_logit_anchor_weight} "
            f"max_batches={args.source_logit_anchor_batches or 'full'}",
            flush=True,
        )
    target_real_ds = None
    target_val_quick_subset = None
    target_val_record_ids: set[str] = set()
    target_train_signals_tc: Optional[np.ndarray] = None
    target_train_labels: Optional[np.ndarray] = None
    target_train_record_ids: Optional[np.ndarray] = None
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
        if real_signals.shape[1:] == (12, args.input_len):
            real_signals = real_signals.transpose(0, 2, 1)
        if real_signals.shape[1:] != (args.input_len, 12):
            raise ValueError(
                f"target_real_npz signals must be (N,{args.input_len},12) "
                f"or (N,12,{args.input_len}), got {real_signals.shape}"
            )
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
            real_record_ids = real_record_ids[train_mask]
        target_train_signals_tc = real_signals.astype(np.float32, copy=False)
        target_train_labels = real_labels.astype(np.float32, copy=False)
        target_train_record_ids = real_record_ids.astype(str, copy=False)
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

    anchor_score_vector: Optional[np.ndarray] = None
    anchor_score_info: Dict[str, Any] = {"enabled": False, "mode": args.anchor_score_mode}
    if args.anchor_score_mode != "none":
        if target_train_signals_tc is None or target_train_labels is None or target_train_record_ids is None:
            raise ValueError("--anchor_score_mode requires --target_real_npz")
        score_ckpt = args.anchor_score_ckpt or args.init_ckpt
        print(
            f"[setup] scoring target-real anchors: mode={args.anchor_score_mode} "
            f"ckpt={score_ckpt} top_frac={args.anchor_score_top_frac}",
            flush=True,
        )
        score_map, score_info = score_target_real_anchors(
            score_ckpt,
            args.model_name,
            target_train_signals_tc,
            target_train_labels,
            target_train_record_ids,
            device=args.device,
            crop_len=args.crop_len,
            mode=args.anchor_score_mode,
            low_margin_weight=args.anchor_score_low_margin_weight,
            batch_size=args.batch_size,
            ecgtwin_wrapper=ecgtwin,
            vae_backend=args.vae_backend,
            latent_preproc_length=latent_preproc_length,
            latent_amp_clamp=args.latent_amp_clamp,
        )
        anchor_score_vector, match_info = build_anchor_score_vector(
            score_map,
            source_meta,
            n_pool=int(synth_labels.shape[0]),
        )
        anchor_score_info = {**score_info, **match_info}
        print(f"[setup] anchor score info: {anchor_score_info}", flush=True)

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5),
        dtype=torch.float32, device=args.device)
    print(f"[loss] pos_weight: {pos_weight.cpu().tolist()}")

    # reduction='none' for mask × bce (-1 sentinel handling)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_perf_kwargs(args),
    )

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
    if args.quick_eval_source == "target_real_val":
        if target_val_quick_subset is None:
            raise RuntimeError("target_real_val quick eval requested but no target val split exists")
        quick_subset = target_val_quick_subset
        print(
            f"[setup] quick_eval_source=target_real_val; "
            f"n={quick_subset[args.center_name]['signals_tc'].shape[0]} "
            f"(K500-internal validation, no PN2021 held-out selection)",
            flush=True,
        )
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

        qe_cache = os.path.join(args.output_dir,
                                f"quick_eval_subset_n{args.quick_eval_n_per_center}.cache")
        quick_subset = build_quick_eval_subset_super5(
            centers=args.quick_eval_centers, data_dir=args.data_dir,
            n_per_center=args.quick_eval_n_per_center, cache_path=qe_cache,
            seed=args.seed, exclude_record_ids=excluded,
        )

    print("[baseline] Computing baseline quick-eval ...")
    baseline_qe = quick_eval_super5(victim.model, quick_subset, args.device,
                                    crop_len=args.crop_len,
                                    min_pos=args.quick_eval_min_pos)
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
            attack_loss_mode=args.attack_loss_mode,
            attack_pos_hide_weight=args.attack_pos_hide_weight,
            attack_neg_add_weight=args.attack_neg_add_weight,
            attack_negative_exclude_indices=attack_negative_exclude_indices,
            attack_neg_topk=args.attack_neg_topk,
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
            attack_loss_mode=args.attack_loss_mode,
            attack_pos_hide_weight=args.attack_pos_hide_weight,
            attack_neg_add_weight=args.attack_neg_add_weight,
            attack_negative_exclude_indices=attack_negative_exclude_indices,
            attack_neg_topk=args.attack_neg_topk,
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
            "accept_min_loss_gain": args.adv_accept_min_loss_gain,
            "accept_gain_mode": args.adv_accept_gain_mode,
            "loss_gain_score_scale": float(args.adv_loss_gain_score_scale),
            "loss_gain_score_strength": float(args.adv_loss_gain_score_strength),
            "hull_mix_label_mode": args.hull_mix_label_mode,
            "hull_label_lambda_y": args.hull_label_lambda_y,
            "hull_label_positive": args.hull_label_positive,
            "hull_label_negative_floor": args.hull_label_negative_floor,
            "hull_label_new_class_cap": args.hull_label_new_class_cap,
        },
        "attack_objective": {
            "mode": args.attack_loss_mode,
            "pos_hide_weight": float(args.attack_pos_hide_weight),
            "neg_add_weight": float(args.attack_neg_add_weight),
            "negative_exclude_classes": list(attack_negative_exclude_classes),
            "negative_exclude_indices": list(attack_negative_exclude_indices),
            "neg_topk": int(args.attack_neg_topk),
        },
        "rank_loss": {
            "weight": float(args.rank_loss_weight),
            "margin": float(args.rank_loss_margin),
            "positive_classes": list(rank_loss_positive_classes),
            "positive_indices": list(rank_loss_positive_indices),
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
            "anchor_score": anchor_score_info,
        },
        "latent_augmix_branch": {
            "enabled": bool(args.enable_latent_augmix_branch),
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "severity": int(args.latent_augmix_severity),
            "latent_weight_cap": float(args.latent_augmix_latent_weight_cap),
            "ops": list(args.latent_augmix_ops),
            "renorm": not bool(args.no_latent_augmix_renorm),
            "clip_abs": float(args.latent_augmix_clip_abs),
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
    if anchor_score_vector is not None:
        anchor_score_apply_info = apply_anchor_scores_to_walker(
            walker,
            anchor_score_vector,
            top_frac=args.anchor_score_top_frac,
            min_per_class=args.anchor_score_min_per_class,
        )
        anchor_score_info["pool_ordering"] = anchor_score_apply_info
        log["anchor_sampling"]["anchor_score"] = anchor_score_info
        print(
            f"[setup] boundary-targeted anchor pools applied: "
            f"{json.dumps(anchor_score_apply_info['per_class'], ensure_ascii=True)}",
            flush=True,
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
                "multilabel_negative_label_asr": float("nan"),
                "sample_any_negative_above_0p5_asr": float("nan"),
                "per_class_negative_label_asr": {},
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
                attack_loss_mode=args.attack_loss_mode,
                attack_pos_hide_weight=args.attack_pos_hide_weight,
                attack_neg_add_weight=args.attack_neg_add_weight,
                attack_negative_exclude_indices=attack_negative_exclude_indices,
                attack_neg_topk=args.attack_neg_topk,
            )
            gain_key = (
                "_attack_target_loss_gain_per_sample"
                if args.adv_accept_gain_mode == "target"
                else "_loss_gain_per_sample"
            )
            loss_gain_per_sample = np.asarray(
                attack_vs_anchor_stats.get(gain_key, []),
                dtype=np.float32,
            )
            # Semantic (Einthoven, HR, QRS)
            sem_info = compute_semantic_gate(
                adv_signals, anc_signals,
                fs=args.sampling_rate,
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
                    loss_gain_per_sample=loss_gain_per_sample,
                    min_loss_gain=args.adv_accept_min_loss_gain,
                    loss_gain_score_scale=args.adv_loss_gain_score_scale,
                    loss_gain_score_strength=args.adv_loss_gain_score_strength,
                )
                if args.enable_latent_augmix_branch:
                    accepted_indices = np.asarray(
                        push_stats.get("accepted_indices", []),
                        dtype=np.int64,
                    )
                    if accepted_indices.size == 0:
                        latent_augmix_signals = np.empty(
                            (0,) + tuple(adv_signals.shape[1:]),
                            dtype=np.float32,
                        )
                        latent_augmix_stats = {
                            "enabled": True,
                            "n_generated": 0,
                            "reason": "no_main_adv_samples_accepted",
                        }
                    else:
                        anchor_for_augmix = anc_signals[accepted_indices]
                        adv_for_augmix = adv_signals[accepted_indices]
                        labels_for_augmix = target_oh[accepted_indices]
                        latent_augmix_signals, latent_augmix_stats = build_latent_augmix_branch_signals(
                            anchor_signals_ct=anchor_for_augmix,
                            adv_signals_ct=adv_for_augmix,
                            copies=args.latent_augmix_copies,
                            severity=args.latent_augmix_severity,
                            width=args.latent_augmix_width,
                            depth=args.latent_augmix_depth,
                            alpha=args.latent_augmix_alpha,
                            latent_weight_cap=args.latent_augmix_latent_weight_cap,
                            ops=list(args.latent_augmix_ops),
                            rng=rng,
                            renorm=not args.no_latent_augmix_renorm,
                            clip_abs=args.latent_augmix_clip_abs,
                        )
                        latent_augmix_stats["accepted_source_count"] = int(accepted_indices.size)
                    if latent_augmix_signals.shape[0] > 0:
                        start = (latent_augmix_signals.shape[-1] - args.crop_len) // 2
                        latent_augmix_ct_crop = latent_augmix_signals[..., start:start + args.crop_len]
                        labels_rep = np.tile(
                            labels_for_augmix,
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
                                      drop_last=True,
                                      **dataloader_perf_kwargs(args))
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
                                      drop_last=True,
                                      **dataloader_perf_kwargs(args))

        # Phase D: train
        train_bce_loss = float("nan")
        train_rank_loss = float("nan")
        if args.rank_loss_weight > 0:
            train_parts = train_one_epoch_masked_bce_rank_aware(
                victim.model,
                train_loader,
                optimizer,
                criterion,
                args.device,
                grad_clip=args.grad_clip,
                trainable_params=trainable_params,
                ewa_params=ewa_params,
                anchor_lambda=args.anchor_lambda,
                ewa_decay=args.ewa_decay,
                rank_loss_weight=args.rank_loss_weight,
                rank_loss_margin=args.rank_loss_margin,
                rank_loss_positive_indices=rank_loss_positive_indices,
                freeze_backbone_eval_fn=freeze_backbone_eval_fn,
            )
            train_loss = train_parts["total"]
            train_bce_loss = train_parts["bce"]
            train_rank_loss = train_parts["rank"]
        elif freeze_backbone_eval_fn is not None:
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
        attack_vs_anchor_log = {
            key: value for key, value in attack_vs_anchor_stats.items()
            if not str(key).startswith("_")
        }
        entry = {
            "epoch": epoch,
            "attack_mode": args.attack_mode,
            "train_loss": round(train_loss, 4),
            "train_bce_loss": round(train_bce_loss, 4)
            if train_bce_loss == train_bce_loss else None,
            "train_rank_loss": round(train_rank_loss, 4)
            if train_rank_loss == train_rank_loss else None,
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_loss_margin": float(args.rank_loss_margin),
            "rank_loss_positive_classes": list(rank_loss_positive_classes),
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
            "atk_anchor": attack_vs_anchor_log.get("success_rate"),
            "attack_vs_anchor": attack_vs_anchor_log,
            "clean_bce": attack_vs_anchor_log.get("clean_bce_mean"),
            "adv_bce": attack_vs_anchor_log.get("adv_bce_mean"),
            "loss_gain": attack_vs_anchor_log.get("loss_gain_mean"),
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
            "multilabel_negative_label_asr": round(
                float(asr_info.get("multilabel_negative_label_asr", float("nan"))), 4
            ),
            "sample_any_negative_above_0p5_asr": round(
                float(asr_info.get("sample_any_negative_above_0p5_asr", float("nan"))), 4
            ),
            "per_class_negative_label_asr": {
                k: round(float(v), 4)
                for k, v in asr_info.get("per_class_negative_label_asr", {}).items()
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
            "adv_weight_effective": round(float(epoch_adv_weight), 6),
            "adv_weight_warmup_epochs": int(args.adv_weight_warmup_epochs),
            "adv_accept_min_loss_gain": args.adv_accept_min_loss_gain,
            "adv_accept_gain_mode": args.adv_accept_gain_mode,
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
                                   crop_len=args.crop_len,
                                   min_pos=args.quick_eval_min_pos)
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
