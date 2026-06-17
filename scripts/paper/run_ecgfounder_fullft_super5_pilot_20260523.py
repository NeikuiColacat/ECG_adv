#!/usr/bin/env python3
"""ECGFounder official-style full fine-tuning Super5 pilot.

This script tests whether ECGFounder's target-center gains come from the
previous frozen-encoder/head-only protocol being too weak. It follows the local
ECGFounder notebook's recommended mode (`linear_prob=False`): initialize the
12-lead checkpoint, replace the dense layer with a Super5 head, and fine-tune
the whole model on PTB-XL source plus K target-center real ECGs.

No ECGTwin VAE adversarial samples are used here. This is the fairness control
that must be understood before attributing ECGFounder gains to VAE-only LH-AT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
for _p in [str(REPO_ROOT), str(ECGFOUNDER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from physionet2021_dataset import EXPECTED_LEADS, TARGET_POINTS  # noqa: E402
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    CHECKPOINT,
    PTBXL_CSV,
    compute_metrics,
    preprocess_record,
    build_pn2021_items,
    build_ptbxl_items,
)
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    linear_warmup_value,
)
from ecg_adv_gen.adaptation.latent_hull_torch import initial_hull_latent  # noqa: E402
from ecg_adv_gen.adaptation.anchor_sampling import (  # noqa: E402
    parse_anchor_class_weight_string,
    signal_anchor_difficulty_scores,
    signal_anchor_positive_boundary_scores,
    weighted_class_quotas_with_caps,
    weighted_sample_indices as package_weighted_sample_indices,
)
from ecg_adv_gen.data import (  # noqa: E402
    build_signal_cache_metadata,
    load_signal_cache,
    load_real_anchor_pool_for_record_ids,
    load_selected_record_ids_from_meta,
    signal_cache_shape,
    write_signal_cache_metadata,
)
from ecg_adv_gen.evaluation import (  # noqa: E402
    compute_source_target_selection_score,
    split_target_train_val_indices,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    SUPER5_TO_IDX,
)
from ecg_adv_gen.labels import pn2021_super5_label_mapping_payload  # noqa: E402
from ecg_adv_gen.models.ecgfounder_heads import (  # noqa: E402
    OperatorConditionedLogitAdapter,
    init_dense_from_head_path,
)
from ecg_adv_gen.models.ecgfounder_inference import evaluate_signal_split  # noqa: E402
from ecg_adv_gen.models.ecgfounder_torch import (  # noqa: E402
    ecg1000_to_ecgfounder_input,
    global_zscore_torch,
    stabilize_ecg_torch,
)
from ecg_adv_gen.training import (  # noqa: E402
    CachedSignalDataset,
    MemorySignalDataset,
    build_weighted_signal_stream_loader_from_datasets,
    build_weighted_signal_stream_loader,
    compute_pos_weight,
    fullft_adv_batch_diagnostics,
    masked_bce_with_logits,
    set_module_requires_grad,
    stream_weighted_masked_bce,
    summarize_fullft_adv_epoch_diagnostics,
)
from ecg_adv_gen.run_naming import build_ecgfounder_fullft_run_leaf  # noqa: E402
from methods.augmix.jsd_loss import jsd_multilabel  # noqa: E402
from scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 import (  # noqa: E402
    RawSignalDataset,
    anchor_signal_npz_path,
    build_profiled_raw_corruption_views,
    load_source_raw_dataset,
)
from scripts.triple_labels.eval_pn2021_corruptions import (  # noqa: E402
    STRESS_PROFILE_CHOICES,
)
from scripts.triple_labels.model_zoo import build_super5_model, normalize_model_name  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_fullft_super5_20260523"
CENTER_DEFAULT = "cpsc_2018"
SUPERVISED_INPUT_MODES = ("cached5000", "raw1000", "source_cached_target_raw1000")
DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE = (
    DATA_ROOT / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
)
DEFAULT_SOURCE_RAW1000_LABEL_CACHE = (
    DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/ptbxl_labels.C5.all.npy"
)
RAW_CORRUPT_OP_CONDITION_NAMES = [
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
]
RAW_CORRUPT_TEACHER_TYPES = ("ecgfounder_fullft", "ecgfounder_feature_head", "efficientnet1dv2")
REAL_ROOTS = [
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v2",
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v1",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def input_stabilizer_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.ecgfounder_input_bandpass_low_hz is not None:
        kwargs["bandpass_low_hz"] = float(args.ecgfounder_input_bandpass_low_hz)
    if args.ecgfounder_input_bandpass_high_hz is not None:
        kwargs["bandpass_high_hz"] = float(args.ecgfounder_input_bandpass_high_hz)
    if bool(args.ecgfounder_input_repair_flat_leads):
        kwargs["repair_flat_leads"] = True
    if args.ecgfounder_input_clip_abs is not None:
        kwargs["clip_abs"] = float(args.ecgfounder_input_clip_abs)
    return kwargs


def prepare_raw_ecgfounder_input(
    x: torch.Tensor,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    return ecg1000_to_ecgfounder_input(x, **(input_stabilizer_kwargs or {}))


def prepare_cached_ecgfounder_input(
    x: torch.Tensor,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    kwargs = dict(input_stabilizer_kwargs or {})
    if not kwargs:
        return x
    y = stabilize_ecg_torch(
        x,
        sample_rate_hz=500.0,
        bandpass_low_hz=kwargs.get("bandpass_low_hz"),
        bandpass_high_hz=kwargs.get("bandpass_high_hz"),
        repair_flat_leads=bool(kwargs.get("repair_flat_leads", False)),
        clip_abs=kwargs.get("clip_abs"),
    )
    return global_zscore_torch(y)


def prepare_supervised_ecgfounder_input(
    x: torch.Tensor,
    supervised_input_mode: str,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    stream: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(supervised_input_mode)
    if mode == "raw1000":
        return prepare_raw_ecgfounder_input(x, input_stabilizer_kwargs)
    if mode == "cached5000":
        return prepare_cached_ecgfounder_input(x, input_stabilizer_kwargs)
    if mode == "source_cached_target_raw1000":
        if stream is None:
            raise ValueError("source_cached_target_raw1000 requires stream ids")
        if x.shape[-1] != TARGET_POINTS:
            raise ValueError(
                "source_cached_target_raw1000 supervised batches must be pre-aligned "
                f"to {TARGET_POINTS} points, got {tuple(x.shape)}"
            )
        stream = stream.to(device=x.device)
        source_mask = stream == 0
        if not bool(source_mask.any()):
            return x
        out = x.clone()
        out[source_mask] = prepare_cached_ecgfounder_input(
            x[source_mask],
            input_stabilizer_kwargs,
        )
        return out
    raise ValueError(f"unknown supervised_input_mode: {mode!r}")


def prepare_adv_stream_signal_for_supervised_mode(
    x_adv_1000: torch.Tensor,
    supervised_input_mode: str,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    mode = str(supervised_input_mode)
    if mode == "raw1000":
        return x_adv_1000
    if mode in {"cached5000", "source_cached_target_raw1000"}:
        return prepare_raw_ecgfounder_input(x_adv_1000, input_stabilizer_kwargs)
    raise ValueError(f"unknown supervised_input_mode: {mode!r}")


class ECGFounderFullFTVictim(nn.Module):
    """Differentiable ECGTwin-latent -> ECGFounder full model logits wrapper."""

    def __init__(
        self,
        model: nn.Module,
        ecgtwin: ECGTwinWrapper,
        input_stabilizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.ecgtwin = ecgtwin
        self.num_classes = len(CLASS_NAMES_SUPER5)
        self.input_stabilizer_kwargs = dict(input_stabilizer_kwargs or {})

    @staticmethod
    def _global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return global_zscore_torch(x, eps=eps)

    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent / 0.18215
        for module in self.ecgtwin.decoder:
            x = module(x)
        return x.transpose(1, 2)

    def _ecgtwin_latent_to_ecg1000(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_tc = self._decode_latent_differentiable(latent)
        ecg_ct = ecg_tc.transpose(-1, -2)
        ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
        ecg_ct = torch.clamp(ecg_ct, min=-3.0, max=3.0)
        ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True)
        return self._global_zscore(ecg_ct)

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_ct = self._ecgtwin_latent_to_ecg1000(latent)
        return self.model(prepare_raw_ecgfounder_input(ecg_ct, self.input_stabilizer_kwargs))


def build_signal_cache(
    items: list[dict[str, Any]],
    signal_path: Path,
    meta_path: Path,
    preprocess_policy: str,
) -> dict[str, np.ndarray]:
    if signal_path.exists() and meta_path.exists():
        return load_signal_cache(signal_path, meta_path)

    signal_path.parent.mkdir(parents=True, exist_ok=True)
    input_shape = (12, TARGET_POINTS)
    metadata = build_signal_cache_metadata(
        items,
        preprocess_policy=preprocess_policy,
        input_shape=input_shape,
        lead_order=EXPECTED_LEADS,
        num_classes=5,
    )
    arr = np.lib.format.open_memmap(
        signal_path,
        mode="w+",
        dtype=np.float32,
        shape=signal_cache_shape(len(items), input_shape),
    )
    for i, item in enumerate(tqdm(items, desc=f"cache {signal_path.name}")):
        arr[i] = preprocess_record(item["path"], preprocess_policy=preprocess_policy)
    arr.flush()
    write_signal_cache_metadata(meta_path, metadata)
    return load_signal_cache(signal_path, meta_path)


def load_selected_ref_ids(ref_meta_json: Path, center: str) -> set[str]:
    return load_selected_record_ids_from_meta(ref_meta_json, center, strict_center=True)


def init_dense_from_head(model: nn.Module, head_path: Path) -> dict[str, Any]:
    return init_dense_from_head_path(model, head_path)


def split_target_train_val(
    target_idx: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    val_count: int,
    split_seed: int,
    split_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    return split_target_train_val_indices(
        target_idx,
        record_ids,
        labels,
        val_count,
        split_seed,
        split_mode,
    )


def load_anchor_pool_for_ids(
    center: str,
    selected_ids: set[str],
    pn_payload: dict[str, np.ndarray],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    return load_real_anchor_pool_for_record_ids(
        center,
        selected_ids=selected_ids,
        pn_payload=pn_payload,
        anchor_base_root=getattr(args, "anchor_base_root", "") if args is not None else None,
        k=int(getattr(args, "k", 500)) if args is not None else 500,
        seed=int(getattr(args, "seed", 42)) if args is not None else 42,
        default_roots=REAL_ROOTS,
        class_names=CLASS_NAMES_SUPER5,
        include_signals=True,
    )


def load_ptbxl_partner_pool(
    ptbxl_payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    cache_path = Path(args.ptbxl_vae_cache)
    rows = torch.load(cache_path, map_location="cpu")
    if not isinstance(rows, list):
        raise RuntimeError(f"{cache_path} should contain a list, got {type(rows)!r}")
    labels = ptbxl_payload["labels"].astype(np.float32)
    folds = ptbxl_payload["folds"].astype(np.int64)
    if len(rows) != len(labels):
        raise RuntimeError(
            f"PTB-XL VAE cache length {len(rows)} does not match PTB-XL labels {len(labels)}"
        )
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    if args.source_partner_limit_per_class > 0:
        rng = np.random.default_rng(args.seed + 90210)
        selected: set[int] = set()
        for class_i in range(labels.shape[1]):
            cls_idx = source_idx[labels[source_idx, class_i] > 0.5]
            if len(cls_idx) == 0:
                continue
            take = min(int(args.source_partner_limit_per_class), len(cls_idx))
            selected.update(int(i) for i in rng.choice(cls_idx, size=take, replace=False))
        if selected:
            source_idx = np.asarray(sorted(selected), dtype=np.int64)
    latents = np.stack([
        rows[int(i)]["data"].detach().cpu().numpy().astype(np.float32)
        for i in source_idx
    ]).astype(np.float32)
    return {
        "latents": latents,
        "labels": labels[source_idx].astype(np.float32),
        "indices": source_idx.astype(np.int64),
    }


def load_source_raw1000_dataset(
    args: argparse.Namespace,
    source_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> RawSignalDataset:
    signal_path = Path(args.source_raw1000_signal_cache)
    label_path = Path(args.source_raw1000_label_cache)
    if not signal_path.exists():
        raise FileNotFoundError(signal_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    signals = np.load(signal_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r").astype(np.float32, copy=False)
    if labels.shape[0] != signals.shape[0]:
        raise ValueError(f"source raw1000 signals/labels length mismatch: {signals.shape} vs {labels.shape}")
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if expected_labels.shape[0] != source_indices.shape[0]:
        raise ValueError(
            f"expected source label rows mismatch: {expected_labels.shape[0]} vs {source_indices.shape[0]}"
        )
    if not np.allclose(labels[source_indices], expected_labels.astype(np.float32), atol=1e-6):
        raise ValueError("source raw1000 labels do not align with ECGFounder supervised cache labels")
    return RawSignalDataset(signals, labels, indices=source_indices)


def load_target_raw_dataset(
    center: str,
    args: argparse.Namespace,
    train_record_ids: list[str],
) -> RawSignalDataset:
    signal_path = anchor_signal_npz_path(center, args)
    with np.load(signal_path, allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        record_ids = (
            data["record_ids"].astype(str)
            if "record_ids" in data.files
            else np.asarray([str(i) for i in range(labels.shape[0])])
        )
    train_ids = set(str(x) for x in train_record_ids)
    indices = np.asarray([i for i, rid in enumerate(record_ids.astype(str)) if str(rid) in train_ids], dtype=np.int64)
    if indices.size == 0:
        raise RuntimeError(f"{center}: no target raw signals matched target train record ids")
    return RawSignalDataset(signals, labels, indices=indices)


def prepare_raw_dataset_as_ecgfounder_dataset(
    dataset: Dataset,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    *,
    batch_size: int = 256,
) -> MemorySignalDataset:
    """Materialize a small raw1000 dataset as ECGFounder 5000-point tensors."""

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    signal_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            prepared = prepare_raw_ecgfounder_input(
                x.float(),
                input_stabilizer_kwargs,
            )
            signal_chunks.append(prepared.detach().cpu().numpy().astype(np.float32, copy=False))
            label_chunks.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
    if not signal_chunks:
        raise RuntimeError("cannot prepare empty raw1000 dataset")
    return MemorySignalDataset(
        np.concatenate(signal_chunks, axis=0).astype(np.float32, copy=False),
        np.concatenate(label_chunks, axis=0).astype(np.float32, copy=False),
    )


def prepare_adv_signals_for_hybrid_loader(
    adv_signals: np.ndarray | None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> np.ndarray | None:
    if adv_signals is None or len(adv_signals) == 0:
        return adv_signals
    arr = np.asarray(adv_signals, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 12:
        raise ValueError(f"adv_signals must have shape (N, 12, L), got {arr.shape}")
    if arr.shape[-1] == TARGET_POINTS:
        return arr
    if arr.shape[-1] != 1000:
        raise ValueError(f"hybrid adv_signals must have length 1000 or {TARGET_POINTS}, got {arr.shape}")
    with torch.no_grad():
        prepared = prepare_raw_ecgfounder_input(
            torch.from_numpy(arr).float(),
            input_stabilizer_kwargs,
        )
    return prepared.detach().cpu().numpy().astype(np.float32, copy=False)


def parse_class_weight_string(raw: str) -> dict[str, float]:
    return parse_anchor_class_weight_string(raw)


def weighted_class_quotas(
    classes: list[str],
    k_total: int,
    class_weights: dict[str, float],
    labels: np.ndarray,
    max_repeat_per_class: int = 0,
) -> dict[str, int]:
    return weighted_class_quotas_with_caps(
        classes,
        k_total,
        class_weights,
        labels,
        max_repeat_per_class=max_repeat_per_class,
        class_to_idx=SUPER5_TO_IDX,
    )


def weighted_sample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    replace_when_needed: bool,
) -> np.ndarray:
    return package_weighted_sample_indices(
        rng,
        indices,
        weights,
        k,
        replace_when_needed=replace_when_needed,
    )


def sample_anchor_indices(
    model: nn.Module,
    anchor_pool: dict[str, Any],
    walker: StratifiedPoolWalker,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(args.anchor_sample_mode)
    classes = list(anchor_pool["classes_in_scope"])
    labels = np.asarray(anchor_pool["labels"], dtype=np.float32)
    class_weights = parse_class_weight_string(args.anchor_class_sample_weights)
    rng = np.random.default_rng(args.seed + int(args.current_epoch) * 1009 + 97)
    quotas = weighted_class_quotas(
        classes,
        int(args.k_anchor),
        class_weights,
        labels,
        int(args.anchor_class_max_repeat),
    )

    difficulty_mode = None
    if mode in {"hard_bce", "stratified_hard_bce"}:
        difficulty_mode = "hard_bce"
    elif mode in {"uncertainty", "stratified_uncertainty"}:
        difficulty_mode = "uncertainty"

    scores = None
    sample_weights = None
    if difficulty_mode is not None:
        scores = signal_anchor_difficulty_scores(
            model,
            anchor_pool["signals"],
            labels,
            pos_weight,
            batch_size=max(1, int(args.eval_batch_size)),
            device=device,
            mode=difficulty_mode,
        )
        centered = scores - float(np.min(scores)) if scores.size else scores
        sample_weights = centered + float(args.anchor_sample_min_weight)
        if float(args.anchor_sample_power) != 1.0:
            sample_weights = np.power(sample_weights, float(args.anchor_sample_power))

    class_boundary_scores = None
    if mode == "stratified_pos_boundary":
        difficulty_mode = "pos_boundary"
        class_boundary_scores = signal_anchor_positive_boundary_scores(
            model,
            anchor_pool["signals"],
            labels,
            batch_size=max(1, int(args.eval_batch_size)),
            device=device,
        )
        scores = class_boundary_scores.max(axis=1) if class_boundary_scores.size else np.empty(0, dtype=np.float32)

    if mode == "stratified":
        drawn = walker.sample(quotas)
        pieces = [drawn[c] for c in classes if drawn.get(c, np.empty(0, dtype=np.int64)).size > 0]
        picks = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        if picks.size > 1:
            rng.shuffle(picks)
    elif mode in {"hard_bce", "uncertainty"}:
        assert sample_weights is not None and scores is not None
        class_cols = [SUPER5_TO_IDX[c] for c in classes]
        eligible = np.nonzero((labels[:, class_cols] > 0.5).any(axis=1))[0]
        picks = weighted_sample_indices(
            rng,
            eligible,
            sample_weights[eligible],
            int(args.k_anchor),
            replace_when_needed=True,
        )
    elif mode in {"stratified_hard_bce", "stratified_uncertainty"}:
        assert sample_weights is not None and scores is not None
        pieces = []
        for cls in classes:
            cls_i = SUPER5_TO_IDX[cls]
            cls_idx = np.nonzero(labels[:, cls_i] > 0.5)[0]
            pieces.append(
                weighted_sample_indices(
                    rng,
                    cls_idx,
                    sample_weights[cls_idx],
                    int(quotas.get(cls, 0)),
                    replace_when_needed=True,
                )
            )
        picks = np.concatenate([p for p in pieces if p.size > 0]) if pieces else np.empty(0, dtype=np.int64)
        if picks.size > 1:
            rng.shuffle(picks)
    elif mode == "stratified_pos_boundary":
        assert class_boundary_scores is not None
        pieces = []
        for cls in classes:
            cls_i = SUPER5_TO_IDX[cls]
            cls_idx = np.nonzero(labels[:, cls_i] > 0.5)[0]
            cls_weights = class_boundary_scores[cls_idx, cls_i] + float(args.anchor_sample_min_weight)
            if float(args.anchor_sample_power) != 1.0:
                cls_weights = np.power(cls_weights, float(args.anchor_sample_power))
            pieces.append(
                weighted_sample_indices(
                    rng,
                    cls_idx,
                    cls_weights,
                    int(quotas.get(cls, 0)),
                    replace_when_needed=True,
                )
            )
        picks = np.concatenate([p for p in pieces if p.size > 0]) if pieces else np.empty(0, dtype=np.int64)
        if picks.size > 1:
            rng.shuffle(picks)
    else:
        raise ValueError(f"unknown --anchor_sample_mode={mode!r}")

    picked_scores = scores[picks] if scores is not None and len(picks) > 0 else np.empty(0, dtype=np.float32)
    picked_weights = (
        sample_weights[picks] if sample_weights is not None and len(picks) > 0 else np.empty(0, dtype=np.float32)
    )
    ess = None
    if picked_weights.size > 0 and float(np.square(picked_weights).sum()) > 0.0:
        ess = float(np.square(picked_weights.sum()) / np.square(picked_weights).sum())
    stats = {
        "anchor_sample_mode": mode,
        "anchor_difficulty_mode": difficulty_mode,
        "anchor_class_sample_weights": class_weights,
        "anchor_class_max_repeat": int(args.anchor_class_max_repeat),
        "anchor_k_per_class": quotas,
        "anchor_picked_unique": int(len(np.unique(picks))) if len(picks) > 0 else 0,
        "anchor_picked_class_counts": {
            cls: int((labels[picks, SUPER5_TO_IDX[cls]] > 0.5).sum()) if len(picks) > 0 else 0
            for cls in classes
        },
        "anchor_score_mean": float(np.mean(picked_scores)) if picked_scores.size > 0 else None,
        "anchor_score_p90": float(np.percentile(picked_scores, 90)) if picked_scores.size > 0 else None,
        "anchor_score_max": float(np.max(picked_scores)) if picked_scores.size > 0 else None,
        "anchor_sample_ess": ess,
    }
    return picks.astype(np.int64), stats


def eval_split(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> dict:
    eval_model: nn.Module = model
    if input_stabilizer_kwargs:
        class _StabilizedModel(nn.Module):
            def __init__(self, base_model: nn.Module, kwargs: dict[str, Any]) -> None:
                super().__init__()
                self.base_model = base_model
                self.kwargs = dict(kwargs)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.base_model(prepare_cached_ecgfounder_input(x, self.kwargs))

        eval_model = _StabilizedModel(model, input_stabilizer_kwargs)
    return evaluate_signal_split(
        eval_model,
        signals,
        labels,
        indices,
        compute_metrics,
        batch_size=batch_size,
        device=device,
        min_pos=1,
    )


def make_train_loader(
    ptbxl_payload: dict[str, np.ndarray],
    pn_payload: dict[str, np.ndarray],
    target_indices: np.ndarray,
    args: argparse.Namespace,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_weight: float | None = None,
    target_train_record_ids: set[str] | None = None,
) -> DataLoader:
    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    if args.source_train_limit > 0:
        source_idx = source_idx[: args.source_train_limit]
    effective_adv_weight = float(args.adv_weight if adv_weight is None else adv_weight)
    if str(args.supervised_input_mode) == "source_cached_target_raw1000":
        if target_train_record_ids is None:
            raise ValueError("source_cached_target_raw1000 supervised input mode requires target_train_record_ids")
        source_ds = CachedSignalDataset(
            ptbxl_payload["signals"],
            ptbxl_payload["labels"],
            source_idx,
        )
        target_raw_ds = load_target_raw_dataset(
            args.center,
            args,
            sorted(str(x) for x in target_train_record_ids),
        )
        target_ds = prepare_raw_dataset_as_ecgfounder_dataset(
            target_raw_ds,
            input_stabilizer_kwargs_from_args(args),
            batch_size=int(args.eval_batch_size),
        )
        adv_signals_for_loader = prepare_adv_signals_for_hybrid_loader(
            adv_signals,
            input_stabilizer_kwargs_from_args(args),
        )
        return build_weighted_signal_stream_loader_from_datasets(
            source_dataset=source_ds,
            target_dataset=target_ds,
            source_weight=float(args.source_weight),
            target_real_weight=float(args.target_real_weight),
            adv_weight=effective_adv_weight,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_classes=len(CLASS_NAMES_SUPER5),
            adv_signals=adv_signals_for_loader,
            adv_labels=adv_labels,
            adv_teacher_logits=adv_teacher_logits,
        )
    if str(args.supervised_input_mode) == "raw1000":
        if target_train_record_ids is None:
            raise ValueError("raw1000 supervised input mode requires target_train_record_ids")
        source_ds = load_source_raw1000_dataset(
            args,
            source_idx,
            ptbxl_payload["labels"][source_idx],
        )
        target_ds = load_target_raw_dataset(
            args.center,
            args,
            sorted(str(x) for x in target_train_record_ids),
        )
        return build_weighted_signal_stream_loader_from_datasets(
            source_dataset=source_ds,
            target_dataset=target_ds,
            source_weight=float(args.source_weight),
            target_real_weight=float(args.target_real_weight),
            adv_weight=effective_adv_weight,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_classes=len(CLASS_NAMES_SUPER5),
            adv_signals=adv_signals,
            adv_labels=adv_labels,
            adv_teacher_logits=adv_teacher_logits,
        )
    return build_weighted_signal_stream_loader(
        source_signals=ptbxl_payload["signals"],
        source_labels=ptbxl_payload["labels"],
        source_indices=source_idx,
        target_signals=pn_payload["signals"],
        target_labels=pn_payload["labels"],
        target_indices=target_indices,
        source_weight=float(args.source_weight),
        target_real_weight=float(args.target_real_weight),
        adv_weight=effective_adv_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=len(CLASS_NAMES_SUPER5),
        adv_signals=adv_signals,
        adv_labels=adv_labels,
        adv_teacher_logits=adv_teacher_logits,
    )


def scheduled_adv_weight(args: argparse.Namespace, epoch: int) -> float:
    return linear_warmup_value(
        args.adv_weight,
        epoch,
        args.adv_weight_warmup_epochs,
        start=args.adv_weight_start,
    )


def build_adv_epoch(
    model: nn.Module,
    victim: ECGFounderFullFTVictim,
    pgd_gen: LatentHullPGDGenerator,
    anchor_pool: dict[str, Any],
    walker: StratifiedPoolWalker,
    index: SameLabelLatentIndex,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_module_requires_grad(model, False)
    model.eval()
    victim.eval()
    adv_signals: list[np.ndarray] = []
    adv_labels: list[np.ndarray] = []
    adv_teacher_logits: list[np.ndarray] = []
    delta_norms: list[float] = []
    batch_diagnostics: list[dict[str, np.ndarray | int]] = []
    anchor_sample_stats: dict[str, Any] = {}
    try:
        picks, anchor_sample_stats = sample_anchor_indices(
            model,
            anchor_pool,
            walker,
            pos_weight,
            args,
            device,
        )
        for start in range(0, len(picks), args.pgd_batch):
            batch_idx = picks[start:start + args.pgd_batch]
            z = torch.from_numpy(anchor_pool["latents"][batch_idx]).float().to(device)
            y = torch.from_numpy(anchor_pool["labels"][batch_idx]).float().to(device)
            cand = torch.from_numpy(index.candidates_for(batch_idx, args.hull_m)).float().to(device)
            with torch.no_grad():
                clean_logits = victim.forward_from_latent_to_logits(z)
                z_init = initial_hull_latent(
                    z,
                    cand,
                    weight_mode=args.hull_weight_mode,
                    hull_lambda=args.hull_lambda,
                    init_logit_gap=pgd_gen.init_logit_gap,
                )
                init_logits = victim.forward_from_latent_to_logits(z_init)
            x_adv_1000, delta = pgd_gen.attack_from_latent(z, y, candidate_latents=cand)
            with torch.no_grad():
                adv_input = prepare_raw_ecgfounder_input(x_adv_1000, victim.input_stabilizer_kwargs)
                adv_logits = victim.model(adv_input)
                batch_diagnostics.append(
                    fullft_adv_batch_diagnostics(clean_logits, init_logits, adv_logits, y)
                )
            x_adv_for_stream = (
                prepare_adv_stream_signal_for_supervised_mode(
                    x_adv_1000,
                    args.supervised_input_mode,
                    victim.input_stabilizer_kwargs,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            adv_signals.append(x_adv_for_stream)
            adv_labels.append(anchor_pool["labels"][batch_idx].astype(np.float32))
            adv_teacher_logits.append(clean_logits.detach().cpu().numpy().astype(np.float32))
            delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())
    finally:
        set_module_requires_grad(model, True)
    signals = (
        np.concatenate(adv_signals, axis=0).astype(np.float32)
        if adv_signals
        else np.empty(
            (0, 12, 1000 if str(args.supervised_input_mode) == "raw1000" else TARGET_POINTS),
            dtype=np.float32,
        )
    )
    labels = (
        np.concatenate(adv_labels, axis=0).astype(np.float32)
        if adv_labels
        else np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    )
    teacher_logits = (
        np.concatenate(adv_teacher_logits, axis=0).astype(np.float32)
        if adv_teacher_logits
        else np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    )
    adv_stats = summarize_fullft_adv_epoch_diagnostics(
        batch_diagnostics,
        delta_norms=delta_norms,
        n_adv=int(len(signals)),
        anchor_sample_stats=anchor_sample_stats,
    )
    return {
        "signals": signals,
        "labels": labels,
        "teacher_logits": teacher_logits,
        **adv_stats,
    }


def forward_logits_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ECGFounder with its penultimate feature output enabled."""

    if not hasattr(model, "return_features"):
        raise RuntimeError("raw feature consistency requires model.return_features support")
    previous = bool(getattr(model, "return_features"))
    try:
        setattr(model, "return_features", True)
        output = model(x)
    finally:
        setattr(model, "return_features", previous)
    if not (isinstance(output, tuple) and len(output) == 2):
        raise RuntimeError("raw feature consistency expected model(x) to return (logits, features)")
    logits, features = output
    return logits, features


def train_fullft_raw_corruption_consistency_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    copies: int,
    severity: int,
    severity_profile: str,
    ops: list[str],
    prob: float,
    consistency_weight: float,
    bce_weight: float,
    consistency_loss: str,
    rng: np.random.Generator,
    trainable_params: list[nn.Parameter],
    grad_clip: float,
    max_batches: int,
    renorm: bool,
    clip_abs: float,
    view_mode: str = "single_op",
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    augmix_mixture_mode: str = "beta",
    augmix_mixture_prob: float = 0.5,
    augmix_mixture_beta_a: float = 0.0,
    augmix_mixture_beta_b: float = 0.0,
    teacher_model: nn.Module | None = None,
    teacher_weight: float = 0.0,
    teacher_loss: str = "soft_bce",
    teacher_view: str = "corrupt",
    feature_consistency_weight: float = 0.0,
    feature_consistency_normalize: bool = False,
    op_adapter: OperatorConditionedLogitAdapter | None = None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the full ECGFounder model on profiled raw corruption views."""

    teacher_weight = float(teacher_weight)
    teacher_enabled = bool(teacher_model is not None and teacher_weight > 0)
    feature_consistency_weight = float(feature_consistency_weight)
    feature_enabled = bool(feature_consistency_weight > 0)
    op_conditioning_enabled = op_adapter is not None
    input_stabilizer_kwargs = dict(input_stabilizer_kwargs or {})
    if copies <= 0 or (
        consistency_weight <= 0
        and bce_weight <= 0
        and not teacher_enabled
        and not feature_enabled
    ):
        return {
            "enabled": False,
            "reason": "disabled_or_zero_weight",
            "loss": float("nan"),
            "bce_loss": float("nan"),
            "consistency_loss": float("nan"),
            "teacher_loss": float("nan"),
            "feature_loss": float("nan"),
            "n_batches": 0,
            "n_generated": 0,
            "n_corrupted": 0,
            "op_counts": {},
        }
    if consistency_loss not in {"soft_bce", "jsd"}:
        raise ValueError(f"unknown raw corruption consistency loss: {consistency_loss}")
    if consistency_loss == "jsd" and copies < 2:
        raise ValueError("--raw_corrupt_consistency_loss jsd requires --raw_corrupt_copies >= 2")
    if teacher_enabled and teacher_loss not in {"soft_bce", "mse_logits"}:
        raise ValueError(f"unknown raw corruption teacher loss: {teacher_loss}")
    if teacher_enabled and teacher_view not in {"clean", "corrupt"}:
        raise ValueError(f"unknown raw corruption teacher view: {teacher_view}")
    if str(severity_profile) not in STRESS_PROFILE_CHOICES:
        raise ValueError(f"unknown raw corruption severity profile: {severity_profile}")

    model.train()
    if op_adapter is not None:
        op_adapter.train()
    if teacher_model is not None:
        teacher_model.eval()
    losses: list[float] = []
    bce_losses: list[float] = []
    consistency_losses: list[float] = []
    teacher_losses: list[float] = []
    feature_losses: list[float] = []
    n_generated = 0
    n_corrupted = 0
    op_conditioned_views = 0
    op_counts: dict[str, int] = {}

    for batch_i, (signals, labels) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()
        if signals.shape[-1] != 1000:
            signals = F.interpolate(signals, size=1000, mode="linear", align_corners=True)

        clean_np = signals.detach().cpu().numpy().astype(np.float32, copy=False)
        corrupt_np, stats = build_profiled_raw_corruption_views(
            clean_np,
            copies=copies,
            severity=severity,
            severity_profile=severity_profile,
            ops=ops,
            prob=prob,
            rng=rng,
            renorm=renorm,
            clip_abs=clip_abs,
            view_mode=view_mode,
            augmix_width=augmix_width,
            augmix_depth=augmix_depth,
            augmix_alpha=augmix_alpha,
            augmix_mixture_mode=augmix_mixture_mode,
            augmix_mixture_prob=augmix_mixture_prob,
            augmix_mixture_beta_a=augmix_mixture_beta_a,
            augmix_mixture_beta_b=augmix_mixture_beta_b,
        )
        if corrupt_np.shape[0] == 0:
            continue

        corrupt = torch.from_numpy(corrupt_np).float().to(device, non_blocking=True)
        labels_rep = labels.repeat((int(copies), 1))
        optimizer.zero_grad(set_to_none=True)
        op_ids: torch.Tensor | None = None
        if op_conditioning_enabled:
            assert op_adapter is not None
            view_ops = stats.get("view_ops")
            if not isinstance(view_ops, list) or len(view_ops) != int(corrupt_np.shape[0]):
                raise RuntimeError(
                    "raw corruption op conditioning requires stats['view_ops'] "
                    f"for every generated view, got {type(view_ops).__name__}"
                )
            op_ids = op_adapter.op_ids_from_names(view_ops, device=device)
            op_conditioned_views += int(op_ids.numel())

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            clean_input = prepare_raw_ecgfounder_input(signals, input_stabilizer_kwargs)
            corrupt_input = prepare_raw_ecgfounder_input(corrupt, input_stabilizer_kwargs)
            if feature_enabled or op_conditioning_enabled:
                clean_logits, clean_features = forward_logits_features(model, clean_input)
                corrupt_logits, corrupt_features = forward_logits_features(model, corrupt_input)
            else:
                clean_logits = model(clean_input)
                corrupt_logits = model(corrupt_input)
                clean_features = None
                corrupt_features = None
            if op_conditioning_enabled:
                assert op_adapter is not None and op_ids is not None and corrupt_features is not None
                corrupt_logits = corrupt_logits + op_adapter(corrupt_features, op_ids)
            if not torch.isfinite(clean_logits).all() or not torch.isfinite(corrupt_logits).all():
                raise RuntimeError(
                    "non-finite raw corruption logits "
                    f"at batch {batch_i}: "
                    f"clean_finite={bool(torch.isfinite(clean_logits).all())} "
                    f"corrupt_finite={bool(torch.isfinite(corrupt_logits).all())}"
                )
            hard_bce = 0.5 * (
                masked_bce_with_logits(clean_logits, labels, pos_weight)
                + masked_bce_with_logits(corrupt_logits, labels_rep, pos_weight)
            )
            if consistency_loss == "soft_bce":
                soft_targets = torch.sigmoid(clean_logits.detach()).repeat((int(copies), 1))
                mask = (labels_rep >= 0).float()
                raw_consistency = (
                    F.binary_cross_entropy_with_logits(
                        corrupt_logits,
                        soft_targets,
                        reduction="none",
                    )
                    * mask
                ).sum() / mask.sum().clamp(min=1.0)
            else:
                logits_views = corrupt_logits.view(int(copies), signals.shape[0], -1)
                jsd_terms = []
                for copy_i in range(int(copies)):
                    copy_j = (copy_i + 1) % int(copies)
                    jsd_terms.append(
                        jsd_multilabel(clean_logits, logits_views[copy_i], logits_views[copy_j])
                    )
                raw_consistency = torch.stack(jsd_terms).mean()
            if teacher_enabled:
                assert teacher_model is not None
                with torch.no_grad():
                    if teacher_view == "clean":
                        teacher_logits = teacher_model(
                            prepare_raw_corruption_teacher_input(
                                teacher_model,
                                signals,
                                input_stabilizer_kwargs,
                            )
                        )
                        teacher_logits = teacher_logits.repeat((int(copies), 1))
                    else:
                        teacher_logits = teacher_model(
                            prepare_raw_corruption_teacher_input(
                                teacher_model,
                                corrupt,
                                input_stabilizer_kwargs,
                            )
                        )
                if not torch.isfinite(teacher_logits).all():
                    raise RuntimeError(
                        "non-finite raw corruption teacher logits "
                        f"at batch {batch_i}: "
                        f"teacher_finite={bool(torch.isfinite(teacher_logits).all())}"
                    )
                if teacher_loss == "soft_bce":
                    teacher_targets = torch.sigmoid(teacher_logits.detach())
                    mask = (labels_rep >= 0).float()
                    raw_teacher_loss = (
                        F.binary_cross_entropy_with_logits(
                            corrupt_logits,
                            teacher_targets,
                            reduction="none",
                        )
                        * mask
                    ).sum() / mask.sum().clamp(min=1.0)
                else:
                    raw_teacher_loss = F.mse_loss(corrupt_logits, teacher_logits.detach())
            else:
                raw_teacher_loss = corrupt_logits.new_zeros(())
            if feature_enabled:
                assert clean_features is not None and corrupt_features is not None
                if not torch.isfinite(clean_features).all() or not torch.isfinite(corrupt_features).all():
                    raise RuntimeError(
                        "non-finite raw corruption features "
                        f"at batch {batch_i}: "
                        f"clean_finite={bool(torch.isfinite(clean_features).all())} "
                        f"corrupt_finite={bool(torch.isfinite(corrupt_features).all())}"
                    )
                clean_feature_targets = clean_features.detach().repeat((int(copies), 1))
                corrupt_feature_values = corrupt_features
                if feature_consistency_normalize:
                    clean_feature_targets = F.normalize(clean_feature_targets.float(), p=2, dim=1)
                    corrupt_feature_values = F.normalize(corrupt_feature_values.float(), p=2, dim=1)
                raw_feature_loss = F.mse_loss(corrupt_feature_values, clean_feature_targets)
            else:
                raw_feature_loss = corrupt_logits.new_zeros(())
            loss = (
                float(bce_weight) * hard_bce
                + float(consistency_weight) * raw_consistency
                + teacher_weight * raw_teacher_loss
                + feature_consistency_weight * raw_feature_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite raw corruption loss "
                    f"at batch {batch_i}: "
                    f"loss={float(loss.detach().cpu())} "
                    f"bce={float(hard_bce.detach().cpu())} "
                    f"consistency={float(raw_consistency.detach().cpu())} "
                    f"teacher={float(raw_teacher_loss.detach().cpu())}"
                    f"feature={float(raw_feature_loss.detach().cpu())}"
                )

        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(
                    "non-finite raw corruption gradient "
                    f"at batch {batch_i}: parameter={name}"
                )
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, float(grad_clip))
        optimizer.step()
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    "non-finite raw corruption parameter "
                    f"after optimizer step at batch {batch_i}: parameter={name}"
                )

        losses.append(float(loss.item()))
        bce_losses.append(float(hard_bce.item()))
        consistency_losses.append(float(raw_consistency.item()))
        if teacher_enabled:
            teacher_losses.append(float(raw_teacher_loss.item()))
        if feature_enabled:
            feature_losses.append(float(raw_feature_loss.item()))
        n_generated += int(stats.get("n_generated", 0))
        n_corrupted += int(stats.get("n_corrupted", 0))
        for op_name, count in dict(stats.get("op_counts", {})).items():
            op_counts[str(op_name)] = op_counts.get(str(op_name), 0) + int(count)
        if max_batches > 0 and batch_i >= max_batches:
            break

    return {
        "enabled": True,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else float("nan"),
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else float("nan"),
        "teacher_loss": float(np.mean(teacher_losses)) if teacher_losses else float("nan"),
        "feature_loss": float(np.mean(feature_losses)) if feature_losses else float("nan"),
        "n_batches": int(len(losses)),
        "n_generated": int(n_generated),
        "n_corrupted": int(n_corrupted),
        "corrupt_fraction": float(n_corrupted / max(1, n_generated)),
        "copies": int(copies),
        "severity": int(severity),
        "severity_profile": str(severity_profile),
        "prob": float(prob),
        "view_mode": str(view_mode),
        "consistency_weight": float(consistency_weight),
        "consistency_objective": str(consistency_loss),
        "bce_weight": float(bce_weight),
        "teacher_enabled": bool(teacher_enabled),
        "teacher_weight": float(teacher_weight),
        "teacher_objective": str(teacher_loss) if teacher_enabled else None,
        "teacher_view": str(teacher_view) if teacher_enabled else None,
        "feature_consistency_enabled": bool(feature_enabled),
        "feature_consistency_weight": float(feature_consistency_weight),
        "feature_consistency_normalize": bool(feature_consistency_normalize),
        "op_conditioning_enabled": bool(op_conditioning_enabled),
        "op_conditioned_views": int(op_conditioned_views),
        "op_condition_names": (
            list(op_adapter.op_names) if op_adapter is not None else []
        ),
        "input_stabilizer": dict(input_stabilizer_kwargs),
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


def _finite_loss_or_none(stats: dict[str, Any]) -> float | None:
    value = float(stats.get("loss", float("nan")))
    if not np.isfinite(value):
        return None
    return value


def prepare_raw_corruption_teacher_input(
    teacher_model: nn.Module,
    x: torch.Tensor,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Return the input tensor expected by a raw-corruption teacher."""

    mode = str(getattr(teacher_model, "raw_corruption_input_mode", "ecgfounder"))
    if mode == "raw1000":
        return x
    if mode == "ecgfounder":
        return prepare_raw_ecgfounder_input(x, input_stabilizer_kwargs)
    raise ValueError(f"unknown raw corruption teacher input mode: {mode}")


def _load_teacher_state_dict(model_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"teacher checkpoint does not contain a state_dict: {model_path}")
    return {str(k).removeprefix("_orig_mod."): v for k, v in payload.items()}


def load_frozen_fullft_teacher(model_path: Path, device: torch.device) -> nn.Module:
    """Load a frozen ECGFounder fullFT model for corrupted-view distillation."""

    teacher = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    state_dict = _load_teacher_state_dict(model_path, device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def load_frozen_feature_head_teacher(head_path: Path, device: torch.device) -> nn.Module:
    """Load a frozen ECGFounder encoder with a linear feature-head checkpoint."""

    teacher = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    head_state = torch.load(head_path, map_location=device)
    if not isinstance(head_state, dict) or "weight" not in head_state or "bias" not in head_state:
        raise ValueError(f"feature-head teacher checkpoint must contain weight/bias: {head_path}")
    dense = getattr(teacher, "dense", None)
    if not isinstance(dense, nn.Linear):
        raise ValueError(f"ECGFounder teacher dense head is not linear: {type(dense).__name__}")
    weight = head_state["weight"].detach().to(device=device, dtype=dense.weight.dtype)
    bias = head_state["bias"].detach().to(device=device, dtype=dense.bias.dtype)
    if tuple(weight.shape) != tuple(dense.weight.shape) or tuple(bias.shape) != tuple(dense.bias.shape):
        raise ValueError(
            f"feature-head teacher shape mismatch: got weight={tuple(weight.shape)} "
            f"bias={tuple(bias.shape)}, expected weight={tuple(dense.weight.shape)} "
            f"bias={tuple(dense.bias.shape)}"
        )
    with torch.no_grad():
        dense.weight.copy_(weight)
        dense.bias.copy_(bias)
    teacher.raw_corruption_input_mode = "ecgfounder"  # type: ignore[attr-defined]
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


class FrozenEfficientNetRawTeacher(nn.Module):
    """Frozen Super5 teacher that consumes raw 100Hz `(B, 12, 1000)` ECG."""

    raw_corruption_input_mode = "raw1000"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_frozen_effnet_teacher(
    model_path: Path,
    device: torch.device,
    model_name: str = "efficientnet1dv2",
) -> nn.Module:
    """Load a frozen EfficientNet-style Super5 model for raw-view distillation."""

    base = build_super5_model(
        normalize_model_name(str(model_name)),
        num_classes=len(CLASS_NAMES_SUPER5),
    ).to(device)
    base.load_state_dict(_load_teacher_state_dict(model_path, device))
    teacher = FrozenEfficientNetRawTeacher(base)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def load_frozen_raw_corruption_teacher(
    model_path: Path,
    device: torch.device,
    *,
    teacher_type: str,
    teacher_model_name: str = "efficientnet1dv2",
) -> nn.Module:
    if teacher_type == "ecgfounder_fullft":
        return load_frozen_fullft_teacher(model_path, device)
    if teacher_type == "ecgfounder_feature_head":
        return load_frozen_feature_head_teacher(model_path, device)
    if teacher_type == "efficientnet1dv2":
        return load_frozen_effnet_teacher(model_path, device, model_name=teacher_model_name)
    raise ValueError(f"unknown raw corruption teacher type: {teacher_type}")


def save_fullft_checkpoint(
    path: Path,
    model: nn.Module,
    op_adapter: OperatorConditionedLogitAdapter | None = None,
) -> None:
    if op_adapter is None:
        torch.save(model.state_dict(), path)
        return
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "op_adapter_state_dict": op_adapter.state_dict(),
            "op_adapter_config": op_adapter.config(),
        },
        path,
    )


def load_fullft_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    op_adapter: OperatorConditionedLogitAdapter | None = None,
) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        if op_adapter is not None and "op_adapter_state_dict" in state:
            op_adapter.load_state_dict(state["op_adapter_state_dict"])
        return
    model.load_state_dict(state)


def train_fullft_raw_corruption_branches_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    trainable_params: list[nn.Parameter],
    aux_teacher_model: nn.Module | None = None,
    op_adapter: OperatorConditionedLogitAdapter | None = None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run primary and optional auxiliary raw-corruption branches for one epoch."""

    primary_stats: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    aux_stats: dict[str, Any] = {"enabled": False, "reason": "disabled"}

    if args.enable_raw_corrupt_consistency:
        primary_stats = train_fullft_raw_corruption_consistency_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            pos_weight=pos_weight,
            device=device,
            copies=args.raw_corrupt_copies,
            severity=args.raw_corrupt_severity,
            severity_profile=args.raw_corrupt_severity_profile,
            ops=list(args.raw_corrupt_ops),
            prob=args.raw_corrupt_prob,
            consistency_weight=args.raw_corrupt_consistency_weight,
            bce_weight=args.raw_corrupt_bce_weight,
            consistency_loss=args.raw_corrupt_consistency_loss,
            rng=np.random.default_rng(args.seed + epoch * 2713),
            trainable_params=trainable_params,
            grad_clip=args.raw_corrupt_grad_clip,
            max_batches=args.raw_corrupt_max_batches,
            renorm=not args.raw_corrupt_no_renorm,
            clip_abs=args.raw_corrupt_clip_abs,
            view_mode=args.raw_corrupt_view_mode,
            augmix_width=args.raw_corrupt_augmix_width,
            augmix_depth=args.raw_corrupt_augmix_depth,
            augmix_alpha=args.raw_corrupt_augmix_alpha,
            augmix_mixture_mode=args.raw_corrupt_augmix_mixture_mode,
            augmix_mixture_prob=args.raw_corrupt_augmix_mixture_prob,
            augmix_mixture_beta_a=args.raw_corrupt_augmix_mixture_beta_a,
            augmix_mixture_beta_b=args.raw_corrupt_augmix_mixture_beta_b,
            teacher_model=None,
            teacher_weight=0.0,
            feature_consistency_weight=args.raw_corrupt_feature_consistency_weight,
            feature_consistency_normalize=args.raw_corrupt_feature_consistency_normalize,
            op_adapter=op_adapter,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        )

    if args.enable_raw_corrupt_aux_consistency:
        if not args.raw_corrupt_aux_ops:
            raise ValueError("--enable_raw_corrupt_aux_consistency requires --raw_corrupt_aux_ops")
        aux_stats = train_fullft_raw_corruption_consistency_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            pos_weight=pos_weight,
            device=device,
            copies=args.raw_corrupt_copies,
            severity=args.raw_corrupt_severity,
            severity_profile=args.raw_corrupt_severity_profile,
            ops=list(args.raw_corrupt_aux_ops),
            prob=args.raw_corrupt_prob,
            consistency_weight=args.raw_corrupt_aux_consistency_weight,
            bce_weight=args.raw_corrupt_aux_bce_weight,
            consistency_loss=args.raw_corrupt_aux_consistency_loss,
            rng=np.random.default_rng(args.seed + epoch * 2713 + 911),
            trainable_params=trainable_params,
            grad_clip=args.raw_corrupt_grad_clip,
            max_batches=args.raw_corrupt_aux_max_batches,
            renorm=not args.raw_corrupt_no_renorm,
            clip_abs=args.raw_corrupt_clip_abs,
            view_mode=args.raw_corrupt_view_mode,
            augmix_width=args.raw_corrupt_augmix_width,
            augmix_depth=args.raw_corrupt_augmix_depth,
            augmix_alpha=args.raw_corrupt_augmix_alpha,
            augmix_mixture_mode=args.raw_corrupt_augmix_mixture_mode,
            augmix_mixture_prob=args.raw_corrupt_augmix_mixture_prob,
            augmix_mixture_beta_a=args.raw_corrupt_augmix_mixture_beta_a,
            augmix_mixture_beta_b=args.raw_corrupt_augmix_mixture_beta_b,
            teacher_model=aux_teacher_model,
            teacher_weight=args.raw_corrupt_aux_teacher_weight,
            teacher_loss=args.raw_corrupt_aux_teacher_loss,
            teacher_view=args.raw_corrupt_aux_teacher_view,
            feature_consistency_weight=args.raw_corrupt_aux_feature_consistency_weight,
            feature_consistency_normalize=args.raw_corrupt_feature_consistency_normalize,
            op_adapter=op_adapter,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        )

    return primary_stats, aux_stats


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--center", default=CENTER_DEFAULT)
    ap.add_argument("--ref_meta_json", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--source_weight", type=float, default=1.0)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--enable_vae_adv_stream", action="store_true")
    ap.add_argument("--adv_weight", type=float, default=20.0)
    ap.add_argument("--adv_weight_start", type=float, default=None)
    ap.add_argument("--adv_weight_warmup_epochs", type=int, default=0)
    ap.add_argument("--source_bce_loss_weight", type=float, default=1.0)
    ap.add_argument("--target_real_bce_loss_weight", type=float, default=1.0)
    ap.add_argument(
        "--adv_bce_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Per-sample BCE multiplier for the VAE adversarial stream after "
            "sampler weighting. Values below 1 make adv samples a gentler branch."
        ),
    )
    ap.add_argument(
        "--adv_clean_logit_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "MSE weight that keeps target-adv logits close to the clean real-anchor "
            "logits captured before the latent-hull attack."
        ),
    )
    ap.add_argument("--k_anchor", type=int, default=100)
    ap.add_argument(
        "--anchor_sample_mode",
        choices=[
            "stratified",
            "hard_bce",
            "uncertainty",
            "stratified_hard_bce",
            "stratified_uncertainty",
            "stratified_pos_boundary",
        ],
        default="stratified",
        help=(
            "How to choose K500 real anchors for the VAE attack. hard/uncertainty "
            "modes score only the random K target subset with the current model."
        ),
    )
    ap.add_argument("--anchor_sample_power", type=float, default=1.0)
    ap.add_argument("--anchor_sample_min_weight", type=float, default=1e-6)
    ap.add_argument(
        "--anchor_class_sample_weights",
        default="",
        help="Optional global class quota weights, e.g. HYP=2,MI=4. Uses K500 labels only.",
    )
    ap.add_argument(
        "--anchor_class_max_repeat",
        type=int,
        default=0,
        help=(
            "Optional per-class quota cap in multiples of the number of unique K500 "
            "positive anchors for that class; 0 disables the cap."
        ),
    )
    ap.add_argument("--vae_classes_in_scope", nargs="*", default=None)
    ap.add_argument("--vae_min_class_count", type=int, default=1)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=3)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--hull_weight_mode", choices=["optimized", "one_hot", "uniform", "dirichlet"], default="optimized")
    ap.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    ap.add_argument(
        "--hull_attack_pos_weight_source",
        choices=["none", "source", "target_train", "source_target"],
        default="none",
        help=(
            "Optional pos_weight source for the inner latent-hull PGD BCE. "
            "target_train/source_target use only the random K target training subset."
        ),
    )
    ap.add_argument("--hull_attack_pos_weight_clip", type=float, default=50.0)
    ap.add_argument("--hull_label_mode", choices=["primary", "exact", "compatible"], default="primary")
    ap.add_argument("--hull_include_anchor", action="store_true")
    ap.add_argument(
        "--hull_partner_pool",
        choices=["target", "target_source"],
        default="target",
        help=(
            "Latent-hull neighbor pool. target uses only the K500 target anchors; "
            "target_source keeps target anchors as attack anchors but adds PTB-XL "
            "source-train latents as same-label/compatible hull partners."
        ),
    )
    ap.add_argument("--ptbxl_vae_cache", default=str(REPO_ROOT / "datasets/PTBXL/PTBXL_vae_multi_nomic.pt"))
    ap.add_argument(
        "--source_partner_limit_per_class",
        type=int,
        default=0,
        help="Optional per-class cap when --hull_partner_pool=target_source; 0 uses all PTB-XL source-train latents.",
    )
    ap.add_argument("--pgd_eps", type=float, default=2.0)
    ap.add_argument("--pgd_batch", type=int, default=4)
    ap.add_argument("--target_val_count", type=int, default=0)
    ap.add_argument("--target_val_seed", type=int, default=None)
    ap.add_argument("--target_val_split_mode", choices=["random", "stratified"], default="random")
    ap.add_argument(
        "--selection_metric",
        choices=["source_auprc", "target_val_auprc", "source_plus_target_val_auprc"],
        default="source_auprc",
    )
    ap.add_argument("--target_val_score_weight", type=float, default=0.5)
    ap.add_argument("--source_train_limit", type=int, default=0)
    ap.add_argument(
        "--cache_dir",
        default="",
        help=(
            "Optional signal-cache directory. When omitted, uses <out_dir>/cache. "
            "This is intended for short engineering smokes that reuse existing "
            "user-owned ECGFounder signal caches without rebuilding raw WFDB data."
        ),
    )
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument("--run_suffix", default="")
    ap.add_argument(
        "--run_name",
        default="",
        help="Optional short run directory name under <out_dir>/runs; useful when config tags exceed filename limits.",
    )
    ap.add_argument("--ecgfounder_input_bandpass_low_hz", type=float, default=None)
    ap.add_argument("--ecgfounder_input_bandpass_high_hz", type=float, default=None)
    ap.add_argument("--ecgfounder_input_repair_flat_leads", action="store_true")
    ap.add_argument("--ecgfounder_input_clip_abs", type=float, default=None)
    ap.add_argument(
        "--supervised_input_mode",
        choices=SUPERVISED_INPUT_MODES,
        default="cached5000",
        help=(
            "cached5000 keeps the historical ECGFounder 12x5000 supervised stream; "
            "raw1000 feeds source/target supervised ECGs through the same 100Hz "
            "ecg1000_to_ecgfounder_input path used by raw-AugMix and PN2021-C eval."
        ),
    )
    ap.add_argument("--source_raw1000_signal_cache", default=str(DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE))
    ap.add_argument("--source_raw1000_label_cache", default=str(DEFAULT_SOURCE_RAW1000_LABEL_CACHE))
    ap.add_argument(
        "--init_head_path",
        default="",
        help="Optional trained Super5 linear head state dict (weight/bias) used to initialize model.dense.",
    )
    ap.add_argument(
        "--anchor_base_root",
        default="",
        help=(
            "Optional root for per-center K-shot ECGTwin VAE anchors. Supports "
            "<root>/<center>/k{k}_seed{seed>/<center>_real_k{k}_seed{seed}."
        ),
    )
    ap.add_argument("--enable_raw_corrupt_consistency", action="store_true")
    ap.add_argument("--raw_corrupt_batch_size", type=int, default=128)
    ap.add_argument("--raw_corrupt_copies", type=int, default=1)
    ap.add_argument("--raw_corrupt_prob", type=float, default=0.5)
    ap.add_argument("--raw_corrupt_severity", type=int, default=4)
    ap.add_argument(
        "--raw_corrupt_severity_profile",
        default="standard",
        choices=STRESS_PROFILE_CHOICES,
    )
    ap.add_argument(
        "--raw_corrupt_ops",
        nargs="+",
        default=[
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
    )
    ap.add_argument("--raw_corrupt_consistency_weight", type=float, default=0.5)
    ap.add_argument(
        "--raw_corrupt_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="soft_bce",
    )
    ap.add_argument("--raw_corrupt_bce_weight", type=float, default=0.1)
    ap.add_argument(
        "--raw_corrupt_feature_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional clean-vs-corrupted penultimate feature MSE weight for "
            "the primary raw-corruption branch."
        ),
    )
    ap.add_argument("--raw_corrupt_max_batches", type=int, default=0)
    ap.add_argument(
        "--raw_corrupt_scope",
        choices=["source", "target", "source_target"],
        default="target",
    )
    ap.add_argument("--raw_corrupt_no_renorm", action="store_true")
    ap.add_argument("--raw_corrupt_clip_abs", type=float, default=6.0)
    ap.add_argument("--raw_corrupt_grad_clip", type=float, default=0.0)
    ap.add_argument(
        "--raw_corrupt_view_mode",
        choices=["single_op", "augmix"],
        default="single_op",
    )
    ap.add_argument("--raw_corrupt_augmix_width", type=int, default=3)
    ap.add_argument("--raw_corrupt_augmix_depth", type=int, default=-1)
    ap.add_argument("--raw_corrupt_augmix_alpha", type=float, default=1.0)
    ap.add_argument(
        "--raw_corrupt_augmix_mixture_mode",
        choices=["beta", "fixed"],
        default="beta",
    )
    ap.add_argument("--raw_corrupt_augmix_mixture_prob", type=float, default=0.5)
    ap.add_argument("--raw_corrupt_augmix_mixture_beta_a", type=float, default=0.0)
    ap.add_argument("--raw_corrupt_augmix_mixture_beta_b", type=float, default=0.0)
    ap.add_argument(
        "--raw_corrupt_feature_consistency_normalize",
        action="store_true",
        help="L2-normalize clean/corrupted features before feature-consistency MSE.",
    )
    ap.add_argument(
        "--raw_corrupt_op_conditioning",
        choices=["none", "explicit_adapter"],
        default="none",
        help=(
            "Optional diagnostic per-operator residual adapter for corrupted "
            "raw-AugMix views. Requires view-level operator metadata."
        ),
    )
    ap.add_argument("--raw_corrupt_op_adapter_hidden", type=int, default=128)
    ap.add_argument("--raw_corrupt_op_adapter_dropout", type=float, default=0.0)
    ap.add_argument("--raw_corrupt_op_adapter_scale", type=float, default=1.0)
    ap.add_argument(
        "--enable_raw_corrupt_aux_consistency",
        action="store_true",
        help=(
            "Optional second raw-corruption branch sharing the same raw loader "
            "and AugMix geometry but using separate ops/loss weights."
        ),
    )
    ap.add_argument("--raw_corrupt_aux_ops", nargs="+", default=[])
    ap.add_argument("--raw_corrupt_aux_consistency_weight", type=float, default=0.0)
    ap.add_argument(
        "--raw_corrupt_aux_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="soft_bce",
    )
    ap.add_argument("--raw_corrupt_aux_bce_weight", type=float, default=0.0)
    ap.add_argument(
        "--raw_corrupt_aux_feature_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional clean-vs-corrupted penultimate feature MSE weight for "
            "the auxiliary raw-corruption branch."
        ),
    )
    ap.add_argument("--raw_corrupt_aux_max_batches", type=int, default=0)
    ap.add_argument(
        "--raw_corrupt_aux_teacher_model_path",
        default="",
        help=(
            "Optional frozen teacher checkpoint used for auxiliary "
            "raw-corruption distillation."
        ),
    )
    ap.add_argument(
        "--raw_corrupt_aux_teacher_type",
        choices=RAW_CORRUPT_TEACHER_TYPES,
        default="ecgfounder_fullft",
    )
    ap.add_argument(
        "--raw_corrupt_aux_teacher_model_name",
        default="efficientnet1dv2",
        help="Model-zoo name used when --raw_corrupt_aux_teacher_type=efficientnet1dv2.",
    )
    ap.add_argument("--raw_corrupt_aux_teacher_weight", type=float, default=0.0)
    ap.add_argument(
        "--raw_corrupt_aux_teacher_loss",
        choices=["soft_bce", "mse_logits"],
        default="soft_bce",
    )
    ap.add_argument(
        "--raw_corrupt_aux_teacher_view",
        choices=["clean", "corrupt"],
        default="corrupt",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--force", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    input_stabilizer_kwargs = input_stabilizer_kwargs_from_args(args)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / "runs" / build_ecgfounder_fullft_run_leaf(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text())
        return
    if input_stabilizer_kwargs:
        print(f"[setup] ECGFounder input stabilizer enabled: {input_stabilizer_kwargs}", flush=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    ptbxl_items, _ = build_ptbxl_items(limit=0)
    pn_items_all = build_pn2021_items(out_dir / "pn2021_manifest.json", limit_per_center=0)
    pn_items = [x for x in pn_items_all if str(x["center"]) == args.center]
    ptbxl = build_signal_cache(
        ptbxl_items,
        cache_dir / f"ptbxl_{args.preprocess_policy}.signals.npy",
        cache_dir / f"ptbxl_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )
    pn = build_signal_cache(
        pn_items,
        cache_dir / f"{args.center}_{args.preprocess_policy}.signals.npy",
        cache_dir / f"{args.center}_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )

    selected_ids = load_selected_ref_ids(Path(args.ref_meta_json), args.center)
    record_ids = pn["record_ids"].astype(str)
    target_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid in selected_ids], dtype=np.int64)
    if len(target_idx) != args.k:
        print(f"[warn] parsed K={len(target_idx)} target records; requested {args.k}", flush=True)
    target_train_idx, target_val_idx, target_train_ids, target_val_ids = split_target_train_val(
        target_idx,
        record_ids,
        pn["labels"],
        args.target_val_count,
        args.seed if args.target_val_seed is None else args.target_val_seed,
        args.target_val_split_mode,
    )
    if args.selection_metric != "source_auprc" and len(target_val_idx) == 0:
        raise RuntimeError("--selection_metric needs --target_val_count > 0 unless source_auprc is used")
    eval_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid not in selected_ids], dtype=np.int64)
    drop_eval_idx = eval_idx[pn["labels"][eval_idx].sum(axis=1) > 0]

    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    init_head_info = None
    if args.init_head_path:
        init_head_info = init_dense_from_head(model, Path(args.init_head_path))
        print(f"[setup] initialized dense Super5 head from {args.init_head_path}", flush=True)
    op_adapter: OperatorConditionedLogitAdapter | None = None
    if args.raw_corrupt_op_conditioning == "explicit_adapter":
        dense = getattr(model, "dense", None)
        if not isinstance(dense, nn.Linear):
            raise RuntimeError("explicit raw-corrupt op conditioning requires ECGFounder model.dense")
        op_adapter = OperatorConditionedLogitAdapter(
            feature_dim=int(dense.in_features),
            num_classes=len(CLASS_NAMES_SUPER5),
            op_names=RAW_CORRUPT_OP_CONDITION_NAMES,
            hidden_dim=int(args.raw_corrupt_op_adapter_hidden),
            dropout=float(args.raw_corrupt_op_adapter_dropout),
            scale=float(args.raw_corrupt_op_adapter_scale),
        ).to(device)
        print(
            "[setup] raw-corrupt explicit op-conditioned adapter enabled: "
            f"ops={RAW_CORRUPT_OP_CONDITION_NAMES} "
            f"hidden={args.raw_corrupt_op_adapter_hidden} "
            f"dropout={args.raw_corrupt_op_adapter_dropout} "
            f"scale={args.raw_corrupt_op_adapter_scale}",
            flush=True,
        )
    model.train()
    if op_adapter is not None:
        op_adapter.train()
    raw_corrupt_aux_teacher_model: nn.Module | None = None
    if float(args.raw_corrupt_aux_teacher_weight) > 0:
        if not args.raw_corrupt_aux_teacher_model_path:
            raise ValueError(
                "--raw_corrupt_aux_teacher_weight > 0 requires "
                "--raw_corrupt_aux_teacher_model_path"
            )
        raw_corrupt_aux_teacher_model = load_frozen_raw_corruption_teacher(
            Path(args.raw_corrupt_aux_teacher_model_path),
            device,
            teacher_type=str(args.raw_corrupt_aux_teacher_type),
            teacher_model_name=str(args.raw_corrupt_aux_teacher_model_name),
        )
        print(
            "[setup] loaded auxiliary raw-corruption frozen teacher: "
            f"type={args.raw_corrupt_aux_teacher_type} "
            f"model_name={args.raw_corrupt_aux_teacher_model_name} "
            f"path={args.raw_corrupt_aux_teacher_model_path} "
            f"weight={args.raw_corrupt_aux_teacher_weight} "
            f"loss={args.raw_corrupt_aux_teacher_loss} "
            f"view={args.raw_corrupt_aux_teacher_view}",
            flush=True,
        )
    anchor_pool = None
    pgd_gen = None
    walker = None
    index = None
    if args.enable_vae_adv_stream:
        print("[setup] loading ECGTwin VAE decoder for full-FT VAE stream", flush=True)
        ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
        for param in ecgtwin.decoder.parameters():
            param.requires_grad_(False)
        victim = ECGFounderFullFTVictim(
            model=model,
            ecgtwin=ecgtwin,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        ).to(device)
        pgd_gen = LatentHullPGDGenerator(
            ecgtwin_wrapper=ecgtwin,
            victim=victim,
            epsilon=args.pgd_eps,
            hull_lambda=args.hull_lambda,
            hull_steps=args.hull_steps,
            hull_lr=args.hull_lr,
            weight_mode=args.hull_weight_mode,
            dirichlet_alpha=args.hull_dirichlet_alpha,
            device=args.device,
        )
        anchor_pool = load_anchor_pool_for_ids(args.center, target_train_ids, pn, args)
        if args.vae_classes_in_scope is not None:
            requested = {str(c) for c in args.vae_classes_in_scope}
            unknown = sorted(requested - set(CLASS_NAMES_SUPER5))
            if unknown:
                raise RuntimeError(f"unknown --vae_classes_in_scope classes: {unknown}")
            anchor_pool["classes_in_scope"] = [
                c for c in anchor_pool["classes_in_scope"] if c in requested
            ]
        if args.vae_min_class_count > 1:
            anchor_pool["classes_in_scope"] = [
                c for c in anchor_pool["classes_in_scope"]
                if int(anchor_pool["label_counts"].get(c, 0)) >= int(args.vae_min_class_count)
            ]
        if not anchor_pool["classes_in_scope"]:
            raise RuntimeError("VAE stream enabled but no classes remain after class filtering")
        walker = StratifiedPoolWalker(
            labels_one_hot=anchor_pool["labels"],
            classes_in_scope=anchor_pool["classes_in_scope"],
            class_to_idx=SUPER5_TO_IDX,
            seed=args.seed,
        )
        index_latents = anchor_pool["latents"]
        index_labels = anchor_pool["labels"]
        source_partner_info = None
        if args.hull_partner_pool == "target_source":
            source_partner_info = load_ptbxl_partner_pool(ptbxl, args)
            index_latents = np.concatenate([index_latents, source_partner_info["latents"]], axis=0)
            index_labels = np.concatenate([index_labels, source_partner_info["labels"]], axis=0)
            print(
                f"[setup] latent partner pool target+source: "
                f"target={len(anchor_pool['latents'])} source={len(source_partner_info['latents'])}",
                flush=True,
            )
        index = SameLabelLatentIndex(
            index_latents,
            index_labels,
            label_mode=args.hull_label_mode,
            seed=args.seed,
            include_self=args.hull_include_anchor,
        )
        set_module_requires_grad(model, True)
    pos_weight = torch.tensor(
        compute_pos_weight(ptbxl["labels"][np.isin(ptbxl["folds"], np.arange(1, 9))], 5, clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )
    if pgd_gen is not None and args.hull_attack_pos_weight_source != "none":
        if args.hull_attack_pos_weight_source == "source":
            attack_weight_labels = ptbxl["labels"][np.isin(ptbxl["folds"], np.arange(1, 9))]
        elif args.hull_attack_pos_weight_source == "target_train":
            attack_weight_labels = pn["labels"][target_train_idx]
        else:
            attack_weight_labels = np.concatenate(
                [
                    ptbxl["labels"][np.isin(ptbxl["folds"], np.arange(1, 9))],
                    pn["labels"][target_train_idx],
                ],
                axis=0,
            )
        attack_pos_weight = torch.tensor(
            compute_pos_weight(
                np.asarray(attack_weight_labels, dtype=np.float32),
                5,
                clip_max=float(args.hull_attack_pos_weight_clip),
            ),
            dtype=torch.float32,
            device=device,
        )
        pgd_gen.attack_pos_weight = attack_pos_weight
        print(
            f"[setup] latent-hull attack pos_weight "
            f"source={args.hull_attack_pos_weight_source}: "
            f"{attack_pos_weight.detach().cpu().tolist()}",
            flush=True,
        )

    raw_corrupt_loader: DataLoader | None = None
    raw_corrupt_setup: dict[str, Any] = {"enabled": False}
    raw_corrupt_aux_setup: dict[str, Any] = {"enabled": False}
    raw_corrupt_any_enabled = bool(
        args.enable_raw_corrupt_consistency or args.enable_raw_corrupt_aux_consistency
    )
    if raw_corrupt_any_enabled:
        if args.enable_raw_corrupt_aux_consistency and not args.raw_corrupt_aux_ops:
            raise ValueError("--enable_raw_corrupt_aux_consistency requires --raw_corrupt_aux_ops")
        raw_datasets: list[Dataset] = []
        folds_for_raw = ptbxl["folds"].astype(np.int64)
        source_raw_idx = np.nonzero(np.isin(folds_for_raw, np.arange(1, 9)))[0]
        if args.source_train_limit > 0:
            source_raw_idx = source_raw_idx[: args.source_train_limit]
        if args.raw_corrupt_scope in {"source", "source_target"}:
            raw_datasets.append(
                load_source_raw_dataset(
                    str(cache_dir),
                    source_raw_idx,
                    ptbxl["labels"][source_raw_idx],
                )
            )
        if args.raw_corrupt_scope in {"target", "source_target"}:
            raw_datasets.append(
                load_target_raw_dataset(
                    args.center,
                    args,
                    sorted(target_train_ids),
                )
            )
        if not raw_datasets:
            raise ValueError(f"unsupported raw_corrupt_scope={args.raw_corrupt_scope!r}")
        raw_corrupt_ds: Dataset = raw_datasets[0] if len(raw_datasets) == 1 else ConcatDataset(raw_datasets)
        raw_corrupt_loader = DataLoader(
            raw_corrupt_ds,
            batch_size=int(args.raw_corrupt_batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        raw_corrupt_common_setup = {
            "scope": str(args.raw_corrupt_scope),
            "n_samples": int(len(raw_corrupt_ds)),
            "batch_size": int(args.raw_corrupt_batch_size),
            "copies": int(args.raw_corrupt_copies),
            "severity": int(args.raw_corrupt_severity),
            "severity_profile": str(args.raw_corrupt_severity_profile),
            "prob": float(args.raw_corrupt_prob),
            "renorm": not bool(args.raw_corrupt_no_renorm),
            "clip_abs": float(args.raw_corrupt_clip_abs),
            "view_mode": str(args.raw_corrupt_view_mode),
            "augmix_width": int(args.raw_corrupt_augmix_width),
            "augmix_depth": int(args.raw_corrupt_augmix_depth),
            "augmix_alpha": float(args.raw_corrupt_augmix_alpha),
            "augmix_mixture_mode": str(args.raw_corrupt_augmix_mixture_mode),
            "augmix_mixture_prob": float(args.raw_corrupt_augmix_mixture_prob),
            "augmix_mixture_beta_a": float(args.raw_corrupt_augmix_mixture_beta_a),
            "augmix_mixture_beta_b": float(args.raw_corrupt_augmix_mixture_beta_b),
            "feature_consistency_normalize": bool(args.raw_corrupt_feature_consistency_normalize),
            "op_conditioning": str(args.raw_corrupt_op_conditioning),
            "op_adapter": None if op_adapter is None else op_adapter.config(),
            "input_stabilizer": dict(input_stabilizer_kwargs),
        }
        raw_corrupt_setup = {
            "enabled": bool(args.enable_raw_corrupt_consistency),
            **raw_corrupt_common_setup,
            "ops": list(args.raw_corrupt_ops),
            "consistency_weight": float(args.raw_corrupt_consistency_weight),
            "bce_weight": float(args.raw_corrupt_bce_weight),
            "feature_consistency_weight": float(args.raw_corrupt_feature_consistency_weight),
            "consistency_loss": str(args.raw_corrupt_consistency_loss),
            "max_batches": int(args.raw_corrupt_max_batches),
        }
        raw_corrupt_aux_setup = {
            "enabled": bool(args.enable_raw_corrupt_aux_consistency),
            **raw_corrupt_common_setup,
            "ops": list(args.raw_corrupt_aux_ops),
            "consistency_weight": float(args.raw_corrupt_aux_consistency_weight),
            "bce_weight": float(args.raw_corrupt_aux_bce_weight),
            "feature_consistency_weight": float(args.raw_corrupt_aux_feature_consistency_weight),
            "consistency_loss": str(args.raw_corrupt_aux_consistency_loss),
            "max_batches": int(args.raw_corrupt_aux_max_batches),
            "teacher_model_path": str(args.raw_corrupt_aux_teacher_model_path),
            "teacher_type": str(args.raw_corrupt_aux_teacher_type),
            "teacher_model_name": str(args.raw_corrupt_aux_teacher_model_name),
            "teacher_weight": float(args.raw_corrupt_aux_teacher_weight),
            "teacher_loss": str(args.raw_corrupt_aux_teacher_loss),
            "teacher_view": str(args.raw_corrupt_aux_teacher_view),
        }
        print(
            f"[setup] fullFT raw-corrupt consistency enabled: scope={args.raw_corrupt_scope} "
            f"n={len(raw_corrupt_ds)} batch={args.raw_corrupt_batch_size} "
            f"copies={args.raw_corrupt_copies} severity={args.raw_corrupt_severity} "
            f"profile={args.raw_corrupt_severity_profile} view={args.raw_corrupt_view_mode} "
            f"w={args.raw_corrupt_augmix_width} d={args.raw_corrupt_augmix_depth} "
            f"mix={args.raw_corrupt_augmix_mixture_mode}:{args.raw_corrupt_augmix_mixture_prob} "
            f"primary={bool(args.enable_raw_corrupt_consistency)} ops={list(args.raw_corrupt_ops)} "
            f"aux={bool(args.enable_raw_corrupt_aux_consistency)} aux_ops={list(args.raw_corrupt_aux_ops)} "
            f"aux_teacher_w={float(args.raw_corrupt_aux_teacher_weight)} "
            f"feat_w={float(args.raw_corrupt_feature_consistency_weight)} "
            f"aux_feat_w={float(args.raw_corrupt_aux_feature_consistency_weight)}",
            flush=True,
        )

    def criterion(logits: torch.Tensor, y: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        return stream_weighted_masked_bce(
            logits,
            y,
            stream,
            pos_weight,
            source_weight=float(args.source_bce_loss_weight),
            target_real_weight=float(args.target_real_bce_loss_weight),
            adv_weight=float(args.adv_bce_loss_weight),
        )

    trainable_params = list(model.parameters())
    if op_adapter is not None:
        trainable_params += list(op_adapter.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    folds = ptbxl["folds"].astype(np.int64)
    val_idx = np.nonzero(folds == 9)[0]
    test_idx = np.nonzero(folds == 10)[0]
    logs = []
    best = -float("inf")
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        adv_info = {
            "signals": None,
            "labels": None,
            "n_adv": 0,
            "delta_mean": None,
            "delta_max": None,
        }
        if args.enable_vae_adv_stream and args.adv_weight > 0 and args.k_anchor > 0:
            assert anchor_pool is not None and pgd_gen is not None and walker is not None and index is not None
            args.current_epoch = epoch
            adv_info = build_adv_epoch(
                model,
                pgd_gen.victim,
                pgd_gen,
                anchor_pool,
                walker,
                index,
                pos_weight,
                args,
                device,
            )
        epoch_adv_weight = scheduled_adv_weight(args, epoch)
        train_loader = make_train_loader(
            ptbxl,
            pn,
            target_train_idx,
            args,
            adv_info["signals"],
            adv_info["labels"],
            adv_info.get("teacher_logits"),
            adv_weight=epoch_adv_weight,
            target_train_record_ids=target_train_ids,
        )
        model.train()
        losses = []
        clean_anchor_losses = []
        for x, y, stream, teacher_logits in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stream = stream.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(
                    prepare_supervised_ecgfounder_input(
                        x,
                        args.supervised_input_mode,
                        input_stabilizer_kwargs,
                        stream=stream,
                    )
                )
                loss = criterion(logits, y, stream)
                if args.adv_clean_logit_anchor_weight > 0:
                    adv_mask = stream == 2
                    if bool(adv_mask.any()):
                        clean_anchor_loss = F.mse_loss(logits[adv_mask], teacher_logits[adv_mask])
                        loss = loss + float(args.adv_clean_logit_anchor_weight) * clean_anchor_loss
                        clean_anchor_losses.append(float(clean_anchor_loss.detach().item()))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        raw_corrupt_stats = {"enabled": False, "reason": "disabled"}
        raw_corrupt_aux_stats = {"enabled": False, "reason": "disabled"}
        if raw_corrupt_any_enabled and raw_corrupt_loader is not None:
            raw_corrupt_stats, raw_corrupt_aux_stats = train_fullft_raw_corruption_branches_epoch(
                model=model,
                loader=raw_corrupt_loader,
                optimizer=opt,
                pos_weight=pos_weight,
                device=device,
                args=args,
                epoch=epoch,
                trainable_params=trainable_params,
                aux_teacher_model=raw_corrupt_aux_teacher_model,
                op_adapter=op_adapter,
                input_stabilizer_kwargs=input_stabilizer_kwargs,
            )
        sched.step()
        val_metrics = eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            val_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        )
        target_val_metrics = (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                target_val_idx,
                args.eval_batch_size,
                device,
                input_stabilizer_kwargs=input_stabilizer_kwargs,
            )
            if len(target_val_idx) > 0
            else None
        )
        target_metrics = eval_split(
            model,
            pn["signals"],
            pn["labels"],
            eval_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        )
        drop_metrics = eval_split(
            model,
            pn["signals"],
            pn["labels"],
            drop_eval_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        )
        selection_score = compute_source_target_selection_score(
            selection_metric=args.selection_metric,
            source_metrics=val_metrics,
            target_val_metrics=target_val_metrics,
            target_val_score_weight=float(args.target_val_score_weight),
        )
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "target_val_macro_auroc": None if target_val_metrics is None else target_val_metrics["macro_auroc"],
            "target_val_macro_auprc": None if target_val_metrics is None else target_val_metrics["macro_auprc"],
            "target_macro_auroc": target_metrics["macro_auroc"],
            "target_macro_auprc": target_metrics["macro_auprc"],
            "target_drop_all_zero_macro_auroc": drop_metrics["macro_auroc"],
            "target_drop_all_zero_macro_auprc": drop_metrics["macro_auprc"],
            "selection_score": selection_score,
            "n_adv": int(adv_info["n_adv"]),
            "adv_weight_effective": float(epoch_adv_weight),
            "adv_bce_loss_weight": float(args.adv_bce_loss_weight),
            "adv_clean_logit_anchor_weight": float(args.adv_clean_logit_anchor_weight),
            "adv_clean_logit_anchor_loss": (
                float(np.mean(clean_anchor_losses)) if clean_anchor_losses else None
            ),
            "adv_delta_mean": adv_info["delta_mean"],
            "adv_delta_max": adv_info["delta_max"],
            "adv_clean_bce_mean": adv_info.get("clean_bce_mean"),
            "adv_init_bce_mean": adv_info.get("init_bce_mean"),
            "adv_adv_bce_mean": adv_info.get("adv_bce_mean"),
            "adv_loss_gain_mean": adv_info.get("loss_gain_mean"),
            "adv_loss_gain_p50": adv_info.get("loss_gain_p50"),
            "adv_loss_gain_p90": adv_info.get("loss_gain_p90"),
            "adv_init_loss_gain_mean": adv_info.get("init_loss_gain_mean"),
            "adv_init_loss_gain_p50": adv_info.get("init_loss_gain_p50"),
            "adv_init_loss_gain_p90": adv_info.get("init_loss_gain_p90"),
            "adv_signed_margin_drop_mean": adv_info.get("signed_margin_drop_mean"),
            "adv_signed_margin_drop_p90": adv_info.get("signed_margin_drop_p90"),
            "adv_pos_signed_margin_drop_mean": adv_info.get("pos_signed_margin_drop_mean"),
            "adv_neg_signed_margin_drop_mean": adv_info.get("neg_signed_margin_drop_mean"),
            "adv_pos_hide_asr": adv_info.get("pos_hide_asr"),
            "adv_neg_add_asr": adv_info.get("neg_add_asr"),
            "adv_sample_anyflip_asr": adv_info.get("sample_anyflip_asr"),
            "adv_pos_correct_count": adv_info.get("pos_correct_count"),
            "adv_neg_correct_count": adv_info.get("neg_correct_count"),
            "adv_sample_clean_correct_count": adv_info.get("sample_clean_correct_count"),
            "adv_anchor_sample_mode": adv_info.get("anchor_sample_mode"),
            "adv_anchor_difficulty_mode": adv_info.get("anchor_difficulty_mode"),
            "adv_hull_attack_pos_weight_source": args.hull_attack_pos_weight_source,
            "adv_anchor_class_sample_weights": adv_info.get("anchor_class_sample_weights"),
            "adv_anchor_class_max_repeat": adv_info.get("anchor_class_max_repeat"),
            "adv_anchor_k_per_class": adv_info.get("anchor_k_per_class"),
            "adv_anchor_picked_unique": adv_info.get("anchor_picked_unique"),
            "adv_anchor_picked_class_counts": adv_info.get("anchor_picked_class_counts"),
            "adv_anchor_score_mean": adv_info.get("anchor_score_mean"),
            "adv_anchor_score_p90": adv_info.get("anchor_score_p90"),
            "adv_anchor_score_max": adv_info.get("anchor_score_max"),
            "adv_anchor_sample_ess": adv_info.get("anchor_sample_ess"),
            "raw_corrupt_loss": _finite_loss_or_none(raw_corrupt_stats),
            "raw_corrupt_stats": raw_corrupt_stats,
            "raw_corrupt_aux_loss": _finite_loss_or_none(raw_corrupt_aux_stats),
            "raw_corrupt_aux_stats": raw_corrupt_aux_stats,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)
        print(
            f"ep={epoch:03d} loss={entry['loss']:.4f} "
            f"val={entry['val_macro_auroc']:.4f}/{entry['val_macro_auprc']:.4f} "
            + (
                f"tval={entry['target_val_macro_auroc']:.4f}/{entry['target_val_macro_auprc']:.4f} "
                if target_val_metrics is not None else ""
            )
            + f"target={entry['target_macro_auroc']:.4f}/{entry['target_macro_auprc']:.4f} "
            f"drop={entry['target_drop_all_zero_macro_auroc']:.4f}/{entry['target_drop_all_zero_macro_auprc']:.4f} "
            + (
                f"adv_gain={entry['adv_loss_gain_mean']:.4f} "
                f"init_gain={entry['adv_init_loss_gain_mean']:.4f} "
                f"margin_drop={entry['adv_signed_margin_drop_mean']:.4f} "
                f"asr={entry['adv_sample_anyflip_asr']:.3f}"
                if entry["adv_loss_gain_mean"] is not None and entry["adv_sample_anyflip_asr"] is not None
                else ""
            ),
            flush=True,
        )
        score = selection_score
        if score > best:
            best = score
            best_epoch = epoch
            save_fullft_checkpoint(run_dir / "best_model.pt", model, op_adapter)

    load_fullft_checkpoint(run_dir / "best_model.pt", model, device, op_adapter)
    result = {
        "method": (
            "ECGFounder official-style full fine-tuning"
            + (" + VAE-only real-anchor latent-hull online AT" if args.enable_vae_adv_stream else "")
            + (" + calibrated/raw corruption consistency" if args.enable_raw_corrupt_consistency else "")
            + (" + auxiliary/raw corruption consistency" if args.enable_raw_corrupt_aux_consistency else "")
            + (" + explicit op-conditioned corruption adapter" if op_adapter is not None else "")
        ),
        "vae_stream_enabled": bool(args.enable_vae_adv_stream),
        "raw_corrupt_consistency_enabled": bool(args.enable_raw_corrupt_consistency),
        "raw_corrupt_aux_consistency_enabled": bool(args.enable_raw_corrupt_aux_consistency),
        "raw_corrupt_op_conditioning_enabled": bool(op_adapter is not None),
        "raw_corrupt_op_adapter": None if op_adapter is None else op_adapter.config(),
        "supervised_input_mode": str(args.supervised_input_mode),
        "input_stabilizer": dict(input_stabilizer_kwargs),
        "label_mapping": pn2021_super5_label_mapping_payload(),
        "center": args.center,
        "K": int(len(target_idx)),
        "target_train_K": int(len(target_train_idx)),
        "target_val_K": int(len(target_val_idx)),
        "selected_ref_record_ids": sorted(selected_ids),
        "target_train_record_ids": sorted(target_train_ids),
        "target_val_record_ids": sorted(target_val_ids),
        "selection_metric": args.selection_metric,
        "target_val_split_mode": args.target_val_split_mode,
        "selection_score_weight": float(args.target_val_score_weight),
        "best_epoch": best_epoch,
        "best_selection_score": float(best),
        "ptbxl_fold10": eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            test_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        ),
        "target_val": (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                target_val_idx,
                args.eval_batch_size,
                device,
                input_stabilizer_kwargs=input_stabilizer_kwargs,
            )
            if len(target_val_idx) > 0
            else None
        ),
        "target_excluding_ref": eval_split(
            model,
            pn["signals"],
            pn["labels"],
            eval_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        ),
        "target_drop_all_zero_excluding_ref": eval_split(
            model,
            pn["signals"],
            pn["labels"],
            drop_eval_idx,
            args.eval_batch_size,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
        ),
        "n_target_eval": int(len(eval_idx)),
        "n_target_drop_all_zero_eval": int(len(drop_eval_idx)),
        "config": vars(args),
        "init_head": init_head_info,
        "raw_corrupt_consistency": raw_corrupt_setup,
        "raw_corrupt_aux_consistency": raw_corrupt_aux_setup,
        "vae_anchor_pool": None if anchor_pool is None else {
            "classes_in_scope": anchor_pool["classes_in_scope"],
            "label_counts": anchor_pool["label_counts"],
            "requested_classes_in_scope": args.vae_classes_in_scope,
            "min_class_count": int(args.vae_min_class_count),
            "source_base": anchor_pool["source_base"],
            "hull_partner_pool": args.hull_partner_pool,
            "source_partner_count": (
                None if args.hull_partner_pool == "target" else int(len(index.latents) - len(anchor_pool["latents"]))
            ),
            "anchor_sample_mode": args.anchor_sample_mode,
            "anchor_sample_power": float(args.anchor_sample_power),
            "anchor_sample_min_weight": float(args.anchor_sample_min_weight),
            "anchor_class_sample_weights": parse_class_weight_string(args.anchor_class_sample_weights),
            "anchor_class_max_repeat": int(args.anchor_class_max_repeat),
            "hull_attack_pos_weight_source": args.hull_attack_pos_weight_source,
            "hull_attack_pos_weight_clip": float(args.hull_attack_pos_weight_clip),
        },
        "official_evidence": {
            "finetune_model": "ft_12lead_ECGFounder(..., linear_prob=False)",
            "notebook_hyperparams": "lr=1e-4, weight_decay=1e-5, Epochs=5",
            "preprocess": "ECGFounder 12 x 5000, official_ptbxl_eval preprocessing",
        },
    }
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
