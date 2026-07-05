#!/usr/bin/env python3
"""ECGFounder official-style full fine-tuning Super5 pilot.

This script tests ECGFounder under full fine-tuning rather than the previous
frozen-encoder/head-only protocol. It follows the local ECGFounder notebook's
recommended mode (`linear_prob=False`): initialize the 12-lead checkpoint,
replace the dense layer with a Super5 head, and fine-tune the whole model on
PTB-XL source plus K target-center real ECGs.

The locked 2026-06-18 protocol also uses this entrypoint for ECGFounder
VAE-LHAT and VAE-LHAT plus three-chain AugMix, while keeping all layers
trainable and selecting only the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
for _p in [str(ECGFOUNDER_ROOT), str(REPO_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from physionet2021_dataset import EXPECTED_LEADS, TARGET_POINTS  # noqa: E402
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from ecg_adv_gen.runner.ecgfounder_linear_probe import (  # noqa: E402
    CHECKPOINT,
    PTBXL_CSV,
    build_pn2021_items,
    build_ptbxl_items,
    compute_metrics,
    preprocess_record,
)
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    build_three_chain_vae_lhat_augmix_views as build_three_chain_vae_lhat_augmix_views_core,
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
from ecg_adv_gen.data.raw_signals import (  # noqa: E402
    RawSignalDataset,
    anchor_signal_npz_path,
    apply_augmix_op_np as _raw_signal_apply_augmix_op_np,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5  # noqa: E402
from ecg_adv_gen.labels.super5_mapping import SUPER5_TO_IDX  # noqa: E402
from ecg_adv_gen.labels import pn2021_super5_label_mapping_payload  # noqa: E402
from ecg_adv_gen.models.ecgfounder_inference import evaluate_signal_split  # noqa: E402
from ecg_adv_gen.models.ecgfounder_torch import (  # noqa: E402
    ecg1000_to_ecgfounder_input,
    global_zscore_torch,
)
from ecg_adv_gen.training import (  # noqa: E402
    build_weighted_signal_stream_loader_from_datasets,
    compute_pos_weight,
    fullft_adv_batch_diagnostics,
    masked_bce_with_logits,
    set_module_requires_grad,
    stream_weighted_masked_bce,
    summarize_fullft_adv_epoch_diagnostics,
)
from ecg_adv_gen.run_naming import build_ecgfounder_fullft_run_leaf  # noqa: E402
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    STRESS_PROFILE_CHOICES,
    resolve_severity_profile_args as _resolve_severity_profile_args,
    severity_profile_metadata as _severity_profile_metadata,
)
from methods.augmix.severity import AVAILABLE_OPS  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_fullft_super5_20260523"
CENTER_DEFAULT = "cpsc_2018"
DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE = (
    DATA_ROOT / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
)
DEFAULT_SOURCE_RAW1000_LABEL_CACHE = (
    DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/ptbxl_labels.C5.all.npy"
)
REAL_ROOTS = [
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v2",
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v1",
]


def _parse_float_csv(value: str | None) -> list[float] | None:
    if value is None or str(value).strip() == "":
        return None
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _resolve_latent_augmix_severity_profile(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    resolver_args = argparse.Namespace(
        severity_profile=args.latent_augmix_severity_profile,
        severity_params_file=args.latent_augmix_severity_params_file,
        severity_params_name=args.latent_augmix_severity_params_name,
        mode="stream",
    )
    params = _resolve_severity_profile_args(resolver_args)
    resolver_args.severity_profile_params = params
    return params, _severity_profile_metadata(resolver_args)


def apply_augmix_op_np(
    sig_ct: np.ndarray,
    op_name: str,
    op_severity: int,
    severity_profile: str = "standard",
    severity_profile_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply one ECG AugMix op, including evaluator-style custom profiles."""

    return _raw_signal_apply_augmix_op_np(
        sig_ct,
        op_name,
        op_severity,
        severity_profile,
        severity_profile_params=severity_profile_params,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_raw_ecgfounder_input(x: torch.Tensor) -> torch.Tensor:
    return ecg1000_to_ecgfounder_input(x)


class ECGFounderFullFTVictim(nn.Module):
    """Differentiable ECGTwin-latent -> ECGFounder full model logits wrapper."""

    def __init__(
        self,
        model: nn.Module,
        ecgtwin: ECGTwinWrapper,
    ) -> None:
        super().__init__()
        self.model = model
        self.ecgtwin = ecgtwin
        self.num_classes = len(CLASS_NAMES_SUPER5)

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
        return self.model(prepare_raw_ecgfounder_input(ecg_ct))


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
    signal_path = (
        Path(args.target_raw1000_npz_override)
        if getattr(args, "target_raw1000_npz_override", "")
        else anchor_signal_npz_path(center, args, default_roots=REAL_ROOTS)
    )
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


def build_locked_three_chain_latent_augmix_views(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    severity_profile: str,
    severity_profile_params: dict[str, Any] | None = None,
    severity_params_file: str | None = None,
    severity_params_name: str | None = None,
    width: int,
    depth: int,
    alpha: float,
    mixture_mode: str = "beta",
    mixture_prob: float = 0.5,
    mixture_beta_a: float | None = None,
    mixture_beta_b: float | None = None,
    op_schedule: str = "random",
    chain_weights: list[float] | None = None,
    ops: list[str],
    rng: np.random.Generator,
    renorm: bool = False,
    clip_abs: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build locked ECGFounder three-chain VAE-LHAT AugMix waveform views."""

    def _apply(sig_ct: np.ndarray, op_name: str, op_severity: int, op_profile: str) -> np.ndarray:
        return apply_augmix_op_np(
            sig_ct,
            op_name,
            op_severity,
            op_profile,
            severity_profile_params=severity_profile_params,
        )

    views, stats = build_three_chain_vae_lhat_augmix_views_core(
        anchor_signals_ct,
        adv_signals_ct,
        copies=copies,
        severity=severity,
        severity_profile=severity_profile,
        width=width,
        depth=depth,
        alpha=alpha,
        ops=ops,
        rng=rng,
        op_apply_fn=_apply,
        available_ops=AVAILABLE_OPS,
        mixture_mode=mixture_mode,
        mixture_prob=mixture_prob,
        mixture_beta_a=mixture_beta_a,
        mixture_beta_b=mixture_beta_b,
        op_schedule=op_schedule,
        chain_weights=chain_weights,
        renorm=renorm,
        clip_abs=clip_abs,
    )
    if str(severity_profile) == "custom":
        stats["severity_params_file"] = str(severity_params_file or "")
        stats["severity_params_name"] = str(severity_params_name or "")
        stats["severity_profile_params"] = severity_profile_params
    return views, stats


def summarize_latent_augmix_epoch_stats(stats_batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-batch locked latent-AugMix stats for logging."""

    enabled_batches = [s for s in stats_batches if bool(s.get("enabled", False))]
    if not enabled_batches:
        return {
            "enabled": False,
            "topology": "locked_three_chain_vae_lhat_augmix",
            "n_generated": 0,
        }
    op_counts: dict[str, int] = {}
    for stats in enabled_batches:
        for op_name, count in dict(stats.get("op_counts", {})).items():
            op_counts[str(op_name)] = op_counts.get(str(op_name), 0) + int(count)

    def _mean_stat(name: str) -> float | None:
        values = [
            float(stats[name])
            for stats in enabled_batches
            if stats.get(name) is not None and np.isfinite(float(stats[name]))
        ]
        return float(np.mean(values)) if values else None

    first = enabled_batches[0]
    return {
        "enabled": True,
        "topology": first.get("topology", "locked_three_chain_vae_lhat_augmix"),
        "view_mode": first.get("view_mode", "vae_lhat_augmix"),
        "n_generated": int(sum(int(s.get("n_generated", 0)) for s in enabled_batches)),
        "copies": int(first.get("copies", 0)),
        "severity": int(first.get("severity", 0)),
        "severity_profile": str(first.get("severity_profile", "")),
        "width": int(first.get("width", 3)),
        "depth": int(first.get("depth", -1)),
        "alpha": float(first.get("alpha", 1.0)),
        "corruption_chain_count": int(first.get("corruption_chain_count", 2)),
        "adversarial_chain_count": int(first.get("adversarial_chain_count", 1)),
        "adversarial_chain_index": int(first.get("adversarial_chain_index", 2)),
        "adversarial_chain_corrupted": bool(first.get("adversarial_chain_corrupted", False)),
        "chain_roles": list(first.get("chain_roles", [])),
        "adv_weight_mean": _mean_stat("adv_weight_mean"),
        "adv_weight_max": _mean_stat("adv_weight_max"),
        "beta_m_mean": _mean_stat("beta_m_mean"),
        "chain_depth_mean": _mean_stat("chain_depth_mean"),
        "ops": list(first.get("ops", [])),
        "op_counts": op_counts,
        "renorm": bool(first.get("renorm", False)),
        "clip_abs": float(first.get("clip_abs", 0.0)),
    }


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
) -> dict:
    return evaluate_signal_split(
        model,
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
    args: argparse.Namespace,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_weight: float | None = None,
    target_train_record_ids: set[str] | None = None,
) -> DataLoader:
    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    effective_adv_weight = float(args.adv_weight if adv_weight is None else adv_weight)
    if target_train_record_ids is None:
        raise ValueError("raw1000 supervised training requires target_train_record_ids")
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


def make_source_only_train_loader(
    ptbxl_payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> DataLoader:
    """Build the locked PTB-XL-only full-FT loader."""

    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    source_ds = load_source_raw1000_dataset(
        args,
        source_idx,
        ptbxl_payload["labels"][source_idx],
    )
    return build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source_ds,
        target_dataset=source_ds,
        source_weight=float(args.source_weight),
        target_real_weight=0.0,
        adv_weight=0.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=len(CLASS_NAMES_SUPER5),
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
    restore_trainable_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    set_module_requires_grad(model, False)
    model.eval()
    victim.eval()
    adv_signals: list[np.ndarray] = []
    adv_labels: list[np.ndarray] = []
    adv_teacher_logits: list[np.ndarray] = []
    delta_norms: list[float] = []
    batch_diagnostics: list[dict[str, np.ndarray | int]] = []
    latent_augmix_stats_batches: list[dict[str, Any]] = []
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
                adv_input = prepare_raw_ecgfounder_input(x_adv_1000)
                adv_logits = victim.model(adv_input)
                batch_diagnostics.append(
                    fullft_adv_batch_diagnostics(clean_logits, init_logits, adv_logits, y)
                )
            if args.enable_latent_augmix_branch:
                with torch.no_grad():
                    anchor_x_1000 = victim._ecgtwin_latent_to_ecg1000(z)
                latent_augmix_rng = np.random.default_rng(
                    int(args.seed) + int(getattr(args, "current_epoch", 0)) * 100003 + int(start)
                )
                latent_augmix_np, latent_augmix_stats = build_locked_three_chain_latent_augmix_views(
                    anchor_x_1000.detach().cpu().numpy().astype(np.float32, copy=False),
                    x_adv_1000.detach().cpu().numpy().astype(np.float32, copy=False),
                    copies=args.latent_augmix_copies,
                    severity=args.latent_augmix_severity,
                    severity_profile=args.latent_augmix_severity_profile,
                    severity_profile_params=getattr(args, "latent_augmix_severity_profile_params", None),
                    severity_params_file=args.latent_augmix_severity_params_file,
                    severity_params_name=args.latent_augmix_severity_params_name,
                    width=args.latent_augmix_width,
                    depth=args.latent_augmix_depth,
                    alpha=args.latent_augmix_alpha,
                    mixture_mode=args.latent_augmix_mixture_mode,
                    mixture_prob=args.latent_augmix_mixture_prob,
                    mixture_beta_a=(
                        None
                        if float(args.latent_augmix_mixture_beta_a) <= 0.0
                        else float(args.latent_augmix_mixture_beta_a)
                    ),
                    mixture_beta_b=(
                        None
                        if float(args.latent_augmix_mixture_beta_b) <= 0.0
                        else float(args.latent_augmix_mixture_beta_b)
                    ),
                    op_schedule=args.latent_augmix_op_schedule,
                    chain_weights=_parse_float_csv(args.latent_augmix_chain_weights),
                    ops=list(args.latent_augmix_ops),
                    rng=latent_augmix_rng,
                    renorm=bool(args.latent_augmix_renorm),
                    clip_abs=float(args.latent_augmix_clip_abs),
                )
                latent_augmix_stats_batches.append(latent_augmix_stats)
                if latent_augmix_np.shape[0] > 0:
                    latent_augmix_t = torch.from_numpy(latent_augmix_np).float().to(device)
                    x_adv_for_stream = latent_augmix_t.detach().cpu().numpy().astype(np.float32)
                    adv_signals.append(x_adv_for_stream)
                    labels_rep = np.tile(
                        anchor_pool["labels"][batch_idx].astype(np.float32),
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[: latent_augmix_np.shape[0]]
                    teacher_rep = np.tile(
                        clean_logits.detach().cpu().numpy().astype(np.float32),
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[: latent_augmix_np.shape[0]]
                    adv_labels.append(labels_rep.astype(np.float32, copy=False))
                    adv_teacher_logits.append(teacher_rep.astype(np.float32, copy=False))
            else:
                x_adv_for_stream = x_adv_1000.detach().cpu().numpy().astype(np.float32)
                adv_signals.append(x_adv_for_stream)
                adv_labels.append(anchor_pool["labels"][batch_idx].astype(np.float32))
                adv_teacher_logits.append(clean_logits.detach().cpu().numpy().astype(np.float32))
            delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())
    finally:
        if restore_trainable_fn is not None:
            restore_trainable_fn()
        else:
            set_module_requires_grad(model, True)
    signals = (
        np.concatenate(adv_signals, axis=0).astype(np.float32)
        if adv_signals
        else np.empty(
            (0, 12, 1000),
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
    latent_augmix_stats = (
        summarize_latent_augmix_epoch_stats(latent_augmix_stats_batches)
        if args.enable_latent_augmix_branch
        else {
            "enabled": False,
            "topology": "locked_three_chain_vae_lhat_augmix",
            "reason": "disabled",
            "n_generated": 0,
        }
    )
    return {
        "signals": signals,
        "labels": labels,
        "teacher_logits": teacher_logits,
        "latent_augmix_stats": latent_augmix_stats,
        **adv_stats,
    }


def save_fullft_checkpoint(
    path: Path,
    model: nn.Module,
) -> None:
    torch.save(model.state_dict(), path)


def load_fullft_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return
    model.load_state_dict(state)


def configure_ecgfounder_full_train(model: nn.Module) -> dict[str, Any]:
    """Lock ECGFounder fine-tuning to the paper mainline: all model weights trainable."""

    set_module_requires_grad(model, True)
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    return {
        "scope": "full",
        "last_n_stages": 0,
        "train_module_names": ["model"],
        "n_trainable_model_tensors": int(len(trainable_model_params)),
        "n_trainable_model_params": int(sum(p.numel() for p in trainable_model_params)),
        "n_total_model_params": int(sum(p.numel() for p in model.parameters())),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--stage", choices=["k500", "ptbxl_source"], default="k500")
    ap.add_argument("--center", default=CENTER_DEFAULT)
    ap.add_argument("--ref_meta_json", default="")
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
    ap.add_argument(
        "--enable_latent_augmix_branch",
        action="store_true",
        help=(
            "After each VAE-LHAT adversarial waveform is generated, build the "
            "locked three-chain AugMix view: two official corruption chains "
            "plus one uncorrupted VAE-LHAT adversarial waveform chain."
        ),
    )
    ap.add_argument(
        "--latent_augmix_topology",
        choices=["locked_three_chain"],
        default="locked_three_chain",
    )
    ap.add_argument("--latent_augmix_copies", type=int, default=1)
    ap.add_argument("--latent_augmix_width", type=int, default=3)
    ap.add_argument("--latent_augmix_depth", type=int, default=-1)
    ap.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    ap.add_argument("--latent_augmix_severity", type=int, default=5)
    ap.add_argument(
        "--latent_augmix_severity_profile",
        default="standard",
        choices=STRESS_PROFILE_CHOICES,
    )
    ap.add_argument(
        "--latent_augmix_severity_params_file",
        default=None,
        help="YAML/JSON profile file used only with --latent_augmix_severity_profile custom.",
    )
    ap.add_argument(
        "--latent_augmix_severity_params_name",
        default=None,
        help="Profile name under top-level profiles used only with --latent_augmix_severity_profile custom.",
    )
    ap.add_argument(
        "--latent_augmix_mixture_mode",
        choices=["beta", "fixed"],
        default="beta",
    )
    ap.add_argument("--latent_augmix_mixture_prob", type=float, default=0.5)
    ap.add_argument("--latent_augmix_mixture_beta_a", type=float, default=0.0)
    ap.add_argument("--latent_augmix_mixture_beta_b", type=float, default=0.0)
    ap.add_argument(
        "--latent_augmix_op_schedule",
        choices=["random", "cycle", "per_op", "official_s5_depth23_composite_cycle"],
        default="random",
    )
    ap.add_argument("--latent_augmix_chain_weights", default="")
    ap.add_argument(
        "--latent_augmix_ops",
        nargs="+",
        default=[
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
        choices=AVAILABLE_OPS,
    )
    ap.add_argument(
        "--latent_augmix_renorm",
        action="store_true",
        help=(
            "Diagnostic only. The locked main protocol leaves generated "
            "AugMix waveforms unnormalized until ECGFounder input conversion."
        ),
    )
    ap.add_argument("--latent_augmix_clip_abs", type=float, default=6.0)
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
    ap.add_argument("--hull_init_logit_gap", type=float, default=4.0)
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
    ap.add_argument("--pgd_eps", type=float, default=2.0)
    ap.add_argument("--pgd_batch", type=int, default=4)
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument(
        "--run_name",
        default="",
        help="Optional short run directory name under <out_dir>/runs; useful when config tags exceed filename limits.",
    )
    ap.add_argument(
        "--init_model_path",
        default="",
        help="Optional full-model ECGFounder checkpoint used before K500 full fine-tuning.",
    )
    ap.add_argument("--source_raw1000_signal_cache", default=str(DEFAULT_SOURCE_RAW1000_SIGNAL_CACHE))
    ap.add_argument("--source_raw1000_label_cache", default=str(DEFAULT_SOURCE_RAW1000_LABEL_CACHE))
    ap.add_argument(
        "--target_raw1000_npz_override",
        default="",
        help=(
            "Optional target-center raw1000 K-shot artifact. Use this for the "
            "locked PN2021-C raw-first protocol so target supervised ECGs are "
            "not loaded from legacy pre-zscored .signals.npz files."
        ),
    )
    ap.add_argument(
        "--anchor_base_root",
        default="",
        help=(
            "Optional root for per-center K-shot ECGTwin VAE anchors. Supports "
            "<root>/<center>/k{k}_seed{seed>/<center>_real_k{k}_seed{seed}."
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260531)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    args.latent_augmix_severity_profile_params = None
    args.latent_augmix_severity_profile_metadata = None
    if args.enable_latent_augmix_branch:
        (
            args.latent_augmix_severity_profile_params,
            args.latent_augmix_severity_profile_metadata,
        ) = _resolve_latent_augmix_severity_profile(args)
    elif args.latent_augmix_severity_params_file or args.latent_augmix_severity_params_name:
        raise ValueError(
            "--latent_augmix_severity_params_file/name require --enable_latent_augmix_branch"
        )

    if args.stage == "k500" and not args.ref_meta_json:
        raise ValueError("--ref_meta_json is required for --stage k500")
    if args.enable_latent_augmix_branch and not args.enable_vae_adv_stream:
        raise ValueError("--enable_latent_augmix_branch requires --enable_vae_adv_stream")
    if args.enable_latent_augmix_branch:
        if args.latent_augmix_topology != "locked_three_chain":
            raise ValueError("--enable_latent_augmix_branch requires --latent_augmix_topology locked_three_chain")
        if int(args.latent_augmix_width) != 3:
            raise ValueError("--latent_augmix_topology locked_three_chain requires exactly three chains")
        if bool(args.latent_augmix_renorm):
            raise ValueError("--latent_augmix_renorm is diagnostic-only and excluded from the locked main protocol")
    if args.stage == "ptbxl_source":
        if args.enable_vae_adv_stream:
            raise ValueError("--stage ptbxl_source does not support VAE adversarial stream")
        if args.enable_latent_augmix_branch:
            raise ValueError("--stage ptbxl_source does not support latent AugMix")
        if args.init_model_path:
            raise ValueError("--stage ptbxl_source must start from the official ECGFounder checkpoint")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / "runs" / build_ecgfounder_fullft_run_leaf(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists():
        print(result_path.read_text())
        return

    cache_dir = out_dir / "cache"
    ptbxl_items, _ = build_ptbxl_items(limit=0)
    ptbxl = build_signal_cache(
        ptbxl_items,
        cache_dir / f"ptbxl_{args.preprocess_policy}.signals.npy",
        cache_dir / f"ptbxl_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )
    if args.stage == "ptbxl_source":
        pn = None
        selected_ids: set[str] = set()
        target_idx = np.empty(0, dtype=np.int64)
        target_train_idx = np.empty(0, dtype=np.int64)
        target_train_ids: set[str] = set()
        eval_idx = np.empty(0, dtype=np.int64)
        drop_eval_idx = np.empty(0, dtype=np.int64)
    else:
        pn_items_all = build_pn2021_items(out_dir / "pn2021_manifest.json", limit_per_center=0)
        pn_items = [x for x in pn_items_all if str(x["center"]) == args.center]
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
        target_train_idx = target_idx
        target_train_ids = set(str(record_ids[i]) for i in target_train_idx)
        eval_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid not in selected_ids], dtype=np.int64)
        drop_eval_idx = eval_idx[pn["labels"][eval_idx].sum(axis=1) > 0]

    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    init_model_info = None
    if args.init_model_path:
        init_model_path = Path(args.init_model_path)
        load_fullft_checkpoint(init_model_path, model, device)
        init_model_info = {
            "path": str(init_model_path),
            "type": "full_model",
        }
        print(f"[setup] initialized full ECGFounder model from {init_model_path}", flush=True)
    model.train()
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
            init_logit_gap=args.hull_init_logit_gap,
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
        index = SameLabelLatentIndex(
            anchor_pool["latents"],
            anchor_pool["labels"],
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

    trainable_setup = configure_ecgfounder_full_train(model)
    train_mode_fn = model.train
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected for ECGFounder run")
    print(
        "[setup] ECGFounder trainable scope: "
        f"scope={trainable_setup['scope']} "
        f"last_n_stages={trainable_setup['last_n_stages']} "
        f"modules={trainable_setup['train_module_names']} "
        f"trainable={trainable_setup['n_trainable_model_params']:,}/"
        f"{trainable_setup['n_total_model_params']:,} model params",
        flush=True,
    )
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    folds = ptbxl["folds"].astype(np.int64)
    val_idx = np.nonzero(folds == 9)[0]
    test_idx = np.nonzero(folds == 10)[0]
    logs = []
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
                restore_trainable_fn=lambda: configure_ecgfounder_full_train(model),
            )
        epoch_adv_weight = scheduled_adv_weight(args, epoch)
        if args.stage == "ptbxl_source":
            train_loader = make_source_only_train_loader(ptbxl, args)
        else:
            assert pn is not None
            train_loader = make_train_loader(
                ptbxl,
                args,
                adv_info["signals"],
                adv_info["labels"],
                adv_info.get("teacher_logits"),
                adv_weight=epoch_adv_weight,
                target_train_record_ids=target_train_ids,
            )
        train_mode_fn()
        losses = []
        clean_anchor_losses = []
        for x, y, stream, teacher_logits in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stream = stream.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(prepare_raw_ecgfounder_input(x))
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
        sched.step()
        val_metrics = eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            val_idx,
            args.eval_batch_size,
            device,
        )
        target_metrics = (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                eval_idx,
                args.eval_batch_size,
                device,
            )
            if pn is not None
            else None
        )
        drop_metrics = (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                drop_eval_idx,
                args.eval_batch_size,
                device,
            )
            if pn is not None
            else None
        )
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "target_macro_auroc": None if target_metrics is None else target_metrics["macro_auroc"],
            "target_macro_auprc": None if target_metrics is None else target_metrics["macro_auprc"],
            "target_drop_all_zero_macro_auroc": None if drop_metrics is None else drop_metrics["macro_auroc"],
            "target_drop_all_zero_macro_auprc": None if drop_metrics is None else drop_metrics["macro_auprc"],
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
            "latent_augmix_stats": adv_info.get("latent_augmix_stats"),
            "trainable_scope": trainable_setup,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)
        print(
            f"ep={epoch:03d} loss={entry['loss']:.4f} "
            f"val={entry['val_macro_auroc']:.4f}/{entry['val_macro_auprc']:.4f} "
            + (
                f"target={entry['target_macro_auroc']:.4f}/{entry['target_macro_auprc']:.4f} "
                f"drop={entry['target_drop_all_zero_macro_auroc']:.4f}/"
                f"{entry['target_drop_all_zero_macro_auprc']:.4f} "
                if entry["target_macro_auroc"] is not None
                else ""
            )
            + (
                f"adv_gain={entry['adv_loss_gain_mean']:.4f} "
                f"init_gain={entry['adv_init_loss_gain_mean']:.4f} "
                f"margin_drop={entry['adv_signed_margin_drop_mean']:.4f} "
                f"asr={entry['adv_sample_anyflip_asr']:.3f}"
                if entry["adv_loss_gain_mean"] is not None and entry["adv_sample_anyflip_asr"] is not None
                else ""
            )
            + (
                f" latmix_n={entry['latent_augmix_stats'].get('n_generated', 0)} "
                f"latmix_wadv={entry['latent_augmix_stats'].get('adv_weight_mean'):.3f}"
                if isinstance(entry.get("latent_augmix_stats"), dict)
                and entry["latent_augmix_stats"].get("enabled")
                and entry["latent_augmix_stats"].get("adv_weight_mean") is not None
                else ""
            ),
            flush=True,
        )
        save_fullft_checkpoint(run_dir / "last_model.pt", model)

    selected_checkpoint_name = "last_model.pt"
    load_fullft_checkpoint(run_dir / selected_checkpoint_name, model, device)
    last_epoch = logs[-1]["epoch"] if logs else None
    result = {
        "method": (
            "ECGFounder official-style full fine-tuning"
            + (" + VAE-only real-anchor latent-hull online AT" if args.enable_vae_adv_stream else "")
            + (" + locked three-chain VAE-LHAT AugMix" if args.enable_latent_augmix_branch else "")
        ),
        "vae_stream_enabled": bool(args.enable_vae_adv_stream),
        "latent_augmix_branch_enabled": bool(args.enable_latent_augmix_branch),
        "trainable_scope": trainable_setup,
        "label_mapping": pn2021_super5_label_mapping_payload(),
        "stage": args.stage,
        "center": None if args.stage == "ptbxl_source" else args.center,
        "K": int(len(target_idx)),
        "target_train_K": int(len(target_train_idx)),
        "selected_ref_record_ids": sorted(selected_ids),
        "target_train_record_ids": sorted(target_train_ids),
        "checkpoint_policy": "last",
        "selected_checkpoint": selected_checkpoint_name,
        "last_epoch": last_epoch,
        "ptbxl_fold10": eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            test_idx,
            args.eval_batch_size,
            device,
        ),
        "target_excluding_ref": (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                eval_idx,
                args.eval_batch_size,
                device,
            )
            if pn is not None
            else None
        ),
        "target_drop_all_zero_excluding_ref": (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                drop_eval_idx,
                args.eval_batch_size,
                device,
            )
            if pn is not None
            else None
        ),
        "n_target_eval": int(len(eval_idx)),
        "n_target_drop_all_zero_eval": int(len(drop_eval_idx)),
        "config": vars(args),
        "init_model": init_model_info,
        "latent_augmix_branch": {
            "enabled": bool(args.enable_latent_augmix_branch),
            "topology": str(args.latent_augmix_topology),
            "chain_roles": [
                "corruption",
                "corruption",
                "vae_lhat_adversarial_waveform",
            ] if args.enable_latent_augmix_branch else [],
            "adversarial_chain_corrupted": False if args.enable_latent_augmix_branch else None,
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "severity": int(args.latent_augmix_severity),
            "severity_profile": str(args.latent_augmix_severity_profile),
            "severity_params_file": args.latent_augmix_severity_params_file,
            "severity_params_name": args.latent_augmix_severity_params_name,
            "severity_profile_metadata": args.latent_augmix_severity_profile_metadata,
            "mixture_mode": str(args.latent_augmix_mixture_mode),
            "mixture_prob": float(args.latent_augmix_mixture_prob),
            "mixture_beta_a": float(args.latent_augmix_mixture_beta_a),
            "mixture_beta_b": float(args.latent_augmix_mixture_beta_b),
            "op_schedule": str(args.latent_augmix_op_schedule),
            "chain_weights": _parse_float_csv(args.latent_augmix_chain_weights),
            "ops": list(args.latent_augmix_ops),
            "renorm": bool(args.latent_augmix_renorm),
            "clip_abs": float(args.latent_augmix_clip_abs),
        },
        "vae_anchor_pool": None if anchor_pool is None else {
            "classes_in_scope": anchor_pool["classes_in_scope"],
            "label_counts": anchor_pool["label_counts"],
            "requested_classes_in_scope": args.vae_classes_in_scope,
            "min_class_count": int(args.vae_min_class_count),
            "source_base": anchor_pool["source_base"],
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
