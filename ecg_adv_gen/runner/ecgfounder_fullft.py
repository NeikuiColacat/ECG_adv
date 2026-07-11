#!/usr/bin/env python3
"""ECGFounder official-style full fine-tuning Super5 pilot.

This script tests ECGFounder under full fine-tuning rather than the previous
frozen-encoder/head-only protocol. It follows the local ECGFounder notebook's
recommended mode (`linear_prob=False`): initialize the 12-lead checkpoint,
replace the dense layer with a Super5 head, and fine-tune the whole model on
PTB-XL source plus K target-center real ECGs.

The locked 2026-06-18 protocol also uses this entrypoint for ECGFounder
VAE-LHAT and VAE-LHAT plus three-chain AugMix. Legacy runs keep last-checkpoint
selection; the optional matched A0/A3/A5 surface uses its explicit best policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
    build_adv_buffer_label,
    build_three_chain_vae_lhat_augmix_views as build_three_chain_vae_lhat_augmix_views_core,
    linear_warmup_value,
)
from ecg_adv_gen.adaptation.latent_hull_torch import initial_hull_latent  # noqa: E402
from ecg_adv_gen.adaptation.anchor_sampling import (  # noqa: E402
    parse_anchor_class_weight_string,
    signal_anchor_difficulty_scores,
    signal_anchor_positive_boundary_scores,
    weighted_class_quotas_with_caps,
    weighted_sample_indices,
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
    optimizer_parameter_step,
    select_matched_auxiliary_clean_examples,
    set_module_requires_grad,
    stream_weighted_masked_bce,
    summarize_fullft_adv_epoch_diagnostics,
    train_matched_auxiliary_epoch,
)
from ecg_adv_gen.training.resume_contract import (  # noqa: E402
    LOCKED_LATENT_AUGMIX_SIGNAL_SPACE,
    validate_resume_contract,
)
from ecg_adv_gen.run_naming import build_ecgfounder_fullft_run_leaf  # noqa: E402
from ecg_adv_gen.matched_ecgfounder import (  # noqa: E402
    MATCHED_ECGFOUNDER_CONTRACT_VERSION,
    build_matched_ecgfounder_training_record,
    build_matched_k500_contract,
    build_view_anchor_identity,
    external_repository_provenance,
    load_source_checkpoint_identity,
    should_evaluate_heldout_target,
    source_fold9_selection_result,
    validate_matched_runtime_args,
)
from ecg_adv_gen.evaluation.selection import update_matched_checkpoint_selection  # noqa: E402
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    STRESS_PROFILE_CHOICES,
    apply_corruption_sequence,
)
from methods.augmix.jsd_loss import jsd_multilabel  # noqa: E402
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latent_augmix_signal_space(
    chain_base_mode: str,
    third_chain_role: str,
    *,
    enabled: bool = True,
) -> str:
    return (
        LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
        if enabled
        and chain_base_mode == "clean_clean_third"
        and third_chain_role == "vae_lhat_adversarial_waveform"
        else "model_zscore"
    )


def validate_existing_run_signal_space(result_path: Path, current_signal_space: str) -> None:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_resume_contract(
        payload.get("config"),
        {"latent_augmix_signal_space": current_signal_space},
        allow_drift=False,
    )


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
        return self.model(ecg1000_to_ecgfounder_input(ecg_ct))


def build_signal_cache(
    items: list[dict[str, Any]],
    signal_path: Path,
    meta_path: Path,
    preprocess_policy: str,
) -> dict[str, np.ndarray]:
    if signal_path.exists() and meta_path.exists():
        return load_signal_cache(signal_path, meta_path)

    signal_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = signal_path.with_name(f"{signal_path.name}.lock")
    owns_lock = False
    while not owns_lock:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(f"pid={os.getpid()} time={time.time():.3f}\n")
            owns_lock = True
        except FileExistsError:
            if signal_path.exists() and meta_path.exists():
                return load_signal_cache(signal_path, meta_path)
            try:
                stale = time.time() - lock_path.stat().st_mtime > 6 * 3600
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            time.sleep(5.0)
    try:
        if signal_path.exists() and meta_path.exists():
            return load_signal_cache(signal_path, meta_path)

        if signal_path.exists() and not meta_path.exists():
            signal_path.unlink()

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
    finally:
        if owns_lock:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def parse_float_sequence(value: str | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


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
    return RawSignalDataset(signals, labels, indices=source_indices, target_len=TARGET_POINTS)


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
    return RawSignalDataset(signals, labels, indices=indices, target_len=TARGET_POINTS)


def build_locked_three_chain_latent_augmix_views(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    severity_profile: str,
    width: int,
    depth: int,
    alpha: float,
    ops: list[str],
    mixture_mode: str,
    mixture_prob: float,
    mixture_beta_a: float | None,
    mixture_beta_b: float | None,
    op_schedule: str,
    chain_weights: list[float] | None,
    renorm: bool,
    clip_abs: float,
    third_chain_role: str,
    chain_base_mode: str,
    adv_base_mix: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build locked ECGFounder three-chain VAE-LHAT AugMix waveform views."""

    def _apply(sig_ct: np.ndarray, op_name: str, op_severity: int, op_profile: str) -> np.ndarray:
        sig_t = torch.from_numpy(sig_ct.copy()).float()
        return apply_corruption_sequence(
            sig_t,
            op_name,
            op_severity,
            op_profile,
            base_seed=int(rng.integers(0, 2**31)),
            seed_parts=("ecgfounder_latent_augmix", op_name),
            sample_rate_hz=500.0,
            severity_profile_params=None,
        ).cpu().numpy().astype(np.float32, copy=False)

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
        third_chain_role=third_chain_role,
        chain_base_mode=chain_base_mode,
        adv_base_mix=adv_base_mix,
    )
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

    def _maybe_int(name: str, default: int | None = None) -> int | None:
        value = first.get(name, default)
        return None if value is None else int(value)

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
        "mixture_mode": str(first.get("mixture_mode", "")),
        "mixture_prob": first.get("mixture_prob"),
        "mixture_beta_a": first.get("mixture_beta_a"),
        "mixture_beta_b": first.get("mixture_beta_b"),
        "op_schedule": str(first.get("op_schedule", "")),
        "chain_base_mode": str(first.get("chain_base_mode", "")),
        "signal_space": str(first.get("signal_space", "")),
        "adv_base_mix": first.get("adv_base_mix"),
        "chain_weight_mode": str(first.get("chain_weight_mode", "")),
        "chain_weights": list(first.get("chain_weights", [])),
        "corruption_chain_count": int(first.get("corruption_chain_count", 2)),
        "adversarial_chain_count": int(first.get("adversarial_chain_count", 1)),
        "clean_anchor_control_chain_count": int(first.get("clean_anchor_control_chain_count", 0)),
        "adversarial_chain_index": _maybe_int("adversarial_chain_index", 2),
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


def train_latent_augmix_consistency_epoch(
    model: nn.Module,
    clean_signals_ct: np.ndarray,
    augmix_signals_ct: np.ndarray,
    labels_np: np.ndarray,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    trainable_params: list[nn.Parameter],
    *,
    copies: int,
    consistency_weight: float,
    bce_weight: float,
    consistency_loss: str,
    batch_size: int,
    max_batches: int = 0,
) -> dict[str, Any]:
    """EffNet-style latent-AugMix BCE/JSD consistency step for ECGFounder."""
    if copies <= 0 or (consistency_weight <= 0 and bce_weight <= 0):
        return {
            "enabled": False,
            "reason": "disabled_or_zero_weight",
            "loss": None,
            "bce_loss": None,
            "consistency_loss": None,
            "n_batches": 0,
            "n_generated": int(len(augmix_signals_ct)),
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
    expected = int(copies) * n
    if n == 0 or aug_np.shape[0] == 0:
        return {
            "enabled": True,
            "reason": "empty_views",
            "loss": None,
            "bce_loss": None,
            "consistency_loss": None,
            "n_batches": 0,
            "n_generated": int(aug_np.shape[0]),
            "copies": int(copies),
        }
    if aug_np.shape[0] != expected:
        raise ValueError(f"latent AugMix view count mismatch: got {aug_np.shape[0]}, expected {expected}")
    if labels_arr.shape[0] != n:
        raise ValueError(f"label count mismatch: got {labels_arr.shape[0]}, expected {n}")

    aug_views_np = aug_np.reshape(int(copies), n, *aug_np.shape[1:])
    labels_t = torch.from_numpy(labels_arr).float()
    order = np.arange(n)
    batch_size = max(1, int(batch_size))
    losses: list[float] = []
    bce_losses: list[float] = []
    consistency_losses: list[float] = []

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
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if consistency_loss == "soft_bce":
                with torch.no_grad():
                    clean_logits = model(ecg1000_to_ecgfounder_input(clean))
                    soft_targets = torch.sigmoid(clean_logits).detach()
                logits = model(ecg1000_to_ecgfounder_input(views))
                soft_rep = soft_targets.repeat((int(copies), 1))

                mask = (labels_rep >= 0).float()
                hard = F.binary_cross_entropy_with_logits(
                    logits,
                    labels_rep.clamp(min=0.0),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                denom = mask.sum().clamp_min(1.0)
                hard_bce = (hard * mask).sum() / denom
                direct_consistency = (
                    F.binary_cross_entropy_with_logits(logits, soft_rep, reduction="none") * mask
                ).sum() / denom
            else:
                clean_logits = model(ecg1000_to_ecgfounder_input(clean))
                logits = model(ecg1000_to_ecgfounder_input(views))
                logits_views = logits.view(int(copies), clean.shape[0], -1)

                mask_clean = (labels >= 0).float()
                clean_hard = F.binary_cross_entropy_with_logits(
                    clean_logits,
                    labels.clamp(min=0.0),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                clean_hard_bce = (clean_hard * mask_clean).sum() / mask_clean.sum().clamp_min(1.0)

                mask_rep = (labels_rep >= 0).float()
                aug_hard = F.binary_cross_entropy_with_logits(
                    logits,
                    labels_rep.clamp(min=0.0),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                aug_hard_bce = (aug_hard * mask_rep).sum() / mask_rep.sum().clamp_min(1.0)
                hard_bce = 0.5 * (clean_hard_bce + aug_hard_bce)

                jsd_terms = []
                for copy_i in range(int(copies)):
                    copy_j = (copy_i + 1) % int(copies)
                    jsd_terms.append(jsd_multilabel(clean_logits, logits_views[copy_i], logits_views[copy_j]))
                direct_consistency = torch.stack(jsd_terms).mean()

            loss = float(bce_weight) * hard_bce + float(consistency_weight) * direct_consistency
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        losses.append(float(loss.detach().item()))
        bce_losses.append(float(hard_bce.detach().item()))
        consistency_losses.append(float(direct_consistency.detach().item()))
        if max_batches > 0 and batch_i >= max_batches:
            break

    return {
        "enabled": True,
        "loss": float(np.mean(losses)) if losses else None,
        "bce_loss": float(np.mean(bce_losses)) if bce_losses else None,
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else None,
        "n_batches": int(len(losses)),
        "n_generated": int(aug_np.shape[0]),
        "copies": int(copies),
        "consistency_weight": float(consistency_weight),
        "consistency_objective": consistency_loss,
        "bce_weight": float(bce_weight),
        "max_batches": int(max_batches),
    }


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
    class_weights = parse_anchor_class_weight_string(args.anchor_class_sample_weights)
    rng = np.random.default_rng(args.seed + int(args.current_epoch) * 1009 + 97)
    quotas = weighted_class_quotas_with_caps(
        classes,
        int(args.k_anchor),
        class_weights,
        labels,
        max_repeat_per_class=int(args.anchor_class_max_repeat),
        class_to_idx=SUPER5_TO_IDX,
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
        "anchor_picked_record_ids": (
            np.asarray(anchor_pool["record_ids"]).astype(str)[picks].tolist()
            if len(picks) > 0
            else []
        ),
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
    adv_sample_weights: np.ndarray | None = None,
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
    matched = str(getattr(args, "matched_contract", "")) == MATCHED_ECGFOUNDER_CONTRACT_VERSION
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
        adv_sample_weights=adv_sample_weights,
        target_adv_fraction=float(args.target_adv_fraction) if matched else None,
        fixed_num_samples=(len(source_ds) + len(target_ds)) if matched else None,
        sampler_seed=(int(args.seed) + int(getattr(args, "current_epoch", 0)) * 1009) if matched else None,
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


def build_clean_anchor_augmix_epoch(
    target_dataset: RawSignalDataset,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    """Build no-VAE three-chain AugMix views from raw K500 anchors."""
    n = len(target_dataset)
    if n <= 0:
        raise RuntimeError("clean-anchor AugMix requires a non-empty target raw dataset")
    k_anchor = max(1, int(args.k_anchor))
    rng = np.random.default_rng(int(args.seed) + int(epoch) * 100003)
    picks = rng.choice(n, size=k_anchor, replace=k_anchor > n)
    clean = []
    labels = []
    for idx in picks:
        sig_ct, label = target_dataset[int(idx)]
        clean.append(sig_ct)
        labels.append(label)
    clean_np = np.stack(clean, axis=0).astype(np.float32, copy=False)
    labels_np = np.stack(labels, axis=0).astype(np.float32, copy=False)
    augmix_np, augmix_stats = build_locked_three_chain_latent_augmix_views(
        clean_np,
        clean_np,
        copies=args.latent_augmix_copies,
        severity=args.latent_augmix_severity,
        severity_profile=args.latent_augmix_severity_profile,
        width=args.latent_augmix_width,
        depth=args.latent_augmix_depth,
        alpha=args.latent_augmix_alpha,
        ops=list(args.latent_augmix_ops),
        mixture_mode="beta",
        mixture_prob=0.5,
        mixture_beta_a=None,
        mixture_beta_b=None,
        op_schedule="random",
        chain_weights=None,
        renorm=False,
        clip_abs=6.0,
        third_chain_role="clean_anchor_control",
        chain_base_mode=args.latent_augmix_chain_base_mode,
        adv_base_mix=float(args.latent_augmix_adv_base_mix),
        rng=rng,
    )
    labels_rep = np.tile(labels_np, (max(1, int(args.latent_augmix_copies)), 1))[: augmix_np.shape[0]]
    return {
        "signals": augmix_np.astype(np.float32, copy=False),
        "labels": labels_rep.astype(np.float32, copy=False),
        "teacher_logits": np.zeros_like(labels_rep, dtype=np.float32),
        "augmix_clean_signals": clean_np,
        "augmix_clean_labels": labels_np,
        "augmix_view_signals": augmix_np.astype(np.float32, copy=False),
        "latent_augmix_stats": augmix_stats,
        "n_adv": int(augmix_np.shape[0]),
        "delta_mean": None,
        "delta_max": None,
        "anchor_sample_stats": {
            "mode": "clean_anchor_control",
            "requested": int(k_anchor),
            "n_unique_available": int(n),
        },
    }


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
    augmix_clean_signals: list[np.ndarray] = []
    augmix_clean_labels: list[np.ndarray] = []
    vae_clean_signals: list[np.ndarray] = []
    vae_adv_signals_for_consistency: list[np.ndarray] = []
    vae_adv_labels_for_consistency: list[np.ndarray] = []
    delta_norms: list[float] = []
    batch_diagnostics: list[dict[str, np.ndarray | int]] = []
    latent_augmix_stats_batches: list[dict[str, Any]] = []
    anchor_sample_stats: dict[str, Any] = {}
    build_vae_augmix_views = (
        bool(args.enable_latent_augmix_branch)
        and args.latent_augmix_chain_base_mode != "all_clean_plus_vae_adv"
    )
    signal_space = latent_augmix_signal_space(
        args.latent_augmix_chain_base_mode,
        args.latent_augmix_third_chain_role,
        enabled=bool(args.enable_latent_augmix_branch),
    )
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
                adv_input = ecg1000_to_ecgfounder_input(x_adv_1000)
                adv_logits = victim.model(adv_input)
                batch_diagnostics.append(
                    fullft_adv_batch_diagnostics(clean_logits, init_logits, adv_logits, y)
                )
                adv_wave_1000 = (
                    pgd_gen._decode_to_ptbxl_1000_raw(z + delta)
                    if signal_space == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
                    else x_adv_1000
                )
                adv_wave_5000 = ecg1000_to_ecgfounder_input(
                    adv_wave_1000,
                    target_points=TARGET_POINTS,
                    apply_global_zscore=False,
                )
            anchor_labels_np = anchor_pool["labels"][batch_idx].astype(np.float32, copy=False)
            clean_probs_np = torch.sigmoid(clean_logits).detach().cpu().numpy().astype(np.float32)
            buffer_labels_np = np.stack(
                [
                    build_adv_buffer_label(
                        anchor_labels_np[j],
                        label_mode=args.adv_label_mode,
                        teacher_probs=clean_probs_np[j],
                        teacher_mix=args.adv_teacher_mix,
                        soft_target_floor=args.adv_soft_target_floor,
                    )
                    for j in range(anchor_labels_np.shape[0])
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            if build_vae_augmix_views:
                with torch.no_grad():
                    anchor_x_1000 = (
                        pgd_gen._decode_to_ptbxl_1000_raw(z)
                        if signal_space == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
                        else victim._ecgtwin_latent_to_ecg1000(z)
                    )
                    anchor_x_5000 = ecg1000_to_ecgfounder_input(
                        anchor_x_1000,
                        target_points=TARGET_POINTS,
                        apply_global_zscore=False,
                    )
            elif float(args.vae_adv_consistency_weight) > 0.0:
                with torch.no_grad():
                    anchor_x_1000 = victim._ecgtwin_latent_to_ecg1000(z)
                    anchor_x_5000 = ecg1000_to_ecgfounder_input(
                        anchor_x_1000,
                        target_points=TARGET_POINTS,
                        apply_global_zscore=False,
                    )
            else:
                anchor_x_1000 = None
                anchor_x_5000 = None
            if anchor_x_1000 is not None and float(args.vae_adv_consistency_weight) > 0.0:
                assert anchor_x_5000 is not None
                vae_clean_signals.append(anchor_x_5000.detach().cpu().numpy().astype(np.float32, copy=False))
                vae_adv_signals_for_consistency.append(adv_wave_5000.detach().cpu().numpy().astype(np.float32, copy=False))
                vae_adv_labels_for_consistency.append(buffer_labels_np)
            if build_vae_augmix_views:
                assert anchor_x_5000 is not None
                latent_augmix_rng = np.random.default_rng(
                    int(args.seed) + int(getattr(args, "current_epoch", 0)) * 100003 + int(start)
                )
                latent_augmix_np, latent_augmix_stats = build_locked_three_chain_latent_augmix_views(
                    anchor_x_5000.detach().cpu().numpy().astype(np.float32, copy=False),
                    adv_wave_5000.detach().cpu().numpy().astype(np.float32, copy=False),
                    copies=args.latent_augmix_copies,
                    severity=args.latent_augmix_severity,
                    severity_profile=args.latent_augmix_severity_profile,
                    width=args.latent_augmix_width,
                    depth=args.latent_augmix_depth,
                    alpha=args.latent_augmix_alpha,
                    ops=list(args.latent_augmix_ops),
                    mixture_mode="beta",
                    mixture_prob=0.5,
                    mixture_beta_a=None,
                    mixture_beta_b=None,
                    op_schedule="random",
                    chain_weights=None,
                    renorm=False,
                    clip_abs=6.0,
                    third_chain_role=args.latent_augmix_third_chain_role,
                    chain_base_mode=args.latent_augmix_chain_base_mode,
                    adv_base_mix=float(args.latent_augmix_adv_base_mix),
                    rng=latent_augmix_rng,
                )
                latent_augmix_stats["signal_space"] = signal_space
                latent_augmix_stats_batches.append(latent_augmix_stats)
                if latent_augmix_np.shape[0] > 0:
                    augmix_clean_signals.append(anchor_x_5000.detach().cpu().numpy().astype(np.float32, copy=False))
                    augmix_clean_labels.append(buffer_labels_np)
                    latent_augmix_t = torch.from_numpy(latent_augmix_np).float().to(device)
                    x_adv_for_stream = latent_augmix_t.detach().cpu().numpy().astype(np.float32)
                    adv_signals.append(x_adv_for_stream)
                    labels_rep = np.tile(
                        buffer_labels_np,
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[: latent_augmix_np.shape[0]]
                    teacher_rep = np.tile(
                        clean_logits.detach().cpu().numpy().astype(np.float32),
                        (max(1, int(args.latent_augmix_copies)), 1),
                    )[: latent_augmix_np.shape[0]]
                    adv_labels.append(labels_rep.astype(np.float32, copy=False))
                    adv_teacher_logits.append(teacher_rep.astype(np.float32, copy=False))
            else:
                x_adv_for_stream = adv_wave_5000.detach().cpu().numpy().astype(np.float32)
                adv_signals.append(x_adv_for_stream)
                adv_labels.append(buffer_labels_np)
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
            (0, 12, TARGET_POINTS),
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
    augmix_clean = (
        np.concatenate(augmix_clean_signals, axis=0).astype(np.float32)
        if augmix_clean_signals
        else np.empty((0, 12, TARGET_POINTS), dtype=np.float32)
    )
    augmix_labels = (
        np.concatenate(augmix_clean_labels, axis=0).astype(np.float32)
        if augmix_clean_labels
        else np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    )
    vae_clean = (
        np.concatenate(vae_clean_signals, axis=0).astype(np.float32)
        if vae_clean_signals
        else np.empty((0, 12, TARGET_POINTS), dtype=np.float32)
    )
    vae_adv_consistency = (
        np.concatenate(vae_adv_signals_for_consistency, axis=0).astype(np.float32)
        if vae_adv_signals_for_consistency
        else np.empty((0, 12, TARGET_POINTS), dtype=np.float32)
    )
    vae_labels_consistency = (
        np.concatenate(vae_adv_labels_for_consistency, axis=0).astype(np.float32)
        if vae_adv_labels_for_consistency
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
            "reason": "decoupled_vae_adv_stream" if args.latent_augmix_chain_base_mode == "all_clean_plus_vae_adv" else "disabled",
            "n_generated": 0,
        }
    )
    return {
        "signals": signals,
        "labels": labels,
        "teacher_logits": teacher_logits,
        "augmix_clean_signals": augmix_clean,
        "augmix_clean_labels": augmix_labels,
        "augmix_view_signals": signals if build_vae_augmix_views else np.empty((0, 12, TARGET_POINTS), dtype=np.float32),
        "vae_clean_signals": vae_clean,
        "vae_adv_signals": vae_adv_consistency,
        "vae_adv_labels": vae_labels_consistency,
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
    ap.add_argument("--matched_contract", default="")
    ap.add_argument("--comparison_arm", choices=["a0", "a3", "a5"], default="a0")
    ap.add_argument("--target_adv_fraction", type=float, default=0.0)
    ap.add_argument("--target_real_val_fraction", type=float, default=0.2)
    ap.add_argument("--target_real_val_seed", type=int, default=20260531)
    ap.add_argument("--selection_metric", choices=["macro_auprc", "macro_auroc"], default="macro_auprc")
    ap.add_argument("--source_floor_max_drop", type=float, default=0.02)
    ap.add_argument("--expected_exact_eligible_count", type=int, default=0)
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
    ap.add_argument("--latent_augmix_copies", type=int, default=1)
    ap.add_argument("--latent_augmix_width", type=int, default=3)
    ap.add_argument("--latent_augmix_depth", type=int, default=-1)
    ap.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    ap.add_argument("--latent_augmix_severity", type=int, default=5)
    ap.add_argument(
        "--latent_augmix_third_chain_role",
        choices=["vae_lhat_adversarial_waveform", "clean_anchor_control"],
        default="vae_lhat_adversarial_waveform",
        help=(
            "Ablation knob for the third AugMix chain. The default is the mainline "
            "VAE-LHAT adversarial waveform; clean_anchor_control disables VAE and "
            "uses the clean K500 anchor as the third chain."
        ),
    )
    ap.add_argument(
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
    ap.add_argument(
        "--latent_augmix_adv_base_mix",
        type=float,
        default=1.0,
        help="For one_adv/all_adv, use (1-rho)*clean + rho*VAE-adv as the corruption base.",
    )
    ap.add_argument(
        "--latent_augmix_severity_profile",
        default="standard",
        choices=[name for name in STRESS_PROFILE_CHOICES if name != "custom"],
    )
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
    ap.add_argument("--latent_augmix_consistency_weight", type=float, default=0.0)
    ap.add_argument(
        "--latent_augmix_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    ap.add_argument("--latent_augmix_bce_weight", type=float, default=0.0)
    ap.add_argument("--latent_augmix_consistency_max_batches", type=int, default=0)
    ap.add_argument(
        "--vae_adv_consistency_weight",
        type=float,
        default=0.0,
        help="Extra soft-BCE consistency from clean VAE anchor to VAE-LHAT adversarial waveform.",
    )
    ap.add_argument(
        "--latent_augmix_consistency_batch_size",
        type=int,
        default=0,
        help=(
            "Optional microbatch size for latent-AugMix BCE/JSD consistency. "
            "Defaults to --batch_size when <= 0."
        ),
    )
    ap.add_argument(
        "--adv_label_mode",
        choices=[
            "hard",
            "multi_hot_hard",
            "latent_soft",
            "mixed_soft",
            "teacher_soft",
            "latent_mixed_teacher",
        ],
        default="multi_hot_hard",
    )
    ap.add_argument("--adv_teacher_mix", type=float, default=0.4)
    ap.add_argument("--adv_soft_target_floor", type=float, default=0.0)
    ap.add_argument("--adv_weight", type=float, default=20.0)
    ap.add_argument(
        "--vae_adv_stream_sample_scale",
        type=float,
        default=1.0,
        help="Sampler multiplier for VAE hard samples when clean AugMix views share the augment stream.",
    )
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
    ap.add_argument("--hull_neighbor_distance_space", choices=["raw", "standardized"], default="raw")
    ap.add_argument("--hull_neighbor_mode", choices=["nearest", "local_random", "random"], default="nearest")
    ap.add_argument("--hull_neighbor_pool_size", type=int, default=0)
    ap.add_argument("--hull_neighbor_pool_multiplier", type=int, default=4)
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
        "--signal_cache_dir",
        default="",
        help="Optional shared directory for ECGFounder preprocessed signal caches.",
    )
    ap.add_argument(
        "--torch_num_threads",
        type=int,
        default=0,
        help="Optional torch intra-op thread cap for shared-server runs; 0 keeps PyTorch default.",
    )
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
    matched_components = validate_matched_runtime_args(args)
    is_matched = str(args.matched_contract) == MATCHED_ECGFOUNDER_CONTRACT_VERSION
    args.latent_augmix_signal_space = latent_augmix_signal_space(
        args.latent_augmix_chain_base_mode,
        args.latent_augmix_third_chain_role,
        enabled=bool(args.enable_latent_augmix_branch),
    )

    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if args.stage == "k500" and not args.ref_meta_json:
        raise ValueError("--ref_meta_json is required for --stage k500")
    if args.enable_latent_augmix_branch:
        if float(args.vae_adv_stream_sample_scale) < 0.0:
            raise ValueError("--vae_adv_stream_sample_scale must be non-negative")
        if not (0.0 <= float(args.latent_augmix_adv_base_mix) <= 1.0):
            raise ValueError("--latent_augmix_adv_base_mix must be in [0, 1]")
        if float(args.vae_adv_consistency_weight) < 0.0:
            raise ValueError("--vae_adv_consistency_weight must be non-negative")
        if (
            args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
            and args.latent_augmix_third_chain_role == "clean_anchor_control"
        ):
            raise ValueError(
                f"--latent_augmix_chain_base_mode {args.latent_augmix_chain_base_mode} "
                "cannot use clean_anchor_control"
            )
        needs_vae_augmix = (
            args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
            or (
                args.latent_augmix_chain_base_mode == "clean_clean_third"
                and args.latent_augmix_third_chain_role == "vae_lhat_adversarial_waveform"
            )
        )
        if needs_vae_augmix and not args.enable_vae_adv_stream:
            raise ValueError("--enable_latent_augmix_branch requires --enable_vae_adv_stream for VAE-LHAT AugMix")
        if args.latent_augmix_chain_base_mode == "all_clean_plus_vae_adv" and not args.enable_vae_adv_stream:
            raise ValueError("--latent_augmix_chain_base_mode all_clean_plus_vae_adv requires --enable_vae_adv_stream")
        if (
            args.latent_augmix_chain_base_mode == "all_clean"
            or (
                args.latent_augmix_third_chain_role == "clean_anchor_control"
                and args.latent_augmix_chain_base_mode != "all_clean_plus_vae_adv"
            )
        ) and args.enable_vae_adv_stream:
            raise ValueError("clean-anchor no-VAE AugMix ablation must not enable --enable_vae_adv_stream")
        if (
            args.latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
            or args.latent_augmix_third_chain_role == "clean_anchor_control"
        ) and float(args.adv_weight) <= 0.0:
            raise ValueError("clean-anchor latent AugMix requires --adv_weight > 0; it samples the augment stream")
        if int(args.latent_augmix_width) != 3:
            raise ValueError("locked latent AugMix requires exactly three chains")
        if (
            (
                float(args.latent_augmix_consistency_weight) > 0.0
                or float(args.latent_augmix_bce_weight) > 0.0
            )
            and args.latent_augmix_consistency_loss == "jsd"
            and int(args.latent_augmix_copies) < 2
        ):
            raise ValueError("--latent_augmix_consistency_loss jsd requires --latent_augmix_copies >= 2")
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
        validate_existing_run_signal_space(result_path, args.latent_augmix_signal_space)
        print(result_path.read_text())
        return

    cache_dir = Path(args.signal_cache_dir) if args.signal_cache_dir else out_dir / "cache"
    ptbxl_items, _ = build_ptbxl_items(limit=0)
    ptbxl = build_signal_cache(
        ptbxl_items,
        cache_dir / f"ptbxl_{args.preprocess_policy}.signals.npy",
        cache_dir / f"ptbxl_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )
    matched_k500: dict[str, Any] | None = None
    exact_eligibility: dict[str, Any] | None = None
    target_val_idx = np.empty(0, dtype=np.int64)
    if args.stage == "ptbxl_source":
        pn = None
        selected_ids: set[str] = set()
        target_idx = np.empty(0, dtype=np.int64)
        target_train_idx = np.empty(0, dtype=np.int64)
        target_train_ids: set[str] = set()
        target_val_ids: set[str] = set()
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

        selected_ids = load_selected_record_ids_from_meta(Path(args.ref_meta_json), args.center, strict_center=True)
        record_ids = pn["record_ids"].astype(str)
        target_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid in selected_ids], dtype=np.int64)
        if len(target_idx) != args.k:
            raise RuntimeError(
                f"{args.center}: matched {len(target_idx)} target records; requested {args.k}"
            )
        if is_matched:
            matched_k500 = build_matched_k500_contract(
                record_ids[target_idx],
                pn["labels"][target_idx],
                val_fraction=float(args.target_real_val_fraction),
                seed=int(args.target_real_val_seed),
            )
            split = matched_k500["split"]
            target_train_idx = target_idx[np.asarray(split["train_indices"], dtype=np.int64)]
            target_val_idx = target_idx[np.asarray(split["val_indices"], dtype=np.int64)]
            exact_eligibility = dict(matched_k500["exact_eligibility"])
            expected_eligible = int(args.expected_exact_eligible_count)
            if expected_eligible > 0 and int(exact_eligibility["eligible_count"]) != expected_eligible:
                raise RuntimeError(
                    f"{args.center}: exact/non-self eligible_count="
                    f"{exact_eligibility['eligible_count']}, expected {expected_eligible}"
                )
            if set(split["train_record_ids"]).intersection(split["val_record_ids"]):
                raise AssertionError("matched K500 train/validation IDs overlap")
            (run_dir / "eligible_anchor_manifest.json").write_text(
                json.dumps(exact_eligibility, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            target_train_idx = target_idx
        target_train_ids = set(str(record_ids[i]) for i in target_train_idx)
        target_val_ids = set(str(record_ids[i]) for i in target_val_idx)
        eval_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid not in selected_ids], dtype=np.int64)
        drop_eval_idx = eval_idx[pn["labels"][eval_idx].sum(axis=1) > 0]

    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    init_model_info = None
    source_checkpoint_identity: dict[str, Any] | None = None
    if args.init_model_path:
        init_model_path = Path(args.init_model_path)
        if is_matched:
            source_checkpoint_identity = load_source_checkpoint_identity(init_model_path)
        load_fullft_checkpoint(init_model_path, model, device)
        init_model_info = {
            "path": str(init_model_path),
            "type": "full_model",
            **(
                {
                    "sha256": source_checkpoint_identity["sha256"],
                    "stage": source_checkpoint_identity["stage"],
                    "selected_checkpoint": source_checkpoint_identity["selected_checkpoint"],
                }
                if source_checkpoint_identity is not None
                else {}
            ),
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
        eligible_indices = None
        if is_matched:
            if exact_eligibility is None:
                raise RuntimeError("matched VAE stream is missing exact eligibility")
            eligible_ids = set(str(value) for value in exact_eligibility["eligible_record_ids"])
            eligible_indices = [
                index
                for index, record_id in enumerate(np.asarray(anchor_pool["record_ids"]).astype(str))
                if str(record_id) in eligible_ids
            ]
            if len(eligible_indices) != int(exact_eligibility["eligible_count"]):
                raise RuntimeError("matched exact eligibility does not align with the train latent pool")
        walker = StratifiedPoolWalker(
            labels_one_hot=anchor_pool["labels"],
            classes_in_scope=anchor_pool["classes_in_scope"],
            class_to_idx=SUPER5_TO_IDX,
            seed=args.seed,
            eligible_indices=eligible_indices,
        )
        index = SameLabelLatentIndex(
            anchor_pool["latents"],
            anchor_pool["labels"],
            label_mode=args.hull_label_mode,
            seed=args.seed,
            include_self=args.hull_include_anchor,
            distance_space=args.hull_neighbor_distance_space,
            neighbor_mode=args.hull_neighbor_mode,
            neighbor_pool_size=args.hull_neighbor_pool_size,
            neighbor_pool_multiplier=args.hull_neighbor_pool_multiplier,
        )
        set_module_requires_grad(model, True)
    clean_anchor_augmix_dataset = None
    if args.enable_latent_augmix_branch and (
        args.latent_augmix_third_chain_role == "clean_anchor_control"
        or args.latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
    ):
        clean_anchor_augmix_dataset = load_target_raw_dataset(
            args.center,
            args,
            sorted(str(x) for x in target_train_ids),
        )
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
    best_metric = -float("inf")
    best_epoch = 0
    source_baseline_metrics: dict[str, Any] = {}
    best_source_floor_result: dict[str, Any] = {}
    selection_history: list[dict[str, Any]] = []
    realized_optimizer_steps = 0
    actual_param_update_steps_total = 0
    scheduler_steps = 0
    optimizer_steps_per_epoch = 0
    aggregate_stream_counts = {"source": 0, "target_clean": 0, "target_adv": 0}
    if is_matched and args.stage == "k500":
        if source_checkpoint_identity is None or matched_k500 is None or len(target_val_idx) == 0:
            raise RuntimeError("matched target run is missing source identity or internal validation split")
        source_baseline_metrics = eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            val_idx,
            args.eval_batch_size,
            device,
        )
        initial_target_val = eval_split(
            model,
            pn["signals"],
            pn["labels"],
            target_val_idx,
            args.eval_batch_size,
            device,
        )
        best_metric = float(initial_target_val[args.selection_metric])
        best_source_floor_result = update_matched_checkpoint_selection(
            best_metric=-float("inf"),
            candidate_metric=best_metric,
            source_metric=float(source_baseline_metrics[args.selection_metric]),
            source_baseline_metric=float(source_baseline_metrics[args.selection_metric]),
            source_max_drop=float(args.source_floor_max_drop),
        )
        best_source_floor_result["epoch"] = 0
        selection_history.append(dict(best_source_floor_result))
        save_fullft_checkpoint(run_dir / "best_model.pt", model)
    logs = []
    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        adv_info = {
            "signals": None,
            "labels": None,
            "teacher_logits": None,
            "sample_weights": None,
            "augmix_clean_signals": None,
            "augmix_clean_labels": None,
            "augmix_view_signals": None,
            "n_adv": 0,
            "delta_mean": None,
            "delta_max": None,
        }
        decoupled_clean_augmix_mode = (
            args.enable_latent_augmix_branch
            and args.latent_augmix_chain_base_mode == "all_clean_plus_vae_adv"
        )
        if args.enable_vae_adv_stream and args.adv_weight > 0 and args.k_anchor > 0:
            assert anchor_pool is not None and pgd_gen is not None and walker is not None and index is not None
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
            if is_matched:
                adv_info["view_anchor_identity"] = build_view_anchor_identity(
                    adv_info.get("anchor_picked_record_ids") or [],
                    train_record_ids=target_train_ids,
                    val_record_ids=target_val_ids,
                )
            if decoupled_clean_augmix_mode:
                assert clean_anchor_augmix_dataset is not None
                clean_augmix_info = build_clean_anchor_augmix_epoch(clean_anchor_augmix_dataset, args, epoch)
                adv_signals = adv_info["signals"]
                adv_labels = adv_info["labels"]
                adv_teacher = adv_info["teacher_logits"]
                clean_signals = clean_augmix_info["signals"]
                clean_labels = clean_augmix_info["labels"]
                clean_teacher = clean_augmix_info["teacher_logits"]
                adv_info["signals"] = np.concatenate([adv_signals, clean_signals], axis=0).astype(np.float32, copy=False)
                adv_info["labels"] = np.concatenate([adv_labels, clean_labels], axis=0).astype(np.float32, copy=False)
                adv_info["teacher_logits"] = np.concatenate([adv_teacher, clean_teacher], axis=0).astype(np.float32, copy=False)
                adv_info["sample_weights"] = np.concatenate(
                    [
                        np.full((adv_signals.shape[0],), float(args.vae_adv_stream_sample_scale), dtype=np.float32),
                        np.ones((clean_signals.shape[0],), dtype=np.float32),
                    ],
                    axis=0,
                )
                adv_info["n_adv"] = int(adv_info["signals"].shape[0])
                adv_info["vae_adv_n"] = int(adv_signals.shape[0])
                adv_info["clean_augmix_n"] = int(clean_signals.shape[0])
                adv_info["vae_adv_stream_sample_scale"] = float(args.vae_adv_stream_sample_scale)
                adv_info["augmix_clean_signals"] = clean_augmix_info["augmix_clean_signals"]
                adv_info["augmix_clean_labels"] = clean_augmix_info["augmix_clean_labels"]
                adv_info["augmix_view_signals"] = clean_augmix_info["augmix_view_signals"]
                augmix_stats = dict(clean_augmix_info["latent_augmix_stats"])
                augmix_stats["decoupled_vae_adv_stream"] = True
                augmix_stats["vae_adv_n"] = int(adv_signals.shape[0])
                augmix_stats["clean_augmix_n"] = int(clean_signals.shape[0])
                adv_info["latent_augmix_stats"] = augmix_stats
        elif (
            args.enable_latent_augmix_branch
            and (
                args.latent_augmix_third_chain_role == "clean_anchor_control"
                or args.latent_augmix_chain_base_mode == "all_clean"
            )
            and args.adv_weight > 0
            and args.k_anchor > 0
        ):
            assert clean_anchor_augmix_dataset is not None
            adv_info = build_clean_anchor_augmix_epoch(clean_anchor_augmix_dataset, args, epoch)
        epoch_adv_weight = linear_warmup_value(
            args.adv_weight,
            epoch,
            args.adv_weight_warmup_epochs,
            start=args.adv_weight_start,
        )
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
                adv_sample_weights=adv_info.get("sample_weights"),
                adv_weight=epoch_adv_weight,
                target_train_record_ids=target_train_ids,
            )
        train_mode_fn()
        losses = []
        clean_anchor_losses = []
        epoch_base_param_update_steps = 0
        epoch_stream_counts = {"source": 0, "target_clean": 0, "target_adv": 0}
        for x, y, stream, teacher_logits in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stream = stream.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(ecg1000_to_ecgfounder_input(x))
                loss = criterion(logits, y, stream)
                if args.adv_clean_logit_anchor_weight > 0:
                    adv_mask = stream == 2
                    if bool(adv_mask.any()):
                        clean_anchor_loss = F.mse_loss(logits[adv_mask], teacher_logits[adv_mask])
                        loss = loss + float(args.adv_clean_logit_anchor_weight) * clean_anchor_loss
                        clean_anchor_losses.append(float(clean_anchor_loss.detach().item()))
            loss.backward()
            stepped_parameter = None
            parameter_step_before = 0
            if is_matched and args.stage == "k500":
                stepped_parameter = next(
                    (
                        parameter
                        for parameter in trainable_params
                        if parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        and bool(torch.count_nonzero(parameter.grad).item())
                    ),
                    None,
                )
                if stepped_parameter is None:
                    raise RuntimeError("base matched optimizer step has no finite nonzero gradient")
                parameter_step_before = optimizer_parameter_step(opt, stepped_parameter)
            opt.step()
            if stepped_parameter is not None:
                if optimizer_parameter_step(opt, stepped_parameter) <= parameter_step_before:
                    raise RuntimeError("base matched optimizer parameter step did not advance")
                epoch_base_param_update_steps += 1
            losses.append(float(loss.item()))
            epoch_stream_counts["source"] += int((stream == 0).sum().item())
            epoch_stream_counts["target_clean"] += int((stream == 1).sum().item())
            epoch_stream_counts["target_adv"] += int((stream == 2).sum().item())
        augmix_clean = adv_info.get("augmix_clean_signals")
        augmix_views = adv_info.get("augmix_view_signals")
        if augmix_views is None:
            augmix_views = adv_info.get("signals")
        matched_auxiliary_exposure: dict[str, Any] | None = None
        if is_matched and args.stage == "k500":
            assert pn is not None
            matched_auxiliary_exposure = select_matched_auxiliary_clean_examples(
                signals=pn["signals"],
                labels=pn["labels"],
                record_ids=pn["record_ids"],
                train_record_ids=target_train_ids,
                val_record_ids=target_val_ids,
                n_samples=int(args.k_anchor),
                seed=int(args.seed),
                epoch=int(epoch),
            )
            use_raw_views = bool(matched_components and matched_components.raw_augmix)
            if use_raw_views and (
                augmix_clean is None
                or augmix_views is None
                or adv_info.get("augmix_clean_labels") is None
            ):
                raise RuntimeError("matched A5 did not produce its declared raw augmented views")
            latent_augmix_consistency_stats = train_matched_auxiliary_epoch(
                model=model,
                clean_signals_ct=matched_auxiliary_exposure["signals"],
                labels_np=matched_auxiliary_exposure["labels"],
                optimizer=opt,
                pos_weight=pos_weight,
                device=device,
                trainable_params=trainable_params,
                batch_size=int(args.batch_size),
                max_batches=int(args.latent_augmix_consistency_max_batches),
                view_clean_signals_ct=augmix_clean if use_raw_views else None,
                augmix_signals_ct=augmix_views if use_raw_views else None,
                view_labels_np=(
                    adv_info.get("augmix_clean_labels") if use_raw_views else None
                ),
                copies=int(args.latent_augmix_copies) if use_raw_views else 0,
                view_bce_weight=(
                    float(args.latent_augmix_bce_weight) if use_raw_views else 0.0
                ),
                consistency_weight=(
                    float(args.latent_augmix_consistency_weight) if use_raw_views else 0.0
                ),
            )
            latent_augmix_consistency_stats["clean_record_ids_sha256"] = (
                matched_auxiliary_exposure["record_ids_sha256"]
            )
            latent_augmix_consistency_stats["clean_val_overlap_count"] = (
                matched_auxiliary_exposure["val_overlap_count"]
            )
        elif (
            args.enable_latent_augmix_branch
            and augmix_clean is not None
            and augmix_views is not None
        ):
            latent_augmix_consistency_stats = train_latent_augmix_consistency_epoch(
                model=model,
                clean_signals_ct=augmix_clean,
                augmix_signals_ct=augmix_views,
                labels_np=adv_info.get("augmix_clean_labels"),
                optimizer=opt,
                pos_weight=pos_weight,
                device=device,
                trainable_params=trainable_params,
                copies=args.latent_augmix_copies,
                consistency_weight=args.latent_augmix_consistency_weight,
                bce_weight=args.latent_augmix_bce_weight,
                consistency_loss=args.latent_augmix_consistency_loss,
                batch_size=(
                    args.latent_augmix_consistency_batch_size
                    if int(args.latent_augmix_consistency_batch_size) > 0
                    else args.batch_size
                ),
                max_batches=args.latent_augmix_consistency_max_batches,
            )
        else:
            latent_augmix_consistency_stats = {
                "enabled": False,
                "reason": "no_latent_augmix_views_this_epoch",
                "loss": None,
                "bce_loss": None,
                "consistency_loss": None,
                "n_batches": 0,
                "n_generated": 0,
            }
        vae_adv_consistency_stats = {
            "enabled": False,
            "reason": "disabled_or_no_vae_adv",
            "loss": None,
            "bce_loss": None,
            "consistency_loss": None,
            "n_batches": 0,
            "n_generated": 0,
        }
        vae_clean = adv_info.get("vae_clean_signals")
        vae_adv = adv_info.get("vae_adv_signals")
        vae_labels = adv_info.get("vae_adv_labels")
        if (
            not (is_matched and args.stage == "k500")
            and
            float(args.vae_adv_consistency_weight) > 0.0
            and vae_clean is not None
            and vae_adv is not None
            and vae_labels is not None
            and vae_adv.shape[0] > 0
            and vae_clean.shape[0] == vae_adv.shape[0]
        ):
            vae_adv_consistency_stats = train_latent_augmix_consistency_epoch(
                model=model,
                clean_signals_ct=vae_clean,
                augmix_signals_ct=vae_adv,
                labels_np=vae_labels,
                optimizer=opt,
                pos_weight=pos_weight,
                device=device,
                trainable_params=trainable_params,
                copies=1,
                consistency_weight=args.vae_adv_consistency_weight,
                bce_weight=0.0,
                consistency_loss="soft_bce",
                batch_size=(
                    args.latent_augmix_consistency_batch_size
                    if int(args.latent_augmix_consistency_batch_size) > 0
                    else args.batch_size
                ),
                max_batches=args.latent_augmix_consistency_max_batches,
            )
        if is_matched and args.stage == "k500":
            assert matched_components is not None
            auxiliary_budget = math.ceil(int(args.k_anchor) / int(args.batch_size))
            if int(args.latent_augmix_consistency_max_batches) > 0:
                auxiliary_budget = min(auxiliary_budget, int(args.latent_augmix_consistency_max_batches))
            if int(latent_augmix_consistency_stats["n_batches"]) != auxiliary_budget:
                raise RuntimeError(
                    "matched auxiliary optimizer-step budget drift: "
                    f"realized={latent_augmix_consistency_stats['n_batches']} expected={auxiliary_budget}"
                )
            if epoch_base_param_update_steps != len(train_loader):
                raise RuntimeError("matched base optimizer parameter-step count drift")
            if int(latent_augmix_consistency_stats["actual_param_update_steps"]) != auxiliary_budget:
                raise RuntimeError("matched auxiliary optimizer calls were not real parameter updates")
            realized_epoch_steps = epoch_base_param_update_steps + int(
                latent_augmix_consistency_stats["actual_param_update_steps"]
            )
            expected_epoch_steps = len(train_loader) + auxiliary_budget
            if realized_epoch_steps != expected_epoch_steps:
                raise RuntimeError(
                    f"matched optimizer-step budget drift: {realized_epoch_steps} != {expected_epoch_steps}"
                )
            if optimizer_steps_per_epoch not in {0, expected_epoch_steps}:
                raise RuntimeError("matched base sampler/optimizer-step budget changed between epochs")
            optimizer_steps_per_epoch = expected_epoch_steps
            realized_optimizer_steps += realized_epoch_steps
            actual_param_update_steps_total += realized_epoch_steps
            for key in aggregate_stream_counts:
                aggregate_stream_counts[key] += int(epoch_stream_counts[key])
        sched.step()
        scheduler_steps += 1
        val_metrics = eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            val_idx,
            args.eval_batch_size,
            device,
        )
        target_val_metrics = (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                target_val_idx,
                args.eval_batch_size,
                device,
            )
            if is_matched and args.stage == "k500"
            else None
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
            if pn is not None and should_evaluate_heldout_target(args.matched_contract)
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
            if pn is not None and should_evaluate_heldout_target(args.matched_contract)
            else None
        )
        source_floor_result: dict[str, Any] = {}
        if is_matched and args.stage == "ptbxl_source":
            candidate_metric = float(val_metrics[args.selection_metric])
            source_floor_result = source_fold9_selection_result(
                best_metric=best_metric,
                candidate_fold9_metric=candidate_metric,
                epoch=epoch,
            )
            selection_history.append(dict(source_floor_result))
            if bool(source_floor_result["selected"]):
                best_metric = candidate_metric
                best_epoch = epoch
                best_source_floor_result = dict(source_floor_result)
                save_fullft_checkpoint(run_dir / "best_model.pt", model)
        elif is_matched and args.stage == "k500":
            assert target_val_metrics is not None and source_baseline_metrics
            source_floor_result = update_matched_checkpoint_selection(
                best_metric=best_metric,
                candidate_metric=float(target_val_metrics[args.selection_metric]),
                source_metric=float(val_metrics[args.selection_metric]),
                source_baseline_metric=float(source_baseline_metrics[args.selection_metric]),
                source_max_drop=float(args.source_floor_max_drop),
            )
            source_floor_result["epoch"] = epoch
            selection_history.append(dict(source_floor_result))
            if bool(source_floor_result["selected"]):
                best_metric = float(source_floor_result["candidate_metric"])
                best_epoch = epoch
                best_source_floor_result = dict(source_floor_result)
                save_fullft_checkpoint(run_dir / "best_model.pt", model)
        target_total = epoch_stream_counts["target_clean"] + epoch_stream_counts["target_adv"]
        realized_target_adv_fraction = (
            float(epoch_stream_counts["target_adv"] / target_total) if target_total else 0.0
        )
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "target_internal_val_macro_auroc": (
                None if target_val_metrics is None else target_val_metrics["macro_auroc"]
            ),
            "target_internal_val_macro_auprc": (
                None if target_val_metrics is None else target_val_metrics["macro_auprc"]
            ),
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
            "atk_anchor": {
                "bce_mean": adv_info.get("clean_bce_mean"),
                "loss_gain_to_adversarial_mean": adv_info.get("loss_gain_mean"),
            },
            "atk_init": {
                "bce_mean": adv_info.get("init_bce_mean"),
                "loss_gain_to_adversarial_mean": adv_info.get("init_loss_gain_mean"),
            },
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
            "effective_anchor_count": adv_info.get("anchor_picked_unique"),
            "view_anchor_identity": adv_info.get("view_anchor_identity"),
            "adv_anchor_picked_class_counts": adv_info.get("anchor_picked_class_counts"),
            "adv_anchor_score_mean": adv_info.get("anchor_score_mean"),
            "adv_anchor_score_p90": adv_info.get("anchor_score_p90"),
            "adv_anchor_score_max": adv_info.get("anchor_score_max"),
            "adv_anchor_sample_ess": adv_info.get("anchor_sample_ess"),
            "latent_augmix_stats": adv_info.get("latent_augmix_stats"),
            "latent_augmix_consistency_stats": latent_augmix_consistency_stats,
            "vae_adv_consistency_stats": vae_adv_consistency_stats,
            "latent_augmix_consistency_loss": latent_augmix_consistency_stats.get("loss"),
            "latent_augmix_consistency_bce_loss": latent_augmix_consistency_stats.get("bce_loss"),
            "latent_augmix_consistency_objective_loss": latent_augmix_consistency_stats.get("consistency_loss"),
            "source_floor_result": source_floor_result or None,
            "realized_stream_counts": dict(epoch_stream_counts),
            "realized_target_adv_fraction": realized_target_adv_fraction,
            "realized_optimizer_steps": (
                len(train_loader)
                + int(latent_augmix_consistency_stats["n_batches"])
                + int(vae_adv_consistency_stats["n_batches"])
            ),
            "realized_optimizer_steps_total": int(realized_optimizer_steps),
            "actual_param_update_steps": (
                epoch_base_param_update_steps
                + int(latent_augmix_consistency_stats.get("actual_param_update_steps", 0))
                if is_matched and args.stage == "k500"
                else None
            ),
            "actual_param_update_steps_total": (
                int(actual_param_update_steps_total)
                if is_matched and args.stage == "k500"
                else None
            ),
            "matched_auxiliary_clean_exposure": (
                {
                    "record_ids_sha256": matched_auxiliary_exposure["record_ids_sha256"],
                    "n_samples": matched_auxiliary_exposure["n_samples"],
                    "val_overlap_count": matched_auxiliary_exposure["val_overlap_count"],
                    "selection_seed": matched_auxiliary_exposure["selection_seed"],
                    "epoch": matched_auxiliary_exposure["epoch"],
                }
                if matched_auxiliary_exposure is not None
                else None
            ),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "scheduler_steps": int(scheduler_steps),
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
            )
            + (
                f" latmix_cons={entry['latent_augmix_consistency_loss']:.4f}"
                if entry["latent_augmix_consistency_loss"] is not None
                else ""
            ),
            flush=True,
        )
        save_fullft_checkpoint(run_dir / "last_model.pt", model)

    selected_checkpoint_name = "best_model.pt" if is_matched else "last_model.pt"
    load_fullft_checkpoint(run_dir / selected_checkpoint_name, model, device)
    last_epoch = logs[-1]["epoch"] if logs else None
    if args.latent_augmix_chain_base_mode == "one_adv":
        latent_augmix_chain_roles = [
            "clean_anchor_corruption",
            "clean_anchor_corruption",
            "vae_lhat_adversarial_corruption",
        ]
    elif args.latent_augmix_chain_base_mode in {"all_clean", "all_adv", "all_clean_plus_vae_adv"}:
        latent_augmix_chain_roles = [f"{args.latent_augmix_chain_base_mode}_corruption"] * 3
    else:
        latent_augmix_chain_roles = ["corruption", "corruption", str(args.latent_augmix_third_chain_role)]
    if args.latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}:
        clean_anchor_chain_count = 3
    elif args.latent_augmix_chain_base_mode == "one_adv":
        clean_anchor_chain_count = 2
    elif args.latent_augmix_third_chain_role == "clean_anchor_control":
        clean_anchor_chain_count = 1
    else:
        clean_anchor_chain_count = 0
    result = {
        "method": (
            "ECGFounder official-style full fine-tuning"
            + (" + VAE-only real-anchor latent-hull online AT" if args.enable_vae_adv_stream else "")
            + (
                " + VAE supervised hard-sample stream with clean-anchor three-chain AugMix"
                if args.enable_latent_augmix_branch
                and args.latent_augmix_chain_base_mode == "all_clean_plus_vae_adv"
                else ""
            )
            + (
                " + locked three-chain VAE-LHAT AugMix"
                if args.enable_latent_augmix_branch
                and args.latent_augmix_chain_base_mode not in {"all_clean", "all_clean_plus_vae_adv"}
                and (
                    args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
                    or args.latent_augmix_third_chain_role == "vae_lhat_adversarial_waveform"
                )
                else ""
            )
            + (
                " + locked three-chain clean-anchor AugMix noVAE ablation"
                if args.enable_latent_augmix_branch
                and args.latent_augmix_chain_base_mode != "all_clean_plus_vae_adv"
                and (
                    args.latent_augmix_third_chain_role == "clean_anchor_control"
                    or args.latent_augmix_chain_base_mode == "all_clean"
                )
                else ""
            )
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
        "checkpoint_policy": (
            "best_fold9_source_val"
            if is_matched and args.stage == "ptbxl_source"
            else "k500_internal_val_plus_source_floor"
            if is_matched
            else "last"
        ),
        "selected_checkpoint": selected_checkpoint_name,
        "selection": (
            {
                "metric": str(args.selection_metric),
                "selection_data": "ptbxl_fold9",
                "report_only_data": ["ptbxl_fold10"],
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "history": selection_history,
            }
            if is_matched and args.stage == "ptbxl_source"
            else {
                "metric": str(args.selection_metric),
                "selection_data": "target_k500_internal_val",
                "source_floor_data": "ptbxl_fold9",
                "forbidden_data": ["pn2021_heldout"],
                "source_floor_max_drop": float(args.source_floor_max_drop),
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
                "source_floor_result": best_source_floor_result,
                "history": selection_history,
            }
            if is_matched
            else None
        ),
        "last_epoch": last_epoch,
        "ptbxl_fold9": eval_split(
            model,
            ptbxl["signals"],
            ptbxl["labels"],
            val_idx,
            args.eval_batch_size,
            device,
        ),
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
            if pn is not None and should_evaluate_heldout_target(args.matched_contract)
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
            if pn is not None and should_evaluate_heldout_target(args.matched_contract)
            else None
        ),
        "target_internal_val": (
            eval_split(
                model,
                pn["signals"],
                pn["labels"],
                target_val_idx,
                args.eval_batch_size,
                device,
            )
            if is_matched and pn is not None
            else None
        ),
        "n_target_eval": 0 if is_matched else int(len(eval_idx)),
        "n_target_drop_all_zero_eval": 0 if is_matched else int(len(drop_eval_idx)),
        "config": vars(args),
        "init_model": init_model_info,
        "matched_contract": MATCHED_ECGFOUNDER_CONTRACT_VERSION if is_matched else None,
        "comparison_arm": args.comparison_arm if is_matched and args.stage == "k500" else None,
        "source_checkpoint": source_checkpoint_identity,
        "k500_split": None if matched_k500 is None else matched_k500["split"],
        "exact_eligibility": exact_eligibility,
        "optimizer_budget": {
            "epochs": int(args.epochs),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "realized_optimizer_steps": int(realized_optimizer_steps),
            "actual_param_update_steps": int(actual_param_update_steps_total),
            "scheduler_steps": int(scheduler_steps),
        },
        "matched_auxiliary_clean_exposure": [
            entry["matched_auxiliary_clean_exposure"]
            for entry in logs
            if entry.get("matched_auxiliary_clean_exposure") is not None
        ],
        "matched_view_anchor_exposure": [
            {
                "epoch": entry["epoch"],
                **entry["view_anchor_identity"],
            }
            for entry in logs
            if entry.get("view_anchor_identity") is not None
        ],
        "realized_stream_counts": dict(aggregate_stream_counts),
        "latent_augmix_branch": {
            "enabled": bool(args.enable_latent_augmix_branch),
            "topology": "locked_three_chain_vae_lhat_augmix",
            "chain_roles": latent_augmix_chain_roles if args.enable_latent_augmix_branch else [],
            "chain_base_mode": str(args.latent_augmix_chain_base_mode),
            "signal_space": str(args.latent_augmix_signal_space),
            "adv_base_mix": float(args.latent_augmix_adv_base_mix),
            "adversarial_chain_corrupted": (
                args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
                if args.enable_latent_augmix_branch
                and (
                    args.latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
                    or args.latent_augmix_third_chain_role == "vae_lhat_adversarial_waveform"
                )
                else None
            ),
            "clean_anchor_control_chain_count": clean_anchor_chain_count if args.enable_latent_augmix_branch else 0,
            "copies": int(args.latent_augmix_copies),
            "width": int(args.latent_augmix_width),
            "depth": int(args.latent_augmix_depth),
            "alpha": float(args.latent_augmix_alpha),
            "severity": int(args.latent_augmix_severity),
            "severity_profile": str(args.latent_augmix_severity_profile),
            "severity_params_file": "",
            "severity_params_name": "",
            "mixture_mode": "beta",
            "mixture_prob": 0.5,
            "mixture_beta_a": None,
            "mixture_beta_b": None,
            "op_schedule": "random",
            "chain_weights": None,
            "ops": list(args.latent_augmix_ops),
            "renorm": False,
            "clip_abs": 6.0,
            "adv_label_mode": str(args.adv_label_mode),
            "adv_teacher_mix": float(args.adv_teacher_mix),
            "adv_soft_target_floor": float(args.adv_soft_target_floor),
            "consistency": {
                "enabled": bool(
                    args.enable_latent_augmix_branch
                    and (
                        float(args.latent_augmix_consistency_weight) > 0.0
                        or float(args.latent_augmix_bce_weight) > 0.0
                    )
                ),
                "consistency_weight": float(args.latent_augmix_consistency_weight),
                "consistency_loss": str(args.latent_augmix_consistency_loss),
                "bce_weight": float(args.latent_augmix_bce_weight),
                "max_batches": int(args.latent_augmix_consistency_max_batches),
                "vae_adv_consistency_weight": float(args.vae_adv_consistency_weight),
                "batch_size": (
                    int(args.latent_augmix_consistency_batch_size)
                    if int(args.latent_augmix_consistency_batch_size) > 0
                    else int(args.batch_size)
                ),
            },
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
            "anchor_class_sample_weights": parse_anchor_class_weight_string(args.anchor_class_sample_weights),
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
    external_provenance = {
        "ecgfounder": external_repository_provenance(ECGFOUNDER_ROOT),
        "ecgtwin": external_repository_provenance(
            os.environ.get("ECGTWIN_ROOT", str(REPO_ROOT / "model" / "ECGTwin"))
        ),
    }
    result["external_repositories"] = external_provenance
    (run_dir / "external_repo_provenance.json").write_text(
        json.dumps(external_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if is_matched:
        selection_record = {
            "contract": MATCHED_ECGFOUNDER_CONTRACT_VERSION,
            "selected_checkpoint": selected_checkpoint_name,
            "checkpoint_policy": result["checkpoint_policy"],
            "selection": result["selection"],
        }
        (run_dir / "selection.json").write_text(
            json.dumps(selection_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if is_matched and args.stage == "k500":
        assert source_checkpoint_identity is not None and matched_k500 is not None and exact_eligibility is not None
        target_total = aggregate_stream_counts["target_clean"] + aggregate_stream_counts["target_adv"]
        aggregate_rho = (
            float(aggregate_stream_counts["target_adv"] / target_total) if target_total else 0.0
        )
        matched_record = build_matched_ecgfounder_training_record(
            comparison_arm=args.comparison_arm,
            source_checkpoint=source_checkpoint_identity,
            split=matched_k500["split"],
            exact_eligibility=exact_eligibility,
            selection_metric=args.selection_metric,
            source_floor_max_drop=args.source_floor_max_drop,
            source_floor_result=best_source_floor_result,
            epochs=args.epochs,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            realized_optimizer_steps=realized_optimizer_steps,
            actual_param_update_steps=actual_param_update_steps_total,
            scheduler_steps=scheduler_steps,
            realized_stream_counts=aggregate_stream_counts,
            realized_target_adv_fraction=aggregate_rho,
            auxiliary_clean_exposure=result["matched_auxiliary_clean_exposure"],
            view_anchor_exposure=result["matched_view_anchor_exposure"],
        )
        result["matched_training_record"] = matched_record
        (run_dir / "matched_training_record.json").write_text(
            json.dumps(matched_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
