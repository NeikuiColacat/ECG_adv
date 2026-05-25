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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
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
from scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 import real_anchor_base  # noqa: E402
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    CHECKPOINT,
    PTBXL_CSV,
    compute_metrics,
    preprocess_record,
    build_pn2021_items,
    build_ptbxl_items,
)
from scripts.pgd_cross_center.synth_online_at_super5 import SameLabelLatentIndex, StratifiedPoolWalker  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, SUPER5_TO_IDX  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_fullft_super5_20260523"
CENTER_DEFAULT = "cpsc_2018"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tag_value(value: float | int | str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


class CachedSignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_i = int(self.indices[idx])
        x = np.asarray(self.signals[src_i], dtype=np.float32).copy()
        y = np.asarray(self.labels[src_i], dtype=np.float32).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class MemorySignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.signals[idx]), torch.from_numpy(self.labels[idx])


class TaggedCachedSignalDataset(Dataset):
    def __init__(self, base: CachedSignalDataset, stream_id: int, num_classes: int) -> None:
        self.base = base
        self.stream_id = int(stream_id)
        self.teacher_logits = torch.zeros((num_classes,), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return x, y, torch.tensor(self.stream_id, dtype=torch.long), self.teacher_logits.clone()


class TaggedMemorySignalDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        stream_id: int,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.stream_id = int(stream_id)
        if teacher_logits is None:
            teacher_logits = np.zeros_like(self.labels, dtype=np.float32)
        teacher_logits = np.asarray(teacher_logits, dtype=np.float32)
        if teacher_logits.shape != self.labels.shape:
            raise ValueError(
                f"teacher_logits shape {teacher_logits.shape} does not match labels {self.labels.shape}"
            )
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.tensor(self.stream_id, dtype=torch.long),
            torch.from_numpy(self.teacher_logits[idx]),
        )


def global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)


def ecg1000_to_ecgfounder_input(ecg_ct_1000: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(ecg_ct_1000, size=TARGET_POINTS, mode="linear", align_corners=True)
    return global_zscore(x)


class ECGFounderFullFTVictim(nn.Module):
    """Differentiable ECGTwin-latent -> ECGFounder full model logits wrapper."""

    def __init__(self, model: nn.Module, ecgtwin: ECGTwinWrapper) -> None:
        super().__init__()
        self.model = model
        self.ecgtwin = ecgtwin
        self.num_classes = len(CLASS_NAMES_SUPER5)

    @staticmethod
    def _global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return global_zscore(x, eps=eps)

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
        return {
            "signals": np.load(signal_path, mmap_mode="r"),
            **{k: v for k, v in np.load(meta_path, allow_pickle=True).items()},
        }

    signal_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        signal_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(items), 12, TARGET_POINTS),
    )
    labels = np.empty((len(items), 5), dtype=np.float32)
    centers, record_ids, folds = [], [], []
    for i, item in enumerate(tqdm(items, desc=f"cache {signal_path.name}")):
        arr[i] = preprocess_record(item["path"], preprocess_policy=preprocess_policy)
        labels[i] = np.asarray(item["label"], dtype=np.float32)
        centers.append(str(item.get("center", "")))
        record_ids.append(str(item.get("record_id", "")))
        folds.append(int(item.get("strat_fold", -1)))
    arr.flush()
    np.savez_compressed(
        meta_path,
        labels=labels,
        centers=np.asarray(centers, dtype=str),
        record_ids=np.asarray(record_ids, dtype=str),
        folds=np.asarray(folds, dtype=np.int64),
        preprocess_policy=np.asarray([preprocess_policy], dtype=str),
        input_shape=np.asarray([12, TARGET_POINTS], dtype=np.int64),
        lead_order=np.asarray(EXPECTED_LEADS, dtype=str),
    )
    return {
        "signals": np.load(signal_path, mmap_mode="r"),
        **{k: v for k, v in np.load(meta_path, allow_pickle=True).items()},
    }


def load_selected_ref_ids(ref_meta_json: Path, center: str) -> set[str]:
    with ref_meta_json.open() as f:
        meta = json.load(f)
    out = set()
    if isinstance(meta, list):
        for row in meta:
            if str(row.get("center", center)) == center:
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    out.add(str(rid))
    elif isinstance(meta, dict):
        for key in ("record_ids", "ref_record_ids", "selected_ref_record_ids"):
            if key in meta:
                out.update(str(x) for x in meta[key])
        if not out and "items" in meta:
            for row in meta["items"]:
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    out.add(str(rid))
    if not out:
        raise RuntimeError(f"could not parse selected ref ids from {ref_meta_json}")
    return out


def init_dense_from_head(model: nn.Module, head_path: Path) -> dict[str, Any]:
    head_state = torch.load(head_path, map_location="cpu")
    if not isinstance(head_state, dict) or "weight" not in head_state or "bias" not in head_state:
        raise RuntimeError(f"{head_path} is not a simple linear-head state dict with weight/bias")
    if not hasattr(model, "dense"):
        raise RuntimeError("ECGFounder model has no dense attribute for Super5 head initialization")
    dense = getattr(model, "dense")
    if not isinstance(dense, nn.Linear):
        raise RuntimeError(f"model.dense is {type(dense)!r}, expected nn.Linear")
    weight = head_state["weight"].detach().to(dtype=dense.weight.dtype)
    bias = head_state["bias"].detach().to(dtype=dense.bias.dtype)
    if tuple(weight.shape) != tuple(dense.weight.shape) or tuple(bias.shape) != tuple(dense.bias.shape):
        raise RuntimeError(
            f"head shape mismatch: got weight={tuple(weight.shape)} bias={tuple(bias.shape)}, "
            f"expected weight={tuple(dense.weight.shape)} bias={tuple(dense.bias.shape)}"
        )
    with torch.no_grad():
        dense.weight.copy_(weight.to(device=dense.weight.device))
        dense.bias.copy_(bias.to(device=dense.bias.device))
    return {
        "path": str(head_path),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
    }


def split_target_train_val(
    target_idx: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    val_count: int,
    split_seed: int,
    split_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    target_idx = np.asarray(target_idx, dtype=np.int64)
    if val_count <= 0:
        train_ids = {str(record_ids[i]) for i in target_idx}
        return target_idx, np.empty(0, dtype=np.int64), train_ids, set()
    if val_count >= len(target_idx):
        raise ValueError(f"target_val_count={val_count} must be smaller than target K={len(target_idx)}")
    rng = np.random.default_rng(split_seed + 1701)
    perm = np.asarray(target_idx, dtype=np.int64).copy()
    rng.shuffle(perm)
    if split_mode == "random":
        val_idx = np.sort(perm[:val_count])
    elif split_mode == "stratified":
        y = np.asarray(labels, dtype=np.float32)[target_idx] > 0.5
        pos_counts = y.sum(axis=0)
        selected: list[int] = []
        selected_set: set[int] = set()
        class_order = [
            int(c)
            for c in np.argsort(pos_counts)
            if int(pos_counts[int(c)]) >= 2
        ]
        for class_i in class_order:
            candidates = [int(i) for i, row in zip(target_idx, y) if row[class_i] and int(i) not in selected_set]
            if not candidates:
                continue
            candidate_scores = []
            for idx in candidates:
                row = np.asarray(labels[idx], dtype=np.float32) > 0.5
                covered_new = sum(
                    1
                    for c in class_order
                    if row[c] and not any(np.asarray(labels[j], dtype=np.float32)[c] > 0.5 for j in selected)
                )
                rarity = float(np.sum([1.0 / max(float(pos_counts[c]), 1.0) for c, present in enumerate(row) if present]))
                candidate_scores.append((-covered_new, -rarity, rng.random(), idx))
            candidate_scores.sort()
            pick = int(candidate_scores[0][3])
            selected.append(pick)
            selected_set.add(pick)
            if len(selected) >= val_count:
                break
        for idx in perm:
            idx = int(idx)
            if len(selected) >= val_count:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
        val_idx = np.sort(np.asarray(selected, dtype=np.int64))
    else:
        raise ValueError(f"unknown target_val_split_mode={split_mode}")
    train_idx = np.sort(perm[val_count:])
    if split_mode == "stratified":
        val_set = set(int(i) for i in val_idx)
        train_idx = np.sort(np.asarray([int(i) for i in target_idx if int(i) not in val_set], dtype=np.int64))
    train_ids = {str(record_ids[i]) for i in train_idx}
    val_ids = {str(record_ids[i]) for i in val_idx}
    return train_idx, val_idx, train_ids, val_ids


def load_anchor_pool_for_ids(
    center: str,
    selected_ids: set[str],
    pn_payload: dict[str, np.ndarray],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    base = real_anchor_base(center, args)
    with np.load(base.with_suffix(".latent.npz"), allow_pickle=True) as d:
        latents_all = d["latents"].astype(np.float32)
        record_ids_all = d["record_ids"].astype(str)

    want = {str(x) for x in selected_ids}
    rid_to_i = {str(rid): i for i, rid in enumerate(record_ids_all)}
    missing = sorted(want - set(rid_to_i))
    if missing:
        raise RuntimeError(
            f"{center}: {len(missing)} selected ids are missing from {base.with_suffix('.latent.npz')}; "
            f"first missing={missing[:5]}"
        )
    chosen = np.asarray([rid_to_i[rid] for rid in sorted(want)], dtype=np.int64)

    pn_record_ids = pn_payload["record_ids"].astype(str)
    pn_labels = pn_payload["labels"].astype(np.float32)
    label_by_rid = {str(rid): pn_labels[i] for i, rid in enumerate(pn_record_ids)}
    signal_index_by_rid = {str(rid): i for i, rid in enumerate(pn_record_ids)}
    labels = []
    signals = []
    for rid in record_ids_all[chosen]:
        if str(rid) not in label_by_rid:
            raise RuntimeError(f"{center}: selected id {rid} missing from PN cache labels")
        if str(rid) not in signal_index_by_rid:
            raise RuntimeError(f"{center}: selected id {rid} missing from PN cache signals")
        labels.append(label_by_rid[str(rid)])
        signals.append(np.asarray(pn_payload["signals"][signal_index_by_rid[str(rid)]], dtype=np.float32))
    labels_arr = np.stack(labels).astype(np.float32)
    signals_arr = np.stack(signals).astype(np.float32)
    present = labels_arr.sum(axis=0) > 0
    classes_in_scope = [c for c, ok in zip(CLASS_NAMES_SUPER5, present) if ok]
    return {
        "latents": latents_all[chosen],
        "labels": labels_arr,
        "signals": signals_arr,
        "record_ids": record_ids_all[chosen],
        "classes_in_scope": classes_in_scope,
        "label_counts": dict(zip(CLASS_NAMES_SUPER5, labels_arr.sum(axis=0).astype(int).tolist())),
        "source_base": str(base),
    }


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


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(flag)


def parse_class_weight_string(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = str(raw or "").strip()
    if not raw:
        return out
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"class weight entry must be NAME=VALUE, got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in SUPER5_TO_IDX:
            raise ValueError(f"unknown class in --anchor_class_sample_weights: {name!r}")
        out[name] = float(value)
    return out


def weighted_class_quotas(
    classes: list[str],
    k_total: int,
    class_weights: dict[str, float],
    labels: np.ndarray,
    max_repeat_per_class: int = 0,
) -> dict[str, int]:
    active = []
    for cls in classes:
        cls_i = SUPER5_TO_IDX[cls]
        count = int((labels[:, cls_i] > 0.5).sum())
        weight = float(class_weights.get(cls, 1.0))
        if count > 0 and weight > 0.0:
            cap = int(k_total)
            if int(max_repeat_per_class) > 0:
                cap = max(1, int(count) * int(max_repeat_per_class))
            active.append((cls, weight, cap))
    if k_total <= 0 or not active:
        return {cls: 0 for cls in classes}

    weights = np.asarray([w for _, w, _ in active], dtype=np.float64)
    weights = weights / weights.sum()
    raw = weights * float(k_total)
    quotas = np.floor(raw).astype(np.int64)
    if k_total >= len(active):
        quotas = np.maximum(quotas, 1)
    while int(quotas.sum()) > k_total:
        i = int(np.argmax(quotas - raw))
        if quotas[i] > 0:
            quotas[i] -= 1
        else:
            break
    caps = np.asarray([cap for _, _, cap in active], dtype=np.int64)
    quotas = np.minimum(quotas, caps)
    order = list(np.argsort(-(raw - np.floor(raw))))
    j = 0
    while int(quotas.sum()) < k_total:
        candidates = [int(i) for i in order if int(quotas[int(i)]) < int(caps[int(i)])]
        if not candidates:
            break
        quotas[candidates[j % len(candidates)]] += 1
        j += 1
    return {cls: int(q) for (cls, _, _), q in zip(active, quotas)}


@torch.no_grad()
def anchor_difficulty_scores(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    pos_weight: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(signals), batch_size):
        x = torch.from_numpy(np.asarray(signals[start:start + batch_size], dtype=np.float32)).to(device)
        y = torch.from_numpy(np.asarray(labels[start:start + batch_size], dtype=np.float32)).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        if mode == "hard_bce":
            score = masked_bce_per_sample(logits.float(), y, pos_weight)
        elif mode == "uncertainty":
            prob = torch.sigmoid(logits.float())
            score = (1.0 - (prob - 0.5).abs() * 2.0).mean(dim=1)
        else:
            raise ValueError(f"unknown anchor difficulty mode={mode!r}")
        scores.append(score.detach().float().cpu().numpy())
    if not scores:
        return np.empty(0, dtype=np.float32)
    out = np.concatenate(scores).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def anchor_pos_boundary_scores(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(signals), batch_size):
        x = torch.from_numpy(np.asarray(signals[start:start + batch_size], dtype=np.float32)).to(device)
        y = torch.from_numpy(np.asarray(labels[start:start + batch_size], dtype=np.float32)).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        prob = torch.sigmoid(logits.float())
        boundary = 1.0 - (prob - 0.5).abs() * 2.0
        clean_positive_correct = (y > 0.5) & (prob >= 0.5)
        score = torch.where(clean_positive_correct, boundary, torch.zeros_like(boundary))
        pieces.append(score.detach().float().cpu().numpy())
    if not pieces:
        return np.empty((0, labels.shape[1]), dtype=np.float32)
    out = np.concatenate(pieces, axis=0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_sample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    replace_when_needed: bool,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if k <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(indices), dtype=np.float64)
    probs = weights / weights.sum()
    replace = bool(replace_when_needed and k > len(indices))
    return rng.choice(indices, size=int(k), replace=replace, p=probs).astype(np.int64)


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
        scores = anchor_difficulty_scores(
            model,
            anchor_pool["signals"],
            labels,
            pos_weight,
            max(1, int(args.eval_batch_size)),
            device,
            difficulty_mode,
        )
        centered = scores - float(np.min(scores)) if scores.size else scores
        sample_weights = centered + float(args.anchor_sample_min_weight)
        if float(args.anchor_sample_power) != 1.0:
            sample_weights = np.power(sample_weights, float(args.anchor_sample_power))

    class_boundary_scores = None
    if mode == "stratified_pos_boundary":
        difficulty_mode = "pos_boundary"
        class_boundary_scores = anchor_pos_boundary_scores(
            model,
            anchor_pool["signals"],
            labels,
            max(1, int(args.eval_batch_size)),
            device,
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


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    labels, scores = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        labels.append(y.numpy())
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        scores.append(probs)
    return np.concatenate(labels, axis=0), np.concatenate(scores, axis=0)


def eval_split(model: nn.Module, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device) -> dict:
    y, p = predict(model, CachedSignalDataset(signals, labels, indices), batch_size, device)
    return compute_metrics(y.astype(np.float32), p.astype(np.float32), min_pos=1)


def make_train_loader(
    ptbxl_payload: dict[str, np.ndarray],
    pn_payload: dict[str, np.ndarray],
    target_indices: np.ndarray,
    args: argparse.Namespace,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_weight: float | None = None,
) -> DataLoader:
    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    if args.source_train_limit > 0:
        source_idx = source_idx[: args.source_train_limit]
    source_ds = TaggedCachedSignalDataset(
        CachedSignalDataset(ptbxl_payload["signals"], ptbxl_payload["labels"], source_idx),
        stream_id=0,
        num_classes=len(CLASS_NAMES_SUPER5),
    )
    target_ds = TaggedCachedSignalDataset(
        CachedSignalDataset(pn_payload["signals"], pn_payload["labels"], target_indices),
        stream_id=1,
        num_classes=len(CLASS_NAMES_SUPER5),
    )
    datasets: list[Dataset] = []
    weights: list[float] = []
    effective_adv_weight = float(args.adv_weight if adv_weight is None else adv_weight)
    if args.source_weight > 0:
        datasets.append(source_ds)
        weights.extend([args.source_weight] * len(source_ds))
    if args.target_real_weight > 0:
        datasets.append(target_ds)
        weights.extend([args.target_real_weight] * len(target_ds))
    if (
        adv_signals is not None
        and adv_labels is not None
        and len(adv_signals) > 0
        and effective_adv_weight > 0
    ):
        adv_ds = TaggedMemorySignalDataset(
            adv_signals,
            adv_labels,
            stream_id=2,
            teacher_logits=adv_teacher_logits,
        )
        datasets.append(adv_ds)
        weights.extend([effective_adv_weight] * len(adv_ds))
    if not datasets:
        raise RuntimeError("no active training streams; check source/target/adv weights")
    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def masked_bce_per_sample(logits: torch.Tensor, labels_mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    mask = (labels_mask >= 0).float()
    labels_safe = torch.where(mask.bool(), labels_mask, torch.zeros_like(labels_mask))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels_safe,
        pos_weight=pos_weight,
        reduction="none",
    )
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (bce * mask).sum(dim=1) / denom


def scheduled_adv_weight(args: argparse.Namespace, epoch: int) -> float:
    target = float(args.adv_weight)
    if target <= 0:
        return 0.0
    warmup_epochs = int(args.adv_weight_warmup_epochs)
    if warmup_epochs <= 0:
        return target
    start = 0.0 if args.adv_weight_start is None else float(args.adv_weight_start)
    if warmup_epochs == 1 or epoch >= warmup_epochs:
        return target
    frac = max(0.0, float(epoch - 1) / float(max(1, warmup_epochs - 1)))
    return start + frac * (target - start)


def initial_hull_latent(
    z0: torch.Tensor,
    cand: torch.Tensor,
    *,
    weight_mode: str,
    hull_lambda: float,
    init_logit_gap: float,
) -> torch.Tensor:
    bsz, m = cand.shape[:2]
    if weight_mode == "optimized":
        logits_a = torch.full((bsz, m), -float(init_logit_gap), device=z0.device)
        logits_a[:, 0] = float(init_logit_gap)
        w = torch.softmax(logits_a, dim=-1)
    elif weight_mode == "one_hot":
        w = torch.zeros((bsz, m), device=z0.device)
        w[:, 0] = 1.0
    elif weight_mode == "uniform":
        w = torch.full((bsz, m), 1.0 / float(m), device=z0.device)
    else:
        return z0
    z_mix = (w.view(bsz, m, 1, 1) * cand).sum(dim=1)
    return (1.0 - float(hull_lambda)) * z0 + float(hull_lambda) * z_mix


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
    set_requires_grad(model, False)
    model.eval()
    victim.eval()
    adv_signals: list[np.ndarray] = []
    adv_labels: list[np.ndarray] = []
    adv_teacher_logits: list[np.ndarray] = []
    delta_norms: list[float] = []
    clean_bces: list[float] = []
    init_bces: list[float] = []
    adv_bces: list[float] = []
    loss_gains: list[float] = []
    init_loss_gains: list[float] = []
    signed_margin_drops: list[float] = []
    pos_signed_margin_drops: list[float] = []
    neg_signed_margin_drops: list[float] = []
    pos_correct = 0
    pos_hide = 0
    neg_correct = 0
    neg_add = 0
    sample_clean_correct = 0
    sample_anyflip = 0
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
                adv_logits = victim.model(ecg1000_to_ecgfounder_input(x_adv_1000))
                clean_bce = F.binary_cross_entropy_with_logits(clean_logits, y, reduction="none").mean(dim=1)
                init_bce = F.binary_cross_entropy_with_logits(init_logits, y, reduction="none").mean(dim=1)
                adv_bce = F.binary_cross_entropy_with_logits(adv_logits, y, reduction="none").mean(dim=1)
                gain = adv_bce - clean_bce
                init_gain = adv_bce - init_bce
                clean_prob = torch.sigmoid(clean_logits)
                adv_prob = torch.sigmoid(adv_logits)
                pos_mask = y > 0.5
                neg_mask = ~pos_mask
                signed = torch.where(pos_mask, torch.ones_like(y), -torch.ones_like(y))
                margin_drop = signed * (clean_logits - adv_logits)
                signed_margin_drops.extend(margin_drop.detach().flatten().cpu().tolist())
                if bool(pos_mask.any()):
                    pos_signed_margin_drops.extend(margin_drop[pos_mask].detach().cpu().tolist())
                if bool(neg_mask.any()):
                    neg_signed_margin_drops.extend(margin_drop[neg_mask].detach().cpu().tolist())
                clean_pos_ok = pos_mask & (clean_prob >= 0.5)
                clean_neg_ok = neg_mask & (clean_prob < 0.5)
                pos_flips = clean_pos_ok & (adv_prob < 0.5)
                neg_flips = clean_neg_ok & (adv_prob >= 0.5)
                sample_ok = (clean_pos_ok | clean_neg_ok).any(dim=1)
                sample_flip = (pos_flips | neg_flips).any(dim=1) & sample_ok
                clean_bces.extend(clean_bce.detach().cpu().tolist())
                init_bces.extend(init_bce.detach().cpu().tolist())
                adv_bces.extend(adv_bce.detach().cpu().tolist())
                loss_gains.extend(gain.detach().cpu().tolist())
                init_loss_gains.extend(init_gain.detach().cpu().tolist())
                pos_correct += int(clean_pos_ok.sum().item())
                pos_hide += int(pos_flips.sum().item())
                neg_correct += int(clean_neg_ok.sum().item())
                neg_add += int(neg_flips.sum().item())
                sample_clean_correct += int(sample_ok.sum().item())
                sample_anyflip += int(sample_flip.sum().item())
            x_adv_5000 = ecg1000_to_ecgfounder_input(x_adv_1000).detach().cpu().numpy().astype(np.float32)
            adv_signals.append(x_adv_5000)
            adv_labels.append(anchor_pool["labels"][batch_idx].astype(np.float32))
            adv_teacher_logits.append(clean_logits.detach().cpu().numpy().astype(np.float32))
            delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())
    finally:
        set_requires_grad(model, True)
    signals = (
        np.concatenate(adv_signals, axis=0).astype(np.float32)
        if adv_signals
        else np.empty((0, 12, TARGET_POINTS), dtype=np.float32)
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
    return {
        "signals": signals,
        "labels": labels,
        "teacher_logits": teacher_logits,
        "n_adv": int(len(signals)),
        "delta_mean": float(np.mean(delta_norms)) if delta_norms else None,
        "delta_max": float(np.max(delta_norms)) if delta_norms else None,
        "clean_bce_mean": float(np.mean(clean_bces)) if clean_bces else None,
        "init_bce_mean": float(np.mean(init_bces)) if init_bces else None,
        "adv_bce_mean": float(np.mean(adv_bces)) if adv_bces else None,
        "loss_gain_mean": float(np.mean(loss_gains)) if loss_gains else None,
        "loss_gain_p50": float(np.percentile(loss_gains, 50)) if loss_gains else None,
        "loss_gain_p90": float(np.percentile(loss_gains, 90)) if loss_gains else None,
        "init_loss_gain_mean": float(np.mean(init_loss_gains)) if init_loss_gains else None,
        "init_loss_gain_p50": float(np.percentile(init_loss_gains, 50)) if init_loss_gains else None,
        "init_loss_gain_p90": float(np.percentile(init_loss_gains, 90)) if init_loss_gains else None,
        "signed_margin_drop_mean": float(np.mean(signed_margin_drops)) if signed_margin_drops else None,
        "signed_margin_drop_p90": float(np.percentile(signed_margin_drops, 90)) if signed_margin_drops else None,
        "pos_signed_margin_drop_mean": (
            float(np.mean(pos_signed_margin_drops)) if pos_signed_margin_drops else None
        ),
        "neg_signed_margin_drop_mean": (
            float(np.mean(neg_signed_margin_drops)) if neg_signed_margin_drops else None
        ),
        "pos_hide_asr": float(pos_hide / pos_correct) if pos_correct else None,
        "neg_add_asr": float(neg_add / neg_correct) if neg_correct else None,
        "sample_anyflip_asr": float(sample_anyflip / sample_clean_correct) if sample_clean_correct else None,
        "pos_correct_count": int(pos_correct),
        "neg_correct_count": int(neg_correct),
        "sample_clean_correct_count": int(sample_clean_correct),
        **anchor_sample_stats,
    }


def main() -> None:
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
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument("--run_suffix", default="")
    ap.add_argument(
        "--run_name",
        default="",
        help="Optional short run directory name under <out_dir>/runs; useful when config tags exceed filename limits.",
    )
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    lr_tag = str(args.lr).replace(".", "p")
    sw_tag = str(args.source_weight).replace(".", "p")
    tw_tag = str(args.target_real_weight).replace(".", "p")
    method_tag = "fullft_vae" if args.enable_vae_adv_stream else "fullft"
    if args.init_head_path:
        method_tag += "_inithead"
    if args.enable_vae_adv_stream and args.vae_classes_in_scope is not None:
        method_tag += "_cls" + "-".join(str(c) for c in args.vae_classes_in_scope)
    if args.enable_vae_adv_stream and args.vae_min_class_count > 1:
        method_tag += f"_mincnt{args.vae_min_class_count}"
    if args.enable_vae_adv_stream:
        method_tag += (
            f"_aw{tag_value(args.adv_weight)}"
            f"_ka{args.k_anchor}"
            f"_M{args.hull_m}"
            f"_lam{tag_value(args.hull_lambda)}"
            f"_hs{args.hull_steps}"
            f"_hlr{tag_value(args.hull_lr)}"
        )
        if args.hull_weight_mode != "optimized":
            method_tag += f"_{args.hull_weight_mode}"
        if args.hull_attack_pos_weight_source != "none":
            method_tag += (
                f"_apw{args.hull_attack_pos_weight_source}"
                f"clip{tag_value(args.hull_attack_pos_weight_clip)}"
            )
        if args.hull_label_mode != "primary":
            method_tag += f"_label{args.hull_label_mode}"
        if args.hull_include_anchor:
            method_tag += "_includeanchor"
        if args.hull_partner_pool != "target":
            method_tag += f"_partners{args.hull_partner_pool}"
        if args.source_partner_limit_per_class > 0:
            method_tag += f"_splim{args.source_partner_limit_per_class}"
        if args.anchor_sample_mode != "stratified":
            method_tag += f"_as{args.anchor_sample_mode}"
        if args.anchor_sample_power != 1.0:
            method_tag += f"_aspow{tag_value(args.anchor_sample_power)}"
        if args.anchor_class_sample_weights:
            compact = args.anchor_class_sample_weights.replace("=", "").replace(",", "-").replace(";", "-")
            method_tag += f"_acw{compact}"
        if args.anchor_class_max_repeat > 0:
            method_tag += f"_acmaxrep{args.anchor_class_max_repeat}"
        if args.adv_weight_warmup_epochs > 0:
            method_tag += f"_awarm{args.adv_weight_warmup_epochs}"
        if args.adv_bce_loss_weight != 1.0:
            method_tag += f"_abce{tag_value(args.adv_bce_loss_weight)}"
        if args.adv_clean_logit_anchor_weight > 0:
            method_tag += f"_aclean{tag_value(args.adv_clean_logit_anchor_weight)}"
    if args.run_suffix:
        method_tag += f"_{args.run_suffix}"
    selection_tag = ""
    if args.target_val_count > 0 or args.selection_metric != "source_auprc":
        selection_tag = f"_tv{args.target_val_count}_{args.selection_metric}"
        if args.target_val_split_mode != "random":
            selection_tag += f"_{args.target_val_split_mode}"
        if args.target_val_seed is not None:
            selection_tag += f"_tvseed{args.target_val_seed}"
    if args.run_name:
        run_dir = out_dir / "runs" / args.run_name
    else:
        run_dir = out_dir / "runs" / (
            f"{args.center}_K{args.k}_fullft_ep{args.epochs}_lr{lr_tag}_"
            f"sw{sw_tag}_tw{tw_tag}_{method_tag}{selection_tag}_seed{args.seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text())
        return

    cache_dir = out_dir / "cache"
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
        victim = ECGFounderFullFTVictim(model=model, ecgtwin=ecgtwin).to(device)
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
        set_requires_grad(model, True)
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
        per_sample = masked_bce_per_sample(logits, y, pos_weight)
        stream_weights = torch.ones_like(per_sample)
        stream_weights = torch.where(
            stream == 0,
            torch.full_like(stream_weights, float(args.source_bce_loss_weight)),
            stream_weights,
        )
        stream_weights = torch.where(
            stream == 1,
            torch.full_like(stream_weights, float(args.target_real_bce_loss_weight)),
            stream_weights,
        )
        stream_weights = torch.where(
            stream == 2,
            torch.full_like(stream_weights, float(args.adv_bce_loss_weight)),
            stream_weights,
        )
        return (per_sample * stream_weights).sum() / stream_weights.sum().clamp_min(1e-6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                logits = model(x)
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
        val_metrics = eval_split(model, ptbxl["signals"], ptbxl["labels"], val_idx, args.eval_batch_size, device)
        target_val_metrics = (
            eval_split(model, pn["signals"], pn["labels"], target_val_idx, args.eval_batch_size, device)
            if len(target_val_idx) > 0
            else None
        )
        target_metrics = eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device)
        drop_metrics = eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device)
        if args.selection_metric == "source_auprc":
            selection_score = float(val_metrics["macro_auprc"])
        elif args.selection_metric == "target_val_auprc":
            assert target_val_metrics is not None
            selection_score = float(target_val_metrics["macro_auprc"])
        else:
            assert target_val_metrics is not None
            w = float(args.target_val_score_weight)
            selection_score = (
                (1.0 - w) * float(val_metrics["macro_auprc"])
                + w * float(target_val_metrics["macro_auprc"])
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
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    result = {
        "method": (
            "ECGFounder official-style full fine-tuning + VAE-only real-anchor latent-hull online AT"
            if args.enable_vae_adv_stream
            else "ECGFounder official-style full fine-tuning, no VAE"
        ),
        "vae_stream_enabled": bool(args.enable_vae_adv_stream),
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
        "ptbxl_fold10": eval_split(model, ptbxl["signals"], ptbxl["labels"], test_idx, args.eval_batch_size, device),
        "target_val": (
            eval_split(model, pn["signals"], pn["labels"], target_val_idx, args.eval_batch_size, device)
            if len(target_val_idx) > 0
            else None
        ),
        "target_excluding_ref": eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device),
        "target_drop_all_zero_excluding_ref": eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device),
        "n_target_eval": int(len(eval_idx)),
        "n_target_drop_all_zero_eval": int(len(drop_eval_idx)),
        "config": vars(args),
        "init_head": init_head_info,
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
