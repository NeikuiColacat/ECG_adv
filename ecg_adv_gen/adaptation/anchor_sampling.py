"""Torch-based anchor sampling helpers for online adaptation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ecg_adv_gen.labels import CLASS_NAMES_SUPER5
from ecg_adv_gen.training.losses import masked_bce_per_sample


SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}


def parse_anchor_class_weight_string(
    raw: str | None,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES_SUPER5,
) -> dict[str, float]:
    """Parse ``CLASS=weight`` anchor sampling weights.

    The parser preserves the ECGFounder full-FT legacy CLI behavior: comma and
    semicolon separators are accepted, unknown Super5 classes raise, and values
    are not clipped.
    """
    out: dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    valid = set(class_names)
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"class weight entry must be NAME=VALUE, got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in valid:
            raise ValueError(f"unknown class in --anchor_class_sample_weights: {name!r}")
        out[name] = float(value)
    return out


def weighted_class_quotas_with_caps(
    classes: list[str],
    k_total: int,
    class_weights: dict[str, float],
    labels: np.ndarray,
    *,
    max_repeat_per_class: int = 0,
    class_to_idx: dict[str, int] | None = None,
) -> dict[str, int]:
    """Allocate per-class anchor counts with optional repeat caps."""
    class_to_idx = SUPER5_TO_IDX if class_to_idx is None else class_to_idx
    labels = np.asarray(labels, dtype=np.float32)
    active = []
    for cls in classes:
        cls_i = class_to_idx[cls]
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
    if int(k_total) >= len(active):
        quotas = np.maximum(quotas, 1)
    while int(quotas.sum()) > int(k_total):
        i = int(np.argmax(quotas - raw))
        if quotas[i] > 0:
            quotas[i] -= 1
        else:
            break
    caps = np.asarray([cap for _, _, cap in active], dtype=np.int64)
    quotas = np.minimum(quotas, caps)
    order = list(np.argsort(-(raw - np.floor(raw))))
    j = 0
    while int(quotas.sum()) < int(k_total):
        candidates = [int(i) for i in order if int(quotas[int(i)]) < int(caps[int(i)])]
        if not candidates:
            break
        quotas[candidates[j % len(candidates)]] += 1
        j += 1
    return {cls: int(q) for (cls, _, _), q in zip(active, quotas)}


def weighted_sample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    replace_when_needed: bool,
) -> np.ndarray:
    """Sample indices using non-negative normalized weights."""
    indices = np.asarray(indices, dtype=np.int64)
    if int(k) <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(indices), dtype=np.float64)
    probs = weights / weights.sum()
    replace = bool(replace_when_needed and int(k) > len(indices))
    return rng.choice(indices, size=int(k), replace=replace, p=probs).astype(np.int64)


@torch.no_grad()
def signal_anchor_difficulty_scores(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    pos_weight: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
    mode: str,
) -> np.ndarray:
    """Score cached ECG anchors using the current signal model."""

    signals = np.asarray(signals, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if len(signals) != len(labels):
        raise ValueError(f"anchor signal/label length mismatch: {len(signals)} vs {len(labels)}")
    if len(signals) == 0:
        return np.empty(0, dtype=np.float32)

    device = torch.device(device)
    batch_size = max(1, int(batch_size))
    scores: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(signals), batch_size):
        x = torch.from_numpy(np.asarray(signals[start:start + batch_size], dtype=np.float32)).to(device)
        y = torch.from_numpy(np.asarray(labels[start:start + batch_size], dtype=np.float32)).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        if mode == "hard_bce":
            score = masked_bce_per_sample(logits.float(), y, pos_weight.to(device))
        elif mode == "uncertainty":
            prob = torch.sigmoid(logits.float())
            score = (1.0 - (prob - 0.5).abs() * 2.0).mean(dim=1)
        else:
            raise ValueError(f"unknown anchor difficulty mode={mode!r}")
        scores.append(score.detach().float().cpu().numpy())

    out = np.concatenate(scores).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def signal_anchor_positive_boundary_scores(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Return class-wise boundary scores for correctly positive anchors."""

    signals = np.asarray(signals, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if len(signals) != len(labels):
        raise ValueError(f"anchor signal/label length mismatch: {len(signals)} vs {len(labels)}")
    if len(signals) == 0:
        n_classes = labels.shape[1] if labels.ndim == 2 else 0
        return np.empty((0, n_classes), dtype=np.float32)

    device = torch.device(device)
    batch_size = max(1, int(batch_size))
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

    out = np.concatenate(pieces, axis=0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def feature_anchor_difficulty_weights(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    power: float,
    min_weight: float,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Return normalized per-anchor weights from feature-head behavior."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if len(features) != len(labels):
        raise ValueError(f"anchor feature/label length mismatch: {len(features)} vs {len(labels)}")
    if len(features) == 0:
        return np.empty((0,), dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {features.shape}")
    if labels.ndim != 2:
        raise ValueError(f"expected 2D label matrix, got shape {labels.shape}")

    device = torch.device(device)
    logits_list: list[torch.Tensor] = []
    head.eval()
    loader = DataLoader(torch.from_numpy(features).float(), batch_size=int(batch_size), shuffle=False)
    for x in loader:
        logits_list.append(head(x.to(device)).detach().float().cpu())
    logits = torch.cat(logits_list, dim=0)
    y = torch.from_numpy(labels)
    if mode == "hard_bce":
        score = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(dim=1).numpy()
    elif mode == "uncertainty":
        prob = torch.sigmoid(logits)
        score = (1.0 - torch.abs(prob - 0.5) * 2.0).mean(dim=1).numpy()
    else:
        raise ValueError(f"unsupported anchor_sample_mode={mode!r}")

    score = np.asarray(score, dtype=np.float64)
    score = np.maximum(score, 0.0)
    if float(power) != 1.0:
        score = np.power(score + 1e-12, float(power))
    score = score + max(float(min_weight), 0.0)
    if not np.all(np.isfinite(score)) or float(score.sum()) <= 0.0:
        return np.full((len(features),), 1.0 / len(features), dtype=np.float32)
    return (score / score.sum()).astype(np.float32)


def sample_hard_feature_anchors(
    *,
    head: nn.Module,
    pool_features: np.ndarray,
    pool_labels: np.ndarray,
    k_anchor: int,
    mode: str,
    power: float,
    min_weight: float,
    batch_size: int,
    device: torch.device | str,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample hard/uncertain anchors and return sampling diagnostics."""
    n_pool = len(pool_labels)
    if n_pool == 0 or int(k_anchor) <= 0:
        return np.empty((0,), dtype=np.int64), {"mean_weight": 0.0, "max_weight": 0.0}
    weights = feature_anchor_difficulty_weights(
        head,
        pool_features,
        pool_labels,
        mode=mode,
        power=power,
        min_weight=min_weight,
        batch_size=batch_size,
        device=device,
    )
    n_take = min(int(k_anchor), n_pool)
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(n_pool, size=n_take, replace=False, p=weights)
    return picks.astype(np.int64), {
        "mean_weight": float(np.mean(weights)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
        "ess": float(1.0 / np.sum(np.square(weights.astype(np.float64)))),
    }


__all__ = [
    "feature_anchor_difficulty_weights",
    "parse_anchor_class_weight_string",
    "sample_hard_feature_anchors",
    "signal_anchor_difficulty_scores",
    "signal_anchor_positive_boundary_scores",
    "weighted_class_quotas_with_caps",
    "weighted_sample_indices",
]
