"""ECGFounder feature-head inference helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ecg_adv_gen.training import CachedSignalDataset


def sigmoid_clipped(logits: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """Numerically stable sigmoid matching legacy ECGFounder head scripts."""
    arr = np.asarray(logits, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(arr, -float(clip), float(clip))))).astype(np.float32)


@torch.no_grad()
def predict_feature_head(
    head: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device | str,
    *,
    clip_logits: float = 50.0,
) -> np.ndarray:
    """Run a linear/adapter ECGFounder head over cached features."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {features.shape}")
    if len(features) == 0:
        return np.empty((0, 0), dtype=np.float32)
    device = torch.device(device)
    head.eval()
    scores: list[np.ndarray] = []
    loader = DataLoader(torch.from_numpy(features).float(), batch_size=int(batch_size), shuffle=False)
    for x in loader:
        logits = head(x.to(device)).detach().float().cpu().numpy()
        scores.append(sigmoid_clipped(logits, clip=clip_logits))
    return np.concatenate(scores, axis=0)


def evaluate_feature_head(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[..., dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device | str,
    min_pos: int = 1,
) -> dict[str, Any]:
    """Predict cached features and pass scores to the supplied metric function."""
    scores = predict_feature_head(head, features, batch_size=batch_size, device=device)
    return metric_fn(np.asarray(labels, dtype=np.float32), scores, min_pos=int(min_pos))


def evaluate_ptbxl_fold_head(
    head: nn.Module,
    payload: dict[str, np.ndarray],
    metric_fn: Callable[..., dict[str, Any]],
    *,
    fold: int,
    batch_size: int,
    device: torch.device | str,
    min_pos: int = 1,
) -> dict[str, Any]:
    """Evaluate a feature head on one PTB-XL fold from a cached feature payload."""
    folds = np.asarray(payload["folds"], dtype=np.int64)
    mask = folds == int(fold)
    return evaluate_feature_head(
        head,
        np.asarray(payload["features"])[mask].astype(np.float32),
        np.asarray(payload["labels"])[mask].astype(np.float32),
        metric_fn,
        batch_size=batch_size,
        device=device,
        min_pos=min_pos,
    )


def evaluate_pn2021_feature_head(
    head: nn.Module,
    payload: dict[str, np.ndarray],
    ref_ids: dict[str, set[str]],
    view_fn: Callable[..., dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device | str,
    report_drop_all_zero: bool = False,
) -> dict[str, Any]:
    """Evaluate a cached PN2021 feature payload through a supplied view function."""
    scores = predict_feature_head(
        head,
        np.asarray(payload["features"], dtype=np.float32),
        batch_size=batch_size,
        device=device,
    )
    return view_fn(
        np.asarray(payload["labels"], dtype=np.float32),
        scores,
        np.asarray(payload["centers"]).astype(str),
        np.asarray(payload["record_ids"]).astype(str),
        ref_ids,
        report_drop_all_zero=report_drop_all_zero,
    )


@torch.no_grad()
def predict_signal_dataset(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a full ECGFounder-style signal model over a labeled dataset."""
    device = torch.device(device)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        labels.append(y.detach().cpu().numpy())
        scores.append(torch.sigmoid(logits).detach().float().cpu().numpy())
    if not labels:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(labels, axis=0), np.concatenate(scores, axis=0)


def evaluate_signal_split(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    metric_fn: Callable[..., dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device | str,
    min_pos: int = 1,
) -> dict[str, Any]:
    """Evaluate a full signal model on an indexed split from cached arrays."""
    y_true, scores = predict_signal_dataset(
        model,
        CachedSignalDataset(signals, labels, indices),
        batch_size=batch_size,
        device=device,
    )
    return metric_fn(y_true.astype(np.float32), scores.astype(np.float32), min_pos=int(min_pos))


__all__ = [
    "evaluate_feature_head",
    "evaluate_pn2021_feature_head",
    "evaluate_signal_split",
    "evaluate_ptbxl_fold_head",
    "predict_feature_head",
    "predict_signal_dataset",
    "sigmoid_clipped",
]
