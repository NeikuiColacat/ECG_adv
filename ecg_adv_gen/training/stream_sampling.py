"""Weighted multi-stream DataLoader helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, WeightedRandomSampler


@dataclass(frozen=True)
class WeightedFeatureStream:
    """One feature/label stream and its sampling policy."""

    features: np.ndarray
    labels: np.ndarray
    stream_id: int
    base_weight: float
    class_weights: dict[str, float] | None = None


def multilabel_stream_sample_weights(
    labels: np.ndarray,
    *,
    base_weight: float,
    class_weights: dict[str, float] | None,
    class_to_idx: dict[str, int],
    num_classes: int | None = None,
) -> list[float]:
    """Per-row sampling weights for multi-label rows.

    If class weights are provided, a positive multi-label row receives the
    maximum configured positive-class multiplier. All-zero rows use
    ``base_weight`` unchanged, matching the legacy ECGFounder VAE-LHAT loader.
    """
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError(f"expected 2D labels, got shape {labels.shape}")
    base_weight = float(base_weight)
    if not class_weights:
        return [base_weight] * len(labels)
    n_classes = int(num_classes if num_classes is not None else labels.shape[1])
    if n_classes != labels.shape[1]:
        raise ValueError(f"num_classes={n_classes} does not match labels columns={labels.shape[1]}")
    weights_vec = np.ones((n_classes,), dtype=np.float32)
    for cls, value in class_weights.items():
        if cls not in class_to_idx:
            raise ValueError(f"unknown class weight key {cls!r}")
        idx = int(class_to_idx[cls])
        if idx < 0 or idx >= n_classes:
            raise ValueError(f"class index for {cls!r} out of range: {idx}")
        weights_vec[idx] = float(value)

    out: list[float] = []
    for row in labels > 0.5:
        if row.any():
            out.append(base_weight * float(np.max(weights_vec[row])))
        else:
            out.append(base_weight)
    return out


def build_weighted_feature_stream_loader(
    streams: list[WeightedFeatureStream],
    *,
    batch_size: int,
    class_to_idx: dict[str, int],
    num_classes: int,
    drop_last: bool = False,
) -> DataLoader:
    """Build a weighted sampler over source/target/adversarial feature streams."""
    datasets = []
    weights: list[float] = []
    for stream in streams:
        if float(stream.base_weight) <= 0.0:
            continue
        features = np.asarray(stream.features, dtype=np.float32)
        labels = np.asarray(stream.labels, dtype=np.float32)
        if len(features) == 0:
            continue
        if len(features) != len(labels):
            raise ValueError(
                f"stream {stream.stream_id} feature/label length mismatch: {len(features)} vs {len(labels)}"
            )
        stream_ids = torch.full((len(features),), int(stream.stream_id), dtype=torch.long)
        ds = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(labels).float(), stream_ids)
        datasets.append(ds)
        weights.extend(
            multilabel_stream_sample_weights(
                labels,
                base_weight=float(stream.base_weight),
                class_weights=stream.class_weights or {},
                class_to_idx=class_to_idx,
                num_classes=num_classes,
            )
        )
    if not datasets:
        raise ValueError("at least one non-empty stream with positive base_weight is required")
    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(combined, batch_size=int(batch_size), sampler=sampler, drop_last=bool(drop_last))


__all__ = [
    "WeightedFeatureStream",
    "build_weighted_feature_stream_loader",
    "multilabel_stream_sample_weights",
]
