"""Small online adversarial-training buffer helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QualityAwareBuffer:
    """FIFO buffer with quality-score eviction for adversarial ECG samples."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.ecg_list: list[torch.Tensor] = []
        self.label_list: list[torch.Tensor] = []
        self.score_list: list[float] = []

    def __len__(self) -> int:
        return len(self.ecg_list)

    def add_one(self, ecg_ct: torch.Tensor, label: torch.Tensor, score: float) -> None:
        self.ecg_list.append(ecg_ct.detach().cpu())
        self.label_list.append(label.detach().cpu())
        self.score_list.append(float(score))
        self._evict()

    def _evict(self) -> None:
        while len(self.ecg_list) > self.max_size:
            idx = int(np.argmin(self.score_list))
            self.ecg_list.pop(idx)
            self.label_list.pop(idx)
            self.score_list.pop(idx)

    def to_dataset(self) -> TensorDataset | None:
        if not self.ecg_list:
            return None
        return TensorDataset(
            torch.stack(self.ecg_list).float(),
            torch.stack(self.label_list).float(),
        )

    @torch.no_grad()
    def rescore(self, victim_model: nn.Module, device: str = "cuda") -> None:
        if not self.ecg_list:
            return
        victim_model.eval()
        all_ecg = torch.stack(self.ecg_list).to(device)
        new_scores: list[float] = []
        for i in range(0, all_ecg.shape[0], 128):
            logits = victim_model(all_ecg[i : i + 128])
            scores = (1.0 / (1.0 + logits.abs().mean(dim=1))).cpu().tolist()
            new_scores.extend(float(x) for x in scores)
        self.score_list = new_scores

    def get_sampling_weights(self) -> list[float]:
        return [max(score, 0.05) for score in self.score_list]


def center_crop_ct(signal_ct: np.ndarray, length: int) -> np.ndarray:
    """Center-crop or zero-pad a channel-time ECG array to ``length``."""

    current = int(signal_ct.shape[-1])
    if current == length:
        return signal_ct
    if current < length:
        pad = length - current
        left = pad // 2
        right = pad - left
        return np.pad(signal_ct, ((0, 0), (left, right)))
    start = (current - length) // 2
    return signal_ct[:, start : start + length]


def train_one_epoch_masked_bce(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
    ewa_params: list[torch.Tensor] | None,
    anchor_lambda: float,
    ewa_decay: float,
) -> float:
    """Train one epoch with BCE entries masked where labels use the -1 sentinel."""

    model.train()
    losses: list[float] = []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        mask = (labels >= 0).float()
        labels_clamp = labels.clamp(min=0.0)
        per_elem = criterion(logits, labels_clamp)
        bce = (per_elem * mask).sum() / mask.sum().clamp(min=1.0)
        if ewa_params is not None and anchor_lambda > 0:
            anchor = sum(
                (param - anchor_param.detach()).pow(2).sum()
                for param, anchor_param in zip(model.parameters(), ewa_params)
            )
            loss = bce + anchor_lambda * anchor
        else:
            loss = bce
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ewa_params is not None and 0 < ewa_decay < 1.0:
            with torch.no_grad():
                for param, anchor_param in zip(model.parameters(), ewa_params):
                    anchor_param.mul_(ewa_decay).add_(param.data, alpha=1 - ewa_decay)
        losses.append(float(bce.item()))
    return float(np.mean(losses)) if losses else float("nan")


__all__ = [
    "QualityAwareBuffer",
    "center_crop_ct",
    "train_one_epoch_masked_bce",
]
