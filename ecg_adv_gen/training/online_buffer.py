"""Small online adversarial-training buffer helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import masked_bce_per_sample


class QualityAwareBuffer:
    """FIFO buffer with quality-score eviction for adversarial ECG samples."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.ecg_list: list[torch.Tensor] = []
        self.label_list: list[torch.Tensor] = []
        self.score_list: list[float] = []
        self.sample_weight_list: list[float] = []

    def __len__(self) -> int:
        return len(self.ecg_list)

    def add_one(
        self,
        ecg_ct: torch.Tensor,
        label: torch.Tensor,
        score: float,
        sample_weight: float = 1.0,
    ) -> None:
        self.ecg_list.append(ecg_ct.detach().cpu())
        self.label_list.append(label.detach().cpu())
        self.score_list.append(float(score))
        self.sample_weight_list.append(float(sample_weight))
        self._evict()

    def _evict(self) -> None:
        while len(self.ecg_list) > self.max_size:
            idx = int(np.argmin(self.score_list))
            self.ecg_list.pop(idx)
            self.label_list.pop(idx)
            self.score_list.pop(idx)
            self.sample_weight_list.pop(idx)

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
        if len(self.sample_weight_list) != len(self.score_list):
            self.sample_weight_list = [1.0] * len(self.score_list)
        return [
            max(score, 0.05) * max(sample_weight, 0.0)
            for score, sample_weight in zip(self.score_list, self.sample_weight_list)
        ]


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


def train_one_epoch_grouped_target_bce(
    model: nn.Module,
    target_clean_loader: DataLoader,
    target_adv_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    *,
    target_adv_fraction: float,
    source_loader: DataLoader | None,
    source_coefficient: float,
    target_coefficient: float,
    grad_clip: float,
    ewa_params: list[torch.Tensor] | None,
    anchor_lambda: float,
    ewa_decay: float,
) -> dict[str, Any]:
    """Train matched groups in one step per clean-target batch."""
    target_adv_fraction = float(target_adv_fraction)
    source_coefficient = float(source_coefficient)
    target_coefficient = float(target_coefficient)
    if not 0.0 <= target_adv_fraction <= 1.0:
        raise ValueError("target_adv_fraction must be in [0, 1]")
    if source_coefficient < 0.0 or target_coefficient <= 0.0:
        raise ValueError("group coefficients require source >= 0 and target > 0")
    adv_active = target_adv_fraction > 0.0
    if len(target_clean_loader) == 0:
        raise ValueError("target clean loader must be nonempty")
    if adv_active and len(target_adv_loader) == 0:
        raise ValueError("active target adversarial loader must be nonempty")
    source_active = source_loader is not None and source_coefficient > 0.0
    if source_active and len(source_loader) == 0:
        raise ValueError("active source loader must be nonempty")

    def next_cycled(loader: DataLoader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    model.train()
    adv_iter = iter(target_adv_loader) if adv_active else None
    source_iter = iter(source_loader) if source_active else None
    step_losses: list[float] = []
    target_step_losses: list[float] = []
    clean_step_losses: list[float] = []
    adv_step_losses: list[float] = []
    source_step_losses: list[float] = []
    clean_losses: list[torch.Tensor] = []
    adv_losses: list[torch.Tensor] = []
    source_losses: list[torch.Tensor] = []
    coefficient_sum = target_coefficient + (source_coefficient if source_active else 0.0)
    source_nominal = source_coefficient / coefficient_sum if source_active else 0.0
    target_nominal = target_coefficient / coefficient_sum
    clean_nominal = target_nominal * (1.0 - target_adv_fraction)
    adv_nominal = target_nominal * target_adv_fraction

    for clean_signals, clean_labels in target_clean_loader:
        clean_signals = clean_signals.to(device, non_blocking=True)
        clean_labels = clean_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        clean_per_sample = masked_bce_per_sample(
            model(clean_signals), clean_labels, getattr(criterion, "pos_weight", None)
        )
        clean_mean = clean_per_sample.mean()
        target_loss = (1.0 - target_adv_fraction) * clean_mean
        adv_per_sample: torch.Tensor | None = None
        if adv_active:
            assert adv_iter is not None
            (adv_signals, adv_labels), adv_iter = next_cycled(
                target_adv_loader, adv_iter
            )
            adv_signals = adv_signals.to(device, non_blocking=True)
            adv_labels = adv_labels.to(device, non_blocking=True)
            adv_per_sample = masked_bce_per_sample(
                model(adv_signals), adv_labels, getattr(criterion, "pos_weight", None)
            )
            adv_mean = adv_per_sample.mean()
            target_loss = target_loss + target_adv_fraction * adv_mean
        primary_bce = target_coefficient * target_loss
        if source_active:
            (source_signals, source_labels), source_iter = next_cycled(
                source_loader, source_iter
            )
            source_signals = source_signals.to(device, non_blocking=True)
            source_labels = source_labels.to(device, non_blocking=True)
            source_per_sample = masked_bce_per_sample(
                model(source_signals), source_labels, getattr(criterion, "pos_weight", None)
            )
            source_batch_mean = source_per_sample.mean()
            primary_bce = primary_bce + source_coefficient * source_batch_mean
            source_losses.append(source_per_sample.detach().cpu())
            source_step_losses.append(float(source_batch_mean.detach().item()))
        primary_bce = primary_bce / coefficient_sum

        if ewa_params is not None and anchor_lambda > 0:
            anchor = sum(
                (param - anchor_param.detach()).pow(2).sum()
                for param, anchor_param in zip(model.parameters(), ewa_params)
            )
            loss = primary_bce + float(anchor_lambda) * anchor
        else:
            loss = primary_bce
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ewa_params is not None and 0 < ewa_decay < 1.0:
            with torch.no_grad():
                for param, anchor_param in zip(model.parameters(), ewa_params):
                    anchor_param.mul_(ewa_decay).add_(param.data, alpha=1 - ewa_decay)

        step_losses.append(float(primary_bce.item()))
        target_step_losses.append(float(target_loss.detach().item()))
        clean_step_losses.append(float(clean_mean.detach().item()))
        clean_losses.append(clean_per_sample.detach().cpu())
        if adv_per_sample is not None:
            adv_step_losses.append(float(adv_mean.detach().item()))
            adv_losses.append(adv_per_sample.detach().cpu())

    clean_all = torch.cat(clean_losses)
    adv_all = torch.cat(adv_losses) if adv_losses else None
    source_all = torch.cat(source_losses) if source_losses else None
    clean_step_mean = float(np.mean(clean_step_losses))
    adv_step_mean = float(np.mean(adv_step_losses)) if adv_step_losses else None
    source_step_mean = float(np.mean(source_step_losses)) if source_step_losses else None
    clean_sample_mean = float(clean_all.mean().item())
    adv_sample_mean = float(adv_all.mean().item()) if adv_all is not None else None
    source_sample_mean = float(source_all.mean().item()) if source_all is not None else None
    source_weighted = (
        source_nominal * source_step_mean if source_step_mean is not None else 0.0
    )
    clean_weighted = clean_nominal * clean_step_mean
    adv_weighted = adv_nominal * adv_step_mean if adv_step_mean is not None else 0.0
    objective = float(np.mean(step_losses))
    target_objective = float(np.mean(target_step_losses))

    def realized(value: float) -> float:
        return value / objective if objective else 0.0

    target_stats: dict[str, Any] = {
        "target_clean_count": int(clean_all.numel()),
        "target_adv_count": int(adv_all.numel()) if adv_all is not None else 0,
        "target_clean_step_count": len(clean_step_losses),
        "target_adv_step_count": len(adv_step_losses),
        "target_clean_loss_mean": clean_step_mean,
        "target_adv_loss_mean": adv_step_mean,
        "target_clean_loss_sample_mean": clean_sample_mean,
        "target_adv_loss_sample_mean": adv_sample_mean,
        "target_clean_weighted_loss": clean_weighted,
        "target_adv_weighted_loss": adv_weighted,
        "target_clean_nominal_fraction": 1.0 - target_adv_fraction,
        "target_adv_nominal_fraction": target_adv_fraction,
        "target_clean_nominal_contribution_fraction": clean_nominal,
        "target_adv_nominal_contribution_fraction": adv_nominal,
        "target_clean_realized_contribution_fraction": realized(clean_weighted),
        "target_adv_realized_contribution_fraction": realized(adv_weighted),
        "target_clean_contribution_fraction": realized(clean_weighted),
        "target_adv_contribution_fraction": realized(adv_weighted),
    }
    return {
        "loss": objective,
        "objective_loss": objective,
        "target_objective_loss": target_objective,
        "n_batches": len(step_losses),
        "source_count": int(source_all.numel()) if source_all is not None else 0,
        "source_step_count": len(source_step_losses),
        "source_loss_mean": source_step_mean,
        "source_loss_sample_mean": source_sample_mean,
        "source_weighted_loss": source_weighted,
        "source_nominal_contribution_fraction": source_nominal,
        "source_realized_contribution_fraction": realized(source_weighted),
        "target_nominal_contribution_fraction": target_nominal,
        "source_coefficient": source_coefficient,
        "target_coefficient": target_coefficient,
        **target_stats,
    }


__all__ = [
    "QualityAwareBuffer",
    "center_crop_ct",
    "train_one_epoch_grouped_target_bce",
    "train_one_epoch_masked_bce",
]
