"""Small online adversarial-training buffer helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def build_roundtrip_anchor_dataset(
    train_ds: Any,
    ecgtwin: Any,
    n_samples: int,
    device: str,
    crop_len: int,
    seed: int = 0,
    cache_path: str | None = None,
) -> TensorDataset | None:
    """Decode a small PTB-XL anchor subset through ECGTwin VAE for source anchoring."""

    if n_samples <= 0:
        return None
    if cache_path and os.path.exists(cache_path):
        print(f"[roundtrip_anchor] cache hit: {cache_path}")
        data = np.load(cache_path)
        label_key = "labels" if "labels" in data.files else "labels_6"
        return TensorDataset(
            torch.from_numpy(data["signals_ct_250"]).float(),
            torch.from_numpy(data[label_key]).float(),
        )

    from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES

    n = min(n_samples, len(train_ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=n, replace=False)

    dev = torch.device(device)
    roundtrip_signals: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    started = time.time()
    print(f"[roundtrip_anchor] building {n}-sample VAE-roundtrip anchor ...")
    for start in range(0, n, 32):
        chunk_idx = indices[start : start + 32]
        signals_tc = np.stack([train_ds.signals[i] for i in chunk_idx], axis=0).astype(np.float32)
        labels = np.stack([train_ds.labels[i] for i in chunk_idx], axis=0).astype(np.float32)
        batch = int(signals_tc.shape[0])

        signals_ct = np.transpose(signals_tc, (0, 2, 1))[:, ECGTWIN_TO_PTBXL_INDICES, :]
        signals_ct_t = torch.from_numpy(signals_ct).float().to(dev)
        signals_ct_t = F.interpolate(signals_ct_t, size=1024, mode="linear", align_corners=True)
        signals_lc_t = signals_ct_t.transpose(1, 2).contiguous()
        with torch.no_grad():
            latent = ecgtwin.encode_ecg(signals_lc_t)
            decoded_lc = ecgtwin.decode_latent(latent)
        decoded_ct = decoded_lc.transpose(1, 2)
        decoded_ct = decoded_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
        decoded_ct = torch.clamp(decoded_ct, min=-3.0, max=3.0)
        decoded_ct = F.interpolate(decoded_ct, size=1000, mode="linear", align_corners=True)

        flat = decoded_ct.reshape(batch, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        decoded_ct = (decoded_ct - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        crop_start = (1000 - crop_len) // 2
        roundtrip_signals.append(decoded_ct[..., crop_start : crop_start + crop_len].cpu())
        labels_list.append(torch.from_numpy(labels))

    signals_tensor = torch.cat(roundtrip_signals, dim=0).float()
    labels_tensor = torch.cat(labels_list, dim=0).float()
    print(
        f"[roundtrip_anchor] done in {time.time() - started:.0f}s. "
        f"shape={tuple(signals_tensor.shape)}"
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            signals_ct_250=signals_tensor.numpy(),
            labels=labels_tensor.numpy(),
            labels_6=labels_tensor.numpy(),
        )
        print(f"[roundtrip_anchor] cached -> {cache_path}")
    return TensorDataset(signals_tensor, labels_tensor)


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
    "build_roundtrip_anchor_dataset",
    "center_crop_ct",
    "train_one_epoch_masked_bce",
]
