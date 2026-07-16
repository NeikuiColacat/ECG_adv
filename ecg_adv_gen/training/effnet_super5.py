"""EfficientNet/Super5 training datasets and torch helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ecg_adv_gen.data import load_synthetic_npz_arrays, normalize_synthetic_signals
from ecg_adv_gen.preprocessing import crop_signal_tc


class PTBXLDatasetScheme(Dataset):
    """PTB-XL dataset emitting ``(12, crop_len)`` tensors plus labels."""

    def __init__(self, signals_1000: np.ndarray, labels: np.ndarray, crop_len: int = 250, mode: str = "train"):
        self.signals = normalize_synthetic_signals(signals_1000)
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode="random" if self.mode == "train" else "center")
        model_input = np.array(crop.T, dtype=np.float32, order="C", copy=True)
        return (
            torch.from_numpy(model_input).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


class SynthNPZDataset(Dataset):
    """Synthetic ECG dataset for Super5 augmentation artifacts."""

    def __init__(self, npz_paths: str, crop_len: int = 250, mode: str = "train"):
        loaded = load_synthetic_npz_arrays(npz_paths)
        self.signals = loaded.signals
        self.labels = loaded.labels
        self.crop_len = int(crop_len)
        self.mode = mode

    _normalize_signals = staticmethod(normalize_synthetic_signals)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode="random" if self.mode == "train" else "center")
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str | torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    labels_np = np.concatenate(all_labels)
    logits_np = np.concatenate(all_logits)
    probs = 1 / (1 + np.exp(-np.clip(logits_np, -50, 50)))
    return total_loss / max(n_batches, 1), labels_np, probs


__all__ = ["PTBXLDatasetScheme", "SynthNPZDataset", "evaluate", "init_weights"]
