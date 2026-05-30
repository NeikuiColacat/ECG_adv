"""
AugMixECGDataset: wrap a base PTBXL-style Dataset to emit (clean, aug1, aug2, label)
triples for AugMix JSD training.

Contract for base_dataset.__getitem__(idx):
    returns (signal: torch.Tensor[12, L], label: Any)
"""
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset

from methods.augmix.augmix import augmix, DEFAULT_OPS


class AugMixECGDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        ops=None,
    ):
        self.base = base_dataset
        self.aug_cfg = dict(
            severity=severity,
            width=width,
            depth=depth,
            alpha=alpha,
            ops=ops if ops is not None else DEFAULT_OPS,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        signal, label = self.base[idx]
        if not isinstance(signal, torch.Tensor):
            signal = torch.as_tensor(signal, dtype=torch.float32)
        aug1 = augmix(signal, **self.aug_cfg)
        aug2 = augmix(signal, **self.aug_cfg)
        return signal, aug1, aug2, label
