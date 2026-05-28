"""Torch ECGFounder preprocessing helpers.

Keep these helpers out of ``ecg_adv_gen.models.__init__`` so lightweight config
and path-contract imports do not load torch unless a runner explicitly needs
model execution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def global_zscore_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply per-sample global z-score normalization over all non-batch values."""
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)


def ecg1000_to_ecgfounder_input(
    ecg_ct_1000: torch.Tensor,
    *,
    target_points: int = 5000,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert channel-time 100Hz ECG tensors to ECGFounder 500Hz normalized input."""
    x = F.interpolate(
        ecg_ct_1000,
        size=int(target_points),
        mode="linear",
        align_corners=True,
    )
    return global_zscore_torch(x, eps=eps)
