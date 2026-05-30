"""
AugMix meta-algorithm for ECG signals.

Source: model/augmix/augment_and_mix.py:augment_and_mix() (Google Research, 2019).
Adapted for torch tensor (12, L) float32 instead of numpy (H, W, C) image/PIL.

Changes vs. upstream:
  1. Input type: torch.Tensor[12, L] instead of np.ndarray image.
  2. Removed normalize() call (ECG preprocessing handled in dataset).
  3. Removed PIL Image conversion in apply_op.
  4. Op pool comes from severity.build_op() instead of a hardcoded list of PIL ops.

Dirichlet/Beta/width/depth logic is preserved verbatim.
"""
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from methods.augmix.severity import AVAILABLE_OPS, build_op


DEFAULT_OPS = [
    "powerline_noise",
    "emg_noise",
    "baseline_shift",
    "baseline_wander",
    "random_leads_masking",
]


def _apply_op(signal: Tensor, op_name: str, severity: int) -> Tensor:
    op = build_op(op_name, severity)
    return op(signal)


def augmix(
    signal: Tensor,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    ops: Optional[List[str]] = None,
) -> Tensor:
    """Perform AugMix and return the mixed ECG signal.

    Mirrors model/augmix/augment_and_mix.py:augment_and_mix() lines 40-69.

    Args:
        signal: Input ECG, torch.Tensor shape (12, L), float32.
        severity: Severity of underlying ops, integer in [1, 10].
        width: Number of augmentation chains to mix (Dirichlet size).
        depth: Depth of each chain. -1 -> uniform random in {1, 2, 3}.
        alpha: Dirichlet / Beta parameter.
        ops: List of op names from severity.AVAILABLE_OPS. Defaults to all 5.

    Returns:
        Mixed tensor, same shape as input.
    """
    if ops is None:
        ops = DEFAULT_OPS
    for op_name in ops:
        if op_name not in AVAILABLE_OPS:
            raise ValueError(f"unknown op: {op_name}")

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = float(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(signal)
    for i in range(width):
        signal_aug = signal.clone()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op_name = np.random.choice(ops)
            signal_aug = _apply_op(signal_aug, op_name, severity)
        mix = mix + float(ws[i]) * signal_aug

    mixed = (1.0 - m) * signal + m * mix
    return mixed.float()
