"""FP32 prediction consistency for the finite ECG/PULSE JSD ablations."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def categorical_jsd(logits: list[torch.Tensor]) -> torch.Tensor:
    """Mean over aligned positions; all three probability branches receive gradients."""
    if len(logits) != 3 or len({tuple(x.shape) for x in logits}) != 1:
        raise ValueError("JSD requires three identically shaped aligned distributions")
    if logits[0].ndim < 2 or logits[0].shape[-1] < 2:
        raise ValueError("JSD requires a categorical probability axis")
    logs = torch.stack([F.log_softmax(x.float(), dim=-1) for x in logits])
    mixture = torch.logsumexp(logs, dim=0) - math.log(3.0)
    return (logs.exp() * (logs - mixture)).sum(dim=-1).mean()


def bernoulli_jsd(logits: list[torch.Tensor]) -> torch.Tensor:
    """Each Super5 label is an independent binary distribution, not a 5-way softmax."""
    return categorical_jsd([torch.stack((torch.zeros_like(x), x), dim=-1) for x in logits])
