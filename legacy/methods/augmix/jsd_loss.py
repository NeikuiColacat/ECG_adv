"""
Multi-label JSD (Jensen-Shannon Divergence) consistency loss for AugMix.

Formula (AugMix paper eq.2, adapted for multi-label Bernoulli):
    p_c, p_1, p_2 = sigmoid(logits_*)               # (B, C)
    M = (p_c + p_1 + p_2) / 3
    JSD = (KL(p_c || M) + KL(p_1 || M) + KL(p_2 || M)) / 3

For multi-label, each of the C classes is an independent Bernoulli,
so KL is computed per-class and summed before averaging over batch.
"""
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def _bernoulli_kl(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-class Bernoulli KL(p || q), summed over class dim.

    p, q: (B, C) probabilities in (0, 1).
    Returns: (B,) KL per sample.

    Note: eps=1e-6 (not 1e-8) because float32 precision is ~1e-7;
    smaller eps can round to 0 or 1 and cause log(0) -> -inf -> NaN.
    """
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    # Use log1p for the (1-p) side to improve stability near p -> 1.
    kl = p * (p.log() - q.log()) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q))
    return kl.sum(dim=-1)


def jsd_multilabel(
    logits_clean: Tensor,
    logits_aug1: Tensor,
    logits_aug2: Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Multi-label 3-way JSD consistency loss.

    Args:
        logits_clean, logits_aug1, logits_aug2: (B, C) logits.
        reduction: "mean" | "sum" | "none".

    Returns:
        Loss. Scalar if reduction in {"mean","sum"}, else (B,).
    """
    p_c = torch.sigmoid(logits_clean)
    p_1 = torch.sigmoid(logits_aug1)
    p_2 = torch.sigmoid(logits_aug2)
    m = (p_c + p_1 + p_2) / 3.0

    jsd = (_bernoulli_kl(p_c, m) + _bernoulli_kl(p_1, m) + _bernoulli_kl(p_2, m)) / 3.0

    if reduction == "mean":
        return jsd.mean()
    if reduction == "sum":
        return jsd.sum()
    return jsd
