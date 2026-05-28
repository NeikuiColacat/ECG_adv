"""Torch helpers for latent-hull online adversarial training."""

from __future__ import annotations

import torch


def initial_hull_latent(
    z0: torch.Tensor,
    cand: torch.Tensor,
    *,
    weight_mode: str,
    hull_lambda: float,
    init_logit_gap: float,
) -> torch.Tensor:
    """Reconstruct the latent-hull optimizer's deterministic starting point.

    ``cand`` is expected to include the anchor candidate at index 0. For
    stochastic ``dirichlet`` starts, return ``z0`` as the deterministic
    diagnostic reference rather than guessing the random draw.
    """
    if cand.ndim < 3:
        raise ValueError(f"expected candidate tensor with batch and candidate axes, got {tuple(cand.shape)}")
    bsz, m = cand.shape[:2]
    if z0.shape[0] != bsz:
        raise ValueError(f"z0 batch {z0.shape[0]} does not match candidate batch {bsz}")
    if weight_mode == "optimized":
        logits_a = torch.full((bsz, m), -float(init_logit_gap), device=z0.device, dtype=z0.dtype)
        logits_a[:, 0] = float(init_logit_gap)
        w = torch.softmax(logits_a, dim=-1)
    elif weight_mode == "one_hot":
        w = torch.zeros((bsz, m), device=z0.device, dtype=z0.dtype)
        w[:, 0] = 1.0
    elif weight_mode == "uniform":
        w = torch.full((bsz, m), 1.0 / float(m), device=z0.device, dtype=z0.dtype)
    else:
        return z0
    view_shape = (bsz, m) + (1,) * (cand.ndim - 2)
    z_mix = (w.view(view_shape) * cand).sum(dim=1)
    return (1.0 - float(hull_lambda)) * z0 + float(hull_lambda) * z_mix


__all__ = ["initial_hull_latent"]
