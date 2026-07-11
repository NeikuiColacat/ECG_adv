"""Torch helpers for latent-hull online adversarial training."""

from __future__ import annotations

import torch


def identity_collapsed_weight_metrics(
    candidate_pool_indices: torch.Tensor,
    weights: torch.Tensor,
    candidate_is_anchor: torch.Tensor | None = None,
) -> dict[str, list[float] | list[int]]:
    """Aggregate positional coefficients by candidate identity before scoring use."""
    ids = torch.as_tensor(candidate_pool_indices).long().cpu()
    values = torch.as_tensor(weights).detach().float().cpu()
    if ids.ndim != 2 or values.shape != ids.shape:
        raise ValueError("candidate_pool_indices and weights must be aligned 2D tensors")
    anchor_mask = (
        torch.zeros_like(ids, dtype=torch.bool)
        if candidate_is_anchor is None
        else torch.as_tensor(candidate_is_anchor).bool().cpu()
    )
    if anchor_mask.shape != ids.shape:
        raise ValueError("candidate_is_anchor must align with candidate identities")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("identity weights must be finite and non-negative")

    unique_nonself: list[int] = []
    duplicate_fraction: list[float] = []
    top1: list[float] = []
    entropy: list[float] = []
    effective_count: list[float] = []
    anchor_count: list[int] = []
    for row_ids, row_weights, row_anchor in zip(ids, values, anchor_mask):
        collapsed: dict[int, float] = {}
        nonself: set[int] = set()
        for identity, weight, is_anchor in zip(row_ids.tolist(), row_weights.tolist(), row_anchor.tolist()):
            collapsed[int(identity)] = collapsed.get(int(identity), 0.0) + float(weight)
            if not is_anchor:
                nonself.add(int(identity))
        collapsed_weights = torch.tensor(list(collapsed.values()), dtype=torch.float64)
        total = float(collapsed_weights.sum())
        if total <= 0.0:
            raise ValueError("identity weights must have positive row sums")
        probs = collapsed_weights / total
        row_entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        unique_nonself.append(len(nonself))
        duplicate_fraction.append(1.0 - len(collapsed) / max(1, len(row_ids)))
        top1.append(float(probs.max()))
        entropy.append(row_entropy)
        effective_count.append(float(torch.exp(torch.tensor(row_entropy, dtype=torch.float64))))
        anchor_count.append(int(row_anchor.sum()))
    return {
        "candidate_unique_nonself_count": unique_nonself,
        "candidate_duplicate_fraction": duplicate_fraction,
        "identity_top1_weight": top1,
        "identity_entropy": entropy,
        "effective_candidate_count": effective_count,
        "candidate_anchor_count": anchor_count,
    }


def initial_hull_weights(
    batch_size: int,
    candidate_count: int,
    *,
    weight_mode: str,
    init_logit_gap: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return the deterministic coefficient state used at attack start."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if weight_mode == "optimized":
        logits = torch.full((batch_size, candidate_count), -float(init_logit_gap), device=device, dtype=dtype)
        logits[:, 0] = float(init_logit_gap)
        return torch.softmax(logits, dim=-1)
    if weight_mode == "one_hot":
        weights = torch.zeros((batch_size, candidate_count), device=device, dtype=dtype)
        weights[:, 0] = 1.0
        return weights
    if weight_mode == "uniform":
        return torch.full((batch_size, candidate_count), 1.0 / candidate_count, device=device, dtype=dtype)
    return None


def latent_hull_geometry(
    z0: torch.Tensor,
    cand: torch.Tensor,
    weights: torch.Tensor,
    *,
    hull_lambda: float,
    epsilon: float | None,
    candidate_is_anchor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply one hull coefficient state and report its projection geometry."""
    batch_size, candidate_count = cand.shape[:2]
    if weights.shape != (batch_size, candidate_count):
        raise ValueError(f"weights must have shape {(batch_size, candidate_count)}, got {tuple(weights.shape)}")
    anchor_mask = (
        torch.zeros_like(weights, dtype=torch.bool)
        if candidate_is_anchor is None
        else candidate_is_anchor
    )
    if anchor_mask.shape != weights.shape:
        raise ValueError("candidate_is_anchor must match weights")
    z_mix = (weights.view((batch_size, candidate_count) + (1,) * (cand.ndim - 2)) * cand).sum(1)
    latent_pre = (1.0 - float(hull_lambda)) * z0 + float(hull_lambda) * z_mix
    delta_pre = latent_pre - z0
    norm_pre = delta_pre.flatten(1).norm(dim=1)
    scale = (
        torch.clamp(float(epsilon) / norm_pre.clamp_min(1e-12), max=1.0)
        if epsilon is not None and float(epsilon) > 0
        else torch.ones_like(norm_pre)
    )
    scale = torch.where(norm_pre > 0, scale, torch.ones_like(scale))
    delta_post = delta_pre * scale.view((batch_size,) + (1,) * (z0.ndim - 1))
    anchor_weight = (weights * anchor_mask.to(weights.dtype)).sum(dim=1)
    effective_lambda = float(hull_lambda) * scale
    geometry = {
        "latent_pre": latent_pre,
        "latent": z0 + delta_post,
        "delta_pre": delta_pre,
        "delta_post": delta_post,
        "pre_projection_norm": norm_pre,
        "post_projection_norm": delta_post.flatten(1).norm(dim=1),
        "projection_scale": scale,
        "effective_lambda": effective_lambda,
        "candidate_anchor_weight": anchor_weight,
        "effective_original_share": 1.0 - effective_lambda * (1.0 - anchor_weight),
    }
    for field in (
        "pre_projection_norm", "post_projection_norm", "projection_scale",
        "effective_lambda", "candidate_anchor_weight", "effective_original_share",
    ):
        bad = (~torch.isfinite(geometry[field])).nonzero(as_tuple=False)
        if bad.numel():
            raise RuntimeError(
                f"latent-hull geometry diagnostic field {field!r} is non-finite "
                f"at sample {int(bad[0, 0])}"
            )
    return geometry


def initial_hull_latent(
    z0: torch.Tensor,
    cand: torch.Tensor,
    *,
    weight_mode: str,
    hull_lambda: float,
    init_logit_gap: float,
) -> torch.Tensor:
    """Reconstruct the latent-hull optimizer's deterministic starting point.

    Legacy optimized/one-hot modes emphasize coefficient position 0; they do
    not infer anchor identity. For stochastic ``dirichlet`` starts, return
    ``z0`` rather than guessing the random draw.
    """
    if cand.ndim < 3:
        raise ValueError(f"expected candidate tensor with batch and candidate axes, got {tuple(cand.shape)}")
    bsz, m = cand.shape[:2]
    if z0.shape[0] != bsz:
        raise ValueError(f"z0 batch {z0.shape[0]} does not match candidate batch {bsz}")
    w = initial_hull_weights(
        bsz,
        m,
        weight_mode=weight_mode,
        init_logit_gap=init_logit_gap,
        device=z0.device,
        dtype=z0.dtype,
    )
    if w is None:
        return z0
    view_shape = (bsz, m) + (1,) * (cand.ndim - 2)
    z_mix = (w.view(view_shape) * cand).sum(dim=1)
    return (1.0 - float(hull_lambda)) * z0 + float(hull_lambda) * z_mix


__all__ = [
    "identity_collapsed_weight_metrics", "initial_hull_latent",
    "initial_hull_weights", "latent_hull_geometry",
]
