"""Pure geometry for the isolated Diverse Latent AugMax sandbox.

The functions in this module do not own models, configuration files, global
random state, or experiment state.  They operate in the *train400-fitted*
ECGTwin latent standardizer domain and make the following geometry explicit:

``clean -> E corrupted endpoints -> greedily selected K directions -> view``.

The first selected endpoint maximizes the classifier Bernoulli-probability
shift from clean.  Subsequent endpoints maximize their minimum weighted
distance to the already selected set.  The default distance is an equal blend
of normalized latent cosine distance and prediction RMS distance.

The final view is a clean-anchored convex combination.  ``m`` controls how far
the view moves away from clean and simplex weights select a direction inside
the chosen endpoint hull.  A calibration-owned global standardized-latent RMS
cap bounds the move.  No batch statistic is used as a substitute for that cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StandardizedEndpointGeometry:
    """Clean and candidate endpoints in one fixed standardized latent domain."""

    clean: torch.Tensor
    endpoints: torch.Tensor
    deltas: torch.Tensor
    unit_directions: torch.Tensor
    radii_rms: torch.Tensor
    finite_mask: torch.Tensor


@dataclass(frozen=True)
class DiversitySelection:
    """Per-record greedy K-of-E endpoint selection and selector diagnostics."""

    indices: torch.Tensor
    endpoints: torch.Tensor
    deltas: torch.Tensor
    unit_directions: torch.Tensor
    radii_rms: torch.Tensor
    prediction_signatures: torch.Tensor
    clean_prediction_shift_rms: torch.Tensor
    selected_pairwise_cosine: torch.Tensor
    selected_pairwise_prediction_rms: torch.Tensor
    selected_pairwise_composite_distance: torch.Tensor
    candidate_prediction_variance: torch.Tensor


@dataclass(frozen=True)
class LatentProjection:
    """One bounded clean-anchored simplex point in standardized latent space."""

    standardized_latent: torch.Tensor
    raw_simplex_weights: torch.Tensor
    raw_m: torch.Tensor
    effective_endpoint_weights: torch.Tensor
    effective_m: torch.Tensor
    pre_projection_radius_rms: torch.Tensor
    post_projection_radius_rms: torch.Tensor
    projection_scale: torch.Tensor
    radius_cap_rms: float | None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("generator and requested tensor device must match")
    if device.type == "cuda":
        current = torch.cuda.current_device()
        expected = current if device.index is None else int(device.index)
        actual = current if generator_device.index is None else int(generator_device.index)
        if expected != actual:
            raise ValueError("generator and requested CUDA device indices must match")


def build_standardized_endpoint_geometry(
    clean_latent: torch.Tensor,
    candidate_latents: torch.Tensor,
    standardizer: Any,
    *,
    epsilon: float = 1.0e-8,
) -> StandardizedEndpointGeometry:
    """Transform clean and E endpoints with the supplied train-only standardizer."""

    if not isinstance(clean_latent, torch.Tensor) or clean_latent.ndim != 3:
        raise TypeError("clean_latent must be a rank-3 torch.Tensor")
    if not isinstance(candidate_latents, torch.Tensor) or candidate_latents.ndim != 4:
        raise TypeError("candidate_latents must be a rank-4 torch.Tensor")
    if candidate_latents.shape[0] != clean_latent.shape[0]:
        raise ValueError("clean and candidate batch sizes differ")
    if tuple(candidate_latents.shape[2:]) != tuple(clean_latent.shape[1:]):
        raise ValueError("clean and candidate latent shapes differ")
    if candidate_latents.shape[1] < 1:
        raise ValueError("at least one candidate endpoint is required")
    if not clean_latent.is_floating_point() or not candidate_latents.is_floating_point():
        raise TypeError("latents must be floating point")
    if clean_latent.device != candidate_latents.device:
        raise ValueError("clean and candidate latents must share a device")
    transform = getattr(standardizer, "transform", None)
    if not callable(transform):
        raise TypeError("standardizer must provide transform(tensor)")
    resolved_epsilon = _finite_nonnegative(epsilon, "epsilon")
    if resolved_epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    clean = transform(clean_latent).to(dtype=torch.float32)
    endpoints = transform(candidate_latents).to(dtype=torch.float32)
    deltas = endpoints - clean.unsqueeze(1)
    flat = deltas.flatten(2)
    dimensions = int(flat.shape[2])
    radii = torch.linalg.vector_norm(flat, dim=2) / math.sqrt(float(dimensions))
    norms = torch.linalg.vector_norm(flat, dim=2, keepdim=True)
    unit = flat / norms.clamp_min(resolved_epsilon)
    finite = (
        torch.isfinite(clean).flatten(1).all(dim=1, keepdim=True)
        & torch.isfinite(endpoints).flatten(2).all(dim=2)
        & torch.isfinite(radii)
        & (norms.squeeze(2) > resolved_epsilon)
    )
    return StandardizedEndpointGeometry(
        clean=torch.nan_to_num(clean).contiguous(),
        endpoints=torch.nan_to_num(endpoints).contiguous(),
        deltas=torch.nan_to_num(deltas).contiguous(),
        unit_directions=torch.nan_to_num(unit).contiguous(),
        radii_rms=torch.nan_to_num(radii).contiguous(),
        finite_mask=finite.contiguous(),
    )


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("candidate gather shapes are incompatible")
    shape = (indices.shape[0], indices.shape[1]) + (1,) * (values.ndim - 2)
    expanded = indices.view(shape).expand(
        indices.shape[0], indices.shape[1], *values.shape[2:]
    )
    return torch.gather(values, 1, expanded).contiguous()


def greedy_select_diverse_directions(
    geometry: StandardizedEndpointGeometry,
    clean_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    *,
    first_scores: torch.Tensor | None = None,
    selected_count: int = 3,
    latent_cosine_weight: float = 0.5,
    prediction_rms_weight: float = 0.5,
) -> DiversitySelection:
    """Select K endpoints with one primary score then greedy max-min diversity.

    Cosine distance is mapped to ``[0, 1]`` as ``(1-cosine)/2`` so its scale
    matches Bernoulli prediction RMS distance.  Ties are deterministic because
    ``torch.argmax`` returns the first maximum and candidate order is stable.
    ``first_scores=None`` preserves the original clean-prediction-shift rule;
    callers may instead supply an aligned per-record score such as supervised
    endpoint BCE.  Invalid or zero-length directions are never selected.
    """

    if not isinstance(geometry, StandardizedEndpointGeometry):
        raise TypeError("geometry must be StandardizedEndpointGeometry")
    batch_size, candidate_count = geometry.radii_rms.shape
    selected = _positive_int(selected_count, "selected_count")
    if selected > candidate_count:
        raise ValueError("selected_count cannot exceed candidate count")
    if clean_probabilities.shape[0] != batch_size or clean_probabilities.ndim != 2:
        raise ValueError("clean_probabilities must have shape (B,C)")
    if candidate_probabilities.shape != (
        batch_size,
        candidate_count,
        clean_probabilities.shape[1],
    ):
        raise ValueError("candidate_probabilities must have shape (B,E,C)")
    if clean_probabilities.device != geometry.clean.device or (
        candidate_probabilities.device != geometry.clean.device
    ):
        raise ValueError("geometry and predictions must share a device")
    latent_weight = _finite_nonnegative(
        latent_cosine_weight, "latent_cosine_weight"
    )
    prediction_weight = _finite_nonnegative(
        prediction_rms_weight, "prediction_rms_weight"
    )
    if not math.isclose(latent_weight + prediction_weight, 1.0, abs_tol=1.0e-8):
        raise ValueError("selector distance weights must sum to one")
    probability_finite = (
        torch.isfinite(clean_probabilities).all(dim=1, keepdim=True)
        & torch.isfinite(candidate_probabilities).all(dim=2)
    )
    if first_scores is None:
        supplied_first_scores = None
    else:
        if first_scores.shape != (batch_size, candidate_count):
            raise ValueError("first_scores must have shape (B,E)")
        if first_scores.device != geometry.clean.device:
            raise ValueError("first_scores must share the geometry device")
        supplied_first_scores = first_scores.to(dtype=torch.float32)
    score_finite = (
        torch.ones_like(geometry.finite_mask)
        if supplied_first_scores is None
        else torch.isfinite(supplied_first_scores)
    )
    eligible = geometry.finite_mask & probability_finite & score_finite
    if bool((eligible.sum(dim=1) < selected).any()):
        raise ValueError("fewer than K finite non-zero candidate directions")

    clean_shift = (
        candidate_probabilities - clean_probabilities.unsqueeze(1)
    ).square().mean(dim=2).sqrt()
    primary_scores = clean_shift if supplied_first_scores is None else supplied_first_scores
    first_scores_masked = torch.where(
        eligible,
        primary_scores,
        torch.full_like(primary_scores, -torch.inf),
    )
    chosen = first_scores_masked.argmax(dim=1, keepdim=True)
    chosen_mask = torch.zeros_like(eligible)
    chosen_mask.scatter_(1, chosen, True)

    unit = geometry.unit_directions
    latent_cosine = torch.einsum("bed,bfd->bef", unit, unit).clamp(-1.0, 1.0)
    latent_distance = (1.0 - latent_cosine) * 0.5
    prediction_delta = (
        candidate_probabilities.unsqueeze(2) - candidate_probabilities.unsqueeze(1)
    )
    prediction_distance = prediction_delta.square().mean(dim=3).sqrt()
    composite = latent_weight * latent_distance + prediction_weight * prediction_distance

    for _ in range(1, selected):
        chosen_so_far = chosen
        pair = torch.gather(
            composite,
            2,
            chosen_so_far.unsqueeze(1).expand(batch_size, candidate_count, -1),
        )
        min_distance = pair.amin(dim=2)
        available = eligible & ~chosen_mask
        scores = torch.where(
            available,
            min_distance,
            torch.full_like(min_distance, -torch.inf),
        )
        next_index = scores.argmax(dim=1, keepdim=True)
        chosen = torch.cat((chosen, next_index), dim=1)
        chosen_mask.scatter_(1, next_index, True)

    selected_endpoints = _gather_candidates(geometry.endpoints, chosen)
    selected_deltas = _gather_candidates(geometry.deltas, chosen)
    selected_unit = _gather_candidates(geometry.unit_directions, chosen)
    selected_radii = torch.gather(geometry.radii_rms, 1, chosen)
    selected_predictions = _gather_candidates(candidate_probabilities, chosen)
    pairwise_cosine = torch.einsum(
        "bkd,bld->bkl", selected_unit, selected_unit
    ).clamp(-1.0, 1.0)
    pairwise_prediction = (
        selected_predictions.unsqueeze(2) - selected_predictions.unsqueeze(1)
    ).square().mean(dim=3).sqrt()
    pairwise_composite = (
        latent_weight * (1.0 - pairwise_cosine) * 0.5
        + prediction_weight * pairwise_prediction
    )
    prediction_variance = candidate_probabilities.var(dim=1, correction=0).mean(dim=1)
    return DiversitySelection(
        indices=chosen.contiguous(),
        endpoints=selected_endpoints,
        deltas=selected_deltas,
        unit_directions=selected_unit,
        radii_rms=selected_radii.contiguous(),
        prediction_signatures=selected_predictions,
        clean_prediction_shift_rms=torch.gather(clean_shift, 1, chosen).contiguous(),
        selected_pairwise_cosine=pairwise_cosine.contiguous(),
        selected_pairwise_prediction_rms=pairwise_prediction.contiguous(),
        selected_pairwise_composite_distance=pairwise_composite.contiguous(),
        candidate_prediction_variance=prediction_variance.contiguous(),
    )


def sample_simplex_state(
    batch_size: int,
    selected_count: int,
    *,
    device: str | torch.device,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw Dirichlet(1) endpoint weights and an independent uniform ``m``."""

    batch = _positive_int(batch_size, "batch_size")
    count = _positive_int(selected_count, "selected_count")
    resolved_device = torch.device(device)
    _validate_generator(generator, resolved_device)
    if not dtype.is_floating_point:
        raise TypeError("simplex state requires a floating dtype")
    uniform = torch.rand(
        batch,
        count,
        device=resolved_device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    exponential = -uniform.log()
    weights = exponential / exponential.sum(dim=1, keepdim=True)
    m = torch.rand(
        batch,
        device=resolved_device,
        dtype=torch.float32,
        generator=generator,
    )
    return weights.to(dtype=dtype).contiguous(), m.to(dtype=dtype).contiguous()


def project_clean_anchored_simplex(
    clean_standardized: torch.Tensor,
    selected_standardized: torch.Tensor,
    simplex_weights: torch.Tensor,
    m: torch.Tensor,
    *,
    radius_cap_rms: float | None,
    smoke_allow_missing_radius_cap: bool = False,
    epsilon: float = 1.0e-8,
) -> LatentProjection:
    """Project a clean-anchored simplex point to one fixed global RMS cap."""

    if clean_standardized.ndim != 3 or selected_standardized.ndim != 4:
        raise ValueError("standardized latents must have shapes (B,...) and (B,K,...)")
    batch_size, selected_count = selected_standardized.shape[:2]
    if clean_standardized.shape[0] != batch_size or tuple(
        selected_standardized.shape[2:]
    ) != tuple(clean_standardized.shape[1:]):
        raise ValueError("clean and selected standardized latents are incompatible")
    if simplex_weights.shape != (batch_size, selected_count):
        raise ValueError("simplex_weights must have shape (B,K)")
    if m.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("m must have shape (B,) or (B,1)")
    if clean_standardized.device != selected_standardized.device or (
        simplex_weights.device != clean_standardized.device
        or m.device != clean_standardized.device
    ):
        raise ValueError("all projection tensors must share a device")
    resolved_epsilon = _finite_nonnegative(epsilon, "epsilon")
    if resolved_epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if radius_cap_rms is None:
        if not smoke_allow_missing_radius_cap:
            raise ValueError(
                "radius_cap_rms is required outside explicit smoke mode"
            )
        cap = None
    else:
        cap = _finite_nonnegative(radius_cap_rms, "radius_cap_rms")
        if cap <= 0.0:
            raise ValueError("radius_cap_rms must be positive")

    weights = simplex_weights.to(dtype=clean_standardized.dtype)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("simplex_weights must be finite and non-negative")
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(resolved_epsilon)
    raw_m = m.reshape(batch_size).to(dtype=clean_standardized.dtype)
    if not bool(torch.isfinite(raw_m).all()) or bool((raw_m < 0).any()) or bool(
        (raw_m > 1).any()
    ):
        raise ValueError("m must be finite and lie in [0,1]")

    deltas = selected_standardized - clean_standardized.unsqueeze(1)
    mixed_delta = (
        deltas
        * weights.view(batch_size, selected_count, *([1] * (deltas.ndim - 2)))
    ).sum(dim=1)
    proposed_delta = mixed_delta * raw_m.view(
        batch_size, *([1] * (clean_standardized.ndim - 1))
    )
    dimensions = int(proposed_delta[0].numel())
    pre_radius = torch.linalg.vector_norm(proposed_delta.flatten(1), dim=1) / math.sqrt(
        float(dimensions)
    )
    if cap is None:
        scale = torch.ones_like(pre_radius)
    else:
        scale = torch.minimum(
            torch.ones_like(pre_radius),
            torch.full_like(pre_radius, cap) / pre_radius.clamp_min(resolved_epsilon),
        )
    projected_delta = proposed_delta * scale.view(
        batch_size, *([1] * (clean_standardized.ndim - 1))
    )
    effective_m = raw_m * scale
    endpoint_weights = weights * effective_m.unsqueeze(1)
    effective_weights = torch.cat(
        (1.0 - effective_m.unsqueeze(1), endpoint_weights), dim=1
    )
    post_radius = torch.linalg.vector_norm(projected_delta.flatten(1), dim=1) / math.sqrt(
        float(dimensions)
    )
    return LatentProjection(
        standardized_latent=(clean_standardized + projected_delta).contiguous(),
        raw_simplex_weights=weights.contiguous(),
        raw_m=raw_m.contiguous(),
        effective_endpoint_weights=effective_weights.contiguous(),
        effective_m=effective_m.contiguous(),
        pre_projection_radius_rms=pre_radius.contiguous(),
        post_projection_radius_rms=post_radius.contiguous(),
        projection_scale=scale.contiguous(),
        radius_cap_rms=cap,
    )


def mix_endpoint_residuals(
    endpoint_residuals: torch.Tensor,
    effective_endpoint_weights: torch.Tensor,
) -> torch.Tensor:
    """Mix clean plus K endpoint residuals with projected convex weights."""

    if endpoint_residuals.ndim < 4:
        raise ValueError("endpoint_residuals must have shape (B,1+K,...)")
    batch_size, endpoint_count = endpoint_residuals.shape[:2]
    if effective_endpoint_weights.shape != (batch_size, endpoint_count):
        raise ValueError("effective endpoint weight shape does not match residuals")
    if endpoint_residuals.device != effective_endpoint_weights.device:
        raise ValueError("residuals and weights must share a device")
    weights = effective_endpoint_weights.to(dtype=endpoint_residuals.dtype).view(
        batch_size, endpoint_count, *([1] * (endpoint_residuals.ndim - 2))
    )
    return (endpoint_residuals * weights).sum(dim=1).contiguous()


__all__ = [
    "DiversitySelection",
    "LatentProjection",
    "StandardizedEndpointGeometry",
    "build_standardized_endpoint_geometry",
    "greedy_select_diverse_directions",
    "mix_endpoint_residuals",
    "project_clean_anchored_simplex",
    "sample_simplex_state",
]
