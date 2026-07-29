"""Sandbox runtime for coverage-preserving fixed20 LHAT/AugMix views.

The locked Direct+fixed20 schedule remains trainer-owned: one clean exposure,
twenty canonical composition exposures, and one optimizer step per base batch.
This adapter changes only the waveform used by each corrupted exposure.  It
keeps the full canonical corruption anchor and adds small, independently
sampled random-corruption and LHAT residual directions.  One LHAT waveform is
generated on the clean exposure and reused across all twenty compositions.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corruption import generate_canonical_corruption
from core.methods.executor import _derive_seed as _derive_executor_seed
from core.methods.contracts import (
    BASE_VIEW_NAME,
    NodeContext,
    Provenance,
    ViewBundle,
    ViewValue,
    WaveformView,
)
from core.methods.registry import CompiledMethod
from core.methods.runtime import MethodViewRuntime, _quality_mask
from models.contracts import validate_model_output
from models.factory import get_model_spec
from models.input_adapter import prepare_canonical_model_input
from models.vae import decode_to_ptbxl_waveform, prepare_ecgtwin_encoder_input
from runtime_adapter import _frozen_classifier_for_attack, _frozen_vae_components


METHOD_ID = "coverage_anchor_m20_lhat_augmix"
CALIBRATED_METHOD_ID = "coverage_anchor_augmax_calibrated"
SUPPORTED_METHOD_IDS = frozenset({METHOD_ID, CALIBRATED_METHOD_ID})
MIX_NODE_ID = "coverage_anchor_mix"
RESOURCE_BRIDGE_NODE_ID = "resource_bridge"
RANDOM_RESIDUAL_SCALE = 0.15
LHAT_RESIDUAL_SCALE = 0.10
CALIBRATED_DOSES = (0.0, 0.5, 1.0, 1.5)
MAX_PER_RECORD_BCE_GAIN = 0.05
DIRECT_FIXED20_PROFILE_NAME = "direct_depth23_fixed20"
DIRECT_FIXED20_RNG_NAMESPACE = "method_direct_depth23_fixed20"
DIRECT_FIXED20_CORRUPTION_NODE_ID = "depth23_corruption"
DIRECT_FIXED20_CORRUPTION_STREAM = "composition_and_operators"
TWO_FACTOR_MATCHED_PROFILE_NAME = "coverage_anchor_two_factor_matched_v1"
TWO_FACTOR_MATCHED_RAW_NAMESPACE = "coverage_anchor_two_factor/raw"
TWO_FACTOR_MATCHED_LHAT_NAMESPACE = "coverage_anchor_two_factor/lhat"
TWO_FACTOR_MATCHED_RAW_NODE_ID = MIX_NODE_ID
TWO_FACTOR_MATCHED_LHAT_NODE_ID = "lhat"
COUPLED_THREECHAIN_DISABLED = "disabled"
COUPLED_THREECHAIN_RAW_CONTROL = "raw_control"
COUPLED_THREECHAIN_ORTHOGONAL_LHAT = "orthogonal_lhat"
COUPLED_THREECHAIN_MODES = frozenset(
    {
        COUPLED_THREECHAIN_DISABLED,
        COUPLED_THREECHAIN_RAW_CONTROL,
        COUPLED_THREECHAIN_ORTHOGONAL_LHAT,
    }
)


@dataclass(frozen=True)
class _FactorizedResidualSelection:
    """One replayable raw/LHAT draw per record without a KxB waveform grid."""

    waveform: torch.Tensor
    assigned_indices: torch.Tensor
    assigned_coefficients: torch.Tensor
    effective_coefficients: torch.Tensor
    quality_accepted: torch.Tensor
    quality_fallback: torch.Tensor
    raw_unavailable: torch.Tensor
    lhat_unavailable: torch.Tensor
    selection_entropy: torch.Tensor


@dataclass(frozen=True)
class _PendingFactorAudit:
    """Device-resident per-composition audit, materialized once per group."""

    composition_index: int
    sample_ids: tuple[str, ...]
    assigned_indices: torch.Tensor
    effective_coefficients: torch.Tensor
    quality_accepted: torch.Tensor
    quality_fallback: torch.Tensor
    raw_unavailable: torch.Tensor
    lhat_unavailable: torch.Tensor


@dataclass(frozen=True)
class _CoupledThreechainCache:
    """One replayable base-group endpoint set reused by all fixed20 views."""

    raw_chain1: torch.Tensor
    raw_chain2: torch.Tensor
    hard_chain: torch.Tensor
    raw_chain1_valid: torch.Tensor
    raw_chain2_valid: torch.Tensor
    hard_chain_valid: torch.Tensor
    raw_chain_depths: torch.Tensor
    raw_chain_operator_mask: torch.Tensor
    diagnostic_values: Mapping[str, float | torch.Tensor]


_FACTOR_AUDIT_RECORDS: list[dict[str, Any]] = []


def _anchor_floored_dirichlet_weights(
    *,
    batch_size: int,
    canonical_mass_floor: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample replayable Dirichlet(1,1,1) weights with a canonical floor."""

    if batch_size <= 0:
        raise ValueError("coupled three-chain batch_size must be positive")
    floor = float(canonical_mass_floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("canonical_mass_floor must lie in [0,1)")
    uniform = torch.rand(
        batch_size,
        3,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    weights = -uniform.log()
    weights = weights / weights.sum(dim=1, keepdim=True)
    residual_mass = 1.0 - floor
    weights[:, 0] = floor + residual_mass * weights[:, 0]
    weights[:, 1:] *= residual_mass
    return weights.to(dtype=dtype).contiguous()


def _masked_batch_absolute_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a device-resident mean absolute cosine and applicability rate."""

    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("cosine inputs must be matching batched tensors")
    if valid.shape != (left.shape[0],):
        raise ValueError("cosine validity mask must have shape (B,)")
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    left_norm = left_flat.norm(p=2, dim=1)
    right_norm = right_flat.norm(p=2, dim=1)
    resolved_epsilon = float(epsilon)
    applicable = (
        valid
        & torch.isfinite(left_flat).all(dim=1)
        & torch.isfinite(right_flat).all(dim=1)
        & (left_norm > resolved_epsilon)
        & (right_norm > resolved_epsilon)
    )
    cosine = (
        (left_flat * right_flat).sum(dim=1).abs()
        / (left_norm * right_norm).clamp_min(resolved_epsilon)
    )
    weight = applicable.to(dtype=cosine.dtype)
    mean = (cosine * weight).sum() / weight.sum().clamp_min(1.0)
    return mean.detach(), weight.mean().detach()


def _orthogonalize_lhat_against_raw_span(
    *,
    clean_standardized: torch.Tensor,
    raw_standardized: torch.Tensor,
    lhat_standardized: torch.Tensor,
    raw_valid: torch.Tensor,
    lhat_valid: torch.Tensor,
    radius_ratio: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Remove LHAT components already represented by two raw-chain directions.

    All geometry is evaluated in the train-pool standardized latent domain.
    Modified Gram-Schmidt is used per record so rank-deficient raw directions
    become zero basis vectors instead of injecting an arbitrary QR basis.
    """

    if clean_standardized.ndim < 2:
        raise ValueError("clean standardized latent must be batched")
    if raw_standardized.ndim != clean_standardized.ndim + 1:
        raise ValueError("raw standardized latents must have shape (B,K,...)")
    if raw_standardized.shape[0] != clean_standardized.shape[0]:
        raise ValueError("raw and clean standardized batch sizes differ")
    if raw_standardized.shape[1] != 2:
        raise ValueError("coupled three-chain geometry requires two raw directions")
    if tuple(raw_standardized.shape[2:]) != tuple(clean_standardized.shape[1:]):
        raise ValueError("raw and clean standardized latent shapes differ")
    if lhat_standardized.shape != clean_standardized.shape:
        raise ValueError("LHAT and clean standardized latent shapes differ")
    batch = int(clean_standardized.shape[0])
    if raw_valid.shape != (batch, 2) or lhat_valid.shape != (batch,):
        raise ValueError("coupled three-chain validity masks have invalid shapes")
    ratio = float(radius_ratio)
    resolved_epsilon = float(epsilon)
    if not torch.isfinite(torch.tensor(ratio)) or ratio <= 0.0:
        raise ValueError("radius_ratio must be finite and positive")
    if not torch.isfinite(torch.tensor(resolved_epsilon)) or resolved_epsilon <= 0.0:
        raise ValueError("orthogonality epsilon must be finite and positive")

    clean_flat = clean_standardized.flatten(1)
    raw_flat = raw_standardized.flatten(2)
    lhat_flat = lhat_standardized.flatten(1)
    raw_directions = raw_flat - clean_flat[:, None, :]
    h = lhat_flat - clean_flat
    basis: list[torch.Tensor] = []
    rank = torch.zeros(batch, device=clean_flat.device, dtype=torch.float32)
    for index in range(2):
        vector = raw_directions[:, index]
        for previous in basis:
            vector = vector - (vector * previous).sum(
                dim=1, keepdim=True
            ) * previous
        norm = vector.norm(p=2, dim=1)
        active = raw_valid[:, index] & (norm > resolved_epsilon)
        unit = torch.where(
            active[:, None],
            vector / norm.clamp_min(resolved_epsilon)[:, None],
            torch.zeros_like(vector),
        )
        basis.append(unit)
        rank = rank + active.to(dtype=torch.float32)

    projection = torch.zeros_like(h)
    for unit in basis:
        projection = projection + (h * unit).sum(dim=1, keepdim=True) * unit
    perpendicular = h - projection
    raw_norms = raw_directions.norm(p=2, dim=2)
    valid_raw_count = raw_valid.sum(dim=1).clamp_min(1)
    reference_radius = (
        raw_norms * raw_valid.to(dtype=raw_norms.dtype)
    ).sum(dim=1) / valid_raw_count.to(dtype=raw_norms.dtype)
    radius_cap = ratio * reference_radius
    perpendicular_norm = perpendicular.norm(p=2, dim=1)
    radius_scale = torch.clamp(
        radius_cap / perpendicular_norm.clamp_min(resolved_epsilon),
        max=1.0,
    )
    final_delta = perpendicular * radius_scale[:, None]
    final_norm = final_delta.norm(p=2, dim=1)
    post_projection = torch.zeros_like(final_delta)
    for unit in basis:
        post_projection = post_projection + (
            final_delta * unit
        ).sum(dim=1, keepdim=True) * unit
    h_norm = h.norm(p=2, dim=1)
    projection_norm = projection.norm(p=2, dim=1)
    post_projection_norm = post_projection.norm(p=2, dim=1)
    valid = (
        raw_valid.all(dim=1)
        & lhat_valid
        & torch.isfinite(final_delta).all(dim=1)
        & (final_norm > resolved_epsilon)
    )
    projected = clean_flat + final_delta
    projected = projected.reshape_as(clean_standardized).contiguous()
    diagnostics = {
        "raw_direction_rank": rank,
        "raw_reference_radius": reference_radius,
        "lhat_original_radius": h_norm,
        "lhat_parallel_fraction": projection_norm
        / h_norm.clamp_min(resolved_epsilon),
        "lhat_retained_fraction": final_norm / h_norm.clamp_min(resolved_epsilon),
        "lhat_radius_scale": radius_scale,
        "lhat_final_radius": final_norm,
        "lhat_post_subspace_fraction": post_projection_norm
        / final_norm.clamp_min(resolved_epsilon),
    }
    return projected, valid, diagnostics


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reset_factor_audit_records() -> None:
    _FACTOR_AUDIT_RECORDS.clear()


def factor_audit_payload() -> dict[str, Any] | None:
    """Return the completed paired-factor audit without mutating it."""

    if not _FACTOR_AUDIT_RECORDS:
        return None
    records = tuple(_FACTOR_AUDIT_RECORDS)
    return {
        "schema_version": 1,
        "audit_scope": "factorized_fixed20_assignment_and_quality",
        "materialization_policy": (
            "one_device_to_host_transfer_after_composition_19_per_base_group"
        ),
        "group_count": len(records),
        "raw_assignment_sha256": _canonical_sha256(
            [record["raw_assignment_sha256"] for record in records]
        ),
        "lhat_assignment_sha256": _canonical_sha256(
            [record["lhat_assignment_sha256"] for record in records]
        ),
        "groups": [dict(record) for record in records],
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    """Persist the compact factor audit before the outer run finalizer."""

    payload = factor_audit_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "factor_assignment_audit.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _anchor_floored_simplex_weights(
    *,
    batch_size: int,
    candidate_count: int,
    anchor_mass_floor: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build replayable three-chain AugMix weights with an exact-anchor fallback."""

    if batch_size <= 0:
        raise ValueError("simplex batch_size must be positive")
    if candidate_count < 4:
        raise ValueError("simplex candidate_count must be at least four")
    if not 0.0 <= float(anchor_mass_floor) < 1.0:
        raise ValueError("simplex anchor_mass_floor must lie in [0,1)")
    weights = torch.zeros(
        candidate_count,
        batch_size,
        3,
        device=device,
        dtype=dtype,
    )
    weights[0, :, 0] = 1.0
    residual_mass = 1.0 - float(anchor_mass_floor)
    weights[1, :, 0] = float(anchor_mass_floor)
    weights[1, :, 1] = residual_mass
    weights[2, :, 0] = float(anchor_mass_floor)
    weights[2, :, 2] = residual_mass
    weights[3, :, 0] = float(anchor_mass_floor)
    weights[3, :, 1:] = 0.5 * residual_mass
    if candidate_count > 4:
        # -log(U) are iid Gamma(1,1), so normalizing them is an exact
        # Dirichlet(1,1,1) draw while retaining an explicit torch generator.
        uniform = torch.rand(
            candidate_count - 4,
            batch_size,
            3,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).clamp_min_(torch.finfo(torch.float32).tiny)
        sampled = -uniform.log()
        sampled = sampled / sampled.sum(dim=2, keepdim=True)
        sampled[:, :, 0] = (
            float(anchor_mass_floor) + residual_mass * sampled[:, :, 0]
        )
        sampled[:, :, 1:] *= residual_mass
        weights[4:] = sampled.to(dtype=dtype)
    return weights


def _sample_replayable_grid_indices(
    admissible: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a configured residual dose without using classifier scores."""

    if admissible.ndim != 2:
        raise ValueError("residual admissibility must have shape (candidates,batch)")
    if probabilities.ndim != 1 or probabilities.shape[0] != admissible.shape[0]:
        raise ValueError("residual probabilities must match the candidate count")
    if not bool(torch.isfinite(probabilities).all()) or bool(
        (probabilities < 0.0).any()
    ):
        raise ValueError("residual probabilities must be finite and non-negative")
    if float(probabilities.sum()) <= 0.0:
        raise ValueError("residual probabilities must contain positive mass")

    weights = admissible.transpose(0, 1).to(dtype=torch.float32)
    weights = weights * probabilities.to(
        device=weights.device, dtype=weights.dtype
    ).unsqueeze(0)
    missing = weights.sum(dim=1) <= 0.0
    if bool(missing.any()):
        weights[missing] = 0.0
        weights[missing, 0] = 1.0
    normalized = weights / weights.sum(dim=1, keepdim=True)
    selected = torch.multinomial(
        normalized,
        1,
        replacement=True,
        generator=generator,
    ).squeeze(1)
    entropy = -(
        normalized * normalized.clamp_min(1.0e-12).log()
    ).sum(dim=1)
    return selected, entropy


def _project_standardized_hull(
    anchor: torch.Tensor,
    candidates: torch.Tensor,
    weight_logits: torch.Tensor,
    *,
    hull_lambda: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.softmax(weight_logits, dim=1)
    mixed = (
        candidates
        * weights.view(weights.shape[0], weights.shape[1], 1, 1)
    ).sum(dim=1)
    proposed = anchor + float(hull_lambda) * (mixed - anchor)
    delta = proposed - anchor
    norm = delta.flatten(1).norm(p=2, dim=1)
    scale = torch.clamp(float(epsilon) / norm.clamp_min(1.0e-12), max=1.0)
    projected = anchor + delta * scale.view(-1, 1, 1)
    return projected, weights, scale


def _finite_contract_float(
    value: Any, *, name: str, minimum: float = 0.0
) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _parse_factor_distribution(
    payload: Any,
    *,
    name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(payload, Mapping) or set(payload) != {"values", "probabilities"}:
        raise TypeError(f"{name} must contain exactly values and probabilities")
    raw_values = payload["values"]
    raw_probabilities = payload["probabilities"]
    if not isinstance(raw_values, list) or not isinstance(raw_probabilities, list):
        raise TypeError(f"{name} values/probabilities must be lists")
    if not raw_values or len(raw_values) != len(raw_probabilities):
        raise ValueError(f"{name} values/probabilities must have equal non-zero length")
    values = tuple(
        _finite_contract_float(value, name=f"{name}.values[{index}]")
        for index, value in enumerate(raw_values)
    )
    probabilities = tuple(
        _finite_contract_float(value, name=f"{name}.probabilities[{index}]")
        for index, value in enumerate(raw_probabilities)
    )
    if values[0] != 0.0 or len(set(values)) != len(values):
        raise ValueError(f"{name} values must begin with zero and be unique")
    if sum(probabilities) <= 0.0:
        raise ValueError(f"{name} probabilities must contain positive mass")
    return values, probabilities


def _sample_factor_coefficients(
    *,
    values: Sequence[float],
    probabilities: Sequence[float],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw one coefficient per record from an isolated factor RNG stream."""

    if batch_size <= 0:
        raise ValueError("factor coefficient batch_size must be positive")
    if not values or len(values) != len(probabilities):
        raise ValueError("factor values/probabilities must have equal non-zero length")
    probability_tensor = torch.as_tensor(
        probabilities,
        device=device,
        dtype=torch.float32,
    )
    probability_tensor = probability_tensor / probability_tensor.sum()
    expanded = probability_tensor.unsqueeze(0).expand(batch_size, -1)
    selected_index = torch.multinomial(
        expanded,
        1,
        replacement=True,
        generator=generator,
    ).squeeze(1)
    value_tensor = torch.as_tensor(values, device=device, dtype=dtype)
    selected = value_tensor.index_select(0, selected_index)
    entropy = -(
        probability_tensor * probability_tensor.clamp_min(1.0e-12).log()
    ).sum().expand(batch_size)
    return selected, selected_index, entropy


def _lhat_adversarial_residual(
    lhat: WaveformView,
    clean: WaveformView,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decode(z_adv)-decode(z_anchor), excluding VAE reconstruction error."""

    anchor_reconstruction = lhat.metadata.get("anchor_waveform_raw")
    if not isinstance(anchor_reconstruction, torch.Tensor):
        raise RuntimeError(
            "VAE-LHAT view is missing its decoded anchor reconstruction"
        )
    if anchor_reconstruction.shape != lhat.waveform.shape:
        raise RuntimeError("VAE-LHAT anchor reconstruction shape changed")
    if anchor_reconstruction.device != lhat.waveform.device:
        raise RuntimeError("VAE-LHAT anchor reconstruction device changed")
    residual = lhat.waveform - anchor_reconstruction.to(
        dtype=lhat.waveform.dtype
    )
    residual = torch.where(
        lhat.valid_mask[:, None, None], residual, torch.zeros_like(residual)
    )
    return residual, anchor_reconstruction


def _materialize_scalar_diagnostics(
    values: Mapping[str, float | int | torch.Tensor],
) -> dict[str, float]:
    """Resolve device scalar diagnostics with at most one host synchronization."""

    resolved: dict[str, float] = {}
    tensor_names: list[str] = []
    tensors: list[torch.Tensor] = []
    device: torch.device | None = None
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            resolved[name] = float(value)
            continue
        if value.numel() != 1:
            raise ValueError(f"diagnostic {name!r} must be scalar")
        if device is None:
            device = value.device
        tensor_names.append(name)
        tensors.append(value.detach().reshape(()))
    if tensors:
        assert device is not None
        packed = torch.stack(
            tuple(value.to(device=device, dtype=torch.float32) for value in tensors)
        )
        host_values = packed.cpu().tolist()
        resolved.update(
            {
                name: float(value)
                for name, value in zip(tensor_names, host_values, strict=True)
            }
        )
    return resolved


def _lhat_availability_fraction(
    valid_mask: torch.Tensor,
    *,
    factor_enabled: bool,
) -> float | torch.Tensor:
    """Keep a disabled identity bridge from masquerading as valid LHAT."""

    if not factor_enabled:
        return 0.0
    return valid_mask.float().mean().detach()


def _scientific_factor_toggles(contracts: Mapping[str, Any]) -> tuple[bool, bool]:
    """Resolve the two causal factors used by the matched A/B/C/D ablation."""

    if "scientific_factors" not in contracts:
        raise ValueError("scientific_factors must be declared explicitly")
    raw = contracts["scientific_factors"]
    if not isinstance(raw, Mapping):
        raise TypeError("scientific_factors must be a mapping")
    if set(raw) != {"raw_diversity", "vae_lhat"}:
        raise ValueError(
            "scientific_factors must contain exactly raw_diversity and vae_lhat"
        )
    if not all(
        isinstance(raw[name], bool)
        for name in ("raw_diversity", "vae_lhat")
    ):
        raise TypeError("scientific factor toggles must be booleans")
    return bool(raw["raw_diversity"]), bool(raw["vae_lhat"])


def _validate_factor_grid_mass(
    grid: Sequence[tuple[float, float]],
    probabilities: Sequence[float],
    *,
    raw_diversity_enabled: bool,
    vae_lhat_enabled: bool,
) -> None:
    """Fail closed when a disabled factor still receives sampling mass."""

    if len(grid) != len(probabilities):
        raise ValueError("factor grid and probability lengths differ")
    for index, ((raw_coefficient, lhat_coefficient), probability) in enumerate(
        zip(grid, probabilities)
    ):
        if probability <= 0.0:
            continue
        if not raw_diversity_enabled and raw_coefficient != 0.0:
            raise ValueError(
                "disabled raw-diversity factor has positive mass at grid index "
                f"{index}"
            )
        if not vae_lhat_enabled and lhat_coefficient != 0.0:
            raise ValueError(
                f"disabled VAE-LHAT factor has positive mass at grid index {index}"
            )


def _difficulty_per_sample(
    logits: torch.Tensor, targets: torch.Tensor, objective: str
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("difficulty logits and targets must have identical shapes")
    elementwise = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    if objective == "multilabel_bce":
        return elementwise.mean(dim=1)
    positive = targets >= 0.5
    negative = ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    positive_loss = (elementwise * positive).sum(dim=1) / positive_count.clamp_min(1)
    negative_loss = (elementwise * negative).sum(dim=1) / negative_count.clamp_min(1)
    if objective == "positive_only_bce":
        return torch.where(positive_count > 0, positive_loss, negative_loss)
    if objective != "polarity_balanced_bce":
        raise ValueError(f"unsupported difficulty objective: {objective!r}")
    both = (positive_count > 0) & (negative_count > 0)
    one_sided = torch.where(positive_count > 0, positive_loss, negative_loss)
    return torch.where(both, 0.5 * (positive_loss + negative_loss), one_sided)


def _composition_index(
    indices: torch.Tensor | None,
    *,
    hint: int | None = None,
) -> int:
    if indices is None or not isinstance(indices, torch.Tensor):
        raise ValueError(f"{METHOD_ID} requires a fixed20 composition tensor")
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("composition_indices must be a non-empty rank-1 tensor")
    if hint is not None:
        if isinstance(hint, bool) or not isinstance(hint, int):
            raise TypeError("composition index hint must be an integer")
        if hint < -1 or hint > 19:
            raise ValueError("composition index must be -1 or lie in [0,19]")
        return hint
    first = int(indices[0].detach().cpu())
    if not bool((indices == first).all().item()):
        raise ValueError("one fixed20 exposure must use one shared composition")
    if first < -1 or first > 19:
        raise ValueError("composition index must be -1 or lie in [0,19]")
    return first


def _group_identity(
    rng_identity: Sequence[str], hash_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stable = tuple(
        str(value)
        for value in rng_identity
        if not str(value).startswith("view_execution_step=")
        and not str(value).startswith("exposure=")
    )
    return stable, tuple(str(value) for value in hash_ids)


class _MultistartNodeContext:
    """Give each LHAT start an isolated replayable RNG and diagnostics scope."""

    def __init__(self, parent: NodeContext, start_index: int) -> None:
        self._parent = parent
        self._prefix = f"multistart_{int(start_index)}"
        self.node_id = parent.node_id
        self.node_type = parent.node_type
        self.params = parent.params
        self.rng_namespace = f"{parent.rng_namespace}/{self._prefix}"

    def source(self, name: str) -> ViewValue:
        return self._parent.source(name)

    def call_adapter(
        self, name: str, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        return self._parent.call_adapter(name, inputs)

    def resource(self, name: str) -> Any:
        return self._parent.resource(name)

    def record_diagnostic(self, name: str, value: Any) -> None:
        self._parent.record_diagnostic(f"{self._prefix}/{name}", value)

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        return self._parent.torch_generator(
            f"{self._prefix}/{stream}", device=device
        )


class _DirectFixed20CorruptionContext:
    """Replay the exact canonical-corruption RNG identity of Direct+fixed20.

    The executor intentionally includes method, namespace and node identities in
    every seed.  A method ablation that keeps the same fixed20 anchor therefore
    needs this narrow context adapter; merely forcing the same composition index
    is insufficient to reproduce the same operator amplitudes and segments.
    """

    def __init__(self, parent: NodeContext) -> None:
        resources = getattr(parent, "_resources", None)
        if resources is None:
            raise TypeError(
                "Direct+fixed20 RNG matching requires the managed executor context"
            )
        base_seed = getattr(resources, "base_seed", None)
        rng_identity = getattr(resources, "rng_identity", None)
        if not isinstance(base_seed, int) or not isinstance(rng_identity, tuple):
            raise TypeError("managed executor RNG resources are malformed")
        self._parent = parent
        self._base_seed = base_seed
        self._rng_identity = tuple(str(value) for value in rng_identity)
        self._generators: dict[tuple[str, str], torch.Generator] = {}
        self.node_id = parent.node_id
        self.node_type = parent.node_type
        self.params = parent.params
        self.rng_namespace = DIRECT_FIXED20_RNG_NAMESPACE

    def source(self, name: str) -> ViewValue:
        return self._parent.source(name)

    def call_adapter(
        self, name: str, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        return self._parent.call_adapter(name, inputs)

    def resource(self, name: str) -> Any:
        return self._parent.resource(name)

    def record_diagnostic(self, name: str, value: Any) -> None:
        self._parent.record_diagnostic(name, value)

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        if stream != DIRECT_FIXED20_CORRUPTION_STREAM:
            raise ValueError(
                "Direct+fixed20 corruption context only owns the canonical "
                f"stream, got {stream!r}"
            )
        resolved_device = torch.device(device)
        key = (stream, str(resolved_device))
        generator = self._generators.get(key)
        if generator is None:
            seed = _derive_executor_seed(
                self._base_seed,
                DIRECT_FIXED20_PROFILE_NAME,
                DIRECT_FIXED20_RNG_NAMESPACE,
                DIRECT_FIXED20_CORRUPTION_NODE_ID,
                stream,
                self._rng_identity,
            )
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(seed)
            self._generators[key] = generator
            self._parent.record_diagnostic(
                "direct_fixed20_matched_corruption_rng",
                {
                    "seed": seed,
                    "profile_name": DIRECT_FIXED20_PROFILE_NAME,
                    "namespace": DIRECT_FIXED20_RNG_NAMESPACE,
                    "node_id": DIRECT_FIXED20_CORRUPTION_NODE_ID,
                    "stream": stream,
                    "execution_identity": list(self._rng_identity),
                },
            )
        return generator


class _MatchedScientificFactorContext:
    """Derive common-random-number streams shared by the factorial arms."""

    def __init__(self, parent: NodeContext, factor: str) -> None:
        resources = getattr(parent, "_resources", None)
        if resources is None:
            raise TypeError(
                "matched scientific-factor RNG requires the managed executor context"
            )
        base_seed = getattr(resources, "base_seed", None)
        rng_identity = getattr(resources, "rng_identity", None)
        if not isinstance(base_seed, int) or not isinstance(rng_identity, tuple):
            raise TypeError("managed executor RNG resources are malformed")
        if factor == "raw":
            namespace = TWO_FACTOR_MATCHED_RAW_NAMESPACE
            seed_node_id = TWO_FACTOR_MATCHED_RAW_NODE_ID
        elif factor == "lhat":
            namespace = TWO_FACTOR_MATCHED_LHAT_NAMESPACE
            seed_node_id = TWO_FACTOR_MATCHED_LHAT_NODE_ID
        else:
            raise ValueError("matched scientific factor must be raw or lhat")
        self._parent = parent
        self._factor = factor
        self._base_seed = base_seed
        self._rng_identity = tuple(str(value) for value in rng_identity)
        self._seed_node_id = seed_node_id
        self._generators: dict[tuple[str, str], torch.Generator] = {}
        self.node_id = parent.node_id
        self.node_type = parent.node_type
        self.params = parent.params
        self.rng_namespace = namespace

    def source(self, name: str) -> ViewValue:
        return self._parent.source(name)

    def call_adapter(
        self, name: str, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        return self._parent.call_adapter(name, inputs)

    def resource(self, name: str) -> Any:
        return self._parent.resource(name)

    def record_diagnostic(self, name: str, value: Any) -> None:
        self._parent.record_diagnostic(name, value)

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        if not isinstance(stream, str) or not stream:
            raise ValueError("matched scientific-factor RNG stream must be non-empty")
        resolved_device = torch.device(device)
        key = (stream, str(resolved_device))
        generator = self._generators.get(key)
        if generator is None:
            seed = _derive_executor_seed(
                self._base_seed,
                TWO_FACTOR_MATCHED_PROFILE_NAME,
                self.rng_namespace,
                self._seed_node_id,
                stream,
                self._rng_identity,
            )
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(seed)
            self._generators[key] = generator
            self._parent.record_diagnostic(
                f"matched_{self._factor}_rng/{stream}",
                {
                    "seed": seed,
                    "profile_name": TWO_FACTOR_MATCHED_PROFILE_NAME,
                    "namespace": self.rng_namespace,
                    "node_id": self._seed_node_id,
                    "stream": stream,
                    "execution_identity": list(self._rng_identity),
                },
            )
        return generator


class CoverageAnchorRuntime(MethodViewRuntime):
    """Reuse one clean-exposure LHAT view across one fixed20 exposure group."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model_spec = get_model_spec(self.model_name)
        contracts = self.method.contracts
        (
            self._raw_diversity_enabled,
            self._vae_lhat_enabled,
        ) = _scientific_factor_toggles(contracts)
        self._match_direct_fixed20_corruption_rng = bool(
            contracts.get("match_direct_fixed20_corruption_rng", False)
        )
        self._match_scientific_factor_rng = bool(
            contracts.get("match_scientific_factor_rng", False)
        )
        scales = contracts.get("residual_base_scales", {})
        if not isinstance(scales, dict):
            raise TypeError("residual_base_scales must be a mapping")
        self._random_residual_scale = _finite_contract_float(
            scales.get("random_corruption", RANDOM_RESIDUAL_SCALE),
            name="residual_base_scales.random_corruption",
        )
        self._lhat_residual_scale = _finite_contract_float(
            scales.get("lhat", LHAT_RESIDUAL_SCALE),
            name="residual_base_scales.lhat",
        )
        self._maximum_bce_gain = _finite_contract_float(
            contracts.get("maximum_per_record_bce_gain", MAX_PER_RECORD_BCE_GAIN),
            name="maximum_per_record_bce_gain",
        )
        self._difficulty_objective = str(
            contracts.get("difficulty_objective", "multilabel_bce")
        )
        self._lhat_attack_objective = str(
            contracts.get("lhat_attack_objective", "label_bce")
        )
        if self._difficulty_objective not in {
            "multilabel_bce",
            "polarity_balanced_bce",
            "positive_only_bce",
        }:
            raise ValueError(
                "difficulty_objective must be multilabel_bce, "
                "polarity_balanced_bce or positive_only_bce"
            )
        if self._lhat_attack_objective not in {
            "label_bce",
            "clean_bernoulli_kl",
        }:
            raise ValueError(
                "lhat_attack_objective must be label_bce or clean_bernoulli_kl"
            )
        self._coefficient_selection = str(
            contracts.get("coefficient_selection", "hard_max")
        )
        if self._coefficient_selection not in {
            "hard_max",
            "loss_softmax_sample",
            "uniform_simplex_sample",
            "replayable_weighted_sample",
            "replayable_factorized_sample",
        }:
            raise ValueError(
                "coefficient_selection must be hard_max, loss_softmax_sample "
                "uniform_simplex_sample, replayable_weighted_sample or "
                "replayable_factorized_sample"
            )
        if (
            self._coefficient_selection == "uniform_simplex_sample"
            and not bool(contracts.get("latent_simplex_augmax", False))
        ):
            raise ValueError("uniform_simplex_sample is reserved for latent simplex")
        self._selection_temperature = _finite_contract_float(
            contracts.get("selection_temperature", 0.05),
            name="selection_temperature",
            minimum=1.0e-8,
        )
        self._condition_on_corruption = bool(
            contracts.get("condition_lhat_on_canonical_corruption", False)
        )
        self._conditioned_strength_min = _finite_contract_float(
            contracts.get("conditioned_augmix_strength_min", 0.0),
            name="conditioned_augmix_strength_min",
        )
        self._conditioned_strength_max = _finite_contract_float(
            contracts.get("conditioned_augmix_strength_max", 1.0),
            name="conditioned_augmix_strength_max",
        )
        self._conditioned_minimum_bce_gain = _finite_contract_float(
            contracts.get("conditioned_minimum_bce_gain", 0.0),
            name="conditioned_minimum_bce_gain",
        )
        self._latent_simplex_augmax = bool(
            contracts.get("latent_simplex_augmax", False)
        )
        self._simplex_candidate_count = int(
            contracts.get("latent_simplex_candidate_count", 8)
        )
        self._simplex_anchor_mass_floor = _finite_contract_float(
            contracts.get("latent_simplex_anchor_mass_floor", 0.5),
            name="latent_simplex_anchor_mass_floor",
        )
        self._lhat_refresh_interval = int(
            contracts.get("lhat_refresh_interval_compositions", 0)
        )
        self._lhat_multistart_count = int(
            contracts.get("lhat_multistart_count", 1)
        )
        self._lhat_margin_preserving = bool(
            contracts.get("lhat_margin_preserving", False)
        )
        self._lhat_augmix_probe_count = int(
            contracts.get("lhat_augmix_probe_count", 0)
        )
        self._latent_simplex_scope = str(
            contracts.get("latent_simplex_scope", "corrupted_exposures")
        )
        self._coupled_threechain_mode = str(
            contracts.get(
                "coupled_threechain_mode", COUPLED_THREECHAIN_DISABLED
            )
        )
        self._coupled_canonical_mass_floor = _finite_contract_float(
            contracts.get("coupled_canonical_mass_floor", 0.5),
            name="coupled_canonical_mass_floor",
        )
        self._coupled_strength_min = _finite_contract_float(
            contracts.get("coupled_strength_min", 0.8),
            name="coupled_strength_min",
        )
        self._coupled_strength_max = _finite_contract_float(
            contracts.get("coupled_strength_max", 1.0),
            name="coupled_strength_max",
        )
        self._coupled_radius_ratio = _finite_contract_float(
            contracts.get("coupled_radius_ratio", 1.0),
            name="coupled_radius_ratio",
            minimum=1.0e-8,
        )
        self._coupled_orthogonality_epsilon = _finite_contract_float(
            contracts.get("coupled_orthogonality_epsilon", 1.0e-6),
            name="coupled_orthogonality_epsilon",
            minimum=1.0e-12,
        )
        if not 0.0 <= self._conditioned_strength_min <= self._conditioned_strength_max <= 1.0:
            raise ValueError("conditioned AugMix strengths must satisfy 0 <= min <= max <= 1")
        if self._condition_on_corruption and self._latent_simplex_augmax:
            raise ValueError(
                "conditioned LHAT and latent-simplex AugMax are exclusive arms"
            )
        if self._coupled_threechain_mode not in COUPLED_THREECHAIN_MODES:
            raise ValueError(
                "coupled_threechain_mode must be disabled, raw_control or "
                "orthogonal_lhat"
            )
        if self._coupled_threechain_mode != COUPLED_THREECHAIN_DISABLED and (
            self._condition_on_corruption or self._latent_simplex_augmax
        ):
            raise ValueError(
                "coupled three-chain, conditioned LHAT and latent simplex are "
                "mutually exclusive mechanisms"
            )
        if self._coupled_threechain_mode == COUPLED_THREECHAIN_RAW_CONTROL and not (
            self._raw_diversity_enabled and not self._vae_lhat_enabled
        ):
            raise ValueError(
                "raw-control coupled three-chain requires raw on and VAE-LHAT off"
            )
        if (
            self._coupled_threechain_mode
            == COUPLED_THREECHAIN_ORTHOGONAL_LHAT
            and not (self._raw_diversity_enabled and self._vae_lhat_enabled)
        ):
            raise ValueError(
                "orthogonal-LHAT coupled three-chain requires both factors"
            )
        if not 0.0 <= self._coupled_canonical_mass_floor < 1.0:
            raise ValueError("coupled canonical mass floor must lie in [0,1)")
        if not 0.0 <= self._coupled_strength_min <= self._coupled_strength_max <= 1.0:
            raise ValueError(
                "coupled strengths must satisfy 0 <= min <= max <= 1"
            )
        if (
            self._coupled_threechain_mode
            == COUPLED_THREECHAIN_ORTHOGONAL_LHAT
            and self.encoder is None
        ):
            raise ValueError(
                "orthogonal-LHAT coupled three-chain requires the managed VAE encoder"
            )
        if self._condition_on_corruption and not (
            self._raw_diversity_enabled and self._vae_lhat_enabled
        ):
            raise ValueError(
                "corruption-conditioned LHAT requires both scientific factors"
            )
        if self._latent_simplex_augmax and not (
            self._raw_diversity_enabled and self._vae_lhat_enabled
        ):
            raise ValueError(
                "latent-simplex AugMax requires both scientific factors"
            )
        if self._simplex_candidate_count < 4:
            raise ValueError("latent_simplex_candidate_count must be at least four")
        if not 0.0 <= self._simplex_anchor_mass_floor < 1.0:
            raise ValueError("latent_simplex_anchor_mass_floor must lie in [0,1)")
        if self._lhat_refresh_interval < 0 or self._lhat_refresh_interval > 20:
            raise ValueError("lhat_refresh_interval_compositions must lie in [0,20]")
        if not 1 <= self._lhat_multistart_count <= 4:
            raise ValueError("lhat_multistart_count must lie in [1,4]")
        if self._lhat_multistart_count > 1 and self._difficulty_objective != "multilabel_bce":
            raise ValueError("multistart LHAT currently requires multilabel_bce")
        if self._lhat_multistart_count > 1 and self._lhat_attack_objective != "label_bce":
            raise ValueError("latent VAT currently uses one LHAT start")
        if self._lhat_margin_preserving and self._lhat_multistart_count == 1:
            raise ValueError("margin-preserving LHAT requires multiple starts")
        if not 0 <= self._lhat_augmix_probe_count <= 3:
            raise ValueError("lhat_augmix_probe_count must lie in [0,3]")
        if self._lhat_augmix_probe_count and self._lhat_multistart_count == 1:
            raise ValueError("AugMix-aware LHAT selection requires multiple starts")
        if self._latent_simplex_scope not in {
            "corrupted_exposures",
            "clean_exposure_only",
        }:
            raise ValueError(
                "latent_simplex_scope must be corrupted_exposures or "
                "clean_exposure_only"
            )
        if (
            self._latent_simplex_scope == "clean_exposure_only"
            and not self._latent_simplex_augmax
        ):
            raise ValueError("clean-exposure simplex scope requires latent simplex")
        if (
            self._latent_simplex_scope == "clean_exposure_only"
            and self._lhat_refresh_interval
        ):
            raise ValueError("clean-exposure simplex cannot refresh unused LHAT views")
        if self._condition_on_corruption and self.encoder is None:
            raise ValueError("corruption-conditioned LHAT requires the managed VAE encoder")
        if self._latent_simplex_augmax and self.encoder is None:
            raise ValueError("latent-simplex AugMax requires the managed VAE encoder")
        raw_grid = contracts.get("residual_coefficient_grid")
        self._residual_coefficient_grid: tuple[tuple[float, float], ...] | None
        if raw_grid is None:
            self._residual_coefficient_grid = None
        else:
            if not isinstance(raw_grid, list) or not raw_grid:
                raise TypeError("residual_coefficient_grid must be a non-empty list")
            parsed: list[tuple[float, float]] = []
            for index, pair in enumerate(raw_grid):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise TypeError(
                        f"residual_coefficient_grid[{index}] must contain raw/lhat"
                    )
                parsed.append(
                    (
                        _finite_contract_float(
                            pair[0], name=f"residual_coefficient_grid[{index}].raw"
                        ),
                        _finite_contract_float(
                            pair[1], name=f"residual_coefficient_grid[{index}].lhat"
                        ),
                    )
                )
            if parsed[0] != (0.0, 0.0):
                raise ValueError(
                    "residual_coefficient_grid must begin with canonical [0, 0] fallback"
                )
            if len(set(parsed)) != len(parsed):
                raise ValueError("residual_coefficient_grid contains duplicates")
            self._residual_coefficient_grid = tuple(parsed)
        raw_probabilities = contracts.get("residual_coefficient_probabilities")
        self._residual_coefficient_probabilities: tuple[float, ...] | None
        if raw_probabilities is None:
            self._residual_coefficient_probabilities = None
        else:
            if self._residual_coefficient_grid is None:
                raise ValueError(
                    "residual coefficient probabilities require a coefficient grid"
                )
            if not isinstance(raw_probabilities, list) or len(raw_probabilities) != len(
                self._residual_coefficient_grid
            ):
                raise ValueError(
                    "residual coefficient probabilities must match the grid length"
                )
            probabilities = tuple(
                _finite_contract_float(
                    value,
                    name=f"residual_coefficient_probabilities[{index}]",
                )
                for index, value in enumerate(raw_probabilities)
            )
            if any(value < 0.0 for value in probabilities) or sum(probabilities) <= 0.0:
                raise ValueError(
                    "residual coefficient probabilities must be non-negative with positive mass"
                )
            self._residual_coefficient_probabilities = probabilities
            _validate_factor_grid_mass(
                self._residual_coefficient_grid,
                probabilities,
                raw_diversity_enabled=self._raw_diversity_enabled,
                vae_lhat_enabled=self._vae_lhat_enabled,
            )
        if (
            self._coefficient_selection == "replayable_weighted_sample"
            and self._residual_coefficient_grid is None
        ):
            raise ValueError("replayable weighted sampling requires a coefficient grid")
        raw_factor_distributions = contracts.get("residual_factor_distributions")
        self._raw_factor_distribution: tuple[
            tuple[float, ...], tuple[float, ...]
        ] | None = None
        self._lhat_factor_distribution: tuple[
            tuple[float, ...], tuple[float, ...]
        ] | None = None
        if raw_factor_distributions is not None:
            if not isinstance(raw_factor_distributions, Mapping) or set(
                raw_factor_distributions
            ) != {"raw", "lhat"}:
                raise TypeError(
                    "residual_factor_distributions must contain exactly raw and lhat"
                )
            self._raw_factor_distribution = _parse_factor_distribution(
                raw_factor_distributions["raw"],
                name="residual_factor_distributions.raw",
            )
            self._lhat_factor_distribution = _parse_factor_distribution(
                raw_factor_distributions["lhat"],
                name="residual_factor_distributions.lhat",
            )
        if self._coefficient_selection == "replayable_factorized_sample":
            if self._residual_coefficient_grid is not None:
                raise ValueError(
                    "factorized sampling forbids a materialized coefficient grid"
                )
            if self._residual_coefficient_probabilities is not None:
                raise ValueError(
                    "factorized sampling forbids joint coefficient probabilities"
                )
            if (
                self._raw_factor_distribution is None
                or self._lhat_factor_distribution is None
            ):
                raise ValueError(
                    "factorized sampling requires raw and LHAT distributions"
                )
            if not self._match_scientific_factor_rng:
                raise ValueError(
                    "factorized A/B/C/D sampling requires matched scientific-factor RNG"
                )
            if any(
                value != 0.0 and probability > 0.0
                for value, probability in zip(*self._raw_factor_distribution)
            ) and not self._raw_diversity_enabled:
                raise ValueError(
                    "disabled raw-diversity factor has positive factorized mass"
                )
            if any(
                value != 0.0 and probability > 0.0
                for value, probability in zip(*self._lhat_factor_distribution)
            ) and not self._vae_lhat_enabled:
                raise ValueError("disabled VAE-LHAT factor has positive factorized mass")
        elif raw_factor_distributions is not None:
            raise ValueError(
                "residual_factor_distributions are reserved for factorized sampling"
            )
        self._composition: int | None = None
        self._active_group: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._cached_lhat: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._cached_lhat_accounting: dict[str, Any] | None = None
        self._coupled_threechain_cache: _CoupledThreechainCache | None = None
        self._pending_factor_audit: list[_PendingFactorAudit] = []

    def _canonical_corruption(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
        *,
        composition_indices: torch.Tensor | None = None,
        composition_index_hint: int | None = None,
    ) -> ViewValue:
        matched_context: NodeContext = context
        if self._match_direct_fixed20_corruption_rng:
            matched_context = _DirectFixed20CorruptionContext(context)
        return MethodViewRuntime._canonical_corruption(
            self,
            matched_context,
            inputs,
            composition_indices=composition_indices,
            composition_index_hint=composition_index_hint,
        )

    def _select_calibrated_dose(
        self,
        *,
        anchor: WaveformView,
        residual: torch.Tensor,
        targets: torch.Tensor,
        classifier: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose the hardest finite dose below a per-record BCE-gain cap."""

        doses = torch.as_tensor(
            CALIBRATED_DOSES,
            device=anchor.waveform.device,
            dtype=anchor.waveform.dtype,
        )
        candidates = anchor.waveform.unsqueeze(0) + doses[:, None, None, None] * residual.unsqueeze(0)
        quality_masks = []
        for candidate in candidates:
            accepted, _ = _quality_mask(
                candidate,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
                return_reasons=False,
            )
            quality_masks.append(accepted & anchor.valid_mask)
        quality = torch.stack(quality_masks, dim=0)

        dose_count, batch_size = int(candidates.shape[0]), int(candidates.shape[1])
        flattened = candidates.reshape(
            dose_count * batch_size, *candidates.shape[2:]
        )
        repeated_targets = targets.unsqueeze(0).expand(dose_count, -1, -1).reshape(
            dose_count * batch_size, -1
        )
        with _frozen_classifier_for_attack(classifier), torch.no_grad(), torch.autocast(
            device_type=anchor.waveform.device.type,
            dtype=torch.bfloat16,
            enabled=anchor.waveform.device.type == "cuda",
        ):
            model_input = prepare_canonical_model_input(
                flattened,
                self._model_spec,
                epsilon=(
                    1.0e-6
                    if self.lhat_config is None
                    else self.lhat_config.normalization_epsilon
                ),
            )
            logits = validate_model_output(
                classifier(model_input),
                self._model_spec,
                batch_size=dose_count * batch_size,
                check_finite=False,
            ).float()
        finite = torch.isfinite(logits).all(dim=1).reshape(dose_count, batch_size)
        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        losses = _difficulty_per_sample(
            safe_logits, repeated_targets, self._difficulty_objective
        ).reshape(dose_count, batch_size)
        gains = losses - losses[0:1]
        admissible = (
            quality
            & finite
            & torch.isfinite(gains)
            & (gains >= 0.0)
            & (gains <= self._maximum_bce_gain)
        )
        admissible[0] = quality[0] & finite[0]
        scores = torch.where(
            admissible,
            gains,
            torch.full_like(gains, -torch.inf),
        )
        selected_index = scores.argmax(dim=0)
        no_valid = ~admissible.any(dim=0)
        selected_index = torch.where(no_valid, torch.zeros_like(selected_index), selected_index)
        batch_index = torch.arange(batch_size, device=anchor.waveform.device)
        selected = candidates.permute(1, 0, 2, 3)[batch_index, selected_index]
        selected_gain = gains.permute(1, 0)[batch_index, selected_index]
        selected_dose = doses.index_select(0, selected_index)
        return selected, selected_dose, selected_gain, admissible

    def _select_calibrated_coefficients(
        self,
        *,
        anchor: WaveformView,
        random_delta: torch.Tensor,
        lhat_delta: torch.Tensor,
        targets: torch.Tensor,
        classifier: nn.Module,
        selection_generator: torch.Generator | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Select the hardest bounded raw/LHAT AugMax coefficient pair."""

        if self._residual_coefficient_grid is None:
            raise RuntimeError("independent coefficient selection lacks a grid")
        coefficients = torch.as_tensor(
            self._residual_coefficient_grid,
            device=anchor.waveform.device,
            dtype=anchor.waveform.dtype,
        )
        candidates = (
            anchor.waveform.unsqueeze(0)
            + coefficients[:, 0, None, None, None]
            * self._random_residual_scale
            * random_delta.unsqueeze(0)
            + coefficients[:, 1, None, None, None]
            * self._lhat_residual_scale
            * lhat_delta.unsqueeze(0)
        )
        quality_masks = []
        for candidate in candidates:
            accepted, _ = _quality_mask(
                candidate,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
                return_reasons=False,
            )
            quality_masks.append(accepted & anchor.valid_mask)
        quality = torch.stack(quality_masks, dim=0)

        candidate_count, batch_size = int(candidates.shape[0]), int(candidates.shape[1])
        if self._coefficient_selection == "replayable_weighted_sample":
            if selection_generator is None:
                raise RuntimeError(
                    "replayable_weighted_sample requires an isolated torch generator"
                )
            configured = self._residual_coefficient_probabilities
            if configured is None:
                configured = tuple(1.0 for _ in self._residual_coefficient_grid)
            admissible = quality
            selected_index, selection_entropy = _sample_replayable_grid_indices(
                admissible,
                torch.as_tensor(
                    configured,
                    device=anchor.waveform.device,
                    dtype=torch.float32,
                ),
                generator=selection_generator,
            )
            no_valid = ~admissible.any(dim=0)
            selected_index = torch.where(
                no_valid, torch.zeros_like(selected_index), selected_index
            )
            batch_index = torch.arange(batch_size, device=anchor.waveform.device)
            selected = candidates.permute(1, 0, 2, 3)[
                batch_index, selected_index
            ]
            selected_coefficients = coefficients.index_select(0, selected_index)
            # No classifier forward is performed merely for coefficient
            # diagnostics. The actual supervised corrupted BCE remains owned
            # by the matched online trainer exposure.
            selected_gain = torch.zeros(
                batch_size,
                device=anchor.waveform.device,
                dtype=torch.float32,
            )
            return (
                selected,
                selected_coefficients,
                selected_gain,
                admissible,
                selected_index,
                selection_entropy,
            )

        flattened = candidates.reshape(candidate_count * batch_size, *candidates.shape[2:])
        repeated_targets = targets.unsqueeze(0).expand(candidate_count, -1, -1).reshape(
            candidate_count * batch_size, -1
        )
        with _frozen_classifier_for_attack(classifier), torch.no_grad(), torch.autocast(
            device_type=anchor.waveform.device.type,
            dtype=torch.bfloat16,
            enabled=anchor.waveform.device.type == "cuda",
        ):
            model_input = prepare_canonical_model_input(
                flattened,
                self._model_spec,
                epsilon=(
                    1.0e-6
                    if self.lhat_config is None
                    else self.lhat_config.normalization_epsilon
                ),
            )
            logits = validate_model_output(
                classifier(model_input),
                self._model_spec,
                batch_size=candidate_count * batch_size,
                check_finite=False,
            ).float()
        finite = torch.isfinite(logits).all(dim=1).reshape(candidate_count, batch_size)
        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        losses = _difficulty_per_sample(
            safe_logits, repeated_targets, self._difficulty_objective
        ).reshape(candidate_count, batch_size)
        gains = losses - losses[0:1]
        admissible = (
            quality
            & finite
            & torch.isfinite(gains)
            & (gains >= 0.0)
            & (gains <= self._maximum_bce_gain)
        )
        admissible[0] = quality[0] & finite[0]
        scores = torch.where(admissible, gains, torch.full_like(gains, -torch.inf))
        no_valid = ~admissible.any(dim=0)
        if self._coefficient_selection == "hard_max":
            selected_index = scores.argmax(dim=0)
            selection_entropy = torch.zeros(
                batch_size, device=anchor.waveform.device, dtype=torch.float32
            )
        elif self._coefficient_selection == "loss_softmax_sample":
            if selection_generator is None:
                raise RuntimeError(
                    "loss_softmax_sample requires an isolated torch generator"
                )
            sampling_scores = scores.clone()
            sampling_scores[0, no_valid] = 0.0
            probability_logits = (
                sampling_scores / self._selection_temperature
            ).transpose(0, 1)
            probabilities = torch.softmax(probability_logits, dim=1)
            selected_index = torch.multinomial(
                probabilities,
                1,
                replacement=True,
                generator=selection_generator,
            ).squeeze(1)
            selection_entropy = -(
                probabilities * probabilities.clamp_min(1.0e-12).log()
            ).sum(dim=1)
        else:
            raise RuntimeError(
                "this coefficient mode cannot select waveform residual coefficients"
            )
        selected_index = torch.where(no_valid, torch.zeros_like(selected_index), selected_index)
        batch_index = torch.arange(batch_size, device=anchor.waveform.device)
        selected = candidates.permute(1, 0, 2, 3)[batch_index, selected_index]
        selected_gain = gains.permute(1, 0)[batch_index, selected_index]
        selected_coefficients = coefficients.index_select(0, selected_index)
        return (
            selected,
            selected_coefficients,
            selected_gain,
            admissible,
            selected_index,
            selection_entropy,
        )

    def _select_factorized_coefficients(
        self,
        *,
        anchor: WaveformView,
        random_delta: torch.Tensor,
        random_valid: torch.Tensor,
        lhat_delta: torch.Tensor,
        lhat_valid: torch.Tensor,
        raw_generator: torch.Generator,
        lhat_generator: torch.Generator,
    ) -> _FactorizedResidualSelection:
        """Draw independent factors once and materialize only one candidate."""

        if (
            self._raw_factor_distribution is None
            or self._lhat_factor_distribution is None
        ):
            raise RuntimeError("factorized coefficient distributions are missing")
        batch_size = anchor.batch_size
        raw_values, raw_probabilities = self._raw_factor_distribution
        lhat_values, lhat_probabilities = self._lhat_factor_distribution
        assigned_raw, assigned_raw_index, raw_entropy = _sample_factor_coefficients(
            values=raw_values,
            probabilities=raw_probabilities,
            batch_size=batch_size,
            device=anchor.waveform.device,
            dtype=anchor.waveform.dtype,
            generator=raw_generator,
        )
        assigned_lhat, assigned_lhat_index, lhat_entropy = _sample_factor_coefficients(
            values=lhat_values,
            probabilities=lhat_probabilities,
            batch_size=batch_size,
            device=anchor.waveform.device,
            dtype=anchor.waveform.dtype,
            generator=lhat_generator,
        )
        assigned = torch.stack((assigned_raw, assigned_lhat), dim=1)
        raw_unavailable = (assigned_raw > 0.0) & ~random_valid
        lhat_unavailable = (assigned_lhat > 0.0) & ~lhat_valid
        available_raw = torch.where(
            random_valid, assigned_raw, torch.zeros_like(assigned_raw)
        )
        available_lhat = torch.where(
            lhat_valid, assigned_lhat, torch.zeros_like(assigned_lhat)
        )
        proposed = (
            anchor.waveform
            + available_raw[:, None, None]
            * self._random_residual_scale
            * random_delta
            + available_lhat[:, None, None]
            * self._lhat_residual_scale
            * lhat_delta
        )
        quality_accepted, _ = _quality_mask(
            proposed,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
            return_reasons=False,
        )
        quality_accepted &= anchor.valid_mask
        available_nonzero = (available_raw > 0.0) | (available_lhat > 0.0)
        quality_fallback = available_nonzero & ~quality_accepted
        effective = torch.stack((available_raw, available_lhat), dim=1)
        effective = torch.where(
            quality_accepted[:, None], effective, torch.zeros_like(effective)
        )
        waveform = torch.where(
            quality_accepted[:, None, None], proposed, anchor.waveform
        ).contiguous()
        return _FactorizedResidualSelection(
            waveform=waveform,
            assigned_indices=torch.stack(
                (assigned_raw_index, assigned_lhat_index), dim=1
            ),
            assigned_coefficients=assigned,
            effective_coefficients=effective,
            quality_accepted=quality_accepted,
            quality_fallback=quality_fallback,
            raw_unavailable=raw_unavailable,
            lhat_unavailable=lhat_unavailable,
            selection_entropy=torch.stack((raw_entropy, lhat_entropy), dim=1),
        )

    def _record_factor_audit(
        self,
        *,
        sample_ids: Sequence[str],
        selection: _FactorizedResidualSelection,
    ) -> None:
        composition = self._composition
        if composition is None or not 0 <= composition <= 19:
            raise RuntimeError("factor audit requires a corrupted fixed20 exposure")
        ids = tuple(str(value) for value in sample_ids)
        if selection.assigned_indices.shape != (len(ids), 2):
            raise RuntimeError("factor audit assignment shape changed")
        if selection.effective_coefficients.shape != (len(ids), 2):
            raise RuntimeError("factor audit effective coefficient shape changed")
        masks = (
            selection.quality_accepted,
            selection.quality_fallback,
            selection.raw_unavailable,
            selection.lhat_unavailable,
        )
        if any(mask.shape != (len(ids),) for mask in masks):
            raise RuntimeError("factor audit mask shape changed")
        if any(item.composition_index == composition for item in self._pending_factor_audit):
            raise RuntimeError(f"factor audit composition {composition} was recorded twice")
        self._pending_factor_audit.append(
            _PendingFactorAudit(
                composition_index=composition,
                sample_ids=ids,
                assigned_indices=selection.assigned_indices.detach(),
                effective_coefficients=selection.effective_coefficients.detach(),
                quality_accepted=selection.quality_accepted.detach(),
                quality_fallback=selection.quality_fallback.detach(),
                raw_unavailable=selection.raw_unavailable.detach(),
                lhat_unavailable=selection.lhat_unavailable.detach(),
            )
        )

    def _finalize_factor_audit_group(self) -> None:
        if self._coefficient_selection != "replayable_factorized_sample":
            return
        ordered = tuple(
            sorted(self._pending_factor_audit, key=lambda item: item.composition_index)
        )
        if tuple(item.composition_index for item in ordered) != tuple(range(20)):
            raise RuntimeError("factor audit did not observe every fixed20 composition")
        if self._active_group is None:
            raise RuntimeError("factor audit has no active group identity")
        stable_identity, group_hash_ids = self._active_group
        if any(item.sample_ids != group_hash_ids for item in ordered):
            raise RuntimeError("factor audit sample order changed inside the group")

        packed = torch.stack(
            tuple(
                torch.cat(
                    (
                        item.assigned_indices.to(dtype=torch.float32),
                        item.effective_coefficients.to(dtype=torch.float32),
                        item.quality_accepted[:, None].to(dtype=torch.float32),
                        item.quality_fallback[:, None].to(dtype=torch.float32),
                        item.raw_unavailable[:, None].to(dtype=torch.float32),
                        item.lhat_unavailable[:, None].to(dtype=torch.float32),
                    ),
                    dim=1,
                )
                for item in ordered
            ),
            dim=0,
        ).detach().cpu()
        host = packed.tolist()
        compositions: list[dict[str, Any]] = []
        raw_bindings: list[list[Any]] = []
        lhat_bindings: list[list[Any]] = []
        effective_bindings: list[list[Any]] = []
        quality_bindings: list[list[Any]] = []
        for row_index, item in enumerate(ordered):
            matrix = host[row_index]
            assigned_raw = [int(row[0]) for row in matrix]
            assigned_lhat = [int(row[1]) for row in matrix]
            effective_raw = [float(row[2]) for row in matrix]
            effective_lhat = [float(row[3]) for row in matrix]
            quality_accepted = [bool(row[4]) for row in matrix]
            quality_fallback = [bool(row[5]) for row in matrix]
            raw_unavailable = [bool(row[6]) for row in matrix]
            lhat_unavailable = [bool(row[7]) for row in matrix]
            composition = item.composition_index
            fallback_hashes = [
                hash_id
                for hash_id, failed in zip(group_hash_ids, quality_fallback, strict=True)
                if failed
            ]
            raw_unavailable_hashes = [
                hash_id
                for hash_id, unavailable in zip(
                    group_hash_ids, raw_unavailable, strict=True
                )
                if unavailable
            ]
            lhat_unavailable_hashes = [
                hash_id
                for hash_id, unavailable in zip(
                    group_hash_ids, lhat_unavailable, strict=True
                )
                if unavailable
            ]
            compositions.append(
                {
                    "composition_index": composition,
                    "assigned_raw_indices": assigned_raw,
                    "assigned_lhat_indices": assigned_lhat,
                    "effective_raw_coefficients": effective_raw,
                    "effective_lhat_coefficients": effective_lhat,
                    "quality_accepted_count": sum(quality_accepted),
                    "quality_fallback_hash_ids": fallback_hashes,
                    "raw_unavailable_hash_ids": raw_unavailable_hashes,
                    "lhat_unavailable_hash_ids": lhat_unavailable_hashes,
                }
            )
            for position, hash_id in enumerate(group_hash_ids):
                raw_bindings.append(
                    [composition, hash_id, assigned_raw[position]]
                )
                lhat_bindings.append(
                    [composition, hash_id, assigned_lhat[position]]
                )
                effective_bindings.append(
                    [
                        composition,
                        hash_id,
                        effective_raw[position],
                        effective_lhat[position],
                    ]
                )
                quality_bindings.append(
                    [
                        composition,
                        hash_id,
                        quality_accepted[position],
                        quality_fallback[position],
                        raw_unavailable[position],
                        lhat_unavailable[position],
                    ]
                )

        lhat_accounting = dict(self._cached_lhat_accounting or {})
        candidate_positions = tuple(
            int(value)
            for value in lhat_accounting.pop("candidate_eligible_positions", ())
        )
        accepted_positions = tuple(
            int(value) for value in lhat_accounting.pop("accepted_positions", ())
        )
        lhat_accounting.update(
            {
                "factor_enabled": self._vae_lhat_enabled,
                "availability_applicable": self._vae_lhat_enabled,
                "candidate_eligible_hash_ids": [
                    group_hash_ids[position] for position in candidate_positions
                ],
                "accepted_hash_ids": [
                    group_hash_ids[position] for position in accepted_positions
                ],
                "available_fraction": (
                    len(accepted_positions) / len(group_hash_ids)
                    if self._vae_lhat_enabled and group_hash_ids
                    else None
                ),
            }
        )
        record = {
            "method_id": self.method.profile_name,
            "scientific_arm": self.method.scientific_arm,
            "scientific_factors": {
                "raw_diversity": self._raw_diversity_enabled,
                "vae_lhat": self._vae_lhat_enabled,
            },
            "stable_rng_identity": list(stable_identity),
            "hash_ids": list(group_hash_ids),
            "ordered_hash_ids_sha256": _canonical_sha256(list(group_hash_ids)),
            "factor_distributions": {
                "raw": {
                    "values": list(self._raw_factor_distribution[0]),
                    "probabilities": list(self._raw_factor_distribution[1]),
                },
                "lhat": {
                    "values": list(self._lhat_factor_distribution[0]),
                    "probabilities": list(self._lhat_factor_distribution[1]),
                },
            },
            "raw_assignment_sha256": _canonical_sha256(raw_bindings),
            "lhat_assignment_sha256": _canonical_sha256(lhat_bindings),
            "effective_assignment_sha256": _canonical_sha256(effective_bindings),
            "quality_identity_sha256": _canonical_sha256(quality_bindings),
            "lhat_accounting": lhat_accounting,
            "compositions": compositions,
        }
        _FACTOR_AUDIT_RECORDS.append(record)

    def _clear_group(self) -> None:
        self._composition = None
        self._active_group = None
        self._cached_lhat = None
        self._cached_lhat_accounting = None
        self._coupled_threechain_cache = None
        self._pending_factor_audit.clear()

    def _multistart_lhat_attack(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        """Mine the hardest valid endpoint across independent exact-label pools."""

        source = inputs[0]
        assert isinstance(source, WaveformView)
        starts = tuple(
            MethodViewRuntime._lhat_attack(
                self,
                _MultistartNodeContext(context, start_index),
                inputs,
            )
            for start_index in range(self._lhat_multistart_count)
        )
        if not all(isinstance(value, WaveformView) for value in starts):
            raise TypeError("multistart LHAT produced a non-waveform view")
        anchor_reconstructions = torch.stack(
            tuple(
                _lhat_adversarial_residual(value, source)[1]
                for value in starts
            ),
            dim=0,
        )
        classifier = context.resource("classifier")
        with _frozen_classifier_for_attack(classifier), torch.no_grad():
            clean_logits = validate_model_output(
                classifier(
                    prepare_canonical_model_input(
                        source.waveform,
                        self._model_spec,
                        epsilon=self.lhat_config.normalization_epsilon,
                    )
                ),
                self._model_spec,
                batch_size=source.batch_size,
                check_finite=False,
            ).float()
            clean_loss = _difficulty_per_sample(
                clean_logits, source.labels, self._difficulty_objective
            )
            losses = []
            candidate_logits = []
            for value in starts:
                logits = validate_model_output(
                    classifier(
                        prepare_canonical_model_input(
                            value.waveform,
                            self._model_spec,
                            epsilon=self.lhat_config.normalization_epsilon,
                        )
                    ),
                    self._model_spec,
                    batch_size=source.batch_size,
                    check_finite=False,
                ).float()
                candidate_logits.append(logits)
                losses.append(
                    _difficulty_per_sample(
                        logits, source.labels, self._difficulty_objective
                    )
                )

        valid = torch.stack(tuple(value.valid_mask for value in starts), dim=0)
        if self._lhat_margin_preserving:
            truth = source.labels >= 0.5
            clean_correct = (torch.sigmoid(clean_logits) >= 0.5) == truth
            no_new_error = torch.stack(
                tuple(
                    ~(
                        ((torch.sigmoid(logits) >= 0.5) != truth)
                        & clean_correct
                    ).any(dim=1)
                    for logits in candidate_logits
                ),
                dim=0,
            )
            context.record_diagnostic(
                "multistart_margin_preserving_fraction",
                float(no_new_error.float().mean().cpu()),
            )
            valid &= no_new_error
        selection_losses = torch.stack(tuple(losses), dim=0)
        if self._lhat_augmix_probe_count:
            operator_parameters = {
                name: self.operator_profile.parameters_for(name)
                for name in self.operator_profile.canonical_order
            }
            robust_losses = torch.zeros_like(selection_losses)
            robust_valid = torch.ones_like(valid)
            for probe_index in range(self._lhat_augmix_probe_count):
                shared_generator = context.torch_generator(
                    f"multistart_augmix_probe_{probe_index}",
                    device=source.waveform.device,
                )
                shared_state = shared_generator.get_state()
                for start_index, value in enumerate(starts):
                    generator = torch.Generator(device=source.waveform.device)
                    generator.set_state(shared_state)
                    corrupted = generate_canonical_corruption(
                        value.waveform,
                        operator_params=operator_parameters,
                        generator=generator,
                        composition_indices=None,
                        _input_prevalidated=True,
                    )
                    robust_valid[start_index] &= (
                        corrupted.diagnostics.output_nonfinite_count == 0
                    )
                    with _frozen_classifier_for_attack(classifier), torch.no_grad():
                        logits = validate_model_output(
                            classifier(
                                prepare_canonical_model_input(
                                    corrupted.waveform_raw_100hz,
                                    self._model_spec,
                                    epsilon=self.lhat_config.normalization_epsilon,
                                )
                            ),
                            self._model_spec,
                            batch_size=source.batch_size,
                            check_finite=False,
                        ).float()
                    robust_losses[start_index] += _difficulty_per_sample(
                        logits, source.labels, self._difficulty_objective
                    )
            selection_losses = robust_losses / float(self._lhat_augmix_probe_count)
            valid &= robust_valid
            context.record_diagnostic(
                "multistart_augmix_probe_count", self._lhat_augmix_probe_count
            )
            context.record_diagnostic(
                "multistart_augmix_probe_gain",
                float(
                    (
                        selection_losses
                        - clean_loss.unsqueeze(0)
                    )[valid].mean().cpu()
                )
                if bool(valid.any())
                else 0.0,
            )
        score = selection_losses.masked_fill(~valid, -torch.inf)
        selected_start = score.argmax(dim=0)
        any_valid = valid.any(dim=0)
        selected_start = torch.where(
            any_valid, selected_start, torch.zeros_like(selected_start)
        )
        waveforms = torch.stack(tuple(value.waveform for value in starts), dim=0)
        batch_index = torch.arange(
            source.batch_size, device=source.waveform.device
        )
        selected_waveform = waveforms[selected_start, batch_index]
        selected_waveform = torch.where(
            any_valid[:, None, None], selected_waveform, source.waveform
        ).contiguous()
        selected_anchor_reconstruction = anchor_reconstructions[
            selected_start, batch_index
        ]
        selected_anchor_reconstruction = torch.where(
            any_valid[:, None, None],
            selected_anchor_reconstruction,
            source.waveform,
        ).contiguous()
        selected_loss = score[selected_start, batch_index]
        selected_gain = torch.where(
            any_valid, selected_loss - clean_loss, torch.zeros_like(clean_loss)
        )
        valid_count = any_valid.sum().clamp_min(1)
        one_hot = F.one_hot(
            selected_start, num_classes=self._lhat_multistart_count
        ).float()
        selected_histogram = (
            (one_hot * any_valid[:, None]).sum(dim=0) / valid_count
        )
        per_record_span = torch.where(
            valid,
            score,
            torch.full_like(score, torch.inf),
        ).amin(dim=0)
        per_record_span = torch.where(
            any_valid, selected_loss - per_record_span, torch.zeros_like(clean_loss)
        )
        context.record_diagnostic("multistart_count", self._lhat_multistart_count)
        context.record_diagnostic(
            "multistart_valid_fraction", float(any_valid.float().mean().cpu())
        )
        context.record_diagnostic(
            "multistart_selected_bce_gain",
            float(selected_gain[any_valid].mean().cpu()) if bool(any_valid.any()) else 0.0,
        )
        context.record_diagnostic(
            "multistart_hardness_span",
            float(per_record_span[any_valid].mean().cpu()) if bool(any_valid.any()) else 0.0,
        )
        for start_index, fraction in enumerate(selected_histogram.cpu().tolist()):
            context.record_diagnostic(
                f"multistart_selected_fraction_{start_index}", float(fraction)
            )
        return WaveformView(
            name=context.node_id,
            waveform=selected_waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=any_valid,
            provenance=Provenance(
                node_id=context.node_id,
                operation="vae_lhat_multistart_hard_mining",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "start_count": self._lhat_multistart_count,
                    "candidate_policy": self.lhat_config.candidate_mode,
                    "selection": "per_record_max_multilabel_bce",
                    "candidate_label_policy": "exact_positive_set",
                    "margin_preserving": self._lhat_margin_preserving,
                    "margin_rule": (
                        "no_new_threshold_error_on_anchor_correct_classes"
                        if self._lhat_margin_preserving
                        else "disabled"
                    ),
                    "difficulty_selection_domain": (
                        f"mean_of_{self._lhat_augmix_probe_count}_matched_depth23_probes"
                        if self._lhat_augmix_probe_count
                        else "decoded_vae_endpoint"
                    ),
                },
            ),
            metadata={
                "selected_start": selected_start.detach(),
                "selected_bce_gain": selected_gain.detach(),
                "anchor_waveform_raw": selected_anchor_reconstruction.detach(),
            },
        )

    def _lhat_attack(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("coverage-anchor LHAT requires one clean WaveformView")
        source = inputs[0]
        if not self._vae_lhat_enabled:
            if self._composition == -1:
                self._cached_lhat = (
                    source.waveform.detach().clone(),
                    source.waveform.detach().clone(),
                    source.valid_mask.detach().clone(),
                )
                self._cached_lhat_accounting = {
                    "candidate_eligible_positions": (),
                    "accepted_positions": (),
                    "ineligible_hash_ids": (),
                    "quality_rejected": (),
                    "factor_disabled": True,
                }
                operation = "vae_lhat_factor_disabled_cache_build"
            else:
                if self._cached_lhat is None:
                    raise RuntimeError(
                        "disabled VAE-LHAT exposure lacks the clean identity cache"
                    )
                operation = "vae_lhat_factor_disabled_cache_reuse"
            context.record_diagnostic("vae_lhat_factor_enabled", 0.0)
            return WaveformView(
                name=context.node_id,
                waveform=source.waveform,
                labels=source.labels,
                sample_ids=source.sample_ids,
                valid_mask=source.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation=operation,
                    parent_names=(source.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={"classifier_attack_executed": False},
                ),
                metadata={
                    "factor_disabled": True,
                    "anchor_waveform_raw": source.waveform.detach(),
                },
            )
        if self._condition_on_corruption:
            context.record_diagnostic("conditioned_placeholder_identity", 1.0)
            return WaveformView(
                name=context.node_id,
                waveform=source.waveform,
                labels=source.labels,
                sample_ids=source.sample_ids,
                valid_mask=source.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="conditioned_lhat_requirement_identity_bridge",
                    parent_names=(source.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={"attack_owner": MIX_NODE_ID},
                ),
            )
        refresh = (
            self._lhat_refresh_interval > 0
            and self._composition is not None
            and self._composition > 0
            and self._composition % self._lhat_refresh_interval == 0
        )
        if self._composition == -1 or refresh:
            attack_context: NodeContext = context
            if self._match_scientific_factor_rng:
                attack_context = _MatchedScientificFactorContext(context, "lhat")
            if self._lhat_multistart_count > 1:
                generated = self._multistart_lhat_attack(attack_context, inputs)
            elif (
                self._difficulty_objective == "multilabel_bce"
                and self._lhat_attack_objective == "label_bce"
            ):
                generated = super()._lhat_attack(attack_context, inputs)
            else:
                # ``core.lhat`` is whitelist-owned and intentionally remains
                # unchanged.  This scoped sandbox patch changes only the
                # attack/search score for this synchronous LHAT call; the
                # outer classifier training loss remains the locked BCE.
                import core.lhat as lhat_module

                original_bce = lhat_module._bce_per_sample

                reference_probabilities: torch.Tensor | None = None

                def sandbox_difficulty(
                    logits: torch.Tensor, targets: torch.Tensor
                ) -> torch.Tensor:
                    nonlocal reference_probabilities
                    if self._lhat_attack_objective == "clean_bernoulli_kl":
                        if reference_probabilities is None:
                            reference_probabilities = torch.sigmoid(
                                logits.detach().float()
                            ).clamp(1.0e-6, 1.0 - 1.0e-6)
                        probabilities = reference_probabilities.to(
                            device=logits.device, dtype=logits.dtype
                        )
                        cross_entropy = F.binary_cross_entropy_with_logits(
                            logits,
                            probabilities,
                            reduction="none",
                        )
                        entropy = F.binary_cross_entropy(
                            probabilities,
                            probabilities,
                            reduction="none",
                        )
                        return (cross_entropy - entropy).mean(dim=1)
                    return _difficulty_per_sample(
                        logits, targets, self._difficulty_objective
                    )

                lhat_module._bce_per_sample = sandbox_difficulty
                try:
                    generated = super()._lhat_attack(attack_context, inputs)
                finally:
                    lhat_module._bce_per_sample = original_bce
                if self._lhat_attack_objective == "clean_bernoulli_kl":
                    if reference_probabilities is None:
                        raise RuntimeError("latent VAT failed to bind its clean reference")
                    context.record_diagnostic("latent_vat_enabled", 1.0)
                    diagnostic_means = generated.metadata.get("diagnostic_means", {})
                    if isinstance(diagnostic_means, Mapping):
                        context.record_diagnostic(
                            "latent_vat_final_kl",
                            float(diagnostic_means.get("final_bce", 0.0)),
                        )
            assert isinstance(generated, WaveformView)
            _, anchor_reconstruction = _lhat_adversarial_residual(
                generated, source
            )
            self._cached_lhat = (
                generated.waveform.detach().clone(),
                anchor_reconstruction.detach().clone(),
                generated.valid_mask.detach().clone(),
            )
            self._cached_lhat_accounting = {
                "candidate_eligible_positions": tuple(
                    int(value)
                    for value in generated.metadata.get(
                        "candidate_eligible_positions", ()
                    )
                ),
                "accepted_positions": tuple(
                    int(value)
                    for value in generated.metadata.get("accepted_positions", ())
                ),
                "ineligible_hash_ids": tuple(
                    str(value)
                    for value in generated.metadata.get("ineligible_hash_ids", ())
                ),
                "quality_rejected": tuple(
                    dict(value)
                    for value in generated.metadata.get("quality_rejected", ())
                ),
            }
            context.record_diagnostic(
                "cache_refresh" if refresh else "cache_build", 1.0
            )
            return generated
        if self._cached_lhat is None:
            raise RuntimeError("fixed20 corrupted exposure lacks clean LHAT cache")
        waveform, anchor_reconstruction, valid_mask = self._cached_lhat
        if (
            waveform.shape != source.waveform.shape
            or anchor_reconstruction.shape != source.waveform.shape
            or valid_mask.shape != source.valid_mask.shape
        ):
            raise RuntimeError("cached LHAT batch shape changed inside exposure group")
        context.record_diagnostic("cache_reuse", 1.0)
        return WaveformView(
            name=context.node_id,
            waveform=waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="clean_exposure_lhat_cache_reuse",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "shared_across_fixed20": True,
                    "detached_before_reuse": True,
                },
            ),
            metadata={
                "cache_reused": True,
                "anchor_waveform_raw": anchor_reconstruction,
            },
        )

    def _resource_bridge(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("resource bridge requires one clean WaveformView")
        source = inputs[0]
        if not (
            self._condition_on_corruption
            or self._latent_simplex_augmax
            or self._coupled_threechain_mode
            == COUPLED_THREECHAIN_ORTHOGONAL_LHAT
        ):
            raise RuntimeError(
                "resource bridge is reserved for encoder-dependent coverage arms"
            )
        if context.resource("vae_encoder") is not self.encoder:
            raise RuntimeError("conditioned LHAT encoder resource identity drifted")
        context.record_diagnostic("identity_only", 1.0)
        return WaveformView(
            name=context.node_id,
            waveform=source.waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="coverage_encoder_requirement_bridge",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={"waveform_mutated": False},
            ),
        )

    def _encode_coupled_waveforms(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("coupled latent geometry has no managed VAE encoder")
        with torch.no_grad():
            encoded = self.encoder(
                prepare_ecgtwin_encoder_input(waveform),
                sample=False,
            )
        if not isinstance(encoded, tuple) or len(encoded) != 3:
            raise TypeError(
                "coupled latent geometry requires encoder output "
                "(scaled_latent, mean, log_variance)"
            )
        latent = encoded[0].float()
        if tuple(latent.shape[1:]) != (4, 128):
            raise ValueError("coupled latent geometry received an invalid VAE latent")
        return latent.contiguous()

    def _build_coupled_threechain_cache(
        self,
        *,
        context: NodeContext,
        clean: WaveformView,
        lhat: WaveformView,
    ) -> _CoupledThreechainCache:
        """Build two raw endpoints and, when enabled, one orthogonal LHAT endpoint."""

        if self._coupled_threechain_mode == COUPLED_THREECHAIN_DISABLED:
            raise RuntimeError("coupled cache requested while the mechanism is disabled")
        if self._coupled_threechain_cache is not None:
            raise RuntimeError("coupled three-chain cache was built twice for one group")
        if self.operator_profile is None:
            raise RuntimeError("coupled three-chain runtime has no operator profile")
        parameters = {
            name: self.operator_profile.parameters_for(name)
            for name in self.operator_profile.canonical_order
        }
        raw_context = _MatchedScientificFactorContext(context, "raw")
        raw_generator = raw_context.torch_generator(
            "coupled_endpoint_generation",
            device=clean.waveform.device,
        )
        batch = clean.batch_size
        raw_result = generate_canonical_corruption(
            torch.cat((clean.waveform, clean.waveform), dim=0),
            operator_params=parameters,
            generator=raw_generator,
            composition_indices=None,
            _input_prevalidated=True,
        )
        raw_chain1, raw_chain2 = raw_result.waveform_raw_100hz.split(batch, dim=0)
        raw_nonfinite1, raw_nonfinite2 = (
            raw_result.diagnostics.output_nonfinite_count.split(batch, dim=0)
        )
        raw_valid1 = clean.valid_mask & (raw_nonfinite1 == 0)
        raw_valid2 = clean.valid_mask & (raw_nonfinite2 == 0)
        raw_depths = raw_result.diagnostics.depth.reshape(2, batch).transpose(0, 1)
        raw_operator_mask = (
            raw_result.diagnostics.operator_mask.reshape(batch * 2, -1)
            .reshape(2, batch, -1)
            .transpose(0, 1)
            .contiguous()
        )
        diagnostics: dict[str, float | torch.Tensor] = {
            "coupled_raw_chain1_available_fraction": raw_valid1.float().mean().detach(),
            "coupled_raw_chain2_available_fraction": raw_valid2.float().mean().detach(),
            "coupled_raw_mean_depth": raw_depths.float().mean().detach(),
        }

        if self._coupled_threechain_mode == COUPLED_THREECHAIN_RAW_CONTROL:
            hard_chain = raw_chain2
            hard_valid = raw_valid2
            diagnostics.update(
                {
                    "coupled_orthogonal_projection_enabled": 0.0,
                    "coupled_hard_available_fraction": hard_valid.float().mean().detach(),
                }
            )
        else:
            if self.latent_pool is None or self.decoder is None:
                raise RuntimeError(
                    "orthogonal-LHAT coupled three-chain requires pool and decoder"
                )
            if context.resource("vae_encoder") is not self.encoder:
                raise RuntimeError("coupled VAE encoder resource identity drifted")
            if context.resource("vae_decoder") is not self.decoder:
                raise RuntimeError("coupled VAE decoder resource identity drifted")
            lhat_source = torch.where(
                lhat.valid_mask[:, None, None],
                lhat.waveform,
                clean.waveform,
            )
            encoded = self._encode_coupled_waveforms(
                torch.cat(
                    (clean.waveform, raw_chain1, raw_chain2, lhat_source),
                    dim=0,
                )
            )
            clean_latent, raw1_latent, raw2_latent, lhat_latent = encoded.split(
                batch, dim=0
            )
            standardizer = self.latent_pool.standardizer
            clean_standardized = standardizer.transform(clean_latent)
            raw_standardized = torch.stack(
                (
                    standardizer.transform(raw1_latent),
                    standardizer.transform(raw2_latent),
                ),
                dim=1,
            )
            lhat_standardized = standardizer.transform(lhat_latent)
            projected_standardized, projection_valid, projection_diagnostics = (
                _orthogonalize_lhat_against_raw_span(
                    clean_standardized=clean_standardized,
                    raw_standardized=raw_standardized,
                    lhat_standardized=lhat_standardized,
                    raw_valid=torch.stack((raw_valid1, raw_valid2), dim=1),
                    lhat_valid=lhat.valid_mask,
                    radius_ratio=self._coupled_radius_ratio,
                    epsilon=self._coupled_orthogonality_epsilon,
                )
            )
            projected_latent = standardizer.inverse_transform(
                projected_standardized
            )
            with torch.no_grad():
                decoded = decode_to_ptbxl_waveform(
                    self.decoder,
                    torch.cat((projected_latent, clean_latent), dim=0),
                    target_points=1000,
                )
            decoded_projected, decoded_clean = decoded.split(batch, dim=0)
            proposed_hard = decoded_projected + (
                clean.waveform.float() - decoded_clean.float()
            )
            hard_quality, _ = _quality_mask(
                proposed_hard,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
                return_reasons=False,
            )
            hard_valid = (
                projection_valid
                & lhat.valid_mask
                & clean.valid_mask
                & hard_quality
            )
            hard_chain = torch.where(
                hard_valid[:, None, None],
                proposed_hard,
                clean.waveform,
            ).to(dtype=clean.waveform.dtype).contiguous()
            diagnostics.update(
                {
                    "coupled_orthogonal_projection_enabled": 1.0,
                    "coupled_hard_available_fraction": hard_valid.float().mean().detach(),
                    "coupled_hard_quality_fraction": hard_quality.float().mean().detach(),
                    **{
                        f"coupled_{name}": value.mean().detach()
                        for name, value in projection_diagnostics.items()
                    },
                }
            )

            classifier = context.resource("classifier")
            with _frozen_classifier_for_attack(classifier), torch.no_grad(), torch.autocast(
                device_type=clean.waveform.device.type,
                dtype=torch.bfloat16,
                enabled=clean.waveform.device.type == "cuda",
            ):
                logits = validate_model_output(
                    classifier(
                        prepare_canonical_model_input(
                            torch.cat((clean.waveform, hard_chain), dim=0),
                            self._model_spec,
                            epsilon=self.lhat_config.normalization_epsilon,
                        )
                    ),
                    self._model_spec,
                    batch_size=2 * batch,
                    check_finite=False,
                ).float()
            clean_logits, hard_logits = logits.split(batch, dim=0)
            clean_loss = _difficulty_per_sample(
                clean_logits, clean.labels, self._difficulty_objective
            )
            hard_loss = _difficulty_per_sample(
                hard_logits, clean.labels, self._difficulty_objective
            )
            truth = clean.labels >= 0.5
            clean_correct = (torch.sigmoid(clean_logits) >= 0.5) == truth
            hard_wrong = (torch.sigmoid(hard_logits) >= 0.5) != truth
            diagnostics.update(
                {
                    "coupled_projected_loss_gain": (
                        hard_loss - clean_loss
                    ).mean().detach(),
                    "coupled_projected_attack_success": (
                        clean_correct & hard_wrong
                    ).any(dim=1).float().mean().detach(),
                }
            )

        return _CoupledThreechainCache(
            raw_chain1=raw_chain1.detach().contiguous(),
            raw_chain2=raw_chain2.detach().contiguous(),
            hard_chain=hard_chain.detach().contiguous(),
            raw_chain1_valid=raw_valid1.detach(),
            raw_chain2_valid=raw_valid2.detach(),
            hard_chain_valid=hard_valid.detach(),
            raw_chain_depths=raw_depths.detach().contiguous(),
            raw_chain_operator_mask=raw_operator_mask.detach(),
            diagnostic_values=diagnostics,
        )

    def _coupled_threechain_view(
        self,
        *,
        context: NodeContext,
        clean: WaveformView,
        anchor: WaveformView,
    ) -> WaveformView:
        cache = self._coupled_threechain_cache
        if cache is None:
            raise RuntimeError("corrupted exposure lacks coupled three-chain cache")
        if cache.raw_chain1.shape != clean.waveform.shape:
            raise RuntimeError("coupled three-chain cache batch shape changed")
        third_chain = (
            cache.raw_chain2
            if self._coupled_threechain_mode == COUPLED_THREECHAIN_RAW_CONTROL
            else cache.hard_chain
        )
        third_valid = (
            cache.raw_chain2_valid
            if self._coupled_threechain_mode == COUPLED_THREECHAIN_RAW_CONTROL
            else cache.hard_chain_valid
        )
        raw1 = torch.where(
            cache.raw_chain1_valid[:, None, None],
            cache.raw_chain1,
            anchor.waveform,
        )
        third = torch.where(
            third_valid[:, None, None],
            third_chain,
            anchor.waveform,
        )
        mix_context = _MatchedScientificFactorContext(context, "raw")
        weights = _anchor_floored_dirichlet_weights(
            batch_size=clean.batch_size,
            canonical_mass_floor=self._coupled_canonical_mass_floor,
            device=clean.waveform.device,
            dtype=torch.float32,
            generator=mix_context.torch_generator(
                "coupled_mixture_weights", device=clean.waveform.device
            ),
        )
        strength = self._coupled_strength_min + (
            self._coupled_strength_max - self._coupled_strength_min
        ) * torch.rand(
            clean.batch_size,
            device=clean.waveform.device,
            dtype=torch.float32,
            generator=mix_context.torch_generator(
                "coupled_mixture_strength", device=clean.waveform.device
            ),
        )
        mixture = (
            weights[:, 0, None, None] * anchor.waveform.float()
            + weights[:, 1, None, None] * raw1.float()
            + weights[:, 2, None, None] * third.float()
        )
        proposed = (
            (1.0 - strength[:, None, None]) * clean.waveform.float()
            + strength[:, None, None] * mixture
        )
        canonical_third_cosine, canonical_third_applicable = (
            _masked_batch_absolute_cosine(
                anchor.waveform - clean.waveform,
                third - clean.waveform,
                anchor.valid_mask & third_valid,
                epsilon=self._coupled_orthogonality_epsilon,
            )
        )
        raw1_third_cosine, raw1_third_applicable = _masked_batch_absolute_cosine(
            raw1 - clean.waveform,
            third - clean.waveform,
            cache.raw_chain1_valid & third_valid,
            epsilon=self._coupled_orthogonality_epsilon,
        )
        quality, _ = _quality_mask(
            proposed,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
            return_reasons=False,
        )
        accepted = anchor.valid_mask & quality
        waveform = torch.where(
            accepted[:, None, None], proposed, anchor.waveform
        ).to(dtype=anchor.waveform.dtype).contiguous()
        diagnostics = {
            **dict(cache.diagnostic_values),
            "coupled_canonical_mass_mean": weights[:, 0].mean().detach(),
            "coupled_raw_mass_mean": weights[:, 1].mean().detach(),
            "coupled_third_mass_mean": weights[:, 2].mean().detach(),
            "coupled_strength_mean": strength.mean().detach(),
            "coupled_canonical_third_waveform_abs_cosine": canonical_third_cosine,
            "coupled_canonical_third_cosine_applicable_fraction": (
                canonical_third_applicable
            ),
            "coupled_raw1_third_waveform_abs_cosine": raw1_third_cosine,
            "coupled_raw1_third_cosine_applicable_fraction": raw1_third_applicable,
            "coupled_raw1_fallback_fraction": (
                ~cache.raw_chain1_valid
            ).float().mean().detach(),
            "coupled_third_fallback_fraction": (
                ~third_valid
            ).float().mean().detach(),
            "coupled_final_quality_fallback_fraction": (
                anchor.valid_mask & ~quality
            ).float().mean().detach(),
        }
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        return WaveformView(
            name=context.node_id,
            waveform=waveform,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=anchor.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="coupled_anchor_preserving_threechain_augmix",
                parent_names=(clean.name, anchor.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "mode": self._coupled_threechain_mode,
                    "chain1": "matching_fixed20_canonical_anchor",
                    "chain2": "shared_independent_depth23_raw",
                    "chain3": (
                        "shared_independent_depth23_raw"
                        if self._coupled_threechain_mode
                        == COUPLED_THREECHAIN_RAW_CONTROL
                        else "shared_orthogonalized_lhat_endpoint"
                    ),
                    "canonical_mass_floor": self._coupled_canonical_mass_floor,
                    "strength_min": self._coupled_strength_min,
                    "strength_max": self._coupled_strength_max,
                    "quality_failure_policy": "canonical_anchor_fallback",
                },
            ),
            metadata={
                "mixture_weights": weights.detach(),
                "augmented_strength": strength.detach(),
                "diagnostic_means": diagnostics,
                "diagnostic_weight": clean.batch_size,
            },
        )

    def _latent_simplex_hard_view(
        self,
        *,
        context: NodeContext,
        clean: WaveformView,
        anchor: WaveformView,
        lhat: WaveformView,
        random_waveform: torch.Tensor,
        random_valid: torch.Tensor,
    ) -> WaveformView:
        """Mix fixed20, random-corruption and VAE-LHAT chains in VAE space."""

        if self.encoder is None or self.decoder is None:
            raise RuntimeError("latent-simplex AugMax VAE resources are incomplete")
        classifier = context.resource("classifier")
        endpoint_valid = torch.stack(
            (anchor.valid_mask, random_valid, lhat.valid_mask), dim=1
        )
        raw_endpoints = torch.stack(
            (anchor.waveform, random_waveform, lhat.waveform), dim=1
        ).float()
        raw_endpoints = torch.where(
            endpoint_valid[:, :, None, None],
            raw_endpoints,
            anchor.waveform[:, None].float(),
        )
        generator = context.torch_generator(
            "latent_simplex_dirichlet", device=clean.waveform.device
        )
        weights = _anchor_floored_simplex_weights(
            batch_size=clean.batch_size,
            candidate_count=self._simplex_candidate_count,
            anchor_mass_floor=self._simplex_anchor_mass_floor,
            device=clean.waveform.device,
            dtype=torch.float32,
            generator=generator,
        )
        batch_size = clean.batch_size
        candidate_count = self._simplex_candidate_count
        with _frozen_classifier_for_attack(classifier), _frozen_vae_components(
            self.encoder, self.decoder
        ), torch.no_grad():
            flat_endpoints = raw_endpoints.reshape(batch_size * 3, 1000, 12)
            encoded = self.encoder(
                prepare_ecgtwin_encoder_input(flat_endpoints), sample=False
            )
            if not isinstance(encoded, tuple) or len(encoded) != 3:
                raise TypeError(
                    "latent-simplex encoder must return latent, mean and log variance"
                )
            endpoint_latents = encoded[0].float().reshape(batch_size, 3, 4, 128)
            mixed_latents = torch.einsum(
                "kbc,bcij->kbij", weights, endpoint_latents
            ).contiguous()
            decoded = decode_to_ptbxl_waveform(
                self.decoder,
                torch.cat(
                    (
                        mixed_latents.reshape(candidate_count * batch_size, 4, 128),
                        endpoint_latents.reshape(batch_size * 3, 4, 128),
                    ),
                    dim=0,
                ),
                target_points=1000,
            ).float()
            decoded_mixed = decoded[: candidate_count * batch_size].reshape(
                candidate_count, batch_size, 1000, 12
            )
            decoded_endpoints = decoded[candidate_count * batch_size :].reshape(
                batch_size, 3, 1000, 12
            )
            endpoint_residual = raw_endpoints - decoded_endpoints
            residual_mixed = torch.einsum(
                "kbc,bctl->kbtl", weights, endpoint_residual
            )
            candidates = decoded_mixed + residual_mixed
            # The first candidate is the exact fixed20 waveform by contract,
            # avoiding any numerical drift from encode/decode correction.
            candidates[0] = anchor.waveform.float()

            quality = torch.stack(
                tuple(
                    _quality_mask(
                        candidate,
                        minimum_std_mV=self.minimum_std_mV,
                        maximum_abs_mV=self.maximum_abs_mV,
                        return_reasons=False,
                    )[0]
                    & anchor.valid_mask
                    for candidate in candidates
                ),
                dim=0,
            )
            flattened = candidates.reshape(
                candidate_count * batch_size, 1000, 12
            )
            repeated_targets = clean.labels.unsqueeze(0).expand(
                candidate_count, -1, -1
            ).reshape(candidate_count * batch_size, -1)
            with torch.autocast(
                device_type=clean.waveform.device.type,
                dtype=torch.bfloat16,
                enabled=clean.waveform.device.type == "cuda",
            ):
                logits = validate_model_output(
                    classifier(
                        prepare_canonical_model_input(
                            flattened,
                            self._model_spec,
                            epsilon=(
                                1.0e-6
                                if self.lhat_config is None
                                else self.lhat_config.normalization_epsilon
                            ),
                        )
                    ),
                    self._model_spec,
                    batch_size=candidate_count * batch_size,
                    check_finite=False,
                ).float()
            finite = torch.isfinite(logits).all(dim=1).reshape(
                candidate_count, batch_size
            )
            losses = _difficulty_per_sample(
                torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0),
                repeated_targets,
                self._difficulty_objective,
            ).reshape(candidate_count, batch_size)
            gains = losses - losses[0:1]
            admissible = quality & finite & torch.isfinite(gains)
            if self._coefficient_selection in {"hard_max", "loss_softmax_sample"}:
                admissible &= (gains >= 0.0) & (
                    gains <= self._maximum_bce_gain
                )
            else:
                # The uniform arm is a classic diversity ablation; the
                # loss-softmax arm above remains genuinely difficulty-aware.
                admissible &= gains <= self._maximum_bce_gain
            admissible[0] = quality[0] & finite[0]
            scores = torch.where(
                admissible, gains, torch.full_like(gains, -torch.inf)
            )
            nonanchor_valid = admissible[1:].any(dim=0)
            if self._coefficient_selection == "hard_max":
                selected_index = scores.argmax(dim=0)
            else:
                selection_generator = context.torch_generator(
                    "latent_simplex_selection", device=clean.waveform.device
                )
                if self._coefficient_selection == "loss_softmax_sample":
                    sampling_logits = (
                        scores[1:] / self._selection_temperature
                    ).transpose(0, 1)
                    sampling_logits = torch.where(
                        nonanchor_valid[:, None],
                        sampling_logits,
                        torch.zeros_like(sampling_logits),
                    )
                    probabilities = torch.softmax(sampling_logits, dim=1)
                else:
                    probabilities = admissible[1:].transpose(0, 1).float()
                    probabilities = torch.where(
                        nonanchor_valid[:, None],
                        probabilities,
                        torch.ones_like(probabilities),
                    )
                    probabilities = probabilities / probabilities.sum(
                        dim=1, keepdim=True
                    )
                selected_index = 1 + torch.multinomial(
                    probabilities,
                    1,
                    replacement=True,
                    generator=selection_generator,
                ).squeeze(1)
                selected_index = torch.where(
                    nonanchor_valid,
                    selected_index,
                    torch.zeros_like(selected_index),
                )
            no_valid = ~admissible.any(dim=0)
            selected_index = torch.where(no_valid, torch.zeros_like(selected_index), selected_index)
            batch_index = torch.arange(batch_size, device=clean.waveform.device)
            waveform = candidates.permute(1, 0, 2, 3)[
                batch_index, selected_index
            ]
            selected_weights = weights.permute(1, 0, 2)[
                batch_index, selected_index
            ]
            selected_gain = gains.permute(1, 0)[batch_index, selected_index]
            entropy = -(
                selected_weights
                * selected_weights.clamp_min(1.0e-12).log()
            ).sum(dim=1)

        diagnostics = {
            "simplex_candidate_count": float(candidate_count),
            "simplex_anchor_mass_mean": float(
                selected_weights[:, 0].mean().detach().cpu()
            ),
            "simplex_random_mass_mean": float(
                selected_weights[:, 1].mean().detach().cpu()
            ),
            "simplex_lhat_mass_mean": float(
                selected_weights[:, 2].mean().detach().cpu()
            ),
            "simplex_entropy_mean": float(entropy.mean().detach().cpu()),
            "simplex_selected_bce_gain_mean": float(
                selected_gain.mean().detach().cpu()
            ),
            "simplex_exact_anchor_fraction": float(
                (selected_index == 0).float().mean().detach().cpu()
            ),
            "simplex_lhat_selected_fraction": float(
                (selected_weights[:, 2] > 1.0e-6).float().mean().detach().cpu()
            ),
            "simplex_admissible_nonanchor_fraction": float(
                admissible[1:].any(dim=0).float().mean().detach().cpu()
            ),
            "simplex_random_endpoint_valid_fraction": float(
                random_valid.float().mean().detach().cpu()
            ),
            "simplex_lhat_endpoint_valid_fraction": float(
                lhat.valid_mask.float().mean().detach().cpu()
            ),
            "simplex_selection_is_stochastic": float(
                self._coefficient_selection != "hard_max"
            ),
        }
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        return WaveformView(
            name=context.node_id,
            waveform=waveform.to(dtype=anchor.waveform.dtype).contiguous(),
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=anchor.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="vae_lhat_latent_simplex_augmax",
                parent_names=(clean.name, anchor.name, lhat.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "chains": [
                        "full_strength_fixed20_anchor",
                        "independent_depth23_corruption",
                        "vae_lhat_hard_waveform",
                    ],
                    "dirichlet_alpha": 1.0,
                    "candidate_count": candidate_count,
                    "anchor_mass_floor": self._simplex_anchor_mass_floor,
                    "latent_simplex_scope": self._latent_simplex_scope,
                    "lhat_refresh_interval_compositions": (
                        self._lhat_refresh_interval
                    ),
                    "selection": (
                        "hardest_nonnegative_bce_gain_under_cap"
                        if self._coefficient_selection == "hard_max"
                        else "replayable_nonanchor_sample_under_bce_gain_cap"
                    ),
                    "selection_mode": self._coefficient_selection,
                    "selection_temperature": self._selection_temperature,
                    "maximum_bce_gain": self._maximum_bce_gain,
                    "reconstruction_correction": (
                        "endpoint_consistent_same_simplex_weights"
                    ),
                    "quality_failure_policy": "canonical_anchor_fallback",
                },
            ),
            metadata={"diagnostic_means": diagnostics},
        )

    def _corruption_conditioned_hard_view(
        self,
        *,
        context: NodeContext,
        anchor: WaveformView,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Attack each canonical corruption in latent space with residual bypass."""

        if self.encoder is None or self.decoder is None or self.lhat_config is None:
            raise RuntimeError("conditioned LHAT resources are incomplete")
        classifier = context.resource("classifier")
        pool = context.resource("latent_pool")
        eligible_universe = pool.eligible_hash_id_set
        positions = tuple(
            index
            for index, hash_id in enumerate(anchor.sample_ids)
            if hash_id in eligible_universe and bool(anchor.valid_mask[index].item())
        )
        full_waveform = anchor.waveform.clone()
        full_valid = torch.zeros_like(anchor.valid_mask)
        diagnostics = {
            "conditioned_eligible_fraction": len(positions) / anchor.batch_size,
            "conditioned_quality_accepted_fraction": 0.0,
            "conditioned_loss_gain": 0.0,
            "conditioned_attack_success": 0.0,
            "conditioned_anchor_l2": 0.0,
            "conditioned_projection_scale": 0.0,
            "conditioned_residual_rms": 0.0,
        }
        if not positions:
            return full_waveform, full_valid, diagnostics

        index = torch.as_tensor(
            positions, device=anchor.waveform.device, dtype=torch.long
        )
        sample_ids = tuple(anchor.sample_ids[position] for position in positions)
        target = anchor.labels.index_select(0, index).float()
        candidate_generator = context.torch_generator(
            "conditioned_candidate_selection", device=anchor.waveform.device
        )
        attack_batch = pool.get_attack_batch_by_hashes(
            sample_ids,
            mode=self.lhat_config.candidate_mode,
            local_pool_size=self.lhat_config.local_pool_size,
            generator=candidate_generator,
        )
        if not torch.equal(
            attack_batch.labels.to(device=target.device, dtype=target.dtype), target
        ):
            raise RuntimeError("conditioned LHAT candidate labels drifted")
        anchor_raw = anchor.waveform.index_select(0, index).float()

        with _frozen_classifier_for_attack(classifier), _frozen_vae_components(
            self.encoder, self.decoder
        ):
            with torch.no_grad():
                encoded = self.encoder(
                    prepare_ecgtwin_encoder_input(anchor_raw), sample=False
                )
                if not isinstance(encoded, tuple) or len(encoded) != 3:
                    raise TypeError(
                        "conditioned LHAT encoder must return latent, mean and log variance"
                    )
                anchor_latent = encoded[0].float()
                if anchor_latent.shape != (len(positions), 4, 128):
                    raise ValueError("conditioned LHAT encoder returned an invalid shape")
                anchor_standardized = pool.standardizer.transform(anchor_latent)
                candidates_standardized = attack_batch.candidates_standardized.to(
                    device=anchor.waveform.device, dtype=torch.float32
                )
                reconstructed_anchor = decode_to_ptbxl_waveform(
                    self.decoder, anchor_latent, target_points=1000
                ).float()
                reconstruction_residual = (anchor_raw - reconstructed_anchor).detach()

            weight_logits = torch.zeros(
                len(positions),
                self.lhat_config.num_candidates,
                device=anchor.waveform.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam(
                (weight_logits,), lr=self.lhat_config.learning_rate
            )

            def waveform_from_standardized(value: torch.Tensor) -> torch.Tensor:
                latent = pool.standardizer.inverse_transform(value)
                decoded = decode_to_ptbxl_waveform(
                    self.decoder, latent, target_points=1000
                ).float()
                return decoded + reconstruction_residual

            def logits_from_standardized(value: torch.Tensor) -> torch.Tensor:
                waveform = waveform_from_standardized(value)
                model_input = prepare_canonical_model_input(
                    waveform,
                    self._model_spec,
                    epsilon=self.lhat_config.normalization_epsilon,
                )
                return validate_model_output(
                    classifier(model_input),
                    self._model_spec,
                    batch_size=len(positions),
                    check_finite=False,
                ).float()

            with torch.no_grad():
                anchor_logits = validate_model_output(
                    classifier(
                        prepare_canonical_model_input(
                            anchor_raw,
                            self._model_spec,
                            epsilon=self.lhat_config.normalization_epsilon,
                        )
                    ),
                    self._model_spec,
                    batch_size=len(positions),
                    check_finite=False,
                ).float()
                anchor_loss = _difficulty_per_sample(
                    anchor_logits, target, self._difficulty_objective
                )

            for _ in range(self.lhat_config.steps):
                optimizer.zero_grad(set_to_none=True)
                proposed, _, _ = _project_standardized_hull(
                    anchor_standardized,
                    candidates_standardized,
                    weight_logits,
                    hull_lambda=self.lhat_config.hull_lambda,
                    epsilon=self.lhat_config.pgd_epsilon,
                )
                loss = _difficulty_per_sample(
                    logits_from_standardized(proposed),
                    target,
                    self._difficulty_objective,
                ).mean()
                (gradient,) = torch.autograd.grad(loss, weight_logits)
                weight_logits.grad = -gradient
                optimizer.step()

            with torch.no_grad():
                final_latent, _, projection_scale = _project_standardized_hull(
                    anchor_standardized,
                    candidates_standardized,
                    weight_logits,
                    hull_lambda=self.lhat_config.hull_lambda,
                    epsilon=self.lhat_config.pgd_epsilon,
                )
                final_waveform = waveform_from_standardized(final_latent)
                final_logits = logits_from_standardized(final_latent)
                final_loss = _difficulty_per_sample(
                    final_logits, target, self._difficulty_objective
                )
                gain = final_loss - anchor_loss
                truth = target >= 0.5
                anchor_correct = (torch.sigmoid(anchor_logits) >= 0.5) == truth
                final_wrong = (torch.sigmoid(final_logits) >= 0.5) != truth
                success = (anchor_correct & final_wrong).any(dim=1)
                anchor_l2 = (final_latent - anchor_standardized).flatten(1).norm(
                    p=2, dim=1
                )

        accepted, _ = _quality_mask(
            final_waveform,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
            return_reasons=False,
        )
        accepted &= (
            torch.isfinite(final_logits).all(dim=1)
            & (gain >= self._conditioned_minimum_bce_gain)
            & (gain <= self._maximum_bce_gain)
        )
        accepted_local = torch.nonzero(accepted, as_tuple=False).flatten()
        if accepted_local.numel():
            accepted_global = index.index_select(0, accepted_local)
            full_waveform.index_copy_(
                0, accepted_global, final_waveform.index_select(0, accepted_local)
            )
            full_valid[accepted_global] = True
        diagnostics.update(
            {
                "conditioned_quality_accepted_fraction": float(
                    accepted.float().mean().detach().cpu()
                ),
                "conditioned_loss_gain": float(gain.mean().detach().cpu()),
                "conditioned_attack_success": float(
                    success.float().mean().detach().cpu()
                ),
                "conditioned_anchor_l2": float(anchor_l2.mean().detach().cpu()),
                "conditioned_projection_scale": float(
                    projection_scale.mean().detach().cpu()
                ),
                "conditioned_residual_rms": float(
                    reconstruction_residual.square()
                    .mean(dim=(1, 2))
                    .sqrt()
                    .mean()
                    .detach()
                    .cpu()
                ),
            }
        )
        return full_waveform, full_valid, diagnostics

    def _coverage_mix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if len(inputs) != 3 or any(not isinstance(value, WaveformView) for value in inputs):
            raise TypeError("coverage anchor mix requires clean, anchor and LHAT views")
        clean, anchor, lhat = inputs
        assert isinstance(clean, WaveformView)
        assert isinstance(anchor, WaveformView)
        assert isinstance(lhat, WaveformView)
        if not (clean.sample_ids == anchor.sample_ids == lhat.sample_ids):
            raise RuntimeError("coverage anchor inputs lost record alignment")
        if not (torch.equal(clean.labels, anchor.labels) and torch.equal(clean.labels, lhat.labels)):
            raise RuntimeError("coverage anchor inputs lost label alignment")
        if self._coupled_threechain_mode != COUPLED_THREECHAIN_DISABLED:
            if self._composition == -1:
                self._coupled_threechain_cache = (
                    self._build_coupled_threechain_cache(
                        context=context,
                        clean=clean,
                        lhat=lhat,
                    )
                )
                context.record_diagnostic("coupled_cache_build", 1.0)
                return WaveformView(
                    name=context.node_id,
                    waveform=clean.waveform,
                    labels=clean.labels,
                    sample_ids=clean.sample_ids,
                    valid_mask=clean.valid_mask,
                    provenance=Provenance(
                        node_id=context.node_id,
                        operation="coupled_fixed20_clean_identity_and_cache_build",
                        parent_names=(clean.name,),
                        rng_namespace=context.rng_namespace,
                        parameters={
                            "composition_index": -1,
                            "mode": self._coupled_threechain_mode,
                            "supervised_waveform_mutated": False,
                        },
                    ),
                )
            context.record_diagnostic("coupled_cache_reuse", 1.0)
            return self._coupled_threechain_view(
                context=context,
                clean=clean,
                anchor=anchor,
            )
        if self._composition == -1 and not (
            self._latent_simplex_augmax
            and self._latent_simplex_scope == "clean_exposure_only"
        ):
            context.record_diagnostic("clean_identity", 1.0)
            return WaveformView(
                name=context.node_id,
                waveform=clean.waveform,
                labels=clean.labels,
                sample_ids=clean.sample_ids,
                valid_mask=clean.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="fixed20_clean_identity",
                    parent_names=(clean.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={"composition_index": -1},
                ),
            )
        if self.operator_profile is None:
            raise RuntimeError("coverage anchor runtime has no operator profile")
        if not self._raw_diversity_enabled and not self._vae_lhat_enabled:
            context.record_diagnostic("factorial_identity_fast_path", 1.0)
            return WaveformView(
                name=context.node_id,
                waveform=anchor.waveform,
                labels=anchor.labels,
                sample_ids=anchor.sample_ids,
                valid_mask=anchor.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="direct_fixed20_factorial_identity",
                    parent_names=(anchor.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={
                        "scientific_factors": {
                            "raw_diversity": False,
                            "vae_lhat": False,
                        },
                        "waveform_mutated": False,
                        "classifier_forward_executed": False,
                        "quality_gate_repeated": False,
                    },
                ),
                metadata={"factorial_identity_fast_path": True},
            )

        if (
            self._latent_simplex_augmax
            and self._latent_simplex_scope == "clean_exposure_only"
        ):
            if self._composition == -1:
                parameters = {
                    name: self.operator_profile.parameters_for(name)
                    for name in self.operator_profile.canonical_order
                }
                generator = context.torch_generator(
                    "clean_branch_random_depth23", device=clean.waveform.device
                )
                random_result = generate_canonical_corruption(
                    clean.waveform,
                    operator_params=parameters,
                    generator=generator,
                    composition_indices=None,
                    _input_prevalidated=True,
                )
                random_valid = clean.valid_mask & (
                    random_result.diagnostics.output_nonfinite_count == 0
                )
                return self._latent_simplex_hard_view(
                    context=context,
                    clean=clean,
                    anchor=clean,
                    lhat=lhat,
                    random_waveform=random_result.waveform_raw_100hz,
                    random_valid=random_valid,
                )
            context.record_diagnostic("fixed20_anchor_unchanged", 1.0)
            return WaveformView(
                name=context.node_id,
                waveform=anchor.waveform,
                labels=anchor.labels,
                sample_ids=anchor.sample_ids,
                valid_mask=anchor.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="clean_branch_augmix_fixed20_anchor_identity",
                    parent_names=(anchor.name,),
                    rng_namespace=context.rng_namespace,
                    parameters={
                        "composition_index": self._composition,
                        "vae_augmix_applied": False,
                    },
                ),
            )

        if self._condition_on_corruption:
            hard_waveform, hard_valid, hard_diagnostics = (
                self._corruption_conditioned_hard_view(
                    context=context,
                    anchor=anchor,
                )
            )
            strength_generator = context.torch_generator(
                "conditioned_augmix_strength", device=clean.waveform.device
            )
            strength = self._conditioned_strength_min + (
                self._conditioned_strength_max - self._conditioned_strength_min
            ) * torch.rand(
                clean.batch_size,
                device=clean.waveform.device,
                dtype=torch.float32,
                generator=strength_generator,
            )
            proposed = anchor.waveform.float() + strength[:, None, None] * (
                hard_waveform.float() - anchor.waveform.float()
            )
            quality, _ = _quality_mask(
                proposed,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
                return_reasons=False,
            )
            accepted = hard_valid & quality & anchor.valid_mask
            waveform = torch.where(
                accepted[:, None, None], proposed, anchor.waveform
            ).to(dtype=anchor.waveform.dtype).contiguous()
            diagnostics = {
                **hard_diagnostics,
                "conditioned_augmix_strength_mean": float(
                    strength.mean().detach().cpu()
                ),
                "conditioned_augmix_strength_std": float(
                    strength.std(correction=0).detach().cpu()
                ),
                "conditioned_augmix_applied_fraction": float(
                    accepted.float().mean().detach().cpu()
                ),
                "quality_fallback_fraction": float(
                    (~accepted).float().mean().detach().cpu()
                ),
            }
            for name, value in diagnostics.items():
                context.record_diagnostic(name, value)
            return WaveformView(
                name=context.node_id,
                waveform=waveform,
                labels=clean.labels,
                sample_ids=clean.sample_ids,
                valid_mask=anchor.valid_mask,
                provenance=Provenance(
                    node_id=context.node_id,
                    operation="corruption_conditioned_vae_lhat_augmix",
                    parent_names=(clean.name, anchor.name, lhat.name),
                    rng_namespace=context.rng_namespace,
                    parameters={
                        "canonical_anchor_scale": 1.0,
                        "latent_anchor": "encoded_canonical_corruption",
                        "residual_bypass": "canonical_corruption_minus_vae_reconstruction",
                        "hull_lambda": self.lhat_config.hull_lambda,
                        "pgd_epsilon_l2_standardized": self.lhat_config.pgd_epsilon,
                        "hard_steps": self.lhat_config.steps,
                        "augmix_strength_min": self._conditioned_strength_min,
                        "augmix_strength_max": self._conditioned_strength_max,
                        "minimum_bce_gain": self._conditioned_minimum_bce_gain,
                        "maximum_bce_gain": self._maximum_bce_gain,
                        "quality_failure_policy": "canonical_anchor_fallback",
                    },
                ),
                metadata={"diagnostic_means": diagnostics},
            )

        if self._raw_diversity_enabled:
            parameters = {
                name: self.operator_profile.parameters_for(name)
                for name in self.operator_profile.canonical_order
            }
            raw_factor_context: NodeContext = context
            if self._match_scientific_factor_rng:
                raw_factor_context = _MatchedScientificFactorContext(context, "raw")
            generator = raw_factor_context.torch_generator(
                "independent_random_depth23_residual", device=clean.waveform.device
            )
            random_result = generate_canonical_corruption(
                clean.waveform,
                operator_params=parameters,
                generator=generator,
                composition_indices=None,
                _input_prevalidated=True,
            )
            random_valid = clean.valid_mask & (
                random_result.diagnostics.output_nonfinite_count == 0
            )
            random_waveform = random_result.waveform_raw_100hz
            random_delta = random_waveform - clean.waveform
            random_delta = torch.where(
                random_valid[:, None, None],
                random_delta,
                torch.zeros_like(random_delta),
            )
            random_mean_depth: float | torch.Tensor = (
                random_result.diagnostics.depth.float().mean().detach()
            )
        else:
            random_valid = clean.valid_mask
            random_waveform = clean.waveform
            random_delta = torch.zeros_like(clean.waveform)
            random_mean_depth = 0.0
        if self._latent_simplex_augmax:
            return self._latent_simplex_hard_view(
                context=context,
                clean=clean,
                anchor=anchor,
                lhat=lhat,
                random_waveform=random_waveform,
                random_valid=random_valid,
            )
        lhat_delta = (
            _lhat_adversarial_residual(lhat, clean)[0]
            if self._vae_lhat_enabled
            else torch.zeros_like(clean.waveform)
        )
        residual = (
            self._random_residual_scale * random_delta
            + self._lhat_residual_scale * lhat_delta
        )
        calibrated = self.method.profile_name == CALIBRATED_METHOD_ID
        selection_diagnostics: dict[str, float | torch.Tensor] = {}
        quality_fallback_fraction_override: float | torch.Tensor | None = None
        if calibrated and self._coefficient_selection == "replayable_factorized_sample":
            raw_selection_context = _MatchedScientificFactorContext(context, "raw")
            lhat_selection_context = _MatchedScientificFactorContext(context, "lhat")
            factorized = self._select_factorized_coefficients(
                anchor=anchor,
                random_delta=random_delta,
                random_valid=random_valid,
                lhat_delta=lhat_delta,
                lhat_valid=lhat.valid_mask,
                raw_generator=raw_selection_context.torch_generator(
                    "coefficient_selection", device=clean.waveform.device
                ),
                lhat_generator=lhat_selection_context.torch_generator(
                    "coefficient_selection", device=clean.waveform.device
                ),
            )
            self._record_factor_audit(
                sample_ids=clean.sample_ids,
                selection=factorized,
            )
            waveform = factorized.waveform
            accepted = factorized.quality_accepted
            assigned = factorized.assigned_coefficients
            effective = factorized.effective_coefficients
            assigned_nonzero = (assigned > 0.0).any(dim=1)
            effective_nonzero = (effective > 0.0).any(dim=1)
            quality_fallback_fraction_override = (
                factorized.quality_fallback.float().mean().detach()
            )
            selection_diagnostics = {
                "assigned_raw_coefficient_mean": assigned[:, 0].mean().detach(),
                "assigned_lhat_coefficient_mean": assigned[:, 1].mean().detach(),
                "effective_raw_coefficient_mean": effective[:, 0].mean().detach(),
                "effective_lhat_coefficient_mean": effective[:, 1].mean().detach(),
                "raw_selection_entropy": (
                    factorized.selection_entropy[:, 0].mean().detach()
                ),
                "lhat_selection_entropy": (
                    factorized.selection_entropy[:, 1].mean().detach()
                ),
                "intended_zero_dose_fraction": (
                    (~assigned_nonzero).float().mean().detach()
                ),
                "effective_zero_dose_fraction": (
                    (~effective_nonzero).float().mean().detach()
                ),
                "raw_factor_unavailable_fraction": (
                    factorized.raw_unavailable.float().mean().detach()
                ),
                "lhat_factor_unavailable_fraction": (
                    factorized.lhat_unavailable.float().mean().detach()
                ),
                "quality_fallback_fraction": quality_fallback_fraction_override,
                "factorized_applied_fraction": (
                    effective_nonzero.float().mean().detach()
                ),
                "joint_selected_fraction": (
                    ((effective[:, 0] > 0.0) & (effective[:, 1] > 0.0))
                    .float()
                    .mean()
                    .detach()
                ),
                "classifier_guided_selection": 0.0,
            }
        elif calibrated and self._residual_coefficient_grid is not None:
            selection_generator = context.torch_generator(
                "loss_aware_coefficient_selection", device=clean.waveform.device
            )
            (
                waveform,
                selected_coefficients,
                selected_gain,
                admissible,
                selected_index,
                selection_entropy,
            ) = (
                self._select_calibrated_coefficients(
                    anchor=anchor,
                    random_delta=random_delta,
                    lhat_delta=lhat_delta,
                    targets=clean.labels,
                    classifier=context.resource("classifier"),
                    selection_generator=selection_generator,
                )
            )
            accepted = anchor.valid_mask
            selection_diagnostics = {
                "selected_raw_coefficient_mean": float(
                    selected_coefficients[:, 0].mean().detach().cpu()
                ),
                "selected_lhat_coefficient_mean": float(
                    selected_coefficients[:, 1].mean().detach().cpu()
                ),
                "selection_entropy_mean": float(
                    selection_entropy.mean().detach().cpu()
                ),
                "zero_dose_fraction": float(
                    (selected_index == 0).float().mean().detach().cpu()
                ),
                "lhat_selected_fraction": float(
                    (selected_coefficients[:, 1] > 0.0).float().mean().detach().cpu()
                ),
                "joint_selected_fraction": float(
                    (
                        (selected_coefficients[:, 0] > 0.0)
                        & (selected_coefficients[:, 1] > 0.0)
                    ).float().mean().detach().cpu()
                ),
                "admissible_nonzero_fraction": float(
                    admissible[1:].any(dim=0).float().mean().detach().cpu()
                ),
            }
            if self._coefficient_selection == "replayable_weighted_sample":
                selection_diagnostics["classifier_guided_selection"] = 0.0
            else:
                selection_diagnostics["classifier_guided_selection"] = 1.0
                selection_diagnostics["selected_bce_gain_mean"] = float(
                    selected_gain.mean().detach().cpu()
                )
        elif calibrated:
            waveform, selected_dose, selected_gain, admissible = (
                self._select_calibrated_dose(
                    anchor=anchor,
                    residual=residual,
                    targets=clean.labels,
                    classifier=context.resource("classifier"),
                )
            )
            accepted = anchor.valid_mask
            selection_diagnostics = {
                "selected_dose_mean": float(selected_dose.mean().detach().cpu()),
                "selected_bce_gain_mean": float(selected_gain.mean().detach().cpu()),
                "zero_dose_fraction": float(
                    (selected_dose == 0.0).float().mean().detach().cpu()
                ),
                "full_or_higher_dose_fraction": float(
                    (selected_dose >= 1.0).float().mean().detach().cpu()
                ),
                "admissible_nonzero_fraction": float(
                    admissible[1:].any(dim=0).float().mean().detach().cpu()
                ),
            }
        else:
            proposed = anchor.waveform + residual
            accepted, _ = _quality_mask(
                proposed,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
                return_reasons=False,
            )
            accepted &= anchor.valid_mask
            waveform = torch.where(
                accepted[:, None, None], proposed, anchor.waveform
            )

        reference_rms = (anchor.waveform - clean.waveform).square().mean(dim=(1, 2)).sqrt()
        added_rms = (waveform - anchor.waveform).square().mean(dim=(1, 2)).sqrt()
        ratio = added_rms / reference_rms.clamp_min(1.0e-8)
        diagnostic_values: dict[str, float | torch.Tensor] = {
            "anchor_weight": 1.0,
            "random_residual_scale": self._random_residual_scale,
            "lhat_residual_scale": self._lhat_residual_scale,
            "raw_diversity_factor_enabled": float(
                self._raw_diversity_enabled
            ),
            "vae_lhat_factor_enabled": float(self._vae_lhat_enabled),
            "random_mean_depth": random_mean_depth,
            "lhat_availability_applicable": float(self._vae_lhat_enabled),
            "lhat_available_fraction": _lhat_availability_fraction(
                lhat.valid_mask,
                factor_enabled=self._vae_lhat_enabled,
            ),
            "added_to_anchor_rms_ratio": ratio.mean().detach(),
            "quality_fallback_fraction": (
                (~accepted).float().mean().detach()
            )
            if quality_fallback_fraction_override is None
            else quality_fallback_fraction_override,
            **selection_diagnostics,
        }
        diagnostics = dict(diagnostic_values)
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        return WaveformView(
            name=context.node_id,
            waveform=waveform,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=anchor.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation=(
                    "coverage_preserving_factorized_raw_diversity_lhat"
                    if self._coefficient_selection == "replayable_factorized_sample"
                    else "coverage_preserving_replayable_stochastic_augmix"
                    if self._coefficient_selection == "replayable_weighted_sample"
                    else "coverage_preserving_loss_calibrated_augmax"
                    if calibrated
                    else "coverage_preserving_residual_threechain_augmix"
                ),
                parent_names=(clean.name, anchor.name, lhat.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "canonical_anchor_scale": 1.0,
                    "scientific_factors": {
                        "raw_diversity": self._raw_diversity_enabled,
                        "vae_lhat": self._vae_lhat_enabled,
                    },
                    "random_corruption_residual_scale": self._random_residual_scale,
                    "lhat_residual_scale": self._lhat_residual_scale,
                    "random_depths": [2, 3],
                    "operator_domain_sampling_rate_hz": 500,
                    "quality_failure_policy": "canonical_anchor_fallback",
                    "difficulty_objective": self._difficulty_objective,
                    "coefficient_selection": self._coefficient_selection,
                    "coefficient_probabilities": (
                        None
                        if self._residual_coefficient_probabilities is None
                        else list(self._residual_coefficient_probabilities)
                    ),
                    "factor_distributions": (
                        {
                            "raw": {
                                "values": list(self._raw_factor_distribution[0]),
                                "probabilities": list(
                                    self._raw_factor_distribution[1]
                                ),
                            },
                            "lhat": {
                                "values": list(self._lhat_factor_distribution[0]),
                                "probabilities": list(
                                    self._lhat_factor_distribution[1]
                                ),
                            },
                        }
                        if self._coefficient_selection
                        == "replayable_factorized_sample"
                        and self._raw_factor_distribution is not None
                        and self._lhat_factor_distribution is not None
                        else None
                    ),
                    "matched_scientific_factor_rng": (
                        self._match_scientific_factor_rng
                    ),
                    "selection_temperature": (
                        None
                        if self._coefficient_selection
                        == "replayable_factorized_sample"
                        else self._selection_temperature
                    ),
                    "dose_selection": (
                        {
                            "doses": (
                                None
                                if self._coefficient_selection
                                == "replayable_factorized_sample"
                                else list(CALIBRATED_DOSES)
                            ),
                            "criterion": (
                                "independent_common_random_factor_draw_then_quality_gate"
                                if self._coefficient_selection
                                == "replayable_factorized_sample"
                                else "configured_replayable_weighted_sample_after_quality_gate"
                                if self._coefficient_selection
                                == "replayable_weighted_sample"
                                else "maximum_nonnegative_per_record_bce_gain_under_cap"
                            ),
                            "maximum_bce_gain": (
                                None
                                if self._coefficient_selection
                                == "replayable_factorized_sample"
                                else self._maximum_bce_gain
                            ),
                            "coefficient_grid": (
                                None
                                if self._residual_coefficient_grid is None
                                else [list(pair) for pair in self._residual_coefficient_grid]
                            ),
                        }
                        if calibrated
                        else "fixed_full_dose"
                    ),
                },
            ),
            metadata={
                "diagnostic_means": diagnostics,
                "diagnostic_weight": clean.batch_size,
            },
        )

    def _dispatch_augmix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if context.node_id == RESOURCE_BRIDGE_NODE_ID:
            return self._resource_bridge(context, inputs)
        if context.node_id == MIX_NODE_ID:
            return self._coverage_mix(context, inputs)
        return super()._dispatch_augmix(context, inputs)

    def generate(
        self,
        *,
        clean_raw: torch.Tensor,
        targets: torch.Tensor,
        hash_ids: Sequence[str],
        classifier: nn.Module,
        base_seed: int,
        rng_identity: Sequence[str],
        composition_indices: torch.Tensor | None = None,
        composition_index_hint: int | None = None,
    ):
        composition = _composition_index(
            composition_indices,
            hint=composition_index_hint,
        )
        identity = _group_identity(rng_identity, hash_ids)
        if composition == -1:
            if self._active_group is not None:
                raise RuntimeError("a new fixed20 group started before the old group ended")
            self._active_group = identity
        elif self._active_group != identity:
            raise RuntimeError("fixed20 exposure identity changed inside one base batch")
        self._composition = composition
        try:
            generated = super().generate(
                clean_raw=clean_raw,
                targets=targets,
                hash_ids=hash_ids,
                classifier=classifier,
                base_seed=base_seed,
                rng_identity=rng_identity,
                composition_indices=composition_indices,
                composition_index_hint=composition_index_hint,
            )
            if (
                composition == -1
                and self._latent_simplex_augmax
                and self._latent_simplex_scope == "clean_exposure_only"
            ):
                augmented = generated.bundle.require("corrupted_view")
                if not isinstance(augmented, WaveformView):
                    raise TypeError("clean-branch AugMix output must be a waveform")
                clean_augmented = WaveformView(
                    name=BASE_VIEW_NAME,
                    waveform=augmented.waveform,
                    labels=augmented.labels,
                    sample_ids=augmented.sample_ids,
                    valid_mask=augmented.valid_mask,
                    provenance=Provenance(
                        node_id="clean_branch_bundle_route",
                        operation="route_vae_lhat_augmix_to_clean_family",
                        parent_names=(augmented.name,),
                        parameters={
                            "composition_index": -1,
                            "fixed20_corrupted_views_mutated": False,
                        },
                    ),
                    metadata=augmented.metadata,
                )
                values = dict(generated.bundle.values)
                values[BASE_VIEW_NAME] = clean_augmented
                generated = replace(
                    generated,
                    bundle=ViewBundle(
                        values=values,
                        node_values=generated.bundle.node_values,
                        diagnostics=generated.bundle.diagnostics,
                    ),
                )
            if (
                composition == -1
                and not self._condition_on_corruption
                and self._vae_lhat_enabled
                and self._cached_lhat is None
            ):
                raise RuntimeError("clean exposure failed to populate LHAT cache")
            if (
                composition == -1
                and self._coupled_threechain_mode != COUPLED_THREECHAIN_DISABLED
                and self._coupled_threechain_cache is None
            ):
                raise RuntimeError(
                    "clean exposure failed to populate coupled three-chain cache"
                )
            if composition == 19:
                self._finalize_factor_audit_group()
            return generated
        except Exception:
            self._clear_group()
            raise
        finally:
            if composition == 19:
                self._clear_group()


def _is_supported(method: CompiledMethod) -> bool:
    return method.profile_name in SUPPORTED_METHOD_IDS


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Install the sandbox factory only for the explicit coverage method."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_exposure_steps = online_trainer._method_exposure_steps
    original_compute_objective = online_trainer._compute_objective

    def patched_exposure_steps(method: CompiledMethod):
        jsd_weight = float(method.contracts.get("augmix_jsd_weight", 0.0))
        dual_view_weight = float(
            method.contracts.get("dual_view_hard_weight", 0.0)
        )
        if not _is_supported(method) or (
            jsd_weight <= 0.0 and dual_view_weight <= 0.0
        ):
            return original_exposure_steps(method)
        names = {term.name for term in method.objective.terms}
        if dual_view_weight > 0.0:
            if jsd_weight > 0.0:
                raise ValueError("dual-view BCE and AugMix JSD are exclusive")
            if names != {"clean_bce", "anchor_bce", "hard_bce"}:
                raise ValueError(
                    "coverage dual-view objective must contain clean, anchor and hard BCE"
                )
        elif names != {"clean_bce", "corrupted_bce", "augmix_jsd"}:
            raise ValueError(
                "coverage AugMix JSD objective must contain clean/corrupted BCE and augmix_jsd"
            )
        order = tuple(int(value) for value in method.contracts.get("composition_indices", ()))
        if order != online_trainer.FIXED20_COMPOSITION_ORDER:
            raise ValueError("coverage AugMix JSD requires canonical fixed20 order")
        weights = method.contracts.get("family_loss_weights")
        if weights != {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.025,
        }:
            raise ValueError("coverage AugMix JSD changed fixed20 family weights")
        if method.contracts.get("optimizer_step_policy") != (
            "accumulate_family_balanced_once_per_base_batch"
        ):
            raise ValueError("coverage AugMix JSD changed optimizer-step policy")
        return (
            online_trainer._ExposureStep("clean", -1, ("clean_bce",)),
            *tuple(
                online_trainer._ExposureStep(
                    f"corruption_{index:02d}",
                    index,
                    (
                        ("anchor_bce", "hard_bce")
                        if dual_view_weight > 0.0
                        else ("corrupted_bce", "augmix_jsd")
                    ),
                )
                for index in order
            ),
        )

    def patched_compute_objective(**kwargs: Any):
        method = kwargs.get("method")
        selected = kwargs.get("objective_term_names")
        jsd_active = (
            isinstance(method, CompiledMethod)
            and _is_supported(method)
            and float(method.contracts.get("augmix_jsd_weight", 0.0)) > 0.0
            and selected is not None
            and "augmix_jsd" in tuple(str(value) for value in selected)
        )
        dual_view_active = (
            isinstance(method, CompiledMethod)
            and _is_supported(method)
            and float(method.contracts.get("dual_view_hard_weight", 0.0)) > 0.0
            and selected is not None
            and {"anchor_bce", "hard_bce"}.issubset(
                {str(value) for value in selected}
            )
        )
        if not jsd_active and not dual_view_active:
            return original_compute_objective(**kwargs)

        # The canonical corrupted view keeps the already-applied fixed20 BN
        # momentum.  Clean/anchor JSD reference forwards receive zero momentum,
        # so AugMix consistency cannot silently change the baseline's running-
        # statistics exposure budget.
        modules = tuple(
            module
            for module in kwargs["model"].modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and bool(module.track_running_stats)
        )
        scheduled = tuple(float(module.momentum) for module in modules)

        class _ScopedReferenceBnPlan:
            exposure_weights = (
                (1.0, 0.0)
                if dual_view_active
                else (1.0, 0.0, 0.0)
            )

            def apply(self, index: int) -> None:
                momentum_scale = 1.0 if index == 0 else 0.0
                for module, momentum in zip(modules, scheduled):
                    module.momentum = momentum_scale * momentum

        patched_kwargs = dict(kwargs)
        patched_kwargs["batch_norm_plan"] = _ScopedReferenceBnPlan()
        try:
            return original_compute_objective(**patched_kwargs)
        finally:
            for module, momentum in zip(modules, scheduled):
                module.momentum = momentum

    def patched_factory(
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: str | Path,
        latent_pool: Any | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        minimum_std_mV: float = 1.0e-4,
        maximum_abs_mV: float = 20.0,
    ) -> MethodViewRuntime:
        if not _is_supported(method):
            return original_factory(
                method,
                model_name=model_name,
                config_root=config_root,
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        return CoverageAnchorRuntime(
            method,
            model_name=model_name,
            config_root=Path(config_root).expanduser().resolve(),
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )

    online_trainer.build_method_runtime = patched_factory
    online_trainer._method_exposure_steps = patched_exposure_steps
    online_trainer._compute_objective = patched_compute_objective
    _reset_factor_audit_records()
    try:
        yield
    finally:
        online_trainer.build_method_runtime = original_factory
        online_trainer._method_exposure_steps = original_exposure_steps
        online_trainer._compute_objective = original_compute_objective


__all__ = [
    "CoverageAnchorRuntime",
    "CALIBRATED_DOSES",
    "CALIBRATED_METHOD_ID",
    "LHAT_RESIDUAL_SCALE",
    "MAX_PER_RECORD_BCE_GAIN",
    "METHOD_ID",
    "MIX_NODE_ID",
    "RANDOM_RESIDUAL_SCALE",
    "SUPPORTED_METHOD_IDS",
    "factor_audit_payload",
    "patch_online_trainer_runtime",
    "write_diagnostics",
]
