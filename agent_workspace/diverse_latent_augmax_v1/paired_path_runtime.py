"""Isolated runtime for the paired clean-to-AugMix latent path candidate.

Each allowlisted managed method profile declares three ordered views at
explicit fractions of one record-local path.  This module owns only the sandbox implementation and a
scoped trainer patch.  The whitelist trainer continues to own data loading,
normalization, optimizer/scheduler steps, validation, checkpointing, and run
records.

For every base batch the runtime:

* looks up each clean latent by its complete train400 hash ID;
* creates three independent depth-1/2/3 waveform-corruption chains;
* combines those chains with record-local Dirichlet(1,1,1) weights and no
  clean Beta mixture, producing one corrupt endpoint per clean record;
* encodes that endpoint with the deterministic VAE posterior mean;
* constructs exact standardized-latent ray points at the profile-declared t values;
* decodes those path latents in fixed chunks and applies linear clean/corrupt
  endpoint residual correction; and
* clears every batch-local tensor and RNG identity in ``finally``.

There is deliberately no cross-record absolute P95 radius cap.  The paired
clean-to-corrupt radius is measured and recorded, while each path point is
constrained by its exact distance ratios to the two paired endpoints.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from core.methods.contracts import NodeContext, Provenance, ViewValue, WaveformView
from core.methods.registry import CompiledMethod
from core.methods.runtime import MethodViewRuntime
from models.vae import decode_to_ptbxl_waveform

try:  # Package import and direct delegate subprocess import are both supported.
    from .canonical_pool import CanonicalPoolSourceIdentity
    from .runtime_adapter import (
        CLASSIFIER_CHUNK_SIZE,
        OUTPUT_POINTS,
        _encode_decode_waveforms,
        _frozen_vae_components,
        _generate_candidate_waveforms,
        _generator,
        _quality_mask,
    )
except ImportError:  # pragma: no cover - direct sibling import in delegate.py
    from canonical_pool import CanonicalPoolSourceIdentity  # type: ignore[no-redef]
    from runtime_adapter import (  # type: ignore[no-redef]
        CLASSIFIER_CHUNK_SIZE,
        OUTPUT_POINTS,
        _encode_decode_waveforms,
        _frozen_vae_components,
        _generate_candidate_waveforms,
        _generator,
        _quality_mask,
    )


METHOD_ID = "paired_augmix_latent_path_010_020_030_v1"
STRONG_ENDPOINT_METHOD_ID = "paired_augmix_latent_path_040_070_100_v1"
SUPPORTED_METHOD_IDS = frozenset({METHOD_ID, STRONG_ENDPOINT_METHOD_ID})
# Backwards-compatible public constants for the original v1 profile.  Runtime
# instances deliberately read their own ordered node/t mapping from the strict
# ``contracts.paired_path.t_values_by_node`` contract below.
PATH_T_BY_NODE: Mapping[str, float] = MappingProxyType(
    {"path_t10": 0.1, "path_t20": 0.2, "path_t30": 0.3}
)
PATH_NODE_IDS = frozenset(PATH_T_BY_NODE)
CHAIN_COUNT = 3
CHAIN_DEPTHS = (1, 2, 3)
LATENT_DIMENSIONS = 4 * 128
OBJECTIVE_GLOBAL_SCALE = 0.5
DECODE_CHUNK_SIZE = CLASSIFIER_CHUNK_SIZE
SEED_NAMESPACE = "paired_augmix_latent_path_shared_endpoint_v1"
SANDBOX_ROOT = Path(__file__).resolve().parent


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positions(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _derive_seed(
    base_seed: int,
    rng_identity: Sequence[str],
    stream: str,
) -> int:
    """Derive a replayable uint32 seed without touching process-global RNG."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not 0 <= base_seed < 2**32:
        raise ValueError("base_seed must be uint32")
    identity = tuple(str(value) for value in rng_identity)
    if not identity or any(not value for value in identity):
        raise ValueError("rng_identity must contain non-empty values")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")
    payload = "|".join(
        (str(base_seed), SEED_NAMESPACE, stream, *identity)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def mix_three_corruption_chains(
    chains: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix ``(B,3,1000,12)`` chains with isolated Dirichlet(1) draws."""

    if not isinstance(chains, torch.Tensor):
        raise TypeError("chains must be a torch.Tensor")
    if chains.ndim != 4 or tuple(chains.shape[1:]) != (3, 1000, 12):
        raise ValueError("chains must have shape (B,3,1000,12)")
    if not chains.is_floating_point() or not bool(torch.isfinite(chains).all()):
        raise ValueError("chains must be finite floating-point raw mV")
    # Independent Exp(1) variates normalized across width are exactly a
    # Dirichlet(1,1,1) draw.  Sampling from the explicit generator keeps this
    # branch independent of process-global RNG state.
    uniform = torch.rand(
        int(chains.shape[0]),
        CHAIN_COUNT,
        device=chains.device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    exponential = -uniform.log()
    weights = exponential / exponential.sum(dim=1, keepdim=True)
    mixed = (chains * weights[:, :, None, None].to(chains.dtype)).sum(dim=1)
    return mixed.contiguous(), weights.contiguous()


@dataclass(frozen=True)
class PairedPathProjection:
    """One exact fixed-t ray point and its paired-lens diagnostics."""

    standardized_latent: torch.Tensor
    configured_t: float
    effective_t: torch.Tensor
    pair_radius_rms: torch.Tensor
    clean_distance_rms: torch.Tensor
    corrupt_distance_rms: torch.Tensor
    clean_distance_ratio: torch.Tensor
    corrupt_distance_ratio: torch.Tensor
    direction_cosine: torch.Tensor
    finite_mask: torch.Tensor


def project_paired_latent_path(
    clean_standardized: torch.Tensor,
    corrupt_standardized: torch.Tensor,
    *,
    t: float,
    epsilon: float = 1.0e-8,
) -> PairedPathProjection:
    """Construct an exact standardized paired ray/lens point.

    No cross-record radius statistic participates in this function.  For a
    non-degenerate pair, the result is exactly ``clean + t*(corrupt-clean)``;
    therefore its clean and corrupt distances are ``t*R`` and ``(1-t)*R``.
    """

    if not isinstance(clean_standardized, torch.Tensor) or not isinstance(
        corrupt_standardized, torch.Tensor
    ):
        raise TypeError("paired standardized latents must be torch.Tensor values")
    if clean_standardized.shape != corrupt_standardized.shape:
        raise ValueError("clean and corrupt standardized latents must align")
    if clean_standardized.ndim != 3 or tuple(clean_standardized.shape[1:]) != (
        4,
        128,
    ):
        raise ValueError("standardized latents must have shape (B,4,128)")
    if not clean_standardized.is_floating_point() or not (
        corrupt_standardized.is_floating_point()
    ):
        raise TypeError("standardized latents must be floating point")
    resolved_t = float(t)
    if not math.isfinite(resolved_t) or not 0.0 <= resolved_t <= 1.0:
        raise ValueError("t must be finite and lie in [0,1]")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")

    clean = clean_standardized.to(dtype=torch.float32)
    corrupt = corrupt_standardized.to(device=clean.device, dtype=torch.float32)
    delta = corrupt - clean
    pair_l2 = delta.flatten(1).norm(p=2, dim=1)
    scale = math.sqrt(LATENT_DIMENSIONS)
    pair_radius = pair_l2 / scale
    projected = clean + resolved_t * delta
    clean_delta = projected - clean
    corrupt_delta = corrupt - projected
    clean_distance = clean_delta.flatten(1).norm(p=2, dim=1) / scale
    corrupt_distance = corrupt_delta.flatten(1).norm(p=2, dim=1) / scale
    safe_radius = pair_radius.clamp_min(float(epsilon))
    clean_ratio = clean_distance / safe_radius
    corrupt_ratio = corrupt_distance / safe_radius
    direction_denominator = (
        clean_delta.flatten(1).norm(p=2, dim=1) * pair_l2
    ).clamp_min(float(epsilon))
    direction_cosine = (
        clean_delta.flatten(1) * delta.flatten(1)
    ).sum(dim=1) / direction_denominator
    finite = (
        torch.isfinite(clean).flatten(1).all(dim=1)
        & torch.isfinite(corrupt).flatten(1).all(dim=1)
        & torch.isfinite(projected).flatten(1).all(dim=1)
        & torch.isfinite(pair_radius)
        & (pair_radius > float(epsilon))
    )
    effective = torch.full_like(pair_radius, resolved_t)
    return PairedPathProjection(
        standardized_latent=torch.nan_to_num(projected).contiguous(),
        configured_t=resolved_t,
        effective_t=effective,
        pair_radius_rms=torch.nan_to_num(pair_radius),
        clean_distance_rms=torch.nan_to_num(clean_distance),
        corrupt_distance_rms=torch.nan_to_num(corrupt_distance),
        clean_distance_ratio=torch.nan_to_num(clean_ratio),
        corrupt_distance_ratio=torch.nan_to_num(corrupt_ratio),
        direction_cosine=torch.nan_to_num(direction_cosine),
        finite_mask=finite.contiguous(),
    )


def apply_linear_endpoint_residual_correction(
    decoded_paths: torch.Tensor,
    *,
    clean_raw: torch.Tensor,
    corrupt_raw: torch.Tensor,
    clean_decoded: torch.Tensor,
    corrupt_decoded: torch.Tensor,
    t_values: Sequence[float],
) -> torch.Tensor:
    """Apply ``D(z_t)+(1-t)r_clean+t*r_corrupt`` in raw waveform space."""

    tensors = (clean_raw, corrupt_raw, clean_decoded, corrupt_decoded)
    if decoded_paths.ndim != 4 or tuple(decoded_paths.shape[2:]) != (1000, 12):
        raise ValueError("decoded_paths must have shape (B,K,1000,12)")
    if any(
        not isinstance(value, torch.Tensor)
        or value.shape != decoded_paths.shape[:1] + decoded_paths.shape[2:]
        for value in tensors
    ):
        raise ValueError("endpoint waveforms must align as (B,1000,12)")
    resolved_t = tuple(float(value) for value in t_values)
    if len(resolved_t) != int(decoded_paths.shape[1]) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in resolved_t
    ):
        raise ValueError("t_values must align with K and lie in [0,1]")
    t_tensor = torch.tensor(
        resolved_t,
        device=decoded_paths.device,
        dtype=decoded_paths.dtype,
    ).view(1, -1, 1, 1)
    clean_residual = (clean_raw - clean_decoded).unsqueeze(1)
    corrupt_residual = (corrupt_raw - corrupt_decoded).unsqueeze(1)
    corrected = (
        decoded_paths
        + (1.0 - t_tensor) * clean_residual
        + t_tensor * corrupt_residual
    )
    return corrected.contiguous()


def _decode_latents_chunked(
    decoder: nn.Module,
    latents: torch.Tensor,
    *,
    chunk_size: int = DECODE_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a logical latent batch in fixed-size, order-preserving chunks."""

    if latents.ndim != 3 or tuple(latents.shape[1:]) != (4, 128):
        raise ValueError("latents must have shape (N,4,128)")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    decoded: list[torch.Tensor] = []
    finite: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(latents.shape[0]), chunk_size):
            latent = latents[start : start + chunk_size]
            waveform = decode_to_ptbxl_waveform(
                decoder, latent, target_points=OUTPUT_POINTS
            )
            valid = (
                torch.isfinite(latent).flatten(1).all(dim=1)
                & torch.isfinite(waveform).flatten(1).all(dim=1)
            )
            decoded.append(torch.nan_to_num(waveform).detach())
            finite.append(valid.detach())
    return (
        torch.cat(decoded, dim=0).contiguous(),
        torch.cat(finite, dim=0).contiguous(),
    )


@dataclass(frozen=True)
class PairedPathContract:
    """Strict parser for ``contracts.paired_path``."""

    chain_count: int
    chain_depths: tuple[int, ...]
    chain_depth_sampling: str
    waveform_chain_mixing: str
    clean_beta_mix: bool
    t_values_by_node: Mapping[str, float]
    standardizer_source: str
    geometry: str
    absolute_radius_cap: None
    posterior: str
    residual_correction: str
    batch_local_cache: bool

    @classmethod
    def from_method(cls, method: CompiledMethod) -> "PairedPathContract":
        if method.profile_name not in SUPPORTED_METHOD_IDS:
            raise ValueError(f"unsupported paired-path method {method.profile_name!r}")
        raw = _mapping(method.contracts.get("paired_path"), "contracts.paired_path")
        expected_keys = {
            "chain_count",
            "chain_depths",
            "chain_depth_sampling",
            "waveform_chain_mixing",
            "clean_beta_mix",
            "t_values_by_node",
            "standardizer_source",
            "geometry",
            "absolute_radius_cap",
            "posterior",
            "residual_correction",
            "batch_local_cache",
        }
        if set(raw) != expected_keys:
            raise ValueError(
                "contracts.paired_path keys must be exactly "
                f"{sorted(expected_keys)}"
            )
        if raw["chain_count"] != CHAIN_COUNT:
            raise ValueError("paired_path.chain_count must be 3")
        depths = tuple(raw["chain_depths"])
        if depths != CHAIN_DEPTHS:
            raise ValueError("paired_path.chain_depths must be [1,2,3]")
        t_values = _mapping(
            raw["t_values_by_node"], "paired_path.t_values_by_node"
        )
        if len(t_values) != 3:
            raise ValueError("paired_path.t_values_by_node must declare exactly 3 views")
        resolved_t_values: dict[str, float] = {}
        previous_t = 0.0
        for node_id, value in t_values.items():
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(
                    "paired_path.t_values_by_node keys must be non-empty node ids"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "paired_path.t_values_by_node values must be numeric"
                )
            resolved_t = float(value)
            if not math.isfinite(resolved_t) or not 0.0 < resolved_t <= 1.0:
                raise ValueError(
                    "paired_path.t_values_by_node values must satisfy 0 < t <= 1"
                )
            if resolved_t <= previous_t:
                raise ValueError(
                    "paired_path.t_values_by_node values must be strictly increasing"
                )
            resolved_t_values[node_id] = resolved_t
            previous_t = resolved_t
        if not set(resolved_t_values) <= set(method.outputs):
            raise ValueError(
                "paired_path.t_values_by_node keys must name exposed method outputs"
            )
        expected_literals = {
            "chain_depth_sampling": "uniform_per_chain",
            "waveform_chain_mixing": "dirichlet_alpha_1",
            "standardizer_source": "per_center_train400",
            "geometry": "standardized_paired_ray_lens",
            "posterior": "deterministic_mean",
            "residual_correction": "linear_clean_corrupt_endpoint",
        }
        for key, expected in expected_literals.items():
            if raw[key] != expected:
                raise ValueError(f"paired_path.{key} must be {expected!r}")
        if raw["clean_beta_mix"] is not False:
            raise ValueError("paired_path.clean_beta_mix must be false")
        if raw["absolute_radius_cap"] is not None:
            raise ValueError("paired_path.absolute_radius_cap must be null")
        if raw["batch_local_cache"] is not True:
            raise ValueError("paired_path.batch_local_cache must be true")
        return cls(
            chain_count=CHAIN_COUNT,
            chain_depths=depths,
            chain_depth_sampling=str(raw["chain_depth_sampling"]),
            waveform_chain_mixing=str(raw["waveform_chain_mixing"]),
            clean_beta_mix=False,
            t_values_by_node=MappingProxyType(resolved_t_values),
            standardizer_source=str(raw["standardizer_source"]),
            geometry=str(raw["geometry"]),
            absolute_radius_cap=None,
            posterior=str(raw["posterior"]),
            residual_correction=str(raw["residual_correction"]),
            batch_local_cache=True,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "chain_count": self.chain_count,
            "chain_depths": list(self.chain_depths),
            "chain_depth_sampling": self.chain_depth_sampling,
            "waveform_chain_mixing": self.waveform_chain_mixing,
            "clean_beta_mix": self.clean_beta_mix,
            "t_values_by_node": dict(self.t_values_by_node),
            "standardizer_source": self.standardizer_source,
            "geometry": self.geometry,
            "absolute_radius_cap": self.absolute_radius_cap,
            "posterior": self.posterior,
            "residual_correction": self.residual_correction,
            "batch_local_cache": self.batch_local_cache,
        }


@dataclass(frozen=True)
class PairedPathBatchCache:
    sample_ids: tuple[str, ...]
    endpoint_seed: int
    corruption_seed: int
    mixing_seed: int
    clean_raw: torch.Tensor
    corrupt_raw: torch.Tensor
    path_waveforms: Mapping[str, torch.Tensor]
    projections: Mapping[str, PairedPathProjection]
    eligible_mask: torch.Tensor
    chain_weights: torch.Tensor
    chain_depths: torch.Tensor
    chain_operator_mask: torch.Tensor
    chain_output_nonfinite_count: torch.Tensor
    clean_residual_rms: torch.Tensor
    corrupt_residual_rms: torch.Tensor


class PairedAugMixLatentPathRuntime(MethodViewRuntime):
    """Shared one-endpoint, three-fixed-path-view sandbox runtime."""

    def __init__(
        self,
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: Path,
        latent_pool: Any | None,
        encoder: nn.Module | None,
        decoder: nn.Module | None,
        minimum_std_mV: float,
        maximum_abs_mV: float,
    ) -> None:
        super().__init__(
            method,
            model_name=model_name,
            config_root=config_root,
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )
        self.paired_path = PairedPathContract.from_method(method)
        self._path_t_by_node = self.paired_path.t_values_by_node
        self._path_node_ids = frozenset(self._path_t_by_node)
        if latent_pool is None or encoder is None or decoder is None:
            raise ValueError(
                "paired path requires train400 latent_pool, encoder and decoder"
            )
        source_identity = getattr(latent_pool, "canonical_source_identity", None)
        # The ordinary whitelist train400 builder returns LatentPool directly;
        # the sandbox CanonicalGeometryPool wrapper is optional for this method.
        # When its managed source identity is present, validate it as additional
        # lineage evidence, but do not require the wrapper merely to recover
        # clean anchors by exact hash.
        if source_identity is not None and not isinstance(
            source_identity, CanonicalPoolSourceIdentity
        ):
            raise TypeError("paired path canonical source identity has wrong type")
        if source_identity is not None and (
            source_identity.partition != "k500_tune_train"
            or source_identity.record_count != 400
        ):
            raise ValueError("paired path currently supports train400 tuning only")
        identity = getattr(latent_pool, "identity", None)
        standardizer = getattr(latent_pool, "standardizer", None)
        if identity is None or standardizer is None:
            raise TypeError("paired path latent_pool lacks identity/standardizer")
        if int(identity.record_count) != 400:
            raise ValueError("paired path requires a 400-record geometry pool")
        # Whitelist LatentPool fits its coordinate statistics on the M=20-
        # eligible subset of the 400-record training pool.  The fit count is
        # therefore center-dependent (and intentionally less than 400 for the
        # current centers), while identity.record_count above remains the hard
        # guard excluding a full-K500/refit pool.
        if not 2 <= int(standardizer.count) <= 400:
            raise ValueError("paired path train400 standardizer fit count is invalid")
        if str(identity.standardizer_sha256) != str(standardizer.identity_sha256):
            raise ValueError("paired path standardizer identity is inconsistent")
        if self.augmix_config is None:
            raise ValueError("paired path requires the managed AugMix config")
        if not math.isclose(
            float(self.augmix_config.dirichlet_alpha),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("paired path requires Dirichlet alpha=1")
        if method.contracts.get("objective_global_scale") != OBJECTIVE_GLOBAL_SCALE:
            raise ValueError("contracts.objective_global_scale must be 0.5")

        node_ids = {node.profile.node_id for node in method.nodes}
        if not self._path_node_ids <= node_ids or "resource_bridge" not in node_ids:
            raise ValueError("paired path graph lacks fixed path nodes/resource bridge")
        if not self._path_node_ids <= set(method.outputs):
            raise ValueError("paired path graph must expose all three path outputs")
        expected_objective = {
            ("bce", ("clean_view",)): 1.0,
            **{
                ("bce", (node_id,)): 1.0 / 3.0
                for node_id in self._path_t_by_node
            },
        }
        actual_objective = {
            (term.kind, tuple(term.views)): float(term.weight)
            for term in method.objective.terms
        }
        if set(actual_objective) != set(expected_objective) or any(
            not math.isclose(
                actual_objective[key], expected, rel_tol=0.0, abs_tol=1.0e-12
            )
            for key, expected in expected_objective.items()
        ):
            raise ValueError("paired path declared objective does not match its contract")
        expected_bn = {
            "clean_view": 0.5,
            **{node_id: 1.0 / 6.0 for node_id in self._path_t_by_node},
        }
        actual_bn = _mapping(
            method.contracts.get("batch_norm_objective_view_weights"),
            "batch_norm_objective_view_weights",
        )
        if set(actual_bn) != set(expected_bn) or any(
            not math.isclose(float(actual_bn[key]), expected, rel_tol=0.0, abs_tol=1e-12)
            for key, expected in expected_bn.items()
        ):
            raise ValueError("paired path BN view weights do not match effective objective")
        if method.contracts.get("one_outer_optimizer_step_per_clean_batch") is not True:
            raise ValueError("paired path requires one outer optimizer step per batch")

        seed_paths = set(self._rng_seed_paths.values())
        if len(seed_paths) != 1:
            raise ValueError("paired path requires exactly one RNG seed config")
        self._seed_config_sha256 = _sha256(next(iter(seed_paths)))
        augmix_path = self._resource_paths.get("augmix_config")
        if augmix_path is None:
            raise ValueError("paired path lacks a resolved augmix_config path")
        self._augmix_config_sha256 = _sha256(augmix_path)
        self._runtime_path = Path(__file__).resolve()
        self._runtime_sha256 = _sha256(self._runtime_path)
        self._shared_cache: PairedPathBatchCache | None = None
        self._generation_state: tuple[int, tuple[str, ...]] | None = None

    @property
    def shared_cache_active(self) -> bool:
        return self._shared_cache is not None

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        runtime_identity = {
            "path": str(self._runtime_path),
            "sha256": self._runtime_sha256,
        }
        base["method_resources"] = {
            **dict(base.get("method_resources", {})),
            "paired_path_runtime_code": runtime_identity,
        }
        base["sandbox_paired_augmix_latent_path"] = {
            "contract": self.paired_path.describe(),
            "runtime_code": runtime_identity,
            "endpoint_construction": {
                "corruption_chains": 3,
                "depths": [1, 2, 3],
                "depth_sampling": "uniform_per_chain",
                "mixing": "record_local_dirichlet_alpha_1",
                "clean_beta_mix": False,
                "posterior": "deterministic_mean",
            },
            "t_values_by_node": dict(self._path_t_by_node),
            "geometry": {
                "domain": "train400_coordinate_standardized_latent",
                "constraint": "exact_paired_ray_lens_distance_ratios",
                "absolute_radius_cap": None,
                "cross_record_absolute_cap_used": False,
                "pair_radius": "diagnostic_only_l2_over_sqrt_512",
            },
            "residual_correction": "linear_clean_corrupt_endpoint",
            "train400_standardizer": {
                **self.latent_pool.standardizer.describe(),
                "fit_population": "M20_eligible_subset_of_per_center_train400",
                "not_full_train400_population": (
                    int(self.latent_pool.standardizer.count) != 400
                ),
            },
            "latent_pool_identity": self.latent_pool.identity.describe(),
            "canonical_pool_source_identity": (
                None
                if getattr(self.latent_pool, "canonical_source_identity", None)
                is None
                else self.latent_pool.canonical_source_identity.describe()
            ),
            "rng": {
                "seed_config_sha256": self._seed_config_sha256,
                "derivation_namespace": SEED_NAMESPACE,
                "effective_endpoint_seed": "recorded_per_batch_in_view_provenance",
                "isolated_substreams": ["corruption_chains", "dirichlet_weights"],
            },
            "logical_decode_batch": "clean_plus_corrupt_plus_three_paths",
            "decode_chunk_size": DECODE_CHUNK_SIZE,
            "cache_scope": "one_generate_call_then_clear_in_finally",
            "objective_global_scale": OBJECTIVE_GLOBAL_SCALE,
            "effective_objective_weights": {
                term.name: float(term.weight) * OBJECTIVE_GLOBAL_SCALE
                for term in self.method.objective.terms
            },
        }
        return base

    def _build_batch_cache(
        self,
        clean_raw: torch.Tensor,
        targets: torch.Tensor,
        hash_ids: Sequence[str],
        *,
        endpoint_seed: int,
        corruption_seed: int,
        mixing_seed: int,
    ) -> PairedPathBatchCache:
        assert self.encoder is not None and self.decoder is not None
        assert self.augmix_config is not None
        if clean_raw.ndim != 3 or tuple(clean_raw.shape[1:]) != (1000, 12):
            raise ValueError("clean_raw must have shape (B,1000,12)")
        if not clean_raw.is_floating_point() or not bool(torch.isfinite(clean_raw).all()):
            raise ValueError("clean_raw must be finite floating-point raw mV")
        batch_size = int(clean_raw.shape[0])
        sample_ids = tuple(str(value) for value in hash_ids)
        if len(sample_ids) != batch_size or len(set(sample_ids)) != batch_size:
            raise ValueError("hash_ids must be unique and align with the clean batch")
        if targets.shape != (batch_size, 5) or not bool(torch.isfinite(targets).all()):
            raise ValueError("targets must be finite and have shape (B,5)")

        pool_indices = self.latent_pool.indices_for_hashes(sample_ids)
        pool_labels = self.latent_pool.labels.index_select(0, pool_indices)
        if not torch.equal(
            pool_labels.detach().to(device="cpu", dtype=torch.float32),
            targets.detach().to(device="cpu", dtype=torch.float32),
        ):
            raise ValueError("clean hash lookup labels differ from outer targets")
        clean = clean_raw.to(dtype=torch.float32).contiguous()
        clean_latent = self.latent_pool.latents.index_select(0, pool_indices).to(
            device=clean.device, dtype=torch.float32
        )
        if clean_latent.shape != (batch_size, 4, 128) or not bool(
            torch.isfinite(clean_latent).all()
        ):
            raise ValueError("train400 clean latent lookup is invalid")

        corruption_generator = _generator(clean.device, corruption_seed)
        chains, depths, operator_mask, chain_nonfinite = _generate_candidate_waveforms(
            clean,
            candidate_count=CHAIN_COUNT,
            allowed_depths=CHAIN_DEPTHS,
            depth_sampling="uniform_per_candidate",
            augmix_config=self.augmix_config,
            generator=corruption_generator,
        )
        mixing_generator = _generator(clean.device, mixing_seed)
        corrupt, chain_weights = mix_three_corruption_chains(
            chains, generator=mixing_generator
        )
        corrupt_latent, corrupt_decoded, corrupt_finite = _encode_decode_waveforms(
            corrupt,
            encoder=self.encoder,
            decoder=self.decoder,
        )
        clean_decoded, clean_decode_finite = _decode_latents_chunked(
            self.decoder, clean_latent, chunk_size=DECODE_CHUNK_SIZE
        )

        standardizer = self.latent_pool.standardizer
        clean_standardized = standardizer.transform(clean_latent)
        corrupt_standardized = standardizer.transform(corrupt_latent)
        projections = {
            node_id: project_paired_latent_path(
                clean_standardized,
                corrupt_standardized,
                t=t_value,
            )
            for node_id, t_value in self._path_t_by_node.items()
        }
        standardized_paths = torch.stack(
            [
                projections[node_id].standardized_latent
                for node_id in self._path_t_by_node
            ],
            dim=1,
        )
        path_latents = standardizer.inverse_transform(
            standardized_paths.reshape(
                batch_size * len(self._path_t_by_node), 4, 128
            )
        )
        decoded_paths_flat, decoded_paths_finite_flat = _decode_latents_chunked(
            self.decoder, path_latents, chunk_size=DECODE_CHUNK_SIZE
        )
        decoded_paths = decoded_paths_flat.reshape(
            batch_size, len(self._path_t_by_node), OUTPUT_POINTS, 12
        )
        decoded_paths_finite = decoded_paths_finite_flat.reshape(
            batch_size, len(self._path_t_by_node)
        )
        corrected_paths = apply_linear_endpoint_residual_correction(
            decoded_paths,
            clean_raw=clean,
            corrupt_raw=corrupt,
            clean_decoded=clean_decoded,
            corrupt_decoded=corrupt_decoded,
            t_values=tuple(self._path_t_by_node.values()),
        )
        corrected_finite = torch.isfinite(corrected_paths).flatten(2).all(dim=2)
        projection_finite = torch.stack(
            [
                projections[node_id].finite_mask
                for node_id in self._path_t_by_node
            ],
            dim=1,
        ).all(dim=1)
        eligible = (
            torch.isfinite(clean).flatten(1).all(dim=1)
            & (chain_nonfinite == 0).all(dim=1)
            & torch.isfinite(corrupt).flatten(1).all(dim=1)
            & corrupt_finite
            & clean_decode_finite
            & projection_finite
            & decoded_paths_finite.all(dim=1)
            & corrected_finite.all(dim=1)
        )
        clean_residual_rms = (
            (clean - clean_decoded).flatten(1).pow(2).mean(dim=1).sqrt()
        )
        corrupt_residual_rms = (
            (corrupt - corrupt_decoded).flatten(1).pow(2).mean(dim=1).sqrt()
        )
        path_waveforms = MappingProxyType(
            {
                node_id: corrected_paths[:, index].contiguous()
                for index, node_id in enumerate(self._path_t_by_node)
            }
        )
        return PairedPathBatchCache(
            sample_ids=sample_ids,
            endpoint_seed=int(endpoint_seed),
            corruption_seed=int(corruption_seed),
            mixing_seed=int(mixing_seed),
            clean_raw=clean,
            corrupt_raw=corrupt,
            path_waveforms=path_waveforms,
            projections=MappingProxyType(projections),
            eligible_mask=eligible.contiguous(),
            chain_weights=chain_weights,
            chain_depths=depths,
            chain_operator_mask=operator_mask,
            chain_output_nonfinite_count=chain_nonfinite,
            clean_residual_rms=clean_residual_rms,
            corrupt_residual_rms=corrupt_residual_rms,
        )

    def _resource_bridge(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("resource_bridge requires one clean WaveformView")
        source = inputs[0]
        context.record_diagnostic("identity_only", 1.0)
        return WaveformView(
            name=context.node_id,
            waveform=source.waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="sandbox_resource_requirement_identity_bridge",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "waveform_mutated": False,
                    "purpose": "compile_encoder_decoder_requirements_only",
                    "paired_path_runtime_sha256": self._runtime_sha256,
                },
            ),
            metadata={"resource_bridge_identity": True},
        )

    def _path_view(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if context.node_id not in self._path_node_ids:
            raise KeyError(f"unknown paired path node: {context.node_id}")
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("paired path node requires one WaveformView")
        if self._shared_cache is None:
            raise RuntimeError("paired path batch cache is unavailable")
        source = inputs[0]
        cache = self._shared_cache
        if source.sample_ids != cache.sample_ids:
            raise RuntimeError("paired path cache/sample order mismatch")
        candidate = cache.path_waveforms[context.node_id]
        quality, quality_reasons = _quality_mask(
            candidate,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
        )
        accepted = (
            cache.eligible_mask
            & quality
            & source.valid_mask.to(device=quality.device)
        )
        waveform = torch.where(
            accepted[:, None, None], candidate, cache.clean_raw
        ).contiguous()
        projection = cache.projections[context.node_id]

        def accepted_mean(value: torch.Tensor) -> float:
            subset = value[accepted]
            return 0.0 if int(subset.numel()) == 0 else float(subset.mean().detach().cpu())

        diagnostics = {
            "configured_t": float(projection.configured_t),
            "effective_t_mean": accepted_mean(projection.effective_t),
            "pair_radius_rms_mean": accepted_mean(projection.pair_radius_rms),
            "clean_distance_rms_mean": accepted_mean(projection.clean_distance_rms),
            "corrupt_distance_rms_mean": accepted_mean(
                projection.corrupt_distance_rms
            ),
            "clean_distance_ratio_mean": accepted_mean(
                projection.clean_distance_ratio
            ),
            "corrupt_distance_ratio_mean": accepted_mean(
                projection.corrupt_distance_ratio
            ),
            "paired_direction_cosine_mean": accepted_mean(
                projection.direction_cosine
            ),
            "chain_depth_mean": accepted_mean(cache.chain_depths.float().mean(dim=1)),
            "clean_residual_rms_mean": accepted_mean(cache.clean_residual_rms),
            "corrupt_residual_rms_mean": accepted_mean(cache.corrupt_residual_rms),
            "eligible_fraction": float(cache.eligible_mask.float().mean().detach().cpu()),
            "accepted_fraction": float(accepted.float().mean().detach().cpu()),
            "absolute_radius_cap_applied": 0.0,
            "decode_chunk_size": float(DECODE_CHUNK_SIZE),
        }
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)

        accepted_positions = _positions(accepted)
        ineligible_positions = _positions(~cache.eligible_mask)
        rejected: list[dict[str, str]] = []
        for position, reason in enumerate(quality_reasons):
            if bool(cache.eligible_mask[position].item()) and not bool(
                accepted[position].item()
            ):
                rejected.append(
                    {
                        "hash_id": cache.sample_ids[position],
                        "reason": reason if reason != "accepted" else "source_invalid",
                    }
                )
        return WaveformView(
            name=context.node_id,
            waveform=waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=accepted,
            provenance=Provenance(
                node_id=context.node_id,
                operation="paired_augmix_standardized_latent_path_decode",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "method_id": self.method.profile_name,
                    "paired_path_runtime_path": str(self._runtime_path),
                    "paired_path_runtime_sha256": self._runtime_sha256,
                    "endpoint_construction": (
                        "three_independent_depth123_corruption_chains_then_"
                        "record_local_dirichlet_alpha_1_waveform_mix"
                    ),
                    "chain_count": CHAIN_COUNT,
                    "chain_depths": list(CHAIN_DEPTHS),
                    "chain_depth_sampling": "uniform_per_chain",
                    "clean_beta_mix": False,
                    "posterior_sample": False,
                    "configured_t": float(projection.configured_t),
                    "all_t_values_by_node": dict(self._path_t_by_node),
                    "standardizer_source": "per_center_train400",
                    "standardizer_identity_sha256": (
                        self.latent_pool.standardizer.identity_sha256
                    ),
                    "geometry": "standardized_paired_ray_lens",
                    "absolute_radius_cap": None,
                    "cross_record_absolute_cap_used": False,
                    "pair_radius_role": "diagnostic_only",
                    "residual_correction": "linear_clean_corrupt_endpoint",
                    "endpoint_seed": cache.endpoint_seed,
                    "corruption_seed": cache.corruption_seed,
                    "mixing_seed": cache.mixing_seed,
                    "seed_config_sha256": self._seed_config_sha256,
                    "augmix_config_sha256": self._augmix_config_sha256,
                    "decode_chunk_size": DECODE_CHUNK_SIZE,
                    "quality_failure_policy": "clean_fallback_and_invalid_view_mask",
                    "objective_global_scale": OBJECTIVE_GLOBAL_SCALE,
                },
            ),
            metadata={
                "candidate_eligible_positions": _positions(cache.eligible_mask),
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": tuple(
                    cache.sample_ids[position] for position in ineligible_positions
                ),
                "quality_rejected": tuple(rejected),
                "diagnostic_means": MappingProxyType(dict(diagnostics)),
                "shared_endpoint_identity": {
                    "endpoint_seed": cache.endpoint_seed,
                    "corruption_seed": cache.corruption_seed,
                    "mixing_seed": cache.mixing_seed,
                    "standardizer_sha256": (
                        self.latent_pool.standardizer.identity_sha256
                    ),
                    "paired_path_runtime_sha256": self._runtime_sha256,
                },
                "chain_depths": cache.chain_depths.detach().cpu(),
                "chain_weights": cache.chain_weights.detach().cpu(),
            },
        )

    def _dispatch_augmix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if context.node_id == "resource_bridge":
            return self._resource_bridge(context, inputs)
        return super()._dispatch_augmix(context, inputs)

    def _lhat_attack(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if context.node_id in self._path_node_ids:
            return self._path_view(context, inputs)
        return super()._lhat_attack(context, inputs)

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
    ):
        if composition_indices is not None:
            raise ValueError("paired path does not accept fixed20 exposures")
        if self._shared_cache is not None or self._generation_state is not None:
            raise RuntimeError("paired path runtime generation is not re-entrant")
        resolved_identity = tuple(str(value) for value in rng_identity)
        endpoint_seed = _derive_seed(
            int(base_seed), resolved_identity, "shared_paired_endpoint"
        )
        corruption_seed = _derive_seed(
            endpoint_seed, resolved_identity, "corruption_chains"
        )
        mixing_seed = _derive_seed(
            endpoint_seed, resolved_identity, "dirichlet_weights"
        )
        self._generation_state = (int(base_seed), resolved_identity)
        try:
            assert self.encoder is not None and self.decoder is not None
            with _frozen_vae_components(self.encoder, self.decoder):
                self._shared_cache = self._build_batch_cache(
                    clean_raw,
                    targets,
                    hash_ids,
                    endpoint_seed=endpoint_seed,
                    corruption_seed=corruption_seed,
                    mixing_seed=mixing_seed,
                )
                return super().generate(
                    clean_raw=clean_raw,
                    targets=targets,
                    hash_ids=hash_ids,
                    classifier=classifier,
                    base_seed=base_seed,
                    rng_identity=resolved_identity,
                    composition_indices=None,
                )
        finally:
            self._shared_cache = None
            self._generation_state = None


def _is_paired_method(method: CompiledMethod) -> bool:
    return method.profile_name in SUPPORTED_METHOD_IDS and isinstance(
        method.contracts.get("paired_path"), Mapping
    )


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Patch factory and declared 0.5 objective scale for this method only."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_compute_objective = online_trainer._compute_objective

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
        if not _is_paired_method(method):
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
        return PairedAugMixLatentPathRuntime(
            method,
            model_name=model_name,
            config_root=Path(config_root).expanduser().resolve(),
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )

    def patched_compute_objective(**kwargs: Any):
        result = original_compute_objective(**kwargs)
        method = kwargs.get("method")
        if not isinstance(method, CompiledMethod) or not _is_paired_method(method):
            return result
        batch_type = type(result)
        return batch_type(
            total=result.total * OBJECTIVE_GLOBAL_SCALE,
            raw_terms=dict(result.raw_terms),
            weighted_terms={
                name: value * OBJECTIVE_GLOBAL_SCALE
                for name, value in result.weighted_terms.items()
            },
            valid_counts=dict(result.valid_counts),
        )

    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_compute_objective
    try:
        yield
    finally:
        online_trainer._compute_objective = original_compute_objective
        online_trainer.build_method_runtime = original_factory


install_runtime_adapter = patch_online_trainer_runtime


__all__ = [
    "CHAIN_COUNT",
    "CHAIN_DEPTHS",
    "DECODE_CHUNK_SIZE",
    "METHOD_ID",
    "OBJECTIVE_GLOBAL_SCALE",
    "PATH_NODE_IDS",
    "PATH_T_BY_NODE",
    "STRONG_ENDPOINT_METHOD_ID",
    "SUPPORTED_METHOD_IDS",
    "PairedAugMixLatentPathRuntime",
    "PairedPathBatchCache",
    "PairedPathContract",
    "PairedPathProjection",
    "apply_linear_endpoint_residual_correction",
    "install_runtime_adapter",
    "mix_three_corruption_chains",
    "patch_online_trainer_runtime",
    "project_paired_latent_path",
]
