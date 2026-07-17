"""Managed adapters from typed method nodes to the ECG implementations.

The graph compiler owns topology and static contracts.  This module is the
only bridge from those symbolic nodes to the existing corruption, LHAT and
AugMix engines.  It returns canonical raw-mV 100 Hz views; model-domain
resampling, normalization, objectives and optimizer steps remain trainer-owned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.augmix import AugMixConfig, generate_three_chain_augmix, load_augmix_config
from core.corruption import generate_canonical_corruption
from core.lhat import LHATConfig, generate_lhat_adversarial, load_lhat_config
from core.methods.contracts import (
    BASE_VIEW_NAME,
    NodeContext,
    Provenance,
    ViewBundle,
    ViewValue,
    WaveformView,
)
from core.methods.executor import ExecutionResources, execute_method
from core.methods.registry import CompiledMethod
from util.augmentations.profile import AugmentationProfile, load_augmentation_profile
from util.config_bundle import resolve_config_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positions(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in torch.nonzero(mask, as_tuple=False).flatten())


def _quality_mask(
    waveform_raw: torch.Tensor,
    *,
    minimum_std_mV: float,
    maximum_abs_mV: float,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    flat = waveform_raw.flatten(1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    standard_deviation = safe.std(dim=1, correction=0)
    maximum_absolute = safe.abs().amax(dim=1)
    accepted = finite & (standard_deviation >= float(minimum_std_mV)) & (
        maximum_absolute <= float(maximum_abs_mV)
    )
    summary = torch.stack(
        (
            finite.float(),
            standard_deviation,
            maximum_absolute,
            accepted.float(),
        ),
        dim=1,
    ).detach().cpu().tolist()
    reasons: list[str] = []
    for finite_value, std_value, maximum_value, accepted_value in summary:
        if bool(accepted_value):
            reasons.append("accepted")
            continue
        failures: list[str] = []
        if not bool(finite_value):
            failures.append("nonfinite")
        if float(std_value) < float(minimum_std_mV):
            failures.append("flatline")
        if float(maximum_value) > float(maximum_abs_mV):
            failures.append("severe_amplitude")
        reasons.append("+".join(failures))
    return accepted, tuple(reasons)


@dataclass(frozen=True)
class GeneratedMethodBatch:
    """Typed views plus record-level method accounting for one base batch."""

    bundle: ViewBundle
    diagnostic_weights: Mapping[str, int]
    candidate_eligible_positions: tuple[int, ...]
    accepted_positions: tuple[int, ...]
    ineligible_hash_ids: tuple[str, ...]
    quality_rejected: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        weights = dict(self.diagnostic_weights)
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
            for name, weight in weights.items()
        ):
            raise ValueError(
                "diagnostic_weights must map non-empty names to positive integers"
            )
        object.__setattr__(self, "diagnostic_weights", MappingProxyType(weights))

    @property
    def batch_size(self) -> int:
        clean = self.bundle.require(BASE_VIEW_NAME)
        assert isinstance(clean, WaveformView)
        return clean.batch_size

    def probe_views(self, batch_position: int) -> dict[str, torch.Tensor]:
        """Return only valid method views for one raw-mV visualization probe."""

        if not 0 <= int(batch_position) < self.batch_size:
            raise IndexError("probe batch_position is out of range")
        result: dict[str, torch.Tensor] = {}
        for output_name, value in self.bundle.values.items():
            if not isinstance(value, WaveformView):
                continue
            if not bool(value.valid_mask[int(batch_position)].item()):
                continue
            name = output_name.removesuffix("_view")
            result[name] = value.waveform[int(batch_position)]
        if "clean" not in result:
            raise RuntimeError("method batch has no valid clean probe view")
        return result


class MethodViewRuntime:
    """One compiled profile with resolved, code-owned node adapters."""

    def __init__(
        self,
        method: CompiledMethod,
        *,
        model_name: str,
        config_root: Path,
        latent_pool: Any | None,
        decoder: nn.Module | None,
        minimum_std_mV: float,
        maximum_abs_mV: float,
    ) -> None:
        if not isinstance(method, CompiledMethod):
            raise TypeError("method must be a CompiledMethod")
        if not method.executable:
            raise ValueError(
                f"method profile {method.profile_name!r} is not executable"
            )
        if method.source_path is None:
            raise ValueError("managed method runtime requires a file-backed profile")
        self.method = method
        self.model_name = str(model_name)
        self.config_root = Path(config_root).expanduser().resolve()
        self.latent_pool = latent_pool
        self.decoder = decoder
        self.minimum_std_mV = float(minimum_std_mV)
        self.maximum_abs_mV = float(maximum_abs_mV)
        if self.minimum_std_mV <= 0.0 or self.maximum_abs_mV <= 0.0:
            raise ValueError("method quality thresholds must be positive")

        self._resource_paths: dict[str, Path] = {}
        for name, raw_resource in method.resources.items():
            if not isinstance(raw_resource, Mapping):
                raise ValueError(f"method resource {name!r} must be a mapping")
            if raw_resource.get("type") != "config_reference":
                continue
            self._resource_paths[name] = resolve_config_reference(
                raw_resource.get("path"),
                owner_config_path=method.source_path,
                config_root=self.config_root,
                description=f"method.resources.{name}",
                must_exist=True,
            )

        self.operator_profile: AugmentationProfile | None = None
        operator_resource = method.resources.get("operator_profile")
        if isinstance(operator_resource, Mapping):
            operator_path = self._resource_paths.get("operator_profile")
            if operator_path is None:
                raise ValueError("operator_profile must be a config_reference")
            self.operator_profile = load_augmentation_profile(
                operator_path,
                profile_name=str(operator_resource.get("profile", "")),
                severity=int(operator_resource.get("severity", 0)),
                config_root=self.config_root,
            )

        self.lhat_config: LHATConfig | None = None
        if "lhat_config" in self._resource_paths:
            self.lhat_config = load_lhat_config(
                self._resource_paths["lhat_config"],
                config_root=self.config_root,
            )
        self.augmix_config: AugMixConfig | None = None
        if "augmix_config" in self._resource_paths:
            self.augmix_config = load_augmix_config(
                self._resource_paths["augmix_config"],
                config_root=self.config_root,
            )

        if method.requirements.latent_pool and latent_pool is None:
            raise ValueError("method requires a train-only latent_pool")
        if method.requirements.vae_decoder and decoder is None:
            raise ValueError("method requires a frozen VAE decoder")
        if not method.requirements.latent_pool and latent_pool is not None:
            raise ValueError("method without latent_pool requirement may not consume one")
        if not method.requirements.vae_decoder and decoder is not None:
            raise ValueError("method without VAE decoder requirement may not consume one")

    @property
    def requires_latent_pool(self) -> bool:
        return bool(self.method.requirements.latent_pool)

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method.describe(),
            "model_name": self.model_name,
            "resource_configs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in self._resource_paths.items()
            },
            "operator_profile": (
                None
                if self.operator_profile is None
                else self.operator_profile.describe()
            ),
            "quality_gate": {
                "minimum_global_std_mV": self.minimum_std_mV,
                "maximum_absolute_mV": self.maximum_abs_mV,
                "rejection_policy": "clean_loss_only",
            },
        }

    def _canonical_corruption(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
        *,
        composition_indices: torch.Tensor | None = None,
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("canonical corruption requires one WaveformView")
        if self.operator_profile is None:
            raise RuntimeError("canonical corruption has no resolved operator profile")
        source = inputs[0]
        if composition_indices is not None:
            if not isinstance(composition_indices, torch.Tensor):
                raise TypeError("composition_indices must be a torch.Tensor or None")
            if composition_indices.device != source.waveform.device:
                raise ValueError(
                    "composition_indices device must match the source waveform device"
                )
            if (
                composition_indices.ndim != 1
                or composition_indices.shape[0] != source.batch_size
            ):
                raise ValueError("composition_indices must have shape (B,)")
            if composition_indices.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError("composition_indices must use an integer dtype")
            identity_mask = composition_indices == -1
            if bool(identity_mask.all().item()):
                context.record_diagnostic("mean_depth", 0.0)
                context.record_diagnostic("repaired_nonfinite_count", 0)
                return WaveformView(
                    name=context.node_id,
                    waveform=source.waveform,
                    labels=source.labels,
                    sample_ids=source.sample_ids,
                    valid_mask=source.valid_mask,
                    provenance=Provenance(
                        node_id=context.node_id,
                        operation=context.node_type,
                        parent_names=(source.name,),
                        rng_namespace=context.rng_namespace,
                        parameters={
                            "operator_profile": self.operator_profile.profile_name,
                            "severity": self.operator_profile.severity,
                            "operator_domain_sampling_rate_hz": 500,
                            "composition_selection": "clean_identity_sentinel",
                            "composition_index": -1,
                        },
                    ),
                    metadata={
                        "corruption_diagnostics": None,
                        "composition_indices": composition_indices.contiguous(),
                        "exposure_kind": "clean_identity",
                    },
                )
            if bool(identity_mask.any().item()):
                raise ValueError(
                    "composition_indices may be all -1 for an identity exposure, "
                    "or all values must be in [0, 19]"
                )
        generator = context.torch_generator(
            "composition_and_operators", device=source.waveform.device
        )
        parameters = {
            name: self.operator_profile.parameters_for(name)
            for name in self.operator_profile.canonical_order
        }
        result = generate_canonical_corruption(
            source.waveform,
            operator_params=parameters,
            generator=generator,
            composition_indices=composition_indices,
            # WaveformView construction immediately above has already enforced
            # the finite canonical raw-mV contract. Avoid a second device-wide
            # finite reduction and host synchronization for every online view.
            _input_prevalidated=True,
        )
        valid = source.valid_mask & (result.diagnostics.output_nonfinite_count == 0)
        context.record_diagnostic(
            "mean_depth", float(result.diagnostics.depth.float().mean().detach().cpu())
        )
        context.record_diagnostic(
            "repaired_nonfinite_count",
            int(result.diagnostics.output_nonfinite_count.sum().detach().cpu()),
        )
        return WaveformView(
            name=context.node_id,
            waveform=result.waveform_raw_100hz,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=valid,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "operator_profile": self.operator_profile.profile_name,
                    "severity": self.operator_profile.severity,
                    "operator_domain_sampling_rate_hz": 500,
                    "composition_selection": (
                        "random" if composition_indices is None else "forced"
                    ),
                },
            ),
            metadata={"corruption_diagnostics": result.diagnostics},
        )

    def _lhat_attack(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("LHAT requires one clean WaveformView")
        if self.latent_pool is None or self.decoder is None or self.lhat_config is None:
            raise RuntimeError("LHAT runtime resources are incomplete")
        source = inputs[0]
        classifier = context.resource("classifier")
        pool = context.resource("latent_pool")
        try:
            all_indices = pool.indices_for_hashes(source.sample_ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "method batch contains a hash outside the bound latent pool"
            ) from exc
        pool_labels = pool.labels.index_select(0, all_indices).to(
            source.labels.device, dtype=torch.float32
        )
        if not torch.equal(pool_labels, source.labels):
            raise RuntimeError("method batch labels differ from latent-pool labels")

        eligible_universe = pool.eligible_hash_id_set
        candidate_positions = tuple(
            index
            for index, hash_id in enumerate(source.sample_ids)
            if hash_id in eligible_universe
        )
        ineligible = tuple(
            hash_id
            for hash_id in source.sample_ids
            if hash_id not in eligible_universe
        )
        full_waveform = source.waveform.clone()
        full_valid = torch.zeros(
            source.batch_size, device=source.waveform.device, dtype=torch.bool
        )
        rejected: list[dict[str, str]] = []
        diagnostic_means: dict[str, float] = {}
        accepted_positions: tuple[int, ...] = ()

        if candidate_positions:
            candidate_index = torch.as_tensor(
                candidate_positions, device=source.waveform.device, dtype=torch.long
            )
            candidate_hashes = tuple(source.sample_ids[index] for index in candidate_positions)
            generator = context.torch_generator(
                "candidate_selection", device=source.waveform.device
            )
            attack_batch = pool.get_attack_batch_by_hashes(
                candidate_hashes,
                mode=self.lhat_config.candidate_mode,
                local_pool_size=self.lhat_config.local_pool_size,
                generator=generator,
            )
            candidate_targets = source.labels.index_select(0, candidate_index)
            if not torch.equal(
                attack_batch.labels.to(
                    source.labels.device, dtype=source.labels.dtype
                ),
                candidate_targets,
            ):
                raise RuntimeError("latent-pool label binding changed")
            attack = generate_lhat_adversarial(
                classifier=classifier,
                decoder=context.resource("vae_decoder"),
                anchor_standardized=attack_batch.anchor_standardized.to(
                    source.waveform.device, dtype=torch.float32
                ),
                candidates_standardized=attack_batch.candidates_standardized.to(
                    source.waveform.device, dtype=torch.float32
                ),
                targets=candidate_targets,
                standardizer=pool.standardizer,
                model_name=self.model_name,
                config=self.lhat_config,
            )
            accepted_local, reasons = _quality_mask(
                attack.waveform_raw,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
            )
            local_positions = _positions(accepted_local)
            accepted_positions = tuple(candidate_positions[index] for index in local_positions)
            if local_positions:
                accepted_local_tensor = torch.as_tensor(
                    local_positions,
                    device=source.waveform.device,
                    dtype=torch.long,
                )
                accepted_batch_tensor = torch.as_tensor(
                    accepted_positions,
                    device=source.waveform.device,
                    dtype=torch.long,
                )
                full_waveform.index_copy_(
                    0,
                    accepted_batch_tensor,
                    attack.waveform_raw.index_select(0, accepted_local_tensor),
                )
                full_valid[accepted_batch_tensor] = True
            for local_index, reason in enumerate(reasons):
                if reason != "accepted":
                    rejected.append(
                        {"hash_id": candidate_hashes[local_index], "reason": reason}
                    )
            diagnostic_means = attack.diagnostics.mean_dict()
            for name, value in diagnostic_means.items():
                context.record_diagnostic(name, float(value))

        context.record_diagnostic("candidate_eligible_count", len(candidate_positions))
        context.record_diagnostic("quality_accepted_count", len(accepted_positions))
        context.record_diagnostic("ineligible_count", len(ineligible))
        context.record_diagnostic("quality_rejected_count", len(rejected))
        return WaveformView(
            name=context.node_id,
            waveform=full_waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=full_valid,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "candidate_mode": self.lhat_config.candidate_mode,
                    "num_candidates": self.lhat_config.num_candidates,
                    "quality_rejection_policy": "clean_loss_only",
                },
            ),
            metadata={
                "candidate_eligible_positions": candidate_positions,
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": ineligible,
                "quality_rejected": tuple(rejected),
                "diagnostic_means": diagnostic_means,
            },
        )

    def _augmix(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if (
            len(inputs) != 2
            or not isinstance(inputs[0], WaveformView)
            or not isinstance(inputs[1], WaveformView)
        ):
            raise TypeError("three-chain AugMix requires clean and LHAT views")
        if self.augmix_config is None:
            raise RuntimeError("AugMix runtime has no resolved config")
        clean, hard = inputs
        if clean.sample_ids != hard.sample_ids or not torch.equal(
            clean.labels, hard.labels
        ):
            raise RuntimeError("AugMix inputs must preserve origin and label alignment")
        accepted_positions = _positions(hard.valid_mask)
        full_waveform = clean.waveform.clone()
        diagnostics: dict[str, float] = {}
        if accepted_positions:
            indices = torch.as_tensor(
                accepted_positions, device=clean.waveform.device, dtype=torch.long
            )
            generator = context.torch_generator(
                "chains_and_mixing", device=clean.waveform.device
            )
            result = generate_three_chain_augmix(
                clean.waveform.index_select(0, indices),
                hard.waveform.index_select(0, indices),
                sampling_rate_hz=100,
                config=self.augmix_config,
                generator=generator,
                # The trainer consumes raw canonical views and applies one
                # shared model-domain z-score afterwards.  Avoid materializing
                # an unused second full waveform here.
                include_normalized=False,
            )
            full_waveform.index_copy_(0, indices, result.mixed_raw)
            names = (
                "chain1_depth",
                "chain2_depth",
                "vae_weight",
                "strength",
            )
            values = torch.stack(
                (
                    result.chain1_depth.float().mean(),
                    result.chain2_depth.float().mean(),
                    result.mixture_weights[:, 2].mean(),
                    result.augmented_strength.mean(),
                )
            ).detach().cpu().tolist()
            diagnostics = dict(zip(names, (float(value) for value in values), strict=True))
            for name, value in diagnostics.items():
                context.record_diagnostic(name, value)
        context.record_diagnostic("accepted_count", len(accepted_positions))
        return WaveformView(
            name=context.node_id,
            waveform=full_waveform,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=hard.valid_mask.clone(),
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(clean.name, hard.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "chain3_source": hard.name,
                    "chain3_additional_corruption": False,
                    "operator_domain_sampling_rate_hz": 500,
                },
            ),
            metadata={
                "accepted_positions": accepted_positions,
                "diagnostic_means": diagnostics,
            },
        )

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
    ) -> GeneratedMethodBatch:
        """Generate all declared raw views before any outer model forward.

        ``composition_indices=None`` preserves the profile's ordinary random
        composition sampling.  A device-local integer tensor in ``[0, 19]``
        forces one canonical depth-2/3 composition per record.  An all-``-1``
        tensor is the explicit clean-identity exposure used by the fixed-20
        direct fine-tuning schedule; mixing ``-1`` with corruptions is rejected.
        """

        clean = WaveformView(
            name="clean_raw",
            waveform=clean_raw,
            labels=targets,
            sample_ids=tuple(str(value) for value in hash_ids),
            provenance=Provenance(
                node_id="batch_source",
                operation="managed_k500_batch",
                parameters={"normalization": "none", "domain": "raw_mV_100Hz"},
            ),
        )
        adapters = {
            "canonical_corruption": lambda context, inputs: self._canonical_corruption(
                context,
                inputs,
                composition_indices=composition_indices,
            ),
            "lhat_attack": self._lhat_attack,
            "augmix": self._augmix,
        }
        bundle = execute_method(
            self.method,
            ExecutionResources(
                sources={"clean_raw": clean},
                adapters=adapters,
                classifier=classifier,
                vae_decoder=self.decoder,
                latent_pool=self.latent_pool,
                base_seed=int(base_seed),
                rng_identity=tuple(str(value) for value in rng_identity),
            ),
        )
        candidate_position_set: set[int] = set()
        accepted_position_set: set[int] = set()
        ineligible_values: list[str] = []
        rejected_values: list[dict[str, str]] = []
        diagnostic_weights: dict[str, int] = {}
        accounting_keys = {
            "candidate_eligible_positions",
            "accepted_positions",
            "ineligible_hash_ids",
            "quality_rejected",
        }
        for value in bundle.values.values():
            if not isinstance(value, WaveformView):
                continue
            metadata = value.metadata
            if accounting_keys <= set(metadata):
                candidate_position_set.update(
                    int(position)
                    for position in metadata["candidate_eligible_positions"]
                )
                accepted_position_set.update(
                    int(position) for position in metadata["accepted_positions"]
                )
                ineligible_values.extend(
                    str(hash_id) for hash_id in metadata["ineligible_hash_ids"]
                )
                rejected_values.extend(
                    dict(item) for item in metadata["quality_rejected"]
                )
            accepted_for_node = metadata.get("accepted_positions")
            if isinstance(accepted_for_node, (tuple, list)):
                node_weight = max(1, len(accepted_for_node))
            else:
                node_weight = max(
                    1, int(value.valid_mask.sum().detach().cpu())
                )
            prefix = f"{value.provenance.node_id}/"
            for diagnostic_name in bundle.diagnostics:
                if diagnostic_name.startswith(prefix):
                    diagnostic_weights[diagnostic_name] = node_weight

        candidate_positions = tuple(sorted(candidate_position_set))
        accepted_positions = tuple(sorted(accepted_position_set))
        ineligible = tuple(dict.fromkeys(ineligible_values))
        rejected = tuple(
            dict(item)
            for item in {
                (entry["hash_id"], entry["reason"]): entry
                for entry in rejected_values
            }.values()
        )
        return GeneratedMethodBatch(
            bundle=bundle,
            diagnostic_weights=diagnostic_weights,
            candidate_eligible_positions=candidate_positions,
            accepted_positions=accepted_positions,
            ineligible_hash_ids=ineligible,
            quality_rejected=rejected,
        )


def build_method_runtime(
    method: CompiledMethod,
    *,
    model_name: str,
    config_root: str | Path,
    latent_pool: Any | None = None,
    decoder: nn.Module | None = None,
    minimum_std_mV: float = 1.0e-4,
    maximum_abs_mV: float = 20.0,
) -> MethodViewRuntime:
    return MethodViewRuntime(
        method,
        model_name=model_name,
        config_root=Path(config_root),
        latent_pool=latent_pool,
        decoder=decoder,
        minimum_std_mV=minimum_std_mV,
        maximum_abs_mV=maximum_abs_mV,
    )


__all__ = [
    "GeneratedMethodBatch",
    "MethodViewRuntime",
    "build_method_runtime",
]
