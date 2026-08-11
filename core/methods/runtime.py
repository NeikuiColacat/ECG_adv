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

from core.augmix import (
    AugMixConfig,
    generate_latent_three_chain_augmix,
    generate_three_chain_augmix,
    load_augmix_config,
)
from core.corruption import generate_canonical_corruption
from core.lhat import (
    LHATConfig,
    contract_lhat_adversarial,
    generate_lhat_adversarial,
    load_lhat_config,
)
from core.methods.contracts import (
    BASE_VIEW_NAME,
    LatentView,
    NodeContext,
    Provenance,
    ViewBundle,
    ViewValue,
    WaveformView,
)
from core.methods.executor import ExecutionResources, execute_method
from core.methods.registry import CompiledMethod
from models.vae import decode_to_ptbxl_waveform, prepare_ecgtwin_encoder_input
from util.augmentations.profile import AugmentationProfile, load_augmentation_profile
from util.config_bundle import resolve_config_reference


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


def _quality_mask(
    waveform_raw: torch.Tensor,
    *,
    minimum_std_mV: float,
    maximum_abs_mV: float,
    return_reasons: bool = True,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    flat = waveform_raw.flatten(1)
    finite = torch.isfinite(flat).all(dim=1)
    safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    standard_deviation = safe.std(dim=1, correction=0)
    maximum_absolute = safe.abs().amax(dim=1)
    accepted = finite & (standard_deviation >= float(minimum_std_mV)) & (
        maximum_absolute <= float(maximum_abs_mV)
    )
    if not return_reasons:
        return accepted, ()
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


def _scoped_lhat_diagnostics(
    attack_diagnostics: Any,
    contract_diagnostics: Any | None,
    accepted_mask: torch.Tensor,
    accepted_count: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, int],
]:
    """Separate candidate-wide attack evidence from accepted training evidence."""

    samples: dict[str, torch.Tensor] = {}
    means: dict[str, torch.Tensor] = {}
    weights: dict[str, int] = {}
    summed = {
        "sample_anyflip_numerator",
        "sample_anyflip_denominator",
        "positive_hide_numerator",
        "positive_hide_denominator",
        "negative_add_numerator",
        "negative_add_denominator",
        "contract_training_anyflip_numerator",
        "contract_training_anyflip_denominator",
    }

    def add(scope: str, diagnostics: Any, count: int, exclude: tuple[str, ...] = ()) -> None:
        local_samples = diagnostics.sample_tensor_dict()
        local_means = diagnostics.mean_tensor_dict()
        for name in exclude:
            local_samples.pop(name, None)
            local_means.pop(name, None)
        samples.update({f"{scope}/{name}": value for name, value in local_samples.items()})
        means.update({f"{scope}/{name}": value for name, value in local_means.items()})
        weights.update(
            {f"{scope}/{name}": 1 if name in summed else count for name in local_means}
        )

    candidate_count = int(accepted_mask.numel())
    add("raw_all_candidate_eligible", attack_diagnostics, candidate_count)
    if contract_diagnostics is not None:
        add("contract_all_candidate_eligible", contract_diagnostics, candidate_count)
        if accepted_count:
            add(
                "contract_training_accepted",
                contract_diagnostics.select(accepted_mask),
                accepted_count,
                ("accepted", "contract_acceptance_rate"),
            )
    return samples, means, weights


@dataclass(frozen=True)
class GeneratedMethodBatch:
    """Typed views plus record-level method accounting for one base batch."""

    bundle: ViewBundle
    diagnostic_weights: Mapping[str, int]
    diagnostic_samples: Mapping[str, torch.Tensor]
    stochastic_trace: Mapping[str, torch.Tensor]
    candidate_eligible_positions: tuple[int, ...]
    accepted_positions: tuple[int, ...]
    quality_view_total_count: int
    quality_view_accepted_count: int
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
        samples = dict(self.diagnostic_samples)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.requires_grad
            for name, value in samples.items()
        ):
            raise ValueError(
                "diagnostic_samples must map names to detached rank-1 tensors"
            )
        object.__setattr__(
            self,
            "diagnostic_samples",
            MappingProxyType(
                {name: value.detach() for name, value in samples.items()}
            ),
        )
        trace = dict(self.stochastic_trace)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or value.requires_grad
            for name, value in trace.items()
        ):
            raise ValueError(
                "stochastic_trace must map names to detached non-scalar tensors"
            )
        object.__setattr__(
            self,
            "stochastic_trace",
            MappingProxyType({name: value.detach() for name, value in trace.items()}),
        )
        if self.quality_view_total_count < 0:
            raise ValueError("quality_view_total_count must be non-negative")
        if not 0 <= self.quality_view_accepted_count <= self.quality_view_total_count:
            raise ValueError(
                "quality_view_accepted_count must be within the total view count"
            )

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
        encoder: nn.Module | None,
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
        self.encoder = encoder
        self.decoder = decoder
        self.minimum_std_mV = float(minimum_std_mV)
        self.maximum_abs_mV = float(maximum_abs_mV)
        if self.minimum_std_mV <= 0.0 or self.maximum_abs_mV <= 0.0:
            raise ValueError("method quality thresholds must be positive")

        self._resource_paths: dict[str, Path] = {}
        self._rng_seed_paths: dict[str, Path] = {}
        for name, raw_resource in method.resources.items():
            if not isinstance(raw_resource, Mapping):
                raise ValueError(f"method resource {name!r} must be a mapping")
            resource_type = raw_resource.get("type")
            if resource_type == "config_reference":
                self._resource_paths[name] = resolve_config_reference(
                    raw_resource.get("path"),
                    owner_config_path=method.source_path,
                    config_root=self.config_root,
                    description=f"method.resources.{name}",
                    must_exist=True,
                )
            elif resource_type == "isolated_torch_generator":
                self._rng_seed_paths[name] = resolve_config_reference(
                    raw_resource.get("seed_config"),
                    owner_config_path=method.source_path,
                    config_root=self.config_root,
                    description=f"method.resources.{name}.seed_config",
                    must_exist=True,
                )
            elif resource_type in {
                "caller_owned_classifier",
                "none",
                "runtime_frozen_decoder",
                "runtime_frozen_encoder",
                "runtime_train_only_exact_label_pool",
            }:
                pass
            else:
                raise ValueError(
                    f"method resource {name!r} has unsupported type "
                    f"{resource_type!r}"
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
        if method.requirements.vae_encoder and encoder is None:
            raise ValueError("method requires a frozen VAE encoder")
        if not method.requirements.latent_pool and latent_pool is not None:
            raise ValueError("method without latent_pool requirement may not consume one")
        if not method.requirements.vae_decoder and decoder is not None:
            raise ValueError("method without VAE decoder requirement may not consume one")
        if not method.requirements.vae_encoder and encoder is not None:
            raise ValueError("method without VAE encoder requirement may not consume one")

    @property
    def requires_latent_pool(self) -> bool:
        return bool(self.method.requirements.latent_pool)

    @property
    def rng_seed_config_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(dict(self._rng_seed_paths))

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method.describe(),
            "model_name": self.model_name,
            "resource_configs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in self._resource_paths.items()
            },
            "rng_seed_configs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in self._rng_seed_paths.items()
            },
            "operator_profile": (
                None
                if self.operator_profile is None
                else self.operator_profile.describe()
            ),
            "augmix": (
                None
                if self.augmix_config is None
                else {
                    "config_path": str(self.augmix_config.config_path),
                    "config_sha256": _sha256(self.augmix_config.config_path),
                    "random_seed_config_path": str(
                        self.augmix_config.random_seed_config_path
                    ),
                    "random_seed_config_sha256": _sha256(
                        self.augmix_config.random_seed_config_path
                    ),
                    "operator_profile": (
                        self.augmix_config.operator_profile_config.describe()
                    ),
                }
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
        composition_index_hint: int | None = None,
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
            if composition_index_hint is not None:
                if isinstance(composition_index_hint, bool) or not isinstance(
                    composition_index_hint, int
                ):
                    raise TypeError("composition_index_hint must be an integer")
                if composition_index_hint < -1 or composition_index_hint > 19:
                    raise ValueError(
                        "composition_index_hint must be -1 or lie in [0,19]"
                    )
                all_identity = composition_index_hint == -1
                any_identity = all_identity
            else:
                identity_mask = composition_indices == -1
                identity_state = (
                    torch.stack((identity_mask.all(), identity_mask.any()))
                    .detach()
                    .cpu()
                    .tolist()
                )
                all_identity = bool(identity_state[0])
                any_identity = bool(identity_state[1])
            if all_identity:
                context.record_diagnostic(
                    "mean_depth", source.waveform.new_zeros(())
                )
                context.record_diagnostic(
                    "repaired_nonfinite_count", source.waveform.new_zeros(())
                )
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
                        "diagnostic_weight": source.batch_size,
                        "stochastic_trace": {
                            "composition_index": composition_indices.detach().contiguous(),
                        },
                    },
                )
            if any_identity:
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
            "mean_depth", result.diagnostics.depth.float().mean().detach()
        )
        context.record_diagnostic(
            "repaired_nonfinite_count",
            result.diagnostics.output_nonfinite_count.float().sum().detach(),
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
            metadata={
                "corruption_diagnostics": result.diagnostics,
                "diagnostic_weight": source.batch_size,
                "stochastic_trace": {
                    "composition_index": (
                        result.diagnostics.composition_index.detach().contiguous()
                    ),
                    "depth": result.diagnostics.depth.detach().contiguous(),
                    "operator_mask": (
                        result.diagnostics.operator_mask.detach().contiguous()
                    ),
                },
            },
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
        attack_then_contract = (
            context.node_type == "vae_lhat_attack_then_contract_view"
        )
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
        full_anchor_reconstruction = source.waveform.clone()
        full_valid = torch.zeros(
            source.batch_size, device=source.waveform.device, dtype=torch.bool
        )
        rejected: list[dict[str, str]] = []
        diagnostic_means: dict[str, torch.Tensor] = {}
        diagnostic_samples: dict[str, torch.Tensor] = {}
        stochastic_trace: dict[str, torch.Tensor] = {}
        local_diagnostic_weights: dict[str, int] = {}
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
                raw_clean_waveform=source.waveform.index_select(
                    0, candidate_index
                ),
                targets=candidate_targets,
                standardizer=pool.standardizer,
                model_name=self.model_name,
                config=self.lhat_config,
            )
            training_waveform = attack.waveform_raw
            contract_result = None
            if attack_then_contract:
                contract_result = contract_lhat_adversarial(
                    classifier=classifier,
                    decoder=context.resource("vae_decoder"),
                    attack=attack,
                    raw_clean_waveform=source.waveform.index_select(
                        0, candidate_index
                    ),
                    targets=candidate_targets,
                    standardizer=pool.standardizer,
                    model_name=self.model_name,
                    minimum_std_mV=self.minimum_std_mV,
                    maximum_abs_mV=self.maximum_abs_mV,
                    config=self.lhat_config,
                )
                training_waveform = contract_result.waveform_raw
            full_anchor_reconstruction.index_copy_(
                0,
                candidate_index,
                attack.anchor_waveform_raw,
            )
            accepted_local, reasons = _quality_mask(
                training_waveform,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
            )
            if contract_result is not None:
                accepted_local &= contract_result.valid_mask
                contract_acceptance = (
                    contract_result.valid_mask.detach().cpu().tolist()
                )
                reasons = tuple(
                    reason
                    if bool(contract_acceptance[index])
                    else "contract_rejected"
                    for index, reason in enumerate(reasons)
                )
            stochastic_trace = {
                "candidate_batch_positions": candidate_index.detach().contiguous(),
                "candidate_pool_indices": (
                    attack_batch.candidate_pool_indices.detach().contiguous()
                ),
                "final_hull_weights": attack.weights.detach().contiguous(),
                "quality_accepted_mask": accepted_local.detach().contiguous(),
            }
            if contract_result is not None:
                stochastic_trace.update(
                    {
                        "contract_accepted_mask": (
                            contract_result.valid_mask.detach().contiguous()
                        ),
                        "contract_selected_t": (
                            contract_result.diagnostics.selected_t.detach().contiguous()
                        ),
                    }
                )
            # ``_quality_mask`` already materialized one compact per-record
            # summary to construct the audit reasons. Reuse that host result
            # instead of synchronizing the accepted mask a second time.
            local_positions = tuple(
                index for index, reason in enumerate(reasons) if reason == "accepted"
            )
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
                    training_waveform.index_select(0, accepted_local_tensor),
                )
                full_valid[accepted_batch_tensor] = True
            diagnostic_samples, diagnostic_means, local_diagnostic_weights = (
                _scoped_lhat_diagnostics(
                    attack.diagnostics,
                    None if contract_result is None else contract_result.diagnostics,
                    accepted_local,
                    len(accepted_positions),
                )
            )
            for name, value in diagnostic_means.items():
                context.record_diagnostic(name, value)
            for local_index, reason in enumerate(reasons):
                if reason != "accepted":
                    rejected.append(
                        {"hash_id": candidate_hashes[local_index], "reason": reason}
                    )
        context.record_diagnostic("candidate_eligible_count", len(candidate_positions))
        context.record_diagnostic("quality_accepted_count", len(accepted_positions))
        context.record_diagnostic("ineligible_count", len(ineligible))
        context.record_diagnostic("quality_rejected_count", len(rejected))
        local_diagnostic_weights.update(
            {
                "candidate_eligible_count": 1,
                "quality_accepted_count": 1,
                "ineligible_count": 1,
                "quality_rejected_count": 1,
            }
        )
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
                    "attack_then_contract": attack_then_contract,
                    "quality_rejection_policy": "clean_loss_only",
                },
            ),
            metadata={
                "candidate_eligible_positions": candidate_positions,
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": ineligible,
                "quality_rejected": tuple(rejected),
                "diagnostic_means": diagnostic_means,
                "diagnostic_samples": diagnostic_samples,
                "diagnostic_weights": local_diagnostic_weights,
                "stochastic_trace": stochastic_trace,
                "anchor_waveform_raw": full_anchor_reconstruction.detach().contiguous(),
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
        recorded_positions = hard.metadata.get("accepted_positions")
        accepted_positions = (
            tuple(int(position) for position in recorded_positions)
            if isinstance(recorded_positions, (tuple, list))
            else _positions(hard.valid_mask)
        )
        full_waveform = clean.waveform.clone()
        diagnostics: dict[str, torch.Tensor] = {}
        generator = context.torch_generator(
            "chains_and_mixing", device=clean.waveform.device
        )
        # Draw chain1, chain2 and mixture parameters over the complete ordered
        # base batch before applying the method-specific hard-view validity
        # mask. This keeps common-random-number trajectories aligned even when
        # two methods reject different LHAT endpoints.
        result = generate_three_chain_augmix(
            clean.waveform,
            hard.waveform,
            sampling_rate_hz=100,
            config=self.augmix_config,
            generator=generator,
            # The trainer consumes raw canonical views and applies one shared
            # model-domain z-score afterwards. Avoid an unused normalized copy.
            include_normalized=False,
        )
        if accepted_positions:
            indices = torch.as_tensor(
                accepted_positions, device=clean.waveform.device, dtype=torch.long
            )
            full_waveform.index_copy_(
                0,
                indices,
                result.mixed_raw.index_select(0, indices),
            )
            diagnostics = {
                "chain1_depth": (
                    result.chain1_depth.index_select(0, indices).float().mean().detach()
                ),
                "chain2_depth": (
                    result.chain2_depth.index_select(0, indices).float().mean().detach()
                ),
                "vae_weight": (
                    result.mixture_weights.index_select(0, indices)[:, 2]
                    .mean()
                    .detach()
                ),
                "strength": (
                    result.augmented_strength.index_select(0, indices)
                    .mean()
                    .detach()
                ),
            }
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
                    "common_random_numbers_before_quality_mask": True,
                    "operator_domain_sampling_rate_hz": 500,
                },
            ),
            metadata={
                "accepted_positions": accepted_positions,
                "diagnostic_means": diagnostics,
                "stochastic_trace": {
                    "chain1_composition_index": (
                        result.chain1_composition_index.detach().contiguous()
                    ),
                    "chain2_composition_index": (
                        result.chain2_composition_index.detach().contiguous()
                    ),
                    "chain1_depth": result.chain1_depth.detach().contiguous(),
                    "chain2_depth": result.chain2_depth.detach().contiguous(),
                    "chain1_operator_mask": (
                        result.chain1_operator_mask.detach().contiguous()
                    ),
                    "chain2_operator_mask": (
                        result.chain2_operator_mask.detach().contiguous()
                    ),
                    "mixture_weights": result.mixture_weights.detach().contiguous(),
                    "augmented_strength": (
                        result.augmented_strength.detach().contiguous()
                    ),
                },
            },
        )

    def _latent_threechain_augmix(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("latent three-chain AugMix requires one clean view")
        if self.encoder is None or self.decoder is None or self.augmix_config is None:
            raise RuntimeError("latent three-chain AugMix resources are incomplete")
        source = inputs[0]
        generator = context.torch_generator(
            "depth_operators_and_latent_mixing",
            device=source.waveform.device,
        )
        result = generate_latent_three_chain_augmix(
            source.waveform,
            encoder=context.resource("vae_encoder"),
            decoder=context.resource("vae_decoder"),
            sampling_rate_hz=100,
            config=self.augmix_config,
            generator=generator,
        )
        accepted, reasons = _quality_mask(
            result.mixed_raw,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
        )
        repaired = result.chain_output_nonfinite_count.sum(dim=1) > 0
        accepted &= ~repaired
        accepted_positions = _positions(accepted)
        full_waveform = source.waveform.clone()
        if accepted_positions:
            positions = torch.as_tensor(
                accepted_positions,
                device=source.waveform.device,
                dtype=torch.long,
            )
            full_waveform.index_copy_(
                0,
                positions,
                result.mixed_raw.index_select(0, positions),
            )
        repaired_cpu = repaired.detach().cpu().tolist()
        rejected: list[dict[str, str]] = []
        for index, quality_reason in enumerate(reasons):
            failures: list[str] = []
            if bool(repaired_cpu[index]):
                failures.append("chain_nonfinite_repaired")
            if quality_reason != "accepted":
                failures.append(quality_reason)
            if failures:
                rejected.append(
                    {
                        "node_id": context.node_id,
                        "hash_id": source.sample_ids[index],
                        "reason": "+".join(failures),
                    }
                )
        diagnostic_values = torch.stack(
            (
                result.chain_depths.float().mean(),
                result.mixture_weights[:, 0].mean(),
                result.mixture_weights[:, 1].mean(),
                result.mixture_weights[:, 2].mean(),
                result.augmented_strength.mean(),
                result.residual_rms_ratio.mean(),
                result.chain_output_nonfinite_count.float().sum(),
            )
        ).detach().cpu().tolist()
        diagnostics = dict(
            zip(
                (
                    "mean_chain_depth",
                    "latent_weight_0",
                    "latent_weight_1",
                    "latent_weight_2",
                    "augmented_strength",
                    "residual_rms_ratio",
                    "repaired_nonfinite_count",
                ),
                (float(value) for value in diagnostic_values),
                strict=True,
            )
        )
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        context.record_diagnostic("quality_accepted_count", len(accepted_positions))
        context.record_diagnostic("quality_rejected_count", len(rejected))
        return WaveformView(
            name=context.node_id,
            waveform=full_waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask & accepted,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "width": self.augmix_config.latent_threechain_width,
                    "depths": list(self.augmix_config.latent_threechain_depths),
                    "dirichlet_alpha": (
                        self.augmix_config.latent_threechain_dirichlet_alpha
                    ),
                    "operator_sampling": "random_subset_without_replacement",
                    "operator_application_order": "canonical_order",
                    "posterior_sample": False,
                    "reconstruction_residual_bypass": (
                        self.augmix_config.latent_threechain_reconstruction_residual_bypass
                    ),
                    "post_decode_clean_beta_mix": (
                        self.augmix_config.latent_threechain_post_decode_clean_beta_mix
                    ),
                    "beta_alpha": self.augmix_config.beta_alpha,
                    "operator_domain_sampling_rate_hz": 500,
                },
            ),
            metadata={
                "candidate_eligible_positions": tuple(range(source.batch_size)),
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": (),
                "quality_rejected": tuple(rejected),
                "diagnostic_means": diagnostics,
            },
        )

    def _dispatch_augmix(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if context.node_type == "latent_threechain_augmix_view":
            return self._latent_threechain_augmix(context, inputs)
        return self._augmix(context, inputs)

    def _vae_encode(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("VAE reconstruction encoder requires one clean waveform")
        source = inputs[0]
        encoder = context.resource("vae_encoder")
        identity = getattr(encoder, "checkpoint_identity", None)
        describe = getattr(identity, "describe", None)
        payload = describe() if callable(describe) else None
        if not isinstance(payload, Mapping) or len(str(payload.get("sha256", ""))) != 64:
            raise ValueError("managed VAE encoder has no valid checkpoint identity")
        with torch.no_grad():
            encoder_input = prepare_ecgtwin_encoder_input(source.waveform)
            latent, _, _ = encoder(encoder_input, sample=False)
        return LatentView(
            name=context.node_id,
            latent=latent.detach().contiguous(),
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask,
            encoder_identity=str(payload["sha256"]),
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(source.name,),
                parameters={"posterior": "deterministic_mean"},
            ),
        )

    def _vae_decode(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], LatentView):
            raise TypeError("VAE reconstruction decoder requires one latent view")
        source = inputs[0]
        with torch.no_grad():
            waveform = decode_to_ptbxl_waveform(
                context.resource("vae_decoder"),
                source.latent,
                target_points=1000,
            ).detach().contiguous()
        accepted, reasons = _quality_mask(
            waveform,
            minimum_std_mV=self.minimum_std_mV,
            maximum_abs_mV=self.maximum_abs_mV,
        )
        accepted_positions = _positions(accepted)
        rejected = tuple(
            {
                "node_id": context.node_id,
                "hash_id": source.sample_ids[index],
                "reason": reason,
            }
            for index, reason in enumerate(reasons)
            if reason != "accepted"
        )
        context.record_diagnostic("quality_accepted_count", len(accepted_positions))
        context.record_diagnostic("quality_rejected_count", len(rejected))
        return WaveformView(
            name=context.node_id,
            waveform=waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask & accepted,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(source.name,),
                parameters={
                    "target_points": 1000,
                    "lead_order": "ptbxl",
                    "quality_rejection_policy": "clean_loss_only",
                },
            ),
            metadata={
                "candidate_eligible_positions": tuple(range(source.batch_size)),
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": (),
                "quality_rejected": rejected,
                "diagnostic_weights": {
                    "quality_accepted_count": 1,
                    "quality_rejected_count": 1,
                },
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
        composition_index_hint: int | None = None,
        objective_term_names: Sequence[str] | None = None,
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

        def canonical_corruption_adapter(
            context: NodeContext, inputs: tuple[ViewValue, ...]
        ) -> ViewValue:
            kwargs: dict[str, Any] = {
                "composition_indices": composition_indices,
            }
            if composition_index_hint is not None:
                kwargs["composition_index_hint"] = composition_index_hint
            return self._canonical_corruption(context, inputs, **kwargs)

        adapters = {
            "canonical_corruption": canonical_corruption_adapter,
            "lhat_attack": self._lhat_attack,
            "augmix": self._dispatch_augmix,
            "vae_encode": self._vae_encode,
            "vae_decode": self._vae_decode,
        }
        selected_terms = (
            self.method.objective.terms
            if objective_term_names is None
            else tuple(
                term
                for term in self.method.objective.terms
                if term.name in set(objective_term_names)
            )
        )
        if objective_term_names is not None:
            requested_names = tuple(str(value) for value in objective_term_names)
            if not requested_names or len(set(requested_names)) != len(requested_names):
                raise ValueError("objective_term_names must be non-empty and unique")
            resolved_names = {term.name for term in selected_terms}
            unknown_names = sorted(set(requested_names) - resolved_names)
            if unknown_names:
                raise ValueError(f"unknown objective term names: {unknown_names}")
        required_outputs = {BASE_VIEW_NAME}
        for term in selected_terms:
            required_outputs.update(term.views)
        bundle = execute_method(
            self.method,
            ExecutionResources(
                sources={"clean_raw": clean},
                adapters=adapters,
                classifier=classifier,
                vae_encoder=self.encoder,
                vae_decoder=self.decoder,
                latent_pool=self.latent_pool,
                base_seed=int(base_seed),
                rng_identity=tuple(str(value) for value in rng_identity),
            ),
            required_outputs=tuple(
                name for name in self.method.outputs if name in required_outputs
            ),
        )
        candidate_position_set: set[int] = set()
        accepted_position_sets: list[set[int]] = []
        quality_view_total_count = 0
        quality_view_accepted_count = 0
        ineligible_values: list[str] = []
        rejected_values: list[dict[str, str]] = []
        diagnostic_weights: dict[str, int] = {}
        diagnostic_sample_chunks: dict[str, list[torch.Tensor]] = {}
        stochastic_trace_chunks: dict[str, list[torch.Tensor]] = {}
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
                accepted_for_accounting = {
                    int(position) for position in metadata["accepted_positions"]
                }
                accepted_position_sets.append(accepted_for_accounting)
                quality_view_total_count += value.batch_size
                quality_view_accepted_count += len(accepted_for_accounting)
                ineligible_values.extend(
                    str(hash_id) for hash_id in metadata["ineligible_hash_ids"]
                )
                for item in metadata["quality_rejected"]:
                    rejection = dict(item)
                    rejection.setdefault("node_id", value.provenance.node_id)
                    rejected_values.append(rejection)
            prefix = f"{value.provenance.node_id}/"
            metadata_samples = metadata.get("diagnostic_samples")
            if isinstance(metadata_samples, Mapping):
                for local_name, sample_values in metadata_samples.items():
                    if not isinstance(local_name, str) or not isinstance(
                        sample_values, torch.Tensor
                    ):
                        raise TypeError("runtime diagnostic_samples metadata drifted")
                    diagnostic_sample_chunks.setdefault(
                        f"{prefix}{local_name}", []
                    ).append(sample_values)
            metadata_trace = metadata.get("stochastic_trace")
            if isinstance(metadata_trace, Mapping):
                for local_name, trace_values in metadata_trace.items():
                    if not isinstance(local_name, str) or not isinstance(
                        trace_values, torch.Tensor
                    ):
                        raise TypeError("runtime stochastic_trace metadata drifted")
                    stochastic_trace_chunks.setdefault(
                        f"{prefix}{local_name}", []
                    ).append(trace_values)
            node_diagnostics = tuple(
                name for name in bundle.diagnostics if name.startswith(prefix)
            )
            if not node_diagnostics:
                continue
            diagnostic_weight = metadata.get("diagnostic_weight")
            per_name_weights = metadata.get("diagnostic_weights")
            accepted_for_node = metadata.get("accepted_positions")
            if (
                isinstance(diagnostic_weight, int)
                and not isinstance(diagnostic_weight, bool)
                and diagnostic_weight > 0
            ):
                node_weight = diagnostic_weight
            elif isinstance(accepted_for_node, (tuple, list)):
                node_weight = max(1, len(accepted_for_node))
            else:
                node_weight = max(
                    1, int(value.valid_mask.sum().detach().cpu())
                )
            for diagnostic_name in node_diagnostics:
                local_name = diagnostic_name.removeprefix(prefix)
                specific_weight = (
                    per_name_weights.get(local_name)
                    if isinstance(per_name_weights, Mapping)
                    else None
                )
                diagnostic_weights[diagnostic_name] = (
                    int(specific_weight)
                    if isinstance(specific_weight, int)
                    and not isinstance(specific_weight, bool)
                    and specific_weight > 0
                    else node_weight
                )

        candidate_positions = tuple(sorted(candidate_position_set))
        accepted_positions = tuple(
            sorted(
                set.intersection(*accepted_position_sets)
                if accepted_position_sets
                else set()
            )
        )
        ineligible = tuple(dict.fromkeys(ineligible_values))
        rejected = tuple(
            dict(item)
            for item in {
                (entry.get("node_id", ""), entry["hash_id"], entry["reason"]): entry
                for entry in rejected_values
            }.values()
        )
        return GeneratedMethodBatch(
            bundle=bundle,
            diagnostic_weights=diagnostic_weights,
            diagnostic_samples={
                name: torch.cat(chunks, dim=0).detach()
                for name, chunks in diagnostic_sample_chunks.items()
            },
            stochastic_trace={
                name: torch.cat(chunks, dim=0).detach()
                for name, chunks in stochastic_trace_chunks.items()
            },
            candidate_eligible_positions=candidate_positions,
            accepted_positions=accepted_positions,
            quality_view_total_count=quality_view_total_count,
            quality_view_accepted_count=quality_view_accepted_count,
            ineligible_hash_ids=ineligible,
            quality_rejected=rejected,
        )


def build_method_runtime(
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
    return MethodViewRuntime(
        method,
        model_name=model_name,
        config_root=Path(config_root),
        latent_pool=latent_pool,
        encoder=encoder,
        decoder=decoder,
        minimum_std_mV=minimum_std_mV,
        maximum_abs_mV=maximum_abs_mV,
    )


__all__ = [
    "GeneratedMethodBatch",
    "MethodViewRuntime",
    "build_method_runtime",
]
