"""Finite-recipe bridge to the retained corruption, LHAT, and latent engines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.augmix import AugMixConfig, load_augmix_config
from core.corruption import generate_canonical_corruption
from core.lhat import (
    LHATConfig,
    contract_lhat_adversarial,
    generate_lhat_adversarial,
    load_lhat_config,
)
from core.methods.contracts import (
    BASE_VIEW_NAME,
    Provenance,
    ViewBundle,
    WaveformView,
)
from core.methods.registry import (
    AuxiliaryVariant,
    RecipeKind,
    RecipeSpec,
)
from util.augmentations.profile import AugmentationProfile, load_augmentation_profile
from util.config_bundle import resolve_config_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derive_seed(
    base_seed: int,
    comparison_rng_identity: str,
    namespace: str,
    node_id: str,
    stream: str,
    execution_identity: tuple[str, ...],
) -> int:
    """Preserve the pre-RecipeSpec semantic seed payload byte-for-byte."""

    payload = "|".join(
        (
            str(base_seed),
            comparison_rng_identity,
            namespace,
            node_id,
            stream,
            *execution_identity,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class _RecipeContext:
    """Narrow local context for one code-owned recipe operation."""

    def __init__(
        self,
        *,
        recipe: RecipeSpec,
        node_id: str,
        node_type: str,
        rng_namespace: str,
        resources: Mapping[str, Any],
        diagnostics: dict[str, Any],
        base_seed: int,
        rng_identity: tuple[str, ...],
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.rng_namespace = rng_namespace
        self._recipe = recipe
        self._resources = resources
        self._diagnostics = diagnostics
        self._base_seed = base_seed
        self._rng_identity = rng_identity
        self._generators: dict[tuple[str, str], torch.Generator] = {}

    def resource(self, name: str) -> Any:
        try:
            value = self._resources[name]
        except KeyError:
            raise KeyError(f"unsupported execution resource: {name}") from None
        if value is None:
            raise ValueError(
                f"recipe operation {self.node_id!r} requires {name!r}"
            )
        return value

    def record_diagnostic(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("diagnostic name must be non-empty")
        key = f"{self.node_id}/{name}"
        if key in self._diagnostics:
            raise ValueError(f"diagnostic already recorded: {key}")
        self._diagnostics[key] = value

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        if not isinstance(stream, str) or not stream:
            raise ValueError("RNG stream must be non-empty")
        resolved_device = torch.device(device)
        key = (stream, str(resolved_device))
        generator = self._generators.get(key)
        if generator is None:
            seed = _derive_seed(
                self._base_seed,
                self._recipe.comparison_rng_identity,
                self.rng_namespace,
                self.node_id,
                stream,
                self._rng_identity,
            )
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(seed)
            self._generators[key] = generator
            self._diagnostics[
                f"{self.node_id}/rng/{stream}/{resolved_device}"
            ] = {
                "seed": seed,
                "namespace": self.rng_namespace,
                "comparison_rng_identity": self._recipe.comparison_rng_identity,
                "execution_identity": list(self._rng_identity),
            }
        return generator


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
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("diagnostic weights must be positive")
        object.__setattr__(self, "diagnostic_weights", MappingProxyType(weights))
        samples = dict(self.diagnostic_samples)
        if any(value.ndim != 1 for value in samples.values()):
            raise ValueError("diagnostic samples must be rank-1")
        object.__setattr__(
            self,
            "diagnostic_samples",
            MappingProxyType({name: value.detach() for name, value in samples.items()}),
        )
        trace = dict(self.stochastic_trace)
        if any(value.ndim < 1 for value in trace.values()):
            raise ValueError("stochastic traces must be non-scalar")
        object.__setattr__(
            self,
            "stochastic_trace",
            MappingProxyType({name: value.detach() for name, value in trace.items()}),
        )
        if not 0 <= self.quality_view_accepted_count <= self.quality_view_total_count:
            raise ValueError("accepted view count must be within the total")

    @property
    def batch_size(self) -> int:
        clean = self.bundle.require(BASE_VIEW_NAME)
        assert isinstance(clean, WaveformView)
        return clean.batch_size


class MethodViewRuntime:
    """One finite recipe with resolved configuration resources."""

    def __init__(
        self,
        recipe: RecipeSpec,
        *,
        model_name: str,
        config_root: Path,
        latent_pool: Any | None,
        decoder: nn.Module | None,
        minimum_std_mV: float,
        maximum_abs_mV: float,
    ) -> None:
        if not isinstance(recipe, RecipeSpec):
            raise TypeError("recipe must be a RecipeSpec")
        self.recipe = recipe
        self.model_name = str(model_name)
        self.config_root = Path(config_root).expanduser().resolve()
        self.latent_pool = latent_pool
        self.decoder = decoder
        self.minimum_std_mV = float(minimum_std_mV)
        self.maximum_abs_mV = float(maximum_abs_mV)
        if self.minimum_std_mV <= 0.0 or self.maximum_abs_mV <= 0.0:
            raise ValueError("method quality thresholds must be positive")
        owner_path = recipe.source_path or (
            self.config_root / "train" / "methods" / f"{recipe.recipe_id}.yaml"
        )

        self._resource_paths: dict[str, Path] = {}
        self._rng_seed_paths: dict[str, Path] = {}
        for name, raw_resource in recipe.resources.items():
            if not isinstance(raw_resource, Mapping):
                raise ValueError(f"method resource {name!r} must be a mapping")
            resource_type = raw_resource.get("type")
            if resource_type == "config_reference":
                self._resource_paths[name] = resolve_config_reference(
                    raw_resource.get("path"),
                    owner_config_path=owner_path,
                    config_root=self.config_root,
                    description=f"method.resources.{name}",
                    must_exist=True,
                )
            elif resource_type == "isolated_torch_generator":
                self._rng_seed_paths[name] = resolve_config_reference(
                    raw_resource.get("seed_config"),
                    owner_config_path=owner_path,
                    config_root=self.config_root,
                    description=f"method.resources.{name}.seed_config",
                    must_exist=True,
                )
            else:
                raise ValueError(
                    f"method resource {name!r} has unsupported type "
                    f"{resource_type!r}"
                )

        self.operator_profile: AugmentationProfile | None = None
        operator_resource = recipe.resources.get("operator_profile")
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

        if recipe.requirements.latent_pool and latent_pool is None:
            raise ValueError("method requires a train-only latent_pool")
        if recipe.requirements.vae_decoder and decoder is None:
            raise ValueError("method requires a frozen VAE decoder")
        if not recipe.requirements.latent_pool and latent_pool is not None:
            raise ValueError("method without latent_pool requirement may not consume one")
        if not recipe.requirements.vae_decoder and decoder is not None:
            raise ValueError("method without VAE decoder requirement may not consume one")

    @property
    def requires_latent_pool(self) -> bool:
        return bool(self.recipe.requirements.latent_pool)

    @property
    def rng_seed_config_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(dict(self._rng_seed_paths))

    def describe(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.describe(),
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
        context: _RecipeContext,
        inputs: tuple[WaveformView, ...],
        *,
        composition_indices: torch.Tensor | None = None,
        composition_index_hint: int | None = None,
    ) -> WaveformView:
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
        context: _RecipeContext,
        inputs: tuple[WaveformView, ...],
    ) -> WaveformView:
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
                    else (
                        "contract_rejected"
                        if reason == "accepted"
                        else f"contract_rejected+{reason}"
                    )
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

        ``composition_indices=None`` preserves the recipe's ordinary random
        composition sampling.  A device-local integer tensor in ``[0, 19]``
        forces one canonical depth-2/3 composition per record.  An all-``-1``
        tensor is the explicit clean-identity exposure used by the fixed-20
        direct fine-tuning schedule; mixing ``-1`` with corruptions is rejected.
        """

        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or not 0 <= base_seed < 2**32
        ):
            raise ValueError("base_seed must be a uint32 integer")
        identity = tuple(rng_identity)
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("rng_identity must contain non-empty strings")
        clean = WaveformView(
            name="clean_identity",
            waveform=clean_raw,
            labels=targets,
            sample_ids=tuple(str(value) for value in hash_ids),
            provenance=Provenance(
                node_id="clean_identity",
                operation="identity_raw100_view",
                parent_names=("clean_raw",),
                parameters={"source": "clean_raw"},
            ),
            metadata={"source_view": "clean_raw"},
        )
        selected_terms = (
            self.recipe.objective.terms
            if objective_term_names is None
            else tuple(
                term
                for term in self.recipe.objective.terms
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
        diagnostics: dict[str, Any] = {
            "recipe/id": self.recipe.recipe_id,
            "recipe/sha256": self.recipe.recipe_sha256,
            "recipe/rng_namespace": self.recipe.rng_namespace,
            "recipe/comparison_rng_identity": self.recipe.comparison_rng_identity,
        }
        execution_resources = {
            "classifier": classifier,
            "vae_decoder": self.decoder,
            "latent_pool": self.latent_pool,
        }

        def context(
            node_id: str, node_type: str, rng_resource: str
        ) -> _RecipeContext:
            try:
                namespace = self.recipe.rng_namespaces[rng_resource]
            except KeyError:
                raise RuntimeError(
                    f"recipe has no RNG namespace for {rng_resource!r}"
                ) from None
            return _RecipeContext(
                recipe=self.recipe,
                node_id=node_id,
                node_type=node_type,
                rng_namespace=namespace,
                resources=execution_resources,
                diagnostics=diagnostics,
                base_seed=base_seed,
                rng_identity=identity,
            )

        generated: dict[str, WaveformView] = {BASE_VIEW_NAME: clean}
        if "corrupted_view" in required_outputs:
            generated["corrupted_view"] = self._canonical_corruption(
                context(
                    "depth23_corruption",
                    "canonical_depth23_corruption_view",
                    "corruption_rng",
                ),
                (clean,),
                composition_indices=composition_indices,
                composition_index_hint=composition_index_hint,
            )
        if "lhat_view" in required_outputs:
            if self.recipe.auxiliary_variant is not AuxiliaryVariant.CONTRACTED_LHAT:
                raise RuntimeError("only the contracted-LHAT recipe can emit lhat_view")
            generated["lhat_view"] = self._lhat_attack(
                context(
                    "lhat",
                    "vae_lhat_attack_then_contract_view",
                    "lhat_rng",
                ),
                (clean,),
            )
        missing_outputs = sorted(required_outputs - set(generated))
        if missing_outputs:
            raise RuntimeError(
                f"recipe dispatch did not produce required outputs: {missing_outputs}"
            )
        bundle = ViewBundle(
            values={
                name: generated[name]
                for name in self.recipe.output_names
                if name in required_outputs
            },
            diagnostics=diagnostics,
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
    recipe: RecipeSpec,
    *,
    model_name: str,
    config_root: str | Path,
    latent_pool: Any | None = None,
    decoder: nn.Module | None = None,
    minimum_std_mV: float = 1.0e-4,
    maximum_abs_mV: float = 20.0,
) -> MethodViewRuntime:
    return MethodViewRuntime(
        recipe,
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
