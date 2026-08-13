"""Fail-closed loader for the finite, code-owned ECG recipe set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from util.config_bundle import require_mapping as _mapping


RECIPE_SCHEMA_VERSION = 2
RECIPE_VERSION = 1
RECIPE_IMPLEMENTATION_IDENTITY = "finite_recipe_spec_v1"
FORBIDDEN_DYNAMIC_KEYS = frozenset(
    {"callable", "class", "class_path", "import", "import_path", "module"}
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain(nested) for nested in value]
    return value


class RecipeKind(str, Enum):
    CLEAN = "clean"
    RANDOM_DEPTH23 = "random_depth23"
    FIXED20 = "fixed20"
    TWO_STAGE_AUGMIX_LHAT = "two_stage_augmix_lhat"


class AuxiliaryVariant(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    CONTRACTED_LHAT = "contracted_lhat"
    MATCHED_NO_VAE = "matched_no_vae"


_RESOURCE_SCHEMAS: Mapping[str, tuple[str, frozenset[str]]] = {
    "operator_profile": (
        "config_reference",
        frozenset({"type", "path", "profile", "severity"}),
    ),
    "vae": ("config_reference", frozenset({"type", "path"})),
    "lhat_config": ("config_reference", frozenset({"type", "path"})),
    "augmix_config": ("config_reference", frozenset({"type", "path"})),
    **{
        name: (
            "isolated_torch_generator",
            frozenset({"type", "seed_config", "namespace"}),
        )
        for name in (
            "corruption_rng",
            "lhat_rng",
        )
    },
}


@dataclass(frozen=True)
class _RecipeDefinition:
    kind: RecipeKind
    auxiliary_variant: AuxiliaryVariant
    scientific_arm: str
    status: str
    resource_names: frozenset[str]
    comparison_rng_identity: str
    output_names: tuple[str, ...]


_OBJECTIVE_NAME_BY_VIEW = MappingProxyType(
    {
        "clean_view": "clean_bce",
        "lhat_view": "lhat_direct_bce",
        "corrupted_view": "corrupted_bce",
    }
)


def _objective_terms(output_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((_OBJECTIVE_NAME_BY_VIEW[view], view) for view in output_names)


def _objective_descriptions(output_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "kind": "bce",
            "views": [view],
            "weight": 1.0,
            "mask_policy": "valid_intersection",
        }
        for name, view in _objective_terms(output_names)
    ]


def _requirement_names(output_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        ("classifier", "vae_decoder", "latent_pool")
        if "lhat_view" in output_names
        else ("classifier",)
    )
_MODEL_ONLY = frozenset()
_CORRUPTION_RESOURCES = frozenset({"operator_profile", "corruption_rng"})
_MAINLINE_RESOURCES = frozenset(
    {
        "operator_profile",
        "augmix_config",
        "vae",
        "lhat_config",
        "corruption_rng",
        "lhat_rng",
    }
)
_NO_VAE_RESOURCES = frozenset(
    {"operator_profile", "augmix_config", "corruption_rng"}
)
_DEFINITIONS: Mapping[str, _RecipeDefinition] = MappingProxyType(
    {
        "a0_clean_v1": _RecipeDefinition(
            RecipeKind.CLEAN,
            AuxiliaryVariant.NOT_APPLICABLE,
            "A0",
            "locked_reference",
            _MODEL_ONLY,
            "a0_clean_v1",
            ("clean_view",),
        ),
        "a3c_depth23_v1": _RecipeDefinition(
            RecipeKind.RANDOM_DEPTH23,
            AuxiliaryVariant.NOT_APPLICABLE,
            "A3c",
            "locked_comparison_candidate",
            _CORRUPTION_RESOURCES,
            "a3c_depth23_v1",
            ("clean_view", "corrupted_view"),
        ),
        "direct_depth23_fixed20": _RecipeDefinition(
            RecipeKind.FIXED20,
            AuxiliaryVariant.NOT_APPLICABLE,
            "DirectDepth23Fixed20FamilyBalanced",
            "locked_family_balanced_baseline",
            _CORRUPTION_RESOURCES,
            "direct_depth23_fixed20",
            ("clean_view", "corrupted_view"),
        ),
        "augmix_simclr_lhat": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "augmix_simclr_lhat",
            "locked_prospective_replication_recipe",
            _MAINLINE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
        ),
        # Deliberately the only non-file-backed future variant.  Keeping the
        # paired identity here permits a matched no-VAE ablation without
        # opening a general kind/variant cross product.
        "augmix_simclr_matched_no_vae": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.MATCHED_NO_VAE,
            "augmix_simclr_matched_no_vae",
            "prospective_matched_ablation",
            _NO_VAE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view"),
        ),
    }
)


def _execution_contract(
    kind: RecipeKind, variant: AuxiliaryVariant
) -> Mapping[str, Any]:
    common = {
        "one_outer_optimizer_step_per_clean_batch": True,
        "complete_k500_base_record_exposure": True,
        "heldout_target_feedback_allowed": False,
    }
    if kind is RecipeKind.CLEAN:
        return MappingProxyType(
            {
                **common,
                "stages": ["supervised_adaptation"],
                "exposure_policy": "clean_once",
                "generated_view_count": 0,
                "batch_norm_policy": "ordinary_clean_forward",
            }
        )
    if kind is RecipeKind.RANDOM_DEPTH23:
        return MappingProxyType(
            {
                **common,
                "stages": ["supervised_adaptation"],
                "exposure_policy": "clean_plus_one_random_depth23",
                "corruption_depths": [2, 3],
                "objective_weights": {"clean_bce": 1.0, "corrupted_bce": 1.0},
                "generated_view_count": 1,
                "batch_norm_policy": "ordinary_objective_view_order",
            }
        )
    if kind is RecipeKind.FIXED20:
        return MappingProxyType(
            {
                **common,
                "stages": ["supervised_adaptation"],
                "exposure_policy": "clean_once_then_exhaustive_depth23",
                "composition_indices": list(range(20)),
                "family_loss_weights": {
                    "clean": 0.5,
                    "corrupted_total": 0.5,
                    "corrupted_per_composition": 0.025,
                },
                "generated_view_count": 20,
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    if kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT:
        contracted = variant is AuxiliaryVariant.CONTRACTED_LHAT
        return MappingProxyType(
            {
                **common,
                "stages": ["augmix_simclr", "supervised_adaptation"],
                "stage1": {
                    "view": "clean_vs_one_twochain_augmix_strong_view",
                    "objective": "simclr",
                    "temperature": 0.5,
                    "classifier_head_trainable": False,
                    "pretrain_logit_anchor_weight": 5.0,
                    "pretrain_logit_anchor_class_weights": [1.0] * 5,
                    "ptbxl_source_replay_weight": 0.0,
                    "vicreg_weight": 0.0,
                    "vae_lhat_tail_fraction": 0.0,
                },
                "stage2_teacher": "post_stage1_pre_stage2_snapshot",
                "stage2_supervised_logit_anchor_weight_by_backbone": {
                    "efficientnet1dv2": 2.0,
                    "ecgfounder": 0.5,
                },
                "exposure_policy": (
                    "clean_aux_once_then_rotating_depth23_2plus2"
                    if contracted
                    else "clean_once_then_rotating_depth23_2plus2"
                ),
                "rotating4_schedule": "epoch_modulo_five_covers_all_depth23_compositions",
                "family_loss_weights": {
                    "clean": 0.5,
                    "corrupted_total": 0.5,
                    "corrupted_per_composition": 0.125,
                },
                "auxiliary": (
                    {
                        "objective_terms": ["lhat_direct_bce"],
                        "alpha": 2.0,
                        "gradient_merge": "direct_sum",
                        "batch_norm_policy": "snapshot_restore",
                        "global_rng_policy": "snapshot_restore",
                        "attack_then_contract_version": "preflip_maxloss_grid_v1",
                        "diagnostic_scopes": [
                            "raw_all_candidate_eligible",
                            "contract_all_candidate_eligible",
                            "contract_training_accepted",
                        ],
                    }
                    if contracted
                    else None
                ),
                "generated_view_count": 5 if contracted else 4,
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    raise AssertionError(f"unsupported recipe kind: {kind}")


@dataclass(frozen=True, init=False)
class RecipeSpec:
    """One loader-owned member of the finite recipe family."""

    profile_name: str
    resources: Mapping[str, Mapping[str, Any]]
    rng_namespaces: Mapping[str, str]
    recipe_sha256: str
    source_path: Path | None
    scientific_contract: Mapping[str, Any]
    _definition: _RecipeDefinition = field(repr=False)

    def __init__(self, *_: Any, **__: Any) -> None:
        raise TypeError("RecipeSpec is loader-owned; use load_recipe_spec")

    def __post_init__(self) -> None:
        if _DEFINITIONS.get(self.profile_name) is not self._definition:
            raise ValueError("recipe definition must match its code-owned profile")
        object.__setattr__(
            self,
            "resources",
            _freeze(self.resources),
        )
        object.__setattr__(
            self, "rng_namespaces", MappingProxyType(dict(self.rng_namespaces))
        )
        object.__setattr__(
            self, "scientific_contract", _freeze(self.scientific_contract)
        )

    @property
    def recipe_id(self) -> str:
        return self.profile_name

    @property
    def kind(self) -> RecipeKind:
        return self._definition.kind

    @property
    def auxiliary_variant(self) -> AuxiliaryVariant:
        return self._definition.auxiliary_variant

    @property
    def scientific_arm(self) -> str:
        return self._definition.scientific_arm

    @property
    def status(self) -> str:
        return self._definition.status

    @property
    def comparison_rng_identity(self) -> str:
        return self._definition.comparison_rng_identity

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._definition.output_names

    @property
    def objective_terms(self) -> tuple[tuple[str, str], ...]:
        return _objective_terms(self.output_names)

    @property
    def requires_vae(self) -> bool:
        return "lhat_view" in self.output_names

    @property
    def schema_version(self) -> int:
        return RECIPE_SCHEMA_VERSION

    @property
    def recipe_version(self) -> int:
        return RECIPE_VERSION

    @property
    def implementation_identity(self) -> str:
        return RECIPE_IMPLEMENTATION_IDENTITY

    @property
    def rng_namespace(self) -> str:
        namespaces = tuple(dict.fromkeys(self.rng_namespaces.values()))
        return namespaces[0] if len(namespaces) == 1 else self.profile_name

    def describe(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "profile_name": self.profile_name,
            "kind": self.kind.value,
            "auxiliary_variant": self.auxiliary_variant.value,
            "scientific_arm": self.scientific_arm,
            "status": self.status,
            "schema_version": self.schema_version,
            "recipe_version": self.recipe_version,
            "implementation_identity": self.implementation_identity,
            "recipe_sha256": self.recipe_sha256,
            "source_path": None if self.source_path is None else str(self.source_path),
            "comparison_rng_identity": self.comparison_rng_identity,
            "rng_namespace": self.rng_namespace,
            "rng_namespaces": dict(self.rng_namespaces),
            "outputs": list(self.output_names),
            "requirements": list(_requirement_names(self.output_names)),
            "objective_terms": _objective_descriptions(self.output_names),
            "resources": _plain(self.resources),
            "scientific_contract": _plain(self.scientific_contract),
        }


def _name(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _reject_dynamic_keys(value: Any, path: str = "recipe") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_DYNAMIC_KEYS:
                raise ValueError(
                    f"{path}.{key} is forbidden; recipes may not select callables"
                )
            _reject_dynamic_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_dynamic_keys(nested, f"{path}[{index}]")


def _validated_resources(
    raw: Any, expected_names: frozenset[str]
) -> dict[str, dict[str, Any]]:
    resources = _mapping(raw, "resources")
    if set(resources) != expected_names:
        raise ValueError(
            "recipe resources must be exactly "
            f"{sorted(expected_names)}, got {sorted(resources)}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for raw_name, raw_resource in resources.items():
        name = _name(raw_name, "resource name")
        expected_type, expected_keys = _RESOURCE_SCHEMAS[name]
        resource = _mapping(raw_resource, f"resources.{name}")
        if set(resource) != expected_keys:
            raise ValueError(
                f"resources.{name} keys must be exactly {sorted(expected_keys)}"
            )
        if resource.get("type") != expected_type:
            raise ValueError(f"resources.{name}.type must be {expected_type!r}")
        for key in ("path", "profile", "seed_config", "namespace"):
            if key in resource:
                _name(resource[key], f"resources.{name}.{key}")
        if "severity" in resource:
            severity = resource["severity"]
            if isinstance(severity, bool) or not isinstance(severity, int) or severity <= 0:
                raise ValueError(f"resources.{name}.severity must be a positive integer")
        validated[name] = dict(resource)
    return validated


def load_recipe_spec(source: str | Path | Mapping[str, Any]) -> RecipeSpec:
    """Resolve an exact finite recipe; arbitrary graphs/plugins are impossible."""

    source_path: Path | None = None
    if isinstance(source, Mapping):
        root = dict(source)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"recipe not found: {source_path}")
        root = _mapping(
            yaml.safe_load(source_path.read_text(encoding="utf-8")), "recipe"
        )
    if set(root) != {"schema_version", "recipe", "resources"}:
        raise ValueError(
            "recipe root keys must be exactly "
            "['recipe', 'resources', 'schema_version']"
        )
    if root.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise ValueError(f"recipe schema_version must be {RECIPE_SCHEMA_VERSION}")
    _reject_dynamic_keys(root)
    recipe = _mapping(root["recipe"], "recipe")
    expected_recipe_keys = {
        "id",
        "kind",
        "auxiliary_variant",
        "scientific_arm",
        "status",
    }
    if set(recipe) != expected_recipe_keys:
        raise ValueError(f"recipe keys must be exactly {sorted(expected_recipe_keys)}")
    recipe_id = _name(recipe["id"], "recipe.id")
    try:
        definition = _DEFINITIONS[recipe_id]
    except KeyError:
        raise ValueError(
            f"unknown recipe.id {recipe_id!r}; allowed={sorted(_DEFINITIONS)}"
        ) from None
    expected_values = {
        "kind": definition.kind.value,
        "auxiliary_variant": definition.auxiliary_variant.value,
        "scientific_arm": definition.scientific_arm,
        "status": definition.status,
    }
    for key, expected in expected_values.items():
        if recipe.get(key) != expected:
            raise ValueError(f"recipe {recipe_id!r} requires {key}={expected!r}")
    resources = _validated_resources(root["resources"], definition.resource_names)
    rng_namespaces = {
        name: str(resource["namespace"])
        for name, resource in resources.items()
        if resource.get("type") == "isolated_torch_generator"
    }
    scientific_contract = _execution_contract(
        definition.kind, definition.auxiliary_variant
    )
    identity_payload = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "recipe_version": RECIPE_VERSION,
        "implementation_identity": RECIPE_IMPLEMENTATION_IDENTITY,
        "declaration": root,
        "comparison_rng_identity": definition.comparison_rng_identity,
        "outputs": list(definition.output_names),
        "requirements": list(_requirement_names(definition.output_names)),
        "objective_terms": _objective_descriptions(definition.output_names),
        "scientific_contract": _plain(scientific_contract),
    }
    try:
        identity = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"recipe must be JSON-serializable: {exc}") from exc
    spec = object.__new__(RecipeSpec)
    for name, value in {
        "profile_name": recipe_id,
        "resources": resources,
        "rng_namespaces": rng_namespaces,
        "recipe_sha256": hashlib.sha256(identity).hexdigest(),
        "source_path": source_path,
        "scientific_contract": scientific_contract,
        "_definition": definition,
    }.items():
        object.__setattr__(spec, name, value)
    spec.__post_init__()
    return spec


__all__ = [
    "AuxiliaryVariant",
    "RECIPE_IMPLEMENTATION_IDENTITY",
    "RECIPE_SCHEMA_VERSION",
    "RECIPE_VERSION",
    "RecipeKind",
    "load_recipe_spec",
]
