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
    SUPERVISED_ROTATING_DEPTH23 = "supervised_rotating_depth23"
    SUPERVISED_ROTATING_DEPTH23_LHAT = "supervised_rotating_depth23_lhat"
    SUPERVISED_ROTATING_DEPTH23_MILD_AUX = (
        "supervised_rotating_depth23_mild_aux"
    )
    SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX = (
        "supervised_rotating_depth23_hardgain_aux"
    )
    FIXED20 = "fixed20"
    TWO_STAGE_AUGMIX_LHAT = "two_stage_augmix_lhat"
    ONE_STAGE_SUPERVISED_AUGMIX = "one_stage_supervised_augmix"
    ONE_STAGE_VAE_LHAT = "one_stage_vae_lhat"
    ONE_STAGE_AUGMIX_LHAT = "one_stage_augmix_lhat"


class AuxiliaryVariant(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    CONTRACTED_LHAT = "contracted_lhat"
    RAW_LHAT = "raw_lhat"
    MATCHED_NO_VAE = "matched_no_vae"


class Stage1Objective(str, Enum):
    SIMCLR = "simclr"
    SUPERVISED_AUGMIX = "supervised_augmix"
    CLEAN_BCE = "clean_bce"
    CLEAN_BCE_JSD = "clean_bce_jsd"
    NOT_APPLICABLE = "not_applicable"


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
    stage1_objective: Stage1Objective = Stage1Objective.NOT_APPLICABLE
    stage1_view: str = "clean_vs_one_twochain_augmix_strong_view"
    lhat_contract_version: str = "nondecreasing_bce_grid_v2"
    augmix_jsd_weight: float = 0.0
    one_stage_lhat_alpha_max: float = 0.25
    lhat_auxiliary_alpha_max: float = 0.1
    augmix_auxiliary_weight: float = 0.125
    augmix_supervised_view_policy: str = "mixed_only"
    rotating_base_scale: float = 1.0
    rotating_clean_weight: float | None = None
    rotating_corrupted_total_weight: float | None = None
    lhat_training_selection: tuple[str, float] | None = None
    lhat_auxiliary_schedule: str = "constant_after_warmup"
    lhat_auxiliary_warmup_epochs: int = 5
    lhat_loss_integration: str = "additive"
    lhat_candidate_source_policy: str = "clean_only"


_OBJECTIVE_NAME_BY_VIEW = MappingProxyType(
    {
        "clean_view": "clean_bce",
        "lhat_view": "lhat_direct_bce",
        "corrupted_view": "corrupted_bce",
        "augmix_view": "augmix_bce",
        "augmix_chain1_view": "augmix_chain1_context",
        "augmix_chain2_view": "augmix_chain2_context",
    }
)


def _objective_terms(output_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((_OBJECTIVE_NAME_BY_VIEW[view], view) for view in output_names)


def _objective_descriptions(
    output_names: tuple[str, ...], *, augmix_jsd_weight: float = 0.0,
    augmix_supervised_view_policy: str = "mixed_only",
) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for name, view in _objective_terms(output_names):
        if name == "augmix_bce" and (
            augmix_jsd_weight > 0.0
            or augmix_supervised_view_policy != "mixed_only"
        ):
            chain_views = [
                value for value in ("augmix_chain1_view", "augmix_chain2_view")
                if value in output_names
            ]
            descriptions.append(
                {
                    "name": name,
                    "kind": (
                        "bce_plus_multilabel_bernoulli_jsd"
                        if augmix_jsd_weight > 0.0
                        else "multiview_bce"
                    ),
                    "views": [
                        *(["clean_view"] if augmix_jsd_weight > 0.0 else []),
                        view,
                        *chain_views,
                    ],
                    "weight": 1.0,
                    "mask_policy": "valid_intersection",
                    **(
                        {"bernoulli_jsd_weight": float(augmix_jsd_weight)}
                        if augmix_jsd_weight > 0.0
                        else {}
                    ),
                    **(
                        {
                            "supervised_view_policy": (
                                augmix_supervised_view_policy
                            )
                        }
                        if augmix_supervised_view_policy != "mixed_only"
                        else {}
                    ),
                }
            )
        else:
            descriptions.append(
                {
                    "name": name,
                    "kind": "context_only" if name.endswith("_context") else "bce",
                    "views": [view],
                    "weight": 0.0 if name.endswith("_context") else 1.0,
                    "mask_policy": "valid_intersection",
                }
            )
    return descriptions


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
_VAE_ONLY_RESOURCES = frozenset(
    {"operator_profile", "vae", "lhat_config", "corruption_rng", "lhat_rng"}
)
_ONE_STAGE_AUGMIX_RESOURCES = frozenset(
    {"augmix_config", "corruption_rng"}
)
_ONE_STAGE_VAE_RESOURCES = frozenset(
    {"vae", "lhat_config", "lhat_rng"}
)
_ONE_STAGE_JOINT_RESOURCES = frozenset(
    {"augmix_config", "vae", "lhat_config", "corruption_rng", "lhat_rng"}
)
_ROT4_AUGMIX_RESOURCES = frozenset(
    {"operator_profile", "augmix_config", "corruption_rng"}
)
_ROT4_LHAT_MILD_RESOURCES = frozenset(
    {"operator_profile", "vae", "lhat_config", "corruption_rng", "lhat_rng"}
)
_ROT4_JOINT_MILD_RESOURCES = frozenset(
    {
        "operator_profile",
        "augmix_config",
        "vae",
        "lhat_config",
        "corruption_rng",
        "lhat_rng",
    }
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
        "a1_corrupt_ft_rot4_v1": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23,
            AuxiliaryVariant.NOT_APPLICABLE,
            "A1",
            "prospective_matched_baseline",
            _CORRUPTION_RESOURCES,
            "augmix_simclr_lhat",
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
            "locked_minimal_final_development_recipe",
            _MAINLINE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
            stage1_objective=Stage1Objective.SIMCLR,
        ),
        "augmix_clean_bce_lhat": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "augmix_clean_bce_lhat", "prospective_stage1_objective_ablation",
            _MAINLINE_RESOURCES, "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
            stage1_objective=Stage1Objective.CLEAN_BCE,
        ),
        "augmix_clean_bce_jsd_lhat": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "augmix_clean_bce_jsd_lhat", "prospective_stage1_objective_ablation",
            _MAINLINE_RESOURCES, "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
            stage1_objective=Stage1Objective.CLEAN_BCE_JSD,
        ),
        "augmix_simclr_matched_no_vae": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.MATCHED_NO_VAE,
            "augmix_simclr_matched_no_vae",
            "prospective_matched_ablation",
            _NO_VAE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view"),
            stage1_objective=Stage1Objective.SIMCLR,
        ),
        "augmix_supervised_lhat": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "augmix_supervised_lhat",
            "prospective_matched_ablation",
            _MAINLINE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
            stage1_objective=Stage1Objective.SUPERVISED_AUGMIX,
        ),
        "augmix_supervised_matched_no_vae": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.MATCHED_NO_VAE,
            "augmix_supervised_matched_no_vae",
            "prospective_matched_ablation",
            _NO_VAE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view"),
            stage1_objective=Stage1Objective.SUPERVISED_AUGMIX,
        ),
        "augmix_supervised_single_chain_matched_no_vae": _RecipeDefinition(
            RecipeKind.TWO_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.MATCHED_NO_VAE,
            "augmix_supervised_single_chain_matched_no_vae",
            "prospective_matched_ablation",
            _NO_VAE_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view"),
            stage1_objective=Stage1Objective.SUPERVISED_AUGMIX,
            stage1_view="clean_vs_one_single_chain_corruption_view",
        ),
        "vae_lhat_only": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "vae_lhat_only",
            "prospective_matched_ablation",
            _VAE_ONLY_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "lhat_view", "corrupted_view"),
        ),
        "a1_rot4_augmix_mild": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_augmix_mild",
            "prospective_mechanism_screen",
            _ROT4_AUGMIX_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "augmix_view"),
        ),
        "a1_rot4_single_chain_mild": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_mild",
            "prospective_matched_chain_ablation",
            _ROT4_AUGMIX_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_corruption_view",
        ),
        "a1_rot4_vae_lhat_mild": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_vae_lhat_mild",
            "prospective_mechanism_screen",
            _ROT4_LHAT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "lhat_view"),
        ),
        "a1_rot4_augmix_vae_lhat_mild": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_augmix_vae_lhat_mild",
            "prospective_mechanism_screen",
            _ROT4_JOINT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "augmix_view", "lhat_view"),
        ),
        "a1_rot4_augmix_jsd": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_augmix_jsd",
            "prospective_repaired_mechanism_screen",
            _ROT4_AUGMIX_RESOURCES,
            "augmix_simclr_lhat",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=12.0,
        ),
        "a1_rot4_single_chain_jsd": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_jsd",
            "prospective_repaired_chain_ablation",
            _ROT4_AUGMIX_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
            augmix_jsd_weight=12.0,
        ),
        "a1_rot4_vae_lhat_puredelta": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_vae_lhat_puredelta",
            "prospective_repaired_mechanism_screen",
            _ROT4_LHAT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "lhat_view"),
            lhat_contract_version="nondecreasing_pure_delta_grid_v3",
        ),
        "a1_rot4_augmix_jsd_vae_lhat_puredelta": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_augmix_jsd_vae_lhat_puredelta",
            "prospective_repaired_mechanism_screen",
            _ROT4_JOINT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
                "lhat_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            lhat_contract_version="nondecreasing_pure_delta_grid_v3",
            augmix_jsd_weight=12.0,
        ),
        "a1_rot4_single_chain_jsd_strong": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_jsd_strong",
            "prospective_a1_pool_strength_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_augmix_lhat_r3",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
            augmix_jsd_weight=12.0,
            augmix_auxiliary_weight=0.5,
        ),
        "a1_rot4_augmix_jsd_endpoint_mean_strong": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_augmix_jsd_endpoint_mean_strong",
            "prospective_a1_pool_strength_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_augmix_lhat_r3",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=12.0,
            augmix_auxiliary_weight=0.5,
            augmix_supervised_view_policy="mixed_plus_chains_mean",
        ),
        "a1_rot4_augmix_jsd_endpoint_mean_strong_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_augmix_jsd_endpoint_mean_strong_vae_lhat_puredelta",
                "prospective_a1_pool_strength_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_augmix_lhat_r3",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=12.0,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
            )
        ),
        "a1_rot4_augmix_jsd_hardview_strong": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_augmix_jsd_hardview_strong",
            "prospective_a1_pool_hardview_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_augmix_lhat_r3",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=12.0,
            augmix_auxiliary_weight=0.5,
            augmix_supervised_view_policy="per_sample_max_mixed_and_chains",
        ),
        "a1_rot4_augmix_jsd_hardview_strong_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_augmix_jsd_hardview_strong_vae_lhat_puredelta",
                "prospective_a1_pool_hardview_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_augmix_lhat_r3",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=12.0,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy=(
                    "per_sample_max_mixed_and_chains"
                ),
            )
        ),
        "a1_rot4_augmix_jsd_complementary_strong": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_augmix_jsd_complementary_strong",
            "prospective_a1_pool_complementary_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_augmix_lhat_r3",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=12.0,
            augmix_auxiliary_weight=0.5,
        ),
        "a1_rot4_augmix_jsd_complementary_strong_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_augmix_jsd_complementary_strong_vae_lhat_puredelta",
                "prospective_a1_pool_complementary_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_augmix_lhat_r3",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=12.0,
                augmix_auxiliary_weight=0.5,
            )
        ),
        "a1_rot4_single_chain_supervised_balanced": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_supervised_balanced",
            "prospective_a1_pool_balanced_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_augmix_lhat_r4",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_corruption_view",
            augmix_auxiliary_weight=0.5,
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_supervised_balanced": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_two_chain_supervised_balanced",
            "prospective_a1_pool_balanced_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_augmix_lhat_r4",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_supervised_chains_mean",
            augmix_auxiliary_weight=0.5,
            augmix_supervised_view_policy="chains_mean",
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_supervised_balanced_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_supervised_balanced_vae_lhat_puredelta",
                "prospective_a1_pool_balanced_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_augmix_lhat_r4",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_supervised_chains_mean",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
            )
        ),
        "a1_rot4_single_chain_balanced_jsd3": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_balanced_jsd3",
            "prospective_a1_pool_balanced_jsd3_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_jsd3_augmix_lhat_r5",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
            augmix_jsd_weight=3.0,
            augmix_auxiliary_weight=0.5,
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_balanced_jsd3": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_two_chain_balanced_jsd3",
            "prospective_a1_pool_balanced_jsd3_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_jsd3_augmix_lhat_r5",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=3.0,
            augmix_auxiliary_weight=0.5,
            augmix_supervised_view_policy="chains_mean",
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_balanced_jsd3_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_balanced_jsd3_vae_lhat_puredelta",
                "prospective_a1_pool_balanced_jsd3_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd3_augmix_lhat_r5",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=3.0,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
            )
        ),
        "a1_rot4_single_chain_balanced_jsd1p5": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_single_chain_balanced_jsd1p5",
            "prospective_a1_pool_balanced_jsd1p5_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
            ("clean_view", "corrupted_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
            augmix_jsd_weight=1.5,
            augmix_auxiliary_weight=0.5,
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_balanced_jsd1p5": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "a1_rot4_two_chain_balanced_jsd1p5",
            "prospective_a1_pool_balanced_jsd1p5_screen",
            _ROT4_AUGMIX_RESOURCES,
            "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
            (
                "clean_view",
                "corrupted_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=1.5,
            augmix_auxiliary_weight=0.5,
            augmix_supervised_view_policy="chains_mean",
            rotating_base_scale=0.5,
        ),
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_puredelta": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_puredelta",
                "prospective_a1_pool_balanced_jsd1p5_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
            )
        ),
        **{
            f"a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace{suffix}": (
                _RecipeDefinition(
                    RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                    AuxiliaryVariant.CONTRACTED_LHAT,
                    f"a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace{suffix}",
                    "prospective_a1_vae_clean_replacement_screen",
                    _ROT4_JOINT_MILD_RESOURCES,
                    "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                    (
                        "clean_view",
                        "corrupted_view",
                        "augmix_view",
                        "augmix_chain1_view",
                        "augmix_chain2_view",
                        "lhat_view",
                    ),
                    stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                    lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                    augmix_jsd_weight=1.5,
                    lhat_auxiliary_alpha_max=mass,
                    augmix_auxiliary_weight=0.5,
                    augmix_supervised_view_policy="chains_mean",
                    rotating_base_scale=0.5,
                    lhat_auxiliary_warmup_epochs=1,
                    lhat_loss_integration="replace_clean_with_lhat_or_clean_fallback",
                )
            )
            for suffix, mass in (("0p05", 0.05), ("0p1", 0.1), ("0p2", 0.2))
        },
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_mixedm20": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_mixedm20",
                "prospective_a1_lhat_candidate_source_ablation",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                (
                    "clean_view", "corrupted_view", "augmix_view",
                    "augmix_chain1_view", "augmix_chain2_view", "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.2,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
                lhat_auxiliary_warmup_epochs=1,
                lhat_loss_integration="replace_clean_with_lhat_or_clean_fallback",
                lhat_candidate_source_policy=(
                    "clean10_corrupted10_same_neighbors_v1"
                ),
            )
        ),
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_nocontract": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.RAW_LHAT,
                "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_nocontract",
                "prospective_a1_lhat_contract_ablation",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                (
                    "clean_view", "corrupted_view", "augmix_view",
                    "augmix_chain1_view", "augmix_chain2_view", "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="raw_attack_no_contract_v1",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.2,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
                lhat_auxiliary_warmup_epochs=1,
                lhat_loss_integration="replace_clean_with_lhat_or_clean_fallback",
            )
        ),
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p25": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p25",
                "prospective_a1_pool_balanced_jsd1p5_alpha_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.25,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
            )
        ),
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p5",
                "prospective_a1_pool_balanced_jsd1p5_alpha_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_pool_balanced_jsd1p5_augmix_lhat_r6",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.5,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_base_scale=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r375_a500_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r375_a500_jsd1p5_vae_lhat",
                "prospective_a1_robust_weight_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_robust_weight_screen_r7",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.375,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat",
                "prospective_a1_robust_weight_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_robust_weight_screen_r7",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c0625_r4375_a500_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c0625_r4375_a500_jsd1p5_vae_lhat",
                "prospective_a1_robust_weight_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_robust_weight_screen_r7",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.5,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.0625,
                rotating_corrupted_total_weight=0.4375,
            )
        ),
        "a1_rot4_single_chain_robust_c125_r500_a375_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_single_chain_robust_c125_r500_a375_jsd1p5",
                "prospective_a1_raw_attack_success_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_raw_attack_success_screen_r8",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                ),
                stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5",
                "prospective_a1_raw_attack_success_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_raw_attack_success_screen_r8",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_rawsuccess": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_rawsuccess",
                "prospective_a1_raw_attack_success_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_raw_attack_success_screen_r8",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.2,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
                lhat_training_selection=("raw_attack_success", 0.0),
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_equalpn_rawsuccess": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_equalpn_rawsuccess",
                "prospective_a1_raw_attack_success_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_raw_attack_success_screen_r8",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.2,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
                lhat_training_selection=("raw_attack_success", 0.0),
            )
        ),
        "a1_rot4_single_chain_robust_c125_r500_a375_mixed_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_single_chain_robust_c125_r500_a375_mixed_jsd1p5",
                "prospective_a1_mixed_view_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_mixed_view_screen_r10",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                ),
                stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5",
                "prospective_a1_mixed_view_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_mixed_view_screen_r10",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5_vae_lhat",
                "prospective_a1_mixed_view_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_mixed_view_screen_r10",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_single_chain_robust_c125_r500_a375_endpointmean_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_single_chain_robust_c125_r500_a375_endpointmean_jsd1p5",
                "prospective_a1_endpoint_mean_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_endpoint_mean_screen_r11",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                ),
                stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5",
                "prospective_a1_endpoint_mean_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_endpoint_mean_screen_r11",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat",
                "prospective_a1_endpoint_mean_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_endpoint_mean_screen_r11",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat_cosdecay": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat_cosdecay",
                "prospective_a1_vae_curriculum_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_endpoint_mean_screen_r11",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
                lhat_auxiliary_schedule="cosine_decay_to_zero",
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat_boundary": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat_boundary",
                "prospective_a1_vae_boundary_outside_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_endpoint_mean_screen_r11",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nearest_boundary_outside_grid_v4",
                augmix_jsd_weight=1.5,
                lhat_auxiliary_alpha_max=0.2,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_mean",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_single_chain_robust_c125_r500_a375_halfendpoint_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_single_chain_robust_c125_r500_a375_halfendpoint_jsd1p5",
                "prospective_a1_half_endpoint_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_half_endpoint_screen_r12",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                ),
                stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_half",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.NOT_APPLICABLE,
                "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5",
                "prospective_a1_half_endpoint_screen",
                _ROT4_AUGMIX_RESOURCES,
                "a1_half_endpoint_screen_r12",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_half",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5_vae_lhat": (
            _RecipeDefinition(
                RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
                AuxiliaryVariant.CONTRACTED_LHAT,
                "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5_vae_lhat",
                "prospective_a1_half_endpoint_screen",
                _ROT4_JOINT_MILD_RESOURCES,
                "a1_half_endpoint_screen_r12",
                (
                    "clean_view",
                    "corrupted_view",
                    "augmix_view",
                    "augmix_chain1_view",
                    "augmix_chain2_view",
                    "lhat_view",
                ),
                stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
                lhat_contract_version="nondecreasing_pure_delta_grid_v3",
                augmix_jsd_weight=1.5,
                augmix_auxiliary_weight=0.375,
                augmix_supervised_view_policy="mixed_plus_chains_half",
                rotating_clean_weight=0.125,
                rotating_corrupted_total_weight=0.5,
            )
        ),
        "a1_rot4_vae_lhat_hardgain": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_vae_lhat_hardgain",
            "prospective_hardgain_mechanism_screen",
            _ROT4_LHAT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "lhat_view"),
        ),
        "a1_rot4_augmix_vae_lhat_hardgain": _RecipeDefinition(
            RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "a1_rot4_augmix_vae_lhat_hardgain",
            "prospective_hardgain_mechanism_screen",
            _ROT4_JOINT_MILD_RESOURCES,
            "augmix_simclr_lhat",
            ("clean_view", "corrupted_view", "augmix_view", "lhat_view"),
        ),
        "one_stage_augmix_supervised": _RecipeDefinition(
            RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "one_stage_augmix_supervised",
            "prospective_one_stage_ablation",
            _ONE_STAGE_AUGMIX_RESOURCES,
            "one_stage_augmix_lhat_mild_r0",
            ("clean_view", "augmix_view"),
        ),
        "one_stage_single_chain_supervised": _RecipeDefinition(
            RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "one_stage_single_chain_supervised",
            "prospective_one_stage_chain_ablation",
            _ONE_STAGE_AUGMIX_RESOURCES,
            "one_stage_augmix_lhat_mild_r0",
            ("clean_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_corruption_view",
        ),
        "one_stage_vae_lhat_mild": _RecipeDefinition(
            RecipeKind.ONE_STAGE_VAE_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "one_stage_vae_lhat_mild",
            "prospective_one_stage_ablation",
            _ONE_STAGE_VAE_RESOURCES,
            "one_stage_augmix_lhat_mild_r0",
            ("clean_view", "lhat_view"),
        ),
        "one_stage_augmix_vae_lhat_mild": _RecipeDefinition(
            RecipeKind.ONE_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "one_stage_augmix_vae_lhat_mild",
            "prospective_one_stage_ablation",
            _ONE_STAGE_JOINT_RESOURCES,
            "one_stage_augmix_lhat_mild_r0",
            ("clean_view", "augmix_view", "lhat_view"),
        ),
        "one_stage_single_chain_vae_lhat_mild": _RecipeDefinition(
            RecipeKind.ONE_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "one_stage_single_chain_vae_lhat_mild",
            "prospective_one_stage_chain_ablation",
            _ONE_STAGE_JOINT_RESOURCES,
            "one_stage_augmix_lhat_mild_r0",
            ("clean_view", "augmix_view", "lhat_view"),
            stage1_view="clean_vs_one_single_chain_corruption_view",
        ),
        "one_pool_augmix_jsd": _RecipeDefinition(
            RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "one_pool_augmix_jsd",
            "prospective_one_pool_mechanism_screen",
            _ONE_STAGE_AUGMIX_RESOURCES,
            "one_pool_augmix_lhat_r2",
            (
                "clean_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            augmix_jsd_weight=12.0,
        ),
        "one_pool_single_chain_jsd": _RecipeDefinition(
            RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
            AuxiliaryVariant.NOT_APPLICABLE,
            "one_pool_single_chain_jsd",
            "prospective_one_pool_chain_ablation",
            _ONE_STAGE_AUGMIX_RESOURCES,
            "one_pool_augmix_lhat_r2",
            ("clean_view", "augmix_view"),
            stage1_view="clean_vs_one_single_chain_bernoulli_jsd",
            augmix_jsd_weight=12.0,
        ),
        "one_pool_vae_lhat_puredelta": _RecipeDefinition(
            RecipeKind.ONE_STAGE_VAE_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "one_pool_vae_lhat_puredelta",
            "prospective_one_pool_mechanism_screen",
            _ONE_STAGE_VAE_RESOURCES,
            "one_pool_augmix_lhat_r2",
            ("clean_view", "lhat_view"),
            lhat_contract_version="nondecreasing_pure_delta_grid_v3",
            one_stage_lhat_alpha_max=0.1,
        ),
        "one_pool_augmix_jsd_vae_lhat_puredelta": _RecipeDefinition(
            RecipeKind.ONE_STAGE_AUGMIX_LHAT,
            AuxiliaryVariant.CONTRACTED_LHAT,
            "one_pool_augmix_jsd_vae_lhat_puredelta",
            "prospective_one_pool_mechanism_screen",
            _ONE_STAGE_JOINT_RESOURCES,
            "one_pool_augmix_lhat_r2",
            (
                "clean_view",
                "augmix_view",
                "augmix_chain1_view",
                "augmix_chain2_view",
                "lhat_view",
            ),
            stage1_view="clean_vs_two_chain_augmix_bernoulli_jsd",
            lhat_contract_version="nondecreasing_pure_delta_grid_v3",
            augmix_jsd_weight=12.0,
            one_stage_lhat_alpha_max=0.1,
        ),
    }
)


def _execution_contract(definition: _RecipeDefinition) -> Mapping[str, Any]:
    kind = definition.kind
    variant = definition.auxiliary_variant
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
    if kind is RecipeKind.SUPERVISED_ROTATING_DEPTH23:
        return MappingProxyType(
            {
                **common,
                "stages": ["supervised_adaptation"],
                "exposure_policy": "clean_once_then_rotating_depth23_2plus2",
                "corruption_depths": [2, 3],
                "rotating4_schedule": (
                    "epoch_modulo_five_covers_all_depth23_compositions"
                ),
                "family_loss_weights": {
                    "clean": 0.5,
                    "corrupted_total": 0.5,
                    "corrupted_per_composition": 0.125,
                },
                "generated_view_count": 4,
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    if kind is RecipeKind.SUPERVISED_ROTATING_DEPTH23_LHAT:
        return MappingProxyType(
            {
                **common,
                "stages": ["supervised_adaptation"],
                "stage2_teacher": "disabled",
                "stage2_supervised_logit_anchor_weight_by_backbone": {
                    "efficientnet1dv2": 0.0,
                    "ecgfounder": 0.0,
                },
                "exposure_policy": "clean_aux_once_then_rotating_depth23_2plus2",
                "corruption_depths": [2, 3],
                "rotating4_schedule": (
                    "epoch_modulo_five_covers_all_depth23_compositions"
                ),
                "family_loss_weights": {
                    "clean": 0.5,
                    "corrupted_total": 0.5,
                    "corrupted_per_composition": 0.125,
                },
                "auxiliary": {
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
                },
                "generated_view_count": 5,
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    if kind in {
        RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
        RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX,
    }:
        hardgain = kind is RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX
        uses_augmix = "augmix_view" in definition.output_names
        uses_lhat = "lhat_view" in definition.output_names
        single_chain = definition.stage1_view.startswith("clean_vs_one_single_chain")
        augmix_consistency = definition.augmix_jsd_weight > 0.0
        augmix_multiview = (
            augmix_consistency
            or definition.augmix_supervised_view_policy != "mixed_only"
        )
        if not (uses_augmix or uses_lhat):
            raise AssertionError("mild auxiliary recipe must enable a component")
        clean_weight = (
            0.5 * float(definition.rotating_base_scale)
            if definition.rotating_clean_weight is None
            else float(definition.rotating_clean_weight)
        )
        corrupted_total_weight = (
            0.5 * float(definition.rotating_base_scale)
            if definition.rotating_corrupted_total_weight is None
            else float(definition.rotating_corrupted_total_weight)
        )
        if clean_weight < 0.0 or corrupted_total_weight < 0.0:
            raise AssertionError("rotating family weights must be non-negative")
        if definition.lhat_auxiliary_warmup_epochs < 1:
            raise AssertionError("LHAT warmup must be a positive integer")
        if definition.lhat_loss_integration not in {
            "additive",
            "replace_clean_with_lhat_or_clean_fallback",
        }:
            raise AssertionError("unsupported LHAT loss integration")
        if (
            definition.lhat_loss_integration
            == "replace_clean_with_lhat_or_clean_fallback"
            and (not uses_lhat or definition.lhat_auxiliary_alpha_max > clean_weight)
        ):
            raise AssertionError(
                "clean-replacement LHAT mass must fit inside the clean family"
            )
        return MappingProxyType(
            {
                **common,
                "stages": ["joint_supervised_adaptation"],
                "stage_boundaries": False,
                "simclr": "disabled",
                "projector": "disabled",
                "source_logit_anchor": "disabled",
                "stage2_teacher": "disabled",
                "stage2_supervised_logit_anchor_weight_by_backbone": {
                    "efficientnet1dv2": 0.0,
                    "ecgfounder": 0.0,
                },
                "exposure_policy": (
                    "a1_rotating4_plus_single_chain_and_lhat_auxiliaries"
                    if single_chain and uses_lhat
                    else "a1_rotating4_plus_single_chain_auxiliary"
                    if single_chain
                    else "a1_rotating4_plus_augmix_and_lhat_auxiliaries"
                    if uses_augmix and uses_lhat
                    else (
                        "a1_rotating4_plus_augmix_auxiliary"
                        if uses_augmix
                        else "a1_rotating4_plus_lhat_auxiliary"
                    )
                ),
                "corruption_depths": [2, 3],
                "rotating4_schedule": (
                    "epoch_modulo_five_covers_all_depth23_compositions"
                ),
                "family_loss_weights": {
                    "clean": clean_weight,
                    "corrupted_total": corrupted_total_weight,
                    "corrupted_per_composition": corrupted_total_weight / 4.0,
                },
                "augmix_auxiliary": (
                    {
                        "objective_terms": [
                            "augmix_bce",
                            *(
                                [
                                    _OBJECTIVE_NAME_BY_VIEW[view]
                                    for view in definition.output_names
                                    if view.startswith("augmix_chain")
                                ]
                                if augmix_multiview
                                else []
                            ),
                        ],
                        "weight": float(definition.augmix_auxiliary_weight),
                        "view_geometry": (
                            (
                                "one_depth23_corruption_chain_with_clean_"
                                "bernoulli_jsd"
                                if augmix_consistency
                                else "one_depth23_corruption_chain_without_mix"
                            )
                            if single_chain
                            else (
                                (
                                    "two_independent_depth23_chains_dirichlet_"
                                    "mix_with_clean_chain_bernoulli_jsd"
                                )
                                if augmix_consistency
                                else (
                                    "two_independent_depth23_chains_"
                                    "supervised_mean_without_jsd"
                                )
                                if definition.augmix_supervised_view_policy == "chains_mean"
                                else (
                                    "two_independent_depth23_chains_dirichlet_mix_"
                                    "without_clean_beta"
                                )
                            )
                        ),
                        **(
                            {
                                "bernoulli_jsd_weight": float(
                                    definition.augmix_jsd_weight
                                )
                            }
                            if augmix_consistency
                            else {}
                        ),
                        **(
                            {
                                "supervised_view_policy": (
                                    definition.augmix_supervised_view_policy
                                )
                            }
                            if definition.augmix_supervised_view_policy != "mixed_only"
                            else {}
                        ),
                        "batch_norm_policy": "zero_momentum",
                        "global_rng_policy": "snapshot_restore",
                    }
                    if uses_augmix
                    else None
                ),
                "lhat_auxiliary": (
                    {
                        "objective_terms": ["lhat_direct_bce"],
                        "alpha_max": (
                            0.25
                            if hardgain
                            else float(definition.lhat_auxiliary_alpha_max)
                        ),
                        "linear_warmup_epochs": int(
                            definition.lhat_auxiliary_warmup_epochs
                        ),
                        **(
                            {
                                "loss_integration": definition.lhat_loss_integration,
                                "replacement_source": "clean_bce",
                                "rejection_fallback": "clean_bce",
                                "batch_norm_reference": (
                                    "pre_replacement_base_family_weights"
                                ),
                            }
                            if definition.lhat_loss_integration
                            == "replace_clean_with_lhat_or_clean_fallback"
                            else {}
                        ),
                        **(
                            {"after_warmup_schedule": definition.lhat_auxiliary_schedule}
                            if definition.lhat_auxiliary_schedule
                            != "constant_after_warmup"
                            else {}
                        ),
                        **(
                            {"training_selection": {
                                "mode": "minimum_bce_gain",
                                "minimum_bce_gain": 0.01,
                            }}
                            if hardgain
                            else {
                                "training_selection": {
                                    "mode": definition.lhat_training_selection[0],
                                    "minimum_bce_gain": definition.lhat_training_selection[1],
                                }
                            }
                            if definition.lhat_training_selection is not None
                            else {}
                        ),
                        "gradient_merge": "direct_sum",
                        "batch_norm_policy": "snapshot_restore",
                        "global_rng_policy": "snapshot_restore",
                        "attack_then_contract_version": (
                            definition.lhat_contract_version
                        ),
                        **(
                            {"candidate_source_policy": definition.lhat_candidate_source_policy}
                            if definition.lhat_candidate_source_policy != "clean_only"
                            else {}
                        ),
                        **(
                            {"contract_enabled": False}
                            if definition.lhat_contract_version
                            == "raw_attack_no_contract_v1"
                            else {}
                        ),
                        "diagnostic_scopes": (
                            ["raw_all_candidate_eligible"]
                            if definition.lhat_contract_version
                            == "raw_attack_no_contract_v1"
                            else [
                                "raw_all_candidate_eligible",
                                "contract_all_candidate_eligible",
                                "contract_training_accepted",
                            ]
                        ),
                    }
                    if uses_lhat
                    else None
                ),
                "generated_view_count": (
                    4
                    + sum(name.startswith("augmix") for name in definition.output_names)
                    + int(uses_lhat)
                ),
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
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
    if kind is RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX:
        single_chain = definition.stage1_view.startswith("clean_vs_one_single_chain")
        augmix_consistency = definition.augmix_jsd_weight > 0.0
        return MappingProxyType(
            {
                **common,
                "stages": ["joint_supervised_adaptation"],
                "stage_boundaries": False,
                "simclr": "disabled",
                "projector": "disabled",
                "source_logit_anchor": "disabled",
                "exposure_policy": (
                    "clean_once_plus_one_single_chain_corruption_view"
                    if single_chain
                    else "clean_once_plus_one_twochain_augmix_view"
                ),
                "family_loss_weights": {"clean": 0.5, "augmix": 0.5},
                **(
                    {
                        "augmix_objective": {
                            "objective_terms": [
                                "augmix_bce",
                                *[
                                    _OBJECTIVE_NAME_BY_VIEW[view]
                                    for view in definition.output_names
                                    if view.startswith("augmix_chain")
                                ],
                            ],
                            "bernoulli_jsd_weight": float(
                                definition.augmix_jsd_weight
                            ),
                        }
                    }
                    if augmix_consistency
                    else {}
                ),
                "generated_view_count": (
                    sum(name.startswith("augmix") for name in definition.output_names)
                    if augmix_consistency
                    else 1
                ),
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    if kind in {RecipeKind.ONE_STAGE_VAE_LHAT, RecipeKind.ONE_STAGE_AUGMIX_LHAT}:
        joint = kind is RecipeKind.ONE_STAGE_AUGMIX_LHAT
        single_chain = definition.stage1_view.startswith("clean_vs_one_single_chain")
        augmix_consistency = joint and definition.augmix_jsd_weight > 0.0
        return MappingProxyType(
            {
                **common,
                "stages": ["joint_supervised_adaptation"],
                "stage_boundaries": False,
                "simclr": "disabled",
                "projector": "disabled",
                "source_logit_anchor": "disabled",
                "stage2_teacher": "disabled",
                "stage2_supervised_logit_anchor_weight_by_backbone": {
                    "efficientnet1dv2": 0.0,
                    "ecgfounder": 0.0,
                },
                "exposure_policy": (
                    (
                        "clean_once_plus_one_single_chain_corruption_view_plus_lhat_auxiliary"
                        if single_chain
                        else "clean_once_plus_one_twochain_augmix_view_plus_lhat_auxiliary"
                    )
                    if joint
                    else "clean_once_plus_lhat_auxiliary"
                ),
                "family_loss_weights": (
                    {"clean": 0.5, "augmix": 0.5}
                    if joint
                    else {"clean": 1.0}
                ),
                **(
                    {
                        "augmix_objective": {
                            "objective_terms": [
                                "augmix_bce",
                                *[
                                    _OBJECTIVE_NAME_BY_VIEW[view]
                                    for view in definition.output_names
                                    if view.startswith("augmix_chain")
                                ],
                            ],
                            "bernoulli_jsd_weight": float(
                                definition.augmix_jsd_weight
                            ),
                        }
                    }
                    if augmix_consistency
                    else {}
                ),
                "auxiliary": {
                    "objective_terms": ["lhat_direct_bce"],
                    "alpha_max": float(definition.one_stage_lhat_alpha_max),
                    "linear_warmup_epochs": 5,
                    "gradient_merge": "direct_sum",
                    "batch_norm_policy": "snapshot_restore",
                    "global_rng_policy": "snapshot_restore",
                    "attack_then_contract_version": definition.lhat_contract_version,
                    "diagnostic_scopes": [
                        "raw_all_candidate_eligible",
                        "contract_all_candidate_eligible",
                        "contract_training_accepted",
                    ],
                },
                "generated_view_count": (
                    sum(name.startswith("augmix") for name in definition.output_names) + 1
                    if augmix_consistency
                    else 2 if joint else 1
                ),
                "batch_norm_policy": "family_loss_weighted_once_per_base_batch",
            }
        )
    if kind is RecipeKind.TWO_STAGE_AUGMIX_LHAT:
        contracted = variant is AuxiliaryVariant.CONTRACTED_LHAT
        if definition.stage1_objective is Stage1Objective.SIMCLR:
            stage1_name = "augmix_simclr"
            stage1_contract = {
                "view": definition.stage1_view,
                "objective": "simclr",
                "temperature": 0.5,
                "classifier_head_trainable": False,
                "pretrain_logit_anchor_weight": 5.0,
                "pretrain_logit_anchor_class_weights": [1.0] * 5,
                "ptbxl_source_replay_weight": 0.0,
                "vicreg_weight": 0.0,
                "vae_lhat_tail_fraction": 0.0,
            }
        elif definition.stage1_objective is Stage1Objective.SUPERVISED_AUGMIX:
            stage1_name = "augmix_supervised"
            stage1_contract = {
                "view": definition.stage1_view,
                "objective": "supervised_augmix",
                "objective_weights": {
                    "clean_bce": 0.5,
                    "strong_view_bce": 0.5,
                },
                "classifier_head_trainable": False,
                "pretrain_logit_anchor_weight": 5.0,
                "pretrain_logit_anchor_class_weights": [1.0] * 5,
                "ptbxl_source_replay_weight": 0.0,
                "vicreg_weight": 0.0,
                "vae_lhat_tail_fraction": 0.0,
            }
        elif definition.stage1_objective in {Stage1Objective.CLEAN_BCE, Stage1Objective.CLEAN_BCE_JSD}:
            stage1_name = definition.stage1_objective.value
            stage1_contract = {
                "view": "clean_plus_two_independent_locked_augmix_strong_views",
                "objective": definition.stage1_objective.value,
                "objective_weights": {"clean_bce": 1.0,
                    "bernoulli_jsd": 12.0 if definition.stage1_objective is Stage1Objective.CLEAN_BCE_JSD else 0.0},
                "jsd_reduction": "mean_over_three_views_records_five_independent_binary_labels",
                "classifier_head_trainable": False,
                "pretrain_logit_anchor_weight": 5.0,
                "pretrain_logit_anchor_class_weights": [1.0] * 5,
                "ptbxl_source_replay_weight": 0.0, "vicreg_weight": 0.0,
                "vae_lhat_tail_fraction": 0.0,
                "model_views_per_record": 3,
                "control_keeps_all_three_forward_passes": True,
            }
        else:
            raise AssertionError("two-stage recipes require an explicit Stage-1 objective")
        return MappingProxyType(
            {
                **common,
                "stages": [stage1_name, "supervised_adaptation"],
                "stage1": stage1_contract,
                "stage2_teacher": "disabled",
                "stage2_supervised_logit_anchor_weight_by_backbone": {
                    "efficientnet1dv2": 0.0,
                    "ecgfounder": 0.0,
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
    def stage1_objective(self) -> Stage1Objective:
        return self._definition.stage1_objective

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
            "stage1_objective": self.stage1_objective.value,
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
            "objective_terms": _objective_descriptions(
                self.output_names,
                augmix_jsd_weight=self._definition.augmix_jsd_weight,
                augmix_supervised_view_policy=(
                    self._definition.augmix_supervised_view_policy
                ),
            ),
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
    scientific_contract = _execution_contract(definition)
    identity_payload = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "recipe_version": RECIPE_VERSION,
        "implementation_identity": RECIPE_IMPLEMENTATION_IDENTITY,
        "declaration": root,
        "comparison_rng_identity": definition.comparison_rng_identity,
        "outputs": list(definition.output_names),
        "requirements": list(_requirement_names(definition.output_names)),
        "objective_terms": _objective_descriptions(
            definition.output_names,
            augmix_jsd_weight=definition.augmix_jsd_weight,
            augmix_supervised_view_policy=(
                definition.augmix_supervised_view_policy
            ),
        ),
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
    "RecipeKind",
    "Stage1Objective",
    "load_recipe_spec",
]
