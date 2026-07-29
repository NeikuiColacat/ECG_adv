"""Scoped fixed20 + LHAT/AugMix auxiliary-gradient sandbox.

This adapter leaves the locked Direct+fixed20 objective, exposure order and
optimizer-step budget intact.  Once per base batch it additionally computes a
low-dose LHAT/three-chain AugMix objective.  Legacy profiles project the
auxiliary gradient away from the fixed20 gradient only when the two conflict.
The direct-sum ablation keeps the same auxiliary objective and optimizer
budget but adds its gradient without conflict projection.
The residualized profile always removes the component parallel to fixed20,
applies the whitelist trainer's Direct-identical clip to the base gradient,
then caps the remaining auxiliary norm relative to that clipped base.  It does
not clip the combined gradient a second time because doing so would silently
shrink the preserved Direct component.  Legacy profiles retain their original
combined-gradient clip semantics.

The patch is deliberately process-local and fail-closed on preregistered
method ids.  It does not alter whitelist source files.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.augmix import AugMixBatch, AugMixConfig, generate_three_chain_augmix
from core.methods import (
    CompiledMethod,
    MethodViewRuntime,
    Provenance,
    ViewValue,
    WaveformView,
    compile_method_profile,
)
from core.methods.contracts import NodeContext
from util.random_seed import derive_seed


SEARCH_METHOD_ID = "fixed20_search_d19_v1"
RESIDUALIZED_METHOD_ID = "fixed20_residualized_augmix_aux_v1"
VAE_POST_METHOD_ID = "fixed20_pcgrad_vae_lhat_post_refine_v1"
HARD_ONLY_PRUNED_METHOD_ID = (
    "fixed20_directsum_vae_lhat_hard_only_pruned_v1"
)
GRADIENT_BALANCED_METHOD_ID = (
    "fixed20_pcgrad_gradbalanced_strong_lhat_augmix_v1"
)
METHOD_ALPHA = {
    "fixed20_pcgrad_lhat_augmix_a025_v1": 0.25,
    "fixed20_pcgrad_lhat_augmix_a050_v1": 0.50,
    "fixed20_pcgrad_lhat_augmix_calibrate_v1": 0.25,
    "fixed20_pcgrad_strong_lhat_augmix_v1": 0.25,
    "fixed20_pcgrad_strong_lhat_direct_augmix_v1": 0.25,
    "fixed20_asl_pcgrad_strong_lhat_augmix_v1": 0.25,
    "fixed20_pcgrad_lhat_poshide_augmix_v1": 0.25,
    "fixed20_pcgrad_lhat_balanced_augmix_v1": 0.25,
    "fixed20_pcgrad_teacher04_lhat_augmix_v1": 0.25,
    "fixed20_pcgrad_teacher04_lhat_direct_augmix_v1": 0.25,
    "fixed20_pcgrad_d13_aux_a050_v1": 0.50,
    "fixed20_pcgrad_d13_aux_a100_v1": 1.00,
    "fixed20_pcgrad_d19_aux_a050_v1": 0.50,
    "fixed20_groupdro_pcgrad_d19_aux_a050_v1": 0.50,
    "fixed20_groupdro_pcgrad_d19_supcon_a050_v1": 0.50,
    "fixed20_pairrank_pcgrad_d13_aux_a050_v1": 0.50,
    "fixed20_pairrank_pcgrad_d19_supcon010_a050_v1": 0.50,
    "fixed20_pairrank_groupdro_pcgrad_d19_aux_a050_v1": 0.50,
    "fixed20_balanced_pairrank_pcgrad_d19_aux_a050_v1": 0.50,
    SEARCH_METHOD_ID: 0.50,
    RESIDUALIZED_METHOD_ID: 1.00,
    VAE_POST_METHOD_ID: 0.25,
    HARD_ONLY_PRUNED_METHOD_ID: 1.50,
    GRADIENT_BALANCED_METHOD_ID: 0.75,
}
METHOD_IDS = frozenset(METHOD_ALPHA)
PCGRAD_MODE = "fixed20_base_plus_projected_lhat_augmix_aux_v1"
VAE_POST_MODE = "fixed20_base_plus_projected_vae_lhat_aux_v1"
DIRECT_SUM_MODE = "fixed20_base_plus_direct_lhat_augmix_aux_v1"
RESIDUALIZED_MODE = "fixed20_base_plus_residualized_augmix_aux_v1"
RESIDUALIZED_PROJECTION_POLICY = (
    "direct_clip_base_then_remove_full_base_parallel_aux_then_"
    "clipped_base_norm_cap_no_second_clip"
)
RESIDUALIZED_FINAL_NORM_POLICY = (
    "l2_bound_sqrt_one_plus_rho_squared_times_direct_clip"
)
RESIDUALIZED_CANDIDATE_SEED_POLICY = (
    "sha256_base_seed_shared_node_namespace_execution_identity_"
    "without_candidate_profile_v1"
)
RESIDUALIZED_AUGMIX_BATCH_POLICY = (
    "full_base_batch_common_random_numbers_before_hard_view_validity_mask"
)
RESIDUALIZED_AVAILABILITY_POLICY = (
    "method_specific_hard_view_validity_clean_fallback_excluded_aux_bce"
)
BASE_EXPOSURE_POLICY = "clean_once_then_exhaustive_depth23"
AUXILIARY_EXPOSURE_POLICY = "clean_aux_once_then_exhaustive_depth23"
ROTATING4_EXPOSURE_POLICY = "clean_aux_once_then_rotating_depth23_2plus2"
BASE_METHOD_ID = "direct_depth23_fixed20"
BASE_TERMS = ("clean_bce", "corrupted_bce")
DEFAULT_AUXILIARY_TERMS = ("augmix_bce", "clean_lhat_augmix_jsd")
DIRECT_LHAT_AUXILIARY_TERMS = (
    "lhat_direct_bce",
    "augmix_bce",
    "clean_lhat_augmix_jsd",
)
VAE_POST_AUXILIARY_TERMS = (
    "lhat_direct_bce",
    "clean_lhat_jsd",
)
VAE_POST_METHOD_IDS = frozenset({VAE_POST_METHOD_ID})
HARD_ONLY_PRUNED_METHOD_IDS = frozenset({HARD_ONLY_PRUNED_METHOD_ID})
DIRECT_LHAT_METHOD_IDS = frozenset(
    {
        "fixed20_pcgrad_strong_lhat_direct_augmix_v1",
        "fixed20_pcgrad_teacher04_lhat_direct_augmix_v1",
    }
)
D13_AUXILIARY_TERMS = (
    "corruption_1_bce",
    "corruption_2_bce",
    "corruption_3_bce",
    "corruption_4_bce",
    "compat_random_bce",
    "compat_hard_bce",
    "clean_raw4_compat_random_hard_jsd",
)
D13_AUXILIARY_METHOD_IDS = frozenset(
    {
        "fixed20_pcgrad_d13_aux_a050_v1",
        "fixed20_pcgrad_d13_aux_a100_v1",
        "fixed20_pairrank_pcgrad_d13_aux_a050_v1",
    }
)
D13_RUNTIME_ALIAS = "diverse_augmax_d13_compat_random_hard_raw40_jsd30_r4_v1"
D13_CANDIDATE_PROFILE = f"train/methods/{D13_RUNTIME_ALIAS}.yaml"
D19_AUXILIARY_TERMS = (
    "corruption_1_bce",
    "compat_random_bce",
    "compat_hard_bce",
    "clean_raw_random_hard_jsd",
)
AUXILIARY_TERM_SCALE_CONTRACT = "pcgrad_auxiliary_term_scales"
AUXILIARY_BCE_MASS_CONTRACT = "pcgrad_auxiliary_bce_mass_preserved"
D19_AUXILIARY_BCE_WEIGHTS = {
    "corruption_1_bce": 0.4,
    "compat_random_bce": 0.3,
    "compat_hard_bce": 0.3,
}
D19_AUXILIARY_METHOD_IDS = frozenset(
    {
        "fixed20_pcgrad_d19_aux_a050_v1",
        "fixed20_groupdro_pcgrad_d19_aux_a050_v1",
        "fixed20_groupdro_pcgrad_d19_supcon_a050_v1",
        "fixed20_pairrank_pcgrad_d19_supcon010_a050_v1",
        "fixed20_pairrank_groupdro_pcgrad_d19_aux_a050_v1",
        "fixed20_balanced_pairrank_pcgrad_d19_aux_a050_v1",
        SEARCH_METHOD_ID,
    }
)
GROUP_DRO_METHOD_IDS = frozenset(
    {
        "fixed20_groupdro_pcgrad_d19_aux_a050_v1",
        "fixed20_groupdro_pcgrad_d19_supcon_a050_v1",
        "fixed20_pairrank_groupdro_pcgrad_d19_aux_a050_v1",
        SEARCH_METHOD_ID,
    }
)
GROUP_DRO_MODE = "online_exponentiated_complete_cycle_v1"
GROUP_DRO_STEP_SIZE = 0.05
GROUP_DRO_UNIFORM_MIX = 0.25
GROUP_DRO_GROUP_COUNT = 20
GROUP_DRO_LOSS_SOURCE = "detached_corrupted_macro_balanced_bce"
PAIRRANK_METHOD_IDS = frozenset(
    {
        "fixed20_pairrank_pcgrad_d13_aux_a050_v1",
        "fixed20_pairrank_pcgrad_d19_supcon010_a050_v1",
        "fixed20_pairrank_groupdro_pcgrad_d19_aux_a050_v1",
        "fixed20_balanced_pairrank_pcgrad_d19_aux_a050_v1",
        SEARCH_METHOD_ID,
    }
)
PAIRRANK_LOSS_ID = "bce_plus_multilabel_pairwise_logistic_v1"
PAIRRANK_WEIGHT = 0.10
PAIRRANK_TEMPERATURE = 1.0
PAIRRANK_REDUCTION = "mean_present_classes_then_all_positive_negative_pairs"
BALANCED_BCE_METHOD_IDS = frozenset(
    {"fixed20_balanced_pairrank_pcgrad_d19_aux_a050_v1"}
)
BALANCED_BCE_LOSS_ID = "soft_class_polarity_balanced_bce_with_logits_v1"
BALANCED_BCE_SCOPE = "corrupted_fixed20_and_auxiliary_bce_views"
BALANCED_BCE_AUXILIARY_SCOPE = "auxiliary_bce_views"
BALANCED_BCE_REDUCTION = "mean_classes_then_equal_positive_negative_mass"
SUPCON_METHOD_IDS = frozenset(
    {
        "fixed20_groupdro_pcgrad_d19_supcon_a050_v1",
        "fixed20_pairrank_pcgrad_d19_supcon010_a050_v1",
        SEARCH_METHOD_ID,
    }
)
SUPCON_LOSS_ID = "stopgrad_clean_key_exact_label_multiview_infonce_v1"
MULSUPCON_LOSS_ID = "stopgrad_clean_key_labelwise_mulsupcon_queue_v1"
SUPPORTED_SUPCON_LOSS_IDS = frozenset({SUPCON_LOSS_ID, MULSUPCON_LOSS_ID})
SUPCON_WEIGHT_BY_METHOD = {
    "fixed20_groupdro_pcgrad_d19_supcon_a050_v1": 0.20,
    "fixed20_pairrank_pcgrad_d19_supcon010_a050_v1": 0.10,
    SEARCH_METHOD_ID: 0.20,
}
SUPCON_TEMPERATURE = 0.20
SUPCON_QUERY_VIEWS = (
    "corruption_view_1",
    "compat_random_view",
    "compat_hard_view",
)
SUPCON_CAPTURE_ORDER = (*SUPCON_QUERY_VIEWS, "clean_view")
MANIFOLD_MIXUP_POLICY = "semantic_compatible_best_in_batch_v1"
FEATURE_PATHMIX_QUERY_VIEWS = SUPCON_QUERY_VIEWS
CROSSBATCH_PAIRRANK_SCOPE = "clean_base_and_d19_bce_queries_v1"
CROSSBATCH_MEMORY_UPDATE = "clean_only_fifo_detached_v1"
ROBUST_VIEW_PAIRRANK_LOSS_ID = (
    "class_balanced_worst_clean_d19_view_pairwise_logistic_queue_v1"
)
ROBUST_VIEW_PAIRRANK_SCOPE = (
    "clean_corruption_compat_random_compat_hard_per_record_class_worst_v1"
)
ROBUST_VIEW_PAIRRANK_MEMORY_UPDATE = (
    "detached_per_record_worst_score_fifo_v1"
)
CANDIDATE_REPLAY_ORDER = tuple(
    composition
    for pair in zip(range(10), range(10, 20), strict=True)
    for composition in pair
)
CANDIDATE_REPLAY_POLICY = (
    "replace_balanced_depth2_depth3_fixed_exposures_with_online_d19_v1"
)
D19_RUNTIME_ALIAS = "diverse_augmax_d19_multistart_threechain_feature_v1"
D19_CANDIDATE_PROFILE = f"train/methods/{D19_RUNTIME_ALIAS}.yaml"
RESIDUALIZED_AUXILIARY_METHOD_IDS = frozenset({RESIDUALIZED_METHOD_ID})
NORM_CAPPED_AUXILIARY_METHOD_IDS = frozenset(
    {GRADIENT_BALANCED_METHOD_ID}
)
NORM_CAPPED_PROJECTION_POLICY = (
    "project_conflicts_then_cap_aux_to_base_before_combined_clip_v1"
)
COMPATIBLE_AUXILIARY_METHOD_IDS = D13_AUXILIARY_METHOD_IDS | D19_AUXILIARY_METHOD_IDS
CANDIDATE_PROFILE_BY_METHOD = {
    **{method_id: D13_CANDIDATE_PROFILE for method_id in D13_AUXILIARY_METHOD_IDS},
    **{method_id: D19_CANDIDATE_PROFILE for method_id in D19_AUXILIARY_METHOD_IDS},
}
CANDIDATE_ALIAS_BY_METHOD = {
    **{method_id: D13_RUNTIME_ALIAS for method_id in D13_AUXILIARY_METHOD_IDS},
    **{method_id: D19_RUNTIME_ALIAS for method_id in D19_AUXILIARY_METHOD_IDS},
}
METHOD_AUXILIARY_TERMS = {
    method_id: (
        ("augmix_bce",)
        if method_id in RESIDUALIZED_AUXILIARY_METHOD_IDS
        else VAE_POST_AUXILIARY_TERMS
        if method_id in HARD_ONLY_PRUNED_METHOD_IDS
        else VAE_POST_AUXILIARY_TERMS
        if method_id in VAE_POST_METHOD_IDS
        else D13_AUXILIARY_TERMS
        if method_id in D13_AUXILIARY_METHOD_IDS
        else D19_AUXILIARY_TERMS
        if method_id in D19_AUXILIARY_METHOD_IDS
        else (
            DIRECT_LHAT_AUXILIARY_TERMS
            if method_id in DIRECT_LHAT_METHOD_IDS
            else DEFAULT_AUXILIARY_TERMS
        )
    )
    for method_id in METHOD_IDS
}
EPSILON = 1.0e-12


def _bounded_float(
    contracts: Mapping[str, Any], name: str, *, minimum: float, maximum: float
) -> float:
    value = contracts.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(
            f"search contract {name} must be finite in [{minimum},{maximum}], "
            f"got {value!r}"
        )
    return float(value)


def _runtime_hyperparameters(method: CompiledMethod) -> dict[str, float]:
    """Resolve the process-local search point from the snapshotted profile."""

    if method.profile_name == RESIDUALIZED_METHOD_ID:
        return {
            "alpha": _bounded_float(
                method.contracts,
                "pcgrad_auxiliary_alpha",
                minimum=0.0,
                maximum=2.0,
            ),
            "residual_norm_ratio_cap": _bounded_float(
                method.contracts,
                "pcgrad_residual_norm_ratio_cap",
                minimum=0.0,
                maximum=1.0,
            ),
            "group_dro_step_size": GROUP_DRO_STEP_SIZE,
            "group_dro_uniform_mix": GROUP_DRO_UNIFORM_MIX,
            "pairrank_weight": PAIRRANK_WEIGHT,
            "pairrank_temperature": PAIRRANK_TEMPERATURE,
            "supcon_weight": 0.0,
            "supcon_temperature": SUPCON_TEMPERATURE,
            "clean_mixup_weight": 0.0,
            "clean_mixup_anchor_mass": 0.75,
            "pathmix_weight": 0.0,
            "pathmix_anchor_mass": 0.75,
            "crossbatch_pairrank_weight": 0.0,
            "crossbatch_pairrank_temperature": 1.0,
            "crossbatch_pairrank_capacity": 512.0,
            "crossbatch_pairrank_hard_fraction": 0.25,
            "candidate_replay_count": 0.0,
        }
    if method.profile_name != SEARCH_METHOD_ID:
        return {
            "alpha": METHOD_ALPHA[method.profile_name],
            "residual_norm_ratio_cap": (
                _bounded_float(
                    method.contracts,
                    "pcgrad_auxiliary_norm_ratio_cap",
                    minimum=0.0,
                    maximum=1.0,
                )
                if method.profile_name in NORM_CAPPED_AUXILIARY_METHOD_IDS
                else 0.0
            ),
            "group_dro_step_size": GROUP_DRO_STEP_SIZE,
            "group_dro_uniform_mix": GROUP_DRO_UNIFORM_MIX,
            "pairrank_weight": PAIRRANK_WEIGHT,
            "pairrank_temperature": PAIRRANK_TEMPERATURE,
            "supcon_weight": SUPCON_WEIGHT_BY_METHOD.get(method.profile_name, 0.0),
            "supcon_temperature": SUPCON_TEMPERATURE,
            "clean_mixup_weight": 0.0,
            "clean_mixup_anchor_mass": 0.75,
            "pathmix_weight": 0.0,
            "pathmix_anchor_mass": 0.75,
            "crossbatch_pairrank_weight": 0.0,
            "crossbatch_pairrank_temperature": 1.0,
            "crossbatch_pairrank_capacity": 512.0,
            "crossbatch_pairrank_hard_fraction": 0.25,
            "candidate_replay_count": 0.0,
        }
    contracts = method.contracts
    return {
        "alpha": _bounded_float(
            contracts, "pcgrad_auxiliary_alpha", minimum=0.0, maximum=2.0
        ),
        "residual_norm_ratio_cap": 0.0,
        "group_dro_step_size": _bounded_float(
            contracts, "group_dro_step_size", minimum=0.0, maximum=0.5
        ),
        "group_dro_uniform_mix": _bounded_float(
            contracts, "group_dro_uniform_mix", minimum=0.0, maximum=1.0
        ),
        "pairrank_weight": _bounded_float(
            contracts, "pairrank_weight", minimum=0.0, maximum=2.0
        ),
        "pairrank_temperature": _bounded_float(
            contracts, "pairrank_temperature", minimum=0.1, maximum=5.0
        ),
        "supcon_weight": _bounded_float(
            contracts, "feature_regularizer_weight", minimum=0.0, maximum=1.0
        ),
        "supcon_temperature": _bounded_float(
            contracts, "feature_regularizer_temperature", minimum=0.05, maximum=2.0
        ),
        "clean_mixup_weight": _bounded_float(
            contracts, "clean_manifold_mixup_weight", minimum=0.0, maximum=2.0
        ),
        "clean_mixup_anchor_mass": _bounded_float(
            contracts, "clean_manifold_mixup_anchor_mass", minimum=0.5, maximum=1.0
        ),
        "pathmix_weight": _bounded_float(
            contracts, "feature_pathmix_weight", minimum=0.0, maximum=2.0
        ),
        "pathmix_anchor_mass": _bounded_float(
            contracts, "feature_pathmix_anchor_mass", minimum=0.5, maximum=1.0
        ),
        "crossbatch_pairrank_weight": _bounded_float(
            contracts, "crossbatch_pairrank_weight", minimum=0.0, maximum=5.0
        ),
        "crossbatch_pairrank_temperature": _bounded_float(
            contracts, "crossbatch_pairrank_temperature", minimum=0.05, maximum=5.0
        ),
        "crossbatch_pairrank_capacity": _bounded_float(
            contracts, "crossbatch_pairrank_capacity", minimum=64.0, maximum=2048.0
        ),
        "crossbatch_pairrank_hard_fraction": _bounded_float(
            contracts,
            "crossbatch_pairrank_hard_fraction",
            minimum=0.01,
            maximum=1.0,
        ),
        "robust_view_pairrank_weight": _bounded_float(
            {
                **contracts,
                "robust_view_pairrank_weight": contracts.get(
                    "robust_view_pairrank_weight", 0.0
                ),
            },
            "robust_view_pairrank_weight",
            minimum=0.0,
            maximum=5.0,
        ),
        "robust_view_pairrank_temperature": _bounded_float(
            {
                **contracts,
                "robust_view_pairrank_temperature": contracts.get(
                    "robust_view_pairrank_temperature", 1.0
                ),
            },
            "robust_view_pairrank_temperature",
            minimum=0.05,
            maximum=5.0,
        ),
        "robust_view_pairrank_capacity": _bounded_float(
            {
                **contracts,
                "robust_view_pairrank_capacity": contracts.get(
                    "robust_view_pairrank_capacity", 512
                ),
            },
            "robust_view_pairrank_capacity",
            minimum=64.0,
            maximum=2048.0,
        ),
        "robust_view_pairrank_hard_fraction": _bounded_float(
            {
                **contracts,
                "robust_view_pairrank_hard_fraction": contracts.get(
                    "robust_view_pairrank_hard_fraction", 0.25
                ),
            },
            "robust_view_pairrank_hard_fraction",
            minimum=0.01,
            maximum=1.0,
        ),
        "candidate_replay_count": _bounded_float(
            {
                **contracts,
                "candidate_replay_count": contracts.get(
                    "candidate_replay_count", 0
                ),
            },
            "candidate_replay_count",
            minimum=0.0,
            maximum=20.0,
        ),
    }


def _active_auxiliary_terms(method: CompiledMethod) -> tuple[str, ...]:
    """Return no VAE/multiview terms for the explicit ``alpha=0`` ablation."""

    if _runtime_hyperparameters(method)["alpha"] == 0.0:
        return ()
    return METHOD_AUXILIARY_TERMS[method.profile_name]


def _auxiliary_term_scales(method: CompiledMethod) -> dict[str, float]:
    """Resolve a mass-preserving D19 raw-to-VAE attribution ablation.

    The default is an exact identity.  A configured search point may move BCE
    mass among the one raw and two VAE views, but may not change their combined
    compiled mass.  This keeps the outer auxiliary alpha, JSD term, optimizer
    budget, and generated views fixed while isolating which view family owns
    the supervised pressure.
    """

    raw = method.contracts.get(AUXILIARY_TERM_SCALE_CONTRACT)
    mass_marker = method.contracts.get(AUXILIARY_BCE_MASS_CONTRACT)
    if raw is None:
        if mass_marker is not None:
            raise ValueError(
                f"{AUXILIARY_BCE_MASS_CONTRACT} requires "
                f"{AUXILIARY_TERM_SCALE_CONTRACT}"
            )
        return {name: 1.0 for name in _active_auxiliary_terms(method)}
    if method.profile_name != SEARCH_METHOD_ID:
        raise ValueError(
            "per-term auxiliary scaling is restricted to the search method"
        )
    if not isinstance(raw, Mapping) or set(raw) != set(D19_AUXILIARY_TERMS):
        raise ValueError(
            f"{AUXILIARY_TERM_SCALE_CONTRACT} must cover exactly "
            f"{D19_AUXILIARY_TERMS}"
        )
    scales: dict[str, float] = {}
    for name in D19_AUXILIARY_TERMS:
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 4.0
        ):
            raise ValueError(
                f"{AUXILIARY_TERM_SCALE_CONTRACT}.{name} must lie in [0,4]"
            )
        scales[name] = float(value)
    if mass_marker is not True:
        raise ValueError(
            f"{AUXILIARY_BCE_MASS_CONTRACT} must be true for scaled D19 terms"
        )
    original_mass = sum(D19_AUXILIARY_BCE_WEIGHTS.values())
    scaled_mass = sum(
        weight * scales[name]
        for name, weight in D19_AUXILIARY_BCE_WEIGHTS.items()
    )
    if not math.isclose(
        scaled_mass,
        original_mass,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "scaled D19 auxiliary BCE mass must remain exactly "
            f"{original_mass}, got {scaled_mass}"
        )
    return scales


def _group_dro_active(method: CompiledMethod) -> bool:
    if method.profile_name not in GROUP_DRO_METHOD_IDS:
        return False
    hyperparameters = _runtime_hyperparameters(method)
    return (
        hyperparameters["group_dro_step_size"] > 0.0
        and hyperparameters["group_dro_uniform_mix"] < 1.0
    )


def _runtime_pairrank_controls(
    method: CompiledMethod,
) -> tuple[tuple[float, float, float, float, float], float]:
    if method.profile_name != SEARCH_METHOD_ID:
        return (1.0, 1.0, 1.0, 1.0, 1.0), 1.0
    raw_weights = method.contracts.get("pairrank_class_weights", [1.0] * 5)
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != 5:
        raise ValueError("pairrank_class_weights must contain five values")
    weights = []
    for value in raw_weights:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 5.0
        ):
            raise ValueError("pairrank class weights must be finite in [0,5]")
        weights.append(float(value))
    if not math.isclose(sum(weights), 5.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("pairrank class weights must sum to 5.0")
    hard_fraction = method.contracts.get("pairrank_hard_fraction", 1.0)
    if (
        isinstance(hard_fraction, bool)
        or not isinstance(hard_fraction, (int, float))
        or not math.isfinite(float(hard_fraction))
        or not 0.05 <= float(hard_fraction) <= 1.0
    ):
        raise ValueError("pairrank_hard_fraction must lie in [0.05,1.0]")
    return tuple(weights), float(hard_fraction)  # type: ignore[return-value]


def _balanced_bce_scope(method: CompiledMethod) -> str | None:
    """Resolve the explicit class/polarity-balanced BCE routing contract."""

    if method.profile_name in BALANCED_BCE_METHOD_IDS:
        return BALANCED_BCE_SCOPE
    contracts = method.contracts
    scope = contracts.get("balanced_bce_scope")
    balanced_keys = {
        "balanced_bce_loss",
        "balanced_bce_scope",
        "balanced_bce_reduction",
        "balanced_bce_clean_base_preserved",
    }
    if scope is None:
        stale = sorted(balanced_keys.intersection(contracts))
        if stale:
            raise ValueError(
                "balanced-BCE contract is incomplete without balanced_bce_scope: "
                f"{stale}"
            )
        return None
    if method.profile_name != SEARCH_METHOD_ID:
        raise ValueError(
            "dynamic balanced-BCE routing is restricted to the search method"
        )
    if scope not in {BALANCED_BCE_AUXILIARY_SCOPE, BALANCED_BCE_SCOPE}:
        raise ValueError(f"unsupported balanced-BCE scope {scope!r}")
    return str(scope)


def _same_float(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _is_pcgrad_method(method: Any) -> bool:
    return isinstance(method, CompiledMethod) and (
        method.profile_name in METHOD_IDS
        or "pcgrad_auxiliary_mode" in method.contracts
    )


def validate_pcgrad_method(method: CompiledMethod) -> float:
    """Reject copied markers or drift from the preregistered mechanism."""

    method_id = method.profile_name
    if method_id not in METHOD_IDS:
        raise ValueError(f"PCGrad method is not allowlisted: {method_id!r}")
    hyperparameters = _runtime_hyperparameters(method)
    alpha = hyperparameters["alpha"]
    contracts = method.contracts
    expected_mode = (
        RESIDUALIZED_MODE
        if method_id in RESIDUALIZED_AUXILIARY_METHOD_IDS
        else DIRECT_SUM_MODE
        if method_id in HARD_ONLY_PRUNED_METHOD_IDS
        else VAE_POST_MODE
        if method_id in VAE_POST_METHOD_IDS
        else PCGRAD_MODE
    )
    configured_mode = contracts.get("pcgrad_auxiliary_mode")
    allowed_modes = {expected_mode}
    if method_id == SEARCH_METHOD_ID:
        allowed_modes.add(DIRECT_SUM_MODE)
    if configured_mode not in allowed_modes:
        raise ValueError(f"{method_id} has an invalid PCGrad mode")
    auxiliary_terms = _active_auxiliary_terms(method)
    rotating4 = "rotating4_schedule" in contracts
    if rotating4 and method_id not in HARD_ONLY_PRUNED_METHOD_IDS:
        raise ValueError(
            "rotating4 PCGrad routing is restricted to the pruned hard-only method"
        )
    expected_exposure_policy = (
        ROTATING4_EXPOSURE_POLICY
        if rotating4
        else
        BASE_EXPOSURE_POLICY
        if method_id in VAE_POST_METHOD_IDS
        else
        AUXILIARY_EXPOSURE_POLICY
        if auxiliary_terms
        else BASE_EXPOSURE_POLICY
    )
    if contracts.get("exposure_policy") != expected_exposure_policy:
        raise ValueError(f"{method_id} has an invalid exposure policy")
    if not _same_float(contracts.get("pcgrad_auxiliary_alpha"), alpha):
        raise ValueError(f"{method_id} has an invalid auxiliary alpha")
    if not _same_float(contracts.get("pcgrad_projection_epsilon"), EPSILON):
        raise ValueError(f"{method_id} has an invalid projection epsilon")
    if tuple(contracts.get("pcgrad_auxiliary_terms", ())) != auxiliary_terms:
        raise ValueError(f"{method_id} has invalid auxiliary terms")
    auxiliary_term_scales = _auxiliary_term_scales(method)
    if contracts.get("pcgrad_batch_norm_policy") != "snapshot_restore_auxiliary":
        raise ValueError(f"{method_id} must restore auxiliary BatchNorm state")
    if contracts.get("pcgrad_rng_policy") != "snapshot_restore_global_rng":
        raise ValueError(f"{method_id} must restore the global RNG after auxiliary")
    if method_id in RESIDUALIZED_AUXILIARY_METHOD_IDS:
        ratio_cap = hyperparameters["residual_norm_ratio_cap"]
        if not _same_float(
            contracts.get("pcgrad_residual_norm_ratio_cap"), ratio_cap
        ):
            raise ValueError(f"{method_id} has an invalid residual norm cap")
        if contracts.get("pcgrad_projection_policy") != (
            RESIDUALIZED_PROJECTION_POLICY
        ):
            raise ValueError(f"{method_id} has an invalid residual projection policy")
        if contracts.get("fixed20_base_gradient_preserved_after_direct_clip") is not True:
            raise ValueError(
                f"{method_id} must preserve the Direct-clipped fixed20 gradient"
            )
        if contracts.get("pcgrad_second_combined_clip_applied") is not False:
            raise ValueError(
                f"{method_id} must not second-clip the residualized gradient"
            )
        if contracts.get("pcgrad_final_gradient_norm_policy") != (
            RESIDUALIZED_FINAL_NORM_POLICY
        ):
            raise ValueError(f"{method_id} has an invalid final-gradient norm policy")
        candidate_profile = contracts.get("pcgrad_candidate_profile")
        candidate_method_id = contracts.get("pcgrad_candidate_method_id")
        if (
            not isinstance(candidate_profile, str)
            or not candidate_profile
            or Path(candidate_profile).is_absolute()
        ):
            raise ValueError(f"{method_id} has an invalid candidate profile")
        if not isinstance(candidate_method_id, str) or not candidate_method_id:
            raise ValueError(f"{method_id} has an invalid candidate method id")
        expected_residualized_contracts = {
            "pcgrad_candidate_seed_policy": RESIDUALIZED_CANDIDATE_SEED_POLICY,
            "pcgrad_augmix_batch_policy": RESIDUALIZED_AUGMIX_BATCH_POLICY,
            "pcgrad_candidate_availability_policy": (
                RESIDUALIZED_AVAILABILITY_POLICY
            ),
        }
        for name, expected in expected_residualized_contracts.items():
            if contracts.get(name) != expected:
                raise ValueError(f"{method_id} has an invalid {name}")
    if method_id in NORM_CAPPED_AUXILIARY_METHOD_IDS:
        ratio_cap = hyperparameters["residual_norm_ratio_cap"]
        if not _same_float(
            contracts.get("pcgrad_auxiliary_norm_ratio_cap"), ratio_cap
        ):
            raise ValueError(f"{method_id} has an invalid auxiliary norm cap")
        if contracts.get("pcgrad_auxiliary_norm_cap_policy") != (
            NORM_CAPPED_PROJECTION_POLICY
        ):
            raise ValueError(
                f"{method_id} has an invalid auxiliary norm-cap policy"
            )
    if contracts.get("optimizer_step_policy") != (
        "accumulate_family_balanced_once_per_base_batch"
    ):
        raise ValueError(f"{method_id} changed the fixed20 optimizer-step policy")
    expected_family = (
        {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.125,
        }
        if rotating4
        else {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.025,
        }
    )
    if dict(contracts.get("family_loss_weights", {})) != expected_family:
        raise ValueError(f"{method_id} changed the fixed20 family weights")
    if tuple(contracts.get("composition_indices", ())) != tuple(range(20)):
        raise ValueError(f"{method_id} changed the fixed20 composition order")
    if contracts.get("heldout_target_feedback_allowed") is not False:
        raise ValueError(f"{method_id} must forbid held-out target feedback")
    if method_id in GROUP_DRO_METHOD_IDS:
        if contracts.get("fixed20_base_gradient_preserved") is not False:
            raise ValueError(
                f"{method_id} must disclose adaptive fixed20 gradient weighting"
            )
        if contracts.get("fixed20_base_exposure_preserved") is not True:
            raise ValueError(f"{method_id} must preserve every fixed20 exposure")
        if contracts.get("group_dro_mode") != GROUP_DRO_MODE:
            raise ValueError(f"{method_id} has an invalid GroupDRO mode")
        expected_group_dro_step = hyperparameters["group_dro_step_size"]
        expected_group_dro_uniform = hyperparameters["group_dro_uniform_mix"]
        if not _same_float(
            contracts.get("group_dro_step_size"), expected_group_dro_step
        ):
            raise ValueError(f"{method_id} has an invalid GroupDRO step size")
        if not _same_float(
            contracts.get("group_dro_uniform_mix"), expected_group_dro_uniform
        ):
            raise ValueError(f"{method_id} has an invalid GroupDRO uniform mix")
        if contracts.get("group_dro_update_policy") != (
            "use_current_complete_cycle_for_next_base_batch"
        ):
            raise ValueError(f"{method_id} has an invalid GroupDRO update policy")
        if contracts.get("group_dro_loss_source") != GROUP_DRO_LOSS_SOURCE:
            raise ValueError(f"{method_id} has an invalid GroupDRO loss source")
        if contracts.get("group_dro_total_corrupted_mass_preserved") is not True:
            raise ValueError(
                f"{method_id} must preserve the total corrupted objective mass"
            )
    if method_id in PAIRRANK_METHOD_IDS:
        pairrank_class_weights, pairrank_hard_fraction = _runtime_pairrank_controls(
            method
        )
        if contracts.get("fixed20_base_gradient_preserved") is not False:
            raise ValueError(
                f"{method_id} must disclose the pairwise-ranking base gradient"
            )
        if contracts.get("fixed20_base_exposure_preserved") is not True:
            raise ValueError(f"{method_id} must preserve every fixed20 exposure")
        expected_pairrank = {
            "pairrank_weight": hyperparameters["pairrank_weight"],
            "pairrank_temperature": hyperparameters["pairrank_temperature"],
            "pairrank_reduction": PAIRRANK_REDUCTION,
            "pairrank_target_policy": "ordered_continuous_relevance",
            "pairrank_scope": "all_bce_views_base_and_auxiliary",
            "pairrank_class_weights": list(pairrank_class_weights),
            "pairrank_hard_fraction": pairrank_hard_fraction,
        }
        if method_id == SEARCH_METHOD_ID:
            if contracts.get("classification_loss") != "asymmetric_loss_multilabel_iccv2021":
                raise ValueError("search method must keep the ASL base loss")
            if contracts.get("pairrank_enabled") is not True:
                raise ValueError("search method must explicitly enable PairRank")
        elif contracts.get("classification_loss") != PAIRRANK_LOSS_ID:
            raise ValueError(f"{method_id} has an invalid pairrank classification loss")
        for name, expected in expected_pairrank.items():
            actual = contracts.get(name, expected)
            valid = (
                _same_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not valid:
                raise ValueError(
                    f"{method_id} has invalid pairrank contract {name}: "
                    f"expected {expected!r}, got {actual!r}"
                )
    balanced_bce_scope = _balanced_bce_scope(method)
    if balanced_bce_scope is not None:
        expected_balanced = {
            "balanced_bce_loss": BALANCED_BCE_LOSS_ID,
            "balanced_bce_scope": balanced_bce_scope,
            "balanced_bce_reduction": BALANCED_BCE_REDUCTION,
            "balanced_bce_clean_base_preserved": True,
        }
        for name, expected in expected_balanced.items():
            if contracts.get(name) != expected:
                raise ValueError(
                    f"{method_id} has invalid balanced-BCE contract {name}"
                )
    if (
        method_id not in GROUP_DRO_METHOD_IDS
        and method_id not in PAIRRANK_METHOD_IDS
        and contracts.get("fixed20_base_gradient_preserved") is not True
    ):
        raise ValueError(f"{method_id} must preserve the fixed20 base gradient")
    if method_id in SUPCON_METHOD_IDS:
        regularizer_id = contracts.get("feature_regularizer")
        if regularizer_id not in SUPPORTED_SUPCON_LOSS_IDS:
            raise ValueError(
                f"{method_id} has unsupported feature regularizer "
                f"{regularizer_id!r}"
            )
        if method_id != SEARCH_METHOD_ID and regularizer_id != SUPCON_LOSS_ID:
            raise ValueError(
                f"{method_id} may only use the locked exact-label SupCon loss"
            )
        positive_policy = (
            "per_active_label_clean_keys_including_same_record"
            if regularizer_id == MULSUPCON_LOSS_ID
            else "exact_positive_set_including_same_record"
        )
        expected_supcon = {
            "feature_regularizer": regularizer_id,
            "feature_regularizer_weight": hyperparameters["supcon_weight"],
            "feature_regularizer_temperature": hyperparameters["supcon_temperature"],
            "feature_regularizer_key_gradient": "stop_gradient",
            "feature_regularizer_positive_policy": positive_policy,
            "feature_regularizer_query_views": list(SUPCON_QUERY_VIEWS),
        }
        for name, expected in expected_supcon.items():
            actual = contracts.get(name)
            valid = (
                _same_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not valid:
                raise ValueError(
                    f"{method_id} has invalid SupCon contract {name}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        if regularizer_id == MULSUPCON_LOSS_ID:
            capacity = contracts.get("feature_regularizer_memory_capacity")
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or not 64 <= capacity <= 4096
            ):
                raise ValueError(
                    "MulSupCon feature memory capacity must be an integer in "
                    "[64,4096]"
                )
            if contracts.get("feature_regularizer_anchor_weighting") not in {
                "equal_per_active_label",
                "equal_per_record",
            }:
                raise ValueError("MulSupCon anchor weighting is invalid")
            if (
                contracts.get("feature_regularizer_all_zero_policy")
                != "same_record_instance_infonce"
            ):
                raise ValueError("MulSupCon all-zero fallback is invalid")
    if method_id == SEARCH_METHOD_ID:
        expected_pathmix = {
            "clean_manifold_mixup_weight": hyperparameters["clean_mixup_weight"],
            "clean_manifold_mixup_anchor_mass": hyperparameters[
                "clean_mixup_anchor_mass"
            ],
            "clean_manifold_mixup_partner_policy": MANIFOLD_MIXUP_POLICY,
            "feature_pathmix_weight": hyperparameters["pathmix_weight"],
            "feature_pathmix_anchor_mass": hyperparameters["pathmix_anchor_mass"],
            "feature_pathmix_query_views": list(FEATURE_PATHMIX_QUERY_VIEWS),
            "crossbatch_pairrank_weight": hyperparameters[
                "crossbatch_pairrank_weight"
            ],
            "crossbatch_pairrank_temperature": hyperparameters[
                "crossbatch_pairrank_temperature"
            ],
            "crossbatch_pairrank_capacity": int(
                hyperparameters["crossbatch_pairrank_capacity"]
            ),
            "crossbatch_pairrank_hard_fraction": hyperparameters[
                "crossbatch_pairrank_hard_fraction"
            ],
            "crossbatch_pairrank_scope": CROSSBATCH_PAIRRANK_SCOPE,
            "crossbatch_memory_update": CROSSBATCH_MEMORY_UPDATE,
            "robust_view_pairrank_weight": hyperparameters[
                "robust_view_pairrank_weight"
            ],
            "robust_view_pairrank_temperature": hyperparameters[
                "robust_view_pairrank_temperature"
            ],
            "robust_view_pairrank_capacity": int(
                hyperparameters["robust_view_pairrank_capacity"]
            ),
            "robust_view_pairrank_hard_fraction": hyperparameters[
                "robust_view_pairrank_hard_fraction"
            ],
            "robust_view_pairrank_loss": ROBUST_VIEW_PAIRRANK_LOSS_ID,
            "robust_view_pairrank_scope": ROBUST_VIEW_PAIRRANK_SCOPE,
            "robust_view_pairrank_memory_update": (
                ROBUST_VIEW_PAIRRANK_MEMORY_UPDATE
            ),
            "candidate_replay_count": int(
                hyperparameters["candidate_replay_count"]
            ),
            "candidate_replay_policy": CANDIDATE_REPLAY_POLICY,
            "candidate_replay_order": list(CANDIDATE_REPLAY_ORDER),
        }
        for name, expected in expected_pathmix.items():
            actual = contracts.get(name, expected)
            valid = (
                _same_float(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not valid:
                raise ValueError(
                    f"{method_id} has invalid path-mix contract {name}: "
                    f"expected {expected!r}, got {actual!r}"
                )
    if method_id in COMPATIBLE_AUXILIARY_METHOD_IDS and auxiliary_terms:
        candidate_profile = contracts.get("pcgrad_candidate_profile")
        if method_id == SEARCH_METHOD_ID:
            if (
                not isinstance(candidate_profile, str)
                or not candidate_profile
                or Path(candidate_profile).is_absolute()
                or ".." in Path(candidate_profile).parts
            ):
                raise ValueError(
                    f"{method_id} has an invalid sandbox candidate profile"
                )
        elif candidate_profile != CANDIDATE_PROFILE_BY_METHOD[method_id]:
            raise ValueError(
                f"{method_id} has an invalid compatible candidate profile"
            )

    actual_terms = tuple(
        (term.name, term.kind, tuple(term.views), float(term.weight))
        for term in method.objective.terms
    )
    if not auxiliary_terms:
        expected_terms = (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
        )
    elif method_id in RESIDUALIZED_AUXILIARY_METHOD_IDS:
        auxiliary_view = contracts.get("pcgrad_auxiliary_view")
        if not isinstance(auxiliary_view, str) or not auxiliary_view:
            raise ValueError(f"{method_id} has an invalid auxiliary view")
        expected_terms = (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
            ("augmix_bce", "bce", (auxiliary_view,), 1.0),
        )
    elif method_id in VAE_POST_METHOD_IDS | HARD_ONLY_PRUNED_METHOD_IDS:
        expected_terms = (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
            ("lhat_direct_bce", "bce", ("lhat_view",), 0.5),
            (
                "clean_lhat_jsd",
                "bernoulli_jsd",
                ("clean_view", "lhat_view"),
                0.5,
            ),
        )
    elif method_id in D13_AUXILIARY_METHOD_IDS:
        expected_terms = (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
            ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
            ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
            ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
            ("compat_random_bce", "bce", ("compat_random_view",), 0.1),
            ("compat_hard_bce", "bce", ("compat_hard_view",), 0.1),
            (
                "clean_raw4_compat_random_hard_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "corruption_view_2",
                    "corruption_view_3",
                    "corruption_view_4",
                    "compat_random_view",
                    "compat_hard_view",
                ),
                0.6,
            ),
        )
    elif method_id in D19_AUXILIARY_METHOD_IDS and auxiliary_terms:
        expected_terms = (
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
            ("corruption_1_bce", "bce", ("corruption_view_1",), 0.4),
            ("compat_random_bce", "bce", ("compat_random_view",), 0.3),
            ("compat_hard_bce", "bce", ("compat_hard_view",), 0.3),
            (
                "clean_raw_random_hard_jsd",
                "bernoulli_jsd",
                (
                    "clean_view",
                    "corruption_view_1",
                    "compat_random_view",
                    "compat_hard_view",
                ),
                0.6,
            ),
        )
    else:
        expected_terms_list = [
            ("clean_bce", "bce", ("clean_view",), 1.0),
            ("corrupted_bce", "bce", ("corrupted_view",), 1.0),
        ]
        if auxiliary_terms == DIRECT_LHAT_AUXILIARY_TERMS:
            expected_terms_list.append(
                ("lhat_direct_bce", "bce", ("lhat_view",), 0.5)
            )
        expected_terms_list.extend(
            [
            ("augmix_bce", "bce", ("augmix_view",), 1.0),
                (
                    "clean_lhat_augmix_jsd",
                    "bernoulli_jsd",
                    ("clean_view", "lhat_view", "augmix_view"),
                    0.5,
                ),
            ]
        )
        expected_terms = tuple(expected_terms_list)
    if method_id == SEARCH_METHOD_ID:
        expected_structure = tuple(
            (name, kind, views) for name, kind, views, _weight in expected_terms
        )
        actual_structure = tuple(
            (name, kind, views) for name, kind, views, _weight in actual_terms
        )
        if actual_structure != expected_structure:
            raise ValueError(f"{method_id} objective structure drifted")
        for name, _kind, _views, weight in actual_terms:
            lower, upper = (1.0, 1.0) if name in BASE_TERMS else (0.0, 2.0)
            if not lower <= float(weight) <= upper:
                raise ValueError(
                    f"{method_id} objective weight {name}={weight} outside [{lower},{upper}]"
                )
    elif actual_terms != expected_terms:
        raise ValueError(f"{method_id} objective drifted from the frozen profile")
    _STATE.active_hyperparameters = dict(hyperparameters)
    _STATE.active_hyperparameters.update(
        {
            f"auxiliary_term_scale_{name}": float(scale)
            for name, scale in auxiliary_term_scales.items()
        }
    )
    (
        _STATE.pairrank_class_weights,
        _STATE.pairrank_hard_fraction,
    ) = _runtime_pairrank_controls(method)
    return alpha


@dataclass(frozen=True)
class _ResidualizedSeedInvocation:
    base_seed: int
    rng_identity: tuple[str, ...]


class _ProfileIndependentNodeContext:
    """Delegate one node while deriving matched seeds without method identity."""

    def __init__(
        self,
        inner: NodeContext,
        *,
        invocation: _ResidualizedSeedInvocation,
        seed_config_path: Path,
    ) -> None:
        self.node_id = inner.node_id
        self.node_type = inner.node_type
        self.params = inner.params
        self.rng_namespace = inner.rng_namespace
        self._inner = inner
        self._invocation = invocation
        self._seed_config_path = seed_config_path
        self._generators: dict[tuple[str, str], torch.Generator] = {}

    def source(self, name: str) -> ViewValue:
        return self._inner.source(name)

    def call_adapter(
        self,
        name: str,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        return self._inner.call_adapter(name, inputs)

    def resource(self, name: str) -> Any:
        return self._inner.resource(name)

    def record_diagnostic(self, name: str, value: Any) -> None:
        self._inner.record_diagnostic(name, value)

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
            seed = derive_seed(
                self.rng_namespace,
                self.node_id,
                stream,
                *self._invocation.rng_identity,
                base_seed=self._invocation.base_seed,
                config_path=self._seed_config_path,
            )
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(seed)
            self._generators[key] = generator
            self._inner.record_diagnostic(
                f"rng/{stream}/{resolved_device}",
                {
                    "seed": seed,
                    "namespace": self.rng_namespace,
                    "execution_identity": list(self._invocation.rng_identity),
                    "seed_config_path": str(self._seed_config_path),
                    "policy": RESIDUALIZED_CANDIDATE_SEED_POLICY,
                },
            )
        return generator


@dataclass(frozen=True)
class _ResidualizedFullBatchAugMix:
    batch: AugMixBatch
    waveform: torch.Tensor
    valid_mask: torch.Tensor


def _mask_positions(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _generate_residualized_full_batch_augmix(
    clean: WaveformView,
    hard: WaveformView,
    *,
    config: AugMixConfig,
    generator: torch.Generator,
) -> _ResidualizedFullBatchAugMix:
    """Draw paired corruption chains/mixing over B before applying validity."""

    if clean.sample_ids != hard.sample_ids or not torch.equal(
        clean.labels, hard.labels
    ):
        raise RuntimeError("residualized AugMix inputs lost origin/label alignment")
    if clean.waveform.shape != hard.waveform.shape:
        raise ValueError("residualized AugMix inputs must have identical shape")
    if clean.valid_mask.shape != hard.valid_mask.shape:
        raise ValueError("residualized AugMix validity masks must have identical shape")
    valid = clean.valid_mask & hard.valid_mask
    expanded_valid = valid.view(-1, 1, 1)
    safe_hard = torch.where(expanded_valid, hard.waveform, clean.waveform)
    batch = generate_three_chain_augmix(
        clean.waveform,
        safe_hard,
        sampling_rate_hz=100,
        config=config,
        generator=generator,
        include_normalized=False,
    )
    waveform = torch.where(
        expanded_valid,
        batch.mixed_raw,
        clean.waveform.to(dtype=torch.float32),
    ).contiguous()
    return _ResidualizedFullBatchAugMix(
        batch=batch,
        waveform=waveform,
        valid_mask=valid.clone(),
    )


class _ResidualizedCandidateRuntime(MethodViewRuntime):
    """Candidate runtime with cross-arm common random numbers over full B."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed_paths = {
            Path(path).expanduser().resolve()
            for path in self.rng_seed_config_paths.values()
        }
        if len(seed_paths) != 1:
            raise ValueError(
                "residualized candidate RNG resources must share one seed config"
            )
        self._residualized_seed_config_path = next(iter(seed_paths))
        self._residualized_seed_invocation: _ResidualizedSeedInvocation | None = None
        self._performance_timer: Any | None = None

    def set_performance_timer(self, timer: Any | None) -> None:
        if timer is not None and (
            not callable(getattr(timer, "start", None))
            or not callable(getattr(timer, "stop", None))
        ):
            raise TypeError("performance timer must provide start/stop methods")
        self._performance_timer = timer

    @contextmanager
    def _timed_phase(self, phase: str) -> Iterator[None]:
        timer = self._performance_timer
        if timer is None:
            yield
            return
        timer.start(phase)
        try:
            yield
        finally:
            timer.stop(phase)

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["residualized_pairing"] = {
            "candidate_seed_policy": RESIDUALIZED_CANDIDATE_SEED_POLICY,
            "augmix_batch_policy": RESIDUALIZED_AUGMIX_BATCH_POLICY,
            "availability_policy": RESIDUALIZED_AVAILABILITY_POLICY,
        }
        return payload

    def _paired_context(self, context: NodeContext) -> NodeContext:
        invocation = self._residualized_seed_invocation
        if invocation is None:
            raise RuntimeError("residualized candidate RNG invocation is inactive")
        return _ProfileIndependentNodeContext(
            context,
            invocation=invocation,
            seed_config_path=self._residualized_seed_config_path,
        )

    def _canonical_corruption(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
        *,
        composition_indices: torch.Tensor | None = None,
        composition_index_hint: int | None = None,
    ) -> ViewValue:
        if context.node_id == "raw_chain3":
            with self._timed_phase("raw_chain3_generation"):
                return super()._canonical_corruption(
                    self._paired_context(context),
                    inputs,
                    composition_indices=composition_indices,
                    composition_index_hint=composition_index_hint,
                )
        return super()._canonical_corruption(
            self._paired_context(context),
            inputs,
            composition_indices=composition_indices,
            composition_index_hint=composition_index_hint,
        )

    def _lhat_attack(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        with self._timed_phase("lhat_search"):
            return super()._lhat_attack(self._paired_context(context), inputs)

    def _augmix(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        with self._timed_phase("augmix"):
            return self._augmix_impl(context, inputs)

    def _augmix_impl(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if (
            len(inputs) != 2
            or not isinstance(inputs[0], WaveformView)
            or not isinstance(inputs[1], WaveformView)
        ):
            raise TypeError("three-chain AugMix requires clean and hard views")
        if self.augmix_config is None:
            raise RuntimeError("AugMix runtime has no resolved config")
        clean, hard = inputs
        paired_context = self._paired_context(context)
        generator = paired_context.torch_generator(
            "chains_and_mixing", device=clean.waveform.device
        )
        generated = _generate_residualized_full_batch_augmix(
            clean,
            hard,
            config=self.augmix_config,
            generator=generator,
        )
        accepted_positions = _mask_positions(generated.valid_mask)
        accepted = torch.nonzero(
            generated.valid_mask, as_tuple=False
        ).flatten()
        if int(accepted.numel()):
            batch = generated.batch
            diagnostic_values = (
                batch.chain1_depth.index_select(0, accepted).float().mean(),
                batch.chain2_depth.index_select(0, accepted).float().mean(),
                batch.mixture_weights.index_select(0, accepted)[:, 2].mean(),
                batch.augmented_strength.index_select(0, accepted).mean(),
            )
        else:
            zero = clean.waveform.new_zeros((), dtype=torch.float32)
            diagnostic_values = (zero, zero, zero, zero)
        diagnostics = dict(
            zip(
                ("chain1_depth", "chain2_depth", "vae_weight", "strength"),
                (
                    float(value.detach().cpu())
                    for value in diagnostic_values
                ),
                strict=True,
            )
        )
        for name, value in diagnostics.items():
            paired_context.record_diagnostic(name, value)
        paired_context.record_diagnostic("accepted_count", len(accepted_positions))
        paired_context.record_diagnostic("full_batch_count", clean.batch_size)
        batch = generated.batch
        trace = {
            "chain1_composition_index": batch.chain1_composition_index.detach().clone(),
            "chain2_composition_index": batch.chain2_composition_index.detach().clone(),
            "chain1_depth": batch.chain1_depth.detach().clone(),
            "chain2_depth": batch.chain2_depth.detach().clone(),
            "chain1_operator_mask": batch.chain1_operator_mask.detach().clone(),
            "chain2_operator_mask": batch.chain2_operator_mask.detach().clone(),
            "mixture_weights": batch.mixture_weights.detach().clone(),
            "augmented_strength": batch.augmented_strength.detach().clone(),
        }
        return WaveformView(
            name=context.node_id,
            waveform=generated.waveform,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=generated.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(clean.name, hard.name),
                rng_namespace=context.rng_namespace,
                parameters={
                    "chain3_source": hard.name,
                    "chain3_additional_corruption": False,
                    "operator_domain_sampling_rate_hz": (
                        batch.operator_domain_sampling_rate_hz
                    ),
                    "candidate_seed_policy": RESIDUALIZED_CANDIDATE_SEED_POLICY,
                    "augmix_batch_policy": RESIDUALIZED_AUGMIX_BATCH_POLICY,
                    "availability_policy": RESIDUALIZED_AVAILABILITY_POLICY,
                },
            ),
            metadata={
                "accepted_positions": accepted_positions,
                "diagnostic_means": diagnostics,
                "paired_augmix_trace": trace,
                "full_batch_count": clean.batch_size,
                "candidate_seed_policy": RESIDUALIZED_CANDIDATE_SEED_POLICY,
                "augmix_batch_policy": RESIDUALIZED_AUGMIX_BATCH_POLICY,
                "availability_policy": RESIDUALIZED_AVAILABILITY_POLICY,
            },
        )

    def generate(self, **kwargs: Any):
        if self._residualized_seed_invocation is not None:
            raise RuntimeError("residualized candidate runtime is not reentrant")
        base_seed = kwargs.get("base_seed")
        rng_identity = kwargs.get("rng_identity")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise TypeError("residualized candidate base_seed must be an integer")
        if isinstance(rng_identity, (str, bytes)) or not isinstance(
            rng_identity, Sequence
        ):
            raise TypeError("residualized candidate rng_identity must be a sequence")
        invocation = _ResidualizedSeedInvocation(
            base_seed=int(base_seed),
            rng_identity=tuple(str(value) for value in rng_identity),
        )
        if any(not value for value in invocation.rng_identity):
            raise ValueError("residualized candidate RNG identity may not be empty")
        self._residualized_seed_invocation = invocation
        try:
            return super().generate(**kwargs)
        finally:
            self._residualized_seed_invocation = None


class _PCGradMethodRuntime:
    """Route clean exposure through LHAT/AugMix and raw exposures through Direct."""

    def __init__(self, candidate: Any, direct: Any, method: CompiledMethod) -> None:
        self.candidate = candidate
        self.direct = direct
        self.method = method
        self.augmix_config = candidate.augmix_config
        paths = {
            f"candidate.{name}": path
            for name, path in candidate.rng_seed_config_paths.items()
        }
        paths.update(
            {
                f"fixed20_base.{name}": path
                for name, path in direct.rng_seed_config_paths.items()
            }
        )
        self.rng_seed_config_paths = paths

    def describe(self) -> dict[str, Any]:
        hyperparameters = _runtime_hyperparameters(self.method)
        balanced_bce_scope = _balanced_bce_scope(self.method)
        return {
            "type": "sandbox_fixed20_pcgrad_lhat_augmix_runtime",
            "method_id": self.method.profile_name,
            "pcgrad": {
                "mode": self.method.contracts["pcgrad_auxiliary_mode"],
                "auxiliary_alpha": hyperparameters["alpha"],
                "residual_norm_ratio_cap": hyperparameters[
                    "residual_norm_ratio_cap"
                ],
                "projection_epsilon": EPSILON,
                "base_method_id": BASE_METHOD_ID,
                "auxiliary_terms": list(_active_auxiliary_terms(self.method)),
                "pairrank": (
                    {
                        "loss_id": PAIRRANK_LOSS_ID,
                        "weight": hyperparameters["pairrank_weight"],
                        "temperature": hyperparameters["pairrank_temperature"],
                        "reduction": PAIRRANK_REDUCTION,
                    }
                    if self.method.profile_name in PAIRRANK_METHOD_IDS
                    else None
                ),
                "balanced_bce": (
                    {
                        "loss_id": BALANCED_BCE_LOSS_ID,
                        "scope": balanced_bce_scope,
                        "reduction": BALANCED_BCE_REDUCTION,
                    }
                    if balanced_bce_scope is not None
                    else None
                ),
                "candidate_replay": {
                    "count": _candidate_replay_count(self.method),
                    "policy": CANDIDATE_REPLAY_POLICY,
                    "composition_indices": list(
                        CANDIDATE_REPLAY_ORDER[
                            : _candidate_replay_count(self.method)
                        ]
                    ),
                },
            },
            "candidate_runtime": self.candidate.describe(),
            "fixed20_base_runtime": self.direct.describe(),
        }

    def set_performance_timer(self, timer: Any | None) -> None:
        if timer is not None and (
            not callable(getattr(timer, "start", None))
            or not callable(getattr(timer, "stop", None))
        ):
            raise TypeError("performance timer must provide start/stop methods")
        _STATE.performance_timer = timer
        candidate_setter = getattr(self.candidate, "set_performance_timer", None)
        if candidate_setter is not None:
            if not callable(candidate_setter):
                raise TypeError(
                    "candidate runtime set_performance_timer must be callable"
                )
            candidate_setter(timer)

    def generate(
        self,
        *,
        composition_index_hint: int | None = None,
        **kwargs: Any,
    ):
        composition_indices = kwargs.get("composition_indices")
        if composition_indices is None:
            return self.candidate.generate(**kwargs)
        if not isinstance(composition_indices, torch.Tensor):
            raise TypeError("composition_indices must be a tensor or None")
        if composition_index_hint is not None:
            if isinstance(composition_index_hint, bool) or not isinstance(
                composition_index_hint, int
            ):
                raise TypeError("composition_index_hint must be an integer")
            if composition_index_hint < -1 or composition_index_hint > 19:
                raise ValueError(
                    "composition_index_hint must be -1 or lie in [0,19]"
                )
            # The whitelist trainer creates composition_indices from this same
            # host integer. Trusting the explicit hint avoids one CUDA
            # synchronization for each of the clean+20 fixed20 exposures.
            composition = composition_index_hint
        else:
            unique = torch.unique(composition_indices.detach())
            if unique.numel() != 1:
                raise ValueError(
                    "one PCGrad exposure must use one composition identity"
                )
            composition = int(unique.item())
        _STATE.current_composition = composition
        if _group_dro_active(self.method):
            if composition == -1 and _STATE.group_cycle_losses:
                raise RuntimeError(
                    "GroupDRO started a new base batch before completing 20 groups"
                )
        if composition == -1 or _is_candidate_replay(
            self.method, composition
        ):
            candidate_kwargs = dict(kwargs)
            if self.method.profile_name in (
                COMPATIBLE_AUXILIARY_METHOD_IDS
                | RESIDUALIZED_AUXILIARY_METHOD_IDS
            ):
                # Clean and replacement exposures delegate to the standalone
                # D19 runtime.  It uses None so every outer exposure receives
                # its own replayable random VAE/AugMix draw.
                candidate_kwargs["composition_indices"] = None
                candidate_kwargs.pop("composition_index_hint", None)
                candidate_kwargs.pop("objective_term_names", None)
            return self.candidate.generate(**candidate_kwargs)
        if not 0 <= composition < 20:
            raise ValueError("fixed20 composition index must be in [0,19]")
        return self.direct.generate(**kwargs)


@dataclass
class _PCGradState:
    pending_parameters: tuple[nn.Parameter, ...] = ()
    pending_aux_grads: tuple[torch.Tensor | None, ...] = ()
    pending_alpha: float | None = None
    pending_projection_mode: str | None = None
    pending_norm_ratio_cap: float | None = None
    pending_aux_loss: float | None = None
    base_method: CompiledMethod | None = None
    candidate_method: CompiledMethod | None = None
    wrapper_method_id: str | None = None
    steps: list[dict[str, float | bool | int | str]] = field(default_factory=list)
    current_composition: int | None = None
    group_log_weights: list[float] = field(
        default_factory=lambda: [0.0] * GROUP_DRO_GROUP_COUNT
    )
    group_cycle_losses: dict[int, float] = field(default_factory=dict)
    group_dro_cycles: list[dict[str, Any]] = field(default_factory=list)
    pairrank_losses: list[float] = field(default_factory=list)
    pairrank_valid_class_counts: list[int] = field(default_factory=list)
    pairrank_class_weights: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    pairrank_hard_fraction: float = 1.0
    pairrank_class_loss_sums: list[float] = field(default_factory=lambda: [0.0] * 5)
    pairrank_class_loss_counts: list[int] = field(default_factory=lambda: [0] * 5)
    balanced_bce_losses: list[float] = field(default_factory=list)
    balanced_bce_scope: str | None = None
    supcon_losses: list[float] = field(default_factory=list)
    supcon_query_counts: list[int] = field(default_factory=list)
    supcon_weights: list[float] = field(default_factory=list)
    supcon_loss_ids: list[str] = field(default_factory=list)
    supcon_label_anchor_counts: list[int] = field(default_factory=list)
    supcon_positive_counts: list[float] = field(default_factory=list)
    supcon_memory_sizes: list[int] = field(default_factory=list)
    supcon_memory_features: torch.Tensor | None = None
    supcon_memory_labels: torch.Tensor | None = None
    clean_mixup_losses: list[float] = field(default_factory=list)
    clean_mixup_pair_counts: list[int] = field(default_factory=list)
    pathmix_losses: list[float] = field(default_factory=list)
    pathmix_valid_counts: list[int] = field(default_factory=list)
    crossbatch_memory_logits: torch.Tensor | None = None
    crossbatch_memory_labels: torch.Tensor | None = None
    crossbatch_losses: list[float] = field(default_factory=list)
    crossbatch_pair_counts: list[int] = field(default_factory=list)
    crossbatch_memory_sizes: list[int] = field(default_factory=list)
    robust_view_memory_scores: torch.Tensor | None = None
    robust_view_memory_labels: torch.Tensor | None = None
    robust_view_losses: list[float] = field(default_factory=list)
    robust_view_pair_counts: list[int] = field(default_factory=list)
    robust_view_valid_class_counts: list[int] = field(default_factory=list)
    robust_view_memory_sizes: list[int] = field(default_factory=list)
    active_hyperparameters: dict[str, float] = field(default_factory=dict)
    performance_timer: Any | None = None

    def clear_pending(self) -> None:
        self.pending_parameters = ()
        self.pending_aux_grads = ()
        self.pending_alpha = None
        self.pending_projection_mode = None
        self.pending_norm_ratio_cap = None
        self.pending_aux_loss = None

    def clear_runtime(self) -> None:
        self.clear_pending()
        self.base_method = None
        self.candidate_method = None
        self.wrapper_method_id = None
        self.current_composition = None
        self.group_log_weights = [0.0] * GROUP_DRO_GROUP_COUNT
        self.group_cycle_losses.clear()
        self.group_dro_cycles.clear()
        self.pairrank_losses.clear()
        self.pairrank_valid_class_counts.clear()
        self.pairrank_class_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.pairrank_hard_fraction = 1.0
        self.pairrank_class_loss_sums = [0.0] * 5
        self.pairrank_class_loss_counts = [0] * 5
        self.balanced_bce_losses.clear()
        self.balanced_bce_scope = None
        self.supcon_losses.clear()
        self.supcon_query_counts.clear()
        self.supcon_weights.clear()
        self.supcon_loss_ids.clear()
        self.supcon_label_anchor_counts.clear()
        self.supcon_positive_counts.clear()
        self.supcon_memory_sizes.clear()
        self.supcon_memory_features = None
        self.supcon_memory_labels = None
        self.clean_mixup_losses.clear()
        self.clean_mixup_pair_counts.clear()
        self.pathmix_losses.clear()
        self.pathmix_valid_counts.clear()
        self.crossbatch_memory_logits = None
        self.crossbatch_memory_labels = None
        self.crossbatch_losses.clear()
        self.crossbatch_pair_counts.clear()
        self.crossbatch_memory_sizes.clear()
        self.robust_view_memory_scores = None
        self.robust_view_memory_labels = None
        self.robust_view_losses.clear()
        self.robust_view_pair_counts.clear()
        self.robust_view_valid_class_counts.clear()
        self.robust_view_memory_sizes.clear()
        self.active_hyperparameters.clear()
        self.performance_timer = None


_STATE = _PCGradState()


def _active_hyperparameter(name: str, fallback: float) -> float:
    return float(_STATE.active_hyperparameters.get(name, fallback))


def _candidate_replay_count(method: CompiledMethod) -> int:
    value = _runtime_hyperparameters(method)["candidate_replay_count"]
    count = int(value)
    if not math.isclose(value, count, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("candidate_replay_count must be an integer")
    return count


def _is_candidate_replay(method: CompiledMethod, composition: int) -> bool:
    if method.profile_name != SEARCH_METHOD_ID or composition < 0:
        return False
    count = _candidate_replay_count(method)
    return composition in CANDIDATE_REPLAY_ORDER[:count]


def _capture_bn_state(model: nn.Module):
    captured = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        captured.append(
            (
                module,
                module.momentum,
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
                None
                if module.num_batches_tracked is None
                else module.num_batches_tracked.detach().clone(),
            )
        )
        module.momentum = 0.0
    return captured


def _restore_bn_state(captured: Sequence[tuple[Any, ...]]) -> None:
    with torch.no_grad():
        for module, momentum, running_mean, running_var, batches in captured:
            module.momentum = momentum
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if batches is not None:
                module.num_batches_tracked.copy_(batches)


def _capture_rng_state(model: nn.Module):
    cpu = torch.random.get_rng_state()
    device = next(model.parameters()).device
    cuda = None
    if device.type == "cuda":
        cuda = torch.cuda.get_rng_state(device)
    return cpu, device, cuda


def _restore_rng_state(state: tuple[torch.Tensor, torch.device, torch.Tensor | None]):
    cpu, device, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def _merged_base_objective(base: Any, auxiliary: Any) -> Any:
    """Expose raw auxiliary losses while keeping trainer total exactly base-only."""

    result_type = type(base)
    raw_terms = dict(base.raw_terms)
    raw_terms.update(auxiliary.raw_terms)
    weighted_terms = dict(base.weighted_terms)
    weighted_terms.update(
        {name: value.detach() * 0.0 for name, value in auxiliary.weighted_terms.items()}
    )
    valid_counts = dict(base.valid_counts)
    valid_counts.update(auxiliary.valid_counts)
    return result_type(
        total=base.total,
        raw_terms=raw_terms,
        weighted_terms=weighted_terms,
        valid_counts=valid_counts,
    )


def _group_dro_probabilities() -> tuple[float, ...]:
    """Return a bounded adaptive distribution over the twenty groups."""

    maximum = max(_STATE.group_log_weights)
    exponentials = [math.exp(value - maximum) for value in _STATE.group_log_weights]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("GroupDRO normalization became non-finite")
    adaptive = [value / total for value in exponentials]
    uniform = 1.0 / GROUP_DRO_GROUP_COUNT
    probabilities = tuple(
        _active_hyperparameter("group_dro_uniform_mix", GROUP_DRO_UNIFORM_MIX) * uniform
        + (
            1.0
            - _active_hyperparameter(
                "group_dro_uniform_mix", GROUP_DRO_UNIFORM_MIX
            )
        )
        * value
        for value in adaptive
    )
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("GroupDRO probabilities do not sum to one")
    return probabilities


def _group_dro_scaled_objective(
    base: Any,
    composition: int,
    *,
    group_loss: float | None = None,
) -> Any:
    """Weight one group using the distribution frozen for the current cycle.

    The outer trainer still multiplies every corrupted exposure by ``0.025``.
    Multiplication by ``20 * q_g`` therefore redistributes, but never changes,
    the total corrupted family mass of ``0.5`` within a complete cycle.
    Losses from the current complete cycle update the distribution only for the
    next base batch, avoiding canonical-order leakage.
    """

    if not 0 <= composition < GROUP_DRO_GROUP_COUNT:
        raise ValueError("GroupDRO composition must be in [0,19]")
    if composition in _STATE.group_cycle_losses:
        raise RuntimeError(f"GroupDRO composition {composition} repeated in a cycle")
    probabilities = _group_dro_probabilities()
    scale = GROUP_DRO_GROUP_COUNT * probabilities[composition]
    raw_loss = (
        _finite_scalar(base.total, "GroupDRO corrupted objective")
        if group_loss is None
        else float(group_loss)
    )
    if not math.isfinite(raw_loss):
        raise FloatingPointError("GroupDRO update loss is non-finite")
    _STATE.group_cycle_losses[composition] = raw_loss

    result_type = type(base)
    weighted_terms = {
        name: value * scale for name, value in base.weighted_terms.items()
    }
    result = result_type(
        total=base.total * scale,
        raw_terms=dict(base.raw_terms),
        weighted_terms=weighted_terms,
        valid_counts=dict(base.valid_counts),
    )

    if composition == GROUP_DRO_GROUP_COUNT - 1:
        if set(_STATE.group_cycle_losses) != set(range(GROUP_DRO_GROUP_COUNT)):
            raise RuntimeError("GroupDRO cycle ended before all twenty groups arrived")
        losses = tuple(
            _STATE.group_cycle_losses[index]
            for index in range(GROUP_DRO_GROUP_COUNT)
        )
        scales = tuple(GROUP_DRO_GROUP_COUNT * value for value in probabilities)
        for index, loss in enumerate(losses):
            _STATE.group_log_weights[index] += _active_hyperparameter(
                "group_dro_step_size", GROUP_DRO_STEP_SIZE
            ) * loss
        recenter = max(_STATE.group_log_weights)
        _STATE.group_log_weights = [
            value - recenter for value in _STATE.group_log_weights
        ]
        updated_probabilities = _group_dro_probabilities()
        _STATE.group_dro_cycles.append(
            {
                "cycle": len(_STATE.group_dro_cycles) + 1,
                "losses": list(losses),
                "scales_used": list(scales),
                "probabilities_next": list(updated_probabilities),
                "scale_sum": float(sum(scales)),
            }
        )
        _STATE.group_cycle_losses.clear()
    return result


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    scalar = float(value.detach().float().cpu())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"PCGrad {name} is non-finite")
    return scalar


def _multilabel_pairwise_logistic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Average positive-vs-negative ranking loss over classes present in a batch."""

    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != 5:
        raise ValueError("pairrank expects aligned [batch,5] logits and targets")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("pairrank logits and targets must be floating point")
    if not bool(torch.isfinite(targets).all().item()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any().item()
    ):
        raise ValueError("pairrank targets must be finite probabilities in [0,1]")
    temperature = _active_hyperparameter(
        "pairrank_temperature", PAIRRANK_TEMPERATURE
    )
    scores = logits.float() / temperature
    terms: list[torch.Tensor] = []
    term_weights: list[float] = []
    for class_index in range(5):
        relevance = targets[:, class_index].float()
        relevance_difference = relevance[:, None] - relevance[None, :]
        ordered = relevance_difference > 1.0e-7
        if not bool(ordered.any().item()):
            continue
        score_difference = (
            scores[:, class_index, None] - scores[None, :, class_index]
        )
        weights = relevance_difference[ordered]
        weighted_losses = F.softplus(-score_difference[ordered]) * weights
        hard_fraction = float(_STATE.pairrank_hard_fraction)
        if hard_fraction < 1.0:
            hard_count = max(1, math.ceil(weighted_losses.numel() * hard_fraction))
            selected = torch.topk(weighted_losses, hard_count, largest=True).indices
            weighted_losses = weighted_losses[selected]
            weights = weights[selected]
        class_loss = (
            weighted_losses.sum()
            / weights.sum().clamp_min(EPSILON)
            * temperature
        )
        class_weight = float(_STATE.pairrank_class_weights[class_index])
        if class_weight <= 0.0:
            continue
        terms.append(class_loss)
        term_weights.append(class_weight)
        _STATE.pairrank_class_loss_sums[class_index] += _finite_scalar(
            class_loss, f"pairrank class {class_index} loss"
        )
        _STATE.pairrank_class_loss_counts[class_index] += 1
    if not terms:
        return logits.sum() * 0.0, 0
    weight_tensor = logits.new_tensor(term_weights, dtype=torch.float32)
    loss = (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum().clamp_min(
        EPSILON
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("pairrank produced a non-finite loss")
    return loss, len(terms)


def _crossbatch_pairwise_logistic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    update_memory: bool,
) -> tuple[torch.Tensor, int]:
    """Rank current samples against a detached FIFO of previous clean batches."""

    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != 5:
        raise ValueError("cross-batch PairRank expects aligned [batch,5] tensors")
    if not bool(torch.isfinite(logits).all().item()) or not bool(
        torch.isfinite(targets).all().item()
    ):
        raise FloatingPointError("cross-batch PairRank inputs must be finite")
    bank_logits = _STATE.crossbatch_memory_logits
    bank_labels = _STATE.crossbatch_memory_labels
    terms: list[torch.Tensor] = []
    term_weights: list[float] = []
    pair_count = 0
    temperature = _active_hyperparameter(
        "crossbatch_pairrank_temperature", 1.0
    )
    hard_fraction = _active_hyperparameter(
        "crossbatch_pairrank_hard_fraction", 0.25
    )
    if bank_logits is not None and bank_labels is not None:
        if bank_logits.device != logits.device or bank_labels.device != logits.device:
            raise RuntimeError("cross-batch PairRank memory moved devices")
        current_scores = logits.float() / temperature
        memory_scores = bank_logits.float() / temperature
        current_truth = targets.float() >= 0.5
        memory_truth = bank_labels.float() >= 0.5
        for class_index in range(5):
            losses: list[torch.Tensor] = []
            current_positive = current_scores[current_truth[:, class_index], class_index]
            current_negative = current_scores[~current_truth[:, class_index], class_index]
            memory_positive = memory_scores[memory_truth[:, class_index], class_index]
            memory_negative = memory_scores[~memory_truth[:, class_index], class_index]
            if current_positive.numel() and memory_negative.numel():
                losses.append(
                    F.softplus(
                        -(current_positive[:, None] - memory_negative[None, :])
                    ).flatten()
                )
            if current_negative.numel() and memory_positive.numel():
                losses.append(
                    F.softplus(
                        current_negative[:, None] - memory_positive[None, :]
                    ).flatten()
                )
            if not losses:
                continue
            class_losses = torch.cat(losses)
            pair_count += int(class_losses.numel())
            if hard_fraction < 1.0:
                hard_count = max(1, math.ceil(class_losses.numel() * hard_fraction))
                class_losses = torch.topk(
                    class_losses, hard_count, largest=True
                ).values
            class_weight = float(_STATE.pairrank_class_weights[class_index])
            if class_weight <= 0.0:
                continue
            terms.append(class_losses.mean() * temperature)
            term_weights.append(class_weight)

    loss = logits.sum() * 0.0
    if terms:
        weights = logits.new_tensor(term_weights, dtype=torch.float32)
        loss = (torch.stack(terms) * weights).sum() / weights.sum().clamp_min(
            EPSILON
        )
    if update_memory:
        detached_logits = logits.detach().float()
        detached_labels = targets.detach().float()
        if bank_logits is None:
            bank_logits = detached_logits
            bank_labels = detached_labels
        else:
            bank_logits = torch.cat((bank_logits, detached_logits), dim=0)
            assert bank_labels is not None
            bank_labels = torch.cat((bank_labels, detached_labels), dim=0)
        capacity = int(
            _active_hyperparameter("crossbatch_pairrank_capacity", 512.0)
        )
        _STATE.crossbatch_memory_logits = bank_logits[-capacity:].contiguous()
        _STATE.crossbatch_memory_labels = bank_labels[-capacity:].contiguous()
        _STATE.crossbatch_memory_sizes.append(
            int(_STATE.crossbatch_memory_logits.shape[0])
        )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("cross-batch PairRank produced a non-finite loss")
    return loss, pair_count


def _worst_view_pairwise_logistic_loss(
    *,
    model: nn.Module,
    captured: Sequence[torch.Tensor],
    bundle: Any,
) -> tuple[torch.Tensor, int, int, int]:
    """Rank the class-wise worst score across clean and three D19 views.

    A positive record contributes its minimum logit across the available
    clean/raw/random/hard views; a negative record contributes its maximum.
    Pairing those scores makes the surrogate target robust AUROC directly
    instead of averaging four independent view-wise ranking losses.
    """

    if len(captured) != len(SUPCON_CAPTURE_ORDER):
        raise RuntimeError(
            "worst-view PairRank expected one feature capture per D19 view"
        )
    feature_by_view = dict(zip(SUPCON_CAPTURE_ORDER, captured, strict=True))
    clean = bundle.require("clean_view")
    labels = clean.labels.float()
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("worst-view PairRank requires [batch,5] labels")
    if not bool(torch.isfinite(labels).all().item()) or bool(
        ((labels < 0.0) | (labels > 1.0)).any().item()
    ):
        raise ValueError("worst-view PairRank labels must be finite in [0,1]")
    batch_size = int(labels.shape[0])
    head = _classifier_head(model)
    clean_features = feature_by_view["clean_view"]
    if clean_features.ndim != 2 or int(clean_features.shape[0]) != batch_size:
        raise RuntimeError("worst-view PairRank clean feature count drifted")
    clean_logits = head(clean_features).float()
    if clean_logits.shape != labels.shape:
        raise RuntimeError("worst-view PairRank clean logits violate Super5 shape")
    score_views = [clean_logits]
    for name in SUPCON_QUERY_VIEWS:
        view = bundle.require(name)
        positions = torch.nonzero(view.valid_mask, as_tuple=False).flatten()
        features = feature_by_view[name]
        if features.ndim != 2:
            raise RuntimeError(f"worst-view PairRank {name} features must be 2D")
        # Invalid generated records are represented only by their clean score;
        # they must not acquire an artificial easy or hard endpoint.
        aligned = clean_logits.clone()
        if int(positions.numel()):
            if int(features.shape[0]) == batch_size:
                selected_features = features.index_select(0, positions)
            elif int(features.shape[0]) == int(positions.numel()):
                selected_features = features
            else:
                raise RuntimeError(
                    f"worst-view PairRank {name} count differs from validity"
                )
            selected_logits = head(selected_features).float()
            aligned = aligned.index_copy(0, positions, selected_logits)
        score_views.append(aligned)
    stacked = torch.stack(score_views, dim=1)
    truth = labels >= 0.5
    worst_positive = stacked.min(dim=1).values
    worst_negative = stacked.max(dim=1).values
    robust_scores = torch.where(truth, worst_positive, worst_negative)

    temperature = _active_hyperparameter(
        "robust_view_pairrank_temperature", 1.0
    )
    hard_fraction = _active_hyperparameter(
        "robust_view_pairrank_hard_fraction", 0.25
    )
    current_scores = robust_scores / temperature
    memory_scores = _STATE.robust_view_memory_scores
    memory_labels = _STATE.robust_view_memory_labels
    prior_memory_size = (
        int(memory_scores.shape[0]) if memory_scores is not None else 0
    )
    if (memory_scores is None) != (memory_labels is None):
        raise RuntimeError("worst-view PairRank score and label memories diverged")
    if memory_scores is not None and memory_labels is not None:
        if (
            memory_scores.ndim != 2
            or memory_labels.ndim != 2
            or memory_scores.shape != memory_labels.shape
            or memory_scores.shape[1] != 5
        ):
            raise RuntimeError("worst-view PairRank memory shape drifted")
        memory_scores = memory_scores.to(current_scores.device) / temperature
        memory_truth = memory_labels.to(labels.device) >= 0.5
    else:
        memory_truth = None

    terms: list[torch.Tensor] = []
    term_weights: list[float] = []
    pair_count = 0
    for class_index in range(5):
        current_positive = current_scores[truth[:, class_index], class_index]
        current_negative = current_scores[~truth[:, class_index], class_index]
        losses: list[torch.Tensor] = []
        if current_positive.numel() and current_negative.numel():
            losses.append(
                F.softplus(
                    -(current_positive[:, None] - current_negative[None, :])
                ).flatten()
            )
        if (
            memory_scores is not None
            and memory_truth is not None
            and current_positive.numel()
        ):
            memory_negative = memory_scores[
                ~memory_truth[:, class_index], class_index
            ]
            if memory_negative.numel():
                losses.append(
                    F.softplus(
                        -(current_positive[:, None] - memory_negative[None, :])
                    ).flatten()
                )
        if (
            memory_scores is not None
            and memory_truth is not None
            and current_negative.numel()
        ):
            memory_positive = memory_scores[
                memory_truth[:, class_index], class_index
            ]
            if memory_positive.numel():
                losses.append(
                    F.softplus(
                        current_negative[:, None] - memory_positive[None, :]
                    ).flatten()
                )
        if not losses:
            continue
        class_losses = torch.cat(losses)
        pair_count += int(class_losses.numel())
        if hard_fraction < 1.0:
            hard_count = max(1, math.ceil(class_losses.numel() * hard_fraction))
            class_losses = torch.topk(
                class_losses, hard_count, largest=True
            ).values
        class_weight = float(_STATE.pairrank_class_weights[class_index])
        if class_weight <= 0.0:
            continue
        terms.append(class_losses.mean() * temperature)
        term_weights.append(class_weight)

    loss = robust_scores.sum() * 0.0
    if terms:
        weights = robust_scores.new_tensor(term_weights, dtype=torch.float32)
        loss = (torch.stack(terms) * weights).sum() / weights.sum().clamp_min(
            EPSILON
        )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("worst-view PairRank produced a non-finite loss")

    with torch.no_grad():
        stored_scores = robust_scores.detach().float()
        stored_labels = labels.detach().float()
        if _STATE.robust_view_memory_scores is not None:
            stored_scores = torch.cat(
                (_STATE.robust_view_memory_scores, stored_scores), dim=0
            )
            assert _STATE.robust_view_memory_labels is not None
            stored_labels = torch.cat(
                (_STATE.robust_view_memory_labels, stored_labels), dim=0
            )
        capacity = int(
            _active_hyperparameter("robust_view_pairrank_capacity", 512.0)
        )
        _STATE.robust_view_memory_scores = stored_scores[-capacity:].contiguous()
        _STATE.robust_view_memory_labels = stored_labels[-capacity:].contiguous()
    return loss, len(terms), pair_count, prior_memory_size


def _soft_class_polarity_balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Give every Super5 class and its positive/negative mass equal weight."""

    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != 5:
        raise ValueError("balanced BCE expects aligned [batch,5] tensors")
    if not bool(torch.isfinite(targets).all().item()) or bool(
        ((targets < 0.0) | (targets > 1.0)).any().item()
    ):
        raise ValueError("balanced BCE targets must be finite probabilities in [0,1]")
    scores = logits.float()
    truth = targets.float()
    positive_terms = F.softplus(-scores)
    negative_terms = F.softplus(scores)
    class_losses: list[torch.Tensor] = []
    for class_index in range(5):
        positive_mass = truth[:, class_index]
        negative_mass = 1.0 - positive_mass
        partitions: list[torch.Tensor] = []
        if bool((positive_mass.sum() > EPSILON).item()):
            partitions.append(
                (positive_terms[:, class_index] * positive_mass).sum()
                / positive_mass.sum().clamp_min(EPSILON)
            )
        if bool((negative_mass.sum() > EPSILON).item()):
            partitions.append(
                (negative_terms[:, class_index] * negative_mass).sum()
                / negative_mass.sum().clamp_min(EPSILON)
            )
        if not partitions:
            raise RuntimeError("balanced BCE class has no probability mass")
        class_losses.append(torch.stack(partitions).mean())
    loss = torch.stack(class_losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("balanced BCE produced a non-finite loss")
    return loss


def _compute_with_pairrank(
    original_objective: Any,
    kwargs: Mapping[str, Any],
    *,
    group_reference_sink: list[float] | None = None,
    balanced_bce: bool = False,
    crossbatch_query: bool = False,
    crossbatch_update: bool = False,
):
    """Scope BCE+pairrank substitution to one objective materialization."""

    original_bce = F.binary_cross_entropy_with_logits

    def patched_bce(
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reduction != "mean" or size_average is not None or reduce is not None:
            raise ValueError("pairrank requires the managed mean BCE reduction")
        if balanced_bce:
            if weight is not None or pos_weight is not None:
                raise ValueError("balanced BCE does not accept BCE weights")
            base = _soft_class_polarity_balanced_bce_with_logits(input, target)
            _STATE.balanced_bce_losses.append(
                _finite_scalar(base, "balanced BCE loss")
            )
        else:
            base = original_bce(
                input,
                target,
                weight=weight,
                size_average=size_average,
                reduce=reduce,
                reduction=reduction,
                pos_weight=pos_weight,
            )
        if group_reference_sink is not None:
            import asl_runtime

            reference = asl_runtime.macro_balanced_bce_with_logits(input, target)
            group_reference_sink.append(
                _finite_scalar(reference, "GroupDRO balanced BCE reference")
            )
        pairrank_weight = _active_hyperparameter(
            "pairrank_weight", PAIRRANK_WEIGHT
        )
        if pairrank_weight > 0.0:
            ranking, valid_classes = _multilabel_pairwise_logistic_loss(input, target)
            _STATE.pairrank_losses.append(_finite_scalar(ranking, "pairrank loss"))
            _STATE.pairrank_valid_class_counts.append(valid_classes)
            result = base + pairrank_weight * ranking
        else:
            result = base
        if crossbatch_query:
            memory_ranking, pair_count = _crossbatch_pairwise_logistic_loss(
                input,
                target,
                update_memory=crossbatch_update,
            )
            weight = _active_hyperparameter("crossbatch_pairrank_weight", 0.0)
            result = result + weight * memory_ranking
            _STATE.crossbatch_losses.append(
                _finite_scalar(memory_ranking, "cross-batch PairRank loss")
            )
            _STATE.crossbatch_pair_counts.append(pair_count)
        return result

    F.binary_cross_entropy_with_logits = patched_bce
    try:
        return original_objective(**dict(kwargs))
    finally:
        F.binary_cross_entropy_with_logits = original_bce


def _classifier_head(model: nn.Module) -> nn.Module:
    """Return the managed penultimate-feature consumer for either backbone."""

    def supported(module: Any) -> bool:
        if isinstance(module, nn.Linear):
            return True
        return (
            isinstance(module, nn.Module)
            and isinstance(getattr(module, "in_features", None), int)
            and getattr(module, "out_features", None) == 5
            and isinstance(getattr(module, "base", None), nn.Linear)
        )

    dense = getattr(model, "dense", None)
    if supported(dense):
        return dense
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and len(classifier):
        head = classifier[-1]
        if supported(head):
            return head
    raise TypeError("SupCon requires the managed EfficientNet or ECGFounder linear head")


def _semantic_partner_indices(labels: torch.Tensor) -> torch.Tensor:
    """Choose one deterministic semantically compatible peer per record.

    Exact-positive-set peers win naturally through Jaccard similarity.  When an
    exact peer is absent, NORM-only, abnormal and label-empty samples stay in
    their own semantic partition.  A deterministic circular tie-break avoids
    module-level RNG and never returns the anchor itself.
    """

    if labels.ndim != 2 or labels.shape[1] != 5 or labels.shape[0] < 2:
        raise ValueError("manifold mixup requires labels shaped [batch>=2,5]")
    truth = labels.float()
    if not bool(torch.isfinite(truth).all().item()):
        raise ValueError("manifold mixup labels must be finite")
    batch_size = int(truth.shape[0])
    abnormal = truth[:, (0, 1, 2, 4)].sum(dim=1) > 0.0
    norm_only = (truth[:, 3] > 0.0) & ~abnormal
    group = torch.where(
        norm_only,
        torch.zeros_like(truth[:, 3], dtype=torch.long),
        torch.where(
            abnormal,
            torch.ones_like(truth[:, 3], dtype=torch.long),
            torch.full_like(truth[:, 3], 2, dtype=torch.long),
        ),
    )
    intersection = truth @ truth.transpose(0, 1)
    union = (
        truth.sum(dim=1, keepdim=True)
        + truth.sum(dim=1, keepdim=True).transpose(0, 1)
        - intersection
    )
    similarity = intersection / union.clamp_min(1.0)
    compatible = group[:, None] == group[None, :]
    compatible.fill_diagonal_(False)
    any_compatible = compatible.any(dim=1)
    nonself = ~torch.eye(batch_size, dtype=torch.bool, device=truth.device)
    eligible = torch.where(any_compatible[:, None], compatible, nonself)

    indices = torch.arange(batch_size, device=truth.device)
    circular = (indices[None, :] - indices[:, None]).remainder(batch_size)
    tie_break = (batch_size - circular).float() * 1.0e-7
    scores = (similarity + tie_break).masked_fill(~eligible, float("-inf"))
    partners = scores.argmax(dim=1)
    if bool(torch.any(partners == indices).item()):
        raise RuntimeError("manifold mixup selected an anchor as its own partner")
    return partners


def _soft_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("soft BCE logits and targets must align")
    loss = (F.softplus(logits.float()) - targets.float() * logits.float()).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("soft BCE produced a non-finite loss")
    return loss


def _compute_clean_with_manifold_mixup(
    original_objective: Any,
    kwargs: Mapping[str, Any],
    *,
    pairrank: bool,
):
    """Add one clean feature-space mixup term without adding optimizer steps."""

    weight = _active_hyperparameter("clean_mixup_weight", 0.0)
    if weight <= 0.0:
        return (
            _compute_with_pairrank(
                original_objective,
                kwargs,
                crossbatch_query=True,
                crossbatch_update=True,
            )
            if pairrank
            else original_objective(**dict(kwargs))
        )
    model = kwargs.get("model")
    if not isinstance(model, nn.Module):
        raise TypeError("clean manifold mixup requires a torch model")
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        if len(arguments) != 1 or arguments[0].ndim != 2:
            raise RuntimeError("managed classifier head received unexpected features")
        captured.append(arguments[0])

    head = _classifier_head(model)
    handle = head.register_forward_pre_hook(capture)
    try:
        objective = (
            _compute_with_pairrank(
                original_objective,
                kwargs,
                crossbatch_query=True,
                crossbatch_update=True,
            )
            if pairrank
            else original_objective(**dict(kwargs))
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"clean manifold mixup expected one feature tensor, got {len(captured)}"
        )
    clean = kwargs["bundle"].require("clean_view")
    labels = clean.labels.float()
    features = captured[0]
    if features.ndim != 2 or int(features.shape[0]) != int(labels.shape[0]):
        raise RuntimeError("clean manifold feature count differs from labels")
    partners = _semantic_partner_indices(labels)
    anchor_mass = _active_hyperparameter("clean_mixup_anchor_mass", 0.75)
    partner_features = features.index_select(0, partners)
    partner_labels = labels.index_select(0, partners)
    mixed_features = anchor_mass * features + (1.0 - anchor_mass) * partner_features
    mixed_labels = anchor_mass * labels + (1.0 - anchor_mass) * partner_labels
    mixup_loss = _soft_bce_with_logits(head(mixed_features), mixed_labels)
    _STATE.clean_mixup_losses.append(_finite_scalar(mixup_loss, "clean mixup loss"))
    _STATE.clean_mixup_pair_counts.append(int(labels.shape[0]))
    result_type = type(objective)
    return result_type(
        total=objective.total + weight * mixup_loss,
        raw_terms=dict(objective.raw_terms),
        weighted_terms=dict(objective.weighted_terms),
        valid_counts=dict(objective.valid_counts),
    )


def _feature_pathmix_loss(
    *,
    model: nn.Module,
    captured: Sequence[torch.Tensor],
    bundle: Any,
) -> tuple[torch.Tensor, int]:
    """Supervise intermediate clean-to-D19 endpoint features."""

    if len(captured) != len(SUPCON_CAPTURE_ORDER):
        raise RuntimeError("feature pathmix received an incomplete D19 feature bundle")
    feature_by_view = dict(zip(SUPCON_CAPTURE_ORDER, captured, strict=True))
    clean = bundle.require("clean_view")
    clean_features = feature_by_view["clean_view"]
    labels = clean.labels.float()
    batch_size = int(labels.shape[0])
    if clean_features.shape[0] != batch_size:
        raise RuntimeError("feature pathmix clean feature count differs from labels")
    anchor_mass = _active_hyperparameter("pathmix_anchor_mass", 0.75)
    head = _classifier_head(model)
    losses: list[torch.Tensor] = []
    valid_count = 0
    for name in FEATURE_PATHMIX_QUERY_VIEWS:
        view = bundle.require(name)
        positions = torch.nonzero(view.valid_mask, as_tuple=False).flatten()
        if not int(positions.numel()):
            continue
        endpoint = feature_by_view[name]
        if int(endpoint.shape[0]) == batch_size:
            endpoint = endpoint.index_select(0, positions)
        elif int(endpoint.shape[0]) != int(positions.numel()):
            raise RuntimeError(f"feature pathmix {name} count differs from validity")
        anchor = clean_features.index_select(0, positions)
        targets = labels.index_select(0, positions)
        mixed = anchor_mass * anchor + (1.0 - anchor_mass) * endpoint
        losses.append(_soft_bce_with_logits(head(mixed), targets))
        valid_count += int(positions.numel())
    if not losses:
        return clean_features.sum() * 0.0, 0
    return torch.stack(losses).mean(), valid_count


def _multiview_supcon_loss(
    *,
    captured: Sequence[torch.Tensor],
    bundle: Any,
    regularizer_id: str = SUPCON_LOSS_ID,
    memory_capacity: int = 0,
    sample_balanced: bool = True,
) -> tuple[torch.Tensor, int, int, float, int]:
    """Contrast generated queries against stop-gradient clean feature keys.

    Exact-label peers are additional positives, while the corresponding clean
    representation of every query is always a positive.  This retains
    instance identity, supplies negatives that plain feature MSE lacks, and
    cannot collapse to one constant representation.
    """

    if len(captured) != len(SUPCON_CAPTURE_ORDER):
        raise RuntimeError(
            "SupCon expected one feature capture for each D19 objective view, "
            f"got {len(captured)} for {SUPCON_CAPTURE_ORDER}"
        )
    feature_by_view = dict(zip(SUPCON_CAPTURE_ORDER, captured, strict=True))
    clean = bundle.require("clean_view")
    batch_size = int(clean.waveform.shape[0])
    labels = clean.labels.float()
    clean_features = feature_by_view["clean_view"]
    if clean_features.ndim != 2 or int(clean_features.shape[0]) != batch_size:
        raise RuntimeError("SupCon clean feature capture violates batch identity")
    current_keys = F.normalize(clean_features.float(), dim=1).detach()
    keys = current_keys
    key_labels = labels
    prior_memory_size = 0
    if regularizer_id == MULSUPCON_LOSS_ID:
        memory_features = _STATE.supcon_memory_features
        memory_labels = _STATE.supcon_memory_labels
        if (memory_features is None) != (memory_labels is None):
            raise RuntimeError("MulSupCon feature and label memories diverged")
        if memory_features is not None and memory_labels is not None:
            if (
                memory_features.ndim != 2
                or memory_labels.ndim != 2
                or int(memory_features.shape[0]) != int(memory_labels.shape[0])
                or int(memory_features.shape[1]) != int(current_keys.shape[1])
                or int(memory_labels.shape[1]) != int(labels.shape[1])
            ):
                raise RuntimeError("MulSupCon memory violates feature-label shape")
            prior_memory_size = int(memory_features.shape[0])
            keys = torch.cat(
                [current_keys, memory_features.to(current_keys.device)], dim=0
            )
            key_labels = torch.cat(
                [labels, memory_labels.to(labels.device)], dim=0
            )
    elif regularizer_id != SUPCON_LOSS_ID:
        raise ValueError(f"unsupported feature regularizer {regularizer_id!r}")
    terms: list[torch.Tensor] = []
    query_count = 0
    label_anchor_count = 0
    positive_count_sum = 0.0
    for name in SUPCON_QUERY_VIEWS:
        view = bundle.require(name)
        positions = torch.nonzero(view.valid_mask, as_tuple=False).flatten()
        if not int(positions.numel()):
            continue
        features = feature_by_view[name]
        if features.ndim != 2:
            raise RuntimeError(f"SupCon {name} features must be two-dimensional")
        if int(features.shape[0]) == batch_size:
            query_features = features.index_select(0, positions)
        elif int(features.shape[0]) == int(positions.numel()):
            query_features = features
        else:
            raise RuntimeError(f"SupCon {name} feature count does not match validity")
        queries = F.normalize(query_features.float(), dim=1)
        query_labels = labels.index_select(0, positions)
        similarities = queries @ keys.transpose(0, 1) / _active_hyperparameter(
            "supcon_temperature", SUPCON_TEMPERATURE
        )
        denominator = torch.logsumexp(similarities, dim=1)
        if regularizer_id == SUPCON_LOSS_ID:
            positive_mask = torch.all(
                query_labels[:, None, :] == labels[None, :, :], dim=2
            )
            # Same-record clean keys make every row valid even when its exact
            # label set is unique within the batch.
            positive_mask[
                torch.arange(int(positions.numel()), device=positions.device),
                positions,
            ] = True
            positive_logsumexp = torch.logsumexp(
                similarities.masked_fill(~positive_mask, float("-inf")), dim=1
            )
            terms.append((denominator - positive_logsumexp).mean())
            positive_count_sum += float(positive_mask.sum().item())
            label_anchor_count += int(positions.numel())
        else:
            log_probabilities = similarities - denominator[:, None]
            view_label_terms: list[torch.Tensor] = []
            view_record_terms: list[torch.Tensor] = []
            for local_index, record_position in enumerate(positions.tolist()):
                active_classes = torch.nonzero(
                    query_labels[local_index] > 0.5, as_tuple=False
                ).flatten()
                per_record: list[torch.Tensor] = []
                for class_index in active_classes.tolist():
                    positive_mask = key_labels[:, class_index] > 0.5
                    positive_count = int(positive_mask.sum().item())
                    if positive_count <= 0:
                        raise RuntimeError(
                            "MulSupCon active label has no clean positive key"
                        )
                    per_label = -log_probabilities[
                        local_index, positive_mask
                    ].mean()
                    per_record.append(per_label)
                    view_label_terms.append(per_label)
                    positive_count_sum += float(positive_count)
                    label_anchor_count += 1
                if not per_record:
                    # K500 is expected to be non-zero-labelled, but keeping a
                    # deterministic instance-positive fallback makes the loss
                    # total for diagnostic all-zero anchors.
                    fallback = -log_probabilities[local_index, record_position]
                    per_record.append(fallback)
                    view_label_terms.append(fallback)
                    positive_count_sum += 1.0
                    label_anchor_count += 1
                view_record_terms.append(torch.stack(per_record).mean())
            selected_terms = (
                view_record_terms if sample_balanced else view_label_terms
            )
            if selected_terms:
                terms.append(torch.stack(selected_terms).mean())
        query_count += int(positions.numel())
    if not terms:
        return clean_features.sum() * 0.0, 0, 0, 0.0, prior_memory_size
    loss = torch.stack(terms).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("SupCon produced a non-finite loss")
    if regularizer_id == MULSUPCON_LOSS_ID:
        if memory_capacity <= 0:
            raise ValueError("MulSupCon requires a positive memory capacity")
        with torch.no_grad():
            stored_features = current_keys
            stored_labels = labels.detach()
            if _STATE.supcon_memory_features is not None:
                stored_features = torch.cat(
                    [_STATE.supcon_memory_features, stored_features], dim=0
                )
                assert _STATE.supcon_memory_labels is not None
                stored_labels = torch.cat(
                    [_STATE.supcon_memory_labels, stored_labels], dim=0
                )
            _STATE.supcon_memory_features = stored_features[-memory_capacity:].detach()
            _STATE.supcon_memory_labels = stored_labels[-memory_capacity:].detach()
    mean_positive_count = (
        positive_count_sum / label_anchor_count if label_anchor_count else 0.0
    )
    return (
        loss,
        query_count,
        label_anchor_count,
        mean_positive_count,
        prior_memory_size,
    )


def _compute_with_supcon(
    original_objective: Any,
    kwargs: Mapping[str, Any],
    *,
    weight: float,
    regularizer_id: str = SUPCON_LOSS_ID,
    memory_capacity: int = 0,
    sample_balanced: bool = True,
    pairrank: bool,
    balanced_bce: bool = False,
    group_reference_sink: list[float] | None = None,
):
    """Materialize the D19 objective and its feature regularizer in one graph."""

    model = kwargs.get("model")
    if not isinstance(model, nn.Module):
        raise TypeError("SupCon objective requires a torch model")
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        if len(arguments) != 1 or arguments[0].ndim != 2:
            raise RuntimeError("managed classifier head received unexpected features")
        captured.append(arguments[0])

    handle = _classifier_head(model).register_forward_pre_hook(capture)
    try:
        objective = (
            _compute_with_pairrank(
                original_objective,
                kwargs,
                balanced_bce=balanced_bce,
                crossbatch_query=True,
                crossbatch_update=False,
                group_reference_sink=group_reference_sink,
            )
            if pairrank
            else original_objective(**dict(kwargs))
        )
    finally:
        handle.remove()
    (
        loss,
        query_count,
        label_anchor_count,
        mean_positive_count,
        prior_memory_size,
    ) = _multiview_supcon_loss(
        captured=captured,
        bundle=kwargs["bundle"],
        regularizer_id=regularizer_id,
        memory_capacity=memory_capacity,
        sample_balanced=sample_balanced,
    )
    contribution = weight * loss
    _STATE.supcon_losses.append(_finite_scalar(loss, "SupCon loss"))
    _STATE.supcon_query_counts.append(query_count)
    _STATE.supcon_weights.append(float(weight))
    _STATE.supcon_loss_ids.append(regularizer_id)
    _STATE.supcon_label_anchor_counts.append(label_anchor_count)
    _STATE.supcon_positive_counts.append(mean_positive_count)
    _STATE.supcon_memory_sizes.append(prior_memory_size)
    pathmix_weight = _active_hyperparameter("pathmix_weight", 0.0)
    if pathmix_weight > 0.0:
        pathmix_loss, pathmix_count = _feature_pathmix_loss(
            model=model,
            captured=captured,
            bundle=kwargs["bundle"],
        )
        contribution = contribution + pathmix_weight * pathmix_loss
        _STATE.pathmix_losses.append(
            _finite_scalar(pathmix_loss, "feature pathmix loss")
        )
        _STATE.pathmix_valid_counts.append(pathmix_count)
    robust_view_weight = _active_hyperparameter(
        "robust_view_pairrank_weight", 0.0
    )
    if robust_view_weight > 0.0:
        (
            robust_view_loss,
            robust_view_valid_classes,
            robust_view_pair_count,
            robust_view_prior_memory,
        ) = _worst_view_pairwise_logistic_loss(
            model=model,
            captured=captured,
            bundle=kwargs["bundle"],
        )
        contribution = contribution + robust_view_weight * robust_view_loss
        _STATE.robust_view_losses.append(
            _finite_scalar(robust_view_loss, "worst-view PairRank loss")
        )
        _STATE.robust_view_valid_class_counts.append(
            robust_view_valid_classes
        )
        _STATE.robust_view_pair_counts.append(robust_view_pair_count)
        _STATE.robust_view_memory_sizes.append(robust_view_prior_memory)
    result_type = type(objective)
    return result_type(
        total=objective.total + contribution,
        # The feature term is deliberately reported in pcgrad_diagnostics.json
        # rather than masquerading as one of the compiler's BCE/JSD terms.
        raw_terms=dict(objective.raw_terms),
        weighted_terms=dict(objective.weighted_terms),
        valid_counts=dict(objective.valid_counts),
    )


def _rescale_candidate_auxiliary(
    objective: Any,
    wrapper_method: CompiledMethod,
) -> Any:
    scales = _auxiliary_term_scales(wrapper_method)
    if all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        for value in scales.values()
    ):
        return objective
    weighted_terms = dict(objective.weighted_terms)
    missing = sorted(set(scales).difference(weighted_terms))
    if missing:
        raise RuntimeError(
            f"scaled auxiliary objective is missing weighted terms: {missing}"
        )
    total = objective.total
    for name, scale in scales.items():
        original = weighted_terms[name]
        scaled = original * scale
        weighted_terms[name] = scaled
        total = total + (scaled - original)
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("scaled auxiliary objective became NaN or Inf")
    result_type = type(objective)
    return result_type(
        total=total,
        raw_terms=dict(objective.raw_terms),
        weighted_terms=weighted_terms,
        valid_counts=dict(objective.valid_counts),
    )


def _compute_candidate_auxiliary(
    original_objective: Any,
    kwargs: Mapping[str, Any],
    *,
    wrapper_method: CompiledMethod,
    pairrank: bool,
    balanced_bce: bool,
    group_reference_sink: list[float] | None = None,
):
    if wrapper_method.profile_name in SUPCON_METHOD_IDS:
        regularizer_id = str(
            wrapper_method.contracts["feature_regularizer"]
        )
        memory_capacity = int(
            wrapper_method.contracts.get(
                "feature_regularizer_memory_capacity", 0
            )
        )
        sample_balanced = (
            wrapper_method.contracts.get(
                "feature_regularizer_anchor_weighting",
                "equal_per_record",
            )
            == "equal_per_record"
        )
        objective = _compute_with_supcon(
            original_objective,
            kwargs,
            weight=_runtime_hyperparameters(wrapper_method)["supcon_weight"],
            regularizer_id=regularizer_id,
            memory_capacity=memory_capacity,
            sample_balanced=sample_balanced,
            pairrank=pairrank,
            balanced_bce=balanced_bce,
            group_reference_sink=group_reference_sink,
        )
    elif pairrank:
        objective = _compute_with_pairrank(
            original_objective,
            kwargs,
            balanced_bce=balanced_bce,
            group_reference_sink=group_reference_sink,
        )
    else:
        objective = original_objective(**dict(kwargs))
    return _rescale_candidate_auxiliary(objective, wrapper_method)


def _apply_projected_auxiliary_and_clip(
    parameters: Sequence[nn.Parameter],
    max_norm: float,
    norm_type: float,
    original_clip: Any,
    foreach: bool | None,
):
    if not _STATE.pending_aux_grads:
        return original_clip(parameters, max_norm, norm_type=norm_type, foreach=foreach)
    if len(parameters) != len(_STATE.pending_parameters) or any(
        left is not right
        for left, right in zip(parameters, _STATE.pending_parameters, strict=True)
    ):
        raise RuntimeError("PCGrad parameter identity changed before optimizer step")
    alpha = _STATE.pending_alpha
    if alpha is None:
        raise RuntimeError("PCGrad auxiliary alpha is missing")
    projection_mode = _STATE.pending_projection_mode
    if projection_mode not in {
        PCGRAD_MODE,
        VAE_POST_MODE,
        DIRECT_SUM_MODE,
        RESIDUALIZED_MODE,
    }:
        raise RuntimeError("PCGrad projection mode is missing or invalid")
    norm_ratio_cap = _STATE.pending_norm_ratio_cap
    if projection_mode == RESIDUALIZED_MODE and norm_ratio_cap is None:
        raise RuntimeError("residualized auxiliary norm cap is missing")
    if (
        projection_mode in {PCGRAD_MODE, VAE_POST_MODE, DIRECT_SUM_MODE}
        and norm_ratio_cap is None
    ):
        norm_ratio_cap = 0.0
    assert norm_ratio_cap is not None

    reference = next(
        (parameter for parameter in parameters if parameter.grad is not None), None
    )
    if reference is None:
        raise RuntimeError("fixed20 base gradient is empty")
    device = reference.grad.device
    base_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    aux_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    dot = torch.zeros((), device=device, dtype=torch.float32)
    for parameter, auxiliary in zip(parameters, _STATE.pending_aux_grads):
        base = parameter.grad
        if base is not None:
            base_float = base.detach().float()
            base_norm_sq += torch.sum(base_float * base_float)
        if auxiliary is not None:
            auxiliary_float = auxiliary.float()
            aux_norm_sq += torch.sum(auxiliary_float * auxiliary_float)
            if base is not None:
                dot += torch.sum(base.detach().float() * auxiliary_float)

    conflict = bool((dot < 0.0).item())
    projection_applied = (
        projection_mode == RESIDUALIZED_MODE
        or (
            projection_mode in {PCGRAD_MODE, VAE_POST_MODE}
            and conflict
        )
    )
    coefficient = (
        dot / (base_norm_sq + EPSILON) if projection_applied else dot * 0.0
    )
    with torch.no_grad():
        projected_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        post_projection_dot = torch.zeros((), device=device, dtype=torch.float32)
        projected_gradients: list[torch.Tensor | None] = []
        for parameter, auxiliary in zip(parameters, _STATE.pending_aux_grads):
            if auxiliary is None:
                projected_gradients.append(None)
                continue
            projected = auxiliary
            if projection_applied and parameter.grad is not None:
                projected = auxiliary - coefficient.to(auxiliary.dtype) * parameter.grad
            projected_gradients.append(projected)
            projected_norm_sq += torch.sum(projected.float() * projected.float())
            if parameter.grad is not None:
                post_projection_dot += torch.sum(
                    projected.float() * parameter.grad.detach().float()
                )

        base_norm = torch.sqrt(base_norm_sq)
        aux_norm = torch.sqrt(aux_norm_sq)
        projected_norm = torch.sqrt(projected_norm_sq)
        addition_scale = torch.as_tensor(
            float(alpha), device=device, dtype=torch.float32
        )
        cap_saturated = False
        maximum_addition_norm: torch.Tensor | None = None

    if projection_mode == RESIDUALIZED_MODE:
        if not math.isclose(float(norm_type), 2.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("residualized auxiliary requires L2 gradient clipping")
        if not math.isfinite(float(max_norm)) or float(max_norm) <= 0.0:
            raise ValueError("residualized auxiliary requires a finite positive max_norm")

        # Match Direct exactly first.  A second clip after adding an orthogonal
        # residual would rescale this base component and invalidate the matched
        # control even though the auxiliary/base dot product is zero.
        clipped_norm = original_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            foreach=foreach,
        )
        with torch.no_grad():
            clipped_base_norm_sq = torch.zeros(
                (), device=device, dtype=torch.float32
            )
            for parameter in parameters:
                if parameter.grad is not None:
                    clipped = parameter.grad.detach().float()
                    clipped_base_norm_sq += torch.sum(clipped * clipped)
            clipped_base_norm = torch.sqrt(clipped_base_norm_sq)
            direct_clip_scale = torch.where(
                base_norm > EPSILON,
                clipped_base_norm / (base_norm + EPSILON),
                torch.ones_like(base_norm),
            )
            maximum_addition_norm = float(norm_ratio_cap) * clipped_base_norm
            requested_addition_norm = addition_scale * projected_norm
            cap_saturated = bool(
                (requested_addition_norm > maximum_addition_norm + EPSILON).item()
            )
            if cap_saturated:
                addition_scale = maximum_addition_norm / (
                    projected_norm + EPSILON
                )

            addition_post_clip_dot = torch.zeros(
                (), device=device, dtype=torch.float32
            )
            for parameter, projected in zip(parameters, projected_gradients):
                if projected is None:
                    continue
                addition = projected.mul(addition_scale.to(projected.dtype))
                if parameter.grad is not None:
                    addition_post_clip_dot += torch.sum(
                        addition.float() * parameter.grad.detach().float()
                    )
                    parameter.grad.add_(addition)
                else:
                    parameter.grad = addition.clone()

            final_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
            for parameter in parameters:
                if parameter.grad is not None:
                    final = parameter.grad.detach().float()
                    final_norm_sq += torch.sum(final * final)
            final_norm = torch.sqrt(final_norm_sq)
            final_norm_bound = torch.sqrt(
                clipped_base_norm_sq + maximum_addition_norm.square()
            )
            addition_norm = addition_scale * projected_norm
            addition_post_clip_cosine = addition_post_clip_dot / (
                clipped_base_norm * addition_norm + EPSILON
            )
            effective_base_component_scale = 1.0 + addition_post_clip_dot / (
                clipped_base_norm_sq + EPSILON
            )
            base_clip_applied = bool(
                (clipped_base_norm + EPSILON < base_norm).item()
            )
    else:
        # Preserve legacy PCGrad byte/semantic behavior: add first, then clip the
        # combined gradient once.  The gradient-balanced profile additionally
        # caps the projected auxiliary addition against the current unclipped
        # Direct+fixed20 base norm before this legacy combined clip.
        with torch.no_grad():
            if float(norm_ratio_cap) > 0.0:
                maximum_addition_norm = float(norm_ratio_cap) * base_norm
                requested_addition_norm = addition_scale * projected_norm
                cap_saturated = bool(
                    (
                        requested_addition_norm
                        > maximum_addition_norm + EPSILON
                    ).item()
                )
                if cap_saturated:
                    addition_scale = maximum_addition_norm / (
                        projected_norm + EPSILON
                    )
            for parameter, projected in zip(parameters, projected_gradients):
                if projected is None:
                    continue
                addition = projected.mul(addition_scale.to(projected.dtype))
                if parameter.grad is None:
                    parameter.grad = addition.clone()
                else:
                    parameter.grad.add_(addition)
        clipped_norm = original_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            foreach=foreach,
        )
        with torch.no_grad():
            final_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
            for parameter in parameters:
                if parameter.grad is not None:
                    final = parameter.grad.detach().float()
                    final_norm_sq += torch.sum(final * final)
            final_norm = torch.sqrt(final_norm_sq)
        addition_norm = addition_scale * projected_norm

    cosine = dot / (base_norm * aux_norm + EPSILON)
    post_projection_cosine = post_projection_dot / (
        base_norm * projected_norm + EPSILON
    )
    step: dict[str, float | bool | int | str] = {
        "step": len(_STATE.steps) + 1,
        "alpha": float(alpha),
        "projection_mode": projection_mode,
        "projection_applied": projection_applied,
        "norm_ratio_cap": float(norm_ratio_cap),
        "auxiliary_loss": float(_STATE.pending_aux_loss or 0.0),
        "base_grad_norm_before_clip": _finite_scalar(base_norm, "base norm"),
        "aux_grad_norm": _finite_scalar(aux_norm, "auxiliary norm"),
        "base_aux_dot": _finite_scalar(dot, "gradient dot"),
        "base_aux_cosine": _finite_scalar(cosine, "gradient cosine"),
        "conflict": conflict,
        "projection_coefficient": _finite_scalar(
            coefficient, "projection coefficient"
        ),
        "post_projection_dot": _finite_scalar(
            post_projection_dot, "post-projection gradient dot"
        ),
        "post_projection_cosine": _finite_scalar(
            post_projection_cosine, "post-projection cosine"
        ),
        "projected_aux_grad_norm": _finite_scalar(
            projected_norm, "projected auxiliary norm"
        ),
        "addition_scale": _finite_scalar(
            addition_scale, "auxiliary addition scale"
        ),
        "addition_grad_norm": _finite_scalar(
            addition_norm, "auxiliary addition norm"
        ),
        "cap_saturated": cap_saturated,
        "combined_grad_norm_before_clip": _finite_scalar(
            torch.as_tensor(clipped_norm), "clip-returned norm"
        ),
        "combined_grad_norm_final": _finite_scalar(final_norm, "final norm"),
    }
    if projection_mode == RESIDUALIZED_MODE:
        step.update(
            {
                "base_grad_norm_after_direct_clip": _finite_scalar(
                    clipped_base_norm, "clipped base norm"
                ),
                "direct_base_clip_scale": _finite_scalar(
                    direct_clip_scale, "Direct base clip scale"
                ),
                "base_clip_applied": base_clip_applied,
                "maximum_addition_grad_norm": _finite_scalar(
                    maximum_addition_norm, "maximum auxiliary addition norm"
                ),
                "addition_post_direct_clip_dot": _finite_scalar(
                    addition_post_clip_dot, "post-Direct-clip addition dot"
                ),
                "addition_post_direct_clip_cosine": _finite_scalar(
                    addition_post_clip_cosine,
                    "post-Direct-clip addition cosine",
                ),
                "effective_base_component_scale": _finite_scalar(
                    effective_base_component_scale,
                    "effective Direct base component scale",
                ),
                "final_grad_norm_bound": _finite_scalar(
                    final_norm_bound, "final gradient norm bound"
                ),
                "second_combined_clip_applied": False,
            }
        )
    else:
        step["second_combined_clip_applied"] = True
        if maximum_addition_norm is not None:
            step["maximum_addition_grad_norm"] = _finite_scalar(
                maximum_addition_norm, "maximum auxiliary addition norm"
            )
    _STATE.steps.append(step)
    _STATE.clear_pending()
    return clipped_norm


def diagnostics_payload() -> dict[str, Any] | None:
    if not _STATE.steps and not _STATE.active_hyperparameters:
        return None
    steps = list(_STATE.steps)
    numeric_names = (
        "alpha",
        "norm_ratio_cap",
        "auxiliary_loss",
        "base_grad_norm_before_clip",
        "aux_grad_norm",
        "base_aux_dot",
        "base_aux_cosine",
        "projection_coefficient",
        "post_projection_dot",
        "post_projection_cosine",
        "projected_aux_grad_norm",
        "addition_scale",
        "addition_grad_norm",
        "combined_grad_norm_before_clip",
        "combined_grad_norm_final",
    )
    means = (
        {
            name: sum(float(step[name]) for step in steps) / len(steps)
            for name in numeric_names
        }
        if steps
        else {}
    )
    modes = sorted({str(step["projection_mode"]) for step in steps})
    optional_numeric_names = (
        "base_grad_norm_after_direct_clip",
        "direct_base_clip_scale",
        "maximum_addition_grad_norm",
        "addition_post_direct_clip_dot",
        "addition_post_direct_clip_cosine",
        "effective_base_component_scale",
        "final_grad_norm_bound",
    )
    for name in optional_numeric_names:
        values = [float(step[name]) for step in steps if name in step]
        if values:
            means[name] = sum(values) / len(values)
    payload = {
        "schema_version": 3,
        "wrapper_method_id": _STATE.wrapper_method_id,
        "mode": (
            modes[0]
            if len(modes) == 1
            else ("disabled_no_auxiliary" if not modes else "mixed")
        ),
        "modes": modes,
        "step_count": len(steps),
        "conflict_count": sum(bool(step["conflict"]) for step in steps),
        "conflict_fraction": (
            sum(bool(step["conflict"]) for step in steps) / len(steps)
            if steps
            else 0.0
        ),
        "projection_count": sum(
            bool(step["projection_applied"]) for step in steps
        ),
        "projection_fraction": (
            sum(bool(step["projection_applied"]) for step in steps) / len(steps)
            if steps
            else 0.0
        ),
        "cap_saturation_count": sum(bool(step["cap_saturated"]) for step in steps),
        "cap_saturation_fraction": (
            sum(bool(step["cap_saturated"]) for step in steps) / len(steps)
            if steps
            else 0.0
        ),
        "means": means,
        "pending_auxiliary_at_exit": bool(_STATE.pending_aux_grads),
        "steps": steps,
    }
    if _STATE.wrapper_method_id == SEARCH_METHOD_ID:
        candidate_replay_count = int(
            _active_hyperparameter("candidate_replay_count", 0.0)
        )
        payload["candidate_replay"] = {
            "count": candidate_replay_count,
            "policy": CANDIDATE_REPLAY_POLICY,
            "composition_indices": list(
                CANDIDATE_REPLAY_ORDER[:candidate_replay_count]
            ),
            "fixed20_composition_count": 20 - candidate_replay_count,
            "optimizer_step_budget_preserved": True,
        }
    if _STATE.wrapper_method_id in D19_AUXILIARY_METHOD_IDS:
        auxiliary_term_scales = {
            name: _active_hyperparameter(
                f"auxiliary_term_scale_{name}", 1.0
            )
            for name in D19_AUXILIARY_BCE_WEIGHTS
        }
        payload["auxiliary_term_scaling"] = {
            "contract": AUXILIARY_TERM_SCALE_CONTRACT,
            "bce_mass_contract": AUXILIARY_BCE_MASS_CONTRACT,
            "base_bce_weights": dict(D19_AUXILIARY_BCE_WEIGHTS),
            "term_scales": auxiliary_term_scales,
            "effective_bce_weights": {
                name: D19_AUXILIARY_BCE_WEIGHTS[name] * scale
                for name, scale in auxiliary_term_scales.items()
            },
            "total_bce_mass_preserved": True,
        }
    if _STATE.group_dro_cycles:
        cycles = list(_STATE.group_dro_cycles)
        final_probabilities = cycles[-1]["probabilities_next"]
        payload["group_dro"] = {
            "mode": GROUP_DRO_MODE,
            "step_size": _active_hyperparameter(
                "group_dro_step_size", GROUP_DRO_STEP_SIZE
            ),
            "uniform_mix": _active_hyperparameter(
                "group_dro_uniform_mix", GROUP_DRO_UNIFORM_MIX
            ),
            "cycle_count": len(cycles),
            "total_corrupted_mass_preserved": True,
            "final_probabilities": final_probabilities,
            "final_scales": [
                GROUP_DRO_GROUP_COUNT * float(value)
                for value in final_probabilities
            ],
            "cycles": cycles,
        }
    if _STATE.pairrank_losses:
        payload["pairrank"] = {
            "loss_id": PAIRRANK_LOSS_ID,
            "weight": _active_hyperparameter("pairrank_weight", PAIRRANK_WEIGHT),
            "temperature": _active_hyperparameter(
                "pairrank_temperature", PAIRRANK_TEMPERATURE
            ),
            "reduction": PAIRRANK_REDUCTION,
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "class_weights": list(_STATE.pairrank_class_weights),
            "hard_fraction": _STATE.pairrank_hard_fraction,
            "mean_class_losses": [
                total / max(count, 1)
                for total, count in zip(
                    _STATE.pairrank_class_loss_sums,
                    _STATE.pairrank_class_loss_counts,
                    strict=True,
                )
            ],
            "call_count": len(_STATE.pairrank_losses),
            "mean_raw_loss": sum(_STATE.pairrank_losses)
            / len(_STATE.pairrank_losses),
            "mean_valid_class_count": sum(_STATE.pairrank_valid_class_counts)
            / len(_STATE.pairrank_valid_class_counts),
            "zero_valid_class_calls": sum(
                value == 0 for value in _STATE.pairrank_valid_class_counts
            ),
        }
    if _STATE.balanced_bce_losses:
        if _STATE.balanced_bce_scope is None:
            raise RuntimeError("balanced-BCE diagnostics lost their routing scope")
        payload["balanced_bce"] = {
            "loss_id": BALANCED_BCE_LOSS_ID,
            "scope": _STATE.balanced_bce_scope,
            "reduction": BALANCED_BCE_REDUCTION,
            "call_count": len(_STATE.balanced_bce_losses),
            "mean_loss": sum(_STATE.balanced_bce_losses)
            / len(_STATE.balanced_bce_losses),
        }
    if _STATE.supcon_losses:
        unique_weights = sorted(set(_STATE.supcon_weights))
        if len(unique_weights) != 1:
            raise RuntimeError(f"one run used multiple SupCon weights: {unique_weights}")
        unique_loss_ids = sorted(set(_STATE.supcon_loss_ids))
        if len(unique_loss_ids) != 1:
            raise RuntimeError(
                f"one run used multiple SupCon losses: {unique_loss_ids}"
            )
        loss_id = unique_loss_ids[0]
        positive_policy = (
            "per_active_label_clean_keys_including_same_record"
            if loss_id == MULSUPCON_LOSS_ID
            else "exact_positive_set_including_same_record"
        )
        payload["supcon"] = {
            "loss_id": loss_id,
            "weight": unique_weights[0],
            "temperature": _active_hyperparameter(
                "supcon_temperature", SUPCON_TEMPERATURE
            ),
            "query_views": list(SUPCON_QUERY_VIEWS),
            "key_gradient": "stop_gradient",
            "positive_policy": positive_policy,
            "call_count": len(_STATE.supcon_losses),
            "mean_raw_loss": sum(_STATE.supcon_losses)
            / len(_STATE.supcon_losses),
            "mean_query_count": sum(_STATE.supcon_query_counts)
            / len(_STATE.supcon_query_counts),
            "zero_query_calls": sum(
                value == 0 for value in _STATE.supcon_query_counts
            ),
            "mean_label_anchor_count": sum(_STATE.supcon_label_anchor_counts)
            / len(_STATE.supcon_label_anchor_counts),
            "mean_positive_keys_per_anchor": sum(_STATE.supcon_positive_counts)
            / len(_STATE.supcon_positive_counts),
            "mean_prior_memory_size": sum(_STATE.supcon_memory_sizes)
            / len(_STATE.supcon_memory_sizes),
        }
    if _STATE.clean_mixup_losses:
        payload["clean_manifold_mixup"] = {
            "policy": MANIFOLD_MIXUP_POLICY,
            "weight": _active_hyperparameter("clean_mixup_weight", 0.0),
            "anchor_mass": _active_hyperparameter(
                "clean_mixup_anchor_mass", 0.75
            ),
            "call_count": len(_STATE.clean_mixup_losses),
            "mean_raw_loss": sum(_STATE.clean_mixup_losses)
            / len(_STATE.clean_mixup_losses),
            "mean_pair_count": sum(_STATE.clean_mixup_pair_counts)
            / len(_STATE.clean_mixup_pair_counts),
        }
    if _STATE.pathmix_losses:
        payload["feature_pathmix"] = {
            "query_views": list(FEATURE_PATHMIX_QUERY_VIEWS),
            "weight": _active_hyperparameter("pathmix_weight", 0.0),
            "anchor_mass": _active_hyperparameter("pathmix_anchor_mass", 0.75),
            "call_count": len(_STATE.pathmix_losses),
            "mean_raw_loss": sum(_STATE.pathmix_losses)
            / len(_STATE.pathmix_losses),
            "mean_valid_count": sum(_STATE.pathmix_valid_counts)
            / len(_STATE.pathmix_valid_counts),
        }
    if _STATE.crossbatch_losses:
        payload["crossbatch_pairrank"] = {
            "scope": CROSSBATCH_PAIRRANK_SCOPE,
            "memory_update": CROSSBATCH_MEMORY_UPDATE,
            "weight": _active_hyperparameter(
                "crossbatch_pairrank_weight", 0.0
            ),
            "temperature": _active_hyperparameter(
                "crossbatch_pairrank_temperature", 1.0
            ),
            "capacity": int(
                _active_hyperparameter("crossbatch_pairrank_capacity", 512.0)
            ),
            "hard_fraction": _active_hyperparameter(
                "crossbatch_pairrank_hard_fraction", 0.25
            ),
            "call_count": len(_STATE.crossbatch_losses),
            "mean_raw_loss": sum(_STATE.crossbatch_losses)
            / len(_STATE.crossbatch_losses),
            "mean_pair_count": sum(_STATE.crossbatch_pair_counts)
            / len(_STATE.crossbatch_pair_counts),
            "final_memory_size": (
                int(_STATE.crossbatch_memory_logits.shape[0])
                if _STATE.crossbatch_memory_logits is not None
                else 0
            ),
            "maximum_memory_size": max(_STATE.crossbatch_memory_sizes, default=0),
        }
    if _STATE.robust_view_losses:
        payload["robust_view_pairrank"] = {
            "loss_id": ROBUST_VIEW_PAIRRANK_LOSS_ID,
            "scope": ROBUST_VIEW_PAIRRANK_SCOPE,
            "memory_update": ROBUST_VIEW_PAIRRANK_MEMORY_UPDATE,
            "weight": _active_hyperparameter(
                "robust_view_pairrank_weight", 0.0
            ),
            "temperature": _active_hyperparameter(
                "robust_view_pairrank_temperature", 1.0
            ),
            "capacity": int(
                _active_hyperparameter(
                    "robust_view_pairrank_capacity", 512.0
                )
            ),
            "hard_fraction": _active_hyperparameter(
                "robust_view_pairrank_hard_fraction", 0.25
            ),
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "class_weights": list(_STATE.pairrank_class_weights),
            "call_count": len(_STATE.robust_view_losses),
            "mean_raw_loss": sum(_STATE.robust_view_losses)
            / len(_STATE.robust_view_losses),
            "mean_valid_class_count": sum(
                _STATE.robust_view_valid_class_counts
            )
            / len(_STATE.robust_view_valid_class_counts),
            "mean_pair_count": sum(_STATE.robust_view_pair_counts)
            / len(_STATE.robust_view_pair_counts),
            "mean_prior_memory_size": sum(_STATE.robust_view_memory_sizes)
            / len(_STATE.robust_view_memory_sizes),
            "final_memory_size": (
                int(_STATE.robust_view_memory_scores.shape[0])
                if _STATE.robust_view_memory_scores is not None
                else 0
            ),
        }
    return payload


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "pcgrad_diagnostics.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Install the fixed20/LHAT PCGrad bridge for one delegate subprocess."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_exposures = online_trainer._method_exposure_steps
    original_objective = online_trainer._compute_objective
    original_clip = torch.nn.utils.clip_grad_norm_
    _STATE.clear_runtime()
    _STATE.steps.clear()

    def patched_exposures(method: CompiledMethod):
        if not _is_pcgrad_method(method):
            return original_exposures(method)
        validate_pcgrad_method(method)
        auxiliary_terms = _active_auxiliary_terms(method)
        return (
            online_trainer._ExposureStep(
                "clean",
                -1,
                ("clean_bce", *auxiliary_terms),
                loss_scale=0.5,
            ),
            *tuple(
                online_trainer._ExposureStep(
                    f"corruption_{index:02d}",
                    index,
                    ("corrupted_bce",),
                    loss_scale=0.025,
                )
                for index in range(20)
            ),
        )

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
    ):
        if not _is_pcgrad_method(method):
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
        validate_pcgrad_method(method)
        root = Path(config_root).expanduser().resolve()
        base_path = (root / "train/methods/direct_depth23_fixed20.yaml").resolve()
        try:
            base_path.relative_to(root)
        except ValueError:
            raise ValueError("fixed20 base profile escapes the config root") from None
        base_method = compile_method_profile(base_path)
        if base_method.profile_name != BASE_METHOD_ID:
            raise ValueError("PCGrad base profile is not direct_depth23_fixed20")
        auxiliary_terms = _active_auxiliary_terms(method)
        if not auxiliary_terms:
            candidate_method = base_method
        elif method.profile_name in RESIDUALIZED_AUXILIARY_METHOD_IDS:
            candidate_profile = method.contracts["pcgrad_candidate_profile"]
            candidate_alias = method.contracts["pcgrad_candidate_method_id"]
            candidate_path = (root / str(candidate_profile)).resolve()
            try:
                candidate_path.relative_to(root)
            except ValueError:
                raise ValueError(
                    "residualized candidate profile escapes the config root"
                ) from None
            candidate_method = compile_method_profile(candidate_path)
            if candidate_method.profile_name != candidate_alias:
                raise ValueError("residualized candidate profile identity drifted")
        elif method.profile_name in COMPATIBLE_AUXILIARY_METHOD_IDS:
            candidate_profile = (
                str(method.contracts["pcgrad_candidate_profile"])
                if method.profile_name == SEARCH_METHOD_ID
                else CANDIDATE_PROFILE_BY_METHOD[method.profile_name]
            )
            candidate_alias = CANDIDATE_ALIAS_BY_METHOD[method.profile_name]
            candidate_path = (root / candidate_profile).resolve()
            try:
                candidate_path.relative_to(root)
            except ValueError:
                raise ValueError(
                    "compatible candidate profile escapes the config root"
                ) from None
            candidate_method = compile_method_profile(candidate_path)
            if candidate_method.profile_name != candidate_alias:
                raise ValueError("compatible candidate profile identity drifted")
        else:
            candidate_method = method
        if not auxiliary_terms:
            candidate_runtime = original_factory(
                base_method,
                model_name=model_name,
                config_root=root,
                latent_pool=None,
                encoder=None,
                decoder=None,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        elif method.profile_name in RESIDUALIZED_AUXILIARY_METHOD_IDS:
            # Keep the strict residualized candidate independent from the broad
            # historical monkeypatch chain.  Its paired seed/batch semantics are
            # explicit here and do not alter whitelist-wide RNG behavior.
            candidate_runtime = _ResidualizedCandidateRuntime(
                candidate_method,
                model_name=model_name,
                config_root=root,
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        elif (
            method.profile_name == SEARCH_METHOD_ID
            and "full_k500_heldout_oracle" in method.contracts
        ):
            import compatible_runtime

            with compatible_runtime.full_k500_heldout_oracle_scope(method):
                candidate_runtime = original_factory(
                    candidate_method,
                    model_name=model_name,
                    config_root=root,
                    latent_pool=latent_pool,
                    encoder=encoder,
                    decoder=decoder,
                    minimum_std_mV=minimum_std_mV,
                    maximum_abs_mV=maximum_abs_mV,
                )
        else:
            candidate_runtime = original_factory(
                candidate_method,
                model_name=model_name,
                config_root=root,
                latent_pool=latent_pool,
                encoder=encoder,
                decoder=decoder,
                minimum_std_mV=minimum_std_mV,
                maximum_abs_mV=maximum_abs_mV,
            )
        direct_runtime = original_factory(
            base_method,
            model_name=model_name,
            config_root=root,
            latent_pool=None,
            encoder=None,
            decoder=None,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )
        _STATE.base_method = base_method
        _STATE.candidate_method = candidate_method
        _STATE.wrapper_method_id = method.profile_name
        return _PCGradMethodRuntime(candidate_runtime, direct_runtime, method)

    def patched_objective(**kwargs: Any):
        method = kwargs.get("method")
        if not _is_pcgrad_method(method):
            return original_objective(**kwargs)
        assert isinstance(method, CompiledMethod)
        alpha = validate_pcgrad_method(method)
        pairrank = method.profile_name in PAIRRANK_METHOD_IDS
        balanced_bce_scope = _balanced_bce_scope(method)
        if balanced_bce_scope is not None:
            if _STATE.balanced_bce_scope is None:
                _STATE.balanced_bce_scope = balanced_bce_scope
            elif _STATE.balanced_bce_scope != balanced_bce_scope:
                raise RuntimeError("balanced-BCE routing scope changed within one run")
        balance_corrupted = balanced_bce_scope == BALANCED_BCE_SCOPE
        balance_auxiliary = balanced_bce_scope in {
            BALANCED_BCE_AUXILIARY_SCOPE,
            BALANCED_BCE_SCOPE,
        }
        if _STATE.base_method is None or _STATE.candidate_method is None:
            raise RuntimeError("PCGrad runtime methods were not initialized")
        auxiliary_terms = _active_auxiliary_terms(method)
        selected = tuple(kwargs.get("objective_term_names") or ())
        if selected == ("corrupted_bce",):
            composition = _STATE.current_composition
            if (
                composition is not None
                and _is_candidate_replay(method, composition)
            ):
                replay_kwargs = dict(kwargs)
                replay_kwargs["method"] = _STATE.candidate_method
                replay_kwargs["objective_term_names"] = auxiliary_terms
                replay_group_references: list[float] = []
                replay = _compute_candidate_auxiliary(
                    original_objective,
                    replay_kwargs,
                    wrapper_method=method,
                    pairrank=pairrank,
                    balanced_bce=balance_auxiliary,
                    group_reference_sink=replay_group_references,
                )
                if not _group_dro_active(method):
                    return replay
                if not replay_group_references:
                    raise RuntimeError(
                        "candidate replay GroupDRO requires at least one "
                        "macro-balanced BCE reference"
                    )
                return _group_dro_scaled_objective(
                    replay,
                    composition,
                    group_loss=(
                        sum(replay_group_references)
                        / len(replay_group_references)
                    ),
                )
            group_reference_start = None
            group_references: list[float] | None = None
            if _group_dro_active(method) and not pairrank:
                import asl_runtime

                group_reference_start = asl_runtime.macro_balanced_bce_call_count()
            elif _group_dro_active(method):
                group_references = []
            base_kwargs = dict(kwargs)
            base_kwargs["method"] = _STATE.base_method
            base = (
                _compute_with_pairrank(
                    original_objective,
                    base_kwargs,
                    group_reference_sink=group_references,
                    balanced_bce=balance_corrupted,
                )
                if pairrank
                else original_objective(**base_kwargs)
            )
            if not _group_dro_active(method):
                return base
            composition = _STATE.current_composition
            if composition is None or composition < 0:
                raise RuntimeError("GroupDRO corrupted objective lacks a group identity")
            if pairrank:
                if group_references is None or len(group_references) != 1:
                    raise RuntimeError(
                        "pairrank GroupDRO corrupted objective must record exactly "
                        "one balanced BCE reference"
                    )
                group_loss = group_references[0]
            else:
                assert group_reference_start is not None
                group_loss = asl_runtime.macro_balanced_bce_since(
                    group_reference_start
                )
            return _group_dro_scaled_objective(
                base,
                composition,
                group_loss=group_loss,
            )
        expected_clean = ("clean_bce", *auxiliary_terms)
        if selected != expected_clean:
            raise ValueError(f"unexpected PCGrad objective routing: {selected}")
        if _STATE.pending_aux_grads:
            raise RuntimeError("previous PCGrad auxiliary gradient was not consumed")

        if not auxiliary_terms:
            base_kwargs = dict(kwargs)
            base_kwargs["method"] = _STATE.base_method
            base_kwargs["objective_term_names"] = ("clean_bce",)
            return _compute_clean_with_manifold_mixup(
                original_objective,
                base_kwargs,
                pairrank=pairrank,
            )

        model = kwargs.get("model")
        if not isinstance(model, nn.Module):
            raise TypeError("PCGrad objective requires a torch model")
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        rng_state = _capture_rng_state(model)
        bn_state = _capture_bn_state(model)
        auxiliary_kwargs = dict(kwargs)
        auxiliary_kwargs["method"] = _STATE.candidate_method
        auxiliary_kwargs["objective_term_names"] = auxiliary_terms
        try:
            auxiliary = _compute_candidate_auxiliary(
                original_objective,
                auxiliary_kwargs,
                wrapper_method=method,
                pairrank=pairrank,
                balanced_bce=balance_auxiliary,
            )
            auxiliary_grads = torch.autograd.grad(
                auxiliary.total,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            _restore_bn_state(bn_state)
            _restore_rng_state(rng_state)

        base_kwargs = dict(kwargs)
        base_kwargs["method"] = _STATE.base_method
        base_kwargs["objective_term_names"] = ("clean_bce",)
        base = _compute_clean_with_manifold_mixup(
            original_objective,
            base_kwargs,
            pairrank=pairrank,
        )
        _STATE.pending_parameters = parameters
        _STATE.pending_aux_grads = tuple(
            None if gradient is None else gradient.detach()
            for gradient in auxiliary_grads
        )
        _STATE.pending_alpha = alpha
        _STATE.pending_projection_mode = str(
            method.contracts["pcgrad_auxiliary_mode"]
        )
        _STATE.pending_norm_ratio_cap = _runtime_hyperparameters(method)[
            "residual_norm_ratio_cap"
        ]
        _STATE.pending_aux_loss = _finite_scalar(
            auxiliary.total, "auxiliary objective"
        )
        return _merged_base_objective(base, auxiliary)

    def patched_clip_grad_norm_(
        parameters: Any,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ):
        if error_if_nonfinite:
            raise ValueError("PCGrad sandbox expects error_if_nonfinite=False")
        resolved = tuple(parameters)
        timer = _STATE.performance_timer
        if timer is not None:
            timer.start("gradient_combine")
        try:
            return _apply_projected_auxiliary_and_clip(
                resolved,
                float(max_norm),
                float(norm_type),
                original_clip,
                foreach,
            )
        finally:
            if timer is not None:
                timer.stop("gradient_combine")

    online_trainer._method_exposure_steps = patched_exposures
    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_objective
    torch.nn.utils.clip_grad_norm_ = patched_clip_grad_norm_
    try:
        yield
        if _STATE.pending_aux_grads:
            raise RuntimeError("PCGrad delegate exited with an unconsumed auxiliary")
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        online_trainer._compute_objective = original_objective
        online_trainer.build_method_runtime = original_factory
        online_trainer._method_exposure_steps = original_exposures
        _STATE.clear_runtime()


__all__ = [
    "BALANCED_BCE_LOSS_ID",
    "BALANCED_BCE_METHOD_IDS",
    "BALANCED_BCE_REDUCTION",
    "BALANCED_BCE_SCOPE",
    "METHOD_ALPHA",
    "METHOD_AUXILIARY_TERMS",
    "D13_AUXILIARY_METHOD_IDS",
    "D13_AUXILIARY_TERMS",
    "D19_AUXILIARY_METHOD_IDS",
    "D19_AUXILIARY_TERMS",
    "HARD_ONLY_PRUNED_METHOD_ID",
    "HARD_ONLY_PRUNED_METHOD_IDS",
    "GROUP_DRO_METHOD_IDS",
    "GROUP_DRO_MODE",
    "GROUP_DRO_LOSS_SOURCE",
    "DIRECT_LHAT_METHOD_IDS",
    "METHOD_IDS",
    "PCGRAD_MODE",
    "DIRECT_SUM_MODE",
    "RESIDUALIZED_AUXILIARY_METHOD_IDS",
    "RESIDUALIZED_METHOD_ID",
    "RESIDUALIZED_MODE",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "validate_pcgrad_method",
    "write_diagnostics",
]
