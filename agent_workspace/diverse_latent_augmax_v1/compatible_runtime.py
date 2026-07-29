"""Scoped compatible-latent runtime for D10 through D19 and F2.

This is an isolated sandbox sidecar.  It reuses the whitelist graph, trainer,
canonical corruption operators, VAE decoder, latent pool, model adapters and
run recorder without changing any whitelist source.  The only altered runtime
semantics are:

* ``compat_random`` and ``compat_hard`` reuse the registered
  ``vae_lhat_hard_view`` node type but dispatch by the fixed node ID;
* tuning candidates come only from the frozen train400 pool; historical
  D10--D18 arms retain their ``NORM_ONLY``/``ABNORMAL``/``OTHER`` grouping,
  while F2 uses the plan-required exact multi-hot positive set;
* D10-D15 retain their frozen shared candidate-selection seed semantics;
* D16-D17 derive independent node-role candidate streams (matched across
  methods for shared role names) and method-and-node-specific Dirichlet streams;
* D18 reuses the D13 graph, BatchNorm plan, candidate stream and geometry while
  moving the random-view BCE mass to the hard view; the random view is still
  generated and remains one of the seven JSD views;
* D19 uses one raw, one random-latent and one multi-start hard-latent branch;
  its hard branch starts from both anchor-biased and random coefficients and
  selects per record against the unoptimized random endpoint, so it is never
  easier than its paired random branch;
* F2 keeps D13's outer objective/optimizer budget but gives each latent view a
  fixed 50/50 convex combination of a compatible-real direction and one of the
  four already-generated raw-corruption directions encoded exactly once;
* soft anchor targets live in ``WaveformView.metadata`` while the typed view's
  ordinary labels remain identical to the clean labels; and
* a scoped objective adapter consumes those soft targets and applies the one
  declared 0.5 global scale.

Full-K500 remains rejected by default.  D13 may use a separately encoded
full-K500 pool only inside a process-local selected-refit scope whose external
promotion sidecar, D13 selection, method/source/VAE identities, E37/T40
contract and matched fixed20-T40 gate all verify before model construction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from core.augmix import multilabel_jsd
from core.corruption import CANONICAL_OPERATORS
from core.methods.contracts import (
    BASE_VIEW_NAME,
    NodeContext,
    Provenance,
    ViewValue,
    WaveformView,
)
from core.methods.registry import CompiledMethod
from core.methods.runtime import MethodViewRuntime, _quality_mask
from models.contracts import ModelSpec, validate_model_output
from models.factory import get_model_spec
from models.input_adapter import prepare_canonical_model_input
from models.vae import decode_to_ptbxl_waveform, prepare_ecgtwin_encoder_input
from runtime_adapter import _frozen_classifier_for_attack, _frozen_vae_components


SANDBOX_ROOT = Path(__file__).resolve().parent
D10_METHOD_ID = "diverse_augmax_d10_compat_hard_v1"
D11_METHOD_ID = "diverse_augmax_d11_compat_random_hard_v1"
D12_METHOD_ID = "diverse_augmax_d12_compat_random_hard_raw40_jsd30_r2_v1"
D13_METHOD_ID = "diverse_augmax_d13_compat_random_hard_raw40_jsd30_r4_v1"
D14_METHOD_ID = (
    "diverse_augmax_d14_compat_random_hard_clean55_raw40_latent05_jsd30_r4_v1"
)
D15_METHOD_ID = "diverse_augmax_d15_compat_random_hard_raw40_jsd30_r8_v1"
D16_METHOD_ID = "diverse_augmax_d16_compat_random2_independent_raw40_jsd30_r4_v1"
D17_METHOD_ID = "diverse_augmax_d17_compat_random3_multishell_raw40_jsd30_r4_v1"
D18_METHOD_ID = "diverse_augmax_d18_compat_hard_focus_raw40_jsd30_r4_v1"
D19_METHOD_ID = "diverse_augmax_d19_multistart_threechain_feature_v1"
F2_METHOD_ID = "diverse_augmax_f2_twoaxis_raw40_jsd30_r4_v1"
SUPPORTED_METHOD_IDS = frozenset(
    {
        D10_METHOD_ID,
        D11_METHOD_ID,
        D12_METHOD_ID,
        D13_METHOD_ID,
        D14_METHOD_ID,
        D15_METHOD_ID,
        D16_METHOD_ID,
        D17_METHOD_ID,
        D18_METHOD_ID,
        D19_METHOD_ID,
        F2_METHOD_ID,
    }
)
CONTRACT_KEY = "compatible_runtime"
FULL_K500_ORACLE_CONTRACT_KEY = "full_k500_heldout_oracle"
FULL_K500_ORACLE_OUTER_METHOD_ID = "fixed20_search_d19_v1"
OBJECTIVE_GLOBAL_SCALE = 0.5


def _generated_search_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-record difficulty used only by latent coefficient search.

    The default deliberately remains the historical five-label mean BCE.  A
    process-scoped sandbox adapter may replace this hook for a generated-view
    search ablation; the supervised trainer never calls it.
    """

    return F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=1)

_METHOD_MODES = {
    D10_METHOD_ID: "d10_compat_hard",
    D11_METHOD_ID: "d11_compat_random_hard",
    D12_METHOD_ID: "d12_compat_random_hard_raw40_jsd30_r2",
    D13_METHOD_ID: "d13_compat_random_hard_raw40_jsd30_r4",
    D14_METHOD_ID: "d14_compat_random_hard_clean55_raw40_latent05_jsd30_r4",
    D15_METHOD_ID: "d15_compat_random_hard_raw40_jsd30_r8",
    D16_METHOD_ID: "d16_compat_random2_independent_raw40_jsd30_r4",
    D17_METHOD_ID: "d17_compat_random3_multishell_raw40_jsd30_r4",
    D18_METHOD_ID: "d18_compat_hard_focus_raw40_jsd30_r4",
    D19_METHOD_ID: "d19_multistart_threechain_feature",
    F2_METHOD_ID: "f2_twoaxis_raw40_jsd30_r4",
}
_CORRUPTION_VIEW_COUNTS = {
    D10_METHOD_ID: 2,
    D11_METHOD_ID: 2,
    D12_METHOD_ID: 2,
    D13_METHOD_ID: 4,
    D14_METHOD_ID: 4,
    D15_METHOD_ID: 8,
    D16_METHOD_ID: 4,
    D17_METHOD_ID: 4,
    D18_METHOD_ID: 4,
    D19_METHOD_ID: 1,
    F2_METHOD_ID: 4,
}
_RANDOM_AND_HARD_METHOD_IDS = frozenset(
    {
        D11_METHOD_ID,
        D12_METHOD_ID,
        D13_METHOD_ID,
        D14_METHOD_ID,
        D15_METHOD_ID,
        D18_METHOD_ID,
        D19_METHOD_ID,
        F2_METHOD_ID,
    }
)
_INDEPENDENT_RANDOM_METHOD_IDS = frozenset({D16_METHOD_ID, D17_METHOD_ID})
_HARD_METHOD_IDS = frozenset(
    {
        D10_METHOD_ID,
        D11_METHOD_ID,
        D12_METHOD_ID,
        D13_METHOD_ID,
        D14_METHOD_ID,
        D15_METHOD_ID,
        D18_METHOD_ID,
        D19_METHOD_ID,
        F2_METHOD_ID,
    }
)
_RANDOM_NODE_LAMBDAS: dict[str, dict[str, float]] = {
    D16_METHOD_ID: {
        "compat_random_1": 0.05,
        "compat_random_2": 0.05,
    },
    D17_METHOD_ID: {
        "compat_random_1": 0.025,
        "compat_random_2": 0.05,
        "compat_random_3": 0.10,
    },
}

F2_REAL_AXIS_MASS = 0.5
F2_NUISANCE_AXIS_MASS = 0.5
F2_NUISANCE_SOURCE_NODES = (
    "corruption_1",
    "corruption_2",
    "corruption_3",
    "corruption_4",
)
F2_NUISANCE_WIDTH = len(F2_NUISANCE_SOURCE_NODES)
F2_HULL_LAMBDA = 0.20
F2_RESIDUAL_CORRECTION = "cached_anchor_decode_delta_bypass"

NORM_ONLY = 0
ABNORMAL = 1
OTHER = 2
GROUP_NAMES = ("NORM_ONLY", "ABNORMAL", "OTHER")
ABNORMAL_CLASS_INDICES = (0, 1, 2, 4)
COARSE_CANDIDATE_GROUP_POLICY = "norm_abnormal_other"
EXACT_CANDIDATE_GROUP_POLICY = "exact_positive_set"

CANDIDATE_COUNT = 20
NONSELF_CANDIDATE_COUNT = CANDIDATE_COUNT - 1
LOCAL_POOL_SIZE = 120
HULL_LAMBDA = 0.05
PGD_EPSILON = 2.0
HARD_STEPS = 3
HARD_LEARNING_RATE = 0.25
HARD_INIT_LOGIT_GAP = 4.0
RANDOM_DIRICHLET_ALPHA = 1.0
ANCHOR_POSITIVE = 0.95
LABEL_MIX_LAMBDA = 0.25
NEW_CLASS_CAP = 0.25
CANDIDATE_RNG_NAMESPACE = "compat_local_candidates_matched_d10_d11_v1"
NODE_CANDIDATE_RNG_NAMESPACE = (
    "compat_local_candidates_node_role_matched_d16_d17_v1"
)
D19_PAIRED_RANDOM_RNG_NAMESPACE = "d19_paired_random_dirichlet_order_independent_v1"
PROMOTION_SIDECAR_ENV = "ECG_DIVERSE_AUGMAX_PROMOTION_SIDECAR"
FULL_K500_RECORD_COUNT = 500
TUNING_RECORD_COUNT = 400
PROMOTED_SELECTED_EPOCH = 37
PROMOTED_SCHEDULER_HORIZON = 40
FULL_K500_PROMOTION_POLICY = (
    "d13_selected_e37_t40_matched_fixed20_all_four_no_worse_"
    "score_strict_v1"
)
A7_ECGFOUNDER_PROMOTION_POLICY = (
    "ecgfounder_a7_g2_internal_k100_selected_one_seed_sanity_v1"
)
LOCKED_CENTERS = (
    "ningbo",
    "chapman_shaoxing",
    "cpsc_2018",
    "georgia",
)
LOCKED_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
LOCKED_MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
LOCKED_MAPPING_HASH = "555ec85d5b51"

D14_STRICT_PROMOTION_GATE = {
    "policy": "all_four_metrics_at_or_above_locked_fixed20",
    "reference": "efficientnet1dv2_direct_fixed20_e23",
    "minimum_metrics": {
        "clean_auroc": 0.8922731926,
        "clean_auprc": 0.7287076884,
        "robust_auroc": 0.8304661431,
        "robust_auprc": 0.6264591620,
    },
    "eligible_epoch_score": {
        "clean_auprc_weight": 0.5,
        "robust_auprc_weight": 0.5,
    },
    "tie_break": "earliest_epoch_on_exact_tie",
    "no_eligible_epoch_policy": "no_promotion",
}


_EXPECTED_CONTRACT = {
    "version": 1,
    "mode": None,
    "full_k500_supported": False,
    "candidate_group_policy": COARSE_CANDIDATE_GROUP_POLICY,
    "candidate_count": CANDIDATE_COUNT,
    "candidate_includes_anchor": True,
    "local_pool_size": LOCAL_POOL_SIZE,
    "hull_lambda": HULL_LAMBDA,
    "pgd_epsilon": PGD_EPSILON,
    "hard_steps": HARD_STEPS,
    "hard_learning_rate": HARD_LEARNING_RATE,
    "hard_init_logit_gap": HARD_INIT_LOGIT_GAP,
    "random_dirichlet_alpha": RANDOM_DIRICHLET_ALPHA,
    "soft_target_mode": "anchor_soft",
    "anchor_positive": ANCHOR_POSITIVE,
    "label_mix_lambda": LABEL_MIX_LAMBDA,
    "new_class_cap": NEW_CLASS_CAP,
    "norm_suppression": True,
    "candidate_rng_comparison_shared": True,
}


def _expected_contract_for_method(method_id: str) -> dict[str, Any]:
    payload = dict(_EXPECTED_CONTRACT)
    payload["mode"] = _METHOD_MODES[method_id]
    if method_id == F2_METHOD_ID:
        payload.update(
            {
                "candidate_group_policy": EXACT_CANDIDATE_GROUP_POLICY,
                "hull_lambda": F2_HULL_LAMBDA,
                "axis_policy": "fixed_equal_real_compatible_and_raw_nuisance",
                "real_axis_mass": F2_REAL_AXIS_MASS,
                "nuisance_axis_mass": F2_NUISANCE_AXIS_MASS,
                "nuisance_source_nodes": list(F2_NUISANCE_SOURCE_NODES),
                "nuisance_encode_policy": (
                    "single_batched_posterior_mean_per_generate"
                ),
                "nuisance_posterior_sample": False,
                "residual_correction": F2_RESIDUAL_CORRECTION,
                "convexity_policy": "nonnegative_unit_sum_no_extrapolation",
                "soft_target_axis_policy": (
                    "exact_label_anchor_soft_no_class_admission"
                ),
            }
        )
    if method_id == D19_METHOD_ID:
        payload.update(
            {
                "hard_start_policy": "anchor_biased_plus_paired_random",
                "hard_start_count": 2,
                "hard_selection_include_unoptimized_random": True,
                "hard_selection_guarantee": "final_bce_gte_paired_random",
                "paired_random_rng_policy": (
                    "shared_candidate_seed_order_independent"
                ),
            }
        )
    if method_id in _INDEPENDENT_RANDOM_METHOD_IDS:
        payload.pop("hull_lambda")
        payload["candidate_rng_comparison_shared"] = False
        payload["candidate_seed_policy"] = "node_role_without_method_id"
        payload["dirichlet_seed_policy"] = "executor_method_and_node_specific"
        payload["role_hull_lambdas"] = dict(_RANDOM_NODE_LAMBDAS[method_id])
    return payload


def _random_roles(method_id: str) -> tuple[str, ...]:
    if method_id in _RANDOM_AND_HARD_METHOD_IDS:
        return ("compat_random",)
    return tuple(_RANDOM_NODE_LAMBDAS.get(method_id, ()))


def _expected_roles(method_id: str) -> set[str]:
    roles = set(_random_roles(method_id))
    if method_id in _HARD_METHOD_IDS:
        roles.add("compat_hard")
    return roles

_EXPECTED_RAW_OBJECTIVES = {
    D10_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.6),
        (
            "clean_compat_hard_jsd",
            "bernoulli_jsd",
            ("clean_view", "compat_hard_view"),
            0.3,
        ),
    ),
    D11_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("compat_random_bce", "bce", ("compat_random_view",), 0.3),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.3),
        (
            "clean_compat_random_hard_jsd",
            "bernoulli_jsd",
            ("clean_view", "compat_random_view", "compat_hard_view"),
            0.3,
        ),
    ),
    D12_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.4),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.4),
        ("compat_random_bce", "bce", ("compat_random_view",), 0.1),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.1),
        (
            "clean_raw2_compat_random_hard_jsd",
            "bernoulli_jsd",
            (
                "clean_view",
                "corruption_view_1",
                "corruption_view_2",
                "compat_random_view",
                "compat_hard_view",
            ),
            0.6,
        ),
    ),
    D13_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
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
    ),
    D14_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("clean_retention_bce", "bce", ("clean_view",), 0.1),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
        ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
        ("compat_random_bce", "bce", ("compat_random_view",), 0.05),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.05),
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
    ),
    D15_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.1),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.1),
        ("corruption_3_bce", "bce", ("corruption_view_3",), 0.1),
        ("corruption_4_bce", "bce", ("corruption_view_4",), 0.1),
        ("corruption_5_bce", "bce", ("corruption_view_5",), 0.1),
        ("corruption_6_bce", "bce", ("corruption_view_6",), 0.1),
        ("corruption_7_bce", "bce", ("corruption_view_7",), 0.1),
        ("corruption_8_bce", "bce", ("corruption_view_8",), 0.1),
        ("compat_random_bce", "bce", ("compat_random_view",), 0.1),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.1),
        (
            "clean_raw8_compat_random_hard_jsd",
            "bernoulli_jsd",
            (
                "clean_view",
                "corruption_view_1",
                "corruption_view_2",
                "corruption_view_3",
                "corruption_view_4",
                "corruption_view_5",
                "corruption_view_6",
                "corruption_view_7",
                "corruption_view_8",
                "compat_random_view",
                "compat_hard_view",
            ),
            0.6,
        ),
    ),
    D16_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
        ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
        ("compat_random_1_bce", "bce", ("compat_random_1_view",), 0.1),
        ("compat_random_2_bce", "bce", ("compat_random_2_view",), 0.1),
        (
            "clean_raw4_compat_random2_jsd",
            "bernoulli_jsd",
            (
                "clean_view",
                "corruption_view_1",
                "corruption_view_2",
                "corruption_view_3",
                "corruption_view_4",
                "compat_random_1_view",
                "compat_random_2_view",
            ),
            0.6,
        ),
    ),
    D17_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
        ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
        (
            "compat_random_1_bce",
            "bce",
            ("compat_random_1_view",),
            1.0 / 15.0,
        ),
        (
            "compat_random_2_bce",
            "bce",
            ("compat_random_2_view",),
            1.0 / 15.0,
        ),
        (
            "compat_random_3_bce",
            "bce",
            ("compat_random_3_view",),
            1.0 / 15.0,
        ),
        (
            "clean_raw4_compat_random3_multishell_jsd",
            "bernoulli_jsd",
            (
                "clean_view",
                "corruption_view_1",
                "corruption_view_2",
                "corruption_view_3",
                "corruption_view_4",
                "compat_random_1_view",
                "compat_random_2_view",
                "compat_random_3_view",
            ),
            0.6,
        ),
    ),
    D18_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
        ("corruption_1_bce", "bce", ("corruption_view_1",), 0.2),
        ("corruption_2_bce", "bce", ("corruption_view_2",), 0.2),
        ("corruption_3_bce", "bce", ("corruption_view_3",), 0.2),
        ("corruption_4_bce", "bce", ("corruption_view_4",), 0.2),
        ("compat_random_bce", "bce", ("compat_random_view",), 0.0),
        ("compat_hard_bce", "bce", ("compat_hard_view",), 0.2),
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
    ),
    D19_METHOD_ID: (
        ("clean_bce", "bce", ("clean_view",), 1.0),
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
    ),
}

# F2 deliberately changes only the construction of D13's two latent views.
# The outer loss, seven-view JSD, optimizer step count and effective weights
# remain byte-for-byte equivalent at the compiled-contract level.
_EXPECTED_RAW_OBJECTIVES[F2_METHOD_ID] = _EXPECTED_RAW_OBJECTIVES[D13_METHOD_ID]

_EXPECTED_BN_WEIGHTS = {
    D10_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "compat_hard_view": 0.3,
    },
    D11_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "compat_random_view": 0.15,
        "compat_hard_view": 0.15,
    },
    D12_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.2,
        "corruption_view_2": 0.2,
        "compat_random_view": 0.05,
        "compat_hard_view": 0.05,
    },
    D13_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "compat_random_view": 0.05,
        "compat_hard_view": 0.05,
    },
    D14_METHOD_ID: {
        "clean_view": 0.55,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "compat_random_view": 0.025,
        "compat_hard_view": 0.025,
    },
    D15_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.05,
        "corruption_view_2": 0.05,
        "corruption_view_3": 0.05,
        "corruption_view_4": 0.05,
        "corruption_view_5": 0.05,
        "corruption_view_6": 0.05,
        "corruption_view_7": 0.05,
        "corruption_view_8": 0.05,
        "compat_random_view": 0.05,
        "compat_hard_view": 0.05,
    },
    D16_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "compat_random_1_view": 0.05,
        "compat_random_2_view": 0.05,
    },
    D17_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "compat_random_1_view": 1.0 / 30.0,
        "compat_random_2_view": 1.0 / 30.0,
        "compat_random_3_view": 1.0 / 30.0,
    },
    D18_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.1,
        "corruption_view_2": 0.1,
        "corruption_view_3": 0.1,
        "corruption_view_4": 0.1,
        "compat_random_view": 0.05,
        "compat_hard_view": 0.05,
    },
    D19_METHOD_ID: {
        "clean_view": 0.5,
        "corruption_view_1": 0.2,
        "compat_random_view": 0.15,
        "compat_hard_view": 0.15,
    },
}

_EXPECTED_BN_WEIGHTS[F2_METHOD_ID] = dict(_EXPECTED_BN_WEIGHTS[D13_METHOD_ID])

_D18_EFFECTIVE_OBJECTIVE_WEIGHTS = {
    "clean_view": 0.5,
    "corruption_view_1": 0.1,
    "corruption_view_2": 0.1,
    "corruption_view_3": 0.1,
    "corruption_view_4": 0.1,
    "compat_random_view": 0.0,
    "compat_hard_view": 0.1,
    "clean_raw4_compat_random_hard_jsd": 0.3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} keys must be exactly {sorted(expected)}")


def _require_sha256(value: Any, description: str) -> str:
    resolved = str(value).strip().lower()
    if len(resolved) != 64 or any(
        character not in "0123456789abcdef" for character in resolved
    ):
        raise ValueError(f"{description} must be a 64-character SHA256")
    return resolved


def _f2_expected_vae_sha256(
    method: CompiledMethod,
    config_root: Path,
) -> str:
    resource = method.resources.get("vae")
    if not isinstance(resource, Mapping):
        raise ValueError("F2 method must declare resources.vae")
    raw_path = resource.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("F2 resources.vae.path must be non-empty")
    config_path = Path(raw_path).expanduser()
    if not config_path.is_absolute():
        config_path = config_root / config_path
    config_path = config_path.resolve()
    try:
        config_path.relative_to(config_root.resolve())
    except ValueError as error:
        raise ValueError("F2 VAE config must stay inside the config root") from error
    if not config_path.is_file() or config_path.is_symlink():
        raise FileNotFoundError(f"F2 VAE config is unavailable: {config_path}")
    payload = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "F2 VAE config",
    )
    model = _mapping(payload.get("model"), "F2 VAE config.model")
    return _require_sha256(
        model.get("checkpoint_sha256"),
        "F2 VAE config.model.checkpoint_sha256",
    )


def _regular_file(path: Any, description: str) -> Path:
    resolved = Path(str(path)).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{description} must be an existing non-symlink file")
    return resolved


def _file_reference(value: Any, description: str) -> tuple[Path, str]:
    reference = _mapping(value, description)
    _exact_keys(reference, {"path", "sha256"}, description)
    path = _regular_file(reference["path"], f"{description}.path")
    expected = _require_sha256(reference["sha256"], f"{description}.sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return path, actual


def _json_artifact(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is invalid JSON: {exc}") from exc
    return _mapping(value, description)


def _finite_metric(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{description} must be a finite number")
    return resolved


def _selection_common(
    value: Mapping[str, Any],
    *,
    artifact_type: str,
    method_id: str,
    selected_epoch: int | None,
    model_family: str = "efficientnet1dv2",
    scheduler_horizon_epochs: int = PROMOTED_SCHEDULER_HORIZON,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if value.get("schema_version") != 1 or value.get("status") != "selected":
        raise ValueError("promotion selection must be selected schema_version=1")
    if value.get("artifact_type") != artifact_type:
        raise ValueError("promotion selection artifact_type mismatch")
    if tuple(value.get("centers", ())) != LOCKED_CENTERS:
        raise ValueError("promotion selection centers mismatch")
    if tuple(value.get("class_order", ())) != LOCKED_CLASS_ORDER:
        raise ValueError("promotion selection class order mismatch")
    if value.get("record_count") != 400 or value.get("composition_count") != 20:
        raise ValueError("promotion selection must contain 400 records and 20 views")
    comparison = _mapping(value.get("comparison_identity"), "comparison_identity")
    if value.get("comparison_identity_sha256") != _canonical_sha256(comparison):
        raise ValueError("promotion selection comparison identity SHA256 mismatch")
    if comparison.get("method_id") != method_id:
        raise ValueError("promotion selection method ID mismatch")
    if comparison.get("model_family") != model_family:
        raise ValueError(
            f"promotion selection must use model family {model_family}"
        )
    parameters = _mapping(
        comparison.get("resolved_training_parameters"),
        "resolved_training_parameters",
    )
    if parameters.get("epochs") != scheduler_horizon_epochs:
        raise ValueError(
            "promotion tuning grid does not match the selected scheduler horizon"
        )
    if parameters.get("scheduler_horizon_epochs") != scheduler_horizon_epochs:
        raise ValueError("promotion scheduler horizon mismatch")
    epoch_grid = value.get("epoch_grid")
    if epoch_grid != list(range(1, scheduler_horizon_epochs + 1)):
        raise ValueError("promotion selection epoch grid must be complete")
    actual_selected = value.get("selected_epoch")
    if (
        isinstance(actual_selected, bool)
        or not isinstance(actual_selected, int)
        or actual_selected <= 0
        or actual_selected > scheduler_horizon_epochs
    ):
        raise ValueError("promotion selection selected_epoch is invalid")
    if selected_epoch is not None and actual_selected != selected_epoch:
        raise ValueError(f"D13 promotion requires selected epoch E{selected_epoch}")
    rule = _mapping(value.get("selection_rule"), "selection_rule")
    if rule.get("heldout_evaluation_used") is not False:
        raise ValueError("held-out target feedback cannot authorize promotion")
    config = _mapping(value.get("config"), "selection config")
    if (
        config.get("mapping_version") != LOCKED_MAPPING_VERSION
        or config.get("mapping_hash") != LOCKED_MAPPING_HASH
    ):
        raise ValueError("promotion selection mapping identity mismatch")
    return comparison, parameters


def _selected_metrics(value: Mapping[str, Any], description: str) -> dict[str, float]:
    return {
        "clean_auroc": _finite_metric(
            value.get("selected_clean_macro_auroc"),
            f"{description}.selected_clean_macro_auroc",
        ),
        "clean_auprc": _finite_metric(
            value.get("selected_clean_macro_auprc"),
            f"{description}.selected_clean_macro_auprc",
        ),
        "robust_auroc": _finite_metric(
            value.get("selected_robust_macro_auroc"),
            f"{description}.selected_robust_macro_auroc",
        ),
        "robust_auprc": _finite_metric(
            value.get("selected_robust_macro_auprc"),
            f"{description}.selected_robust_macro_auprc",
        ),
        "score": _finite_metric(
            value.get("selected_score"), f"{description}.selected_score"
        ),
    }


@dataclass(frozen=True)
class FullK500Promotion:
    sidecar_path: Path
    sidecar_sha256: str
    selection_path: Path
    selection_sha256: str
    control_selection_path: Path
    control_selection_sha256: str
    method_profile_path: Path
    method_profile_sha256: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    vae_checkpoint_path: Path
    vae_checkpoint_sha256: str
    selected_epoch: int
    scheduler_horizon_epochs: int
    method_metrics: Mapping[str, float]
    control_metrics: Mapping[str, float]
    metric_deltas: Mapping[str, float]
    policy: str = FULL_K500_PROMOTION_POLICY
    method_id: str = D13_METHOD_ID
    model_family: str = "efficientnet1dv2"
    invocation_method_profile_path: Path | None = None
    invocation_method_profile_sha256: str | None = None

    def describe(self) -> dict[str, Any]:
        description = {
            "authorization": "external_verified_promotion_sidecar",
            "policy": self.policy,
            "method_id": self.method_id,
            "model_family": self.model_family,
            "sidecar": {
                "path": str(self.sidecar_path),
                "sha256": self.sidecar_sha256,
            },
            "selection": {
                "path": str(self.selection_path),
                "sha256": self.selection_sha256,
                "selected_epoch": self.selected_epoch,
                "scheduler_horizon_epochs": self.scheduler_horizon_epochs,
            },
            "matched_control_selection": {
                "path": str(self.control_selection_path),
                "sha256": self.control_selection_sha256,
            },
            "method_profile": {
                "path": str(self.method_profile_path),
                "sha256": self.method_profile_sha256,
            },
            "source_checkpoint": {
                "path": str(self.source_checkpoint_path),
                "sha256": self.source_checkpoint_sha256,
            },
            "vae_checkpoint": {
                "path": str(self.vae_checkpoint_path),
                "sha256": self.vae_checkpoint_sha256,
            },
            "gate": {
                "passed": True,
                "method_metrics": dict(self.method_metrics),
                "control_metrics": dict(self.control_metrics),
                "method_minus_control": dict(self.metric_deltas),
            },
        }
        if self.invocation_method_profile_path is not None:
            description["invocation_method_profile"] = {
                "path": str(self.invocation_method_profile_path),
                "sha256": self.invocation_method_profile_sha256,
            }
        return description


_ACTIVE_FULL_K500_PROMOTION: ContextVar[FullK500Promotion | None] = ContextVar(
    "diverse_augmax_full_k500_promotion",
    default=None,
)
_ACTIVE_FULL_K500_ORACLE: ContextVar[dict[str, Any] | None] = ContextVar(
    "diverse_augmax_full_k500_heldout_oracle",
    default=None,
)


def active_full_k500_promotion() -> FullK500Promotion | None:
    return _ACTIVE_FULL_K500_PROMOTION.get()


def active_full_k500_heldout_oracle() -> dict[str, Any] | None:
    return _ACTIVE_FULL_K500_ORACLE.get()


@contextmanager
def full_k500_heldout_oracle_scope(
    outer_method: CompiledMethod,
) -> Iterator[dict[str, Any]]:
    """Authorize one explicit diagnostic-only full-K500 search construction."""

    if outer_method.profile_name != FULL_K500_ORACLE_OUTER_METHOD_ID:
        raise ValueError("full-K500 heldout oracle is restricted to the search method")
    raw = outer_method.contracts.get(FULL_K500_ORACLE_CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"contracts.{FULL_K500_ORACLE_CONTRACT_KEY} must be a mapping"
        )
    expected = {
        "authorized": True,
        "training_partition": "k500",
        "selection_protocol": "refexcluded_clean_and_pn2021c_heldout_oracle",
        "evidence_status": "heldout_tuned_diagnostic_only",
        "candidate_method_id": D19_METHOD_ID,
    }
    if set(raw) != set(expected) or any(raw[key] != value for key, value in expected.items()):
        raise ValueError("full-K500 heldout oracle contract is incomplete or drifted")
    if active_full_k500_promotion() is not None:
        raise RuntimeError("heldout oracle and selected-refit promotion are exclusive")
    if active_full_k500_heldout_oracle() is not None:
        raise RuntimeError("full-K500 heldout oracle scope is already active")
    identity = {
        **expected,
        "outer_method_profile_sha256": outer_method.profile_sha256,
    }
    token = _ACTIVE_FULL_K500_ORACLE.set(identity)
    try:
        yield identity
    finally:
        _ACTIVE_FULL_K500_ORACLE.reset(token)


def _verify_d13_full_k500_promotion(
    sidecar_path: str | Path,
    *,
    invocation: Mapping[str, Any],
) -> FullK500Promotion:
    """Verify one external D13 full-K500 authorization without loading data/GPU."""

    sidecar = _regular_file(sidecar_path, "promotion sidecar")
    payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    root = _mapping(payload, "promotion sidecar")
    _exact_keys(root, {"schema_version", "promotion"}, "promotion sidecar")
    if root.get("schema_version") != 1:
        raise ValueError("promotion sidecar schema_version must be 1")
    promotion = _mapping(root.get("promotion"), "promotion")
    _exact_keys(
        promotion,
        {
            "authorized",
            "policy",
            "method_id",
            "model_family",
            "centers",
            "class_order",
            "mapping_version",
            "mapping_hash",
            "selected_epoch",
            "scheduler_horizon_epochs",
            "heldout_target_feedback_allowed",
            "exclude_all_k500_from_final_evaluation",
            "geometry",
            "selection",
            "method_profile",
            "source_checkpoint",
            "vae_checkpoint",
            "matched_control",
        },
        "promotion",
    )
    if promotion.get("authorized") is not True:
        raise ValueError("promotion sidecar must explicitly authorize the refit")
    if promotion.get("policy") != FULL_K500_PROMOTION_POLICY:
        raise ValueError("promotion sidecar policy mismatch")
    if promotion.get("method_id") != D13_METHOD_ID:
        raise ValueError("only D13 may receive this promotion")
    if promotion.get("model_family") != "efficientnet1dv2":
        raise ValueError("D13 E37/T40 promotion is EfficientNet-only")
    if tuple(promotion.get("centers", ())) != LOCKED_CENTERS:
        raise ValueError("promotion sidecar centers mismatch")
    if tuple(promotion.get("class_order", ())) != LOCKED_CLASS_ORDER:
        raise ValueError("promotion sidecar class order mismatch")
    if (
        promotion.get("mapping_version") != LOCKED_MAPPING_VERSION
        or promotion.get("mapping_hash") != LOCKED_MAPPING_HASH
    ):
        raise ValueError("promotion sidecar mapping identity mismatch")
    if promotion.get("selected_epoch") != PROMOTED_SELECTED_EPOCH:
        raise ValueError("promotion sidecar must lock E37")
    if promotion.get("scheduler_horizon_epochs") != PROMOTED_SCHEDULER_HORIZON:
        raise ValueError("promotion sidecar must lock T40")
    if promotion.get("heldout_target_feedback_allowed") is not False:
        raise ValueError("promotion sidecar cannot allow held-out target feedback")
    if promotion.get("exclude_all_k500_from_final_evaluation") is not True:
        raise ValueError("promotion sidecar must require full K500 ref exclusion")
    geometry = _mapping(promotion.get("geometry"), "promotion.geometry")
    _exact_keys(
        geometry,
        {
            "partition",
            "record_count",
            "standardizer_source_population",
            "standardizer_fit_policy",
            "validation100_phase",
        },
        "promotion.geometry",
    )
    if geometry != {
        "partition": "k500",
        "record_count": FULL_K500_RECORD_COUNT,
        "standardizer_source_population": "full_k500",
        "standardizer_fit_policy": "whitelist_exact_label_m20_eligible",
        "validation100_phase": "post_selection_refit_only",
    }:
        raise ValueError("promotion full-K500 geometry contract mismatch")

    required_invocation = {
        "entrypoint",
        "model_family",
        "method_config_path",
        "selection_json_path",
        "source_checkpoint_path",
        "epochs",
        "scheduler_horizon_epochs",
        "trainable_scope",
    }
    _exact_keys(invocation, required_invocation, "promotion invocation")
    if invocation.get("entrypoint") != "train_pn2021":
        raise ValueError("full-K500 promotion is valid only for train_pn2021")
    if invocation.get("model_family") != "efficientnet1dv2":
        raise ValueError("promotion invocation model mismatch")
    if invocation.get("epochs") != PROMOTED_SELECTED_EPOCH:
        raise ValueError("promotion invocation must explicitly request E37")
    if invocation.get("scheduler_horizon_epochs") != PROMOTED_SCHEDULER_HORIZON:
        raise ValueError("promotion invocation must explicitly request T40")
    if invocation.get("trainable_scope") != "full":
        raise ValueError("promotion invocation must use full fine-tuning")

    selection_path, selection_sha256 = _file_reference(
        promotion["selection"], "promotion.selection"
    )
    method_path, method_sha256 = _file_reference(
        promotion["method_profile"], "promotion.method_profile"
    )
    source_path, source_sha256 = _file_reference(
        promotion["source_checkpoint"], "promotion.source_checkpoint"
    )
    vae_path, vae_sha256 = _file_reference(
        promotion["vae_checkpoint"], "promotion.vae_checkpoint"
    )
    if _regular_file(
        invocation["selection_json_path"], "invocation selection"
    ) != selection_path:
        raise ValueError("invocation selection differs from promotion sidecar")
    if _regular_file(
        invocation["method_config_path"], "invocation method profile"
    ) != method_path:
        raise ValueError("invocation method profile differs from promotion sidecar")
    if _regular_file(
        invocation["source_checkpoint_path"], "invocation source checkpoint"
    ) != source_path:
        raise ValueError("invocation source checkpoint differs from promotion sidecar")

    selection = _json_artifact(selection_path, "D13 selection")
    comparison, parameters = _selection_common(
        selection,
        artifact_type="pn2021_k500_pooled_epoch_selection",
        method_id=D13_METHOD_ID,
        selected_epoch=PROMOTED_SELECTED_EPOCH,
    )
    if comparison.get("method_profile_sha256") != method_sha256:
        raise ValueError("D13 selection method profile SHA256 mismatch")
    source_identity = _mapping(
        comparison.get("source_checkpoint_identity"),
        "D13 source checkpoint identity",
    )
    if (
        Path(str(source_identity.get("path", ""))).expanduser().resolve() != source_path
        or source_identity.get("sha256") != source_sha256
    ):
        raise ValueError("D13 selection source checkpoint identity mismatch")
    vae_encoder = _mapping(
        comparison.get("vae_encoder_checkpoint"), "D13 VAE encoder identity"
    )
    vae_decoder = _mapping(
        comparison.get("vae_decoder_checkpoint"), "D13 VAE decoder identity"
    )
    if (
        vae_encoder.get("sha256") != vae_sha256
        or vae_decoder.get("sha256") != vae_sha256
        or Path(str(vae_decoder.get("path", ""))).expanduser().resolve() != vae_path
    ):
        raise ValueError("D13 selection VAE checkpoint identity mismatch")

    matched = _mapping(promotion.get("matched_control"), "matched_control")
    _exact_keys(
        matched,
        {"selection", "method_id", "scheduler_horizon_epochs", "gate_policy"},
        "matched_control",
    )
    if matched.get("method_id") != "direct_depth23_fixed20":
        raise ValueError("matched control must be Direct+fixed20")
    if matched.get("scheduler_horizon_epochs") != PROMOTED_SCHEDULER_HORIZON:
        raise ValueError("matched control must use T40")
    if matched.get("gate_policy") != FULL_K500_PROMOTION_POLICY:
        raise ValueError("matched control gate policy mismatch")
    control_path, control_sha256 = _file_reference(
        matched["selection"], "matched_control.selection"
    )
    control = _json_artifact(control_path, "fixed20-T40 selection")
    control_comparison, control_parameters = _selection_common(
        control,
        artifact_type="direct_k500_pooled_epoch_selection",
        method_id="direct_depth23_fixed20",
        selected_epoch=None,
    )
    if comparison.get("model_spec") != control_comparison.get("model_spec"):
        raise ValueError("D13 and fixed20 model specifications differ")
    if comparison.get("replicate_id") != control_comparison.get("replicate_id"):
        raise ValueError("D13 and fixed20 replicate IDs differ")
    if comparison.get("pos_weight") != control_comparison.get("pos_weight"):
        raise ValueError("D13 and fixed20 pos_weight differ")
    if comparison.get("source_checkpoint_identity") != control_comparison.get(
        "source_checkpoint_identity"
    ):
        raise ValueError("D13 and fixed20 source checkpoints differ")
    matched_training_keys = {
        "amp_dtype",
        "amp_enabled",
        "batch_size",
        "epochs",
        "gradient_clip_norm",
        "learning_rate",
        "minimum_learning_rate_ratio",
        "scheduler_horizon_epochs",
        "weight_decay",
    }
    if set(parameters) != matched_training_keys or set(control_parameters) != matched_training_keys:
        raise ValueError("D13/fixed20 training parameter keys are not the locked set")
    for key in matched_training_keys:
        left = parameters[key]
        right = control_parameters[key]
        if isinstance(left, float) or isinstance(right, float):
            matches = _same_float(left, float(right))
        else:
            matches = left == right
        if not matches:
            raise ValueError(f"D13 and fixed20 training parameter {key} differs")
    for key in (
        "composition_ids",
        "frozen_corruption_identity_sha256",
        "per_center_validation_identity",
        "selection_rule",
    ):
        if selection.get(key) != control.get(key):
            raise ValueError(f"D13 and fixed20 {key} differ")

    method_metrics = _selected_metrics(selection, "D13 selection")
    control_metrics = _selected_metrics(control, "fixed20-T40 selection")
    metric_deltas = {
        name: method_metrics[name] - control_metrics[name]
        for name in method_metrics
    }
    for name in ("clean_auroc", "clean_auprc", "robust_auroc", "robust_auprc"):
        if metric_deltas[name] < 0.0:
            raise ValueError(f"D13 fails matched fixed20-T40 gate on {name}")
    if metric_deltas["score"] <= 0.0:
        raise ValueError("D13 must strictly exceed the matched fixed20-T40 score")

    return FullK500Promotion(
        sidecar_path=sidecar,
        sidecar_sha256=_sha256(sidecar),
        selection_path=selection_path,
        selection_sha256=selection_sha256,
        control_selection_path=control_path,
        control_selection_sha256=control_sha256,
        method_profile_path=method_path,
        method_profile_sha256=method_sha256,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha256,
        vae_checkpoint_path=vae_path,
        vae_checkpoint_sha256=vae_sha256,
        selected_epoch=PROMOTED_SELECTED_EPOCH,
        scheduler_horizon_epochs=PROMOTED_SCHEDULER_HORIZON,
        method_metrics=method_metrics,
        control_metrics=control_metrics,
        metric_deltas=metric_deltas,
    )


def _verify_a7_ecgfounder_full_k500_promotion(
    sidecar_path: str | Path,
    *,
    invocation: Mapping[str, Any],
) -> FullK500Promotion:
    """Verify the K100-selected ECGFounder G2 one-seed full-K500 sanity refit."""

    sidecar = _regular_file(sidecar_path, "A7 promotion sidecar")
    payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    root = _mapping(payload, "A7 promotion sidecar")
    _exact_keys(root, {"schema_version", "promotion"}, "A7 promotion sidecar")
    if root.get("schema_version") != 1:
        raise ValueError("A7 promotion sidecar schema_version must be 1")
    promotion = _mapping(root.get("promotion"), "A7 promotion")
    _exact_keys(
        promotion,
        {
            "authorized",
            "policy",
            "method_id",
            "outer_method_id",
            "model_family",
            "centers",
            "class_order",
            "mapping_version",
            "mapping_hash",
            "selected_epoch",
            "scheduler_horizon_epochs",
            "heldout_target_feedback_allowed",
            "exclude_all_k500_from_final_evaluation",
            "evidence_status",
            "geometry",
            "selection",
            "tuning_method_profile",
            "refit_method_profile",
            "runtime_method_profile",
            "source_checkpoint",
            "vae_checkpoint",
            "matched_control",
        },
        "A7 promotion",
    )
    if promotion.get("authorized") is not True:
        raise ValueError("A7 promotion must explicitly authorize the refit")
    if promotion.get("policy") != A7_ECGFOUNDER_PROMOTION_POLICY:
        raise ValueError("A7 promotion policy mismatch")
    if promotion.get("method_id") != D19_METHOD_ID:
        raise ValueError("A7 promotion runtime method must be D19")
    if promotion.get("outer_method_id") != FULL_K500_ORACLE_OUTER_METHOD_ID:
        raise ValueError("A7 promotion outer method identity mismatch")
    if promotion.get("model_family") != "ecgfounder":
        raise ValueError("A7 promotion is ECGFounder-only")
    if tuple(promotion.get("centers", ())) != LOCKED_CENTERS:
        raise ValueError("A7 promotion centers mismatch")
    if tuple(promotion.get("class_order", ())) != LOCKED_CLASS_ORDER:
        raise ValueError("A7 promotion class order mismatch")
    if (
        promotion.get("mapping_version") != LOCKED_MAPPING_VERSION
        or promotion.get("mapping_hash") != LOCKED_MAPPING_HASH
    ):
        raise ValueError("A7 promotion mapping identity mismatch")
    selected_epoch = promotion.get("selected_epoch")
    scheduler_horizon = promotion.get("scheduler_horizon_epochs")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch <= 0
        or selected_epoch > PROMOTED_SCHEDULER_HORIZON
    ):
        raise ValueError("A7 promotion selected_epoch is invalid")
    if scheduler_horizon != PROMOTED_SCHEDULER_HORIZON:
        raise ValueError("A7 promotion must retain the frozen T40 horizon")
    if promotion.get("heldout_target_feedback_allowed") is not False:
        raise ValueError("A7 promotion cannot use held-out target feedback")
    if promotion.get("exclude_all_k500_from_final_evaluation") is not True:
        raise ValueError("A7 promotion must require full K500 ref exclusion")
    if promotion.get("evidence_status") != "internal_selected_one_seed_sanity":
        raise ValueError("A7 promotion evidence status mismatch")

    geometry = _mapping(promotion.get("geometry"), "A7 promotion.geometry")
    _exact_keys(
        geometry,
        {
            "partition",
            "record_count",
            "standardizer_source_population",
            "standardizer_fit_policy",
            "validation100_phase",
        },
        "A7 promotion.geometry",
    )
    if geometry != {
        "partition": "k500",
        "record_count": FULL_K500_RECORD_COUNT,
        "standardizer_source_population": "full_k500",
        "standardizer_fit_policy": "whitelist_exact_label_m20_eligible",
        "validation100_phase": "post_selection_refit_only",
    }:
        raise ValueError("A7 promotion full-K500 geometry contract mismatch")

    required_invocation = {
        "entrypoint",
        "model_family",
        "method_config_path",
        "selection_json_path",
        "source_checkpoint_path",
        "epochs",
        "scheduler_horizon_epochs",
        "trainable_scope",
    }
    _exact_keys(invocation, required_invocation, "A7 promotion invocation")
    if invocation.get("entrypoint") != "train_pn2021":
        raise ValueError("A7 promotion is valid only for train_pn2021")
    if invocation.get("model_family") != "ecgfounder":
        raise ValueError("A7 promotion invocation model mismatch")
    if invocation.get("epochs") != selected_epoch:
        raise ValueError("A7 promotion invocation epoch differs from selection")
    if invocation.get("scheduler_horizon_epochs") != scheduler_horizon:
        raise ValueError("A7 promotion invocation scheduler horizon mismatch")
    if invocation.get("trainable_scope") != "full":
        raise ValueError("A7 promotion invocation must use full fine-tuning")

    selection_path, selection_sha256 = _file_reference(
        promotion["selection"], "A7 promotion.selection"
    )
    tuning_path, tuning_sha256 = _file_reference(
        promotion["tuning_method_profile"],
        "A7 promotion.tuning_method_profile",
    )
    refit_path, refit_sha256 = _file_reference(
        promotion["refit_method_profile"],
        "A7 promotion.refit_method_profile",
    )
    runtime_path, runtime_sha256 = _file_reference(
        promotion["runtime_method_profile"],
        "A7 promotion.runtime_method_profile",
    )
    source_path, source_sha256 = _file_reference(
        promotion["source_checkpoint"], "A7 promotion.source_checkpoint"
    )
    vae_path, vae_sha256 = _file_reference(
        promotion["vae_checkpoint"], "A7 promotion.vae_checkpoint"
    )
    if _regular_file(
        invocation["selection_json_path"], "A7 invocation selection"
    ) != selection_path:
        raise ValueError("A7 invocation selection differs from promotion sidecar")
    if _regular_file(
        invocation["method_config_path"], "A7 invocation method profile"
    ) != refit_path:
        raise ValueError("A7 invocation method differs from promotion sidecar")
    if _regular_file(
        invocation["source_checkpoint_path"], "A7 invocation source checkpoint"
    ) != source_path:
        raise ValueError("A7 invocation source differs from promotion sidecar")

    selection = _json_artifact(selection_path, "A7 G2 selection")
    comparison, parameters = _selection_common(
        selection,
        artifact_type="pn2021_k500_pooled_epoch_selection",
        method_id=FULL_K500_ORACLE_OUTER_METHOD_ID,
        selected_epoch=int(selected_epoch),
        model_family="ecgfounder",
        scheduler_horizon_epochs=int(scheduler_horizon),
    )
    if comparison.get("method_profile_sha256") != tuning_sha256:
        raise ValueError("A7 selection tuning method profile SHA256 mismatch")
    source_identity = _mapping(
        comparison.get("source_checkpoint_identity"),
        "A7 source checkpoint identity",
    )
    if (
        Path(str(source_identity.get("path", ""))).expanduser().resolve()
        != source_path
        or source_identity.get("sha256") != source_sha256
    ):
        raise ValueError("A7 selection source checkpoint identity mismatch")
    vae_encoder = _mapping(
        comparison.get("vae_encoder_checkpoint"), "A7 VAE encoder identity"
    )
    vae_decoder = _mapping(
        comparison.get("vae_decoder_checkpoint"), "A7 VAE decoder identity"
    )
    if (
        vae_encoder.get("sha256") != vae_sha256
        or vae_decoder.get("sha256") != vae_sha256
        or Path(str(vae_decoder.get("path", ""))).expanduser().resolve()
        != vae_path
    ):
        raise ValueError("A7 selection VAE checkpoint identity mismatch")

    tuning_profile = _mapping(
        yaml.safe_load(tuning_path.read_text(encoding="utf-8")),
        "A7 tuning method profile",
    )
    refit_profile = _mapping(
        yaml.safe_load(refit_path.read_text(encoding="utf-8")),
        "A7 refit method profile",
    )
    runtime_profile = _mapping(
        yaml.safe_load(runtime_path.read_text(encoding="utf-8")),
        "A7 runtime method profile",
    )
    if _mapping(tuning_profile.get("method"), "A7 tuning method").get("id") != (
        FULL_K500_ORACLE_OUTER_METHOD_ID
    ):
        raise ValueError("A7 tuning outer method ID mismatch")
    refit_method = _mapping(refit_profile.get("method"), "A7 refit method")
    if (
        refit_method.get("id") != FULL_K500_ORACLE_OUTER_METHOD_ID
        or refit_method.get("scientific_arm") != "g2_bv_full_k500_refit"
    ):
        raise ValueError("A7 refit outer method identity mismatch")
    if _mapping(runtime_profile.get("method"), "A7 runtime method").get("id") != (
        D19_METHOD_ID
    ):
        raise ValueError("A7 runtime candidate method ID mismatch")
    for section in ("nodes", "inputs", "outputs", "objective", "resources"):
        if tuning_profile.get(section) != refit_profile.get(section):
            raise ValueError(
                f"A7 refit changed frozen method section {section!r}"
            )
    tuning_contracts = copy.deepcopy(
        _mapping(tuning_profile.get("contracts"), "A7 tuning contracts")
    )
    refit_contracts = copy.deepcopy(
        _mapping(refit_profile.get("contracts"), "A7 refit contracts")
    )
    if tuning_contracts.pop("tuning_partition_only", None) != "k500_tune_train":
        raise ValueError("A7 tuning profile partition contract mismatch")
    if tuning_contracts.pop("full_k500_refit_supported", None) is not False:
        raise ValueError("A7 tuning profile full-K500 contract mismatch")
    if refit_contracts.pop("full_k500_refit_supported", None) is not True:
        raise ValueError("A7 refit profile lacks full-K500 contract")
    if refit_contracts.pop("tuning_partition_only", None) is not None:
        raise ValueError("A7 refit profile retained tuning-only partition")
    tuning_teacher = _mapping(
        tuning_contracts.get("unlabeled_teacher"), "A7 tuning teacher"
    )
    refit_teacher = _mapping(
        refit_contracts.get("unlabeled_teacher"), "A7 refit teacher"
    )
    for key in ("evidence_status", "pool_partition"):
        tuning_teacher.pop(key, None)
        refit_teacher.pop(key, None)
    tuning_contracts["unlabeled_teacher"] = tuning_teacher
    refit_contracts["unlabeled_teacher"] = refit_teacher
    if tuning_contracts != refit_contracts:
        raise ValueError("A7 refit changed contracts beyond the refit boundary")
    if (
        _mapping(refit_profile.get("contracts"), "A7 refit contracts").get(
            "heldout_target_feedback_allowed"
        )
        is not False
    ):
        raise ValueError("A7 refit method permits held-out feedback")

    matched = _mapping(
        promotion.get("matched_control"), "A7 matched_control"
    )
    _exact_keys(
        matched,
        {"selection", "method_id", "scheduler_horizon_epochs", "gate_policy"},
        "A7 matched_control",
    )
    if matched.get("method_id") != "direct_depth23_fixed20":
        raise ValueError("A7 matched control must be Direct+fixed20")
    if matched.get("scheduler_horizon_epochs") != scheduler_horizon:
        raise ValueError("A7 matched control scheduler horizon mismatch")
    if matched.get("gate_policy") != A7_ECGFOUNDER_PROMOTION_POLICY:
        raise ValueError("A7 matched control gate policy mismatch")
    control_path, control_sha256 = _file_reference(
        matched["selection"], "A7 matched_control.selection"
    )
    control = _json_artifact(control_path, "A7 G0-budget selection")
    control_comparison, control_parameters = _selection_common(
        control,
        artifact_type="direct_k500_pooled_epoch_selection",
        method_id="direct_depth23_fixed20",
        selected_epoch=None,
        model_family="ecgfounder",
        scheduler_horizon_epochs=int(scheduler_horizon),
    )
    for key in (
        "model_spec",
        "replicate_id",
        "pos_weight",
        "source_checkpoint_identity",
    ):
        if comparison.get(key) != control_comparison.get(key):
            raise ValueError(f"A7 and matched control {key} differ")
    if parameters != control_parameters:
        raise ValueError("A7 and matched control training parameters differ")
    for key in (
        "composition_ids",
        "frozen_corruption_identity_sha256",
        "per_center_validation_identity",
        "selection_rule",
    ):
        if selection.get(key) != control.get(key):
            raise ValueError(f"A7 and matched control {key} differ")

    method_metrics = _selected_metrics(selection, "A7 G2 selection")
    control_metrics = _selected_metrics(control, "A7 G0-budget selection")
    metric_deltas = {
        name: method_metrics[name] - control_metrics[name]
        for name in method_metrics
    }
    if metric_deltas["score"] <= 0.0:
        raise ValueError("A7 G2 must improve the internal pooled selection score")
    if metric_deltas["clean_auprc"] < -0.01:
        raise ValueError("A7 G2 violates the preregistered clean AUPRC floor")

    return FullK500Promotion(
        sidecar_path=sidecar,
        sidecar_sha256=_sha256(sidecar),
        selection_path=selection_path,
        selection_sha256=selection_sha256,
        control_selection_path=control_path,
        control_selection_sha256=control_sha256,
        method_profile_path=runtime_path,
        method_profile_sha256=runtime_sha256,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_sha256,
        vae_checkpoint_path=vae_path,
        vae_checkpoint_sha256=vae_sha256,
        selected_epoch=int(selected_epoch),
        scheduler_horizon_epochs=int(scheduler_horizon),
        method_metrics=method_metrics,
        control_metrics=control_metrics,
        metric_deltas=metric_deltas,
        policy=A7_ECGFOUNDER_PROMOTION_POLICY,
        method_id=D19_METHOD_ID,
        model_family="ecgfounder",
        invocation_method_profile_path=refit_path,
        invocation_method_profile_sha256=refit_sha256,
    )


def verify_full_k500_promotion(
    sidecar_path: str | Path,
    *,
    invocation: Mapping[str, Any],
) -> FullK500Promotion:
    """Dispatch to the exact fail-closed selected-refit promotion policy."""

    sidecar = _regular_file(sidecar_path, "promotion sidecar")
    payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    root = _mapping(payload, "promotion sidecar")
    promotion = _mapping(root.get("promotion"), "promotion")
    policy = promotion.get("policy")
    if policy == A7_ECGFOUNDER_PROMOTION_POLICY:
        return _verify_a7_ecgfounder_full_k500_promotion(
            sidecar,
            invocation=invocation,
        )
    return _verify_d13_full_k500_promotion(
        sidecar,
        invocation=invocation,
    )


@contextmanager
def selected_full_k500_refit_scope(
    sidecar_path: str | Path,
    *,
    invocation: Mapping[str, Any],
) -> Iterator[FullK500Promotion]:
    if active_full_k500_promotion() is not None:
        raise RuntimeError("nested full-K500 promotion scopes are forbidden")
    promotion = verify_full_k500_promotion(sidecar_path, invocation=invocation)
    token = _ACTIVE_FULL_K500_PROMOTION.set(promotion)
    try:
        yield promotion
    finally:
        _ACTIVE_FULL_K500_PROMOTION.reset(token)


def _same_float(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)
    )


def _positions(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in torch.nonzero(mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _strict_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("compatible labels must have shape (N,5)")
    if not labels.is_floating_point() or not bool(torch.isfinite(labels).all()):
        raise ValueError("compatible labels must be finite floating point")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("compatible pool labels must be binary multi-hot targets")
    return labels


def classify_compatibility_groups(labels: torch.Tensor) -> torch.Tensor:
    """Return NORM_ONLY/ABNORMAL/OTHER group codes with strict NORM semantics."""

    values = _strict_binary_labels(labels)
    positive = values > 0.5
    abnormal = positive[:, ABNORMAL_CLASS_INDICES].any(dim=1)
    norm = positive[:, 3]
    if bool((abnormal & norm).any()):
        raise ValueError("compatible pool contains simultaneous NORM and abnormal labels")
    groups = torch.full(
        (values.shape[0],), OTHER, device=values.device, dtype=torch.int64
    )
    groups[norm] = NORM_ONLY
    groups[abnormal] = ABNORMAL
    return groups


def classify_exact_label_groups(labels: torch.Tensor) -> torch.Tensor:
    """Return a stable five-bit code for each exact Super5 positive set."""

    values = _strict_binary_labels(labels)
    # Reuse the strict NORM-vs-abnormal invariant before assigning signatures.
    classify_compatibility_groups(values)
    bits = torch.tensor(
        (1, 2, 4, 8, 16), device=values.device, dtype=torch.int64
    )
    return ((values > 0.5).to(dtype=torch.int64) * bits).sum(dim=1)


def _exact_group_name(code: int) -> str:
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 32:
        raise ValueError("exact-label group code must be an integer in [0,31]")
    positives = [
        name for index, name in enumerate(LOCKED_CLASS_ORDER) if code & (1 << index)
    ]
    return "+".join(positives) if positives else "EMPTY"


@dataclass(frozen=True)
class CompatibleNeighborTable:
    indices: torch.Tensor
    counts: torch.Tensor
    groups: torch.Tensor


def build_compatible_neighbor_table(
    standardized_latents: torch.Tensor,
    labels: torch.Tensor,
    *,
    local_pool_size: int = LOCAL_POOL_SIZE,
    expected_record_count: int = TUNING_RECORD_COUNT,
    candidate_group_policy: str = COARSE_CANDIDATE_GROUP_POLICY,
) -> CompatibleNeighborTable:
    """Precompute stable nearest non-self neighbors inside the declared group."""

    if (
        not isinstance(standardized_latents, torch.Tensor)
        or standardized_latents.ndim != 3
        or tuple(standardized_latents.shape[1:]) != (4, 128)
    ):
        raise ValueError("standardized_latents must have shape (N,4,128)")
    if standardized_latents.shape[0] != labels.shape[0]:
        raise ValueError("latent and label populations differ")
    if not standardized_latents.is_floating_point() or not bool(
        torch.isfinite(standardized_latents).all()
    ):
        raise ValueError("standardized latents must be finite floating point")
    if isinstance(local_pool_size, bool) or int(local_pool_size) < NONSELF_CANDIDATE_COUNT:
        raise ValueError("local_pool_size must cover nineteen non-self candidates")
    if expected_record_count not in {TUNING_RECORD_COUNT, FULL_K500_RECORD_COUNT}:
        raise ValueError("compatible geometry expectation must be train400 or full-K500")
    count = int(standardized_latents.shape[0])
    if count != expected_record_count:
        raise ValueError(
            "compatible geometry population mismatch: "
            f"expected={expected_record_count}, actual={count}"
        )
    width = min(int(local_pool_size), count - 1)
    if candidate_group_policy == COARSE_CANDIDATE_GROUP_POLICY:
        groups = classify_compatibility_groups(labels)
    elif candidate_group_policy == EXACT_CANDIDATE_GROUP_POLICY:
        groups = classify_exact_label_groups(labels)
    else:
        raise ValueError(
            "candidate_group_policy must be norm_abnormal_other or "
            "exact_positive_set"
        )
    flat = standardized_latents.flatten(1).to(dtype=torch.float32)
    squared_norm = flat.square().sum(dim=1, keepdim=True)
    distance = (
        squared_norm + squared_norm.transpose(0, 1) - 2.0 * (flat @ flat.transpose(0, 1))
    ).clamp_min(0.0)
    compatible = groups[:, None] == groups[None, :]
    compatible.fill_diagonal_(False)
    counts = compatible.sum(dim=1).to(dtype=torch.int64)
    masked = distance.masked_fill(~compatible, float("inf"))
    ordered = torch.argsort(masked, dim=1, stable=True)[:, :width]
    finite = torch.gather(compatible, 1, ordered)
    table = torch.where(finite, ordered, torch.full_like(ordered, -1))
    return CompatibleNeighborTable(
        indices=table.contiguous(),
        counts=counts.contiguous(),
        groups=groups.contiguous(),
    )


def derive_matched_candidate_seed(
    base_seed: int,
    rng_identity: Sequence[str],
    hash_ids: Sequence[str],
) -> int:
    """Derive the D10/D11-shared candidate seed without a method identifier."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    identity = tuple(str(value) for value in rng_identity)
    hashes = tuple(str(value) for value in hash_ids)
    if not identity or not hashes or any(not value for value in (*identity, *hashes)):
        raise ValueError("candidate seed identity values must be non-empty")
    payload = "|".join(
        (CANDIDATE_RNG_NAMESPACE, str(base_seed), *identity, *hashes)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def derive_d19_paired_random_seed(
    candidate_seed: int,
    sample_ids: Sequence[str],
) -> int:
    """Derive D19's shared Dirichlet seed without a graph-node identity.

    The executor is free to topologically schedule ``compat_hard`` before
    ``compat_random``.  Seeding from the shared candidate batch makes both
    nodes consume the same paired random endpoint regardless of that order.
    """

    if isinstance(candidate_seed, bool) or not isinstance(candidate_seed, int):
        raise TypeError("candidate_seed must be an integer")
    resolved_ids = tuple(str(value) for value in sample_ids)
    if not resolved_ids or any(not value for value in resolved_ids):
        raise ValueError("D19 paired-random sample IDs must be non-empty")
    payload = "|".join(
        (D19_PAIRED_RANDOM_RNG_NAMESPACE, str(candidate_seed), *resolved_ids)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def derive_node_candidate_seed(
    base_seed: int,
    method_id: str,
    node_id: str,
    rng_identity: Sequence[str],
    hash_ids: Sequence[str],
) -> int:
    """Derive a replayable D16/D17 candidate seed scoped to node, not method."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if method_id not in _INDEPENDENT_RANDOM_METHOD_IDS:
        raise ValueError("node-specific candidate seeds are only valid for D16/D17")
    if node_id not in _RANDOM_NODE_LAMBDAS[method_id]:
        raise ValueError("node_id is not a declared compatible-random role")
    identity = tuple(str(value) for value in rng_identity)
    hashes = tuple(str(value) for value in hash_ids)
    if not identity or not hashes or any(not value for value in (*identity, *hashes)):
        raise ValueError("candidate seed identity values must be non-empty")
    payload = "|".join(
        (
            NODE_CANDIDATE_RNG_NAMESPACE,
            str(base_seed),
            node_id,
            *identity,
            *hashes,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass(frozen=True)
class SharedCandidateBatch:
    sample_ids: tuple[str, ...]
    anchor_pool_indices: torch.Tensor
    candidate_pool_indices: torch.Tensor
    anchor_standardized: torch.Tensor
    candidates_standardized: torch.Tensor
    candidate_labels: torch.Tensor
    eligible_mask: torch.Tensor
    seed: int


def select_shared_candidate_batch(
    *,
    pool: Any,
    neighbor_table: CompatibleNeighborTable,
    hash_ids: Sequence[str],
    seed: int,
) -> SharedCandidateBatch:
    """Select self+19 from the nearest compatible top-120 without replacement."""

    sample_ids = tuple(str(value) for value in hash_ids)
    anchor_indices = pool.indices_for_hashes(sample_ids)
    ranked = neighbor_table.indices.index_select(0, anchor_indices)
    eligible = neighbor_table.counts.index_select(0, anchor_indices) >= (
        NONSELF_CANDIDATE_COUNT
    )
    generator = torch.Generator(device=ranked.device)
    generator.manual_seed(int(seed))
    random_keys = torch.rand(
        ranked.shape, device=ranked.device, generator=generator, dtype=torch.float32
    )
    random_keys = random_keys.masked_fill(ranked < 0, float("inf"))
    sampled_positions = torch.argsort(random_keys, dim=1, stable=True)[
        :, :NONSELF_CANDIDATE_COUNT
    ]
    selected = torch.gather(ranked, 1, sampled_positions)
    selected = torch.where(
        (selected >= 0) & eligible[:, None],
        selected,
        anchor_indices[:, None],
    )
    candidate_indices = torch.cat((anchor_indices[:, None], selected), dim=1)
    standardized = pool.standardized_latents
    labels = pool.labels
    candidates = standardized.index_select(0, candidate_indices.flatten()).reshape(
        candidate_indices.shape[0], CANDIDATE_COUNT, 4, 128
    )
    candidate_labels = labels.index_select(0, candidate_indices.flatten()).reshape(
        candidate_indices.shape[0], CANDIDATE_COUNT, 5
    )
    return SharedCandidateBatch(
        sample_ids=sample_ids,
        anchor_pool_indices=anchor_indices.contiguous(),
        candidate_pool_indices=candidate_indices.contiguous(),
        anchor_standardized=standardized.index_select(0, anchor_indices).contiguous(),
        candidates_standardized=candidates.contiguous(),
        candidate_labels=candidate_labels.contiguous(),
        eligible_mask=eligible.contiguous(),
        seed=int(seed),
    )


def sample_dirichlet_alpha_one(
    batch_size: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample Dirichlet(1) with an explicit device-local generator."""

    uniform = torch.rand(
        (int(batch_size), int(width)),
        device=device,
        dtype=dtype,
        generator=generator,
    ).clamp_min(torch.finfo(dtype).tiny)
    exponential = -uniform.log()
    return (exponential / exponential.sum(dim=1, keepdim=True)).contiguous()


def project_hull(
    anchor: torch.Tensor,
    candidates: torch.Tensor,
    weights: torch.Tensor,
    *,
    hull_lambda: float = HULL_LAMBDA,
    epsilon: float = PGD_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project an anchor-dominant candidate mixture into the L2 ball."""

    if candidates.shape[:2] != weights.shape or candidates.shape[0] != anchor.shape[0]:
        raise ValueError("anchor/candidate/weight batch shapes are inconsistent")
    if tuple(candidates.shape[2:]) != tuple(anchor.shape[1:]):
        raise ValueError("candidate latent shape differs from anchor")
    view_shape = weights.shape + (1,) * (candidates.ndim - 2)
    mixed = (weights.view(view_shape) * candidates).sum(dim=1)
    proposed = anchor + float(hull_lambda) * (mixed - anchor)
    delta = proposed - anchor
    norm = delta.flatten(1).norm(p=2, dim=1)
    projection_scale = torch.clamp(
        float(epsilon) / norm.clamp_min(1.0e-12), max=1.0
    )
    broadcast = projection_scale.view(
        (projection_scale.shape[0],) + (1,) * (anchor.ndim - 1)
    )
    projected = anchor + delta * broadcast
    return projected, projection_scale, projection_scale * float(hull_lambda)


@dataclass(frozen=True)
class HardHullResult:
    standardized_latent: torch.Tensor
    weights: torch.Tensor
    projection_scale: torch.Tensor
    effective_lambda: torch.Tensor
    initial_score: torch.Tensor


@dataclass(frozen=True)
class MultiStartHardHullResult:
    standardized_latent: torch.Tensor
    weights: torch.Tensor
    projection_scale: torch.Tensor
    effective_lambda: torch.Tensor
    paired_random_score: torch.Tensor
    final_score: torch.Tensor
    winner: torch.Tensor


def optimize_hard_hull(
    anchor: torch.Tensor,
    candidates: torch.Tensor,
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_weights: torch.Tensor | None = None,
    steps: int = HARD_STEPS,
    learning_rate: float = HARD_LEARNING_RATE,
    init_logit_gap: float = HARD_INIT_LOGIT_GAP,
    hull_lambda: float = HULL_LAMBDA,
    epsilon: float = PGD_EPSILON,
) -> HardHullResult:
    """Optimize candidate logits while leaving classifier/decoder grads untouched."""

    batch, width = candidates.shape[:2]
    if initial_weights is None:
        initial_logits = torch.zeros(
            batch, width, device=anchor.device, dtype=anchor.dtype
        )
        initial_logits[:, 0] = float(init_logit_gap)
    else:
        if initial_weights.shape != (batch, width):
            raise ValueError("initial hard-hull weights have the wrong shape")
        if not bool(torch.isfinite(initial_weights).all()) or bool(
            (initial_weights <= 0.0).any()
        ):
            raise ValueError("initial hard-hull weights must be finite and positive")
        if not torch.allclose(
            initial_weights.sum(dim=1),
            torch.ones(batch, device=anchor.device, dtype=anchor.dtype),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError("initial hard-hull weights must sum to one")
        initial_logits = initial_weights.detach().log().to(
            device=anchor.device, dtype=anchor.dtype
        )
    logits = initial_logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate))
    with torch.no_grad():
        initial_weights = torch.softmax(logits, dim=1)
        initial_latent, _, _ = project_hull(
            anchor,
            candidates,
            initial_weights,
            hull_lambda=hull_lambda,
            epsilon=epsilon,
        )
        initial_score = score_fn(initial_latent).detach()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=1)
        projected, _, _ = project_hull(
            anchor,
            candidates,
            weights,
            hull_lambda=hull_lambda,
            epsilon=epsilon,
        )
        score = score_fn(projected)
        if score.shape != (batch,) or not bool(torch.isfinite(score).all()):
            raise ValueError("hard-hull score_fn must return finite shape (B,)")
        (gradient,) = torch.autograd.grad(score.mean(), logits)
        logits.grad = -gradient
        optimizer.step()
    with torch.no_grad():
        final_weights = torch.softmax(logits, dim=1)
        final_latent, projection_scale, effective_lambda = project_hull(
            anchor,
            candidates,
            final_weights,
            hull_lambda=hull_lambda,
            epsilon=epsilon,
        )
    return HardHullResult(
        standardized_latent=final_latent.detach().contiguous(),
        weights=final_weights.detach().contiguous(),
        projection_scale=projection_scale.detach().contiguous(),
        effective_lambda=effective_lambda.detach().contiguous(),
        initial_score=initial_score.contiguous(),
    )


def optimize_multistart_hard_hull(
    anchor: torch.Tensor,
    candidates: torch.Tensor,
    paired_random_weights: torch.Tensor,
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = HARD_STEPS,
    learning_rate: float = HARD_LEARNING_RATE,
    init_logit_gap: float = HARD_INIT_LOGIT_GAP,
    hull_lambda: float = HULL_LAMBDA,
    epsilon: float = PGD_EPSILON,
) -> MultiStartHardHullResult:
    """Select the hardest endpoint from random, anchor-start and random-start PGD.

    Candidate index zero is the unoptimized paired random endpoint.  Including
    it in the per-record argmax makes the advertised lower bound exact rather
    than relying on a monotonic optimizer step.
    """

    paired_random_latent, paired_scale, paired_lambda = project_hull(
        anchor,
        candidates,
        paired_random_weights,
        hull_lambda=hull_lambda,
        epsilon=epsilon,
    )
    with torch.no_grad():
        paired_random_score = score_fn(paired_random_latent).detach()
    anchor_start = optimize_hard_hull(
        anchor,
        candidates,
        score_fn=score_fn,
        steps=steps,
        learning_rate=learning_rate,
        init_logit_gap=init_logit_gap,
        hull_lambda=hull_lambda,
        epsilon=epsilon,
    )
    random_start = optimize_hard_hull(
        anchor,
        candidates,
        score_fn=score_fn,
        initial_weights=paired_random_weights,
        steps=steps,
        learning_rate=learning_rate,
        init_logit_gap=init_logit_gap,
        hull_lambda=hull_lambda,
        epsilon=epsilon,
    )
    with torch.no_grad():
        anchor_score = score_fn(anchor_start.standardized_latent).detach()
        random_start_score = score_fn(random_start.standardized_latent).detach()
        scores = torch.stack(
            (paired_random_score, anchor_score, random_start_score), dim=1
        )
        final_score, winner = scores.max(dim=1)
        latent_options = torch.stack(
            (
                paired_random_latent,
                anchor_start.standardized_latent,
                random_start.standardized_latent,
            ),
            dim=1,
        )
        weight_options = torch.stack(
            (
                paired_random_weights,
                anchor_start.weights,
                random_start.weights,
            ),
            dim=1,
        )
        scale_options = torch.stack(
            (paired_scale, anchor_start.projection_scale, random_start.projection_scale),
            dim=1,
        )
        lambda_options = torch.stack(
            (paired_lambda, anchor_start.effective_lambda, random_start.effective_lambda),
            dim=1,
        )
        batch_indices = torch.arange(anchor.shape[0], device=anchor.device)
        selected_latent = latent_options[batch_indices, winner]
        selected_weights = weight_options[batch_indices, winner]
        selected_scale = scale_options[batch_indices, winner]
        selected_lambda = lambda_options[batch_indices, winner]
    if bool((final_score + 1.0e-7 < paired_random_score).any()):
        raise RuntimeError("multi-start hard selection violated its random lower bound")
    return MultiStartHardHullResult(
        standardized_latent=selected_latent.detach().contiguous(),
        weights=selected_weights.detach().contiguous(),
        projection_scale=selected_scale.detach().contiguous(),
        effective_lambda=selected_lambda.detach().contiguous(),
        paired_random_score=paired_random_score.contiguous(),
        final_score=final_score.contiguous(),
        winner=winner.detach().contiguous(),
    )


@dataclass(frozen=True)
class TwoAxisProjection:
    standardized_latent: torch.Tensor
    real_direction: torch.Tensor
    nuisance_direction: torch.Tensor
    axis_cosine: torch.Tensor
    axis_coverage: torch.Tensor
    projection_scale: torch.Tensor
    effective_lambda: torch.Tensor
    convex_coefficients: torch.Tensor


def _validate_simplex(weights: torch.Tensor, *, width: int, name: str) -> None:
    if weights.ndim != 2 or weights.shape[1] != int(width):
        raise ValueError(f"{name} must have shape (B,{width})")
    if not weights.is_floating_point() or not bool(torch.isfinite(weights).all()):
        raise ValueError(f"{name} must be finite floating point")
    if bool((weights < -1.0e-7).any()):
        raise ValueError(f"{name} must be non-negative")
    expected = torch.ones(
        weights.shape[0], device=weights.device, dtype=weights.dtype
    )
    if not torch.allclose(weights.sum(dim=1), expected, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} must sum to one")


def project_two_axis_hull(
    anchor: torch.Tensor,
    real_candidates: torch.Tensor,
    nuisance_candidates: torch.Tensor,
    real_weights: torch.Tensor,
    nuisance_weights: torch.Tensor,
    *,
    hull_lambda: float = F2_HULL_LAMBDA,
    epsilon: float = PGD_EPSILON,
    validate: bool = True,
) -> TwoAxisProjection:
    """Project a fixed 50/50 real+nuisance endpoint without extrapolation."""

    if anchor.ndim != 3 or tuple(anchor.shape[1:]) != (4, 128):
        raise ValueError("two-axis anchor must have shape (B,4,128)")
    batch_size = int(anchor.shape[0])
    if real_candidates.shape != (batch_size, CANDIDATE_COUNT, 4, 128):
        raise ValueError("two-axis real candidates have the wrong shape")
    if nuisance_candidates.shape != (batch_size, F2_NUISANCE_WIDTH, 4, 128):
        raise ValueError("two-axis nuisance candidates have the wrong shape")
    if validate:
        if not all(
            bool(torch.isfinite(value).all())
            for value in (anchor, real_candidates, nuisance_candidates)
        ):
            raise ValueError("two-axis latent geometry must be finite")
        _validate_simplex(real_weights, width=CANDIDATE_COUNT, name="real_weights")
        _validate_simplex(
            nuisance_weights,
            width=F2_NUISANCE_WIDTH,
            name="nuisance_weights",
        )
    if not _same_float(
        F2_REAL_AXIS_MASS + F2_NUISANCE_AXIS_MASS, 1.0
    ):
        raise RuntimeError("F2 axis masses must sum to one")
    if float(hull_lambda) < 0.0 or float(hull_lambda) > F2_HULL_LAMBDA:
        raise ValueError(
            f"F2 hull_lambda must lie in [0,{F2_HULL_LAMBDA}]"
        )
    if float(epsilon) <= 0.0:
        raise ValueError("F2 epsilon must be positive")

    real_endpoint = (
        real_weights[:, :, None, None] * real_candidates
    ).sum(dim=1)
    nuisance_endpoint = (
        nuisance_weights[:, :, None, None] * nuisance_candidates
    ).sum(dim=1)
    real_direction = real_endpoint - anchor
    nuisance_direction = nuisance_endpoint - anchor
    combined_direction = (
        F2_REAL_AXIS_MASS * real_direction
        + F2_NUISANCE_AXIS_MASS * nuisance_direction
    )
    delta = float(hull_lambda) * combined_direction
    norm = delta.flatten(1).norm(p=2, dim=1)
    projection_scale = torch.clamp(
        float(epsilon) / norm.clamp_min(1.0e-12), max=1.0
    )
    projected = anchor + delta * projection_scale[:, None, None]
    effective_lambda = projection_scale * float(hull_lambda)
    anchor_coefficient = 1.0 - effective_lambda
    convex_coefficients = torch.cat(
        (
            anchor_coefficient[:, None],
            effective_lambda[:, None]
            * F2_REAL_AXIS_MASS
            * real_weights,
            effective_lambda[:, None]
            * F2_NUISANCE_AXIS_MASS
            * nuisance_weights,
        ),
        dim=1,
    )
    if validate:
        if bool((convex_coefficients < -1.0e-7).any()) or not torch.allclose(
            convex_coefficients.sum(dim=1),
            torch.ones_like(effective_lambda),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("F2 projection violated its convex-hull contract")

    real_norm = real_direction.flatten(1).norm(p=2, dim=1)
    nuisance_norm = nuisance_direction.flatten(1).norm(p=2, dim=1)
    axis_coverage = (real_norm > 1.0e-8) & (nuisance_norm > 1.0e-8)
    axis_cosine = F.cosine_similarity(
        real_direction.flatten(1), nuisance_direction.flatten(1), dim=1, eps=1.0e-8
    )
    axis_cosine = torch.where(
        axis_coverage, axis_cosine, torch.zeros_like(axis_cosine)
    )
    return TwoAxisProjection(
        standardized_latent=projected.contiguous(),
        real_direction=real_direction.contiguous(),
        nuisance_direction=nuisance_direction.contiguous(),
        axis_cosine=axis_cosine.contiguous(),
        axis_coverage=axis_coverage.contiguous(),
        projection_scale=projection_scale.contiguous(),
        effective_lambda=effective_lambda.contiguous(),
        convex_coefficients=convex_coefficients.contiguous(),
    )


@dataclass(frozen=True)
class TwoAxisHardResult:
    projection: TwoAxisProjection
    real_weights: torch.Tensor
    nuisance_weights: torch.Tensor
    initial_score: torch.Tensor


def optimize_two_axis_hard_hull(
    anchor: torch.Tensor,
    real_candidates: torch.Tensor,
    nuisance_candidates: torch.Tensor,
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = HARD_STEPS,
    learning_rate: float = HARD_LEARNING_RATE,
    init_logit_gap: float = HARD_INIT_LOGIT_GAP,
    hull_lambda: float = F2_HULL_LAMBDA,
    epsilon: float = PGD_EPSILON,
) -> TwoAxisHardResult:
    """Optimize independent non-negative real/nuisance logits at fixed mass."""

    batch_size = int(anchor.shape[0])
    real_logits = torch.zeros(
        batch_size,
        CANDIDATE_COUNT,
        device=anchor.device,
        dtype=anchor.dtype,
    )
    real_logits[:, 0] = float(init_logit_gap)
    real_logits.requires_grad_(True)
    nuisance_logits = torch.zeros(
        batch_size,
        F2_NUISANCE_WIDTH,
        device=anchor.device,
        dtype=anchor.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        (real_logits, nuisance_logits), lr=float(learning_rate)
    )
    with torch.no_grad():
        initial_projection = project_two_axis_hull(
            anchor,
            real_candidates,
            nuisance_candidates,
            torch.softmax(real_logits, dim=1),
            torch.softmax(nuisance_logits, dim=1),
            hull_lambda=hull_lambda,
            epsilon=epsilon,
            validate=True,
        )
        initial_score = score_fn(initial_projection.standardized_latent).detach()
        if initial_score.shape != (batch_size,) or not bool(
            torch.isfinite(initial_score).all()
        ):
            raise ValueError("two-axis score_fn must return finite shape (B,)")
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        real_weights = torch.softmax(real_logits, dim=1)
        nuisance_weights = torch.softmax(nuisance_logits, dim=1)
        projection = project_two_axis_hull(
            anchor,
            real_candidates,
            nuisance_candidates,
            real_weights,
            nuisance_weights,
            hull_lambda=hull_lambda,
            epsilon=epsilon,
            validate=False,
        )
        score = score_fn(projection.standardized_latent)
        if score.shape != (batch_size,) or not bool(torch.isfinite(score).all()):
            raise ValueError("two-axis score_fn must return finite shape (B,)")
        real_gradient, nuisance_gradient = torch.autograd.grad(
            score.mean(), (real_logits, nuisance_logits)
        )
        real_logits.grad = -real_gradient
        nuisance_logits.grad = -nuisance_gradient
        optimizer.step()
    with torch.no_grad():
        final_real_weights = torch.softmax(real_logits, dim=1)
        final_nuisance_weights = torch.softmax(nuisance_logits, dim=1)
        final_projection = project_two_axis_hull(
            anchor,
            real_candidates,
            nuisance_candidates,
            final_real_weights,
            final_nuisance_weights,
            hull_lambda=hull_lambda,
            epsilon=epsilon,
            validate=False,
        )
    return TwoAxisHardResult(
        projection=TwoAxisProjection(
            **{
                name: value.detach().contiguous()
                for name, value in final_projection.__dict__.items()
            }
        ),
        real_weights=final_real_weights.detach().contiguous(),
        nuisance_weights=final_nuisance_weights.detach().contiguous(),
        initial_score=initial_score.contiguous(),
    )


def build_anchor_soft_targets(
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    anchor_positive: float = ANCHOR_POSITIVE,
    label_mix_lambda: float = LABEL_MIX_LAMBDA,
    new_class_cap: float = NEW_CLASS_CAP,
) -> torch.Tensor:
    """Preserve anchor positives and softly admit compatible candidate classes."""

    anchor = _strict_binary_labels(anchor_labels)
    if candidate_labels.shape != (anchor.shape[0], weights.shape[1], 5):
        raise ValueError("candidate labels must have shape (B,M,5)")
    if not bool(torch.isfinite(candidate_labels).all()) or not bool(
        ((candidate_labels == 0.0) | (candidate_labels == 1.0)).all()
    ):
        raise ValueError("candidate labels must be finite binary targets")
    if weights.shape[:2] != candidate_labels.shape[:2] or not bool(
        torch.isfinite(weights).all()
    ):
        raise ValueError("candidate weights are invalid")
    if not torch.allclose(
        weights.sum(dim=1),
        torch.ones(weights.shape[0], device=weights.device, dtype=weights.dtype),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("candidate weights must sum to one")
    candidate_mix = (weights.unsqueeze(-1) * candidate_labels).sum(dim=1)
    target = (float(label_mix_lambda) * candidate_mix).clamp(
        min=0.0, max=float(new_class_cap)
    )
    target = torch.where(
        anchor > 0.5,
        torch.full_like(target, float(anchor_positive)),
        target,
    )
    anchor_abnormal = (anchor[:, ABNORMAL_CLASS_INDICES] > 0.5).any(dim=1)
    target[:, 3] = torch.where(
        anchor_abnormal, torch.zeros_like(target[:, 3]), target[:, 3]
    )
    return target.detach().to(dtype=torch.float32).contiguous()


def build_two_axis_soft_targets(
    anchor_labels: torch.Tensor,
    real_candidate_labels: torch.Tensor,
    real_weights: torch.Tensor,
) -> torch.Tensor:
    """Use anchor-soft targets after proving every real endpoint is exact-label."""

    anchor = _strict_binary_labels(anchor_labels)
    if real_candidate_labels.shape != (anchor.shape[0], real_weights.shape[1], 5):
        raise ValueError("F2 real candidate labels must have shape (B,M,5)")
    if not bool(
        (
            real_candidate_labels
            == anchor[:, None, :].expand_as(real_candidate_labels)
        ).all()
    ):
        raise ValueError("F2 real candidates must match the anchor positive set exactly")
    _validate_simplex(
        real_weights,
        width=int(real_candidate_labels.shape[1]),
        name="F2 real target weights",
    )
    anchor_soft = torch.where(
        anchor > 0.5,
        torch.full_like(anchor, ANCHOR_POSITIVE),
        torch.zeros_like(anchor),
    )
    target = anchor_soft
    anchor_abnormal = (anchor[:, ABNORMAL_CLASS_INDICES] > 0.5).any(dim=1)
    target[:, 3] = torch.where(
        anchor_abnormal, torch.zeros_like(target[:, 3]), target[:, 3]
    )
    return target.detach().to(dtype=torch.float32).contiguous()


def _f2_hard_diagnostic_values(
    *,
    anchor_latent: torch.Tensor,
    initial_latent: torch.Tensor,
    final_latent: torch.Tensor,
    clean_raw: torch.Tensor,
    anchor_decoded_raw: torch.Tensor,
    final_decoded_raw: torch.Tensor,
    anchor_logits: torch.Tensor,
    initial_logits: torch.Tensor,
    final_logits: torch.Tensor,
    targets: torch.Tensor,
    accepted_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, int]]:
    """Summarize F2 hard search without changing its objective or RNG stream."""

    batch_size = int(targets.shape[0])
    expected_latent = (batch_size, 4, 128)
    expected_raw = (batch_size, 1000, 12)
    expected_logits = (batch_size, 5)
    if any(
        value.shape != expected_latent
        for value in (anchor_latent, initial_latent, final_latent)
    ):
        raise ValueError("F2 hard diagnostic latents must align as (B,4,128)")
    if any(
        value.shape != expected_raw
        for value in (clean_raw, anchor_decoded_raw, final_decoded_raw)
    ):
        raise ValueError("F2 hard diagnostic waveforms must align as (B,1000,12)")
    if any(
        value.shape != expected_logits
        for value in (anchor_logits, initial_logits, final_logits, targets)
    ):
        raise ValueError("F2 hard diagnostic logits/targets must align as (B,5)")
    if accepted_mask.shape != (batch_size,) or accepted_mask.dtype != torch.bool:
        raise ValueError("F2 hard diagnostic accepted_mask must be bool shape (B,)")
    finite = accepted_mask & torch.stack(
        [
            torch.isfinite(value).flatten(1).all(dim=1)
            for value in (
                anchor_latent,
                initial_latent,
                final_latent,
                clean_raw,
                anchor_decoded_raw,
                final_decoded_raw,
                anchor_logits,
                initial_logits,
                final_logits,
                targets,
            )
        ],
        dim=0,
    ).all(dim=0)
    count = int(finite.sum().detach().cpu())
    values: dict[str, float] = {
        "hard_diagnostic_finite_accepted_batch_count": float(count),
        "hard_diagnostic_finite_accepted_fraction": float(
            finite.float().mean().detach().cpu()
        ),
    }
    weights: dict[str, int] = {
        "hard_diagnostic_finite_accepted_batch_count": 1,
        "hard_diagnostic_finite_accepted_fraction": batch_size,
    }
    if count == 0:
        return values, weights

    anchor_latent = anchor_latent[finite]
    initial_latent = initial_latent[finite]
    final_latent = final_latent[finite]
    clean_raw = clean_raw[finite]
    anchor_decoded_raw = anchor_decoded_raw[finite]
    final_decoded_raw = final_decoded_raw[finite]
    anchor_logits = anchor_logits[finite]
    initial_logits = initial_logits[finite]
    final_logits = final_logits[finite]
    targets = targets[finite]

    anchor_bce = F.binary_cross_entropy_with_logits(
        anchor_logits, targets, reduction="none"
    ).mean(dim=1)
    initial_bce = F.binary_cross_entropy_with_logits(
        initial_logits, targets, reduction="none"
    ).mean(dim=1)
    final_bce = F.binary_cross_entropy_with_logits(
        final_logits, targets, reduction="none"
    ).mean(dim=1)
    hard_clean_loss_gain = final_bce - anchor_bce
    search_loss_gain = final_bce - initial_bce
    atk_init_l2 = (final_latent - initial_latent).flatten(1).norm(p=2, dim=1)
    atk_anchor_l2 = (final_latent - anchor_latent).flatten(1).norm(p=2, dim=1)

    truth = targets >= 0.5
    anchor_positive = torch.sigmoid(anchor_logits) >= 0.5
    final_positive = torch.sigmoid(final_logits) >= 0.5
    per_label_clean_correct = anchor_positive == truth
    final_wrong = final_positive != truth
    attack_success = (per_label_clean_correct & final_wrong).any(dim=1)
    clean_correct_record = per_label_clean_correct.all(dim=1)
    asr_denominator = int(clean_correct_record.sum().detach().cpu())
    asr_numerator = int(
        (clean_correct_record & final_wrong.any(dim=1)).sum().detach().cpu()
    )
    positive_denominator_mask = clean_correct_record[:, None] & truth
    negative_denominator_mask = clean_correct_record[:, None] & ~truth
    positive_hide = positive_denominator_mask & ~final_positive
    negative_add = negative_denominator_mask & final_positive
    positive_denominator = int(positive_denominator_mask.sum().detach().cpu())
    negative_denominator = int(negative_denominator_mask.sum().detach().cpu())
    positive_numerator = int(positive_hide.sum().detach().cpu())
    negative_numerator = int(negative_add.sum().detach().cpu())

    roundtrip_rmse = (anchor_decoded_raw - clean_raw).square().flatten(1).mean(
        dim=1
    ).sqrt()
    latent_move_decode_rmse = (
        final_decoded_raw - anchor_decoded_raw
    ).square().flatten(1).mean(dim=1).sqrt()
    values.update(
        {
            "clean_bce_recomputed": float(anchor_bce.mean().detach().cpu()),
            "initial_bce_recomputed": float(initial_bce.mean().detach().cpu()),
            "final_bce_recomputed": float(final_bce.mean().detach().cpu()),
            "hard_clean_loss_gain_mean": float(
                hard_clean_loss_gain.mean().detach().cpu()
            ),
            "hard_clean_loss_gain_batch_median": float(
                torch.quantile(hard_clean_loss_gain, 0.5).detach().cpu()
            ),
            "hard_clean_loss_gain_batch_p90": float(
                torch.quantile(hard_clean_loss_gain, 0.9).detach().cpu()
            ),
            "hard_clean_loss_gain_positive_fraction": float(
                (hard_clean_loss_gain > 0.0).float().mean().detach().cpu()
            ),
            "search_loss_gain_mean": float(
                search_loss_gain.mean().detach().cpu()
            ),
            "search_loss_gain_batch_median": float(
                torch.quantile(search_loss_gain, 0.5).detach().cpu()
            ),
            "search_loss_gain_batch_p90": float(
                torch.quantile(search_loss_gain, 0.9).detach().cpu()
            ),
            "search_loss_gain_positive_fraction": float(
                (search_loss_gain > 0.0).float().mean().detach().cpu()
            ),
            "atk_init_l2": float(atk_init_l2.mean().detach().cpu()),
            "atk_anchor_l2": float(atk_anchor_l2.mean().detach().cpu()),
            "atk_anchor_l2_batch_max": float(atk_anchor_l2.max().detach().cpu()),
            "attack_success": float(attack_success.float().mean().detach().cpu()),
            "clean_correct_multilabel_asr_denominator_batch_count": float(
                asr_denominator
            ),
            "clean_correct_multilabel_asr_numerator_batch_count": float(
                asr_numerator
            ),
            "clean_correct_multilabel_asr_denominator_fraction": float(
                asr_denominator / count
            ),
            "positive_hide_denominator_batch_count": float(positive_denominator),
            "positive_hide_numerator_batch_count": float(positive_numerator),
            "negative_add_denominator_batch_count": float(negative_denominator),
            "negative_add_numerator_batch_count": float(negative_numerator),
            "vae_cached_anchor_decode_residual_rmse_mV": float(
                roundtrip_rmse.mean().detach().cpu()
            ),
            "latent_move_decode_rmse_mV": float(
                latent_move_decode_rmse.mean().detach().cpu()
            ),
            "latent_move_to_roundtrip_rmse_ratio": float(
                (
                    latent_move_decode_rmse
                    / roundtrip_rmse.clamp_min(1.0e-8)
                ).mean().detach().cpu()
            ),
        }
    )
    accepted_mean_keys = {
        "clean_bce_recomputed",
        "initial_bce_recomputed",
        "final_bce_recomputed",
        "hard_clean_loss_gain_mean",
        "hard_clean_loss_gain_batch_median",
        "hard_clean_loss_gain_batch_p90",
        "hard_clean_loss_gain_positive_fraction",
        "search_loss_gain_mean",
        "search_loss_gain_batch_median",
        "search_loss_gain_batch_p90",
        "search_loss_gain_positive_fraction",
        "atk_init_l2",
        "atk_anchor_l2",
        "atk_anchor_l2_batch_max",
        "attack_success",
        "clean_correct_multilabel_asr_denominator_fraction",
        "vae_cached_anchor_decode_residual_rmse_mV",
        "latent_move_decode_rmse_mV",
        "latent_move_to_roundtrip_rmse_ratio",
    }
    weights.update({name: count for name in accepted_mean_keys})
    for name in (
        "clean_correct_multilabel_asr_denominator_batch_count",
        "clean_correct_multilabel_asr_numerator_batch_count",
        "positive_hide_denominator_batch_count",
        "positive_hide_numerator_batch_count",
        "negative_add_denominator_batch_count",
        "negative_add_numerator_batch_count",
    ):
        weights[name] = 1
    if asr_denominator > 0:
        values["clean_correct_multilabel_asr"] = float(
            asr_numerator / asr_denominator
        )
        weights["clean_correct_multilabel_asr"] = asr_denominator
    if positive_denominator > 0:
        values["positive_hide_rate"] = float(
            positive_numerator / positive_denominator
        )
        weights["positive_hide_rate"] = positive_denominator
    if negative_denominator > 0:
        values["negative_add_rate"] = float(
            negative_numerator / negative_denominator
        )
        weights["negative_add_rate"] = negative_denominator
    return values, weights


@dataclass(frozen=True)
class NuisanceAxisBatch:
    sample_ids: tuple[str, ...]
    standardized_latents: torch.Tensor
    anchor_decoded_raw: torch.Tensor
    finite_mask: torch.Tensor
    endpoint_finite_mask: torch.Tensor
    encoder_call_count: int
    encoded_endpoint_count: int
    encode_host_enqueue_seconds: float


@dataclass(frozen=True)
class F2HardScoreBatch:
    sample_ids: tuple[str, ...]
    final_bce: torch.Tensor
    accepted_mask: torch.Tensor


@dataclass(frozen=True)
class CompatibleRuntimeContract:
    mode: str
    method_id: str
    local_pool_size: int
    candidate_group_policy: str

    @classmethod
    def from_method(cls, method: CompiledMethod) -> "CompatibleRuntimeContract":
        if method.profile_name not in SUPPORTED_METHOD_IDS:
            raise ValueError(
                f"unsupported compatible-runtime method {method.profile_name!r}"
            )
        raw = method.contracts.get(CONTRACT_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError(f"contracts.{CONTRACT_KEY} must be a mapping")
        expected = _expected_contract_for_method(method.profile_name)
        if set(raw) != set(expected):
            raise ValueError(
                f"contracts.{CONTRACT_KEY} keys must be exactly {sorted(expected)}"
            )
        for key, expected_value in expected.items():
            actual = raw[key]
            if (
                key == "candidate_group_policy"
                and method.profile_name == D19_METHOD_ID
            ):
                matches = actual in {
                    COARSE_CANDIDATE_GROUP_POLICY,
                    EXACT_CANDIDATE_GROUP_POLICY,
                }
            elif key == "local_pool_size" and method.profile_name == D19_METHOD_ID:
                matches = (
                    not isinstance(actual, bool)
                    and isinstance(actual, int)
                    and NONSELF_CANDIDATE_COUNT <= actual
                    <= FULL_K500_RECORD_COUNT - 1
                )
            elif isinstance(expected_value, float):
                matches = _same_float(actual, expected_value)
            else:
                matches = actual == expected_value
            if not matches:
                if (
                    key == "candidate_group_policy"
                    and method.profile_name == D19_METHOD_ID
                ):
                    raise ValueError(
                        f"contracts.{CONTRACT_KEY}.candidate_group_policy must "
                        "be norm_abnormal_other or exact_positive_set"
                    )
                if key == "local_pool_size" and method.profile_name == D19_METHOD_ID:
                    raise ValueError(
                        f"contracts.{CONTRACT_KEY}.local_pool_size must be an "
                        f"integer in [{NONSELF_CANDIDATE_COUNT},"
                        f"{FULL_K500_RECORD_COUNT - 1}]"
                    )
                raise ValueError(
                    f"contracts.{CONTRACT_KEY}.{key} must be {expected_value!r}"
                )

        required_common = {
            "fixed20_train_views": 0,
            "corruption_views_per_base_record_step": _CORRUPTION_VIEW_COUNTS[
                method.profile_name
            ],
            "teacher_enabled": False,
            "tuning_partition_only": "k500_tune_train",
            "full_k500_refit_supported": False,
            "one_outer_optimizer_step_per_clean_batch": True,
            "heldout_target_feedback_allowed": False,
            "quality_rejected_latent_policy": "retain_clean_and_raw_terms_only",
            "batch_norm_running_stats_policy": (
                "objective_view_weighted_once_per_base_batch"
            ),
        }
        for key, expected_value in required_common.items():
            if method.contracts.get(key) != expected_value:
                raise ValueError(f"contracts.{key} must be {expected_value!r}")
        if not _same_float(
            method.contracts.get("objective_global_scale"), OBJECTIVE_GLOBAL_SCALE
        ):
            raise ValueError("contracts.objective_global_scale must be 0.5")

        expected_nodes = {
            "clean_identity": "identity_raw100_view",
            **{
                f"corruption_{index}": "canonical_depth23_corruption_view"
                for index in range(
                    1, _CORRUPTION_VIEW_COUNTS[method.profile_name] + 1
                )
            },
        }
        if method.profile_name == F2_METHOD_ID:
            expected_nodes.update(
                {
                    "resource_bridge": "latent_threechain_augmix_view",
                    "nuisance_bridge": "paired_threechain_augmix_views",
                }
            )
        for role in _expected_roles(method.profile_name):
            expected_nodes[role] = "vae_lhat_hard_view"
        actual_nodes = {
            node.profile.node_id: node.profile.node_type for node in method.nodes
        }
        if actual_nodes != expected_nodes:
            raise ValueError("compatible runtime graph nodes differ from frozen arm")
        lhat_nodes = {
            name for name, node_type in actual_nodes.items() if node_type == "vae_lhat_hard_view"
        }
        expected_lhat_nodes = _expected_roles(method.profile_name)
        if lhat_nodes != expected_lhat_nodes:
            raise ValueError("compatible runtime has unexpected LHAT node IDs")
        if method.profile_name == F2_METHOD_ID:
            expected_order = (
                "clean_identity",
                *F2_NUISANCE_SOURCE_NODES,
                "resource_bridge",
                "nuisance_bridge",
                "compat_hard",
                "compat_random",
            )
            if tuple(actual_nodes) != expected_order:
                raise ValueError("F2 graph node order differs from frozen two-axis arm")
            expected_inputs = {
                "clean_identity": (),
                **{
                    f"corruption_{index}": ("clean_identity",)
                    for index in range(1, F2_NUISANCE_WIDTH + 1)
                },
                "resource_bridge": ("clean_identity",),
                "nuisance_bridge": (
                    "resource_bridge",
                    *F2_NUISANCE_SOURCE_NODES,
                ),
                "compat_random": ("nuisance_bridge",),
                "compat_hard": ("nuisance_bridge",),
            }
            actual_inputs = {
                node.profile.node_id: tuple(node.profile.inputs)
                for node in method.nodes
            }
            if actual_inputs != expected_inputs:
                raise ValueError("F2 graph dependencies differ from frozen two-axis arm")

        actual_objective = tuple(
            (term.name, term.kind, tuple(term.views), float(term.weight))
            for term in method.objective.terms
        )
        if actual_objective != _EXPECTED_RAW_OBJECTIVES[method.profile_name]:
            raise ValueError("compatible runtime objective differs from frozen arm")
        actual_bn = method.contracts.get("batch_norm_objective_view_weights")
        expected_bn = _EXPECTED_BN_WEIGHTS[method.profile_name]
        if not isinstance(actual_bn, Mapping) or set(actual_bn) != set(expected_bn):
            raise ValueError("compatible runtime BN weights cover the wrong views")
        if any(
            not _same_float(actual_bn[name], expected)
            for name, expected in expected_bn.items()
        ):
            raise ValueError("compatible runtime BN weights differ from frozen arm")
        if method.profile_name == D18_METHOD_ID:
            actual_effective = method.contracts.get("effective_objective_weights")
            if not isinstance(actual_effective, Mapping) or set(actual_effective) != set(
                _D18_EFFECTIVE_OBJECTIVE_WEIGHTS
            ):
                raise ValueError("D18 effective objective weights cover the wrong terms")
            if any(
                not _same_float(actual_effective[name], expected)
                for name, expected in _D18_EFFECTIVE_OBJECTIVE_WEIGHTS.items()
            ):
                raise ValueError("D18 effective objective weights differ from frozen arm")
        if method.profile_name == D14_METHOD_ID:
            if method.contracts.get("strict_promotion_gate") != D14_STRICT_PROMOTION_GATE:
                raise ValueError(
                    "D14 contracts.strict_promotion_gate differs from the preregistered gate"
                )
        return cls(
            mode=str(expected["mode"]),
            method_id=method.profile_name,
            local_pool_size=int(raw["local_pool_size"]),
            candidate_group_policy=str(raw["candidate_group_policy"]),
        )

    def describe(self) -> dict[str, Any]:
        payload = _expected_contract_for_method(self.method_id)
        payload["local_pool_size"] = self.local_pool_size
        payload["candidate_group_policy"] = self.candidate_group_policy
        return payload


def _declares_compatible_runtime(method: CompiledMethod) -> bool:
    return method.profile_name in SUPPORTED_METHOD_IDS or CONTRACT_KEY in method.contracts


class CompatibleLocalHullRuntime(MethodViewRuntime):
    """D10-D19/F2 compatible local-hull runtime with frozen RNG rules."""

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
        self.compatible = CompatibleRuntimeContract.from_method(method)
        if latent_pool is None or decoder is None:
            raise ValueError("compatible runtime requires latent_pool and VAE decoder")
        if method.profile_name == F2_METHOD_ID and encoder is None:
            raise ValueError("F2 compatible runtime requires a frozen VAE encoder")
        if method.profile_name != F2_METHOD_ID and encoder is not None:
            raise ValueError("compatible tuning runtime must not receive an encoder")
        geometry = getattr(latent_pool, "geometry_pool", latent_pool)
        self._f2_vae_checkpoint_sha256: str | None = None
        if method.profile_name == F2_METHOD_ID:
            expected_vae_sha256 = _f2_expected_vae_sha256(method, config_root)
            component_identities = {
                "encoder": getattr(
                    getattr(encoder, "checkpoint_identity", None), "sha256", None
                ),
                "decoder": getattr(
                    getattr(decoder, "checkpoint_identity", None), "sha256", None
                ),
                "latent_pool": getattr(
                    getattr(geometry, "identity", None), "encoder_identity", None
                ),
            }
            mismatched = {
                name: value
                for name, value in component_identities.items()
                if value != expected_vae_sha256
            }
            if mismatched:
                raise ValueError(
                    "F2 VAE checkpoint identity differs from the pinned config: "
                    f"{mismatched}"
                )
            self._f2_vae_checkpoint_sha256 = expected_vae_sha256
        record_count = len(geometry.hash_ids)
        if int(geometry.identity.record_count) != record_count:
            raise ValueError("compatible geometry identity record count mismatch")
        outer_hashes = tuple(str(value) for value in latent_pool.hash_ids)
        if len(outer_hashes) != record_count or set(outer_hashes) != set(
            geometry.hash_ids
        ):
            raise ValueError("compatible outer membership differs from geometry")
        promotion = active_full_k500_promotion()
        heldout_oracle = active_full_k500_heldout_oracle()
        if record_count == TUNING_RECORD_COUNT:
            if promotion is not None or heldout_oracle is not None:
                raise ValueError(
                    "full-K500 authorization requires full-K500, not train400 geometry"
                )
            geometry_population = "frozen_train400_only"
        elif record_count == FULL_K500_RECORD_COUNT:
            if promotion is not None:
                if method.profile_name != promotion.method_id:
                    raise ValueError(
                        "runtime method differs from the verified D13 selected-refit "
                        "or A7 promotion"
                    )
                if self.model_name != promotion.model_family:
                    raise ValueError(
                        "selected-refit model family differs from the promotion"
                    )
                if (
                    method.source_path is None
                    or method.source_path.resolve() != promotion.method_profile_path
                    or _sha256(method.source_path) != promotion.method_profile_sha256
                ):
                    raise ValueError(
                        "full-K500 runtime method profile differs from the promotion"
                    )
                if (
                    geometry.identity.encoder_identity
                    != promotion.vae_checkpoint_sha256
                ):
                    raise ValueError(
                        "full-K500 latent pool VAE identity differs from the promotion"
                    )
                decoder_identity = getattr(decoder, "checkpoint_identity", None)
                if (
                    getattr(decoder_identity, "sha256", None)
                    != promotion.vae_checkpoint_sha256
                ):
                    raise ValueError(
                        "full-K500 decoder VAE identity differs from the promotion"
                    )
                geometry_population = "selected_refit_full_k500"
            elif heldout_oracle is not None:
                if method.profile_name != D19_METHOD_ID:
                    raise ValueError(
                        "heldout oracle is restricted to the D19 candidate runtime"
                    )
                if self.model_name not in {"efficientnet1dv2", "ecgfounder"}:
                    raise ValueError(
                        "heldout oracle supports only managed "
                        "EfficientNet1DV2 and ECGFounder backbones"
                    )
                decoder_identity = getattr(decoder, "checkpoint_identity", None)
                decoder_sha256 = getattr(decoder_identity, "sha256", None)
                if geometry.identity.encoder_identity != decoder_sha256:
                    raise ValueError(
                        "heldout-oracle latent pool and decoder VAE identities differ"
                    )
                geometry_population = "heldout_oracle_full_k500"
            else:
                raise ValueError(
                    "compatible runtime rejects full-K500 without a verified "
                    "D13 selected-refit promotion, verified A7 selected-refit "
                    "promotion, or explicit heldout-oracle scope"
                )
        else:
            raise ValueError(
                "compatible runtime geometry must contain exactly train400 or full-K500"
            )
        standardizer_count = int(geometry.standardizer.count)
        if not 2 <= standardizer_count <= record_count:
            raise ValueError(
                "compatible standardizer fit count must be within the geometry population"
            )
        if (
            geometry.identity.standardizer_sha256
            != geometry.standardizer.identity_sha256
        ):
            raise ValueError("compatible standardizer identity mismatch")
        if self.lhat_config is None or self.lhat_config.num_candidates != CANDIDATE_COUNT:
            raise ValueError("compatible runtime requires the managed M=20 LHAT config")
        self._geometry_pool = geometry
        self._geometry_population = geometry_population
        self._promotion = promotion
        self._heldout_oracle = heldout_oracle
        candidate_group_policy = str(
            self.compatible.describe()["candidate_group_policy"]
        )
        self._neighbor_table = build_compatible_neighbor_table(
            geometry.standardized_latents,
            geometry.labels,
            local_pool_size=self.compatible.local_pool_size,
            expected_record_count=record_count,
            candidate_group_policy=candidate_group_policy,
        )
        self._candidate_group_policy = candidate_group_policy
        self._model_spec = get_model_spec(self.model_name)
        self._shared_candidates: SharedCandidateBatch | None = None
        self._candidate_seed: int | None = None
        self._node_candidates: dict[str, SharedCandidateBatch] = {}
        self._node_candidate_seeds: dict[str, int] = {}
        self._generation_identity: tuple[str, ...] | None = None
        self._seen_roles: set[str] = set()
        self._d19_paired_random_weights: torch.Tensor | None = None
        self._d19_paired_random_score: torch.Tensor | None = None
        self._d19_paired_random_seed: int | None = None
        self._nuisance_axis_batch: NuisanceAxisBatch | None = None
        self._f2_hard_score_batch: F2HardScoreBatch | None = None

    def _d19_paired_weights(
        self,
        anchor: torch.Tensor,
        shared: SharedCandidateBatch,
    ) -> tuple[torch.Tensor, int]:
        if self.method.profile_name != D19_METHOD_ID:
            raise RuntimeError("D19 paired weights requested by a different method")
        seed = derive_d19_paired_random_seed(int(shared.seed), shared.sample_ids)
        weights = self._d19_paired_random_weights
        if weights is None:
            generator = torch.Generator(device=anchor.device)
            generator.manual_seed(seed)
            weights = sample_dirichlet_alpha_one(
                int(anchor.shape[0]),
                CANDIDATE_COUNT,
                device=anchor.device,
                dtype=anchor.dtype,
                generator=generator,
            ).detach().contiguous()
            self._d19_paired_random_weights = weights
            self._d19_paired_random_seed = seed
        else:
            if self._d19_paired_random_seed != seed:
                raise RuntimeError("D19 paired-random seed changed inside one generation")
            if weights.shape != (int(anchor.shape[0]), CANDIDATE_COUNT):
                raise RuntimeError("D19 paired-random weights changed shape across nodes")
            if weights.device != anchor.device or weights.dtype != anchor.dtype:
                raise RuntimeError("D19 paired-random weights changed device or dtype")
        return weights, seed

    def _observe_d19_paired_score(self, score: torch.Tensor) -> None:
        observed = score.detach().contiguous()
        if self._d19_paired_random_score is None:
            self._d19_paired_random_score = observed
            return
        if not torch.allclose(
            observed,
            self._d19_paired_random_score,
            atol=1.0e-6,
            rtol=1.0e-5,
        ):
            raise RuntimeError("D19 paired random score changed across graph nodes")

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        independent = self.method.profile_name in _INDEPENDENT_RANDOM_METHOD_IDS
        runtime_description = {
            "contract": self.compatible.describe(),
            "runtime_path": str(Path(__file__).resolve()),
            "runtime_sha256": _sha256(Path(__file__).resolve()),
            "candidate_rng_namespace": (
                NODE_CANDIDATE_RNG_NAMESPACE if independent else CANDIDATE_RNG_NAMESPACE
            ),
            "candidate_rng_method_id_in_payload": False,
            "candidate_reuse_scope": (
                "one_generate_call_per_compatible_random_node"
                if independent
                else "one_generate_call_random_and_hard"
            ),
            "geometry_population": self._geometry_population,
            "geometry_record_count": len(self._geometry_pool.hash_ids),
            "geometry_identity": self._geometry_pool.identity.describe(),
            "standardizer_identity": self._geometry_pool.standardizer.describe(),
            "standardizer_source_population": self._geometry_population,
            "standardizer_fit_policy": "whitelist_exact_label_m20_eligible",
            "profile_full_k500_supported": False,
            "full_k500_supported": (
                self._promotion is not None or self._heldout_oracle is not None
            ),
            "selected_refit_full_k500_authorized": self._promotion is not None,
            "heldout_oracle_full_k500_authorized": self._heldout_oracle is not None,
            "promotion": (
                None if self._promotion is None else self._promotion.describe()
            ),
            "heldout_oracle": self._heldout_oracle,
            "group_names": (
                list(GROUP_NAMES)
                if self._candidate_group_policy == COARSE_CANDIDATE_GROUP_POLICY
                else [
                    _exact_group_name(int(value))
                    for value in torch.unique(
                        self._neighbor_table.groups, sorted=True
                    ).detach().cpu().tolist()
                ]
            ),
            "group_counts": {
                (
                    GROUP_NAMES[int(value)]
                    if self._candidate_group_policy
                    == COARSE_CANDIDATE_GROUP_POLICY
                    else _exact_group_name(int(value))
                ): int((self._neighbor_table.groups == int(value)).sum())
                for value in torch.unique(
                    self._neighbor_table.groups, sorted=True
                ).detach().cpu().tolist()
            },
            "soft_targets_metadata_key": "soft_targets",
            "objective_global_scale": OBJECTIVE_GLOBAL_SCALE,
            "effective_objective_weights": {
                term.name: float(term.weight) * OBJECTIVE_GLOBAL_SCALE
                for term in self.method.objective.terms
            },
        }
        if independent:
            runtime_description["candidate_rng_node_id_in_payload"] = True
            runtime_description["role_hull_lambdas"] = dict(
                _RANDOM_NODE_LAMBDAS[self.method.profile_name]
            )
        if self.method.profile_name == F2_METHOD_ID:
            runtime_description["two_axis"] = {
                "real_axis_mass": F2_REAL_AXIS_MASS,
                "nuisance_axis_mass": F2_NUISANCE_AXIS_MASS,
                "nuisance_source_nodes": list(F2_NUISANCE_SOURCE_NODES),
                "encoder_calls_per_generate": 1,
                "encoder_posterior_sample": False,
                "nuisance_endpoint_count_per_record": F2_NUISANCE_WIDTH,
                "residual_correction": F2_RESIDUAL_CORRECTION,
                "convexity_policy": "nonnegative_unit_sum_no_extrapolation",
                "soft_target_axis_policy": (
                    "exact_label_anchor_soft_no_class_admission"
                ),
                "vae_checkpoint_sha256": self._f2_vae_checkpoint_sha256,
            }
        base["sandbox_compatible_runtime"] = runtime_description
        return base

    def _get_candidates(
        self, source: WaveformView, *, role: str
    ) -> SharedCandidateBatch:
        if self._generation_identity is None:
            raise RuntimeError("compatible candidate generation is outside generate()")
        if self.method.profile_name in _INDEPENDENT_RANDOM_METHOD_IDS:
            seed = self._node_candidate_seeds.get(role)
            if seed is None:
                raise RuntimeError("compatible node has no node-specific candidate seed")
            if role not in self._node_candidates:
                self._node_candidates[role] = select_shared_candidate_batch(
                    pool=self._geometry_pool,
                    neighbor_table=self._neighbor_table,
                    hash_ids=source.sample_ids,
                    seed=seed,
                )
            cached = self._node_candidates[role]
        else:
            if self._candidate_seed is None:
                raise RuntimeError("compatible shared candidate seed is unavailable")
            if self._shared_candidates is None:
                self._shared_candidates = select_shared_candidate_batch(
                    pool=self._geometry_pool,
                    neighbor_table=self._neighbor_table,
                    hash_ids=source.sample_ids,
                    seed=self._candidate_seed,
                )
            cached = self._shared_candidates
        if cached.sample_ids != source.sample_ids:
            raise RuntimeError("compatible candidate cache changed batch identity")
        bound_labels = self._geometry_pool.labels.index_select(
            0, cached.anchor_pool_indices
        ).to(device=source.labels.device, dtype=source.labels.dtype)
        if not torch.equal(bound_labels, source.labels):
            raise RuntimeError("compatible batch labels differ from train400 pool labels")
        return cached

    def _decode_standardized(self, value: torch.Tensor) -> torch.Tensor:
        assert self.decoder is not None
        latent = self._geometry_pool.standardizer.inverse_transform(value)
        return decode_to_ptbxl_waveform(self.decoder, latent, target_points=1000)

    def _classifier_logits(
        self, waveform: torch.Tensor, classifier: nn.Module
    ) -> torch.Tensor:
        assert self.lhat_config is not None
        model_input = prepare_canonical_model_input(
            waveform,
            self._model_spec,
            epsilon=self.lhat_config.normalization_epsilon,
        )
        return validate_model_output(
            classifier(model_input),
            self._model_spec,
            batch_size=int(waveform.shape[0]),
            check_finite=False,
        )

    def _resource_bridge(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if self.method.profile_name != F2_METHOD_ID:
            raise RuntimeError("resource_bridge is reserved for F2")
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("F2 resource_bridge requires one clean WaveformView")
        source = inputs[0]
        if context.resource("vae_encoder") is not self.encoder:
            raise RuntimeError("F2 resource_bridge encoder identity drifted")
        context.record_diagnostic("identity_only", 1.0)
        return WaveformView(
            name=context.node_id,
            waveform=source.waveform,
            labels=source.labels,
            sample_ids=source.sample_ids,
            valid_mask=source.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="f2_encoder_requirement_identity_bridge",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "waveform_mutated": False,
                    "purpose": "compile_frozen_encoder_requirement",
                },
            ),
            metadata={"resource_bridge_identity": True},
        )

    def _nuisance_bridge(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> WaveformView:
        if self.method.profile_name != F2_METHOD_ID:
            raise RuntimeError("nuisance_bridge is reserved for F2")
        if len(inputs) != 1 + F2_NUISANCE_WIDTH or not all(
            isinstance(value, WaveformView) for value in inputs
        ):
            raise TypeError("F2 nuisance_bridge requires clean plus four raw views")
        if self._nuisance_axis_batch is not None:
            raise RuntimeError("F2 nuisance endpoints were encoded more than once")
        clean = inputs[0]
        raw_views = inputs[1:]
        assert isinstance(clean, WaveformView)
        if clean.name != "resource_bridge" or tuple(
            value.name for value in raw_views
        ) != F2_NUISANCE_SOURCE_NODES:
            raise RuntimeError("F2 nuisance_bridge received unexpected graph parents")
        for value in raw_views:
            assert isinstance(value, WaveformView)
            if value.sample_ids != clean.sample_ids or not torch.equal(
                value.labels, clean.labels
            ):
                raise RuntimeError("F2 nuisance endpoint alignment drifted")

        raw = torch.stack(
            tuple(value.waveform for value in raw_views), dim=1
        ).contiguous()
        endpoint_valid = torch.stack(
            tuple(value.valid_mask for value in raw_views), dim=1
        )
        endpoint_finite = torch.isfinite(raw).flatten(2).all(dim=2)
        endpoint_valid = endpoint_valid & endpoint_finite
        safe_raw = torch.where(
            endpoint_valid[:, :, None, None],
            raw,
            clean.waveform[:, None].expand_as(raw),
        ).contiguous()
        flat = safe_raw.reshape(
            clean.batch_size * F2_NUISANCE_WIDTH, 1000, 12
        )
        assert self.encoder is not None and self.decoder is not None
        anchor_pool_indices = self._geometry_pool.indices_for_hashes(
            clean.sample_ids
        )
        anchor_standardized = self._geometry_pool.standardized_latents.index_select(
            0, anchor_pool_indices
        )
        started = time.perf_counter()
        with _frozen_vae_components(self.encoder, self.decoder), torch.no_grad():
            encoded = self.encoder(
                prepare_ecgtwin_encoder_input(flat), sample=False
            )
            anchor_decoded_raw = self._decode_standardized(anchor_standardized)
        encode_host_enqueue_seconds = time.perf_counter() - started
        if not isinstance(encoded, tuple) or len(encoded) != 3:
            raise TypeError(
                "F2 VAE encoder must return (scaled_latent, mean, log_variance)"
            )
        scaled_latent = encoded[0]
        if scaled_latent.shape != (
            clean.batch_size * F2_NUISANCE_WIDTH,
            4,
            128,
        ):
            raise ValueError("F2 VAE encoder returned an unexpected latent shape")
        latent_finite = torch.isfinite(scaled_latent).flatten(1).all(dim=1)
        standardized_flat = self._geometry_pool.standardizer.transform(
            torch.nan_to_num(scaled_latent.detach())
        )
        standardized_finite = torch.isfinite(standardized_flat).flatten(1).all(dim=1)
        endpoint_latent_finite = (latent_finite & standardized_finite).reshape(
            clean.batch_size, F2_NUISANCE_WIDTH
        )
        endpoint_valid = endpoint_valid & endpoint_latent_finite
        standardized = standardized_flat.reshape(
            clean.batch_size, F2_NUISANCE_WIDTH, 4, 128
        ).contiguous()
        anchor_decode_finite = torch.isfinite(anchor_decoded_raw).flatten(1).all(dim=1)
        finite_mask = (
            endpoint_valid.all(dim=1) & clean.valid_mask & anchor_decode_finite
        )
        self._nuisance_axis_batch = NuisanceAxisBatch(
            sample_ids=clean.sample_ids,
            standardized_latents=standardized,
            anchor_decoded_raw=anchor_decoded_raw.detach().contiguous(),
            finite_mask=finite_mask.contiguous(),
            endpoint_finite_mask=endpoint_valid.contiguous(),
            encoder_call_count=1,
            encoded_endpoint_count=int(flat.shape[0]),
            encode_host_enqueue_seconds=float(encode_host_enqueue_seconds),
        )
        roundtrip_rmse = (
            anchor_decoded_raw - clean.waveform
        ).square().flatten(1).mean(dim=1).sqrt()
        finite_values = torch.stack(
            (
                endpoint_valid.float().mean(),
                anchor_decode_finite.float().mean(),
                finite_mask.float().mean(),
                roundtrip_rmse.mean(),
            )
        ).detach().cpu().tolist()
        diagnostics = {
            "encoder_call_count": 1.0,
            "encoded_endpoint_count": float(flat.shape[0]),
            "endpoint_finite_fraction": float(finite_values[0]),
            "anchor_decode_finite_fraction": float(finite_values[1]),
            "record_finite_fraction": float(finite_values[2]),
            "vae_cached_anchor_decode_residual_rmse_mV": float(finite_values[3]),
            # CUDA kernels are asynchronous.  This deliberately records only
            # host enqueue time so the runtime does not add a synchronization;
            # trainer wall-clock diagnostics remain the efficiency gate.
            "encode_host_enqueue_seconds": float(encode_host_enqueue_seconds),
        }
        for name, value in diagnostics.items():
            context.record_diagnostic(name, value)
        return WaveformView(
            name=context.node_id,
            waveform=clean.waveform,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            valid_mask=clean.valid_mask,
            provenance=Provenance(
                node_id=context.node_id,
                operation="f2_four_raw_endpoints_single_batch_vae_encode",
                parent_names=tuple(value.name for value in inputs),
                rng_namespace=context.rng_namespace,
                parameters={
                    "posterior_sample": False,
                    "encoder_call_count": 1,
                    "endpoint_count_per_record": F2_NUISANCE_WIDTH,
                    "standardizer_source": "frozen_train400_only",
                    "standardizer_sha256": (
                        self._geometry_pool.standardizer.identity_sha256
                    ),
                    "residual_correction": F2_RESIDUAL_CORRECTION,
                },
            ),
            metadata={
                "nuisance_bridge_identity": True,
                "endpoint_finite_mask": endpoint_valid.detach(),
                "diagnostic_means": diagnostics,
            },
        )

    def _dispatch_augmix(
        self, context: NodeContext, inputs: tuple[ViewValue, ...]
    ) -> ViewValue:
        if self.method.profile_name == F2_METHOD_ID:
            if context.node_id == "resource_bridge":
                return self._resource_bridge(context, inputs)
            if context.node_id == "nuisance_bridge":
                return self._nuisance_bridge(context, inputs)
            raise RuntimeError(f"unexpected F2 augmix bridge {context.node_id!r}")
        return super()._dispatch_augmix(context, inputs)

    def _canonical_corruption(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
        *,
        composition_indices: torch.Tensor | None = None,
    ) -> ViewValue:
        value = super()._canonical_corruption(
            context,
            inputs,
            composition_indices=composition_indices,
        )
        if self.method.profile_name != F2_METHOD_ID:
            return value
        if not isinstance(value, WaveformView):
            raise TypeError("F2 canonical corruption must return a WaveformView")
        diagnostics = value.metadata.get("corruption_diagnostics")
        if diagnostics is None:
            raise RuntimeError("F2 random corruption lacks device diagnostics")
        operator_mask = diagnostics.operator_mask
        if operator_mask.shape != (value.batch_size, len(CANONICAL_OPERATORS)):
            raise RuntimeError("F2 corruption operator mask has an invalid shape")
        usage = operator_mask.float().mean(dim=0).detach().cpu().tolist()
        for operator_name, fraction in zip(
            CANONICAL_OPERATORS, usage, strict=True
        ):
            context.record_diagnostic(
                f"operator_usage_fraction_{operator_name}", float(fraction)
            )
        return value

    def _build_two_axis_view(
        self,
        context: NodeContext,
        source: WaveformView,
        *,
        role: str,
        shared: SharedCandidateBatch,
    ) -> WaveformView:
        cache = self._nuisance_axis_batch
        if cache is None or cache.sample_ids != source.sample_ids:
            raise RuntimeError("F2 nuisance cache is unavailable or misaligned")
        eligible_mask = (
            shared.eligible_mask.to(device=source.waveform.device)
            & cache.finite_mask.to(device=source.waveform.device)
            & source.valid_mask
        )
        eligible_positions = _positions(eligible_mask)
        full_waveform = source.waveform.clone()
        full_valid = torch.zeros(
            source.batch_size, device=source.waveform.device, dtype=torch.bool
        )
        full_soft_targets = source.labels.detach().clone().to(dtype=torch.float32)
        rejected: list[dict[str, str]] = []
        accepted_positions: tuple[int, ...] = ()
        diagnostic_values: dict[str, float] = {
            "real_axis_mass": F2_REAL_AXIS_MASS,
            "nuisance_axis_mass": F2_NUISANCE_AXIS_MASS,
            "nuisance_encoder_call_count": float(cache.encoder_call_count),
            "nuisance_encoded_endpoint_count": float(cache.encoded_endpoint_count),
            "nuisance_encode_host_enqueue_seconds": float(
                cache.encode_host_enqueue_seconds
            ),
            "nuisance_endpoint_finite_fraction": float(
                cache.endpoint_finite_mask.float().mean().detach().cpu()
            ),
            "joint_eligible_fraction": float(
                eligible_mask.float().mean().detach().cpu()
            ),
        }
        stored_real_weights = source.waveform.new_empty((0, CANDIDATE_COUNT))
        stored_nuisance_weights = source.waveform.new_empty(
            (0, F2_NUISANCE_WIDTH)
        )
        stored_coefficients = source.waveform.new_empty(
            (0, 1 + CANDIDATE_COUNT + F2_NUISANCE_WIDTH)
        )
        hard_diagnostic_inputs: dict[str, torch.Tensor] | None = None
        hard_diagnostic_weights: dict[str, int] = {}
        random_score: torch.Tensor | None = None

        if eligible_positions:
            positions = torch.as_tensor(
                eligible_positions, device=source.waveform.device, dtype=torch.int64
            )
            anchor = shared.anchor_standardized.index_select(0, positions)
            real_candidates = shared.candidates_standardized.index_select(
                0, positions
            )
            real_candidate_labels = shared.candidate_labels.index_select(0, positions)
            nuisance_candidates = cache.standardized_latents.index_select(0, positions)
            targets = source.labels.index_select(0, positions)
            clean_for_probe = source.waveform.index_select(0, positions)
            anchor_decoded_for_probe = cache.anchor_decoded_raw.index_select(
                0, positions
            )
            classifier = context.resource("classifier")
            assert self.decoder is not None
            started = time.perf_counter()
            with (
                _frozen_classifier_for_attack(classifier),
                _frozen_vae_components(self.decoder),
            ):
                if role == "compat_random":
                    real_generator = context.torch_generator(
                        "two_axis_real_dirichlet", device=source.waveform.device
                    )
                    nuisance_generator = context.torch_generator(
                        "two_axis_nuisance_dirichlet", device=source.waveform.device
                    )
                    real_weights = sample_dirichlet_alpha_one(
                        len(eligible_positions),
                        CANDIDATE_COUNT,
                        device=source.waveform.device,
                        dtype=anchor.dtype,
                        generator=real_generator,
                    )
                    nuisance_weights = sample_dirichlet_alpha_one(
                        len(eligible_positions),
                        F2_NUISANCE_WIDTH,
                        device=source.waveform.device,
                        dtype=anchor.dtype,
                        generator=nuisance_generator,
                    )
                    projection = project_two_axis_hull(
                        anchor,
                        real_candidates,
                        nuisance_candidates,
                        real_weights,
                        nuisance_weights,
                        hull_lambda=F2_HULL_LAMBDA,
                        epsilon=PGD_EPSILON,
                    )
                    with torch.no_grad():
                        decoded_raw = self._decode_standardized(
                            projection.standardized_latent
                        )
                        generated_raw = clean_for_probe + (
                            decoded_raw - anchor_decoded_for_probe
                        )
                        random_logits = self._classifier_logits(
                            generated_raw, classifier
                        )
                        random_score = _generated_search_loss(random_logits, targets)
                    initial_score = torch.zeros(
                        len(eligible_positions),
                        device=anchor.device,
                        dtype=anchor.dtype,
                    )
                    final_score = initial_score
                elif role == "compat_hard":
                    initial_probe: dict[str, torch.Tensor] = {}

                    def score_fn(value: torch.Tensor) -> torch.Tensor:
                        if not torch.is_grad_enabled() and not initial_probe:
                            decoded_raw = self._decode_standardized(value)
                            raw = clean_for_probe + (
                                decoded_raw - anchor_decoded_for_probe
                            )
                            paired_logits = self._classifier_logits(
                                torch.cat((clean_for_probe, raw), dim=0),
                                classifier,
                            )
                            clean_logits, logits = paired_logits.split(
                                len(eligible_positions), dim=0
                            )
                            initial_probe.update(
                                {
                                    "initial_latent": value.detach().contiguous(),
                                    "anchor_decoded_raw": (
                                        anchor_decoded_for_probe.detach().contiguous()
                                    ),
                                    "initial_decoded_raw": (
                                        decoded_raw.detach().contiguous()
                                    ),
                                    "clean_logits": clean_logits.detach().contiguous(),
                                    "initial_logits": logits.detach().contiguous(),
                                }
                            )
                        else:
                            decoded_raw = self._decode_standardized(value)
                            raw = clean_for_probe + (
                                decoded_raw - anchor_decoded_for_probe
                            )
                            logits = self._classifier_logits(raw, classifier)
                        return _generated_search_loss(logits, targets)

                    hard = optimize_two_axis_hard_hull(
                        anchor,
                        real_candidates,
                        nuisance_candidates,
                        score_fn=score_fn,
                        steps=HARD_STEPS,
                        learning_rate=HARD_LEARNING_RATE,
                        init_logit_gap=HARD_INIT_LOGIT_GAP,
                        hull_lambda=F2_HULL_LAMBDA,
                        epsilon=PGD_EPSILON,
                    )
                    projection = hard.projection
                    real_weights = hard.real_weights
                    nuisance_weights = hard.nuisance_weights
                    initial_score = hard.initial_score
                    required_probe_keys = {
                        "initial_latent",
                        "anchor_decoded_raw",
                        "initial_decoded_raw",
                        "clean_logits",
                        "initial_logits",
                    }
                    if set(initial_probe) != required_probe_keys:
                        raise RuntimeError("F2 hard initial probe was not captured exactly once")
                    initial_probe_bce = _generated_search_loss(
                        initial_probe["initial_logits"], targets
                    )
                    if not torch.allclose(
                        initial_probe_bce,
                        initial_score,
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        raise RuntimeError("F2 hard initial probe changed the attack score")
                    with torch.no_grad():
                        final_decoded_raw = self._decode_standardized(
                            projection.standardized_latent
                        )
                        generated_raw = clean_for_probe + (
                            final_decoded_raw - anchor_decoded_for_probe
                        )
                        final_logits = self._classifier_logits(
                            generated_raw, classifier
                        )
                        final_score = _generated_search_loss(final_logits, targets)
                    hard_diagnostic_inputs = {
                        "anchor_latent": anchor.detach(),
                        "initial_latent": initial_probe["initial_latent"],
                        "final_latent": projection.standardized_latent.detach(),
                        "clean_raw": clean_for_probe.detach(),
                        "anchor_decoded_raw": initial_probe[
                            "anchor_decoded_raw"
                        ],
                        "final_decoded_raw": final_decoded_raw.detach(),
                        "anchor_logits": initial_probe["clean_logits"],
                        "initial_logits": initial_probe["initial_logits"],
                        "final_logits": final_logits.detach(),
                        "targets": targets.detach(),
                    }
                else:
                    raise RuntimeError(f"unsupported F2 role {role!r}")
            latent_seconds = time.perf_counter() - started

            soft_targets = build_two_axis_soft_targets(
                targets,
                real_candidate_labels,
                real_weights,
            )
            accepted_local, reasons = _quality_mask(
                generated_raw,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
            )
            accepted_local_positions = _positions(accepted_local)
            accepted_positions = tuple(
                eligible_positions[index] for index in accepted_local_positions
            )
            if accepted_local_positions:
                accepted_local_tensor = torch.as_tensor(
                    accepted_local_positions,
                    device=source.waveform.device,
                    dtype=torch.int64,
                )
                accepted_batch_tensor = torch.as_tensor(
                    accepted_positions,
                    device=source.waveform.device,
                    dtype=torch.int64,
                )
                full_waveform.index_copy_(
                    0,
                    accepted_batch_tensor,
                    generated_raw.index_select(0, accepted_local_tensor),
                )
                full_soft_targets.index_copy_(
                    0,
                    accepted_batch_tensor,
                    soft_targets.index_select(0, accepted_local_tensor),
                )
                full_valid[accepted_batch_tensor] = True
            for local_index, reason in enumerate(reasons):
                if reason != "accepted":
                    rejected.append(
                        {
                            "hash_id": source.sample_ids[
                                eligible_positions[local_index]
                            ],
                            "reason": reason,
                        }
                    )
            if role == "compat_hard":
                full_hard_score = torch.full(
                    (source.batch_size,),
                    float("nan"),
                    device=source.waveform.device,
                    dtype=final_score.dtype,
                )
                full_hard_score.index_copy_(0, positions, final_score.detach())
                full_hard_accepted = torch.zeros(
                    source.batch_size,
                    device=source.waveform.device,
                    dtype=torch.bool,
                )
                if accepted_positions:
                    full_hard_accepted[
                        torch.as_tensor(
                            accepted_positions,
                            device=source.waveform.device,
                            dtype=torch.int64,
                        )
                    ] = True
                self._f2_hard_score_batch = F2HardScoreBatch(
                    sample_ids=source.sample_ids,
                    final_bce=full_hard_score.contiguous(),
                    accepted_mask=full_hard_accepted.contiguous(),
                )
            elif role == "compat_random":
                hard_scores = self._f2_hard_score_batch
                if (
                    hard_scores is None
                    or hard_scores.sample_ids != source.sample_ids
                    or random_score is None
                ):
                    raise RuntimeError(
                        "F2 random-hard comparison lacks the preceding hard scores"
                    )
                hard_local = hard_scores.final_bce.index_select(0, positions)
                hard_valid_local = hard_scores.accepted_mask.index_select(0, positions)
                comparison_mask = (
                    accepted_local
                    & hard_valid_local
                    & torch.isfinite(random_score)
                    & torch.isfinite(hard_local)
                )
                comparison_count = int(comparison_mask.sum().detach().cpu())
                diagnostic_values[
                    "random_hard_comparison_batch_count"
                ] = float(comparison_count)
                hard_diagnostic_weights[
                    "random_hard_comparison_batch_count"
                ] = 1
                if comparison_count > 0:
                    selected_random = random_score[comparison_mask]
                    selected_hard = hard_local[comparison_mask]
                    difference = selected_hard - selected_random
                    diagnostic_values.update(
                        {
                            "random_bce_comparator_mean": float(
                                selected_random.mean().detach().cpu()
                            ),
                            "hard_bce_comparator_mean": float(
                                selected_hard.mean().detach().cpu()
                            ),
                            "hard_minus_random_bce_mean": float(
                                difference.mean().detach().cpu()
                            ),
                            "hard_minus_random_bce_batch_median": float(
                                torch.quantile(difference, 0.5).detach().cpu()
                            ),
                            "hard_harder_than_random_fraction": float(
                                (difference > 0.0).float().mean().detach().cpu()
                            ),
                        }
                    )
                    hard_diagnostic_weights.update(
                        {
                            name: comparison_count
                            for name in (
                                "random_bce_comparator_mean",
                                "hard_bce_comparator_mean",
                                "hard_minus_random_bce_mean",
                                "hard_minus_random_bce_batch_median",
                                "hard_harder_than_random_fraction",
                            )
                        }
                    )
            coefficient_sum_error = (
                projection.convex_coefficients.sum(dim=1) - 1.0
            ).abs()
            real_axis_norm = projection.real_direction.flatten(1).norm(p=2, dim=1)
            nuisance_axis_norm = projection.nuisance_direction.flatten(1).norm(
                p=2, dim=1
            )
            effective_real_norm = (
                projection.effective_lambda
                * F2_REAL_AXIS_MASS
                * real_axis_norm
            )
            effective_nuisance_norm = (
                projection.effective_lambda
                * F2_NUISANCE_AXIS_MASS
                * nuisance_axis_norm
            )
            actual_move_norm = (
                projection.standardized_latent - anchor
            ).flatten(1).norm(p=2, dim=1)
            means = torch.stack(
                (
                    projection.axis_cosine.mean(),
                    projection.axis_cosine.abs().mean(),
                    (
                        projection.axis_cosine.abs() < 0.95
                    ).float().mean(),
                    projection.axis_coverage.float().mean(),
                    projection.projection_scale.mean(),
                    projection.effective_lambda.mean(),
                    projection.effective_lambda.max(),
                    projection.convex_coefficients.min(),
                    coefficient_sum_error.max(),
                    real_weights.max(dim=1).values.mean(),
                    nuisance_weights.max(dim=1).values.mean(),
                    initial_score.mean(),
                    final_score.mean(),
                    (final_score - initial_score).mean(),
                    real_axis_norm.mean(),
                    nuisance_axis_norm.mean(),
                    effective_real_norm.mean(),
                    effective_nuisance_norm.mean(),
                    (
                        effective_nuisance_norm
                        / effective_real_norm.clamp_min(1.0e-12)
                    ).mean(),
                    actual_move_norm.mean(),
                    (projection.projection_scale < 1.0 - 1.0e-7).float().mean(),
                    (
                        projection.effective_lambda * F2_REAL_AXIS_MASS
                    ).mean(),
                    (
                        projection.effective_lambda * F2_NUISANCE_AXIS_MASS
                    ).mean(),
                )
            ).detach().cpu().tolist()
            diagnostic_values.update(
                dict(
                    zip(
                        (
                            "axis_cosine",
                            "axis_abs_cosine_mean",
                            "axis_noncollinear_fraction_abs_lt_0p95",
                            "axis_coverage",
                            "projection_scale",
                            "effective_lambda",
                            "effective_lambda_max",
                            "coefficient_min",
                            "coefficient_sum_error_max",
                            "real_coefficient_top1",
                            "nuisance_coefficient_top1",
                            "initial_bce",
                            "final_bce",
                            "search_loss_gain",
                            "real_axis_l2",
                            "nuisance_axis_l2",
                            "effective_real_contribution_l2",
                            "effective_nuisance_contribution_l2",
                            "effective_nuisance_to_real_l2_ratio",
                            "actual_latent_move_l2",
                            "epsilon_cap_hit_fraction",
                            "real_endpoint_coefficient_mass",
                            "nuisance_endpoint_coefficient_mass",
                        ),
                        (float(value) for value in means),
                        strict=True,
                    )
                )
            )
            projection_diagnostic_names = (
                "axis_cosine",
                "axis_abs_cosine_mean",
                "axis_noncollinear_fraction_abs_lt_0p95",
                "axis_coverage",
                "projection_scale",
                "effective_lambda",
                "effective_lambda_max",
                "coefficient_min",
                "coefficient_sum_error_max",
                "real_coefficient_top1",
                "nuisance_coefficient_top1",
                "initial_bce",
                "final_bce",
                "search_loss_gain",
                "real_axis_l2",
                "nuisance_axis_l2",
                "effective_real_contribution_l2",
                "effective_nuisance_contribution_l2",
                "effective_nuisance_to_real_l2_ratio",
                "actual_latent_move_l2",
                "epsilon_cap_hit_fraction",
                "real_endpoint_coefficient_mass",
                "nuisance_endpoint_coefficient_mass",
            )
            hard_diagnostic_weights.update(
                {
                    name: len(eligible_positions)
                    for name in projection_diagnostic_names
                }
            )
            diagnostic_values["latent_view_host_enqueue_seconds"] = float(
                latent_seconds
            )
            hard_diagnostic_weights["latent_view_host_enqueue_seconds"] = 1
            if hard_diagnostic_inputs is not None:
                hard_values, accepted_diagnostic_weights = _f2_hard_diagnostic_values(
                    **hard_diagnostic_inputs,
                    accepted_mask=accepted_local,
                )
                diagnostic_values.update(hard_values)
                hard_diagnostic_weights.update(accepted_diagnostic_weights)
            stored_real_weights = real_weights.detach().contiguous()
            stored_nuisance_weights = nuisance_weights.detach().contiguous()
            stored_coefficients = projection.convex_coefficients.detach().contiguous()

        for name, value in diagnostic_values.items():
            context.record_diagnostic(name, value)
        ineligible_positions = _positions(~eligible_mask)
        ineligible = tuple(source.sample_ids[index] for index in ineligible_positions)
        context.record_diagnostic("candidate_seed", int(shared.seed))
        context.record_diagnostic("candidate_eligible_count", len(eligible_positions))
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
                operation="f2_fixed_equal_mass_two_axis_convex_latent_decode",
                parent_names=(source.name,),
                rng_namespace=context.rng_namespace,
                parameters={
                    "role": role,
                    "candidate_group_policy": self._candidate_group_policy,
                    "real_candidate_count": CANDIDATE_COUNT,
                    "nuisance_candidate_count": F2_NUISANCE_WIDTH,
                    "real_axis_mass": F2_REAL_AXIS_MASS,
                    "nuisance_axis_mass": F2_NUISANCE_AXIS_MASS,
                    "hull_lambda": F2_HULL_LAMBDA,
                    "pgd_epsilon": PGD_EPSILON,
                    "posterior_sample": False,
                    "residual_correction": F2_RESIDUAL_CORRECTION,
                    "convexity_policy": (
                        "nonnegative_unit_sum_no_extrapolation"
                    ),
                    "soft_target_axis_policy": (
                        "exact_label_anchor_soft_no_class_admission"
                    ),
                    "candidate_rng_namespace": CANDIDATE_RNG_NAMESPACE,
                    "candidate_rng_method_id_in_payload": False,
                },
            ),
            metadata={
                "soft_targets": full_soft_targets.detach().contiguous(),
                "candidate_eligible_positions": eligible_positions,
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": ineligible,
                "quality_rejected": tuple(rejected),
                "diagnostic_means": diagnostic_values,
                "diagnostic_weights": hard_diagnostic_weights,
                "candidate_seed": int(shared.seed),
                "real_axis_weights": stored_real_weights,
                "nuisance_axis_weights": stored_nuisance_weights,
                "convex_coefficients": stored_coefficients,
                "nuisance_endpoint_finite_mask": (
                    cache.endpoint_finite_mask.detach()
                ),
            },
        )

    def _build_view(
        self,
        context: NodeContext,
        source: WaveformView,
        *,
        role: str,
    ) -> WaveformView:
        shared = self._get_candidates(source, role=role)
        if self.method.profile_name == F2_METHOD_ID:
            return self._build_two_axis_view(
                context,
                source,
                role=role,
                shared=shared,
            )
        hull_lambda = _RANDOM_NODE_LAMBDAS.get(
            self.method.profile_name, {}
        ).get(role, HULL_LAMBDA)
        eligible_positions = _positions(shared.eligible_mask)
        full_waveform = source.waveform.clone()
        full_valid = torch.zeros(
            source.batch_size, device=source.waveform.device, dtype=torch.bool
        )
        full_soft_targets = source.labels.detach().clone().to(dtype=torch.float32)
        rejected: list[dict[str, str]] = []
        diagnostic_values: dict[str, float] = {}
        multistart_winner_fractions: tuple[float, float, float] | None = None
        paired_random_seed: int | None = None
        accepted_positions: tuple[int, ...] = ()

        if eligible_positions:
            positions = torch.as_tensor(
                eligible_positions, device=source.waveform.device, dtype=torch.int64
            )
            anchor = shared.anchor_standardized.index_select(0, positions)
            candidates = shared.candidates_standardized.index_select(0, positions)
            candidate_labels = shared.candidate_labels.index_select(0, positions)
            targets = source.labels.index_select(0, positions)
            classifier = context.resource("classifier")
            assert self.decoder is not None
            with (
                _frozen_classifier_for_attack(classifier),
                _frozen_vae_components(self.decoder),
            ):
                if role in _random_roles(self.method.profile_name):
                    if self.method.profile_name == D19_METHOD_ID:
                        weights, paired_random_seed = self._d19_paired_weights(
                            anchor, shared
                        )
                        context.record_diagnostic(
                            f"rng/paired_random_dirichlet/{source.waveform.device}",
                            {
                                "seed": paired_random_seed,
                                "namespace": D19_PAIRED_RANDOM_RNG_NAMESPACE,
                                "execution_identity": list(
                                    self._generation_identity or ()
                                ),
                            },
                        )
                    else:
                        generator = context.torch_generator(
                            "random_dirichlet", device=source.waveform.device
                        )
                        weights = sample_dirichlet_alpha_one(
                            len(eligible_positions),
                            CANDIDATE_COUNT,
                            device=source.waveform.device,
                            dtype=anchor.dtype,
                            generator=generator,
                        )
                    standardized, projection_scale, effective_lambda = project_hull(
                        anchor,
                        candidates,
                        weights,
                        hull_lambda=hull_lambda,
                        epsilon=PGD_EPSILON,
                    )
                    with torch.no_grad():
                        generated_raw = self._decode_standardized(standardized)
                        if self.method.profile_name == D19_METHOD_ID:
                            random_logits = self._classifier_logits(
                                generated_raw, classifier
                            )
                            random_score = _generated_search_loss(
                                random_logits, targets
                            )
                        else:
                            random_score = torch.zeros(
                                len(eligible_positions),
                                device=anchor.device,
                                dtype=anchor.dtype,
                            )
                    initial_score = random_score
                    final_score = random_score
                    if self.method.profile_name == D19_METHOD_ID:
                        self._observe_d19_paired_score(random_score)
                elif role == "compat_hard":
                    def score_fn(value: torch.Tensor) -> torch.Tensor:
                        raw = self._decode_standardized(value)
                        logits = self._classifier_logits(raw, classifier)
                        return _generated_search_loss(logits, targets)

                    if self.method.profile_name == D19_METHOD_ID:
                        paired_weights, paired_random_seed = self._d19_paired_weights(
                            anchor, shared
                        )
                        context.record_diagnostic(
                            f"rng/paired_random_dirichlet/{source.waveform.device}",
                            {
                                "seed": paired_random_seed,
                                "namespace": D19_PAIRED_RANDOM_RNG_NAMESPACE,
                                "execution_identity": list(
                                    self._generation_identity or ()
                                ),
                            },
                        )
                        hard_multi = optimize_multistart_hard_hull(
                            anchor,
                            candidates,
                            paired_weights,
                            score_fn=score_fn,
                            steps=HARD_STEPS,
                            learning_rate=HARD_LEARNING_RATE,
                            init_logit_gap=HARD_INIT_LOGIT_GAP,
                            hull_lambda=HULL_LAMBDA,
                            epsilon=PGD_EPSILON,
                        )
                        self._observe_d19_paired_score(
                            hard_multi.paired_random_score
                        )
                        weights = hard_multi.weights
                        projection_scale = hard_multi.projection_scale
                        effective_lambda = hard_multi.effective_lambda
                        initial_score = hard_multi.paired_random_score
                        final_score = hard_multi.final_score
                        winner_counts = torch.bincount(
                            hard_multi.winner, minlength=3
                        ).float() / max(1, len(eligible_positions))
                        multistart_winner_fractions = tuple(
                            float(value)
                            for value in winner_counts.detach().cpu().tolist()
                        )
                        with torch.no_grad():
                            generated_raw = self._decode_standardized(
                                hard_multi.standardized_latent
                            )
                    else:
                        hard = optimize_hard_hull(
                            anchor,
                            candidates,
                            score_fn=score_fn,
                            steps=HARD_STEPS,
                            learning_rate=HARD_LEARNING_RATE,
                            init_logit_gap=HARD_INIT_LOGIT_GAP,
                            hull_lambda=HULL_LAMBDA,
                            epsilon=PGD_EPSILON,
                        )
                        weights = hard.weights
                        projection_scale = hard.projection_scale
                        effective_lambda = hard.effective_lambda
                        initial_score = hard.initial_score
                        with torch.no_grad():
                            generated_raw = self._decode_standardized(
                                hard.standardized_latent
                            )
                            final_logits = self._classifier_logits(
                                generated_raw, classifier
                            )
                            final_score = _generated_search_loss(
                                final_logits, targets
                            )
                else:
                    raise RuntimeError(f"unsupported compatible role {role!r}")

            soft_targets = build_anchor_soft_targets(
                targets,
                candidate_labels,
                weights,
            )
            accepted_local, reasons = _quality_mask(
                generated_raw,
                minimum_std_mV=self.minimum_std_mV,
                maximum_abs_mV=self.maximum_abs_mV,
            )
            accepted_local_positions = _positions(accepted_local)
            accepted_positions = tuple(
                eligible_positions[index] for index in accepted_local_positions
            )
            if accepted_local_positions:
                accepted_local_tensor = torch.as_tensor(
                    accepted_local_positions,
                    device=source.waveform.device,
                    dtype=torch.int64,
                )
                accepted_batch_tensor = torch.as_tensor(
                    accepted_positions,
                    device=source.waveform.device,
                    dtype=torch.int64,
                )
                full_waveform.index_copy_(
                    0,
                    accepted_batch_tensor,
                    generated_raw.index_select(0, accepted_local_tensor),
                )
                full_soft_targets.index_copy_(
                    0,
                    accepted_batch_tensor,
                    soft_targets.index_select(0, accepted_local_tensor),
                )
                full_valid[accepted_batch_tensor] = True
            for local_index, reason in enumerate(reasons):
                if reason != "accepted":
                    rejected.append(
                        {
                            "hash_id": source.sample_ids[eligible_positions[local_index]],
                            "reason": reason,
                        }
                    )
            means = torch.stack(
                (
                    weights[:, 0].mean(),
                    weights.max(dim=1).values.mean(),
                    projection_scale.mean(),
                    effective_lambda.mean(),
                    initial_score.mean(),
                    final_score.mean(),
                    (final_score - initial_score).mean(),
                )
            ).detach().cpu().tolist()
            diagnostic_values = dict(
                zip(
                    (
                        "self_weight",
                        "coefficient_top1",
                        "projection_scale",
                        "effective_lambda",
                        "initial_bce",
                        "final_bce",
                        "loss_gain",
                    ),
                    (float(value) for value in means),
                    strict=True,
                )
            )
            if multistart_winner_fractions is not None:
                diagnostic_values.update(
                    {
                        "winner_paired_random_fraction": multistart_winner_fractions[0],
                        "winner_anchor_start_fraction": multistart_winner_fractions[1],
                        "winner_random_start_fraction": multistart_winner_fractions[2],
                    }
                )
            for name, value in diagnostic_values.items():
                context.record_diagnostic(name, value)

        ineligible = tuple(
            source.sample_ids[index] for index in _positions(~shared.eligible_mask)
        )
        context.record_diagnostic("candidate_seed", int(shared.seed))
        context.record_diagnostic("candidate_eligible_count", len(eligible_positions))
        context.record_diagnostic("quality_accepted_count", len(accepted_positions))
        context.record_diagnostic("ineligible_count", len(ineligible))
        context.record_diagnostic("quality_rejected_count", len(rejected))
        provenance_parameters: dict[str, Any] = {
            "role": role,
            "candidate_group_policy": self._candidate_group_policy,
            "candidate_count": CANDIDATE_COUNT,
            "candidate_includes_anchor": True,
            "local_pool_size": self.compatible.local_pool_size,
            "hull_lambda": hull_lambda,
            "pgd_epsilon": PGD_EPSILON,
            "soft_target_mode": "anchor_soft",
            "candidate_rng_namespace": (
                NODE_CANDIDATE_RNG_NAMESPACE
                if self.method.profile_name in _INDEPENDENT_RANDOM_METHOD_IDS
                else CANDIDATE_RNG_NAMESPACE
            ),
            "candidate_rng_method_id_in_payload": False,
        }
        if self.method.profile_name in _INDEPENDENT_RANDOM_METHOD_IDS:
            provenance_parameters["candidate_rng_node_id_in_payload"] = True
        if self.method.profile_name == D19_METHOD_ID and role == "compat_hard":
            provenance_parameters.update(
                {
                    "hard_start_policy": "anchor_biased_plus_paired_random",
                    "hard_start_count": 2,
                    "hard_selection_include_unoptimized_random": True,
                    "hard_selection_guarantee": "final_bce_gte_paired_random",
                }
            )
        if self.method.profile_name == D19_METHOD_ID:
            if paired_random_seed is None:
                raise RuntimeError("D19 paired-random seed was not materialized")
            provenance_parameters.update(
                {
                    "paired_random_rng_namespace": D19_PAIRED_RANDOM_RNG_NAMESPACE,
                    "paired_random_seed": paired_random_seed,
                    "paired_random_order_independent": True,
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
                parameters=provenance_parameters,
            ),
            metadata={
                "soft_targets": full_soft_targets.detach().contiguous(),
                "candidate_eligible_positions": eligible_positions,
                "accepted_positions": accepted_positions,
                "ineligible_hash_ids": ineligible,
                "quality_rejected": tuple(rejected),
                "diagnostic_means": diagnostic_values,
                "candidate_seed": int(shared.seed),
                **(
                    {"paired_random_seed": paired_random_seed}
                    if paired_random_seed is not None
                    else {}
                ),
            },
        )

    def _lhat_attack(
        self,
        context: NodeContext,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        if len(inputs) != 1 or not isinstance(inputs[0], WaveformView):
            raise TypeError("compatible local hull requires one clean WaveformView")
        role = context.node_id
        allowed = _expected_roles(self.method.profile_name)
        if role not in allowed or role in self._seen_roles:
            raise RuntimeError("compatible node role is missing, duplicated or unexpected")
        self._seen_roles.add(role)
        return self._build_view(context, inputs[0], role=role)

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
            raise ValueError("compatible D10-D19/F2 do not support fixed20 exposures")
        expected_roles = _expected_roles(self.method.profile_name)
        self._shared_candidates = None
        self._node_candidates = {}
        self._node_candidate_seeds = {}
        self._nuisance_axis_batch = None
        self._f2_hard_score_batch = None
        if self.method.profile_name in _INDEPENDENT_RANDOM_METHOD_IDS:
            self._candidate_seed = None
            self._node_candidate_seeds = {
                role: derive_node_candidate_seed(
                    int(base_seed),
                    self.method.profile_name,
                    role,
                    rng_identity,
                    hash_ids,
                )
                for role in _random_roles(self.method.profile_name)
            }
        else:
            self._candidate_seed = derive_matched_candidate_seed(
                int(base_seed), rng_identity, hash_ids
            )
        self._generation_identity = tuple(str(value) for value in rng_identity)
        self._seen_roles = set()
        self._d19_paired_random_weights = None
        self._d19_paired_random_score = None
        self._d19_paired_random_seed = None
        try:
            generated = super().generate(
                clean_raw=clean_raw,
                targets=targets,
                hash_ids=hash_ids,
                classifier=classifier,
                base_seed=base_seed,
                rng_identity=rng_identity,
                composition_indices=None,
            )
            if self._seen_roles != expected_roles:
                raise RuntimeError("compatible graph did not execute every frozen role once")
            diagnostic_weights = dict(generated.diagnostic_weights)
            for value in generated.bundle.values.values():
                if not isinstance(value, WaveformView):
                    continue
                local_weights = value.metadata.get("diagnostic_weights")
                if not isinstance(local_weights, Mapping):
                    continue
                prefix = f"{value.provenance.node_id}/"
                for local_name, weight in local_weights.items():
                    full_name = f"{prefix}{local_name}"
                    if full_name not in generated.bundle.diagnostics:
                        raise RuntimeError(
                            "custom diagnostic weight lacks a recorded scalar: "
                            f"{full_name}"
                        )
                    if (
                        isinstance(weight, bool)
                        or not isinstance(weight, int)
                        or weight < 1
                    ):
                        raise ValueError(
                            f"diagnostic weight {full_name} must be a positive integer"
                        )
                    diagnostic_weights[full_name] = int(weight)
            generated = replace(generated, diagnostic_weights=diagnostic_weights)
            return generated
        finally:
            self._shared_candidates = None
            self._candidate_seed = None
            self._node_candidates = {}
            self._node_candidate_seeds = {}
            self._nuisance_axis_batch = None
            self._f2_hard_score_batch = None
            self._generation_identity = None
            self._seen_roles = set()
            self._d19_paired_random_weights = None
            self._d19_paired_random_score = None
            self._d19_paired_random_seed = None


def _soft_targets_for_view(view: WaveformView, clean: WaveformView) -> torch.Tensor:
    raw = view.metadata.get("soft_targets")
    if raw is None:
        return view.labels
    if view.name == BASE_VIEW_NAME:
        raise ValueError("clean_view may not override its hard targets")
    if not isinstance(raw, torch.Tensor) or raw.shape != clean.labels.shape:
        raise ValueError("metadata.soft_targets must have shape (B,5)")
    if raw.device != view.waveform.device or not raw.is_floating_point():
        raise ValueError("metadata.soft_targets must be floating and device aligned")
    if raw.requires_grad or raw.grad_fn is not None:
        raise ValueError("metadata.soft_targets must be detached")
    if not bool(torch.isfinite(raw).all()) or bool(((raw < 0.0) | (raw > 1.0)).any()):
        raise ValueError("metadata.soft_targets must be finite probabilities")
    return raw.to(dtype=view.labels.dtype)


def _compute_soft_objective(
    *,
    method: CompiledMethod,
    bundle: Any,
    model: nn.Module,
    spec: ModelSpec,
    normalization_epsilon: float,
    pos_weight: torch.Tensor | None,
    objective_term_names: Sequence[str] | None = None,
    batch_norm_plan: Any | None = None,
):
    """Whitelist objective logic with per-view detached soft BCE targets."""

    import core.online_trainer as online_trainer

    clean = bundle.require(BASE_VIEW_NAME)
    if not isinstance(clean, WaveformView):
        raise TypeError("method clean_view must be a WaveformView")
    batch_size = clean.batch_size
    if not bool(clean.valid_mask.all()):
        raise ValueError("clean_view must be valid for every base record")
    selected_names = (
        None
        if objective_term_names is None
        else tuple(str(value) for value in objective_term_names)
    )
    if selected_names is not None:
        if not selected_names or len(set(selected_names)) != len(selected_names):
            raise ValueError("objective_term_names must be non-empty and unique")
        known = {term.name for term in method.objective.terms}
        unknown = sorted(set(selected_names) - known)
        if unknown:
            raise ValueError(f"unknown objective term names: {unknown}")
    selected_terms = tuple(
        term
        for term in method.objective.terms
        if selected_names is None or term.name in selected_names
    )
    ordered_views: list[str] = []
    for term in selected_terms:
        for name in term.views:
            if name not in ordered_views:
                ordered_views.append(name)
    if not ordered_views:
        raise ValueError("method objective must contain at least one term")
    if batch_norm_plan is not None and len(batch_norm_plan.exposure_weights) != len(
        ordered_views
    ):
        raise ValueError("objective-view BatchNorm plan does not match objective views")

    views: dict[str, WaveformView] = {}
    objective_targets: dict[str, torch.Tensor] = {}
    positions: dict[str, torch.Tensor] = {}
    logits: dict[str, torch.Tensor] = {}
    full_to_local: dict[str, torch.Tensor] = {}
    for view_index, name in enumerate(ordered_views):
        value = bundle.require(name)
        if not isinstance(value, WaveformView):
            raise TypeError(f"objective view {name!r} must be a WaveformView")
        if value.sample_ids != clean.sample_ids:
            raise RuntimeError(f"objective view {name!r} changed origin order")
        if not torch.equal(value.labels, clean.labels):
            raise RuntimeError(f"objective view {name!r} changed typed labels")
        valid_positions = torch.nonzero(value.valid_mask, as_tuple=False).flatten()
        views[name] = value
        objective_targets[name] = _soft_targets_for_view(value, clean)
        positions[name] = valid_positions
        forward_positions = (
            torch.arange(batch_size, device=value.waveform.device, dtype=torch.int64)
            if batch_norm_plan is not None
            else valid_positions
        )
        lookup = torch.full(
            (batch_size,), -1, device=value.waveform.device, dtype=torch.int64
        )
        if forward_positions.numel():
            lookup[forward_positions] = torch.arange(
                forward_positions.numel(),
                device=value.waveform.device,
                dtype=torch.int64,
            )
            model_input = prepare_canonical_model_input(
                value.waveform.index_select(0, forward_positions),
                spec,
                epsilon=normalization_epsilon,
            )
            if batch_norm_plan is not None:
                batch_norm_plan.apply(view_index)
            logits[name] = validate_model_output(
                model(model_input),
                spec,
                batch_size=int(forward_positions.numel()),
                check_finite=False,
            )
        else:
            logits[name] = clean.waveform.new_empty((0, 5))
        full_to_local[name] = lookup

    reference = logits[ordered_views[0]]
    total = reference.sum() * 0.0
    raw_terms: dict[str, torch.Tensor] = {}
    weighted_terms: dict[str, torch.Tensor] = {}
    valid_counts: dict[str, int] = {}
    for term in selected_terms:
        if term.kind == "bce":
            name = term.views[0]
            count = int(positions[name].numel())
            if count:
                selected_targets = objective_targets[name].index_select(
                    0, positions[name]
                )
                selected_logits = logits[name].index_select(
                    0,
                    full_to_local[name].index_select(0, positions[name]),
                )
                raw_loss = F.binary_cross_entropy_with_logits(
                    selected_logits, selected_targets, pos_weight=pos_weight
                )
            else:
                raw_loss = reference.sum() * 0.0
        elif term.kind == "bernoulli_jsd":
            common = torch.ones(
                batch_size, device=clean.waveform.device, dtype=torch.bool
            )
            for name in term.views:
                common &= views[name].valid_mask
            common_positions = torch.nonzero(common, as_tuple=False).flatten()
            count = int(common_positions.numel())
            if count:
                aligned = tuple(
                    logits[name].index_select(
                        0,
                        full_to_local[name].index_select(0, common_positions),
                    )
                    for name in term.views
                )
                raw_loss = multilabel_jsd(aligned)
            else:
                raw_loss = reference.sum() * 0.0
        else:
            raise RuntimeError(f"unsupported compiled objective kind: {term.kind}")
        contribution = float(term.weight) * (float(count) / batch_size) * raw_loss
        raw_terms[term.name] = raw_loss
        weighted_terms[term.name] = contribution
        valid_counts[term.name] = count
        total = total + contribution
    return online_trainer._ObjectiveBatch(
        total=total,
        raw_terms=raw_terms,
        weighted_terms=weighted_terms,
        valid_counts=valid_counts,
    )


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Install D10-D18/F2 factory and objective adapters, then restore them."""

    import core.online_trainer as online_trainer

    original_factory = online_trainer.build_method_runtime
    original_objective = online_trainer._compute_objective

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
        if not _declares_compatible_runtime(method):
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
        CompatibleRuntimeContract.from_method(method)
        return CompatibleLocalHullRuntime(
            method,
            model_name=model_name,
            config_root=Path(config_root).expanduser().resolve(),
            latent_pool=latent_pool,
            encoder=encoder,
            decoder=decoder,
            minimum_std_mV=minimum_std_mV,
            maximum_abs_mV=maximum_abs_mV,
        )

    def patched_objective(**kwargs: Any):
        method = kwargs.get("method")
        if not isinstance(method, CompiledMethod) or not _declares_compatible_runtime(
            method
        ):
            return original_objective(**kwargs)
        CompatibleRuntimeContract.from_method(method)
        result = _compute_soft_objective(**kwargs)
        result_type = type(result)
        return result_type(
            total=result.total * OBJECTIVE_GLOBAL_SCALE,
            raw_terms=dict(result.raw_terms),
            weighted_terms={
                name: value * OBJECTIVE_GLOBAL_SCALE
                for name, value in result.weighted_terms.items()
            },
            valid_counts=dict(result.valid_counts),
        )

    online_trainer.build_method_runtime = patched_factory
    online_trainer._compute_objective = patched_objective
    try:
        yield
    finally:
        online_trainer._compute_objective = original_objective
        online_trainer.build_method_runtime = original_factory


install_compatible_runtime = patch_online_trainer_runtime


__all__ = [
    "A7_ECGFOUNDER_PROMOTION_POLICY",
    "ABNORMAL",
    "ANCHOR_POSITIVE",
    "CANDIDATE_COUNT",
    "CANDIDATE_RNG_NAMESPACE",
    "CompatibleLocalHullRuntime",
    "CompatibleNeighborTable",
    "CompatibleRuntimeContract",
    "D10_METHOD_ID",
    "D11_METHOD_ID",
    "D12_METHOD_ID",
    "D13_METHOD_ID",
    "D14_METHOD_ID",
    "D15_METHOD_ID",
    "D16_METHOD_ID",
    "D17_METHOD_ID",
    "D18_METHOD_ID",
    "D19_METHOD_ID",
    "D19_PAIRED_RANDOM_RNG_NAMESPACE",
    "F2_METHOD_ID",
    "F2_NUISANCE_AXIS_MASS",
    "F2_NUISANCE_SOURCE_NODES",
    "F2_REAL_AXIS_MASS",
    "D14_STRICT_PROMOTION_GATE",
    "FULL_K500_PROMOTION_POLICY",
    "FULL_K500_RECORD_COUNT",
    "FullK500Promotion",
    "GROUP_NAMES",
    "HARD_INIT_LOGIT_GAP",
    "HARD_LEARNING_RATE",
    "HARD_STEPS",
    "HULL_LAMBDA",
    "LABEL_MIX_LAMBDA",
    "LOCAL_POOL_SIZE",
    "MultiStartHardHullResult",
    "NEW_CLASS_CAP",
    "NODE_CANDIDATE_RNG_NAMESPACE",
    "NORM_ONLY",
    "OBJECTIVE_GLOBAL_SCALE",
    "OTHER",
    "PGD_EPSILON",
    "PROMOTED_SCHEDULER_HORIZON",
    "PROMOTED_SELECTED_EPOCH",
    "PROMOTION_SIDECAR_ENV",
    "SUPPORTED_METHOD_IDS",
    "TUNING_RECORD_COUNT",
    "active_full_k500_heldout_oracle",
    "active_full_k500_promotion",
    "build_anchor_soft_targets",
    "build_two_axis_soft_targets",
    "build_compatible_neighbor_table",
    "classify_compatibility_groups",
    "derive_matched_candidate_seed",
    "derive_d19_paired_random_seed",
    "derive_node_candidate_seed",
    "full_k500_heldout_oracle_scope",
    "install_compatible_runtime",
    "optimize_hard_hull",
    "optimize_multistart_hard_hull",
    "optimize_two_axis_hard_hull",
    "patch_online_trainer_runtime",
    "project_hull",
    "project_two_axis_hull",
    "sample_dirichlet_alpha_one",
    "selected_full_k500_refit_scope",
    "select_shared_candidate_batch",
    "verify_full_k500_promotion",
]
