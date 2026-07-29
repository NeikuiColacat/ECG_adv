"""Exploratory unlabeled-target Mean Teacher bridge for EfficientNet search.

The bridge is enabled only by an explicit method contract.  It cycles through
the clean ref-excluded target-center pool, discards its labels, obtains weak
targets from an EMA teacher, and trains the student on independently generated
AugMix-style strong views.  The existing labelled D19 branch still supplies
VAE-LHAT samples, so this bridge adds target-scale diversity rather than
replacing the thesis mechanism.

Using the final ref-excluded pool in this way is transductive exploration.  It
must not be reported as inductive held-out evidence; a paper candidate needs a
disjoint unlabeled-adaptation/evaluation split before promotion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.augmix import load_augmix_config
from core.corruption import (
    CANONICAL_OPERATORS,
    CORRUPTION_DOMAIN_POINTS,
    CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
    INTERPOLATION_ALIGN_CORNERS,
    INTERPOLATION_MODE,
    OUTPUT_POINTS,
    generate_canonical_corruption,
)
from core.methods import CompiledMethod
from data_preprocess.data_runtime import get_dataloader
from models.checkpoints import extract_state_dict, load_checkpoint_payload
from models.contracts import ModelSpec, validate_model_output
from models.input_adapter import prepare_canonical_model_input
from util.augmentations.torch_operators import apply_operator_batch_prevalidated
from util.random_seed import derive_seed, make_torch_generator


SEARCH_METHOD_ID = "fixed20_search_d19_v1"
TARGETSSL_D19_METHOD_ID = "fixed20_pcgrad_d19_aux_a050_v1"
STRONG_LHAT_AUGMIX_METHOD_ID = "fixed20_pcgrad_strong_lhat_augmix_v1"
GRADIENT_BALANCED_STRONG_LHAT_AUGMIX_METHOD_ID = (
    "fixed20_pcgrad_gradbalanced_strong_lhat_augmix_v1"
)
DIRECT_METHOD_ID = "direct_depth23_fixed20"
HARD_ONLY_PRUNED_METHOD_ID = (
    "fixed20_directsum_vae_lhat_hard_only_pruned_v1"
)
CLEAN_TARGET_SSL_METHOD_ID = "a0_clean_v1"
VAE_LHAT_CLEAN_POLISH_METHOD_ID = "vae_lhat_clean_polish_v1"
VAE_LHAT_CLEAN_DIRECTSUM_METHOD_ID = "vae_lhat_clean_directsum_v1"
ALLOWED_METHOD_IDS = frozenset(
    {
        DIRECT_METHOD_ID,
        SEARCH_METHOD_ID,
        TARGETSSL_D19_METHOD_ID,
        STRONG_LHAT_AUGMIX_METHOD_ID,
        GRADIENT_BALANCED_STRONG_LHAT_AUGMIX_METHOD_ID,
        HARD_ONLY_PRUNED_METHOD_ID,
        CLEAN_TARGET_SSL_METHOD_ID,
        VAE_LHAT_CLEAN_POLISH_METHOD_ID,
        VAE_LHAT_CLEAN_DIRECTSUM_METHOD_ID,
    }
)
CONTRACT_KEY = "unlabeled_teacher"
CLEAN_FAMILY_WEIGHT = 0.5
POLICY = "ema_teacher_clean_to_twochain_augmix_strong_v1"
CLASSIC_AUGMIX_POLICY = "clean_to_classic_augmix3_depth13_simclr_v1"
IDENTITY_PRETRAIN_POLICY = "clean_identity_simclr_vicreg_control_v1"
CLASSIC_AUGMIX_VIEW_MODE = "classic_augmix3_depth13"
PRETRAIN_VIEW_MODES = frozenset(
    {"clean_identity", "twochain_augmix", CLASSIC_AUGMIX_VIEW_MODE}
)
PRETRAIN_POLICY_BY_VIEW_MODE = {
    "clean_identity": IDENTITY_PRETRAIN_POLICY,
    "twochain_augmix": POLICY,
    CLASSIC_AUGMIX_VIEW_MODE: CLASSIC_AUGMIX_POLICY,
}
CLASSIC_AUGMIX_TOPOLOGY = {
    "mixture_width": 3,
    "depth_sampling": "uniform_integer_1_to_3",
    "operator_sampling": "uniform_with_replacement",
    "operator_application_order": "sampled_sequence",
    "operator_domain_sampling_rate_hz": 500,
}
POOL_EVIDENCE_STATUS = {
    "pn2021_all_zero_kept_refexcluded": (
        "transductive_exploration_not_inductive_heldout_evidence"
    ),
    "k500_tune_train": "k500_train_only_internal_screen",
    "k500": "k500_only_full_refit",
}


@dataclass(frozen=True)
class TeacherConfig:
    pool_partition: str
    evidence_status: str
    policy: str
    online_enabled: bool
    labels_consumed: bool
    total_weight: float
    ema_decay: float
    confidence_threshold: float
    temperature: float
    strong_views: int
    dirichlet_alpha: float
    beta_alpha: float
    unlabeled_batches_per_clean: int
    simclr_weight: float
    simclr_temperature: float
    pretrain_epochs: int
    pretrain_steps_per_epoch: int
    pretrain_learning_rate: float
    pretrain_weight_decay: float
    pretrain_clean_anchor: bool
    pretrain_view_mode: str
    pretrain_augmix_topology: Mapping[str, Any] | None
    pretrain_logit_anchor_weight: float
    pretrain_logit_anchor_class_weights: tuple[float, float, float, float, float]
    supervised_logit_anchor_weight: float
    supervised_logit_anchor_class_weights: tuple[float, float, float, float, float]
    pretrain_prototype_weight: float
    pretrain_prototype_temperature: float
    pretrain_prototype_quantile: float
    pretrain_prototype_bank_batches: int
    pretrain_rank_weight: float
    pretrain_rank_margin: float
    pretrain_rank_temperature: float
    pretrain_strong_pseudo_weight: float
    residual_head_mode: str
    residual_head_width: int
    residual_head_dropout: float
    residual_head_scale: float
    residual_head_freeze_base: bool
    pretrain_ssl_objective: str
    pretrain_vicreg_invariance_weight: float
    pretrain_vicreg_variance_weight: float
    pretrain_vicreg_covariance_weight: float
    pretrain_vicreg_mix_weight: float
    pretrain_barlow_offdiagonal_weight: float
    pretrain_ssl_weight: float
    pretrain_source_replay_weight: float
    pretrain_source_batches_per_step: int
    pseudo_reduction: str
    pseudo_quantile: float


@dataclass
class _TeacherState:
    active: bool = False
    config: TeacherConfig | None = None
    teacher: nn.Module | None = None
    supervised_anchor_teacher: nn.Module | None = None
    loader: Any | None = None
    iterator: Iterator[Any] | None = None
    source_loader: Any | None = None
    source_iterator: Iterator[Any] | None = None
    operator_params: Mapping[str, Mapping[str, Any]] | None = None
    generator: torch.Generator | None = None
    seed_config_path: Path | None = None
    center: str | None = None
    model_name: str | None = None
    method_id: str | None = None
    replicate_id: int = 0
    loader_identity: Mapping[str, Any] | None = None
    source_loader_identity: Mapping[str, Any] | None = None
    objective_calls: int = 0
    ema_updates: int = 0
    unlabeled_batches: int = 0
    unlabeled_samples: int = 0
    strong_forwards: int = 0
    accepted_elements: int = 0
    total_elements: int = 0
    accepted_positive_by_class: list[int] = field(default_factory=lambda: [0] * 5)
    accepted_negative_by_class: list[int] = field(default_factory=lambda: [0] * 5)
    loss_sum: float = 0.0
    pseudo_loss_sum: float = 0.0
    simclr_loss_sum: float = 0.0
    pretrain_steps: int = 0
    pretrain_samples: int = 0
    pretrain_loss_sum: float = 0.0
    pretrain_logit_anchor_loss_sum: float = 0.0
    pretrain_prototype_loss_sum: float = 0.0
    pretrain_rank_loss_sum: float = 0.0
    pretrain_strong_pseudo_loss_sum: float = 0.0
    pretrain_ssl_loss_sum: float = 0.0
    pretrain_source_loss_sum: float = 0.0
    pretrain_source_samples: int = 0
    supervised_anchor_calls: int = 0
    supervised_anchor_loss_sum: float = 0.0
    prototype_positive_counts: list[int] = field(default_factory=list)
    prototype_negative_counts: list[int] = field(default_factory=list)
    prototype_positive_thresholds: list[float] = field(default_factory=list)
    prototype_negative_thresholds: list[float] = field(default_factory=list)
    residual_head_parameter_count: int = 0
    residual_head_trainable_parameter_count: int = 0
    confidence_sum: float = 0.0
    hash_digests: list[str] = field(default_factory=list)
    source_hash_digests: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.active = False
        self.config = None
        self.teacher = None
        self.supervised_anchor_teacher = None
        self.loader = None
        self.iterator = None
        self.source_loader = None
        self.source_iterator = None
        self.operator_params = None
        self.generator = None
        self.seed_config_path = None
        self.center = None
        self.model_name = None
        self.method_id = None
        self.replicate_id = 0
        self.loader_identity = None
        self.source_loader_identity = None
        self.objective_calls = 0
        self.ema_updates = 0
        self.unlabeled_batches = 0
        self.unlabeled_samples = 0
        self.strong_forwards = 0
        self.accepted_elements = 0
        self.total_elements = 0
        self.accepted_positive_by_class = [0] * 5
        self.accepted_negative_by_class = [0] * 5
        self.loss_sum = 0.0
        self.pseudo_loss_sum = 0.0
        self.simclr_loss_sum = 0.0
        self.pretrain_steps = 0
        self.pretrain_samples = 0
        self.pretrain_loss_sum = 0.0
        self.pretrain_logit_anchor_loss_sum = 0.0
        self.pretrain_prototype_loss_sum = 0.0
        self.pretrain_rank_loss_sum = 0.0
        self.pretrain_strong_pseudo_loss_sum = 0.0
        self.pretrain_ssl_loss_sum = 0.0
        self.pretrain_source_loss_sum = 0.0
        self.pretrain_source_samples = 0
        self.supervised_anchor_calls = 0
        self.supervised_anchor_loss_sum = 0.0
        self.prototype_positive_counts.clear()
        self.prototype_negative_counts.clear()
        self.prototype_positive_thresholds.clear()
        self.prototype_negative_thresholds.clear()
        self.residual_head_parameter_count = 0
        self.residual_head_trainable_parameter_count = 0
        self.confidence_sum = 0.0
        self.hash_digests.clear()
        self.source_hash_digests.clear()


_STATE = _TeacherState()


def _bounded_float(
    payload: Mapping[str, Any], name: str, *, minimum: float, maximum: float
) -> float:
    value = payload.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{CONTRACT_KEY}.{name} must lie in [{minimum},{maximum}]")
    return float(value)


def _teacher_config(method: CompiledMethod) -> TeacherConfig | None:
    raw = method.contracts.get(CONTRACT_KEY)
    if raw is None:
        return None
    if method.profile_name not in ALLOWED_METHOD_IDS:
        raise ValueError(
            "unlabeled teacher is allowed only on explicit Direct/D19/"
            "strong-LHAT-AugMix methods"
        )
    if not isinstance(raw, Mapping):
        raise ValueError(f"{CONTRACT_KEY} must be a mapping")
    expected = {
        "enabled",
        "policy",
        "evidence_status",
        "pool_partition",
        "labels_consumed",
        "total_weight",
        "ema_decay",
        "confidence_threshold",
        "temperature",
        "strong_views",
        "dirichlet_alpha",
        "beta_alpha",
    }
    allowed = expected | {
        "online_enabled",
        "unlabeled_batches_per_clean",
        "simclr_weight",
        "simclr_temperature",
        "pretrain_epochs",
        "pretrain_steps_per_epoch",
        "pretrain_learning_rate",
        "pretrain_weight_decay",
        "pretrain_clean_anchor",
        "pretrain_view_mode",
        "pretrain_augmix_topology",
        "pretrain_logit_anchor_weight",
        "pretrain_logit_anchor_class_weights",
        "supervised_logit_anchor_weight",
        "supervised_logit_anchor_class_weights",
        "pretrain_prototype_weight",
        "pretrain_prototype_temperature",
        "pretrain_prototype_quantile",
        "pretrain_prototype_bank_batches",
        "pretrain_rank_weight",
        "pretrain_rank_margin",
        "pretrain_rank_temperature",
        "pretrain_strong_pseudo_weight",
        "residual_head_mode",
        "residual_head_width",
        "residual_head_dropout",
        "residual_head_scale",
        "residual_head_freeze_base",
        "pretrain_ssl_objective",
        "pretrain_vicreg_invariance_weight",
        "pretrain_vicreg_variance_weight",
        "pretrain_vicreg_covariance_weight",
        "pretrain_vicreg_mix_weight",
        "pretrain_barlow_offdiagonal_weight",
        "pretrain_ssl_weight",
        "pretrain_source_replay_weight",
        "pretrain_source_batches_per_step",
        "pseudo_reduction",
        "pseudo_quantile",
    }
    if not expected.issubset(raw) or not set(raw).issubset(allowed):
        raise ValueError(f"{CONTRACT_KEY} keys are incomplete or unexpected")
    if raw.get("enabled") is not True:
        return None
    pretrain_view_mode = raw.get("pretrain_view_mode", "twochain_augmix")
    if pretrain_view_mode not in PRETRAIN_VIEW_MODES:
        raise ValueError(
            "unlabeled_teacher.pretrain_view_mode must be "
            "clean_identity/twochain_augmix/classic_augmix3_depth13"
        )
    expected_policy = PRETRAIN_POLICY_BY_VIEW_MODE[str(pretrain_view_mode)]
    if raw.get("policy") != expected_policy:
        raise ValueError("unlabeled teacher policy disclosure is invalid")
    raw_topology = raw.get("pretrain_augmix_topology")
    if pretrain_view_mode == CLASSIC_AUGMIX_VIEW_MODE:
        if not isinstance(raw_topology, Mapping):
            raise ValueError(
                "classic AugMix pretraining requires pretrain_augmix_topology"
            )
        topology = dict(raw_topology)
        if topology != CLASSIC_AUGMIX_TOPOLOGY:
            raise ValueError(
                "classic AugMix topology must be width=3, stochastic depth "
                "1-3, and uniform operator sampling with replacement"
            )
    else:
        if raw_topology is not None:
            raise ValueError(
                "pretrain_augmix_topology is reserved for classic_augmix3_depth13"
            )
        topology = None
    pool_partition = str(raw.get("pool_partition"))
    expected_evidence_status = POOL_EVIDENCE_STATUS.get(pool_partition)
    if expected_evidence_status is None:
        raise ValueError("unlabeled teacher pool partition is invalid")
    if raw.get("evidence_status") != expected_evidence_status:
        raise ValueError(
            "unlabeled teacher evidence status does not match its pool partition"
        )
    online_enabled = raw.get("online_enabled", True)
    if not isinstance(online_enabled, bool):
        raise ValueError("unlabeled_teacher.online_enabled must be boolean")
    labels_consumed = raw.get("labels_consumed")
    if not isinstance(labels_consumed, bool):
        raise ValueError("unlabeled_teacher.labels_consumed must be boolean")
    strong_views = raw.get("strong_views")
    if isinstance(strong_views, bool) or not isinstance(strong_views, int):
        raise ValueError("unlabeled_teacher.strong_views must be an integer")
    if not 1 <= strong_views <= 3:
        raise ValueError("unlabeled_teacher.strong_views must lie in [1,3]")
    batches_per_clean = raw.get("unlabeled_batches_per_clean", 1)
    if (
        isinstance(batches_per_clean, bool)
        or not isinstance(batches_per_clean, int)
        or not 1 <= batches_per_clean <= 4
    ):
        raise ValueError(
            "unlabeled_teacher.unlabeled_batches_per_clean must lie in [1,4]; "
            "eight retained graphs exceed one RTX 4090"
        )
    simclr_weight = _bounded_float(
        raw, "simclr_weight", minimum=0.0, maximum=2.0
    ) if "simclr_weight" in raw else 0.0
    simclr_temperature = _bounded_float(
        raw, "simclr_temperature", minimum=0.05, maximum=2.0
    ) if "simclr_temperature" in raw else 0.2
    if simclr_weight > 0.0 and strong_views < 2:
        raise ValueError("unlabeled SimCLR requires at least two strong views")
    if batches_per_clean * strong_views > 8:
        raise ValueError(
            "unlabeled_batches_per_clean * strong_views must not exceed 8 "
            "on one RTX 4090"
        )
    pretrain_epochs = raw.get("pretrain_epochs", 0)
    pretrain_steps = raw.get("pretrain_steps_per_epoch", 0)
    for name, value, maximum in (
        ("pretrain_epochs", pretrain_epochs, 20),
        # The global K500 scaling study uses one final long-horizon SimCLR
        # boundary point.  The loop and cosine scheduler are total-step based,
        # so 4096 does not change allocation or checkpoint semantics.
        ("pretrain_steps_per_epoch", pretrain_steps, 4096),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"unlabeled_teacher.{name} must lie in [0,{maximum}]")
    clean_anchor = raw.get("pretrain_clean_anchor", True)
    if not isinstance(clean_anchor, bool):
        raise ValueError("unlabeled_teacher.pretrain_clean_anchor must be boolean")
    logit_anchor_weight = (
        _bounded_float(
            raw, "pretrain_logit_anchor_weight", minimum=0.0, maximum=20.0
        )
        if "pretrain_logit_anchor_weight" in raw
        else 0.0
    )
    if logit_anchor_weight > 0.0 and not clean_anchor:
        raise ValueError("pretrain logit anchoring requires pretrain_clean_anchor=true")
    raw_anchor_class_weights = raw.get(
        "pretrain_logit_anchor_class_weights", [1.0] * 5
    )
    if not isinstance(raw_anchor_class_weights, (list, tuple)) or len(
        raw_anchor_class_weights
    ) != 5:
        raise ValueError(
            "pretrain_logit_anchor_class_weights must contain five values"
        )
    anchor_class_weights = tuple(float(value) for value in raw_anchor_class_weights)
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 5.0
        for value in anchor_class_weights
    ) or not math.isclose(sum(anchor_class_weights), 5.0, abs_tol=1.0e-6):
        raise ValueError(
            "pretrain_logit_anchor_class_weights must be finite in [0,5] and sum to 5"
        )
    supervised_anchor_weight = (
        _bounded_float(
            raw, "supervised_logit_anchor_weight", minimum=0.0, maximum=2.0
        )
        if "supervised_logit_anchor_weight" in raw
        else 0.0
    )
    raw_supervised_class_weights = raw.get(
        "supervised_logit_anchor_class_weights", [1.0] * 5
    )
    if not isinstance(raw_supervised_class_weights, (list, tuple)) or len(
        raw_supervised_class_weights
    ) != 5:
        raise ValueError(
            "supervised_logit_anchor_class_weights must contain five values"
        )
    supervised_class_weights = tuple(
        float(value) for value in raw_supervised_class_weights
    )
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 5.0
        for value in supervised_class_weights
    ) or not math.isclose(sum(supervised_class_weights), 5.0, abs_tol=1.0e-6):
        raise ValueError(
            "supervised_logit_anchor_class_weights must be finite in [0,5] and sum to 5"
        )
    pretrain_lr = (
        _bounded_float(
            raw, "pretrain_learning_rate", minimum=1.0e-7, maximum=1.0e-2
        )
        if pretrain_epochs > 0
        else float(raw.get("pretrain_learning_rate", 1.0e-4))
    )
    pretrain_weight_decay = (
        _bounded_float(
            raw, "pretrain_weight_decay", minimum=0.0, maximum=1.0
        )
        if pretrain_epochs > 0
        else float(raw.get("pretrain_weight_decay", 1.0e-4))
    )
    prototype_weight = (
        _bounded_float(raw, "pretrain_prototype_weight", minimum=0.0, maximum=20.0)
        if "pretrain_prototype_weight" in raw
        else 0.0
    )
    prototype_temperature = (
        _bounded_float(
            raw, "pretrain_prototype_temperature", minimum=0.02, maximum=5.0
        )
        if "pretrain_prototype_temperature" in raw
        else 0.2
    )
    prototype_quantile = (
        _bounded_float(raw, "pretrain_prototype_quantile", minimum=0.02, maximum=0.49)
        if "pretrain_prototype_quantile" in raw
        else 0.2
    )
    prototype_bank_batches = raw.get("pretrain_prototype_bank_batches", 0)
    if (
        isinstance(prototype_bank_batches, bool)
        or not isinstance(prototype_bank_batches, int)
        or not 0 <= prototype_bank_batches <= 512
    ):
        raise ValueError(
            "unlabeled_teacher.pretrain_prototype_bank_batches must lie in [0,512]"
        )
    rank_weight = (
        _bounded_float(raw, "pretrain_rank_weight", minimum=0.0, maximum=20.0)
        if "pretrain_rank_weight" in raw
        else 0.0
    )
    rank_margin = (
        _bounded_float(raw, "pretrain_rank_margin", minimum=0.0, maximum=10.0)
        if "pretrain_rank_margin" in raw
        else 0.5
    )
    rank_temperature = (
        _bounded_float(raw, "pretrain_rank_temperature", minimum=0.05, maximum=5.0)
        if "pretrain_rank_temperature" in raw
        else 1.0
    )
    strong_pseudo_weight = (
        _bounded_float(
            raw, "pretrain_strong_pseudo_weight", minimum=0.0, maximum=20.0
        )
        if "pretrain_strong_pseudo_weight" in raw
        else 0.0
    )
    residual_head_mode = raw.get("residual_head_mode", "none")
    if residual_head_mode not in {"none", "feature", "label", "both"}:
        raise ValueError(
            "unlabeled_teacher.residual_head_mode must be none/feature/label/both"
        )
    residual_head_width = raw.get("residual_head_width", 64)
    if (
        isinstance(residual_head_width, bool)
        or not isinstance(residual_head_width, int)
        or not 4 <= residual_head_width <= 1024
    ):
        raise ValueError("unlabeled_teacher.residual_head_width must lie in [4,1024]")
    residual_head_dropout = (
        _bounded_float(raw, "residual_head_dropout", minimum=0.0, maximum=0.8)
        if "residual_head_dropout" in raw
        else 0.1
    )
    residual_head_scale = (
        _bounded_float(raw, "residual_head_scale", minimum=0.001, maximum=2.0)
        if "residual_head_scale" in raw
        else 0.1
    )
    residual_head_freeze_base = raw.get("residual_head_freeze_base", False)
    if not isinstance(residual_head_freeze_base, bool):
        raise ValueError("unlabeled_teacher.residual_head_freeze_base must be boolean")
    pretrain_ssl_objective = raw.get("pretrain_ssl_objective", "simclr")
    if pretrain_ssl_objective not in {
        "simclr",
        "simclr_vicreg",
        "vicreg",
        "barlow_twins",
        "supervised_bce_budget_filler",
    }:
        raise ValueError(
            "unlabeled_teacher.pretrain_ssl_objective must be "
            "simclr/simclr_vicreg/vicreg/barlow_twins/"
            "supervised_bce_budget_filler"
        )
    vicreg_invariance_weight = (
        _bounded_float(
            raw, "pretrain_vicreg_invariance_weight", minimum=0.0, maximum=100.0
        )
        if "pretrain_vicreg_invariance_weight" in raw
        else 25.0
    )
    vicreg_variance_weight = (
        _bounded_float(
            raw, "pretrain_vicreg_variance_weight", minimum=0.0, maximum=100.0
        )
        if "pretrain_vicreg_variance_weight" in raw
        else 25.0
    )
    vicreg_covariance_weight = (
        _bounded_float(
            raw, "pretrain_vicreg_covariance_weight", minimum=0.0, maximum=100.0
        )
        if "pretrain_vicreg_covariance_weight" in raw
        else 1.0
    )
    vicreg_mix_weight = (
        _bounded_float(
            raw, "pretrain_vicreg_mix_weight", minimum=0.0, maximum=1.0
        )
        if "pretrain_vicreg_mix_weight" in raw
        else 0.03
    )
    if pretrain_ssl_objective == "simclr_vicreg" and vicreg_mix_weight <= 0.0:
        raise ValueError(
            "unlabeled_teacher.pretrain_vicreg_mix_weight must be positive "
            "for simclr_vicreg"
        )
    barlow_offdiagonal_weight = (
        _bounded_float(
            raw, "pretrain_barlow_offdiagonal_weight", minimum=0.0001, maximum=1.0
        )
        if "pretrain_barlow_offdiagonal_weight" in raw
        else 0.005
    )
    pretrain_ssl_weight = (
        _bounded_float(raw, "pretrain_ssl_weight", minimum=0.0001, maximum=10.0)
        if "pretrain_ssl_weight" in raw
        else 1.0
    )
    pretrain_source_replay_weight = (
        _bounded_float(
            raw, "pretrain_source_replay_weight", minimum=0.0, maximum=10.0
        )
        if "pretrain_source_replay_weight" in raw
        else 0.0
    )
    pretrain_source_batches_per_step = raw.get(
        "pretrain_source_batches_per_step", 1
    )
    if (
        isinstance(pretrain_source_batches_per_step, bool)
        or not isinstance(pretrain_source_batches_per_step, int)
        or not 1 <= pretrain_source_batches_per_step <= 4
    ):
        raise ValueError(
            "unlabeled_teacher.pretrain_source_batches_per_step must lie in [1,4]"
        )
    if pretrain_source_replay_weight > 0.0 and pretrain_epochs == 0:
        raise ValueError("source semantic replay requires pretrain_epochs > 0")
    pseudo_reduction = raw.get("pseudo_reduction", "element_mean_threshold")
    if pseudo_reduction not in {
        "element_mean_threshold",
        "class_polarity_balanced_quantile",
    }:
        raise ValueError(
            "unlabeled_teacher.pseudo_reduction must be "
            "element_mean_threshold/class_polarity_balanced_quantile"
        )
    pseudo_quantile = (
        _bounded_float(raw, "pseudo_quantile", minimum=0.01, maximum=0.49)
        if "pseudo_quantile" in raw
        else 0.10
    )
    class_conditional_enabled = any(
        value > 0.0
        for value in (prototype_weight, rank_weight, strong_pseudo_weight)
    )
    if class_conditional_enabled and pretrain_epochs == 0:
        raise ValueError("class-conditional target adaptation requires pretrain_epochs > 0")
    if class_conditional_enabled and prototype_bank_batches == 0:
        raise ValueError(
            "class-conditional target adaptation requires a nonzero prototype bank"
        )
    if class_conditional_enabled and not clean_anchor:
        raise ValueError(
            "class-conditional target adaptation requires pretrain_clean_anchor=true"
        )
    budget_filler = pretrain_ssl_objective == "supervised_bce_budget_filler"
    if labels_consumed != budget_filler:
        raise ValueError(
            "labels_consumed must be true only for supervised_bce_budget_filler"
        )
    if budget_filler:
        if online_enabled:
            raise ValueError("supervised budget filler must set online_enabled=false")
        if pretrain_epochs == 0 or pretrain_steps == 0:
            raise ValueError("supervised budget filler requires explicit pretrain steps")
        disabled_weights = {
            "pretrain_logit_anchor_weight": logit_anchor_weight,
            "supervised_logit_anchor_weight": supervised_anchor_weight,
            "pretrain_prototype_weight": prototype_weight,
            "pretrain_rank_weight": rank_weight,
            "pretrain_strong_pseudo_weight": strong_pseudo_weight,
            "pretrain_source_replay_weight": pretrain_source_replay_weight,
            "simclr_weight": simclr_weight,
        }
        if any(value != 0.0 for value in disabled_weights.values()):
            raise ValueError(
                "supervised budget filler forbids teacher/source/auxiliary weights"
            )
        if residual_head_mode != "none":
            raise ValueError("supervised budget filler forbids a residual head")
    elif not online_enabled:
        if pretrain_epochs == 0 or pretrain_steps == 0:
            raise ValueError(
                "pretrain-only target SSL requires explicit pretrain steps"
            )
        if float(raw.get("total_weight", 0.0)) != 0.0:
            raise ValueError(
                "pretrain-only target SSL must set total_weight=0"
            )
    return TeacherConfig(
        pool_partition=pool_partition,
        evidence_status=expected_evidence_status,
        policy=expected_policy,
        online_enabled=online_enabled,
        labels_consumed=labels_consumed,
        total_weight=_bounded_float(
            raw,
            "total_weight",
            minimum=0.0 if budget_filler or not online_enabled else 0.05,
            maximum=0.0 if budget_filler or not online_enabled else 2.0,
        ),
        ema_decay=_bounded_float(raw, "ema_decay", minimum=0.90, maximum=0.9999),
        confidence_threshold=_bounded_float(
            raw, "confidence_threshold", minimum=0.50, maximum=0.999
        ),
        temperature=_bounded_float(raw, "temperature", minimum=0.25, maximum=2.0),
        strong_views=strong_views,
        dirichlet_alpha=_bounded_float(
            raw, "dirichlet_alpha", minimum=0.10, maximum=5.0
        ),
        beta_alpha=_bounded_float(raw, "beta_alpha", minimum=0.10, maximum=5.0),
        unlabeled_batches_per_clean=batches_per_clean,
        simclr_weight=simclr_weight,
        simclr_temperature=simclr_temperature,
        pretrain_epochs=int(pretrain_epochs),
        pretrain_steps_per_epoch=int(pretrain_steps),
        pretrain_learning_rate=pretrain_lr,
        pretrain_weight_decay=pretrain_weight_decay,
        pretrain_clean_anchor=clean_anchor,
        pretrain_view_mode=str(pretrain_view_mode),
        pretrain_augmix_topology=topology,
        pretrain_logit_anchor_weight=logit_anchor_weight,
        pretrain_logit_anchor_class_weights=anchor_class_weights,
        supervised_logit_anchor_weight=supervised_anchor_weight,
        supervised_logit_anchor_class_weights=supervised_class_weights,
        pretrain_prototype_weight=prototype_weight,
        pretrain_prototype_temperature=prototype_temperature,
        pretrain_prototype_quantile=prototype_quantile,
        pretrain_prototype_bank_batches=int(prototype_bank_batches),
        pretrain_rank_weight=rank_weight,
        pretrain_rank_margin=rank_margin,
        pretrain_rank_temperature=rank_temperature,
        pretrain_strong_pseudo_weight=strong_pseudo_weight,
        residual_head_mode=str(residual_head_mode),
        residual_head_width=int(residual_head_width),
        residual_head_dropout=residual_head_dropout,
        residual_head_scale=residual_head_scale,
        residual_head_freeze_base=residual_head_freeze_base,
        pretrain_ssl_objective=str(pretrain_ssl_objective),
        pretrain_vicreg_invariance_weight=vicreg_invariance_weight,
        pretrain_vicreg_variance_weight=vicreg_variance_weight,
        pretrain_vicreg_covariance_weight=vicreg_covariance_weight,
        pretrain_vicreg_mix_weight=vicreg_mix_weight,
        pretrain_barlow_offdiagonal_weight=barlow_offdiagonal_weight,
        pretrain_ssl_weight=pretrain_ssl_weight,
        pretrain_source_replay_weight=pretrain_source_replay_weight,
        pretrain_source_batches_per_step=int(pretrain_source_batches_per_step),
        pseudo_reduction=str(pseudo_reduction),
        pseudo_quantile=pseudo_quantile,
    )


def _is_clean_objective(names: Any) -> bool:
    selected = tuple(str(value) for value in (names or ()))
    return bool(selected) and selected[0] == "clean_bce" and (
        "corrupted_bce" not in selected
    )


def _next_batch() -> Mapping[str, Any]:
    if _STATE.loader is None:
        raise RuntimeError("unlabeled loader is unavailable")
    if _STATE.iterator is None:
        _STATE.iterator = iter(_STATE.loader)
    try:
        batch = next(_STATE.iterator)
    except StopIteration:
        _STATE.iterator = iter(_STATE.loader)
        batch = next(_STATE.iterator)
    if not isinstance(batch, Mapping):
        raise TypeError("unlabeled target batch must be a mapping")
    return batch


def _next_source_batch() -> Mapping[str, Any]:
    if _STATE.source_loader is None:
        raise RuntimeError("pretrain source replay loader is unavailable")
    if _STATE.source_iterator is None:
        _STATE.source_iterator = iter(_STATE.source_loader)
    try:
        batch = next(_STATE.source_iterator)
    except StopIteration:
        _STATE.source_iterator = iter(_STATE.source_loader)
        batch = next(_STATE.source_iterator)
    if not isinstance(batch, Mapping):
        raise TypeError("pretrain source replay batch must be a mapping")
    return batch


def _batch_digest(batch: Mapping[str, Any], batch_size: int) -> str:
    raw = batch.get("hash_id")
    if not isinstance(raw, (list, tuple)) or len(raw) != batch_size:
        raise ValueError("batch hash IDs do not align to the batch")
    return hashlib.sha256("\n".join(str(value) for value in raw).encode()).hexdigest()


def _disable_bn_running_stats(model: nn.Module) -> tuple[tuple[nn.Module, bool], ...]:
    captured = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            captured.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    return tuple(captured)


def _restore_bn_running_stats(captured: Sequence[tuple[nn.Module, bool]]) -> None:
    for module, value in captured:
        module.track_running_stats = value


def _capture_rng(model: nn.Module) -> tuple[torch.Tensor, torch.device, torch.Tensor | None]:
    device = next(model.parameters()).device
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.random.get_rng_state(), device, cuda


def _restore_rng(state: tuple[torch.Tensor, torch.device, torch.Tensor | None]) -> None:
    cpu, device, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


class _ResidualTargetHead(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        mode: str,
        width: int,
        dropout: float,
        scale: float,
        freeze_base: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.mode = mode
        self.base = base
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
        self.feature_adapter = (
            nn.Sequential(
                nn.LayerNorm(self.in_features),
                nn.Dropout(dropout),
                nn.Linear(self.in_features, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, self.out_features),
            )
            if mode in {"feature", "both"}
            else None
        )
        self.label_adapter = (
            nn.Sequential(
                nn.LayerNorm(self.out_features),
                nn.Linear(self.out_features, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, self.out_features),
            )
            if mode in {"label", "both"}
            else None
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(scale)))
        for adapter in (self.feature_adapter, self.label_adapter):
            if adapter is None:
                continue
            output = adapter[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("residual adapter output must be linear")
            nn.init.normal_(output.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = self.base(features)
        residual = torch.zeros_like(base_logits)
        if self.feature_adapter is not None:
            residual = residual + self.feature_adapter(features)
        if self.label_adapter is not None:
            residual = residual + self.label_adapter(base_logits)
        return base_logits + self.residual_scale * residual


def _classifier_head(model: nn.Module) -> nn.Module:
    dense = getattr(model, "dense", None)
    if isinstance(dense, (nn.Linear, _ResidualTargetHead)):
        return dense
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and len(classifier):
        head = classifier[-1]
        if isinstance(head, (nn.Linear, _ResidualTargetHead)):
            return head
    raise TypeError("unlabeled SimCLR requires the managed linear classifier head")


def _install_residual_target_head(model: nn.Module, config: TeacherConfig) -> None:
    if config.residual_head_mode == "none":
        return
    current = _classifier_head(model)
    if isinstance(current, _ResidualTargetHead):
        raise RuntimeError("residual target head was already installed")
    if not isinstance(current, nn.Linear):
        raise TypeError("residual target head requires a linear source classifier")
    if _STATE.seed_config_path is None:
        raise RuntimeError("residual target head seed config is unavailable")
    device = current.weight.device
    seed = derive_seed(
        "residual_target_head_v1",
        _STATE.center,
        _STATE.model_name,
        _STATE.replicate_id,
        config.residual_head_mode,
        config.residual_head_width,
        config.residual_head_dropout,
        config.residual_head_scale,
        config.residual_head_freeze_base,
        config_path=_STATE.seed_config_path,
    )
    cuda_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=cuda_devices if device.type == "cuda" else []):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        replacement = _ResidualTargetHead(
            current,
            mode=config.residual_head_mode,
            width=config.residual_head_width,
            dropout=config.residual_head_dropout,
            scale=config.residual_head_scale,
            freeze_base=config.residual_head_freeze_base,
        ).to(device=device, dtype=current.weight.dtype)
    if getattr(model, "dense", None) is current:
        model.dense = replacement
    else:
        classifier = getattr(model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or classifier[-1] is not current:
            raise RuntimeError("managed classifier ownership changed")
        classifier[-1] = replacement
    _STATE.residual_head_parameter_count = sum(
        parameter.numel() for parameter in replacement.parameters()
    )
    _STATE.residual_head_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in replacement.parameters()
        if parameter.requires_grad
    )


def _install_checkpoint_residual_head(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> bool:
    """Recreate an exploratory residual head before strict checkpoint loading."""

    current = _classifier_head(model)
    if isinstance(current, _ResidualTargetHead):
        raise RuntimeError("evaluation model already owns a residual target head")
    if not isinstance(current, nn.Linear):
        raise TypeError("checkpoint residual head requires a linear classifier")
    if getattr(model, "dense", None) is current:
        prefix = "dense."
        owner = "dense"
    else:
        classifier = getattr(model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or classifier[-1] is not current:
            raise RuntimeError("evaluation classifier ownership changed")
        prefix = f"classifier.{len(classifier) - 1}."
        owner = "classifier"
    residual_key = f"{prefix}residual_scale"
    residual_keys = sorted(key for key in state if key.endswith("residual_scale"))
    if not residual_keys:
        return False
    if residual_keys != [residual_key]:
        raise ValueError(
            "checkpoint residual-head path does not match the managed classifier: "
            f"expected {residual_key!r}, found {residual_keys!r}"
        )
    has_feature = any(key.startswith(f"{prefix}feature_adapter.") for key in state)
    has_label = any(key.startswith(f"{prefix}label_adapter.") for key in state)
    if has_feature and has_label:
        mode = "both"
    elif has_feature:
        mode = "feature"
    elif has_label:
        mode = "label"
    else:
        raise ValueError("residual checkpoint contains no feature or label adapter")
    width_key = (
        f"{prefix}feature_adapter.2.weight"
        if has_feature
        else f"{prefix}label_adapter.1.weight"
    )
    width_tensor = state.get(width_key)
    if not isinstance(width_tensor, torch.Tensor) or width_tensor.ndim != 2:
        raise ValueError(f"residual checkpoint is missing width tensor {width_key!r}")
    width = int(width_tensor.shape[0])
    if width <= 0:
        raise ValueError("residual checkpoint width must be positive")
    cpu_rng = torch.random.get_rng_state()
    try:
        replacement = _ResidualTargetHead(
            current,
            mode=mode,
            width=width,
            # Dropout has no numerical effect under evaluator ``model.eval()``;
            # all learned tensors and residual_scale are loaded strictly below.
            dropout=0.0,
            scale=float(state[residual_key].detach().cpu()),
            freeze_base=False,
        ).to(device=current.weight.device, dtype=current.weight.dtype)
    finally:
        torch.random.set_rng_state(cpu_rng)
    if owner == "dense":
        model.dense = replacement
    else:
        classifier = getattr(model, "classifier")
        classifier[-1] = replacement
    return True


@contextmanager
def patch_pn2021_evaluation_runtime(
    checkpoint_path: str | Path,
) -> Iterator[None]:
    """Teach the sandbox evaluator to reconstruct residual-head checkpoints."""

    from boot_scripts import evaluate_pn2021

    _, payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    state = extract_state_dict(payload)
    original_build_model = evaluate_pn2021.build_model

    def patched_build_model(*args: Any, **kwargs: Any) -> nn.Module:
        model = original_build_model(*args, **kwargs)
        _install_checkpoint_residual_head(model, state)
        return model

    evaluate_pn2021.build_model = patched_build_model
    try:
        yield
    finally:
        evaluate_pn2021.build_model = original_build_model


def _paired_simclr_loss(
    first: torch.Tensor, second: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("unlabeled SimCLR features must be paired [B,D] tensors")
    batch_size = int(first.shape[0])
    features = F.normalize(torch.cat((first.float(), second.float()), dim=0), dim=1)
    similarities = features @ features.transpose(0, 1) / float(temperature)
    diagonal = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
    similarities = similarities.masked_fill(diagonal, float("-inf"))
    positive_indices = torch.cat(
        (
            torch.arange(batch_size, 2 * batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device),
        )
    )
    row = torch.arange(2 * batch_size, device=features.device)
    positives = similarities[row, positive_indices]
    loss = (torch.logsumexp(similarities, dim=1) - positives).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("unlabeled SimCLR loss became NaN or Inf")
    return loss


def _weighted_logit_anchor_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    class_weights: Sequence[float],
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("logit anchor expects aligned [batch,class] logits")
    elements = F.binary_cross_entropy_with_logits(
        student_logits.float(), torch.sigmoid(teacher_logits.float()), reduction="none"
    )
    weights = torch.as_tensor(
        class_weights, dtype=elements.dtype, device=elements.device
    )
    if tuple(weights.shape) != (int(elements.shape[1]),):
        raise ValueError("logit anchor class weights do not match the output width")
    loss = (elements * weights.unsqueeze(0)).sum(dim=1).div(weights.sum()).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("logit anchor loss became NaN or Inf")
    return loss


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("off-diagonal extraction expects a square matrix")
    size = int(matrix.shape[0])
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def _vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> torch.Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("VICReg features must be paired [B,D] tensors")
    first = first.float()
    second = second.float()
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1.0e-4)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1.0e-4)
    variance = 0.5 * (
        F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
    )
    denominator = max(int(first.shape[0]) - 1, 1)
    first_covariance = first_centered.transpose(0, 1) @ first_centered / denominator
    second_covariance = second_centered.transpose(0, 1) @ second_centered / denominator
    dimension = int(first.shape[1])
    covariance = (
        _off_diagonal(first_covariance).square().sum()
        + _off_diagonal(second_covariance).square().sum()
    ) / (2.0 * dimension)
    loss = (
        float(invariance_weight) * invariance
        + float(variance_weight) * variance
        + float(covariance_weight) * covariance
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("VICReg loss became NaN or Inf")
    return loss


def _barlow_twins_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    offdiagonal_weight: float,
) -> torch.Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("Barlow Twins features must be paired [B,D] tensors")
    first = first.float()
    second = second.float()
    first = (first - first.mean(dim=0)) / first.std(
        dim=0, unbiased=False
    ).clamp_min(1.0e-4)
    second = (second - second.mean(dim=0)) / second.std(
        dim=0, unbiased=False
    ).clamp_min(1.0e-4)
    correlation = first.transpose(0, 1) @ second / float(first.shape[0])
    diagonal = torch.diagonal(correlation).add(-1.0).square().sum()
    offdiagonal = _off_diagonal(correlation).square().sum()
    loss = diagonal + float(offdiagonal_weight) * offdiagonal
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Barlow Twins loss became NaN or Inf")
    return loss


@dataclass(frozen=True)
class _PrototypeBank:
    positive: torch.Tensor
    negative: torch.Tensor
    positive_thresholds: torch.Tensor
    negative_thresholds: torch.Tensor


def _source_features_and_logits(
    model: nn.Module,
    head: nn.Module,
    model_input: torch.Tensor,
    spec: ModelSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        if len(arguments) != 1 or arguments[0].ndim != 2:
            raise RuntimeError("source prototype feature capture is invalid")
        captured.append(arguments[0])

    handle = head.register_forward_pre_hook(capture)
    try:
        logits = validate_model_output(
            model(model_input),
            spec,
            batch_size=int(model_input.shape[0]),
            check_finite=True,
        ).float()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("source prototype expected exactly one feature tensor")
    return captured[0].float(), logits


def _build_prototype_bank(
    source_teacher: nn.Module,
    spec: ModelSpec,
    config: TeacherConfig,
    *,
    normalization_epsilon: float,
) -> _PrototypeBank | None:
    enabled = any(
        value > 0.0
        for value in (
            config.pretrain_prototype_weight,
            config.pretrain_rank_weight,
            config.pretrain_strong_pseudo_weight,
        )
    )
    if not enabled:
        return None
    head = _classifier_head(source_teacher)
    device = next(source_teacher.parameters()).device
    features: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    _STATE.iterator = None
    with torch.no_grad():
        for _ in range(config.pretrain_prototype_bank_batches):
            batch = _next_batch()
            raw = batch.get("waveform")
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape[1:]) != (1000, 12):
                raise ValueError("prototype bank waveform must have shape (B,1000,12)")
            raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
            model_input = prepare_canonical_model_input(
                raw, spec, epsilon=normalization_epsilon
            )
            feature, logits = _source_features_and_logits(
                source_teacher, head, model_input, spec
            )
            features.append(F.normalize(feature.float(), dim=1).cpu())
            probabilities.append(torch.sigmoid(logits).cpu())
    _STATE.iterator = None
    feature_bank = torch.cat(features, dim=0)
    probability_bank = torch.cat(probabilities, dim=0)
    q = config.pretrain_prototype_quantile
    negative_thresholds = torch.quantile(probability_bank, q, dim=0)
    positive_thresholds = torch.quantile(probability_bank, 1.0 - q, dim=0)
    positive = []
    negative = []
    positive_counts = []
    negative_counts = []
    for class_index in range(probability_bank.shape[1]):
        pos_mask = probability_bank[:, class_index] >= positive_thresholds[class_index]
        neg_mask = probability_bank[:, class_index] <= negative_thresholds[class_index]
        positive_counts.append(int(pos_mask.sum()))
        negative_counts.append(int(neg_mask.sum()))
        positive.append(F.normalize(feature_bank[pos_mask].mean(dim=0), dim=0))
        negative.append(F.normalize(feature_bank[neg_mask].mean(dim=0), dim=0))
    _STATE.prototype_positive_counts[:] = positive_counts
    _STATE.prototype_negative_counts[:] = negative_counts
    _STATE.prototype_positive_thresholds[:] = [
        float(value) for value in positive_thresholds
    ]
    _STATE.prototype_negative_thresholds[:] = [
        float(value) for value in negative_thresholds
    ]
    return _PrototypeBank(
        positive=torch.stack(positive).to(device=device),
        negative=torch.stack(negative).to(device=device),
        positive_thresholds=positive_thresholds.to(device=device),
        negative_thresholds=negative_thresholds.to(device=device),
    )


def _prototype_classification_loss(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    bank: _PrototypeBank,
    *,
    temperature: float,
) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=1)
    positive_similarity = normalized @ bank.positive.transpose(0, 1)
    negative_similarity = normalized @ bank.negative.transpose(0, 1)
    logits = (positive_similarity - negative_similarity) / float(temperature)
    positive_mask = probabilities >= bank.positive_thresholds
    negative_mask = probabilities <= bank.negative_thresholds
    losses = []
    for class_index in range(logits.shape[1]):
        mask = positive_mask[:, class_index] | negative_mask[:, class_index]
        if not bool(mask.any().item()):
            continue
        target = positive_mask[mask, class_index].to(dtype=torch.float32)
        losses.append(
            F.binary_cross_entropy_with_logits(logits[mask, class_index], target)
        )
    if not losses:
        return logits.sum() * 0.0
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("prototype classification loss became NaN or Inf")
    return loss


def _class_balanced_rank_loss(
    student_logits: torch.Tensor,
    probabilities: torch.Tensor,
    bank: _PrototypeBank,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for class_index in range(student_logits.shape[1]):
        positive = student_logits[
            probabilities[:, class_index] >= bank.positive_thresholds[class_index],
            class_index,
        ].float()
        negative = student_logits[
            probabilities[:, class_index] <= bank.negative_thresholds[class_index],
            class_index,
        ].float()
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        difference = (positive[:, None] - negative[None, :]) / float(temperature)
        losses.append(F.softplus(float(margin) - difference).mean())
    if not losses:
        return student_logits.sum() * 0.0
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("class-balanced ranking loss became NaN or Inf")
    return loss


def _masked_strong_pseudo_loss(
    student_logits: torch.Tensor,
    probabilities: torch.Tensor,
    bank: _PrototypeBank,
) -> torch.Tensor:
    mask = (probabilities >= bank.positive_thresholds) | (
        probabilities <= bank.negative_thresholds
    )
    if not bool(mask.any().item()):
        return student_logits.sum() * 0.0
    elementwise = F.binary_cross_entropy_with_logits(
        student_logits.float(), probabilities, reduction="none"
    )
    loss = elementwise[mask].mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("strong pseudo-label loss became NaN or Inf")
    return loss


def _update_teacher(model: nn.Module, decay: float) -> nn.Module:
    if _STATE.teacher is None:
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        _STATE.teacher = teacher
        return teacher
    teacher = _STATE.teacher
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), model.parameters(), strict=True
        ):
            teacher_parameter.mul_(decay).add_(
                student_parameter.detach(), alpha=1.0 - decay
            )
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(), model.buffers(), strict=True
        ):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(decay).add_(
                    student_buffer.detach(), alpha=1.0 - decay
                )
            else:
                teacher_buffer.copy_(student_buffer)
    teacher.eval()
    _STATE.ema_updates += 1
    return teacher


def _beta_sample(
    batch_size: int, alpha: float, *, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    concentration = torch.full((batch_size, 2), alpha, device=device)
    gamma = torch._standard_gamma(concentration, generator=generator)
    return (gamma[:, 0] / gamma.sum(dim=1).clamp_min(1.0e-12)).view(-1, 1, 1)


def _dirichlet_sample(
    batch_size: int,
    width: int,
    alpha: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    concentration = torch.full((batch_size, width), alpha, device=device)
    gamma = torch._standard_gamma(concentration, generator=generator)
    return gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def _ensure_augmix_generator(
    raw: torch.Tensor,
    config: TeacherConfig,
) -> torch.Generator:
    if _STATE.operator_params is None:
        raise RuntimeError("unlabeled AugMix resources are unavailable")
    if _STATE.generator is None:
        if _STATE.seed_config_path is None or not _STATE.center or not _STATE.model_name:
            raise RuntimeError("unlabeled AugMix seed identity is unavailable")
        _STATE.generator = make_torch_generator(
            raw.device,
            "unlabeled_teacher_augmix_v1",
            _STATE.center,
            _STATE.model_name,
            _STATE.replicate_id,
            config_path=_STATE.seed_config_path,
        )
    if torch.device(_STATE.generator.device).type != raw.device.type:
        raise RuntimeError("unlabeled AugMix generator device drifted")
    return _STATE.generator


def _twochain_strong_view(
    raw: torch.Tensor,
    config: TeacherConfig,
) -> torch.Tensor:
    generator = _ensure_augmix_generator(raw, config)
    first = generate_canonical_corruption(
        raw,
        operator_params=_STATE.operator_params,
        generator=generator,
        _input_prevalidated=True,
    ).waveform_raw_100hz
    second = generate_canonical_corruption(
        raw,
        operator_params=_STATE.operator_params,
        generator=generator,
        _input_prevalidated=True,
    ).waveform_raw_100hz
    weights = _dirichlet_sample(
        int(raw.shape[0]),
        2,
        config.dirichlet_alpha,
        device=raw.device,
        generator=generator,
    )
    mixed = (
        weights[:, 0].view(-1, 1, 1) * first
        + weights[:, 1].view(-1, 1, 1) * second
    )
    strength = _beta_sample(
        int(raw.shape[0]),
        config.beta_alpha,
        device=raw.device,
        generator=generator,
    )
    return ((1.0 - strength) * raw + strength * mixed).contiguous()


def _resample_btc(signal: torch.Tensor, *, points: int) -> torch.Tensor:
    return F.interpolate(
        signal.transpose(1, 2),
        size=int(points),
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()


def _classic_augmix_chain(
    raw: torch.Tensor,
    config: TeacherConfig,
) -> torch.Tensor:
    """Generate one classic AugMix chain with per-record depth and op draws."""

    generator = _ensure_augmix_generator(raw, config)
    if _STATE.operator_params is None:  # narrowed by _ensure_augmix_generator
        raise RuntimeError("unlabeled AugMix resources are unavailable")
    batch_size = int(raw.shape[0])
    device = raw.device
    depths = torch.randint(
        1,
        4,
        (batch_size,),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    operator_indices = torch.randint(
        0,
        len(CANONICAL_OPERATORS),
        (batch_size, 3),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    waveform = _resample_btc(
        raw.to(dtype=torch.float32),
        points=CORRUPTION_DOMAIN_POINTS,
    )
    for step in range(3):
        active = depths > step
        for operator_index, operator in enumerate(CANONICAL_OPERATORS):
            selected = torch.nonzero(
                active & (operator_indices[:, step] == operator_index),
                as_tuple=False,
            ).flatten()
            if selected.numel() == 0:
                continue
            subset = waveform.index_select(0, selected)
            augmented = apply_operator_batch_prevalidated(
                operator,
                subset,
                params=dict(_STATE.operator_params[operator]),
                sampling_rate_hz=CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
                rng=generator,
            )
            waveform = waveform.index_copy(0, selected, augmented)
    output = _resample_btc(waveform, points=OUTPUT_POINTS)
    return torch.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()


def _classic_augmix3_depth13_view(
    raw: torch.Tensor,
    config: TeacherConfig,
) -> torch.Tensor:
    generator = _ensure_augmix_generator(raw, config)
    chains = tuple(_classic_augmix_chain(raw, config) for _ in range(3))
    weights = _dirichlet_sample(
        int(raw.shape[0]),
        3,
        config.dirichlet_alpha,
        device=raw.device,
        generator=generator,
    )
    mixed = sum(
        weights[:, index].view(-1, 1, 1) * chain
        for index, chain in enumerate(chains)
    )
    strength = _beta_sample(
        int(raw.shape[0]),
        config.beta_alpha,
        device=raw.device,
        generator=generator,
    )
    return ((1.0 - strength) * raw + strength * mixed).contiguous()


def _strong_view(raw: torch.Tensor, config: TeacherConfig) -> torch.Tensor:
    if config.pretrain_view_mode == "twochain_augmix":
        return _twochain_strong_view(raw, config)
    if config.pretrain_view_mode == CLASSIC_AUGMIX_VIEW_MODE:
        return _classic_augmix3_depth13_view(raw, config)
    raise AssertionError(config.pretrain_view_mode)


def _pretrain_view(raw: torch.Tensor, config: TeacherConfig) -> torch.Tensor:
    """Build one Stage-1 view without changing the shared batch/RNG identity.

    ``clean_identity`` is the strict no-AugMix control. It intentionally keeps
    the same SimCLR/VICReg objective, optimizer and number of updates as the
    ``twochain_augmix`` arm; only the view generator differs.
    """

    if config.pretrain_view_mode == "clean_identity":
        return raw
    if config.pretrain_view_mode in {
        "twochain_augmix",
        CLASSIC_AUGMIX_VIEW_MODE,
    }:
        return _strong_view(raw, config)
    raise AssertionError(config.pretrain_view_mode)


def _pretrain_source_semantic_loss(
    model: nn.Module,
    spec: ModelSpec,
    config: TeacherConfig,
    *,
    normalization_epsilon: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    losses = []
    for _ in range(config.pretrain_source_batches_per_step):
        batch = _next_source_batch()
        raw = batch.get("waveform")
        targets = batch.get("label")
        if not isinstance(raw, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("source semantic replay requires waveform and label tensors")
        if tuple(raw.shape[1:]) != (1000, 12) or targets.shape != (raw.shape[0], 5):
            raise ValueError("source semantic replay expects (B,1000,12) and (B,5)")
        raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        if not bool(torch.isfinite(raw).all().item()) or not bool(
            torch.isfinite(targets).all().item()
        ):
            raise FloatingPointError("source semantic replay contains NaN or Inf")
        rng_state = _capture_rng(model)
        bn_state = _disable_bn_running_stats(model)
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = validate_model_output(
                    model(
                        prepare_canonical_model_input(
                            raw, spec, epsilon=normalization_epsilon
                        )
                    ),
                    spec,
                    batch_size=int(raw.shape[0]),
                    check_finite=True,
                )
                losses.append(
                    F.binary_cross_entropy_with_logits(logits.float(), targets)
                )
        finally:
            _restore_bn_running_stats(bn_state)
            _restore_rng(rng_state)
        _STATE.pretrain_source_samples += int(raw.shape[0])
        _STATE.source_hash_digests.append(
            _batch_digest(batch, int(raw.shape[0]))
        )
    if not losses:
        raise RuntimeError("source semantic replay produced no losses")
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("source semantic replay loss became NaN or Inf")
    return loss


def _pretrain_supervised_budget_filler(
    model: nn.Module,
    spec: ModelSpec,
    config: TeacherConfig,
    *,
    normalization_epsilon: float,
) -> None:
    """Spend the matched 128 K400-only optimizer steps without enabling T."""

    if _STATE.loader is None or _STATE.seed_config_path is None:
        raise RuntimeError("supervised budget filler resources are unavailable")
    device = next(model.parameters()).device
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable:
        raise ValueError("supervised budget filler found no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.pretrain_learning_rate,
        weight_decay=config.pretrain_weight_decay,
    )
    total_steps = config.pretrain_epochs * config.pretrain_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=config.pretrain_learning_rate * 0.01,
    )
    amp_enabled = device.type == "cuda"
    model.train()
    for _epoch in range(config.pretrain_epochs):
        for _step in range(config.pretrain_steps_per_epoch):
            batch = _next_batch()
            raw = batch.get("waveform")
            targets = batch.get("label")
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape[1:]) != (1000, 12):
                raise ValueError(
                    "supervised budget filler waveform must have shape (B,1000,12)"
                )
            if not isinstance(targets, torch.Tensor) or tuple(targets.shape[1:]) != (5,):
                raise ValueError(
                    "supervised budget filler labels must have shape (B,5)"
                )
            raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if not bool(torch.isfinite(raw).all().item()) or not bool(
                torch.isfinite(targets).all().item()
            ):
                raise FloatingPointError(
                    "supervised budget filler batch contains NaN or Inf"
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = validate_model_output(
                    model(
                        prepare_canonical_model_input(
                            raw,
                            spec,
                            epsilon=normalization_epsilon,
                        )
                    ),
                    spec,
                    batch_size=int(raw.shape[0]),
                    check_finite=True,
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), targets
                )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "supervised budget filler loss became NaN or Inf"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            scheduler.step()
            batch_size = int(raw.shape[0])
            loss_value = float(loss.detach().cpu())
            _STATE.pretrain_steps += 1
            _STATE.pretrain_samples += batch_size
            _STATE.pretrain_loss_sum += loss_value
            _STATE.pretrain_ssl_loss_sum += loss_value
            _STATE.hash_digests.append(_batch_digest(batch, batch_size))
    _STATE.iterator = None


def _pretrain_projector_seed(config: TeacherConfig) -> int:
    """Keep projector initialization paired across source-replay ablations."""

    if (
        _STATE.center is None
        or _STATE.model_name is None
        or _STATE.seed_config_path is None
    ):
        raise RuntimeError("target SSL projector seed identity is unavailable")
    return derive_seed(
        "unlabeled_target_simclr_pretrain_v1",
        _STATE.center,
        _STATE.model_name,
        _STATE.replicate_id,
        config.pool_partition,
        config.pretrain_epochs,
        config.pretrain_steps_per_epoch,
        config.pretrain_learning_rate,
        config.pretrain_weight_decay,
        config.pretrain_clean_anchor,
        config.pretrain_logit_anchor_weight,
        config.pretrain_logit_anchor_class_weights,
        config.pretrain_prototype_weight,
        config.pretrain_prototype_temperature,
        config.pretrain_prototype_quantile,
        config.pretrain_prototype_bank_batches,
        config.pretrain_rank_weight,
        config.pretrain_rank_margin,
        config.pretrain_rank_temperature,
        config.pretrain_strong_pseudo_weight,
        config.pretrain_ssl_objective,
        config.pretrain_vicreg_invariance_weight,
        config.pretrain_vicreg_variance_weight,
        config.pretrain_vicreg_covariance_weight,
        config.pretrain_vicreg_mix_weight,
        config.pretrain_barlow_offdiagonal_weight,
        config.pretrain_ssl_weight,
        # Freeze the source-replay ablation seed to the historical replay-off
        # identity. Source replay changes the loss only, not projector init.
        0.0,
        config.pretrain_source_batches_per_step,
        config_path=_STATE.seed_config_path,
    )


def _pretrain_target_representation(
    model: nn.Module,
    spec: ModelSpec,
    *,
    normalization_epsilon: float,
) -> None:
    """Run an explicit target-domain SimCLR phase before supervised tuning.

    The temporary projection head is discarded.  Only the managed classifier
    backbone (including its normalization statistics) is carried into the
    labelled fixed20/LHAT stage; the five-class head remains exactly at the
    locked PTB-XL source checkpoint initialization.  Both managed
    EfficientNet1DV2 and ECGFounder expose their penultimate features through
    the input of a linear five-class head, so the same feature-capture contract
    applies to both backbones.
    """

    config = _STATE.config
    if config is None or config.pretrain_epochs == 0:
        return
    if config.pretrain_ssl_objective == "supervised_bce_budget_filler":
        _pretrain_supervised_budget_filler(
            model,
            spec,
            config,
            normalization_epsilon=normalization_epsilon,
        )
        return
    if _STATE.loader is None or _STATE.seed_config_path is None:
        raise RuntimeError("target SSL pretraining resources are unavailable")
    device = next(model.parameters()).device
    head = _classifier_head(model)
    feature_dim = int(head.in_features)
    projection_dim = min(128, feature_dim)
    source_teacher = None
    source_guidance_enabled = any(
        value > 0.0
        for value in (
            config.pretrain_logit_anchor_weight,
            config.pretrain_prototype_weight,
            config.pretrain_rank_weight,
            config.pretrain_strong_pseudo_weight,
        )
    )
    if source_guidance_enabled:
        source_teacher = copy.deepcopy(model).to(device).eval()
        for parameter in source_teacher.parameters():
            parameter.requires_grad_(False)
    prototype_bank = (
        _build_prototype_bank(
            source_teacher,
            spec,
            config,
            normalization_epsilon=normalization_epsilon,
        )
        if source_teacher is not None
        else None
    )
    seed = _pretrain_projector_seed(config)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices if device.type == "cuda" else []):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        ).to(device)

    original_requires_grad = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in {id(p) for p in head.parameters()}
    ]
    if not trainable:
        raise ValueError("target SSL pretraining found no trainable backbone parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable},
            {"params": list(projector.parameters())},
        ],
        lr=config.pretrain_learning_rate,
        weight_decay=config.pretrain_weight_decay,
    )
    steps_per_epoch = (
        config.pretrain_steps_per_epoch
        if config.pretrain_steps_per_epoch > 0
        else len(_STATE.loader)
    )
    total_steps = config.pretrain_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
        eta_min=config.pretrain_learning_rate * 0.01,
    )
    amp_enabled = device.type == "cuda"
    model.train()
    projector.train()
    try:
        for _epoch in range(config.pretrain_epochs):
            for _step in range(steps_per_epoch):
                batch = _next_batch()
                raw = batch.get("waveform")
                if not isinstance(raw, torch.Tensor) or tuple(raw.shape[1:]) != (1000, 12):
                    raise ValueError("target SSL waveform must have shape (B,1000,12)")
                raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
                if not bool(torch.isfinite(raw).all().item()):
                    raise FloatingPointError("target SSL waveform contains NaN or Inf")
                first = (
                    raw
                    if config.pretrain_clean_anchor
                    else _pretrain_view(raw, config)
                )
                second = _pretrain_view(raw, config)
                source_loss = (
                    _pretrain_source_semantic_loss(
                        model,
                        spec,
                        config,
                        normalization_epsilon=normalization_epsilon,
                    )
                    if config.pretrain_source_replay_weight > 0.0
                    else raw.sum() * 0.0
                )
                captured: list[torch.Tensor] = []

                def capture(
                    _module: nn.Module, arguments: tuple[torch.Tensor, ...]
                ) -> None:
                    if len(arguments) != 1 or arguments[0].ndim != 2:
                        raise RuntimeError("target SSL feature capture is invalid")
                    captured.append(arguments[0])

                handle = head.register_forward_pre_hook(capture)
                optimizer.zero_grad(set_to_none=True)
                try:
                    source_logits = None
                    if source_teacher is not None:
                        with torch.no_grad():
                            source_logits = validate_model_output(
                                source_teacher(
                                    prepare_canonical_model_input(
                                        raw,
                                        spec,
                                        epsilon=normalization_epsilon,
                                    )
                                ),
                                spec,
                                batch_size=int(raw.shape[0]),
                                check_finite=True,
                            ).float()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        student_outputs = []
                        for view in (first, second):
                            model_input = prepare_canonical_model_input(
                                view,
                                spec,
                                epsilon=normalization_epsilon,
                            )
                            student_outputs.append(model(model_input))
                        if len(captured) != 2:
                            raise RuntimeError("target SSL expected exactly two views")
                        first_projection = projector(captured[0])
                        second_projection = projector(captured[1])
                        if config.pretrain_ssl_objective == "simclr":
                            ssl_loss = _paired_simclr_loss(
                                first_projection,
                                second_projection,
                                temperature=config.simclr_temperature,
                            )
                        elif config.pretrain_ssl_objective == "simclr_vicreg":
                            simclr_loss = _paired_simclr_loss(
                                first_projection,
                                second_projection,
                                temperature=config.simclr_temperature,
                            )
                            vicreg_loss = _vicreg_loss(
                                first_projection,
                                second_projection,
                                invariance_weight=config.pretrain_vicreg_invariance_weight,
                                variance_weight=config.pretrain_vicreg_variance_weight,
                                covariance_weight=config.pretrain_vicreg_covariance_weight,
                            )
                            ssl_loss = (
                                simclr_loss
                                + config.pretrain_vicreg_mix_weight * vicreg_loss
                            )
                        elif config.pretrain_ssl_objective == "vicreg":
                            ssl_loss = _vicreg_loss(
                                first_projection,
                                second_projection,
                                invariance_weight=config.pretrain_vicreg_invariance_weight,
                                variance_weight=config.pretrain_vicreg_variance_weight,
                                covariance_weight=config.pretrain_vicreg_covariance_weight,
                            )
                        else:
                            ssl_loss = _barlow_twins_loss(
                                first_projection,
                                second_projection,
                                offdiagonal_weight=config.pretrain_barlow_offdiagonal_weight,
                            )
                        if source_logits is not None:
                            anchor_loss = _weighted_logit_anchor_loss(
                                student_outputs[0].float(),
                                source_logits,
                                config.pretrain_logit_anchor_class_weights,
                            )
                        else:
                            anchor_loss = ssl_loss.detach() * 0.0
                        source_probabilities = (
                            torch.sigmoid(source_logits)
                            if source_logits is not None
                            else None
                        )
                        if prototype_bank is not None and source_probabilities is not None:
                            prototype_loss = torch.stack(
                                [
                                    _prototype_classification_loss(
                                        feature,
                                        source_probabilities,
                                        prototype_bank,
                                        temperature=config.pretrain_prototype_temperature,
                                    )
                                    for feature in captured
                                ]
                            ).mean()
                            rank_loss = torch.stack(
                                [
                                    _class_balanced_rank_loss(
                                        output,
                                        source_probabilities,
                                        prototype_bank,
                                        margin=config.pretrain_rank_margin,
                                        temperature=config.pretrain_rank_temperature,
                                    )
                                    for output in student_outputs
                                ]
                            ).mean()
                            strong_pseudo_loss = _masked_strong_pseudo_loss(
                                student_outputs[1],
                                source_probabilities,
                                prototype_bank,
                            )
                        else:
                            prototype_loss = ssl_loss.detach() * 0.0
                            rank_loss = ssl_loss.detach() * 0.0
                            strong_pseudo_loss = ssl_loss.detach() * 0.0
                        loss = (
                            config.pretrain_ssl_weight * ssl_loss
                            + config.pretrain_logit_anchor_weight * anchor_loss
                            + config.pretrain_prototype_weight * prototype_loss
                            + config.pretrain_rank_weight * rank_loss
                            + config.pretrain_strong_pseudo_weight
                            * strong_pseudo_loss
                            + config.pretrain_source_replay_weight * source_loss
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*trainable, *projector.parameters()], max_norm=5.0
                    )
                    optimizer.step()
                    scheduler.step()
                finally:
                    handle.remove()
                _STATE.pretrain_steps += 1
                _STATE.pretrain_samples += int(raw.shape[0])
                _STATE.pretrain_loss_sum += float(loss.detach().cpu())
                _STATE.pretrain_ssl_loss_sum += float(ssl_loss.detach().cpu())
                _STATE.pretrain_logit_anchor_loss_sum += float(
                    anchor_loss.detach().cpu()
                )
                _STATE.pretrain_prototype_loss_sum += float(
                    prototype_loss.detach().cpu()
                )
                _STATE.pretrain_rank_loss_sum += float(rank_loss.detach().cpu())
                _STATE.pretrain_strong_pseudo_loss_sum += float(
                    strong_pseudo_loss.detach().cpu()
                )
                _STATE.pretrain_source_loss_sum += float(source_loss.detach().cpu())
                _STATE.hash_digests.append(
                    _batch_digest(batch, int(raw.shape[0]))
                )
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
        _STATE.iterator = None
        _STATE.source_iterator = None
        _STATE.teacher = None


def _unlabeled_loss(
    kwargs: Mapping[str, Any], *, update_teacher: bool
) -> torch.Tensor:
    config = _STATE.config
    if config is None:
        raise RuntimeError("unlabeled teacher config is unavailable")
    model = kwargs.get("model")
    spec = kwargs.get("spec")
    epsilon = kwargs.get("normalization_epsilon")
    if not isinstance(model, nn.Module) or not isinstance(spec, ModelSpec):
        raise TypeError("unlabeled teacher requires model and ModelSpec")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("normalization epsilon must be numeric")
    batch = _next_batch()
    raw = batch.get("waveform")
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape[1:]) != (1000, 12):
        raise ValueError("unlabeled waveform must have shape (B,1000,12)")
    batch_size = int(raw.shape[0])
    device = next(model.parameters()).device
    raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
    if not bool(torch.isfinite(raw).all().item()):
        raise FloatingPointError("unlabeled target waveform contains NaN or Inf")

    teacher = (
        _update_teacher(model, config.ema_decay)
        if update_teacher
        else _STATE.teacher
    )
    if teacher is None:
        raise RuntimeError("EMA teacher was not initialized")
    rng_state = _capture_rng(model)
    bn_state = _disable_bn_running_stats(model)
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        if len(arguments) != 1 or arguments[0].ndim != 2:
            raise RuntimeError("unlabeled student feature capture is invalid")
        captured.append(arguments[0])

    handle = _classifier_head(model).register_forward_pre_hook(capture)
    try:
        # ``no_grad`` keeps the pseudo targets as ordinary tensors. PyTorch
        # inference tensors cannot be saved by BCE for the student backward.
        with torch.no_grad():
            weak_input = prepare_canonical_model_input(raw, spec, epsilon=float(epsilon))
            teacher_logits = validate_model_output(
                teacher(weak_input), spec, batch_size=batch_size, check_finite=True
            )
            probabilities = torch.sigmoid(teacher_logits.float() / config.temperature)
            confidence = torch.maximum(probabilities, 1.0 - probabilities)
            if config.pseudo_reduction == "class_polarity_balanced_quantile":
                lower = torch.quantile(
                    probabilities, config.pseudo_quantile, dim=0, keepdim=True
                )
                upper = torch.quantile(
                    probabilities,
                    1.0 - config.pseudo_quantile,
                    dim=0,
                    keepdim=True,
                )
                # Quantile selection is deliberately class-relative.  A fixed
                # 0.5 gate silently removes nearly every positive for rare
                # classes such as HYP even when the teacher ranks them well.
                positive_mask = probabilities >= upper
                negative_mask = probabilities <= lower
                mask = positive_mask | negative_mask
            else:
                mask = confidence >= config.confidence_threshold
                positive_mask = mask & (probabilities >= 0.5)
                negative_mask = mask & (probabilities < 0.5)
        numerator = raw.new_zeros(())
        denominator = mask.sum().clamp_min(1).to(dtype=torch.float32)
        for _ in range(config.strong_views):
            strong = _strong_view(raw, config)
            strong_input = prepare_canonical_model_input(
                strong, spec, epsilon=float(epsilon)
            )
            student_logits = validate_model_output(
                model(strong_input), spec, batch_size=batch_size, check_finite=True
            )
            elementwise = F.binary_cross_entropy_with_logits(
                student_logits.float(), probabilities, reduction="none"
            )
            if config.pseudo_reduction == "class_polarity_balanced_quantile":
                class_losses: list[torch.Tensor] = []
                for class_index in range(5):
                    partitions: list[torch.Tensor] = []
                    positive = positive_mask[:, class_index]
                    negative = negative_mask[:, class_index]
                    if bool(positive.any().item()):
                        partitions.append(elementwise[positive, class_index].mean())
                    if bool(negative.any().item()):
                        partitions.append(elementwise[negative, class_index].mean())
                    if partitions:
                        class_losses.append(torch.stack(partitions).mean())
                balanced = (
                    torch.stack(class_losses).mean()
                    if class_losses
                    else elementwise.sum() * 0.0
                )
                numerator = numerator + balanced
            else:
                numerator = numerator + (elementwise * mask).sum() / denominator
        pseudo_loss = numerator / float(config.strong_views)
        if len(captured) != config.strong_views:
            raise RuntimeError("unlabeled student feature capture count drifted")
        simclr_loss = (
            _paired_simclr_loss(
                captured[0], captured[1], temperature=config.simclr_temperature
            )
            if config.simclr_weight > 0.0
            else pseudo_loss.detach() * 0.0
        )
        loss = pseudo_loss + config.simclr_weight * simclr_loss
    finally:
        handle.remove()
        _restore_bn_running_stats(bn_state)
        _restore_rng(rng_state)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("unlabeled teacher loss became NaN or Inf")
    accepted = int(mask.sum().detach().cpu())
    _STATE.objective_calls += 1
    _STATE.unlabeled_batches += 1
    _STATE.unlabeled_samples += batch_size
    _STATE.strong_forwards += config.strong_views
    _STATE.accepted_elements += accepted
    _STATE.total_elements += int(mask.numel())
    for class_index in range(5):
        _STATE.accepted_positive_by_class[class_index] += int(
            positive_mask[:, class_index].sum().detach().cpu()
        )
        _STATE.accepted_negative_by_class[class_index] += int(
            negative_mask[:, class_index].sum().detach().cpu()
        )
    _STATE.confidence_sum += float(confidence.mean().detach().cpu())
    _STATE.loss_sum += float(loss.detach().cpu())
    _STATE.pseudo_loss_sum += float(pseudo_loss.detach().cpu())
    _STATE.simclr_loss_sum += float(simclr_loss.detach().cpu())
    _STATE.hash_digests.append(_batch_digest(batch, batch_size))
    return loss


def _supervised_logit_anchor_loss(kwargs: Mapping[str, Any]) -> torch.Tensor:
    config = _STATE.config
    teacher = _STATE.supervised_anchor_teacher
    model = kwargs.get("model")
    spec = kwargs.get("spec")
    bundle = kwargs.get("bundle")
    epsilon = kwargs.get("normalization_epsilon")
    if config is None or teacher is None:
        raise RuntimeError("supervised logit anchor state is unavailable")
    if not isinstance(model, nn.Module) or not isinstance(spec, ModelSpec):
        raise TypeError("supervised logit anchor requires model and ModelSpec")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("normalization epsilon must be numeric")
    clean = bundle.require("clean_view")
    raw = getattr(clean, "waveform", None)
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape[1:]) != (1000, 12):
        raise ValueError("supervised logit anchor requires clean (B,1000,12) waveform")
    model_input = prepare_canonical_model_input(raw, spec, epsilon=float(epsilon))
    rng_state = _capture_rng(model)
    bn_state = _disable_bn_running_stats(model)
    try:
        with torch.no_grad():
            teacher_logits = validate_model_output(
                teacher(model_input),
                spec,
                batch_size=int(raw.shape[0]),
                check_finite=True,
            )
        student_logits = validate_model_output(
            model(model_input),
            spec,
            batch_size=int(raw.shape[0]),
            check_finite=True,
        )
        return _weighted_logit_anchor_loss(
            student_logits,
            teacher_logits,
            config.supervised_logit_anchor_class_weights,
        )
    finally:
        _restore_bn_running_stats(bn_state)
        _restore_rng(rng_state)


def diagnostics_payload() -> dict[str, Any] | None:
    if _STATE.config is None:
        return None
    calls = max(_STATE.objective_calls, 1)
    return {
        "schema_version": 1,
        "policy": _STATE.config.policy,
        "evidence_status": _STATE.config.evidence_status,
        "warning": (
            "Ref-excluded evaluation records were consumed without labels; this is "
            "transductive exploration and cannot be cited as inductive held-out evidence."
            if _STATE.config.pool_partition == "pn2021_all_zero_kept_refexcluded"
            else "Only the declared K500 adaptation partition was consumed."
        ),
        "method_id": _STATE.method_id,
        "center": _STATE.center,
        "model_name": _STATE.model_name,
        "replicate_id": _STATE.replicate_id,
        "pool_partition": _STATE.config.pool_partition,
        "labels_consumed": _STATE.config.labels_consumed,
        "online_enabled": _STATE.config.online_enabled,
        "config": _STATE.config.__dict__,
        "objective_calls": _STATE.objective_calls,
        "ema_updates": _STATE.ema_updates,
        "unlabeled_batches": _STATE.unlabeled_batches,
        "unlabeled_samples": _STATE.unlabeled_samples,
        "strong_forwards": _STATE.strong_forwards,
        "accepted_element_fraction": _STATE.accepted_elements
        / max(_STATE.total_elements, 1),
        "accepted_positive_by_class": list(_STATE.accepted_positive_by_class),
        "accepted_negative_by_class": list(_STATE.accepted_negative_by_class),
        "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
        "mean_teacher_confidence": _STATE.confidence_sum / calls,
        "mean_unlabeled_loss": _STATE.loss_sum / calls,
        "mean_pseudo_label_loss": _STATE.pseudo_loss_sum / calls,
        "mean_simclr_loss": _STATE.simclr_loss_sum / calls,
        "pretrain": {
            "steps": _STATE.pretrain_steps,
            "samples": _STATE.pretrain_samples,
            "mean_loss": _STATE.pretrain_loss_sum / max(_STATE.pretrain_steps, 1),
            "ssl_objective": _STATE.config.pretrain_ssl_objective,
            "ssl_weight": _STATE.config.pretrain_ssl_weight,
            "view_mode": _STATE.config.pretrain_view_mode,
            "mean_ssl_loss": _STATE.pretrain_ssl_loss_sum
            / max(_STATE.pretrain_steps, 1),
            "mean_logit_anchor_loss": _STATE.pretrain_logit_anchor_loss_sum
            / max(_STATE.pretrain_steps, 1),
            "logit_anchor_class_weights": list(
                _STATE.config.pretrain_logit_anchor_class_weights
            ),
            "mean_prototype_loss": _STATE.pretrain_prototype_loss_sum
            / max(_STATE.pretrain_steps, 1),
            "mean_rank_loss": _STATE.pretrain_rank_loss_sum
            / max(_STATE.pretrain_steps, 1),
            "mean_strong_pseudo_loss": _STATE.pretrain_strong_pseudo_loss_sum
            / max(_STATE.pretrain_steps, 1),
            "source_semantic_replay": {
                "weight": _STATE.config.pretrain_source_replay_weight,
                "batches_per_step": _STATE.config.pretrain_source_batches_per_step,
                "samples": _STATE.pretrain_source_samples,
                "mean_bce": _STATE.pretrain_source_loss_sum
                / max(_STATE.pretrain_steps, 1),
                "dataset": "ptbxl",
                "partition": "train",
                "labels_consumed": (
                    _STATE.config.pretrain_source_replay_weight > 0.0
                ),
                "batch_norm_policy": "disable_running_stat_tracking",
                "rng_policy": "snapshot_restore_source_forward",
                "loader": dict(_STATE.source_loader_identity or {}),
                "ordered_batch_hashes_sha256": (
                    hashlib.sha256(
                        "\n".join(_STATE.source_hash_digests).encode()
                    ).hexdigest()
                    if _STATE.config.pretrain_source_replay_weight > 0.0
                    else None
                ),
            },
            "prototype_bank": {
                "positive_counts": list(_STATE.prototype_positive_counts),
                "negative_counts": list(_STATE.prototype_negative_counts),
                "positive_thresholds": list(
                    _STATE.prototype_positive_thresholds
                ),
                "negative_thresholds": list(
                    _STATE.prototype_negative_thresholds
                ),
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "projection_head_persisted": False,
            "classifier_head_updated": (
                _STATE.config.pretrain_ssl_objective
                == "supervised_bce_budget_filler"
            ),
            "pair": (
                (
                    "clean_to_clean_identity"
                    if _STATE.config.pretrain_clean_anchor
                    else "clean_identity_to_clean_identity"
                )
                if _STATE.config.pretrain_view_mode == "clean_identity"
                else (
                    "clean_to_augmix_strong"
                    if _STATE.config.pretrain_clean_anchor
                    else "augmix_strong_to_augmix_strong"
                )
            ),
        },
        "supervised_logit_anchor": {
            "weight": _STATE.config.supervised_logit_anchor_weight,
            "class_weights": list(
                _STATE.config.supervised_logit_anchor_class_weights
            ),
            "calls": _STATE.supervised_anchor_calls,
            "mean_loss": _STATE.supervised_anchor_loss_sum
            / max(_STATE.supervised_anchor_calls, 1),
            "teacher_snapshot": "post_target_ssl_pre_supervised",
            "views": ["clean_view"],
        },
        "residual_target_head": {
            "mode": _STATE.config.residual_head_mode,
            "width": _STATE.config.residual_head_width,
            "dropout": _STATE.config.residual_head_dropout,
            "initial_scale": _STATE.config.residual_head_scale,
            "freeze_base": _STATE.config.residual_head_freeze_base,
            "parameter_count": _STATE.residual_head_parameter_count,
            "trainable_parameter_count": _STATE.residual_head_trainable_parameter_count,
        },
        "ordered_batch_hashes_sha256": hashlib.sha256(
            "\n".join(_STATE.hash_digests).encode()
        ).hexdigest(),
        "loader": dict(_STATE.loader_identity or {}),
        "batch_norm_policy": "disable_running_stat_tracking_for_student_strong_views",
        "rng_policy": "isolated_corruption_generator_plus_snapshot_restore_model_rng",
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    destination = Path(output_dir).expanduser().resolve() / "unlabeled_teacher_diagnostics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    import core.online_trainer as online_trainer
    import core.pn2021_tuning as pn2021_tuning
    import core.train_PN2021 as train_pn2021

    original_train = online_trainer.train_online_model
    original_tuning_alias = pn2021_tuning.train_online_model
    original_train_alias = train_pn2021.train_online_model
    original_objective = online_trainer._compute_objective
    _STATE.reset()

    def patched_objective(**kwargs: Any):
        result = original_objective(**kwargs)
        if not _STATE.active or not _is_clean_objective(kwargs.get("objective_term_names")):
            return result
        method = kwargs.get("method")
        if not isinstance(method, CompiledMethod) or method.profile_name != _STATE.method_id:
            return result
        config = _STATE.config
        if config is None:
            raise RuntimeError("unlabeled teacher state is incomplete")
        if config.online_enabled:
            consistency_terms = [
                _unlabeled_loss(kwargs, update_teacher=index == 0)
                for index in range(config.unlabeled_batches_per_clean)
            ]
            consistency = torch.stack(consistency_terms).mean()
        else:
            consistency = result.total.detach() * 0.0
        supervised_anchor = (
            _supervised_logit_anchor_loss(kwargs)
            if config.supervised_logit_anchor_weight > 0.0
            else consistency.detach() * 0.0
        )
        if config.supervised_logit_anchor_weight > 0.0:
            _STATE.supervised_anchor_calls += 1
            _STATE.supervised_anchor_loss_sum += float(
                supervised_anchor.detach().cpu()
            )
        result_type = type(result)
        scale = config.total_weight / CLEAN_FAMILY_WEIGHT
        anchor_scale = config.supervised_logit_anchor_weight / CLEAN_FAMILY_WEIGHT
        return result_type(
            total=(
                result.total
                + consistency * scale
                + supervised_anchor * anchor_scale
            ),
            raw_terms=dict(result.raw_terms),
            weighted_terms=dict(result.weighted_terms),
            valid_counts=dict(result.valid_counts),
        )

    def patched_train(*args: Any, **kwargs: Any):
        method_path = Path(str(kwargs.get("method_config_path"))).expanduser()
        config_path = kwargs.get("config_path", online_trainer.DEFAULT_ONLINE_CONFIG_PATH)
        config_root = kwargs.get("config_root")
        config = online_trainer.load_online_train_config(config_path, config_root=config_root)
        if not method_path.is_absolute():
            method_path = config.config_root / method_path
        method = online_trainer.compile_method_profile(method_path.resolve())
        teacher_config = _teacher_config(method)
        if teacher_config is None:
            return original_train(*args, **kwargs)
        if _STATE.active:
            raise RuntimeError("nested unlabeled teacher runs are not supported")
        train_loader = args[1] if len(args) > 1 else kwargs.get("train_dataloader")
        batch_size = getattr(train_loader, "batch_size", None)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("unlabeled teacher requires a positive labelled batch size")
        model = args[0] if args else kwargs.get("model")
        if not isinstance(model, nn.Module):
            raise TypeError("unlabeled teacher requires a torch model")
        center = str(kwargs.get("center", ""))
        if not center:
            raise ValueError("unlabeled teacher requires a center")
        spec = online_trainer._model_spec(model)
        if spec.name not in {"efficientnet1dv2", "ecgfounder"}:
            raise ValueError(
                "unlabeled teacher exploration supports only managed "
                "EfficientNet1DV2 and ECGFounder models"
            )
        seed_path = config.references["random_seed_config"]
        random_seed = config.payload.get("random_seed")
        if not isinstance(random_seed, Mapping):
            raise ValueError("unlabeled teacher requires random_seed mapping")
        replicate_id = random_seed.get("replicate_id")
        if (
            isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or replicate_id < 0
        ):
            raise ValueError(
                "unlabeled teacher random_seed.replicate_id must be "
                "a non-negative integer"
            )
        augmix = load_augmix_config(
            config.config_root / "train" / "augmix.yaml",
            config_root=config.config_root,
        )
        operator_params = {
            name: augmix.operator_profile_config.parameters_for(name)
            for name in augmix.canonical_operators
        }
        loader = get_dataloader(
            dataset="pn2021",
            partition=teacher_config.pool_partition,
            logical_center=center,
            batch_size=batch_size,
            sampling_rate_hz=100,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            persistent_workers=False,
            cache_mode="mmap",
            validate_values="sample",
            selection_resident=True,
            selection_resident_pin_memory=True,
            prepare_for_model=False,
            sanitize=False,
            global_zscore=False,
            output_layout="time_channel",
            seed_namespace=(
                "unlabeled_teacher_pool_v1:"
                f"{teacher_config.pool_partition}:{center}:{spec.name}:"
                f"replicate{replicate_id}"
            ),
            seed_config_path=seed_path,
            config_root=config.config_root,
        )
        source_loader = None
        if teacher_config.pretrain_source_replay_weight > 0.0:
            source_loader = get_dataloader(
                dataset="ptbxl",
                partition="train",
                batch_size=batch_size,
                sampling_rate_hz=100,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                pin_memory=True,
                persistent_workers=False,
                cache_mode="ram",
                validate_values="sample",
                selection_resident=False,
                prepare_for_model=False,
                sanitize=False,
                global_zscore=False,
                output_layout="time_channel",
                seed_namespace=(
                    f"target_ssl_source_semantic_v1:{center}:{spec.name}:"
                    f"replicate{replicate_id}"
                ),
                seed_config_path=seed_path,
                config_root=config.config_root,
            )
        _STATE.active = True
        _STATE.config = teacher_config
        _STATE.loader = loader
        _STATE.source_loader = source_loader
        _STATE.operator_params = operator_params
        _STATE.seed_config_path = Path(seed_path)
        _STATE.center = center
        _STATE.model_name = spec.name
        _STATE.method_id = method.profile_name
        _STATE.replicate_id = replicate_id
        describe = getattr(loader, "describe", None)
        _STATE.loader_identity = describe() if callable(describe) else {
            "type": loader.__class__.__name__, "batch_size": batch_size
        }
        if source_loader is not None:
            describe_source = getattr(source_loader, "describe", None)
            _STATE.source_loader_identity = (
                describe_source()
                if callable(describe_source)
                else {
                    "type": source_loader.__class__.__name__,
                    "batch_size": batch_size,
                }
            )
        else:
            _STATE.source_loader_identity = None
        try:
            requested_device = str(
                kwargs.get("device") or config.payload["training"]["device"]
            )
            if requested_device == "auto":
                requested_device = "cuda" if torch.cuda.is_available() else "cpu"
            resolved_device = torch.device(requested_device)
            if resolved_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("target SSL pretraining requested unavailable CUDA")
            model.to(resolved_device)
            _install_residual_target_head(model, teacher_config)
            _pretrain_target_representation(
                model,
                spec,
                normalization_epsilon=float(
                    config.payload["data"]["normalization_epsilon"]
                ),
            )
            if teacher_config.supervised_logit_anchor_weight > 0.0:
                _STATE.supervised_anchor_teacher = copy.deepcopy(model).eval()
                for parameter in _STATE.supervised_anchor_teacher.parameters():
                    parameter.requires_grad_(False)
            return original_train(*args, **kwargs)
        finally:
            _STATE.active = False
            _STATE.iterator = None
            _STATE.supervised_anchor_teacher = None
            close = getattr(loader, "close", None)
            if callable(close):
                close()
            _STATE.loader = None
            _STATE.source_iterator = None
            if source_loader is not None:
                close_source = getattr(source_loader, "close", None)
                if callable(close_source):
                    close_source()
            _STATE.source_loader = None
            _STATE.generator = None

    online_trainer._compute_objective = patched_objective
    online_trainer.train_online_model = patched_train
    pn2021_tuning.train_online_model = patched_train
    train_pn2021.train_online_model = patched_train
    try:
        yield
    finally:
        train_pn2021.train_online_model = original_train_alias
        pn2021_tuning.train_online_model = original_tuning_alias
        online_trainer.train_online_model = original_train
        online_trainer._compute_objective = original_objective
        if _STATE.active:
            raise RuntimeError("unlabeled teacher adapter exited during an active run")


__all__ = [
    "CONTRACT_KEY",
    "EVIDENCE_STATUS",
    "POLICY",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "patch_pn2021_evaluation_runtime",
    "write_diagnostics",
]
