"""Global, K500-only A7 VAE-LHAT + AugMix search controller.

This file owns only derived exploratory YAMLs, resource gates, parallel managed
launches, and macro-metric summaries. The existing sandbox launcher and
manual-refactor whitelist continue to own model construction, training, data
loading, evaluation, and run records.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SANDBOX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SANDBOX_ROOT.parents[1]
CONFIG_ROOT = SANDBOX_ROOT / "configs"
GENERATED_ROOT = CONFIG_ROOT / "generated_search" / "a7_global_search_20260726"
RUN_ROOT = (
    Path("/home/linbinhao/ECG_adv_data/runs/agent_workspace")
    / "a7_global_search_20260726"
)
TMP_ROOT = Path("/dev/shm/a7_global_search_20260726")
PYTHON = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
LAUNCHER = SANDBOX_ROOT / "launch.py"
BASE_A7_PARENT = (
    CONFIG_ROOT
    / "generated_ablation"
    / "point29_a7_equal5_bothbackbones_fullk500_e40_20260725_v1"
)
STRONG_LHAT_AUGMIX_METHOD = (
    CONFIG_ROOT
    / "train"
    / "methods"
    / "fixed20_pcgrad_strong_lhat_augmix_v1.yaml"
)
STRONG_LHAT_DIRECT_AUGMIX_METHOD = (
    CONFIG_ROOT
    / "train"
    / "methods"
    / "fixed20_pcgrad_strong_lhat_direct_augmix_v1.yaml"
)
DIRECT_FIXED20_METHOD = (
    CONFIG_ROOT / "train" / "methods" / "direct_depth23_fixed20.yaml"
)
CLEAN_ONLY_METHOD = (
    PROJECT_ROOT / "configs" / "train" / "methods" / "a0_clean_v1.yaml"
)
PURE_M20_BOUNDED_METHOD = (
    CONFIG_ROOT
    / "train"
    / "methods"
    / "pure_m20_lhat_augmax_bounded.yaml"
)
BASE_LHAT_CONFIG = CONFIG_ROOT / "train" / "lhat.yaml"
STRONG_LHAT_CONFIG = CONFIG_ROOT / "train" / "lhat_h060_s5_eps8.yaml"
VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID = (
    "vae_lhat_bounded_augmax_polish_v1"
)
HARD_ONLY_PRUNED_METHOD_ID = (
    "fixed20_directsum_vae_lhat_hard_only_pruned_v1"
)
CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID = (
    "clean_bounded_augmax_polish_control_v1"
)
VAE_CLEAN_POLISH_METHOD_ID = "vae_lhat_clean_polish_v1"
VAE_CLEAN_DIRECTSUM_METHOD_ID = "vae_lhat_clean_directsum_v1"
CLEAN_TARGET_SSL_METHOD_ID = "a0_clean_v1"
VAE_CLEAN_POLISH_EFFECTIVE_WEIGHTS = {
    "clean_bce": 0.50,
    "lhat_direct_bce": 0.25,
    "clean_lhat_jsd": 0.25,
}
VAE_CLEAN_DIRECTSUM_AUXILIARY_ALPHAS = frozenset({0.5, 1.0, 1.5})
VAE_TEACHER_DISTILL_POLISH_METHOD_ID = (
    "vae_lhat_teacher_distill_polish_v1"
)
CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID = (
    "clean_teacher_distill_polish_control_v1"
)
VAE_HARD_BCE_POLISH_METHOD_ID = "vae_lhat_hard_bce_polish_v1"
CLEAN_HARD_BCE_CONTROL_METHOD_ID = "clean_hard_bce_polish_control_v1"
VAE_PATH_BCE_POLISH_METHOD_ID = "vae_lhat_path_bce_polish_v1"
CLEAN_PATH_BCE_CONTROL_METHOD_ID = "clean_lhat_path_bce_polish_control_v1"
VAE_CONSISTENCY_POLISH_METHOD_ID = "vae_lhat_consistency_polish_v1"
CLEAN_CONSISTENCY_CONTROL_METHOD_ID = "clean_consistency_polish_control_v1"
VAE_FEATURE_INVARIANCE_POLISH_METHOD_ID = (
    "vae_lhat_feature_invariance_polish_v1"
)
CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID = (
    "clean_feature_invariance_polish_control_v1"
)
VAE_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID = (
    "vae_lhat_local_anchor_soft_polish_v1"
)
CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID = (
    "clean_local_anchor_soft_polish_control_v1"
)
VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID = (
    "vae_lhat_calibrated_local_anchor_soft_polish_v1"
)
CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID = (
    "clean_calibrated_local_anchor_soft_polish_control_v1"
)
LHAT_PATH_T_VALUES = [0.4, 0.7, 1.0]
VAE_TEACHER_DISTILL_EFFECTIVE_WEIGHTS = {
    "clean_bce": 0.50,
    "clean_preserve_bce": 0.25,
    "lhat_teacher_distill_bce": 0.25,
}
VAE_CHAIN3_POLISH_METHOD_ID = "vae_lhat_chain3_polish_v1"
CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID = "clean_chain3_polish_control_v1"
VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID = (
    "vae_lhat_chain3_teacher04_polish_v1"
)
CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID = (
    "clean_chain3_teacher04_polish_control_v1"
)
CHAIN3_POLISH_EFFECTIVE_WEIGHTS = {
    "clean_bce": 0.50,
    "clean_preserve_bce": 0.25,
    "augmix_bce": 0.125,
    "consistency_jsd": 0.125,
}
LOCKED_SOURCE_REGISTRY = CONFIG_ROOT / "baselines" / "ptbxl_source_v1.yaml"
SOURCE_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train" / "PTBXL.yaml"
SOURCE_CHECKPOINTS = {
    "efficientnet1dv2": Path(
        "/data/linbinhao/ECG_manual_refactor_runs/manual_refactor_20260721/"
        "managed_ptbxl_effnet_source_v1/training/checkpoints/best.pt"
    ),
    "ecgfounder": Path(
        "/data/linbinhao/ECG_manual_refactor_runs/manual_refactor_20260721/"
        "managed_ptbxl_ecgfounder_source_v1/training/checkpoints/best.pt"
    ),
}
EVAL_BATCH_SIZES = {
    "efficientnet1dv2": 1024,
    "ecgfounder": 128,
}
VAE_CHECKPOINT = Path(
    "/home/linbinhao/ECG_adv_data/models/ECGTwin/checkpoints/vae_model.pth"
)
VAE_CHECKPOINT_MANIFEST_FAMILIES = frozenset(
    {
        "d19_separate_views",
        "strong_lhat_augmix_chain3",
        "vae_lhat_post_refine",
        "vae_lhat_clean_polish",
        "vae_lhat_chain3_polish",
        "vae_lhat_chain3_teacher04_polish",
        "vae_lhat_bounded_augmax_polish",
    }
)
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
PTBXL_LEAD_ORDER = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"
MIN_AVAILABLE_RAM_GIB = 48.0
MIN_SHM_AVAILABLE_GIB = 24.0
MIN_NVME_AVAILABLE_GIB = 150.0
MIN_PERSISTENT_AVAILABLE_GIB = 300.0
MIN_FREE_GPU_MIB = 20_000
CLASSIC_AUGMIX_VIEW_MODE = "classic_augmix3_depth13"
FIXED20_SUPERVISED_SCHEDULE = "fixed20_exhaustive_v1"
ROTATING4_SUPERVISED_SCHEDULE = "rotating_depth23_2plus2_v1"
ROTATING4_EXPOSURE_POLICY = "clean_aux_once_then_rotating_depth23_2plus2"
ROTATING4_EXECUTION_SLOT_SENTINELS = (0, 1, 10, 19)
PRETRAIN_POLICY_BY_VIEW_MODE = {
    "clean_identity": "clean_identity_simclr_vicreg_control_v1",
    "twochain_augmix": "ema_teacher_clean_to_twochain_augmix_strong_v1",
    CLASSIC_AUGMIX_VIEW_MODE: "clean_to_classic_augmix3_depth13_simclr_v1",
}
CLASSIC_AUGMIX_TOPOLOGY = {
    "mixture_width": 3,
    "depth_sampling": "uniform_integer_1_to_3",
    "operator_sampling": "uniform_with_replacement",
    "operator_application_order": "sampled_sequence",
    "operator_domain_sampling_rate_hz": 500,
}
LHAT_ATTACK_OBJECTIVES = {
    "maximize_multilabel_bce_with_logits",
    "maximize_equal_positive_negative_bce_with_logits",
    "maximize_bernoulli_kl_from_decoded_anchor",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    learning_rate: float
    batch_size: int
    model_name: str = "efficientnet1dv2"
    initializer_candidate: str | None = None
    epochs: int = 40
    scheduler_horizon_epochs: int = 40
    weight_decay: float = 1.0e-4
    pretrain_epochs: int = 1
    pretrain_steps: int = 128
    pretrain_learning_rate: float = 1.0e-3
    pretrain_logit_anchor_weight: float = 5.0
    supervised_logit_anchor_weight: float = 2.0
    auxiliary_alpha: float = 1.0
    clean_direct_sum_auxiliary: bool = False
    augmix_dirichlet_alpha: float = 0.5
    augmix_beta_alpha: float = 0.5
    pretrain_ssl_objective: str = "simclr"
    pretrain_ssl_weight: float = 1.0
    pretrain_vicreg_mix_weight: float = 0.03
    pretrain_view_mode: str | None = None
    pretrain_source_replay_weight: float | None = None
    pretrain_source_batches_per_step: int = 1
    supervised_source_replay_weight: float | None = None
    raw_auxiliary_scale: float = 1.0
    vae_random_auxiliary_scale: float = 1.0
    vae_hard_auxiliary_scale: float = 1.0
    jsd_auxiliary_scale: float = 1.0
    method_family: str = "d19_separate_views"
    checkpoint_blend_alpha: float | None = None
    augmax_hard_hull_lambda: float | None = None
    lhat_pgd_epsilon_l2_standardized: float = 2.0
    comparison_rng_identity: str | None = None
    frozen_teacher_mix: float | None = None
    polish_auxiliary_target: str | None = None
    local_lhat_geometry_preset: str | None = None
    lhat_geometry_preset: str | None = None
    pcgrad_auxiliary_norm_ratio_cap: float | None = None
    lhat_search_mode: str = "optimized_softmax"
    lhat_attack_objective: str = "maximize_multilabel_bce_with_logits"
    supervised_corruption_schedule: str = FIXED20_SUPERVISED_SCHEDULE
    replicate_id: int = 0


CANDIDATES: Mapping[str, Candidate] = {
    "p00_lr1p5e4_bs64": Candidate(
        candidate_id="p00_lr1p5e4_bs64",
        learning_rate=1.5e-4,
        batch_size=64,
    ),
    "p01_aux150": Candidate(
        candidate_id="p01_aux150",
        learning_rate=1.5e-4,
        batch_size=64,
        auxiliary_alpha=1.5,
    ),
    "p02_ssl256": Candidate(
        candidate_id="p02_ssl256",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
    ),
    "p03_ssl256_aux150": Candidate(
        candidate_id="p03_ssl256_aux150",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        auxiliary_alpha=1.5,
    ),
    "p04_anchor_light": Candidate(
        candidate_id="p04_anchor_light",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_logit_anchor_weight=2.0,
        supervised_logit_anchor_weight=1.0,
    ),
    "p05_anchor_strong": Candidate(
        candidate_id="p05_anchor_strong",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_logit_anchor_weight=10.0,
        supervised_logit_anchor_weight=4.0,
    ),
    "p06_ssl256_aux150_augmix11": Candidate(
        candidate_id="p06_ssl256_aux150_augmix11",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        auxiliary_alpha=1.5,
        augmix_dirichlet_alpha=1.0,
        augmix_beta_alpha=1.0,
    ),
    "p07_ssl256_aux150_vicreg": Candidate(
        candidate_id="p07_ssl256_aux150_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="vicreg",
        pretrain_ssl_weight=0.03,
    ),
    "p08_ssl256_vae_focus": Candidate(
        candidate_id="p08_ssl256_vae_focus",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        raw_auxiliary_scale=0.5,
        vae_random_auxiliary_scale=4.0 / 3.0,
        vae_hard_auxiliary_scale=4.0 / 3.0,
    ),
    "p09_ssl256_vae_hard_focus": Candidate(
        candidate_id="p09_ssl256_vae_hard_focus",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        raw_auxiliary_scale=0.5,
        vae_random_auxiliary_scale=2.0 / 3.0,
        vae_hard_auxiliary_scale=2.0,
    ),
    "p10_ssl256_aux150_barlow": Candidate(
        candidate_id="p10_ssl256_aux150_barlow",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="barlow_twins",
        pretrain_ssl_weight=0.03,
    ),
    "p11_ssl512_aux150": Candidate(
        candidate_id="p11_ssl512_aux150",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=1.5,
    ),
    "p12_ssl256_strong_lhat_chain3": Candidate(
        candidate_id="p12_ssl256_strong_lhat_chain3",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        auxiliary_alpha=0.25,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
    ),
    "p13_ssl256_direct_control": Candidate(
        candidate_id="p13_ssl256_direct_control",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=256,
        method_family="direct_target_ssl_control",
    ),
    "p14_ssl512_strong_lhat_chain3": Candidate(
        candidate_id="p14_ssl512_strong_lhat_chain3",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=0.25,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
    ),
    "p15_ssl512_aux150_anchor_strong": Candidate(
        candidate_id="p15_ssl512_aux150_anchor_strong",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        pretrain_logit_anchor_weight=10.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=1.5,
    ),
    "p16_ssl512_aux150_simclr_vicreg": Candidate(
        candidate_id="p16_ssl512_aux150_simclr_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p17_ssl512_aux150_simclr_vicreg006": Candidate(
        candidate_id="p17_ssl512_aux150_simclr_vicreg006",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.06,
    ),
    "p18_ssl512_simclr_vicreg_strong_d19_lhat": Candidate(
        candidate_id="p18_ssl512_simclr_vicreg_strong_d19_lhat",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        method_family="invalid_d19_outer_lhat_override",
    ),
    "p19_ssl512_simclr_vicreg_strong_lhat_chain3": Candidate(
        candidate_id="p19_ssl512_simclr_vicreg_strong_lhat_chain3",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=512,
        auxiliary_alpha=0.25,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
    ),
    "p20_ssl1024_aux150_simclr_vicreg": Candidate(
        candidate_id="p20_ssl1024_aux150_simclr_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=1024,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p21_ssl2048_aux150_simclr_vicreg": Candidate(
        candidate_id="p21_ssl2048_aux150_simclr_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=2048,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p22_ssl4096_aux150_simclr_vicreg": Candidate(
        candidate_id="p22_ssl4096_aux150_simclr_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_source_replay_weight=0.30,
    ),
    "p22m0_current_rng_vicreg": Candidate(
        candidate_id="p22m0_current_rng_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_source_replay_weight=0.30,
    ),
    "p22r1_simclr_no_vicreg": Candidate(
        candidate_id="p22r1_simclr_no_vicreg",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.0,
        pretrain_source_replay_weight=0.30,
    ),
    "p22r2_simclr_no_raw_bce": Candidate(
        candidate_id="p22r2_simclr_no_raw_bce",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.0,
        pretrain_source_replay_weight=0.30,
        raw_auxiliary_scale=0.0,
        vae_random_auxiliary_scale=5.0 / 3.0,
        vae_hard_auxiliary_scale=5.0 / 3.0,
    ),
    "p22r3_simclr_hard_only_bce_screen": Candidate(
        candidate_id="p22r3_simclr_hard_only_bce_screen",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.0,
        pretrain_source_replay_weight=0.30,
        raw_auxiliary_scale=0.0,
        vae_random_auxiliary_scale=0.0,
        vae_hard_auxiliary_scale=10.0 / 3.0,
    ),
    "p22r4_simclr_hard_only_pruned": Candidate(
        candidate_id="p22r4_simclr_hard_only_pruned",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.0,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.30,
        method_family="vae_lhat_post_refine",
    ),
    "p22r5_classic_augmix3_depth13_hard_only_pruned": Candidate(
        candidate_id="p22r5_classic_augmix3_depth13_hard_only_pruned",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        augmix_dirichlet_alpha=1.0,
        augmix_beta_alpha=1.0,
        pretrain_ssl_objective="simclr",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.0,
        pretrain_view_mode=CLASSIC_AUGMIX_VIEW_MODE,
        pretrain_source_replay_weight=0.30,
        method_family="vae_lhat_post_refine",
    ),
    "p23_direct_budget4096": Candidate(
        candidate_id="p23_direct_budget4096",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        pretrain_learning_rate=1.5e-4,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        method_family="direct_budget_filler",
    ),
    "p24_ecgfounder_ssl1024_aux150_simclr_vicreg": Candidate(
        candidate_id="p24_ecgfounder_ssl1024_aux150_simclr_vicreg",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        pretrain_steps=1024,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p25_ecgfounder_ssl4096_aux150_simclr_vicreg": Candidate(
        candidate_id="p25_ecgfounder_ssl4096_aux150_simclr_vicreg",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        pretrain_steps=4096,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p26_ecgfounder_ssl1024_aux100_simclr_vicreg": Candidate(
        candidate_id="p26_ecgfounder_ssl1024_aux100_simclr_vicreg",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        pretrain_steps=1024,
        auxiliary_alpha=1.0,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p27_ecgfounder_blend25_p24": Candidate(
        candidate_id="p27_ecgfounder_blend25_p24",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        method_family="checkpoint_blend",
        checkpoint_blend_alpha=0.25,
    ),
    "p28_ecgfounder_blend50_p24": Candidate(
        candidate_id="p28_ecgfounder_blend50_p24",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        method_family="checkpoint_blend",
        checkpoint_blend_alpha=0.50,
    ),
    "p29_ecgfounder_blend75_p24": Candidate(
        candidate_id="p29_ecgfounder_blend75_p24",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        method_family="checkpoint_blend",
        checkpoint_blend_alpha=0.75,
    ),
    "p30_ecgfounder_ssl2048_aux150_simclr_vicreg": Candidate(
        candidate_id="p30_ecgfounder_ssl2048_aux150_simclr_vicreg",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        pretrain_steps=2048,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p31_ecgfounder_ssl2048_lr5e4_aux150_simclr_vicreg": Candidate(
        candidate_id="p31_ecgfounder_ssl2048_lr5e4_aux150_simclr_vicreg",
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        batch_size=64,
        pretrain_steps=2048,
        pretrain_learning_rate=5.0e-4,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "p32_ssl1024_aux150_simclr_vicreg_lhat010": Candidate(
        candidate_id="p32_ssl1024_aux150_simclr_vicreg_lhat010",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=1024,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        augmax_hard_hull_lambda=0.10,
    ),
    "p33_ssl1024_aux150_simclr_vicreg_lhat015": Candidate(
        candidate_id="p33_ssl1024_aux150_simclr_vicreg_lhat015",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=1024,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        augmax_hard_hull_lambda=0.15,
    ),
    "p34_p22_vae_refine8_lr3e5": Candidate(
        candidate_id="p34_p22_vae_refine8_lr3e5",
        initializer_candidate="p22_ssl4096_aux150_simclr_vicreg",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=8,
        scheduler_horizon_epochs=8,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
    ),
    "s00_effnet_ssl4096_identity_e40": Candidate(
        candidate_id="s00_effnet_ssl4096_identity_e40",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="clean_identity",
        pretrain_source_replay_weight=0.0,
        method_family="direct_target_ssl_control",
        comparison_rng_identity="sequential_effnet_stage1_v1",
    ),
    "s01_effnet_ssl4096_augmix_e40": Candidate(
        candidate_id="s01_effnet_ssl4096_augmix_e40",
        learning_rate=1.5e-4,
        batch_size=64,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.0,
        method_family="direct_target_ssl_control",
        comparison_rng_identity="sequential_effnet_stage1_v1",
    ),
    "s02_effnet_identity_vae_refine8": Candidate(
        candidate_id="s02_effnet_identity_vae_refine8",
        initializer_candidate="s00_effnet_ssl4096_identity_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=8,
        scheduler_horizon_epochs=8,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=0.25,
        method_family="vae_lhat_post_refine",
        comparison_rng_identity="sequential_effnet_stage2_v1",
    ),
    "s03_effnet_augmix_vae_refine8": Candidate(
        candidate_id="s03_effnet_augmix_vae_refine8",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=8,
        scheduler_horizon_epochs=8,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=0.25,
        method_family="vae_lhat_post_refine",
        comparison_rng_identity="sequential_effnet_stage2_v1",
    ),
    "g1i_effnet_identity_vae_eps8_refine8": Candidate(
        candidate_id="g1i_effnet_identity_vae_eps8_refine8",
        initializer_candidate="s00_effnet_ssl4096_identity_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=8,
        scheduler_horizon_epochs=8,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=0.25,
        method_family="vae_lhat_post_refine",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity="sequential_effnet_stage2_v1",
    ),
    "g1a_effnet_augmix_vae_eps8_refine8": Candidate(
        candidate_id="g1a_effnet_augmix_vae_eps8_refine8",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=8,
        scheduler_horizon_epochs=8,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=0.25,
        method_family="vae_lhat_post_refine",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity="sequential_effnet_stage2_v1",
    ),
    "h2i_effnet_identity_vae_clean_polish4": Candidate(
        candidate_id="h2i_effnet_identity_vae_clean_polish4",
        initializer_candidate="s00_effnet_ssl4096_identity_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=1.0,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity="sequential_effnet_stage2_clean_polish_v1",
    ),
    "h2a_effnet_augmix_vae_clean_polish4": Candidate(
        candidate_id="h2a_effnet_augmix_vae_clean_polish4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=1.0,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity="sequential_effnet_stage2_clean_polish_v1",
    ),
    "j3a_effnet_augmix_vae_chain3_polish4": Candidate(
        candidate_id="j3a_effnet_augmix_vae_chain3_polish4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=1.0,
        method_family="vae_lhat_chain3_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity="sequential_effnet_stage2_chain3_polish_v1",
    ),
    "j4a_effnet_augmix_vae_chain3_teacher04_polish4": Candidate(
        candidate_id="j4a_effnet_augmix_vae_chain3_teacher04_polish4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        auxiliary_alpha=1.0,
        method_family="vae_lhat_chain3_teacher04_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_chain3_teacher04_polish_v1"
        ),
        frozen_teacher_mix=0.4,
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "p22r6_no_stage1_source_replay": replace(
        CANDIDATES["p22r4_simclr_hard_only_pruned"],
        candidate_id="p22r6_no_stage1_source_replay",
        pretrain_source_replay_weight=0.0,
    ),
    "p22r7_no_logit_anchors": replace(
        CANDIDATES["p22r4_simclr_hard_only_pruned"],
        candidate_id="p22r7_no_logit_anchors",
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
    ),
    "p22r8_no_vae_lhat_stage2": replace(
        CANDIDATES["p22r4_simclr_hard_only_pruned"],
        candidate_id="p22r8_no_vae_lhat_stage2",
        method_family="direct_target_ssl_control",
    ),
    "p22r9_identity_pretrain_with_vae_lhat": replace(
        CANDIDATES["p22r4_simclr_hard_only_pruned"],
        candidate_id="p22r9_identity_pretrain_with_vae_lhat",
        pretrain_view_mode="clean_identity",
    ),
    "s10_effnet_ssl4096_identity_e20": replace(
        CANDIDATES["s00_effnet_ssl4096_identity_e40"],
        candidate_id="s10_effnet_ssl4096_identity_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        comparison_rng_identity="sequential_effnet_stage1_e20_v1",
    ),
    "s11_effnet_ssl4096_augmix_e20": replace(
        CANDIDATES["s01_effnet_ssl4096_augmix_e40"],
        candidate_id="s11_effnet_ssl4096_augmix_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        comparison_rng_identity="sequential_effnet_stage1_e20_v1",
    ),
    "s12_effnet_ssl8192_identity_e20": replace(
        CANDIDATES["s00_effnet_ssl4096_identity_e40"],
        candidate_id="s12_effnet_ssl8192_identity_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=2,
        pretrain_steps=4096,
        comparison_rng_identity=(
            "sequential_effnet_stage1_ssl8192_e20_v1"
        ),
    ),
    "s13_effnet_ssl8192_augmix_e20": replace(
        CANDIDATES["s01_effnet_ssl4096_augmix_e40"],
        candidate_id="s13_effnet_ssl8192_augmix_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=2,
        pretrain_steps=4096,
        comparison_rng_identity=(
            "sequential_effnet_stage1_ssl8192_e20_v1"
        ),
    ),
    "s14_effnet_ssl8192_shared_direct_e20": replace(
        CANDIDATES["s00_effnet_ssl4096_identity_e40"],
        candidate_id="s14_effnet_ssl8192_shared_direct_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=2,
        pretrain_steps=4096,
        comparison_rng_identity=(
            "sequential_effnet_stage1_ssl8192_e20_v1"
        ),
    ),
    "s15_effnet_ssl4096_augmix_alpha100_e20": replace(
        CANDIDATES["s01_effnet_ssl4096_augmix_e40"],
        candidate_id="s15_effnet_ssl4096_augmix_alpha100_e20",
        epochs=20,
        scheduler_horizon_epochs=20,
        comparison_rng_identity="sequential_effnet_stage1_e20_v1",
        augmix_dirichlet_alpha=1.0,
        augmix_beta_alpha=1.0,
    ),
    "r00_effnet_ssl4096_identity_replay030_e40": replace(
        CANDIDATES["s00_effnet_ssl4096_identity_e40"],
        candidate_id="r00_effnet_ssl4096_identity_replay030_e40",
        pretrain_source_replay_weight=0.30,
    ),
    "r01_effnet_ssl4096_augmix_replay030_e40": replace(
        CANDIDATES["s01_effnet_ssl4096_augmix_e40"],
        candidate_id="r01_effnet_ssl4096_augmix_replay030_e40",
        pretrain_source_replay_weight=0.30,
    ),
    "j4b_effnet_augmix_vae_chain3_teacher04_bnmatched_polish4": replace(
        CANDIDATES["j4a_effnet_augmix_vae_chain3_teacher04_polish4"],
        candidate_id=(
            "j4b_effnet_augmix_vae_chain3_teacher04_bnmatched_polish4"
        ),
    ),
    "k1_effnet_augmix_vae_teacher_distill_polish4": Candidate(
        candidate_id="k1_effnet_augmix_vae_teacher_distill_polish4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=0.25,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_teacher_distill_polish_v1"
        ),
        frozen_teacher_mix=1.0,
    ),
    "k2i_effnet_identity_vae_teacher_distill_lowdose4": Candidate(
        candidate_id="k2i_effnet_identity_vae_teacher_distill_lowdose4",
        initializer_candidate="s00_effnet_ssl4096_identity_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=0.10,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_teacher_distill_lowdose_v1"
        ),
        frozen_teacher_mix=1.0,
    ),
    "k2a_effnet_augmix_vae_teacher_distill_lowdose4": Candidate(
        candidate_id="k2a_effnet_augmix_vae_teacher_distill_lowdose4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=0.10,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_teacher_distill_lowdose_v1"
        ),
        frozen_teacher_mix=1.0,
    ),
    "k3i_effnet_identity_vae_hard_bce_lowdose4": Candidate(
        candidate_id="k3i_effnet_identity_vae_hard_bce_lowdose4",
        initializer_candidate="s00_effnet_ssl4096_identity_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=0.10,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_hard_bce_lowdose_v1"
        ),
        polish_auxiliary_target="hard_label",
    ),
    "k3a_effnet_augmix_vae_hard_bce_lowdose4": Candidate(
        candidate_id="k3a_effnet_augmix_vae_hard_bce_lowdose4",
        initializer_candidate="s01_effnet_ssl4096_augmix_e40",
        learning_rate=3.0e-5,
        batch_size=64,
        epochs=4,
        scheduler_horizon_epochs=4,
        pretrain_epochs=0,
        pretrain_steps=0,
        auxiliary_alpha=0.10,
        method_family="vae_lhat_clean_polish",
        lhat_pgd_epsilon_l2_standardized=8.0,
        comparison_rng_identity=(
            "sequential_effnet_stage2_hard_bce_lowdose_v1"
        ),
        polish_auxiliary_target="hard_label",
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "k4i_effnet_e20_identity_vae_hard_bce_lowdose4": replace(
        CANDIDATES["k3i_effnet_identity_vae_hard_bce_lowdose4"],
        candidate_id="k4i_effnet_e20_identity_vae_hard_bce_lowdose4",
        initializer_candidate="s10_effnet_ssl4096_identity_e20",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_hard_bce_lowdose_v1"
        ),
    ),
    "k4a_effnet_e20_augmix_vae_hard_bce_lowdose4": replace(
        CANDIDATES["k3a_effnet_augmix_vae_hard_bce_lowdose4"],
        candidate_id="k4a_effnet_e20_augmix_vae_hard_bce_lowdose4",
        initializer_candidate="s11_effnet_ssl4096_augmix_e20",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_hard_bce_lowdose_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "k5i_effnet_e20_identity_vae_consistency_lowdose4": replace(
        CANDIDATES["k4i_effnet_e20_identity_vae_hard_bce_lowdose4"],
        candidate_id="k5i_effnet_e20_identity_vae_consistency_lowdose4",
        polish_auxiliary_target="consistency",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_consistency_lowdose_v1"
        ),
    ),
    "k5a_effnet_e20_augmix_vae_consistency_lowdose4": replace(
        CANDIDATES["k4a_effnet_e20_augmix_vae_hard_bce_lowdose4"],
        candidate_id="k5a_effnet_e20_augmix_vae_consistency_lowdose4",
        polish_auxiliary_target="consistency",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_consistency_lowdose_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l6i_effnet_e20_identity_vae_lhat_path8": replace(
        CANDIDATES["k4i_effnet_e20_identity_vae_hard_bce_lowdose4"],
        candidate_id="l6i_effnet_e20_identity_vae_lhat_path8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        polish_auxiliary_target="latent_path_hard_label",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_lhat_path_v1"
        ),
    ),
    "l6a_effnet_e20_augmix_vae_lhat_path8": replace(
        CANDIDATES["k4a_effnet_e20_augmix_vae_hard_bce_lowdose4"],
        candidate_id="l6a_effnet_e20_augmix_vae_lhat_path8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        polish_auxiliary_target="latent_path_hard_label",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_lhat_path_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l7i_effnet_e20_identity_vae_feature8": replace(
        CANDIDATES["k4i_effnet_e20_identity_vae_hard_bce_lowdose4"],
        candidate_id="l7i_effnet_e20_identity_vae_feature8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        polish_auxiliary_target="feature_invariance",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_lhat_feature_v1"
        ),
    ),
    "l7a_effnet_e20_augmix_vae_feature8": replace(
        CANDIDATES["k4a_effnet_e20_augmix_vae_hard_bce_lowdose4"],
        candidate_id="l7a_effnet_e20_augmix_vae_feature8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        polish_auxiliary_target="feature_invariance",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_lhat_feature_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l8ri_effnet_e20_identity_vae_local_anchor_soft8": replace(
        CANDIDATES["k4i_effnet_e20_identity_vae_hard_bce_lowdose4"],
        candidate_id="l8ri_effnet_e20_identity_vae_local_anchor_soft8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        lhat_pgd_epsilon_l2_standardized=2.0,
        polish_auxiliary_target="local_anchor_soft",
        local_lhat_geometry_preset="local_anchor_soft_v1",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_local_anchor_soft_v1"
        ),
    ),
    "l8ra_effnet_e20_augmix_vae_local_anchor_soft8": replace(
        CANDIDATES["k4a_effnet_e20_augmix_vae_hard_bce_lowdose4"],
        candidate_id="l8ra_effnet_e20_augmix_vae_local_anchor_soft8",
        epochs=8,
        scheduler_horizon_epochs=8,
        auxiliary_alpha=0.25,
        lhat_pgd_epsilon_l2_standardized=2.0,
        polish_auxiliary_target="local_anchor_soft",
        local_lhat_geometry_preset="local_anchor_soft_v1",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_local_anchor_soft_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l9ri_effnet_e20_identity_vae_calibrated_anchor_soft8": replace(
        CANDIDATES["l8ri_effnet_e20_identity_vae_local_anchor_soft8"],
        candidate_id="l9ri_effnet_e20_identity_vae_calibrated_anchor_soft8",
        lhat_pgd_epsilon_l2_standardized=4.0,
        local_lhat_geometry_preset="historical_gate2_calibrated_v1",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_calibrated_anchor_soft_v1"
        ),
    ),
    "l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8": replace(
        CANDIDATES["l8ra_effnet_e20_augmix_vae_local_anchor_soft8"],
        candidate_id="l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8",
        lhat_pgd_epsilon_l2_standardized=4.0,
        local_lhat_geometry_preset="historical_gate2_calibrated_v1",
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_calibrated_anchor_soft_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l10ri_effnet_e20_identity_vae_bounded_augmax6": Candidate(
        candidate_id="l10ri_effnet_e20_identity_vae_bounded_augmax6",
        initializer_candidate="s10_effnet_ssl4096_identity_e20",
        learning_rate=5.0e-5,
        batch_size=128,
        epochs=6,
        scheduler_horizon_epochs=40,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        method_family="vae_lhat_bounded_augmax_polish",
        lhat_pgd_epsilon_l2_standardized=10.0,
        lhat_geometry_preset=(
            "k500_internal_bounded_augmax_lam35_eps10_v1"
        ),
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_bounded_augmax_v1"
        ),
    ),
    "l10ra_effnet_e20_augmix_vae_bounded_augmax6": Candidate(
        candidate_id="l10ra_effnet_e20_augmix_vae_bounded_augmax6",
        initializer_candidate="s11_effnet_ssl4096_augmix_e20",
        learning_rate=5.0e-5,
        batch_size=128,
        epochs=6,
        scheduler_horizon_epochs=40,
        pretrain_epochs=0,
        pretrain_steps=0,
        pretrain_logit_anchor_weight=0.0,
        supervised_logit_anchor_weight=0.0,
        method_family="vae_lhat_bounded_augmax_polish",
        lhat_pgd_epsilon_l2_standardized=10.0,
        lhat_geometry_preset=(
            "k500_internal_bounded_augmax_lam35_eps10_v1"
        ),
        comparison_rng_identity=(
            "sequential_effnet_stage2_e20_bounded_augmax_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l11c_effnet_ssl4096_identity_vae_mild_chain3_e20": Candidate(
        candidate_id="l11c_effnet_ssl4096_identity_vae_mild_chain3_e20",
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=0.25,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="clean_identity",
        pretrain_source_replay_weight=0.0,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
        comparison_rng_identity="sequential_effnet_stage1_e20_v1",
    ),
    "l11d_effnet_ssl4096_augmix_vae_mild_chain3_e20": Candidate(
        candidate_id="l11d_effnet_ssl4096_augmix_vae_mild_chain3_e20",
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=0.25,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.0,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
        comparison_rng_identity="sequential_effnet_stage1_e20_v1",
    ),
    "l12d_effnet_ssl4096_augmix_replay030_vae_mild_chain3_e20": Candidate(
        candidate_id=(
            "l12d_effnet_ssl4096_augmix_replay030_vae_mild_chain3_e20"
        ),
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=0.25,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.30,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
        comparison_rng_identity=(
            "sequential_effnet_stage1_replay030_e20_v1"
        ),
    ),
    "l13d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20": Candidate(
        candidate_id=(
            "l13d_effnet_ssl4096_augmix_replay030_"
            "vae_gradbalanced_chain3_e20"
        ),
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=0.75,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.30,
        lhat_pgd_epsilon_l2_standardized=8.0,
        method_family="strong_lhat_augmix_chain3",
        comparison_rng_identity=(
            "sequential_effnet_stage1_replay030_e20_v1"
        ),
        pcgrad_auxiliary_norm_ratio_cap=1.0,
    ),
    "l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20": Candidate(
        candidate_id=(
            "l14d_effnet_ssl4096_augmix_replay030_"
            "vae_diverse_d19_e20"
        ),
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.30,
        raw_auxiliary_scale=1.0,
        vae_random_auxiliary_scale=1.0,
        vae_hard_auxiliary_scale=1.0,
        jsd_auxiliary_scale=1.0,
        method_family="d19_separate_views",
        comparison_rng_identity=(
            "sequential_effnet_stage1_replay030_d19_e20_v1"
        ),
    ),
    "l15d_effnet_ssl4096_augmix_replay030_stage2replay010_vae_diverse_d19_e20": Candidate(
        candidate_id=(
            "l15d_effnet_ssl4096_augmix_replay030_"
            "stage2replay010_vae_diverse_d19_e20"
        ),
        learning_rate=1.5e-4,
        batch_size=64,
        epochs=20,
        scheduler_horizon_epochs=20,
        pretrain_epochs=1,
        pretrain_steps=4096,
        pretrain_learning_rate=1.0e-3,
        pretrain_logit_anchor_weight=5.0,
        supervised_logit_anchor_weight=2.0,
        auxiliary_alpha=1.5,
        pretrain_ssl_objective="simclr_vicreg",
        pretrain_ssl_weight=1.0,
        pretrain_vicreg_mix_weight=0.03,
        pretrain_view_mode="twochain_augmix",
        pretrain_source_replay_weight=0.30,
        pretrain_source_batches_per_step=1,
        supervised_source_replay_weight=0.10,
        raw_auxiliary_scale=1.0,
        vae_random_auxiliary_scale=1.0,
        vae_hard_auxiliary_scale=1.0,
        jsd_auxiliary_scale=1.0,
        method_family="d19_separate_views",
        comparison_rng_identity=(
            "sequential_effnet_stage1_replay030_d19_e20_v1"
        ),
    ),
}
CANDIDATES = {
    **CANDIDATES,
    "l18d_ecgfounder_ssl4096_augmix_replay030_vae_diverse_d19_e20": replace(
        CANDIDATES[
            "l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20"
        ],
        candidate_id=(
            "l18d_ecgfounder_ssl4096_augmix_replay030_"
            "vae_diverse_d19_e20"
        ),
        model_name="ecgfounder",
        learning_rate=2.0e-5,
        comparison_rng_identity=(
            "sequential_ecgfounder_stage1_replay030_d19_e20_v1"
        ),
    ),
    "l16d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20_seed1": replace(
        CANDIDATES[
            "l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20"
        ],
        candidate_id=(
            "l16d_effnet_ssl4096_augmix_replay030_"
            "vae_diverse_d19_e20_seed1"
        ),
        replicate_id=1,
    ),
    "l17d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20_seed1_fullrng": (
        replace(
            CANDIDATES[
                "l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20"
            ],
            candidate_id=(
                "l17d_effnet_ssl4096_augmix_replay030_"
                "vae_diverse_d19_e20_seed1_fullrng"
            ),
            replicate_id=1,
        )
    ),
    "l19d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed1": (
        replace(
            CANDIDATES[
                "l13d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20"
            ],
            candidate_id=(
                "l19d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20_seed1"
            ),
            replicate_id=1,
        )
    ),
    "l20d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed2": (
        replace(
            CANDIDATES[
                "l13d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20"
            ],
            candidate_id=(
                "l20d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20_seed2"
            ),
            replicate_id=2,
        )
    ),
    "l21d_effnet_ssl4096_augmix_replay030_vae_uniform_chain3_e20_seed2": (
        replace(
            CANDIDATES[
                "l13d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20"
            ],
            candidate_id=(
                "l21d_effnet_ssl4096_augmix_replay030_"
                "vae_uniform_chain3_e20_seed2"
            ),
            lhat_search_mode="uniform_compute_matched",
            replicate_id=2,
        )
    ),
    "l22d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed2_replay": (
        replace(
            CANDIDATES[
                "l13d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20"
            ],
            candidate_id=(
                "l22d_effnet_ssl4096_augmix_replay030_"
                "vae_gradbalanced_chain3_e20_seed2_replay"
            ),
            replicate_id=2,
        )
    ),
    "l23d_effnet_ssl4096_augmix_replay030_vae_direct_uniform_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l23d_effnet_ssl4096_augmix_replay030_"
                "vae_direct_uniform_e40"
            ),
            lhat_search_mode="uniform_compute_matched",
        )
    ),
    "l24d_effnet_ssl4096_augmix_replay030_vae_direct_localhard_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l24d_effnet_ssl4096_augmix_replay030_"
                "vae_direct_localhard_e40"
            ),
            lhat_geometry_preset="lowdose_h010_s3_v1",
        )
    ),
    "l25d_effnet_ssl4096_augmix_replay030_vae_direct_localuniform_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l25d_effnet_ssl4096_augmix_replay030_"
                "vae_direct_localuniform_e40"
            ),
            lhat_geometry_preset="lowdose_h010_s3_v1",
            lhat_search_mode="uniform_compute_matched",
        )
    ),
    "l26d_effnet_ssl4096_augmix_replay030_vae_direct_balancedhard_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l26d_effnet_ssl4096_augmix_replay030_"
                "vae_direct_balancedhard_e40"
            ),
            lhat_geometry_preset="lowdose_h010_s3_v1",
            lhat_attack_objective=(
                "maximize_equal_positive_negative_bce_with_logits"
            ),
        )
    ),
    "l27d_effnet_ssl4096_augmix_replay030_vae_direct_vathard_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l27d_effnet_ssl4096_augmix_replay030_"
                "vae_direct_vathard_e40"
            ),
            lhat_geometry_preset="lowdose_h010_s3_v1",
            lhat_attack_objective=(
                "maximize_bernoulli_kl_from_decoded_anchor"
            ),
        )
    ),
    "l28d_ecgfounder_ssl1024_augmix_replay030_no_vae_e40": (
        replace(
            CANDIDATES["p22r8_no_vae_lhat_stage2"],
            candidate_id=(
                "l28d_ecgfounder_ssl1024_augmix_replay030_"
                "no_vae_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l29d_ecgfounder_ssl1024_augmix_replay030_vae_hard_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l29d_ecgfounder_ssl1024_augmix_replay030_"
                "vae_hard_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l30d_ecgfounder_ssl1024_augmix_replay030_clean_only_e40": (
        replace(
            CANDIDATES["p22r8_no_vae_lhat_stage2"],
            candidate_id=(
                "l30d_ecgfounder_ssl1024_augmix_replay030_"
                "clean_only_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            auxiliary_alpha=1.5,
            clean_direct_sum_auxiliary=True,
            method_family="clean_target_ssl_control",
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l31d_ecgfounder_ssl1024_augmix_replay030_clean_vae_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l31d_ecgfounder_ssl1024_augmix_replay030_"
                "clean_vae_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            auxiliary_alpha=1.5,
            clean_direct_sum_auxiliary=True,
            method_family="vae_lhat_clean_polish",
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l32d_ecgfounder_ssl1024_augmix_replay030_clean_vae_a050_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l32d_ecgfounder_ssl1024_augmix_replay030_"
                "clean_vae_a050_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            auxiliary_alpha=0.5,
            clean_direct_sum_auxiliary=True,
            method_family="vae_lhat_clean_polish",
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l33d_ecgfounder_ssl1024_augmix_replay030_clean_vae_a100_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l33d_ecgfounder_ssl1024_augmix_replay030_"
                "clean_vae_a100_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            auxiliary_alpha=1.0,
            clean_direct_sum_auxiliary=True,
            method_family="vae_lhat_clean_polish",
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "pruned_lhat_pair_e40_v1"
            ),
        )
    ),
    "l34d_ecgfounder_ssl1024_augmix_replay030_rot4_no_vae_e40": (
        replace(
            CANDIDATES["p22r8_no_vae_lhat_stage2"],
            candidate_id=(
                "l34d_ecgfounder_ssl1024_augmix_replay030_"
                "rot4_no_vae_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            supervised_corruption_schedule=ROTATING4_SUPERVISED_SCHEDULE,
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "rotating4_pair_e40_v1"
            ),
        )
    ),
    "l35d_ecgfounder_ssl1024_augmix_replay030_rot4_vae_hard_e40": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l35d_ecgfounder_ssl1024_augmix_replay030_"
                "rot4_vae_hard_e40"
            ),
            model_name="ecgfounder",
            learning_rate=2.0e-5,
            pretrain_steps=1024,
            supervised_corruption_schedule=ROTATING4_SUPERVISED_SCHEDULE,
            comparison_rng_identity=(
                "sequential_ecgfounder_ssl1024_replay030_"
                "rotating4_pair_e40_v1"
            ),
        )
    ),
    "l36d_effnet_ssl1024_augmix_replay030_rot4_no_vae_h30_e23": (
        replace(
            CANDIDATES["p22r8_no_vae_lhat_stage2"],
            candidate_id=(
                "l36d_effnet_ssl1024_augmix_replay030_"
                "rot4_no_vae_h30_e23"
            ),
            model_name="efficientnet1dv2",
            learning_rate=5.0e-5,
            batch_size=128,
            epochs=23,
            scheduler_horizon_epochs=30,
            pretrain_steps=1024,
            supervised_corruption_schedule=ROTATING4_SUPERVISED_SCHEDULE,
            comparison_rng_identity=(
                "sequential_effnet_ssl1024_replay030_"
                "rotating4_pair_h30_e23_v1"
            ),
        )
    ),
    "l37d_effnet_ssl1024_augmix_replay030_rot4_vae_hard_h30_e23": (
        replace(
            CANDIDATES["p22r4_simclr_hard_only_pruned"],
            candidate_id=(
                "l37d_effnet_ssl1024_augmix_replay030_"
                "rot4_vae_hard_h30_e23"
            ),
            model_name="efficientnet1dv2",
            learning_rate=5.0e-5,
            batch_size=128,
            epochs=23,
            scheduler_horizon_epochs=30,
            pretrain_steps=1024,
            supervised_corruption_schedule=ROTATING4_SUPERVISED_SCHEDULE,
            comparison_rng_identity=(
                "sequential_effnet_ssl1024_replay030_"
                "rotating4_pair_h30_e23_v1"
            ),
        )
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _local_anchor_soft_geometry(candidate: Candidate) -> dict[str, Any]:
    if candidate.polish_auxiliary_target != "local_anchor_soft":
        raise ValueError(
            "local anchor-soft geometry is only valid for local_anchor_soft"
        )
    preset = candidate.local_lhat_geometry_preset
    geometries = {
        "local_anchor_soft_v1": {
            "preset": "local_anchor_soft_v1",
            "hull_lambda": 0.05,
            "steps": 3,
            "learning_rate": 0.25,
            "pgd_epsilon_l2_standardized": 2.0,
            "minimum_effective_anchor_mass": 0.95,
        },
        "historical_gate2_calibrated_v1": {
            "preset": "historical_gate2_calibrated_v1",
            "hull_lambda": 0.20,
            "steps": 10,
            "learning_rate": 0.40,
            "pgd_epsilon_l2_standardized": 4.0,
            "minimum_effective_anchor_mass": 0.80,
        },
    }
    if preset not in geometries:
        raise ValueError(
            "local anchor-soft geometry must use one frozen preset: "
            f"{sorted(geometries)}, got {preset!r}"
        )
    geometry = copy.deepcopy(geometries[preset])
    if not math.isclose(
        candidate.lhat_pgd_epsilon_l2_standardized,
        geometry["pgd_epsilon_l2_standardized"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "candidate epsilon does not match its frozen local-LHAT geometry"
        )
    return geometry


def _local_anchor_soft_method_ids(candidate: Candidate) -> tuple[str, str]:
    geometry = _local_anchor_soft_geometry(candidate)
    if geometry["preset"] == "local_anchor_soft_v1":
        return (
            VAE_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID,
            CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        )
    return (
        VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID,
        CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
    )


def _bounded_augmax_geometry(candidate: Candidate) -> dict[str, Any]:
    if candidate.method_family != "vae_lhat_bounded_augmax_polish":
        raise ValueError(
            "bounded AugMax geometry is restricted to its frozen method family"
        )
    expected_preset = "k500_internal_bounded_augmax_lam35_eps10_v1"
    if candidate.lhat_geometry_preset != expected_preset:
        raise ValueError(
            "bounded AugMax must use the frozen K500-internal geometry preset"
        )
    geometry = {
        "preset": expected_preset,
        "hull_lambda": 0.35,
        "steps": 10,
        "learning_rate": 0.25,
        "pgd_epsilon_l2_standardized": 10.0,
    }
    if not math.isclose(
        candidate.lhat_pgd_epsilon_l2_standardized,
        geometry["pgd_epsilon_l2_standardized"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "bounded AugMax candidate epsilon drifted from the frozen geometry"
        )
    return geometry


def _lhat_config_for(candidate: Candidate) -> dict[str, Any]:
    if candidate.lhat_attack_objective not in LHAT_ATTACK_OBJECTIVES:
        raise ValueError(
            f"unsupported candidate LHAT objective: "
            f"{candidate.lhat_attack_objective!r}"
        )
    config = _load_yaml(
        STRONG_LHAT_CONFIG
        if candidate.method_family == "strong_lhat_augmix_chain3"
        else BASE_LHAT_CONFIG
    )
    config["hull_attack"]["pgd_epsilon_l2_standardized"] = (
        candidate.lhat_pgd_epsilon_l2_standardized
    )
    if candidate.method_family == "vae_lhat_bounded_augmax_polish":
        geometry = _bounded_augmax_geometry(candidate)
        config["candidates"].update(
            {
                "label_policy": "exact_positive_set",
                "mode": "nearest",
                "local_pool_size": 80,
                "num_candidates": 20,
                "include_anchor": False,
                "require_distinct": True,
                "require_non_self": True,
                "forbid_norm_abnormal_mix": True,
                "validation_records_allowed": False,
            }
        )
        config["hull_attack"].update(
            {
                "weight_mode": "optimized_softmax",
                "init_logit_gap": 0.0,
                "hull_lambda": geometry["hull_lambda"],
                "steps": geometry["steps"],
                "learning_rate": geometry["learning_rate"],
                "pgd_epsilon_l2_standardized": geometry[
                    "pgd_epsilon_l2_standardized"
                ],
                "objective": "maximize_multilabel_bce_with_logits",
                "optimizer": "adam",
            }
        )
    elif candidate.polish_auxiliary_target == "local_anchor_soft":
        geometry = _local_anchor_soft_geometry(candidate)
        config["candidates"].update(
            {
                "label_policy": "exact_positive_set",
                "mode": "nearest",
                "local_pool_size": 80,
                "num_candidates": 20,
                "include_anchor": False,
                "require_distinct": True,
                "require_non_self": True,
                "forbid_norm_abnormal_mix": True,
                "validation_records_allowed": False,
            }
        )
        config["hull_attack"].update(
            {
                "weight_mode": "optimized_softmax",
                "init_logit_gap": 0.0,
                "hull_lambda": geometry["hull_lambda"],
                "steps": geometry["steps"],
                "learning_rate": geometry["learning_rate"],
                "pgd_epsilon_l2_standardized": geometry[
                    "pgd_epsilon_l2_standardized"
                ],
                "objective": "maximize_multilabel_bce_with_logits",
                "optimizer": "adam",
            }
        )
    elif (
        candidate.method_family == "vae_lhat_post_refine"
        and candidate.lhat_geometry_preset is not None
    ):
        if candidate.lhat_geometry_preset != "lowdose_h010_s3_v1":
            raise ValueError(
                "VAE-LHAT direct supervision supports only the frozen "
                "lowdose_h010_s3_v1 geometry preset"
            )
        config["hull_attack"].update(
            {
                "weight_mode": "optimized_softmax",
                "init_logit_gap": 0.0,
                "hull_lambda": 0.10,
                "steps": 3,
                "learning_rate": 0.25,
                "pgd_epsilon_l2_standardized": 2.0,
                "objective": "maximize_multilabel_bce_with_logits",
                "optimizer": "adam",
            }
        )
    config["hull_attack"]["objective"] = candidate.lhat_attack_objective
    if candidate.lhat_search_mode == "uniform_compute_matched":
        if candidate.method_family not in {
            "strong_lhat_augmix_chain3",
            "vae_lhat_post_refine",
        }:
            raise ValueError(
                "the uniform compute-matched LHAT control is restricted to "
                "the strong chain-three or direct-supervision LHAT families"
            )
        config["hull_attack"].update(
            {
                "weight_mode": "uniform_compute_matched",
                # Keep all five decoder/classifier/backward passes while making
                # the float32 softmax logits numerically stationary.
                "learning_rate": 1.0e-30,
                "objective": "compute_matched_uniform_latent_control",
            }
        )
    elif candidate.lhat_search_mode != "optimized_softmax":
        raise ValueError(
            f"unsupported LHAT search mode: {candidate.lhat_search_mode!r}"
        )
    return config


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(value),
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_argument(arguments: list[Any], flag: str, value: Any) -> None:
    positions = [i for i, item in enumerate(arguments) if str(item) == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"expected exactly one {flag} argument")
    arguments[positions[0] + 1] = str(value)


def _remove_argument(arguments: list[Any], flag: str) -> None:
    positions = [i for i, item in enumerate(arguments) if str(item) == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"expected exactly one {flag} argument")
    del arguments[positions[0] : positions[0] + 2]


def _argument_value(arguments: Sequence[Any], flag: str) -> str:
    positions = [i for i, item in enumerate(arguments) if str(item) == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"expected exactly one {flag} argument")
    return str(arguments[positions[0] + 1])


def _snapshot_evidence(
    run_record: Mapping[str, Any],
    path: Path,
    *,
    role: str,
    snapshot_root: Path | None = None,
    file_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshots = run_record.get("config_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("run manifest does not contain config snapshots")
    resolved = path.resolve()
    matches = [
        record
        for record in snapshots
        if isinstance(record, Mapping)
        and record.get("role") == role
        and Path(str(record.get("source_path", ""))).resolve() == resolved
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {role} snapshot for {path}, found {len(matches)}"
        )
    record = matches[0]
    current_sha256 = _sha256(path)
    if record.get("sha256") != current_sha256:
        raise ValueError(f"runtime snapshot is stale for {path}")
    snapshot_path_raw = record.get("snapshot_path")
    if not isinstance(snapshot_path_raw, str) or not snapshot_path_raw:
        raise ValueError(f"runtime snapshot path is missing for {path}")
    snapshot_path = Path(snapshot_path_raw)
    if snapshot_path.is_absolute() or ".." in snapshot_path.parts:
        raise ValueError(f"runtime snapshot path escapes its run root: {snapshot_path}")
    if snapshot_root is not None:
        artifact = snapshot_root / snapshot_path
        root_resolved = snapshot_root.resolve()
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or not artifact.resolve().is_relative_to(root_resolved)
        ):
            raise ValueError(f"runtime snapshot artifact is missing: {artifact}")
        if _sha256(artifact) != current_sha256:
            raise ValueError(f"runtime snapshot artifact hash drifted: {artifact}")
        if int(record.get("size_bytes", -1)) != artifact.stat().st_size:
            raise ValueError(f"runtime snapshot artifact size drifted: {artifact}")
        if file_index is None:
            raise ValueError("snapshot artifact validation requires a run file index")
        indexed = file_index.get(snapshot_path.as_posix())
        if (
            not isinstance(indexed, Mapping)
            or indexed.get("role") != "config_snapshot"
            or indexed.get("sha256") != current_sha256
            or int(indexed.get("size_bytes", -1)) != artifact.stat().st_size
        ):
            raise ValueError(f"run file index does not bind snapshot: {artifact}")
    return {
        "path": str(path),
        "role": role,
        "sha256": current_sha256,
        "snapshot_path": snapshot_path.as_posix(),
    }


def _run_file_index_evidence(
    train_root: Path,
    run_record: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    path = train_root / "run_file_index.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run file index: {path}")
    sha256 = _sha256(path)
    if run_record.get("run_file_index_sha256") != sha256:
        raise ValueError("run manifest does not bind its run file index")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(files, list):
        raise ValueError("run file index has an unsupported schema")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("run file index contains an invalid record")
        key = str(record["path"])
        if key in indexed:
            raise ValueError(f"run file index contains a duplicate path: {key}")
        indexed[key] = record
    return indexed, {"path": str(path), "sha256": sha256}


def _indexed_artifact_evidence(
    train_root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    relative_path: str,
) -> dict[str, Any]:
    path = train_root / relative_path
    record = file_index.get(relative_path)
    if not path.is_file() or not isinstance(record, Mapping):
        raise ValueError(f"run file index is missing artifact: {relative_path}")
    sha256 = _sha256(path)
    if (
        record.get("role") != "delegate_artifact"
        or record.get("sha256") != sha256
        or int(record.get("size_bytes", -1)) != path.stat().st_size
    ):
        raise ValueError(f"run file index artifact drifted: {relative_path}")
    return {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
    }


def _candidate_relroot(candidate: Candidate) -> Path:
    return Path("generated_search") / "a7_global_search_20260726" / candidate.candidate_id


def _candidate_root(candidate: Candidate) -> Path:
    return CONFIG_ROOT / _candidate_relroot(candidate)


def _persistent_root(candidate: Candidate) -> Path:
    return RUN_ROOT / candidate.candidate_id


def _base_a7_root(candidate: Candidate) -> Path:
    if candidate.model_name not in SOURCE_CHECKPOINTS:
        raise ValueError(f"unsupported model: {candidate.model_name}")
    return BASE_A7_PARENT / candidate.model_name


def _source_checkpoint(candidate: Candidate, center: str | None = None) -> Path:
    if candidate.initializer_candidate is None:
        return SOURCE_CHECKPOINTS[candidate.model_name]
    if center not in CENTERS:
        raise ValueError(
            "a candidate initializer requires an explicit logical center"
        )
    return (
        RUN_ROOT
        / candidate.initializer_candidate
        / "candidate"
        / str(center)
        / "train"
        / "training"
        / "checkpoints"
        / "last.pt"
    )


def _validated_source_loader_identity(
    replay: Mapping[str, Any],
    *,
    enabled: bool,
    center: str,
    model_name: str,
    replicate_id: int = 0,
) -> dict[str, Any]:
    raw_loader = replay.get("loader")
    if not enabled:
        return dict(raw_loader) if isinstance(raw_loader, Mapping) else {}
    if not isinstance(raw_loader, Mapping) or not raw_loader:
        raise ValueError("replay-on diagnostics omit source loader identity")
    dataset = raw_loader.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("source loader identity omits its dataset contract")
    cache = dataset.get("cache")
    selection = dataset.get("selection")
    if not isinstance(cache, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("source loader identity omits cache/selection evidence")
    cache_drift = {
        key: (cache.get(key), expected)
        for key, expected in {
            "dataset": "ptbxl",
            "record_count": 21_799,
            "sampling_rate_hz": 100,
            "duration_seconds": 10.0,
            "signal_shape": [21_799, 1000, 12],
            "lead_order": list(PTBXL_LEAD_ORDER),
            "class_order": list(CLASS_ORDER),
            "physical_unit": "mV",
            "normalization": "none",
            "storage_mode": "ram",
        }.items()
        if cache.get(key) != expected
    }
    selection_drift = {
        key: (selection.get(key), expected)
        for key, expected in {
            "dataset": "ptbxl",
            "cache_dataset": "ptbxl",
            "partition": "train",
            "record_count": 17_084,
            "split_id": "ptbxl_super5_official_folds_v1",
            "class_order": list(CLASS_ORDER),
            "mapping_version": None,
            "mapping_hash": None,
            "ref_excluded_evaluation": False,
        }.items()
        if selection.get(key) != expected
    }
    if cache_drift or selection_drift:
        raise ValueError(
            "source loader PTB-XL contract drifted: "
            f"cache={cache_drift}, selection={selection_drift}"
        )
    for label, value in {
        "cache manifest": cache.get("manifest_sha256"),
        "split manifest": selection.get("split_manifest_sha256"),
        "source manifest": selection.get("source_manifest_sha256"),
        "hash-id set": selection.get("hash_id_set_sha256"),
    }.items():
        if not _is_sha256(value):
            raise ValueError(f"source loader {label} is not a SHA256")
    if selection.get("source_manifest_sha256") != cache.get("manifest_sha256"):
        raise ValueError("source loader cache/source manifest identity differs")
    if (
        raw_loader.get("batch_size") != 64
        or raw_loader.get("num_workers") != 0
        or raw_loader.get("drop_last") is not True
    ):
        raise ValueError("source loader batching contract drifted")
    seed = raw_loader.get("seed")
    expected_namespace = (
        f"target_ssl_source_semantic_v1:{center}:{model_name}:"
        f"replicate{replicate_id}"
    )
    if (
        not isinstance(seed, Mapping)
        or seed.get("namespace") != expected_namespace
        or not _is_sha256(seed.get("config_sha256"))
    ):
        raise ValueError("source loader seed identity drifted")
    stable_loader = dict(raw_loader)
    stable_dataset = dict(dataset)
    stable_cache = dict(cache)
    # This is runtime telemetry, not part of the replay population or order.
    # Concurrent arms legitimately observe different free-memory values.
    stable_cache.pop("available_memory_at_open_bytes", None)
    stable_dataset["cache"] = stable_cache
    stable_loader["dataset"] = stable_dataset
    return stable_loader


def _parent_evidence(candidate: Candidate, center: str) -> dict[str, Any]:
    """Resolve the exact K500/run identity behind a continuation checkpoint."""

    if candidate.initializer_candidate is None:
        raise ValueError("parent evidence requires an initializer candidate")
    parent = CANDIDATES.get(candidate.initializer_candidate)
    if parent is None:
        raise ValueError(
            f"unknown initializer candidate: {candidate.initializer_candidate}"
        )
    if parent.method_family != "direct_target_ssl_control":
        raise ValueError(
            "sequential continuation requires a VAE-free "
            "direct_target_ssl_control initializer"
        )
    checkpoint = _source_checkpoint(candidate, center)
    parent_manifest = _candidate_root(parent) / "manifest.json"
    train_root = (
        _persistent_root(parent) / "candidate" / center / "train"
    )
    train_result = train_root / "training" / "train_result.json"
    teacher_diagnostics = (
        train_root / "training" / "unlabeled_teacher_diagnostics.json"
    )
    training_history = train_root / "training" / "training_history.json"
    run_manifest = train_root / "run_manifest.json"
    run_file_index = train_root / "run_file_index.json"
    required = (
        checkpoint,
        parent_manifest,
        train_result,
        teacher_diagnostics,
        training_history,
        run_manifest,
        run_file_index,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "continuation initializer evidence is incomplete: " + ", ".join(missing)
        )
    result = json.loads(train_result.read_text(encoding="utf-8"))
    diagnostics = json.loads(teacher_diagnostics.read_text(encoding="utf-8"))
    history = json.loads(training_history.read_text(encoding="utf-8"))
    run_record = json.loads(run_manifest.read_text(encoding="utf-8"))
    parent_method_path = _candidate_root(parent) / "methods" / f"{center}.yaml"
    parent_online_path = _candidate_root(parent) / "online" / f"{center}.yaml"
    parent_experiment_path = (
        _candidate_root(parent)
        / "experiments"
        / "candidate"
        / f"{center}_train.yaml"
    )
    parent_method = _load_yaml(parent_method_path)
    parent_manifest_record = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if (
        run_record.get("status") != "complete"
        or run_record.get("exit_code") != 0
        or run_record.get("error") is not None
    ):
        raise ValueError("initializer run manifest is not a successful completed run")
    forbidden_nodes = {"lhat", "threechain_augmix"}.intersection(
        parent_method.get("nodes", {})
    )
    forbidden_resources = {
        "vae",
        "latent_pool",
        "vae_decoder",
        "lhat_config",
    }.intersection(parent_method.get("resources", {}))
    if forbidden_nodes or forbidden_resources:
        raise ValueError(
            "sequential Stage-1 initializer is VAE-exposed: "
            f"nodes={sorted(forbidden_nodes)}, "
            f"resources={sorted(forbidden_resources)}"
        )
    delegate_argv = [str(value) for value in run_record.get("exact_delegate_argv", [])]
    if "--vae-checkpoint" in delegate_argv:
        raise ValueError("sequential Stage-1 initializer received a VAE checkpoint")
    expected_arguments = {
        "--config": parent_online_path,
        "--config-root": CONFIG_ROOT,
        "--output-dir": train_root / "training",
        "--model": parent.model_name,
        "--method-config": (
            _candidate_relroot(parent) / "methods" / f"{center}.yaml"
        ),
        "--center": center,
        "--source-checkpoint": SOURCE_CHECKPOINTS[parent.model_name],
        "--epochs": parent.epochs,
        "--scheduler-horizon-epochs": parent.scheduler_horizon_epochs,
        "--trainable-scope": "full",
        "--num-workers": 0,
        "--learning-rate": parent.learning_rate,
        "--weight-decay": parent.weight_decay,
        "--batch-size": parent.batch_size,
    }
    path_arguments = {
        "--config",
        "--config-root",
        "--output-dir",
        "--source-checkpoint",
    }
    argument_drift: dict[str, tuple[str, str]] = {}
    for flag, expected in expected_arguments.items():
        actual = _argument_value(delegate_argv, flag)
        matches = (
            Path(actual).resolve() == Path(str(expected)).resolve()
            if flag in path_arguments
            else actual == str(expected)
        )
        if not matches:
            argument_drift[flag] = (actual, str(expected))
    if argument_drift:
        raise ValueError(f"initializer delegate argv drifted: {argument_drift}")
    source_record = parent_manifest_record.get("source_checkpoint")
    expected_source = SOURCE_CHECKPOINTS[parent.model_name]
    if not isinstance(source_record, Mapping) or (
        source_record.get("path") != str(expected_source)
        or source_record.get("sha256") != _sha256(expected_source)
    ):
        raise ValueError("initializer manifest does not bind the locked source model")
    file_index, file_index_evidence = _run_file_index_evidence(
        train_root,
        run_record,
    )
    snapshot_closure = {
        "experiment": _snapshot_evidence(
            run_record,
            parent_experiment_path,
            role="experiment_entry_config",
            snapshot_root=train_root,
            file_index=file_index,
        ),
        "online": _snapshot_evidence(
            run_record,
            parent_online_path,
            role="delegate_entry_config",
            snapshot_root=train_root,
            file_index=file_index,
        ),
        "method": _snapshot_evidence(
            run_record,
            parent_method_path,
            role="referenced_config",
            snapshot_root=train_root,
            file_index=file_index,
        ),
        "source_registry": _snapshot_evidence(
            run_record,
            LOCKED_SOURCE_REGISTRY,
            role="referenced_config",
            snapshot_root=train_root,
            file_index=file_index,
        ),
        "run_file_index": file_index_evidence,
    }
    artifact_closure = {
        name: _indexed_artifact_evidence(train_root, file_index, relative_path)
        for name, relative_path in {
            "checkpoint": "training/checkpoints/last.pt",
            "train_result": "training/train_result.json",
            "training_history": "training/training_history.json",
            "teacher_diagnostics": (
                "training/unlabeled_teacher_diagnostics.json"
            ),
        }.items()
    }
    diagnostic_config = diagnostics.get("config", {})
    if diagnostic_config.get("pretrain_view_mode") != parent.pretrain_view_mode:
        raise ValueError("initializer diagnostics report the wrong Stage-1 view mode")
    expected_policy = PRETRAIN_POLICY_BY_VIEW_MODE[parent.pretrain_view_mode]
    if diagnostic_config.get("policy") != expected_policy:
        raise ValueError("initializer diagnostics report the wrong Stage-1 policy")
    expected_replay_weight = float(parent.pretrain_source_replay_weight or 0.0)
    expected_source_batches = parent.pretrain_source_batches_per_step
    if expected_replay_weight not in {0.0, 0.30}:
        raise ValueError(
            "sequential Stage-1 initializer uses an unregistered source replay "
            f"weight: {expected_replay_weight}"
        )
    if (
        diagnostic_config.get("pretrain_source_replay_weight")
        != expected_replay_weight
        or diagnostic_config.get("pretrain_source_batches_per_step")
        != expected_source_batches
    ):
        raise ValueError(
            "initializer diagnostics report the wrong PTB-XL source replay "
            "configuration"
        )
    replay = diagnostics.get("pretrain", {}).get("source_semantic_replay", {})
    expected_source_samples = (
        parent.pretrain_steps
        * parent.batch_size
        * expected_source_batches
        if expected_replay_weight > 0.0
        else 0
    )
    replay_drift = {
        key: (replay.get(key), expected)
        for key, expected in {
            "weight": expected_replay_weight,
            "batches_per_step": expected_source_batches,
            "samples": expected_source_samples,
            "dataset": "ptbxl",
            "partition": "train",
        }.items()
        if replay.get(key) != expected
    }
    if replay_drift:
        raise ValueError(
            f"initializer source replay evidence drifted: {replay_drift}"
        )
    raw_source_labels = replay.get("labels_consumed")
    if expected_replay_weight > 0.0 and raw_source_labels is not True:
        raise ValueError("replay-on initializer does not declare source labels")
    if expected_replay_weight == 0.0 and raw_source_labels not in {False, True}:
        raise ValueError("replay-off initializer has invalid source-label evidence")
    source_order_hash = replay.get("ordered_batch_hashes_sha256")
    source_loader_identity = _validated_source_loader_identity(
        replay,
        enabled=expected_replay_weight > 0.0,
        center=center,
        model_name=parent.model_name,
    )
    if expected_replay_weight > 0.0:
        if (
            not math.isfinite(float(replay.get("mean_bce", math.nan)))
            or not _is_sha256(source_order_hash)
            or source_order_hash == hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError(
                "initializer does not prove its ordered PTB-XL source replay"
            )
    elif source_order_hash is not None:
        raise ValueError(
            "replay-off initializer unexpectedly contains a source order hash"
        )
    if diagnostics.get("labels_consumed") is not False:
        raise ValueError("sequential Stage-1 initializer consumed target labels in SSL")
    if diagnostics.get("pretrain", {}).get("steps") != parent.pretrain_steps:
        raise ValueError("initializer did not complete the frozen Stage-1 step budget")
    ordered_batch_hash = diagnostics.get("ordered_batch_hashes_sha256")
    if (
        not _is_sha256(ordered_batch_hash)
        or ordered_batch_hash == hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("initializer does not prove its ordered K500 pretrain batches")
    selection = (
        diagnostics.get("loader", {})
        .get("dataset", {})
        .get("selection", {})
    )
    required_selection = {
        "logical_center": center,
        "partition": "k500",
        "record_count": 500,
        "mapping_hash": MAPPING_HASH,
        "mapping_version": MAPPING_VERSION,
    }
    drift = {
        key: (selection.get(key), expected)
        for key, expected in required_selection.items()
        if selection.get(key) != expected
    }
    if drift:
        raise ValueError(f"initializer K500 identity drifted: {drift}")
    selected = result.get("selection", {}).get("selected_checkpoint", {})
    checkpoint_sha256 = _sha256(checkpoint)
    if selected.get("sha256") != checkpoint_sha256:
        raise ValueError(
            "initializer checkpoint does not match its parent train_result.json"
        )
    if int(result.get("selection", {}).get("selected_epoch", -1)) != parent.epochs:
        raise ValueError("initializer selected epoch does not match parent candidate")
    history_rows = _assert_epoch_count(
        history,
        parent,
        label=f"{center}/{parent.candidate_id}/initializer",
    )
    if (
        history.get("method_id") != "direct_depth23_fixed20"
        or result.get("epochs_completed") != len(history_rows)
        or result.get("optimizer_steps") != history.get("optimizer_steps")
    ):
        raise ValueError("initializer train_result and training history drifted")
    return {
        "parent_candidate": parent.candidate_id,
        "parent_candidate_config": parent.__dict__,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "parent_manifest": str(parent_manifest),
        "parent_manifest_sha256": _sha256(parent_manifest),
        "parent_train_result": str(train_result),
        "parent_train_result_sha256": _sha256(train_result),
        "parent_run_manifest": str(run_manifest),
        "parent_run_manifest_sha256": _sha256(run_manifest),
        "parent_run_file_index": str(run_file_index),
        "parent_run_file_index_sha256": _sha256(run_file_index),
        "parent_method": str(parent_method_path),
        "parent_method_sha256": _sha256(parent_method_path),
        "runtime_config_snapshot_closure": snapshot_closure,
        "runtime_artifact_closure": artifact_closure,
        "parent_teacher_diagnostics": str(teacher_diagnostics),
        "parent_teacher_diagnostics_sha256": _sha256(teacher_diagnostics),
        "parent_selected_epoch": parent.epochs,
        "k500_selection": {
            key: selection.get(key)
            for key in (
                "logical_center",
                "source_centers",
                "partition",
                "record_count",
                "hash_id_set_sha256",
                "source_manifest_sha256",
                "split_id",
                "split_manifest_path",
                "split_manifest_sha256",
                "mapping_version",
                "mapping_hash",
                "class_order",
            )
        },
        "ordered_pretrain_batch_hashes_sha256": ordered_batch_hash,
        "source_replay_evidence": {
            "enabled": expected_replay_weight > 0.0,
            "weight": expected_replay_weight,
            "batches_per_step": expected_source_batches,
            "samples": expected_source_samples,
            "dataset": "ptbxl",
            "partition": "train",
            "labels_consumed": expected_replay_weight > 0.0,
            "raw_labels_consumed": raw_source_labels,
            "legacy_replay_off_literal_normalized": (
                expected_replay_weight == 0.0 and raw_source_labels is True
            ),
            "ordered_batch_hashes_sha256": source_order_hash,
            "loader": source_loader_identity,
        },
        "outside_k500_target_model_access": False,
        "outside_k500_model_access": False,
    }


def _source_registry_for(candidate: Candidate, center: str) -> dict[str, Any]:
    registry = _load_yaml(LOCKED_SOURCE_REGISTRY)
    if candidate.initializer_candidate is None:
        return registry
    checkpoint = _source_checkpoint(candidate, center)
    evidence = _parent_evidence(candidate, center)
    parent_candidate = CANDIDATES[candidate.initializer_candidate]
    parent = copy.deepcopy(registry["models"][candidate.model_name])
    parent.update(
        {
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": evidence["checkpoint_sha256"],
            "selected_epoch": parent_candidate.epochs,
            "initializer_candidate": candidate.initializer_candidate,
            "logical_center": center,
            "selection_rule": (
                f"fixed_global_epoch_{parent_candidate.epochs}_before_continuation"
            ),
            "evidence_status": "heldout_tuned_development_only",
            "initializer_evidence": evidence,
        }
    )
    for key in ("fold9", "fold10", "tensorboard_log_dir"):
        parent.pop(key, None)
    registry["baseline"] = {
        "id": (
            f"{candidate.initializer_candidate}_{center}_"
            "continuation_initializer"
        ),
        "status": "derived_k500_initializer_development_only",
        "parent_source_registry": str(LOCKED_SOURCE_REGISTRY),
        "parent_source_registry_sha256": _sha256(LOCKED_SOURCE_REGISTRY),
        "initializer_candidate": candidate.initializer_candidate,
        "logical_center": center,
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
        "class_order": list(CLASS_ORDER),
        "selection_rule": (
            f"fixed_global_epoch_{parent_candidate.epochs}_before_continuation"
        ),
        "initializer_evidence": evidence,
        "outside_k500_model_access": False,
    }
    registry["models"] = {candidate.model_name: parent}
    registry["downstream_policy"] = {
        "checkpoint_field": "selected_checkpoint",
        "require_sha256_match": True,
        "prohibit_last_checkpoint": False,
        "allowed_role": "matched_continuation_initializer",
        "paper_status": "development_only_until_independent_validation",
    }
    registry.pop("sensitivity_evidence", None)
    return registry


def _apply_supervised_corruption_schedule(
    method: dict[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    """Derive the frozen supervised corruption schedule for one candidate.

    The rotating schedule remains an isolated sandbox contract.  It preserves
    the clean/corruption family mass and one outer optimizer step per base
    batch, but executes two depth-2 and two depth-3 slots per epoch.  A scoped
    runtime adapter resolves each slot to per-record canonical compositions.
    """

    schedule = candidate.supervised_corruption_schedule
    if schedule == FIXED20_SUPERVISED_SCHEDULE:
        return method
    if schedule != ROTATING4_SUPERVISED_SCHEDULE:
        raise ValueError(
            f"unsupported supervised corruption schedule: {schedule!r}"
        )
    if candidate.method_family not in {
        "direct_target_ssl_control",
        "vae_lhat_post_refine",
    }:
        raise ValueError(
            "rotating4 is restricted to the matched Direct/VAE-LHAT pair"
        )

    result = copy.deepcopy(method)
    has_lhat = "lhat" in result["nodes"]
    result["method"]["scientific_arm"] = (
        "augmix_simclr_then_rotating4_plus_vae_lhat_hard_only"
        if has_lhat
        else "augmix_simclr_then_rotating4_supervised_adaptation"
    )
    result["method"]["description"] = (
        "Two-chain AugMix SimCLR pretraining followed by clean supervision "
        "and a deterministic depth-balanced rotating corruption schedule "
        "(two depth-2 plus two depth-3 views per epoch; five epochs cover all "
        "twenty canonical compositions). "
        + (
            "One exact-label VAE-LHAT hard view contributes direct BCE and "
            "clean/hard Bernoulli JSD through the same direct-sum auxiliary "
            "used by the matched fixed20 arm. "
            if has_lhat
            else "No VAE/LHAT branch is active. "
        )
        + "The clean/corruption family mass remains 0.5/0.5 and all views "
        "accumulate into one optimizer update per base batch."
    )
    contracts = result["contracts"]
    contracts.update(
        {
            "exposure_policy": ROTATING4_EXPOSURE_POLICY,
            "clean_exposures_per_base_record": 1,
            "corrupted_exposures_per_base_record": 4,
            "total_exposures_per_base_record": 5,
            # This is the canonical universe, not the per-epoch execution
            # order.  The four host sentinels below delimit the accumulated
            # group while the runtime resolves actual per-record indices.
            "composition_indices": list(range(20)),
            "rotating4_execution_slot_sentinels": list(
                ROTATING4_EXECUTION_SLOT_SENTINELS
            ),
            "tensorboard_probe_composition_indices": [0, 10, 19],
            "family_loss_weights": {
                "clean": 0.5,
                "corrupted_total": 0.5,
                "corrupted_per_composition": 0.125,
            },
            "optimizer_step_policy": (
                "accumulate_family_balanced_once_per_base_batch"
            ),
            "batch_norm_running_stats_policy": (
                "family_loss_weighted_once_per_base_batch"
            ),
            "exposure_budget_match": (
                "matched_one_optimizer_step_per_base_batch"
            ),
            "rotating4_schedule": {
                "schema_version": 1,
                "policy": (
                    "stable_hash_depth_balanced_rotating_2plus2_v1"
                ),
                "depth2_composition_indices": list(range(10)),
                "depth3_composition_indices": list(range(10, 20)),
                "views_per_family_per_epoch": 2,
                "cycle_epochs": 5,
                "epoch_stride_within_family": 2,
                "record_offset": (
                    "sha256(namespace_pipe_hash_id)_uint64_be_mod10"
                ),
                "offset_namespace": (
                    "pn2021_rotating_depth23_2plus2_v1"
                ),
                "same_offset_for_depth2_and_depth3": True,
                "coverage": (
                    "each_record_sees_all_10_depth2_and_all_10_depth3_"
                    "compositions_once_per_5_epochs"
                ),
                "parameter_rng": (
                    "existing_comparison_rng_identity_and_exposure_identity"
                ),
            },
            "fixed20_endpoint_supervision": False,
            "rotating4_endpoint_supervision": True,
        }
    )
    if has_lhat:
        contracts["rotating4_base_gradient_preserved"] = True
    return result


def _method_for(candidate: Candidate, center: str) -> dict[str, Any]:
    if (
        candidate.polish_auxiliary_target is not None
        and candidate.method_family != "vae_lhat_clean_polish"
    ):
        raise ValueError(
            "polish_auxiliary_target is only valid for vae_lhat_clean_polish"
        )
    if (
        candidate.pcgrad_auxiliary_norm_ratio_cap is not None
        and candidate.method_family != "strong_lhat_augmix_chain3"
    ):
        raise ValueError(
            "pcgrad_auxiliary_norm_ratio_cap is restricted to "
            "strong_lhat_augmix_chain3"
        )
    if (
        candidate.clean_direct_sum_auxiliary
        and candidate.method_family
        not in {"vae_lhat_clean_polish", "clean_target_ssl_control"}
    ):
        raise ValueError(
            "clean direct-sum auxiliary is restricted to the VAE-LHAT "
            "clean-polish pair"
        )
    if (
        candidate.lhat_search_mode != "optimized_softmax"
        and candidate.method_family
        not in {"strong_lhat_augmix_chain3", "vae_lhat_post_refine"}
    ):
        raise ValueError(
            "non-default LHAT search modes are restricted to "
            "strong chain-three or direct-supervision LHAT"
        )
    a7_method = _load_yaml(
        _base_a7_root(candidate) / "methods" / f"{center}.yaml"
    )
    if candidate.method_family == "invalid_d19_outer_lhat_override":
        raise ValueError(
            "P18 is invalid: D19 consumes compatible_runtime from its candidate "
            "profile, so overriding only resources.lhat_config is inert"
        )
    if candidate.method_family in {"d19_separate_views", "checkpoint_blend"}:
        method = a7_method
    elif candidate.method_family == "strong_lhat_augmix_chain3":
        if candidate.model_name != "efficientnet1dv2":
            raise ValueError(
                "strong_lhat_augmix_chain3 is only validated for efficientnet1dv2"
            )
        method = _load_yaml(STRONG_LHAT_AUGMIX_METHOD)
        if candidate.lhat_search_mode == "uniform_compute_matched":
            method["resources"]["lhat_config"]["path"] = str(
                _candidate_relroot(candidate) / "lhat.yaml"
            )
            method["method"]["description"] = (
                "Compute-matched non-adversarial control for the strong LHAT "
                "third-chain method. It retains the same five search forwards "
                "and backwards but keeps the candidate softmax weights "
                "numerically uniform."
            )
        elif candidate.lhat_search_mode != "optimized_softmax":
            raise ValueError(
                f"unsupported strong LHAT search mode: "
                f"{candidate.lhat_search_mode!r}"
            )
        lhat_resource_path = Path(method["resources"]["lhat_config"]["path"])
        if lhat_resource_path.is_absolute() or ".." in lhat_resource_path.parts:
            raise ValueError("strong LHAT config must stay inside config root")
        runtime_lhat = _load_yaml(CONFIG_ROOT / lhat_resource_path)
        runtime_epsilon = float(
            runtime_lhat["hull_attack"]["pgd_epsilon_l2_standardized"]
        )
        runtime_search_mode = str(
            runtime_lhat["hull_attack"].get("weight_mode", "")
        )
        if runtime_search_mode != candidate.lhat_search_mode:
            raise ValueError(
                "strong LHAT candidate metadata/runtime search-mode mismatch: "
                f"{candidate.lhat_search_mode!r} versus "
                f"{runtime_search_mode!r}"
            )
        if not math.isclose(
            candidate.lhat_pgd_epsilon_l2_standardized,
            runtime_epsilon,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "strong LHAT candidate metadata/runtime epsilon mismatch: "
                f"{candidate.lhat_pgd_epsilon_l2_standardized} versus "
                f"{runtime_epsilon}"
            )
        method["contracts"]["lhat_pgd_epsilon_l2_standardized"] = (
            runtime_epsilon
        )
        method["contracts"]["lhat_config_path"] = str(lhat_resource_path)
        method["contracts"]["lhat_search_mode"] = runtime_search_mode
        method["contracts"]["lhat_search_learning_rate"] = float(
            runtime_lhat["hull_attack"]["learning_rate"]
        )
        method["contracts"]["lhat_attack_objective"] = str(
            runtime_lhat["hull_attack"]["objective"]
        )
        if candidate.pcgrad_auxiliary_norm_ratio_cap is not None:
            if not math.isclose(
                candidate.auxiliary_alpha,
                0.75,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "gradient-balanced LHAT chain3 requires auxiliary alpha 0.75"
                )
            if not math.isclose(
                candidate.pcgrad_auxiliary_norm_ratio_cap,
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "gradient-balanced LHAT chain3 requires norm-ratio cap 1.0"
                )
            method["method"].update(
                {
                    "id": (
                        "fixed20_pcgrad_gradbalanced_"
                        "strong_lhat_augmix_v1"
                    ),
                    "scientific_arm": (
                        "fixed20_base_plus_pcgrad_gradient_balanced_"
                        "strong_lhat_augmix"
                    ),
                    "description": (
                        "Preserve family-balanced Direct+fixed20 and add the "
                        "same exact-label LHAT third-chain AugMix auxiliary. "
                        "After conflict projection, request alpha 0.75 but cap "
                        "the auxiliary gradient norm at the current fixed20 "
                        "base-gradient norm before the ordinary combined clip."
                    ),
                }
            )
            method["contracts"].update(
                {
                    "pcgrad_auxiliary_norm_ratio_cap": 1.0,
                    "pcgrad_auxiliary_norm_cap_policy": (
                        "project_conflicts_then_cap_aux_to_base_"
                        "before_combined_clip_v1"
                    ),
                }
            )
        method["contracts"]["unlabeled_teacher"] = copy.deepcopy(
            a7_method["contracts"]["unlabeled_teacher"]
        )
        if candidate.pretrain_view_mode is None:
            method["contracts"]["full_k500_heldout_oracle"] = copy.deepcopy(
                a7_method["contracts"]["full_k500_heldout_oracle"]
            )
        else:
            method["contracts"].pop("full_k500_heldout_oracle", None)
        method["contracts"]["tuning_partition_only"] = "k500"
        method["contracts"]["full_k500_refit_supported"] = True
        method["contracts"]["exposure_policy"] = (
            "clean_aux_once_then_exhaustive_depth23"
        )
    elif candidate.method_family == "clean_target_ssl_control":
        method = _load_yaml(CLEAN_ONLY_METHOD)
        method["method"].update(
            {
                "scientific_arm": (
                    "augmix_simclr_then_clean_only_supervised_adaptation"
                ),
                "status": "heldout_tuned_development_only",
                "description": (
                    "K500-only two-chain AugMix SimCLR pretraining followed "
                    "by clean-only supervised adaptation. Fixed20 and every "
                    "VAE/LHAT branch are absent."
                ),
            }
        )
        method["contracts"]["unlabeled_teacher"] = copy.deepcopy(
            a7_method["contracts"]["unlabeled_teacher"]
        )
        method["contracts"].update(
            {
                "tuning_partition_only": "k500",
                "full_k500_refit_supported": True,
                "fixed20_train_views": 0,
                "augmix_train_views": 0,
                "stage1_components": [
                    candidate.pretrain_view_mode,
                    "simclr",
                    "ptbxl_source_logit_anchor",
                ],
            }
        )
    elif candidate.method_family == "direct_target_ssl_control":
        method = _load_yaml(DIRECT_FIXED20_METHOD)
        method["contracts"]["unlabeled_teacher"] = copy.deepcopy(
            a7_method["contracts"]["unlabeled_teacher"]
        )
        if candidate.pretrain_view_mode is None:
            method["contracts"]["full_k500_heldout_oracle"] = copy.deepcopy(
                a7_method["contracts"]["full_k500_heldout_oracle"]
            )
        method["contracts"]["tuning_partition_only"] = "k500"
        method["contracts"]["full_k500_refit_supported"] = True
    elif candidate.method_family == "direct_budget_filler":
        method = _load_yaml(DIRECT_FIXED20_METHOD)
        method["contracts"]["unlabeled_teacher"] = copy.deepcopy(
            a7_method["contracts"]["unlabeled_teacher"]
        )
        method["contracts"]["full_k500_heldout_oracle"] = copy.deepcopy(
            a7_method["contracts"]["full_k500_heldout_oracle"]
        )
        method["contracts"]["tuning_partition_only"] = "k500"
        method["contracts"]["full_k500_refit_supported"] = True
    elif candidate.method_family == "vae_lhat_bounded_augmax_polish":
        if candidate.model_name != "efficientnet1dv2":
            raise ValueError(
                "bounded AugMax must pass the frozen EfficientNet gate before "
                "cross-backbone transfer"
            )
        geometry = _bounded_augmax_geometry(candidate)
        method = _load_yaml(PURE_M20_BOUNDED_METHOD)
        method["method"].update(
            {
                "id": VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID,
                "scientific_arm": "vae_lhat_bounded_augmax_polish",
                "status": "heldout_tuned_development_only",
                "description": (
                    "K500-only bounded AugMax-M20 continuation. Each of twenty "
                    "views uses two online PN2021-C chains plus one shared "
                    "exact-label VAE-LHAT chain; six frozen convex candidates "
                    "are selected by current BCE under a [0,0.15] gain cap. "
                    "No fixed20 endpoints, teacher, replay, residual head, "
                    "PCGrad, VAE-random, SupCon, or class weighting enter."
                ),
            }
        )
        method["resources"]["lhat_config"]["path"] = str(
            _candidate_relroot(candidate) / "lhat.yaml"
        )
        method["contracts"].update(
            {
                "complete_k500_base_record_exposure": True,
                "lhat_hull_lambda": geometry["hull_lambda"],
                "lhat_steps": geometry["steps"],
                "lhat_pgd_epsilon_l2_standardized": geometry[
                    "pgd_epsilon_l2_standardized"
                ],
                "local_lhat_geometry_preset": geometry["preset"],
                "post_refine_components": [
                    "online_depth23_m20_coverage",
                    "shared_exact_label_vae_lhat_chain3",
                    "bounded_per_record_augmax_coefficient_search",
                    "classic_bernoulli_jsd",
                ],
                "post_refine_excluded_components": [
                    "fixed20_endpoint_supervision",
                    "teacher",
                    "source_replay",
                    "residual_head",
                    "pcgrad",
                    "vae_random",
                    "supcon",
                    "class_reweighting",
                ],
                "tuning_partition_only": "k500",
                "full_k500_refit_supported": True,
                "heldout_target_feedback_allowed": False,
                "matched_control_method_id": (
                    CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID
                ),
            }
        )
    elif candidate.method_family == "vae_lhat_post_refine":
        if candidate.lhat_pgd_epsilon_l2_standardized not in {2.0, 4.0, 8.0}:
            raise ValueError(
                "VAE-LHAT post-refine epsilon must be one of 2, 4, or 8"
            )
        joint_hard_only = bool(
            candidate.pretrain_epochs or candidate.pretrain_steps
        )
        expected_alpha = 1.5 if joint_hard_only else 0.25
        if not math.isclose(
            candidate.auxiliary_alpha,
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "VAE-LHAT hard-only joint training requires alpha=1.5; "
                "short post-refine requires alpha=0.25"
            )
        method = _load_yaml(STRONG_LHAT_DIRECT_AUGMIX_METHOD)
        method["method"].update(
            {
                "id": (
                    HARD_ONLY_PRUNED_METHOD_ID
                    if joint_hard_only
                    else "fixed20_pcgrad_vae_lhat_post_refine_v1"
                ),
                "scientific_arm": (
                    "augmix_simclr_then_fixed20_plus_vae_lhat_hard_only"
                    if joint_hard_only
                    else "fixed20_plus_vae_lhat_post_refine"
                ),
                "status": "heldout_tuned_development_only",
                "description": (
                    (
                        (
                            "Classic width-3, stochastic-depth-1-to-3 AugMix "
                            "SimCLR pretraining followed by "
                            if candidate.pretrain_view_mode
                            == CLASSIC_AUGMIX_VIEW_MODE
                            else "Two-chain AugMix SimCLR pretraining followed by "
                        )
                        +
                        "Direct+fixed20 with one exact-label VAE-LHAT hard "
                        "view and clean/LHAT JSD. "
                        if joint_hard_only
                        else (
                            "Direct+fixed20 continuation with one exact-label "
                            "VAE-LHAT auxiliary view and clean/LHAT JSD. "
                        )
                    )
                    + "No AugMix, raw auxiliary view, VAE-random view, or "
                    "source replay is active during the supervised stage. "
                    f"Standardized L2 epsilon="
                    f"{candidate.lhat_pgd_epsilon_l2_standardized:g}."
                ),
            }
        )
        method["nodes"].pop("threechain_augmix")
        for output_name in ("augmix_view", "augmix_diagnostics"):
            method["outputs"].pop(output_name)
        method["objective"]["terms"] = [
            term
            for term in method["objective"]["terms"]
            if term["id"] in {"clean_bce", "corrupted_bce", "lhat_direct_bce"}
        ]
        method["objective"]["terms"].append(
            {
                "id": "clean_lhat_jsd",
                "type": "multilabel_bernoulli_jsd",
                "views": ["clean_view", "lhat_view"],
                "weight": 0.5,
            }
        )
        for resource_name in ("augmix_config", "augmix_rng"):
            method["resources"].pop(resource_name)
        if (
            candidate.lhat_pgd_epsilon_l2_standardized == 2.0
            and candidate.lhat_geometry_preset is None
            and candidate.lhat_search_mode == "optimized_softmax"
        ):
            lhat_config_path = "train/lhat.yaml"
        else:
            lhat_config_path = str(_candidate_relroot(candidate) / "lhat.yaml")
        method["resources"]["lhat_config"]["path"] = lhat_config_path
        runtime_lhat = _load_yaml(CONFIG_ROOT / lhat_config_path)
        runtime_attack = runtime_lhat["hull_attack"]
        runtime_mode = str(runtime_attack.get("weight_mode", ""))
        if runtime_mode != candidate.lhat_search_mode:
            raise ValueError(
                "direct LHAT metadata/runtime search-mode mismatch: "
                f"{candidate.lhat_search_mode!r} versus {runtime_mode!r}"
            )
        contracts = method["contracts"]
        contracts.pop("full_k500_heldout_oracle", None)
        contracts["tuning_partition_only"] = "k500"
        contracts["full_k500_refit_supported"] = True
        # PCGrad evaluates the LHAT auxiliary objective inside the clean
        # exposure and combines its gradient at the single base optimizer
        # step. It is not a 22nd dataloader exposure.
        contracts["exposure_policy"] = (
            "clean_aux_once_then_exhaustive_depth23"
            if joint_hard_only
            else "clean_once_then_exhaustive_depth23"
        )
        contracts["pcgrad_auxiliary_mode"] = (
            "fixed20_base_plus_direct_lhat_augmix_aux_v1"
            if joint_hard_only
            else "fixed20_base_plus_projected_vae_lhat_aux_v1"
        )
        contracts["pcgrad_auxiliary_alpha"] = candidate.auxiliary_alpha
        contracts["pcgrad_auxiliary_terms"] = [
            "lhat_direct_bce",
            "clean_lhat_jsd",
        ]
        contracts["lhat_hull_lambda"] = float(runtime_attack["hull_lambda"])
        contracts["lhat_steps"] = int(runtime_attack["steps"])
        contracts["lhat_pgd_epsilon_l2_standardized"] = float(
            runtime_attack["pgd_epsilon_l2_standardized"]
        )
        contracts["lhat_search_mode"] = runtime_mode
        contracts["lhat_search_learning_rate"] = float(
            runtime_attack["learning_rate"]
        )
        contracts["lhat_attack_objective"] = str(
            runtime_attack["objective"]
        )
        contracts["lhat_config_path"] = lhat_config_path
        if candidate.lhat_geometry_preset is not None:
            contracts["lhat_geometry_preset"] = candidate.lhat_geometry_preset
        if candidate.lhat_search_mode == "uniform_compute_matched":
            method["method"]["description"] += (
                " This compute-matched non-adversarial control executes the "
                "same LHAT search graph while keeping neighbor weights "
                "numerically uniform."
            )
        contracts["post_refine_components"] = [
            "vae_lhat_exact_label_nonself",
            "lhat_direct_bce",
            "clean_lhat_jsd",
        ]
        contracts["post_refine_excluded_components"] = [
            "augmix",
            "raw_auxiliary",
            "vae_random",
            "source_replay",
        ]
        contracts.pop("augmix_chain3_source", None)
        if joint_hard_only:
            contracts["unlabeled_teacher"] = copy.deepcopy(
                a7_method["contracts"]["unlabeled_teacher"]
            )
            contracts["stage1_components"] = [
                candidate.pretrain_view_mode,
                "simclr",
                "ptbxl_source_logit_anchor",
            ]
        else:
            contracts.pop("unlabeled_teacher", None)
    elif candidate.method_family == "vae_lhat_clean_polish":
        if candidate.model_name not in {"efficientnet1dv2", "ecgfounder"}:
            raise ValueError(
                "VAE-LHAT clean polish supports only EfficientNet1DV2 and "
                "ECGFounder"
            )
        if candidate.lhat_pgd_epsilon_l2_standardized not in {2.0, 4.0, 8.0}:
            raise ValueError(
                "VAE-LHAT clean-polish epsilon must be one of 2, 4, or 8"
            )
        teacher_distill = candidate.frozen_teacher_mix is not None
        hard_label_lowdose = candidate.polish_auxiliary_target == "hard_label"
        latent_path_lowdose = (
            candidate.polish_auxiliary_target == "latent_path_hard_label"
        )
        consistency_lowdose = (
            candidate.polish_auxiliary_target == "consistency"
        )
        feature_invariance_lowdose = (
            candidate.polish_auxiliary_target == "feature_invariance"
        )
        local_anchor_soft_lowdose = (
            candidate.polish_auxiliary_target == "local_anchor_soft"
        )
        local_anchor_soft_geometry = (
            _local_anchor_soft_geometry(candidate)
            if local_anchor_soft_lowdose
            else None
        )
        local_anchor_soft_method_ids = (
            _local_anchor_soft_method_ids(candidate)
            if local_anchor_soft_lowdose
            else (None, None)
        )
        if candidate.polish_auxiliary_target not in {
            None,
            "hard_label",
            "latent_path_hard_label",
            "consistency",
            "feature_invariance",
            "local_anchor_soft",
        }:
            raise ValueError(
                "VAE-LHAT clean-polish auxiliary target must be hard_label, "
                "latent_path_hard_label, consistency, feature_invariance, "
                "local_anchor_soft, or omitted"
            )
        if teacher_distill and (
            hard_label_lowdose
            or latent_path_lowdose
            or consistency_lowdose
            or feature_invariance_lowdose
            or local_anchor_soft_lowdose
        ):
            raise ValueError(
                "VAE-LHAT polish cannot use teacher and another auxiliary "
                "target simultaneously"
            )
        if teacher_distill and candidate.frozen_teacher_mix != 1.0:
            raise ValueError(
                "VAE-LHAT teacher distillation requires frozen_teacher_mix=1.0"
            )
        lowdose_auxiliary = (
            teacher_distill
            or hard_label_lowdose
            or latent_path_lowdose
            or consistency_lowdose
            or feature_invariance_lowdose
            or local_anchor_soft_lowdose
        )
        direct_sum_auxiliary = candidate.clean_direct_sum_auxiliary
        if direct_sum_auxiliary and lowdose_auxiliary:
            raise ValueError(
                "clean direct-sum auxiliary cannot be combined with a "
                "low-dose auxiliary target"
            )
        if (
            direct_sum_auxiliary
            and float(candidate.auxiliary_alpha)
            not in VAE_CLEAN_DIRECTSUM_AUXILIARY_ALPHAS
        ):
            raise ValueError(
                "clean direct-sum auxiliary alpha must be one of "
                f"{sorted(VAE_CLEAN_DIRECTSUM_AUXILIARY_ALPHAS)}"
            )
        if (
            not lowdose_auxiliary
            and not direct_sum_auxiliary
            and not math.isclose(
                float(candidate.auxiliary_alpha),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "VAE-LHAT clean polish requires auxiliary_alpha=1.0 unless "
                "an explicit low-dose or direct-sum contract is selected"
            )
        direct_sum_term_weight = 0.5 * float(candidate.auxiliary_alpha)
        lowdose_auxiliary_weight = (
            float(candidate.auxiliary_alpha) if lowdose_auxiliary else 0.0
        )
        if lowdose_auxiliary and not 0.0 < lowdose_auxiliary_weight < 0.5:
            raise ValueError(
                "VAE-LHAT low-dose auxiliary weight must lie in (0,0.5)"
            )
        lowdose_clean_preserve_raw = 1.0 - 2.0 * lowdose_auxiliary_weight
        lowdose_auxiliary_raw = 2.0 * lowdose_auxiliary_weight
        lowdose_auxiliary_term = (
            "lhat_teacher_distill_bce"
            if teacher_distill
            else (
                "lhat_path_hard_bce"
                if latent_path_lowdose
                else (
                    "lhat_anchor_soft_bce"
                    if local_anchor_soft_lowdose
                    else (
                        "lhat_hard_bce"
                        if hard_label_lowdose
                        else (
                            "clean_lhat_feature_cosine"
                            if feature_invariance_lowdose
                            else "clean_lhat_consistency_jsd"
                        )
                    )
                )
            )
        )
        lowdose_effective_weights = (
            {
                "clean_bce": 0.50,
                "clean_preserve_bce": (
                    0.50 - lowdose_auxiliary_weight
                ),
                lowdose_auxiliary_term: lowdose_auxiliary_weight,
            }
            if consistency_lowdose or feature_invariance_lowdose
            else {
                "clean_bce": 0.50,
                "clean_preserve_bce": 0.50 - lowdose_auxiliary_weight,
                lowdose_auxiliary_term: lowdose_auxiliary_weight,
            }
        )
        method = _load_yaml(STRONG_LHAT_DIRECT_AUGMIX_METHOD)
        method["method"].update(
            {
                "id": (
                    VAE_TEACHER_DISTILL_POLISH_METHOD_ID
                    if teacher_distill
                    else (
                        VAE_PATH_BCE_POLISH_METHOD_ID
                        if latent_path_lowdose
                        else (
                            VAE_HARD_BCE_POLISH_METHOD_ID
                            if hard_label_lowdose
                            else (
                                local_anchor_soft_method_ids[0]
                                if local_anchor_soft_lowdose
                                else (
                                    VAE_CONSISTENCY_POLISH_METHOD_ID
                                    if consistency_lowdose
                                    else (
                                    VAE_FEATURE_INVARIANCE_POLISH_METHOD_ID
                                    if feature_invariance_lowdose
                                    else (
                                        VAE_CLEAN_DIRECTSUM_METHOD_ID
                                        if direct_sum_auxiliary
                                        else VAE_CLEAN_POLISH_METHOD_ID
                                    )
                                    )
                                )
                            )
                        )
                    )
                ),
                "scientific_arm": (
                    "vae_lhat_teacher_distill_polish"
                    if teacher_distill
                    else (
                        "vae_lhat_path_bce_polish"
                        if latent_path_lowdose
                        else (
                            "vae_lhat_hard_bce_polish"
                            if hard_label_lowdose
                            else (
                                (
                                    "vae_lhat_calibrated_local_anchor_soft_polish"
                                    if local_anchor_soft_geometry["preset"]
                                    == "historical_gate2_calibrated_v1"
                                    else "vae_lhat_local_anchor_soft_polish"
                                )
                                if local_anchor_soft_lowdose
                                else (
                                    "vae_lhat_consistency_polish"
                                    if consistency_lowdose
                                    else (
                                        "vae_lhat_feature_invariance_polish"
                                        if feature_invariance_lowdose
                                        else (
                                            "vae_lhat_clean_directsum"
                                            if direct_sum_auxiliary
                                            else "vae_lhat_clean_polish"
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "status": "heldout_tuned_development_only",
                "description": (
                    (
                        "Single-exposure K500 polish with effective objective "
                        f"{1.0 - lowdose_auxiliary_weight:.2f} clean hard-label "
                        f"BCE + {lowdose_auxiliary_weight:.2f} pure frozen-"
                        "initializer distillation BCE on the direct VAE-LHAT "
                        "endpoint. The outer LHAT term has no hard-label "
                        "remainder, pos_weight, JSD, or AugMix. "
                        if teacher_distill
                        else (
                            "Single-exposure K500 polish with effective objective "
                            f"{1.0 - lowdose_auxiliary_weight:.2f} clean hard-label "
                            f"BCE + {lowdose_auxiliary_weight:.2f} hard BCE on one "
                            "uniformly sampled standardized latent path point "
                            "t in {0.4,0.7,1.0} between clean and the exact-label "
                            "VAE-LHAT endpoint. There is no teacher, JSD, or "
                            "AugMix in this stage. "
                            if latent_path_lowdose
                            else (
                                "Single-exposure K500 polish with effective "
                                f"objective {1.0 - lowdose_auxiliary_weight:.2f} "
                                "clean hard-label BCE + "
                                f"{lowdose_auxiliary_weight:.2f} exact-label "
                                "hard BCE on the direct VAE-LHAT endpoint. There "
                                "is no teacher, JSD, or AugMix in this stage. "
                                if hard_label_lowdose
                                else (
                                    "Single-exposure K500 polish with effective "
                                    f"objective {1.0 - lowdose_auxiliary_weight:.2f} "
                                    "clean hard-label BCE + "
                                    f"{lowdose_auxiliary_weight:.2f} anchor-soft "
                                    "BCE on a local exact-positive-set VAE-LHAT "
                                    "view. The target is 0.95 on anchor positives "
                                    "and 0 otherwise; the matched identity control "
                                    "uses the same target. "
                                    if local_anchor_soft_lowdose
                                    else (
                                        "Single-exposure K500 polish with effective "
                                        f"objective {1.0 - lowdose_auxiliary_weight:.2f} "
                                        "clean hard-label BCE + "
                                        f"{lowdose_auxiliary_weight:.2f} symmetric "
                                        "Bernoulli JSD between current clean and direct "
                                        "VAE-LHAT predictions. The LHAT endpoint has "
                                        "no direct label or teacher loss. "
                                        if consistency_lowdose
                                        else (
                                            "Single-exposure K500 polish with effective "
                                            f"objective {1.0 - lowdose_auxiliary_weight:.2f} "
                                            "clean hard-label BCE + "
                                            f"{lowdose_auxiliary_weight:.2f} mean "
                                            "one-minus-cosine distance between current "
                                            "clean and direct VAE-LHAT penultimate "
                                            "features. The LHAT endpoint has no direct "
                                            "label, teacher or prediction-JSD loss. "
                                            if feature_invariance_lowdose
                                            else (
                                                "Single-exposure K500 adaptation "
                                                "with one clean-base gradient plus "
                                                f"{candidate.auxiliary_alpha:g} "
                                                "times the same VAE auxiliary "
                                                "used by the matched fixed20 arm: "
                                                "0.5 exact-label VAE-LHAT BCE + "
                                                "0.5 clean/LHAT Bernoulli JSD. "
                                                if direct_sum_auxiliary
                                                else
                                                "Single-exposure K500 polish with "
                                                "effective objective 0.50 clean BCE "
                                                "+ 0.25 exact-label VAE-LHAT BCE + "
                                                "0.25 clean/LHAT Bernoulli JSD. "
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    + "Fixed20, post-stage AugMix, PCGrad, VAE-random, raw "
                    "auxiliary and source replay are absent. "
                    f"Standardized L2 epsilon="
                    f"{candidate.lhat_pgd_epsilon_l2_standardized:g}."
                ),
            }
        )
        for node_name in ("depth23_corruption", "threechain_augmix"):
            method["nodes"].pop(node_name)
        for output_name in (
            "corrupted_view",
            "corruption_provenance",
            "augmix_view",
            "augmix_diagnostics",
        ):
            method["outputs"].pop(output_name)
        method["objective"]["terms"] = (
            [
                {
                    "id": "clean_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "clean_view",
                    "targets": "targets",
                    "weight": 1.0,
                },
                {
                    "id": "clean_preserve_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "clean_view",
                    "targets": "targets",
                    "weight": lowdose_clean_preserve_raw,
                },
                {
                    "id": lowdose_auxiliary_term,
                    "type": "multilabel_bernoulli_jsd",
                    "views": ["clean_view", "lhat_view"],
                    "weight": lowdose_auxiliary_raw,
                },
            ]
            if consistency_lowdose or feature_invariance_lowdose
            else [
                {
                    "id": "clean_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "clean_view",
                    "targets": "targets",
                    "weight": 1.0,
                },
                {
                    "id": "clean_preserve_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "clean_view",
                    "targets": "targets",
                    "weight": lowdose_clean_preserve_raw,
                },
                {
                    "id": lowdose_auxiliary_term,
                    "type": "multilabel_bce_with_logits",
                    "view": "lhat_view",
                    "targets": "targets",
                    "weight": lowdose_auxiliary_raw,
                },
            ]
            if lowdose_auxiliary
            else [
                {
                    "id": "clean_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "clean_view",
                    "targets": "targets",
                    "weight": 1.0,
                },
                {
                    "id": "lhat_direct_bce",
                    "type": "multilabel_bce_with_logits",
                    "view": "lhat_view",
                    "targets": "targets",
                    "weight": (
                        direct_sum_term_weight
                        if direct_sum_auxiliary
                        else 0.5
                    ),
                },
                {
                    "id": "clean_lhat_jsd",
                    "type": "multilabel_bernoulli_jsd",
                    "views": ["clean_view", "lhat_view"],
                    "weight": (
                        direct_sum_term_weight
                        if direct_sum_auxiliary
                        else 0.5
                    ),
                },
            ]
        )
        for resource_name in (
            "operator_profile",
            "corruption_rng",
            "augmix_config",
            "augmix_rng",
        ):
            method["resources"].pop(resource_name)
        lhat_config_path = (
            str(_candidate_relroot(candidate) / "lhat.yaml")
            if local_anchor_soft_lowdose
            else (
                "train/lhat.yaml"
                if candidate.lhat_pgd_epsilon_l2_standardized == 2.0
                else str(_candidate_relroot(candidate) / "lhat.yaml")
            )
        )
        method["resources"]["lhat_config"]["path"] = lhat_config_path
        method["contracts"] = {
            "executable": True,
            "normalization_owner": "online_trainer",
            "optimizer_owner": "online_trainer",
            "canonical_input_sampling_rate_hz": 100,
            "canonical_output_sampling_rate_hz": 100,
            "fixed20_train_views": 0,
            "augmix_train_views": 0,
            "one_outer_optimizer_step_per_clean_batch": True,
            "complete_k500_base_record_exposure": True,
            "objective_scale_adapter_mode": "scale_only",
            "objective_global_scale": 0.5,
            "effective_objective_weights": copy.deepcopy(
                lowdose_effective_weights
                if lowdose_auxiliary
                else (
                    {
                        "clean_bce": 1.0,
                        "lhat_direct_bce": direct_sum_term_weight,
                        "clean_lhat_jsd": direct_sum_term_weight,
                    }
                    if direct_sum_auxiliary
                    else VAE_CLEAN_POLISH_EFFECTIVE_WEIGHTS
                )
            ),
            "batch_norm_running_stats_policy": (
                "objective_view_weighted_once_per_base_batch"
            ),
            "batch_norm_objective_view_weights": {
                "clean_view": (
                    1.0 - lowdose_auxiliary_weight
                    if lowdose_auxiliary
                    else 0.5
                ),
                "lhat_view": (
                    lowdose_auxiliary_weight if lowdose_auxiliary else 0.5
                ),
            },
            "lhat_label_policy": "exact_positive_set_nonself",
            "lhat_hull_lambda": (
                local_anchor_soft_geometry["hull_lambda"]
                if local_anchor_soft_lowdose
                else 0.60
            ),
            "lhat_steps": (
                local_anchor_soft_geometry["steps"]
                if local_anchor_soft_lowdose
                else 5
            ),
            "lhat_pgd_epsilon_l2_standardized": (
                candidate.lhat_pgd_epsilon_l2_standardized
            ),
            "post_refine_components": [
                "vae_lhat_exact_label_nonself",
                *(
                    ["frozen_initializer_lhat_distillation"]
                    if teacher_distill
                    else (
                        ["standardized_lhat_path_sampling", "lhat_path_hard_bce"]
                        if latent_path_lowdose
                        else (
                            ["lhat_hard_bce"]
                            if hard_label_lowdose
                            else (
                                [
                                    "local_anchor_dominant_hull",
                                    "lhat_anchor_soft_bce",
                                ]
                                if local_anchor_soft_lowdose
                                else (
                                    ["clean_lhat_consistency_jsd"]
                                    if consistency_lowdose
                                    else (
                                        ["clean_lhat_penultimate_feature_cosine"]
                                        if feature_invariance_lowdose
                                        else ["lhat_direct_bce", "clean_lhat_jsd"]
                                    )
                                )
                            )
                        )
                    )
                ),
            ],
            "post_refine_excluded_components": [
                "fixed20",
                "augmix",
                "pcgrad",
                "raw_auxiliary",
                "vae_random",
                "source_replay",
            ],
            "tuning_partition_only": "k500",
            "full_k500_refit_supported": True,
            "heldout_target_feedback_allowed": False,
        }
        if direct_sum_auxiliary:
            method["contracts"].pop("objective_scale_adapter_mode")
            method["contracts"].pop("objective_global_scale")
            method["contracts"].update(
                {
                    "direct_sum_auxiliary_alpha": (
                        float(candidate.auxiliary_alpha)
                    ),
                    "matched_fixed20_candidate_method_id": (
                        HARD_ONLY_PRUNED_METHOD_ID
                    ),
                }
            )
        if teacher_distill:
            method["contracts"].update(
                {
                    "auxiliary_label_policy": (
                        "frozen_stage2_initializer_probability_only"
                    ),
                    "auxiliary_teacher_policy": (
                        "frozen_stage2_initializer_snapshot_before_first_"
                        "optimizer_step"
                    ),
                    "auxiliary_teacher_mix": 1.0,
                    "auxiliary_teacher_scope": "lhat_distill_bce_only",
                    "auxiliary_soft_bce_pos_weight": None,
                    "auxiliary_effective_weight": (
                        lowdose_auxiliary_weight
                    ),
                    "clean_hard_label_bce_preserved": True,
                    "auxiliary_teacher_initializer_sha256": _sha256(
                        _source_checkpoint(candidate, center)
                    ),
                    "matched_control_method_id": (
                        CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID
                    ),
                }
            )
        elif latent_path_lowdose:
            method["contracts"].update(
                {
                    "hard_auxiliary_target_policy": (
                        "anchor_ground_truth_multihot"
                    ),
                    "auxiliary_effective_weight": lowdose_auxiliary_weight,
                    "clean_hard_label_bce_preserved": True,
                    "lhat_path_polish": {
                        "enabled": True,
                        "t_values": list(LHAT_PATH_T_VALUES),
                        "sampling": (
                            "uniform_valid_per_eligible_record_per_epoch"
                        ),
                        "geometry": "standardized_clean_to_lhat_ray",
                        "residual_correction": "linear_clean_hard_endpoint",
                        "candidate_training_view": "one_sampled_path_point",
                        "matched_compute_in_control": True,
                    },
                    "matched_control_method_id": (
                        CLEAN_PATH_BCE_CONTROL_METHOD_ID
                    ),
                }
            )
        elif hard_label_lowdose:
            method["contracts"].update(
                {
                    "hard_auxiliary_target_policy": (
                        "anchor_ground_truth_multihot"
                    ),
                    "auxiliary_effective_weight": lowdose_auxiliary_weight,
                    "clean_hard_label_bce_preserved": True,
                    "matched_control_method_id": (
                        CLEAN_HARD_BCE_CONTROL_METHOD_ID
                    ),
                }
            )
        elif local_anchor_soft_lowdose:
            method["contracts"].update(
                {
                    "anchor_soft_target_policy": (
                        "anchor_soft_exact_positive_set_no_class_admission"
                    ),
                    "auxiliary_positive_value": 0.95,
                    "auxiliary_negative_value": 0.0,
                    "auxiliary_soft_bce_pos_weight": None,
                    "auxiliary_effective_weight": lowdose_auxiliary_weight,
                    "clean_hard_label_bce_preserved": True,
                    "local_anchor_soft_polish": {
                        "candidate_policy": "exact_positive_set_nonself",
                        "candidate_count": 20,
                        "candidate_includes_anchor": False,
                        "anchor_in_outer_interpolation": True,
                        "local_pool_size": 80,
                        "hull_lambda": local_anchor_soft_geometry[
                            "hull_lambda"
                        ],
                        "pgd_epsilon_l2_standardized": (
                            local_anchor_soft_geometry[
                                "pgd_epsilon_l2_standardized"
                            ]
                        ),
                        "steps": local_anchor_soft_geometry["steps"],
                        "learning_rate": local_anchor_soft_geometry[
                            "learning_rate"
                        ],
                        "minimum_effective_anchor_mass": (
                            local_anchor_soft_geometry[
                                "minimum_effective_anchor_mass"
                            ]
                        ),
                        "matched_compute_in_control": True,
                    },
                    "local_lhat_geometry_preset": (
                        local_anchor_soft_geometry["preset"]
                    ),
                    "matched_control_method_id": local_anchor_soft_method_ids[1],
                }
            )
        elif consistency_lowdose:
            method["contracts"].update(
                {
                    "clean_lhat_consistency_policy": (
                        "symmetric_bernoulli_jsd_current_model"
                    ),
                    "auxiliary_effective_weight": lowdose_auxiliary_weight,
                    "clean_hard_label_bce_preserved": True,
                    "lhat_direct_label_supervision": False,
                    "matched_control_method_id": (
                        CLEAN_CONSISTENCY_CONTROL_METHOD_ID
                    ),
                }
            )
        elif feature_invariance_lowdose:
            method["contracts"].update(
                {
                    "auxiliary_effective_weight": lowdose_auxiliary_weight,
                    "clean_hard_label_bce_preserved": True,
                    "feature_invariance_polish": {
                        "space": "managed_classifier_penultimate_feature",
                        "distance": "one_minus_cosine_similarity",
                        "validity": (
                            "exact_matched_eligibility_intersection"
                        ),
                        "projection_head": False,
                        "direct_lhat_label_supervision": False,
                        "typed_objective_slot": (
                            "bernoulli_jsd_replaced_before_objective_scaling"
                        ),
                        "matched_compute_in_control": True,
                    },
                    "matched_control_method_id": (
                        CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID
                    ),
                }
            )
        if candidate.pretrain_epochs or candidate.pretrain_steps:
            method["contracts"]["unlabeled_teacher"] = copy.deepcopy(
                a7_method["contracts"]["unlabeled_teacher"]
            )
            method["contracts"]["stage1_components"] = [
                candidate.pretrain_view_mode,
                "simclr",
                "ptbxl_source_logit_anchor",
            ]
    elif candidate.method_family == "vae_lhat_chain3_polish":
        if candidate.model_name != "efficientnet1dv2":
            raise ValueError(
                "VAE-LHAT chain3 polish must pass the EfficientNet gate before "
                "cross-backbone transfer"
            )
        if candidate.lhat_pgd_epsilon_l2_standardized not in {2.0, 4.0, 8.0}:
            raise ValueError(
                "VAE-LHAT chain3-polish epsilon must be one of 2, 4, or 8"
            )
        method = _load_yaml(STRONG_LHAT_AUGMIX_METHOD)
        method["method"].update(
            {
                "id": VAE_CHAIN3_POLISH_METHOD_ID,
                "scientific_arm": "vae_lhat_chain3_polish",
                "status": "heldout_tuned_development_only",
                "description": (
                    "Single-exposure K500 polish in which exact-label "
                    "VAE-LHAT is used only as AugMix chain three and in the "
                    "three-view Bernoulli JSD. LHAT receives no direct BCE. "
                    "The effective objective is 0.50 clean BCE + 0.25 clean "
                    "preservation BCE + 0.125 AugMix BCE + 0.125 JSD. "
                    "Fixed20, PCGrad, VAE-random, raw auxiliary and source "
                    "replay are absent. "
                    f"Standardized L2 epsilon="
                    f"{candidate.lhat_pgd_epsilon_l2_standardized:g}."
                ),
            }
        )
        method["nodes"].pop("depth23_corruption")
        for output_name in (
            "corrupted_view",
            "corruption_provenance",
        ):
            method["outputs"].pop(output_name)
        method["objective"]["terms"] = [
            {
                "id": "clean_bce",
                "type": "multilabel_bce_with_logits",
                "view": "clean_view",
                "targets": "targets",
                "weight": 1.0,
            },
            {
                "id": "clean_preserve_bce",
                "type": "multilabel_bce_with_logits",
                "view": "clean_view",
                "targets": "targets",
                "weight": 0.5,
            },
            {
                "id": "augmix_bce",
                "type": "multilabel_bce_with_logits",
                "view": "augmix_view",
                "targets": "targets",
                "weight": 0.25,
            },
            {
                "id": "clean_lhat_augmix_jsd",
                "type": "multilabel_bernoulli_jsd",
                "views": ["clean_view", "lhat_view", "augmix_view"],
                "weight": 0.25,
            },
        ]
        for resource_name in ("operator_profile", "corruption_rng"):
            method["resources"].pop(resource_name)
        lhat_config_path = (
            "train/lhat.yaml"
            if candidate.lhat_pgd_epsilon_l2_standardized == 2.0
            else str(_candidate_relroot(candidate) / "lhat.yaml")
        )
        method["resources"]["lhat_config"]["path"] = lhat_config_path
        method["contracts"] = {
            "executable": True,
            "normalization_owner": "online_trainer",
            "optimizer_owner": "online_trainer",
            "canonical_input_sampling_rate_hz": 100,
            "augmix_operator_domain_sampling_rate_hz": 500,
            "canonical_output_sampling_rate_hz": 100,
            "fixed20_train_views": 0,
            "augmix_train_views": 0,
            "one_outer_optimizer_step_per_clean_batch": True,
            "complete_k500_base_record_exposure": True,
            "objective_scale_adapter_mode": "scale_only",
            "objective_global_scale": 0.5,
            "effective_objective_weights": copy.deepcopy(
                CHAIN3_POLISH_EFFECTIVE_WEIGHTS
            ),
            "batch_norm_running_stats_policy": (
                "objective_view_weighted_once_per_base_batch"
            ),
            "batch_norm_objective_view_weights": {
                "clean_view": 0.75,
                "lhat_view": 0.125,
                "augmix_view": 0.125,
            },
            "lhat_label_policy": "exact_positive_set_nonself",
            "lhat_hull_lambda": 0.60,
            "lhat_steps": 5,
            "lhat_pgd_epsilon_l2_standardized": (
                candidate.lhat_pgd_epsilon_l2_standardized
            ),
            "lhat_direct_supervision": False,
            "lhat_view_single_generation_per_record_step": True,
            "augmix_chain3_source": "lhat_view",
            "augmix_chain3_additional_corruption": False,
            "augmix_full_base_batch_random_draw_before_quality_mask": True,
            "quality_rejected_anchor_policy": "clean_loss_only",
            "online_gpu_generation_required": True,
            "materialized_dataset_expansion_allowed": False,
            "post_refine_components": [
                "vae_lhat_exact_label_nonself",
                "lhat_as_augmix_chain3",
                "clean_lhat_augmix_jsd",
            ],
            "post_refine_excluded_components": [
                "fixed20",
                "lhat_direct_bce",
                "pcgrad",
                "raw_auxiliary",
                "vae_random",
                "source_replay",
            ],
            "tuning_partition_only": "k500",
            "full_k500_refit_supported": True,
            "heldout_target_feedback_allowed": False,
        }
    elif candidate.method_family == "vae_lhat_chain3_teacher04_polish":
        if candidate.frozen_teacher_mix != 0.4:
            raise ValueError("J4 requires frozen_teacher_mix=0.4")
        base_candidate = replace(
            candidate,
            method_family="vae_lhat_chain3_polish",
            frozen_teacher_mix=None,
        )
        method = _method_for(base_candidate, center)
        method["method"].update(
            {
                "id": VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID,
                "scientific_arm": "vae_lhat_chain3_teacher04_polish",
                "description": (
                    "J4 single-exposure K500 VAE-LHAT chain3 polish. "
                    "Only AugMix BCE uses a 0.40 frozen Stage-2 initializer "
                    "teacher target; every non-AugMix supervised term remains "
                    "hard-label trained. Objective weights and chain geometry "
                    "remain identical to J3. "
                    f"Candidate={candidate.candidate_id}."
                ),
            }
        )
        method["contracts"].update(
            {
                "auxiliary_label_policy": (
                    "teacher_mix_times_stage2_initializer_probability_plus_"
                    "hard_remainder"
                ),
                "auxiliary_teacher_policy": (
                    "frozen_stage2_initializer_snapshot_before_first_"
                    "optimizer_step"
                ),
                "auxiliary_teacher_mix": candidate.frozen_teacher_mix,
                "auxiliary_teacher_scope": "augmix_bce_only",
                "non_augmix_bce_hard_labels_preserved": True,
                "auxiliary_teacher_initializer_sha256": _sha256(
                    _source_checkpoint(candidate, center)
                ),
                "matched_control_method_id": (
                    CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID
                ),
            }
        )
        return method
    else:
        raise ValueError(f"unsupported method family: {candidate.method_family}")
    profile = method["method"]
    contracts = method["contracts"]
    teacher = contracts.get("unlabeled_teacher")
    if candidate.pretrain_view_mode is not None:
        replay_tag = (
            "source_replay_off"
            if not candidate.pretrain_source_replay_weight
            else (
                "source_replay_"
                f"{candidate.pretrain_source_replay_weight:.2f}".replace(".", "p")
            )
        )
        profile["scientific_arm"] = (
            f"sequential_stage1_{candidate.pretrain_view_mode}_ssl_"
            f"{replay_tag}_then_"
            + (
                "fixed20_plus_mild_lhat_chain3"
                if candidate.method_family == "strong_lhat_augmix_chain3"
                else (
                    "fixed20_plus_vae_random_hard_triview"
                    if candidate.method_family == "d19_separate_views"
                    else (
                        "fixed20_plus_vae_lhat_hard_only"
                        if candidate.method_family == "vae_lhat_post_refine"
                        else (
                            "clean_plus_vae_lhat"
                            if candidate.method_family
                            == "vae_lhat_clean_polish"
                            else (
                                "clean_only"
                                if candidate.method_family
                                == "clean_target_ssl_control"
                                else "fixed20"
                            )
                        )
                    )
                )
            )
        )
    elif candidate.method_family not in {
        "vae_lhat_post_refine",
        "vae_lhat_clean_polish",
        "vae_lhat_chain3_polish",
        "vae_lhat_chain3_teacher04_polish",
        "vae_lhat_bounded_augmax_polish",
    }:
        profile["scientific_arm"] = "a7_global_vae_lhat_augmix_equal5"
    profile["status"] = "heldout_tuned_development_only"
    profile["description"] = (
        f"{profile['description']} Candidate={candidate.candidate_id}."
    )
    if candidate.comparison_rng_identity is not None:
        contracts["comparison_rng_identity"] = candidate.comparison_rng_identity
    if teacher is not None:
        teacher["pool_partition"] = "k500"
        teacher["labels_consumed"] = False
        teacher["pretrain_epochs"] = candidate.pretrain_epochs
        teacher["pretrain_steps_per_epoch"] = candidate.pretrain_steps
        teacher["pretrain_learning_rate"] = candidate.pretrain_learning_rate
        teacher["pretrain_logit_anchor_weight"] = (
            candidate.pretrain_logit_anchor_weight
        )
        teacher["pretrain_logit_anchor_class_weights"] = [1.0] * 5
        teacher["supervised_logit_anchor_weight"] = (
            candidate.supervised_logit_anchor_weight
        )
        teacher["supervised_logit_anchor_class_weights"] = [1.0] * 5
        teacher["dirichlet_alpha"] = candidate.augmix_dirichlet_alpha
        teacher["beta_alpha"] = candidate.augmix_beta_alpha
        teacher["pretrain_ssl_objective"] = candidate.pretrain_ssl_objective
        teacher["pretrain_ssl_weight"] = candidate.pretrain_ssl_weight
        teacher["pretrain_vicreg_mix_weight"] = (
            candidate.pretrain_vicreg_mix_weight
        )
        if candidate.pretrain_view_mode is not None:
            teacher["pretrain_view_mode"] = candidate.pretrain_view_mode
            teacher["policy"] = PRETRAIN_POLICY_BY_VIEW_MODE[
                candidate.pretrain_view_mode
            ]
            if candidate.pretrain_view_mode == CLASSIC_AUGMIX_VIEW_MODE:
                teacher["pretrain_augmix_topology"] = copy.deepcopy(
                    CLASSIC_AUGMIX_TOPOLOGY
                )
            else:
                teacher.pop("pretrain_augmix_topology", None)
        if candidate.pretrain_source_replay_weight is not None:
            teacher["pretrain_source_replay_weight"] = (
                candidate.pretrain_source_replay_weight
            )
        teacher["pretrain_source_batches_per_step"] = (
            candidate.pretrain_source_batches_per_step
        )
        teacher["online_enabled"] = False
    elif candidate.pretrain_epochs or candidate.pretrain_steps:
        raise ValueError(
            f"{candidate.method_family} does not support a Stage-1 pretrain phase"
        )
    if candidate.method_family == "direct_budget_filler":
        assert teacher is not None
        teacher.update(
            {
                "labels_consumed": True,
                "total_weight": 0.0,
                "strong_views": 1,
                "unlabeled_batches_per_clean": 1,
                "simclr_weight": 0.0,
                "pretrain_clean_anchor": True,
                "pretrain_logit_anchor_weight": 0.0,
                "supervised_logit_anchor_weight": 0.0,
                "pretrain_prototype_weight": 0.0,
                "pretrain_prototype_bank_batches": 0,
                "pretrain_rank_weight": 0.0,
                "pretrain_strong_pseudo_weight": 0.0,
                "residual_head_mode": "none",
                "residual_head_freeze_base": False,
                "pretrain_ssl_objective": "supervised_bce_budget_filler",
                "pretrain_ssl_weight": 1.0,
                "pretrain_source_replay_weight": 0.0,
                "pretrain_source_batches_per_step": 1,
            }
        )
    if candidate.method_family in {"d19_separate_views", "checkpoint_blend"}:
        contracts["pcgrad_auxiliary_alpha"] = candidate.auxiliary_alpha
        contracts["exposure_policy"] = (
            "clean_aux_once_then_exhaustive_depth23"
        )
        auxiliary_scales = {
            "corruption_1_bce": candidate.raw_auxiliary_scale,
            "compat_random_bce": candidate.vae_random_auxiliary_scale,
            "compat_hard_bce": candidate.vae_hard_auxiliary_scale,
            "clean_raw_random_hard_jsd": candidate.jsd_auxiliary_scale,
        }
        if any(abs(value - 1.0) > 1.0e-12 for value in auxiliary_scales.values()):
            contracts["pcgrad_auxiliary_term_scales"] = auxiliary_scales
            contracts["pcgrad_auxiliary_bce_mass_preserved"] = True
    elif candidate.method_family == "strong_lhat_augmix_chain3":
        contracts["pcgrad_auxiliary_alpha"] = candidate.auxiliary_alpha
    elif candidate.method_family == "vae_lhat_post_refine":
        contracts["pcgrad_auxiliary_alpha"] = candidate.auxiliary_alpha
    if candidate.augmax_hard_hull_lambda is not None:
        if candidate.method_family != "d19_separate_views":
            raise ValueError(
                "AugMax hard geometry is restricted to the D19 separate-view method"
            )
        if candidate.augmax_hard_hull_lambda not in {0.10, 0.15}:
            raise ValueError("AugMax hard hull lambda must be 0.10 or 0.15")
        contracts.update(
            {
                "augmax_hard_search_loss": "multilabel_bce",
                "augmax_hard_search_loss_scope": "coefficient_search_only",
                "augmax_hard_search_positive_requirement": (
                    "at_least_one_positive_per_record"
                ),
                "augmax_hard_search_geometry_override": {
                    "scope": "compatible_d19_generated_views_only",
                    "hull_lambda": candidate.augmax_hard_hull_lambda,
                    "hard_steps": 5,
                    "hard_learning_rate": 0.25,
                    "pgd_epsilon": 2.0,
                },
            }
        )
    requires_legacy_oracle = (
        candidate.method_family
        in {
            "d19_separate_views",
            "checkpoint_blend",
            "direct_budget_filler",
        }
        or (
            candidate.method_family == "strong_lhat_augmix_chain3"
            and candidate.pretrain_view_mode is None
        )
        or (
            candidate.method_family == "direct_target_ssl_control"
            and candidate.pretrain_view_mode is None
        )
    )
    if requires_legacy_oracle and not isinstance(
        contracts.get("full_k500_heldout_oracle"), dict
    ):
        raise ValueError("legacy A7 full-K500 oracle contract must remain a mapping")
    if not requires_legacy_oracle and "full_k500_heldout_oracle" in contracts:
        raise ValueError("sequential factorial methods must not carry heldout oracle")
    contracts["heldout_target_feedback_allowed"] = False
    return _apply_supervised_corruption_schedule(method, candidate)


def _direct_method_for(candidate: Candidate, center: str) -> dict[str, Any]:
    if candidate.method_family == "clean_target_ssl_control":
        method = _method_for(candidate, center)
        method["method"]["description"] = (
            f"{method['method']['description']} Self-matched clean-only "
            f"control for {candidate.candidate_id}."
        )
        return method
    if (
        candidate.method_family == "vae_lhat_clean_polish"
        and (candidate.pretrain_epochs or candidate.pretrain_steps)
    ):
        control_candidate = replace(
            candidate,
            method_family="clean_target_ssl_control",
            auxiliary_alpha=1.0,
            frozen_teacher_mix=None,
            polish_auxiliary_target=None,
        )
        method = _method_for(control_candidate, center)
        method["method"]["description"] = (
            f"{method['method']['description']} Stage-1-, optimizer-, "
            "source-, seed-, K500- and epoch-matched clean-only control for "
            f"{candidate.candidate_id}; the supervised VAE-LHAT branch alone "
            "is absent."
        )
        method["contracts"]["matched_candidate_method_id"] = (
            VAE_CLEAN_DIRECTSUM_METHOD_ID
            if candidate.clean_direct_sum_auxiliary
            else VAE_CLEAN_POLISH_METHOD_ID
        )
        return method
    if candidate.method_family == "vae_lhat_bounded_augmax_polish":
        method = _method_for(candidate, center)
        method["method"].update(
            {
                "id": CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID,
                "scientific_arm": "clean_bounded_augmax_polish_control",
                "status": "heldout_tuned_development_only",
                "description": (
                    "Strict bounded AugMax-M20 control. It executes the same "
                    "LHAT candidate search, PGD, decode, hard QC, twenty raw "
                    "chain pairs, six-way coefficient search, classifier "
                    "forwards, RNG streams, objective, BatchNorm policy, "
                    "optimizer, and record order as the candidate. Only after "
                    "LHAT search and QC is its accepted waveform replaced by "
                    "the aligned clean waveform before chain-three mixing."
                ),
            }
        )
        contracts = method["contracts"]
        contracts.pop("matched_control_method_id")
        contracts.update(
            {
                "matched_candidate_method_id": (
                    VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID
                ),
                "matched_chain3_control_mode": (
                    "replace_lhat_with_clean_after_full_search_and_qc"
                ),
                "control_executes_full_lhat_search": True,
                "post_refine_components": [
                    "online_depth23_m20_coverage",
                    "full_matched_vae_lhat_compute",
                    "clean_replacement_after_lhat_qc",
                    "bounded_per_record_augmax_coefficient_search",
                    "classic_bernoulli_jsd",
                ],
            }
        )
        return method
    if (
        candidate.method_family == "strong_lhat_augmix_chain3"
        and candidate.pretrain_view_mode is not None
    ):
        control_candidate = replace(
            candidate,
            method_family="direct_target_ssl_control",
            auxiliary_alpha=1.0,
            pcgrad_auxiliary_norm_ratio_cap=None,
            lhat_search_mode="optimized_softmax",
        )
        method = _method_for(control_candidate, center)
        method["method"]["description"] = (
            f"{method['method']['description']} Optimizer-, source-, seed-, "
            "K500-, pretraining-, fixed20- and epoch-matched control for "
            f"{candidate.candidate_id}; the VAE-LHAT third-chain auxiliary "
            "alone is absent."
        )
        matched_candidate = _method_for(candidate, center)
        method["contracts"]["matched_candidate_method_id"] = (
            matched_candidate["method"]["id"]
        )
        method["contracts"]["matched_candidate_auxiliary_alpha"] = (
            candidate.auxiliary_alpha
        )
        method["contracts"]["matched_candidate_lhat_search_mode"] = (
            candidate.lhat_search_mode
        )
        if candidate.pcgrad_auxiliary_norm_ratio_cap is not None:
            method["contracts"]["matched_candidate_auxiliary_norm_ratio_cap"] = (
                candidate.pcgrad_auxiliary_norm_ratio_cap
            )
        return method
    if (
        candidate.method_family == "d19_separate_views"
        and candidate.pretrain_view_mode is not None
    ):
        control_candidate = replace(
            candidate,
            method_family="direct_target_ssl_control",
            auxiliary_alpha=1.0,
            raw_auxiliary_scale=1.0,
            vae_random_auxiliary_scale=1.0,
            vae_hard_auxiliary_scale=1.0,
            jsd_auxiliary_scale=1.0,
        )
        method = _method_for(control_candidate, center)
        method["method"]["description"] = (
            f"{method['method']['description']} Optimizer-, source-, seed-, "
            "K500-, pretraining-, fixed20- and epoch-matched control for "
            f"{candidate.candidate_id}; the post-training raw-reference, "
            "VAE-random, VAE-LHAT-hard and JSD auxiliary views alone are "
            "absent."
        )
        matched_candidate = _method_for(candidate, center)
        method["contracts"]["matched_candidate_method_id"] = (
            matched_candidate["method"]["id"]
        )
        method["contracts"]["matched_candidate_auxiliary_alpha"] = (
            candidate.auxiliary_alpha
        )
        return method
    method = _load_yaml(
        STRONG_LHAT_AUGMIX_METHOD
        if candidate.method_family
        in {
            "vae_lhat_chain3_polish",
            "vae_lhat_chain3_teacher04_polish",
        }
        else DIRECT_FIXED20_METHOD
    )
    teacher_distill = candidate.frozen_teacher_mix is not None
    hard_label_lowdose = candidate.polish_auxiliary_target == "hard_label"
    latent_path_lowdose = (
        candidate.polish_auxiliary_target == "latent_path_hard_label"
    )
    consistency_lowdose = (
        candidate.polish_auxiliary_target == "consistency"
    )
    feature_invariance_lowdose = (
        candidate.polish_auxiliary_target == "feature_invariance"
    )
    local_anchor_soft_lowdose = (
        candidate.polish_auxiliary_target == "local_anchor_soft"
    )
    local_anchor_soft_geometry = (
        _local_anchor_soft_geometry(candidate)
        if local_anchor_soft_lowdose
        else None
    )
    local_anchor_soft_method_ids = (
        _local_anchor_soft_method_ids(candidate)
        if local_anchor_soft_lowdose
        else (None, None)
    )
    if (
        candidate.method_family == "vae_lhat_clean_polish"
        and (
            teacher_distill
            or hard_label_lowdose
            or latent_path_lowdose
            or consistency_lowdose
            or feature_invariance_lowdose
            or local_anchor_soft_lowdose
        )
    ):
        lowdose_auxiliary_weight = float(candidate.auxiliary_alpha)
        candidate_auxiliary_term = (
            "lhat_teacher_distill_bce"
            if teacher_distill
            else (
                "lhat_path_hard_bce"
                if latent_path_lowdose
                else (
                    "lhat_anchor_soft_bce"
                    if local_anchor_soft_lowdose
                    else (
                        "lhat_hard_bce"
                        if hard_label_lowdose
                        else (
                            "clean_lhat_feature_cosine"
                            if feature_invariance_lowdose
                            else "clean_lhat_consistency_jsd"
                        )
                    )
                )
            )
        )
        control_auxiliary_term = (
            "clean_teacher_distill_control_bce"
            if teacher_distill
            else (
                "clean_lhat_path_control_bce"
                if latent_path_lowdose
                else (
                    "clean_anchor_soft_control_bce"
                    if local_anchor_soft_lowdose
                    else (
                        "clean_hard_control_bce"
                        if hard_label_lowdose
                        else (
                            "clean_identity_feature_cosine"
                            if feature_invariance_lowdose
                            else "clean_identity_consistency_jsd"
                        )
                    )
                )
            )
        )
        method = _method_for(candidate, center)
        method["method"].update(
            {
                "id": (
                    CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID
                    if teacher_distill
                    else (
                        CLEAN_PATH_BCE_CONTROL_METHOD_ID
                        if latent_path_lowdose
                        else (
                            CLEAN_HARD_BCE_CONTROL_METHOD_ID
                            if hard_label_lowdose
                            else (
                                local_anchor_soft_method_ids[1]
                                if local_anchor_soft_lowdose
                                else (
                                    CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID
                                    if feature_invariance_lowdose
                                    else CLEAN_CONSISTENCY_CONTROL_METHOD_ID
                                )
                            )
                        )
                    )
                ),
                "scientific_arm": (
                    "matched_clean_teacher_distill_polish_control"
                    if teacher_distill
                    else (
                        "matched_clean_lhat_path_bce_polish_control"
                        if latent_path_lowdose
                        else (
                            "matched_clean_hard_bce_polish_control"
                            if hard_label_lowdose
                            else (
                                "matched_clean_local_anchor_soft_polish_control"
                                if local_anchor_soft_lowdose
                                else (
                                    "matched_clean_feature_invariance_polish_control"
                                    if feature_invariance_lowdose
                                    else "matched_clean_consistency_polish_control"
                                )
                            )
                        )
                    )
                ),
                "description": (
                    "Matched identity control. The Stage-2 initializer, "
                    + (
                        "frozen teacher, "
                        if teacher_distill
                        else ""
                    )
                    + "latent-pool eligibility, two model forwards, objective "
                    "weights, and BatchNorm weights match the VAE-LHAT "
                    "candidate; the auxiliary waveform alone is replaced by "
                    "clean identity."
                ),
            }
        )
        if local_anchor_soft_lowdose:
            # Force the matched control to execute the same LHAT search/decode
            # graph. The scoped runtime replaces this output with clean
            # identity only after preserving LHAT eligibility and hard-QC
            # accounting.
            method["nodes"]["lhat"]["outputs"]["waveform"] = (
                "lhat_control_view"
            )
            method["outputs"].pop("lhat_view")
            method["outputs"]["lhat_control_view"] = {
                "type": "ecg_waveform",
                "source": "lhat.waveform",
                "domain": "raw_mV",
                "sampling_rate_hz": 100,
                "points": 1000,
            }
        else:
            method["nodes"]["lhat_control_identity"] = {
                "type": "identity_raw100_view",
                "implementation": "builtin",
                "implementation_version": 1,
                "status": "implemented",
                "inputs": {"waveform": "clean_raw"},
                "outputs": {"waveform": "lhat_control_view"},
            }
            method["outputs"]["lhat_control_view"] = {
                "type": "ecg_waveform",
                "source": "lhat_control_identity.waveform",
                "domain": "raw_mV",
                "sampling_rate_hz": 100,
                "points": 1000,
            }
        for term in method["objective"]["terms"]:
            if term["id"] == candidate_auxiliary_term:
                if consistency_lowdose or feature_invariance_lowdose:
                    term.update(
                        {
                            "id": control_auxiliary_term,
                            "views": ["clean_view", "lhat_control_view"],
                        }
                    )
                else:
                    term.update(
                        {
                            "id": control_auxiliary_term,
                            "view": "lhat_control_view",
                        }
                    )
        contracts = method["contracts"]
        contracts.pop("matched_control_method_id", None)
        contracts["post_refine_components"] = [
            (
                "matched_identity_frozen_teacher_distillation"
                if teacher_distill
                else (
                    "matched_identity_lhat_path_bce"
                    if latent_path_lowdose
                    else (
                    "matched_identity_hard_bce"
                    if hard_label_lowdose
                    else (
                        "matched_identity_local_anchor_soft_bce"
                        if local_anchor_soft_lowdose
                        else (
                            "matched_identity_penultimate_feature_cosine"
                            if feature_invariance_lowdose
                            else "matched_identity_consistency_jsd"
                        )
                    )
                    )
                )
            )
        ]
        contracts.update(
            {
                "batch_norm_objective_view_weights": {
                    "clean_view": 1.0 - lowdose_auxiliary_weight,
                    "lhat_control_view": lowdose_auxiliary_weight,
                },
                "effective_objective_weights": {
                    **(
                        {
                            "clean_bce": 0.50,
                            "clean_preserve_bce": (
                                0.50 - lowdose_auxiliary_weight
                            ),
                            control_auxiliary_term: lowdose_auxiliary_weight,
                        }
                        if consistency_lowdose or feature_invariance_lowdose
                        else {
                            "clean_bce": 0.50,
                            "clean_preserve_bce": (
                                0.50 - lowdose_auxiliary_weight
                            ),
                            control_auxiliary_term: lowdose_auxiliary_weight,
                        }
                    ),
                },
                "matched_candidate_method_id": (
                    VAE_TEACHER_DISTILL_POLISH_METHOD_ID
                    if teacher_distill
                    else (
                        VAE_PATH_BCE_POLISH_METHOD_ID
                        if latent_path_lowdose
                        else (
                            VAE_HARD_BCE_POLISH_METHOD_ID
                            if hard_label_lowdose
                            else (
                                local_anchor_soft_method_ids[0]
                                if local_anchor_soft_lowdose
                                else (
                                    VAE_FEATURE_INVARIANCE_POLISH_METHOD_ID
                                    if feature_invariance_lowdose
                                    else VAE_CONSISTENCY_POLISH_METHOD_ID
                                )
                            )
                        )
                    )
                ),
            }
        )
        if local_anchor_soft_lowdose:
            contracts["control_executes_full_lhat_search"] = True
        else:
            contracts[
                "unused_lhat_node_retained_for_latent_pool_construction"
            ] = True
        if teacher_distill:
            contracts["auxiliary_teacher_scope"] = (
                "clean_identity_distill_control_bce_only"
            )
        else:
            if latent_path_lowdose:
                contracts["hard_auxiliary_target_policy"] = (
                    "anchor_ground_truth_multihot"
                )
                contracts["unused_lhat_path_computation_for_matched_compute"] = (
                    True
                )
            elif hard_label_lowdose:
                contracts["hard_auxiliary_target_policy"] = (
                    "anchor_ground_truth_multihot"
                )
            elif local_anchor_soft_lowdose:
                contracts["anchor_soft_target_policy"] = (
                    "anchor_soft_exact_positive_set_no_class_admission"
                )
                contracts["auxiliary_positive_value"] = 0.95
                contracts["auxiliary_negative_value"] = 0.0
                contracts["auxiliary_soft_bce_pos_weight"] = None
                contracts["local_anchor_soft_polish"] = {
                    "candidate_policy": "exact_positive_set_nonself",
                    "candidate_count": 20,
                    "candidate_includes_anchor": False,
                    "anchor_in_outer_interpolation": True,
                    "local_pool_size": 80,
                    "hull_lambda": local_anchor_soft_geometry["hull_lambda"],
                    "pgd_epsilon_l2_standardized": (
                        local_anchor_soft_geometry[
                            "pgd_epsilon_l2_standardized"
                        ]
                    ),
                    "steps": local_anchor_soft_geometry["steps"],
                    "learning_rate": local_anchor_soft_geometry[
                        "learning_rate"
                    ],
                    "minimum_effective_anchor_mass": (
                        local_anchor_soft_geometry[
                            "minimum_effective_anchor_mass"
                        ]
                    ),
                    "matched_compute_in_control": True,
                }
                contracts["local_lhat_geometry_preset"] = (
                    local_anchor_soft_geometry["preset"]
                )
            elif feature_invariance_lowdose:
                contracts["feature_invariance_polish"] = {
                    "space": "managed_classifier_penultimate_feature",
                    "distance": "one_minus_cosine_similarity",
                    "validity": (
                        "exact_matched_eligibility_intersection"
                    ),
                    "projection_head": False,
                    "direct_lhat_label_supervision": False,
                    "typed_objective_slot": (
                        "bernoulli_jsd_replaced_before_objective_scaling"
                    ),
                    "matched_compute_in_control": True,
                }
            else:
                contracts["clean_lhat_consistency_policy"] = (
                    "symmetric_bernoulli_jsd_current_model"
                )
        return method
    if candidate.method_family == "vae_lhat_clean_polish":
        method["method"].update(
            {
                "id": "clean_polish_control_v1",
                "scientific_arm": "matched_clean_only_polish_control",
                "status": "heldout_tuned_development_only",
                "description": (
                    "Single-exposure clean-only K500 polish matched to "
                    f"{candidate.candidate_id} on initializer, record order, "
                    "optimizer, learning-rate schedule, epoch count and "
                    "optimizer-step budget."
                ),
            }
        )
        method["nodes"].pop("depth23_corruption")
        for output_name in ("corrupted_view", "corruption_provenance"):
            method["outputs"].pop(output_name)
        method["objective"]["terms"] = [
            term
            for term in method["objective"]["terms"]
            if term["id"] == "clean_bce"
        ]
        for resource_name in ("operator_profile", "corruption_rng"):
            method["resources"].pop(resource_name)
        method["contracts"] = {
            "executable": True,
            "normalization_owner": "online_trainer",
            "optimizer_owner": "online_trainer",
            "canonical_input_sampling_rate_hz": 100,
            "canonical_output_sampling_rate_hz": 100,
            "fixed20_train_views": 0,
            "augmix_train_views": 0,
            "one_outer_optimizer_step_per_clean_batch": True,
            "complete_k500_base_record_exposure": True,
            "effective_objective_weights": {"clean_bce": 1.0},
            "tuning_partition_only": "k500",
            "full_k500_refit_supported": True,
            "heldout_target_feedback_allowed": False,
        }
    elif candidate.method_family == "vae_lhat_chain3_teacher04_polish":
        method = _method_for(candidate, center)
        method["method"].update(
            {
                "id": CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID,
                "scientific_arm": (
                    "matched_clean_chain3_teacher04_polish_control"
                ),
                "description": (
                    "J4 matched chain3 control. The VAE-LHAT node and all "
                    "VAE/latent/LHAT resources remain present so latent-pool "
                    "construction and initialization are matched, but AugMix "
                    "chain three and the JSD third view consume a clean_raw "
                    "identity view. Only AugMix BCE uses the frozen Stage-2 "
                    "initializer teacher target."
                ),
            }
        )
        method["nodes"]["chain3_control_identity"] = {
            "type": "identity_raw100_view",
            "implementation": "builtin",
            "implementation_version": 1,
            "status": "implemented",
            "inputs": {"waveform": "clean_raw"},
            "outputs": {"waveform": "chain3_control_view"},
        }
        method["nodes"]["threechain_augmix"]["inputs"]["chain3_waveform"] = (
            "chain3_control_view"
        )
        method["outputs"]["chain3_control_view"] = {
            "type": "ecg_waveform",
            "source": "chain3_control_identity.waveform",
            "domain": "raw_mV",
            "sampling_rate_hz": 100,
            "points": 1000,
        }
        for term in method["objective"]["terms"]:
            if term["id"] == "clean_lhat_augmix_jsd":
                term.update(
                    {
                        "id": "clean_control_augmix_jsd",
                        "views": [
                            "clean_view",
                            "chain3_control_view",
                            "augmix_view",
                        ],
                    }
                )
        contracts = method["contracts"]
        contracts.pop("matched_control_method_id", None)
        contracts.update(
            {
                "batch_norm_objective_view_weights": {
                    "clean_view": 0.75,
                    "chain3_control_view": 0.125,
                    "augmix_view": 0.125,
                },
                "chain3_control_source": "clean_raw_identity",
                "augmix_chain3_source": "chain3_control_view",
                "unused_lhat_node_retained_for_latent_pool_construction": True,
                "matched_candidate_method_id": (
                    VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID
                ),
            }
        )
    elif candidate.method_family == "vae_lhat_chain3_polish":
        method["method"].update(
            {
                "id": CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID,
                "scientific_arm": "matched_clean_chain3_polish_control",
                "status": "heldout_tuned_development_only",
                "description": (
                    "Single-exposure chain3 control matched to "
                    f"{candidate.candidate_id}. The third AugMix chain and "
                    "the corresponding JSD view are a second identity view "
                    "of clean_raw instead of VAE-LHAT. Raw chain1/chain2, "
                    "AugMix RNG, objective weights, three model forwards, "
                    "initializer, record order, optimizer and step budget "
                    "remain matched."
                ),
            }
        )
        method["nodes"].pop("depth23_corruption")
        method["nodes"].pop("lhat")
        method["nodes"]["chain3_control_identity"] = {
            "type": "identity_raw100_view",
            "implementation": "builtin",
            "implementation_version": 1,
            "status": "implemented",
            "inputs": {"waveform": "clean_raw"},
            "outputs": {"waveform": "chain3_control_view"},
        }
        method["nodes"]["threechain_augmix"]["inputs"]["chain3_waveform"] = (
            "chain3_control_view"
        )
        for output_name in (
            "corrupted_view",
            "corruption_provenance",
            "lhat_view",
            "lhat_diagnostics",
        ):
            method["outputs"].pop(output_name)
        method["outputs"]["chain3_control_view"] = {
            "type": "ecg_waveform",
            "source": "chain3_control_identity.waveform",
            "domain": "raw_mV",
            "sampling_rate_hz": 100,
            "points": 1000,
        }
        method["objective"]["terms"] = [
            {
                "id": "clean_bce",
                "type": "multilabel_bce_with_logits",
                "view": "clean_view",
                "targets": "targets",
                "weight": 1.0,
            },
            {
                "id": "clean_preserve_bce",
                "type": "multilabel_bce_with_logits",
                "view": "clean_view",
                "targets": "targets",
                "weight": 0.5,
            },
            {
                "id": "augmix_bce",
                "type": "multilabel_bce_with_logits",
                "view": "augmix_view",
                "targets": "targets",
                "weight": 0.25,
            },
            {
                "id": "clean_control_augmix_jsd",
                "type": "multilabel_bernoulli_jsd",
                "views": [
                    "clean_view",
                    "chain3_control_view",
                    "augmix_view",
                ],
                "weight": 0.25,
            },
        ]
        for resource_name in (
            "vae",
            "latent_pool",
            "vae_decoder",
            "lhat_config",
            "operator_profile",
            "corruption_rng",
            "lhat_rng",
        ):
            method["resources"].pop(resource_name)
        method["contracts"] = {
            "executable": True,
            "normalization_owner": "online_trainer",
            "optimizer_owner": "online_trainer",
            "canonical_input_sampling_rate_hz": 100,
            "augmix_operator_domain_sampling_rate_hz": 500,
            "canonical_output_sampling_rate_hz": 100,
            "fixed20_train_views": 0,
            "augmix_train_views": 0,
            "one_outer_optimizer_step_per_clean_batch": True,
            "complete_k500_base_record_exposure": True,
            "objective_scale_adapter_mode": "scale_only",
            "objective_global_scale": 0.5,
            "effective_objective_weights": copy.deepcopy(
                CHAIN3_POLISH_EFFECTIVE_WEIGHTS
            ),
            "batch_norm_running_stats_policy": (
                "objective_view_weighted_once_per_base_batch"
            ),
            "batch_norm_objective_view_weights": {
                "clean_view": 0.75,
                "chain3_control_view": 0.125,
                "augmix_view": 0.125,
            },
            "chain3_control_source": "clean_raw_identity",
            "augmix_chain3_source": "chain3_control_view",
            "augmix_chain3_additional_corruption": False,
            "augmix_full_base_batch_random_draw_before_quality_mask": True,
            "online_gpu_generation_required": True,
            "materialized_dataset_expansion_allowed": False,
            "matched_candidate_method_id": VAE_CHAIN3_POLISH_METHOD_ID,
            "tuning_partition_only": "k500",
            "full_k500_refit_supported": True,
            "heldout_target_feedback_allowed": False,
        }
    if candidate.comparison_rng_identity is not None:
        method["contracts"]["comparison_rng_identity"] = (
            candidate.comparison_rng_identity
        )
    if candidate.method_family not in {
        "vae_lhat_clean_polish",
        "vae_lhat_chain3_polish",
        "vae_lhat_chain3_teacher04_polish",
        "vae_lhat_bounded_augmax_polish",
    }:
        method["method"]["description"] = (
            f"{method['method']['description']} Matched control for "
            f"{candidate.candidate_id}."
        )
    return method


def _online_for(candidate: Candidate, center: str) -> dict[str, Any]:
    online = _load_yaml(
        _base_a7_root(candidate) / "online" / f"{center}.yaml"
    )
    if candidate.supervised_source_replay_weight is None:
        online["profile_name"] = f"a7_global_{candidate.candidate_id}_{center}"
    else:
        if abs(candidate.supervised_source_replay_weight - 0.10) > 1.0e-12:
            raise ValueError(
                "supervised source replay currently supports only the "
                "predeclared total weight 0.10"
            )
        online["profile_name"] = (
            "pn2021_fixed20_source_replay_w010_sandbox"
        )
    if candidate.initializer_candidate is not None:
        online["references"]["source_baseline_registry"] = str(
            _candidate_relroot(candidate) / "baselines" / f"{center}.yaml"
        )
    protocol = online["protocol"]
    protocol["partition"] = "k500"
    protocol["use_all_k500"] = True
    protocol["validation_split"] = False
    protocol["heldout_ref_exclusion_required"] = True
    protocol["mapping_version"] = MAPPING_VERSION
    protocol["mapping_hash"] = MAPPING_HASH
    protocol["class_order"] = list(CLASS_ORDER)
    if candidate.replicate_id < 0:
        raise ValueError("replicate_id must be non-negative")
    online["random_seed"]["replicate_id"] = candidate.replicate_id
    training = online["training"]
    profile = training["model_profiles"][candidate.model_name]
    profile["epochs"] = candidate.epochs
    profile["batch_size"] = candidate.batch_size
    profile["learning_rate"] = candidate.learning_rate
    profile["weight_decay"] = candidate.weight_decay
    # Search runs keep complete epoch aggregates and final checkpoints while
    # avoiding a device-to-host synchronization on nearly every fixed20 base
    # group.  The managed trainer accumulates attack/quality diagnostics
    # independently of these observer intervals, so this changes monitoring
    # overhead only, not samples, RNG streams, losses, or optimizer steps.
    online["diagnostics"]["diagnostics_every_steps"] = 1000
    online["diagnostics"]["performance_timing"]["interval_steps"] = 40
    tensorboard = online["logging"]["tensorboard"]
    tensorboard["scalars"]["batch_loss_interval_steps"] = 1000
    tensorboard["ecg_figures"]["enabled"] = False
    tensorboard["ecg_figures"]["save_png"] = False
    tensorboard["ecg_figures"]["save_raw_arrays"] = False
    online["output"]["root_dir"] = str(
        _persistent_root(candidate) / "candidate" / center / "_online_placeholder"
    )
    online["output"]["checkpoint_write_policy"] = "final_only"
    return online


def _direct_training_config_for(
    candidate: Candidate, center: str
) -> dict[str, Any]:
    # Keep loader, diagnostics, TensorBoard cadence, output policy and every
    # optimizer-facing field byte-for-byte aligned with the candidate arm.
    # The method YAML and output directory remain the only arm-specific inputs.
    return _online_for(candidate, center)


def _candidate_experiments(
    candidate: Candidate,
    center: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relroot = _candidate_relroot(candidate)
    train = _load_yaml(
        _base_a7_root(candidate) / "experiments" / f"{center}_train.yaml"
    )
    train["experiment"]["name"] = (
        f"a7_global_{candidate.candidate_id}_{center}_train"
    )
    if (
        candidate.supervised_corruption_schedule
        == ROTATING4_SUPERVISED_SCHEDULE
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR pretraining followed by clean "
            "plus deterministic rotating corruption supervision: two depth-2 "
            "and two depth-3 canonical views per epoch, with per-record "
            "SHA256 offsets and complete twenty-composition coverage every "
            "five epochs. "
            + (
                "One exact-label online VAE-LHAT hard view adds direct BCE "
                "and clean/hard Bernoulli-JSD at multiplier 1.5."
                if candidate.method_family == "vae_lhat_post_refine"
                else "No VAE/LHAT branch is active."
            )
        )
    elif (
        candidate.pretrain_view_mode == "clean_identity"
        and candidate.method_family == "strong_lhat_augmix_chain3"
    ):
        purpose = (
            "K500-only clean-identity SimCLR/VICReg control pretraining "
            "followed by E20 Direct+fixed20 plus a low-dose exact-label "
            "VAE-LHAT ordinary AugMix third-chain auxiliary."
        )
    elif (
        candidate.pretrain_view_mode == "twochain_augmix"
        and candidate.method_family == "strong_lhat_augmix_chain3"
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR/VICReg pretraining followed "
            "by E20 Direct+fixed20 plus a low-dose exact-label VAE-LHAT "
            "ordinary AugMix third-chain auxiliary."
        )
    elif (
        candidate.pretrain_view_mode == "twochain_augmix"
        and candidate.method_family == "d19_separate_views"
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR/VICReg pretraining followed "
            "by E20 Direct+fixed20 plus one post-training raw corruption "
            "reference, one VAE-random view, one VAE-LHAT-hard view and "
            "their Bernoulli-JSD consistency. GroupDRO, SupCon, pair ranking, "
            "class weighting, residual heads and checkpoint blending are off."
        )
    elif candidate.pretrain_view_mode == "clean_identity":
        purpose = (
            "K500-only matched clean-clean SimCLR/VICReg control followed by "
            "Direct+fixed20 supervised adaptation."
        )
    elif (
        candidate.pretrain_view_mode == CLASSIC_AUGMIX_VIEW_MODE
        and candidate.method_family == "vae_lhat_post_refine"
    ):
        purpose = (
            "K500-only clean-versus-classic-AugMix SimCLR pretraining. Each "
            "AugMix view mixes three independent ECG corruption chains; every "
            "chain samples depth uniformly from 1 to 3 and operators uniformly "
            "with replacement, using Dirichlet(1) and Beta(1). This is followed "
            "by Direct+fixed20 plus one physically pruned VAE-LHAT hard view and "
            "clean/LHAT Bernoulli-JSD. No raw auxiliary or VAE-random view is "
            "generated during supervised adaptation."
        )
    elif (
        candidate.pretrain_view_mode == "twochain_augmix"
        and candidate.method_family == "vae_lhat_post_refine"
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR pretraining followed by "
            "Direct+fixed20 plus one physically pruned VAE-LHAT hard view and "
            "clean/LHAT Bernoulli-JSD. No raw auxiliary or VAE-random view is "
            "generated during supervised adaptation."
        )
    elif (
        candidate.pretrain_view_mode == "twochain_augmix"
        and candidate.method_family == "clean_target_ssl_control"
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR pretraining followed by "
            "clean-only supervised adaptation. Fixed20 and every VAE/LHAT "
            "branch are absent."
        )
    elif (
        candidate.pretrain_view_mode == "twochain_augmix"
        and candidate.method_family == "vae_lhat_clean_polish"
    ):
        purpose = (
            "K500-only two-chain AugMix SimCLR pretraining followed by "
            "single-exposure clean plus one exact-label VAE-LHAT hard view "
            "and clean/LHAT Bernoulli JSD. The VAE auxiliary is multiplied "
            f"by {candidate.auxiliary_alpha:g}. Fixed20, "
            "supervised-stage AugMix, conflict projection, raw auxiliary and "
            "VAE-random are absent."
        )
    elif candidate.pretrain_view_mode == "twochain_augmix":
        purpose = (
            "K500-only AugMix SimCLR/VICReg pretraining followed by "
            "Direct+fixed20 supervised adaptation."
        )
    elif candidate.method_family == "vae_lhat_post_refine":
        purpose = (
            "K500-only short VAE-LHAT post-training refinement with no AugMix, "
            "raw auxiliary, VAE-random, or source-replay branch."
        )
    elif candidate.method_family == "vae_lhat_clean_polish":
        purpose = (
            (
                "K500-only single-exposure clean plus direct VAE-LHAT frozen-"
                "initializer distillation polish. The outer LHAT objective "
                "uses pure soft targets without hard-label remainder, "
                "pos_weight, JSD, or AugMix."
                if candidate.frozen_teacher_mix is not None
                else (
                    "K500-only single-exposure clean plus one uniformly sampled "
                    "point on the standardized clean-to-LHAT latent path, with "
                    "exact-label hard BCE. There is no teacher, JSD, fixed20, "
                    "post-stage AugMix, PCGrad, raw auxiliary, VAE-random or "
                    "source replay."
                    if candidate.polish_auxiliary_target
                    == "latent_path_hard_label"
                    else (
                        "K500-only single-exposure clean plus direct VAE-LHAT "
                        "exact-label hard-BCE polish. There is no teacher, JSD, "
                        "fixed20, post-stage AugMix, PCGrad, raw auxiliary, "
                        "VAE-random or source replay."
                        if candidate.polish_auxiliary_target == "hard_label"
                        else (
                            "K500-only single-exposure clean plus local "
                            "anchor-dominant exact-label VAE-LHAT anchor-soft "
                            "polish. There is no teacher, JSD, fixed20, "
                            "post-stage AugMix, PCGrad, raw auxiliary, "
                            "VAE-random or source replay."
                            if candidate.polish_auxiliary_target
                            == "local_anchor_soft"
                            else (
                                "K500-only single-exposure clean plus direct VAE-LHAT "
                                "prediction-consistency polish. The LHAT endpoint has "
                                "no direct label or teacher loss, and there is no "
                                "fixed20, post-stage AugMix, PCGrad, raw auxiliary, "
                                "VAE-random or source replay."
                                if candidate.polish_auxiliary_target == "consistency"
                                else (
                                    "K500-only single-exposure clean plus direct "
                                    "VAE-LHAT penultimate-feature invariance polish. "
                                    "The endpoint has no direct label, teacher or "
                                    "prediction-JSD loss, and there is no fixed20, "
                                    "post-stage AugMix, PCGrad, raw auxiliary, "
                                    "VAE-random or source replay."
                                    if candidate.polish_auxiliary_target
                                    == "feature_invariance"
                                    else
                                    "K500-only single-exposure clean plus VAE-LHAT "
                                    "short polish with no fixed20, post-stage AugMix, "
                                    "PCGrad, raw auxiliary, VAE-random or source replay."
                                )
                            )
                        )
                    )
                )
            )
        )
    elif candidate.method_family == "vae_lhat_chain3_polish":
        purpose = (
            "K500-only single-exposure VAE-LHAT chain3 short polish. LHAT is "
            "used only inside three-chain AugMix and JSD, with no direct LHAT "
            "BCE, fixed20, PCGrad, VAE-random, raw auxiliary or source replay."
        )
    elif candidate.method_family == "vae_lhat_chain3_teacher04_polish":
        purpose = (
            "K500-only J4 VAE-LHAT chain3 short polish. Only AugMix BCE uses "
            "the 0.40 frozen Stage-2 initializer teacher target; non-AugMix "
            "supervision stays hard-label trained."
        )
    elif candidate.method_family == "vae_lhat_bounded_augmax_polish":
        purpose = (
            "K500-only bounded AugMax-M20 continuation with online depth2/3 "
            "coverage and one shared exact-label VAE-LHAT third chain. Six "
            "frozen convex mixtures are searched under a bounded BCE-gain cap."
        )
    else:
        purpose = (
            "K500-only A7 VAE-LHAT plus AugMix global heldout-tuned "
            "development run."
        )
    if candidate.pretrain_view_mode is not None and (
        candidate.pretrain_source_replay_weight
    ):
        purpose = (
            f"{purpose} Stage-1 additionally consumes labeled PTB-XL train "
            "semantic replay at weight "
            f"{candidate.pretrain_source_replay_weight:g} with "
            f"{candidate.pretrain_source_batches_per_step} source batch per "
            "target step; no target-center records outside K500 are consumed."
        )
    if candidate.supervised_source_replay_weight is not None:
        purpose = (
            f"{purpose} Stage-2 candidate and Direct arms additionally use "
            "the same deterministic labeled PTB-XL replay stream at total "
            f"loss weight {candidate.supervised_source_replay_weight:g}; "
            "source replay is a stability control, not a novelty claim."
        )
    train["experiment"]["purpose"] = purpose
    train["entrypoint"]["config"] = str(relroot / "online" / f"{center}.yaml")
    arguments = train["entrypoint"]["arguments"]
    _replace_argument(arguments, "--model", candidate.model_name)
    _replace_argument(
        arguments,
        "--method-config",
        relroot / "methods" / f"{center}.yaml",
    )
    _replace_argument(arguments, "--center", center)
    _replace_argument(
        arguments,
        "--source-checkpoint",
        _source_checkpoint(candidate, center),
    )
    if candidate.method_family in {
        "clean_target_ssl_control",
        "direct_target_ssl_control",
        "direct_budget_filler",
    }:
        _remove_argument(arguments, "--vae-checkpoint")
    else:
        _replace_argument(arguments, "--vae-checkpoint", VAE_CHECKPOINT)
    _replace_argument(arguments, "--epochs", candidate.epochs)
    _replace_argument(
        arguments,
        "--scheduler-horizon-epochs",
        candidate.scheduler_horizon_epochs,
    )
    _replace_argument(arguments, "--learning-rate", candidate.learning_rate)
    if "--weight-decay" in [str(value) for value in arguments]:
        _replace_argument(arguments, "--weight-decay", candidate.weight_decay)
    else:
        arguments.extend(["--weight-decay", str(candidate.weight_decay)])
    _replace_argument(arguments, "--batch-size", candidate.batch_size)
    _replace_argument(arguments, "--num-workers", 0)
    train["output"]["run_dir"] = str(
        _persistent_root(candidate) / "candidate" / center / "train"
    )
    train["output"]["if_exists"] = "error"

    evaluate = _load_yaml(
        _base_a7_root(candidate) / "experiments" / f"{center}_eval.yaml"
    )
    evaluate["experiment"]["name"] = (
        f"a7_global_{candidate.candidate_id}_{center}_eval"
    )
    evaluate["experiment"]["purpose"] = (
        "Metric-only ref-excluded Clean and PN2021-C development evaluation."
    )
    eval_arguments = evaluate["entrypoint"]["arguments"]
    _replace_argument(eval_arguments, "--model", candidate.model_name)
    _replace_argument(eval_arguments, "--center", center)
    _replace_argument(
        eval_arguments,
        "--checkpoint",
        _persistent_root(candidate)
        / "candidate"
        / center
        / "train"
        / "training"
        / "checkpoints"
        / "last.pt",
    )
    _replace_argument(
        eval_arguments,
        "--batch-size",
        EVAL_BATCH_SIZES[candidate.model_name],
    )
    _replace_argument(eval_arguments, "--num-workers", 0)
    evaluate["output"]["run_dir"] = str(
        _persistent_root(candidate) / "candidate" / center / "eval"
    )
    evaluate["output"]["if_exists"] = "error"
    return train, evaluate


def _direct_experiments(
    candidate: Candidate,
    center: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = {
        "schema_version": 1,
        "experiment": {
            "name": f"a7_global_{candidate.candidate_id}_{center}_direct_train",
            "purpose": (
                "Optimizer-, budget-, source-, seed-, and K500-matched "
                + (
                    "full-LHAT-compute clean-chain3 bounded AugMax-M20 control."
                    if candidate.method_family
                    == "vae_lhat_bounded_augmax_polish"
                    else (
                    (
                        "clean-identity auxiliary control matched to LHAT-path "
                        "generation and hard-BCE weight."
                        if candidate.polish_auxiliary_target
                        == "latent_path_hard_label"
                        else (
                            "clean-identity auxiliary hard-BCE short-polish "
                            "control."
                            if candidate.polish_auxiliary_target == "hard_label"
                            else (
                                "clean-identity anchor-soft control with matched "
                                "local LHAT search and decode."
                                if candidate.polish_auxiliary_target
                                == "local_anchor_soft"
                                else (
                                    "clean-identity consistency short-polish control."
                                    if candidate.polish_auxiliary_target
                                    == "consistency"
                                    else (
                                        "clean-identity penultimate-feature "
                                        "invariance short-polish control."
                                        if candidate.polish_auxiliary_target
                                        == "feature_invariance"
                                        else "clean-only short-polish control."
                                    )
                                )
                            )
                        )
                    )
                    if candidate.method_family == "vae_lhat_clean_polish"
                    else (
                        "clean-identity chain3 short-polish control."
                        if candidate.method_family
                        in {
                            "vae_lhat_chain3_polish",
                            "vae_lhat_chain3_teacher04_polish",
                        }
                        else "Direct+fixed20 control."
                    )
                    )
                )
            ),
        },
        "entrypoint": {
            "name": "train_pn2021",
            "config": str(
                _candidate_relroot(candidate)
                / "direct_train"
                / f"{center}.yaml"
            ),
            "arguments": [
                "--model",
                candidate.model_name,
                "--method-config",
                str(
                    _candidate_relroot(candidate)
                    / "direct_methods"
                    / f"{center}.yaml"
                ),
                "--center",
                center,
                "--source-checkpoint",
                str(_source_checkpoint(candidate, center)),
                "--epochs",
                str(candidate.epochs),
                "--scheduler-horizon-epochs",
                str(candidate.scheduler_horizon_epochs),
                "--trainable-scope",
                "full",
                "--num-workers",
                "0",
                "--learning-rate",
                str(candidate.learning_rate),
                "--weight-decay",
                str(candidate.weight_decay),
                "--batch-size",
                str(candidate.batch_size),
            ],
        },
        "output": {
            "run_dir": str(
                _persistent_root(candidate) / "direct" / center / "train"
            ),
            "if_exists": "error",
            "delegate_output_subdir": "training",
        },
    }
    if (
        candidate.method_family
        in {
            "vae_lhat_chain3_teacher04_polish",
            "vae_lhat_bounded_augmax_polish",
        }
        or (
            candidate.method_family == "vae_lhat_clean_polish"
            and (
                candidate.frozen_teacher_mix is not None
                or candidate.polish_auxiliary_target
                in {
                    "hard_label",
                    "latent_path_hard_label",
                    "consistency",
                    "feature_invariance",
                    "local_anchor_soft",
                }
            )
        )
    ):
        train["entrypoint"]["arguments"].extend(
            ["--vae-checkpoint", str(VAE_CHECKPOINT)]
        )
    evaluate = {
        "schema_version": 1,
        "experiment": {
            "name": f"a7_global_{candidate.candidate_id}_{center}_direct_eval",
            "purpose": (
                "Ref-excluded Clean and PN2021-C evaluation of the matched "
                + (
                    "full-LHAT-compute clean-chain3 bounded AugMax-M20 control."
                    if candidate.method_family
                    == "vae_lhat_bounded_augmax_polish"
                    else (
                    (
                        "clean-identity auxiliary control matched to LHAT-path "
                        "generation and hard-BCE weight."
                        if candidate.polish_auxiliary_target
                        == "latent_path_hard_label"
                        else (
                            "clean-identity auxiliary hard-BCE short-polish "
                            "control."
                            if candidate.polish_auxiliary_target == "hard_label"
                            else (
                                "clean-identity anchor-soft control with matched "
                                "local LHAT search and decode."
                                if candidate.polish_auxiliary_target
                                == "local_anchor_soft"
                                else (
                                    "clean-identity consistency short-polish control."
                                    if candidate.polish_auxiliary_target
                                    == "consistency"
                                    else (
                                        "clean-identity penultimate-feature "
                                        "invariance short-polish control."
                                        if candidate.polish_auxiliary_target
                                        == "feature_invariance"
                                        else "clean-only short-polish control."
                                    )
                                )
                            )
                        )
                    )
                    if candidate.method_family == "vae_lhat_clean_polish"
                    else (
                        "clean-identity chain3 short-polish control."
                        if candidate.method_family
                        in {
                            "vae_lhat_chain3_polish",
                            "vae_lhat_chain3_teacher04_polish",
                        }
                        else "Direct+fixed20 control."
                    )
                    )
                )
            ),
        },
        "entrypoint": {
            "name": "evaluate_pn2021",
            "config": "eval/PN2021.yaml",
            "arguments": [
                "--model",
                candidate.model_name,
                "--checkpoint",
                str(
                    _persistent_root(candidate)
                    / "direct"
                    / center
                    / "train"
                    / "training"
                    / "checkpoints"
                    / "last.pt"
                ),
                "--center",
                center,
                "--batch-size",
                str(EVAL_BATCH_SIZES[candidate.model_name]),
                "--num-workers",
                "0",
            ],
        },
        "output": {
            "run_dir": str(
                _persistent_root(candidate) / "direct" / center / "eval"
            ),
            "if_exists": "error",
            "delegate_output_subdir": "evaluation",
        },
    }
    return train, evaluate


def _vae_checkpoint_manifest_for(candidate: Candidate) -> dict[str, str] | None:
    if candidate.method_family not in VAE_CHECKPOINT_MANIFEST_FAMILIES:
        return None
    return {
        "path": str(VAE_CHECKPOINT),
        "sha256": _sha256(VAE_CHECKPOINT),
    }


def prepare(candidate: Candidate, *, force: bool) -> Path:
    root = _candidate_root(candidate)
    if root.exists():
        if not force:
            raise FileExistsError(f"generated candidate already exists: {root}")
        shutil.rmtree(root)
    if (
        (
            candidate.method_family
            in {
                "vae_lhat_post_refine",
                "vae_lhat_clean_polish",
                "vae_lhat_chain3_polish",
                "vae_lhat_chain3_teacher04_polish",
                "vae_lhat_bounded_augmax_polish",
            }
            and (
                candidate.lhat_pgd_epsilon_l2_standardized != 2.0
                or candidate.polish_auxiliary_target == "local_anchor_soft"
                or candidate.lhat_geometry_preset is not None
            )
        )
        or candidate.lhat_search_mode != "optimized_softmax"
    ):
        _write_yaml(root / "lhat.yaml", _lhat_config_for(candidate))
    for center in CENTERS:
        if candidate.initializer_candidate is not None:
            _write_yaml(
                root / "baselines" / f"{center}.yaml",
                _source_registry_for(candidate, center),
            )
        _write_yaml(root / "methods" / f"{center}.yaml", _method_for(candidate, center))
        _write_yaml(
            root / "direct_methods" / f"{center}.yaml",
            _direct_method_for(candidate, center),
        )
        _write_yaml(root / "online" / f"{center}.yaml", _online_for(candidate, center))
        _write_yaml(
            root / "direct_train" / f"{center}.yaml",
            _direct_training_config_for(candidate, center),
        )
        candidate_train, candidate_eval = _candidate_experiments(
            candidate, center
        )
        direct_train, direct_eval = _direct_experiments(candidate, center)
        _write_yaml(
            root / "experiments" / "candidate" / f"{center}_train.yaml",
            candidate_train,
        )
        _write_yaml(
            root / "experiments" / "candidate" / f"{center}_eval.yaml",
            candidate_eval,
        )
        _write_yaml(
            root / "experiments" / "direct" / f"{center}_train.yaml",
            direct_train,
        )
        _write_yaml(
            root / "experiments" / "direct" / f"{center}_eval.yaml",
            direct_eval,
        )
    manifest = {
        "schema_version": 1,
        "candidate": candidate.__dict__,
        "status": "heldout_tuned_development_only",
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "per_center_or_per_class_tuning": False,
        "target_data_contract": {
            "adaptation_records": "per_center_fixed_k500_only",
            "outside_k500_model_access": False,
            "ptbxl_source_replay_weight": candidate.pretrain_source_replay_weight,
            "supervised_ptbxl_source_replay_weight": (
                candidate.supervised_source_replay_weight
            ),
            "source_checkpoint_allowed": True,
            "source_teacher_logits_on_k500_allowed": (
                candidate.frozen_teacher_mix is not None
            ),
        },
        "class_order": list(CLASS_ORDER),
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
        "source_checkpoint": (
            {
                "path": str(_source_checkpoint(candidate)),
                "sha256": _sha256(_source_checkpoint(candidate)),
            }
            if candidate.initializer_candidate is None
            else {
                "initializer_candidate": candidate.initializer_candidate,
                "per_center": {
                    center: {
                        "path": str(_source_checkpoint(candidate, center)),
                        "sha256": _sha256(_source_checkpoint(candidate, center)),
                    }
                    for center in CENTERS
                },
            }
        ),
        "source_baseline_registry": (
            {
                "path": str(LOCKED_SOURCE_REGISTRY),
                "sha256": _sha256(LOCKED_SOURCE_REGISTRY),
            }
            if candidate.initializer_candidate is None
            else {
                "parent_path": str(LOCKED_SOURCE_REGISTRY),
                "parent_sha256": _sha256(LOCKED_SOURCE_REGISTRY),
                "per_center": {
                    center: {
                        "path": str(root / "baselines" / f"{center}.yaml"),
                        "sha256": _sha256(root / "baselines" / f"{center}.yaml"),
                    }
                    for center in CENTERS
                },
            }
        ),
        "vae_checkpoint": _vae_checkpoint_manifest_for(candidate),
        "managed_launcher": str(LAUNCHER),
        "persistent_root": str(_persistent_root(candidate)),
        "temporary_log_root": str(TMP_ROOT / candidate.candidate_id),
        "resource_floors": {
            "available_ram_gib": MIN_AVAILABLE_RAM_GIB,
            "available_shm_gib": MIN_SHM_AVAILABLE_GIB,
            "available_nvme_gib": MIN_NVME_AVAILABLE_GIB,
            "available_persistent_gib": MIN_PERSISTENT_AVAILABLE_GIB,
            "free_gpu_memory_mib": MIN_FREE_GPU_MIB,
        },
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _available_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024**2)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def _gpu_rows() -> dict[int, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: dict[int, dict[str, int]] = {}
    for raw in result.stdout.splitlines():
        index, total, used, utilization = (
            int(value.strip()) for value in raw.split(",")
        )
        rows[index] = {
            "total_mib": total,
            "used_mib": used,
            "free_mib": total - used,
            "utilization": utilization,
        }
    return rows


def _resource_gate(
    gpus: Sequence[int],
    *,
    require_free_gpu: bool = True,
) -> dict[str, Any]:
    if not 1 <= len(gpus) <= len(CENTERS) or len(set(gpus)) != len(gpus):
        raise ValueError(
            f"select between one and {len(CENTERS)} unique GPUs"
        )
    rows = _gpu_rows()
    missing = [gpu for gpu in gpus if gpu not in rows]
    if missing:
        raise ValueError(f"unknown GPU indices: {missing}")
    if require_free_gpu:
        busy = {
            gpu: rows[gpu]
            for gpu in gpus
            if rows[gpu]["free_mib"] < MIN_FREE_GPU_MIB
        }
        if busy:
            raise RuntimeError(f"selected GPUs are not free enough: {busy}")
    available_ram = _available_ram_gib()
    available_shm = _available_gib(Path("/dev/shm"))
    available_nvme = _available_gib(Path("/home/linbinhao"))
    available_persistent = _available_gib(RUN_ROOT.parent)
    if available_ram < MIN_AVAILABLE_RAM_GIB:
        raise RuntimeError(
            f"available RAM {available_ram:.1f} GiB is below the floor"
        )
    if available_shm < MIN_SHM_AVAILABLE_GIB:
        raise RuntimeError(
            f"available /dev/shm {available_shm:.1f} GiB is below the floor"
        )
    if available_nvme < MIN_NVME_AVAILABLE_GIB:
        raise RuntimeError(
            f"available NVMe {available_nvme:.1f} GiB is below the floor"
        )
    if available_persistent < MIN_PERSISTENT_AVAILABLE_GIB:
        raise RuntimeError(
            "available persistent-run space "
            f"{available_persistent:.1f} GiB is below the floor"
        )
    return {
        "gpus": {str(gpu): rows[gpu] for gpu in gpus},
        "available_ram_gib": available_ram,
        "available_shm_gib": available_shm,
        "available_nvme_gib": available_nvme,
        "available_persistent_gib": available_persistent,
    }


def _is_complete(run_dir: Path, stage: str) -> bool:
    result_path = (
        run_dir / "training" / "train_result.json"
        if stage == "train"
        else run_dir / "evaluation" / "evaluation_result.json"
    )
    run_manifest = run_dir / "run_manifest.json"
    run_file_index = run_dir / "run_file_index.json"
    required = [result_path, run_manifest, run_file_index]
    checkpoint = run_dir / "training" / "checkpoints" / "last.pt"
    if stage == "train":
        required.append(checkpoint)
    if any(not path.is_file() for path in required):
        return False
    try:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("exit_code") != 0
            or manifest.get("error") is not None
            or manifest.get("run_file_index_sha256") != _sha256(run_file_index)
        ):
            return False
        index = json.loads(run_file_index.read_text(encoding="utf-8"))
        indexed = {
            record.get("path"): record
            for record in index.get("files", [])
            if isinstance(record, Mapping)
        }
        relative_result = result_path.relative_to(run_dir).as_posix()
        result_record = indexed.get(relative_result)
        if (
            not isinstance(result_record, Mapping)
            or result_record.get("sha256") != _sha256(result_path)
        ):
            return False
        if stage == "train":
            checkpoint_record = indexed.get(
                checkpoint.relative_to(run_dir).as_posix()
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            selected = result.get("selection", {}).get("selected_checkpoint", {})
            checkpoint_sha256 = _sha256(checkpoint)
            if (
                not isinstance(checkpoint_record, Mapping)
                or checkpoint_record.get("sha256") != checkpoint_sha256
                or selected.get("sha256") != checkpoint_sha256
            ):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _config_path(
    candidate: Candidate,
    arm: str,
    center: str,
    stage: str,
) -> Path:
    return (
        _candidate_root(candidate)
        / "experiments"
        / arm
        / f"{center}_{stage}.yaml"
    )


def _run_dir(candidate: Candidate, arm: str, center: str, stage: str) -> Path:
    return _persistent_root(candidate) / arm / center / stage


def run_wave(
    candidate: Candidate,
    *,
    arm: str,
    stage: str,
    gpus: Sequence[int],
    cpu_threads: int,
    jobs_per_gpu: int = 1,
) -> None:
    if arm not in {"candidate", "direct"}:
        raise ValueError("arm must be candidate or direct")
    if stage not in {"train", "eval"}:
        raise ValueError("stage must be train or eval")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if jobs_per_gpu <= 0:
        raise ValueError("jobs_per_gpu must be positive")
    if len(gpus) * jobs_per_gpu > len(CENTERS):
        raise ValueError(
            f"at most {len(CENTERS)} concurrent center jobs are allowed"
        )
    if not (_candidate_root(candidate) / "manifest.json").is_file():
        raise FileNotFoundError("prepare the candidate before launching")
    resource_snapshot = _resource_gate(gpus)
    log_root = TMP_ROOT / candidate.candidate_id / arm / stage
    log_root.mkdir(parents=True, exist_ok=True)
    _write_json(log_root / "resource_snapshot.json", resource_snapshot)
    pending: list[str] = []
    for center in CENTERS:
        run_dir = _run_dir(candidate, arm, center, stage)
        if _is_complete(run_dir, stage):
            print(f"[skip] {arm}/{stage}/{center} already complete", flush=True)
            continue
        if run_dir.exists():
            raise FileExistsError(
                f"incomplete run exists; inspect it instead of overwriting: {run_dir}"
            )
        pending.append(center)
    slots = tuple(
        (gpu, slot)
        for slot in range(jobs_per_gpu)
        for gpu in gpus
    )
    _write_json(
        log_root / "launch_plan.json",
        {
            "schema_version": 1,
            "candidate": candidate.candidate_id,
            "arm": arm,
            "stage": stage,
            "gpus": list(gpus),
            "jobs_per_gpu": jobs_per_gpu,
            "cpu_threads_per_job": cpu_threads,
            "planned_assignment": {
                center: {"gpu": gpu, "slot": slot}
                for center, (gpu, slot) in zip(pending, slots, strict=False)
            },
        },
    )
    running: dict[
        tuple[int, int],
        tuple[str, subprocess.Popen[str], Any, Path],
    ] = {}
    failures: list[str] = []
    last_report = 0.0
    while pending or running:
        for gpu, slot in slots:
            slot_key = (gpu, slot)
            if slot_key in running or not pending:
                continue
            center = pending.pop(0)
            config = _config_path(candidate, arm, center, stage)
            log_path = log_root / f"{center}.log"
            handle = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "ECG_DIVERSE_AUGMAX_CPU_THREADS": str(cpu_threads),
                    "OMP_NUM_THREADS": str(cpu_threads),
                    "MKL_NUM_THREADS": str(cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(cpu_threads),
                    "NUMEXPR_NUM_THREADS": str(cpu_threads),
                    "TMPDIR": str(TMP_ROOT / candidate.candidate_id / "tmp"),
                }
            )
            Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON),
                "-u",
                str(LAUNCHER),
                "--config",
                str(config),
                "--config-root",
                str(CONFIG_ROOT),
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[slot_key] = (center, process, handle, log_path)
            print(
                f"[launch] {arm}/{stage}/{center} gpu={gpu} slot={slot} "
                f"pid={process.pid} log={log_path}",
                flush=True,
            )
        for slot_key, (center, process, handle, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[slot_key]
            if return_code != 0:
                failures.append(f"{center}: exit={return_code}, log={log_path}")
            print(
                f"[done] {arm}/{stage}/{center} exit={return_code}",
                flush=True,
            )
        now = time.monotonic()
        if running and now - last_report >= 30.0:
            resource_snapshot = _resource_gate(
                gpus,
                require_free_gpu=False,
            )
            active = ", ".join(
                f"{center}:{process.pid}@gpu{gpu}/slot{slot}"
                for (gpu, slot), (center, process, _, _) in running.items()
            )
            print(
                f"[progress] active={active} "
                f"ram={resource_snapshot['available_ram_gib']:.1f}GiB "
                f"nvme={resource_snapshot['available_nvme_gib']:.1f}GiB "
                "persistent="
                f"{resource_snapshot['available_persistent_gib']:.1f}GiB "
                f"shm={resource_snapshot['available_shm_gib']:.1f}GiB",
                flush=True,
            )
            last_report = now
        if pending or running:
            time.sleep(2.0)
    if failures:
        raise RuntimeError("; ".join(failures))


def _source_floor_output(
    candidate: Candidate,
    arm: str,
    center: str,
) -> Path:
    if arm not in {"candidate", "direct"}:
        raise ValueError("arm must be candidate or direct")
    if center not in CENTERS:
        raise ValueError(f"unknown center: {center}")
    return _persistent_root(candidate) / "source_floor" / arm / f"{center}.json"


def evaluate_source_floor_one(
    candidate: Candidate,
    *,
    arm: str,
    center: str,
) -> Path:
    """Evaluate one frozen target-adapted checkpoint on PTB-XL fold10."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import torch

    from core.supervised_trainer import _run_epoch
    from core.train_PTBXL import build_ptbxl_dataloaders
    from models import build_model
    from models.checkpoints import load_model_checkpoint, sha256_file

    train_run = _run_dir(candidate, arm, center, "train")
    if not _is_complete(train_run, "train"):
        raise RuntimeError(
            f"source-floor checkpoint is not a complete managed run: {train_run}"
        )
    checkpoint = train_run / "training" / "checkpoints" / "last.pt"
    train_result_path = train_run / "training" / "train_result.json"
    train_result = json.loads(train_result_path.read_text(encoding="utf-8"))
    if int(train_result.get("epochs_completed", -1)) != candidate.epochs:
        raise ValueError(
            f"epoch mismatch for {candidate.candidate_id}/{arm}/{center}: "
            f"{train_result.get('epochs_completed')} != {candidate.epochs}"
        )

    model = build_model(
        candidate.model_name,
        config_root=PROJECT_ROOT / "configs",
        seed_namespace="a7_global_search_source_floor",
        seed_identity=(candidate.candidate_id, arm, center),
    )
    checkpoint_identity = load_model_checkpoint(
        model,
        checkpoint,
        map_location="cpu",
        strict=True,
    )
    loaders = build_ptbxl_dataloaders(
        model,
        config_path=SOURCE_TRAIN_CONFIG,
        config_root=PROJECT_ROOT / "configs",
        dataloader_parameters={
            "train_batch_size": 128,
            "eval_batch_size": EVAL_BATCH_SIZES[candidate.model_name],
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "cache_mode": "mmap",
            "validate_values": "sample",
            "drop_last": False,
        },
    )
    try:
        if loaders.test is None:
            raise RuntimeError("PTB-XL fold10 loader was not constructed")
        device = torch.device("cuda")
        model.to(device)
        metrics = _run_epoch(
            model,
            loaders.test,
            device=device,
            model_spec=model.model_spec,
            input_key="waveform",
            target_key="label",
            class_names=CLASS_ORDER,
            pos_weight=None,
            amp_enabled=True,
            amp_dtype=torch.bfloat16,
            undefined_class_policy="strict",
        )
        loader_identity = loaders.test.describe()
    finally:
        loaders.close()

    baseline = _load_yaml(LOCKED_SOURCE_REGISTRY)["models"][
        candidate.model_name
    ]["fold10"]
    payload = {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "arm": arm,
        "center": center,
        "model_name": candidate.model_name,
        "selected_epoch": candidate.epochs,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "load_identity": checkpoint_identity.describe(),
        },
        "training_result": {
            "path": str(train_result_path),
            "sha256": sha256_file(train_result_path),
            "optimizer_steps": train_result.get("optimizer_steps"),
        },
        "partition": "ptbxl_fold10",
        "selection_use": "source_floor_only",
        "metrics": metrics,
        "locked_source_baseline": baseline,
        "delta": {
            "macro_auroc": (
                float(metrics["macro_auroc"])
                - float(baseline["macro_auroc"])
            ),
            "macro_auprc": (
                float(metrics["macro_auprc"])
                - float(baseline["macro_auprc"])
            ),
        },
        "loader": loader_identity,
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = _source_floor_output(candidate, arm, center)
    _write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return output


def summarize_source_floor(candidate: Candidate, *, arm: str) -> Path:
    records = {
        center: json.loads(
            _source_floor_output(candidate, arm, center).read_text(
                encoding="utf-8"
            )
        )
        for center in CENTERS
    }
    metric_names = ("macro_auroc", "macro_auprc")
    means = {
        name: sum(
            float(record["metrics"][name]) for record in records.values()
        )
        / len(records)
        for name in metric_names
    }
    baseline = records[CENTERS[0]]["locked_source_baseline"]
    if any(
        record["locked_source_baseline"] != baseline
        for record in records.values()
    ):
        raise ValueError("source-floor baseline identity drifted across centers")
    deltas = {
        name: means[name] - float(baseline[name]) for name in metric_names
    }
    individual_auprc_deltas = {
        center: float(record["delta"]["macro_auprc"])
        for center, record in records.items()
    }
    absolute_rule = {
        "mean_macro_auroc_delta_at_least_minus_0p01": (
            deltas["macro_auroc"] >= -0.01
        ),
        "mean_macro_auprc_delta_at_least_minus_0p01": (
            deltas["macro_auprc"] >= -0.01
        ),
        "all_individual_macro_auprc_deltas_at_least_minus_0p015": all(
            delta >= -0.015 for delta in individual_auprc_deltas.values()
        ),
    }
    absolute_rule["passed"] = all(absolute_rule.values())
    summary = {
        "schema_version": 1,
        "candidate": candidate.__dict__,
        "arm": arm,
        "partition": "ptbxl_fold10",
        "selection_use": "source_floor_only",
        "four_center_mean": means,
        "locked_source_baseline": baseline,
        "mean_delta": deltas,
        "individual_macro_auprc_delta": individual_auprc_deltas,
        "predeclared_absolute_retention_rule": absolute_rule,
        "records": {
            center: {
                "path": str(_source_floor_output(candidate, arm, center)),
                "checkpoint_sha256": record["checkpoint"]["sha256"],
                "metrics": record["metrics"],
                "delta": record["delta"],
            }
            for center, record in records.items()
        },
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = (
        _persistent_root(candidate)
        / "source_floor"
        / arm
        / "summary.json"
    )
    _write_json(output, summary)
    return output


def run_source_floor(
    candidate: Candidate,
    *,
    arm: str,
    gpus: Sequence[int],
    cpu_threads: int,
) -> Path:
    if arm not in {"candidate", "direct"}:
        raise ValueError("arm must be candidate or direct")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if len(gpus) > len(CENTERS):
        raise ValueError(f"at most {len(CENTERS)} GPUs are allowed")
    resource_snapshot = _resource_gate(gpus)
    log_root = TMP_ROOT / candidate.candidate_id / "source_floor" / arm
    log_root.mkdir(parents=True, exist_ok=True)
    _write_json(log_root / "resource_snapshot.json", resource_snapshot)

    pending = [
        center
        for center in CENTERS
        if not _source_floor_output(candidate, arm, center).is_file()
    ]
    running: dict[int, tuple[str, subprocess.Popen[str], Any, Path]] = {}
    failures: list[str] = []
    last_report = 0.0
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            center = pending.pop(0)
            log_path = log_root / f"{center}.log"
            handle = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": str(cpu_threads),
                    "MKL_NUM_THREADS": str(cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(cpu_threads),
                    "NUMEXPR_NUM_THREADS": str(cpu_threads),
                }
            )
            command = [
                str(PYTHON),
                "-u",
                str(SANDBOX_ROOT / "a7_global_search.py"),
                "source-floor-worker",
                "--candidate",
                candidate.candidate_id,
                "--arm",
                arm,
                "--center",
                center,
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = (center, process, handle, log_path)
            print(
                f"[launch] source-floor/{arm}/{center} gpu={gpu} "
                f"pid={process.pid} log={log_path}",
                flush=True,
            )
        for gpu, (center, process, handle, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[gpu]
            if return_code != 0:
                failures.append(f"{center}: exit={return_code}, log={log_path}")
            print(
                f"[done] source-floor/{arm}/{center} exit={return_code}",
                flush=True,
            )
        now = time.monotonic()
        if running and now - last_report >= 30.0:
            snapshot = _resource_gate(gpus, require_free_gpu=False)
            print(
                f"[progress] source-floor/{arm} active={len(running)} "
                f"ram={snapshot['available_ram_gib']:.1f}GiB "
                f"nvme={snapshot['available_nvme_gib']:.1f}GiB "
                f"persistent={snapshot['available_persistent_gib']:.1f}GiB",
                flush=True,
            )
            last_report = now
        if pending or running:
            time.sleep(2.0)
    if failures:
        raise RuntimeError("; ".join(failures))
    return summarize_source_floor(candidate, arm=arm)


def _primary_metrics(path: Path) -> tuple[float, float, float, float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    clean = result["clean"]["four_center_mean"][
        "pn2021_all_zero_kept_refexcluded"
    ]
    corrupted = result["corrupted"]["aggregates"]["depth23"][
        "four_center_mean"
    ]["pn2021c_all_zero_kept_corrupted_refexcluded"]
    return (
        float(clean["macro_auroc"]),
        float(clean["macro_auprc"]),
        float(corrupted["macro_auroc"]),
        float(corrupted["macro_auprc"]),
    )


def _mean_rows(rows: Iterable[Sequence[float]]) -> list[float]:
    values = [tuple(float(v) for v in row) for row in rows]
    if not values:
        raise ValueError("cannot aggregate an empty metric list")
    return [
        sum(row[index] for row in values) / len(values)
        for index in range(len(values[0]))
    ]


def _assert_matched_direct(
    candidate: Candidate,
    direct_candidate: Candidate,
) -> None:
    fields = (
        "model_name",
        "learning_rate",
        "batch_size",
        "epochs",
        "scheduler_horizon_epochs",
        "weight_decay",
    )
    mismatched = {
        field: (getattr(candidate, field), getattr(direct_candidate, field))
        for field in fields
        if getattr(candidate, field) != getattr(direct_candidate, field)
    }
    if mismatched:
        raise ValueError(f"Direct reference is not optimizer matched: {mismatched}")


def summarize(
    candidate: Candidate,
    *,
    direct_candidate: Candidate | None = None,
    direct_arm: str = "direct",
) -> Path:
    if direct_arm not in {"direct", "candidate"}:
        raise ValueError("direct_arm must be direct or candidate")
    direct_candidate = candidate if direct_candidate is None else direct_candidate
    _assert_matched_direct(candidate, direct_candidate)
    per_arm: dict[str, dict[str, list[float]]] = {}
    for arm in ("direct", "candidate"):
        per_center: dict[str, list[float]] = {}
        for center in CENTERS:
            owner = direct_candidate if arm == "direct" else candidate
            owner_arm = direct_arm if arm == "direct" else "candidate"
            path = (
                _run_dir(owner, owner_arm, center, "eval")
                / "evaluation"
                / "evaluation_result.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing evaluation: {path}")
            per_center[center] = list(_primary_metrics(path))
        per_arm[arm] = per_center
    direct_mean = _mean_rows(per_arm["direct"].values())
    candidate_mean = _mean_rows(per_arm["candidate"].values())
    delta_pp = [
        100.0 * (candidate_mean[index] - direct_mean[index])
        for index in range(4)
    ]
    center_delta_pp = {
        center: [
            100.0
            * (
                per_arm["candidate"][center][index]
                - per_arm["direct"][center][index]
            )
            for index in range(4)
        ]
        for center in CENTERS
    }
    stage2_pair_evidence: dict[str, Any] | None = None
    if (
        direct_arm == "direct"
        and candidate.candidate_id == direct_candidate.candidate_id
        and candidate.method_family
        in {
            "vae_lhat_post_refine",
            "vae_lhat_clean_polish",
            "vae_lhat_bounded_augmax_polish",
        }
    ):
        stage2_pair_evidence = {
            center: _assert_stage2_optimizer_pair(candidate, center)
            for center in CENTERS
        }
    elif (
        direct_arm == "direct"
        and candidate.candidate_id == direct_candidate.candidate_id
        and candidate.method_family == "d19_separate_views"
    ):
        stage2_pair_evidence = {
            center: _assert_fixed20_outer_pair(
                candidate,
                "direct",
                candidate,
                "candidate",
                center,
                label="D19 Direct versus candidate",
            )
            for center in CENTERS
        }
    summary = {
        "schema_version": 1,
        "candidate": candidate.__dict__,
        "direct_reference_candidate": direct_candidate.candidate_id,
        "direct_reference_arm": direct_arm,
        "status": "heldout_tuned_development_only",
        "metric_order": [
            "clean_macro_auroc",
            "clean_macro_auprc",
            "pn2021c_macro_auroc",
            "pn2021c_macro_auprc",
        ],
        "direct_four_center_mean": direct_mean,
        "candidate_four_center_mean": candidate_mean,
        "candidate_minus_direct_pp": delta_pp,
        "per_center_metrics": per_arm,
        "per_center_candidate_minus_direct_pp": center_delta_pp,
        "stage2_pair_evidence": stage2_pair_evidence,
        "selection_uses_per_center_or_per_class_weights": False,
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = _persistent_root(candidate) / "summary.json"
    _write_json(output, summary)
    return output


def _sample_std(rows: Sequence[Sequence[float]]) -> list[float]:
    if len(rows) < 2:
        return [0.0 for _ in rows[0]]
    means = _mean_rows(rows)
    return [
        math.sqrt(
            sum(
                (float(row[index]) - means[index]) ** 2
                for row in rows
            )
            / (len(rows) - 1)
        )
        for index in range(len(means))
    ]


def _replicate_seed_evidence(
    candidate: Candidate,
    direct_candidate: Candidate,
    center: str,
    *,
    allow_legacy_source_loader_gap: bool = False,
) -> dict[str, Any]:
    candidate_online = _load_yaml(
        _candidate_root(candidate) / "online" / f"{center}.yaml"
    )
    direct_online = _load_yaml(
        _candidate_root(direct_candidate)
        / "direct_train"
        / f"{center}.yaml"
    )
    candidate_seed = candidate_online.get("random_seed", {})
    direct_seed = direct_online.get("random_seed", {})
    if (
        candidate_seed != direct_seed
        or candidate_seed.get("replicate_id") != candidate.replicate_id
    ):
        raise ValueError(
            f"replicate seed identity drifted for {candidate.candidate_id}/{center}"
        )
    direct_stage1 = _stage1_replay_evidence(
        direct_candidate,
        center,
        arm="direct",
        allow_legacy_source_loader_gap=allow_legacy_source_loader_gap,
    )
    candidate_stage1 = _stage1_replay_evidence(
        candidate,
        center,
        arm="candidate",
        allow_legacy_source_loader_gap=allow_legacy_source_loader_gap,
    )
    for key in ("k500_selection", "target_order_sha256", "source_replay"):
        if direct_stage1[key] != candidate_stage1[key]:
            raise ValueError(
                f"replicate Stage-1 candidate/Direct drifted for "
                f"{candidate.candidate_id}/{center}/{key}"
            )
    return {
        "random_seed": candidate_seed,
        "stage1": {
            "k500_selection": candidate_stage1["k500_selection"],
            "target_order_sha256": candidate_stage1["target_order_sha256"],
            "source_replay": candidate_stage1["source_replay"],
        },
        "stage2": _assert_fixed20_outer_pair(
            direct_candidate,
            "direct",
            candidate,
            "candidate",
            center,
            label=f"replicate-{candidate.replicate_id} Direct versus D19",
        ),
    }


def summarize_replicates(
    reference: Candidate,
    replicate: Candidate,
    *,
    benchmark: Candidate,
    reference_direct: Candidate | None = None,
    replicate_direct: Candidate | None = None,
    additional_pairs: Sequence[tuple[Candidate, Candidate]] = (),
) -> Path:
    reference_direct = (
        reference if reference_direct is None else reference_direct
    )
    replicate_direct = (
        replicate if replicate_direct is None else replicate_direct
    )
    supported_method_families = {
        "d19_separate_views",
        "strong_lhat_augmix_chain3",
    }
    if (
        reference.method_family != replicate.method_family
        or reference.method_family not in supported_method_families
    ):
        raise ValueError(
            "replicate summary requires one frozen supported method family"
        )
    replicate_pairs = [
        (reference, reference_direct),
        (replicate, replicate_direct),
        *additional_pairs,
    ]
    replicate_ids = [candidate.replicate_id for candidate, _ in replicate_pairs]
    if reference.replicate_id != 0 or len(set(replicate_ids)) != len(
        replicate_ids
    ):
        raise ValueError(
            "replicate summary requires reference id 0 and unique replicate ids"
        )
    frozen_fields = tuple(
        field
        for field in Candidate.__dataclass_fields__
        if field not in {"candidate_id", "replicate_id"}
    )
    for candidate, _ in replicate_pairs[1:]:
        drift = {
            field: (getattr(reference, field), getattr(candidate, field))
            for field in frozen_fields
            if getattr(reference, field) != getattr(candidate, field)
        }
        if drift:
            raise ValueError(f"replicate recipe drifted: {drift}")
        if _sha256(_source_checkpoint(reference)) != _sha256(
            _source_checkpoint(candidate)
        ):
            raise ValueError("replicates use different source checkpoints")
    for candidate, direct_candidate in replicate_pairs:
        _assert_matched_direct(candidate, direct_candidate)
        if _sha256(_source_checkpoint(candidate)) != _sha256(
            _source_checkpoint(direct_candidate)
        ):
            raise ValueError(
                f"{candidate.candidate_id} and its Direct reference use "
                "different source checkpoints"
            )

    seeds = sorted(
        replicate_pairs,
        key=lambda item: item[0].replicate_id,
    )
    benchmark_mean = _mean_rows(
        _arm_metrics(benchmark, "candidate").values()
    )
    seed_records: dict[str, Any] = {}
    candidate_seed_means: list[list[float]] = []
    direct_seed_means: list[list[float]] = []
    evidence: dict[str, Any] = {}
    for candidate, direct_candidate in seeds:
        candidate_rows = _arm_metrics(candidate, "candidate")
        direct_rows = _arm_metrics(direct_candidate, "direct")
        candidate_mean = _mean_rows(candidate_rows.values())
        direct_mean = _mean_rows(direct_rows.values())
        delta_pp = _difference_pp(candidate_mean, direct_mean)
        per_center_delta = {
            center: _difference_pp(
                candidate_rows[center],
                direct_rows[center],
            )
            for center in CENTERS
        }
        if reference.method_family == "d19_separate_views":
            gate = {
                "clean_auprc_gain_at_least_0p25pp": delta_pp[1] >= 0.25,
                "pn2021c_auprc_gain_at_least_0p50pp": delta_pp[3] >= 0.50,
                "all_centers_nonnegative_pn2021c_auprc": all(
                    row[3] >= -1.0e-12 for row in per_center_delta.values()
                ),
                "exceeds_p22_clean_auprc": (
                    candidate_mean[1] > benchmark_mean[1]
                ),
                "exceeds_p22_pn2021c_auprc": (
                    candidate_mean[3] > benchmark_mean[3]
                ),
            }
        else:
            gate = {
                "positive_clean_auprc_gain": delta_pp[1] > 0.0,
                "positive_pn2021c_auprc_gain": delta_pp[3] > 0.0,
                "at_least_three_centers_nonnegative_pn2021c_auprc": (
                    sum(
                        row[3] >= -1.0e-12
                        for row in per_center_delta.values()
                    )
                    >= 3
                ),
            }
        gate["passed"] = all(gate.values())
        seed_key = f"replicate_{candidate.replicate_id}"
        seed_records[seed_key] = {
            "candidate_id": candidate.candidate_id,
            "direct_reference_candidate": direct_candidate.candidate_id,
            "direct_four_center_mean": direct_mean,
            "candidate_four_center_mean": candidate_mean,
            "candidate_minus_direct_pp": delta_pp,
            "per_center_candidate_minus_direct_pp": per_center_delta,
            "predeclared_gate": gate,
        }
        candidate_seed_means.append(candidate_mean)
        direct_seed_means.append(direct_mean)
        evidence[seed_key] = {
            center: _replicate_seed_evidence(
                candidate,
                direct_candidate,
                center,
                allow_legacy_source_loader_gap=(
                    candidate.replicate_id == 0
                ),
            )
            for center in CENTERS
        }

    for center in CENTERS:
        stage1_rows = [
            evidence[f"replicate_{candidate.replicate_id}"][center]["stage1"]
            for candidate, _ in seeds
        ]
        if any(
            stage1_rows[0]["k500_selection"] != row["k500_selection"]
            for row in stage1_rows[1:]
        ):
            raise ValueError(
                f"replicate K500 identities differ for {center}"
            )
        target_hashes = {
            row["target_order_sha256"] for row in stage1_rows
        }
        source_hashes = {
            row["source_replay"]["ordered_batch_hashes_sha256"]
            for row in stage1_rows
        }
        if (
            len(target_hashes) != len(stage1_rows)
            or len(source_hashes) != len(stage1_rows)
        ):
            raise ValueError(
                f"replicate identities are not independent for {center}"
            )

    candidate_mean = _mean_rows(candidate_seed_means)
    direct_mean = _mean_rows(direct_seed_means)
    all_seeds_passed = all(
        record["predeclared_gate"]["passed"]
        for record in seed_records.values()
    )
    summary = {
        "schema_version": 1,
        "status": (
            (
                f"frozen_target_specialist_{len(seeds)}_seed"
                if reference.method_family == "d19_separate_views"
                else f"{len(seeds)}_seed_lhat_target_gate_passed"
            )
            if all_seeds_passed
            else "replication_gate_failed"
        ),
        "claim_boundary": (
            "per-center K500-only target-specialist adaptation; "
            "no source-preservation claim"
        ),
        "metric_order": [
            "clean_macro_auroc",
            "clean_macro_auprc",
            "pn2021c_macro_auroc",
            "pn2021c_macro_auprc",
        ],
        "replicates": seed_records,
        "replicate_count": len(seeds),
        "replicate_candidate_mean": candidate_mean,
        "replicate_candidate_sample_std": _sample_std(candidate_seed_means),
        "replicate_direct_mean": direct_mean,
        "replicate_direct_sample_std": _sample_std(direct_seed_means),
        "replicate_candidate_minus_direct_pp": _difference_pp(
            candidate_mean,
            direct_mean,
        ),
        "benchmark": {
            "candidate_id": benchmark.candidate_id,
            "four_center_mean": benchmark_mean,
            "replicate_candidate_minus_benchmark_pp": _difference_pp(
                candidate_mean,
                benchmark_mean,
            ),
        },
        "predeclared_gate": {
            "all_replicates_passed_independently": all_seeds_passed,
            "passed": all_seeds_passed,
        },
        "identity_evidence": evidence,
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
        "source_floor": {
            "claim": "not_source_preserving",
            "reference_source_floor": str(
                _persistent_root(reference)
                / "source_floor"
                / "candidate"
                / "summary.json"
            ),
            "reported_as_limitation": True,
        },
    }
    output = _persistent_root(seeds[-1][0]) / "replication_summary.json"
    _write_json(output, summary)
    return output


def _arm_metrics(candidate: Candidate, arm: str) -> dict[str, list[float]]:
    if arm not in {"candidate", "direct"}:
        raise ValueError(f"unsupported arm: {arm}")
    metrics: dict[str, list[float]] = {}
    for center in CENTERS:
        path = (
            _run_dir(candidate, arm, center, "eval")
            / "evaluation"
            / "evaluation_result.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation: {path}")
        metrics[center] = list(_primary_metrics(path))
    return metrics


def _difference_pp(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        100.0 * (float(left[index]) - float(right[index]))
        for index in range(len(left))
    ]


def _training_history(candidate: Candidate, arm: str, center: str) -> dict[str, Any]:
    path = (
        _run_dir(candidate, arm, center, "train")
        / "training"
        / "training_history.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing training history: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, dict) or not isinstance(history.get("epochs"), list):
        raise ValueError(f"invalid training history: {path}")
    return history


def _assert_epoch_count(
    history: Mapping[str, Any],
    candidate: Candidate,
    *,
    label: str,
) -> list[Mapping[str, Any]]:
    rows = history.get("epochs")
    expected_steps_per_epoch = math.ceil(500 / candidate.batch_size)
    expected_optimizer_steps = candidate.epochs * expected_steps_per_epoch
    if (
        history.get("schema_version") != 2
        or history.get("selection") != "last"
        or history.get("optimizer_steps") != expected_optimizer_steps
    ):
        raise ValueError(f"{label} training-history budget/schema drifted")
    if not isinstance(rows, list) or len(rows) != candidate.epochs:
        raise ValueError(f"{label} did not complete its frozen epoch budget")
    if [row.get("epoch") for row in rows if isinstance(row, Mapping)] != list(
        range(1, candidate.epochs + 1)
    ):
        raise ValueError(f"{label} training-history epochs are not contiguous")
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_stage1_supervised_pair(
    identity_stage1: Candidate,
    augmix_stage1: Candidate,
    center: str,
) -> dict[str, Any]:
    identity = _training_history(identity_stage1, "candidate", center)
    augmix = _training_history(augmix_stage1, "candidate", center)
    identity_rows = _assert_epoch_count(
        identity, identity_stage1, label=f"{center}/identity Stage-1"
    )
    augmix_rows = _assert_epoch_count(
        augmix, augmix_stage1, label=f"{center}/AugMix Stage-1"
    )
    if identity.get("optimizer_steps") != augmix.get("optimizer_steps"):
        raise ValueError(f"Stage-1 optimizer-step budget differs for {center}")
    expected_steps_per_epoch = math.ceil(500 / identity_stage1.batch_size)
    epoch_evidence: list[dict[str, Any]] = []
    for identity_row, augmix_row in zip(identity_rows, augmix_rows, strict=True):
        if identity_row.get("learning_rate") != augmix_row.get("learning_rate"):
            raise ValueError(f"Stage-1 LR schedule differs for {center}")
        if identity_row.get("exposure") != augmix_row.get("exposure"):
            raise ValueError(f"Stage-1 supervised exposure differs for {center}")
        exposure = identity_row.get("exposure")
        if (
            not isinstance(exposure, Mapping)
            or exposure.get("base_record_count") != 500
            or exposure.get("clean_count") != 500
            or exposure.get("corrupted_count") != 10_000
            or exposure.get("total_count") != 10_500
            or exposure.get("view_executions_per_base_batch") != 21
            or exposure.get("optimizer_steps_per_base_batch") != 1
            or exposure.get("optimizer_steps_this_epoch")
            != expected_steps_per_epoch
        ):
            raise ValueError(f"Stage-1 supervised exposure budget drifted for {center}")
        identity_trace = identity_row.get("stochastic_trace", {})
        augmix_trace = augmix_row.get("stochastic_trace", {})
        if (
            not _is_sha256(identity_trace.get("input_identity_sha256"))
            or not _is_sha256(augmix_trace.get("input_identity_sha256"))
            or identity_trace.get("input_record_count") != 10_500
            or augmix_trace.get("input_record_count") != 10_500
            or
            identity_trace.get("input_identity_sha256")
            != augmix_trace.get("input_identity_sha256")
            or identity_trace.get("input_record_count")
            != augmix_trace.get("input_record_count")
        ):
            raise ValueError(
                f"Stage-1 supervised record order differs for {center}"
            )
        epoch_evidence.append(
            {
                "epoch": identity_row.get("epoch"),
                "input_identity_sha256": identity_trace.get(
                    "input_identity_sha256"
                ),
                "input_record_count": identity_trace.get("input_record_count"),
                "optimizer_steps_this_epoch": identity_row.get("exposure", {}).get(
                    "optimizer_steps_this_epoch"
                ),
            }
        )
    return {
        "optimizer_steps": identity.get("optimizer_steps"),
        "epochs": epoch_evidence,
    }


def _assert_fixed20_outer_pair(
    left: Candidate,
    left_arm: str,
    right: Candidate,
    right_arm: str,
    center: str,
    *,
    label: str,
) -> dict[str, Any]:
    """Prove equal outer updates while allowing LHAT-only trace additions."""

    left_history = _training_history(left, left_arm, center)
    right_history = _training_history(right, right_arm, center)
    left_rows = _assert_epoch_count(
        left_history,
        left,
        label=f"{center}/{label}/left",
    )
    right_rows = _assert_epoch_count(
        right_history,
        right,
        label=f"{center}/{label}/right",
    )
    if left_history.get("optimizer_steps") != right_history.get(
        "optimizer_steps"
    ):
        raise ValueError(f"{label} optimizer-step budget differs for {center}")
    fixed20_trace_names = (
        "depth23_corruption/composition_index",
        "depth23_corruption/depth",
        "depth23_corruption/operator_mask",
    )
    epoch_evidence: list[dict[str, Any]] = []
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        if left_row.get("learning_rate") != right_row.get("learning_rate"):
            raise ValueError(f"{label} LR schedule differs for {center}")
        if left_row.get("exposure") != right_row.get("exposure"):
            raise ValueError(f"{label} fixed20 exposure differs for {center}")
        left_trace = left_row.get("stochastic_trace", {})
        right_trace = right_row.get("stochastic_trace", {})
        if (
            left_trace.get("input_record_count") != 10_500
            or right_trace.get("input_record_count") != 10_500
        ):
            raise ValueError(f"{label} input record count drifted for {center}")
        left_tensors = left_trace.get("tensors", {})
        right_tensors = right_trace.get("tensors", {})
        if not isinstance(left_tensors, Mapping) or not isinstance(
            right_tensors, Mapping
        ):
            raise ValueError(f"{label} stochastic trace is missing for {center}")
        fixed20_trace = {}
        for name in fixed20_trace_names:
            if (
                name not in left_tensors
                or name not in right_tensors
                or left_tensors[name] != right_tensors[name]
                or not _is_sha256(left_tensors[name].get("sha256"))
            ):
                raise ValueError(
                    f"{label} fixed20 RNG trace differs for {center}/{name}"
                )
            fixed20_trace[name] = left_tensors[name]
        epoch_evidence.append(
            {
                "epoch": left_row.get("epoch"),
                "learning_rate": left_row.get("learning_rate"),
                "fixed20_trace": fixed20_trace,
            }
        )
    return {
        "optimizer_steps": left_history.get("optimizer_steps"),
        "epochs": epoch_evidence,
    }


def _stage1_replay_evidence(
    candidate: Candidate,
    center: str,
    *,
    arm: str = "candidate",
    allow_legacy_source_loader_gap: bool = False,
) -> dict[str, Any]:
    if candidate.method_family not in {
        "direct_target_ssl_control",
        "strong_lhat_augmix_chain3",
        "d19_separate_views",
    }:
        raise ValueError(
            "Stage-1 replay evidence requires a direct, mild-chain3, or D19 "
            "SSL candidate"
        )
    if arm not in {"candidate", "direct"}:
        raise ValueError("Stage-1 replay evidence arm must be candidate or direct")
    train_root = _run_dir(candidate, arm, center, "train")
    diagnostics_path = (
        train_root / "training" / "unlabeled_teacher_diagnostics.json"
    )
    if not _is_complete(train_root, "train") or not diagnostics_path.is_file():
        raise FileNotFoundError(
            "incomplete Stage-1 replay evidence: "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    config = diagnostics.get("config", {})
    expected_weight = float(candidate.pretrain_source_replay_weight or 0.0)
    expected_batches = candidate.pretrain_source_batches_per_step
    expected_total_steps = (
        candidate.pretrain_epochs * candidate.pretrain_steps
    )
    expected_samples = (
        expected_total_steps * candidate.batch_size * expected_batches
        if expected_weight > 0.0
        else 0
    )
    expected_policy = PRETRAIN_POLICY_BY_VIEW_MODE[
        candidate.pretrain_view_mode
    ]
    config_drift = {
        key: (config.get(key), expected)
        for key, expected in {
            "pretrain_view_mode": candidate.pretrain_view_mode,
            "policy": expected_policy,
            "pretrain_source_replay_weight": expected_weight,
            "pretrain_source_batches_per_step": expected_batches,
        }.items()
        if config.get(key) != expected
    }
    if config_drift:
        raise ValueError(
            f"Stage-1 replay config drifted for {candidate.candidate_id}/"
            f"{arm}/{center}: {config_drift}"
        )
    if (
        diagnostics.get("labels_consumed") is not False
        or diagnostics.get("pretrain", {}).get("steps")
        != expected_total_steps
    ):
        raise ValueError(
            f"Stage-1 target SSL contract drifted for "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    target_order_hash = diagnostics.get("ordered_batch_hashes_sha256")
    if (
        not _is_sha256(target_order_hash)
        or target_order_hash == hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError(
            f"Stage-1 target order is not proven for "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    replay = diagnostics.get("pretrain", {}).get("source_semantic_replay", {})
    replay_drift = {
        key: (replay.get(key), expected)
        for key, expected in {
            "weight": expected_weight,
            "batches_per_step": expected_batches,
            "samples": expected_samples,
            "dataset": "ptbxl",
            "partition": "train",
        }.items()
        if replay.get(key) != expected
    }
    if replay_drift:
        raise ValueError(
            f"Stage-1 source replay evidence drifted for "
            f"{candidate.candidate_id}/{arm}/{center}: {replay_drift}"
        )
    raw_source_labels = replay.get("labels_consumed")
    if expected_weight > 0.0 and raw_source_labels is not True:
        raise ValueError(
            f"Replay-on source labels are not declared for "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    if expected_weight == 0.0 and raw_source_labels not in {False, True}:
        raise ValueError(
            f"Replay-off source-label evidence is invalid for "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    source_order_hash = replay.get("ordered_batch_hashes_sha256")
    raw_source_loader = replay.get("loader")
    legacy_source_loader_gap = bool(
        allow_legacy_source_loader_gap
        and expected_weight > 0.0
        and isinstance(raw_source_loader, Mapping)
        and not raw_source_loader
    )
    source_loader_identity = (
        {}
        if legacy_source_loader_gap
        else _validated_source_loader_identity(
            replay,
            enabled=expected_weight > 0.0,
            center=center,
            model_name=candidate.model_name,
            replicate_id=candidate.replicate_id,
        )
    )
    if expected_weight > 0.0:
        if (
            not _is_sha256(source_order_hash)
            or source_order_hash == hashlib.sha256(b"").hexdigest()
            or not math.isfinite(float(replay.get("mean_bce", math.nan)))
        ):
            raise ValueError(
                f"Stage-1 source replay order/loss is not proven for "
                f"{candidate.candidate_id}/{arm}/{center}"
            )
    elif source_order_hash is not None:
        raise ValueError(
            f"Replay-off Stage-1 unexpectedly has a source order hash for "
            f"{candidate.candidate_id}/{arm}/{center}"
        )
    selection = (
        diagnostics.get("loader", {})
        .get("dataset", {})
        .get("selection", {})
    )
    required_selection = {
        "logical_center": center,
        "partition": "k500",
        "record_count": 500,
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    selection_drift = {
        key: (selection.get(key), expected)
        for key, expected in required_selection.items()
        if selection.get(key) != expected
    }
    if selection_drift:
        raise ValueError(
            f"Stage-1 K500 selection drifted for "
            f"{candidate.candidate_id}/{arm}/{center}: {selection_drift}"
        )
    return {
        "checkpoint_sha256": _sha256(
            train_root / "training" / "checkpoints" / "last.pt"
        ),
        "k500_selection": {
            key: selection.get(key)
            for key in (
                "logical_center",
                "partition",
                "record_count",
                "hash_id_set_sha256",
                "source_manifest_sha256",
                "split_manifest_sha256",
                "mapping_version",
                "mapping_hash",
            )
        },
        "target_order_sha256": target_order_hash,
        "source_replay": {
            "enabled": expected_weight > 0.0,
            "weight": expected_weight,
            "batches_per_step": expected_batches,
            "samples": expected_samples,
            "labels_consumed": expected_weight > 0.0,
            "raw_labels_consumed": raw_source_labels,
            "legacy_replay_off_literal_normalized": (
                expected_weight == 0.0 and raw_source_labels is True
            ),
            "ordered_batch_hashes_sha256": source_order_hash,
            "loader": source_loader_identity,
            "loader_evidence_status": (
                "legacy_missing_field_order_hash_present"
                if legacy_source_loader_gap
                else "complete"
            ),
        },
    }


def summarize_stage1_replay_factorial(
    no_replay_identity: Candidate,
    no_replay_augmix: Candidate,
    replay_identity: Candidate,
    replay_augmix: Candidate,
) -> Path:
    cells = {
        "I0_identity_replay_off": no_replay_identity,
        "A0_augmix_replay_off": no_replay_augmix,
        "I1_identity_replay_on": replay_identity,
        "A1_augmix_replay_on": replay_augmix,
    }
    expected_design = {
        "I0_identity_replay_off": ("clean_identity", 0.0),
        "A0_augmix_replay_off": ("twochain_augmix", 0.0),
        "I1_identity_replay_on": ("clean_identity", 0.30),
        "A1_augmix_replay_on": ("twochain_augmix", 0.30),
    }
    frozen_fields = tuple(
        field
        for field in Candidate.__dataclass_fields__
        if field
        not in {
            "candidate_id",
            "pretrain_view_mode",
            "pretrain_source_replay_weight",
        }
    )
    reference = no_replay_identity
    for name, candidate in cells.items():
        expected_view, expected_weight = expected_design[name]
        if (
            candidate.method_family != "direct_target_ssl_control"
            or candidate.pretrain_view_mode != expected_view
            or float(candidate.pretrain_source_replay_weight or 0.0)
            != expected_weight
        ):
            raise ValueError(f"invalid Stage-1 replay factorial cell: {name}")
        drift = {
            field: (getattr(candidate, field), getattr(reference, field))
            for field in frozen_fields
            if getattr(candidate, field) != getattr(reference, field)
        }
        if drift:
            raise ValueError(f"Stage-1 replay factorial drift in {name}: {drift}")
        if _sha256(_source_checkpoint(candidate)) != _sha256(
            _source_checkpoint(reference)
        ):
            raise ValueError(f"Stage-1 source checkpoint drift in {name}")

    evidence: dict[str, Any] = {}
    for center in CENTERS:
        cell_evidence = {
            name: _stage1_replay_evidence(candidate, center)
            for name, candidate in cells.items()
        }
        selections = [
            item["k500_selection"] for item in cell_evidence.values()
        ]
        if any(selection != selections[0] for selection in selections[1:]):
            raise ValueError(f"Stage-1 K500 identity differs for {center}")
        target_hashes = [
            item["target_order_sha256"] for item in cell_evidence.values()
        ]
        if any(value != target_hashes[0] for value in target_hashes[1:]):
            raise ValueError(f"Stage-1 target batch order differs for {center}")
        source_hashes = [
            cell_evidence[name]["source_replay"][
                "ordered_batch_hashes_sha256"
            ]
            for name in ("I1_identity_replay_on", "A1_augmix_replay_on")
        ]
        if source_hashes[0] != source_hashes[1]:
            raise ValueError(f"Stage-1 source replay order differs for {center}")
        no_replay_supervised = _assert_stage1_supervised_pair(
            no_replay_identity, no_replay_augmix, center
        )
        replay_supervised = _assert_stage1_supervised_pair(
            replay_identity, replay_augmix, center
        )
        if no_replay_supervised["epochs"] != replay_supervised["epochs"]:
            raise ValueError(
                f"Stage-1 supervised record order differs by replay for {center}"
            )
        evidence[center] = {
            "cells": cell_evidence,
            "supervised_pair": no_replay_supervised,
        }

    per_cell = {
        name: _arm_metrics(candidate, "candidate")
        for name, candidate in cells.items()
    }
    means = {
        name: _mean_rows(rows.values()) for name, rows in per_cell.items()
    }
    effects = {
        "augmix_at_replay_off_A0_minus_I0_pp": _difference_pp(
            means["A0_augmix_replay_off"],
            means["I0_identity_replay_off"],
        ),
        "augmix_at_replay_on_A1_minus_I1_pp": _difference_pp(
            means["A1_augmix_replay_on"],
            means["I1_identity_replay_on"],
        ),
        "replay_under_identity_I1_minus_I0_pp": _difference_pp(
            means["I1_identity_replay_on"],
            means["I0_identity_replay_off"],
        ),
        "replay_under_augmix_A1_minus_A0_pp": _difference_pp(
            means["A1_augmix_replay_on"],
            means["A0_augmix_replay_off"],
        ),
    }
    effects["interaction_pp"] = [
        effects["augmix_at_replay_on_A1_minus_I1_pp"][index]
        - effects["augmix_at_replay_off_A0_minus_I0_pp"][index]
        for index in range(4)
    ]
    summary = {
        "schema_version": 1,
        "status": "heldout_tuned_development_only_single_seed",
        "design": "stage1_pretrain_view_by_ptbxl_source_replay_2x2",
        "metric_order": [
            "clean_macro_auroc",
            "clean_macro_auprc",
            "pn2021c_macro_auroc",
            "pn2021c_macro_auprc",
        ],
        "cells": {name: candidate.__dict__ for name, candidate in cells.items()},
        "four_center_mean": means,
        "per_center_metrics": per_cell,
        "effects": effects,
        "identity_evidence": evidence,
        "target_center_access": "per_center_k500_only",
        "source_access": (
            "PTB-XL train labeled replay, one batch per target SSL step "
            "in replay-on cells"
        ),
        "outside_k500_target_model_access": False,
        "checkpoint_selection": "fixed_last_epoch_no_heldout_oracle",
        "heldout_use": "global_macro_metric_comparison_only",
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = _persistent_root(replay_augmix) / "stage1_replay_factorial.json"
    _write_json(output, summary)
    return output


def _assert_stage2_optimizer_pair(
    candidate: Candidate,
    center: str,
) -> dict[str, Any]:
    if candidate.method_family not in {
        "vae_lhat_post_refine",
        "vae_lhat_clean_polish",
        "vae_lhat_bounded_augmax_polish",
    }:
        raise ValueError(
            f"unsupported sequential Stage-2 family: {candidate.method_family}"
        )
    clean_polish = candidate.method_family == "vae_lhat_clean_polish"
    bounded_augmax = (
        candidate.method_family == "vae_lhat_bounded_augmax_polish"
    )
    single_exposure = clean_polish or bounded_augmax
    local_anchor_soft = (
        candidate.polish_auxiliary_target == "local_anchor_soft"
    )
    local_anchor_soft_geometry = (
        _local_anchor_soft_geometry(candidate)
        if local_anchor_soft
        else None
    )
    direct = _training_history(candidate, "direct", center)
    vae = _training_history(candidate, "candidate", center)
    direct_rows = _assert_epoch_count(
        direct, candidate, label=f"{center}/{candidate.candidate_id}/Direct"
    )
    vae_rows = _assert_epoch_count(
        vae, candidate, label=f"{center}/{candidate.candidate_id}/VAE"
    )
    if direct.get("optimizer_steps") != vae.get("optimizer_steps"):
        raise ValueError(
            f"Stage-2 optimizer-step budget differs for "
            f"{center}/{candidate.candidate_id}"
        )
    online_path = _candidate_root(candidate) / "online" / f"{center}.yaml"
    direct_path = _candidate_root(candidate) / "direct_train" / f"{center}.yaml"
    if _sha256(online_path) != _sha256(direct_path):
        raise ValueError(
            f"Stage-2 Direct/VAE training config differs for "
            f"{center}/{candidate.candidate_id}"
        )
    expected_optimizer_steps = candidate.epochs * math.ceil(
        500 / candidate.batch_size
    )
    diagnostics_evidence: dict[str, Any]
    if single_exposure:
        method_contracts = _method_for(candidate, center)["contracts"]
        diagnostics_evidence = {
            "mode": (
                "native_single_exposure_bounded_augmax_m20"
                if bounded_augmax
                else "native_single_exposure_vae_lhat_clean_polish"
            ),
            "objective_effective_weights": copy.deepcopy(
                method_contracts["effective_objective_weights"]
            ),
            "attack_epochs": [],
        }
    else:
        diagnostics_path = (
            _run_dir(candidate, "candidate", center, "train")
            / "training"
            / "pcgrad_diagnostics.json"
        )
        if not diagnostics_path.is_file():
            raise FileNotFoundError(
                f"missing VAE PCGrad diagnostics: {diagnostics_path}"
            )
        pcgrad = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if (
            pcgrad.get("schema_version") != 3
            or pcgrad.get("mode")
            != "fixed20_base_plus_projected_vae_lhat_aux_v1"
            or pcgrad.get("modes")
            != ["fixed20_base_plus_projected_vae_lhat_aux_v1"]
            or pcgrad.get("step_count") != expected_optimizer_steps
            or pcgrad.get("pending_auxiliary_at_exit") is not False
            or len(pcgrad.get("steps", ())) != expected_optimizer_steps
        ):
            raise ValueError(
                f"Stage-2 VAE auxiliary-step diagnostics drifted for "
                f"{center}/{candidate.candidate_id}"
            )
        diagnostics_evidence = {
            "mode": "fixed20_pcgrad_vae_lhat_auxiliary",
            "pcgrad_diagnostics_sha256": _sha256(diagnostics_path),
        }
    epoch_evidence: list[dict[str, Any]] = []
    for direct_row, vae_row in zip(direct_rows, vae_rows, strict=True):
        if direct_row.get("learning_rate") != vae_row.get("learning_rate"):
            raise ValueError(
                f"Stage-2 LR schedule differs for "
                f"{center}/{candidate.candidate_id}"
            )
        direct_exposure = direct_row.get("exposure")
        vae_exposure = vae_row.get("exposure")
        if (
            not isinstance(direct_exposure, Mapping)
            or not isinstance(vae_exposure, Mapping)
            or dict(direct_exposure) != dict(vae_exposure)
        ):
            raise ValueError("Stage-2 exposure evidence is missing")
        expected_steps_per_epoch = math.ceil(500 / candidate.batch_size)
        expected_corrupted = 0 if single_exposure else 10_000
        expected_total = 500 if single_exposure else 10_500
        expected_view_executions = 1 if single_exposure else 21
        if (
            direct_exposure.get("base_record_count") != 500
            or direct_exposure.get("clean_count") != 500
            or direct_exposure.get("corrupted_count") != expected_corrupted
            or direct_exposure.get("total_count") != expected_total
            or direct_exposure.get("view_executions_per_base_batch")
            != expected_view_executions
            or direct_exposure.get("optimizer_steps_per_base_batch") != 1
            or direct_exposure.get("optimizer_steps_this_epoch")
            != expected_steps_per_epoch
        ):
            raise ValueError(
                f"Stage-2 clean/fixed20 budget drifted for "
                f"{center}/{candidate.candidate_id}"
            )
        direct_trace = direct_row.get("stochastic_trace", {})
        vae_trace = vae_row.get("stochastic_trace", {})
        if (
            not _is_sha256(direct_trace.get("input_identity_sha256"))
            or not _is_sha256(vae_trace.get("input_identity_sha256"))
            or direct_trace.get("input_record_count") != expected_total
            or vae_trace.get("input_record_count") != expected_total
            or direct_trace.get("input_identity_sha256")
            != vae_trace.get("input_identity_sha256")
        ):
            raise ValueError(
                f"Stage-2 input order differs for "
                f"{center}/{candidate.candidate_id}"
            )
        epoch_evidence.append(
            {
                "epoch": direct_row.get("epoch"),
                "optimizer_steps_this_epoch": direct_exposure.get(
                    "optimizer_steps_this_epoch"
                ),
                "direct_input_identity_sha256": direct_trace.get(
                    "input_identity_sha256"
                ),
                "input_record_count": direct_trace.get("input_record_count"),
            }
        )
        if bounded_augmax:
            arm_attacks: dict[str, Any] = {}
            arm_train_counts: dict[str, tuple[int, int]] = {}
            for arm_name, row in (
                ("direct", direct_row),
                ("candidate", vae_row),
            ):
                diagnostics = row.get("diagnostics")
                train_payload = row.get("train")
                rejected_payload = row.get("quality_rejected")
                if (
                    not isinstance(diagnostics, Mapping)
                    or not isinstance(train_payload, Mapping)
                    or not isinstance(rejected_payload, list)
                ):
                    raise ValueError(
                        "bounded AugMax matched-compute diagnostics are missing"
                    )
                rejected_count = train_payload.get("quality_rejected_count")
                eligible_count = train_payload.get("candidate_eligible_count")
                accepted_count = train_payload.get("quality_accepted_count")
                if (
                    rejected_count != 0
                    or rejected_payload
                    or isinstance(eligible_count, bool)
                    or not isinstance(eligible_count, int)
                    or isinstance(accepted_count, bool)
                    or not isinstance(accepted_count, int)
                    or not 0 < eligible_count <= 500
                    or accepted_count != eligible_count
                ):
                    raise ValueError(
                        "bounded AugMax LHAT eligibility/QC accounting drifted: "
                        f"{arm_name} eligible={eligible_count!r}, "
                        f"accepted={accepted_count!r}, "
                        f"rejected={rejected_count!r}"
                    )
                arm_train_counts[arm_name] = (
                    eligible_count,
                    accepted_count,
                )
                attack = {
                    "loss_gain": float(
                        diagnostics.get("lhat/loss_gain", math.nan)
                    ),
                    "atk_anchor_l2": float(
                        diagnostics.get("lhat/atk_anchor_l2", math.nan)
                    ),
                    "attack_success": float(
                        diagnostics.get(
                            "lhat/decoded_anchor_sample_anyflip_asr",
                            math.nan,
                        )
                    ),
                    "decoded_invalid_rate": float(
                        diagnostics.get(
                            "lhat/decoded_invalid_rate", math.nan
                        )
                    ),
                    "quality_rejected_count": float(
                        diagnostics.get(
                            "lhat/quality_rejected_count", math.nan
                        )
                    ),
                    "effective_anchor_share": float(
                        diagnostics.get(
                            "lhat/effective_anchor_share", math.nan
                        )
                    ),
                }
                hardness_keys = (
                    "hardness_available_fraction",
                    "hardness_selected_bce_gain",
                    "hardness_selected_nonclassic_fraction",
                    "vae_weight",
                    "strength",
                )
                for key in hardness_keys:
                    values = [
                        float(
                            diagnostics.get(
                                f"augmix_{index:02d}/{key}", math.nan
                            )
                        )
                        for index in range(1, 21)
                    ]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(
                            "bounded AugMax hardness diagnostics are incomplete: "
                            f"{arm_name}/{key}"
                        )
                    if key == "hardness_selected_bce_gain" and any(
                        not 0.0 <= value <= 0.150001 for value in values
                    ):
                        raise ValueError(
                            "bounded AugMax selected BCE gain escaped [0,0.15]"
                        )
                    if key != "hardness_selected_bce_gain" and any(
                        not 0.0 <= value <= 1.0 for value in values
                    ):
                        raise ValueError(
                            f"bounded AugMax diagnostic {key} escaped [0,1]"
                        )
                    attack[f"mean_{key}"] = sum(values) / len(values)
                if (
                    not all(
                        math.isfinite(float(value))
                        for value in attack.values()
                    )
                    or attack["loss_gain"] <= 0.0
                    or not 0.0
                    < attack["atk_anchor_l2"]
                    <= candidate.lhat_pgd_epsilon_l2_standardized + 1.0e-3
                    or not 0.0 <= attack["attack_success"] <= 1.0
                    or attack["decoded_invalid_rate"] != 0.0
                    or attack["quality_rejected_count"] != 0.0
                    or not 0.0 < attack["effective_anchor_share"] <= 1.0
                ):
                    raise ValueError(
                        "bounded AugMax LHAT diagnostic gate failed: "
                        f"{arm_name}={attack}"
                    )
                arm_attacks[arm_name] = attack
            if arm_train_counts["direct"] != arm_train_counts["candidate"]:
                raise ValueError(
                    "bounded AugMax control/candidate eligibility differs"
                )
            diagnostics_evidence["attack_epochs"].append(
                {
                    "epoch": direct_row.get("epoch"),
                    "direct": arm_attacks["direct"],
                    "candidate": arm_attacks["candidate"],
                    "eligible_count": arm_train_counts["candidate"][0],
                }
            )
        if clean_polish:
            diagnostics = vae_row.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ValueError("VAE clean-polish attack diagnostics are missing")
            attack = {
                "epoch": vae_row.get("epoch"),
                "loss_gain": float(diagnostics.get("lhat/loss_gain", math.nan)),
                "atk_anchor_l2": float(
                    diagnostics.get("lhat/atk_anchor_l2", math.nan)
                ),
                "sample_anyflip_asr": float(
                    diagnostics.get(
                        "lhat/decoded_anchor_sample_anyflip_asr", math.nan
                    )
                ),
                "decoded_invalid_rate": float(
                    diagnostics.get("lhat/decoded_invalid_rate", math.nan)
                ),
                "quality_rejected_count": float(
                    diagnostics.get("lhat/quality_rejected_count", math.nan)
                ),
            }
            if local_anchor_soft:
                attack["effective_anchor_share"] = float(
                    diagnostics.get("lhat/effective_anchor_share", math.nan)
                )
                allowed_quality_reasons = {
                    "nonfinite",
                    "flatline",
                    "severe_amplitude",
                }
                for arm_name, row in (
                    ("direct", direct_row),
                    ("candidate", vae_row),
                ):
                    train_payload = row.get("train")
                    rejected_payload = row.get("quality_rejected")
                    if not isinstance(train_payload, Mapping) or not isinstance(
                        rejected_payload, list
                    ):
                        raise ValueError(
                            "local anchor-soft hard-QC accounting is missing"
                        )
                    rejected_count = train_payload.get(
                        "quality_rejected_count"
                    )
                    if (
                        isinstance(rejected_count, bool)
                        or not isinstance(rejected_count, int)
                        or rejected_count != len(rejected_payload)
                        or not 0 <= rejected_count <= 50
                    ):
                        raise ValueError(
                            "local anchor-soft hard-QC rejection count exceeds "
                            f"the frozen 10% cap: {arm_name}={rejected_count!r}"
                        )
                    for item in rejected_payload:
                        if not isinstance(item, Mapping):
                            raise TypeError(
                                "local anchor-soft hard-QC entry is malformed"
                            )
                        reason = item.get("reason")
                        if (
                            not isinstance(reason, str)
                            or not set(reason.split("+"))
                            <= allowed_quality_reasons
                        ):
                            raise ValueError(
                                "local anchor-soft hard-QC reason escaped the "
                                f"allowlist: {reason!r}"
                            )
                    attack[f"{arm_name}_hard_qc_rejected_count"] = float(
                        rejected_count
                    )
                    attack[f"{arm_name}_hard_qc_rejected_fraction"] = (
                        float(rejected_count) / 500.0
                    )
            if (
                candidate.polish_auxiliary_target
                == "latent_path_hard_label"
            ):
                attack.update(
                    {
                        "latent_path_t_mean": float(
                            diagnostics.get(
                                "lhat/latent_path_t_mean", math.nan
                            )
                        ),
                        "latent_path_bank_valid_fraction": float(
                            diagnostics.get(
                                "lhat/latent_path_bank_valid_fraction", math.nan
                            )
                        ),
                        "latent_path_selected_valid_fraction": float(
                            diagnostics.get(
                                "lhat/latent_path_selected_valid_fraction",
                                math.nan,
                            )
                        ),
                        "latent_path_t_040_fraction": float(
                            diagnostics.get(
                                "lhat/latent_path_t_040_fraction", math.nan
                            )
                        ),
                        "latent_path_t_070_fraction": float(
                            diagnostics.get(
                                "lhat/latent_path_t_070_fraction", math.nan
                            )
                        ),
                        "latent_path_t_100_fraction": float(
                            diagnostics.get(
                                "lhat/latent_path_t_100_fraction", math.nan
                            )
                        ),
                    }
                )
            if (
                candidate.polish_auxiliary_target
                == "feature_invariance"
            ):
                direct_train = direct_row.get("train")
                vae_train = vae_row.get("train")
                direct_terms = (
                    direct_train.get("objective_terms")
                    if isinstance(direct_train, Mapping)
                    else None
                )
                vae_terms = (
                    vae_train.get("objective_terms")
                    if isinstance(vae_train, Mapping)
                    else None
                )
                if not isinstance(direct_terms, Mapping) or not isinstance(
                    vae_terms, Mapping
                ):
                    raise ValueError(
                        "feature-invariance objective diagnostics are missing"
                    )
                control_cosine = float(
                    direct_terms.get(
                        "clean_identity_feature_cosine", math.nan
                    )
                )
                lhat_cosine = float(
                    vae_terms.get("clean_lhat_feature_cosine", math.nan)
                )
                if (
                    not math.isfinite(control_cosine)
                    or not math.isfinite(lhat_cosine)
                    or not 0.0 <= control_cosine <= 2.0
                    or not 0.0 < lhat_cosine <= 2.0
                ):
                    raise ValueError(
                        "feature-invariance cosine diagnostics are invalid: "
                        f"control={control_cosine}, lhat={lhat_cosine}"
                    )
                attack.update(
                    {
                        "clean_identity_feature_cosine": control_cosine,
                        "clean_lhat_feature_cosine": lhat_cosine,
                    }
                )
            if (
                not all(math.isfinite(float(value)) for value in attack.values())
                or attack["loss_gain"] <= 0.0
                or (
                    (
                        not 0.0
                        < attack["atk_anchor_l2"]
                        <= candidate.lhat_pgd_epsilon_l2_standardized + 1.0e-3
                        or attack["effective_anchor_share"]
                        < local_anchor_soft_geometry[
                            "minimum_effective_anchor_mass"
                        ]
                        - 1.0e-6
                    )
                    if local_anchor_soft
                    else not math.isclose(
                        attack["atk_anchor_l2"],
                        candidate.lhat_pgd_epsilon_l2_standardized,
                        rel_tol=0.0,
                        abs_tol=1.0e-3,
                    )
                )
                or not 0.0 <= attack["sample_anyflip_asr"] <= 1.0
                or attack["decoded_invalid_rate"] != 0.0
                or (
                    not local_anchor_soft
                    and attack["quality_rejected_count"] != 0.0
                )
            ):
                raise ValueError(
                    f"VAE clean-polish attack gate failed for "
                    f"{center}/{candidate.candidate_id}: {attack}"
                )
            if (
                candidate.polish_auxiliary_target
                == "latent_path_hard_label"
            ):
                fractions = tuple(
                    attack[name]
                    for name in (
                        "latent_path_t_040_fraction",
                        "latent_path_t_070_fraction",
                        "latent_path_t_100_fraction",
                    )
                )
                if (
                    not 0.4 <= attack["latent_path_t_mean"] <= 1.0
                    or not math.isclose(
                        attack["latent_path_selected_valid_fraction"],
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    or not 1.0 / 3.0
                    <= attack["latent_path_bank_valid_fraction"]
                    <= 1.0
                    or not math.isclose(
                        sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-6
                    )
                ):
                    raise ValueError(
                        f"VAE latent-path gate failed for "
                        f"{center}/{candidate.candidate_id}: {attack}"
                    )
            diagnostics_evidence["attack_epochs"].append(attack)
    return {
        "optimizer_steps": direct.get("optimizer_steps"),
        "training_config_sha256": _sha256(online_path),
        "diagnostics": diagnostics_evidence,
        "epochs": epoch_evidence,
    }


def _assert_sequential_factorial(
    identity_post: Candidate,
    augmix_post: Candidate,
) -> tuple[Candidate, Candidate, dict[str, Any]]:
    if (
        identity_post.method_family == "strong_lhat_augmix_chain3"
        or augmix_post.method_family == "strong_lhat_augmix_chain3"
    ):
        if (
            identity_post.method_family != "strong_lhat_augmix_chain3"
            or augmix_post.method_family != "strong_lhat_augmix_chain3"
        ):
            raise ValueError(
                "mild-chain3 factorial requires the same method in C and D"
            )
        if (
            identity_post.initializer_candidate is not None
            or augmix_post.initializer_candidate is not None
        ):
            raise ValueError(
                "integrated mild-chain3 factorial must start from the locked "
                "PTB-XL source checkpoint"
            )
        if identity_post.pretrain_view_mode != "clean_identity":
            raise ValueError("mild-chain3 C arm must use clean-identity SSL")
        if augmix_post.pretrain_view_mode != "twochain_augmix":
            raise ValueError("mild-chain3 D arm must use two-chain AugMix SSL")
        frozen_fields = tuple(
            field
            for field in Candidate.__dataclass_fields__
            if field not in {"candidate_id", "pretrain_view_mode"}
        )
        drift = {
            field: (
                getattr(identity_post, field),
                getattr(augmix_post, field),
            )
            for field in frozen_fields
            if getattr(identity_post, field) != getattr(augmix_post, field)
        }
        if drift:
            raise ValueError(f"mild-chain3 factorial settings drifted: {drift}")
        if (
            identity_post.pretrain_steps != 4096
            or identity_post.pretrain_epochs != 1
            or identity_post.pretrain_ssl_objective != "simclr_vicreg"
            or not math.isclose(
                identity_post.pretrain_vicreg_mix_weight,
                0.03,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or identity_post.epochs != 20
            or identity_post.scheduler_horizon_epochs != 20
            or not math.isclose(
                identity_post.auxiliary_alpha,
                0.25,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or float(identity_post.pretrain_source_replay_weight or 0.0) != 0.0
        ):
            raise ValueError("mild-chain3 factorial departed from frozen L11")
        if _sha256(_source_checkpoint(identity_post)) != _sha256(
            _source_checkpoint(augmix_post)
        ):
            raise ValueError("mild-chain3 source checkpoint differs across views")

        center_evidence: dict[str, Any] = {}
        for center in CENTERS:
            cells = {
                "A_identity_direct": _stage1_replay_evidence(
                    identity_post,
                    center,
                    arm="direct",
                ),
                "B_augmix_direct": _stage1_replay_evidence(
                    augmix_post,
                    center,
                    arm="direct",
                ),
                "C_identity_vae": _stage1_replay_evidence(
                    identity_post,
                    center,
                    arm="candidate",
                ),
                "D_augmix_vae": _stage1_replay_evidence(
                    augmix_post,
                    center,
                    arm="candidate",
                ),
            }
            selections = [cell["k500_selection"] for cell in cells.values()]
            if any(selection != selections[0] for selection in selections[1:]):
                raise ValueError(f"L11 K500 identity differs for {center}")
            target_orders = [
                cell["target_order_sha256"] for cell in cells.values()
            ]
            if any(value != target_orders[0] for value in target_orders[1:]):
                raise ValueError(f"L11 target SSL order differs for {center}")
            if any(
                cell["source_replay"]["enabled"] for cell in cells.values()
            ):
                raise ValueError(f"L11 unexpectedly used source replay for {center}")
            center_evidence[center] = {
                "cells": cells,
                "identity_control_vs_vae_outer": _assert_fixed20_outer_pair(
                    identity_post,
                    "direct",
                    identity_post,
                    "candidate",
                    center,
                    label="L11 identity Direct versus VAE",
                ),
                "augmix_control_vs_vae_outer": _assert_fixed20_outer_pair(
                    augmix_post,
                    "direct",
                    augmix_post,
                    "candidate",
                    center,
                    label="L11 AugMix Direct versus VAE",
                ),
                "direct_identity_vs_augmix_outer": _assert_fixed20_outer_pair(
                    identity_post,
                    "direct",
                    augmix_post,
                    "direct",
                    center,
                    label="L11 Direct identity versus AugMix",
                ),
                "vae_identity_vs_augmix_outer": _assert_fixed20_outer_pair(
                    identity_post,
                    "candidate",
                    augmix_post,
                    "candidate",
                    center,
                    label="L11 VAE identity versus AugMix",
                ),
            }
        return identity_post, augmix_post, center_evidence

    supported_stage2 = {
        "vae_lhat_post_refine",
        "vae_lhat_clean_polish",
        "vae_lhat_bounded_augmax_polish",
    }
    if (
        identity_post.method_family not in supported_stage2
        or augmix_post.method_family != identity_post.method_family
    ):
        raise ValueError(
            "both Stage-2 candidates must use the same supported VAE-LHAT "
            "refinement family"
        )
    if (
        identity_post.initializer_candidate is None
        or augmix_post.initializer_candidate is None
    ):
        raise ValueError("both Stage-2 candidates require Stage-1 initializers")
    identity_stage1 = CANDIDATES[identity_post.initializer_candidate]
    augmix_stage1 = CANDIDATES[augmix_post.initializer_candidate]
    if identity_stage1.pretrain_view_mode != "clean_identity":
        raise ValueError("identity Stage-1 arm is not clean_identity")
    if augmix_stage1.pretrain_view_mode != "twochain_augmix":
        raise ValueError("AugMix Stage-1 arm is not twochain_augmix")
    replay_weight = float(
        identity_stage1.pretrain_source_replay_weight or 0.0
    )
    if replay_weight not in {0.0, 0.30}:
        raise ValueError(
            "sequential factorial uses an unregistered PTB-XL source replay "
            f"weight: {replay_weight}"
        )
    if (
        float(augmix_stage1.pretrain_source_replay_weight or 0.0)
        != replay_weight
        or identity_stage1.pretrain_source_batches_per_step
        != augmix_stage1.pretrain_source_batches_per_step
    ):
        raise ValueError(
            "sequential factorial Stage-1 arms use different source replay "
            "budgets"
        )
    stage1_fields = (
        "model_name",
        "learning_rate",
        "batch_size",
        "epochs",
        "scheduler_horizon_epochs",
        "weight_decay",
        "pretrain_epochs",
        "pretrain_steps",
        "pretrain_learning_rate",
        "pretrain_logit_anchor_weight",
        "supervised_logit_anchor_weight",
        "pretrain_ssl_objective",
        "pretrain_ssl_weight",
        "pretrain_vicreg_mix_weight",
        "pretrain_source_replay_weight",
        "pretrain_source_batches_per_step",
        "method_family",
        "comparison_rng_identity",
    )
    stage2_fields = (
        "model_name",
        "learning_rate",
        "batch_size",
        "epochs",
        "scheduler_horizon_epochs",
        "weight_decay",
        "pretrain_epochs",
        "pretrain_steps",
        "auxiliary_alpha",
        "method_family",
        "comparison_rng_identity",
        "polish_auxiliary_target",
        "frozen_teacher_mix",
        "lhat_pgd_epsilon_l2_standardized",
        "local_lhat_geometry_preset",
        "lhat_geometry_preset",
    )
    mismatches: dict[str, Any] = {}
    for prefix, left, right, fields in (
        ("stage1", identity_stage1, augmix_stage1, stage1_fields),
        ("stage2", identity_post, augmix_post, stage2_fields),
    ):
        for field in fields:
            if getattr(left, field) != getattr(right, field):
                mismatches[f"{prefix}.{field}"] = (
                    getattr(left, field),
                    getattr(right, field),
                )
    if mismatches:
        raise ValueError(f"sequential factorial settings drifted: {mismatches}")
    center_evidence: dict[str, Any] = {}
    for center in CENTERS:
        identity_evidence = _parent_evidence(identity_post, center)
        augmix_evidence = _parent_evidence(augmix_post, center)
        identity_k500 = identity_evidence["k500_selection"]
        augmix_k500 = augmix_evidence["k500_selection"]
        for key in (
            "hash_id_set_sha256",
            "split_manifest_sha256",
            "source_manifest_sha256",
            "mapping_hash",
            "mapping_version",
            "record_count",
        ):
            if identity_k500.get(key) != augmix_k500.get(key):
                raise ValueError(
                    f"Stage-1 K500 identity differs for {center}/{key}"
                )
        if (
            identity_evidence["ordered_pretrain_batch_hashes_sha256"]
            != augmix_evidence["ordered_pretrain_batch_hashes_sha256"]
        ):
            raise ValueError(
                f"Stage-1 pretrain record order differs for {center}"
            )
        if (
            identity_evidence["source_replay_evidence"]
            ["ordered_batch_hashes_sha256"]
            != augmix_evidence["source_replay_evidence"]
            ["ordered_batch_hashes_sha256"]
        ):
            raise ValueError(
                f"Stage-1 source replay order differs for {center}"
            )
        stage1_supervised = _assert_stage1_supervised_pair(
            identity_stage1,
            augmix_stage1,
            center,
        )
        identity_stage2 = _assert_stage2_optimizer_pair(identity_post, center)
        augmix_stage2 = _assert_stage2_optimizer_pair(augmix_post, center)
        if identity_stage2["epochs"] != augmix_stage2["epochs"]:
            raise ValueError(
                f"Stage-2 input order differs across initializers for {center}"
            )
        center_evidence[center] = {
            "k500_selection": identity_k500,
            "ordered_pretrain_batch_hashes_sha256": identity_evidence[
                "ordered_pretrain_batch_hashes_sha256"
            ],
            "source_replay_evidence": identity_evidence[
                "source_replay_evidence"
            ],
            "identity_initializer_sha256": identity_evidence["checkpoint_sha256"],
            "augmix_initializer_sha256": augmix_evidence["checkpoint_sha256"],
            "stage1_supervised_pair": stage1_supervised,
            "identity_initializer_stage2_pair": identity_stage2,
            "augmix_initializer_stage2_pair": augmix_stage2,
        }
    return identity_stage1, augmix_stage1, center_evidence


def summarize_stage1_scaling_pair(
    identity: Candidate,
    augmix: Candidate,
    *,
    direct: Candidate,
    reference_augmix: Candidate,
    benchmark: Candidate | None,
) -> Path:
    cells = {
        "D_shared_direct": direct,
        "I_identity_ssl": identity,
        "A_augmix_ssl": augmix,
        "R_previous_augmix": reference_augmix,
    }
    if (
        identity.method_family != "direct_target_ssl_control"
        or augmix.method_family != "direct_target_ssl_control"
        or identity.pretrain_view_mode != "clean_identity"
        or augmix.pretrain_view_mode != "twochain_augmix"
    ):
        raise ValueError("invalid Stage-1 identity/AugMix scaling pair")
    frozen_fields = tuple(
        field
        for field in Candidate.__dataclass_fields__
        if field not in {"candidate_id", "pretrain_view_mode"}
    )
    drift = {
        field: (getattr(identity, field), getattr(augmix, field))
        for field in frozen_fields
        if getattr(identity, field) != getattr(augmix, field)
    }
    if drift:
        raise ValueError(f"Stage-1 scaling pair drifted: {drift}")
    total_steps = identity.pretrain_epochs * identity.pretrain_steps
    if total_steps != augmix.pretrain_epochs * augmix.pretrain_steps:
        raise ValueError("Stage-1 total SSL update budget differs")
    if (
        float(identity.pretrain_source_replay_weight or 0.0) != 0.0
        or float(augmix.pretrain_source_replay_weight or 0.0) != 0.0
    ):
        raise ValueError("simple Stage-1 scaling pair must not use source replay")
    _assert_matched_direct(identity, direct)
    _assert_matched_direct(augmix, direct)
    if _sha256(_source_checkpoint(identity)) != _sha256(
        _source_checkpoint(augmix)
    ):
        raise ValueError("Stage-1 source checkpoint differs across views")

    evidence: dict[str, Any] = {}
    for center in CENTERS:
        identity_evidence = _stage1_replay_evidence(identity, center)
        augmix_evidence = _stage1_replay_evidence(augmix, center)
        if (
            identity_evidence["k500_selection"]
            != augmix_evidence["k500_selection"]
        ):
            raise ValueError(f"Stage-1 K500 identity differs for {center}")
        if (
            identity_evidence["target_order_sha256"]
            != augmix_evidence["target_order_sha256"]
        ):
            raise ValueError(f"Stage-1 target batch order differs for {center}")
        if (
            identity_evidence["source_replay"]["enabled"]
            or augmix_evidence["source_replay"]["enabled"]
        ):
            raise ValueError(
                f"Stage-1 source replay unexpectedly active for {center}"
            )
        evidence[center] = {
            "identity": identity_evidence,
            "augmix": augmix_evidence,
            "supervised_pair": _assert_stage1_supervised_pair(
                identity, augmix, center
            ),
        }

    per_cell = {
        "D_shared_direct": _arm_metrics(direct, "direct"),
        "I_identity_ssl": _arm_metrics(identity, "candidate"),
        "A_augmix_ssl": _arm_metrics(augmix, "candidate"),
        "R_previous_augmix": _arm_metrics(reference_augmix, "candidate"),
    }
    means = {
        name: _mean_rows(rows.values()) for name, rows in per_cell.items()
    }
    effects = {
        "augmix_minus_identity_pp": _difference_pp(
            means["A_augmix_ssl"], means["I_identity_ssl"]
        ),
        "identity_minus_direct_pp": _difference_pp(
            means["I_identity_ssl"], means["D_shared_direct"]
        ),
        "augmix_minus_direct_pp": _difference_pp(
            means["A_augmix_ssl"], means["D_shared_direct"]
        ),
        "augmix_minus_previous_augmix_pp": _difference_pp(
            means["A_augmix_ssl"], means["R_previous_augmix"]
        ),
    }
    benchmark_payload = None
    if benchmark is not None:
        benchmark_mean = _mean_rows(
            _arm_metrics(benchmark, "candidate").values()
        )
        benchmark_payload = {
            "candidate_id": benchmark.candidate_id,
            "four_center_mean": benchmark_mean,
            "augmix_minus_benchmark_pp": _difference_pp(
                means["A_augmix_ssl"], benchmark_mean
            ),
            "warning": (
                "Historical compound heldout-tuned single-seed reference; "
                "not part of the simple causal pair."
            ),
        }
    summary = {
        "schema_version": 1,
        "status": "heldout_tuned_development_only_single_seed",
        "design": "stage1_total_ssl_updates_identity_vs_augmix_pair",
        "metric_order": [
            "clean_macro_auroc",
            "clean_macro_auprc",
            "pn2021c_macro_auroc",
            "pn2021c_macro_auprc",
        ],
        "cells": {name: candidate.__dict__ for name, candidate in cells.items()},
        "total_ssl_optimizer_steps": total_steps,
        "four_center_mean": means,
        "balanced_clean_robust_auprc": {
            name: 0.5 * (row[1] + row[3]) for name, row in means.items()
        },
        "per_center_metrics": per_cell,
        "effects": effects,
        "benchmark": benchmark_payload,
        "identity_evidence": evidence,
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = _persistent_root(augmix) / "stage1_ssl_scaling_comparison.json"
    _write_json(output, summary)
    return output


def summarize_factorial(
    identity_post: Candidate,
    augmix_post: Candidate,
    *,
    benchmark: Candidate | None,
) -> Path:
    identity_stage1, augmix_stage1, evidence = _assert_sequential_factorial(
        identity_post,
        augmix_post,
    )
    per_arm = {
        "A_identity_ssl_direct_control": _arm_metrics(identity_post, "direct"),
        "B_augmix_ssl_direct": _arm_metrics(augmix_post, "direct"),
        "C_identity_ssl_then_vae": _arm_metrics(identity_post, "candidate"),
        "D_augmix_ssl_then_vae": _arm_metrics(augmix_post, "candidate"),
    }
    means = {
        name: _mean_rows(center_rows.values())
        for name, center_rows in per_arm.items()
    }
    effects = {
        "augmix_without_vae_B_minus_A_pp": _difference_pp(
            means["B_augmix_ssl_direct"],
            means["A_identity_ssl_direct_control"],
        ),
        "vae_after_identity_C_minus_A_pp": _difference_pp(
            means["C_identity_ssl_then_vae"],
            means["A_identity_ssl_direct_control"],
        ),
        "vae_after_augmix_D_minus_B_pp": _difference_pp(
            means["D_augmix_ssl_then_vae"],
            means["B_augmix_ssl_direct"],
        ),
        "augmix_with_vae_D_minus_C_pp": _difference_pp(
            means["D_augmix_ssl_then_vae"],
            means["C_identity_ssl_then_vae"],
        ),
    }
    effects["interaction_D_minus_B_minus_C_minus_A_pp"] = [
        effects["vae_after_augmix_D_minus_B_pp"][index]
        - effects["vae_after_identity_C_minus_A_pp"][index]
        for index in range(4)
    ]
    benchmark_payload = None
    if benchmark is not None:
        benchmark_rows = _arm_metrics(benchmark, "candidate")
        benchmark_mean = _mean_rows(benchmark_rows.values())
        benchmark_payload = {
            "candidate_id": benchmark.candidate_id,
            "four_center_mean": benchmark_mean,
            "D_minus_benchmark_pp": _difference_pp(
                means["D_augmix_ssl_then_vae"], benchmark_mean
            ),
            "warning": (
                "Historical benchmark only; it is a compound heldout-tuned "
                "single-seed method and is not part of the causal factorial."
            ),
        }
    integrated_mild_chain3 = (
        identity_post.method_family == "strong_lhat_augmix_chain3"
    )
    vae_after_augmix_by_center = {
        center: _difference_pp(
            per_arm["D_augmix_ssl_then_vae"][center],
            per_arm["B_augmix_ssl_direct"][center],
        )
        for center in CENTERS
    }
    nonnegative_robust_auprc_centers = sum(
        delta[3] >= -1.0e-12
        for delta in vae_after_augmix_by_center.values()
    )
    mechanism_gate = {
        "robust_auprc_gain_at_least_0p15pp": (
            effects["vae_after_augmix_D_minus_B_pp"][3] >= 0.15
        ),
        "clean_auprc_loss_no_worse_than_0p25pp": (
            effects["vae_after_augmix_D_minus_B_pp"][1] >= -0.25
        ),
        "nonnegative_robust_auprc_centers": (
            nonnegative_robust_auprc_centers
        ),
        "at_least_three_nonnegative_robust_auprc_centers": (
            nonnegative_robust_auprc_centers >= 3
        ),
    }
    mechanism_gate["passed"] = all(
        (
            mechanism_gate["robust_auprc_gain_at_least_0p15pp"],
            mechanism_gate["clean_auprc_loss_no_worse_than_0p25pp"],
            mechanism_gate[
                "at_least_three_nonnegative_robust_auprc_centers"
            ],
        )
    )
    transfer_gate = {
        "benchmark_available": benchmark_payload is not None,
        "D_exceeds_benchmark_clean_auprc": (
            benchmark_payload is not None
            and means["D_augmix_ssl_then_vae"][1]
            > benchmark_payload["four_center_mean"][1]
        ),
        "D_exceeds_benchmark_pn2021c_auprc": (
            benchmark_payload is not None
            and means["D_augmix_ssl_then_vae"][3]
            > benchmark_payload["four_center_mean"][3]
        ),
    }
    transfer_gate["passed"] = all(
        (
            mechanism_gate["passed"],
            transfer_gate["D_exceeds_benchmark_clean_auprc"],
            transfer_gate["D_exceeds_benchmark_pn2021c_auprc"],
        )
    )
    summary = {
        "schema_version": 1,
        "status": "heldout_tuned_development_only_single_seed",
        "design": (
            "integrated_ssl_then_mild_lhat_chain3_2x2"
            if integrated_mild_chain3
            else "sequential_stage1_augmix_by_stage2_vae_lhat_2x2"
        ),
        "metric_order": [
            "clean_macro_auroc",
            "clean_macro_auprc",
            "pn2021c_macro_auroc",
            "pn2021c_macro_auprc",
        ],
        "stage1": {
            "identity": identity_stage1.__dict__,
            "augmix": augmix_stage1.__dict__,
            "only_intended_mechanism_difference": (
                "clean_identity versus twochain_augmix pretrain views"
            ),
            "ptbxl_source_replay": {
                "enabled": bool(
                    identity_stage1.pretrain_source_replay_weight
                ),
                "weight": float(
                    identity_stage1.pretrain_source_replay_weight or 0.0
                ),
                "batches_per_step": (
                    identity_stage1.pretrain_source_batches_per_step
                ),
                "samples_per_center": (
                    identity_stage1.pretrain_steps
                    * identity_stage1.batch_size
                    * identity_stage1.pretrain_source_batches_per_step
                    if identity_stage1.pretrain_source_replay_weight
                    else 0
                ),
            },
            "identity_arm_note": (
                "Equal-step clean-clean SimCLR/VICReg control; this is not a "
                "plain no-SSL Direct run."
            ),
        },
        "stage2": {
            "identity_initializer_pair": identity_post.__dict__,
            "augmix_initializer_pair": augmix_post.__dict__,
            "direct_and_vae_optimizer_budget_matched": True,
            "vae_extra_attack_compute_reported_separately": True,
            "execution": (
                "same-run supervised phase after Stage-1 SSL"
                if integrated_mild_chain3
                else "checkpoint continuation after completed Stage-1"
            ),
            "mild_chain3_contract": (
                {
                    "lhat_label_policy": "exact_positive_set_nonself",
                    "hull_lambda": 0.60,
                    "hard_steps": 5,
                    "pgd_epsilon_l2_standardized": 8.0,
                    "auxiliary_alpha": 0.25,
                    "coupling": "ordinary_threechain_augmix_plus_jsd",
                    "fixed20_base_gradient_preserved": True,
                }
                if integrated_mild_chain3
                else None
            ),
        },
        "four_center_mean": means,
        "per_center_metrics": per_arm,
        "effects": effects,
        "vae_after_augmix_by_center_pp": vae_after_augmix_by_center,
        "predeclared_mechanism_gate": mechanism_gate,
        "ecgfounder_transfer_gate": transfer_gate,
        "benchmark": benchmark_payload,
        "identity_evidence": evidence,
        "ptbxl_source_floor": {
            "status": "not_evaluated",
            "recipe_freeze_allowed": False,
            "reason": (
                "Target factorial results may be inspected for development, but "
                "the recipe cannot be frozen or claimed source-preserving until "
                "all four adapted arms receive the same PTB-XL source evaluation."
            ),
        },
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
    }
    output = _persistent_root(augmix_post) / "factorial_summary.json"
    _write_json(output, summary)
    return output


def blend_checkpoints(
    candidate: Candidate,
    *,
    direct_candidate: Candidate,
    method_candidate: Candidate,
) -> Path:
    """Materialize minimal, auditable Direct-to-method weight interpolations."""

    import torch

    alpha = candidate.checkpoint_blend_alpha
    if candidate.method_family != "checkpoint_blend" or alpha is None:
        raise ValueError("candidate is not a checkpoint blend")
    if not 0.0 < alpha < 1.0:
        raise ValueError("checkpoint blend alpha must lie strictly in (0,1)")
    if not (
        candidate.model_name
        == direct_candidate.model_name
        == method_candidate.model_name
    ):
        raise ValueError("checkpoint blend model families must match")

    blend_records: dict[str, Any] = {}
    for center in CENTERS:
        direct_path = (
            _persistent_root(direct_candidate)
            / "direct"
            / center
            / "train"
            / "training"
            / "checkpoints"
            / "last.pt"
        )
        method_path = (
            _persistent_root(method_candidate)
            / "candidate"
            / center
            / "train"
            / "training"
            / "checkpoints"
            / "last.pt"
        )
        if not direct_path.is_file() or not method_path.is_file():
            raise FileNotFoundError(
                f"missing blend endpoint for {center}: "
                f"direct={direct_path.is_file()} method={method_path.is_file()}"
            )
        direct_payload = torch.load(
            direct_path, map_location="cpu", weights_only=True
        )
        method_payload = torch.load(
            method_path, map_location="cpu", weights_only=True
        )
        direct_state = direct_payload["model_state_dict"]
        method_state = method_payload["model_state_dict"]
        if direct_state.keys() != method_state.keys():
            raise ValueError(f"checkpoint key mismatch for {center}")
        blended: dict[str, Any] = {}
        for key in direct_state:
            direct_tensor = direct_state[key]
            method_tensor = method_state[key]
            if (
                direct_tensor.shape != method_tensor.shape
                or direct_tensor.dtype != method_tensor.dtype
            ):
                raise ValueError(f"checkpoint tensor mismatch: {center}/{key}")
            if direct_tensor.is_floating_point():
                blended[key] = direct_tensor.mul(1.0 - alpha).add(
                    method_tensor, alpha=alpha
                )
            else:
                blended[key] = method_tensor.clone()
        output = (
            _persistent_root(candidate)
            / "candidate"
            / center
            / "train"
            / "training"
            / "checkpoints"
            / "last.pt"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_state_dict": blended,
                "center": center,
                "blend": {
                    "direct_checkpoint": str(direct_path),
                    "direct_sha256": _sha256(direct_path),
                    "method_checkpoint": str(method_path),
                    "method_sha256": _sha256(method_path),
                    "method_weight": alpha,
                    "direct_weight": 1.0 - alpha,
                },
            },
            output,
        )
        blend_records[center] = {
            "checkpoint": str(output),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        }
        del direct_payload, method_payload, direct_state, method_state, blended
    manifest = {
        "schema_version": 1,
        "candidate": candidate.__dict__,
        "direct_candidate": direct_candidate.candidate_id,
        "method_candidate": method_candidate.candidate_id,
        "adaptation_population": "per_center_k500_only",
        "outside_k500_model_access": False,
        "heldout_use": "global_macro_metric_comparison_only",
        "centers": blend_records,
    }
    output_manifest = _persistent_root(candidate) / "blend_manifest.json"
    _write_json(output_manifest, manifest)
    return output_manifest


def clean_failed(
    candidate: Candidate,
    *,
    arm: str,
    center: str | None = None,
) -> Path:
    """Remove only owned incomplete outputs after the relevant worker exited."""

    if arm not in {"candidate", "direct"}:
        raise ValueError("arm must be candidate or direct")
    if center is not None and center not in CENTERS:
        raise ValueError(f"unknown center: {center}")
    arm_root = (_persistent_root(candidate) / arm).resolve()
    expected_parent = _persistent_root(candidate).resolve()
    if arm_root.parent != expected_parent:
        raise RuntimeError(f"refusing unexpected cleanup root: {arm_root}")
    root = arm_root if center is None else (arm_root / center).resolve()
    if center is not None and root.parent != arm_root:
        raise RuntimeError(f"refusing unexpected center cleanup root: {root}")
    if not root.exists():
        return root
    completed: list[Path] = []
    checkpoints: list[Path] = []
    targets = CENTERS if center is None else (center,)
    for target_center in targets:
        center_root = arm_root / target_center
        train_result = (
            center_root / "train" / "training" / "train_result.json"
        )
        eval_result = (
            center_root / "eval" / "evaluation" / "evaluation_result.json"
        )
        checkpoint = (
            center_root
            / "train"
            / "training"
            / "checkpoints"
            / "last.pt"
        )
        completed.extend(path for path in (train_result, eval_result) if path.exists())
        if checkpoint.exists():
            checkpoints.append(checkpoint)
    if completed or checkpoints:
        raise RuntimeError(
            "refusing to remove an arm with completed artifacts: "
            f"results={completed}, checkpoints={checkpoints}"
        )
    shutil.rmtree(root)
    return root


def _candidate_from_name(name: str) -> Candidate:
    try:
        return CANDIDATES[name]
    except KeyError:
        raise ValueError(
            f"unknown candidate {name!r}; choose from {sorted(CANDIDATES)}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--candidate", required=True)
    prepare_parser.add_argument("--force", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--candidate", required=True)
    run_parser.add_argument("--arm", choices=("candidate", "direct"), required=True)
    run_parser.add_argument("--stage", choices=("train", "eval"), required=True)
    run_parser.add_argument("--gpus", nargs="+", type=int, required=True)
    run_parser.add_argument("--cpu-threads", type=int, default=8)
    run_parser.add_argument("--jobs-per-gpu", type=int, default=1)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--candidate", required=True)
    summarize_parser.add_argument("--direct-candidate")
    summarize_parser.add_argument(
        "--direct-arm",
        choices=("direct", "candidate"),
        default="direct",
    )
    replicate_parser = subparsers.add_parser("summarize-replicates")
    replicate_parser.add_argument("--reference-candidate", required=True)
    replicate_parser.add_argument("--reference-direct-candidate")
    replicate_parser.add_argument("--replicate-candidate", required=True)
    replicate_parser.add_argument("--replicate-direct-candidate")
    replicate_parser.add_argument(
        "--additional-candidate",
        action="append",
        default=[],
    )
    replicate_parser.add_argument(
        "--additional-direct-candidate",
        action="append",
        default=[],
    )
    replicate_parser.add_argument(
        "--benchmark-candidate",
        default="p22_ssl4096_aux150_simclr_vicreg",
    )
    factorial_parser = subparsers.add_parser("summarize-factorial")
    factorial_parser.add_argument("--identity-post-candidate", required=True)
    factorial_parser.add_argument("--augmix-post-candidate", required=True)
    factorial_parser.add_argument(
        "--benchmark-candidate",
        default="none",
    )
    replay_factorial_parser = subparsers.add_parser(
        "summarize-stage1-replay-factorial"
    )
    replay_factorial_parser.add_argument(
        "--no-replay-identity-candidate", required=True
    )
    replay_factorial_parser.add_argument(
        "--no-replay-augmix-candidate", required=True
    )
    replay_factorial_parser.add_argument(
        "--replay-identity-candidate", required=True
    )
    replay_factorial_parser.add_argument(
        "--replay-augmix-candidate", required=True
    )
    scaling_parser = subparsers.add_parser("summarize-stage1-scaling")
    scaling_parser.add_argument("--identity-candidate", required=True)
    scaling_parser.add_argument("--augmix-candidate", required=True)
    scaling_parser.add_argument("--direct-candidate", required=True)
    scaling_parser.add_argument("--reference-augmix-candidate", required=True)
    scaling_parser.add_argument("--benchmark-candidate", default="none")
    blend_parser = subparsers.add_parser("blend")
    blend_parser.add_argument("--candidate", required=True)
    blend_parser.add_argument("--direct-candidate", required=True)
    blend_parser.add_argument("--method-candidate", required=True)
    clean_parser = subparsers.add_parser("clean-failed")
    clean_parser.add_argument("--candidate", required=True)
    clean_parser.add_argument("--arm", choices=("candidate", "direct"), required=True)
    clean_parser.add_argument("--center", choices=CENTERS)
    source_floor_parser = subparsers.add_parser("source-floor")
    source_floor_parser.add_argument("--candidate", required=True)
    source_floor_parser.add_argument(
        "--arm",
        choices=("candidate", "direct"),
        required=True,
    )
    source_floor_parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
    )
    source_floor_parser.add_argument("--cpu-threads", type=int, default=4)
    source_floor_worker = subparsers.add_parser("source-floor-worker")
    source_floor_worker.add_argument("--candidate", required=True)
    source_floor_worker.add_argument(
        "--arm",
        choices=("candidate", "direct"),
        required=True,
    )
    source_floor_worker.add_argument("--center", choices=CENTERS, required=True)
    subparsers.add_parser("list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print(json.dumps({k: v.__dict__ for k, v in CANDIDATES.items()}, indent=2))
        return 0
    if args.command == "source-floor-worker":
        candidate = _candidate_from_name(args.candidate)
        print(
            evaluate_source_floor_one(
                candidate,
                arm=args.arm,
                center=args.center,
            )
        )
        return 0
    if args.command == "summarize-factorial":
        print(
            summarize_factorial(
                _candidate_from_name(args.identity_post_candidate),
                _candidate_from_name(args.augmix_post_candidate),
                benchmark=(
                    None
                    if args.benchmark_candidate == "none"
                    else _candidate_from_name(args.benchmark_candidate)
                ),
            )
        )
        return 0
    if args.command == "summarize-replicates":
        if len(args.additional_candidate) != len(
            args.additional_direct_candidate
        ):
            raise ValueError(
                "each additional candidate requires one additional Direct owner"
            )
        print(
            summarize_replicates(
                _candidate_from_name(args.reference_candidate),
                _candidate_from_name(args.replicate_candidate),
                benchmark=_candidate_from_name(args.benchmark_candidate),
                reference_direct=(
                    None
                    if args.reference_direct_candidate is None
                    else _candidate_from_name(
                        args.reference_direct_candidate
                    )
                ),
                replicate_direct=(
                    None
                    if args.replicate_direct_candidate is None
                    else _candidate_from_name(
                        args.replicate_direct_candidate
                    )
                ),
                additional_pairs=tuple(
                    (
                        _candidate_from_name(candidate_name),
                        _candidate_from_name(direct_name),
                    )
                    for candidate_name, direct_name in zip(
                        args.additional_candidate,
                        args.additional_direct_candidate,
                        strict=True,
                    )
                ),
            )
        )
        return 0
    if args.command == "summarize-stage1-replay-factorial":
        print(
            summarize_stage1_replay_factorial(
                _candidate_from_name(args.no_replay_identity_candidate),
                _candidate_from_name(args.no_replay_augmix_candidate),
                _candidate_from_name(args.replay_identity_candidate),
                _candidate_from_name(args.replay_augmix_candidate),
            )
        )
        return 0
    if args.command == "summarize-stage1-scaling":
        print(
            summarize_stage1_scaling_pair(
                _candidate_from_name(args.identity_candidate),
                _candidate_from_name(args.augmix_candidate),
                direct=_candidate_from_name(args.direct_candidate),
                reference_augmix=_candidate_from_name(
                    args.reference_augmix_candidate
                ),
                benchmark=(
                    None
                    if args.benchmark_candidate == "none"
                    else _candidate_from_name(args.benchmark_candidate)
                ),
            )
        )
        return 0
    candidate = _candidate_from_name(args.candidate)
    if args.command == "prepare":
        print(prepare(candidate, force=args.force))
    elif args.command == "run":
        run_wave(
            candidate,
            arm=args.arm,
            stage=args.stage,
            gpus=args.gpus,
            cpu_threads=args.cpu_threads,
            jobs_per_gpu=args.jobs_per_gpu,
        )
    elif args.command == "summarize":
        direct_candidate = (
            None
            if args.direct_candidate is None
            else _candidate_from_name(args.direct_candidate)
        )
        print(
            summarize(
                candidate,
                direct_candidate=direct_candidate,
                direct_arm=args.direct_arm,
            )
        )
    elif args.command == "blend":
        print(
            blend_checkpoints(
                candidate,
                direct_candidate=_candidate_from_name(args.direct_candidate),
                method_candidate=_candidate_from_name(args.method_candidate),
            )
        )
    elif args.command == "clean-failed":
        print(clean_failed(candidate, arm=args.arm, center=args.center))
    elif args.command == "source-floor":
        print(
            run_source_floor(
                candidate,
                arm=args.arm,
                gpus=args.gpus,
                cpu_threads=args.cpu_threads,
            )
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
