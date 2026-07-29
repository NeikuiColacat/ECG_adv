from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SANDBOX_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT))

import a7_global_search as controller  # noqa: E402
import core.online_trainer as online_trainer  # noqa: E402
import objective_scale_adapter as objective_scale  # noqa: E402
import pcgrad_runtime as pcgrad  # noqa: E402
import rotating4_runtime as rotating4  # noqa: E402
import soft_teacher_runtime as soft_teacher  # noqa: E402
import source_replay_runtime as source_replay  # noqa: E402
import unlabeled_teacher_runtime as unlabeled_teacher  # noqa: E402
from core.methods.contracts import Provenance, ViewBundle, WaveformView  # noqa: E402
from core.methods.registry import compile_method_profile  # noqa: E402
from models.contracts import EFFICIENTNET1DV2_SPEC  # noqa: E402


STAGE1_IDS = (
    "s00_effnet_ssl4096_identity_e40",
    "s01_effnet_ssl4096_augmix_e40",
)
SHORT_STAGE1_IDS = (
    "s10_effnet_ssl4096_identity_e20",
    "s11_effnet_ssl4096_augmix_e20",
)
EXTENDED_STAGE1_IDS = (
    "s12_effnet_ssl8192_identity_e20",
    "s13_effnet_ssl8192_augmix_e20",
)
EXTENDED_STAGE1_SHARED_DIRECT_ID = "s14_effnet_ssl8192_shared_direct_e20"
CLASSIC_ALPHA_STAGE1_ID = "s15_effnet_ssl4096_augmix_alpha100_e20"
REPLAY_STAGE1_IDS = (
    "r00_effnet_ssl4096_identity_replay030_e40",
    "r01_effnet_ssl4096_augmix_replay030_e40",
)
STAGE2_IDS = (
    "s02_effnet_identity_vae_refine8",
    "s03_effnet_augmix_vae_refine8",
)
STRONG_STAGE2_IDS = (
    "g1i_effnet_identity_vae_eps8_refine8",
    "g1a_effnet_augmix_vae_eps8_refine8",
)
CLEAN_POLISH_IDS = (
    "h2i_effnet_identity_vae_clean_polish4",
    "h2a_effnet_augmix_vae_clean_polish4",
)
CHAIN3_POLISH_IDS = ("j3a_effnet_augmix_vae_chain3_polish4",)
CHAIN3_TEACHER04_POLISH_IDS = (
    "j4a_effnet_augmix_vae_chain3_teacher04_polish4",
    "j4b_effnet_augmix_vae_chain3_teacher04_bnmatched_polish4",
)
TEACHER_DISTILL_POLISH_IDS = (
    "k1_effnet_augmix_vae_teacher_distill_polish4",
    "k2i_effnet_identity_vae_teacher_distill_lowdose4",
    "k2a_effnet_augmix_vae_teacher_distill_lowdose4",
)
HARD_BCE_POLISH_IDS = (
    "k3i_effnet_identity_vae_hard_bce_lowdose4",
    "k3a_effnet_augmix_vae_hard_bce_lowdose4",
    "k4i_effnet_e20_identity_vae_hard_bce_lowdose4",
    "k4a_effnet_e20_augmix_vae_hard_bce_lowdose4",
)
CONSISTENCY_POLISH_IDS = (
    "k5i_effnet_e20_identity_vae_consistency_lowdose4",
    "k5a_effnet_e20_augmix_vae_consistency_lowdose4",
)
LHAT_PATH_POLISH_IDS = (
    "l6i_effnet_e20_identity_vae_lhat_path8",
    "l6a_effnet_e20_augmix_vae_lhat_path8",
)
FEATURE_INVARIANCE_POLISH_IDS = (
    "l7i_effnet_e20_identity_vae_feature8",
    "l7a_effnet_e20_augmix_vae_feature8",
)
LOCAL_ANCHOR_SOFT_POLISH_IDS = (
    "l8ri_effnet_e20_identity_vae_local_anchor_soft8",
    "l8ra_effnet_e20_augmix_vae_local_anchor_soft8",
)
CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_IDS = (
    "l9ri_effnet_e20_identity_vae_calibrated_anchor_soft8",
    "l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8",
)
BOUNDED_AUGMAX_POLISH_IDS = (
    "l10ri_effnet_e20_identity_vae_bounded_augmax6",
    "l10ra_effnet_e20_augmix_vae_bounded_augmax6",
)
MILD_CHAIN3_FACTORIAL_IDS = (
    "l11c_effnet_ssl4096_identity_vae_mild_chain3_e20",
    "l11d_effnet_ssl4096_augmix_vae_mild_chain3_e20",
)
SOURCE_REPLAY_MILD_CHAIN3_ID = (
    "l12d_effnet_ssl4096_augmix_replay030_vae_mild_chain3_e20"
)
GRADIENT_BALANCED_MILD_CHAIN3_ID = (
    "l13d_effnet_ssl4096_augmix_replay030_"
    "vae_gradbalanced_chain3_e20"
)
DIVERSE_VAE_POSTTRAIN_ID = (
    "l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20"
)
SOURCE_STABILIZED_DIVERSE_VAE_POSTTRAIN_ID = (
    "l15d_effnet_ssl4096_augmix_replay030_"
    "stage2replay010_vae_diverse_d19_e20"
)
INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID = (
    "l16d_effnet_ssl4096_augmix_replay030_"
    "vae_diverse_d19_e20_seed1"
)
FULLRNG_INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID = (
    "l17d_effnet_ssl4096_augmix_replay030_"
    "vae_diverse_d19_e20_seed1_fullrng"
)
INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID = (
    "l19d_effnet_ssl4096_augmix_replay030_"
    "vae_gradbalanced_chain3_e20_seed1"
)
SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID = (
    "l20d_effnet_ssl4096_augmix_replay030_"
    "vae_gradbalanced_chain3_e20_seed2"
)
COMPUTE_MATCHED_UNIFORM_CHAIN3_ID = (
    "l21d_effnet_ssl4096_augmix_replay030_"
    "vae_uniform_chain3_e20_seed2"
)
SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_REPLAY_ID = (
    "l22d_effnet_ssl4096_augmix_replay030_"
    "vae_gradbalanced_chain3_e20_seed2_replay"
)
DIRECT_LHAT_UNIFORM_ID = (
    "l23d_effnet_ssl4096_augmix_replay030_vae_direct_uniform_e40"
)
DIRECT_LHAT_LOCAL_HARD_ID = (
    "l24d_effnet_ssl4096_augmix_replay030_vae_direct_localhard_e40"
)
DIRECT_LHAT_LOCAL_UNIFORM_ID = (
    "l25d_effnet_ssl4096_augmix_replay030_vae_direct_localuniform_e40"
)
DIRECT_LHAT_BALANCED_HARD_ID = (
    "l26d_effnet_ssl4096_augmix_replay030_vae_direct_balancedhard_e40"
)
DIRECT_LHAT_VAT_HARD_ID = (
    "l27d_effnet_ssl4096_augmix_replay030_vae_direct_vathard_e40"
)
ECGFOUNDER_PRUNED_NO_VAE_ID = (
    "l28d_ecgfounder_ssl1024_augmix_replay030_no_vae_e40"
)
ECGFOUNDER_PRUNED_VAE_HARD_ID = (
    "l29d_ecgfounder_ssl1024_augmix_replay030_vae_hard_e40"
)
ECGFOUNDER_CLEAN_ONLY_ID = (
    "l30d_ecgfounder_ssl1024_augmix_replay030_clean_only_e40"
)
ECGFOUNDER_CLEAN_VAE_ID = (
    "l31d_ecgfounder_ssl1024_augmix_replay030_clean_vae_e40"
)
ECGFOUNDER_CLEAN_VAE_A050_ID = (
    "l32d_ecgfounder_ssl1024_augmix_replay030_clean_vae_a050_e40"
)
ECGFOUNDER_CLEAN_VAE_A100_ID = (
    "l33d_ecgfounder_ssl1024_augmix_replay030_clean_vae_a100_e40"
)
ECGFOUNDER_ROTATING4_NO_VAE_ID = (
    "l34d_ecgfounder_ssl1024_augmix_replay030_rot4_no_vae_e40"
)
ECGFOUNDER_ROTATING4_VAE_ID = (
    "l35d_ecgfounder_ssl1024_augmix_replay030_rot4_vae_hard_e40"
)
EFFNET_ROTATING4_NO_VAE_ID = (
    "l36d_effnet_ssl1024_augmix_replay030_rot4_no_vae_h30_e23"
)
EFFNET_ROTATING4_VAE_ID = (
    "l37d_effnet_ssl1024_augmix_replay030_rot4_vae_hard_h30_e23"
)
ECGFOUNDER_DIVERSE_VAE_POSTTRAIN_ID = (
    "l18d_ecgfounder_ssl4096_augmix_replay030_"
    "vae_diverse_d19_e20"
)


def _normalize_stage1_explanation_fields(method: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(method)
    normalized["method"]["scientific_arm"] = "<stage1-view-arm>"
    normalized["method"]["description"] = "<candidate-description>"
    teacher = normalized["contracts"]["unlabeled_teacher"]
    teacher["pretrain_view_mode"] = "<pretrain-view>"
    teacher["policy"] = "<pretrain-policy>"
    return normalized


def _normalize_j4_clone_payload(value: Any, candidate_id: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<candidate-description>"
                if key == "description"
                else _normalize_j4_clone_payload(item, candidate_id)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_j4_clone_payload(item, candidate_id)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _normalize_j4_clone_payload(item, candidate_id)
            for item in value
        )
    if isinstance(value, str):
        return value.replace(candidate_id, "<candidate-id>")
    return value


def test_sequential_stage1_pair_differs_only_by_view_policy_and_description() -> None:
    identity = controller._method_for(
        controller.CANDIDATES[STAGE1_IDS[0]], "ningbo"
    )
    augmix = controller._method_for(
        controller.CANDIDATES[STAGE1_IDS[1]], "ningbo"
    )

    assert _normalize_stage1_explanation_fields(identity) == (
        _normalize_stage1_explanation_fields(augmix)
    )
    assert identity["method"]["scientific_arm"] != (
        augmix["method"]["scientific_arm"]
    )
    assert identity["method"]["description"] != augmix["method"]["description"]

    expected_views = {
        STAGE1_IDS[0]: (
            "clean_identity",
            "clean_identity_simclr_vicreg_control_v1",
        ),
        STAGE1_IDS[1]: (
            "twochain_augmix",
            "ema_teacher_clean_to_twochain_augmix_strong_v1",
        ),
    }
    for candidate_id, method in zip(STAGE1_IDS, (identity, augmix), strict=True):
        teacher = method["contracts"]["unlabeled_teacher"]
        assert (
            teacher["pretrain_view_mode"],
            teacher["policy"],
        ) == expected_views[candidate_id]
        assert teacher["pretrain_source_replay_weight"] == 0.0
        assert teacher["labels_consumed"] is False
        assert "full_k500_heldout_oracle" not in method["contracts"]
        assert method["contracts"]["heldout_target_feedback_allowed"] is False
        assert not {"lhat", "threechain_augmix"}.intersection(method["nodes"])
        assert not {
            "vae",
            "latent_pool",
            "vae_decoder",
            "lhat_config",
            "augmix_config",
        }.intersection(method["resources"])


@pytest.mark.parametrize(
    ("candidate_id", "expected_view"),
    (
        (MILD_CHAIN3_FACTORIAL_IDS[0], "clean_identity"),
        (MILD_CHAIN3_FACTORIAL_IDS[1], "twochain_augmix"),
    ),
)
def test_l11_mild_chain3_and_direct_control_share_frozen_stage1(
    candidate_id: str,
    expected_view: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    candidate_path = tmp_path / f"{candidate_id}_candidate.yaml"
    control_path = tmp_path / f"{candidate_id}_control.yaml"
    candidate_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    control_path.write_text(
        yaml.safe_dump(control, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(candidate_path)
    compiled_control = compile_method_profile(control_path)

    assert compiled.profile_name == "fixed20_pcgrad_strong_lhat_augmix_v1"
    assert compiled_control.profile_name == "direct_depth23_fixed20"
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(0.25)
    assert candidate.pretrain_steps == 4096
    assert candidate.pretrain_epochs == 1
    assert candidate.epochs == 20
    assert candidate.scheduler_horizon_epochs == 20
    assert candidate.pretrain_ssl_objective == "simclr_vicreg"
    assert candidate.pretrain_vicreg_mix_weight == pytest.approx(0.03)
    assert candidate.pretrain_source_replay_weight == pytest.approx(0.0)

    candidate_teacher = unlabeled_teacher._teacher_config(compiled)
    control_teacher = unlabeled_teacher._teacher_config(compiled_control)
    assert candidate_teacher is not None
    assert control_teacher is not None
    for teacher in (candidate_teacher, control_teacher):
        assert teacher.pretrain_view_mode == expected_view
        assert teacher.pretrain_steps_per_epoch == 4096
        assert teacher.pretrain_source_replay_weight == pytest.approx(0.0)
        assert teacher.pretrain_ssl_objective == "simclr_vicreg"
        assert teacher.pretrain_vicreg_mix_weight == pytest.approx(0.03)

    assert "full_k500_heldout_oracle" not in method["contracts"]
    assert "full_k500_heldout_oracle" not in control["contracts"]
    assert method["contracts"]["lhat_label_policy"] == (
        "exact_positive_set_nonself"
    )
    assert method["contracts"]["lhat_hull_lambda"] == pytest.approx(0.60)
    assert method["contracts"]["lhat_steps"] == 5
    assert method["contracts"]["lhat_pgd_epsilon_l2_standardized"] == (
        pytest.approx(8.0)
    )
    assert method["contracts"]["lhat_config_path"] == (
        "train/lhat_h060_s5_eps8.yaml"
    )
    assert candidate.lhat_pgd_epsilon_l2_standardized == pytest.approx(8.0)
    assert method["contracts"]["pcgrad_auxiliary_alpha"] == pytest.approx(0.25)
    assert method["contracts"]["pcgrad_auxiliary_terms"] == [
        "augmix_bce",
        "clean_lhat_augmix_jsd",
    ]
    assert control["contracts"]["matched_candidate_method_id"] == (
        "fixed20_pcgrad_strong_lhat_augmix_v1"
    )
    assert {"lhat", "threechain_augmix"}.issubset(method["nodes"])
    assert not {"lhat", "threechain_augmix"}.intersection(control["nodes"])

    candidate_train, _ = controller._candidate_experiments(
        candidate,
        "ningbo",
    )
    control_train, _ = controller._direct_experiments(candidate, "ningbo")
    candidate_args = candidate_train["entrypoint"]["arguments"]
    control_args = control_train["entrypoint"]["arguments"]
    assert candidate_args.count("--vae-checkpoint") == 1
    assert "--vae-checkpoint" not in control_args


def test_l12_source_replay_is_the_only_l11d_recipe_change(
    tmp_path: Path,
) -> None:
    reference = controller.CANDIDATES[MILD_CHAIN3_FACTORIAL_IDS[1]]
    candidate = controller.CANDIDATES[SOURCE_REPLAY_MILD_CHAIN3_ID]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == MILD_CHAIN3_FACTORIAL_IDS[1]
    assert candidate_fields.pop("candidate_id") == SOURCE_REPLAY_MILD_CHAIN3_ID
    assert reference_fields.pop("pretrain_source_replay_weight") == pytest.approx(
        0.0
    )
    assert candidate_fields.pop("pretrain_source_replay_weight") == pytest.approx(
        0.30
    )
    assert reference_fields.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_e20_v1"
    )
    assert candidate_fields.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_replay030_e20_v1"
    )
    assert candidate_fields == reference_fields

    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    candidate_path = tmp_path / "l12_candidate.yaml"
    control_path = tmp_path / "l12_control.yaml"
    candidate_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    control_path.write_text(
        yaml.safe_dump(control, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(candidate_path)
    compiled_control = compile_method_profile(control_path)
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(0.25)
    candidate_teacher = unlabeled_teacher._teacher_config(compiled)
    control_teacher = unlabeled_teacher._teacher_config(compiled_control)
    assert candidate_teacher is not None
    assert control_teacher is not None
    for teacher in (candidate_teacher, control_teacher):
        assert teacher.pretrain_view_mode == "twochain_augmix"
        assert teacher.pretrain_source_replay_weight == pytest.approx(0.30)
        assert teacher.pretrain_source_batches_per_step == 1
    assert method["contracts"]["heldout_target_feedback_allowed"] is False
    assert control["contracts"]["heldout_target_feedback_allowed"] is False
    assert "full_k500_heldout_oracle" not in method["contracts"]
    assert "full_k500_heldout_oracle" not in control["contracts"]


def test_l13_changes_only_l12_via_predeclared_gradient_budget(
    tmp_path: Path,
) -> None:
    reference = controller.CANDIDATES[SOURCE_REPLAY_MILD_CHAIN3_ID]
    candidate = controller.CANDIDATES[GRADIENT_BALANCED_MILD_CHAIN3_ID]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == SOURCE_REPLAY_MILD_CHAIN3_ID
    assert candidate_fields.pop("candidate_id") == (
        GRADIENT_BALANCED_MILD_CHAIN3_ID
    )
    assert reference_fields.pop("auxiliary_alpha") == pytest.approx(0.25)
    assert candidate_fields.pop("auxiliary_alpha") == pytest.approx(0.75)
    assert reference_fields.pop("pcgrad_auxiliary_norm_ratio_cap") is None
    assert candidate_fields.pop("pcgrad_auxiliary_norm_ratio_cap") == (
        pytest.approx(1.0)
    )
    assert candidate_fields == reference_fields

    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    candidate_path = tmp_path / "l13_candidate.yaml"
    control_path = tmp_path / "l13_control.yaml"
    candidate_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    control_path.write_text(
        yaml.safe_dump(control, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(candidate_path)
    compiled_control = compile_method_profile(control_path)

    assert compiled.profile_name == (
        "fixed20_pcgrad_gradbalanced_strong_lhat_augmix_v1"
    )
    assert compiled_control.profile_name == "direct_depth23_fixed20"
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(0.75)
    assert compiled.contracts["pcgrad_auxiliary_norm_ratio_cap"] == (
        pytest.approx(1.0)
    )
    assert compiled.contracts["pcgrad_auxiliary_norm_cap_policy"] == (
        pcgrad.NORM_CAPPED_PROJECTION_POLICY
    )
    assert compiled_control.contracts["matched_candidate_method_id"] == (
        compiled.profile_name
    )
    for compiled_method in (compiled, compiled_control):
        teacher = unlabeled_teacher._teacher_config(compiled_method)
        assert teacher is not None
        assert teacher.pretrain_view_mode == "twochain_augmix"
        assert teacher.pretrain_source_replay_weight == pytest.approx(0.30)
        assert teacher.pretrain_source_batches_per_step == 1
        assert compiled_method.contracts["heldout_target_feedback_allowed"] is False


def test_l14_restores_only_frozen_d19_diverse_vae_posttrain(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    assert candidate.model_name == "efficientnet1dv2"
    assert candidate.epochs == 20
    assert candidate.scheduler_horizon_epochs == 20
    assert candidate.pretrain_steps == 4096
    assert candidate.pretrain_view_mode == "twochain_augmix"
    assert candidate.pretrain_source_replay_weight == pytest.approx(0.30)
    assert candidate.method_family == "d19_separate_views"
    assert candidate.auxiliary_alpha == pytest.approx(1.5)
    assert (
        candidate.raw_auxiliary_scale,
        candidate.vae_random_auxiliary_scale,
        candidate.vae_hard_auxiliary_scale,
        candidate.jsd_auxiliary_scale,
    ) == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert candidate.pcgrad_auxiliary_norm_ratio_cap is None

    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    candidate_path = tmp_path / "l14_candidate.yaml"
    control_path = tmp_path / "l14_control.yaml"
    candidate_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    control_path.write_text(
        yaml.safe_dump(control, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(candidate_path)
    compiled_control = compile_method_profile(control_path)

    assert compiled.profile_name == pcgrad.SEARCH_METHOD_ID
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(1.5)
    assert compiled.contracts["pcgrad_auxiliary_terms"] == list(
        pcgrad.D19_AUXILIARY_TERMS
    )
    assert compiled.contracts["exposure_policy"] == (
        pcgrad.AUXILIARY_EXPOSURE_POLICY
    )
    assert compiled.contracts["group_dro_step_size"] == pytest.approx(0.0)
    assert compiled.contracts["feature_regularizer_weight"] == pytest.approx(0.0)
    assert compiled.contracts["pairrank_weight"] == pytest.approx(0.0)
    assert compiled.contracts["candidate_replay_count"] == 0
    assert compiled_control.profile_name == "direct_depth23_fixed20"
    assert compiled_control.contracts["matched_candidate_method_id"] == (
        pcgrad.SEARCH_METHOD_ID
    )
    for compiled_method in (compiled, compiled_control):
        teacher = unlabeled_teacher._teacher_config(compiled_method)
        assert teacher is not None
        assert teacher.pretrain_view_mode == "twochain_augmix"
        assert teacher.pretrain_source_replay_weight == pytest.approx(0.30)
        assert teacher.pretrain_source_batches_per_step == 1
        assert compiled_method.contracts["heldout_target_feedback_allowed"] is False


def test_current_rng_vicreg_control_differs_from_r1_only_by_ssl_objective() -> None:
    control = controller.CANDIDATES["p22m0_current_rng_vicreg"]
    pruning = controller.CANDIDATES["p22r1_simclr_no_vicreg"]
    control_fields = control.__dict__.copy()
    pruning_fields = pruning.__dict__.copy()
    assert control_fields.pop("candidate_id") == "p22m0_current_rng_vicreg"
    assert pruning_fields.pop("candidate_id") == "p22r1_simclr_no_vicreg"
    assert control_fields.pop("pretrain_ssl_objective") == "simclr_vicreg"
    assert pruning_fields.pop("pretrain_ssl_objective") == "simclr"
    assert control_fields.pop("pretrain_vicreg_mix_weight") == pytest.approx(0.03)
    assert pruning_fields.pop("pretrain_vicreg_mix_weight") == pytest.approx(0.0)
    assert control_fields == pruning_fields


@pytest.mark.parametrize(
    ("candidate_id", "objective", "expected_scales"),
    (
        (
            "p22r1_simclr_no_vicreg",
            "simclr",
            None,
        ),
        (
            "p22r2_simclr_no_raw_bce",
            "simclr",
            {
                "corruption_1_bce": 0.0,
                "compat_random_bce": 5.0 / 3.0,
                "compat_hard_bce": 5.0 / 3.0,
                "clean_raw_random_hard_jsd": 1.0,
            },
        ),
        (
            "p22r3_simclr_hard_only_bce_screen",
            "simclr",
            {
                "corruption_1_bce": 0.0,
                "compat_random_bce": 0.0,
                "compat_hard_bce": 10.0 / 3.0,
                "clean_raw_random_hard_jsd": 1.0,
            },
        ),
    ),
)
def test_p22_pruning_screens_preserve_d19_bce_mass(
    candidate_id: str,
    objective: str,
    expected_scales: dict[str, float] | None,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    assert candidate.pretrain_steps == 4096
    assert candidate.pretrain_ssl_objective == objective
    assert candidate.pretrain_vicreg_mix_weight == pytest.approx(0.0)

    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)

    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(1.5)
    if expected_scales is None:
        assert "pcgrad_auxiliary_term_scales" not in compiled.contracts
        return
    assert compiled.contracts["pcgrad_auxiliary_bce_mass_preserved"] is True
    assert compiled.contracts["pcgrad_auxiliary_term_scales"] == pytest.approx(
        expected_scales
    )
    original_mass = sum(pcgrad.D19_AUXILIARY_BCE_WEIGHTS.values())
    scaled_mass = sum(
        weight * expected_scales[name]
        for name, weight in pcgrad.D19_AUXILIARY_BCE_WEIGHTS.items()
    )
    assert scaled_mass == pytest.approx(original_mass)


def test_l18_transfers_l14_to_ecgfounder_without_method_drift() -> None:
    reference = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    candidate = controller.CANDIDATES[ECGFOUNDER_DIVERSE_VAE_POSTTRAIN_ID]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == DIVERSE_VAE_POSTTRAIN_ID
    assert candidate_fields.pop("candidate_id") == (
        ECGFOUNDER_DIVERSE_VAE_POSTTRAIN_ID
    )
    assert reference_fields.pop("model_name") == "efficientnet1dv2"
    assert candidate_fields.pop("model_name") == "ecgfounder"
    assert reference_fields.pop("learning_rate") == pytest.approx(1.5e-4)
    assert candidate_fields.pop("learning_rate") == pytest.approx(2.0e-5)
    assert reference_fields.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_replay030_d19_e20_v1"
    )
    assert candidate_fields.pop("comparison_rng_identity") == (
        "sequential_ecgfounder_stage1_replay030_d19_e20_v1"
    )
    assert candidate_fields == reference_fields

    train, evaluate = controller._candidate_experiments(candidate, "ningbo")
    train_arguments = [str(value) for value in train["entrypoint"]["arguments"]]
    eval_arguments = [str(value) for value in evaluate["entrypoint"]["arguments"]]
    assert train_arguments[train_arguments.index("--model") + 1] == "ecgfounder"
    assert train_arguments[train_arguments.index("--source-checkpoint") + 1] == (
        str(controller.SOURCE_CHECKPOINTS["ecgfounder"])
    )
    assert train_arguments[train_arguments.index("--learning-rate") + 1] == "2e-05"
    assert eval_arguments[eval_arguments.index("--batch-size") + 1] == "128"


def test_l15_adds_only_matched_stage2_source_replay_to_l14() -> None:
    reference = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    candidate = controller.CANDIDATES[
        SOURCE_STABILIZED_DIVERSE_VAE_POSTTRAIN_ID
    ]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == DIVERSE_VAE_POSTTRAIN_ID
    assert candidate_fields.pop("candidate_id") == (
        SOURCE_STABILIZED_DIVERSE_VAE_POSTTRAIN_ID
    )
    assert reference_fields.pop("supervised_source_replay_weight") is None
    assert candidate_fields.pop(
        "supervised_source_replay_weight"
    ) == pytest.approx(0.10)
    assert candidate_fields == reference_fields

    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    assert method["method"]["id"] == pcgrad.SEARCH_METHOD_ID
    assert control["method"]["id"] == "direct_depth23_fixed20"
    assert method["method"]["id"] in source_replay.ALLOWED_METHOD_IDS
    assert control["method"]["id"] in source_replay.ALLOWED_METHOD_IDS

    online = controller._online_for(candidate, "ningbo")
    direct_online = controller._direct_training_config_for(
        candidate,
        "ningbo",
    )
    expected_profile = "pn2021_fixed20_source_replay_w010_sandbox"
    assert online["profile_name"] == expected_profile
    assert direct_online["profile_name"] == expected_profile
    assert source_replay.PROFILE_TOTAL_WEIGHTS[expected_profile] == (
        pytest.approx(0.10)
    )


def test_l16_is_an_exact_l14_replay_with_only_replicate_id_changed() -> None:
    reference = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    candidate = controller.CANDIDATES[INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == DIVERSE_VAE_POSTTRAIN_ID
    assert candidate_fields.pop("candidate_id") == (
        INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID
    )
    assert reference_fields.pop("replicate_id") == 0
    assert candidate_fields.pop("replicate_id") == 1
    assert candidate_fields == reference_fields

    reference_online = controller._online_for(reference, "ningbo")
    candidate_online = controller._online_for(candidate, "ningbo")
    candidate_direct_online = controller._direct_training_config_for(
        candidate,
        "ningbo",
    )
    assert reference_online["random_seed"]["replicate_id"] == 0
    assert candidate_online["random_seed"]["replicate_id"] == 1
    assert candidate_direct_online["random_seed"] == candidate_online["random_seed"]


def test_l17_is_an_exact_l14_replay_with_full_rng_replicate() -> None:
    reference = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    candidate = controller.CANDIDATES[
        FULLRNG_INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID
    ]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    reference_fields.pop("candidate_id")
    candidate_fields.pop("candidate_id")
    assert reference_fields.pop("replicate_id") == 0
    assert candidate_fields.pop("replicate_id") == 1
    assert candidate_fields == reference_fields


def test_l19_is_an_exact_l13_replay_with_full_rng_replicate() -> None:
    reference = controller.CANDIDATES[GRADIENT_BALANCED_MILD_CHAIN3_ID]
    candidate = controller.CANDIDATES[
        INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    assert reference_fields.pop("candidate_id") == (
        GRADIENT_BALANCED_MILD_CHAIN3_ID
    )
    assert candidate_fields.pop("candidate_id") == (
        INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    )
    assert reference_fields.pop("replicate_id") == 0
    assert candidate_fields.pop("replicate_id") == 1
    assert candidate_fields == reference_fields

    candidate_online = controller._online_for(candidate, "ningbo")
    candidate_direct_online = controller._direct_training_config_for(
        candidate,
        "ningbo",
    )
    assert candidate_online["random_seed"]["replicate_id"] == 1
    assert candidate_direct_online["random_seed"] == (
        candidate_online["random_seed"]
    )


def test_l20_is_an_exact_l13_replay_with_second_full_rng_replicate() -> None:
    reference = controller.CANDIDATES[GRADIENT_BALANCED_MILD_CHAIN3_ID]
    candidate = controller.CANDIDATES[
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    reference_fields.pop("candidate_id")
    candidate_fields.pop("candidate_id")
    assert reference_fields.pop("replicate_id") == 0
    assert candidate_fields.pop("replicate_id") == 2
    assert candidate_fields == reference_fields

    candidate_online = controller._online_for(candidate, "ningbo")
    candidate_direct_online = controller._direct_training_config_for(
        candidate,
        "ningbo",
    )
    assert candidate_online["random_seed"]["replicate_id"] == 2
    assert candidate_direct_online["random_seed"] == (
        candidate_online["random_seed"]
    )


def test_l21_changes_only_lhat_search_to_compute_matched_uniform_control() -> None:
    hard = controller.CANDIDATES[
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]
    control = controller.CANDIDATES[COMPUTE_MATCHED_UNIFORM_CHAIN3_ID]
    hard_fields = hard.__dict__.copy()
    control_fields = control.__dict__.copy()
    hard_fields.pop("candidate_id")
    control_fields.pop("candidate_id")
    assert hard_fields.pop("lhat_search_mode") == "optimized_softmax"
    assert control_fields.pop("lhat_search_mode") == (
        "uniform_compute_matched"
    )
    assert control_fields == hard_fields

    lhat = controller._lhat_config_for(control)
    assert lhat["hull_attack"]["hull_lambda"] == pytest.approx(0.60)
    assert lhat["hull_attack"]["steps"] == 5
    assert lhat["hull_attack"]["pgd_epsilon_l2_standardized"] == pytest.approx(
        8.0
    )
    assert lhat["hull_attack"]["weight_mode"] == "uniform_compute_matched"
    assert lhat["hull_attack"]["learning_rate"] == pytest.approx(1.0e-30)


def test_l22_replays_l20_without_changing_the_scientific_recipe() -> None:
    failed_attempt = controller.CANDIDATES[
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]
    replay = controller.CANDIDATES[
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_REPLAY_ID
    ]
    failed_fields = failed_attempt.__dict__.copy()
    replay_fields = replay.__dict__.copy()
    failed_fields.pop("candidate_id")
    replay_fields.pop("candidate_id")
    assert replay_fields == failed_fields
    assert replay.replicate_id == 2

    replay_online = controller._online_for(replay, "ningbo")
    replay_direct_online = controller._direct_training_config_for(
        replay,
        "ningbo",
    )
    assert replay_online["random_seed"]["replicate_id"] == 2
    assert replay_direct_online["random_seed"] == replay_online["random_seed"]


def test_l23_is_compute_matched_uniform_control_for_direct_lhat() -> None:
    hard = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    control = controller.CANDIDATES[DIRECT_LHAT_UNIFORM_ID]
    hard_fields = hard.__dict__.copy()
    control_fields = control.__dict__.copy()
    hard_fields.pop("candidate_id")
    control_fields.pop("candidate_id")
    assert hard_fields.pop("lhat_search_mode") == "optimized_softmax"
    assert control_fields.pop("lhat_search_mode") == "uniform_compute_matched"
    assert control_fields == hard_fields

    lhat = controller._lhat_config_for(control)
    assert lhat["hull_attack"]["hull_lambda"] == pytest.approx(0.60)
    assert lhat["hull_attack"]["steps"] == 5
    assert lhat["hull_attack"]["pgd_epsilon_l2_standardized"] == pytest.approx(
        2.0
    )
    assert lhat["hull_attack"]["weight_mode"] == "uniform_compute_matched"
    assert lhat["hull_attack"]["learning_rate"] == pytest.approx(1.0e-30)


def test_l24_l25_change_only_frozen_local_geometry_and_search_mode() -> None:
    reference = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    hard = controller.CANDIDATES[DIRECT_LHAT_LOCAL_HARD_ID]
    uniform = controller.CANDIDATES[DIRECT_LHAT_LOCAL_UNIFORM_ID]
    reference_fields = reference.__dict__.copy()
    hard_fields = hard.__dict__.copy()
    uniform_fields = uniform.__dict__.copy()
    for fields in (reference_fields, hard_fields, uniform_fields):
        fields.pop("candidate_id")
    assert reference_fields.pop("lhat_geometry_preset") is None
    assert hard_fields.pop("lhat_geometry_preset") == "lowdose_h010_s3_v1"
    assert uniform_fields.pop("lhat_geometry_preset") == "lowdose_h010_s3_v1"
    assert reference_fields.pop("lhat_search_mode") == "optimized_softmax"
    assert hard_fields.pop("lhat_search_mode") == "optimized_softmax"
    assert uniform_fields.pop("lhat_search_mode") == "uniform_compute_matched"
    assert hard_fields == reference_fields
    assert uniform_fields == reference_fields

    hard_lhat = controller._lhat_config_for(hard)
    uniform_lhat = controller._lhat_config_for(uniform)
    for lhat in (hard_lhat, uniform_lhat):
        attack = lhat["hull_attack"]
        assert attack["hull_lambda"] == pytest.approx(0.10)
        assert attack["steps"] == 3
        assert attack["pgd_epsilon_l2_standardized"] == pytest.approx(2.0)
    assert hard_lhat["hull_attack"]["weight_mode"] == "optimized_softmax"
    assert uniform_lhat["hull_attack"]["weight_mode"] == (
        "uniform_compute_matched"
    )


def test_l26_l27_change_only_the_effective_lhat_attack_objective() -> None:
    reference = controller.CANDIDATES[DIRECT_LHAT_LOCAL_HARD_ID]
    balanced = controller.CANDIDATES[DIRECT_LHAT_BALANCED_HARD_ID]
    vat = controller.CANDIDATES[DIRECT_LHAT_VAT_HARD_ID]
    reference_fields = reference.__dict__.copy()
    balanced_fields = balanced.__dict__.copy()
    vat_fields = vat.__dict__.copy()
    for fields in (reference_fields, balanced_fields, vat_fields):
        fields.pop("candidate_id")
    assert (
        reference_fields.pop("lhat_attack_objective")
        == "maximize_multilabel_bce_with_logits"
    )
    assert (
        balanced_fields.pop("lhat_attack_objective")
        == "maximize_equal_positive_negative_bce_with_logits"
    )
    assert (
        vat_fields.pop("lhat_attack_objective")
        == "maximize_bernoulli_kl_from_decoded_anchor"
    )
    assert balanced_fields == reference_fields
    assert vat_fields == reference_fields

    balanced_lhat = controller._lhat_config_for(balanced)
    vat_lhat = controller._lhat_config_for(vat)
    assert balanced_lhat["hull_attack"]["objective"] == (
        "maximize_equal_positive_negative_bce_with_logits"
    )
    assert vat_lhat["hull_attack"]["objective"] == (
        "maximize_bernoulli_kl_from_decoded_anchor"
    )
    for lhat in (balanced_lhat, vat_lhat):
        assert lhat["hull_attack"]["hull_lambda"] == pytest.approx(0.10)
        assert lhat["hull_attack"]["steps"] == 3
        assert lhat["hull_attack"]["pgd_epsilon_l2_standardized"] == (
            pytest.approx(2.0)
        )


def test_l28_l29_are_a_matched_ecgfounder_vae_knockout_pair(
    tmp_path: Path,
) -> None:
    no_vae = controller.CANDIDATES[ECGFOUNDER_PRUNED_NO_VAE_ID]
    vae_hard = controller.CANDIDATES[ECGFOUNDER_PRUNED_VAE_HARD_ID]
    no_vae_fields = no_vae.__dict__.copy()
    vae_hard_fields = vae_hard.__dict__.copy()
    no_vae_fields.pop("candidate_id")
    vae_hard_fields.pop("candidate_id")
    assert no_vae_fields.pop("method_family") == "direct_target_ssl_control"
    assert vae_hard_fields.pop("method_family") == "vae_lhat_post_refine"
    assert no_vae_fields == vae_hard_fields

    for candidate, expects_lhat in (
        (no_vae, False),
        (vae_hard, True),
    ):
        assert candidate.model_name == "ecgfounder"
        assert candidate.learning_rate == pytest.approx(2.0e-5)
        assert candidate.pretrain_steps == 1024
        assert candidate.pretrain_view_mode == "twochain_augmix"
        assert candidate.pretrain_ssl_objective == "simclr"
        assert candidate.pretrain_vicreg_mix_weight == pytest.approx(0.0)
        assert candidate.pretrain_source_replay_weight == pytest.approx(0.30)
        assert candidate.epochs == 40
        method = controller._method_for(candidate, "ningbo")
        path = tmp_path / f"{candidate.candidate_id}.yaml"
        path.write_text(
            yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compiled = compile_method_profile(path)
        assert ("lhat" in method["nodes"]) is expects_lhat
        assert (
            compiled.profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
        ) is expects_lhat
        teacher = method["contracts"]["unlabeled_teacher"]
        assert teacher["pretrain_steps_per_epoch"] == 1024
        assert teacher["pretrain_source_replay_weight"] == pytest.approx(0.30)


def test_l30_l31_are_the_clean_only_fixed20_dependency_pair(
    tmp_path: Path,
) -> None:
    clean_only = controller.CANDIDATES[ECGFOUNDER_CLEAN_ONLY_ID]
    clean_vae = controller.CANDIDATES[ECGFOUNDER_CLEAN_VAE_ID]
    clean_fields = clean_only.__dict__.copy()
    vae_fields = clean_vae.__dict__.copy()
    clean_fields.pop("candidate_id")
    vae_fields.pop("candidate_id")
    assert clean_fields.pop("method_family") == "clean_target_ssl_control"
    assert vae_fields.pop("method_family") == "vae_lhat_clean_polish"
    assert clean_fields == vae_fields
    assert clean_only.auxiliary_alpha == pytest.approx(1.5)
    assert clean_vae.auxiliary_alpha == pytest.approx(1.5)

    fixed20_no_vae = controller.CANDIDATES[ECGFOUNDER_PRUNED_NO_VAE_ID]
    fixed20_vae = controller.CANDIDATES[ECGFOUNDER_PRUNED_VAE_HARD_ID]
    assert clean_only.comparison_rng_identity == (
        fixed20_no_vae.comparison_rng_identity
    )
    assert clean_vae.comparison_rng_identity == (
        fixed20_vae.comparison_rng_identity
    )

    clean_method = controller._method_for(clean_only, "ningbo")
    clean_vae_method = controller._method_for(clean_vae, "ningbo")
    matched_control = controller._direct_method_for(clean_vae, "ningbo")
    compiled_methods = {}
    for name, method in (
        ("clean", clean_method),
        ("clean_vae", clean_vae_method),
        ("matched_control", matched_control),
    ):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compiled = compile_method_profile(path)
        compiled_methods[name] = compiled
        assert compiled.profile_name in unlabeled_teacher.ALLOWED_METHOD_IDS
        teacher = unlabeled_teacher._teacher_config(compiled)
        assert teacher.pretrain_steps_per_epoch == 1024
        assert teacher.pretrain_view_mode == "twochain_augmix"
        assert teacher.pretrain_source_replay_weight == pytest.approx(0.30)
    assert (
        compiled_methods["clean_vae"].profile_name
        not in objective_scale.SCALE_ONLY_METHOD_IDS
    )

    assert clean_method["method"]["id"] == controller.CLEAN_TARGET_SSL_METHOD_ID
    assert "depth23_corruption" not in clean_method["nodes"]
    assert "lhat" not in clean_method["nodes"]
    assert clean_vae_method["method"]["id"] == (
        controller.VAE_CLEAN_DIRECTSUM_METHOD_ID
    )
    assert "depth23_corruption" not in clean_vae_method["nodes"]
    assert "lhat" in clean_vae_method["nodes"]
    clean_bce_terms = [
        term
        for term in clean_vae_method["objective"]["terms"]
        if term["id"] == "clean_bce"
    ]
    assert len(clean_bce_terms) == 1
    assert clean_bce_terms[0]["view"] == "clean_view"
    assert clean_bce_terms[0]["weight"] == pytest.approx(1.0)
    assert clean_vae_method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(1.0),
        "lhat_direct_bce": pytest.approx(0.75),
        "clean_lhat_jsd": pytest.approx(0.75),
    }
    assert clean_vae_method["contracts"]["direct_sum_auxiliary_alpha"] == (
        pytest.approx(1.5)
    )
    assert "objective_scale_adapter_mode" not in clean_vae_method["contracts"]
    assert "objective_global_scale" not in clean_vae_method["contracts"]
    assert matched_control["method"]["id"] == (
        controller.CLEAN_TARGET_SSL_METHOD_ID
    )

    clean_train, _ = controller._candidate_experiments(clean_only, "ningbo")
    vae_train, _ = controller._candidate_experiments(clean_vae, "ningbo")
    assert "--vae-checkpoint" not in clean_train["entrypoint"]["arguments"]
    assert "--vae-checkpoint" in vae_train["entrypoint"]["arguments"]


@pytest.mark.parametrize(
    ("candidate_id", "auxiliary_alpha"),
    [
        (ECGFOUNDER_CLEAN_VAE_A050_ID, 0.5),
        (ECGFOUNDER_CLEAN_VAE_A100_ID, 1.0),
    ],
)
def test_l32_l33_are_matched_clean_directsum_dose_candidates(
    tmp_path: Path,
    candidate_id: str,
    auxiliary_alpha: float,
) -> None:
    reference = controller.CANDIDATES[ECGFOUNDER_CLEAN_VAE_ID]
    candidate = controller.CANDIDATES[candidate_id]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    for field in ("candidate_id", "auxiliary_alpha"):
        reference_fields.pop(field)
        candidate_fields.pop(field)
    assert candidate_fields == reference_fields
    assert candidate.clean_direct_sum_auxiliary is True
    assert candidate.auxiliary_alpha == pytest.approx(auxiliary_alpha)

    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)
    assert compiled.profile_name == controller.VAE_CLEAN_DIRECTSUM_METHOD_ID
    assert compiled.profile_name in unlabeled_teacher.ALLOWED_METHOD_IDS
    assert compiled.profile_name not in objective_scale.SCALE_ONLY_METHOD_IDS
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(1.0),
        "lhat_direct_bce": pytest.approx(auxiliary_alpha / 2.0),
        "clean_lhat_jsd": pytest.approx(auxiliary_alpha / 2.0),
    }
    assert method["contracts"]["direct_sum_auxiliary_alpha"] == pytest.approx(
        auxiliary_alpha
    )
    assert "objective_scale_adapter_mode" not in method["contracts"]
    assert "objective_global_scale" not in method["contracts"]

    objective_weights = {
        term["id"]: term["weight"] for term in method["objective"]["terms"]
    }
    assert objective_weights == {
        "clean_bce": pytest.approx(1.0),
        "lhat_direct_bce": pytest.approx(auxiliary_alpha / 2.0),
        "clean_lhat_jsd": pytest.approx(auxiliary_alpha / 2.0),
    }
    matched_control = controller._direct_method_for(candidate, "ningbo")
    assert matched_control["method"]["id"] == (
        controller.CLEAN_TARGET_SSL_METHOD_ID
    )
    assert matched_control["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_CLEAN_DIRECTSUM_METHOD_ID
    )


@pytest.mark.parametrize(
    ("no_vae_id", "vae_id"),
    (
        (ECGFOUNDER_ROTATING4_NO_VAE_ID, ECGFOUNDER_ROTATING4_VAE_ID),
        (EFFNET_ROTATING4_NO_VAE_ID, EFFNET_ROTATING4_VAE_ID),
    ),
)
def test_rotating4_candidates_are_matched_pairs(
    tmp_path: Path,
    no_vae_id: str,
    vae_id: str,
) -> None:
    no_vae = controller.CANDIDATES[no_vae_id]
    vae_hard = controller.CANDIDATES[vae_id]
    no_vae_fields = no_vae.__dict__.copy()
    vae_fields = vae_hard.__dict__.copy()
    no_vae_fields.pop("candidate_id")
    vae_fields.pop("candidate_id")
    assert no_vae_fields.pop("method_family") == "direct_target_ssl_control"
    assert vae_fields.pop("method_family") == "vae_lhat_post_refine"
    assert no_vae_fields == vae_fields
    assert no_vae.supervised_corruption_schedule == (
        controller.ROTATING4_SUPERVISED_SCHEDULE
    )
    assert no_vae.comparison_rng_identity == vae_hard.comparison_rng_identity

    compiled: dict[str, Any] = {}
    for name, candidate in (("no_vae", no_vae), ("vae", vae_hard)):
        method = controller._method_for(candidate, "ningbo")
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compiled[name] = compile_method_profile(path)
        rotating4.validate_rotating4_method(compiled[name])
        contracts = method["contracts"]
        assert contracts["corrupted_exposures_per_base_record"] == 4
        assert contracts["total_exposures_per_base_record"] == 5
        assert contracts["family_loss_weights"] == {
            "clean": pytest.approx(0.5),
            "corrupted_total": pytest.approx(0.5),
            "corrupted_per_composition": pytest.approx(0.125),
        }

    assert compiled["no_vae"].profile_name == "direct_depth23_fixed20"
    assert compiled["vae"].profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
    assert pcgrad.validate_pcgrad_method(compiled["vae"]) == pytest.approx(1.5)

    with rotating4.patch_online_trainer_runtime():
        for method in compiled.values():
            steps = online_trainer._method_exposure_steps(method)
            assert [step.name for step in steps] == [
                "clean",
                "rotating_depth2_a",
                "rotating_depth2_b",
                "rotating_depth3_a",
                "rotating_depth3_b",
            ]
            assert [step.composition_index for step in steps] == [
                -1,
                0,
                1,
                10,
                19,
            ]
            assert sum(step.loss_scale for step in steps) == pytest.approx(1.0)
        assert rotating4.ROTATING4_EXPOSURE_POLICY in (
            online_trainer.FIXED20_GROUPED_EXPOSURE_POLICIES
        )
    assert rotating4.ROTATING4_EXPOSURE_POLICY not in (
        online_trainer.FIXED20_GROUPED_EXPOSURE_POLICIES
    )


def test_l36_l37_match_locked_effnet_stage2_budget() -> None:
    for candidate_id in (EFFNET_ROTATING4_NO_VAE_ID, EFFNET_ROTATING4_VAE_ID):
        candidate = controller.CANDIDATES[candidate_id]
        assert candidate.model_name == "efficientnet1dv2"
        assert candidate.learning_rate == pytest.approx(5.0e-5)
        assert candidate.batch_size == 128
        assert candidate.epochs == 23
        assert candidate.scheduler_horizon_epochs == 30
        assert candidate.pretrain_steps == 1024
        assert candidate.pretrain_ssl_objective == "simclr"
        assert candidate.pretrain_vicreg_mix_weight == pytest.approx(0.0)
        assert candidate.pretrain_source_replay_weight == pytest.approx(0.30)


def test_rotating4_hash_schedule_is_deterministic_and_covers_twenty() -> None:
    hash_ids = ("record-a", "record-b", "record-c")
    for position, hash_id in enumerate(hash_ids):
        depth2: list[int] = []
        depth3: list[int] = []
        for epoch in range(1, 6):
            for sentinel in (0, 1):
                values = rotating4.rotating_composition_indices(
                    hash_ids,
                    epoch=epoch,
                    slot_sentinel=sentinel,
                    device=torch.device("cpu"),
                )
                depth2.append(int(values[position]))
            for sentinel in (10, 19):
                values = rotating4.rotating_composition_indices(
                    hash_ids,
                    epoch=epoch,
                    slot_sentinel=sentinel,
                    device=torch.device("cpu"),
                )
                depth3.append(int(values[position]))
        assert set(depth2) == set(range(10))
        assert set(depth3) == set(range(10, 20))
        assert len(depth2) == len(set(depth2)) == 10
        assert len(depth3) == len(set(depth3)) == 10

    epoch1 = [
        rotating4.rotating_composition_indices(
            hash_ids,
            epoch=1,
            slot_sentinel=sentinel,
            device=torch.device("cpu"),
        )
        for sentinel in rotating4.ROTATING4_EXECUTION_SLOT_SENTINELS
    ]
    epoch6 = [
        rotating4.rotating_composition_indices(
            hash_ids,
            epoch=6,
            slot_sentinel=sentinel,
            device=torch.device("cpu"),
        )
        for sentinel in rotating4.ROTATING4_EXECUTION_SLOT_SENTINELS
    ]
    assert all(torch.equal(left, right) for left, right in zip(epoch1, epoch6))


def test_summary_parser_supports_candidate_arm_as_matched_reference() -> None:
    args = controller.build_parser().parse_args(
        [
            "summarize",
            "--candidate",
            DIRECT_LHAT_BALANCED_HARD_ID,
            "--direct-candidate",
            DIRECT_LHAT_LOCAL_UNIFORM_ID,
            "--direct-arm",
            "candidate",
        ]
    )
    assert args.direct_candidate == DIRECT_LHAT_LOCAL_UNIFORM_ID
    assert args.direct_arm == "candidate"


def test_replicate_summary_parser_supports_external_direct_owners() -> None:
    args = controller.build_parser().parse_args(
        [
            "summarize-replicates",
            "--reference-candidate",
            DIVERSE_VAE_POSTTRAIN_ID,
            "--reference-direct-candidate",
            SOURCE_REPLAY_MILD_CHAIN3_ID,
            "--replicate-candidate",
            INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID,
            "--replicate-direct-candidate",
            INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID,
        ]
    )

    assert args.reference_direct_candidate == SOURCE_REPLAY_MILD_CHAIN3_ID
    assert args.replicate_direct_candidate == (
        INDEPENDENT_DIVERSE_VAE_POSTTRAIN_ID
    )


def test_replicate_summary_parser_supports_a_third_seed() -> None:
    args = controller.build_parser().parse_args(
        [
            "summarize-replicates",
            "--reference-candidate",
            GRADIENT_BALANCED_MILD_CHAIN3_ID,
            "--reference-direct-candidate",
            SOURCE_REPLAY_MILD_CHAIN3_ID,
            "--replicate-candidate",
            INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID,
            "--replicate-direct-candidate",
            INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID,
            "--additional-candidate",
            SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID,
            "--additional-direct-candidate",
            SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID,
        ]
    )

    assert args.additional_candidate == [
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]
    assert args.additional_direct_candidate == [
        SECOND_INDEPENDENT_GRADIENT_BALANCED_CHAIN3_ID
    ]


def test_pcgrad_diagnostics_emit_only_active_method_metadata() -> None:
    pcgrad._STATE.clear_runtime()
    pcgrad._STATE.steps.clear()
    pcgrad._STATE.wrapper_method_id = pcgrad.GRADIENT_BALANCED_METHOD_ID
    pcgrad._STATE.active_hyperparameters["alpha"] = 0.75
    strong_payload = pcgrad.diagnostics_payload()
    assert strong_payload is not None
    assert strong_payload["wrapper_method_id"] == (
        pcgrad.GRADIENT_BALANCED_METHOD_ID
    )
    assert "candidate_replay" not in strong_payload
    assert "auxiliary_term_scaling" not in strong_payload

    pcgrad._STATE.clear_runtime()
    pcgrad._STATE.steps.clear()
    pcgrad._STATE.wrapper_method_id = pcgrad.SEARCH_METHOD_ID
    pcgrad._STATE.active_hyperparameters["alpha"] = 1.5
    d19_payload = pcgrad.diagnostics_payload()
    assert d19_payload is not None
    assert d19_payload["wrapper_method_id"] == pcgrad.SEARCH_METHOD_ID
    assert d19_payload["candidate_replay"]["count"] == 0
    assert d19_payload["auxiliary_term_scaling"]["effective_bce_weights"] == (
        pcgrad.D19_AUXILIARY_BCE_WEIGHTS
    )
    pcgrad._STATE.clear_runtime()
    pcgrad._STATE.steps.clear()


@pytest.mark.parametrize(
    ("full_id", "short_id"),
    zip(STAGE1_IDS, SHORT_STAGE1_IDS, strict=True),
)
def test_short_stage1_changes_only_supervised_horizon_and_identity(
    full_id: str,
    short_id: str,
) -> None:
    full = controller.CANDIDATES[full_id].__dict__.copy()
    short = controller.CANDIDATES[short_id].__dict__.copy()
    assert full.pop("candidate_id") == full_id
    assert short.pop("candidate_id") == short_id
    assert full.pop("epochs") == 40
    assert short.pop("epochs") == 20
    assert full.pop("scheduler_horizon_epochs") == 40
    assert short.pop("scheduler_horizon_epochs") == 20
    assert full.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_v1"
    )
    assert short.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_e20_v1"
    )
    assert full == short


def test_short_stage1_pair_differs_only_by_view_policy_and_description() -> None:
    identity = controller._method_for(
        controller.CANDIDATES[SHORT_STAGE1_IDS[0]], "ningbo"
    )
    augmix = controller._method_for(
        controller.CANDIDATES[SHORT_STAGE1_IDS[1]], "ningbo"
    )
    assert _normalize_stage1_explanation_fields(identity) == (
        _normalize_stage1_explanation_fields(augmix)
    )
    assert identity["contracts"]["unlabeled_teacher"]["pretrain_view_mode"] == (
        "clean_identity"
    )
    assert augmix["contracts"]["unlabeled_teacher"]["pretrain_view_mode"] == (
        "twochain_augmix"
    )
    for candidate_id in SHORT_STAGE1_IDS:
        candidate = controller.CANDIDATES[candidate_id]
        assert candidate.epochs == candidate.scheduler_horizon_epochs == 20
        assert candidate.pretrain_steps == 4096
        assert candidate.pretrain_source_replay_weight == 0.0
        assert candidate.method_family == "direct_target_ssl_control"


@pytest.mark.parametrize(
    ("short_id", "extended_id"),
    zip(SHORT_STAGE1_IDS, EXTENDED_STAGE1_IDS, strict=True),
)
def test_extended_stage1_changes_only_ssl_budget_and_rng_identity(
    short_id: str,
    extended_id: str,
) -> None:
    short = controller.CANDIDATES[short_id].__dict__.copy()
    extended = controller.CANDIDATES[extended_id].__dict__.copy()
    assert short.pop("candidate_id") == short_id
    assert extended.pop("candidate_id") == extended_id
    assert short.pop("pretrain_epochs") == 1
    assert extended.pop("pretrain_epochs") == 2
    assert short.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_e20_v1"
    )
    assert extended.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_ssl8192_e20_v1"
    )
    assert short == extended


def test_extended_stage1_pair_differs_only_by_view_policy_and_description() -> None:
    identity = controller._method_for(
        controller.CANDIDATES[EXTENDED_STAGE1_IDS[0]], "ningbo"
    )
    augmix = controller._method_for(
        controller.CANDIDATES[EXTENDED_STAGE1_IDS[1]], "ningbo"
    )
    assert _normalize_stage1_explanation_fields(identity) == (
        _normalize_stage1_explanation_fields(augmix)
    )
    assert identity["contracts"]["unlabeled_teacher"]["pretrain_view_mode"] == (
        "clean_identity"
    )
    assert augmix["contracts"]["unlabeled_teacher"]["pretrain_view_mode"] == (
        "twochain_augmix"
    )
    for candidate_id in EXTENDED_STAGE1_IDS:
        candidate = controller.CANDIDATES[candidate_id]
        assert candidate.epochs == candidate.scheduler_horizon_epochs == 20
        assert candidate.pretrain_epochs == 2
        assert candidate.pretrain_steps == 4096
        assert candidate.pretrain_epochs * candidate.pretrain_steps == 8192
        assert candidate.pretrain_source_replay_weight == 0.0
        assert candidate.method_family == "direct_target_ssl_control"


def test_extended_stage1_shared_direct_matches_identity_candidate_contract() -> None:
    identity = controller.CANDIDATES[EXTENDED_STAGE1_IDS[0]].__dict__.copy()
    shared_direct = controller.CANDIDATES[
        EXTENDED_STAGE1_SHARED_DIRECT_ID
    ].__dict__.copy()
    assert identity.pop("candidate_id") == EXTENDED_STAGE1_IDS[0]
    assert shared_direct.pop("candidate_id") == EXTENDED_STAGE1_SHARED_DIRECT_ID
    assert identity == shared_direct


def test_classic_alpha_stage1_changes_only_augmix_mixing_alpha() -> None:
    baseline = controller.CANDIDATES[SHORT_STAGE1_IDS[1]].__dict__.copy()
    classic = controller.CANDIDATES[CLASSIC_ALPHA_STAGE1_ID].__dict__.copy()
    assert baseline.pop("candidate_id") == SHORT_STAGE1_IDS[1]
    assert classic.pop("candidate_id") == CLASSIC_ALPHA_STAGE1_ID
    assert baseline.pop("augmix_dirichlet_alpha") == pytest.approx(0.5)
    assert classic.pop("augmix_dirichlet_alpha") == pytest.approx(1.0)
    assert baseline.pop("augmix_beta_alpha") == pytest.approx(0.5)
    assert classic.pop("augmix_beta_alpha") == pytest.approx(1.0)
    assert baseline == classic

    method = controller._method_for(
        controller.CANDIDATES[CLASSIC_ALPHA_STAGE1_ID], "ningbo"
    )
    teacher = method["contracts"]["unlabeled_teacher"]
    assert teacher["pretrain_view_mode"] == "twochain_augmix"
    assert teacher["dirichlet_alpha"] == pytest.approx(1.0)
    assert teacher["beta_alpha"] == pytest.approx(1.0)
    assert teacher["pretrain_source_replay_weight"] == 0.0
    assert teacher["labels_consumed"] is False
    assert not {"lhat", "threechain_augmix"}.intersection(method["nodes"])
    assert not {
        "vae",
        "latent_pool",
        "vae_decoder",
        "lhat_config",
        "augmix_config",
    }.intersection(method["resources"])


@pytest.mark.parametrize("candidate_id", EXTENDED_STAGE1_IDS)
def test_extended_stage1_runtime_accepts_frozen_8192_step_boundary(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    method = controller._method_for(
        controller.CANDIDATES[candidate_id], "ningbo"
    )
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    config = unlabeled_teacher._teacher_config(compile_method_profile(path))
    assert config is not None
    assert config.pretrain_epochs == 2
    assert config.pretrain_steps_per_epoch == 4096
    assert config.pretrain_epochs * config.pretrain_steps_per_epoch == 8192


def test_replay_stage1_pair_differs_only_by_view_policy_and_description() -> None:
    identity = controller._method_for(
        controller.CANDIDATES[REPLAY_STAGE1_IDS[0]], "ningbo"
    )
    augmix = controller._method_for(
        controller.CANDIDATES[REPLAY_STAGE1_IDS[1]], "ningbo"
    )

    assert _normalize_stage1_explanation_fields(identity) == (
        _normalize_stage1_explanation_fields(augmix)
    )
    assert identity["method"]["scientific_arm"] != (
        augmix["method"]["scientific_arm"]
    )
    assert identity["method"]["description"] != augmix["method"]["description"]

    expected_views = {
        REPLAY_STAGE1_IDS[0]: (
            "clean_identity",
            "clean_identity_simclr_vicreg_control_v1",
        ),
        REPLAY_STAGE1_IDS[1]: (
            "twochain_augmix",
            "ema_teacher_clean_to_twochain_augmix_strong_v1",
        ),
    }
    for candidate_id, method in zip(
        REPLAY_STAGE1_IDS, (identity, augmix), strict=True
    ):
        candidate = controller.CANDIDATES[candidate_id]
        teacher = method["contracts"]["unlabeled_teacher"]
        assert (
            teacher["pretrain_view_mode"],
            teacher["policy"],
        ) == expected_views[candidate_id]
        assert teacher["pretrain_source_replay_weight"] == pytest.approx(0.30)
        assert teacher["pretrain_source_batches_per_step"] == 1
        assert teacher["labels_consumed"] is False
        assert candidate.comparison_rng_identity == (
            "sequential_effnet_stage1_v1"
        )
        assert "full_k500_heldout_oracle" not in method["contracts"]
        assert method["contracts"]["heldout_target_feedback_allowed"] is False
        assert not {"lhat", "threechain_augmix"}.intersection(method["nodes"])
        assert not {
            "vae",
            "latent_pool",
            "vae_decoder",
            "lhat_config",
            "augmix_config",
        }.intersection(method["resources"])


@pytest.mark.parametrize(
    ("baseline_id", "replay_id"),
    zip(STAGE1_IDS, REPLAY_STAGE1_IDS, strict=True),
)
def test_source_replay_stage1_changes_only_replay_and_rng_identity(
    baseline_id: str,
    replay_id: str,
) -> None:
    baseline = controller.CANDIDATES[baseline_id].__dict__.copy()
    replay = controller.CANDIDATES[replay_id].__dict__.copy()
    assert baseline.pop("candidate_id") == baseline_id
    assert replay.pop("candidate_id") == replay_id
    assert baseline.pop("pretrain_source_replay_weight") == 0.0
    assert replay.pop("pretrain_source_replay_weight") == pytest.approx(0.30)
    assert baseline.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_v1"
    )
    assert replay.pop("comparison_rng_identity") == (
        "sequential_effnet_stage1_v1"
    )
    assert baseline == replay


def test_source_replay_records_ordered_source_batch_hash(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[REPLAY_STAGE1_IDS[0]]
    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / "replay_method.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    config = unlabeled_teacher._teacher_config(compile_method_profile(path))
    assert config is not None

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(5))

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.bias.unsqueeze(0).expand(int(waveform.shape[0]), -1)

    batch_size = 4
    batch = {
        "waveform": torch.zeros(batch_size, 1000, 12),
        "label": torch.zeros(batch_size, 5),
        "hash_id": [f"source-{index}" for index in range(batch_size)],
    }
    expected = unlabeled_teacher._batch_digest(batch, batch_size)
    unlabeled_teacher._STATE.reset()
    try:
        unlabeled_teacher._STATE.config = config
        unlabeled_teacher._STATE.source_loader = [batch]
        loss = unlabeled_teacher._pretrain_source_semantic_loss(
            DummyModel(),
            EFFICIENTNET1DV2_SPEC,
            config,
            normalization_epsilon=1.0e-6,
        )
        assert torch.isfinite(loss)
        assert unlabeled_teacher._STATE.pretrain_source_samples == batch_size
        assert unlabeled_teacher._STATE.source_hash_digests == [expected]
        diagnostics = unlabeled_teacher.diagnostics_payload()
        assert diagnostics is not None
        replay = diagnostics["pretrain"]["source_semantic_replay"]
        assert replay["ordered_batch_hashes_sha256"] == (
            hashlib.sha256(expected.encode()).hexdigest()
        )
    finally:
        unlabeled_teacher._STATE.reset()


def test_source_replay_does_not_change_projector_or_supervised_rng_identity(
    tmp_path: Path,
) -> None:
    configs = []
    for candidate_id in (STAGE1_IDS[0], REPLAY_STAGE1_IDS[0]):
        candidate = controller.CANDIDATES[candidate_id]
        method = controller._method_for(candidate, "ningbo")
        path = tmp_path / f"{candidate_id}.yaml"
        path.write_text(
            yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        config = unlabeled_teacher._teacher_config(compile_method_profile(path))
        assert config is not None
        configs.append(config)
    assert (
        controller.CANDIDATES[STAGE1_IDS[0]].comparison_rng_identity
        == controller.CANDIDATES[REPLAY_STAGE1_IDS[0]].comparison_rng_identity
    )
    unlabeled_teacher._STATE.reset()
    try:
        unlabeled_teacher._STATE.center = "ningbo"
        unlabeled_teacher._STATE.model_name = "efficientnet1dv2"
        unlabeled_teacher._STATE.seed_config_path = (
            SANDBOX_ROOT / "configs" / "random_seed.yaml"
        )
        unlabeled_teacher._STATE.replicate_id = 0
        assert unlabeled_teacher._pretrain_projector_seed(
            configs[0]
        ) == unlabeled_teacher._pretrain_projector_seed(configs[1])
        seed0 = unlabeled_teacher._pretrain_projector_seed(configs[0])
        unlabeled_teacher._STATE.replicate_id = 1
        assert unlabeled_teacher._pretrain_projector_seed(configs[0]) != seed0
    finally:
        unlabeled_teacher._STATE.reset()


def test_source_loader_identity_is_fail_closed() -> None:
    manifest_sha = "a" * 64
    replay = {
        "loader": {
            "dataset": {
                "cache": {
                    "dataset": "ptbxl",
                    "record_count": 21_799,
                    "sampling_rate_hz": 100,
                    "duration_seconds": 10.0,
                    "signal_shape": [21_799, 1000, 12],
                    "lead_order": list(controller.PTBXL_LEAD_ORDER),
                    "class_order": list(controller.CLASS_ORDER),
                    "physical_unit": "mV",
                    "normalization": "none",
                    "storage_mode": "ram",
                    "manifest_sha256": manifest_sha,
                    "available_memory_at_open_bytes": 123_456,
                },
                "selection": {
                    "dataset": "ptbxl",
                    "cache_dataset": "ptbxl",
                    "partition": "train",
                    "record_count": 17_084,
                    "split_id": "ptbxl_super5_official_folds_v1",
                    "class_order": list(controller.CLASS_ORDER),
                    "mapping_version": None,
                    "mapping_hash": None,
                    "ref_excluded_evaluation": False,
                    "split_manifest_sha256": "b" * 64,
                    "source_manifest_sha256": manifest_sha,
                    "hash_id_set_sha256": "c" * 64,
                },
            },
            "batch_size": 64,
            "num_workers": 0,
            "drop_last": True,
            "seed": {
                "namespace": (
                    "target_ssl_source_semantic_v1:"
                    "ningbo:efficientnet1dv2:replicate0"
                ),
                "config_sha256": "d" * 64,
            },
        }
    }
    identity = controller._validated_source_loader_identity(
        replay,
        enabled=True,
        center="ningbo",
        model_name="efficientnet1dv2",
    )
    assert identity["dataset"]["selection"]["partition"] == "train"
    assert (
        "available_memory_at_open_bytes"
        not in identity["dataset"]["cache"]
    )
    drifted = copy.deepcopy(replay)
    drifted["loader"]["dataset"]["cache"]["sampling_rate_hz"] = 500
    with pytest.raises(ValueError, match="PTB-XL contract drifted"):
        controller._validated_source_loader_identity(
            drifted,
            enabled=True,
            center="ningbo",
            model_name="efficientnet1dv2",
        )


def test_stage1_replay_factorial_summary_computes_main_effects_and_interaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"locked-source")
    monkeypatch.setattr(controller, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(controller, "_source_checkpoint", lambda candidate: source)

    def evidence(candidate: controller.Candidate, center: str) -> dict[str, Any]:
        replay_on = bool(candidate.pretrain_source_replay_weight)
        return {
            "checkpoint_sha256": "a" * 64,
            "k500_selection": {
                "logical_center": center,
                "partition": "k500",
                "record_count": 500,
                "hash_id_set_sha256": "b" * 64,
                "source_manifest_sha256": "c" * 64,
                "split_manifest_sha256": "d" * 64,
                "mapping_version": controller.MAPPING_VERSION,
                "mapping_hash": controller.MAPPING_HASH,
            },
            "target_order_sha256": "e" * 64,
            "source_replay": {
                "enabled": replay_on,
                "weight": float(candidate.pretrain_source_replay_weight or 0.0),
                "batches_per_step": 1,
                "samples": 262_144 if replay_on else 0,
                "ordered_batch_hashes_sha256": "f" * 64 if replay_on else None,
            },
        }

    monkeypatch.setattr(controller, "_stage1_replay_evidence", evidence)
    monkeypatch.setattr(
        controller,
        "_assert_stage1_supervised_pair",
        lambda *args, **kwargs: {
            "optimizer_steps": 320,
            "epochs": [{"epoch": index} for index in range(1, 41)],
        },
    )
    metrics = {
        STAGE1_IDS[0]: [0.80, 0.50, 0.70, 0.40],
        STAGE1_IDS[1]: [0.79, 0.49, 0.74, 0.44],
        REPLAY_STAGE1_IDS[0]: [0.82, 0.52, 0.71, 0.41],
        REPLAY_STAGE1_IDS[1]: [0.82, 0.53, 0.76, 0.47],
    }
    monkeypatch.setattr(
        controller,
        "_arm_metrics",
        lambda candidate, arm: {
            center: metrics[candidate.candidate_id]
            for center in controller.CENTERS
        },
    )
    output = controller.summarize_stage1_replay_factorial(
        controller.CANDIDATES[STAGE1_IDS[0]],
        controller.CANDIDATES[STAGE1_IDS[1]],
        controller.CANDIDATES[REPLAY_STAGE1_IDS[0]],
        controller.CANDIDATES[REPLAY_STAGE1_IDS[1]],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["effects"]["augmix_at_replay_off_A0_minus_I0_pp"] == (
        pytest.approx([-1.0, -1.0, 4.0, 4.0])
    )
    assert payload["effects"]["augmix_at_replay_on_A1_minus_I1_pp"] == (
        pytest.approx([0.0, 1.0, 5.0, 6.0])
    )
    assert payload["effects"]["replay_under_augmix_A1_minus_A0_pp"] == (
        pytest.approx([3.0, 4.0, 2.0, 3.0])
    )
    assert payload["effects"]["interaction_pp"] == (
        pytest.approx([1.0, 2.0, 1.0, 2.0])
    )


@pytest.mark.parametrize("candidate_id", STAGE2_IDS)
def test_sequential_vae_post_profile_compiles_and_is_allowlisted(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    method = controller._method_for(
        controller.CANDIDATES[candidate_id], "ningbo"
    )
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    compiled = compile_method_profile(path)
    assert compiled.profile_name == pcgrad.VAE_POST_METHOD_ID
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(0.25)

    assert method["resources"]["lhat_config"]["path"] == "train/lhat.yaml"
    assert method["contracts"]["lhat_pgd_epsilon_l2_standardized"] == 2.0
    assert method["contracts"]["pcgrad_auxiliary_terms"] == [
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    assert method["contracts"]["exposure_policy"] == (
        "clean_once_then_exhaustive_depth23"
    )
    assert method["contracts"]["total_exposures_per_base_record"] == 21
    assert method["contracts"]["optimizer_step_policy"] == (
        "accumulate_family_balanced_once_per_base_batch"
    )
    assert [term["id"] for term in method["objective"]["terms"]] == [
        "clean_bce",
        "corrupted_bce",
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    assert set(method["nodes"]) == {
        "clean_identity",
        "depth23_corruption",
        "lhat",
    }
    assert not {"augmix_config", "augmix_rng"}.intersection(method["resources"])
    assert not {"augmix_view", "augmix_diagnostics"}.intersection(
        method["outputs"]
    )
    assert "full_k500_heldout_oracle" not in method["contracts"]
    assert set(method["contracts"]["post_refine_excluded_components"]) == {
        "augmix",
        "raw_auxiliary",
        "vae_random",
        "source_replay",
    }


def test_p22_hard_only_pruned_profile_keeps_stage1_and_prunes_stage2(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / "p22r4_simclr_hard_only_pruned.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    compiled = compile_method_profile(path)
    assert compiled.profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
    assert pcgrad.validate_pcgrad_method(compiled) == pytest.approx(1.5)
    assert method["contracts"]["pcgrad_auxiliary_mode"] == pcgrad.DIRECT_SUM_MODE
    assert method["contracts"]["exposure_policy"] == (
        "clean_aux_once_then_exhaustive_depth23"
    )
    assert method["contracts"]["pcgrad_auxiliary_terms"] == [
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    assert [term["id"] for term in method["objective"]["terms"]] == [
        "clean_bce",
        "corrupted_bce",
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    assert set(method["nodes"]) == {
        "clean_identity",
        "depth23_corruption",
        "lhat",
    }
    assert not {"augmix_config", "augmix_rng"}.intersection(method["resources"])
    assert not {"augmix_view", "augmix_diagnostics"}.intersection(
        method["outputs"]
    )

    teacher = method["contracts"]["unlabeled_teacher"]
    assert teacher["pretrain_view_mode"] == "twochain_augmix"
    assert teacher["pretrain_ssl_objective"] == "simclr"
    assert teacher["pretrain_vicreg_mix_weight"] == pytest.approx(0.0)
    assert teacher["pretrain_source_replay_weight"] == pytest.approx(0.30)
    assert teacher["pretrain_steps_per_epoch"] == 4096
    assert set(method["contracts"]["post_refine_excluded_components"]) == {
        "augmix",
        "raw_auxiliary",
        "vae_random",
        "source_replay",
    }


def test_p22r5_changes_only_stage1_to_classic_augmix3_depth13(
    tmp_path: Path,
) -> None:
    r4 = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    r5 = controller.CANDIDATES[
        "p22r5_classic_augmix3_depth13_hard_only_pruned"
    ]
    r4_fields = r4.__dict__.copy()
    r5_fields = r5.__dict__.copy()
    r4_fields.pop("candidate_id")
    r5_fields.pop("candidate_id")
    assert {
        key
        for key in r4_fields
        if r4_fields[key] != r5_fields[key]
    } == {
        "augmix_dirichlet_alpha",
        "augmix_beta_alpha",
        "pretrain_view_mode",
    }

    method = controller._method_for(r5, "ningbo")
    path = tmp_path / "p22r5_classic_augmix3_depth13.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)
    config = unlabeled_teacher._teacher_config(compiled)
    assert config is not None
    assert compiled.profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
    assert compiled.profile_name in unlabeled_teacher.ALLOWED_METHOD_IDS
    assert config.pretrain_clean_anchor is True
    assert config.pretrain_view_mode == unlabeled_teacher.CLASSIC_AUGMIX_VIEW_MODE
    assert config.policy == unlabeled_teacher.CLASSIC_AUGMIX_POLICY
    assert config.dirichlet_alpha == pytest.approx(1.0)
    assert config.beta_alpha == pytest.approx(1.0)
    assert config.pretrain_augmix_topology == (
        unlabeled_teacher.CLASSIC_AUGMIX_TOPOLOGY
    )
    assert method["contracts"]["pcgrad_auxiliary_terms"] == [
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    assert not {"augmix_view", "augmix_diagnostics"}.intersection(
        method["outputs"]
    )


@pytest.mark.parametrize(
    (
        "candidate_id",
        "changed_fields",
        "pretrain_replay_weight",
        "pretrain_anchor_weight",
        "supervised_anchor_weight",
    ),
    [
        (
            "p22r6_no_stage1_source_replay",
            {"pretrain_source_replay_weight"},
            0.0,
            5.0,
            2.0,
        ),
        (
            "p22r7_no_logit_anchors",
            {
                "pretrain_logit_anchor_weight",
                "supervised_logit_anchor_weight",
            },
            0.30,
            0.0,
            0.0,
        ),
    ],
)
def test_p22_stability_controls_are_single_group_prunes(
    candidate_id: str,
    changed_fields: set[str],
    pretrain_replay_weight: float,
    pretrain_anchor_weight: float,
    supervised_anchor_weight: float,
    tmp_path: Path,
) -> None:
    reference = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    candidate = controller.CANDIDATES[candidate_id]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    reference_fields.pop("candidate_id")
    candidate_fields.pop("candidate_id")
    assert {
        key
        for key in reference_fields
        if reference_fields[key] != candidate_fields[key]
    } == changed_fields

    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)
    assert compiled.profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
    assert [term["id"] for term in method["objective"]["terms"]] == [
        "clean_bce",
        "corrupted_bce",
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]
    teacher = method["contracts"]["unlabeled_teacher"]
    assert teacher["pretrain_source_replay_weight"] == pytest.approx(
        pretrain_replay_weight
    )
    assert teacher["pretrain_logit_anchor_weight"] == pytest.approx(
        pretrain_anchor_weight
    )
    assert teacher["supervised_logit_anchor_weight"] == pytest.approx(
        supervised_anchor_weight
    )


@pytest.mark.parametrize(
    ("candidate_id", "changed_fields"),
    [
        ("p22r8_no_vae_lhat_stage2", {"method_family"}),
        ("p22r9_identity_pretrain_with_vae_lhat", {"pretrain_view_mode"}),
    ],
)
def test_p22_core_component_knockouts_change_one_declared_group(
    candidate_id: str,
    changed_fields: set[str],
    tmp_path: Path,
) -> None:
    reference = controller.CANDIDATES["p22r4_simclr_hard_only_pruned"]
    candidate = controller.CANDIDATES[candidate_id]
    reference_fields = reference.__dict__.copy()
    candidate_fields = candidate.__dict__.copy()
    reference_fields.pop("candidate_id")
    candidate_fields.pop("candidate_id")
    assert {
        key
        for key in reference_fields
        if reference_fields[key] != candidate_fields[key]
    } == changed_fields

    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)
    teacher = method["contracts"]["unlabeled_teacher"]
    assert teacher["pretrain_source_replay_weight"] == pytest.approx(0.30)
    assert teacher["pretrain_logit_anchor_weight"] == pytest.approx(5.0)
    assert teacher["supervised_logit_anchor_weight"] == pytest.approx(2.0)

    if candidate_id == "p22r8_no_vae_lhat_stage2":
        assert compiled.profile_name != pcgrad.HARD_ONLY_PRUNED_METHOD_ID
        assert [term["id"] for term in method["objective"]["terms"]] == [
            "clean_bce",
            "corrupted_bce",
        ]
        assert "lhat" not in method["nodes"]
        assert "vae" not in method["resources"]
        assert teacher["pretrain_view_mode"] == "twochain_augmix"
    else:
        assert compiled.profile_name == pcgrad.HARD_ONLY_PRUNED_METHOD_ID
        assert [term["id"] for term in method["objective"]["terms"]] == [
            "clean_bce",
            "corrupted_bce",
            "lhat_direct_bce",
            "clean_lhat_jsd",
        ]
        assert "lhat" in method["nodes"]
        assert teacher["pretrain_view_mode"] == "clean_identity"


def test_classic_augmix_chain_samples_depth_1_to_3_on_raw_ecg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_operator(
        operator: str,
        signal: torch.Tensor,
        *,
        params: dict[str, object],
        sampling_rate_hz: int,
        rng: torch.Generator,
    ) -> torch.Tensor:
        del params, rng
        calls.append((operator, sampling_rate_hz))
        return signal + 1.0

    monkeypatch.setattr(
        unlabeled_teacher,
        "apply_operator_batch_prevalidated",
        fake_operator,
    )
    candidate = controller.CANDIDATES[
        "p22r5_classic_augmix3_depth13_hard_only_pruned"
    ]
    method = controller._method_for(candidate, "ningbo")
    compiled_path = tmp_path / "p22r5_runtime_test.yaml"
    compiled_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    try:
        config = unlabeled_teacher._teacher_config(
            compile_method_profile(compiled_path)
        )
        assert config is not None
        raw = torch.zeros(64, 1000, 12)
        unlabeled_teacher._STATE.reset()
        unlabeled_teacher._STATE.operator_params = {
            name: {} for name in unlabeled_teacher.CANONICAL_OPERATORS
        }
        unlabeled_teacher._STATE.generator = torch.Generator().manual_seed(7)
        output = unlabeled_teacher._classic_augmix_chain(raw, config)
        assert tuple(output.shape) == tuple(raw.shape)
        assert torch.isfinite(output).all()
        assert set(output[:, 0, 0].tolist()) == {1.0, 2.0, 3.0}
        assert calls
        assert all(rate == 500 for _, rate in calls)
    finally:
        unlabeled_teacher._STATE.reset()
        compiled_path.unlink(missing_ok=True)


@pytest.mark.parametrize("candidate_id", STRONG_STAGE2_IDS)
def test_strong_vae_post_changes_only_generated_lhat_epsilon(
    candidate_id: str,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")

    assert method["resources"]["lhat_config"]["path"].endswith(
        f"{candidate_id}/lhat.yaml"
    )
    assert method["contracts"]["lhat_pgd_epsilon_l2_standardized"] == 8.0

    generated = controller._lhat_config_for(candidate)
    baseline = yaml.safe_load(
        controller.BASE_LHAT_CONFIG.read_text(encoding="utf-8")
    )
    assert generated["hull_attack"]["pgd_epsilon_l2_standardized"] == 8.0
    generated["hull_attack"]["pgd_epsilon_l2_standardized"] = 2.0
    assert generated == baseline


@pytest.mark.parametrize("candidate_id", CLEAN_POLISH_IDS)
def test_vae_clean_polish_is_native_single_exposure_and_scaled_once(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    path = tmp_path / f"{candidate_id}.yaml"
    path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)

    assert compiled.profile_name == controller.VAE_CLEAN_POLISH_METHOD_ID
    objective_scale.validate_scale_only_method(compiled)
    assert set(method["nodes"]) == {"clean_identity", "lhat"}
    assert set(method["outputs"]) == {
        "clean_view",
        "lhat_view",
        "lhat_diagnostics",
    }
    assert set(method["resources"]) == {
        "model",
        "vae",
        "latent_pool",
        "vae_decoder",
        "lhat_config",
        "lhat_rng",
    }
    assert "exposure_policy" not in method["contracts"]
    assert "pcgrad_auxiliary_mode" not in method["contracts"]
    assert method["contracts"]["objective_global_scale"] == pytest.approx(0.5)
    assert method["contracts"]["effective_objective_weights"] == (
        controller.VAE_CLEAN_POLISH_EFFECTIVE_WEIGHTS
    )
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": 0.5,
        "lhat_view": 0.5,
    }
    assert [term["id"] for term in method["objective"]["terms"]] == [
        "clean_bce",
        "lhat_direct_bce",
        "clean_lhat_jsd",
    ]

    direct = controller._direct_method_for(candidate, "ningbo")
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_compiled = compile_method_profile(direct_path)
    assert direct_compiled.profile_name == "clean_polish_control_v1"
    assert set(direct["nodes"]) == {"clean_identity"}
    assert set(direct["outputs"]) == {"clean_view"}
    assert set(direct["resources"]) == {"model"}
    assert [term["id"] for term in direct["objective"]["terms"]] == [
        "clean_bce"
    ]
    assert "exposure_policy" not in direct["contracts"]
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": 1.0
    }

    generated = controller._lhat_config_for(candidate)
    assert generated["hull_attack"]["pgd_epsilon_l2_standardized"] == 8.0


@pytest.mark.parametrize("candidate_id", TEACHER_DISTILL_POLISH_IDS)
def test_vae_teacher_distill_polish_is_pure_soft_and_matched(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)
    assert compiled.profile_name == (
        controller.VAE_TEACHER_DISTILL_POLISH_METHOD_ID
    )
    assert direct_compiled.profile_name == (
        controller.CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID
    )
    expected_initializer = (
        "s00_effnet_ssl4096_identity_e40"
        if candidate_id.startswith("k2i_")
        else "s01_effnet_ssl4096_augmix_e40"
    )
    assert candidate.initializer_candidate == expected_initializer
    assert candidate.epochs == candidate.scheduler_horizon_epochs == 4
    assert candidate.learning_rate == pytest.approx(3.0e-5)
    assert candidate.lhat_pgd_epsilon_l2_standardized == 8.0
    assert candidate.frozen_teacher_mix == pytest.approx(1.0)
    auxiliary_weight = float(candidate.auxiliary_alpha)
    clean_preserve_raw = 1.0 - 2.0 * auxiliary_weight
    auxiliary_raw = 2.0 * auxiliary_weight
    assert [
        (term["id"], term["weight"]) for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", pytest.approx(clean_preserve_raw)),
        ("lhat_teacher_distill_bce", pytest.approx(auxiliary_raw)),
    ]
    assert [
        (term["id"], term["weight"]) for term in direct["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", pytest.approx(clean_preserve_raw)),
        (
            "clean_teacher_distill_control_bce",
            pytest.approx(auxiliary_raw),
        ),
    ]
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.50),
        "clean_preserve_bce": pytest.approx(0.50 - auxiliary_weight),
        "lhat_teacher_distill_bce": pytest.approx(auxiliary_weight),
    }
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(1.0 - auxiliary_weight),
        "lhat_view": pytest.approx(auxiliary_weight),
    }
    assert direct["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(1.0 - auxiliary_weight),
        "lhat_control_view": pytest.approx(auxiliary_weight),
    }
    for payload in (method, direct):
        assert payload["contracts"]["auxiliary_teacher_mix"] == 1.0
        assert payload["contracts"]["auxiliary_soft_bce_pos_weight"] is None
        assert payload["contracts"][
            "auxiliary_teacher_initializer_sha256"
        ] == controller._sha256(controller._source_checkpoint(candidate, center))
        assert "clean_lhat_jsd" not in {
            term["id"] for term in payload["objective"]["terms"]
        }
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_TEACHER_DISTILL_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_TEACHER_DISTILL_POLISH_METHOD_ID
    )
    assert "lhat" in method["nodes"] and "lhat" in direct["nodes"]
    assert "lhat_control_identity" in direct["nodes"]
    assert set(method["resources"]) == set(direct["resources"])

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1
        assert arguments[arguments.index("--vae-checkpoint") + 1] == str(
            controller.VAE_CHECKPOINT
        )


@pytest.mark.parametrize("candidate_id", HARD_BCE_POLISH_IDS)
def test_vae_hard_bce_polish_is_lowdose_and_matched(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)

    assert compiled.profile_name == controller.VAE_HARD_BCE_POLISH_METHOD_ID
    assert direct_compiled.profile_name == (
        controller.CLEAN_HARD_BCE_CONTROL_METHOD_ID
    )
    assert not soft_teacher._is_teacher_method(compiled)
    assert not soft_teacher._is_teacher_method(direct_compiled)
    assert candidate.polish_auxiliary_target == "hard_label"
    assert candidate.frozen_teacher_mix is None
    assert [
        (term["id"], term["weight"]) for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", pytest.approx(0.8)),
        ("lhat_hard_bce", pytest.approx(0.2)),
    ]
    assert [
        (term["id"], term["weight"]) for term in direct["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", pytest.approx(0.8)),
        ("clean_hard_control_bce", pytest.approx(0.2)),
    ]
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.4),
        "lhat_hard_bce": pytest.approx(0.1),
    }
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.4),
        "clean_hard_control_bce": pytest.approx(0.1),
    }
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.9),
        "lhat_view": pytest.approx(0.1),
    }
    assert direct["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.9),
        "lhat_control_view": pytest.approx(0.1),
    }
    forbidden_contracts = {
        "auxiliary_label_policy",
        "auxiliary_teacher_policy",
        "auxiliary_teacher_mix",
        "auxiliary_teacher_scope",
        "auxiliary_teacher_initializer_sha256",
    }
    for payload in (method, direct):
        assert forbidden_contracts.isdisjoint(payload["contracts"])
        assert "clean_lhat_jsd" not in {
            term["id"] for term in payload["objective"]["terms"]
        }
        excluded = set(payload["contracts"]["post_refine_excluded_components"])
        assert {
            "fixed20",
            "augmix",
            "pcgrad",
            "raw_auxiliary",
            "vae_random",
            "source_replay",
        } <= excluded

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1
        assert arguments[arguments.index("--vae-checkpoint") + 1] == str(
            controller.VAE_CHECKPOINT
        )


@pytest.mark.parametrize(
    ("e40_id", "e20_id", "initializer"),
    (
        (
            "k3i_effnet_identity_vae_hard_bce_lowdose4",
            "k4i_effnet_e20_identity_vae_hard_bce_lowdose4",
            "s10_effnet_ssl4096_identity_e20",
        ),
        (
            "k3a_effnet_augmix_vae_hard_bce_lowdose4",
            "k4a_effnet_e20_augmix_vae_hard_bce_lowdose4",
            "s11_effnet_ssl4096_augmix_e20",
        ),
    ),
)
def test_e20_hard_bce_polish_changes_only_initializer_and_identity(
    e40_id: str,
    e20_id: str,
    initializer: str,
) -> None:
    e40 = controller.CANDIDATES[e40_id].__dict__.copy()
    e20 = controller.CANDIDATES[e20_id].__dict__.copy()
    assert e40.pop("candidate_id") == e40_id
    assert e20.pop("candidate_id") == e20_id
    e40.pop("initializer_candidate")
    assert e20.pop("initializer_candidate") == initializer
    assert e40.pop("comparison_rng_identity") == (
        "sequential_effnet_stage2_hard_bce_lowdose_v1"
    )
    assert e20.pop("comparison_rng_identity") == (
        "sequential_effnet_stage2_e20_hard_bce_lowdose_v1"
    )
    assert e40 == e20


@pytest.mark.parametrize("candidate_id", LHAT_PATH_POLISH_IDS)
def test_lhat_path_polish_is_simple_budget_matched_and_compilable(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)

    assert compiled.profile_name == controller.VAE_PATH_BCE_POLISH_METHOD_ID
    assert direct_compiled.profile_name == (
        controller.CLEAN_PATH_BCE_CONTROL_METHOD_ID
    )
    assert candidate.polish_auxiliary_target == "latent_path_hard_label"
    assert candidate.epochs == candidate.scheduler_horizon_epochs == 8
    assert candidate.auxiliary_alpha == pytest.approx(0.25)
    assert [
        (term["id"], term["weight"]) for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", pytest.approx(1.0)),
        ("clean_preserve_bce", pytest.approx(0.5)),
        ("lhat_path_hard_bce", pytest.approx(0.5)),
    ]
    assert [
        (term["id"], term["weight"]) for term in direct["objective"]["terms"]
    ] == [
        ("clean_bce", pytest.approx(1.0)),
        ("clean_preserve_bce", pytest.approx(0.5)),
        ("clean_lhat_path_control_bce", pytest.approx(0.5)),
    ]
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "lhat_path_hard_bce": pytest.approx(0.25),
    }
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "clean_lhat_path_control_bce": pytest.approx(0.25),
    }
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.75),
        "lhat_view": pytest.approx(0.25),
    }
    assert direct["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.75),
        "lhat_control_view": pytest.approx(0.25),
    }
    expected_path = {
        "enabled": True,
        "t_values": controller.LHAT_PATH_T_VALUES,
        "sampling": "uniform_valid_per_eligible_record_per_epoch",
        "geometry": "standardized_clean_to_lhat_ray",
        "residual_correction": "linear_clean_hard_endpoint",
        "candidate_training_view": "one_sampled_path_point",
        "matched_compute_in_control": True,
    }
    assert method["contracts"]["lhat_path_polish"] == expected_path
    assert direct["contracts"]["lhat_path_polish"] == expected_path
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_PATH_BCE_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_PATH_BCE_POLISH_METHOD_ID
    )
    assert direct["contracts"][
        "unused_lhat_path_computation_for_matched_compute"
    ] is True
    for payload in (method, direct):
        ids = {term["id"] for term in payload["objective"]["terms"]}
        assert ids.isdisjoint(
            {"clean_lhat_jsd", "clean_lhat_consistency_jsd", "augmix_bce"}
        )
        assert payload["contracts"]["hard_auxiliary_target_policy"] == (
            "anchor_ground_truth_multihot"
        )

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1


@pytest.mark.parametrize("candidate_id", CONSISTENCY_POLISH_IDS)
def test_vae_consistency_polish_is_lowdose_label_free_and_matched(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)

    assert compiled.profile_name == (
        controller.VAE_CONSISTENCY_POLISH_METHOD_ID
    )
    assert direct_compiled.profile_name == (
        controller.CLEAN_CONSISTENCY_CONTROL_METHOD_ID
    )
    assert not soft_teacher._is_teacher_method(compiled)
    assert not soft_teacher._is_teacher_method(direct_compiled)
    assert candidate.polish_auxiliary_target == "consistency"
    assert candidate.auxiliary_alpha == pytest.approx(0.10)
    assert [
        (term["id"], term["weight"]) for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", pytest.approx(1.0)),
        ("clean_preserve_bce", pytest.approx(0.8)),
        ("clean_lhat_consistency_jsd", pytest.approx(0.2)),
    ]
    assert method["objective"]["terms"][2]["views"] == [
        "clean_view",
        "lhat_view",
    ]
    assert [
        (term["id"], term["weight"]) for term in direct["objective"]["terms"]
    ] == [
        ("clean_bce", pytest.approx(1.0)),
        ("clean_preserve_bce", pytest.approx(0.8)),
        ("clean_identity_consistency_jsd", pytest.approx(0.2)),
    ]
    assert direct["objective"]["terms"][2]["views"] == [
        "clean_view",
        "lhat_control_view",
    ]
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.4),
        "clean_lhat_consistency_jsd": pytest.approx(0.1),
    }
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.4),
        "clean_identity_consistency_jsd": pytest.approx(0.1),
    }
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.9),
        "lhat_view": pytest.approx(0.1),
    }
    assert direct["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.9),
        "lhat_control_view": pytest.approx(0.1),
    }
    for payload in (method, direct):
        contracts = payload["contracts"]
        assert contracts["clean_lhat_consistency_policy"] == (
            "symmetric_bernoulli_jsd_current_model"
        )
        assert contracts["lhat_direct_label_supervision"] is False
        assert {
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "hard_auxiliary_target_policy",
        }.isdisjoint(contracts)
        assert "lhat_hard_bce" not in {
            term["id"] for term in payload["objective"]["terms"]
        }
    assert "lhat_control_identity" in direct["nodes"]
    assert set(method["resources"]) == set(direct["resources"])

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1
        assert arguments[arguments.index("--vae-checkpoint") + 1] == str(
            controller.VAE_CHECKPOINT
        )


@pytest.mark.parametrize(
    ("hard_id", "consistency_id"),
    zip(HARD_BCE_POLISH_IDS[-2:], CONSISTENCY_POLISH_IDS, strict=True),
)
def test_consistency_polish_changes_only_target_and_rng_identity(
    hard_id: str,
    consistency_id: str,
) -> None:
    hard = controller.CANDIDATES[hard_id].__dict__.copy()
    consistency = controller.CANDIDATES[consistency_id].__dict__.copy()
    hard.pop("candidate_id")
    consistency.pop("candidate_id")
    assert hard.pop("polish_auxiliary_target") == "hard_label"
    assert consistency.pop("polish_auxiliary_target") == "consistency"
    assert hard.pop("comparison_rng_identity") == (
        "sequential_effnet_stage2_e20_hard_bce_lowdose_v1"
    )
    assert consistency.pop("comparison_rng_identity") == (
        "sequential_effnet_stage2_e20_consistency_lowdose_v1"
    )
    assert hard == consistency


@pytest.mark.parametrize("candidate_id", LOCAL_ANCHOR_SOFT_POLISH_IDS)
def test_local_anchor_soft_polish_is_local_matched_and_compilable(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    direct = controller._direct_method_for(candidate, "ningbo")
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)
    assert soft_teacher._is_teacher_method(compiled) is False
    assert soft_teacher._is_teacher_method(direct_compiled) is False

    assert compiled.profile_name == (
        controller.VAE_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID
    )
    assert direct_compiled.profile_name == (
        controller.CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID
    )
    assert candidate.epochs == candidate.scheduler_horizon_epochs == 8
    assert candidate.auxiliary_alpha == pytest.approx(0.25)
    assert candidate.lhat_pgd_epsilon_l2_standardized == pytest.approx(2.0)
    assert [
        (term["id"], term["type"], term["weight"])
        for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", "multilabel_bce_with_logits", pytest.approx(1.0)),
        (
            "clean_preserve_bce",
            "multilabel_bce_with_logits",
            pytest.approx(0.5),
        ),
        (
            "lhat_anchor_soft_bce",
            "multilabel_bce_with_logits",
            pytest.approx(0.5),
        ),
    ]
    assert direct["objective"]["terms"][-1] == {
        "id": "clean_anchor_soft_control_bce",
        "type": "multilabel_bce_with_logits",
        "view": "lhat_control_view",
        "targets": "targets",
        "weight": pytest.approx(0.5),
    }
    expected_policy = {
        "candidate_policy": "exact_positive_set_nonself",
        "candidate_count": 20,
        "candidate_includes_anchor": False,
        "anchor_in_outer_interpolation": True,
        "local_pool_size": 80,
        "hull_lambda": pytest.approx(0.05),
        "pgd_epsilon_l2_standardized": pytest.approx(2.0),
        "steps": 3,
        "learning_rate": pytest.approx(0.25),
        "minimum_effective_anchor_mass": pytest.approx(0.95),
        "matched_compute_in_control": True,
    }
    for payload in (method, direct):
        contracts = payload["contracts"]
        assert contracts["local_anchor_soft_polish"] == expected_policy
        assert contracts["anchor_soft_target_policy"] == (
            "anchor_soft_exact_positive_set_no_class_admission"
        )
        assert "auxiliary_label_policy" not in contracts
        assert contracts["auxiliary_positive_value"] == pytest.approx(0.95)
        assert contracts["auxiliary_negative_value"] == pytest.approx(0.0)
        assert contracts["auxiliary_soft_bce_pos_weight"] is None
        assert contracts["fixed20_train_views"] == 0
        assert contracts["augmix_train_views"] == 0
        assert set(contracts["post_refine_excluded_components"]) >= {
            "fixed20",
            "augmix",
            "pcgrad",
            "raw_auxiliary",
            "vae_random",
            "source_replay",
        }
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "lhat_anchor_soft_bce": pytest.approx(0.25),
    }
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "clean_anchor_soft_control_bce": pytest.approx(0.25),
    }
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID
    )
    assert direct["contracts"]["control_executes_full_lhat_search"] is True
    assert (
        "unused_lhat_node_retained_for_latent_pool_construction"
        not in direct["contracts"]
    )
    assert set(method["resources"]) == set(direct["resources"])
    assert set(method["nodes"]) == {"clean_identity", "lhat"}
    assert set(direct["nodes"]) == {"clean_identity", "lhat"}
    assert direct["outputs"]["lhat_control_view"]["source"] == "lhat.waveform"

    lhat = controller._lhat_config_for(candidate)
    assert lhat["candidates"] == {
        "mode": "nearest",
        "num_candidates": 20,
        "include_anchor": False,
        "label_policy": "exact_positive_set",
        "forbid_norm_abnormal_mix": True,
        "require_distinct": True,
        "require_non_self": True,
        "local_pool_size": 80,
        "validation_records_allowed": False,
    }
    assert lhat["hull_attack"] == {
        "weight_mode": "optimized_softmax",
        "init_logit_gap": pytest.approx(0.0),
        "hull_lambda": pytest.approx(0.05),
        "steps": 3,
        "learning_rate": pytest.approx(0.25),
        "pgd_epsilon_l2_standardized": pytest.approx(2.0),
        "objective": "maximize_multilabel_bce_with_logits",
        "optimizer": "adam",
    }


@pytest.mark.parametrize(
    "candidate_id", CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_IDS
)
def test_calibrated_local_anchor_soft_uses_only_frozen_gate2_geometry(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    direct = controller._direct_method_for(candidate, "ningbo")
    expected_policy = {
        "candidate_policy": "exact_positive_set_nonself",
        "candidate_count": 20,
        "candidate_includes_anchor": False,
        "anchor_in_outer_interpolation": True,
        "local_pool_size": 80,
        "hull_lambda": pytest.approx(0.20),
        "pgd_epsilon_l2_standardized": pytest.approx(4.0),
        "steps": 10,
        "learning_rate": pytest.approx(0.40),
        "minimum_effective_anchor_mass": pytest.approx(0.80),
        "matched_compute_in_control": True,
    }
    for name, payload, expected_id in (
        (
            "candidate",
            method,
            controller.VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID,
        ),
        (
            "control",
            direct,
            controller.CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID,
        ),
    ):
        path = tmp_path / f"{candidate_id}_{name}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compiled = compile_method_profile(path)
        objective_scale.validate_scale_only_method(compiled)
        assert compiled.profile_name == expected_id
        assert compiled.contracts["local_anchor_soft_polish"] == expected_policy
        assert compiled.contracts["local_lhat_geometry_preset"] == (
            "historical_gate2_calibrated_v1"
        )
        assert soft_teacher._is_teacher_method(compiled) is False
    assert direct["contracts"]["control_executes_full_lhat_search"] is True
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_CALIBRATED_LOCAL_ANCHOR_SOFT_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_CALIBRATED_LOCAL_ANCHOR_SOFT_POLISH_METHOD_ID
    )
    assert set(method["nodes"]) == set(direct["nodes"]) == {
        "clean_identity",
        "lhat",
    }
    lhat = controller._lhat_config_for(candidate)
    assert lhat["hull_attack"] == {
        "weight_mode": "optimized_softmax",
        "init_logit_gap": pytest.approx(0.0),
        "hull_lambda": pytest.approx(0.20),
        "steps": 10,
        "learning_rate": pytest.approx(0.40),
        "pgd_epsilon_l2_standardized": pytest.approx(4.0),
        "objective": "maximize_multilabel_bce_with_logits",
        "optimizer": "adam",
    }


@pytest.mark.parametrize("candidate_id", BOUNDED_AUGMAX_POLISH_IDS)
def test_bounded_augmax_candidate_and_full_compute_control_are_matched(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    control = controller._direct_method_for(candidate, "ningbo")
    compiled_by_arm = {}
    for arm, payload in (("candidate", method), ("control", control)):
        path = tmp_path / f"{candidate_id}_{arm}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compiled = compile_method_profile(path)
        objective_scale.validate_scale_only_method(compiled)
        compiled_by_arm[arm] = compiled

    assert compiled_by_arm["candidate"].profile_name == (
        controller.VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID
    )
    assert compiled_by_arm["control"].profile_name == (
        controller.CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID
    )
    assert candidate.epochs == 6
    assert candidate.scheduler_horizon_epochs == 40
    assert candidate.batch_size == 128
    assert candidate.learning_rate == pytest.approx(5.0e-5)
    assert set(method["nodes"]) == set(control["nodes"])
    assert set(method["resources"]) == set(control["resources"])
    assert method["objective"] == control["objective"]
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_BOUNDED_AUGMAX_CONTROL_METHOD_ID
    )
    assert control["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_LHAT_BOUNDED_AUGMAX_POLISH_METHOD_ID
    )
    assert control["contracts"]["matched_chain3_control_mode"] == (
        "replace_lhat_with_clean_after_full_search_and_qc"
    )
    assert control["contracts"]["control_executes_full_lhat_search"] is True
    for payload in (method, control):
        contracts = payload["contracts"]
        assert contracts["fixed20_train_views"] == 0
        assert contracts["augmix_train_views"] == 0
        assert contracts["tuning_partition_only"] == "k500"
        assert contracts["full_k500_refit_supported"] is True
        assert contracts["local_lhat_geometry_preset"] == (
            "k500_internal_bounded_augmax_lam35_eps10_v1"
        )
        assert contracts["lhat_hull_lambda"] == pytest.approx(0.35)
        assert contracts["lhat_steps"] == 10
        assert contracts["lhat_pgd_epsilon_l2_standardized"] == pytest.approx(
            10.0
        )

    lhat = controller._lhat_config_for(candidate)
    assert lhat["hull_attack"]["hull_lambda"] == pytest.approx(0.35)
    assert lhat["hull_attack"]["steps"] == 10
    assert lhat["hull_attack"]["learning_rate"] == pytest.approx(0.25)
    assert lhat["hull_attack"]["pgd_epsilon_l2_standardized"] == pytest.approx(
        10.0
    )
    candidate_train, _ = controller._candidate_experiments(
        candidate, "ningbo"
    )
    control_train, _ = controller._direct_experiments(candidate, "ningbo")
    for experiment in (candidate_train, control_train):
        arguments = experiment["entrypoint"]["arguments"]
        assert arguments.count("--vae-checkpoint") == 1
        assert arguments[arguments.index("--vae-checkpoint") + 1] == str(
            controller.VAE_CHECKPOINT
        )
    assert controller._online_for(
        candidate, "ningbo"
    ) == controller._direct_training_config_for(candidate, "ningbo")


def test_bounded_augmax_control_replaces_only_post_qc_lhat_waveform(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[
        "l10ra_effnet_e20_augmix_vae_bounded_augmax6"
    ]
    del candidate, tmp_path
    runtimes = {}
    for arm in ("candidate", "control"):
        runtime = object.__new__(objective_scale._A5CalibratedMixRuntime)
        runtime._matched_clean_chain3_control = arm == "control"
        runtimes[arm] = runtime

    labels = torch.zeros(2, 5)
    clean = WaveformView(
        name="clean_view",
        waveform=torch.zeros(2, 1000, 12),
        labels=labels,
        sample_ids=("a", "b"),
        valid_mask=torch.ones(2, dtype=torch.bool),
        provenance=Provenance(node_id="clean", operation="fixture"),
    )
    hard = WaveformView(
        name="lhat_view",
        waveform=torch.ones(2, 1000, 12),
        labels=labels,
        sample_ids=("a", "b"),
        valid_mask=torch.tensor([True, False]),
        provenance=Provenance(node_id="lhat", operation="full_lhat_fixture"),
        metadata={"diagnostic_means": {"loss_gain": 0.25}},
    )
    candidate_view = runtimes[
        "candidate"
    ]._replace_lhat_with_clean_for_matched_control(hard, (clean,))
    control_view = runtimes[
        "control"
    ]._replace_lhat_with_clean_for_matched_control(hard, (clean,))
    assert candidate_view is hard
    torch.testing.assert_close(control_view.waveform, clean.waveform)
    assert torch.equal(control_view.valid_mask, hard.valid_mask)
    assert control_view.metadata["diagnostic_means"] == {"loss_gain": 0.25}
    assert control_view.metadata["full_lhat_search_executed"] is True
    assert control_view.provenance.operation == (
        "full_lhat_search_then_clean_chain3_control"
    )


def test_bounded_augmax_control_contract_fails_closed(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[
        "l10ra_effnet_e20_augmix_vae_bounded_augmax6"
    ]
    payload = controller._direct_method_for(candidate, "ningbo")
    payload["contracts"]["matched_chain3_control_mode"] = "identity_before_lhat"
    path = tmp_path / "invalid_bounded_control.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(path)
    with pytest.raises(ValueError, match="full-search control contract"):
        objective_scale.validate_scale_only_method(compiled)


@pytest.mark.parametrize(
    ("method_id", "auxiliary_view_name", "auxiliary_term_name"),
    (
        (
            "l8ra_effnet_e20_augmix_vae_local_anchor_soft8",
            "lhat_view",
            "lhat_anchor_soft_bce",
        ),
        (
            "l8ri_effnet_e20_identity_vae_local_anchor_soft8",
            "lhat_control_view",
            "clean_anchor_soft_control_bce",
        ),
        (
            "l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8",
            "lhat_view",
            "lhat_anchor_soft_bce",
        ),
        (
            "l9ri_effnet_e20_identity_vae_calibrated_anchor_soft8",
            "lhat_control_view",
            "clean_anchor_soft_control_bce",
        ),
    ),
)
def test_local_anchor_soft_runtime_replaces_hard_target_without_extra_forward(
    method_id: str,
    auxiliary_view_name: str,
    auxiliary_term_name: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[method_id]
    payload = (
        controller._method_for(candidate, "ningbo")
        if auxiliary_view_name == "lhat_view"
        else controller._direct_method_for(candidate, "ningbo")
    )
    path = tmp_path / f"{method_id}_{auxiliary_view_name}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(path)

    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    generator = torch.Generator().manual_seed(20260727)
    waveform = torch.randn(3, 1000, 12, generator=generator)
    sample_ids = ("a", "b", "c")
    clean = WaveformView(
        name="clean_identity",
        waveform=waveform,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="clean_identity",
            operation="local_anchor_soft_runtime_test",
        ),
    )
    auxiliary = WaveformView(
        name=(
            "lhat"
            if auxiliary_view_name == "lhat_view"
            else "lhat_control_identity"
        ),
        waveform=waveform.clone(),
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id=(
                "lhat"
                if auxiliary_view_name == "lhat_view"
                else "lhat_control_identity"
            ),
            operation="local_anchor_soft_runtime_test",
        ),
    )
    bundle = ViewBundle(
        values={
            "clean_view": clean,
            auxiliary_view_name: auxiliary,
        },
        node_values={
            "clean_identity": clean,
            auxiliary.name: auxiliary,
        },
    )

    class CountingClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(
                torch.tensor([2.0, -1.0, 0.5, -2.0, 1.5])
            )
            self.forward_calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.bias.unsqueeze(0).expand(int(value.shape[0]), -1)

    model = CountingClassifier()
    soft_targets = torch.where(
        labels > 0.5,
        torch.full_like(labels, 0.95),
        torch.zeros_like(labels),
    )
    expected_soft = F.binary_cross_entropy_with_logits(
        model.bias.unsqueeze(0).expand(3, -1),
        soft_targets,
    )
    hard_bce = F.binary_cross_entropy_with_logits(
        model.bias.unsqueeze(0).expand(3, -1),
        labels,
    )
    with objective_scale.patch_online_trainer_runtime():
        objective = online_trainer._compute_objective(
            method=method,
            bundle=bundle,
            model=model,
            spec=EFFICIENTNET1DV2_SPEC,
            normalization_epsilon=1.0e-6,
            pos_weight=None,
        )

    assert model.forward_calls == 2
    torch.testing.assert_close(
        objective.raw_terms[auxiliary_term_name],
        expected_soft,
    )
    torch.testing.assert_close(
        objective.weighted_terms[auxiliary_term_name],
        0.25 * expected_soft,
    )
    assert not torch.isclose(expected_soft, hard_bce)
    objective.total.backward()
    assert model.bias.grad is not None
    assert bool(torch.isfinite(model.bias.grad).all())


def test_local_anchor_soft_contract_drift_fails_closed(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[
        "l8ri_effnet_e20_identity_vae_local_anchor_soft8"
    ]
    payload = controller._method_for(candidate, "ningbo")
    path = tmp_path / "local_anchor_soft.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(path)
    invalid = replace(
        method,
        contracts={
            **dict(method.contracts),
            "auxiliary_positive_value": 0.90,
        },
    )
    with pytest.raises(ValueError, match="auxiliary_positive_value drifted"):
        objective_scale.validate_scale_only_method(invalid)


@pytest.mark.parametrize("candidate_id", FEATURE_INVARIANCE_POLISH_IDS)
def test_feature_invariance_polish_is_label_free_matched_and_compilable(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)

    assert compiled.profile_name == (
        controller.VAE_FEATURE_INVARIANCE_POLISH_METHOD_ID
    )
    assert direct_compiled.profile_name == (
        controller.CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID
    )
    assert candidate.polish_auxiliary_target == "feature_invariance"
    assert candidate.epochs == candidate.scheduler_horizon_epochs == 8
    assert candidate.auxiliary_alpha == pytest.approx(0.25)
    assert [
        (term["id"], term["type"], term["weight"])
        for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", "multilabel_bce_with_logits", pytest.approx(1.0)),
        (
            "clean_preserve_bce",
            "multilabel_bce_with_logits",
            pytest.approx(0.5),
        ),
        (
            "clean_lhat_feature_cosine",
            "multilabel_bernoulli_jsd",
            pytest.approx(0.5),
        ),
    ]
    assert method["objective"]["terms"][-1]["views"] == [
        "clean_view",
        "lhat_view",
    ]
    assert direct["objective"]["terms"][-1] == {
        "id": "clean_identity_feature_cosine",
        "type": "multilabel_bernoulli_jsd",
        "views": ["clean_view", "lhat_control_view"],
        "weight": pytest.approx(0.5),
    }
    assert method["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "clean_lhat_feature_cosine": pytest.approx(0.25),
    }
    assert direct["contracts"]["effective_objective_weights"] == {
        "clean_bce": pytest.approx(0.5),
        "clean_preserve_bce": pytest.approx(0.25),
        "clean_identity_feature_cosine": pytest.approx(0.25),
    }
    expected_policy = {
        "space": "managed_classifier_penultimate_feature",
        "distance": "one_minus_cosine_similarity",
        "validity": "exact_matched_eligibility_intersection",
        "projection_head": False,
        "direct_lhat_label_supervision": False,
        "typed_objective_slot": (
            "bernoulli_jsd_replaced_before_objective_scaling"
        ),
        "matched_compute_in_control": True,
    }
    for payload in (method, direct):
        contracts = payload["contracts"]
        assert contracts["feature_invariance_polish"] == expected_policy
        assert {
            "auxiliary_label_policy",
            "auxiliary_teacher_policy",
            "hard_auxiliary_target_policy",
            "clean_lhat_consistency_policy",
        }.isdisjoint(contracts)
        assert set(contracts["post_refine_excluded_components"]) >= {
            "fixed20",
            "augmix",
            "raw_auxiliary",
            "vae_random",
            "source_replay",
        }
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_FEATURE_INVARIANCE_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_FEATURE_INVARIANCE_POLISH_METHOD_ID
    )
    assert "lhat_control_identity" in direct["nodes"]
    assert set(method["resources"]) == set(direct["resources"])

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1
        assert arguments[arguments.index("--vae-checkpoint") + 1] == str(
            controller.VAE_CHECKPOINT
        )


def test_feature_invariance_runtime_replaces_jsd_with_penultimate_cosine(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[
        "l7a_effnet_e20_augmix_vae_feature8"
    ]
    method_payload = controller._method_for(candidate, "ningbo")
    method_path = tmp_path / "feature_method.yaml"
    method_path.write_text(
        yaml.safe_dump(method_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(method_path)

    generator = torch.Generator().manual_seed(20260727)
    clean_waveform = torch.randn(3, 1000, 12, generator=generator)
    auxiliary_waveform = clean_waveform.clone()
    auxiliary_waveform[:, 0, :] += torch.linspace(-2.0, 2.0, 12)
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    sample_ids = ("a", "b", "c")
    clean = WaveformView(
        name="clean_identity",
        waveform=clean_waveform,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="clean_identity",
            operation="feature_runtime_test",
        ),
    )
    auxiliary = WaveformView(
        name="lhat",
        waveform=auxiliary_waveform,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="lhat",
            operation="feature_runtime_test",
        ),
    )
    bundle = ViewBundle(
        values={"clean_view": clean, "lhat_view": auxiliary},
        node_values={"clean_identity": clean, "lhat": auxiliary},
    )

    class TinyFeatureClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(12, 5, bias=False)
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.classifier(value[:, :, 0])

    model = TinyFeatureClassifier()
    clean_input = online_trainer.prepare_canonical_model_input(
        clean_waveform,
        EFFICIENTNET1DV2_SPEC,
        epsilon=1.0e-6,
    )
    auxiliary_input = online_trainer.prepare_canonical_model_input(
        auxiliary_waveform,
        EFFICIENTNET1DV2_SPEC,
        epsilon=1.0e-6,
    )
    expected = (
        1.0
        - F.cosine_similarity(
            clean_input[:, :, 0],
            auxiliary_input[:, :, 0],
            dim=1,
            eps=1.0e-8,
        )
    ).mean()

    with objective_scale.patch_online_trainer_runtime():
        objective = online_trainer._compute_objective(
            method=method,
            bundle=bundle,
            model=model,
            spec=EFFICIENTNET1DV2_SPEC,
            normalization_epsilon=1.0e-6,
            pos_weight=None,
        )

    term = "clean_lhat_feature_cosine"
    torch.testing.assert_close(objective.raw_terms[term], expected)
    torch.testing.assert_close(
        objective.weighted_terms[term],
        0.25 * expected,
    )
    assert objective.valid_counts[term] == 3
    assert expected.item() > 0.0


def test_feature_invariance_contract_drift_fails_closed(
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[
        "l7i_effnet_e20_identity_vae_feature8"
    ]
    method_payload = controller._method_for(candidate, "ningbo")
    method_path = tmp_path / "feature_method.yaml"
    method_path.write_text(
        yaml.safe_dump(method_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(method_path)
    policy = dict(method.contracts["feature_invariance_polish"])
    policy["projection_head"] = True
    invalid = replace(
        method,
        contracts={
            **dict(method.contracts),
            "feature_invariance_polish": policy,
        },
    )
    with pytest.raises(ValueError, match="feature-invariance contract drifted"):
        objective_scale.validate_scale_only_method(invalid)


def test_feature_invariance_completed_history_keeps_objective_terms_under_train(
) -> None:
    root = (
        controller.RUN_ROOT
        / "l7a_effnet_e20_augmix_vae_feature8"
    )
    for arm, term_name in (
        ("candidate", "clean_lhat_feature_cosine"),
        ("direct", "clean_identity_feature_cosine"),
    ):
        history_path = (
            root
            / arm
            / "ningbo"
            / "train"
            / "training"
            / "training_history.json"
        )
        history = json.loads(history_path.read_text(encoding="utf-8"))
        rows = history["epochs"]
        assert len(rows) == 8
        assert all(
            term_name in row["train"]["objective_terms"] for row in rows
        )
        assert all(term_name not in row for row in rows)


@pytest.mark.parametrize("candidate_id", CHAIN3_POLISH_IDS)
def test_vae_chain3_polish_is_matched_without_direct_lhat_bce(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, "ningbo")
    direct = controller._direct_method_for(candidate, "ningbo")
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    assert compiled.profile_name == controller.VAE_CHAIN3_POLISH_METHOD_ID
    assert direct_compiled.profile_name == (
        controller.CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID
    )
    objective_scale.validate_scale_only_method(compiled)
    objective_scale.validate_scale_only_method(direct_compiled)

    assert set(method["nodes"]) == {
        "clean_identity",
        "lhat",
        "threechain_augmix",
    }
    assert set(direct["nodes"]) == {
        "clean_identity",
        "chain3_control_identity",
        "threechain_augmix",
    }
    assert method["nodes"]["threechain_augmix"]["inputs"][
        "chain3_waveform"
    ] == "lhat_view"
    assert direct["nodes"]["threechain_augmix"]["inputs"][
        "chain3_waveform"
    ] == "chain3_control_view"
    assert direct["nodes"]["chain3_control_identity"]["inputs"] == {
        "waveform": "clean_raw"
    }
    assert "lhat_direct_bce" not in {
        term["id"] for term in method["objective"]["terms"]
    }
    assert method["contracts"]["effective_objective_weights"] == (
        controller.CHAIN3_POLISH_EFFECTIVE_WEIGHTS
    )
    assert direct["contracts"]["effective_objective_weights"] == (
        controller.CHAIN3_POLISH_EFFECTIVE_WEIGHTS
    )
    assert method["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": 0.75,
        "lhat_view": 0.125,
        "augmix_view": 0.125,
    }
    assert direct["contracts"]["batch_norm_objective_view_weights"] == {
        "clean_view": 0.75,
        "chain3_control_view": 0.125,
        "augmix_view": 0.125,
    }
    assert method["contracts"]["comparison_rng_identity"] == (
        direct["contracts"]["comparison_rng_identity"]
    )
    assert method["nodes"]["threechain_augmix"][
        "inputs"
    ]["rng"] == direct["nodes"]["threechain_augmix"]["inputs"]["rng"]
    assert method["resources"]["augmix_rng"] == direct["resources"]["augmix_rng"]
    assert set(method["resources"]) == {
        "model",
        "vae",
        "latent_pool",
        "vae_decoder",
        "lhat_config",
        "augmix_config",
        "lhat_rng",
        "augmix_rng",
    }
    assert set(direct["resources"]) == {
        "model",
        "augmix_config",
        "augmix_rng",
    }
    assert method["resources"]["lhat_config"]["path"].endswith(
        f"{candidate_id}/lhat.yaml"
    )
    generated = controller._lhat_config_for(candidate)
    assert generated["hull_attack"]["pgd_epsilon_l2_standardized"] == 8.0


def test_j4b_is_an_isolated_exact_j4a_bnmatched_rerun() -> None:
    center = "ningbo"
    j4a_id, j4b_id = CHAIN3_TEACHER04_POLISH_IDS
    j4a = controller.CANDIDATES[j4a_id]
    j4b = controller.CANDIDATES[j4b_id]

    j4a_candidate = dict(j4a.__dict__)
    j4b_candidate = dict(j4b.__dict__)
    assert j4a_candidate.pop("candidate_id") == j4a_id
    assert j4b_candidate.pop("candidate_id") == j4b_id
    assert j4b_candidate == j4a_candidate
    assert j4b.comparison_rng_identity == j4a.comparison_rng_identity

    paired_payloads = (
        (
            controller._method_for(j4a, center),
            controller._method_for(j4b, center),
        ),
        (
            controller._direct_method_for(j4a, center),
            controller._direct_method_for(j4b, center),
        ),
        (
            controller._online_for(j4a, center),
            controller._online_for(j4b, center),
        ),
        (
            controller._direct_training_config_for(j4a, center),
            controller._direct_training_config_for(j4b, center),
        ),
        (
            controller._candidate_experiments(j4a, center),
            controller._candidate_experiments(j4b, center),
        ),
        (
            controller._direct_experiments(j4a, center),
            controller._direct_experiments(j4b, center),
        ),
    )
    for j4a_payload, j4b_payload in paired_payloads:
        assert _normalize_j4_clone_payload(j4a_payload, j4a_id) == (
            _normalize_j4_clone_payload(j4b_payload, j4b_id)
        )

    assert controller._candidate_root(j4a) != controller._candidate_root(j4b)
    assert controller._persistent_root(j4a) != controller._persistent_root(j4b)
    assert controller._candidate_root(j4a).name == j4a_id
    assert controller._candidate_root(j4b).name == j4b_id
    assert controller._persistent_root(j4a).name == j4a_id
    assert controller._persistent_root(j4b).name == j4b_id

    j4a_vae = controller._vae_checkpoint_manifest_for(j4a)
    j4b_vae = controller._vae_checkpoint_manifest_for(j4b)
    assert j4a_vae is not None
    assert j4b_vae == j4a_vae
    assert j4b_vae["path"] == str(controller.VAE_CHECKPOINT)
    assert len(j4b_vae["sha256"]) == 64

    j4b_method = controller._method_for(j4b, center)
    j4b_direct = controller._direct_method_for(j4b, center)
    assert j4b_method["method"]["id"] == (
        controller.VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID
    )
    assert j4b_direct["method"]["id"] == (
        controller.CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID
    )
    for payload in (j4b_method, j4b_direct):
        assert payload["contracts"]["auxiliary_teacher_mix"] == 0.4
        assert payload["contracts"]["comparison_rng_identity"] == (
            j4a.comparison_rng_identity
        )
        assert {
            "vae",
            "latent_pool",
            "vae_decoder",
            "lhat_config",
            "lhat_rng",
        }.issubset(payload["resources"])


@pytest.mark.parametrize("candidate_id", CHAIN3_TEACHER04_POLISH_IDS)
def test_j4_teacher04_chain3_candidate_and_control_are_mechanically_matched(
    candidate_id: str,
    tmp_path: Path,
) -> None:
    center = "ningbo"
    candidate = controller.CANDIDATES[candidate_id]
    method = controller._method_for(candidate, center)
    direct = controller._direct_method_for(candidate, center)
    method_path = tmp_path / f"{candidate_id}.yaml"
    direct_path = tmp_path / f"{candidate_id}_direct.yaml"
    method_path.write_text(
        yaml.safe_dump(method, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    direct_path.write_text(
        yaml.safe_dump(direct, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    compiled = compile_method_profile(method_path)
    direct_compiled = compile_method_profile(direct_path)
    assert compiled.profile_name == (
        controller.VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID
    )
    assert direct_compiled.profile_name == (
        controller.CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID
    )
    assert compiled.profile_name != controller.VAE_CHAIN3_POLISH_METHOD_ID
    assert direct_compiled.profile_name != (
        controller.CLEAN_CHAIN3_POLISH_CONTROL_METHOD_ID
    )

    expected_teacher = {
        "auxiliary_label_policy": (
            "teacher_mix_times_stage2_initializer_probability_plus_"
            "hard_remainder"
        ),
        "auxiliary_teacher_policy": (
            "frozen_stage2_initializer_snapshot_before_first_optimizer_step"
        ),
        "auxiliary_teacher_mix": 0.4,
        "auxiliary_teacher_scope": "augmix_bce_only",
        "non_augmix_bce_hard_labels_preserved": True,
        "auxiliary_teacher_initializer_sha256": controller._sha256(
            controller._source_checkpoint(candidate, center)
        ),
    }
    for name, expected in expected_teacher.items():
        assert method["contracts"][name] == expected
        assert direct["contracts"][name] == expected
    assert len(
        method["contracts"]["auxiliary_teacher_initializer_sha256"]
    ) == 64

    assert candidate.initializer_candidate == "s01_effnet_ssl4096_augmix_e40"
    assert candidate.epochs == candidate.scheduler_horizon_epochs == 4
    assert candidate.learning_rate == pytest.approx(3.0e-5)
    assert candidate.batch_size == 64
    assert candidate.lhat_pgd_epsilon_l2_standardized == 8.0
    assert candidate.frozen_teacher_mix == pytest.approx(0.4)
    vae_manifest_record = controller._vae_checkpoint_manifest_for(candidate)
    assert vae_manifest_record is not None
    assert vae_manifest_record["path"] == str(controller.VAE_CHECKPOINT)
    assert vae_manifest_record["sha256"] == controller._sha256(
        controller.VAE_CHECKPOINT
    )
    assert method["contracts"]["effective_objective_weights"] == (
        controller.CHAIN3_POLISH_EFFECTIVE_WEIGHTS
    )
    assert direct["contracts"]["effective_objective_weights"] == (
        controller.CHAIN3_POLISH_EFFECTIVE_WEIGHTS
    )
    assert [
        (term["id"], term["weight"]) for term in method["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", 0.5),
        ("augmix_bce", 0.25),
        ("clean_lhat_augmix_jsd", 0.25),
    ]
    assert [
        (term["id"], term["weight"]) for term in direct["objective"]["terms"]
    ] == [
        ("clean_bce", 1.0),
        ("clean_preserve_bce", 0.5),
        ("augmix_bce", 0.25),
        ("clean_control_augmix_jsd", 0.25),
    ]

    required_vae_resources = {
        "vae",
        "latent_pool",
        "vae_decoder",
        "lhat_config",
        "lhat_rng",
    }
    assert required_vae_resources.issubset(method["resources"])
    assert required_vae_resources.issubset(direct["resources"])
    assert set(method["resources"]) == set(direct["resources"])
    assert "lhat" in method["nodes"]
    assert "lhat" in direct["nodes"]
    assert direct["contracts"][
        "unused_lhat_node_retained_for_latent_pool_construction"
    ] is True
    assert method["nodes"]["threechain_augmix"]["inputs"][
        "chain3_waveform"
    ] == "lhat_view"
    assert direct["nodes"]["threechain_augmix"]["inputs"][
        "chain3_waveform"
    ] == "chain3_control_view"
    assert direct["nodes"]["chain3_control_identity"]["inputs"] == {
        "waveform": "clean_raw"
    }
    assert method["contracts"]["matched_control_method_id"] == (
        controller.CLEAN_CHAIN3_TEACHER04_POLISH_CONTROL_METHOD_ID
    )
    assert direct["contracts"]["matched_candidate_method_id"] == (
        controller.VAE_CHAIN3_TEACHER04_POLISH_METHOD_ID
    )

    candidate_train, _ = controller._candidate_experiments(candidate, center)
    direct_train, _ = controller._direct_experiments(candidate, center)
    for experiment in (candidate_train, direct_train):
        arguments = [str(value) for value in experiment["entrypoint"]["arguments"]]
        assert arguments.count("--vae-checkpoint") == 1
        vae_index = arguments.index("--vae-checkpoint")
        assert arguments[vae_index + 1] == str(controller.VAE_CHECKPOINT)
        source_index = arguments.index("--source-checkpoint")
        assert arguments[source_index + 1] == str(
            controller._source_checkpoint(candidate, center)
        )


def test_j4_soft_teacher_runtime_freezes_and_routes_only_augmix_bce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = controller.CANDIDATES[CHAIN3_TEACHER04_POLISH_IDS[0]]
    method_payload = controller._method_for(candidate, "ningbo")
    method_path = tmp_path / "j4_method.yaml"
    method_path.write_text(
        yaml.safe_dump(method_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(method_path)
    initializer_sha256 = method.contracts[
        "auxiliary_teacher_initializer_sha256"
    ]
    initial_logits = torch.tensor(
        [-1.25, -0.25, 0.50, 1.25, 2.00],
        dtype=torch.float32,
    )

    class TinyStudent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(initial_logits.clone())
            self.checkpoint_identity = {"sha256": initializer_sha256}

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.logits.unsqueeze(0).expand(waveform.shape[0], -1)

    delegated_generate_observations: list[dict[str, Any]] = []
    generated_token = object()

    class FakeRuntime:
        def generate(self, **kwargs: Any) -> object:
            teacher = soft_teacher._STATE.teacher
            assert teacher is not None
            assert teacher is not kwargs["classifier"]
            assert teacher.training is False
            assert all(
                parameter.requires_grad is False
                for parameter in teacher.parameters()
            )
            delegated_generate_observations.append(
                {
                    "generate_calls": soft_teacher._STATE.generate_calls,
                    "teacher_logits": teacher.logits.detach().clone(),
                }
            )
            return generated_token

    runtime = FakeRuntime()
    original_generate = runtime.generate

    def fake_factory(*args: Any, **kwargs: Any) -> FakeRuntime:
        assert args[0] is method
        return runtime

    monkeypatch.setattr(
        online_trainer,
        "build_method_runtime",
        fake_factory,
    )
    restored_factory = online_trainer.build_method_runtime
    restored_objective = online_trainer._compute_objective
    torch_bce = F.binary_cross_entropy_with_logits
    resolved_bce_targets: list[torch.Tensor] = []

    def recording_bce(
        input: torch.Tensor,
        target: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        resolved_bce_targets.append(target.detach().clone())
        return torch_bce(input, target, *args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", recording_bce)
    restored_bce = F.binary_cross_entropy_with_logits

    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sample_ids = ("j4-sample-0", "j4-sample-1")
    clean_waveform = torch.zeros(2, 1000, 12, dtype=torch.float32)
    clean_waveform[:, :, 0] = torch.linspace(-1.0, 1.0, 1000)

    def waveform_view(name: str, waveform: torch.Tensor) -> WaveformView:
        return WaveformView(
            name=name,
            waveform=waveform,
            labels=labels,
            sample_ids=sample_ids,
            provenance=Provenance(
                node_id=name,
                operation="j4_cpu_runtime_test",
            ),
        )

    clean = waveform_view("clean_identity", clean_waveform)
    lhat = waveform_view("lhat", clean_waveform + 0.05)
    augmix = waveform_view("threechain_augmix", clean_waveform - 0.05)
    bundle = ViewBundle(
        values={
            "clean_view": clean,
            "lhat_view": lhat,
            "augmix_view": augmix,
        }
    )
    student = TinyStudent()

    with soft_teacher.patch_online_trainer_runtime():
        assert online_trainer.build_method_runtime is not restored_factory
        assert online_trainer._compute_objective is not restored_objective
        patched_runtime = online_trainer.build_method_runtime(
            method,
            model_name="efficientnet1dv2",
            config_root=tmp_path,
        )
        assert patched_runtime is runtime
        assert runtime.generate is not original_generate

        generated = runtime.generate(
            clean_raw=clean.waveform,
            targets=labels,
            hash_ids=sample_ids,
            classifier=student,
            base_seed=1042,
            rng_identity=sample_ids,
        )
        assert generated is generated_token
        assert len(delegated_generate_observations) == 1
        assert delegated_generate_observations[0]["generate_calls"] == 1
        assert torch.equal(
            delegated_generate_observations[0]["teacher_logits"],
            initial_logits,
        )

        teacher = soft_teacher._STATE.teacher
        assert teacher is not None
        assert torch.equal(teacher.logits.detach(), initial_logits)
        with torch.no_grad():
            student.logits.add_(7.0)
        assert not torch.equal(student.logits.detach(), initial_logits)
        assert torch.equal(teacher.logits.detach(), initial_logits)

        objective = online_trainer._compute_objective(
            method=method,
            bundle=bundle,
            model=student,
            spec=EFFICIENTNET1DV2_SPEC,
            normalization_epsilon=1.0e-6,
            pos_weight=None,
        )
        assert objective.total.ndim == 0
        assert F.binary_cross_entropy_with_logits is restored_bce
        assert len(resolved_bce_targets) == 3
        assert torch.equal(resolved_bce_targets[0], labels)
        assert torch.equal(resolved_bce_targets[1], labels)
        expected_augmix_target = (
            0.6 * labels
            + 0.4 * torch.sigmoid(initial_logits).unsqueeze(0).expand_as(labels)
        )
        torch.testing.assert_close(
            resolved_bce_targets[2],
            expected_augmix_target,
            rtol=0.0,
            atol=1.0e-7,
        )
        assert soft_teacher._STATE.soft_bce_calls == 1
        assert torch.equal(teacher.logits.detach(), initial_logits)

    assert online_trainer.build_method_runtime is restored_factory
    assert online_trainer._compute_objective is restored_objective
    assert F.binary_cross_entropy_with_logits is restored_bce
    assert runtime.generate.__self__ is original_generate.__self__
    assert runtime.generate.__func__ is original_generate.__func__


def test_k1_soft_teacher_runtime_uses_full_teacher_and_drops_pos_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = controller.CANDIDATES[TEACHER_DISTILL_POLISH_IDS[0]]
    payload = controller._method_for(candidate, "ningbo")
    path = tmp_path / "k1_method.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    method = compile_method_profile(path)
    initializer_sha256 = method.contracts[
        "auxiliary_teacher_initializer_sha256"
    ]
    initial_logits = torch.tensor(
        [-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float32
    )

    class TinyStudent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(initial_logits.clone())
            self.checkpoint_identity = {"sha256": initializer_sha256}

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.logits.unsqueeze(0).expand(waveform.shape[0], -1)

    class FakeRuntime:
        def generate(self, **kwargs: Any) -> object:
            return object()

    runtime = FakeRuntime()
    monkeypatch.setattr(
        online_trainer,
        "build_method_runtime",
        lambda *args, **kwargs: runtime,
    )
    torch_bce = F.binary_cross_entropy_with_logits
    resolved: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def recording_bce(
        input: torch.Tensor,
        target: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        pos_weight = kwargs.get("pos_weight")
        resolved.append(
            (
                target.detach().clone(),
                None if pos_weight is None else pos_weight.detach().clone(),
            )
        )
        return torch_bce(input, target, *args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", recording_bce)
    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sample_ids = ("k1-eligible", "k1-ineligible")
    waveform = torch.zeros(2, 1000, 12, dtype=torch.float32)
    waveform[:, :, 0] = torch.linspace(-1.0, 1.0, 1000)
    clean = WaveformView(
        name="clean_identity",
        waveform=waveform,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="clean_identity",
            operation="k1_cpu_runtime_test",
        ),
    )
    lhat = WaveformView(
        name="lhat",
        waveform=waveform + 0.05,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="lhat",
            operation="k1_cpu_runtime_test",
        ),
        valid_mask=torch.tensor([True, False]),
    )
    bundle = ViewBundle(values={"clean_view": clean, "lhat_view": lhat})
    student = TinyStudent()
    pos_weight = torch.full((5,), 7.0)

    with soft_teacher.patch_online_trainer_runtime():
        patched_runtime = online_trainer.build_method_runtime(
            method,
            model_name="efficientnet1dv2",
            config_root=tmp_path,
        )
        patched_runtime.generate(
            clean_raw=waveform,
            targets=labels,
            hash_ids=sample_ids,
            classifier=student,
            base_seed=1042,
            rng_identity=sample_ids,
        )
        with torch.no_grad():
            student.logits.add_(4.0)
        online_trainer._compute_objective(
            method=method,
            bundle=bundle,
            model=student,
            spec=EFFICIENTNET1DV2_SPEC,
            normalization_epsilon=1.0e-6,
            pos_weight=pos_weight,
        )
        diagnostics = soft_teacher.diagnostics_payload()

    assert len(resolved) == 3
    assert torch.equal(resolved[0][0], labels)
    assert torch.equal(resolved[0][1], pos_weight)
    assert torch.equal(resolved[1][0], labels)
    assert torch.equal(resolved[1][1], pos_weight)
    expected_soft = torch.sigmoid(initial_logits).unsqueeze(0)
    torch.testing.assert_close(
        resolved[2][0],
        expected_soft,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert resolved[2][1] is None
    assert diagnostics is not None
    assert diagnostics["teacher_mix"] == 1.0
    assert diagnostics["generate_calls"] == 1
    assert diagnostics["objective_calls"] == 1
    assert diagnostics["soft_bce_calls"] == 1


def test_snapshot_evidence_fails_closed_on_stale_or_ambiguous_snapshot(
    tmp_path: Path,
) -> None:
    config = tmp_path / "method.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    snapshot = {
        "role": "referenced_config",
        "source_path": str(config),
        "sha256": controller._sha256(config),
        "snapshot_path": "configs/method.yaml",
    }
    run_record = {"config_snapshots": [snapshot]}

    evidence = controller._snapshot_evidence(
        run_record,
        config,
        role="referenced_config",
    )
    assert evidence["sha256"] == controller._sha256(config)
    assert evidence["snapshot_path"] == "configs/method.yaml"

    stale = copy.deepcopy(run_record)
    stale["config_snapshots"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime snapshot is stale"):
        controller._snapshot_evidence(
            stale,
            config,
            role="referenced_config",
        )

    ambiguous = {"config_snapshots": [snapshot, copy.deepcopy(snapshot)]}
    with pytest.raises(ValueError, match="expected one referenced_config snapshot"):
        controller._snapshot_evidence(
            ambiguous,
            config,
            role="referenced_config",
        )


@pytest.mark.parametrize(
    "candidate_id",
    (
        *STAGE2_IDS,
        *CLEAN_POLISH_IDS,
        *CHAIN3_POLISH_IDS,
        *CHAIN3_TEACHER04_POLISH_IDS,
        *LHAT_PATH_POLISH_IDS,
        *FEATURE_INVARIANCE_POLISH_IDS,
        *LOCAL_ANCHOR_SOFT_POLISH_IDS,
    ),
)
def test_stage2_direct_and_candidate_share_online_training_history(
    candidate_id: str,
) -> None:
    candidate = controller.CANDIDATES[candidate_id]
    candidate_online = controller._online_for(candidate, "ningbo")
    direct_online = controller._direct_training_config_for(candidate, "ningbo")

    assert direct_online == candidate_online
    assert direct_online["training"] == candidate_online["training"]
    assert direct_online["random_seed"] == candidate_online["random_seed"]
    assert direct_online["diagnostics"] == candidate_online["diagnostics"]
    assert direct_online["logging"] == candidate_online["logging"]


def _write_stage2_pair_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift: str | None = None,
) -> tuple[controller.Candidate, str]:
    candidate = controller.CANDIDATES["s02_effnet_identity_vae_refine8"]
    center = "ningbo"
    run_root = tmp_path / "runs"
    config_root = tmp_path / "configs"
    monkeypatch.setattr(controller, "RUN_ROOT", run_root)
    monkeypatch.setattr(controller, "CONFIG_ROOT", config_root)

    candidate_root = controller._candidate_root(candidate)
    online = candidate_root / "online" / f"{center}.yaml"
    direct_train = candidate_root / "direct_train" / f"{center}.yaml"
    online.parent.mkdir(parents=True, exist_ok=True)
    direct_train.parent.mkdir(parents=True, exist_ok=True)
    shared_training_config = "profile_name: sequential_stage2_test\n"
    online.write_text(shared_training_config, encoding="utf-8")
    direct_train.write_text(shared_training_config, encoding="utf-8")

    steps_per_epoch = 8
    digest = "a" * 64
    exposure = {
        "base_record_count": 500,
        "clean_count": 500,
        "corrupted_count": 10_000,
        "total_count": 10_500,
        "view_executions_per_base_batch": 21,
        "optimizer_steps_per_base_batch": 1,
        "optimizer_steps_this_epoch": steps_per_epoch,
    }
    histories: dict[str, dict[str, Any]] = {}
    for arm in ("direct", "candidate"):
        rows = [
            {
                "epoch": epoch,
                "learning_rate": 3.0e-5,
                "exposure": copy.deepcopy(exposure),
                "stochastic_trace": {
                    "input_identity_sha256": digest,
                    "input_record_count": 10_500,
                },
            }
            for epoch in range(1, candidate.epochs + 1)
        ]
        histories[arm] = {
            "schema_version": 2,
            "selection": "last",
            "optimizer_steps": candidate.epochs * steps_per_epoch,
            "epochs": rows,
        }

    if drift == "exposure":
        for history in histories.values():
            history["epochs"][0]["exposure"]["total_count"] = 10_499
    elif drift == "input_digest":
        histories["candidate"]["epochs"][0]["stochastic_trace"][
            "input_identity_sha256"
        ] = "b" * 64

    for arm, history in histories.items():
        training = (
            controller._run_dir(candidate, arm, center, "train") / "training"
        )
        training.mkdir(parents=True, exist_ok=True)
        (training / "training_history.json").write_text(
            json.dumps(history),
            encoding="utf-8",
        )

    expected_steps = candidate.epochs * steps_per_epoch
    diagnostics = {
        "schema_version": 3,
        "mode": pcgrad.VAE_POST_MODE,
        "modes": [pcgrad.VAE_POST_MODE],
        "step_count": expected_steps,
        "pending_auxiliary_at_exit": False,
        "steps": [{"step": step} for step in range(expected_steps)],
    }
    if drift == "step_count":
        diagnostics["step_count"] = expected_steps - 1
    diagnostics_path = (
        controller._run_dir(candidate, "candidate", center, "train")
        / "training"
        / "pcgrad_diagnostics.json"
    )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    return candidate, center


def test_stage2_optimizer_pair_accepts_complete_matched_synthetic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, center = _write_stage2_pair_fixture(tmp_path, monkeypatch)

    evidence = controller._assert_stage2_optimizer_pair(candidate, center)

    assert evidence["optimizer_steps"] == 64
    assert len(evidence["epochs"]) == 8
    assert all(
        row["optimizer_steps_this_epoch"] == 8
        and row["input_record_count"] == 10_500
        and row["direct_input_identity_sha256"] == "a" * 64
        for row in evidence["epochs"]
    )


@pytest.mark.parametrize("drift", ("exposure", "input_digest", "step_count"))
def test_stage2_optimizer_pair_fails_closed_on_evidence_drift(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, center = _write_stage2_pair_fixture(
        tmp_path,
        monkeypatch,
        drift=drift,
    )

    with pytest.raises(ValueError):
        controller._assert_stage2_optimizer_pair(candidate, center)


def _write_snapshot_closure_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], Path]:
    source = tmp_path / "source" / "method.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("schema_version: 1\n", encoding="utf-8")
    source_sha256 = controller._sha256(source)

    run_root = tmp_path / "run"
    snapshot_relative = Path("configs/method.yaml")
    artifact = run_root / snapshot_relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(source.read_bytes())
    size_bytes = artifact.stat().st_size

    index_path = run_root / "run_file_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": snapshot_relative.as_posix(),
                        "role": "config_snapshot",
                        "sha256": source_sha256,
                        "size_bytes": size_bytes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_record = {
        "run_file_index_sha256": controller._sha256(index_path),
        "config_snapshots": [
            {
                "role": "referenced_config",
                "source_path": str(source),
                "sha256": source_sha256,
                "size_bytes": size_bytes,
                "snapshot_path": snapshot_relative.as_posix(),
            }
        ],
    }
    return source, run_root, run_record, artifact


def test_snapshot_artifact_hash_size_and_file_index_close_the_loop(
    tmp_path: Path,
) -> None:
    source, run_root, run_record, artifact = _write_snapshot_closure_fixture(
        tmp_path
    )
    file_index, index_evidence = controller._run_file_index_evidence(
        run_root,
        run_record,
    )

    evidence = controller._snapshot_evidence(
        run_record,
        source,
        role="referenced_config",
        snapshot_root=run_root,
        file_index=file_index,
    )

    assert evidence["sha256"] == controller._sha256(source)
    assert evidence["snapshot_path"] == "configs/method.yaml"
    assert controller._sha256(artifact) == evidence["sha256"]
    assert index_evidence["sha256"] == run_record["run_file_index_sha256"]


@pytest.mark.parametrize("drift", ("artifact_hash", "size", "index"))
def test_snapshot_artifact_closure_fails_closed_on_drift(
    drift: str,
    tmp_path: Path,
) -> None:
    source, run_root, run_record, artifact = _write_snapshot_closure_fixture(
        tmp_path
    )
    index_path = run_root / "run_file_index.json"
    if drift == "artifact_hash":
        artifact.write_text("schema_version: 2\n", encoding="utf-8")
    elif drift == "size":
        run_record["config_snapshots"][0]["size_bytes"] += 1
    elif drift == "index":
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        payload["files"][0]["role"] = "delegate_artifact"
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        run_record["run_file_index_sha256"] = controller._sha256(index_path)

    file_index, _ = controller._run_file_index_evidence(run_root, run_record)
    with pytest.raises(ValueError):
        controller._snapshot_evidence(
            run_record,
            source,
            role="referenced_config",
            snapshot_root=run_root,
            file_index=file_index,
        )


def test_source_floor_parser_exposes_controller_and_worker_contracts() -> None:
    controller_args = controller.build_parser().parse_args(
        [
            "source-floor",
            "--candidate",
            DIVERSE_VAE_POSTTRAIN_ID,
            "--arm",
            "candidate",
            "--gpus",
            "4",
            "5",
            "6",
            "7",
            "--cpu-threads",
            "2",
        ]
    )
    worker_args = controller.build_parser().parse_args(
        [
            "source-floor-worker",
            "--candidate",
            DIVERSE_VAE_POSTTRAIN_ID,
            "--arm",
            "candidate",
            "--center",
            "ningbo",
        ]
    )

    assert controller_args.gpus == [4, 5, 6, 7]
    assert controller_args.cpu_threads == 2
    assert worker_args.center == "ningbo"


def test_source_floor_summary_applies_predeclared_absolute_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = controller.CANDIDATES[DIVERSE_VAE_POSTTRAIN_ID]
    monkeypatch.setattr(controller, "RUN_ROOT", tmp_path / "runs")
    baseline = {
        "sample_count": 2158,
        "macro_auroc": 0.90,
        "macro_auprc": 0.77,
    }
    for index, center in enumerate(controller.CENTERS):
        output = controller._source_floor_output(
            candidate,
            "candidate",
            center,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        auroc = 0.895 - index * 0.001
        auprc = 0.765 - index * 0.001
        output.write_text(
            json.dumps(
                {
                    "checkpoint": {"sha256": str(index) * 64},
                    "metrics": {
                        "macro_auroc": auroc,
                        "macro_auprc": auprc,
                    },
                    "locked_source_baseline": baseline,
                    "delta": {
                        "macro_auroc": auroc - baseline["macro_auroc"],
                        "macro_auprc": auprc - baseline["macro_auprc"],
                    },
                }
            ),
            encoding="utf-8",
        )

    output = controller.summarize_source_floor(candidate, arm="candidate")
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert summary["predeclared_absolute_retention_rule"]["passed"] is True
    assert summary["four_center_mean"]["macro_auprc"] == pytest.approx(0.7635)
    assert set(summary["records"]) == set(controller.CENTERS)
