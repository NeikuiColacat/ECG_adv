"""Small CPU-only helpers for resume-safe training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


RESUME_CONTRACT_KEYS = (
    "center_name",
    "ref_meta_json",
    "synth_npz",
    "target_real_npz",
    "init_ckpt",
    "model_name",
    "comparison_arm",
    "enable_vae_lhat",
    "enable_raw_augmix",
    "hull_M",
    "hull_lambda",
    "hull_steps",
    "hull_lr",
    "hull_init_logit_gap",
    "hull_label_mode",
    "hull_mix_label_mode",
    "hull_label_lambda_y",
    "hull_include_anchor",
    "hull_neighbor_distance_space",
    "hull_neighbor_mode",
    "hull_neighbor_pool_size",
    "pgd_eps",
    "asr_low_threshold",
    "asr_high_threshold",
    "target_adv_fraction",
    "enable_latent_augmix_consistency",
    "final_checkpoint_only",
    "adv_label_mode",
    "adv_teacher_mix",
    "latent_augmix_width",
    "latent_augmix_depth",
    "latent_augmix_severity",
    "latent_augmix_third_chain_role",
    "latent_augmix_chain_base_mode",
    "latent_augmix_ops",
    "latent_augmix_chain_weights",
    "latent_augmix_consistency_weight",
    "latent_augmix_consistency_loss",
    "latent_augmix_bce_weight",
    "vae_adv_consistency_weight",
    "latent_augmix_signal_space",
    "classes_in_scope",
    "seed",
    "crop_len",
)
LOCKED_ATTACK_MODE = "latent_hull"
LOCKED_LATENT_AUGMIX_SIGNAL_SPACE = "raw_pre_zscore"
LOCKED_LEGACY_ARGS = {
    "enable_latent_augmix_branch": True,
    "enable_latent_augmix_consistency": True,
    "latent_augmix_topology": "locked_three_chain",
    "latent_augmix_mixture_mode": "beta",
    "latent_augmix_mixture_prob": 0.5,
    "latent_augmix_mixture_beta_a": 0.0,
    "latent_augmix_mixture_beta_b": 0.0,
    "latent_augmix_op_schedule": "random",
    "latent_augmix_corruption_source": "vae_decode",
    "latent_augmix_severity_params_file": "",
    "latent_augmix_severity_params_name": "",
    "no_latent_augmix_renorm": False,
    "latent_augmix_clip_abs": 6.0,
}


def normalize_resume_contract_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [normalize_resume_contract_value(v) for v in value]
    if isinstance(value, list):
        return [normalize_resume_contract_value(v) for v in value]
    return value


def resume_contract_mismatches(
    saved_args: dict[str, Any],
    current_args: dict[str, Any],
    *,
    keys: tuple[str, ...] = RESUME_CONTRACT_KEYS,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if "attack_mode" in saved_args:
        saved_attack_mode = normalize_resume_contract_value(saved_args["attack_mode"])
        if saved_attack_mode != LOCKED_ATTACK_MODE:
            mismatches.append({
                "key": "attack_mode",
                "saved": saved_attack_mode,
                "current": LOCKED_ATTACK_MODE,
            })
    for key, current in LOCKED_LEGACY_ARGS.items():
        if key not in saved_args:
            continue
        saved = normalize_resume_contract_value(saved_args[key])
        if saved != current:
            mismatches.append({"key": key, "saved": saved, "current": current})
    for key in keys:
        if key not in current_args:
            continue
        if key not in saved_args:
            if (
                key == "latent_augmix_signal_space"
                and current_args[key] == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
            ):
                mismatches.append({"key": key, "saved": None, "current": current_args[key]})
            continue
        saved = normalize_resume_contract_value(saved_args[key])
        current = normalize_resume_contract_value(current_args[key])
        if saved != current:
            mismatches.append({"key": key, "saved": saved, "current": current})
    return mismatches


def validate_resume_contract(
    saved_args: Any,
    current_args: dict[str, Any],
    *,
    allow_drift: bool,
) -> list[dict[str, Any]]:
    """Validate critical resume args and return mismatches when allowed."""
    if not isinstance(saved_args, dict):
        if allow_drift:
            return [{"key": "<missing_args>", "saved": None, "current": "<present>"}]
        raise ValueError("resume checkpoint has no args contract; rerun with --allow_resume_config_drift to override")
    mismatches = resume_contract_mismatches(saved_args, current_args)
    if not mismatches or allow_drift:
        return mismatches
    preview = mismatches[:8]
    raise ValueError(
        "resume checkpoint args disagree with current command on critical keys: "
        f"{preview}. Use --allow_resume_config_drift only for intentional recovery."
    )
