"""Small CPU-only helpers for resume-safe training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecg_adv_gen.matched_effnet import (
    F004_FROZEN_TOPOLOGY,
    F004_RHO_SWEEP_PROTOCOL,
    f004_identity,
    validate_f004_kshot_identity,
)


RESUME_CONTRACT_KEYS = (
    "center_name",
    "ref_meta_json",
    "synth_npz",
    "target_real_npz",
    "init_ckpt",
    "model_name",
    "comparison_arm",
    "comparison_protocol",
    "comparison_variant",
    "comparison_topology_version",
    "comparison_topology_sha256",
    "study_scope",
    "mechanism_variant",
    "eligibility_manifest_sha256",
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
    "hull_neighbor_pool_multiplier",
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
    "latent_augmix_copies",
    "latent_augmix_adv_base_mix",
    "latent_augmix_alpha",
    "latent_augmix_severity",
    "latent_augmix_severity_profile",
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
    f004_current = current_args.get("comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
    f004_required = {
        "comparison_protocol", "comparison_variant", "comparison_topology_version",
        "comparison_topology_sha256", "target_adv_fraction",
        "seed", "ref_meta_json", "synth_npz", "target_real_npz",
        *F004_FROZEN_TOPOLOGY,
    }
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
            if f004_current and key in f004_required:
                mismatches.append({
                    "key": key,
                    "saved": None,
                    "current": normalize_resume_contract_value(current_args[key]),
                })
                continue
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


def validate_f004_checkpoint_identity(
    checkpoint: Any, current_args: dict[str, Any]
) -> dict[str, Any] | None:
    """Hard-gate F-004 top-level and args identity, independent of drift overrides."""

    current_f004 = current_args.get("comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    top_identity = checkpoint.get("comparison_identity") if isinstance(checkpoint, dict) else None
    checkpoint_f004 = (
        isinstance(saved_args, dict)
        and saved_args.get("comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
    ) or (
        isinstance(top_identity, dict)
        and top_identity.get("comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
    )
    if not current_f004:
        if checkpoint_f004:
            raise ValueError("F-004 checkpoint identity cannot resume into a non-F004 run")
        return None
    try:
        expected = f004_identity(current_args["target_adv_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"F-004 checkpoint identity current command is incomplete: {exc}") from exc
    current_projection = {
        "comparison_protocol": current_args.get("comparison_protocol"),
        "variant": current_args.get("comparison_variant"),
        "rho": current_args.get("target_adv_fraction"),
        "topology_version": current_args.get("comparison_topology_version"),
        "topology_sha256": current_args.get("comparison_topology_sha256"),
    }
    expected_projection = {
        key: expected[key]
        for key in (
            "comparison_protocol", "variant", "rho", "topology_version", "topology_sha256"
        )
    }
    if current_projection != expected_projection:
        raise ValueError(
            "F-004 checkpoint identity current command mismatch: "
            f"{current_projection!r} != {expected_projection!r}"
        )
    if top_identity != expected:
        raise ValueError(
            f"F-004 checkpoint identity top-level mismatch: {top_identity!r} != {expected!r}"
        )
    if not isinstance(saved_args, dict):
        raise ValueError("F-004 checkpoint identity args mapping is missing")
    args_projection = {
        "comparison_protocol": saved_args.get("comparison_protocol"),
        "variant": saved_args.get("comparison_variant"),
        "rho": saved_args.get("target_adv_fraction"),
        "topology_version": saved_args.get("comparison_topology_version"),
        "topology_sha256": saved_args.get("comparison_topology_sha256"),
    }
    if args_projection != expected_projection:
        raise ValueError(
            "F-004 checkpoint identity args mismatch: "
            f"{args_projection!r} != {expected_projection!r}"
        )
    hard_runtime_keys = (
        "seed",
        "ref_meta_json",
        "synth_npz",
        "target_real_npz",
        *F004_FROZEN_TOPOLOGY,
    )
    for layer, args in (("current", current_args), ("saved args", saved_args)):
        missing = [key for key in hard_runtime_keys if key not in args]
        if missing:
            raise ValueError(
                f"F-004 checkpoint identity {layer} missing hard runtime fields: {missing}"
            )
    current_hard = {
        key: normalize_resume_contract_value(current_args[key])
        for key in hard_runtime_keys
    }
    saved_hard = {
        key: normalize_resume_contract_value(saved_args[key])
        for key in hard_runtime_keys
    }
    runtime_drift = {
        key: {"saved": saved_hard[key], "current": current_hard[key]}
        for key in hard_runtime_keys
        if saved_hard[key] != current_hard[key]
    }
    if runtime_drift:
        raise ValueError(
            f"F-004 checkpoint identity hard runtime mismatch: {runtime_drift}"
        )
    expected_topology = {
        key: normalize_resume_contract_value(value)
        for key, value in F004_FROZEN_TOPOLOGY.items()
    }
    topology_drift = {
        key: {"current": current_hard[key], "expected": expected}
        for key, expected in expected_topology.items()
        if current_hard[key] != expected
    }
    if topology_drift:
        raise ValueError(
            f"F-004 checkpoint identity current frozen topology mismatch: {topology_drift}"
        )
    for path_key in ("ref_meta_json", "synth_npz", "target_real_npz"):
        try:
            validate_f004_kshot_identity(
                kshot_seed=int(current_hard["seed"]),
                kshot_path=str(current_hard[path_key]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"F-004 checkpoint identity {path_key} mismatch: {exc}"
            ) from exc
    return expected
