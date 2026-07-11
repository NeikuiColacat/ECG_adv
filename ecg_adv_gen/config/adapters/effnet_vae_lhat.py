"""Audit adapter for YAML-managed EfficientNet VAE-LHAT commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ecg_adv_gen.f004_contract import project_f004_adapter_config
from ecg_adv_gen.matched_effnet import (
    F004_RHO_SWEEP_PROTOCOL,
    F004_FROZEN_TOPOLOGY_SHA256,
    MATCHED_EFFNET_CONTRACT_VERSION,
    MATCHED_EFFNET_NON_VAE_HULL_LABEL_MODE,
    f004_variant_for_rho,
    is_matched_effnet_arm,
    matched_effnet_arm,
    validate_f004_runtime,
    validate_matched_effnet_case,
    validate_matched_effnet_runtime,
)
from ecg_adv_gen.f005_control import F005_STUDY_SCOPE, validate_f005_case

from .common import (
    argv_option_map,
    audit_equals,
    audit_require_options,
    matrix_case,
    opt_first,
)


def _append_optional_value(argv: list[Any], option: str, value: Any) -> None:
    if value is not None:
        argv.extend([option, value])


def _append_optional_sequence(argv: list[Any], option: str, values: Any) -> None:
    if values:
        argv.append(option)
        argv.extend(list(values))


def _resolve_arm(case: Mapping[str, Any] | None, fallback: str):
    if case is not None:
        return validate_matched_effnet_case(case)
    return fallback, matched_effnet_arm(fallback) if is_matched_effnet_arm(fallback) else None


def build_effnet_vae_lhat_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build managed argv for the EfficientNet VAE-LHAT runner from typed config fields."""

    matrix = context.get("matrix") or {}
    case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else None
    center = matrix.get("center")
    if not center:
        raise ValueError("effnet_vae_lhat adapter requires runner.matrix.center")

    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    selection = paper["selection"]
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    adaptation = config["adaptation"]
    evaluation = config["evaluation"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]

    k = kshot["k"]
    seed = int(case.get("seed", kshot["seed"])) if case is not None else kshot["seed"]
    comparison_protocol = str(paper.get("comparison_protocol") or "")
    is_f004 = comparison_protocol == F004_RHO_SWEEP_PROTOCOL
    out_root = f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}"
    if is_f004:
        out_root = f"{out_root}/runs/{center}"
    kshot_subset_root = data.get("kshot_subset_root") or f"{paths['data_root']}/paper_vae_only_latenthull_sweep_20260516/subsets"
    init_ckpt = model["init_checkpoint"]
    anchor_base = (
        f"{kshot_subset_root}/{center}/k{k}_seed{seed}/"
        f"{center}_real_k{k}_seed{seed}"
    )
    target_real_npz_override = str(data.get("target_real_npz_override") or "")
    target_real_npz_suffix = str(data.get("target_real_npz_suffix") or "")
    if not target_real_npz_override and target_real_npz_suffix:
        suffix = target_real_npz_suffix if target_real_npz_suffix.startswith(".") else f".{target_real_npz_suffix}"
        target_real_npz_override = f"{anchor_base}{suffix}"
    synth_npz_override = str(data.get("synth_npz_override") or "")
    if synth_npz_override and not Path(synth_npz_override).is_absolute():
        synth_npz_override = str(Path(paths["data_root"]) / synth_npz_override)
    latent_augmix = adaptation["latent_augmix"]
    latent_augmix_consistency = latent_augmix["consistency"]
    anchors = adaptation["anchors"]
    hull = adaptation["hull"]
    attack = adaptation["attack"]
    asr_low, asr_high = attack["target_asr_range"]
    if not 0.0 <= float(asr_low) <= float(asr_high) <= 1.0:
        raise ValueError("adaptation.attack.target_asr_range must be ordered within [0, 1]")
    study_scope = ""
    mechanism_variant = ""
    resolved_hull_lambda = hull["lambda"]
    resolved_hull_label_mode = hull["label_mode"]
    resolved_hull_include_anchor = bool(hull["include_anchor"])
    if is_f004:
        comparison_arm, arm_components = "historical_unmatched", matched_effnet_arm("a5")
        target_adv_fraction = float(matrix["rho"])
        comparison_variant = f004_variant_for_rho(target_adv_fraction)
        declared_topology_sha256 = str(paper.get("topology_sha256") or "")
        validate_f004_runtime(
            comparison_protocol=comparison_protocol,
            comparison_arm=comparison_arm,
            comparison_variant=comparison_variant,
            comparison_topology_sha256=declared_topology_sha256,
            target_adv_fraction=target_adv_fraction,
            kshot_seed=seed,
            kshot_path=kshot_subset_root,
            behavior_projection=project_f004_adapter_config(
                config,
                comparison_arm=comparison_arm,
                comparison_protocol=comparison_protocol,
                seed=seed,
            ),
        )
    else:
        comparison_arm, arm_components = _resolve_arm(
            case, matrix.get("comparison_arm", adaptation.get("comparison_arm", "historical_unmatched"))
        )
        if case is not None and case.get("study_scope"):
            mechanism_variant, seed, study_variant = validate_f005_case(case)
            study_scope = F005_STUDY_SCOPE
            resolved_hull_lambda = study_variant.hull_lambda
            resolved_hull_label_mode = study_variant.label_mode
            resolved_hull_include_anchor = study_variant.include_anchor
        target_adv_fraction = float(
            matrix.get("target_adv_fraction", adaptation["loss"]["target_adv_fraction"])
        )
        comparison_variant = ""
    if target_adv_fraction not in {0.0, 0.25, 0.5}:
        raise ValueError("target_adv_fraction matrix value must be one of 0, 0.25, 0.5")
    if case is not None:
        target_adv_fraction = arm_components.target_adv_fraction
    elif arm_components is not None and not arm_components.vae_lhat:
        target_adv_fraction = arm_components.target_adv_fraction
    if (
        arm_components is not None
        and is_matched_effnet_arm(comparison_arm)
        and not arm_components.vae_lhat
    ):
        resolved_hull_label_mode = MATCHED_EFFNET_NON_VAE_HULL_LABEL_MODE
    third_chain_route = (
        arm_components.third_chain_route
        if arm_components is not None
        else latent_augmix.get("third_chain_role", "vae_lhat_adversarial_waveform")
    )
    bce_weight = latent_augmix_consistency.get("bce_weight")
    consistency_weight = latent_augmix_consistency.get("consistency_weight")
    if arm_components is not None:
        bce_weight = bce_weight if arm_components.augmix_view_bce else 0.0
        consistency_weight = consistency_weight if arm_components.jsd else 0.0
    if is_matched_effnet_arm(comparison_arm):
        validate_matched_effnet_runtime(
            comparison_arm,
            enable_vae_lhat=arm_components.vae_lhat,
            enable_raw_augmix=arm_components.raw_augmix,
            enable_auxiliary_steps=True,
            bce_weight=bce_weight,
            jsd_weight=consistency_weight,
            third_chain_route=third_chain_route,
            hull_label_mode=resolved_hull_label_mode,
            hull_include_anchor=resolved_hull_include_anchor,
            target_adv_fraction=target_adv_fraction,
        )

    argv: list[Any] = [
        "--center",
        center,
        "--comparison_arm",
        comparison_arm,
        *(["--study_scope", study_scope, "--mechanism_variant", mechanism_variant]
          if study_scope else []),
        "--epochs",
        training["epochs"],
        "--seed",
        seed,
        "--device",
        "cuda",
        "--num_workers",
        training["num_workers"],
        "--data_root",
        paths["data_root"],
        "--out_root",
        out_root,
        "--init_ckpt",
        init_ckpt,
        "--init_checkpoint_sha256",
        (model.get("init_lineage") or {}).get("checkpoint_sha256", ""),
        "--init_lineage_stage",
        (model.get("init_lineage") or {}).get("stage", ""),
        "--anchor_base",
        anchor_base,
        "--hull_steps",
        hull["steps"],
        "--hull_M",
        hull["M"],
        "--hull_lambda",
        resolved_hull_lambda,
        "--hull_lr",
        hull["lr"],
        "--hull_init_logit_gap",
        hull["init_logit_gap"],
        "--hull_label_mode",
        resolved_hull_label_mode,
        "--hull_mix_label_mode",
        hull["mix_label_mode"],
        "--hull_label_lambda_y",
        hull["label_lambda_y"],
        "--hull_label_new_class_cap",
        hull["label_new_class_cap"],
        "--hull_neighbor_distance_space",
        hull["neighbor_distance_space"],
        "--hull_neighbor_mode",
        hull["neighbor_mode"],
        "--hull_neighbor_pool_size",
        hull["neighbor_pool_size"],
        "--hull_neighbor_pool_multiplier",
        hull["neighbor_pool_multiplier"],
        "--k_anchor",
        anchors["k_anchor"],
        "--pgd_eps",
        attack["pgd_eps"],
        "--pgd_batch",
        attack["pgd_batch"],
        "--asr_low_threshold",
        asr_low,
        "--asr_high_threshold",
        asr_high,
        "--target_real_weight",
        adaptation["loss"]["target_real_weight"],
        "--target_adv_fraction",
        target_adv_fraction,
        "--adv_weight",
        adaptation["loss"]["adv_weight"],
        "--vae_adv_stream_sample_scale",
        adaptation["loss"].get("vae_adv_stream_sample_scale", 1.0),
        "--vae_adv_consistency_weight",
        adaptation["loss"].get("vae_adv_consistency_weight", 0.0),
        "--adv_weight_warmup_epochs",
        adaptation["loss"]["adv_weight_warmup_epochs"],
        "--adv_label_mode",
        adaptation["loss"]["label_mode"],
        "--adv_teacher_mix",
        adaptation["loss"]["teacher_mix"],
        "--lr",
        training["optimizer"]["lr"],
        "--train_batch_size",
        training["batch_size"],
        "--ptbxl_weight",
        adaptation["loss"].get("ptbxl_weight", "1.0"),
        "--target_real_val_fraction",
        selection.get("validation_fraction", 0.2),
        "--target_real_val_seed",
        seed if study_scope else selection.get("seed", kshot["seed"]),
        "--selection_metric",
        selection.get("metric", "macro_auprc"),
        "--source_floor_max_drop",
        (selection.get("source_floor") or {}).get("max_drop", 0.02),
        "--latent_augmix_width",
        latent_augmix["width"],
        "--latent_augmix_depth",
        latent_augmix["depth"],
        "--latent_augmix_alpha",
        latent_augmix["alpha"],
        "--latent_augmix_severity",
        latent_augmix["severity"],
        "--latent_augmix_third_chain_role",
        third_chain_route,
        "--latent_augmix_chain_base_mode",
        latent_augmix.get("chain_base_mode", "clean_clean_third"),
        "--latent_augmix_adv_base_mix",
        latent_augmix.get("adv_base_mix", 1.0),
        "--eval_batch_size",
        training["eval_batch_size"],
        "--eval_min_pos",
        evaluation["min_pos"],
        "--eval_pn2021_limit",
        evaluation["pn2021_limit"],
    ]
    if is_f004:
        argv[4:4] = [
            "--comparison_protocol", comparison_protocol,
            "--comparison_variant", comparison_variant,
            "--comparison_topology_version", MATCHED_EFFNET_CONTRACT_VERSION,
            "--comparison_topology_sha256", F004_FROZEN_TOPOLOGY_SHA256,
        ]
    if resolved_hull_include_anchor:
        argv.append("--hull_include_anchor")
    if arm_components is not None:
        argv.append("--enable_vae_lhat" if arm_components.vae_lhat else "--disable_vae_lhat")
        argv.append("--enable_raw_augmix" if arm_components.raw_augmix else "--disable_raw_augmix")
    argv.append(
        "--enable_latent_augmix_consistency"
        if arm_components is not None or bool(latent_augmix_consistency["enabled"])
        else "--disable_latent_augmix_consistency"
    )
    _append_optional_value(argv, "--target_real_norm_mode", data.get("target_real_norm_mode"))
    _append_optional_value(argv, "--synth_npz_override", synth_npz_override or None)
    _append_optional_value(argv, "--target_real_npz_override", target_real_npz_override or None)
    _append_optional_value(
        argv,
        "--latent_augmix_copies",
        latent_augmix.get("copies"),
    )
    _append_optional_value(
        argv,
        "--latent_augmix_severity_profile",
        latent_augmix.get("severity_profile"),
    )
    _append_optional_sequence(argv, "--latent_augmix_ops", latent_augmix.get("ops"))
    _append_optional_value(
        argv,
        "--latent_augmix_chain_weights",
        latent_augmix.get("chain_weights"),
    )
    _append_optional_value(
        argv,
        "--latent_augmix_consistency_weight",
        consistency_weight,
    )
    _append_optional_value(
        argv,
        "--latent_augmix_consistency_loss",
        latent_augmix_consistency.get("consistency_loss"),
    )
    _append_optional_value(
        argv,
        "--latent_augmix_bce_weight",
        bce_weight,
    )
    _append_optional_value(
        argv,
        "--latent_augmix_consistency_max_batches",
        latent_augmix_consistency.get("max_batches"),
    )
    run_tag_extra = adaptation.get("run_tag_extra")
    _append_optional_value(argv, "--run_tag_extra", run_tag_extra)
    if paper["selection"]["policy"] == "last_checkpoint_only":
        argv.append("--final_checkpoint_only")
    return argv


def audit_effnet_vae_lhat_command(
    command: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return protocol-audit errors and warnings for managed EfficientNet VAE-LHAT commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = Path(argv[1]).name
    opts = argv_option_map(argv)
    case = matrix_case(command)
    paper = config["paper_protocol"]
    adaptation = config["adaptation"]
    hull = adaptation["hull"]
    attack = adaptation["attack"]
    consistency = adaptation["latent_augmix"]["consistency"]
    selection = paper["selection"]
    kshot = paper["kshot"]
    expected_k = int(kshot["k"])
    expected_seed = int(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(paper["centers"]["target_4"])
    expected_command_k = str(case.get("k", expected_k))
    expected_command_seed = str(case.get("seed", expected_seed))
    study_variant = None
    if case.get("study_scope"):
        try:
            variant_name, _, study_variant = validate_f005_case(case)
        except ValueError as exc:
            errors.append(f"{script}: {exc}")
            variant_name = str(case.get("variant") or "")
        audit_equals(errors, script, opts, "--study_scope", F005_STUDY_SCOPE)
        audit_equals(errors, script, opts, "--mechanism_variant", variant_name)

    if script != "effnet_vae_lhat_augmix.py":
        errors.append(
            f"{script}: EfficientNet VAE-LHAT adapter only accepts "
            "effnet_vae_lhat_augmix.py"
        )
        return {"errors": errors, "warnings": warnings}

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--center",
            "--comparison_arm",
            "--seed",
            "--data_root",
            "--out_root",
            "--init_ckpt",
            "--anchor_base",
            "--hull_lambda",
            "--hull_init_logit_gap",
            "--hull_neighbor_distance_space",
            "--hull_neighbor_mode",
            "--hull_neighbor_pool_size",
            "--pgd_eps",
            "--asr_low_threshold",
            "--asr_high_threshold",
            "--target_real_val_fraction",
            "--target_real_val_seed",
            "--selection_metric",
            "--source_floor_max_drop",
            "--target_adv_fraction",
        ],
    )
    center = str(opt_first(opts, "--center", ""))
    matrix_center = str(case.get("center") or "")
    if matrix_center and center != matrix_center:
        errors.append(f"{script}: matrix center {matrix_center!r} must match --center {center!r}")
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    audit_equals(errors, script, opts, "--seed", expected_command_seed)
    expected_init_ckpt = str(config["model"]["init_checkpoint"]).replace("${matrix.center}", center)
    audit_equals(errors, script, opts, "--hull_init_logit_gap", hull["init_logit_gap"])
    audit_equals(
        errors,
        script,
        opts,
        "--hull_lambda",
        study_variant.hull_lambda if study_variant is not None else hull["lambda"],
    )
    expected_hull_label_mode = (
        study_variant.label_mode if study_variant is not None else hull["label_mode"]
    )
    canonical_case_arm = str(case.get("arm") or "")
    if (
        is_matched_effnet_arm(canonical_case_arm)
        and not matched_effnet_arm(canonical_case_arm).vae_lhat
    ):
        expected_hull_label_mode = MATCHED_EFFNET_NON_VAE_HULL_LABEL_MODE
    audit_equals(errors, script, opts, "--hull_label_mode", expected_hull_label_mode)
    audit_equals(errors, script, opts, "--pgd_eps", attack["pgd_eps"])
    asr_low, asr_high = attack["target_asr_range"]
    audit_equals(errors, script, opts, "--asr_low_threshold", asr_low)
    audit_equals(errors, script, opts, "--asr_high_threshold", asr_high)
    source_floor = selection.get("source_floor") or {}
    fallback_arm = case.get("comparison_arm") or adaptation.get("comparison_arm", "historical_unmatched")
    comparison_protocol = str(paper.get("comparison_protocol") or "")
    is_f004 = comparison_protocol == F004_RHO_SWEEP_PROTOCOL
    if is_f004:
        comparison_arm, arm_components = "historical_unmatched", matched_effnet_arm("a5")
    else:
        comparison_arm, arm_components = _resolve_arm(case if "arm" in case else None, fallback_arm)
    target_adv_fraction = float(
        case.get("rho" if is_f004 else "target_adv_fraction", adaptation["loss"]["target_adv_fraction"])
    )
    if arm_components is not None and ("arm" in case or not arm_components.vae_lhat):
        target_adv_fraction = arm_components.target_adv_fraction
    expected_options = {
        "--comparison_arm": comparison_arm,
        "--init_ckpt": expected_init_ckpt,
        "--target_real_val_fraction": selection.get("validation_fraction", 0.2),
        "--target_real_val_seed": (
            expected_command_seed if study_variant is not None
            else selection.get("seed", kshot["seed"])
        ),
        "--selection_metric": selection.get("metric", "macro_auprc"),
        "--source_floor_max_drop": source_floor.get("max_drop", 0.02),
        "--target_adv_fraction": target_adv_fraction,
    }
    if is_f004:
        expected_options.update({
            "--comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
            "--comparison_variant": f004_variant_for_rho(target_adv_fraction),
            "--comparison_topology_version": MATCHED_EFFNET_CONTRACT_VERSION,
            "--comparison_topology_sha256": F004_FROZEN_TOPOLOGY_SHA256,
        })
    for option, expected in expected_options.items():
        audit_equals(errors, script, opts, option, expected)
    if arm_components is not None:
        component_options = {
            "--latent_augmix_third_chain_role": arm_components.third_chain_route,
            "--latent_augmix_bce_weight": (
                consistency.get("bce_weight") if arm_components.augmix_view_bce else 0.0
            ),
            "--latent_augmix_consistency_weight": (
                consistency.get("consistency_weight") if arm_components.jsd else 0.0
            ),
        }
        for option, expected in component_options.items():
            audit_equals(errors, script, opts, option, expected)
    expected_flags = {
        "--hull_include_anchor": (
            study_variant.include_anchor if study_variant is not None else bool(hull["include_anchor"])
        ),
        "--final_checkpoint_only": paper["selection"]["policy"] == "last_checkpoint_only",
        "--enable_latent_augmix_consistency": (
            arm_components is not None or bool(consistency["enabled"])
        ),
        "--disable_latent_augmix_consistency": (
            arm_components is None and not bool(consistency["enabled"])
        ),
    }
    if arm_components is not None:
        expected_flags.update({
            "--enable_vae_lhat": arm_components.vae_lhat,
            "--disable_vae_lhat": not arm_components.vae_lhat,
            "--enable_raw_augmix": arm_components.raw_augmix,
            "--disable_raw_augmix": not arm_components.raw_augmix,
        })
    for flag, expected in expected_flags.items():
        if (flag in opts) != expected:
            errors.append(f"{script}: {flag} presence must be {expected}")
    forbidden_ablation_flags = [
        "--enable_raw_corrupt_consistency",
        "--enable_mask_shift_consistency",
        "--raw_input_repair_flat_leads",
        "--raw_input_bandpass_low_hz",
        "--raw_input_bandpass_high_hz",
    ]
    present_forbidden = [flag for flag in forbidden_ablation_flags if flag in opts]
    if present_forbidden:
        errors.append(f"{script}: locked mainline must not enable raw/mask-shift ablation flags: {present_forbidden}")
    if opt_first(opts, "--hull_neighbor_distance_space") not in {"raw", "standardized"}:
        errors.append(f"{script}: invalid --hull_neighbor_distance_space")
    if opt_first(opts, "--hull_neighbor_mode") not in {"nearest", "local_random", "random"}:
        errors.append(f"{script}: invalid --hull_neighbor_mode")
    anchor_base = str(opt_first(opts, "--anchor_base", ""))
    if f"k{expected_command_k}_seed{expected_command_seed}" not in anchor_base:
        errors.append(f"{script}: anchor_base does not encode K{expected_command_k}/seed{expected_command_seed}")
    init_ckpt = str(opt_first(opts, "--init_ckpt", ""))
    protocol = str(case.get("protocol") or "")
    if protocol:
        if f"effnet_direct_{protocol}_v7_sjr_rgq" not in init_ckpt:
            errors.append(f"{script}: init_ckpt does not match protocol {protocol!r}")
        if f"{center}_K{expected_command_k}_direct_ft_ep" not in init_ckpt:
            errors.append(f"{script}: init_ckpt does not encode {center}/K{expected_command_k}")
    return {"errors": errors, "warnings": warnings}
