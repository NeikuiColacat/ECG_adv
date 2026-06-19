"""Audit adapter for YAML-managed EfficientNet VAE-LHAT commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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


def build_effnet_vae_lhat_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build legacy argv for the EfficientNet VAE-LHAT wrapper from typed config fields."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("effnet_vae_lhat adapter requires runner.matrix.center")

    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    adaptation = config["adaptation"]
    evaluation = config["evaluation"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]

    k = kshot["k"]
    seed = kshot["seed"]
    out_root = f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}"
    direct_init_root = model.get("direct_init_root") or f"{paths['data_root']}/paper_direct_finetune_k500_20260516/runs"
    kshot_subset_root = data.get("kshot_subset_root") or f"{paths['data_root']}/paper_vae_only_latenthull_sweep_20260516/subsets"
    init_ckpt = (
        f"{direct_init_root}/{center}_K{k}_direct_ft_ep30_seed{seed}_val0.2/"
        "best_model.pt"
    )
    anchor_base = (
        f"{kshot_subset_root}/{center}/k{k}_seed{seed}/"
        f"{center}_real_k{k}_seed{seed}"
    )
    target_real_npz_override = str(data.get("target_real_npz_override") or "")
    target_real_npz_suffix = str(data.get("target_real_npz_suffix") or "")
    if not target_real_npz_override and target_real_npz_suffix:
        suffix = target_real_npz_suffix if target_real_npz_suffix.startswith(".") else f".{target_real_npz_suffix}"
        target_real_npz_override = f"{anchor_base}{suffix}"
    latent_augmix = adaptation["latent_augmix"]
    latent_augmix_consistency = latent_augmix.get("consistency") or {}
    selection = adaptation.get("selection") or {}
    raw_corrupt = adaptation.get("raw_corrupt_consistency") or {}
    raw_augmix = raw_corrupt.get("augmix") or {}
    raw_input_stabilizer = raw_corrupt.get("input_stabilizer") or {}
    mask_shift = adaptation.get("mask_shift_consistency") or {}

    argv: list[Any] = [
        "--center",
        center,
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
        "--anchor_base",
        anchor_base,
        "--hull_steps",
        adaptation["hull"]["steps"],
        "--hull_M",
        adaptation["hull"]["M"],
        "--hull_lambda",
        adaptation["hull"]["lambda"],
        "--hull_lr",
        adaptation["hull"]["lr"],
        "--hull_include_anchor",
        "--hull_label_mode",
        adaptation["hull"]["label_mode"],
        "--hull_mix_label_mode",
        adaptation["hull"]["mix_label_mode"],
        "--hull_label_lambda_y",
        adaptation["hull"]["label_lambda_y"],
        "--hull_label_new_class_cap",
        adaptation["hull"]["label_new_class_cap"],
        "--hull_neighbor_distance_space",
        adaptation["hull"]["neighbor_distance_space"],
        "--hull_neighbor_mode",
        adaptation["hull"]["neighbor_mode"],
        "--hull_neighbor_pool_size",
        adaptation["hull"]["neighbor_pool_size"],
        "--hull_neighbor_pool_multiplier",
        adaptation["hull"]["neighbor_pool_multiplier"],
        "--k_anchor",
        adaptation["anchors"]["k_anchor"],
        "--pgd_batch",
        adaptation["attack"]["pgd_batch"],
        "--target_real_weight",
        adaptation["loss"]["target_real_weight"],
        "--adv_weight",
        adaptation["loss"]["adv_weight"],
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
        "1.0",
        "--latent_augmix_latent_weight_cap",
        latent_augmix["latent_weight_cap"],
        "--latent_augmix_width",
        latent_augmix["width"],
        "--latent_augmix_depth",
        latent_augmix["depth"],
        "--latent_augmix_alpha",
        latent_augmix["alpha"],
        "--latent_augmix_severity",
        latent_augmix["severity"],
        "--quick_eval_source",
        selection.get("quick_eval_source", "target_real_val"),
        "--target_real_val_fraction",
        selection.get("target_real_val_fraction", 0.2),
        "--target_real_val_seed",
        seed,
        "--eval_batch_size",
        training["eval_batch_size"],
        "--eval_min_pos",
        evaluation["min_pos"],
        "--eval_pn2021_limit",
        evaluation["pn2021_limit"],
    ]
    _append_optional_value(argv, "--target_real_norm_mode", data.get("target_real_norm_mode"))
    _append_optional_value(argv, "--target_real_npz_override", target_real_npz_override or None)
    _append_optional_value(
        argv,
        "--checkpoint_policy",
        selection.get("checkpoint_policy"),
    )
    _append_optional_value(
        argv,
        "--latent_augmix_topology",
        latent_augmix.get("topology"),
    )
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
    if latent_augmix.get("no_renorm"):
        argv.append("--no_latent_augmix_renorm")
    if latent_augmix_consistency.get("enabled"):
        argv.append("--enable_latent_augmix_consistency")
        _append_optional_value(
            argv,
            "--latent_augmix_consistency_weight",
            latent_augmix_consistency.get("consistency_weight"),
        )
        _append_optional_value(
            argv,
            "--latent_augmix_consistency_loss",
            latent_augmix_consistency.get("consistency_loss"),
        )
        _append_optional_value(
            argv,
            "--latent_augmix_bce_weight",
            latent_augmix_consistency.get("bce_weight"),
        )
        _append_optional_value(
            argv,
            "--latent_augmix_consistency_max_batches",
            latent_augmix_consistency.get("max_batches"),
        )
    if raw_corrupt.get("enabled"):
        argv.append("--enable_raw_corrupt_consistency")
        _append_optional_value(argv, "--raw_corrupt_copies", raw_corrupt.get("copies"))
        _append_optional_value(argv, "--raw_corrupt_prob", raw_corrupt.get("prob"))
        _append_optional_value(argv, "--raw_corrupt_severity", raw_corrupt.get("severity"))
        _append_optional_value(
            argv,
            "--raw_corrupt_severity_profile",
            raw_corrupt.get("severity_profile", "standard"),
        )
        _append_optional_sequence(argv, "--raw_corrupt_ops", raw_corrupt.get("ops"))
        _append_optional_value(
            argv,
            "--raw_corrupt_consistency_weight",
            raw_corrupt.get("consistency_weight"),
        )
        _append_optional_value(argv, "--raw_corrupt_consistency_loss", raw_corrupt.get("consistency_loss"))
        _append_optional_value(argv, "--raw_corrupt_bce_weight", raw_corrupt.get("bce_weight"))
        _append_optional_value(argv, "--raw_corrupt_max_batches", raw_corrupt.get("max_batches"))
        _append_optional_value(argv, "--raw_corrupt_scope", raw_corrupt.get("scope"))
        _append_optional_value(argv, "--raw_corrupt_clip_abs", raw_corrupt.get("clip_abs"))
        _append_optional_value(argv, "--raw_corrupt_view_mode", raw_corrupt.get("view_mode"))
        _append_optional_value(argv, "--raw_augmix_width", raw_augmix.get("width"))
        _append_optional_value(argv, "--raw_augmix_depth", raw_augmix.get("depth"))
        _append_optional_value(argv, "--raw_augmix_alpha", raw_augmix.get("alpha"))
        _append_optional_value(argv, "--raw_augmix_mixture_mode", raw_augmix.get("mixture_mode"))
        _append_optional_value(argv, "--raw_augmix_mixture_prob", raw_augmix.get("mixture_prob"))
        _append_optional_value(argv, "--raw_augmix_mixture_beta_a", raw_augmix.get("mixture_beta_a"))
        _append_optional_value(argv, "--raw_augmix_mixture_beta_b", raw_augmix.get("mixture_beta_b"))
        if raw_input_stabilizer:
            _append_optional_value(
                argv,
                "--raw_input_bandpass_low_hz",
                raw_input_stabilizer.get("bandpass_low_hz"),
            )
            _append_optional_value(
                argv,
                "--raw_input_bandpass_high_hz",
                raw_input_stabilizer.get("bandpass_high_hz"),
            )
            if raw_input_stabilizer.get("repair_flat_leads"):
                argv.append("--raw_input_repair_flat_leads")
            _append_optional_value(argv, "--raw_input_clip_abs", raw_input_stabilizer.get("clip_abs"))
            if raw_input_stabilizer.get("renorm_after_stabilizer"):
                argv.append("--raw_input_renorm_after_stabilizer")
            _append_optional_value(
                argv,
                "--raw_input_sample_rate_hz",
                raw_input_stabilizer.get("sample_rate_hz"),
            )
        if raw_corrupt.get("no_renorm"):
            argv.append("--raw_corrupt_no_renorm")
    if mask_shift.get("enabled"):
        argv.append("--enable_mask_shift_consistency")
        _append_optional_value(argv, "--mask_shift_copies", mask_shift.get("copies"))
        _append_optional_value(argv, "--mask_shift_mask_severity", mask_shift.get("mask_severity"))
        _append_optional_value(argv, "--mask_shift_shift_severity", mask_shift.get("shift_severity"))
        _append_optional_value(
            argv,
            "--mask_shift_consistency_weight",
            mask_shift.get("consistency_weight"),
        )
        _append_optional_value(argv, "--mask_shift_consistency_loss", mask_shift.get("consistency_loss"))
        _append_optional_value(argv, "--mask_shift_bce_weight", mask_shift.get("bce_weight"))
        _append_optional_value(argv, "--mask_shift_max_batches", mask_shift.get("max_batches"))
        _append_optional_value(argv, "--mask_shift_scope", mask_shift.get("scope"))
        _append_optional_value(argv, "--mask_shift_clip_abs", mask_shift.get("clip_abs"))
        if mask_shift.get("no_renorm"):
            argv.append("--mask_shift_no_renorm")
    run_tag_extra = (
        adaptation.get("run_tag_extra")
        or mask_shift.get("run_tag_extra")
        or raw_corrupt.get("run_tag_extra")
    )
    _append_optional_value(argv, "--run_tag_extra", run_tag_extra)
    return argv


def audit_effnet_vae_lhat_command(
    command: Mapping[str, Any],
    *,
    expected_k: int | str,
    expected_seed: int | str,
    target_centers: set[str],
) -> dict[str, list[str]]:
    """Return protocol-audit errors and warnings for managed EfficientNet VAE-LHAT commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = Path(argv[1]).name
    opts = argv_option_map(argv)
    case = matrix_case(command)
    expected_command_k = str(case.get("k", expected_k))
    expected_command_seed = str(case.get("seed", expected_seed))

    if script != "run_effnet_latent_augmix_stage3_20260524.py":
        errors.append(
            f"{script}: EfficientNet VAE-LHAT adapter only accepts "
            "run_effnet_latent_augmix_stage3_20260524.py"
        )
        return {"errors": errors, "warnings": warnings}

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--center",
            "--seed",
            "--data_root",
            "--out_root",
            "--init_ckpt",
            "--anchor_base",
            "--hull_lambda",
            "--hull_neighbor_distance_space",
            "--hull_neighbor_mode",
            "--hull_neighbor_pool_size",
            "--quick_eval_source",
            "--target_real_val_fraction",
            "--target_real_val_seed",
        ],
    )
    center = str(opt_first(opts, "--center", ""))
    matrix_center = str(case.get("center") or "")
    if matrix_center and center != matrix_center:
        errors.append(f"{script}: matrix center {matrix_center!r} must match --center {center!r}")
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    audit_equals(errors, script, opts, "--seed", expected_command_seed)
    checkpoint_policy = str(opt_first(opts, "--checkpoint_policy", "best"))
    quick_eval_source = str(opt_first(opts, "--quick_eval_source", ""))
    if checkpoint_policy == "last":
        audit_equals(errors, script, opts, "--quick_eval_source", "none")
        audit_equals(errors, script, opts, "--target_real_val_fraction", "0.0")
    else:
        audit_equals(errors, script, opts, "--quick_eval_source", "target_real_val")
    audit_equals(errors, script, opts, "--target_real_val_seed", expected_command_seed)
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
    if "paper_direct_finetune_k500_20260516" not in init_ckpt:
        warnings.append(f"{script}: init_ckpt is not the historical direct-K500 run root")
    return {"errors": errors, "warnings": warnings}
