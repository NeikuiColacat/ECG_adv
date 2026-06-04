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

    return [
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
        adaptation["latent_augmix"]["latent_weight_cap"],
        "--latent_augmix_width",
        adaptation["latent_augmix"]["width"],
        "--latent_augmix_depth",
        adaptation["latent_augmix"]["depth"],
        "--latent_augmix_alpha",
        adaptation["latent_augmix"]["alpha"],
        "--latent_augmix_severity",
        adaptation["latent_augmix"]["severity"],
        "--quick_eval_source",
        "target_real_val",
        "--target_real_val_fraction",
        "0.2",
        "--target_real_val_seed",
        seed,
        "--eval_batch_size",
        training["eval_batch_size"],
        "--eval_min_pos",
        evaluation["min_pos"],
        "--eval_pn2021_limit",
        evaluation["pn2021_limit"],
    ]


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
