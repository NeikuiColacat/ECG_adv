"""Typed YAML adapter for ECGFounder VAE-LHAT legacy commands."""

from __future__ import annotations

from typing import Any, Mapping


def _append_flag(argv: list[Any], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def build_ecgfounder_vae_lhat_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build legacy argv for the ECGFounder VAE-LHAT runner from typed config fields."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("ecgfounder_vae_lhat adapter requires runner.matrix.center")

    paper = config["paper_protocol"]
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    adaptation = config["adaptation"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]

    seed = paper["kshot"]["seed"]
    hull = adaptation["hull"]
    anchors = adaptation["anchors"]
    attack = adaptation["attack"]
    ecgfounder = adaptation["ecgfounder"]
    loss = adaptation["loss"]
    selection = adaptation["selection"]
    optimizer = training["optimizer"]
    latent_augmix = adaptation.get("latent_augmix") or {}
    raw_corrupt = adaptation.get("raw_corrupt_consistency") or {}

    argv: list[Any] = [
        "--centers",
        center,
        "--out_dir",
        f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}/{center}",
        "--linear_probe_dir",
        model["linear_probe_dir"],
        "--checkpoint",
        model["checkpoint"],
        "--preprocess_policy",
        model["preprocess_policy"],
        "--k",
        paper["kshot"]["k"],
        "--k_anchor",
        anchors["k_anchor"],
        "--anchor_sample_mode",
        anchors["sample_mode"],
        "--anchor_sample_power",
        anchors["sample_power"],
        "--anchor_sample_min_weight",
        anchors["sample_min_weight"],
        "--hull_m",
        hull["M"],
        "--hull_lambda",
        hull["lambda"],
        "--hull_steps",
        hull["steps"],
        "--hull_lr",
        hull["lr"],
        "--hull_weight_mode",
        hull["weight_mode"],
        "--hull_dirichlet_alpha",
        hull["dirichlet_alpha"],
        "--hull_label_mode",
        hull["label_mode"],
        "--hull_mix_label_mode",
        hull["mix_label_mode"],
        "--hull_label_lambda_y",
        hull["label_lambda_y"],
        "--hull_label_positive",
        hull["label_positive"],
        "--hull_label_negative_floor",
        hull["label_negative_floor"],
        "--hull_label_new_class_cap",
        hull["label_new_class_cap"],
    ]
    _append_flag(argv, "--hull_include_anchor", bool(hull.get("include_anchor", False)))
    argv.extend(
        [
            "--hull_neighbor_distance_space",
            hull["neighbor_distance_space"],
            "--hull_neighbor_mode",
            hull["neighbor_mode"],
            "--hull_neighbor_pool_size",
            hull["neighbor_pool_size"],
            "--hull_neighbor_pool_multiplier",
            hull["neighbor_pool_multiplier"],
            "--pgd_eps",
            attack["pgd_eps"],
            "--pgd_batch",
            attack["pgd_batch"],
            "--attack_success_margin",
            attack["attack_success_margin"],
            "--epochs",
            training["epochs"],
            "--eval_every",
            training["eval_every"],
            "--lr",
            optimizer["lr"],
            "--weight_decay",
            optimizer["weight_decay"],
            "--batch_size",
            training["batch_size"],
            "--eval_batch_size",
            training["eval_batch_size"],
            "--source_train_limit",
            training.get("source_train_limit", 0),
            "--head_type",
            ecgfounder["head_type"],
            "--adapter_hidden",
            ecgfounder["adapter_hidden"],
            "--adapter_dropout",
            ecgfounder["adapter_dropout"],
            "--adapter_scale",
            ecgfounder["adapter_scale"],
        ]
    )
    _append_flag(argv, "--freeze_base_head", bool(ecgfounder.get("freeze_base_head", False)))
    argv.extend(
        [
            "--init_base_head_from_k500_root",
            model["k500_head_root"],
            "--source_weight",
            loss["source_weight"],
            "--target_real_weight",
            loss["target_real_weight"],
            "--adv_weight",
            loss["adv_weight"],
            "--source_logit_anchor_weight",
            loss["source_logit_anchor_weight"],
            "--target_logit_anchor_weight",
            loss["target_logit_anchor_weight"],
            "--selection_metric",
            selection["metric"],
            "--selection_source",
            selection["source"],
            "--target_real_val_fraction",
            selection["target_real_val_fraction"],
            "--target_real_val_seed",
            seed,
            "--source_selection_weight",
            selection["source_selection_weight"],
            "--source_auprc_floor",
            selection["source_auprc_floor"],
            "--source_floor_penalty",
            selection["source_floor_penalty"],
            "--anchor_base_root",
            anchors["base_root"],
            "--ref_root",
            data["kshot_subset_root"],
            "--seed",
            seed,
            "--device",
            "cuda",
            "--report_drop_all_zero_pn2021",
        ]
    )
    if bool(latent_augmix.get("enabled", False)):
        argv.extend(
            [
                "--enable_latent_augmix_branch",
                "--latent_augmix_latent_weight_cap",
                latent_augmix.get("latent_weight_cap", 0.25),
                "--latent_augmix_width",
                latent_augmix.get("width", 3),
                "--latent_augmix_depth",
                latent_augmix.get("depth", -1),
                "--latent_augmix_alpha",
                latent_augmix.get("alpha", 1.0),
                "--latent_augmix_severity",
                latent_augmix.get("severity", 2),
                "--latent_augmix_severity_profile",
                latent_augmix.get("severity_profile", "standard"),
            ]
        )
        if latent_augmix.get("ops"):
            argv.append("--latent_augmix_ops")
            argv.extend(latent_augmix["ops"])
        if bool(latent_augmix.get("no_renorm", False)):
            argv.append("--no_latent_augmix_renorm")
        if latent_augmix.get("clip_abs") is not None:
            argv.extend(["--latent_augmix_clip_abs", latent_augmix.get("clip_abs")])
    if bool(raw_corrupt.get("enabled", False)):
        raw_augmix = raw_corrupt.get("augmix") or {}
        argv.extend(
            [
                "--enable_raw_corrupt_consistency",
                "--raw_corrupt_batch_size",
                raw_corrupt.get("batch_size", 128),
                "--raw_corrupt_copies",
                raw_corrupt.get("copies", 1),
                "--raw_corrupt_prob",
                raw_corrupt.get("prob", 0.5),
                "--raw_corrupt_severity",
                raw_corrupt.get("severity", 4),
                "--raw_corrupt_severity_profile",
                raw_corrupt.get("severity_profile", "standard"),
                "--raw_corrupt_consistency_weight",
                raw_corrupt.get("consistency_weight", 0.5),
                "--raw_corrupt_consistency_loss",
                raw_corrupt.get("consistency_loss", "soft_bce"),
                "--raw_corrupt_bce_weight",
                raw_corrupt.get("bce_weight", 0.1),
                "--raw_corrupt_max_batches",
                raw_corrupt.get("max_batches", 0),
                "--raw_corrupt_scope",
                raw_corrupt.get("scope", "target"),
                "--raw_corrupt_clip_abs",
                raw_corrupt.get("clip_abs", 6.0),
                "--raw_corrupt_grad_clip",
                raw_corrupt.get("grad_clip", 0.0),
                "--source_signal_cache_dir",
                raw_corrupt.get("source_signal_cache_dir", ""),
                "--raw_corrupt_view_mode",
                raw_corrupt.get("view_mode", "single_op"),
                "--raw_augmix_width",
                raw_augmix.get("width", 3),
                "--raw_augmix_depth",
                raw_augmix.get("depth", -1),
                "--raw_augmix_alpha",
                raw_augmix.get("alpha", 1.0),
                "--raw_augmix_mixture_mode",
                raw_augmix.get("mixture_mode", "beta"),
                "--raw_augmix_mixture_prob",
                raw_augmix.get("mixture_prob", 0.5),
                "--raw_augmix_mixture_beta_a",
                raw_augmix.get("mixture_beta_a", 0.0),
                "--raw_augmix_mixture_beta_b",
                raw_augmix.get("mixture_beta_b", 0.0),
            ]
        )
        if raw_corrupt.get("ops"):
            argv.append("--raw_corrupt_ops")
            argv.extend(raw_corrupt["ops"])
        if bool(raw_corrupt.get("no_renorm", False)):
            argv.append("--raw_corrupt_no_renorm")
    return argv
