"""Typed YAML adapter for ECGFounder full fine-tuning commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import audit_equals, audit_require_options, argv_option_map, opt_first


def _append_option(argv: list[Any], flag: str, value: Any) -> None:
    if value is not None and value != "":
        argv.extend([flag, value])


def _append_flag(argv: list[Any], flag: str, enabled: Any) -> None:
    if bool(enabled):
        argv.append(flag)


def _append_list_option(argv: list[Any], flag: str, values: Any) -> None:
    if values:
        argv.append(flag)
        argv.extend(list(values))


def build_ecgfounder_fullft_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build locked ECGFounder full-FT argv from typed YAML fields."""

    paths = config["paths"]
    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    training = config["training"]
    optimizer = training["optimizer"]
    data = config["data"]
    model = config["model"]
    adaptation = config.get("adaptation") or {}
    loss = adaptation.get("loss") or {}
    vae = adaptation.get("vae") or {}
    hull = adaptation.get("hull") or {}
    attack = adaptation.get("attack") or {}
    latent_augmix = adaptation.get("latent_augmix") or {}
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]
    stage = str(adaptation.get("stage", "k500"))

    seed = kshot["seed"]
    subset_seed = kshot.get("subset_seed", seed)
    if stage == "ptbxl_source":
        argv: list[Any] = [
            "--out_dir",
            f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}",
            "--stage",
            "ptbxl_source",
            "--epochs",
            training["epochs"],
            "--seed",
            seed,
            "--device",
            "cuda",
            "--num_workers",
            training["num_workers"],
            "--batch_size",
            training["batch_size"],
            "--eval_batch_size",
            training["eval_batch_size"],
            "--lr",
            optimizer["lr"],
            "--weight_decay",
            optimizer["weight_decay"],
            "--source_weight",
            loss.get("source_weight", 1.0),
            "--target_real_weight",
            0.0,
            "--run_name",
            adaptation.get("run_name", "ptbxl_super5_fullft_locked"),
        ]
        return argv

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("ecgfounder_fullft adapter requires runner.matrix.center for K500 stage")

    ref_meta = (
        f"{data['kshot_subset_root']}/{center}/k{kshot['k']}_seed{subset_seed}/"
        f"{center}_real_k{kshot['k']}_seed{subset_seed}.ref_meta.json"
    )
    anchor_base = (
        f"{data['kshot_subset_root']}/{center}/k{kshot['k']}_seed{subset_seed}/"
        f"{center}_real_k{kshot['k']}_seed{subset_seed}"
    )
    target_raw1000_npz_override = (
        data.get("target_raw1000_npz_override")
        or data.get("target_real_npz_override")
        or ""
    )
    target_raw1000_npz_suffix = str(data.get("target_raw1000_npz_suffix") or data.get("target_real_npz_suffix") or "")
    if not target_raw1000_npz_override and target_raw1000_npz_suffix:
        suffix = target_raw1000_npz_suffix if target_raw1000_npz_suffix.startswith(".") else f".{target_raw1000_npz_suffix}"
        target_raw1000_npz_override = f"{anchor_base}{suffix}"
    argv: list[Any] = [
        "--out_dir",
        f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}",
        "--center",
        center,
        "--ref_meta_json",
        ref_meta,
        "--k",
        kshot["k"],
        "--epochs",
        training["epochs"],
        "--seed",
        seed,
        "--device",
        "cuda",
        "--num_workers",
        training["num_workers"],
        "--batch_size",
        training["batch_size"],
        "--eval_batch_size",
        training["eval_batch_size"],
        "--lr",
        optimizer["lr"],
        "--weight_decay",
        optimizer["weight_decay"],
        "--source_weight",
        loss.get("source_weight", 1.0),
        "--target_real_weight",
        loss.get("target_real_weight", 40.0),
    ]
    _append_option(argv, "--target_raw1000_npz_override", target_raw1000_npz_override)
    _append_option(argv, "--init_model_path", model.get("init_model_path", ""))
    _append_option(argv, "--run_name", adaptation.get("run_name", ""))
    if bool(vae.get("enabled", False)):
        argv.append("--enable_vae_adv_stream")
        _append_option(argv, "--adv_weight", vae.get("adv_weight", 20.0))
        _append_option(argv, "--vae_adv_stream_sample_scale", vae.get("adv_stream_sample_scale"))
        _append_option(argv, "--vae_adv_consistency_weight", vae.get("adv_consistency_weight"))
        _append_option(argv, "--adv_weight_start", vae.get("adv_weight_start"))
        _append_option(argv, "--adv_weight_warmup_epochs", vae.get("adv_weight_warmup_epochs", 0))
        _append_option(argv, "--source_bce_loss_weight", loss.get("source_bce_loss_weight", 1.0))
        _append_option(argv, "--target_real_bce_loss_weight", loss.get("target_real_bce_loss_weight", 1.0))
        _append_option(argv, "--adv_bce_loss_weight", loss.get("adv_bce_loss_weight", 1.0))
        _append_option(argv, "--adv_clean_logit_anchor_weight", loss.get("adv_clean_logit_anchor_weight", 0.0))
        _append_option(argv, "--adv_label_mode", loss.get("label_mode"))
        _append_option(argv, "--adv_teacher_mix", loss.get("teacher_mix"))
        _append_option(argv, "--adv_soft_target_floor", loss.get("soft_target_floor"))
        _append_option(argv, "--k_anchor", vae.get("k_anchor", 100))
        _append_option(argv, "--anchor_sample_mode", vae.get("anchor_sample_mode", "stratified"))
        _append_option(argv, "--anchor_sample_power", vae.get("anchor_sample_power", 1.0))
        _append_option(argv, "--anchor_sample_min_weight", vae.get("anchor_sample_min_weight", 1e-6))
        _append_option(argv, "--anchor_class_sample_weights", vae.get("anchor_class_sample_weights", ""))
        _append_option(argv, "--anchor_class_max_repeat", vae.get("anchor_class_max_repeat", 0))
        _append_list_option(argv, "--vae_classes_in_scope", vae.get("classes_in_scope"))
        _append_option(argv, "--vae_min_class_count", vae.get("min_class_count", 1))
        _append_option(argv, "--anchor_base_root", vae.get("anchor_base_root", ""))
        _append_option(argv, "--hull_m", hull.get("M", 20))
        _append_option(argv, "--hull_lambda", hull.get("lambda", 0.15))
        _append_option(argv, "--hull_steps", hull.get("steps", 3))
        _append_option(argv, "--hull_lr", hull.get("lr", 0.25))
        _append_option(argv, "--hull_init_logit_gap", hull.get("init_logit_gap", 4.0))
        _append_option(argv, "--hull_weight_mode", hull.get("weight_mode", "optimized"))
        _append_option(argv, "--hull_dirichlet_alpha", hull.get("dirichlet_alpha", 1.0))
        _append_option(argv, "--hull_attack_pos_weight_source", hull.get("attack_pos_weight_source", "none"))
        _append_option(argv, "--hull_attack_pos_weight_clip", hull.get("attack_pos_weight_clip", 50.0))
        _append_option(argv, "--hull_label_mode", hull.get("label_mode", "primary"))
        _append_flag(argv, "--hull_include_anchor", hull.get("include_anchor", False))
        _append_option(argv, "--pgd_eps", attack.get("pgd_eps", 2.0))
        _append_option(argv, "--pgd_batch", attack.get("pgd_batch", 4))
    if bool(latent_augmix.get("enabled", False)):
        argv.append("--enable_latent_augmix_branch")
        _append_option(argv, "--latent_augmix_copies", latent_augmix.get("copies", 1))
        _append_option(argv, "--latent_augmix_width", latent_augmix.get("width", 3))
        _append_option(argv, "--latent_augmix_depth", latent_augmix.get("depth", -1))
        _append_option(argv, "--latent_augmix_alpha", latent_augmix.get("alpha", 1.0))
        _append_option(argv, "--latent_augmix_severity", latent_augmix.get("severity", 5))
        _append_option(argv, "--latent_augmix_third_chain_role", latent_augmix.get("third_chain_role"))
        _append_option(argv, "--latent_augmix_chain_base_mode", latent_augmix.get("chain_base_mode"))
        _append_option(argv, "--latent_augmix_adv_base_mix", latent_augmix.get("adv_base_mix"))
        _append_option(argv, "--latent_augmix_severity_profile", latent_augmix.get("severity_profile", "standard"))
        _append_list_option(argv, "--latent_augmix_ops", latent_augmix.get("ops", []))
        consistency = latent_augmix.get("consistency") or {}
        _append_option(argv, "--latent_augmix_consistency_weight", consistency.get("consistency_weight"))
        _append_option(argv, "--latent_augmix_consistency_loss", consistency.get("consistency_loss"))
        _append_option(argv, "--latent_augmix_bce_weight", consistency.get("bce_weight"))
        _append_option(argv, "--latent_augmix_consistency_max_batches", consistency.get("max_batches"))
        _append_option(argv, "--latent_augmix_consistency_batch_size", consistency.get("batch_size"))
    return argv


def audit_ecgfounder_fullft_command(
    command: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return protocol audit errors for ECGFounder full-FT commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = argv[1].split("/")[-1] if len(argv) > 1 else ""
    opts = argv_option_map(argv)
    kshot = config["paper_protocol"]["kshot"]
    expected_k = int(kshot["k"])
    expected_seed = int(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])

    if script != "ecgfounder_fullft.py":
        return {
            "errors": [f"{script}: ecgfounder_fullft audit cannot handle this script"],
            "warnings": warnings,
        }

    forbidden_options = (
        "--target_val_count",
        "--target_val_seed",
        "--target_val_split_mode",
        "--target_val_score_weight",
        "--selection_metric",
        "--checkpoint_policy",
        "--init_head_path",
        "--trainable_scope",
        "--trainable_last_n_stages",
        "--ecgfounder_input_repair_flat_leads",
        "--ecgfounder_input_bandpass_low_hz",
        "--ecgfounder_input_bandpass_high_hz",
        "--ecgfounder_input_clip_abs",
        "--run_suffix",
        "--force",
    )
    for opt in forbidden_options:
        if opt in opts:
            errors.append(f"{script}: locked full-FT command must not pass legacy option {opt}")

    stage = str(opt_first(opts, "--stage", "k500"))
    if stage == "ptbxl_source":
        audit_require_options(
            errors,
            script,
            opts,
            [
                "--out_dir",
                "--stage",
                "--epochs",
                "--seed",
                "--run_name",
            ],
        )
        audit_equals(errors, script, opts, "--run_name", "ptbxl_super5_fullft_locked")
        if "--ref_meta_json" in opts or "--center" in opts:
            errors.append(f"{script}: PTB-XL source full-FT stage must not pass K500 center/ref_meta options")
        if "--init_model_path" in opts or "--init_head_path" in opts:
            errors.append(f"{script}: PTB-XL source full-FT stage must start from official full ECGFounder checkpoint")
        joined = " ".join(argv)
        if "best_head.pt" in joined or "residual_adapter" in joined:
            errors.append(f"{script}: locked PTB-XL full-FT command contains historical head-only/residual-adapter route")
        return {"errors": errors, "warnings": warnings}

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--out_dir",
            "--center",
            "--ref_meta_json",
            "--k",
            "--seed",
            "--init_model_path",
        ],
    )
    center = str(opt_first(opts, "--center", ""))
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    audit_equals(errors, script, opts, "--k", str(expected_k))
    audit_equals(errors, script, opts, "--seed", str(expected_seed))
    experiment_name = str((config.get("experiment") or {}).get("name") or "")
    official_composite_supervised = experiment_name.startswith(
        "ecgfounder_direct_corrupted_k500_supervised_officials5_depth23_ep10_"
    )

    target_raw1000 = str(opt_first(opts, "--target_raw1000_npz_override", ""))
    if not target_raw1000:
        errors.append(f"{script}: locked raw1000 full-FT command must pass --target_raw1000_npz_override")
    elif official_composite_supervised and not target_raw1000.endswith(".signals.npz"):
        errors.append(
            f"{script}: official composite supervised baseline must use materialized .signals.npz, "
            f"got {target_raw1000}"
        )
    elif not official_composite_supervised and not target_raw1000.endswith(".raw1000.npz"):
        errors.append(f"{script}: target_raw1000_npz_override must point at .raw1000.npz, got {target_raw1000}")

    if "--enable_raw_corrupt_consistency" in opts or "--enable_raw_corrupt_aux_consistency" in opts:
        errors.append(f"{script}: locked full-FT baseline must not enable raw corruption branches")
    joined = " ".join(argv)
    if "best_head.pt" in joined or "residual_adapter" in joined:
        errors.append(f"{script}: locked full-FT command contains historical head-only/residual-adapter route")
    latent_augmix_third_chain_role = str(
        opt_first(opts, "--latent_augmix_third_chain_role", "vae_lhat_adversarial_waveform")
    )
    latent_augmix_chain_base_mode = str(
        opt_first(opts, "--latent_augmix_chain_base_mode", "clean_clean_third")
    )
    if "--enable_latent_augmix_branch" in opts:
        third_chain_role = latent_augmix_third_chain_role
        numeric_values: dict[str, int | float | None] = {}
        for option, default, parse in (
            ("--latent_augmix_copies", 1, int),
            ("--latent_augmix_consistency_weight", 0.0, float),
            ("--latent_augmix_bce_weight", 0.0, float),
        ):
            raw_value = opt_first(opts, option, default)
            try:
                if isinstance(raw_value, bool):
                    raise ValueError
                numeric_values[option] = parse(raw_value)
            except (TypeError, ValueError):
                errors.append(f"{script}: {option} has invalid numeric value {raw_value!r}")
                numeric_values[option] = None
        consistency_loss = str(opt_first(opts, "--latent_augmix_consistency_loss", "jsd"))
        copies = numeric_values["--latent_augmix_copies"]
        consistency_weight = numeric_values["--latent_augmix_consistency_weight"]
        bce_weight = numeric_values["--latent_augmix_bce_weight"]
        if (
            copies is not None
            and consistency_weight is not None
            and bce_weight is not None
            and (consistency_weight > 0.0 or bce_weight > 0.0)
            and consistency_loss == "jsd"
            and copies < 2
        ):
            errors.append(
                f"{script}: --latent_augmix_consistency_loss jsd requires "
                "--latent_augmix_copies >= 2 when either consistency weight is positive"
            )
        if latent_augmix_chain_base_mode in {"one_adv", "all_adv"} and third_chain_role == "clean_anchor_control":
            errors.append(f"{script}: {latent_augmix_chain_base_mode} latent AugMix cannot use clean_anchor_control")
        needs_vae_augmix = (
            latent_augmix_chain_base_mode in {"one_adv", "all_adv"}
            or (
                latent_augmix_chain_base_mode == "clean_clean_third"
                and third_chain_role == "vae_lhat_adversarial_waveform"
            )
        )
        if needs_vae_augmix and "--enable_vae_adv_stream" not in opts:
            errors.append(f"{script}: locked latent AugMix requires --enable_vae_adv_stream")
        if latent_augmix_chain_base_mode == "all_clean_plus_vae_adv" and "--enable_vae_adv_stream" not in opts:
            errors.append(f"{script}: all_clean_plus_vae_adv requires --enable_vae_adv_stream")
        if (
            latent_augmix_chain_base_mode == "all_clean"
            or (
                third_chain_role == "clean_anchor_control"
                and latent_augmix_chain_base_mode != "all_clean_plus_vae_adv"
            )
        ) and "--enable_vae_adv_stream" in opts:
            errors.append(f"{script}: clean-anchor no-VAE AugMix ablation must not enable --enable_vae_adv_stream")
        if (
            latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
            or third_chain_role == "clean_anchor_control"
        ):
            try:
                adv_weight = float(opt_first(opts, "--adv_weight", "0"))
            except (TypeError, ValueError):
                adv_weight = 0.0
            if adv_weight <= 0.0:
                errors.append(f"{script}: clean-anchor latent AugMix requires --adv_weight > 0")
        audit_equals(errors, script, opts, "--latent_augmix_width", "3")
        audit_equals(errors, script, opts, "--latent_augmix_severity", "5")
        latent_augmix = ((config.get("adaptation") or {}).get("latent_augmix")) or {}
        expected_profile = str(latent_augmix.get("severity_profile", "standard"))
        audit_equals(errors, script, opts, "--latent_augmix_severity_profile", expected_profile)
        if expected_profile == "custom":
            errors.append(f"{script}: locked latent AugMix training does not support custom severity profiles")
        removed_latent_augmix_opts = [
            "--latent_augmix_topology",
            "--latent_augmix_severity_params_file",
            "--latent_augmix_severity_params_name",
        ]
        present_removed = [opt for opt in removed_latent_augmix_opts if opt in opts]
        if present_removed:
            errors.append(f"{script}: locked latent AugMix command exposes removed knobs {present_removed}")
        expected_ops = list(
            latent_augmix.get(
                "ops",
                [
                    "powerline_noise",
                    "emg_noise",
                    "baseline_wander",
                    "baseline_shift",
                    "random_leads_masking",
                ],
            )
        )
        actual_ops = list(opts.get("--latent_augmix_ops", []))
        if actual_ops != expected_ops:
            errors.append(f"{script}: locked latent AugMix ops must be {expected_ops}, got {actual_ops}")

    ref_meta = str(opt_first(opts, "--ref_meta_json", ""))
    if official_composite_supervised:
        expected_ref = (
            f"/pn2021c_official_s5_composite_k500_direct_supervised_20260624/subsets/{center}/"
            f"k{expected_k}_seed{expected_seed}/{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
        )
    else:
        expected_ref = (
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{center}/"
            f"k{expected_k}_seed{expected_seed}/{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
        )
    if not ref_meta.endswith(expected_ref):
        errors.append(f"{script}: ref_meta_json does not use locked K500 subset path")

    init_model = str(opt_first(opts, "--init_model_path", ""))
    run_id = str((config.get("runtime") or {}).get("run_id") or "")
    upstream_run_id = str(((config.get("model") or {}).get("upstream_run_id")) or run_id)
    uses_k500_fullft_init = (
            "--enable_vae_adv_stream" in opts
            or (
                "--enable_latent_augmix_branch" in opts
                and (
                    latent_augmix_third_chain_role == "clean_anchor_control"
                    or latent_augmix_chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
                )
            )
        )
    if uses_k500_fullft_init:
        expected_init = (
            f"/ecgfounder_k500_fullft_locked/{upstream_run_id}/runs/"
            f"{center}_k500_fullft_locked/last_model.pt"
        )
    else:
        expected_init = (
            f"/ecgfounder_ptbxl_super5_fullft_locked/{upstream_run_id}/runs/"
            "ptbxl_super5_fullft_locked/last_model.pt"
        )
    if upstream_run_id and not init_model.endswith(expected_init):
        errors.append(
            f"{script}: init_model_path must consume the locked upstream last_model.pt "
            "from runtime.run_id or model.upstream_run_id"
        )
    return {"errors": errors, "warnings": warnings}
