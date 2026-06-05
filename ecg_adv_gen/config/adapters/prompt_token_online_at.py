"""Typed YAML adapter for prompt-token gated-pool online AT commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, opt_first, opt_list


def build_prompt_token_online_at_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``scripts/pgd_cross_center/synth_online_at_super5.py``."""

    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    adaptation = config["adaptation"]
    prompt = adaptation["prompt_token"]
    hull = adaptation["hull"]
    loss = adaptation["loss"]
    latent_augmix = adaptation["latent_augmix"]
    center = prompt["center"]
    gated_dir = prompt["gated_dir"]

    return [
        "--center_name",
        center,
        "--ref_meta_json",
        f"{gated_dir}/gated_samples.ref_meta.json",
        "--synth_npz",
        f"{gated_dir}/gated_samples.latent.npz",
        "--target_real_npz",
        (
            f"{data['kshot_subset_root']}/{center}/k{kshot['k']}_seed{kshot['seed']}/"
            f"{center}_real_k{kshot['k']}_seed{kshot['seed']}.signals.npz"
        ),
        "--class_trust",
        f"{gated_dir}/gated_samples.class_trust.json",
        "--init_ckpt",
        f"{model['direct_init_root']}/{center}_K{kshot['k']}_direct_ft_ep30_seed{kshot['seed']}_val0.2/best_model.pt",
        "--model_name",
        model["name"],
        "--output_dir",
        f"{prompt['online_at_root']}/{center}",
        "--data_dir",
        f"{paths['data_root']}/physionet2021/training",
        "--quick_eval_source",
        "target_real_val",
        "--quick_eval_centers",
        center,
        "--quick_eval_n_per_center",
        0,
        "--target_real_val_fraction",
        "0.2",
        "--target_real_val_seed",
        kshot["seed"],
        "--ptbxl_raw",
        f"{paths['data_root']}/ptbxl/raw100.npy",
        "--ptbxl_csv",
        f"{paths['data_root']}/ptbxl/ptbxl_database.csv",
        "--ptbxl_prep",
        f"{paths['data_root']}/crosscenter_v2/ptbxl_preprocessed.npy",
        "--attack_mode",
        "latent_hull",
        "--hull_M",
        hull["M"],
        "--hull_lambda",
        hull["lambda"],
        "--hull_steps",
        hull["steps"],
        "--hull_lr",
        hull["lr"],
        "--hull_include_anchor",
        "--hull_label_mode",
        hull["label_mode"],
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
        "--K_anchor",
        adaptation["anchors"]["k_anchor"],
        "--pgd_batch",
        adaptation["attack"]["pgd_batch"],
        "--target_real_weight",
        loss["target_real_weight"],
        "--ptbxl_weight",
        loss["clean_weight"],
        "--adv_weight",
        loss["adv_weight"],
        "--adv_weight_warmup_epochs",
        loss["adv_weight_warmup_epochs"],
        "--adv_label_mode",
        loss["label_mode"],
        "--adv_teacher_mix",
        loss["teacher_mix"],
        "--classes_in_scope",
        *list(prompt["classes"]),
        "--enable_latent_augmix_branch",
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
        "--n_epochs",
        training["epochs"],
        "--patience",
        8,
        "--es_metric",
        "target_macro_auprc",
        "--lr",
        training["optimizer"]["lr"],
        "--weight_decay",
        training["optimizer"]["weight_decay"],
        "--batch_size",
        training["batch_size"],
        "--num_workers",
        training["num_workers"],
        "--grad_clip",
        training["grad_clip"],
        "--eval_every",
        1,
        "--rescore_interval",
        1,
        "--seed",
        kshot["seed"],
        "--device",
        "cuda",
    ]


def audit_prompt_token_online_at_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return protocol audit errors for prompt-token online AT commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    opts = argv_option_map(argv)
    script = "synth_online_at_super5.py"
    kshot = config["paper_protocol"]["kshot"]
    expected_k = str(kshot["k"])
    expected_seed = str(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])
    run_id = str((config.get("runtime") or {}).get("run_id") or "")

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--center_name",
            "--ref_meta_json",
            "--synth_npz",
            "--target_real_npz",
            "--class_trust",
            "--init_ckpt",
            "--model_name",
            "--output_dir",
            "--data_dir",
            "--quick_eval_source",
            "--quick_eval_centers",
            "--target_real_val_fraction",
            "--target_real_val_seed",
            "--ptbxl_raw",
            "--ptbxl_csv",
            "--ptbxl_prep",
            "--attack_mode",
            "--hull_M",
            "--hull_lambda",
            "--hull_steps",
            "--hull_lr",
            "--hull_label_mode",
            "--hull_mix_label_mode",
            "--hull_neighbor_distance_space",
            "--hull_neighbor_mode",
            "--hull_neighbor_pool_size",
            "--K_anchor",
            "--pgd_batch",
            "--target_real_weight",
            "--ptbxl_weight",
            "--adv_weight",
            "--adv_weight_warmup_epochs",
            "--adv_label_mode",
            "--adv_teacher_mix",
            "--classes_in_scope",
            "--n_epochs",
            "--num_workers",
            "--seed",
            "--device",
        ],
    )
    center = str(opt_first(opts, "--center_name", ""))
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    audit_equals(errors, script, opts, "--seed", expected_seed)
    audit_equals(errors, script, opts, "--target_real_val_seed", expected_seed)
    audit_equals(errors, script, opts, "--quick_eval_source", "target_real_val")
    audit_equals(errors, script, opts, "--attack_mode", "latent_hull")
    audit_equals(errors, script, opts, "--hull_label_mode", "compatible")
    audit_equals(errors, script, opts, "--hull_mix_label_mode", "anchor_soft")
    audit_equals(errors, script, opts, "--adv_label_mode", "latent_mixed_teacher")
    audit_equals(errors, script, opts, "--adv_teacher_mix", "0.4")
    if opt_list(opts, "--quick_eval_centers") != [center]:
        errors.append(f"{script}: --quick_eval_centers must contain only the target center")
    if opt_list(opts, "--classes_in_scope") != ["NORM", "MI", "STTC"]:
        errors.append(f"{script}: classes_in_scope must be ['NORM', 'MI', 'STTC']")
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
    if "--build_class_trust" in opts:
        errors.append(f"{script}: managed online-AT config must consume a gated class_trust artifact")
    if "--disable_adv_stream" in opts:
        errors.append(f"{script}: prompt-token online-AT config must keep the adversarial stream enabled")

    model_name = str(opt_first(opts, "--model_name", ""))
    expected_model = str((config.get("model") or {}).get("name") or "")
    if expected_model and model_name != expected_model:
        errors.append(f"{script}: --model_name={model_name!r}, expected {expected_model!r}")

    gated_dir = Path(str(opt_first(opts, "--synth_npz", ""))).parent
    expected_gated_fragment = f"/ecgtwin_prompt_token_minimal/{run_id}/generated/target_token/{center}/gated"
    if run_id and expected_gated_fragment not in str(gated_dir):
        errors.append(f"{script}: synth_npz must consume same-run prompt-token gated latents")
    expected_files = {
        "--synth_npz": "gated_samples.latent.npz",
        "--class_trust": "gated_samples.class_trust.json",
        "--ref_meta_json": "gated_samples.ref_meta.json",
    }
    for option, expected_name in expected_files.items():
        value = Path(str(opt_first(opts, option, "")))
        if value.name != expected_name:
            errors.append(f"{script}: {option} must point at {expected_name}")
        if str(value.parent) != str(gated_dir):
            errors.append(f"{script}: {option} must come from the same gated directory")

    target_real_npz = str(opt_first(opts, "--target_real_npz", ""))
    expected_real_suffix = (
        f"/{center}/k{expected_k}_seed{expected_seed}/"
        f"{center}_real_k{expected_k}_seed{expected_seed}.signals.npz"
    )
    if not target_real_npz.endswith(expected_real_suffix):
        errors.append(f"{script}: target_real_npz must point at the v7 K-shot signals artifact")

    init_ckpt = str(opt_first(opts, "--init_ckpt", ""))
    expected_init_suffix = (
        f"/effnet_direct_k500_v7_sjr_rgq/{run_id}/runs/"
        f"{center}_K{expected_k}_direct_ft_ep30_seed{expected_seed}_val0.2/best_model.pt"
    )
    if run_id and not init_ckpt.endswith(expected_init_suffix):
        errors.append(f"{script}: init_ckpt must consume same-run EfficientNet Direct K500 checkpoint")
    output_dir = str(opt_first(opts, "--output_dir", ""))
    if run_id and not output_dir.endswith(f"/ecgtwin_prompt_token_online_at_minimal/{run_id}/{center}"):
        errors.append(f"{script}: output_dir must be run-id-scoped under ecgtwin_prompt_token_online_at_minimal")
    return {"errors": errors, "warnings": warnings}
