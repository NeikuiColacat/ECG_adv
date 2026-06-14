"""Typed YAML adapter for benchmark-backbone VAE-LHAT commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ecg_adv_gen.run_naming import build_benchmark_direct_run_leaf

from .common import argv_option_map, audit_equals, audit_require_options, matrix_case, opt_first


def _benchmark_direct_init_ckpt(
    *,
    output_root: str,
    model_name: str,
    center: str,
    k: int,
    seed: int,
    epochs: int,
) -> str:
    leaf = build_benchmark_direct_run_leaf(
        {
            "center": center,
            "k": k,
            "model_name": model_name,
            "epochs": epochs,
            "seed": seed,
            "val_fraction": 0.2,
        }
    )
    return (
        f"{output_root}/{model_name}_direct_k500_v7_sjr_rgq/"
        f"${{runtime.run_id}}/{center}/runs/{leaf}/best_model.pt"
    )


def build_benchmark_vae_lhat_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build legacy argv for a benchmark-backbone VAE-LHAT wrapper command."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("benchmark_vae_lhat adapter requires runner.matrix.center")

    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    adaptation = config["adaptation"]
    evaluation = config["evaluation"]
    runtime = config.get("runtime") or {}

    k = int(kshot["k"])
    seed = int(kshot.get("subset_seed", kshot["seed"]))
    model_name = str(model["name"])
    output_family = str(model.get("output_family") or f"{model_name}_vae_lhat_k500_v7_sjr_rgq")
    latent_augmix = adaptation.get("latent_augmix") or {}
    direct_init_ckpt = model.get("direct_init_ckpt") or _benchmark_direct_init_ckpt(
        output_root=str(paths["output_root"]),
        model_name=model_name,
        center=str(center),
        k=k,
        seed=seed,
        epochs=int(training["epochs"]),
    ).replace("${runtime.run_id}", str(runtime["run_id"]))
    anchor_base = (
        f"{data['kshot_subset_root']}/{center}/k{k}_seed{seed}/"
        f"{center}_real_k{k}_seed{seed}"
    )

    argv: list[Any] = [
        "--center",
        center,
        "--model_name",
        model_name,
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
        f"{paths['output_root']}/{output_family}/{runtime['run_id']}",
        "--init_ckpt",
        direct_init_ckpt,
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
    ]
    if bool(latent_augmix.get("enabled", True)):
        argv.extend(
            [
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
            ]
        )
    else:
        argv.append("--disable_latent_augmix_branch")
        argv.extend(["--run_tag_extra", latent_augmix.get("run_tag_extra", "noaugmix")])
    argv.extend(
        [
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
    )
    return argv


def audit_benchmark_vae_lhat_command(
    command: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return protocol-audit errors for managed benchmark VAE-LHAT commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = Path(argv[1]).name if len(argv) > 1 else ""
    opts = argv_option_map(argv)
    case = matrix_case(command)
    kshot = config["paper_protocol"]["kshot"]
    expected_k = int(case.get("k", kshot["k"]))
    expected_seed = int(case.get("seed", kshot.get("subset_seed", kshot["seed"])))
    expected_model = str((config.get("model") or {}).get("name") or "")
    expected_output_family = str(
        (config.get("model") or {}).get("output_family")
        or f"{expected_model}_vae_lhat_k500_v7_sjr_rgq"
    )
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])
    run_id = str((config.get("runtime") or {}).get("run_id") or "")
    latent_augmix = (config.get("adaptation") or {}).get("latent_augmix") or {}
    latent_augmix_enabled = bool(latent_augmix.get("enabled", True))

    if script != "run_effnet_latent_augmix_stage3_20260524.py":
        errors.append(
            f"{script}: benchmark VAE-LHAT adapter only accepts "
            "run_effnet_latent_augmix_stage3_20260524.py"
        )
        return {"errors": errors, "warnings": warnings}

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--center",
            "--model_name",
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
    audit_equals(errors, script, opts, "--model_name", expected_model)
    audit_equals(errors, script, opts, "--seed", str(expected_seed))
    audit_equals(errors, script, opts, "--quick_eval_source", "target_real_val")
    audit_equals(errors, script, opts, "--target_real_val_seed", str(expected_seed))
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")

    init_ckpt = str(opt_first(opts, "--init_ckpt", ""))
    expected_init_fragment = f"/{expected_model}_direct_k500_v7_sjr_rgq/{run_id}/{center}/runs/"
    if expected_model and expected_init_fragment not in init_ckpt:
        errors.append(f"{script}: init_ckpt must consume same-run benchmark Direct K500 checkpoint")
    if expected_model and expected_model not in init_ckpt:
        errors.append(f"{script}: init_ckpt does not encode model_name {expected_model!r}")
    if f"seed{expected_seed}" not in init_ckpt:
        errors.append(f"{script}: init_ckpt does not encode seed{expected_seed}")

    anchor_base = str(opt_first(opts, "--anchor_base", ""))
    if f"k{expected_k}_seed{expected_seed}" not in anchor_base:
        errors.append(f"{script}: anchor_base does not encode K{expected_k}/seed{expected_seed}")
    out_root = str(opt_first(opts, "--out_root", ""))
    if expected_output_family and f"/{expected_output_family}/{run_id}" not in out_root:
        errors.append(f"{script}: out_root must be scoped to benchmark VAE-LHAT model/run_id")
    if latent_augmix_enabled:
        if "--disable_latent_augmix_branch" in opts:
            errors.append(f"{script}: latent AugMix enabled config must not pass --disable_latent_augmix_branch")
        for name in [
            "--latent_augmix_latent_weight_cap",
            "--latent_augmix_width",
            "--latent_augmix_depth",
            "--latent_augmix_alpha",
            "--latent_augmix_severity",
        ]:
            if name not in opts:
                errors.append(f"{script}: latent AugMix enabled config missing {name}")
    else:
        if "--disable_latent_augmix_branch" not in opts:
            errors.append(f"{script}: latent AugMix disabled config must pass --disable_latent_augmix_branch")
        if str(opt_first(opts, "--run_tag_extra", "")) != "noaugmix":
            errors.append(f"{script}: latent AugMix disabled config must tag runs with --run_tag_extra noaugmix")
        for name in [
            "--latent_augmix_latent_weight_cap",
            "--latent_augmix_width",
            "--latent_augmix_depth",
            "--latent_augmix_alpha",
            "--latent_augmix_severity",
        ]:
            if name in opts:
                errors.append(f"{script}: latent AugMix disabled config must not pass {name}")
    return {"errors": errors, "warnings": warnings}
