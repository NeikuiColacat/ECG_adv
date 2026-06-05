"""Typed YAML adapter for EfficientNet direct K-shot finetune commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, matrix_case, opt_first, opt_list
from .direct import audit_direct_finetune_command as audit_legacy_direct_finetune_command


def build_direct_finetune_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``scripts/paper/run_direct_finetune_k500_20260516.py``."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("direct_finetune adapter requires runner.matrix.center")

    paths = config["paths"]
    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    preprocess = config["preprocess"]
    training = config["training"]
    data = config["data"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]

    return [
        "--centers",
        center,
        "--k",
        kshot["k"],
        "--subset_seed",
        kshot.get("subset_seed", kshot["seed"]),
        "--seed",
        kshot["seed"],
        "--epochs",
        training["epochs"],
        "--lr",
        training["optimizer"]["lr"],
        "--weight_decay",
        training["optimizer"]["weight_decay"],
        "--grad_clip",
        training["grad_clip"],
        "--batch_size",
        training["batch_size"],
        "--eval_batch_size",
        training["eval_batch_size"],
        "--crop_len",
        preprocess["crop_len"],
        "--val_fraction",
        "0.2",
        "--num_workers",
        training["num_workers"],
        "--device",
        "cuda",
        "--pos_weight_clip_max",
        training["pos_weight_clip_max"],
        "--subset_root",
        data["kshot_subset_root"],
        "--out_root",
        f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}/{center}",
    ]


def audit_direct_finetune_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return protocol audit errors for managed Direct K-shot finetune commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = argv[1].split("/")[-1] if len(argv) > 1 else ""
    kshot = config["paper_protocol"]["kshot"]
    expected_k = int(kshot["k"])
    expected_seed = int(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])

    if script == "run_direct_finetune_k500_20260516.py":
        errors.extend(
            audit_legacy_direct_finetune_command(
                command,
                expected_k=expected_k,
                expected_seed=expected_seed,
                target_centers=target_centers,
            )
        )
        return {"errors": errors, "warnings": warnings}

    if script != "run_benchmark_direct_finetune_v7_20260530.py":
        errors.append(f"{script}: direct_finetune audit cannot handle this script")
        return {"errors": errors, "warnings": warnings}

    opts = argv_option_map(argv)
    matrix = command.get("matrix") or {}
    case = matrix_case(command)
    expected_command_k = str(case.get("k", expected_k))
    expected_command_seed = str(case.get("seed", expected_seed))

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--centers",
            "--k",
            "--subset_seed",
            "--seed",
            "--model_name",
            "--init_ckpt",
            "--epochs",
            "--val_fraction",
            "--out_root",
            "--subset_root",
            "--data_root",
            "--python",
        ],
    )
    audit_equals(errors, script, opts, "--k", expected_command_k)
    audit_equals(errors, script, opts, "--subset_seed", expected_command_seed)
    audit_equals(errors, script, opts, "--seed", expected_command_seed)
    centers = set(opt_list(opts, "--centers"))
    matrix_center = str(matrix.get("center") or case.get("center") or "")
    if matrix_center:
        if centers != {matrix_center}:
            errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {sorted(centers)!r}")
        if matrix_center not in target_centers:
            errors.append(f"{script}: unexpected matrix center {matrix_center!r}")
    elif centers != target_centers:
        errors.append(f"{script}: centers={sorted(centers)!r}, expected {sorted(target_centers)!r}")

    model_name = str(opt_first(opts, "--model_name", ""))
    expected_model = str((config.get("model") or {}).get("name") or "")
    if expected_model and model_name != expected_model:
        errors.append(f"{script}: --model_name={model_name!r}, expected {expected_model!r}")
    init_ckpt = str(opt_first(opts, "--init_ckpt", ""))
    if "benchmark_source_v7_sjr_rgq" not in init_ckpt:
        errors.append(f"{script}: init_ckpt must come from benchmark_source_v7_sjr_rgq")
    if model_name and model_name not in init_ckpt:
        errors.append(f"{script}: init_ckpt does not encode model_name {model_name!r}")
    if f"seed{expected_command_seed}" not in init_ckpt:
        errors.append(f"{script}: init_ckpt does not encode seed{expected_command_seed}")
    subset_root = str(opt_first(opts, "--subset_root", ""))
    if subset_root:
        if f"k{expected_command_k}_seed{expected_command_seed}" in subset_root:
            errors.append(f"{script}: subset_root should be the root directory, not one center-specific K-shot base")
        if "subsets" not in subset_root:
            errors.append(f"{script}: subset_root does not point at a K-shot subsets directory")
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
    if "--force" in opts:
        errors.append(f"{script}: benchmark direct managed config must not pass --force")
    return {"errors": errors, "warnings": warnings}
