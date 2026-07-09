"""Typed YAML adapter for direct K-shot finetune commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, matrix_case, opt_first, opt_list


def build_direct_finetune_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for package-owned direct K-shot finetune."""

    matrix = context.get("matrix") or {}
    case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else {}
    center = matrix.get("center") or case.get("center")

    paths = config["paths"]
    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    preprocess = config["preprocess"]
    training = config["training"]
    data = config["data"]
    evaluation = config.get("evaluation") or {}
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]
    centers = [center] if center else list(paper["centers"]["target_4"])
    k_value = case.get("k", kshot["k"])
    protocol = case.get("protocol")
    out_root = f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}"
    if center and protocol:
        out_root = f"{paths['output_root']}/effnet_direct_{protocol}_v7_sjr_rgq/{runtime['run_id']}/{center}"
    elif center:
        out_root = f"{out_root}/{center}"

    argv = [
        "--centers",
        *centers,
        "--k",
        k_value,
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
        "--out_root",
        out_root,
    ]
    if data.get("kshot_subset_root"):
        argv.extend(["--subset_root", data["kshot_subset_root"]])
    if experiment.get("status") == "smoke-only":
        argv.extend(["--eval_min_pos", evaluation["min_pos"], "--eval_pn2021_limit", evaluation["pn2021_limit"]])
    return argv


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

    if script != "effnet_direct_finetune.py":
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
        ["--centers", "--k", "--subset_seed", "--seed", "--val_fraction", "--out_root"],
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
    subset_root = str(opt_first(opts, "--subset_root", ""))
    if subset_root:
        if f"k{expected_command_k}_seed{expected_command_seed}" in subset_root:
            errors.append(f"{script}: subset_root should be the root directory, not one center-specific K-shot base")
        if "subsets" not in subset_root:
            errors.append(f"{script}: subset_root does not point at a K-shot subsets directory")
    return {"errors": errors, "warnings": warnings}
