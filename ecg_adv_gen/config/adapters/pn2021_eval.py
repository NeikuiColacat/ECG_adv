"""Typed YAML adapter for clean PN2021 ref-excluded eval commands."""

from __future__ import annotations

from typing import Any, Mapping

from ecg_adv_gen.run_naming import (
    build_f005_control_producer_dir,
    build_matched_effnet_producer_dir,
)
from ecg_adv_gen.f005_control import validate_f005_case

from .common import argv_option_map, audit_equals, audit_require_options, opt_first, opt_list


def build_pn2021_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``ecg_adv_gen/runner/pn2021_clean_eval.py``."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else None
    arm = str((case or {}).get("arm") or matrix.get("arm") or "")
    if not center:
        raise ValueError("pn2021_eval adapter requires runner.matrix.center")
    if not arm:
        raise ValueError("pn2021_eval adapter requires runner.matrix.arm")

    paths = config["paths"]
    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    preprocess = config["preprocess"]
    model = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    data = config["data"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]

    if case is not None and case.get("study_scope"):
        variant, seed, _ = validate_f005_case(case)
        model_dir = build_f005_control_producer_dir(
            config, center=str(center), seed=seed, variant=variant
        )
        ref_exclusion_meta = [
            f"{data['kshot_subset_root']}/{target}/k{kshot['k']}_seed{seed}/"
            f"{target}_real_k{kshot['k']}_seed{seed}.ref_meta.json"
            for target in paper["centers"]["target_4"]
        ]
        output_role = f"seed{seed}/{variant}"
    else:
        model_dir = build_matched_effnet_producer_dir(config, center=str(center), arm=arm)
        ref_exclusion_meta = list(evaluation["ref_exclusion_meta"])
        output_role = arm
    return [
        "--scheme",
        "super5",
        "--model_dir",
        model_dir,
        "--checkpoint_name",
        model.get("checkpoint_name", "best_model.pt"),
        "--model_name",
        model["backbone"],
        "--device",
        "cuda",
        "--crop_len",
        preprocess["crop_len"],
        "--batch_size",
        training["eval_batch_size"],
        "--num_workers",
        training["num_workers"],
        "--min_pos",
        evaluation["min_pos"],
        "--ptbxl_csv",
        f"{paths['data_root']}/ptbxl/ptbxl_database.csv",
        "--ptbxl_cache",
        f"{paths['data_root']}/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy",
        "--preprocess_mode",
        preprocess["mode"],
        "--norm_mode",
        preprocess["norm_mode"],
        "--pn2021_root",
        f"{paths['data_root']}/physionet2021",
        "--pn2021_cache_dir",
        data["cache"]["pn2021_cache_dir"],
        "--pn2021_mmap_cache_dir",
        data["cache"]["pn2021_mmap_cache_dir"],
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
        "--eval_protocol",
        "paper_refexcluded",
        "--min_target_ref_excluded",
        kshot["k"],
        "--exclude_ref_ids",
        *ref_exclusion_meta,
        "--output_path",
        f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}/{center}/{output_role}/eval_result_v7_super5_sjr_rgq_refexcluded.json",
    ]


def audit_pn2021_eval_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return protocol audit errors for clean PN2021 ref-excluded eval commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    opts = argv_option_map(argv)
    script = "pn2021_clean_eval.py"
    kshot = config["paper_protocol"]["kshot"]
    expected_k = str(kshot["k"])
    expected_seed = str(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--scheme",
            "--model_dir",
            "--model_name",
            "--device",
            "--crop_len",
            "--batch_size",
            "--num_workers",
            "--ptbxl_csv",
            "--ptbxl_cache",
            "--preprocess_mode",
            "--norm_mode",
            "--pn2021_root",
            "--pn2021_cache_dir",
            "--pn2021_mmap_cache_dir",
            "--exclude_ref_ids",
            "--eval_protocol",
            "--output_path",
        ],
    )
    audit_equals(errors, script, opts, "--scheme", "super5")
    audit_equals(errors, script, opts, "--crop_len", config["preprocess"]["crop_len"])
    audit_equals(errors, script, opts, "--preprocess_mode", config["preprocess"]["mode"])
    audit_equals(errors, script, opts, "--norm_mode", config["preprocess"]["norm_mode"])
    audit_equals(errors, script, opts, "--eval_protocol", "paper_refexcluded")
    if "--skip_mimic" not in opts:
        errors.append(f"{script}: missing --skip_mimic for managed PN2021 eval")
    if "--skip_pn2021" in opts:
        errors.append(f"{script}: managed PN2021 eval must not pass --skip_pn2021")
    if "--report_drop_all_zero_pn2021" not in opts:
        errors.append(f"{script}: missing --report_drop_all_zero_pn2021")

    ref_metas = opt_list(opts, "--exclude_ref_ids")
    command_matrix = command.get("matrix") or {}
    command_case = command_matrix.get("case") if isinstance(command_matrix.get("case"), Mapping) else None
    study_variant = None
    study_seed = None
    if command_case is not None and command_case.get("study_scope"):
        try:
            study_variant, study_seed, _ = validate_f005_case(command_case)
            expected_seed = str(study_seed)
        except ValueError as exc:
            errors.append(f"{script}: {exc}")
    expected_suffixes = {
        f"{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
        for center in target_centers
    }
    found_suffixes = {str(path).split("/")[-1] for path in ref_metas}
    if expected_suffixes != found_suffixes:
        errors.append(
            f"{script}: exclude_ref_ids={sorted(found_suffixes)!r}, "
            f"expected {sorted(expected_suffixes)!r}"
        )

    matrix_center = str(command_matrix.get("center", ""))
    matrix_arm = str((command_case or {}).get("arm") or command_matrix.get("arm", ""))
    if matrix_center and matrix_center not in target_centers:
        errors.append(f"{script}: unexpected matrix center {matrix_center!r}")
    if matrix_arm:
        if "--checkpoint_name" not in opts:
            errors.append(f"{script}: missing required option --checkpoint_name")
        try:
            expected_model_dir = (
                build_f005_control_producer_dir(
                    config,
                    center=matrix_center,
                    seed=int(study_seed),
                    variant=str(study_variant),
                )
                if study_variant is not None and study_seed is not None
                else build_matched_effnet_producer_dir(
                    config, center=matrix_center, arm=matrix_arm
                )
            )
            audit_equals(errors, script, opts, "--model_dir", expected_model_dir)
        except ValueError as exc:
            errors.append(f"{script}: {exc}")
        audit_equals(errors, script, opts, "--checkpoint_name", "best_model.pt")
        output_path = str(opt_first(opts, "--output_path", ""))
        expected_role = str(study_variant) if study_variant is not None else matrix_arm
        if f"/{expected_role}/" not in output_path:
            errors.append(f"{script}: output_path must include role {expected_role!r}")

    return {"errors": errors, "warnings": warnings}
