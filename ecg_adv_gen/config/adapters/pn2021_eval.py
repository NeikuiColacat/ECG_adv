"""Typed YAML adapter for clean PN2021 ref-excluded eval commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, opt_first, opt_list


def build_pn2021_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``scripts/triple_labels/eval_crosscenter.py``."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    if not center:
        raise ValueError("pn2021_eval adapter requires runner.matrix.center")

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

    return [
        "--scheme",
        "super5",
        "--model_dir",
        f"{model['direct_init_root']}/{center}_K{kshot['k']}_direct_ft_ep30_seed{kshot['seed']}_val0.2",
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
        *list(evaluation["ref_exclusion_meta"]),
        "--output_path",
        f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}/{center}/eval_result_v7_super5_sjr_rgq_refexcluded.json",
    ]


def audit_pn2021_eval_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return protocol audit errors for clean PN2021 ref-excluded eval commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    opts = argv_option_map(argv)
    script = "eval_crosscenter.py"
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

    matrix_center = str((command.get("matrix") or {}).get("center", ""))
    if matrix_center and matrix_center not in target_centers:
        errors.append(f"{script}: unexpected matrix center {matrix_center!r}")

    experiment_name = str((config.get("experiment") or {}).get("name") or "")
    run_id = str((config.get("runtime") or {}).get("run_id") or "")
    if experiment_name == "ecgtwin_prompt_token_online_at_minimal_eval":
        center = matrix_center or str(opt_first(opts, "--center", ""))
        model_dir = str(opt_first(opts, "--model_dir", ""))
        output_path = str(opt_first(opts, "--output_path", ""))
        expected_model_dir_suffix = f"/ecgtwin_prompt_token_online_at_minimal/{run_id}/{center}"
        expected_output_suffix = (
            f"/ecgtwin_prompt_token_online_at_minimal_eval/{run_id}/{center}/"
            "eval_result_v7_super5_sjr_rgq_refexcluded.json"
        )
        audit_equals(errors, script, opts, "--model_name", "efficientnet1dv2")
        audit_equals(errors, script, opts, "--device", "cuda")
        if run_id and not model_dir.endswith(expected_model_dir_suffix):
            errors.append(f"{script}: model_dir must point at same-run prompt-token online-AT output")
        if run_id and not output_path.endswith(expected_output_suffix):
            errors.append(f"{script}: output_path must be run-id-scoped under ecgtwin_prompt_token_online_at_minimal_eval")

    return {"errors": errors, "warnings": warnings}
