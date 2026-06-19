"""Typed YAML adapter for locked ECGFounder PN2021-C corruption eval commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, opt_first, opt_list
from .pn2021c_eval import PN2021C_REQUIRED_CACHE_VERSION


def _format_run_dir_template(
    template: str,
    *,
    center: str,
    method: Mapping[str, Any],
    paths: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    return template.format(
        center=center,
        method_name=method.get("name", ""),
        family=method.get("family", ""),
        output_root=paths["output_root"],
        run_id=runtime.get("run_id", ""),
    )


def build_ecgfounder_pn2021c_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py``."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    method = matrix.get("method") or {}
    if not center:
        raise ValueError("ecgfounder_pn2021c_eval adapter requires runner.matrix.center")
    if not isinstance(method, Mapping) or not method.get("family") or not method.get("name"):
        raise ValueError("ecgfounder_pn2021c_eval adapter requires runner.matrix.method name/family")

    paths = config["paths"]
    paper = config["paper_protocol"]
    model = config["model"]
    training = config["training"]
    data = config["data"]
    evaluation = config["evaluation"]
    preprocess = config["preprocess"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]
    severity_profile = str(evaluation["severity_profile"])
    output_stem = str(
        evaluation.get(
            "output_stem",
            f"eval_pn2021_c_ecgfounder_v7_refexcluded_stream_{severity_profile}",
        )
    )
    corruption_input = str(evaluation.get("corruption_input", "bottleneck5000"))
    run_dir_template = method.get("run_dir_template") or method.get("run_dir")
    if not run_dir_template:
        raise ValueError("ecgfounder_pn2021c_eval adapter requires method.run_dir_template")
    run_dir = _format_run_dir_template(
        str(run_dir_template),
        center=str(center),
        method=method,
        paths=paths,
        runtime=runtime,
    )

    argv: list[Any] = [
        "--scheme",
        "super5",
        "--run_dir",
        run_dir,
        "--variant",
        method["name"],
        "--checkpoint",
        model["checkpoint"],
        "--clean_mmap_cache_dir",
        data["cache"]["pn2021_clean_mmap_cache_dir"],
        "--clean_cache_dir",
        data["cache"]["pn2021_clean_cache_dir"],
        "--required_cache_version",
        PN2021C_REQUIRED_CACHE_VERSION,
        "--centers",
        center,
        "--corruptions",
        *list(evaluation["corruptions"]),
        "--severities",
        *list(evaluation["severities"]),
        "--severity_profile",
        evaluation["severity_profile"],
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
        "--seed",
        evaluation["corruption_seed"],
        "--corruption_input",
        corruption_input,
        "--pn2021_root",
        f"{paths['data_root']}/physionet2021",
        "--output_path",
        (
            f"{paths['output_root']}/{experiment['name']}/{runtime['run_id']}/{center}/"
            f"{method['name']}/{output_stem}.json"
        ),
    ]
    if severity_profile == "custom":
        argv.extend(
            [
                "--severity_params_file",
                evaluation["severity_params_file"],
                "--severity_params_name",
                evaluation["severity_params_name"],
            ]
        )
    return argv


def audit_ecgfounder_pn2021c_eval_command(
    command: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return protocol audit errors for locked ECGFounder PN2021-C eval commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    opts = argv_option_map(argv)
    script = "eval_ecgfounder_pn2021_corruptions.py"
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
            "--run_dir",
            "--variant",
            "--checkpoint",
            "--clean_mmap_cache_dir",
            "--clean_cache_dir",
            "--required_cache_version",
            "--centers",
            "--corruptions",
            "--severities",
            "--severity_profile",
            "--device",
            "--crop_len",
            "--batch_size",
            "--num_workers",
            "--min_pos",
            "--seed",
            "--corruption_input",
            "--pn2021_root",
            "--output_path",
        ],
    )
    audit_equals(errors, script, opts, "--scheme", "super5")
    audit_equals(errors, script, opts, "--required_cache_version", PN2021C_REQUIRED_CACHE_VERSION)
    audit_equals(errors, script, opts, "--severity_profile", config["evaluation"]["severity_profile"])
    if str(config["evaluation"]["severity_profile"]) == "custom":
        audit_require_options(errors, script, opts, ["--severity_params_file", "--severity_params_name"])
        audit_equals(errors, script, opts, "--severity_params_file", config["evaluation"]["severity_params_file"])
        audit_equals(errors, script, opts, "--severity_params_name", config["evaluation"]["severity_params_name"])
    audit_equals(errors, script, opts, "--crop_len", config["preprocess"]["crop_len"])
    audit_equals(errors, script, opts, "--min_pos", config["evaluation"]["min_pos"])
    audit_equals(errors, script, opts, "--corruption_input", config["evaluation"].get("corruption_input", ""))
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")

    forbidden = [
        "--ecgfounder_input_bandpass_low_hz",
        "--ecgfounder_input_bandpass_high_hz",
        "--ecgfounder_input_repair_flat_leads",
        "--ecgfounder_input_clip_abs",
        "--recompute_clean_with_input_stabilizer",
    ]
    present_forbidden = [name for name in forbidden if name in opts]
    if present_forbidden:
        errors.append(f"{script}: locked main eval must not enable input stabilizer options: {present_forbidden}")

    centers = opt_list(opts, "--centers")
    matrix_center = str((command.get("matrix") or {}).get("center", ""))
    if len(centers) != 1:
        errors.append(f"{script}: managed ECGFounder PN2021-C eval must target exactly one center per command")
    center = centers[0] if centers else matrix_center
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    if matrix_center and center != matrix_center:
        errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {centers!r}")

    output_path = str(opt_first(opts, "--output_path", ""))
    if "official_s5_locked" not in output_path:
        errors.append(f"{script}: locked PN2021-C output_path must include official_s5_locked")

    run_dir = str(opt_first(opts, "--run_dir", ""))
    method = (command.get("matrix") or {}).get("method") or {}
    run_id = str((config.get("runtime") or {}).get("run_id") or "")
    if isinstance(method, Mapping):
        template = method.get("run_dir_template") or method.get("run_dir")
        if template:
            expected_run_dir = _format_run_dir_template(
                str(template),
                center=center,
                method=method,
                paths=config["paths"],
                runtime=config.get("runtime") or {},
            )
            if run_dir != expected_run_dir:
                errors.append(f"{script}: run_dir does not match locked method template")
    if run_id and "/ecgfounder_vae_lhat_augmix_threechain_locked_k500/" not in run_dir:
        errors.append(f"{script}: run_dir must target the locked ECGFounder three-chain run family")

    if "--limit" in opts and str(opt_first(opts, "--limit")) not in {"0", ""}:
        errors.append(f"{script}: managed main ECGFounder PN2021-C eval must not limit samples")
    return {"errors": errors, "warnings": warnings}
