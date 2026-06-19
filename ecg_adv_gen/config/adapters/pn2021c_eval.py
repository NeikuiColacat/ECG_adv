"""Typed YAML adapter for PN2021-C corruption eval commands."""

from __future__ import annotations

from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, opt_first, opt_list


PN2021C_REQUIRED_CACHE_VERSION = "v7_refexcluded_100hz1000"


def _format_model_dir_template(
    template: str,
    *,
    center: str,
    method: Mapping[str, Any],
    paths: Mapping[str, Any],
    runtime: Mapping[str, Any],
    eval_seed: Any,
    epochs: Any,
    k: Any,
) -> str:
    return template.format(
        center=center,
        method_name=method.get("name", ""),
        family=method.get("family", ""),
        output_root=paths["output_root"],
        run_id=runtime.get("run_id", ""),
        eval_seed=eval_seed,
        epochs=epochs,
        k=k,
    )


def build_pn2021c_eval_argv(config: Mapping[str, Any], context: Mapping[str, Any]) -> list[Any]:
    """Build argv for ``scripts/triple_labels/eval_pn2021_corruptions.py``."""

    matrix = context.get("matrix") or {}
    center = matrix.get("center")
    method = matrix.get("method") or {}
    if not center:
        raise ValueError("pn2021c_eval adapter requires runner.matrix.center")
    if not isinstance(method, Mapping) or not method.get("family") or not method.get("name"):
        raise ValueError("pn2021c_eval adapter requires runner.matrix.method name/family")

    paths = config["paths"]
    paper = config["paper_protocol"]
    kshot = paper["kshot"]
    model = config["model"]
    training = config["training"]
    data = config["data"]
    evaluation = config["evaluation"]
    preprocess = config["preprocess"]
    runtime = config.get("runtime") or {}
    experiment = config["experiment"]
    severity_profile = str(evaluation["severity_profile"])
    corruption_input = evaluation.get("corruption_input")
    output_stem = str(
        evaluation.get(
            "output_stem",
            f"eval_pn2021_c_v7_refexcluded_stream_{severity_profile}",
        )
    )
    noaug_suffix = method.get("noaug_suffix", "")
    eval_seed = method.get("eval_seed", model["eval_seed"])
    epochs = method.get("epochs", model["epochs"])
    run_stamp = method.get("run_stamp", model["run_stamp"])
    run_leaf_stem = method.get("run_leaf_stem", model["run_leaf_stem"])
    clean_eval_name = method.get("clean_eval_name", model["clean_eval_name"])
    selection_policy = str((paper.get("selection") or {}).get("policy", ""))
    checkpoint_name = str(
        method.get("checkpoint_name")
        or model.get("checkpoint_name")
        or ("last_model.pt" if selection_policy == "last_checkpoint_only" else "best_model.pt")
    )
    run_leaf = (
        f"{center}_{run_leaf_stem}_fullft_k{kshot['k']}"
        f"{noaug_suffix}_ep{epochs}_seed{eval_seed}"
    )
    model_dir_template = method.get("model_dir_template") or method.get("model_dir")
    if model_dir_template:
        method_root = _format_model_dir_template(
            str(model_dir_template),
            center=str(center),
            method=method,
            paths=paths,
            runtime=runtime,
            eval_seed=eval_seed,
            epochs=epochs,
            k=kshot["k"],
        )
    else:
        method_root = (
            f"{paths['output_root']}/{method['family']}/"
            f"seed{eval_seed}_k{kshot['k']}_v7_sjr_rgq_{run_stamp}/{run_leaf}"
        )

    argv: list[Any] = [
        "--mode",
        "stream",
        "--scheme",
        "super5",
        "--model_dir",
        method_root,
        "--clean_mmap_cache_dir",
        data["cache"]["pn2021_clean_mmap_cache_dir"],
        "--clean_cache_dir",
        data["cache"]["pn2021_clean_cache_dir"],
        "--clean_eval_json",
        f"{method_root}/{clean_eval_name}",
        "--checkpoint_name",
        checkpoint_name,
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
        "--exclude_ref_ids",
        (
            f"{data['kshot_subset_root']}/{center}/k{kshot['k']}_seed{kshot['seed']}/"
            f"{center}_real_k{kshot['k']}_seed{kshot['seed']}.ref_meta.json"
        ),
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
    if corruption_input:
        argv.extend(
            [
                "--corruption_input",
                str(corruption_input),
                "--pn2021_root",
                f"{paths['data_root']}/physionet2021",
            ]
        )
    return argv


def audit_pn2021c_eval_command(command: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return protocol audit errors for PN2021-C ref-excluded eval commands."""

    errors: list[str] = []
    warnings: list[str] = []
    argv = [str(x) for x in command["argv"]]
    opts = argv_option_map(argv)
    script = "eval_pn2021_corruptions.py"
    kshot = config["paper_protocol"]["kshot"]
    expected_k = str(kshot["k"])
    expected_seed = str(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--mode",
            "--scheme",
            "--model_dir",
            "--clean_mmap_cache_dir",
            "--clean_cache_dir",
            "--clean_eval_json",
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
            "--exclude_ref_ids",
            "--output_path",
        ],
    )
    audit_equals(errors, script, opts, "--mode", "stream")
    audit_equals(errors, script, opts, "--scheme", "super5")
    audit_equals(errors, script, opts, "--required_cache_version", PN2021C_REQUIRED_CACHE_VERSION)
    audit_equals(errors, script, opts, "--severity_profile", config["evaluation"]["severity_profile"])
    if str(config["evaluation"]["severity_profile"]) == "custom":
        audit_require_options(errors, script, opts, ["--severity_params_file", "--severity_params_name"])
        audit_equals(errors, script, opts, "--severity_params_file", config["evaluation"]["severity_params_file"])
        audit_equals(errors, script, opts, "--severity_params_name", config["evaluation"]["severity_params_name"])
    audit_equals(errors, script, opts, "--crop_len", config["preprocess"]["crop_len"])
    audit_equals(errors, script, opts, "--min_pos", config["evaluation"]["min_pos"])
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
    configured_corruption_input = config["evaluation"].get("corruption_input")
    if configured_corruption_input:
        audit_equals(errors, script, opts, "--corruption_input", configured_corruption_input)
        audit_require_options(errors, script, opts, ["--pn2021_root"])
        if str(configured_corruption_input) == "raw_first" and "official_s5_locked" in str(
            config.get("experiment", {}).get("name", "")
        ):
            output_path = str(opt_first(opts, "--output_path", ""))
            if "official_s5_locked" not in output_path:
                errors.append(f"{script}: locked PN2021-C output_path must include official_s5_locked")

    centers = opt_list(opts, "--centers")
    matrix_center = str((command.get("matrix") or {}).get("center", ""))
    if len(centers) != 1:
        errors.append(f"{script}: managed PN2021-C eval must target exactly one center per command")
    center = centers[0] if centers else matrix_center
    if center not in target_centers:
        errors.append(f"{script}: unexpected center {center!r}")
    if matrix_center and center != matrix_center:
        errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {centers!r}")

    ref_metas = opt_list(opts, "--exclude_ref_ids")
    expected_ref_name = f"{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
    if [str(path).split("/")[-1] for path in ref_metas] != [expected_ref_name]:
        errors.append(
            f"{script}: exclude_ref_ids={[str(path).split('/')[-1] for path in ref_metas]!r}, "
            f"expected {[expected_ref_name]!r}"
        )

    clean_eval = str(opt_first(opts, "--clean_eval_json", ""))
    model_dir = str(opt_first(opts, "--model_dir", ""))
    method = (command.get("matrix") or {}).get("method") or {}
    expected_clean_name = "eval_result_v7_exclrefs_crop1000.json"
    if isinstance(method, Mapping):
        expected_clean_name = str(
            method.get("clean_eval_name")
            or config.get("model", {}).get("clean_eval_name")
            or expected_clean_name
        )
    if model_dir and clean_eval != f"{model_dir}/{expected_clean_name}":
        errors.append(f"{script}: clean_eval_json must point inside --model_dir")
    if "--diagnostic_without_clean" in opts:
        errors.append(f"{script}: managed PN2021-C eval must require a clean paper-safe eval JSON")
    if "--limit" in opts and str(opt_first(opts, "--limit")) not in {"0", ""}:
        errors.append(f"{script}: managed main PN2021-C eval must not limit samples")
    return {"errors": errors, "warnings": warnings}
