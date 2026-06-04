"""Audit adapter for YAML-managed PTB-XL source training commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import argv_option_map, audit_equals, audit_require_options, opt_first


def audit_train_ptbxl_command(command: Mapping[str, Any], *, expected_seed: int | str) -> list[str]:
    """Return protocol-audit errors for managed ``train_ptbxl.py`` commands."""

    errors: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = Path(argv[1]).name
    opts = argv_option_map(argv)
    matrix = command.get("matrix") or {}
    expected_command_seed = str(matrix.get("seed") or expected_seed)

    if script != "train_ptbxl.py":
        errors.append(f"{script}: source-training adapter only accepts train_ptbxl.py")
        return errors

    audit_require_options(
        errors,
        script,
        opts,
        [
            "--scheme",
            "--model_name",
            "--output_dir",
            "--data_path",
            "--csv_path",
            "--cache_path",
            "--preprocess_mode",
            "--norm_mode",
            "--device",
            "--crop_len",
            "--batch_size",
            "--epochs",
            "--lr",
            "--weight_decay",
            "--cosine_tmax",
            "--patience",
            "--num_workers",
            "--seed",
            "--checkpoint_metric",
            "--pos_weight_clip_max",
        ],
    )
    audit_equals(errors, script, opts, "--scheme", "super5")
    audit_equals(errors, script, opts, "--seed", expected_command_seed)
    audit_equals(errors, script, opts, "--preprocess_mode", "minimal_resample")
    audit_equals(errors, script, opts, "--norm_mode", "per_sample_global")
    audit_equals(errors, script, opts, "--crop_len", "1000")
    audit_equals(errors, script, opts, "--checkpoint_metric", "auprc")
    if str(opt_first(opts, "--device", "")) != "cuda":
        errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
    model_name = str(opt_first(opts, "--model_name", ""))
    matrix_model = str(matrix.get("model") or "")
    if matrix_model and model_name != matrix_model:
        errors.append(f"{script}: matrix model {matrix_model!r} must match --model_name={model_name!r}")
    output_dir = str(opt_first(opts, "--output_dir", ""))
    if "benchmark_source_v7_sjr_rgq" not in output_dir:
        errors.append(f"{script}: output_dir must be under benchmark_source_v7_sjr_rgq")
    if model_name and model_name not in output_dir:
        errors.append(f"{script}: output_dir does not encode model_name {model_name!r}")
    if f"seed{expected_command_seed}" not in output_dir:
        errors.append(f"{script}: output_dir does not encode seed{expected_command_seed}")
    if str(opt_first(opts, "--checkpoint_metric", "")) != "auprc":
        errors.append(f"{script}: benchmark source managed config must use --checkpoint_metric auprc")
    data_path = str(opt_first(opts, "--data_path", ""))
    csv_path = str(opt_first(opts, "--csv_path", ""))
    cache_path = str(opt_first(opts, "--cache_path", ""))
    if not data_path.endswith("/ptbxl/raw100.npy"):
        errors.append(f"{script}: data_path should point at ptbxl/raw100.npy")
    if not csv_path.endswith("/ptbxl/ptbxl_database.csv"):
        errors.append(f"{script}: csv_path should point at ptbxl/ptbxl_database.csv")
    if "ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy" not in cache_path:
        errors.append(f"{script}: cache_path should use the minimal_resample/per_sample_global fs100 len1000 cache")
    if "--synth_npz" in opts or "--synthetic_only" in opts:
        errors.append(f"{script}: benchmark source pretrain must not include synthetic data options")
    if "--init_ckpt" in opts:
        errors.append(f"{script}: benchmark source pretrain must not initialize from a target checkpoint")
    return errors
