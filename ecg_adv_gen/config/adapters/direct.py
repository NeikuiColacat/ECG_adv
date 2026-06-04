"""Audit adapter for YAML-managed EfficientNet direct fine-tune commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import (
    argv_option_map,
    audit_equals,
    audit_require_options,
    matrix_case,
    opt_first,
    opt_list,
)


def audit_direct_finetune_command(
    command: Mapping[str, Any],
    *,
    expected_k: int | str,
    expected_seed: int | str,
    target_centers: set[str],
) -> list[str]:
    """Return protocol-audit errors for managed Direct K-shot fine-tune commands."""

    errors: list[str] = []
    argv = [str(x) for x in command["argv"]]
    script = Path(argv[1]).name
    opts = argv_option_map(argv)
    matrix = command.get("matrix") or {}
    case = matrix_case(command)
    expected_command_k = str(case.get("k", expected_k))
    expected_command_seed = str(case.get("seed", expected_seed))

    if script != "run_direct_finetune_k500_20260516.py":
        errors.append(f"{script}: direct adapter only accepts run_direct_finetune_k500_20260516.py")
        return errors

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
    return errors
