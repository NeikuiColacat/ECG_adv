"""Run-plan file materialization for YAML-managed experiment launches."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from ecg_adv_gen.config import (
    LaunchError,
    attach_launch_artifacts,
    write_k500_ref_ids_artifact,
    write_selection_record_artifact,
)
from ecg_adv_gen.data import write_data_path_manifest
from ecg_adv_gen.evidence import finalize_run_record

from .process import render_command


def _declared_launch_artifact_names(manifest: dict[str, Any]) -> set[str]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    declared = expected.get("launch_artifacts_declared") or []
    return {str(name) for name in declared}


def _should_write_launch_artifact(manifest: dict[str, Any], artifact_name: str) -> bool:
    declared = _declared_launch_artifact_names(manifest)
    return not declared or artifact_name in declared


def render_launch_command(command: dict[str, Any]) -> str:
    """Render a managed child/postprocess command for durable plan files."""

    return render_command(command["argv"], env=command.get("env") or {})


def experiment_purpose(config: dict[str, Any]) -> str:
    """Return the human-readable run purpose from a resolved experiment config."""

    experiment = config.get("experiment") or {}
    return str(experiment.get("purpose") or experiment.get("description") or "")


def _package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for key, distribution in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("wfdb", "wfdb"),
        ("pyyaml", "PyYAML"),
    ):
        try:
            packages[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return packages


def _write_env_snapshot(out_dir: Path) -> None:
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    (out_dir / "env.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_launch_plan_files(
    out_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    commands: list[dict[str, Any]],
    postprocess_commands: list[dict[str, Any]],
    *,
    bind_matrix_resume: bool = False,
) -> dict[str, Any]:
    """Write resolved config, command manifest, and agent-readable run records."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    (out_dir / "run_config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )

    command_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    command_text += "# Legacy child commands\n"
    command_text += "\n\n".join(render_launch_command(cmd) for cmd in commands) + "\n"
    if postprocess_commands:
        command_text += "\n# Managed postprocess commands\n"
        command_text += "\n\n".join(render_launch_command(cmd) for cmd in postprocess_commands) + "\n"
    (out_dir / "command.sh").write_text(command_text, encoding="utf-8")

    write_data_path_manifest(
        config,
        out_dir / "data_manifest.json",
        local_paths=manifest.get("local_paths") or {},
    )
    _write_env_snapshot(out_dir)
    if _should_write_launch_artifact(manifest, "k500_ref_ids.json"):
        write_k500_ref_ids_artifact(manifest, out_dir / "k500_ref_ids.json")
    if _should_write_launch_artifact(manifest, "selection.json"):
        write_selection_record_artifact(manifest, out_dir / "selection.json")
    manifest = attach_launch_artifacts(manifest, run_dir=out_dir)
    if bind_matrix_resume:
        # Local import avoids coupling the general launch-plan module to the
        # matrix executor unless this explicit launch mode is requested.
        from .matrix_parallel import bind_matrix_resume_contract

        manifest = bind_matrix_resume_contract(manifest, run_dir=out_dir)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    finalize_run_record(
        out_dir,
        purpose=experiment_purpose(config),
        result_summary="Run plan files were written; execution has not completed yet.",
        outcome=str(manifest.get("status", "dry_run")),
    )
    try:
        return json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LaunchError(f"run_manifest.json is not valid JSON after plan write: {out_dir}") from exc
