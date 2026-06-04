"""Run-plan file materialization for YAML-managed legacy launches."""

from __future__ import annotations

import json
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


def render_launch_command(command: dict[str, Any]) -> str:
    """Render a managed child/postprocess command for durable plan files."""

    return render_command(command["argv"], env=command.get("env") or {})


def experiment_purpose(config: dict[str, Any]) -> str:
    """Return the human-readable run purpose from a resolved experiment config."""

    experiment = config.get("experiment") or {}
    return str(experiment.get("purpose") or experiment.get("description") or "")


def write_launch_plan_files(
    out_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    commands: list[dict[str, Any]],
    postprocess_commands: list[dict[str, Any]],
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
    write_k500_ref_ids_artifact(manifest, out_dir / "k500_ref_ids.json")
    write_selection_record_artifact(manifest, out_dir / "selection.json")
    manifest = attach_launch_artifacts(manifest, run_dir=out_dir)
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
