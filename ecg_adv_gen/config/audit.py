"""CPU-only audit helpers for YAML-managed experiment configs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .launch import verify_required_inputs
from .loader import (
    ConfigError,
    build_postprocess_commands,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)


def load_active_script_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"active script index must be a mapping: {path}")
    return data


def managed_experiment_items(index: dict[str, Any], *, status: str = "active_wrapped") -> list[dict[str, Any]]:
    items = index.get("managed_experiments") or []
    if not isinstance(items, list):
        raise ConfigError("configs/active_scripts.yaml managed_experiments must be a list")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ConfigError("managed_experiments entries must be mappings")
        if item.get("status") == status:
            out.append(item)
    return out


def _count_records(records: list[dict[str, Any]], role: str) -> int:
    return sum(1 for record in records if record.get("role") == role)


def audit_managed_experiment(
    *,
    repo_root: Path,
    local_config_path: Path,
    item: dict[str, Any],
    require_existing_inputs: bool = False,
) -> dict[str, Any]:
    name = str(item.get("name") or "")
    config_rel = str(item.get("config") or "")
    if not name or not config_rel:
        raise ConfigError(f"managed experiment item is missing name/config: {item}")

    config_path = repo_root / config_rel
    config = load_experiment_config(
        config_path,
        local_config_path,
        runtime_context={"run_id": f"audit_{name}"},
    )
    local_paths = validate_experiment_config(config, repo_root=repo_root)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=local_paths,
        run_id=f"audit_{name}",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        postprocess_commands=postprocess_commands,
    )
    trace = manifest["artifact_trace"]
    inputs = trace["inputs"]
    child_runs = trace["expected_outputs"]["child_runs"]
    postprocess_runs = trace["expected_outputs"].get("postprocess_runs") or []
    command_audit = trace["protocol_audit"]["command_audit"]
    input_verification = verify_required_inputs(manifest) if require_existing_inputs else None

    passed = bool(command_audit.get("passed")) and bool(trace["protocol_audit"].get("passed"))
    if input_verification is not None:
        passed = passed and bool(input_verification.get("passed"))

    checkpoints = inputs.get("checkpoints") or []
    init_heads = inputs.get("init_heads") or []
    k500_refs = inputs.get("k500_refs") or []
    data_caches = inputs.get("data_caches") or []
    return {
        "name": name,
        "status": item.get("status"),
        "config": config_rel,
        "legacy_entrypoint": item.get("legacy_entrypoint"),
        "method_family": item.get("method_family"),
        "mapping_version": trace["metrics"]["mapping_version"],
        "mapping_hash": trace["metrics"]["mapping_hash"],
        "command_count": len(commands),
        "child_run_count": len(child_runs),
        "postprocess_command_count": len(postprocess_commands),
        "postprocess_artifact_count": sum(len(run.get("expected_artifacts") or []) for run in postprocess_runs),
        "k500_ref_count": len(k500_refs),
        "checkpoint_count": len(checkpoints),
        "init_head_count": len(init_heads),
        "data_cache_count": len(data_caches),
        "required_input_count": (
            _count_records(checkpoints, "model.init_checkpoint")
            + _count_records(checkpoints, "model.checkpoint")
            + len([r for r in checkpoints if r.get("required") is not False and r.get("role") not in {"model.init_checkpoint", "model.checkpoint"}])
            + len([r for r in init_heads if r.get("required") is not False])
            + len([r for r in data_caches if r.get("required") is not False])
            + sum(
                int(ref.get("ref_meta_json") is not None)
                + int(ref.get("signals_npz") is not None)
                + int(ref.get("latent_npz") is not None)
                for ref in k500_refs
            )
        ),
        "command_audit_passed": bool(command_audit.get("passed")),
        "protocol_audit_passed": bool(trace["protocol_audit"].get("passed")),
        "input_verification_passed": None if input_verification is None else bool(input_verification.get("passed")),
        "missing_input_count": None if input_verification is None else len(input_verification.get("missing") or []),
        "passed": passed,
    }


def audit_active_managed_configs(
    *,
    repo_root: Path,
    index_path: Path,
    local_config_path: Path,
    require_existing_inputs: bool = False,
) -> dict[str, Any]:
    index = load_active_script_index(index_path)
    rows = [
        audit_managed_experiment(
            repo_root=repo_root,
            local_config_path=local_config_path,
            item=item,
            require_existing_inputs=require_existing_inputs,
        )
        for item in managed_experiment_items(index)
    ]
    return {
        "schema_version": 1,
        "index": str(index_path),
        "local_config": str(local_config_path),
        "require_existing_inputs": require_existing_inputs,
        "managed_experiment_count": len(rows),
        "passed_count": sum(1 for row in rows if row["passed"]),
        "failed_count": sum(1 for row in rows if not row["passed"]),
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def write_audit_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "managed_config_audit.json"
    csv_path = output_dir / "managed_config_audit.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "name",
        "status",
        "config",
        "legacy_entrypoint",
        "method_family",
        "mapping_version",
        "mapping_hash",
        "command_count",
        "child_run_count",
        "postprocess_command_count",
        "postprocess_artifact_count",
        "k500_ref_count",
        "checkpoint_count",
        "init_head_count",
        "data_cache_count",
        "required_input_count",
        "command_audit_passed",
        "protocol_audit_passed",
        "input_verification_passed",
        "missing_input_count",
        "passed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return {"json": str(json_path), "csv": str(csv_path)}
