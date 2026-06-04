"""CPU-only audit helpers for YAML-managed experiment configs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
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


def _git_file_status(repo_root: Path, rel_path: str) -> dict[str, Any]:
    tracked_proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    status_proc = subprocess.run(
        ["git", "status", "--short", "--", rel_path],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    tracked = tracked_proc.returncode == 0
    if status_proc.returncode != 0:
        return {
            "config_exists": (repo_root / rel_path).exists(),
            "config_tracked_by_git": tracked,
            "config_git_status": "git_status_failed",
            "config_index_status": "?",
            "config_worktree_status": "?",
            "config_intent_to_add": False,
        }
    lines = [line for line in status_proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "config_exists": (repo_root / rel_path).exists(),
            "config_tracked_by_git": tracked,
            "config_git_status": "clean" if tracked else "untracked_or_missing",
            "config_index_status": " ",
            "config_worktree_status": " ",
            "config_intent_to_add": False,
        }

    raw_status = lines[0][:2]
    index_status = raw_status[0] if len(raw_status) > 0 else " "
    worktree_status = raw_status[1] if len(raw_status) > 1 else " "
    intent_to_add = tracked and index_status == " " and worktree_status == "A"
    return {
        "config_exists": (repo_root / rel_path).exists(),
        "config_tracked_by_git": tracked,
        "config_git_status": raw_status.strip() or "modified",
        "config_index_status": index_status,
        "config_worktree_status": worktree_status,
        "config_intent_to_add": intent_to_add,
    }


def _summarize_config_git_status(
    items: list[dict[str, Any]],
    *,
    tracked_yaml_required: bool,
) -> dict[str, Any]:
    untracked_paths: list[str] = []
    missing_paths: list[str] = []
    dirty_paths: list[str] = []
    intent_to_add_paths: list[str] = []
    staged_content_paths: list[str] = []
    unstaged_content_paths: list[str] = []
    clean_count = 0
    tracked_count = 0
    by_git_status: dict[str, int] = {}

    for item in items:
        path = str(item["config"])
        git_status = str(item["config_git_status"])
        by_git_status[git_status] = by_git_status.get(git_status, 0) + 1
        if bool(item["config_tracked_by_git"]):
            tracked_count += 1
        if not bool(item["config_exists"]):
            missing_paths.append(path)
        if not bool(item["config_tracked_by_git"]):
            untracked_paths.append(path)
        if bool(item["config_intent_to_add"]):
            intent_to_add_paths.append(path)

        has_staged_content = item["config_index_status"] not in {" ", "?"}
        has_unstaged_content = item["config_worktree_status"] not in {" ", "?"}
        if has_staged_content and not bool(item["config_intent_to_add"]):
            staged_content_paths.append(path)
        if has_unstaged_content:
            unstaged_content_paths.append(path)
        if git_status == "clean":
            clean_count += 1
        else:
            dirty_paths.append(path)

    return {
        "tracked_yaml_required": tracked_yaml_required,
        "total_count": len(items),
        "tracked_count": tracked_count,
        "clean_count": clean_count,
        "dirty_count": len(dirty_paths),
        "untracked_count": len(untracked_paths),
        "missing_count": len(missing_paths),
        "intent_to_add_count": len(intent_to_add_paths),
        "staged_content_count": len(staged_content_paths),
        "unstaged_content_count": len(unstaged_content_paths),
        "by_git_status": dict(sorted(by_git_status.items())),
        "dirty_paths": dirty_paths,
        "untracked_paths": untracked_paths,
        "missing_paths": missing_paths,
        "intent_to_add_paths": intent_to_add_paths,
        "staged_content_paths": staged_content_paths,
        "unstaged_content_paths": unstaged_content_paths,
        "requires_attention": bool(
            untracked_paths
            or missing_paths
            or intent_to_add_paths
            or staged_content_paths
            or unstaged_content_paths
        ),
    }


def _config_git_inventory(
    *,
    repo_root: Path,
    config_roots: list[str],
    managed_config_paths: set[str],
) -> list[dict[str, Any]]:
    paths = set(managed_config_paths)
    normalized_roots = [root.strip("/") for root in config_roots if root]
    for root in normalized_roots:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        paths.update(str(path.relative_to(repo_root)) for path in sorted(root_path.glob("*.yaml")))

    out: list[dict[str, Any]] = []
    for rel_path in sorted(paths):
        roots = [
            root
            for root in normalized_roots
            if rel_path == root or rel_path.startswith(f"{root}/")
        ]
        out.append(
            {
                "config": rel_path,
                "in_managed_experiments": rel_path in managed_config_paths,
                "config_roots": roots,
                **_git_file_status(repo_root, rel_path),
            }
        )
    return out


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
    config_git_status = _git_file_status(repo_root, config_rel)
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
        **config_git_status,
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
    launch_surface_policy = index.get("launch_surface_policy") or {}
    items = managed_experiment_items(index)
    rows = [
        audit_managed_experiment(
            repo_root=repo_root,
            local_config_path=local_config_path,
            item=item,
            require_existing_inputs=require_existing_inputs,
        )
        for item in items
    ]
    config_roots = launch_surface_policy.get("config_roots") or []
    if not isinstance(config_roots, list):
        config_roots = []
    managed_config_paths = {str(item.get("config") or "") for item in items if item.get("config")}
    config_git_inventory = _config_git_inventory(
        repo_root=repo_root,
        config_roots=[str(root) for root in config_roots],
        managed_config_paths=managed_config_paths,
    )
    config_git_summary = _summarize_config_git_status(
        config_git_inventory,
        tracked_yaml_required=bool(launch_surface_policy.get("tracked_yaml_required")),
    )
    return {
        "schema_version": 1,
        "index": str(index_path),
        "local_config": str(local_config_path),
        "require_existing_inputs": require_existing_inputs,
        "launch_surface_policy": launch_surface_policy,
        "documentation_surface_policy": index.get("documentation_surface_policy") or {},
        "implementation_surface_policy": index.get("implementation_surface_policy") or {},
        "managed_experiment_count": len(rows),
        "passed_count": sum(1 for row in rows if row["passed"]),
        "failed_count": sum(1 for row in rows if not row["passed"]),
        "config_git_inventory": config_git_inventory,
        "config_git_summary": config_git_summary,
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
        "config_exists",
        "config_tracked_by_git",
        "config_git_status",
        "config_index_status",
        "config_worktree_status",
        "config_intent_to_add",
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
