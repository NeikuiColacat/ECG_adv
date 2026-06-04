#!/usr/bin/env python3
"""CPU-only audit of the active agent operating layer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import ConfigError, audit_active_managed_configs  # noqa: E402
from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import EvidenceAuditError, audit_active_evidence_registry  # noqa: E402
from ecg_adv_gen.evidence.registry import _dirty_status_layer  # noqa: E402


def current_handoff_note_status() -> dict:
    rel_path = "docs/codex-handoffs/current_workspace_handoff.md"
    note_path = REPO_ROOT / rel_path
    tracked, git_status, index_status, worktree_status, intent_to_add = _git_ls_file_status(rel_path)
    return {
        "path": rel_path,
        "exists": note_path.exists(),
        "tracked_by_git": tracked,
        "git_status": git_status,
        "index_status": index_status,
        "worktree_status": worktree_status,
        "intent_to_add": intent_to_add,
    }


def _repo_relative_text(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path_text


def _git_ls_file_status(rel_path: str) -> tuple[bool, str, str, str, bool]:
    tracked_proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status_proc = subprocess.run(
        ["git", "status", "--short", "--", rel_path],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    tracked = tracked_proc.returncode == 0
    if status_proc.returncode != 0:
        return tracked, "git_status_failed", "?", "?", False
    lines = [line for line in status_proc.stdout.splitlines() if line.strip()]
    if not lines:
        return tracked, "clean" if tracked else "untracked_or_missing", " ", " ", False
    raw_status = lines[0][:2]
    index_status = raw_status[0] if len(raw_status) > 0 else " "
    worktree_status = raw_status[1] if len(raw_status) > 1 else " "
    git_status = raw_status.strip() or "modified"
    intent_to_add = tracked and index_status == " " and worktree_status == "A"
    return tracked, git_status, index_status, worktree_status, intent_to_add


def _git_diff_numstat(rel_path: str, *, cached: bool = False) -> dict:
    cmd = ["git", "diff", "--numstat"]
    if cached:
        cmd.append("--cached")
    cmd.extend(["--", rel_path])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    out = {
        "has_diff": False,
        "added_lines": 0,
        "deleted_lines": 0,
        "binary": False,
    }
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        out["has_diff"] = True
        added, deleted = parts[0], parts[1]
        if added == "-" or deleted == "-":
            out["binary"] = True
            continue
        out["added_lines"] += int(added)
        out["deleted_lines"] += int(deleted)
    return out


def _source_of_truth_items(active_scripts_index: str) -> list[tuple[str, str]]:
    active_scripts_index = _repo_relative_text(active_scripts_index)
    items = [("active_evidence_registry", "configs/active_evidence_registry.yaml")]
    if active_scripts_index:
        items.append(("active_scripts_index", active_scripts_index))
    index_path = REPO_ROOT / active_scripts_index if active_scripts_index else None
    if index_path is None or not index_path.exists():
        return items
    data = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    for name, rel_path in (data.get("source_of_truth") or {}).items():
        item = (str(name), str(rel_path))
        if item not in items:
            items.append(item)
    return items


def source_of_truth_status(active_scripts_index: str) -> list[dict]:
    out: list[dict] = []
    for name, rel_path in _source_of_truth_items(active_scripts_index):
        rel_path = _repo_relative_text(rel_path)
        path = REPO_ROOT / rel_path
        tracked, git_status, index_status, worktree_status, intent_to_add = _git_ls_file_status(rel_path)
        out.append(
            {
                "name": name,
                "path": rel_path,
                "exists": path.exists(),
                "tracked_by_git": tracked,
                "git_status": git_status,
                "index_status": index_status,
                "worktree_status": worktree_status,
                "intent_to_add": intent_to_add,
                "staged_diff": _git_diff_numstat(rel_path, cached=True),
                "unstaged_diff": _git_diff_numstat(rel_path),
                "layer": _dirty_status_layer(rel_path),
                "required_for_handoff": True,
            }
        )
    return out


def summarize_source_of_truth_status(items: list[dict]) -> dict:
    by_git_status: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    by_layer_status: dict[str, dict] = {}
    missing_paths: list[str] = []
    untracked_paths: list[str] = []
    dirty_paths: list[str] = []
    intent_to_add_paths: list[str] = []
    staged_content_paths: list[str] = []
    unstaged_content_paths: list[str] = []
    mixed_index_worktree_paths: list[str] = []
    clean_count = 0

    for item in items:
        git_status = str(item["git_status"])
        layer = str(item["layer"])
        by_git_status[git_status] = by_git_status.get(git_status, 0) + 1
        by_layer[layer] = by_layer.get(layer, 0) + 1
        layer_summary = by_layer_status.setdefault(
            layer,
            {
                "total_count": 0,
                "clean_count": 0,
                "dirty_count": 0,
                "intent_to_add_count": 0,
                "staged_content_count": 0,
                "unstaged_content_count": 0,
                "mixed_index_worktree_count": 0,
                "paths": [],
            },
        )
        layer_summary["total_count"] += 1
        layer_summary["paths"].append(str(item["path"]))

        path = str(item["path"])
        has_staged_content = str(item.get("index_status", " ")) not in {" ", "?"}
        has_unstaged_content = str(item.get("worktree_status", " ")) not in {" ", "?"}
        if not bool(item["exists"]):
            missing_paths.append(path)
        elif not bool(item["tracked_by_git"]):
            untracked_paths.append(path)
        elif git_status == "clean":
            clean_count += 1
            layer_summary["clean_count"] += 1
        else:
            dirty_paths.append(path)
            layer_summary["dirty_count"] += 1
            if has_staged_content:
                staged_content_paths.append(path)
                layer_summary["staged_content_count"] += 1
            if has_unstaged_content:
                unstaged_content_paths.append(path)
                layer_summary["unstaged_content_count"] += 1
            if has_staged_content and has_unstaged_content:
                mixed_index_worktree_paths.append(path)
                layer_summary["mixed_index_worktree_count"] += 1
            if bool(item.get("intent_to_add")):
                intent_to_add_paths.append(path)
                layer_summary["intent_to_add_count"] += 1

    return {
        "total_count": len(items),
        "clean_count": clean_count,
        "dirty_count": len(dirty_paths),
        "intent_to_add_count": len(intent_to_add_paths),
        "staged_content_count": len(staged_content_paths),
        "unstaged_content_count": len(unstaged_content_paths),
        "mixed_index_worktree_count": len(mixed_index_worktree_paths),
        "untracked_count": len(untracked_paths),
        "missing_count": len(missing_paths),
        "by_git_status": dict(sorted(by_git_status.items())),
        "by_layer": dict(sorted(by_layer.items())),
        "by_layer_status": dict(sorted(by_layer_status.items())),
        "dirty_paths": dirty_paths,
        "intent_to_add_paths": intent_to_add_paths,
        "staged_content_paths": staged_content_paths,
        "unstaged_content_paths": unstaged_content_paths,
        "mixed_index_worktree_paths": mixed_index_worktree_paths,
        "untracked_paths": untracked_paths,
        "missing_paths": missing_paths,
        "requires_attention": bool(dirty_paths or untracked_paths or missing_paths),
        "handoff_action": "review_dirty_or_untracked_source_of_truth_paths_before_handoff",
    }


def build_source_of_truth_review_queue(items: list[dict]) -> list[dict]:
    priority = {
        "mixed_index_worktree": 0,
        "staged_content": 1,
        "intent_to_add": 2,
        "unstaged_content": 3,
        "dirty": 4,
    }

    def review_state(item: dict) -> tuple[str, str]:
        has_staged = str(item.get("index_status", " ")) not in {" ", "?"}
        has_unstaged = str(item.get("worktree_status", " ")) not in {" ", "?"}
        if has_staged and has_unstaged:
            return (
                "mixed_index_worktree",
                "review_staged_and_unstaged_diffs_then_stage_or_unstage_consistently",
            )
        if has_staged:
            return ("staged_content", "review_staged_diff_before_commit")
        if bool(item.get("intent_to_add")):
            return ("intent_to_add", "review_worktree_diff_then_stage_or_declassify")
        if has_unstaged:
            return ("unstaged_content", "review_worktree_diff_then_stage_or_declassify")
        return ("dirty", "review_git_status_before_handoff")

    queue: list[dict] = []
    for item in items:
        if str(item.get("git_status")) == "clean":
            continue
        state, action = review_state(item)
        queue.append(
            {
                "path": str(item["path"]),
                "name": str(item["name"]),
                "layer": str(item["layer"]),
                "git_status": str(item["git_status"]),
                "review_state": state,
                "next_action": action,
                "staged_diff": item.get("staged_diff") or {},
                "unstaged_diff": item.get("unstaged_diff") or {},
            }
        )
    return sorted(queue, key=lambda item: (priority.get(str(item["review_state"]), 99), str(item["path"])))


def add_source_of_truth_issues(report: dict) -> None:
    contract = report.get("handoff_contract") or {}
    summary = contract.get("source_of_truth_summary") or {}
    issues = report.setdefault("issues", [])

    if summary.get("missing_paths"):
        issues.append(
            {
                "level": "error",
                "code": "source_of_truth_missing",
                "message": "Declared source-of-truth paths are missing",
                "paths": list(summary["missing_paths"]),
            }
        )
    if summary.get("untracked_paths"):
        issues.append(
            {
                "level": "warning",
                "code": "source_of_truth_untracked",
                "message": "Declared source-of-truth paths exist but are not tracked by git",
                "paths": list(summary["untracked_paths"]),
            }
        )
    if summary.get("dirty_paths"):
        issues.append(
            {
                "level": "warning",
                "code": "source_of_truth_dirty",
                "message": "Declared source-of-truth paths are dirty and need review before handoff",
                "paths": list(summary["dirty_paths"]),
            }
        )
    if summary.get("intent_to_add_paths"):
        issues.append(
            {
                "level": "warning",
                "code": "source_of_truth_intent_to_add",
                "message": "Declared source-of-truth paths are intent-to-add candidates without staged content",
                "paths": list(summary["intent_to_add_paths"]),
            }
        )
    if summary.get("staged_content_paths"):
        issues.append(
            {
                "level": "warning",
                "code": "source_of_truth_staged_content",
                "message": "Declared source-of-truth paths have staged content that needs review before commit",
                "paths": list(summary["staged_content_paths"]),
            }
        )

    error_count = sum(1 for issue in issues if issue["level"] == "error")
    warning_count = sum(1 for issue in issues if issue["level"] == "warning")
    report["error_count"] = error_count
    report["warning_count"] = warning_count
    report["passed"] = bool(report.get("passed")) and error_count == 0


def build_handoff_readiness(report: dict, source_summary: dict) -> dict:
    reasons: list[dict] = []

    def add_reason(code: str, severity: str, count: int, paths: list[str], action: str) -> None:
        if count <= 0:
            return
        reasons.append(
            {
                "code": code,
                "severity": severity,
                "count": count,
                "paths": paths,
                "handoff_action": action,
            }
        )

    add_reason(
        "source_of_truth_missing",
        "error",
        int(source_summary.get("missing_count") or 0),
        list(source_summary.get("missing_paths") or []),
        "restore_or_remove_missing_source_of_truth_paths",
    )
    add_reason(
        "source_of_truth_untracked",
        "warning",
        int(source_summary.get("untracked_count") or 0),
        list(source_summary.get("untracked_paths") or []),
        "track_or_declassify_source_of_truth_paths_before_handoff",
    )
    add_reason(
        "source_of_truth_dirty",
        "warning",
        int(source_summary.get("dirty_count") or 0),
        list(source_summary.get("dirty_paths") or []),
        "review_source_of_truth_diffs_before_handoff",
    )
    add_reason(
        "source_of_truth_intent_to_add",
        "warning",
        int(source_summary.get("intent_to_add_count") or 0),
        list(source_summary.get("intent_to_add_paths") or []),
        "review_intent_to_add_paths_then_stage_or_declassify",
    )
    add_reason(
        "source_of_truth_staged_content",
        "warning",
        int(source_summary.get("staged_content_count") or 0),
        list(source_summary.get("staged_content_paths") or []),
        "review_staged_source_of_truth_content_before_commit",
    )

    git = report.get("git") or {}
    for risk in git.get("blocking_artifact_risks") or []:
        add_reason(
            str(risk.get("risk") or "artifact_risk"),
            str(risk.get("severity") or "warning"),
            int(risk.get("count") or 0),
            list(risk.get("paths") or []),
            str(risk.get("handoff_action") or "review_artifact_risk"),
        )

    dirty_gate = ((git.get("dirty_summary") or {}).get("handoff_gate") or {})
    if dirty_gate.get("requires_attention"):
        add_reason(
            "dirty_workspace_requires_attention",
            "warning",
            int((git.get("dirty_summary") or {}).get("total_entries") or 0),
            [],
            "inspect_git_dirty_summary_handoff_gate",
        )

    config_git_summary = ((report.get("active_scripts") or {}).get("config_git_summary") or {})
    if config_git_summary.get("requires_attention"):
        add_reason(
            "managed_config_git_requires_attention",
            "warning",
            int(config_git_summary.get("dirty_count") or 0),
            list(config_git_summary.get("dirty_paths") or []),
            "inspect_active_scripts_config_git_summary",
        )

    hard_reason_count = sum(1 for reason in reasons if reason["severity"] == "error")
    warning_reason_count = sum(1 for reason in reasons if reason["severity"] == "warning")
    ready = not reasons
    source_control_ready = (
        bool(report.get("passed"))
        and hard_reason_count == 0
        and int(source_summary.get("missing_count") or 0) == 0
        and int(source_summary.get("untracked_count") or 0) == 0
        and int(source_summary.get("dirty_count") or 0) == 0
        and int(source_summary.get("intent_to_add_count") or 0) == 0
        and int(source_summary.get("staged_content_count") or 0) == 0
        and int(source_summary.get("unstaged_content_count") or 0) == 0
        and int(source_summary.get("mixed_index_worktree_count") or 0) == 0
        and int(config_git_summary.get("missing_count") or 0) == 0
        and int(config_git_summary.get("untracked_count") or 0) == 0
        and int(config_git_summary.get("dirty_count") or 0) == 0
        and int(config_git_summary.get("intent_to_add_count") or 0) == 0
        and int(config_git_summary.get("staged_content_count") or 0) == 0
        and int(config_git_summary.get("unstaged_content_count") or 0) == 0
        and not (git.get("guarded_staged_paths") or [])
    )
    return {
        "ready_for_handoff": ready,
        "ready_for_commit": ready,
        "source_control_ready": source_control_ready,
        "status": "ready" if ready else ("blocked" if hard_reason_count else "attention_required"),
        "hard_reason_count": hard_reason_count,
        "warning_reason_count": warning_reason_count,
        "reasons": reasons,
        "must_be_empty": [
            "git.guarded_staged_paths",
            "git.blocking_artifact_risks[severity=error].paths",
            "handoff_contract.source_of_truth_summary.missing_paths",
            "active_scripts.config_git_summary.missing_paths",
        ],
        "review_required": [
            "handoff_contract.source_of_truth_summary.untracked_paths",
            "handoff_contract.source_of_truth_review_queue",
            "handoff_contract.source_of_truth_summary.dirty_paths",
            "handoff_contract.source_of_truth_summary.intent_to_add_paths",
            "handoff_contract.source_of_truth_summary.staged_content_paths",
            "handoff_contract.source_of_truth_summary.unstaged_content_paths",
            "active_scripts.config_git_summary.untracked_paths",
            "active_scripts.config_git_summary.intent_to_add_paths",
            "active_scripts.config_git_summary.staged_content_paths",
            "active_scripts.config_git_summary.unstaged_content_paths",
            "git.dirty_summary.handoff_gate",
            "git.dirty_summary.by_layer",
            "git.dirty_summary.by_layer.*.paths",
        ],
    }


def build_handoff_contract(report: dict) -> dict:
    active_scripts = report.get("active_scripts") or {}
    if active_scripts.get("skipped"):
        policies = {}
        active_scripts_index = ""
    else:
        policies = {
            "evidence_surface_policy": report.get("evidence_surface_policy") or {},
            "launch_surface_policy": active_scripts.get("launch_surface_policy") or {},
            "documentation_surface_policy": active_scripts.get("documentation_surface_policy") or {},
            "implementation_surface_policy": active_scripts.get("implementation_surface_policy") or {},
        }
        active_scripts_index = str(active_scripts.get("index") or "")

    source_status = source_of_truth_status(active_scripts_index)
    source_summary = summarize_source_of_truth_status(source_status)
    return {
        "schema_version": 1,
        "startup_sequence": [
            "read_AGENTS_first_100_lines",
            "run_scripts_agent_audit_agent_workspace",
            "inspect_git_dirty_summary_handoff_gate",
            "inspect_active_scripts_policies",
            "inspect_active_scripts_config_git_summary",
        ],
        "source_of_truth": {
            "active_evidence_registry": str(report.get("registry_path") or ""),
            "active_scripts_index": active_scripts_index,
        },
        "source_of_truth_status": source_status,
        "source_of_truth_summary": source_summary,
        "source_of_truth_review_queue": build_source_of_truth_review_queue(source_status),
        "handoff_readiness": build_handoff_readiness(report, source_summary),
        "dirty_gate_json_path": "git.dirty_summary.handoff_gate",
        "current_handoff_note": current_handoff_note_status(),
        "active_scripts_json_path": "active_scripts",
        "policies": policies,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="configs/active_evidence_registry.yaml")
    p.add_argument("--active-scripts", default="configs/active_scripts.yaml")
    p.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
    p.add_argument("--output-dir", default="")
    p.add_argument("--skip-existing-artifacts", action="store_true")
    p.add_argument("--skip-git", action="store_true")
    p.add_argument("--skip-active-scripts", action="store_true")
    p.add_argument("--require-managed-inputs", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None and not is_under(output_dir, Path(args.write_boundary).expanduser().resolve()):
        print(f"[agent-audit-error] output-dir is outside write boundary: {output_dir}", file=sys.stderr)
        return 2
    try:
        report = audit_active_evidence_registry(
            repo_root=REPO_ROOT,
            registry_path=REPO_ROOT / args.registry,
            local_config_path=REPO_ROOT / args.local_config,
            require_existing_artifacts=not args.skip_existing_artifacts,
            check_git=not args.skip_git,
        )
        if args.skip_active_scripts:
            report["active_scripts"] = {"skipped": True}
        else:
            active_scripts = audit_active_managed_configs(
                repo_root=REPO_ROOT,
                index_path=REPO_ROOT / args.active_scripts,
                local_config_path=REPO_ROOT / args.local_config,
                require_existing_inputs=args.require_managed_inputs,
            )
            report["active_scripts"] = active_scripts
            report["passed"] = bool(report["passed"]) and bool(active_scripts["passed"])
            if not active_scripts["passed"]:
                report["error_count"] += int(active_scripts["failed_count"])
        report["handoff_contract"] = build_handoff_contract(report)
        add_source_of_truth_issues(report)
    except (EvidenceAuditError, ConfigError) as exc:
        print(f"[agent-audit-error] {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    print(text)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "agent_workspace_audit.json").write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
