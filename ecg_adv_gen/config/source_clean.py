"""Fail-closed source-control checks for managed experiment execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from .launch import LaunchError


class ExecutionSourceError(LaunchError):
    """Raised when an execution-defining source is not clean and tracked."""


def _resolve(path: object, repo_root: Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _source_paths(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    index_path: Path | None,
) -> tuple[list[tuple[Path, tuple[str, ...]]], list[Path], list[str]]:
    roles: dict[Path, set[str]] = {}
    errors: list[str] = []

    def add(path: object, role: str) -> None:
        if not str(path or "").strip():
            errors.append(f"missing declared {role} path")
            return
        resolved = _resolve(path, repo_root)
        roles.setdefault(resolved, set()).add(role)

    excluded_local = [
        _resolve(source, repo_root)
        for source in config.get("_local_config_sources") or ()
        if str(source or "").strip()
    ]
    excluded_local_set = set(excluded_local)
    add(index_path or repo_root / "configs/active_scripts.yaml", "active_script_index")
    add(config.get("_entry_config"), "entry_config")
    for source in config.get("_config_sources") or ():
        if str(source or "").strip() and _resolve(source, repo_root) in excluded_local_set:
            continue
        add(source, "experiment_config_source")
    ordered = [
        (path, tuple(sorted(path_roles)))
        for path, path_roles in sorted(roles.items(), key=lambda item: str(item[0]))
    ]
    return ordered, excluded_local, errors


def _inspect_path(repo_root: Path, path: Path, roles: tuple[str, ...]) -> dict[str, Any]:
    try:
        repo_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        return {
            "path": str(path),
            "repo_path": "",
            "roles": list(roles),
            "exists": path.exists(),
            "tracked": False,
            "staged": False,
            "unstaged": False,
            "untracked": False,
            "missing": not path.exists(),
            "outside_repo": True,
            "state": "outside_repo",
            "git_status": [],
        }

    status_result = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", repo_path],
    )
    tracked_result = _run_git(repo_root, ["ls-files", "--error-unmatch", "--", repo_path])
    git_error = status_result.returncode != 0
    status_lines = [line for line in status_result.stdout.splitlines() if line]
    codes = [line[:2] for line in status_lines]
    exists = path.exists()
    tracked = tracked_result.returncode == 0
    staged = any(code[0] not in {" ", "?", "!"} for code in codes)
    unstaged = any(code[1] not in {" ", "?", "!"} for code in codes)
    untracked = any(code == "??" for code in codes) or (exists and not tracked)
    missing = not exists
    states = [
        name
        for name, active in (
            ("missing", missing),
            ("staged", staged),
            ("unstaged", unstaged),
            ("untracked", untracked),
            ("git_error", git_error),
        )
        if active
    ]
    return {
        "path": str(path),
        "repo_path": repo_path,
        "roles": list(roles),
        "exists": exists,
        "tracked": tracked,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "missing": missing,
        "outside_repo": False,
        "state": "+".join(states) if states else "clean",
        "git_status": status_lines,
        **(
            {"git_error": status_result.stderr.strip() or "git status failed"}
            if git_error
            else {}
        ),
    }


def inspect_execution_sources(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect only execution-defining tracked YAML sources.

    Local machine overlays are deliberately recorded but excluded.  Unrelated
    workspace dirt, including external ``model/*`` handles, is never queried.
    """

    root = Path(repo_root).expanduser().resolve()
    source_paths, excluded_local, errors = _source_paths(
        config, repo_root=root, index_path=index_path
    )
    sources = [_inspect_path(root, path, roles) for path, roles in source_paths]
    blockers = [row for row in sources if row["state"] != "clean"]
    return {
        "passed": not errors and not blockers,
        "repo_root": str(root),
        "sources": sources,
        "blocking_sources": blockers,
        "errors": errors,
        "excluded_local_config_sources": [str(path) for path in excluded_local],
    }


def require_clean_execution_sources(
    report: Mapping[str, Any], *, phase: str
) -> None:
    """Raise before execution when any managed experiment source is unsafe."""

    if report.get("passed") is True:
        return
    blockers = report.get("blocking_sources") or ()
    rendered = ", ".join(
        f"{row.get('repo_path') or row.get('path')}[{row.get('state')}]"
        for row in blockers
    )
    errors = "; ".join(str(item) for item in report.get("errors") or ())
    details = "; ".join(item for item in (rendered, errors) if item)
    raise ExecutionSourceError(
        f"{phase}: managed execution sources must be clean, tracked, and present: {details}"
    )
