"""External model link contracts for host-local repository handles."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import os
import subprocess
from typing import Any

from .loader import (
    ConfigError,
    _load_with_extends,
    _validate_json_schema,
    _validate_local_config_overlay,
    interpolate_config,
)
from .paths import is_under, translate_legacy_path


DEFAULT_EXTERNAL_MODEL_REPOS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("DeepECG", "https://github.com/HeartWise-AI/DeepECG_Docker.git"),
        ("ECGTwin", "https://github.com/Raiiyf/ECGTwin.git"),
        ("advdiff", "https://github.com/EricDai0/advdiff.git"),
        ("ecg_ptbxl_benchmarking", "https://github.com/helme/ecg_ptbxl_benchmarking.git"),
        ("ecgfounder", "https://github.com/PKUDigitalHealth/ECGFounder.git"),
    ]
)


def _issue(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _load_local_external_model_config(local_config_path: Path, repo_root: Path) -> dict[str, Any]:
    local_raw = _load_with_extends(local_config_path)
    _validate_local_config_overlay(local_raw)
    _validate_json_schema(local_raw, repo_root / "configs" / "schemas" / "local_config.schema.json")
    return interpolate_config(local_raw)


def _resolve_config_path(raw_value: Any, *, repo_root: Path) -> Path:
    translated = translate_legacy_path(raw_value)
    path = Path(str(translated)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _resolve_existing_link_target(handle: Path) -> Path | None:
    if not handle.is_symlink():
        return None
    raw_target = handle.readlink()
    if raw_target.is_absolute():
        return raw_target.expanduser().resolve(strict=False)
    return (handle.parent / raw_target).resolve(strict=False)


def _is_git_checkout(path: Path) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def external_model_specs(
    *,
    repo_root: Path,
    local_config_path: Path,
) -> list[dict[str, str]]:
    """Return the configured external model bootstrap/link plan."""

    repo_root = repo_root.resolve()
    config = _load_local_external_model_config(local_config_path.resolve(), repo_root)
    paths_cfg = config.get("paths") or {}
    external_cfg = config.get("external_models") or {}
    repos_cfg = external_cfg.get("repos") or {}
    model_root = _resolve_config_path(paths_cfg.get("model_root"), repo_root=repo_root)

    specs: list[dict[str, str]] = []
    for name, default_url in DEFAULT_EXTERNAL_MODEL_REPOS.items():
        item = repos_cfg.get(name) or {}
        if not isinstance(item, dict):
            raise ConfigError(f"external_models.repos.{name} must be a mapping")
        default_target = model_root / name
        target = _resolve_config_path(item.get("target") or default_target, repo_root=repo_root)
        specs.append(
            {
                "name": name,
                "url": str(item.get("url") or default_url),
                "target": str(target),
                "handle_path": str(Path("model") / name),
            }
        )
    return specs


def audit_external_model_links(
    *,
    repo_root: Path,
    local_config_path: Path,
    require_existing: bool = True,
) -> dict[str, Any]:
    """Validate tracked ``model/*`` handles against local YAML path contracts."""

    repo_root = repo_root.resolve()
    config = _load_local_external_model_config(local_config_path.resolve(), repo_root)
    paths_cfg = config.get("paths") or {}
    safety_cfg = config.get("safety") or {}
    external_cfg = config.get("external_models") or {}
    link_root = _resolve_config_path(external_cfg.get("link_root") or "model", repo_root=repo_root)
    allowed_roots_raw = external_cfg.get("allowed_target_roots") or [
        paths_cfg.get("model_root"),
        paths_cfg.get("data_root"),
    ]
    allowed_roots = [
        _resolve_config_path(root, repo_root=repo_root)
        for root in allowed_roots_raw
        if root not in (None, "")
    ]
    write_boundary = _resolve_config_path(
        safety_cfg.get("write_boundary") or paths_cfg.get("home_root") or os.environ.get("HOME") or "/home/linbinhao",
        repo_root=repo_root,
    )
    specs = external_model_specs(repo_root=repo_root, local_config_path=local_config_path)

    models: list[dict[str, Any]] = []
    verified_handle_paths: list[str] = []
    issues: list[dict[str, str]] = []

    for spec in specs:
        name = spec["name"]
        handle_path = Path("model") / name
        handle = link_root / name
        expected_target = Path(spec["target"]).resolve(strict=False)
        actual_target = _resolve_existing_link_target(handle)
        target_for_checks = actual_target or expected_target
        item_issues: list[dict[str, str]] = []

        handle_exists = handle.exists() or handle.is_symlink()
        if not handle_exists:
            item_issues.append(_issue("error", "external_model_handle_missing", f"Missing external model handle: {handle}"))
        if handle_exists and not handle.is_symlink():
            item_issues.append(_issue("error", "external_model_handle_not_symlink", f"External model handle is not a symlink: {handle}"))
        target_matches_config = actual_target == expected_target if actual_target is not None else False
        if handle.is_symlink() and not target_matches_config:
            item_issues.append(
                _issue(
                    "error",
                    "external_model_target_mismatch",
                    f"{handle_path} points to {actual_target}, expected {expected_target}",
                )
            )

        target_exists = target_for_checks.exists()
        if require_existing and not target_exists:
            item_issues.append(_issue("error", "external_model_target_missing", f"Missing external model target: {target_for_checks}"))

        target_under_allowed_root = bool(allowed_roots) and any(is_under(target_for_checks, root) for root in allowed_roots)
        if not target_under_allowed_root:
            item_issues.append(
                _issue(
                    "error",
                    "external_model_target_outside_allowed_roots",
                    f"{handle_path} target {target_for_checks} is outside configured allowed roots",
                )
            )
        target_under_write_boundary = is_under(target_for_checks, write_boundary)
        if not target_under_write_boundary:
            item_issues.append(
                _issue(
                    "error",
                    "external_model_target_outside_write_boundary",
                    f"{handle_path} target {target_for_checks} is outside write boundary {write_boundary}",
                )
            )

        target_is_git_checkout = bool(target_exists and _is_git_checkout(target_for_checks))
        if require_existing and target_exists and not target_is_git_checkout:
            item_issues.append(
                _issue(
                    "warning",
                    "external_model_target_not_git_checkout",
                    f"{handle_path} target exists but is not a git checkout: {target_for_checks}",
                )
            )

        error_count = sum(1 for issue in item_issues if issue["level"] == "error")
        if error_count == 0:
            verified_handle_paths.append(str(handle_path))
        issues.extend(item_issues)
        models.append(
            {
                "name": name,
                "url": spec["url"],
                "handle": str(handle),
                "handle_path": str(handle_path),
                "exists": handle_exists,
                "is_symlink": handle.is_symlink(),
                "target": str(actual_target or expected_target),
                "configured_target": str(expected_target),
                "target_exists": target_exists,
                "target_is_git_checkout": target_is_git_checkout,
                "target_matches_config": target_matches_config,
                "target_under_allowed_root": target_under_allowed_root,
                "target_under_write_boundary": target_under_write_boundary,
                "issues": item_issues,
            }
        )

    error_count = sum(1 for issue in issues if issue["level"] == "error")
    warning_count = sum(1 for issue in issues if issue["level"] == "warning")
    return {
        "schema_version": 1,
        "local_config": str(local_config_path),
        "link_root": str(link_root),
        "allowed_target_roots": [str(root) for root in allowed_roots],
        "write_boundary": str(write_boundary),
        "model_count": len(models),
        "passed": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "verified_handle_paths": verified_handle_paths,
        "models": models,
    }
