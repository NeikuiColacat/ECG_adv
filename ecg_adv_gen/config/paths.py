"""Path resolution and safety checks for shared-server runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PathSafetyError(ValueError):
    """Raised when a configured path would violate shared-server boundaries."""


def _as_path(value: Any, *, field: str) -> Path:
    if value in (None, ""):
        raise PathSafetyError(f"Missing required path field: {field}")
    return Path(str(value)).expanduser().resolve()


def is_under(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
        return True
    except ValueError:
        return False


def reject_untranslated_root_path(raw_value: Any, *, field: str) -> None:
    if not isinstance(raw_value, str):
        return
    expanded = raw_value.strip()
    if expanded.startswith("/root/"):
        raise PathSafetyError(
            f"{field} still points at root-era path {expanded!r}; "
            "move this to current host-local YAML"
        )


def resolve_local_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths_cfg = config.get("paths") or {}
    safety_cfg = config.get("safety") or {}
    python_cfg = config.get("python") or {}

    for key, value in paths_cfg.items():
        reject_untranslated_root_path(value, field=f"paths.{key}")
    reject_untranslated_root_path(python_cfg.get("executable"), field="python.executable")

    write_boundary = _as_path(
        safety_cfg.get("write_boundary") or paths_cfg.get("home_root") or "/home/linbinhao",
        field="safety.write_boundary",
    )
    configured_project_root = paths_cfg.get("project_root")
    project_root = _as_path(
        config.get("_project_root") or configured_project_root,
        field="paths.project_root",
    )
    if configured_project_root and _as_path(
        configured_project_root,
        field="paths.project_root",
    ) != project_root:
        raise PathSafetyError(
            f"paths.project_root={configured_project_root} must match "
            f"experiment config checkout {project_root}"
        )
    data_root = _as_path(paths_cfg.get("data_root"), field="paths.data_root")
    model_root = _as_path(paths_cfg.get("model_root") or data_root / "models", field="paths.model_root")
    output_root = _as_path(paths_cfg.get("output_root") or data_root / "runs", field="paths.output_root")
    cache_root = _as_path(paths_cfg.get("cache_root") or data_root / "cache", field="paths.cache_root")
    tmp_root = _as_path(paths_cfg.get("tmp_root") or data_root / "tmp", field="paths.tmp_root")
    short_tmp_root = (
        _as_path(paths_cfg.get("short_tmp_root"), field="paths.short_tmp_root")
        if paths_cfg.get("short_tmp_root")
        else tmp_root
    )
    python_exe = _as_path(python_cfg.get("executable"), field="python.executable")

    resolved = {
        "write_boundary": write_boundary,
        "project_root": project_root,
        "data_root": data_root,
        "model_root": model_root,
        "output_root": output_root,
        "cache_root": cache_root,
        "tmp_root": tmp_root,
        "short_tmp_root": short_tmp_root,
        "python_executable": python_exe,
    }
    return resolved


def validate_local_paths(config: dict[str, Any]) -> dict[str, str]:
    resolved = resolve_local_paths(config)
    boundary = resolved["write_boundary"]

    project_root = resolved["project_root"]
    if not (project_root / "AGENTS.md").exists() or not (project_root / "scripts").is_dir():
        raise PathSafetyError(
            f"paths.project_root={project_root} does not look like ECG_adv_Gen"
        )

    python_exe = resolved["python_executable"]
    if not python_exe.exists():
        raise PathSafetyError(f"python.executable does not exist: {python_exe}")

    for field in resolved:
        path = resolved[field]
        if not is_under(path, boundary):
            raise PathSafetyError(f"{field}={path} is outside write boundary {boundary}")

    return {key: str(value) for key, value in resolved.items()}
