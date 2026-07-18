"""Resolve internal YAML references inside a portable copied config bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "configs"
CONFIG_SECTION_NAMES = {
    "augmentation",
    "data",
    "defaults",
    "eval",
    "experiments",
    "local",
    "train",
}


def resolve_entry_config_path(path: str | Path) -> Path:
    """Resolve a user-selected entry YAML; relative paths are repo-relative."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def config_bundle_root(
    owner_config_path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> Path:
    """Return the root of a ``configs``-shaped directory tree.

    For ``<root>/train/x.yaml`` or ``<root>/data/x.yaml`` the inferred root is
    ``<root>``.  Callers may pass ``config_root`` explicitly for non-standard
    layouts.
    """

    owner = resolve_entry_config_path(owner_config_path)
    if config_root is not None:
        root = Path(config_root).expanduser().resolve()
    else:
        # A portable bundle may organize one section more deeply, for example
        # ``<bundle>/train/methods/a5.yaml``.  Locate the nearest ancestor for
        # which the owner's first relative component is a known top-level
        # section instead of assuming every YAML is directly inside it.
        root = owner.parent
        for candidate in owner.parents:
            relative = owner.relative_to(candidate)
            if len(relative.parts) >= 2 and relative.parts[0] in CONFIG_SECTION_NAMES:
                root = candidate
                break
    try:
        owner.relative_to(root)
    except ValueError:
        raise ValueError(
            f"owner config {owner} is outside declared config root {root}"
        ) from None
    return root


def resolve_config_reference(
    raw: Any,
    *,
    owner_config_path: str | Path,
    description: str,
    config_root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve one path relative to the selected config bundle root."""

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{description} must be a non-empty path")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        root = config_bundle_root(owner_config_path, config_root=config_root)
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                f"{description} escapes config bundle {root}: {raw}"
            ) from None
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _yaml_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        if value.get("config_closure") == "snapshot_only":
            return references
        for child in value.values():
            references.extend(_yaml_references(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            references.extend(_yaml_references(child))
    elif isinstance(value, str) and Path(value).suffix.lower() in {".yaml", ".yml"}:
        references.append(value)
    return references


def resolve_yaml_config_closure(
    entry_paths: Sequence[str | Path],
    *,
    config_root: str | Path,
) -> tuple[Path, ...]:
    """Resolve the transitive YAML closure used by managed run snapshots."""

    root = Path(config_root).expanduser().resolve()
    ordered = [resolve_entry_config_path(path) for path in entry_paths]
    seen: set[Path] = set()
    index = 0
    while index < len(ordered):
        path = ordered[index].resolve()
        index += 1
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"config closure entry not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"config closure YAML must be a mapping: {path}")
        for raw_reference in _yaml_references(payload):
            referenced = resolve_config_reference(
                raw_reference,
                owner_config_path=path,
                config_root=root,
                description=f"YAML reference in {path.name}",
                must_exist=True,
            )
            if referenced not in seen and referenced not in ordered:
                ordered.append(referenced)
    return tuple(path.resolve() for path in ordered)


__all__ = [
    "CONFIG_SECTION_NAMES",
    "DEFAULT_CONFIG_ROOT",
    "config_bundle_root",
    "resolve_config_reference",
    "resolve_entry_config_path",
    "resolve_yaml_config_closure",
]
