"""Resolve internal YAML references inside a portable copied config bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "configs"
CONFIG_SECTION_NAMES = {
    "augmentation",
    "data",
    "defaults",
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
    elif owner.parent.name in CONFIG_SECTION_NAMES:
        root = owner.parent.parent
    else:
        root = owner.parent
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


__all__ = [
    "CONFIG_SECTION_NAMES",
    "DEFAULT_CONFIG_ROOT",
    "config_bundle_root",
    "resolve_config_reference",
    "resolve_entry_config_path",
]
