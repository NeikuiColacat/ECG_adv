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
SNAPSHOT_ONLY_CLOSURE_MODES = frozenset(
    {"snapshot_only", "managed_run_snapshots_and_sha256"}
)
DECLARED_REFERENCE_KEY_PATHS = (
    ("metadata", "random_seed_file"),
    ("method", "random_seed_file"),
    ("random_seed", "file"),
    ("corruption", "operators_config"),
    ("corruption", "random_seed_file"),
    ("corruption_chains", "operator_config"),
    ("label_mapping_file",),
)


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
    root = config_bundle_root(owner_config_path, config_root=config_root)
    if candidate.is_absolute():
        raise ValueError(
            f"{description} must be relative to config bundle {root}: {raw}"
        )
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


def _nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _declared_yaml_references(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return only schema-owned YAML references, never YAML-looking prose."""

    if payload.get("config_closure") in SNAPSHOT_ONLY_CLOSURE_MODES:
        return ()
    references: list[str] = []
    declared = payload.get("references")
    if isinstance(declared, dict):
        for value in declared.values():
            raw = value.get("path") if isinstance(value, dict) else value
            if isinstance(raw, str):
                references.append(raw)
    resources = payload.get("resources")
    if isinstance(resources, dict):
        for resource in resources.values():
            if not isinstance(resource, dict):
                continue
            if resource.get("type") == "config_reference":
                raw = resource.get("path")
                if isinstance(raw, str):
                    references.append(raw)
            if resource.get("type") == "isolated_torch_generator":
                raw = resource.get("seed_config")
                if isinstance(raw, str):
                    references.append(raw)
    for keys in DECLARED_REFERENCE_KEY_PATHS:
        raw = _nested_value(payload, keys)
        if isinstance(raw, str):
            references.append(raw)
    return tuple(dict.fromkeys(references))


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
        for raw_reference in _declared_yaml_references(payload):
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
