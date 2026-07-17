"""Strict shared loader for the tracked five-operator ECG profile.

Both offline NumPy cache construction and online Torch AugMix consume this
module.  It resolves the profile's seed YAML inside the same portable config
bundle and records hashes for both files without mutating any RNG state.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from util.config_bundle import resolve_config_reference, resolve_entry_config_path
from util.random_seed import RandomSeedConfig, load_random_seed_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATOR_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "augmentation" / "operators.yaml"
)
CANONICAL_OPERATOR_ORDER = (
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class AugmentationProfile:
    """One fully resolved and identity-tracked operator severity profile."""

    config_path: Path
    config_sha256: str
    schema_version: int
    profile_name: str
    severity: int
    canonical_order: tuple[str, ...]
    random_seed_config: RandomSeedConfig
    operator_parameters: Mapping[str, Mapping[str, Any]]
    _document: dict[str, Any] = field(repr=False, compare=False)

    def parameters_for(self, operator_name: str) -> dict[str, Any]:
        """Return a mutable copy of one operator's validated parameters."""

        name = str(operator_name)
        try:
            parameters = self.operator_parameters[name]
        except KeyError as exc:
            raise ValueError(
                f"operator {name!r} is not in profile {self.profile_name!r}"
            ) from exc
        return copy.deepcopy(dict(parameters))

    def snapshot(self) -> dict[str, Any]:
        """Return an independent YAML-serializable copy of the source document."""

        return copy.deepcopy(self._document)

    def describe(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "severity": self.severity,
            "canonical_order": list(self.canonical_order),
            "random_seed_config_path": str(self.random_seed_config.path),
            "random_seed_sha256": self.random_seed_config.sha256,
            "random_seed": self.random_seed_config.base_seed,
        }


def load_augmentation_profile(
    path: str | Path = DEFAULT_OPERATOR_CONFIG_PATH,
    *,
    profile_name: str | None = None,
    severity: int | None = None,
    expected_canonical_order: tuple[str, ...] = CANONICAL_OPERATOR_ORDER,
    expected_seed_config_path: str | Path | None = None,
    config_root: str | Path | None = None,
) -> AugmentationProfile:
    """Load and strictly validate the shared five-operator profile contract."""

    config_path = resolve_entry_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"augmentation operator config not found: {config_path}"
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document = _mapping(payload, "augmentation operator config")
    if set(document) != {"schema_version", "metadata", "profiles"}:
        raise ValueError(
            "augmentation operator config must contain exactly schema_version, "
            "metadata and profiles"
        )
    if document.get("schema_version") != 1:
        raise ValueError("augmentation operator schema_version must be 1")

    metadata = _mapping(document.get("metadata"), "operator config.metadata")
    profiles = _mapping(document.get("profiles"), "operator config.profiles")
    declared_profile = metadata.get("profile_name")
    if not isinstance(declared_profile, str) or not declared_profile:
        raise ValueError("operator profile name must be a non-empty string")
    requested_profile = declared_profile if profile_name is None else str(profile_name)
    if requested_profile != declared_profile:
        raise ValueError(
            "requested operator profile does not match metadata.profile_name: "
            f"{requested_profile!r} != {declared_profile!r}"
        )
    if set(profiles) != {declared_profile}:
        raise ValueError(
            "operator profiles must contain exactly metadata.profile_name"
        )

    declared_severity = _positive_integer(
        metadata.get("public_severity"), "operator public severity"
    )
    requested_severity = (
        declared_severity
        if severity is None
        else _positive_integer(severity, "requested operator severity")
    )
    if requested_severity != declared_severity:
        raise ValueError(
            "requested operator severity does not match metadata.public_severity: "
            f"{requested_severity} != {declared_severity}"
        )

    composite = _mapping(metadata.get("composite"), "operator config.composite")
    canonical = tuple(str(value) for value in composite.get("canonical_order", ()))
    expected = tuple(str(value) for value in expected_canonical_order)
    if canonical != expected or canonical != CANONICAL_OPERATOR_ORDER:
        raise ValueError(
            "operator canonical order must be "
            f"{list(CANONICAL_OPERATOR_ORDER)}, got {list(canonical)}"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError("operator canonical order contains duplicates")
    depths = tuple(int(value) for value in composite.get("depths", ()))
    if depths != (2, 3):
        raise ValueError("operator composite depths must be [2, 3]")
    if _positive_integer(composite.get("n_depth2"), "composite.n_depth2") != math.comb(
        len(canonical), 2
    ):
        raise ValueError("composite.n_depth2 does not match canonical combinations")
    if _positive_integer(composite.get("n_depth3"), "composite.n_depth3") != math.comb(
        len(canonical), 3
    ):
        raise ValueError("composite.n_depth3 does not match canonical combinations")

    seed_path = resolve_config_reference(
        metadata.get("random_seed_file"),
        owner_config_path=config_path,
        config_root=config_root,
        description="operator profile random seed config",
        must_exist=True,
    )
    if expected_seed_config_path is not None:
        expected_seed_path = resolve_entry_config_path(expected_seed_config_path)
        if seed_path != expected_seed_path:
            raise ValueError(
                "operator profile and consumer must reference the same random seed "
                f"config: {seed_path} != {expected_seed_path}"
            )
    seed_config = load_random_seed_config(seed_path)

    selected_profile = _mapping(
        profiles.get(declared_profile), f"operator profile {declared_profile}"
    )
    if set(selected_profile) != set(canonical):
        raise ValueError(
            "operator profile keys must exactly match the canonical order"
        )
    resolved_parameters: dict[str, Mapping[str, Any]] = {}
    for operator_name in canonical:
        severities = _mapping(
            selected_profile.get(operator_name),
            f"operator profile {declared_profile}.{operator_name}",
        )
        raw_parameters = severities.get(requested_severity)
        if raw_parameters is None:
            raw_parameters = severities.get(str(requested_severity))
        parameters = _mapping(
            raw_parameters,
            f"operator profile {declared_profile}.{operator_name}.{requested_severity}",
        )
        resolved_parameters[operator_name] = MappingProxyType(
            copy.deepcopy(parameters)
        )

    return AugmentationProfile(
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        schema_version=1,
        profile_name=declared_profile,
        severity=requested_severity,
        canonical_order=canonical,
        random_seed_config=seed_config,
        operator_parameters=MappingProxyType(resolved_parameters),
        _document=copy.deepcopy(document),
    )


__all__ = [
    "AugmentationProfile",
    "CANONICAL_OPERATOR_ORDER",
    "DEFAULT_OPERATOR_CONFIG_PATH",
    "load_augmentation_profile",
]
