"""Fail-closed F-004 producer/evaluation identity helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ecg_adv_gen.matched_effnet import F004_RHO_SWEEP_PROTOCOL, f004_identity


class ComparisonIdentityError(ValueError):
    """Raised when an F-004 artifact loses or changes producer identity."""


def validate_comparison_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonIdentityError("comparison_identity must be a JSON object")
    identity = dict(value)
    if identity.get("comparison_protocol") != F004_RHO_SWEEP_PROTOCOL:
        raise ComparisonIdentityError(
            f"unsupported comparison protocol: {identity.get('comparison_protocol')!r}"
        )
    try:
        expected = f004_identity(identity.get("rho"))
    except (TypeError, ValueError) as exc:
        raise ComparisonIdentityError(str(exc)) from exc
    if identity != expected:
        raise ComparisonIdentityError(
            f"comparison_identity mismatch: observed={identity!r}, expected={expected!r}"
        )
    return expected


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonIdentityError(f"invalid comparison identity JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonIdentityError(f"comparison identity JSON root must be an object: {path}")
    return payload


def parse_expected_comparison_identity(raw: str) -> dict[str, Any] | None:
    if not str(raw or "").strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ComparisonIdentityError(f"invalid --comparison_identity_json: {exc}") from exc
    return validate_comparison_identity(payload)


def resolve_producer_comparison_identity(
    model_dir: str | Path, expected_json: str = ""
) -> dict[str, Any] | None:
    expected = parse_expected_comparison_identity(expected_json)
    run_config_path = Path(model_dir) / "run_config.json"
    if not run_config_path.is_file():
        if expected is not None:
            raise ComparisonIdentityError(
                f"F-004 producer is missing run_config.json: {run_config_path}"
            )
        return None
    run_config = _load_json_object(run_config_path)
    contract = run_config.get("comparison_contract")
    producer_raw = contract.get("comparison_identity") if isinstance(contract, Mapping) else None
    if producer_raw is None:
        if expected is not None:
            raise ComparisonIdentityError(
                f"F-004 producer run_config.json is missing comparison_identity: {run_config_path}"
            )
        return None
    producer = validate_comparison_identity(producer_raw)
    if expected is not None and producer != expected:
        raise ComparisonIdentityError(
            f"comparison_identity mismatch between CLI and producer: {expected!r} != {producer!r}"
        )
    return producer


def require_matching_comparison_identity(
    expected: dict[str, Any], observed: Any, *, source: str
) -> dict[str, Any]:
    actual = validate_comparison_identity(observed)
    if actual != expected:
        raise ComparisonIdentityError(
            f"comparison_identity mismatch for {source}: {actual!r} != {expected!r}"
        )
    return actual


def flatten_comparison_identity(identity: Any) -> dict[str, Any]:
    if identity is None:
        return {
            "comparison_protocol": "", "comparison_variant": "", "comparison_rho": "",
            "comparison_topology_version": "", "comparison_topology_sha256": "",
            "comparison_identity_json": "",
        }
    valid = validate_comparison_identity(identity)
    return {
        "comparison_protocol": valid["comparison_protocol"],
        "comparison_variant": valid["variant"],
        "comparison_rho": valid["rho"],
        "comparison_topology_version": valid["topology_version"],
        "comparison_topology_sha256": valid["topology_sha256"],
        "comparison_identity_json": json.dumps(
            valid, sort_keys=True, separators=(",", ":")
        ),
    }
