"""Strict, traceable checkpoint loading shared by the rebuilt model package."""

from __future__ import annotations

import hashlib
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC
from util.pn2021_artifact_contract import (
    validate_checkpoint_root,
    validate_training_lineage,
)


_MODEL_SPECS = {
    spec.name: spec.describe() for spec in (EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC)
}


def _legacy_training_identity(
    payload: Any, schema_version: int | None
) -> dict[str, Any] | None:
    if schema_version != 2 or not isinstance(payload, Mapping):
        return None
    if not {"center", "method_id", "scientific_arm", "selection", "run_identity"}.issubset(payload):
        return None
    center, method_id = payload["center"], payload["method_id"]
    scientific_arm = payload["scientific_arm"]
    run_identity = payload["run_identity"]
    run_model = run_identity.get("model") if isinstance(run_identity, Mapping) else None
    run_spec = run_model.get("spec") if isinstance(run_model, Mapping) else None
    model_name = run_spec.get("name") if isinstance(run_spec, Mapping) else None
    method_resources = run_identity.get("method") if isinstance(run_identity, Mapping) else None
    legacy_method = method_resources.get("method") if isinstance(method_resources, Mapping) else None
    if (
        payload["selection"] != "last"
        or model_name not in _MODEL_SPECS
        or run_spec != _MODEL_SPECS.get(model_name)
        or not isinstance(legacy_method, Mapping)
        or run_identity.get("center") != center
        or run_identity.get("scientific_arm") != scientific_arm
        or method_resources.get("model_name") != model_name
        or legacy_method.get("profile_name") != method_id
        or legacy_method.get("scientific_arm") != scientific_arm
    ):
        raise ValueError("legacy checkpoint root and run_identity disagree")
    return {"model": model_name, "center": center, "method_id": method_id,
            "scientific_arm": scientific_arm, "selection": "last"}


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    sha256: str
    state_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    checkpoint_schema_version: int | None = None
    lineage: dict[str, Any] | None = None
    legacy_training_identity: dict[str, Any] | None = None

    def describe(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": str(self.path),
            "sha256": self.sha256,
            "state_key_count": self.state_key_count,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }
        for name, value in (("checkpoint_schema_version", self.checkpoint_schema_version),
                            ("lineage", self.lineage),
                            ("legacy_training_identity", self.legacy_training_identity)):
            if value is not None:
                result[name] = deepcopy(value)
        return result


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    trusted: bool = False,
) -> tuple[Path, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {resolved}")
    try:
        payload = torch.load(
            resolved,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(resolved, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if not trusted:
            raise ValueError(
                "checkpoint requires unsafe pickle loading; only an explicitly "
                f"trusted source may be loaded: {resolved}"
            ) from exc
        payload = torch.load(
            resolved,
            map_location=map_location,
            weights_only=False,
        )
    return resolved, payload


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract common raw, training, and official checkpoint state layouts."""

    candidate = payload
    if isinstance(payload, Mapping):
        if "model_state_dict" in payload:
            candidate = payload["model_state_dict"]
        elif "state_dict" in payload:
            candidate = payload["state_dict"]
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("checkpoint does not contain a non-empty model state dict")
    state: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state dict must map string keys to tensors")
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key in state:
            raise ValueError(f"checkpoint key collision after prefix removal: {key}")
        state[key] = value
    return state


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointIdentity:
    """Load a task checkpoint and return an auditable identity."""

    resolved, payload = load_checkpoint_payload(path, map_location=map_location)
    raw_schema = payload.get("schema_version") if isinstance(payload, Mapping) else None
    schema_version = raw_schema if type(raw_schema) is int else None
    lineage = validate_checkpoint_root(payload, schema_version=schema_version)
    legacy_identity = _legacy_training_identity(payload, schema_version)
    state = extract_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=bool(strict))
    return CheckpointIdentity(
        path=resolved,
        sha256=sha256_file(resolved),
        state_key_count=len(state),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        checkpoint_schema_version=schema_version,
        lineage=lineage,
        legacy_training_identity=legacy_identity,
    )


__all__ = [
    "CheckpointIdentity",
    "extract_state_dict",
    "load_checkpoint_payload",
    "load_model_checkpoint",
    "sha256_file",
    "validate_training_lineage",
]
