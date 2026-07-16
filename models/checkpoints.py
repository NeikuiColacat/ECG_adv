"""Strict, traceable checkpoint loading shared by the rebuilt model package."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    sha256: str
    state_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def describe(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "state_key_count": self.state_key_count,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


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
    state = extract_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=bool(strict))
    return CheckpointIdentity(
        path=resolved,
        sha256=sha256_file(resolved),
        state_key_count=len(state),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


__all__ = [
    "CheckpointIdentity",
    "extract_state_dict",
    "load_checkpoint_payload",
    "load_model_checkpoint",
    "sha256_file",
]
