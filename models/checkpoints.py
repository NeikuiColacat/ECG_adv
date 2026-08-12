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

from models.contracts import CLASS_ORDER, ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC


_LINEAGE_KEYS = frozenset("""schema_version scope model center method comparison seed
source_checkpoint training_config_sha256 adaptation_data selection""".split())
_METHOD_KEYS = frozenset("""recipe_id scientific_arm recipe_spec_sha256 implementation_identity
recipe_version kind auxiliary_variant schema_version""".split())
_ADAPTATION_KEYS = frozenset("""dataset partition logical_center source_centers record_count split_id
hash_id_set_sha256 split_manifest_sha256 source_manifest_sha256 mapping_version mapping_hash class_order""".split())
_MODEL_SPECS = {
    spec.name: spec.describe() for spec in (EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC)
}
_CENTER_SOURCES = {
    "ningbo": ["ningbo"],
    "chapman_shaoxing": ["chapman_shaoxing"],
    "cpsc_2018": ["cpsc_2018", "cpsc_2018_extra"],
    "georgia": ["georgia"],
}


def _exact_mapping(value: Any, keys: set[str] | frozenset[str], description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{description} keys are incomplete or unexpected")
    return value


def _sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA256 digest")
    return value


def validate_training_lineage(value: Any) -> dict[str, Any]:
    """Validate and copy one portable PN2021 center-adaptation identity."""

    lineage = _exact_mapping(value, _LINEAGE_KEYS, "training lineage")
    model = _exact_mapping(lineage["model"], {"name", "spec"}, "lineage.model")
    method = _exact_mapping(lineage["method"], _METHOD_KEYS, "lineage.method")
    comparison = _exact_mapping(lineage["comparison"], {"group", "replicate_id"}, "lineage.comparison")
    seed = _exact_mapping(lineage["seed"], {"base_seed", "effective_seed", "namespace", "config_sha256"}, "lineage.seed")
    source = _exact_mapping(lineage["source_checkpoint"], {"sha256"}, "lineage.source_checkpoint")
    adaptation = _exact_mapping(lineage["adaptation_data"], _ADAPTATION_KEYS, "lineage.adaptation_data")
    selection = _exact_mapping(lineage["selection"], {"policy", "heldout_evaluation_used_for_selection"}, "lineage.selection")
    model_name, center = model["name"], lineage["center"]
    if (
        type(lineage["schema_version"]) is not int
        or lineage["schema_version"] != 1
        or lineage["scope"] != "pn2021_k500_center_adaptation"
        or not isinstance(model_name, str)
        or model["spec"] != _MODEL_SPECS.get(model_name)
        or not isinstance(center, str)
        or center not in _CENTER_SOURCES
    ):
        raise ValueError("training lineage scope, model, or center is invalid")
    if any(
        not isinstance(method[key], str) or not method[key]
        for key in (
            "recipe_id", "scientific_arm", "implementation_identity", "kind",
            "auxiliary_variant",
        )
    ):
        raise ValueError("lineage method identity is invalid")
    if any(type(method[key]) is not int or method[key] <= 0
           for key in ("recipe_version", "schema_version")):
        raise ValueError("lineage method versions must be positive integers")
    replicate_id = comparison["replicate_id"]
    if (not isinstance(comparison["group"], str) or not comparison["group"]
            or type(replicate_id) is not int or replicate_id < 0):
        raise ValueError("lineage comparison identity is invalid")
    if (not isinstance(seed["namespace"], str) or not seed["namespace"]
            or any(
            type(seed[key]) is not int or not 0 <= seed[key] < 2**32
            for key in ("base_seed", "effective_seed")
        )):
        raise ValueError("lineage seed identity is invalid")
    if (
        adaptation["dataset"] != "pn2021"
        or adaptation["partition"] != "k500"
        or adaptation["logical_center"] != center
        or adaptation["source_centers"] != _CENTER_SOURCES[center]
        or adaptation["record_count"] != 500
        or not isinstance(adaptation["split_id"], str)
        or not adaptation["split_id"]
    ):
        raise ValueError("lineage adaptation is not the selected center's PN2021 K500")
    if (adaptation["mapping_version"] != "v7_super5_sjr_rgq_review_20260528"
            or adaptation["mapping_hash"] != "555ec85d5b51"
            or adaptation["class_order"] != list(CLASS_ORDER)):
        raise ValueError("lineage adaptation label mapping is incorrect")
    if (selection["policy"] != "last"
            or selection["heldout_evaluation_used_for_selection"] is not False):
        raise ValueError("lineage selection must be heldout-free last")
    for digest in (
        method["recipe_spec_sha256"], seed["config_sha256"], source["sha256"],
        lineage["training_config_sha256"], adaptation["hash_id_set_sha256"],
        adaptation["split_manifest_sha256"], adaptation["source_manifest_sha256"],
    ):
        _sha256(digest, "lineage digest")
    return deepcopy(dict(lineage))


def _lineage_from_payload(
    payload: Any, schema_version: int | None
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if schema_version != 3:
        if "lineage" in payload:
            raise ValueError("checkpoint lineage requires checkpoint schema_version=3")
        return None
    lineage = validate_training_lineage(payload.get("lineage"))
    method = lineage["method"]
    if (
        payload.get("center") != lineage["center"]
        or payload.get("method_id") != method["recipe_id"]
        or payload.get("scientific_arm") != method["scientific_arm"]
        or payload.get("selection") != "last"
    ):
        raise ValueError("checkpoint root identity differs from its lineage")
    run_identity = payload.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("schema-3 checkpoint requires run_identity")
    run_model, run_recipe, run_seed, run_config = (
        run_identity.get(key) for key in ("model", "recipe", "seed", "config")
    )
    run_spec = run_model.get("spec") if isinstance(run_model, Mapping) else None
    source_key = {EFFICIENTNET1DV2_SPEC.name: "checkpoint_identity",
                  ECGFOUNDER_SPEC.name: "task_checkpoint_identity"}[lineage["model"]["name"]]
    run_source = run_model.get(source_key) if isinstance(run_model, Mapping) else None
    if (
        run_identity.get("center") != lineage["center"]
        or run_identity.get("scientific_arm") != method["scientific_arm"]
        or run_spec != lineage["model"]["spec"]
        or not isinstance(run_recipe, Mapping)
        or run_recipe.get("recipe_id") != method["recipe_id"]
        or run_recipe.get("recipe_spec_sha256") != method["recipe_spec_sha256"]
        or not isinstance(run_seed, Mapping)
        or any(run_seed.get(key) != value for key, value in lineage["seed"].items())
        or not isinstance(run_config, Mapping)
        or run_config.get("sha256") != lineage["training_config_sha256"]
        or not isinstance(run_source, Mapping)
        or run_source.get("sha256") != lineage["source_checkpoint"]["sha256"]
    ):
        raise ValueError("checkpoint run_identity differs from its lineage")
    return lineage


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
    lineage = _lineage_from_payload(payload, schema_version)
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
