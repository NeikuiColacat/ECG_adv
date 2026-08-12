"""Portable, fail-closed contracts for PN2021 evidence artifacts.

This module deliberately uses only the Python standard library.  Artifact
inspection and matrix reduction therefore never import Torch or construct a
model merely to validate provenance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


CLASS_ORDER = ["CD", "HYP", "MI", "NORM", "STTC"]
LOGICAL_CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"
CENTER_SOURCES = {
    "ningbo": ["ningbo"],
    "chapman_shaoxing": ["chapman_shaoxing"],
    "cpsc_2018": ["cpsc_2018", "cpsc_2018_extra"],
    "georgia": ["georgia"],
}
MODEL_SPECS = {
    "efficientnet1dv2": {
        "name": "efficientnet1dv2", "input_shape": [12, 1000],
        "input_layout": "channel_time", "sampling_rate_hz": 100,
        "num_classes": 5, "class_order": CLASS_ORDER, "output_type": "raw_logits",
    },
    "ecgfounder": {
        "name": "ecgfounder", "input_shape": [12, 5000],
        "input_layout": "channel_time", "sampling_rate_hz": 500,
        "num_classes": 5, "class_order": CLASS_ORDER, "output_type": "raw_logits",
    },
}
_OPERATORS = (
    "powerline_noise", "emg_noise", "baseline_wander", "baseline_shift",
    "random_leads_masking",
)
CANONICAL_CORRUPTION_VIEWS = tuple(
    (index, depth, f"d{depth}__{'__'.join(operators)}", operators)
    for index, (depth, operators) in enumerate(
        ((depth, item) for depth in (2, 3) for item in itertools.combinations(_OPERATORS, depth))
    )
)
_LINEAGE_KEYS = frozenset("schema_version scope model center method comparison seed source_checkpoint training_config_sha256 adaptation_data selection".split())
_METHOD_KEYS = frozenset("recipe_id scientific_arm recipe_spec_sha256 implementation_identity recipe_version kind auxiliary_variant schema_version".split())
_ADAPTATION_KEYS = frozenset("dataset partition logical_center source_centers record_count split_id hash_id_set_sha256 split_manifest_sha256 source_manifest_sha256 mapping_version mapping_hash class_order".split())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, keys: set[str] | frozenset[str] | None, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (keys is not None and set(value) != keys):
        raise ValueError(f"{name} keys are incomplete or unexpected")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_json_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    """Load one JSON object with a caller-owned artifact description."""

    resolved = Path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} not found: {resolved}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {resolved}") from exc
    return dict(_mapping(value, None, name))


def build_artifact_reference(path: str | Path) -> dict[str, str]:
    """Build the canonical path/SHA reference without rewriting its path text."""

    return {"path": str(path), "sha256": sha256_file(path)}


def resolve_artifact_reference(value: Any, *, owner: Path, name: str) -> tuple[Path, str]:
    reference = _mapping(value, {"path", "sha256"}, name)
    digest = _sha(reference["sha256"], f"{name}.sha256")
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise ValueError(f"{name}.path must be a non-empty string")
    unresolved = Path(reference["path"]).expanduser()
    candidate = unresolved if unresolved.is_absolute() else owner.parent / unresolved
    if candidate.is_symlink():
        raise ValueError(f"{name} cannot be a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    if sha256_file(resolved) != digest:
        raise ValueError(f"{name} SHA256 mismatch: digest does not match the file")
    return resolved, digest


def validate_training_lineage(value: Any) -> dict[str, Any]:
    """Validate and copy one portable prospective K500 lineage."""

    lineage = _mapping(value, _LINEAGE_KEYS, "training lineage")
    model = _mapping(lineage["model"], {"name", "spec"}, "lineage.model")
    method = _mapping(lineage["method"], _METHOD_KEYS, "lineage.method")
    comparison = _mapping(lineage["comparison"], {"group", "replicate_id"}, "lineage.comparison")
    seed = _mapping(lineage["seed"], {"base_seed", "effective_seed", "namespace", "config_sha256"}, "lineage.seed")
    source = _mapping(lineage["source_checkpoint"], {"sha256"}, "lineage.source_checkpoint")
    adaptation = _mapping(lineage["adaptation_data"], _ADAPTATION_KEYS, "lineage.adaptation_data")
    selection = _mapping(lineage["selection"], {"policy", "heldout_evaluation_used_for_selection"}, "lineage.selection")
    model_name, center = model["name"], lineage["center"]
    if (type(lineage["schema_version"]) is not int or lineage["schema_version"] != 1
            or lineage["scope"] != "pn2021_k500_center_adaptation"
            or not isinstance(model_name, str) or model.get("spec") != MODEL_SPECS.get(model_name)
            or not isinstance(center, str) or center not in CENTER_SOURCES):
        raise ValueError("training lineage scope, model, or center is invalid")
    if any(not isinstance(method[key], str) or not method[key] for key in
           ("recipe_id", "scientific_arm", "implementation_identity", "kind", "auxiliary_variant")):
        raise ValueError("lineage method identity is invalid")
    if any(type(method[key]) is not int or method[key] <= 0 for key in ("recipe_version", "schema_version")):
        raise ValueError("lineage method versions must be positive integers")
    if (not isinstance(comparison["group"], str) or not comparison["group"]
            or type(comparison["replicate_id"]) is not int or comparison["replicate_id"] < 0):
        raise ValueError("lineage comparison identity is invalid")
    if (not isinstance(seed["namespace"], str) or not seed["namespace"] or any(
            type(seed[key]) is not int or not 0 <= seed[key] < 2**32
            for key in ("base_seed", "effective_seed"))):
        raise ValueError("lineage seed identity is invalid")
    if (adaptation["dataset"] != "pn2021" or adaptation["partition"] != "k500"
            or adaptation["logical_center"] != center
            or adaptation["source_centers"] != CENTER_SOURCES[center]
            or type(adaptation["record_count"]) is not int or adaptation["record_count"] != 500
            or not isinstance(adaptation["split_id"], str) or not adaptation["split_id"]):
        raise ValueError("lineage adaptation is not the selected center's PN2021 K500")
    if (adaptation["mapping_version"] != MAPPING_VERSION or adaptation["mapping_hash"] != MAPPING_HASH
            or adaptation["class_order"] != CLASS_ORDER):
        raise ValueError("lineage adaptation label mapping is incorrect")
    if (selection["policy"] != "last"
            or selection["heldout_evaluation_used_for_selection"] is not False):
        raise ValueError("lineage selection must be heldout-free last")
    for digest in (method["recipe_spec_sha256"], seed["config_sha256"], source["sha256"],
                   lineage["training_config_sha256"], adaptation["hash_id_set_sha256"],
                   adaptation["split_manifest_sha256"], adaptation["source_manifest_sha256"]):
        _sha(digest, "lineage digest")
    return deepcopy(dict(lineage))


def validate_checkpoint_root(payload: Any, *, schema_version: int | None = None) -> dict[str, Any] | None:
    """Validate the portable root identity of a loaded checkpoint payload."""

    if not isinstance(payload, Mapping):
        return None
    version = payload.get("schema_version") if schema_version is None else schema_version
    if version != 3:
        if "lineage" in payload:
            raise ValueError("checkpoint lineage requires checkpoint schema_version=3")
        return None
    lineage = validate_training_lineage(payload.get("lineage"))
    method, run = lineage["method"], _mapping(payload.get("run_identity"), None, "checkpoint.run_identity")
    run_model = _mapping(run.get("model"), None, "checkpoint.run_identity.model")
    run_recipe = _mapping(run.get("recipe"), None, "checkpoint.run_identity.recipe")
    run_seed = _mapping(run.get("seed"), None, "checkpoint.run_identity.seed")
    run_config = _mapping(run.get("config"), None, "checkpoint.run_identity.config")
    source_key = "checkpoint_identity" if lineage["model"]["name"] == "efficientnet1dv2" else "task_checkpoint_identity"
    run_source = _mapping(run_model.get(source_key), None, f"checkpoint.run_identity.model.{source_key}")
    if (payload.get("center") != lineage["center"] or payload.get("method_id") != method["recipe_id"]
            or payload.get("scientific_arm") != method["scientific_arm"] or payload.get("selection") != "last"
            or run.get("center") != lineage["center"] or run.get("scientific_arm") != method["scientific_arm"]
            or run_model.get("spec") != lineage["model"]["spec"]
            or run_recipe.get("recipe_id") != method["recipe_id"]
            or run_recipe.get("recipe_spec_sha256") != method["recipe_spec_sha256"]
            or any(run_seed.get(key) != item for key, item in lineage["seed"].items())
            or run_config.get("sha256") != lineage["training_config_sha256"]
            or run_source.get("sha256") != lineage["source_checkpoint"]["sha256"]):
        raise ValueError("checkpoint root or run_identity differs from its lineage")
    return lineage


def validate_pn2021_train_result(payload: Any, *, result_path: Path,
                                 verify_checkpoint: bool = True) -> dict[str, Any]:
    """Validate a prospective train result and its selected checkpoint bytes."""

    result = _mapping(payload, None, "train_result")
    lineage = validate_training_lineage(result.get("lineage"))
    model = _mapping(result.get("model"), None, "train_result.model")
    training = _mapping(_mapping(result.get("config"), None, "train_result.config").get("training"), None, "train_result.config.training")
    seed = _mapping(result.get("seed"), None, "train_result.seed")
    selection = _mapping(result.get("selection"), None, "train_result.selection")
    last = _mapping(result.get("last_checkpoint"), {"path", "sha256"}, "train_result.last_checkpoint")
    epochs = _positive(result.get("epochs_completed"), "train_result.epochs_completed")
    steps = _positive(result.get("optimizer_steps"), "train_result.optimizer_steps")
    if any(seed.get(key) != item for key, item in lineage["seed"].items()):
        raise ValueError("train_result seed identity differs from lineage")
    if (selection.get("policy") != "last" or selection.get("selected_epoch") != epochs
            or selection.get("selected_checkpoint") != last
            or selection.get("heldout_evaluation_used_for_selection") is not False):
        raise ValueError("train_result selection identity differs: require heldout-free last checkpoint")
    if (result.get("schema_version") != 1 or result.get("artifact_type") != "pn2021_train_result"
            or result.get("center") != lineage["center"] or result.get("method_id") != lineage["method"]["recipe_id"]
            or result.get("scientific_arm") != lineage["method"]["scientific_arm"]
            or model.get("spec") != lineage["model"]["spec"]
            or training.get("sha256") != lineage["training_config_sha256"]):
        raise ValueError("train_result identity differs from its lineage")
    checkpoint_path, checkpoint_sha = (resolve_artifact_reference(last, owner=result_path, name="train_result.last_checkpoint")
                                       if verify_checkpoint else (Path(str(last["path"])).expanduser().resolve(), _sha(last["sha256"], "checkpoint SHA256")))
    if checkpoint_path.parent.parent != result_path.parent.resolve():
        raise ValueError("train_result checkpoint must belong to the same training output directory")
    return {"lineage": lineage, "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha,
            "epochs_completed": epochs, "optimizer_steps": steps}


def _complete_centers(value: Any, centers: Sequence[str], name: str) -> Mapping[str, Any]:
    items = _mapping(value, None, name)
    if list(items) != list(centers) or any(not isinstance(item, Mapping)
            or not isinstance(item.get("metrics"), Mapping) or not item["metrics"]
            or not isinstance(item.get("identity"), Mapping) or not item["identity"] for item in items.values()):
        raise ValueError(f"{name} does not contain the expected centers")
    return items


def _view_signatures(views: Any, centers: Sequence[str]) -> list[tuple[Any, ...]]:
    if not isinstance(views, list) or len(views) != 20:
        raise ValueError("evaluation must contain exactly 20 corruption views")
    signatures = []
    for index, view in enumerate(views):
        item = _mapping(view, None, f"corrupted view {index}")
        _complete_centers(item.get("per_center"), centers, f"corrupted view {index}.per_center")
        operators = item.get("operators")
        if not isinstance(operators, list) or not all(isinstance(op, str) and op for op in operators):
            raise ValueError("corruption view operators must be non-empty strings")
        signatures.append((item.get("view_index"), item.get("depth"), item.get("composition_id"), tuple(operators)))
    if tuple(signatures) != CANONICAL_CORRUPTION_VIEWS:
        raise ValueError("corruption views must be the canonical depth2/depth3 set")
    return signatures


def _validate_center_evidence(entry: Mapping[str, Any], *, corrupted: bool) -> None:
    identity = _mapping(entry.get("identity"), None, "evaluation identity")
    metrics = _mapping(entry.get("metrics"), None, "evaluation metrics")
    aliases = (
        ("pn2021c_all_zero_kept_corrupted_refexcluded",
         "pn2021c_drop_all_zero_corrupted_refexcluded")
        if corrupted else
        ("pn2021_all_zero_kept_refexcluded", "pn2021_drop_all_zero_refexcluded")
    )
    if tuple(metrics) != aliases:
        raise ValueError("evaluation metric aliases are not canonical")
    reference = _mapping(identity.get("reference_exclusion"), None, "reference_exclusion")
    kept, dropped = (metrics[alias] for alias in aliases)
    for metric, alias, policy in zip((kept, dropped), aliases, ("kept", "drop"), strict=True):
        record = _mapping(metric.get("record_identity"), None, f"{alias}.record_identity")
        reference_key = (
            "all_zero_kept_hash_id_set_sha256"
            if policy == "kept" else "drop_all_zero_hash_id_set_sha256"
        )
        if (metric.get("canonical_view") != alias or metric.get("all_zero_policy") != policy
                or metric.get("excluded_reference_count") != 500
                or metric.get("class_order") != CLASS_ORDER
                or metric.get("evaluation_slice") not in ({"clean"} if not corrupted else {"depth2", "depth3"})
                or record.get("hash_id_set_sha256") != reference.get(reference_key)):
            raise ValueError("evaluation metric identity differs from its reference exclusion")
    kept_record = kept["record_identity"]
    if (identity.get("hash_id_set_sha256") != kept_record.get("hash_id_set_sha256")
            or identity.get("hash_label_set_sha256") != kept_record.get("hash_label_set_sha256")
            or identity.get("ordered_hash_ids_sha256") != kept_record.get("ordered_hash_ids_sha256")
            or identity.get("evaluated_sample_count") != kept.get("sample_count")):
        raise ValueError("evaluation top-level identity differs from kept metric records")


def _prospective_cohort(result: Mapping[str, Any], lineage: Mapping[str, Any],
                        train: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(result.get("config"), None, "evaluation config")
    resolved = _mapping(config.get("resolved"), None, "evaluation config.resolved")
    seed, adaptation = lineage["seed"], lineage["adaptation_data"]
    return {
        "model_name": lineage["model"]["name"], "method": lineage["method"],
        "comparison": lineage["comparison"], "source_checkpoint": lineage["source_checkpoint"],
        "selection": lineage["selection"],
        "seed": {key: seed[key] for key in ("base_seed", "namespace", "config_sha256")},
        "adaptation": {key: adaptation[key] for key in ("dataset", "partition", "record_count", "split_id",
            "source_manifest_sha256", "mapping_version", "mapping_hash", "class_order")},
        "training": {"config_sha256": lineage["training_config_sha256"],
                     "epochs_completed": train["epochs_completed"], "optimizer_steps": train["optimizer_steps"]},
        "evaluation": {"config_sha256": config.get("sha256"), "artifact_locks": resolved.get("artifact_locks")},
    }


def validate_expected_cohort(cohort: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Match immutable, config-owned cohort fields; epochs/steps stay equality gates."""

    keys = {"model_name", "method", "comparison", "source_checkpoint", "selection", "seed", "adaptation",
            "center_adaptation", "training_config_sha256", "evaluation_config_sha256", "artifact_locks"}
    locked = _mapping(expected, keys, "expected_cohort")
    observed = {key: cohort[key] for key in ("model_name", "method", "comparison", "source_checkpoint", "selection", "seed", "adaptation")}
    observed.update(training_config_sha256=cohort["training"]["config_sha256"],
                    evaluation_config_sha256=cohort["evaluation"]["config_sha256"],
                    artifact_locks=cohort["evaluation"]["artifact_locks"])
    if observed != {key: locked[key] for key in observed}:
        raise ValueError("artifact cohort differs from the config-owned expected cohort")


def validate_pn2021_evaluation_result(
    payload: Any, *, result_path: Path, expected_cohort: Mapping[str, Any] | None = None,
    prospective_only: bool = False, verify_checkpoint: bool = True,
    registry_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Validate one complete schema-3 evaluation and all referenced bytes."""

    result = _mapping(payload, None, "evaluation_result")
    if (result.get("schema_version"), result.get("artifact_type"), result.get("status")) != (3, "pn2021_evaluation_result", "complete"):
        raise ValueError("evaluation_result must be complete schema_version=3")
    subject = _mapping(result.get("subject"), {"mode", "train_result", "lineage"}, "evaluation subject")
    mode, lineage = subject["mode"], _mapping(subject["lineage"], None, "evaluation lineage")
    if mode not in {"prospective_train_result", "legacy_center_adapted", "source_registry"} or (prospective_only and mode != "prospective_train_result"):
        raise ValueError("evaluation subject mode is not allowed")
    protocol = _mapping(result.get("protocol"), None, "evaluation protocol")
    centers = protocol.get("logical_centers")
    if (not isinstance(centers, list) or not centers or len(set(centers)) != len(centers)
            or any(center not in LOGICAL_CENTERS for center in centers)
            or protocol.get("mapping_version") != MAPPING_VERSION or protocol.get("mapping_hash") != MAPPING_HASH
            or protocol.get("class_order") != CLASS_ORDER or protocol.get("reference_count_per_center") != 500):
        raise ValueError("evaluation protocol is not the locked PN2021 contract")
    expected_aliases = [
        "pn2021_all_zero_kept_refexcluded",
        "pn2021_drop_all_zero_refexcluded",
        "pn2021c_all_zero_kept_corrupted_refexcluded",
        "pn2021c_drop_all_zero_corrupted_refexcluded",
    ]
    if protocol.get("canonical_views") != expected_aliases:
        raise ValueError("evaluation canonical metric views are not locked")
    checkpoint = _mapping(result.get("checkpoint"), None, "evaluation checkpoint")
    checkpoint_ref = {"path": checkpoint.get("path"), "sha256": checkpoint.get("sha256")}
    checkpoint_path, checkpoint_sha = (resolve_artifact_reference(checkpoint_ref, owner=result_path, name="evaluation checkpoint")
                                       if verify_checkpoint else (Path(str(checkpoint_ref["path"])).expanduser().resolve(), _sha(checkpoint_ref["sha256"], "checkpoint SHA256")))
    train_payload: Mapping[str, Any] | None = None
    train_context: dict[str, Any] | None = None
    cohort = None
    if mode == "prospective_train_result":
        lineage = validate_training_lineage(lineage)
        if centers != [lineage["center"]] or result.get("model") != lineage["model"]["spec"]:
            raise ValueError("evaluation model or center differs from prospective lineage")
        train_path, _ = resolve_artifact_reference(subject["train_result"], owner=result_path, name="evaluation train_result")
        train_payload = load_json_mapping(train_path, name="train_result")
        train_context = validate_pn2021_train_result(train_payload, result_path=train_path, verify_checkpoint=verify_checkpoint)
        if train_context["lineage"] != lineage or train_context["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("evaluation differs from its prospective train_result")
        if checkpoint.get("checkpoint_schema_version") != 3 or checkpoint.get("lineage") != lineage:
            raise ValueError("prospective evaluation checkpoint must carry matching schema-3 lineage")
        cohort = _prospective_cohort(result, lineage, train_context)
        if expected_cohort is not None:
            validate_expected_cohort(cohort, expected_cohort)
            center_locks = _mapping(expected_cohort["center_adaptation"], set(LOGICAL_CENTERS), "expected center_adaptation")
            expected_center = _mapping(center_locks[lineage["center"]],
                                       {"source_centers", "hash_id_set_sha256", "split_manifest_sha256"},
                                       "expected center adaptation")
            adaptation = lineage["adaptation_data"]
            if any(adaptation[key] != expected_center[key] for key in expected_center):
                raise ValueError("artifact uses the wrong center-specific K500 evidence")
    elif mode == "legacy_center_adapted":
        if set(lineage) != {"scope", "model", "center", "method", "adaptation_data",
                            "selection", "checkpoint"}:
            raise ValueError("legacy lineage keys are incomplete or unexpected")
        legacy_model = _mapping(lineage.get("model"), {"name", "spec"}, "legacy lineage.model")
        method = _mapping(lineage.get("method"), {"recipe_id", "scientific_arm"}, "legacy lineage.method")
        legacy_adaptation = _mapping(
            lineage.get("adaptation_data"),
            {"partition", "mapping_version", "mapping_hash"},
            "legacy lineage.adaptation_data",
        )
        allowed = {"a0_clean_v1": "A0", "direct_depth23_fixed20": "DirectDepth23Fixed20FamilyBalanced"}
        selection_identity = _mapping(lineage.get("selection"), None, "legacy lineage.selection")
        checkpoint_identity = _mapping(lineage.get("checkpoint"), {"sha256"}, "legacy lineage.checkpoint")
        legacy_checkpoint = _mapping(
            checkpoint.get("legacy_training_identity"),
            {"model", "center", "method_id", "scientific_arm", "selection"},
            "legacy checkpoint identity",
        )
        expected_checkpoint = {
            "model": legacy_model["name"], "center": lineage.get("center"),
            "method_id": method["recipe_id"], "scientific_arm": method["scientific_arm"],
            "selection": "last",
        }
        if (len(centers) != 1 or lineage.get("scope") != "legacy_pn2021_k500_center_adaptation"
                or lineage.get("center") != centers[0] or checkpoint.get("checkpoint_schema_version") != 2
                or legacy_model.get("spec") != MODEL_SPECS.get(legacy_model.get("name"))
                or result.get("model") != legacy_model["spec"]
                or allowed.get(method.get("recipe_id")) != method.get("scientific_arm")
                or legacy_adaptation != {"partition": "k500", "mapping_version": MAPPING_VERSION,
                                         "mapping_hash": MAPPING_HASH}
                or selection_identity.get("policy") != "last"
                or selection_identity.get("heldout_evaluation_used_for_selection") is not False
                or checkpoint_identity.get("sha256") != checkpoint_sha
                or dict(legacy_checkpoint) != expected_checkpoint):
            raise ValueError("legacy evaluation checkpoint identity is invalid")
        train_path, _ = resolve_artifact_reference(subject["train_result"], owner=result_path, name="legacy train_result")
        train_payload = load_json_mapping(train_path, name="legacy train_result")
        if any(key in train_payload for key in ("schema_version", "artifact_type", "lineage")):
            raise ValueError("schema-1 train_result cannot use legacy evaluation mode")
        last = _mapping(train_payload.get("last_checkpoint"), {"path", "sha256"}, "legacy last_checkpoint")
        last_path, last_sha = resolve_artifact_reference(last, owner=train_path, name="legacy last_checkpoint")
        selection = _mapping(train_payload.get("selection"), None, "legacy selection")
        config = _mapping(train_payload.get("config"), None, "legacy config")
        method_config = _mapping(config.get("method"), None, "legacy config.method")
        training = _mapping(config.get("training"), None, "legacy config.training")
        resolved = _mapping(training.get("resolved"), None, "legacy training config")
        legacy_protocol = _mapping(resolved.get("protocol"), None, "legacy protocol")
        locked_protocol = {"partition": "k500", "use_all_k500": True, "validation_split": False,
            "checkpoint_selection": "last", "heldout_ref_exclusion_required": True,
            "merge_cpsc_2018_extra_into_cpsc_2018": True, "centers": list(LOGICAL_CENTERS),
            "class_order": CLASS_ORDER, "mapping_version": MAPPING_VERSION, "mapping_hash": MAPPING_HASH}
        if (last_sha != checkpoint_sha or selection.get("policy") != "last"
                or selection.get("selected_checkpoint") != last or selection.get("heldout_evaluation_used_for_selection") is not False
                or last_path != checkpoint_path
                or train_payload.get("center") != centers[0] or train_payload.get("method_id") != method.get("recipe_id")
                or train_payload.get("scientific_arm") != method.get("scientific_arm")
                or method_config.get("profile_name") != method.get("recipe_id")
                or method_config.get("scientific_arm") != method.get("scientific_arm")
                or _mapping(train_payload.get("model"), None, "legacy model").get("spec") != result.get("model")
                or any(legacy_protocol.get(key) != item for key, item in locked_protocol.items())):
            raise ValueError("legacy evaluation differs from its train_result")
    else:
        if (set(lineage) != {"scope", "model", "source_checkpoint", "registry", "selection"}
                or subject["train_result"] is not None or centers != list(LOGICAL_CENTERS)
                or checkpoint.get("checkpoint_schema_version") != 1
                or checkpoint.get("lineage") is not None
                or checkpoint.get("legacy_training_identity") is not None):
            raise ValueError("source evaluation subject identity is invalid")
        source = _mapping(lineage.get("source_checkpoint"), {"sha256"}, "source checkpoint")
        if source["sha256"] != checkpoint_sha:
            raise ValueError("source evaluation checkpoint differs from lineage")
        registry_path, _ = resolve_artifact_reference(lineage.get("registry"), owner=result_path, name="source registry")
        if registry_loader is None:
            raise ValueError("source registry evaluation requires a registry loader")
        registry = _mapping(registry_loader(registry_path), None, "source registry")
        baseline = _mapping(registry.get("baseline"), None, "source registry.baseline")
        policy = _mapping(registry.get("downstream_policy"), None, "source registry.downstream_policy")
        models = _mapping(registry.get("models"), None, "source registry.models")
        source_model = _mapping(lineage.get("model"), {"name", "spec"}, "source model")
        entry = _mapping(models.get(source_model["name"]), None, "source registry model")
        selection_identity = _mapping(lineage.get("selection"), None, "source selection")
        selection_policy = selection_identity.get("policy")
        registered_path = entry.get("selected_checkpoint")
        if not isinstance(registered_path, str) or not registered_path:
            raise ValueError("source registry selected checkpoint path is invalid")
        if (lineage.get("scope") != "ptbxl_source_global"
                or registry.get("schema_version") != 1
                or source_model.get("spec") != MODEL_SPECS.get(source_model["name"])
                or result.get("model") != source_model["spec"] or baseline.get("class_order") != CLASS_ORDER
                or policy.get("checkpoint_field") != "selected_checkpoint"
                or policy.get("require_sha256_match") is not True or policy.get("prohibit_last_checkpoint") is not True
                or entry.get("selected_checkpoint_sha256") != checkpoint_sha
                or Path(registered_path).expanduser().resolve() != checkpoint_path
                or not isinstance(selection_policy, str) or not selection_policy
                or selection_policy != baseline.get("selection_rule")
                or selection_identity.get("heldout_evaluation_used_for_selection") is not False):
            raise ValueError("source registry differs from evaluation checkpoint")
    clean_centers = _complete_centers(_mapping(result.get("clean"), None, "clean").get("per_center"), centers, "clean.per_center")
    views = _mapping(result.get("corrupted"), None, "corrupted").get("per_view")
    signatures = _view_signatures(views, centers)
    for center in centers:
        _validate_center_evidence(clean_centers[center], corrupted=False)
        clean_identity = _mapping(clean_centers[center].get("identity"), None, "clean identity")
        for view in views:
            corrupted_entry = view["per_center"][center]
            _validate_center_evidence(corrupted_entry, corrupted=True)
            identity = _mapping(corrupted_entry.get("identity"), None, "corrupted identity")
            if any(identity.get(key) != clean_identity.get(key) for key in ("hash_id_set_sha256", "hash_label_set_sha256")):
                raise ValueError("clean and corruption record/label identities differ")
            for candidate in (clean_identity, identity):
                exclusion = _mapping(candidate.get("reference_exclusion"), None, "reference_exclusion")
                if exclusion.get("excluded_reference_count") != 500:
                    raise ValueError("evaluation does not exclude exactly K500")
                if mode == "prospective_train_result" and exclusion.get("k500_hash_id_set_sha256") != lineage["adaptation_data"]["hash_id_set_sha256"]:
                    raise ValueError("evaluation excludes a different K500 identity")
    # Local import avoids a package-level NumPy/sklearn dependency and cycle;
    # the contract module itself remains portable for training/checkpoint use.
    from util.evaluation.metrics import validate_evaluation_aggregates
    validate_evaluation_aggregates(result)
    return {"mode": mode, "lineage": deepcopy(dict(lineage)), "logical_centers": tuple(centers),
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha,
            "train_result": train_payload, "train_context": train_context,
            "signatures": signatures, "cohort": cohort}


__all__ = ["CANONICAL_CORRUPTION_VIEWS", "CLASS_ORDER", "LOGICAL_CENTERS", "MAPPING_HASH", "MAPPING_VERSION",
           "MODEL_SPECS", "build_artifact_reference", "load_json_mapping",
           "resolve_artifact_reference", "sha256_file",
           "validate_checkpoint_root", "validate_expected_cohort",
           "validate_pn2021_evaluation_result", "validate_pn2021_train_result",
           "validate_training_lineage"]
