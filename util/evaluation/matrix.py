"""Fail-closed diagonal aggregation for four prospective PN2021 evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from models.checkpoints import validate_training_lineage
from models.contracts import CLASS_ORDER
from util.evaluation.metrics import aggregate_corruption_views, mean_metric_views
from util.evaluation.pn2021 import (
    LOGICAL_CENTERS,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {path}") from exc
    return dict(_mapping(value, name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(value: Any, owner: Path, name: str) -> tuple[Path, str]:
    reference = _mapping(value, name)
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain only path and sha256")
    path = Path(str(reference["path"])).expanduser()
    candidate = owner.parent / path if not path.is_absolute() else path
    if candidate.is_symlink():
        raise ValueError(f"{name} cannot be a symbolic link")
    path = candidate.resolve()
    digest = _sha256(path)
    if digest != reference["sha256"]:
        raise ValueError(f"{name} SHA256 mismatch")
    return path, digest


def _cohort(result: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict[str, Any]:
    seed = _mapping(lineage.get("seed"), "lineage.seed")
    comparison = _mapping(lineage.get("comparison"), "lineage.comparison")
    config = _mapping(result.get("config"), "evaluation config")
    resolved = _mapping(config.get("resolved"), "evaluation config.resolved")
    protocol = _mapping(result.get("protocol"), "evaluation protocol")
    adaptation = _mapping(lineage.get("adaptation_data"), "lineage.adaptation_data")
    return {
        "model": result["model"],
        "method": lineage["method"],
        "comparison": {key: comparison[key] for key in ("group", "replicate_id")},
        "source_checkpoint": lineage["source_checkpoint"],
        "selection": lineage["selection"],
        "seed": {key: seed[key] for key in ("base_seed", "namespace", "config_sha256")},
        "adaptation": {
            key: adaptation[key]
            for key in (
                "dataset", "partition", "record_count", "split_id",
                "split_manifest_sha256", "source_manifest_sha256",
                "mapping_version", "mapping_hash", "class_order",
            )
        },
        "evaluation": {
            "config_sha256": config["sha256"],
            "artifact_locks": resolved["artifact_locks"],
            "mapping_version": protocol["mapping_version"],
            "mapping_hash": protocol["mapping_hash"],
            "class_order": protocol["class_order"],
        },
    }


def _member(path: Path, expected_center: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _load(path, "evaluation result")
    if (result.get("schema_version"), result.get("artifact_type"), result.get("status")) != (
        3, "pn2021_evaluation_result", "complete"
    ):
        raise ValueError("matrix members must be complete schema-3 PN2021 evaluations")
    subject = _mapping(result.get("subject"), "evaluation subject")
    if subject.get("mode") != "prospective_train_result":
        raise ValueError("matrix accepts only prospective_train_result evaluations")
    lineage = validate_training_lineage(subject.get("lineage"))
    adaptation = _mapping(lineage.get("adaptation_data"), "lineage.adaptation_data")
    protocol = _mapping(result.get("protocol"), "evaluation protocol")
    model = _mapping(lineage.get("model"), "lineage.model")
    if (
        lineage.get("center") != expected_center
        or adaptation.get("logical_center") != expected_center
        or protocol.get("logical_centers") != [expected_center]
        or model.get("spec") != result.get("model")
        or adaptation.get("dataset") != "pn2021"
        or adaptation.get("partition") != "k500"
        or adaptation.get("record_count") != 500
        or protocol.get("mapping_version") != PN2021_MAPPING_VERSION
        or protocol.get("mapping_hash") != PN2021_MAPPING_HASH
        or protocol.get("class_order") != list(CLASS_ORDER)
        or protocol.get("reference_count_per_center") != 500
    ):
        raise ValueError(f"{expected_center} member identity is inconsistent")
    train_path, train_sha = _reference(subject.get("train_result"), path, "train_result")
    train = _load(train_path, "train_result")
    last_checkpoint = _mapping(train.get("last_checkpoint"), "train_result.last_checkpoint")
    checkpoint = _mapping(result.get("checkpoint"), "evaluation checkpoint")
    method = lineage["method"]
    train_model = _mapping(train.get("model"), "train_result.model")
    train_config = _mapping(train.get("config"), "train_result.config")
    training_config = _mapping(train_config.get("training"), "train_result.config.training")
    train_seed = _mapping(train.get("seed"), "train_result.seed")
    selection = _mapping(train.get("selection"), "train_result.selection")
    epochs, steps = train.get("epochs_completed"), train.get("optimizer_steps")
    if (
        train.get("schema_version") != 1
        or train.get("artifact_type") != "pn2021_train_result"
        or train.get("lineage") != lineage
        or checkpoint.get("checkpoint_schema_version") != 3
        or checkpoint.get("lineage") != lineage
        or train.get("center") != expected_center
        or train.get("method_id") != method["recipe_id"]
        or train.get("scientific_arm") != method["scientific_arm"]
        or train_model.get("spec") != lineage["model"]["spec"]
        or training_config.get("sha256") != lineage["training_config_sha256"]
        or any(train_seed.get(key) != value for key, value in lineage["seed"].items())
        or type(epochs) is not int or epochs <= 0
        or type(steps) is not int or steps <= 0
        or selection.get("policy") != "last"
        or selection.get("selected_epoch") != epochs
        or selection.get("selected_checkpoint") != last_checkpoint
        or selection.get("heldout_evaluation_used_for_selection") is not False
        or last_checkpoint.get("sha256") != checkpoint.get("sha256")
    ):
        raise ValueError(f"{expected_center} evaluation differs from its train_result")
    clean = _mapping(result.get("clean"), "clean")
    clean_centers = _mapping(clean.get("per_center"), "clean.per_center")
    if list(clean_centers) != [expected_center]:
        raise ValueError("clean result must contain exactly its adapted center")
    clean_entry = _mapping(clean_centers[expected_center], "clean center")
    clean_identity = _mapping(clean_entry.get("identity"), "clean identity")
    k500 = adaptation.get("hash_id_set_sha256")
    views = _mapping(result.get("corrupted"), "corrupted").get("per_view")
    if not isinstance(views, list) or len(views) != 20:
        raise ValueError("each matrix member must contain exactly 20 corruption views")
    signatures: list[tuple[Any, ...]] = []
    for index, view_value in enumerate(views):
        view = _mapping(view_value, f"corrupted view {index}")
        per_center = _mapping(view.get("per_center"), "view.per_center")
        if list(per_center) != [expected_center]:
            raise ValueError("each corruption view must contain its adapted center only")
        identity = _mapping(_mapping(per_center[expected_center], "view center").get("identity"), "view identity")
        for candidate in (clean_identity, identity):
            exclusion = _mapping(candidate.get("reference_exclusion"), "reference_exclusion")
            if exclusion.get("excluded_reference_count") != 500 or exclusion.get("k500_hash_id_set_sha256") != k500:
                raise ValueError("evaluation does not exclude the adapted K500 identity")
        if any(identity.get(key) != clean_identity.get(key) for key in ("hash_id_set_sha256", "hash_label_set_sha256")):
            raise ValueError("clean and corruption record/label identities differ")
        operators = view.get("operators")
        if not isinstance(operators, list) or not all(
            isinstance(operator, str) and operator for operator in operators
        ):
            raise ValueError("corruption view operators must be a non-empty list")
        signatures.append(
            (view.get("view_index"), view.get("depth"), view.get("composition_id"), tuple(operators))
        )
    if (
        [item[0] for item in signatures] != list(range(20))
        or [item[1] for item in signatures] != [2] * 10 + [3] * 10
        or len({item[2] for item in signatures}) != 20
        or any(not isinstance(item[2], str) or not item[2] for item in signatures)
        or any(len(item[3]) != item[1] for item in signatures)
    ):
        raise ValueError("corruption views must be the canonical depth2/depth3 set")
    summary = {
        "center": expected_center,
        "evaluation_result": {"path": str(path), "sha256": _sha256(path)},
        "train_result": {"path": str(train_path), "sha256": train_sha},
        "checkpoint": {"sha256": checkpoint["sha256"]},
        "adaptation_data": adaptation,
    }
    cohort = _cohort(result, lineage)
    cohort["training"] = {
        "config_sha256": lineage["training_config_sha256"],
        "epochs_completed": epochs,
        "optimizer_steps": steps,
    }
    return result, {"summary": summary, "cohort": cohort, "signatures": signatures}


def aggregate_pn2021_matrix(result_paths: Sequence[str | Path], *, profile_name: str) -> dict[str, Any]:
    """Validate and recompute the canonical four-center diagonal matrix."""

    candidates = [Path(value).expanduser() for value in result_paths]
    if any(path.is_symlink() for path in candidates):
        raise ValueError("matrix evaluation results cannot be symbolic links")
    paths = [path.resolve() for path in candidates]
    if len(paths) != 4:
        raise ValueError("matrix aggregation requires exactly four evaluation results")
    loaded = [_member(path, center) for path, center in zip(paths, LOGICAL_CENTERS, strict=True)]
    results, identities = zip(*loaded, strict=True)
    if any(item["cohort"] != identities[0]["cohort"] for item in identities[1:]):
        raise ValueError("matrix members do not share one locked evaluation cohort")
    if any(item["signatures"] != identities[0]["signatures"] for item in identities[1:]):
        raise ValueError("matrix members use different corruption views")
    clean_per_center = {center: result["clean"]["per_center"][center] for center, result in zip(LOGICAL_CENTERS, results, strict=True)}
    clean_mean = mean_metric_views([clean_per_center[center]["metrics"] for center in LOGICAL_CENTERS])
    per_view: list[dict[str, Any]] = []
    for index, signature in enumerate(identities[0]["signatures"]):
        view = {key: results[0]["corrupted"]["per_view"][index][key] for key in ("view_index", "depth", "operators", "composition_id")}
        view["evaluation_slice"] = f"depth{signature[1]}"
        view["per_center"] = {center: result["corrupted"]["per_view"][index]["per_center"][center] for center, result in zip(LOGICAL_CENTERS, results, strict=True)}
        view["evaluated_center_mean"] = mean_metric_views([view["per_center"][center]["metrics"] for center in LOGICAL_CENTERS])
        view.update(center_count=4, center_order=list(LOGICAL_CENTERS), four_center_mean=view["evaluated_center_mean"])
        per_view.append(view)
    aggregates = aggregate_corruption_views(per_view, center_order=LOGICAL_CENTERS, canonical_four_center_order=LOGICAL_CENTERS)
    return {
        "schema_version": 1, "artifact_type": "pn2021_diagonal_four_center_evaluation", "status": "complete",
        "profile_name": profile_name, "model": results[0]["model"], "cohort": identities[0]["cohort"],
        "centers": list(LOGICAL_CENTERS), "members": [item["summary"] for item in identities],
        "clean": {"evaluation_slice": "clean", "per_center": clean_per_center, "center_count": 4, "center_order": list(LOGICAL_CENTERS), "evaluated_center_mean": clean_mean, "four_center_mean": clean_mean},
        "corrupted": {"per_view": per_view, "aggregates": aggregates},
        "aggregation": {"policy": "equal_views_then_equal_centers", "checkpoint_policy": "one_adapted_checkpoint_per_corresponding_center", "off_diagonal": "not_evaluated"},
    }


__all__ = ["aggregate_pn2021_matrix"]
