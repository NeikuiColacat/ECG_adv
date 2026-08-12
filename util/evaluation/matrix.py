"""Fail-closed diagonal aggregation for four prospective PN2021 evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from util.evaluation.metrics import aggregate_corruption_views, mean_metric_views
from util.pn2021_artifact_contract import (
    LOGICAL_CENTERS,
    build_artifact_reference,
    load_json_mapping,
    validate_pn2021_evaluation_result,
)


def _load(path: Path) -> dict[str, Any]:
    return load_json_mapping(path, name="evaluation result")


def _member(path: Path, center: str, expected: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _load(path)
    context = validate_pn2021_evaluation_result(
        result, result_path=path, expected_cohort=expected, prospective_only=True
    )
    if context["logical_centers"] != (center,):
        raise ValueError(f"matrix member order differs at {center}")
    lineage, train = context["lineage"], context["train_context"]
    assert isinstance(train, dict)
    return result, {
        "summary": {
            "center": center,
            "evaluation_result": build_artifact_reference(path),
            "train_result": dict(result["subject"]["train_result"]),
            "checkpoint": {
                "path": str(context["checkpoint_path"]),
                "sha256": context["checkpoint_sha256"],
            },
            "adaptation_data": lineage["adaptation_data"],
        },
        "cohort": context["cohort"],
        "signatures": context["signatures"],
    }


def aggregate_pn2021_matrix(
    result_paths: Sequence[str | Path], *, profile_name: str,
    expected_cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and recompute the canonical four-center diagonal matrix."""

    candidates = [Path(value).expanduser() for value in result_paths]
    if any(path.is_symlink() for path in candidates):
        raise ValueError("matrix evaluation results cannot be symbolic links")
    paths = [path.resolve() for path in candidates]
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("matrix aggregation requires four distinct evaluation results")
    loaded = [_member(path, center, expected_cohort)
              for path, center in zip(paths, LOGICAL_CENTERS, strict=True)]
    results, identities = zip(*loaded, strict=True)
    if any(item["cohort"] != identities[0]["cohort"] for item in identities[1:]):
        raise ValueError("matrix members do not share one locked evaluation cohort")
    if any(item["signatures"] != identities[0]["signatures"] for item in identities[1:]):
        raise ValueError("matrix members use different corruption views")
    clean_per_center = {center: result["clean"]["per_center"][center]
                        for center, result in zip(LOGICAL_CENTERS, results, strict=True)}
    clean_mean = mean_metric_views([clean_per_center[center]["metrics"] for center in LOGICAL_CENTERS])
    per_view: list[dict[str, Any]] = []
    for index, signature in enumerate(identities[0]["signatures"]):
        view = {key: results[0]["corrupted"]["per_view"][index][key]
                for key in ("view_index", "depth", "operators", "composition_id")}
        view["evaluation_slice"] = f"depth{signature[1]}"
        view["per_center"] = {center: result["corrupted"]["per_view"][index]["per_center"][center]
                              for center, result in zip(LOGICAL_CENTERS, results, strict=True)}
        mean = mean_metric_views([view["per_center"][center]["metrics"] for center in LOGICAL_CENTERS])
        view.update(evaluated_center_mean=mean, center_count=4,
                    center_order=list(LOGICAL_CENTERS), four_center_mean=mean)
        per_view.append(view)
    aggregates = aggregate_corruption_views(
        per_view, center_order=LOGICAL_CENTERS,
        canonical_four_center_order=LOGICAL_CENTERS,
    )
    return {
        "schema_version": 1, "artifact_type": "pn2021_diagonal_four_center_evaluation",
        "status": "complete", "profile_name": profile_name, "model": results[0]["model"],
        "cohort": identities[0]["cohort"], "centers": list(LOGICAL_CENTERS),
        "members": [item["summary"] for item in identities],
        "clean": {"evaluation_slice": "clean", "per_center": clean_per_center,
                  "center_count": 4, "center_order": list(LOGICAL_CENTERS),
                  "evaluated_center_mean": clean_mean, "four_center_mean": clean_mean},
        "corrupted": {"per_view": per_view, "aggregates": aggregates},
        "aggregation": {"policy": "equal_views_then_equal_centers",
                        "checkpoint_policy": "one_adapted_checkpoint_per_corresponding_center",
                        "off_diagonal": "not_evaluated"},
    }


__all__ = ["aggregate_pn2021_matrix"]
