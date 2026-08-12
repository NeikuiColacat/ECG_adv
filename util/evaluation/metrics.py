"""Strict Super5 metric views and equal-weight PN2021 aggregations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics import average_precision_score, roc_auc_score

from models.contracts import CLASS_ORDER


CLEAN_VIEW_ALIASES = (
    "pn2021_all_zero_kept_refexcluded",
    "pn2021_drop_all_zero_refexcluded",
)
CORRUPTED_VIEW_ALIASES = (
    "pn2021c_all_zero_kept_corrupted_refexcluded",
    "pn2021c_drop_all_zero_corrupted_refexcluded",
)
UndefinedClassPolicy = Literal["strict", "skip_undefined"]
METRIC_DEFINITION = {
    "score_input": "raw_logits",
    "auroc": "sklearn.metrics.roc_auc_score",
    "auprc": "sklearn.metrics.average_precision_score",
    "auprc_definition": "average_precision_not_trapezoidal_pr_auc",
    "class_aggregation": "unweighted_arithmetic_mean_over_classes_used",
    "sklearn_version": SKLEARN_VERSION,
}


def _sha256_lines(values: Sequence[str], *, sort: bool) -> str:
    normalized = [str(value) for value in values]
    if sort:
        normalized.sort()
    payload = "".join(f"{value}\n" for value in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_label_set_sha256(hash_ids: Sequence[str], targets: np.ndarray) -> str:
    rows = sorted(
        (str(hash_id), "".join(str(int(value)) for value in target))
        for hash_id, target in zip(hash_ids, targets, strict=True)
    )
    payload = "".join(f"{hash_id}|{label}\n" for hash_id, label in rows).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _validated_arrays(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    class_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(targets)
    classes = tuple(str(value) for value in class_order)
    if classes != CLASS_ORDER:
        raise ValueError(f"class_order must be {CLASS_ORDER}, got {classes}")
    if scores.ndim != 2 or truth.ndim != 2 or scores.shape != truth.shape:
        raise ValueError("logits and targets must share shape (N,C)")
    if scores.shape[0] <= 0 or scores.shape[1] != len(classes):
        raise ValueError("metric arrays must contain at least one Super5 record")
    if not np.isfinite(scores).all():
        raise ValueError("logits contain NaN or Inf")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("targets must contain only binary 0/1 values")
    return scores, truth.astype(np.uint8, copy=False), classes


def compute_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    class_order: Sequence[str] = CLASS_ORDER,
    undefined_class_policy: UndefinedClassPolicy = "strict",
) -> dict[str, Any]:
    """Compute Super5 one-vs-rest AUROC and Average Precision from raw logits.

    ``strict`` requires every class to contain at least one positive and one
    negative record. ``skip_undefined`` omits such classes from both macro
    metrics so AUROC and AUPRC always use the same explicit class denominator.
    """

    scores, truth, classes = _validated_arrays(
        logits,
        targets,
        class_order=class_order,
    )
    if undefined_class_policy not in {"strict", "skip_undefined"}:
        raise ValueError(
            "undefined_class_policy must be 'strict' or 'skip_undefined'"
        )
    per_class: dict[str, dict[str, Any]] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    classes_used: list[str] = []
    for index, class_name in enumerate(classes):
        target = truth[:, index]
        score = scores[:, index]
        positives = int(target.sum())
        negatives = int(target.size - positives)
        if positives <= 0 or negatives <= 0:
            reason = "no_positive_examples" if positives <= 0 else "no_negative_examples"
            if undefined_class_policy == "strict":
                raise ValueError(
                    f"class {class_name} requires both positive and negative "
                    f"examples: positives={positives}, negatives={negatives}"
                )
            per_class[class_name] = {
                "positives": positives,
                "negatives": negatives,
                "defined": False,
                "undefined_reason": reason,
                "auroc": None,
                "auprc": None,
            }
            continue
        auroc = float(roc_auc_score(target, score))
        auprc = float(average_precision_score(target, score))
        per_class[class_name] = {
            "positives": positives,
            "negatives": negatives,
            "defined": True,
            "undefined_reason": None,
            "auroc": auroc,
            "auprc": auprc,
        }
        aurocs.append(auroc)
        auprcs.append(auprc)
        classes_used.append(class_name)
    if not classes_used:
        raise ValueError("no class contains both positive and negative examples")
    return {
        "sample_count": int(truth.shape[0]),
        "metric_definition": dict(METRIC_DEFINITION),
        "undefined_class_policy": undefined_class_policy,
        "class_order": list(classes),
        "classes_used": classes_used,
        "n_classes_used": len(classes_used),
        "n_classes_total": len(classes),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_auprc": float(np.mean(auprcs)),
        "per_class": per_class,
    }


def compute_metric_views(
    logits: np.ndarray,
    targets: np.ndarray,
    hash_ids: Sequence[str],
    *,
    corrupted: bool,
    class_order: Sequence[str] = CLASS_ORDER,
) -> dict[str, dict[str, Any]]:
    """Emit the canonical all-zero-kept and drop-all-zero metric views."""

    scores, truth, _ = _validated_arrays(
        logits,
        targets,
        class_order=class_order,
    )
    hashes = tuple(str(value) for value in hash_ids)
    if len(hashes) != truth.shape[0] or len(set(hashes)) != len(hashes):
        raise ValueError("hash_ids must be unique and aligned with metric records")
    keep_mask = np.ones(truth.shape[0], dtype=bool)
    drop_mask = truth.sum(axis=1) > 0
    if not bool(drop_mask.any()):
        raise ValueError("drop-all-zero metric view would be empty")
    aliases = CORRUPTED_VIEW_ALIASES if corrupted else CLEAN_VIEW_ALIASES
    result: dict[str, dict[str, Any]] = {}
    all_zero_count = int((~drop_mask).sum())
    for alias, policy, mask in (
        (aliases[0], "kept", keep_mask),
        (aliases[1], "drop", drop_mask),
    ):
        selected_hashes = tuple(
            value for value, selected in zip(hashes, mask, strict=True) if selected
        )
        selected_truth = truth[mask]
        payload = compute_classification_metrics(
            scores[mask],
            selected_truth,
            class_order=class_order,
        )
        payload.update(
            {
                "canonical_view": alias,
                "all_zero_policy": policy,
                "all_zero_count_in_source_selection": all_zero_count,
                "record_identity": {
                    "hash_id_set_sha256": _sha256_lines(
                        selected_hashes, sort=True
                    ),
                    "ordered_hash_ids_sha256": _sha256_lines(
                        selected_hashes, sort=False
                    ),
                    "hash_label_set_sha256": _hash_label_set_sha256(
                        selected_hashes, selected_truth
                    ),
                },
            }
        )
        result[alias] = payload
    return result


def _mean_metric_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one metric payload is required")
    alias = str(payloads[0].get("canonical_view", ""))
    policy = str(payloads[0].get("all_zero_policy", ""))
    if not alias or not policy:
        raise ValueError("metric payloads require canonical view and all-zero policy")
    if any(
        str(item.get("canonical_view")) != alias
        or str(item.get("all_zero_policy")) != policy
        for item in payloads
    ):
        raise ValueError("cannot aggregate different canonical metric views")
    metric_definition = dict(payloads[0].get("metric_definition", {}))
    undefined_class_policy = str(payloads[0].get("undefined_class_policy", ""))
    classes_used = tuple(str(value) for value in payloads[0].get("classes_used", ()))
    if (
        not metric_definition
        or not undefined_class_policy
        or not classes_used
        or any(
            dict(item.get("metric_definition", {})) != metric_definition
            or str(item.get("undefined_class_policy", ""))
            != undefined_class_policy
            or tuple(str(value) for value in item.get("classes_used", ()))
            != classes_used
            for item in payloads
        )
    ):
        raise ValueError("metric payloads use incompatible metric definitions")
    per_class: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_ORDER:
        entries = [item["per_class"][class_name] for item in payloads]
        is_defined = class_name in classes_used
        per_class[class_name] = {
            "positives_sum": int(
                sum(
                    int(item.get("positives_sum", item.get("positives", 0)))
                    for item in entries
                )
            ),
            "negatives_sum": int(
                sum(
                    int(item.get("negatives_sum", item.get("negatives", 0)))
                    for item in entries
                )
            ),
            "defined": is_defined,
            "undefined_reason": (
                None if is_defined else str(entries[0].get("undefined_reason"))
            ),
            "auroc": (
                float(np.mean([float(item["auroc"]) for item in entries]))
                if is_defined
                else None
            ),
            "auprc": (
                float(np.mean([float(item["auprc"]) for item in entries]))
                if is_defined
                else None
            ),
        }
    return {
        "canonical_view": alias,
        "all_zero_policy": policy,
        "metric_definition": metric_definition,
        "undefined_class_policy": undefined_class_policy,
        "class_order": list(CLASS_ORDER),
        "classes_used": list(classes_used),
        "n_classes_used": len(classes_used),
        "n_classes_total": len(CLASS_ORDER),
        "aggregation": "equal_weighted_arithmetic_mean",
        "member_count": len(payloads),
        "source_member_count_sum": int(
            sum(int(item.get("member_count", 1)) for item in payloads)
        ),
        "sample_count_sum": int(
            sum(
                int(item.get("sample_count_sum", item.get("sample_count", 0)))
                for item in payloads
            )
        ),
        "all_zero_count_sum": int(
            sum(
                int(
                    item.get(
                        "all_zero_count_sum",
                        item.get("all_zero_count_in_source_selection", 0),
                    )
                )
                for item in payloads
            )
        ),
        "macro_auroc": float(
            np.mean([float(item["macro_auroc"]) for item in payloads])
        ),
        "macro_auprc": float(
            np.mean([float(item["macro_auprc"]) for item in payloads])
        ),
        "per_class": per_class,
    }


def mean_metric_views(
    members: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Equal-weight a sequence of complete canonical metric-view mappings."""

    if not members:
        raise ValueError("at least one metric-view mapping is required")
    keys = tuple(members[0])
    if any(tuple(member) != keys for member in members):
        raise ValueError("metric-view mappings must have identical ordered keys")
    return {
        key: _mean_metric_payloads([member[key] for member in members]) for key in keys
    }


def resolve_evaluated_center_mean(
    payload: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Read the schema-v2 center mean or its schema-v1 compatibility alias.

    Historical evaluation JSONs called every selected-center mean
    ``four_center_mean``, including one-center runs.  The legacy field is
    therefore accepted only as a generic evaluated-center mean; callers must
    not infer that it represents four centers without separately validating
    the result's center count/order.
    """

    current = payload.get("evaluated_center_mean")
    if isinstance(current, Mapping):
        return current
    legacy = payload.get("four_center_mean")
    if isinstance(legacy, Mapping):
        return legacy
    raise ValueError(
        "center aggregate requires evaluated_center_mean or legacy four_center_mean"
    )


def _annotate_slice(
    views: Mapping[str, Mapping[str, Any]], evaluation_slice: str
) -> dict[str, dict[str, Any]]:
    return {
        alias: {**dict(payload), "evaluation_slice": evaluation_slice}
        for alias, payload in views.items()
    }


def aggregate_corruption_views(
    per_view: Sequence[Mapping[str, Any]],
    *,
    center_order: Sequence[str],
    canonical_four_center_order: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate depth2, depth3, and depth23 with equal view/center weight."""

    centers = tuple(str(value) for value in center_order)
    if not per_view or not centers or len(set(centers)) != len(centers):
        raise ValueError("corruption aggregation requires views and unique centers")
    groups = {
        "depth2": [item for item in per_view if int(item["depth"]) == 2],
        "depth3": [item for item in per_view if int(item["depth"]) == 3],
        "depth23": list(per_view),
    }
    result: dict[str, dict[str, Any]] = {}
    for slice_name, entries in groups.items():
        if not entries:
            raise ValueError(f"corruption aggregate {slice_name} has no views")
        center_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for center in centers:
            try:
                members = [entry["per_center"][center]["metrics"] for entry in entries]
            except KeyError as exc:
                raise ValueError(f"corruption view is missing center {center}") from exc
            center_metrics[center] = _annotate_slice(
                mean_metric_views(members), slice_name
            )
        evaluated_center_mean = _annotate_slice(
            mean_metric_views([center_metrics[center] for center in centers]),
            slice_name,
        )
        canonical_centers = (
            ()
            if canonical_four_center_order is None
            else tuple(str(value) for value in canonical_four_center_order)
        )
        is_exact_four_center = centers == canonical_centers and len(centers) == 4
        result[slice_name] = {
            "evaluation_slice": slice_name,
            "aggregation": "equal_views_then_equal_centers",
            "view_count": len(entries),
            "composition_ids": [str(item["composition_id"]) for item in entries],
            "per_center": center_metrics,
            "center_count": len(centers),
            "center_order": list(centers),
            "evaluated_center_mean": evaluated_center_mean,
            "four_center_mean": (
                evaluated_center_mean if is_exact_four_center else None
            ),
        }
    return result


__all__ = [
    "CLEAN_VIEW_ALIASES",
    "CORRUPTED_VIEW_ALIASES",
    "METRIC_DEFINITION",
    "UndefinedClassPolicy",
    "aggregate_corruption_views",
    "compute_classification_metrics",
    "compute_metric_views",
    "mean_metric_views",
    "resolve_evaluated_center_mean",
]
