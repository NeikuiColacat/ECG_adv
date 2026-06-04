"""Pure PN2021 metric-view assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ecg_adv_gen.data import PN2021_EVAL_CENTERS_7, PN2021_TARGET_CENTERS_4

MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, Any]]

DROP_ALL_ZERO_POLICY = "rows with no positive Super5 label are excluded from PN2021 metric calculation"


@dataclass(frozen=True)
class FilteredPN2021CenterRecords:
    """Center-local PN2021 arrays after include/ref-exclusion filters."""

    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    n_total: int
    n_excluded_ref: int
    n_include_kept: int


def _finite_mean(values: Sequence[Any], *, none_if_empty: bool) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None if none_if_empty else float("nan")
    return float(np.mean(finite))


def _record_mask(
    centers: np.ndarray,
    record_ids: np.ndarray,
    *,
    center: str,
    ref_ids: set[str] | None = None,
    include_ids: set[str] | None = None,
) -> tuple[np.ndarray, int, int, int]:
    mask = centers == center
    n_total = int(mask.sum())
    n_include_kept = 0
    if include_ids:
        include_mask = np.asarray([str(rid) in include_ids for rid in record_ids], dtype=bool)
        mask = mask & include_mask
        n_include_kept = int(mask.sum())
    n_excluded_ref = 0
    if ref_ids:
        exclude_mask = np.asarray([str(rid) in ref_ids for rid in record_ids], dtype=bool)
        n_excluded_ref = int((mask & exclude_mask).sum())
        mask = mask & ~exclude_mask
    return mask, n_total, n_excluded_ref, n_include_kept


def filter_pn2021_center_records(
    signals: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    *,
    ref_ids: set[str] | None = None,
    include_ids: set[str] | None = None,
) -> FilteredPN2021CenterRecords:
    """Apply center-local include filtering before K-shot ref exclusion."""

    signals = np.asarray(signals)
    labels = np.asarray(labels)
    record_ids = np.asarray(record_ids).astype(str)
    keep_mask = np.ones(record_ids.shape[0], dtype=bool)
    n_include_kept = 0
    if include_ids:
        include_set = {str(rid) for rid in include_ids}
        include_mask = np.asarray([str(rid) in include_set for rid in record_ids], dtype=bool)
        keep_mask &= include_mask
        n_include_kept = int(keep_mask.sum())
    n_excluded_ref = 0
    if ref_ids:
        ref_set = {str(rid) for rid in ref_ids}
        ref_mask = np.asarray([str(rid) in ref_set for rid in record_ids], dtype=bool)
        n_excluded_ref = int((keep_mask & ref_mask).sum())
        keep_mask &= ~ref_mask
    return FilteredPN2021CenterRecords(
        signals=signals[keep_mask],
        labels=labels[keep_mask],
        record_ids=record_ids[keep_mask],
        n_total=int(record_ids.shape[0]),
        n_excluded_ref=n_excluded_ref,
        n_include_kept=n_include_kept,
    )


def summarize_center_view(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    metric_fn: MetricFn,
    n_total: int,
    n_excluded_ref: int = 0,
    n_include_kept: int = 0,
    extra_fields: dict[str, Any] | None = None,
    report_drop_all_zero: bool = False,
    drop_none_if_empty: bool = False,
    always_include_count_fields: bool = False,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    metric = metric_fn(labels, scores)
    nonzero_mask = np.sum(labels > 0.5, axis=1) > 0
    n_all_zero = int((~nonzero_mask).sum())
    n_nonzero = int(nonzero_mask.sum())
    out = {
        "n_records": int(n_total),
        "n_excluded_ref": int(n_excluded_ref),
        "effective_n": int(labels.shape[0]),
        "n_all_zero_labels": n_all_zero,
        "n_nonzero_labels": n_nonzero,
        **(extra_fields or {}),
        **metric,
    }
    if n_include_kept or always_include_count_fields:
        out["n_include_kept"] = int(n_include_kept)
    if report_drop_all_zero:
        if n_nonzero > 0:
            drop_metric = metric_fn(labels[nonzero_mask], scores[nonzero_mask])
            out.update({
                "drop_all_zero_macro_auroc": drop_metric["macro_auroc"],
                "drop_all_zero_macro_auprc": drop_metric["macro_auprc"],
                "drop_all_zero_n_records": n_nonzero,
                "drop_all_zero_n_classes_used": drop_metric["n_classes_used"],
                "drop_all_zero_per_class": drop_metric["per_class"],
            })
        else:
            empty_value = None if drop_none_if_empty else float("nan")
            out.update({
                "drop_all_zero_macro_auroc": empty_value,
                "drop_all_zero_macro_auprc": empty_value,
                "drop_all_zero_n_records": 0,
                "drop_all_zero_n_classes_used": 0,
                "drop_all_zero_per_class": {},
            })
    return out


def assemble_pn2021_center_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    *,
    metric_fn: MetricFn,
    eval_centers: Sequence[str] = PN2021_EVAL_CENTERS_7,
    ref_ids_by_center: dict[str, set[str]] | None = None,
    include_ids_by_center: dict[str, set[str]] | None = None,
    report_drop_all_zero: bool = False,
    drop_none_if_empty: bool = False,
    extra_by_center: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    centers = np.asarray(centers).astype(str)
    record_ids = np.asarray(record_ids).astype(str)
    ref_ids_by_center = ref_ids_by_center or {}
    include_ids_by_center = include_ids_by_center or {}
    extra_by_center = extra_by_center or {}

    per_center = {}
    macro_aurocs, macro_auprcs = [], []
    drop_aurocs, drop_auprcs = [], []
    for center in eval_centers:
        mask, n_total, n_excluded_ref, n_include_kept = _record_mask(
            centers,
            record_ids,
            center=str(center),
            ref_ids=ref_ids_by_center.get(str(center), set()),
            include_ids=include_ids_by_center.get(str(center), set()),
        )
        if int(mask.sum()) == 0:
            continue
        center_metrics = summarize_center_view(
            labels[mask],
            scores[mask],
            metric_fn=metric_fn,
            n_total=n_total,
            n_excluded_ref=n_excluded_ref,
            n_include_kept=n_include_kept,
            extra_fields=extra_by_center.get(str(center), {}),
            report_drop_all_zero=report_drop_all_zero,
            drop_none_if_empty=drop_none_if_empty,
        )
        per_center[str(center)] = center_metrics
        if center_metrics["macro_auroc"] is not None and np.isfinite(center_metrics["macro_auroc"]):
            macro_aurocs.append(center_metrics["macro_auroc"])
        if center_metrics["macro_auprc"] is not None and np.isfinite(center_metrics["macro_auprc"]):
            macro_auprcs.append(center_metrics["macro_auprc"])
        if report_drop_all_zero:
            d_auroc = center_metrics["drop_all_zero_macro_auroc"]
            d_auprc = center_metrics["drop_all_zero_macro_auprc"]
            if d_auroc is not None and np.isfinite(d_auroc):
                drop_aurocs.append(d_auroc)
            if d_auprc is not None and np.isfinite(d_auprc):
                drop_auprcs.append(d_auprc)

    out = {
        "avg_macro_auroc": _finite_mean(macro_aurocs, none_if_empty=False),
        "avg_macro_auprc": _finite_mean(macro_auprcs, none_if_empty=False),
        "per_center": per_center,
    }
    if report_drop_all_zero:
        out.update({
            "drop_all_zero_policy": DROP_ALL_ZERO_POLICY,
            "avg_drop_all_zero_macro_auroc": _finite_mean(drop_aurocs, none_if_empty=drop_none_if_empty),
            "avg_drop_all_zero_macro_auprc": _finite_mean(drop_auprcs, none_if_empty=drop_none_if_empty),
        })
    return out


def assemble_target_refexcluded_views(
    labels: np.ndarray,
    scores: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    *,
    metric_fn: MetricFn,
    ref_ids_by_center: dict[str, set[str]],
    target_centers: Sequence[str] = PN2021_TARGET_CENTERS_4,
    eval_centers: Sequence[str] = PN2021_EVAL_CENTERS_7,
    report_drop_all_zero: bool = False,
    drop_none_if_empty: bool = True,
) -> dict[str, dict[str, Any]]:
    views = {}
    for target_center in target_centers:
        view = assemble_pn2021_center_metrics(
            labels,
            scores,
            centers,
            record_ids,
            metric_fn=metric_fn,
            eval_centers=eval_centers,
            ref_ids_by_center={str(target_center): set(ref_ids_by_center.get(str(target_center), set()))},
            report_drop_all_zero=report_drop_all_zero,
            drop_none_if_empty=drop_none_if_empty,
        )
        view["target_center"] = str(target_center)
        views[str(target_center)] = view
    return views
