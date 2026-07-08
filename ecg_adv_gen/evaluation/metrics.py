"""Shared ECG macro AUROC/AUPRC helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5


@dataclass(frozen=True)
class MetricRow:
    macro_auroc: float | None
    macro_auprc: float | None
    n_classes_used: int
    per_class: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_auroc": self.macro_auroc,
            "macro_auprc": self.macro_auprc,
            "n_classes_used": self.n_classes_used,
            "per_class": self.per_class,
        }


def _validate_metric_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    names = tuple(str(name) for name in class_names)
    if labels.ndim != 2:
        raise ValueError(f"y_true must be 2D, got shape {labels.shape}")
    if scores.shape != labels.shape:
        raise ValueError(f"y_score shape {scores.shape} does not match y_true shape {labels.shape}")
    if len(names) != labels.shape[1]:
        raise ValueError(f"class_names has {len(names)} entries for {labels.shape[1]} columns")
    return labels, scores, names


def compute_macro_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
    min_pos: int = 10,
    valid_label_min: float | None = None,
    empty_value: float | None = None,
    skip_metric_errors: bool = False,
) -> MetricRow:
    """Compute macro AUROC/AUPRC with legacy ECG class gates.

    A class is used only when it has at least ``min_pos`` positives and at
    least two label values after optional masking. The per-class payload
    intentionally keeps ``n_valid`` rather than ``n_neg`` because existing ECG
    artifacts use that schema.
    """
    labels, scores, names = _validate_metric_inputs(y_true, y_score, class_names)
    aurocs: list[float] = []
    auprcs: list[float] = []
    per_class: dict[str, dict[str, Any]] = {}
    for col, class_name in enumerate(names):
        raw_yt = labels[:, col]
        raw_ys = scores[:, col]
        if valid_label_min is None:
            yt = raw_yt
            ys = raw_ys
            n_valid = int(len(yt))
        else:
            valid = raw_yt >= float(valid_label_min)
            yt = raw_yt[valid]
            ys = raw_ys[valid]
            n_valid = int(valid.sum())
        n_pos = int((yt == 1).sum())
        n_label_values = len(np.unique(yt)) if yt.size else 0
        if n_pos < int(min_pos) or n_label_values < 2:
            per_class[class_name] = {
                "auroc": None,
                "auprc": None,
                "n_pos": n_pos,
                "n_valid": n_valid,
            }
            continue
        try:
            auroc = float(roc_auc_score(yt, ys))
            auprc = float(average_precision_score(yt, ys))
        except Exception:
            if not skip_metric_errors:
                raise
            per_class[class_name] = {
                "auroc": None,
                "auprc": None,
                "n_pos": n_pos,
                "n_valid": n_valid,
            }
            continue
        per_class[class_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "n_pos": n_pos,
            "n_valid": n_valid,
        }
        aurocs.append(auroc)
        auprcs.append(auprc)

    return MetricRow(
        macro_auroc=float(np.mean(aurocs)) if aurocs else empty_value,
        macro_auprc=float(np.mean(auprcs)) if auprcs else empty_value,
        n_classes_used=len(aurocs),
        per_class=per_class,
    )


def compute_macro_metric_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
    min_pos: int = 10,
    valid_label_min: float | None = None,
    empty_value: float | None = None,
    skip_metric_errors: bool = False,
) -> dict[str, Any]:
    return compute_macro_metrics(
        y_true,
        y_score,
        class_names=class_names,
        min_pos=min_pos,
        valid_label_min=valid_label_min,
        empty_value=empty_value,
        skip_metric_errors=skip_metric_errors,
    ).to_dict()


def compute_macro_auroc_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    min_pos: int = 10,
) -> dict[str, Any]:
    """EffNet Super5 macro AUROC/AUPRC wrapper with the historical signature."""
    return compute_macro_metric_dict(
        y_true,
        y_score,
        class_names=class_names,
        min_pos=min_pos,
        valid_label_min=0.0,
        empty_value=float("nan"),
        skip_metric_errors=True,
    )
