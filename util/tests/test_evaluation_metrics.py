from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from models.contracts import CLASS_ORDER
from util.evaluation.metrics import (
    CLEAN_VIEW_ALIASES,
    compute_classification_metrics,
    compute_metric_views,
    resolve_evaluated_center_mean,
)


def _five_columns(values: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(values)[:, None], (1, len(CLASS_ORDER)))


def test_metrics_use_raw_logits_without_sigmoid_saturation() -> None:
    targets = _five_columns(np.asarray([0, 1, 0, 1], dtype=np.uint8))
    logits = _five_columns(np.asarray([17.0, 20.0, 18.0, 21.0], dtype=np.float32))

    result = compute_classification_metrics(logits, targets)

    assert roc_auc_score(targets, logits, average="macro") == 1.0
    assert average_precision_score(targets, logits, average="macro") == 1.0
    assert result["macro_auroc"] == 1.0
    assert result["macro_auprc"] == 1.0
    assert result["metric_definition"]["score_input"] == "raw_logits"


def test_auprc_is_explicitly_sklearn_average_precision_not_trapezoidal_auc() -> None:
    target = np.asarray([0, 1, 0, 1, 1], dtype=np.uint8)
    score = np.asarray([0.1, 0.4, 0.35, 0.8, 0.2], dtype=np.float64)
    targets = _five_columns(target)
    logits = _five_columns(score)
    precision, recall, _ = precision_recall_curve(target, score)
    expected_ap = average_precision_score(target, score)
    trapezoidal = auc(recall, precision)

    result = compute_classification_metrics(logits, targets)

    assert expected_ap != pytest.approx(trapezoidal)
    assert result["macro_auprc"] == pytest.approx(expected_ap)
    assert (
        result["metric_definition"]["auprc_definition"]
        == "average_precision_not_trapezoidal_pr_auc"
    )


def test_undefined_class_policy_is_explicit_and_uses_one_macro_denominator() -> None:
    targets = np.asarray(
        [
            [0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    logits = np.arange(targets.size, dtype=np.float64).reshape(targets.shape)

    with pytest.raises(ValueError, match="class HYP requires both positive and negative"):
        compute_classification_metrics(logits, targets, undefined_class_policy="strict")

    result = compute_classification_metrics(
        logits,
        targets,
        undefined_class_policy="skip_undefined",
    )

    assert result["classes_used"] == ["CD", "NORM", "STTC"]
    assert result["n_classes_used"] == 3
    assert result["n_classes_total"] == 5
    assert result["per_class"]["HYP"] == {
        "positives": 4,
        "negatives": 0,
        "defined": False,
        "undefined_reason": "no_negative_examples",
        "auroc": None,
        "auprc": None,
    }
    assert result["per_class"]["MI"]["undefined_reason"] == "no_positive_examples"


def test_metrics_reject_noncanonical_or_duplicate_class_order() -> None:
    targets = _five_columns(np.asarray([0, 1, 0, 1], dtype=np.uint8))
    logits = np.zeros_like(targets, dtype=np.float64)

    with pytest.raises(ValueError, match="class_order must be"):
        compute_classification_metrics(
            logits,
            targets,
            class_order=("CD", "CD", "MI", "NORM", "STTC"),
        )


def test_drop_all_zero_view_filters_only_rows_without_any_super5_label() -> None:
    targets = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    logits = targets.astype(np.float64) * 2.0 - 1.0
    hashes = [f"hash-{index}" for index in range(len(targets))]

    views = compute_metric_views(logits, targets, hashes, corrupted=False)
    kept = views[CLEAN_VIEW_ALIASES[0]]
    dropped = views[CLEAN_VIEW_ALIASES[1]]

    assert kept["sample_count"] == 7
    assert dropped["sample_count"] == 6
    assert kept["all_zero_count_in_source_selection"] == 1
    assert dropped["all_zero_count_in_source_selection"] == 1
    assert kept["all_zero_policy"] == "kept"
    assert dropped["all_zero_policy"] == "drop"
    assert (
        kept["record_identity"]["hash_id_set_sha256"]
        != dropped["record_identity"]["hash_id_set_sha256"]
    )


def test_legacy_four_center_field_is_only_read_as_generic_center_mean() -> None:
    metric = {"pn2021_drop_all_zero_refexcluded": {"macro_auroc": 0.75}}

    assert resolve_evaluated_center_mean({"evaluated_center_mean": metric}) is metric
    assert resolve_evaluated_center_mean({"four_center_mean": metric}) is metric
    with pytest.raises(ValueError, match="center aggregate requires"):
        resolve_evaluated_center_mean({"four_center_mean": None})
