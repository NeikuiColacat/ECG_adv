"""CPU-only tests for shared ECG metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_adv_gen.evaluation import compute_macro_metric_dict, compute_macro_metrics
from scripts.triple_labels.train_ptbxl import compute_macro_auroc_auprc as legacy_effnet_macro


def test_compute_macro_metrics_matches_sklearn_per_class_values():
    labels = np.asarray(
        [
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float32,
    )
    scores = np.asarray(
        [
            [0.10, 0.90],
            [0.40, 0.20],
            [0.35, 0.30],
            [0.80, 0.70],
        ],
        dtype=np.float32,
    )

    row = compute_macro_metrics(labels, scores, class_names=["A", "B"], min_pos=1)

    expected_aurocs = [
        roc_auc_score(labels[:, 0], scores[:, 0]),
        roc_auc_score(labels[:, 1], scores[:, 1]),
    ]
    expected_auprcs = [
        average_precision_score(labels[:, 0], scores[:, 0]),
        average_precision_score(labels[:, 1], scores[:, 1]),
    ]
    assert row.n_classes_used == 2
    assert row.macro_auroc == pytest.approx(float(np.mean(expected_aurocs)))
    assert row.macro_auprc == pytest.approx(float(np.mean(expected_auprcs)))
    assert row.per_class["A"]["n_pos"] == 2
    assert row.per_class["A"]["n_valid"] == 4
    assert row.per_class["A"]["auroc"] == pytest.approx(expected_aurocs[0])
    assert row.per_class["B"]["auprc"] == pytest.approx(expected_auprcs[1])


def test_compute_macro_metrics_applies_min_pos_and_all_positive_gate():
    labels = np.asarray(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    scores = np.zeros_like(labels)

    row = compute_macro_metrics(labels, scores, class_names=["low_pos", "all_pos", "none"], min_pos=2)

    assert row.macro_auroc is None
    assert row.macro_auprc is None
    assert row.n_classes_used == 0
    assert row.per_class["low_pos"] == {"auroc": None, "auprc": None, "n_pos": 1, "n_valid": 4}
    assert row.per_class["all_pos"] == {"auroc": None, "auprc": None, "n_pos": 4, "n_valid": 4}
    assert row.per_class["none"] == {"auroc": None, "auprc": None, "n_pos": 0, "n_valid": 4}


def test_compute_macro_metric_dict_shape_and_input_validation():
    labels = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    scores = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)

    metric = compute_macro_metric_dict(labels, scores, class_names=["A", "B"], min_pos=1)

    assert set(metric) == {"macro_auroc", "macro_auprc", "n_classes_used", "per_class"}
    assert metric["n_classes_used"] == 2
    assert set(metric["per_class"]) == {"A", "B"}
    with pytest.raises(ValueError, match="does not match"):
        compute_macro_metrics(labels, scores[:, :1], class_names=["A", "B"])
    with pytest.raises(ValueError, match="class_names"):
        compute_macro_metrics(labels, scores, class_names=["A"])


def test_compute_macro_metrics_masks_unknown_labels_for_effnet_legacy_schema():
    labels = np.asarray(
        [
            [1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray(
        [
            [0.9, 0.1, 0.2],
            [0.2, 0.4, 0.8],
            [0.7, 0.3, 0.6],
            [0.8, 0.9, 0.9],
        ],
        dtype=np.float32,
    )

    metric = compute_macro_metric_dict(
        labels,
        scores,
        class_names=["valid", "unknown", "all_pos"],
        min_pos=1,
        valid_label_min=0.0,
        empty_value=float("nan"),
        skip_metric_errors=True,
    )

    assert metric["n_classes_used"] == 1
    assert metric["per_class"]["valid"]["n_pos"] == 2
    assert metric["per_class"]["valid"]["n_valid"] == 3
    assert metric["per_class"]["valid"]["auroc"] == pytest.approx(1.0)
    assert metric["per_class"]["unknown"] == {"auroc": None, "auprc": None, "n_pos": 0, "n_valid": 0}
    assert metric["per_class"]["all_pos"] == {"auroc": None, "auprc": None, "n_pos": 4, "n_valid": 4}


def test_effnet_legacy_macro_wrapper_matches_package_helper():
    labels = np.asarray(
        [
            [1.0, -1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray(
        [
            [0.8, 0.3],
            [0.4, 0.9],
            [0.7, 0.2],
        ],
        dtype=np.float32,
    )

    expected = compute_macro_metric_dict(
        labels,
        scores,
        class_names=["A", "B"],
        min_pos=1,
        valid_label_min=0.0,
        empty_value=float("nan"),
        skip_metric_errors=True,
    )

    assert legacy_effnet_macro(labels, scores, ["A", "B"], min_pos=1) == expected


def test_effnet_legacy_macro_wrapper_returns_nan_when_no_class_is_usable():
    labels = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    scores = np.zeros_like(labels)

    metric = legacy_effnet_macro(labels, scores, ["unknown", "all_pos"], min_pos=1)

    assert np.isnan(metric["macro_auroc"])
    assert np.isnan(metric["macro_auprc"])
    assert metric["n_classes_used"] == 0
