"""CPU-only tests for PN2021 metric view assembly."""

from __future__ import annotations

import numpy as np

from ecg_adv_gen.evaluation import (
    DROP_ALL_ZERO_POLICY,
    assemble_pn2021_center_metrics,
    assemble_target_refexcluded_views,
    filter_pn2021_center_records,
    summarize_center_view,
)


def fake_metric(labels: np.ndarray, scores: np.ndarray) -> dict:
    n = labels.shape[0]
    positives = labels.sum(axis=0)
    used = int((positives > 0).sum())
    return {
        "macro_auroc": float(n) / 100.0,
        "macro_auprc": float(used) / 10.0,
        "n_classes_used": used,
        "per_class": {
            str(i): {"n_pos": int(v), "n_valid": int(n), "auroc": None, "auprc": None}
            for i, v in enumerate(positives)
        },
    }


def test_filter_pn2021_center_records_applies_include_then_ref_exclusion():
    signals = np.asarray(
        [
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[4.0]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    record_ids = np.asarray(["N1", "N2", "N3", "N4"])

    filtered = filter_pn2021_center_records(
        signals,
        labels,
        record_ids,
        include_ids={"N1", "N2", "N3"},
        ref_ids={"N2", "N4"},
    )

    assert filtered.n_total == 4
    assert filtered.n_include_kept == 3
    assert filtered.n_excluded_ref == 1
    assert filtered.record_ids.tolist() == ["N1", "N3"]
    assert filtered.signals[:, 0, 0].tolist() == [1.0, 3.0]
    assert filtered.labels.tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_assemble_pn2021_center_metrics_applies_include_then_ref_exclusion():
    labels = np.asarray(
        [
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    scores = np.zeros_like(labels)
    centers = np.asarray(["ningbo", "ningbo", "ningbo", "georgia", "georgia"])
    record_ids = np.asarray(["N1", "N2", "N3", "G1", "G2"])

    result = assemble_pn2021_center_metrics(
        labels,
        scores,
        centers,
        record_ids,
        metric_fn=fake_metric,
        eval_centers=["ningbo", "georgia"],
        ref_ids_by_center={"ningbo": {"N2"}},
        include_ids_by_center={"ningbo": {"N1", "N2"}},
        report_drop_all_zero=True,
    )

    assert result["avg_macro_auroc"] == 0.015
    assert result["avg_macro_auprc"] == 0.1
    assert result["drop_all_zero_policy"] == DROP_ALL_ZERO_POLICY
    assert result["avg_drop_all_zero_macro_auroc"] == 0.01
    assert result["per_center"]["ningbo"]["n_records"] == 3
    assert result["per_center"]["ningbo"]["n_include_kept"] == 2
    assert result["per_center"]["ningbo"]["n_excluded_ref"] == 1
    assert result["per_center"]["ningbo"]["effective_n"] == 1
    assert result["per_center"]["ningbo"]["n_all_zero_labels"] == 0
    assert result["per_center"]["georgia"]["n_all_zero_labels"] == 1
    assert result["per_center"]["georgia"]["drop_all_zero_n_records"] == 1


def test_summarize_center_view_matches_eval_crosscenter_legacy_row_shape():
    labels = np.asarray(
        [
            [1, 0],
            [0, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    scores = np.zeros_like(labels)

    row = summarize_center_view(
        labels,
        scores,
        metric_fn=fake_metric,
        n_total=3,
        n_excluded_ref=1,
        n_include_kept=0,
        extra_fields={"n_scanned": 5, "load_time_s": 1.2, "cache_hit": True, "cache_kind": "npz"},
        report_drop_all_zero=True,
        always_include_count_fields=True,
    )

    assert row == {
        "n_records": 3,
        "n_excluded_ref": 1,
        "effective_n": 3,
        "n_all_zero_labels": 1,
        "n_nonzero_labels": 2,
        "n_scanned": 5,
        "load_time_s": 1.2,
        "cache_hit": True,
        "cache_kind": "npz",
        "macro_auroc": 0.03,
        "macro_auprc": 0.2,
        "n_classes_used": 2,
        "per_class": {
            "0": {"n_pos": 1, "n_valid": 3, "auroc": None, "auprc": None},
            "1": {"n_pos": 1, "n_valid": 3, "auroc": None, "auprc": None},
        },
        "n_include_kept": 0,
        "drop_all_zero_macro_auroc": 0.02,
        "drop_all_zero_macro_auprc": 0.2,
        "drop_all_zero_n_records": 2,
        "drop_all_zero_n_classes_used": 2,
        "drop_all_zero_per_class": {
            "0": {"n_pos": 1, "n_valid": 2, "auroc": None, "auprc": None},
            "1": {"n_pos": 1, "n_valid": 2, "auroc": None, "auprc": None},
        },
    }


def test_assemble_target_refexcluded_views_only_excludes_target_center_refs():
    labels = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 0],
        ],
        dtype=np.float32,
    )
    scores = np.zeros_like(labels)
    centers = np.asarray(["ningbo", "georgia", "ningbo", "georgia"])
    record_ids = np.asarray(["N1", "G1", "N2", "G2"])

    views = assemble_target_refexcluded_views(
        labels,
        scores,
        centers,
        record_ids,
        metric_fn=fake_metric,
        ref_ids_by_center={"ningbo": {"N1"}, "georgia": {"G1"}},
        target_centers=["ningbo", "georgia"],
        eval_centers=["ningbo", "georgia"],
        report_drop_all_zero=True,
    )

    assert views["ningbo"]["target_center"] == "ningbo"
    assert views["ningbo"]["per_center"]["ningbo"]["effective_n"] == 1
    assert views["ningbo"]["per_center"]["georgia"]["effective_n"] == 2
    assert views["georgia"]["per_center"]["ningbo"]["effective_n"] == 2
    assert views["georgia"]["per_center"]["georgia"]["effective_n"] == 1
    assert views["georgia"]["avg_drop_all_zero_macro_auroc"] == 0.02


def test_assemble_target_refexcluded_views_matches_legacy_loop_shape():
    labels = np.asarray(
        [
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    scores = np.zeros_like(labels)
    centers = np.asarray(["ningbo", "ningbo", "georgia", "georgia", "cpsc_2018"])
    record_ids = np.asarray(["N1", "N2", "G1", "G2", "C1"])
    ref_ids = {"ningbo": {"N2"}, "georgia": {"G1"}}
    eval_centers = ["ningbo", "georgia", "cpsc_2018"]
    target_centers = ["ningbo", "georgia"]

    views = assemble_target_refexcluded_views(
        labels,
        scores,
        centers,
        record_ids,
        metric_fn=fake_metric,
        ref_ids_by_center=ref_ids,
        target_centers=target_centers,
        eval_centers=eval_centers,
        report_drop_all_zero=True,
    )

    def legacy_view(target_center: str) -> dict:
        per_center = {}
        avg_aurocs, avg_auprcs = [], []
        drop_aurocs, drop_auprcs = [], []
        for center in eval_centers:
            mask = centers == center
            n_total = int(mask.sum())
            n_excluded = 0
            if center == target_center:
                exclude = np.asarray([rid in ref_ids[target_center] for rid in record_ids], dtype=bool)
                n_excluded = int((mask & exclude).sum())
                mask = mask & ~exclude
            if int(mask.sum()) == 0:
                continue
            center_labels = labels[mask]
            center_scores = scores[mask]
            metric = fake_metric(center_labels, center_scores)
            nonzero = np.sum(center_labels > 0.5, axis=1) > 0
            row = {
                "n_records": n_total,
                "n_excluded_ref": n_excluded,
                "effective_n": int(mask.sum()),
                "n_all_zero_labels": int((~nonzero).sum()),
                "n_nonzero_labels": int(nonzero.sum()),
                **metric,
            }
            if int(nonzero.sum()) > 0:
                drop = fake_metric(center_labels[nonzero], center_scores[nonzero])
                row.update({
                    "drop_all_zero_macro_auroc": drop["macro_auroc"],
                    "drop_all_zero_macro_auprc": drop["macro_auprc"],
                    "drop_all_zero_n_records": int(nonzero.sum()),
                    "drop_all_zero_n_classes_used": drop["n_classes_used"],
                    "drop_all_zero_per_class": drop["per_class"],
                })
                drop_aurocs.append(drop["macro_auroc"])
                drop_auprcs.append(drop["macro_auprc"])
            else:
                row.update({
                    "drop_all_zero_macro_auroc": None,
                    "drop_all_zero_macro_auprc": None,
                    "drop_all_zero_n_records": 0,
                    "drop_all_zero_n_classes_used": 0,
                    "drop_all_zero_per_class": {},
                })
            per_center[center] = row
            avg_aurocs.append(metric["macro_auroc"])
            avg_auprcs.append(metric["macro_auprc"])
        return {
            "target_center": target_center,
            "avg_macro_auroc": float(np.mean(avg_aurocs)),
            "avg_macro_auprc": float(np.mean(avg_auprcs)),
            "per_center": per_center,
            "drop_all_zero_policy": DROP_ALL_ZERO_POLICY,
            "avg_drop_all_zero_macro_auroc": float(np.mean(drop_aurocs)),
            "avg_drop_all_zero_macro_auprc": float(np.mean(drop_auprcs)),
        }

    assert views["ningbo"] == legacy_view("ningbo")
    assert views["georgia"] == legacy_view("georgia")
