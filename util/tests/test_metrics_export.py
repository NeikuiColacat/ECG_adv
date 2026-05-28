"""CPU-only tests for metrics_long exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ecg_adv_gen.reporting import (
    MetricsExportError,
    MetricsMergeError,
    export_metrics,
    merge_metrics_long,
)


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _eval_crosscenter(mapping_version: str, mapping_hash: str) -> dict:
    return {
        "scheme": "super5",
        "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
        "label_mapping": {
            "pn2021_super5": {
                "mapping_version": mapping_version,
                "mapping_hash": mapping_hash,
                "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
            }
        },
        "ptbxl_test": {
            "n_records": 10,
            "macro_auroc": 0.91,
            "macro_auprc": 0.77,
            "n_classes_used": 5,
            "per_class": {"CD": {"auroc": 0.9, "auprc": 0.8, "n_pos": 2, "n_valid": 10}},
        },
        "pn2021": {
            "avg_macro_auroc": 0.81,
            "avg_macro_auprc": 0.51,
            "avg_drop_all_zero_macro_auroc": 0.86,
            "avg_drop_all_zero_macro_auprc": 0.61,
            "per_center": {
                "ningbo": {
                    "n_records": 100,
                    "n_excluded_ref": 500,
                    "n_all_zero_labels": 40,
                    "n_nonzero_labels": 60,
                    "macro_auroc": 0.82,
                    "macro_auprc": 0.52,
                    "n_classes_used": 5,
                    "per_class": {"CD": {"auroc": 0.83, "auprc": 0.53, "n_pos": 20, "n_valid": 100}},
                    "drop_all_zero_macro_auroc": 0.87,
                    "drop_all_zero_macro_auprc": 0.62,
                    "drop_all_zero_n_records": 60,
                    "drop_all_zero_n_classes_used": 5,
                    "drop_all_zero_per_class": {
                        "CD": {"auroc": 0.88, "auprc": 0.63, "n_pos": 20, "n_valid": 60}
                    },
                }
            },
        },
    }


def _write_metrics_long(path: Path, rows: list[dict[str, str]]) -> Path:
    from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {key: "" for key in METRICS_FIELDNAMES}
            full.update(row)
            writer.writerow(full)
    return path


def test_export_metrics_long_separates_allzero_views(tmp_path: Path):
    src = _write_json(
        tmp_path / "eval_result_v6.json",
        _eval_crosscenter("v7_super5_sjr_rgq_review_20260528", "555ec85d5b51"),
    )
    manifest = export_metrics([src], tmp_path / "out")

    assert manifest["n_artifacts"] == 1
    assert manifest["views"]["pn2021_all_zero_kept_refexcluded"] > 0
    assert manifest["views"]["pn2021_drop_all_zero_refexcluded"] > 0
    assert manifest["canonical_views"]["pn2021_all_zero_kept_refexcluded"] > 0
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_long.csv").open()))
    assert any(r["view"] == "ptbxl_fold10_source_floor" for r in rows)
    assert any(r["view"] == "pn2021_all_zero_kept_refexcluded" and r["center"] == "ningbo" for r in rows)
    assert any(r["view"] == "pn2021_drop_all_zero_refexcluded" and r["center"] == "ningbo" for r in rows)
    assert all(r["canonical_view"] == r["view"] for r in rows if r["view"].startswith("pn2021_"))
    assert {r["mapping_hash"] for r in rows} == {"555ec85d5b51"}
    drop_center = next(
        r for r in rows
        if r["view"] == "pn2021_drop_all_zero_refexcluded"
        and r["center"] == "ningbo"
        and r["metric"] == "drop_all_zero_macro_auprc"
    )
    assert drop_center["n_records"] == "60"
    assert drop_center["n_classes_used"] == "5"


def test_export_rejects_mixed_mapping_versions(tmp_path: Path):
    v6 = _write_json(tmp_path / "eval_result_v6.json", _eval_crosscenter("v6", "hash6"))
    v5 = _write_json(tmp_path / "eval_result_v5.json", _eval_crosscenter("v5", "hash5"))

    with pytest.raises(MetricsExportError, match="mix mapping"):
        export_metrics([v6, v5], tmp_path / "out")


def test_export_can_fill_expected_mapping_for_legacy_single_run(tmp_path: Path):
    legacy = _write_json(
        tmp_path / "eval_result.json",
        {
            "center": "georgia",
            "target_excluding_ref": {"macro_auroc": 0.9, "macro_auprc": 0.7, "per_class": {}},
            "target_drop_all_zero_excluding_ref": {"macro_auroc": 0.93, "macro_auprc": 0.75},
            "ptbxl_fold10": {"macro_auroc": 0.91, "macro_auprc": 0.76},
        },
    )

    manifest = export_metrics(
        [legacy],
        tmp_path / "out",
        expected_mapping_version="v7_super5_sjr_rgq_review_20260528",
        expected_mapping_hash="555ec85d5b51",
    )

    assert manifest["mapping_pairs"] == ["v7_super5_sjr_rgq_review_20260528|555ec85d5b51"]
    assert manifest["canonical_views"]["pn2021_all_zero_kept_refexcluded"] > 0
    assert manifest["canonical_views"]["pn2021_drop_all_zero_refexcluded"] > 0
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_long.csv").open()))
    kept = next(r for r in rows if r["view"] == "target_all_zero_kept_refexcluded")
    drop = next(r for r in rows if r["view"] == "target_drop_all_zero_refexcluded")
    assert kept["canonical_view"] == "pn2021_all_zero_kept_refexcluded"
    assert drop["canonical_view"] == "pn2021_drop_all_zero_refexcluded"


def test_export_requires_mapping_for_metric_artifacts(tmp_path: Path):
    legacy = _write_json(
        tmp_path / "eval_result.json",
        {"target_excluding_ref": {"macro_auroc": 0.9, "macro_auprc": 0.7}},
    )

    with pytest.raises(MetricsExportError, match="Missing complete mapping metadata"):
        export_metrics([legacy], tmp_path / "out")


def test_export_rejects_partial_mapping_metadata(tmp_path: Path):
    partial = _eval_crosscenter("v7_super5_sjr_rgq_review_20260528", "555ec85d5b51")
    del partial["label_mapping"]["pn2021_super5"]["mapping_hash"]
    src = _write_json(tmp_path / "eval_result_partial.json", partial)

    with pytest.raises(MetricsExportError, match="Missing complete mapping metadata"):
        export_metrics([src], tmp_path / "out")


def test_export_run_id_override_and_target_center_filter(tmp_path: Path):
    run_dir = tmp_path / "ningbo_K500_direct_ft_ep30_seed20260531_val0.2"
    run_dir.mkdir()
    payload = _eval_crosscenter("v7_super5_sjr_rgq_review_20260528", "555ec85d5b51")
    payload["pn2021"]["per_center"]["georgia"] = dict(payload["pn2021"]["per_center"]["ningbo"])
    payload["pn2021"]["per_center"]["georgia"]["macro_auroc"] = 0.7
    src = _write_json(run_dir / "eval_result_v6.json", payload)

    manifest = export_metrics(
        [src],
        tmp_path / "out",
        run_id_override="effnet_direct_k500_v6",
        filter_to_target_center=True,
        target_centers=["ningbo", "georgia"],
    )

    assert manifest["run_id_override"] == "effnet_direct_k500_v6"
    assert manifest["filter_to_target_center"] is True
    assert manifest["artifacts"][0]["source_run_id"] == run_dir.name
    assert manifest["artifacts"][0]["target_center"] == "ningbo"
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_long.csv").open()))
    assert rows
    assert {r["run_id"] for r in rows} == {"effnet_direct_k500_v6"}
    assert {r["center"] for r in rows} == {"ningbo"}
    assert not any(r["scope"] == "dataset" for r in rows)


def test_export_ecgfounder_lhat_final_views_with_target_filter(tmp_path: Path):
    run_dir = tmp_path / "ningbo" / "runs" / "ningbo_K500_M4_lam0p05_ep1_seed20260531"
    run_dir.mkdir(parents=True)
    payload = _eval_crosscenter("v7_super5_sjr_rgq_review_20260528", "555ec85d5b51")
    _write_json(
        run_dir / "eval_result.json",
        {
            "center": "ningbo",
            "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
            "label_mapping": payload["label_mapping"],
            "final_ptbxl_fold10": payload["ptbxl_test"],
            "final_pn2021_views": {"ningbo": payload["pn2021"]},
        },
    )

    manifest = export_metrics(
        [tmp_path / "ningbo"],
        tmp_path / "out",
        run_id_override="ecgfounder_vae_lhat_k500_v6_smoke",
        filter_to_target_center=True,
        target_centers=["ningbo", "georgia"],
        expected_mapping_version="v7_super5_sjr_rgq_review_20260528",
        expected_mapping_hash="555ec85d5b51",
    )

    assert manifest["n_metric_rows"] > 0
    assert manifest["views"]["pn2021_all_zero_kept_refexcluded"] > 0
    assert manifest["views"]["pn2021_drop_all_zero_refexcluded"] > 0
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_long.csv").open()))
    assert {r["run_id"] for r in rows} == {"ecgfounder_vae_lhat_k500_v6_smoke"}
    assert {r["center"] for r in rows} == {"ningbo"}
    assert not any(r["scope"] == "dataset" for r in rows)
    kept = next(
        r for r in rows
        if r["view"] == "pn2021_all_zero_kept_refexcluded"
        and r["metric"] == "macro_auprc"
    )
    assert kept["value"] == "0.52"


def test_export_directory_ignores_training_log_lists(tmp_path: Path):
    run_dir = tmp_path / "runs" / "chapman_shaoxing_K500_fromK500_headft_ep2_seed20260531"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "eval_result.json",
        {
            "center": "chapman_shaoxing",
            "label_mapping": {
                "pn2021_super5": {
                    "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                    "mapping_hash": "555ec85d5b51",
                    "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
                }
            },
            "target_view": {
                "target_center": "chapman_shaoxing",
                "avg_macro_auroc": 0.88,
                "avg_macro_auprc": 0.66,
                "per_center": {
                    "chapman_shaoxing": {
                        "n_records": 20,
                        "n_excluded_ref": 5,
                        "macro_auroc": 0.9,
                        "macro_auprc": 0.7,
                    },
                    "ningbo": {
                        "n_records": 20,
                        "macro_auroc": 0.5,
                        "macro_auprc": 0.4,
                    },
                },
            },
        },
    )
    (run_dir / "training_log.json").write_text(
        json.dumps([{"epoch": 1, "loss": 0.5}]),
        encoding="utf-8",
    )

    manifest = export_metrics(
        [tmp_path / "runs"],
        tmp_path / "out",
        run_id_override="ecgfounder_direct_k500_v6",
        filter_to_target_center=True,
        target_centers=["chapman_shaoxing"],
        expected_mapping_version="v7_super5_sjr_rgq_review_20260528",
        expected_mapping_hash="555ec85d5b51",
    )

    assert manifest["n_artifacts"] == 1
    assert manifest["artifacts"][0]["artifact_type"] == "eval_result"
    assert manifest["n_metric_rows"] == 2
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_long.csv").open()))
    assert {r["center"] for r in rows} == {"chapman_shaoxing"}
    assert {r["canonical_view"] for r in rows} == {"pn2021_all_zero_kept_refexcluded"}


def test_export_target_center_filter_requires_inference(tmp_path: Path):
    src = _write_json(tmp_path / "eval_result_v6.json", _eval_crosscenter("v6", "hash6"))

    with pytest.raises(MetricsExportError, match="Could not infer target center"):
        export_metrics([src], tmp_path / "out", filter_to_target_center=True)


def test_merge_metrics_long_combines_methods(tmp_path: Path):
    a = _write_metrics_long(
        tmp_path / "a.csv",
        [
            {
                "run_id": "direct",
                "dataset": "pn2021",
                "view": "pn2021_all_zero_kept_refexcluded",
                "canonical_view": "pn2021_all_zero_kept_refexcluded",
                "scope": "center",
                "center": "ningbo",
                "metric": "macro_auroc",
                "value": "0.8",
                "mapping_version": "v6",
                "mapping_hash": "hash6",
                "class_order": "CD|HYP|MI|NORM|STTC",
                "source_file": "/x/direct.json",
            }
        ],
    )
    b = _write_metrics_long(
        tmp_path / "b.csv",
        [
            {
                "run_id": "lhat",
                "dataset": "pn2021",
                "view": "target_all_zero_kept_refexcluded",
                "canonical_view": "pn2021_all_zero_kept_refexcluded",
                "scope": "center",
                "center": "ningbo",
                "metric": "macro_auroc",
                "value": "0.82",
                "mapping_version": "v6",
                "mapping_hash": "hash6",
                "class_order": "CD|HYP|MI|NORM|STTC",
                "source_file": "/x/lhat.json",
            }
        ],
    )

    manifest = merge_metrics_long([a, b], tmp_path / "merged")

    assert manifest["n_input_files"] == 2
    assert manifest["n_metric_rows"] == 2
    assert set(manifest["run_ids"]) == {"direct", "lhat"}
    rows = list(csv.DictReader((tmp_path / "merged" / "metrics_long.csv").open()))
    assert len(rows) == 2


def test_merge_metrics_long_rejects_duplicate_metric_keys(tmp_path: Path):
    row = {
        "run_id": "direct",
        "dataset": "pn2021",
        "view": "pn2021_all_zero_kept_refexcluded",
        "canonical_view": "pn2021_all_zero_kept_refexcluded",
        "scope": "center",
        "center": "ningbo",
        "metric": "macro_auroc",
        "value": "0.8",
        "mapping_version": "v6",
        "mapping_hash": "hash6",
        "class_order": "CD|HYP|MI|NORM|STTC",
        "source_file": "/x/a.json",
    }
    a = _write_metrics_long(tmp_path / "a.csv", [row])
    b = _write_metrics_long(tmp_path / "b.csv", [dict(row, value="0.81", source_file="/x/b.json")])

    with pytest.raises(MetricsMergeError, match="Duplicate metric row"):
        merge_metrics_long([a, b], tmp_path / "merged")
