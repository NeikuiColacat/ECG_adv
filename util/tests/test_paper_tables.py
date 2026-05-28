"""CPU-only tests for paper table export from metrics_long.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ecg_adv_gen.reporting import PaperTableError, export_paper_table


FIELDS = [
    "run_id",
    "source_file",
    "artifact_type",
    "dataset",
    "view",
    "canonical_view",
    "scope",
    "center",
    "class_name",
    "metric",
    "value",
    "n_records",
    "n_excluded_ref",
    "n_all_zero_labels",
    "n_nonzero_labels",
    "n_classes_used",
    "n_pos",
    "n_valid",
    "mapping_version",
    "mapping_hash",
    "class_order",
]


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            full = {key: "" for key in FIELDS}
            full.update(row)
            writer.writerow(full)
    return path


def _row(run: str, view: str, center: str, metric: str, value: float, *, mapping_hash: str = "hash6"):
    canonical_view = {
        "target_all_zero_kept_refexcluded": "pn2021_all_zero_kept_refexcluded",
        "target_drop_all_zero_refexcluded": "pn2021_drop_all_zero_refexcluded",
    }.get(view, view)
    return {
        "run_id": run,
        "source_file": f"/x/{run}.json",
        "artifact_type": "eval_result",
        "dataset": "pn2021",
        "view": view,
        "canonical_view": canonical_view,
        "scope": "center",
        "center": center,
        "metric": metric,
        "value": str(value),
        "mapping_version": "v6",
        "mapping_hash": mapping_hash,
        "class_order": "CD|HYP|MI|NORM|STTC",
    }


def test_export_paper_table_center_mean_and_delta(tmp_path: Path):
    centers = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
    rows = []
    for i, center in enumerate(centers):
        rows.append(_row("direct", "pn2021_all_zero_kept_refexcluded", center, "macro_auroc", 0.80 + i * 0.01))
        rows.append(_row("direct", "pn2021_all_zero_kept_refexcluded", center, "macro_auprc", 0.50 + i * 0.01))
        rows.append(_row("lhat", "pn2021_all_zero_kept_refexcluded", center, "macro_auroc", 0.82 + i * 0.01))
        rows.append(_row("lhat", "pn2021_all_zero_kept_refexcluded", center, "macro_auprc", 0.54 + i * 0.01))
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    manifest = export_paper_table(
        metrics,
        tmp_path / "out",
        view="pn2021_all_zero_kept_refexcluded",
        dataset="pn2021",
        centers=centers,
        include_dataset_rows=False,
        baseline_run_id="direct",
    )

    assert manifest["n_table_rows"] == 2
    table = list(csv.DictReader((tmp_path / "out" / "paper_table.csv").open()))
    lhat = next(r for r in table if r["run_id"] == "lhat")
    assert lhat["scope"] == "center_mean"
    assert lhat["n_centers"] == "4"
    assert abs(float(lhat["macro_auroc"]) - 0.835) < 1e-9
    assert abs(float(lhat["delta_macro_auroc_vs_baseline"]) - 0.02) < 1e-9
    assert abs(float(lhat["delta_macro_auprc_vs_baseline"]) - 0.04) < 1e-9


def test_export_paper_table_keeps_views_separate(tmp_path: Path):
    rows = [
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8),
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auprc", 0.5),
        _row("run", "pn2021_drop_all_zero_refexcluded", "ningbo", "drop_all_zero_macro_auroc", 0.9),
        _row("run", "pn2021_drop_all_zero_refexcluded", "ningbo", "drop_all_zero_macro_auprc", 0.6),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    export_paper_table(
        metrics,
        tmp_path / "out",
        view="pn2021_drop_all_zero_refexcluded",
        dataset="pn2021",
        include_dataset_rows=False,
    )
    table = list(csv.DictReader((tmp_path / "out" / "paper_table.csv").open()))
    assert len(table) == 1
    assert table[0]["view"] == "pn2021_drop_all_zero_refexcluded"
    assert table[0]["macro_auroc"] == "0.9"
    assert table[0]["raw_views"] == "pn2021_drop_all_zero_refexcluded"


def test_export_paper_table_uses_canonical_view_aliases(tmp_path: Path):
    rows = [
        _row("direct", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8),
        _row("direct", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auprc", 0.5),
        _row("ecgfounder", "target_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.84),
        _row("ecgfounder", "target_all_zero_kept_refexcluded", "ningbo", "macro_auprc", 0.56),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    export_paper_table(
        metrics,
        tmp_path / "out",
        view="pn2021_all_zero_kept_refexcluded",
        dataset="pn2021",
        centers=["ningbo"],
        include_dataset_rows=False,
        baseline_run_id="direct",
    )
    table = list(csv.DictReader((tmp_path / "out" / "paper_table.csv").open()))
    ecgfounder = next(r for r in table if r["run_id"] == "ecgfounder")
    assert ecgfounder["view"] == "pn2021_all_zero_kept_refexcluded"
    assert ecgfounder["raw_views"] == "target_all_zero_kept_refexcluded"
    assert abs(float(ecgfounder["delta_macro_auprc_vs_baseline"]) - 0.06) < 1e-9


def test_export_paper_table_rejects_mixed_mapping(tmp_path: Path):
    rows = [
        _row("a", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8, mapping_hash="hash6"),
        _row("b", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.7, mapping_hash="hash5"),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    with pytest.raises(PaperTableError, match="mixed mapping"):
        export_paper_table(metrics, tmp_path / "out", view="pn2021_all_zero_kept_refexcluded")


def test_export_paper_table_rejects_missing_requested_center(tmp_path: Path):
    rows = [
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8),
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auprc", 0.5),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    with pytest.raises(PaperTableError, match="missing centers"):
        export_paper_table(
            metrics,
            tmp_path / "out",
            view="pn2021_all_zero_kept_refexcluded",
            centers=["ningbo", "georgia"],
            include_dataset_rows=False,
        )


def test_export_paper_table_rejects_missing_metric_for_requested_center(tmp_path: Path):
    rows = [
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8),
        _row("run", "pn2021_all_zero_kept_refexcluded", "georgia", "macro_auroc", 0.7),
        _row("run", "pn2021_all_zero_kept_refexcluded", "georgia", "macro_auprc", 0.4),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    with pytest.raises(PaperTableError, match="missing center metrics"):
        export_paper_table(
            metrics,
            tmp_path / "out",
            view="pn2021_all_zero_kept_refexcluded",
            centers=["ningbo", "georgia"],
            include_dataset_rows=False,
        )


def test_export_paper_table_rejects_missing_baseline(tmp_path: Path):
    rows = [
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auroc", 0.8),
        _row("run", "pn2021_all_zero_kept_refexcluded", "ningbo", "macro_auprc", 0.5),
    ]
    metrics = _write_metrics(tmp_path / "metrics_long.csv", rows)

    with pytest.raises(PaperTableError, match="baseline_run_id"):
        export_paper_table(
            metrics,
            tmp_path / "out",
            view="pn2021_all_zero_kept_refexcluded",
            baseline_run_id="direct",
            include_dataset_rows=False,
        )
