from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ecg_adv_gen.evaluation import (
    PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
    PN2021_DROP_ALL_ZERO_REFEXCLUDED,
    PN2021C_ALL_ZERO_KEPT_CORRUPTED_REFEXCLUDED,
    PN2021C_DROP_ALL_ZERO_CORRUPTED_REFEXCLUDED,
)
from ecg_adv_gen.reporting import PaperTableError, export_metrics, export_paper_table


MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metric_values(path: Path, *, scope: str = "center_mean") -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(row for row in rows if row["scope"] == scope)
    return {
        metric: float(selected[metric])
        for metric in ("macro_auroc", "macro_auprc")
    }


def test_reporting_pipeline_keeps_four_all_zero_views_distinct(tmp_path: Path) -> None:
    metadata = {
        "run_id": "fixture-run",
        "mapping": {
            "mapping_version": MAPPING_VERSION,
            "mapping_hash": MAPPING_HASH,
        },
        "class_names": CLASS_NAMES,
    }
    pn2021 = _write_json(
        tmp_path / "eval_result_pn2021.json",
        {
            **metadata,
            "pn2021": {
                "avg_macro_auroc": 0.11,
                "avg_macro_auprc": 0.12,
                "avg_drop_all_zero_macro_auroc": 0.21,
                "avg_drop_all_zero_macro_auprc": 0.22,
                "per_center": {
                    "ningbo": {
                        "macro_auroc": 0.13,
                        "macro_auprc": 0.14,
                        "drop_all_zero_macro_auroc": 0.23,
                        "drop_all_zero_macro_auprc": 0.24,
                    }
                },
            },
        },
    )
    pn2021c = _write_json(
        tmp_path / "eval_pn2021_c_fixture.json",
        {
            **metadata,
            "per_center": {
                "ningbo": {
                    "powerline_noise": {
                        "5": {
                            "macro_auroc": 0.31,
                            "macro_auprc": 0.32,
                            "drop_all_zero_macro_auroc": 0.41,
                            "drop_all_zero_macro_auprc": 0.42,
                        }
                    }
                }
            },
        },
    )

    metrics_manifest = export_metrics([pn2021, pn2021c], tmp_path / "metrics")
    metrics_long = Path(metrics_manifest["metrics_long"])
    expected = {
        ("pn2021", PN2021_ALL_ZERO_KEPT_REFEXCLUDED): {
            "macro_auroc": 0.13,
            "macro_auprc": 0.14,
        },
        ("pn2021", PN2021_DROP_ALL_ZERO_REFEXCLUDED): {
            "macro_auroc": 0.23,
            "macro_auprc": 0.24,
        },
        ("pn2021c", PN2021C_ALL_ZERO_KEPT_CORRUPTED_REFEXCLUDED): {
            "macro_auroc": 0.31,
            "macro_auprc": 0.32,
        },
        ("pn2021c", PN2021C_DROP_ALL_ZERO_CORRUPTED_REFEXCLUDED): {
            "macro_auroc": 0.41,
            "macro_auprc": 0.42,
        },
    }
    pn2021_dataset_expected = {
        PN2021_ALL_ZERO_KEPT_REFEXCLUDED: {
            "macro_auroc": 0.11,
            "macro_auprc": 0.12,
        },
        PN2021_DROP_ALL_ZERO_REFEXCLUDED: {
            "macro_auroc": 0.21,
            "macro_auprc": 0.22,
        },
    }

    assert metrics_manifest["canonical_views"] == {
        view: 4 if dataset == "pn2021" else 2
        for (dataset, view) in expected
    }
    for (dataset, view), values in expected.items():
        manifest = export_paper_table(
            metrics_long,
            tmp_path / f"table-{view}",
            dataset=dataset,
            view=view,
        )
        assert manifest["view"] == view
        assert _metric_values(Path(manifest["paper_table"])) == values
        if dataset == "pn2021":
            assert _metric_values(
                Path(manifest["paper_table"]),
                scope="dataset",
            ) == pn2021_dataset_expected[view]


def test_paper_table_rejects_blank_view(tmp_path: Path) -> None:
    metrics_long = tmp_path / "metrics_long.csv"
    metrics_long.write_text(",".join(["run_id", "dataset", "view"]) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="view"):
        export_paper_table(metrics_long, tmp_path / "omitted")  # type: ignore[call-arg]
    with pytest.raises(PaperTableError, match="view must be explicitly specified"):
        export_paper_table(metrics_long, tmp_path / "table", view="")
