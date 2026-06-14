import csv
import json
from pathlib import Path

from scripts.triple_labels.export_pn2021c_recovery_report import (
    DEFAULT_EVAL_FILENAME,
    export_report,
)


def _record(
    *,
    center: str,
    clean_auroc: float,
    clean_auprc: float,
    corrupt_auroc: float,
    corrupt_auprc: float,
    clean_daz_auroc: float,
    clean_daz_auprc: float,
    corrupt_daz_auroc: float,
    corrupt_daz_auprc: float,
) -> dict:
    return {
        "n_records": 100,
        "clean_metric_source": "same_filtered_subset",
        "clean_macro_auroc": clean_auroc,
        "clean_macro_auprc": clean_auprc,
        "macro_auroc": corrupt_auroc,
        "macro_auprc": corrupt_auprc,
        "auroc_drop_vs_clean": clean_auroc - corrupt_auroc,
        "auprc_drop_vs_clean": clean_auprc - corrupt_auprc,
        "drop_all_zero_macro_auroc": corrupt_daz_auroc,
        "drop_all_zero_macro_auprc": corrupt_daz_auprc,
        "metadata": {
            "pn2021": {
                "per_center": {
                    center: {
                        "drop_all_zero_macro_auroc": clean_daz_auroc,
                        "drop_all_zero_macro_auprc": clean_daz_auprc,
                    }
                }
            }
        },
    }


def _write_eval(path: Path, center: str, record: dict) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "label_mapping": {
            "version": "v7_super5_sjr_rgq_review_20260528",
            "hash": "555ec85d5b51",
        },
        "per_center": {
            center: {
                "baseline_wander": {
                    "5": record,
                }
            }
        },
    }))


def test_pn2021c_recovery_report_exports_macro_and_drop_all_zero(tmp_path: Path):
    run_root = tmp_path / "run"
    center = "ningbo"
    _write_eval(
        run_root / center / "vae_noaug" / DEFAULT_EVAL_FILENAME,
        center,
        _record(
            center=center,
            clean_auroc=0.90,
            clean_auprc=0.70,
            corrupt_auroc=0.70,
            corrupt_auprc=0.50,
            clean_daz_auroc=0.95,
            clean_daz_auprc=0.80,
            corrupt_daz_auroc=0.80,
            corrupt_daz_auprc=0.60,
        ),
    )
    _write_eval(
        run_root / center / "vae_lhat" / DEFAULT_EVAL_FILENAME,
        center,
        _record(
            center=center,
            clean_auroc=0.91,
            clean_auprc=0.72,
            corrupt_auroc=0.75,
            corrupt_auprc=0.56,
            clean_daz_auroc=0.96,
            clean_daz_auprc=0.82,
            corrupt_daz_auroc=0.86,
            corrupt_daz_auprc=0.68,
        ),
    )

    out = tmp_path / "report"
    export_report(run_root=run_root, output_dir=out)

    rows = list(csv.DictReader((out / "pn2021c_recovery_by_center_corruption.csv").open()))
    macro = next(row for row in rows if row["view"] == "macro")
    daz = next(row for row in rows if row["view"] == "drop_all_zero")

    assert float(macro["candidate_minus_baseline_corrupt_auroc"]) == 0.05
    assert float(macro["candidate_minus_baseline_corrupt_auprc"]) == 0.06
    assert round(float(macro["candidate_corrupt_auroc_recovery_fraction"]), 6) == 0.25
    assert round(float(macro["candidate_drop_reduction_auroc"]), 6) == 0.04
    assert float(daz["baseline_clean_auroc"]) == 0.95
    assert float(daz["candidate_clean_auprc"]) == 0.82
    assert round(float(daz["candidate_corrupt_auprc_recovery_fraction"]), 6) == 0.4

    summary = (out / "pn2021c_recovery_summary.md").read_text()
    assert "meets the +6.00 pp corrupted macro AUPRC recovery target" in summary
    assert "negative recovery result" not in summary
