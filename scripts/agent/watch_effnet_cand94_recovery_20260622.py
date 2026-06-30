#!/usr/bin/env python3
"""Wait for cand94 PN2021-C evals and export recovery against direct K500."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CENTERS = {
    "cpsc_2018": {
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_rawfirst_cpsc",
        "run_id": "cand94_cpsc_postcand93_rawfirst_eval_20260622_gpu0",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_goal",
    },
    "chapman_shaoxing": {
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_rawfirst_chapman_shaoxing",
        "run_id": "cand94_chapman_postcand93_rawfirst_eval_20260622_gpu1",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_goal",
    },
    "georgia": {
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_rawfirst_georgia",
        "run_id": "cand94_georgia_postcand93_rawfirst_eval_20260622_gpu2",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_goal",
    },
    "ningbo": {
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_rawfirst_ningbo",
        "run_id": "cand94_ningbo_postcand93_rawfirst_eval_20260622_gpu3",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_goal",
    },
}

CORRUPTIONS = [
    "baseline_shift",
    "baseline_wander",
    "emg_noise",
    "powerline_noise",
    "random_leads_masking",
]

EVAL_NAME = "eval_pn2021_c_v7_refexcluded_stream_dual_model_10to15pp_v1_raw_first.json"


def _as_float(value: Any) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _expected_jsons(output_root: Path) -> dict[str, Path]:
    return {
        center: output_root / spec["run_root"] / spec["run_id"] / center / spec["method"] / EVAL_NAME
        for center, spec in CENTERS.items()
    }


def _load_direct_baseline(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["center"], row["operator"])
            baseline[key] = {
                "direct_clean_auroc": _as_float(row["direct_clean_auroc"]),
                "direct_clean_auprc": _as_float(row["direct_clean_auprc"]),
                "direct_corrupted_auroc": _as_float(row["direct_corrupted_auroc"]),
                "direct_corrupted_auprc": _as_float(row["direct_corrupted_auprc"]),
            }
    return baseline


def _build_rows(output_root: Path, baseline_csv: Path) -> list[dict[str, Any]]:
    baseline = _load_direct_baseline(baseline_csv)
    rows: list[dict[str, Any]] = []
    for center, json_path in _expected_jsons(output_root).items():
        data = json.loads(json_path.read_text())
        mapping = data.get("label_mapping") or {}
        for op in CORRUPTIONS:
            rec = data["per_center"][center][op]["5"]
            direct = baseline[(center, op)]
            method_clean_auroc = _as_float(rec["clean_macro_auroc"])
            method_clean_auprc = _as_float(rec["clean_macro_auprc"])
            method_corrupted_auroc = _as_float(rec["macro_auroc"])
            method_corrupted_auprc = _as_float(rec["macro_auprc"])
            rows.append(
                {
                    "center": center,
                    "operator": op,
                    "cand": "cand94",
                    "variant": CENTERS[center]["method"],
                    "json": str(json_path),
                    "mapping_version": mapping.get("version", ""),
                    "mapping_hash": mapping.get("hash", ""),
                    **direct,
                    "method_clean_auroc": method_clean_auroc,
                    "method_clean_auprc": method_clean_auprc,
                    "method_corrupted_auroc": method_corrupted_auroc,
                    "method_corrupted_auprc": method_corrupted_auprc,
                    "recovery_auroc_pp": (method_corrupted_auroc - direct["direct_corrupted_auroc"]) * 100.0,
                    "recovery_auprc_pp": (method_corrupted_auprc - direct["direct_corrupted_auprc"]) * 100.0,
                    "method_drop_auroc_pp": (method_clean_auroc - method_corrupted_auroc) * 100.0,
                    "method_drop_auprc_pp": (method_clean_auprc - method_corrupted_auprc) * 100.0,
                    "clean_delta_auroc_pp": (method_clean_auroc - direct["direct_clean_auroc"]) * 100.0,
                    "clean_delta_auprc_pp": (method_clean_auprc - direct["direct_clean_auprc"]) * 100.0,
                }
            )
    return rows


def _write_outputs(rows: list[dict[str, Any]], report_dir: Path, stamp: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"cand94_perop_recovery_snapshot_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_auroc = _mean([_as_float(r["recovery_auroc_pp"]) for r in rows])
    mean_auprc = _mean([_as_float(r["recovery_auprc_pp"]) for r in rows])
    center_lines = []
    for center in CENTERS:
        crows = [r for r in rows if r["center"] == center]
        center_lines.append(
            f"- {center}: recovery "
            f"{_mean([_as_float(r['recovery_auroc_pp']) for r in crows]):.2f}/"
            f"{_mean([_as_float(r['recovery_auprc_pp']) for r in crows]):.2f} pp"
        )
    summary_path = report_dir / f"cand94_perop_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand94 Per-Op PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix, per-op raw corruption exposure, dual_model_10to15pp_v1 raw-first eval.",
                "",
                "## By Center",
                *center_lines,
                "",
                f"CSV: {csv_path}",
                "",
            ]
        )
    )
    return csv_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/home/linbinhao/ECG_adv_data/runs"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("/home/linbinhao/ECG_adv_data/runs/pn2021c_sota_ablation_effnet_ecgfounder_20260621"),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "/home/linbinhao/ECG_adv_data/runs/pn2021c_sota_ablation_effnet_ecgfounder_20260621/"
            "effnet_completed_recovery_snapshot_20260621_2252.csv"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_seconds
    expected = _expected_jsons(args.output_root)
    while True:
        missing = [str(path) for path in expected.values() if not path.exists()]
        if not missing:
            rows = _build_rows(args.output_root, args.baseline_csv)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_path, summary_path = _write_outputs(rows, args.report_dir, stamp)
            print(f"wrote {csv_path}")
            print(f"wrote {summary_path}")
            return 0
        if time.time() >= deadline:
            print("timeout waiting for cand94 eval JSONs")
            for item in missing:
                print(f"missing {item}")
            return 2
        print(f"waiting for {len(missing)} cand94 eval JSONs")
        for item in missing:
            print(f"missing {item}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
