#!/usr/bin/env python3
"""Wait for cand93 PN2021-C evals and export recovery against direct K500."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CENTERS = {
    "chapman_shaoxing": {
        "run_root": "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_chapman_shaoxing",
        "run_id": "cand93_chapman_depth2_rawfirst_eval_20260622_gpu1",
        "method": "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal",
    },
    "cpsc_2018": {
        "run_root": "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_cpsc",
        "run_id": "cand93_cpsc_depth2_rawfirst_eval_20260622_gpu0",
        "method": "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal",
    },
    "georgia": {
        "run_root": "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_georgia",
        "run_id": "cand93_georgia_depth2_rawfirst_eval_20260622_gpu2",
        "method": "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal",
    },
    "ningbo": {
        "run_root": "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_ningbo",
        "run_id": "cand93_ningbo_depth2_rawfirst_eval_20260622_gpu3",
        "method": "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_float(value: Any) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _expected_jsons(output_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for center, spec in CENTERS.items():
        paths[center] = (
            output_root
            / spec["run_root"]
            / spec["run_id"]
            / center
            / spec["method"]
            / EVAL_NAME
        )
    return paths


def _load_direct_baseline(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["center"], row["operator"])
            values = {
                "direct_clean_auroc": _as_float(row["direct_clean_auroc"]),
                "direct_clean_auprc": _as_float(row["direct_clean_auprc"]),
                "direct_corrupted_auroc": _as_float(row["direct_corrupted_auroc"]),
                "direct_corrupted_auprc": _as_float(row["direct_corrupted_auprc"]),
            }
            if key in baseline and baseline[key] != values:
                raise ValueError(f"non-unique direct baseline for {key}: {baseline[key]} vs {values}")
            baseline[key] = values
    missing = [
        (center, op)
        for center in CENTERS
        for op in CORRUPTIONS
        if (center, op) not in baseline
    ]
    if missing:
        raise ValueError(f"missing direct baseline rows: {missing}")
    return baseline


def _build_rows(output_root: Path, baseline_csv: Path) -> list[dict[str, Any]]:
    baseline = _load_direct_baseline(baseline_csv)
    rows: list[dict[str, Any]] = []
    for center, json_path in _expected_jsons(output_root).items():
        data = _load_json(json_path)
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
                    "cand": "cand93",
                    "variant": CENTERS[center]["method"],
                    "json": str(json_path),
                    "mapping_version": mapping.get("version", ""),
                    "mapping_hash": mapping.get("hash", ""),
                    **direct,
                    "method_clean_auroc": method_clean_auroc,
                    "method_clean_auprc": method_clean_auprc,
                    "method_corrupted_auroc": method_corrupted_auroc,
                    "method_corrupted_auprc": method_corrupted_auprc,
                    "recovery_auroc_pp": (
                        method_corrupted_auroc - direct["direct_corrupted_auroc"]
                    )
                    * 100.0,
                    "recovery_auprc_pp": (
                        method_corrupted_auprc - direct["direct_corrupted_auprc"]
                    )
                    * 100.0,
                    "method_drop_auroc_pp": (
                        method_clean_auroc - method_corrupted_auroc
                    )
                    * 100.0,
                    "method_drop_auprc_pp": (
                        method_clean_auprc - method_corrupted_auprc
                    )
                    * 100.0,
                    "clean_delta_auroc_pp": (
                        method_clean_auroc - direct["direct_clean_auroc"]
                    )
                    * 100.0,
                    "clean_delta_auprc_pp": (
                        method_clean_auprc - direct["direct_clean_auprc"]
                    )
                    * 100.0,
                }
            )
    return rows


def _write_outputs(rows: list[dict[str, Any]], report_dir: Path, stamp: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"cand93_recovery_snapshot_{stamp}.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    op_lines = []
    for op in CORRUPTIONS:
        orows = [r for r in rows if r["operator"] == op]
        op_lines.append(
            f"- {op}: recovery "
            f"{_mean([_as_float(r['recovery_auroc_pp']) for r in orows]):.2f}/"
            f"{_mean([_as_float(r['recovery_auprc_pp']) for r in orows]):.2f} pp"
        )

    summary_path = report_dir / f"cand93_recovery_summary_{stamp}.md"
    status = "ACHIEVED" if mean_auroc >= 9.5 or mean_auprc >= 9.5 else "NOT_ACHIEVED"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand93 PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix, dual_model_10to15pp_v1 raw-first eval.",
                "",
                "## By Center",
                *center_lines,
                "",
                "## By Operator",
                *op_lines,
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
            print("timeout waiting for cand93 eval JSONs")
            for item in missing:
                print(f"missing {item}")
            return 2
        print(f"waiting for {len(missing)} cand93 eval JSONs")
        for item in missing:
            print(f"missing {item}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
