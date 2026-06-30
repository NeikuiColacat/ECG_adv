#!/usr/bin/env python3
"""Launch cand98 fixed-m=1.0 only if cand97 misses the recovery gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/linbinhao/ECG_adv_Gen")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.agent.watch_effnet_cand97_fixedm085_4g_20260622 as runner


CAND97_METHOD = runner.METHOD_NAME
CAND98_METHOD = "cand98_q8192_hull24_perop_fixedm100_c15_b20_ep45_4g"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _cand97_ready() -> bool:
    return all(runner._expected_eval_json(center).exists() for center in runner.CENTERS)


def _cand97_recovery() -> tuple[float, float]:
    rows = runner._build_recovery_rows()
    return (
        _mean([float(row["recovery_auroc_pp"]) for row in rows]),
        _mean([float(row["recovery_auprc_pp"]) for row in rows]),
    )


def _patch_runner_for_cand98() -> None:
    runner.METHOD_NAME = CAND98_METHOD
    runner.CENTERS = {
        "cpsc_2018": {
            "gpu": "0",
            "short": "cpsc",
            "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand98_q8192_hull24_perop_fixedm100_c15_b20_ep45_cpsc.yaml",
            "train_run_id": "cand98_cpsc_fixedm100_4g_20260622_gpu0",
            "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_cpsc.yaml",
            "eval_run_id": "cand98_cpsc_fixedm100_rawfirst_eval_20260622_gpu0",
            "eval_root": "pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_cpsc",
        },
        "chapman_shaoxing": {
            "gpu": "1",
            "short": "chapman",
            "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand98_q8192_hull24_perop_fixedm100_c15_b20_ep45_chapman_shaoxing.yaml",
            "train_run_id": "cand98_chapman_fixedm100_4g_20260622_gpu1",
            "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_chapman_shaoxing.yaml",
            "eval_run_id": "cand98_chapman_fixedm100_rawfirst_eval_20260622_gpu1",
            "eval_root": "pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_chapman_shaoxing",
        },
        "georgia": {
            "gpu": "2",
            "short": "georgia",
            "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand98_q8192_hull24_perop_fixedm100_c15_b20_ep45_georgia.yaml",
            "train_run_id": "cand98_georgia_fixedm100_4g_20260622_gpu2",
            "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_georgia.yaml",
            "eval_run_id": "cand98_georgia_fixedm100_rawfirst_eval_20260622_gpu2",
            "eval_root": "pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_georgia",
        },
        "ningbo": {
            "gpu": "3",
            "short": "ningbo",
            "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand98_q8192_hull24_perop_fixedm100_c15_b20_ep45_ningbo.yaml",
            "train_run_id": "cand98_ningbo_fixedm100_4g_20260622_gpu3",
            "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_ningbo.yaml",
            "eval_run_id": "cand98_ningbo_fixedm100_rawfirst_eval_20260622_gpu3",
            "eval_root": "pn2021c_effnet_dual3ch_cand98_fixedm100_rawfirst_ningbo",
        },
    }
    runner._build_recovery_rows = _build_recovery_rows_cand98  # type: ignore[assignment]
    runner._write_recovery = _write_recovery_cand98  # type: ignore[assignment]


def _build_recovery_rows_cand98() -> list[dict[str, Any]]:
    baseline = runner._load_direct_baseline(runner.BASELINE_CSV)
    rows: list[dict[str, Any]] = []
    for center in runner.CENTERS:
        json_path = runner._expected_eval_json(center)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        mapping = data.get("label_mapping") or {}
        for op in runner.CORRUPTIONS:
            rec = data["per_center"][center][op]["5"]
            direct = baseline[(center, op)]
            clean_auroc = float(rec["clean_macro_auroc"])
            clean_auprc = float(rec["clean_macro_auprc"])
            corrupted_auroc = float(rec["macro_auroc"])
            corrupted_auprc = float(rec["macro_auprc"])
            rows.append(
                {
                    "center": center,
                    "operator": op,
                    "cand": "cand98_fixedm100",
                    "json": str(json_path),
                    "mapping_version": mapping.get("version", ""),
                    "mapping_hash": mapping.get("hash", ""),
                    **direct,
                    "method_clean_auroc": clean_auroc,
                    "method_clean_auprc": clean_auprc,
                    "method_corrupted_auroc": corrupted_auroc,
                    "method_corrupted_auprc": corrupted_auprc,
                    "recovery_auroc_pp": (corrupted_auroc - direct["direct_corrupted_auroc"]) * 100.0,
                    "recovery_auprc_pp": (corrupted_auprc - direct["direct_corrupted_auprc"]) * 100.0,
                    "method_drop_auroc_pp": (clean_auroc - corrupted_auroc) * 100.0,
                    "method_drop_auprc_pp": (clean_auprc - corrupted_auprc) * 100.0,
                    "clean_delta_auroc_pp": (clean_auroc - direct["direct_clean_auroc"]) * 100.0,
                    "clean_delta_auprc_pp": (clean_auprc - direct["direct_clean_auprc"]) * 100.0,
                }
            )
    return rows


def _write_recovery_cand98(rows: list[dict[str, Any]], threshold_pp: float) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = runner.REPORT_DIR / f"cand98_fixedm100_recovery_snapshot_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_auroc = _mean([float(row["recovery_auroc_pp"]) for row in rows])
    mean_auprc = _mean([float(row["recovery_auprc_pp"]) for row in rows])
    status = "ACHIEVED" if mean_auroc >= threshold_pp and mean_auprc >= threshold_pp else "NOT_ACHIEVED"
    center_lines = []
    for center in runner.CENTERS:
        crows = [row for row in rows if row["center"] == center]
        center_lines.append(
            f"- {center}: recovery "
            f"{_mean([float(row['recovery_auroc_pp']) for row in crows]):.2f}/"
            f"{_mean([float(row['recovery_auprc_pp']) for row in crows]):.2f} pp"
        )
    op_lines = []
    for op in runner.CORRUPTIONS:
        orows = [row for row in rows if row["operator"] == op]
        op_lines.append(
            f"- {op}: recovery "
            f"{_mean([float(row['recovery_auroc_pp']) for row in orows]):.2f}/"
            f"{_mean([float(row['recovery_auprc_pp']) for row in orows]):.2f} pp"
        )

    summary_path = runner.REPORT_DIR / f"cand98_fixedm100_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand98 Fixed-M=1.0 PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix with fixed final AugMix mixture m=1.0.",
                "No stabilizer, preprocessing repair, operator oracle, heldout selector, or separate raw-supervised branch.",
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
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    parser.add_argument("--max-load-before-launch", type=float, default=220.0)
    parser.add_argument("--max-load-before-eval", type=float, default=0.0)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1000)
    args = parser.parse_args(argv)

    runner.WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    gate_log = runner.WATCHDOG_DIR / "cand98_fixedm100_after_cand97_gate_20260622.log"
    while not _cand97_ready():
        missing = [str(runner._expected_eval_json(center)) for center in runner.CENTERS if not runner._expected_eval_json(center).exists()]
        msg = f"[{runner._stamp()}] waiting cand97 gate: {len(missing)} eval JSONs missing\n"
        gate_log.write_text(gate_log.read_text(encoding="utf-8") + msg if gate_log.exists() else msg, encoding="utf-8")
        time.sleep(args.poll_seconds)

    mean_auroc, mean_auprc = _cand97_recovery()
    msg = (
        f"[{runner._stamp()}] cand97 recovery mean={mean_auroc:.2f}/{mean_auprc:.2f} pp; "
        f"threshold={args.goal_threshold_pp:.2f} pp\n"
    )
    gate_log.write_text(gate_log.read_text(encoding="utf-8") + msg if gate_log.exists() else msg, encoding="utf-8")
    if mean_auroc >= args.goal_threshold_pp and mean_auprc >= args.goal_threshold_pp:
        runner._log("cand97 reached both recovery thresholds; cand98 fallback will not launch")
        return 0

    runner._log("cand97 missed at least one recovery threshold; launching cand98 fixedm100 fallback")
    _patch_runner_for_cand98()
    return runner.main(
        [
            "--poll-seconds",
            str(args.poll_seconds),
            "--goal-threshold-pp",
            str(args.goal_threshold_pp),
            "--gpu-free-threshold-mib",
            str(args.gpu_free_threshold_mib),
            "--max-load-before-launch",
            str(args.max_load_before_launch),
            "--max-load-before-eval",
            str(args.max_load_before_eval),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
