#!/usr/bin/env python3
"""Gate cand94 4-GPU launch on cand93 recovery, then summarize recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any


PROJECT_ROOT = Path("/home/linbinhao/ECG_adv_Gen")
PYTHON_BIN = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
LOCAL_CONFIG = "configs/local/linbinhao_server.example.yaml"
OUTPUT_ROOT = Path("/home/linbinhao/ECG_adv_data/runs")
REPORT_DIR = OUTPUT_ROOT / "pn2021c_sota_ablation_effnet_ecgfounder_20260621"
BASELINE_CSV = REPORT_DIR / "effnet_completed_recovery_snapshot_20260621_2252.csv"
WATCHDOG_DIR = REPORT_DIR / "watchdogs"

EVAL_NAME = "eval_pn2021_c_v7_refexcluded_stream_dual_model_10to15pp_v1_raw_first.json"
CORRUPTIONS = [
    "baseline_shift",
    "baseline_wander",
    "emg_noise",
    "powerline_noise",
    "random_leads_masking",
]

CENTERS = {
    "cpsc_2018": {
        "gpu": "0",
        "short": "cpsc",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_cpsc.yaml",
        "train_run_id": "cand94_cpsc_postcand93_4g_20260622_gpu0",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_cpsc.yaml",
        "eval_run_id": "cand94_cpsc_postcand93_4g_rawfirst_eval_20260622_gpu0",
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_cpsc",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g",
    },
    "chapman_shaoxing": {
        "gpu": "1",
        "short": "chapman",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_chapman_shaoxing.yaml",
        "train_run_id": "cand94_chapman_postcand93_4g_20260622_gpu1",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_chapman_shaoxing.yaml",
        "eval_run_id": "cand94_chapman_postcand93_4g_rawfirst_eval_20260622_gpu1",
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_chapman_shaoxing",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g",
    },
    "georgia": {
        "gpu": "2",
        "short": "georgia",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_georgia.yaml",
        "train_run_id": "cand94_georgia_postcand93_4g_20260622_gpu2",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_georgia.yaml",
        "eval_run_id": "cand94_georgia_postcand93_4g_rawfirst_eval_20260622_gpu2",
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_georgia",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g",
    },
    "ningbo": {
        "gpu": "3",
        "short": "ningbo",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_ningbo.yaml",
        "train_run_id": "cand94_ningbo_postcand93_4g_20260622_gpu3",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_ningbo.yaml",
        "eval_run_id": "cand94_ningbo_postcand93_4g_rawfirst_eval_20260622_gpu3",
        "run_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_ningbo",
        "method": "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g",
    },
}


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _as_float(value: Any) -> float:
    return float(value)


def _latest_cand93_summary(report_dir: Path) -> Path | None:
    summaries = sorted(
        report_dir.glob("cand93_recovery_summary_*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def _cand93_achieved(summary_path: Path, threshold_pp: float) -> bool | None:
    text = summary_path.read_text(encoding="utf-8")
    if "Goal gate status: ACHIEVED" in text:
        return True
    if "Goal gate status: NOT_ACHIEVED" in text:
        return False
    match = re.search(
        r"Four-center/operator mean recovery:\s*([+-]?\d+(?:\.\d+)?) pp AUROC / ([+-]?\d+(?:\.\d+)?) pp AUPRC",
        text,
    )
    if not match:
        return None
    return float(match.group(1)) >= threshold_pp or float(match.group(2)) >= threshold_pp


def _wait_for_cand93_gate(report_dir: Path, poll_seconds: int, threshold_pp: float) -> bool:
    while True:
        summary = _latest_cand93_summary(report_dir)
        if summary is None:
            _log("waiting for cand93 recovery summary")
            time.sleep(poll_seconds)
            continue
        achieved = _cand93_achieved(summary, threshold_pp)
        if achieved is None:
            _log(f"cand93 summary found but unparsable: {summary}")
            time.sleep(poll_seconds)
            continue
        _log(f"cand93 gate summary={summary} achieved={achieved}")
        return achieved


def _gpu_memory_mib() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    usage: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",", maxsplit=1)]
        usage[index] = int(used)
    return usage


def _wait_for_target_gpus(gpus: list[str], max_used_mib: int, poll_seconds: int) -> None:
    while True:
        usage = _gpu_memory_mib()
        busy = {gpu: usage.get(gpu, -1) for gpu in gpus if usage.get(gpu, 0) > max_used_mib}
        if not busy:
            _log(f"target GPUs free: {gpus}")
            subprocess.run(["nvidia-smi"], check=False)
            return
        _log(f"waiting for target GPUs to free below {max_used_mib} MiB: {busy}")
        time.sleep(poll_seconds)


def _run_launch(config: str, run_id: str, gpu: str, log_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "ECG_ADV_GEN_DATA_ROOT": "/home/linbinhao/ECG_adv_data",
            "TMPDIR": "/home/linbinhao/tmp_ecg",
            "XDG_CACHE_HOME": "/home/linbinhao/ECG_adv_data/cache",
        }
    )
    cmd = [
        str(PYTHON_BIN),
        "scripts/run_experiment.py",
        "--config",
        config,
        "--local-config",
        LOCAL_CONFIG,
        "--run-id",
        run_id,
        "--execute",
        "--write-plan",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_stamp()}] launch config={config} run_id={run_id} gpu={gpu}\n")
        log.flush()
        subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"[{_stamp()}] done config={config} run_id={run_id}\n")


def _run_center(center: str, spec: dict[str, str], errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand94_{spec['short']}_postcand93_4g_train_eval_20260622_gpu{spec['gpu']}.log"
    try:
        _run_launch(spec["train_config"], spec["train_run_id"], spec["gpu"], log_path)
        _run_launch(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log_path)
    except Exception as exc:  # pragma: no cover - long-running watchdog path
        errors.append(f"{center}: {exc}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_stamp()}] ERROR {center}: {exc}\n")


def _expected_jsons(output_root: Path) -> dict[str, Path]:
    return {
        center: output_root / spec["run_root"] / spec["eval_run_id"] / center / spec["method"] / EVAL_NAME
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
    missing = [
        (center, op)
        for center in CENTERS
        for op in CORRUPTIONS
        if (center, op) not in baseline
    ]
    if missing:
        raise ValueError(f"missing direct baseline rows: {missing}")
    return baseline


def _build_recovery_rows(output_root: Path, baseline_csv: Path) -> list[dict[str, Any]]:
    baseline = _load_direct_baseline(baseline_csv)
    rows: list[dict[str, Any]] = []
    for center, json_path in _expected_jsons(output_root).items():
        data = json.loads(json_path.read_text(encoding="utf-8"))
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
                    "cand": "cand94_4g",
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


def _write_recovery(rows: list[dict[str, Any]], report_dir: Path, threshold_pp: float) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = report_dir / f"cand94_4g_recovery_snapshot_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_auroc = _mean([_as_float(r["recovery_auroc_pp"]) for r in rows])
    mean_auprc = _mean([_as_float(r["recovery_auprc_pp"]) for r in rows])
    status = "ACHIEVED" if mean_auroc >= threshold_pp or mean_auprc >= threshold_pp else "NOT_ACHIEVED"
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

    summary_path = report_dir / f"cand94_4g_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand94 4G PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix, per-op raw corruption exposure, dual_model_10to15pp_v1 raw-first eval.",
                "Launch policy: started only if cand93 did not reach the recovery gate; GPU0-3 only.",
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


def _wait_for_eval_jsons(output_root: Path, poll_seconds: int) -> None:
    while True:
        missing = [path for path in _expected_jsons(output_root).values() if not path.exists()]
        if not missing:
            return
        _log(f"waiting for cand94 eval JSONs: {len(missing)} missing")
        for path in missing:
            _log(f"missing {path}")
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1000)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    args = parser.parse_args(argv)

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("cand94 post-cand93 4g watcher start")
    achieved = _wait_for_cand93_gate(REPORT_DIR, args.poll_seconds, args.goal_threshold_pp)
    if achieved:
        _log("cand93 already reached goal gate; cand94 4g launch skipped")
        return 0

    gpus = [spec["gpu"] for spec in CENTERS.values()]
    _wait_for_target_gpus(gpus, args.gpu_free_threshold_mib, args.poll_seconds)
    _log("launching cand94 4g train+eval workers")
    errors: list[str] = []
    threads = [
        threading.Thread(target=_run_center, args=(center, spec, errors), daemon=False)
        for center, spec in CENTERS.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        for err in errors:
            _log(f"worker failed: {err}")
        return 1

    _wait_for_eval_jsons(OUTPUT_ROOT, args.poll_seconds)
    rows = _build_recovery_rows(OUTPUT_ROOT, BASELINE_CSV)
    csv_path, summary_path = _write_recovery(rows, REPORT_DIR, args.goal_threshold_pp)
    _log(f"wrote {csv_path}")
    _log(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
