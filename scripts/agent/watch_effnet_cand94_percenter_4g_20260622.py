#!/usr/bin/env python3
"""Per-center cand94 scheduler after cand93 raw-first eval readiness.

This is intentionally a scheduler only. It does not change the method:
EfficientNet1DV2 VAE-LH online AT + locked three-chain AugMix, then raw-first
dual_model_10to15pp_v1 PN2021-C evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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

CENTERS: dict[str, dict[str, str]] = {
    "cpsc_2018": {
        "gpu": "0",
        "short": "cpsc",
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_cpsc"
            / "cand93_cpsc_depth2_rawfirst_eval_20260622_gpu0"
            / "cpsc_2018"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
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
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_chapman_shaoxing"
            / "cand93_chapman_depth2_rawfirst_eval_20260622_gpu1"
            / "chapman_shaoxing"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
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
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_georgia"
            / "cand93_georgia_depth2_rawfirst_eval_20260622_gpu2"
            / "georgia"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
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
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_ningbo"
            / "cand93_ningbo_depth2_rawfirst_eval_20260622_gpu3"
            / "ningbo"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
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


def _gpu_memory_mib() -> dict[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
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


def _wait_for_gpu(gpu: str, max_used_mib: int, poll_seconds: int) -> None:
    while True:
        used = _gpu_memory_mib().get(gpu, 0)
        if used <= max_used_mib:
            _log(f"GPU{gpu} free enough: {used} MiB")
            subprocess.run(["nvidia-smi"], check=False)
            return
        _log(f"waiting GPU{gpu} below {max_used_mib} MiB; now {used} MiB")
        time.sleep(poll_seconds)


def _wait_for_file(path: Path, poll_seconds: int) -> None:
    while not path.exists():
        _log(f"waiting prerequisite JSON: {path}")
        time.sleep(poll_seconds)
    json.loads(path.read_text(encoding="utf-8"))
    _log(f"prerequisite JSON ready: {path}")


def _expected_eval_json(center: str) -> Path:
    spec = CENTERS[center]
    return (
        OUTPUT_ROOT
        / spec["run_root"]
        / spec["eval_run_id"]
        / center
        / spec["method"]
        / EVAL_NAME
    )


def _run_id_active(run_id: str) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return run_id in result.stdout


def _run_launch(config: str, run_id: str, gpu: str, log_path: Path) -> None:
    if _run_id_active(run_id):
        _log(f"run already active; skip duplicate launch: {run_id}")
        return
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


def _run_center(center: str, spec: dict[str, str], args: argparse.Namespace, errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand94_{spec['short']}_percenter_4g_train_eval_20260622_gpu{spec['gpu']}.log"
    try:
        final_json = _expected_eval_json(center)
        if final_json.exists():
            _log(f"{center}: cand94 eval already exists; skip train/eval")
            return
        _wait_for_file(Path(spec["cand93_eval_json"]), args.poll_seconds)
        _wait_for_gpu(spec["gpu"], args.gpu_free_threshold_mib, args.poll_seconds)
        _run_launch(spec["train_config"], spec["train_run_id"], spec["gpu"], log_path)
        if final_json.exists():
            _log(f"{center}: cand94 eval exists after train step; skip eval")
            return
        _wait_for_gpu(spec["gpu"], args.gpu_free_threshold_mib, args.poll_seconds)
        _run_launch(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log_path)
    except Exception as exc:  # pragma: no cover - long-running watchdog path
        errors.append(f"{center}: {exc}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_stamp()}] ERROR {center}: {exc}\n")


def _load_direct_baseline(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            baseline[(row["center"], row["operator"])] = {
                "direct_clean_auroc": _as_float(row["direct_clean_auroc"]),
                "direct_clean_auprc": _as_float(row["direct_clean_auprc"]),
                "direct_corrupted_auroc": _as_float(row["direct_corrupted_auroc"]),
                "direct_corrupted_auprc": _as_float(row["direct_corrupted_auprc"]),
            }
    return baseline


def _build_recovery_rows() -> list[dict[str, Any]]:
    baseline = _load_direct_baseline(BASELINE_CSV)
    rows: list[dict[str, Any]] = []
    for center, spec in CENTERS.items():
        json_path = _expected_eval_json(center)
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
                    "cand": "cand94_4g_percenter",
                    "variant": spec["method"],
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


def _write_recovery(rows: list[dict[str, Any]], threshold_pp: float) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = REPORT_DIR / f"cand94_4g_percenter_recovery_snapshot_{stamp}.csv"
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

    summary_path = REPORT_DIR / f"cand94_4g_percenter_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand94 4G Per-Center PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix, per-op raw corruption exposure, dual_model_10to15pp_v1 raw-first eval.",
                "Launch policy: each center started after its cand93 raw-first eval JSON existed and its assigned GPU was free; GPU0-3 only.",
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


def _wait_for_all_eval_jsons(poll_seconds: int) -> None:
    while True:
        missing = [path for center in CENTERS for path in [_expected_eval_json(center)] if not path.exists()]
        if not missing:
            return
        _log(f"waiting for cand94 eval JSONs: {len(missing)} missing")
        for path in missing:
            _log(f"missing {path}")
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1000)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    args = parser.parse_args(argv)

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("cand94 per-center 4g watcher start")
    errors: list[str] = []
    threads = [
        threading.Thread(target=_run_center, args=(center, spec, args, errors), daemon=False)
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

    _wait_for_all_eval_jsons(args.poll_seconds)
    rows = _build_recovery_rows()
    csv_path, summary_path = _write_recovery(rows, args.goal_threshold_pp)
    _log(f"wrote {csv_path}")
    _log(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
