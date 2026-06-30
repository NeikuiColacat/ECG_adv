#!/usr/bin/env python3
"""Run cand97 fixed-m AugMix on GPU0-GPU3 and summarize PN2021-C recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime
from typing import Any


PROJECT_ROOT = Path("/home/linbinhao/ECG_adv_Gen")
PYTHON_BIN = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
LOCAL_CONFIG = "configs/local/linbinhao_server.example.yaml"
OUTPUT_ROOT = Path("/home/linbinhao/ECG_adv_data/runs")
REPORT_DIR = OUTPUT_ROOT / "pn2021c_sota_ablation_effnet_ecgfounder_20260621"
WATCHDOG_DIR = REPORT_DIR / "watchdogs"
BASELINE_CSV = REPORT_DIR / "effnet_completed_recovery_snapshot_20260621_2252.csv"
EVAL_NAME = "eval_pn2021_c_v7_refexcluded_stream_dual_model_10to15pp_v1_raw_first.json"
METHOD_NAME = "cand97_q8192_hull24_perop_fixedm085_c15_b20_ep45_4g"
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
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand97_q8192_hull24_perop_fixedm085_c15_b20_ep45_cpsc.yaml",
        "train_run_id": "cand97_cpsc_fixedm085_4g_20260622_gpu0",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_cpsc.yaml",
        "eval_run_id": "cand97_cpsc_fixedm085_rawfirst_eval_20260622_gpu0",
        "eval_root": "pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_cpsc",
    },
    "chapman_shaoxing": {
        "gpu": "1",
        "short": "chapman",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand97_q8192_hull24_perop_fixedm085_c15_b20_ep45_chapman_shaoxing.yaml",
        "train_run_id": "cand97_chapman_fixedm085_4g_20260622_gpu1",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_chapman_shaoxing.yaml",
        "eval_run_id": "cand97_chapman_fixedm085_rawfirst_eval_20260622_gpu1",
        "eval_root": "pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_chapman_shaoxing",
    },
    "georgia": {
        "gpu": "2",
        "short": "georgia",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand97_q8192_hull24_perop_fixedm085_c15_b20_ep45_georgia.yaml",
        "train_run_id": "cand97_georgia_fixedm085_4g_20260622_gpu2",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_georgia.yaml",
        "eval_run_id": "cand97_georgia_fixedm085_rawfirst_eval_20260622_gpu2",
        "eval_root": "pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_georgia",
    },
    "ningbo": {
        "gpu": "3",
        "short": "ningbo",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand97_q8192_hull24_perop_fixedm085_c15_b20_ep45_ningbo.yaml",
        "train_run_id": "cand97_ningbo_fixedm085_4g_20260622_gpu3",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_ningbo.yaml",
        "eval_run_id": "cand97_ningbo_fixedm085_rawfirst_eval_20260622_gpu3",
        "eval_root": "pn2021c_effnet_dual3ch_cand97_fixedm085_rawfirst_ningbo",
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
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    usage: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",", maxsplit=1)]
        usage[index] = int(used)
    return usage


def _host_load_1min() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def _run_id_active(run_id: str) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return run_id in result.stdout


def _wait_for_slot(gpu: str, max_used_mib: int, max_load: float, poll_seconds: int) -> None:
    while True:
        used = _gpu_memory_mib().get(gpu, 0)
        load = _host_load_1min()
        if used <= max_used_mib and (max_load <= 0 or load <= max_load):
            _log(f"GPU{gpu} ready: used={used} MiB load1={load:.2f}")
            subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, check=False)
            return
        _log(f"waiting GPU{gpu}: used={used} MiB load1={load:.2f}")
        time.sleep(poll_seconds)


def _expected_eval_json(center: str) -> Path:
    spec = CENTERS[center]
    return OUTPUT_ROOT / spec["eval_root"] / spec["eval_run_id"] / center / METHOD_NAME / EVAL_NAME


def _run_launch(config: str, run_id: str, gpu: str, log_path: Path) -> None:
    if _run_id_active(run_id):
        _log(f"run already active, waiting: {run_id}")
        while _run_id_active(run_id):
            time.sleep(60)
        return
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "ECG_ADV_GEN_DATA_ROOT": "/home/linbinhao/ECG_adv_data",
            "TMPDIR": "/home/linbinhao/tmp_ecg",
            "XDG_CACHE_HOME": "/home/linbinhao/ECG_adv_data/cache",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
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
        log.write(f"[{_stamp()}] launch run_id={run_id} config={config} gpu={gpu}\n")
        log.flush()
        subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"[{_stamp()}] finished run_id={run_id}\n")


def _run_center(center: str, spec: dict[str, str], args: argparse.Namespace, errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand97_{spec['short']}_fixedm085_train_eval_20260622_gpu{spec['gpu']}.log"
    try:
        final_json = _expected_eval_json(center)
        if final_json.exists():
            _log(f"{center}: cand97 eval exists; skip")
            return
        _wait_for_slot(spec["gpu"], args.gpu_free_threshold_mib, args.max_load_before_launch, args.poll_seconds)
        _run_launch(spec["train_config"], spec["train_run_id"], spec["gpu"], log_path)
        if final_json.exists():
            _log(f"{center}: cand97 eval exists after train step; skip eval")
            return
        _wait_for_slot(spec["gpu"], args.gpu_free_threshold_mib, args.max_load_before_eval, args.poll_seconds)
        _run_launch(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log_path)
    except Exception as exc:  # pragma: no cover - operational watchdog path
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
    for center in CENTERS:
        json_path = _expected_eval_json(center)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        mapping = data.get("label_mapping") or {}
        for op in CORRUPTIONS:
            rec = data["per_center"][center][op]["5"]
            direct = baseline[(center, op)]
            clean_auroc = _as_float(rec["clean_macro_auroc"])
            clean_auprc = _as_float(rec["clean_macro_auprc"])
            corrupted_auroc = _as_float(rec["macro_auroc"])
            corrupted_auprc = _as_float(rec["macro_auprc"])
            rows.append(
                {
                    "center": center,
                    "operator": op,
                    "cand": "cand97_fixedm085",
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


def _write_recovery(rows: list[dict[str, Any]], threshold_pp: float) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = REPORT_DIR / f"cand97_fixedm085_recovery_snapshot_{stamp}.csv"
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

    summary_path = REPORT_DIR / f"cand97_fixedm085_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand97 Fixed-M PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix with fixed final AugMix mixture m=0.85.",
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


def _wait_for_all_eval_jsons(poll_seconds: int) -> None:
    while True:
        missing = [path for center in CENTERS for path in [_expected_eval_json(center)] if not path.exists()]
        if not missing:
            return
        _log(f"waiting for cand97 eval JSONs: {len(missing)} missing")
        for path in missing:
            _log(f"missing {path}")
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1000)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    parser.add_argument("--max-load-before-launch", type=float, default=220.0)
    parser.add_argument("--max-load-before-eval", type=float, default=0.0)
    args = parser.parse_args(argv)

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("cand97 fixedm085 four-GPU watcher start")
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
            _log(f"ERROR {err}")
        return 2
    _wait_for_all_eval_jsons(args.poll_seconds)
    rows = _build_recovery_rows()
    csv_path, summary_path = _write_recovery(rows, args.goal_threshold_pp)
    _log(f"wrote {csv_path}")
    _log(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
