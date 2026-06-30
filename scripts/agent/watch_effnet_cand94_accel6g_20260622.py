#!/usr/bin/env python3
"""Robust cand94 scheduler after the temporary over-4-GPU rollback.

This only changes scheduling:
- keep the already-running CPSC cand94 train on GPU0 and evaluate it after finish;
- launch Chapman/Georgia/Ningbo only after each center's cand93 raw-first eval
  JSON is complete and the corresponding GPU0-3 slot is free.

The training method remains EfficientNet1DV2 VAE-LH online AT + locked
three-chain AugMix.
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

COMMON_TRAIN_FAMILY_PREFIX = (
    "effnet_vae_lhat_augmix_threechain_dualmodel_cand94_"
    "q8192_hull24_perop_maskpower_c15_b20_ep45"
)
METHOD_4G = "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g"

CENTERS: dict[str, dict[str, str]] = {
    "cpsc_2018": {
        "gpu": "0",
        "short": "cpsc",
        "mode": "existing_or_launch",
        "train_config": f"configs/experiments/{COMMON_TRAIN_FAMILY_PREFIX}_cpsc.yaml",
        "train_run_id": "cand94_cpsc_postcand93_4g_20260622_gpu0",
        "train_family": f"{COMMON_TRAIN_FAMILY_PREFIX}_cpsc",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_cpsc.yaml",
        "eval_run_id": "cand94_cpsc_postcand93_4g_rawfirst_eval_20260622_gpu0",
        "eval_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_cpsc",
        "method": METHOD_4G,
    },
    "chapman_shaoxing": {
        "gpu": "1",
        "short": "chapman",
        "mode": "wait_cand93_then_launch",
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_chapman_shaoxing"
            / "cand93_chapman_depth2_rawfirst_eval_20260622_gpu1"
            / "chapman_shaoxing"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
        "train_config": f"configs/experiments/{COMMON_TRAIN_FAMILY_PREFIX}_chapman_shaoxing.yaml",
        "train_run_id": "cand94_chapman_postcand93_4g_20260622_gpu1",
        "train_family": f"{COMMON_TRAIN_FAMILY_PREFIX}_chapman_shaoxing",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_chapman_shaoxing.yaml",
        "eval_run_id": "cand94_chapman_postcand93_4g_rawfirst_eval_20260622_gpu1",
        "eval_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_chapman_shaoxing",
        "method": METHOD_4G,
    },
    "georgia": {
        "gpu": "2",
        "short": "georgia",
        "mode": "wait_cand93_then_launch",
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_georgia"
            / "cand93_georgia_depth2_rawfirst_eval_20260622_gpu2"
            / "georgia"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
        "train_config": f"configs/experiments/{COMMON_TRAIN_FAMILY_PREFIX}_georgia.yaml",
        "train_run_id": "cand94_georgia_postcand93_4g_20260622_gpu2",
        "train_family": f"{COMMON_TRAIN_FAMILY_PREFIX}_georgia",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_georgia.yaml",
        "eval_run_id": "cand94_georgia_postcand93_4g_rawfirst_eval_20260622_gpu2",
        "eval_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_georgia",
        "method": METHOD_4G,
    },
    "ningbo": {
        "gpu": "3",
        "short": "ningbo",
        "mode": "wait_cand93_then_launch",
        "cand93_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand93_goal_rawfirst_ningbo"
            / "cand93_ningbo_depth2_rawfirst_eval_20260622_gpu3"
            / "ningbo"
            / "cand93_q8192_hull24_combo_depth2_c12_b24_ep45_goal"
            / EVAL_NAME
        ),
        "train_config": f"configs/experiments/{COMMON_TRAIN_FAMILY_PREFIX}_ningbo.yaml",
        "train_run_id": "cand94_ningbo_postcand93_4g_20260622_gpu3",
        "train_family": f"{COMMON_TRAIN_FAMILY_PREFIX}_ningbo",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_ningbo.yaml",
        "eval_run_id": "cand94_ningbo_postcand93_4g_rawfirst_eval_20260622_gpu3",
        "eval_root": "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_ningbo",
        "method": METHOD_4G,
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


def _host_load_1min() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def _wait_for_host_load(max_load: float, poll_seconds: int, reason: str) -> None:
    if max_load <= 0:
        return
    while True:
        load = _host_load_1min()
        if load <= max_load:
            _log(f"host load ok for {reason}: load1={load:.2f} <= {max_load:.2f}")
            return
        _log(f"waiting host load for {reason}: load1={load:.2f} > {max_load:.2f}")
        time.sleep(poll_seconds)


def _wait_for_gpu(gpu: str, max_used_mib: int, poll_seconds: int) -> None:
    while True:
        used = _gpu_memory_mib().get(gpu, 0)
        if used <= max_used_mib:
            _log(f"GPU{gpu} free enough: {used} MiB")
            subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, check=False)
            return
        _log(f"waiting GPU{gpu} below {max_used_mib} MiB; now {used} MiB")
        time.sleep(poll_seconds)


def _wait_for_json(path: Path, poll_seconds: int) -> None:
    while not path.exists():
        _log(f"waiting prerequisite JSON: {path}")
        time.sleep(poll_seconds)
    json.loads(path.read_text(encoding="utf-8"))
    _log(f"prerequisite JSON ready: {path}")


def _run_id_active(run_id: str) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return run_id in result.stdout


def _train_base(spec: dict[str, str]) -> Path:
    return OUTPUT_ROOT / spec["train_family"] / spec["train_run_id"]


def _latest_checkpoint(spec: dict[str, str]) -> Path | None:
    base = _train_base(spec)
    matches = sorted(base.glob("*/last_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _expected_eval_json(center: str) -> Path:
    spec = CENTERS[center]
    return OUTPUT_ROOT / spec["eval_root"] / spec["eval_run_id"] / center / spec["method"] / EVAL_NAME


def _run_launch(config: str, run_id: str, gpu: str, log_path: Path) -> None:
    if _run_id_active(run_id):
        _log(f"run already active; wait instead of duplicate launch: {run_id}")
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
        log.write(f"[{_stamp()}] launch config={config} run_id={run_id} gpu={gpu}\n")
        log.flush()
        subprocess.run(["nvidia-smi"], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"[{_stamp()}] done config={config} run_id={run_id}\n")


def _ensure_train(center: str, spec: dict[str, str], args: argparse.Namespace, log_path: Path) -> None:
    mode = spec["mode"]
    if mode == "wait_cand93_then_launch":
        _wait_for_json(Path(spec["cand93_eval_json"]), args.poll_seconds)

    if _run_id_active(spec["train_run_id"]):
        _log(f"{center}: train active, waiting: {spec['train_run_id']}")
        while _run_id_active(spec["train_run_id"]):
            time.sleep(args.poll_seconds)
        ckpt = _latest_checkpoint(spec)
        if ckpt is None:
            raise RuntimeError(f"{center}: train ended but last_model.pt not found under {_train_base(spec)}")
        _log(f"{center}: train completed: {ckpt}")
        return

    ckpt = _latest_checkpoint(spec)
    if ckpt is not None:
        _log(f"{center}: train checkpoint already exists and no active run remains: {ckpt}")
        return

    if mode == "existing_or_launch" and _train_base(spec).exists():
        raise RuntimeError(f"{center}: partial train dir exists but no active run or last_model.pt: {_train_base(spec)}")

    _wait_for_host_load(args.max_load_before_launch, args.poll_seconds, f"{center} train")
    _wait_for_gpu(spec["gpu"], args.gpu_free_threshold_mib, args.poll_seconds)
    _run_launch(spec["train_config"], spec["train_run_id"], spec["gpu"], log_path)
    ckpt = _latest_checkpoint(spec)
    if ckpt is None:
        raise RuntimeError(f"{center}: train launch finished but last_model.pt not found under {_train_base(spec)}")
    _log(f"{center}: train completed: {ckpt}")


def _run_center(center: str, spec: dict[str, str], args: argparse.Namespace, errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand94_{spec['short']}_robust4g_train_eval_20260622_gpu{spec['gpu']}.log"
    try:
        final_json = _expected_eval_json(center)
        if final_json.exists():
            _log(f"{center}: eval already exists; skip")
            return
        _ensure_train(center, spec, args, log_path)
        if final_json.exists():
            _log(f"{center}: eval exists after train wait; skip")
            return
        _wait_for_host_load(args.max_load_before_eval, args.poll_seconds, f"{center} eval")
        _wait_for_gpu(spec["gpu"], args.gpu_free_threshold_mib, args.poll_seconds)
        _run_launch(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log_path)
        if not final_json.exists():
            raise RuntimeError(f"{center}: eval launch finished but JSON missing: {final_json}")
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
                    "cand": "cand94_robust4g",
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
    csv_path = REPORT_DIR / f"cand94_robust4g_recovery_snapshot_{stamp}.csv"
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

    summary_path = REPORT_DIR / f"cand94_robust4g_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Cand94 Robust4G PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Method: VAE-LH online AT + locked three-chain AugMix, per-op raw corruption exposure, dual_model_10to15pp_v1 raw-first eval.",
                "Launch policy: max-4-GPU goal execution. CPSC keeps GPU0; Chapman/Georgia/Ningbo use GPU1/GPU2/GPU3 after cand93 prerequisites. The temporary GPU4/GPU7 queue is disabled.",
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
    parser.add_argument("--gpu-free-threshold-mib", type=int, default=1000)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    parser.add_argument(
        "--max-load-before-launch",
        type=float,
        default=120.0,
        help="Wait until /proc/loadavg 1-minute load is at or below this value before launching train/eval. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-load-before-eval",
        type=float,
        default=0.0,
        help="Separate load gate for post-training PN2021-C eval. Use <=0 to launch as soon as the assigned GPU is free.",
    )
    args = parser.parse_args(argv)

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("cand94 robust4g watcher start")
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

    rows = _build_recovery_rows()
    csv_path, summary_path = _write_recovery(rows, args.goal_threshold_pp)
    _log(f"wrote {csv_path}")
    _log(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
