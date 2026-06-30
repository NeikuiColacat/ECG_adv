#!/usr/bin/env python3
"""Wait for cand94, then run cand96 only if the +10 pp recovery gate is missed.

This watcher is intentionally max-4-GPU only. It uses GPU0-GPU3 and never
touches GPU4-GPU7. The method remains EfficientNet1DV2 VAE-LH online AT plus
locked three-chain AugMix.
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
WATCHDOG_DIR = REPORT_DIR / "watchdogs"
BASELINE_CSV = REPORT_DIR / "effnet_completed_recovery_snapshot_20260621_2252.csv"
EVAL_NAME = "eval_pn2021_c_v7_refexcluded_stream_dual_model_10to15pp_v1_raw_first.json"
CORRUPTIONS = [
    "baseline_shift",
    "baseline_wander",
    "emg_noise",
    "powerline_noise",
    "random_leads_masking",
]


CAND94_METHOD = "cand94_q8192_hull24_perop_maskpower_c15_b20_ep45_4g"
CAND96_METHOD = "cand96_q8192_hull24_balancemask_c18_b24_ep36"


CENTERS: dict[str, dict[str, str]] = {
    "cpsc_2018": {
        "gpu": "0",
        "short": "cpsc",
        "cand94_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_cpsc"
            / "cand94_cpsc_postcand93_4g_rawfirst_eval_20260622_gpu0"
            / "cpsc_2018"
            / CAND94_METHOD
            / EVAL_NAME
        ),
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_cpsc.yaml",
        "train_run_id": "cand96_cpsc_after_cand94_20260622_gpu0",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_cpsc",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_cpsc.yaml",
        "eval_run_id": "cand96_cpsc_balancemask_rawfirst_eval_20260622_gpu0",
        "eval_root": "pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_cpsc",
    },
    "chapman_shaoxing": {
        "gpu": "1",
        "short": "chapman",
        "cand94_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_chapman_shaoxing"
            / "cand94_chapman_postcand93_4g_rawfirst_eval_20260622_gpu1"
            / "chapman_shaoxing"
            / CAND94_METHOD
            / EVAL_NAME
        ),
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_chapman_shaoxing.yaml",
        "train_run_id": "cand96_chapman_after_cand94_20260622_gpu1",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_chapman_shaoxing",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_chapman_shaoxing.yaml",
        "eval_run_id": "cand96_chapman_balancemask_rawfirst_eval_20260622_gpu1",
        "eval_root": "pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_chapman_shaoxing",
    },
    "georgia": {
        "gpu": "2",
        "short": "georgia",
        "cand94_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_georgia"
            / "cand94_georgia_postcand93_4g_rawfirst_eval_20260622_gpu2"
            / "georgia"
            / CAND94_METHOD
            / EVAL_NAME
        ),
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_georgia.yaml",
        "train_run_id": "cand96_georgia_after_cand94_20260622_gpu2",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_georgia",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_georgia.yaml",
        "eval_run_id": "cand96_georgia_balancemask_rawfirst_eval_20260622_gpu2",
        "eval_root": "pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_georgia",
    },
    "ningbo": {
        "gpu": "3",
        "short": "ningbo",
        "cand94_eval_json": str(
            OUTPUT_ROOT
            / "pn2021c_effnet_dual3ch_cand94_postcand93_4g_rawfirst_ningbo"
            / "cand94_ningbo_postcand93_4g_rawfirst_eval_20260622_gpu3"
            / "ningbo"
            / CAND94_METHOD
            / EVAL_NAME
        ),
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_ningbo.yaml",
        "train_run_id": "cand96_ningbo_after_cand94_20260622_gpu3",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand96_q8192_hull24_balancemask_c18_b24_ep36_ningbo",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_ningbo.yaml",
        "eval_run_id": "cand96_ningbo_balancemask_rawfirst_eval_20260622_gpu3",
        "eval_root": "pn2021c_effnet_dual3ch_cand96_balancemask_rawfirst_ningbo",
    },
}


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _load_direct_baseline(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            baseline[(row["center"], row["operator"])] = {
                "direct_clean_auroc": float(row["direct_clean_auroc"]),
                "direct_clean_auprc": float(row["direct_clean_auprc"]),
                "direct_corrupted_auroc": float(row["direct_corrupted_auroc"]),
                "direct_corrupted_auprc": float(row["direct_corrupted_auprc"]),
            }
    return baseline


def _wait_for_json(path: Path, poll_seconds: int) -> None:
    while not path.exists():
        _log(f"waiting JSON: {path}")
        time.sleep(poll_seconds)
    json.loads(path.read_text(encoding="utf-8"))
    _log(f"JSON ready: {path}")


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


def _wait_for_slot(gpu: str, max_load: float, poll_seconds: int, reason: str) -> None:
    while True:
        used = _gpu_memory_mib().get(gpu, 0)
        load = _host_load_1min()
        if used <= 1000 and (max_load <= 0 or load <= max_load):
            _log(f"slot ready for {reason}: GPU{gpu} used={used} MiB load1={load:.2f}")
            return
        _log(f"waiting slot for {reason}: GPU{gpu} used={used} MiB load1={load:.2f} max_load={max_load:.2f}")
        time.sleep(poll_seconds)


def _train_base(spec: dict[str, str]) -> Path:
    return OUTPUT_ROOT / spec["train_family"] / spec["train_run_id"]


def _latest_checkpoint(spec: dict[str, str]) -> Path | None:
    base = _train_base(spec)
    matches = sorted(base.glob("*/last_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _expected_cand96_eval_json(center: str) -> Path:
    spec = CENTERS[center]
    return OUTPUT_ROOT / spec["eval_root"] / spec["eval_run_id"] / center / CAND96_METHOD / EVAL_NAME


def _run_stage(config: str, run_id: str, gpu: str, log_path: Path) -> None:
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
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"[{_stamp()}] finished run_id={run_id}\n")


def _build_recovery_rows(method: str, json_paths: dict[str, Path]) -> list[dict[str, Any]]:
    baseline = _load_direct_baseline(BASELINE_CSV)
    rows: list[dict[str, Any]] = []
    for center, path in json_paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = data.get("label_mapping") or {}
        for op in CORRUPTIONS:
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
                    "cand": method,
                    "json": str(path),
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


def _write_recovery(method: str, rows: list[dict[str, Any]], threshold_pp: float) -> tuple[bool, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = REPORT_DIR / f"{method}_recovery_snapshot_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_auroc = _mean([float(r["recovery_auroc_pp"]) for r in rows])
    mean_auprc = _mean([float(r["recovery_auprc_pp"]) for r in rows])
    achieved = mean_auroc >= threshold_pp or mean_auprc >= threshold_pp
    status = "ACHIEVED" if achieved else "NOT_ACHIEVED"

    center_lines = []
    centers_with_rows = [center for center in CENTERS if any(r["center"] == center for r in rows)]
    for center in centers_with_rows:
        crows = [r for r in rows if r["center"] == center]
        center_lines.append(
            f"- {center}: "
            f"{_mean([float(r['recovery_auroc_pp']) for r in crows]):.2f}/"
            f"{_mean([float(r['recovery_auprc_pp']) for r in crows]):.2f} pp"
        )
    op_lines = []
    for op in CORRUPTIONS:
        orows = [r for r in rows if r["operator"] == op]
        op_lines.append(
            f"- {op}: "
            f"{_mean([float(r['recovery_auroc_pp']) for r in orows]):.2f}/"
            f"{_mean([float(r['recovery_auprc_pp']) for r in orows]):.2f} pp"
        )

    summary_path = REPORT_DIR / f"{method}_recovery_summary_{stamp}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# {method} PN2021-C Recovery Snapshot - {stamp}",
                "",
                f"Goal gate status: {status}",
                f"Four-center/operator mean recovery: {mean_auroc:.2f} pp AUROC / {mean_auprc:.2f} pp AUPRC.",
                "Baseline: EfficientNet1DV2 direct K500 fullFT snapshot.",
                "Evaluation: dual_model_10to15pp_v1 raw-first PN2021-C, ref-excluded.",
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
    return achieved, csv_path, summary_path


def _center_recovery_means(center: str, json_path: Path) -> tuple[float, float]:
    rows = _build_recovery_rows("cand94_center_gate", {center: json_path})
    mean_auroc = _mean([float(r["recovery_auroc_pp"]) for r in rows])
    mean_auprc = _mean([float(r["recovery_auprc_pp"]) for r in rows])
    return mean_auroc, mean_auprc


def _run_cand96_center(center: str, spec: dict[str, str], args: argparse.Namespace, errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand96_{spec['short']}_after_cand94_4g_20260622_gpu{spec['gpu']}.log"
    try:
        final_json = _expected_cand96_eval_json(center)
        if final_json.exists():
            _log(f"{center}: cand96 eval already exists; skip")
            return
        if _run_id_active(spec["train_run_id"]):
            _log(f"{center}: cand96 train active; waiting")
            while _run_id_active(spec["train_run_id"]):
                time.sleep(args.poll_seconds)
        elif _latest_checkpoint(spec) is None:
            _wait_for_slot(spec["gpu"], args.max_load_before_train, args.poll_seconds, f"{center} cand96 train")
            _run_stage(spec["train_config"], spec["train_run_id"], spec["gpu"], log_path)
        ckpt = _latest_checkpoint(spec)
        if ckpt is None:
            raise RuntimeError(f"{center}: cand96 train ended but last_model.pt not found")
        _log(f"{center}: cand96 checkpoint ready: {ckpt}")

        if final_json.exists():
            return
        _wait_for_slot(spec["gpu"], args.max_load_before_eval, args.poll_seconds, f"{center} cand96 eval")
        _run_stage(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log_path)
        if not final_json.exists():
            raise RuntimeError(f"{center}: cand96 eval JSON missing after eval: {final_json}")
    except Exception as exc:  # pragma: no cover - long-running watcher path
        errors.append(f"{center}: {exc}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_stamp()}] ERROR {center}: {exc}\n")


def _run_per_center_gate(center: str, spec: dict[str, str], args: argparse.Namespace, errors: list[str]) -> None:
    try:
        cand94_path = Path(spec["cand94_eval_json"])
        _wait_for_json(cand94_path, args.poll_seconds)
        mean_auroc, mean_auprc = _center_recovery_means(center, cand94_path)
        _log(
            f"{center}: cand94 center recovery gate "
            f"{mean_auroc:.2f}/{mean_auprc:.2f} pp "
            f"(threshold={args.goal_threshold_pp:.2f})"
        )
        if mean_auroc >= args.goal_threshold_pp or mean_auprc >= args.goal_threshold_pp:
            _log(f"{center}: cand94 center gate achieved; skip cand96 fallback for this GPU slot")
            return
        _log(f"{center}: cand94 center gate missed; launch cand96 fallback on GPU{spec['gpu']}")
        _run_cand96_center(center, spec, args, errors)
    except Exception as exc:  # pragma: no cover - long-running watchdog path
        errors.append(f"{center}: {exc}")
        _log(f"{center}: per-center gate failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--goal-threshold-pp", type=float, default=9.5)
    parser.add_argument("--max-load-before-train", type=float, default=170.0)
    parser.add_argument("--max-load-before-eval", type=float, default=0.0)
    args = parser.parse_args(argv)

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    _log("cand96 after-cand94 max-4-GPU watcher start")
    _log("mode=per-center fallback: each freed GPU slot can launch cand96 after its cand94 eval gate")

    cand94_paths = {center: Path(spec["cand94_eval_json"]) for center, spec in CENTERS.items()}
    errors: list[str] = []
    threads = [
        threading.Thread(target=_run_per_center_gate, args=(center, spec, args, errors), daemon=False)
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

    achieved, csv_path, summary_path = _write_recovery("cand94_robust4g_gate", _build_recovery_rows("cand94_robust4g_gate", cand94_paths), args.goal_threshold_pp)
    _log(f"cand94 gate CSV: {csv_path}")
    _log(f"cand94 gate summary: {summary_path}")
    if achieved:
        _log("cand94 achieved the global goal gate")
        return 0

    cand96_paths = {center: _expected_cand96_eval_json(center) for center in CENTERS if _expected_cand96_eval_json(center).exists()}
    if cand96_paths:
        achieved, csv_path, summary_path = _write_recovery("cand96_balancemask", _build_recovery_rows("cand96_balancemask", cand96_paths), args.goal_threshold_pp)
        _log(f"cand96 CSV: {csv_path}")
        _log(f"cand96 summary: {summary_path}")
        _log(f"cand96 goal status over completed fallback centers: {'ACHIEVED' if achieved else 'NOT_ACHIEVED'}")
    else:
        _log("cand94 missed the global gate, but no cand96 fallback JSONs were produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
