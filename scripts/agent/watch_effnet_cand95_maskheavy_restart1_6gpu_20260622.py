#!/usr/bin/env python3
"""Restart cand95 mask-heavy pilots on GPU4/GPU7 with fresh run ids."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import threading
import time


PROJECT_ROOT = Path("/home/linbinhao/ECG_adv_Gen")
PYTHON_BIN = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
LOCAL_CONFIG = "configs/local/linbinhao_server.example.yaml"
OUTPUT_ROOT = Path("/home/linbinhao/ECG_adv_data/runs")
REPORT_DIR = OUTPUT_ROOT / "pn2021c_sota_ablation_effnet_ecgfounder_20260621"
WATCHDOG_DIR = REPORT_DIR / "watchdogs"
EVAL_NAME = "eval_pn2021_c_v7_refexcluded_stream_dual_model_10to15pp_v1_raw_first.json"
METHOD_NAME = "cand95_maskheavy_c20_b20_ep18_restart1"


PILOTS = {
    "georgia": {
        "gpu": "4",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_georgia.yaml",
        "train_run_id": "cand95_maskheavy_georgia_restart1_20260622_gpu4",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_georgia",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_georgia_restart1.yaml",
        "eval_run_id": "cand95_maskheavy_georgia_restart1_rawfirst_eval_20260622_gpu4",
        "eval_root": "pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_georgia_restart1",
    },
    "chapman_shaoxing": {
        "gpu": "7",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_chapman_shaoxing.yaml",
        "train_run_id": "cand95_maskheavy_chapman_restart1_20260622_gpu7",
        "train_family": "effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_chapman_shaoxing",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_chapman_shaoxing_restart1.yaml",
        "eval_run_id": "cand95_maskheavy_chapman_restart1_rawfirst_eval_20260622_gpu7",
        "eval_root": "pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_chapman_shaoxing_restart1",
    },
}


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _host_load_1min() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


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
        idx, used = [part.strip() for part in line.split(",", maxsplit=1)]
        usage[idx] = int(used)
    return usage


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


def _train_base(spec: dict[str, str]) -> Path:
    return OUTPUT_ROOT / spec["train_family"] / spec["train_run_id"]


def _latest_checkpoint(spec: dict[str, str]) -> Path | None:
    matches = sorted(_train_base(spec).glob("*/last_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _eval_json(center: str, spec: dict[str, str]) -> Path:
    return OUTPUT_ROOT / spec["eval_root"] / spec["eval_run_id"] / center / METHOD_NAME / EVAL_NAME


def _wait_for_slot(gpu: str, max_load: float, poll_seconds: int, log) -> None:
    while True:
        load = _host_load_1min()
        used = _gpu_memory_mib().get(gpu, 0)
        if used <= 500 and (max_load <= 0 or load <= max_load):
            print(f"[{_stamp()}] slot ready GPU{gpu}: used={used} MiB load1={load:.2f}", file=log, flush=True)
            return
        print(
            f"[{_stamp()}] waiting GPU{gpu}: used={used} MiB load1={load:.2f} max_load={max_load:.2f}",
            file=log,
            flush=True,
        )
        time.sleep(poll_seconds)


def _run_stage(config: str, run_id: str, gpu: str, log, stage: str) -> None:
    if _run_id_active(run_id):
        print(f"[{_stamp()}] {stage} already active: {run_id}", file=log, flush=True)
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
    print(f"[{_stamp()}] start {stage}: run_id={run_id} gpu={gpu} config={config}", file=log, flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"[{_stamp()}] finished {stage}: run_id={run_id}", file=log, flush=True)


def _worker(center: str, spec: dict[str, str], max_load: float, poll_seconds: int, errors: list[str]) -> None:
    log_path = WATCHDOG_DIR / f"cand95_{center}_restart1_6gpu_20260622_gpu{spec['gpu']}.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            print(f"[{_stamp()}] cand95 restart1 worker start center={center}", file=log, flush=True)
            if _eval_json(center, spec).exists():
                print(f"[{_stamp()}] eval exists; skip: {_eval_json(center, spec)}", file=log, flush=True)
                return
            if _latest_checkpoint(spec) is None:
                _wait_for_slot(spec["gpu"], max_load, poll_seconds, log)
                _run_stage(spec["train_config"], spec["train_run_id"], spec["gpu"], log, "train")
            ckpt = _latest_checkpoint(spec)
            if ckpt is None:
                raise RuntimeError(f"missing last_model.pt under {_train_base(spec)}")
            print(f"[{_stamp()}] checkpoint ready: {ckpt}", file=log, flush=True)
            _wait_for_slot(spec["gpu"], max_load, poll_seconds, log)
            _run_stage(spec["eval_config"], spec["eval_run_id"], spec["gpu"], log, "eval")
            if not _eval_json(center, spec).exists():
                raise RuntimeError(f"eval JSON missing: {_eval_json(center, spec)}")
            print(f"[{_stamp()}] cand95 restart1 worker done center={center}", file=log, flush=True)
    except Exception as exc:  # pragma: no cover - long-running scheduler path
        errors.append(f"{center}: {exc}")
        with log_path.open("a", encoding="utf-8") as log:
            print(f"[{_stamp()}] ERROR {center}: {exc}", file=log, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-load-before-launch", type=float, default=260.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    threads = [
        threading.Thread(target=_worker, args=(center, spec, args.max_load_before_launch, args.poll_seconds, errors))
        for center, spec in PILOTS.items()
    ]
    print(f"[{_stamp()}] cand95 restart1 6gpu watcher start", flush=True)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        for err in errors:
            print(f"[{_stamp()}] worker failed: {err}", flush=True)
        return 1
    print(f"[{_stamp()}] cand95 restart1 6gpu watcher done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
