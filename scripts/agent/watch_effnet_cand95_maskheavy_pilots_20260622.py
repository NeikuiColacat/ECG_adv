#!/usr/bin/env python3
"""Load-gated launcher for cand95 mask-heavy short pilots on GPU4/GPU7."""

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


PILOTS = {
    "georgia": {
        "gpu": "4",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_georgia.yaml",
        "train_run_id": "cand95_maskheavy_georgia_pilot_20260622_gpu4",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_georgia.yaml",
        "eval_run_id": "cand95_maskheavy_georgia_rawfirst_eval_20260622_gpu4",
        "eval_json": OUTPUT_ROOT
        / "pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_georgia"
        / "cand95_maskheavy_georgia_rawfirst_eval_20260622_gpu4"
        / "georgia"
        / "cand95_maskheavy_c20_b20_ep18_pilot"
        / EVAL_NAME,
    },
    "chapman_shaoxing": {
        "gpu": "7",
        "train_config": "configs/experiments/effnet_vae_lhat_augmix_threechain_dualmodel_cand95_maskheavy_c20_b20_ep18_chapman_shaoxing.yaml",
        "train_run_id": "cand95_maskheavy_chapman_pilot_20260622_gpu7",
        "eval_config": "configs/experiments/pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_chapman_shaoxing.yaml",
        "eval_run_id": "cand95_maskheavy_chapman_rawfirst_eval_20260622_gpu7",
        "eval_json": OUTPUT_ROOT
        / "pn2021c_effnet_dual3ch_cand95_maskheavy_rawfirst_chapman_shaoxing"
        / "cand95_maskheavy_chapman_rawfirst_eval_20260622_gpu7"
        / "chapman_shaoxing"
        / "cand95_maskheavy_c20_b20_ep18_pilot"
        / EVAL_NAME,
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
    out: dict[str, int] = {}
    for line in result.stdout.splitlines():
        idx, used = [part.strip() for part in line.split(",", maxsplit=1)]
        out[idx] = int(used)
    return out


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


def _wait_for_slot(gpu: str, max_load: float, poll_seconds: int, log) -> None:
    while True:
        load = _host_load_1min()
        used = _gpu_memory_mib().get(gpu, 0)
        if load <= max_load and used <= 500:
            print(
                f"[{_stamp()}] slot ready: GPU{gpu} used={used} MiB load1={load:.2f}",
                file=log,
                flush=True,
            )
            return
        print(
            f"[{_stamp()}] waiting slot: GPU{gpu} used={used} MiB load1={load:.2f} max_load={max_load:.2f}",
            file=log,
            flush=True,
        )
        time.sleep(poll_seconds)


def _run_stage(config: str, run_id: str, gpu: str, log, stage_name: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
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
    print(f"[{_stamp()}] start {stage_name}: {' '.join(cmd)}", file=log, flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"[{_stamp()}] finished {stage_name}", file=log, flush=True)


def _worker(center: str, spec: dict[str, object], max_load: float, poll_seconds: int) -> None:
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = WATCHDOG_DIR / f"cand95_maskheavy_{center}_gpu{spec['gpu']}_20260622.log"
    with log_path.open("a", encoding="utf-8") as log:
        print(f"[{_stamp()}] cand95 worker start center={center}", file=log, flush=True)
        if Path(spec["eval_json"]).exists():
            print(f"[{_stamp()}] eval already exists, skip: {spec['eval_json']}", file=log, flush=True)
            return
        if _run_id_active(str(spec["train_run_id"])) or _run_id_active(str(spec["eval_run_id"])):
            print(f"[{_stamp()}] run id already active, watcher will not duplicate launch", file=log, flush=True)
            return
        _wait_for_slot(str(spec["gpu"]), max_load, poll_seconds, log)
        _run_stage(str(spec["train_config"]), str(spec["train_run_id"]), str(spec["gpu"]), log, "train")
        _wait_for_slot(str(spec["gpu"]), max_load, poll_seconds, log)
        _run_stage(str(spec["eval_config"]), str(spec["eval_run_id"]), str(spec["gpu"]), log, "eval")
        print(f"[{_stamp()}] cand95 worker done center={center}", file=log, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-load-before-launch", type=float, default=170.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    threads = [
        threading.Thread(
            target=_worker,
            args=(center, spec, args.max_load_before_launch, args.poll_seconds),
            daemon=False,
        )
        for center, spec in PILOTS.items()
    ]
    print(f"[{_stamp()}] cand95 maskheavy watcher start", flush=True)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"[{_stamp()}] cand95 maskheavy watcher done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
