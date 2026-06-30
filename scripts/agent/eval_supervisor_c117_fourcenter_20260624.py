#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PY = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")
REPO = Path("/home/linbinhao/ECG_adv_Gen")
ROOT = Path(
    "/home/linbinhao/ECG_adv_data/runs/"
    "pn2021c_goal_effnet_officials5_c117_fourcenter_20260624/"
    "eval_c117_officials5_fourcenter"
)
LOGDIR = Path("/dev/shm/ecg_goal_c117_20260624")
DIRECT_ROOT = Path(
    "/home/linbinhao/ECG_adv_data/runs/"
    "pn2021c_official_s5_composite_direct_drop_20260622/effnet"
)

COMBOS = [
    "baseline_shift+baseline_wander",
    "baseline_shift+emg_noise",
    "baseline_shift+powerline_noise",
    "baseline_shift+random_leads_masking",
    "baseline_wander+emg_noise",
    "baseline_wander+powerline_noise",
    "baseline_wander+random_leads_masking",
    "emg_noise+powerline_noise",
    "emg_noise+random_leads_masking",
    "powerline_noise+random_leads_masking",
    "baseline_shift+baseline_wander+emg_noise",
    "baseline_shift+baseline_wander+powerline_noise",
    "baseline_shift+baseline_wander+random_leads_masking",
    "baseline_shift+emg_noise+powerline_noise",
    "baseline_shift+emg_noise+random_leads_masking",
    "baseline_shift+powerline_noise+random_leads_masking",
    "baseline_wander+emg_noise+powerline_noise",
    "baseline_wander+emg_noise+random_leads_masking",
    "baseline_wander+powerline_noise+random_leads_masking",
    "emg_noise+powerline_noise+random_leads_masking",
]


@dataclass(frozen=True)
class EvalTask:
    center: str
    gpu: int
    train_root: Path
    tag: str
    ref_meta: Path

    @property
    def out_dir(self) -> Path:
        return ROOT / self.center

    @property
    def out_json(self) -> Path:
        return self.out_dir / "eval_official_s5_locked_composite_rawfirst.json"

    @property
    def log_path(self) -> Path:
        return LOGDIR / f"c117_{self.center}_eval_watch.log"

    def model_dir(self) -> Path | None:
        if not self.train_root.exists():
            return None
        matches = sorted(
            p for p in self.train_root.glob(f"*fullft_{self.tag}_ep60_seed20260601")
            if p.is_dir()
        )
        return matches[-1] if matches else None

    def clean_eval(self) -> Path | None:
        model_dir = self.model_dir()
        if model_dir is None:
            return None
        return model_dir / "eval_result_v7_exclrefs_crop1000.json"

    def train_stdout(self) -> Path | None:
        model_dir = self.model_dir()
        if model_dir is None:
            return None
        return model_dir / "train_stdout.log"


def ref_meta(center: str) -> Path:
    return Path(
        "/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/"
        f"subsets/{center}/k500_seed20260601/{center}_real_k500_seed20260601.ref_meta.json"
    )


TASKS = [
    EvalTask(
        "chapman_shaoxing",
        1,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand117_triplecover_c24_strongjsd_b20_ep60_chapman_shaoxing/"
            "goal_c117_effnet_chapman_20260624_train_threads4"
        ),
        "c117officials5_triplecover_c24_strongjsd_b20_ep60",
        ref_meta("chapman_shaoxing"),
    ),
    EvalTask(
        "cpsc_2018",
        0,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand117_triplecover_c24_strongjsd_b20_ep60_cpsc/"
            "goal_c117_effnet_cpsc_20260624_train_threads4"
        ),
        "c117officials5_triplecover_c24_strongjsd_b20_ep60",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "georgia",
        2,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand117_triplecover_c24_strongjsd_b20_ep60_georgia/"
            "goal_c117_effnet_georgia_20260624_train_threads4"
        ),
        "c117officials5_triplecover_c24_strongjsd_b20_ep60",
        ref_meta("georgia"),
    ),
    EvalTask(
        "ningbo",
        3,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand117_triplecover_c24_strongjsd_b20_ep60_ningbo/"
            "goal_c117_effnet_ningbo_20260624_train_threads4"
        ),
        "c117officials5_triplecover_c24_strongjsd_b20_ep60",
        ref_meta("ningbo"),
    ),
]


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp()}] {msg}\n")


def gpu_mem_mib(gpu: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0].strip())


def training_still_running(task: EvalTask) -> bool:
    pattern = f"goal_c117_effnet_{task.center.split('_')[0]}"
    if task.center == "cpsc_2018":
        pattern = "goal_c117_effnet_cpsc"
    if task.center == "chapman_shaoxing":
        pattern = "goal_c117_effnet_chapman"
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", pattern],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return (
        "synth_online_at_super5.py" in out
        or "run_effnet_latent_augmix_stage3_20260524.py" in out
        or "run_experiment.py" in out
    )


def task_ready(task: EvalTask) -> tuple[bool, str]:
    model_dir = task.model_dir()
    if model_dir is None:
        return False, "missing_model_dir"
    clean_eval = task.clean_eval()
    if clean_eval is None or not clean_eval.exists() or clean_eval.stat().st_size == 0:
        return False, "missing_clean_eval"
    last_ckpt = model_dir / "last_model.pt"
    if not last_ckpt.exists() or last_ckpt.stat().st_size == 0:
        return False, "missing_last_model"
    if training_still_running(task):
        return False, "training_still_running"
    return True, "ready"


def eval_command(task: EvalTask, *, model_dir: Path, clean_eval: Path) -> list[str]:
    return [
        str(PY),
        "-u",
        "scripts/triple_labels/eval_pn2021_corruptions.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(model_dir),
        "--checkpoint_name",
        "last_model.pt",
        "--model_name",
        "efficientnet1dv2",
        "--mode",
        "stream",
        "--corruption_input",
        "raw_first",
        "--pn2021_root",
        "/home/linbinhao/ECG_adv_data/physionet2021",
        "--clean_eval_json",
        str(clean_eval),
        "--exclude_ref_ids",
        str(task.ref_meta),
        "--centers",
        task.center,
        "--corruptions",
        *COMBOS,
        "--severities",
        "5",
        "--severity_profile",
        "standard",
        "--crop_len",
        "1000",
        "--batch_size",
        "256",
        "--num_workers",
        "2",
        "--device",
        "cuda",
        "--output_path",
        str(task.out_json),
    ]


def summary_command() -> list[str]:
    cmd = [
        str(PY),
        "scripts/agent/summarize_official_s5_composite_recovery.py",
        "--model",
        "effnet",
        "--method",
        "c117_triplecover_strongjsd_ep60",
    ]
    for center in ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"]:
        cmd.extend(
            [
                "--direct-json",
                f"{center}={DIRECT_ROOT / center / 'eval_official_s5_composite_direct_rawfirst.json'}",
            ]
        )
    for center in ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"]:
        cmd.extend(
            [
                "--method-json",
                f"{center}={ROOT / center / 'eval_official_s5_locked_composite_rawfirst.json'}",
            ]
        )
    cmd.extend(
        [
            "--output-dir",
            str(ROOT / "summary"),
            "--prefix",
            "c117_fourcenter_official_s5_composite_method",
        ]
    )
    return cmd


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    master_log = LOGDIR / "eval_supervisor_c117_fourcenter_20260624.log"
    append(master_log, "supervisor_start max_eval_gpus=4 threads=4")

    running: dict[str, subprocess.Popen] = {}
    complete: set[str] = set()
    failed: set[str] = set()
    last_status: dict[str, str] = {}
    summary_done = False

    while True:
        for task in TASKS:
            key = task.center
            if key in complete or key in failed:
                continue
            proc = running.get(key)
            if proc is not None:
                rc = proc.poll()
                if rc is None:
                    continue
                running.pop(key, None)
                if rc == 0 and task.out_json.exists() and task.out_json.stat().st_size > 0:
                    complete.add(key)
                    append(master_log, f"eval_complete key={key}")
                else:
                    failed.add(key)
                    append(master_log, f"eval_failed key={key} rc={rc}")
                continue

            if task.out_json.exists() and task.out_json.stat().st_size > 0:
                complete.add(key)
                append(master_log, f"eval_output_exists key={key}")
                continue

            ready, reason = task_ready(task)
            if not ready:
                if last_status.get(key) != reason:
                    append(master_log, f"wait key={key} reason={reason}")
                    last_status[key] = reason
                continue

            if gpu_mem_mib(task.gpu) >= 1000:
                if last_status.get(key) != "gpu_busy":
                    append(master_log, f"wait key={key} reason=gpu_busy gpu={task.gpu}")
                    last_status[key] = "gpu_busy"
                continue

            model_dir = task.model_dir()
            clean_eval = task.clean_eval()
            if model_dir is None or clean_eval is None:
                failed.add(key)
                append(master_log, f"unexpected_missing_paths key={key}")
                continue

            task.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
            env.update(
                {
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                    "TORCH_NUM_THREADS": "4",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log_f = task.log_path.open("a", encoding="utf-8")
            append(task.log_path, f"eval_start key={key} gpu={task.gpu} model_dir={model_dir}")
            running[key] = subprocess.Popen(
                eval_command(task, model_dir=model_dir, clean_eval=clean_eval),
                cwd=REPO,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            append(master_log, f"eval_launched key={key} pid={running[key].pid} gpu={task.gpu}")

        if len(complete) == len(TASKS) and not summary_done:
            log_path = LOGDIR / "c117_fourcenter_summary_watch.log"
            log_f = log_path.open("a", encoding="utf-8")
            append(log_path, "summary_start")
            rc = subprocess.Popen(
                summary_command(),
                cwd=REPO,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            ).wait()
            if rc == 0:
                summary_done = True
                append(master_log, "summary_done")
            else:
                failed.add("summary")
                append(master_log, f"summary_failed rc={rc}")

        if summary_done:
            append(master_log, "supervisor_done")
            return 0
        if failed:
            append(master_log, f"supervisor_has_failures failed={sorted(failed)}")
            return 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
