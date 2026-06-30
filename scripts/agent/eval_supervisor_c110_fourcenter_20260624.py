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
    "pn2021c_goal_effnet_officials5_c110_fourcenter_20260624/"
    "eval_c110_officials5_fourcenter"
)
LOGDIR = Path("/dev/shm/ecg_goal_20260624_c110_fourcenter_6gpu")
DIRECT_ROOT = Path(
    "/home/linbinhao/ECG_adv_data/runs/"
    "pn2021c_official_s5_composite_direct_drop_20260622/effnet"
)
OLD_C110_ROOT = Path(
    "/home/linbinhao/ECG_adv_data/runs/"
    "pn2021c_goal_effnet_officials5_pilots_20260624/"
    "eval_c109_c110_officials5/c110"
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
        return LOGDIR / f"c110_{self.center}_eval_watch.log"

    def model_dir(self) -> Path | None:
        if not self.train_root.exists():
            return None
        matches = sorted(
            p for p in self.train_root.glob(f"*fullft_{self.tag}_ep45_seed20260601") if p.is_dir()
        )
        return matches[-1] if matches else None

    def clean_eval(self) -> Path | None:
        model_dir = self.model_dir()
        if model_dir is None:
            return None
        return model_dir / "eval_result_v7_exclrefs_crop1000.json"


TASKS = [
    EvalTask(
        "chapman_shaoxing",
        4,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand110_weakcombo_cover_c24_strongjsd_b20_ep45_chapman_shaoxing/"
            "goal_c110_effnet_chapman_20260624_train_6gpu"
        ),
        "c110officials5_weakcombo_cover_c24_strongjsd_b20_ep45",
        Path(
            "/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/"
            "subsets/chapman_shaoxing/k500_seed20260601/"
            "chapman_shaoxing_real_k500_seed20260601.ref_meta.json"
        ),
    ),
    EvalTask(
        "ningbo",
        5,
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand110_weakcombo_cover_c24_strongjsd_b20_ep45_ningbo/"
            "goal_c110_effnet_ningbo_20260624_train_6gpu"
        ),
        "c110officials5_weakcombo_cover_c24_strongjsd_b20_ep45",
        Path(
            "/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/"
            "subsets/ningbo/k500_seed20260601/"
            "ningbo_real_k500_seed20260601.ref_meta.json"
        ),
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
    method_jsons = {
        "cpsc_2018": OLD_C110_ROOT
        / "cpsc_2018/eval_official_s5_locked_composite_rawfirst.json",
        "georgia": OLD_C110_ROOT / "georgia/eval_official_s5_locked_composite_rawfirst.json",
        "chapman_shaoxing": ROOT
        / "chapman_shaoxing/eval_official_s5_locked_composite_rawfirst.json",
        "ningbo": ROOT / "ningbo/eval_official_s5_locked_composite_rawfirst.json",
    }
    cmd = [
        str(PY),
        "scripts/agent/summarize_official_s5_composite_recovery.py",
        "--model",
        "effnet",
        "--method",
        "c110_weakcombo_cover",
    ]
    for center in ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"]:
        cmd.extend(
            [
                "--direct-json",
                f"{center}={DIRECT_ROOT / center / 'eval_official_s5_composite_direct_rawfirst.json'}",
            ]
        )
    for center in ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"]:
        cmd.extend(["--method-json", f"{center}={method_jsons[center]}"])
    cmd.extend(
        [
            "--output-dir",
            str(ROOT / "summary"),
            "--prefix",
            "c110_fourcenter_official_s5_composite_method",
        ]
    )
    return cmd


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    master_log = LOGDIR / "eval_supervisor_c110_fourcenter_20260624.log"
    append(master_log, "supervisor_start")

    running: dict[str, subprocess.Popen] = {}
    complete: set[str] = set()
    failed: set[str] = set()
    summary_done = False

    while True:
        for task in TASKS:
            key = task.center
            if key in complete or key in failed:
                continue
            if task.out_json.exists() and task.out_json.stat().st_size > 0:
                complete.add(key)
                append(master_log, f"eval_output_exists key={key}")
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

            model_dir = task.model_dir()
            clean_eval = task.clean_eval()
            if model_dir is None or clean_eval is None or not clean_eval.exists():
                continue
            try:
                mem = gpu_mem_mib(task.gpu)
            except Exception as exc:
                append(master_log, f"gpu_query_failed key={key} error={exc}")
                continue
            if mem >= 1000:
                continue

            task.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
            env.update(
                {
                    "OMP_NUM_THREADS": "6",
                    "MKL_NUM_THREADS": "6",
                    "OPENBLAS_NUM_THREADS": "6",
                    "NUMEXPR_NUM_THREADS": "6",
                    "TORCH_NUM_THREADS": "6",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log_f = task.log_path.open("a", encoding="utf-8")
            append(task.log_path, f"eval_start key={key} gpu={task.gpu}")
            running[key] = subprocess.Popen(
                eval_command(task, model_dir=model_dir, clean_eval=clean_eval),
                cwd=REPO,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            append(master_log, f"eval_launched key={key} pid={running[key].pid} gpu={task.gpu}")

        if len(complete) == len(TASKS) and not summary_done:
            log_path = LOGDIR / "c110_fourcenter_summary_watch.log"
            log_f = log_path.open("a", encoding="utf-8")
            append(log_path, "summary_start")
            proc = subprocess.Popen(summary_command(), cwd=REPO, stdout=log_f, stderr=subprocess.STDOUT)
            rc = proc.wait()
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
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
