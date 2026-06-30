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
    "pn2021c_goal_effnet_officials5_c111_c115_pilots_20260624/"
    "eval_cpsc_georgia_officials5"
)
LOGDIR = Path("/dev/shm/ecg_goal_20260624_c111_c115_pilots")
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
    cand: str
    center: str
    train_root: Path
    tag: str
    ref_meta: Path

    @property
    def out_dir(self) -> Path:
        return ROOT / self.cand / self.center

    @property
    def out_json(self) -> Path:
        return self.out_dir / f"{self.cand}_{self.center}_official_s5_composite_method_rawfirst.json"

    @property
    def log_path(self) -> Path:
        return LOGDIR / f"{self.cand}_{self.center}_eval_watch.log"

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


def ref_meta(center: str) -> Path:
    return Path(
        "/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/"
        f"subsets/{center}/k500_seed20260601/{center}_real_k500_seed20260601.ref_meta.json"
    )


TASKS = [
    EvalTask(
        "c111",
        "cpsc_2018",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand111_midlatent20_aw28_c24_b20_ep45_cpsc/"
            "goal_c111r1_effnet_cpsc_20260624_train"
        ),
        "c111officials5_midlatent20_aw28_c24_b20_ep45",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "c111",
        "georgia",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand111_midlatent20_aw28_c24_b20_ep45_georgia/"
            "goal_c111r1_effnet_georgia_20260624_train"
        ),
        "c111officials5_midlatent20_aw28_c24_b20_ep45",
        ref_meta("georgia"),
    ),
    EvalTask(
        "c112",
        "cpsc_2018",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand112_safewarm_c24_b20_aw20_ep45_cpsc/"
            "goal_c112r1_effnet_cpsc_20260624_train"
        ),
        "c112officials5_safewarm_c24_b20_aw20_ep45",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "c112",
        "georgia",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand112_safewarm_c24_b20_aw20_ep45_georgia/"
            "goal_c112r1_effnet_georgia_20260624_train"
        ),
        "c112officials5_safewarm_c24_b20_aw20_ep45",
        ref_meta("georgia"),
    ),
    EvalTask(
        "c113",
        "cpsc_2018",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand113_rawcons_hi_bce_c24_b20_aw20_ep45_cpsc/"
            "goal_c113r1_effnet_cpsc_20260624_train"
        ),
        "c113officials5_rawcons_hi_bce_c24_b20_aw20_ep45",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "c113",
        "georgia",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand113_rawcons_hi_bce_c24_b20_aw20_ep45_georgia/"
            "goal_c113r1_effnet_georgia_20260624_train"
        ),
        "c113officials5_rawcons_hi_bce_c24_b20_aw20_ep45",
        ref_meta("georgia"),
    ),
    EvalTask(
        "c114",
        "cpsc_2018",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand114_weakcombo_safewarm_c24_b20_aw20_ep45_cpsc/"
            "goal_c114_effnet_cpsc_20260624_train_retry_threads8"
        ),
        "c114officials5_weakcombo_safewarm_c24_b20_aw20_ep45",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "c114",
        "georgia",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand114_weakcombo_safewarm_c24_b20_aw20_ep45_georgia/"
            "goal_c114_effnet_georgia_20260624_train_retry_threads8"
        ),
        "c114officials5_weakcombo_safewarm_c24_b20_aw20_ep45",
        ref_meta("georgia"),
    ),
    EvalTask(
        "c115",
        "cpsc_2018",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand115_weakcombo_latentchain28_c24_b20_aw24_ep45_cpsc/"
            "goal_c115_effnet_cpsc_20260624_train_retry_threads8"
        ),
        "c115officials5_weakcombo_latentchain28_c24_b20_aw24_ep45",
        ref_meta("cpsc_2018"),
    ),
    EvalTask(
        "c115",
        "georgia",
        Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand115_weakcombo_latentchain28_c24_b20_aw24_ep45_georgia/"
            "goal_c115_effnet_georgia_20260624_train_retry_threads8"
        ),
        "c115officials5_weakcombo_latentchain28_c24_b20_aw24_ep45",
        ref_meta("georgia"),
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


def free_gpus(max_gpu: int = 6) -> list[int]:
    free: list[int] = []
    for gpu in range(max_gpu):
        try:
            if gpu_mem_mib(gpu) < 1000:
                free.append(gpu)
        except Exception:
            continue
    return free


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


def summary_command(cand: str) -> list[str]:
    return [
        str(PY),
        "scripts/agent/summarize_official_s5_composite_recovery.py",
        "--model",
        "effnet",
        "--method",
        cand,
        "--direct-json",
        f"cpsc_2018={DIRECT_ROOT / 'cpsc_2018/eval_official_s5_composite_direct_rawfirst.json'}",
        "--direct-json",
        f"georgia={DIRECT_ROOT / 'georgia/eval_official_s5_composite_direct_rawfirst.json'}",
        "--method-json",
        f"cpsc_2018={ROOT / cand / 'cpsc_2018' / f'{cand}_cpsc_2018_official_s5_composite_method_rawfirst.json'}",
        "--method-json",
        f"georgia={ROOT / cand / 'georgia' / f'{cand}_georgia_official_s5_composite_method_rawfirst.json'}",
        "--output-dir",
        str(ROOT / cand),
        "--prefix",
        f"{cand}_cpsc_georgia",
    ]


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    master_log = LOGDIR / "eval_supervisor_c111_c115_cpsc_georgia_20260624.log"
    append(master_log, "supervisor_start max_gpus=6")

    pending = list(TASKS)
    running: dict[str, tuple[EvalTask, int, subprocess.Popen]] = {}
    complete: set[str] = set()
    failed: set[str] = set()
    summaries_done: set[str] = set()

    while True:
        for key, (task, gpu, proc) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            running.pop(key, None)
            if rc == 0 and task.out_json.exists() and task.out_json.stat().st_size > 0:
                complete.add(key)
                append(master_log, f"eval_complete key={key} gpu={gpu}")
            else:
                failed.add(key)
                append(master_log, f"eval_failed key={key} gpu={gpu} rc={rc}")

        for task in list(pending):
            key = f"{task.cand}_{task.center}"
            if task.out_json.exists() and task.out_json.stat().st_size > 0:
                pending.remove(task)
                complete.add(key)
                append(master_log, f"eval_output_exists key={key}")
                continue

            available = [gpu for gpu in free_gpus(6) if all(gpu != item[1] for item in running.values())]
            if not available:
                break

            model_dir = task.model_dir()
            clean_eval = task.clean_eval()
            if model_dir is None or clean_eval is None or not clean_eval.exists() or clean_eval.stat().st_size == 0:
                failed.add(key)
                pending.remove(task)
                append(master_log, f"eval_not_ready key={key} train_root={task.train_root}")
                continue

            gpu = available[0]
            task.out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
            append(task.log_path, f"eval_start key={key} gpu={gpu} model_dir={model_dir}")
            proc = subprocess.Popen(
                eval_command(task, model_dir=model_dir, clean_eval=clean_eval),
                cwd=REPO,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running[key] = (task, gpu, proc)
            pending.remove(task)
            append(master_log, f"eval_launched key={key} pid={proc.pid} gpu={gpu}")

        for cand in sorted({task.cand for task in TASKS}):
            if cand in summaries_done:
                continue
            cpsc_key = f"{cand}_cpsc_2018"
            georgia_key = f"{cand}_georgia"
            if cpsc_key in complete and georgia_key in complete:
                log_path = LOGDIR / f"{cand}_summary_watch.log"
                log_f = log_path.open("a", encoding="utf-8")
                append(log_path, "summary_start")
                rc = subprocess.Popen(
                    summary_command(cand),
                    cwd=REPO,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                ).wait()
                if rc == 0:
                    summaries_done.add(cand)
                    append(master_log, f"summary_done cand={cand}")
                else:
                    failed.add(f"{cand}_summary")
                    append(master_log, f"summary_failed cand={cand} rc={rc}")

        if len(complete) == len(TASKS) and len(summaries_done) == len({task.cand for task in TASKS}):
            append(master_log, "supervisor_done")
            return 0
        if failed:
            append(master_log, f"supervisor_has_failures failed={sorted(failed)}")
            return 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
