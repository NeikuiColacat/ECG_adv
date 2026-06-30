#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OLD = Path(
    "/home/linbinhao/ECG_adv_data/runs/"
    "pn2021c_goal_effnet_officials5_pilots_20260624/"
    "eval_supervisor_c114_c115_20260624.py"
)


def main() -> int:
    spec = importlib.util.spec_from_file_location("c114_c115_supervisor", OLD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load supervisor module: {OLD}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    mod.ROOT = Path(
        "/home/linbinhao/ECG_adv_data/runs/"
        "pn2021c_goal_effnet_officials5_pilots_20260624_retry_threads8/"
        "eval_c114_c115_officials5"
    )
    mod.LOGDIR = Path("/dev/shm/ecg_goal_20260624_c114_c115_retry_threads8")

    retry_roots = {
        ("c114", "cpsc_2018"): Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand114_weakcombo_safewarm_c24_b20_aw20_ep45_cpsc/"
            "goal_c114_effnet_cpsc_20260624_train_retry_threads8"
        ),
        ("c114", "georgia"): Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand114_weakcombo_safewarm_c24_b20_aw20_ep45_georgia/"
            "goal_c114_effnet_georgia_20260624_train_retry_threads8"
        ),
        ("c115", "cpsc_2018"): Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand115_weakcombo_latentchain28_c24_b20_aw24_ep45_cpsc/"
            "goal_c115_effnet_cpsc_20260624_train_retry_threads8"
        ),
        ("c115", "georgia"): Path(
            "/home/linbinhao/ECG_adv_data/runs/"
            "effnet_vae_lhat_augmix_threechain_officials5_cand115_weakcombo_latentchain28_c24_b20_aw24_ep45_georgia/"
            "goal_c115_effnet_georgia_20260624_train_retry_threads8"
        ),
    }
    mod.TASKS = [
        mod.EvalTask(
            task.cand,
            task.center,
            task.gpu,
            retry_roots[(task.cand, task.center)],
            task.tag,
            task.ref_meta,
        )
        for task in mod.TASKS
    ]
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
