#!/usr/bin/env python3
"""Run one EfficientNet1DV2 stage-3 latent-branch AugMix LHAT test.

This is a narrow pilot for the current migrated host:

  target real anchors -> VAE Latent-Hull x_adv
  -> x_adv as one AugMix branch + ECG corruption chains
  -> EfficientNet1DV2 online AT
  -> full PN2021 ref-excluded evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def write_real_all_present_trust(signal_npz: Path, out_dir: Path, center: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{center}_real_k500_seed42.real_all_present.class_trust.json"
    with np.load(signal_npz) as data:
        labels = data["labels"].astype(np.float32)
    counts = labels.sum(axis=0).astype(int).tolist()
    blob = {
        "tag": f"{center}_real_k500_seed42_real_all_present",
        "center": center,
        "signal_npz": str(signal_npz),
        "class_trust": {
            cls: (1.0 if counts[i] > 0 else 0.0)
            for i, cls in enumerate(CLASS_NAMES)
        },
        "class_counts": dict(zip(CLASS_NAMES, counts)),
        "policy": (
            "VAE-only real-anchor stage-3 AugMix test; trust every Super5 "
            "class present in the real target-center K subset."
        ),
    }
    out_path.write_text(json.dumps(blob, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="cpsc_2018")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--out_root", default="")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else (
        data_root / "paper_effnet_latent_augmix_stage3_20260524"
    )
    center = args.center
    out_dir = (
        out_root
        / f"{center}_realall_targetheavy_M20_lam015_augmix_s2_wlat030"
          f"_ep{args.epochs}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_base = (
        data_root
        / "ecgtwin_prompt_token_super5/real_anchor_selected_v2"
        / center
        / f"{center}_real_k500_seed42"
    )
    signal_npz = anchor_base.with_suffix(".signals.npz")
    latent_npz = anchor_base.with_suffix(".latent.npz")
    ref_meta = anchor_base.with_suffix(".ref_meta.json")
    class_trust = write_real_all_present_trust(
        signal_npz,
        out_root / "config",
        center,
    )

    env = os.environ.copy()
    # PyTorch multiprocessing creates AF_UNIX sockets under TMPDIR. The
    # migrated data path is too long for that socket path, so use a short
    # user-owned temp directory while keeping large caches on the data disk.
    env.setdefault("TMPDIR", "/tmp/linbinhao_ecg_tmp")
    env.setdefault("XDG_CACHE_HOME", str(data_root / "cache"))
    env.setdefault("DEEPECG_NOTEBOOKS", str(REPO / "model" / "DeepECG" / "notebooks"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    python = sys.executable
    train_cmd = [
        python, "-u", "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name", center,
        "--ref_meta_json", str(ref_meta),
        "--synth_npz", str(latent_npz),
        "--target_real_npz", str(signal_npz),
        "--class_trust", str(class_trust),
        "--init_ckpt", str(data_root / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"),
        "--model_name", "efficientnet1dv2",
        "--output_dir", str(out_dir),
        "--data_dir", str(data_root / "physionet2021/training"),
        "--quick_eval_centers", center,
        "--quick_eval_n_per_center", "500",
        "--ptbxl_raw", str(data_root / "ptbxl/raw100.npy"),
        "--ptbxl_csv", str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_prep", str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--attack_mode", "latent_hull",
        "--hull_M", "20",
        "--hull_lambda", "0.15",
        "--hull_steps", "3",
        "--hull_lr", "0.25",
        "--source_sampling_strategy", "source_weighted",
        "--source_weights", "real_anchor=1.0",
        "--K_anchor", "300",
        "--pgd_batch", "32",
        "--classes_in_scope", "CD", "HYP", "MI", "NORM", "STTC",
        "--allow_hyp_cd_trust",
        "--target_real_weight", "80",
        "--adv_weight", "0.2",
        "--ptbxl_weight", "1.0",
        "--roundtrip_weight", "0.0",
        "--roundtrip_anchor_n", "0",
        "--adv_label_mode", "mixed_soft",
        "--adv_teacher_mix", "0.3",
        "--enable_latent_augmix_branch",
        "--latent_augmix_copies", "1",
        "--latent_augmix_width", "3",
        "--latent_augmix_severity", "2",
        "--latent_augmix_latent_weight_cap", "0.3",
        "--disable_quality_gate",
        "--lr", "5e-5",
        "--weight_decay", "1e-4",
        "--batch_size", "128",
        "--n_epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--eval_every", "2",
        "--es_metric", "target_macro_auprc",
        "--ewa_decay", "0.999",
        "--anchor_lambda", "0.05",
        "--asr_consec_low_max", "999",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--crop_len", "1000",
        "--device", args.device,
    ]
    eval_cmd = [
        python, "-u", "scripts/triple_labels/eval_crosscenter.py",
        "--scheme", "super5",
        "--model_dir", str(out_dir),
        "--model_name", "efficientnet1dv2",
        "--device", "cuda",
        "--crop_len", "1000",
        "--batch_size", "192",
        "--num_workers", str(args.num_workers),
        "--ptbxl_csv", str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache", str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--preprocess_mode", "minimal_resample",
        "--norm_mode", "per_sample_global",
        "--pn2021_root", str(data_root / "physionet2021"),
        "--pn2021_cache_dir", str(data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--pn2021_mmap_cache_dir", str(data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
        "--exclude_ref_ids", str(ref_meta),
        "--output_path", str(out_dir / "eval_result_v5_exclrefs_crop1000.json"),
    ]

    (out_dir / "launch_config.json").write_text(json.dumps({
        "args": vars(args),
        "class_trust": str(class_trust),
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
    }, indent=2))

    run(train_cmd, out_dir / "train_stdout.log", env)
    run(eval_cmd, out_dir / "eval_full.log", env)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
