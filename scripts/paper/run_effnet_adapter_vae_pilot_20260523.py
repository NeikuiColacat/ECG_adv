#!/usr/bin/env python3
"""EfficientNet1DV2 frozen-backbone adapter pilot for VAE-only LH-AT.

This runner compares two simple target-center adaptation arms under the same
protocol:

  real-only: PTB-XL source + K target real ECG, no adversarial stream.
  VAE-only:  same, plus real-anchor ECGTwin VAE latent-hull online AT samples.

The model update is intentionally small: freeze EfficientNet1DV2 backbone and
train classifier (+ optional final_norm affine). This avoids full-model source
forgetting and tests whether VAE-only adds value beyond target-center head
adaptation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.paper.run_vae_only_latenthull_sweep_20260516 as subset_mod
from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PN2021_ROOT,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PTBXL_RAW,
    PYTHON,
)


OUT_ROOT = Path("/root/autodl-tmp/paper_effnet_classifieronly_vae_pilot_20260523")
CLASS_NAMES = np.asarray(["CD", "HYP", "MI", "NORM", "STTC"])


def run(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def prepare_subset(center: str, k: int, seed: int) -> dict[str, Path]:
    subset_mod.OUT_ROOT = OUT_ROOT
    return subset_mod.prepare_subset(
        center=center,
        k=k,
        seed=seed,
        trust_policy="real_all_present",
    )


def present_classes(signal_npz: Path) -> list[str]:
    with np.load(signal_npz, allow_pickle=True) as d:
        labels = d["labels"].astype(np.float32, copy=False)
    counts = labels.sum(axis=0)
    return [str(CLASS_NAMES[i]) for i, n in enumerate(counts) if n > 0]


def train_one(center: str, method: str, args: argparse.Namespace) -> Path:
    subset = prepare_subset(center, args.K, args.subset_seed)
    classes = present_classes(subset["signals"])
    class_tag = "".join(c.lower() for c in classes)
    adapter_tag = "headfn" if args.train_final_norm else "head"
    if args.classifier_adapter_type != "linear":
        adapter_tag = f"{adapter_tag}_{args.classifier_adapter_type}r{args.classifier_lora_rank}"
    label_tag = args.adv_label_mode.replace("_", "")
    tag = (
        f"{center}_K{args.K}_{method}_{adapter_tag}_lam015_M20_"
        f"{label_tag}_ep{args.epochs}_{class_tag}_seed{args.seed}"
    )
    out_dir = OUT_ROOT / "runs" / tag
    eval_path = out_dir / "eval_result_v5_exclrefs_crop1000_dropzero.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {tag}")
        return eval_path

    train_cmd = [
        PYTHON,
        "-u",
        "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name", center,
        "--ref_meta_json", str(subset["meta"]),
        "--synth_npz", str(subset["latents"]),
        "--target_real_npz", str(subset["signals"]),
        "--class_trust", str(subset["trust"]),
        "--init_ckpt", BASELINE_CKPT,
        "--output_dir", str(out_dir),
        "--quick_eval_centers", center,
        "--quick_eval_n_per_center", str(args.quick_eval_n),
        "--ptbxl_raw", PTBXL_RAW,
        "--ptbxl_csv", PTBXL_CSV,
        "--ptbxl_prep", PTBXL_PREP,
        "--attack_mode", "latent_hull",
        "--hull_M", "20",
        "--hull_lambda", "0.15",
        "--hull_steps", str(args.hull_steps),
        "--hull_lr", "0.25",
        "--source_sampling_strategy", "source_weighted",
        "--source_weights", "real_anchor=1.0",
        "--K_anchor", str(args.k_anchor),
        "--pgd_batch", "32",
        "--classes_in_scope", *classes,
        "--allow_hyp_cd_trust",
        "--freeze_backbone_classifier_only",
        "--classifier_adapter_type", args.classifier_adapter_type,
        "--classifier_lora_rank", str(args.classifier_lora_rank),
        "--classifier_lora_alpha", str(args.classifier_lora_alpha),
        "--target_real_weight", str(args.target_real_weight),
        "--ptbxl_weight", "1.0",
        "--roundtrip_weight", "0.0",
        "--roundtrip_anchor_n", "0",
        "--source_logit_anchor_weight", str(args.source_logit_anchor_weight),
        "--source_logit_anchor_batches", str(args.source_logit_anchor_batches),
        "--lr", str(args.lr),
        "--weight_decay", "1e-4",
        "--batch_size", str(args.batch_size),
        "--n_epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--eval_every", "2",
        "--es_metric", "target_macro_auprc",
        "--ewa_decay", "0.999",
        "--anchor_lambda", "0.05",
        "--asr_consec_low_max", "999",
        "--disable_quality_gate",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--crop_len", "1000",
        "--device", args.device,
    ]
    if args.train_final_norm:
        train_cmd.append("--classifier_only_train_final_norm")
    if method == "real_only":
        train_cmd.extend(["--disable_adv_stream", "--adv_weight", "0.0"])
    elif method == "vae_only":
        train_cmd.extend([
            "--adv_label_mode", args.adv_label_mode,
            "--adv_teacher_mix", str(args.adv_teacher_mix),
            "--adv_soft_target_floor", str(args.adv_soft_target_floor),
            "--adv_weight", str(args.adv_weight),
        ])
    else:
        raise ValueError(f"unknown method={method!r}")

    run(train_cmd, out_dir / "train.log", dry_run=args.dry_run)
    if args.dry_run:
        return eval_path

    eval_cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme", "super5",
        "--model_dir", str(out_dir),
        "--device", "cuda",
        "--crop_len", "1000",
        "--batch_size", "192",
        "--num_workers", str(args.num_workers),
        "--ptbxl_cache", PTBXL_PREP,
        "--preprocess_mode", "minimal_resample",
        "--norm_mode", "per_sample_global",
        "--pn2021_cache_dir", PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir", PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
        "--exclude_ref_ids", str(subset["meta"]),
        "--output_path", str(eval_path),
    ]
    run(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def metrics_from_eval(eval_path: Path, center: str) -> dict[str, Any]:
    with eval_path.open() as f:
        d = json.load(f)
    ptbxl = d["ptbxl_test"]
    pn = d["pn2021"]
    target = pn["per_center"][center]
    return {
        "ptbxl_auroc": ptbxl["macro_auroc"],
        "ptbxl_auprc": ptbxl["macro_auprc"],
        "target_auroc": target["macro_auroc"],
        "target_auprc": target["macro_auprc"],
        "target_dropzero_auroc": target.get("drop_all_zero_macro_auroc"),
        "target_dropzero_auprc": target.get("drop_all_zero_macro_auprc"),
        "target_n": target.get("n_records"),
        "target_dropzero_n": target.get("drop_all_zero_n_records"),
        "pn_avg_auroc": pn["avg_macro_auroc"],
        "pn_avg_auprc": pn["avg_macro_auprc"],
        "pn_avg_dropzero_auroc": pn.get("avg_drop_all_zero_macro_auroc"),
        "pn_avg_dropzero_auprc": pn.get("avg_drop_all_zero_macro_auprc"),
    }


def write_summary(rows: list[dict[str, Any]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "effnet_adapter_vae_pilot_20260523.csv"
    fields = [
        "center", "K", "method", "adapter", "classes", "ptbxl_auroc", "ptbxl_auprc",
        "target_auroc", "target_auprc", "target_dropzero_auroc",
        "target_dropzero_auprc", "target_n", "target_dropzero_n",
        "pn_avg_auroc", "pn_avg_auprc",
        "pn_avg_dropzero_auroc", "pn_avg_dropzero_auprc", "eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "effnet_adapter_vae_pilot_20260523.md"
    with md_path.open("w") as f:
        f.write("# EfficientNet Adapter VAE-only Pilot\n\n")
        f.write("| center | K | method | adapter | PTB-XL | target | target drop-zero | 7-center avg |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {center} | {K} | {method} | {adapter} | "
                "{ptbxl_auroc:.4f} / {ptbxl_auprc:.4f} | "
                "{target_auroc:.4f} / {target_auprc:.4f} | "
                "{target_dropzero_auroc:.4f} / {target_dropzero_auprc:.4f} | "
                "{pn_avg_auroc:.4f} / {pn_avg_auprc:.4f} |\n".format(**r)
            )
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "georgia"])
    ap.add_argument("--methods", nargs="+", choices=["real_only", "vae_only"], default=["real_only", "vae_only"])
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--subset_seed", type=int, default=20260531)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--quick_eval_n", type=int, default=500)
    ap.add_argument("--k_anchor", type=int, default=150)
    ap.add_argument("--hull_steps", type=int, default=10)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--adv_weight", type=float, default=20.0)
    ap.add_argument(
        "--adv_label_mode",
        choices=[
            "hard", "multi_hot_hard", "latent_soft", "mixed_soft",
            "teacher_soft", "latent_mixed_teacher",
        ],
        default="multi_hot_hard",
    )
    ap.add_argument("--adv_teacher_mix", type=float, default=0.7)
    ap.add_argument("--adv_soft_target_floor", type=float, default=0.0)
    ap.add_argument("--source_logit_anchor_weight", type=float, default=0.2)
    ap.add_argument("--source_logit_anchor_batches", type=int, default=10)
    ap.add_argument("--train_final_norm", action="store_true", default=True)
    ap.add_argument("--no_train_final_norm", action="store_false", dest="train_final_norm")
    ap.add_argument("--classifier_adapter_type", choices=["linear", "lora"], default="linear")
    ap.add_argument("--classifier_lora_rank", type=int, default=16)
    ap.add_argument("--classifier_lora_alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)

    (OUT_ROOT / "summaries").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "effnet_adapter_pilot_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    rows: list[dict[str, Any]] = []
    for center in args.centers:
        for method in args.methods:
            eval_path = train_one(center, method, args)
            if args.dry_run:
                continue
            subset = prepare_subset(center, args.K, args.subset_seed)
            row = {
                "center": center,
                "K": args.K,
                "method": method,
                "adapter": (
                    ("head+final_norm" if args.train_final_norm else "head")
                    if args.classifier_adapter_type == "linear"
                    else (
                        ("head+final_norm" if args.train_final_norm else "head")
                        + f"+lora_r{args.classifier_lora_rank}"
                    )
                ),
                "classes": " ".join(present_classes(subset["signals"])),
                "eval_path": str(eval_path),
            }
            row.update(metrics_from_eval(eval_path, center))
            rows.append(row)
            write_summary(rows)
    if rows:
        write_summary(rows)


if __name__ == "__main__":
    main()
