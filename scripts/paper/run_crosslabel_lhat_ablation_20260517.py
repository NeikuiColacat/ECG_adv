#!/usr/bin/env python3
"""Run exact-label vs compatible-soft latent-hull AT ablations.

This is the controlled experiment for cross-label ECGTwin VAE latent-hull AT:

  1. exact_multi_hot:
     exact same multi-hot candidate pool, hard full multi-hot labels.
  2. compatible_latent_soft:
     NORM-only or abnormal-compatible candidate pool, anchor-preserving soft
     labels from optimized latent-hull coefficients.
  3. compatible_teacher_soft:
     same compatible latent pool and soft labels, blended with the frozen
     initial EfficientNet teacher probabilities.

All runs use target-center real K=500 anchors only. Reference record IDs are
excluded from the final PN2021 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.paper.run_vae_only_latenthull_sweep_20260516 import (  # noqa: E402
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PN2021_ROOT,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PTBXL_RAW,
    PYTHON,
    eval_metrics,
    prepare_subset,
)


OUT_ROOT = Path("/root/autodl-tmp/paper_crosslabel_lhat_ablation_20260517")

VARIANTS = {
    "exact_multi_hot": {
        "hull_label_mode": "exact",
        "hull_mix_label_mode": "anchor",
        "adv_label_mode": "multi_hot_hard",
        "adv_teacher_mix": 0.0,
        "description": "exact same multi-hot latent hull; adversarial labels keep full anchor multi-hot GT",
    },
    "compatible_latent_soft": {
        "hull_label_mode": "compatible",
        "hull_mix_label_mode": "anchor_soft",
        "adv_label_mode": "latent_soft",
        "adv_teacher_mix": 0.0,
        "description": "NORM-only/abnormal-compatible latent hull; labels are anchor-preserving fractional targets",
    },
    "compatible_teacher_soft": {
        "hull_label_mode": "compatible",
        "hull_mix_label_mode": "anchor_soft",
        "adv_label_mode": "latent_mixed_teacher",
        "adv_teacher_mix": 0.3,
        "description": "compatible latent soft labels blended with frozen initial teacher probabilities",
    },
}


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
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


def one_run(center: str, variant: str, args: argparse.Namespace) -> Path:
    subset = prepare_subset(center, args.k, seed=args.subset_seed)
    spec = VARIANTS[variant]
    tag = (
        f"{center}_K{args.k}_{variant}_M{args.hull_m}_"
        f"lam{str(args.hull_lambda).replace('.', 'p')}_ep{args.n_epochs}_seed{args.seed}"
    )
    out_dir = OUT_ROOT / "runs" / tag
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {tag}", flush=True)
        return eval_path

    quick_eval_centers = []
    for c in [center, "chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"]:
        if c not in quick_eval_centers:
            quick_eval_centers.append(c)

    train_cmd = [
        PYTHON,
        "-u",
        "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name", center,
        "--ref_meta_json", str(subset["meta"]),
        "--synth_npz", str(subset["latents"]),
        "--init_ckpt", BASELINE_CKPT,
        "--output_dir", str(out_dir),
        "--target_real_npz", str(subset["signals"]),
        "--data_dir", PN2021_ROOT,
        "--quick_eval_centers", *quick_eval_centers,
        "--quick_eval_n_per_center", str(args.quick_eval_n_per_center),
        "--ptbxl_raw", PTBXL_RAW,
        "--ptbxl_csv", PTBXL_CSV,
        "--ptbxl_prep", PTBXL_PREP,
        "--pgd_eps", "2.0",
        "--pgd_K", "10",
        "--pgd_batch", str(args.pgd_batch),
        "--K_anchor", str(args.k_anchor),
        "--attack_mode", "latent_hull",
        "--hull_M", str(args.hull_m),
        "--hull_lambda", str(args.hull_lambda),
        "--hull_steps", str(args.hull_steps),
        "--hull_lr", str(args.hull_lr),
        "--hull_weight_mode", "optimized",
        "--hull_dirichlet_alpha", "1.0",
        "--hull_label_mode", spec["hull_label_mode"],
        "--hull_mix_label_mode", spec["hull_mix_label_mode"],
        "--hull_label_lambda_y", str(args.hull_label_lambda_y),
        "--hull_label_positive", str(args.hull_label_positive),
        "--hull_label_negative_floor", str(args.hull_label_negative_floor),
        "--hull_label_new_class_cap", str(args.hull_label_new_class_cap),
        "--source_sampling_strategy", "source_weighted",
        "--source_weights", "real_anchor=1.0",
        "--class_trust", str(subset["trust"]),
        "--classes_in_scope", "NORM", "MI", "STTC",
        "--adv_label_mode", spec["adv_label_mode"],
        "--adv_teacher_mix", str(spec["adv_teacher_mix"]),
        "--adv_soft_target_floor", "0.0",
        "--boundary_prob_min", "0.0",
        "--boundary_prob_max", "1.0",
        "--target_real_weight", str(args.target_real_weight),
        "--ptbxl_weight", "1.0",
        "--roundtrip_weight", str(args.roundtrip_weight),
        "--roundtrip_anchor_n", str(args.roundtrip_anchor_n),
        "--adv_weight", str(args.adv_weight),
        "--lr", str(args.lr),
        "--weight_decay", "0.0001",
        "--batch_size", str(args.batch_size),
        "--n_epochs", str(args.n_epochs),
        "--patience", str(args.n_epochs),
        "--eval_every", str(args.eval_every),
        "--es_metric", "target_macro_auprc",
        "--ewa_decay", "0.999",
        "--rescore_interval", "3",
        "--qab_size", "2048",
        "--anchor_lambda", "0.05",
        "--asr_low_threshold", "0.3",
        "--asr_consec_low_max", "999",
        "--einthoven_p95_max", "0.5",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--crop_len", "1000",
        "--device", "cuda:0",
    ]
    if args.disable_quality_gate:
        train_cmd.append("--disable_quality_gate")
    run(train_cmd, out_dir / "train.log")

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
        "--exclude_ref_ids", str(subset["meta"]),
        "--output_path", str(eval_path),
    ]
    run(eval_cmd, out_dir / "eval_full.log")
    return eval_path


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    csv_path = summary_dir / "crosslabel_lhat_ablation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "crosslabel_lhat_ablation.md"
    key_fields = [
        "center",
        "variant",
        "target_auroc",
        "target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    with md_path.open("w") as f:
        f.write("# Cross-Label Latent-Hull AT Ablation\n\n")
        f.write("| " + " | ".join(key_fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(key_fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in key_fields) + " |\n")
        f.write("\n## Variant Definitions\n\n")
        for name, spec in VARIANTS.items():
            f.write(f"- `{name}`: {spec['description']}\n")
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {md_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--centers",
        nargs="+",
        default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
    )
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--hull_label_lambda_y", type=float, default=0.5)
    ap.add_argument("--hull_label_positive", type=float, default=0.95)
    ap.add_argument("--hull_label_negative_floor", type=float, default=0.0)
    ap.add_argument("--hull_label_new_class_cap", type=float, default=0.5)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--pgd_batch", type=int, default=32)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_weight", type=float, default=0.5)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--subset_seed", type=int, default=20260531)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--disable_quality_gate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for center in args.centers:
        for variant in args.variants:
            if variant not in VARIANTS:
                raise SystemExit(f"unknown variant {variant!r}; valid={sorted(VARIANTS)}")
            eval_path = one_run(center, variant, args)
            metrics = eval_metrics(eval_path)
            row = {
                "center": center,
                "variant": variant,
                "K": args.k,
                "M": args.hull_m,
                "lambda": args.hull_lambda,
                "epochs": args.n_epochs,
                "disable_quality_gate": bool(args.disable_quality_gate),
                "eval_path": str(eval_path),
            }
            row.update(metrics)
            row["target_auroc"] = metrics.get(f"{center}_auroc")
            row["target_auprc"] = metrics.get(f"{center}_auprc")
            rows.append(row)
            write_summary(rows)

    write_summary(rows)


if __name__ == "__main__":
    main()
