#!/usr/bin/env python3
"""Run VAE-only real-anchor online AT on five PTB-XL benchmark backbones.

For each model:
  1. train on PTB-XL Super5 with the current paper input protocol;
  2. evaluate the source model on PN2021 with each target center's K=500
     adaptation records excluded;
  3. run VAE-only real-anchor latent-hull online AT for each target center;
  4. evaluate the adapted checkpoint on full PN2021 with the same exclusion.

This script intentionally uses no DiT synthetic samples and no center-token
samples.  It tests whether the VAE latent-manifold online AT gain transfers
beyond EfficientNet1DV2.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    CLASS_NAMES,
    DATA_ROOT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PN2021_ROOT,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PTBXL_RAW,
    PYTHON,
    eval_metrics,
)


OUT_ROOT = DATA_ROOT / "paper_multibackbone_vae_only_lhat_20260522"
REAL_ROOTS = [
    DATA_ROOT / "ecgtwin_prompt_token_super5" / "real_anchor_selected_v2",
    DATA_ROOT / "ecgtwin_prompt_token_super5" / "real_anchor_selected_v1",
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v2"),
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v1"),
]

DEFAULT_MODELS = [
    "benchmark_fcn_wang",
    "benchmark_resnet1d_wang",
    "benchmark_inception1d",
    "benchmark_lstm",
    "benchmark_xresnet1d101",
]

MODEL_LR = {
    "benchmark_lstm": 1e-3,
}
SUMMARY_SUFFIX = ""


def run(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))
    env.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / "cache"))
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


def real_anchor_base(center: str) -> Path:
    for root in REAL_ROOTS:
        base = root / center / f"{center}_real_k500_seed42"
        if base.with_suffix(".signals.npz").exists() and base.with_suffix(".latent.npz").exists():
            return base
    searched = ", ".join(str(root / center / f"{center}_real_k500_seed42") for root in REAL_ROOTS)
    raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")


def anchor_paths(center: str) -> dict[str, Path]:
    base = real_anchor_base(center)
    return {
        "signals": base.with_suffix(".signals.npz"),
        "latents": base.with_suffix(".latent.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
        "trust": base.with_suffix(".class_trust.json"),
    }


def train_source_model(model_name: str, args: argparse.Namespace) -> Path:
    out_dir = OUT_ROOT / "source_models" / model_name
    if (out_dir / "best_model.pt").exists() and (out_dir / "train_result.json").exists() and not args.force_train:
        print(f"[skip] source train {model_name}", flush=True)
        return out_dir
    lr = MODEL_LR.get(model_name, args.lr)
    cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/train_ptbxl.py",
        "--scheme",
        "super5",
        "--model_name",
        model_name,
        "--data_path",
        PTBXL_RAW,
        "--csv_path",
        PTBXL_CSV,
        "--output_dir",
        str(out_dir),
        "--cache_path",
        PTBXL_PREP,
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.train_batch_size),
        "--epochs",
        str(args.train_epochs),
        "--lr",
        str(lr),
        "--weight_decay",
        "0.01",
        "--cosine_tmax",
        str(args.cosine_tmax),
        "--patience",
        str(args.train_patience),
        "--num_workers",
        str(args.num_workers),
        "--checkpoint_metric",
        "auprc",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    run(cmd, out_dir / "train.log", dry_run=args.dry_run)
    return out_dir


def eval_model(
    model_name: str,
    model_dir: Path,
    center: str,
    ref_meta: Path,
    tag: str,
    args: argparse.Namespace,
) -> Path:
    out_dir = OUT_ROOT / "evals" / model_name / center
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / f"{tag}{args.eval_suffix}.json"
    if eval_path.exists() and not args.force_eval:
        print(f"[skip] eval {model_name} {center} {tag}", flush=True)
        return eval_path
    cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_name",
        model_name,
        "--model_dir",
        str(model_dir),
        "--device",
        args.device,
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_csv",
        PTBXL_CSV,
        "--ptbxl_cache",
        PTBXL_PREP,
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir",
        PN2021_MMAP_CACHE_DIR,
        "--pn2021_root",
        str(DATA_ROOT / "physionet2021"),
        "--skip_mimic",
        "--exclude_ref_ids",
        str(ref_meta),
        "--output_path",
        str(eval_path),
    ]
    run(cmd, out_dir / f"{tag}.log", dry_run=args.dry_run)
    return eval_path


def online_at(
    model_name: str,
    source_dir: Path,
    center: str,
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> Path:
    tag = (
        f"{model_name}_{center}_K500_vaeonly_M{args.hull_m}"
        f"_lam{str(args.hull_lambda).replace('.', 'p')}_ep{args.at_epochs}"
        f"_seed{args.seed}"
    )
    out_dir = OUT_ROOT / "online_at" / tag
    if (out_dir / "best_model.pt").exists() and (out_dir / "train_result.json").exists() and not args.force_at:
        print(f"[skip] online AT {tag}", flush=True)
        return out_dir

    quick_eval_centers = []
    for c in [center, "chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"]:
        if c not in quick_eval_centers:
            quick_eval_centers.append(c)

    cmd = [
        PYTHON,
        "-u",
        "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name",
        center,
        "--ref_meta_json",
        str(paths["meta"]),
        "--synth_npz",
        str(paths["latents"]),
        "--init_ckpt",
        str(source_dir / "best_model.pt"),
        "--model_name",
        model_name,
        "--output_dir",
        str(out_dir),
        "--target_real_npz",
        str(paths["signals"]),
        "--data_dir",
        PN2021_ROOT,
        "--quick_eval_centers",
        *quick_eval_centers,
        "--quick_eval_n_per_center",
        str(args.quick_eval_n_per_center),
        "--ptbxl_raw",
        PTBXL_RAW,
        "--ptbxl_csv",
        PTBXL_CSV,
        "--ptbxl_prep",
        PTBXL_PREP,
        "--pgd_eps",
        "2.0",
        "--pgd_batch",
        str(args.pgd_batch),
        "--K_anchor",
        str(args.k_anchor),
        "--attack_mode",
        "latent_hull",
        "--hull_M",
        str(args.hull_m),
        "--hull_lambda",
        str(args.hull_lambda),
        "--hull_steps",
        str(args.hull_steps),
        "--hull_lr",
        str(args.hull_lr),
        "--hull_weight_mode",
        "optimized",
        "--hull_label_mode",
        "primary",
        "--source_sampling_strategy",
        "source_weighted",
        "--source_weights",
        "real_anchor=1.0",
        "--class_trust",
        str(paths["trust"]),
        "--classes_in_scope",
        "NORM",
        "MI",
        "STTC",
        "--adv_label_mode",
        "mixed_soft",
        "--adv_teacher_mix",
        "0.3",
        "--adv_weight",
        str(args.adv_weight),
        "--target_real_weight",
        str(args.target_real_weight),
        "--ptbxl_weight",
        "1.0",
        "--roundtrip_weight",
        "0.5",
        "--roundtrip_anchor_n",
        str(args.roundtrip_anchor_n),
        "--lr",
        str(args.at_lr),
        "--weight_decay",
        "0.0001",
        "--batch_size",
        str(args.at_batch_size),
        "--n_epochs",
        str(args.at_epochs),
        "--patience",
        str(args.at_epochs),
        "--eval_every",
        "2",
        "--es_metric",
        "target_macro_auprc",
        "--ewa_decay",
        "0.999",
        "--rescore_interval",
        "3",
        "--qab_size",
        "2048",
        "--anchor_lambda",
        "0.05",
        "--asr_low_threshold",
        "0.3",
        "--asr_consec_low_max",
        "999",
        "--einthoven_p95_max",
        "0.5",
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--crop_len",
        "1000",
        "--device",
        args.device,
        "--disable_quality_gate",
    ]
    run(cmd, out_dir / "train.log", dry_run=args.dry_run)
    return out_dir


def target_metrics(center: str, eval_path: Path) -> dict[str, float]:
    m = eval_metrics(eval_path)
    return {
        "target_auroc": m[f"{center}_auroc"],
        "target_auprc": m[f"{center}_auprc"],
        "pn2021_avg_auroc": m["pn2021_avg_auroc"],
        "pn2021_avg_auprc": m["pn2021_avg_auprc"],
        "ptbxl_auroc": m["ptbxl_auroc"],
        "ptbxl_auprc": m["ptbxl_auprc"],
    }


def write_summary(rows: list[dict[str, Any]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"multibackbone_vae_only_lhat{SUMMARY_SUFFIX}.csv"
    fields = [
        "model_name",
        "center",
        "baseline_target_auroc",
        "baseline_target_auprc",
        "lhat_target_auroc",
        "lhat_target_auprc",
        "delta_target_auroc",
        "delta_target_auprc",
        "baseline_pn2021_avg_auroc",
        "baseline_pn2021_avg_auprc",
        "lhat_pn2021_avg_auroc",
        "lhat_pn2021_avg_auprc",
        "baseline_eval_path",
        "lhat_eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_path = summary_dir / f"multibackbone_vae_only_lhat{SUMMARY_SUFFIX}.md"
    with md_path.open("w") as f:
        f.write("# Multi-Backbone VAE-Only Online AT\n\n")
        f.write(
            "Protocol: PTB-XL Super5 source training, minimal_resample, "
            "per_sample_global, 100Hz, 1000 samples, then K=500 target-center "
            "real-anchor VAE-only latent-hull online AT. Target-center eval "
            "excludes the 500 adaptation records.\n\n"
        )
        f.write("| model | center | baseline target | online AT target | delta | PN2021 avg baseline -> AT |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda x: (x["model_name"], x["center"])):
            f.write(
                "| {model_name} | {center} | "
                "{baseline_target_auroc:.4f} / {baseline_target_auprc:.4f} | "
                "{lhat_target_auroc:.4f} / {lhat_target_auprc:.4f} | "
                "{delta_target_auroc:+.4f} / {delta_target_auprc:+.4f} | "
                "{baseline_pn2021_avg_auroc:.4f} / {baseline_pn2021_avg_auprc:.4f} -> "
                "{lhat_pn2021_avg_auroc:.4f} / {lhat_pn2021_avg_auprc:.4f} |\n".format(**r)
            )
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {md_path}", flush=True)


def main() -> None:
    global OUT_ROOT, SUMMARY_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    ap.add_argument("--train_epochs", type=int, default=50)
    ap.add_argument("--train_patience", type=int, default=8)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--cosine_tmax", type=int, default=20)
    ap.add_argument("--at_epochs", type=int, default=10)
    ap.add_argument("--at_batch_size", type=int, default=128)
    ap.add_argument("--at_lr", type=float, default=5e-5)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--pgd_batch", type=int, default=32)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    ap.add_argument("--eval_suffix", default="",
                    help="Optional suffix before .json for preserving older "
                         "evaluation files, e.g. _v5_20260524.")
    ap.add_argument("--force_at", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)
    SUMMARY_SUFFIX = args.eval_suffix
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        source_dir = train_source_model(model_name, args)
        for center in args.centers:
            paths = anchor_paths(center)
            baseline_eval = eval_model(
                model_name, source_dir, center, paths["meta"], "baseline_exclrefs", args
            )
            at_dir = online_at(model_name, source_dir, center, paths, args)
            at_eval = eval_model(
                model_name, at_dir, center, paths["meta"], "vaeonly_lhat_exclrefs", args
            )
            if args.dry_run:
                continue
            b = target_metrics(center, baseline_eval)
            m = target_metrics(center, at_eval)
            rows.append({
                "model_name": model_name,
                "center": center,
                "baseline_target_auroc": b["target_auroc"],
                "baseline_target_auprc": b["target_auprc"],
                "lhat_target_auroc": m["target_auroc"],
                "lhat_target_auprc": m["target_auprc"],
                "delta_target_auroc": m["target_auroc"] - b["target_auroc"],
                "delta_target_auprc": m["target_auprc"] - b["target_auprc"],
                "baseline_pn2021_avg_auroc": b["pn2021_avg_auroc"],
                "baseline_pn2021_avg_auprc": b["pn2021_avg_auprc"],
                "lhat_pn2021_avg_auroc": m["pn2021_avg_auroc"],
                "lhat_pn2021_avg_auprc": m["pn2021_avg_auprc"],
                "baseline_eval_path": str(baseline_eval),
                "lhat_eval_path": str(at_eval),
            })
            write_summary(rows)
    if rows:
        write_summary(rows)


if __name__ == "__main__":
    main()
