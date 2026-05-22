#!/usr/bin/env python3
"""Run the current VAE-only real-anchor LH-AT K curve.

This is the paper-facing K-shot curve for the current main method:

  * no ECGTwin DiT synthetic samples;
  * no center token samples;
  * target-center real ECGTwin VAE latents only;
  * quality gate disabled by default;
  * target ref ids excluded from the target-center evaluation.

For every center/K pair, the runner evaluates the frozen PTB-XL baseline with
the same excluded refs, then trains/evaluates LH-AT and writes a combined CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    BASELINE_CKPT,
    CLASS_NAMES,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PN2021_ROOT,
    PROJECT_ROOT,
    PTBXL_CSV,
    PTBXL_PREP,
    PTBXL_RAW,
    PYTHON,
    eval_metrics,
    proportional_stratified_indices,
)
from scripts.paper.run_vae_only_latenthull_sweep_20260516 import _find_real_anchor_base  # noqa: E402


OUT_ROOT = Path("/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518")
np_load = np.load
np_savez_compressed = np.savez_compressed


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


def prepare_subset(center: str, k: int, subset_seed: int) -> dict[str, Path]:
    base = _find_real_anchor_base(center)
    signal_path = base.with_suffix(".signals.npz")
    latent_path = base.with_suffix(".latent.npz")
    meta_path = base.with_suffix(".ref_meta.json")

    out_dir = OUT_ROOT / "subsets" / center / f"k{k}_seed{subset_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_signal = out_dir / f"{center}_real_k{k}_seed{subset_seed}.signals.npz"
    out_latent = out_dir / f"{center}_real_k{k}_seed{subset_seed}.latent.npz"
    out_meta = out_dir / f"{center}_real_k{k}_seed{subset_seed}.ref_meta.json"
    out_trust = out_dir / f"{center}_real_k{k}_seed{subset_seed}.class_trust.json"
    if out_signal.exists() and out_latent.exists() and out_meta.exists() and out_trust.exists():
        return {"signals": out_signal, "latents": out_latent, "meta": out_meta, "trust": out_trust}

    with np_load(signal_path, allow_pickle=True) as sig_npz:
        signals = sig_npz["signals"].astype("float32", copy=False)
        labels = sig_npz["labels"].astype("float32", copy=False)
        record_ids = sig_npz["record_ids"].astype(str)
    with np_load(latent_path, allow_pickle=True) as lat_npz:
        latents = lat_npz["latents"].astype("float32", copy=False)
        source_indices = (
            lat_npz["source_indices"]
            if "source_indices" in lat_npz
            else np.arange(len(labels), dtype=np.int64)
        )
        primary_class = (
            lat_npz["primary_class"].astype(str)
            if "primary_class" in lat_npz
            else CLASS_NAMES[labels.argmax(axis=1)]
        )

    if k > len(labels):
        raise ValueError(f"K={k} exceeds available anchors for {center}: {len(labels)}")
    idx = proportional_stratified_indices(labels, k, seed=subset_seed)

    np_savez_compressed(
        out_signal,
        signals=signals[idx],
        labels=labels[idx],
        record_ids=record_ids[idx],
        center_name=center,
        class_names=CLASS_NAMES,
    )
    np_savez_compressed(
        out_latent,
        latents=latents[idx],
        labels=labels[idx],
        center_name=center,
        class_names=CLASS_NAMES,
        source_indices=np.asarray(source_indices)[idx],
        record_ids=record_ids[idx],
        primary_class=primary_class[idx],
        source_ids=np.zeros((len(idx),), dtype=np.int16),
        source_names=np.asarray(["real_anchor"]),
        source_local_indices=np.arange(len(idx), dtype=np.int32),
    )
    with out_meta.open("w") as f:
        json.dump(
            {
                "center": center,
                "K": int(k),
                "selection_seed": int(subset_seed),
                "parent": str(meta_path),
                "ref_record_ids": record_ids[idx].tolist(),
                "source_indices": np.asarray(source_indices)[idx].astype(int).tolist(),
                "policy": "proportional stratified subset from current available K=500 real anchors",
            },
            f,
            indent=2,
        )
    counts = labels[idx].sum(axis=0).astype(int).tolist()
    with out_trust.open("w") as f:
        json.dump(
            {
                "tag": f"{center}_real_k{k}",
                "class_trust": {"CD": 0.0, "HYP": 0.0, "MI": 1.0, "NORM": 1.0, "STTC": 1.0},
                "label_counts": dict(zip(CLASS_NAMES.tolist(), counts)),
                "policy": "VAE-only real-anchor K curve; HYP/CD disabled for adv generation",
            },
            f,
            indent=2,
        )
    return {"signals": out_signal, "latents": out_latent, "meta": out_meta, "trust": out_trust}


def baseline_eval(center: str, subset: dict[str, Path], args: argparse.Namespace) -> Path:
    out_dir = OUT_ROOT / "baseline_eval" / center
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / f"{center}_K{args.current_k}_baseline_exclrefs.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] baseline {center} K={args.current_k}", flush=True)
        return eval_path
    cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(Path(BASELINE_CKPT).parent),
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
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
        "--skip_mimic",
        "--exclude_ref_ids",
        str(subset["meta"]),
        "--output_path",
        str(eval_path),
    ]
    run(cmd, out_dir / f"{center}_K{args.current_k}_baseline_eval.log", dry_run=args.dry_run)
    return eval_path


def train_eval(center: str, subset: dict[str, Path], args: argparse.Namespace) -> Path:
    tag = (
        f"{center}_K{args.current_k}_lambda{str(args.hull_lambda).replace('.', 'p')}"
        f"_M{args.hull_m}_ep{args.n_epochs}_seed{args.seed}"
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
        "--center_name",
        center,
        "--ref_meta_json",
        str(subset["meta"]),
        "--synth_npz",
        str(subset["latents"]),
        "--init_ckpt",
        BASELINE_CKPT,
        "--output_dir",
        str(out_dir),
        "--target_real_npz",
        str(subset["signals"]),
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
        "--pgd_K",
        "10",
        "--pgd_batch",
        "32",
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
        args.hull_label_mode,
        "--source_sampling_strategy",
        "source_weighted",
        "--source_weights",
        "real_anchor=1.0",
        "--class_trust",
        str(subset["trust"]),
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
        "5e-05",
        "--weight_decay",
        "0.0001",
        "--batch_size",
        "128",
        "--n_epochs",
        str(args.n_epochs),
        "--patience",
        str(args.n_epochs),
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
        "cuda:0",
        "--disable_quality_gate",
    ]
    run(train_cmd, out_dir / "train.log", dry_run=args.dry_run)
    if args.dry_run:
        return eval_path

    eval_cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(out_dir),
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
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
        "--skip_mimic",
        "--exclude_ref_ids",
        str(subset["meta"]),
        "--output_path",
        str(eval_path),
    ]
    run(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def _target_metrics(center: str, eval_path: Path) -> dict[str, float]:
    metrics = eval_metrics(eval_path)
    return {
        "target_auroc": metrics[f"{center}_auroc"],
        "target_auprc": metrics[f"{center}_auprc"],
        "pn2021_avg_auroc": metrics["pn2021_avg_auroc"],
        "pn2021_avg_auprc": metrics["pn2021_avg_auprc"],
        "ptbxl_auroc": metrics["ptbxl_auroc"],
        "ptbxl_auprc": metrics["ptbxl_auprc"],
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r["center"]), int(r["K"])))

    csv_path = summary_dir / "vae_only_lhat_kcurve.csv"
    fields = [
        "center",
        "K",
        "baseline_target_auroc",
        "baseline_target_auprc",
        "lhat_target_auroc",
        "lhat_target_auprc",
        "delta_target_auroc",
        "delta_target_auprc",
        "lhat_pn2021_avg_auroc",
        "lhat_pn2021_avg_auprc",
        "lhat_ptbxl_auroc",
        "lhat_ptbxl_auprc",
        "baseline_eval_path",
        "lhat_eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "vae_only_lhat_kcurve.md"
    with md_path.open("w") as f:
        f.write("# VAE-Only Real-Anchor LH-AT K Curve\n\n")
        f.write(
            "Config: no DiT synthetic, no center token, quality gate disabled, "
            "M=20, lambda=0.15, epochs=10 by default. Target-center metrics "
            "exclude the K adaptation refs.\n\n"
        )
        f.write(
            "| center | K | baseline target | LH-AT target | delta target | "
            "PN2021 avg after LH-AT | PTB-XL after LH-AT |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {center} | {K} | {baseline_target_auroc:.4f} / {baseline_target_auprc:.4f} "
                "| {lhat_target_auroc:.4f} / {lhat_target_auprc:.4f} "
                "| {delta_target_auroc:+.4f} / {delta_target_auprc:+.4f} "
                "| {lhat_pn2021_avg_auroc:.4f} / {lhat_pn2021_avg_auprc:.4f} "
                "| {lhat_ptbxl_auroc:.4f} / {lhat_ptbxl_auprc:.4f} |\n".format(**r)
            )
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {md_path}", flush=True)


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100, 200, 300, 400, 500])
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--subset_seed", type=int, default=20260531)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--hull_label_mode", choices=["primary", "exact"], default="primary")
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)

    rows: list[dict[str, object]] = []
    config_path = OUT_ROOT / "kcurve_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)

    for center in args.centers:
        for k in args.ks:
            args.current_k = int(k)
            subset = prepare_subset(center, k, args.subset_seed)
            baseline_path = baseline_eval(center, subset, args)
            lhat_path = train_eval(center, subset, args)
            if args.dry_run:
                continue
            b = _target_metrics(center, baseline_path)
            m = _target_metrics(center, lhat_path)
            row = {
                "center": center,
                "K": int(k),
                "baseline_target_auroc": b["target_auroc"],
                "baseline_target_auprc": b["target_auprc"],
                "lhat_target_auroc": m["target_auroc"],
                "lhat_target_auprc": m["target_auprc"],
                "delta_target_auroc": m["target_auroc"] - b["target_auroc"],
                "delta_target_auprc": m["target_auprc"] - b["target_auprc"],
                "lhat_pn2021_avg_auroc": m["pn2021_avg_auroc"],
                "lhat_pn2021_avg_auprc": m["pn2021_avg_auprc"],
                "lhat_ptbxl_auroc": m["ptbxl_auroc"],
                "lhat_ptbxl_auprc": m["ptbxl_auprc"],
                "baseline_eval_path": str(baseline_path),
                "lhat_eval_path": str(lhat_path),
            }
            rows.append(row)
            write_summary(rows)

    if rows:
        write_summary(rows)


if __name__ == "__main__":
    main()
