#!/usr/bin/env python3
"""Rerun the four-way K=500 LH-AT comparison with hard quality gate disabled.

Arms:
  1. bare PTB-XL baseline, evaluated with each center's K=500 refs excluded;
  2. VAE-only real-anchor LH-AT;
  3. matched no-token ECGTwin candidates + LH-AT;
  4. target-token ECGTwin candidates + LH-AT.

The run intentionally keeps the historical table's M=10/lambda=0.15/10-epoch
recipe, but passes --disable_quality_gate to the online AT trainer. Gate metrics
are still logged by the trainer; failed medical gates no longer skip buffer
pushes.
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
)


OUT_ROOT = Path("/root/autodl-tmp/paper_fourway_lhat_nogate_20260518")

REAL_ROOTS = [
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v2"),
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v1"),
]

TOKEN_ROOT = Path("/root/autodl-tmp/ecgtwin_prompt_token_super5")


CENTER_SPECS: dict[str, dict[str, Any]] = {
    "ningbo": {
        "seed": 20260531,
        "target_real_weight": 20.0,
        "k_anchor": 300,
        "pgd_batch": 32,
        "quick_eval_centers": ["chapman_shaoxing", "cpsc_2018_extra", "georgia", "ningbo"],
        "classes_in_scope": ["NORM", "MI", "STTC"],
        "boundary": (0.0, 1.0),
        "source_weights": {
            "no_token": "real_anchor=1.0,v45_no_token=1.0",
            "target_token": "real_anchor=1.0,v45_target_token=1.0",
        },
        "pools": {
            "no_token": TOKEN_ROOT
            / "merged_v45_norm_mi_real500_ningbo_20260504/no_token",
            "target_token": TOKEN_ROOT
            / "merged_v45_norm_mi_real500_ningbo_20260504/target_token",
        },
    },
    "chapman_shaoxing": {
        "seed": 20260522,
        "target_real_weight": 40.0,
        "k_anchor": 90,
        "pgd_batch": 16,
        "quick_eval_centers": ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"],
        "classes_in_scope": ["NORM", "MI", "STTC"],
        "boundary": (0.45, 0.70),
        "source_weights": {
            "no_token": "real_anchor=1.0,prompt_token=0.4",
            "target_token": "real_anchor=1.0,prompt_token=0.4",
        },
        "pools": {
            "no_token": TOKEN_ROOT / "merged_real_token_v2/chapman_real500_v42_vanilla",
            "target_token": TOKEN_ROOT / "merged_real_token_v2/chapman_real500_v42_target",
        },
    },
    "cpsc_2018": {
        "seed": 20260523,
        "target_real_weight": 40.0,
        "k_anchor": 90,
        "pgd_batch": 16,
        "quick_eval_centers": ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"],
        "classes_in_scope": ["NORM", "STTC"],
        "boundary": (0.45, 0.70),
        "source_weights": {
            "no_token": "real_anchor=1.0,prompt_token=0.4",
            "target_token": "real_anchor=1.0,prompt_token=0.4",
        },
        "pools": {
            "no_token": TOKEN_ROOT / "merged_real_token_v2/cpsc_real500_v42_vanilla",
            "target_token": TOKEN_ROOT / "merged_real_token_v2/cpsc_real500_v42_target",
        },
    },
    "georgia": {
        "seed": 20260521,
        "target_real_weight": 40.0,
        "k_anchor": 90,
        "pgd_batch": 16,
        "quick_eval_centers": ["chapman_shaoxing", "cpsc_2018", "georgia", "ningbo"],
        "classes_in_scope": ["NORM", "MI", "STTC"],
        "boundary": (0.45, 0.70),
        "source_weights": {
            "no_token": "real_anchor=1.0,prompt_token=0.4",
            "target_token": "real_anchor=1.0,prompt_token=0.4",
        },
        "pools": {
            "no_token": TOKEN_ROOT / "merged_real_token_v2/georgia_real500_v42_vanilla",
            "target_token": TOKEN_ROOT / "merged_real_token_v2/georgia_real500_v42_target",
        },
    },
}


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


def real_anchor_base(center: str) -> Path:
    for root in REAL_ROOTS:
        base = root / center / f"{center}_real_k500_seed42"
        if base.with_suffix(".signals.npz").exists() and base.with_suffix(".latent.npz").exists():
            return base
    searched = ", ".join(str(root / center / f"{center}_real_k500_seed42") for root in REAL_ROOTS)
    raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")


def real_anchor_paths(center: str, out_root: Path) -> dict[str, Path]:
    base = real_anchor_base(center)
    trust_path = out_root / "real_anchor_trust" / center / "real_anchor.class_trust.json"
    if not trust_path.exists():
        trust_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(base.with_suffix(".signals.npz"), allow_pickle=True) as z:
            labels = np.asarray(z["labels"], dtype=np.float32)
        counts = labels.sum(axis=0).astype(int).tolist()
        with trust_path.open("w") as f:
            json.dump(
                {
                    "tag": f"{center}_real_k500",
                    "class_trust": {"CD": 0.0, "HYP": 0.0, "MI": 1.0, "NORM": 1.0, "STTC": 1.0},
                    "label_counts": dict(zip(CLASS_NAMES.tolist(), counts)),
                    "policy": "four-way no-gate rerun; trusted classes limited to NORM/MI/STTC",
                },
                f,
                indent=2,
            )
    return {
        "signals": base.with_suffix(".signals.npz"),
        "latents": base.with_suffix(".latent.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
        "trust": trust_path,
    }


def baseline_eval(center: str, ref_meta: Path, out_root: Path, args: argparse.Namespace) -> Path:
    out_dir = out_root / "baseline_eval" / center
    eval_path = out_dir / f"{center}_baseline_exclrefs.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] baseline {center}", flush=True)
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
        str(ref_meta),
        "--output_path",
        str(eval_path),
    ]
    run(cmd, out_dir / "baseline_eval.log", dry_run=args.dry_run)
    return eval_path


def online_at(
    center: str,
    arm: str,
    synth_npz: Path,
    class_trust: Path,
    ref_meta: Path,
    target_real_npz: Path,
    out_root: Path,
    args: argparse.Namespace,
    source_weights: str,
    target_real_weight: float,
    classes_in_scope: list[str],
    quick_eval_centers: list[str],
    seed: int,
    k_anchor: int,
    pgd_batch: int,
    boundary: tuple[float, float],
) -> Path:
    tag = (
        f"{center}_{arm}_nogate_M{args.hull_m}_lam{str(args.hull_lambda).replace('.', 'p')}"
        f"_ep{args.n_epochs}_seed{seed}"
    )
    out_dir = out_root / "runs" / tag
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {tag}", flush=True)
        return eval_path

    cmd = [
        PYTHON,
        "-u",
        "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name",
        center,
        "--ref_meta_json",
        str(ref_meta),
        "--synth_npz",
        str(synth_npz),
        "--init_ckpt",
        BASELINE_CKPT,
        "--output_dir",
        str(out_dir),
        "--target_real_npz",
        str(target_real_npz),
        "--data_dir",
        PN2021_ROOT,
        "--quick_eval_centers",
        *quick_eval_centers,
        "--quick_eval_n_per_center",
        "500",
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
        str(pgd_batch),
        "--K_anchor",
        str(k_anchor),
        "--delta_init_scale",
        "0.1",
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
        "--hull_dirichlet_alpha",
        "1.0",
        "--hull_label_mode",
        "primary",
        "--source_sampling_strategy",
        "source_weighted",
        "--source_weights",
        source_weights,
        "--class_trust",
        str(class_trust),
        "--classes_in_scope",
        *classes_in_scope,
        "--boundary_prob_min",
        str(boundary[0]),
        "--boundary_prob_max",
        str(boundary[1]),
        "--adv_label_mode",
        "mixed_soft",
        "--adv_teacher_mix",
        "0.3",
        "--adv_soft_target_floor",
        "0.0",
        "--ptbxl_weight",
        "1.0",
        "--target_real_weight",
        str(target_real_weight),
        "--roundtrip_weight",
        "0.5",
        "--adv_weight",
        str(args.adv_weight),
        "--roundtrip_anchor_n",
        str(args.roundtrip_anchor_n),
        "--qab_size",
        "2048",
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
        "--es_metric",
        "target_macro_auprc",
        "--ewa_decay",
        "0.999",
        "--rescore_interval",
        "3",
        "--anchor_lambda",
        "0.05",
        "--asr_low_threshold",
        "0.3",
        "--asr_consec_low_max",
        str(args.asr_consec_low_max),
        "--einthoven_p95_max",
        "0.5",
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--crop_len",
        "1000",
        "--device",
        "cuda:0",
        "--disable_quality_gate",
    ]
    run(cmd, out_dir / "train.log", dry_run=args.dry_run)
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
        str(ref_meta),
        "--output_path",
        str(eval_path),
    ]
    run(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def extract(center: str, eval_path: Path) -> dict[str, float]:
    m = eval_metrics(eval_path)
    return {
        "target_auroc": m[f"{center}_auroc"],
        "target_auprc": m[f"{center}_auprc"],
        "pn2021_avg_auroc": m["pn2021_avg_auroc"],
        "pn2021_avg_auprc": m["pn2021_avg_auprc"],
        "ptbxl_auroc": m["ptbxl_auroc"],
        "ptbxl_auprc": m["ptbxl_auprc"],
    }


def write_summary(rows: list[dict[str, Any]], out_root: Path) -> None:
    if not rows:
        return
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["center"], r["arm"]))
    fields = [
        "center",
        "arm",
        "target_auroc",
        "target_auprc",
        "delta_target_auroc",
        "delta_target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    csv_path = summary_dir / "fourway_lhat_nogate.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_center: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_center.setdefault(row["center"], {})[row["arm"]] = row

    md_path = summary_dir / "fourway_lhat_nogate.md"
    with md_path.open("w") as f:
        f.write("# Four-Way LH-AT No-Gate Rerun\n\n")
        f.write(
            "Config: K=500 target refs, target refs excluded from evaluation, "
            "M=10, lambda=0.15, 10 epochs, `--disable_quality_gate`, "
            "minimal_resample + per_sample_global + 100Hz + 10s. Gate metrics "
            "are logged but medical/semantic gate failure does not skip buffer pushes.\n\n"
        )
        f.write("| center | baseline | VAE-only / real-anchor LH-AT | no-token + LH-AT | target-token + LH-AT |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for center in sorted(by_center):
            arms = by_center[center]
            def fmt(arm: str) -> str:
                r = arms.get(arm)
                if not r:
                    return "pending"
                if arm == "baseline":
                    return f"{r['target_auroc']:.4f} / {r['target_auprc']:.4f}"
                return (
                    f"{r['target_auroc']:.4f} / {r['target_auprc']:.4f} "
                    f"({r['delta_target_auroc'] * 100:+.2f}pp / {r['delta_target_auprc'] * 100:+.2f}pp)"
                )
            f.write(
                f"| {center} | {fmt('baseline')} | {fmt('vae_only')} | "
                f"{fmt('no_token')} | {fmt('target_token')} |\n"
            )
        f.write("\n## Paths\n\n")
        f.write(f"- CSV: `{csv_path}`\n")
        for row in rows:
            f.write(f"- {row['center']} / {row['arm']}: `{row['eval_path']}`\n")
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {md_path}", flush=True)


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    ap.add_argument(
        "--arms",
        nargs="+",
        choices=["baseline", "vae_only", "no_token", "target_token"],
        default=["baseline", "vae_only", "no_token", "target_token"],
    )
    ap.add_argument("--hull_m", type=int, default=10)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--asr_consec_low_max", type=int, default=999)
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summaries").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "fourway_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    rows: list[dict[str, Any]] = []
    for center in args.centers:
        spec = CENTER_SPECS[center]
        real = real_anchor_paths(center, OUT_ROOT)
        baseline_path = None
        if "baseline" in args.arms:
            baseline_path = baseline_eval(center, real["meta"], OUT_ROOT, args)
            if not args.dry_run:
                b = extract(center, baseline_path)
                rows.append(
                    {
                        "center": center,
                        "arm": "baseline",
                        **b,
                        "delta_target_auroc": 0.0,
                        "delta_target_auprc": 0.0,
                        "eval_path": str(baseline_path),
                    }
                )
                write_summary(rows, OUT_ROOT)
        else:
            baseline_path = baseline_eval(center, real["meta"], OUT_ROOT, args)
        if args.dry_run:
            continue
        baseline_metrics = extract(center, baseline_path)

        if "vae_only" in args.arms:
            eval_path = online_at(
                center=center,
                arm="vae_only",
                synth_npz=real["latents"],
                class_trust=real["trust"],
                ref_meta=real["meta"],
                target_real_npz=real["signals"],
                out_root=OUT_ROOT,
                args=args,
                source_weights="real_anchor=1.0",
                target_real_weight=40.0,
                classes_in_scope=["NORM", "MI", "STTC"],
                quick_eval_centers=spec["quick_eval_centers"],
                seed=20260531,
                k_anchor=300,
                pgd_batch=32,
                boundary=(0.0, 1.0),
            )
            m = extract(center, eval_path)
            rows.append(
                {
                    "center": center,
                    "arm": "vae_only",
                    **m,
                    "delta_target_auroc": m["target_auroc"] - baseline_metrics["target_auroc"],
                    "delta_target_auprc": m["target_auprc"] - baseline_metrics["target_auprc"],
                    "eval_path": str(eval_path),
                }
            )
            write_summary(rows, OUT_ROOT)

        for arm in ["no_token", "target_token"]:
            if arm not in args.arms:
                continue
            pool_dir = Path(spec["pools"][arm])
            eval_path = online_at(
                center=center,
                arm=arm,
                synth_npz=pool_dir / "merged.latent.npz",
                class_trust=pool_dir / "merged.class_trust.json",
                ref_meta=pool_dir / "merged.ref_meta.json",
                target_real_npz=real["signals"],
                out_root=OUT_ROOT,
                args=args,
                source_weights=spec["source_weights"][arm],
                target_real_weight=float(spec["target_real_weight"]),
                classes_in_scope=list(spec["classes_in_scope"]),
                quick_eval_centers=list(spec["quick_eval_centers"]),
                seed=int(spec["seed"]),
                k_anchor=int(spec["k_anchor"]),
                pgd_batch=int(spec["pgd_batch"]),
                boundary=tuple(spec["boundary"]),
            )
            m = extract(center, eval_path)
            rows.append(
                {
                    "center": center,
                    "arm": arm,
                    **m,
                    "delta_target_auroc": m["target_auroc"] - baseline_metrics["target_auroc"],
                    "delta_target_auprc": m["target_auprc"] - baseline_metrics["target_auprc"],
                    "eval_path": str(eval_path),
                }
            )
            write_summary(rows, OUT_ROOT)

    write_summary(rows, OUT_ROOT)


if __name__ == "__main__":
    main()
