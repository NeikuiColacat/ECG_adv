#!/usr/bin/env python3
"""Run VAE-only real-anchor Latent-Hull AT sweeps for the paper route.

This intentionally excludes ECGTwin DiT synthetic samples and center tokens.
Every adversarial sample is produced by mixing target-center real ECGTwin VAE
latents from the same primary super5 class.
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

from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (
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


OUT_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516")
REAL_ROOTS = [
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v2"),
    Path("/root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v1"),
]
np_load = np.load
np_savez_compressed = np.savez_compressed


VARIANTS = {
    "lambda015": {
        "hull_lambda": 0.15,
        "include_anchor": False,
        "description": "anchored LH-AT: z_adv=(1-lambda)z0+lambda*sum softmax(a_i)z_i",
    },
    "convex_neighbors": {
        "hull_lambda": 1.0,
        "include_anchor": False,
        "description": "no-lambda neighbor hull: z_adv=sum softmax(a_i)z_i, z0 excluded",
    },
    "convex_anchor": {
        "hull_lambda": 1.0,
        "include_anchor": True,
        "description": "no-lambda anchor-included hull: z_adv=sum softmax(a_i)z_i, z0 is candidate 0",
    },
}


def _find_real_anchor_base(center: str) -> Path:
    for root in REAL_ROOTS:
        base = root / center / f"{center}_real_k500_seed42"
        if base.with_suffix(".signals.npz").exists() and base.with_suffix(".latent.npz").exists():
            return base
    searched = ", ".join(str(root / center / f"{center}_real_k500_seed42") for root in REAL_ROOTS)
    raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")


def prepare_subset(center: str, k: int, seed: int) -> dict[str, Path]:
    """Prepare a deterministic K-subset, supporting mixed v1/v2 anchor roots."""
    base = _find_real_anchor_base(center)
    signal_path = base.with_suffix(".signals.npz")
    latent_path = base.with_suffix(".latent.npz")
    meta_path = base.with_suffix(".ref_meta.json")

    out_dir = OUT_ROOT / "subsets" / center / f"k{k}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_signal = out_dir / f"{center}_real_k{k}_seed{seed}.signals.npz"
    out_latent = out_dir / f"{center}_real_k{k}_seed{seed}.latent.npz"
    out_meta = out_dir / f"{center}_real_k{k}_seed{seed}.ref_meta.json"
    out_trust = out_dir / f"{center}_real_k{k}_seed{seed}.class_trust.json"
    if out_signal.exists() and out_latent.exists() and out_meta.exists() and out_trust.exists():
        return {"signals": out_signal, "latents": out_latent, "meta": out_meta, "trust": out_trust}

    with np_load(signal_path) as sig_npz:
        signals = sig_npz["signals"].astype("float32", copy=False)
        labels = sig_npz["labels"].astype("float32", copy=False)
        record_ids = sig_npz["record_ids"].astype(str)
    with np_load(latent_path) as lat_npz:
        latents = lat_npz["latents"].astype("float32", copy=False)
        source_indices = lat_npz["source_indices"] if "source_indices" in lat_npz else list(range(len(labels)))
        primary_class = lat_npz["primary_class"].astype(str) if "primary_class" in lat_npz else CLASS_NAMES[labels.argmax(axis=1)]

    if k > len(labels):
        raise ValueError(f"K={k} exceeds available anchors for {center}: {len(labels)}")
    idx = proportional_stratified_indices(labels, k, seed=seed)
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
        source_indices=source_indices[idx],
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
                "selection_seed": int(seed),
                "parent": str(meta_path),
                "ref_record_ids": record_ids[idx].tolist(),
                "source_indices": np.asarray(source_indices)[idx].astype(int).tolist(),
                "policy": "proportional stratified subset from available v2/v1 real anchors",
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
                "policy": "VAE-only real-anchor sweep; HYP/CD disabled to match trusted-class setting",
            },
            f,
            indent=2,
        )
    return {"signals": out_signal, "latents": out_latent, "meta": out_meta, "trust": out_trust}


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


def one_run(
    center: str,
    k: int,
    variant: str,
    hull_m: int,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
) -> Path:
    subset = prepare_subset(center, k, seed=args.subset_seed)
    spec = VARIANTS[variant]
    weight_suffix = "" if args.hull_weight_mode == "optimized" else f"_w{args.hull_weight_mode}"
    tag = f"{center}_K{k}_{variant}_M{hull_m}{weight_suffix}_ep{epochs}_seed{seed}"
    out_dir = OUT_ROOT / "runs" / tag
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {tag}")
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
        "32",
        "--K_anchor",
        str(args.k_anchor),
        "--attack_mode",
        "latent_hull",
        "--hull_M",
        str(hull_m),
        "--hull_lambda",
        str(spec["hull_lambda"]),
        "--hull_steps",
        str(args.hull_steps),
        "--hull_lr",
        str(args.hull_lr),
        "--hull_weight_mode",
        args.hull_weight_mode,
        "--hull_dirichlet_alpha",
        str(args.hull_dirichlet_alpha),
        "--hull_label_mode",
        "primary",
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
        str(epochs),
        "--patience",
        str(epochs),
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
        str(seed),
        "--crop_len",
        "1000",
        "--device",
        "cuda:0",
    ]
    if spec["include_anchor"]:
        train_cmd.append("--hull_include_anchor")
    if not args.enable_quality_gate:
        train_cmd.append("--disable_quality_gate")

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
        "192",
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


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "vae_only_latenthull_sweep.csv"
    preferred_fields = [
        "center",
        "K",
        "variant",
        "hull_M",
        "epochs",
        "target_auroc",
        "target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    all_fields = sorted({key for row in rows for key in row.keys()})
    fields = preferred_fields + [key for key in all_fields if key not in preferred_fields]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "vae_only_latenthull_sweep.md"
    with md_path.open("w") as f:
        f.write("# VAE-Only Real-Anchor Latent-Hull Sweep\n\n")
        f.write("| center | K | variant | M | epochs | target AUROC/AUPRC | PN2021 avg | PTB-XL |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {center} | {K} | {variant} | {hull_M} | {epochs} | "
                "{target_auroc:.4f} / {target_auprc:.4f} | "
                "{pn2021_avg_auroc:.4f} / {pn2021_avg_auprc:.4f} | "
                "{ptbxl_auroc:.4f} / {ptbxl_auprc:.4f} |\n".format(**r)
            )
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_root",
        default=str(OUT_ROOT),
        help="Output root for sweep artifacts. Defaults to the original 20260516 sweep root.",
    )
    ap.add_argument("--centers", nargs="+", default=["ningbo", "cpsc_2018"])
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["lambda015", "convex_anchor"],
    )
    ap.add_argument("--Ms", nargs="+", type=int, default=[10, 20, 40, 80])
    ap.add_argument("--epochs", nargs="+", type=int, default=[30])
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--subset_seed", type=int, default=20260531)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument(
        "--hull_weight_mode",
        choices=["optimized", "one_hot", "uniform", "dirichlet"],
        default="optimized",
        help="optimized is the main LH-AT method; fixed modes are no-adversarial-weight ablations.",
    )
    ap.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument(
        "--enable_quality_gate",
        action="store_true",
        help=(
            "Enable hard semantic/quality-gate filtering during real-anchor "
            "LH-AT. The paper mainline keeps this disabled by default and uses "
            "gate metrics as monitoring only."
        ),
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)

    (OUT_ROOT / "summaries").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "sweep_config.json").open("w") as f:
        json.dump(
            {
                "args": vars(args),
                "variants": VARIANTS,
                "note": (
                    "VAE-only real-anchor LH-AT; no ECGTwin DiT synthetic or "
                    "center-token samples. Hard quality-gate filtering is "
                    "disabled by default; gate metrics are logged for monitoring."
                ),
            },
            f,
            indent=2,
        )

    rows: list[dict[str, object]] = []
    for center in args.centers:
        for variant in args.variants:
            for hull_m in args.Ms:
                for epochs in args.epochs:
                    eval_path = one_run(center, args.K, variant, hull_m, epochs, args.seed, args)
                    if args.dry_run:
                        continue
                    metrics = eval_metrics(eval_path)
                    row = {
                        "center": center,
                        "K": args.K,
                        "variant": variant,
                        "hull_M": hull_m,
                        "hull_weight_mode": args.hull_weight_mode,
                        "epochs": epochs,
                        "eval_path": str(eval_path),
                    }
                    row.update(metrics)
                    row["target_auroc"] = metrics[f"{center}_auroc"]
                    row["target_auprc"] = metrics[f"{center}_auprc"]
                    rows.append(row)
                    write_summary(rows)

    if rows:
        write_summary(rows)


if __name__ == "__main__":
    main()
