#!/usr/bin/env python3
"""Seeded K-fold stability check for real-anchor Latent-Hull AT.

This runner performs cross-validation over the existing target-center K=500
real-anchor latent pool. For each center/seed/fold:

  * split the 500 target-center anchors into train and held-out folds;
  * train real-anchor LH-AT on the train folds only;
  * evaluate the fine-tuned model on the held-out target-center fold only;
  * evaluate the frozen PTB-XL baseline on the same held-out fold;
  * write per-run metrics plus center-level mean/std summaries.

The evaluation meta JSON contains both ``ref_record_ids`` and
``heldout_record_ids``. ``eval_crosscenter.py`` uses the former for leakage
exclusion and the latter for explicit fold evaluation.
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

from scripts.paper.run_vae_only_latenthull_sweep_20260516 import (  # noqa: E402
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
    _find_real_anchor_base,
    eval_metrics,
)


OUT_ROOT = Path("/root/autodl-tmp/paper_real_anchor_lhat_seed_kfold_20260517")
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


def _stratified_kfold(labels: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """Return deterministic primary-label stratified fold indices."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > len(labels):
        raise ValueError(f"n_splits={n_splits} exceeds N={len(labels)}")
    rng = np.random.default_rng(seed)
    primary = labels.argmax(axis=1)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for class_idx in range(labels.shape[1]):
        idx = np.where(primary == class_idx)[0]
        rng.shuffle(idx)
        for rank, sample_idx in enumerate(idx):
            folds[rank % n_splits].append(int(sample_idx))
    out = []
    for fold in folds:
        fold_arr = np.asarray(fold, dtype=np.int64)
        rng.shuffle(fold_arr)
        out.append(fold_arr)
    return out


def prepare_fold_subset(
    center: str,
    seed: int,
    fold: int,
    n_splits: int,
) -> dict[str, Path | int]:
    """Create train-fold real-anchor files plus held-out fold metadata."""
    base = _find_real_anchor_base(center)
    signal_path = base.with_suffix(".signals.npz")
    latent_path = base.with_suffix(".latent.npz")
    meta_path = base.with_suffix(".ref_meta.json")

    out_dir = OUT_ROOT / "fold_subsets" / center / f"seed{seed}_fold{fold}of{n_splits}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_signal = out_dir / f"{center}_seed{seed}_fold{fold}of{n_splits}_train.signals.npz"
    out_latent = out_dir / f"{center}_seed{seed}_fold{fold}of{n_splits}_train.latent.npz"
    out_meta = out_dir / f"{center}_seed{seed}_fold{fold}of{n_splits}.ref_meta.json"
    out_trust = out_dir / f"{center}_seed{seed}_fold{fold}of{n_splits}.class_trust.json"
    if out_signal.exists() and out_latent.exists() and out_meta.exists() and out_trust.exists():
        with out_meta.open() as f:
            meta = json.load(f)
        return {
            "signals": out_signal,
            "latents": out_latent,
            "meta": out_meta,
            "trust": out_trust,
            "train_n": int(meta["K"]),
            "heldout_n": int(meta["heldout_K"]),
        }

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

    folds = _stratified_kfold(labels, n_splits=n_splits, seed=seed)
    heldout_idx = folds[fold]
    heldout_set = set(int(i) for i in heldout_idx)
    train_idx = np.asarray(
        [i for i in range(len(labels)) if i not in heldout_set],
        dtype=np.int64,
    )

    np_savez_compressed(
        out_signal,
        signals=signals[train_idx],
        labels=labels[train_idx],
        record_ids=record_ids[train_idx],
        center_name=center,
        class_names=CLASS_NAMES,
    )
    np_savez_compressed(
        out_latent,
        latents=latents[train_idx],
        labels=labels[train_idx],
        center_name=center,
        class_names=CLASS_NAMES,
        source_indices=np.asarray(source_indices)[train_idx],
        record_ids=record_ids[train_idx],
        primary_class=primary_class[train_idx],
        source_ids=np.zeros((len(train_idx),), dtype=np.int16),
        source_names=np.asarray(["real_anchor"]),
        source_local_indices=np.arange(len(train_idx), dtype=np.int32),
    )
    with out_meta.open("w") as f:
        json.dump(
            {
                "center": center,
                "K": int(len(train_idx)),
                "heldout_K": int(len(heldout_idx)),
                "selection_seed": int(seed),
                "fold": int(fold),
                "n_splits": int(n_splits),
                "parent": str(meta_path),
                "ref_record_ids": record_ids[train_idx].tolist(),
                "heldout_record_ids": record_ids[heldout_idx].tolist(),
                "source_indices": np.asarray(source_indices)[train_idx].astype(int).tolist(),
                "heldout_source_indices": np.asarray(source_indices)[heldout_idx].astype(int).tolist(),
                "policy": (
                    "primary-label stratified K-fold split over existing "
                    "target-center K=500 real-anchor ECGTwin VAE latent pool"
                ),
            },
            f,
            indent=2,
        )
    counts = labels[train_idx].sum(axis=0).astype(int).tolist()
    heldout_counts = labels[heldout_idx].sum(axis=0).astype(int).tolist()
    with out_trust.open("w") as f:
        json.dump(
            {
                "tag": f"{center}_seed{seed}_fold{fold}of{n_splits}",
                "class_trust": {"CD": 0.0, "HYP": 0.0, "MI": 1.0, "NORM": 1.0, "STTC": 1.0},
                "label_counts": dict(zip(CLASS_NAMES.tolist(), counts)),
                "heldout_label_counts": dict(zip(CLASS_NAMES.tolist(), heldout_counts)),
                "policy": "real-anchor LH-AT K-fold stability; HYP/CD disabled for adv generation",
            },
            f,
            indent=2,
        )
    return {
        "signals": out_signal,
        "latents": out_latent,
        "meta": out_meta,
        "trust": out_trust,
        "train_n": int(len(train_idx)),
        "heldout_n": int(len(heldout_idx)),
    }


def baseline_eval(center: str, subset: dict[str, Path | int], args: argparse.Namespace) -> Path:
    out_dir = OUT_ROOT / "baseline_eval" / center
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = Path(subset["meta"])
    eval_path = out_dir / f"{meta.stem}_baseline_heldout.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] baseline {center} {meta.stem}", flush=True)
        return eval_path
    cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme", "super5",
        "--model_dir", str(Path(BASELINE_CKPT).parent),
        "--device", "cuda",
        "--crop_len", "1000",
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--ptbxl_cache", PTBXL_PREP,
        "--preprocess_mode", "minimal_resample",
        "--norm_mode", "per_sample_global",
        "--pn2021_cache_dir", PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir", PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--include_record_ids", str(meta),
        "--output_path", str(eval_path),
    ]
    run(cmd, out_dir / f"{meta.stem}_baseline_eval.log", dry_run=args.dry_run)
    return eval_path


def train_eval(center: str, subset: dict[str, Path | int], args: argparse.Namespace) -> Path:
    meta = Path(subset["meta"])
    tag = (
        f"{center}_seed{args.current_seed}_fold{args.current_fold}of{args.folds}_"
        f"K{subset['train_n']}_M{args.hull_m}_lam{str(args.hull_lambda).replace('.', 'p')}_"
        f"ep{args.n_epochs}"
    )
    out_dir = OUT_ROOT / "runs" / tag
    eval_path = out_dir / "eval_result_v3_super5_foldheldout_crop1000.json"
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
        "--ref_meta_json", str(meta),
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
        "--hull_label_mode", args.hull_label_mode,
        "--hull_mix_label_mode", "anchor",
        "--source_sampling_strategy", "source_weighted",
        "--source_weights", "real_anchor=1.0",
        "--class_trust", str(subset["trust"]),
        "--classes_in_scope", "NORM", "MI", "STTC",
        "--adv_label_mode", "multi_hot_hard",
        "--adv_teacher_mix", "0.0",
        "--adv_weight", str(args.adv_weight),
        "--target_real_weight", str(args.target_real_weight),
        "--ptbxl_weight", "1.0",
        "--roundtrip_weight", str(args.roundtrip_weight),
        "--roundtrip_anchor_n", str(args.roundtrip_anchor_n),
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
        "--seed", str(args.current_seed),
        "--crop_len", "1000",
        "--device", "cuda:0",
    ]
    if args.disable_quality_gate:
        train_cmd.append("--disable_quality_gate")
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
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--ptbxl_cache", PTBXL_PREP,
        "--preprocess_mode", "minimal_resample",
        "--norm_mode", "per_sample_global",
        "--pn2021_cache_dir", PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir", PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--exclude_ref_ids", str(meta),
        "--include_record_ids", str(meta),
        "--output_path", str(eval_path),
    ]
    run(eval_cmd, out_dir / "eval_foldheldout.log", dry_run=args.dry_run)
    return eval_path


def _target_metrics(eval_path: Path, center: str) -> dict[str, float | int]:
    metrics = eval_metrics(eval_path)
    return {
        "target_auroc": metrics[f"{center}_auroc"],
        "target_auprc": metrics[f"{center}_auprc"],
        "target_n": int(metrics.get(f"{center}_n", 0)),
        "ptbxl_auroc": metrics["ptbxl_auroc"],
        "ptbxl_auprc": metrics["ptbxl_auprc"],
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    csv_path = summary_dir / "real_anchor_lhat_seed_kfold.csv"
    preferred = [
        "center", "seed", "fold", "folds", "train_n", "heldout_n",
        "baseline_auroc", "baseline_auprc", "lhat_auroc", "lhat_auprc",
        "delta_auroc", "delta_auprc", "target_n", "eval_path",
    ]
    all_fields = sorted({key for row in rows for key in row.keys()})
    fields = preferred + [key for key in all_fields if key not in preferred]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["center"]), []).append(row)

    md_path = summary_dir / "real_anchor_lhat_seed_kfold.md"
    with md_path.open("w") as f:
        f.write("# Real-Anchor LH-AT Seed/K-Fold Stability\n\n")
        f.write(
            "Evaluation is fold-heldout for the target center. Each run trains "
            "on train-fold target anchors and evaluates the target center only "
            "on held-out anchor IDs; train anchor IDs are excluded.\n\n"
        )
        f.write("## Center Summary\n\n")
        f.write("| center | n_runs | baseline AUROC/AUPRC mean | LH-AT AUROC/AUPRC mean | delta AUROC/AUPRC mean | delta AUROC/AUPRC std |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for center, group in grouped.items():
            def arr(key: str) -> np.ndarray:
                return np.asarray([float(r[key]) for r in group], dtype=np.float64)

            b_auroc, b_auprc = arr("baseline_auroc"), arr("baseline_auprc")
            l_auroc, l_auprc = arr("lhat_auroc"), arr("lhat_auprc")
            d_auroc, d_auprc = arr("delta_auroc"), arr("delta_auprc")
            f.write(
                f"| {center} | {len(group)} | "
                f"{b_auroc.mean():.4f} / {b_auprc.mean():.4f} | "
                f"{l_auroc.mean():.4f} / {l_auprc.mean():.4f} | "
                f"{d_auroc.mean():+.4f} / {d_auprc.mean():+.4f} | "
                f"{d_auroc.std(ddof=0):.4f} / {d_auprc.std(ddof=0):.4f} |\n"
            )

        f.write("\n## Per Run\n\n")
        f.write("| center | seed | fold | train_n | heldout_n | baseline | LH-AT | delta |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {center} | {seed} | {fold}/{folds} | {train_n} | {heldout_n} | "
                "{baseline_auroc:.4f} / {baseline_auprc:.4f} | "
                "{lhat_auroc:.4f} / {lhat_auprc:.4f} | "
                "{delta_auroc:+.4f} / {delta_auprc:+.4f} |\n".format(**row)
            )
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {md_path}", flush=True)


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument(
        "--centers",
        nargs="+",
        default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260531, 20260541, 20260551])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", nargs="+", type=int, default=None)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=5)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument(
        "--hull_label_mode",
        choices=["primary", "exact", "compatible"],
        default="exact",
    )
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--pgd_batch", type=int, default=32)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_weight", type=float, default=0.5)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--adv_weight", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--disable_quality_gate", action="store_true", default=True)
    ap.add_argument("--enable_quality_gate", action="store_false", dest="disable_quality_gate")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = Path(args.out_root)
    (OUT_ROOT / "summaries").mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "seed_kfold_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    requested_folds = args.only_folds if args.only_folds is not None else list(range(args.folds))
    rows: list[dict[str, object]] = []
    for center in args.centers:
        for seed in args.seeds:
            for fold in requested_folds:
                if fold < 0 or fold >= args.folds:
                    raise ValueError(f"fold {fold} outside [0, {args.folds})")
                args.current_seed = int(seed)
                args.current_fold = int(fold)
                subset = prepare_fold_subset(center, seed, fold, args.folds)
                base_eval_path = baseline_eval(center, subset, args)
                lhat_eval_path = train_eval(center, subset, args)
                if args.dry_run:
                    continue
                base = _target_metrics(base_eval_path, center)
                lhat = _target_metrics(lhat_eval_path, center)
                row = {
                    "center": center,
                    "seed": int(seed),
                    "fold": int(fold),
                    "folds": int(args.folds),
                    "train_n": int(subset["train_n"]),
                    "heldout_n": int(subset["heldout_n"]),
                    "target_n": int(lhat["target_n"]),
                    "baseline_auroc": float(base["target_auroc"]),
                    "baseline_auprc": float(base["target_auprc"]),
                    "lhat_auroc": float(lhat["target_auroc"]),
                    "lhat_auprc": float(lhat["target_auprc"]),
                    "delta_auroc": float(lhat["target_auroc"]) - float(base["target_auroc"]),
                    "delta_auprc": float(lhat["target_auprc"]) - float(base["target_auprc"]),
                    "baseline_ptbxl_auroc": float(base["ptbxl_auroc"]),
                    "baseline_ptbxl_auprc": float(base["ptbxl_auprc"]),
                    "lhat_ptbxl_auroc": float(lhat["ptbxl_auroc"]),
                    "lhat_ptbxl_auprc": float(lhat["ptbxl_auprc"]),
                    "eval_path": str(lhat_eval_path),
                    "baseline_eval_path": str(base_eval_path),
                    "meta_path": str(subset["meta"]),
                }
                rows.append(row)
                write_summary(rows)

    if rows:
        write_summary(rows)


if __name__ == "__main__":
    main()
