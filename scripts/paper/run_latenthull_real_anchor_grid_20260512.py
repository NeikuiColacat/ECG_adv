#!/usr/bin/env python3
"""Run clean real-anchor Latent-Hull online AT grid for paper ablations.

This script intentionally uses only target-center real anchors for the K sweep.
It does not reuse prompt-token synthetic pools generated from K=500 refs, so a
K=100 run only trains on and excludes those 100 target-center records.
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


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
_DEFAULT_DATA_ROOT = _MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")

PYTHON = os.environ.get(
    "ECGTWIN_PYTHON",
    "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
    if Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python").exists()
    else "/root/miniforge3/envs/ECGTwin/bin/python",
)
PROJECT_ROOT = Path(os.environ.get("ECG_ADV_GEN_PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)))
DATA_ROOT = Path(os.environ.get("ECG_ADV_GEN_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))
BASELINE_CKPT = str(
    DATA_ROOT
    / "triple_labels"
    / "super5_minresample_full10_perglobal_20260503"
    / "best_model.pt"
)
PTBXL_RAW = str(DATA_ROOT / "ptbxl" / "raw100.npy")
PTBXL_CSV = str(DATA_ROOT / "ptbxl" / "ptbxl_database.csv")
PTBXL_PREP = str(
    DATA_ROOT
    / "triple_labels"
    / "cache"
    / "ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
)
PN2021_CACHE_DIR = str(DATA_ROOT / "triple_labels" / "pn2021_eval_cache_minresample_perglobal")
PN2021_MMAP_CACHE_DIR = str(
    DATA_ROOT / "triple_labels" / "pn2021_eval_cache_mmap_minresample_perglobal"
)
PN2021_ROOT = str(DATA_ROOT / "physionet2021" / "training")
REAL_ROOT = DATA_ROOT / "ecgtwin_prompt_token_super5" / "real_anchor_selected_v2"
OUT_ROOT = DATA_ROOT / "paper_latenthull_grid_20260512"

CLASS_NAMES = np.asarray(["CD", "HYP", "MI", "NORM", "STTC"])


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))
    env.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / "cache"))
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


def proportional_stratified_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Deterministically select roughly class-proportional primary-label subset."""
    rng = np.random.default_rng(seed)
    primary = labels.argmax(axis=1)
    pools = []
    for c in range(labels.shape[1]):
        idx = np.where(primary == c)[0]
        rng.shuffle(idx)
        pools.append(idx)

    counts = np.asarray([len(p) for p in pools], dtype=np.float64)
    raw = counts / max(counts.sum(), 1.0) * k
    take = np.floor(raw).astype(int)
    remainders = raw - take
    for c in np.argsort(-remainders):
        if take.sum() >= k:
            break
        if take[c] < len(pools[c]):
            take[c] += 1
    for c in range(len(take)):
        take[c] = min(take[c], len(pools[c]))

    selected = np.concatenate([pools[c][: take[c]] for c in range(len(pools))])
    if selected.size < k:
        already = set(int(i) for i in selected)
        rest = np.asarray([i for i in range(labels.shape[0]) if i not in already])
        rng.shuffle(rest)
        selected = np.concatenate([selected, rest[: k - selected.size]])
    rng.shuffle(selected)
    return selected[:k].astype(np.int64)


def prepare_subset(center: str, k: int, seed: int) -> dict[str, Path]:
    base = REAL_ROOT / center / f"{center}_real_k500_seed42"
    signal_path = base.with_suffix(".signals.npz")
    latent_path = base.with_suffix(".latent.npz")
    meta_path = base.with_suffix(".ref_meta.json")
    if not signal_path.exists() or not latent_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing real-anchor files for {center}: {base}")

    out_dir = OUT_ROOT / "subsets" / center / f"k{k}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_signal = out_dir / f"{center}_real_k{k}_seed{seed}.signals.npz"
    out_latent = out_dir / f"{center}_real_k{k}_seed{seed}.latent.npz"
    out_meta = out_dir / f"{center}_real_k{k}_seed{seed}.ref_meta.json"
    out_trust = out_dir / f"{center}_real_k{k}_seed{seed}.class_trust.json"

    if out_signal.exists() and out_latent.exists() and out_meta.exists() and out_trust.exists():
        return {
            "signals": out_signal,
            "latents": out_latent,
            "meta": out_meta,
            "trust": out_trust,
        }

    with np.load(signal_path, allow_pickle=True) as sig_npz:
        signals = np.asarray(sig_npz["signals"], dtype=np.float32)
        labels = np.asarray(sig_npz["labels"], dtype=np.float32)
        record_ids = np.asarray(sig_npz["record_ids"]).astype(str)
    with np.load(latent_path, allow_pickle=True) as lat_npz:
        latents = np.asarray(lat_npz["latents"], dtype=np.float32)
        source_indices = np.asarray(lat_npz["source_indices"]) if "source_indices" in lat_npz else np.arange(len(labels))
        primary_class = np.asarray(lat_npz["primary_class"]).astype(str) if "primary_class" in lat_npz else CLASS_NAMES[labels.argmax(axis=1)]

    idx = proportional_stratified_indices(labels, k, seed=seed)
    np.savez_compressed(
        out_signal,
        signals=signals[idx],
        labels=labels[idx],
        record_ids=record_ids[idx],
        center_name=np.asarray(center),
        class_names=CLASS_NAMES,
    )
    np.savez_compressed(
        out_latent,
        latents=latents[idx],
        labels=labels[idx],
        center_name=np.asarray(center),
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
                "source_indices": source_indices[idx].astype(int).tolist(),
                "policy": "proportional stratified subset from real_anchor_selected_v2 K=500",
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
                "policy": "real-anchor clean K sweep; HYP/CD disabled to match main trusted-class setting",
            },
            f,
            indent=2,
        )
    return {
        "signals": out_signal,
        "latents": out_latent,
        "meta": out_meta,
        "trust": out_trust,
    }


def eval_metrics(eval_path: Path) -> dict[str, float]:
    d = json.load(eval_path.open())
    pt = d["ptbxl_test"]
    pn = d["pn2021"]
    out = {
        "ptbxl_auroc": float(pt["macro_auroc"]),
        "ptbxl_auprc": float(pt["macro_auprc"]),
        "pn2021_avg_auroc": float(pn["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(pn["avg_macro_auprc"]),
    }
    for center, m in pn["per_center"].items():
        out[f"{center}_auroc"] = float(m["macro_auroc"])
        out[f"{center}_auprc"] = float(m["macro_auprc"])
        out[f"{center}_n"] = int(m.get("effective_n", m.get("n_records", 0)))
    return out


def one_run(center: str, k: int, lam: float, seed: int, args: argparse.Namespace) -> Path:
    subset = prepare_subset(center, k, seed=args.subset_seed)
    tag = f"{center}_realK{k}_lambda{str(lam).replace('.', 'p')}_seed{seed}"
    out_dir = OUT_ROOT / "runs" / tag
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_20260512.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {tag} already evaluated")
        return eval_path

    out_dir.mkdir(parents=True, exist_ok=True)
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
        "--delta_init_scale",
        "0.1",
        "--attack_mode",
        "latent_hull",
        "--hull_M",
        "10",
        "--hull_lambda",
        str(lam),
        "--hull_steps",
        "5",
        "--hull_lr",
        "0.25",
        "--hull_weight_mode",
        "optimized",
        "--hull_dirichlet_alpha",
        "1.0",
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
        "--adv_soft_target_floor",
        "0.0",
        "--boundary_prob_min",
        "0.0",
        "--boundary_prob_max",
        "1.0",
        "--target_real_weight",
        str(args.target_real_weight),
        "--ptbxl_weight",
        "1.0",
        "--roundtrip_weight",
        "0.5",
        "--roundtrip_anchor_n",
        str(args.roundtrip_anchor_n),
        "--adv_weight",
        "0.06",
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
    ]
    run(train_cmd, out_dir / "train.log")

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
    run(eval_cmd, out_dir / "eval_full.log")
    return eval_path


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_dir = OUT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    csv_path = summary_dir / "real_anchor_k_lambda_grid.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = summary_dir / "real_anchor_k_lambda_grid.md"
    key_fields = [
        "center",
        "K",
        "lambda",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "target_auroc",
        "target_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
    ]
    with md_path.open("w") as f:
        f.write("# Real-Anchor Latent-Hull K/Lambda Grid\n\n")
        f.write("| " + " | ".join(key_fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(key_fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in key_fields) + " |\n")
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=["chapman_shaoxing", "cpsc_2018", "georgia"])
    ap.add_argument("--ks", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.25, 0.35])
    ap.add_argument("--main_lambda", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--subset_seed", type=int, default=20260531)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument(
        "--asr_consec_low_max",
        type=int,
        default=999,
        help="Use a large value for sweep mode so low-ASR settings are recorded instead of aborting.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for center in args.centers:
        # K sweep at main lambda.
        for k in args.ks:
            eval_path = one_run(center, k, args.main_lambda, args.seed, args)
            metrics = eval_metrics(eval_path)
            row = {"center": center, "K": k, "lambda": args.main_lambda, "eval_path": str(eval_path)}
            row.update(metrics)
            row["target_auroc"] = metrics.get(f"{center}_auroc")
            row["target_auprc"] = metrics.get(f"{center}_auprc")
            rows.append(row)
            write_summary(rows)
        # Lambda sweep at K=max(Ks), skipping duplicate main lambda.
        k = max(args.ks)
        for lam in args.lambdas:
            if abs(lam - args.main_lambda) < 1e-12:
                continue
            eval_path = one_run(center, k, lam, args.seed, args)
            metrics = eval_metrics(eval_path)
            row = {"center": center, "K": k, "lambda": lam, "eval_path": str(eval_path)}
            row.update(metrics)
            row["target_auroc"] = metrics.get(f"{center}_auroc")
            row["target_auprc"] = metrics.get(f"{center}_auprc")
            rows.append(row)
            write_summary(rows)

    write_summary(rows)


if __name__ == "__main__":
    main()
