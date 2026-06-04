#!/usr/bin/env python3
"""Evaluate EfficientNet v7 VAE-only vs VAE+AugMix on target-center PN2021-C.

This runner evaluates one adapted model per target center.  For each center it
runs the true VAE-only control and the current latent-branch AugMix variant on
that center's corrupted PN2021 records, excluding the same K-shot reference ids
used for adaptation.
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


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
PYTHON = "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_METHODS = ["vae_noaug", "vae_lhat"]
DEFAULT_CORRUPTIONS = [
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
]


def run_stream(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(cmd)
    print(f"[run] {rendered}", flush=True)
    if dry_run:
        log_path.write_text(rendered + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log:
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


def run_id(args: argparse.Namespace) -> str:
    return f"seed{args.seed}_{args.protocol}_v7_sjr_rgq_{args.run_stamp}"


def method_run_id(args: argparse.Namespace, method: str) -> str:
    stamp = args.run_stamp
    if method == "vae_noaug" and args.noaug_run_stamp:
        stamp = args.noaug_run_stamp
    elif method != "vae_noaug" and args.lhat_run_stamp:
        stamp = args.lhat_run_stamp
    return f"seed{args.seed}_{args.protocol}_v7_sjr_rgq_{stamp}"


def ref_meta(args: argparse.Namespace, center: str, k: int) -> Path:
    return (
        args.subset_root
        / center
        / f"k{k}_seed{args.seed}"
        / f"{center}_real_k{k}_seed{args.seed}.ref_meta.json"
    )


def subset_k(args: argparse.Namespace, center: str) -> int:
    pattern = args.subset_root / center / f"k*_seed{args.seed}" / f"{center}_real_k*_seed{args.seed}.ref_meta.json"
    matches = sorted(pattern.parent.parent.glob(pattern.name))
    if args.protocol == "k500":
        return 500
    if args.protocol == "p10":
        suffix = "_p10"
    elif args.protocol == "p20":
        suffix = "_p20"
    else:
        raise ValueError(f"unknown protocol {args.protocol}")
    matches = sorted((args.subset_root / center).glob(f"k*_seed{args.seed}{suffix}/{center}_real_k*_seed{args.seed}.ref_meta.json"))
    if not matches:
        raise FileNotFoundError(f"missing subset for {center} {args.protocol} seed={args.seed}")
    return int(matches[0].parent.name.split("_", 1)[0].removeprefix("k"))


def model_dir(args: argparse.Namespace, center: str, method: str) -> Path:
    root = args.data_root / "runs" / f"effnet_{method}_{args.protocol}_v7_sjr_rgq" / method_run_id(args, method)
    if method == "vae_noaug":
        pattern = f"{center}_realall_*_fullft_{args.protocol}*_noaugmix_ep{args.epochs}_seed{args.seed}"
    else:
        pattern = f"{center}_realall_*_fullft_{args.protocol}*_ep{args.epochs}_seed{args.seed}"
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"missing model dir: {root}/{pattern}")
    if len(matches) > 1:
        print(f"[warn] multiple model dirs for {center} {method}; using {matches[0]}", flush=True)
    return matches[0]


def clean_target_metrics(path: Path, center: str) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    vals = data["pn2021"]["per_center"][center]
    return float(vals["macro_auroc"]), float(vals["macro_auprc"])


def corrupted_mean(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for corruption_map in data["aggregate_by_corruption_severity"].values():
        for vals in corruption_map.values():
            rows.append((float(vals["mean_macro_auroc"]), float(vals["mean_macro_auprc"])))
    if not rows:
        return float("nan"), float("nan")
    return (
        sum(r[0] for r in rows) / len(rows),
        sum(r[1] for r in rows) / len(rows),
    )


def eval_one(args: argparse.Namespace, center: str, method: str, gpu: str) -> dict[str, Any]:
    k = subset_k(args, center)
    mdir = model_dir(args, center, method)
    clean_eval = mdir / "eval_result_v7_exclrefs_crop1000.json"
    if not clean_eval.exists():
        raise FileNotFoundError(clean_eval)
    meta = ref_meta(args, center, k)
    if not meta.exists():
        raise FileNotFoundError(meta)
    out_path = mdir / f"eval_pn2021_c_v7_{center}_exclrefs_stream_standard.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ECG_ADV_GEN_DATA_ROOT"] = str(args.data_root)
    env["TMPDIR"] = str(args.tmp_root)
    env["XDG_CACHE_HOME"] = str(args.cache_root)
    env["PYTHONUNBUFFERED"] = "1"
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-u",
        "scripts/triple_labels/eval_pn2021_corruptions.py",
        "--mode",
        "stream",
        "--scheme",
        "super5",
        "--model_dir",
        str(mdir),
        "--clean_mmap_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--clean_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--clean_eval_json",
        str(clean_eval),
        "--centers",
        center,
        "--corruptions",
        *args.corruptions,
        "--severities",
        *[str(v) for v in args.severities],
        "--exclude_ref_ids",
        str(meta),
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--min_pos",
        str(args.min_pos),
        "--seed",
        str(args.corruption_seed),
        "--output_path",
        str(out_path),
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if out_path.exists() and not args.force:
        print(f"[skip] {center} {method} PN2021-C exists: {out_path}", flush=True)
    else:
        run_stream(
            cmd,
            env=env,
            log_path=args.log_root / run_id(args) / f"{center}_{method}_pn2021c.log",
            dry_run=args.dry_run,
        )
    clean_auroc, clean_auprc = (float("nan"), float("nan"))
    corrupt_auroc, corrupt_auprc = (float("nan"), float("nan"))
    if not args.dry_run:
        clean_auroc, clean_auprc = clean_target_metrics(clean_eval, center)
        corrupt_auroc, corrupt_auprc = corrupted_mean(out_path)
    return {
        "protocol": args.protocol,
        "seed": args.seed,
        "center": center,
        "K": k,
        "method": method,
        "clean_macro_auroc": clean_auroc,
        "clean_macro_auprc": clean_auprc,
        "corrupted_macro_auroc": corrupt_auroc,
        "corrupted_macro_auprc": corrupt_auprc,
        "drop_macro_auroc": clean_auroc - corrupt_auroc,
        "drop_macro_auprc": clean_auprc - corrupt_auprc,
        "model_dir": str(mdir),
        "eval_json": str(out_path),
        "ref_meta": str(meta),
    }


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.summary_dir / f"effnet_v7_{args.protocol}_seed{args.seed}_{args.run_stamp}_pn2021c_augmix_vs_noaug.csv"
    fields = [
        "protocol",
        "seed",
        "center",
        "K",
        "method",
        "clean_macro_auroc",
        "clean_macro_auprc",
        "corrupted_macro_auroc",
        "corrupted_macro_auprc",
        "drop_macro_auroc",
        "drop_macro_auprc",
        "delta_corrupted_auroc_vs_noaug_pp",
        "delta_corrupted_auprc_vs_noaug_pp",
        "delta_clean_auroc_vs_noaug_pp",
        "delta_clean_auprc_vs_noaug_pp",
        "model_dir",
        "eval_json",
        "ref_meta",
    ]
    by_center = {(r["center"], r["method"]): r for r in rows}
    enriched = []
    for row in rows:
        out = dict(row)
        base = by_center.get((row["center"], "vae_noaug"))
        if base and row["method"] != "vae_noaug":
            out["delta_corrupted_auroc_vs_noaug_pp"] = (
                row["corrupted_macro_auroc"] - base["corrupted_macro_auroc"]
            ) * 100.0
            out["delta_corrupted_auprc_vs_noaug_pp"] = (
                row["corrupted_macro_auprc"] - base["corrupted_macro_auprc"]
            ) * 100.0
            out["delta_clean_auroc_vs_noaug_pp"] = (
                row["clean_macro_auroc"] - base["clean_macro_auroc"]
            ) * 100.0
            out["delta_clean_auprc_vs_noaug_pp"] = (
                row["clean_macro_auprc"] - base["clean_macro_auprc"]
            ) * 100.0
        else:
            out["delta_corrupted_auroc_vs_noaug_pp"] = ""
            out["delta_corrupted_auprc_vs_noaug_pp"] = ""
            out["delta_clean_auroc_vs_noaug_pp"] = ""
            out["delta_clean_auprc_vs_noaug_pp"] = ""
        enriched.append(out)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)
    print(f"[summary] {csv_path}", flush=True)


def run_center_worker(args_path: Path, center: str, gpu: str) -> None:
    args = parse_worker_args(args_path)
    rows = []
    for method in args.methods:
        rows.append(eval_one(args, center, method, gpu))
        write_summary(args, collect_existing_rows(args, rows))


def collect_existing_rows(args: argparse.Namespace, current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(current_rows)
    seen = {(r["center"], r["method"]) for r in rows}
    for center in args.centers:
        for method in args.methods:
            if (center, method) in seen:
                continue
            try:
                mdir = model_dir(args, center, method)
                out_path = mdir / f"eval_pn2021_c_v7_{center}_exclrefs_stream_standard.json"
                if not out_path.exists():
                    continue
                clean_eval = mdir / "eval_result_v7_exclrefs_crop1000.json"
                clean_auroc, clean_auprc = clean_target_metrics(clean_eval, center)
                corrupt_auroc, corrupt_auprc = corrupted_mean(out_path)
                k = subset_k(args, center)
                rows.append({
                    "protocol": args.protocol,
                    "seed": args.seed,
                    "center": center,
                    "K": k,
                    "method": method,
                    "clean_macro_auroc": clean_auroc,
                    "clean_macro_auprc": clean_auprc,
                    "corrupted_macro_auroc": corrupt_auroc,
                    "corrupted_macro_auprc": corrupt_auprc,
                    "drop_macro_auroc": clean_auroc - corrupt_auroc,
                    "drop_macro_auprc": clean_auprc - corrupt_auprc,
                    "model_dir": str(mdir),
                    "eval_json": str(out_path),
                    "ref_meta": str(ref_meta(args, center, k)),
                })
            except (FileNotFoundError, KeyError):
                continue
    return sorted(rows, key=lambda r: (r["center"], r["method"]))


def write_worker_args(args: argparse.Namespace) -> Path:
    path = args.log_root / run_id(args) / "pn2021c_worker_args.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = vars(args).copy()
    for key in ["data_root", "subset_root", "log_root", "summary_dir", "tmp_root", "cache_root"]:
        data[key] = str(data[key])
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_worker_args(path: Path) -> argparse.Namespace:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["data_root", "subset_root", "log_root", "summary_dir", "tmp_root", "cache_root"]:
        data[key] = Path(data[key])
    return argparse.Namespace(**data)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["k500", "p10", "p20"], default="k500")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--run-stamp", default="20260530_rep1")
    ap.add_argument(
        "--noaug-run-stamp",
        default="",
        help="Use a different run stamp for the vae_noaug baseline.",
    )
    ap.add_argument(
        "--lhat-run-stamp",
        default="",
        help="Use a different run stamp for non-noAug VAE/AugMix methods.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--subset-root", type=Path, default=DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets")
    ap.add_argument("--log-root", type=Path, default=DATA_ROOT / "runs/logs/effnet_v7_pn2021c_augmix_vs_noaug_20260530")
    ap.add_argument("--summary-dir", type=Path, default=DATA_ROOT / "runs/pn2021_c_comparisons")
    ap.add_argument("--tmp-root", type=Path, default=Path("/home/linbinhao/tmp_ecg"))
    ap.add_argument("--cache-root", type=Path, default=DATA_ROOT / "cache")
    ap.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--corruption-seed", type=int, default=20260501)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(args.gpus) < len(args.centers):
        raise SystemExit("--gpus must provide at least one id per center")
    args.gpus = args.gpus[: len(args.centers)]
    return args


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    worker_args = write_worker_args(args)
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for center, gpu in zip(args.centers, args.gpus):
        cmd = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from scripts.paper.eval_effnet_v7_pn2021c_augmix_vs_noaug_20260530 "
                "import run_center_worker; "
                "import sys; run_center_worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])"
            ),
            str(worker_args),
            center,
            str(gpu),
        ]
        proc = subprocess.Popen(cmd, cwd=str(REPO), text=True)
        procs.append((center, proc))
        print(f"[launched] center={center} gpu={gpu} pid={proc.pid}", flush=True)
    failures = []
    for center, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failures.append((center, ret))
    rows = collect_existing_rows(args, [])
    if rows:
        write_summary(args, rows)
    if failures:
        raise RuntimeError(f"PN2021-C workers failed: {failures}")


if __name__ == "__main__":
    main()
