#!/usr/bin/env python3
"""Run ECGFounder v7 Direct/VAE-LHAT K-shot replication matrix.

This mirrors the EfficientNet1DV2 v7 matrix runner, but uses ECGFounder
feature caches.  It keeps the ECGFounder encoder frozen:

1. ensure a v7 PN2021-label feature cache exists;
2. export matched v7 K-shot/percent ECGTwin anchor subsets;
3. run direct target-center head fine-tuning;
4. run residual-adapter VAE latent-hull online AT from that matched direct head;
5. export metrics_long and paper_table artifacts.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ecg_adv_gen.runner.process import build_process_env, run_stream as run_stream_command  # noqa: E402


DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
PYTHON = "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
DEFAULT_SUBSET_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
DEFAULT_SOURCE_LINEAR_PROBE = (
    DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_linear_probe_v6_from_legacy_cache"
)
DEFAULT_V7_LINEAR_PROBE = (
    DATA_ROOT / "paper_foundation_baselines_20260530/ecgfounder_linear_probe_v7_from_v6_cache"
)
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_GPUS = ["0", "1", "2", "3"]
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"


PROTOCOLS = {
    "k500": {"fixed_k": 500, "percent": None},
    "p10": {"fixed_k": None, "percent": 0.10},
    "p20": {"fixed_k": None, "percent": 0.20},
}


def run_stream(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    cwd: Path = REPO,
    dry_run: bool = False,
    append: bool = False,
) -> None:
    run_stream_command(
        cmd,
        log_path=log_path,
        env=env,
        cwd=cwd,
        dry_run=dry_run,
        append=append,
    )


def base_env(args: argparse.Namespace) -> dict[str, str]:
    return build_process_env(
        updates={
            "ECG_ADV_GEN_DATA_ROOT": args.data_root,
            "TMPDIR": args.tmp_root,
            "XDG_CACHE_HOME": args.cache_root,
        },
    )


def mapping_ok(metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta = data.get("pn2021_mapping", {})
    return meta.get("mapping_version") == MAPPING_VERSION and meta.get("mapping_hash") == MAPPING_HASH


def ensure_v7_linear_probe(args: argparse.Namespace) -> None:
    meta = args.linear_probe_dir / "cache_relabel_metadata.json"
    required = [
        args.linear_probe_dir / "best_head.pt",
        args.linear_probe_dir / "ptbxl_ecgfounder_features_official_ptbxl_eval.npz",
        args.linear_probe_dir / "pn2021_ecgfounder_features_official_ptbxl_eval.npz",
    ]
    if all(p.exists() for p in required) and mapping_ok(meta):
        print(f"[cache] ECGFounder v7 feature cache ready: {args.linear_probe_dir}", flush=True)
        return
    cmd = [
        args.python,
        "scripts/paper/relabel_ecgfounder_feature_cache_super5_v5_20260524.py",
        "--source_dir",
        str(args.source_linear_probe_dir),
        "--out_dir",
        str(args.linear_probe_dir),
        "--preprocess_policy",
        "official_ptbxl_eval",
    ]
    if args.force_relabel:
        cmd.append("--force")
    run_stream(
        cmd,
        log_path=args.log_root / "relabel_ecgfounder_v7_cache.log",
        env=base_env(args),
        dry_run=args.dry_run,
    )


def export_subsets(seed: int, protocols: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    percents = [PROTOCOLS[p]["percent"] for p in protocols if PROTOCOLS[p]["percent"] is not None]
    fixed_ks = [PROTOCOLS[p]["fixed_k"] for p in protocols if PROTOCOLS[p]["fixed_k"] is not None]
    summary_path = args.subset_root / f"replication_v7_ecgfounder_seed{seed}_{'_'.join(protocols)}_summary.json"
    cmd = [
        args.python,
        "scripts/paper/export_percent_kshot_v7_sjr_rgq_20260530.py",
        "--centers",
        *args.centers,
        "--percents",
        *[str(p) for p in percents],
        "--seed",
        str(seed),
        "--output-root",
        str(args.subset_root),
        "--summary-path",
        str(summary_path),
    ]
    if fixed_ks:
        cmd.extend(["--fixed-ks", *[str(k) for k in fixed_ks]])
    run_stream(
        cmd,
        log_path=args.log_root / f"export_seed{seed}_{'_'.join(protocols)}.log",
        env=base_env(args),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    by_key: dict[str, dict[str, Any]] = {}
    for item in data["items"]:
        protocol = "k500" if item.get("fixed_k") == 500 else item["protocol_tag"]
        by_key[f"{item['center']}::{protocol}"] = item
    return by_key


def run_id_for(protocol: str, seed: int, args: argparse.Namespace) -> str:
    return f"seed{seed}_{protocol}_v7_sjr_rgq_{args.run_stamp}"


def direct_run_root(args: argparse.Namespace, protocol: str, seed: int, center: str) -> Path:
    return args.data_root / "runs" / f"ecgfounder_direct_{protocol}_v7_sjr_rgq" / run_id_for(protocol, seed, args) / center


def vae_run_root(args: argparse.Namespace, protocol: str, seed: int, center: str) -> Path:
    return args.data_root / "runs" / f"ecgfounder_vae_lhat_{protocol}_v7_sjr_rgq" / run_id_for(protocol, seed, args) / center


def run_center_pair(
    *,
    center: str,
    k: int,
    protocol: str,
    seed: int,
    gpu: str,
    args: argparse.Namespace,
) -> None:
    run_id = run_id_for(protocol, seed, args)
    direct_out = direct_run_root(args, protocol, seed, center)
    vae_out = vae_run_root(args, protocol, seed, center)
    direct_runs = direct_out / "runs"
    log = args.log_root / run_id / f"{center}.log"
    env = base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    direct_cmd = [
        args.python,
        "-u",
        "scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py",
        "--centers",
        center,
        "--linear_probe_dir",
        str(args.linear_probe_dir),
        "--preprocess_policy",
        "official_ptbxl_eval",
        "--out_dir",
        str(direct_out),
        "--ref_root",
        str(args.subset_root),
        "--k",
        str(k),
        "--source_k",
        str(k),
        "--subset_seed",
        str(seed),
        "--seed",
        str(seed),
        "--epochs",
        str(args.direct_epochs),
        "--lr",
        str(args.direct_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--val_fraction",
        "0.2",
        "--batch_size",
        str(args.direct_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--device",
        "cuda",
    ]
    vae_cmd = [
        args.python,
        "-u",
        "scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py",
        "--centers",
        center,
        "--out_dir",
        str(vae_out),
        "--linear_probe_dir",
        str(args.linear_probe_dir),
        "--preprocess_policy",
        "official_ptbxl_eval",
        "--k",
        str(k),
        "--k_anchor",
        str(args.k_anchor),
        "--anchor_sample_mode",
        "hard_bce",
        "--anchor_sample_power",
        "1.5",
        "--anchor_sample_min_weight",
        "0.0001",
        "--hull_m",
        "20",
        "--hull_lambda",
        "0.15",
        "--hull_steps",
        "5",
        "--hull_lr",
        "0.25",
        "--hull_weight_mode",
        "optimized",
        "--hull_label_mode",
        "primary",
        "--hull_mix_label_mode",
        "anchor",
        "--hull_include_anchor",
        "--pgd_batch",
        str(args.pgd_batch),
        "--epochs",
        str(args.vae_epochs),
        "--eval_every",
        "1",
        "--lr",
        str(args.vae_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--batch_size",
        str(args.vae_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--head_type",
        "residual_adapter",
        "--adapter_hidden",
        "128",
        "--adapter_dropout",
        "0.0",
        "--adapter_scale",
        "1.0",
        "--freeze_base_head",
        "--init_base_head_from_k500_root",
        str(direct_runs),
        "--source_weight",
        "1.0",
        "--target_real_weight",
        "120.0",
        "--adv_weight",
        "80.0",
        "--selection_metric",
        "target_auprc",
        "--selection_source",
        "target_real_val",
        "--target_real_val_fraction",
        "0.2",
        "--target_real_val_seed",
        str(seed),
        "--source_selection_weight",
        "0.25",
        "--source_auprc_floor",
        "0.79",
        "--source_floor_penalty",
        "5.0",
        "--anchor_base_root",
        str(args.subset_root),
        "--ref_root",
        str(args.subset_root),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--report_drop_all_zero_pn2021",
    ]
    run_stream(direct_cmd, log_path=log, env=env, dry_run=args.dry_run)
    run_stream(vae_cmd, log_path=log, env=env, dry_run=args.dry_run, append=True)
    print(f"[done] {run_id} center={center} K={k} gpu={gpu}", flush=True)


def write_worker_args(args: argparse.Namespace, run_id: str) -> Path:
    path = args.log_root / run_id / "worker_args.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "python": args.python,
        "data_root": str(args.data_root),
        "subset_root": str(args.subset_root),
        "source_linear_probe_dir": str(args.source_linear_probe_dir),
        "linear_probe_dir": str(args.linear_probe_dir),
        "log_root": str(args.log_root),
        "tmp_root": str(args.tmp_root),
        "cache_root": str(args.cache_root),
        "centers": args.centers,
        "gpus": args.gpus,
        "direct_epochs": args.direct_epochs,
        "vae_epochs": args.vae_epochs,
        "direct_lr": args.direct_lr,
        "vae_lr": args.vae_lr,
        "weight_decay": args.weight_decay,
        "direct_batch_size": args.direct_batch_size,
        "vae_batch_size": args.vae_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "k_anchor": args.k_anchor,
        "pgd_batch": args.pgd_batch,
        "run_stamp": args.run_stamp,
        "dry_run": args.dry_run,
        "force_relabel": args.force_relabel,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args_for_worker(path: Path) -> argparse.Namespace:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in [
        "data_root",
        "subset_root",
        "source_linear_probe_dir",
        "linear_probe_dir",
        "log_root",
        "tmp_root",
        "cache_root",
    ]:
        data[key] = Path(data[key])
    return argparse.Namespace(**data)


def run_protocol_seed(seed: int, protocol: str, subset_items: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    run_id = run_id_for(protocol, seed, args)
    manifest = {
        "run_id": run_id,
        "protocol": protocol,
        "seed": seed,
        "centers": args.centers,
        "gpus": args.gpus,
        "mapping_version": MAPPING_VERSION,
        "mapping_hash": MAPPING_HASH,
        "items": [],
    }
    procs: list[tuple[str, subprocess.Popen[str], Path]] = []
    for center, gpu in zip(args.centers, args.gpus):
        item = subset_items.get(f"{center}::{protocol}")
        if item is None:
            if not args.dry_run:
                raise KeyError(f"missing exported subset summary for {center}::{protocol}")
            k = int(PROTOCOLS[protocol]["fixed_k"] or 0)
            subset_base = "<dry-run:subset-not-exported>"
        else:
            k = int(item["K"])
            subset_base = item["base"]
        manifest["items"].append({"center": center, "K": k, "gpu": gpu, "subset_base": subset_base})
        if args.dry_run:
            run_center_pair(center=center, k=k, protocol=protocol, seed=seed, gpu=gpu, args=args)
            continue
        wrapper = [
            sys.executable,
            "-c",
            (
                "import json, pathlib, sys; "
                "from scripts.paper.run_ecgfounder_v7_multiseed_kshot_matrix_20260530 import "
                "run_center_pair, parse_args_for_worker; "
                "args=parse_args_for_worker(pathlib.Path(sys.argv[1])); "
                "spec=json.loads(sys.argv[2]); "
                "run_center_pair(center=spec['center'], k=spec['k'], protocol=spec['protocol'], "
                "seed=spec['seed'], gpu=spec['gpu'], args=args)"
            ),
            str(write_worker_args(args, run_id)),
            json.dumps({"center": center, "k": k, "protocol": protocol, "seed": seed, "gpu": gpu}),
        ]
        proc = subprocess.Popen(wrapper, cwd=str(REPO), env=base_env(args), text=True)
        procs.append((center, proc, args.log_root / run_id / f"{center}.log"))
        print(f"[launched] {run_id} center={center} K={k} gpu={gpu} pid={proc.pid}", flush=True)
    failures = []
    for center, proc, log_path in procs:
        ret = proc.wait()
        if ret != 0:
            failures.append((center, ret, str(log_path)))
    manifest_path = args.log_root / run_id / "run_matrix_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{run_id} failed: {failures}")
    postprocess(protocol, seed, args)


def postprocess(protocol: str, seed: int, args: argparse.Namespace) -> None:
    run_id = run_id_for(protocol, seed, args)
    for method in ["direct", "vae_lhat"]:
        exp_name = f"ecgfounder_{method}_{protocol}_v7_sjr_rgq"
        input_root = args.data_root / "runs" / exp_name / run_id
        out_dir = args.data_root / "runs" / "metrics_export" / f"{run_id}_{exp_name}"
        cmd = [
            args.python,
            "scripts/export_metrics_long.py",
            "--input",
            str(input_root),
            "--output-dir",
            str(out_dir),
            "--run-id",
            exp_name,
            "--filter-to-target-center",
            "--target-centers",
            *args.centers,
            "--expected-mapping-version",
            MAPPING_VERSION,
            "--expected-mapping-hash",
            MAPPING_HASH,
        ]
        run_stream(
            cmd,
            log_path=args.log_root / run_id / f"postprocess_{method}_metrics_long.log",
            env=base_env(args),
            dry_run=args.dry_run,
        )
        metrics_long = out_dir / "metrics_long.csv"
        if args.dry_run:
            available_views = {"pn2021_all_zero_kept_refexcluded", "pn2021_drop_all_zero_refexcluded"}
        else:
            with metrics_long.open(newline="") as f:
                available_views = {row["canonical_view"] for row in csv.DictReader(f)}
        for view, suffix in [
            ("pn2021_all_zero_kept_refexcluded", "all_zero_kept"),
            ("pn2021_drop_all_zero_refexcluded", "drop_all_zero"),
        ]:
            if view not in available_views:
                print(f"[skip-paper-table] {run_id} {method} missing view={view}", flush=True)
                continue
            table_dir = args.data_root / "runs" / "paper_table" / f"{run_id}_{exp_name}_{suffix}"
            table_cmd = [
                args.python,
                "scripts/export_paper_table.py",
                "--metrics-long",
                str(metrics_long),
                "--output-dir",
                str(table_dir),
                "--view",
                view,
                "--dataset",
                "pn2021",
                "--centers",
                *args.centers,
                "--no-dataset-rows",
            ]
            run_stream(
                table_cmd,
                log_path=args.log_root / run_id / f"postprocess_{method}_{suffix}.log",
                env=base_env(args),
                dry_run=args.dry_run,
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--protocols", nargs="+", choices=sorted(PROTOCOLS), default=["k500", "p10", "p20"])
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET_ROOT)
    ap.add_argument("--source-linear-probe-dir", type=Path, default=DEFAULT_SOURCE_LINEAR_PROBE)
    ap.add_argument("--linear-probe-dir", type=Path, default=DEFAULT_V7_LINEAR_PROBE)
    ap.add_argument("--log-root", type=Path, default=DATA_ROOT / "runs/logs/ecgfounder_v7_multiseed_matrix_20260530")
    ap.add_argument("--tmp-root", type=Path, default=Path("/home/linbinhao/tmp_ecg"))
    ap.add_argument("--cache-root", type=Path, default=DATA_ROOT / "cache")
    ap.add_argument("--run-stamp", default="20260530")
    ap.add_argument("--direct-epochs", type=int, default=50)
    ap.add_argument("--vae-epochs", type=int, default=20)
    ap.add_argument("--direct-lr", type=float, default=5e-4)
    ap.add_argument("--vae-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--direct-batch-size", type=int, default=128)
    ap.add_argument("--vae-batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--k-anchor", type=int, default=300)
    ap.add_argument("--pgd-batch", type=int, default=16)
    ap.add_argument("--force-relabel", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(args.gpus) < len(args.centers):
        raise SystemExit("--gpus must provide at least one GPU id per center")
    args.gpus = args.gpus[: len(args.centers)]
    return args


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    ensure_v7_linear_probe(args)
    for seed in args.seeds:
        subset_items = export_subsets(seed, args.protocols, args)
        for protocol in args.protocols:
            run_protocol_seed(seed, protocol, subset_items, args)


if __name__ == "__main__":
    main()
