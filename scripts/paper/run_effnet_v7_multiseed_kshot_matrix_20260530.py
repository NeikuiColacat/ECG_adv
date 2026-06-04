#!/usr/bin/env python3
"""Run EfficientNet1DV2 v7 Direct/VAE-LHAT K-shot replication matrix.

This runner replaces the temporary shell snippets used during the first
K500/10%/20% checks.  For each seed and sampling protocol it:

1. exports matched v7 real-signal and ECGTwin-latent anchor subsets;
2. runs per-center Direct fine-tuning followed by VAE-LHAT from the matched
   Direct checkpoint;
3. exports metrics_long and paper_table artifacts for both methods.

It intentionally keeps the current frozen EfficientNet recipe unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
PYTHON = "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
DEFAULT_SUBSET_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_GPUS = ["0", "1", "2", "3"]
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"


PROTOCOLS = {
    "k500": {"fixed_k": 500, "percent": None, "method_tag": "k500"},
    "p10": {"fixed_k": None, "percent": 0.10, "method_tag": "p10"},
    "p20": {"fixed_k": None, "percent": 0.20, "method_tag": "p20"},
}


def run_stream(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    cwd: Path = REPO,
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
            cwd=str(cwd),
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


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["ECG_ADV_GEN_DATA_ROOT"] = str(args.data_root)
    env["TMPDIR"] = str(args.tmp_root)
    env["XDG_CACHE_HOME"] = str(args.cache_root)
    env["PYTHONUNBUFFERED"] = "1"
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    return env


def export_subsets(seed: int, protocols: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    percents = [PROTOCOLS[p]["percent"] for p in protocols if PROTOCOLS[p]["percent"] is not None]
    fixed_ks = [PROTOCOLS[p]["fixed_k"] for p in protocols if PROTOCOLS[p]["fixed_k"] is not None]
    summary_path = args.subset_root / f"replication_v7_seed{seed}_{'_'.join(protocols)}_summary.json"
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


def direct_eval_path(args: argparse.Namespace, protocol: str, seed: int, center: str, k: int) -> Path:
    run_id = direct_run_id_for(protocol, seed, args)
    return (
        args.data_root
        / "runs"
        / f"effnet_direct_{protocol}_v7_sjr_rgq"
        / run_id
        / "runs"
        / f"{center}_K{k}_direct_ft_ep30_seed{seed}_val0.2"
        / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json"
    )


def vae_eval_path(args: argparse.Namespace, protocol: str, seed: int, center: str) -> Path:
    run_id = run_id_for(protocol, seed, args)
    root = args.data_root / "runs" / f"effnet_{vae_method_name(args)}_{protocol}_v7_sjr_rgq" / run_id
    matches = sorted(root.glob(f"{center}_realall_*_fullft_{protocol}_ep30_seed{seed}/eval_result_v7_exclrefs_crop1000.json"))
    return matches[0] if matches else root / f"{center}_MISSING_eval_result_v7_exclrefs_crop1000.json"


def run_id_for(protocol: str, seed: int, args: argparse.Namespace) -> str:
    return f"seed{seed}_{protocol}_v7_sjr_rgq_{args.run_stamp}"


def direct_run_id_for(protocol: str, seed: int, args: argparse.Namespace) -> str:
    stamp = getattr(args, "direct_run_stamp", "") or args.run_stamp
    return f"seed{seed}_{protocol}_v7_sjr_rgq_{stamp}"


def vae_method_name(args: argparse.Namespace) -> str:
    return "vae_noaug" if getattr(args, "disable_latent_augmix_branch", False) else "vae_lhat"


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
    direct_run_id = direct_run_id_for(protocol, seed, args)
    direct_out = args.data_root / "runs" / f"effnet_direct_{protocol}_v7_sjr_rgq" / direct_run_id
    vae_out = args.data_root / "runs" / f"effnet_{vae_method_name(args)}_{protocol}_v7_sjr_rgq" / run_id
    env = base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = args.log_root / run_id / f"{center}.log"
    direct_ckpt = direct_out / "runs" / f"{center}_K{k}_direct_ft_ep30_seed{seed}_val0.2" / "best_model.pt"
    direct_cmd = [
        args.python,
        "-u",
        "scripts/paper/run_direct_finetune_k500_20260516.py",
        "--centers",
        center,
        "--k",
        str(k),
        "--subset_seed",
        str(seed),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--crop_len",
        "1000",
        "--val_fraction",
        "0.2",
        "--num_workers",
        str(args.num_workers),
        "--device",
        "cuda",
        "--pos_weight_clip_max",
        "50.0",
        "--subset_root",
        str(args.subset_root),
        "--out_root",
        str(direct_out),
    ]
    vae_cmd = [
        args.python,
        "-u",
        "scripts/paper/run_effnet_latent_augmix_stage3_20260524.py",
        "--center",
        center,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--num_workers",
        str(args.num_workers),
        "--data_root",
        str(args.data_root),
        "--out_root",
        str(vae_out),
        "--init_ckpt",
        str(direct_ckpt),
        "--anchor_base",
        str(args.subset_root / center / f"k{k}_seed{seed}" / f"{center}_real_k{k}_seed{seed}"),
        "--hull_steps",
        "3",
        "--hull_M",
        "20",
        "--hull_lambda",
        "0.05",
        "--hull_lr",
        "0.25",
        "--hull_include_anchor",
        "--hull_label_mode",
        "compatible",
        "--hull_mix_label_mode",
        "anchor_soft",
        "--hull_label_lambda_y",
        "0.25",
        "--hull_label_new_class_cap",
        "0.25",
        "--hull_neighbor_distance_space",
        "standardized",
        "--hull_neighbor_mode",
        "local_random",
        "--hull_neighbor_pool_size",
        "120",
        "--hull_neighbor_pool_multiplier",
        "4",
        "--k_anchor",
        "300",
        "--pgd_batch",
        "32",
        "--target_real_weight",
        "80.0",
        "--adv_weight",
        "0.3",
        "--adv_weight_warmup_epochs",
        "10",
        "--adv_label_mode",
        "latent_mixed_teacher",
        "--adv_teacher_mix",
        "0.4",
        "--lr",
        str(args.lr),
        "--train_batch_size",
        str(args.batch_size),
        "--ptbxl_weight",
        "1.0",
        "--latent_augmix_latent_weight_cap",
        str(args.latent_augmix_latent_weight_cap),
        "--latent_augmix_copies",
        str(args.latent_augmix_copies),
        "--latent_augmix_width",
        str(args.latent_augmix_width),
        "--latent_augmix_depth",
        str(args.latent_augmix_depth),
        "--latent_augmix_alpha",
        str(args.latent_augmix_alpha),
        "--latent_augmix_severity",
        str(args.latent_augmix_severity),
        "--latent_augmix_ops",
        *args.latent_augmix_ops,
        "--quick_eval_source",
        "target_real_val",
        "--target_real_val_fraction",
        "0.2",
        "--target_real_val_seed",
        str(seed),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--eval_min_pos",
        "10",
        "--eval_pn2021_limit",
        "0",
        "--run_tag_extra",
        f"{protocol}_noaugmix" if args.disable_latent_augmix_branch else protocol,
    ]
    if args.disable_latent_augmix_branch:
        vae_cmd.append("--disable_latent_augmix_branch")
    if args.enable_raw_corrupt_consistency:
        vae_cmd.extend([
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_copies",
            str(args.raw_corrupt_copies),
            "--raw_corrupt_prob",
            str(args.raw_corrupt_prob),
            "--raw_corrupt_severity",
            str(args.raw_corrupt_severity),
            "--raw_corrupt_ops",
            *args.raw_corrupt_ops,
            "--raw_corrupt_consistency_weight",
            str(args.raw_corrupt_consistency_weight),
            "--raw_corrupt_consistency_loss",
            str(args.raw_corrupt_consistency_loss),
            "--raw_corrupt_bce_weight",
            str(args.raw_corrupt_bce_weight),
            "--raw_corrupt_max_batches",
            str(args.raw_corrupt_max_batches),
            "--raw_corrupt_scope",
            str(args.raw_corrupt_scope),
            "--raw_corrupt_clip_abs",
            str(args.raw_corrupt_clip_abs),
        ])
        if args.raw_corrupt_no_renorm:
            vae_cmd.append("--raw_corrupt_no_renorm")
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"echo '[start]' $(date -Is) protocol={protocol} seed={seed} center={center} K={k} gpu={gpu}\n"
        + f"if test -s {direct_ckpt}; then echo '[skip direct] existing {direct_ckpt}'; else "
        + " ".join(direct_cmd)
        + "; fi"
        + "\n"
        f"test -s {direct_ckpt}\n"
        + " ".join(vae_cmd)
        + "\n"
        f"echo '[done]' $(date -Is) protocol={protocol} seed={seed} center={center} K={k} gpu={gpu}\n"
    )
    tmp_script = args.log_root / run_id / f"{center}.sh"
    tmp_script.parent.mkdir(parents=True, exist_ok=True)
    tmp_script.write_text(script, encoding="utf-8")
    run_stream(["bash", str(tmp_script)], log_path=log, env=env, dry_run=args.dry_run)


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
    args.log_root.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen[str], Path]] = []
    for center, gpu in zip(args.centers, args.gpus):
        item = subset_items.get(f"{center}::{protocol}")
        if item is None:
            if not args.dry_run:
                raise KeyError(f"missing exported subset summary for {center}::{protocol}")
            fixed_k = PROTOCOLS[protocol]["fixed_k"]
            k = int(fixed_k) if fixed_k is not None else 0
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
                "from scripts.paper.run_effnet_v7_multiseed_kshot_matrix_20260530 import run_center_pair, parse_args_for_worker; "
                "args=parse_args_for_worker(pathlib.Path(sys.argv[1])); "
                "spec=json.loads(sys.argv[2]); "
                "run_center_pair(center=spec['center'], k=spec['k'], protocol=spec['protocol'], seed=spec['seed'], gpu=spec['gpu'], args=args)"
            ),
            str(write_worker_args(args, run_id)),
            json.dumps({"center": center, "k": k, "protocol": protocol, "seed": seed, "gpu": gpu}),
        ]
        env = base_env(args)
        proc = subprocess.Popen(wrapper, cwd=str(REPO), env=env, text=True)
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


def write_worker_args(args: argparse.Namespace, run_id: str) -> Path:
    path = args.log_root / run_id / "worker_args.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "python": args.python,
        "data_root": str(args.data_root),
        "subset_root": str(args.subset_root),
        "log_root": str(args.log_root),
        "tmp_root": str(args.tmp_root),
        "cache_root": str(args.cache_root),
        "centers": args.centers,
        "gpus": args.gpus,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "run_stamp": args.run_stamp,
        "direct_run_stamp": args.direct_run_stamp,
        "disable_latent_augmix_branch": args.disable_latent_augmix_branch,
        "latent_augmix_latent_weight_cap": args.latent_augmix_latent_weight_cap,
        "latent_augmix_copies": args.latent_augmix_copies,
        "latent_augmix_width": args.latent_augmix_width,
        "latent_augmix_depth": args.latent_augmix_depth,
        "latent_augmix_alpha": args.latent_augmix_alpha,
        "latent_augmix_severity": args.latent_augmix_severity,
        "latent_augmix_ops": list(args.latent_augmix_ops),
        "enable_raw_corrupt_consistency": args.enable_raw_corrupt_consistency,
        "raw_corrupt_copies": args.raw_corrupt_copies,
        "raw_corrupt_prob": args.raw_corrupt_prob,
        "raw_corrupt_severity": args.raw_corrupt_severity,
        "raw_corrupt_ops": list(args.raw_corrupt_ops),
        "raw_corrupt_consistency_weight": args.raw_corrupt_consistency_weight,
        "raw_corrupt_consistency_loss": args.raw_corrupt_consistency_loss,
        "raw_corrupt_bce_weight": args.raw_corrupt_bce_weight,
        "raw_corrupt_max_batches": args.raw_corrupt_max_batches,
        "raw_corrupt_scope": args.raw_corrupt_scope,
        "raw_corrupt_no_renorm": args.raw_corrupt_no_renorm,
        "raw_corrupt_clip_abs": args.raw_corrupt_clip_abs,
        "dry_run": args.dry_run,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args_for_worker(path: Path) -> argparse.Namespace:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["data_root", "subset_root", "log_root", "tmp_root", "cache_root"]:
        data[key] = Path(data[key])
    return argparse.Namespace(**data)


def postprocess(protocol: str, seed: int, args: argparse.Namespace) -> None:
    run_id = run_id_for(protocol, seed, args)
    for method in ["direct", vae_method_name(args)]:
        exp_name = f"effnet_{method}_{protocol}_v7_sjr_rgq"
        input_run_id = direct_run_id_for(protocol, seed, args) if method == "direct" else run_id
        input_root = args.data_root / "runs" / exp_name / input_run_id
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
        for view, suffix in [
            ("pn2021_all_zero_kept_refexcluded", "all_zero_kept"),
            ("pn2021_drop_all_zero_refexcluded", "drop_all_zero"),
        ]:
            table_dir = args.data_root / "runs" / "paper_table" / f"{run_id}_{exp_name}_{suffix}"
            table_cmd = [
                args.python,
                "scripts/export_paper_table.py",
                "--metrics-long",
                str(out_dir / "metrics_long.csv"),
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
    ap.add_argument("--log-root", type=Path, default=DATA_ROOT / "runs/logs/effnet_v7_multiseed_matrix_20260530")
    ap.add_argument("--tmp-root", type=Path, default=Path("/home/linbinhao/tmp_ecg"))
    ap.add_argument("--cache-root", type=Path, default=DATA_ROOT / "cache")
    ap.add_argument("--run-stamp", default="20260530")
    ap.add_argument(
        "--direct-run-stamp",
        default="",
        help=(
            "Optional existing Direct run stamp to reuse as the init checkpoint "
            "for VAE/AugMix sweeps. Existing checkpoints are skipped instead "
            "of retrained."
        ),
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=192)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--disable-latent-augmix-branch",
        action="store_true",
        help=(
            "Run the VAE branch without latent-branch AugMix and store it as "
            "effnet_vae_noaug_<protocol>_v7_sjr_rgq. Direct controls are "
            "unchanged and skipped when already present."
        ),
    )
    ap.add_argument("--latent-augmix-latent-weight-cap", type=float, default=0.25)
    ap.add_argument("--latent-augmix-copies", type=int, default=1)
    ap.add_argument("--latent-augmix-width", type=int, default=3)
    ap.add_argument("--latent-augmix-depth", type=int, default=-1)
    ap.add_argument("--latent-augmix-alpha", type=float, default=1.0)
    ap.add_argument("--latent-augmix-severity", type=int, default=2)
    ap.add_argument(
        "--latent-augmix-ops",
        nargs="+",
        default=["powerline_noise", "emg_noise", "baseline_wander", "baseline_shift"],
        help=(
            "Forwarded to the VAE-LHAT wrapper. Pass all five PN2021-C ops, "
            "including random_leads_masking, for robustness sweeps."
        ),
    )
    ap.add_argument("--enable-raw-corrupt-consistency", action="store_true")
    ap.add_argument("--raw-corrupt-copies", type=int, default=1)
    ap.add_argument("--raw-corrupt-prob", type=float, default=0.5)
    ap.add_argument("--raw-corrupt-severity", type=int, default=4)
    ap.add_argument(
        "--raw-corrupt-ops",
        nargs="+",
        default=[
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
        help="Forwarded to the target-real raw corruption consistency branch.",
    )
    ap.add_argument("--raw-corrupt-consistency-weight", type=float, default=0.5)
    ap.add_argument(
        "--raw-corrupt-consistency-loss",
        choices=["soft_bce", "jsd"],
        default="soft_bce",
    )
    ap.add_argument("--raw-corrupt-bce-weight", type=float, default=0.1)
    ap.add_argument("--raw-corrupt-max-batches", type=int, default=0)
    ap.add_argument(
        "--raw-corrupt-scope",
        choices=["target", "source", "source_target"],
        default="target",
    )
    ap.add_argument("--raw-corrupt-no-renorm", action="store_true")
    ap.add_argument("--raw-corrupt-clip-abs", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(args.gpus) < len(args.centers):
        raise SystemExit("--gpus must provide at least one GPU id per center")
    args.gpus = args.gpus[: len(args.centers)]
    return args


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        subset_items = export_subsets(seed, args.protocols, args)
        for protocol in args.protocols:
            run_protocol_seed(seed, protocol, subset_items, args)


if __name__ == "__main__":
    main()
