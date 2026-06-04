#!/usr/bin/env python3
"""Run v7 Direct/VAE-LHAT K-shot matrix for classic benchmark ECG backbones.

This runner is intentionally separate from the EfficientNet-specific runner so
the ongoing mainline replication scripts remain unchanged.  It supports any
architecture registered in ``scripts.triple_labels.model_zoo`` and uses the
same v7 PN2021 ref-excluded evaluation contract.
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

from ecg_adv_gen.data.class_trust import write_real_all_present_trust  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from scripts.triple_labels.model_zoo import available_model_names, normalize_model_name  # noqa: E402


DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
PYTHON = "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
DEFAULT_SUBSET_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_GPUS = ["0", "1", "2", "3"]
DEFAULT_MODELS = ["benchmark_resnet1d_wang", "benchmark_inception1d", "benchmark_fcn_wang"]
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"


PROTOCOLS = {
    "k500": {"fixed_k": 500, "percent": None},
    "p10": {"fixed_k": None, "percent": 0.10},
    "p20": {"fixed_k": None, "percent": 0.20},
}


def model_slug(model_name: str) -> str:
    return normalize_model_name(model_name).replace("/", "_").replace(" ", "_")


def direct_leaf(center: str, k: int, model_name: str, epochs: int, seed: int, val_fraction: float) -> str:
    val_tag = str(val_fraction).replace(".", "p")
    return f"{center}_K{k}_direct_ft_{model_slug(model_name)}_ep{epochs}_seed{seed}_val{val_tag}"


def vae_leaf(center: str, model_name: str, protocol: str, epochs: int, seed: int) -> str:
    return f"{center}_realall_{model_slug(model_name)}_fullft_{protocol}_ep{epochs}_seed{seed}"


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


def source_run_id(model_name: str, seed: int, args: argparse.Namespace) -> str:
    return f"{model_slug(model_name)}_seed{seed}_v7_sjr_rgq_{args.run_stamp}"


def run_id_for(model_name: str, protocol: str, seed: int, args: argparse.Namespace) -> str:
    return f"{model_slug(model_name)}_seed{seed}_{protocol}_v7_sjr_rgq_{args.run_stamp}"


def source_dir(model_name: str, seed: int, args: argparse.Namespace) -> Path:
    return args.data_root / "runs" / "benchmark_source_v7_sjr_rgq" / source_run_id(model_name, seed, args)


def train_source_model(model_name: str, seed: int, args: argparse.Namespace) -> Path:
    out_dir = source_dir(model_name, seed, args)
    ckpt = out_dir / "best_model.pt"
    if ckpt.exists() and not args.force_source:
        print(f"[skip source] {model_name} seed={seed}: {ckpt}", flush=True)
        return ckpt
    env = base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = str(args.source_gpu)
    cmd = [
        args.python,
        "-u",
        "scripts/triple_labels/train_ptbxl.py",
        "--scheme",
        "super5",
        "--model_name",
        normalize_model_name(model_name),
        "--output_dir",
        str(out_dir),
        "--data_path",
        str(args.data_root / "ptbxl/raw100.npy"),
        "--csv_path",
        str(args.data_root / "ptbxl/ptbxl_database.csv"),
        "--cache_path",
        str(args.data_root / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.source_batch_size),
        "--epochs",
        str(args.source_epochs),
        "--lr",
        str(args.source_lr),
        "--weight_decay",
        str(args.source_weight_decay),
        "--cosine_tmax",
        str(max(args.source_epochs, 1)),
        "--patience",
        str(args.source_patience),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--checkpoint_metric",
        args.source_checkpoint_metric,
        "--pos_weight_clip_max",
        "50.0",
    ]
    run_stream(
        cmd,
        log_path=args.log_root / source_run_id(model_name, seed, args) / "source_train.log",
        env=env,
        dry_run=args.dry_run,
    )
    return ckpt


def export_subsets(seed: int, protocols: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    percents = [PROTOCOLS[p]["percent"] for p in protocols if PROTOCOLS[p]["percent"] is not None]
    fixed_ks = [PROTOCOLS[p]["fixed_k"] for p in protocols if PROTOCOLS[p]["fixed_k"] is not None]
    summary_path = args.subset_root / f"benchmark_v7_seed{seed}_{'_'.join(protocols)}_summary.json"
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


def run_center_pair(
    *,
    model_name: str,
    source_ckpt: Path,
    center: str,
    k: int,
    protocol: str,
    seed: int,
    gpu: str,
    args: argparse.Namespace,
) -> None:
    slug = model_slug(model_name)
    run_id = run_id_for(model_name, protocol, seed, args)
    direct_out = args.data_root / "runs" / f"{slug}_direct_{protocol}_v7_sjr_rgq" / run_id
    vae_out = args.data_root / "runs" / f"{slug}_vae_lhat_{protocol}_v7_sjr_rgq" / run_id
    direct_ckpt = direct_out / "runs" / direct_leaf(center, k, model_name, args.epochs, seed, args.val_fraction) / "best_model.pt"
    direct_eval = direct_ckpt.parent / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json"
    vae_dir = vae_out / vae_leaf(center, model_name, protocol, args.epochs, seed)
    vae_eval = vae_dir / "eval_result_v7_exclrefs_crop1000.json"
    anchor_base = args.subset_root / center / f"k{k}_seed{seed}" / f"{center}_real_k{k}_seed{seed}"
    signal_npz = anchor_base.with_suffix(".signals.npz")
    latent_npz = anchor_base.with_suffix(".latent.npz")
    ref_meta = anchor_base.with_suffix(".ref_meta.json")
    class_trust = write_real_all_present_trust(
        signal_npz,
        vae_out / "config",
        center,
        policy="benchmark-backbone v7 real-anchor VAE-LHAT; trust classes present in the K-shot subset",
    )

    env = base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    direct_cmd = [
        args.python,
        "-u",
        "scripts/paper/run_benchmark_direct_finetune_v7_20260530.py",
        "--centers",
        center,
        "--k",
        str(k),
        "--subset_seed",
        str(seed),
        "--seed",
        str(seed),
        "--model_name",
        normalize_model_name(model_name),
        "--init_ckpt",
        str(source_ckpt),
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
        str(args.val_fraction),
        "--num_workers",
        str(args.num_workers),
        "--device",
        "cuda",
        "--data_root",
        str(args.data_root),
        "--python",
        args.python,
        "--subset_root",
        str(args.subset_root),
        "--out_root",
        str(direct_out),
    ]
    if args.force_direct:
        direct_cmd.append("--force")

    train_cmd = [
        args.python,
        "-u",
        "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name",
        center,
        "--ref_meta_json",
        str(ref_meta),
        "--synth_npz",
        str(latent_npz),
        "--target_real_npz",
        str(signal_npz),
        "--class_trust",
        str(class_trust),
        "--init_ckpt",
        str(direct_ckpt),
        "--model_name",
        normalize_model_name(model_name),
        "--output_dir",
        str(vae_dir),
        "--data_dir",
        str(args.data_root / "physionet2021/training"),
        "--quick_eval_source",
        "target_real_val",
        "--quick_eval_centers",
        center,
        "--quick_eval_n_per_center",
        str(args.quick_eval_n_per_center),
        "--target_real_val_fraction",
        str(args.val_fraction),
        "--target_real_val_seed",
        str(seed),
        "--ptbxl_raw",
        str(args.data_root / "ptbxl/raw100.npy"),
        "--ptbxl_csv",
        str(args.data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_prep",
        str(args.data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--attack_mode",
        "latent_hull",
        "--hull_M",
        "20",
        "--hull_lambda",
        "0.05",
        "--hull_steps",
        "3",
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
        "--source_sampling_strategy",
        "source_weighted",
        "--source_weights",
        "real_anchor=1.0",
        "--anchor_class_weight_mode",
        "inv_freq_kshot",
        "--anchor_class_weight_reference_source",
        "real_anchor",
        "--anchor_class_weight_gamma",
        "0.5",
        "--anchor_class_weight_min",
        "0.35",
        "--anchor_class_weight_cap",
        "4.0",
        "--anchor_class_missing_weight",
        "0.35",
        "--K_anchor",
        "300",
        "--pgd_batch",
        "32",
        "--classes_in_scope",
        *CLASS_NAMES_SUPER5,
        "--allow_hyp_cd_trust",
        "--target_real_weight",
        "80.0",
        "--adv_weight",
        "0.3",
        "--adv_weight_warmup_epochs",
        "10",
        "--ptbxl_weight",
        "1.0",
        "--roundtrip_weight",
        "0.0",
        "--roundtrip_anchor_n",
        "0",
        "--adv_label_mode",
        "latent_mixed_teacher",
        "--adv_teacher_mix",
        "0.4",
        "--disable_quality_gate",
        "--lr",
        str(args.lr),
        "--weight_decay",
        "1e-4",
        "--batch_size",
        str(args.batch_size),
        "--n_epochs",
        str(args.epochs),
        "--patience",
        str(args.epochs),
        "--eval_every",
        "2",
        "--es_metric",
        "target_macro_auprc",
        "--ewa_decay",
        "0.999",
        "--anchor_lambda",
        "0.05",
        "--asr_consec_low_max",
        "999",
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--crop_len",
        "1000",
        "--device",
        "cuda",
    ]
    if not args.disable_latent_augmix_branch:
        train_cmd.extend([
            "--enable_latent_augmix_branch",
            "--latent_augmix_copies",
            "1",
            "--latent_augmix_width",
            "3",
            "--latent_augmix_depth",
            "-1",
            "--latent_augmix_alpha",
            "1.0",
            "--latent_augmix_severity",
            "2",
            "--latent_augmix_latent_weight_cap",
            "0.25",
            "--latent_augmix_ops",
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
        ])

    eval_cmd = [
        args.python,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(vae_dir),
        "--model_name",
        normalize_model_name(model_name),
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.eval_batch_size),
        "--min_pos",
        "10",
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_csv",
        str(args.data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache",
        str(args.data_root / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_root",
        str(args.data_root / "physionet2021"),
        "--pn2021_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--pn2021_mmap_cache_dir",
        str(args.data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
        "--exclude_ref_ids",
        str(ref_meta),
        "--output_path",
        str(vae_eval),
    ]
    if args.eval_pn2021_limit:
        eval_cmd.extend(["--pn2021_limit", str(args.eval_pn2021_limit)])

    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"echo '[start]' $(date -Is) model={slug} protocol={protocol} seed={seed} center={center} K={k} gpu={gpu}\n"
        + f"if test -s {direct_eval} && test -s {direct_ckpt}; then echo '[skip direct] existing {direct_eval}'; else "
        + " ".join(direct_cmd)
        + "; fi\n"
        + f"test -s {direct_ckpt}\n"
        + f"if test -s {vae_eval}; then echo '[skip vae] existing {vae_eval}'; else "
        + " ".join(train_cmd)
        + "; "
        + " ".join(eval_cmd)
        + "; fi\n"
        + f"echo '[done]' $(date -Is) model={slug} protocol={protocol} seed={seed} center={center} K={k} gpu={gpu}\n"
    )
    tmp_script = args.log_root / run_id / f"{center}.sh"
    tmp_script.parent.mkdir(parents=True, exist_ok=True)
    tmp_script.write_text(script, encoding="utf-8")
    run_stream(["bash", str(tmp_script)], log_path=args.log_root / run_id / f"{center}.log", env=env, dry_run=args.dry_run)


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
        "models": args.models,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "run_stamp": args.run_stamp,
        "val_fraction": args.val_fraction,
        "quick_eval_n_per_center": args.quick_eval_n_per_center,
        "disable_latent_augmix_branch": args.disable_latent_augmix_branch,
        "eval_pn2021_limit": args.eval_pn2021_limit,
        "force_direct": args.force_direct,
        "dry_run": args.dry_run,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args_for_worker(path: Path) -> argparse.Namespace:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["data_root", "subset_root", "log_root", "tmp_root", "cache_root"]:
        data[key] = Path(data[key])
    return argparse.Namespace(**data)


def run_model_protocol_seed(
    model_name: str,
    seed: int,
    protocol: str,
    source_ckpt: Path,
    subset_items: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    run_id = run_id_for(model_name, protocol, seed, args)
    manifest = {
        "run_id": run_id,
        "model_name": normalize_model_name(model_name),
        "protocol": protocol,
        "seed": seed,
        "source_ckpt": str(source_ckpt),
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
            fixed_k = PROTOCOLS[protocol]["fixed_k"]
            k = int(fixed_k) if fixed_k is not None else 0
            subset_base = "<dry-run:subset-not-exported>"
        else:
            k = int(item["K"])
            subset_base = item["base"]
        manifest["items"].append({"center": center, "K": k, "gpu": gpu, "subset_base": subset_base})
        if args.dry_run:
            run_center_pair(
                model_name=model_name,
                source_ckpt=source_ckpt,
                center=center,
                k=k,
                protocol=protocol,
                seed=seed,
                gpu=gpu,
                args=args,
            )
            continue
        wrapper = [
            sys.executable,
            "-c",
            (
                "import json, pathlib, sys; "
                "from scripts.paper.run_benchmark_backbone_v7_kshot_matrix_20260530 import run_center_pair, parse_args_for_worker; "
                "args=parse_args_for_worker(pathlib.Path(sys.argv[1])); "
                "spec=json.loads(sys.argv[2]); "
                "run_center_pair(model_name=spec['model_name'], source_ckpt=pathlib.Path(spec['source_ckpt']), "
                "center=spec['center'], k=spec['k'], protocol=spec['protocol'], seed=spec['seed'], gpu=spec['gpu'], args=args)"
            ),
            str(write_worker_args(args, run_id)),
            json.dumps(
                {
                    "model_name": model_name,
                    "source_ckpt": str(source_ckpt),
                    "center": center,
                    "k": k,
                    "protocol": protocol,
                    "seed": seed,
                    "gpu": gpu,
                }
            ),
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
    postprocess(model_name, protocol, seed, args)


def postprocess(model_name: str, protocol: str, seed: int, args: argparse.Namespace) -> None:
    slug = model_slug(model_name)
    run_id = run_id_for(model_name, protocol, seed, args)
    for method in ["direct", "vae_lhat"]:
        exp_name = f"{slug}_{method}_{protocol}_v7_sjr_rgq"
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
    ap.add_argument("--models", nargs="+", choices=available_model_names(), default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--protocols", nargs="+", choices=sorted(PROTOCOLS), default=["k500"])
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    ap.add_argument("--source-gpu", default="0")
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET_ROOT)
    ap.add_argument("--log-root", type=Path, default=DATA_ROOT / "runs/logs/benchmark_v7_multiseed_matrix_20260530")
    ap.add_argument("--tmp-root", type=Path, default=Path("/home/linbinhao/tmp_ecg"))
    ap.add_argument("--cache-root", type=Path, default=DATA_ROOT / "cache")
    ap.add_argument("--run-stamp", default="20260530")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=192)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--source-epochs", type=int, default=30)
    ap.add_argument("--source-batch-size", type=int, default=128)
    ap.add_argument("--source-lr", type=float, default=1e-3)
    ap.add_argument("--source-weight-decay", type=float, default=1e-2)
    ap.add_argument("--source-patience", type=int, default=8)
    ap.add_argument("--source-checkpoint-metric", choices=["auroc", "auprc"], default="auprc")
    ap.add_argument("--quick-eval-n-per-center", type=int, default=500)
    ap.add_argument("--eval-pn2021-limit", type=int, default=0)
    ap.add_argument("--disable-latent-augmix-branch", action="store_true")
    ap.add_argument("--force-source", action="store_true")
    ap.add_argument("--force-direct", action="store_true")
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
        for model_name in args.models:
            source_ckpt = train_source_model(model_name, seed, args)
            for protocol in args.protocols:
                run_model_protocol_seed(model_name, seed, protocol, source_ckpt, subset_items, args)


if __name__ == "__main__":
    main()
