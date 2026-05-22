#!/usr/bin/env python3
"""Adaptive BatchNorm K=500 target-center baseline for Super5 PN2021.

Protocol:
  * initialize from the PTB-XL Super5 EfficientNet1DV2 checkpoint;
  * use the same K=500 target-center records as LH-AT, labels ignored;
  * reset BN running statistics and estimate them from the K=500 target records;
  * evaluate PN2021 with those K=500 ref ids excluded from the target center.

This answers whether target-center gains can be explained by only adapting
BatchNorm statistics, without changing classifier weights or using labels.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc  # noqa: E402
from scripts.paper.run_latenthull_real_anchor_grid_20260512 import (  # noqa: E402
    BASELINE_CKPT,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PROJECT_ROOT,
    PTBXL_PREP,
    PYTHON,
)
from scripts.triple_labels.label_schemes import NUM_SUPER5  # noqa: E402


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_tta_baselines_20260517/adabn_k500")
SUBSET_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")


class TargetNPZDataset(Dataset):
    def __init__(self, signals: np.ndarray, crop_len: int) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.crop_len = int(crop_len)

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        crop = crop_signal_tc(self.signals[idx], self.crop_len, mode="center")
        return torch.from_numpy(np.ascontiguousarray(crop.T)).float()


def subset_paths(center: str, k: int, seed: int) -> dict[str, Path]:
    base_dir = SUBSET_ROOT / center / f"k{k}_seed{seed}"
    base = base_dir / f"{center}_real_k{k}_seed{seed}"
    return {
        "signals": base.with_suffix(".signals.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
    }


def load_model(device: torch.device) -> nn.Module:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=NUM_SUPER5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    state = torch.load(BASELINE_CKPT, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def configure_adabn(model: nn.Module) -> int:
    """Reset BN running stats and make cumulative averages over target batches."""
    model.train()
    model.requires_grad_(False)
    n_bn = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.track_running_stats = True
            module.momentum = None
            n_bn += 1
    if n_bn == 0:
        raise RuntimeError("AdaBN requires BatchNorm1d layers, found none")
    return n_bn


@torch.no_grad()
def estimate_target_bn(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.train()
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            _ = model(x)


def run_cmd(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
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


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    out_dir = Path(args.out_root) / "runs" / (
        f"{center}_K{args.k}_adabn_seed{args.seed}_subset{args.subset_seed}"
    )
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} AdaBN already evaluated")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    with np.load(paths["signals"], allow_pickle=True) as data:
        target_signals = data["signals"].astype(np.float32, copy=False)
    ds = TargetNPZDataset(target_signals, crop_len=args.crop_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = load_model(device)
    n_bn = configure_adabn(model)
    print(f"[data] center={center} target_unlabeled={len(ds)} BN_layers={n_bn}", flush=True)
    if not args.dry_run:
        estimate_target_bn(model, loader, device)
        torch.save(model.state_dict(), out_dir / "best_model.pt")
        with (out_dir / "train_result.json").open("w") as f:
            json.dump(
                {
                    "method": "adabn",
                    "center": center,
                    "K": int(args.k),
                    "n_target_unlabeled": int(len(ds)),
                    "n_bn_layers": int(n_bn),
                    "init_checkpoint": str(BASELINE_CKPT),
                    "subset_signals": str(paths["signals"]),
                    "ref_meta": str(paths["meta"]),
                    "protocol": "BN running stats estimated from target K records; labels ignored",
                },
                f,
                indent=2,
            )
        with (out_dir / "run_config.json").open("w") as f:
            json.dump({**vars(args), "center": center, "method": "adabn"}, f, indent=2)

    eval_cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(out_dir),
        "--device",
        args.device,
        "--crop_len",
        str(args.crop_len),
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
        str(paths["meta"]),
        "--output_path",
        str(eval_path),
    ]
    run_cmd(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def parse_eval(center: str, args: argparse.Namespace, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    return {
        "method": "adabn",
        "center": center,
        "K": int(args.k),
        "target_auroc": float(data["pn2021"]["per_center"][center]["macro_auroc"]),
        "target_auprc": float(data["pn2021"]["per_center"][center]["macro_auprc"]),
        "pn2021_avg_auroc": float(data["pn2021"]["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(data["pn2021"]["avg_macro_auprc"]),
        "ptbxl_auroc": float(data["ptbxl_test"]["macro_auroc"]),
        "ptbxl_auprc": float(data["ptbxl_test"]["macro_auprc"]),
        "eval_path": str(eval_path),
    }


def write_summary(rows: list[dict], out_root: Path) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "adabn_k500.csv"
    fields = [
        "method",
        "center",
        "K",
        "target_auroc",
        "target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["center"]):
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[summary] wrote {csv_path}", flush=True)


def load_existing_rows(out_root: Path) -> list[dict]:
    rows = []
    for eval_path in sorted((out_root / "runs").glob("*/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json")):
        cfg_path = eval_path.parent / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            rows.append(parse_eval(str(cfg["center"]), argparse.Namespace(**cfg), eval_path))
        except Exception as exc:
            print(f"[warn] failed to parse {eval_path}: {exc}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--subset_seed", type=int, default=20260531)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_root)
    for center in args.centers:
        eval_path = train_one(center, args)
        if not args.dry_run:
            rows = [r for r in rows if r["center"] != center]
            rows.append(parse_eval(center, args, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
