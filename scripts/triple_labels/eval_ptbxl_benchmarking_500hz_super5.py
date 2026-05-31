#!/usr/bin/env python3
"""Evaluate 500Hz ecg_ptbxl_benchmarking Super5 checkpoints on PN2021."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data import load_kshot_ref_record_ids  # noqa: E402
from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021_CENTERS,
    PN2021_FORBIDDEN,
    _load_or_build_pn2021_center,
)
from scripts.triple_labels.label_schemes import get_scheme  # noqa: E402
from scripts.triple_labels.train_ptbxl_benchmarking_500hz_super5 import (  # noqa: E402
    CLASS_NAMES,
    compute_metrics,
    model_builder,
)


class PNArrayDataset(Dataset):
    def __init__(self, signals: np.ndarray, mean: float, std: float) -> None:
        self.signals = signals
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return int(len(self.signals))

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.asarray(self.signals[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.transpose(x, (1, 0)).copy())


@torch.no_grad()
def predict(model: torch.nn.Module, signals: np.ndarray, mean: float, std: float, args) -> np.ndarray:
    ds = PNArrayDataset(signals, mean, std)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and args.device.startswith("cuda")),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(ds, **loader_kwargs)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model.eval()
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            logits = model(x)
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def metric_view(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, *, drop_all_zero: bool) -> dict:
    if drop_all_zero:
        mask = mask & (labels.sum(axis=1) > 0)
    return compute_metrics(labels[mask], scores[mask])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--pn2021_root", default="/root/autodl-tmp/physionet2021")
    p.add_argument("--pn2021_cache_dir", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_benchmark_none")
    p.add_argument("--pn2021_mmap_cache_dir", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_benchmark_none_mmap")
    p.add_argument("--exclude_ref_root", default="/root/autodl-tmp/vae500_lhat_v7/anchors_k500")
    p.add_argument("--ref_k", type=int, default=500)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--load_center_in_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    p.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_path", default="")
    args = p.parse_args()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    model_dir = Path(args.model_dir)
    train_result = json.load(open(model_dir / "train_result.json"))
    model_name = train_result["model_name"]
    mean = float(train_result["normalization"]["train_mean"])
    std = float(train_result["normalization"]["train_std"])
    model = model_builder(model_name)()
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location="cpu"))
    model.to(args.device)
    if args.compile_model:
        model = torch.compile(model, mode=args.compile_mode)

    scheme = get_scheme("super5")
    cache_args = SimpleNamespace(
        scheme="super5",
        sampling_rate=500,
        input_len=5000,
        crop_len=5000,
        preprocess_mode="minimal_resample",
        norm_mode="none",
        pn2021_cache_dir=args.pn2021_cache_dir,
        pn2021_mmap_cache_dir=args.pn2021_mmap_cache_dir,
    )
    per_center = {}
    all_labels, all_scores, all_centers, all_record_ids = [], [], [], []
    t0 = time.time()
    for center in PN2021_CENTERS:
        if center.lower() in PN2021_FORBIDDEN:
            continue
        center_dir = Path(args.pn2021_root) / "training" / center
        signals, labels, record_ids, fail, load_time, from_cache, cache_kind = _load_or_build_pn2021_center(
            center,
            str(center_dir),
            scheme,
            cache_args,
        )
        if args.load_center_in_memory:
            signals = np.ascontiguousarray(np.asarray(signals, dtype=np.float32))
            labels = np.ascontiguousarray(np.asarray(labels, dtype=np.float32))
        scores = predict(model, signals, mean, std, args)
        mask = np.ones((len(labels),), dtype=bool)
        m = metric_view(labels, scores, mask, drop_all_zero=False)
        md = metric_view(labels, scores, mask, drop_all_zero=True)
        per_center[center] = {
            "n": int(len(labels)),
            "cache_kind": cache_kind,
            "macro_auroc": m["macro_auroc"],
            "macro_auprc": m["macro_auprc"],
            "drop_all_zero_macro_auroc": md["macro_auroc"],
            "drop_all_zero_macro_auprc": md["macro_auprc"],
        }
        all_labels.append(np.asarray(labels))
        all_scores.append(scores)
        all_centers.append(np.asarray([center] * len(labels), dtype=str))
        all_record_ids.append(np.asarray(record_ids).astype(str))
        print(f"{center:22s} {m['macro_auroc']:.4f}/{m['macro_auprc']:.4f} n={len(labels)} {cache_kind}", flush=True)

    centers = np.concatenate(all_centers)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    record_ids = np.concatenate(all_record_ids)
    target_views = {}
    for target in ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]:
        ref_ids = set(load_kshot_ref_record_ids(args.exclude_ref_root, target, k=args.ref_k, seed=args.ref_seed))
        view = {}
        aus, aps, daus, daps = [], [], [], []
        for center in PN2021_CENTERS:
            if center.lower() in PN2021_FORBIDDEN:
                continue
            mask = centers == center
            n_excl = 0
            if center == target:
                excl = np.asarray([rid in ref_ids for rid in record_ids], dtype=bool)
                n_excl = int((mask & excl).sum())
                mask = mask & ~excl
            m = metric_view(labels, scores, mask, drop_all_zero=False)
            md = metric_view(labels, scores, mask, drop_all_zero=True)
            view[center] = {
                "n": int(mask.sum()),
                "n_excluded_ref": n_excl,
                "macro_auroc": m["macro_auroc"],
                "macro_auprc": m["macro_auprc"],
                "drop_all_zero_macro_auroc": md["macro_auroc"],
                "drop_all_zero_macro_auprc": md["macro_auprc"],
            }
            aus.append(m["macro_auroc"])
            aps.append(m["macro_auprc"])
            daus.append(md["macro_auroc"])
            daps.append(md["macro_auprc"])
        target_views[target] = {
            "per_center": view,
            "avg_macro_auroc": float(np.nanmean(aus)),
            "avg_macro_auprc": float(np.nanmean(aps)),
            "drop_all_zero_avg_macro_auroc": float(np.nanmean(daus)),
            "drop_all_zero_avg_macro_auprc": float(np.nanmean(daps)),
        }

    out = {
        "model_name": model_name,
        "model_dir": str(model_dir),
        "class_names": CLASS_NAMES,
        "normalization": train_result["normalization"],
        "preprocess": {
            "sampling_rate": 500,
            "input_len": 5000,
            "preprocess_mode": "minimal_resample",
            "norm_mode_before_model_scaler": "none",
        },
        "per_center": per_center,
        "target_views": target_views,
        "runtime_sec": time.time() - t0,
    }
    output_path = Path(args.output_path) if args.output_path else model_dir / "pn2021_eval_v7_500hz_benchmark.json"
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved {output_path}")


if __name__ == "__main__":
    main()
