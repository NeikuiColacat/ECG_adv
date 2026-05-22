#!/usr/bin/env python3
"""ECGFounder official-style full fine-tuning Super5 pilot.

This script tests whether ECGFounder's target-center gains come from the
previous frozen-encoder/head-only protocol being too weak. It follows the local
ECGFounder notebook's recommended mode (`linear_prob=False`): initialize the
12-lead checkpoint, replace the dense layer with a Super5 head, and fine-tune
the whole model on PTB-XL source plus K target-center real ECGs.

No ECGTwin VAE adversarial samples are used here. This is the fairness control
that must be understood before attributing ECGFounder gains to VAE-only LH-AT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ECGFOUNDER_ROOT = Path("/root/autodl-tmp/ecgfounder")
for _p in [str(REPO_ROOT), str(ECGFOUNDER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from physionet2021_dataset import EXPECTED_LEADS, TARGET_POINTS  # noqa: E402
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    CHECKPOINT,
    PTBXL_CSV,
    compute_metrics,
    preprocess_record,
    build_pn2021_items,
    build_ptbxl_items,
)
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402


DEFAULT_OUT_DIR = Path("/root/autodl-tmp/paper_ecgfounder_fullft_super5_20260523")
CENTER_DEFAULT = "cpsc_2018"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedSignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_i = int(self.indices[idx])
        x = np.asarray(self.signals[src_i], dtype=np.float32).copy()
        y = np.asarray(self.labels[src_i], dtype=np.float32).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def build_signal_cache(
    items: list[dict[str, Any]],
    signal_path: Path,
    meta_path: Path,
    preprocess_policy: str,
) -> dict[str, np.ndarray]:
    if signal_path.exists() and meta_path.exists():
        return {
            "signals": np.load(signal_path, mmap_mode="r"),
            **{k: v for k, v in np.load(meta_path, allow_pickle=True).items()},
        }

    signal_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        signal_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(items), 12, TARGET_POINTS),
    )
    labels = np.empty((len(items), 5), dtype=np.float32)
    centers, record_ids, folds = [], [], []
    for i, item in enumerate(tqdm(items, desc=f"cache {signal_path.name}")):
        arr[i] = preprocess_record(item["path"], preprocess_policy=preprocess_policy)
        labels[i] = np.asarray(item["label"], dtype=np.float32)
        centers.append(str(item.get("center", "")))
        record_ids.append(str(item.get("record_id", "")))
        folds.append(int(item.get("strat_fold", -1)))
    arr.flush()
    np.savez_compressed(
        meta_path,
        labels=labels,
        centers=np.asarray(centers, dtype=str),
        record_ids=np.asarray(record_ids, dtype=str),
        folds=np.asarray(folds, dtype=np.int64),
        preprocess_policy=np.asarray([preprocess_policy], dtype=str),
        input_shape=np.asarray([12, TARGET_POINTS], dtype=np.int64),
        lead_order=np.asarray(EXPECTED_LEADS, dtype=str),
    )
    return {
        "signals": np.load(signal_path, mmap_mode="r"),
        **{k: v for k, v in np.load(meta_path, allow_pickle=True).items()},
    }


def load_selected_ref_ids(ref_meta_json: Path, center: str) -> set[str]:
    with ref_meta_json.open() as f:
        meta = json.load(f)
    out = set()
    if isinstance(meta, list):
        for row in meta:
            if str(row.get("center", center)) == center:
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    out.add(str(rid))
    elif isinstance(meta, dict):
        for key in ("record_ids", "ref_record_ids", "selected_ref_record_ids"):
            if key in meta:
                out.update(str(x) for x in meta[key])
        if not out and "items" in meta:
            for row in meta["items"]:
                rid = row.get("record_id") or row.get("record")
                if rid is not None:
                    out.add(str(rid))
    if not out:
        raise RuntimeError(f"could not parse selected ref ids from {ref_meta_json}")
    return out


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    labels, scores = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        labels.append(y.numpy())
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        scores.append(probs)
    return np.concatenate(labels, axis=0), np.concatenate(scores, axis=0)


def eval_split(model: nn.Module, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device) -> dict:
    y, p = predict(model, CachedSignalDataset(signals, labels, indices), batch_size, device)
    return compute_metrics(y.astype(np.float32), p.astype(np.float32), min_pos=1)


def make_train_loader(
    ptbxl_payload: dict[str, np.ndarray],
    pn_payload: dict[str, np.ndarray],
    target_indices: np.ndarray,
    args: argparse.Namespace,
) -> DataLoader:
    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    if args.source_train_limit > 0:
        source_idx = source_idx[: args.source_train_limit]
    source_ds = CachedSignalDataset(ptbxl_payload["signals"], ptbxl_payload["labels"], source_idx)
    target_ds = CachedSignalDataset(pn_payload["signals"], pn_payload["labels"], target_indices)
    combined = ConcatDataset([source_ds, target_ds])
    weights = [args.source_weight] * len(source_ds) + [args.target_real_weight] * len(target_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--center", default=CENTER_DEFAULT)
    ap.add_argument("--ref_meta_json", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--source_weight", type=float, default=1.0)
    ap.add_argument("--target_real_weight", type=float, default=40.0)
    ap.add_argument("--source_train_limit", type=int, default=0)
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    lr_tag = str(args.lr).replace(".", "p")
    sw_tag = str(args.source_weight).replace(".", "p")
    tw_tag = str(args.target_real_weight).replace(".", "p")
    run_dir = out_dir / "runs" / (
        f"{args.center}_K{args.k}_fullft_ep{args.epochs}_lr{lr_tag}_"
        f"sw{sw_tag}_tw{tw_tag}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text())
        return

    cache_dir = out_dir / "cache"
    ptbxl_items, _ = build_ptbxl_items(limit=0)
    pn_items_all = build_pn2021_items(out_dir / "pn2021_manifest.json", limit_per_center=0)
    pn_items = [x for x in pn_items_all if str(x["center"]) == args.center]
    ptbxl = build_signal_cache(
        ptbxl_items,
        cache_dir / f"ptbxl_{args.preprocess_policy}.signals.npy",
        cache_dir / f"ptbxl_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )
    pn = build_signal_cache(
        pn_items,
        cache_dir / f"{args.center}_{args.preprocess_policy}.signals.npy",
        cache_dir / f"{args.center}_{args.preprocess_policy}.meta.npz",
        args.preprocess_policy,
    )

    selected_ids = load_selected_ref_ids(Path(args.ref_meta_json), args.center)
    record_ids = pn["record_ids"].astype(str)
    target_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid in selected_ids], dtype=np.int64)
    if len(target_idx) != args.k:
        print(f"[warn] parsed K={len(target_idx)} target records; requested {args.k}", flush=True)
    eval_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid not in selected_ids], dtype=np.int64)
    drop_eval_idx = eval_idx[pn["labels"][eval_idx].sum(axis=1) > 0]

    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    model.train()
    pos_weight = torch.tensor(
        compute_pos_weight(ptbxl["labels"][np.isin(ptbxl["folds"], np.arange(1, 9))], 5, clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return masked_bce_with_logits(logits, y, pos_weight)

    train_loader = make_train_loader(ptbxl, pn, target_idx, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    folds = ptbxl["folds"].astype(np.int64)
    val_idx = np.nonzero(folds == 9)[0]
    test_idx = np.nonzero(folds == 10)[0]
    logs = []
    best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_metrics = eval_split(model, ptbxl["signals"], ptbxl["labels"], val_idx, args.eval_batch_size, device)
        target_metrics = eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device)
        drop_metrics = eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device)
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "target_macro_auroc": target_metrics["macro_auroc"],
            "target_macro_auprc": target_metrics["macro_auprc"],
            "target_drop_all_zero_macro_auroc": drop_metrics["macro_auroc"],
            "target_drop_all_zero_macro_auprc": drop_metrics["macro_auprc"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)
        print(
            f"ep={epoch:03d} loss={entry['loss']:.4f} "
            f"val={entry['val_macro_auroc']:.4f}/{entry['val_macro_auprc']:.4f} "
            f"target={entry['target_macro_auroc']:.4f}/{entry['target_macro_auprc']:.4f} "
            f"drop={entry['target_drop_all_zero_macro_auroc']:.4f}/{entry['target_drop_all_zero_macro_auprc']:.4f}",
            flush=True,
        )
        score = float(val_metrics["macro_auprc"])
        if score > best:
            best = score
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    result = {
        "method": "ECGFounder official-style full fine-tuning, no VAE",
        "center": args.center,
        "K": int(len(target_idx)),
        "selected_ref_record_ids": sorted(selected_ids),
        "ptbxl_fold10": eval_split(model, ptbxl["signals"], ptbxl["labels"], test_idx, args.eval_batch_size, device),
        "target_excluding_ref": eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device),
        "target_drop_all_zero_excluding_ref": eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device),
        "n_target_eval": int(len(eval_idx)),
        "n_target_drop_all_zero_eval": int(len(drop_eval_idx)),
        "config": vars(args),
        "official_evidence": {
            "finetune_model": "ft_12lead_ECGFounder(..., linear_prob=False)",
            "notebook_hyperparams": "lr=1e-4, weight_decay=1e-5, Epochs=5",
            "preprocess": "ECGFounder 12 x 5000, official_ptbxl_eval preprocessing",
        },
    }
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
