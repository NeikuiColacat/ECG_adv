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
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ECGFOUNDER_ROOT = Path("/root/autodl-tmp/ecgfounder")
for _p in [str(REPO_ROOT), str(ECGFOUNDER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from physionet2021_dataset import EXPECTED_LEADS, TARGET_POINTS  # noqa: E402
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 import real_anchor_base  # noqa: E402
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    CHECKPOINT,
    PTBXL_CSV,
    compute_metrics,
    preprocess_record,
    build_pn2021_items,
    build_ptbxl_items,
)
from scripts.pgd_cross_center.synth_online_at_super5 import SameLabelLatentIndex, StratifiedPoolWalker  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, SUPER5_TO_IDX  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


DEFAULT_OUT_DIR = Path("/root/autodl-tmp/paper_ecgfounder_fullft_super5_20260523")
CENTER_DEFAULT = "cpsc_2018"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tag_value(value: float | int | str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


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


class MemorySignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.signals[idx]), torch.from_numpy(self.labels[idx])


def global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)


def ecg1000_to_ecgfounder_input(ecg_ct_1000: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(ecg_ct_1000, size=TARGET_POINTS, mode="linear", align_corners=True)
    return global_zscore(x)


class ECGFounderFullFTVictim(nn.Module):
    """Differentiable ECGTwin-latent -> ECGFounder full model logits wrapper."""

    def __init__(self, model: nn.Module, ecgtwin: ECGTwinWrapper) -> None:
        super().__init__()
        self.model = model
        self.ecgtwin = ecgtwin
        self.num_classes = len(CLASS_NAMES_SUPER5)

    @staticmethod
    def _global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return global_zscore(x, eps=eps)

    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent / 0.18215
        for module in self.ecgtwin.decoder:
            x = module(x)
        return x.transpose(1, 2)

    def _ecgtwin_latent_to_ecg1000(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_tc = self._decode_latent_differentiable(latent)
        ecg_ct = ecg_tc.transpose(-1, -2)
        ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
        ecg_ct = torch.clamp(ecg_ct, min=-3.0, max=3.0)
        ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True)
        return self._global_zscore(ecg_ct)

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_ct = self._ecgtwin_latent_to_ecg1000(latent)
        return self.model(ecg1000_to_ecgfounder_input(ecg_ct))


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


def split_target_train_val(
    target_idx: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    val_count: int,
    split_seed: int,
    split_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    target_idx = np.asarray(target_idx, dtype=np.int64)
    if val_count <= 0:
        train_ids = {str(record_ids[i]) for i in target_idx}
        return target_idx, np.empty(0, dtype=np.int64), train_ids, set()
    if val_count >= len(target_idx):
        raise ValueError(f"target_val_count={val_count} must be smaller than target K={len(target_idx)}")
    rng = np.random.default_rng(split_seed + 1701)
    perm = np.asarray(target_idx, dtype=np.int64).copy()
    rng.shuffle(perm)
    if split_mode == "random":
        val_idx = np.sort(perm[:val_count])
    elif split_mode == "stratified":
        y = np.asarray(labels, dtype=np.float32)[target_idx] > 0.5
        pos_counts = y.sum(axis=0)
        selected: list[int] = []
        selected_set: set[int] = set()
        class_order = [
            int(c)
            for c in np.argsort(pos_counts)
            if int(pos_counts[int(c)]) >= 2
        ]
        for class_i in class_order:
            candidates = [int(i) for i, row in zip(target_idx, y) if row[class_i] and int(i) not in selected_set]
            if not candidates:
                continue
            candidate_scores = []
            for idx in candidates:
                row = np.asarray(labels[idx], dtype=np.float32) > 0.5
                covered_new = sum(
                    1
                    for c in class_order
                    if row[c] and not any(np.asarray(labels[j], dtype=np.float32)[c] > 0.5 for j in selected)
                )
                rarity = float(np.sum([1.0 / max(float(pos_counts[c]), 1.0) for c, present in enumerate(row) if present]))
                candidate_scores.append((-covered_new, -rarity, rng.random(), idx))
            candidate_scores.sort()
            pick = int(candidate_scores[0][3])
            selected.append(pick)
            selected_set.add(pick)
            if len(selected) >= val_count:
                break
        for idx in perm:
            idx = int(idx)
            if len(selected) >= val_count:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
        val_idx = np.sort(np.asarray(selected, dtype=np.int64))
    else:
        raise ValueError(f"unknown target_val_split_mode={split_mode}")
    train_idx = np.sort(perm[val_count:])
    if split_mode == "stratified":
        val_set = set(int(i) for i in val_idx)
        train_idx = np.sort(np.asarray([int(i) for i in target_idx if int(i) not in val_set], dtype=np.int64))
    train_ids = {str(record_ids[i]) for i in train_idx}
    val_ids = {str(record_ids[i]) for i in val_idx}
    return train_idx, val_idx, train_ids, val_ids


def load_anchor_pool_for_ids(
    center: str,
    selected_ids: set[str],
    pn_payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    base = real_anchor_base(center)
    with np.load(base.with_suffix(".latent.npz"), allow_pickle=True) as d:
        latents_all = d["latents"].astype(np.float32)
        record_ids_all = d["record_ids"].astype(str)

    want = {str(x) for x in selected_ids}
    rid_to_i = {str(rid): i for i, rid in enumerate(record_ids_all)}
    missing = sorted(want - set(rid_to_i))
    if missing:
        raise RuntimeError(
            f"{center}: {len(missing)} selected ids are missing from {base.with_suffix('.latent.npz')}; "
            f"first missing={missing[:5]}"
        )
    chosen = np.asarray([rid_to_i[rid] for rid in sorted(want)], dtype=np.int64)

    pn_record_ids = pn_payload["record_ids"].astype(str)
    pn_labels = pn_payload["labels"].astype(np.float32)
    label_by_rid = {str(rid): pn_labels[i] for i, rid in enumerate(pn_record_ids)}
    labels = []
    for rid in record_ids_all[chosen]:
        if str(rid) not in label_by_rid:
            raise RuntimeError(f"{center}: selected id {rid} missing from PN cache labels")
        labels.append(label_by_rid[str(rid)])
    labels_arr = np.stack(labels).astype(np.float32)
    present = labels_arr.sum(axis=0) > 0
    classes_in_scope = [c for c, ok in zip(CLASS_NAMES_SUPER5, present) if ok]
    return {
        "latents": latents_all[chosen],
        "labels": labels_arr,
        "record_ids": record_ids_all[chosen],
        "classes_in_scope": classes_in_scope,
        "label_counts": dict(zip(CLASS_NAMES_SUPER5, labels_arr.sum(axis=0).astype(int).tolist())),
        "source_base": str(base),
    }


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(flag)


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
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
) -> DataLoader:
    folds = ptbxl_payload["folds"].astype(np.int64)
    source_idx = np.nonzero(np.isin(folds, np.arange(1, 9)))[0]
    if args.source_train_limit > 0:
        source_idx = source_idx[: args.source_train_limit]
    source_ds = CachedSignalDataset(ptbxl_payload["signals"], ptbxl_payload["labels"], source_idx)
    target_ds = CachedSignalDataset(pn_payload["signals"], pn_payload["labels"], target_indices)
    datasets: list[Dataset] = []
    weights: list[float] = []
    if args.source_weight > 0:
        datasets.append(source_ds)
        weights.extend([args.source_weight] * len(source_ds))
    if args.target_real_weight > 0:
        datasets.append(target_ds)
        weights.extend([args.target_real_weight] * len(target_ds))
    if (
        adv_signals is not None
        and adv_labels is not None
        and len(adv_signals) > 0
        and args.adv_weight > 0
    ):
        adv_ds = MemorySignalDataset(adv_signals, adv_labels)
        datasets.append(adv_ds)
        weights.extend([args.adv_weight] * len(adv_ds))
    if not datasets:
        raise RuntimeError("no active training streams; check source/target/adv weights")
    combined = ConcatDataset(datasets)
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


def build_adv_epoch(
    model: nn.Module,
    victim: ECGFounderFullFTVictim,
    pgd_gen: LatentHullPGDGenerator,
    anchor_pool: dict[str, Any],
    walker: StratifiedPoolWalker,
    index: SameLabelLatentIndex,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_requires_grad(model, False)
    model.eval()
    victim.eval()
    adv_signals: list[np.ndarray] = []
    adv_labels: list[np.ndarray] = []
    delta_norms: list[float] = []
    try:
        classes = list(anchor_pool["classes_in_scope"])
        per_cls = max(1, args.k_anchor // max(len(classes), 1))
        k_per_cls = {c: per_cls for c in classes}
        for i in range(args.k_anchor - per_cls * len(classes)):
            k_per_cls[classes[i % len(classes)]] += 1
        drawn = walker.sample(k_per_cls)
        picks = np.concatenate([drawn[c] for c in classes if drawn[c].size > 0])
        if picks.size > 1:
            np.random.default_rng(args.seed + int(args.current_epoch)).shuffle(picks)
        for start in range(0, len(picks), args.pgd_batch):
            batch_idx = picks[start:start + args.pgd_batch]
            z = torch.from_numpy(anchor_pool["latents"][batch_idx]).float().to(device)
            y = torch.from_numpy(anchor_pool["labels"][batch_idx]).float().to(device)
            cand = torch.from_numpy(index.candidates_for(batch_idx, args.hull_m)).float().to(device)
            x_adv_1000, delta = pgd_gen.attack_from_latent(z, y, candidate_latents=cand)
            x_adv_5000 = ecg1000_to_ecgfounder_input(x_adv_1000).detach().cpu().numpy().astype(np.float32)
            adv_signals.append(x_adv_5000)
            adv_labels.append(anchor_pool["labels"][batch_idx].astype(np.float32))
            delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())
    finally:
        set_requires_grad(model, True)
    signals = (
        np.concatenate(adv_signals, axis=0).astype(np.float32)
        if adv_signals
        else np.empty((0, 12, TARGET_POINTS), dtype=np.float32)
    )
    labels = (
        np.concatenate(adv_labels, axis=0).astype(np.float32)
        if adv_labels
        else np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    )
    return {
        "signals": signals,
        "labels": labels,
        "n_adv": int(len(signals)),
        "delta_mean": float(np.mean(delta_norms)) if delta_norms else None,
        "delta_max": float(np.max(delta_norms)) if delta_norms else None,
    }


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
    ap.add_argument("--enable_vae_adv_stream", action="store_true")
    ap.add_argument("--adv_weight", type=float, default=20.0)
    ap.add_argument("--k_anchor", type=int, default=100)
    ap.add_argument("--vae_classes_in_scope", nargs="*", default=None)
    ap.add_argument("--vae_min_class_count", type=int, default=1)
    ap.add_argument("--hull_m", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_steps", type=int, default=3)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--hull_weight_mode", choices=["optimized", "one_hot", "uniform", "dirichlet"], default="optimized")
    ap.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--hull_label_mode", choices=["primary", "exact", "compatible"], default="primary")
    ap.add_argument("--hull_include_anchor", action="store_true")
    ap.add_argument("--pgd_eps", type=float, default=2.0)
    ap.add_argument("--pgd_batch", type=int, default=4)
    ap.add_argument("--target_val_count", type=int, default=0)
    ap.add_argument("--target_val_seed", type=int, default=None)
    ap.add_argument("--target_val_split_mode", choices=["random", "stratified"], default="random")
    ap.add_argument(
        "--selection_metric",
        choices=["source_auprc", "target_val_auprc", "source_plus_target_val_auprc"],
        default="source_auprc",
    )
    ap.add_argument("--target_val_score_weight", type=float, default=0.5)
    ap.add_argument("--source_train_limit", type=int, default=0)
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument("--run_suffix", default="")
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
    method_tag = "fullft_vae" if args.enable_vae_adv_stream else "fullft"
    if args.enable_vae_adv_stream and args.vae_classes_in_scope is not None:
        method_tag += "_cls" + "-".join(str(c) for c in args.vae_classes_in_scope)
    if args.enable_vae_adv_stream and args.vae_min_class_count > 1:
        method_tag += f"_mincnt{args.vae_min_class_count}"
    if args.enable_vae_adv_stream:
        method_tag += (
            f"_aw{tag_value(args.adv_weight)}"
            f"_ka{args.k_anchor}"
            f"_M{args.hull_m}"
            f"_lam{tag_value(args.hull_lambda)}"
            f"_hs{args.hull_steps}"
            f"_hlr{tag_value(args.hull_lr)}"
        )
        if args.hull_weight_mode != "optimized":
            method_tag += f"_{args.hull_weight_mode}"
        if args.hull_label_mode != "primary":
            method_tag += f"_label{args.hull_label_mode}"
        if args.hull_include_anchor:
            method_tag += "_includeanchor"
    if args.run_suffix:
        method_tag += f"_{args.run_suffix}"
    selection_tag = ""
    if args.target_val_count > 0 or args.selection_metric != "source_auprc":
        selection_tag = f"_tv{args.target_val_count}_{args.selection_metric}"
        if args.target_val_split_mode != "random":
            selection_tag += f"_{args.target_val_split_mode}"
        if args.target_val_seed is not None:
            selection_tag += f"_tvseed{args.target_val_seed}"
    run_dir = out_dir / "runs" / (
        f"{args.center}_K{args.k}_fullft_ep{args.epochs}_lr{lr_tag}_"
        f"sw{sw_tag}_tw{tw_tag}_{method_tag}{selection_tag}_seed{args.seed}"
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
    target_train_idx, target_val_idx, target_train_ids, target_val_ids = split_target_train_val(
        target_idx,
        record_ids,
        pn["labels"],
        args.target_val_count,
        args.seed if args.target_val_seed is None else args.target_val_seed,
        args.target_val_split_mode,
    )
    if args.selection_metric != "source_auprc" and len(target_val_idx) == 0:
        raise RuntimeError("--selection_metric needs --target_val_count > 0 unless source_auprc is used")
    eval_idx = np.asarray([i for i, rid in enumerate(record_ids) if rid not in selected_ids], dtype=np.int64)
    drop_eval_idx = eval_idx[pn["labels"][eval_idx].sum(axis=1) > 0]

    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    model.train()
    anchor_pool = None
    pgd_gen = None
    walker = None
    index = None
    if args.enable_vae_adv_stream:
        print("[setup] loading ECGTwin VAE decoder for full-FT VAE stream", flush=True)
        ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
        for param in ecgtwin.decoder.parameters():
            param.requires_grad_(False)
        victim = ECGFounderFullFTVictim(model=model, ecgtwin=ecgtwin).to(device)
        pgd_gen = LatentHullPGDGenerator(
            ecgtwin_wrapper=ecgtwin,
            victim=victim,
            epsilon=args.pgd_eps,
            hull_lambda=args.hull_lambda,
            hull_steps=args.hull_steps,
            hull_lr=args.hull_lr,
            weight_mode=args.hull_weight_mode,
            dirichlet_alpha=args.hull_dirichlet_alpha,
            device=args.device,
        )
        anchor_pool = load_anchor_pool_for_ids(args.center, target_train_ids, pn)
        if args.vae_classes_in_scope is not None:
            requested = {str(c) for c in args.vae_classes_in_scope}
            unknown = sorted(requested - set(CLASS_NAMES_SUPER5))
            if unknown:
                raise RuntimeError(f"unknown --vae_classes_in_scope classes: {unknown}")
            anchor_pool["classes_in_scope"] = [
                c for c in anchor_pool["classes_in_scope"] if c in requested
            ]
        if args.vae_min_class_count > 1:
            anchor_pool["classes_in_scope"] = [
                c for c in anchor_pool["classes_in_scope"]
                if int(anchor_pool["label_counts"].get(c, 0)) >= int(args.vae_min_class_count)
            ]
        if not anchor_pool["classes_in_scope"]:
            raise RuntimeError("VAE stream enabled but no classes remain after class filtering")
        walker = StratifiedPoolWalker(
            labels_one_hot=anchor_pool["labels"],
            classes_in_scope=anchor_pool["classes_in_scope"],
            class_to_idx=SUPER5_TO_IDX,
            seed=args.seed,
        )
        index = SameLabelLatentIndex(
            anchor_pool["latents"],
            anchor_pool["labels"],
            label_mode=args.hull_label_mode,
            seed=args.seed,
            include_self=args.hull_include_anchor,
        )
        set_requires_grad(model, True)
    pos_weight = torch.tensor(
        compute_pos_weight(ptbxl["labels"][np.isin(ptbxl["folds"], np.arange(1, 9))], 5, clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return masked_bce_with_logits(logits, y, pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    folds = ptbxl["folds"].astype(np.int64)
    val_idx = np.nonzero(folds == 9)[0]
    test_idx = np.nonzero(folds == 10)[0]
    logs = []
    best = -float("inf")
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        adv_info = {
            "signals": None,
            "labels": None,
            "n_adv": 0,
            "delta_mean": None,
            "delta_max": None,
        }
        if args.enable_vae_adv_stream and args.adv_weight > 0 and args.k_anchor > 0:
            assert anchor_pool is not None and pgd_gen is not None and walker is not None and index is not None
            args.current_epoch = epoch
            adv_info = build_adv_epoch(model, pgd_gen.victim, pgd_gen, anchor_pool, walker, index, args, device)
        train_loader = make_train_loader(
            ptbxl,
            pn,
            target_train_idx,
            args,
            adv_info["signals"],
            adv_info["labels"],
        )
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
        target_val_metrics = (
            eval_split(model, pn["signals"], pn["labels"], target_val_idx, args.eval_batch_size, device)
            if len(target_val_idx) > 0
            else None
        )
        target_metrics = eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device)
        drop_metrics = eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device)
        if args.selection_metric == "source_auprc":
            selection_score = float(val_metrics["macro_auprc"])
        elif args.selection_metric == "target_val_auprc":
            assert target_val_metrics is not None
            selection_score = float(target_val_metrics["macro_auprc"])
        else:
            assert target_val_metrics is not None
            w = float(args.target_val_score_weight)
            selection_score = (
                (1.0 - w) * float(val_metrics["macro_auprc"])
                + w * float(target_val_metrics["macro_auprc"])
            )
        entry = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "target_val_macro_auroc": None if target_val_metrics is None else target_val_metrics["macro_auroc"],
            "target_val_macro_auprc": None if target_val_metrics is None else target_val_metrics["macro_auprc"],
            "target_macro_auroc": target_metrics["macro_auroc"],
            "target_macro_auprc": target_metrics["macro_auprc"],
            "target_drop_all_zero_macro_auroc": drop_metrics["macro_auroc"],
            "target_drop_all_zero_macro_auprc": drop_metrics["macro_auprc"],
            "selection_score": selection_score,
            "n_adv": int(adv_info["n_adv"]),
            "adv_delta_mean": adv_info["delta_mean"],
            "adv_delta_max": adv_info["delta_max"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)
        print(
            f"ep={epoch:03d} loss={entry['loss']:.4f} "
            f"val={entry['val_macro_auroc']:.4f}/{entry['val_macro_auprc']:.4f} "
            + (
                f"tval={entry['target_val_macro_auroc']:.4f}/{entry['target_val_macro_auprc']:.4f} "
                if target_val_metrics is not None else ""
            )
            + f"target={entry['target_macro_auroc']:.4f}/{entry['target_macro_auprc']:.4f} "
            f"drop={entry['target_drop_all_zero_macro_auroc']:.4f}/{entry['target_drop_all_zero_macro_auprc']:.4f}",
            flush=True,
        )
        score = selection_score
        if score > best:
            best = score
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    result = {
        "method": (
            "ECGFounder official-style full fine-tuning + VAE-only real-anchor latent-hull online AT"
            if args.enable_vae_adv_stream
            else "ECGFounder official-style full fine-tuning, no VAE"
        ),
        "vae_stream_enabled": bool(args.enable_vae_adv_stream),
        "center": args.center,
        "K": int(len(target_idx)),
        "target_train_K": int(len(target_train_idx)),
        "target_val_K": int(len(target_val_idx)),
        "selected_ref_record_ids": sorted(selected_ids),
        "target_train_record_ids": sorted(target_train_ids),
        "target_val_record_ids": sorted(target_val_ids),
        "selection_metric": args.selection_metric,
        "target_val_split_mode": args.target_val_split_mode,
        "selection_score_weight": float(args.target_val_score_weight),
        "best_epoch": best_epoch,
        "best_selection_score": float(best),
        "ptbxl_fold10": eval_split(model, ptbxl["signals"], ptbxl["labels"], test_idx, args.eval_batch_size, device),
        "target_val": (
            eval_split(model, pn["signals"], pn["labels"], target_val_idx, args.eval_batch_size, device)
            if len(target_val_idx) > 0
            else None
        ),
        "target_excluding_ref": eval_split(model, pn["signals"], pn["labels"], eval_idx, args.eval_batch_size, device),
        "target_drop_all_zero_excluding_ref": eval_split(model, pn["signals"], pn["labels"], drop_eval_idx, args.eval_batch_size, device),
        "n_target_eval": int(len(eval_idx)),
        "n_target_drop_all_zero_eval": int(len(drop_eval_idx)),
        "config": vars(args),
        "vae_anchor_pool": None if anchor_pool is None else {
            "classes_in_scope": anchor_pool["classes_in_scope"],
            "label_counts": anchor_pool["label_counts"],
            "requested_classes_in_scope": args.vae_classes_in_scope,
            "min_class_count": int(args.vae_min_class_count),
            "source_base": anchor_pool["source_base"],
        },
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
