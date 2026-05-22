#!/usr/bin/env python3
"""ECGFounder frozen-encoder Super5 linear probe baseline.

This is a stronger foundation-model baseline than the zero-shot keyword-pool
mapping: reuse ECGFounder's official Net1D encoder and preprocessing, extract
1024-d features, train only a PTB-XL Super5 linear head, then evaluate PN2021
with the same K=500 ref-id exclusion views used by LH-AT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import wfdb


REPO_ROOT = Path(__file__).resolve().parents[2]
ECGFOUNDER_ROOT = Path("/root/autodl-tmp/ecgfounder")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ECGFOUNDER_ROOT))

from net1d import Net1D  # noqa: E402
from physionet2021_dataset import (  # noqa: E402
    EXPECTED_LEADS,
    TARGET_POINTS,
    filter_bandpass,
    resample_to_target,
    scan_records,
    z_score_normalize,
)
from scripts.paper.eval_ecgfounder_super5_zero_shot_20260517 import (  # noqa: E402
    PN2021_CENTERS,
    PN2021_FORBIDDEN,
    TARGET_CENTERS,
    center_from_record_path,
    compute_macro,
    load_ref_ids,
    record_id_from_record_path,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_super5_pn2021_mapping_metadata,
    get_scheme,
    snomed_list_to_super5,
)
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402


DEFAULT_OUT_DIR = Path("/root/autodl-tmp/paper_foundation_baselines_20260517/ecgfounder_linear_probe_super5")
PTBXL_ROOT = Path("/root/autodl-tmp/ptbxl")
PTBXL_CSV = PTBXL_ROOT / "ptbxl_database.csv"
PN2021_ROOT = Path("/root/autodl-tmp/physionet2021/training")
CHECKPOINT = ECGFOUNDER_ROOT / "checkpoint/12_lead_ECGFounder.pth"
REF_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_ecgfounder_feature_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def normalize_lead_name(name: str) -> str:
    mapping = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "avr": "aVR",
        "avl": "aVL",
        "avf": "aVF",
    }
    return mapping.get(str(name), str(name))


def preprocess_record(
    record_path: str,
    max_duration_sec: int = 10,
    preprocess_policy: str = "official_ptbxl_eval",
) -> np.ndarray:
    data, fields = wfdb.rdsamp(record_path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).T
    fs = int(fields.get("fs", 500))
    sig_names = [normalize_lead_name(x) for x in fields.get("sig_name", [])]
    if sig_names and sig_names != EXPECTED_LEADS:
        try:
            reorder_idx = [sig_names.index(lead) for lead in EXPECTED_LEADS]
            data = data[reorder_idx]
        except ValueError:
            pass
    max_samples = int(fs * max_duration_sec)
    if data.shape[1] > max_samples:
        data = data[:, :max_samples]
    if preprocess_policy == "filtered_dataset" and data.shape[1] > 10 and fs > 0:
        try:
            data = filter_bandpass(data, fs)
        except Exception:
            pass
    if data.shape[1] < max_samples:
        pad_width = max_samples - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
    data = resample_to_target(data, data.shape[1], TARGET_POINTS)
    data = z_score_normalize(data)
    return data.astype(np.float32, copy=False)


class RecordFeatureDataset(Dataset):
    def __init__(self, items: list[dict], preprocess_policy: str) -> None:
        self.items = items
        self.preprocess_policy = preprocess_policy

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        signal = preprocess_record(item["path"], preprocess_policy=self.preprocess_policy)
        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(np.asarray(item["label"], dtype=np.float32)).float(),
            item.get("center", ""),
            item.get("record_id", ""),
        )


def collate_records(batch):
    signals, labels, centers, record_ids = zip(*batch)
    return torch.stack(signals), torch.stack(labels), list(centers), list(record_ids)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    items: list[dict],
    out_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    preprocess_policy: str,
) -> dict[str, np.ndarray]:
    if out_path.exists():
        data = np.load(out_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        RecordFeatureDataset(items, preprocess_policy=preprocess_policy),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_records,
    )
    feats, labels, centers, record_ids = [], [], [], []
    for x, y, c, rid in tqdm(loader, desc=f"Extract {out_path.name}"):
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            _, f = model(x)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
        centers.extend(c)
        record_ids.extend(rid)
    payload = {
        "features": np.concatenate(feats, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.float32),
        "centers": np.asarray(centers, dtype=str),
        "record_ids": np.asarray(record_ids, dtype=str),
    }
    np.savez_compressed(out_path, **payload)
    return payload


def build_ptbxl_items(limit: int = 0) -> tuple[list[dict], np.ndarray]:
    df = pd.read_csv(PTBXL_CSV)
    scheme = get_scheme("super5")
    labels = np.stack([scheme["ptbxl_fn"](x) for x in df.scp_codes]).astype(np.float32)
    items = []
    for idx, row in df.iterrows():
        items.append(
            {
                "path": str(PTBXL_ROOT / row.filename_hr),
                "label": labels[idx],
                "center": "ptbxl",
                "record_id": str(row.ecg_id),
                "strat_fold": int(row.strat_fold),
            }
        )
    if limit > 0:
        items = items[:limit]
        labels = labels[:limit]
    folds = np.asarray([item["strat_fold"] for item in items], dtype=np.int64)
    return items, folds


def build_pn2021_items(manifest_cache: Path, limit_per_center: int = 0) -> list[dict]:
    if manifest_cache.exists():
        with manifest_cache.open() as f:
            records = json.load(f)
    else:
        records = scan_records(str(PN2021_ROOT))
        manifest_cache.parent.mkdir(parents=True, exist_ok=True)
        with manifest_cache.open("w") as f:
            json.dump(records, f)
    counts = {c: 0 for c in PN2021_CENTERS}
    items = []
    for rec in records:
        center = center_from_record_path(rec["path"])
        if center.lower() in PN2021_FORBIDDEN:
            continue
        if center not in counts:
            continue
        if limit_per_center and counts[center] >= limit_per_center:
            continue
        counts[center] += 1
        items.append(
            {
                "path": rec["path"],
                "label": snomed_list_to_super5(rec.get("snomed", [])),
                "center": center,
                "record_id": record_id_from_record_path(rec["path"]),
            }
        )
    return items


def compute_metrics(labels: np.ndarray, scores: np.ndarray, min_pos: int = 10) -> dict:
    row = compute_macro(labels, scores, min_pos=min_pos)
    return {
        "macro_auroc": row.macro_auroc,
        "macro_auprc": row.macro_auprc,
        "n_classes_used": row.n_classes_used,
        "per_class": row.per_class,
    }


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> tuple[nn.Module, dict]:
    train_mask = np.isin(folds, np.arange(1, 9))
    val_mask = folds == 9
    test_mask = folds == 10
    x_train = torch.from_numpy(features[train_mask]).float()
    y_train = torch.from_numpy(labels[train_mask]).float()
    x_val = torch.from_numpy(features[val_mask]).float()
    y_val = torch.from_numpy(labels[val_mask]).float()
    x_test = torch.from_numpy(features[test_mask]).float()
    y_test = torch.from_numpy(labels[test_mask]).float()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.head_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.head_batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.head_batch_size, shuffle=False)

    head = nn.Linear(features.shape[1], len(CLASS_NAMES_SUPER5)).to(device)
    pos_weight = torch.tensor(
        compute_pos_weight(labels[train_mask], len(CLASS_NAMES_SUPER5), clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits, y):
        return masked_bce_with_logits(logits, y, pos_weight)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    best = -float("inf")
    logs = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_metrics = eval_head(head, val_loader, device)
        score = (
            float(val_metrics["macro_auprc"])
            if val_metrics["macro_auprc"] is not None
            else -float("inf")
        )
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(entry)
        print(
            f"Head ep {epoch:03d}/{args.epochs} loss={entry['train_loss']:.4f} "
            f"val={entry['val_macro_auroc']}/{entry['val_macro_auprc']}",
            flush=True,
        )
        if score > best or not (out_dir / "best_head.pt").exists():
            best = score
            torch.save(head.state_dict(), out_dir / "best_head.pt")
        with (out_dir / "head_training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)

    head.load_state_dict(torch.load(out_dir / "best_head.pt", map_location=device))
    test_metrics = eval_head(head, test_loader, device)
    train_result = {
        "best_val_macro_auprc": best,
        "ptbxl_test": test_metrics,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "pos_weight": pos_weight.detach().cpu().tolist(),
    }
    with (out_dir / "train_result.json").open("w") as f:
        json.dump(train_result, f, indent=2)
    return head, train_result


@torch.no_grad()
def eval_head(head: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    head.eval()
    labels, scores = [], []
    for x, y in loader:
        logits = head(x.to(device)).cpu().numpy()
        labels.append(y.numpy())
        scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    return compute_metrics(np.concatenate(labels, axis=0), np.concatenate(scores, axis=0))


@torch.no_grad()
def predict_head(head: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    head.eval()
    loader = DataLoader(torch.from_numpy(features).float(), batch_size=batch_size, shuffle=False)
    scores = []
    for x in loader:
        logits = head(x.to(device)).cpu().numpy()
        scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    return np.concatenate(scores, axis=0)


def evaluate_pn2021_views(
    labels: np.ndarray,
    scores: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    ref_ids_by_center: dict[str, set[str]],
) -> dict:
    views = {}
    for target_center in TARGET_CENTERS:
        ref_ids = ref_ids_by_center[target_center]
        per_center = {}
        avg_aurocs, avg_auprcs = [], []
        for center in PN2021_CENTERS:
            if center.lower() in PN2021_FORBIDDEN:
                continue
            mask = centers == center
            n_total = int(mask.sum())
            n_excluded = 0
            if center == target_center:
                exclude = np.asarray([rid in ref_ids for rid in record_ids], dtype=bool)
                n_excluded = int((mask & exclude).sum())
                mask = mask & ~exclude
            if int(mask.sum()) == 0:
                continue
            metric = compute_metrics(labels[mask], scores[mask])
            per_center[center] = {
                "n_records": n_total,
                "n_excluded_ref": n_excluded,
                "effective_n": int(mask.sum()),
                **metric,
            }
            if metric["macro_auroc"] is not None:
                avg_aurocs.append(metric["macro_auroc"])
                avg_auprcs.append(metric["macro_auprc"])
        views[target_center] = {
            "target_center": target_center,
            "avg_macro_auroc": float(np.mean(avg_aurocs)),
            "avg_macro_auprc": float(np.mean(avg_auprcs)),
            "per_center": per_center,
        }
    return views


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--manifest_cache", default="/root/autodl-tmp/ecgfounder/physionet2021_manifest.json")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--head_batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit_ptbxl", type=int, default=0)
    p.add_argument("--limit_per_center", type=int, default=0)
    p.add_argument("--force_features", action="store_true")
    p.add_argument(
        "--preprocess_policy",
        choices=["official_ptbxl_eval", "filtered_dataset"],
        default="official_ptbxl_eval",
        help=(
            "official_ptbxl_eval matches ECGFounder's PTB-XL eval style: "
            "12x5000 + per-record z-score, no extra filter. filtered_dataset "
            "adds ECGFounder's util.py notch/bandpass/median baseline chain."
        ),
    )
    args = p.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ptbxl_feature_path = out_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz"
    pn_feature_path = out_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz"
    if args.force_features:
        for pth in [ptbxl_feature_path, pn_feature_path]:
            if pth.exists():
                pth.unlink()

    model = build_ecgfounder_feature_model(Path(args.checkpoint), device)
    ptbxl_items, folds = build_ptbxl_items(limit=args.limit_ptbxl)
    ptbxl_payload = extract_features(
        model,
        ptbxl_items,
        ptbxl_feature_path,
        args.batch_size,
        args.num_workers,
        device,
        args.preprocess_policy,
    )
    if "folds" not in ptbxl_payload:
        ptbxl_payload["folds"] = folds
        np.savez_compressed(ptbxl_feature_path, **ptbxl_payload)
    else:
        folds = ptbxl_payload["folds"].astype(np.int64)

    head, train_result = train_head(
        ptbxl_payload["features"],
        ptbxl_payload["labels"],
        folds,
        args,
        device,
        out_dir,
    )

    pn_items = build_pn2021_items(Path(args.manifest_cache), limit_per_center=args.limit_per_center)
    pn_payload = extract_features(
        model,
        pn_items,
        pn_feature_path,
        args.batch_size,
        args.num_workers,
        device,
        args.preprocess_policy,
    )
    pn_scores = predict_head(head, pn_payload["features"], args.head_batch_size, device)
    ref_ids_by_center = load_ref_ids(REF_ROOT, TARGET_CENTERS)
    views = evaluate_pn2021_views(
        pn_payload["labels"],
        pn_scores,
        pn_payload["centers"].astype(str),
        pn_payload["record_ids"].astype(str),
        ref_ids_by_center,
    )

    output = {
        "method": "ECGFounder frozen encoder + PTB-XL Super5 linear head",
        "class_names": list(CLASS_NAMES_SUPER5),
        "config": vars(args),
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "pn2021_mapping": get_super5_pn2021_mapping_metadata(),
        "preprocess": {
            "policy": args.preprocess_policy,
            "input_shape": [12, TARGET_POINTS],
            "duration_sec": 10,
            "ptbxl_source": "records500 / filename_hr",
            "lead_order": EXPECTED_LEADS,
            "normalization": "per-sample global z-score",
            "filtered_dataset": args.preprocess_policy == "filtered_dataset",
        },
        "train_result": train_result,
        "pn2021_views": views,
        "feature_paths": {
            "ptbxl": str(ptbxl_feature_path),
            "pn2021": str(pn_feature_path),
        },
    }
    out_json = out_dir / "ecgfounder_linear_probe_super5_ref_excluded.json"
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    csv_path = out_dir / "ecgfounder_linear_probe_super5_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_view", "target_auroc", "target_auprc", "pn2021_avg_auroc", "pn2021_avg_auprc"])
        for target in TARGET_CENTERS:
            view = views[target]
            target_row = view["per_center"][target]
            writer.writerow([
                target,
                target_row["macro_auroc"],
                target_row["macro_auprc"],
                view["avg_macro_auroc"],
                view["avg_macro_auprc"],
            ])
    print(f"[done] saved {out_json}")
    print(f"[done] saved {csv_path}")


if __name__ == "__main__":
    main()
