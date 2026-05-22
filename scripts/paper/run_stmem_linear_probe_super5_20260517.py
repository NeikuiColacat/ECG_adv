#!/usr/bin/env python3
"""ST-MEM frozen-encoder Super5 linear probe baseline.

This follows the official ST-MEM downstream input protocol as closely as is
practical inside this repo: 250 Hz, 9 s crops (2250 samples), 3 evaluation crops
per record, 0.67 Hz high-pass + 40 Hz low-pass filtering, and per-sample global
standardization.  We freeze the official ST-MEM ViT-B encoder, train only a
PTB-XL Super5 linear head, then evaluate PN2021 using the same K=500 target
ref-id exclusion views used by the latent-hull AT experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
STMEM_ROOT = Path("/root/autodl-tmp/external_repos/ST-MEM")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(STMEM_ROOT))

import models.encoder as stmem_encoder  # noqa: E402
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
    get_scheme,
    snomed_list_to_super5,
)
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402


DEFAULT_OUT_DIR = Path("/root/autodl-tmp/paper_foundation_baselines_20260517/stmem_linear_probe_super5")
STMEM_CKPT = Path("/root/autodl-tmp/st_mem/encoder.pth")
PTBXL_ROOT = Path("/root/autodl-tmp/ptbxl")
PTBXL_CSV = PTBXL_ROOT / "ptbxl_database.csv"
PN2021_ROOT = Path("/root/autodl-tmp/physionet2021/training")
REF_ROOT = Path("/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 250
RECORD_LEN = 2500
CROP_LEN = 2250
N_CROPS = 3


def normalize_lead_name(name: str) -> str:
    return {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "avr": "aVR", "avl": "aVL", "avf": "aVF"}.get(
        str(name),
        str(name),
    )


def reorder_leads(data_ct: np.ndarray, sig_names: list[str]) -> np.ndarray:
    names = [normalize_lead_name(x) for x in sig_names]
    if names and set(LEAD_ORDER).issubset(set(names)):
        idx = [names.index(lead) for lead in LEAD_ORDER]
        return data_ct[idx]
    if data_ct.shape[0] == len(LEAD_ORDER):
        return data_ct
    out = np.zeros((len(LEAD_ORDER), data_ct.shape[1]), dtype=np.float32)
    n_copy = min(out.shape[0], data_ct.shape[0])
    out[:n_copy] = data_ct[:n_copy]
    return out


def resample_to_250(data_ct: np.ndarray, fs: int) -> np.ndarray:
    if int(fs) == TARGET_FS:
        return data_ct.astype(np.float32, copy=False)
    g = gcd(int(TARGET_FS), int(fs))
    up, down = int(TARGET_FS // g), int(fs // g)
    return resample_poly(data_ct, up, down, axis=1).astype(np.float32, copy=False)


_HIGHPASS_SOS = butter(5, 0.67, btype="highpass", fs=TARGET_FS, output="sos")
_LOWPASS_SOS = butter(5, 40.0, btype="lowpass", fs=TARGET_FS, output="sos")


def n_crop(data_ct: np.ndarray) -> np.ndarray:
    if data_ct.shape[1] >= RECORD_LEN:
        start = max((data_ct.shape[1] - RECORD_LEN) // 2, 0)
        data_ct = data_ct[:, start : start + RECORD_LEN]
    else:
        pad = RECORD_LEN - data_ct.shape[1]
        left = pad // 2
        right = pad - left
        data_ct = np.pad(data_ct, ((0, 0), (left, right)), mode="constant")
    starts = np.linspace(0, RECORD_LEN - CROP_LEN, N_CROPS).round().astype(int)
    return np.stack([data_ct[:, s : s + CROP_LEN] for s in starts], axis=0)


def official_eval_preprocess(crops: np.ndarray) -> np.ndarray:
    """Apply ST-MEM eval filters and standardization to (crop, lead, time)."""
    try:
        crops = sosfiltfilt(_HIGHPASS_SOS, crops, axis=-1)
        crops = sosfiltfilt(_LOWPASS_SOS, crops, axis=-1)
    except Exception:
        crops = np.nan_to_num(crops, nan=0.0, posinf=0.0, neginf=0.0)
    loc = np.mean(crops, axis=(-1, -2), keepdims=True)
    scale = np.std(crops, axis=(-1, -2), keepdims=True)
    crops = np.divide(crops - loc, scale, out=np.zeros_like(crops), where=scale != 0)
    return np.nan_to_num(crops, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def three_crops_from_record(record_path: str) -> np.ndarray:
    data, fields = wfdb.rdsamp(record_path)
    data_ct = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).T
    data_ct = reorder_leads(data_ct, list(fields.get("sig_name", [])))
    fs = int(fields.get("fs", TARGET_FS))
    data_ct = resample_to_250(data_ct, fs)
    return official_eval_preprocess(n_crop(data_ct))


class RecordCropsDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return (
            torch.from_numpy(three_crops_from_record(item["path"])).float(),
            torch.from_numpy(np.asarray(item["label"], dtype=np.float32)).float(),
            item.get("center", ""),
            item.get("record_id", ""),
        )


def collate_crops(batch):
    crops, labels, centers, record_ids = zip(*batch)
    return torch.stack(crops), torch.stack(labels), list(centers), list(record_ids)


def build_stmem_model(device: torch.device) -> nn.Module:
    model = stmem_encoder.__dict__["st_mem_vit_base"](
        num_leads=12,
        num_classes=None,
        seq_len=CROP_LEN,
        patch_size=75,
    )
    checkpoint = torch.load(STMEM_CKPT, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    msg = model.load_state_dict(state, strict=False)
    print(f"[model] ST-MEM loaded: {msg}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(
    model: nn.Module,
    items: list[dict],
    out_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if out_path.exists():
        data = np.load(out_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        RecordCropsDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_crops,
        persistent_workers=num_workers > 0,
    )
    feats, labels, centers, record_ids = [], [], [], []
    for crops, y, c, rid in tqdm(loader, desc=f"Extract {out_path.name}"):
        b, nc, nlead, nt = crops.shape
        x = crops.view(b * nc, nlead, nt).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            emb = model.forward_encoding(x).float().view(b, nc, -1).mean(dim=1)
        feats.append(emb.cpu().numpy())
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


def scan_records(root: str) -> list[dict]:
    records = []
    for hea in sorted(Path(root).rglob("*.hea")):
        rec = str(hea.with_suffix(""))
        snomed = []
        with hea.open(errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line and ":" in line:
                    codes = [x.strip() for x in line.split(":", 1)[1].strip().split(",") if x.strip()]
                    try:
                        snomed = [int(x) for x in codes]
                    except ValueError:
                        snomed = []
                    break
        records.append({"path": rec, "snomed": snomed})
    return records


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
        if center.lower() in PN2021_FORBIDDEN or center not in counts:
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


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def predict_head(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)
    for (xb,) in loader:
        xb = xb.to(device)
        outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def fmt_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def train_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    train_mask = np.isin(folds, np.arange(1, 9))
    val_mask = folds == 9
    x_train, y_train = features[train_mask], labels[train_mask]
    x_val, y_val = features[val_mask], labels[val_mask]
    model = LinearHead(features.shape[1], labels.shape[1]).to(device)
    pos_weight = torch.tensor(
        compute_pos_weight(y_train, labels.shape[1], clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    best = -float("inf")
    log_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = masked_bce_with_logits(model(xb), yb, pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict_head(model, x_val, batch_size, device)
        val_metric = compute_macro(y_val, val_pred)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_auroc": val_metric.macro_auroc,
            "val_macro_auprc": val_metric.macro_auprc,
        }
        log_rows.append(row)
        print(
            f"Ep {epoch:03d}/{epochs} loss={row['train_loss']:.4f} "
            f"val={fmt_metric(row['val_macro_auroc'])}/{fmt_metric(row['val_macro_auprc'])}",
            flush=True,
        )
        if epoch == 1 or (row["val_macro_auprc"] is not None and row["val_macro_auprc"] > best):
            if row["val_macro_auprc"] is not None:
                best = row["val_macro_auprc"]
            torch.save(model.state_dict(), out_dir / "linear_head_best.pt")
        with (out_dir / "linear_train_log.json").open("w") as f:
            json.dump(log_rows, f, indent=2)
    model.load_state_dict(torch.load(out_dir / "linear_head_best.pt", map_location=device))
    return model


def evaluate_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_stmem_model(device)
    ptbxl_items, folds = build_ptbxl_items(args.ptbxl_limit)
    pn_items = build_pn2021_items(out_dir / "pn2021_manifest.json", args.pn2021_limit_per_center)
    print(f"[data] PTBXL={len(ptbxl_items)} PN2021={len(pn_items)}")
    ptb = extract_features(
        model,
        ptbxl_items,
        out_dir / "ptbxl_stmem_features.npz",
        args.extract_batch_size,
        args.num_workers,
        device,
    )
    pn = extract_features(
        model,
        pn_items,
        out_dir / "pn2021_stmem_features.npz",
        args.extract_batch_size,
        args.num_workers,
        device,
    )

    head = train_linear_head(
        ptb["features"],
        ptb["labels"],
        folds,
        out_dir,
        args.head_epochs,
        args.head_batch_size,
        args.head_lr,
        args.head_weight_decay,
        device,
    )
    ptb_pred = predict_head(head, ptb["features"], args.head_batch_size, device)
    test_mask = folds == 10
    ptb_metric = compute_macro(ptb["labels"][test_mask], ptb_pred[test_mask])

    pn_pred = predict_head(head, pn["features"], args.head_batch_size, device)
    ref_ids = load_ref_ids(REF_ROOT, TARGET_CENTERS)
    views = {}
    rows = []
    for target in TARGET_CENTERS:
        per_center = {}
        for center in PN2021_CENTERS:
            mask = pn["centers"] == center
            n_excl = 0
            if center == target:
                excl = ref_ids.get(center, set())
                keep = np.asarray([rid not in excl for rid in pn["record_ids"][mask]], dtype=bool)
                idx = np.where(mask)[0][keep]
                n_excl = int((~keep).sum())
            else:
                idx = np.where(mask)[0]
            metric = compute_macro(pn["labels"][idx], pn_pred[idx])
            per_center[center] = {
                "n_records": int(len(idx)),
                "n_excluded_ref": n_excl,
                "macro_auroc": metric.macro_auroc,
                "macro_auprc": metric.macro_auprc,
                "n_classes_used": metric.n_classes_used,
                "per_class": metric.per_class,
            }
        valid_aurocs = [per_center[c]["macro_auroc"] for c in PN2021_CENTERS if per_center[c]["macro_auroc"] is not None]
        valid_auprcs = [per_center[c]["macro_auprc"] for c in PN2021_CENTERS if per_center[c]["macro_auprc"] is not None]
        views[target] = {
            "target_center": target,
            "avg_macro_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "avg_macro_auprc": float(np.mean(valid_auprcs)) if valid_auprcs else None,
            "per_center": per_center,
        }
        rows.append(
            {
                "target_view": target,
                "target_auroc": per_center[target]["macro_auroc"],
                "target_auprc": per_center[target]["macro_auprc"],
                "pn2021_avg_auroc": views[target]["avg_macro_auroc"],
                "pn2021_avg_auprc": views[target]["avg_macro_auprc"],
            }
        )

    result = {
        "method": "STMEM_frozen_encoder_super5_linear_probe",
        "checkpoint": str(STMEM_CKPT),
        "input_protocol": "official-like ST-MEM eval: 250Hz, 2250-sample crops, 3 crops averaged, 0.67-40Hz filters, per-sample global standardization",
        "ptbxl_test": {
            "macro_auroc": ptb_metric.macro_auroc,
            "macro_auprc": ptb_metric.macro_auprc,
            "n_classes_used": ptb_metric.n_classes_used,
            "per_class": ptb_metric.per_class,
        },
        "views": views,
    }
    with (out_dir / "stmem_linear_probe_result.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (out_dir / "stmem_linear_probe_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {out_dir / 'stmem_linear_probe_summary.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--extract_batch_size", type=int, default=8, help="records per batch; each has three 9s crops")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--head_epochs", type=int, default=80)
    p.add_argument("--head_batch_size", type=int, default=512)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--head_weight_decay", type=float, default=1e-4)
    p.add_argument("--ptbxl_limit", type=int, default=0)
    p.add_argument("--pn2021_limit_per_center", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    parsed = parse_args()
    evaluate_all(parsed)
    print(f"[elapsed] {time.time() - t0:.1f}s")
