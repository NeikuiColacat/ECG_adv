#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.classifier_backend import build_efficientnet_super5  # noqa: E402
from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, crop_or_pad_ct, global_zscore_ct  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
    x = np.stack([global_zscore_ct(crop_or_pad_ct(sig, target_len=1000)) for sig in signals])
    return x.astype(np.float32), labels.astype(np.float32)


def load_synth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        key = "signals" if "signals" in data.files else data.files[0]
        signals = data[key].astype(np.float32)
        labels = data["labels"].astype(np.float32)
    x = np.stack([global_zscore_ct(crop_or_pad_ct(sig, target_len=1000)) for sig in signals])
    return x.astype(np.float32), labels.astype(np.float32)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if n >= 4 else max(1, n // 2)
    return idx[n_val:], idx[:n_val]


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        aurocs = []
        auprcs = []
        for i, cls in enumerate(CLASS_NAMES):
            if len(np.unique(y_true[:, i])) < 2:
                out[f"{cls}_auroc"] = None
                out[f"{cls}_auprc"] = None
                continue
            auroc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            auprc = float(average_precision_score(y_true[:, i], y_prob[:, i]))
            out[f"{cls}_auroc"] = auroc
            out[f"{cls}_auprc"] = auprc
            aurocs.append(auroc)
            auprcs.append(auprc)
        out["macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
        out["macro_auprc"] = float(np.mean(auprcs)) if auprcs else None
    except Exception as exc:
        out["metric_error"] = str(exc)
        out["macro_auroc"] = None
        out["macro_auprc"] = None
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    ys = []
    ps = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(float(loss.item()))
            ys.append(yb.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    return float(np.mean(losses)), metrics(y_true, y_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--synth_npz", default=None)
    ap.add_argument("--synth_ratio", type=float, default=1.0)
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    x_real, y_real = load_dataset(args.dataset_npz)
    train_idx, val_idx = split_indices(len(x_real), args.val_fraction, args.seed)
    x_train = x_real[train_idx]
    y_train = y_real[train_idx]
    x_val = x_real[val_idx]
    y_val = y_real[val_idx]

    synth_used = 0
    if args.synth_npz:
        x_syn, y_syn = load_synth(args.synth_npz)
        max_syn = int(round(len(x_train) * max(0.0, args.synth_ratio)))
        if max_syn > 0 and len(x_syn):
            rng = np.random.default_rng(args.seed)
            chosen = rng.choice(np.arange(len(x_syn)), size=min(max_syn, len(x_syn)), replace=False)
            x_train = np.concatenate([x_train, x_syn[chosen]], axis=0)
            y_train = np.concatenate([y_train, y_syn[chosen]], axis=0)
            synth_used = int(len(chosen))

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_efficientnet_super5()
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score = -1.0
    logs = []
    best_path = out_dir / "best_model.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        val_loss, val_metrics = evaluate(model, val_loader, device)
        score = val_metrics.get("macro_auprc")
        score_f = float(score) if score is not None else -val_loss
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "elapsed_sec": time.time() - t0,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
        if score_f > best_score:
            best_score = score_f
            torch.save(model.state_dict(), best_path)

    train_result = {
        "dataset_npz": str(args.dataset_npz),
        "synth_npz": str(args.synth_npz or ""),
        "synth_used": synth_used,
        "n_real": int(len(x_real)),
        "n_train_total": int(len(x_train)),
        "n_val": int(len(x_val)),
        "class_names": list(CLASS_NAMES),
        "best_score": best_score,
        "best_model": str(best_path),
        "logs": logs,
    }
    (out_dir / "training_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    (out_dir / "train_result.json").write_text(json.dumps(train_result, indent=2), encoding="utf-8")
    (out_dir / "model_card.json").write_text(
        json.dumps({
            "intended_use": "Target-hospital ECG super5 research deployment prototype",
            "warning": "Not a clinical diagnostic device. Validate on held-out target-center data before use.",
            "class_names": list(CLASS_NAMES),
            "preprocess": "crop_or_pad_1000 + per_sample_global_zscore",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[done] saved best model to {best_path}")


if __name__ == "__main__":
    main()

