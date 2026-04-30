"""Per-center style classifier (7-way) for validating CenterToken v2.

Trains an EfficientNet1DV2 head on real PN2021 records (ptb-xl shard hard-excluded
per CLAUDE.md / data integrity rule) to predict acquisition center identity.

Used POST-HOC in CT v2 smoking gun: generate K synth ECGs with CT_v2_extra hooked,
run through this classifier, expect top-1 = 'extra' substantially above chance.
If the classifier cannot distinguish CT_v2 outputs by center, CT v2 is not
encoding per-center style despite passing signal-level G1/G2/G4 gates.

PN2021 7 centers (ptb-xl forbidden):
  chapman_shaoxing, cpsc_2018, cpsc_2018_extra, georgia, ningbo, ptb, st_petersburg_incart

Caches preprocessed signals to a single .npz so the slow wfdb load happens once.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/train_per_center_style_classifier.py \
      --output_dir /root/autodl-tmp/per_center_style_classifier \
      --max_per_center 3000 --epochs 30 --batch_size 96
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model" / "DeepECG" / "notebooks"))

from scripts.crosscenter_v2.preprocess_utils import (  # noqa: E402
    unified_preprocess_to_1000, crop_signal_tc,
)
from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


PN2021_CENTERS = [
    "chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra",
    "georgia", "ningbo", "ptb", "st_petersburg_incart",
]
PN2021_FORBIDDEN = {"ptb-xl", "ptbxl"}


def scan_center_records(center_dir):
    paths = []
    for root, _, files in os.walk(center_dir):
        for f in files:
            if f.endswith(".hea"):
                paths.append(os.path.join(root, f)[:-4])
    return sorted(paths)


def load_and_preprocess(record_path):
    try:
        rec = wfdb.rdrecord(record_path)
    except Exception:
        return None
    sig = rec.p_signal
    if sig is None or sig.shape[1] < 12:
        return None
    sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
    proc = unified_preprocess_to_1000(
        sig.astype(np.float32), fs=rec.fs, source_leads=sig_names,
        target_fs=100, target_len=1000,
        apply_filter=True, apply_zscore=True,
    )
    return proc  # (1000, 12) or None


def build_or_load_cache(args):
    cache_path = Path(args.output_dir) / f"cache_max{args.max_per_center}.npz"
    if cache_path.exists() and not args.force_rebuild_cache:
        print(f"[cache] loading {cache_path}")
        z = np.load(cache_path)
        return z["signals"], z["labels"], list(z["center_names"])

    print(f"[cache] building cache at {cache_path}")
    rng = np.random.default_rng(args.seed)
    all_signals, all_labels = [], []
    for ci, center in enumerate(PN2021_CENTERS):
        assert center.lower() not in PN2021_FORBIDDEN, \
            f"FORBIDDEN center {center} would leak PTB-XL"
        center_dir = os.path.join(args.pn2021_root, "training", center)
        if not os.path.isdir(center_dir):
            print(f"  [{center}] missing dir, skipped")
            continue
        paths = scan_center_records(center_dir)
        n_scanned = len(paths)
        if args.max_per_center and len(paths) > args.max_per_center:
            paths = list(rng.choice(paths, args.max_per_center, replace=False))
        t0 = time.time()
        kept = 0
        for p in paths:
            sig = load_and_preprocess(p)
            if sig is None:
                continue
            all_signals.append(sig)  # (1000, 12)
            all_labels.append(ci)
            kept += 1
        print(f"  [{center:<22}] scanned={n_scanned}  kept={kept}  "
              f"({time.time()-t0:.0f}s)")

    signals = np.stack(all_signals).astype(np.float32)  # (N, 1000, 12)
    labels = np.array(all_labels, dtype=np.int64)
    center_names = list(PN2021_CENTERS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path,
                        signals=signals, labels=labels, center_names=center_names)
    print(f"[cache] wrote {cache_path}  signals={signals.shape}  labels={labels.shape}")
    return signals, labels, center_names


class CenterStyleDataset(Dataset):
    def __init__(self, signals_1000, labels, crop_len=250, mode="train"):
        self.signals = signals_1000
        self.labels = labels
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len,
                              mode="random" if self.mode == "train" else "center")
        sig_ct = crop.T  # (12, crop_len)
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.tensor(self.labels[idx], dtype=torch.long))


def stratified_split(labels, frac_train=0.8, frac_val=0.1, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * frac_train)
        n_va = int(n * frac_val)
        train_idx.extend(idx[:n_tr])
        val_idx.extend(idx[n_tr:n_tr + n_va])
        test_idx.extend(idx[n_tr + n_va:])
    return (np.array(sorted(train_idx)),
            np.array(sorted(val_idx)),
            np.array(sorted(test_idx)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred, all_logits = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_logits.append(logits.cpu().numpy())
        all_pred.append(pred)
        all_y.append(y.numpy())
    return (np.concatenate(all_y), np.concatenate(all_pred),
            np.concatenate(all_logits))


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pn2021_root", default="/root/autodl-tmp/physionet2021")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_per_center", type=int, default=3000,
                   help="cap each center to this many records (balance ningbo 34k)")
    p.add_argument("--crop_len", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force_rebuild_cache", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    signals, labels, center_names = build_or_load_cache(args)
    num_classes = len(center_names)
    print(f"\n[data] N={len(signals)}  C={num_classes}  centers={center_names}")
    print("[data] per-center counts: " + "  ".join(
        f"{c}={int((labels==i).sum())}" for i, c in enumerate(center_names)
    ))

    tr_idx, va_idx, te_idx = stratified_split(labels, seed=args.seed)
    print(f"[split] train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    train_ds = CenterStyleDataset(signals[tr_idx], labels[tr_idx],
                                  crop_len=args.crop_len, mode="train")
    val_ds = CenterStyleDataset(signals[va_idx], labels[va_idx],
                                crop_len=args.crop_len, mode="eval")
    test_ds = CenterStyleDataset(signals[te_idx], labels[te_idx],
                                 crop_len=args.crop_len, mode="eval")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    device = args.device
    model = EfficientNet1DV2(
        variant="s_v2", input_channels=12, num_classes=num_classes,
        activation="leaky_relu", stochastic_depth_prob=0.304,
        dropout_rate=0.0, use_se=True, norm_type="batch",
    ).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] EfficientNet1DV2 s_v2  params={n_params:,}  num_classes={num_classes}")

    cls_counts = np.bincount(labels[tr_idx], minlength=num_classes).astype(np.float32)
    cls_weights = torch.tensor(cls_counts.sum() / (cls_counts * num_classes),
                               dtype=torch.float32, device=device)
    print(f"[loss] class weights: " + "  ".join(
        f"{c}={w:.2f}" for c, w in zip(center_names, cls_weights.cpu().tolist())
    ))
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None

    best_val_acc = -1.0
    patience_ctr = 0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss, n_b = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()
            n_b += 1
        scheduler.step()
        train_loss /= max(n_b, 1)

        vy, vp, _ = evaluate(model, val_loader, device)
        val_acc = accuracy_score(vy, vp)
        val_f1 = f1_score(vy, vp, average="macro")
        elapsed = time.time() - t0

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(),
                       Path(args.output_dir) / "best_model.pt")
        else:
            patience_ctr += 1

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time": round(elapsed, 1),
        }
        log.append(entry)
        marker = " *" if improved else ""
        print(f"  Ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.5f}  ({elapsed:.0f}s){marker}")
        with open(Path(args.output_dir) / "training_log.json", "w") as f:
            json.dump(log, f, indent=2)
        if patience_ctr >= args.patience:
            print(f"  Early stop @ ep {epoch} (patience={args.patience})")
            break

    # Test
    print("\n[test] reloading best model")
    model.load_state_dict(torch.load(Path(args.output_dir) / "best_model.pt",
                                     map_location=device))
    ty, tp, tlogits = evaluate(model, test_loader, device)
    test_acc = accuracy_score(ty, tp)
    test_f1 = f1_score(ty, tp, average="macro")
    cm = confusion_matrix(ty, tp).tolist()
    cls_report = classification_report(ty, tp, target_names=center_names,
                                       output_dict=True, zero_division=0)
    print(f"  test_acc={test_acc:.4f}  test_macro_f1={test_f1:.4f}")
    print("  per-center accuracy:")
    for ci, c in enumerate(center_names):
        n = int((ty == ci).sum())
        if n == 0:
            continue
        per = (tp[ty == ci] == ci).sum() / n
        print(f"    {c:<22}  acc={per:.4f}  (n={n})")

    result = {
        "config": vars(args),
        "center_names": center_names,
        "n_total": len(signals),
        "n_per_center": {c: int((labels == i).sum())
                         for i, c in enumerate(center_names)},
        "split_sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)),
                        "test": int(len(te_idx))},
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_confusion_matrix": cm,
        "test_classification_report": cls_report,
        "epochs_trained": len(log),
        "n_params": n_params,
    }
    with open(Path(args.output_dir) / "train_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
