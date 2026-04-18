"""
PTBXL v2 training — strong source-domain baseline for cross-center gap study.

Improvements over baseline (scripts/crosscenter/train_ptbxl.py):
  1. Unified preprocessing pipeline (bandpass + per-sample z-score) via
     preprocess_utils.unified_preprocess_to_1000 — identical to PN2021/MIMIC
     so residual gap = genuine domain shift (not preprocessing artifact).
  2. 26-class SNOMED head (PN2021 official scored) with label masking (-1
     for classes PTBXL doesn't cover: Brady/PRWP/RAD).
  3. MaskedFocalLoss (γ=2, α=0.25) — handles -1 ignore & class imbalance.
  4. CosineAnnealingLR(T_max=15) per-epoch — DeepECG/ECGFounder consensus.
  5. dropout_rate=0.0 (only stochastic_depth for regularization).
  6. Kaiming normal init (Conv1d fan_out) + Xavier normal (Linear) + BN 1/0.
  7. Presence-based SCP labels (threshold=0) so label semantics match PN2021.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter/train_ptbxl_v2.py
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    NUM_CLASSES_26, CLASS_NAMES_26, TIER1_IDX, TIER2_IDX, ALL_IDX,
    TIER1, TIER2, get_ptbxl_26_labels,
)
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, crop_signal_tc,
)
from EfficientNetv2 import EfficientNet1DV2


# ── Masked Focal Loss ─────────────────────────────────────────────────────────
class MaskedFocalLoss(nn.Module):
    """Binary focal loss with -1 ignore index.

    BCE_with_logits * focal_weight, summed over active (label != -1) entries.
    alpha balances positive class; gamma focuses on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_value=-1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_value = ignore_value
        self.reduction = reduction

    def forward(self, logits, targets):
        mask = (targets != self.ignore_value).float()
        targets_safe = torch.where(mask.bool(), targets, torch.zeros_like(targets))

        bce = F.binary_cross_entropy_with_logits(logits, targets_safe, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets_safe + (1 - p) * (1 - targets_safe)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * targets_safe + (1 - self.alpha) * (1 - targets_safe)
        loss = alpha_t * focal_weight * bce

        loss = loss * mask
        if self.reduction == 'mean':
            n = mask.sum().clamp(min=1.0)
            return loss.sum() / n
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ── Dataset ───────────────────────────────────────────────────────────────────
class PTBXLDatasetV2(Dataset):
    """PTBXL dataset over pre-processed (N, 1000, 12) tensor.

    Signals are already run through unified_preprocess_to_1000 (bandpass
    at native 100Hz + per-sample z-score), so per-epoch work is just crop.
    """

    def __init__(self, signals_1000, labels, crop_len=250, mode='train'):
        self.signals = signals_1000  # (N, 1000, 12) float32
        self.labels = labels          # (N, 26) float32
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]  # (1000, 12)
        crop = crop_signal_tc(sig_tc, self.crop_len,
                              mode='random' if self.mode == 'train' else 'center')
        sig_ct = crop.T  # (12, crop_len)
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels[idx]).float())


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_tiered_metrics(y_true, y_score, tier_indices, tier_name):
    """Compute macro AUROC/AUPRC over given class indices, skipping masked/degenerate.

    y_true may contain -1 values → filter per class to only rows where label is 0 or 1.
    """
    aurocs, auprcs, used = [], [], []
    for i in tier_indices:
        col_t = y_true[:, i]
        col_s = y_score[:, i]
        valid = col_t != -1.0
        t, s = col_t[valid], col_s[valid]
        if len(t) == 0 or len(np.unique(t)) < 2:
            continue
        aurocs.append(roc_auc_score(t, s))
        auprcs.append(average_precision_score(t, s))
        used.append(i)
    return {
        'tier': tier_name,
        'macro_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(used),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_labels, all_logits = [], []
    total_loss, n_batches = 0.0, 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    all_probs = 1 / (1 + np.exp(-np.clip(all_logits, -50, 50)))
    return (total_loss / max(n_batches, 1),
            compute_tiered_metrics(all_labels, all_probs, TIER1_IDX, 'tier1'),
            compute_tiered_metrics(all_labels, all_probs, TIER2_IDX, 'tier2'),
            compute_tiered_metrics(all_labels, all_probs, ALL_IDX, 'all'),
            all_labels, all_probs)


# ── Weight init (DeepECG sl_e2e_training.ipynb cell-8) ────────────────────────
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ── Preprocess PTBXL once (cache to disk for reuse) ───────────────────────────
def preprocess_ptbxl_all(raw_path, cache_path, target_fs=100, target_len=1000):
    """Run unified_preprocess on every PTBXL record once; cache to npy.

    PTBXL raw100.npy is already (21799, 1000, 12) @ 100Hz with canonical lead
    order, so reorder is skipped (source_leads=None).
    """
    if os.path.exists(cache_path):
        print(f"[preprocess] cache hit: {cache_path}")
        return np.load(cache_path, mmap_mode='r')

    print(f"[preprocess] preprocessing all PTBXL → {cache_path} (first run, ~1-2 min)")
    raw = np.load(raw_path, allow_pickle=True).astype(np.float32)
    N = raw.shape[0]
    out = np.zeros((N, target_len, 12), dtype=np.float32)
    fails = []
    for i in range(N):
        proc = unified_preprocess_to_1000(
            raw[i], fs=100, source_leads=None,
            target_fs=target_fs, target_len=target_len,
            apply_filter=True, apply_zscore=True,
        )
        if proc is None:
            fails.append(i)
            # fallback: just per-sample z-score without filter
            sig = raw[i]
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            out[i] = sig.astype(np.float32)
        else:
            out[i] = proc
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{N}")
    if fails:
        print(f"[preprocess] WARN: {len(fails)} records fell back (filter failed)")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, out)
    print(f"[preprocess] saved cache: {cache_path}")
    return out


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Labels ────────────────────────────────────────────────────────────
    print("Loading PTBXL labels (26-class, presence-based)...")
    train_idx, train_labels, _ = get_ptbxl_26_labels(args.csv_path, folds=list(range(1, 9)))
    val_idx,   val_labels,   _ = get_ptbxl_26_labels(args.csv_path, folds=[9])
    test_idx,  test_labels,  _ = get_ptbxl_26_labels(args.csv_path, folds=[10])
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    # Show Tier-1 positive counts
    for i, name in zip(TIER1_IDX, TIER1):
        pos = int((train_labels[:, i] == 1.0).sum())
        print(f"    Tier-1 {name}: {pos} train+")

    # ── Preprocess (cache on autodl-tmp) ──────────────────────────────────
    print("Preprocessing PTBXL via unified pipeline...")
    cache_path = os.path.join(args.output_dir, 'ptbxl_preprocessed.npy')
    all_sig = preprocess_ptbxl_all(args.data_path, cache_path)

    train_signals = np.asarray(all_sig[train_idx])
    val_signals   = np.asarray(all_sig[val_idx])
    test_signals  = np.asarray(all_sig[test_idx])

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = PTBXLDatasetV2(train_signals, train_labels, crop_len=args.crop_len, mode='train')
    val_ds   = PTBXLDatasetV2(val_signals,   val_labels,   crop_len=args.crop_len, mode='eval')
    test_ds  = PTBXLDatasetV2(test_signals,  test_labels,  crop_len=args.crop_len, mode='eval')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = EfficientNet1DV2(
        variant='s_v2',
        input_channels=12,
        num_classes=NUM_CLASSES_26,
        activation='leaky_relu',
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type='batch',
    ).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: EfficientNet1DV2 s_v2, {n_params:,} params, {NUM_CLASSES_26} classes")

    # ── Loss / Opt / Sched ───────────────────────────────────────────────
    criterion = MaskedFocalLoss(alpha=0.25, gamma=2.0, ignore_value=-1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01
    )
    scaler_amp = torch.cuda.amp.GradScaler() if 'cuda' in args.device else None

    # ── Training loop ────────────────────────────────────────────────────
    best_val_auroc = -1.0
    patience_counter = 0
    log = []
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()

        for signals, labels in train_loader:
            signals, labels = signals.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(signals)
                    loss = criterion(logits, labels)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()  # per-epoch
        train_loss /= max(n_batches, 1)
        val_loss, t1, t2, ta, *_ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'val_tier1_auroc': round(t1['macro_auroc'], 4),
            'val_tier1_auprc': round(t1['macro_auprc'], 4),
            'val_tier2_auroc': round(t2['macro_auroc'], 4),
            'val_all_auroc':   round(ta['macro_auroc'], 4),
            'lr': round(optimizer.param_groups[0]['lr'], 6),
            'time': round(elapsed, 1),
        }
        log.append(entry)

        improved = t1['macro_auroc'] > best_val_auroc
        if improved:
            best_val_auroc = t1['macro_auroc']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        marker = ' *' if improved else ''
        print(f"  Ep {epoch:3d}/{args.epochs} | loss={train_loss:.4f} val={val_loss:.4f} "
              f"t1_auroc={t1['macro_auroc']:.4f} t1_auprc={t1['macro_auprc']:.4f} "
              f"t2_auroc={t2['macro_auroc']:.4f} all_auroc={ta['macro_auroc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.5f} ({elapsed:.0f}s){marker}")

        if patience_counter >= args.patience:
            print(f"  Early stopping @ ep {epoch} (patience={args.patience})")
            break

    # Log
    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # ── Test (fold 10) ───────────────────────────────────────────────────
    print("\nEvaluating on test set (fold 10)...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                                     map_location=device))
    test_loss, t1, t2, ta, y_true, y_score = evaluate(model, test_loader, criterion, device)

    print(f"  Test Tier-1 AUROC={t1['macro_auroc']:.4f} AUPRC={t1['macro_auprc']:.4f}")
    print(f"  Test Tier-2 AUROC={t2['macro_auroc']:.4f} AUPRC={t2['macro_auprc']:.4f}")
    print(f"  Test ALL    AUROC={ta['macro_auroc']:.4f} AUPRC={ta['macro_auprc']:.4f}")

    print("\n  Per-class AUROC/AUPRC (PTBXL fold 10):")
    per_class = {}
    for i, name in enumerate(CLASS_NAMES_26):
        col_t = y_true[:, i]
        valid = col_t != -1.0
        t, s = col_t[valid], y_score[valid, i]
        n_pos = int((t == 1.0).sum())
        if len(t) == 0 or len(np.unique(t)) < 2:
            auc, ap = None, None
            print(f"    {name:<8} AUROC=N/A AUPRC=N/A (n_pos={n_pos}, masked={(col_t==-1).sum()})")
        else:
            auc = float(roc_auc_score(t, s))
            ap = float(average_precision_score(t, s))
            print(f"    {name:<8} AUROC={auc:.4f} AUPRC={ap:.4f} (n_pos={n_pos})")
        per_class[name] = {'auroc': auc, 'auprc': ap, 'n_pos': n_pos}

    result = {
        'test_loss': round(test_loss, 5),
        'tier1': t1, 'tier2': t2, 'all': ta,
        'best_val_tier1_auroc': round(best_val_auroc, 4),
        'epochs_trained': len(log),
        'num_classes': NUM_CLASSES_26,
        'class_names': CLASS_NAMES_26,
        'per_class': per_class,
        'config': vars(args),
    }
    with open(os.path.join(args.output_dir, 'train_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/root/ECG_adv_Gen/datasets/PTBXL/raw100.npy')
    p.add_argument('--csv_path',  default='/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv')
    p.add_argument('--output_dir', default='/root/autodl-tmp/crosscenter_v2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--cosine_tmax', type=int, default=15)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
