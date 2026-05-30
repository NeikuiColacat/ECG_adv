"""MIMIC-IV-ECG Tier-M 6-class training (apples-to-apples vs train_ptbxl_tierM.py).

All model-quality hyperparameters are locked to the same defaults as
train_ptbxl_tierM.py. Only data source + label extraction + split key differ
(these are the unavoidable dataset-swap variables).

Reads from /root/autodl-tmp/mimic_tierM/ produced by build_mimic_cache.py:
  - mimic_preprocessed_f16.npy: (N, 1000, 12) float16 mmap
  - mimic_index.npz: labels_6, subject_ids, valid_mask, split (0=train, 1=val, 2=test)

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/train_mimic_tierM.py
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    TIER_M, NUM_CLASSES_TIER_M,
)
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc
from scripts.crosscenter_tierM.train_ptbxl_tierM import SynthCenterDataset
from EfficientNetv2 import EfficientNet1DV2


class MIMICTierMDataset(Dataset):
    """Reads MIMIC from float16 mmap, returns (12, crop_len) float32 + (6,) labels.

    valid_indices restricts to split-specific valid records.
    """

    def __init__(self, signals_mmap, labels_6_full, valid_indices, crop_len=250, mode='train'):
        self.signals = signals_mmap
        self.labels_6 = labels_6_full.astype(np.float32)
        self.valid_indices = np.asarray(valid_indices, dtype=np.int64)
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = int(self.valid_indices[idx])
        sig_tc = np.asarray(self.signals[real_idx]).astype(np.float32)
        crop = crop_signal_tc(
            sig_tc, self.crop_len,
            mode='random' if self.mode == 'train' else 'center',
        )
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels_6[real_idx]).float())


def compute_macro_auroc_auprc(y_true, y_score):
    aurocs, auprcs, per_class = [], [], {}
    for i, name in enumerate(TIER_M):
        t, s = y_true[:, i], y_score[:, i]
        n_pos = int((t == 1.0).sum())
        if len(np.unique(t)) < 2:
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': n_pos}
            continue
        auc = float(roc_auc_score(t, s))
        ap = float(average_precision_score(t, s))
        aurocs.append(auc)
        auprcs.append(ap)
        per_class[name] = {'auroc': auc, 'auprc': ap, 'n_pos': n_pos}
    return {
        'macro_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(aurocs),
        'per_class': per_class,
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
            compute_macro_auroc_auprc(all_labels, all_probs),
            all_labels, all_probs)


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


def compute_pos_weight(labels_6, clip_max=50.0):
    n = labels_6.shape[0]
    pw = np.zeros(NUM_CLASSES_TIER_M, dtype=np.float32)
    for i in range(NUM_CLASSES_TIER_M):
        n_pos = int((labels_6[:, i] == 1.0).sum())
        n_neg = n - n_pos
        if n_pos == 0:
            pw[i] = clip_max
        else:
            pw[i] = min(max(n_neg / n_pos, 1.0), clip_max)
    return pw


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    print(f"Loading MIMIC index from {args.index_path}")
    idx = np.load(args.index_path, allow_pickle=True)
    labels_6 = idx['labels_6'].astype(np.float32)
    valid_mask = idx['valid_mask']
    split = idx['split']
    N = labels_6.shape[0]
    print(f"  records: {N}, valid: {int(valid_mask.sum())} ({100*valid_mask.sum()/N:.2f}%)")

    train_idx = np.nonzero(valid_mask & (split == 0))[0]
    val_idx   = np.nonzero(valid_mask & (split == 1))[0]
    test_idx  = np.nonzero(valid_mask & (split == 2))[0]
    print(f"  split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    train_labels_6 = labels_6[train_idx]
    print(f"  Tier-M class positives in train:")
    for i, name in enumerate(TIER_M):
        n_pos = int((train_labels_6[:, i] == 1.0).sum())
        n_neg = int((train_labels_6[:, i] == 0.0).sum())
        print(f"    {name:<6} pos={n_pos:>7} neg={n_neg:>7}")

    print(f"\nOpening cache mmap {args.cache_path}")
    signals_mmap = np.load(args.cache_path, mmap_mode='r')
    assert signals_mmap.shape[0] == N, \
        f"cache shape {signals_mmap.shape} vs labels {N}"
    assert signals_mmap.shape[1:] == (1000, 12), \
        f"cache shape {signals_mmap.shape} unexpected"

    train_ds = MIMICTierMDataset(signals_mmap, labels_6, train_idx,
                                 crop_len=args.crop_len, mode='train')
    val_ds   = MIMICTierMDataset(signals_mmap, labels_6, val_idx,
                                 crop_len=args.crop_len, mode='eval')
    test_ds  = MIMICTierMDataset(signals_mmap, labels_6, test_idx,
                                 crop_len=args.crop_len, mode='eval')

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        pin_memory_device=args.device if 'cuda' in args.device else '',
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Optional: mix center-token synthetic samples via WeightedRandomSampler
    # (mirror of train_ptbxl_tierM.py:318-342). Synth .npz is loaded fully into
    # RAM; fine for ~6000-sample per-center outputs.
    synth_paths = []
    if args.synth_center_npz:
        synth_paths = [p.strip() for p in args.synth_center_npz.split(',') if p.strip()]
    if synth_paths:
        synth_ds = SynthCenterDataset(synth_paths, crop_len=args.crop_len)
        N_real = len(train_ds)
        N_synth = len(synth_ds)
        synth_ratio = float(args.synth_ratio)
        assert 0.0 < synth_ratio < 1.0, "synth_ratio must be in (0, 1)"
        # For finetuning we typically don't need a full-N_real epoch (641k on MIMIC
        # is ~15GB of mmap reads). Let --samples_per_epoch override; default=N_real.
        num_samples = int(args.samples_per_epoch) if args.samples_per_epoch > 0 else N_real
        print(f"[synth] mixing {N_synth} synthetic samples from {synth_paths} "
              f"with {N_real} real @ synth_ratio={synth_ratio} "
              f"(samples_per_epoch={num_samples})")
        combined_ds = ConcatDataset([train_ds, synth_ds])
        w_real = (1.0 - synth_ratio) / max(N_real, 1)
        w_synth = synth_ratio / max(N_synth, 1)
        weights = torch.cat([
            torch.full((N_real,), w_real, dtype=torch.double),
            torch.full((N_synth,), w_synth, dtype=torch.double),
        ])
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                                  sampler=sampler, drop_last=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              **loader_kwargs)

    model = EfficientNet1DV2(
        variant='s_v2',
        input_channels=12,
        num_classes=NUM_CLASSES_TIER_M,
        activation='leaky_relu',
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type='batch',
    ).to(device)
    if args.resume_from:
        print(f"[resume] loading weights from {args.resume_from} (finetune mode)")
        state = torch.load(args.resume_from, map_location=device)
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: EfficientNet1DV2 s_v2, {n_params:,} params, {NUM_CLASSES_TIER_M} classes ({TIER_M})")

    if args.compile:
        print("[compile] torch.compile(model, mode='default') — first batch will pay compile cost")
        model = torch.compile(model, mode='default')

    pos_weight_np = compute_pos_weight(train_labels_6, clip_max=50.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print(f"pos_weight per class: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(TIER_M, pos_weight_np)
    ))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01
    )
    scaler_amp = torch.cuda.amp.GradScaler() if 'cuda' in args.device else None

    best_val_auroc = -1.0
    patience_counter = 0
    log = []
    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()

        for signals, labels in train_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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

        scheduler.step()
        train_loss /= max(n_batches, 1)
        val_loss, val_metrics, *_ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'val_macro_auroc': round(val_metrics['macro_auroc'], 4),
            'val_macro_auprc': round(val_metrics['macro_auprc'], 4),
            'lr': round(optimizer.param_groups[0]['lr'], 6),
            'time': round(elapsed, 1),
        }
        log.append(entry)

        improved = val_metrics['macro_auroc'] > best_val_auroc
        if improved:
            best_val_auroc = val_metrics['macro_auroc']
            patience_counter = 0
            save_model = getattr(model, '_orig_mod', model)
            torch.save(save_model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        marker = ' *' if improved else ''
        print(f"  Ep {epoch:3d}/{args.epochs} | loss={train_loss:.4f} val={val_loss:.4f} "
              f"macro_auroc={val_metrics['macro_auroc']:.4f} "
              f"macro_auprc={val_metrics['macro_auprc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.5f} ({elapsed:.0f}s){marker}")

        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        if patience_counter >= args.patience:
            print(f"  Early stopping @ ep {epoch} (patience={args.patience})")
            break

    print("\nEvaluating on MIMIC test split (in-domain)...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                                     map_location=device))
    test_loss, test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

    print(f"  Test macro AUROC={test_metrics['macro_auroc']:.4f} "
          f"AUPRC={test_metrics['macro_auprc']:.4f}")
    print("\n  Per-class AUROC/AUPRC (MIMIC test split):")
    for name, stats in test_metrics['per_class'].items():
        if stats['auroc'] is None:
            print(f"    {name:<6} AUROC=N/A AUPRC=N/A (n_pos={stats['n_pos']})")
        else:
            print(f"    {name:<6} AUROC={stats['auroc']:.4f} AUPRC={stats['auprc']:.4f} "
                  f"(n_pos={stats['n_pos']})")

    result = {
        'test_loss': round(test_loss, 5),
        'test_macro_auroc': round(test_metrics['macro_auroc'], 4),
        'test_macro_auprc': round(test_metrics['macro_auprc'], 4),
        'test_per_class': test_metrics['per_class'],
        'best_val_macro_auroc': round(best_val_auroc, 4),
        'epochs_trained': len(log),
        'num_classes': NUM_CLASSES_TIER_M,
        'class_names': TIER_M,
        'pos_weight': pos_weight_np.tolist(),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'n_test': int(len(test_idx)),
        'config': vars(args),
    }
    with open(os.path.join(args.output_dir, 'train_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_path',
                   default='/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy')
    p.add_argument('--index_path',
                   default='/root/autodl-tmp/mimic_tierM/mimic_index.npz')
    p.add_argument('--output_dir', default='/root/autodl-tmp/mimic_tierM')
    p.add_argument('--device', default='cuda')
    # Model-quality hyperparams — locked to train_ptbxl_tierM.py defaults for apples-to-apples.
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--cosine_tmax', type=int, default=15)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    # Throughput-only knobs (do not affect gradient/loss).
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                   help='torch.compile the model (default ON; use --no-compile to disable). '
                        'May introduce tiny numeric diffs vs PTBXL baseline.')
    # Optional: mix in center-token synthetic samples (mirrors train_ptbxl_tierM.py).
    p.add_argument('--synth_center_npz', default='',
                   help='Comma-separated .npz paths from center-token generation. '
                        'Each must contain signals (N,12,1000) and labels (N,6).')
    p.add_argument('--synth_ratio', type=float, default=0.3,
                   help='Target fraction of synthetic samples per batch, in (0, 1).')
    # Optional: finetune from a trained checkpoint instead of from-scratch init.
    p.add_argument('--resume_from', default='',
                   help='Path to a best_model.pt to warm-start from (finetune mode). '
                        'Empty means train from scratch (default).')
    p.add_argument('--samples_per_epoch', type=int, default=0,
                   help='Override WeightedRandomSampler num_samples (only used when '
                        'synth_center_npz is set). 0 = use N_real (full-size epoch).')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
