"""PTB-XL trainer for the thesis super5 label scheme.

Forked from the historical Tier-M trainer now archived under legacy/ with:
  - --scheme flag dispatches to label_schemes.SCHEME_REGISTRY
  - masked BCE loss for compatibility with historical label helpers
  - per-class pos_weight computed only on valid (non -1) labels
  - cudnn.benchmark = True; optional torch.compile
  - Historical augmentation ablations are kept out of the thesis training path

Usage:
    uv run python scripts/triple_labels/train_ptbxl.py \
        --scheme super5 --batch_size 96 --output_dir ${ECG_ADV_TRIPLE_ROOT}/super5
"""

import os
import sys
import json
import time
import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'DeepECG', 'notebooks'))

from scripts.triple_labels.label_schemes import get_scheme
from util.ecg_preprocessing import (
    unified_preprocess_to_1000, crop_signal_tc,
)
from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


DATA_ROOT = os.path.expanduser(os.environ.get('ECG_ADV_DATA_ROOT', '~/autodl-tmp'))
PTBXL_ROOT = os.path.expanduser(os.environ.get('ECG_ADV_PTBXL_ROOT', os.path.join(DATA_ROOT, 'ptbxl')))
TRIPLE_ROOT = os.path.expanduser(os.environ.get('ECG_ADV_TRIPLE_ROOT', os.path.join(DATA_ROOT, 'triple_labels')))


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class PTBXLDatasetScheme(Dataset):
    """PTB-XL dataset emitting (12, crop_len) signals + (C,) labels for any scheme."""

    def __init__(self, signals_1000, labels, crop_len=250, mode='train'):
        # signals_1000: (N, 1000, 12) float32
        # labels: (N, C) float32 with values in {-1, 0, 1}
        self.signals = signals_1000
        self.labels = labels.astype(np.float32, copy=False)
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


class SynthNPZDataset(Dataset):
    """Synthetic ECG dataset for super5 augmentation.

    Supports either flat keys:
      signals: (N,1000,12) or (N,12,1000), labels: (N,C)
    or cached per-center keys:
      <center>__signals, <center>__labels5
    """

    def __init__(self, npz_paths, crop_len=250, mode='train'):
        if isinstance(npz_paths, str):
            npz_paths = [p for p in npz_paths.split(',') if p]
        signals_all, labels_all = [], []
        for path in npz_paths:
            data = np.load(path)
            if 'signals' in data:
                label_key = 'labels' if 'labels' in data else 'labels5'
                signals_all.append(self._normalize_signals(data['signals']))
                labels_all.append(np.asarray(data[label_key], dtype=np.float32))
                continue

            signal_keys = sorted(k for k in data.files if k.endswith('__signals'))
            for sig_key in signal_keys:
                prefix = sig_key[:-len('__signals')]
                label_key = f'{prefix}__labels5'
                if label_key not in data:
                    continue
                signals_all.append(self._normalize_signals(data[sig_key]))
                labels_all.append(np.asarray(data[label_key], dtype=np.float32))

        if not signals_all:
            raise ValueError(f"No synthetic signals found in {npz_paths}")
        self.signals = np.concatenate(signals_all, axis=0).astype(np.float32, copy=False)
        self.labels = np.concatenate(labels_all, axis=0).astype(np.float32, copy=False)
        self.crop_len = crop_len
        self.mode = mode
        if self.signals.shape[0] != self.labels.shape[0]:
            raise ValueError(f"synth signals/labels length mismatch: {self.signals.shape} vs {self.labels.shape}")

    @staticmethod
    def _normalize_signals(signals):
        arr = np.asarray(signals, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected synth signals ndim=3, got {arr.shape}")
        if arr.shape[1:] == (12, 1000):
            arr = arr.transpose(0, 2, 1)
        if arr.shape[1:] != (1000, 12):
            raise ValueError(f"Expected synth signals as (N,1000,12) or (N,12,1000), got {arr.shape}")
        return arr

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len,
                              mode='random' if self.mode == 'train' else 'center')
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels[idx]).float())


# ────────────────────────────────────────────────────────────────────────────
# Loss + metrics
# ────────────────────────────────────────────────────────────────────────────

def masked_bce_with_logits(logits, labels_mask, pos_weight):
    """BCE-with-logits that ignores entries where label == -1."""
    mask = (labels_mask >= 0).float()
    labels_safe = torch.where(mask.bool(), labels_mask, torch.zeros_like(labels_mask))
    bce = F.binary_cross_entropy_with_logits(
        logits, labels_safe, pos_weight=pos_weight, reduction='none'
    )
    denom = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denom


def compute_pos_weight(labels_mat, num_classes, clip_max=50.0):
    """Per-class pos_weight = N_neg / N_pos (only on valid 0/1 entries),
    clipped to [1, clip_max]."""
    pw = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        col = labels_mat[:, i]
        n_pos = int((col == 1.0).sum())
        n_neg = int((col == 0.0).sum())
        if n_pos == 0:
            pw[i] = clip_max
        else:
            pw[i] = float(np.clip(n_neg / max(n_pos, 1), 1.0, clip_max))
    return pw


def compute_macro_auroc_auprc(y_true, y_score, class_names, min_pos=10):
    """Per-class AUROC/AUPRC with -1 masking. Skips classes with < min_pos."""
    aurocs, auprcs, per_class = [], [], {}
    for i, name in enumerate(class_names):
        t = y_true[:, i]
        s = y_score[:, i]
        valid = t >= 0
        t_v = t[valid]
        s_v = s[valid]
        n_pos = int((t_v == 1.0).sum())
        n_classes_unique = len(np.unique(t_v)) if t_v.size else 0
        if n_pos < min_pos or n_classes_unique < 2:
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': n_pos,
                               'n_valid': int(valid.sum())}
            continue
        try:
            auc = float(roc_auc_score(t_v, s_v))
            ap = float(average_precision_score(t_v, s_v))
        except Exception:
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': n_pos,
                               'n_valid': int(valid.sum())}
            continue
        aurocs.append(auc)
        auprcs.append(ap)
        per_class[name] = {'auroc': auc, 'auprc': ap, 'n_pos': n_pos,
                           'n_valid': int(valid.sum())}
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
    return total_loss / max(n_batches, 1), all_labels, all_probs


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


# ────────────────────────────────────────────────────────────────────────────
# Data preparation
# ────────────────────────────────────────────────────────────────────────────

def preprocess_ptbxl_all(
    raw_path,
    cache_path,
    target_fs=100,
    target_len=1000,
    preprocess_mode='legacy_ecgfounder_filter',
    norm_mode='per_sample_global',
):
    if os.path.exists(cache_path):
        print(f"[preprocess] cache hit: {cache_path}")
        return np.load(cache_path, mmap_mode='r')

    print(
        f"[preprocess] preprocessing all PTBXL → {cache_path} (first run) "
        f"mode={preprocess_mode} norm={norm_mode}"
    )
    raw = np.load(raw_path, allow_pickle=True).astype(np.float32)
    N = raw.shape[0]
    out = np.zeros((N, target_len, 12), dtype=np.float32)
    fails = []
    for i in range(N):
        proc = unified_preprocess_to_1000(
            raw[i], fs=100, source_leads=None,
            target_fs=target_fs, target_len=target_len,
            preprocess_mode=preprocess_mode, norm_mode=norm_mode,
        )
        if proc is None:
            fails.append(i)
            sig = raw[i]
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            if norm_mode == 'per_sample_global':
                sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            out[i] = sig.astype(np.float32)
        else:
            out[i] = proc
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{N}")
    if fails:
        print(f"[preprocess] WARN: {len(fails)} records fell back (filter failed)")
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(cache_path, out)
    print(f"[preprocess] saved cache: {cache_path}")
    return out


def get_ptbxl_labels_for_scheme(csv_path, scheme, label_cache_path, folds=None):
    """Load PTB-XL metadata, generate scheme-specific labels (cached on disk)."""
    import pandas as pd
    df_full = pd.read_csv(csv_path)
    if folds is not None:
        mask = df_full.strat_fold.isin(folds)
        indices = np.where(mask)[0].tolist()
        df = df_full[mask].reset_index(drop=True)
    else:
        indices = list(range(len(df_full)))
        df = df_full.copy()

    # Cache key includes class count so different preprocessing modes cannot
    # silently load each other's array.
    cache_key = f"{label_cache_path}.C{scheme['num_classes']}.all.npy"
    if os.path.exists(cache_key):
        all_labels = np.load(cache_key)
        assert all_labels.shape[1] == scheme['num_classes'], (
            f"label cache {cache_key} has shape {all_labels.shape}, "
            f"expected C={scheme['num_classes']}"
        )
    else:
        if folds is not None:
            df_for_labels = df_full
        else:
            df_for_labels = df
        all_labels = np.stack([scheme['ptbxl_fn'](r) for r in df_for_labels.scp_codes])
        os.makedirs(os.path.dirname(cache_key) or '.', exist_ok=True)
        np.save(cache_key, all_labels)
    if folds is not None:
        labels = all_labels[indices]
    else:
        labels = all_labels
    return indices, labels.astype(np.float32), df


def _load_split_json(path):
    with open(path) as f:
        split = json.load(f)
    required = ['train_indices', 'val_indices', 'test_indices']
    for key in required:
        if key not in split:
            raise ValueError(f"split_json missing {key}: {path}")
    return split


def _default_cache_path(args):
    safe_mode = str(args.preprocess_mode).replace('/', '_')
    safe_norm = str(args.norm_mode).replace('/', '_')
    cache_dir = os.path.join(TRIPLE_ROOT, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"ptbxl_{safe_mode}_{safe_norm}_fs100_len1000.npy",
    )


# ────────────────────────────────────────────────────────────────────────────
# Training driver
# ────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    scheme = get_scheme(args.scheme)
    num_classes = scheme['num_classes']
    class_names = scheme['class_names']
    print(f"[scheme] {args.scheme}  num_classes={num_classes}  class_names={class_names}")

    label_cache_path = os.path.join(args.output_dir, 'ptbxl_labels')
    print(f"[labels] generating PTB-XL labels for scheme={args.scheme}")
    split_meta = None
    if args.split_json:
        all_idx, all_labels, _ = get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=None
        )
        split_meta = _load_split_json(args.split_json)
        train_idx = [int(i) for i in split_meta['train_indices']]
        val_idx = [int(i) for i in split_meta['val_indices']]
        test_idx = [int(i) for i in split_meta['test_indices']]
        max_idx = len(all_idx) - 1
        for name, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            if any(i < 0 or i > max_idx for i in idxs):
                raise ValueError(f"{name} split has index outside [0,{max_idx}]")
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        test_labels = all_labels[test_idx]
        shutil.copyfile(args.split_json, os.path.join(args.output_dir, 'split.json'))
        print(f"[split] custom split loaded: {args.split_json}")
    else:
        train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=list(range(1, 9))
        )
        val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=[9]
        )
        test_idx, test_labels, _ = get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=[10]
        )
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print("  Per-class positives in train:")
    for i, name in enumerate(class_names):
        col = train_labels[:, i]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_mask = int((col == -1).sum())
        print(f"    {name:<12} pos={n_pos:>5}  neg={n_neg:>5}  mask={n_mask}")

    cache_path = args.cache_path or _default_cache_path(args)
    print(f"[preprocess] using cache {cache_path}")
    all_sig = preprocess_ptbxl_all(
        args.data_path,
        cache_path,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )

    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])
    test_signals = np.asarray(all_sig[test_idx])

    train_ds = PTBXLDatasetScheme(train_signals, train_labels,
                                  crop_len=args.crop_len, mode='train')
    val_ds = PTBXLDatasetScheme(val_signals, val_labels,
                                crop_len=args.crop_len, mode='eval')
    test_ds = PTBXLDatasetScheme(test_signals, test_labels,
                                 crop_len=args.crop_len, mode='eval')

    loss_labels = train_labels
    if args.synthetic_only and not args.synth_npz:
        raise ValueError("--synthetic_only requires --synth_npz")

    if args.synth_npz:
        if args.scheme != 'super5':
            raise ValueError("--synth_npz currently supports only --scheme super5")
        synth_ds = SynthNPZDataset(args.synth_npz, crop_len=args.crop_len, mode='train')
        if args.synthetic_only:
            loss_labels = synth_ds.labels
            print(f"[synth] synthetic-only training with {len(synth_ds)} samples from {args.synth_npz}")
            train_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True, persistent_workers=args.num_workers > 0)
        else:
            train_combo = ConcatDataset([train_ds, synth_ds])
            real_weight = np.ones(len(train_ds), dtype=np.float64) / max(len(train_ds), 1)
            synth_weight = (
                np.ones(len(synth_ds), dtype=np.float64)
                * float(args.synth_ratio)
                / max(len(synth_ds), 1)
            )
            sampler = WeightedRandomSampler(
                weights=np.concatenate([real_weight, synth_weight]),
                num_samples=len(train_ds),
                replacement=True,
            )
            print(f"[synth] using {len(synth_ds)} synthetic samples from {args.synth_npz}")
            print(f"[synth] target synth:real ratio={args.synth_ratio:g}; epoch samples={len(train_ds)}")
            train_loader = DataLoader(train_combo, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True, persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    model = EfficientNet1DV2(
        variant='s_v2',
        input_channels=12,
        num_classes=num_classes,
        activation='leaky_relu',
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type='batch',
    ).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] EfficientNet1DV2 s_v2  params={n_params:,}  num_classes={num_classes}")

    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        elif isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"[model] initialized from {args.init_ckpt}")

    if args.compile:
        print("[model] torch.compile(mode='default')")
        model = torch.compile(model, mode='default')

    pos_weight_np = compute_pos_weight(loss_labels, num_classes,
                                       clip_max=args.pos_weight_clip_max)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print("[loss] pos_weight: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(class_names, pos_weight_np)
    ))

    def criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01
    )
    scaler_amp = torch.cuda.amp.GradScaler() if 'cuda' in args.device else None

    best_val_auroc = -1.0
    best_val_auprc = -1.0
    best_selected_metric = -1.0
    patience_counter = 0
    log = []
    print(f"\n[train] {args.epochs} epochs  patience={args.patience}  "
          f"batch={args.batch_size}  amp={scaler_amp is not None}  "
          f"checkpoint_metric={args.checkpoint_metric}")

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
        val_loss, vy_true, vy_score = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_macro_auroc_auprc(vy_true, vy_score, class_names)
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

        sd = None
        improved_auroc = val_metrics['macro_auroc'] > best_val_auroc
        improved_auprc = val_metrics['macro_auprc'] > best_val_auprc
        selected_metric = val_metrics[f"macro_{args.checkpoint_metric}"]
        improved_selected = selected_metric > best_selected_metric

        if improved_auroc or improved_auprc:
            # Save uncompiled state_dict (strip _orig_mod. prefix if compiled)
            sd = model.state_dict()
            sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}

        if improved_auroc:
            best_val_auroc = val_metrics['macro_auroc']
            torch.save(sd, os.path.join(args.output_dir, 'best_model_auroc.pt'))
        if improved_auprc:
            best_val_auprc = val_metrics['macro_auprc']
            torch.save(sd, os.path.join(args.output_dir, 'best_model_auprc.pt'))

        if improved_selected:
            best_selected_metric = selected_metric
            patience_counter = 0
            if sd is None:
                sd = model.state_dict()
                sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
            torch.save(sd, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        marker_parts = []
        if improved_selected:
            marker_parts.append('*')
        if improved_auroc:
            marker_parts.append('A')
        if improved_auprc:
            marker_parts.append('P')
        marker = f" {'/'.join(marker_parts)}" if marker_parts else ''
        print(f"  Ep {epoch:3d}/{args.epochs} | loss={train_loss:.4f} val={val_loss:.4f} "
              f"auroc={val_metrics['macro_auroc']:.4f} "
              f"auprc={val_metrics['macro_auprc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.5f} ({elapsed:.0f}s){marker}")

        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        if patience_counter >= args.patience:
            print(f"  Early stopping @ ep {epoch} (patience={args.patience})")
            break

    # Reload best for test
    ckpt_name = f"best_model_{args.checkpoint_metric}.pt"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        ckpt_name = 'best_model.pt'
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
    test_name = "custom split test" if args.split_json else "PTB-XL fold 10"
    print(f"\n[test] evaluating {ckpt_name} on {test_name}")
    sd = torch.load(ckpt_path, map_location=device)
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    target.load_state_dict(sd)
    test_loss, ty_true, ty_score = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_macro_auroc_auprc(ty_true, ty_score, class_names)

    print(f"  Test macro AUROC={test_metrics['macro_auroc']:.4f}  "
          f"AUPRC={test_metrics['macro_auprc']:.4f}")
    print("  Per-class:")
    for name, stats in test_metrics['per_class'].items():
        if stats['auroc'] is None:
            print(f"    {name:<12} AUROC=N/A AUPRC=N/A "
                  f"(n_pos={stats['n_pos']}, n_valid={stats['n_valid']})")
        else:
            print(f"    {name:<12} AUROC={stats['auroc']:.4f} "
                  f"AUPRC={stats['auprc']:.4f} "
                  f"(n_pos={stats['n_pos']}, n_valid={stats['n_valid']})")

    result = {
        'scheme': args.scheme,
        'num_classes': num_classes,
        'class_names': list(class_names),
        'test_loss': round(test_loss, 5),
        'test_macro_auroc': round(test_metrics['macro_auroc'], 4),
        'test_macro_auprc': round(test_metrics['macro_auprc'], 4),
        'test_per_class': test_metrics['per_class'],
        'best_val_macro_auroc': round(best_val_auroc, 4),
        'best_val_macro_auprc': round(best_val_auprc, 4),
        'checkpoint_metric': args.checkpoint_metric,
        'checkpoint_path': ckpt_path,
        'epochs_trained': len(log),
        'pos_weight': pos_weight_np.tolist(),
        'config': vars(args),
        'split': split_meta,
    }
    with open(os.path.join(args.output_dir, 'train_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] results saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', required=True, choices=['super5'])
    p.add_argument('--data_path', default=os.path.join(PTBXL_ROOT, 'raw100.npy'))
    p.add_argument('--csv_path', default=os.path.join(PTBXL_ROOT, 'ptbxl_database.csv'))
    p.add_argument('--output_dir', required=True)
    p.add_argument('--cache_path', default=None)
    p.add_argument('--split_json', default=None,
                   help='Optional custom split JSON with train_indices/val_indices/test_indices')
    p.add_argument('--preprocess_mode', default='legacy_ecgfounder_filter',
                   choices=['minimal_resample', 'legacy_ecgfounder_filter',
                            'raw_for_generation_or_digital'])
    p.add_argument('--norm_mode', default='per_sample_global',
                   choices=['per_sample_global', 'none'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--cosine_tmax', type=int, default=15)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', type=_str2bool, default=False,
                   help='Use torch.compile (default off; set true for sequential mode)')
    p.add_argument('--pos_weight_clip_max', type=float, default=50.0)
    p.add_argument('--checkpoint_metric', default='auroc', choices=['auroc', 'auprc'],
                   help='Validation metric used for best_model.pt, early stopping, and test reload')
    p.add_argument('--synth_npz', default=None,
                   help='Comma-separated synthetic npz files with signals/labels or <center>__signals/<center>__labels5')
    p.add_argument('--synth_ratio', type=float, default=0.25,
                   help='Target synthetic:real sampling ratio when --synth_npz is set')
    p.add_argument('--synthetic_only', action='store_true',
                   help='Train on --synth_npz only while validating/testing on real PTB-XL splits')
    p.add_argument('--init_ckpt', default=None,
                   help='Optional EfficientNet1DV2 state_dict used to initialize training')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
