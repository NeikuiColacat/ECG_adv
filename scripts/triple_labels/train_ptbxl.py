"""Parametrized PTB-XL trainer for the 3 label schemes (super5/sub23/pn26).

Forked from scripts/crosscenter_tierM/train_ptbxl_tierM.py with:
  - --scheme flag dispatches to label_schemes.SCHEME_REGISTRY
  - masked BCE loss (handles -1 unknown labels in pn26's Brady/PRWP/RAD)
  - per-class pos_weight computed only on valid (non -1) labels
  - cudnn.benchmark = True; optional torch.compile
  - Removed AugMix / SynthCenter (independent ablations)

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
        --scheme super5 --batch_size 96 --output_dir /root/autodl-tmp/triple_labels/super5
"""

import os
import sys
import json
import time
import argparse
import contextlib
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'DeepECG', 'notebooks'))

from scripts.triple_labels.label_schemes import get_scheme
from scripts.triple_labels.model_zoo import (
    available_model_names, build_super5_model, normalize_model_name,
)
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, crop_signal_tc,
)
from ecg_adv_gen.evaluation import compute_macro_metric_dict
from ecg_adv_gen.training import compute_pos_weight, masked_bce_with_logits


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
        # signals_1000: (N, T, 12) float32. Historical default T=1000.
        # labels: (N, C) float32 with values in {-1, 0, 1}
        self.signals = signals_1000
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]  # (T, 12)
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

def compute_macro_auroc_auprc(y_true, y_score, class_names, min_pos=10):
    """Per-class AUROC/AUPRC with -1 masking. Skips classes with < min_pos."""
    return compute_macro_metric_dict(
        y_true,
        y_score,
        class_names=class_names,
        min_pos=min_pos,
        valid_label_min=0.0,
        empty_value=float('nan'),
        skip_metric_errors=True,
    )


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
    csv_path=None,
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
        f"fs={target_fs} len={target_len} mode={preprocess_mode} norm={norm_mode}"
    )
    use_records500 = int(target_fs) == 500
    if use_records500:
        import pandas as pd
        import wfdb

        if not csv_path:
            raise ValueError("csv_path is required when target_fs=500")
        df = pd.read_csv(csv_path)
        if 'filename_hr' not in df.columns:
            raise ValueError(f"{csv_path} has no filename_hr column")
        ptbxl_root = os.path.dirname(os.path.abspath(csv_path))
        raw = None
        N = len(df)
    else:
        raw = np.load(raw_path, allow_pickle=True).astype(np.float32)
        df = None
        ptbxl_root = None
        N = raw.shape[0]
    out = np.zeros((N, target_len, 12), dtype=np.float32)
    fails = []
    for i in range(N):
        fs = 100
        source_leads = None
        if use_records500:
            row = df.iloc[i]
            rec_base = os.path.join(ptbxl_root, str(row.filename_hr))
            try:
                rec = wfdb.rdrecord(rec_base)
                sig = np.asarray(rec.p_signal, dtype=np.float32)
                fs = int(round(float(rec.fs)))
                source_leads = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
            except Exception:
                fails.append(i)
                sig = np.zeros((target_len, 12), dtype=np.float32)
                fs = target_fs
        else:
            sig = raw[i]
        proc = unified_preprocess_to_1000(
            sig, fs=fs, source_leads=source_leads,
            target_fs=target_fs, target_len=target_len,
            preprocess_mode=preprocess_mode, norm_mode=norm_mode,
        )
        if proc is None:
            fails.append(i)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            if sig.shape[0] < target_len:
                pad = np.zeros((target_len - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
                sig = np.concatenate([sig, pad], axis=0)
            elif sig.shape[0] > target_len:
                sig = sig[:target_len]
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

    # Cache key includes class count so different schemes can't silently load
    # each other's array (super5 C=5, sub23 C=23, pn26 C=26).
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
    cache_dir = '/root/autodl-tmp/triple_labels/cache'
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"ptbxl_{safe_mode}_{safe_norm}_fs{int(args.sampling_rate)}_len{int(args.input_len)}.npy",
    )


# ────────────────────────────────────────────────────────────────────────────
# Training driver
# ────────────────────────────────────────────────────────────────────────────

def _loader_kwargs(args, *, train):
    kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': bool(args.pin_memory and 'cuda' in str(args.device)),
        'drop_last': bool(train and args.drop_last),
        'persistent_workers': bool(args.num_workers > 0 and args.persistent_workers),
    }
    if args.num_workers > 0:
        kwargs['prefetch_factor'] = args.prefetch_factor
    return kwargs


def _amp_context(args):
    if 'cuda' not in str(args.device) or not args.amp:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        if args.matmul_precision:
            torch.set_float32_matmul_precision(args.matmul_precision)

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
    print(f"  Per-class positives in train:")
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
        csv_path=args.csv_path,
        target_fs=args.sampling_rate,
        target_len=args.input_len,
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
            train_loader = DataLoader(
                synth_ds,
                shuffle=True,
                **_loader_kwargs(args, train=True),
            )
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
            train_loader = DataLoader(
                train_combo,
                sampler=sampler,
                **_loader_kwargs(args, train=True),
            )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **_loader_kwargs(args, train=True),
        )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **_loader_kwargs(args, train=False),
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **_loader_kwargs(args, train=False),
    )

    model_name = normalize_model_name(args.model_name)
    model = build_super5_model(model_name, num_classes=num_classes).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_name}  params={n_params:,}  num_classes={num_classes}")

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
        print(f"[model] torch.compile(mode={args.compile_mode!r})")
        model = torch.compile(model, mode=args.compile_mode)

    pos_weight_np = compute_pos_weight(loss_labels, num_classes,
                                       clip_max=args.pos_weight_clip_max)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print(f"[loss] pos_weight: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(class_names, pos_weight_np)
    ))

    def criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    optim_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.fused_adamw and 'cuda' in args.device:
        optim_kwargs['fused'] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    except TypeError:
        optim_kwargs.pop('fused', None)
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.cosine_tmax, eta_min=args.lr * 0.01
    )
    scaler_enabled = 'cuda' in args.device and args.amp and args.amp_dtype == 'fp16'
    try:
        scaler_amp = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler_amp = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_val_auroc = -1.0
    best_val_auprc = -1.0
    best_selected_metric = -1.0
    patience_counter = 0
    log = []
    print(f"\n[train] {args.epochs} epochs  patience={args.patience}  "
          f"batch={args.batch_size}  amp={args.amp} dtype={args.amp_dtype} "
          f"checkpoint_metric={args.checkpoint_metric}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()

        for signals, labels in train_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler_amp.is_enabled():
                with _amp_context(args):
                    logits = model(signals)
                    loss = criterion(logits, labels)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                with _amp_context(args):
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
    print(f"\n[test] evaluating {ckpt_name} on PTB-XL fold 10")
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
        'model_name': model_name,
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
    p.add_argument('--scheme', required=True, choices=['super5', 'sub23', 'pn26'])
    p.add_argument('--data_path', default='/root/autodl-tmp/ptbxl/raw100.npy')
    p.add_argument('--csv_path', default='/root/autodl-tmp/ptbxl/ptbxl_database.csv')
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
    p.add_argument('--model_name', default='efficientnet1dv2',
                   choices=available_model_names())
    p.add_argument('--sampling_rate', type=int, default=100,
                   help='Target sampling rate for PTB-XL preprocessing/cache. Default keeps legacy 100Hz behavior.')
    p.add_argument('--input_len', type=int, default=1000,
                   help='Target pre-crop sequence length. Use 5000 with --sampling_rate 500 for records500.')
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=96)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--cosine_tmax', type=int, default=15)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--pin_memory', type=_str2bool, default=True)
    p.add_argument('--persistent_workers', type=_str2bool, default=True)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--drop_last', type=_str2bool, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', type=_str2bool, default=False,
                   help='Use torch.compile (default off; set true for sequential mode)')
    p.add_argument('--compile_mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--amp', type=_str2bool, default=True,
                   help='Use CUDA autocast when device is cuda.')
    p.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16'],
                   help='Autocast dtype. bf16 is the default for RTX 4090 class GPUs.')
    p.add_argument('--allow_tf32', type=_str2bool, default=True,
                   help='Enable TF32 matmul/cuDNN paths on Ampere/Ada GPUs.')
    p.add_argument('--matmul_precision', default='high',
                   choices=['highest', 'high', 'medium', ''],
                   help='torch.set_float32_matmul_precision value; empty string disables it.')
    p.add_argument('--fused_adamw', type=_str2bool, default=True,
                   help='Use fused AdamW when available on CUDA.')
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
                   help='Optional matching-architecture state_dict used to initialize training')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
