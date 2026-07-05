"""PTB-XL Super5 trainer.

Package-owned source trainer with:
  - --scheme is locked to super5 for the current paper mainline
  - masked BCE loss (handles -1 unknown labels in pn26's Brady/PRWP/RAD)
  - per-class pos_weight computed only on valid (non -1) labels
  - cudnn.benchmark = True; optional torch.compile
  - Removed AugMix / SynthCenter (independent ablations)

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python ecg_adv_gen/runner/ptbxl_source_train.py \
        --scheme super5 --batch_size 96 --output_dir /root/autodl-tmp/triple_labels/super5
"""

import os
import sys
import json
import time
import argparse
import shutil

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'DeepECG', 'notebooks'))

from ecg_adv_gen.labels import get_super5_scheme
from ecg_adv_gen.models.super5_model_zoo import (
    available_model_names, build_super5_model, normalize_model_name,
)
import ecg_adv_gen.evaluation as evaluation_helpers
import ecg_adv_gen.training as training_helpers
import ecg_adv_gen.data.ptbxl as ptbxl_data


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')


# ────────────────────────────────────────────────────────────────────────────
# Data preparation
# ────────────────────────────────────────────────────────────────────────────

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

    scheme = get_super5_scheme()
    num_classes = scheme['num_classes']
    class_names = scheme['class_names']
    print(f"[scheme] {args.scheme}  num_classes={num_classes}  class_names={class_names}")

    label_cache_path = os.path.join(args.output_dir, 'ptbxl_labels')
    print(f"[labels] generating PTB-XL labels for scheme={args.scheme}")
    split_meta = None
    if args.split_json:
        all_idx, all_labels, _ = ptbxl_data.get_ptbxl_labels_for_scheme(
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
        train_idx, train_labels, _ = ptbxl_data.get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=list(range(1, 9))
        )
        val_idx, val_labels, _ = ptbxl_data.get_ptbxl_labels_for_scheme(
            args.csv_path, scheme, label_cache_path, folds=[9]
        )
        test_idx, test_labels, _ = ptbxl_data.get_ptbxl_labels_for_scheme(
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
    all_sig = ptbxl_data.preprocess_ptbxl_all(
        args.data_path,
        cache_path,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )

    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])
    test_signals = np.asarray(all_sig[test_idx])

    train_ds = training_helpers.PTBXLDatasetScheme(train_signals, train_labels,
                                                   crop_len=args.crop_len, mode='train')
    val_ds = training_helpers.PTBXLDatasetScheme(val_signals, val_labels,
                                                 crop_len=args.crop_len, mode='eval')
    test_ds = training_helpers.PTBXLDatasetScheme(test_signals, test_labels,
                                                  crop_len=args.crop_len, mode='eval')

    loss_labels = train_labels
    if args.synthetic_only and not args.synth_npz:
        raise ValueError("--synthetic_only requires --synth_npz")

    if args.synth_npz:
        if args.scheme != 'super5':
            raise ValueError("--synth_npz currently supports only --scheme super5")
        synth_ds = training_helpers.SynthNPZDataset(args.synth_npz, crop_len=args.crop_len, mode='train')
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

    model_name = normalize_model_name(args.model_name)
    model = build_super5_model(model_name, num_classes=num_classes).to(device)
    model.apply(training_helpers.init_weights)
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
        print("[model] torch.compile(mode='default')")
        model = torch.compile(model, mode='default')

    pos_weight_np = training_helpers.compute_pos_weight(loss_labels, num_classes,
                                                        clip_max=args.pos_weight_clip_max)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print(f"[loss] pos_weight: " + ", ".join(
        f"{n}={w:.2f}" for n, w in zip(class_names, pos_weight_np)
    ))

    def criterion(logits, labels):
        return training_helpers.masked_bce_with_logits(logits, labels, pos_weight)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)
        val_loss, vy_true, vy_score = training_helpers.evaluate(model, val_loader, criterion, device)
        val_metrics = evaluation_helpers.compute_macro_auroc_auprc(vy_true, vy_score, class_names)
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
    test_loss, ty_true, ty_score = training_helpers.evaluate(model, test_loader, criterion, device)
    test_metrics = evaluation_helpers.compute_macro_auroc_auprc(ty_true, ty_score, class_names)

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
    p.add_argument('--scheme', required=True, choices=['super5'])
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
                   help='Optional matching-architecture state_dict used to initialize training')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
