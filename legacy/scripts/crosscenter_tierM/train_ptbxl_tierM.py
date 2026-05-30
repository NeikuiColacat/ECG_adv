"""PTBXL Tier-M 6-class training (ECGTwin-friendly victim baseline).

Fork of scripts/crosscenter_v2/train_ptbxl_v2.py. Differences:
  1. 6-class head (TIER_M = NSR/STach/AF/IAVB/LBBB/RBBB) — all PTBXL-covered.
  2. BCEWithLogitsLoss with per-class pos_weight (no -1 mask needed).
  3. Early stopping on val 6-class macro AUROC.
  4. Re-uses PTBXL preprocessed cache from crosscenter_v2 (same preprocessing).

See docs/label_selection_research.md §7.6 for the label choice rationale.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/train_ptbxl_tierM.py
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
    TIER_M, TIER_M_IDX, NUM_CLASSES_TIER_M, get_ptbxl_26_labels,
)
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, crop_signal_tc,
)
from methods.augmix.augmix import augmix as _augmix_fn
from methods.augmix.latent_viz.latent_augmix import latent_augmix_batch
from util.ecgtwin_utils import ECGTwinWrapper
from EfficientNetv2 import EfficientNet1DV2


def _random_crop_batch(sigs_bct: torch.Tensor, crop_len: int) -> torch.Tensor:
    """Per-sample random crop along the last dim.

    (B, 12, L) -> (B, 12, crop_len) — each sample gets an independent random start.
    """
    B, C, L = sigs_bct.shape
    if L == crop_len:
        return sigs_bct
    if L < crop_len:
        pad = crop_len - L
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(sigs_bct, (left, right))
    max_start = L - crop_len
    starts = torch.randint(0, max_start + 1, (B,), device=sigs_bct.device)
    out = torch.empty((B, C, crop_len), dtype=sigs_bct.dtype, device=sigs_bct.device)
    for b in range(B):
        s = int(starts[b].item())
        out[b] = sigs_bct[b, :, s:s + crop_len]
    return out


def make_augmix_collate(prob: float, severity: int, width: int, depth: int):
    """Per-sample Bernoulli AugMix collate. prob=0 -> identity (but we gate upstream)."""
    def _collate(batch):
        signals, labels = zip(*batch)
        signals = list(signals)
        if prob > 0.0:
            mask = np.random.random(len(signals)) < prob
            for i, apply in enumerate(mask):
                if apply:
                    signals[i] = _augmix_fn(signals[i], severity=severity,
                                            width=width, depth=depth)
        return torch.stack(signals), torch.stack(list(labels))
    return _collate


class PTBXLDatasetTierM(Dataset):
    """PTBXL dataset returning 6-class Tier-M labels sliced from the 26-d vector.

    If ``return_uncropped=True``, returns the full (12, 1000) signal with no
    cropping. Used by latent-AugMix pipeline where augmentation requires the
    full 10s window and cropping happens in the training loop afterwards.
    """

    def __init__(self, signals_1000, labels_26, crop_len=250, mode='train',
                 return_uncropped=False):
        self.signals = signals_1000
        self.labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)
        self.crop_len = crop_len
        self.mode = mode
        self.return_uncropped = return_uncropped

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        if self.return_uncropped:
            sig_ct = np.asarray(sig_tc).T                    # (12, 1000)
        else:
            crop = crop_signal_tc(sig_tc, self.crop_len,
                                  mode='random' if self.mode == 'train' else 'center')
            sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels_6[idx]).float())


class SynthCenterDataset(Dataset):
    """Synthetic ECGs from one or more center-token generations.

    Loads .npz files with ``signals: (N, 12, 1000) float32`` and
    ``labels: (N, 6) float32``. Returns random-cropped (12, crop_len) per sample.
    """

    def __init__(self, npz_paths, crop_len=250):
        sig_list, lbl_list = [], []
        for p in npz_paths:
            d = np.load(p)
            sig_list.append(d['signals'].astype(np.float32))
            lbl_list.append(d['labels'].astype(np.float32))
        self.signals = np.concatenate(sig_list, axis=0)  # (N, 12, 1000)
        self.labels = np.concatenate(lbl_list, axis=0)   # (N, 6)
        self.crop_len = crop_len
        assert self.signals.ndim == 3 and self.signals.shape[1] == 12
        assert self.labels.shape[0] == self.signals.shape[0]

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        sig = self.signals[idx]   # (12, 1000)
        L = sig.shape[1]
        if L > self.crop_len:
            start = np.random.randint(0, L - self.crop_len + 1)
            sig = sig[:, start:start + self.crop_len]
        return (torch.from_numpy(np.ascontiguousarray(sig)).float(),
                torch.from_numpy(self.labels[idx]).float())


def compute_macro_auroc_auprc(y_true, y_score):
    """Macro AUROC/AUPRC over all 6 Tier-M classes. All classes are covered by PTBXL
    (no -1), so a class can only be skipped if val/test split happens to have
    fewer than 2 unique labels (extremely rare)."""
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


def preprocess_ptbxl_all(raw_path, cache_path, target_fs=100, target_len=1000):
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


def compute_pos_weight(labels_6, clip_max=50.0):
    """Per-class pos_weight = N_neg / N_pos, clipped to [1, clip_max]."""
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

    print("Loading PTBXL labels (26-class, presence-based; will slice to Tier-M 6)...")
    train_idx, train_labels_26, _ = get_ptbxl_26_labels(args.csv_path, folds=list(range(1, 9)))
    val_idx,   val_labels_26,   _ = get_ptbxl_26_labels(args.csv_path, folds=[9])
    test_idx,  test_labels_26,  _ = get_ptbxl_26_labels(args.csv_path, folds=[10])
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    train_labels_6 = train_labels_26[:, TIER_M_IDX].astype(np.float32)
    print(f"  Tier-M class positives in train:")
    for i, name in enumerate(TIER_M):
        n_pos = int((train_labels_6[:, i] == 1.0).sum())
        n_neg = int((train_labels_6[:, i] == 0.0).sum())
        n_masked = int((train_labels_6[:, i] == -1.0).sum())
        print(f"    {name:<6} pos={n_pos:>5} neg={n_neg:>5} masked={n_masked}")
    assert int((train_labels_6 == -1.0).sum()) == 0, \
        "Tier-M expected to have no -1 labels (all 6 classes PTBXL-covered)"

    # Re-use PTBXL preprocessed cache from crosscenter_v2 if available
    cache_path = args.cache_path
    if cache_path is None:
        default_v2_cache = '/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy'
        if os.path.exists(default_v2_cache):
            cache_path = default_v2_cache
            print(f"  reusing v2 preprocessed cache: {cache_path}")
        else:
            cache_path = os.path.join(args.output_dir, 'ptbxl_preprocessed.npy')
    print("Preprocessing PTBXL via unified pipeline...")
    all_sig = preprocess_ptbxl_all(args.data_path, cache_path)

    train_signals = np.asarray(all_sig[train_idx])
    val_signals   = np.asarray(all_sig[val_idx])
    test_signals  = np.asarray(all_sig[test_idx])

    # Augmentation mode dispatch
    augmix_active = (args.augmix_mode != 'off' and args.augmix_prob > 0.0)
    is_latent = (args.augmix_mode == 'latent' and augmix_active)
    is_time = (args.augmix_mode == 'time' and augmix_active)
    if augmix_active:
        print(f"[augmix] mode={args.augmix_mode}  prob={args.augmix_prob} "
              f"severity={args.augmix_severity} width={args.augmix_width} "
              f"depth={args.augmix_depth}")
    else:
        print(f"[augmix] disabled (mode={args.augmix_mode}, prob={args.augmix_prob})")

    train_ds = PTBXLDatasetTierM(train_signals, train_labels_26, crop_len=args.crop_len,
                                 mode='train', return_uncropped=is_latent)
    val_ds   = PTBXLDatasetTierM(val_signals,   val_labels_26,   crop_len=args.crop_len, mode='eval')
    test_ds  = PTBXLDatasetTierM(test_signals,  test_labels_26,  crop_len=args.crop_len, mode='eval')

    ecgtwin = None
    if is_latent:
        print("[augmix-latent] loading ECGTwin VAE wrapper (load_text_model=False)...")
        ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
        print("[augmix-latent] wrapper ready.")

    train_collate = None
    if is_time:
        train_collate = make_augmix_collate(
            prob=args.augmix_prob,
            severity=args.augmix_severity,
            width=args.augmix_width,
            depth=args.augmix_depth,
        )

    # Optional: mix in center-token synthetic samples via WeightedRandomSampler.
    synth_paths = []
    if args.synth_center_npz:
        synth_paths = [p.strip() for p in args.synth_center_npz.split(',') if p.strip()]
    if synth_paths:
        assert not is_latent, "synth_center_npz + augmix_mode=latent not supported together"
        synth_ds = SynthCenterDataset(synth_paths, crop_len=args.crop_len)
        N_real = len(train_ds)
        N_synth = len(synth_ds)
        synth_ratio = float(args.synth_ratio)
        assert 0.0 < synth_ratio < 1.0, "synth_ratio must be in (0, 1)"
        print(f"[synth] mixing {N_synth} synthetic samples from {synth_paths} "
              f"with {N_real} real at synth_ratio={synth_ratio}")
        combined_ds = ConcatDataset([train_ds, synth_ds])
        w_real = (1.0 - synth_ratio) / max(N_real, 1)
        w_synth = synth_ratio / max(N_synth, 1)
        weights = torch.cat([
            torch.full((N_real,), w_real, dtype=torch.double),
            torch.full((N_synth,), w_synth, dtype=torch.double),
        ])
        sampler = WeightedRandomSampler(weights=weights, num_samples=N_real, replacement=True)
        train_loader = DataLoader(combined_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True,
                                  collate_fn=train_collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                  collate_fn=train_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

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

            # Latent AugMix happens here (signals arrive uncropped (B,12,1000))
            if is_latent:
                B = signals.shape[0]
                mask = torch.rand(B, device=device) < args.augmix_prob
                aug_idx = mask.nonzero(as_tuple=True)[0]
                if aug_idx.numel() > 0:
                    augd = latent_augmix_batch(
                        ecgtwin, signals[aug_idx],
                        severity=args.augmix_severity,
                        width=args.augmix_width,
                        depth=args.augmix_depth,
                    )
                    signals = signals.clone()
                    signals[aug_idx] = augd
                signals = _random_crop_batch(signals, args.crop_len)

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
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        marker = ' *' if improved else ''
        print(f"  Ep {epoch:3d}/{args.epochs} | loss={train_loss:.4f} val={val_loss:.4f} "
              f"macro_auroc={val_metrics['macro_auroc']:.4f} "
              f"macro_auprc={val_metrics['macro_auprc']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.5f} ({elapsed:.0f}s){marker}")

        if patience_counter >= args.patience:
            print(f"  Early stopping @ ep {epoch} (patience={args.patience})")
            break

    with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print("\nEvaluating on test set (fold 10)...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                                     map_location=device))
    test_loss, test_metrics, y_true, y_score = evaluate(model, test_loader, criterion, device)

    print(f"  Test macro AUROC={test_metrics['macro_auroc']:.4f} "
          f"AUPRC={test_metrics['macro_auprc']:.4f}")
    print("\n  Per-class AUROC/AUPRC (PTBXL fold 10):")
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
        'tier_m_idx_in_26': TIER_M_IDX,
        'pos_weight': pos_weight_np.tolist(),
        'config': vars(args),
    }
    with open(os.path.join(args.output_dir, 'train_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/root/ECG_adv_Gen/datasets/PTBXL/raw100.npy')
    p.add_argument('--csv_path',  default='/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv')
    p.add_argument('--output_dir', default='/root/autodl-tmp/crosscenter_tierM')
    p.add_argument('--cache_path', default=None,
                   help='PTBXL preprocessed cache path (default: reuse crosscenter_v2)')
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
    p.add_argument('--augmix_mode', choices=['off', 'time', 'latent'], default='off',
                   help='off = no aug; time = time-domain AugMix; latent = VAE-latent AugMix.')
    p.add_argument('--augmix_prob', type=float, default=0.0,
                   help='Per-sample probability to apply AugMix. 0 = baseline, no augmentation.')
    p.add_argument('--augmix_severity', type=int, default=5,
                   help='AugMix op severity, integer in [1,10].')
    p.add_argument('--augmix_width', type=int, default=3,
                   help='Number of AugMix chains to Dirichlet-mix.')
    p.add_argument('--augmix_depth', type=int, default=-1,
                   help='AugMix chain depth. -1 = random in {1,2,3}.')
    p.add_argument('--synth_center_npz', type=str, default='',
                   help='Comma-separated list of synthetic-center .npz paths to mix into training.')
    p.add_argument('--synth_ratio', type=float, default=0.3,
                   help='Target fraction of synth samples per batch (WeightedRandomSampler).')
    p.add_argument('--resume_from', default='',
                   help='Path to a .pt state_dict to resume from (finetune mode). '
                        'When set, skips random init. Consider lowering --lr to 1e-4 and '
                        '--epochs to 5 for proper finetune.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
