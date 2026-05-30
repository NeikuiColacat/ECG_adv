"""
v2 cross-center evaluation — 26-class head, unified preprocessing, tiered metrics.

Key differences from eval_crosscenter.py:
  - Uses unified_preprocess_to_1000 (bandpass at native fs + per-sample z-score)
    — identical pipeline to training, so any gap reflects real domain shift.
  - 26-class SNOMED head from label_alignment_v2.
  - Tiered reporting: Tier-1 (5 core), Tier-2 (15 legacy), ALL (26).
  - AUROC + AUPRC per tier with bootstrap n=1000 95% CI.
  - Splits 5 "main" centers (avg reported) vs 2 "small" centers (listed, not averaged).

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter/eval_crosscenter_v2.py \
        --model_dir /root/autodl-tmp/crosscenter_v2 \
        --data_dir  datasets/physionet2021
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    NUM_CLASSES_26, CLASS_NAMES_26, TIER1_IDX, TIER2_IDX, ALL_IDX,
    TIER1, TIER2, ALL_SCORED_SNOMED, snomed_to_26, has_any_scored_class,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000, crop_signal_tc
from EfficientNetv2 import EfficientNet1DV2


# ── Centers ───────────────────────────────────────────────────────────────────
MAIN_CENTERS = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra', 'georgia', 'ningbo']
SMALL_CENTERS = ['ptb', 'st_petersburg_incart']
ALL_CENTERS = MAIN_CENTERS + SMALL_CENTERS


# ── Header parsing ────────────────────────────────────────────────────────────
def parse_header_snomed(header_path):
    with open(header_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and 'Dx' in line:
                idx = line.index('Dx')
                codes_str = line[idx:].split(':', 1)[1].strip()
                try:
                    return [int(c.strip()) for c in codes_str.split(',') if c.strip()]
                except ValueError:
                    return []
    return []


def scan_center_records(center_dir):
    record_paths, snomed_lists = [], []
    for root, dirs, files in os.walk(center_dir):
        for f in files:
            if f.endswith('.hea'):
                hea_path = os.path.join(root, f)
                record_path = hea_path[:-4]
                snomed = parse_header_snomed(hea_path)
                if has_any_scored_class(snomed):
                    record_paths.append(record_path)
                    snomed_lists.append(snomed)
    return record_paths, snomed_lists


# ── Dataset ───────────────────────────────────────────────────────────────────
class PN2021CenterDatasetV2(Dataset):
    """Loads records, applies unified_preprocess_to_1000 eagerly, caches (N,1000,12)."""

    def __init__(self, record_paths, labels, crop_len=250):
        self.signals = []
        self.valid_indices = []
        fail = 0
        t0 = time.time()
        for i, path in enumerate(record_paths):
            try:
                rec = wfdb.rdrecord(path)
            except Exception:
                fail += 1
                continue
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                fail += 1
                continue
            fs = rec.fs
            sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None

            proc = unified_preprocess_to_1000(
                sig.astype(np.float32),
                fs=fs,
                source_leads=sig_names,
                target_fs=100,
                target_len=1000,
                apply_filter=True,
                apply_zscore=True,
            )
            if proc is None:
                fail += 1
                continue
            self.signals.append(proc)
            self.valid_indices.append(i)
        self.labels = labels
        self.crop_len = crop_len
        self._fail = fail
        self._load_time = time.time() - t0

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        label = self.labels[self.valid_indices[idx]]
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(label).float())


# ── Metrics ───────────────────────────────────────────────────────────────────
def _class_metrics(t, s):
    """Return (auroc, auprc) or (None, None) if unsuitable."""
    if len(t) == 0 or len(np.unique(t)) < 2:
        return None, None
    return float(roc_auc_score(t, s)), float(average_precision_score(t, s))


def compute_tiered(y_true, y_score, indices):
    """Compute macro AUROC & AUPRC on the given class indices (skip masked/degenerate)."""
    aurocs, auprcs = [], []
    for i in indices:
        col_t = y_true[:, i]
        valid = col_t != -1.0
        auc, ap = _class_metrics(col_t[valid], y_score[valid, i])
        if auc is not None:
            aurocs.append(auc)
            auprcs.append(ap)
    return {
        'macro_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(aurocs),
    }


def bootstrap_tiered(y_true, y_score, indices, n_bootstrap=1000, seed=42):
    """Bootstrap macro AUROC + AUPRC over the given tier."""
    point = compute_tiered(y_true, y_score, indices)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_auroc, boot_auprc = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        t, s = y_true[idx], y_score[idx]
        m = compute_tiered(t, s, indices)
        if not np.isnan(m['macro_auroc']):
            boot_auroc.append(m['macro_auroc'])
            boot_auprc.append(m['macro_auprc'])
    if boot_auroc:
        point['auroc_ci_low']  = float(np.percentile(boot_auroc, 2.5))
        point['auroc_ci_high'] = float(np.percentile(boot_auroc, 97.5))
        point['auprc_ci_low']  = float(np.percentile(boot_auprc, 2.5))
        point['auprc_ci_high'] = float(np.percentile(boot_auprc, 97.5))
    else:
        point.update({'auroc_ci_low': None, 'auroc_ci_high': None,
                      'auprc_ci_low': None, 'auprc_ci_high': None})
    return point


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_dataset(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())
    y_true = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    y_score = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return y_true, y_score


def evaluate_center(model, center_name, center_dir, device, args):
    print(f"\n  [{center_name}] scanning...", end=' ', flush=True)
    t0 = time.time()
    paths, snomed_lists = scan_center_records(center_dir)
    print(f"{len(paths)} scored records ({time.time()-t0:.0f}s)")
    if not paths:
        return None
    labels = np.stack([snomed_to_26(s) for s in snomed_lists])
    ds = PN2021CenterDatasetV2(paths, labels, crop_len=args.crop_len)
    print(f"  [{center_name}] loaded {len(ds)}/{len(paths)} (fail={ds._fail}, {ds._load_time:.0f}s)")
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    y_true, y_score = infer_dataset(model, loader, device)

    result = {
        'center': center_name,
        'n_records': len(ds),
        'n_scored_records': len(paths),
        'tier1': bootstrap_tiered(y_true, y_score, TIER1_IDX, n_bootstrap=args.n_bootstrap),
        'tier2': bootstrap_tiered(y_true, y_score, TIER2_IDX, n_bootstrap=args.n_bootstrap),
        'all':   bootstrap_tiered(y_true, y_score, ALL_IDX,   n_bootstrap=args.n_bootstrap),
    }

    per_class = {}
    for i, name in enumerate(CLASS_NAMES_26):
        col_t = y_true[:, i]
        valid = col_t != -1.0
        t, s = col_t[valid], y_score[valid, i]
        auc, ap = _class_metrics(t, s)
        per_class[name] = {
            'auroc': round(auc, 4) if auc is not None else None,
            'auprc': round(ap, 4) if ap is not None else None,
            'n_pos': int((t == 1.0).sum()),
        }
    result['per_class'] = per_class
    return result


def _fmt_tier(t):
    if t.get('auroc_ci_low') is None:
        return f"{t['macro_auroc']:.4f} AUPRC={t['macro_auprc']:.4f}"
    return (f"{t['macro_auroc']:.4f} ({t['auroc_ci_low']:.4f}-{t['auroc_ci_high']:.4f}) "
            f"AUPRC={t['macro_auprc']:.4f}")


def main(args):
    device = torch.device(args.device)

    # Model (v2: 26 classes, dropout 0.0)
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
    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    results = {}
    for center in ALL_CENTERS:
        center_dir = os.path.join(args.data_dir, 'training', center)
        if not os.path.isdir(center_dir):
            # Some datasets put centers under different dir names
            center_dir_alt = os.path.join(args.data_dir, center)
            if os.path.isdir(center_dir_alt):
                center_dir = center_dir_alt
            else:
                print(f"\n  Skipping {center} (dir not found)")
                continue
        r = evaluate_center(model, center, center_dir, device, args)
        if r is None:
            continue
        results[center] = r
        print(f"  [{center}] Tier-1 AUROC={_fmt_tier(r['tier1'])}")
        print(f"  [{center}] Tier-2 AUROC={_fmt_tier(r['tier2'])}")
        print(f"  [{center}] ALL    AUROC={_fmt_tier(r['all'])}")

    # Main-centers aggregate
    def avg(key_tier, key_metric):
        vals = [results[c][key_tier][key_metric] for c in MAIN_CENTERS if c in results]
        return float(np.mean(vals)) if vals else float('nan')

    main_agg = {
        'tier1': {
            'macro_auroc_avg': avg('tier1', 'macro_auroc'),
            'macro_auprc_avg': avg('tier1', 'macro_auprc'),
        },
        'tier2': {
            'macro_auroc_avg': avg('tier2', 'macro_auroc'),
            'macro_auprc_avg': avg('tier2', 'macro_auprc'),
        },
        'all': {
            'macro_auroc_avg': avg('all', 'macro_auroc'),
            'macro_auprc_avg': avg('all', 'macro_auprc'),
        },
    }

    print("\n" + "=" * 90)
    print("Summary: PhysioNet 2021 cross-center zero-shot")
    print("=" * 90)
    print(f"{'Center':<22} {'N':>7} | "
          f"{'T1 AUROC':>10} {'T1 AUPRC':>10} | "
          f"{'T2 AUROC':>10} {'T2 AUPRC':>10} | "
          f"{'ALL AUROC':>10} {'ALL AUPRC':>10}")
    print("-" * 90)
    for c in ALL_CENTERS:
        if c not in results:
            continue
        r = results[c]
        tag = '' if c in MAIN_CENTERS else ' (small)'
        print(f"{c + tag:<22} {r['n_records']:>7} | "
              f"{r['tier1']['macro_auroc']:>10.4f} {r['tier1']['macro_auprc']:>10.4f} | "
              f"{r['tier2']['macro_auroc']:>10.4f} {r['tier2']['macro_auprc']:>10.4f} | "
              f"{r['all']['macro_auroc']:>10.4f} {r['all']['macro_auprc']:>10.4f}")
    print("-" * 90)
    print(f"{'MAIN avg (5 centers)':<22} {'':>7} | "
          f"{main_agg['tier1']['macro_auroc_avg']:>10.4f} {main_agg['tier1']['macro_auprc_avg']:>10.4f} | "
          f"{main_agg['tier2']['macro_auroc_avg']:>10.4f} {main_agg['tier2']['macro_auprc_avg']:>10.4f} | "
          f"{main_agg['all']['macro_auroc_avg']:>10.4f} {main_agg['all']['macro_auprc_avg']:>10.4f}")
    print("=" * 90)

    output = {
        'centers': results,
        'main_centers_avg': main_agg,
        'main_centers': MAIN_CENTERS,
        'small_centers': SMALL_CENTERS,
        'config': vars(args),
    }
    output_path = os.path.join(args.model_dir, 'eval_crosscenter.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='/root/autodl-tmp/crosscenter_v2')
    p.add_argument('--data_dir',  default='datasets/physionet2021')
    p.add_argument('--device',    default='cuda')
    p.add_argument('--crop_len',  type=int, default=250)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_bootstrap', type=int, default=1000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
