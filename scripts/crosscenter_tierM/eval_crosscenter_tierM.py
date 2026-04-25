"""Tier-M 6-class cross-center eval on PhysioNet 2021.

Fork of scripts/crosscenter_v2/eval_crosscenter_v2.py. Differences:
  1. num_classes=6 for model and labels (slice snomed_to_26 output to TIER_M_IDX).
  2. Single Tier-M macro metric (no Tier-1/Tier-2/ALL split).
  3. Per-class table over the 6 Tier-M classes with null cells for (center, class)
     combinations where the center has 0 positives.

The ningbo AF↔AFL dual mapping is applied automatically via label_alignment_v2.py
(which patches _SNOMED_TO_IDX at import time). See
/root/.claude/projects/-root-ECG-adv-Gen/memory/pn2021_labeling_quirks.md.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/eval_crosscenter_tierM.py \
        --model_dir /root/autodl-tmp/crosscenter_tierM
"""

import os
import sys
import json
import argparse
import hashlib
import time

import numpy as np
import wfdb
import torch


def _fold_from_hash(record_id: str) -> int:
    """Deterministic fold 1..10 from record_id (matches prep_center_dataset.py)."""
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    TIER_M, TIER_M_IDX, NUM_CLASSES_TIER_M,
    ALL_SCORED_SNOMED, snomed_to_26, has_any_scored_class,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000, crop_signal_tc
from EfficientNetv2 import EfficientNet1DV2


MAIN_CENTERS = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra', 'georgia', 'ningbo']
SMALL_CENTERS = ['ptb', 'st_petersburg_incart']
ALL_CENTERS = MAIN_CENTERS + SMALL_CENTERS


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


class PN2021CenterDatasetTierM(Dataset):
    """Loads PN2021 records and returns (signal, tier_m_label_6) pairs."""

    def __init__(self, record_paths, labels_6, crop_len=250):
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
        self.labels_6 = labels_6
        self.crop_len = crop_len
        self._fail = fail
        self._load_time = time.time() - t0

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        label = self.labels_6[self.valid_indices[idx]]
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(label).float())


def _class_metrics(t, s):
    if len(t) == 0 or len(np.unique(t)) < 2:
        return None, None
    return float(roc_auc_score(t, s)), float(average_precision_score(t, s))


def compute_macro(y_true, y_score):
    """Macro AUROC/AUPRC over 6 Tier-M classes, skipping classes with 0 positives."""
    aurocs, auprcs = [], []
    for i in range(NUM_CLASSES_TIER_M):
        auc, ap = _class_metrics(y_true[:, i], y_score[:, i])
        if auc is not None:
            aurocs.append(auc)
            auprcs.append(ap)
    return {
        'macro_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(aurocs),
    }


def bootstrap_macro(y_true, y_score, n_bootstrap=1000, seed=42):
    point = compute_macro(y_true, y_score)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_auroc, boot_auprc = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        m = compute_macro(y_true[idx], y_score[idx])
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
    n_scanned = len(paths)
    print(f"{n_scanned} scored records ({time.time()-t0:.0f}s)")
    if not paths:
        return None
    if getattr(args, 'eval_half_only', False):
        keep = [i for i, p in enumerate(paths) if _fold_from_hash(os.path.basename(p)) >= 6]
        paths = [paths[i] for i in keep]
        snomed_lists = [snomed_lists[i] for i in keep]
        print(f"  [{center_name}] eval_half filter: {len(paths)}/{n_scanned} kept (fold>=6)")
        if not paths:
            return None
    labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
    labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)
    ds = PN2021CenterDatasetTierM(paths, labels_6, crop_len=args.crop_len)
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
        'macro': bootstrap_macro(y_true, y_score, n_bootstrap=args.n_bootstrap),
    }

    per_class = {}
    for i, name in enumerate(TIER_M):
        t, s = y_true[:, i], y_score[:, i]
        auc, ap = _class_metrics(t, s)
        per_class[name] = {
            'auroc': round(auc, 4) if auc is not None else None,
            'auprc': round(ap, 4) if ap is not None else None,
            'n_pos': int((t == 1.0).sum()),
        }
    result['per_class'] = per_class
    return result


def _fmt_macro(m):
    if m.get('auroc_ci_low') is None:
        return f"AUROC={m['macro_auroc']:.4f} AUPRC={m['macro_auprc']:.4f}"
    return (f"AUROC={m['macro_auroc']:.4f} ({m['auroc_ci_low']:.4f}-{m['auroc_ci_high']:.4f}) "
            f"AUPRC={m['macro_auprc']:.4f}")


def main(args):
    device = torch.device(args.device)

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
    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model loaded from {ckpt_path}  (Tier-M: {TIER_M})")

    results = {}
    for center in ALL_CENTERS:
        # PTB-XL is the training set; the PN2021 ptb-xl shard is exactly that
        # data and must never enter cross-center eval. (ptb is the older small
        # dataset, allowed.)
        assert center.lower() not in {'ptb-xl', 'ptbxl'}, (
            f"refusing to eval on PN2021 shard {center!r} (PTB-XL leak)"
        )
        center_dir = os.path.join(args.data_dir, 'training', center)
        if not os.path.isdir(center_dir):
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
        print(f"  [{center}] Tier-M {_fmt_macro(r['macro'])}")

    # Main-centers aggregate (average AUROC/AUPRC across the 5 main centers)
    main_macro_auroc = [results[c]['macro']['macro_auroc'] for c in MAIN_CENTERS if c in results]
    main_macro_auprc = [results[c]['macro']['macro_auprc'] for c in MAIN_CENTERS if c in results]
    main_agg = {
        'macro_auroc_avg': float(np.mean(main_macro_auroc)) if main_macro_auroc else float('nan'),
        'macro_auprc_avg': float(np.mean(main_macro_auprc)) if main_macro_auprc else float('nan'),
        'n_centers': len(main_macro_auroc),
    }

    print("\n" + "=" * 110)
    print(f"Summary: Tier-M 6-class ({' / '.join(TIER_M)})  —  PN2021 cross-center zero-shot")
    print("=" * 110)
    header = f"{'Center':<22} {'N':>6} | {'Macro AUROC':>12} {'Macro AUPRC':>12} | "
    header += " ".join(f"{n:>14}" for n in TIER_M)
    print(header)
    print("-" * 110)
    for c in ALL_CENTERS:
        if c not in results:
            continue
        r = results[c]
        tag = '' if c in MAIN_CENTERS else ' (small)'
        pc = r['per_class']
        row = f"{c + tag:<22} {r['n_records']:>6} | " \
              f"{r['macro']['macro_auroc']:>12.4f} {r['macro']['macro_auprc']:>12.4f} | "
        for name in TIER_M:
            auc = pc[name]['auroc']
            n_pos = pc[name]['n_pos']
            cell = f"{auc:.3f}|{n_pos}" if auc is not None else f"N/A|{n_pos}"
            row += f"{cell:>14} "
        print(row)
    print("-" * 110)
    print(f"{'MAIN avg (5 centers)':<22} {'':>6} | "
          f"{main_agg['macro_auroc_avg']:>12.4f} {main_agg['macro_auprc_avg']:>12.4f}")
    print("=" * 110)

    output = {
        'centers': results,
        'main_centers_avg': main_agg,
        'main_centers': MAIN_CENTERS,
        'small_centers': SMALL_CENTERS,
        'tier_m': TIER_M,
        'config': vars(args),
    }
    output_path = args.output_path or os.path.join(args.model_dir, 'eval_crosscenter.json')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='/root/autodl-tmp/crosscenter_tierM')
    p.add_argument('--data_dir',  default='datasets/physionet2021')
    p.add_argument('--device',    default='cuda')
    p.add_argument('--crop_len',  type=int, default=250)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_bootstrap', type=int, default=1000)
    p.add_argument('--eval_half_only', action='store_true',
                   help='Filter records to fold>=6 (eval_half); used for R3 clean ablation to exclude records used as refs.')
    p.add_argument('--output_path', default=None,
                   help='Override output JSON path. Defaults to <model_dir>/eval_crosscenter.json.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
