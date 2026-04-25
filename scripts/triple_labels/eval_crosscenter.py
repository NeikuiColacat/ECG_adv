"""Cross-center evaluator for the 3 label schemes.

Evaluates a trained model on:
  1. PTB-XL fold 10 test split (in-domain)
  2. PN2021 7 centers, ptb-xl shard hard-excluded
  3. MIMIC-IV ECG test split (split=2 from build_mimic_cache patient-level partition)

Per-class AUROC/AUPRC with -1 masking, plus macro across classes that have
>= min_pos positives.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
        --scheme super5 --model_dir /root/autodl-tmp/triple_labels/super5
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'DeepECG', 'notebooks'))

from scripts.triple_labels.label_schemes import get_scheme
from scripts.triple_labels.train_ptbxl import (
    PTBXLDatasetScheme, compute_macro_auroc_auprc, masked_bce_with_logits,
    get_ptbxl_labels_for_scheme,
)
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, crop_signal_tc,
)
from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


# PN2021 centers — ptb-xl explicitly excluded (data leakage with PTB-XL train)
PN2021_CENTERS = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra',
                  'georgia', 'ningbo', 'ptb', 'st_petersburg_incart']
PN2021_FORBIDDEN = {'ptb-xl', 'ptbxl'}


# ────────────────────────────────────────────────────────────────────────────
# PN2021 evaluation
# ────────────────────────────────────────────────────────────────────────────

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
    paths, snomeds = [], []
    for root, _, files in os.walk(center_dir):
        for f in files:
            if f.endswith('.hea'):
                hea = os.path.join(root, f)
                paths.append(hea[:-4])
                snomeds.append(parse_header_snomed(hea))
    return paths, snomeds


class PN2021CenterDataset(Dataset):
    def __init__(self, record_paths, labels, crop_len=250):
        self.signals = []
        self.kept_indices = []
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
            sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
            proc = unified_preprocess_to_1000(
                sig.astype(np.float32), fs=rec.fs, source_leads=sig_names,
                target_fs=100, target_len=1000,
                apply_filter=True, apply_zscore=True,
            )
            if proc is None:
                fail += 1
                continue
            self.signals.append(proc)
            self.kept_indices.append(i)
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = crop_len
        self._fail = fail
        self._load_time = time.time() - t0

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        label = self.labels[self.kept_indices[idx]]
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(label).float())


@torch.no_grad()
def infer_dataset(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        with torch.cuda.amp.autocast(enabled='cuda' in str(device)):
            logits = model(signals)
        all_labels.append(labels.numpy())
        all_logits.append(logits.float().cpu().numpy())
    y_true = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    y_score = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return y_true, y_score


def eval_pn2021(model, scheme, args, device):
    print(f"\n[pn2021] evaluating {len(PN2021_CENTERS)} centers (ptb-xl excluded)")
    pn2021_root = args.pn2021_root
    per_center = {}
    macro_aurocs, macro_auprcs = [], []
    for center in PN2021_CENTERS:
        assert center.lower() not in PN2021_FORBIDDEN, \
            f"FORBIDDEN center {center} would leak PTB-XL data"
        center_dir = os.path.join(pn2021_root, 'training', center)
        if not os.path.isdir(center_dir):
            print(f"  {center}: missing dir, skipped")
            continue
        t0 = time.time()
        paths, snomeds = scan_center_records(center_dir)
        if args.pn2021_limit and args.pn2021_limit < len(paths):
            paths = paths[:args.pn2021_limit]
            snomeds = snomeds[:args.pn2021_limit]
        labels = np.stack([scheme['pn2021_fn'](s) for s in snomeds])
        ds = PN2021CenterDataset(paths, labels, crop_len=args.crop_len)
        if len(ds) == 0:
            print(f"  {center}: no valid records")
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        y_true, y_score = infer_dataset(model, loader, device)
        m = compute_macro_auroc_auprc(y_true, y_score, scheme['class_names'],
                                      min_pos=args.min_pos)
        per_center[center] = {
            'n_records': len(ds),
            'n_scanned': len(paths),
            'load_time_s': round(ds._load_time, 1),
            'macro_auroc': m['macro_auroc'],
            'macro_auprc': m['macro_auprc'],
            'n_classes_used': m['n_classes_used'],
            'per_class': m['per_class'],
        }
        macro_aurocs.append(m['macro_auroc'])
        macro_auprcs.append(m['macro_auprc'])
        print(f"  {center:<22} n={len(ds):>5}  "
              f"AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
              f"n_classes={m['n_classes_used']}  ({time.time()-t0:.0f}s)")
    avg_auroc = float(np.nanmean(macro_aurocs)) if macro_aurocs else float('nan')
    avg_auprc = float(np.nanmean(macro_auprcs)) if macro_auprcs else float('nan')
    print(f"  ──── 7-center average AUROC={avg_auroc:.4f}  AUPRC={avg_auprc:.4f}")
    return {
        'avg_macro_auroc': avg_auroc,
        'avg_macro_auprc': avg_auprc,
        'per_center': per_center,
    }


# ────────────────────────────────────────────────────────────────────────────
# MIMIC evaluation (test split = 2 from build_mimic_cache)
# ────────────────────────────────────────────────────────────────────────────

def build_mimic_test_index(scheme):
    """Reproduces build_mimic_cache.py merge order to get scheme-specific labels.

    Returns (cache_indices, labels) where cache_indices is the row indices into
    mimic_preprocessed_f16.npy for the test split.
    """
    print("[mimic] reproducing record_list × machine_measurements merge")
    rec = pd.read_csv('/root/autodl-tmp/MIMIC/record_list.csv',
                      usecols=['subject_id', 'study_id', 'path'])
    report_cols = [f'report_{i}' for i in range(18)]
    mm = pd.read_csv('/root/autodl-tmp/MIMIC/machine_measurements.csv',
                     usecols=['study_id'] + report_cols, low_memory=False)

    def _join(row):
        parts = [str(v).strip() for v in row.values
                 if isinstance(v, str) and v.strip()]
        return ' | '.join(parts)
    mm['report_text'] = mm[report_cols].apply(_join, axis=1)
    mm = mm[['study_id', 'report_text']]
    merged = rec.merge(mm, on='study_id', how='inner')
    print(f"[mimic] merged rows: {len(merged)}")

    cache = np.load('/root/autodl-tmp/mimic_tierM/mimic_index.npz', allow_pickle=True)
    valid_mask = cache['valid_mask']
    split = cache['split']
    test_mask = (split == 2) & valid_mask
    assert len(merged) == len(valid_mask), \
        f"row count mismatch: merged={len(merged)} cache={len(valid_mask)}"

    test_indices = np.nonzero(test_mask)[0]
    print(f"[mimic] test split: {len(test_indices)} valid records")

    print(f"[mimic] generating scheme labels for test records")
    labels = np.stack([scheme['mimic_fn'](merged.report_text.iloc[i])
                       for i in test_indices])
    return test_indices, labels


class MIMICCacheDataset(Dataset):
    """Reads preprocessed signals from f16 mmap for given indices."""

    def __init__(self, cache_indices, labels, crop_len=250):
        self.cache = np.load('/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy',
                             mmap_mode='r')
        self.indices = cache_indices
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = crop_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sig_tc = self.cache[self.indices[i]].astype(np.float32)  # (1000, 12) f16 → f32
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels[i]).float())


def eval_mimic(model, scheme, args, device):
    print("\n[mimic] evaluating test split")
    cache_idx, labels = build_mimic_test_index(scheme)
    if args.mimic_limit and args.mimic_limit < len(cache_idx):
        cache_idx = cache_idx[:args.mimic_limit]
        labels = labels[:args.mimic_limit]
        print(f"[mimic] limited to first {len(cache_idx)} records")
    ds = MIMICCacheDataset(cache_idx, labels, crop_len=args.crop_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    t0 = time.time()
    y_true, y_score = infer_dataset(model, loader, device)
    print(f"[mimic] inference done ({time.time()-t0:.0f}s)")
    m = compute_macro_auroc_auprc(y_true, y_score, scheme['class_names'],
                                  min_pos=args.min_pos)
    print(f"[mimic] AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
          f"n_classes={m['n_classes_used']}/{scheme['num_classes']}")
    return {
        'n_records': int(len(ds)),
        'macro_auroc': m['macro_auroc'],
        'macro_auprc': m['macro_auprc'],
        'n_classes_used': m['n_classes_used'],
        'per_class': m['per_class'],
    }


# ────────────────────────────────────────────────────────────────────────────
# PTB-XL fold 10 evaluation
# ────────────────────────────────────────────────────────────────────────────

def eval_ptbxl_test(model, scheme, args, device):
    print("\n[ptbxl] evaluating fold 10 test split")
    label_cache = os.path.join(args.model_dir, 'ptbxl_labels')
    test_idx, test_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=[10]
    )
    cache_path = args.ptbxl_cache or '/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy'
    all_sig = np.load(cache_path, mmap_mode='r')
    test_signals = np.asarray(all_sig[test_idx])
    ds = PTBXLDatasetScheme(test_signals, test_labels,
                            crop_len=args.crop_len, mode='eval')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    y_true, y_score = infer_dataset(model, loader, device)
    m = compute_macro_auroc_auprc(y_true, y_score, scheme['class_names'],
                                  min_pos=args.min_pos)
    print(f"[ptbxl] AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
          f"n_classes={m['n_classes_used']}/{scheme['num_classes']}")
    return {
        'n_records': int(len(ds)),
        'macro_auroc': m['macro_auroc'],
        'macro_auprc': m['macro_auprc'],
        'n_classes_used': m['n_classes_used'],
        'per_class': m['per_class'],
    }


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', required=True, choices=['super5', 'sub23', 'pn26'])
    p.add_argument('--model_dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--min_pos', type=int, default=10,
                   help='Skip class metrics if n_pos < min_pos')
    p.add_argument('--ptbxl_csv', default='/root/autodl-tmp/ptbxl/ptbxl_database.csv')
    p.add_argument('--ptbxl_cache', default=None)
    p.add_argument('--pn2021_root', default='/root/autodl-tmp/physionet2021')
    p.add_argument('--pn2021_limit', type=int, default=None,
                   help='Cap records per center (for smoke test)')
    p.add_argument('--mimic_limit', type=int, default=None,
                   help='Cap MIMIC test records (for smoke test)')
    p.add_argument('--skip_pn2021', action='store_true')
    p.add_argument('--skip_mimic', action='store_true')
    p.add_argument('--output_path', default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    scheme = get_scheme(args.scheme)
    print(f"[scheme] {args.scheme}  num_classes={scheme['num_classes']}")

    model = EfficientNet1DV2(
        variant='s_v2', input_channels=12, num_classes=scheme['num_classes'],
        activation='leaky_relu', stochastic_depth_prob=0.304,
        dropout_rate=0.0, use_se=True, norm_type='batch',
    ).to(device)
    ckpt = os.path.join(args.model_dir, 'best_model.pt')
    sd = torch.load(ckpt, map_location=device)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {ckpt}")

    output = {
        'scheme': args.scheme,
        'num_classes': scheme['num_classes'],
        'class_names': list(scheme['class_names']),
        'config': vars(args),
    }

    output['ptbxl_test'] = eval_ptbxl_test(model, scheme, args, device)
    if not args.skip_pn2021:
        output['pn2021'] = eval_pn2021(model, scheme, args, device)
    if not args.skip_mimic:
        output['mimic_test'] = eval_mimic(model, scheme, args, device)

    out_path = args.output_path or os.path.join(args.model_dir, 'eval_result.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] saved {out_path}")
    print(f"\n  PTB-XL test  AUROC={output['ptbxl_test']['macro_auroc']:.4f}  "
          f"AUPRC={output['ptbxl_test']['macro_auprc']:.4f}")
    if 'pn2021' in output:
        print(f"  PN2021 avg   AUROC={output['pn2021']['avg_macro_auroc']:.4f}  "
              f"AUPRC={output['pn2021']['avg_macro_auprc']:.4f}")
    if 'mimic_test' in output:
        print(f"  MIMIC test   AUROC={output['mimic_test']['macro_auroc']:.4f}  "
              f"AUPRC={output['mimic_test']['macro_auprc']:.4f}")


if __name__ == '__main__':
    main()
