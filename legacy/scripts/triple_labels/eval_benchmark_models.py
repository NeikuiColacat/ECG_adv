"""Cross-center evaluation of PTB-XL benchmark pretrained models on PN2021 + MIMIC.

Tests `fastai_resnet1d_wang` (exp1.1.1, 5-class superdiagnostic) head-to-head
against our super5 model.

Benchmark preprocessing:
  - Raw signal at 100 Hz (no bandpass/notch/baseline filter)
  - 12-lead canonical order
  - Global StandardScaler fit on PTB-XL train (mean=-0.000825, std=0.232)
  - Chunkify inference: 250-sample windows with stride 125, max-aggregate per class

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_benchmark_models.py \
        --ptbxl --pn2021 --mimic
"""

import os
import sys
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wfdb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'ecg_ptbxl_benchmarking', 'code'))

from scripts.triple_labels.label_schemes import get_scheme
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, reorder_leads_tc, resample_tc,
    pad_or_truncate_tc,
)
from models.resnet1d import resnet1d_wang  # noqa: E402

PN2021_CENTERS = ['chapman_shaoxing', 'cpsc_2018', 'cpsc_2018_extra',
                  'georgia', 'ningbo', 'ptb', 'st_petersburg_incart']
PN2021_FORBIDDEN = {'ptb-xl', 'ptbxl'}

BENCH_ROOT = '/root/ECG_adv_Gen/model/ecg_ptbxl_benchmarking/output/exp1.1.1'
SCALER_PATH = f'{BENCH_ROOT}/data/standard_scaler.pkl'
MLB_PATH = f'{BENCH_ROOT}/data/mlb.pkl'
WEIGHT_PATH = f'{BENCH_ROOT}/models/fastai_resnet1d_wang/models/fastai_resnet1d_wang.pth'

CHUNK_LEN = 250
CHUNK_STRIDE = 125


# ────────────────────────────────────────────────────────────────────────────
# Model loader + chunkify inference
# ────────────────────────────────────────────────────────────────────────────

def load_benchmark_model(num_classes=5, device='cuda'):
    model = resnet1d_wang(
        num_classes=num_classes, input_channels=12,
        lin_ftrs_head=[128], ps_head=0.5, kernel_size=5,
    )
    ckpt = torch.load(WEIGHT_PATH, map_location='cpu')
    sd = ckpt['model']
    res = model.load_state_dict(sd, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"weight mismatch: {res}")
    return model.to(device).eval()


def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        ss = pickle.load(f)
    mean, std = float(ss.mean_[0]), float(ss.scale_[0])
    print(f"[scaler] mean={mean:.6f}  std={std:.6f}")
    return mean, std


def benchmark_preprocess(signal_tc, fs, source_leads, scaler_mean, scaler_std):
    """Apply benchmark's preprocessing: reorder + resample → 100Hz +
    pad/truncate → 1000 samples + global standardize.

    Returns: (1000, 12) float32 or None.
    """
    sig = np.asarray(signal_tc, dtype=np.float32)
    if np.isnan(sig).any():
        sig = np.nan_to_num(sig, nan=0.0)
    if source_leads is not None:
        reordered = reorder_leads_tc(sig, source_leads)
        if reordered is None:
            return None
        sig = reordered
    if sig.shape[1] != 12:
        return None
    sig = resample_tc(sig, fs, 100)
    sig = pad_or_truncate_tc(sig, 1000)
    sig = (sig - scaler_mean) / scaler_std
    if not np.isfinite(sig).all():
        return None
    return sig.astype(np.float32)


@torch.no_grad()
def infer_chunkify(model, signal_1000_12, device, chunk_len=CHUNK_LEN,
                   stride=CHUNK_STRIDE):
    """Run model on (1000, 12) signal via chunkify + max aggregate per class.

    Returns sigmoid probabilities (num_classes,).
    """
    T = signal_1000_12.shape[0]
    starts = list(range(0, T - chunk_len + 1, stride))
    if starts[-1] != T - chunk_len:
        starts.append(T - chunk_len)
    chunks = np.stack(
        [signal_1000_12[s:s + chunk_len].T for s in starts]
    )                                                     # (n_chunks, 12, chunk_len)
    x = torch.from_numpy(chunks).float().to(device)
    logits = model(x)                                     # (n_chunks, C)
    probs = torch.sigmoid(logits)                         # (n_chunks, C)
    return probs.max(dim=0).values.cpu().numpy()          # (C,)


@torch.no_grad()
def infer_chunkify_batch(model, signals_1000_12, device, chunk_len=CHUNK_LEN,
                         stride=CHUNK_STRIDE, batch_size=128):
    """Vectorised version: process N records at once. Returns (N, C) probs."""
    if signals_1000_12.ndim == 2:
        signals_1000_12 = signals_1000_12[None]
    N, T, C = signals_1000_12.shape
    starts = list(range(0, T - chunk_len + 1, stride))
    if starts[-1] != T - chunk_len:
        starts.append(T - chunk_len)
    n_chunks = len(starts)
    # Build (N*n_chunks, 12, chunk_len)
    chunks = np.stack(
        [signals_1000_12[:, s:s + chunk_len, :].transpose(0, 2, 1)
         for s in starts], axis=1
    )                                                     # (N, n_chunks, 12, chunk_len)
    chunks = chunks.reshape(N * n_chunks, 12, chunk_len)
    out = np.empty((N * n_chunks, model[-1][-1].out_features), dtype=np.float32)
    for i in range(0, N * n_chunks, batch_size):
        x = torch.from_numpy(chunks[i:i + batch_size]).float().to(device)
        out[i:i + batch_size] = torch.sigmoid(model(x)).cpu().numpy()
    out = out.reshape(N, n_chunks, -1)
    return out.max(axis=1)                                # (N, C)


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────

def compute_macro(y_true, y_score, class_names, min_pos=10):
    per_class = {}
    aurocs, auprcs = [], []
    for k, name in enumerate(class_names):
        yt, ys = y_true[:, k], y_score[:, k]
        valid = yt >= 0
        if not valid.any():
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': 0}
            continue
        yt_v = yt[valid]
        ys_v = ys[valid]
        n_pos = int((yt_v == 1).sum())
        if n_pos < min_pos or n_pos == len(yt_v):
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': n_pos}
            continue
        try:
            au = roc_auc_score(yt_v, ys_v)
            ap = average_precision_score(yt_v, ys_v)
            per_class[name] = {'auroc': float(au), 'auprc': float(ap),
                               'n_pos': n_pos}
            aurocs.append(au)
            auprcs.append(ap)
        except Exception:
            per_class[name] = {'auroc': None, 'auprc': None, 'n_pos': n_pos}
    return {
        'macro_auroc': float(np.nanmean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.nanmean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(aurocs),
        'per_class': per_class,
    }


# ────────────────────────────────────────────────────────────────────────────
# PTB-XL fold 10 (sanity)
# ────────────────────────────────────────────────────────────────────────────

def eval_ptbxl(model, scheme, scaler_mean, scaler_std, device, batch_size=64):
    print("\n[ptbxl] fold 10 in-domain sanity")
    raw = np.load('/root/autodl-tmp/ptbxl/raw100.npy', allow_pickle=True)
    raw = raw.astype(np.float32)                                # (N, 1000, 12)
    df = pd.read_csv('/root/autodl-tmp/ptbxl/ptbxl_database.csv', index_col='ecg_id')
    test_mask = (df.strat_fold == 10).values
    print(f"[ptbxl] loaded raw100.npy {raw.shape}, fold10 n={test_mask.sum()}")

    # Standardize all PTB-XL test in one go (memory ok: 2198 * 1000 * 12 * 4 ≈ 100 MB)
    sigs = (raw[test_mask] - scaler_mean) / scaler_std
    # Build labels with super5 scheme
    import ast
    scp_codes_list = df.loc[test_mask, 'scp_codes'].apply(ast.literal_eval).values
    labels = np.stack([scheme['ptbxl_fn'](s) for s in scp_codes_list])
    print(f"[ptbxl] running chunkify inference (batch_size={batch_size})")
    t0 = time.time()
    all_probs = []
    for i in range(0, len(sigs), batch_size):
        probs = infer_chunkify_batch(model, sigs[i:i + batch_size], device)
        all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    print(f"[ptbxl] inference {time.time()-t0:.1f}s")
    m = compute_macro(labels, probs, scheme['class_names'])
    print(f"[ptbxl] AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
          f"n_classes={m['n_classes_used']}")
    for name in scheme['class_names']:
        pc = m['per_class'][name]
        if pc['auroc'] is not None:
            print(f"  {name:6} auroc={pc['auroc']:.3f} auprc={pc['auprc']:.3f} n={pc['n_pos']}")
    return m


# ────────────────────────────────────────────────────────────────────────────
# PN2021 (7 centers, ptb-xl excluded)
# ────────────────────────────────────────────────────────────────────────────

def parse_header_snomed(hea_path):
    """PhysioNet/CinC 2021 hea uses '# Dx: <codes>' (note the space).
    Match both '#Dx:' and '# Dx:' for robustness."""
    with open(hea_path) as f:
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


def eval_pn2021(model, scheme, scaler_mean, scaler_std, device,
                pn2021_root='/root/autodl-tmp/physionet2021', batch_size=64,
                center_limit=None):
    print(f"\n[pn2021] evaluating {len(PN2021_CENTERS)} centers (ptb-xl excluded)")
    per_center = {}
    macro_aurocs, macro_auprcs = [], []
    for center in PN2021_CENTERS:
        assert center.lower() not in PN2021_FORBIDDEN, \
            f"FORBIDDEN center {center}"
        center_dir = os.path.join(pn2021_root, 'training', center)
        if not os.path.isdir(center_dir):
            print(f"  {center}: missing dir, skipped")
            continue
        t0 = time.time()
        paths, snomeds = scan_center_records(center_dir)
        if center_limit and center_limit < len(paths):
            paths = paths[:center_limit]
            snomeds = snomeds[:center_limit]
        labels = np.stack([scheme['pn2021_fn'](s) for s in snomeds])
        signals_kept = []
        labels_kept = []
        fail = 0
        for i, p in enumerate(paths):
            try:
                rec = wfdb.rdrecord(p)
                sig = rec.p_signal
                names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
                proc = benchmark_preprocess(sig, rec.fs, names,
                                            scaler_mean, scaler_std)
                if proc is None:
                    fail += 1
                    continue
                signals_kept.append(proc)
                labels_kept.append(labels[i])
            except Exception:
                fail += 1
        if not signals_kept:
            print(f"  {center}: no valid records")
            continue
        sigs = np.stack(signals_kept)
        labs = np.stack(labels_kept)
        probs_list = []
        for i in range(0, len(sigs), batch_size):
            probs_list.append(infer_chunkify_batch(model, sigs[i:i + batch_size], device))
        probs = np.concatenate(probs_list, axis=0)
        m = compute_macro(labs, probs, scheme['class_names'])
        per_center[center] = {
            'n_records': len(sigs), 'n_scanned': len(paths),
            'fail_load': fail,
            'macro_auroc': m['macro_auroc'], 'macro_auprc': m['macro_auprc'],
            'n_classes_used': m['n_classes_used'],
            'per_class': m['per_class'],
        }
        macro_aurocs.append(m['macro_auroc'])
        macro_auprcs.append(m['macro_auprc'])
        print(f"  {center:<22} n={len(sigs):>5}  AUROC={m['macro_auroc']:.4f}  "
              f"AUPRC={m['macro_auprc']:.4f}  ({time.time()-t0:.0f}s)")
    avg_auroc = float(np.nanmean(macro_aurocs)) if macro_aurocs else float('nan')
    avg_auprc = float(np.nanmean(macro_auprcs)) if macro_auprcs else float('nan')
    print(f"  ──── 7-center average AUROC={avg_auroc:.4f}  AUPRC={avg_auprc:.4f}")
    return {
        'avg_macro_auroc': avg_auroc, 'avg_macro_auprc': avg_auprc,
        'per_center': per_center,
    }


# ────────────────────────────────────────────────────────────────────────────
# MIMIC test split — use raw recordings (NOT our preprocessed cache, since
# benchmark needs no-filter preprocessing)
# ────────────────────────────────────────────────────────────────────────────

def eval_mimic(model, scheme, scaler_mean, scaler_std, device,
               mimic_root='/root/autodl-tmp/MIMIC', batch_size=64,
               mimic_limit=None):
    print("\n[mimic] evaluating test split")
    rec_df = pd.read_csv(f'{mimic_root}/record_list.csv',
                         usecols=['subject_id', 'study_id', 'path'])
    report_cols = [f'report_{i}' for i in range(18)]
    mm = pd.read_csv(f'{mimic_root}/machine_measurements.csv',
                     usecols=['study_id'] + report_cols, low_memory=False)

    def _join(row):
        parts = [str(v).strip() for v in row.values
                 if isinstance(v, str) and v.strip()]
        return ' | '.join(parts)
    mm['report_text'] = mm[report_cols].apply(_join, axis=1)
    mm = mm[['study_id', 'report_text']]
    merged = rec_df.merge(mm, on='study_id', how='inner')

    cache = np.load('/root/autodl-tmp/mimic_tierM/mimic_index.npz', allow_pickle=True)
    valid_mask = cache['valid_mask']
    split = cache['split']
    test_mask = (split == 2) & valid_mask
    assert len(merged) == len(valid_mask), \
        f"row mismatch: {len(merged)} vs {len(valid_mask)}"
    test_indices = np.nonzero(test_mask)[0]
    if mimic_limit and mimic_limit < len(test_indices):
        test_indices = test_indices[:mimic_limit]
    print(f"[mimic] test split: {len(test_indices)} records "
          f"(limit={mimic_limit})")

    labels = np.stack([scheme['mimic_fn'](merged.report_text.iloc[i])
                       for i in test_indices])

    signals_kept = []
    labels_kept = []
    fail = 0
    t0 = time.time()
    for j, i in enumerate(test_indices):
        rel = merged.path.iloc[i]
        path = os.path.join(mimic_root, rel) if not os.path.isabs(rel) else rel
        try:
            rec = wfdb.rdrecord(path)
            sig = rec.p_signal
            names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
            proc = benchmark_preprocess(sig, rec.fs, names,
                                        scaler_mean, scaler_std)
            if proc is None:
                fail += 1
                continue
            signals_kept.append(proc)
            labels_kept.append(labels[j])
        except Exception:
            fail += 1
        if (j + 1) % 5000 == 0:
            print(f"  ... {j+1}/{len(test_indices)} ({time.time()-t0:.0f}s, "
                  f"fail={fail})")
    sigs = np.stack(signals_kept)
    labs = np.stack(labels_kept)
    print(f"[mimic] preprocessing done {time.time()-t0:.0f}s, n={len(sigs)} "
          f"fail={fail}")

    probs_list = []
    for i in range(0, len(sigs), batch_size):
        probs_list.append(infer_chunkify_batch(model, sigs[i:i + batch_size], device))
    probs = np.concatenate(probs_list, axis=0)
    m = compute_macro(labs, probs, scheme['class_names'])
    print(f"[mimic] AUROC={m['macro_auroc']:.4f}  AUPRC={m['macro_auprc']:.4f}  "
          f"n_classes={m['n_classes_used']}/{scheme['num_classes']}  n={len(sigs)}")
    for name in scheme['class_names']:
        pc = m['per_class'][name]
        if pc['auroc'] is not None:
            print(f"  {name:6} auroc={pc['auroc']:.3f} auprc={pc['auprc']:.3f} n={pc['n_pos']}")
    return {
        'macro_auroc': m['macro_auroc'], 'macro_auprc': m['macro_auprc'],
        'n_classes_used': m['n_classes_used'], 'n_records': len(sigs),
        'per_class': m['per_class'],
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scheme', default='super5')
    p.add_argument('--ptbxl', action='store_true')
    p.add_argument('--pn2021', action='store_true')
    p.add_argument('--mimic', action='store_true')
    p.add_argument('--all', action='store_true')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mimic_limit', type=int, default=0)
    p.add_argument('--center_limit', type=int, default=0)
    p.add_argument('--out_path', default='/root/autodl-tmp/triple_labels/'
                                          'benchmark_resnet1d_wang/eval_result.json')
    args = p.parse_args()
    if args.all:
        args.ptbxl = args.pn2021 = args.mimic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheme = get_scheme(args.scheme)
    print(f"[main] scheme={args.scheme}  classes={scheme['class_names']}")
    model = load_benchmark_model(num_classes=scheme['num_classes'], device=device)
    scaler_mean, scaler_std = load_scaler()

    result = {
        'scheme': args.scheme, 'num_classes': scheme['num_classes'],
        'class_names': list(scheme['class_names']),
        'model_source': 'ptbxl_benchmarking_exp1.1.1_fastai_resnet1d_wang',
    }
    if args.ptbxl:
        result['ptbxl_test'] = eval_ptbxl(
            model, scheme, scaler_mean, scaler_std, device,
            batch_size=args.batch_size,
        )
    if args.pn2021:
        result['pn2021'] = eval_pn2021(
            model, scheme, scaler_mean, scaler_std, device,
            batch_size=args.batch_size,
            center_limit=args.center_limit or None,
        )
    if args.mimic:
        result['mimic_test'] = eval_mimic(
            model, scheme, scaler_mean, scaler_std, device,
            batch_size=args.batch_size,
            mimic_limit=args.mimic_limit or None,
        )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] saved {args.out_path}")


if __name__ == '__main__':
    main()
