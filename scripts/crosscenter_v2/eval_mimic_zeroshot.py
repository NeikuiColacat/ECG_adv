"""
MIMIC-IV ECG 800k zero-shot evaluation.

Streams WFDB records on-the-fly (no caching), labels from keyword-extracted
machine_measurements.csv reports. Uses the unified preprocessing pipeline so
the MIMIC evaluation is apples-to-apples with PN2021/PTBXL.

Note: MIMIC keyword-label is intrinsically noisy (regex over free-text reports).
We report it as a second, noisier OOD reference alongside PN2021.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter/eval_mimic_zeroshot.py \
        --model_dir /root/autodl-tmp/crosscenter_v2 \
        --data_dir  /root/ECG_adv_Gen/datasets/MIMIC
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
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    NUM_CLASSES_26, CLASS_NAMES_26, TIER1_IDX, TIER2_IDX, ALL_IDX,
    mimic_report_to_26, mimic_row_to_report_text,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000, crop_signal_tc
from EfficientNetv2 import EfficientNet1DV2


class MIMICStreamDataset(Dataset):
    """Streams WFDB records from disk. Labels precomputed from machine report text.

    Returns (12, crop_len) tensor + (26,) label vector. None-result records
    (read/preprocess failure) are returned as zeros with an -1 marker in label
    position [0] (this is safe because label[0] is IAVB and would normally be
    0/1 but we'll drop these records via the valid_mask).
    """

    def __init__(self, paths, labels, crop_len=250):
        self.paths = paths
        self.labels = labels
        self.crop_len = crop_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            rec = wfdb.rdrecord(self.paths[idx])
        except Exception:
            return self._fail(idx)
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            return self._fail(idx)
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
            return self._fail(idx)
        crop = crop_signal_tc(proc, self.crop_len, mode='center')
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels[idx]).float(),
                torch.tensor(1, dtype=torch.int32))  # valid=1

    def _fail(self, idx):
        sig_ct = np.zeros((12, self.crop_len), dtype=np.float32)
        label = self.labels[idx]
        return (torch.from_numpy(sig_ct).float(),
                torch.from_numpy(label).float(),
                torch.tensor(0, dtype=torch.int32))


def build_labels_and_paths(data_dir, limit=None):
    """Load MIMIC metadata, build (path, label_vec) list."""
    mm_csv = os.path.join(data_dir, 'machine_measurements.csv')
    rec_csv = os.path.join(data_dir, 'record_list.csv')
    print(f"Loading {rec_csv}...")
    rec_df = pd.read_csv(rec_csv, usecols=['study_id', 'path'])
    print(f"Loading {mm_csv}... (this may take a moment)")
    mm_df = pd.read_csv(mm_csv)

    # Build report text per study
    print(f"Extracting report text for {len(mm_df)} studies...")
    report_cols = [c for c in mm_df.columns if c.startswith('report_')]
    texts = []
    for _, row in mm_df[report_cols].iterrows():
        parts = [str(v).strip() for v in row.values if isinstance(v, str) and v.strip()]
        texts.append(' | '.join(parts))
    mm_df = mm_df[['study_id']].copy()
    mm_df['report_text'] = texts

    # Merge with record paths
    merged = mm_df.merge(rec_df, on='study_id', how='inner')
    if limit is not None and limit < len(merged):
        merged = merged.iloc[:limit].reset_index(drop=True)

    print(f"Total records to evaluate: {len(merged)}")
    # Build label matrix
    print(f"Computing keyword labels...")
    t0 = time.time()
    labels = np.stack([mimic_report_to_26(t) for t in merged.report_text.values])
    print(f"  done ({time.time()-t0:.0f}s)")

    paths = [os.path.join(data_dir, p) for p in merged.path.values]
    return paths, labels


@torch.no_grad()
def stream_infer(model, loader, device, n_records):
    model.eval()
    y_true_all = np.zeros((n_records, NUM_CLASSES_26), dtype=np.float32)
    y_score_all = np.zeros((n_records, NUM_CLASSES_26), dtype=np.float32)
    valid_all = np.zeros(n_records, dtype=np.int32)

    pos = 0
    t0 = time.time()
    for bi, (signals, labels, valid) in enumerate(loader):
        bs = signals.size(0)
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_score_all[pos:pos+bs] = probs
        y_true_all[pos:pos+bs]  = labels.numpy()
        valid_all[pos:pos+bs]   = valid.numpy()
        pos += bs
        if (bi + 1) % 50 == 0:
            dt = time.time() - t0
            rate = pos / max(dt, 0.1)
            eta = (n_records - pos) / max(rate, 0.1)
            print(f"  batch {bi+1} | {pos}/{n_records} records | "
                  f"{rate:.0f} rec/s | ETA {eta/60:.1f} min")
    return y_true_all[:pos], y_score_all[:pos], valid_all[:pos]


def _class_metrics(t, s):
    if len(t) == 0 or len(np.unique(t)) < 2:
        return None, None
    return float(roc_auc_score(t, s)), float(average_precision_score(t, s))


def compute_tiered(y_true, y_score, indices):
    aurocs, auprcs = [], []
    for i in indices:
        col_t = y_true[:, i]
        valid = col_t != -1.0
        auc, ap = _class_metrics(col_t[valid], y_score[valid, i])
        if auc is not None:
            aurocs.append(auc); auprcs.append(ap)
    return {
        'macro_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'macro_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'n_classes_used': len(aurocs),
    }


def bootstrap_tiered(y_true, y_score, indices, n_bootstrap=1000, seed=42):
    point = compute_tiered(y_true, y_score, indices)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    b_auroc, b_auprc = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        m = compute_tiered(y_true[idx], y_score[idx], indices)
        if not np.isnan(m['macro_auroc']):
            b_auroc.append(m['macro_auroc'])
            b_auprc.append(m['macro_auprc'])
    if b_auroc:
        point.update({
            'auroc_ci_low':  float(np.percentile(b_auroc, 2.5)),
            'auroc_ci_high': float(np.percentile(b_auroc, 97.5)),
            'auprc_ci_low':  float(np.percentile(b_auprc, 2.5)),
            'auprc_ci_high': float(np.percentile(b_auprc, 97.5)),
        })
    return point


def main(args):
    device = torch.device(args.device)
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

    paths, labels = build_labels_and_paths(args.data_dir, limit=args.limit)
    print(f"Keyword label stats (Tier-1):")
    for i, name in zip(TIER1_IDX, ['IAVB','AF','AFL','BBB','Brady','LBBB','RBBB']
                       if False else [CLASS_NAMES_26[j] for j in TIER1_IDX]):
        pos = int((labels[:, i] == 1.0).sum())
        print(f"  {name:10s}: {pos} positives ({100*pos/len(labels):.3f}%)")

    ds = MIMICStreamDataset(paths, labels, crop_len=args.crop_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"\nStreaming inference over {len(ds)} records...")
    t0 = time.time()
    y_true, y_score, valid = stream_infer(model, loader, device, len(ds))
    dt = time.time() - t0
    print(f"Inference done in {dt/60:.1f} min. Valid: {int(valid.sum())}/{len(valid)}")

    # Filter to valid records
    m = valid == 1
    y_true, y_score = y_true[m], y_score[m]
    print(f"After filter: {len(y_true)} records")

    t1 = bootstrap_tiered(y_true, y_score, TIER1_IDX, n_bootstrap=args.n_bootstrap)
    t2 = bootstrap_tiered(y_true, y_score, TIER2_IDX, n_bootstrap=args.n_bootstrap)
    ta = bootstrap_tiered(y_true, y_score, ALL_IDX,   n_bootstrap=args.n_bootstrap)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES_26):
        col_t = y_true[:, i]
        v = col_t != -1.0
        t, s = col_t[v], y_score[v, i]
        auc, ap = _class_metrics(t, s)
        per_class[name] = {
            'auroc': round(auc, 4) if auc is not None else None,
            'auprc': round(ap, 4) if ap is not None else None,
            'n_pos': int((t == 1.0).sum()),
        }

    result = {
        'n_records': int(len(y_true)),
        'tier1': t1, 'tier2': t2, 'all': ta,
        'per_class': per_class,
        'inference_minutes': round(dt / 60, 2),
        'config': vars(args),
    }

    def fmt_t(t):
        if t.get('auroc_ci_low') is None:
            return f"{t['macro_auroc']:.4f} / AUPRC {t['macro_auprc']:.4f}"
        return (f"{t['macro_auroc']:.4f} ({t['auroc_ci_low']:.4f}-{t['auroc_ci_high']:.4f}) / "
                f"AUPRC {t['macro_auprc']:.4f} ({t['auprc_ci_low']:.4f}-{t['auprc_ci_high']:.4f})")

    print("\n" + "=" * 70)
    print(f"MIMIC 800k zero-shot — N={len(y_true)}")
    print("=" * 70)
    print(f"Tier-1 AUROC={fmt_t(t1)}")
    print(f"Tier-2 AUROC={fmt_t(t2)}")
    print(f"ALL    AUROC={fmt_t(ta)}")

    output_path = os.path.join(args.model_dir, 'eval_mimic.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='/root/autodl-tmp/crosscenter_v2')
    p.add_argument('--data_dir',  default='/root/ECG_adv_Gen/datasets/MIMIC')
    p.add_argument('--device',    default='cuda')
    p.add_argument('--crop_len',  type=int, default=250)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--n_bootstrap', type=int, default=1000)
    p.add_argument('--limit', type=int, default=None,
                   help='optional cap for debug (e.g., 10000)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
