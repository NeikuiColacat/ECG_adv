"""Offline MIMIC-IV-ECG cache builder for Tier-M training.

Produces two artifacts in /root/autodl-tmp/mimic_tierM/:
  - mimic_preprocessed_f16.npy: (N, 1000, 12) float16 mmap of preprocessed signals.
  - mimic_index.npz: labels_6, subject_ids, valid_mask, split (0=train, 1=val, 2=test).

Preprocessing is bit-identical to PTBXL Tier-M (unified_preprocess_to_1000 @ 100Hz).
Labels come from regex on concatenated machine report text (mimic_report_to_26),
then sliced to Tier-M 6 classes (NSR, STach, AF, IAVB, LBBB, RBBB).

Patient-level 8:1:1 split on subject_id with seed=42 — records from the same
patient never cross splits.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/build_mimic_cache.py --workers 12
    # resume from interrupted run:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/build_mimic_cache.py --workers 12 --resume
"""

import os
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    NUM_CLASSES_TIER_M, TIER_M, TIER_M_IDX,
    mimic_report_to_26,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000


MIMIC_SRC = '/root/autodl-tmp/MIMIC'
OUT_DIR = '/root/autodl-tmp/mimic_tierM'
TARGET_LEN = 1000
TARGET_FS = 100


def _assert_safe_paths():
    out_abs = os.path.abspath(OUT_DIR)
    src_abs = os.path.abspath(MIMIC_SRC)
    assert out_abs.startswith('/root/autodl-tmp/mimic_tierM'), \
        f"OUT_DIR unsafe: {out_abs}"
    assert out_abs != src_abs and not out_abs.startswith(src_abs + os.sep), \
        f"OUT_DIR must not be inside MIMIC source: {out_abs}"


def preprocess_one(task):
    idx, path = task
    try:
        rec = wfdb.rdrecord(path)
    except Exception:
        return (idx, None)
    sig = rec.p_signal
    if sig is None or sig.shape[1] < 12:
        return (idx, None)
    sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
    proc = unified_preprocess_to_1000(
        sig.astype(np.float32),
        fs=rec.fs,
        source_leads=sig_names,
        target_fs=TARGET_FS,
        target_len=TARGET_LEN,
        apply_filter=True,
        apply_zscore=True,
    )
    if proc is None:
        return (idx, None)
    return (idx, proc.astype(np.float16))


def build_metadata(limit=None):
    rec_path = os.path.join(MIMIC_SRC, 'record_list.csv')
    mm_path = os.path.join(MIMIC_SRC, 'machine_measurements.csv')

    print(f"[meta] loading {rec_path}")
    rec_df = pd.read_csv(rec_path, usecols=['subject_id', 'study_id', 'path'])

    print(f"[meta] loading {mm_path}")
    report_cols = [f'report_{i}' for i in range(18)]
    mm_df = pd.read_csv(mm_path, usecols=['study_id'] + report_cols, low_memory=False)

    print(f"[meta] joining report text ({len(mm_df)} rows)")
    def _join(row):
        parts = [str(v).strip() for v in row.values if isinstance(v, str) and v.strip()]
        return ' | '.join(parts)
    mm_df['report_text'] = mm_df[report_cols].apply(_join, axis=1)
    mm_df = mm_df[['study_id', 'report_text']]

    merged = rec_df.merge(mm_df, on='study_id', how='inner')
    if limit is not None and limit < len(merged):
        merged = merged.iloc[:limit].reset_index(drop=True)
    print(f"[meta] merged records: {len(merged)}")

    print(f"[meta] extracting Tier-M labels via regex")
    t0 = time.time()
    N = len(merged)
    labels_6 = np.zeros((N, NUM_CLASSES_TIER_M), dtype=np.int8)
    texts = merged.report_text.values
    for i in range(N):
        lab26 = mimic_report_to_26(texts[i])
        labels_6[i] = lab26[TIER_M_IDX].astype(np.int8)
        if (i + 1) % 200000 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)")
    assert (labels_6 == -1).sum() == 0, "Tier-M should have no -1 labels"
    print(f"[meta] labels done in {time.time()-t0:.0f}s")

    paths = [os.path.join(MIMIC_SRC, p) for p in merged.path.values]
    subject_ids = merged.subject_id.values.astype(np.int64)
    return paths, labels_6, subject_ids


def patient_split(subject_ids, seed=42, ratios=(0.8, 0.1, 0.1)):
    unique = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    unique = unique[perm]
    n = len(unique)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_pats = unique[:n_train]
    val_pats = unique[n_train:n_train + n_val]
    test_pats = unique[n_train + n_val:]

    pat_to_split = np.full(int(unique.max()) + 1, -1, dtype=np.int8)
    pat_to_split[train_pats] = 0
    pat_to_split[val_pats] = 1
    pat_to_split[test_pats] = 2
    split = pat_to_split[subject_ids]
    assert (split >= 0).all()
    return split, (len(train_pats), len(val_pats), len(test_pats))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=None, help='debug cap on records')
    parser.add_argument('--resume', action='store_true',
                        help='keep existing cache and skip already-valid indices')
    parser.add_argument('--chunksize', type=int, default=32,
                        help='imap_unordered chunksize per worker')
    args = parser.parse_args()

    _assert_safe_paths()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[safe] MIMIC_SRC={MIMIC_SRC} (read-only)")
    print(f"[safe] OUT_DIR={OUT_DIR}")

    paths, labels_6, subject_ids = build_metadata(limit=args.limit)
    N = len(paths)

    print(f"\n[stats] Tier-M positive counts across all {N} records:")
    for i, name in enumerate(TIER_M):
        n_pos = int((labels_6[:, i] == 1).sum())
        print(f"  {name:<6} {n_pos:>8}  ({100*n_pos/N:.2f}%)")

    print(f"\n[split] patient-level 8:1:1 (seed=42)")
    split, (n_tr_p, n_va_p, n_te_p) = patient_split(subject_ids, seed=42)
    n_tr = int((split == 0).sum())
    n_va = int((split == 1).sum())
    n_te = int((split == 2).sum())
    print(f"  patients  train={n_tr_p}  val={n_va_p}  test={n_te_p}")
    print(f"  records   train={n_tr}  val={n_va}  test={n_te}")
    tr_set = set(subject_ids[split == 0].tolist())
    va_set = set(subject_ids[split == 1].tolist())
    te_set = set(subject_ids[split == 2].tolist())
    assert not (tr_set & va_set), "LEAKAGE train ∩ val"
    assert not (tr_set & te_set), "LEAKAGE train ∩ test"
    assert not (va_set & te_set), "LEAKAGE val ∩ test"
    print(f"  leakage check: PASSED")

    cache_path = os.path.join(OUT_DIR, 'mimic_preprocessed_f16.npy')
    partial_mask_path = os.path.join(OUT_DIR, '_partial_valid_mask.npy')
    valid_mask = np.zeros(N, dtype=bool)

    if args.resume and os.path.exists(cache_path):
        existing = np.load(cache_path, mmap_mode='r')
        if existing.shape == (N, TARGET_LEN, 12) and existing.dtype == np.float16:
            print(f"[resume] reusing cache {cache_path}")
            if os.path.exists(partial_mask_path):
                saved = np.load(partial_mask_path)
                if saved.shape == (N,):
                    valid_mask = saved
                    print(f"[resume] {valid_mask.sum()}/{N} records already valid")
            signals = np.load(cache_path, mmap_mode='r+')
        else:
            print(f"[resume] shape/dtype mismatch, rebuilding")
            signals = None
    else:
        signals = None

    if signals is None:
        expected_gb = N * TARGET_LEN * 12 * 2 / 1024 ** 3
        print(f"[alloc] creating {cache_path} shape=({N},{TARGET_LEN},12) dtype=f16 ~{expected_gb:.1f} GB")
        signals = np.lib.format.open_memmap(
            cache_path, mode='w+', dtype=np.float16,
            shape=(N, TARGET_LEN, 12),
        )

    todo = np.nonzero(~valid_mask)[0].tolist()
    print(f"\n[run] preprocessing {len(todo)} records with {args.workers} workers")
    tasks = [(i, paths[i]) for i in todo]
    t0 = time.time()
    snapshot_every = 20000

    with Pool(args.workers) as pool:
        with tqdm(total=len(tasks), unit='rec', smoothing=0.05) as pbar:
            done = 0
            for idx, result in pool.imap_unordered(preprocess_one, tasks, chunksize=args.chunksize):
                if result is not None:
                    signals[idx] = result
                    valid_mask[idx] = True
                done += 1
                pbar.update(1)
                if done % snapshot_every == 0:
                    np.save(partial_mask_path, valid_mask)

    if hasattr(signals, 'flush'):
        signals.flush()
    del signals

    elapsed = time.time() - t0
    print(f"\n[run] preprocessing done in {elapsed/60:.1f} min")

    index_path = os.path.join(OUT_DIR, 'mimic_index.npz')
    print(f"[save] writing {index_path}")
    np.savez(
        index_path,
        labels_6=labels_6,
        subject_ids=subject_ids,
        valid_mask=valid_mask,
        split=split,
        tier_m_names=np.array(TIER_M),
    )

    if os.path.exists(partial_mask_path):
        os.remove(partial_mask_path)

    print(f"\n=== Done ===")
    print(f"  records total:     {N}")
    print(f"  records valid:     {int(valid_mask.sum())}  ({100*valid_mask.sum()/N:.2f}%)")
    print(f"  valid in train:    {int((valid_mask & (split==0)).sum())}")
    print(f"  valid in val:      {int((valid_mask & (split==1)).sum())}")
    print(f"  valid in test:     {int((valid_mask & (split==2)).sum())}")
    print(f"\n  Tier-M pos counts on valid train split:")
    tr_valid = valid_mask & (split == 0)
    for i, name in enumerate(TIER_M):
        n_pos = int(((labels_6[:, i] == 1) & tr_valid).sum())
        print(f"    {name:<6} {n_pos:>7}  ({100*n_pos/max(tr_valid.sum(),1):.2f}%)")


if __name__ == '__main__':
    main()
