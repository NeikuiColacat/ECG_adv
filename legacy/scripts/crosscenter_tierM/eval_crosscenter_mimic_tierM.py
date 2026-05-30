"""Cross-dataset evaluation for a MIMIC-trained Tier-M model.

Pairs with train_mimic_tierM.py. Runs the 6-class head against:
  - MIMIC test split           (in-domain)
  - PTBXL fold 10               (zero-shot cross-dataset)
  - PhysioNet 2021 5 centers    (zero-shot cross-center)

Uses the same unified preprocessing as training, so any gap reflects real
domain shift rather than pipeline differences.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_tierM/eval_crosscenter_mimic_tierM.py
"""

import os
import sys
import json
import argparse
import hashlib
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _fold_from_hash(record_id: str) -> int:
    """Deterministic fold 1..10 from record_id (matches eval_crosscenter_tierM.py)."""
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'DeepECG', 'notebooks'))

from scripts.crosscenter_v2.label_alignment_v2 import (
    TIER_M, TIER_M_IDX, NUM_CLASSES_TIER_M,
    snomed_to_26, get_ptbxl_26_labels,
)
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc
from scripts.crosscenter_v2.eval_crosscenter_v2 import (
    MAIN_CENTERS, SMALL_CENTERS, ALL_CENTERS,
    scan_center_records, PN2021CenterDatasetV2,
    compute_tiered, bootstrap_tiered, _class_metrics, infer_dataset,
)
from scripts.crosscenter_tierM.train_mimic_tierM import MIMICTierMDataset
from EfficientNetv2 import EfficientNet1DV2


TIER_M_INDICES = list(range(NUM_CLASSES_TIER_M))  # model outputs 6 classes, so within-model indices are 0..5


class PTBXLCachedDataset(Dataset):
    """Reads preprocessed PTBXL cache (from crosscenter_v2), returns 6-class labels.

    signals: (N_all, 1000, 12) float32 npy / mmap
    labels_26: (N_all, 26) float32 with -1 for masked (PTBXL uncovered classes)
    indices: int array selecting fold-10 test records
    """

    def __init__(self, signals_all, labels_26_all, indices, crop_len=250):
        self.signals = signals_all
        self.labels_6 = labels_26_all[:, TIER_M_IDX].astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = crop_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real = int(self.indices[idx])
        sig_tc = np.asarray(self.signals[real]).astype(np.float32)
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels_6[real]).float())


def _class_wise(y_true, y_score):
    per_class = {}
    for i, name in enumerate(TIER_M):
        col_t = y_true[:, i]
        valid = col_t != -1.0
        t, s = col_t[valid], y_score[valid, i]
        auc, ap = _class_metrics(t, s)
        per_class[name] = {
            'auroc': round(auc, 4) if auc is not None else None,
            'auprc': round(ap, 4) if ap is not None else None,
            'n_pos': int((t == 1.0).sum()),
            'n_eval': int(valid.sum()),
        }
    return per_class


def _pack(y_true, y_score, n_bootstrap, seed=42):
    return {
        'n_records': int(len(y_true)),
        'tier_m': bootstrap_tiered(y_true, y_score, TIER_M_INDICES, n_bootstrap=n_bootstrap, seed=seed),
        'per_class': _class_wise(y_true, y_score),
    }


def eval_mimic_test(model, args, device):
    print("\n[MIMIC test split] (in-domain)")
    idx = np.load(args.mimic_index, allow_pickle=True)
    labels_6 = idx['labels_6'].astype(np.float32)
    valid_mask = idx['valid_mask']
    split = idx['split']
    test_idx = np.nonzero(valid_mask & (split == 2))[0]
    print(f"  valid test records: {len(test_idx)}")

    signals = np.load(args.mimic_cache, mmap_mode='r')
    ds = MIMICTierMDataset(signals, labels_6, test_idx, crop_len=args.crop_len, mode='eval')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    t0 = time.time()
    y_true, y_score = infer_dataset(model, loader, device)
    print(f"  inference done ({(time.time()-t0)/60:.1f} min)")
    return _pack(y_true, y_score, args.n_bootstrap)


def eval_ptbxl_fold10(model, args, device):
    print("\n[PTBXL fold 10] (zero-shot cross-dataset)")
    test_idx_arr, test_labels_26, _ = get_ptbxl_26_labels(args.ptbxl_csv, folds=[10])
    print(f"  fold 10 records with labels: {len(test_idx_arr)}")

    signals_all = np.load(args.ptbxl_cache, mmap_mode='r')
    sub_signals = np.asarray(signals_all[test_idx_arr])
    ds = PTBXLCachedDataset(sub_signals, test_labels_26, np.arange(len(test_idx_arr)),
                            crop_len=args.crop_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(2, args.num_workers // 2), pin_memory=True)
    t0 = time.time()
    y_true, y_score = infer_dataset(model, loader, device)
    print(f"  inference done ({time.time()-t0:.0f}s)")
    return _pack(y_true, y_score, args.n_bootstrap)


def eval_pn2021(model, args, device):
    print("\n[PN2021] (zero-shot cross-center)")
    results = {}
    for center in ALL_CENTERS:
        # PTB-XL is the training set; the PN2021 ptb-xl shard is exactly that
        # data and must never enter cross-center eval. (ptb is the older small
        # dataset, allowed.)
        assert center.lower() not in {'ptb-xl', 'ptbxl'}, (
            f"refusing to eval on PN2021 shard {center!r} (PTB-XL leak)"
        )
        center_dir = os.path.join(args.pn2021_dir, 'training', center)
        if not os.path.isdir(center_dir):
            alt = os.path.join(args.pn2021_dir, center)
            if os.path.isdir(alt):
                center_dir = alt
            else:
                print(f"  [{center}] dir not found, skipping")
                continue

        print(f"  [{center}] scanning...", end=' ', flush=True)
        t0 = time.time()
        paths, snomed_lists = scan_center_records(center_dir)
        n_scanned = len(paths)
        print(f"{n_scanned} scored records ({time.time()-t0:.0f}s)")
        if not paths:
            continue
        if getattr(args, 'eval_half_only', False):
            keep = [i for i, p in enumerate(paths) if _fold_from_hash(os.path.basename(p)) >= 6]
            paths = [paths[i] for i in keep]
            snomed_lists = [snomed_lists[i] for i in keep]
            print(f"  [{center}] eval_half filter: {len(paths)}/{n_scanned} kept (fold>=6)")
            if not paths:
                continue

        labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
        # Reuse existing PN2021 dataset (preprocesses eagerly); it returns 26-d labels.
        ds = PN2021CenterDatasetV2(paths, labels_26, crop_len=args.crop_len)
        print(f"  [{center}] loaded {len(ds)}/{len(paths)} (fail={ds._fail}, {ds._load_time:.0f}s)")
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        y_true_26, y_score_6 = infer_dataset(model, loader, device)
        # y_true_26 shape (N, 26), y_score_6 shape (N, 6). Slice labels to Tier-M.
        y_true_6 = y_true_26[:, TIER_M_IDX]
        results[center] = _pack(y_true_6, y_score_6, args.n_bootstrap)
        t = results[center]['tier_m']
        print(f"  [{center}] TierM AUROC={t['macro_auroc']:.4f} AUPRC={t['macro_auprc']:.4f}")

    if results:
        main_aur = [results[c]['tier_m']['macro_auroc'] for c in MAIN_CENTERS if c in results]
        main_apr = [results[c]['tier_m']['macro_auprc'] for c in MAIN_CENTERS if c in results]
        main_agg = {
            'macro_auroc_avg': float(np.mean(main_aur)) if main_aur else float('nan'),
            'macro_auprc_avg': float(np.mean(main_apr)) if main_apr else float('nan'),
            'n_centers_averaged': len(main_aur),
        }
    else:
        main_agg = {'macro_auroc_avg': float('nan'), 'macro_auprc_avg': float('nan'),
                    'n_centers_averaged': 0}

    return {'centers': results, 'main_centers_avg': main_agg,
            'main_centers': MAIN_CENTERS, 'small_centers': SMALL_CENTERS}


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
    ckpt = os.path.join(args.model_dir, 'best_model.pt')
    state = torch.load(ckpt, map_location=device)
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded MIMIC Tier-M model from {ckpt}")

    output = {'classes': TIER_M, 'tier_m_idx_in_26': TIER_M_IDX}

    if args.eval_mimic:
        output['mimic_test'] = eval_mimic_test(model, args, device)
    if args.eval_ptbxl:
        output['ptbxl_fold10'] = eval_ptbxl_fold10(model, args, device)
    if args.eval_pn2021:
        output['pn2021'] = eval_pn2021(model, args, device)

    print("\n" + "=" * 80)
    print("Summary: MIMIC Tier-M model — cross-dataset")
    print("=" * 80)
    if 'mimic_test' in output:
        t = output['mimic_test']['tier_m']
        print(f"  MIMIC test  (in-domain)    AUROC={t['macro_auroc']:.4f}  AUPRC={t['macro_auprc']:.4f}  "
              f"N={output['mimic_test']['n_records']}")
    if 'ptbxl_fold10' in output:
        t = output['ptbxl_fold10']['tier_m']
        print(f"  PTBXL fold10 (zero-shot)    AUROC={t['macro_auroc']:.4f}  AUPRC={t['macro_auprc']:.4f}  "
              f"N={output['ptbxl_fold10']['n_records']}")
    if 'pn2021' in output:
        agg = output['pn2021']['main_centers_avg']
        print(f"  PN2021 main-avg (zero-shot) AUROC={agg['macro_auroc_avg']:.4f}  "
              f"AUPRC={agg['macro_auprc_avg']:.4f}  ({agg['n_centers_averaged']} centers)")
        for c in ALL_CENTERS:
            if c in output['pn2021']['centers']:
                r = output['pn2021']['centers'][c]
                tag = '' if c in MAIN_CENTERS else ' (small)'
                t = r['tier_m']
                print(f"    [{c + tag:<22}] N={r['n_records']:>6}  "
                      f"AUROC={t['macro_auroc']:.4f}  AUPRC={t['macro_auprc']:.4f}")
    print("=" * 80)

    out_path = args.output_path or os.path.join(args.model_dir, 'eval_crosscenter_mimic_tierM.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='/root/autodl-tmp/mimic_tierM')
    p.add_argument('--mimic_cache', default='/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy')
    p.add_argument('--mimic_index', default='/root/autodl-tmp/mimic_tierM/mimic_index.npz')
    p.add_argument('--ptbxl_cache', default='/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy')
    p.add_argument('--ptbxl_csv',   default='/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv')
    p.add_argument('--pn2021_dir',  default='/root/ECG_adv_Gen/datasets/physionet2021')
    p.add_argument('--device', default='cuda')
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--n_bootstrap', type=int, default=1000)
    p.add_argument('--eval_mimic',  action='store_true', default=True)
    p.add_argument('--eval_ptbxl',  action='store_true', default=True)
    p.add_argument('--eval_pn2021', action='store_true', default=True)
    p.add_argument('--no_mimic',    dest='eval_mimic',  action='store_false')
    p.add_argument('--no_ptbxl',    dest='eval_ptbxl',  action='store_false')
    p.add_argument('--no_pn2021',   dest='eval_pn2021', action='store_false')
    p.add_argument('--eval_half_only', action='store_true',
                   help='Filter PN2021 records to fold>=6 (eval_half) for clean ablation vs refs from fold<=5.')
    p.add_argument('--output_path', default=None,
                   help='Override output JSON path. Defaults to <model_dir>/eval_crosscenter_mimic_tierM.json.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
