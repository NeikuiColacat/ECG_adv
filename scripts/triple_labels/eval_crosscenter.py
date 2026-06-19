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
import hashlib

import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'model', 'DeepECG', 'notebooks'))

from scripts.triple_labels.label_schemes import (
    get_scheme, get_super5_pn2021_mapping_metadata,
)
from scripts.triple_labels.model_zoo import (
    available_model_names, build_super5_model, normalize_model_name,
)
from scripts.triple_labels.train_ptbxl import (
    PTBXLDatasetScheme, compute_macro_auroc_auprc, masked_bce_with_logits,
    get_ptbxl_labels_for_scheme,
)
from scripts.crosscenter_v2.preprocess_utils import (
    unified_preprocess_to_1000, crop_signal_tc, _resolve_preprocess_flags,
)
from ecg_adv_gen.data import (
    PN2021_EVAL_CENTERS_7,
    PN2021_LEAK_EXCLUDED_CENTERS,
    PN2021_TARGET_CENTERS_4,
    load_include_record_ids_from_meta,
    load_ref_record_ids_from_meta,
    materialize_pn2021_center_records,
    parse_header_snomeds,
    scan_pn2021_center_records,
)
from ecg_adv_gen.evaluation import (
    ALLOWED_EVAL_PROTOCOLS,
    DROP_ALL_ZERO_POLICY,
    filter_pn2021_center_records,
    normalize_paper_preprocess_mode,
    summarize_center_view,
    validate_pn2021_eval_protocol,
)
from ecg_adv_gen.evaluation.pn2021_eval_cache import (
    PN2021_EVAL_CACHE_VERSION,
    build_pn2021_cache_metadata,
    build_pn2021_preprocess_config,
    legacy_default_metadata_variant,
    load_existing_pn2021_eval_cache,
    load_npz_metadata,
    metadata_matches_expected,
    pn2021_mmap_cache_path,
    pn2021_npz_cache_path,
    write_pn2021_mmap_cache,
)
from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID


# PN2021 centers — ptb-xl explicitly excluded (data leakage with PTB-XL train)
PN2021_CENTERS = list(PN2021_EVAL_CENTERS_7)
PN2021_FORBIDDEN = set(PN2021_LEAK_EXCLUDED_CENTERS)


# ────────────────────────────────────────────────────────────────────────────
# PN2021 evaluation
# ────────────────────────────────────────────────────────────────────────────

def parse_header_snomed(header_path):
    return parse_header_snomeds(header_path)


def scan_center_records(center_dir):
    records = scan_pn2021_center_records(center_dir)
    return [str(record.record_path) for record in records], [list(record.snomeds) for record in records]


class PN2021CenterDataset(Dataset):
    def __init__(
        self,
        record_paths,
        labels,
        crop_len=250,
        preprocess_mode='legacy_ecgfounder_filter',
        norm_mode='per_sample_global',
    ):
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
                preprocess_mode=preprocess_mode, norm_mode=norm_mode,
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


class PN2021CachedCenterDataset(Dataset):
    def __init__(self, signals, labels, crop_len=250):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.crop_len = crop_len
        self._fail = 0
        self._load_time = 0.0

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len, mode='center')
        sig_ct = crop.T
        label = np.array(self.labels[idx], dtype=np.float32, copy=True)
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(label).float())


def _pn2021_preprocess_config(args, include_crop=False):
    apply_filter, apply_zscore = _resolve_preprocess_flags(
        apply_filter=True,
        apply_zscore=True,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )
    return build_pn2021_preprocess_config(
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
        apply_filter=apply_filter,
        apply_zscore=apply_zscore,
        include_crop=include_crop,
        crop_len=getattr(args, 'crop_len', None),
    )


def _legacy_default_metadata_variant(metadata):
    """Return the pre-2026-05 metadata shape for default preprocessing caches."""
    return legacy_default_metadata_variant(metadata)


def _metadata_matches_expected(found, expected):
    return metadata_matches_expected(found, expected)


def _pn2021_cache_path(args, scheme, center):
    cache_dir = getattr(args, 'pn2021_cache_dir', None)
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return pn2021_npz_cache_path(cache_dir, args.scheme, center)


def _pn2021_mmap_cache_path(args, scheme, center):
    cache_dir = getattr(args, 'pn2021_mmap_cache_dir', None)
    if not cache_dir:
        return None
    return pn2021_mmap_cache_path(cache_dir, args.scheme, center)


def _expected_pn2021_cache_metadata(args, scheme, center):
    return build_pn2021_cache_metadata(
        scheme_name=args.scheme,
        center=center,
        class_names=scheme['class_names'],
        preprocess_config=_pn2021_preprocess_config(args, include_crop=True),
        pn2021_mapping=(
            get_super5_pn2021_mapping_metadata()
            if args.scheme == 'super5'
            else None
        ),
    )


def _expected_pn2021_mmap_metadata(args, scheme, center):
    return build_pn2021_cache_metadata(
        scheme_name=args.scheme,
        center=center,
        class_names=scheme['class_names'],
        preprocess_config=_pn2021_preprocess_config(args, include_crop=False),
        pn2021_mapping=(
            get_super5_pn2021_mapping_metadata()
            if args.scheme == 'super5'
            else None
        ),
        layout='mmap_v1',
    )


def _load_cache_metadata(data):
    return load_npz_metadata(data)


def _cache_metadata_matches(found, expected):
    return _metadata_matches_expected(found, expected)


def _load_or_build_pn2021_center(center, center_dir, scheme, args):
    cache_path = _pn2021_cache_path(args, scheme, center)
    expected_metadata = _expected_pn2021_cache_metadata(args, scheme, center)
    mmap_cache_root = _pn2021_mmap_cache_path(args, scheme, center)
    expected_mmap_metadata = _expected_pn2021_mmap_metadata(args, scheme, center)

    cache_load = load_existing_pn2021_eval_cache(
        npz_cache_path=cache_path,
        npz_expected_metadata=expected_metadata,
        mmap_cache_root=mmap_cache_root,
        mmap_expected_metadata=expected_mmap_metadata,
    )
    if cache_load is not None:
        if cache_load.converted_npz_to_mmap:
            print(f"  {center}: converted PN2021 cache to mmap → {mmap_cache_root}")
        loaded = cache_load.cache
        return (
            loaded.signals,
            loaded.labels,
            loaded.record_ids,
            0,
            0.0,
            loaded.cache_hit,
            loaded.cache_kind,
        )
    if cache_path and os.path.exists(cache_path):
        print(f"  {center}: cache metadata mismatch, rebuilding {cache_path}")

    paths, snomeds = scan_center_records(center_dir)
    materialized = materialize_pn2021_center_records(
        paths,
        snomeds,
        read_record=wfdb.rdrecord,
        preprocess_signal=unified_preprocess_to_1000,
        label_fn=scheme['pn2021_fn'],
        num_classes=scheme['num_classes'],
        preprocess_kwargs={
            'target_fs': 100,
            'target_len': 1000,
            'preprocess_mode': args.preprocess_mode,
            'norm_mode': args.norm_mode,
        },
    )
    signals = materialized.signals
    labels = materialized.labels
    record_ids = materialized.record_ids
    fail = materialized.fail
    load_time = materialized.load_time_s
    if mmap_cache_root and len(signals) > 0:
        write_pn2021_mmap_cache(
            mmap_cache_root, signals, labels, record_ids, expected_mmap_metadata,
        )
        print(f"  {center}: cached preprocessed PN2021 mmap → {mmap_cache_root}")
    if cache_path and len(signals) > 0:
        np.savez_compressed(
            cache_path,
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            metadata_json=json.dumps(expected_metadata, sort_keys=True),
        )
        print(f"  {center}: cached preprocessed PN2021 → {cache_path}")
    return signals, labels, record_ids, fail, load_time, False, 'built'


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


def _load_excluded_ref_ids(meta_paths):
    """Plan Rev 7 Issue #39 / Rev 13.2: load `ref_record_ids` from one or more
    {tag}_k{K}.meta.json files and group by `center`. Records present in any
    ref pool for a given center are excluded from that center's PN2021 eval
    so the model never sees its own training refs at test time.

    Returns: {center_name: set(record_id)}
    """
    out = {}
    if not meta_paths:
        return out
    for mp in meta_paths:
        center, ids = load_ref_record_ids_from_meta(mp)
        if not ids:
            print(f"[exclude] WARN {mp} has no ref_record_ids — skipped")
            continue
        out.setdefault(center, set()).update(ids)
        print(f"[exclude] loaded {len(ids)} ref ids for center='{center}' from {mp}")
    return out


def _ref_id_hashes_by_center(excluded_by_center):
    out = {}
    for center, ids in sorted((excluded_by_center or {}).items()):
        values = sorted(str(item) for item in ids)
        payload = ("\n".join(values) + "\n").encode("utf-8")
        out[str(center)] = hashlib.sha256(payload).hexdigest()
    return out


def _load_included_record_ids(meta_paths):
    """Load explicit PN2021 evaluation IDs grouped by center.

    This is mainly used by seed/fold stability checks: each fold trains on
    ``ref_record_ids`` and evaluates the target center only on
    ``heldout_record_ids``. Centers absent from this mapping are evaluated
    normally.
    """
    out = {}
    if not meta_paths:
        return out
    include_keys = (
        'heldout_record_ids',
        'eval_record_ids',
        'include_record_ids',
        'test_record_ids',
    )
    for mp in meta_paths:
        center, ids, key_used = load_include_record_ids_from_meta(mp, include_keys=include_keys)
        if not ids:
            print(f"[include] WARN {mp} has no heldout/eval record ids — skipped")
            continue
        out.setdefault(center, set()).update(ids)
        print(
            f"[include] loaded {len(ids)} {key_used} for center='{center}' "
            f"from {mp}"
        )
    return out


def eval_pn2021(model, scheme, args, device):
    print(f"\n[pn2021] evaluating {len(PN2021_CENTERS)} centers (ptb-xl excluded)")
    pn2021_root = args.pn2021_root
    excluded_by_center = getattr(args, '_excluded_by_center', {}) or {}
    included_by_center = getattr(args, '_included_by_center', {}) or {}
    report_drop_all_zero = bool(getattr(args, 'report_drop_all_zero_pn2021', False))
    if excluded_by_center:
        total_excl = sum(len(v) for v in excluded_by_center.values())
        print(f"[pn2021] ref-id exclusion active: {total_excl} ids across "
              f"{len(excluded_by_center)} centers")
    if included_by_center:
        total_incl = sum(len(v) for v in included_by_center.values())
        print(f"[pn2021] explicit include-id filter active: {total_incl} ids across "
              f"{len(included_by_center)} centers")
    per_center = {}
    macro_aurocs, macro_auprcs = [], []
    drop_all_zero_aurocs, drop_all_zero_auprcs = [], []
    metric_fn = lambda y_true, y_score: compute_macro_auroc_auprc(
        y_true,
        y_score,
        scheme['class_names'],
        min_pos=args.min_pos,
    )
    for center in PN2021_CENTERS:
        assert center.lower() not in PN2021_FORBIDDEN, \
            f"FORBIDDEN center {center} would leak PTB-XL data"
        center_dir = os.path.join(pn2021_root, 'training', center)
        if not os.path.isdir(center_dir):
            print(f"  {center}: missing dir, skipped")
            continue
        t0 = time.time()

        # Plan Rev 7 Issue #39 / Rev 13.2: filter ref records (basename match).
        # The preprocessing cache must remain center-complete; apply exclusions
        # after loading/building it so each model can use its own ref-id set.
        excluded_set = excluded_by_center.get(center, set())
        included_set = included_by_center.get(center, set())

        signals, labels, record_ids, fail, load_time, cache_hit, cache_kind = _load_or_build_pn2021_center(
            center, center_dir, scheme, args
        )
        n_scanned = int(len(record_ids) + fail)
        filtered = filter_pn2021_center_records(
            signals,
            labels,
            record_ids,
            include_ids=included_set,
            ref_ids=excluded_set,
        )
        signals = filtered.signals
        labels = filtered.labels
        record_ids = filtered.record_ids
        n_excluded_ref = filtered.n_excluded_ref
        n_include_kept = filtered.n_include_kept
        if args.pn2021_limit and args.pn2021_limit < len(signals):
            signals = signals[:args.pn2021_limit]
            labels = labels[:args.pn2021_limit]
            record_ids = record_ids[:args.pn2021_limit]
        ds = PN2021CachedCenterDataset(signals, labels, crop_len=args.crop_len)
        ds._fail = fail
        ds._load_time = load_time
        if len(ds) == 0:
            print(f"  {center}: no valid records")
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        y_true, y_score = infer_dataset(model, loader, device)
        center_row = summarize_center_view(
            y_true,
            y_score,
            metric_fn=metric_fn,
            n_total=len(ds),
            n_excluded_ref=n_excluded_ref,
            n_include_kept=n_include_kept,
            extra_fields={
                'n_scanned': n_scanned,
                'load_time_s': round(ds._load_time, 1),
                'cache_hit': cache_hit,
                'cache_kind': cache_kind,
            },
            report_drop_all_zero=report_drop_all_zero,
            drop_none_if_empty=False,
            always_include_count_fields=True,
        )
        per_center[center] = center_row
        macro_aurocs.append(center_row['macro_auroc'])
        macro_auprcs.append(center_row['macro_auprc'])
        n_all_zero = center_row['n_all_zero_labels']
        n_nonzero = center_row['n_nonzero_labels']
        if report_drop_all_zero and n_nonzero > 0:
            drop_all_zero_aurocs.append(center_row['drop_all_zero_macro_auroc'])
            drop_all_zero_auprcs.append(center_row['drop_all_zero_macro_auprc'])
        excl_tag = f" (-{n_excluded_ref} ref)" if n_excluded_ref > 0 else ""
        incl_tag = f" include={n_include_kept}" if included_set else ""
        cache_tag = f" {cache_kind}" if cache_hit else ""
        print(f"  {center:<22} n={len(ds):>5}{incl_tag}{excl_tag}{cache_tag}  "
              f"AUROC={center_row['macro_auroc']:.4f}  "
              f"AUPRC={center_row['macro_auprc']:.4f}  "
              f"n_classes={center_row['n_classes_used']}  ({time.time()-t0:.0f}s)")
        if report_drop_all_zero and n_nonzero > 0:
            print(f"    drop-all-zero n={n_nonzero:>5} (-{n_all_zero})  "
                  f"AUROC={center_row['drop_all_zero_macro_auroc']:.4f}  "
                  f"AUPRC={center_row['drop_all_zero_macro_auprc']:.4f}  "
                  f"n_classes={center_row['drop_all_zero_n_classes_used']}")
    finite_aurocs = [v for v in macro_aurocs if np.isfinite(v)]
    finite_auprcs = [v for v in macro_auprcs if np.isfinite(v)]
    avg_auroc = float(np.mean(finite_aurocs)) if finite_aurocs else float('nan')
    avg_auprc = float(np.mean(finite_auprcs)) if finite_auprcs else float('nan')
    print(f"  ──── 7-center average AUROC={avg_auroc:.4f}  AUPRC={avg_auprc:.4f}")
    result = {
        'avg_macro_auroc': avg_auroc,
        'avg_macro_auprc': avg_auprc,
        'per_center': per_center,
    }
    if report_drop_all_zero:
        finite_drop_aurocs = [v for v in drop_all_zero_aurocs if np.isfinite(v)]
        finite_drop_auprcs = [v for v in drop_all_zero_auprcs if np.isfinite(v)]
        avg_drop_auroc = float(np.mean(finite_drop_aurocs)) if finite_drop_aurocs else float('nan')
        avg_drop_auprc = float(np.mean(finite_drop_auprcs)) if finite_drop_auprcs else float('nan')
        result.update({
            'drop_all_zero_policy': DROP_ALL_ZERO_POLICY,
            'avg_drop_all_zero_macro_auroc': avg_drop_auroc,
            'avg_drop_all_zero_macro_auprc': avg_drop_auprc,
        })
        print(f"  ──── drop-all-zero 7-center average "
              f"AUROC={avg_drop_auroc:.4f}  AUPRC={avg_drop_auprc:.4f}")
    protocol_preprocess_mode = normalize_paper_preprocess_mode(
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
        crop_len=args.crop_len,
    )
    result['eval_protocol'] = validate_pn2021_eval_protocol(
        eval_protocol=args.eval_protocol,
        per_center=per_center,
        target_centers=PN2021_TARGET_CENTERS_4,
        eval_centers=PN2021_CENTERS,
        preprocess_mode=protocol_preprocess_mode,
        crop_len=args.crop_len,
        diagnostic_legacy_eval=args.diagnostic_legacy_eval,
        allow_missing_centers=args.allow_missing_centers,
        min_target_ref_excluded=args.min_target_ref_excluded,
    )
    result['eval_protocol']['actual_preprocess_mode'] = args.preprocess_mode
    result['eval_protocol']['actual_norm_mode'] = args.norm_mode
    result['eval_protocol']['target_ref_id_hashes'] = _ref_id_hashes_by_center(
        getattr(args, '_excluded_by_center', {}) or {}
    )
    return result


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
    p.add_argument('--checkpoint_name', default='best_model.pt',
                   help='Checkpoint filename under --model_dir.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--model_name', default='efficientnet1dv2',
                   choices=available_model_names())
    p.add_argument('--crop_len', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--min_pos', type=int, default=10,
                   help='Skip class metrics if n_pos < min_pos')
    p.add_argument('--ptbxl_csv', default='/root/autodl-tmp/ptbxl/ptbxl_database.csv')
    p.add_argument('--ptbxl_cache', default=None)
    p.add_argument('--preprocess_mode', default='legacy_ecgfounder_filter',
                   choices=['minimal_resample', 'legacy_ecgfounder_filter',
                            'raw_for_generation_or_digital'],
                   help='Named preprocessing branch for PN2021 cache builds. '
                        'PTB-XL eval uses --ptbxl_cache and must be built separately.')
    p.add_argument('--norm_mode', default='per_sample_global',
                   choices=['per_sample_global', 'none'],
                   help='Normalization branch for PN2021 cache builds.')
    p.add_argument('--pn2021_root', default='/root/autodl-tmp/physionet2021')
    p.add_argument('--pn2021_cache_dir', default='/root/autodl-tmp/triple_labels/pn2021_eval_cache',
                   help='Cache preprocessed PN2021 center signals/labels for repeated model evals')
    p.add_argument('--pn2021_mmap_cache_dir',
                   default='/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap',
                   help='Mmap-friendly PN2021 cache root. Preferred over compressed npz when present.')
    p.add_argument('--pn2021_limit', type=int, default=None,
                   help='Cap records per center (for smoke test)')
    p.add_argument('--mimic_limit', type=int, default=None,
                   help='Cap MIMIC test records (for smoke test)')
    p.add_argument('--skip_pn2021', action='store_true')
    p.add_argument('--skip_mimic', action='store_true')
    p.add_argument('--output_path', default=None)
    p.add_argument('--exclude_ref_ids', nargs='+', default=[],
                   help='Plan Rev 7 Issue #39: paths to one or more '
                        '{tag}_k{K}.meta.json files. Records whose '
                        'basename matches any meta\'s ref_record_ids will be '
                        'excluded from that center\'s PN2021 eval, so the '
                        'fine-tuned model never tests on its own training refs.')
    p.add_argument('--include_record_ids', nargs='+', default=[],
                   help='Optional paths to meta JSON files containing '
                        'heldout_record_ids/eval_record_ids/include_record_ids. '
                        'For centers present in these files, PN2021 eval is '
                        'restricted to those record IDs after loading the full '
                        'cache. This is intended for seed/fold stability checks.')
    p.add_argument('--report_drop_all_zero_pn2021', action='store_true',
                   help='Also report a PN2021 metric view that removes rows '
                        'with no positive Super5 label before AUROC/AUPRC.')
    p.add_argument('--eval_protocol', choices=ALLOWED_EVAL_PROTOCOLS,
                   default='paper_refexcluded',
                   help='PN2021 eval safety protocol. paper_refexcluded requires '
                        'all eval centers, paper preprocessing, and target ref exclusion.')
    p.add_argument('--diagnostic_legacy_eval', action='store_true',
                   help='Allow legacy preprocessing in diagnostic_unrefexcluded mode only.')
    p.add_argument('--allow_missing_centers', action='store_true',
                   help='Allow PN2021 center directories with no evaluated rows; diagnostic use only.')
    p.add_argument('--min_target_ref_excluded', type=int, default=500,
                   help='Minimum excluded K-shot refs required per target center in paper mode.')
    args = p.parse_args()
    args._excluded_by_center = _load_excluded_ref_ids(args.exclude_ref_ids)
    args._included_by_center = _load_included_record_ids(args.include_record_ids)

    device = torch.device(args.device)
    scheme = get_scheme(args.scheme)
    print(f"[scheme] {args.scheme}  num_classes={scheme['num_classes']}")

    model_name = normalize_model_name(args.model_name)
    model = build_super5_model(model_name, num_classes=scheme['num_classes']).to(device)
    ckpt = os.path.join(args.model_dir, args.checkpoint_name)
    sd = torch.load(ckpt, map_location=device)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {ckpt} ({model_name})")

    # Exclude private fields like `_excluded_by_center` (sets are not JSON serializable)
    output = {
        'scheme': args.scheme,
        'num_classes': scheme['num_classes'],
        'class_names': list(scheme['class_names']),
        'cache_versions': {
            'pn2021_eval': PN2021_EVAL_CACHE_VERSION,
        },
        'config': {k: v for k, v in vars(args).items() if not k.startswith('_')},
    }
    output['preprocess'] = {
        'contract_id': PREPROCESS_CONTRACT_ID,
        'preprocess_mode': args.preprocess_mode,
        'norm_mode': args.norm_mode,
        'crop_len': int(args.crop_len),
        'target_fs': 100,
        'target_len': 1000,
    }
    if args.scheme == 'super5':
        mapping_metadata = get_super5_pn2021_mapping_metadata()
        output['label_mapping'] = {
            'version': mapping_metadata.get('mapping_version'),
            'hash': mapping_metadata.get('mapping_hash'),
            'pn2021_super5': mapping_metadata,
        }

    output['ptbxl_test'] = eval_ptbxl_test(model, scheme, args, device)
    if not args.skip_pn2021:
        output['pn2021'] = eval_pn2021(model, scheme, args, device)
    if not args.skip_mimic:
        output['mimic_test'] = eval_mimic(model, scheme, args, device)

    out_path = args.output_path or os.path.join(args.model_dir, 'eval_result.json')
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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
