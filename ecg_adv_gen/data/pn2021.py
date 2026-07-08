"""Small PN2021 data helpers."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
import time

import numpy as np


def apply_ref_exclusion(record_ids: Iterable[str], *, excluded_record_ids: set[str]) -> list[str]:
    """Return record IDs after K-shot ref exclusion, preserving input order."""

    return [record_id for record_id in record_ids if record_id not in excluded_record_ids]


def pn2021_preprocess_config(args, *, include_crop=False):
    from ecg_adv_gen.evaluation.pn2021_eval_cache import build_pn2021_preprocess_config
    from ecg_adv_gen.preprocessing.classifier import _resolve_preprocess_flags

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
        crop_len=getattr(args, "crop_len", None),
    )


def pn2021_cache_path(args, center):
    from ecg_adv_gen.evaluation.pn2021_eval_cache import pn2021_npz_cache_path

    cache_dir = getattr(args, "pn2021_cache_dir", None)
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return pn2021_npz_cache_path(cache_dir, args.scheme, center)


def pn2021_mmap_cache_path_for_args(args, center):
    from ecg_adv_gen.evaluation.pn2021_eval_cache import pn2021_mmap_cache_path

    cache_dir = getattr(args, "pn2021_mmap_cache_dir", None)
    if not cache_dir:
        return None
    return pn2021_mmap_cache_path(cache_dir, args.scheme, center)


def expected_pn2021_cache_metadata(args, scheme, center):
    from ecg_adv_gen.evaluation.pn2021_eval_cache import build_pn2021_cache_metadata
    from ecg_adv_gen.labels.super5_mapping import get_super5_pn2021_mapping_metadata

    return build_pn2021_cache_metadata(
        scheme_name=args.scheme,
        center=center,
        class_names=scheme["class_names"],
        preprocess_config=pn2021_preprocess_config(args, include_crop=True),
        pn2021_mapping=(
            get_super5_pn2021_mapping_metadata()
            if args.scheme == "super5"
            else None
        ),
    )


def expected_pn2021_mmap_metadata(args, scheme, center):
    from ecg_adv_gen.evaluation.pn2021_eval_cache import build_pn2021_cache_metadata
    from ecg_adv_gen.labels.super5_mapping import get_super5_pn2021_mapping_metadata

    return build_pn2021_cache_metadata(
        scheme_name=args.scheme,
        center=center,
        class_names=scheme["class_names"],
        preprocess_config=pn2021_preprocess_config(args, include_crop=False),
        pn2021_mapping=(
            get_super5_pn2021_mapping_metadata()
            if args.scheme == "super5"
            else None
        ),
        layout="mmap_v1",
    )


def load_or_build_pn2021_center(center, center_dir, scheme, args):
    """Load a PN2021 clean eval center from cache or build the 100 Hz cache."""

    from ecg_adv_gen.data.pn2021_index import scan_pn2021_center_records
    from ecg_adv_gen.data.pn2021_waveforms import materialize_pn2021_center_records
    from ecg_adv_gen.evaluation.pn2021_eval_cache import (
        load_existing_pn2021_eval_cache,
        write_pn2021_mmap_cache,
    )
    from ecg_adv_gen.preprocessing import unified_preprocess_to_1000

    cache_path = pn2021_cache_path(args, center)
    expected_metadata = expected_pn2021_cache_metadata(args, scheme, center)
    mmap_cache_root = pn2021_mmap_cache_path_for_args(args, center)
    expected_mmap_metadata = expected_pn2021_mmap_metadata(args, scheme, center)

    cache_load = load_existing_pn2021_eval_cache(
        npz_cache_path=cache_path,
        npz_expected_metadata=expected_metadata,
        mmap_cache_root=mmap_cache_root,
        mmap_expected_metadata=expected_mmap_metadata,
    )
    if cache_load is not None:
        if cache_load.converted_npz_to_mmap:
            print(f"  {center}: converted PN2021 cache to mmap -> {mmap_cache_root}")
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

    import wfdb

    records = scan_pn2021_center_records(center_dir)
    materialized = materialize_pn2021_center_records(
        [str(record.record_path) for record in records],
        [list(record.snomeds) for record in records],
        read_record=wfdb.rdrecord,
        preprocess_signal=unified_preprocess_to_1000,
        label_fn=scheme["pn2021_fn"],
        num_classes=scheme["num_classes"],
        preprocess_kwargs={
            "target_fs": 100,
            "target_len": 1000,
            "preprocess_mode": args.preprocess_mode,
            "norm_mode": args.norm_mode,
        },
    )
    signals = materialized.signals
    labels = materialized.labels
    record_ids = materialized.record_ids
    if mmap_cache_root and len(signals) > 0:
        write_pn2021_mmap_cache(
            mmap_cache_root,
            signals,
            labels,
            record_ids,
            expected_mmap_metadata,
        )
        print(f"  {center}: cached preprocessed PN2021 mmap -> {mmap_cache_root}")
    if cache_path and len(signals) > 0:
        np.savez_compressed(
            cache_path,
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            metadata_json=json.dumps(expected_metadata, sort_keys=True),
        )
        print(f"  {center}: cached preprocessed PN2021 -> {cache_path}")
    return (
        signals,
        labels,
        record_ids,
        materialized.fail,
        materialized.load_time_s,
        False,
        "built",
    )


def load_clean_center(args, center):
    """Load clean PN2021 eval cache, preferring mmap over legacy NPZ."""

    from ecg_adv_gen.evaluation.pn2021_corruptions import (
        clean_mmap_cache_path,
        clean_npz_cache_path,
        load_npz_metadata,
    )
    from ecg_adv_gen.evaluation.pn2021_eval_cache import PN2021_EVAL_CACHE_VERSION

    mmap_root = clean_mmap_cache_path(
        args.clean_mmap_cache_dir,
        args.scheme,
        center,
        PN2021_EVAL_CACHE_VERSION,
    )
    if os.path.isdir(mmap_root):
        sig_path = os.path.join(mmap_root, "signals.npy")
        lab_path = os.path.join(mmap_root, "labels.npy")
        rid_path = os.path.join(mmap_root, "record_ids.npy")
        meta_path = os.path.join(mmap_root, "metadata.json")
        if os.path.exists(sig_path) and os.path.exists(lab_path) and os.path.exists(rid_path):
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
            return (
                np.load(sig_path, mmap_mode="r"),
                np.load(lab_path, mmap_mode="r"),
                np.load(rid_path, allow_pickle=True).astype(str),
                metadata,
                "mmap",
            )
    npz_path = clean_npz_cache_path(
        args.clean_cache_dir,
        args.scheme,
        center,
        PN2021_EVAL_CACHE_VERSION,
    )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"clean cache not found for {center}: {mmap_root} or {npz_path}"
        )
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["signals"].astype(np.float32, copy=False),
        data["labels"].astype(np.float32, copy=False),
        data["record_ids"].astype(str),
        load_npz_metadata(data),
        "npz",
    )


def load_raw_first_center(args, scheme, center):
    """Load PN2021 center as 100 Hz / 1000 point raw-first signals without z-score."""

    import wfdb

    from ecg_adv_gen.data.pn2021_index import scan_pn2021_center_records
    from ecg_adv_gen.evaluation.pn2021_eval_cache import PN2021_EVAL_CACHE_VERSION
    from ecg_adv_gen.labels.super5_mapping import get_super5_pn2021_mapping_metadata
    from ecg_adv_gen.preprocessing import unified_preprocess_to_1000

    cache = getattr(args, "_raw_first_center_cache", None)
    if cache is None:
        cache = {}
        args._raw_first_center_cache = cache
    if center in cache:
        return cache[center]

    center_dir = os.path.join(args.pn2021_root, "training", center)
    if not os.path.isdir(center_dir):
        raise FileNotFoundError(f"PN2021 center directory not found: {center_dir}")

    records = scan_pn2021_center_records(center_dir)
    signals = []
    labels = []
    record_ids = []
    fail = 0
    t0 = time.time()
    for record in records:
        try:
            rec = wfdb.rdrecord(str(record.record_path))
        except Exception:
            fail += 1
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            fail += 1
            continue
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
        proc = unified_preprocess_to_1000(
            sig.astype(np.float32),
            fs=rec.fs,
            source_leads=sig_names,
            target_fs=100,
            target_len=1000,
            preprocess_mode="minimal_resample",
            norm_mode="none",
        )
        if proc is None:
            fail += 1
            continue
        signals.append(proc.astype(np.float32, copy=False))
        labels.append(scheme["pn2021_fn"](record.snomeds).astype(np.float32, copy=False))
        record_ids.append(str(record.record_id))

    if not signals:
        raise RuntimeError(f"raw-first PN2021 load produced no records for {center}")
    metadata = {
        "scheme": str(args.scheme),
        "center": str(center),
        "class_names": list(scheme["class_names"]),
        "cache_version": str(PN2021_EVAL_CACHE_VERSION),
        "source": "wfdb_raw_first",
        "source_center_dir": center_dir,
        "n_scanned_records": int(len(records)),
        "n_failed_records": int(fail),
        "load_time_s": float(time.time() - t0),
        "preprocess_config": {
            "target_fs": 100,
            "target_len": 1000,
            "apply_filter": False,
            "apply_zscore": False,
            "preprocess_mode": "minimal_resample",
            "norm_mode": "none",
        },
        "model_input_preprocess_after_corruption": {
            "apply_zscore": True,
            "norm_mode": "per_sample_global",
            "crop_len": int(args.crop_len),
            "crop_mode": "center",
        },
        "corruption_order": [
            "wfdb_read",
            "lead_reorder_nan_guard_resample_pad_no_zscore",
            "corruption",
            "per_sample_global_zscore",
            "center_crop",
            "model",
        ],
    }
    if args.scheme == "super5":
        metadata["pn2021_mapping"] = get_super5_pn2021_mapping_metadata()

    loaded = (
        np.stack(signals, axis=0).astype(np.float32),
        np.stack(labels, axis=0).astype(np.float32),
        np.asarray(record_ids, dtype=str),
        metadata,
        "raw_first:wfdb",
    )
    cache[center] = loaded
    print(
        f"  {center}: loaded raw-first PN2021 n={len(record_ids)} "
        f"fail={fail} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    return loaded


def load_native_raw_first_center(args, scheme, center):
    """Load PN2021 center as native-rate raw ECGs for native-first corruptions."""

    import wfdb

    from ecg_adv_gen.data import waveform_datasets
    from ecg_adv_gen.data.pn2021_index import scan_pn2021_center_records
    from ecg_adv_gen.evaluation.pn2021_eval_cache import PN2021_EVAL_CACHE_VERSION
    from ecg_adv_gen.labels.super5_mapping import get_super5_pn2021_mapping_metadata

    cache = getattr(args, "_native_raw_first_center_cache", None)
    if cache is None:
        cache = {}
        args._native_raw_first_center_cache = cache
    if center in cache:
        return cache[center]

    center_dir = os.path.join(args.pn2021_root, "training", center)
    if not os.path.isdir(center_dir):
        raise FileNotFoundError(f"PN2021 center directory not found: {center_dir}")

    records = scan_pn2021_center_records(center_dir)
    signals = []
    sample_rates = []
    labels = []
    record_ids = []
    fail = 0
    t0 = time.time()
    for record in records:
        try:
            rec = wfdb.rdrecord(str(record.record_path))
        except Exception:
            fail += 1
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            fail += 1
            continue
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
        proc = waveform_datasets.prepare_native_raw_signal_tc(
            sig.astype(np.float32),
            sig_names,
        )
        if proc is None:
            fail += 1
            continue
        signals.append(proc)
        sample_rates.append(float(rec.fs))
        labels.append(scheme["pn2021_fn"](record.snomeds).astype(np.float32, copy=False))
        record_ids.append(str(record.record_id))

    if not signals:
        raise RuntimeError(f"native raw-first PN2021 load produced no records for {center}")
    metadata = {
        "scheme": str(args.scheme),
        "center": str(center),
        "class_names": list(scheme["class_names"]),
        "cache_version": str(PN2021_EVAL_CACHE_VERSION),
        "source": "wfdb_native_raw_first",
        "source_center_dir": center_dir,
        "n_scanned_records": int(len(records)),
        "n_failed_records": int(fail),
        "load_time_s": float(time.time() - t0),
        "pre_corruption_config": {
            "lead_reorder": True,
            "nan_guard": True,
            "apply_filter": False,
            "apply_resample": False,
            "apply_pad_truncate": False,
            "apply_zscore": False,
            "sample_rate_hz": "native_per_record",
            "signal_length": "native_per_record",
        },
        "model_input_preprocess_after_corruption": {
            "target_fs": 100,
            "target_len": 1000,
            "preprocess_mode": "minimal_resample",
            "apply_filter": False,
            "apply_zscore": True,
            "norm_mode": "per_sample_global",
            "crop_len": int(args.crop_len),
            "crop_mode": "center",
        },
        "corruption_order": [
            "wfdb_read_native_fs_native_length",
            "lead_reorder_nan_guard_no_resample_no_zscore",
            "corruption_with_native_sample_rate",
            "resample_pad_or_truncate_to_100hz_1000",
            "per_sample_global_zscore",
            "center_crop",
            "model",
        ],
    }
    if args.scheme == "super5":
        metadata["pn2021_mapping"] = get_super5_pn2021_mapping_metadata()

    loaded = (
        signals,
        np.stack(labels, axis=0).astype(np.float32),
        np.asarray(record_ids, dtype=str),
        metadata,
        "native_raw_first:wfdb",
        np.asarray(sample_rates, dtype=np.float32),
    )
    cache[center] = loaded
    print(
        f"  {center}: loaded native raw-first PN2021 n={len(record_ids)} "
        f"fail={fail} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    return loaded
