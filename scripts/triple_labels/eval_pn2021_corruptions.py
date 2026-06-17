"""Evaluate EfficientNet1DV2 on PN2021-C corruption caches.

This script intentionally stays separate from eval_crosscenter.py so the clean
PN2021 benchmark remains unchanged.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "model", "DeepECG", "notebooks"))
from scripts.triple_labels.build_pn2021_corruptions import (  # noqa: E402
    DEFAULT_CENTERS,
    DEFAULT_CORRUPTIONS,
    PUBLIC_TO_INTERNAL_SEVERITY,
    PN2021_C_CACHE_VERSION,
)
from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021_EVAL_CACHE_VERSION,
    PN2021CachedCenterDataset,
    compute_macro_auroc_auprc,
    infer_dataset,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    get_scheme,
    get_super5_pn2021_mapping_metadata,
)
from scripts.triple_labels.model_zoo import (  # noqa: E402
    available_model_names,
    build_super5_model,
    normalize_model_name,
)
from methods.augmix.ecg_ops import (  # noqa: E402
    BaselineShift,
    BaselineWander,
    EMGNoise,
    PowerlineNoise,
    RandomLeadsMask,
)
from methods.augmix.severity import build_op  # noqa: E402
from ecg_adv_gen.data.kshot import load_ref_record_ids_by_center  # noqa: E402
from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID  # noqa: E402
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    aggregate_corruption_summary,
    clean_mmap_cache_path,
    clean_npz_cache_path,
    corruption_cache_path,
    filter_record_indices,
    load_json_payload,
    load_clean_metric_lookup,
    load_npz_metadata,
    require_clean_eval_json,
    stable_corruption_seed,
)
from ecg_adv_gen.evaluation.pn2021c_metadata import (  # noqa: E402
    PN2021CMetadataError,
    build_center_scoped_clean_eval_payload,
    build_pn2021c_metadata_payload,
    validate_pn2021c_metadata_compatibility,
)
from ecg_adv_gen.models.ecgfounder_torch import (  # noqa: E402
    global_zscore_torch,
    stabilize_ecg_torch,
)


STRESS_PROFILE_CHOICES = ("standard", "stress_v2", "calibrated_10to20pp")


_STRESS_V2_PARAMS = {
    # Stress profile calibrated for z-scored 100Hz/10s ECG. It is deliberately
    # stronger than the training-time AugMix severity table and should only be
    # used as a corruption benchmark.
    "powerline_noise": {
        1: {"max_amplitude": 0.15},
        2: {"max_amplitude": 0.30},
        3: {"max_amplitude": 0.60},
        4: {"max_amplitude": 0.90},
        5: {"max_amplitude": 1.20},
    },
    "emg_noise": {
        1: {"max_amplitude": 0.10},
        2: {"max_amplitude": 0.25},
        3: {"max_amplitude": 0.50},
        4: {"max_amplitude": 0.80},
        5: {"max_amplitude": 1.20},
    },
    "baseline_wander": {
        1: {"max_amplitude": 0.20, "k": 3},
        2: {"max_amplitude": 0.40, "k": 3},
        3: {"max_amplitude": 0.80, "k": 4},
        4: {"max_amplitude": 1.20, "k": 4},
        5: {"max_amplitude": 1.60, "k": 5},
    },
    "baseline_shift": {
        1: {"max_amplitude": 0.20, "shift_ratio": 0.10, "num_segment": 1},
        2: {"max_amplitude": 0.50, "shift_ratio": 0.20, "num_segment": 1},
        3: {"max_amplitude": 0.80, "shift_ratio": 0.35, "num_segment": 2},
        4: {"max_amplitude": 1.10, "shift_ratio": 0.55, "num_segment": 2},
        5: {"max_amplitude": 1.40, "shift_ratio": 0.75, "num_segment": 3},
    },
    "random_leads_masking": {
        1: {"mask_leads_prob": 0.15},
        2: {"mask_leads_prob": 0.30},
        3: {"mask_leads_prob": 0.50},
        4: {"mask_leads_prob": 0.70},
        5: {"mask_leads_prob": 0.85},
    },
}


_CALIBRATED_10TO20PP_PARAMS = {
    # Fixed single-operator profile calibrated on 2026-06-06 to induce roughly
    # 10-20 pp mean AUROC/AUPRC drops on the four-center v7 K500 mainline.
    # These parameters are validated only for public severity=5.
    "powerline_noise": {
        5: {"max_amplitude": 8.0},
    },
    "emg_noise": {
        5: {"max_amplitude": 2.3},
    },
    "baseline_wander": {
        5: {"max_amplitude": 2.5, "k": 6, "max_freq": 0.8},
    },
    "baseline_shift": {
        5: {"max_amplitude": 2.4, "shift_ratio": 0.9, "num_segment": 6},
    },
    "random_leads_masking": {
        5: {"mask_leads_prob": 0.57},
    },
}


def _build_profile_op(profile_params, profile_name, corruption, public_severity):
    public_severity = int(public_severity)
    if corruption not in profile_params:
        raise ValueError(f"{profile_name} does not define corruption: {corruption}")
    if public_severity not in profile_params[corruption]:
        raise ValueError(
            f"{profile_name} defines {corruption} only for severities "
            f"{sorted(profile_params[corruption])}; got {public_severity}"
        )
    params = dict(profile_params[corruption][public_severity])
    params.setdefault("p", 1.0)
    if corruption == "powerline_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return PowerlineNoise(**params)
    if corruption == "emg_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("dependency", False)
        return EMGNoise(**params)
    if corruption == "baseline_wander":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("min_freq", 0.03)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return BaselineWander(**params)
    if corruption == "baseline_shift":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return BaselineShift(**params)
    if corruption == "random_leads_masking":
        params.setdefault("mask_leads_selection", "random")
        return RandomLeadsMask(**params)
    raise ValueError(f"unknown corruption: {corruption}")


def _build_stress_v2_op(corruption, public_severity):
    return _build_profile_op(_STRESS_V2_PARAMS, "stress_v2", corruption, public_severity)


def _build_calibrated_10to20pp_op(corruption, public_severity):
    return _build_profile_op(
        _CALIBRATED_10TO20PP_PARAMS,
        "calibrated_10to20pp",
        corruption,
        public_severity,
    )


def _build_corruption_op(corruption, public_severity, severity_profile):
    if severity_profile == "standard":
        return build_op(corruption, PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)])
    if severity_profile == "stress_v2":
        return _build_stress_v2_op(corruption, public_severity)
    if severity_profile == "calibrated_10to20pp":
        return _build_calibrated_10to20pp_op(corruption, public_severity)
    raise ValueError(f"unknown severity_profile: {severity_profile}")


def _input_stabilizer_config(args):
    return {
        "bandpass_low_hz": args.input_bandpass_low_hz,
        "bandpass_high_hz": args.input_bandpass_high_hz,
        "repair_flat_leads": bool(args.input_repair_flat_leads),
        "clip_abs": args.input_clip_abs,
        "renorm_after_stabilizer": bool(args.input_renorm_after_stabilizer),
        "sample_rate_hz": float(args.input_sample_rate_hz),
        "stage": str(args.input_stabilizer_stage),
    }


def _has_input_stabilizer(config):
    return bool(
        config.get("bandpass_low_hz") is not None
        or config.get("bandpass_high_hz") is not None
        or config.get("repair_flat_leads")
        or config.get("clip_abs") is not None
        or config.get("renorm_after_stabilizer")
    )


def apply_effnet_input_stabilizer(ecg_ct, config):
    """Apply optional diagnostic input stabilizer to EfficientNet ECG tensors.

    The standard EfficientNet PN2021-C evaluator remains unchanged unless one
    of the explicit stabilizer flags is enabled.  Inputs are channel-time ECG
    tensors with shape ``(C, T)`` or batched ``(B, C, T)``.
    """
    config = dict(config or {})
    if not _has_input_stabilizer(config):
        return ecg_ct
    squeeze = False
    x = ecg_ct
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    if x.ndim != 3:
        raise ValueError(f"expected ECG tensor with shape (C,T) or (B,C,T), got {tuple(ecg_ct.shape)}")
    y = stabilize_ecg_torch(
        x,
        sample_rate_hz=float(config.get("sample_rate_hz", 100.0)),
        bandpass_low_hz=config.get("bandpass_low_hz"),
        bandpass_high_hz=config.get("bandpass_high_hz"),
        repair_flat_leads=bool(config.get("repair_flat_leads", False)),
        clip_abs=config.get("clip_abs"),
    )
    if bool(config.get("renorm_after_stabilizer", False)):
        y = global_zscore_torch(y)
    if squeeze:
        y = y.squeeze(0)
    return y.to(dtype=ecg_ct.dtype)


def _center_crop_ct(ecg_ct, crop_len):
    if crop_len and int(crop_len) < ecg_ct.shape[-1]:
        start = max((ecg_ct.shape[-1] - int(crop_len)) // 2, 0)
        return ecg_ct[..., start:start + int(crop_len)]
    return ecg_ct


def _cache_path(cache_dir, scheme, center, corruption, severity, cache_version=None):
    return corruption_cache_path(
        cache_dir,
        scheme,
        center,
        corruption,
        severity,
        cache_version or PN2021_C_CACHE_VERSION,
    )


def _clean_mmap_path(cache_dir, scheme, center):
    return clean_mmap_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _clean_npz_path(cache_dir, scheme, center):
    return clean_npz_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _load_metadata(data):
    return load_npz_metadata(data)


def _stable_seed(base_seed, *parts):
    return stable_corruption_seed(base_seed, *parts)


class StreamingCorruptedPN2021Dataset(Dataset):
    def __init__(self, signals, labels, corruption, public_severity,
                 seed=20260501, crop_len=250, severity_profile="standard",
                 indices=None, input_stabilizer_config=None):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = (
            np.arange(len(signals), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.severity_profile = severity_profile
        self.seed = int(seed)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sample_seed = _stable_seed(self.seed, self.corruption, self.public_severity, real_idx)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        op = _build_corruption_op(
            self.corruption, self.public_severity, self.severity_profile
        )

        sig_tc = self.signals[real_idx]
        stage = self.input_stabilizer_config.get("stage", "post_crop")
        if stage == "pre_crop":
            sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        else:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            crop = sig_tc[start:start + self.crop_len]
            sig_ct = torch.from_numpy(np.ascontiguousarray(crop.T)).float()
        corrupt_ct = op(sig_ct)
        corrupt_ct = apply_effnet_input_stabilizer(
            corrupt_ct,
            self.input_stabilizer_config,
        )
        if stage == "pre_crop":
            corrupt_ct = _center_crop_ct(corrupt_ct, self.crop_len)
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


class PN2021IndexedCenterDataset(Dataset):
    def __init__(self, signals, labels, indices, crop_len=250, input_stabilizer_config=None):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = crop_len
        self.input_stabilizer_config = dict(input_stabilizer_config or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        stage = self.input_stabilizer_config.get("stage", "post_crop")
        if stage == "pre_crop":
            sig_ct_np = sig_tc.T
        else:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            crop = sig_tc[start:start + self.crop_len]
            sig_ct_np = crop.T
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_ct_np)).float()
        sig_ct = apply_effnet_input_stabilizer(
            sig_ct,
            self.input_stabilizer_config,
        )
        if stage == "pre_crop":
            sig_ct = _center_crop_ct(sig_ct, self.crop_len)
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return (
            sig_ct.float(),
            torch.from_numpy(label).float(),
        )


def _filter_indices(record_ids, exclude_ids, limit=None):
    return filter_record_indices(record_ids, exclude_ids, limit)


def _load_clean_center(args, center):
    mmap_root = _clean_mmap_path(args.clean_mmap_cache_dir, args.scheme, center)
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
    npz_path = _clean_npz_path(args.clean_cache_dir, args.scheme, center)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"clean cache not found for {center}: {mmap_root} or {npz_path}"
        )
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["signals"].astype(np.float32, copy=False),
        data["labels"].astype(np.float32, copy=False),
        data["record_ids"].astype(str),
        _load_metadata(data),
        "npz",
    )


def _load_model(args, scheme, device):
    model_name = normalize_model_name(args.model_name)
    model = build_super5_model(
        model_name,
        num_classes=scheme["num_classes"],
    ).to(device)
    ckpt = os.path.join(args.model_dir, "best_model.pt")
    sd = torch.load(ckpt, map_location=device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {ckpt} ({model_name})")
    return model


def _clean_lookup(clean_eval_json):
    return load_clean_metric_lookup(clean_eval_json)


def _merge_clean_metadata(clean_cache_metadata, clean_eval_payload):
    metadata = dict(clean_cache_metadata)
    if clean_eval_payload:
        if "label_mapping" in clean_eval_payload:
            metadata["label_mapping"] = clean_eval_payload["label_mapping"]
        if "preprocess" in clean_eval_payload:
            metadata["preprocess"] = clean_eval_payload["preprocess"]
    return metadata


def _ref_ids_sha256(ids):
    values = sorted(str(item) for item in ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _compute_clean_subset_metrics(model, scheme, args, device, signals, labels, indices):
    ds = PN2021IndexedCenterDataset(
        signals,
        labels,
        indices,
        crop_len=args.crop_len,
        input_stabilizer_config=_input_stabilizer_config(args),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    y_true, y_score = infer_dataset(model, loader, device)
    metrics = compute_macro_auroc_auprc(
        y_true, y_score, scheme["class_names"], min_pos=args.min_pos
    )
    nonzero_mask = np.asarray(y_true).sum(axis=1) > 0
    drop_all_zero_metrics = None
    if np.any(nonzero_mask):
        drop_all_zero_metrics = compute_macro_auroc_auprc(
            y_true[nonzero_mask],
            y_score[nonzero_mask],
            scheme["class_names"],
            min_pos=args.min_pos,
        )
    return {
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "drop_all_zero_n_records": int(nonzero_mask.sum()),
        "drop_all_zero_macro_auroc": (
            drop_all_zero_metrics["macro_auroc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_macro_auprc": (
            drop_all_zero_metrics["macro_auprc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_n_classes_used": (
            drop_all_zero_metrics["n_classes_used"]
            if drop_all_zero_metrics is not None
            else 0
        ),
        "drop_all_zero_per_class": (
            drop_all_zero_metrics["per_class"]
            if drop_all_zero_metrics is not None
            else {}
        ),
        "clean_metric_source": "same_filtered_subset",
    }


def _get_clean_subset_metrics(model, scheme, args, device, center, signals, labels, indices):
    cache = getattr(args, "clean_subset_metrics_by_center", None)
    if cache is None:
        cache = {}
        args.clean_subset_metrics_by_center = cache
    if center not in cache:
        cache[center] = _compute_clean_subset_metrics(
            model, scheme, args, device, signals, labels, indices
        )
    return cache[center]


def eval_one(model, scheme, args, device, center, corruption, severity, clean_by_center, clean_eval_payload):
    path = None
    metadata = {}
    cache_source = args.mode
    exclude_ids = getattr(args, "exclude_ref_ids_by_center", {}).get(center, set())
    if args.mode == "stream":
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
        cache_source = f"stream:{clean_kind}"
        indices = _filter_indices(record_ids, exclude_ids, args.limit)
        ds = StreamingCorruptedPN2021Dataset(
            signals, labels, corruption, severity,
            seed=args.seed, crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
            input_stabilizer_config=_input_stabilizer_config(args),
        )
    else:
        if args.severity_profile != "standard":
            raise ValueError(
                "cache mode only supports --severity_profile standard because "
                "prebuilt PN2021-C caches encode the standard profile"
            )
        path = _cache_path(
            args.cache_dir,
            args.scheme,
            center,
            corruption,
            severity,
            args.required_cache_version,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing PN2021-C cache: {path}")
        data = np.load(path, allow_pickle=True)
        signals = data["signals"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        record_ids = data["record_ids"].astype(str)
        indices = _filter_indices(record_ids, exclude_ids, args.limit)
        signals = signals[indices]
        labels = labels[indices]
        ds = PN2021CachedCenterDataset(signals, labels, crop_len=args.crop_len)
        metadata = _load_metadata(data)
    if clean_eval_payload:
        clean_eval_for_validation = clean_eval_payload
        if args.mode == "stream":
            clean = _get_clean_subset_metrics(
                model, scheme, args, device, center, signals, labels, indices
            )
            clean_eval_for_validation = build_center_scoped_clean_eval_payload(
                _merge_clean_metadata(metadata, clean_eval_payload),
                center=center,
                n_excluded_ref=len(exclude_ids),
                ref_record_ids_sha256=_ref_ids_sha256(exclude_ids),
                preprocess_contract_id=PREPROCESS_CONTRACT_ID,
                preprocess_mode="minimal_resample",
                norm_mode="per_sample_global",
                crop_len=args.crop_len,
                target_len=1000,
                clean_metrics=clean,
            )
            metadata = build_pn2021c_metadata_payload(
                clean_metadata=clean_eval_for_validation,
                center=center,
                corruption=corruption,
                public_severity=int(severity),
                internal_severity=int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)]),
                cache_version=args.required_cache_version,
                n_excluded_ref=len(exclude_ids),
                preprocess_contract_id=PREPROCESS_CONTRACT_ID,
                ref_record_ids_sha256=_ref_ids_sha256(exclude_ids),
                source_clean_cache=cache_source,
                seed=int(args.seed),
                limit=args.limit,
                severity_profile=args.severity_profile,
            )
        compatibility = validate_pn2021c_metadata_compatibility(
            clean_eval=clean_eval_for_validation,
            corrupt_metadata=metadata,
            center=center,
            required_cache_version=args.required_cache_version,
        )
        if int(len(exclude_ids)) != int(compatibility["n_excluded_ref"]):
            raise PN2021CMetadataError(
                "n_excluded_ref mismatch for actual PN2021-C eval: "
                f"clean={compatibility['n_excluded_ref']!r}, "
                f"exclude_ref_ids={len(exclude_ids)!r}, center={center}"
            )
        actual_ref_hash = _ref_ids_sha256(exclude_ids)
        if actual_ref_hash != compatibility["ref_record_ids_sha256"]:
            raise PN2021CMetadataError(
                "ref_record_ids_sha256 mismatch for actual PN2021-C eval: "
                f"clean={compatibility['ref_record_ids_sha256']!r}, "
                f"exclude_ref_ids={actual_ref_hash!r}, center={center}"
            )
    else:
        compatibility = {
            "compatible": None,
            "reason": "diagnostic_without_clean",
            "center": center,
            "cache_version": args.required_cache_version,
        }
        clean = clean_by_center.get(center, {})
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    t0 = time.time()
    y_true, y_score = infer_dataset(model, loader, device)
    metrics = compute_macro_auroc_auprc(
        y_true, y_score, scheme["class_names"], min_pos=args.min_pos
    )
    nonzero_mask = np.asarray(y_true).sum(axis=1) > 0
    drop_all_zero_metrics = None
    if np.any(nonzero_mask):
        drop_all_zero_metrics = compute_macro_auroc_auprc(
            y_true[nonzero_mask],
            y_score[nonzero_mask],
            scheme["class_names"],
            min_pos=args.min_pos,
        )
    if args.mode != "stream":
        clean = clean_by_center.get(center, {})
    auroc_drop = None
    auprc_drop = None
    if clean.get("macro_auroc") is not None:
        auroc_drop = float(clean["macro_auroc"] - metrics["macro_auroc"])
    if clean.get("macro_auprc") is not None:
        auprc_drop = float(clean["macro_auprc"] - metrics["macro_auprc"])

    print(
        f"  {center:<18} {corruption:<22} s{severity} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f} "
        f"drop=({auroc_drop if auroc_drop is not None else float('nan'):.4f}, "
        f"{auprc_drop if auprc_drop is not None else float('nan'):.4f}) "
        f"({time.time() - t0:.0f}s)"
    )
    return {
        "cache_path": path,
        "cache_source": cache_source,
        "metadata": metadata,
        "metadata_compatibility": compatibility,
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "corruption": {
            "name": corruption,
            "severity_profile": args.severity_profile,
            "public_severity": int(severity),
            "internal_severity": int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)]),
            "seed": int(args.seed),
        },
        "n_records": int(len(ds)),
        "n_all_zero_labels": int((~nonzero_mask).sum()),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "drop_all_zero_n_records": int(nonzero_mask.sum()),
        "drop_all_zero_macro_auroc": (
            drop_all_zero_metrics["macro_auroc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_macro_auprc": (
            drop_all_zero_metrics["macro_auprc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_n_classes_used": (
            drop_all_zero_metrics["n_classes_used"]
            if drop_all_zero_metrics is not None
            else 0
        ),
        "drop_all_zero_per_class": (
            drop_all_zero_metrics["per_class"]
            if drop_all_zero_metrics is not None
            else {}
        ),
        "clean_metric_source": clean.get("clean_metric_source", "clean_eval_json"),
        "clean_macro_auroc": clean.get("macro_auroc"),
        "clean_macro_auprc": clean.get("macro_auprc"),
        "auroc_drop_vs_clean": auroc_drop,
        "auprc_drop_vs_clean": auprc_drop,
    }


def _aggregate(results):
    return aggregate_corruption_summary(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="super5", choices=["super5", "sub23", "pn26"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name", default="efficientnet1dv2",
                   choices=available_model_names())
    p.add_argument("--mode", default="stream", choices=["stream", "cache"],
                   help="stream: corrupt clean cache on the fly; cache: read prebuilt PN2021-C npz files")
    p.add_argument("--cache_dir", default="/root/autodl-tmp/triple_labels/pn2021_c_cache")
    p.add_argument("--clean_mmap_cache_dir",
                   default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap")
    p.add_argument("--clean_cache_dir",
                   default="/root/autodl-tmp/triple_labels/pn2021_eval_cache")
    p.add_argument("--clean_eval_json", default=None)
    p.add_argument("--diagnostic_without_clean", action="store_true")
    p.add_argument("--required_cache_version", default=PN2021_C_CACHE_VERSION)
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--severity_profile", default="standard",
                   choices=STRESS_PROFILE_CHOICES,
                   help="standard uses methods/augmix/severity.py via public "
                        "severity 1..5 -> internal 2/4/6/8/10; stress_v2 is "
                        "a stronger streaming-only robustness sweep; "
                        "calibrated_10to20pp is a fixed severity=5 profile "
                        "calibrated to induce roughly 10-20 pp drops. These "
                        "profiles do not affect training-time AugMix defaults.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop_len", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument(
        "--input_bandpass_low_hz",
        type=float,
        default=None,
        help="Optional diagnostic bandpass low cutoff applied after crop/corruption before EfficientNet inference.",
    )
    p.add_argument(
        "--input_bandpass_high_hz",
        type=float,
        default=None,
        help="Optional diagnostic bandpass high cutoff applied after crop/corruption before EfficientNet inference.",
    )
    p.add_argument(
        "--input_repair_flat_leads",
        action="store_true",
        help="Optionally repair flat/masked ECG leads before EfficientNet inference.",
    )
    p.add_argument(
        "--input_clip_abs",
        type=float,
        default=None,
        help="Optional absolute clipping after diagnostic input stabilization.",
    )
    p.add_argument(
        "--input_renorm_after_stabilizer",
        action="store_true",
        help="Apply per-sample global z-score after diagnostic input stabilization.",
    )
    p.add_argument(
        "--input_sample_rate_hz",
        type=float,
        default=100.0,
        help="Sample rate for diagnostic input stabilization filters.",
    )
    p.add_argument(
        "--input_stabilizer_stage",
        choices=["post_crop", "pre_crop"],
        default="post_crop",
        help=(
            "Apply diagnostic input stabilization after the normal center crop "
            "or on the full 1000-point signal before center cropping."
        ),
    )
    p.add_argument(
        "--exclude_ref_ids",
        nargs="*",
        default=[],
        help=(
            "Optional K-shot ref-meta JSON file(s). Records listed under "
            "ref_record_ids are excluded from the matching center before "
            "corruption evaluation."
        ),
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N records per cache for smoke tests.")
    p.add_argument("--output_path", default=None)
    args = p.parse_args()
    try:
        require_clean_eval_json(
            args.clean_eval_json,
            diagnostic_without_clean=args.diagnostic_without_clean,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    clean_eval_payload = load_json_payload(args.clean_eval_json)

    device = torch.device(args.device)
    scheme = get_scheme(args.scheme)
    model = _load_model(args, scheme, device)
    clean_by_center = _clean_lookup(args.clean_eval_json)
    args.exclude_ref_ids_by_center = load_ref_record_ids_by_center(args.exclude_ref_ids)
    if args.exclude_ref_ids_by_center:
        print(
            "[exclude] "
            + ", ".join(
                f"{center}:{len(ids)}"
                for center, ids in sorted(args.exclude_ref_ids_by_center.items())
            )
        )

    output = {
        "scheme": args.scheme,
        "model_dir": args.model_dir,
        "model_name": normalize_model_name(args.model_name),
        "mode": args.mode,
        "cache_dir": args.cache_dir,
        "clean_mmap_cache_dir": args.clean_mmap_cache_dir,
        "clean_cache_dir": args.clean_cache_dir,
        "clean_eval_json": args.clean_eval_json,
        "diagnostic_without_clean": bool(args.diagnostic_without_clean),
        "required_cache_version": args.required_cache_version,
        "centers": list(args.centers),
        "corruptions": list(args.corruptions),
        "severities": list(args.severities),
        "severity_profile": args.severity_profile,
        "exclude_ref_ids": list(args.exclude_ref_ids),
        "exclude_ref_ids_by_center_counts": {
            center: len(ids)
            for center, ids in sorted(args.exclude_ref_ids_by_center.items())
        },
        "pn2021_c_cache_version": args.required_cache_version,
        "per_center": {},
    }
    output["preprocess"] = {
        "contract_id": PREPROCESS_CONTRACT_ID,
        "preprocess_mode": "minimal_resample",
        "norm_mode": "per_sample_global",
        "crop_len": int(args.crop_len),
        "target_fs": 100,
        "target_len": 1000,
        "diagnostic_input_stabilizer": _input_stabilizer_config(args),
    }
    if args.scheme == "super5":
        mapping_metadata = get_super5_pn2021_mapping_metadata()
        output["label_mapping"] = {
            "version": mapping_metadata.get("mapping_version"),
            "hash": mapping_metadata.get("mapping_hash"),
            "pn2021_super5": mapping_metadata,
        }

    for center in args.centers:
        output["per_center"].setdefault(center, {})
        for corruption in args.corruptions:
            output["per_center"][center].setdefault(corruption, {})
            for severity in args.severities:
                output["per_center"][center][corruption][str(severity)] = eval_one(
                    model,
                    scheme,
                    args,
                    device,
                    center,
                    corruption,
                    severity,
                    clean_by_center,
                    clean_eval_payload,
                )

    output["aggregate_by_corruption_severity"] = _aggregate(output)

    out_path = args.output_path or os.path.join(args.model_dir, "eval_pn2021_c.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
