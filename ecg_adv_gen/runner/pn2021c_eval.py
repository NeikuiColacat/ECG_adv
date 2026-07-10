"""Evaluate EfficientNet1DV2 on PN2021-C corruption caches.

This script intentionally stays separate from pn2021_clean_eval.py so the clean
PN2021 benchmark remains unchanged.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "model", "DeepECG", "notebooks"))
import ecg_adv_gen.data.waveform_datasets as waveform_datasets  # noqa: E402
from ecg_adv_gen.data.waveform_datasets import (  # noqa: E402
    NativeRawFirstCleanPN2021Dataset,
    NativeRawFirstCorruptedPN2021Dataset,
    RawFirstCleanPN2021Dataset,
    RawFirstCorruptedPN2021Dataset,
    StreamingCorruptedPN2021Dataset,
)
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc, summarize_center_view  # noqa: E402
from ecg_adv_gen.evaluation.inference import infer_dataset  # noqa: E402
from ecg_adv_gen.labels import (  # noqa: E402
    get_super5_scheme,
    get_super5_pn2021_mapping_metadata,
)
from ecg_adv_gen.models.super5_model_zoo import (  # noqa: E402
    available_model_names,
    build_super5_model,
    normalize_model_name,
)
from ecg_adv_gen.data.kshot import load_ref_record_ids_by_center  # noqa: E402
from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID  # noqa: E402
from ecg_adv_gen.data.pn2021 import (  # noqa: E402
    load_clean_center as _load_clean_center,
    load_native_raw_first_center as _load_native_raw_first_center,
    load_raw_first_center as _load_raw_first_center,
)
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    aggregate_corruption_summary,
    corruption_cache_path,
    filter_record_indices,
    load_json_payload,
    load_clean_metric_lookup,
    load_npz_metadata,
    ref_ids_sha256,
    require_clean_eval_json,
)
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    PN2021C_DEFAULT_CENTERS as DEFAULT_CENTERS,
    PN2021C_DEFAULT_CORRUPTIONS as DEFAULT_CORRUPTIONS,
    PN2021_C_CACHE_VERSION,
    PUBLIC_TO_INTERNAL_SEVERITY,
    STRESS_PROFILE_CHOICES,
    load_custom_severity_profile as _load_custom_severity_profile,
    resolve_corruption_profile_params as _resolve_corruption_profile_params,
    resolve_severity_profile_args as _resolve_severity_profile_args,
    severity_profile_metadata as _severity_profile_metadata,
)
from ecg_adv_gen.evaluation.pn2021c_protocol import (  # noqa: E402
    LOCKED_EFFNET_CORRUPTION_INPUT,
    LOCKED_MAIN_INPUT_ORDER_ID,
    locked_protocol_metadata,
)
from ecg_adv_gen.evaluation.pn2021c_metadata import (  # noqa: E402
    PN2021CMetadataError,
    build_center_scoped_clean_eval_payload,
    build_pn2021c_metadata_payload,
    validate_pn2021c_metadata_compatibility,
)


DEFAULT_DATA_ROOT = os.environ.get("ECG_ADV_GEN_DATA_ROOT", "/home/linbinhao/ECG_adv_data")


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


def _cache_path(cache_dir, scheme, center, corruption, severity, cache_version=None):
    return corruption_cache_path(
        cache_dir,
        scheme,
        center,
        corruption,
        severity,
        cache_version or PN2021_C_CACHE_VERSION,
    )


def _filter_indices(record_ids, exclude_ids, limit=None):
    return filter_record_indices(record_ids, exclude_ids, limit)


def _load_model(args, scheme, device):
    model_name = normalize_model_name(args.model_name)
    model = build_super5_model(
        model_name,
        num_classes=scheme["num_classes"],
    ).to(device)
    ckpt = os.path.join(args.model_dir, args.checkpoint_name)
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
        clean_protocol = (clean_eval_payload.get("pn2021") or {}).get("eval_protocol")
        if clean_protocol:
            metadata["pn2021"] = {"eval_protocol": dict(clean_protocol)}
    return metadata


def _compute_clean_subset_metrics(model, scheme, args, device, signals, labels, indices, sample_rates=None):
    if getattr(args, "corruption_input", "preprocessed_cache") == "native_raw_first":
        ds = NativeRawFirstCleanPN2021Dataset(
            signals,
            labels,
            sample_rates,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config=_input_stabilizer_config(args),
        )
    elif getattr(args, "corruption_input", "preprocessed_cache") == "raw_first":
        ds = RawFirstCleanPN2021Dataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config=_input_stabilizer_config(args),
        )
    else:
        ds = waveform_datasets.PN2021IndexedCenterDataset(
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
    metrics = summarize_center_view(
        y_true,
        y_score,
        metric_fn=lambda yt, ys: compute_macro_auroc_auprc(
            yt, ys, scheme["class_names"], min_pos=args.min_pos
        ),
        n_total=len(ds),
        report_drop_all_zero=True,
        drop_none_if_empty=True,
    )
    return {
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "drop_all_zero_n_records": metrics["drop_all_zero_n_records"],
        "drop_all_zero_macro_auroc": metrics["drop_all_zero_macro_auroc"],
        "drop_all_zero_macro_auprc": metrics["drop_all_zero_macro_auprc"],
        "drop_all_zero_n_classes_used": metrics["drop_all_zero_n_classes_used"],
        "drop_all_zero_per_class": metrics["drop_all_zero_per_class"],
        "clean_metric_source": "same_filtered_subset",
    }


def _get_clean_subset_metrics(model, scheme, args, device, center, signals, labels, indices, sample_rates=None):
    cache = getattr(args, "clean_subset_metrics_by_center", None)
    if cache is None:
        cache = {}
        args.clean_subset_metrics_by_center = cache
    if center not in cache:
        cache[center] = _compute_clean_subset_metrics(
            model, scheme, args, device, signals, labels, indices, sample_rates=sample_rates
        )
    return cache[center]


def eval_one(model, scheme, args, device, center, corruption, severity, clean_by_center, clean_eval_payload):
    path = None
    metadata = {}
    cache_source = args.mode
    exclude_ids = getattr(args, "exclude_ref_ids_by_center", {}).get(center, set())
    sample_rates = None
    if args.mode == "stream":
        if args.corruption_input == "native_raw_first":
            (
                signals,
                labels,
                record_ids,
                metadata,
                clean_kind,
                sample_rates,
            ) = _load_native_raw_first_center(args, scheme, center)
        elif args.corruption_input == "raw_first":
            signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
                args, scheme, center
            )
        else:
            signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
        cache_source = f"stream:{clean_kind}"
        indices = _filter_indices(record_ids, exclude_ids, args.limit)
        if args.corruption_input == "native_raw_first":
            ds = NativeRawFirstCorruptedPN2021Dataset(
                signals, labels, sample_rates, corruption, severity,
                seed=args.seed, crop_len=args.crop_len,
                severity_profile=args.severity_profile,
                indices=indices,
                input_stabilizer_config=_input_stabilizer_config(args),
                severity_profile_params=getattr(args, "severity_profile_params", None),
            )
        elif args.corruption_input == "raw_first":
            ds = RawFirstCorruptedPN2021Dataset(
                signals, labels, corruption, severity,
                seed=args.seed, crop_len=args.crop_len,
                severity_profile=args.severity_profile,
                indices=indices,
                input_stabilizer_config=_input_stabilizer_config(args),
                severity_profile_params=getattr(args, "severity_profile_params", None),
            )
        else:
            ds = StreamingCorruptedPN2021Dataset(
                signals, labels, corruption, severity,
                seed=args.seed, crop_len=args.crop_len,
                severity_profile=args.severity_profile,
                indices=indices,
                input_stabilizer_config=_input_stabilizer_config(args),
                severity_profile_params=getattr(args, "severity_profile_params", None),
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
        ds = waveform_datasets.PN2021CachedCenterDataset(signals, labels, crop_len=args.crop_len)
        metadata = load_npz_metadata(data)
    if clean_eval_payload:
        clean_eval_for_validation = clean_eval_payload
        if args.mode == "stream":
            clean = _get_clean_subset_metrics(
                model,
                scheme,
                args,
                device,
                center,
                signals,
                labels,
                indices,
                sample_rates=sample_rates,
            )
            clean_eval_for_validation = build_center_scoped_clean_eval_payload(
                _merge_clean_metadata(metadata, clean_eval_payload),
                center=center,
                n_excluded_ref=len(exclude_ids),
                ref_record_ids_sha256=ref_ids_sha256(exclude_ids),
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
                ref_record_ids_sha256=ref_ids_sha256(exclude_ids),
                source_clean_cache=cache_source,
                seed=int(args.seed),
                limit=args.limit,
                severity_profile=args.severity_profile,
                protocol_metadata=(
                    locked_protocol_metadata("efficientnet1dv2")
                    if args.corruption_input == LOCKED_EFFNET_CORRUPTION_INPUT
                    else None
                ),
            )
        compatibility = validate_pn2021c_metadata_compatibility(
            clean_eval=clean_eval_for_validation,
            corrupt_metadata=metadata,
            center=center,
            required_cache_version=args.required_cache_version,
            required_input_order_id=(
                LOCKED_MAIN_INPUT_ORDER_ID
                if args.corruption_input == LOCKED_EFFNET_CORRUPTION_INPUT
                else None
            ),
        )
        if int(len(exclude_ids)) != int(compatibility["n_excluded_ref"]):
            raise PN2021CMetadataError(
                "n_excluded_ref mismatch for actual PN2021-C eval: "
                f"clean={compatibility['n_excluded_ref']!r}, "
                f"exclude_ref_ids={len(exclude_ids)!r}, center={center}"
            )
        actual_ref_hash = ref_ids_sha256(exclude_ids)
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
    metrics = summarize_center_view(
        y_true,
        y_score,
        metric_fn=lambda yt, ys: compute_macro_auroc_auprc(
            yt, ys, scheme["class_names"], min_pos=args.min_pos
        ),
        n_total=len(ds),
        report_drop_all_zero=True,
        drop_none_if_empty=True,
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
            "severity_params_file": getattr(args, "severity_params_file", None),
            "severity_params_name": getattr(args, "severity_params_name", None),
            "resolved_params": _resolve_corruption_profile_params(
                corruption,
                severity,
                args.severity_profile,
                getattr(args, "severity_profile_params", None),
            ),
            "public_severity": int(severity),
            "internal_severity": int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)]),
            "seed": int(args.seed),
        },
        "n_records": int(len(ds)),
        "n_all_zero_labels": metrics["n_all_zero_labels"],
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "drop_all_zero_n_records": metrics["drop_all_zero_n_records"],
        "drop_all_zero_macro_auroc": metrics["drop_all_zero_macro_auroc"],
        "drop_all_zero_macro_auprc": metrics["drop_all_zero_macro_auprc"],
        "drop_all_zero_n_classes_used": metrics["drop_all_zero_n_classes_used"],
        "drop_all_zero_per_class": metrics["drop_all_zero_per_class"],
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
    p.add_argument("--scheme", default="super5", choices=["super5"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--checkpoint_name", default="best_model.pt")
    p.add_argument("--model_name", default="efficientnet1dv2",
                   choices=available_model_names())
    p.add_argument("--mode", default="stream", choices=["stream", "cache"],
                   help="stream: corrupt clean cache on the fly; cache: read prebuilt PN2021-C npz files")
    p.add_argument(
        "--corruption_input",
        default="preprocessed_cache",
        choices=["preprocessed_cache", "raw_first", "native_raw_first"],
        help=(
            "preprocessed_cache keeps the historical behavior: corrupt the "
            "100Hz/1000 per-sample-z-scored clean cache. raw_first reads WFDB "
            "records, resamples/pads without z-score, applies corruption, then "
            "applies the normal per-sample global z-score before model input. "
            "native_raw_first applies corruption on WFDB native-fs/native-length "
            "signals after only lead reorder and NaN repair, then runs the "
            "normal model preprocessing."
        ),
    )
    p.add_argument(
        "--pn2021_root",
        default=os.path.join(DEFAULT_DATA_ROOT, "physionet2021"),
        help="PN2021 root containing training/<center>; used by --corruption_input raw_first.",
    )
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
                        "calibrated to induce roughly 10-20 pp drops; custom "
                        "loads operator parameters from --severity_params_file. These "
                        "profiles do not affect training-time AugMix defaults.")
    p.add_argument(
        "--severity_params_file",
        default=None,
        help="YAML/JSON profile file used only with --severity_profile custom.",
    )
    p.add_argument(
        "--severity_params_name",
        default=None,
        help="Profile name under top-level profiles used only with --severity_profile custom.",
    )
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
        args.severity_profile_params = _resolve_severity_profile_args(args)
        args.severity_profile_metadata = _severity_profile_metadata(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.mode != "stream" and args.corruption_input in {"raw_first", "native_raw_first"}:
        raise SystemExit("--corruption_input raw_first/native_raw_first requires --mode stream")
    try:
        require_clean_eval_json(
            args.clean_eval_json,
            diagnostic_without_clean=args.diagnostic_without_clean,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    clean_eval_payload = load_json_payload(args.clean_eval_json)

    device = torch.device(args.device)
    scheme = get_super5_scheme()
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
        "corruption_input": args.corruption_input,
        "pn2021_root": (
            args.pn2021_root
            if args.corruption_input in {"raw_first", "native_raw_first"}
            else None
        ),
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
        "severity_params_file": args.severity_params_file,
        "severity_params_name": args.severity_params_name,
        "severity_profile_metadata": args.severity_profile_metadata,
        "exclude_ref_ids": list(args.exclude_ref_ids),
        "exclude_ref_ids_by_center_counts": {
            center: len(ids)
            for center, ids in sorted(args.exclude_ref_ids_by_center.items())
        },
        "pn2021_c_cache_version": args.required_cache_version,
        "pn2021c_protocol": (
            locked_protocol_metadata("efficientnet1dv2")
            if args.corruption_input == LOCKED_EFFNET_CORRUPTION_INPUT
            else None
        ),
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
        "corruption_order": (
            [
                "wfdb_read_native_fs_native_length",
                "lead_reorder_nan_guard_no_resample_no_zscore",
                "corruption_with_native_sample_rate",
                "resample_pad_or_truncate_to_100hz_1000",
                "per_sample_global_zscore",
                "center_crop",
                "model",
            ]
            if args.corruption_input == "native_raw_first"
            else (
                [
                    "wfdb_read",
                    "lead_reorder_nan_guard_resample_pad_no_zscore",
                    "corruption",
                    "per_sample_global_zscore",
                    "center_crop",
                    "model",
                ]
                if args.corruption_input == "raw_first"
                else [
                    "load_100hz1000_per_sample_global_zscore_cache",
                    "center_crop_or_full_signal",
                    "corruption",
                    "model",
                ]
            )
        ),
    }
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
