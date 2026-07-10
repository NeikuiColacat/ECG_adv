#!/usr/bin/env python3
"""Evaluate ECGFounder PN2021-C corruptions.

This evaluator supports historical frozen-feature heads and the locked full
fine-tuning mainline. It keeps the PN2021-C aggregation contract while routing
corrupted ECGs through the ECGFounder 500 Hz input path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        "/home/linbinhao/ECG_adv_data",
    )
)
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
for path in (ECGFOUNDER_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from physionet2021_dataset import TARGET_POINTS  # noqa: E402

from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID  # noqa: E402
from ecg_adv_gen.data.kshot import load_ref_record_ids_by_center  # noqa: E402
from ecg_adv_gen.data.waveform_datasets import (  # noqa: E402
    ECGFounderBottleneck5000CleanDataset,
    ECGFounderBottleneck5000CorruptedDataset,
    ECGFounderCleanDataset,
    ECGFounderStreamingCorruptedDataset,
    NativeRawFirstCleanPN2021Dataset,
    NativeRawFirstCorruptedPN2021Dataset,
    RawFirstCleanPN2021Dataset,
    RawFirstCorruptedPN2021Dataset,
)
from ecg_adv_gen.data.pn2021 import (  # noqa: E402
    load_clean_center as _load_clean_center,
    load_native_raw_first_center as _load_native_raw_first_center,
    load_raw_first_center as _load_raw_first_center,
)
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    aggregate_corruption_summary,
    filter_record_indices,
    ref_ids_sha256,
)
from ecg_adv_gen.evaluation.pn2021_metric_views import (  # noqa: E402
    select_center_clean_metric_row,
)
from ecg_adv_gen.evaluation.pn2021c_metadata import (  # noqa: E402
    build_pn2021c_metadata_payload,
    validate_target_init_k500_identity,
)
from ecg_adv_gen.evaluation.pn2021c_protocol import (  # noqa: E402
    LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
    ecgfounder_pn2021c_corruption_order,
    locked_protocol_metadata,
)
from ecg_adv_gen.labels import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_super5_scheme,
    pn2021_super5_label_mapping_payload,
)
from ecg_adv_gen.models.ecgfounder_inference import (  # noqa: E402
    infer_ecgfounder,
    infer_ecgfounder_fullft,
)
from ecg_adv_gen.models.ecgfounder_runtime import (  # noqa: E402
    build_ecgfounder_feature_model,
    build_ecgfounder_fullft_model,
    build_ecgfounder_head,
    detect_eval_mode,
    fullft_model_path,
)
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc, summarize_center_view  # noqa: E402
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    PN2021C_DEFAULT_CENTERS as DEFAULT_CENTERS,
    PN2021C_DEFAULT_CORRUPTIONS as DEFAULT_CORRUPTIONS,
    PN2021_C_CACHE_VERSION,
    PUBLIC_TO_INTERNAL_SEVERITY,
    STRESS_PROFILE_CHOICES,
    resolve_corruption_profile_params as _resolve_corruption_profile_params,
    resolve_severity_profile_args as _resolve_severity_profile_args,
    severity_profile_metadata as _severity_profile_metadata,
)


CHECKPOINT = ECGFOUNDER_ROOT / "checkpoint/12_lead_ECGFounder.pth"


def _pn2021_scheme(args: argparse.Namespace) -> dict[str, Any]:
    if args.scheme != "super5":
        raise ValueError(f"unsupported ECGFounder PN2021-C scheme: {args.scheme!r}")
    return get_super5_scheme()


def _load_result(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with result_path.open() as f:
        result = json.load(f)
    if "center" not in result:
        raise ValueError(f"{result_path} does not record center")
    init_model = result.get("init_model") or {}
    init_path = init_model.get("path") if isinstance(init_model, dict) else None
    if init_path:
        init_result_path = Path(str(init_path)).parent / "eval_result.json"
        if not init_result_path.is_file():
            raise ValueError(f"initialization checkpoint does not expose eval_result.json: {init_result_path}")
        with init_result_path.open() as f:
            init_result = json.load(f)
        if not isinstance(init_result, dict):
            raise ValueError(f"initialization eval_result.json root must be an object: {init_result_path}")
        validate_target_init_k500_identity(result, init_result)
    return result


def _excluded_ref_ids_for_center(
    result: dict[str, Any],
    center: str,
    min_target_ref_excluded: int = 0,
    record_ids: Any = None,
    evaluation_ref_ids: set[str] | None = None,
) -> set[str]:
    if center != str(result["center"]):
        return set()
    selected_ids = {str(x) for x in result.get("selected_ref_record_ids", [])}
    exclude_ids = selected_ids if evaluation_ref_ids is None else {str(x) for x in evaluation_ref_ids}
    if exclude_ids != selected_ids:
        raise ValueError(
            f"training/evaluation K500 identity mismatch for {center}: "
            f"training={len(selected_ids)} ids, evaluation={len(exclude_ids)} ids"
        )
    matched_ids = exclude_ids
    if record_ids is not None:
        matched_ids = selected_ids.intersection(set(np.asarray(record_ids).astype(str)))
    if len(matched_ids) < int(min_target_ref_excluded or 0):
        raise ValueError(
            "eval_result.json selected_ref_record_ids has "
            f"{len(selected_ids)} ids for {center}; matched {len(matched_ids)} loaded records; "
            f"expected at least {int(min_target_ref_excluded)} for K-shot ref exclusion"
        )
    return exclude_ids


def _input_stabilizer_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.ecgfounder_input_bandpass_low_hz is not None:
        kwargs["bandpass_low_hz"] = float(args.ecgfounder_input_bandpass_low_hz)
    if args.ecgfounder_input_bandpass_high_hz is not None:
        kwargs["bandpass_high_hz"] = float(args.ecgfounder_input_bandpass_high_hz)
    if bool(args.ecgfounder_input_repair_flat_leads):
        kwargs["repair_flat_leads"] = True
    if args.ecgfounder_input_clip_abs is not None:
        kwargs["clip_abs"] = float(args.ecgfounder_input_clip_abs)
    return kwargs


def eval_clean_center(
    eval_mode: str,
    model_or_feature_model: nn.Module,
    head: nn.Module | None,
    result: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    center: str,
    *,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
    input_already_ecgfounder: bool = False,
) -> dict[str, Any]:
    sample_rates = None
    if args.corruption_input == "native_raw_first":
        (
            signals,
            labels,
            record_ids,
            metadata,
            clean_kind,
            sample_rates,
        ) = _load_native_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == "raw_first":
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT:
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    else:
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
    exclude_ids = _excluded_ref_ids_for_center(
        result,
        center,
        getattr(args, "min_target_ref_excluded", 0),
        record_ids,
        evaluation_ref_ids=getattr(args, "exclude_ref_ids_by_center", {}).get(center),
    )
    indices = filter_record_indices(record_ids, exclude_ids, args.limit)
    if args.corruption_input == "native_raw_first":
        ds = NativeRawFirstCleanPN2021Dataset(
            signals,
            labels,
            sample_rates,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config={},
        )
    elif args.corruption_input == "raw_first":
        ds = RawFirstCleanPN2021Dataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config={},
        )
    elif args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT:
        ds = ECGFounderBottleneck5000CleanDataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
            target_points=TARGET_POINTS,
        )
    else:
        ds = ECGFounderCleanDataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if eval_mode == "feature_head":
        if head is None:
            raise RuntimeError("feature_head evaluation requires a head module")
        y_true, y_score = infer_ecgfounder(
            model_or_feature_model,
            head,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
            input_already_ecgfounder=input_already_ecgfounder,
            target_points=TARGET_POINTS,
        )
    elif eval_mode == "fullft_model":
        y_true, y_score = infer_ecgfounder_fullft(
            model_or_feature_model,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
            input_already_ecgfounder=input_already_ecgfounder,
            target_points=TARGET_POINTS,
        )
    else:
        raise ValueError(f"unknown ECGFounder eval mode: {eval_mode}")
    metrics = compute_macro_auroc_auprc(
        y_true,
        y_score,
        CLASS_NAMES_SUPER5,
        min_pos=args.min_pos,
    )
    print(
        f"  {center:<18} clean{'':<22} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f}",
        flush=True,
    )
    return {
        "cache_source": f"stream:{clean_kind}",
        "metadata": metadata,
        "n_records": int(len(ds)),
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "ref_record_ids_sha256": ref_ids_sha256(exclude_ids),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
    }


def eval_one(
    eval_mode: str,
    model_or_feature_model: nn.Module,
    head: nn.Module | None,
    result: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    center: str,
    corruption: str,
    severity: int,
    *,
    clean_metric_override: dict[str, Any] | None = None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
    input_already_ecgfounder: bool = False,
) -> dict[str, Any]:
    sample_rates = None
    if args.corruption_input == "native_raw_first":
        (
            signals,
            labels,
            record_ids,
            metadata,
            clean_kind,
            sample_rates,
        ) = _load_native_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == "raw_first":
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT:
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    else:
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
    exclude_ids = _excluded_ref_ids_for_center(
        result,
        center,
        getattr(args, "min_target_ref_excluded", 0),
        record_ids,
        evaluation_ref_ids=getattr(args, "exclude_ref_ids_by_center", {}).get(center),
    )
    indices = filter_record_indices(record_ids, exclude_ids, args.limit)
    if args.corruption_input == "native_raw_first":
        ds = NativeRawFirstCorruptedPN2021Dataset(
            signals,
            labels,
            sample_rates,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
            input_stabilizer_config={},
            severity_profile_params=getattr(args, "severity_profile_params", None),
        )
    elif args.corruption_input == "raw_first":
        ds = RawFirstCorruptedPN2021Dataset(
            signals,
            labels,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
            input_stabilizer_config={},
            severity_profile_params=getattr(args, "severity_profile_params", None),
        )
    elif args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT:
        ds = ECGFounderBottleneck5000CorruptedDataset(
            signals,
            labels,
            indices,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            severity_profile_params=getattr(args, "severity_profile_params", None),
            target_points=TARGET_POINTS,
        )
    else:
        ds = ECGFounderStreamingCorruptedDataset(
            signals,
            labels,
            indices,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            severity_profile_params=getattr(args, "severity_profile_params", None),
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    t0 = time.time()
    if eval_mode == "feature_head":
        if head is None:
            raise RuntimeError("feature_head evaluation requires a head module")
        y_true, y_score = infer_ecgfounder(
            model_or_feature_model,
            head,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
            input_already_ecgfounder=input_already_ecgfounder,
            target_points=TARGET_POINTS,
        )
    elif eval_mode == "fullft_model":
        y_true, y_score = infer_ecgfounder_fullft(
            model_or_feature_model,
            loader,
            device,
            operator_name=corruption,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
            input_already_ecgfounder=input_already_ecgfounder,
            target_points=TARGET_POINTS,
        )
    else:
        raise ValueError(f"unknown ECGFounder eval mode: {eval_mode}")
    metrics = summarize_center_view(
        y_true,
        y_score,
        metric_fn=lambda yt, ys: compute_macro_auroc_auprc(
            yt,
            ys,
            CLASS_NAMES_SUPER5,
            min_pos=args.min_pos,
        ),
        n_total=len(ds),
        report_drop_all_zero=True,
        drop_none_if_empty=True,
    )
    clean = (
        clean_metric_override
        if clean_metric_override is not None
        else select_center_clean_metric_row(result, center)
    )
    if clean_metric_override is not None:
        clean_metric_source = (
            "recomputed_native_raw_first_clean"
            if args.corruption_input == "native_raw_first"
            else "recomputed_raw_first_clean"
            if args.corruption_input == "raw_first"
            else "recomputed_bottleneck5000_clean"
            if args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT
            else "recomputed_input_stabilizer"
        )
    else:
        clean_metric_source = "saved_eval_result"
    auroc_drop = None
    auprc_drop = None
    if clean.get("macro_auroc") is not None:
        auroc_drop = float(clean["macro_auroc"] - metrics["macro_auroc"])
    if clean.get("macro_auprc") is not None:
        auprc_drop = float(clean["macro_auprc"] - metrics["macro_auprc"])
    result_metadata = dict(metadata)
    result_protocol = (
        locked_protocol_metadata("ecgfounder")
        if args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT
        else None
    )
    ref_hash = ref_ids_sha256(exclude_ids)
    result_metadata = build_pn2021c_metadata_payload(
        clean_metadata=metadata,
        center=center,
        corruption=corruption,
        public_severity=severity,
        internal_severity=PUBLIC_TO_INTERNAL_SEVERITY[int(severity)],
        cache_version=args.required_cache_version,
        n_excluded_ref=len(exclude_ids),
        preprocess_contract_id=PREPROCESS_CONTRACT_ID,
        ref_record_ids_sha256=ref_hash,
        source_clean_cache=f"stream:{clean_kind}",
        seed=args.seed,
        limit=args.limit,
        severity_profile=args.severity_profile,
        protocol_metadata=result_protocol,
    )
    print(
        f"  {center:<18} {corruption:<22} s{severity} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f} "
        f"drop=({auroc_drop if auroc_drop is not None else float('nan'):.4f}, "
        f"{auprc_drop if auprc_drop is not None else float('nan'):.4f}) "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {
        "cache_path": None,
        "cache_source": f"stream:{clean_kind}",
        "metadata": result_metadata,
        "metadata_compatibility": {
            "compatible": None,
            "reason": "ecgfounder_clean_eval_result",
            "cache_version": result_metadata["pn2021c"]["cache_version"],
            "preprocess_contract_id": result_metadata["preprocess"]["contract_id"],
            "input_order_id": result_metadata["pn2021c"].get("input_order_id"),
            "ref_record_ids_sha256": ref_hash,
        },
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "ref_record_ids_sha256": ref_hash,
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
        "clean_macro_auroc": clean.get("macro_auroc"),
        "clean_macro_auprc": clean.get("macro_auprc"),
        "clean_metric_source": clean_metric_source,
        "auroc_drop_vs_clean": auroc_drop,
        "auprc_drop_vs_clean": auprc_drop,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="super5", choices=["super5"])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--variant", default="")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument(
        "--clean_mmap_cache_dir",
        default=str(DATA_ROOT / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
    )
    p.add_argument(
        "--clean_cache_dir",
        default=str(DATA_ROOT / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
    )
    p.add_argument(
        "--corruption_input",
        default="preprocessed_cache",
        choices=["preprocessed_cache", "raw_first", "native_raw_first", LOCKED_ECGFOUNDER_CORRUPTION_INPUT],
        help=(
            "preprocessed_cache keeps historical behavior: corrupt the "
            "100Hz/1000 per-sample-z-scored clean cache and let ECGFounder "
            "input conversion z-score again. raw_first reads WFDB records, "
            "resamples/pads without z-score, applies corruption, z-scores once, "
            "then only resamples to ECGFounder length. native_raw_first applies "
            "corruption on WFDB native-fs/native-length signals after only lead "
            "reorder and NaN repair, then runs model preprocessing. "
            "bottleneck5000 is the locked main protocol: raw -> 100Hz/1000 "
            "without z-score -> ECGFounder 500Hz/5000 interpolation -> "
            "corruption -> z-score -> model."
        ),
    )
    p.add_argument(
        "--pn2021_root",
        default=str(DATA_ROOT / "physionet2021"),
        help="PN2021 root containing training/<center>; used by --corruption_input raw_first.",
    )
    p.add_argument("--required_cache_version", default=PN2021_C_CACHE_VERSION)
    p.add_argument("--centers", nargs="+", default=None)
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--severity_profile", default="standard", choices=STRESS_PROFILE_CHOICES)
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
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min_target_ref_excluded", type=int, default=0)
    p.add_argument("--exclude_ref_ids", nargs="+", default=[])
    p.add_argument("--ecgfounder_input_bandpass_low_hz", type=float, default=None)
    p.add_argument("--ecgfounder_input_bandpass_high_hz", type=float, default=None)
    p.add_argument("--ecgfounder_input_repair_flat_leads", action="store_true")
    p.add_argument("--ecgfounder_input_clip_abs", type=float, default=None)
    p.add_argument(
        "--recompute_clean_with_input_stabilizer",
        action="store_true",
        help="Recompute clean PN2021 metrics through the same ECGFounder input stabilizer before reporting drops.",
    )
    p.add_argument("--output_path", default=None)
    args = p.parse_args()
    try:
        args.severity_profile_params = _resolve_severity_profile_args(args)
        args.severity_profile_metadata = _severity_profile_metadata(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = Path(args.run_dir)
    result = _load_result(run_dir)
    args.exclude_ref_ids_by_center = load_ref_record_ids_by_center(args.exclude_ref_ids)
    centers = list(args.centers or [str(result["center"])])
    unknown = sorted(set(centers).difference(DEFAULT_CENTERS))
    if unknown:
        raise ValueError(f"unknown PN2021-C centers: {unknown}")

    device = torch.device(args.device)
    eval_mode = detect_eval_mode(run_dir)
    if eval_mode == "feature_head":
        model_or_feature_model = build_ecgfounder_feature_model(Path(args.checkpoint), device)
        head = build_ecgfounder_head(
            run_dir,
            result,
            device,
            num_classes=len(CLASS_NAMES_SUPER5),
        )
    else:
        model_or_feature_model = build_ecgfounder_fullft_model(
            run_dir,
            Path(args.checkpoint),
            device,
            num_classes=len(CLASS_NAMES_SUPER5),
        )
        head = None
    input_stabilizer_kwargs = _input_stabilizer_kwargs_from_args(args)
    input_already_ecgfounder = args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT
    apply_input_zscore = args.corruption_input not in {
        "raw_first",
        "native_raw_first",
        LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
    }
    clean_metric_overrides: dict[str, dict[str, Any]] = {}
    if (
        args.recompute_clean_with_input_stabilizer
        or args.corruption_input in {"raw_first", "native_raw_first", LOCKED_ECGFOUNDER_CORRUPTION_INPUT}
    ):
        for center in centers:
            clean_metric_overrides[center] = eval_clean_center(
                eval_mode,
                model_or_feature_model,
                head,
                result,
                args,
                device,
                center,
                input_stabilizer_kwargs=input_stabilizer_kwargs,
                apply_input_zscore=apply_input_zscore,
                input_already_ecgfounder=input_already_ecgfounder,
            )

    protocol_metadata = (
        locked_protocol_metadata("ecgfounder")
        if args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT
        else None
    )
    output: dict[str, Any] = {
        "scheme": args.scheme,
        "model_name": "ECGFounder",
        "eval_mode": eval_mode,
        "variant": args.variant,
        "run_dir": str(run_dir),
        "corruption_input": args.corruption_input,
        "pn2021_root": (
            args.pn2021_root
            if args.corruption_input in {"raw_first", "native_raw_first", LOCKED_ECGFOUNDER_CORRUPTION_INPUT}
            else None
        ),
        "clean_eval_json": str(run_dir / "eval_result.json"),
        "head_path": str(run_dir / "best_head.pt") if (run_dir / "best_head.pt").exists() else None,
        "model_path": str(fullft_model_path(run_dir)) if fullft_model_path(run_dir).exists() else None,
        "checkpoint": str(Path(args.checkpoint)),
        "center_from_run": str(result["center"]),
        "centers": centers,
        "corruptions": list(args.corruptions),
        "severities": list(args.severities),
        "severity_profile": args.severity_profile,
        "severity_params_file": args.severity_params_file,
        "severity_params_name": args.severity_params_name,
        "severity_profile_metadata": args.severity_profile_metadata,
        "crop_len": int(args.crop_len),
        "required_cache_version": args.required_cache_version,
        "pn2021_c_cache_version": args.required_cache_version,
        "min_target_ref_excluded": int(args.min_target_ref_excluded),
        "input_stabilizer": {
            "kwargs": dict(input_stabilizer_kwargs),
            "recompute_clean_with_input_stabilizer": bool(args.recompute_clean_with_input_stabilizer),
            "ecgfounder_apply_input_zscore": bool(apply_input_zscore),
        },
        "pn2021c_protocol": protocol_metadata,
        "corruption_order": ecgfounder_pn2021c_corruption_order(args.corruption_input),
        "selected_ref_record_ids_count": int(len(result.get("selected_ref_record_ids", []))),
        "label_mapping": pn2021_super5_label_mapping_payload(),
        "class_names": list(CLASS_NAMES_SUPER5),
        "clean_metric_overrides": clean_metric_overrides,
        "per_center": {},
    }
    for center in centers:
        output["per_center"].setdefault(center, {})
        for corruption in args.corruptions:
            output["per_center"][center].setdefault(corruption, {})
            for severity in args.severities:
                output["per_center"][center][corruption][str(severity)] = eval_one(
                    eval_mode,
                    model_or_feature_model,
                    head,
                    result,
                    args,
                    device,
                    center,
                    corruption,
                    int(severity),
                    clean_metric_override=clean_metric_overrides.get(center),
                    input_stabilizer_kwargs=input_stabilizer_kwargs,
                    apply_input_zscore=apply_input_zscore,
                    input_already_ecgfounder=input_already_ecgfounder,
                )
    output["aggregate_by_corruption_severity"] = aggregate_corruption_summary(output)

    out_path = Path(args.output_path) if args.output_path else run_dir / "eval_pn2021_c_ecgfounder.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[done] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
