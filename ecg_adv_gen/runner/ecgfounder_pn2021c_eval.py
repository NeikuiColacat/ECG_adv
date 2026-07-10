#!/usr/bin/env python3
"""Evaluate ECGFounder PN2021-C corruptions.

This evaluator supports historical frozen-feature heads and the locked full
fine-tuning mainline. It keeps the PN2021-C aggregation contract while routing
corrupted ECGs through the ECGFounder 500 Hz input path.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
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
_RESUME_SCHEMA_VERSION = 2
_RESULT_KEYS = tuple(
    "cache_path cache_source metadata metadata_compatibility n_excluded_ref_ids_for_center "
    "ref_record_ids_sha256 corruption n_records n_all_zero_labels macro_auroc macro_auprc "
    "n_classes_used per_class drop_all_zero_n_records drop_all_zero_macro_auroc drop_all_zero_macro_auprc "
    "drop_all_zero_n_classes_used drop_all_zero_per_class clean_macro_auroc clean_macro_auprc "
    "clean_metric_source auroc_drop_vs_clean auprc_drop_vs_clean".split()
)


def _normalize_identity(payload: Any) -> Any:
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("resume identity must contain only finite JSON values") from exc


def _json_sha256(payload: Any, *, finite: bool = False) -> str:
    if finite:
        payload = _normalize_identity(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_paths(output_path: str | Path) -> tuple[Path, Path]:
    output = Path(output_path).expanduser().resolve()
    return output.with_name(f".{output.name}.resume"), output.with_name(f".{output.name}.resume.lock")


def _atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(output_path: str | Path):
    _, lock_path = _resume_paths(output_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"PN2021-C output is already locked: {output_path}") from exc
        yield
    finally:
        os.close(fd)


def _file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    fingerprint = lambda value: {field: int(getattr(value, field)) for field in fields}
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = fingerprint(os.fstat(handle.fileno()))
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = fingerprint(os.fstat(handle.fileno()))
    if before != after or after != fingerprint(path.stat()):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "stat": after}


def _checkpoint_identity(eval_mode: str, run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    if eval_mode == "fullft_model":
        return {"scored_checkpoint": _file_identity(fullft_model_path(run_dir))}
    if eval_mode == "feature_head":
        return {
            "scored_head": _file_identity(run_dir / "best_head.pt"),
            "scored_backbone": _file_identity(checkpoint),
        }
    raise ValueError(f"unknown ECGFounder eval mode: {eval_mode}")


def _build_runtime_with_checkpoint_guard(
    expected: Mapping[str, Any], *, eval_mode: str, run_dir: Path,
    checkpoint: Path, factory: Callable[[], Any],
) -> Any:
    def verify(phase: str) -> None:
        if expected != _checkpoint_identity(eval_mode, run_dir, checkpoint):
            raise RuntimeError(f"checkpoint identity changed {phase} runtime construction")

    verify("before")
    runtime = factory()
    verify("during")
    return runtime


def _unit_key(center: str, corruption: str, severity: int) -> str:
    return json.dumps([str(center), str(corruption), int(severity)], separators=(",", ":"))


def _validate_result(
    result: Any, unit: tuple[str, str, int], run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    center, corruption, severity = unit
    key = _unit_key(*unit)
    if not isinstance(result, dict) or tuple(result) != _RESULT_KEYS:
        raise ValueError(f"invalid result schema for resume unit {key}")
    metadata = result.get("metadata")
    compatibility = result.get("metadata_compatibility")
    corruption_payload = result.get("corruption")
    if not all(isinstance(item, Mapping) for item in (metadata, compatibility, corruption_payload)):
        raise ValueError(f"invalid nested result schema for resume unit {key}")
    pn2021c = metadata.get("pn2021c")
    pn2021c_alias = metadata.get("pn2021_c")
    if not isinstance(pn2021c, Mapping) or not isinstance(pn2021c_alias, Mapping):
        raise ValueError(f"invalid PN2021-C metadata for resume unit {key}")
    expected_ref = run_identity.get("ref_record_ids_sha256_by_center", {}).get(center)
    internal_severity = int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)])
    expected_metadata = {
        "center": center, "corruption": corruption,
        "severity": severity, "public_severity": severity,
        "internal_severity": internal_severity, "seed": run_identity.get("seed"),
        "n_excluded_ref": result["n_excluded_ref_ids_for_center"],
        "ref_record_ids_sha256": expected_ref,
    }
    for namespace, payload in (("pn2021c", pn2021c), ("pn2021_c", pn2021c_alias)):
        if {field: payload.get(field) for field in expected_metadata} != expected_metadata:
            raise ValueError(f"invalid result identity in resume unit {key} at {namespace}")
    expected_corruption = {
        "name": corruption, "public_severity": severity,
        "internal_severity": internal_severity, "seed": run_identity.get("seed"),
    }
    if {field: corruption_payload.get(field) for field in expected_corruption} != expected_corruption:
        raise ValueError(f"invalid result identity in resume unit {key} at corruption")
    ref_hashes = result.get("ref_record_ids_sha256"), compatibility.get("ref_record_ids_sha256")
    if any(value != expected_ref for value in ref_hashes):
        raise ValueError(f"invalid result identity in resume unit {key} at ref_record_ids_sha256")
    protocol = run_identity.get("resolved_evaluator_semantic_config", {}).get("pn2021c_protocol")
    if protocol is not None:
        input_order_id = protocol.get("input_order_id")
        actual_protocol = metadata.get("pn2021c_protocol")
        if not isinstance(actual_protocol, Mapping) or actual_protocol != protocol:
            raise ValueError(f"invalid result identity in resume unit {key} at pn2021c_protocol")
        if (
            pn2021c.get("input_order_id") != input_order_id
            or compatibility.get("input_order_id") != input_order_id
            or actual_protocol.get("waveform_order") != run_identity.get("input_order")
        ):
            raise ValueError(f"invalid result identity in resume unit {key} at input_order")
    return result


def _validate_snapshot(
    snapshot: Any, run_identity: dict[str, Any], ordered_units: list[tuple[str, str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_keys = {
        "resume_schema_version", "run_identity", "run_identity_sha256", "base_output",
        "base_output_sha256", "completed_units",
    }
    if (not isinstance(snapshot, dict) or set(snapshot) != expected_keys
            or snapshot["resume_schema_version"] != _RESUME_SCHEMA_VERSION):
        raise ValueError("resume snapshot has invalid top-level schema or version")
    try:
        saved_digest = _json_sha256(snapshot["run_identity"], finite=True)
    except ValueError as exc:
        raise ValueError("resume snapshot contains invalid run identity") from exc
    if snapshot["run_identity_sha256"] != saved_digest or run_identity != snapshot["run_identity"]:
        raise ValueError("resume snapshot identity mismatch")
    base_output = snapshot["base_output"]
    if (not isinstance(base_output, dict) or base_output.get("per_center") != {}
            or "aggregate_by_corruption_severity" in base_output
            or snapshot["base_output_sha256"] != _json_sha256(base_output)):
        raise ValueError("resume snapshot has invalid base output")
    entries = snapshot["completed_units"]
    if not isinstance(entries, list) or len(entries) > len(ordered_units):
        raise ValueError("resume snapshot has invalid completed units")
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        unit = ordered_units[index]
        key = _unit_key(*unit)
        entry_keys = {"unit_key", "unit_identity_sha256", "result_sha256", "result"}
        if (not isinstance(entry, dict) or set(entry) != entry_keys
                or entry["unit_key"] != key
                or entry["unit_identity_sha256"] != _json_sha256([saved_digest, *unit], finite=True)
                or entry["result_sha256"] != _json_sha256(entry["result"])):
            raise ValueError(f"resume snapshot has invalid completed unit at index {index}")
        results.append(_validate_result(entry["result"], unit, run_identity))
    return base_output, results


def _run_resume_safe_evaluation(
    *, output_path: str | Path, ordered_units: Sequence[tuple[str, str, int]],
    run_identity: dict[str, Any],
    base_output_factory: Callable[[], dict[str, Any]],
    evaluate_unit: Callable[[str, str, int, dict[str, Any]], dict[str, Any]],
    after_commit: Callable[[tuple[str, str, int], int], None] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path).expanduser().resolve()
    units = [(str(center), str(corruption), int(severity)) for center, corruption, severity in ordered_units]
    if len(units) != len(set(units)):
        raise ValueError("ordered resume units must be unique")
    run_identity = _normalize_identity(run_identity)
    run_digest = _json_sha256(run_identity, finite=True)
    sidecar_path, _ = _resume_paths(output_path)
    with _output_lock(output_path):
        if sidecar_path.exists():
            try:
                with sidecar_path.open(encoding="utf-8") as handle:
                    snapshot = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid resume snapshot {sidecar_path}: {exc}") from exc
            base_output, results = _validate_snapshot(snapshot, run_identity, units)
        else:
            base_output = base_output_factory()
            if not isinstance(base_output, dict) or base_output.get("per_center") != {}:
                raise ValueError("base output must be a public output object with empty per_center")
            snapshot = {
                "resume_schema_version": _RESUME_SCHEMA_VERSION, "run_identity": run_identity,
                "run_identity_sha256": run_digest, "base_output": base_output,
                "base_output_sha256": _json_sha256(base_output), "completed_units": [],
            }
            _atomic_write_json(sidecar_path, snapshot)
            results = []
        for unit in units[len(results):]:
            result = _validate_result(evaluate_unit(*unit, base_output), unit, run_identity)
            snapshot["completed_units"].append({
                "unit_key": _unit_key(*unit),
                "unit_identity_sha256": _json_sha256([run_digest, *unit], finite=True),
                "result_sha256": _json_sha256(result), "result": result,
            })
            _atomic_write_json(sidecar_path, snapshot)
            results.append(result)
            if after_commit is not None:
                after_commit(unit, len(results))
        output = copy.deepcopy(base_output)
        output["per_center"] = {}
        for unit, result in zip(units, results):
            center, corruption, severity = unit
            output["per_center"].setdefault(center, {}).setdefault(corruption, {})[str(severity)] = result
        output["aggregate_by_corruption_severity"] = aggregate_corruption_summary(output)
        _atomic_write_json(output_path, output)
        return output


def _build_run_identity(
    args: argparse.Namespace, eval_mode: str, run_dir: Path,
    base_output: Mapping[str, Any], ref_hashes: Mapping[str, str],
    checkpoint_identity: Mapping[str, Any] | None = None,
    eval_result_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    semantic = {key: copy.deepcopy(value) for key, value in base_output.items()
                if key not in {"clean_metric_overrides", "per_center"}}
    semantic.update({
        "batch_size": int(args.batch_size), "device": str(args.device),
        "min_pos": int(args.min_pos), "seed": int(args.seed),
        "limit": int(args.limit) if args.limit is not None else None,
        "clean_mmap_cache_dir": str(args.clean_mmap_cache_dir),
        "clean_cache_dir": str(args.clean_cache_dir),
        "resolved_severity_profile": args.severity_profile_params,
    })
    identity = {
        "identity_schema_version": _RESUME_SCHEMA_VERSION,
        "resolved_evaluator_semantic_config": semantic,
        "seed": int(args.seed), "ref_record_ids_sha256_by_center": dict(ref_hashes),
        "label_mapping": copy.deepcopy(base_output["label_mapping"]),
        "input_order": copy.deepcopy(base_output["corruption_order"]),
        "eval_result_json": copy.deepcopy(eval_result_identity or _file_identity(run_dir / "eval_result.json")),
    }
    identity.update(copy.deepcopy(checkpoint_identity or _checkpoint_identity(
        eval_mode, run_dir, Path(args.checkpoint))))
    _json_sha256(identity, finite=True)
    return identity


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
    if not init_path:
        if str(result.get("stage") or "") == "ptbxl_source":
            validate_target_init_k500_identity(result, result)
            return result
        raise ValueError(f"target-adapted eval_result.json does not record init_model.path: {result_path}")
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
    if evaluation_ref_ids is None and (selected_ids or int(min_target_ref_excluded or 0) > 0):
        raise ValueError(f"{center}: target K500 evaluation requires explicit --exclude_ref_ids")
    exclude_ids = {str(x) for x in evaluation_ref_ids or set()}
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
        clean_metadata=(clean_metric_override or {}).get("metadata", metadata),
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
    eval_result_identity = _file_identity(run_dir / "eval_result.json")
    result = _load_result(run_dir)
    if eval_result_identity != _file_identity(run_dir / "eval_result.json"):
        raise RuntimeError("eval_result.json changed while loading")
    args.exclude_ref_ids_by_center = load_ref_record_ids_by_center(args.exclude_ref_ids)
    centers = list(args.centers or [str(result["center"])])
    unknown = sorted(set(centers).difference(DEFAULT_CENTERS))
    if unknown:
        raise ValueError(f"unknown PN2021-C centers: {unknown}")
    excluded_refs = {
        center: _excluded_ref_ids_for_center(
            result,
            center,
            args.min_target_ref_excluded,
            evaluation_ref_ids=args.exclude_ref_ids_by_center.get(center),
        )
        for center in centers
    }

    device = torch.device(args.device)
    eval_mode = detect_eval_mode(run_dir)
    checkpoint_identity = _checkpoint_identity(eval_mode, run_dir, Path(args.checkpoint))
    input_stabilizer_kwargs = _input_stabilizer_kwargs_from_args(args)
    input_already_ecgfounder = args.corruption_input == LOCKED_ECGFOUNDER_CORRUPTION_INPUT
    apply_input_zscore = args.corruption_input not in {
        "raw_first",
        "native_raw_first",
        LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
    }
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
        "clean_metric_overrides": {},
        "per_center": {},
    }
    out_path = Path(args.output_path) if args.output_path else run_dir / "eval_pn2021_c_ecgfounder.json"
    ref_hashes = {center: ref_ids_sha256(ids) for center, ids in excluded_refs.items()}
    run_identity = _build_run_identity(
        args, eval_mode, run_dir, output, ref_hashes,
        checkpoint_identity, eval_result_identity,
    )
    model_or_feature_model: nn.Module | None = None
    head: nn.Module | None = None

    def ensure_runtime() -> tuple[nn.Module, nn.Module | None]:
        nonlocal model_or_feature_model, head
        if model_or_feature_model is None:
            def build_runtime() -> tuple[nn.Module, nn.Module | None]:
                if eval_mode == "feature_head":
                    return build_ecgfounder_feature_model(Path(args.checkpoint), device), build_ecgfounder_head(
                        run_dir, result, device, num_classes=len(CLASS_NAMES_SUPER5)
                    )
                return build_ecgfounder_fullft_model(
                    run_dir, Path(args.checkpoint), device,
                    num_classes=len(CLASS_NAMES_SUPER5),
                ), None

            model_or_feature_model, head = _build_runtime_with_checkpoint_guard(
                checkpoint_identity,
                eval_mode=eval_mode,
                run_dir=run_dir,
                checkpoint=Path(args.checkpoint),
                factory=build_runtime,
            )
        return model_or_feature_model, head

    def build_base_output() -> dict[str, Any]:
        base_output = copy.deepcopy(output)
        if (
            args.recompute_clean_with_input_stabilizer
            or args.corruption_input
            in {"raw_first", "native_raw_first", LOCKED_ECGFOUNDER_CORRUPTION_INPUT}
        ):
            model, runtime_head = ensure_runtime()
            for center in centers:
                base_output["clean_metric_overrides"][center] = eval_clean_center(
                    eval_mode, model, runtime_head, result, args, device, center,
                    input_stabilizer_kwargs=input_stabilizer_kwargs,
                    apply_input_zscore=apply_input_zscore,
                    input_already_ecgfounder=input_already_ecgfounder,
                )
        return base_output

    def evaluate_unit(
        center: str,
        corruption: str,
        severity: int,
        base_output: dict[str, Any],
    ) -> dict[str, Any]:
        model, runtime_head = ensure_runtime()
        return eval_one(
            eval_mode, model, runtime_head, result, args, device, center, corruption, severity,
            clean_metric_override=base_output["clean_metric_overrides"].get(center),
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
            input_already_ecgfounder=input_already_ecgfounder,
        )

    ordered_units = [
        (center, corruption, int(severity))
        for center in centers
        for corruption in args.corruptions
        for severity in args.severities
    ]
    _run_resume_safe_evaluation(
        output_path=out_path,
        ordered_units=ordered_units,
        run_identity=run_identity,
        base_output_factory=build_base_output,
        evaluate_unit=evaluate_unit,
    )
    print(f"[done] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
