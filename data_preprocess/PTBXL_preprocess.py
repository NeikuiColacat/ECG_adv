"""Build a contiguous, model-independent PTB-XL NumPy cache."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "PTBXL.yaml"
CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
DATA_ROOT_TOKEN = "${data_root}"


class WaveformQualityError(ValueError):
    """A record-level waveform rejection with machine-readable QC details."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _repair_nonfinite_per_lead(
    signal_tc: np.ndarray,
    *,
    max_record_fraction: float,
    max_lead_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair sparse NaN/Inf values using within-lead linear interpolation.

    ``numpy.interp`` uses the nearest valid value outside the valid index range,
    which gives the intended edge-fill behavior without inventing cross-lead
    information. Records with an all-invalid lead or excessive corruption are
    rejected before interpolation.
    """

    repaired = np.asarray(signal_tc, dtype=np.float64).copy()
    if repaired.ndim != 2 or repaired.shape[0] == 0 or repaired.shape[1] == 0:
        raise WaveformQualityError(
            "invalid_waveform_shape",
            {"shape": list(repaired.shape)},
        )

    invalid = ~np.isfinite(repaired)
    invalid_count = int(invalid.sum())
    total_values = int(repaired.size)
    record_fraction = invalid_count / total_values
    per_lead_counts = invalid.sum(axis=0).astype(int)
    per_lead_fractions = per_lead_counts / repaired.shape[0]
    details: dict[str, Any] = {
        "nonfinite_count": invalid_count,
        "nonfinite_fraction": record_fraction,
        "nonfinite_count_by_lead": per_lead_counts.tolist(),
        "nonfinite_fraction_by_lead": per_lead_fractions.tolist(),
        "repaired_nonfinite_count": 0,
        "repair_method": "none",
    }
    if invalid_count == 0:
        return repaired, details
    if record_fraction > max_record_fraction:
        raise WaveformQualityError("record_nonfinite_fraction_exceeded", details)

    sample_indices = np.arange(repaired.shape[0])
    for lead_index in range(repaired.shape[1]):
        invalid_lead = invalid[:, lead_index]
        if not invalid_lead.any():
            continue
        valid_lead = ~invalid_lead
        if not valid_lead.any():
            raise WaveformQualityError("all_nonfinite_lead", details)
        if per_lead_fractions[lead_index] > max_lead_fraction:
            raise WaveformQualityError("lead_nonfinite_fraction_exceeded", details)
        repaired[invalid_lead, lead_index] = np.interp(
            sample_indices[invalid_lead],
            sample_indices[valid_lead],
            repaired[valid_lead, lead_index],
        )

    if not np.isfinite(repaired).all():
        raise WaveformQualityError("nonfinite_remains_after_repair", details)
    details["repaired_nonfinite_count"] = invalid_count
    details["repair_method"] = "linear_per_lead_nearest_edge"
    return repaired, details


def _validate_nonfinite_policy(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("nonfinite_policy")
    if not isinstance(policy, dict):
        raise ValueError("PTB-XL config must define nonfinite_policy")
    if str(policy.get("repair_method")) != "linear_per_lead_nearest_edge":
        raise ValueError(
            "PTB-XL nonfinite repair_method must be "
            "'linear_per_lead_nearest_edge'"
        )
    max_record_fraction = float(policy["max_record_fraction"])
    max_lead_fraction = float(policy["max_lead_fraction"])
    for name, value in (
        ("max_record_fraction", max_record_fraction),
        ("max_lead_fraction", max_lead_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"PTB-XL {name} must be in [0, 1], got {value}")
    return {
        "repair_method": "linear_per_lead_nearest_edge",
        "max_record_fraction": max_record_fraction,
        "max_lead_fraction": max_lead_fraction,
        "reject_all_nonfinite_lead": True,
    }


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PTB-XL config must be a YAML mapping: {path}")
    return payload


def _resolve_path(value: str, data_root: str | Path | None) -> Path:
    text = str(value)
    if DATA_ROOT_TOKEN in text:
        if data_root is None:
            raise ValueError(
                f"Path {text!r} requires data_root in the YAML config"
            )
        text = text.replace(DATA_ROOT_TOKEN, str(Path(data_root).expanduser()))
    return Path(text).expanduser().resolve()


def _normalise_lead_name(name: str) -> str:
    compact = str(name).strip().replace(" ", "").upper()
    aliases = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    return aliases.get(compact, compact)


def _reorder_leads(
    signal_tc: np.ndarray,
    source_leads: list[str],
    target_leads: list[str],
) -> np.ndarray:
    normalised_source = [_normalise_lead_name(name) for name in source_leads]
    normalised_target = [_normalise_lead_name(name) for name in target_leads]
    if len(set(normalised_source)) != len(normalised_source):
        raise ValueError(f"Duplicate source leads: {source_leads}")
    try:
        indices = [normalised_source.index(name) for name in normalised_target]
    except ValueError as exc:
        raise ValueError(
            f"Cannot map source leads {source_leads} to target leads {target_leads}"
        ) from exc
    return np.asarray(signal_tc)[:, indices]


def _read_wfdb(record_path: Path) -> tuple[np.ndarray, float, list[str]]:
    import wfdb

    signal_tc, fields = wfdb.rdsamp(str(record_path))
    return (
        np.asarray(signal_tc),
        float(fields["fs"]),
        [str(name) for name in fields["sig_name"]],
    )


def _record_hash(dataset: str, dataset_version: str, record_key: str) -> str:
    identity = f"{dataset}\0{dataset_version}\0{record_key}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _fixed_unicode_array(values: list[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _linear_interpolate_time_batch(
    signals_ntc: np.ndarray,
    target_num_samples: int,
) -> np.ndarray:
    """Resize a batch with the current ECGFounder aligned-corner interpolation."""

    signals_ntc = np.asarray(signals_ntc)
    if signals_ntc.ndim != 3 or signals_ntc.shape[1] < 2:
        raise ValueError(
            "Expected batched time-channel signals with at least two time points, "
            f"got {signals_ntc.shape}"
        )
    if target_num_samples < 2:
        raise ValueError(f"target_num_samples must be at least 2, got {target_num_samples}")
    import torch
    import torch.nn.functional as F

    source_array = np.array(signals_ntc, copy=True, order="C")
    source = torch.from_numpy(source_array).permute(0, 2, 1)
    with torch.no_grad():
        derived = F.interpolate(
            source,
            size=int(target_num_samples),
            mode="linear",
            align_corners=True,
        )
    return derived.permute(0, 2, 1).contiguous().numpy()


def _load_super5_map(scp_statements_path: Path) -> dict[str, int]:
    frame = pd.read_csv(scp_statements_path, index_col=0)
    mapping: dict[str, int] = {}
    class_to_index = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    for code, row in frame.iterrows():
        if float(row.get("diagnostic", 0.0)) != 1.0:
            continue
        diagnostic_class = row.get("diagnostic_class")
        if isinstance(diagnostic_class, str) and diagnostic_class in class_to_index:
            mapping[str(code)] = class_to_index[diagnostic_class]
    return mapping


def _build_labels(metadata: pd.DataFrame, scp_map: dict[str, int]) -> np.ndarray:
    labels = np.zeros((len(metadata), len(CLASS_ORDER)), dtype=np.uint8)
    for row_index, value in enumerate(metadata["scp_codes"]):
        codes = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(codes, dict):
            raise ValueError(f"Invalid scp_codes at row {row_index}: {value!r}")
        for code, confidence in codes.items():
            class_index = scp_map.get(str(code))
            if class_index is not None and float(confidence) >= 0.0:
                labels[row_index, class_index] = 1
    return labels


def _validate_cache_contract(
    config: dict[str, Any],
) -> tuple[int, int, int, int, int, int, list[str], np.dtype]:
    sampling_rate = int(config["sampling_rate_hz"])
    duration_seconds = int(config["duration_seconds"])
    num_samples = int(config["num_samples"])
    if sampling_rate * duration_seconds != num_samples:
        raise ValueError(
            "PTB-XL sampling contract is inconsistent: "
            f"{sampling_rate} Hz * {duration_seconds} s != {num_samples} samples"
        )
    derived_sampling_rate = int(config["derived_sampling_rate_hz"])
    derived_num_samples = int(config["derived_num_samples"])
    if derived_sampling_rate * duration_seconds != derived_num_samples:
        raise ValueError(
            "PTB-XL derived sampling contract is inconsistent: "
            f"{derived_sampling_rate} Hz * {duration_seconds} s "
            f"!= {derived_num_samples} samples"
        )
    if str(config["derived_interpolation"]) != "linear_align_corners":
        raise ValueError(
            "PTB-XL derived_interpolation must be 'linear_align_corners'"
        )
    derived_batch_size = int(config["derived_batch_size"])
    if derived_batch_size <= 0:
        raise ValueError("PTB-XL derived_batch_size must be positive")
    lead_order = [str(name) for name in config["lead_order"]]
    if len(lead_order) != 12 or len(set(lead_order)) != 12:
        raise ValueError(f"PTB-XL lead_order must contain 12 unique leads: {lead_order}")
    dtype = np.dtype(config.get("cache_dtype", "float32"))
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"cache_dtype must be float32 or float64, got {dtype}")
    return (
        sampling_rate,
        duration_seconds,
        num_samples,
        derived_sampling_rate,
        derived_num_samples,
        derived_batch_size,
        lead_order,
        dtype,
    )


def build_ptbxl_npy_cache(
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    """Materialize PTB-XL into one contiguous ``signals.npy`` cache.

    The cache keeps physical-mV waveforms in PTB-XL lead order. It intentionally
    does not filter, normalize, split, or convert to model-specific layouts.
    Existing cache directories are never overwritten.
    """

    config = _load_config(config_path)
    (
        sampling_rate,
        duration_seconds,
        num_samples,
        derived_sampling_rate,
        derived_num_samples,
        derived_batch_size,
        lead_order,
        cache_dtype,
    ) = _validate_cache_contract(config)
    nonfinite_policy = _validate_nonfinite_policy(config)
    dataset = str(config["dataset"])
    dataset_version = str(config["dataset_version"])
    data_root = config.get("data_root")
    source_root = _resolve_path(str(config["root"]), data_root)
    cache_dir = _resolve_path(str(config["cache_dir"]), data_root)
    metadata_path = source_root / str(config["metadata_file"])
    scp_statements_path = source_root / str(config["scp_statements_file"])

    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {cache_dir}")
    for required in (source_root, metadata_path, scp_statements_path):
        if not required.exists():
            raise FileNotFoundError(required)

    metadata = pd.read_csv(metadata_path)
    required_columns = {"ecg_id", "filename_lr", "scp_codes"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"PTB-XL metadata missing columns: {missing_columns}")
    if metadata["ecg_id"].duplicated().any():
        duplicates = metadata.loc[metadata["ecg_id"].duplicated(), "ecg_id"].tolist()
        raise ValueError(f"Duplicate PTB-XL ecg_id values: {duplicates[:10]}")

    source_records = [str(value) for value in metadata["filename_lr"].tolist()]
    record_ids = [str(int(value)) for value in metadata["ecg_id"].tolist()]
    record_keys = [
        f"ecg_id={record_id};source={source_record}"
        for record_id, source_record in zip(record_ids, source_records)
    ]
    hash_ids = [
        _record_hash(dataset, dataset_version, record_key)
        for record_key in record_keys
    ]
    if len(set(hash_ids)) != len(hash_ids):
        raise ValueError("PTB-XL record hash collision detected")

    scp_map = _load_super5_map(scp_statements_path)
    labels = _build_labels(metadata, scp_map)
    source_record_count = len(metadata)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir.with_name(f".{cache_dir.name}.building-{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir()

    try:
        build_signal_path = staging_dir / "signals.build.npy"
        final_signal_path = staging_dir / "signals.npy"
        signals = np.lib.format.open_memmap(
            build_signal_path,
            mode="w+",
            dtype=cache_dtype,
            shape=(len(metadata), num_samples, len(lead_order)),
        )
        kept_indices: list[int] = []
        quality_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        for index, source_record in enumerate(source_records):
            try:
                signal_tc, source_fs, source_leads = _read_wfdb(
                    source_root / source_record
                )
                if int(round(source_fs)) != sampling_rate:
                    raise ValueError(
                        f"Unexpected sampling rate: {source_fs} Hz"
                    )
                signal_tc = _reorder_leads(signal_tc, source_leads, lead_order)
                if signal_tc.shape != (num_samples, len(lead_order)):
                    raise ValueError(f"Unexpected waveform shape: {signal_tc.shape}")
                signal_tc, quality = _repair_nonfinite_per_lead(
                    signal_tc,
                    max_record_fraction=nonfinite_policy[
                        "max_record_fraction"
                    ],
                    max_lead_fraction=nonfinite_policy["max_lead_fraction"],
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    cache_signal = signal_tc.astype(cache_dtype, copy=False)
                if not np.isfinite(cache_signal).all():
                    raise WaveformQualityError(
                        "nonfinite_after_cache_dtype_cast",
                        quality,
                    )
            except (WaveformQualityError, ValueError, OSError, EOFError) as exc:
                failure = {
                    "source_index": index,
                    "record_id": record_ids[index],
                    "record_key": record_keys[index],
                    "source_record": source_record,
                    "error_type": type(exc).__name__,
                    "reason": getattr(exc, "reason", str(exc)),
                }
                failure.update(getattr(exc, "details", {}))
                failed_rows.append(failure)
                continue

            cache_index = len(kept_indices)
            signals[cache_index] = cache_signal
            kept_indices.append(index)
            quality_rows.append(
                {
                    "quality_status": (
                        "repaired"
                        if quality["repaired_nonfinite_count"] > 0
                        else "clean"
                    ),
                    **quality,
                }
            )
            if (index + 1) % 1000 == 0 or index + 1 == len(metadata):
                print(
                    f"[PTB-XL] scanned={index + 1}/{len(metadata)} "
                    f"kept={len(kept_indices)} failed={len(failed_rows)}"
                )
        signals.flush()
        del signals

        record_count = len(kept_indices)
        if record_count == 0:
            raise ValueError("No valid PTB-XL records remained after quality control")
        if record_count == source_record_count:
            build_signal_path.replace(final_signal_path)
        else:
            build_signals = np.load(build_signal_path, mmap_mode="r")
            compact_signals = np.lib.format.open_memmap(
                final_signal_path,
                mode="w+",
                dtype=cache_dtype,
                shape=(record_count, num_samples, len(lead_order)),
            )
            compact_batch_size = max(derived_batch_size, 1)
            for start in range(0, record_count, compact_batch_size):
                end = min(start + compact_batch_size, record_count)
                compact_signals[start:end] = build_signals[start:end]
            compact_signals.flush()
            del compact_signals
            del build_signals
            build_signal_path.unlink()

        signals = np.load(final_signal_path, mmap_mode="r")

        signals_500hz = np.lib.format.open_memmap(
            staging_dir / "signals_500hz.npy",
            mode="w+",
            dtype=cache_dtype,
            shape=(record_count, derived_num_samples, len(lead_order)),
        )
        for start in range(0, record_count, derived_batch_size):
            end = min(start + derived_batch_size, record_count)
            derived_batch = _linear_interpolate_time_batch(
                signals[start:end],
                derived_num_samples,
            ).astype(cache_dtype, copy=False)
            if not np.isfinite(derived_batch).all():
                raise ValueError(
                    f"Non-finite PTB-XL 500 Hz values in cache rows {start}:{end}"
                )
            signals_500hz[start:end] = derived_batch
            if end == record_count or end % (derived_batch_size * 20) == 0:
                print(f"[PTB-XL 500 Hz] {end}/{record_count} records")
        signals_500hz.flush()
        del signals_500hz
        del signals

        kept_array = np.asarray(kept_indices, dtype=np.int64)
        labels = labels[kept_array]
        record_ids = [record_ids[index] for index in kept_indices]
        record_keys = [record_keys[index] for index in kept_indices]
        hash_ids = [hash_ids[index] for index in kept_indices]
        kept_source_records = [source_records[index] for index in kept_indices]
        np.save(staging_dir / "labels_super5.npy", labels, allow_pickle=False)
        np.save(
            staging_dir / "record_ids.npy",
            _fixed_unicode_array(record_ids),
            allow_pickle=False,
        )
        np.save(
            staging_dir / "hash_ids.npy",
            np.asarray(hash_ids, dtype="<U64"),
            allow_pickle=False,
        )

        records = metadata.iloc[kept_array].copy().reset_index(drop=True)
        records.insert(0, "cache_index", np.arange(len(records), dtype=np.int64))
        records["record_id"] = record_ids
        records["record_key"] = record_keys
        records["hash_id"] = hash_ids
        records["source_record"] = kept_source_records
        quality_frame = pd.DataFrame(quality_rows)
        for column in quality_frame.columns:
            records[column] = quality_frame[column]
        records.to_parquet(staging_dir / "records.parquet", index=False)
        failed_path = staging_dir / "failed_records.jsonl"
        failed_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in failed_rows
            ),
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 3,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "dataset_version": dataset_version,
            "source_record_count": source_record_count,
            "record_count": record_count,
            "excluded_record_count": len(failed_rows),
            "waveform": {
                "file": "signals.npy",
                "shape": [record_count, num_samples, len(lead_order)],
                "dtype": cache_dtype.name,
                "sampling_rate_hz": sampling_rate,
                "layout": "time_channel",
                "lead_order": lead_order,
                "physical_unit": "mV",
                "filtering": "none",
                "normalization": "none",
            },
            "derived_waveforms": {
                "500hz_linear": {
                    "file": "signals_500hz.npy",
                    "source_file": "signals.npy",
                    "shape": [
                        record_count,
                        derived_num_samples,
                        len(lead_order),
                    ],
                    "dtype": cache_dtype.name,
                    "sampling_rate_hz": derived_sampling_rate,
                    "duration_seconds": duration_seconds,
                    "layout": "time_channel",
                    "lead_order": lead_order,
                    "physical_unit": "mV",
                    "interpolation": "linear",
                    "align_corners": True,
                    "implementation": "torch.nn.functional.interpolate",
                    "source_sampling_rate_hz": sampling_rate,
                    "information_note": "100hz_grid_adapter_no_high_frequency_recovery",
                    "filtering": "none",
                    "normalization": "none",
                }
            },
            "record_hash": {
                "file": "hash_ids.npy",
                "algorithm": "sha256",
                "identity_fields": ["dataset", "dataset_version", "record_key"],
                "unique": True,
            },
            "quality_control": {
                **nonfinite_policy,
                "repaired_record_count": sum(
                    row["quality_status"] == "repaired" for row in quality_rows
                ),
                "failed_record_count": len(failed_rows),
                "failed_records_file": "failed_records.jsonl",
                "post_resample_finite_check": True,
                "post_dtype_cast_finite_check": True,
            },
            "labels": {
                "file": "labels_super5.npy",
                "dtype": "uint8",
                "class_order": list(CLASS_ORDER),
                "scp_confidence_comparison": ">= 0.0",
            },
            "files": {
                "record_ids": "record_ids.npy",
                "metadata": "records.parquet",
                "signals_500hz": "signals_500hz.npy",
                "failed_records": "failed_records.jsonl",
            },
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(cache_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"[PTB-XL] cache written to {cache_dir}")
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="PTB-XL preprocessing YAML (a copied config bundle is supported)",
    )
    args = parser.parse_args()
    build_ptbxl_npy_cache(args.config)


if __name__ == "__main__":
    main()
