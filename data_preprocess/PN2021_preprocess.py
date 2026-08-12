"""Build a contiguous, model-independent PN2021 NumPy cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.pn2021_metadata import (  # noqa: E402
    parse_header_snomeds,
    parse_pn2021_header_metadata,
)
from data_preprocess import preprocess_primitives as primitives  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "PN2021.yaml"
EXPECTED_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")


def _stable_mapping_hash(
    *,
    version: str,
    direct_positive: dict[int, str],
    norm_candidate: set[int],
    norm_suppress: set[int],
    ignored: set[int],
) -> str:
    payload = json.dumps(
        {
            "version": version,
            "positive": direct_positive,
            "norm_positive": sorted(norm_candidate),
            "norm_suppress": sorted(norm_suppress),
            "ignored": sorted(ignored),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _load_super5_mapping(
    config_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    mapping_value = config.get("label_mapping_file")
    if not mapping_value:
        raise ValueError("PN2021 config must define label_mapping_file")
    mapping_path = Path(str(mapping_value)).expanduser()
    if not mapping_path.is_absolute():
        mapping_path = Path(config_path).resolve().parent / mapping_path
    mapping_path = mapping_path.resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PN2021 label mapping must be a YAML mapping: {mapping_path}")

    version = str(payload["mapping_version"])
    declared_hash = str(payload["mapping_hash"])
    class_order = tuple(str(name) for name in payload["class_order"])
    if class_order != EXPECTED_CLASS_ORDER:
        raise ValueError(
            "PN2021 Super5 class_order must be "
            f"{EXPECTED_CLASS_ORDER}, got {class_order}"
        )

    raw_direct = payload.get("direct_positive", {})
    if not isinstance(raw_direct, dict):
        raise ValueError("direct_positive must be a SNOMED-to-class mapping")
    direct_positive = {int(code): str(name) for code, name in raw_direct.items()}
    invalid_classes = sorted(set(direct_positive.values()) - set(class_order))
    if invalid_classes:
        raise ValueError(f"Unknown direct-positive Super5 classes: {invalid_classes}")
    if "NORM" in direct_positive.values():
        raise ValueError("NORM codes belong in norm_candidate, not direct_positive")

    norm_candidate = {int(code) for code in payload.get("norm_candidate", [])}
    norm_suppress = {int(code) for code in payload.get("norm_suppress", [])}
    ignored = {int(code) for code in payload.get("ignored", [])}
    decision_groups = {
        "direct_positive": set(direct_positive),
        "norm_candidate": norm_candidate,
        "norm_suppress": norm_suppress,
        "ignored": ignored,
    }
    group_names = list(decision_groups)
    for index, left_name in enumerate(group_names):
        for right_name in group_names[index + 1 :]:
            overlap = decision_groups[left_name] & decision_groups[right_name]
            if overlap:
                raise ValueError(
                    f"SNOMED codes overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)}"
                )

    all_zero_policy = str(payload.get("all_zero_policy", ""))
    if all_zero_policy != "kept":
        raise ValueError(
            f"PN2021 all_zero_policy must be 'kept', got {all_zero_policy!r}"
        )
    computed_hash = _stable_mapping_hash(
        version=version,
        direct_positive=direct_positive,
        norm_candidate=norm_candidate,
        norm_suppress=norm_suppress,
        ignored=ignored,
    )
    if computed_hash != declared_hash:
        raise ValueError(
            "PN2021 mapping hash mismatch: "
            f"declared {declared_hash}, computed {computed_hash}"
        )
    return {
        "path": mapping_path,
        "version": version,
        "hash": computed_hash,
        "class_order": class_order,
        "class_to_index": {name: index for index, name in enumerate(class_order)},
        "direct_positive": direct_positive,
        "norm_candidate": norm_candidate,
        "norm_suppress": norm_suppress,
        "ignored": ignored,
        "all_zero_policy": all_zero_policy,
    }


def _snomed_list_to_super5(
    snomed_codes: list[int] | tuple[int, ...],
    mapping: dict[str, Any],
) -> np.ndarray:
    label = np.zeros(len(mapping["class_order"]), dtype=np.uint8)
    has_norm_candidate = False
    has_norm_suppress = False
    for raw_code in snomed_codes:
        code = int(raw_code)
        class_name = mapping["direct_positive"].get(code)
        if class_name is not None:
            label[mapping["class_to_index"][class_name]] = 1
        if code in mapping["norm_candidate"]:
            has_norm_candidate = True
        if code in mapping["norm_suppress"]:
            has_norm_suppress = True
    abnormal_indices = [
        index
        for index, name in enumerate(mapping["class_order"])
        if name != "NORM"
    ]
    norm_index = mapping["class_to_index"]["NORM"]
    label[norm_index] = int(
        has_norm_candidate
        and not label[abnormal_indices].any()
        and not has_norm_suppress
    )
    return label


def _resample_crop_pad(
    signal_tc: np.ndarray,
    *,
    source_fs: float,
    target_fs: int,
    duration_seconds: int,
    target_num_samples: int,
    window_policy: str,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Center-crop long records, linearly interpolate, then right-pad short records."""

    if source_fs <= 0:
        raise ValueError(f"Invalid source sampling rate: {source_fs}")
    if window_policy != "center":
        raise ValueError(f"Unsupported PN2021 window_policy: {window_policy!r}")
    original_num_samples = int(signal_tc.shape[0])
    max_source_samples = int(round(source_fs * duration_seconds))
    used_source_samples = min(original_num_samples, max_source_samples)
    source_start_sample = max((original_num_samples - used_source_samples) // 2, 0)
    source_end_sample = source_start_sample + used_source_samples
    cropped = np.asarray(
        signal_tc[source_start_sample:source_end_sample],
        dtype=np.float64,
    )

    if int(round(source_fs)) == target_fs and abs(source_fs - target_fs) < 1e-6:
        resampled = cropped
    else:
        resized_samples = max(
            1,
            int(round(cropped.shape[0] * target_fs / source_fs)),
        )
        if cropped.shape[0] == 1 or resized_samples == 1:
            resampled = cropped[:1]
        else:
            resampled = primitives._linear_interpolate_time_batch(
                cropped[None, ...],
                resized_samples,
            )[0]

    valid_target_samples = min(int(resampled.shape[0]), target_num_samples)
    output = np.zeros((target_num_samples, signal_tc.shape[1]), dtype=np.float64)
    output[:valid_target_samples] = resampled[:valid_target_samples]
    details: dict[str, int | bool] = {
        "original_num_samples": original_num_samples,
        "used_source_samples": used_source_samples,
        "source_start_sample": source_start_sample,
        "source_end_sample": source_end_sample,
        "valid_target_samples": valid_target_samples,
        "was_padded": valid_target_samples < target_num_samples,
        "was_truncated": original_num_samples > max_source_samples,
    }
    return output, details


def _validate_cache_contract(
    config: dict[str, Any],
) -> tuple[
    int,
    int,
    int,
    str,
    int,
    int,
    int,
    list[str],
    np.dtype,
    list[str],
    set[str],
]:
    target_fs = int(config["target_sampling_rate_hz"])
    duration_seconds = int(config["target_duration_seconds"])
    target_num_samples = int(config["target_num_samples"])
    if target_fs * duration_seconds != target_num_samples:
        raise ValueError(
            "PN2021 sampling contract is inconsistent: "
            f"{target_fs} Hz * {duration_seconds} s != {target_num_samples} samples"
        )
    window_policy = str(config["window_policy"])
    if window_policy != "center":
        raise ValueError(
            f"PN2021 window_policy must be 'center', got {window_policy!r}"
        )
    if str(config["target_interpolation"]) != "linear_align_corners":
        raise ValueError(
            "PN2021 target_interpolation must be 'linear_align_corners'"
        )
    derived_sampling_rate = int(config["derived_sampling_rate_hz"])
    derived_num_samples = int(config["derived_num_samples"])
    if derived_sampling_rate * duration_seconds != derived_num_samples:
        raise ValueError(
            "PN2021 derived sampling contract is inconsistent: "
            f"{derived_sampling_rate} Hz * {duration_seconds} s "
            f"!= {derived_num_samples} samples"
        )
    if str(config["derived_interpolation"]) != "linear_align_corners":
        raise ValueError(
            "PN2021 derived_interpolation must be 'linear_align_corners'"
        )
    derived_batch_size = int(config["derived_batch_size"])
    if derived_batch_size <= 0:
        raise ValueError("PN2021 derived_batch_size must be positive")
    lead_order = [str(name) for name in config["lead_order"]]
    if len(lead_order) != 12 or len(set(lead_order)) != 12:
        raise ValueError(f"PN2021 lead_order must contain 12 unique leads: {lead_order}")
    dtype = np.dtype(config.get("cache_dtype", "float32"))
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"cache_dtype must be float32 or float64, got {dtype}")
    centers = [str(center) for center in config["centers"]]
    excluded = {str(center).lower() for center in config.get("exclude_centers", [])}
    forbidden = [center for center in centers if center.lower() in excluded]
    if forbidden:
        raise ValueError(f"Excluded PN2021 centers cannot be cached: {forbidden}")
    if len(centers) != len(set(centers)):
        raise ValueError(f"Duplicate PN2021 centers: {centers}")
    return (
        target_fs,
        duration_seconds,
        target_num_samples,
        window_policy,
        derived_sampling_rate,
        derived_num_samples,
        derived_batch_size,
        lead_order,
        dtype,
        centers,
        excluded,
    )


def _scan_records(waveform_root: Path, centers: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for center in centers:
        center_dir = waveform_root / center
        if not center_dir.is_dir():
            raise FileNotFoundError(center_dir)
        for header_path in sorted(center_dir.rglob("*.hea")):
            record_path = header_path.with_suffix("")
            source_record = record_path.relative_to(waveform_root).as_posix()
            record_id = record_path.name
            snomed_codes = parse_header_snomeds(header_path)
            demographics = parse_pn2021_header_metadata(header_path)
            records.append(
                {
                    "center": center,
                    "record_id": record_id,
                    "record_key": source_record,
                    "source_record": source_record,
                    "record_path": record_path,
                    "header_path": header_path,
                    "snomed_codes": snomed_codes,
                    "age": demographics.get("age"),
                    "sex": demographics.get("sex"),
                }
            )
    return records


def build_pn2021_npy_cache(
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    """Materialize the seven external PN2021 centers into one NumPy cache.

    Long records are center-cropped to ten seconds before aligned-corner linear
    interpolation. Short records are right-padded afterward. The cache keeps
    physical-mV scale and does not filter, normalize, perform K-shot exclusion,
    or drop all-zero labels. Existing cache directories are never overwritten.
    """

    config = primitives._load_yaml_mapping("PN2021", config_path)
    label_mapping = _load_super5_mapping(config_path, config)
    (
        target_fs,
        duration_seconds,
        target_num_samples,
        window_policy,
        derived_sampling_rate,
        derived_num_samples,
        derived_batch_size,
        lead_order,
        cache_dtype,
        centers,
        excluded_centers,
    ) = _validate_cache_contract(config)
    nonfinite_policy = primitives._validate_nonfinite_policy("PN2021", config)
    dataset = str(config["dataset"])
    dataset_version = str(config["dataset_version"])
    data_root = config.get("data_root")
    source_root = primitives._resolve_path(str(config["root"]), data_root)
    waveform_root = source_root / str(config["waveform_dir"])
    cache_dir = primitives._resolve_path(str(config["cache_dir"]), data_root)

    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {cache_dir}")
    if not waveform_root.is_dir():
        raise FileNotFoundError(waveform_root)

    records = _scan_records(waveform_root, centers)
    if not records:
        raise ValueError(f"No PN2021 records found under {waveform_root}")
    record_ids = [str(record["record_id"]) for record in records]
    record_keys = [str(record["record_key"]) for record in records]
    hash_ids = [
        primitives._record_hash(dataset, dataset_version, record_key)
        for record_key in record_keys
    ]
    if len(set(record_keys)) != len(record_keys):
        raise ValueError("Duplicate PN2021 record_key values detected")
    if len(set(hash_ids)) != len(hash_ids):
        raise ValueError("PN2021 record hash collision detected")

    labels = np.stack(
        [
            _snomed_list_to_super5(record["snomed_codes"], label_mapping)
            for record in records
        ]
    )
    source_record_count = len(records)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir.with_name(f".{cache_dir.name}.building-{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir()

    metadata_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    try:
        build_signal_path = staging_dir / "signals.build.npy"
        final_signal_path = staging_dir / "signals.npy"
        signals = np.lib.format.open_memmap(
            build_signal_path,
            mode="w+",
            dtype=cache_dtype,
            shape=(len(records), target_num_samples, len(lead_order)),
        )
        for index, record in enumerate(records):
            try:
                signal_tc, source_fs, source_leads = primitives._read_wfdb(
                    record["record_path"]
                )
                if signal_tc.ndim != 2:
                    raise ValueError(
                        f"Invalid waveform dimensions: {signal_tc.shape}"
                    )
                signal_tc = primitives._reorder_leads(
                    signal_tc, source_leads, lead_order
                )
                signal_tc, quality = primitives._repair_nonfinite_per_lead(
                    signal_tc,
                    max_record_fraction=nonfinite_policy[
                        "max_record_fraction"
                    ],
                    max_lead_fraction=nonfinite_policy["max_lead_fraction"],
                )
                processed, details = _resample_crop_pad(
                    signal_tc,
                    source_fs=source_fs,
                    target_fs=target_fs,
                    duration_seconds=duration_seconds,
                    target_num_samples=target_num_samples,
                    window_policy=window_policy,
                )
                if processed.shape != (target_num_samples, len(lead_order)):
                    raise ValueError(
                        f"Unexpected processed shape: {processed.shape}"
                    )
                if not np.isfinite(processed).all():
                    raise primitives.WaveformQualityError(
                        "nonfinite_after_resample_crop_pad",
                        quality,
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    cache_signal = processed.astype(cache_dtype, copy=False)
                if not np.isfinite(cache_signal).all():
                    raise primitives.WaveformQualityError(
                        "nonfinite_after_cache_dtype_cast",
                        quality,
                    )
            except (
                primitives.WaveformQualityError,
                ValueError,
                OSError,
                EOFError,
            ) as exc:
                failure = {
                    "source_index": index,
                    "record_id": record["record_id"],
                    "record_key": record["record_key"],
                    "center": record["center"],
                    "source_record": record["source_record"],
                    "error_type": type(exc).__name__,
                    "reason": getattr(exc, "reason", str(exc)),
                }
                failure.update(getattr(exc, "details", {}))
                failed_rows.append(failure)
                continue

            cache_index = len(kept_indices)
            signals[cache_index] = cache_signal
            kept_indices.append(index)
            metadata_rows.append(
                {
                    "cache_index": cache_index,
                    "record_id": record["record_id"],
                    "record_key": record["record_key"],
                    "hash_id": hash_ids[index],
                    "center": record["center"],
                    "source_record": record["source_record"],
                    "original_fs": source_fs,
                    "original_num_samples": details["original_num_samples"],
                    "original_duration_seconds": details["original_num_samples"] / source_fs,
                    "used_source_samples": details["used_source_samples"],
                    "source_start_sample": details["source_start_sample"],
                    "source_end_sample": details["source_end_sample"],
                    "source_start_seconds": details["source_start_sample"] / source_fs,
                    "source_end_seconds": details["source_end_sample"] / source_fs,
                    "valid_target_samples": details["valid_target_samples"],
                    "was_padded": details["was_padded"],
                    "was_truncated": details["was_truncated"],
                    "snomed_codes": record["snomed_codes"],
                    "age": record["age"],
                    "sex": record["sex"],
                    "quality_status": (
                        "repaired"
                        if quality["repaired_nonfinite_count"] > 0
                        else "clean"
                    ),
                    **quality,
                }
            )
            if (index + 1) % 1000 == 0 or index + 1 == len(records):
                print(
                    f"[PN2021] scanned={index + 1}/{len(records)} "
                    f"kept={len(kept_indices)} failed={len(failed_rows)}"
                )
        signals.flush()
        del signals

        record_count = len(kept_indices)
        if record_count == 0:
            raise ValueError("No valid PN2021 records remained after quality control")
        if record_count == source_record_count:
            build_signal_path.replace(final_signal_path)
        else:
            build_signals = np.load(build_signal_path, mmap_mode="r")
            compact_signals = np.lib.format.open_memmap(
                final_signal_path,
                mode="w+",
                dtype=cache_dtype,
                shape=(record_count, target_num_samples, len(lead_order)),
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
            derived_batch = primitives._linear_interpolate_time_batch(
                signals[start:end],
                derived_num_samples,
            ).astype(cache_dtype, copy=False)
            if not np.isfinite(derived_batch).all():
                raise ValueError(
                    f"Non-finite PN2021 500 Hz values in cache rows {start}:{end}"
                )
            signals_500hz[start:end] = derived_batch
            if end == record_count or end % (derived_batch_size * 20) == 0:
                print(f"[PN2021 500 Hz] {end}/{record_count} records")
        signals_500hz.flush()
        del signals_500hz
        del signals

        kept_array = np.asarray(kept_indices, dtype=np.int64)
        labels = labels[kept_array]
        record_ids = [record_ids[index] for index in kept_indices]
        record_keys = [record_keys[index] for index in kept_indices]
        hash_ids = [hash_ids[index] for index in kept_indices]
        np.save(staging_dir / "labels_super5.npy", labels, allow_pickle=False)
        np.save(
            staging_dir / "record_ids.npy",
            primitives._fixed_unicode_array(record_ids),
            allow_pickle=False,
        )
        np.save(
            staging_dir / "hash_ids.npy",
            np.asarray(hash_ids, dtype="<U64"),
            allow_pickle=False,
        )
        pd.DataFrame(metadata_rows).to_parquet(
            staging_dir / "records.parquet", index=False
        )
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
            "centers": centers,
            "excluded_centers": sorted(excluded_centers),
            "waveform": {
                "file": "signals.npy",
                "shape": [record_count, target_num_samples, len(lead_order)],
                "dtype": cache_dtype.name,
                "sampling_rate_hz": target_fs,
                "duration_seconds": duration_seconds,
                "layout": "time_channel",
                "lead_order": lead_order,
                "physical_unit": "mV",
                "window_policy": "center_crop_long_then_linear_interpolate_right_pad_short",
                "resampling": "linear",
                "align_corners": True,
                "implementation": "torch.nn.functional.interpolate",
                "source_sampling_rate_hz": "native",
                "information_note": "linear_grid_adapter_without_antialias_filter",
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
                    "source_sampling_rate_hz": target_fs,
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
                    row["quality_status"] == "repaired" for row in metadata_rows
                ),
                "failed_record_count": len(failed_rows),
                "failed_records_file": "failed_records.jsonl",
                "post_resample_finite_check": True,
                "post_dtype_cast_finite_check": True,
            },
            "labels": {
                "file": "labels_super5.npy",
                "dtype": "uint8",
                "class_order": list(label_mapping["class_order"]),
                "mapping_file": str(config["label_mapping_file"]),
                "mapping_version": label_mapping["version"],
                "mapping_hash": label_mapping["hash"],
                "all_zero_policy": label_mapping["all_zero_policy"],
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

    print(f"[PN2021] cache written to {cache_dir}")
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="PN2021 preprocessing YAML (a copied config bundle is supported)",
    )
    args = parser.parse_args()
    build_pn2021_npy_cache(args.config)


if __name__ == "__main__":
    main()
