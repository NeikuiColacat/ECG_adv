"""Build the manifest-owned five-center PN2021-C depth-2/3 NumPy cache.

Runtime dependencies are intentionally restricted to the manual-refactor
whitelist: the new PN2021 cache artifact, tracked augmentation/seed YAML files,
and ``util.augmentations``. No legacy PN2021-C loader or AugMix bridge is used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.load_cache import (  # noqa: E402
    EXPECTED_CLASS_ORDER,
    EXPECTED_LEADS,
)
from util.augmentations.operators import (  # noqa: E402
    UPSTREAM_COMMIT,
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)
from util.augmentations.profile import (  # noqa: E402
    AugmentationProfile,
    load_augmentation_profile,
)
from util.config_bundle import (  # noqa: E402
    load_yaml_mapping as _read_yaml_mapping,
    resolve_config_reference,
    resolve_entry_config_path,
)
from util import pn2021_artifact_contract as _artifact_contract  # noqa: E402
from util.random_seed import (  # noqa: E402
    DEFAULT_RANDOM_SEED_CONFIG_PATH,
    make_numpy_rng,
    make_python_rng,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "augmentation" / "cache.yaml"
OPERATOR_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "powerline_noise": powerline_noise,
    "emg_noise": emg_noise,
    "baseline_wander": baseline_wander,
    "baseline_shift": baseline_shift,
    "random_leads_masking": random_leads_masking,
}
OPERATORS_WITH_FREQ = {
    "powerline_noise",
    "baseline_wander",
    "baseline_shift",
}


def load_cache_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked PN2021-C cache contract."""

    config = _read_yaml_mapping(path, description="augmentation cache config")
    if config.get("schema_version") != 1:
        raise ValueError("augmentation cache schema_version must be 1")
    if config.get("dataset") != "pn2021c":
        raise ValueError("augmentation cache dataset must be 'pn2021c'")

    source = config.get("source")
    corruption = config.get("corruption")
    output = config.get("output")
    execution = config.get("execution")
    if not all(isinstance(item, dict) for item in (source, corruption, output, execution)):
        raise ValueError("source, corruption, output and execution must be mappings")

    centers = [str(value) for value in source["centers"]]
    if len(centers) != 5 or len(set(centers)) != 5:
        raise ValueError(f"source.centers must contain five unique centers: {centers}")
    if centers != [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
    ]:
        raise ValueError(f"unexpected five-center order: {centers}")

    if int(corruption["domain_sampling_rate_hz"]) != 500:
        raise ValueError("corruption domain must be 500 Hz")
    if [int(value) for value in corruption["output_sampling_rates_hz"]] != [100, 500]:
        raise ValueError("output sampling rates must be [100, 500]")
    if [int(value) for value in corruption["depths"]] != [2, 3]:
        raise ValueError("corruption depths must be [2, 3]")
    if corruption["combination_policy"] != "all_canonical_combinations_without_replacement":
        raise ValueError("unsupported combination_policy")
    if corruption["operator_order_policy"] != "canonical_order":
        raise ValueError("unsupported operator_order_policy")
    if corruption["normalization"] != "none" or corruption["signal_units"] != "mV":
        raise ValueError("PN2021-C cache must remain unnormalized physical-mV data")

    if output["layout"] != "view_record_time_channel":
        raise ValueError("output.layout must be 'view_record_time_channel'")
    if np.dtype(output["dtype"]) != np.dtype("float32"):
        raise ValueError("output dtype must be float32")
    if output["interpolation_500_to_100"] != "linear_align_corners":
        raise ValueError("500-to-100 interpolation must be linear_align_corners")
    if execution["backend"] != "numpy":
        raise ValueError("only the whitelist NumPy backend is supported")
    if int(execution["chunk_size"]) <= 0:
        raise ValueError("execution.chunk_size must be positive")
    if int(execution["checkpoint_every_chunks"]) <= 0:
        raise ValueError("execution.checkpoint_every_chunks must be positive")
    return config


def load_operators_profile(
    path: str | Path,
    *,
    owner_config_path: str | Path = DEFAULT_CONFIG,
    config_root: str | Path | None = None,
    profile_name: str | None = None,
    severity: int | None = None,
    expected_seed_config_path: str | Path | None = None,
) -> AugmentationProfile:
    """Resolve one operators reference through the shared strict loader."""

    resolved = resolve_config_reference(
        str(path),
        owner_config_path=owner_config_path,
        config_root=config_root,
        description="augmentation operators config",
        must_exist=True,
    )
    return load_augmentation_profile(
        resolved,
        profile_name=profile_name,
        severity=severity,
        expected_canonical_order=tuple(OPERATOR_FUNCTIONS),
        expected_seed_config_path=expected_seed_config_path,
        config_root=config_root,
    )
def build_compositions(
    config: dict[str, Any],
    operator_config: AugmentationProfile,
) -> list[dict[str, Any]]:
    """Expand five operators into all 10 pairs and all 10 triples."""

    canonical = list(operator_config.canonical_order)
    corruption = config["corruption"]
    declared_counts = {
        int(depth): int(count)
        for depth, count in corruption["combinations_per_depth"].items()
    }
    compositions: list[dict[str, Any]] = []
    for depth in [int(value) for value in corruption["depths"]]:
        combinations = list(itertools.combinations(canonical, depth))
        if len(combinations) != declared_counts[depth]:
            raise ValueError(
                f"depth {depth} declares {declared_counts[depth]} combinations, "
                f"but canonical expansion produces {len(combinations)}"
            )
        for operators in combinations:
            compositions.append(
                {
                    "view_index": len(compositions),
                    "depth": depth,
                    "operators": list(operators),
                    "composition_id": f"d{depth}__{'__'.join(operators)}",
                }
            )
    if len(compositions) != 20:
        raise ValueError(f"depth2+3 must produce 20 views, got {len(compositions)}")
    return compositions


def _severity_params(
    *,
    operator_name: str,
    config: dict[str, Any],
    operator_config: AugmentationProfile,
) -> dict[str, Any]:
    if str(config["corruption"]["profile"]) != operator_config.profile_name:
        raise ValueError("cache profile does not match the loaded operator profile")
    if int(config["corruption"]["severity"]) != operator_config.severity:
        raise ValueError("cache severity does not match the loaded operator profile")
    return operator_config.parameters_for(operator_name)


def apply_composition(
    signal_500hz: np.ndarray,
    *,
    source_hash: str,
    composition: dict[str, Any],
    config: dict[str, Any],
    operator_config: AugmentationProfile,
    seed_config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> np.ndarray:
    """Apply one deterministic composition to one raw-mV 500 Hz ECG."""

    output = np.asarray(signal_500hz)
    if output.shape != (5000, 12):
        raise ValueError(f"500 Hz source signal must have shape (5000, 12), got {output.shape}")
    if not np.issubdtype(output.dtype, np.floating):
        raise TypeError(f"source signal must be floating point, got {output.dtype}")
    if not np.isfinite(output).all():
        raise ValueError("source signal contains NaN or infinity")
    output = np.ascontiguousarray(output, dtype=np.float32)

    corruption = config["corruption"]
    seed_identity = (
        config["cache_version"],
        corruption["profile"],
        int(corruption["severity"]),
        int(corruption["domain_sampling_rate_hz"]),
        str(source_hash),
        composition["composition_id"],
    )
    for operator_index, operator_name in enumerate(composition["operators"]):
        function = OPERATOR_FUNCTIONS[str(operator_name)]
        params = _severity_params(
            operator_name=str(operator_name),
            config=config,
            operator_config=operator_config,
        )
        params["rng"] = make_numpy_rng(
            "augmentations_cache",
            *seed_identity,
            operator_index,
            operator_name,
            config_path=seed_config_path,
        )
        if operator_name in OPERATORS_WITH_FREQ:
            params["freq"] = float(corruption["domain_sampling_rate_hz"])
        if operator_name == "random_leads_masking":
            params["python_rng"] = make_python_rng(
                "augmentations_cache_python",
                *seed_identity,
                operator_index,
                operator_name,
                config_path=seed_config_path,
            )
        output = function(output, **params)
        if output.shape != (5000, 12) or not np.isfinite(output).all():
            raise ValueError(
                f"operator {operator_name} produced an invalid waveform for "
                f"{source_hash}/{composition['composition_id']}"
            )
    return np.ascontiguousarray(output, dtype=np.float32)


def linear_interpolate_time_batch(
    signals_ntc: np.ndarray,
    target_num_samples: int,
) -> np.ndarray:
    """Aligned-corner linear resize for a float32 ``(N,time,12)`` batch."""

    signals = np.asarray(signals_ntc)
    if signals.ndim != 3 or signals.shape[1] < 2 or signals.shape[2] != 12:
        raise ValueError(f"expected (N,time,12) signals, got {signals.shape}")
    if int(target_num_samples) < 2:
        raise ValueError("target_num_samples must be at least 2")
    if not np.isfinite(signals).all():
        raise ValueError("cannot interpolate non-finite signals")

    import torch
    import torch.nn.functional as torch_functional

    source = torch.from_numpy(
        np.ascontiguousarray(signals, dtype=np.float32)
    ).permute(0, 2, 1)
    with torch.no_grad():
        resized = torch_functional.interpolate(
            source,
            size=int(target_num_samples),
            mode="linear",
            align_corners=True,
        )
    output = resized.permute(0, 2, 1).contiguous().numpy()
    return np.ascontiguousarray(output, dtype=np.float32)


def _stable_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_source_contract(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    source = config["source"]
    source_dir = resolve_entry_config_path(source["cache_dir"])
    manifest_path = source_dir / str(source["manifest_file"])
    if not source_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"whitelist-produced PN2021 cache is incomplete: {source_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3 or manifest.get("dataset") != "pn2021":
        raise ValueError("source must be the schema-v3 manual-refactor PN2021 cache")
    waveform = manifest.get("waveform", {})
    derived = manifest.get("derived_waveforms", {}).get("500hz_linear", {})
    labels = manifest.get("labels", {})
    if waveform.get("shape", [None])[1:] != [1000, 12]:
        raise ValueError(f"unexpected source 100 Hz shape: {waveform.get('shape')}")
    if waveform.get("sampling_rate_hz") != 100:
        raise ValueError("source base waveform must be 100 Hz")
    if waveform.get("lead_order") != list(EXPECTED_LEADS):
        raise ValueError("source lead order does not match PTB-XL order")
    if waveform.get("physical_unit") != "mV" or waveform.get("normalization") != "none":
        raise ValueError("source waveform must be unnormalized physical-mV data")
    if derived.get("shape", [None])[1:] != [5000, 12]:
        raise ValueError(f"unexpected source 500 Hz shape: {derived.get('shape')}")
    if derived.get("sampling_rate_hz") != 500 or derived.get("source_file") != waveform.get("file"):
        raise ValueError("500 Hz source must be derived from the canonical 100 Hz cache")
    if labels.get("class_order") != list(EXPECTED_CLASS_ORDER):
        raise ValueError("source Super5 class order mismatch")
    if labels.get("mapping_version") != source["required_mapping_version"]:
        raise ValueError("source Super5 mapping version mismatch")
    if labels.get("mapping_hash") != source["required_mapping_hash"]:
        raise ValueError("source Super5 mapping hash mismatch")
    return source_dir, manifest, _artifact_contract.sha256_file(manifest_path)


def _select_source_records(
    *,
    source_dir: Path,
    config: dict[str, Any],
    source_record_count: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    source = config["source"]
    records = pd.read_parquet(source_dir / str(source["metadata_file"]))
    if len(records) != source_record_count:
        raise ValueError(
            f"source metadata has {len(records)} rows, expected {source_record_count}"
        )
    required_columns = {"cache_index", "record_id", "record_key", "hash_id", "center"}
    missing = sorted(required_columns - set(records.columns))
    if missing:
        raise ValueError(f"source metadata missing columns: {missing}")
    centers = set(str(value) for value in source["centers"])
    selected = records.loc[records["center"].isin(centers)].copy()
    present_centers = set(str(value) for value in selected["center"].unique())
    if present_centers != centers:
        raise ValueError(
            f"source cache center mismatch: expected {sorted(centers)}, "
            f"found {sorted(present_centers)}"
        )
    source_indices = selected["cache_index"].to_numpy(dtype=np.int64, copy=True)
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("selected source cache indices are not unique")
    selected.insert(0, "source_cache_index", source_indices)
    selected["cache_index"] = np.arange(len(selected), dtype=np.int64)
    selected.reset_index(drop=True, inplace=True)
    return selected, source_indices


def _validate_source_arrays(
    *,
    source_dir: Path,
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    selected_records: pd.DataFrame,
    source_indices: np.ndarray,
) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    source = config["source"]
    record_count = int(source_manifest["record_count"])
    signals_100 = np.load(
        source_dir / str(source["waveform_100hz_file"]), mmap_mode="r"
    )
    signals_500 = np.load(
        source_dir / str(source["waveform_500hz_file"]), mmap_mode="r"
    )
    labels = np.load(source_dir / str(source["labels_file"]), mmap_mode="r")
    record_ids = np.load(source_dir / str(source["record_ids_file"]), mmap_mode="r")
    hash_ids = np.load(source_dir / str(source["hash_ids_file"]), mmap_mode="r")
    if signals_100.shape != (record_count, 1000, 12):
        raise ValueError(f"source 100 Hz array shape mismatch: {signals_100.shape}")
    if signals_500.shape != (record_count, 5000, 12):
        raise ValueError(f"source 500 Hz array shape mismatch: {signals_500.shape}")
    if labels.shape != (record_count, 5):
        raise ValueError(f"source labels shape mismatch: {labels.shape}")
    if record_ids.shape != (record_count,) or hash_ids.shape != (record_count,):
        raise ValueError("source ID arrays do not match source record_count")
    if signals_100.dtype != np.float32 or signals_500.dtype != np.float32:
        raise ValueError("source waveform arrays must be float32")

    selected_labels = np.asarray(labels[source_indices], dtype=np.uint8)
    selected_record_ids = np.asarray(record_ids[source_indices])
    selected_hash_ids = np.asarray(hash_ids[source_indices])
    if selected_records["record_id"].astype(str).tolist() != selected_record_ids.astype(str).tolist():
        raise ValueError("record_ids.npy is not aligned with records.parquet")
    if selected_records["hash_id"].astype(str).tolist() != selected_hash_ids.astype(str).tolist():
        raise ValueError("hash_ids.npy is not aligned with records.parquet")
    if len(set(selected_hash_ids.astype(str))) != len(selected_hash_ids):
        raise ValueError("selected source hash IDs are not unique")
    return signals_500, selected_labels, selected_record_ids, selected_hash_ids


def _prepare_staging(
    *,
    output_dir: Path,
    config_identity: dict[str, Any],
    config: dict[str, Any],
    compositions: list[dict[str, Any]],
    record_count: int,
) -> tuple[Path, np.memmap, np.memmap, dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed cache: {output_dir}")
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")
    state_path = staging_dir / "build_state.json"
    shape_100 = (len(compositions), record_count, 1000, 12)
    shape_500 = (len(compositions), record_count, 5000, 12)
    resume = bool(config["execution"]["resume"])

    if staging_dir.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError(
                f"staging directory exists without an allowed resume: {staging_dir}"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_identity_hash") != config_identity["config_identity_hash"]:
            raise ValueError("staging config identity does not match current inputs")
        signals_100 = np.load(
            staging_dir / str(config["output"]["waveform_100hz_file"]),
            mmap_mode="r+",
        )
        signals_500 = np.load(
            staging_dir / str(config["output"]["waveform_500hz_file"]),
            mmap_mode="r+",
        )
        if signals_100.shape != shape_100 or signals_500.shape != shape_500:
            raise ValueError("staging waveform shapes do not match the current contract")
        return staging_dir, signals_100, signals_500, state

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = (
        math.prod(shape_100) + math.prod(shape_500)
    ) * np.dtype("float32").itemsize
    disk = shutil.disk_usage(output_dir.parent)
    reserve_bytes = max(int(disk.total * 0.10), 50 * 1024**3)
    if disk.free - estimated_bytes < reserve_bytes:
        raise OSError(
            "insufficient target-disk safety margin: "
            f"free={disk.free}, estimated_cache={estimated_bytes}, "
            f"required_reserve={reserve_bytes}"
        )

    staging_dir.mkdir()
    signals_100 = np.lib.format.open_memmap(
        staging_dir / str(config["output"]["waveform_100hz_file"]),
        mode="w+",
        dtype=np.float32,
        shape=shape_100,
    )
    signals_500 = np.lib.format.open_memmap(
        staging_dir / str(config["output"]["waveform_500hz_file"]),
        mode="w+",
        dtype=np.float32,
        shape=shape_500,
    )
    state = {
        **config_identity,
        "status": "building",
        "next_view_index": 0,
        "next_record_index": 0,
        "shape_100hz": list(shape_100),
        "shape_500hz": list(shape_500),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(state_path, state)
    return staging_dir, signals_100, signals_500, state


def _validate_complete_output(
    signals: np.ndarray,
    *,
    name: str,
    chunk_records: int,
) -> None:
    for view_index in range(signals.shape[0]):
        for start in range(0, signals.shape[1], chunk_records):
            end = min(start + chunk_records, signals.shape[1])
            if not np.isfinite(signals[view_index, start:end]).all():
                raise ValueError(
                    f"non-finite values in completed {name} cache at "
                    f"view={view_index}, records={start}:{end}"
                )


def _load_cache_operator_profile(
    *,
    config_path: Path,
    config: dict[str, Any],
) -> AugmentationProfile:
    corruption = config["corruption"]
    seed_path = resolve_config_reference(
        corruption["random_seed_file"],
        owner_config_path=config_path,
        description="corruption.random_seed_file",
        must_exist=True,
    )
    return load_operators_profile(
        corruption["operators_config"],
        owner_config_path=config_path,
        profile_name=str(corruption["profile"]),
        severity=int(corruption["severity"]),
        expected_seed_config_path=seed_path,
    )


def describe_cache_plan(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Resolve record counts, shapes and disk bytes without creating output."""

    config_path = resolve_entry_config_path(config_path)
    config = load_cache_config(config_path)
    operator_profile = _load_cache_operator_profile(
        config_path=config_path,
        config=config,
    )
    compositions = build_compositions(config, operator_profile)
    source_dir, source_manifest, source_manifest_sha256 = _load_source_contract(config)
    selected_records, _ = _select_source_records(
        source_dir=source_dir,
        config=config,
        source_record_count=int(source_manifest["record_count"]),
    )
    shape_100 = (len(compositions), len(selected_records), 1000, 12)
    shape_500 = (len(compositions), len(selected_records), 5000, 12)
    bytes_100 = math.prod(shape_100) * np.dtype("float32").itemsize
    bytes_500 = math.prod(shape_500) * np.dtype("float32").itemsize
    return {
        "dataset": config["dataset"],
        "cache_version": config["cache_version"],
        "source_cache_dir": str(source_dir),
        "source_manifest_sha256": source_manifest_sha256,
        "output_cache_dir": str(resolve_entry_config_path(config["output"]["cache_dir"])),
        "centers": list(config["source"]["centers"]),
        "center_counts": {
            str(center): int(count)
            for center, count in selected_records["center"].value_counts().sort_index().items()
        },
        "record_count": len(selected_records),
        "view_count": len(compositions),
        "shape_100hz": list(shape_100),
        "shape_500hz": list(shape_500),
        "estimated_bytes_100hz": bytes_100,
        "estimated_bytes_500hz": bytes_500,
        "estimated_bytes_total": bytes_100 + bytes_500,
    }


def build_augmentations_cache(
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    """Materialize deterministic five-center PN2021-C at 100 and 500 Hz."""

    config_path = resolve_entry_config_path(config_path)
    config = load_cache_config(config_path)
    operator_profile = _load_cache_operator_profile(
        config_path=config_path,
        config=config,
    )
    compositions = build_compositions(config, operator_profile)
    profile_name = operator_profile.profile_name
    declared_seed = operator_profile.random_seed_config
    seed_path = declared_seed.path

    source_dir, source_manifest, source_manifest_sha256 = _load_source_contract(config)
    source_record_count = int(source_manifest["record_count"])
    selected_records, source_indices = _select_source_records(
        source_dir=source_dir,
        config=config,
        source_record_count=source_record_count,
    )
    (
        source_signals_500,
        selected_labels,
        selected_record_ids,
        selected_hash_ids,
    ) = _validate_source_arrays(
        source_dir=source_dir,
        config=config,
        source_manifest=source_manifest,
        selected_records=selected_records,
        source_indices=source_indices,
    )

    config_identity = {
        "config_sha256": _artifact_contract.sha256_file(config_path),
        "operators_sha256": operator_profile.config_sha256,
        "random_seed_sha256": declared_seed.sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    config_identity["config_identity_hash"] = _stable_payload_hash(config_identity)
    output_dir = resolve_entry_config_path(config["output"]["cache_dir"])
    staging_dir, output_100, output_500, state = _prepare_staging(
        output_dir=output_dir,
        config_identity=config_identity,
        config=config,
        compositions=compositions,
        record_count=len(selected_records),
    )
    state_path = staging_dir / "build_state.json"

    if not (staging_dir / "source_indices.npy").exists():
        np.save(staging_dir / "source_indices.npy", source_indices, allow_pickle=False)
        np.save(staging_dir / "labels_super5.npy", selected_labels, allow_pickle=False)
        np.save(staging_dir / "record_ids.npy", selected_record_ids, allow_pickle=False)
        np.save(staging_dir / "hash_ids.npy", selected_hash_ids, allow_pickle=False)
        selected_records.to_parquet(staging_dir / "records.parquet", index=False)
        (staging_dir / "compositions.json").write_text(
            json.dumps(compositions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "resolved_cache_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (staging_dir / "operators_config_snapshot.yaml").write_text(
            yaml.safe_dump(
                operator_profile.snapshot(), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )

    chunk_size = int(config["execution"]["chunk_size"])
    checkpoint_every = int(config["execution"]["checkpoint_every_chunks"])
    checkpoint_counter = 0
    start_view = int(state["next_view_index"])
    start_record = int(state["next_record_index"])
    total_records = len(selected_records)

    for view_index in range(start_view, len(compositions)):
        composition = compositions[view_index]
        record_start = start_record if view_index == start_view else 0
        for start in range(record_start, total_records, chunk_size):
            end = min(start + chunk_size, total_records)
            source_batch = np.asarray(
                source_signals_500[source_indices[start:end]], dtype=np.float32
            )
            corrupted_500 = np.empty_like(source_batch, dtype=np.float32)
            for local_index, source_signal in enumerate(source_batch):
                record_index = start + local_index
                corrupted_500[local_index] = apply_composition(
                    source_signal,
                    source_hash=str(selected_hash_ids[record_index]),
                    composition=composition,
                    config=config,
                    operator_config=operator_profile,
                    seed_config_path=seed_path,
                )
            if not np.isfinite(corrupted_500).all():
                raise ValueError(
                    f"non-finite 500 Hz corruption at view={view_index}, "
                    f"records={start}:{end}"
                )
            corrupted_100 = linear_interpolate_time_batch(corrupted_500, 1000)
            if not np.isfinite(corrupted_100).all():
                raise ValueError(
                    f"non-finite 100 Hz corruption at view={view_index}, "
                    f"records={start}:{end}"
                )
            output_500[view_index, start:end] = corrupted_500
            output_100[view_index, start:end] = corrupted_100
            checkpoint_counter += 1

            next_view = view_index
            next_record = end
            if end == total_records:
                next_view = view_index + 1
                next_record = 0
            if checkpoint_counter >= checkpoint_every or end == total_records:
                output_500.flush()
                output_100.flush()
                state.update(
                    {
                        "next_view_index": next_view,
                        "next_record_index": next_record,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                _write_json_atomic(state_path, state)
                checkpoint_counter = 0
                print(
                    f"[PN2021-C] view={view_index + 1}/{len(compositions)} "
                    f"records={end}/{total_records} "
                    f"composition={composition['composition_id']}"
                )
        start_record = 0

    output_500.flush()
    output_100.flush()
    _validate_complete_output(
        output_500,
        name="500 Hz",
        chunk_records=max(chunk_size, 64),
    )
    _validate_complete_output(
        output_100,
        name="100 Hz",
        chunk_records=max(chunk_size, 64),
    )

    center_counts = {
        str(center): int(count)
        for center, count in selected_records["center"].value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config["dataset"],
        "cache_version": config["cache_version"],
        "record_count": total_records,
        "view_count": len(compositions),
        "corrupted_record_view_count": total_records * len(compositions),
        "centers": list(config["source"]["centers"]),
        "center_counts": center_counts,
        "source": {
            "cache_dir": str(source_dir),
            "manifest_sha256": source_manifest_sha256,
            "source_record_count": source_record_count,
            "source_indices_file": "source_indices.npy",
            "source_waveform_500hz_file": str(config["source"]["waveform_500hz_file"]),
            "mapping_version": source_manifest["labels"]["mapping_version"],
            "mapping_hash": source_manifest["labels"]["mapping_hash"],
            "class_order": list(EXPECTED_CLASS_ORDER),
            "lead_order": list(EXPECTED_LEADS),
            "physical_unit": "mV",
            "normalization": "none",
        },
        "corruption": {
            "profile": profile_name,
            "severity": int(config["corruption"]["severity"]),
            "domain_sampling_rate_hz": 500,
            "depths": [2, 3],
            "combination_policy": config["corruption"]["combination_policy"],
            "operator_order_policy": config["corruption"]["operator_order_policy"],
            "compositions_file": "compositions.json",
            "operators_upstream_commit": UPSTREAM_COMMIT,
            "random_seed": int(declared_seed.base_seed),
            "random_seed_derivation": (
                "sha256(base_seed,cache_version,profile,severity,domain_hz,"
                "source_hash,composition_id,operator_index,operator_name)"
            ),
        },
        "waveforms": {
            "100hz": {
                "file": str(config["output"]["waveform_100hz_file"]),
                "shape": list(output_100.shape),
                "dtype": "float32",
                "sampling_rate_hz": 100,
                "duration_seconds": 10,
                "layout": config["output"]["layout"],
                "derived_from": "corrupted_500hz",
                "interpolation": "linear",
                "align_corners": True,
                "normalization": "none",
            },
            "500hz": {
                "file": str(config["output"]["waveform_500hz_file"]),
                "shape": list(output_500.shape),
                "dtype": "float32",
                "sampling_rate_hz": 500,
                "duration_seconds": 10,
                "layout": config["output"]["layout"],
                "corruption_domain": True,
                "normalization": "none",
            },
        },
        "identity": config_identity,
        "files": {
            "labels": "labels_super5.npy",
            "record_ids": "record_ids.npy",
            "hash_ids": "hash_ids.npy",
            "metadata": "records.parquet",
            "source_indices": "source_indices.npy",
            "compositions": "compositions.json",
            "resolved_config": "resolved_cache_config.yaml",
            "operators_snapshot": "operators_config_snapshot.yaml",
            "build_state": "build_state.json",
        },
        "validation": {
            "full_100hz_isfinite": True,
            "full_500hz_isfinite": True,
            "source_hash_unique": True,
            "metadata_alignment": True,
        },
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    state.update(
        {
            "status": "complete",
            "next_view_index": len(compositions),
            "next_record_index": 0,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json_atomic(state_path, state)
    del output_100
    del output_500
    del source_signals_500
    staging_dir.replace(output_dir)
    print(f"[PN2021-C] cache written to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="tracked PN2021-C cache YAML",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="resolve identities, shapes and disk estimate without writing a cache",
    )
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(describe_cache_plan(args.config), ensure_ascii=False, indent=2))
    else:
        build_augmentations_cache(args.config)


if __name__ == "__main__":
    main()
