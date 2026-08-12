"""Validated runtime boundary from ECG caches and splits to PyTorch batches.

This module intentionally owns only the data-runtime layer:

1. verify an immutable split against the cache that produced it;
2. resolve stable hash IDs to cache-local integer indices;
3. expose a worker-safe PyTorch ``Dataset``;
4. apply model-side transforms in the fixed order
   raw mV -> optional augmentation -> non-finite sanitization -> global z-score;
5. build a deterministically seeded ``DataLoader``.

It does not resample waveforms, create splits, compute metrics, or import legacy
training code. Callers choose the already-built 100 Hz or 500 Hz cache view.
PN2021-C callers must also choose one explicit corruption composition.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data_preprocess.load_cache import (
    EXPECTED_CLASS_ORDER,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    ECGCache,
    StorageMode,
    ValueValidation,
    ViewSelector,
    load_cache,
)
from data_preprocess.split_cache import load_split_config
from util.config_bundle import (
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.random_seed import (
    DEFAULT_RANDOM_SEED_CONFIG_PATH,
    derive_seed,
    load_random_seed_config,
    seed_dataloader_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "configs" / "data" / "splits.yaml"
DEFAULT_DATA_LOAD_CONFIG = PROJECT_ROOT / "configs" / "data" / "data_load.yaml"
DEFAULT_CORRUPTION_CACHE_CONFIG = (
    PROJECT_ROOT / "configs" / "augmentation" / "cache.yaml"
)
RuntimeLayout = Literal["time_channel", "channel_time"]
RuntimeDatasetName = Literal["ptbxl", "pn2021", "pn2021c"]
MMapAccessOrder = Literal["split", "cache_index"]
MMapAdvice = Literal["none", "sequential_willneed"]
PTBXL_PARTITIONS = ("train", "validation", "test")
PN2021_K500_PARTITIONS = (
    "k500",
    "k500_tune_train",
    "k500_tune_validation",
)
PN2021_PARTITIONS = (
    *PN2021_K500_PARTITIONS,
    "evaluation_all_zero_kept",
    "evaluation_drop_all_zero",
)
CANONICAL_EVALUATION_PARTITIONS = {
    "pn2021_all_zero_kept_refexcluded": (
        "pn2021",
        "evaluation_all_zero_kept",
    ),
    "pn2021_drop_all_zero_refexcluded": (
        "pn2021",
        "evaluation_drop_all_zero",
    ),
    "pn2021c_all_zero_kept_corrupted_refexcluded": (
        "pn2021c",
        "evaluation_all_zero_kept",
    ),
    "pn2021c_drop_all_zero_corrupted_refexcluded": (
        "pn2021c",
        "evaluation_drop_all_zero",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_id_set_sha256(values: Sequence[str] | np.ndarray) -> str:
    ordered = sorted(str(value) for value in values)
    payload = "".join(f"{value}\n" for value in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON mapping: {path}")
    return payload


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _resolve_artifact(base_dir: Path, descriptor: Any, *, description: str) -> Path:
    entry = _require_mapping(descriptor, description=description)
    raw_file = entry.get("file")
    if not isinstance(raw_file, str) or not raw_file:
        raise ValueError(f"{description}.file must be a non-empty relative path")
    member = Path(raw_file)
    if member.is_absolute():
        raise ValueError(f"{description}.file must be relative: {raw_file}")
    path = (base_dir / member).resolve()
    try:
        path.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"{description}.file escapes split directory") from None
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    expected_sha = str(entry.get("sha256", ""))
    actual_sha = _sha256_file(path)
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(
            f"{description} SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    return path


def _load_artifact_array(
    base_dir: Path,
    descriptor: Any,
    *,
    description: str,
) -> np.ndarray:
    entry = _require_mapping(descriptor, description=description)
    path = _resolve_artifact(base_dir, entry, description=description)
    array = np.load(path, allow_pickle=False)
    expected_shape = tuple(int(value) for value in entry.get("shape", ()))
    expected_dtype = str(entry.get("dtype", ""))
    if array.shape != expected_shape or str(array.dtype) != expected_dtype:
        raise ValueError(
            f"{description} array contract mismatch: "
            f"actual={array.shape}/{array.dtype}, "
            f"expected={expected_shape}/{expected_dtype}"
        )
    if array.ndim != 1:
        raise ValueError(f"{description} must be one-dimensional")
    return array


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.flags.writeable = False
    return result


def _source_manifest_sha256(cache: ECGCache) -> str:
    if not cache.is_corruption:
        return cache.identity.manifest_sha256
    source = _require_mapping(
        cache.identity.raw_manifest.get("source"),
        description="PN2021-C manifest.source",
    )
    value = str(source.get("manifest_sha256", ""))
    if len(value) != 64:
        raise ValueError("PN2021-C source manifest SHA256 is missing or invalid")
    return value


def _mapping_identity(cache: ECGCache) -> tuple[str | None, str | None]:
    if not cache.is_corruption:
        return cache.identity.mapping_version, cache.identity.mapping_hash
    source = _require_mapping(
        cache.identity.raw_manifest.get("source"),
        description="PN2021-C manifest.source",
    )
    return str(source.get("mapping_version")), str(source.get("mapping_hash"))


@dataclass(frozen=True)
class ECGSelection:
    """A verified, immutable split selection resolved for one open cache."""

    dataset: str
    cache_dataset: str
    partition: str
    logical_center: str | None
    source_centers: tuple[str, ...]
    indices: np.ndarray
    hash_ids: np.ndarray
    record_ids: np.ndarray
    split_id: str
    split_manifest_path: Path
    split_manifest_sha256: str
    source_manifest_sha256: str
    hash_id_set_sha256: str
    class_order: tuple[str, ...]
    mapping_version: str | None
    mapping_hash: str | None

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def is_ref_excluded_evaluation(self) -> bool:
        return self.partition in {
            "evaluation_all_zero_kept",
            "evaluation_drop_all_zero",
        }

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cache_dataset": self.cache_dataset,
            "partition": self.partition,
            "logical_center": self.logical_center,
            "source_centers": list(self.source_centers),
            "record_count": len(self),
            "split_id": self.split_id,
            "split_manifest_path": str(self.split_manifest_path),
            "split_manifest_sha256": self.split_manifest_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "hash_id_set_sha256": self.hash_id_set_sha256,
            "class_order": list(self.class_order),
            "mapping_version": self.mapping_version,
            "mapping_hash": self.mapping_hash,
            "ref_excluded_evaluation": self.is_ref_excluded_evaluation,
        }


def _validate_split_cache_identity(
    cache: ECGCache,
    manifest: Mapping[str, Any],
    *,
    split_dataset: str,
) -> str:
    declared_source_sha = str(manifest.get("source_manifest_sha256", ""))
    actual_source_sha = _source_manifest_sha256(cache)
    if declared_source_sha != actual_source_sha:
        raise ValueError(
            "split/cache source manifest mismatch: "
            f"split={declared_source_sha}, cache_source={actual_source_sha}"
        )
    if tuple(manifest.get("class_order", ())) != EXPECTED_CLASS_ORDER:
        raise ValueError("split class order does not match CD/HYP/MI/NORM/STTC")
    if split_dataset == "ptbxl":
        if cache.dataset != "ptbxl":
            raise ValueError("PTB-XL splits can only be applied to a PTB-XL cache")
    else:
        if cache.dataset not in {"pn2021", "pn2021c"}:
            raise ValueError("PN2021 splits require a PN2021 or PN2021-C cache")
        split_mapping = (
            str(manifest.get("mapping_version", "")),
            str(manifest.get("mapping_hash", "")),
        )
        cache_mapping = _mapping_identity(cache)
        if split_mapping != cache_mapping:
            raise ValueError(
                f"split/cache label mapping mismatch: split={split_mapping}, "
                f"cache={cache_mapping}"
            )
        if split_mapping != (PN2021_MAPPING_VERSION, PN2021_MAPPING_HASH):
            raise ValueError(f"unsupported PN2021 mapping identity: {split_mapping}")
    return declared_source_sha


def _partition_hash_key(partition: str) -> str:
    return {
        "k500": "k500_hash_id_set_sha256",
        "k500_tune_train": "k500_tune_train_hash_id_set_sha256",
        "k500_tune_validation": (
            "k500_tune_validation_hash_id_set_sha256"
        ),
        "evaluation_all_zero_kept": (
            "evaluation_all_zero_kept_hash_id_set_sha256"
        ),
        "evaluation_drop_all_zero": (
            "evaluation_drop_all_zero_hash_id_set_sha256"
        ),
    }[partition]


def _validate_k500_tuning_contract(
    artifact_base: Path,
    center_manifest: Mapping[str, Any],
    file_groups: Mapping[str, Any],
) -> None:
    evidence = _require_mapping(
        center_manifest.get("k500_tuning_split"),
        description="k500_tuning_split",
    )
    if evidence.get("parent_partition") != "k500":
        raise ValueError("K500 tuning parent partition must be k500")
    if evidence.get("train_partition") != "k500_tune_train":
        raise ValueError("K500 tuning train partition identity mismatch")
    if evidence.get("validation_partition") != "k500_tune_validation":
        raise ValueError("K500 tuning validation partition identity mismatch")
    hashes: dict[str, np.ndarray] = {}
    for partition in PN2021_K500_PARTITIONS:
        files = _require_mapping(
            file_groups.get(partition),
            description=f"partition {partition} files",
        )
        hashes[partition] = _load_artifact_array(
            artifact_base,
            files.get("hash_ids"),
            description=f"{partition} hash IDs",
        ).astype(str, copy=False)
    parent = set(hashes["k500"].tolist())
    train = set(hashes["k500_tune_train"].tolist())
    validation = set(hashes["k500_tune_validation"].tolist())
    overlap = train.intersection(validation)
    union = train.union(validation)
    expected = {
        "k500": (
            "parent_count",
            "parent_hash_id_set_sha256",
            "k500_count",
            "k500_hash_id_set_sha256",
        ),
        "k500_tune_train": (
            "train_count",
            "train_hash_id_set_sha256",
            "k500_tune_train_count",
            "k500_tune_train_hash_id_set_sha256",
        ),
        "k500_tune_validation": (
            "validation_count",
            "validation_hash_id_set_sha256",
            "k500_tune_validation_count",
            "k500_tune_validation_hash_id_set_sha256",
        ),
    }
    for partition, (
        evidence_count_key,
        evidence_hash_key,
        manifest_count_key,
        manifest_hash_key,
    ) in expected.items():
        values = hashes[partition]
        actual_hash = _hash_id_set_sha256(values)
        if int(evidence.get(evidence_count_key, -1)) != values.size:
            raise ValueError(f"{partition} evidence count mismatch")
        if int(center_manifest.get(manifest_count_key, -1)) != values.size:
            raise ValueError(f"{partition} manifest count mismatch")
        if str(evidence.get(evidence_hash_key, "")) != actual_hash:
            raise ValueError(f"{partition} evidence hash mismatch")
        if str(center_manifest.get(manifest_hash_key, "")) != actual_hash:
            raise ValueError(f"{partition} manifest hash mismatch")
    if overlap or union != parent:
        raise ValueError(
            "K500 tuning partitions are not disjoint or do not cover the parent"
        )
    union_hash = _hash_id_set_sha256(tuple(union))
    if (
        int(evidence.get("overlap_count", -1)) != 0
        or int(evidence.get("union_count", -1)) != len(parent)
        or evidence.get("union_matches_parent") is not True
        or str(evidence.get("union_hash_id_set_sha256", "")) != union_hash
    ):
        raise ValueError("K500 tuning union evidence mismatch")


def load_selection(
    cache: ECGCache,
    split_dir: str | Path,
    *,
    partition: str,
    logical_center: str | None = None,
) -> ECGSelection:
    """Load one split partition and bind it to ``cache`` by stable hash ID.

    ``split_dir`` is the dataset directory containing ``split_manifest.json``.
    For PN2021, ``logical_center`` is mandatory and ``cpsc_2018`` represents
    both physical sources ``cpsc_2018`` and ``cpsc_2018_extra``.
    """

    root = Path(split_dir).expanduser().resolve()
    manifest_path = root / "split_manifest.json"
    manifest = _read_json_mapping(manifest_path, description="split manifest")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("split manifest schema_version must be 1")
    split_dataset = str(manifest.get("dataset", ""))
    if split_dataset not in {"ptbxl", "pn2021"}:
        raise ValueError(f"unsupported split dataset: {split_dataset!r}")
    source_sha = _validate_split_cache_identity(
        cache, manifest, split_dataset=split_dataset
    )
    split_id = str(manifest.get("split_id", ""))
    if not split_id:
        raise ValueError("split manifest split_id is missing")

    if split_dataset == "ptbxl":
        if logical_center is not None:
            raise ValueError("PTB-XL split selection does not accept logical_center")
        if partition not in PTBXL_PARTITIONS:
            raise ValueError(f"PTB-XL partition must be one of {PTBXL_PARTITIONS}")
        partitions = _require_mapping(manifest.get("splits"), description="splits")
        overlap_counts = _require_mapping(
            manifest.get("patient_overlap_counts"),
            description="PTB-XL patient_overlap_counts",
        )
        if any(int(value) != 0 for value in overlap_counts.values()):
            raise ValueError(f"PTB-XL split declares patient leakage: {overlap_counts}")
        selection_manifest = _require_mapping(
            partitions.get(partition), description=f"split {partition}"
        )
        artifact_base = root
        expected_hash_set = str(
            selection_manifest.get("hash_id_set_sha256", "")
        )
        source_centers: tuple[str, ...] = ()
    else:
        if logical_center is None:
            raise ValueError("PN2021 split selection requires logical_center")
        if partition not in PN2021_PARTITIONS:
            raise ValueError(f"PN2021 partition must be one of {PN2021_PARTITIONS}")
        if cache.is_corruption and partition in {"k500", "k500_tune_train"}:
            raise ValueError(
                "PN2021-C may not supply K500 training records; only the frozen "
                "k500_tune_validation partition or ref-excluded evaluation is allowed"
            )
        centers = _require_mapping(
            manifest.get("logical_centers"), description="logical_centers"
        )
        parent_center = _require_mapping(
            centers.get(logical_center),
            description=f"logical center {logical_center}",
        )
        artifact_base = (root / str(logical_center)).resolve()
        try:
            artifact_base.relative_to(root)
        except ValueError:
            raise ValueError("logical_center escapes split directory") from None
        center_manifest_path = artifact_base / "split_manifest.json"
        expected_center_sha = str(parent_center.get("manifest_sha256", ""))
        actual_center_sha = _sha256_file(center_manifest_path)
        if expected_center_sha != actual_center_sha:
            raise ValueError(
                "logical-center manifest SHA256 mismatch: "
                f"expected={expected_center_sha}, actual={actual_center_sha}"
            )
        center_manifest = _read_json_mapping(
            center_manifest_path, description="logical-center split manifest"
        )
        if str(center_manifest.get("logical_center", "")) != str(logical_center):
            raise ValueError("logical-center manifest identity mismatch")
        source_centers = tuple(
            str(value) for value in center_manifest.get("source_centers", ())
        )
        if not source_centers or {"ptb-xl", "ptbxl"}.intersection(source_centers):
            raise ValueError("invalid or forbidden PN2021 physical source centers")
        file_groups = _require_mapping(
            center_manifest.get("files"), description="center files"
        )
        if partition in {"k500_tune_train", "k500_tune_validation"}:
            _validate_k500_tuning_contract(
                artifact_base,
                center_manifest,
                file_groups,
            )
        selection_manifest = {
            "files": _require_mapping(
                file_groups.get(partition), description=f"partition {partition} files"
            )
        }
        expected_hash_set = str(
            center_manifest.get(_partition_hash_key(partition), "")
        )
        reference_hash_descriptor = None
        if partition.startswith("evaluation_"):
            k500_files = _require_mapping(
                file_groups.get("k500"), description="partition k500 files"
            )
            reference_hash_descriptor = k500_files.get("hash_ids")

    if split_dataset == "ptbxl":
        reference_hash_descriptor = None

    files = _require_mapping(
        selection_manifest.get("files"), description=f"{partition}.files"
    )
    stored_indices = _load_artifact_array(
        artifact_base, files.get("indices"), description=f"{partition} indices"
    )
    hash_ids = _load_artifact_array(
        artifact_base, files.get("hash_ids"), description=f"{partition} hash IDs"
    ).astype(str, copy=False)
    record_ids = _load_artifact_array(
        artifact_base,
        files.get("record_ids"),
        description=f"{partition} record IDs",
    ).astype(str, copy=False)
    if not (stored_indices.size == hash_ids.size == record_ids.size):
        raise ValueError("split identity arrays have inconsistent lengths")
    if stored_indices.size == 0:
        raise ValueError(f"split partition {partition} is empty")
    if len(set(hash_ids.tolist())) != hash_ids.size:
        raise ValueError("split contains duplicate hash IDs")
    if reference_hash_descriptor is not None:
        reference_hashes = _load_artifact_array(
            artifact_base,
            reference_hash_descriptor,
            description="K500 reference hash IDs",
        ).astype(str, copy=False)
        overlap = set(hash_ids.tolist()).intersection(reference_hashes.tolist())
        if overlap:
            raise ValueError(
                "PN2021 evaluation contains K500 reference leakage: "
                f"count={len(overlap)}, preview={sorted(overlap)[:5]}"
            )
    actual_hash_set = _hash_id_set_sha256(hash_ids)
    if not expected_hash_set or actual_hash_set != expected_hash_set:
        raise ValueError(
            f"split hash-ID set mismatch: expected={expected_hash_set}, "
            f"actual={actual_hash_set}"
        )

    indices = cache.indices_for_hashes(hash_ids)
    if not cache.is_corruption and not np.array_equal(
        stored_indices.astype(np.int64, copy=False), indices
    ):
        raise ValueError("split indices no longer align with the source cache")
    cache_record_ids = np.asarray(cache.record_ids[indices]).astype(str, copy=False)
    if not np.array_equal(cache_record_ids, record_ids):
        raise ValueError("split record IDs no longer align with resolved hash IDs")
    labels = np.asarray(cache.labels[indices])
    if partition in {
        *PN2021_K500_PARTITIONS,
        "evaluation_drop_all_zero",
    }:
        if np.any(labels.sum(axis=1) == 0):
            raise ValueError(f"partition {partition} unexpectedly contains all-zero labels")
    if source_centers:
        if "center" not in cache.records.columns:
            raise ValueError("PN2021 cache lacks physical center metadata")
        actual_centers = set(
            cache.records.iloc[indices]["center"].astype(str).tolist()
        )
        if not actual_centers.issubset(set(source_centers)):
            raise ValueError(
                f"selection contains records outside logical center: {actual_centers}"
            )

    indices = _readonly(indices.astype(np.int64, copy=False))
    hash_ids = _readonly(np.asarray(hash_ids))
    record_ids = _readonly(np.asarray(record_ids))
    return ECGSelection(
        dataset=split_dataset,
        cache_dataset=cache.dataset,
        partition=str(partition),
        logical_center=None if logical_center is None else str(logical_center),
        source_centers=source_centers,
        indices=indices,
        hash_ids=hash_ids,
        record_ids=record_ids,
        split_id=split_id,
        split_manifest_path=manifest_path,
        split_manifest_sha256=_sha256_file(manifest_path),
        source_manifest_sha256=source_sha,
        hash_id_set_sha256=actual_hash_set,
        class_order=EXPECTED_CLASS_ORDER,
        mapping_version=_mapping_identity(cache)[0],
        mapping_hash=_mapping_identity(cache)[1],
    )


def assert_disjoint_selections(*selections: ECGSelection) -> None:
    """Raise if any two selections share a source ECG hash ID."""

    if len(selections) < 2:
        raise ValueError("at least two selections are required")
    for left_index, left in enumerate(selections[:-1]):
        left_hashes = set(left.hash_ids.tolist())
        for right in selections[left_index + 1 :]:
            overlap = left_hashes.intersection(right.hash_ids.tolist())
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(
                    f"selection leakage between {left.partition!r} and "
                    f"{right.partition!r}: count={len(overlap)}, preview={preview}"
                )


def sanitize_nonfinite(signal: torch.Tensor) -> torch.Tensor:
    """Replace NaN and +/-Inf after augmentation without mutating the input."""

    if not torch.is_floating_point(signal):
        signal = signal.to(dtype=torch.float32)
    return torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def per_sample_global_zscore(
    signal: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Z-score each ECG over all time points and leads together.

    The final two dimensions are treated as the waveform dimensions, so both
    ``(..., time, lead)`` and ``(..., lead, time)`` layouts are supported.
    Flat samples become zeros rather than NaN.
    """

    if signal.ndim < 2:
        raise ValueError("ECG tensor must have at least two dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    value = signal if torch.is_floating_point(signal) else signal.float()
    mean = value.mean(dim=(-2, -1), keepdim=True)
    std = value.std(dim=(-2, -1), correction=0, keepdim=True)
    return (value - mean) / std.clamp_min(float(epsilon))


def convert_layout(
    signal: torch.Tensor,
    *,
    source_layout: RuntimeLayout = "time_channel",
    output_layout: RuntimeLayout = "time_channel",
) -> torch.Tensor:
    """Convert the final two waveform axes without changing leading axes."""

    valid = {"time_channel", "channel_time"}
    if source_layout not in valid or output_layout not in valid:
        raise ValueError(f"layout must be one of {sorted(valid)}")
    if signal.ndim < 2:
        raise ValueError("ECG tensor must have at least two dimensions")
    if source_layout == output_layout:
        return signal.contiguous()
    return signal.transpose(-1, -2).contiguous()


def prepare_model_input(
    signal: np.ndarray | torch.Tensor,
    *,
    augmentation: Callable[[torch.Tensor], np.ndarray | torch.Tensor] | None = None,
    sanitize: bool = True,
    global_zscore: bool = True,
    epsilon: float = 1e-6,
    source_layout: RuntimeLayout = "time_channel",
    output_layout: RuntimeLayout = "channel_time",
) -> torch.Tensor:
    """Create one model tensor using the fixed runtime preprocessing order."""

    if isinstance(signal, torch.Tensor):
        value = signal.detach().clone().to(dtype=torch.float32)
    else:
        value = torch.as_tensor(np.array(signal, copy=True), dtype=torch.float32)
    original_shape = tuple(value.shape)
    if value.ndim < 2:
        raise ValueError("ECG waveform must have at least two dimensions")
    if augmentation is not None:
        augmented = augmentation(value)
        if isinstance(augmented, torch.Tensor):
            value = augmented.to(device=value.device, dtype=torch.float32)
        else:
            value = torch.as_tensor(augmented, device=value.device, dtype=torch.float32)
        if tuple(value.shape) != original_shape:
            raise ValueError(
                f"augmentation changed ECG shape from {original_shape} to {tuple(value.shape)}"
            )
    if sanitize:
        value = sanitize_nonfinite(value)
    elif not torch.isfinite(value).all():
        raise ValueError("non-finite values remain after runtime augmentation")
    if global_zscore:
        value = per_sample_global_zscore(value, epsilon=epsilon)
    if not torch.isfinite(value).all():
        raise ValueError("runtime transform produced a non-finite model input")
    return convert_layout(
        value, source_layout=source_layout, output_layout=output_layout
    )


@dataclass
class ECGModelTransform:
    """Pickle-friendly callable implementing the standard model-side transform."""

    augmentation: Callable[[torch.Tensor], np.ndarray | torch.Tensor] | None = None
    sanitize: bool = True
    global_zscore: bool = True
    epsilon: float = 1e-6
    source_layout: RuntimeLayout = "time_channel"
    output_layout: RuntimeLayout = "channel_time"

    def __call__(self, signal: np.ndarray | torch.Tensor) -> torch.Tensor:
        return prepare_model_input(
            signal,
            augmentation=self.augmentation,
            sanitize=self.sanitize,
            global_zscore=self.global_zscore,
            epsilon=self.epsilon,
            source_layout=self.source_layout,
            output_layout=self.output_layout,
        )


@dataclass(frozen=True)
class MMapPrefetchConfig:
    """Bounded Linux page-cache hints for ordered mmap evaluation reads."""

    enabled: bool = False
    advice: MMapAdvice = "sequential_willneed"
    window_mib: int = 1024
    ahead_batches: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("mmap prefetch enabled must be boolean")
        if self.advice not in {"none", "sequential_willneed"}:
            raise ValueError(
                "mmap prefetch advice must be none or sequential_willneed"
            )
        if (
            isinstance(self.window_mib, bool)
            or not isinstance(self.window_mib, int)
            or self.window_mib <= 0
        ):
            raise ValueError("mmap prefetch window_mib must be a positive integer")
        if (
            isinstance(self.ahead_batches, bool)
            or not isinstance(self.ahead_batches, int)
            or self.ahead_batches <= 0
        ):
            raise ValueError(
                "mmap prefetch ahead_batches must be a positive integer"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "advice": self.advice,
            "window_mib": self.window_mib,
            "ahead_batches": self.ahead_batches,
        }


class MMapPrefetcher:
    """Issue best-effort Linux access hints for one contiguous cache view.

    The helper owns a process-local read-only file descriptor. Advice affects
    only the kernel page-cache policy; it never changes waveform bytes or the
    split selection. Unsupported platforms and filesystems safely become a
    no-op.
    """

    def __init__(
        self,
        cache: ECGCache,
        *,
        view_index: int | None,
        config: MMapPrefetchConfig,
    ) -> None:
        self.config = config
        self._fd: int | None = None
        self._supported = False
        self._last_prefetch_end = 0
        self.record_bytes = 0
        self.view_offset = 0
        self.view_bytes = 0
        if (
            not config.enabled
            or config.advice == "none"
            or cache.storage_mode != "mmap"
            or not isinstance(cache.signals, np.memmap)
            or not hasattr(os, "posix_fadvise")
            or not hasattr(os, "POSIX_FADV_SEQUENTIAL")
            or not hasattr(os, "POSIX_FADV_WILLNEED")
        ):
            return

        record_shape = (
            cache.signals.shape[2:]
            if cache.is_corruption
            else cache.signals.shape[1:]
        )
        self.record_bytes = int(np.prod(record_shape)) * int(
            cache.signals.dtype.itemsize
        )
        self.view_bytes = len(cache) * self.record_bytes
        data_offset = int(cache.signals.offset)
        if cache.is_corruption:
            if view_index is None:
                raise ValueError("PN2021-C mmap prefetch requires a resolved view")
            self.view_offset = data_offset + int(view_index) * self.view_bytes
        else:
            self.view_offset = data_offset
        try:
            self._fd = os.open(cache.identity.signals_path, os.O_RDONLY)
            os.posix_fadvise(
                self._fd,
                self.view_offset,
                self.view_bytes,
                os.POSIX_FADV_SEQUENTIAL,
            )
            self._supported = True
        except (AttributeError, OSError):
            self.close()

    def prefetch_after(self, cache_indices: np.ndarray) -> None:
        """Hint a bounded window immediately after one increasing batch."""

        if not self._supported or self._fd is None:
            return
        indices = np.asarray(cache_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            return
        if indices.size > 1 and not np.all(np.diff(indices) > 0):
            return
        next_record = int(indices[-1]) + 1
        if next_record * self.record_bytes >= self.view_bytes:
            return
        requested = min(
            self.config.window_mib * 1024**2,
            int(indices.size) * self.config.ahead_batches * self.record_bytes,
            self.view_bytes - next_record * self.record_bytes,
        )
        start = self.view_offset + next_record * self.record_bytes
        if self._last_prefetch_end > start:
            overlap = self._last_prefetch_end - start
            start = self._last_prefetch_end
            requested -= overlap
        if requested <= 0:
            return
        try:
            os.posix_fadvise(
                self._fd,
                start,
                requested,
                os.POSIX_FADV_WILLNEED,
            )
            self._last_prefetch_end = start + requested
        except (AttributeError, OSError):
            self.close()

    def describe(self) -> dict[str, Any]:
        return {
            **self.config.describe(),
            "supported": self._supported,
            "record_bytes": self.record_bytes,
            "view_offset": self.view_offset,
            "view_bytes": self.view_bytes,
        }

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._supported = False


class RuntimeECGDataset(Dataset[dict[str, Any]]):
    """Lazy, worker-safe Dataset backed by the validated NumPy cache."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        split_dir: str | Path,
        partition: str,
        logical_center: str | None = None,
        sampling_rate_hz: int = 100,
        cache_mode: StorageMode = "auto",
        view: ViewSelector = None,
        transform: Callable[[np.ndarray | torch.Tensor], torch.Tensor] | None = None,
        validate_values: ValueValidation = "sample",
        access_order: MMapAccessOrder = "split",
        batch_read: bool = True,
        mmap_prefetch: MMapPrefetchConfig | None = None,
        shared_cache: ECGCache | None = None,
        shared_selection: ECGSelection | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.partition = str(partition)
        self.logical_center = logical_center
        self.sampling_rate_hz = int(sampling_rate_hz)
        self.requested_cache_mode = cache_mode
        self.view = view
        self.transform = transform
        self.validate_values = validate_values
        if access_order not in {"split", "cache_index"}:
            raise ValueError("access_order must be split or cache_index")
        if not isinstance(batch_read, bool):
            raise ValueError("batch_read must be boolean")
        if mmap_prefetch is not None and not isinstance(
            mmap_prefetch, MMapPrefetchConfig
        ):
            raise TypeError("mmap_prefetch must be an MMapPrefetchConfig")
        self.access_order = access_order
        self.batch_read = batch_read
        self.mmap_prefetch = mmap_prefetch or MMapPrefetchConfig()
        self._prefetcher: MMapPrefetcher | None = None
        self._prefetcher_pid: int | None = None
        self._resolved_view_index: int | None = None
        if (shared_cache is None) != (shared_selection is None):
            raise ValueError(
                "shared_cache and shared_selection must be supplied together"
            )
        self._owns_cache = shared_cache is None
        self._cache: ECGCache | None = (
            load_cache(
                self.cache_dir,
                sampling_rate_hz=self.sampling_rate_hz,
                mode=self.requested_cache_mode,
                validate_values=self.validate_values,
            )
            if shared_cache is None
            else shared_cache
        )
        self._cache_pid: int | None = os.getpid()
        try:
            if self._cache.identity.cache_dir != self.cache_dir:
                raise ValueError("shared cache directory differs from dataset cache_dir")
            if self._cache.sampling_rate_hz != self.sampling_rate_hz:
                raise ValueError("shared cache sampling rate differs from dataset request")
            self.cache_mode = self._cache.storage_mode
            self.selection = (
                load_selection(
                    self._cache,
                    self.split_dir,
                    partition=self.partition,
                    logical_center=self.logical_center,
                )
                if shared_selection is None
                else shared_selection
            )
            if (
                self.selection.partition != self.partition
                or self.selection.logical_center != self.logical_center
                or self.selection.source_manifest_sha256
                != _source_manifest_sha256(self._cache)
            ):
                raise ValueError("shared selection differs from dataset request/cache")
            if self.access_order == "cache_index":
                order = np.argsort(self.selection.indices, kind="stable")
                reordered = replace(
                    self.selection,
                    indices=_readonly(self.selection.indices[order]),
                    hash_ids=_readonly(self.selection.hash_ids[order]),
                    record_ids=_readonly(self.selection.record_ids[order]),
                )
                if (
                    _hash_id_set_sha256(reordered.hash_ids)
                    != self.selection.hash_id_set_sha256
                ):
                    raise RuntimeError("cache-index ordering changed split membership")
                self.selection = reordered
            if self._cache.is_corruption and self.view is None:
                raise ValueError("PN2021-C RuntimeECGDataset requires an explicit view")
            if not self._cache.is_corruption and self.view is not None:
                raise ValueError("clean RuntimeECGDataset does not accept a view")
            if self._cache.is_corruption:
                first_record = self._cache.get_record(
                    int(self.selection.indices[0]),
                    view=self.view,
                    copy=False,
                    check_finite=False,
                )
                self._resolved_view_index = first_record.view_index
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self.selection)

    def __getstate__(self) -> dict[str, Any]:
        if not self._owns_cache:
            raise RuntimeError(
                "session-owned datasets require num_workers=0 and cannot be pickled"
            )
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_cache_pid"] = None
        state["_prefetcher"] = None
        state["_prefetcher_pid"] = None
        return state

    def _get_cache(self) -> ECGCache:
        current_pid = os.getpid()
        if self._cache is None or self._cache_pid != current_pid:
            if not self._owns_cache:
                raise RuntimeError(
                    "session-owned cache access is restricted to its creating process"
                )
            if self._cache is not None:
                self._cache.close()
            self._cache = load_cache(
                self.cache_dir,
                sampling_rate_hz=self.sampling_rate_hz,
                mode=self.cache_mode,
                validate_values="none",
            )
            self._cache_pid = current_pid
        return self._cache

    def _get_prefetcher(self, cache: ECGCache) -> MMapPrefetcher | None:
        current_pid = os.getpid()
        if self._prefetcher is not None and self._prefetcher_pid != current_pid:
            self._prefetcher.close()
            self._prefetcher = None
            self._prefetcher_pid = None
        if (
            self._prefetcher is None
            and self.mmap_prefetch.enabled
            and self.access_order == "cache_index"
            and self.batch_read
            and cache.storage_mode == "mmap"
        ):
            self._prefetcher = MMapPrefetcher(
                cache,
                view_index=self._resolved_view_index,
                config=self.mmap_prefetch,
            )
            self._prefetcher_pid = current_pid
        return self._prefetcher

    def __getitem__(self, position: int) -> dict[str, Any]:
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise TypeError("dataset position must be an integer")
        position = int(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(f"dataset position {position} is outside [0, {len(self)})")
        cache = self._get_cache()
        cache_index = int(self.selection.indices[position])
        record = cache.get_record(
            cache_index,
            view=self.view,
            layout="time_channel",
            copy=True,
            check_finite=True,
        )
        expected_hash = str(self.selection.hash_ids[position])
        expected_record = str(self.selection.record_ids[position])
        if record.hash_id != expected_hash or record.record_id != expected_record:
            raise RuntimeError("cache identity changed after dataset initialization")
        waveform = (
            torch.as_tensor(record.signal, dtype=torch.float32)
            if self.transform is None
            else self.transform(record.signal)
        )
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("runtime transform must return a torch.Tensor")
        center = str(record.metadata.get("center", ""))
        return {
            "waveform": waveform,
            "label": torch.as_tensor(record.label, dtype=torch.float32),
            "selection_index": torch.tensor(position, dtype=torch.int64),
            "cache_index": torch.tensor(cache_index, dtype=torch.int64),
            "record_id": record.record_id,
            "hash_id": record.hash_id,
            "dataset": cache.dataset,
            "logical_center": self.selection.logical_center or "",
            "source_center": center,
            "sampling_rate_hz": torch.tensor(
                cache.sampling_rate_hz, dtype=torch.int64
            ),
            "view_index": torch.tensor(
                -1 if record.view_index is None else record.view_index,
                dtype=torch.int64,
            ),
            "composition_id": record.composition_id or "",
        }

    def __getitems__(self, positions: Sequence[int]) -> list[dict[str, Any]]:
        """Fetch one DataLoader batch with a single validated mmap operation.

        PyTorch calls this optional map-style Dataset hook when automatic
        batching is enabled.  Runtime augmentation keeps the scalar path so its
        random-call semantics do not change; clean model transforms are safe to
        apply over the leading batch dimension and are therefore vectorized.
        """

        normalized_positions: list[int] = []
        for raw_position in positions:
            if isinstance(raw_position, bool) or not isinstance(
                raw_position, (int, np.integer)
            ):
                raise TypeError("dataset positions must be integers")
            position = int(raw_position)
            if position < 0:
                position += len(self)
            if not 0 <= position < len(self):
                raise IndexError(
                    f"dataset position {position} is outside [0, {len(self)})"
                )
            normalized_positions.append(position)
        if not normalized_positions:
            return []
        if not self.batch_read:
            return [self[position] for position in normalized_positions]
        if self.transform is not None and not isinstance(
            self.transform, ECGModelTransform
        ):
            return [self[position] for position in normalized_positions]
        if isinstance(self.transform, ECGModelTransform) and (
            self.transform.augmentation is not None
        ):
            return [self[position] for position in normalized_positions]

        cache = self._get_cache()
        selection_positions = np.asarray(normalized_positions, dtype=np.int64)
        cache_indices = np.asarray(
            self.selection.indices[selection_positions], dtype=np.int64
        )
        strictly_increasing = bool(
            cache_indices.size == 1 or np.all(np.diff(cache_indices) > 0)
        )
        dense_start = int(cache_indices[0])
        dense_stop = int(cache_indices[-1]) + 1
        dense_span = dense_stop - dense_start
        use_dense_read = strictly_increasing and dense_span <= (
            len(cache_indices) + max(16, len(cache_indices) // 4)
        )
        if use_dense_read:
            dense_batch = cache.get_batch(
                slice(dense_start, dense_stop),
                view=self.view,
                layout="time_channel",
                copy=True,
                check_finite=True,
            )
            offsets = cache_indices - dense_start
            signals = dense_batch.signals[offsets]
            labels_array = dense_batch.labels[offsets]
            record_ids = dense_batch.record_ids[offsets]
            hash_ids = dense_batch.hash_ids[offsets]
            resolved_indices = dense_batch.indices[offsets]
            view_index = dense_batch.view_index
            composition_id = dense_batch.composition_id
        else:
            sparse_batch = cache.get_batch(
                cache_indices,
                view=self.view,
                layout="time_channel",
                copy=True,
                check_finite=True,
            )
            signals = sparse_batch.signals
            labels_array = sparse_batch.labels
            record_ids = sparse_batch.record_ids
            hash_ids = sparse_batch.hash_ids
            resolved_indices = sparse_batch.indices
            view_index = sparse_batch.view_index
            composition_id = sparse_batch.composition_id
        expected_hashes = self.selection.hash_ids[selection_positions].astype(str)
        expected_records = self.selection.record_ids[selection_positions].astype(str)
        if not np.array_equal(hash_ids, expected_hashes) or not np.array_equal(
            record_ids, expected_records
        ):
            raise RuntimeError("cache identity changed after dataset initialization")
        prefetcher = self._get_prefetcher(cache)
        if prefetcher is not None:
            prefetcher.prefetch_after(cache_indices)
        waveforms = (
            torch.as_tensor(signals, dtype=torch.float32)
            if self.transform is None
            else self.transform(signals)
        )
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("runtime transform must return a torch.Tensor")
        if waveforms.ndim < 3 or int(waveforms.shape[0]) != len(
            normalized_positions
        ):
            raise ValueError("batched runtime transform changed the leading batch axis")
        labels = torch.as_tensor(labels_array, dtype=torch.float32)
        centers = (
            cache.records.iloc[resolved_indices]["center"].astype(str).to_numpy()
            if "center" in cache.records.columns
            else np.full(len(normalized_positions), "", dtype=str)
        )
        return [
            {
                "waveform": waveforms[offset],
                "label": labels[offset],
                "selection_index": torch.tensor(position, dtype=torch.int64),
                "cache_index": torch.tensor(
                    int(resolved_indices[offset]), dtype=torch.int64
                ),
                "record_id": str(record_ids[offset]),
                "hash_id": str(hash_ids[offset]),
                "dataset": cache.dataset,
                "logical_center": self.selection.logical_center or "",
                "source_center": str(centers[offset]),
                "sampling_rate_hz": torch.tensor(
                    cache.sampling_rate_hz, dtype=torch.int64
                ),
                "view_index": torch.tensor(
                    -1 if view_index is None else view_index,
                    dtype=torch.int64,
                ),
                "composition_id": composition_id or "",
            }
            for offset, position in enumerate(normalized_positions)
        ]

    def describe(self) -> dict[str, Any]:
        cache = self._get_cache()
        return {
            "cache": cache.describe(),
            "selection": self.selection.describe(),
            "view": self.view,
            "mmap": {
                "access_order": self.access_order,
                "batch_read": self.batch_read,
                "prefetch": self.mmap_prefetch.describe(),
            },
            "transform": None
            if self.transform is None
            else self.transform.__class__.__name__,
        }

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
            self._prefetcher_pid = None
        if self._cache is not None and self._owns_cache:
            self._cache.close()
        self._cache = None
        self._cache_pid = None

    def __enter__(self) -> RuntimeECGDataset:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SelectionResidentECGDataset(Dataset[dict[str, Any]]):
    """Contiguous CPU copy of one immutable PN2021 selection.

    Only records named by the already-validated split are gathered. Global
    cache indices, stable hashes, record IDs and physical source centers remain
    unchanged, so replacing the mmap-backed dataset cannot alter sampler
    positions or random-number streams. The source cache is not retained by
    this object and can be closed immediately after construction.
    """

    def __init__(
        self,
        source: RuntimeECGDataset,
        *,
        pin_memory: bool = False,
    ) -> None:
        if not isinstance(source, RuntimeECGDataset):
            raise TypeError("source must be a RuntimeECGDataset")
        cache = source._get_cache()
        clean_k500 = (
            cache.dataset == "pn2021"
            and not cache.is_corruption
            and source.selection.partition in PN2021_K500_PARTITIONS
            and source.view is None
        )
        frozen_corruption_validation = (
            cache.dataset == "pn2021c"
            and cache.is_corruption
            and source.selection.partition == "k500_tune_validation"
            and source.view is not None
        )
        if not (clean_k500 or frozen_corruption_validation):
            raise ValueError(
                "selection residency supports clean PN2021 K500 splits or one "
                "explicit PN2021-C k500_tune_validation view"
            )
        if source.transform is not None:
            raise ValueError(
                "selection residency stores canonical raw data and rejects transforms"
            )
        if cache.storage_mode != "mmap":
            raise ValueError("selection residency requires a mmap source cache")
        if not isinstance(pin_memory, bool):
            raise ValueError("selection residency pin_memory must be boolean")

        cache_indices = np.asarray(source.selection.indices, dtype=np.int64)
        batch = cache.get_batch(
            cache_indices,
            view=source.view,
            layout="time_channel",
            copy=True,
            check_finite=True,
        )
        expected_hashes = source.selection.hash_ids.astype(str)
        expected_records = source.selection.record_ids.astype(str)
        if not np.array_equal(batch.indices, cache_indices):
            raise RuntimeError("resident gather changed global cache indices")
        if not np.array_equal(batch.hash_ids, expected_hashes):
            raise RuntimeError("resident gather changed K500 hash order")
        if not np.array_equal(batch.record_ids, expected_records):
            raise RuntimeError("resident gather changed K500 record order")

        waveform_array = np.array(
            batch.signals,
            dtype=np.float32,
            order="C",
            copy=False,
        )
        label_array = np.array(
            batch.labels,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        waveforms = torch.from_numpy(waveform_array)
        labels = torch.from_numpy(label_array)
        if pin_memory:
            waveforms = waveforms.pin_memory()
            labels = labels.pin_memory()
        if waveforms.device.type != "cpu" or not waveforms.is_contiguous():
            raise RuntimeError("resident K500 waveforms must be contiguous CPU data")
        if labels.device.type != "cpu" or not labels.is_contiguous():
            raise RuntimeError("resident K500 labels must be contiguous CPU data")

        self.selection = source.selection
        self.sampling_rate_hz = source.sampling_rate_hz
        self.transform = source.transform
        self.cache_mode = "selection_ram"
        self.view = source.view
        self._dataset_name = cache.dataset
        self._view_index = None if batch.view_index is None else int(batch.view_index)
        self._composition_id = str(batch.composition_id or "")
        self._waveforms = waveforms
        self._labels = labels
        self._cache_indices = cache_indices.copy()
        self._record_ids = expected_records.copy()
        self._hash_ids = expected_hashes.copy()
        self._source_centers = (
            cache.records.iloc[cache_indices]["center"].astype(str).to_numpy(copy=True)
            if "center" in cache.records.columns
            else np.full(len(cache_indices), "", dtype=str)
        )
        self._cache_description = cache.describe()
        self._access_order = source.access_order
        self._batch_read = source.batch_read
        self._pin_memory = pin_memory
        self._closed = False

    def __len__(self) -> int:
        return len(self.selection)

    @property
    def waveforms(self) -> torch.Tensor:
        """The immutable selection-sized contiguous CPU tensor."""

        self._require_open()
        return self._waveforms

    @property
    def labels(self) -> torch.Tensor:
        self._require_open()
        return self._labels

    @property
    def is_pinned(self) -> bool:
        return bool(self._pin_memory)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("selection-resident dataset is closed")

    def _normalize_position(self, position: int) -> int:
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise TypeError("dataset position must be an integer")
        resolved = int(position)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError(f"dataset position {position} is outside [0, {len(self)})")
        return resolved

    def _item(self, position: int) -> dict[str, Any]:
        self._require_open()
        waveform = self._waveforms[position]
        if self.transform is not None:
            waveform = self.transform(waveform)
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("runtime transform must return a torch.Tensor")
        return {
            "waveform": waveform,
            "label": self._labels[position],
            "selection_index": torch.tensor(position, dtype=torch.int64),
            "cache_index": torch.tensor(
                int(self._cache_indices[position]), dtype=torch.int64
            ),
            "record_id": str(self._record_ids[position]),
            "hash_id": str(self._hash_ids[position]),
            "dataset": self._dataset_name,
            "logical_center": self.selection.logical_center or "",
            "source_center": str(self._source_centers[position]),
            "sampling_rate_hz": torch.tensor(
                self.sampling_rate_hz, dtype=torch.int64
            ),
            "view_index": torch.tensor(
                -1 if self._view_index is None else self._view_index,
                dtype=torch.int64,
            ),
            "composition_id": self._composition_id,
        }

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self._item(self._normalize_position(position))

    def __getitems__(self, positions: Sequence[int]) -> list[dict[str, Any]]:
        # Keeping the scalar item contract delegates batching/shuffle entirely
        # to the unchanged PyTorch sampler and collate path.
        return [self._item(self._normalize_position(value)) for value in positions]

    def describe(self) -> dict[str, Any]:
        return {
            "cache": dict(self._cache_description),
            "selection": self.selection.describe(),
            "view": self.view,
            "composition_id": self._composition_id,
            "mmap": {
                "access_order": self._access_order,
                "batch_read": self._batch_read,
                "prefetch": MMapPrefetchConfig(enabled=False).describe(),
            },
            "residency": {
                "mode": "selection_ram",
                "record_count": len(self),
                "waveform_shape": list(self._waveforms.shape),
                "waveform_dtype": str(self._waveforms.dtype),
                "contiguous": self._waveforms.is_contiguous(),
                "pinned": self._pin_memory,
                "source_cache_closed_after_gather": True,
            },
            "transform": None
            if self.transform is None
            else self.transform.__class__.__name__,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._waveforms = torch.empty((0, 0, 0), dtype=torch.float32)
        self._labels = torch.empty((0, 0), dtype=torch.float32)
        self._closed = True


@dataclass(frozen=True)
class DataLoaderSeedIdentity:
    base_seed: int
    effective_seed: int
    namespace: str
    config_path: Path
    config_sha256: str

    def describe(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "effective_seed": self.effective_seed,
            "namespace": self.namespace,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "derivation": "sha256_first_uint32_little_endian",
        }


@dataclass(frozen=True)
class DataContentLedgerConfig:
    path: Path
    sha256: str

    def describe(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class DataLoadConfig:
    """Strictly validated defaults loaded from ``configs/data/data_load.yaml``."""

    path: Path
    sha256: str
    content_ledger: DataContentLedgerConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    drop_last: bool
    cache_mode: StorageMode
    validate_values: ValueValidation
    shuffle_partitions: tuple[str, ...]
    mmap_access_order: MMapAccessOrder
    mmap_batch_read: bool
    mmap_prefetch: MMapPrefetchConfig
    selection_resident: bool
    selection_resident_pin_memory: bool
    prepare_for_model: bool
    sanitize: bool
    global_zscore: bool
    output_layout: RuntimeLayout
    epsilon: float

    def describe(self) -> dict[str, Any]:
        return {
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "content_ledger": self.content_ledger.describe(),
            "defaults": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
                "drop_last": self.drop_last,
                "cache_mode": self.cache_mode,
                "validate_values": self.validate_values,
                "shuffle_partitions": list(self.shuffle_partitions),
                "mmap": {
                    "access_order": self.mmap_access_order,
                    "batch_read": self.mmap_batch_read,
                    "prefetch": self.mmap_prefetch.describe(),
                },
                "selection_residency": {
                    "enabled": self.selection_resident,
                    "pin_memory": self.selection_resident_pin_memory,
                },
                "prepare_for_model": self.prepare_for_model,
                "sanitize": self.sanitize,
                "global_zscore": self.global_zscore,
                "output_layout": self.output_layout,
                "epsilon": self.epsilon,
            },
        }


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_bool(value: Any, *, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be boolean")
    return value


def _require_positive_int(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


def _require_nonnegative_int(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return int(value)


def load_data_load_config(
    path: str | Path = DEFAULT_DATA_LOAD_CONFIG,
) -> DataLoadConfig:
    """Load and strictly validate the shared DataLoader/runtime defaults."""

    config_path = resolve_entry_config_path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"data-load config not found: {config_path}") from None
    if not isinstance(payload, dict):
        raise ValueError("data-load config must be a YAML mapping")
    required_root_keys = {
        "schema_version",
        "content_ledger",
        "dataloader",
        "model_input",
        "mmap",
    }
    optional_root_keys = {"selection_residency"}
    actual_root_keys = set(payload)
    missing_root_keys = required_root_keys - actual_root_keys
    unexpected_root_keys = actual_root_keys - required_root_keys - optional_root_keys
    if missing_root_keys or unexpected_root_keys:
        raise ValueError(
            "data-load config keys mismatch: "
            f"missing={sorted(missing_root_keys)}, "
            f"unexpected={sorted(unexpected_root_keys)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("data-load config schema_version must be 1")
    dataloader = _require_mapping(
        payload["dataloader"], description="data-load config.dataloader"
    )
    content_ledger = _require_mapping(
        payload["content_ledger"], description="data-load config.content_ledger"
    )
    _require_exact_keys(
        content_ledger,
        expected={"path", "sha256"},
        description="data-load config.content_ledger",
    )
    ledger_path = resolve_config_reference(
        content_ledger["path"],
        owner_config_path=config_path,
        description="data-load config.content_ledger.path",
    )
    ledger_sha256 = content_ledger["sha256"]
    if (
        not isinstance(ledger_sha256, str)
        or len(ledger_sha256) != 64
        or set(ledger_sha256).difference("0123456789abcdef")
    ):
        raise ValueError("data-load config.content_ledger.sha256 is invalid")
    model_input = _require_mapping(
        payload["model_input"], description="data-load config.model_input"
    )
    mmap_config = _require_mapping(
        payload["mmap"], description="data-load config.mmap"
    )
    selection_residency = _require_mapping(
        payload.get(
            "selection_residency",
            {"enabled": False, "pin_memory": False},
        ),
        description="data-load config.selection_residency",
    )
    _require_exact_keys(
        dataloader,
        expected={
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "drop_last",
            "cache_mode",
            "validate_values",
            "shuffle_partitions",
        },
        description="data-load config.dataloader",
    )
    _require_exact_keys(
        model_input,
        expected={
            "prepare_for_model",
            "sanitize",
            "global_zscore",
            "output_layout",
            "epsilon",
        },
        description="data-load config.model_input",
    )
    _require_exact_keys(
        mmap_config,
        expected={"access_order", "batch_read", "prefetch"},
        description="data-load config.mmap",
    )
    mmap_prefetch = _require_mapping(
        mmap_config["prefetch"], description="data-load config.mmap.prefetch"
    )
    _require_exact_keys(
        mmap_prefetch,
        expected={"enabled", "advice", "window_mib", "ahead_batches"},
        description="data-load config.mmap.prefetch",
    )
    _require_exact_keys(
        selection_residency,
        expected={"enabled", "pin_memory"},
        description="data-load config.selection_residency",
    )
    batch_size = _require_positive_int(
        dataloader["batch_size"], description="dataloader.batch_size"
    )
    num_workers = _require_nonnegative_int(
        dataloader["num_workers"], description="dataloader.num_workers"
    )
    pin_memory = _require_bool(
        dataloader["pin_memory"], description="dataloader.pin_memory"
    )
    persistent_workers = _require_bool(
        dataloader["persistent_workers"],
        description="dataloader.persistent_workers",
    )
    prefetch_factor = _require_positive_int(
        dataloader["prefetch_factor"], description="dataloader.prefetch_factor"
    )
    drop_last = _require_bool(
        dataloader["drop_last"], description="dataloader.drop_last"
    )
    cache_mode = str(dataloader["cache_mode"])
    if cache_mode not in {"auto", "ram", "mmap"}:
        raise ValueError("dataloader.cache_mode must be auto, ram or mmap")
    validate_values = str(dataloader["validate_values"])
    if validate_values not in {"none", "sample", "full"}:
        raise ValueError(
            "dataloader.validate_values must be none, sample or full"
        )
    raw_shuffle = dataloader["shuffle_partitions"]
    if not isinstance(raw_shuffle, list) or any(
        not isinstance(value, str) for value in raw_shuffle
    ):
        raise ValueError("dataloader.shuffle_partitions must be a string list")
    shuffle_partitions = tuple(raw_shuffle)
    if len(set(shuffle_partitions)) != len(shuffle_partitions):
        raise ValueError("dataloader.shuffle_partitions contains duplicates")
    allowed_shuffle = {"train", "k500"}
    unexpected_shuffle = set(shuffle_partitions) - allowed_shuffle
    if unexpected_shuffle:
        raise ValueError(
            "only train and k500 may be shuffled by shared defaults; "
            f"got {sorted(unexpected_shuffle)}"
        )
    mmap_access_order = str(mmap_config["access_order"])
    if mmap_access_order not in {"split", "cache_index"}:
        raise ValueError("mmap.access_order must be split or cache_index")
    mmap_batch_read = _require_bool(
        mmap_config["batch_read"], description="mmap.batch_read"
    )
    mmap_prefetch_enabled = _require_bool(
        mmap_prefetch["enabled"], description="mmap.prefetch.enabled"
    )
    mmap_prefetch_advice = str(mmap_prefetch["advice"])
    if mmap_prefetch_advice not in {"none", "sequential_willneed"}:
        raise ValueError(
            "mmap.prefetch.advice must be none or sequential_willneed"
        )
    mmap_prefetch_config = MMapPrefetchConfig(
        enabled=mmap_prefetch_enabled,
        advice=cast(MMapAdvice, mmap_prefetch_advice),
        window_mib=_require_positive_int(
            mmap_prefetch["window_mib"], description="mmap.prefetch.window_mib"
        ),
        ahead_batches=_require_positive_int(
            mmap_prefetch["ahead_batches"],
            description="mmap.prefetch.ahead_batches",
        ),
    )
    selection_resident = _require_bool(
        selection_residency["enabled"],
        description="selection_residency.enabled",
    )
    selection_resident_pin_memory = _require_bool(
        selection_residency["pin_memory"],
        description="selection_residency.pin_memory",
    )
    if selection_resident_pin_memory and not selection_resident:
        raise ValueError(
            "selection_residency.pin_memory requires selection residency enabled"
        )
    prepare_for_model = _require_bool(
        model_input["prepare_for_model"],
        description="model_input.prepare_for_model",
    )
    sanitize = _require_bool(
        model_input["sanitize"], description="model_input.sanitize"
    )
    global_zscore = _require_bool(
        model_input["global_zscore"], description="model_input.global_zscore"
    )
    output_layout = str(model_input["output_layout"])
    if output_layout not in {"time_channel", "channel_time"}:
        raise ValueError(
            "model_input.output_layout must be time_channel or channel_time"
        )
    epsilon_value = model_input["epsilon"]
    if isinstance(epsilon_value, bool) or not isinstance(
        epsilon_value, (int, float)
    ):
        raise ValueError("model_input.epsilon must be numeric")
    epsilon = float(epsilon_value)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("model_input.epsilon must be finite and positive")
    if num_workers == 0 and persistent_workers:
        raise ValueError(
            "dataloader.persistent_workers must be false when num_workers is 0"
        )
    if num_workers > 0 and cache_mode != "mmap":
        raise ValueError(
            "multi-worker YAML defaults require dataloader.cache_mode=mmap"
        )
    return DataLoadConfig(
        path=config_path,
        sha256=_sha256_file(config_path),
        content_ledger=DataContentLedgerConfig(ledger_path, ledger_sha256),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        cache_mode=cast(StorageMode, cache_mode),
        validate_values=cast(ValueValidation, validate_values),
        shuffle_partitions=shuffle_partitions,
        mmap_access_order=cast(MMapAccessOrder, mmap_access_order),
        mmap_batch_read=mmap_batch_read,
        mmap_prefetch=mmap_prefetch_config,
        selection_resident=selection_resident,
        selection_resident_pin_memory=selection_resident_pin_memory,
        prepare_for_model=prepare_for_model,
        sanitize=sanitize,
        global_zscore=global_zscore,
        output_layout=cast(RuntimeLayout, output_layout),
        epsilon=epsilon,
    )


class RuntimeDataLoader(DataLoader[dict[str, Any]]):
    """DataLoader carrying its reproducibility identity and close helper."""

    def __init__(
        self,
        *args: Any,
        runtime_seed_identity: DataLoaderSeedIdentity,
        **kwargs: Any,
    ) -> None:
        self.runtime_seed_identity = runtime_seed_identity
        self.runtime_config_identity: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def describe(self) -> dict[str, Any]:
        dataset = self.dataset
        describe = getattr(dataset, "describe", None)
        return {
            "dataset": describe() if callable(describe) else dataset.__class__.__name__,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
            "seed": self.runtime_seed_identity.describe(),
            "runtime_config": self.runtime_config_identity,
        }

    def close(self) -> None:
        # PyTorch keeps a persistent multiprocessing iterator on the loader.
        # Closing only the mmap dataset leaves those workers alive when a
        # caller creates one loader per evaluation view. Shut them down
        # explicitly so sequential clean/corruption evaluation cannot leak
        # four processes (and their inherited mappings) per view.
        iterator = getattr(self, "_iterator", None)
        try:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        finally:
            self._iterator = None
            close = getattr(self.dataset, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> RuntimeDataLoader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def build_dataloader(
    dataset: RuntimeECGDataset | SelectionResidentECGDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    seed_namespace: str = "data_runtime",
    seed_config_path: str | Path = DEFAULT_RANDOM_SEED_CONFIG_PATH,
) -> tuple[RuntimeDataLoader, DataLoaderSeedIdentity]:
    """Build a deterministic DataLoader and return its recorded seed identity."""

    if not isinstance(dataset, (RuntimeECGDataset, SelectionResidentECGDataset)):
        raise TypeError(
            "dataset must be a RuntimeECGDataset or SelectionResidentECGDataset"
        )
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    num_workers = int(num_workers)
    if num_workers > 0 and dataset.cache_mode != "mmap":
        raise ValueError(
            "multi-worker loading requires cache_mode='mmap' to avoid one full "
            "RAM copy per worker"
        )
    if not str(seed_namespace):
        raise ValueError("seed_namespace must be non-empty")
    seed_path = Path(seed_config_path).expanduser().resolve()
    base_seed = load_random_seed_config(seed_path).base_seed
    effective_seed = derive_seed(
        "data_runtime",
        str(seed_namespace),
        base_seed=base_seed,
        config_path=seed_path,
    )
    seed_identity = DataLoaderSeedIdentity(
        base_seed=base_seed,
        effective_seed=effective_seed,
        namespace=str(seed_namespace),
        config_path=seed_path,
        config_sha256=_sha256_file(seed_path),
    )
    generator = torch.Generator(device="cpu").manual_seed(effective_seed)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "drop_last": bool(drop_last),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    if num_workers > 0:
        if int(prefetch_factor) <= 0:
            raise ValueError("prefetch_factor must be positive")
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return (
        RuntimeDataLoader(runtime_seed_identity=seed_identity, **kwargs),
        seed_identity,
    )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _normalize_runtime_partition(
    dataset: RuntimeDatasetName,
    partition: str,
) -> str:
    requested = str(partition)
    canonical = CANONICAL_EVALUATION_PARTITIONS.get(requested)
    if canonical is not None:
        required_dataset, resolved = canonical
        if dataset != required_dataset:
            raise ValueError(
                f"canonical view {requested!r} belongs to {required_dataset}, "
                f"not {dataset}"
            )
        return resolved
    if dataset == "ptbxl":
        if requested not in PTBXL_PARTITIONS:
            raise ValueError(
                f"PTB-XL partition must be one of {PTBXL_PARTITIONS}"
            )
        return requested
    if requested not in PN2021_PARTITIONS:
        canonical_names = tuple(
            name
            for name, (owner, _) in CANONICAL_EVALUATION_PARTITIONS.items()
            if owner == dataset
        )
        raise ValueError(
            f"{dataset} partition must be one of {PN2021_PARTITIONS} or "
            f"canonical views {canonical_names}"
        )
    return requested


def _read_corruption_cache_dir(config_path: str | Path) -> Path:
    path = _resolve_project_path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"corruption cache config not found: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError("corruption cache config must be a YAML mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("corruption cache config schema_version must be 1")
    if str(payload.get("dataset", "")) != "pn2021c":
        raise ValueError("corruption cache config dataset must be pn2021c")
    output = _require_mapping(
        payload.get("output"), description="corruption cache config.output"
    )
    cache_dir = output.get("cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("corruption cache config.output.cache_dir is required")
    return _resolve_project_path(cache_dir)


def _resolve_runtime_locations(
    *,
    dataset: RuntimeDatasetName,
    cache_dir: str | Path | None,
    split_dir: str | Path | None,
    split_config_path: str | Path,
    corruption_cache_config_path: str | Path,
) -> tuple[Path, Path]:
    split_config: dict[str, Any] | None = None
    if split_dir is None or (cache_dir is None and dataset != "pn2021c"):
        split_config = load_split_config(split_config_path)
    if split_dir is None:
        assert split_config is not None
        split_root = _resolve_project_path(split_config["output"]["root_dir"])
        split_key = "ptbxl" if dataset == "ptbxl" else "pn2021"
        resolved_split_dir = split_root / str(
            split_config[split_key]["output_subdir"]
        )
    else:
        resolved_split_dir = _resolve_project_path(split_dir)

    if cache_dir is not None:
        resolved_cache_dir = _resolve_project_path(cache_dir)
    elif dataset == "pn2021c":
        resolved_cache_dir = _read_corruption_cache_dir(
            corruption_cache_config_path
        )
    else:
        assert split_config is not None
        resolved_cache_dir = _resolve_project_path(
            split_config[dataset]["cache_dir"]
        )
    return resolved_cache_dir, resolved_split_dir.resolve()


def _resolve_runtime_seed_config(
    seed_config_path: str | Path | None,
    *,
    split_config_path: str | Path,
) -> Path:
    if seed_config_path is not None:
        return resolve_entry_config_path(seed_config_path)
    owner = resolve_entry_config_path(split_config_path)
    split_config = load_split_config(owner)
    return resolve_config_reference(
        split_config["random_seed"]["file"],
        owner_config_path=owner,
        description="random_seed.file",
        must_exist=True,
    )


class SequentialEvaluationDataSession:
    """Reuse validated mmap caches and split selections across eval views.

    ``get_dataloader`` has the same keyword interface as the module-level
    function and is intended to be passed directly as an evaluation loader
    factory. Loaders are strictly single-process and sequential. Closing one
    loader releases only its lightweight dataset wrapper; the session owns and
    closes the shared mmap handles on context exit.
    """

    def __init__(self) -> None:
        self._caches: dict[tuple[Path, int], ECGCache] = {}
        self._validation_levels: dict[tuple[Path, int], ValueValidation] = {}
        self._selections: dict[
            tuple[Path, int, Path, str, str | None], ECGSelection
        ] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("sequential evaluation data session is closed")

    def _acquire(
        self,
        *,
        cache_dir: Path,
        split_dir: Path,
        partition: str,
        logical_center: str | None,
        sampling_rate_hz: int,
        validate_values: ValueValidation,
    ) -> tuple[ECGCache, ECGSelection]:
        self._require_open()
        cache_key = (cache_dir, int(sampling_rate_hz))
        cache = self._caches.get(cache_key)
        if cache is None:
            cache = load_cache(
                cache_dir,
                sampling_rate_hz=sampling_rate_hz,
                mode="mmap",
                validate_values=validate_values,
            )
            self._caches[cache_key] = cache
            self._validation_levels[cache_key] = validate_values
        else:
            rank = {"none": 0, "sample": 1, "full": 2}
            completed = self._validation_levels[cache_key]
            if rank[validate_values] > rank[completed]:
                cache.validate_waveforms(full=validate_values == "full")
                self._validation_levels[cache_key] = validate_values

        selection_key = (
            cache_dir,
            int(sampling_rate_hz),
            split_dir,
            str(partition),
            logical_center,
        )
        selection = self._selections.get(selection_key)
        if selection is None:
            selection = load_selection(
                cache,
                split_dir,
                partition=partition,
                logical_center=logical_center,
            )
            self._selections[selection_key] = selection
        return cache, selection

    def get_dataloader(self, **kwargs: Any) -> RuntimeDataLoader:
        """Build one sequential loader while retaining shared session state."""

        self._require_open()
        if "evaluation_session" in kwargs:
            raise ValueError("evaluation_session is owned by this factory")
        return get_dataloader(evaluation_session=self, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            "closed": self._closed,
            "cache_open_count": len(self._caches),
            "selection_count": len(self._selections),
            "caches": [
                {
                    **cache.describe(),
                    "validated_values": self._validation_levels[key],
                }
                for key, cache in self._caches.items()
            ],
        }

    def close(self) -> None:
        if self._closed:
            return
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()
        self._validation_levels.clear()
        self._selections.clear()
        self._closed = True

    def __enter__(self) -> SequentialEvaluationDataSession:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def get_dataloader(
    *,
    dataset: RuntimeDatasetName,
    partition: str,
    batch_size: int | None = None,
    logical_center: str | None = None,
    sampling_rate_hz: int = 100,
    view: ViewSelector = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    drop_last: bool | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    cache_mode: StorageMode | None = None,
    validate_values: ValueValidation | None = None,
    selection_resident: bool | None = None,
    selection_resident_pin_memory: bool | None = None,
    mmap_access_order: MMapAccessOrder | None = None,
    mmap_batch_read: bool | None = None,
    mmap_prefetch_enabled: bool | None = None,
    mmap_prefetch_advice: MMapAdvice | None = None,
    mmap_prefetch_window_mib: int | None = None,
    mmap_prefetch_ahead_batches: int | None = None,
    prepare_for_model: bool | None = None,
    augmentation: Callable[[torch.Tensor], np.ndarray | torch.Tensor] | None = None,
    sanitize: bool | None = None,
    global_zscore: bool | None = None,
    output_layout: RuntimeLayout | None = None,
    epsilon: float | None = None,
    seed_namespace: str | None = None,
    seed_config_path: str | Path | None = None,
    config_root: str | Path | None = None,
    split_config_path: str | Path | None = None,
    data_load_config_path: str | Path | None = None,
    corruption_cache_config_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    evaluation_session: SequentialEvaluationDataSession | None = None,
) -> RuntimeDataLoader:
    """Return one ready-to-iterate loader for any managed ECG split.

    Supported requests are:

    - ``dataset="ptbxl"`` with ``train``, ``validation`` or ``test``;
    - ``dataset="pn2021"`` with the full ``k500``, either managed K500 tuning
      partition, or either ref-excluded evaluation;
    - ``dataset="pn2021c"`` with frozen ``k500_tune_validation`` or corrupted
      ref-excluded evaluation and an explicit ``view``.

    Omitted runtime arguments are loaded from ``config_root`` (the repository
    ``configs/`` by default); explicit config paths take precedence. Canonical
    evaluation aliases from the project protocol are accepted. Model-ready
    batches use augmentation -> sanitization -> per-sample global z-score. Set
    ``prepare_for_model=False`` only when the caller explicitly needs untouched
    raw-mV time-channel data.
    """

    selected_config_root = (
        PROJECT_ROOT / "configs"
        if config_root is None
        else Path(config_root).expanduser().resolve()
    )
    resolved_split_config_path = (
        selected_config_root / "data" / "splits.yaml"
        if split_config_path is None
        else resolve_entry_config_path(split_config_path)
    )
    resolved_data_load_config_path = (
        selected_config_root / "data" / "data_load.yaml"
        if data_load_config_path is None
        else resolve_entry_config_path(data_load_config_path)
    )
    resolved_corruption_cache_config_path = (
        selected_config_root / "augmentation" / "cache.yaml"
        if corruption_cache_config_path is None
        else resolve_entry_config_path(corruption_cache_config_path)
    )
    if dataset not in {"ptbxl", "pn2021", "pn2021c"}:
        raise ValueError("dataset must be 'ptbxl', 'pn2021' or 'pn2021c'")
    resolved_partition = _normalize_runtime_partition(dataset, partition)
    data_load_config = load_data_load_config(resolved_data_load_config_path)
    resolved_batch_size = _require_positive_int(
        data_load_config.batch_size if batch_size is None else batch_size,
        description="resolved batch_size",
    )
    resolved_num_workers = _require_nonnegative_int(
        data_load_config.num_workers if num_workers is None else num_workers,
        description="resolved num_workers",
    )
    resolved_pin_memory = _require_bool(
        data_load_config.pin_memory if pin_memory is None else pin_memory,
        description="resolved pin_memory",
    )
    resolved_drop_last = _require_bool(
        data_load_config.drop_last if drop_last is None else drop_last,
        description="resolved drop_last",
    )
    if persistent_workers is None:
        resolved_persistent_workers = (
            data_load_config.persistent_workers
            if resolved_num_workers > 0
            else False
        )
    else:
        resolved_persistent_workers = _require_bool(
            persistent_workers, description="resolved persistent_workers"
        )
    resolved_prefetch_factor = _require_positive_int(
        data_load_config.prefetch_factor
        if prefetch_factor is None
        else prefetch_factor,
        description="resolved prefetch_factor",
    )
    resolved_cache_mode = (
        data_load_config.cache_mode if cache_mode is None else cache_mode
    )
    if resolved_cache_mode not in {"auto", "ram", "mmap"}:
        raise ValueError("resolved cache_mode must be auto, ram or mmap")
    resolved_validate_values = (
        data_load_config.validate_values
        if validate_values is None
        else validate_values
    )
    if resolved_validate_values not in {"none", "sample", "full"}:
        raise ValueError(
            "resolved validate_values must be none, sample or full"
        )
    resolved_selection_resident = _require_bool(
        data_load_config.selection_resident
        if selection_resident is None
        else selection_resident,
        description="resolved selection_resident",
    )
    resolved_selection_resident_pin_memory = _require_bool(
        data_load_config.selection_resident_pin_memory
        if selection_resident_pin_memory is None
        else selection_resident_pin_memory,
        description="resolved selection_resident_pin_memory",
    )
    if resolved_selection_resident_pin_memory and not resolved_selection_resident:
        raise ValueError(
            "selection_resident_pin_memory requires selection_resident=true"
        )
    if shuffle is None:
        resolved_shuffle = (
            resolved_partition in data_load_config.shuffle_partitions
            or (
                resolved_partition == "k500_tune_train"
                and "k500" in data_load_config.shuffle_partitions
            )
        )
    else:
        resolved_shuffle = _require_bool(shuffle, description="resolved shuffle")
    requested_mmap_access_order = (
        data_load_config.mmap_access_order
        if mmap_access_order is None
        else mmap_access_order
    )
    if requested_mmap_access_order not in {"split", "cache_index"}:
        raise ValueError("resolved mmap_access_order must be split or cache_index")
    resolved_mmap_batch_read = _require_bool(
        data_load_config.mmap_batch_read
        if mmap_batch_read is None
        else mmap_batch_read,
        description="resolved mmap_batch_read",
    )
    requested_prefetch_enabled = _require_bool(
        data_load_config.mmap_prefetch.enabled
        if mmap_prefetch_enabled is None
        else mmap_prefetch_enabled,
        description="resolved mmap_prefetch_enabled",
    )
    resolved_prefetch_advice = (
        data_load_config.mmap_prefetch.advice
        if mmap_prefetch_advice is None
        else mmap_prefetch_advice
    )
    if resolved_prefetch_advice not in {"none", "sequential_willneed"}:
        raise ValueError(
            "resolved mmap_prefetch_advice must be none or sequential_willneed"
        )
    resolved_prefetch_window_mib = _require_positive_int(
        data_load_config.mmap_prefetch.window_mib
        if mmap_prefetch_window_mib is None
        else mmap_prefetch_window_mib,
        description="resolved mmap_prefetch_window_mib",
    )
    resolved_prefetch_ahead_batches = _require_positive_int(
        data_load_config.mmap_prefetch.ahead_batches
        if mmap_prefetch_ahead_batches is None
        else mmap_prefetch_ahead_batches,
        description="resolved mmap_prefetch_ahead_batches",
    )
    # Sorting a shuffled split would change its seeded training semantics. Keep
    # training/K500 shuffle in persisted split order and reserve page-cache
    # look-ahead for deterministic evaluation traversal.
    resolved_mmap_access_order: MMapAccessOrder = (
        "split" if resolved_shuffle else requested_mmap_access_order
    )
    resolved_mmap_prefetch = MMapPrefetchConfig(
        enabled=(
            requested_prefetch_enabled
            and not resolved_shuffle
            and resolved_cache_mode == "mmap"
        ),
        advice=cast(MMapAdvice, resolved_prefetch_advice),
        window_mib=resolved_prefetch_window_mib,
        ahead_batches=resolved_prefetch_ahead_batches,
    )
    resolved_prepare_for_model = _require_bool(
        data_load_config.prepare_for_model
        if prepare_for_model is None
        else prepare_for_model,
        description="resolved prepare_for_model",
    )
    resolved_sanitize = _require_bool(
        data_load_config.sanitize if sanitize is None else sanitize,
        description="resolved sanitize",
    )
    resolved_global_zscore = _require_bool(
        data_load_config.global_zscore
        if global_zscore is None
        else global_zscore,
        description="resolved global_zscore",
    )
    resolved_output_layout = (
        data_load_config.output_layout
        if output_layout is None
        else output_layout
    )
    if resolved_output_layout not in {"time_channel", "channel_time"}:
        raise ValueError(
            "resolved output_layout must be time_channel or channel_time"
        )
    raw_epsilon = data_load_config.epsilon if epsilon is None else epsilon
    if isinstance(raw_epsilon, bool) or not isinstance(
        raw_epsilon, (int, float)
    ):
        raise ValueError("resolved epsilon must be numeric")
    resolved_epsilon = float(raw_epsilon)
    if not np.isfinite(resolved_epsilon) or resolved_epsilon <= 0:
        raise ValueError("resolved epsilon must be finite and positive")
    if dataset == "ptbxl":
        if logical_center is not None:
            raise ValueError("PTB-XL loader does not accept logical_center")
        if view is not None:
            raise ValueError("PTB-XL loader does not accept corruption view")
    else:
        if not logical_center:
            raise ValueError(f"{dataset} loader requires logical_center")
        if dataset == "pn2021" and view is not None:
            raise ValueError("clean PN2021 loader does not accept corruption view")
        if dataset == "pn2021c" and view is None:
            raise ValueError("PN2021-C loader requires an explicit corruption view")
    if not resolved_prepare_for_model and augmentation is not None:
        raise ValueError("augmentation requires prepare_for_model=True")
    if resolved_num_workers > 0 and resolved_cache_mode != "mmap":
        raise ValueError(
            "num_workers > 0 requires cache_mode='mmap' to avoid one full "
            "cache copy per worker"
        )
    if resolved_selection_resident:
        clean_k500_residency = (
            dataset == "pn2021"
            and resolved_partition in PN2021_K500_PARTITIONS
            and view is None
        )
        frozen_corruption_validation_residency = (
            dataset == "pn2021c"
            and resolved_partition == "k500_tune_validation"
            and view is not None
        )
        if not (
            clean_k500_residency or frozen_corruption_validation_residency
        ):
            raise ValueError(
                "selection residency supports clean PN2021 K500 splits or one "
                "explicit PN2021-C k500_tune_validation view"
            )
        if resolved_num_workers != 0:
            raise ValueError("selection residency requires num_workers=0")
        if resolved_persistent_workers:
            raise ValueError("selection residency requires persistent_workers=false")
        if resolved_cache_mode != "mmap":
            raise ValueError("selection residency requires a mmap source cache")
        if resolved_prepare_for_model:
            raise ValueError(
                "selection residency requires prepare_for_model=false raw K500 data"
            )
    if evaluation_session is not None:
        if not isinstance(evaluation_session, SequentialEvaluationDataSession):
            raise TypeError(
                "evaluation_session must be a SequentialEvaluationDataSession"
            )
        if resolved_shuffle:
            raise ValueError("sequential evaluation sessions require shuffle=false")
        if resolved_num_workers != 0:
            raise ValueError("sequential evaluation sessions require num_workers=0")
        if resolved_persistent_workers:
            raise ValueError(
                "sequential evaluation sessions require persistent_workers=false"
            )
        if resolved_cache_mode != "mmap":
            raise ValueError("sequential evaluation sessions require cache_mode=mmap")
        if resolved_selection_resident:
            raise ValueError(
                "sequential evaluation sessions reject selection residency"
            )

    resolved_cache_dir, resolved_split_dir = _resolve_runtime_locations(
        dataset=dataset,
        cache_dir=cache_dir,
        split_dir=split_dir,
        split_config_path=resolved_split_config_path,
        corruption_cache_config_path=resolved_corruption_cache_config_path,
    )
    transform: ECGModelTransform | None
    if resolved_prepare_for_model:
        transform = ECGModelTransform(
            augmentation=augmentation,
            sanitize=resolved_sanitize,
            global_zscore=resolved_global_zscore,
            epsilon=float(resolved_epsilon),
            source_layout="time_channel",
            output_layout=resolved_output_layout,
        )
    else:
        transform = None
    shared_cache: ECGCache | None = None
    shared_selection: ECGSelection | None = None
    if evaluation_session is not None:
        shared_cache, shared_selection = evaluation_session._acquire(
            cache_dir=resolved_cache_dir,
            split_dir=resolved_split_dir,
            partition=resolved_partition,
            logical_center=logical_center,
            sampling_rate_hz=sampling_rate_hz,
            validate_values=cast(ValueValidation, resolved_validate_values),
        )
    runtime_dataset = RuntimeECGDataset(
        cache_dir=resolved_cache_dir,
        split_dir=resolved_split_dir,
        partition=resolved_partition,
        logical_center=logical_center,
        sampling_rate_hz=sampling_rate_hz,
        cache_mode=resolved_cache_mode,
        view=view,
        transform=transform,
        validate_values=resolved_validate_values,
        access_order=resolved_mmap_access_order,
        batch_read=resolved_mmap_batch_read,
        mmap_prefetch=resolved_mmap_prefetch,
        shared_cache=shared_cache,
        shared_selection=shared_selection,
    )
    runtime_data: RuntimeECGDataset | SelectionResidentECGDataset
    if resolved_selection_resident:
        try:
            runtime_data = SelectionResidentECGDataset(
                runtime_dataset,
                pin_memory=resolved_selection_resident_pin_memory,
            )
        finally:
            runtime_dataset.close()
    else:
        runtime_data = runtime_dataset
    effective_namespace = seed_namespace or ":".join(
        (
            dataset,
            resolved_partition,
            logical_center or "global",
            f"{int(sampling_rate_hz)}hz",
            "clean" if view is None else str(view),
        )
    )
    resolved_seed_config_path = _resolve_runtime_seed_config(
        seed_config_path,
        split_config_path=resolved_split_config_path,
    )
    try:
        loader, _ = build_dataloader(
            runtime_data,
            batch_size=resolved_batch_size,
            shuffle=resolved_shuffle,
            num_workers=resolved_num_workers,
            drop_last=resolved_drop_last,
            pin_memory=resolved_pin_memory,
            persistent_workers=resolved_persistent_workers,
            prefetch_factor=resolved_prefetch_factor,
            seed_namespace=effective_namespace,
            seed_config_path=resolved_seed_config_path,
        )
        config_identity = data_load_config.describe()
        config_identity["resolved"] = {
            "batch_size": int(resolved_batch_size),
            "num_workers": int(resolved_num_workers),
            "pin_memory": resolved_pin_memory,
            "persistent_workers": resolved_persistent_workers,
            "prefetch_factor": int(resolved_prefetch_factor),
            "drop_last": resolved_drop_last,
            "shuffle": resolved_shuffle,
            "cache_mode": resolved_cache_mode,
            "validate_values": resolved_validate_values,
            "selection_residency": {
                "enabled": resolved_selection_resident,
                "pin_memory": resolved_selection_resident_pin_memory,
            },
            "mmap": {
                "access_order": resolved_mmap_access_order,
                "batch_read": resolved_mmap_batch_read,
                "prefetch": resolved_mmap_prefetch.describe(),
            },
            "prepare_for_model": resolved_prepare_for_model,
            "sanitize": resolved_sanitize,
            "global_zscore": resolved_global_zscore,
            "output_layout": resolved_output_layout,
            "epsilon": float(resolved_epsilon),
        }
        loader.runtime_config_identity = config_identity
        return loader
    except Exception:
        runtime_data.close()
        raise


__all__ = [
    "CANONICAL_EVALUATION_PARTITIONS",
    "DEFAULT_CORRUPTION_CACHE_CONFIG",
    "DEFAULT_DATA_LOAD_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "DataContentLedgerConfig",
    "DataLoadConfig",
    "DataLoaderSeedIdentity",
    "ECGModelTransform",
    "ECGSelection",
    "MMapAccessOrder",
    "MMapAdvice",
    "MMapPrefetchConfig",
    "MMapPrefetcher",
    "PN2021_K500_PARTITIONS",
    "PN2021_PARTITIONS",
    "PTBXL_PARTITIONS",
    "RuntimeECGDataset",
    "RuntimeDataLoader",
    "SelectionResidentECGDataset",
    "SequentialEvaluationDataSession",
    "assert_disjoint_selections",
    "build_dataloader",
    "convert_layout",
    "get_dataloader",
    "load_data_load_config",
    "load_selection",
    "per_sample_global_zscore",
    "prepare_model_input",
    "sanitize_nonfinite",
]
