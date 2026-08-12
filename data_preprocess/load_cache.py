"""Validated, read-only access to manual-refactor ECG NumPy caches.

This module is the single data-access boundary for the new PTB-XL, PN2021 and
PN2021-C caches. It validates the persisted manifest before exposing arrays and
never performs filtering, resampling, augmentation or normalization. Signals
therefore remain raw physical-mV waveforms until the model input pipeline.

Arrays are always opened as read-only NumPy memmaps. One cache handle represents
one sampling rate, so callers explicitly choose 100 or 500 Hz.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


EXPECTED_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
EXPECTED_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
PN2021_MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
PN2021_MAPPING_HASH = "555ec85d5b51"

StorageMode = Literal["mmap"]
ValueValidation = Literal["none", "sample", "full"]
SignalLayout = Literal["time_channel", "channel_time"]
ViewSelector = int | str | None
RecordSelector = int | slice | Sequence[int] | np.ndarray


@dataclass(frozen=True)
class CacheManifest:
    """Resolved and validated identity for one cache sampling-rate view."""

    cache_dir: Path
    manifest_path: Path
    manifest_sha256: str
    raw_manifest: dict[str, Any]
    schema_version: int
    dataset: str
    cache_version: str | None
    record_count: int
    view_count: int
    sampling_rate_hz: int
    duration_seconds: float
    signal_shape: tuple[int, ...]
    signal_dtype: np.dtype
    storage_layout: str
    lead_order: tuple[str, ...]
    physical_unit: str
    normalization: str
    class_order: tuple[str, ...]
    mapping_version: str | None
    mapping_hash: str | None
    signals_path: Path
    labels_path: Path
    record_ids_path: Path
    hash_ids_path: Path
    metadata_path: Path
    compositions_path: Path | None
    build_state_path: Path | None

    @property
    def is_corruption(self) -> bool:
        return self.dataset == "pn2021c"


@dataclass(frozen=True)
class ECGRecord:
    """One accessed ECG together with its immutable cache identity fields."""

    signal: np.ndarray
    label: np.ndarray
    index: int
    record_id: str
    hash_id: str
    metadata: dict[str, Any]
    view_index: int | None
    composition_id: str | None


@dataclass(frozen=True)
class ECGBatch:
    """A batch returned by :meth:`ECGCache.get_batch`."""

    signals: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    record_ids: np.ndarray
    hash_ids: np.ndarray
    view_index: int | None
    composition_id: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _resolve_member(cache_dir: Path, value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    member = Path(value)
    if member.is_absolute():
        raise ValueError(f"{description} must stay inside the cache: {value}")
    resolved = (cache_dir / member).resolve()
    try:
        resolved.relative_to(cache_dir)
    except ValueError:
        raise ValueError(f"{description} escapes the cache directory: {value}") from None
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _validate_common_waveform_contract(
    *,
    entry: dict[str, Any],
    source: dict[str, Any],
    sampling_rate_hz: int,
    expected_shape: tuple[int, ...],
) -> tuple[np.dtype, tuple[str, ...], str, str, float]:
    shape = tuple(int(value) for value in entry.get("shape", ()))
    if shape != expected_shape:
        raise ValueError(
            f"waveform shape mismatch: manifest={shape}, expected={expected_shape}"
        )
    if int(entry.get("sampling_rate_hz", -1)) != sampling_rate_hz:
        raise ValueError("waveform sampling rate does not match the requested rate")
    dtype = np.dtype(entry.get("dtype", ""))
    if dtype != np.dtype(np.float32):
        raise ValueError(f"waveform dtype must be float32, got {dtype}")

    lead_order = tuple(entry.get("lead_order", source.get("lead_order", ())))
    if lead_order != EXPECTED_LEADS:
        raise ValueError(
            f"waveform lead order must be {EXPECTED_LEADS}, got {lead_order}"
        )
    physical_unit = str(
        entry.get("physical_unit", source.get("physical_unit", ""))
    )
    if physical_unit != "mV":
        raise ValueError(f"waveform physical unit must be mV, got {physical_unit!r}")
    normalization = str(
        entry.get("normalization", source.get("normalization", ""))
    )
    if normalization != "none":
        raise ValueError(
            "cache waveform must remain unnormalized raw mV; "
            f"got normalization={normalization!r}"
        )
    time_size = expected_shape[-2]
    inferred_duration = float(time_size) / float(sampling_rate_hz)
    duration = float(entry.get("duration_seconds", inferred_duration))
    if not np.isclose(duration, 10.0) or time_size != sampling_rate_hz * 10:
        raise ValueError(
            f"waveform must contain exactly ten seconds, got {duration}s/{time_size}"
        )
    return dtype, lead_order, physical_unit, normalization, duration


def load_cache_manifest(
    cache_dir: str | Path,
    *,
    sampling_rate_hz: int = 100,
) -> CacheManifest:
    """Resolve and validate one cache without opening its waveform array."""

    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"cache directory not found: {root}")
    if int(sampling_rate_hz) not in {100, 500}:
        raise ValueError("sampling_rate_hz must be either 100 or 500")
    sampling_rate_hz = int(sampling_rate_hz)

    manifest_path = root / "manifest.json"
    manifest = _read_json_mapping(manifest_path, description="cache manifest")
    dataset = str(manifest.get("dataset", ""))
    schema_version = int(manifest.get("schema_version", -1))
    if dataset not in {"ptbxl", "pn2021", "pn2021c"}:
        raise ValueError(f"unsupported cache dataset: {dataset!r}")
    if (dataset == "pn2021c" and schema_version != 1) or (
        dataset != "pn2021c" and schema_version != 3
    ):
        raise ValueError(
            f"unsupported {dataset} cache schema_version: {schema_version}"
        )

    record_count = int(manifest.get("record_count", 0))
    if record_count <= 0:
        raise ValueError("cache record_count must be positive")
    files = _require_mapping(manifest.get("files"), description="manifest.files")

    if dataset == "pn2021c":
        view_count = int(manifest.get("view_count", 0))
        if view_count <= 0:
            raise ValueError("PN2021-C view_count must be positive")
        source = _require_mapping(manifest.get("source"), description="manifest.source")
        waveforms = _require_mapping(
            manifest.get("waveforms"), description="manifest.waveforms"
        )
        entry = _require_mapping(
            waveforms.get(f"{sampling_rate_hz}hz"),
            description=f"manifest.waveforms.{sampling_rate_hz}hz",
        )
        expected_shape = (
            view_count,
            record_count,
            sampling_rate_hz * 10,
            12,
        )
        expected_layout = "view_record_time_channel"
        class_order = tuple(source.get("class_order", ()))
        mapping_version = source.get("mapping_version")
        mapping_hash = source.get("mapping_hash")
        labels_file = files.get("labels")
        compositions_path = _resolve_member(
            root, files.get("compositions"), description="compositions file"
        )
        build_state_path = _resolve_member(
            root, files.get("build_state"), description="build-state file"
        )
        build_state = _read_json_mapping(
            build_state_path, description="cache build state"
        )
        if build_state.get("status") != "complete":
            raise ValueError(
                "PN2021-C cache is not complete: "
                f"status={build_state.get('status')!r}"
            )
    else:
        view_count = 1
        source = {}
        if sampling_rate_hz == 100:
            entry = _require_mapping(
                manifest.get("waveform"), description="manifest.waveform"
            )
        else:
            derived = _require_mapping(
                manifest.get("derived_waveforms"),
                description="manifest.derived_waveforms",
            )
            entry = _require_mapping(
                derived.get("500hz_linear"),
                description="manifest.derived_waveforms.500hz_linear",
            )
        expected_shape = (record_count, sampling_rate_hz * 10, 12)
        expected_layout = "time_channel"
        labels_manifest = _require_mapping(
            manifest.get("labels"), description="manifest.labels"
        )
        class_order = tuple(labels_manifest.get("class_order", ()))
        mapping_version = labels_manifest.get("mapping_version")
        mapping_hash = labels_manifest.get("mapping_hash")
        labels_file = labels_manifest.get("file")
        compositions_path = None
        build_state_path = None

    if str(entry.get("layout", "")) != expected_layout:
        raise ValueError(
            f"waveform layout must be {expected_layout!r}, "
            f"got {entry.get('layout')!r}"
        )
    (
        signal_dtype,
        lead_order,
        physical_unit,
        normalization,
        duration,
    ) = _validate_common_waveform_contract(
        entry=entry,
        source=source,
        sampling_rate_hz=sampling_rate_hz,
        expected_shape=expected_shape,
    )
    if class_order != EXPECTED_CLASS_ORDER:
        raise ValueError(
            f"Super5 class order must be {EXPECTED_CLASS_ORDER}, got {class_order}"
        )
    if dataset in {"pn2021", "pn2021c"} and (
        mapping_version != PN2021_MAPPING_VERSION
        or mapping_hash != PN2021_MAPPING_HASH
    ):
        raise ValueError(
            "PN2021 Super5 mapping identity mismatch: "
            f"{mapping_version!r}/{mapping_hash!r}"
        )

    signals_path = _resolve_member(
        root, entry.get("file"), description=f"{sampling_rate_hz} Hz signals file"
    )
    labels_path = _resolve_member(root, labels_file, description="labels file")
    record_ids_path = _resolve_member(
        root, files.get("record_ids"), description="record-IDs file"
    )
    hash_ids_path = _resolve_member(
        root, files.get("hash_ids", "hash_ids.npy"), description="hash-IDs file"
    )
    metadata_path = _resolve_member(
        root, files.get("metadata"), description="records metadata file"
    )
    cache_version_value = manifest.get(
        "cache_version", manifest.get("dataset_version")
    )
    cache_version = (
        None if cache_version_value is None else str(cache_version_value)
    )
    return CacheManifest(
        cache_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        raw_manifest=manifest,
        schema_version=schema_version,
        dataset=dataset,
        cache_version=cache_version,
        record_count=record_count,
        view_count=view_count,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration,
        signal_shape=expected_shape,
        signal_dtype=signal_dtype,
        storage_layout=expected_layout,
        lead_order=lead_order,
        physical_unit=physical_unit,
        normalization=normalization,
        class_order=class_order,
        mapping_version=None if mapping_version is None else str(mapping_version),
        mapping_hash=None if mapping_hash is None else str(mapping_hash),
        signals_path=signals_path,
        labels_path=labels_path,
        record_ids_path=record_ids_path,
        hash_ids_path=hash_ids_path,
        metadata_path=metadata_path,
        compositions_path=compositions_path,
        build_state_path=build_state_path,
    )


def _open_npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _close_memmap(array: np.ndarray | None) -> None:
    if isinstance(array, np.memmap):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


class ECGCache:
    """Read-only cache handle with record, center and corruption-view access."""

    storage_mode: Literal["mmap"] = "mmap"

    def __init__(
        self,
        *,
        identity: CacheManifest,
        signals: np.ndarray,
        labels: np.ndarray,
        record_ids: np.ndarray,
        hash_ids: np.ndarray,
        records: pd.DataFrame,
        compositions: tuple[dict[str, Any], ...],
    ) -> None:
        self.identity = identity
        self.signals = signals
        self.labels = labels
        self.record_ids = record_ids
        self.hash_ids = hash_ids
        self.records = records
        self.compositions = compositions
        self._hash_to_index = {
            str(value): index for index, value in enumerate(hash_ids.astype(str))
        }
        self._composition_to_view = {
            str(item["composition_id"]): int(item["view_index"])
            for item in compositions
        }
        self._closed = False

    def __len__(self) -> int:
        return self.identity.record_count

    @property
    def dataset(self) -> str:
        return self.identity.dataset

    @property
    def sampling_rate_hz(self) -> int:
        return self.identity.sampling_rate_hz

    @property
    def view_count(self) -> int:
        return self.identity.view_count

    @property
    def is_corruption(self) -> bool:
        return self.identity.is_corruption

    @property
    def available_centers(self) -> tuple[str, ...]:
        if "center" not in self.records.columns:
            return ()
        return tuple(sorted(self.records["center"].astype(str).unique()))

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable description for run manifests."""

        return {
            "cache_dir": str(self.identity.cache_dir),
            "manifest_sha256": self.identity.manifest_sha256,
            "dataset": self.dataset,
            "cache_version": self.identity.cache_version,
            "record_count": len(self),
            "view_count": self.view_count,
            "sampling_rate_hz": self.sampling_rate_hz,
            "duration_seconds": self.identity.duration_seconds,
            "signal_shape": list(self.identity.signal_shape),
            "lead_order": list(self.identity.lead_order),
            "class_order": list(self.identity.class_order),
            "mapping_version": self.identity.mapping_version,
            "mapping_hash": self.identity.mapping_hash,
            "physical_unit": self.identity.physical_unit,
            "normalization": self.identity.normalization,
            "storage_mode": self.storage_mode,
            "centers": list(self.available_centers),
        }

    def indices_for_hashes(
        self,
        hash_ids: Sequence[str] | np.ndarray,
    ) -> np.ndarray:
        """Resolve a one-dimensional hash sequence to ordered int64 indices.

        Input order and duplicate hashes are preserved. Resolution is atomic:
        if any hash is absent, the method raises instead of returning a partial
        result. Call this once when materializing a split, then use integer
        indices inside the training loop.
        """

        if isinstance(hash_ids, (str, bytes, np.str_, np.bytes_)):
            raise TypeError("hash_ids must be a one-dimensional sequence")
        values = np.asarray(hash_ids)
        if values.ndim != 1:
            raise ValueError("hash_ids must be one-dimensional")
        resolved = np.empty(values.size, dtype=np.int64)
        missing: list[str] = []
        for position, value in enumerate(values):
            if isinstance(value, (bytes, np.bytes_)):
                key = bytes(value).decode("ascii")
            elif isinstance(value, (str, np.str_)):
                key = str(value)
            else:
                raise TypeError("hash_ids must contain only string values")
            index = self._hash_to_index.get(key)
            if index is None:
                missing.append(key)
            else:
                resolved[position] = index
        if missing:
            preview = missing[:5]
            raise KeyError(
                f"missing hash_id values: count={len(missing)}, preview={preview}"
            )
        return resolved

    def _resolve_view(self, view: ViewSelector) -> tuple[int | None, str | None]:
        if not self.is_corruption:
            if view is not None:
                raise ValueError("clean caches do not accept a corruption view")
            return None, None
        if view is None:
            raise ValueError("view is required when accessing a PN2021-C cache")
        if isinstance(view, bool):
            raise TypeError("view must be an integer index or composition ID")
        if isinstance(view, str):
            try:
                view_index = self._composition_to_view[view]
            except KeyError:
                raise KeyError(f"unknown PN2021-C composition_id: {view}") from None
        elif isinstance(view, (int, np.integer)):
            view_index = int(view)
        else:
            raise TypeError("view must be an integer index or composition ID")
        if not 0 <= view_index < self.view_count:
            raise IndexError(
                f"view index {view_index} is outside [0, {self.view_count})"
            )
        item = self.compositions[view_index]
        return view_index, str(item["composition_id"])

    def _normalize_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("record index must be an integer")
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError(f"record index {index} is outside [0, {len(self)})")
        return resolved

    def _normalize_indices(
        self, indices: RecordSelector
    ) -> tuple[int | slice | np.ndarray, np.ndarray]:
        if isinstance(indices, slice):
            resolved = np.arange(len(self), dtype=np.int64)[indices]
            if resolved.size == 0:
                raise ValueError("record batch cannot be empty")
            return indices, resolved
        if isinstance(indices, (int, np.integer)) and not isinstance(indices, bool):
            resolved_index = self._normalize_index(int(indices))
            resolved = np.asarray([resolved_index], dtype=np.int64)
            return resolved, resolved
        array = np.asarray(indices)
        if array.ndim != 1:
            raise ValueError("record indices must be one-dimensional")
        if array.dtype == np.bool_:
            if len(array) != len(self):
                raise ValueError("boolean record mask must match cache record_count")
            resolved = np.flatnonzero(array).astype(np.int64, copy=False)
        else:
            if not np.issubdtype(array.dtype, np.integer):
                raise TypeError("record indices must contain integers")
            resolved = array.astype(np.int64, copy=True)
            resolved[resolved < 0] += len(self)
            if np.any((resolved < 0) | (resolved >= len(self))):
                raise IndexError("one or more record indices are outside the cache")
        if resolved.size == 0:
            raise ValueError("record batch cannot be empty")
        return resolved, resolved

    @staticmethod
    def _format_signals(
        signals: np.ndarray,
        *,
        layout: SignalLayout,
        copy: bool,
    ) -> np.ndarray:
        if layout not in {"time_channel", "channel_time"}:
            raise ValueError("layout must be 'time_channel' or 'channel_time'")
        output = np.asarray(signals)
        if layout == "channel_time":
            output = np.swapaxes(output, -1, -2)
        if copy:
            output = np.array(output, dtype=np.float32, order="C", copy=True)
        elif layout == "channel_time":
            output = np.ascontiguousarray(output, dtype=np.float32)
        return output

    def get_record(
        self,
        index: int,
        *,
        view: ViewSelector = None,
        layout: SignalLayout = "time_channel",
        copy: bool = True,
        check_finite: bool = True,
    ) -> ECGRecord:
        """Access one record without applying model-side normalization."""

        resolved_index = self._normalize_index(index)
        view_index, composition_id = self._resolve_view(view)
        signal = (
            self.signals[resolved_index]
            if view_index is None
            else self.signals[view_index, resolved_index]
        )
        signal = self._format_signals(signal, layout=layout, copy=copy)
        if check_finite and not np.isfinite(signal).all():
            raise ValueError(
                f"non-finite waveform at record={resolved_index}, view={view_index}"
            )
        return ECGRecord(
            signal=signal,
            label=np.asarray(self.labels[resolved_index], dtype=np.uint8).copy(),
            index=resolved_index,
            record_id=str(self.record_ids[resolved_index]),
            hash_id=str(self.hash_ids[resolved_index]),
            metadata=self.records.iloc[resolved_index].to_dict(),
            view_index=view_index,
            composition_id=composition_id,
        )

    def get_batch(
        self,
        indices: RecordSelector,
        *,
        view: ViewSelector = None,
        layout: SignalLayout = "time_channel",
        copy: bool = True,
        check_finite: bool = True,
    ) -> ECGBatch:
        """Access a record batch in classifier-friendly NTC or NCT layout."""

        selector, resolved_indices = self._normalize_indices(indices)
        view_index, composition_id = self._resolve_view(view)
        signals = (
            self.signals[selector]
            if view_index is None
            else self.signals[view_index, selector]
        )
        signals = self._format_signals(signals, layout=layout, copy=copy)
        if check_finite and not np.isfinite(signals).all():
            raise ValueError(
                "non-finite waveform in requested batch: "
                f"records={resolved_indices.tolist()}, view={view_index}"
            )
        labels = np.asarray(self.labels[selector], dtype=np.uint8)
        record_ids = np.asarray(self.record_ids[selector]).astype(str, copy=False)
        hash_ids = np.asarray(self.hash_ids[selector]).astype(str, copy=False)
        if copy:
            labels = labels.copy()
            record_ids = record_ids.copy()
            hash_ids = hash_ids.copy()
        return ECGBatch(
            signals=signals,
            labels=labels,
            indices=resolved_indices.copy(),
            record_ids=record_ids,
            hash_ids=hash_ids,
            view_index=view_index,
            composition_id=composition_id,
        )

    def validate_waveforms(
        self,
        *,
        full: bool = False,
        chunk_records: int = 64,
    ) -> int:
        """Check deterministic samples or the complete cache for finite values."""

        if isinstance(chunk_records, bool) or int(chunk_records) <= 0:
            raise ValueError("chunk_records must be a positive integer")
        checked = 0
        view_indices: tuple[int | None, ...]
        if self.is_corruption:
            if full:
                view_indices = tuple(range(self.view_count))
            else:
                view_indices = tuple(
                    sorted({0, self.view_count // 2, self.view_count - 1})
                )
        else:
            view_indices = (None,)
        if not full:
            record_indices = tuple(sorted({0, len(self) // 2, len(self) - 1}))
            for view_index in view_indices:
                for record_index in record_indices:
                    signal = (
                        self.signals[record_index]
                        if view_index is None
                        else self.signals[view_index, record_index]
                    )
                    if not np.isfinite(signal).all():
                        raise ValueError(
                            "non-finite waveform during sample validation: "
                            f"record={record_index}, view={view_index}"
                        )
                    checked += 1
            return checked

        for view_index in view_indices:
            for start in range(0, len(self), int(chunk_records)):
                end = min(start + int(chunk_records), len(self))
                batch = (
                    self.signals[start:end]
                    if view_index is None
                    else self.signals[view_index, start:end]
                )
                if not np.isfinite(batch).all():
                    raise ValueError(
                        "non-finite waveform during full validation: "
                        f"records={start}:{end}, view={view_index}"
                    )
                checked += end - start
        return checked

    def close(self) -> None:
        """Close read-only memmaps owned by this handle."""

        if self._closed:
            return
        for array in (self.signals, self.labels, self.record_ids, self.hash_ids):
            _close_memmap(array)
        self._closed = True

    def __enter__(self) -> ECGCache:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _validate_opened_arrays(
    *,
    identity: CacheManifest,
    signals: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    hash_ids: np.ndarray,
    records: pd.DataFrame,
) -> None:
    if signals.shape != identity.signal_shape or signals.dtype != np.float32:
        raise ValueError(
            f"signals array contract mismatch: {signals.shape}/{signals.dtype}, "
            f"expected {identity.signal_shape}/float32"
        )
    if labels.shape != (identity.record_count, 5) or labels.dtype != np.uint8:
        raise ValueError(
            f"labels array must be ({identity.record_count}, 5) uint8, "
            f"got {labels.shape}/{labels.dtype}"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Super5 labels must contain only binary 0/1 values")
    if identity.dataset in {"pn2021", "pn2021c"}:
        norm_positive = labels[:, 3] == 1
        abnormal_positive = labels[:, [0, 1, 2, 4]].any(axis=1)
        if np.any(norm_positive & abnormal_positive):
            raise ValueError(
                "PN2021 v7 NORM labels cannot coexist with abnormal Super5 labels"
            )
    expected_vector_shape = (identity.record_count,)
    if record_ids.shape != expected_vector_shape or hash_ids.shape != expected_vector_shape:
        raise ValueError("record/hash ID arrays do not match cache record_count")
    if len(records) != identity.record_count:
        raise ValueError("records metadata does not match cache record_count")
    required_columns = {"cache_index", "record_id", "record_key", "hash_id"}
    missing = sorted(required_columns - set(records.columns))
    if missing:
        raise ValueError(f"records metadata is missing required columns: {missing}")
    cache_indices = records["cache_index"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(cache_indices, np.arange(identity.record_count)):
        raise ValueError("records.cache_index must be contiguous and position-aligned")
    record_strings = record_ids.astype(str)
    hash_strings = hash_ids.astype(str)
    if records["record_id"].astype(str).tolist() != record_strings.tolist():
        raise ValueError("record_ids.npy is not aligned with records metadata")
    if records["hash_id"].astype(str).tolist() != hash_strings.tolist():
        raise ValueError("hash_ids.npy is not aligned with records metadata")
    if len(set(hash_strings.tolist())) != identity.record_count:
        raise ValueError("hash IDs must be unique within the cache")
    hexadecimal = set("0123456789abcdef")
    if any(len(value) != 64 or not set(value.lower()) <= hexadecimal for value in hash_strings):
        raise ValueError("each hash ID must be a 64-character hexadecimal SHA256 value")


def _load_compositions(identity: CacheManifest) -> tuple[dict[str, Any], ...]:
    if not identity.is_corruption:
        return ()
    assert identity.compositions_path is not None
    try:
        payload = json.loads(identity.compositions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid compositions JSON: {exc}") from exc
    if not isinstance(payload, list) or len(payload) != identity.view_count:
        raise ValueError("compositions must contain exactly one item per view")
    compositions: list[dict[str, Any]] = []
    ids: set[str] = set()
    for expected_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("each composition must be a mapping")
        if int(item.get("view_index", -1)) != expected_index:
            raise ValueError("composition view indices must be contiguous and ordered")
        depth = int(item.get("depth", -1))
        operators = item.get("operators")
        composition_id = str(item.get("composition_id", ""))
        if depth not in {2, 3} or not isinstance(operators, list):
            raise ValueError("each composition requires depth 2/3 and an operator list")
        if len(operators) != depth:
            raise ValueError("composition operator count must equal its depth")
        if not composition_id or composition_id in ids:
            raise ValueError("composition IDs must be non-empty and unique")
        ids.add(composition_id)
        compositions.append(dict(item))
    return tuple(compositions)


def load_cache(
    cache_dir: str | Path,
    *,
    sampling_rate_hz: int = 100,
    mode: StorageMode = "mmap",
    validate_values: ValueValidation = "sample",
) -> ECGCache:
    """Open a validated ECG cache as read-only memory maps."""

    if type(mode) is not str or mode != "mmap":
        raise ValueError("cache mode must be mmap")
    if validate_values not in {"none", "sample", "full"}:
        raise ValueError("validate_values must be 'none', 'sample' or 'full'")
    identity = load_cache_manifest(
        cache_dir, sampling_rate_hz=sampling_rate_hz
    )
    signals: np.ndarray | None = None
    labels: np.ndarray | None = None
    record_ids: np.ndarray | None = None
    hash_ids: np.ndarray | None = None
    try:
        signals = _open_npy(identity.signals_path)
        labels = _open_npy(identity.labels_path)
        record_ids = _open_npy(identity.record_ids_path)
        hash_ids = _open_npy(identity.hash_ids_path)
        records = pd.read_parquet(identity.metadata_path)
        _validate_opened_arrays(
            identity=identity,
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            hash_ids=hash_ids,
            records=records,
        )
        compositions = _load_compositions(identity)
        cache = ECGCache(
            identity=identity,
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            hash_ids=hash_ids,
            records=records,
            compositions=compositions,
        )
        if validate_values == "sample":
            cache.validate_waveforms(full=False)
        elif validate_values == "full":
            cache.validate_waveforms(full=True)
        return cache
    except Exception:
        for array in (signals, labels, record_ids, hash_ids):
            _close_memmap(array)
        raise


__all__ = [
    "EXPECTED_CLASS_ORDER",
    "EXPECTED_LEADS",
    "PN2021_MAPPING_HASH",
    "PN2021_MAPPING_VERSION",
    "CacheManifest",
    "ECGBatch",
    "ECGCache",
    "ECGRecord",
    "load_cache",
    "load_cache_manifest",
]
