"""CPU-safe helpers for PN2021 clean evaluation caches."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PN2021_EVAL_CACHE_VERSION = "v7_super5_sjr_rgq_review"


@dataclass(frozen=True)
class PN2021EvalCache:
    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    cache_hit: bool
    cache_kind: str


@dataclass(frozen=True)
class PN2021EvalCacheLoad:
    cache: PN2021EvalCache
    converted_npz_to_mmap: bool = False


def pn2021_npz_cache_path(
    cache_dir: os.PathLike[str] | str | None,
    scheme_name: str,
    center: str,
    cache_version: str = PN2021_EVAL_CACHE_VERSION,
) -> str | None:
    if not cache_dir:
        return None
    return os.path.join(
        os.fspath(cache_dir),
        f"{scheme_name}_{center}_100hz1000_{cache_version}.npz",
    )


def pn2021_mmap_cache_path(
    cache_dir: os.PathLike[str] | str | None,
    scheme_name: str,
    center: str,
    cache_version: str = PN2021_EVAL_CACHE_VERSION,
) -> str | None:
    if not cache_dir:
        return None
    return os.path.join(
        os.fspath(cache_dir),
        f"{scheme_name}_{center}_100hz1000_{cache_version}",
    )


def build_pn2021_preprocess_config(
    *,
    preprocess_mode: str,
    norm_mode: str,
    apply_filter: bool,
    apply_zscore: bool,
    include_crop: bool = False,
    crop_len: int | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "target_fs": 100,
        "target_len": 1000,
        "apply_filter": bool(apply_filter),
        "apply_zscore": bool(apply_zscore),
        "preprocess_mode": str(preprocess_mode),
        "norm_mode": str(norm_mode),
    }
    if include_crop:
        config.update({"crop_len": int(crop_len), "crop_mode": "center"})
    return config


def build_pn2021_cache_metadata(
    *,
    scheme_name: str,
    center: str,
    class_names: Sequence[str],
    preprocess_config: Mapping[str, Any],
    pn2021_mapping: Mapping[str, Any] | None = None,
    cache_version: str = PN2021_EVAL_CACHE_VERSION,
    layout: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "scheme": str(scheme_name),
        "center": str(center),
        "class_names": [str(name) for name in class_names],
        "cache_version": str(cache_version),
        "preprocess_config": dict(preprocess_config),
    }
    if layout is not None:
        metadata["layout"] = str(layout)
    if pn2021_mapping is not None:
        metadata["pn2021_mapping"] = dict(pn2021_mapping)
    return metadata


def legacy_default_metadata_variant(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the legacy default-preprocessing metadata shape, if applicable."""
    config = metadata.get("preprocess_config")
    if not isinstance(config, Mapping):
        return None
    if (
        config.get("preprocess_mode") != "legacy_ecgfounder_filter"
        or config.get("norm_mode") != "per_sample_global"
    ):
        return None
    legacy = json.loads(json.dumps(dict(metadata)))
    legacy_config = legacy["preprocess_config"]
    legacy_config.pop("preprocess_mode", None)
    legacy_config.pop("norm_mode", None)
    legacy_config["apply_filter"] = True
    legacy_config["apply_zscore"] = True
    return legacy


def metadata_matches_expected(
    found: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> bool:
    if found is None:
        return False
    found_dict = dict(found)
    expected_dict = dict(expected)
    if found_dict == expected_dict:
        return True
    legacy = legacy_default_metadata_variant(expected_dict)
    return legacy is not None and found_dict == legacy


def load_npz_metadata(data: Any) -> dict[str, Any] | None:
    if "metadata_json" not in getattr(data, "files", []):
        return None
    raw = data["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        decoded = json.loads(str(raw))
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def write_pn2021_mmap_cache(
    cache_root: os.PathLike[str] | str,
    signals: np.ndarray,
    labels: np.ndarray,
    record_ids: Sequence[object] | np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "signals.npy", np.asarray(signals).astype(np.float32, copy=False))
    np.save(root / "labels.npy", np.asarray(labels).astype(np.float32, copy=False))
    np.save(root / "record_ids.npy", np.asarray(record_ids, dtype=str))
    (root / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_pn2021_mmap_cache(
    cache_root: os.PathLike[str] | str | None,
    expected_metadata: Mapping[str, Any],
) -> PN2021EvalCache | None:
    if not cache_root:
        return None
    root = Path(cache_root)
    meta_path = root / "metadata.json"
    signals_path = root / "signals.npy"
    labels_path = root / "labels.npy"
    record_ids_path = root / "record_ids.npy"
    if not (
        meta_path.exists()
        and signals_path.exists()
        and labels_path.exists()
        and record_ids_path.exists()
    ):
        return None
    try:
        found = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not metadata_matches_expected(found, expected_metadata):
        return None
    return PN2021EvalCache(
        signals=np.load(signals_path, mmap_mode="r"),
        labels=np.load(labels_path, mmap_mode="r"),
        record_ids=np.load(record_ids_path, allow_pickle=False).astype(str),
        cache_hit=True,
        cache_kind="mmap",
    )


def load_pn2021_npz_cache(
    cache_path: os.PathLike[str] | str | None,
    expected_metadata: Mapping[str, Any],
) -> PN2021EvalCache | None:
    if not cache_path or not Path(cache_path).exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    if not metadata_matches_expected(load_npz_metadata(data), expected_metadata):
        return None
    return PN2021EvalCache(
        signals=data["signals"].astype(np.float32, copy=False),
        labels=data["labels"].astype(np.float32, copy=False),
        record_ids=data["record_ids"].astype(str),
        cache_hit=True,
        cache_kind="npz",
    )


def load_existing_pn2021_eval_cache(
    *,
    npz_cache_path: os.PathLike[str] | str | None,
    npz_expected_metadata: Mapping[str, Any],
    mmap_cache_root: os.PathLike[str] | str | None,
    mmap_expected_metadata: Mapping[str, Any],
    upgrade_npz_to_mmap: bool = True,
) -> PN2021EvalCacheLoad | None:
    """Load an existing PN2021 cache, preferring mmap and upgrading valid NPZ."""

    mmap_loaded = load_pn2021_mmap_cache(mmap_cache_root, mmap_expected_metadata)
    if mmap_loaded is not None:
        return PN2021EvalCacheLoad(cache=mmap_loaded)

    npz_loaded = load_pn2021_npz_cache(npz_cache_path, npz_expected_metadata)
    if npz_loaded is None:
        return None

    if upgrade_npz_to_mmap and mmap_cache_root:
        write_pn2021_mmap_cache(
            mmap_cache_root,
            npz_loaded.signals,
            npz_loaded.labels,
            npz_loaded.record_ids,
            mmap_expected_metadata,
        )
        mmap_loaded = load_pn2021_mmap_cache(mmap_cache_root, mmap_expected_metadata)
        if mmap_loaded is not None:
            return PN2021EvalCacheLoad(
                cache=mmap_loaded,
                converted_npz_to_mmap=True,
            )

    return PN2021EvalCacheLoad(cache=npz_loaded)
