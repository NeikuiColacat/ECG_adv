from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preprocess.load_cache import (
    EXPECTED_CLASS_ORDER,
    EXPECTED_LEADS,
    ECGCache,
    load_cache,
    load_cache_manifest,
)


def _base_arrays(record_count: int) -> tuple[np.ndarray, ...]:
    signals_100 = np.arange(
        record_count * 1000 * 12, dtype=np.float32
    ).reshape(record_count, 1000, 12)
    signals_100 /= float(signals_100.size)
    signals_500 = np.repeat(signals_100, 5, axis=1)
    labels = np.eye(5, dtype=np.uint8)[:record_count]
    record_ids = np.asarray([f"record-{index}" for index in range(record_count)])
    hash_ids = np.asarray([f"{index:064x}" for index in range(record_count)])
    return signals_100, signals_500, labels, record_ids, hash_ids


def _write_clean_cache(root: Path, *, dataset: str = "pn2021") -> Path:
    root.mkdir()
    record_count = 4
    signals_100, signals_500, labels, record_ids, hash_ids = _base_arrays(
        record_count
    )
    np.save(root / "signals.npy", signals_100, allow_pickle=False)
    np.save(root / "signals_500hz.npy", signals_500, allow_pickle=False)
    np.save(root / "labels_super5.npy", labels, allow_pickle=False)
    np.save(root / "record_ids.npy", record_ids, allow_pickle=False)
    np.save(root / "hash_ids.npy", hash_ids, allow_pickle=False)
    pd.DataFrame(
        {
            "cache_index": np.arange(record_count),
            "record_id": record_ids,
            "record_key": [f"center-{index % 2}/{value}" for index, value in enumerate(record_ids)],
            "hash_id": hash_ids,
            "center": [f"center-{index % 2}" for index in range(record_count)],
        }
    ).to_parquet(root / "records.parquet", index=False)
    labels_manifest = {
        "file": "labels_super5.npy",
        "dtype": "uint8",
        "class_order": list(EXPECTED_CLASS_ORDER),
    }
    if dataset == "pn2021":
        labels_manifest.update(
            {
                "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                "mapping_hash": "555ec85d5b51",
                "all_zero_policy": "kept",
            }
        )
    manifest = {
        "schema_version": 3,
        "dataset": dataset,
        "record_count": record_count,
        "waveform": {
            "file": "signals.npy",
            "shape": [record_count, 1000, 12],
            "dtype": "float32",
            "sampling_rate_hz": 100,
            "duration_seconds": 10,
            "layout": "time_channel",
            "lead_order": list(EXPECTED_LEADS),
            "physical_unit": "mV",
            "normalization": "none",
        },
        "derived_waveforms": {
            "500hz_linear": {
                "file": "signals_500hz.npy",
                "source_file": "signals.npy",
                "shape": [record_count, 5000, 12],
                "dtype": "float32",
                "sampling_rate_hz": 500,
                "duration_seconds": 10,
                "layout": "time_channel",
                "lead_order": list(EXPECTED_LEADS),
                "physical_unit": "mV",
                "interpolation": "linear",
                "align_corners": True,
                "normalization": "none",
            }
        },
        "labels": labels_manifest,
        "files": {
            "record_ids": "record_ids.npy",
            "metadata": "records.parquet",
            "signals_500hz": "signals_500hz.npy",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_corruption_cache(root: Path, *, status: str = "complete") -> Path:
    root.mkdir()
    record_count = 4
    view_count = 2
    signals_100, signals_500, labels, record_ids, hash_ids = _base_arrays(
        record_count
    )
    corrupted_100 = np.stack((signals_100 + 1.0, signals_100 + 2.0))
    corrupted_500 = np.stack((signals_500 + 1.0, signals_500 + 2.0))
    np.save(root / "signals_100hz.npy", corrupted_100, allow_pickle=False)
    np.save(root / "signals_500hz.npy", corrupted_500, allow_pickle=False)
    np.save(root / "labels_super5.npy", labels, allow_pickle=False)
    np.save(root / "record_ids.npy", record_ids, allow_pickle=False)
    np.save(root / "hash_ids.npy", hash_ids, allow_pickle=False)
    pd.DataFrame(
        {
            "cache_index": np.arange(record_count),
            "source_cache_index": np.arange(record_count),
            "record_id": record_ids,
            "record_key": [f"center-{index % 2}/{value}" for index, value in enumerate(record_ids)],
            "hash_id": hash_ids,
            "center": [f"center-{index % 2}" for index in range(record_count)],
        }
    ).to_parquet(root / "records.parquet", index=False)
    compositions = [
        {
            "view_index": 0,
            "depth": 2,
            "operators": ["powerline_noise", "emg_noise"],
            "composition_id": "d2__powerline_noise__emg_noise",
        },
        {
            "view_index": 1,
            "depth": 3,
            "operators": [
                "baseline_wander",
                "baseline_shift",
                "random_leads_masking",
            ],
            "composition_id": (
                "d3__baseline_wander__baseline_shift__random_leads_masking"
            ),
        },
    ]
    (root / "compositions.json").write_text(
        json.dumps(compositions), encoding="utf-8"
    )
    (root / "build_state.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "dataset": "pn2021c",
        "cache_version": "test-pn2021c",
        "record_count": record_count,
        "view_count": view_count,
        "source": {
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
            "class_order": list(EXPECTED_CLASS_ORDER),
            "lead_order": list(EXPECTED_LEADS),
            "physical_unit": "mV",
            "normalization": "none",
        },
        "waveforms": {
            "100hz": {
                "file": "signals_100hz.npy",
                "shape": [view_count, record_count, 1000, 12],
                "dtype": "float32",
                "sampling_rate_hz": 100,
                "duration_seconds": 10,
                "layout": "view_record_time_channel",
                "normalization": "none",
            },
            "500hz": {
                "file": "signals_500hz.npy",
                "shape": [view_count, record_count, 5000, 12],
                "dtype": "float32",
                "sampling_rate_hz": 500,
                "duration_seconds": 10,
                "layout": "view_record_time_channel",
                "normalization": "none",
            },
        },
        "files": {
            "labels": "labels_super5.npy",
            "record_ids": "record_ids.npy",
            "hash_ids": "hash_ids.npy",
            "metadata": "records.parquet",
            "compositions": "compositions.json",
            "build_state": "build_state.json",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_clean_cache_supports_manifest_mmap_ram_and_record_access(tmp_path: Path):
    root = _write_clean_cache(tmp_path / "clean")
    identity = load_cache_manifest(root, sampling_rate_hz=100)

    assert identity.dataset == "pn2021"
    assert identity.signal_shape == (4, 1000, 12)
    assert identity.view_count == 1

    with load_cache(root, sampling_rate_hz=100, mode="mmap") as cache:
        assert isinstance(cache, ECGCache)
        assert isinstance(cache.signals, np.memmap)
        assert len(cache) == 4
        assert cache.available_centers == ("center-0", "center-1")
        assert cache.index_for_hash(f"{2:064x}") == 2
        np.testing.assert_array_equal(
            cache.indices_for_hashes(
                [f"{2:064x}", f"{0:064x}", f"{2:064x}"]
            ),
            [2, 0, 2],
        )
        with pytest.raises(KeyError, match="missing hash_id"):
            cache.indices_for_hashes([f"{0:064x}", "missing-hash"])
        np.testing.assert_array_equal(cache.indices_for_center("center-0"), [0, 2])

        record = cache.get_record(1)
        assert record.signal.shape == (1000, 12)
        assert record.signal.flags.owndata
        assert record.label.shape == (5,)
        assert record.record_id == "record-1"
        assert record.hash_id == f"{1:064x}"
        assert record.metadata["cache_index"] == 1

        batch = cache.get_batch([0, 2], layout="channel_time")
        assert batch.signals.shape == (2, 12, 1000)
        assert batch.signals.flags.c_contiguous
        assert batch.labels.shape == (2, 5)
        assert batch.view_index is None

    with load_cache(root, sampling_rate_hz=500, mode="ram") as cache_500:
        assert not isinstance(cache_500.signals, np.memmap)
        assert cache_500.signals.shape == (4, 5000, 12)
        assert not cache_500.signals.flags.writeable


def test_corruption_cache_requires_explicit_view_and_supports_composition_id(
    tmp_path: Path,
):
    root = _write_corruption_cache(tmp_path / "corrupted")
    with load_cache(root, sampling_rate_hz=100, mode="mmap") as cache:
        assert cache.is_corruption
        assert cache.view_count == 2
        assert cache.list_views(depth=2)[0]["view_index"] == 0
        with pytest.raises(ValueError, match="view is required"):
            cache.get_record(0)

        composition_id = "d3__baseline_wander__baseline_shift__random_leads_masking"
        record = cache.get_record(0, view=composition_id)
        assert record.view_index == 1
        assert record.composition_id == composition_id
        assert float(record.signal[0, 0]) == pytest.approx(2.0)

        batch = cache.get_batch(slice(1, 3), view=0)
        assert batch.signals.shape == (2, 1000, 12)
        assert batch.signals.flags.owndata
        assert batch.view_index == 0
        assert batch.composition_id == "d2__powerline_noise__emg_noise"


def test_loader_rejects_incomplete_corruption_cache(tmp_path: Path):
    root = _write_corruption_cache(tmp_path / "building", status="building")
    with pytest.raises(ValueError, match="not complete"):
        load_cache(root)


def test_loader_rejects_waveform_contract_mismatch(tmp_path: Path):
    root = _write_clean_cache(tmp_path / "invalid")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["waveform"]["lead_order"][0] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lead order"):
        load_cache(root)


def test_ptbxl_preserves_source_multilabel_norm_cooccurrence(tmp_path: Path):
    root = _write_clean_cache(tmp_path / "ptbxl", dataset="ptbxl")
    labels_path = root / "labels_super5.npy"
    labels = np.load(labels_path)
    labels[0, 0] = 1
    labels[0, 3] = 1
    np.save(labels_path, labels, allow_pickle=False)

    with load_cache(root, mode="mmap") as cache:
        np.testing.assert_array_equal(cache.labels[0], [1, 0, 0, 1, 0])


def test_ram_mode_enforces_available_memory_reserve(tmp_path: Path, monkeypatch):
    root = _write_clean_cache(tmp_path / "memory")
    monkeypatch.setattr(
        "data_preprocess.load_cache._available_memory_bytes", lambda: 1024
    )
    with pytest.raises(MemoryError, match="RAM safety reserve"):
        load_cache(root, mode="ram", minimum_free_ram_bytes=1024)


def test_auto_mode_chooses_ram_or_mmap_from_live_available_memory(
    tmp_path: Path, monkeypatch
):
    root = _write_clean_cache(tmp_path / "automatic")
    monkeypatch.setattr(
        "data_preprocess.load_cache._available_memory_bytes", lambda: 1024**4
    )
    with load_cache(root, mode="auto", minimum_free_ram_bytes=1024) as cache:
        assert cache.storage_mode == "ram"

    monkeypatch.setattr(
        "data_preprocess.load_cache._available_memory_bytes", lambda: 1024
    )
    with load_cache(root, mode="auto", minimum_free_ram_bytes=1024) as cache:
        assert cache.storage_mode == "mmap"
