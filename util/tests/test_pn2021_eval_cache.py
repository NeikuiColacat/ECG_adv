"""CPU-only tests for PN2021 clean eval cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ecg_adv_gen.data.pn2021 import (
    expected_pn2021_mmap_metadata,
    load_or_build_pn2021_center,
)
from ecg_adv_gen.evaluation.pn2021_eval_cache import (
    PN2021_EVAL_CACHE_VERSION,
    build_pn2021_cache_metadata,
    build_pn2021_preprocess_config,
    load_existing_pn2021_eval_cache,
    load_npz_metadata,
    load_pn2021_mmap_cache,
    metadata_matches_expected,
    pn2021_mmap_cache_path,
    pn2021_npz_cache_path,
    write_pn2021_mmap_cache,
)


class _NpzLike:
    def __init__(self, payload):
        self.files = list(payload)
        self._payload = payload

    def __getitem__(self, key):
        return self._payload[key]


def test_pn2021_clean_cache_paths_are_versioned_and_stable():
    assert PN2021_EVAL_CACHE_VERSION == "v7_super5_sjr_rgq_review"
    assert pn2021_npz_cache_path("/npz", "super5", "ningbo") == (
        "/npz/super5_ningbo_100hz1000_v7_super5_sjr_rgq_review.npz"
    )
    assert pn2021_mmap_cache_path("/mmap", "super5", "georgia") == (
        "/mmap/super5_georgia_100hz1000_v7_super5_sjr_rgq_review"
    )
    assert pn2021_npz_cache_path(None, "super5", "ningbo") is None
    assert pn2021_mmap_cache_path("", "super5", "ningbo") is None


def test_build_pn2021_cache_metadata_includes_crop_and_mapping_when_requested():
    preprocess = build_pn2021_preprocess_config(
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
        apply_filter=False,
        apply_zscore=True,
        include_crop=True,
        crop_len=250,
    )
    metadata = build_pn2021_cache_metadata(
        scheme_name="super5",
        center="ningbo",
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
        preprocess_config=preprocess,
        pn2021_mapping={"mapping_version": "vtest", "mapping_hash": "htest"},
    )

    assert metadata == {
        "scheme": "super5",
        "center": "ningbo",
        "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
        "cache_version": PN2021_EVAL_CACHE_VERSION,
        "preprocess_config": {
            "target_fs": 100,
            "target_len": 1000,
            "apply_filter": False,
            "apply_zscore": True,
            "preprocess_mode": "minimal_resample",
            "norm_mode": "per_sample_global",
            "crop_len": 250,
            "crop_mode": "center",
        },
        "pn2021_mapping": {"mapping_version": "vtest", "mapping_hash": "htest"},
    }


def test_metadata_matches_expected_accepts_legacy_default_variant():
    expected = build_pn2021_cache_metadata(
        scheme_name="super5",
        center="ningbo",
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
        preprocess_config=build_pn2021_preprocess_config(
            preprocess_mode="legacy_ecgfounder_filter",
            norm_mode="per_sample_global",
            apply_filter=True,
            apply_zscore=True,
            include_crop=False,
        ),
    )
    legacy = json.loads(json.dumps(expected))
    legacy["preprocess_config"].pop("preprocess_mode")
    legacy["preprocess_config"].pop("norm_mode")

    assert metadata_matches_expected(expected, expected)
    assert metadata_matches_expected(legacy, expected)
    assert not metadata_matches_expected({"center": "georgia"}, expected)
    assert not metadata_matches_expected(None, expected)


def test_load_npz_metadata_decodes_json_scalars_and_rejects_non_dicts():
    assert load_npz_metadata(_NpzLike({})) is None
    assert load_npz_metadata(
        _NpzLike({"metadata_json": np.array(json.dumps({"center": "ningbo"}))})
    ) == {"center": "ningbo"}
    assert load_npz_metadata(_NpzLike({"metadata_json": np.array(b'{"center":"georgia"}')})) == {
        "center": "georgia"
    }
    assert load_npz_metadata(_NpzLike({"metadata_json": np.array("[1, 2]")})) is None
    assert load_npz_metadata(_NpzLike({"metadata_json": np.array("bad")})) is None


def test_write_and_load_pn2021_mmap_cache_round_trip_and_rejects_metadata(tmp_path: Path):
    metadata = build_pn2021_cache_metadata(
        scheme_name="super5",
        center="ningbo",
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
        preprocess_config=build_pn2021_preprocess_config(
            preprocess_mode="minimal_resample",
            norm_mode="per_sample_global",
            apply_filter=False,
            apply_zscore=True,
        ),
        layout="mmap_v1",
    )
    signals = np.zeros((2, 1000, 12), dtype=np.float32)
    labels = np.eye(5, dtype=np.float32)[:2]
    record_ids = np.array(["N1", "N2"], dtype=str)

    write_pn2021_mmap_cache(tmp_path, signals, labels, record_ids, metadata)
    loaded = load_pn2021_mmap_cache(tmp_path, metadata)

    assert loaded is not None
    assert loaded.cache_kind == "mmap"
    assert loaded.cache_hit is True
    np.testing.assert_array_equal(loaded.record_ids, record_ids)
    assert loaded.signals.shape == (2, 1000, 12)
    assert loaded.labels.shape == (2, 5)

    wrong = dict(metadata)
    wrong["center"] = "georgia"
    assert load_pn2021_mmap_cache(tmp_path, wrong) is None


def test_package_load_or_build_pn2021_center_uses_existing_mmap_cache(tmp_path: Path):
    args = SimpleNamespace(
        scheme="super5",
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
        crop_len=250,
        pn2021_cache_dir=str(tmp_path / "npz"),
        pn2021_mmap_cache_dir=str(tmp_path / "mmap"),
    )
    scheme = {
        "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
        "pn2021_fn": lambda snomeds: np.zeros(5, dtype=np.float32),
        "num_classes": 5,
    }
    mmap_root = Path(pn2021_mmap_cache_path(args.pn2021_mmap_cache_dir, "super5", "ningbo"))
    metadata = expected_pn2021_mmap_metadata(args, scheme, "ningbo")
    signals = np.zeros((2, 1000, 12), dtype=np.float32)
    labels = np.eye(5, dtype=np.float32)[:2]
    record_ids = np.asarray(["N1", "N2"], dtype=str)
    write_pn2021_mmap_cache(mmap_root, signals, labels, record_ids, metadata)

    loaded = load_or_build_pn2021_center("ningbo", tmp_path / "missing_center", scheme, args)

    assert loaded[3:] == (0, 0.0, True, "mmap")
    np.testing.assert_array_equal(loaded[2], record_ids)


def test_load_existing_pn2021_eval_cache_upgrades_valid_npz_to_mmap(tmp_path: Path):
    npz_metadata = build_pn2021_cache_metadata(
        scheme_name="super5",
        center="ningbo",
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
        preprocess_config=build_pn2021_preprocess_config(
            preprocess_mode="legacy_ecgfounder_filter",
            norm_mode="per_sample_global",
            apply_filter=True,
            apply_zscore=True,
            include_crop=True,
            crop_len=250,
        ),
    )
    mmap_metadata = build_pn2021_cache_metadata(
        scheme_name="super5",
        center="ningbo",
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
        preprocess_config=build_pn2021_preprocess_config(
            preprocess_mode="legacy_ecgfounder_filter",
            norm_mode="per_sample_global",
            apply_filter=True,
            apply_zscore=True,
            include_crop=False,
        ),
        layout="mmap_v1",
    )
    npz_path = tmp_path / "center.npz"
    mmap_root = tmp_path / "center_mmap"
    signals = np.arange(24, dtype=np.float32).reshape(2, 1, 12)
    labels = np.eye(5, dtype=np.float32)[:2]
    record_ids = np.asarray(["N1", "N2"], dtype=str)
    np.savez_compressed(
        npz_path,
        signals=signals,
        labels=labels,
        record_ids=record_ids,
        metadata_json=json.dumps(npz_metadata, sort_keys=True),
    )

    result = load_existing_pn2021_eval_cache(
        npz_cache_path=npz_path,
        npz_expected_metadata=npz_metadata,
        mmap_cache_root=mmap_root,
        mmap_expected_metadata=mmap_metadata,
    )

    assert result is not None
    assert result.converted_npz_to_mmap is True
    assert result.cache.cache_kind == "mmap"
    assert (mmap_root / "metadata.json").exists()
    np.testing.assert_array_equal(result.cache.record_ids, record_ids)
    np.testing.assert_array_equal(result.cache.labels, labels)
