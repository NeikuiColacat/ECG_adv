"""CPU-only tests for reusable signal-cache metadata helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_adv_gen.data import (
    build_signal_cache_metadata,
    load_signal_cache,
    signal_cache_shape,
    write_signal_cache_metadata,
)


def test_build_signal_cache_metadata_preserves_fullft_schema_and_defaults():
    items = [
        {
            "label": [1, 0, 0, 0, 1],
            "center": "ningbo",
            "record_id": "N001",
            "strat_fold": 3,
        },
        {"label": np.asarray([0, 1, 0, 0, 0], dtype=np.float64)},
    ]

    metadata = build_signal_cache_metadata(
        items,
        preprocess_policy="minimal_resample",
        input_shape=(12, 1000),
        lead_order=["I", "II", "III"],
        num_classes=5,
    )

    assert set(metadata) == {
        "labels",
        "centers",
        "record_ids",
        "folds",
        "preprocess_policy",
        "input_shape",
        "lead_order",
    }
    assert metadata["labels"].dtype == np.float32
    assert metadata["labels"].shape == (2, 5)
    np.testing.assert_allclose(metadata["labels"][0], np.asarray([1, 0, 0, 0, 1], dtype=np.float32))
    assert metadata["centers"].tolist() == ["ningbo", ""]
    assert metadata["record_ids"].tolist() == ["N001", ""]
    assert metadata["folds"].dtype == np.int64
    assert metadata["folds"].tolist() == [3, -1]
    assert metadata["preprocess_policy"].tolist() == ["minimal_resample"]
    assert metadata["input_shape"].dtype == np.int64
    assert metadata["input_shape"].tolist() == [12, 1000]
    assert metadata["lead_order"].tolist() == ["I", "II", "III"]


def test_build_signal_cache_metadata_rejects_wrong_label_shape():
    with pytest.raises(ValueError, match="label shape"):
        build_signal_cache_metadata(
            [{"label": [1, 0, 0]}],
            preprocess_policy="minimal_resample",
            input_shape=(12, 1000),
            lead_order=["I"],
            num_classes=5,
        )


def test_signal_cache_shape_and_metadata_roundtrip(tmp_path):
    signal_path = tmp_path / "signals.npy"
    meta_path = tmp_path / "signals_meta.npz"
    signals = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4)
    np.save(signal_path, signals)
    metadata = build_signal_cache_metadata(
        [
            {"label": [1, 0, 0, 0, 0], "center": "cpsc_2018", "record_id": "A001"},
            {"label": [0, 0, 1, 0, 0], "center": "georgia", "record_id": "G001"},
        ],
        preprocess_policy="cache_policy",
        input_shape=(12, 4),
        lead_order=["I", "II"],
        num_classes=5,
    )

    write_signal_cache_metadata(meta_path, metadata)
    loaded = load_signal_cache(signal_path, meta_path)

    assert signal_cache_shape(2, (12, 4)) == signals.shape
    np.testing.assert_allclose(loaded["signals"], signals)
    np.testing.assert_allclose(loaded["labels"], metadata["labels"])
    assert loaded["centers"].tolist() == ["cpsc_2018", "georgia"]
    assert loaded["record_ids"].tolist() == ["A001", "G001"]
    assert loaded["preprocess_policy"].tolist() == ["cache_policy"]
