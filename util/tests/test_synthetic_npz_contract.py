"""CPU-only tests for synthetic classifier NPZ contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg_adv_gen.data import load_synthetic_npz_arrays, normalize_synthetic_signals


def test_normalize_synthetic_signals_accepts_channels_first_and_time_first():
    channels_first = np.arange(2 * 12 * 1000, dtype=np.float64).reshape(2, 12, 1000)
    time_first = np.arange(2 * 1000 * 12, dtype=np.float64).reshape(2, 1000, 12)

    normalized_channels_first = normalize_synthetic_signals(channels_first)
    normalized_time_first = normalize_synthetic_signals(time_first)

    assert normalized_channels_first.shape == (2, 1000, 12)
    assert normalized_channels_first.dtype == np.float32
    assert normalized_channels_first[0, 0, 1] == channels_first[0, 1, 0]
    assert normalized_time_first.shape == (2, 1000, 12)
    assert normalized_time_first.dtype == np.float32
    assert normalized_time_first[1, 9, 3] == time_first[1, 9, 3]


def test_load_synthetic_npz_arrays_accepts_flat_labels_and_labels5(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(
        first,
        signals=np.ones((2, 12, 1000), dtype=np.float32),
        labels=np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
    )
    np.savez(
        second,
        signals=np.full((1, 1000, 12), 2.0, dtype=np.float32),
        labels5=np.asarray([[0, 0, 1, 0, 0]], dtype=np.float32),
    )

    loaded = load_synthetic_npz_arrays(f"{first},{second}")

    assert loaded.signals.shape == (3, 1000, 12)
    assert loaded.labels.tolist() == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    assert loaded.source_count == 2
    assert loaded.paths == (first, second)


def test_load_synthetic_npz_arrays_accepts_per_center_keys_in_stable_order(tmp_path: Path):
    path = tmp_path / "centers.npz"
    np.savez(
        path,
        ningbo__signals=np.full((1, 1000, 12), 1.0, dtype=np.float32),
        ningbo__labels5=np.asarray([[1, 0, 0, 0, 0]], dtype=np.float32),
        georgia__signals=np.full((2, 12, 1000), 2.0, dtype=np.float32),
        georgia__labels5=np.asarray([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32),
    )

    loaded = load_synthetic_npz_arrays([path])

    assert loaded.signals.shape == (3, 1000, 12)
    assert loaded.signals[:, 0, 0].tolist() == [2.0, 2.0, 1.0]
    assert loaded.labels.tolist() == [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]


def test_load_synthetic_npz_arrays_rejects_missing_labels(tmp_path: Path):
    path = tmp_path / "missing_labels.npz"
    np.savez(path, signals=np.ones((1, 1000, 12), dtype=np.float32))

    with pytest.raises(ValueError, match="missing labels"):
        load_synthetic_npz_arrays(path)


def test_load_synthetic_npz_arrays_rejects_length_mismatch(tmp_path: Path):
    path = tmp_path / "mismatch.npz"
    np.savez(
        path,
        signals=np.ones((2, 1000, 12), dtype=np.float32),
        labels=np.ones((1, 5), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="length mismatch"):
        load_synthetic_npz_arrays(path)


def test_normalize_synthetic_signals_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Expected synth signals"):
        normalize_synthetic_signals(np.ones((3, 999, 12), dtype=np.float32))
