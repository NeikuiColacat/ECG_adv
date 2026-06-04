from __future__ import annotations

import numpy as np
import pytest

from ecg_adv_gen.data.latent_pools import LatentPoolError, load_synth_pool


def _latents(n: int = 3) -> np.ndarray:
    return np.arange(n * 4 * 128, dtype=np.float64).reshape(n, 4, 128)


def _labels(n: int = 3) -> np.ndarray:
    labels = np.zeros((n, 5), dtype=np.float64)
    labels[:, 0] = 1.0
    return labels


def test_load_synth_pool_resolves_signal_npz_to_latent_sidecar(tmp_path):
    signal_path = tmp_path / "chapman_pool300.npz"
    latent_path = tmp_path / "chapman_pool300.latent.npz"
    signal_path.write_bytes(b"placeholder")
    np.savez(
        latent_path,
        latents=_latents(),
        labels=_labels(),
        center_name=np.array("chapman_shaoxing"),
        source_ids=np.array([0, 1, 3]),
        source_names=np.array(["real", "synthetic"], dtype=object),
        record_ids=np.array(["r001", 22, b"r003"], dtype=object),
    )

    latents, labels, center, source_meta = load_synth_pool(signal_path)

    assert latents.dtype == np.float32
    assert labels.dtype == np.float32
    assert center == "chapman_shaoxing"
    np.testing.assert_array_equal(source_meta["source_ids"], np.array([0, 1, 3]))
    assert source_meta["source_names"] == ["real", "synthetic"]
    np.testing.assert_array_equal(
        source_meta["source_labels"],
        np.array(["real", "synthetic", "source_3"]),
    )
    np.testing.assert_array_equal(
        source_meta["record_ids"],
        np.array(["r001", "22", "r003"]),
    )
    assert source_meta["has_source_metadata"] is True


def test_load_synth_pool_accepts_latent_npz_directly_and_defaults_metadata(tmp_path):
    latent_path = tmp_path / "ningbo_pool.latent.npz"
    np.savez(latent_path, latents=_latents(2), labels=_labels(2))

    latents, labels, center, source_meta = load_synth_pool(latent_path)

    assert latents.shape == (2, 4, 128)
    assert labels.shape == (2, 5)
    assert center == "?"
    np.testing.assert_array_equal(source_meta["source_ids"], np.zeros(2, dtype=np.int64))
    assert source_meta["source_names"] == ["unknown"]
    np.testing.assert_array_equal(
        source_meta["source_labels"],
        np.array(["unknown", "unknown"]),
    )
    assert source_meta["has_source_metadata"] is False


def test_load_synth_pool_rejects_missing_sidecar(tmp_path):
    with pytest.raises(LatentPoolError, match="latent pool .npz not found"):
        load_synth_pool(tmp_path / "missing_signal_pool.npz")


def test_load_synth_pool_validates_required_keys_and_shapes(tmp_path):
    missing_labels = tmp_path / "missing_labels.latent.npz"
    np.savez(missing_labels, latents=_latents())
    with pytest.raises(LatentPoolError, match="missing required key"):
        load_synth_pool(missing_labels)

    bad_shape = tmp_path / "bad_shape.latent.npz"
    np.savez(bad_shape, latents=np.zeros((3, 128, 4)), labels=_labels())
    with pytest.raises(LatentPoolError, match="bad synth latent shape"):
        load_synth_pool(bad_shape)

    bad_label_len = tmp_path / "bad_label_len.latent.npz"
    np.savez(bad_label_len, latents=_latents(3), labels=_labels(2))
    with pytest.raises(LatentPoolError, match="labels length does not match latents"):
        load_synth_pool(bad_label_len)


def test_load_synth_pool_validates_source_and_record_lengths(tmp_path):
    bad_source_len = tmp_path / "bad_source_len.latent.npz"
    np.savez(
        bad_source_len,
        latents=_latents(3),
        labels=_labels(3),
        source_ids=np.array([0, 1]),
        source_names=np.array(["a", "b"]),
    )
    with pytest.raises(LatentPoolError, match="source_ids length does not match latents"):
        load_synth_pool(bad_source_len)

    bad_record_len = tmp_path / "bad_record_len.latent.npz"
    np.savez(
        bad_record_len,
        latents=_latents(3),
        labels=_labels(3),
        record_ids=np.array(["r1", "r2"]),
    )
    with pytest.raises(LatentPoolError, match="record_ids length does not match latents"):
        load_synth_pool(bad_record_len)
