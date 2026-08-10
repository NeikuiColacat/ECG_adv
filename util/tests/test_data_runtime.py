"""Small CPU contracts for the retained runtime data boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data_preprocess.data_runtime import (
    ECGSelection,
    MMapPrefetchConfig,
    assert_disjoint_selections,
    load_data_load_config,
    per_sample_global_zscore,
    prepare_model_input,
)
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER


REPO = Path(__file__).resolve().parents[2]


def _selection(partition: str, hash_ids: tuple[str, ...]) -> ECGSelection:
    count = len(hash_ids)
    return ECGSelection(
        dataset="pn2021",
        cache_dataset="pn2021",
        partition=partition,
        logical_center="ningbo",
        source_centers=("ningbo",),
        indices=np.arange(count, dtype=np.int64),
        hash_ids=np.asarray(hash_ids),
        record_ids=np.asarray([f"record-{index}" for index in range(count)]),
        split_id="fixture",
        split_manifest_path=Path("fixture.json"),
        split_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        hash_id_set_sha256="c" * 64,
        class_order=EXPECTED_CLASS_ORDER,
        mapping_version="v7_super5_sjr_rgq_review_20260528",
        mapping_hash="555ec85d5b51",
    )


def test_canonical_data_load_config_is_the_single_runtime_profile() -> None:
    config_path = REPO / "configs" / "data" / "data_load.yaml"
    config = load_data_load_config(config_path)

    assert config.path == config_path.resolve()
    assert config.cache_mode == "mmap"
    assert config.mmap_access_order == "cache_index"
    assert config.mmap_batch_read is True
    assert config.mmap_prefetch.enabled is True
    assert config.selection_resident is False
    assert config.prepare_for_model is True
    assert config.output_layout == "channel_time"


def test_runtime_transform_applies_augmentation_then_sanitize_zscore_and_layout() -> None:
    source = torch.tensor(
        [[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )
    original = source.clone()
    saw_nonfinite: list[bool] = []

    def augmentation(value: torch.Tensor) -> torch.Tensor:
        saw_nonfinite.append(not bool(torch.isfinite(value).all()))
        return value + 1.0

    transformed = prepare_model_input(source, augmentation=augmentation)

    assert saw_nonfinite == [True]
    assert transformed.shape == (2, 4)
    assert transformed.is_contiguous()
    assert bool(torch.isfinite(transformed).all())
    torch.testing.assert_close(
        transformed.mean(),
        torch.tensor(0.0),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed.std(correction=0),
        torch.tensor(1.0),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(source, original, equal_nan=True)


def test_runtime_transform_rejects_shape_drift_and_unsanitized_nonfinite() -> None:
    source = torch.ones((8, 12), dtype=torch.float32)

    with pytest.raises(ValueError, match="augmentation changed ECG shape"):
        prepare_model_input(source, augmentation=lambda value: value[:-1])

    source[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite values remain"):
        prepare_model_input(source, sanitize=False)


def test_flat_samples_zscore_to_finite_zeros() -> None:
    flat = torch.full((2, 1000, 12), 3.0)
    normalized = per_sample_global_zscore(flat)

    assert bool(torch.isfinite(normalized).all())
    assert torch.count_nonzero(normalized).item() == 0


def test_selection_overlap_is_a_fail_closed_leakage_error() -> None:
    train = _selection("k500_tune_train", ("a", "b"))
    validation = _selection("k500_tune_validation", ("c", "d"))
    assert_disjoint_selections(train, validation)

    leaked = _selection("evaluation_drop_all_zero", ("d", "e"))
    with pytest.raises(ValueError, match="selection leakage"):
        assert_disjoint_selections(validation, leaked)


def test_mmap_prefetch_config_rejects_unbounded_or_unknown_policy() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MMapPrefetchConfig(window_mib=0)
    with pytest.raises(ValueError, match="none or sequential_willneed"):
        MMapPrefetchConfig(advice="aggressive")  # type: ignore[arg-type]
