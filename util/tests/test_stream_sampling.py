"""CPU-only tests for weighted feature stream loaders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_adv_gen.training import (
    WeightedFeatureStream,
    build_weighted_feature_stream_loader,
    multilabel_stream_sample_weights,
)


CLASS_TO_IDX = {"CD": 0, "HYP": 1, "MI": 2, "NORM": 3, "STTC": 4}


def test_multilabel_stream_sample_weights_uses_max_positive_class_weight():
    labels = np.zeros((4, 5), dtype=np.float32)
    labels[0, CLASS_TO_IDX["NORM"]] = 1.0
    labels[1, CLASS_TO_IDX["MI"]] = 1.0
    labels[1, CLASS_TO_IDX["STTC"]] = 1.0
    labels[2, CLASS_TO_IDX["CD"]] = 1.0

    weights = multilabel_stream_sample_weights(
        labels,
        base_weight=2.0,
        class_weights={"MI": 3.0, "STTC": 0.5, "CD": 0.25},
        class_to_idx=CLASS_TO_IDX,
        num_classes=5,
    )

    assert weights == [2.0, 6.0, 0.5, 2.0]


def test_multilabel_stream_sample_weights_validates_inputs():
    labels = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown class"):
        multilabel_stream_sample_weights(
            labels,
            base_weight=1.0,
            class_weights={"BAD": 1.0},
            class_to_idx=CLASS_TO_IDX,
            num_classes=5,
        )
    with pytest.raises(ValueError, match="num_classes"):
        multilabel_stream_sample_weights(
            labels,
            base_weight=1.0,
            class_weights={"NORM": 1.0},
            class_to_idx=CLASS_TO_IDX,
            num_classes=4,
        )
    with pytest.raises(ValueError, match="2D labels"):
        multilabel_stream_sample_weights(
            np.zeros((5,), dtype=np.float32),
            base_weight=1.0,
            class_weights={},
            class_to_idx=CLASS_TO_IDX,
        )


def test_build_weighted_feature_stream_loader_preserves_stream_ids_and_weights():
    source_x = np.ones((2, 3), dtype=np.float32)
    source_y = np.zeros((2, 5), dtype=np.float32)
    target_x = np.full((2, 3), 2.0, dtype=np.float32)
    target_y = np.zeros((2, 5), dtype=np.float32)
    target_y[0, CLASS_TO_IDX["MI"]] = 1.0
    target_y[1, CLASS_TO_IDX["NORM"]] = 1.0
    adv_x = np.full((1, 3), 3.0, dtype=np.float32)
    adv_y = np.zeros((1, 5), dtype=np.float32)
    adv_y[0, CLASS_TO_IDX["STTC"]] = 1.0

    loader = build_weighted_feature_stream_loader(
        [
            WeightedFeatureStream(source_x, source_y, stream_id=0, base_weight=1.0),
            WeightedFeatureStream(
                target_x,
                target_y,
                stream_id=1,
                base_weight=2.0,
                class_weights={"MI": 3.0},
            ),
            WeightedFeatureStream(
                adv_x,
                adv_y,
                stream_id=2,
                base_weight=4.0,
                class_weights={"STTC": 0.5},
            ),
        ],
        batch_size=5,
        class_to_idx=CLASS_TO_IDX,
        num_classes=5,
    )

    assert len(loader.dataset) == 5
    sampler_weights = [float(w) for w in loader.sampler.weights]
    assert sampler_weights == pytest.approx([1.0, 1.0, 6.0, 2.0, 2.0])
    batch_x, batch_y, batch_stream = next(iter(loader))
    assert batch_x.shape[1:] == torch.Size([3])
    assert batch_y.shape[1:] == torch.Size([5])
    assert set(batch_stream.tolist()).issubset({0, 1, 2})


def test_build_weighted_feature_stream_loader_skips_empty_or_zero_weight_streams():
    loader = build_weighted_feature_stream_loader(
        [
            WeightedFeatureStream(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 5), dtype=np.float32),
                stream_id=0,
                base_weight=1.0,
            ),
            WeightedFeatureStream(
                np.ones((2, 3), dtype=np.float32),
                np.zeros((2, 5), dtype=np.float32),
                stream_id=1,
                base_weight=0.0,
            ),
            WeightedFeatureStream(
                np.full((1, 3), 2.0, dtype=np.float32),
                np.zeros((1, 5), dtype=np.float32),
                stream_id=2,
                base_weight=1.0,
            ),
        ],
        batch_size=1,
        class_to_idx=CLASS_TO_IDX,
        num_classes=5,
    )

    assert len(loader.dataset) == 1
    with pytest.raises(ValueError, match="at least one non-empty stream"):
        build_weighted_feature_stream_loader(
            [
                WeightedFeatureStream(
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 5), dtype=np.float32),
                    stream_id=0,
                    base_weight=1.0,
                )
            ],
            batch_size=1,
            class_to_idx=CLASS_TO_IDX,
            num_classes=5,
        )


def test_build_weighted_feature_stream_loader_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        build_weighted_feature_stream_loader(
            [
                WeightedFeatureStream(
                    np.zeros((2, 3), dtype=np.float32),
                    np.zeros((1, 5), dtype=np.float32),
                    stream_id=0,
                    base_weight=1.0,
                )
            ],
            batch_size=1,
            class_to_idx=CLASS_TO_IDX,
            num_classes=5,
        )
