"""CPU-only tests for signal-level stream datasets and loaders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_adv_gen.training import (
    CachedSignalDataset,
    MemorySignalDataset,
    TaggedCachedSignalDataset,
    TaggedMemorySignalDataset,
    build_weighted_signal_stream_loader,
)


def test_cached_signal_dataset_indexes_and_returns_detached_float32_copy():
    signals = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    labels = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
    ds = CachedSignalDataset(signals, labels, np.asarray([2, 0], dtype=np.int64))

    x, y = ds[0]

    assert len(ds) == 2
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    torch.testing.assert_close(x, torch.from_numpy(signals[2].astype(np.float32)))
    torch.testing.assert_close(y, torch.from_numpy(labels[2].astype(np.float32)))
    x[0, 0] = -999.0
    assert signals[2, 0, 0] != -999.0


def test_memory_and_tagged_signal_datasets_emit_stream_and_teacher_logits():
    signals = np.ones((2, 2, 3), dtype=np.float32)
    labels = np.eye(5, dtype=np.float32)[:2]
    memory = MemorySignalDataset(signals, labels)
    x, y = memory[1]
    torch.testing.assert_close(x, torch.from_numpy(signals[1]))
    torch.testing.assert_close(y, torch.from_numpy(labels[1]))

    tagged_cached = TaggedCachedSignalDataset(
        CachedSignalDataset(signals, labels, np.asarray([0, 1], dtype=np.int64)),
        stream_id=7,
        num_classes=5,
    )
    _, _, stream, teacher = tagged_cached[0]
    assert stream.item() == 7
    torch.testing.assert_close(teacher, torch.zeros(5))

    teacher_logits = np.full_like(labels, 0.25, dtype=np.float32)
    tagged_memory = TaggedMemorySignalDataset(signals, labels, stream_id=2, teacher_logits=teacher_logits)
    _, _, stream, teacher = tagged_memory[1]
    assert stream.item() == 2
    torch.testing.assert_close(teacher, torch.from_numpy(teacher_logits[1]))

    with pytest.raises(ValueError, match="teacher_logits shape"):
        TaggedMemorySignalDataset(signals, labels, stream_id=2, teacher_logits=np.zeros((1, 5), dtype=np.float32))


def test_build_weighted_signal_stream_loader_preserves_weights_and_four_tensors():
    source_x = np.ones((3, 2, 4), dtype=np.float32)
    source_y = np.zeros((3, 5), dtype=np.float32)
    target_x = np.full((2, 2, 4), 2.0, dtype=np.float32)
    target_y = np.zeros((2, 5), dtype=np.float32)
    adv_x = np.full((1, 2, 4), 3.0, dtype=np.float32)
    adv_y = np.zeros((1, 5), dtype=np.float32)
    adv_teacher = np.full((1, 5), 0.75, dtype=np.float32)

    loader = build_weighted_signal_stream_loader(
        source_signals=source_x,
        source_labels=source_y,
        source_indices=np.asarray([0, 2], dtype=np.int64),
        target_signals=target_x,
        target_labels=target_y,
        target_indices=np.asarray([1], dtype=np.int64),
        source_weight=1.5,
        target_real_weight=2.0,
        adv_weight=4.0,
        batch_size=4,
        num_workers=0,
        num_classes=5,
        adv_signals=adv_x,
        adv_labels=adv_y,
        adv_teacher_logits=adv_teacher,
        pin_memory=False,
    )

    assert len(loader.dataset) == 4
    assert [float(w) for w in loader.sampler.weights] == pytest.approx([1.5, 1.5, 2.0, 4.0])
    batch = next(iter(loader))
    assert len(batch) == 4
    x, y, stream, teacher = batch
    assert x.shape[1:] == torch.Size([2, 4])
    assert y.shape[1:] == torch.Size([5])
    assert teacher.shape[1:] == torch.Size([5])
    assert set(stream.tolist()).issubset({0, 1, 2})


def test_build_weighted_signal_stream_loader_rejects_no_active_streams():
    with pytest.raises(RuntimeError, match="no active training streams"):
        build_weighted_signal_stream_loader(
            source_signals=np.ones((1, 2, 4), dtype=np.float32),
            source_labels=np.zeros((1, 5), dtype=np.float32),
            source_indices=np.asarray([0], dtype=np.int64),
            target_signals=np.ones((1, 2, 4), dtype=np.float32),
            target_labels=np.zeros((1, 5), dtype=np.float32),
            target_indices=np.asarray([0], dtype=np.int64),
            source_weight=0.0,
            target_real_weight=0.0,
            adv_weight=0.0,
            batch_size=1,
            num_workers=0,
            num_classes=5,
            pin_memory=False,
        )
