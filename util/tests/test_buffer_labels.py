"""CPU-only tests for adversarial buffer label construction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_adv_gen.adaptation.buffer_labels import build_adv_buffer_label
from scripts.pgd_cross_center.synth_online_at_super5 import push_adv_to_buffer


class _FakeBuffer:
    def __init__(self):
        self.rows = []

    def add_one(self, signal, label, score):
        self.rows.append((signal, label, score))


def test_hard_label_masks_non_target_classes_with_sentinel():
    target = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    label = build_adv_buffer_label(target, label_mode="hard")

    assert label.tolist() == [-1.0, -1.0, 1.0, -1.0, -1.0]


def test_multi_hot_hard_and_latent_soft_preserve_multilabel_targets():
    target = np.asarray([0.2, 0.0, 1.0, 0.0, 0.7], dtype=np.float32)

    hard = build_adv_buffer_label(target, label_mode="multi_hot_hard")
    soft = build_adv_buffer_label(target, label_mode="latent_soft")

    assert hard.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert soft.tolist() == pytest.approx([0.2, 0.0, 1.0, 0.0, 0.7])


def test_teacher_and_mixed_labels_clip_teacher_and_clamp_target_floor():
    target = np.asarray([0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32)
    teacher = np.asarray([-1.0, 0.2, 0.1, 1.2, 0.8], dtype=np.float32)

    teacher_soft = build_adv_buffer_label(
        target,
        label_mode="teacher_soft",
        teacher_probs=teacher,
        soft_target_floor=0.4,
    )
    mixed_soft = build_adv_buffer_label(
        target,
        label_mode="mixed_soft",
        teacher_probs=teacher,
        teacher_mix=0.25,
    )
    latent_mixed = build_adv_buffer_label(
        target,
        label_mode="latent_mixed_teacher",
        teacher_probs=teacher,
        teacher_mix=0.25,
    )

    assert teacher_soft.tolist() == pytest.approx([0.0, 0.2, 0.4, 1.0, 0.8])
    assert mixed_soft.tolist() == pytest.approx([0.0, 0.05, 0.775, 0.25, 0.2])
    assert latent_mixed.tolist() == pytest.approx([0.0, 0.05, 0.775, 0.25, 0.575])


def test_teacher_mix_is_clamped_and_teacher_is_required_for_teacher_modes():
    target = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    teacher = np.asarray([0.8, 0.1, 0.2], dtype=np.float32)

    all_teacher = build_adv_buffer_label(
        target,
        label_mode="mixed_soft",
        teacher_probs=teacher,
        teacher_mix=2.0,
    )
    all_hard = build_adv_buffer_label(
        target,
        label_mode="mixed_soft",
        teacher_probs=teacher,
        teacher_mix=-1.0,
    )

    assert all_teacher.tolist() == pytest.approx([0.8, 0.1, 0.2])
    assert all_hard.tolist() == pytest.approx([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="teacher_probs required"):
        build_adv_buffer_label(target, label_mode="teacher_soft")


def test_label_builder_rejects_unsupported_modes_and_shape_mismatch():
    target = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="teacher_probs required"):
        build_adv_buffer_label(target, label_mode="bad")
    with pytest.raises(ValueError, match="unsupported adv label_mode"):
        build_adv_buffer_label(
            target,
            label_mode="bad",
            teacher_probs=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="same shape"):
        build_adv_buffer_label(
            target,
            label_mode="mixed_soft",
            teacher_probs=np.asarray([0.1, 0.2], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("label_mode", "target", "teacher", "expected"),
    [
        (
            "hard",
            [0.0, 1.0, 0.0, 0.0, 0.0],
            None,
            [-1.0, 1.0, -1.0, -1.0, -1.0],
        ),
        (
            "multi_hot_hard",
            [1.0, 0.0, 0.7, 0.0, 0.0],
            None,
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ),
        (
            "latent_mixed_teacher",
            [0.0, 0.0, 1.0, 0.0, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.8],
            [0.025, 0.05, 0.825, 0.1, 0.575],
        ),
    ],
)
def test_push_adv_to_buffer_preserves_legacy_label_modes(label_mode, target, teacher, expected):
    buffer = _FakeBuffer()
    signals = np.arange(12 * 8, dtype=np.float32).reshape(1, 12, 8)
    target_arr = np.asarray([target], dtype=np.float32)
    logits = np.zeros((1, 5), dtype=np.float32)
    teacher_arr = None if teacher is None else np.asarray([teacher], dtype=np.float32)

    stats = push_adv_to_buffer(
        buffer,
        signals,
        target_arr,
        logits,
        crop_len=4,
        teacher_probs=teacher_arr,
        label_mode=label_mode,
        teacher_mix=0.25,
    )

    assert stats["n_pushed"] == 1
    assert stats["label_mode"] == label_mode
    assert len(buffer.rows) == 1
    signal, label, score = buffer.rows[0]
    assert isinstance(signal, torch.Tensor)
    assert signal.shape == (12, 4)
    assert label.tolist() == pytest.approx(expected)
    assert score == pytest.approx(1.0)
