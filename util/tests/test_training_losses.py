"""CPU-only tests for shared training loss helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ecg_adv_gen.training import (
    attack_success_stats,
    compute_pos_weight,
    fullft_adv_batch_diagnostics,
    masked_bce_per_sample,
    masked_bce_with_logits,
    merge_attack_success_stats,
    multilabel_bce_per_sample,
    pairwise_rank_loss,
    stream_weighted_masked_bce,
    summarize_fullft_adv_epoch_diagnostics,
)
from scripts.triple_labels.train_ptbxl import (
    compute_pos_weight as legacy_compute_pos_weight,
)
from scripts.triple_labels.train_ptbxl import (
    masked_bce_with_logits as legacy_masked_bce_with_logits,
)


def test_compute_pos_weight_matches_legacy_and_masks_unknown_labels():
    labels = np.asarray(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    weights = compute_pos_weight(labels, num_classes=4, clip_max=7.0)

    assert weights.dtype == np.float32
    assert weights.tolist() == pytest.approx([1.0, 2.0, 7.0, 2.0])
    np.testing.assert_allclose(weights, legacy_compute_pos_weight(labels, 4, clip_max=7.0))


def test_compute_pos_weight_rejects_invalid_shape_or_class_count():
    with pytest.raises(ValueError, match="2D"):
        compute_pos_weight(np.asarray([1.0, 0.0], dtype=np.float32), num_classes=1)
    with pytest.raises(ValueError, match="exceeds"):
        compute_pos_weight(np.zeros((2, 1), dtype=np.float32), num_classes=2)


def test_masked_bce_with_logits_matches_manual_and_legacy():
    logits = torch.tensor(
        [[0.2, -0.7, 1.3], [1.1, -0.4, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
        dtype=torch.float32,
    )
    pos_weight = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    mask = (labels >= 0).float()
    labels_safe = torch.where(mask.bool(), labels, torch.zeros_like(labels))
    manual = (
        F.binary_cross_entropy_with_logits(
            logits,
            labels_safe,
            pos_weight=pos_weight,
            reduction="none",
        )
        * mask
    ).sum() / mask.sum()

    loss = masked_bce_with_logits(logits, labels, pos_weight)

    assert loss.item() == pytest.approx(manual.item())
    assert loss.item() == pytest.approx(
        legacy_masked_bce_with_logits(logits, labels, pos_weight).item()
    )


def test_masked_bce_with_logits_handles_all_unknown_labels():
    logits = torch.zeros((2, 3), dtype=torch.float32)
    labels = torch.full((2, 3), -1.0, dtype=torch.float32)
    pos_weight = torch.ones((3,), dtype=torch.float32)

    assert masked_bce_with_logits(logits, labels, pos_weight).item() == pytest.approx(0.0)


def test_masked_bce_per_sample_matches_manual_and_handles_all_unknown_rows():
    logits = torch.tensor(
        [[0.2, -0.7, 1.3], [1.1, -0.4, 0.0], [0.5, -0.5, 0.25]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, -1.0, -1.0]],
        dtype=torch.float32,
    )
    pos_weight = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    mask = (labels >= 0).float()
    labels_safe = torch.where(mask.bool(), labels, torch.zeros_like(labels))
    manual = (
        F.binary_cross_entropy_with_logits(
            logits,
            labels_safe,
            pos_weight=pos_weight,
            reduction="none",
        )
        * mask
    ).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    torch.testing.assert_close(masked_bce_per_sample(logits, labels, pos_weight), manual)
    assert masked_bce_per_sample(logits, labels, pos_weight)[2].item() == pytest.approx(0.0)


def test_stream_weighted_masked_bce_matches_manual_stream_weights():
    logits = torch.tensor(
        [[0.2, -0.7], [1.1, -0.4], [0.5, -0.5], [0.0, 0.25]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    stream = torch.tensor([0, 1, 2, 99], dtype=torch.long)
    pos_weight = torch.tensor([2.0, 3.0], dtype=torch.float32)

    per_sample = masked_bce_per_sample(logits, labels, pos_weight)
    weights = torch.tensor([0.5, 2.0, 4.0, 1.0], dtype=torch.float32)
    expected = (per_sample * weights).sum() / weights.sum()

    loss = stream_weighted_masked_bce(
        logits,
        labels,
        stream,
        pos_weight,
        source_weight=0.5,
        target_real_weight=2.0,
        adv_weight=4.0,
    )

    assert loss.item() == pytest.approx(float(expected.item()))


def test_multilabel_bce_per_sample_matches_torch_reduction():
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)

    torch.testing.assert_close(multilabel_bce_per_sample(logits, labels), expected)


def test_attack_success_stats_and_merge_are_sample_weighted():
    labels = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    clean_logits = torch.tensor(
        [[2.0, -2.0], [-1.5, 1.0], [0.5, -0.5]],
        dtype=torch.float32,
    )
    adv_logits = torch.tensor(
        [[0.0, 0.5], [0.5, -0.5], [-0.5, 0.5]],
        dtype=torch.float32,
    )

    stats = attack_success_stats(clean_logits, adv_logits, labels, margin=0.0)
    clean_loss = F.binary_cross_entropy_with_logits(clean_logits, labels, reduction="none").mean(dim=1)
    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, labels, reduction="none").mean(dim=1)
    gain = adv_loss - clean_loss

    assert stats["n"] == 3.0
    assert stats["success_count"] == float((gain > 0.0).float().sum().item())
    assert stats["loss_gain_mean"] == pytest.approx(float(gain.mean().item()))
    assert stats["pos_prob_drop_mean"] > 0.0
    assert stats["neg_prob_rise_mean"] > 0.0

    merged = merge_attack_success_stats([stats, {**stats, "n": 1.0, "success_count": 0.0}])
    assert merged["success_rate"] == pytest.approx(stats["success_count"] / 4.0)
    assert merged["loss_gain_mean"] == pytest.approx(stats["loss_gain_mean"])
    assert merge_attack_success_stats([])["success_rate"] is None


def test_fullft_adv_batch_diagnostics_match_legacy_attack_metrics():
    labels = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    clean_logits = torch.tensor(
        [[2.0, -2.0, 0.25], [-1.0, 1.5, -0.2]],
        dtype=torch.float32,
    )
    init_logits = torch.tensor(
        [[1.0, -1.0, 0.0], [-0.5, 1.0, 0.1]],
        dtype=torch.float32,
    )
    adv_logits = torch.tensor(
        [[-0.2, 0.6, -0.4], [0.4, -0.1, 0.7]],
        dtype=torch.float32,
    )

    stats = fullft_adv_batch_diagnostics(clean_logits, init_logits, adv_logits, labels)
    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, labels, reduction="none").mean(dim=1)
    init_bce = F.binary_cross_entropy_with_logits(init_logits, labels, reduction="none").mean(dim=1)
    adv_bce = F.binary_cross_entropy_with_logits(adv_logits, labels, reduction="none").mean(dim=1)
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    signed = torch.where(pos_mask, torch.ones_like(labels), -torch.ones_like(labels))
    margin_drop = signed * (clean_logits - adv_logits)
    clean_prob = torch.sigmoid(clean_logits)
    adv_prob = torch.sigmoid(adv_logits)
    clean_pos_ok = pos_mask & (clean_prob >= 0.5)
    clean_neg_ok = neg_mask & (clean_prob < 0.5)
    pos_flips = clean_pos_ok & (adv_prob < 0.5)
    neg_flips = clean_neg_ok & (adv_prob >= 0.5)
    sample_ok = (clean_pos_ok | clean_neg_ok).any(dim=1)
    sample_flip = (pos_flips | neg_flips).any(dim=1) & sample_ok

    np.testing.assert_allclose(stats["clean_bce"], clean_bce.numpy(), rtol=1e-6)
    np.testing.assert_allclose(stats["init_bce"], init_bce.numpy(), rtol=1e-6)
    np.testing.assert_allclose(stats["adv_bce"], adv_bce.numpy(), rtol=1e-6)
    np.testing.assert_allclose(stats["loss_gain"], (adv_bce - clean_bce).numpy(), rtol=1e-6)
    np.testing.assert_allclose(stats["init_loss_gain"], (adv_bce - init_bce).numpy(), rtol=1e-6)
    np.testing.assert_allclose(stats["signed_margin_drop"], margin_drop.flatten().numpy(), rtol=1e-6)
    assert stats["pos_correct_count"] == int(clean_pos_ok.sum().item())
    assert stats["pos_hide_count"] == int(pos_flips.sum().item())
    assert stats["neg_correct_count"] == int(clean_neg_ok.sum().item())
    assert stats["neg_add_count"] == int(neg_flips.sum().item())
    assert stats["sample_clean_correct_count"] == int(sample_ok.sum().item())
    assert stats["sample_anyflip_count"] == int(sample_flip.sum().item())


def test_summarize_fullft_adv_epoch_diagnostics_preserves_epoch_fields():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    clean_logits = torch.tensor([[2.0, -2.0], [-1.0, 1.0]], dtype=torch.float32)
    init_logits = torch.tensor([[1.0, -1.0], [-0.5, 0.5]], dtype=torch.float32)
    adv_logits = torch.tensor([[-0.5, 0.5], [0.5, -0.5]], dtype=torch.float32)
    batch = fullft_adv_batch_diagnostics(clean_logits, init_logits, adv_logits, labels)

    summary = summarize_fullft_adv_epoch_diagnostics(
        [batch],
        delta_norms=[1.0, 3.0],
        n_adv=2,
        anchor_sample_stats={"anchor_sample_mode": "hard_bce"},
    )

    assert summary["n_adv"] == 2
    assert summary["delta_mean"] == pytest.approx(2.0)
    assert summary["delta_max"] == pytest.approx(3.0)
    assert summary["clean_bce_mean"] == pytest.approx(float(np.mean(batch["clean_bce"])))
    assert summary["loss_gain_p50"] == pytest.approx(float(np.percentile(batch["loss_gain"], 50)))
    assert summary["loss_gain_p90"] == pytest.approx(float(np.percentile(batch["loss_gain"], 90)))
    assert summary["pos_hide_asr"] == pytest.approx(batch["pos_hide_count"] / batch["pos_correct_count"])
    assert summary["neg_add_asr"] == pytest.approx(batch["neg_add_count"] / batch["neg_correct_count"])
    assert summary["sample_anyflip_asr"] == pytest.approx(
        batch["sample_anyflip_count"] / batch["sample_clean_correct_count"]
    )
    assert summary["anchor_sample_mode"] == "hard_bce"

    empty = summarize_fullft_adv_epoch_diagnostics([], delta_norms=[], n_adv=0)
    assert empty["n_adv"] == 0
    assert empty["delta_mean"] is None
    assert empty["pos_hide_asr"] is None


def test_pairwise_rank_loss_matches_manual_and_returns_zero_when_no_pairs():
    logits = torch.tensor(
        [[2.0, 0.0], [0.5, 1.0], [-1.0, 3.0], [1.0, -0.5]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    temperature = 2.0

    pos = logits[labels[:, 0] > 0.5, 0]
    neg = logits[labels[:, 0] <= 0.0, 0]
    expected_cls0 = F.softplus(-((pos[:, None] - neg[None, :]) / temperature)).mean()
    pos = logits[labels[:, 1] > 0.5, 1]
    neg = logits[labels[:, 1] <= 0.0, 1]
    expected_cls1 = F.softplus(-((pos[:, None] - neg[None, :]) / temperature)).mean()
    expected = (expected_cls0 + 2.0 * expected_cls1) / 3.0

    loss = pairwise_rank_loss(
        logits,
        labels,
        [0, 1],
        class_weights={1: 2.0},
        temperature=temperature,
    )

    assert loss.item() == pytest.approx(float(expected.item()))
    assert pairwise_rank_loss(logits, torch.ones_like(labels), [0, 1], temperature=1.0).item() == 0.0
