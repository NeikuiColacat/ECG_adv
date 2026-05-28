"""CPU-only tests for torch-based anchor sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_adv_gen.adaptation.anchor_sampling import (
    feature_anchor_difficulty_weights,
    parse_anchor_class_weight_string,
    sample_hard_feature_anchors,
    signal_anchor_difficulty_scores,
    signal_anchor_positive_boundary_scores,
    weighted_class_quotas_with_caps,
    weighted_sample_indices,
)


def _tiny_head() -> nn.Linear:
    head = nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.25, 0.25]]))
        head.bias.copy_(torch.tensor([0.0, 0.5]))
    return head


class _TinySignalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))
            self.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def test_parse_anchor_class_weight_string_matches_fullft_cli_semantics():
    assert parse_anchor_class_weight_string("NORM=2;MI=-1, STTC=0.5") == {
        "NORM": 2.0,
        "MI": -1.0,
        "STTC": 0.5,
    }
    assert parse_anchor_class_weight_string("") == {}
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_anchor_class_weight_string("NORM")
    with pytest.raises(ValueError, match="unknown class"):
        parse_anchor_class_weight_string("BAD=1")


def test_weighted_class_quotas_with_caps_preserves_fullft_allocation():
    labels = np.zeros((6, 5), dtype=np.float32)
    labels[:3, 3] = 1.0  # NORM
    labels[3:5, 2] = 1.0  # MI
    labels[5:, 4] = 1.0  # STTC

    quotas = weighted_class_quotas_with_caps(
        ["NORM", "MI", "STTC"],
        5,
        {"NORM": 1.0, "MI": 2.0, "STTC": 1.0},
        labels,
    )
    capped = weighted_class_quotas_with_caps(
        ["NORM", "MI", "STTC"],
        5,
        {"NORM": 1.0, "MI": 2.0, "STTC": 1.0},
        labels,
        max_repeat_per_class=1,
    )

    assert quotas == {"NORM": 1, "MI": 3, "STTC": 1}
    assert capped == {"NORM": 2, "MI": 2, "STTC": 1}
    assert weighted_class_quotas_with_caps(["NORM"], 5, {}, np.zeros((2, 5), dtype=np.float32)) == {"NORM": 0}


def test_weighted_sample_indices_handles_bad_weights_and_replacement():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    indices = np.asarray([10, 11, 12], dtype=np.int64)
    bad_weights = np.asarray([0.0, np.nan, -1.0], dtype=np.float64)

    picks1 = weighted_sample_indices(rng1, indices, bad_weights, 2, replace_when_needed=False)
    picks2 = weighted_sample_indices(rng2, indices, bad_weights, 2, replace_when_needed=False)
    repeated = weighted_sample_indices(
        np.random.default_rng(1),
        indices[:2],
        np.asarray([0.0, 1.0]),
        4,
        replace_when_needed=True,
    )

    assert picks1.tolist() == picks2.tolist()
    assert set(picks1).issubset(set(indices))
    assert repeated.shape == (4,)
    assert set(repeated).issubset({10, 11})
    assert weighted_sample_indices(np.random.default_rng(0), indices, np.ones(3), 0, replace_when_needed=True).size == 0


def test_signal_anchor_difficulty_scores_match_fullft_modes():
    model = _TinySignalModel()
    signals = np.asarray([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]], dtype=np.float32)
    labels = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pos_weight = torch.tensor([1.0, 2.0], dtype=torch.float32)

    hard = signal_anchor_difficulty_scores(
        model,
        signals,
        labels,
        pos_weight,
        batch_size=2,
        device="cpu",
        mode="hard_bce",
    )
    logits = model(torch.from_numpy(signals)).detach()
    expected_hard = F.binary_cross_entropy_with_logits(
        logits,
        torch.from_numpy(labels),
        pos_weight=pos_weight,
        reduction="none",
    ).mean(dim=1).numpy()

    uncertainty = signal_anchor_difficulty_scores(
        model,
        signals,
        labels,
        pos_weight,
        batch_size=2,
        device="cpu",
        mode="uncertainty",
    )
    prob = torch.sigmoid(logits)
    expected_uncertainty = (1.0 - (prob - 0.5).abs() * 2.0).mean(dim=1).numpy()

    np.testing.assert_allclose(hard, expected_hard.astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(uncertainty, expected_uncertainty.astype(np.float32), rtol=1e-6)
    assert signal_anchor_difficulty_scores(
        model,
        signals[:0],
        labels[:0],
        pos_weight,
        batch_size=0,
        device="cpu",
        mode="hard_bce",
    ).size == 0
    with pytest.raises(ValueError, match="unknown anchor difficulty mode"):
        signal_anchor_difficulty_scores(
            model,
            signals,
            labels,
            pos_weight,
            batch_size=2,
            device="cpu",
            mode="bad",
        )


def test_signal_anchor_positive_boundary_scores_are_classwise_and_positive_correct_only():
    model = _TinySignalModel()
    signals = np.asarray([[[2.0, 0.0]], [[0.0, 2.0]], [[-2.0, -2.0]]], dtype=np.float32)
    labels = np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    scores = signal_anchor_positive_boundary_scores(
        model,
        signals,
        labels,
        batch_size=1,
        device="cpu",
    )
    logits = model(torch.from_numpy(signals)).detach()
    prob = torch.sigmoid(logits)
    boundary = 1.0 - (prob - 0.5).abs() * 2.0
    expected = torch.where((torch.from_numpy(labels) > 0.5) & (prob >= 0.5), boundary, torch.zeros_like(boundary))

    np.testing.assert_allclose(scores, expected.numpy().astype(np.float32), rtol=1e-6)
    assert scores.shape == labels.shape
    assert signal_anchor_positive_boundary_scores(
        model,
        signals[:0],
        labels[:0],
        batch_size=0,
        device="cpu",
    ).shape == (0, 2)


def test_feature_anchor_difficulty_weights_hard_bce_matches_manual():
    head = _tiny_head()
    features = np.asarray(
        [[2.0, 0.0, 1.0], [0.5, 0.5, -1.0], [-1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    weights = feature_anchor_difficulty_weights(
        head,
        features,
        labels,
        mode="hard_bce",
        power=1.5,
        min_weight=0.01,
        batch_size=2,
        device="cpu",
    )
    logits = head(torch.from_numpy(features)).detach()
    raw = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels), reduction="none").mean(dim=1).numpy()
    expected = np.power(np.maximum(raw, 0.0) + 1e-12, 1.5) + 0.01
    expected = expected / expected.sum()

    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, expected.astype(np.float32), rtol=1e-6)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_feature_anchor_difficulty_weights_uncertainty_and_validation():
    head = _tiny_head()
    features = np.asarray([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    labels = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    weights = feature_anchor_difficulty_weights(
        head,
        features,
        labels,
        mode="uncertainty",
        power=1.0,
        min_weight=0.0,
        batch_size=1,
        device="cpu",
    )

    assert weights.shape == (2,)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert feature_anchor_difficulty_weights(head, np.empty((0, 3), dtype=np.float32), labels[:0], mode="hard_bce", power=1.0, min_weight=0.0, batch_size=2, device="cpu").size == 0
    with pytest.raises(ValueError, match="length mismatch"):
        feature_anchor_difficulty_weights(head, features, labels[:1], mode="hard_bce", power=1.0, min_weight=0.0, batch_size=2, device="cpu")
    with pytest.raises(ValueError, match="unsupported anchor_sample_mode"):
        feature_anchor_difficulty_weights(head, features, labels, mode="bad", power=1.0, min_weight=0.0, batch_size=2, device="cpu")


def test_sample_hard_feature_anchors_is_deterministic_and_reports_stats():
    head = _tiny_head()
    features = np.asarray(
        [[2.0, 0.0, 1.0], [0.5, 0.5, -1.0], [-1.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    picks1, stats1 = sample_hard_feature_anchors(
        head=head,
        pool_features=features,
        pool_labels=labels,
        k_anchor=3,
        mode="hard_bce",
        power=1.0,
        min_weight=0.05,
        batch_size=2,
        device="cpu",
        seed=123,
    )
    picks2, stats2 = sample_hard_feature_anchors(
        head=head,
        pool_features=features,
        pool_labels=labels,
        k_anchor=3,
        mode="hard_bce",
        power=1.0,
        min_weight=0.05,
        batch_size=2,
        device="cpu",
        seed=123,
    )

    assert picks1.tolist() == picks2.tolist()
    assert len(np.unique(picks1)) == 3
    assert stats1 == stats2
    assert set(stats1) == {"mean_weight", "max_weight", "min_weight", "ess"}
    assert stats1["ess"] > 0.0


def test_sample_hard_feature_anchors_handles_empty_or_zero_k():
    head = _tiny_head()
    empty_x = np.empty((0, 3), dtype=np.float32)
    empty_y = np.empty((0, 2), dtype=np.float32)

    picks, stats = sample_hard_feature_anchors(
        head=head,
        pool_features=empty_x,
        pool_labels=empty_y,
        k_anchor=3,
        mode="hard_bce",
        power=1.0,
        min_weight=0.0,
        batch_size=2,
        device="cpu",
        seed=1,
    )
    assert picks.size == 0
    assert stats == {"mean_weight": 0.0, "max_weight": 0.0}

    picks, stats = sample_hard_feature_anchors(
        head=head,
        pool_features=np.zeros((2, 3), dtype=np.float32),
        pool_labels=np.zeros((2, 2), dtype=np.float32),
        k_anchor=0,
        mode="hard_bce",
        power=1.0,
        min_weight=0.0,
        batch_size=2,
        device="cpu",
        seed=1,
    )
    assert picks.size == 0
    assert stats == {"mean_weight": 0.0, "max_weight": 0.0}
