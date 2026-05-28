"""CPU-only tests for torch latent-hull helpers."""

from __future__ import annotations

import pytest
import torch

from ecg_adv_gen.adaptation.latent_hull_torch import initial_hull_latent


def test_initial_hull_latent_one_hot_uses_anchor_candidate():
    z0 = torch.zeros((2, 4, 3), dtype=torch.float32)
    cand = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 3)

    out = initial_hull_latent(
        z0,
        cand,
        weight_mode="one_hot",
        hull_lambda=0.25,
        init_logit_gap=2.0,
    )

    torch.testing.assert_close(out, cand[:, 0] * 0.25)


def test_initial_hull_latent_uniform_matches_candidate_mean():
    z0 = torch.ones((1, 2, 2), dtype=torch.float64)
    cand = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        dtype=torch.float64,
    )

    out = initial_hull_latent(
        z0,
        cand,
        weight_mode="uniform",
        hull_lambda=0.5,
        init_logit_gap=1.0,
    )
    expected = 0.5 * z0 + 0.5 * cand.mean(dim=1)

    assert out.dtype == torch.float64
    torch.testing.assert_close(out, expected)


def test_initial_hull_latent_optimized_is_anchor_dominant_when_gap_positive():
    z0 = torch.zeros((1, 1), dtype=torch.float32)
    cand = torch.tensor([[[10.0], [0.0], [0.0]]], dtype=torch.float32)

    out = initial_hull_latent(
        z0,
        cand,
        weight_mode="optimized",
        hull_lambda=1.0,
        init_logit_gap=2.0,
    )

    assert float(out.item()) > 9.0


def test_initial_hull_latent_unknown_mode_returns_z0_and_validates_shapes():
    z0 = torch.ones((1, 2), dtype=torch.float32)
    cand = torch.zeros((1, 2, 2), dtype=torch.float32)

    assert initial_hull_latent(z0, cand, weight_mode="dirichlet", hull_lambda=0.5, init_logit_gap=1.0) is z0
    with pytest.raises(ValueError, match="candidate tensor"):
        initial_hull_latent(z0, torch.zeros((2,)), weight_mode="one_hot", hull_lambda=0.5, init_logit_gap=1.0)
    with pytest.raises(ValueError, match="does not match"):
        initial_hull_latent(torch.ones((2, 2)), cand, weight_mode="one_hot", hull_lambda=0.5, init_logit_gap=1.0)
