from __future__ import annotations

import pytest
import torch

from ecg_adv_gen.training.losses import target_clean_adv_objective


@pytest.mark.parametrize(
    ("rho", "expected"),
    [(0.0, 2.0), (0.25, 2.5), (0.5, 3.0)],
)
def test_target_clean_adv_objective_uses_declared_fraction(rho: float, expected: float):
    objective, stats = target_clean_adv_objective(
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0, 6.0]),
        rho,
    )

    assert objective.item() == pytest.approx(expected)
    assert stats["target_clean_count"] == 2
    assert stats["target_adv_count"] == 2
    assert stats["target_clean_loss_mean"] == pytest.approx(2.0)
    assert stats["target_adv_loss_mean"] == pytest.approx(4.0)
    assert stats["target_clean_weighted_loss"] == pytest.approx((1.0 - rho) * 2.0)
    assert stats["target_adv_weighted_loss"] == pytest.approx(rho * 4.0)
    assert stats["target_clean_nominal_fraction"] == pytest.approx(1.0 - rho)
    assert stats["target_adv_nominal_fraction"] == pytest.approx(rho)


def test_target_half_fraction_is_source_count_independent_with_measured_contributions():
    results = []
    for source_rows in (1, 100):
        source_losses = torch.ones(source_rows)
        assert source_losses.shape[0] == source_rows
        objective, stats = target_clean_adv_objective(
            torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 2.0]),
            0.5,
        )
        results.append((objective.item(), stats))

    assert results[0] == results[1]
    assert results[0][0] == pytest.approx(2.0)
    assert results[0][1]["target_clean_contribution_fraction"] == pytest.approx(0.5)
    assert results[0][1]["target_adv_contribution_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("rho", [-0.01, 1.01])
def test_target_clean_adv_objective_rejects_invalid_fraction(rho: float):
    with pytest.raises(ValueError, match="rho must be in"):
        target_clean_adv_objective(torch.ones(1), torch.ones(1), rho)


@pytest.mark.parametrize("empty_stream", ["clean", "adv"])
def test_target_clean_adv_objective_rejects_empty_stream(empty_stream: str):
    clean = torch.empty(0) if empty_stream == "clean" else torch.ones(1)
    adv = torch.empty(0) if empty_stream == "adv" else torch.ones(1)
    with pytest.raises(ValueError, match="losses must be nonempty"):
        target_clean_adv_objective(clean, adv, 0.5)
