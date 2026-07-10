import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_adv_gen.training import online_buffer
from ecg_adv_gen.training.online_buffer import QualityAwareBuffer


def test_quality_score_evicts_separately_from_sampling_weight():
    buf = QualityAwareBuffer(max_size=2)
    x = torch.zeros(12, 8)
    y = torch.zeros(5)

    buf.add_one(x, y, score=0.90, sample_weight=0.10)
    buf.add_one(x + 1, y, score=0.40, sample_weight=1.00)
    buf.add_one(x + 2, y, score=0.80, sample_weight=1.00)

    assert buf.score_list == [0.90, 0.80]
    assert np.allclose(buf.get_sampling_weights(), [0.09, 0.80])


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.realized_steps = 0

    def step(self, closure=None):
        self.realized_steps += 1
        return super().step(closure)


def _value_loader(values: list[float], batch_size: int = 1) -> DataLoader:
    signals = torch.tensor(values).reshape(-1, 1, 1)
    labels = torch.zeros(len(values), 1)
    return DataLoader(TensorDataset(signals, labels), batch_size=batch_size, shuffle=False)


def test_primary_grouped_target_objective_is_independent_of_source_row_count():
    target_clean = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
    target_adv = target_clean + 0.5
    target_labels = torch.tensor([[0.0], [1.0]])

    def run(source_rows: int) -> tuple[dict, int]:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1))
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
        optimizer = _CountingSGD(model.parameters())
        source_signals = torch.tensor([[[0.25, 0.75]]]).repeat(source_rows, 1, 1)
        source_labels = torch.zeros(source_rows, 1)
        stats = online_buffer.train_one_epoch_grouped_target_bce(
            model=model,
            target_clean_loader=DataLoader(
                TensorDataset(target_clean, target_labels), batch_size=2, shuffle=False
            ),
            target_adv_loader=DataLoader(
                TensorDataset(target_adv, target_labels), batch_size=2, shuffle=False
            ),
            source_loader=DataLoader(
                TensorDataset(source_signals, source_labels),
                batch_size=source_rows,
                shuffle=False,
            ),
            optimizer=optimizer,
            criterion=nn.BCEWithLogitsLoss(reduction="none"),
            device="cpu",
            target_adv_fraction=0.5,
            source_coefficient=1.0,
            target_coefficient=1.0,
            grad_clip=0.0,
            ewa_params=None,
            anchor_lambda=0.0,
            ewa_decay=0.0,
        )
        return stats, optimizer.realized_steps

    one, one_steps = run(1)
    hundred, hundred_steps = run(100)

    assert one_steps == hundred_steps == 1
    assert one["source_count"] == 1
    assert hundred["source_count"] == 100
    for key in (
        "target_objective_loss",
        "target_clean_count",
        "target_adv_count",
        "target_clean_loss_mean",
        "target_adv_loss_mean",
        "target_clean_weighted_loss",
        "target_adv_weighted_loss",
        "target_clean_nominal_fraction",
        "target_adv_nominal_fraction",
        "target_clean_nominal_contribution_fraction",
        "target_adv_nominal_contribution_fraction",
        "target_clean_realized_contribution_fraction",
        "target_adv_realized_contribution_fraction",
    ):
        assert one[key] == pytest.approx(hundred[key])
    assert one["target_clean_count"] == one["target_adv_count"] == 2
    assert one["target_clean_nominal_fraction"] == one["target_adv_nominal_fraction"] == 0.5
    assert one["source_nominal_contribution_fraction"] == 0.5
    assert one["target_clean_nominal_contribution_fraction"] == 0.25
    assert one["target_adv_nominal_contribution_fraction"] == 0.25
    assert one["source_realized_contribution_fraction"] == pytest.approx(0.5)
    assert one["target_clean_realized_contribution_fraction"] == pytest.approx(0.25)
    assert one["target_adv_realized_contribution_fraction"] == pytest.approx(0.25)
    assert sum(
        one[f"{stream}_realized_contribution_fraction"]
        for stream in ("source", "target_clean", "target_adv")
    ) == pytest.approx(1.0)
    assert one["source_weighted_loss"] == pytest.approx(hundred["source_weighted_loss"])
    assert one["source_loss_mean"] == pytest.approx(hundred["source_loss_mean"])


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[float, ...]] = []

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(float(value) for value in signals[:, 0, 0]))
        return self.bias.expand(signals.shape[0], 1)


def test_primary_grouped_target_objective_cycles_auxiliary_groups_to_clean_budget():
    model = _RecordingModel()
    optimizer = _CountingSGD(model.parameters())
    stats = online_buffer.train_one_epoch_grouped_target_bce(
        model=model,
        target_clean_loader=_value_loader([0.0, 1.0, 2.0, 3.0]),
        target_adv_loader=_value_loader([10.0, 11.0]),
        source_loader=_value_loader([20.0, 21.0, 22.0]),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device="cpu",
        target_adv_fraction=0.25,
        source_coefficient=1.0,
        target_coefficient=1.0,
        grad_clip=0.0,
        ewa_params=None,
        anchor_lambda=0.0,
        ewa_decay=0.0,
    )

    assert optimizer.realized_steps == stats["n_batches"] == 4
    assert stats["target_clean_count"] == stats["target_adv_count"] == stats["source_count"] == 4
    assert model.calls[0::3] == [(0.0,), (1.0,), (2.0,), (3.0,)]
    assert model.calls[1::3] == [(10.0,), (11.0,), (10.0,), (11.0,)]
    assert model.calls[2::3] == [(20.0,), (21.0,), (22.0,), (20.0,)]


class _BatchNormRecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(1, momentum=1.0)
        self.calls: list[tuple[float, ...]] = []

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(float(value) for value in signals[:, 0, 0]))
        return self.batch_norm(signals).mean(dim=2)


def test_primary_grouped_rho_zero_skips_adversarial_model_forward():
    model = _BatchNormRecordingModel()
    optimizer = _CountingSGD(model.parameters())
    stats = online_buffer.train_one_epoch_grouped_target_bce(
        model=model,
        target_clean_loader=_value_loader([0.0, 1.0, 2.0, 3.0], 2),
        target_adv_loader=_value_loader([100.0, 101.0], 2),
        source_loader=None,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device="cpu",
        target_adv_fraction=0.0,
        source_coefficient=0.0,
        target_coefficient=1.0,
        grad_clip=0.0,
        ewa_params=None,
        anchor_lambda=0.0,
        ewa_decay=0.0,
    )

    assert optimizer.realized_steps == stats["n_batches"] == 2
    assert model.calls == [(0.0, 1.0), (2.0, 3.0)]
    assert model.batch_norm.running_mean.item() == pytest.approx(2.5)
    assert stats["target_clean_count"] == 4
    assert stats["target_adv_count"] == 0
    assert stats["target_adv_step_count"] == 0
    assert stats["target_adv_loss_mean"] is None
    assert stats["target_adv_weighted_loss"] == 0.0
    assert stats["target_adv_realized_contribution_fraction"] == 0.0


class _InputLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return signals[:, 0, 0].unsqueeze(1) + self.bias * 0.0


def test_primary_grouped_objective_uses_true_step_means_for_uneven_cycled_batches():
    def batch_bce(values: list[float]) -> float:
        logits = torch.tensor(values).reshape(-1, 1)
        labels = torch.zeros_like(logits)
        return float(
            nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
        )

    clean_steps = [[0.0, 2.0], [4.0, 6.0], [8.0]]
    adv_steps = [[1.0, 3.0], [5.0], [1.0, 3.0]]
    source_steps = [[10.0, 12.0, 14.0], [16.0], [10.0, 12.0, 14.0]]
    clean_step_mean = float(np.mean([batch_bce(values) for values in clean_steps]))
    adv_step_mean = float(np.mean([batch_bce(values) for values in adv_steps]))
    source_step_mean = float(np.mean([batch_bce(values) for values in source_steps]))
    clean_sample_mean = batch_bce([value for step in clean_steps for value in step])
    adv_sample_mean = batch_bce([value for step in adv_steps for value in step])
    source_sample_mean = batch_bce([value for step in source_steps for value in step])
    rho = 0.25
    source_coefficient = 1.0
    target_coefficient = 3.0
    coefficient_sum = source_coefficient + target_coefficient
    source_nominal = source_coefficient / coefficient_sum
    clean_nominal = target_coefficient / coefficient_sum * (1.0 - rho)
    adv_nominal = target_coefficient / coefficient_sum * rho
    expected_target = (1.0 - rho) * clean_step_mean + rho * adv_step_mean
    expected_objective = (
        source_nominal * source_step_mean
        + clean_nominal * clean_step_mean
        + adv_nominal * adv_step_mean
    )

    model = _InputLogitModel()
    optimizer = _CountingSGD(model.parameters())
    stats = online_buffer.train_one_epoch_grouped_target_bce(
        model=model,
        target_clean_loader=_value_loader([0.0, 2.0, 4.0, 6.0, 8.0], 2),
        target_adv_loader=_value_loader([1.0, 3.0, 5.0], 2),
        source_loader=_value_loader([10.0, 12.0, 14.0, 16.0], 3),
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device="cpu",
        target_adv_fraction=rho,
        source_coefficient=source_coefficient,
        target_coefficient=target_coefficient,
        grad_clip=0.0,
        ewa_params=None,
        anchor_lambda=0.0,
        ewa_decay=0.0,
    )

    assert optimizer.realized_steps == stats["n_batches"] == 3
    expected_groups = {
        "target_clean": (5, clean_step_mean, clean_sample_mean, clean_nominal),
        "target_adv": (5, adv_step_mean, adv_sample_mean, adv_nominal),
        "source": (7, source_step_mean, source_sample_mean, source_nominal),
    }
    for stream, (count, step_mean, sample_mean, nominal) in expected_groups.items():
        assert stats[f"{stream}_count"] == count
        assert stats[f"{stream}_step_count"] == 3
        assert stats[f"{stream}_loss_mean"] == pytest.approx(step_mean)
        assert stats[f"{stream}_loss_sample_mean"] == pytest.approx(sample_mean)
        assert stats[f"{stream}_loss_mean"] != pytest.approx(sample_mean)
        assert stats[f"{stream}_weighted_loss"] == pytest.approx(nominal * step_mean)
        assert stats[f"{stream}_realized_contribution_fraction"] == pytest.approx(
            nominal * step_mean / expected_objective
        )
    assert stats["target_objective_loss"] == pytest.approx(expected_target)
    assert stats["loss"] == pytest.approx(expected_objective)
    assert stats["objective_loss"] == pytest.approx(expected_objective)
    assert sum(
        stats[f"{stream}_realized_contribution_fraction"]
        for stream in ("source", "target_clean", "target_adv")
    ) == pytest.approx(1.0)
