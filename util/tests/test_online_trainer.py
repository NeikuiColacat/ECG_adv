"""Fast contracts for the retained online-training control plane."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import core.online_trainer as online_trainer
from core.methods import compile_method_profile
from core.online_trainer import (
    load_online_train_config,
    resolve_online_training_parameters,
)


REPO = Path(__file__).resolve().parents[2]
MAINLINE_METHOD = REPO / "configs" / "train" / "methods" / "augmix_simclr_lhat.yaml"


def test_online_config_resolves_only_tracked_bundle_references() -> None:
    config_path = REPO / "configs" / "train" / "PN2021.yaml"
    config = load_online_train_config(config_path)
    resolved, supplied = resolve_online_training_parameters(
        config,
        "efficientnet1dv2",
    )

    assert config.path == config_path.resolve()
    assert config.config_root == (REPO / "configs").resolve()
    assert set(config.references) == {
        "split_config",
        "data_load_config",
        "random_seed_config",
        "source_baseline_registry",
    }
    assert all(path.is_file() for path in config.references.values())
    assert resolved["epochs"] == 23
    assert resolved["scheduler_horizon_epochs"] == 30
    assert resolved["stage1_steps"] == 1024
    assert supplied == {}


def test_online_parameter_overrides_are_closed_and_budget_checked() -> None:
    config = load_online_train_config(REPO / "configs" / "train" / "PN2021.yaml")

    with pytest.raises(ValueError, match="unknown online training parameters"):
        resolve_online_training_parameters(config, "efficientnet1dv2", {"typo": 1})
    with pytest.raises(ValueError, match="greater than or equal"):
        resolve_online_training_parameters(
            config,
            "efficientnet1dv2",
            {"epochs": 4, "scheduler_horizon_epochs": 3},
        )


def test_mainline_rotating_four_covers_all_compositions_in_five_epochs() -> None:
    method = compile_method_profile(MAINLINE_METHOD)
    schedules = [
        online_trainer._method_exposure_steps(method, epoch=epoch)
        for epoch in range(1, 6)
    ]
    compositions = [
        step.composition_index
        for schedule in schedules
        for step in schedule
        if step.name.startswith("corruption_")
    ]

    assert sorted(compositions) == list(range(20))
    assert all(len(schedule) == 6 for schedule in schedules)
    assert all(schedule[0].name == "clean" for schedule in schedules)
    assert all(schedule[0].loss_scale == pytest.approx(0.5) for schedule in schedules)
    assert all(schedule[-1].name == "auxiliary" for schedule in schedules)
    assert all(schedule[-1].loss_scale == pytest.approx(2.0) for schedule in schedules)


def test_exposure_expansion_reuses_one_loaded_batch_and_tags_each_view() -> None:
    method = compile_method_profile(MAINLINE_METHOD)
    steps = online_trainer._method_exposure_steps(method, epoch=1)
    waveform = torch.zeros((2, 1000, 12), dtype=torch.float32)
    reads: list[str] = []

    def loader():
        reads.append("read")
        yield {"waveform": waveform, "hash_id": ("a", "b")}

    batches = list(online_trainer._iter_exposure_batches(loader(), steps))

    assert reads == ["read"]
    assert len(batches) == 6
    assert all(batch["waveform"] is waveform for batch in batches)
    assert [batch["__exposure_index"] for batch in batches] == list(range(6))
    assert [batch["__composition_index"] for batch in batches[1:5]] == [0, 1, 10, 11]
    assert batches[-1]["__objective_terms"] == ("lhat_direct_bce",)


def test_family_balanced_batch_norm_schedule_preserves_declared_weights() -> None:
    weights = (0.5, 0.125, 0.125, 0.125, 0.125)
    momenta = online_trainer._family_balanced_batch_norm_momenta(0.1, weights)

    old_coefficient = 1.0
    contributions: list[float] = []
    for momentum in momenta:
        contributions = [value * (1.0 - momentum) for value in contributions]
        contributions.append(momentum)
        old_coefficient *= 1.0 - momentum

    assert old_coefficient == pytest.approx(0.9)
    assert contributions == pytest.approx([0.1 * weight for weight in weights])


def test_invalid_rotating_epoch_fails_before_any_training_side_effect() -> None:
    method = compile_method_profile(MAINLINE_METHOD)

    with pytest.raises(ValueError, match="positive integer"):
        online_trainer._method_exposure_steps(method, epoch=0)
