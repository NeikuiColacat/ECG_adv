"""Fast contracts for the retained online-training control plane."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import core.online_trainer as online_trainer
from core.methods import compile_method_profile
from core.methods.executor import _derive_seed
from core.online_trainer import (
    load_online_train_config,
    resolve_online_training_parameters,
)
from core.train_PN2021 import _validate_locked_source_checkpoint
from models.checkpoints import CheckpointIdentity
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC
from util.random_seed import load_random_seed_config


REPO = Path(__file__).resolve().parents[2]
MAINLINE_METHOD = REPO / "configs" / "train" / "methods" / "augmix_simclr_lhat.yaml"


def test_source_checkpoint_lock_covers_both_backbones(tmp_path: Path) -> None:
    cases = (
        (
            EFFICIENTNET1DV2_SPEC,
            "checkpoint_identity",
            "1" * 64,
            tmp_path / "effnet.pt",
        ),
        (
            ECGFOUNDER_SPEC,
            "task_checkpoint_identity",
            "2" * 64,
            tmp_path / "founder.pt",
        ),
    )
    registry = tmp_path / "source.yaml"
    registry.write_text(
        "schema_version: 1\nmodels:\n"
        + "".join(
            f"  {spec.name}:\n"
            f"    selected_checkpoint: {path}\n"
            f"    selected_checkpoint_sha256: '{sha256}'\n"
            for spec, _, sha256, path in cases
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(references={"source_baseline_registry": registry})
    for spec, attribute, sha256, path in cases:
        identity = CheckpointIdentity(path, sha256, 1, (), ())
        _validate_locked_source_checkpoint(
            SimpleNamespace(**{attribute: identity}), spec, config
        )
        bad = CheckpointIdentity(path, "f" * 64, 1, (), ())
        with pytest.raises(ValueError, match="SHA256"):
            _validate_locked_source_checkpoint(
                SimpleNamespace(**{attribute: bad}), spec, config
            )
        wrong_path = CheckpointIdentity(tmp_path / "wrong.pt", sha256, 1, (), ())
        with pytest.raises(ValueError, match="path"):
            _validate_locked_source_checkpoint(
                SimpleNamespace(**{attribute: wrong_path}), spec, config
            )


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


def test_direct_corruption_seed_uses_semantic_identity() -> None:
    method = compile_method_profile(
        REPO / "configs" / "train" / "methods" / "direct_depth23_fixed20.yaml"
    )
    schedule = online_trainer._method_exposure_steps(method)
    exposure_index = next(
        index
        for index, step in enumerate(schedule)
        if step.composition_index == 7
    )
    assert len(schedule) == 21
    assert schedule[exposure_index].composition_index == 7
    assert method.comparison_rng_identity == "direct_depth23_fixed20"

    config = load_online_train_config(REPO / "configs" / "train" / "PN2021.yaml")
    random_seed = config.payload["random_seed"]
    batch_digest = hashlib.sha256(b"second-a\nsecond-b").hexdigest()
    arguments = {
        "comparison_group": random_seed["comparison_group"],
        "replicate_id": random_seed["replicate_id"],
        "center": "ningbo",
        "model_name": "efficientnet1dv2",
        "epoch": 2,
        "batch_hash_sha256": batch_digest,
        "exposure_name": "corruption_07",
        "composition_index": 7,
    }
    identity = online_trainer._method_rng_identity(**arguments)
    assert all("view_execution_step" not in value for value in identity)
    node = next(
        item for item in method.nodes if item.profile.node_id == "depth23_corruption"
    )
    assert node.profile.params["rng_namespace"] == "method_direct_depth23_fixed20"
    assert _derive_seed(
        load_random_seed_config(config.references["random_seed_config"]).base_seed,
        method.comparison_rng_identity,
        str(node.profile.params["rng_namespace"]),
        node.profile.node_id,
        "composition_and_operators",
        identity,
    ) == 1342437248


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
