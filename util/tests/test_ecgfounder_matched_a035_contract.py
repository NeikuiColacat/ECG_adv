from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from ecg_adv_gen.config import (
    attach_replication_preflight,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)


REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
SOURCE_CONFIG = "configs/defaults/ecgfounder_matched_a035_v1.yaml"
STAGE_CONFIGS = {
    "train": "configs/replications/ecgfounder_matched_train_seed20260531.yaml",
    "clean": "configs/replications/ecgfounder_matched_clean_seed20260531.yaml",
    "s5": "configs/replications/ecgfounder_matched_s5_seed20260531.yaml",
    "depth23": "configs/replications/ecgfounder_matched_depth23_seed20260531.yaml",
}
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
ARMS = ("a0", "a3", "a5")
INDEX = REPO / "configs" / "active_scripts.yaml"


def _load(path: str, run_id: str = "pytest_ecgfounder_matched") -> dict:
    config = load_experiment_config(
        REPO / path,
        LOCAL,
        runtime_context={"run_id": run_id},
    )
    validate_experiment_config(config, repo_root=REPO)
    return config


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _command_key(command: dict) -> tuple[str, str]:
    matrix = command["matrix"]
    case = matrix.get("case") or {}
    return str(matrix["center"]), str(case["arm"])


def _manifest(path: str, run_id: str = "pytest_ecgfounder_binding") -> tuple[dict, dict]:
    config = _load(path, run_id)
    commands = build_runner_commands(config)
    return config, make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=run_id,
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )


def _attach(path: str, run_id: str = "pytest_ecgfounder_binding") -> tuple[dict, dict]:
    config, current = _manifest(path, run_id)
    return config, attach_replication_preflight(
        current,
        config,
        repo_root=REPO,
        index_path=INDEX,
    )


def _replace_option(argv: list[str], name: str, value: str) -> None:
    argv[argv.index(name) + 1] = value


def _remove_option(argv: list[str], name: str) -> None:
    index = argv.index(name)
    del argv[index : index + 2]


def test_matched_ecgfounder_arm_allowlist_and_components_are_exact():
    from ecg_adv_gen.matched_ecgfounder import matched_ecgfounder_arm

    expected = {
        "a0": (False, False, False, False, 0.0),
        "a3": (True, False, False, False, 0.5),
        "a5": (True, True, True, True, 0.5),
    }
    for arm, values in expected.items():
        row = matched_ecgfounder_arm(arm)
        assert (
            row.vae_lhat,
            row.raw_augmix,
            row.augmix_view_bce,
            row.jsd,
            row.target_adv_fraction,
        ) == values

    for forbidden in ("a1", "a2", "a4", "", "historical_unmatched"):
        with pytest.raises(ValueError, match="A0/A3/A5"):
            matched_ecgfounder_arm(forbidden)


def test_matched_target_adv_fraction_is_copy_invariant_and_base_budget_is_fixed():
    from ecg_adv_gen.training.signal_streams import matched_signal_stream_weights

    one_copy = matched_signal_stream_weights(
        source_count=100,
        target_clean_count=20,
        target_adv_count=10,
        source_weight=1.0,
        target_weight=4.0,
        target_adv_fraction=0.5,
    )
    two_copies = matched_signal_stream_weights(
        source_count=100,
        target_clean_count=20,
        target_adv_count=20,
        source_weight=1.0,
        target_weight=4.0,
        target_adv_fraction=0.5,
    )

    assert one_copy["num_samples"] == two_copies["num_samples"] == 120
    assert one_copy["target_adv_mass_fraction"] == pytest.approx(0.5)
    assert two_copies["target_adv_mass_fraction"] == pytest.approx(0.5)
    assert one_copy["target_total_mass"] == pytest.approx(two_copies["target_total_mass"])
    assert one_copy["target_adv_row_weight"] == pytest.approx(
        2.0 * two_copies["target_adv_row_weight"]
    )

    clean = matched_signal_stream_weights(
        source_count=100,
        target_clean_count=20,
        target_adv_count=0,
        source_weight=1.0,
        target_weight=4.0,
        target_adv_fraction=0.0,
    )
    assert clean["num_samples"] == 120
    assert clean["target_adv_mass_fraction"] == 0.0


def test_matched_loader_exposes_fixed_sampler_contract():
    from ecg_adv_gen.training.signal_streams import (
        build_weighted_signal_stream_loader_from_datasets,
    )

    source = TensorDataset(torch.zeros((8, 1)), torch.zeros((8, 1)))
    target = TensorDataset(torch.zeros((4, 1)), torch.zeros((4, 1)))
    loader = build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source,
        target_dataset=target,
        source_weight=1.0,
        target_real_weight=2.0,
        adv_weight=99.0,
        batch_size=3,
        num_workers=0,
        num_classes=1,
        adv_signals=np.zeros((8, 1), dtype=np.float32),
        adv_labels=np.zeros((8, 1), dtype=np.float32),
        target_adv_fraction=0.5,
        fixed_num_samples=12,
        sampler_seed=17,
    )

    assert len(loader.sampler) == 12
    assert loader.stream_sampling_contract["target_adv_mass_fraction"] == pytest.approx(0.5)
    assert loader.stream_sampling_contract["num_samples"] == 12


def test_ecgfounder_runtime_consumes_recorded_selected_checkpoint(tmp_path: Path):
    from ecg_adv_gen.models.ecgfounder_runtime import fullft_model_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"best")
    (run_dir / "last_model.pt").write_bytes(b"last")
    (run_dir / "eval_result.json").write_text(
        json.dumps({"selected_checkpoint": "best_model.pt"}),
        encoding="utf-8",
    )
    assert fullft_model_path(run_dir) == run_dir / "best_model.pt"

    (run_dir / "best_model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="selected checkpoint"):
        fullft_model_path(run_dir)


def test_ecgfounder_runtime_preserves_legacy_last_without_selection_record(tmp_path: Path):
    from ecg_adv_gen.models.ecgfounder_runtime import fullft_model_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"best")
    (run_dir / "last_model.pt").write_bytes(b"last")

    assert fullft_model_path(run_dir) == run_dir / "last_model.pt"


def test_matched_source_and_train_configs_emit_one_source_and_exact_a035_grid():
    source = _load(SOURCE_CONFIG)
    source_commands = build_runner_commands(source)
    assert len(source_commands) == 1
    source_argv = source_commands[0]["argv"]
    assert _option(source_argv, "--stage") == "ptbxl_source"
    assert _option(source_argv, "--matched_contract") == "ecgfounder_matched_a035_v1"
    assert _option(source_argv, "--selection_metric") == "macro_auprc"
    assert _option(source_argv, "--run_name") == "ptbxl_source_fullft"
    assert "--init_model_path" not in source_argv

    train = _load(STAGE_CONFIGS["train"])
    commands = build_runner_commands(train)
    assert len(commands) == 12
    assert Counter(_command_key(command) for command in commands) == Counter(
        (center, arm) for center in CENTERS for arm in ARMS
    )

    source_paths = set()
    for command in commands:
        center, arm = _command_key(command)
        argv = command["argv"]
        assert _option(argv, "--comparison_arm") == arm
        assert _option(argv, "--matched_contract") == "ecgfounder_matched_a035_v1"
        assert _option(argv, "--epochs") == "10"
        assert _option(argv, "--seed") == "20260531"
        assert _option(argv, "--target_real_val_fraction") == "0.2"
        assert _option(argv, "--target_real_val_seed") == "20260531"
        assert _option(argv, "--selection_metric") == "macro_auprc"
        assert _option(argv, "--source_floor_max_drop") == "0.02"
        assert _option(argv, "--run_name") == f"{center}_{arm}"
        assert "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets" in _option(
            argv, "--ref_meta_json"
        )
        source_paths.add(_option(argv, "--init_model_path"))

    assert len(source_paths) == 1
    assert next(iter(source_paths)).endswith(
        "/ecgfounder_matched_a035_v1/pytest_ecgfounder_matched/"
        "runs/ptbxl_source_fullft/best_model.pt"
    )


def test_matched_a0_a3_a5_commands_have_exact_component_topology():
    commands = build_runner_commands(_load(STAGE_CONFIGS["train"]))
    by_arm = {arm: argv for _, arm, argv in (
        (*_command_key(command), command["argv"]) for command in commands[:3]
    )}
    if set(by_arm) != set(ARMS):
        by_arm = {}
        for command in commands:
            _, arm = _command_key(command)
            by_arm.setdefault(arm, command["argv"])

    a0, a3, a5 = (by_arm[arm] for arm in ARMS)
    assert _option(a0, "--target_adv_fraction") == "0.0"
    assert "--enable_vae_adv_stream" not in a0
    assert "--enable_latent_augmix_branch" not in a0

    assert _option(a3, "--target_adv_fraction") == "0.5"
    assert "--enable_vae_adv_stream" in a3
    assert "--enable_latent_augmix_branch" not in a3

    assert _option(a5, "--target_adv_fraction") == "0.5"
    assert "--enable_vae_adv_stream" in a5
    assert "--enable_latent_augmix_branch" in a5
    assert _option(a5, "--latent_augmix_copies") == "2"
    assert _option(a5, "--latent_augmix_bce_weight") == "1.0"
    assert _option(a5, "--latent_augmix_consistency_weight") == "2.0"
    assert _option(a5, "--latent_augmix_consistency_loss") == "jsd"

    for argv in (a3, a5):
        assert _option(argv, "--hull_label_mode") == "exact"
        assert "--hull_include_anchor" not in argv
        assert _option(argv, "--hull_m") == "20"
        assert _option(argv, "--hull_lambda") == "0.6"
        assert _option(argv, "--hull_steps") == "5"
        assert _option(argv, "--hull_init_logit_gap") == "0.0"
        assert _option(argv, "--hull_neighbor_distance_space") == "standardized"
        assert _option(argv, "--hull_neighbor_mode") == "local_random"


def test_matched_consumers_bind_every_producer_run_and_require_best_checkpoint():
    expected = Counter((center, arm) for center in CENTERS for arm in ARMS)
    for stage in ("clean", "s5", "depth23"):
        commands = build_runner_commands(_load(STAGE_CONFIGS[stage]))
        assert Counter(_command_key(command) for command in commands) == expected
        for command in commands:
            center, arm = _command_key(command)
            argv = command["argv"]
            assert _option(argv, "--run_dir").endswith(
                f"/ecgfounder_matched_a035_v1/pytest_ecgfounder_matched/runs/{center}_{arm}"
            )
            assert "--require_selected_checkpoint" in argv
            assert ("--clean_only" in argv) is (stage == "clean")


def test_active_surface_tracks_exact_wrappers_without_changing_latest_golden():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    surface = next(
        item for item in index["replication_surfaces"]
        if item["name"] == "ecgfounder_matched_a035_v1"
    )
    assert surface["source_config"] == SOURCE_CONFIG
    assert surface["command_count_per_stage"] == 12
    assert surface["pipeline_contract"] == {
        "stage_order": ["train", "clean", "s5", "depth23"],
        "producer_stage": "train",
        "consumer_stages": ["clean", "s5", "depth23"],
        "matrix": {"centers": list(CENTERS), "arms": list(ARMS)},
    }
    replicate = surface["replicates"][0]
    assert replicate["seed"] == 20260531
    assert replicate["stages"] == STAGE_CONFIGS

    committed = subprocess.check_output(
        ["git", "show", "3039996:configs/golden/latest_mainline_contract_v1.json"],
        cwd=REPO,
    )
    assert (REPO / "configs" / "golden" / "latest_mainline_contract_v1.json").read_bytes() == committed


def test_matched_k500_split_and_exact_eligibility_never_include_validation_ids():
    from ecg_adv_gen.matched_ecgfounder import build_matched_k500_contract

    record_ids = np.asarray([f"r{i:03d}" for i in range(60)])
    labels = np.zeros((60, 5), dtype=np.float32)
    labels[:30, 3] = 1.0
    labels[30:45, 2] = 1.0
    labels[45:, 4] = 1.0

    first = build_matched_k500_contract(
        record_ids,
        labels,
        val_fraction=0.2,
        seed=20260531,
    )
    second = build_matched_k500_contract(
        record_ids,
        labels,
        val_fraction=0.2,
        seed=20260531,
    )

    assert first == second
    train_ids = set(first["split"]["train_record_ids"])
    val_ids = set(first["split"]["val_record_ids"])
    eligible_ids = set(first["exact_eligibility"]["eligible_record_ids"])
    assert train_ids.isdisjoint(val_ids)
    assert eligible_ids <= train_ids
    assert eligible_ids.isdisjoint(val_ids)
    assert first["exact_eligibility"]["partner_policy"] == "exact_nonself"
    assert first["exact_eligibility"]["min_nonself"] == 2


def test_source_initialization_contract_requires_selected_source_best(tmp_path: Path):
    from ecg_adv_gen.matched_ecgfounder import load_source_checkpoint_identity

    run_dir = tmp_path / "ptbxl_source_fullft"
    run_dir.mkdir()
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"shared-source")
    result = {
        "stage": "ptbxl_source",
        "selected_checkpoint": "best_model.pt",
        "checkpoint_policy": "best_fold9_source_val",
        "selection": {
            "metric": "macro_auprc",
            "selection_data": "ptbxl_fold9",
            "report_only_data": ["ptbxl_fold10"],
            "best_epoch": 3,
        },
    }
    (run_dir / "eval_result.json").write_text(json.dumps(result), encoding="utf-8")

    identity = load_source_checkpoint_identity(checkpoint)
    assert identity["path"] == str(checkpoint)
    assert len(identity["sha256"]) == 64
    assert identity["stage"] == "ptbxl_source"
    assert identity["selected_checkpoint"] == "best_model.pt"
    assert identity["selection_data"] == "ptbxl_fold9"
    assert identity["report_only_data"] == ["ptbxl_fold10"]

    result["stage"] = "k500"
    (run_dir / "eval_result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="ptbxl_source"):
        load_source_checkpoint_identity(checkpoint)


def test_matched_training_record_carries_selection_budget_stream_and_source_identity():
    from ecg_adv_gen.matched_ecgfounder import build_matched_ecgfounder_training_record

    split = {
        "train_record_ids": ["a", "b", "c"],
        "val_record_ids": ["d"],
        "train_record_ids_sha256": "1" * 64,
        "val_record_ids_sha256": "2" * 64,
        "val_fraction": 0.2,
        "seed": 20260531,
    }
    record = build_matched_ecgfounder_training_record(
        comparison_arm="a5",
        source_checkpoint={
            "path": "/runs/source/best_model.pt",
            "sha256": "3" * 64,
            "stage": "ptbxl_source",
            "selected_checkpoint": "best_model.pt",
        },
        split=split,
        exact_eligibility={"eligible_count": 3, "manifest_sha256": "4" * 64},
        selection_metric="macro_auprc",
        source_floor_max_drop=0.02,
        source_floor_result={"source_floor_passed": False, "selected": False},
        epochs=10,
        optimizer_steps_per_epoch=282,
        realized_optimizer_steps=2820,
        actual_param_update_steps=2820,
        scheduler_steps=10,
        realized_stream_counts={"source": 100, "target_clean": 50, "target_adv": 50},
        realized_target_adv_fraction=0.5,
        auxiliary_clean_exposure=[
            {
                "record_ids_sha256": "5" * 64,
                "n_samples": 160,
                "val_overlap_count": 0,
                "selection_seed": 20260531,
                "epoch": 1,
            }
        ],
        view_anchor_exposure=[
            {
                "epoch": 1,
                "record_ids_sha256": "6" * 64,
                "count": 160,
                "unique_count": 150,
                "val_overlap_count": 0,
                "scope": "k500_train_only",
            }
        ],
    )

    assert record["contract"] == "ecgfounder_matched_a035_v1"
    assert record["source_checkpoint"]["sha256"] == "3" * 64
    assert record["source_checkpoint"]["selected_checkpoint"] == "best_model.pt"
    assert record["k500_split"]["train_count"] == 3
    assert record["selection"]["source_floor_result"]["selected"] is False
    assert record["budget"] == {
        "epochs": 10,
        "optimizer_steps_per_epoch": 282,
        "realized_optimizer_steps": 2820,
        "actual_param_update_steps": 2820,
        "scheduler_steps": 10,
    }
    assert record["auxiliary_clean_exposure"][0]["val_overlap_count"] == 0
    assert record["view_anchor_exposure"][0]["scope"] == "k500_train_only"
    assert record["stream"]["realized_target_adv_fraction"] == 0.5


def test_ecgfounder_parser_and_runtime_reject_matched_component_drift(monkeypatch):
    monkeypatch.setenv("ECGTWIN_ROOT", "/home/linbinhao/ECG_adv_data/models/ECGTwin")
    monkeypatch.setenv("ECG_ADV_GEN_DATA_ROOT", "/home/linbinhao/ECG_adv_data")
    monkeypatch.setenv("ECGFOUNDER_ROOT", "/home/linbinhao/ECG_adv_data/ecgfounder")
    from ecg_adv_gen.runner.ecgfounder_fullft import (
        build_arg_parser,
        validate_matched_runtime_args,
    )

    commands = build_runner_commands(_load(STAGE_CONFIGS["train"]))
    a5_command = next(command for command in commands if _command_key(command)[1] == "a5")
    args = build_arg_parser().parse_args(a5_command["argv"][2:])

    row = validate_matched_runtime_args(args)
    assert row.vae_lhat and row.raw_augmix and row.jsd
    assert args.target_adv_fraction == 0.5
    assert args.target_real_val_fraction == 0.2
    assert args.source_floor_max_drop == 0.02

    args.latent_augmix_copies = 1
    with pytest.raises(ValueError, match="copies=2"):
        validate_matched_runtime_args(args)


def test_matched_evaluator_requires_recorded_best_and_exposes_clean_only(tmp_path: Path):
    from ecg_adv_gen.runner.ecgfounder_pn2021c_eval import (
        add_matched_evaluation_args,
        require_selected_checkpoint,
    )

    parser = argparse.ArgumentParser()
    add_matched_evaluation_args(parser)
    args = parser.parse_args(["--clean_only", "--require_selected_checkpoint"])
    assert args.clean_only is True
    assert args.require_selected_checkpoint is True

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"best")
    (run_dir / "last_model.pt").write_bytes(b"last")
    (run_dir / "eval_result.json").write_text(
        json.dumps({"selected_checkpoint": "best_model.pt"}),
        encoding="utf-8",
    )
    assert require_selected_checkpoint(run_dir).name == "best_model.pt"

    (run_dir / "eval_result.json").write_text(
        json.dumps({"selected_checkpoint": "last_model.pt"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="best_model.pt"):
        require_selected_checkpoint(run_dir)


def test_matched_clean_auxiliary_control_updates_parameters_and_optimizer_state(monkeypatch):
    monkeypatch.setenv("ECGTWIN_ROOT", "/home/linbinhao/ECG_adv_data/models/ECGTwin")
    from ecg_adv_gen.runner.ecgfounder_fullft import train_matched_auxiliary_epoch

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    signals = np.linspace(-1.0, 1.0, 320, dtype=np.float32).reshape(160, 2)
    labels = (signals[:, :1] > 0).astype(np.float32)
    stats = train_matched_auxiliary_epoch(
        model=model,
        clean_signals_ct=signals,
        labels_np=labels,
        optimizer=optimizer,
        pos_weight=torch.ones(1),
        device=torch.device("cpu"),
        trainable_params=list(model.parameters()),
        batch_size=64,
        input_transform=lambda value: value,
    )

    assert stats["n_batches"] == 3
    assert stats["control"] == "train_only_clean_bce"
    assert stats["actual_param_update_steps"] == 3
    assert stats["optimizer_parameter_step_advances"] >= 3
    assert any(not torch.equal(actual, expected) for actual, expected in zip(model.parameters(), before))
    assert all(int(optimizer.state[p]["step"].item()) == 3 for p in model.parameters())


def test_matched_auxiliary_clean_exposure_is_identical_and_excludes_internal_val():
    from ecg_adv_gen.runner.ecgfounder_fullft import select_matched_auxiliary_clean_examples

    record_ids = np.asarray([f"r{i:03d}" for i in range(12)])
    signals = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.eye(2, dtype=np.float32)[np.arange(12) % 2]
    train_ids = set(record_ids[:9].tolist())
    val_ids = set(record_ids[9:].tolist())
    arm_rows = [
        select_matched_auxiliary_clean_examples(
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            train_record_ids=train_ids,
            val_record_ids=val_ids,
            n_samples=6,
            seed=20260531,
            epoch=2,
        )
        for _arm in ("a0", "a3", "a5")
    ]
    assert len({row["record_ids_sha256"] for row in arm_rows}) == 1
    assert all(row["n_samples"] == 6 for row in arm_rows)
    assert all(row["val_overlap_count"] == 0 for row in arm_rows)
    assert all(set(row["record_ids"]).isdisjoint(val_ids) for row in arm_rows)


def test_matched_a5_auxiliary_keeps_clean_batch_budget_while_adding_views(monkeypatch):
    monkeypatch.setenv("ECGTWIN_ROOT", "/home/linbinhao/ECG_adv_data/models/ECGTwin")
    from ecg_adv_gen.runner.ecgfounder_fullft import train_matched_auxiliary_epoch

    generator = np.random.default_rng(7)
    clean = generator.normal(size=(160, 2)).astype(np.float32)
    labels = (clean[:, :1] > 0).astype(np.float32)
    views = np.concatenate([clean + 0.1, clean - 0.1], axis=0).astype(np.float32)

    def run(with_views: bool) -> dict:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        return train_matched_auxiliary_epoch(
            model=model,
            clean_signals_ct=clean,
            labels_np=labels,
            optimizer=optimizer,
            pos_weight=torch.ones(1),
            device=torch.device("cpu"),
            trainable_params=list(model.parameters()),
            batch_size=64,
            input_transform=lambda value: value,
            view_clean_signals_ct=clean if with_views else None,
            augmix_signals_ct=views if with_views else None,
            view_labels_np=labels if with_views else None,
            copies=2 if with_views else 0,
            view_bce_weight=1.0 if with_views else 0.0,
            consistency_weight=2.0 if with_views else 0.0,
        )

    clean_stats, a5_stats = run(False), run(True)
    assert clean_stats["n_batches"] == a5_stats["n_batches"] == 3
    assert clean_stats["actual_param_update_steps"] == a5_stats["actual_param_update_steps"] == 3
    assert clean_stats["clean_samples_exposed"] == a5_stats["clean_samples_exposed"] == 160
    assert clean_stats["view_samples_exposed"] == 0
    assert a5_stats["view_samples_exposed"] == 320
    assert a5_stats["view_bce_loss"] is not None
    assert a5_stats["consistency_loss"] is not None


def test_a3_a5_stratified_vae_anchor_ids_are_same_order_and_train_only(monkeypatch):
    monkeypatch.setenv("ECGTWIN_ROOT", "/home/linbinhao/ECG_adv_data/models/ECGTwin")
    from types import SimpleNamespace

    from ecg_adv_gen.adaptation import StratifiedPoolWalker
    from ecg_adv_gen.labels.super5_mapping import SUPER5_TO_IDX
    from ecg_adv_gen.matched_ecgfounder import build_view_anchor_identity
    from ecg_adv_gen.runner.ecgfounder_fullft import sample_anchor_indices

    labels = np.zeros((20, 5), dtype=np.float32)
    labels[np.arange(20), np.arange(20) % 5] = 1.0
    ids = np.asarray([f"train_{index:02d}" for index in range(20)])
    pool = {"labels": labels, "record_ids": ids}
    args = SimpleNamespace(
        anchor_sample_mode="stratified",
        anchor_class_sample_weights="",
        anchor_class_max_repeat=0,
        anchor_sample_min_weight=1e-6,
        anchor_sample_power=1.0,
        k_anchor=10,
        seed=20260531,
        current_epoch=3,
        eval_batch_size=8,
    )

    def picks_for_arm() -> np.ndarray:
        walker = StratifiedPoolWalker(
            labels_one_hot=labels,
            classes_in_scope=list(SUPER5_TO_IDX),
            class_to_idx=SUPER5_TO_IDX,
            seed=20260531,
        )
        picks, _stats = sample_anchor_indices(
            torch.nn.Linear(1, 1),
            {**pool, "classes_in_scope": list(SUPER5_TO_IDX)},
            walker,
            torch.ones(5),
            args,
            torch.device("cpu"),
        )
        return picks

    a3, a5 = picks_for_arm(), picks_for_arm()
    assert a3.tolist() == a5.tolist()
    train_ids = set(ids.tolist())
    val_ids = {"val_00", "val_01"}
    a3_identity = build_view_anchor_identity(
        ids[a3], train_record_ids=train_ids, val_record_ids=val_ids
    )
    a5_identity = build_view_anchor_identity(
        ids[a5], train_record_ids=train_ids, val_record_ids=val_ids
    )
    assert a3_identity == a5_identity
    assert a3_identity["count"] == 10
    assert a3_identity["val_overlap_count"] == 0


def test_matched_eval_metadata_accepts_train_split_but_keeps_all_500_refs(tmp_path: Path):
    from ecg_adv_gen.runner.ecgfounder_pn2021c_eval import _load_result

    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    source_checkpoint = source_dir / "best_model.pt"
    source_checkpoint.write_bytes(b"source")
    source_result = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
    }
    (source_dir / "eval_result.json").write_text(json.dumps(source_result), encoding="utf-8")

    selected = [f"r{i:03d}" for i in range(500)]
    train = selected[:379]
    target_result = {
        "stage": "k500",
        "center": "ningbo",
        "K": 500,
        "target_train_K": 379,
        "selected_ref_record_ids": selected,
        "target_train_record_ids": train,
        "matched_contract": "ecgfounder_matched_a035_v1",
        "k500_split": {
            "train_record_ids": train,
            "val_record_ids": selected[379:],
        },
        "config": {
            "stage": "k500",
            "seed": 20260531,
            "matched_contract": "ecgfounder_matched_a035_v1",
        },
        "init_model": {"path": str(source_checkpoint)},
    }
    (target_dir / "eval_result.json").write_text(json.dumps(target_result), encoding="utf-8")

    assert _load_result(target_dir)["target_train_K"] == 379


def test_matched_manifests_trace_best_selection_and_exact_producer_consumers():
    from ecg_adv_gen.config.replication import (
        audit_replication_producer_consumers,
        audit_replication_source_checkpoint_binding,
    )

    def manifest(path: str) -> dict:
        config = _load(path)
        commands = build_runner_commands(config)
        return make_dry_run_manifest(
            config,
            commands=commands,
            local_paths=validate_experiment_config(config, repo_root=REPO),
            run_id="pytest_ecgfounder_matched",
            cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        )

    source = manifest(SOURCE_CONFIG)
    source_child = source["artifact_trace"]["expected_outputs"]["child_runs"][0]
    assert {item["role"] for item in source_child["expected_artifacts"]} >= {
        "best_model", "last_model", "selection", "eval_result", "external_repo_provenance"
    }

    train = manifest(STAGE_CONFIGS["train"])
    children = train["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(children) == 12
    producer_dirs = {
        (child["matrix"]["center"], child["matrix"]["case"]["arm"]): child["child_run_dir"]
        for child in children
    }
    assert len(producer_dirs) == 12
    source_binding = audit_replication_source_checkpoint_binding(
        source["artifact_trace"]["expected_outputs"]["child_runs"],
        train["commands"],
    )
    assert source_binding["passed"], source_binding["errors"]
    assert source_binding["bound_training_command_count"] == 12
    assert source_binding["source_best_model"].endswith(
        "/runs/ptbxl_source_fullft/best_model.pt"
    )
    for child in children:
        assert {item["role"] for item in child["expected_artifacts"]} >= {
            "best_model", "last_model", "selection", "matched_training_record",
            "eligible_anchor_manifest", "eval_result", "external_repo_provenance",
        }

    consumer_commands = {}
    for stage in ("clean", "s5", "depth23"):
        consumer = manifest(STAGE_CONFIGS[stage])
        consumer_commands[stage] = consumer["commands"]
        for command in consumer["commands"]:
            key = _command_key(command)
            assert _option(command["argv"], "--run_dir") == producer_dirs[key]
        checkpoint_roles = {
            item["role"]
            for item in consumer["artifact_trace"]["inputs"]["checkpoints"]
        }
        assert "command.run_dir.best_model" in checkpoint_roles
        assert "command.run_dir.last_model" not in checkpoint_roles
    binding = audit_replication_producer_consumers(children, consumer_commands)
    assert binding["passed"], binding["errors"]


@pytest.mark.parametrize("stage", ("train", "clean", "s5", "depth23"))
def test_ecgfounder_replication_preflight_binds_each_command_input_exactly(
    stage: str,
):
    _config, current = _attach(
        STAGE_CONFIGS[stage],
        run_id=f"pytest_ecgfounder_binding_{stage}",
    )
    trace = current["artifact_trace"]
    bindings = trace["inputs"]["replication_command_input_bindings"]
    contract = trace["replication_preflight"]["command_input_contract"]

    assert len(current["commands"]) == len(bindings) == 12
    assert len(trace["inputs"]["replication_k500_groups"]) == 4
    assert contract == {
        "passed": True,
        "per_command_passed": True,
        "expected_cell_count": 12,
        "actual_argv_cell_count": 12,
        "manifest_ref_cell_count": 12,
        "attached_group_cell_count": 12,
    }
    for binding in bindings:
        assert binding["runner_family"] == "ecgfounder"
        assert binding["input_index"] == 0
        assert binding["ref_meta_json"] == binding["actual_consumed"]["ref_meta_json"]
        assert binding["ref_meta_json"].endswith(".ref_meta.json")
        if stage == "train":
            expected_latent = binding["arm"] in {"a3", "a5"}
            consumed = binding["actual_consumed"]
            companions = binding["manifest_companions"]
            assert binding["consume_latent"] is expected_latent
            assert consumed["raw1000_npz"] == str(
                Path(binding["anchor_base"]).with_suffix(".raw1000.npz")
            )
            assert companions["signals_npz"] == consumed["raw1000_npz"]
            assert consumed["class_trust"] is None
            assert bool(consumed["latent_npz"]) is expected_latent
            assert companions["latent_npz"] == consumed["latent_npz"]
            if expected_latent:
                assert consumed["latent_npz"] == str(
                    Path(binding["anchor_base"]).with_suffix(".latent.npz")
                )
        else:
            assert binding["consume_latent"] is False
            assert binding["actual_consumed"] == {
                "ref_meta_json": binding["ref_meta_json"],
                "raw1000_npz": None,
                "latent_npz": None,
                "class_trust": None,
            }
            assert binding["manifest_companions"]["signals_npz"].endswith(
                ".signals.npz"
            )
            assert binding["manifest_companions"]["latent_npz"] is None


@pytest.mark.parametrize(
    "drift",
    (
        "foreign_ref_meta",
        "foreign_target_raw1000",
        "foreign_anchor_root",
        "a0_enables_latent",
        "a3_missing_anchor_root",
    ),
)
def test_ecgfounder_replication_rejects_actual_argv_path_or_arm_drift(
    drift: str,
):
    config, current = _manifest(STAGE_CONFIGS["train"], f"pytest_ecg_actual_{drift}")
    commands = current["commands"]
    a0 = next(command for command in commands if command["matrix"]["case"]["arm"] == "a0")
    a3 = next(command for command in commands if command["matrix"]["case"]["arm"] == "a3")

    if drift == "foreign_ref_meta":
        _replace_option(a3["argv"], "--ref_meta_json", "/dev/shm/foreign.ref_meta.json")
    elif drift == "foreign_target_raw1000":
        _replace_option(
            a3["argv"],
            "--target_raw1000_npz_override",
            "/dev/shm/foreign.raw1000.npz",
        )
    elif drift == "foreign_anchor_root":
        _replace_option(a3["argv"], "--anchor_base_root", "/dev/shm/foreign/subsets")
    elif drift == "a0_enables_latent":
        a0["argv"].extend(
            ["--anchor_base_root", str(config["data"]["kshot_subset_root"]), "--enable_vae_adv_stream"]
        )
    else:
        _remove_option(a3["argv"], "--anchor_base_root")

    with pytest.raises(ValueError, match="actual child|canonical|latent|arm"):
        attach_replication_preflight(
            current,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize(
    "drift",
    ("duplicate_identity", "missing_a3_latent", "extra_a0_latent", "wrong_ref_role"),
)
def test_ecgfounder_replication_rejects_manifest_identity_or_artifact_drift(
    drift: str,
):
    config, current = _manifest(STAGE_CONFIGS["train"], f"pytest_ecg_manifest_{drift}")
    refs = current["artifact_trace"]["inputs"]["k500_refs"]
    commands = current["commands"]
    command_arm = {
        index: command["matrix"]["case"]["arm"]
        for index, command in enumerate(commands)
    }
    a0 = next(ref for ref in refs if command_arm[ref["command_index"]] == "a0")
    a3 = next(ref for ref in refs if command_arm[ref["command_index"]] == "a3")
    if drift == "duplicate_identity":
        refs.append(copy.deepcopy(refs[0]))
    elif drift == "missing_a3_latent":
        a3["latent_npz"] = None
    elif drift == "extra_a0_latent":
        a0["latent_npz"] = copy.deepcopy(a3["latent_npz"])
    else:
        a3["ref_meta_json"]["role"] = "forged_ref_meta"

    with pytest.raises(ValueError, match="manifest K500 refs|input cell"):
        attach_replication_preflight(
            current,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_ecgfounder_replication_rejects_foreign_materialization_root():
    config, current = _manifest(STAGE_CONFIGS["train"], "pytest_ecg_foreign_group")
    config["data"]["kshot_subset_root"] = "/dev/shm/foreign_family/subsets"
    with pytest.raises(ValueError, match="canonical input root"):
        attach_replication_preflight(
            current,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize("drift", ("foreign_ref", "duplicate_ref"))
def test_ecgfounder_eval_requires_one_canonical_ref_per_command(drift: str):
    config, current = _manifest(STAGE_CONFIGS["s5"], f"pytest_ecg_eval_{drift}")
    argv = current["commands"][0]["argv"]
    index = argv.index("--exclude_ref_ids")
    if drift == "foreign_ref":
        argv[index + 1] = "/dev/shm/foreign.ref_meta.json"
    else:
        argv.insert(index + 2, argv[index + 1])
    with pytest.raises(ValueError, match="exclude_ref_ids|canonical|count"):
        attach_replication_preflight(
            current,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_ecgfounder_source_only_stays_outside_k500_preflight():
    config, current = _manifest(SOURCE_CONFIG, "pytest_ecg_source_only_binding")
    attached = attach_replication_preflight(
        current,
        config,
        repo_root=REPO,
        index_path=INDEX,
    )
    assert attached["artifact_trace"]["inputs"]["k500_refs"] == []
    assert "replication_preflight" not in attached["artifact_trace"]


def test_active_replication_audit_binds_one_source_best_to_all_training_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import ecg_adv_gen.config.replication as replication_module

    index = yaml.safe_load((REPO / "configs/active_scripts.yaml").read_text(encoding="utf-8"))
    index["replication_surfaces"] = [
        item
        for item in index["replication_surfaces"]
        if item["name"] == "ecgfounder_matched_a035_v1"
    ]
    index["study_surfaces"] = []
    index_path = tmp_path / "active_scripts.yaml"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(replication_module, "_git_tracked", lambda *_args: True)
    monkeypatch.setattr(
        replication_module,
        "verify_replication_k500_groups",
        lambda *_args, **_kwargs: {"passed": True, "missing": [], "identity_errors": []},
    )
    monkeypatch.setattr(
        replication_module,
        "verify_replication_validation_report",
        lambda contract, _groups: {"passed": True, "path": contract["path"], "errors": []},
    )

    report = replication_module.audit_replication_surfaces(
        repo_root=REPO,
        index_path=index_path,
        local_config_path=REPO / "configs/local/linbinhao_server.example.yaml",
    )
    assert report["contract_passed"] is True, report
    source = report["surfaces"][0]["source_config"]
    assert source["config"] == SOURCE_CONFIG
    assert source["command_count"] == 1
    assert source["config_tracked_by_git"] is True
    assert source["checkpoint"] == "best_model.pt"
    seed = report["surfaces"][0]["replicates"][0]
    assert seed["source_checkpoint_binding"]["passed"] is True
    assert seed["source_checkpoint_binding"]["bound_training_command_count"] == 12


def test_indexed_source_config_is_a_runnable_cpu_dry_run_surface():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_experiment.py"),
            "--config",
            str(REPO / SOURCE_CONFIG),
            "--local-config",
            str(LOCAL),
            "--run-id",
            "pytest_ecgfounder_source",
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads(proc.stdout.split("\n\n# Managed commands", 1)[0])
    assert len(manifest["commands"]) == 1
    assert manifest["commands"][0]["matrix"] == {}
    assert "replication_preflight" not in manifest["artifact_trace"]


def test_source_selection_uses_fold9_only_and_matched_training_skips_heldout(monkeypatch):
    monkeypatch.setenv("ECGTWIN_ROOT", "/home/linbinhao/ECG_adv_data/models/ECGTwin")
    from ecg_adv_gen.runner.ecgfounder_fullft import (
        source_fold9_selection_result,
        should_evaluate_heldout_target,
    )

    selected = source_fold9_selection_result(
        best_metric=0.71,
        candidate_fold9_metric=0.73,
        epoch=2,
    )
    assert selected == {
        "epoch": 2,
        "candidate_metric": 0.73,
        "selection_data": "ptbxl_fold9",
        "report_only_data": ["ptbxl_fold10"],
        "source_floor_passed": True,
        "selected": True,
    }
    assert should_evaluate_heldout_target("ecgfounder_matched_a035_v1") is False
    assert should_evaluate_heldout_target("") is True


def test_source_floor_rejection_cannot_select_a_better_target_candidate():
    from ecg_adv_gen.evaluation.selection import update_matched_checkpoint_selection

    result = update_matched_checkpoint_selection(
        best_metric=0.60,
        candidate_metric=0.90,
        source_metric=0.77,
        source_baseline_metric=0.80,
        source_max_drop=0.02,
    )
    assert result["source_floor_threshold"] == pytest.approx(0.78)
    assert result["source_floor_passed"] is False
    assert result["selected"] is False
