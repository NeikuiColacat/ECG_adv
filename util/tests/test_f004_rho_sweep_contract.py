"""Mechanical contract for the independent F-004 full-topology rho sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_adv_gen.config import (
    ConfigError,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)
from ecg_adv_gen.evaluation import selection
from ecg_adv_gen.matched_effnet import (
    F004_RHO_SWEEP_PROTOCOL,
    F004_RHO_VALUES,
    F004_VARIANTS,
    MATCHED_EFFNET_ARMS,
    f004_identity,
    validate_f004_runtime,
)
from ecg_adv_gen.training.online_buffer import train_one_epoch_grouped_target_bce
from ecg_adv_gen.training.resume_contract import resume_contract_mismatches


REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"
CANONICAL = REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml"
SMOKE = REPO / "configs/experiments/effnet_f004_rho_sweep_onecenter_smoke.yaml"
TRAIN = REPO / "configs/experiments/effnet_f004_rho_sweep_k500.yaml"
S5 = REPO / "configs/experiments/pn2021c_effnet_f004_rho_sweep_official_s5.yaml"
DEPTH23 = REPO / "configs/experiments/pn2021c_effnet_f004_rho_sweep_depth23.yaml"


def _split() -> dict:
    return {
        "train_record_ids_sha256": "a" * 64,
        "val_record_ids_sha256": "b" * 64,
        "train_record_ids": ["r1"],
        "val_record_ids": ["r2"],
        "val_fraction": 0.5,
        "seed": 7,
    }


@pytest.mark.parametrize("rho", [0.0, 0.25])
def test_generic_fake_a5_rho_matrix_is_rejected_before_command_expansion(rho: float):
    config = load_experiment_config(CANONICAL, LOCAL, runtime_context={"run_id": "fake-a5"})
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "comparison_arm": ["a5"],
        "target_adv_fraction": [rho],
    }

    with pytest.raises(ConfigError, match="canonical matched EffNet"):
        build_runner_commands(config)


@pytest.mark.parametrize("rho", [0.0, 0.25])
def test_canonical_a5_record_rejects_runtime_rho_drift(rho: float):
    with pytest.raises(ValueError, match="canonical matched EffNet arm a5 requires target_adv_fraction=0.5"):
        selection.build_matched_training_record(
            comparison_arm="a5",
            runtime_target_adv_fraction=rho,
            source_checkpoint_path="/data/source.pt",
            source_checkpoint_sha256="c" * 64,
            split=_split(),
            selection_metric="macro_auprc",
            source_floor_max_drop=0.02,
            epochs=2,
            optimizer_steps_per_epoch=3,
            realized_optimizer_steps=6,
            scheduler_steps=2,
            source_floor_result={"source_floor_passed": True},
        )


def _load(path: Path, run_id: str = "pytest-f004") -> dict:
    return load_experiment_config(path, LOCAL, runtime_context={"run_id": run_id})


def _option(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _manifest(path: Path, run_id: str = "pytest-f004") -> tuple[dict, list[dict]]:
    config = _load(path, run_id)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=run_id,
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    return manifest, commands


def test_f004_is_independent_from_canonical_arms_and_has_stable_identity():
    assert F004_RHO_SWEEP_PROTOCOL not in MATCHED_EFFNET_ARMS
    assert F004_RHO_VALUES == (0.0, 0.25, 0.5)
    assert F004_VARIANTS == ("f004_rho0", "f004_rho0p25", "f004_rho0p5")
    assert [f004_identity(rho)["variant"] for rho in F004_RHO_VALUES] == list(F004_VARIANTS)
    assert all(f004_identity(rho)["canonical_arm"] is None for rho in F004_RHO_VALUES)


@pytest.mark.parametrize(
    ("path", "centers", "count"),
    [(SMOKE, ["cpsc_2018"], 3), (TRAIN, None, 12), (S5, None, 12), (DEPTH23, None, 12)],
)
def test_f004_managed_matrices_are_exact_center_by_rho(path: Path, centers, count: int):
    config = _load(path)
    matrix = config["runner"]["matrix"]
    assert set(matrix) == {"center", "rho"}
    assert matrix["rho"] == list(F004_RHO_VALUES)
    if centers is not None:
        assert matrix["center"] == centers
    commands = build_runner_commands(config)
    assert len(commands) == count
    assert len({(item["matrix"]["center"], item["matrix"]["rho"]) for item in commands}) == count


@pytest.mark.parametrize(
    "matrix",
    [
        {"center": ["cpsc_2018"], "rho": [0.0, 0.25, 0.5], "arm": ["a5"]},
        {"center": ["cpsc_2018"], "rho": [0.0, 0.5]},
        {"center": ["cpsc_2018"], "rho": [0.5, 0.25, 0.0]},
        {"center": ["cpsc_2018", "cpsc_2018"], "rho": [0.0, 0.25, 0.5]},
    ],
)
def test_f004_matrix_rejects_extra_missing_reordered_and_duplicate_axes(matrix: dict):
    config = _load(SMOKE)
    config["runner"]["matrix"] = matrix
    with pytest.raises(ConfigError, match="F-004"):
        build_runner_commands(config)


def test_f004_train_commands_freeze_full_topology_and_only_vary_rho_within_center():
    commands = [item for item in build_runner_commands(_load(TRAIN)) if item["matrix"]["center"] == "cpsc_2018"]
    assert len(commands) == 3
    normalized = []
    for command, rho, variant in zip(commands, F004_RHO_VALUES, F004_VARIANTS):
        argv = command["argv"]
        assert _option(argv, "--comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
        assert _option(argv, "--comparison_variant") == variant
        assert _option(argv, "--target_adv_fraction") == str(rho)
        assert _option(argv, "--comparison_arm") == "historical_unmatched"
        assert "--enable_vae_lhat" in argv and "--enable_raw_augmix" in argv
        assert "--enable_latent_augmix_consistency" in argv
        assert float(_option(argv, "--latent_augmix_bce_weight")) > 0
        assert float(_option(argv, "--latent_augmix_consistency_weight")) > 0
        assert _option(argv, "--latent_augmix_third_chain_role") == "vae_lhat_adversarial_waveform"
        stripped = list(argv)
        for option in ("--comparison_variant", "--target_adv_fraction"):
            index = stripped.index(option)
            del stripped[index:index + 2]
        normalized.append(stripped)
    assert normalized[0] == normalized[1] == normalized[2]


def test_f004_manifest_paths_are_unique_short_leaves_and_use_best_checkpoint():
    manifest, _ = _manifest(TRAIN)
    children = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(children) == 12
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        selected = [child for child in children if child["center"] == center]
        assert {Path(child["child_run_dir"]).name for child in selected} == set(F004_VARIANTS)
        assert all(f"/runs/{center}/" in child["child_run_dir"] for child in selected)
        assert all(any(row["role"] == "best_model" for row in child["expected_artifacts"]) for child in selected)
        assert {child["comparison_identity"]["rho"] for child in selected} == set(F004_RHO_VALUES)


def test_f004_wrapper_propagates_identity_to_child_command():
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )

    command = next(
        item for item in build_runner_commands(_load(TRAIN))
        if item["matrix"] == {"center": "cpsc_2018", "rho": 0.25}
    )
    args = wrapper.parse_args(command["argv"][2:])
    paths = resolve_effnet_vae_lhat_paths(
        args, data_root=Path(args.data_root), out_root=Path(args.out_root)
    )
    child = build_effnet_vae_lhat_train_cmd(
        args,
        python="python",
        data_root=Path(args.data_root),
        paths=paths,
        class_trust=Path("/dev/shm/f004-class-trust.json"),
    )
    assert _option(child, "--comparison_protocol") == F004_RHO_SWEEP_PROTOCOL
    assert _option(child, "--comparison_variant") == "f004_rho0p25"
    assert _option(child, "--comparison_topology_version") == "matched_effnet_a0_a2_a3_a4_a5_v2"
    assert _option(child, "--target_adv_fraction") == "0.25"
    assert Path(_option(child, "--output_dir")).name == "f004_rho0p25"


@pytest.mark.parametrize("eval_path", [S5, DEPTH23])
def test_f004_eval_consumers_match_exact_train_producers(eval_path: Path):
    train_manifest, _ = _manifest(TRAIN)
    eval_manifest, eval_commands = _manifest(eval_path)
    producers = {
        (child["center"], child["comparison_identity"]["rho"]): child["child_run_dir"]
        for child in train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    }
    consumers = {
        (command["matrix"]["center"], command["matrix"]["rho"]): _option(command["argv"], "--model_dir")
        for command in eval_commands
    }
    assert consumers == producers
    assert all(_option(command["argv"], "--checkpoint_name") == "best_model.pt" for command in eval_commands)
    assert len(eval_manifest["artifact_trace"]["expected_outputs"]["child_runs"]) == 12


def test_f004_runtime_rejects_topology_drift_and_arm_masquerading():
    base = dict(
        comparison_protocol=F004_RHO_SWEEP_PROTOCOL,
        comparison_arm="historical_unmatched",
        comparison_variant="f004_rho0p25",
        target_adv_fraction=0.25,
        enable_vae_lhat=True,
        enable_raw_augmix=True,
        enable_auxiliary_steps=True,
        bce_weight=1.0,
        jsd_weight=2.0,
        third_chain_route="vae_lhat_adversarial_waveform",
        latent_augmix_signal_space="raw_pre_zscore",
    )
    assert validate_f004_runtime(**base)["rho"] == 0.25
    for key, value in [("comparison_arm", "a5"), ("enable_raw_augmix", False), ("jsd_weight", 0.0)]:
        drifted = {**base, key: value}
        with pytest.raises(ValueError, match="F-004"):
            validate_f004_runtime(**drifted)


def test_f004_record_reports_actual_rho_and_preserves_source_floor_and_budget():
    record = selection.build_f004_training_record(
        target_adv_fraction=0.25,
        source_checkpoint_path="/data/source.pt",
        source_checkpoint_sha256="c" * 64,
        split=_split(),
        selection_metric="macro_auprc",
        source_floor_max_drop=0.02,
        epochs=30,
        optimizer_steps_per_epoch=7,
        realized_optimizer_steps=210,
        scheduler_steps=30,
        source_floor_result={"source_floor_passed": True},
    )
    assert record["comparison_arm"] is None
    assert record["comparison_identity"] == f004_identity(0.25)
    assert record["target_adv_fraction"] == 0.25
    assert record["selection"]["checkpoint"] == "best_model.pt"
    assert record["selection"]["source_floor_result"]["source_floor_passed"] is True
    assert record["budget"]["realized_optimizer_steps"] == 210


def test_f004_resume_contract_rejects_protocol_variant_topology_and_rho_drift():
    current = {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        "comparison_variant": "f004_rho0p25",
        "comparison_topology_version": "matched_effnet_a0_a2_a3_a4_a5_v2",
        "target_adv_fraction": 0.25,
    }
    for key, value in [
        ("comparison_protocol", "other"),
        ("comparison_variant", "f004_rho0p5"),
        ("comparison_topology_version", "other"),
        ("target_adv_fraction", 0.5),
    ]:
        saved = {**current, key: value}
        assert [row["key"] for row in resume_contract_mismatches(saved, current)] == [key]


def test_realized_within_target_fractions_exclude_source_objective():
    clean = TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [1.0]]))
    adv = TensorDataset(torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0], [1.0]]))
    source = TensorDataset(torch.tensor([[2.0], [2.0]]), torch.tensor([[1.0], [1.0]]))
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stats = train_one_epoch_grouped_target_bce(
        model,
        DataLoader(clean, batch_size=2),
        DataLoader(adv, batch_size=2),
        optimizer,
        nn.BCEWithLogitsLoss(reduction="none"),
        "cpu",
        target_adv_fraction=0.25,
        source_loader=DataLoader(source, batch_size=2),
        source_coefficient=3.0,
        target_coefficient=1.0,
        grad_clip=0.0,
        ewa_params=None,
        anchor_lambda=0.0,
        ewa_decay=0.0,
    )
    assert stats["target_clean_realized_within_target_fraction"] + stats["target_adv_realized_within_target_fraction"] == pytest.approx(1.0)
    assert stats["target_adv_realized_within_target_fraction"] != pytest.approx(stats["target_adv_realized_contribution_fraction"])


def test_target_center_filter_retags_source_floor_row_without_duplication(tmp_path: Path):
    from ecg_adv_gen.reporting.metrics_export import _filter_rows_to_target_center

    rows = [
        {"dataset": "ptbxl", "view": "ptbxl_fold10_source_floor", "scope": "dataset", "center": "", "metric": "macro_auprc"},
        {"dataset": "pn2021", "view": "pn2021_all_zero_kept_refexcluded", "scope": "center", "center": "ningbo", "metric": "macro_auprc"},
        {"dataset": "pn2021", "view": "pn2021_all_zero_kept_refexcluded", "scope": "center", "center": "georgia", "metric": "macro_auprc"},
    ]
    filtered = _filter_rows_to_target_center(rows, path=tmp_path / "f004_rho0p25" / "eval.json", target_center="ningbo")
    assert len(filtered) == 2
    source = [row for row in filtered if row["dataset"] == "ptbxl"]
    assert len(source) == 1
    assert source[0]["center"] == "ningbo" and source[0]["scope"] == "center"
