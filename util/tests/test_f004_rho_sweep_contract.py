"""Mechanical contract for the independent F-004 full-topology rho sweep."""

from __future__ import annotations

import argparse
import csv
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from ecg_adv_gen.config import (
    ConfigError,
    build_postprocess_commands,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)
from ecg_adv_gen.evaluation import selection
from ecg_adv_gen.matched_effnet import (
    F004_FROZEN_TOPOLOGY,
    F004_RHO_SWEEP_PROTOCOL,
    F004_RHO_VALUES,
    F004_VARIANTS,
    MATCHED_EFFNET_ARMS,
    f004_identity,
    validate_f004_runtime,
)
from ecg_adv_gen.training.online_buffer import train_one_epoch_grouped_target_bce
from ecg_adv_gen.training.resume_contract import RESUME_CONTRACT_KEYS, resume_contract_mismatches


REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"
CANONICAL = REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml"
CLEAN = REPO / "configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml"
SMOKE = REPO / "configs/studies/effnet_f004_rho_sweep_onecenter_smoke.yaml"
TRAIN = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
S5 = REPO / "configs/studies/pn2021c_effnet_f004_rho_sweep_official_s5.yaml"
DEPTH23 = REPO / "configs/studies/pn2021c_effnet_f004_rho_sweep_depth23.yaml"
F004_DEFAULT = REPO / "configs/defaults/effnet_f004_rho_sweep_locked.yaml"
F004_EXPECTED_KSHOT_SEED = 20260601
F004_EXPECTED_KSHOT_ROOT_FAMILY = (
    "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets"
)
F004_REVIEWER_MISSING_SCIENTIFIC_FIELDS = (
    "adv_soft_target_floor",
    "anchor_class_missing_weight",
    "anchor_class_weight_cap",
    "anchor_class_weight_gamma",
    "anchor_class_weight_min",
    "anchor_class_weight_mode",
    "anchor_class_weight_reference_source",
    "anchor_class_weights",
    "boundary_prob_max",
    "boundary_prob_min",
    "einthoven_p95_max",
    "grad_clip",
    "hull_dirichlet_alpha",
    "hull_weight_mode",
)


def test_f004_contract_has_package_owned_projection_module():
    assert importlib.util.find_spec("ecg_adv_gen.f004_contract") is not None


def test_f004_contract_exposes_single_projection_api():
    contract = importlib.import_module("ecg_adv_gen.f004_contract")
    assert {
        "F004_FROZEN_PROJECTION",
        "F004_FROZEN_PROJECTION_SHA256",
        "F004_EXEMPT_CHILD_OPTIONS",
        "project_f004_child_argv",
        "project_f004_runtime_args",
        "validate_f004_parser_destinations",
        "validate_f004_projection",
    } <= set(dir(contract))


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


def _f004_child_command(rho: float = 0.25) -> list[str]:
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )

    command = next(
        item
        for item in build_runner_commands(_load(TRAIN, "pytest-f004-projection"))
        if item["matrix"] == {"center": "cpsc_2018", "rho": rho}
    )
    args = wrapper.parse_args(command["argv"][2:])
    paths = resolve_effnet_vae_lhat_paths(
        args, data_root=Path(args.data_root), out_root=Path(args.out_root)
    )
    return build_effnet_vae_lhat_train_cmd(
        args,
        python="python",
        data_root=Path(args.data_root),
        paths=paths,
        class_trust=Path("/dev/shm/f004-projection-class-trust.json"),
    )


def _child_option_names(argv: list[str]) -> set[str]:
    return {str(token)[2:] for token in argv[3:] if str(token).startswith("--")}


def _set_nested(config: dict, dotted: str, value) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def test_f004_projection_covers_every_actual_child_behavior_option():
    from ecg_adv_gen.f004_contract import (
        F004_EXEMPT_CHILD_OPTIONS,
        F004_FROZEN_PROJECTION,
        project_f004_child_argv,
    )

    assert F004_EXEMPT_CHILD_OPTIONS == {
        "allow_resume_config_drift",
        "center_name",
        "class_trust",
        "comparison_topology_sha256",
        "comparison_topology_version",
        "comparison_variant",
        "device",
        "init_ckpt",
        "mechanism_variant",
        "num_workers",
        "output_dir",
        "ptbxl_csv",
        "ptbxl_prep",
        "ptbxl_raw",
        "ref_meta_json",
        "resume",
        "synth_npz",
        "study_scope",
        "target_adv_fraction",
        "target_real_npz",
    }
    child = _f004_child_command()
    actual = project_f004_child_argv(child)
    child_behavior = _child_option_names(child) - F004_EXEMPT_CHILD_OPTIONS

    assert actual == dict(F004_FROZEN_PROJECTION)
    assert child_behavior <= set(actual)
    assert set(actual) - child_behavior == {
        "final_checkpoint_only",
        "hull_include_anchor",
        "latent_augmix_signal_space",
    }
    assert {
        "K_anchor",
        "latent_augmix_chain_weights",
        "latent_augmix_consistency_max_batches",
        "weight_decay",
        "anchor_lambda",
        "hull_label_positive",
        "hull_label_negative_floor",
        *F004_REVIEWER_MISSING_SCIENTIFIC_FIELDS,
    } <= set(actual)


def test_f004_parser_destination_classifier_fails_closed_for_new_fields():
    from ecg_adv_gen.f004_contract import (
        F004_EXEMPT_CHILD_OPTIONS,
        F004_FROZEN_PROJECTION,
        validate_f004_parser_destinations,
    )

    classified = {
        *(set(F004_FROZEN_PROJECTION) - {"latent_augmix_signal_space"}),
        *F004_EXEMPT_CHILD_OPTIONS,
    }
    validate_f004_parser_destinations(classified)
    with pytest.raises(ValueError, match="unclassified.*future_scientific_knob"):
        validate_f004_parser_destinations({*classified, "future_scientific_knob"})


@pytest.mark.parametrize("field", F004_REVIEWER_MISSING_SCIENTIFIC_FIELDS)
def test_f004_direct_child_parser_rejects_reviewer_field_tamper(
    field: str, capsys: pytest.CaptureFixture[str]
):
    from ecg_adv_gen.f004_contract import F004_FROZEN_PROJECTION
    from ecg_adv_gen.runner import synth_online_at_super5 as child_runner

    argv = _f004_child_command()[3:]
    option = f"--{field}"
    assert option in argv
    index = argv.index(option)
    tampered = list(argv)
    alternate = {
        "anchor_class_weight_mode": "inv_freq_kshot",
        "hull_weight_mode": "uniform",
    }.get(field, _drift_value(F004_FROZEN_PROJECTION[field]))
    tampered[index + 1] = str(alternate)

    with pytest.raises(SystemExit):
        child_runner.parse_args(tampered)
    error = capsys.readouterr().err
    assert "F-004 frozen projection" in error
    assert field in error


def test_f004_projection_is_rho_invariant_and_sha_covers_all_fields():
    import hashlib

    from ecg_adv_gen.f004_contract import (
        F004_FROZEN_PROJECTION,
        F004_FROZEN_PROJECTION_SHA256,
        project_f004_child_argv,
    )

    projections = [
        project_f004_child_argv(_f004_child_command(rho))
        for rho in F004_RHO_VALUES
    ]
    assert projections[0] == projections[1] == projections[2]
    rendered = json.dumps(
        dict(F004_FROZEN_PROJECTION), sort_keys=True, separators=(",", ":")
    )
    assert F004_FROZEN_PROJECTION_SHA256 == hashlib.sha256(
        rendered.encode()
    ).hexdigest()


def test_f004_projection_rejects_every_field_drift():
    from ecg_adv_gen.f004_contract import (
        F004_FROZEN_PROJECTION,
        validate_f004_projection,
    )

    assert F004_FROZEN_PROJECTION
    for field, value in F004_FROZEN_PROJECTION.items():
        drifted = dict(F004_FROZEN_PROJECTION)
        drifted[field] = _drift_value(value)
        with pytest.raises(ValueError, match=f"F-004 frozen projection.*{field}"):
            validate_f004_projection(drifted, source="pytest")


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("adaptation.anchors.k_anchor", 299, "K_anchor"),
        ("adaptation.latent_augmix.chain_weights", "1,0,0", "latent_augmix_chain_weights"),
        ("adaptation.latent_augmix.consistency.max_batches", 1, "latent_augmix_consistency_max_batches"),
        ("training.optimizer.weight_decay", 0.0002, "weight_decay"),
        ("adaptation.hull.label_lambda_y", 0.3, "hull_label_lambda_y"),
        ("adaptation.attack.pgd_batch", 16, "pgd_batch"),
        ("adaptation.loss.target_real_weight", 81.0, "target_real_weight"),
        ("adaptation.loss.adv_weight", 0.4, "adv_weight"),
        ("adaptation.loss.adv_weight_warmup_epochs", 9, "adv_weight_warmup_epochs"),
        ("adaptation.loss.label_mode", "hard", "adv_label_mode"),
        ("adaptation.loss.teacher_mix", 0.2, "adv_teacher_mix"),
        ("adaptation.loss.ptbxl_weight", 0.1, "ptbxl_weight"),
        ("adaptation.loss.vae_adv_stream_sample_scale", 0.9, "vae_adv_stream_sample_scale"),
        ("adaptation.loss.vae_adv_consistency_weight", 0.1, "vae_adv_consistency_weight"),
        ("training.batch_size", 64, "batch_size"),
        ("training.epochs", 29, "n_epochs"),
        ("training.optimizer.lr", 0.0001, "lr"),
        ("paper_protocol.selection.validation_fraction", 0.25, "target_real_val_fraction"),
        ("paper_protocol.selection.source_floor.max_drop", 0.03, "source_floor_max_drop"),
        ("data.target_real_norm_mode", "pre_zscored", "target_real_norm_mode"),
        ("adaptation.loss.soft_target_floor", 0.2, "adv_soft_target_floor"),
        ("adaptation.anchors.class_missing_weight", 0.4, "anchor_class_missing_weight"),
        ("adaptation.anchors.class_weight_cap", 5.0, "anchor_class_weight_cap"),
        ("adaptation.anchors.class_weight_gamma", 0.7, "anchor_class_weight_gamma"),
        ("adaptation.anchors.class_weight_min", 0.4, "anchor_class_weight_min"),
        ("adaptation.anchors.class_weight_mode", "inv_freq_kshot", "anchor_class_weight_mode"),
        ("adaptation.anchors.class_weight_reference_source", "all", "anchor_class_weight_reference_source"),
        ("adaptation.anchors.class_weights", "HYP=2", "anchor_class_weights"),
        ("adaptation.attack.boundary_prob_max", 0.9, "boundary_prob_max"),
        ("adaptation.attack.boundary_prob_min", 0.1, "boundary_prob_min"),
        ("adaptation.attack.einthoven_p95_max", 0.6, "einthoven_p95_max"),
        ("training.grad_clip", 2.0, "grad_clip"),
        ("adaptation.hull.dirichlet_alpha", 2.0, "hull_dirichlet_alpha"),
        ("adaptation.hull.weight_mode", "uniform", "hull_weight_mode"),
    ],
)
def test_f004_managed_build_rejects_every_config_sourced_projection_drift(
    path: str, value, field: str
):
    config = _load(SMOKE, "pytest-f004-managed-projection-drift")
    _set_nested(config, path, value)
    with pytest.raises(ConfigError, match=f"F-004 frozen projection.*{field}"):
        build_runner_commands(config)


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
    assert all(
        f004_identity(rho)["kshot_seed"] == F004_EXPECTED_KSHOT_SEED
        for rho in F004_RHO_VALUES
    )
    assert all(
        f004_identity(rho)["kshot_subset_root_family"]
        == F004_EXPECTED_KSHOT_ROOT_FAMILY
        for rho in F004_RHO_VALUES
    )


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


@pytest.mark.parametrize("path", [SMOKE, TRAIN, S5, DEPTH23])
def test_f004_uses_canonical_fresh_k500_root_without_five_arm_cross_product(path: Path):
    config = _load(path)

    assert config["data"]["kshot_subset_root"].endswith(
        "/paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets"
    )
    assert config["paper_protocol"]["kshot"]["seed"] == 20260601
    assert config["paper_protocol"]["kshot"]["subset_seed"] == 20260601
    assert set(config["runner"]["matrix"]) == {"center", "rho"}

    drifted = copy.deepcopy(config)
    drifted["data"]["kshot_subset_root"] = (
        "${paths.data_root}/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
    )
    with pytest.raises(ConfigError, match="F-004 K500 identity"):
        build_runner_commands(drifted)


def test_f004_clean_consumer_rejects_old_ref_exclusion_root_before_build():
    from ecg_adv_gen.config.adapters.pn2021_eval import build_pn2021_eval_argv

    config = _load(CLEAN, "pytest-f004-clean-root-drift")
    config["paper_protocol"]["comparison_protocol"] = F004_RHO_SWEEP_PROTOCOL
    config["data"]["kshot_subset_root"] = (
        "/data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
    )

    with pytest.raises(ValueError, match="F-004 K500 identity"):
        build_pn2021_eval_argv(
            config,
            {"matrix": {"center": "ningbo", "arm": "a0"}},
        )


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
    assert _option(child, "--comparison_topology_version") == "matched_effnet_a0_a2_a3_a4_a5_v5"
    assert _option(child, "--comparison_topology_sha256") == f004_identity(0.25)["topology_sha256"]
    assert _option(child, "--target_adv_fraction") == "0.25"
    assert _option(child, "--hull_label_mode") == "exact"
    assert _option(child, "--hull_mix_label_mode") == "anchor_soft"
    assert _option(child, "--hull_lr") == "0.25"
    assert _option(child, "--hull_neighbor_distance_space") == "standardized"
    assert _option(child, "--hull_neighbor_mode") == "local_random"
    assert _option(child, "--hull_neighbor_pool_size") == "120"
    assert _option(child, "--hull_neighbor_pool_multiplier") == "4"
    assert "--hull_include_anchor" not in child
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
    base = _f004_runtime_kwargs()
    assert validate_f004_runtime(**base)["rho"] == 0.25
    drifted_cases = [{**base, "comparison_arm": "a5"}]
    for key, value in [
        ("enable_raw_augmix", False),
        ("latent_augmix_consistency_weight", 0.0),
    ]:
        projection = {**base["behavior_projection"], key: value}
        drifted_cases.append({**base, "behavior_projection": projection})
    for drifted in drifted_cases:
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
    from ecg_adv_gen import matched_effnet

    current = {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        "comparison_variant": "f004_rho0p25",
        "comparison_topology_version": "matched_effnet_a0_a2_a3_a4_a5_v5",
        "comparison_topology_sha256": matched_effnet.F004_FROZEN_TOPOLOGY_SHA256,
        "target_adv_fraction": 0.25,
    }
    for key, value in [
        ("comparison_protocol", "other"),
        ("comparison_variant", "f004_rho0p5"),
        ("comparison_topology_version", "other"),
        ("comparison_topology_sha256", "0" * 64),
        ("target_adv_fraction", 0.5),
    ]:
        saved = {**current, key: value}
        assert [row["key"] for row in resume_contract_mismatches(saved, current)] == [key]


@pytest.mark.parametrize(
    "missing_key",
    [
        "comparison_protocol",
        "comparison_variant",
        "comparison_topology_version",
        "comparison_topology_sha256",
        "target_adv_fraction",
    ],
)
def test_f004_resume_contract_treats_missing_identity_as_drift(missing_key: str):
    from ecg_adv_gen import matched_effnet

    current = {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        "comparison_variant": "f004_rho0p25",
        "comparison_topology_version": "matched_effnet_a0_a2_a3_a4_a5_v5",
        "comparison_topology_sha256": matched_effnet.F004_FROZEN_TOPOLOGY_SHA256,
        "target_adv_fraction": 0.25,
    }
    saved = {key: value for key, value in current.items() if key != missing_key}
    assert resume_contract_mismatches(saved, current) == [
        {"key": missing_key, "saved": None, "current": current[missing_key]}
    ]


def _f004_runtime_kwargs() -> dict:
    from ecg_adv_gen import matched_effnet

    topology = dict(matched_effnet.F004_FROZEN_TOPOLOGY)
    return {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        "comparison_arm": "historical_unmatched",
        "comparison_variant": "f004_rho0p25",
        "comparison_topology_sha256": matched_effnet.F004_FROZEN_TOPOLOGY_SHA256,
        "target_adv_fraction": 0.25,
        "kshot_seed": F004_EXPECTED_KSHOT_SEED,
        "kshot_path": f"/data/{F004_EXPECTED_KSHOT_ROOT_FAMILY}",
        "behavior_projection": topology,
    }


@pytest.mark.parametrize(
    ("key", "drift"),
    [
        ("enable_vae_lhat", False),
        ("enable_raw_augmix", False),
        ("enable_latent_augmix_consistency", False),
        ("latent_augmix_bce_weight", 0.0),
        ("latent_augmix_consistency_weight", 0.0),
        ("latent_augmix_consistency_loss", "soft_bce"),
        ("latent_augmix_third_chain_role", "clean_anchor_control"),
        ("latent_augmix_width", 1),
        ("latent_augmix_depth", 1),
        ("latent_augmix_copies", 1),
        ("latent_augmix_chain_base_mode", "all_clean"),
        ("latent_augmix_adv_base_mix", 0.5),
        ("latent_augmix_alpha", 0.5),
        ("latent_augmix_severity", 1),
        ("latent_augmix_severity_profile", "calibrated_10to20pp"),
        ("latent_augmix_ops", ["powerline_noise"]),
        ("latent_augmix_signal_space", "model_zscore"),
        ("hull_M", 5),
        ("hull_lambda", 0.5),
        ("hull_steps", 3),
        ("hull_include_anchor", True),
        ("hull_init_logit_gap", 4.0),
        ("hull_label_mode", "compatible"),
        ("hull_mix_label_mode", "anchor"),
        ("hull_lr", 0.5),
        ("hull_neighbor_distance_space", "raw"),
        ("hull_neighbor_mode", "nearest"),
        ("hull_neighbor_pool_size", 20),
        ("hull_neighbor_pool_multiplier", 2),
        ("pgd_eps", 1.0),
    ],
)
def test_f004_shared_frozen_topology_rejects_every_component_drift(key: str, drift):
    kwargs = _f004_runtime_kwargs()
    kwargs["behavior_projection"] = {**kwargs["behavior_projection"], key: drift}
    with pytest.raises(ValueError, match=f"F-004 frozen projection.*{key}"):
        validate_f004_runtime(**kwargs)


def test_f004_managed_config_declares_shared_topology_fingerprint_and_rejects_drift():
    from ecg_adv_gen import matched_effnet

    config = _load(TRAIN)
    assert config["paper_protocol"]["topology_sha256"] == matched_effnet.F004_FROZEN_TOPOLOGY_SHA256
    commands = build_runner_commands(config)
    assert {
        _option(command["argv"], "--comparison_topology_sha256")
        for command in commands
    } == {matched_effnet.F004_FROZEN_TOPOLOGY_SHA256}

    drifted = copy.deepcopy(config)
    drifted["adaptation"]["latent_augmix"]["width"] = 1
    with pytest.raises(ConfigError, match="F-004 frozen projection.*latent_augmix_width"):
        build_runner_commands(drifted)


def test_f004_default_explicitly_locks_all_hull_identity_fields():
    raw = yaml.safe_load(F004_DEFAULT.read_text(encoding="utf-8"))
    hull = (raw.get("adaptation") or {}).get("hull") or {}

    assert {
        "label_mode": hull.get("label_mode"),
        "mix_label_mode": hull.get("mix_label_mode"),
        "lr": hull.get("lr"),
        "include_anchor": hull.get("include_anchor"),
        "neighbor_distance_space": hull.get("neighbor_distance_space"),
        "neighbor_mode": hull.get("neighbor_mode"),
        "neighbor_pool_size": hull.get("neighbor_pool_size"),
        "neighbor_pool_multiplier": hull.get("neighbor_pool_multiplier"),
    } == {
        "label_mode": "exact",
        "mix_label_mode": "anchor_soft",
        "lr": 0.25,
        "include_anchor": False,
        "neighbor_distance_space": "standardized",
        "neighbor_mode": "local_random",
        "neighbor_pool_size": 120,
        "neighbor_pool_multiplier": 4,
    }


def test_f004_child_locks_make_parent_f005_default_drift_command_invariant(monkeypatch):
    from ecg_adv_gen.config import loader

    options = (
        "--hull_label_mode",
        "--hull_mix_label_mode",
        "--hull_lr",
        "--hull_neighbor_distance_space",
        "--hull_neighbor_mode",
        "--hull_neighbor_pool_size",
        "--hull_neighbor_pool_multiplier",
        "--comparison_topology_version",
        "--comparison_topology_sha256",
    )
    baseline = build_runner_commands(_load(TRAIN, "pytest-f004-parent-baseline"))[0]["argv"]
    real_read_yaml = loader._read_yaml

    def drifting_read_yaml(path: Path) -> dict:
        raw = real_read_yaml(path)
        if path.name == "effnet_matched_f005_locked.yaml":
            raw = copy.deepcopy(raw)
            raw["adaptation"]["hull"].update({
                "label_mode": "primary",
                "mix_label_mode": "anchor",
                "lr": 0.9,
                "include_anchor": True,
                "neighbor_distance_space": "raw",
                "neighbor_mode": "nearest",
                "neighbor_pool_size": 5,
                "neighbor_pool_multiplier": 2,
            })
        return raw

    monkeypatch.setattr(loader, "_read_yaml", drifting_read_yaml)
    drifted = build_runner_commands(_load(TRAIN, "pytest-f004-parent-drift"))[0]["argv"]
    for option in options:
        assert _option(drifted, option) == _option(baseline, option)
    assert ("--hull_include_anchor" in drifted) is False


@pytest.mark.parametrize("key", tuple(F004_FROZEN_TOPOLOGY))
def test_f004_resume_contract_requires_every_frozen_topology_field(key: str):
    assert key in RESUME_CONTRACT_KEYS
    current = {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        key: F004_FROZEN_TOPOLOGY[key],
    }
    saved = (
        {}
        if key == "comparison_protocol"
        else {"comparison_protocol": F004_RHO_SWEEP_PROTOCOL}
    )
    mismatches = resume_contract_mismatches(saved, current)
    assert [row["key"] for row in mismatches] == [key]


def test_f004_eval_identity_resolver_is_fail_closed(tmp_path: Path):
    from ecg_adv_gen.evaluation.comparison_identity import (
        ComparisonIdentityError,
        resolve_producer_comparison_identity,
    )

    identity = f004_identity(0.25)
    producer = tmp_path / "producer"
    producer.mkdir()
    expected = json.dumps(identity, sort_keys=True)
    with pytest.raises(ComparisonIdentityError, match="run_config.json"):
        resolve_producer_comparison_identity(producer, expected)

    (producer / "run_config.json").write_text(
        json.dumps({"comparison_contract": {"comparison_identity": identity}}),
        encoding="utf-8",
    )
    assert resolve_producer_comparison_identity(producer, expected) == identity

    mismatch = {**identity, "variant": "f004_rho0p5"}
    with pytest.raises(ComparisonIdentityError, match="mismatch"):
        resolve_producer_comparison_identity(producer, json.dumps(mismatch))


def test_f004_eval_commands_carry_exact_identity_for_clean_s5_and_depth23():
    train_commands = build_runner_commands(_load(TRAIN))
    for command in train_commands:
        argv = command["argv"]
        expected = f004_identity(command["matrix"]["rho"])
        from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
        from ecg_adv_gen.runner.effnet_vae_lhat import (
            build_effnet_vae_lhat_eval_cmd,
            resolve_effnet_vae_lhat_paths,
        )

        args = wrapper.parse_args(argv[2:])
        paths = resolve_effnet_vae_lhat_paths(
            args, data_root=Path(args.data_root), out_root=Path(args.out_root)
        )
        clean_argv = build_effnet_vae_lhat_eval_cmd(
            args, python="python", data_root=Path(args.data_root), paths=paths
        )
        assert json.loads(_option(clean_argv, "--comparison_identity_json")) == expected

    for config_path in (S5, DEPTH23):
        for command in build_runner_commands(_load(config_path)):
            expected = f004_identity(command["matrix"]["rho"])
            assert json.loads(_option(command["argv"], "--comparison_identity_json")) == expected


def test_f004_metrics_and_table_preserve_exact_identity(tmp_path: Path):
    from ecg_adv_gen.evaluation import PN2021_ALL_ZERO_KEPT_REFEXCLUDED
    from ecg_adv_gen.reporting import export_metrics, export_paper_table

    centers = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
    for rho in F004_RHO_VALUES:
        identity = f004_identity(rho)
        for center in centers:
            path = tmp_path / center / identity["variant"] / "eval_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "comparison_identity": identity,
                "mapping": {"mapping_version": "v7_super5_sjr_rgq_review_20260528", "mapping_hash": "555ec85d5b51"},
                "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
                "pn2021": {"per_center": {center: {"macro_auroc": 0.8, "macro_auprc": 0.5}}},
            }), encoding="utf-8")

    metrics = export_metrics(
        [tmp_path],
        tmp_path / "metrics",
        required_comparison_protocol=F004_RHO_SWEEP_PROTOCOL,
        filter_to_target_center=True,
        target_centers=centers,
    )
    with Path(metrics["metrics_long"]).open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert {row["comparison_variant"] for row in metric_rows} == set(F004_VARIANTS)
    assert all(row["comparison_topology_sha256"] == f004_identity(float(row["comparison_rho"]))["topology_sha256"] for row in metric_rows)

    table = export_paper_table(
        Path(metrics["metrics_long"]),
        tmp_path / "table",
        view=PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
        dataset="pn2021",
        centers=centers,
        include_dataset_rows=False,
        baseline_run_id="f004_rho0",
    )
    with Path(table["paper_table"]).open(encoding="utf-8", newline="") as handle:
        table_rows = list(csv.DictReader(handle))
    assert {row["comparison_variant"] for row in table_rows} == set(F004_VARIANTS)
    assert all(row["comparison_protocol"] == F004_RHO_SWEEP_PROTOCOL for row in table_rows)


@pytest.mark.parametrize("config_path", [S5, DEPTH23])
def test_f004_corruption_stages_export_identity_preserving_metrics_and_tables(config_path: Path):
    commands = build_postprocess_commands(_load(config_path))
    assert len(commands) == 3
    metrics_argv = commands[0]["argv"]
    assert _option(metrics_argv, "--require-comparison-protocol") == F004_RHO_SWEEP_PROTOCOL
    assert "--run-id" not in metrics_argv
    assert {
        _option(command["argv"], "--baseline-run-id")
        for command in commands[1:]
    } == {"f004_rho0"}


def _f004_checkpoint_runtime_args() -> dict:
    identity = f004_identity(0.25)
    anchor = (
        f"/data/{F004_EXPECTED_KSHOT_ROOT_FAMILY}/cpsc_2018/"
        f"k500_seed{F004_EXPECTED_KSHOT_SEED}/cpsc_2018_real_k500_"
        f"seed{F004_EXPECTED_KSHOT_SEED}"
    )
    return {
        "comparison_protocol": identity["comparison_protocol"],
        "comparison_variant": identity["variant"],
        "comparison_topology_version": identity["topology_version"],
        "comparison_topology_sha256": identity["topology_sha256"],
        "target_adv_fraction": identity["rho"],
        "seed": F004_EXPECTED_KSHOT_SEED,
        "ref_meta_json": f"{anchor}.ref_meta.json",
        "synth_npz": f"{anchor}.latent.npz",
        "target_real_npz": f"{anchor}.raw1000.npz",
        **dict(F004_FROZEN_TOPOLOGY),
    }


def _drift_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, tuple):
        return ("intentional_drift",)
    return f"{value}_intentional_drift"


@pytest.mark.parametrize(
    ("layer", "key", "value"),
    [
        ("top", "comparison_protocol", "other"),
        ("top", "variant", "f004_rho0p5"),
        ("top", "topology_version", "other"),
        ("top", "topology_sha256", "0" * 64),
        ("top", "rho", 0.5),
        ("args", "comparison_protocol", "other"),
        ("args", "comparison_variant", "f004_rho0p5"),
        ("args", "comparison_topology_version", "other"),
        ("args", "comparison_topology_sha256", "0" * 64),
        ("args", "target_adv_fraction", 0.5),
    ],
)
def test_f004_checkpoint_identity_hard_gate_rejects_tampering_even_with_drift_override(
    layer: str, key: str, value
):
    from ecg_adv_gen.training.resume_contract import validate_f004_checkpoint_identity

    identity = f004_identity(0.25)
    current = {**_f004_checkpoint_runtime_args(), "allow_resume_config_drift": True}
    checkpoint = {"comparison_identity": dict(identity), "args": dict(current)}
    checkpoint[layer if layer == "args" else "comparison_identity"][key] = value
    with pytest.raises(ValueError, match="F-004 checkpoint identity"):
        validate_f004_checkpoint_identity(checkpoint, current)


@pytest.mark.parametrize(
    "key",
    (
        "seed",
        "ref_meta_json",
        "synth_npz",
        "target_real_npz",
        *F004_FROZEN_TOPOLOGY,
    ),
)
def test_f004_checkpoint_hard_gate_rejects_runtime_tamper_despite_override(key: str):
    from ecg_adv_gen.training.resume_contract import validate_f004_checkpoint_identity

    current = {**_f004_checkpoint_runtime_args(), "allow_resume_config_drift": True}
    checkpoint = {
        "comparison_identity": f004_identity(0.25),
        "args": dict(current),
    }
    checkpoint["args"][key] = _drift_value(checkpoint["args"][key])

    with pytest.raises(ValueError, match=f"F-004 checkpoint identity.*{key}"):
        validate_f004_checkpoint_identity(checkpoint, current)


@pytest.mark.parametrize(
    "key",
    (
        "seed",
        "ref_meta_json",
        "synth_npz",
        "target_real_npz",
        *F004_FROZEN_TOPOLOGY,
    ),
)
def test_f004_checkpoint_hard_gate_rejects_missing_runtime_field_despite_override(
    key: str,
):
    from ecg_adv_gen.training.resume_contract import validate_f004_checkpoint_identity

    current = {**_f004_checkpoint_runtime_args(), "allow_resume_config_drift": True}
    checkpoint = {
        "comparison_identity": f004_identity(0.25),
        "args": dict(current),
    }
    checkpoint["args"].pop(key)

    with pytest.raises(ValueError, match=f"F-004 checkpoint identity.*{key}"):
        validate_f004_checkpoint_identity(checkpoint, current)


@pytest.mark.parametrize(
    ("layer", "key"),
    [
        ("top", "comparison_identity"),
        ("args", "comparison_protocol"),
        ("args", "comparison_variant"),
        ("args", "comparison_topology_version"),
        ("args", "comparison_topology_sha256"),
        ("args", "target_adv_fraction"),
    ],
)
def test_f004_checkpoint_identity_hard_gate_rejects_missing_fields(layer: str, key: str):
    from ecg_adv_gen.training.resume_contract import validate_f004_checkpoint_identity

    identity = f004_identity(0.25)
    current = _f004_checkpoint_runtime_args()
    checkpoint = {"comparison_identity": dict(identity), "args": dict(current)}
    if layer == "top":
        checkpoint.pop(key)
    else:
        checkpoint["args"].pop(key)
    with pytest.raises(ValueError, match="F-004 checkpoint identity"):
        validate_f004_checkpoint_identity(checkpoint, current)


def test_non_f004_legacy_checkpoint_keeps_resume_compatibility():
    from ecg_adv_gen.training.resume_contract import validate_f004_checkpoint_identity

    assert validate_f004_checkpoint_identity(
        {"args": {"center_name": "ningbo"}}, {"center_name": "ningbo"}
    ) is None


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("topology_reference", "a5"),
        ("topology_version", "wrong_version"),
        ("only_varied_parameter", "hull_lambda"),
    ],
)
def test_f004_managed_yaml_rejects_false_protocol_declarations(field: str, wrong: str):
    config = _load(TRAIN)
    config["paper_protocol"][field] = wrong
    with pytest.raises(ConfigError, match=field):
        build_runner_commands(config)


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
