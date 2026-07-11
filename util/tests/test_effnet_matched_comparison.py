from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_adv_gen.config import (
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)
from ecg_adv_gen.config.launch import verify_required_inputs
from ecg_adv_gen.data import kshot
from ecg_adv_gen.evaluation import selection
from ecg_adv_gen.runner import synth_online_at_super5 as synth_runner
from ecg_adv_gen.runner.synth_online_at_super5 import train_latent_augmix_consistency_epoch


REPO = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO / "configs/local/linbinhao_server.example.yaml"


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _ids_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def test_matched_a0_a5_share_split_and_validation_never_enters_latent_candidates():
    record_ids = np.asarray([f"r{i:03d}" for i in range(500)])
    labels = np.eye(5, dtype=np.float32)[np.arange(500) % 5]

    a0 = kshot.matched_k500_split(record_ids, labels, val_fraction=0.2, seed=20260601)
    a5 = kshot.matched_k500_split(record_ids, labels, val_fraction=0.2, seed=20260601)

    assert a0 == a5
    assert len(a0["train_record_ids"]) == 400
    assert len(a0["val_record_ids"]) == 100
    assert set(a0["train_record_ids"]).isdisjoint(a0["val_record_ids"])
    assert a0["train_record_ids_sha256"] == _ids_hash(a0["train_record_ids"])
    assert a0["val_record_ids_sha256"] == _ids_hash(a0["val_record_ids"])

    order = np.random.default_rng(9).permutation(500)
    latents = np.arange(500 * 2, dtype=np.float32).reshape(500, 2)[order]
    latent_labels = labels[order]
    source_meta = {"record_ids": record_ids[order], "source_ids": order}
    kept_latents, kept_labels, kept_meta = kshot.filter_latent_candidates(
        latents,
        latent_labels,
        source_meta,
        train_record_ids=a0["train_record_ids"],
        validation_record_ids=a0["val_record_ids"],
    )

    assert kept_latents.shape[0] == kept_labels.shape[0] == 400
    assert set(kept_meta["record_ids"]) == set(a0["train_record_ids"])
    assert set(kept_meta["record_ids"]).isdisjoint(a0["val_record_ids"])


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)
        self.realized_steps = 0

    def step(self, closure=None):
        self.realized_steps += 1
        return super().step(closure)


def test_a0_clean_control_and_a5_augmix_have_equal_auxiliary_optimizer_steps():
    torch.manual_seed(3)
    clean = np.random.default_rng(3).normal(size=(12, 1, 2)).astype(np.float32)
    labels = (np.arange(12) % 2).astype(np.float32)[:, None]
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    def run(views: np.ndarray) -> tuple[int, dict]:
        model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1))
        optimizer = _CountingSGD(model.parameters())
        stats = train_latent_augmix_consistency_epoch(
            model=model,
            clean_signals_ct=clean,
            augmix_signals_ct=views,
            labels_np=labels,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            copies=2,
            consistency_weight=0.5,
            bce_weight=0.0,
            consistency_loss="jsd",
            batch_size=4,
            crop_len=2,
            grad_clip=0.0,
            trainable_params=list(model.parameters()),
        )
        return optimizer.realized_steps, stats

    a0_steps, a0 = run(np.tile(clean, (2, 1, 1)))
    a5_steps, a5 = run(np.tile(clean + 0.1, (2, 1, 1)))

    assert a0_steps == a5_steps == 3
    assert a0["n_batches"] == a5["n_batches"] == 3
    assert a0["bce_weight"] == a5["bce_weight"] == 0.0
    assert abs(a0["consistency_loss"]) < 1e-7
    assert a5["consistency_loss"] > 0.0


def test_matched_auxiliary_step_does_not_apply_hard_bce():
    torch.manual_seed(4)
    clean = np.asarray([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32)
    views = np.tile(clean + 2.0, (2, 1, 1))
    labels = np.asarray([[0.0], [1.0]], dtype=np.float32)
    model = nn.Sequential(nn.Flatten(), nn.Linear(2, 1))
    optimizer = _CountingSGD(model.parameters())
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    stats = train_latent_augmix_consistency_epoch(
        model=model,
        clean_signals_ct=clean,
        augmix_signals_ct=views,
        labels_np=labels,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        copies=2,
        consistency_weight=0.5,
        bce_weight=0.0,
        consistency_loss="jsd",
        batch_size=2,
        crop_len=2,
        grad_clip=0.0,
        trainable_params=list(model.parameters()),
    )

    assert optimizer.realized_steps == stats["n_batches"] == 1
    assert stats["n_generated"] == 4
    assert stats["bce_weight"] == 0.0
    assert stats["bce_loss"] > 0.0
    assert stats["loss"] == pytest.approx(0.5 * stats["consistency_loss"])
    assert "target_adv_count" not in stats


def test_primary_adversarial_dataset_center_crops_and_pads_to_crop_len():
    labels = np.eye(5, dtype=np.float32)[:2]
    signals = np.arange(2 * 12 * 1000, dtype=np.float32).reshape(2, 12, 1000)
    builder = getattr(synth_runner, "_build_primary_adv_dataset", None)

    assert callable(builder), "runner must expose primary adversarial dataset construction"
    cropped = builder(signals, labels, crop_len=250)
    cropped_signals, cropped_labels = cropped.tensors
    assert cropped_signals.shape == (2, 12, 250)
    assert torch.equal(cropped_signals, torch.from_numpy(signals[..., 375:625]))
    assert torch.equal(cropped_labels, torch.from_numpy(labels))

    unchanged_signals, _ = builder(signals, labels, crop_len=1000).tensors
    assert torch.equal(unchanged_signals, torch.from_numpy(signals))

    short = signals[..., :200]
    padded_signals, _ = builder(short, labels, crop_len=250).tensors
    assert padded_signals.shape == (2, 12, 250)
    assert torch.count_nonzero(padded_signals[..., :25]) == 0
    assert torch.equal(padded_signals[..., 25:225], torch.from_numpy(short))
    assert torch.count_nonzero(padded_signals[..., 225:]) == 0


def test_best_source_floor_result_tracks_selected_epoch_and_survives_resume():
    from ecg_adv_gen.runner.synth_online_at_super5 import (
        restore_best_selection_state,
        update_best_selection_state,
    )

    accepted = selection.update_matched_checkpoint_selection(
        best_metric=0.51, candidate_metric=0.55, source_metric=0.75,
        source_baseline_metric=0.76, source_max_drop=0.02,
    )
    rejected = selection.update_matched_checkpoint_selection(
        best_metric=0.55, candidate_metric=0.60, source_metric=0.73,
        source_baseline_metric=0.76, source_max_drop=0.02,
    )
    assert accepted["source_floor_passed"] is accepted["selected"] is True
    assert rejected["source_floor_passed"] is rejected["selected"] is False
    state = update_best_selection_state(0.51, 0, {}, accepted, epoch=1)
    state = update_best_selection_state(*state, rejected, epoch=2)

    assert state[0] == 0.55
    assert state[1] == 1
    assert state[2] == accepted
    resumed = restore_best_selection_state(
        {
            "best_metric": state[0],
            "best_epoch": state[1],
            "best_source_floor_result": state[2],
        },
        (-1.0, 0, {}),
    )
    assert resumed == state


def test_preflight_accepts_verified_ptbxl_source_lineage_without_k500_identity(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"source-only-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evidence = tmp_path / "train_result.json"
    evidence.write_text(
        json.dumps(
            {
                "scheme": "super5",
                "num_classes": 5,
                "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
                "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
                "config": {
                    "data_path": "/data/ptbxl/raw100.npy",
                    "synth_npz": None,
                    "init_ckpt": None,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "artifact_trace": {
            "inputs": {
                "checkpoints": [
                    {
                        "role": "model.init_checkpoint",
                        "path": str(checkpoint),
                        "required": True,
                        "sha256": checkpoint_sha,
                    }
                ],
                "k500_refs": [],
                "data_caches": [],
            },
            "initialization": {
                "stage": "ptbxl_source",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "evidence_path": str(evidence),
            },
        },
        "commands": [],
    }

    report = verify_required_inputs(manifest)

    assert report["passed"] is True, report
    assert report["initialization_lineage"]["stage"] == "ptbxl_source"
    assert report["initialization_lineage"]["source_only"] is True


def test_preflight_rejects_source_evidence_bound_to_a_different_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"selected-source-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    other = tmp_path / "other_model.pt"
    other.write_bytes(b"other-source-checkpoint")
    evidence = tmp_path / "train_result.json"
    evidence.write_text(
        json.dumps(
            {
                "scheme": "super5",
                "num_classes": 5,
                "checkpoint": {
                    "path": str(other),
                    "sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
                },
                "config": {
                    "data_path": "/data/ptbxl/raw100.npy",
                    "synth_npz": None,
                    "init_ckpt": None,
                },
            }
        ),
        encoding="utf-8",
    )
    report = verify_required_inputs(
        {
            "artifact_trace": {
                "inputs": {"checkpoints": [], "k500_refs": [], "data_caches": []},
                "initialization": {
                    "stage": "ptbxl_source",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "evidence_path": str(evidence),
                },
            },
            "commands": [],
        }
    )

    assert report["passed"] is False
    assert "evidence checkpoint binding mismatch" in report["lineage_errors"][0]["error"]


def test_managed_matched_a0_a5_commands_share_every_non_method_contract():
    config_path = REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml"
    config = load_experiment_config(
        config_path,
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_matched"},
    )
    config["runner"]["matrix"]["case"] = [
        case for case in config["runner"]["matrix"]["case"] if case["arm"] in {"a0", "a5"}
    ]
    commands = build_runner_commands(config)
    pairs: dict[str, dict[str, list[str]]] = {}
    for command in commands:
        pairs.setdefault(command["matrix"]["center"], {})[
            command["matrix"]["case"]["arm"]
        ] = command["argv"]

    assert set(pairs) == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    shared_options = (
        "--init_ckpt",
        "--epochs",
        "--seed",
        "--train_batch_size",
        "--lr",
        "--ptbxl_weight",
        "--target_real_weight",
        "--target_real_val_fraction",
        "--target_real_val_seed",
        "--selection_metric",
        "--source_floor_max_drop",
    )
    for arms in pairs.values():
        assert set(arms) == {"a0", "a5"}
        assert _option(arms["a0"], "--comparison_arm") == "a0"
        assert _option(arms["a5"], "--comparison_arm") == "a5"
        for option in shared_options:
            assert _option(arms["a0"], option) == _option(arms["a5"], option)
        assert _option(arms["a0"], "--ptbxl_weight") == "0.0"
        assert _option(arms["a0"], "--target_adv_fraction") == "0.0"
        assert _option(arms["a5"], "--target_adv_fraction") == "0.5"


@pytest.mark.parametrize("rho", [0.0, 0.25])
def test_managed_fake_a5_rho_matrix_is_rejected_before_path_collision(rho: float):
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_target_rho"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "comparison_arm": ["a5"],
        "target_adv_fraction": [rho],
    }

    with pytest.raises(ValueError, match="canonical matched EffNet|matrix keys"):
        build_runner_commands(config)


@pytest.mark.parametrize("rho", [0.0, 0.25])
def test_canonical_a5_record_cannot_report_a_noncanonical_runtime_rho(rho: float):
    split = {
        "train_record_ids_sha256": "a" * 64,
        "val_record_ids_sha256": "b" * 64,
        "train_record_ids": ["r1"],
        "val_record_ids": ["r2"],
        "val_fraction": 0.5,
        "seed": 7,
    }
    with pytest.raises(ValueError, match="canonical matched EffNet arm a5 requires target_adv_fraction=0.5"):
        selection.build_matched_training_record(
            comparison_arm="a5",
            runtime_target_adv_fraction=rho,
            source_checkpoint_path="/data/source.pt",
            source_checkpoint_sha256="c" * 64,
            split=split,
            selection_metric="macro_auprc",
            source_floor_max_drop=0.02,
            epochs=2,
            optimizer_steps_per_epoch=3,
            realized_optimizer_steps=6,
            scheduler_steps=2,
            source_floor_result={"source_floor_passed": True},
        )


def test_managed_target_adv_fraction_rejects_undeclared_matrix_value():
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_target_rho_invalid"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "comparison_arm": ["a5"],
        "target_adv_fraction": [0.75],
    }

    with pytest.raises(ValueError, match="target_adv_fraction"):
        build_runner_commands(config)

def test_matched_manifest_child_paths_equal_runtime_arm_paths():
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner.effnet_vae_lhat import resolve_effnet_vae_lhat_paths

    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_manifest_arms"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "case": [
            case for case in config["runner"]["matrix"]["case"] if case["arm"] in {"a0", "a5"}
        ],
    }
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id="pytest_manifest_arms",
        cli_args=Namespace(dry_run=True, write_plan=True),
    )
    child_paths = {
        child["matrix"]["case"]["arm"]: child["child_run_dir"]
        for child in manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    }
    assert child_paths["a0"] != child_paths["a5"]
    assert "_arma0_" in child_paths["a0"]
    assert "_arma5_" in child_paths["a5"]
    init = manifest["artifact_trace"]["initialization"]
    assert init["stage"] == "ptbxl_source"
    assert len(init["checkpoint_sha256"]) == 64
    assert init["checkpoint_path"].endswith("/best_model.pt")

    for command in commands:
        args = wrapper.parse_args(command["argv"][2:])
        runtime_path = resolve_effnet_vae_lhat_paths(
            args, data_root=Path(args.data_root), out_root=Path(args.out_root)
        ).out_dir
        arm = command["matrix"]["case"]["arm"]
        assert child_paths[arm] == str(runtime_path)


def test_historical_direct_runner_cannot_be_mistaken_for_matched_a0():
    source = (REPO / "ecg_adv_gen/runner/effnet_direct_finetune.py").read_text(encoding="utf-8")

    assert "historical_unmatched" in source


def test_matched_contract_reaches_the_shared_child_parser(monkeypatch):
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_matched"},
    )
    config["runner"]["matrix"]["case"] = [
        case for case in config["runner"]["matrix"]["case"] if case["arm"] in {"a0", "a5"}
    ]
    command = next(
        item
        for item in build_runner_commands(config)
        if item["matrix"]["center"] == "ningbo" and item["matrix"]["case"]["arm"] == "a5"
    )
    args = wrapper.parse_args(command["argv"][2:])
    paths = resolve_effnet_vae_lhat_paths(
        args,
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
    )
    child_command = build_effnet_vae_lhat_train_cmd(
        args,
        python=sys.executable,
        data_root=Path(args.data_root),
        paths=paths,
        class_trust=Path("/tmp/class_trust.json"),
    )
    monkeypatch.setattr(sys, "argv", ["synth_online_at_super5.py", *child_command[3:]])

    child_args = child.parse_args()

    assert child_args.comparison_arm == "a5"
    assert child_args.init_lineage_stage == "ptbxl_source"
    assert len(child_args.init_checkpoint_sha256) == 64
    assert child_args.target_real_val_fraction == 0.2
    assert child_args.target_real_val_seed == 20260601
    assert child_args.selection_metric == "macro_auprc"
    assert child_args.source_floor_max_drop == 0.02
    assert args.target_adv_fraction == child_args.target_adv_fraction == 0.5


def test_unmanaged_child_target_adv_fraction_defaults_to_legacy_none(monkeypatch):
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth_online_at_super5.py",
            "--center_name",
            "ningbo",
            "--synth_npz",
            "unused.npz",
            "--output_dir",
            "unused",
        ],
    )

    assert child.parse_args().target_adv_fraction is None


def test_unmanaged_wrapper_target_adv_fraction_defaults_to_legacy_none():
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper

    assert wrapper.parse_args([]).target_adv_fraction is None


def test_run_record_contract_differs_only_by_declared_arm():
    split = {
        "train_record_ids_sha256": "a" * 64,
        "val_record_ids_sha256": "b" * 64,
        "train_record_ids": ["r1", "r2"],
        "val_record_ids": ["r3"],
        "val_fraction": 1 / 3,
        "seed": 7,
    }
    kwargs = {
        "source_checkpoint_path": "/data/source.pt",
        "source_checkpoint_sha256": "c" * 64,
        "split": split,
        "selection_metric": "macro_auprc",
        "source_floor_max_drop": 0.02,
        "epochs": 2,
        "optimizer_steps_per_epoch": 3,
        "realized_optimizer_steps": 6,
        "scheduler_steps": 2,
        "source_floor_result": {"source_floor_passed": True},
    }
    a0 = selection.build_matched_training_record(comparison_arm="a0", **kwargs)
    a5 = selection.build_matched_training_record(comparison_arm="a5", **kwargs)

    assert a0["comparison_arm"] == "a0"
    assert a5["comparison_arm"] == "a5"
    allowed_differences = {
        "comparison_arm", "method_components", "role",
        "target_adv_fraction", "third_chain_route",
    }
    assert {k: v for k, v in a0.items() if k not in allowed_differences} == {
        k: v for k, v in a5.items() if k not in allowed_differences
    }
    assert a0["method_components"] == {
        "vae_lhat": False, "raw_augmix": False, "augmix_view_bce": False, "jsd": False,
    }
    assert a5["method_components"] == {
        "vae_lhat": True, "raw_augmix": True, "augmix_view_bce": True, "jsd": True,
    }
    assert a0["source_checkpoint"]["stage"] == "ptbxl_source"
    assert a0["k500_split"]["train_record_ids_sha256"] == "a" * 64
    assert a0["selection"]["source_floor_result"]["source_floor_passed"] is True
    assert a0["budget"]["realized_optimizer_steps"] == 6
