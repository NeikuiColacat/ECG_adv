from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

import boot_scripts.select_pn2021_direct as select_module
import boot_scripts.tune_pn2021_direct as tune_module
from boot_scripts.refit_pn2021_direct import main as refit_main
from boot_scripts.run_experiment import load_experiment_plan
from boot_scripts.select_pn2021_direct import main as select_main
from boot_scripts.tune_pn2021_direct import main as tune_main
from util.config_bundle import resolve_yaml_config_closure
from util.evaluation.direct_baseline_selection import locked_pooled_selection_rule
from util.run_record import build_run_file_index, sha256_file, snapshot_yaml_files


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021_direct_tune.yaml"
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")


def _latent_tuning_bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", root)
    path = root / "train" / "PN2021_direct_tune.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol_id = "pn2021_latent_threechain_augmix_residual_depth23_aug075_tuning"
    payload["profile_name"] = protocol_id
    payload["protocol_lock"]["protocol_id"] = protocol_id
    payload["protocol_lock"]["method_id"] = (
        "latent_threechain_augmix_residual_depth23_aug075"
    )
    payload["references"]["method_config"] = (
        "train/methods/exp_paired_augmix_latent_bridge_v1.yaml"
    )
    payload["training"]["method"] = (
        "latent_threechain_augmix_residual_depth23_aug075"
    )
    payload["training"].pop("family_loss_weights", None)
    payload["training"]["objective_source"] = "method_profile"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, root


def _a5_tuning_bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", root)
    path = root / "train" / "PN2021_direct_tune.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol_id = "pn2021_a5_lhat_threechain_tuning"
    payload["profile_name"] = protocol_id
    payload["protocol_lock"]["protocol_id"] = protocol_id
    payload["protocol_lock"]["method_id"] = "augmix_simclr_lhat"
    payload["references"]["method_config"] = (
        "train/methods/augmix_simclr_lhat.yaml"
    )
    payload["training"]["method"] = "augmix_simclr_lhat"
    payload["training"].pop("family_loss_weights", None)
    payload["training"]["objective_source"] = "method_profile"
    payload["pooled_selection"]["output_file"] = (
        "a5_lhat_threechain_selection.json"
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, root


def _selection(
    tmp_path: Path,
    *,
    online_config_sha256: str | None = None,
) -> Path:
    source = tmp_path / "source.pt"
    source.write_bytes(b"locked-source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    method_path = (
        REPO / "configs" / "train" / "methods" / "direct_depth23_fixed20.yaml"
    )
    online_config_path = REPO / "configs" / "train" / "PN2021_fixed20.yaml"
    tuning_config_path = REPO / "configs" / "train" / "PN2021_direct_tune.yaml"
    comparison = {
        "protocol_id": "pn2021_direct_family_balanced_tuning",
        "method_id": "direct_depth23_fixed20",
        "comparison_group": "pn2021_direct_depth23_fixed20_family_balanced",
        "replicate_id": 0,
        "pos_weight": None,
        "model_family": "efficientnet1dv2",
        "method_profile_sha256": sha256_file(method_path),
        "online_training_config_sha256": (
            online_config_sha256 or sha256_file(online_config_path)
        ),
        "tuning_config_sha256": sha256_file(tuning_config_path),
        "source_checkpoint_identity": {
            "path": str(source),
            "sha256": source_sha,
        },
        "resolved_training_parameters": {
            "epochs": 6,
            "scheduler_horizon_epochs": 6,
            "batch_size": 128,
            "learning_rate": 5.0e-5,
            "weight_decay": 1.0e-4,
            "minimum_learning_rate_ratio": 0.01,
            "gradient_clip_norm": 1.0,
            "amp_enabled": True,
            "amp_dtype": "bfloat16",
        },
    }
    per_epoch = [
        {
            "epoch": epoch,
            "clean": {"macro_auprc": 0.75, "macro_auroc": 0.82},
            "robust": {
                "macro_auprc": 0.65 if epoch == 3 else 0.5 + epoch * 1.0e-3,
                "macro_auroc": 0.76 if epoch == 3 else 0.6 + epoch * 1.0e-3,
            },
            "score": 0.7 if epoch == 3 else 0.625 + epoch * 5.0e-4,
            "clean_floor": 0.72,
            "eligible": True,
        }
        for epoch in range(1, 7)
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "direct_k500_pooled_epoch_selection",
        "status": "selected",
        "centers": list(CENTERS),
        "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
        "record_count": 400,
        "composition_count": 20,
        "selection_rule": locked_pooled_selection_rule(),
        "selected_epoch": 3,
        "selected_score": 0.7,
        "selected_clean_macro_auprc": 0.75,
        "selected_robust_macro_auprc": 0.65,
        "epoch_grid": list(range(1, 7)),
        "per_epoch": per_epoch,
        "comparison_identity": comparison,
        "comparison_identity_sha256": hashlib.sha256(
            json.dumps(
                comparison,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    run_root = tmp_path / "managed-selection"
    training = run_root / "training"
    training.mkdir(parents=True)
    path = training / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    closure = resolve_yaml_config_closure(
        (
            REPO / "configs" / "train" / "methods" / "direct_depth23_fixed20.yaml",
            online_config_path,
            tuning_config_path,
            REPO / "configs" / "data" / "splits.yaml",
            REPO / "configs" / "data" / "data_load.yaml",
            REPO / "configs" / "random_seed.yaml",
            REPO / "configs" / "baselines" / "ptbxl_source_v1.yaml",
        ),
        config_root=REPO / "configs",
    )
    snapshots = snapshot_yaml_files(
        run_root,
        [(item, "referenced_config") for item in closure],
        config_root=REPO / "configs",
    )
    (run_root / "run_card.json").write_text(
        json.dumps({"status": "complete", "exit_code": 0}),
        encoding="utf-8",
    )
    index_path = run_root / "run_file_index.json"
    index_path.write_text(
        json.dumps(build_run_file_index(run_root)), encoding="utf-8"
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_file_index_sha256": sha256_file(index_path),
                "config_snapshots": snapshots,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_direct_tuning_boot_dry_run_uses_locked_model_profile(tmp_path, capsys):
    output = tmp_path / "training"

    assert tune_main(
        [
            "--config",
            str(CONFIG),
            "--model",
            "efficientnet1dv2",
            "--center",
            "ningbo",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["action"] == "pn2021_direct_k500_internal_tuning"
    assert payload["center"] == "ningbo"
    assert payload["model"]["sampling_rate_hz"] == 100
    assert payload["training_parameters"]["epochs"] == 30
    assert payload["training_parameters"]["scheduler_horizon_epochs"] == 30
    assert payload["config"]["resolved"]["protocol_lock"]["protocol_id"] == (
        "pn2021_direct_family_balanced_tuning"
    )
    assert payload["selection"] == (
        "deferred_to_four_center_pooled_raw_predictions"
    )
    assert payload["method"]["profile_name"] == "direct_depth23_fixed20"
    assert payload["vae_requirements"] == {
        "encoder": False,
        "decoder": False,
        "latent_pool": False,
    }
    assert not output.exists()


def test_latent_tuning_dry_run_declares_vae_without_loading_weights(
    monkeypatch, tmp_path, capsys
):
    config, root = _latent_tuning_bundle(tmp_path)
    monkeypatch.setattr(
        tune_module,
        "build_ecgtwin_vae",
        lambda **kwargs: pytest.fail("dry-run must not load VAE weights"),
    )
    vae_checkpoint = tmp_path / "vae.pt"

    assert tune_module.main(
        [
            "--config",
            str(config),
            "--config-root",
            str(root),
            "--model",
            "efficientnet1dv2",
            "--center",
            "ningbo",
            "--vae-checkpoint",
            str(vae_checkpoint),
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["action"] == "pn2021_k500_internal_tuning"
    assert payload["method"]["profile_name"] == (
        "latent_threechain_augmix_residual_depth23_aug075"
    )
    assert payload["vae_checkpoint"] == str(vae_checkpoint)
    assert payload["vae_requirements"] == {
        "encoder": True,
        "decoder": True,
        "latent_pool": False,
    }


def test_a5_tuning_dry_run_declares_train400_pool_without_loading_weights(
    monkeypatch, tmp_path, capsys
):
    config, root = _a5_tuning_bundle(tmp_path)
    monkeypatch.setattr(
        tune_module,
        "build_ecgtwin_vae",
        lambda **kwargs: pytest.fail("dry-run must not load VAE weights"),
    )
    vae_checkpoint = tmp_path / "vae.pt"

    assert tune_module.main(
        [
            "--config",
            str(config),
            "--config-root",
            str(root),
            "--model",
            "efficientnet1dv2",
            "--center",
            "ningbo",
            "--vae-checkpoint",
            str(vae_checkpoint),
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["action"] == "pn2021_k500_internal_tuning"
    assert payload["method"]["profile_name"] == "augmix_simclr_lhat"
    assert payload["vae_requirements"] == {
        "encoder": True,
        "decoder": True,
        "latent_pool": True,
    }


def test_direct_selector_boot_dry_run_requires_all_four_centers(tmp_path, capsys):
    arguments = ["--config", str(CONFIG)]
    for center in CENTERS:
        arguments.extend(
            ["--center-run-dir", f"{center}={tmp_path / center}"]
        )
    arguments.append("--dry-run")

    assert select_main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["aggregation"] == "pool_four_centers_per_clean_or_composition"
    assert payload["metric"] == (
        "0.5_clean_plus_0.5_mean20_macro_auprc_with_clean_floor"
    )
    assert set(payload["center_run_dirs"]) == set(CENTERS)
    assert payload["action"] == "select_pn2021_direct_global_epoch"


def test_latent_selector_reuses_entrypoint_without_direct_action(tmp_path, capsys):
    config, root = _latent_tuning_bundle(tmp_path)
    arguments = ["--config", str(config), "--config-root", str(root)]
    for center in CENTERS:
        arguments.extend(
            ["--center-run-dir", f"{center}={tmp_path / center}"]
        )
    arguments.append("--dry-run")

    assert select_main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["action"] == "select_pn2021_global_epoch"
    assert payload["config"]["method_id"] == (
        "latent_threechain_augmix_residual_depth23_aug075"
    )


def test_direct_selector_returns_nonzero_when_clean_floor_has_no_epoch(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        select_module,
        "discover_validation_prediction_artifacts",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        select_module,
        "select_direct_baseline_epoch",
        lambda *args, **kwargs: {
            "status": "failed_clean_floor",
            "selected_epoch": None,
            "per_epoch": [],
        },
    )
    arguments = ["--config", str(CONFIG), "--output-dir", str(tmp_path / "selection")]
    for center in CENTERS:
        arguments.extend(["--center-run-dir", f"{center}={tmp_path / center}"])

    assert select_module.main(arguments) == select_module.FAILED_CLEAN_FLOOR_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed_clean_floor"
    assert payload["selected_epoch"] is None


def test_direct_tuning_and_selector_are_managed_entrypoints(tmp_path):
    tuning = load_experiment_plan(
        REPO
        / "configs"
        / "experiments"
        / "manual_refactor_pn2021_effnet_direct_tune_ningbo.yaml",
        run_dir=tmp_path / "tune-run",
    )
    selector = load_experiment_plan(
        REPO
        / "configs"
        / "experiments"
        / "manual_refactor_pn2021_effnet_direct_select.yaml",
        run_dir=tmp_path / "select-run",
    )

    assert tuning.entrypoint_name == "tune_pn2021_direct"
    assert selector.entrypoint_name == "select_pn2021_direct"
    assert not tuning.run_dir.exists()
    assert not selector.run_dir.exists()


def test_direct_refit_dry_run_consumes_selection_epoch_and_horizon(tmp_path, capsys):
    selection = _selection(tmp_path)
    output = tmp_path / "refit"
    assert refit_main(
        [
            "--config",
            str(REPO / "configs" / "train" / "PN2021_fixed20.yaml"),
            "--model",
            "efficientnet1dv2",
            "--method-config",
            "train/methods/direct_depth23_fixed20.yaml",
            "--center",
            "ningbo",
            "--selection-json",
            str(selection),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["training_parameters"]["epochs"] == 3
    assert payload["training_parameters"]["scheduler_horizon_epochs"] == 6
    assert payload["training_parameters"]["batch_size"] == 128
    assert payload["method"]["profile_name"] == "direct_depth23_fixed20"
    assert payload["selection"]["selected_score"] == 0.7
    assert payload["selection"]["selected_clean_macro_auprc"] == 0.75
    assert payload["selection"]["selected_robust_macro_auprc"] == 0.65
    assert payload["selection"]["heldout_evaluation_used"] is False
    assert not output.exists()


def test_direct_refit_rejects_online_config_sha_drift(tmp_path):
    selection = _selection(
        tmp_path,
        online_config_sha256=sha256_file(
            REPO / "configs" / "train" / "PN2021_direct_tune.yaml"
        ),
    )
    with pytest.raises(ValueError, match="online-training config SHA256 differs"):
        refit_main(
            [
                "--config",
                str(REPO / "configs" / "train" / "PN2021_fixed20.yaml"),
                "--model",
                "efficientnet1dv2",
                "--method-config",
                "train/methods/direct_depth23_fixed20.yaml",
                "--center",
                "ningbo",
                "--selection-json",
                str(selection),
                "--dry-run",
            ]
        )


def test_removed_fixed20_cycle_flag_is_rejected(tmp_path):
    selection = _selection(tmp_path)
    with pytest.raises(SystemExit):
        refit_main(
            [
                "--config",
                str(REPO / "configs" / "train" / "PN2021_fixed20.yaml"),
                "--model",
                "efficientnet1dv2",
                "--method-config",
                "train/methods/direct_depth23_fixed20.yaml",
                "--center",
                "ningbo",
                "--selection-json",
                str(selection),
                "--fixed20-cycles",
                "2",
                "--dry-run",
            ]
        )
