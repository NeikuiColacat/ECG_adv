from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import boot_scripts.select_pn2021_direct as select_module
from boot_scripts.refit_pn2021_direct import main as refit_main
from boot_scripts.run_experiment import load_experiment_plan
from boot_scripts.select_pn2021_direct import main as select_main
from boot_scripts.tune_pn2021_direct import main as tune_main


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021_direct_tune.yaml"
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")


def _selection(
    tmp_path: Path,
) -> Path:
    source = tmp_path / "source.pt"
    source.write_bytes(b"locked-source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    comparison = {
        "protocol_id": "pn2021_direct_family_balanced_tuning",
        "method_id": "direct_depth23_fixed20",
        "comparison_group": "pn2021_direct_depth23_fixed20_family_balanced",
        "model_family": "efficientnet1dv2",
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
        "selection_rule": {
            "training_partition": "k500_tune_train",
            "validation_partition": "k500_tune_validation",
            "clean_aggregation": "concatenate_four_centers_before_metric",
            "corrupted_aggregation": "concatenate_four_centers_per_composition_before_metric",
            "robust_aggregation": "mean_of_20_composition_macro_auprc",
            "score": "0.5_clean_macro_auprc_plus_0.5_robust_macro_auprc",
            "clean_floor": "same_backbone_locked_clean_macro_auprc_minus_0.01",
            "metric_definition": "sklearn_average_precision",
            "input_type": "raw_logits",
            "strict_all_five_classes": True,
            "tie_break": "earliest_epoch_on_exact_tie",
            "heldout_evaluation_used": False,
        },
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
    selection_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_root / "run_file_index.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "training/selection.json",
                        "sha256": selection_sha,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_root / "run_card.json").write_text(
        json.dumps({"status": "complete", "exit_code": 0}),
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
    assert not output.exists()


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
