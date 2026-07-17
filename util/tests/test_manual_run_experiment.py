from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from boot_scripts.run_experiment import (
    execute_experiment,
    load_experiment_plan,
    main,
)
from util.run_record import verify_run_file_index


REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path, *, run_dir: Path | None = None) -> Path:
    root = tmp_path / "strategy_configs"
    experiment = root / "experiments" / "manual_test.yaml"
    train = root / "train" / "PTBXL.yaml"
    seed = root / "random_seed.yaml"
    experiment.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    seed.write_text(
        yaml.safe_dump({"schema_version": 1, "random_seed": 20260501}),
        encoding="utf-8",
    )
    train.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "references": {"random_seed_config": "random_seed.yaml"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    experiment.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": {
                    "name": "manual_test",
                    "purpose": "Exercise the whitelist managed launcher.",
                },
                "entrypoint": {
                    "name": "train_ptbxl_effnet",
                    "config": "train/PTBXL.yaml",
                    "arguments": ["--epochs", "1", "--num-workers", "0"],
                },
                "output": {
                    "run_dir": str(run_dir or (tmp_path / "runs" / "run-1")),
                    "if_exists": "error",
                    "delegate_output_subdir": "training",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return experiment


def test_dry_run_is_side_effect_free_and_resolves_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    experiment = _write_bundle(tmp_path)
    run_dir = tmp_path / "override" / "dry-run"

    assert main(
        [
            "--config",
            str(experiment),
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "dry_run"
    assert payload["experiment_name"] == "manual_test"
    assert payload["config_root"] == str(experiment.parents[1].resolve())
    assert payload["entrypoint"]["name"] == "train_ptbxl_effnet"
    assert payload["entrypoint"]["config_path"] == str(
        (experiment.parents[1] / "train" / "PTBXL.yaml").resolve()
    )
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["would_create_directory"] is True
    assert not run_dir.exists()


def test_execute_delegates_and_writes_integrity_checked_run_record(
    tmp_path: Path,
) -> None:
    experiment = _write_bundle(tmp_path)
    plan = load_experiment_plan(experiment)
    calls: list[list[str]] = []

    def fake_delegate(argv: list[str], log_path: Path) -> int:
        calls.append(argv)
        delegate_output = Path(argv[argv.index("--output-dir") + 1])
        delegate_output.mkdir(parents=True)
        (delegate_output / "data_manifest.json").write_text(
            json.dumps({"schema_version": 1, "dataset": "fixture"}),
            encoding="utf-8",
        )
        (delegate_output / "selection.json").write_text(
            json.dumps({"schema_version": 1, "selected": "fixture"}),
            encoding="utf-8",
        )
        log_path.write_text("fake delegate completed\n", encoding="utf-8")
        return 0

    exit_code = execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py", "--config", str(experiment)],
        delegate_runner=fake_delegate,
    )

    assert exit_code == 0
    assert len(calls) == 1
    delegated = calls[0]
    assert delegated[0].endswith("python")
    assert delegated[1] == str((REPO / "boot_scripts" / "train_ptbxl_effnet.py").resolve())
    assert delegated[delegated.index("--config") + 1] == str(plan.entry_config_path)
    assert delegated[delegated.index("--config-root") + 1] == str(plan.config_root)
    assert delegated[delegated.index("--output-dir") + 1] == str(
        plan.run_dir / "training"
    )

    required = {
        "run_card.json",
        "run_manifest.json",
        "run_file_index.json",
        "summary.md",
    }
    assert required <= {path.name for path in plan.run_dir.iterdir()}
    manifest = json.loads(
        (plan.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    card = json.loads((plan.run_dir / "run_card.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["exit_code"] == 0
    assert manifest["exact_launcher_argv"] == [
        "python",
        "boot_scripts/run_experiment.py",
        "--config",
        str(experiment),
    ]
    assert manifest["exact_delegate_argv"] == delegated
    assert manifest["cwd"] == str(REPO)
    assert manifest["environment"]["python_executable"]
    assert manifest["environment"]["python_version"]
    assert manifest["environment"]["platform"]
    assert manifest["environment"]["torch_version"]
    assert {"commit", "branch", "dirty", "dirty_diff_sha256"} <= set(
        manifest["git"]
    )
    assert card["status"] == "complete"

    snapshots = manifest["config_snapshots"]
    assert len(snapshots) == 3
    for item in snapshots:
        snapshot = plan.run_dir / item["snapshot_path"]
        assert snapshot.is_file()
        assert item["sha256"] == _sha256(snapshot)
    index_path = plan.run_dir / "run_file_index.json"
    assert manifest["run_file_index_sha256"] == _sha256(index_path)
    indexed_files = {
        item["path"]
        for item in json.loads(index_path.read_text(encoding="utf-8"))["files"]
    }
    assert "training/data_manifest.json" in indexed_files
    assert "training/selection.json" in indexed_files
    assert verify_run_file_index(plan.run_dir) == []

    (plan.run_dir / "summary.md").write_text("tampered\n", encoding="utf-8")
    assert any("summary.md" in error for error in verify_run_file_index(plan.run_dir))


def test_run_record_uses_typed_method_identity_and_last_selection(
    tmp_path: Path,
) -> None:
    experiment = _write_bundle(tmp_path)
    plan = load_experiment_plan(experiment)

    def fake_delegate(argv: list[str], log_path: Path) -> int:
        delegate_output = Path(argv[argv.index("--output-dir") + 1])
        checkpoint = delegate_output / "checkpoints" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"typed-method-checkpoint")
        (delegate_output / "train_result.json").write_text(
            json.dumps(
                {
                    "method_id": "a5_lhat_threechain_v1",
                    "scientific_arm": "A5",
                    "center": "ningbo",
                    "epochs_completed": 3,
                    "optimizer_steps": 6,
                    "last_checkpoint": {
                        "path": str(checkpoint),
                        "sha256": _sha256(checkpoint),
                    },
                    "selection": {
                        "policy": "last",
                        "selected_epoch": 3,
                        "selected_metric": None,
                        "selected_checkpoint": {
                            "path": str(checkpoint),
                            "sha256": _sha256(checkpoint),
                        },
                        "heldout_evaluation_used_for_selection": False,
                    },
                    "model": {"name": "efficientnet1dv2"},
                    "config": {"method": {"profile_sha256": "a" * 64}},
                    "seed": {"effective_seed": 7},
                    "latent_pool": {"identity_sha256": "b" * 64},
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("typed method completed\n", encoding="utf-8")
        return 0

    assert execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=fake_delegate,
    ) == 0

    data_manifest = json.loads(
        (plan.run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (plan.run_dir / "selection.json").read_text(encoding="utf-8")
    )
    assert data_manifest["method_id"] == "a5_lhat_threechain_v1"
    assert data_manifest["scientific_arm"] == "A5"
    assert "arm" not in data_manifest
    assert selection["policy"] == "last"
    assert selection["selected_epoch"] == 3
    assert selection["selected_checkpoint"]["sha256"] == _sha256(
        plan.run_dir / "training" / "checkpoints" / "last.pt"
    )


def test_run_record_extracts_direct_pooled_selection(tmp_path: Path) -> None:
    experiment = _write_bundle(tmp_path)
    plan = load_experiment_plan(experiment)

    def fake_delegate(argv: list[str], log_path: Path) -> int:
        delegate_output = Path(argv[argv.index("--output-dir") + 1])
        delegate_output.mkdir(parents=True)
        (delegate_output / "direct_baseline_selection.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "direct_k500_pooled_epoch_selection",
                    "status": "selected",
                    "selection_rule": {
                        "heldout_evaluation_used": False,
                        "robust_aggregation": "mean_of_20_composition_macro_auprc",
                        "score": "0.5_clean_macro_auprc_plus_0.5_robust_macro_auprc",
                    },
                    "selected_epoch": 7,
                    "selected_score": 0.61,
                    "selected_clean_macro_auprc": 0.70,
                    "selected_robust_macro_auprc": 0.52,
                    "clean_floor": 0.69,
                    "record_count": 400,
                    "composition_count": 20,
                    "composition_ids": [f"composition_{index:02d}" for index in range(20)],
                    "centers": [
                        "ningbo",
                        "chapman_shaoxing",
                        "cpsc_2018",
                        "georgia",
                    ],
                    "config": {
                        "protocol_id": "pn2021_direct_family_balanced_tuning"
                    },
                    "comparison_identity": {
                        "protocol_id": "pn2021_direct_family_balanced_tuning",
                        "method_id": "direct_depth23_fixed20",
                        "model_family": "efficientnet1dv2",
                    },
                    "frozen_corruption_identity_sha256": "fixture-cache-sha",
                    "per_center_validation_identity": {},
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("selector completed\n", encoding="utf-8")
        return 0

    assert execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=fake_delegate,
    ) == 0

    selection = json.loads(
        (plan.run_dir / "selection.json").read_text(encoding="utf-8")
    )
    data_manifest = json.loads(
        (plan.run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    assert selection["policy"] == "four_center_pooled_clean_robust_family_balanced"
    assert selection["selected_epoch"] == 7
    assert selection["selected_score"] == 0.61
    assert selection["selected_clean_macro_auprc"] == 0.70
    assert selection["selected_robust_macro_auprc"] == 0.52
    assert selection["heldout_evaluation_used_for_selection"] is False
    assert data_manifest["record_count"] == 400
    assert data_manifest["composition_count"] == 20


def test_existing_or_worktree_output_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "runs" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        load_experiment_plan(_write_bundle(tmp_path / "existing-case", run_dir=existing))

    inside_worktree = REPO / ".forbidden_manual_run"
    with pytest.raises(ValueError, match="outside the Git worktree"):
        load_experiment_plan(
            _write_bundle(tmp_path / "inside-case", run_dir=inside_worktree)
        )
    assert not inside_worktree.exists()


def test_launcher_rejects_owned_flags_and_legacy_imports(tmp_path: Path) -> None:
    experiment = _write_bundle(tmp_path)
    payload = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    payload["entrypoint"]["arguments"] = ["--output-dir", "/tmp/bypass"]
    experiment.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="launcher-owned"):
        load_experiment_plan(experiment)

    payload["entrypoint"]["arguments"] = ["--fixed20-cycles", "2"]
    experiment.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="removed protocol"):
        load_experiment_plan(experiment)

    for relative in ("util/run_record.py", "boot_scripts/run_experiment.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "ecg_adv_gen" not in source
        assert "scripts.run_experiment" not in source


@pytest.mark.parametrize(
    ("entrypoint_name", "config_relative", "arguments", "script_name"),
    [
        (
            "train_pn2021",
            "train/PN2021.yaml",
            [
                "--model",
                "efficientnet1dv2",
                "--method-config",
                "train/methods/a0_clean_v1.yaml",
                "--center",
                "ningbo",
                "--source-checkpoint",
                "source.pt",
            ],
            "train_pn2021.py",
        ),
        (
            "evaluate_pn2021",
            "eval/PN2021.yaml",
            [
                "--model",
                "efficientnet1dv2",
                "--checkpoint",
                "checkpoint.pt",
            ],
            "evaluate_pn2021.py",
        ),
    ],
)
def test_pn2021_entrypoint_plan_is_allowlisted_without_execution(
    tmp_path: Path,
    entrypoint_name: str,
    config_relative: str,
    arguments: list[str],
    script_name: str,
) -> None:
    config_root = tmp_path / "pn2021_configs"
    experiment_path = config_root / "experiments" / f"{entrypoint_name}.yaml"
    entry_config_path = config_root / config_relative
    experiment_path.parent.mkdir(parents=True)
    entry_config_path.parent.mkdir(parents=True)
    entry_config_path.write_text("schema_version: 1\n", encoding="utf-8")
    if entrypoint_name == "train_pn2021":
        method_path = config_root / "train" / "methods" / "a0_clean_v1.yaml"
        method_path.parent.mkdir(parents=True)
        method_path.write_text("schema_version: 1\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / entrypoint_name
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": {
                    "name": f"fixture_{entrypoint_name}",
                    "purpose": "Validate a PN2021 managed entrypoint plan only.",
                },
                "entrypoint": {
                    "name": entrypoint_name,
                    "config": config_relative,
                    "arguments": arguments,
                },
                "output": {
                    "run_dir": str(run_dir),
                    "if_exists": "error",
                    "delegate_output_subdir": "training",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = load_experiment_plan(experiment_path)

    assert plan.entrypoint_name == entrypoint_name
    assert plan.entrypoint_path == (REPO / "boot_scripts" / script_name).resolve()
    assert plan.entry_config_path == entry_config_path.resolve()
    assert plan.entry_arguments == tuple(arguments)
    assert plan.delegate_argv()[1] == str(plan.entrypoint_path)
    assert plan.delegate_argv()[-len(arguments) :] == arguments
    assert not run_dir.exists()
