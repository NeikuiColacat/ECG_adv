"""CPU-only contracts for the single managed experiment launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from boot_scripts.run_experiment import (
    ExperimentPlan,
    execute_experiment,
    load_experiment_plan,
    main,
)
from util.run_record import verify_run_file_index
from util.evaluation.direct_baseline_selection import locked_pooled_selection_rule


REPO = Path(__file__).resolve().parents[2]


def _write_bundle(tmp_path: Path, *, run_dir: Path) -> Path:
    config_root = tmp_path / "configs"
    experiment_path = config_root / "experiments" / "fixture.yaml"
    entry_config_path = config_root / "train" / "PTBXL.yaml"
    seed_path = config_root / "random_seed.yaml"
    experiment_path.parent.mkdir(parents=True)
    entry_config_path.parent.mkdir(parents=True)
    seed_path.write_text(
        "schema_version: 1\nrandom_seed: 20260501\n",
        encoding="utf-8",
    )
    entry_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "references": {"random_seed_config": "random_seed.yaml"},
                "notes": {"historical_example": "not-a-runtime-config.yaml"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": {
                    "name": "launcher_fixture",
                    "purpose": "Exercise the public managed launcher contract.",
                },
                "entrypoint": {
                    "name": "train_ptbxl_effnet",
                    "config": "train/PTBXL.yaml",
                    "arguments": ["--epochs", "1", "--num-workers", "0"],
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
    return experiment_path


def _action_plan(
    tmp_path: Path,
    entrypoint_name: str,
    *,
    delegate_subdir: str = "training",
) -> tuple[Path, ExperimentPlan]:
    run_dir = tmp_path / "runs" / entrypoint_name
    experiment_path = _write_bundle(tmp_path, run_dir=run_dir)
    payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    payload["entrypoint"]["name"] = entrypoint_name
    payload["entrypoint"]["arguments"] = []
    payload["output"]["delegate_output_subdir"] = delegate_subdir
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return run_dir, load_experiment_plan(experiment_path)


def _typed_train_result(*, heldout: bool = False) -> dict:
    checkpoint = {"path": "checkpoints/last.pt", "sha256": "0" * 64}
    return {
        "config": {},
        "epochs_completed": 1,
        "model": {"name": "fixture"},
        "center": "ningbo",
        "method_id": "fixture_method",
        "scientific_arm": "fixture",
        "optimizer_steps": 1,
        "last_checkpoint": checkpoint,
        "selection": {
            "policy": "last",
            "selected_epoch": 1,
            "selected_checkpoint": checkpoint,
            "heldout_evaluation_used_for_selection": heldout,
        },
    }


def _pooled_selection(*, status: str = "selected") -> dict:
    selected = status == "selected"
    value = 0.5 if selected else None
    return {
        "schema_version": 1,
        "artifact_type": "direct_k500_pooled_epoch_selection",
        "status": status,
        "selection_rule": locked_pooled_selection_rule(),
        "comparison_identity": {
            "protocol_id": "fixture_protocol",
            "method_id": "direct_depth23_fixed20",
            "model_family": "efficientnet1dv2",
            "replicate_id": 0,
        },
        "config": {"protocol_id": "fixture_protocol"},
        "record_count": 400,
        "composition_count": 20,
        "centers": ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
        "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
        "selected_epoch": 1 if selected else None,
        "selected_score": value,
        "selected_clean_macro_auprc": value,
        "selected_clean_macro_auroc": value,
        "selected_robust_macro_auprc": value,
        "selected_robust_macro_auroc": value,
    }


def _record_payload(
    plan: ExperimentPlan,
    relative_name: str,
    payload: dict,
    *,
    delegate_exit_code: int = 0,
) -> tuple[int, dict]:
    def delegate(argv: list[str], log_path: Path) -> int:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / relative_name).write_text(json.dumps(payload), encoding="utf-8")
        log_path.write_text("fixture delegate\n", encoding="utf-8")
        return delegate_exit_code

    exit_code = execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=delegate,
    )
    manifest = json.loads(
        (plan.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    return exit_code, manifest


def test_dry_run_resolves_the_config_closure_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs" / "new"
    experiment_path = _write_bundle(tmp_path, run_dir=run_dir)

    assert main(["--config", str(experiment_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "dry_run"
    assert payload["entrypoint"]["name"] == "train_ptbxl_effnet"
    assert {Path(item["path"]).name for item in payload["config_closure"]} == {
        "fixture.yaml",
        "PTBXL.yaml",
        "random_seed.yaml",
    }
    assert payload["run_dir_collision"] is False
    assert payload["would_create_directory"] is True
    assert payload["would_fail_execution"] is False
    assert payload["execution_blocker"] is None
    assert payload["loads_data"] is False
    assert payload["loads_model_or_gpu"] is False
    assert payload["expected_result"] == {
        "path": "training/train_result.json",
        "type": "train_result",
    }
    assert not run_dir.exists()


def test_dry_run_reports_existing_run_dir_collision_without_mutating_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs" / "complete"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    experiment_path = _write_bundle(tmp_path / "bundle", run_dir=run_dir)

    assert main(["--config", str(experiment_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["run_dir_collision"] is True
    assert payload["would_create_directory"] is False
    assert payload["would_fail_execution"] is True
    assert payload["execution_blocker"] == "run_directory_exists"
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert {path.name for path in run_dir.iterdir()} == {"sentinel.txt"}


def test_existing_run_dir_is_still_fail_closed_for_real_execution(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "existing"
    run_dir.mkdir(parents=True)
    experiment_path = _write_bundle(tmp_path / "bundle", run_dir=run_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        load_experiment_plan(experiment_path)

    dry_run_plan = load_experiment_plan(
        experiment_path,
        allow_existing_run_dir=True,
    )

    def forbidden_delegate(argv: list[str], log_path: Path) -> int:
        del argv, log_path
        raise AssertionError("delegate must not run on a colliding output")

    with pytest.raises(FileExistsError, match="already exists"):
        execute_experiment(
            dry_run_plan,
            launcher_argv=["python", "boot_scripts/run_experiment.py"],
            delegate_runner=forbidden_delegate,
        )


def test_successful_execution_records_the_exact_delegate_and_file_index(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "fresh"
    experiment_path = _write_bundle(tmp_path, run_dir=run_dir)
    plan = load_experiment_plan(experiment_path)
    calls: list[list[str]] = []

    def fake_delegate(argv: list[str], log_path: Path) -> int:
        calls.append(argv)
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "train_result.json").write_text(
            json.dumps(
                {
                    "config": {}, "epochs_completed": 1,
                    "model": {"name": "fixture"}, "selected_epoch": 1,
                    "selected_checkpoint": {"path": "checkpoints/best.pt"},
                }
            ),
            encoding="utf-8",
        )
        log_path.write_text("fixture complete\n", encoding="utf-8")
        return 0

    launcher_argv = ["python", "boot_scripts/run_experiment.py", "--config", str(experiment_path)]
    assert execute_experiment(
        plan,
        launcher_argv=launcher_argv,
        delegate_runner=fake_delegate,
    ) == 0

    assert len(calls) == 1
    assert calls[0][1] == str((REPO / "boot_scripts" / "train_ptbxl_effnet.py").resolve())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["exact_launcher_argv"] == launcher_argv
    assert manifest["exact_delegate_argv"] == calls[0]
    assert manifest["delegate_result"]["path"] == "training/train_result.json"
    assert set(manifest["delegate_result"]) >= {"identity", "selection", "sha256"}
    assert verify_run_file_index(run_dir) == []


def test_exit_zero_without_declared_result_is_a_failed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-result"
    experiment_path = _write_bundle(tmp_path, run_dir=run_dir)
    plan = load_experiment_plan(experiment_path)

    def empty_delegate(argv: list[str], log_path: Path) -> int:
        del argv
        log_path.write_text("delegate returned zero without a result\n", encoding="utf-8")
        return 0

    assert execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=empty_delegate,
    ) == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["delegate_exit_code"] == 0
    assert manifest["exit_code"] == 1
    assert manifest["delegate_result"] is None
    assert "expected result is missing" in manifest["error"]
    assert verify_run_file_index(run_dir) == []


def test_typed_train_result_rejects_heldout_selection(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(tmp_path, "train_pn2021")
    exit_code, manifest = _record_payload(
        plan, "train_result.json", _typed_train_result(heldout=True)
    )
    assert exit_code == 1 and manifest["status"] == "failed"
    assert manifest["delegate_result"] is None
    assert "heldout-free last checkpoint" in manifest["error"]


@pytest.mark.parametrize("missing", ["center", "scientific_arm", "optimizer_steps"])
def test_typed_train_result_requires_identity(tmp_path: Path, missing: str) -> None:
    _, plan = _action_plan(tmp_path, "train_pn2021")
    payload = _typed_train_result()
    payload.pop(missing)
    exit_code, manifest = _record_payload(plan, "train_result.json", payload)
    assert exit_code == 1
    assert manifest["delegate_result"] is None


def test_bogus_pooled_selection_is_rejected(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(tmp_path, "select_pn2021_direct")
    payload = _pooled_selection()
    payload["selection_rule"] = {"heldout_evaluation_used": False}
    exit_code, manifest = _record_payload(
        plan, "direct_baseline_selection.json", payload
    )
    assert exit_code == 1
    assert manifest["delegate_result"] is None
    assert "pooled-selection identity" in manifest["error"]


def test_failed_clean_floor_result_is_validated_and_recorded(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(tmp_path, "select_pn2021_direct")
    exit_code, manifest = _record_payload(
        plan,
        "direct_baseline_selection.json",
        _pooled_selection(status="failed_clean_floor"),
        delegate_exit_code=3,
    )
    assert exit_code == 3
    assert manifest["status"] == "failed"
    assert manifest["delegate_exit_code"] == 3
    assert manifest["delegate_result"]["selection"]["status"] == "failed_clean_floor"


def test_bogus_evaluation_result_is_rejected(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(
        tmp_path, "evaluate_pn2021", delegate_subdir="evaluation"
    )

    payload = {"schema_version": 1, "status": "complete", "checkpoint": {},
               "clean": {}, "corrupted": {}, "protocol": {}}
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1
    assert manifest["delegate_result"] is None
    assert "schema_version=2" in manifest["error"]


def test_refit_requires_heldout_free_refit_contract(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(tmp_path, "refit_pn2021_direct")
    assert plan.expected_result_relative_path.as_posix() == "training/refit_contract.json"
    assert plan.expected_result_type == "refit_contract"

    payload = {
        "schema_version": 1,
        "artifact_type": "pn2021_direct_refit_contract",
        "model": {"name": "fixture"}, "center": "ningbo",
        "method": {"id": "direct_depth23_fixed20"},
        "selection": {"selected_epoch": 1, "heldout_evaluation_used": True},
        "result": _typed_train_result(),
    }
    exit_code, manifest = _record_payload(plan, "refit_contract.json", payload)
    assert exit_code == 1
    assert manifest["delegate_result"] is None
    assert "heldout-free Direct refit contract" in manifest["error"]


def test_internal_absolute_yaml_reference_is_rejected(tmp_path: Path) -> None:
    experiment_path = _write_bundle(tmp_path, run_dir=tmp_path / "runs" / "new")
    entry_config = tmp_path / "configs" / "train" / "PTBXL.yaml"
    payload = yaml.safe_load(entry_config.read_text(encoding="utf-8"))
    payload["references"]["random_seed_config"] = str(
        (tmp_path / "configs" / "random_seed.yaml").resolve()
    )
    entry_config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be relative to config bundle"):
        load_experiment_plan(experiment_path)


def test_all_tracked_experiment_jobs_resolve_with_explicit_results(
    tmp_path: Path,
) -> None:
    experiment_paths = sorted((REPO / "configs" / "experiments").glob("*.yaml"))
    assert len(experiment_paths) == 59
    for experiment_path in experiment_paths:
        plan = load_experiment_plan(
            experiment_path,
            run_dir=tmp_path / experiment_path.stem,
        )
        assert plan.expected_result_relative_path.parts[0] in {"training", "evaluation"}
        assert plan.expected_result_type in {
            "train_result",
            "evaluation_result",
            "pooled_selection",
            "refit_contract",
        }


def test_mainline_closure_includes_stage1_augmix_config(tmp_path: Path) -> None:
    experiment_path = (
        REPO
        / "configs"
        / "experiments"
        / "manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml"
    )
    plan = load_experiment_plan(experiment_path, run_dir=tmp_path / "mainline")
    closure = {path.relative_to(plan.config_root).as_posix() for path, _ in plan.config_sources}
    assert "train/augmix.yaml" in closure


def test_launcher_rejects_owned_and_removed_delegate_flags(tmp_path: Path) -> None:
    experiment_path = _write_bundle(tmp_path, run_dir=tmp_path / "runs" / "new")
    payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))

    payload["entrypoint"]["arguments"] = ["--output-dir", "/tmp/bypass"]
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="launcher-owned"):
        load_experiment_plan(experiment_path)

    payload["entrypoint"]["arguments"] = ["--fixed20-cycles", "2"]
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="removed protocol"):
        load_experiment_plan(experiment_path)
