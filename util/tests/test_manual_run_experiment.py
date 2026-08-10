"""CPU-only contracts for the single managed experiment launcher."""

from __future__ import annotations

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
        (output_dir / "result.json").write_text(
            json.dumps({"status": "fixture"}),
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
    assert verify_run_file_index(run_dir) == []


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
