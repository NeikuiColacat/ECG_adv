"""CPU-only contracts for the single managed experiment launcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from data_preprocess import data_ledger
from boot_scripts.run_experiment import (
    ExperimentPlan,
    execute_experiment,
    load_experiment_plan,
    main,
)
from models.contracts import EFFICIENTNET1DV2_SPEC
from util.run_record import verify_run_file_index


REPO = Path(__file__).resolve().parents[2]
FIXTURE_CHECKPOINT = b"fixture checkpoint\n"
FIXTURE_CHECKPOINT_SHA256 = hashlib.sha256(FIXTURE_CHECKPOINT).hexdigest()


def _write_bundle(tmp_path: Path, *, run_dir: Path) -> Path:
    config_root = tmp_path / "configs"
    experiment_path = config_root / "experiments" / "fixture.yaml"
    entry_config_path = config_root / "train" / "PTBXL.yaml"
    seed_path = config_root / "random_seed.yaml"
    experiment_path.parent.mkdir(parents=True)
    entry_config_path.parent.mkdir(parents=True)
    (config_root / "data").mkdir()
    (config_root / "augmentation").mkdir()
    roots = {}
    for name in data_ledger.ROOT_NAMES:
        root = tmp_path / "data_roots" / name
        root.mkdir(parents=True)
        (root / "member.bin").write_bytes(name.encode())
        roots[name] = root
    (config_root / "data" / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "ptbxl": {"cache_dir": str(roots["ptbxl_cache"])},
                "pn2021": {"cache_dir": str(roots["pn2021_cache"])},
                "output": {"root_dir": str(roots["split_artifacts"])},
            }
        ),
        encoding="utf-8",
    )
    (config_root / "augmentation" / "cache.yaml").write_text(
        yaml.safe_dump({"output": {"cache_dir": str(roots["pn2021c_cache"])}}),
        encoding="utf-8",
    )
    ledger_path = config_root / "data" / "content_ledger.jsonl"
    ledger = data_ledger.generate_ledger(roots, ledger_path)
    (config_root / "data" / "data_load.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "content_ledger": {
                "path": "data/content_ledger.jsonl",
                "sha256": ledger["ledger_sha256"],
            }}
        ),
        encoding="utf-8",
    )
    seed_path.write_text(
        "schema_version: 1\nrandom_seed: 20260501\n",
        encoding="utf-8",
    )
    entry_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "references": {
                    "random_seed_config": "random_seed.yaml",
                    "split_config": "data/splits.yaml",
                    "data_load_config": "data/data_load.yaml",
                    "corruption_cache_config": "augmentation/cache.yaml",
                },
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


def _training_lineage() -> dict:
    model = EFFICIENTNET1DV2_SPEC.describe()
    return {
        "schema_version": 1, "scope": "pn2021_k500_center_adaptation",
        "model": {"name": EFFICIENTNET1DV2_SPEC.name, "spec": model}, "center": "ningbo",
        "method": {"recipe_id": "fixture", "scientific_arm": "fixture",
                   "recipe_spec_sha256": "0" * 64, "implementation_identity": "fixture",
                   "recipe_version": 1, "kind": "fixture", "auxiliary_variant": "fixture",
                   "schema_version": 1},
        "comparison": {"group": "fixture", "replicate_id": 0},
        "seed": {"base_seed": 0, "effective_seed": 0, "namespace": "fixture",
                 "config_sha256": "0" * 64},
        "source_checkpoint": {"sha256": "f" * 64},
        "training_config_sha256": "0" * 64,
        "adaptation_data": {"dataset": "pn2021", "partition": "k500",
            "logical_center": "ningbo", "source_centers": ["ningbo"], "record_count": 500,
            "split_id": "fixture", "hash_id_set_sha256": "0" * 64,
            "split_manifest_sha256": "0" * 64, "source_manifest_sha256": "0" * 64,
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51", "class_order": ["CD", "HYP", "MI", "NORM", "STTC"]},
        "selection": {"policy": "last", "heldout_evaluation_used_for_selection": False},
    }


def _typed_train_result(*, heldout: bool = False) -> dict:
    checkpoint = {
        "path": "checkpoints/last.pt",
        "sha256": FIXTURE_CHECKPOINT_SHA256,
    }
    return {
        "schema_version": 1,
        "artifact_type": "pn2021_train_result",
        "lineage": _training_lineage(),
        "config": {"training": {"sha256": "0" * 64}},
        "epochs_completed": 1,
        "model": {"spec": EFFICIENTNET1DV2_SPEC.describe()},
        "center": "ningbo",
        "method_id": "fixture",
        "scientific_arm": "fixture",
        "optimizer_steps": 1,
        "last_checkpoint": checkpoint,
        "seed": {"base_seed": 0, "effective_seed": 0, "namespace": "fixture",
                 "config_sha256": "0" * 64},
        "selection": {
            "policy": "last",
            "selected_epoch": 1,
            "selected_checkpoint": checkpoint,
            "heldout_evaluation_used_for_selection": heldout,
        },
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
        checkpoint = payload.get("last_checkpoint")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("path"), str):
            raw_checkpoint = Path(checkpoint["path"])
            checkpoint_path = (
                raw_checkpoint
                if raw_checkpoint.is_absolute()
                else output_dir / raw_checkpoint
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(FIXTURE_CHECKPOINT)
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "new"
    experiment_path = _write_bundle(tmp_path, run_dir=run_dir)
    data_load = tmp_path / "configs" / "data" / "data_load.yaml"
    ledger_path = tmp_path / "configs" / "data" / "content_ledger.jsonl"
    ledger_path.unlink()
    monkeypatch.setattr(data_ledger, "load_roots", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(data_ledger, "verify_ledger", lambda *args, **kwargs: 1 / 0)

    assert main(["--config", str(experiment_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "dry_run"
    assert payload["entrypoint"]["name"] == "train_ptbxl_effnet"
    assert {Path(item["path"]).name for item in payload["config_closure"]} == {
        "fixture.yaml",
        "PTBXL.yaml",
        "random_seed.yaml",
        "splits.yaml",
        "data_load.yaml",
        "cache.yaml",
    }
    assert payload["data_content"] == {
        "path": str(ledger_path),
        "config_relative_path": "data/content_ledger.jsonl",
        "expected_sha256": yaml.safe_load(data_load.read_text())["content_ledger"]["sha256"],
        "required_roots": ["ptbxl_cache", "split_artifacts"],
        "verification_mode": "deferred_quick_inventory_size",
        "reads_ledger": False,
        "stats_data": False,
        "config_closure_sha256": {
            item["path"]: item["sha256"] for item in payload["config_closure"]
        },
    }
    assert payload["run_dir_collision"] is False
    assert payload["would_create_directory"] is True
    assert payload["would_fail_execution"] is False
    assert payload["execution_blocker"] is None
    assert payload["loads_data"] is False
    assert payload["loads_model_or_gpu"] is False
    assert payload["expected_result"] == {
        "path": "training/train_result.json",
        "type": "supervised_train_result",
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
    assert calls[0][calls[0].index("--config") + 1] == str(
        run_dir / "configs" / "train" / "PTBXL.yaml"
    )
    assert calls[0][calls[0].index("--config-root") + 1] == str(
        run_dir / "configs"
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["exact_launcher_argv"] == launcher_argv
    assert manifest["exact_delegate_argv"] == calls[0]
    assert manifest["delegate_result"]["path"] == "training/train_result.json"
    assert set(manifest["delegate_result"]) >= {"identity", "selection", "sha256"}
    data_content = manifest["data_content"]
    snapshot = run_dir / data_content["ledger_snapshot"]
    config_snapshot = run_dir / data_content["config_ledger_snapshot"]
    assert snapshot.read_bytes() == plan.data_ledger.path.read_bytes()
    assert config_snapshot.read_bytes() == snapshot.read_bytes()
    assert data_content["expected_ledger_sha256"] == data_content["verified_ledger_sha256"]
    assert data_content["required_roots"] == ["ptbxl_cache", "split_artifacts"]
    assert data_content["execution_verification"]["mode"] == "quick_exact_inventory_and_size"
    file_index = json.loads((run_dir / "run_file_index.json").read_text())
    assert any(item["path"] == "manifests/data_content_ledger.jsonl"
               and item["role"] == "data_content_ledger_snapshot"
               for item in file_index["files"])
    assert verify_run_file_index(run_dir) == []


@pytest.mark.parametrize(
    ("relative", "delete"),
    (
        ("manifests/data_content_ledger.jsonl", False),
        ("manifests/data_content_ledger.jsonl", True),
        ("configs/data/content_ledger.jsonl", False),
        ("configs/data/content_ledger.jsonl", True),
    ),
)
def test_delegate_cannot_tamper_with_the_data_ledger_snapshot(
    tmp_path: Path, relative: str, delete: bool
) -> None:
    case = ("delete" if delete else "rewrite") + "_" + Path(relative).parent.name
    run_dir = tmp_path / "runs" / case
    plan = load_experiment_plan(_write_bundle(tmp_path / case, run_dir=run_dir))

    def delegate(argv: list[str], log_path: Path) -> int:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "train_result.json").write_text(
            json.dumps({"config": {}, "epochs_completed": 1,
                        "model": {"name": "fixture"}, "selected_epoch": 1,
                        "selected_checkpoint": {"path": "checkpoints/best.pt"}})
        )
        snapshot = run_dir / relative
        snapshot.unlink() if delete else snapshot.write_bytes(b"tampered\n")
        log_path.write_text("tampered ledger snapshot\n")
        return 0

    assert execute_experiment(
        plan, launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=delegate,
    ) == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "data content ledger snapshot" in manifest["error"]


def test_data_verification_failure_precedes_run_directory_and_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "blocked"
    plan = load_experiment_plan(_write_bundle(tmp_path, run_dir=run_dir))
    calls = []

    def fail(*args: object, **kwargs: object) -> dict:
        assert not run_dir.exists()
        raise ValueError("fixture ledger drift")

    monkeypatch.setattr(data_ledger, "verify_ledger", fail)
    with pytest.raises(ValueError, match="ledger drift"):
        execute_experiment(
            plan,
            launcher_argv=["python", "boot_scripts/run_experiment.py"],
            delegate_runner=lambda argv, log: calls.append((argv, log)) or 0,
        )
    assert calls == []
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "relative", ("data/data_load.yaml", "data/splits.yaml", "train/PTBXL.yaml")
)
def test_data_config_closure_drift_is_rejected_before_run_directory(
    tmp_path: Path, relative: str
) -> None:
    run_dir = tmp_path / "runs" / Path(relative).stem
    plan = load_experiment_plan(_write_bundle(tmp_path, run_dir=run_dir))
    source = tmp_path / "configs" / relative
    source.write_text(source.read_text() + "\n# drift after plan resolution\n")

    with pytest.raises(ValueError, match="config closure changed"):
        execute_experiment(
            plan, launcher_argv=["python", "boot_scripts/run_experiment.py"],
            delegate_runner=lambda argv, log: 0,
        )
    assert not run_dir.exists()


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
    assert "identity differs" in manifest["error"]


def test_typed_train_result_records_lineage_and_verifies_checkpoint(
    tmp_path: Path,
) -> None:
    _, plan = _action_plan(tmp_path, "train_pn2021")
    payload = _typed_train_result()
    exit_code, manifest = _record_payload(plan, "train_result.json", payload)

    assert exit_code == 0
    assert manifest["delegate_result"]["identity"]["lineage"] == payload["lineage"]


@pytest.mark.parametrize(
    ("checkpoint_path", "checkpoint_sha256", "message"),
    (
        ("../outside.pt", FIXTURE_CHECKPOINT_SHA256, "same training output"),
        ("checkpoints/last.pt", "f" * 64, "SHA256 mismatch"),
    ),
)
def test_typed_train_result_verifies_checkpoint_artifact(
    tmp_path: Path, checkpoint_path: str, checkpoint_sha256: str, message: str
) -> None:
    _, plan = _action_plan(tmp_path, "train_pn2021")
    payload = _typed_train_result()
    checkpoint = {"path": checkpoint_path, "sha256": checkpoint_sha256}
    payload["last_checkpoint"] = checkpoint
    payload["selection"]["selected_checkpoint"] = checkpoint
    exit_code, manifest = _record_payload(plan, "train_result.json", payload)
    assert exit_code == 1
    assert message in manifest["error"]


@pytest.mark.parametrize(
    "missing",
    [
        "schema_version",
        "artifact_type",
        "lineage",
        "method_id",
        "center",
        "scientific_arm",
        "optimizer_steps",
    ],
)
def test_typed_train_result_requires_identity(tmp_path: Path, missing: str) -> None:
    _, plan = _action_plan(tmp_path, "train_pn2021")
    payload = _typed_train_result()
    payload.pop(missing)
    exit_code, manifest = _record_payload(plan, "train_result.json", payload)
    assert exit_code == 1
    assert manifest["delegate_result"] is None


def test_typed_train_result_rejects_invalid_canonical_lineage(tmp_path: Path) -> None:
    _, plan = _action_plan(tmp_path, "train_pn2021")
    payload = _typed_train_result()
    payload["lineage"]["seed"] = None
    exit_code, manifest = _record_payload(plan, "train_result.json", payload)
    assert exit_code == 1 and manifest["delegate_result"] is None


def test_matrix_result_is_registered(tmp_path: Path) -> None:
    _, plan = _action_plan(tmp_path, "aggregate_pn2021", delegate_subdir="evaluation")
    assert plan.expected_result_relative_path == Path("evaluation/matrix_result.json")
    assert plan.expected_result_type == "pn2021_matrix_result"
    assert plan.data_ledger is None


@pytest.mark.parametrize(
    ("entrypoint", "required_roots"),
    (
        ("train_ptbxl_effnet", ("ptbxl_cache", "split_artifacts")),
        ("train_ptbxl_ecgfounder", ("ptbxl_cache", "split_artifacts")),
        ("train_pn2021", ("pn2021_cache", "split_artifacts")),
        (
            "evaluate_pn2021",
            ("pn2021_cache", "pn2021c_cache", "split_artifacts"),
        ),
    ),
)
def test_entrypoints_own_their_exact_data_roots(
    tmp_path: Path, entrypoint: str, required_roots: tuple[str, ...]
) -> None:
    _, plan = _action_plan(tmp_path, entrypoint)
    assert plan.data_ledger is not None
    assert plan.data_ledger.required_roots == required_roots
    assert plan.data_ledger.path == (
        tmp_path / "configs" / "data" / "content_ledger.jsonl"
    ).resolve()
    closure_paths = {path for path, _ in plan.data_ledger.closure_sha256}
    if entrypoint == "evaluate_pn2021":
        assert tmp_path / "configs" / "augmentation" / "cache.yaml" in closure_paths


def test_aggregate_bypasses_data_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, plan = _action_plan(
        tmp_path, "aggregate_pn2021", delegate_subdir="evaluation"
    )
    monkeypatch.setattr(data_ledger, "load_roots", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(data_ledger, "verify_ledger", lambda *args, **kwargs: 1 / 0)
    calls = []

    def failed_delegate(argv: list[str], log_path: Path) -> int:
        calls.append(argv)
        log_path.write_text("expected fixture failure\n", encoding="utf-8")
        return 2

    assert execute_experiment(
        plan,
        launcher_argv=["python", "boot_scripts/run_experiment.py"],
        delegate_runner=failed_delegate,
    ) == 2
    assert len(calls) == 1
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["data_content"] is None


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
    assert len(experiment_paths) == 40
    for experiment_path in experiment_paths:
        plan = load_experiment_plan(
            experiment_path,
            run_dir=tmp_path / experiment_path.stem,
        )
        assert plan.expected_result_relative_path.parts[0] in {"training", "evaluation"}
        assert plan.expected_result_type in {
            "supervised_train_result",
            "pn2021_train_result",
            "evaluation_result",
            "pn2021_matrix_result",
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


def test_launcher_snapshots_source_registry_but_not_train_result(tmp_path: Path) -> None:
    experiment_path = _write_bundle(tmp_path, run_dir=tmp_path / "runs" / "new")
    registry = tmp_path / "configs" / "baselines" / "source.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    payload["entrypoint"]["name"] = "evaluate_pn2021"
    payload["entrypoint"]["arguments"] = [
        "--source-registry",
        "baselines/source.yaml",
        "--train-result",
        str(tmp_path / "external" / "train_result.json"),
    ]
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    plan = load_experiment_plan(experiment_path)
    closure = {path.relative_to(plan.config_root).as_posix() for path, _ in plan.config_sources}
    assert "baselines/source.yaml" in closure
    assert all(path.suffix in {".yaml", ".yml"} for path, _ in plan.config_sources)


def test_launcher_rejects_owned_delegate_flags(tmp_path: Path) -> None:
    experiment_path = _write_bundle(tmp_path, run_dir=tmp_path / "runs" / "new")
    payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))

    payload["entrypoint"]["arguments"] = ["--output-dir", "/tmp/bypass"]
    experiment_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="launcher-owned"):
        load_experiment_plan(experiment_path)
