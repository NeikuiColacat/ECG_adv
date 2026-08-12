"""CPU-only contracts for the single managed experiment launcher."""

from __future__ import annotations

import hashlib
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


def _legacy_protocol() -> dict:
    return {
        "partition": "k500", "use_all_k500": True, "validation_split": False,
        "checkpoint_selection": "last", "heldout_ref_exclusion_required": True,
        "merge_cpsc_2018_extra_into_cpsc_2018": True,
        "centers": ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"],
        "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
        "mapping_hash": "555ec85d5b51",
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
        "config": {"training": {"sha256": "0" * 64,
                                  "resolved": {"protocol": _legacy_protocol()}}},
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


def _metric_body() -> dict:
    return {"metrics": {"drop_all_zero": {"macro_auroc": 0.5}},
            "identity": {"record_count": 1}}


def _evaluation_result() -> dict:
    model = EFFICIENTNET1DV2_SPEC.describe()
    lineage = _training_lineage()
    return {
        "schema_version": 3,
        "artifact_type": "pn2021_evaluation_result",
        "status": "complete",
        "model": model,
        "checkpoint": {"path": "checkpoints/last.pt", "sha256": FIXTURE_CHECKPOINT_SHA256},
        "subject": {
            "mode": "prospective_train_result",
            "train_result": {"path": "train_result.json", "sha256": "pending"},
            "lineage": lineage,
        },
        "clean": {"per_center": {"ningbo": _metric_body()}},
        "corrupted": {
            "per_view": [{"view_index": index, "per_center": {"ningbo": _metric_body()}}
                         for index in range(20)],
            "aggregates": {key: {"view_count": count, "center_order": ["ningbo"],
                                 "evaluated_center_mean": {"drop_all_zero": {}}}
                           for key, count in (("depth2", 10), ("depth3", 10), ("depth23", 20))},
        },
        "protocol": {
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "logical_centers": ["ningbo"],
        },
    }


def _matrix_result(tmp_path: Path) -> dict:
    centers = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
    mean = {"drop_all_zero": {"macro_auroc": 0.5}}
    config = tmp_path / "matrix.yaml"; config.write_text("schema_version: 1\n", encoding="utf-8")
    members = []
    for center in centers:
        artifact = tmp_path / f"{center}_evaluation_result.json"
        artifact.write_text(json.dumps({"schema_version": 3, "artifact_type":
            "pn2021_evaluation_result", "status": "complete"}), encoding="utf-8")
        members.append({
            "center": center,
            "evaluation_result": {"path": str(artifact),
                                  "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
            "train_result": {"path": f"{center}_train_result.json", "sha256": "0" * 64},
            "checkpoint": {"sha256": "0" * 64},
            "adaptation_data": {"split_id": center},
        })
    return {
        "schema_version": 1,
        "artifact_type": "pn2021_diagonal_four_center_evaluation",
        "status": "complete",
        "profile_name": "fixture_matrix",
        "config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "model": EFFICIENTNET1DV2_SPEC.describe(),
        "cohort": {
            "model": EFFICIENTNET1DV2_SPEC.describe(),
            "method": {"recipe_id": "fixture"},
            "comparison": {"group": "fixture", "replicate_id": 0},
            "source_checkpoint": {"sha256": "0" * 64}, "selection": {"policy": "last"},
            "seed": {"base_seed": 0}, "evaluation": {"config_sha256": "0" * 64},
        },
        "centers": centers,
        "members": members,
        "clean": {"per_center": {center: _metric_body() for center in centers},
                  "center_count": 4, "center_order": centers,
                  "evaluated_center_mean": mean, "four_center_mean": mean},
        "corrupted": {"per_view": [
            {"view_index": index, "per_center": {center: _metric_body() for center in centers},
             "center_count": 4, "center_order": centers,
             "evaluated_center_mean": mean, "four_center_mean": mean}
            for index in range(20)
        ], "aggregates": {
            key: {"view_count": count, "center_count": 4, "center_order": centers,
                  "evaluated_center_mean": mean, "four_center_mean": mean}
            for key, count in (("depth2", 10), ("depth3", 10), ("depth23", 20))
        }},
        "aggregation": {"policy": "equal_views_then_equal_centers",
            "checkpoint_policy": "one_adapted_checkpoint_per_corresponding_center",
            "off_diagonal": "not_evaluated",
        },
    }


def _stub_matrix_recompute(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    expected = json.loads(json.dumps({key: value for key, value in payload.items()
                                     if key not in {"config", "output"}}))
    monkeypatch.setattr("util.evaluation.matrix.aggregate_pn2021_matrix",
                        lambda paths, *, profile_name: expected)


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
        subject = payload.get("subject")
        train_result = subject.get("train_result") if isinstance(subject, dict) else None
        if isinstance(train_result, dict):
            artifact = Path(train_result["path"])
            if not artifact.is_absolute():
                artifact = output_dir / Path(relative_name).parent / artifact
                artifact.parent.mkdir(parents=True, exist_ok=True)
                referenced = _typed_train_result()
                if isinstance(payload.get("checkpoint"), dict):
                    referenced["last_checkpoint"] = dict(payload["checkpoint"])
                    referenced["selection"]["selected_checkpoint"] = dict(payload["checkpoint"])
                referenced["lineage"] = payload["subject"]["lineage"]
                encoded = (json.dumps(referenced, sort_keys=True) + "\n").encode()
                artifact.write_bytes(encoded)
                train_result["sha256"] = hashlib.sha256(encoded).hexdigest()
        lineage = subject.get("lineage") if isinstance(subject, dict) else None
        registry = lineage.get("registry") if isinstance(lineage, dict) else None
        if isinstance(registry, dict):
            artifact = output_dir / Path(relative_name).parent / registry["path"]
            artifact.parent.mkdir(parents=True, exist_ok=True)
            encoded = yaml.safe_dump({"schema_version": 1, "models": {
                payload["model"]["name"]: {
                    "selected_checkpoint_sha256": payload["checkpoint"]["sha256"]
                }
            }}).encode()
            artifact.write_bytes(encoded)
            registry["sha256"] = hashlib.sha256(encoded).hexdigest()
        evaluation_checkpoint = payload.get("checkpoint")
        if isinstance(evaluation_checkpoint, dict) and isinstance(evaluation_checkpoint.get("path"), str):
            artifact = Path(evaluation_checkpoint["path"])
            if not artifact.is_absolute():
                artifact = output_dir / Path(relative_name).parent / artifact
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(FIXTURE_CHECKPOINT)
                evaluation_checkpoint["sha256"] = FIXTURE_CHECKPOINT_SHA256
        if payload.get("artifact_type") == "pn2021_diagonal_four_center_evaluation":
            result_path = output_dir / relative_name
            payload["output"] = {"directory": str(output_dir), "result_file": str(result_path)}
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
        ("../outside.pt", FIXTURE_CHECKPOINT_SHA256, "output directory"),
        ("checkpoints/last.pt", "f" * 64, "SHA256 does not match"),
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


def test_bogus_evaluation_result_is_rejected(tmp_path: Path) -> None:
    run_dir, plan = _action_plan(
        tmp_path, "evaluate_pn2021", delegate_subdir="evaluation"
    )

    payload = _evaluation_result()
    payload["schema_version"] = 1
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1
    assert manifest["delegate_result"] is None
    assert "schema_version=3" in manifest["error"]


def test_evaluation_result_requires_checkpoint_and_metric_bodies(tmp_path: Path) -> None:
    _, valid_plan = _action_plan(
        tmp_path / "valid", "evaluate_pn2021", delegate_subdir="evaluation"
    )
    exit_code, manifest = _record_payload(
        valid_plan, "evaluation_result.json", _evaluation_result()
    )
    assert exit_code == 0 and manifest["status"] == "complete"

    _, legacy_plan = _action_plan(
        tmp_path / "legacy", "evaluate_pn2021", delegate_subdir="evaluation"
    )
    legacy = _evaluation_result()
    legacy["subject"]["mode"] = "legacy_center_adapted"
    legacy["subject"]["lineage"].update(
        adaptation_data={"mapping_version": "v7_super5_sjr_rgq_review_20260528",
                         "mapping_hash": "555ec85d5b51"},
        checkpoint={"sha256": FIXTURE_CHECKPOINT_SHA256},
    )
    exit_code, _ = _record_payload(legacy_plan, "evaluation_result.json", legacy)
    assert exit_code == 0

    _, source_plan = _action_plan(
        tmp_path / "source", "evaluate_pn2021", delegate_subdir="evaluation"
    )
    source = _evaluation_result()
    source["subject"] = {
        "mode": "source_registry",
        "train_result": None,
        "lineage": {
            "model": {"name": EFFICIENTNET1DV2_SPEC.name,
                      "spec": EFFICIENTNET1DV2_SPEC.describe()},
            "source_checkpoint": {"sha256": FIXTURE_CHECKPOINT_SHA256},
            "registry": {"path": "source_registry.yaml", "sha256": "pending"},
            "selection": {"policy": "fold9_best_macro_auprc",
                          "heldout_evaluation_used_for_selection": False},
        },
    }
    source_centers = [
        "ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"
    ]
    source["protocol"]["logical_centers"] = source_centers
    source["clean"]["per_center"] = {center: _metric_body() for center in source_centers}
    for view in source["corrupted"]["per_view"]:
        view["per_center"] = {center: _metric_body() for center in source_centers}
    for aggregate in source["corrupted"]["aggregates"].values():
        aggregate["center_order"] = source_centers
    exit_code, _ = _record_payload(source_plan, "evaluation_result.json", source)
    assert exit_code == 0

    _, invalid_plan = _action_plan(
        tmp_path / "invalid", "evaluate_pn2021", delegate_subdir="evaluation"
    )
    payload = _evaluation_result()
    payload.update(checkpoint=None, clean=None, corrupted="not-a-metric-body")
    exit_code, manifest = _record_payload(
        invalid_plan, "evaluation_result.json", payload
    )
    assert exit_code == 1
    assert manifest["delegate_result"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", {"name": "wrong"}, "model differs"),
        ("centers", ["georgia"], "centers differ"),
        ("subject", {"mode": "unknown", "train_result": None, "lineage": None}, "subject contract"),
    ),
)
def test_evaluation_result_rejects_subject_mismatch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _, plan = _action_plan(
        tmp_path, "evaluate_pn2021", delegate_subdir="evaluation"
    )
    payload = _evaluation_result()
    if field == "centers":
        payload["protocol"]["logical_centers"] = value
    elif field == "subject":
        payload["subject"] = value
    else:
        payload[field] = value
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1
    assert message in manifest["error"]


@pytest.mark.parametrize(
    "attack", ["missing_train_result", "unbound_train_result", "invalid_lineage",
               "wrong_clean_center", "empty_view_bodies", "empty_aggregates", "checkpoint_sha"]
)
def test_evaluation_result_rejects_unbound_or_incomplete_evidence(
    tmp_path: Path, attack: str
) -> None:
    _, plan = _action_plan(tmp_path / attack, "evaluate_pn2021", delegate_subdir="evaluation")
    payload = _evaluation_result()
    if attack in {"missing_train_result", "unbound_train_result"}:
        artifact = tmp_path / f"{attack}.json"
        if attack == "unbound_train_result":
            artifact.write_text("{}\n", encoding="utf-8")
        payload["subject"]["train_result"] = {
            "path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()
            if artifact.exists() else "0" * 64,
        }
    elif attack == "invalid_lineage":
        payload["subject"]["lineage"]["seed"] = None
    elif attack == "wrong_clean_center":
        payload["clean"]["per_center"] = {"not_ningbo": _metric_body()}
    elif attack == "empty_view_bodies":
        payload["corrupted"]["per_view"][0]["per_center"]["ningbo"] = {}
    elif attack == "empty_aggregates":
        payload["corrupted"]["aggregates"] = {}
    else:
        artifact = tmp_path / "wrong.pt"
        artifact.write_bytes(b"wrong checkpoint")
        payload["checkpoint"] = {"path": str(artifact), "sha256": "0" * 64}
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1 and manifest["delegate_result"] is None


@pytest.mark.parametrize("mode", ["legacy_center_adapted", "source_registry"])
def test_evaluation_result_rejects_mode_specific_checkpoint_mismatch(
    tmp_path: Path, mode: str
) -> None:
    _, plan = _action_plan(
        tmp_path / mode, "evaluate_pn2021", delegate_subdir="evaluation"
    )
    payload = _evaluation_result()
    payload["subject"]["mode"] = mode
    if mode == "legacy_center_adapted":
        payload["subject"]["lineage"]["checkpoint"] = {"sha256": "f" * 64}
    else:
        payload["subject"]["train_result"] = None
        payload["subject"]["lineage"] = {
            "model": {"name": EFFICIENTNET1DV2_SPEC.name,
                      "spec": EFFICIENTNET1DV2_SPEC.describe()},
            "checkpoint": {"sha256": FIXTURE_CHECKPOINT_SHA256},
            "source_checkpoint": {"sha256": "f" * 64},
            "registry": {"path": "source_registry.yaml", "sha256": "pending"},
            "selection": {"policy": "fold9_best_macro_auprc",
                          "heldout_evaluation_used_for_selection": False},
        }
        source_centers = [
            "ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"
        ]
        payload["protocol"]["logical_centers"] = source_centers
        payload["clean"]["per_center"] = {center: _metric_body() for center in source_centers}
        for view in payload["corrupted"]["per_view"]:
            view["per_center"] = {center: _metric_body() for center in source_centers}
        for aggregate in payload["corrupted"]["aggregates"].values():
            aggregate["center_order"] = source_centers
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1
    assert "checkpoint differs" in manifest["error"]


def test_matrix_result_is_registered_and_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, plan = _action_plan(tmp_path, "aggregate_pn2021", delegate_subdir="evaluation")
    assert plan.expected_result_relative_path == Path("evaluation/matrix_result.json")
    assert plan.expected_result_type == "pn2021_matrix_result"
    payload = _matrix_result(tmp_path)
    _stub_matrix_recompute(monkeypatch, payload)
    exit_code, manifest = _record_payload(plan, "matrix_result.json", payload)
    assert exit_code == 0
    assert manifest["delegate_result"]["identity"]["centers"] == payload["centers"]


@pytest.mark.parametrize("attack", ["artifact", "order", "model", "cohort", "member_sha", "aggregates"])
def test_matrix_result_rejects_invalid_contract(
    tmp_path: Path, attack: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _, plan = _action_plan(tmp_path / attack, "aggregate_pn2021", delegate_subdir="evaluation")
    payload = _matrix_result(tmp_path / attack)
    _stub_matrix_recompute(monkeypatch, payload)
    if attack == "artifact":
        payload["artifact_type"] = "pn2021_evaluation_result"
    elif attack == "order":
        payload["centers"] = list(reversed(payload["centers"]))
    elif attack == "model":
        payload["model"]["name"] = "unknown"
    elif attack == "cohort":
        payload["cohort"] = {}
    elif attack == "member_sha":
        payload["members"][0]["evaluation_result"]["sha256"] = "f" * 64
    else:
        payload["corrupted"]["aggregates"] = {}
    exit_code, manifest = _record_payload(plan, "matrix_result.json", payload)
    assert exit_code == 1 and manifest["delegate_result"] is None


@pytest.mark.parametrize(
    ("key", "value"),
    (("class_order", ["NORM", "CD", "HYP", "MI", "STTC"]),
     ("heldout_ref_exclusion_required", False)),
)
def test_legacy_evaluation_rejects_unlocked_training_protocol(
    tmp_path: Path, key: str, value: object
) -> None:
    _, plan = _action_plan(tmp_path, "evaluate_pn2021", delegate_subdir="evaluation")
    payload = _evaluation_result()
    payload["subject"]["mode"] = "legacy_center_adapted"
    payload["subject"]["lineage"].update(checkpoint={"sha256": FIXTURE_CHECKPOINT_SHA256})
    referenced = _typed_train_result()
    for field in ("schema_version", "artifact_type", "lineage", "seed"):
        referenced.pop(field)
    referenced["config"]["training"]["resolved"]["protocol"][key] = value
    artifact = tmp_path / "legacy_train_result.json"
    artifact.write_text(json.dumps(referenced), encoding="utf-8")
    payload["subject"]["train_result"] = {
        "path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()
    }
    exit_code, manifest = _record_payload(plan, "evaluation_result.json", payload)
    assert exit_code == 1
    assert "locked K500 protocol" in manifest["error"]


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
