import hashlib
import json
from pathlib import Path

import pytest

from ecg_adv_gen.config import attach_launch_artifacts
from ecg_adv_gen.config.launch import verify_required_inputs
from ecg_adv_gen.evidence import run_record


def _manifest_with_resolved_config(path: Path, *extra_artifacts: dict) -> dict:
    return {
        "artifact_trace": {
            "expected_outputs": {
                "launch_artifacts": [
                    {
                        "path": str(path),
                        "role": "resolved_config",
                        "required": True,
                    },
                    *extra_artifacts,
                ]
            }
        }
    }


def _record_for(index: dict, relative_path: str) -> dict:
    return next(
        record
        for records in index["categories"].values()
        for record in records
        if record["relative_path"] == relative_path
    )


def _record_for_path(index: dict, path: Path) -> dict:
    resolved = str(path.resolve())
    return next(
        record
        for records in index["categories"].values()
        for record in records
        if record["path"] == resolved
    )


def _write_indexed_run(run_dir: Path) -> Path:
    run_dir.mkdir()
    resolved_config = run_dir / "run_config.resolved.yaml"
    resolved_config.write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "env.json").write_text("{}\n", encoding="utf-8")
    manifest = _manifest_with_resolved_config(resolved_config)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    index = run_record._build_file_index(run_dir, manifest)
    (run_dir / "run_file_index.json").write_text(json.dumps(index), encoding="utf-8")
    return resolved_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _required_file(path: Path, role: str) -> dict:
    return {"path": str(path), "role": role, "required": True}


def _ecgfounder_preflight_manifest(ref_meta: Path, init_checkpoint: Path) -> dict:
    return {
        "commands": [
            {
                "argv": [
                    "python",
                    "ecgfounder_fullft.py",
                    "--stage",
                    "k500",
                    "--center",
                    "ningbo",
                    "--ref_meta_json",
                    str(ref_meta),
                    "--init_model_path",
                    str(init_checkpoint),
                ]
            }
        ],
        "artifact_trace": {
            "inputs": {
                "checkpoints": [_required_file(init_checkpoint, "command.init_model_path")],
                "k500_refs": [
                    {
                        "center": "ningbo",
                        "ref_meta_json": _required_file(ref_meta, "kshot_ref_meta"),
                    }
                ],
            }
        },
    }


def test_preflight_rejects_target_adapted_init_with_different_k500_identity(tmp_path: Path):
    ref_meta = tmp_path / "current.ref_meta.json"
    _write_json(
        ref_meta,
        {"center": "ningbo", "K": 2, "selection_seed": 20260531, "ref_record_ids": ["r1", "r2"]},
    )
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    init_checkpoint = init_dir / "last_model.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    _write_json(
        init_dir / "eval_result.json",
        {
            "stage": "k500",
            "center": "ningbo",
            "selected_ref_record_ids": ["r2", "r3"],
            "config": {"seed": 20260601},
        },
    )

    report = verify_required_inputs(_ecgfounder_preflight_manifest(ref_meta, init_checkpoint))

    assert report["passed"] is False
    assert "target-adapted initialization K500 identity mismatch" in report["lineage_errors"][0]["error"]


def test_preflight_allows_explicit_source_only_init(tmp_path: Path):
    ref_meta = tmp_path / "current.ref_meta.json"
    _write_json(
        ref_meta,
        {"center": "ningbo", "K": 1, "selection_seed": 20260531, "ref_record_ids": ["r1"]},
    )
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    init_checkpoint = init_dir / "last_model.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    _write_json(
        init_dir / "eval_result.json",
        {
            "stage": "ptbxl_source",
            "center": None,
            "K": 0,
            "target_train_K": 0,
            "selected_ref_record_ids": [],
            "target_train_record_ids": [],
            "config": {"stage": "ptbxl_source"},
        },
    )

    assert verify_required_inputs(_ecgfounder_preflight_manifest(ref_meta, init_checkpoint))["passed"] is True


def _make_registerable_run(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    output_root = tmp_path / "output"
    run_dir = output_root / "integrity" / "run1"
    run_dir.mkdir(parents=True)
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    paths = {
        "env": run_dir / "env.json",
        "checkpoint": inputs_dir / "init.pt",
        "ref_meta": inputs_dir / "refs.json",
        "signals": inputs_dir / "signals.npz",
        "latents": inputs_dir / "latents.npz",
    }
    payloads = {
        "env": b'{"python":"3.11"}\n',
        "checkpoint": b"checkpoint-v1",
        "ref_meta": b'{"ids":["r1"]}\n',
        "signals": b"signals-v1",
        "latents": b"latents-v1",
    }
    for name, path in paths.items():
        path.write_bytes(payloads[name])

    (run_dir / "run_config.resolved.yaml").write_text("seed: 42\n", encoding="utf-8")
    _write_json(run_dir / "run_config.resolved.json", {"seed": 42})
    (run_dir / "command.sh").write_text("python train.py\n", encoding="utf-8")
    _write_json(run_dir / "data_manifest.json", {"dataset": "fixture"})
    policy = "k500_internal_source_floor_v1"
    _write_json(
        run_dir / "selection.json",
        {
            "selection_policy": {"policy": policy},
            "selection_safety": {
                "heldout_target_labels_used_for_selection": False,
                "full_target_distribution_used_for_tuning": False,
                "forbidden_reference_found": False,
            },
            "paper_protocol": {"mapping_version": "v7", "mapping_hash": "555ec85d5b51"},
        },
    )
    _write_json(
        run_dir / "k500_ref_ids.json",
        {
            "paper_protocol": {
                "mapping_version": "v7",
                "mapping_hash": "555ec85d5b51",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "centers": {
                "ningbo": {
                    "k": 1,
                    "selection_seed": 42,
                    "ref_record_ids_ordered": ["r1"],
                    "ref_record_ids_sorted": ["r1"],
                    "ref_record_ids_sha256": hashlib.sha256(b"r1\n").hexdigest(),
                    "source_ref_meta_path": str(paths["ref_meta"]),
                    "source_ref_meta_sha256": hashlib.sha256(payloads["ref_meta"]).hexdigest(),
                }
            },
        },
    )
    eval_result = run_dir / "eval" / "eval_result.json"
    metrics_source = run_dir / "eval" / "metrics_source.json"
    _write_json(eval_result, {"status": "ok"})
    _write_json(metrics_source, {"metrics": "fixture"})
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "manifest_schema_version": 2,
        "run_id": "run1",
        "status": "succeeded",
        "git": {"commit": "a" * 40},
        "config_hash_sha256": "b" * 64,
        "commands": [{"argv": ["python", "train.py"], "cwd": str(run_dir)}],
        "experiment": {"name": "integrity", "purpose": "Exercise registration integrity."},
        "paper_protocol": {
            "mapping_version": "v7",
            "mapping_hash": "555ec85d5b51",
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "target_centers": ["ningbo"],
            "kshot": {"k": 1, "seed": 42},
            "selection": {"policy": policy},
        },
        "artifact_verification": {"passed": True},
        "artifact_trace": {
            "schema_version": 1,
            "inputs": {
                "checkpoints": [_required_file(paths["checkpoint"], "model.checkpoint")],
                "k500_refs": [
                    {
                        "ref_meta_json": _required_file(paths["ref_meta"], "kshot_ref_meta"),
                        "signals_npz": _required_file(paths["signals"], "kshot_signals"),
                        "latent_npz": _required_file(paths["latents"], "kshot_latents"),
                    }
                ],
                "data_caches": [],
            },
            "selection_policy": {"policy": policy},
            "expected_outputs": {
                "child_runs": [
                    {
                        "center": "ningbo",
                        "expected_artifacts": [
                            _required_file(eval_result, "eval_result"),
                            _required_file(metrics_source, "metrics_source"),
                        ],
                    }
                ]
            },
        },
        "run_record": {
            "purpose": "Exercise registration integrity.",
            "result_summary": "The fixture completed with synthetic evidence.",
            "outcome": "provisional",
        },
    }
    manifest = attach_launch_artifacts(manifest, run_dir=run_dir)
    _write_json(manifest_path, manifest)
    run_record.finalize_run_record(run_dir)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("{}\n", encoding="utf-8")
    local_config_path = tmp_path / "local.yaml"
    local_config_path.write_text(f"paths:\n  output_root: {output_root}\n", encoding="utf-8")
    paths["run_dir"] = run_dir
    return {
        "registry_path": registry_path,
        "local_config_path": local_config_path,
        "run_dir": run_dir,
    }, paths


def test_file_index_hashes_only_declared_reproduction_critical_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved_config = run_dir / "run_config.resolved.yaml"
    resolved_config.write_text("seed: 42\n", encoding="utf-8")
    selected_checkpoint = run_dir / "checkpoints" / "best.pt"
    selected_checkpoint.parent.mkdir()
    selected_checkpoint.write_bytes(b"selected-checkpoint")
    undeclared_checkpoint = run_dir / "checkpoints" / "last.pt"
    undeclared_checkpoint.write_bytes(b"undeclared-checkpoint")

    index = run_record._build_file_index(
        run_dir,
        _manifest_with_resolved_config(
            resolved_config,
            {
                "path": str(selected_checkpoint),
                "role": "selected_checkpoint",
                "required": True,
            },
        ),
    )

    config_record = _record_for(index, "run_config.resolved.yaml")
    assert config_record["sha256"] == hashlib.sha256(b"seed: 42\n").hexdigest()
    assert _record_for(index, "checkpoints/best.pt")["sha256"] == hashlib.sha256(
        b"selected-checkpoint"
    ).hexdigest()
    assert "sha256" not in _record_for(index, "checkpoints/last.pt")


def test_file_index_hashes_declared_checkpoint_and_result_chain_roles(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved_config = run_dir / "run_config.resolved.yaml"
    resolved_config.write_text("seed: 42\n", encoding="utf-8")
    role_paths = {
        "best_model": run_dir / "runs/model/best_model.pt",
        "last_model": run_dir / "runs/model/last_model.pt",
        "selected_checkpoint": run_dir / "runs/model/selected.pt",
        "eval_result": run_dir / "runs/model/eval_result.json",
        "training_log": run_dir / "runs/model/training_log.json",
        "artifact_manifest": run_dir / "runs/model/artifact_manifest.json",
        "paper_table": run_dir / "reports/paper_table.csv",
        "paper_table_manifest": run_dir / "reports/paper_table_manifest.json",
    }
    for role, path in role_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("utf-8"))

    index = run_record._build_file_index(
        run_dir,
        _manifest_with_resolved_config(
            resolved_config,
            *(
                {"path": str(path), "role": role, "required": True}
                for role, path in role_paths.items()
            ),
        ),
    )

    for role, path in role_paths.items():
        record = _record_for(index, path.relative_to(run_dir).as_posix())
        assert record["sha256"] == hashlib.sha256(role.encode("utf-8")).hexdigest()


def test_file_index_hashes_every_required_expected_output_role(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "env.json").write_bytes(b"env")
    resolved_config = run_dir / "run_config.resolved.yaml"
    resolved_config.write_bytes(b"config")
    launch_config = run_dir / "launch_config.json"
    launch_config.write_bytes(b"launch-v1")
    manifest = _manifest_with_resolved_config(
        resolved_config,
        _required_file(launch_config, "launch_config"),
    )
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "run_file_index.json", run_record._build_file_index(run_dir, manifest))

    assert _record_for_path(
        json.loads((run_dir / "run_file_index.json").read_text(encoding="utf-8")),
        launch_config,
    )["sha256"] == hashlib.sha256(b"launch-v1").hexdigest()
    launch_config.write_bytes(b"launch-v2")
    assert run_record.verify_run_file_index(run_dir)["passed"] is False


def test_verify_file_index_detects_reproduction_critical_sha256_mismatch(tmp_path: Path):
    run_dir = tmp_path / "run"
    resolved_config = _write_indexed_run(run_dir)

    verifier = getattr(run_record, "verify_run_file_index", None)
    assert verifier is not None, "run file index has no integrity verifier"
    assert verifier(run_dir)["passed"] is True

    resolved_config.write_text("seed: 43\n", encoding="utf-8")
    report = verifier(run_dir)

    assert report["passed"] is False
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_verify_file_index_detects_size_mismatch(tmp_path: Path):
    run_dir = tmp_path / "run"
    resolved_config = _write_indexed_run(run_dir)
    resolved_config.write_text("seed: 4200\n", encoding="utf-8")

    verifier = getattr(run_record, "verify_run_file_index", None)
    assert verifier is not None, "run file index has no integrity verifier"
    report = verifier(run_dir)

    assert report["passed"] is False
    assert any("size_bytes mismatch" in error for error in report["errors"])


def test_verify_file_index_detects_missing_file(tmp_path: Path):
    run_dir = tmp_path / "run"
    resolved_config = _write_indexed_run(run_dir)
    resolved_config.unlink()

    verifier = getattr(run_record, "verify_run_file_index", None)
    assert verifier is not None, "run file index has no integrity verifier"
    report = verifier(run_dir)

    assert report["passed"] is False
    assert any("missing" in error for error in report["errors"])


def test_verify_file_index_detects_same_size_change_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    resolved_config = _write_indexed_run(run_dir)
    real_sha256 = run_record._sha256_file

    def mutate_after_read(path: Path) -> str:
        digest = real_sha256(path)
        if path.resolve() == resolved_config.resolve():
            resolved_config.write_text("seed: 43\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(run_record, "_sha256_file", mutate_after_read)
    report = run_record.verify_run_file_index(run_dir)

    assert report["passed"] is False
    assert any("changed during verification" in error for error in report["errors"])


def test_verify_file_index_marks_schema_v1_unchecked(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_indexed_run(run_dir)
    index_path = run_dir / "run_file_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    verifier = getattr(run_record, "verify_run_file_index", None)
    assert verifier is not None, "run file index has no integrity verifier"
    report = verifier(run_dir)

    assert report["passed"] is False
    assert report["status"] == "unchecked"
    assert report["errors"] == []
    assert any("schema_version=1" in warning for warning in report["warnings"])


def test_finalize_run_record_writes_immediately_verifiable_v2_index(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved_config = run_dir / "run_config.resolved.yaml"
    resolved_config.write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "command.sh").write_text("python train.py\n", encoding="utf-8")
    (run_dir / "env.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "selection.json").write_text("{}\n", encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    selected_checkpoint = run_dir / "selected.pt"
    selected_checkpoint.write_bytes(b"selected-checkpoint")
    manifest = {
        "manifest_schema_version": 2,
        "run_id": "integrity-fixture",
        "status": "dry_run",
        "git": {"commit": "a" * 40},
        "config_hash_sha256": "b" * 64,
        "commands": [{"argv": ["python", "train.py"], "cwd": str(run_dir)}],
        "experiment": {"name": "integrity-fixture"},
        "paper_protocol": {
            "mapping_version": "v7",
            "mapping_hash": "555ec85d5b51",
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "target_centers": ["ningbo"],
            "kshot": {"k": 500, "seed": 42},
            "selection": {"policy": "k500_internal"},
        },
        "artifact_trace": {
            "schema_version": 1,
            "inputs": {},
            "expected_outputs": {
                "launch_artifacts": [
                    {"path": str(manifest_path), "role": "run_manifest", "required": True},
                    {"path": str(resolved_config), "role": "resolved_config", "required": True},
                    {
                        "path": str(selected_checkpoint),
                        "role": "selected_checkpoint",
                        "required": True,
                    },
                ]
            },
        },
        "run_record": {
            "purpose": "Exercise the run file integrity contract.",
            "result_summary": "The synthetic dry-run fixture was finalized.",
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hashed_paths = []
    original_sha256_file = run_record._sha256_file

    def record_sha256_call(path: Path) -> str:
        hashed_paths.append(path.resolve())
        return original_sha256_file(path)

    monkeypatch.setattr(run_record, "_sha256_file", record_sha256_call)

    run_record.finalize_run_record(run_dir)
    report = run_record.verify_run_file_index(run_dir)

    assert report["passed"] is True
    assert report["status"] == "verified"
    assert report["checked_count"] == 4
    assert hashed_paths.count(selected_checkpoint.resolve()) == 2  # index build + verification


def test_file_index_hashes_required_nested_inputs_and_mandatory_env(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env_path = run_dir / "env.json"
    env_path.write_bytes(b"environment")
    config_path = run_dir / "run_config.resolved.yaml"
    config_path.write_bytes(b"config")
    checkpoint = tmp_path / "checkpoint.pt"
    ref_meta = tmp_path / "refs.json"
    signals = tmp_path / "signals.npz"
    latents = tmp_path / "latents.npz"
    for path in (checkpoint, ref_meta, signals, latents):
        path.write_bytes(path.name.encode("utf-8"))
    optional_cache = tmp_path / "cache"
    optional_cache.mkdir()
    (optional_cache / "do-not-hash.bin").write_bytes(b"cache")
    manifest = _manifest_with_resolved_config(config_path)
    manifest["artifact_trace"]["inputs"] = {
        "checkpoints": [_required_file(checkpoint, "model.checkpoint")],
        "k500_refs": [
            {
                "ref_meta_json": _required_file(ref_meta, "kshot_ref_meta"),
                "signals_npz": _required_file(signals, "kshot_signals"),
                "latent_npz": _required_file(latents, "kshot_latents"),
            }
        ],
        "data_caches": [
            {"path": str(optional_cache), "role": "cache_dir", "required": False}
        ],
    }

    index = run_record._build_file_index(run_dir, manifest)

    for path in (env_path, config_path, checkpoint, ref_meta, signals, latents):
        assert _record_for_path(index, path)["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(
        record["path"] != str(optional_cache.resolve())
        for records in index["categories"].values()
        for record in records
    )


def test_verifier_ignores_stale_hash_outside_current_manifest_scope(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "env.json").write_bytes(b"env")
    current_file = run_dir / "current.json"
    stale_file = run_dir / "stale.json"
    current_file.write_bytes(b"current-v1")
    stale_file.write_bytes(b"stale-v1")
    original_manifest = _manifest_with_resolved_config(
        current_file,
        _required_file(stale_file, "data_manifest"),
    )
    _write_json(run_dir / "run_manifest.json", original_manifest)
    _write_json(run_dir / "run_file_index.json", run_record._build_file_index(run_dir, original_manifest))
    current_manifest = _manifest_with_resolved_config(current_file)
    _write_json(run_dir / "run_manifest.json", current_manifest)
    stale_file.write_bytes(b"stale-v2")

    report = run_record.verify_run_file_index(run_dir)

    assert report["passed"] is True
    assert report["checked_count"] == 2  # current file + mandatory env.json


def test_file_index_omits_root_and_mirrored_self_index(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "env.json").write_bytes(b"env")
    config_path = run_dir / "run_config.resolved.yaml"
    config_path.write_bytes(b"config")
    (run_dir / "run_file_index.json").write_bytes(b"old-root-index")
    mirror = run_dir / "manifests" / "run_file_index.json"
    mirror.parent.mkdir()
    mirror.write_bytes(b"old-mirror-index")

    index = run_record._build_file_index(run_dir, _manifest_with_resolved_config(config_path))
    indexed_paths = {
        record["path"]
        for records in index["categories"].values()
        for record in records
    }

    assert str((run_dir / "run_file_index.json").resolve()) not in indexed_paths
    assert str(mirror.resolve()) not in indexed_paths


def test_mirror_refuses_symlink_destination_without_touching_target(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True)
    (run_dir / "run_card.json").write_text("source\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    (manifests / "run_card.json").symlink_to(outside)

    with pytest.raises(run_record.RunRecordError, match="symbolic link"):
        run_record._mirror_known_small_files(run_dir)

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_atomic_write_rechecks_parent_after_containment_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    parent = run_dir / "manifests"
    parent.mkdir(parents=True)
    moved_parent = run_dir / "original-manifests"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_check = run_record._assert_safe_write_path

    def swap_parent(path: Path, *, root: Path | None = None) -> None:
        real_check(path, root=root)
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(run_record, "_assert_safe_write_path", swap_parent)
    with pytest.raises((run_record.RunRecordError, OSError)):
        run_record._atomic_write_text(parent / "target.json", "payload\n", root=run_dir)

    assert not (outside / "target.json").exists()


def test_small_copy_temp_creation_is_exclusive_and_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True)
    src = run_dir / "run_card.json"
    src.write_text("source\n", encoding="utf-8")
    victim = tmp_path / "victim.json"
    victim.write_text("victim\n", encoding="utf-8")
    fixed = type("FixedUUID", (), {"hex": "fixed"})()
    monkeypatch.setattr(run_record.uuid, "uuid4", lambda: fixed)
    (manifests / ".run_card.json.fixed.tmp").symlink_to(victim)

    with pytest.raises((FileExistsError, run_record.RunRecordError)):
        run_record._copy_small_file(src, manifests / "run_card.json", root=run_dir)

    assert victim.read_text(encoding="utf-8") == "victim\n"


@pytest.mark.parametrize("mutated_name", ["env", "checkpoint", "signals", "latents"])
def test_registration_rejects_same_size_critical_input_mutation(tmp_path: Path, mutated_name: str):
    registration, paths = _make_registerable_run(tmp_path)
    path = paths[mutated_name]
    payload = path.read_bytes()
    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(run_record.RunRecordError, match="sha256 mismatch"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_rejects_schema_v1_as_unchecked(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    index_path = paths["run_dir"] / "run_file_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = 1
    _write_json(index_path, index)

    with pytest.raises(run_record.RunRecordError, match="schema_version=1"):
        run_record.register_run_in_registry(**registration, status="trusted")


def test_registration_allows_schema_v1_for_exploratory_compatibility(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    index_path = paths["run_dir"] / "run_file_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = 1
    _write_json(index_path, index)

    registry = run_record.register_run_in_registry(**registration, status="exploratory")

    assert registry["managed_runs"][0]["status"] == "exploratory"


def test_registration_rejects_unknown_status(tmp_path: Path):
    registration, _ = _make_registerable_run(tmp_path)

    with pytest.raises(run_record.RunRecordError, match="Unsupported registration status"):
        run_record.register_run_in_registry(**registration, status="unverified")


def test_registration_requires_run_card_and_summary_roles(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    manifest_path = paths["run_dir"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    launch = manifest["artifact_trace"]["expected_outputs"]["launch_artifacts"]
    manifest["artifact_trace"]["expected_outputs"]["launch_artifacts"] = [
        item for item in launch if item.get("role") not in {"run_card", "run_summary"}
    ]
    _write_json(manifest_path, manifest)

    with pytest.raises(run_record.RunRecordError, match="run_card"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_rejects_card_manifest_mismatch_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration, paths = _make_registerable_run(tmp_path)
    card_path = paths["run_dir"] / "run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["run_id"] = "forged-run"
    _write_json(card_path, card)

    def reject_hash_call(path: Path) -> str:
        raise AssertionError(f"hash called before card/manifest contract for {path}")

    monkeypatch.setattr(run_record, "_sha256_file", reject_hash_call)
    with pytest.raises(run_record.RunRecordError, match="run_card.json disagrees.*run_id"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_rejects_reindexed_mutation_against_existing_anchor(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    first = run_record.register_run_in_registry(**registration, status="provisional")
    assert first["managed_runs"][0]["run_file_index_sha256"]

    checkpoint = paths["checkpoint"]
    checkpoint.write_bytes(b"checkpoint-v2")
    run_dir = paths["run_dir"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    _write_json(run_dir / "run_file_index.json", run_record._build_file_index(run_dir, manifest))

    with pytest.raises(run_record.RunRecordError, match="anchored run_file_index SHA-256"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_rejects_identity_rename_for_anchored_run_dir(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    run_record.register_run_in_registry(**registration, status="provisional")
    run_dir = paths["run_dir"]
    manifest_path = run_dir / "run_manifest.json"
    card_path = run_dir / "run_card.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    manifest["run_id"] = card["run_id"] = "renamed-run"
    _write_json(manifest_path, manifest)
    _write_json(card_path, card)
    _write_json(run_dir / "run_file_index.json", run_record._build_file_index(run_dir, manifest))

    with pytest.raises(run_record.RunRecordError, match="run_dir is already anchored"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_anchors_the_exact_index_verified_before_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration, paths = _make_registerable_run(tmp_path)
    index_path = paths["run_dir"] / "run_file_index.json"
    original_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    real_verify = run_record.verify_run_file_index

    def verify_then_swap(run_dir: Path) -> dict:
        report = real_verify(run_dir)
        _write_json(index_path, {"forged": True})
        return report

    monkeypatch.setattr(run_record, "verify_run_file_index", verify_then_swap)
    registry = run_record.register_run_in_registry(**registration, status="provisional")

    anchored = registry["managed_runs"][0]["run_file_index_sha256"]
    assert anchored == original_sha
    assert anchored != hashlib.sha256(index_path.read_bytes()).hexdigest()


def test_verifier_and_registration_reject_required_input_symlink(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    checkpoint = paths["checkpoint"]
    payload = checkpoint.read_bytes()
    target = checkpoint.with_name("target.pt")
    target.write_bytes(payload)
    checkpoint.unlink()
    checkpoint.symlink_to(target)

    report = run_record.verify_run_file_index(paths["run_dir"])
    assert report["passed"] is False
    assert any("symbolic link" in error for error in report["errors"])
    with pytest.raises(run_record.RunRecordError, match="symbolic link"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_preflights_output_root_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration, _ = _make_registerable_run(tmp_path)
    registration["local_config_path"].write_text("paths: {}\n", encoding="utf-8")

    def reject_hash_call(path: Path) -> str:
        raise AssertionError(f"hash called before output-root preflight for {path}")

    monkeypatch.setattr(run_record, "_sha256_file", reject_hash_call)
    with pytest.raises(run_record.RunRecordError, match="no paths.output_root"):
        run_record.register_run_in_registry(**registration, status="provisional")


def test_registration_resolves_local_output_root_interpolation(tmp_path: Path):
    registration, paths = _make_registerable_run(tmp_path)
    registration["local_config_path"].write_text(
        f"paths:\n  data_root: {tmp_path / 'output'}\n  output_root: ${{paths.data_root}}\n",
        encoding="utf-8",
    )

    registry = run_record.register_run_in_registry(**registration, status="provisional")

    assert registry["managed_runs"][0]["run_dir"].startswith("${paths.output_root}/")
    assert paths["run_dir"].is_dir()


def test_registration_rejects_cheap_contract_before_hashing(tmp_path: Path, monkeypatch):
    registration, paths = _make_registerable_run(tmp_path)
    manifest_path = paths["run_dir"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "dry_run"
    _write_json(manifest_path, manifest)

    def reject_hash_call(path: Path) -> str:
        raise AssertionError(f"hash called before cheap contract for {path}")

    monkeypatch.setattr(run_record, "_sha256_file", reject_hash_call)
    with pytest.raises(run_record.RunRecordError, match="run_card.json disagrees|trusted/provisional registration"):
        run_record.register_run_in_registry(**registration, status="provisional")
