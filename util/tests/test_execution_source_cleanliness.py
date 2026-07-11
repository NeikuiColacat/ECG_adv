from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    repo = tmp_path / "repo"
    paths = {
        "index": repo / "configs/active_scripts.yaml",
        "base": repo / "configs/defaults/base.yaml",
        "entry": repo / "configs/experiments/study.yaml",
        "local": repo / "configs/local/server.yaml",
        "model": repo / "model/external/status.txt",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"name: {path.stem}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "pytest")
    _git(repo, "config", "user.email", "pytest@example.invalid")
    _git(repo, "add", "configs/active_scripts.yaml", "configs/defaults/base.yaml", "configs/experiments/study.yaml")
    _git(repo, "commit", "-qm", "fixture")
    config = {
        "_entry_config": str(paths["entry"]),
        "_config_sources": [str(paths["base"]), str(paths["entry"])],
        "_local_config_sources": [str(paths["local"])],
    }
    return repo, config, paths


def test_execution_source_helper_exists():
    from ecg_adv_gen.config.source_clean import inspect_execution_sources

    assert callable(inspect_execution_sources)


def test_clean_report_has_only_managed_experiment_sources(source_repo):
    from ecg_adv_gen.config.source_clean import inspect_execution_sources

    repo, config, paths = source_repo
    report = inspect_execution_sources(config, repo_root=repo)

    assert report["passed"] is True
    assert {row["path"] for row in report["sources"]} == {
        str(paths["index"].resolve()),
        str(paths["base"].resolve()),
        str(paths["entry"].resolve()),
    }
    assert report["excluded_local_config_sources"] == [str(paths["local"].resolve())]
    assert all(row["state"] == "clean" for row in report["sources"])


@pytest.mark.parametrize("role", ["index", "base", "entry"])
@pytest.mark.parametrize("state", ["staged", "unstaged", "untracked", "missing"])
def test_every_nonclean_source_state_fails_closed(source_repo, role: str, state: str):
    from ecg_adv_gen.config.source_clean import inspect_execution_sources

    repo, config, paths = source_repo
    target = paths[role]
    repo_path = target.relative_to(repo).as_posix()
    if state == "staged":
        target.write_text("name: staged\n", encoding="utf-8")
        _git(repo, "add", repo_path)
    elif state == "unstaged":
        target.write_text("name: unstaged\n", encoding="utf-8")
    elif state == "untracked":
        _git(repo, "rm", "--cached", "-q", repo_path)
    else:
        target.unlink()

    report = inspect_execution_sources(config, repo_root=repo)
    row = next(item for item in report["sources"] if item["path"] == str(target.resolve()))
    assert report["passed"] is False
    assert row[state] is True
    assert state in row["state"]


def test_dirty_local_overlay_and_unrelated_model_path_do_not_block(source_repo):
    from ecg_adv_gen.config.source_clean import inspect_execution_sources

    repo, config, paths = source_repo
    config["_config_sources"].append(str(paths["local"]))
    paths["local"].write_text("name: dirty-local\n", encoding="utf-8")
    paths["model"].write_text("dirty external model status\n", encoding="utf-8")

    report = inspect_execution_sources(config, repo_root=repo)

    assert report["passed"] is True
    assert all("model/" not in row["repo_path"] for row in report["sources"])


@pytest.mark.parametrize(
    ("layer_name", "reserved_key"),
    [
        ("effnet_f004_rho_sweep_onecenter_smoke.yaml", "_config_sources"),
        ("effnet_f004_rho_sweep_locked.yaml", "_local_config_sources"),
        ("effnet_matched_f005_locked.yaml", "_entry_config"),
        ("vae_lhat_defaults.yaml", "_project_root"),
        ("effnet_f004_rho_sweep_locked.yaml", "_cli_overrides"),
        ("effnet_f004_rho_sweep_onecenter_smoke.yaml", "_local_config"),
    ],
)
def test_loader_rejects_reserved_internal_key_in_entry_or_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    layer_name: str,
    reserved_key: str,
) -> None:
    from ecg_adv_gen.config import ConfigError, load_experiment_config
    from ecg_adv_gen.config import loader

    repo = Path(__file__).resolve().parents[2]
    entry = repo / "configs/studies/effnet_f004_rho_sweep_onecenter_smoke.yaml"
    local = repo / "configs/local/linbinhao_server.example.yaml"
    real_read_yaml = loader._read_yaml

    def injecting_read_yaml(path: Path) -> dict:
        data = real_read_yaml(path)
        if path.name == layer_name:
            data[reserved_key] = ["forged.yaml"]
        return data

    monkeypatch.setattr(loader, "_read_yaml", injecting_read_yaml)
    with pytest.raises(ConfigError, match=f"reserved.*{reserved_key}"):
        load_experiment_config(
            entry,
            local,
            runtime_context={"run_id": "pytest-reserved-provenance"},
        )


def test_loader_accumulates_ancestor_provenance_outside_merged_yaml_and_blocks_dirty_parent(
    source_repo,
) -> None:
    from ecg_adv_gen.config.loader import _load_with_extends
    from ecg_adv_gen.config.source_clean import inspect_execution_sources

    repo, _config, paths = source_repo
    paths["base"].write_text("base_value: 1\n", encoding="utf-8")
    paths["entry"].write_text(
        "extends: ../defaults/base.yaml\nentry_value: 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", "configs/defaults/base.yaml", "configs/experiments/study.yaml")
    _git(repo, "commit", "-qm", "add extends chain")

    sources: list[str] = []
    merged = _load_with_extends(paths["entry"], sources=sources)
    assert merged == {"base_value": 1, "entry_value": 2}
    assert not any(str(key).startswith("_") for key in merged)
    assert sources == [str(paths["base"].resolve()), str(paths["entry"].resolve())]

    paths["base"].write_text("base_value: dirty\n", encoding="utf-8")
    report = inspect_execution_sources(
        {
            "_entry_config": str(paths["entry"]),
            "_config_sources": sources,
            "_local_config_sources": [str(paths["local"])],
        },
        repo_root=repo,
    )
    assert report["passed"] is False
    parent = next(row for row in report["sources"] if row["path"] == str(paths["base"].resolve()))
    assert parent["unstaged"] is True


def test_require_clean_execution_sources_reports_phase(source_repo):
    from ecg_adv_gen.config.source_clean import (
        ExecutionSourceError,
        inspect_execution_sources,
        require_clean_execution_sources,
    )

    repo, config, paths = source_repo
    paths["index"].write_text("dirty: true\n", encoding="utf-8")
    report = inspect_execution_sources(config, repo_root=repo)
    with pytest.raises(ExecutionSourceError, match="pre_execute.*active_scripts.yaml"):
        require_clean_execution_sources(report, phase="pre_execute")


def test_launcher_dry_run_records_dirty_source_diagnostics(monkeypatch, capsys, source_repo):
    import scripts.run_experiment as launcher

    repo, config, paths = source_repo
    paths["entry"].write_text("dirty: true\n", encoding="utf-8")
    args = Namespace(
        config=str(paths["entry"]), local_config=str(paths["local"]), run_id="dry",
        output_dir="", dry_run=True, execute=False, write_plan=False,
        resume=False, force=False, set_overrides=[],
    )
    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(launcher, "parse_args", lambda: args)
    monkeypatch.setattr(launcher, "load_experiment_config", lambda *a, **k: config)
    monkeypatch.setattr(launcher, "validate_experiment_config", lambda *a, **k: {})
    monkeypatch.setattr(launcher, "build_runner_commands", lambda *a, **k: [])
    monkeypatch.setattr(launcher, "build_postprocess_commands", lambda *a, **k: [])
    monkeypatch.setattr(launcher, "make_dry_run_manifest", lambda *a, **k: {"status": "dry_run"})
    monkeypatch.setattr(launcher, "attach_replication_preflight", lambda manifest, *a, **k: manifest)

    assert launcher.main() == 0
    rendered = capsys.readouterr().out.split("\n\n# Managed commands", 1)[0]
    manifest = json.loads(rendered)
    check = manifest["execution_source_cleanliness"]["checks"][0]
    assert check["phase"] == "diagnostic"
    assert check["passed"] is False
    assert any(row["unstaged"] for row in check["sources"])


def test_launcher_rechecks_sources_immediately_before_child_invocation(
    monkeypatch, tmp_path: Path, source_repo
):
    import scripts.run_experiment as launcher

    repo, config, paths = source_repo
    out_dir = tmp_path / "run"
    args = Namespace(
        config=str(paths["entry"]), local_config=str(paths["local"]), run_id="execute",
        output_dir=str(out_dir), dry_run=False, execute=True, write_plan=False,
        resume=False, force=False, set_overrides=[],
    )
    manifest = {
        "status": "dry_run",
        "launcher": {},
        "safety": {},
        "config_hash_sha256": "fixture",
    }
    invoked: list[bool] = []
    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(launcher, "parse_args", lambda: args)
    monkeypatch.setattr(launcher, "load_experiment_config", lambda *a, **k: config)
    monkeypatch.setattr(launcher, "validate_experiment_config", lambda *a, **k: {})
    monkeypatch.setattr(launcher, "build_runner_commands", lambda *a, **k: [])
    monkeypatch.setattr(launcher, "build_postprocess_commands", lambda *a, **k: [])
    monkeypatch.setattr(launcher, "make_dry_run_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(launcher, "attach_replication_preflight", lambda value, *a, **k: value)
    monkeypatch.setattr(launcher, "require_cuda_visible_devices", lambda: "0")
    monkeypatch.setattr(launcher, "check_nvidia_smi", lambda: "fixture")
    monkeypatch.setattr(launcher, "prepare_output_dir", lambda *a, **k: out_dir)

    def write_plan(*_args, **_kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        return manifest

    def verify_then_dirty(_manifest):
        paths["base"].write_text("dirty: after-initial-check\n", encoding="utf-8")
        return {"passed": True}

    monkeypatch.setattr(launcher, "write_launch_plan_files", write_plan)
    monkeypatch.setattr(launcher, "verify_required_inputs", verify_then_dirty)
    monkeypatch.setattr(launcher, "run_managed_commands", lambda *a, **k: invoked.append(True))

    assert launcher.main() == 3
    assert invoked == []
    persisted = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checks = persisted["execution_source_cleanliness"]["checks"]
    assert [row["phase"] for row in checks] == ["pre_execute", "pre_child_invocation"]
    assert checks[0]["passed"] is True
    assert checks[1]["passed"] is False
    assert persisted["safety"]["managed_child_commands_invoked"] is False


def _matrix_args(out_dir: Path, *, execute: bool, resume: bool = False) -> Namespace:
    return Namespace(
        config="experiment.yaml",
        local_config="local.yaml",
        run_id="matrix-source-gate",
        gpus="0",
        max_parallel=1,
        matrix_key="center",
        output_dir=str(out_dir),
        min_free_memory_mb=0,
        max_utilization_pct=100,
        dry_run=not execute,
        execute=execute,
        resume=resume,
        force=False,
        write_plan=True,
    )


def _matrix_command(script: Path, output: Path, index: int, *extra: str) -> dict:
    return {
        "name": f"child_{index}",
        "argv": [sys.executable, str(script), str(output), *extra],
        "cwd": str(script.parent),
        "env": {},
        "matrix": {"center": f"center_{index}"},
    }


def _matrix_manifest(commands: list[dict], outputs: list[Path]) -> dict:
    return {
        "manifest_schema_version": 2,
        "run_id": "matrix-source-gate",
        "status": "launch_prepared",
        "config_hash_sha256": "a" * 64,
        "commands": commands,
        "postprocess_commands": [],
        "launcher": {},
        "safety": {"managed_child_commands_invoked": False},
        "artifact_trace": {
            "metrics": {},
            "expected_outputs": {
                "launch_artifacts": [],
                "child_runs": [
                    {
                        "command_index": index,
                        "name": command["name"],
                        "expected_artifacts": [
                            {"role": "result", "path": str(output), "required": True}
                        ],
                    }
                    for index, (command, output) in enumerate(
                        zip(commands, outputs, strict=True)
                    )
                ],
                "postprocess_runs": [],
            },
        },
    }


def _patch_matrix_config(
    monkeypatch,
    launcher,
    *,
    repo: Path,
    config: dict,
    args: Namespace,
    manifest: dict,
    commands: list[dict],
) -> None:
    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(launcher, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(launcher, "load_experiment_config", lambda *a, **k: config)
    monkeypatch.setattr(
        launcher,
        "validate_experiment_config",
        lambda *a, **k: {
            "output_root": str(Path(args.output_dir).parent),
            "write_boundary": str(Path(args.output_dir).parent),
        },
    )
    monkeypatch.setattr(launcher, "build_runner_commands", lambda *_a, **_k: commands)
    monkeypatch.setattr(launcher, "build_postprocess_commands", lambda *_a, **_k: [])
    monkeypatch.setattr(
        launcher, "make_dry_run_manifest", lambda *_a, **_k: json.loads(json.dumps(manifest))
    )
    monkeypatch.setattr(
        launcher, "attach_replication_preflight", lambda value, *_a, **_k: value
    )


def test_matrix_dry_run_records_dirty_source_diagnostic_without_blocking(
    monkeypatch, tmp_path: Path, source_repo
) -> None:
    from scripts.agent import run_matrix_parallel as launcher

    repo, config, paths = source_repo
    paths["entry"].write_text("dirty: diagnostic\n", encoding="utf-8")
    out_dir = tmp_path / "dry-run"
    args = _matrix_args(out_dir, execute=False)
    commands = [_matrix_command(paths["entry"], tmp_path / "unused", 0)]
    manifest = _matrix_manifest(commands, [tmp_path / "unused"])
    captured: dict = {}
    _patch_matrix_config(
        monkeypatch,
        launcher,
        repo=repo,
        config=config,
        args=args,
        manifest=manifest,
        commands=commands,
    )

    def capture_plan(**kwargs):
        captured.update(kwargs["manifest"])
        return kwargs["out_dir"], kwargs["manifest"]

    monkeypatch.setattr(launcher, "_prepare_plan", capture_plan)

    assert launcher.main([]) == 0
    check = captured["execution_source_cleanliness"]["checks"][0]
    assert check["phase"] == "diagnostic"
    assert check["passed"] is False
    assert captured["safety"]["managed_child_commands_invoked"] is False


@pytest.mark.parametrize("resume", [False, True])
def test_matrix_execute_rechecks_sources_after_plan_before_gpu_preflight(
    monkeypatch, tmp_path: Path, source_repo, resume: bool
) -> None:
    from scripts.agent import run_matrix_parallel as launcher

    repo, config, paths = source_repo
    out_dir = tmp_path / ("resume" if resume else "fresh")
    args = _matrix_args(out_dir, execute=True, resume=resume)
    commands = [_matrix_command(paths["entry"], tmp_path / "unused", 0)]
    manifest = _matrix_manifest(commands, [tmp_path / "unused"])
    _patch_matrix_config(
        monkeypatch,
        launcher,
        repo=repo,
        config=config,
        args=args,
        manifest=manifest,
        commands=commands,
    )

    def write_plan_then_dirty(**kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_manifest.json").write_text(
            json.dumps(kwargs["manifest"]), encoding="utf-8"
        )
        paths["base"].write_text("dirty: after-plan\n", encoding="utf-8")
        return out_dir, kwargs["manifest"]

    gpu_calls: list[bool] = []
    monkeypatch.setattr(launcher, "_prepare_plan", write_plan_then_dirty)
    monkeypatch.setattr(launcher, "check_nvidia_smi", lambda: gpu_calls.append(True))

    assert launcher.main([]) == 3
    assert gpu_calls == []
    persisted = json.loads((out_dir / "run_manifest.json").read_text())
    checks = persisted["execution_source_cleanliness"]["checks"]
    assert [row["phase"] for row in checks] == ["diagnostic", "pre_execute"]
    assert checks[0]["passed"] is True
    assert checks[1]["passed"] is False
    assert persisted["status"] == "failed"
    assert persisted["execution_lifecycle"]["phase"] == "pre_execute"
    assert persisted["safety"]["managed_child_commands_invoked"] is False


def test_matrix_execute_rechecks_after_inputs_before_first_child(
    monkeypatch, tmp_path: Path, source_repo
) -> None:
    from ecg_adv_gen.config import LaunchError
    from scripts.agent import run_matrix_parallel as launcher

    repo, config, paths = source_repo
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _matrix_manifest([], [])
    manifest["execution_source_cleanliness"] = {"policy": "fixture", "checks": []}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = _matrix_args(run_dir, execute=True)
    queue_calls: list[bool] = []
    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(
        launcher,
        "check_nvidia_smi",
        lambda: {
            "gpus": [
                {
                    "index": "0",
                    "memory_used_mb": "0",
                    "memory_total_mb": "24000",
                    "utilization_gpu_pct": "0",
                }
            ]
        },
    )

    def verify_then_dirty(_manifest):
        paths["base"].write_text("dirty: after-inputs\n", encoding="utf-8")
        return {"passed": True}

    monkeypatch.setattr(launcher, "verify_required_inputs", verify_then_dirty)
    monkeypatch.setattr(
        launcher, "run_matrix_queue", lambda *a, **k: queue_calls.append(True)
    )

    with pytest.raises(LaunchError, match="pre_child"):
        launcher._execute_pipeline(
            args=args,
            config=config,
            commands=[],
            postprocess_commands=[],
            out_dir=run_dir,
            manifest_path=manifest_path,
            gpus=["0"],
        )

    assert queue_calls == []
    persisted = json.loads(manifest_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["execution_lifecycle"]["phase"] == "pre_child"
    assert persisted["execution_source_cleanliness"]["checks"][-1]["passed"] is False
    assert persisted["safety"]["managed_child_commands_invoked"] is False


def test_matrix_queue_blocks_later_child_when_source_becomes_dirty(
    monkeypatch, tmp_path: Path, source_repo
) -> None:
    from ecg_adv_gen.config import LaunchError
    from scripts.agent import run_matrix_parallel as launcher

    repo, config, paths = source_repo
    script = tmp_path / "child.py"
    script.write_text(
        """
import sys
from pathlib import Path

output = Path(sys.argv[1])
output.write_text("ok\\n", encoding="utf-8")
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text("dirty: by-first-child\\n", encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    outputs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    commands = [
        _matrix_command(script, outputs[0], 0, str(paths["base"])),
        _matrix_command(script, outputs[1], 1),
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _matrix_manifest(commands, outputs)
    manifest["execution_source_cleanliness"] = {"policy": "fixture", "checks": []}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = _matrix_args(run_dir, execute=True)
    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(
        launcher,
        "check_nvidia_smi",
        lambda: {
            "gpus": [
                {
                    "index": "0",
                    "memory_used_mb": "0",
                    "memory_total_mb": "24000",
                    "utilization_gpu_pct": "0",
                }
            ]
        },
    )
    monkeypatch.setattr(launcher, "verify_required_inputs", lambda _manifest: {"passed": True})
    monkeypatch.setattr(
        launcher,
        "run_postprocess_serial",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("postprocess must not run")),
    )

    with pytest.raises(LaunchError, match="pre_child"):
        launcher._execute_pipeline(
            args=args,
            config=config,
            commands=commands,
            postprocess_commands=[],
            out_dir=run_dir,
            manifest_path=manifest_path,
            gpus=["0"],
        )

    assert outputs[0].is_file()
    assert not outputs[1].exists()
    persisted = json.loads(manifest_path.read_text())
    dispatch_checks = [
        row
        for row in persisted["execution_source_cleanliness"]["checks"]
        if "command_index" in row
    ]
    assert [row["command_index"] for row in dispatch_checks] == [0, 1]
    assert [row["passed"] for row in dispatch_checks] == [True, False]
    assert persisted["status"] == "failed"
    assert persisted["execution_lifecycle"]["phase"] == "pre_child"
    assert persisted["safety"]["managed_child_commands_invoked"] is True
