from __future__ import annotations

import json
import subprocess
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
