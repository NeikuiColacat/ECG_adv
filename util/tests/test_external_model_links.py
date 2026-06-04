"""CPU-only checks for host-local external model handles."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from ecg_adv_gen.config.external_models import audit_external_model_links


REPO = Path(__file__).resolve().parents[2]
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _write_local_config(tmp_path: Path, repo_root: Path, *, ecgfounder_target: Path | None = None) -> Path:
    data_root = tmp_path / "data"
    model_root = data_root / "models"
    ecgfounder_target = ecgfounder_target or data_root / "ecgfounder"
    local = {
        "paths": {
            "project_root": str(repo_root),
            "data_root": str(data_root),
            "model_root": str(model_root),
            "output_root": str(data_root / "runs"),
            "cache_root": str(data_root / "cache"),
            "tmp_root": str(data_root / "tmp"),
        },
        "python": {"executable": sys.executable},
        "safety": {"write_boundary": str(tmp_path)},
        "external_models": {
            "link_root": str(repo_root / "model"),
            "allowed_target_roots": [
                "${paths.model_root}",
                "${paths.data_root}",
            ],
            "repos": {
                "DeepECG": {"target": "${paths.model_root}/DeepECG"},
                "ECGTwin": {"target": "${paths.model_root}/ECGTwin"},
                "advdiff": {"target": "${paths.model_root}/advdiff"},
                "ecg_ptbxl_benchmarking": {"target": "${paths.model_root}/ecg_ptbxl_benchmarking"},
                "ecgfounder": {"target": str(ecgfounder_target)},
            },
        },
    }
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    return path


def _materialize_external_model_layout(repo_root: Path, local_config: Path) -> None:
    raw = yaml.safe_load(local_config.read_text(encoding="utf-8"))
    paths = raw["paths"]
    model_root = Path(paths["model_root"])
    data_root = Path(paths["data_root"])
    targets = {
        "DeepECG": model_root / "DeepECG",
        "ECGTwin": model_root / "ECGTwin",
        "advdiff": model_root / "advdiff",
        "ecg_ptbxl_benchmarking": model_root / "ecg_ptbxl_benchmarking",
        "ecgfounder": data_root / "ecgfounder",
    }
    (repo_root / "model").mkdir(parents=True, exist_ok=True)
    for name, target in targets.items():
        _git_init(target)
        (repo_root / "model" / name).symlink_to(target)


def test_external_model_audit_resolves_configured_model_handles(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    local_config = _write_local_config(tmp_path, repo_root)
    _materialize_external_model_layout(repo_root, local_config)

    report = audit_external_model_links(
        repo_root=repo_root,
        local_config_path=local_config,
        require_existing=True,
    )

    assert report["passed"] is True
    assert report["model_count"] == 5
    assert report["verified_handle_paths"] == [
        "model/DeepECG",
        "model/ECGTwin",
        "model/advdiff",
        "model/ecg_ptbxl_benchmarking",
        "model/ecgfounder",
    ]
    by_name = {item["name"]: item for item in report["models"]}
    assert by_name["ecgfounder"]["target"] == str(tmp_path / "data" / "ecgfounder")
    assert by_name["ecgfounder"]["target_matches_config"] is True
    assert by_name["ecgfounder"]["target_under_allowed_root"] is True
    assert by_name["ECGTwin"]["url"] == "https://github.com/Raiiyf/ECGTwin.git"


def test_external_model_audit_rejects_unconfigured_symlink_targets(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    local_config = _write_local_config(tmp_path, repo_root)
    _materialize_external_model_layout(repo_root, local_config)
    wrong_target = tmp_path / "wrong" / "ECGTwin"
    _git_init(wrong_target)
    handle = repo_root / "model" / "ECGTwin"
    handle.unlink()
    handle.symlink_to(wrong_target)

    report = audit_external_model_links(
        repo_root=repo_root,
        local_config_path=local_config,
        require_existing=True,
    )

    assert report["passed"] is False
    by_name = {item["name"]: item for item in report["models"]}
    assert by_name["ECGTwin"]["target_matches_config"] is False
    assert any(issue["code"] == "external_model_target_mismatch" for issue in by_name["ECGTwin"]["issues"])


def test_external_model_checker_cli_reports_current_repo_handles():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "agent" / "check_external_models.py"),
            "--local-config",
            str(LOCAL_EXAMPLE),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["model_count"] == 5
    assert "model/ecgfounder" in report["verified_handle_paths"]


def test_bootstrap_model_repos_check_only_uses_env_targets_and_includes_ecgfounder(tmp_path: Path):
    repo_root = tmp_path / "repo"
    model_root = tmp_path / "models"
    ecgfounder_root = tmp_path / "ecgfounder"
    (repo_root / "model").mkdir(parents=True)
    for name in ["DeepECG", "ECGTwin", "advdiff", "ecg_ptbxl_benchmarking"]:
        target = model_root / name
        _git_init(target)
        (repo_root / "model" / name).symlink_to(target)
    _git_init(ecgfounder_root)
    (repo_root / "model" / "ecgfounder").symlink_to(ecgfounder_root)

    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "bootstrap_model_repos.sh"),
            "--check-only",
            "--repo-root",
            str(repo_root),
            "--model-root",
            str(model_root),
            "--ecgfounder-root",
            str(ecgfounder_root),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[ok] model/ecgfounder ->" in result.stdout
    assert "[clone]" not in result.stdout


def test_bootstrap_model_repos_data_root_updates_default_targets(tmp_path: Path):
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    model_root = data_root / "models"
    ecgfounder_root = data_root / "ecgfounder"
    (repo_root / "model").mkdir(parents=True)
    for name in ["DeepECG", "ECGTwin", "advdiff", "ecg_ptbxl_benchmarking"]:
        target = model_root / name
        _git_init(target)
        (repo_root / "model" / name).symlink_to(target)
    _git_init(ecgfounder_root)
    (repo_root / "model" / "ecgfounder").symlink_to(ecgfounder_root)

    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "bootstrap_model_repos.sh"),
            "--check-only",
            "--repo-root",
            str(repo_root),
            "--data-root",
            str(data_root),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"[ok] model/DeepECG -> {model_root / 'DeepECG'}" in result.stdout
    assert f"[ok] model/ecgfounder -> {ecgfounder_root}" in result.stdout
