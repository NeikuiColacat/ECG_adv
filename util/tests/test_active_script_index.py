"""Tests for the YAML refactor active script index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ecg_adv_gen.config import audit_active_managed_configs, load_experiment_config, write_audit_report


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "configs" / "active_scripts.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
TRACKED_CONFIG_YAMLS = [
    INDEX,
    REPO / "configs" / "active_evidence_registry.yaml",
    *sorted((REPO / "configs" / "defaults").glob("*.yaml")),
    *sorted((REPO / "configs" / "experiments").glob("*.yaml")),
]


def _load_index() -> dict[str, Any]:
    data = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _walk_strings(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)
    elif isinstance(obj, str):
        yield obj


def _walk_key_values(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            child = f"{path}.{key_str}" if path else key_str
            yield child, key_str, value
            yield from _walk_key_values(value, child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{path}[{idx}]"
            yield from _walk_key_values(value, child)


def test_active_script_index_has_no_host_absolute_paths():
    data = _load_index()
    hits = [
        value for value in _walk_strings(data)
        if value.startswith("/home/") or value.startswith("/root/")
    ]
    assert hits == []


def test_tracked_config_sources_have_no_machine_local_runtime_fields():
    """Tracked YAML defines paper protocol, not this host's launch environment."""
    hits = []
    forbidden_key_names = {"python", "python_executable", "executable", "cuda_visible_devices"}
    forbidden_value_fragments = ("CUDA_VISIBLE_DEVICES", ".ssh/", "id_ed25519", "token")
    for path in TRACKED_CONFIG_YAMLS:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item_path, key, value in _walk_key_values(data):
            key_lower = key.lower()
            if key_lower in forbidden_key_names:
                hits.append(f"{path.relative_to(REPO)}:{item_path}: forbidden key {key}")
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith(("/home/", "/root/")):
                    hits.append(f"{path.relative_to(REPO)}:{item_path}: host path {stripped}")
                if any(fragment in stripped for fragment in forbidden_value_fragments):
                    hits.append(f"{path.relative_to(REPO)}:{item_path}: forbidden value {stripped}")
    assert hits == []


def test_local_config_gitignore_allows_only_examples():
    gitignore = (REPO / "configs" / "local" / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert gitignore == ["*", "!.gitignore", "!*.example.yaml"]
    local_files = {path.name for path in (REPO / "configs" / "local").iterdir() if path.is_file()}
    assert ".gitignore" in local_files
    assert "linbinhao_server.example.yaml" in local_files
    assert all(name == ".gitignore" or name.endswith(".example.yaml") for name in local_files)


def test_source_of_truth_paths_exist():
    data = _load_index()
    for key, rel_path in data["source_of_truth"].items():
        assert (REPO / rel_path).exists(), f"{key}: {rel_path}"


def test_managed_experiment_configs_match_runner_entrypoints():
    data = _load_index()
    allowed_status = {"active_wrapped", "planned", "deprecated"}
    for item in data["managed_experiments"]:
        assert item["status"] in allowed_status
        config_path = REPO / item["config"]
        entrypoint = REPO / item["legacy_entrypoint"]
        launcher = REPO / item["launcher"]
        assert config_path.exists()
        assert entrypoint.exists()
        assert launcher.exists()
        config = load_experiment_config(
            config_path,
            LOCAL_EXAMPLE,
            runtime_context={"run_id": f"pytest_{item['name']}"},
        )
        assert config["runner"]["entrypoint"] == item["legacy_entrypoint"]
        assert config["paper_protocol"]["mapping_version"] == item["mapping_version"]


def test_reporting_tools_paths_exist_and_have_outputs():
    data = _load_index()
    for item in data["reporting_tools"]:
        assert item["status"] == "active"
        assert (REPO / item["script"]).exists()
        assert (REPO / item["package"]).exists()
        assert item["output_artifacts"]
        assert item["safety_rules"]


def test_legacy_hold_and_do_not_move_paths_are_tracked_or_present():
    data = _load_index()
    allowed_status = {"do_not_move_yet", "archived", "deprecated"}
    for item in data["legacy_hold"]:
        assert item["status"] in allowed_status
        assert (REPO / item["path"]).exists()
        assert item["future_package_target"]
    for item in data["do_not_commit_or_move"]:
        assert (REPO / item["path"]).exists()
        assert item["reason"]


def test_pipeline_readme_links_refactor_source_doc():
    readme = (REPO / "docs" / "pipelines" / "README.md").read_text(encoding="utf-8")
    assert "refactor_source_of_truth_20260527.md" in readme


def test_active_script_index_file_exists():
    assert INDEX.exists()


def test_active_managed_config_audit_passes_and_writes_reports(tmp_path: Path):
    report = audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
    )
    assert report["passed"] is True
    assert report["failed_count"] == 0
    assert report["managed_experiment_count"] == len(_load_index()["managed_experiments"])
    assert {row["name"] for row in report["rows"]} >= {
        "effnet_direct_k500_v6",
        "effnet_vae_lhat_k500_v6",
        "ecgfounder_direct_k500_v6",
        "ecgfounder_inithead_fullft_k500_v6",
        "ecgfounder_vae_lhat_k500_v6",
        "pn2021_eval_v6_refexcluded",
    }
    assert all(row["command_count"] >= 1 for row in report["rows"])
    assert all(row["command_audit_passed"] is True for row in report["rows"])

    paths = write_audit_report(report, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
