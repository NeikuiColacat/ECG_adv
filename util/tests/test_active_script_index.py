"""Tests for the public latest-mainline active script index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ecg_adv_gen.config import (
    audit_active_managed_configs,
    build_runner_commands,
    load_experiment_config,
    validate_experiment_config,
)
from ecg_adv_gen.config.entrypoints import managed_runner_script_names, managed_script_profile


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "configs" / "active_scripts.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
LEGACY_SCRIPT_ROOTS = {
    "scripts/paper",
    "scripts/triple_labels",
    "scripts/pgd_cross_center",
    "scripts/ecgtwin_gen",
    "scripts/ecgtwin_author_repro",
}


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
            yield from _walk_key_values(value, f"{path}[{idx}]")


def _tracked_public_yaml() -> list[Path]:
    return [
        INDEX,
        REPO / "configs" / "active_evidence_registry.yaml",
        *sorted((REPO / "configs" / "defaults").glob("*.yaml")),
        *sorted((REPO / "configs" / "experiments").glob("*.yaml")),
    ]


def _latest_configs(data: dict[str, Any]) -> set[str]:
    return {stage["config"] for stage in data["latest_mainline"]["stages"]}


def _resolve_registry_path(raw: str) -> Path:
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8")) or {}
    context = {"paths": local["paths"]}
    for _ in range(8):
        updated = yaml.safe_load(yaml.safe_dump(context))

        def repl(match: re.Match[str]) -> str:
            cur: Any = updated
            for part in match.group(1).split("."):
                cur = cur[part]
            return str(cur)

        next_context = {
            "paths": {
                key: re.sub(r"\$\{([^}]+)\}", repl, str(value))
                for key, value in updated["paths"].items()
            }
        }
        if next_context == context:
            break
        context = next_context

    def repl_final(match: re.Match[str]) -> str:
        cur: Any = context
        for part in match.group(1).split("."):
            cur = cur[part]
        return str(cur)

    return Path(re.sub(r"\$\{([^}]+)\}", repl_final, raw))


def test_active_script_index_has_no_host_absolute_paths():
    hits = [
        value for value in _walk_strings(_load_index())
        if value.startswith("/home/") or value.startswith("/root/")
    ]

    assert hits == []


def test_tracked_public_yaml_has_no_machine_local_runtime_fields():
    forbidden_key_names = {"python", "python_executable", "executable", "cuda_visible_devices"}
    forbidden_value_fragments = ("CUDA_VISIBLE_DEVICES", ".ssh/", "id_ed25519")
    hits = []
    for path in _tracked_public_yaml():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item_path, key, value in _walk_key_values(data):
            if key.lower() in forbidden_key_names:
                hits.append(f"{path.relative_to(REPO)}:{item_path}: forbidden key {key}")
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped.startswith(("/home/", "/root/")):
                hits.append(f"{path.relative_to(REPO)}:{item_path}: host path {stripped}")
            for fragment in forbidden_value_fragments:
                if fragment in stripped:
                    hits.append(f"{path.relative_to(REPO)}:{item_path}: forbidden value {stripped}")

    assert hits == []


def test_source_of_truth_paths_exist_and_do_not_reference_archives():
    data = _load_index()
    for key, rel_path in data["source_of_truth"].items():
        assert "trash/" not in rel_path, key
        assert "docs/reports/archive" not in rel_path, key
        assert (REPO / rel_path).exists(), f"{key}: {rel_path}"


def test_latest_mainline_declares_locked_protocol_identity():
    latest = _load_index()["latest_mainline"]

    assert latest["method"] == "vae_lhat_threechain_augmix_pn2021c"
    assert latest["launcher"] == "scripts/run_experiment.py"
    assert latest["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert latest["mapping_hash"] == "555ec85d5b51"
    assert latest["class_order"] == ["CD", "HYP", "MI", "NORM", "STTC"]
    assert len(latest["stages"]) == 10


def test_active_evidence_comparison_bundle_records_paper_table_manifests():
    registry = yaml.safe_load((REPO / "configs" / "active_evidence_registry.yaml").read_text(encoding="utf-8"))
    claim = next(
        item for item in registry["active_claims"]
        if item["claim_id"] == "effnet_v7_vae_lhat_improves_direct_k500"
    )
    bundle = claim["comparison_bundle"]
    output_dir = _resolve_registry_path(bundle["output_dir"])

    assert all((output_dir / name).exists() for name in bundle["managed_artifacts"])

    manifest = json.loads((output_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    paper_table_manifests = manifest["artifacts"]["paper_table_manifests"]
    assert set(paper_table_manifests) == {
        "pn2021_all_zero_kept_refexcluded",
        "pn2021_drop_all_zero_refexcluded",
    }
    assert all(Path(path).exists() for path in paper_table_manifests.values())


def test_managed_experiments_are_exactly_latest_mainline_stages():
    data = _load_index()
    latest_configs = _latest_configs(data)
    active_configs = {item["config"] for item in data["managed_experiments"]}

    assert active_configs == latest_configs
    assert len(active_configs) == 10


def test_latest_mainline_configs_use_managed_package_runners():
    data = _load_index()
    managed_by_config = {item["config"]: item for item in data["managed_experiments"]}
    for stage in data["latest_mainline"]["stages"]:
        item = managed_by_config[stage["config"]]
        assert item["status"] == "active_managed"
        assert item["launcher"] == "scripts/run_experiment.py"
        assert item["runner_entrypoint"].startswith("ecg_adv_gen/runner/")

        config_path = REPO / stage["config"]
        config = load_experiment_config(
            config_path,
            LOCAL_EXAMPLE,
            runtime_context={"run_id": f"pytest_latest_{stage['name']}"},
        )
        assert config["runner"]["entrypoint"] == item["runner_entrypoint"]
        assert config["runner"].get("adapter")
        assert "argv" not in config["runner"]
        validate_experiment_config(config, repo_root=REPO)
        assert build_runner_commands(config)


def test_public_experiment_config_inventory_is_latest_mainline_only():
    data = _load_index()
    latest_configs = _latest_configs(data)
    actual_configs = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "configs" / "experiments").glob("*.yaml")
    }

    assert "inactive_experiment_configs" not in data
    assert actual_configs == latest_configs


def test_active_index_no_longer_tracks_historical_inventory():
    forbidden = (
        "trash/",
        "legacy_scripts_20260701",
        "historical_unmanaged",
        "sota_replay_reference",
        "smoke_only",
        "superseded_by_v7",
    )
    hits = [
        value for value in _walk_strings(_load_index())
        if any(marker in value for marker in forbidden)
    ]

    assert hits == []


def test_public_tree_has_no_trash_dependency():
    assert not (REPO / "trash").exists()
    for root in LEGACY_SCRIPT_ROOTS:
        assert not (REPO / root).exists()


def test_managed_runner_profiles_match_active_managed_entrypoints_only():
    data = _load_index()
    active_script_names = {
        Path(item["runner_entrypoint"]).name
        for item in data["managed_experiments"]
    }

    assert managed_runner_script_names() == active_script_names
    for script_name in active_script_names:
        profile = managed_script_profile(script_name)
        assert profile.runner_root == "ecg_adv_gen/runner"
        assert (REPO / profile.relative_path).exists()


def test_reporting_tools_paths_exist():
    data = _load_index()
    for item in data["reporting_tools"]:
        assert item["status"] == "active"
        assert (REPO / item["script"]).exists()
        assert (REPO / item["package"]).exists()
        assert item["output_artifacts"]
        assert item["safety_rules"]


def test_active_managed_audit_reflects_public_mainline_contract():
    report = audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_inputs=False,
    )

    assert report["managed_experiment_count"] == 10
    assert report["latest_mainline"]["passed"] is True
    assert report["passed"] is True
    for item in report["config_git_inventory"]:
        rel_path = item["config"]
        assert not rel_path.startswith("trash/")
        assert rel_path.startswith(("configs/defaults/", "configs/experiments/"))
