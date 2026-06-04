"""Tests for the YAML refactor active script index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ecg_adv_gen.config import audit_active_managed_configs, load_experiment_config, write_audit_report
from ecg_adv_gen.config.entrypoints import managed_runner_script_names, managed_script_profile


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


def _allowed_prompt_token_reference(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    return (
        "prompt_token" in normalized
        or "center_token" in normalized
        or "token_bank" in normalized
        or "target_token" in normalized
    )


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
                for fragment in forbidden_value_fragments:
                    if fragment not in stripped:
                        continue
                    if fragment == "token" and _allowed_prompt_token_reference(stripped):
                        continue
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
    assert data["source_of_truth"]["pn2021_record_filtering"] == "ecg_adv_gen/evaluation/pn2021_metric_views.py"
    assert data["source_of_truth"]["pn2021_waveform_materialization"] == "ecg_adv_gen/data/pn2021_waveforms.py"
    assert data["source_of_truth"]["synthetic_npz_dataset_contract"] == "ecg_adv_gen/data/synthetic_npz.py"
    assert data["source_of_truth"]["current_handoff_note"] == "docs/codex-handoffs/current_workspace_handoff.md"


def test_active_script_index_declares_yaml_only_launch_surface():
    data = _load_index()
    policy = data["launch_surface_policy"]

    assert policy["default_launcher"] == "scripts/run_experiment.py"
    assert (REPO / policy["default_launcher"]).exists()
    assert policy["managed_index"] == "configs/active_scripts.yaml"
    assert policy["tracked_yaml_required"] is True
    assert policy["new_long_bash_launchers_allowed"] is False
    assert policy["new_dated_matrix_scripts_allowed"] is False
    assert policy["human_exception_requires_index_update"] is True
    assert policy["gpu_binding_location"] == "external_environment_after_nvidia_smi"
    assert set(policy["config_roots"]) == {
        "configs/defaults",
        "configs/experiments",
    }
    assert {
        "no_tracked_gpu_ids",
        "no_host_absolute_paths",
        "run_id_scoped_outputs",
        "legacy_entrypoint_kept_as_thin_wrapper",
    }.issubset(set(policy["safety_rules"]))


def test_active_script_index_declares_documentation_surface_policy():
    data = _load_index()
    policy = data["documentation_surface_policy"]

    assert policy["durable_process_docs"] == "docs/pipelines"
    assert policy["historical_report_archive"] == "docs/reports/archive"
    assert policy["handoff_notes"] == "docs/codex-handoffs"
    assert policy["temporary_report_dirs_allowed"] is False
    assert policy["dated_archive_subdirs_required"] is True
    assert policy["archive_reports_are_active_evidence_by_default"] is False
    assert set(policy["active_evidence_sources"]) == {
        "configs/active_evidence_registry.yaml",
        "registered_run_records",
    }
    assert {
        "move_historical_reports_to_docs_reports_archive_YYYYMMDD",
        "cite_archived_reports_from_pipeline_docs_or_registry",
        "keep_raw_tmp_reports_out_of_default_handoff",
    }.issubset(set(policy["safety_rules"]))
    for key in [
        "durable_process_docs",
        "historical_report_archive",
        "handoff_notes",
    ]:
        assert (REPO / policy[key]).exists()


def test_active_script_index_declares_implementation_surface_policy():
    data = _load_index()
    policy = data["implementation_surface_policy"]

    assert policy["package_root"] == "ecg_adv_gen"
    assert policy["tests_root"] == "util/tests"
    assert set(policy["legacy_wrapper_roots"]) == {
        "scripts/paper",
        "scripts/pgd_cross_center",
        "scripts/triple_labels",
        "scripts/ecgtwin_author_repro",
        "scripts/ecgtwin_gen",
    }
    assert policy["new_shared_logic_target"] == "ecg_adv_gen"
    assert policy["legacy_entrypoints_can_move_without_index_update"] is False
    assert policy["legacy_wrappers_own_new_shared_abstractions"] is False
    assert policy["package_helpers_require_cpu_tests"] is True
    assert {
        "extract_small_cpu_tested_helpers_before_training_loop_moves",
        "keep_script_level_compatibility_exports",
        "update_active_scripts_and_tests_with_entrypoint_moves",
        "do_not_move_full_gpu_orchestration_in_one_step",
    }.issubset(set(policy["safety_rules"]))
    assert (REPO / policy["package_root"]).exists()
    assert (REPO / policy["tests_root"]).exists()
    for rel_path in policy["legacy_wrapper_roots"]:
        assert (REPO / rel_path).exists()


def test_agents_startup_block_points_to_current_agent_handoff_audits():
    first_100 = "\n".join((REPO / "AGENTS.md").read_text(encoding="utf-8").splitlines()[:100])

    assert "configs/active_scripts.yaml" in first_100
    assert "configs/active_evidence_registry.yaml" in first_100
    assert "scripts/agent/audit_agent_workspace.py" in first_100
    assert "handoff_contract.handoff_readiness" in first_100
    assert "handoff_contract.source_of_truth_summary" in first_100
    assert "handoff_contract.source_of_truth_status" in first_100
    assert "active_scripts.config_git_summary" in first_100
    assert "git.blocking_artifact_risks" in first_100
    assert "git.dirty_summary.handoff_gate" in first_100
    assert "git.dirty_summary.by_layer" in first_100
    assert "docs/codex-handoffs/current_workspace_handoff.md" in first_100


def test_current_workspace_handoff_points_to_agent_safe_entrypoints():
    handoff = REPO / "docs" / "codex-handoffs" / "current_workspace_handoff.md"

    assert handoff.exists()
    content = handoff.read_text(encoding="utf-8")
    for required in [
        "scripts/agent/audit_agent_workspace.py",
        "handoff_contract.handoff_readiness",
        "handoff_contract.source_of_truth_summary",
        "handoff_contract.source_of_truth_status",
        "handoff_contract.source_of_truth_review_queue",
        "source_of_truth_missing",
        "source_of_truth_untracked",
        "source_of_truth_dirty",
        "source_of_truth_intent_to_add",
        "intent_to_add_paths",
        "source_of_truth_staged_content",
        "staged_content_paths",
        "unstaged_content_paths",
        "mixed_index_worktree_paths",
        "by_layer_status",
        "staged_diff",
        "unstaged_diff",
        "git.dirty_summary.by_layer.*.paths",
        "status_entries",
        "active_scripts.config_git_inventory",
        "active_scripts.config_git_summary",
        "config_git_summary.untracked_paths",
        "git add -N",
        "git.blocking_artifact_risks",
        "git.dirty_summary.handoff_gate",
        "git.dirty_summary.by_layer",
        "git.guarded_staged_paths",
        "handoff_contract",
        "current_handoff_note",
        "configs/active_evidence_registry.yaml",
        "configs/active_scripts.yaml",
        "scripts/run_experiment.py",
        "ecg_adv_gen/",
        "docs/reports/archive/",
        "model/*",
        "keep_local_only_do_not_stage",
        "micromamba run -n ECGTwin python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts",
    ]:
        assert required in content


def test_eval_crosscenter_uses_package_owned_pn2021_record_filtering():
    content = (REPO / "scripts" / "triple_labels" / "eval_crosscenter.py").read_text(encoding="utf-8")

    assert "filter_pn2021_center_records" in content
    assert "filtered = filter_pn2021_center_records" in content


def test_eval_crosscenter_uses_package_owned_pn2021_cache_selection():
    content = (REPO / "scripts" / "triple_labels" / "eval_crosscenter.py").read_text(encoding="utf-8")

    assert "load_existing_pn2021_eval_cache" in content
    assert "cache_load = load_existing_pn2021_eval_cache" in content


def test_eval_crosscenter_uses_package_owned_pn2021_waveform_materialization():
    content = (REPO / "scripts" / "triple_labels" / "eval_crosscenter.py").read_text(encoding="utf-8")

    assert "materialize_pn2021_center_records" in content
    assert "materialized = materialize_pn2021_center_records" in content


def test_train_ptbxl_uses_package_owned_synthetic_npz_contract():
    content = (REPO / "scripts" / "triple_labels" / "train_ptbxl.py").read_text(encoding="utf-8")

    assert "load_synthetic_npz_arrays" in content
    assert "loaded = load_synthetic_npz_arrays" in content


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


def test_active_legacy_entrypoints_are_registered_as_managed_runner_profiles():
    data = _load_index()
    registered_script_names = managed_runner_script_names()

    assert "synth_online_at_super5.py" in registered_script_names
    for item in data["managed_experiments"]:
        script_name = Path(item["legacy_entrypoint"]).name
        assert script_name in registered_script_names
        profile = managed_script_profile(script_name)
        assert profile.script_name == script_name
        assert profile.family
        assert profile.wrapper_root in data["implementation_surface_policy"]["legacy_wrapper_roots"]


def test_managed_experiments_declare_agent_evidence_scope():
    data = _load_index()
    allowed_scope = {
        "trusted_registry_linked",
        "launch_only",
        "prep_only",
        "evaluation_only",
    }
    assert set(data["evidence_scope_definitions"]) == allowed_scope
    scopes = {item["name"]: item.get("evidence_scope") for item in data["managed_experiments"]}

    assert all(scope in allowed_scope for scope in scopes.values())
    assert {
        name for name, scope in scopes.items()
        if scope == "trusted_registry_linked"
    } == {
        "effnet_direct_k500_v7_sjr_rgq",
        "effnet_vae_lhat_k500_v7_sjr_rgq",
    }
    assert scopes["effnet_v7_sjr_rgq_subset_export"] == "prep_only"
    assert scopes["pn2021_eval_v7_sjr_rgq_refexcluded"] == "evaluation_only"
    assert scopes["pn2021c_effnet_v7_augmix_vs_noaug"] == "evaluation_only"
    assert scopes["ecgfounder_direct_k500_v7_sjr_rgq_matrix"] == "launch_only"
    assert scopes["ecgfounder_vae_lhat_k500_v7_sjr_rgq"] == "launch_only"


def test_experiment_config_inventory_covers_every_tracked_yaml():
    data = _load_index()
    active_configs = {item["config"] for item in data["managed_experiments"]}
    inactive_items = data["inactive_experiment_configs"]
    allowed_status = {"smoke_only", "superseded_by_v7", "historical_unmanaged"}

    inactive_configs = {item["config"] for item in inactive_items}
    actual_configs = {
        str(path.relative_to(REPO))
        for path in sorted((REPO / "configs" / "experiments").glob("*.yaml"))
    }

    assert active_configs.isdisjoint(inactive_configs)
    assert active_configs | inactive_configs == actual_configs
    assert len(inactive_configs) == len(inactive_items)
    for item in inactive_items:
        assert item["status"] in allowed_status
        assert (REPO / item["config"]).exists()
        assert item["reason"]
        if item["config"].endswith("_smoke.yaml"):
            assert item["status"] == "smoke_only"


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
    assert report["launch_surface_policy"] == _load_index()["launch_surface_policy"]
    assert report["launch_surface_policy"]["default_launcher"] == "scripts/run_experiment.py"
    assert report["launch_surface_policy"]["new_long_bash_launchers_allowed"] is False
    assert report["documentation_surface_policy"] == _load_index()["documentation_surface_policy"]
    assert report["documentation_surface_policy"]["historical_report_archive"] == "docs/reports/archive"
    assert report["documentation_surface_policy"]["archive_reports_are_active_evidence_by_default"] is False
    assert report["implementation_surface_policy"] == _load_index()["implementation_surface_policy"]
    assert report["implementation_surface_policy"]["package_root"] == "ecg_adv_gen"
    assert report["implementation_surface_policy"]["legacy_wrappers_own_new_shared_abstractions"] is False
    config_inventory = report["config_git_inventory"]
    inventory_by_path = {item["config"]: item for item in config_inventory}
    expected_config_root_yamls = {
        str(path.relative_to(REPO))
        for root in report["launch_surface_policy"]["config_roots"]
        for path in sorted((REPO / root).glob("*.yaml"))
    }
    assert set(inventory_by_path) == expected_config_root_yamls
    assert {
        "configs/defaults/benchmark_backbone_v7_sjr_rgq_defaults.yaml",
        "configs/defaults/ecgfounder_v7_sjr_rgq_defaults.yaml",
    }.issubset(set(inventory_by_path))
    git_summary = report["config_git_summary"]
    assert git_summary["tracked_yaml_required"] is True
    assert git_summary["total_count"] == len(config_inventory)
    assert git_summary["clean_count"] == sum(1 for item in config_inventory if item["config_git_status"] == "clean")
    assert git_summary["untracked_count"] == sum(1 for item in config_inventory if item["config_git_status"] == "??")
    assert git_summary["tracked_count"] == sum(1 for item in config_inventory if item["config_tracked_by_git"])
    assert git_summary["intent_to_add_count"] == sum(1 for item in config_inventory if item["config_intent_to_add"])
    assert git_summary["staged_content_count"] == sum(
        1 for item in config_inventory
        if item["config_index_status"] not in {" ", "?"} and not item["config_intent_to_add"]
    )
    assert git_summary["unstaged_content_count"] == sum(
        1 for item in config_inventory
        if item["config_worktree_status"] not in {" ", "?"}
    )
    assert git_summary["requires_attention"] is (
        git_summary["untracked_count"] > 0
        or git_summary["missing_count"] > 0
        or git_summary["intent_to_add_count"] > 0
        or git_summary["staged_content_count"] > 0
        or git_summary["unstaged_content_count"] > 0
    )
    assert set(git_summary["untracked_paths"]) == {
        item["config"] for item in config_inventory if item["config_git_status"] == "??"
    }
    assert set(git_summary["intent_to_add_paths"]) == {
        item["config"] for item in config_inventory if item["config_intent_to_add"]
    }
    assert {row["name"] for row in report["rows"]} >= {
        "benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix",
        "effnet_direct_k500_v7_sjr_rgq",
        "effnet_direct_k500_v7_sjr_rgq_matrix",
        "effnet_direct_percent_v7_sjr_rgq_matrix",
        "effnet_v7_sjr_rgq_subset_export",
        "effnet_vae_lhat_k500_v7_sjr_rgq",
        "effnet_vae_lhat_percent_v7_sjr_rgq",
        "ecgfounder_direct_k500_v7_sjr_rgq_matrix",
        "ecgfounder_vae_lhat_k500_v7_sjr_rgq",
        "ecgfounder_direct_k500_v6",
        "ecgfounder_inithead_fullft_k500_v6",
        "ecgfounder_vae_lhat_k500_v6",
        "pn2021_eval_v7_sjr_rgq_refexcluded",
    }
    assert all(row["command_count"] >= 1 for row in report["rows"])
    assert all(row["command_audit_passed"] is True for row in report["rows"])
    assert all(row["config_git_status"] for row in report["rows"])
    assert all(row["config_index_status"] is not None for row in report["rows"])
    assert all(row["config_worktree_status"] is not None for row in report["rows"])
    assert all(row["config"] in inventory_by_path for row in report["rows"])
    assert all(
        row["config_git_status"] == inventory_by_path[row["config"]]["config_git_status"]
        for row in report["rows"]
    )

    paths = write_audit_report(report, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
