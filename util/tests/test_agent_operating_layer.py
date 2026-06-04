"""CPU-only tests for the agent operating layer registry and bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml
import pytest

from ecg_adv_gen.evidence import (
    audit_active_evidence_registry,
    build_comparison_bundle,
    build_legacy_vae_lhat_manifest,
    EvidenceAuditError,
    load_evidence_registry,
)
from ecg_adv_gen.evidence.registry import _audit_manifest, _summarize_dirty_workspace
from ecg_adv_gen.evidence.registry import _guarded_staged_paths
from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES


REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs" / "active_evidence_registry.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _walk_strings(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)
    elif isinstance(obj, str):
        yield obj


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {key: "" for key in METRICS_FIELDNAMES}
            full.update(row)
            writer.writerow(full)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_rows(run_id: str, auroc: float, auprc: float) -> list[dict[str, str]]:
    rows = []
    for center in ["ningbo", "georgia"]:
        for metric, value in [("macro_auroc", auroc), ("macro_auprc", auprc)]:
            rows.append(
                {
                    "run_id": run_id,
                    "source_file": f"/tmp/{run_id}_{center}.json",
                    "artifact_type": "eval_result",
                    "dataset": "pn2021",
                    "view": "pn2021_all_zero_kept_refexcluded",
                    "canonical_view": "pn2021_all_zero_kept_refexcluded",
                    "scope": "center",
                    "center": center,
                    "metric": metric,
                    "value": str(value),
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": "CD|HYP|MI|NORM|STTC",
                }
            )
        for metric, value in [("drop_all_zero_macro_auroc", auroc + 0.1), ("drop_all_zero_macro_auprc", auprc + 0.1)]:
            rows.append(
                {
                    "run_id": run_id,
                    "source_file": f"/tmp/{run_id}_{center}.json",
                    "artifact_type": "eval_result",
                    "dataset": "pn2021",
                    "view": "pn2021_drop_all_zero_refexcluded",
                    "canonical_view": "pn2021_drop_all_zero_refexcluded",
                    "scope": "center",
                    "center": center,
                    "metric": metric,
                    "value": str(value),
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": "CD|HYP|MI|NORM|STTC",
                }
            )
    return rows


def test_active_evidence_registry_is_tracked_and_path_safe():
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    host_paths = [
        value for value in _walk_strings(raw)
        if value.startswith("/home/") or value.startswith("/root/")
    ]
    assert host_paths == []
    assert raw["active_claims"][0]["protocol"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert raw["active_claims"][0]["comparison_bundle"]["status"] == "built"
    policy = raw["evidence_surface_policy"]
    assert policy["trusted_claims_source"] == "active_claims"
    assert policy["paper_tables_source"] == "registered_artifacts"
    assert policy["archived_reports_are_evidence_by_default"] is False
    assert policy["managed_launch_configs_are_evidence_by_default"] is False
    assert {
        "cite_metrics_through_registry_or_registered_run_records",
        "do_not_promote_archived_reports_without_registry_link",
        "do_not_treat_launch_only_configs_as_paper_evidence",
    }.issubset(set(policy["safety_rules"]))


def test_registry_resolves_with_local_config():
    resolved = load_evidence_registry(REGISTRY, LOCAL_EXAMPLE)
    claim = resolved["active_claims"][0]
    out_dir = claim["comparison_bundle"]["output_dir"]
    assert out_dir.startswith("/home/linbinhao/")
    assert claim["protocol"]["kshot"]["ref_meta_files"]["ningbo"].endswith(".ref_meta.json")


def test_agent_workspace_audit_cpu_only_without_artifact_scan():
    report = audit_active_evidence_registry(
        repo_root=REPO,
        registry_path=REGISTRY,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_artifacts=False,
        check_git=False,
    )
    assert report["passed"] is True
    assert report["claim_count"] == 1
    policy = report["evidence_surface_policy"]
    assert policy["trusted_claims_source"] == "active_claims"
    assert policy["archived_reports_are_evidence_by_default"] is False


def test_agent_workspace_git_audit_reports_guarded_staged_paths():
    report = audit_active_evidence_registry(
        repo_root=REPO,
        registry_path=REGISTRY,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_artifacts=False,
        check_git=True,
    )

    assert "guarded_staged_count" in report["git"]
    assert "guarded_staged_paths" in report["git"]
    assert report["git"]["guarded_staged_paths"] == []
    assert not [issue for issue in report["issues"] if issue["code"] == "guarded_path_staged"]
    assert "blocking_artifact_risks" in report["git"]
    risks = report["git"]["blocking_artifact_risks"]
    assert {"risk", "count", "paths"}.issubset(risks[0])
    risk_names = {item["risk"] for item in risks}
    assert {
        "blocked_tracked_artifacts",
        "blocked_staged_artifacts",
        "guarded_staged_paths",
        "dirty_guarded_local_only_paths",
    }.issubset(risk_names)


def test_agent_workspace_cli_combines_registry_and_active_script_audits():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "agent" / "audit_agent_workspace.py"),
            "--skip-existing-artifacts",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["claim_count"] == 1
    active_scripts = report["active_scripts"]
    assert active_scripts["passed"] is True
    assert active_scripts["failed_count"] == 0
    assert active_scripts["managed_experiment_count"] >= 25
    assert active_scripts["index"].endswith("configs/active_scripts.yaml")
    config_git_inventory = active_scripts["config_git_inventory"]
    inventory_paths = {item["config"] for item in config_git_inventory}
    config_git_summary = active_scripts["config_git_summary"]
    assert config_git_summary["tracked_yaml_required"] is True
    assert config_git_summary["total_count"] == len(config_git_inventory)
    assert config_git_summary["untracked_count"] == len(config_git_summary["untracked_paths"])
    assert all(row["config"] in inventory_paths for row in active_scripts["rows"])
    assert "configs/defaults/benchmark_backbone_v7_sjr_rgq_defaults.yaml" in inventory_paths
    assert "configs/defaults/ecgfounder_v7_sjr_rgq_defaults.yaml" in inventory_paths
    contract = report["handoff_contract"]
    assert contract["schema_version"] == 1
    assert contract["startup_sequence"] == [
        "read_AGENTS_first_100_lines",
        "run_scripts_agent_audit_agent_workspace",
        "inspect_git_dirty_summary_handoff_gate",
        "inspect_active_scripts_policies",
        "inspect_active_scripts_config_git_summary",
    ]
    assert contract["source_of_truth"]["active_evidence_registry"].endswith("configs/active_evidence_registry.yaml")
    assert contract["source_of_truth"]["active_scripts_index"].endswith("configs/active_scripts.yaml")
    source_status = contract["source_of_truth_status"]
    assert source_status
    status_by_name = {item["name"]: item for item in source_status}
    assert status_by_name["active_evidence_registry"]["path"] == "configs/active_evidence_registry.yaml"
    assert status_by_name["active_evidence_registry"]["exists"] is True
    assert isinstance(status_by_name["active_evidence_registry"]["tracked_by_git"], bool)
    assert status_by_name["active_evidence_registry"]["git_status"]
    assert status_by_name["active_evidence_registry"]["layer"] == "configs"
    assert status_by_name["active_scripts_index"]["path"] == "configs/active_scripts.yaml"
    assert status_by_name["active_scripts_index"]["layer"] == "configs"
    assert status_by_name["yaml_refactor_plan"]["path"] == "docs/pipelines/ai_agent_workspace_refactor_plan_20260604.md"
    assert status_by_name["yaml_refactor_plan"]["layer"] == "docs"
    assert status_by_name["synthetic_npz_dataset_contract"]["path"] == "ecg_adv_gen/data/synthetic_npz.py"
    assert status_by_name["synthetic_npz_dataset_contract"]["layer"] == "package"
    assert all(item["required_for_handoff"] is True for item in source_status)
    assert all(not Path(item["path"]).is_absolute() for item in source_status)
    assert all("index_status" in item for item in source_status)
    assert all("worktree_status" in item for item in source_status)
    assert all(isinstance(item["intent_to_add"], bool) for item in source_status)
    for item in source_status:
        assert set(item["staged_diff"]) == {
            "has_diff",
            "added_lines",
            "deleted_lines",
            "binary",
        }
        assert set(item["unstaged_diff"]) == {
            "has_diff",
            "added_lines",
            "deleted_lines",
            "binary",
        }
        assert isinstance(item["staged_diff"]["has_diff"], bool)
        assert isinstance(item["unstaged_diff"]["has_diff"], bool)
    source_summary = contract["source_of_truth_summary"]
    assert source_summary["total_count"] == len(source_status)
    assert source_summary["missing_count"] == len(source_summary["missing_paths"])
    assert source_summary["untracked_count"] == len(source_summary["untracked_paths"])
    assert source_summary["dirty_count"] == len(source_summary["dirty_paths"])
    assert source_summary["intent_to_add_count"] == len(source_summary["intent_to_add_paths"])
    assert source_summary["staged_content_count"] == len(source_summary["staged_content_paths"])
    assert source_summary["unstaged_content_count"] == len(source_summary["unstaged_content_paths"])
    assert source_summary["mixed_index_worktree_count"] == len(source_summary["mixed_index_worktree_paths"])
    intent_to_add_items = [item for item in source_status if item["intent_to_add"]]
    staged_content_items = [item for item in source_status if item["index_status"] not in {" ", "?"}]
    unstaged_content_items = [item for item in source_status if item["worktree_status"] not in {" ", "?"}]
    mixed_index_worktree_items = [
        item
        for item in source_status
        if item["index_status"] not in {" ", "?"} and item["worktree_status"] not in {" ", "?"}
    ]
    assert source_summary["intent_to_add_count"] == len(intent_to_add_items)
    assert source_summary["intent_to_add_paths"] == [item["path"] for item in intent_to_add_items]
    assert source_summary["staged_content_count"] == len(staged_content_items)
    assert source_summary["staged_content_paths"] == [item["path"] for item in staged_content_items]
    assert source_summary["unstaged_content_count"] == len(unstaged_content_items)
    assert source_summary["unstaged_content_paths"] == [item["path"] for item in unstaged_content_items]
    assert source_summary["mixed_index_worktree_count"] == len(mixed_index_worktree_items)
    assert source_summary["mixed_index_worktree_paths"] == [item["path"] for item in mixed_index_worktree_items]
    if intent_to_add_items:
        assert all(item["git_status"] == "A" for item in intent_to_add_items)
        assert all(item["index_status"] == " " for item in intent_to_add_items)
        assert all(item["worktree_status"] == "A" for item in intent_to_add_items)
        assert all(item["staged_diff"]["has_diff"] is False for item in intent_to_add_items)
        assert all(item["unstaged_diff"]["has_diff"] is True for item in intent_to_add_items)
    for item in mixed_index_worktree_items:
        assert item["staged_diff"]["has_diff"] is True
        assert item["unstaged_diff"]["has_diff"] is True
    review_queue = contract["source_of_truth_review_queue"]
    assert len(review_queue) == source_summary["dirty_count"]
    queue_by_path = {item["path"]: item for item in review_queue}
    assert set(queue_by_path) == set(source_summary["dirty_paths"])
    for item in review_queue:
        assert set(item) == {
            "path",
            "name",
            "layer",
            "git_status",
            "review_state",
            "next_action",
            "staged_diff",
            "unstaged_diff",
        }
        assert item["next_action"]
    for item in mixed_index_worktree_items:
        queue_item = queue_by_path[item["path"]]
        assert queue_item["review_state"] == "mixed_index_worktree"
        assert queue_item["next_action"] == "review_staged_and_unstaged_diffs_then_stage_or_unstage_consistently"
        assert queue_item["staged_diff"] == item["staged_diff"]
        assert queue_item["unstaged_diff"] == item["unstaged_diff"]
    for item in intent_to_add_items:
        queue_item = queue_by_path[item["path"]]
        assert queue_item["review_state"] == "intent_to_add"
        assert queue_item["next_action"] == "review_worktree_diff_then_stage_or_declassify"
    priority_order = {
        "mixed_index_worktree": 0,
        "staged_content": 1,
        "intent_to_add": 2,
        "unstaged_content": 3,
        "dirty": 4,
    }
    assert [priority_order[item["review_state"]] for item in review_queue] == sorted(
        priority_order[item["review_state"]] for item in review_queue
    )
    assert source_summary["requires_attention"] is (source_summary["dirty_count"] > 0)
    assert "configs" in source_summary["by_layer"]
    layer_status = source_summary["by_layer_status"]
    assert set(layer_status) == set(source_summary["by_layer"])
    for layer, layer_item in layer_status.items():
        layer_items = [item for item in source_status if item["layer"] == layer]
        assert layer_item["total_count"] == len(layer_items)
        assert layer_item["clean_count"] == sum(1 for item in layer_items if item["git_status"] == "clean")
        assert layer_item["dirty_count"] == sum(1 for item in layer_items if item["git_status"] != "clean")
        assert layer_item["intent_to_add_count"] == sum(1 for item in layer_items if item["intent_to_add"])
        assert layer_item["staged_content_count"] == sum(1 for item in layer_items if item["index_status"] not in {" ", "?"})
        assert layer_item["unstaged_content_count"] == sum(1 for item in layer_items if item["worktree_status"] not in {" ", "?"})
        assert layer_item["mixed_index_worktree_count"] == sum(
            1
            for item in layer_items
            if item["index_status"] not in {" ", "?"} and item["worktree_status"] not in {" ", "?"}
        )
        assert layer_item["paths"] == [item["path"] for item in layer_items]
    assert set(source_summary["by_git_status"]) >= {"clean"}
    issue_codes = {issue["code"] for issue in report["issues"]}
    if source_summary["untracked_count"]:
        assert "source_of_truth_untracked" in issue_codes
    else:
        assert "source_of_truth_untracked" not in issue_codes
    if source_summary["dirty_count"]:
        assert "source_of_truth_dirty" in issue_codes
    else:
        assert "source_of_truth_dirty" not in issue_codes
    if source_summary["intent_to_add_count"]:
        assert "source_of_truth_intent_to_add" in issue_codes
    else:
        assert "source_of_truth_intent_to_add" not in issue_codes
    if source_summary["staged_content_count"]:
        assert "source_of_truth_staged_content" in issue_codes
    else:
        assert "source_of_truth_staged_content" not in issue_codes
    assert "source_of_truth_missing" not in issue_codes
    readiness = contract["handoff_readiness"]
    assert readiness["ready_for_handoff"] is False
    assert readiness["ready_for_commit"] is False
    source_control_ready_expected = (
        source_summary["dirty_count"] == 0
        and source_summary["staged_content_count"] == 0
        and source_summary["unstaged_content_count"] == 0
        and source_summary["untracked_count"] == 0
        and source_summary["missing_count"] == 0
        and config_git_summary["dirty_count"] == 0
        and config_git_summary["staged_content_count"] == 0
        and config_git_summary["unstaged_content_count"] == 0
        and config_git_summary["untracked_count"] == 0
        and config_git_summary["missing_count"] == 0
        and readiness["hard_reason_count"] == 0
    )
    assert readiness["source_control_ready"] is source_control_ready_expected
    assert readiness["status"] == "attention_required"
    readiness_reasons = {item["code"]: item for item in readiness["reasons"]}
    if source_summary["untracked_count"]:
        assert readiness_reasons["source_of_truth_untracked"]["count"] == source_summary["untracked_count"]
    if source_summary["dirty_count"]:
        assert readiness_reasons["source_of_truth_dirty"]["count"] == source_summary["dirty_count"]
    if source_summary["intent_to_add_count"]:
        assert readiness_reasons["source_of_truth_intent_to_add"]["count"] == source_summary["intent_to_add_count"]
    if source_summary["staged_content_count"]:
        assert readiness_reasons["source_of_truth_staged_content"]["count"] == source_summary["staged_content_count"]
    if config_git_summary["requires_attention"]:
        assert readiness_reasons["managed_config_git_requires_attention"]["count"] == config_git_summary["dirty_count"]
        assert readiness_reasons["managed_config_git_requires_attention"]["paths"] == config_git_summary["dirty_paths"]
    assert readiness_reasons["dirty_guarded_local_only_paths"]["count"] == 5
    assert "git.guarded_staged_paths" in readiness["must_be_empty"]
    assert "handoff_contract.source_of_truth_summary.missing_paths" in readiness["must_be_empty"]
    assert "active_scripts.config_git_summary.missing_paths" in readiness["must_be_empty"]
    assert "handoff_contract.source_of_truth_summary.untracked_paths" in readiness["review_required"]
    assert "handoff_contract.source_of_truth_review_queue" in readiness["review_required"]
    assert "handoff_contract.source_of_truth_summary.staged_content_paths" in readiness["review_required"]
    assert "handoff_contract.source_of_truth_summary.unstaged_content_paths" in readiness["review_required"]
    assert "active_scripts.config_git_summary.untracked_paths" in readiness["review_required"]
    assert "active_scripts.config_git_summary.intent_to_add_paths" in readiness["review_required"]
    assert "active_scripts.config_git_summary.staged_content_paths" in readiness["review_required"]
    assert "active_scripts.config_git_summary.unstaged_content_paths" in readiness["review_required"]
    assert "git.dirty_summary.handoff_gate" in readiness["review_required"]
    assert "git.dirty_summary.by_layer" in readiness["review_required"]
    assert "git.dirty_summary.by_layer.*.paths" in readiness["review_required"]
    assert contract["dirty_gate_json_path"] == "git.dirty_summary.handoff_gate"
    assert contract["current_handoff_note"]["path"] == "docs/codex-handoffs/current_workspace_handoff.md"
    assert contract["current_handoff_note"]["exists"] is True
    assert isinstance(contract["current_handoff_note"]["tracked_by_git"], bool)
    assert contract["current_handoff_note"]["git_status"]
    assert "index_status" in contract["current_handoff_note"]
    assert "worktree_status" in contract["current_handoff_note"]
    assert isinstance(contract["current_handoff_note"]["intent_to_add"], bool)
    assert contract["policies"]["evidence_surface_policy"]["trusted_claims_source"] == "active_claims"
    assert contract["policies"]["evidence_surface_policy"]["managed_launch_configs_are_evidence_by_default"] is False
    assert contract["policies"]["launch_surface_policy"]["default_launcher"] == "scripts/run_experiment.py"
    assert contract["policies"]["documentation_surface_policy"]["historical_report_archive"] == "docs/reports/archive"
    assert contract["policies"]["implementation_surface_policy"]["package_root"] == "ecg_adv_gen"


def test_dirty_workspace_summary_groups_agent_handoff_layers():
    summary = _summarize_dirty_workspace(
        [
            "MM configs/active_scripts.yaml",
            " M ecg_adv_gen/config/loader.py",
            "?? configs/experiments/new_eval.yaml",
            " M model/ECGTwin",
            "R  docs/tmp_md/old.md -> docs/reports/archive/20260501/old.md",
            " M util/tests/test_config_loader.py",
            " M scripts/run_experiment.py",
            "A  .codex/skills/ecg-agent-retrospective/SKILL.md",
            " M AGENTS.md",
        ],
        do_not_commit={"model/ECGTwin"},
    )

    assert summary["total_entries"] == 9
    assert summary["staged_entries"] == 3
    assert summary["unstaged_entries"] == 6
    assert summary["untracked_entries"] == 1
    assert summary["guarded_dirty_count"] == 1
    assert summary["guarded_dirty_paths"] == ["model/ECGTwin"]

    by_layer = summary["by_layer"]
    assert by_layer["configs"]["total_entries"] == 2
    assert by_layer["configs"]["staged_entries"] == 1
    assert by_layer["configs"]["untracked_entries"] == 1
    assert by_layer["configs"]["paths"] == [
        "configs/active_scripts.yaml",
        "configs/experiments/new_eval.yaml",
    ]
    assert by_layer["configs"]["staged_paths"] == ["configs/active_scripts.yaml"]
    assert by_layer["configs"]["unstaged_paths"] == ["configs/active_scripts.yaml"]
    assert by_layer["configs"]["untracked_paths"] == ["configs/experiments/new_eval.yaml"]
    assert by_layer["configs"]["status_entries"] == [
        {
            "path": "configs/active_scripts.yaml",
            "git_status": "MM",
            "index_status": "M",
            "worktree_status": "M",
        },
        {
            "path": "configs/experiments/new_eval.yaml",
            "git_status": "??",
            "index_status": "?",
            "worktree_status": "?",
        },
    ]
    assert by_layer["package"]["unstaged_entries"] == 1
    assert by_layer["package"]["paths"] == ["ecg_adv_gen/config/loader.py"]
    assert by_layer["package"]["staged_paths"] == []
    assert by_layer["package"]["unstaged_paths"] == ["ecg_adv_gen/config/loader.py"]
    assert by_layer["external_models"]["guarded_dirty_count"] == 1
    assert by_layer["external_models"]["guarded_dirty_paths"] == ["model/ECGTwin"]
    assert by_layer["docs"]["staged_entries"] == 1
    assert by_layer["docs"]["paths"] == ["docs/reports/archive/20260501/old.md"]
    assert by_layer["tests"]["unstaged_entries"] == 1
    assert by_layer["scripts"]["unstaged_entries"] == 1
    assert by_layer["agent_instructions"]["unstaged_entries"] == 1
    assert by_layer["codex_skills"]["staged_entries"] == 1
    assert by_layer["external_models"]["handoff_action"] == "keep_local_only_do_not_stage"
    assert by_layer["configs"]["handoff_action"] == "verify_yaml_index_and_path_boundaries"
    assert by_layer["package"]["handoff_action"] == "run_focused_cpu_tests_before_handoff"
    assert by_layer["scripts"]["handoff_action"] == "keep_legacy_wrappers_thin_and_index_managed"
    assert by_layer["docs"]["handoff_action"] == "separate_pipeline_docs_from_archived_reports"
    assert by_layer["tests"]["handoff_action"] == "run_related_cpu_tests"
    assert by_layer["agent_instructions"]["handoff_action"] == "review_first_100_lines_before_writes"

    gate = summary["handoff_gate"]
    assert gate["requires_attention"] is True
    assert gate["dirty_layer_count"] == len(by_layer)
    assert gate["guarded_paths_require_local_only"] == ["model/ECGTwin"]
    assert {
        "layer": "external_models",
        "handoff_action": "keep_local_only_do_not_stage",
        "dirty_entries": 1,
        "guarded_dirty_count": 1,
    } in gate["actions"]
    assert {
        "layer": "configs",
        "handoff_action": "verify_yaml_index_and_path_boundaries",
        "dirty_entries": 2,
        "guarded_dirty_count": 0,
    } in gate["actions"]


def test_guarded_staged_paths_match_exact_prefix_and_glob_policy_entries():
    guarded = _guarded_staged_paths(
        [
            "model/ECGTwin",
            "datasets/PTBXL/cache.npy",
            "docs/reports/archive/20260501/report.md",
            "outputs/run/latest.json",
            "ecg_adv_gen/evidence/registry.py",
        ],
        [
            "model/ECGTwin",
            "datasets/",
            "outputs/",
            "*.npy",
        ],
    )

    assert guarded == [
        "model/ECGTwin",
        "datasets/PTBXL/cache.npy",
        "outputs/run/latest.json",
    ]


def test_trusted_mainline_missing_manifest_is_an_error():
    issues = []
    summary = _audit_manifest(
        "vae_lhat",
        {"status": "trusted", "manifest": ""},
        {"claim_id": "claim", "status": "trusted", "paper_use": "mainline"},
        issues,
    )
    assert summary == {"declared": False, "path": ""}
    assert issues[0]["level"] == "error"
    assert issues[0]["code"] == "manifest_not_declared"


def test_legacy_backfilled_manifest_contract(tmp_path: Path):
    artifact_records = {}
    for name in [
        "launch_config.json",
        "train_result.json",
        "early_stop_info.json",
        "eval_result_v7_exclrefs_crop1000.json",
        "best_model.pt",
    ]:
        path = tmp_path / name
        path.write_bytes(f"{name}\n".encode("utf-8"))
        artifact_records[name] = {
            "exists": True,
            "path": str(path),
            "sha256": _sha256(path),
        }
    manifest_path = tmp_path / "run_manifest.backfilled.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_kind": "legacy_backfilled_manifest",
                "status": "succeeded",
                "claim_id": "claim",
                "method_key": "vae_lhat",
                "run_id": "run",
                "protocol": {
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                    "target_centers": ["ningbo"],
                },
                "centers": [
                    {
                        "center": "ningbo",
                        "final_eval": {
                            "mapping_version": "vtest",
                            "mapping_hash": "htest",
                        },
                        "artifacts": artifact_records,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    issues = []
    summary = _audit_manifest(
        "vae_lhat",
        {
            "status": "trusted",
            "traceability": "legacy_backfilled_manifest",
            "run_id": "run",
            "manifest": str(manifest_path),
        },
        {
            "claim_id": "claim",
            "status": "trusted",
            "paper_use": "mainline",
            "protocol": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                "target_centers": ["ningbo"],
            },
        },
        issues,
    )
    assert summary["manifest_kind"] == "legacy_backfilled_manifest"
    assert issues == []


def test_build_legacy_vae_lhat_manifest_from_fake_run(tmp_path: Path):
    output_root = tmp_path / "out"
    run_root = output_root / "effnet_vae_lhat_k500_v7_sjr_rgq" / "run"
    center_dir = run_root / "ningbo_run"
    center_dir.mkdir(parents=True)
    launch = {
        "args": {"center": "ningbo", "seed": 7, "init_ckpt": "/tmp/init.pt"},
        "train_cmd": ["python", "train.py"],
        "eval_cmd": ["python", "eval.py"],
    }
    train_result = {
        "args": {
            "ref_meta_json": "/tmp/ref.json",
            "target_real_npz": "/tmp/signals.npz",
            "synth_npz": "/tmp/latent.npz",
        }
    }
    early_stop = {
        "es_metric": "target_macro_auprc",
        "best_epoch": 2,
        "best_metric": 0.7,
        "stopped_epoch": 4,
        "early_stopped": False,
    }
    eval_result = {
        "pn2021": {
            "avg_macro_auroc": 0.8,
            "avg_macro_auprc": 0.6,
            "avg_drop_all_zero_macro_auroc": 0.9,
            "avg_drop_all_zero_macro_auprc": 0.7,
        },
        "label_mapping": {
            "pn2021_super5": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
            }
        },
    }
    for name, payload in [
        ("launch_config.json", launch),
        ("train_result.json", train_result),
        ("early_stop_info.json", early_stop),
        ("eval_result_v7_exclrefs_crop1000.json", eval_result),
    ]:
        (center_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (center_dir / "best_model.pt").write_bytes(b"fake checkpoint")
    for extra in ["metrics_long.csv", "artifact_manifest.json", "paper_table.csv"]:
        (run_root / extra).write_text("x\n", encoding="utf-8")

    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {output_root}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
schema_version: 1
active_claims:
  - claim_id: claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo]
      kshot: {{k: 500, seed: 7, ref_excluded: true}}
      selection: {{policy: fake}}
    methods:
      vae_lhat:
        status: trusted
        traceability: legacy_backfilled_manifest
        method_family: vae_latent_hull_online_at
        run_id: run
        config: configs/experiments/fake.yaml
        run_root: ${{paths.output_root}}/effnet_vae_lhat_k500_v7_sjr_rgq/run
        manifest: ${{paths.output_root}}/effnet_vae_lhat_k500_v7_sjr_rgq/run/run_manifest.backfilled.json
        metrics_long: {run_root / 'metrics_long.csv'}
        artifact_manifest: {run_root / 'artifact_manifest.json'}
        paper_tables:
          pn2021_all_zero_kept_refexcluded: {run_root / 'paper_table.csv'}
""",
        encoding="utf-8",
    )
    manifest = build_legacy_vae_lhat_manifest(
        repo_root=REPO,
        registry_path=registry,
        local_config_path=local_config,
        claim_id="claim",
        method_key="vae_lhat",
        write_boundary=tmp_path,
    )
    assert manifest["manifest_kind"] == "legacy_backfilled_manifest"
    assert manifest["centers"][0]["center"] == "ningbo"
    assert manifest["centers"][0]["artifacts"]["best_model.pt"]["sha256"]
    assert manifest["centers"][0]["final_eval"]["mapping_version"] == "vtest"
    assert (run_root / "run_manifest.backfilled.json").exists()


def test_backfill_manifest_respects_write_boundary(tmp_path: Path):
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {tmp_path / 'out'}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
schema_version: 1
active_claims:
  - claim_id: claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo]
      kshot: {k: 500, seed: 7, ref_excluded: true}
      selection: {policy: fake}
    methods:
      vae_lhat:
        status: trusted
        traceability: legacy_backfilled_manifest
        method_family: vae_latent_hull_online_at
        run_id: run
        config: configs/experiments/fake.yaml
        run_root: ${paths.output_root}/missing
        manifest: ${paths.output_root}/missing/run_manifest.backfilled.json
        metrics_long: ${paths.output_root}/metrics.csv
        artifact_manifest: ${paths.output_root}/artifact_manifest.json
        paper_tables: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceAuditError, match="outside write boundary"):
        build_legacy_vae_lhat_manifest(
            repo_root=REPO,
            registry_path=registry,
            local_config_path=local_config,
            claim_id="claim",
            method_key="vae_lhat",
            write_boundary=tmp_path / "other",
        )


def test_build_comparison_bundle_from_registry(tmp_path: Path):
    direct_metrics = _write_metrics(tmp_path / "direct.csv", _metric_rows("direct", 0.8, 0.5))
    vae_metrics = _write_metrics(tmp_path / "vae.csv", _metric_rows("vae", 0.83, 0.56))
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {tmp_path / 'out'}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
schema_version: 1
active_claims:
  - claim_id: synthetic_claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo, georgia]
      evaluation_views:
        - pn2021_all_zero_kept_refexcluded
        - pn2021_drop_all_zero_refexcluded
    methods:
      direct:
        metrics_long: {direct_metrics}
      vae:
        metrics_long: {vae_metrics}
    comparison_bundle:
      comparison_id: synthetic_comparison
      baseline_method: direct
      candidate_method: vae
      baseline_run_id: direct
      candidate_run_id: vae
      output_dir: ${{paths.output_root}}/synthetic_comparison
""",
        encoding="utf-8",
    )

    manifest = build_comparison_bundle(
        registry_path=registry,
        local_config_path=local_config,
        force=True,
    )

    assert Path(manifest["artifacts"]["comparison_delta"]).exists()
    mean = [
        row for row in manifest["delta_rows"]
        if row["view"] == "pn2021_all_zero_kept_refexcluded"
        and row["scope"] == "center_mean"
    ][0]
    assert abs(mean["delta_macro_auroc_pp"] - 3.0) < 1e-9
    assert abs(mean["delta_macro_auprc_pp"] - 6.0) < 1e-9
