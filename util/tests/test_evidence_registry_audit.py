from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import ecg_adv_gen.evidence.registry as registry_module
from ecg_adv_gen.evidence import audit_active_evidence_registry


MAPPING = ("v7_super5_sjr_rgq_review_20260528", "555ec85d5b51")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    output = tmp_path / "runs"
    run_dir = output / "managed" / "run-1"
    support = output / "support"
    for path in (repo, run_dir, support, support / "training", support / "evaluation"):
        path.mkdir(parents=True, exist_ok=True)

    (repo / "evidence.md").write_text("evidence\n", encoding="utf-8")
    (repo / "method.yaml").write_text("method: test\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "evidence.md", "method.yaml")
    _git(
        repo,
        "-c",
        "user.name=Registry Test",
        "-c",
        "user.email=registry@example.invalid",
        "commit",
        "-qm",
        "seed evidence",
    )
    commit = _git(repo, "rev-parse", "HEAD")

    card = {
        "run_id": "run-1",
        "experiment": {"name": "experiment-1"},
        "protocol": {"mapping_version": MAPPING[0], "mapping_hash": MAPPING[1]},
    }
    _write_json(run_dir / "run_card.json", card)
    _write_json(run_dir / "run_file_index.json", {})
    (run_dir / "summary.md").write_text("summary\n", encoding="utf-8")
    run_file_index_sha256 = hashlib.sha256((run_dir / "run_file_index.json").read_bytes()).hexdigest()

    for name, content in {
        "metrics.csv": "metric,value\nmacro_auroc,0.8\n",
        "table.csv": "metric,value\nmacro_auroc,0.8\n",
        "center.csv": "center,value\nningbo,0.8\n",
        "provenance.json": "{}\n",
        "payload.csv": "metric,value\nmacro_auprc,0.7\n",
        "normal.yaml": "kind: ordinary-paper-manifest\n",
    }.items():
        (support / name).write_text(content, encoding="utf-8")
    _write_json(support / "summary.json", ["ordinary", "summary"])
    _write_json(support / "paper_manifest.json", ["ordinary", "paper", "manifest"])
    payload_sha = hashlib.sha256((support / "payload.csv").read_bytes()).hexdigest()
    strict_manifest = {"artifacts": [{"path": "payload.csv", "sha256": payload_sha}]}
    _write_json(support / "artifact_manifest.json", strict_manifest)
    _write_json(support / "artifact_integrity.json", strict_manifest)
    _write_json(support / "nested_artifact_manifest.json", strict_manifest)
    ref_meta = support / "ningbo_ref.json"
    _write_json(
        ref_meta,
        {
            "center": "ningbo",
            "K": 1,
            "selection_seed": 1,
            "mapping_version": MAPPING[0],
            "mapping_hash": MAPPING[1],
            "ref_record_ids": ["r1"],
        },
    )
    ref_hash = hashlib.sha256(b"r1\n").hexdigest()
    k500_ids = support / "k500_ref_ids.json"
    _write_json(
        k500_ids,
        {
            "paper_protocol": {"seed": 1, "subset_seed": 1},
            "centers": {
                "ningbo": {
                    "center": "ningbo",
                    "consumers": ["evaluation"],
                    "k": 1,
                    "selection_seed": 1,
                    "ref_record_ids_sha256": ref_hash,
                }
            },
        },
    )
    method_manifest = support / "method_run_manifest.json"
    _write_json(
        method_manifest,
        {
            "status": "succeeded",
            "manifest_kind": "managed_launcher_manifest",
            "artifact_trace": {
                "expected_outputs": {
                    "launch_artifacts": [
                        {"role": "k500_ref_ids", "path": str(k500_ids), "required": True}
                    ]
                }
            },
        },
    )
    eval_result = support / "pn2021c_eval.json"
    _write_json(
        eval_result,
        {
            "per_center": {
                "ningbo": {
                    "emg_noise": {
                        "5": {
                            "metadata_compatibility": {
                                "n_excluded_ref": 1,
                                "ref_record_ids_sha256": ref_hash,
                            }
                        }
                    }
                }
            }
        },
    )
    eval_manifest = support / "pn2021c_artifact_manifest.json"
    _write_json(
        eval_manifest,
        {
            "artifacts": [
                {
                    "artifact_type": "eval_result",
                    "path": str(eval_result),
                    "sha256": hashlib.sha256(eval_result.read_bytes()).hexdigest(),
                }
            ]
        },
    )

    registry = {
        "managed_runs": [
            {
                "run_id": "run-1",
                "experiment_name": "experiment-1",
                "status": "trusted",
                "replay_status": "replay_ready",
                "run_dir": "${paths.output_root}/managed/run-1",
                "run_card": "${paths.output_root}/managed/run-1/run_card.json",
                "summary": "${paths.output_root}/managed/run-1/summary.md",
                "run_file_index": "${paths.output_root}/managed/run-1/run_file_index.json",
                "run_file_index_sha256": run_file_index_sha256,
                "mapping_version": MAPPING[0],
                "mapping_hash": MAPPING[1],
            }
        ],
        "active_claims": [
            {
                "claim_id": "claim-1",
                "status": "provisional",
                "paper_use": "supporting",
                "protocol": {
                    "mapping_version": MAPPING[0],
                    "mapping_hash": MAPPING[1],
                    "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                    "model_backbone": "fixture",
                    "target_centers": ["ningbo"],
                    "evaluation_views": ["pn2021_all_zero_kept_refexcluded"],
                    "kshot": {
                        "k": 1,
                        "seed": 1,
                        "ref_excluded": True,
                        "ref_meta_files": {"ningbo": str(ref_meta)},
                    },
                },
                "methods": {
                    "baseline": {
                        "run_id": "run-1",
                        "experiment_name": "experiment-1",
                        "status": "provisional",
                        "config": "method.yaml",
                        "manifest": str(method_manifest),
                    }
                },
                "required_reporting_artifacts": {
                    "method_artifact_manifests": {"baseline": str(eval_manifest)}
                },
                "supporting_reporting_artifacts": {
                    "support-1": {
                        "status": "provisional_single_seed",
                        "replay_status": "oneoff_not_managed_run_record",
                        "evidence_doc": "evidence.md",
                        "implementation_commit": commit,
                        "method_config": "method.yaml",
                        "training_root": "${paths.output_root}/support/training",
                        "evaluation_root": "${paths.output_root}/support/evaluation",
                        "metrics_long": "${paths.output_root}/support/metrics.csv",
                        "data_manifest": "${paths.output_root}/support/normal.yaml",
                        "artifact_manifest": "${paths.output_root}/support/artifact_manifest.json",
                        "method_artifact_manifests": {
                            "direct": "${paths.output_root}/support/nested_artifact_manifest.json"
                        },
                        "paper_tables": {"kept": "${paths.output_root}/support/table.csv"},
                        "paper_table_manifests": {
                            "kept": "${paths.output_root}/support/paper_manifest.json",
                            "yaml": "${paths.output_root}/support/normal.yaml",
                        },
                        "summary_csv": "${paths.output_root}/support/table.csv",
                        "center_csv": "${paths.output_root}/support/center.csv",
                        "summary_json": "${paths.output_root}/support/summary.json",
                        "provenance_gap": "${paths.output_root}/support/provenance.json",
                        "artifact_integrity": "${paths.output_root}/support/artifact_integrity.json",
                        "artifact_integrity_sha256": hashlib.sha256(
                            (support / "artifact_integrity.json").read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ],
    }
    registry_path = repo / "registry.yaml"
    local_config = tmp_path / "local.yaml"
    local_config.write_text(
        yaml.safe_dump({"paths": {"output_root": str(output), "data_root": str(tmp_path / "data")}}),
        encoding="utf-8",
    )

    verify_calls: list[Path] = []

    def verified(path: Path) -> dict[str, object]:
        verify_calls.append(path)
        return {"passed": True, "status": "verified", "checked_count": 3, "warnings": [], "errors": []}

    monkeypatch.setattr(registry_module, "verify_run_file_index", verified, raising=False)
    monkeypatch.setattr(registry_module, "_audit_config", lambda *args, **kwargs: None)

    def audit(*, check_files: bool = True, check_git: bool = True) -> dict[str, object]:
        registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
        return audit_active_evidence_registry(
            repo_root=repo,
            registry_path=registry_path,
            local_config_path=local_config,
            require_existing_artifacts=check_files,
            check_git=check_git,
        )

    return {
        "audit": audit,
        "registry": registry,
        "repo": repo,
        "run_dir": run_dir,
        "support": support,
        "k500_ids": k500_ids,
        "eval_result": eval_result,
        "eval_manifest": eval_manifest,
        "verify_calls": verify_calls,
    }


def _codes(report: dict[str, object]) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def _support(case: dict[str, object]) -> dict[str, object]:
    return case["registry"]["active_claims"][0]["supporting_reporting_artifacts"]["support-1"]


def _rewrite_integrity(case: dict[str, object], value: object) -> None:
    path = case["support"] / "artifact_integrity.json"
    _write_json(path, value)
    _support(case)["artifact_integrity_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_eval_result(case: dict[str, object], value: object) -> None:
    path = case["eval_result"]
    _write_json(path, value)
    manifest_path = case["eval_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)


def test_valid_registry_audits_explicit_paths_and_integrity(evidence_case: dict[str, object]) -> None:
    report = evidence_case["audit"]()

    assert report["passed"] is True
    assert report["managed_runs"]["valid_count"] == 1
    assert report["managed_runs"]["entries"][0]["integrity"]["status"] == "verified"
    assert evidence_case["verify_calls"] == [evidence_case["run_dir"]]
    support = report["claims"][0]["supporting_reporting_artifacts"]
    assert support["valid_count"] == 1
    assert support["entries"][0]["integrity"] == {"checked": 3, "missing": 0, "mismatch": 0}
    assert report["artifact_hashing"] == {"computed": 3, "cache_hits": 2}
    assert not _codes(report) & {"supporting_manifest_invalid", "supporting_file_missing"}


@pytest.mark.parametrize("value", [("", ""), ("bogus", "bad")])
def test_managed_run_mapping_must_match_active_claim(
    evidence_case: dict[str, object], value: tuple[str, str]
) -> None:
    item = evidence_case["registry"]["managed_runs"][0]
    item["mapping_version"], item["mapping_hash"] = value
    card = json.loads((evidence_case["run_dir"] / "run_card.json").read_text(encoding="utf-8"))
    card["protocol"]["mapping_version"], card["protocol"]["mapping_hash"] = value
    _write_json(evidence_case["run_dir"] / "run_card.json", card)

    assert "managed_run_claim_mapping_mismatch" in _codes(evidence_case["audit"]())


def test_managed_run_rejects_duplicate_and_conflicting_claim_mapping(evidence_case: dict[str, object]) -> None:
    registry = evidence_case["registry"]
    registry["managed_runs"].append(deepcopy(registry["managed_runs"][0]))
    conflict = deepcopy(registry["active_claims"][0])
    conflict["claim_id"] = "claim-2"
    conflict["protocol"]["mapping_hash"] = "conflict"
    conflict["supporting_reporting_artifacts"] = {}
    registry["active_claims"].append(conflict)

    codes = _codes(evidence_case["audit"]())
    assert "managed_run_duplicate_key" in codes
    assert "managed_run_claim_mapping_conflict" in codes


def test_managed_files_must_use_conventional_paths_inside_run_dir(evidence_case: dict[str, object]) -> None:
    outside = evidence_case["repo"] / "run_card.json"
    outside.write_bytes((evidence_case["run_dir"] / "run_card.json").read_bytes())
    evidence_case["registry"]["managed_runs"][0]["run_card"] = str(outside)

    assert "managed_run_path_mismatch" in _codes(evidence_case["audit"]())


def test_managed_run_index_root_sha_is_required_and_verified(evidence_case: dict[str, object]) -> None:
    item = evidence_case["registry"]["managed_runs"][0]
    item.pop("run_file_index_sha256")
    assert "managed_run_index_sha_invalid" in _codes(evidence_case["audit"]())

    item["run_file_index_sha256"] = "0" * 64
    report = evidence_case["audit"]()
    assert "managed_run_index_sha_mismatch" in _codes(report)
    assert report["managed_runs"]["entries"][0]["replay_ready"] is False


def test_managed_run_is_bound_to_claim_experiment(evidence_case: dict[str, object]) -> None:
    item = evidence_case["registry"]["managed_runs"][0]
    item["experiment_name"] = "forged-experiment"
    card = json.loads((evidence_case["run_dir"] / "run_card.json").read_text(encoding="utf-8"))
    card["experiment"]["name"] = "forged-experiment"
    _write_json(evidence_case["run_dir"] / "run_card.json", card)

    assert "managed_run_claim_mapping_mismatch" in _codes(evidence_case["audit"]())


def test_skip_flags_keep_structure_but_skip_all_io_and_git(evidence_case: dict[str, object]) -> None:
    item = _support(evidence_case)
    item["evidence_doc"] = "missing.md"
    item["implementation_commit"] = "0" * 40
    item["method_config"] = "missing.yaml"
    item["training_root"] = "${paths.output_root}/missing-training"
    item["artifact_integrity"] = "${paths.output_root}/missing-integrity.json"

    report = evidence_case["audit"](check_files=False, check_git=False)

    assert report["passed"] is True
    assert not _codes(report) & {
        "supporting_file_missing",
        "supporting_directory_missing",
        "supporting_evidence_doc_missing",
        "supporting_implementation_commit_unresolvable",
    }
    assert evidence_case["verify_calls"] == []
    assert report["artifact_hashing"] == {"computed": 0, "cache_hits": 0}


def test_skip_files_still_rejects_non_path_directory_declaration(evidence_case: dict[str, object]) -> None:
    _support(evidence_case)["training_root"] = ["not", "a", "path"]

    assert "supporting_path_invalid" in _codes(
        evidence_case["audit"](check_files=False, check_git=False)
    )


@pytest.mark.parametrize(
    ("field", "code"),
    [("method_config", "supporting_file_missing"), ("data_manifest", "supporting_file_missing"),
     ("training_root", "supporting_directory_missing")],
)
def test_explicit_supporting_config_and_roots_are_checked(
    evidence_case: dict[str, object], field: str, code: str
) -> None:
    item = _support(evidence_case)
    item[field] = "${paths.output_root}/does-not-exist"

    assert code in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("manifest", "code"),
    [({}, "supporting_manifest_artifacts_invalid"), ({"artifacts": []}, "supporting_manifest_artifacts_invalid"),
     ({"artifacts": "bad"}, "supporting_manifest_artifacts_invalid")],
)
def test_integrity_manifest_requires_nonempty_artifacts(
    evidence_case: dict[str, object], manifest: object, code: str
) -> None:
    _rewrite_integrity(evidence_case, manifest)

    assert code in _codes(evidence_case["audit"]())


def test_artifact_manifest_also_requires_nonempty_artifacts(evidence_case: dict[str, object]) -> None:
    _write_json(evidence_case["support"] / "artifact_manifest.json", {})

    assert "supporting_manifest_artifacts_invalid" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize("declared", ["0" * 64, "not-a-sha"])
def test_integrity_manifest_root_sha_is_verified(evidence_case: dict[str, object], declared: str) -> None:
    _support(evidence_case)["artifact_integrity_sha256"] = declared

    assert "supporting_integrity_root_sha_mismatch" in _codes(evidence_case["audit"]())


def test_integrity_payload_missing_and_mutation_are_rejected(evidence_case: dict[str, object]) -> None:
    (evidence_case["support"] / "payload.csv").unlink()
    assert "supporting_manifest_artifact_missing" in _codes(evidence_case["audit"]())

    (evidence_case["support"] / "payload.csv").write_text("mutated\n", encoding="utf-8")
    assert "supporting_manifest_sha256_mismatch" in _codes(evidence_case["audit"]())


def test_manifest_path_ref_falls_back_to_resolved_local_context(evidence_case: dict[str, object]) -> None:
    payload_sha = hashlib.sha256((evidence_case["support"] / "payload.csv").read_bytes()).hexdigest()
    entry = {
        "path": "/missing/old-host/payload.csv",
        "path_ref": "${paths.output_root}/support/payload.csv",
        "sha256": payload_sha,
    }
    _rewrite_integrity(evidence_case, {"artifacts": [entry]})
    _write_json(evidence_case["support"] / "artifact_manifest.json", {"artifacts": [entry]})

    assert evidence_case["audit"]()["passed"] is True


def test_manifest_path_ref_with_unknown_placeholder_is_structural_error(
    evidence_case: dict[str, object],
) -> None:
    payload_sha = hashlib.sha256((evidence_case["support"] / "payload.csv").read_bytes()).hexdigest()
    _rewrite_integrity(
        evidence_case,
        {"artifacts": [{"path_ref": "${paths.unknown}/payload.csv", "sha256": payload_sha}]},
    )

    assert "supporting_manifest_path_ref_unresolvable" in _codes(evidence_case["audit"]())


def test_duplicate_artifact_paths_are_hashed_once(
    evidence_case: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real_sha = registry_module._sha256

    def counting_sha(path: Path) -> str:
        calls.append(path)
        return real_sha(path)

    monkeypatch.setattr(registry_module, "_sha256", counting_sha)
    report = evidence_case["audit"]()

    assert calls.count(evidence_case["support"] / "payload.csv") == 1
    assert report["artifact_hashing"]["cache_hits"] == 2


def test_nested_method_artifact_manifest_detects_payload_mutation(evidence_case: dict[str, object]) -> None:
    payload = evidence_case["support"] / "payload.csv"
    payload.write_text("mutated\n", encoding="utf-8")
    new_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {"artifacts": [{"path": "payload.csv", "sha256": new_sha}]}
    _write_json(evidence_case["support"] / "artifact_manifest.json", manifest)
    _rewrite_integrity(evidence_case, manifest)

    assert "supporting_manifest_sha256_mismatch" in _codes(evidence_case["audit"]())


def test_deprecated_non_ready_run_need_not_be_in_active_claim(evidence_case: dict[str, object]) -> None:
    item = evidence_case["registry"]["managed_runs"][0]
    item["status"] = "deprecated"
    item["replay_status"] = "artifact_missing"
    evidence_case["registry"]["active_claims"] = []

    report = evidence_case["audit"]()
    assert report["passed"] is True
    assert report["managed_runs"]["valid_count"] == 1
    assert report["managed_runs"]["entries"][0]["replay_ready"] is False


@pytest.mark.parametrize(
    ("status", "replay_status"),
    [("trusted", "artifact_missing"), ("provisional", "historical_evaluation_only"),
     ("failed", "replay_ready"), ("deprecated", "replay_ready")],
)
def test_managed_run_rejects_inconsistent_status_and_replay_state(
    evidence_case: dict[str, object], status: str, replay_status: str
) -> None:
    item = evidence_case["registry"]["managed_runs"][0]
    item["status"] = status
    item["replay_status"] = replay_status

    assert "managed_run_replay_status_invalid" in _codes(evidence_case["audit"]())


def test_managed_run_rejects_unknown_status(evidence_case: dict[str, object]) -> None:
    evidence_case["registry"]["managed_runs"][0]["status"] = "unverified"

    assert "managed_run_status_invalid" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("field", "bad_value", "code"),
    [
        ("claim_id", "", "active_claim_id_missing"),
        ("methods", [], "active_claim_methods_invalid"),
        ("protocol", [], "active_claim_protocol_invalid"),
    ],
)
def test_malformed_claim_fields_return_audit_errors(
    evidence_case: dict[str, object], field: str, bad_value: object, code: str
) -> None:
    evidence_case["registry"]["active_claims"][0][field] = bad_value

    assert code in _codes(evidence_case["audit"]())


def test_malformed_nested_claim_fields_do_not_crash(evidence_case: dict[str, object]) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["methods"]["baseline"] = "bad"
    assert "active_claim_method_invalid" in _codes(evidence_case["audit"]())

    claim["methods"] = {}
    claim["protocol"] = {}
    assert "active_claim_protocol_invalid" in _codes(evidence_case["audit"]())


def test_active_claim_requires_nonempty_review_protocol(evidence_case: dict[str, object]) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["methods"] = {}
    claim["protocol"]["class_order"] = []
    claim["protocol"]["target_centers"] = []
    claim["protocol"]["evaluation_views"] = []
    claim["protocol"]["kshot"] = {"ref_meta_files": {}}

    codes = _codes(evidence_case["audit"]())
    assert "active_claim_methods_invalid" in codes
    assert "active_claim_protocol_invalid" in codes


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_centers", [{}]), ("evaluation_views", [{}]), ("class_order", [""])],
)
def test_active_claim_protocol_list_items_must_be_nonempty_strings(
    evidence_case: dict[str, object], field: str, value: list[object]
) -> None:
    evidence_case["registry"]["active_claims"][0]["protocol"][field] = value

    assert "active_claim_protocol_invalid" in _codes(evidence_case["audit"]())


def test_active_claim_requires_ref_meta_for_every_center(evidence_case: dict[str, object]) -> None:
    evidence_case["registry"]["active_claims"][0]["protocol"]["kshot"]["ref_meta_files"] = {}

    assert "active_claim_protocol_invalid" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("surface", "code"),
    [("paper_tables", "active_claim_method_field_invalid"),
     ("comparison_bundle", "active_claim_comparison_bundle_invalid")],
)
def test_nested_claim_maps_must_be_mappings(
    evidence_case: dict[str, object], surface: str, code: str
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    if surface == "paper_tables":
        claim["methods"]["baseline"][surface] = []
    else:
        claim[surface] = ["bad"]

    assert code in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    "payload",
    [
        ["bad-root"],
        {"center": "ningbo", "K": "bogus", "selection_seed": 1,
         "mapping_version": MAPPING[0], "mapping_hash": MAPPING[1], "ref_record_ids": ["r1"]},
        {"center": "ningbo", "K": 1, "selection_seed": 1,
         "mapping_version": MAPPING[0], "mapping_hash": MAPPING[1], "ref_record_ids": 1},
    ],
)
def test_malformed_k500_ref_json_returns_error_not_exception(
    evidence_case: dict[str, object], payload: object
) -> None:
    ref_path = Path(
        evidence_case["registry"]["active_claims"][0]["protocol"]["kshot"]["ref_meta_files"]["ningbo"]
    )
    _write_json(ref_path, payload)

    assert "k500_ref_invalid" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("k", "record_ids"),
    [(1, [""]), (2, ["r1", "r1"])],
)
def test_k500_ref_ids_must_be_nonempty_and_unique(
    evidence_case: dict[str, object], k: int, record_ids: list[str]
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["protocol"]["kshot"]["k"] = k
    ref_path = Path(claim["protocol"]["kshot"]["ref_meta_files"]["ningbo"])
    _write_json(
        ref_path,
        {
            "center": "ningbo", "K": k, "selection_seed": 1,
            "mapping_version": MAPPING[0], "mapping_hash": MAPPING[1],
            "ref_record_ids": record_ids,
        },
    )

    assert "k500_ref_invalid" in _codes(evidence_case["audit"]())


def test_active_claim_audits_recorded_evaluation_k500_identity(
    evidence_case: dict[str, object],
) -> None:
    support = evidence_case["support"]
    expected_hash = hashlib.sha256(b"r1\n").hexdigest()
    k500_ids = support / "k500_ref_ids.json"
    _write_json(
        k500_ids,
        {
            "paper_protocol": {"seed": 1, "subset_seed": 1},
            "centers": {
                "ningbo": {
                    "k": 1,
                    "selection_seed": 1,
                    "ref_record_ids_sha256": expected_hash,
                }
            },
        },
    )
    method_manifest = support / "method_run_manifest.json"
    _write_json(
        method_manifest,
        {
            "status": "succeeded",
            "manifest_kind": "managed_launcher_manifest",
            "artifact_trace": {
                "expected_outputs": {
                    "launch_artifacts": [
                        {"role": "k500_ref_ids", "path": str(k500_ids), "required": True}
                    ]
                }
            },
        },
    )
    eval_result = support / "pn2021c_eval.json"
    _write_json(
        eval_result,
        {
            "per_center": {
                "ningbo": {
                    "emg_noise": {
                        "5": {
                            "metadata_compatibility": {
                                "n_excluded_ref": 1,
                                "ref_record_ids_sha256": "b" * 64,
                            }
                        }
                    }
                }
            }
        },
    )
    eval_manifest = support / "pn2021c_artifact_manifest.json"
    _write_json(
        eval_manifest,
        {
            "artifacts": [
                {
                    "artifact_type": "eval_result",
                    "path": str(eval_result),
                    "sha256": hashlib.sha256(eval_result.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    claim = evidence_case["registry"]["active_claims"][0]
    claim["methods"]["baseline"]["manifest"] = str(method_manifest)
    claim["required_reporting_artifacts"] = {
        "method_artifact_manifests": {"baseline": str(eval_manifest)}
    }

    report = evidence_case["audit"]()

    assert "evaluation_k500_identity_mismatch" in _codes(report)


def test_active_method_requires_evaluation_artifact_manifest(
    evidence_case: dict[str, object],
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["required_reporting_artifacts"] = {"method_artifact_manifests": {}}

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


def test_active_method_rejects_non_mapping_evaluation_artifact_manifests(
    evidence_case: dict[str, object],
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["required_reporting_artifacts"] = ["bad"]

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


def test_active_method_rejects_empty_training_k500_identities(
    evidence_case: dict[str, object],
) -> None:
    _write_json(evidence_case["k500_ids"], {"centers": {}})

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


def test_active_method_binds_training_identity_to_claim_ref_ids(
    evidence_case: dict[str, object],
) -> None:
    wrong_hash = "b" * 64
    k500 = json.loads(evidence_case["k500_ids"].read_text(encoding="utf-8"))
    k500["centers"]["ningbo"]["ref_record_ids_sha256"] = wrong_hash
    _write_json(evidence_case["k500_ids"], k500)
    _rewrite_eval_result(
        evidence_case,
        {
            "per_center": {
                "ningbo": {
                    "emg_noise": {
                        "5": {
                            "metadata_compatibility": {
                                "n_excluded_ref": 1,
                                "ref_record_ids_sha256": wrong_hash,
                            }
                        }
                    }
                }
            }
        },
    )

    assert "evaluation_k500_identity_mismatch" in _codes(evidence_case["audit"]())


def test_active_method_binds_training_seed_to_claim(
    evidence_case: dict[str, object],
) -> None:
    k500 = json.loads(evidence_case["k500_ids"].read_text(encoding="utf-8"))
    k500["centers"]["ningbo"]["selection_seed"] = 2
    _write_json(evidence_case["k500_ids"], k500)

    assert "evaluation_k500_identity_mismatch" in _codes(evidence_case["audit"]())


def test_active_method_rejects_unexpected_evaluation_center(
    evidence_case: dict[str, object],
) -> None:
    ref_hash = hashlib.sha256(b"r1\n").hexdigest()
    row = {
        "emg_noise": {
            "5": {
                "metadata_compatibility": {
                    "n_excluded_ref": 1,
                    "ref_record_ids_sha256": ref_hash,
                }
            }
        }
    }
    _rewrite_eval_result(
        evidence_case,
        {"per_center": {"ningbo": row, "georgia": row}},
    )

    assert "evaluation_k500_identity_mismatch" in _codes(evidence_case["audit"]())


def test_active_method_requires_recorded_evaluation_artifacts(
    evidence_case: dict[str, object],
) -> None:
    _write_json(evidence_case["eval_manifest"], {"artifacts": []})

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize("declared_sha", [None, "0" * 64])
def test_active_method_validates_eval_artifact_sha_before_read(
    evidence_case: dict[str, object], declared_sha: str | None,
) -> None:
    manifest = json.loads(evidence_case["eval_manifest"].read_text(encoding="utf-8"))
    if declared_sha is None:
        manifest["artifacts"][0].pop("sha256")
    else:
        manifest["artifacts"][0]["sha256"] = declared_sha
    _write_json(evidence_case["eval_manifest"], manifest)

    assert "evaluation_artifact_sha256_mismatch" in _codes(evidence_case["audit"]())


def test_active_method_malformed_evaluation_count_is_fail_closed(
    evidence_case: dict[str, object],
) -> None:
    _rewrite_eval_result(
        evidence_case,
        {
            "per_center": {
                "ningbo": {
                    "emg_noise": {
                        "5": {
                            "metadata_compatibility": {
                                "n_excluded_ref": "not-an-int",
                                "ref_record_ids_sha256": hashlib.sha256(b"r1\n").hexdigest(),
                            }
                        }
                    }
                }
            }
        },
    )

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


def test_trusted_claim_supporting_method_cannot_skip_lineage_gate(
    evidence_case: dict[str, object],
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["status"] = "trusted"
    method = claim["methods"]["baseline"]
    method["status"] = "supporting"
    method.pop("manifest")
    claim.pop("required_reporting_artifacts")

    assert "evaluation_k500_identity_missing" in _codes(evidence_case["audit"]())


def test_active_method_prefers_subset_seed_over_seed(
    evidence_case: dict[str, object],
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["protocol"]["kshot"]["subset_seed"] = 2
    ref_path = Path(claim["protocol"]["kshot"]["ref_meta_files"]["ningbo"])
    ref_meta = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_meta["selection_seed"] = 2
    _write_json(ref_path, ref_meta)
    k500 = json.loads(evidence_case["k500_ids"].read_text(encoding="utf-8"))
    k500["paper_protocol"]["subset_seed"] = 2
    k500["centers"]["ningbo"]["selection_seed"] = 2
    _write_json(evidence_case["k500_ids"], k500)

    codes = _codes(evidence_case["audit"]())
    assert "k500_ref_protocol_mismatch" not in codes
    assert "evaluation_k500_identity_mismatch" not in codes


def test_trusted_method_keeps_managed_run_mapping_when_comparison_is_deprecated(
    evidence_case: dict[str, object],
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    claim["status"] = "deprecated"
    claim["paper_use"] = "prohibited_protocol_invalid"
    claim["methods"]["baseline"]["status"] = "trusted"

    claim.pop("required_reporting_artifacts")
    codes = _codes(evidence_case["audit"]())
    assert "managed_run_claim_mapping_mismatch" not in codes
    assert not any(code.startswith("evaluation_k500_") for code in codes)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("required_reporting_views", "active_claim_protocol_invalid"),
        ("managed_artifacts", "active_claim_comparison_bundle_invalid"),
        ("method_key", "active_claim_method_key_invalid"),
    ],
)
def test_additional_nested_schema_errors_do_not_escape(
    evidence_case: dict[str, object], mutation: str, code: str
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    if mutation == "required_reporting_views":
        claim["protocol"]["required_reporting_views"] = 1
    elif mutation == "managed_artifacts":
        claim["comparison_bundle"] = {"managed_artifacts": 1}
    else:
        claim["methods"][1] = deepcopy(claim["methods"]["baseline"])

    assert code in _codes(evidence_case["audit"]())


def test_comparison_manifest_root_must_be_object(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    _write_json(output / "comparison_manifest.json", ["bad-root"])
    issues: list[dict[str, object]] = []

    registry_module._audit_comparison_bundle(
        {
            "claim_id": "claim",
            "comparison_bundle": {
                "status": "built",
                "output_dir": str(output),
                "managed_artifacts": [],
            },
        },
        issues,
    )

    assert "comparison_manifest_invalid" in {issue["code"] for issue in issues}


@pytest.mark.parametrize(
    ("surface", "code"),
    [
        ("claim_status", "active_claim_status_invalid"),
        ("support_status", "supporting_status_invalid"),
        ("bundle_status", "active_claim_comparison_bundle_invalid"),
        ("summary_metrics", "active_claim_summary_metrics_invalid"),
    ],
)
def test_unhashable_nested_status_and_summary_values_return_errors(
    evidence_case: dict[str, object], surface: str, code: str
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    if surface == "claim_status":
        claim["status"] = []
    elif surface == "support_status":
        _support(evidence_case)["status"] = []
    elif surface == "bundle_status":
        claim["comparison_bundle"] = {"status": []}
    else:
        claim["summary_metrics"] = []

    assert code in _codes(evidence_case["audit"]())


@pytest.mark.parametrize("value", [[], {"do_not_commit": 1}])
def test_artifact_policy_must_be_mapping(evidence_case: dict[str, object], value: object) -> None:
    evidence_case["registry"]["artifact_policy"] = value

    assert "artifact_policy_invalid" in _codes(evidence_case["audit"]())


def test_method_manifest_root_must_be_object(evidence_case: dict[str, object]) -> None:
    manifest = evidence_case["repo"] / "method_manifest.json"
    _write_json(manifest, ["bad-root"])
    evidence_case["registry"]["active_claims"][0]["methods"]["baseline"]["manifest"] = str(manifest)

    assert "manifest_invalid" in _codes(evidence_case["audit"]())


def test_method_manifest_status_must_be_string(evidence_case: dict[str, object]) -> None:
    manifest = evidence_case["repo"] / "method_manifest.json"
    _write_json(manifest, {"status": []})
    evidence_case["registry"]["active_claims"][0]["methods"]["baseline"]["manifest"] = str(manifest)

    assert "manifest_status_invalid" in _codes(evidence_case["audit"]())


def test_metrics_with_missing_columns_returns_error_not_exception(evidence_case: dict[str, object]) -> None:
    metrics = evidence_case["support"] / "broken_metrics.csv"
    metrics.write_text(f"mapping_version\n{MAPPING[0]}\n", encoding="utf-8")
    evidence_case["registry"]["active_claims"][0]["methods"]["baseline"]["metrics_long"] = str(metrics)

    assert "metrics_long_bad_columns" in _codes(evidence_case["audit"]())


def test_metrics_with_truncated_row_returns_error_not_exception(evidence_case: dict[str, object]) -> None:
    metrics = evidence_case["support"] / "truncated_metrics.csv"
    metrics.write_text(
        ",".join(registry_module.METRICS_FIELDNAMES)
        + "\n"
        + ",".join([""] * 18 + [MAPPING[0]])
        + "\n",
        encoding="utf-8",
    )
    evidence_case["registry"]["active_claims"][0]["methods"]["baseline"]["metrics_long"] = str(metrics)

    assert "metrics_long_bad_values" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("surface", "code"),
    [
        ("metrics", "metrics_long_invalid"),
        ("paper_table", "paper_table_invalid"),
        ("k500_ref", "k500_ref_bad_json"),
        ("comparison_manifest", "comparison_manifest_bad_json"),
        ("artifact_integrity", "supporting_manifest_invalid"),
        ("run_card", "managed_run_card_invalid"),
    ],
)
def test_non_utf8_artifacts_return_errors(
    evidence_case: dict[str, object], surface: str, code: str
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    method = claim["methods"]["baseline"]
    if surface == "metrics":
        path = evidence_case["support"] / "bad_metrics.csv"
        method["metrics_long"] = str(path)
    elif surface == "paper_table":
        path = evidence_case["support"] / "bad_table.csv"
        method["paper_tables"] = {"pn2021_all_zero_kept_refexcluded": str(path)}
    elif surface == "k500_ref":
        path = Path(claim["protocol"]["kshot"]["ref_meta_files"]["ningbo"])
    elif surface == "comparison_manifest":
        output = evidence_case["support"] / "comparison"
        output.mkdir()
        path = output / "comparison_manifest.json"
        claim["comparison_bundle"] = {
            "status": "built", "output_dir": str(output), "managed_artifacts": []
        }
    elif surface == "artifact_integrity":
        path = evidence_case["support"] / "artifact_integrity.json"
    else:
        path = evidence_case["run_dir"] / "run_card.json"

    path.write_bytes(b"\xff")
    if surface == "artifact_integrity":
        _support(evidence_case)["artifact_integrity_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert code in _codes(evidence_case["audit"]())


def test_claim_id_must_be_nonempty_string(evidence_case: dict[str, object]) -> None:
    evidence_case["registry"]["active_claims"][0]["claim_id"] = {}

    assert "active_claim_id_missing" in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("surface", "code"),
    [("claim", "active_claim_status_invalid"), ("method", "active_claim_method_status_invalid")],
)
def test_claim_and_method_status_are_validated(
    evidence_case: dict[str, object], surface: str, code: str
) -> None:
    claim = evidence_case["registry"]["active_claims"][0]
    if surface == "claim":
        claim["status"] = "trustd"
    else:
        claim["methods"]["baseline"]["status"] = "trustd"

    assert code in _codes(evidence_case["audit"]())


@pytest.mark.parametrize(
    ("surface", "bad_value", "code"),
    [
        ("managed_runs", "bad", "managed_runs_invalid"),
        ("managed_runs", {}, "managed_runs_invalid"),
        ("managed_runs", ["bad"], "managed_run_invalid"),
        ("active_claims", "bad", "active_claims_invalid"),
        ("active_claims", {}, "active_claims_invalid"),
        ("active_claims", ["bad"], "active_claim_invalid"),
        ("supporting_reporting_artifacts", [], "supporting_artifacts_invalid"),
        ("supporting_reporting_artifacts", "bad", "supporting_artifacts_invalid"),
    ],
)
def test_malformed_registry_shapes_return_audit_errors(
    evidence_case: dict[str, object], surface: str, bad_value: object, code: str
) -> None:
    if surface in {"managed_runs", "active_claims"}:
        evidence_case["registry"][surface] = bad_value
    else:
        evidence_case["registry"]["active_claims"][0][surface] = bad_value

    assert code in _codes(evidence_case["audit"]())


def test_managed_run_integrity_failure_is_reported(
    evidence_case: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        registry_module,
        "verify_run_file_index",
        lambda _: {"passed": False, "status": "failed", "checked_count": 1,
                   "warnings": [], "errors": ["mismatch"]},
    )

    report = evidence_case["audit"]()
    assert "managed_run_integrity_failed" in _codes(report)
    assert report["managed_runs"]["entries"][0]["integrity"]["status"] == "failed"


def test_declared_historical_run_can_be_non_ready_without_error(
    evidence_case: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_case["registry"]["managed_runs"][0]["status"] = "deprecated"
    evidence_case["registry"]["managed_runs"][0]["replay_status"] = "historical_evaluation_only"
    monkeypatch.setattr(
        registry_module,
        "verify_run_file_index",
        lambda _: {"passed": False, "status": "unchecked", "checked_count": 0,
                   "warnings": ["legacy schema"], "errors": []},
    )

    report = evidence_case["audit"]()
    assert report["passed"] is True
    assert report["managed_runs"]["entries"][0]["replay_ready"] is False
    assert report["managed_runs"]["entries"][0]["integrity"]["status"] in {"skipped", "unchecked"}


def test_provisional_support_cannot_bypass_integrity_with_managed_prefix(
    evidence_case: dict[str, object]
) -> None:
    support = _support(evidence_case)
    support["replay_status"] = "managed_bogus_prefix"
    for field in ("provenance_gap", "artifact_integrity", "artifact_integrity_sha256"):
        support.pop(field, None)

    codes = _codes(evidence_case["audit"]())
    assert "supporting_provenance_gap_not_declared" in codes
    assert "supporting_artifact_integrity_not_declared" in codes


def test_trusted_support_requires_strict_integrity(evidence_case: dict[str, object]) -> None:
    support = _support(evidence_case)
    support["status"] = "trusted"
    for field in ("provenance_gap", "artifact_integrity", "artifact_integrity_sha256"):
        support.pop(field, None)

    assert "supporting_artifact_integrity_not_declared" in _codes(evidence_case["audit"]())


def test_supporting_status_typo_cannot_bypass_integrity(evidence_case: dict[str, object]) -> None:
    support = _support(evidence_case)
    support["status"] = "provisional_single_seeed"
    for field in ("provenance_gap", "artifact_integrity", "artifact_integrity_sha256"):
        support.pop(field, None)

    assert "supporting_status_invalid" in _codes(evidence_case["audit"]())


def test_implementation_commit_must_be_full_sha_even_when_git_check_is_skipped(
    evidence_case: dict[str, object]
) -> None:
    _support(evidence_case)["implementation_commit"] = "HEAD"

    assert "supporting_implementation_commit_invalid" in _codes(
        evidence_case["audit"](check_files=False, check_git=False)
    )


def test_method_experiment_name_is_bound_to_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "method.yaml").write_text("experiment: {}\n", encoding="utf-8")
    config = {
        "experiment": {"name": "resolved-experiment"},
        "paper_protocol": {
            "mapping_version": MAPPING[0], "mapping_hash": MAPPING[1],
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "centers": {"target_4": ["ningbo"]},
            "kshot": {"k": 1, "seed": 1, "exclude_refs_from_eval": True},
        },
        "model": {"backbone": "fixture"},
    }
    monkeypatch.setattr(registry_module, "load_experiment_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(registry_module, "validate_experiment_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry_module, "build_runner_commands", lambda _: [])
    monkeypatch.setattr(registry_module, "build_postprocess_commands", lambda _: [])
    claim = {
        "protocol": {
            "mapping_version": MAPPING[0], "mapping_hash": MAPPING[1],
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "target_centers": ["ningbo"], "kshot": {"k": 1, "seed": 1, "ref_excluded": True},
            "model_backbone": "fixture",
        }
    }
    issues: list[dict[str, object]] = []

    registry_module._audit_config(
        repo, tmp_path / "local.yaml", claim, "baseline",
        {"config": "method.yaml", "run_id": "run-1", "experiment_name": "forged-experiment"},
        issues,
    )

    assert "config_experiment_mismatch" in {issue["code"] for issue in issues}
