"""Tests for agent-readable run finalization and registration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ecg_adv_gen.evidence import RunRecordError, finalize_run_record, register_run_in_registry
from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES


def _write_metrics_long(path: Path, *, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for center, auroc, auprc in [("ningbo", 0.81, 0.52), ("georgia", 0.85, 0.56)]:
        source_file = path.parent / f"{center}_eval_result.json"
        source_file.write_text(json.dumps({"center": center, "macro_auroc": auroc, "macro_auprc": auprc}), encoding="utf-8")
        for metric, value in [("macro_auroc", auroc), ("macro_auprc", auprc)]:
            row = {key: "" for key in METRICS_FIELDNAMES}
            row.update(
                {
                    "run_id": run_id,
                    "source_file": str(source_file),
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
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_k500_ref_ids(path: Path, *, tmp_path: Path) -> None:
    refs_root = tmp_path / "refs"
    centers = {}
    for center in ["ningbo", "georgia"]:
        ids = [f"{center}_{idx:03d}" for idx in range(500)]
        meta_path = refs_root / f"{center}.ref_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps({"K": 500, "selection_seed": 7, "ref_record_ids": ids}, indent=2),
            encoding="utf-8",
        )
        centers[center] = {
            "center": center,
            "k": 500,
            "selection_seed": 7,
            "selection_policy": "k500_internal_val_plus_source_floor",
            "ref_record_ids_ordered": ids,
            "ref_record_ids_sorted": sorted(ids),
            "ref_record_ids_sha256": _sha256_lines(sorted(ids)),
            "source_ref_meta_path": str(meta_path),
            "source_ref_meta_sha256": _sha256_file(meta_path),
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paper_protocol": {
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                    "target_centers": ["ningbo", "georgia"],
                    "k": 500,
                    "seed": 7,
                    "subset_seed": 7,
                    "exclude_refs_from_eval": True,
                },
                "centers": centers,
                "ref_record_id_hashes": {
                    center: payload["ref_record_ids_sha256"]
                    for center, payload in centers.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_paper_table(path: Path, *, metrics_long: Path) -> None:
    source_files = sorted(
        {
            row["source_file"]
            for row in csv.DictReader(metrics_long.open("r", encoding="utf-8", newline=""))
            if row.get("source_file")
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "dataset", "view", "metric", "value", "source_files"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "unit_experiment",
                "dataset": "pn2021",
                "view": "pn2021_all_zero_kept_refexcluded",
                "metric": "macro_auroc",
                "value": "0.83",
                "source_files": "|".join(source_files),
            }
        )
    (path.parent / "paper_table_manifest.json").write_text(
        json.dumps({"metrics_long": str(metrics_long), "paper_table": str(path)}, indent=2),
        encoding="utf-8",
    )


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "20260529" / "unit_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.resolved.yaml").write_text("experiment:\n  name: unit_experiment\n", encoding="utf-8")
    (run_dir / "run_config.resolved.json").write_text(
        json.dumps({"experiment": {"name": "unit_experiment"}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\npython train.py\n", encoding="utf-8")
    (run_dir / "data_manifest.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    _write_k500_ref_ids(run_dir / "k500_ref_ids.json", tmp_path=tmp_path)
    (run_dir / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_policy": {"policy": "k500_internal_val_plus_source_floor"},
                "selection_safety": {
                    "heldout_target_labels_used_for_selection": False,
                    "full_target_distribution_used_for_tuning": False,
                    "forbidden_reference_found": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "command_00.log").write_text("training complete\n", encoding="utf-8")
    diagnostics = run_dir / "diagnostics_epoch.jsonl"
    diagnostics.write_text('{"epoch": 1, "asr_overall": 0.45}\n', encoding="utf-8")
    checkpoint = run_dir / "center_runs" / "ningbo" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"tiny fake checkpoint")
    metrics_long = run_dir / "eval" / "metrics_long.csv"
    _write_metrics_long(metrics_long, run_id="unit_experiment")
    paper_table = run_dir / "eval" / "paper_table.csv"
    _write_paper_table(paper_table, metrics_long=metrics_long)
    external_metrics = tmp_path / "metrics_export" / "unit_run" / "metrics_long.csv"

    manifest = {
        "manifest_schema_version": 2,
        "run_id": "unit_run",
        "status": "succeeded",
        "config_hash_sha256": "0" * 64,
        "git": {"commit": "16446f14b071e6fc06898b92db374ba05fe42cb0", "branch": "main", "status_short": ""},
        "commands": [
            {
                "name": "unit_child",
                "cwd": str(run_dir),
                "argv": ["python", "train.py", "--seed", "7"],
                "env": {},
            }
        ],
        "command_runs": [
            {
                "command_index": 0,
                "name": "unit_child",
                "returncode": 0,
                "status": "succeeded",
                "log_path": str(logs_dir / "command_00.log"),
            }
        ],
        "experiment": {
            "name": "unit_experiment",
            "description": "Check that run finalization records purpose and results.",
        },
        "paper_protocol": {
            "mapping_version": "vtest",
            "mapping_hash": "htest",
            "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            "centers": {"target_4": ["ningbo", "georgia"]},
            "kshot": {"mode": "fixed_k", "k": 500, "seed": 7, "exclude_refs_from_eval": True},
            "selection": {
                "policy": "k500_internal_val_plus_source_floor",
                "allowed_data": ["target_k500_internal_val", "ptbxl_source_floor"],
            },
        },
        "artifact_trace": {
            "schema_version": 1,
            "metrics": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "selection_policy": {
                "policy": "k500_internal_val_plus_source_floor",
                "allowed_data": ["target_k500_internal_val", "ptbxl_source_floor"],
            },
            "inputs": {
                "checkpoints": [],
                "data_caches": [],
                "init_heads": [],
                "k500_refs": [
                    {"center": "ningbo", "anchor_base": str(tmp_path / "refs" / "ningbo")},
                    {"center": "georgia", "anchor_base": str(tmp_path / "refs" / "georgia")},
                ],
            },
            "expected_outputs": {
                "launch_artifacts": [
                    {"role": "run_manifest", "path": str(run_dir / "run_manifest.json"), "required": True},
                    {"role": "run_config_yaml", "path": str(run_dir / "run_config.resolved.yaml"), "required": True},
                    {"role": "run_config_json", "path": str(run_dir / "run_config.resolved.json"), "required": True},
                    {"role": "command_sh", "path": str(run_dir / "command.sh"), "required": True},
                    {"role": "data_manifest", "path": str(run_dir / "data_manifest.json"), "required": True},
                    {"role": "k500_ref_ids", "path": str(run_dir / "k500_ref_ids.json"), "required": True},
                    {"role": "selection_record", "path": str(run_dir / "selection.json"), "required": True},
                ],
                "child_runs": [
                    {
                        "command_index": 0,
                        "center": "ningbo",
                        "expected_artifacts": [
                            {"role": "best_model", "path": str(checkpoint), "required": True},
                            {"role": "diagnostics_epoch", "path": str(diagnostics), "required": True},
                        ],
                    }
                ],
                "postprocess_runs": [
                    {
                        "command_index": 0,
                        "expected_artifacts": [
                            {"role": "metrics_long", "path": str(metrics_long), "required": True},
                            {"role": "paper_table", "path": str(paper_table), "required": True},
                            {"role": "paper_table_manifest", "path": str(paper_table.parent / "paper_table_manifest.json"), "required": True},
                            {"role": "metrics_long", "path": str(external_metrics), "required": True},
                        ],
                    }
                ],
            },
        },
        "artifact_verification": {
            "passed": True,
            "verified_artifacts": [
                {"role": "metrics_long", "path": str(metrics_long), "exists": True, "content_verified": True}
            ],
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_finalize_run_record_writes_agent_readable_card_and_file_index(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)

    card = finalize_run_record(
        run_dir,
        purpose="Test whether the run manager records experiment intent.",
        result_summary="The synthetic test run produced two center metrics.",
        outcome="provisional",
    )

    assert card["run_id"] == "unit_run"
    assert card["experiment"]["purpose"] == "Test whether the run manager records experiment intent."
    assert card["result"]["summary"] == "The synthetic test run produced two center metrics."
    assert card["result"]["outcome"] == "provisional"
    assert card["metric_summary"]["pn2021_all_zero_kept_refexcluded"]["macro_auroc"] == 0.83
    assert card["metric_summary"]["pn2021_all_zero_kept_refexcluded"]["macro_auprc"] == 0.54

    assert (run_dir / "run_card.json").exists()
    assert (run_dir / "run_file_index.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "configs" / "run_config.resolved.yaml").exists()
    assert (run_dir / "manifests" / "run_manifest.snapshot.json").exists()
    assert (run_dir / "reports" / "summary.md").exists()

    index = json.loads((run_dir / "run_file_index.json").read_text(encoding="utf-8"))
    categories = index["categories"]
    assert any(item["relative_path"].endswith("command_00.log") for item in categories["logs"])
    assert any(item["relative_path"].endswith("best_model.pt") for item in categories["checkpoints"])
    assert any(item["relative_path"].endswith("metrics_long.csv") for item in categories["eval"])
    assert any(item["external"] and item["role"] == "metrics_long" for item in categories["eval"])
    assert any(item["relative_path"].endswith("diagnostics_epoch.jsonl") for item in categories["diagnostics"])


def test_finalize_run_record_uses_manifest_run_record_defaults(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    manifest = _load_manifest(run_dir)
    manifest["run_record"] = {
        "purpose": "Use the YAML-derived purpose when finalizer args are omitted.",
        "result_summary": "The manifest run-record summary finalized this run.",
        "outcome": "provisional",
    }
    _write_manifest(run_dir, manifest)

    card = finalize_run_record(run_dir)

    assert card["experiment"]["purpose"] == "Use the YAML-derived purpose when finalizer args are omitted."
    assert card["result"]["summary"] == "The manifest run-record summary finalized this run."
    assert card["result"]["outcome"] == "provisional"


def test_finalize_run_record_requires_explicit_result_summary(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)

    with pytest.raises(RunRecordError, match="result_summary"):
        finalize_run_record(
            run_dir,
            purpose="This purpose is not enough to make the result reproducible.",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["git"].pop("commit"), "git.commit"),
        (lambda manifest: manifest.update({"commands": []}), "commands"),
        (lambda manifest: manifest["artifact_trace"]["inputs"].update({"k500_refs": []}), "k500_refs"),
        (lambda manifest: manifest["artifact_trace"]["expected_outputs"].update({"postprocess_runs": []}), "expected eval"),
    ],
)
def test_finalize_run_record_rejects_missing_replay_contract(tmp_path: Path, mutation, message: str):
    run_dir = _make_run_dir(tmp_path)
    manifest = _load_manifest(run_dir)
    mutation(manifest)
    _write_manifest(run_dir, manifest)

    with pytest.raises(RunRecordError, match=message):
        finalize_run_record(
            run_dir,
            purpose="Test whether replay contract gaps are rejected.",
            result_summary="This run should not finalize because replay evidence is incomplete.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_missing_selection_record(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "selection.json").unlink()

    with pytest.raises(RunRecordError, match="selection.json"):
        finalize_run_record(
            run_dir,
            purpose="Test whether selection provenance is mandatory.",
            result_summary="This run should not finalize without selection provenance.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_forbidden_selection_content(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    selection["selection_safety"]["heldout_target_labels_used_for_selection"] = True
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    with pytest.raises(RunRecordError, match="selection.*held-out"):
        finalize_run_record(
            run_dir,
            purpose="Test whether forbidden selection content is rejected.",
            result_summary="This run should not finalize with held-out selection leakage.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_bad_k500_ref_content(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    payload = json.loads((run_dir / "k500_ref_ids.json").read_text(encoding="utf-8"))
    payload["centers"]["georgia"]["ref_record_ids_sha256"] = "bad"
    (run_dir / "k500_ref_ids.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(RunRecordError, match="k500_ref_ids.*sha256"):
        finalize_run_record(
            run_dir,
            purpose="Test whether K500 content hashes are enforced.",
            result_summary="This run should not finalize with a bad K500 hash.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_metrics_long_mapping_mismatch(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    metrics_path = run_dir / "eval" / "metrics_long.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline="")))
    rows[0]["mapping_hash"] = "wrong"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RunRecordError, match="metrics_long.*mapping_hash"):
        finalize_run_record(
            run_dir,
            purpose="Test whether metrics mapping metadata is enforced.",
            result_summary="This run should not finalize with mismatched metrics metadata.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_metrics_long_missing_mapping_metadata(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    metrics_path = run_dir / "eval" / "metrics_long.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline="")))
    rows[0]["mapping_version"] = ""
    rows[0]["mapping_hash"] = ""
    rows[0]["class_order"] = ""
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RunRecordError, match="metrics_long.*mapping_version"):
        finalize_run_record(
            run_dir,
            purpose="Test whether metrics mapping metadata is mandatory.",
            result_summary="This run should not finalize with missing metrics metadata.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_k500_missing_class_order(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    payload = json.loads((run_dir / "k500_ref_ids.json").read_text(encoding="utf-8"))
    payload["paper_protocol"].pop("class_order")
    (run_dir / "k500_ref_ids.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(RunRecordError, match="k500_ref_ids.*class_order"):
        finalize_run_record(
            run_dir,
            purpose="Test whether K500 mapping metadata is mandatory.",
            result_summary="This run should not finalize with missing K500 class order.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_paper_table_unknown_source_file(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    table_path = run_dir / "eval" / "paper_table.csv"
    rows = list(csv.DictReader(table_path.open("r", encoding="utf-8", newline="")))
    rows[0]["source_files"] = str(tmp_path / "missing_eval_result.json")
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "dataset", "view", "metric", "value", "source_files"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RunRecordError, match="paper_table.*source_files"):
        finalize_run_record(
            run_dir,
            purpose="Test whether paper table sources are traced.",
            result_summary="This run should not finalize with unknown paper table sources.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_paper_table_empty_source_files(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    table_path = run_dir / "eval" / "paper_table.csv"
    rows = list(csv.DictReader(table_path.open("r", encoding="utf-8", newline="")))
    rows[0]["source_files"] = ""
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "dataset", "view", "metric", "value", "source_files"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RunRecordError, match="paper_table.*source_files"):
        finalize_run_record(
            run_dir,
            purpose="Test whether paper table sources are mandatory.",
            result_summary="This run should not finalize with empty paper table sources.",
            outcome="provisional",
        )


def test_finalize_run_record_rejects_empty_paper_table_manifest_metrics_long(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    manifest_path = run_dir / "eval" / "paper_table_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metrics_long"] = ""
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(RunRecordError, match="paper_table_manifest.*metrics_long"):
        finalize_run_record(
            run_dir,
            purpose="Test whether paper table manifest provenance is mandatory.",
            result_summary="This run should not finalize with empty paper-table provenance.",
            outcome="provisional",
        )


def test_register_run_in_registry_requires_existing_finalized_record(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    registry_path = tmp_path / "registry.yaml"
    local_config_path = tmp_path / "local.yaml"
    registry_path.write_text("schema_version: 1\nrun_catalog:\n  provisional: []\n", encoding="utf-8")
    local_config_path.write_text(f"paths:\n  output_root: {tmp_path / 'runs'}\n", encoding="utf-8")

    with pytest.raises(RunRecordError, match="run_card.json"):
        register_run_in_registry(
            registry_path=registry_path,
            local_config_path=local_config_path,
            run_dir=run_dir,
            status="provisional",
        )


def test_register_run_in_registry_uses_output_root_placeholder(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    finalize_run_record(
        run_dir,
        purpose="Register this run without host-specific tracked paths.",
        result_summary="Registration succeeded.",
        outcome="trusted",
    )
    registry_path = tmp_path / "registry.yaml"
    local_config_path = tmp_path / "local.yaml"
    registry_path.write_text("schema_version: 1\nrun_catalog:\n  trusted: []\n", encoding="utf-8")
    local_config_path.write_text(f"paths:\n  output_root: {tmp_path / 'runs'}\n", encoding="utf-8")

    updated = register_run_in_registry(
        registry_path=registry_path,
        local_config_path=local_config_path,
        run_dir=run_dir,
        status="trusted",
    )
    updated_again = register_run_in_registry(
        registry_path=registry_path,
        local_config_path=local_config_path,
        run_dir=run_dir,
        status="trusted",
    )

    assert len(updated["managed_runs"]) == 1
    assert len(updated_again["managed_runs"]) == 1
    entry = updated_again["managed_runs"][0]
    assert entry["run_dir"] == "${paths.output_root}/20260529/unit_run"
    assert entry["run_card"] == "${paths.output_root}/20260529/unit_run/run_card.json"
    assert entry["summary"] == "${paths.output_root}/20260529/unit_run/summary.md"
    assert entry["purpose"] == "Register this run without host-specific tracked paths."

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    dumped = yaml.safe_dump(raw)
    assert str(tmp_path) not in dumped


def test_register_run_in_registry_can_auto_resolve_status_from_run_record(tmp_path: Path):
    run_dir = _make_run_dir(tmp_path)
    manifest = _load_manifest(run_dir)
    manifest["run_record"] = {
        "purpose": "Register with auto status from YAML-derived metadata.",
        "result_summary": "Auto registration selected the manifest status.",
        "outcome": "provisional",
        "registration_status": "trusted",
    }
    _write_manifest(run_dir, manifest)
    finalize_run_record(run_dir)
    registry_path = tmp_path / "registry.yaml"
    local_config_path = tmp_path / "local.yaml"
    registry_path.write_text("schema_version: 1\nrun_catalog:\n  trusted: []\n", encoding="utf-8")
    local_config_path.write_text(f"paths:\n  output_root: {tmp_path / 'runs'}\n", encoding="utf-8")

    updated = register_run_in_registry(
        registry_path=registry_path,
        local_config_path=local_config_path,
        run_dir=run_dir,
        status="auto",
    )

    entry = updated["managed_runs"][0]
    assert entry["status"] == "trusted"
    assert "unit_experiment/unit_run" in updated["run_catalog"]["trusted"]
    assert "auto" not in updated["run_catalog"]
