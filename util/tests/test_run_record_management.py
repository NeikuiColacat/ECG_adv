"""Tests for agent-readable run finalization and registration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from ecg_adv_gen.evidence import finalize_run_record, register_run_in_registry
from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES


def _write_metrics_long(path: Path, *, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for center, auroc, auprc in [("ningbo", 0.81, 0.52), ("georgia", 0.85, 0.56)]:
        for metric, value in [("macro_auroc", auroc), ("macro_auprc", auprc)]:
            row = {key: "" for key in METRICS_FIELDNAMES}
            row.update(
                {
                    "run_id": run_id,
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


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "20260529" / "unit_run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.resolved.yaml").write_text("experiment:\n  name: unit_experiment\n", encoding="utf-8")
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\npython train.py\n", encoding="utf-8")
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
    external_metrics = tmp_path / "metrics_export" / "unit_run" / "metrics_long.csv"

    manifest = {
        "schema_version": 1,
        "run_id": "unit_run",
        "status": "succeeded",
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
        },
        "artifact_trace": {
            "metrics": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "expected_outputs": {
                "launch_artifacts": [
                    {"role": "run_manifest", "path": str(run_dir / "run_manifest.json"), "required": True},
                    {"role": "run_config_yaml", "path": str(run_dir / "run_config.resolved.yaml"), "required": True},
                    {"role": "command_sh", "path": str(run_dir / "command.sh"), "required": True},
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
                            {"role": "metrics_long", "path": str(external_metrics), "required": True},
                        ],
                    }
                ],
            },
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


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
