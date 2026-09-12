"""CPU-only configuration, fixed-task and prediction contracts for PULSE SFT."""
from __future__ import annotations

import json
from pathlib import Path

from util.evaluation.ecg_image_artifact import parse_response
from util.evaluation.ecg_image_queue import fixed_tasks
from util.pn2021_artifact_contract import sha256_file
from util.pulse_training_contract import CENTERS, CLASS_ORDER

ARMS = ("w1", "w2")
MODEL_NAMES = ("PULSE-SFT-single-chain", "PULSE-SFT-two-chain")


def validate_config(config):
    if set(config) != {"schema_version", "stage", "references", "baseline", "cohort", "training_results",
                       "admission_training", "paths", "runtime"} or config["schema_version"] != 1:
        raise ValueError("invalid paired PULSE benchmark schema")
    if config["stage"] not in {"admission", "benchmark"}:
        raise ValueError("unsupported paired PULSE stage")
    required = {"evaluation_config", "operators_config", "data_load_config", "split_config"}
    if config["stage"] == "benchmark":
        required.add("admission_config")
    if set(config["references"]) != required or config["cohort"] != {"mode": "full_pulse"}:
        raise ValueError("incomplete fixed input/config closure")
    if set(config["training_results"]) != set(CENTERS) or set(config["admission_training"]) != set(ARMS):
        raise ValueError("all four independent center pairs and the engineering pair are required")
    for pair in config["training_results"].values():
        if set(pair) != set(ARMS):
            raise ValueError("missing matched training arm")
    expected = {"batch_size": 2, "max_new_tokens": 32, "loader_workers": 2, "cpu_threads": 2,
                "records_per_task": 8, "development_records_per_center": 128, "max_workers": 4,
                "poll_seconds": 30, "idle_seconds": 30, "min_available_ram_gib": 48,
                "max_load_fraction": 0.9, "min_disk_free_gib": 20,
                "bootstrap_repeats": 1000, "bootstrap_seed": 20260909}
    if config["runtime"] != expected:
        raise ValueError("paired inference, development subset and resource budgets must be frozen")
    if set(config["paths"]) != {"state_root", "raw_root", "toolkit_dir", "temporary_root"}:
        raise ValueError("invalid paired benchmark paths")
    paths = [*config["paths"].values(), *config["admission_training"].values(), config["baseline"]["run_dir"]]
    paths.extend(path for pair in config["training_results"].values() for path in pair.values())
    for raw in paths:
        Path(raw).resolve().relative_to(Path("/home/linbinhao/ECG_adv_data"))


def center_tasks(rows, conditions, protocol_identity, *, records_per_task=8, batch_size=2, development_records=128):
    tasks = []
    for center in CENTERS:
        selected = [r for r in rows if r["logical_center"] == center]
        if len(selected) < development_records or development_records % records_per_task:
            raise ValueError("development subset must be fixed complete record tasks")
        group = fixed_tasks(selected, records_per_task=records_per_task, batch_size=batch_size,
                            condition_ids=[c["condition_id"] for c in conditions], protocol_identity=protocol_identity)
        for task in group:
            task["development"] = task["index"] < development_records // records_per_task
            task["center"] = center
            task["index"] = len(tasks)
            tasks.append(task)
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate paired task identity")
    return tasks


def validate_pair_row(row, sample, condition):
    if (row["sample_key"] != sample["sample_key"] or row["logical_center"] != sample["logical_center"]
            or row["record_id"] != sample["record_id"] or row["hash_id"] != sample["hash_id"]
            or row["true_labels"] != sample["label_names"] or row["condition_id"] != condition["condition_id"]
            or row["operators"] != condition["operators"] or row["depth"] != condition["depth"]
            or set(row["arms"]) != set(ARMS)):
        raise ValueError("paired prediction sample/condition/arm identity mismatch")
    for key in ("clean_waveform_sha256", "input_waveform_sha256"):
        digest = row.get(key, "")
        if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
            raise ValueError("paired input waveform fingerprint missing")
    for answer in row["arms"].values():
        parsed = parse_response(answer["response"])
        if any(answer.get(k) != value for k, value in parsed.items()):
            raise ValueError("paired response differs from its frozen parser")
        if type(answer["generated_tokens"]) is not int or not 1 <= answer["generated_tokens"] <= 32:
            raise ValueError("invalid generated-answer length")
        if type(answer["hit_max_new_tokens"]) is not bool:
            raise ValueError("missing truncation audit")


def validate_result(payload, path):
    if (payload.get("artifact_type") != "pulse_benchmark_result" or payload.get("schema_version") != 1
            or payload.get("status") != "complete" or payload.get("stage") not in {"admission", "benchmark"}):
        raise ValueError("invalid paired benchmark artifact")
    if tuple(payload["protocol"]["class_order"]) != CLASS_ORDER or payload["protocol"]["mapping_hash"] != "555ec85d5b51":
        raise ValueError("paired result mapping changed")
    required = {"engineering_predictions.json", "rendering.json"} if payload["stage"] == "admission" else {
        "cohort.json", "paired_predictions.jsonl", "metrics.json", "bootstrap.json", "exact_bootstrap.json",
        "quality.json", "data_audit.json", "training_lineage.json", "task_index.json", "runtime.json",
        "gpu_admission.json", "development_freeze.json"}
    if not required <= set(payload["files"]):
        raise ValueError("paired benchmark products missing")
    for relative, digest in payload["files"].items():
        member = (path.parent / relative).resolve()
        member.relative_to(path.parent.resolve())
        if sha256_file(member) != digest:
            raise ValueError("paired benchmark result member changed")
    if payload["stage"] == "admission":
        expected = {f"{arm}_batch{batch}" for arm in ARMS for batch in (1, 2)}
        checks = payload["admission"]["author_equivalence"]
        if set(checks) != expected or any(v != {"packing_exact": True, "generated_token_ids_exact": True} for v in checks.values()):
            raise ValueError("missing GPU author-equivalence admission")
        if payload["admission"]["repeated_adapter_switch_exact"] is not True:
            raise ValueError("adapter switching did not replay exactly")
        return
    if payload["protocol"]["metric_view"] != "drop_all_zero" or payload["protocol"]["k500_ref_excluded"] is not True:
        raise ValueError("paired metric/exclusion contract missing")
    if payload["records"] != 39879 or payload["predictions_per_arm"] != 837459:
        raise ValueError("full paired benchmark is incomplete")
    cohort = json.loads((path.parent / "cohort.json").read_text())
    samples = {row["sample_key"]: row for row in cohort}
    conditions = {c["condition_id"]: c for c in payload["protocol"]["conditions"]}
    if len(samples) != 39879 or len(conditions) != 21:
        raise ValueError("full paired cohort/condition identity missing")
    seen = set()
    with (path.parent / "paired_predictions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["sample_key"], row["condition_id"])
            if key in seen or key[0] not in samples or key[1] not in conditions:
                raise ValueError("duplicate or foreign paired prediction")
            validate_pair_row(row, samples[key[0]], conditions[key[1]])
            seen.add(key)
    if len(seen) != 837459:
        raise ValueError("paired prediction grid is incomplete")
