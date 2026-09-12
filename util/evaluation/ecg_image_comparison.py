"""Read-only paired hard-label analysis for image-LLM pilots and full cohorts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import yaml

from util.evaluation.ecg_image_artifact import LABEL_PATTERN, parse_response, validate_result
from util.evaluation.ecg_image_metrics import CLASS_ORDER, paired_bootstrap, paired_metrics
from util.pn2021_artifact_contract import sha256_file
from util.run_record import verify_run_file_index

CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")


def audit_parser(rows: list[dict], *, pulse: bool) -> int:
    """Verify stored labels and count changes under one common final-answer parser.

    No prediction is rewritten. PULSE historically uppercases before token
    extraction; new frozen inference accepts uppercase codes and strips reasoning.
    """
    changes = 0
    for row in rows:
        text = row["response"]
        expected = ([label for label in CLASS_ORDER if label in set(LABEL_PATTERN.findall(text.upper()))]
                    if pulse else parse_response(text)["predicted_labels"])
        if row["predicted_labels"] != expected:
            raise ValueError("stored labels differ from the recorded backend parser")
        normalized = re.sub(LABEL_PATTERN.pattern, lambda match: match.group().upper(), text, flags=re.IGNORECASE)
        common = parse_response(normalized)["predicted_labels"]
        changes += common != expected
    return changes


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_diagnostics(items: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, evidence = [], []
    for item in items:
        run_dir = Path(item["run_dir"])
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if manifest["status"] not in {"complete", "failed"} or verify_run_file_index(run_dir):
            raise ValueError("diagnostic run is unfinished or its file index changed")
        performance = {}
        result_path = run_dir / "evaluation" / "evaluation_result.json"
        if manifest["status"] == "complete":
            result = json.loads(result_path.read_text())
            validate_result(result, result_path)
            if result["protocol"]["stage"] != "smoke":
                raise ValueError("admission diagnostics must use only smoke records")
            performance = result["performance"]
        log_path = run_dir / "logs" / "delegate.log"
        log = log_path.read_text()
        reason = "CUDA OOM" if "torch.cuda.OutOfMemoryError" in log else (
            "delegate_failed_see_log" if manifest["status"] == "failed" else "smoke_only_not_accuracy")
        rows.append({"probe": item["name"], "status": manifest["status"],
                     "predictions": performance.get("completed"), "no_code_outputs": performance.get("parse_failures"),
                     "non_list_outputs": performance.get("strict_list_failures"),
                     "token_limit_hits": performance.get("token_limit_hits"),
                     "images_per_second": performance.get("images_per_second"), "note": reason})
        evidence.append({"probe": item["name"], "run_dir": str(run_dir),
                         "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                         "delegate_log_sha256": sha256_file(log_path),
                         "result_sha256": sha256_file(result_path) if result_path.exists() else None})
    return rows, evidence


def join_shards(paths: list[Path]) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """Reject incomplete, mixed-protocol, duplicate-rank, or unrecorded shards."""
    if len(paths) == 1:
        path = paths[0]
        result = json.loads(path.read_text())
        if result.get("schema_version") == 2:
            validate_result(result, path)
            if result["protocol"]["stage"] != "evaluate" or result["protocol"].get("cohort_mode") != "full_pulse":
                raise ValueError("smoke predictions cannot enter formal comparison")
            run_dir = path.parent.parent
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            if manifest["status"] != "complete" or verify_run_file_index(run_dir):
                raise ValueError("elastic run recorder verification failed")
            return (read_jsonl(path.parent / result["prediction_artifact"]["path"]),
                    read_jsonl(path.parent / "cohort.jsonl"), result,
                    [{"result_path": str(path), "result_sha256": sha256_file(path),
                      "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                      "performance": result["performance"], "scheduler": "fixed_task_queue_v1"}])
    rows, evidence, ranks = [], [], set()
    first = None
    identity = None
    cohort = None
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("schema_version") != 1:
            raise ValueError("elastic full results cannot be mixed with legacy rank shards")
        validate_result(result, path)
        protocol = result["protocol"]
        if protocol["stage"] != "evaluate":
            raise ValueError("smoke predictions cannot enter formal comparison")
        run_dir = path.parent.parent
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if manifest["status"] != "complete" or verify_run_file_index(run_dir):
            raise ValueError("shard run recorder verification failed")
        current = {"model": result["model"], "protocol": {k: v for k, v in protocol.items() if k != "rank"}}
        # Asset verification cache provenance may differ across replicas, but
        # the actual checkpoint identity must agree exactly.
        current["model"] = {k: v for k, v in result["model"].items() if k != "assets"}
        assets = result["model"]["assets"]
        current["model"]["asset_identity"] = {
            "repo_id": assets["repo_id"], "revision": assets["revision"],
            "config_files": assets["config_files"],
            "shards": [{key: shard[key] for key in ("name", "size", "sha256")} for shard in assets["shards"]],
        }
        if first is None:
            first, identity = result, current
            cohort = read_jsonl(path.parent / "cohort.jsonl")
        elif current != identity:
            raise ValueError("mixed model or protocol in shard group")
        if protocol["rank"] in ranks:
            raise ValueError("duplicate shard rank")
        ranks.add(protocol["rank"])
        rows.extend(read_jsonl(path.parent / result["prediction_artifact"]["path"]))
        evidence.append({"result_path": str(path), "result_sha256": sha256_file(path),
                         "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                         "performance": result["performance"], "rank": protocol["rank"]})
    if first is None or ranks != set(range(first["protocol"]["world_size"])):
        raise ValueError("all declared ranks must be complete")
    expected = {(r["sample_key"], c["condition_id"]) for r in cohort for c in first["protocol"]["conditions"]}
    keys = [(r["sample_key"], r["condition_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("shard union differs from full cohort-condition grid")
    return rows, cohort, first, evidence


def select_pulse(directory: Path, cohort: list[dict], conditions: list[dict], protocol_sha: str) -> tuple[list[dict], list[dict]]:
    if sha256_file(directory / "protocol.json") != protocol_sha:
        raise ValueError("PULSE protocol changed")
    samples = {r["sample_key"]: r for r in cohort}
    expected = {(key, c["condition_id"]) for key in samples for c in conditions}
    selected, evidence, seen = [], [], set()
    for rank in range(4):
        path = directory / "predictions" / f"predictions.rank-{rank:03d}.jsonl"
        runtime_path = directory / f"runtime.rank-{rank:03d}.json"
        runtime = json.loads(runtime_path.read_text())
        digest = sha256_file(path)
        if runtime["status"] != "complete" or digest != runtime["predictions_sha256"]:
            raise ValueError("PULSE frozen prediction file is incomplete or changed")
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["sample_key"] not in samples:
                    continue
                key = (row["sample_key"], row["condition_id"])
                sample = samples[key[0]]
                if key in seen or key not in expected or set(row["true_labels"]) != set(sample["label_names"]):
                    raise ValueError("PULSE duplicate/foreign prediction or label mismatch")
                if row["logical_center"] != sample["logical_center"] or row["hash_id"] != sample["hash_id"]:
                    raise ValueError("PULSE sample identity mismatch")
                selected.append(row)
                seen.add(key)
        evidence.append({"prediction_path": str(path), "sha256": digest,
                         "runtime_sha256": sha256_file(runtime_path),
                         "historical_images_per_second": runtime["predictions_written_this_process"] / runtime["elapsed_seconds"],
                         "peak_allocated_bytes": runtime["peak_allocated_bytes"]})
    if seen != expected:
        raise ValueError("PULSE has not covered the exact selected grid")
    return selected, evidence


def build_arrays(cohort: list[dict], conditions: list[str], by_model: dict[str, list[dict]],
                 *, require_balanced: bool = True) -> tuple[dict, dict]:
    truth, prediction = {}, {}
    expected = {(r["sample_key"], condition) for r in cohort for condition in conditions}
    lookup = {}
    for model, rows in by_model.items():
        indexed = {(r["sample_key"], r["condition_id"]): r for r in rows}
        if len(rows) != len(indexed) or set(indexed) != expected:
            raise ValueError("comparison requires complete identical paired grids")
        for row in rows:
            if not set(row["predicted_labels"]) <= set(CLASS_ORDER):
                raise ValueError("foreign predicted class")
        lookup[model] = indexed
    for center in CENTERS:
        samples = [r for r in cohort if r["logical_center"] == center]
        if not samples or any(not any(r["label"]) for r in samples):
            raise ValueError("empty center or non-dropzero cohort")
        truth[center] = np.asarray([r["label"] for r in samples], dtype=np.uint8)
        prediction[center] = np.asarray([
            [[[int(label in lookup[model][(r["sample_key"], condition)]["predicted_labels"])
               for label in CLASS_ORDER] for condition in conditions] for r in samples]
            for model in by_model
        ], dtype=np.uint8)
    if require_balanced and len({len(v) for v in truth.values()}) != 1:
        raise ValueError("primary pilot requires center-balanced sampling")
    return truth, prediction


def make_overview(metrics: dict, bootstrap: dict) -> list[dict]:
    rows = []
    for model, scores in metrics["models"].items():
        for center, values in [("center_equal", scores["center_equal"]), *scores["centers"].items()]:
            clean, corrupt = values["clean"], values["corruption_view_mean"]
            ci = (bootstrap if center == "center_equal" else bootstrap["centers"][center])["models"][model]
            rows.append({"model": model, "center": center,
                         "clean_macro_f1": clean["macro_f1"], "corrupted_macro_f1": corrupt["macro_f1"],
                         "drop_pp": 100 * (clean["macro_f1"] - corrupt["macro_f1"]),
                         "drop_pp_ci95_low": ci["drop_pp_ci95"][0], "drop_pp_ci95_high": ci["drop_pp_ci95"][1],
                         "clean_micro_f1": clean["micro_f1"], "corrupted_micro_f1": corrupt["micro_f1"],
                         "clean_exact_match": clean["exact_match_accuracy"], "corrupted_exact_match": corrupt["exact_match_accuracy"],
                         "clean_hamming_loss": clean["hamming_loss"], "corrupted_hamming_loss": corrupt["hamming_loss"]})
    return rows


def exclude_prior_records(cohort: list[dict], truth: dict, prediction: dict,
                          exposed: list[str]) -> tuple[dict, dict, dict]:
    """Apply an identity-only exclusion to every model and every view together."""
    if len(exposed) != len(set(exposed)) or not set(exposed) <= {r["sample_key"] for r in cohort}:
        raise ValueError("prior exposure identities must be unique members of the full cohort")
    excluded = set(exposed)
    kept_truth, kept_prediction, counts = {}, {}, {}
    for center in CENTERS:
        records = [r for r in cohort if r["logical_center"] == center]
        keep = np.asarray([r["sample_key"] not in excluded for r in records])
        if len(keep) != len(truth[center]) or not keep.any():
            raise ValueError("sensitivity exclusion emptied or mismatched a center")
        kept_truth[center], kept_prediction[center] = truth[center][keep], prediction[center][:, keep]
        counts[center] = {"full_records": len(keep), "excluded_records": int((~keep).sum()),
                          "retained_records": int(keep.sum())}
    return kept_truth, kept_prediction, counts


def run(config_path: Path, config_root: Path, output_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    required = {"schema_version", "models", "baseline_run", "bootstrap", "limitations", "report"}
    if not required <= set(config) <= required | {"diagnostics", "sensitivity"} or config["schema_version"] != 1:
        raise ValueError("invalid image comparison config")
    if config["report"] != {"renderer": "data_analytics_portable_v1"}:
        raise ValueError("unsupported report renderer")
    if not config["models"] or not 100 <= config["bootstrap"]["repeats"] <= 2000:
        raise ValueError("invalid comparison size")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_model, provenance, quality = {}, {}, []
    cohort, common, input_fingerprints = None, None, None
    cohort_modes, exposure_files, exposed_ids = set(), [], None
    for item in config["models"]:
        if item["name"] in by_model or item["name"] == "PULSE-7B" or Path(item["name"]).name != item["name"]:
            raise ValueError("duplicate/reserved model name")
        rows, selected, result, evidence = join_shards([Path(p) for p in item["results"]])
        protocol = result["protocol"]
        cohort_modes.add(protocol.get("cohort_mode", "balanced_pilot"))
        if protocol.get("cohort_mode") == "full_pulse":
            exposure_path = Path(item["results"][0]).parent / "prior_exposed_records.json"
            current_exposure = json.loads(exposure_path.read_text())
            if exposed_ids is not None and current_exposure != exposed_ids:
                raise ValueError("models disagree about prior exposure identity set")
            exposed_ids = current_exposure
            exposure_files.append({"path": str(exposure_path), "sha256": sha256_file(exposure_path)})
        shared = {k: protocol[k] for k in ("cohort_sha256", "baseline_protocol_sha256", "conditions", "mapping_hash", "metric_view")}
        if common is None:
            cohort, common = selected, shared
        elif shared != common:
            raise ValueError("models evaluated on different inputs or cohorts")
        fingerprints = {(row["sample_key"], row["condition_id"]): (row["clean_waveform_sha256"], row["input_waveform_sha256"]) for row in rows}
        if input_fingerprints is None:
            input_fingerprints = fingerprints
        elif input_fingerprints != fingerprints:
            raise ValueError("new models consumed different actual input waveform bytes")
        by_model[item["name"]] = rows
        provenance[item["name"]] = {"model": result["model"], "shards": evidence}
    pulse, pulse_evidence = select_pulse(Path(config["baseline_run"]), cohort, common["conditions"], common["baseline_protocol_sha256"])
    by_model = {"PULSE-7B": pulse, **by_model}
    provenance["PULSE-7B"] = {
        "frozen_prediction_files": pulse_evidence,
        "frozen_protocol_sha256": common["baseline_protocol_sha256"],
        "frozen_protocol": json.loads((Path(config["baseline_run"]) / "protocol.json").read_text()),
        "runtime_note": "historical throughput, not a new matched speed run",
    }
    for model, rows in by_model.items():
        for center in CENTERS:
            subset = [r for r in rows if r["logical_center"] == center]
            failed = sum(not r["predicted_labels"] for r in subset)
            quality.append({"model": model, "center": center, "n_predictions": len(subset),
                            "parse_failures": failed, "parse_failure_rate": failed / len(subset),
                            "non_list_outputs": sum(not r["strict_label_list"] for r in subset) if model != "PULSE-7B" else None,
                            "common_parser_label_changes": audit_parser(subset, pulse=model == "PULSE-7B"),
                            "normal_abnormal_conflicts": sum("NORM" in r["predicted_labels"] and len(r["predicted_labels"]) > 1 for r in subset),
                            "token_limit_hits": sum(r.get("hit_token_limit", False) for r in subset) if model != "PULSE-7B" else None})
        with (output_dir / f"predictions.{model}.jsonl").open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    conditions = [c["condition_id"] for c in common["conditions"]]
    if len(cohort_modes) != 1:
        raise ValueError("cannot mix full and pilot cohort modes")
    full = cohort_modes == {"full_pulse"}
    if full and config.get("sensitivity") != {"exclude_prior_pilot_and_smoke": True}:
        raise ValueError("full comparison requires the predeclared prior-exposure sensitivity check")
    if not full and "sensitivity" in config:
        raise ValueError("prior-exposure sensitivity is only defined for full cohorts")
    truth, prediction = build_arrays(cohort, conditions, by_model, require_balanced=not full)
    metrics = paired_metrics(truth, prediction, list(by_model), conditions)
    bootstrap = paired_bootstrap(truth, prediction, list(by_model), **config["bootstrap"])
    class_support = [{"center": center, "n_records": len(values), **dict(zip(CLASS_ORDER, map(int, values.sum(0))))}
                     for center, values in truth.items()]
    overview = make_overview(metrics, bootstrap)
    sensitivity = None
    if full:
        retained_truth, retained_prediction, counts = exclude_prior_records(cohort, truth, prediction, exposed_ids)
        sensitivity_metrics = paired_metrics(retained_truth, retained_prediction, list(by_model), conditions)
        sensitivity_bootstrap = paired_bootstrap(retained_truth, retained_prediction, list(by_model), **config["bootstrap"])
        sensitivity = {"rule": "exclude_prior_pilot_and_smoke_by_record_identity_no_outcome_selection",
                       "excluded_record_ids": exposed_ids, "exclusion_sources": exposure_files, "center_counts": counts,
                       "full_records": len(cohort), "excluded_records": len(exposed_ids),
                       "retained_records": sum(len(v) for v in retained_truth.values()),
                       "overview": make_overview(sensitivity_metrics, sensitivity_bootstrap)}
        write_json(output_dir / "sensitivity_metrics.json", sensitivity_metrics)
        write_json(output_dir / "sensitivity_bootstrap.json", sensitivity_bootstrap)
        write_json(output_dir / "sensitivity.json", sensitivity)
        write_csv(output_dir / "sensitivity_overview.csv", sensitivity["overview"])
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "bootstrap.json", bootstrap)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "parse_quality.json", quality)
    write_json(output_dir / "cohort.json", cohort)
    diagnostics, diagnostic_evidence = collect_diagnostics(config.get("diagnostics", []))
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "diagnostic_provenance.json", diagnostic_evidence)
    for name, rows in (("overview", overview), ("condition_metrics", metrics["condition_rows"]),
                       ("class_metrics", metrics["class_rows"]), ("class_support", class_support), ("parse_quality", quality)):
        write_csv(output_dir / f"{name}.csv", rows)
    from util.evaluation.ecg_image_report import build_report

    build_report(output_dir, metrics, bootstrap, overview, quality, class_support, provenance, config,
                 diagnostics=diagnostics, cohort_mode="full_pulse" if full else "balanced_pilot", sensitivity=sensitivity)
    files = {p.name: sha256_file(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    write_json(output_dir / "comparison_result.json", {
        "schema_version": 1, "artifact_type": "ecg_image_comparison_result", "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(), "models": list(by_model),
        "cohort": {"records": len(cohort), "per_center": None if full else len(cohort) // 4,
                   "center_counts": {c: len(v) for c, v in truth.items()}, "mode": "full_pulse" if full else "balanced_pilot",
                   "conditions": conditions, **common},
        "files": files, "limitations": config["limitations"],
        "summary": {model: value["center_equal"] for model, value in metrics["models"].items()},
        "evidence_level": "full_development_not_blind_not_strict_ood" if full else "development_pilot_not_paper_final",
    })
