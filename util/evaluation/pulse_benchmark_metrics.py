"""Bounded-memory paired SFT metrics from immutable task commits, never images."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from util.evaluation.ecg_image_metrics import paired_metrics, paired_bootstrap
from util.evaluation.ecg_image_queue import atomic_json
from util.pn2021_artifact_contract import sha256_file
from util.pulse_benchmark_contract import ARMS, MODEL_NAMES, validate_pair_row, validate_result
from util.pulse_training_contract import CENTERS, CLASS_ORDER


def collect(prepared, queue, *, development):
    tasks = [t for t in queue.tasks if t["development"] or not development]
    selected = {key for t in tasks for batch in t["batches"] for key in batch}
    rows = [r for r in prepared["rows"] if r["sample_key"] in selected]
    by_key = {r["sample_key"]: r for r in rows}
    conditions = {c["condition_id"]: c for c in prepared["conditions"]}
    condition_index = {key: i for i, key in enumerate(conditions)}
    truth, predictions, indexes, seen = {}, {}, {}, {}
    quality = {c: {arm: {k: 0 for k in ("predictions", "parse_invalid", "not_strict_label_list",
        "normal_abnormal_conflict", "incomplete_reasoning", "truncated")} for arm in ARMS} for c in CENTERS}
    for center in CENTERS:
        members = [r for r in rows if r["logical_center"] == center]
        indexes.update({r["sample_key"]: (center, i) for i, r in enumerate(members)})
        truth[center] = np.array([[label in r["label_names"] for label in CLASS_ORDER] for r in members], dtype=bool)
        predictions[center] = np.zeros((2, len(members), 21, 5), dtype=bool)
        seen[center] = np.zeros((len(members), 21), dtype=bool)
    for task in tasks:
        payload = queue.validate_completed(task)
        for row in payload["predictions"]:
            validate_pair_row(row, by_key[row["sample_key"]], conditions[row["condition_id"]])
            center, i = indexes[row["sample_key"]]
            j = condition_index[row["condition_id"]]
            if seen[center][i, j]:
                raise ValueError("duplicate paired sample condition")
            seen[center][i, j] = True
            for a, arm in enumerate(ARMS):
                answer, counters = row["arms"][arm], quality[center][arm]
                predictions[center][a, i, j] = [label in answer["predicted_labels"] for label in CLASS_ORDER]
                counters["predictions"] += 1
                counters["parse_invalid"] += not answer["parse_valid"]
                counters["not_strict_label_list"] += not answer["strict_label_list"]
                counters["normal_abnormal_conflict"] += answer["normal_abnormal_conflict"]
                counters["incomplete_reasoning"] += answer["incomplete_reasoning"]
                counters["truncated"] += answer["hit_max_new_tokens"]
    if any(not grid.all() for grid in seen.values()):
        raise ValueError("incomplete paired metric grid")
    return rows, truth, predictions, quality


def exact_bootstrap(truth, predictions, *, repeats, seed):
    """Same record draw for both arms and all views, stratified by center."""
    rng, draws = np.random.default_rng(seed), {}
    for center in CENTERS:
        count = len(truth[center])
        exact = (predictions[center] == truth[center][None, :, None, :]).all(axis=-1)
        weights = rng.multinomial(count, np.full(count, 1 / count), size=repeats).astype(float)
        draws[center] = (weights @ exact.transpose(1, 0, 2).reshape(count, -1).astype(float)).reshape(repeats, 2, 21) / count
    def summarize(values):
        clean, corrupted = values[:, :, 0], values[:, :, 1:].mean(axis=-1)
        interval = lambda x: [float(v) for v in np.quantile(x, (.025, .975))]
        return {"models": {name: {"clean_exact_match_ci95": interval(clean[:, a]),
            "corrupted_exact_match_ci95": interval(corrupted[:, a])} for a, name in enumerate(MODEL_NAMES)},
            "two_minus_single": {"clean_delta_pp_ci95": interval((clean[:, 1] - clean[:, 0]) * 100),
                "corrupted_delta_pp_ci95": interval((corrupted[:, 1] - corrupted[:, 0]) * 100)}}
    return {"method": "percentile_stratified_paired_record_bootstrap", "unit": "record_not_view",
        "patient_cluster_adjustment": "unavailable_not_claimed", "repeats": repeats, "seed": seed,
        "center_equal": summarize(np.mean(list(draws.values()), axis=0)),
        "centers": {c: summarize(d) for c, d in draws.items()}}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_stage(output, prepared, queue, config, *, development):
    output.mkdir(parents=True, exist_ok=True)
    rows, truth, predictions, quality = collect(prepared, queue, development=development)
    metrics = paired_metrics(truth, predictions, list(MODEL_NAMES),
                             [c["condition_id"] for c in prepared["conditions"]])
    # Independent point verification using sklearn, not the reporting reducer.
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss
    max_error = 0.0
    for item in metrics["condition_rows"]:
        a = MODEL_NAMES.index(item["model"])
        j = metrics["condition_ids"].index(item["condition"])
        y, p = truth[item["center"]], predictions[item["center"]][a, :, j]
        independent = {"macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "micro_f1": f1_score(y, p, average="micro", zero_division=0),
            "exact_match_accuracy": accuracy_score(y, p), "hamming_loss": hamming_loss(y, p)}
        max_error = max(max_error, *(abs(item[k] - v) for k, v in independent.items()))
    if max_error > 1e-12:
        raise ValueError("paired metric implementation disagrees with sklearn")
    runtime = config["runtime"]
    kwargs = {"repeats": runtime["bootstrap_repeats"], "seed": runtime["bootstrap_seed"]}
    bootstrap = paired_bootstrap(truth, predictions, list(MODEL_NAMES), **kwargs)
    exact = exact_bootstrap(truth, predictions, **kwargs)
    metrics["evidence"] = {"stage": "development" if development else "full",
        "records": len(rows), "protocol_identity": prepared["protocol_identity"],
        "sklearn_point_checks": len(metrics["condition_rows"]) * 4, "max_point_error": max_error,
        "primary_comparison": "two_chain_minus_single_chain", "single_seed_development_only": True,
        "matched_direct_sft_control": False, "heldout_tuning": False,
        "bootstrap_is_not_training_seed_variability": True,
        "zero_shot_reference": "historical_only_generation_equivalence_not_admitted"}
    for name, value in (("metrics", metrics), ("bootstrap", bootstrap), ("exact_bootstrap", exact), ("quality", quality)):
        atomic_json(output / f"{name}.json", value)
    write_csv(output / "per_condition.csv", metrics["condition_rows"])
    write_csv(output / "per_class.csv", metrics["class_rows"])
    overview = []
    for name in MODEL_NAMES:
        for center, values in {"four_center_equal": metrics["models"][name]["center_equal"],
                               **metrics["models"][name]["centers"]}.items():
            for view in ("clean", "corruption_view_mean"):
                overview.append({"model": name, "center": center, "view": view, **values[view]})
    write_csv(output / "overview.csv", overview)
    return metrics


def finalize(output, state, prepared, queue, config, sources):
    from util.evaluation.pulse_benchmark import snapshot_sources
    metrics = write_stage(output, prepared, queue, config, development=False)
    atomic_json(output / "cohort.json", prepared["rows"])
    atomic_json(output / "data_audit.json", prepared["audit"])
    atomic_json(output / "training_lineage.json", prepared["training"])
    shutil.copyfile(state / "development_freeze.json", output / "development_freeze.json")
    shutil.copytree(state / "development", output / "development", dirs_exist_ok=True)
    shutil.copyfile(state / "gpu_admission.json", output / "gpu_admission.json")
    snapshot_sources(output, sources)
    task_index, task_seconds = [], 0.0
    with (output / "paired_predictions.jsonl").open("w") as handle:
        for task in queue.tasks:
            payload = queue.validate_completed(task)
            task_seconds += payload["performance"]["seconds"]
            task_index.append({"task_id": task["id"], "path": str(queue.result_path(task)),
                "sha256": sha256_file(queue.result_path(task)), "rows": len(payload["predictions"])})
            for row in payload["predictions"]:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    atomic_json(output / "task_index.json", task_index)
    workers = []
    for launch_path in sorted((state / "workers").glob("*.launch.json")):
        launch = json.loads(launch_path.read_text())
        worker_id = launch["worker_id"]
        item = {"launch": launch}
        for kind in ("exit", "final"):
            path = state / "workers" / f"{worker_id}.{kind}.json"
            if path.exists():
                item[kind] = json.loads(path.read_text())
        workers.append(item)
    train_runtime = {c: {a: json.loads((Path(v["path"]).parent / "runtime.json").read_text())
        for a, v in pair.items()} for c, pair in prepared["training"].items()}
    runtime = {"training": train_runtime, "workers": workers, "committed_task_gpu_hours": task_seconds / 3600,
        "inference_process_wall_gpu_hours": sum(w.get("exit", {}).get("wall_seconds", w.get("final", {}).get("seconds", 0))
            for w in workers if w["launch"]["kind"] == "inference") / 3600,
        "missing_worker_timing": [w["launch"]["worker_id"] for w in workers if not ({"exit", "final"} & set(w))],
        "paired_shared_vision_cost_cannot_be_assigned_independently_to_each_arm": True}
    atomic_json(output / "runtime.json", runtime)
    protocol = {"class_order": list(CLASS_ORDER), "mapping_hash": "555ec85d5b51",
        "mapping_version": "v7_super5_sjr_rgq_review_20260528", "metric_view": "drop_all_zero",
        "k500_ref_excluded": True, "k500_overlap": prepared["k500_overlap"], "conditions": prepared["conditions"],
        "identity": prepared["protocol_identity"], "source_identity": sources, "evidence": metrics["evidence"]}
    result = {"artifact_type": "pulse_benchmark_result", "schema_version": 1, "status": "complete", "stage": "benchmark",
        "protocol": protocol, "records": len(prepared["rows"]), "predictions_per_arm": queue.counts()["expected_predictions"],
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*")
            if p.is_file() and p.name != "benchmark_result.json"}}
    validate_result(result, output / "benchmark_result.json")
    atomic_json(output / "benchmark_result.json", result)
