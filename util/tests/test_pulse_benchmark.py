"""Paired adapter scheduler and metric contracts; no live GPU calls."""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from boot_scripts.run_experiment import load_experiment_plan
from util.evaluation.ecg_image_artifact import parse_response
from util.evaluation.ecg_image_queue import TaskQueue
from util.evaluation.pulse_benchmark_metrics import collect, exact_bootstrap, write_stage
from util.pulse_benchmark_contract import ARMS, MODEL_NAMES, center_tasks, validate_config, validate_pair_row
from util.pulse_training_contract import CENTERS, CLASS_ORDER

REPO = Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((REPO / "configs/eval/pulse_adapter_full.yaml").read_text())


@pytest.mark.parametrize("name", ["admission", "full"])
def test_managed_pair_config_closure(tmp_path, name):
    path = REPO / f"configs/experiments/pulse_adapter_{name}.yaml"
    plan = load_experiment_plan(path, run_dir=tmp_path / name)
    assert plan.entrypoint_name == "evaluate_pulse_adapters"
    assert plan.expected_result_type == "pulse_benchmark_result"
    validate_config(yaml.safe_load((REPO / f"configs/eval/pulse_adapter_{name}.yaml").read_text()))


@pytest.mark.parametrize("key,value", [("batch_size", 4), ("max_new_tokens", 64), ("max_workers", 5),
    ("development_records_per_center", 500), ("bootstrap_seed", 20260910)])
def test_unmatched_or_unbounded_runtime_rejected(key, value):
    payload = config()
    payload["runtime"][key] = value
    with pytest.raises(ValueError, match="frozen"):
        validate_config(payload)


def fixture_rows(count):
    return [{"sample_key": f"{center}/{i}", "logical_center": center, "record_id": str(i),
             "hash_id": f"{center}:{i}", "label_names": [CLASS_ORDER[i % 5]]}
            for center in CENTERS for i in range(count)]


def conditions():
    return [{"condition_id": "clean" if i == 0 else f"c{i}", "condition_index": i,
             "depth": 0 if i == 0 else 2, "operators": [] if i == 0 else ["a", "b"]} for i in range(21)]


def prediction(sample, condition):
    arms = {}
    for arm in ARMS:
        answer = sample["label_names"][0] if arm == "w2" else "no recognized label"
        arms[arm] = {"response": answer, "generated_tokens": 3, "hit_max_new_tokens": False,
                     **parse_response(answer)}
    return {**{k: sample[k] for k in ("sample_key", "logical_center", "record_id", "hash_id")},
        **condition, "true_labels": sample["label_names"], "clean_waveform_sha256": "a" * 64,
        "input_waveform_sha256": "b" * 64, "arms": arms}


def test_development_reuses_unchanged_full_batch_and_tasks(tmp_path):
    rows = fixture_rows(137)
    tasks = center_tasks(rows, conditions(), "frozen")
    assert sum(t["development"] for t in tasks) == 64
    assert sum(len(b) for t in tasks if t["development"] for b in t["batches"]) == 512
    assert tasks[-1]["batches"] == [["georgia/136"]]
    queue = TaskQueue(tmp_path, tasks)
    selected = {tasks[-1]["id"]}
    with queue.claim("test", allowed_task_ids=selected) as lease:
        assert lease.task == tasks[-1]
    assert queue.claim("test", allowed_task_ids=set()) is None
    with pytest.raises(ValueError, match="foreign"):
        queue.claim("test", allowed_task_ids={"wrong"})


@pytest.mark.parametrize("field", ["record_id", "hash_id", "logical_center", "true_labels", "input_waveform_sha256"])
def test_prediction_identity_rejects_drift(field):
    sample, condition = fixture_rows(1)[0], conditions()[0]
    row = prediction(sample, condition)
    validate_pair_row(row, sample, condition)
    row[field] = "changed"
    with pytest.raises(ValueError):
        validate_pair_row(row, sample, condition)


def prediction_variant(kind):
    from util.evaluation.pulse_subset import validate_original_row
    from util.evaluation.pulse_visual_subset import validate_row

    sample, condition = fixture_rows(1)[0], conditions()[0]
    row = prediction(sample, condition)
    def answer(text):
        return {"response": text, "generated_tokens": 3,
                "hit_max_new_tokens": False, **parse_response(text)}
    if kind == "pair":
        row["arms"] = {"w2": answer("MI"), "w1": answer("CD")}
        return row, sample, condition, validate_pair_row, ["MI", "CD"]
    if kind == "original":
        row["answer"] = answer("CD")
        # Extra fields were historically ignored, including an unrelated arms field.
        row["arms"] = {"unused": None}
        return row, sample, condition, validate_original_row, ["CD"]
    row["arms"] = {"three": answer("HYP"), "single": answer("MI"), "clean": answer("CD")}
    return row, sample, condition, validate_row, ["CD", "MI", "HYP"]


@pytest.mark.parametrize("kind", ["pair", "original", "visual"])
def test_prediction_validation_parses_real_answers_once_in_original_order(kind, monkeypatch):
    from util import pulse_benchmark_contract as contract

    row, sample, condition, validate, expected = prediction_variant(kind)
    original = deepcopy((row, sample, condition))
    calls = []

    def parse(text):
        calls.append(text)
        return parse_response(text)

    monkeypatch.setattr(contract, "parse_response", parse)
    validate(row, sample, condition)
    assert calls == expected
    assert (row, sample, condition) == original


@pytest.mark.parametrize("kind", ["pair", "original", "visual"])
@pytest.mark.parametrize("failure,message", [
    ("identity", "paired prediction sample/condition/arm identity mismatch"),
    ("fingerprint", "paired input waveform fingerprint missing"),
    ("parser", "paired response differs from its frozen parser"),
    ("length", "invalid generated-answer length"),
    ("truncation", "missing truncation audit"),
])
def test_prediction_validation_keeps_failure_priority(kind, failure, message):
    row, sample, condition, validate, _ = prediction_variant(kind)
    first = row["answer"] if kind == "original" else row["arms"]["w2" if kind == "pair" else "clean"]
    first["hit_max_new_tokens"] = 1
    if failure in {"identity", "fingerprint", "parser", "length"}:
        first["generated_tokens"] = 0
    if failure in {"identity", "fingerprint", "parser"}:
        first["predicted_labels"] = ["STTC"]
    if failure in {"identity", "fingerprint"}:
        row["input_waveform_sha256"] = ""
    if failure == "identity":
        row["record_id"] = "foreign"
    with pytest.raises(ValueError, match=f"^{message}$"):
        validate(row, sample, condition)


@pytest.mark.parametrize("kind", ["pair", "visual"])
def test_prediction_validation_keeps_arm_check_before_fingerprints(kind):
    row, sample, condition, validate, _ = prediction_variant(kind)
    row["arms"]["foreign"] = row["arms"].pop("w1" if kind == "pair" else "single")
    row["input_waveform_sha256"] = ""
    message = ("paired prediction sample/condition/arm identity mismatch" if kind == "pair"
               else "visual prediction arm or condition-index mismatch")
    with pytest.raises(ValueError, match=f"^{message}$"):
        validate(row, sample, condition)


def test_paired_reducer_and_confidence_intervals_match_known_fixture(tmp_path):
    rows, views = fixture_rows(8), conditions()
    tasks = center_tasks(rows, views, "fixture", development_records=8)
    queue = TaskQueue(tmp_path / "queue", tasks)
    lookup = {r["sample_key"]: r for r in rows}
    for task in tasks:
        with queue.claim("fixture") as lease:
            predictions = [prediction(lookup[key], c) for b in lease.task["batches"] for key in b for c in views]
            lease.commit(predictions, {"seconds": 1.0})
    prepared = {"rows": rows, "conditions": views, "protocol_identity": "fixture"}
    payload = config()
    payload["runtime"]["bootstrap_repeats"] = 100
    scores = write_stage(tmp_path / "metrics", prepared, queue, payload, development=True)
    assert scores["models"][MODEL_NAMES[0]]["center_equal"]["clean"]["macro_f1"] == 0
    assert scores["models"][MODEL_NAMES[1]]["center_equal"]["clean"]["macro_f1"] == 1
    assert scores["evidence"]["max_point_error"] == 0
    exact = json.loads((tmp_path / "metrics/exact_bootstrap.json").read_text())
    assert exact["center_equal"]["two_minus_single"]["clean_delta_pp_ci95"] == [100, 100]
    quality = json.loads((tmp_path / "metrics/quality.json").read_text())
    assert quality["ningbo"]["w1"]["parse_invalid"] == 8 * 21
    assert quality["ningbo"]["w2"]["parse_invalid"] == 0
    assert collect(prepared, queue, development=False)[0] == rows


def test_bootstrap_is_paired_across_identical_arms():
    rng = np.random.default_rng(101)
    truth = {c: rng.integers(0, 2, size=(16, 5)) for c in CENTERS}
    predictions = {c: np.repeat(rng.integers(0, 2, size=(1, 16, 21, 5)), 2, axis=0) for c in CENTERS}
    exact = exact_bootstrap(truth, predictions, repeats=100, seed=51)
    assert exact["center_equal"]["two_minus_single"] == {"clean_delta_pp_ci95": [0, 0], "corrupted_delta_pp_ci95": [0, 0]}


def test_bootstrap_intervals_match_explicit_record_resampling_and_sklearn():
    """Independent slow oracle: expand record draws, never use weighted reducers."""
    from sklearn.metrics import accuracy_score, f1_score
    from util.evaluation.ecg_image_metrics import paired_bootstrap

    fixture_rng = np.random.default_rng(17)
    truth, predictions = {}, {}
    for i, center in enumerate(CENTERS):
        count = i + 3  # Unequal sizes detect accidental record-weighted centers.
        truth[center] = fixture_rng.integers(0, 2, size=(count, 5))
        predictions[center] = fixture_rng.integers(0, 2, size=(2, count, 21, 5))
    kwargs = {"repeats": 100, "seed": 21}
    f1_result = paired_bootstrap(truth, predictions, list(MODEL_NAMES), **kwargs)
    exact_result = exact_bootstrap(truth, predictions, **kwargs)
    rng = np.random.default_rng(kwargs["seed"])
    f1_draws, exact_draws = {}, {}
    for center in CENTERS:
        count = len(truth[center])
        weights = rng.multinomial(count, np.full(count, 1 / count), size=kwargs["repeats"])
        f1_draws[center] = np.empty((kwargs["repeats"], 2, 21))
        exact_draws[center] = np.empty_like(f1_draws[center])
        cache = {}
        for repeat, weight in enumerate(weights):
            key = tuple(weight)
            if key not in cache:
                indexes = np.repeat(np.arange(count), weight)
                y = truth[center][indexes]
                f1_values, exact_values = np.empty((2, 21)), np.empty((2, 21))
                for arm in range(2):
                    for view in range(21):
                        p = predictions[center][arm, indexes, view]
                        f1_values[arm, view] = f1_score(y, p, average="macro", zero_division=0)
                        exact_values[arm, view] = accuracy_score(y, p)
                cache[key] = f1_values, exact_values
            f1_draws[center][repeat], exact_draws[center][repeat] = cache[key]

    def interval(values):
        return np.quantile(values, (0.025, 0.975))

    for center in (*CENTERS, "center_equal"):
        f1 = np.mean(list(f1_draws.values()), axis=0) if center == "center_equal" else f1_draws[center]
        exact = np.mean(list(exact_draws.values()), axis=0) if center == "center_equal" else exact_draws[center]
        f1_actual = f1_result if center == "center_equal" else f1_result["centers"][center]
        exact_actual = exact_result[center] if center == "center_equal" else exact_result["centers"][center]
        clean, corrupted = f1[:, :, 0], f1[:, :, 1:].mean(axis=-1)
        clean_exact, corrupted_exact = exact[:, :, 0], exact[:, :, 1:].mean(axis=-1)
        for arm, name in enumerate(MODEL_NAMES):
            for key, values in (("clean_macro_f1_ci95", clean[:, arm]),
                                ("corrupted_macro_f1_ci95", corrupted[:, arm]),
                                ("drop_pp_ci95", (clean[:, arm] - corrupted[:, arm]) * 100)):
                np.testing.assert_allclose(f1_actual["models"][name][key], interval(values), atol=1e-12, rtol=0)
            for key, values in (("clean_exact_match_ci95", clean_exact[:, arm]),
                                ("corrupted_exact_match_ci95", corrupted_exact[:, arm])):
                np.testing.assert_allclose(exact_actual["models"][name][key], interval(values), atol=1e-12, rtol=0)
        for key, values in (("clean_delta_pp_ci95", (clean[:, 1] - clean[:, 0]) * 100),
                            ("corrupted_delta_pp_ci95", (corrupted[:, 1] - corrupted[:, 0]) * 100),
                            ("extra_drop_pp_ci95", ((clean[:, 1] - corrupted[:, 1]) - (clean[:, 0] - corrupted[:, 0])) * 100)):
            np.testing.assert_allclose(f1_actual["paired_vs_first_model"][MODEL_NAMES[1]][key], interval(values), atol=1e-12, rtol=0)
        for key, values in (("clean_delta_pp_ci95", (clean_exact[:, 1] - clean_exact[:, 0]) * 100),
                            ("corrupted_delta_pp_ci95", (corrupted_exact[:, 1] - corrupted_exact[:, 0]) * 100)):
            np.testing.assert_allclose(exact_actual["two_minus_single"][key], interval(values), atol=1e-12, rtol=0)


def test_cpu_coordinator_waits_without_launching_before_training_complete(tmp_path, monkeypatch):
    from util.evaluation import pulse_benchmark as module
    payload = config()
    payload["paths"]["state_root"] = str(tmp_path / "state")
    payload["training_results"]["ningbo"]["w1"] = str(tmp_path / "not_completed.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module, "allowed_gpus", lambda: [6])
    monkeypatch.setattr(module, "source_identity", lambda: {})
    monkeypatch.setattr(module, "verify_sources", lambda expected: None)
    monkeypatch.setattr(module, "resource_snapshot", lambda: {"gpus": {}, "available_ram_gib": 100,
        "load1": 1, "cpu_affinity_count": 8})
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    def sleep_once(seconds):
        raise RuntimeError("test_poll_complete")
    monkeypatch.setattr(module.time, "sleep", sleep_once)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected GPU launch"))
    with pytest.raises(RuntimeError, match="test_poll_complete"):
        module.run_coordinator(payload, REPO / "configs/eval/pulse_adapter_full.yaml", REPO / "configs", tmp_path / "output")
    status = json.loads((tmp_path / "state/status.json").read_text())
    assert not status["training_ready"] and status["active_workers"] == []


@pytest.mark.skipif(os.environ.get("PULSE_VERIFY_ZERO_SHOT_REFERENCE") != "1",
                    reason="explicit read-only full historical prediction audit")
def test_full_zero_shot_reference_reuse_contract():
    from util.evaluation.ecg_image_comparison import select_pulse, audit_parser
    from util.evaluation.ecg_image_data import audit_inputs, QUESTION
    from util.evaluation.ecg_image_queue import atomic_json
    from util.pn2021_artifact_contract import sha256_file
    from core.pulse_finetune import MODEL_HASHES
    payload = config()
    samples, views, _, baseline, audit = audit_inputs(payload,
        REPO / "configs/eval/pulse_adapter_full.yaml", REPO / "configs")
    directory = Path(payload["baseline"]["run_dir"])
    predictions, evidence = select_pulse(directory, samples, views, payload["baseline"]["protocol_sha256"])
    assert len(samples) == 39879 and len(predictions) == 837459
    assert {v["name"]: v["sha256"] for v in baseline["assets"]["model_shards"]} == MODEL_HASHES
    assert baseline["cohort"]["k500_reference_records_excluded"] is True
    assert baseline["cohort"]["all_zero_labels_dropped"] is True
    stored_parser_changes = audit_parser(predictions, pulse=True)
    strict_current_parser_changes = sum(parse_response(r["response"])["predicted_labels"] != r["predicted_labels"]
                                        for r in predictions)
    # Read historical code only as hash-checked provenance; never import it.
    old_source = REPO / "agent_workspace/pulse_pn2021_probe/run_full.py"
    old_renderer_runner = REPO / "agent_workspace/pulse_pn2021_probe/run_pn2021c_gpu.py"
    for source in (old_source, old_renderer_runner):
        assert sha256_file(source) == baseline["implementation_files"][source.relative_to(REPO).as_posix()]
    assignment = next(node for node in ast.parse(old_source.read_text()).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "QUESTION" for target in node.targets))
    assert ast.literal_eval(assignment.value) == QUESTION
    result = {"status": "passed", "records": len(samples), "predictions": len(predictions),
        "baseline_protocol_sha256": sha256_file(directory / "protocol.json"), "input_audit": audit,
        "prediction_evidence": evidence, "model_shard_identities_match": True, "question_text_exact": True,
        "stored_historical_parser_verified": True, "normalized_common_parser_changes": stored_parser_changes,
        "current_strict_parser_label_changes": strict_current_parser_changes,
        "historical_hard_label_scores_reusable_without_reparsing": strict_current_parser_changes == 0,
        "reuse_policy": "historical_pre_finetuning_reference_only_not_matched_direct_sft",
        "strict_matched_current_inference_baseline": False,
        "unproven_equivalence_axes": ["historical_handoff_regrouped_batches",
            "historical_no_autocast_vs_current_fp32_trainables_with_autocast",
            "historical_loader_default_attention_vs_current_explicit_sdpa"],
        "new_sft_generation_or_performance_verified": False}
    destination = Path(os.environ["PULSE_REFERENCE_AUDIT_OUTPUT"]).resolve()
    destination.relative_to(Path("/home/linbinhao/ECG_adv_data/runs/pulse_augmix_sft_20260909"))
    assert not destination.exists(), "do not overwrite an earlier audit receipt"
    atomic_json(destination, result)
