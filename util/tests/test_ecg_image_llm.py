"""Small CPU contracts for the separate image-LLM development surface."""
from pathlib import Path
import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import pytest
import numpy as np
import torch
import yaml

from boot_scripts.run_experiment import load_experiment_plan
from util.evaluation.ecg_image_data import CENTERS, corrupt_native500_one, derived_seed, select_cohort
from util.evaluation.ecg_image_llm import audit_model, parse_response, validate_config
from util.evaluation.ecg_image_metrics import binary_metrics, paired_bootstrap, paired_metrics
from util.evaluation.ecg_image_comparison import audit_parser, build_arrays, join_shards
from util.evaluation.ecg_image_artifact import validate_result
from util.pn2021_artifact_contract import sha256_file

REPO = Path(__file__).resolve().parents[2]


def fixture_rows():
    return [{"sample_key": f"{center}:{i}", "logical_center": center, "label": [int(i % 2)] * 5}
            for center in CENTERS for i in range(30)]


def test_selection_is_nested_center_balanced_disjoint_and_label_independent():
    rows = fixture_rows()
    small = select_cohort(rows, per_center=3, smoke=False)
    large = select_cohort(rows, per_center=5, smoke=False)
    smoke = select_cohort(rows, per_center=8, smoke=True)
    keys = lambda rs: {r["sample_key"] for r in rs}
    assert keys(small) < keys(large)
    assert not keys(smoke).intersection(keys(large))
    assert [r["logical_center"] for r in small] == list(CENTERS) * 3
    relabeled = [{**r, "label": [1, 0, 0, 0, 0]} for r in reversed(rows)]
    assert keys(select_cohort(relabeled, per_center=3, smoke=False)) == keys(small)


def test_selection_rejects_duplicates_and_unavailable_counts():
    with pytest.raises(ValueError, match="duplicate"):
        select_cohort(fixture_rows() * 2, per_center=2, smoke=False)
    with pytest.raises(ValueError, match="insufficient"):
        select_cohort(fixture_rows(), per_center=29, smoke=False)


@pytest.mark.parametrize("text,labels,valid,strict", [
    ("CD; STTC", ["CD", "STTC"], True, True),
    ("NORM", ["NORM"], True, True),
    ("", [], False, False),
    ("I cannot interpret this.", [], False, False),
    ("The diagnosis is MI.", ["MI"], True, False),
    ("NORMAL", [], False, False),
    ("<think>MI or STTC?</think><answer>NORM</answer>", ["NORM"], True, True),
    ("<think>MI may be present", [], False, False),
])
def test_parse_response_keeps_failure_separate_from_normal(text, labels, valid, strict):
    result = parse_response(text)
    assert result["predicted_labels"] == labels
    assert result["parse_valid"] is valid
    assert result["strict_label_list"] is strict


def test_normal_conflict_is_not_silently_rewritten():
    result = parse_response("NORM; MI")
    assert result["normal_abnormal_conflict"]
    assert result["predicted_labels"] == ["MI", "NORM"]


def test_parser_audit_preserves_frozen_labels_and_exposes_case_reasoning_differences():
    assert audit_parser([{"response": "CD; STTC", "predicted_labels": ["CD", "STTC"]}], pulse=True) == 0
    assert audit_parser([{"response": "cd", "predicted_labels": []}], pulse=False) == 1
    assert audit_parser([{"response": "<think>MI?</think><answer>NORM</answer>", "predicted_labels": ["NORM"]}], pulse=False) == 0
    assert audit_parser([{"response": "<think>MI?</think>NORM", "predicted_labels": ["MI", "NORM"]}], pulse=True) == 1
    with pytest.raises(ValueError, match="recorded backend parser"):
        audit_parser([{"response": "MI", "predicted_labels": ["NORM"]}], pulse=True)


def test_cpu_comparison_parser_does_not_import_torch_before_sqlite():
    code = ("import sys; from util.evaluation.ecg_image_comparison import audit_parser; "
            "assert audit_parser([{'response':'CD','predicted_labels':['CD']}],pulse=True)==0; "
            "assert 'torch' not in sys.modules; import sqlite3; "
            "assert sqlite3.connect(':memory:').execute('SELECT 1').fetchone()==(1,)")
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True, text=True)


def test_native500_shape_and_seed_namespace():
    with pytest.raises(ValueError, match="native-500"):
        corrupt_native500_one(torch.zeros(1, 1000, 12), source_hash="h", condition={}, profile=None, base_seed=1)
    assert derived_seed(1, "a", "h") == derived_seed(1, "a", "h")
    assert derived_seed(1, "a", "h") != derived_seed(1, "b", "h")


def test_image_llm_config_and_dry_run_contract(tmp_path):
    path = REPO / "configs/eval/ecg_image_llama_smoke.yaml"
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    plan = load_experiment_plan(REPO / "configs/experiments/ecg_image_llama_smoke.yaml", run_dir=tmp_path / "not_created")
    description = plan.describe()
    assert not description["loads_data"]
    assert not description["loads_model_or_gpu"]
    assert not plan.run_dir.exists()
    assert plan.entry_arguments == ()
    assert plan.expected_result_type == "ecg_image_evaluation_result"
    assert plan.data_ledger.required_roots == ("pn2021_cache", "split_artifacts")
    config["model"]["backend"] = "arbitrary.module"
    with pytest.raises(ValueError, match="code-owned"):
        validate_config(config)


def test_two_gpu_placement_is_only_a_full_precision_llama_smoke():
    config = yaml.safe_load((REPO / "configs/eval/ecg_image_llama_smoke.yaml").read_text())
    config["model"]["device_strategy"] = "two_gpu_balanced"
    validate_config(config)
    config["model"]["precision"] = "nf4_fp16"
    with pytest.raises(ValueError, match="FP16 Llama"):
        validate_config(config)
    config["model"]["precision"] = "float16"
    config["stage"] = "evaluate"
    with pytest.raises(ValueError, match="FP16 Llama"):
        validate_config(config)


def test_hard_metrics_match_independent_sklearn():
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss

    rng = np.random.default_rng(12)
    truth = rng.integers(0, 2, (41, 5))
    prediction = rng.integers(0, 2, (41, 5))
    result = binary_metrics(truth, prediction)
    assert result["macro_f1"] == pytest.approx(f1_score(truth, prediction, average="macro", zero_division=0))
    assert result["micro_f1"] == pytest.approx(f1_score(truth, prediction, average="micro", zero_division=0))
    assert result["exact_match_accuracy"] == accuracy_score(truth, prediction)
    assert result["hamming_loss"] == hamming_loss(truth, prediction)


def test_paired_metrics_and_bootstrap_keep_models_and_views_paired():
    truth = {center: np.ones((10, 5), dtype=np.uint8) for center in CENTERS}
    prediction = {center: np.zeros((2, 10, 21, 5), dtype=np.uint8) for center in CENTERS}
    for values in prediction.values():
        values[:, :, 0, :] = 1
    result = paired_metrics(truth, prediction, ["first", "identical"], ["clean"] + [f"c{i}" for i in range(20)])
    assert result["models"]["first"]["center_equal"]["clean_minus_corruption_pp"]["macro_f1"] == 100
    bootstrap = paired_bootstrap(truth, prediction, ["first", "identical"], repeats=100)
    assert bootstrap["models"]["first"]["drop_pp_ci95"] == [100, 100]
    assert bootstrap["paired_vs_first_model"]["identical"]["extra_drop_pp_ci95"] == [0, 0]
    assert set(bootstrap["centers"]) == set(CENTERS)
    for center in CENTERS:
        assert bootstrap["centers"][center]["models"]["first"]["drop_pp_ci95"] == [100, 100]
        assert bootstrap["centers"][center]["paired_vs_first_model"]["identical"]["corrupted_delta_pp_ci95"] == [0, 0]


def test_comparison_rejects_unpaired_grids_and_foreign_labels():
    cohort = [{"sample_key": c, "logical_center": c, "label": [1, 0, 0, 0, 0]} for c in CENTERS]
    conditions = ["clean"] + [f"c{i}" for i in range(20)]
    rows = [{"sample_key": c, "condition_id": k, "predicted_labels": ["CD"]} for c in CENTERS for k in conditions]
    truth, predictions = build_arrays(cohort, conditions, {"baseline": rows, "new": rows})
    assert truth[CENTERS[0]].shape == (1, 5)
    assert predictions[CENTERS[0]].shape == (2, 1, 21, 5)
    with pytest.raises(ValueError, match="paired grids"):
        build_arrays(cohort, conditions, {"baseline": rows[:-1]})
    with pytest.raises(ValueError, match="paired grids"):
        build_arrays(cohort, conditions, {"baseline": rows + rows[:1]})
    rows[0] = {**rows[0], "predicted_labels": ["NORMAL"]}
    with pytest.raises(ValueError, match="foreign"):
        build_arrays(cohort, conditions, {"baseline": rows})


def test_shard_validator_and_smoke_promotion_guard(tmp_path):
    cohort = [{"sample_key": c, "logical_center": c, "label_names": ["CD"]} for c in CENTERS]
    cohort_path = tmp_path / "cohort.jsonl"
    cohort_path.write_text("".join(json.dumps(r) + "\n" for r in cohort))
    rows = [{"sample_key": c, "condition_id": "clean", "true_labels": ["CD"],
             "clean_waveform_sha256": "0" * 64, "input_waveform_sha256": "1" * 64} for c in CENTERS]
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload = {"schema_version": 1, "artifact_type": "ecg_image_evaluation_result", "status": "complete",
               "expected_predictions": 4, "completed_predictions": 4,
               "prediction_artifact": {"path": "predictions.jsonl", "sha256": sha256_file(prediction_path)},
               "protocol": {"cohort_sha256": sha256_file(cohort_path), "rank": 0, "world_size": 1,
                            "conditions": [{"condition_id": "clean"}], "mapping_hash": "555ec85d5b51",
                            "metric_view": "drop_all_zero", "k500_ref_excluded": True, "stage": "smoke"}}
    result_path = tmp_path / "evaluation_result.json"
    result_path.write_text(json.dumps(payload))
    validate_result(payload, result_path)
    with pytest.raises(ValueError, match="smoke"):
        join_shards([result_path])
    rows[0]["input_waveform_sha256"] = "not_a_hash"
    prediction_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload["prediction_artifact"]["sha256"] = sha256_file(prediction_path)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_result(payload, result_path)


def test_shared_asset_audit_serializes_replica_cache_updates(tmp_path):
    name = "model-00001-of-00001.safetensors"
    content = b"fixture weight bytes"
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / name).write_bytes(content)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"test": name}}))
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    (metadata / (name + ".metadata")).write_text(f"revision\n{digest}\n0\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: audit_model(tmp_path, "revision", "fixture"), range(12)))
    assert all(result["shards"][0]["sha256"] == digest for result in results)
    assert json.loads((tmp_path / "local_verified_assets.json").read_text())["shards"][0]["sha256"] == digest


def test_join_checks_weight_bytes_but_ignores_cache_verification_method(tmp_path, monkeypatch):
    from util.evaluation import ecg_image_comparison as module

    monkeypatch.setattr(module, "validate_result", lambda *args: None)
    monkeypatch.setattr(module, "verify_run_file_index", lambda *args: [])
    cohort = [{"sample_key": f"s{i}"} for i in range(8)]
    paths = []
    for rank in range(2):
        run = tmp_path / f"r{rank}"
        out = run / "evaluation"
        out.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
        (out / "cohort.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in cohort))
        (out / "predictions.jsonl").write_text(''.join(json.dumps({**r,"condition_id":"clean"})+'\n' for i,r in enumerate(cohort) if i//4 == rank))
        result = {"schema_version": 1, "model": {"assets": {"repo_id": "fixture", "revision": "r", "config_files": {},
                    "shards": [{"name": "weight", "size": 1, "sha256": "a"*64, "verification": str(rank)}]}},
                  "protocol": {"stage": "evaluate", "rank": rank, "world_size": 2,
                               "conditions": [{"condition_id": "clean"}]},
                  "prediction_artifact": {"path": "predictions.jsonl"}, "performance": {}}
        path = out / "evaluation_result.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    assert len(join_shards(paths)[0]) == 8
    result["model"]["assets"]["shards"][0]["sha256"] = "b"*64
    paths[-1].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="mixed model"):
        join_shards(paths)


def test_report_payload_uses_canonical_encodings_and_context(tmp_path, monkeypatch):
    """Payload contract only; the real delivery receipt owns renderer QA."""
    if os.environ.get("ECG_IMAGE_PAYLOAD_TEST_CHILD") != "1":
        # Torch can preload an older system libstdc++ before this environment's
        # SQLite/ICU. Match the CPU-only report process in an isolated interpreter.
        code = ("import sqlite3, pytest; raise SystemExit(pytest.main("
                + repr(["-q", f"{__file__}::test_report_payload_uses_canonical_encodings_and_context",
                        "--basetemp", str(tmp_path / "isolated")]) + "))")
        checked = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                                 env={**os.environ, "ECG_IMAGE_PAYLOAD_TEST_CHILD": "1"})
        assert checked.returncode == 0, checked.stdout + checked.stderr
        return
    from types import SimpleNamespace
    from util.evaluation.ecg_image_comparison import make_overview
    from util.evaluation.ecg_image_report import build_report

    truth = {c: np.ones((3, 5), dtype=np.uint8) for c in CENTERS}
    prediction = {c: np.zeros((2, 3, 21, 5), dtype=np.uint8) for c in CENTERS}
    names = ["PULSE-7B", "ECG-R1-8B-RL-image-only-BF16"]
    conditions = ["clean", *[f"condition-{i}" for i in range(20)]]
    metrics = paired_metrics(truth, prediction, names, conditions)
    bootstrap = paired_bootstrap(truth, prediction, names, repeats=100)
    support = [{"center": c, "n_records": 3, **{k: 3 for k in ("CD", "HYP", "MI", "NORM", "STTC")}}
               for c in CENTERS]

    def fake_delivery(*args, **kwargs):
        (tmp_path / "report.html").write_text("fixture: renderer deliberately mocked")
        return SimpleNamespace(returncode=0, stdout="fixture", stderr="")

    monkeypatch.setattr("util.evaluation.ecg_image_report.subprocess.run", fake_delivery)
    build_report(tmp_path, metrics, bootstrap, make_overview(metrics, bootstrap),
                 [{"model": names[0], "center": CENTERS[0], "n_predictions": 63}], support, {},
                 {"limitations": ["Synthetic payload test only"]})
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    blocks = artifact["manifest"]["blocks"]
    for chart in artifact["manifest"]["charts"]:
        assert "xField" not in chart and "series" not in chart
        assert chart["encodings"]["x"]["field"] == "center"
        assert chart["labels"] == {"values": "all"}
        rows = artifact["snapshot"]["datasets"][chart["dataset"]]
        assert len(rows) == 8 and all(row["n_records"] == 3 for row in rows)
        index = next(i for i, b in enumerate(blocks) if b.get("chartId") == chart["id"])
        assert blocks[index - 1]["type"] == "markdown"
    for table in artifact["manifest"]["tables"]:
        assert table["defaultSort"]["field"] in {c["field"] for c in table["columns"]}


@pytest.mark.skipif(not os.environ.get("ECG_IMAGE_REPORT_VERIFY_ROOT"), reason="explicit completed-report CPU audit opt-in")
@pytest.mark.parametrize("exclude_prior", [False, True], ids=["primary", "prior_exposure_sensitivity"])
def test_completed_report_with_sklearn_and_literal_paired_record_resampling(exclude_prior):
    """Independent actual-artifact check: no production metric/bootstrap helper."""
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_recall_fscore_support

    root = Path(os.environ["ECG_IMAGE_REPORT_VERIFY_ROOT"]).resolve()
    root.relative_to(Path("/home/linbinhao"))
    result = json.loads((root / "comparison_result.json").read_text())
    assert result["status"] == "complete"
    for name, expected_sha in result["files"].items():
        assert sha256_file(root / name) == expected_sha
    if exclude_prior and result["cohort"].get("mode") != "full_pulse":
        pytest.skip("prior-exposure sensitivity is only defined for full cohorts")
    prefix = "sensitivity_" if exclude_prior else ""
    metrics = json.loads((root / f"{prefix}metrics.json").read_text())
    intervals = json.loads((root / f"{prefix}bootstrap.json").read_text())
    cohort = json.loads((root / "cohort.json").read_text())
    models, conditions = result["models"], metrics["condition_ids"]
    classes = ("CD", "HYP", "MI", "NORM", "STTC")
    assert classes == tuple(metrics["class_order"])
    assert conditions == [row["condition_id"] for row in result["cohort"]["conditions"]]
    assert result["cohort"]["mapping_hash"] == "555ec85d5b51"
    assert result["cohort"]["metric_view"] == "drop_all_zero"
    selected = {r["sample_key"]: r for r in cohort}
    assert len(selected) == len(cohort)
    assert all(any(r["label"]) for r in cohort)
    if result["cohort"].get("mode") == "full_pulse":
        provenance = json.loads((root / "provenance.json").read_text())
        r1 = provenance["ECG-R1-8B-RL-image-only-BF16"]["shards"][0]
        split_config = Path(r1["result_path"]).parent.parent / "configs/data/splits.yaml"
        split = yaml.safe_load(split_config.read_text())
        split_root = Path(split["output"]["root_dir"]) / split["pn2021"]["output_subdir"]
        assert split["pn2021"]["required_mapping_hash"] == "555ec85d5b51"
        for center in CENTERS:
            hashes = {r["hash_id"] for r in cohort if r["logical_center"] == center}
            reference = np.load(split_root / center / "k500_hash_ids.npy", allow_pickle=False)
            expected = np.load(split_root / center / "evaluation_drop_all_zero_hash_ids.npy", allow_pickle=False)
            assert len(reference) == len(set(reference)) == 500
            assert hashes.isdisjoint(reference)
            assert hashes == set(expected)
    indexed = {}
    for model in models:
        with (root / f"predictions.{model}.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        indexed[model] = {(r["sample_key"], r["condition_id"]): r for r in rows}
        assert len(indexed[model]) == len(rows) == len(cohort) * 21
        assert set(indexed[model]) == {(r["sample_key"], c) for r in cohort for c in conditions}
        for row in rows:
            source = selected[row["sample_key"]]
            assert row["true_labels"] == source["label_names"]
            assert row["logical_center"] == source["logical_center"]
    if exclude_prior:
        exclusion = json.loads((root / "sensitivity.json").read_text())
        excluded = exclusion["excluded_record_ids"]
        assert len(excluded) == len(set(excluded)) == exclusion["excluded_records"] == 2032
        for source in exclusion["exclusion_sources"]:
            assert sha256_file(Path(source["path"])) == source["sha256"]
            assert json.loads(Path(source["path"]).read_text()) == excluded
        cohort = [r for r in cohort if r["sample_key"] not in set(excluded)]
        assert len(cohort) == exclusion["retained_records"] == 37847
        for center in CENTERS:
            assert sum(r["logical_center"] == center for r in cohort) == exclusion["center_counts"][center]["retained_records"]
    saved = {(r["model"], r["center"], r["condition"]): r for r in metrics["condition_rows"]}
    saved_classes = {(r["model"], r["center"], r["class"], r["metric"]): r for r in metrics["class_rows"]}
    rng = np.random.default_rng(intervals["seed"])
    draws_by_center, point_errors, independent_centers = [], [], {}
    for center in CENTERS:
        samples = [r for r in cohort if r["logical_center"] == center]
        truth = np.asarray([r["label"] for r in samples], dtype=bool)
        for row in samples:
            assert [label for label, bit in zip(classes, row["label"]) if bit] == row["label_names"]
        prediction = np.asarray([[[[label in indexed[m][(r["sample_key"], c)]["predicted_labels"]
                                    for label in classes] for c in conditions] for r in samples] for m in models], dtype=bool)
        for mi, model in enumerate(models):
            point, class_points = [], []
            for ci, condition in enumerate(conditions):
                pred = prediction[mi, :, ci]
                independent = {"macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                               "micro_f1": f1_score(truth, pred, average="micro", zero_division=0),
                               "exact_match_accuracy": accuracy_score(truth, pred),
                               "hamming_loss": hamming_loss(truth, pred)}
                point.append(independent)
                class_points.append(precision_recall_fscore_support(truth, pred, zero_division=0)[:3])
                point_errors.extend(abs(value - saved[(model, center, condition)][key]) for key, value in independent.items())
            scores = metrics["models"][model]["centers"][center]
            for key in point[0]:
                point_errors.append(abs(point[0][key] - scores["clean"][key]))
                point_errors.append(abs(np.mean([p[key] for p in point[1:]]) - scores["corruption_view_mean"][key]))
            independent_centers[(model, center)] = point
            for label_index, label in enumerate(classes):
                for metric_index, metric in enumerate(("precision", "recall", "f1")):
                    saved_class = saved_classes[(model, center, label, metric)]
                    point_errors.append(abs(class_points[0][metric_index][label_index] - saved_class["clean"]))
                    point_errors.append(abs(np.mean([p[metric_index][label_index] for p in class_points[1:]])
                                            - saved_class["corruption_view_mean"]))
        # Same seeded draws, but explicitly repeat records, unlike production's
        # matrix-multiplication weighting. Every model and view stays paired.
        weights = rng.multinomial(len(truth), np.full(len(truth), 1 / len(truth)), size=intervals["repeats"])
        draws = []
        for weights_row in weights:
            indices = np.repeat(np.arange(len(truth)), weights_row)
            y, p = truth[indices], prediction[:, indices]
            tp = (p & y[None, :, None, :]).sum(axis=1)
            denominator = p.sum(axis=1) + y.sum(axis=0)[None, None, :]
            f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
            draws.append(f1.mean(axis=-1))
        draws_by_center.append(np.asarray(draws))
    for model in models:
        aggregate = metrics["models"][model]["center_equal"]
        for key in point[0]:
            clean = np.mean([independent_centers[(model, c)][0][key] for c in CENTERS])
            corrupt = np.mean([np.mean([p[key] for p in independent_centers[(model, c)][1:]]) for c in CENTERS])
            point_errors.extend((abs(clean - aggregate["clean"][key]),
                                 abs(corrupt - aggregate["corruption_view_mean"][key]),
                                 abs(100 * (clean - corrupt) - aggregate["clean_minus_corruption_pp"][key])))
    assert max(point_errors) < 1e-12
    interval_errors = []
    for scope, draws in [(intervals, np.mean(draws_by_center, axis=0)),
                         *[(intervals["centers"][c], d) for c, d in zip(CENTERS, draws_by_center)]]:
        clean, corrupt = draws[:, :, 0], draws[:, :, 1:].mean(axis=-1)
        drop = (clean - corrupt) * 100
        for i, model in enumerate(models):
            for key, values in (("clean_macro_f1_ci95", clean[:, i]), ("corrupted_macro_f1_ci95", corrupt[:, i]),
                                ("drop_pp_ci95", drop[:, i])):
                interval_errors.extend(abs(np.quantile(values, (0.025, 0.975)) - scope["models"][model][key]))
            if i:
                for key, values in (("clean_delta_pp_ci95", 100 * (clean[:, i] - clean[:, 0])),
                                    ("corrupted_delta_pp_ci95", 100 * (corrupt[:, i] - corrupt[:, 0])),
                                    ("extra_drop_pp_ci95", drop[:, i] - drop[:, 0])):
                    interval_errors.extend(abs(np.quantile(values, (0.025, 0.975)) - scope["paired_vs_first_model"][model][key]))
    assert max(interval_errors) < 1e-12
    print(json.dumps({"scope": "prior_exposure_sensitivity" if exclude_prior else "primary",
                      "independent_metric_checks": len(point_errors), "max_point_error": max(point_errors),
                      "bootstrap_repeats": intervals["repeats"], "interval_endpoint_checks": len(interval_errors),
                      "max_interval_error": max(interval_errors)}))
