"""Committed-record metrics and restart behavior; no GPU or trained models."""
import copy
import json
from types import SimpleNamespace

import pytest

from util.evaluation.pulse_hybrid_development import metrics_from_predictions
from util.pulse_hybrid_workflow import LiveC5Metrics
from util.pulse_training_contract import CLASS_ORDER


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def block(root, center="ningbo", count=2, start=0):
    output = root / center / str(start)
    conditions = [{"condition_id": "clean", "family": "clean"}]
    conditions += [{"condition_id": f"image_{i}", "family": "image"} for i in range(5)]
    samples, rows = [], []
    for index in range(start, start + count):
        labels = [CLASS_ORDER[index % len(CLASS_ORDER)]]
        sample = {"sample_key": f"{center}:{index}", "hash_id": str(index),
                  "label_names": labels, "label": [int(c in labels) for c in CLASS_ORDER]}
        samples.append(sample)
        for condition_index, condition in enumerate(conditions):
            answers = {}
            for arm_index, arm in enumerate(LiveC5Metrics.arms):
                predicted = labels if (index + condition_index + arm_index) % 3 else []
                answers[arm] = {"predicted_labels": predicted, "parse_valid": bool(predicted),
                                "hit_max_new_tokens": condition_index == arm_index}
            rows.append({"sample_key": sample["sample_key"], "true_labels": labels,
                         "condition_id": condition["condition_id"], "arms": answers})
    write(output / "cohort.json", {"samples": samples, "conditions": conditions})
    for offset in range(count):
        write(output / "batches" / f"{offset:04d}.json", rows[offset * 6:(offset + 1) * 6])
    write(output / "progress.json", {"completed_records": count, "total_records": count})
    job = {"plan": SimpleNamespace(entrypoint_name="pulse_hybrid", delegate_output_dir=output),
           "merge_config": {"center": center}}
    return output, job, samples, conditions, rows


def read_live(state):
    return json.loads((state / "live_metrics.json").read_text())


def test_refresh_matches_offline_metrics_with_unequal_center_sizes(tmp_path):
    fixtures = {c: block(tmp_path, c, count=n) for c, n in (("ningbo", 2), ("georgia", 1))}
    jobs = {c: value[1] for c, value in fixtures.items()}
    complete = {c: {"path": str(value[0] / "hybrid_result.json")} for c, value in fixtures.items()}
    tracker = LiveC5Metrics()
    state = tmp_path / "state"
    tracker.refresh(jobs, complete, {}, {}, state)
    observed = read_live(state)
    assert observed["progress"]["completed_records"] == 3
    assert observed["provisional"] is True
    expected = {}
    for center, (_, _, samples, conditions, rows) in fixtures.items():
        expected[center] = metrics_from_predictions(rows, samples, conditions, arms=tracker.arms)
        actual = observed["metrics"]["per_center"][center]
        for entry in expected[center]["per_condition"]:
            result = actual["per_condition_metrics"][entry["arm"]][entry["condition_id"]]
            for name in ("macro_f1", "exact_match", "hamming_loss", "parse_failure_rate", "truncation_rate"):
                assert result[name] == pytest.approx(entry[name], abs=1e-15)
    for arm in tracker.arms:
        for family in ("clean", "image"):
            value = sum(e["families"][arm][family] for e in expected.values()) / 2
            assert observed["metrics"]["current_equal_center_mean"]["arms"][arm][family] == pytest.approx(value)
    # Repeated refresh and a new coordinator must produce the same statistics.
    for reader in (tracker, LiveC5Metrics()):
        reader.refresh(jobs, complete, {}, {}, state)
        refreshed = read_live(state)
        assert refreshed["progress"] == observed["progress"]
        assert refreshed["metrics"] == observed["metrics"]


def test_live_reader_waits_for_committed_progress(tmp_path):
    output, job, *_ = block(tmp_path)
    tracker = LiveC5Metrics()
    state = tmp_path / "state"
    for count in (0, 1, 1, 2):
        write(output / "progress.json", {"completed_records": count, "total_records": 2})
        tracker.refresh({"job": job}, {}, {"job": {}}, {}, state)
        assert read_live(state)["progress"]["completed_records"] == count


def test_missing_batch_waits_live_but_rejects_completed_job(tmp_path):
    output, job, *_ = block(tmp_path, count=1)
    (output / "batches/0000.json").unlink()
    tracker = LiveC5Metrics()
    tracker.refresh({"job": job}, {}, {"job": {}}, {}, tmp_path / "state")
    assert read_live(tmp_path / "state")["progress"]["completed_records"] == 0
    with pytest.raises(ValueError, match="uncommitted live records"):
        tracker.refresh({"job": job}, {"job": {"path": str(output / "hybrid_result.json")}},
                        {}, {}, tmp_path / "state")


@pytest.mark.parametrize("damage", ["missing_view", "duplicate_view", "wrong_truth",
                                    "missing_arm", "unknown_label", "duplicate_label"])
def test_invalid_paired_batch_is_rejected(tmp_path, damage):
    output, job, _, _, rows = block(tmp_path, count=1)
    rows = copy.deepcopy(rows)
    if damage == "missing_view":
        rows.pop()
    elif damage == "duplicate_view":
        rows.append(rows[0])
    elif damage == "wrong_truth":
        rows[0]["true_labels"] = ["MI"]
    elif damage == "missing_arm":
        rows[0]["arms"].pop("single")
    else:
        rows[0]["arms"]["single"]["predicted_labels"] = (
            ["unknown"] if damage == "unknown_label" else ["CD", "CD"])
    write(output / "batches/0000.json", rows)
    with pytest.raises(ValueError):
        LiveC5Metrics().refresh({"job": job}, {}, {"job": {}}, {}, tmp_path / "state")


@pytest.mark.parametrize("count", [-1, 3, True, 1.5])
def test_invalid_progress_is_rejected(tmp_path, count):
    output, job, *_ = block(tmp_path)
    write(output / "progress.json", {"completed_records": count, "total_records": 2})
    with pytest.raises(ValueError, match="invalid live evaluation progress"):
        LiveC5Metrics().refresh({"job": job}, {}, {"job": {}}, {}, tmp_path / "state")


def test_overlapping_blocks_are_rejected(tmp_path):
    _, first, *_ = block(tmp_path / "first")
    _, duplicate, *_ = block(tmp_path / "duplicate")
    with pytest.raises(ValueError, match="overlapping live evaluation blocks"):
        LiveC5Metrics().refresh({"first": first, "duplicate": duplicate}, {},
                               {"first": {}, "duplicate": {}}, {}, tmp_path / "state")
