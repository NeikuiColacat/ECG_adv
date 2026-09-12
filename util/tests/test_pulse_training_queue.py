"""CPU-only scheduling/adoption contracts; never launches a training child."""
from pathlib import Path
import copy
import json
import os

import pytest
import yaml

from boot_scripts.run_experiment import load_experiment_plan
from util.pulse_training_queue import JOB_KEYS, allowed_gpus, job_state, live_training_jobs, load_jobs, validate_config

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/train/pulse_training_queue.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text())


def test_queue_closure_and_exact_paired_jobs(tmp_path):
    payload = config()
    validate_config(payload)
    jobs = load_jobs(payload, CONFIG, REPO / "configs")
    assert tuple(jobs) == JOB_KEYS
    assert len({j["run"] for j in jobs.values()}) == 8
    plan = load_experiment_plan(REPO / "configs/experiments/pulse_training_queue.yaml", run_dir=tmp_path / "output")
    assert len(plan.config_sources) == 22
    assert plan.expected_result_type == "pulse_training_queue_result"
    assert plan.data_ledger.required_roots == ("pn2021_cache", "split_artifacts")
    assert not plan.describe()["loads_model_or_gpu"]
    assert not plan.run_dir.exists()


@pytest.mark.parametrize("change", ["extra_job", "max_workers", "missing_admission", "wrong_order"])
def test_queue_rejects_scope_and_admission_changes(change):
    payload = copy.deepcopy(config())
    if change == "extra_job":
        payload["references"]["w3_ningbo"] = "experiments/pulse_augmix_w2_ningbo.yaml"
    elif change == "max_workers":
        payload["runtime"]["max_workers"] = 8
    elif change == "missing_admission":
        del payload["admission_results"]["resume"]
    else:
        payload["references"] = dict(reversed(list(payload["references"].items())))
    with pytest.raises(ValueError):
        validate_config(payload)


@pytest.mark.parametrize("value", ["", "0,0", "8", "-1"])
def test_gpu_allowlist_rejects_implicit_or_invalid_devices(monkeypatch, value):
    monkeypatch.setenv("PULSE_GPU_ALLOWLIST", value)
    with pytest.raises(ValueError):
        allowed_gpus()


def test_gpu_allowlist_is_explicit(monkeypatch):
    monkeypatch.setenv("PULSE_GPU_ALLOWLIST", "6,5,4,0")
    assert allowed_gpus() == [6, 5, 4, 0]


def test_stale_progress_is_not_live_and_is_not_relaunched(tmp_path):
    job = {"run": tmp_path / "run", "delegate": tmp_path / "run/training"}
    assert job_state(job, live=False) == "pending"
    job["delegate"].mkdir(parents=True)
    (job["delegate"] / "progress.json").write_text(json.dumps({"pid": os.getpid(), "step": 7}))
    assert live_training_jobs({"test": job}) == {}
    assert job_state(job, live=False) == "stopped_incomplete_requires_explicit_resume"
    assert job_state(job, live=True) == "running"
    (job["run"] / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
    assert job_state(job, live=False) == "complete"


def test_cpu_loading_children_and_adopted_jobs_reserve_their_devices():
    from util.pulse_training_queue import unclaimed_training_devices

    devices = {"loading": 0, "adopted": 2, "completed": 1}
    # Loading launchers need not appear in the live training delegates yet.
    assert unclaimed_training_devices(
        [0, 1, 2, 3], devices, live={"adopted": {}}, children={"loading": {}},
    ) == [1, 3]
    # A delegate may be both discovered and owned; it still claims only one GPU.
    assert unclaimed_training_devices(
        [3, 2, 1, 0], devices, live={"loading": {}, "adopted": {}}, children={"loading": {}},
    ) == [3, 1]
    # Historical device assignments must not hold a completed job's GPU forever.
    assert unclaimed_training_devices([0, 1, 2, 3], devices, live={}, children={}) == [0, 1, 2, 3]


@pytest.mark.parametrize("live,children", [({"adopted": {}}, {}), ({}, {"loading": {}})])
def test_active_job_without_device_identity_blocks_new_gpu_assignments(live, children):
    from util.pulse_training_queue import unclaimed_training_devices

    assert unclaimed_training_devices([0, 1], {}, live=live, children=children) == []


@pytest.mark.parametrize("schema,expected_policy,count", [
    (1, "all_eight_matched_last_checkpoints", 8),
    (2, "all_eight_matched_last_checkpoints", 8),
    (3, "all_44_dual_jsd_training_and_evaluation_jobs", 44),
    (4, "width1_width3_four_centers_last_checkpoints_frozen_width2_baselines", 16),
])
def test_queue_summary_uses_validated_schema(monkeypatch, tmp_path, schema, expected_policy, count):
    from util import pulse_training_queue as queue
    from util.founder_width_queue import KEYS as WIDTH_KEYS
    from util.run_record import _result_summary

    keys = queue.DUAL_KEYS if schema == 3 else WIDTH_KEYS if schema == 4 else JOB_KEYS
    assert len(keys) == count
    payload = {"artifact_type": "pulse_training_queue_result", "schema_version": schema,
               "status": "complete", "jobs": dict.fromkeys(keys, {}), "admission": {}}
    path = tmp_path / "queue_result.json"
    seen = []
    # Result validators own artifact checks; isolate only the summary projection.
    monkeypatch.setattr(queue, "validate_queue_result", lambda value, member: seen.append((value, member)))
    summary = _result_summary(payload, "pulse_training_queue_result", result_path=path)
    assert seen == [(payload, path)]
    assert summary == {"identity": payload["jobs"], "selection": {"policy": expected_policy}}

    def reject(value, member):
        raise ValueError("invalid queue evidence")

    monkeypatch.setattr(queue, "validate_queue_result", reject)
    with pytest.raises(ValueError, match="invalid queue evidence"):
        _result_summary(payload, "pulse_training_queue_result", result_path=path)
