"""CPU scheduling/recovery contracts. Never allocate CUDA or contact a network."""
from __future__ import annotations

import copy
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest
import yaml

from util.evaluation.ecg_image_data import CENTERS, select_full_cohort
from util.evaluation.ecg_image_elastic import free_device, gpu_allowlist, live_workers, load_control, process_identity, validate_config
from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, fixed_tasks

REPO = Path(__file__).resolve().parents[2]


def test_idle_context_requires_exact_permit_identity_and_small_memory(tmp_path, monkeypatch):
    import util.evaluation.ecg_image_elastic as module
    from util.pn2021_artifact_contract import sha256_file
    monkeypatch.setattr(module, "DATA", tmp_path)
    raw = tmp_path / "runs/permit.json"
    raw.parent.mkdir()
    pid = os.getpid()
    identity = {**process_identity(pid), "uid": os.getuid()}
    permit = {"schema_version":1, "user_authorized":True, "gpu_uuids":["GPU-a","GPU-b","GPU-c","GPU-d"],
              "process":identity, "max_context_mib":32}
    atomic_json(raw, permit)
    monkeypatch.setenv("PULSE_IDLE_CONTEXT_PERMIT", str(raw))
    monkeypatch.setenv("PULSE_IDLE_CONTEXT_PERMIT_SHA256", sha256_file(raw))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: f"GPU-a, {pid}, 17\n")
    device = {"name":"NVIDIA GeForce RTX 4090", "uuid":"GPU-a", "pids":[pid],
              "memory_used_mib":46, "memory_total_mib":24564, "utilization":0}
    assert module.authorized_idle_contexts("GPU-a") == {pid}
    assert free_device(device)
    assert not free_device({**device, "pids":[pid,999999]})
    assert not free_device({**device, "utilization":50})
    assert module.authorized_idle_contexts("GPU-z") == set()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: f"GPU-a, {pid}, 33\n")
    assert not free_device(device)
    permit["process"]["start_ticks"] = "wrong"
    atomic_json(raw, permit)
    with pytest.raises(ValueError, match="permit identity"):
        free_device(device)
    monkeypatch.setenv("PULSE_IDLE_CONTEXT_PERMIT_SHA256", sha256_file(raw))
    assert not free_device(device)
    monkeypatch.delenv("PULSE_IDLE_CONTEXT_PERMIT")
    assert not free_device(device)


def _tasks(count=48):
    return fixed_tasks([{"sample_key": f"s{i}"} for i in range(count)], records_per_task=8,
                       batch_size=8, condition_ids=["clean", "noise"], protocol_identity="frozen")


def _rows(task):
    return [{"sample_key": key, "condition_id": condition} for batch in task["batches"]
            for key in batch for condition in task["condition_ids"]]


def _consume(root, tasks, worker):
    queue = TaskQueue(Path(root), tasks)
    while (lease := queue.claim(worker)) is not None:
        with lease:
            time.sleep(0.002)
            lease.commit(_rows(lease.task), {"test": True})


def _claim_and_wait(root, tasks, ready):
    queue = TaskQueue(Path(root), tasks)
    with queue.claim("crash-me") as lease:
        ready.send(lease.task["id"])
        time.sleep(30)


def test_tasks_preserve_batch_identity_and_tail():
    tasks = _tasks(19)
    assert [len(t["batches"][0]) for t in tasks] == [8, 8, 3]
    assert tasks == _tasks(19)
    assert sum(t["expected_predictions"] for t in tasks) == 38
    assert all("gpu" not in t and "world_size" not in t for t in tasks)


def test_full_cohort_keeps_unequal_centers_and_all_records():
    rows = [{"sample_key": f"{center}-{i}", "logical_center": center}
            for n, center in enumerate(CENTERS) for i in range(n + 1)]
    selected = select_full_cohort(rows)
    assert len(selected) == 10
    assert {r["sample_key"] for r in selected} == {r["sample_key"] for r in rows}
    assert [r["logical_center"] for r in selected[:4]] == list(CENTERS)
    assert selected == select_full_cohort(list(reversed(rows)))
    with pytest.raises(ValueError, match="duplicate"):
        select_full_cohort(rows + rows[:1])


def test_queue_atomic_claim_commit_and_resume(tmp_path):
    tasks = _tasks(16)
    queue = TaskQueue(tmp_path, tasks)
    first, second = queue.claim("one"), queue.claim("two")
    assert first.task["id"] != second.task["id"]
    assert queue.claim("three") is None
    first.commit(_rows(first.task), {})
    with pytest.raises(FileExistsError):
        first.commit(_rows(first.task), {})
    first.close()
    second.close()
    with TaskQueue(tmp_path, tasks).claim("replacement") as replacement:
        assert replacement.task == second.task
        replacement.commit(_rows(replacement.task), {})
    assert queue.counts()["completed_predictions"] == 32
    assert queue.claim("finished") is None
    assert len(list(queue.completed_payloads())) == 2
    with pytest.raises(ValueError, match="expired"):
        first.commit(_rows(first.task), {})


def test_commit_rejects_duplicates_and_content_corruption(tmp_path):
    queue = TaskQueue(tmp_path, _tasks(8))
    with queue.claim("one") as lease:
        rows = _rows(lease.task)
        with pytest.raises(ValueError, match="incomplete"):
            lease.commit(rows[:-1], {})
        with pytest.raises(ValueError, match="duplicate"):
            lease.commit(rows + rows[:1], {})
        lease.commit(rows, {})
    path = queue.result_path(queue.tasks[0])
    payload = json.loads(path.read_text())
    payload["predictions"][0]["extra"] = "changed"
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="content changed"):
        queue.validate_completed(queue.tasks[0])


def test_dynamic_process_pool_produces_exactly_one_accepted_result(tmp_path):
    tasks = _tasks(400)
    TaskQueue(tmp_path, tasks)
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_consume, args=(str(tmp_path), tasks, f"w{i}")) for i in range(4)]
    # Add workers after the first has already started; no repartitioning.
    processes[0].start()
    time.sleep(0.015)
    for process in processes[1:]:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    queue = TaskQueue(tmp_path, tasks)
    assert queue.counts()["completed_predictions"] == 800
    assert len(list(queue.completed_payloads())) == 50


def test_worker_crash_releases_lock_and_retries_only_uncommitted_task(tmp_path):
    context = multiprocessing.get_context("fork")
    tasks = _tasks(8)
    queue = TaskQueue(tmp_path, tasks)
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_claim_and_wait, args=(str(tmp_path), tasks, send))
    child.start()
    try:
        assert receive.poll(5)
        task_id = receive.recv()
        assert queue.claim("cannot-steal") is None
        # This exact PID was created above for this owned crash-recovery test.
        child.terminate()
        child.join(timeout=5)
        assert not child.is_alive()
        with queue.claim("replacement") as lease:
            assert lease.task["id"] == task_id
            lease.commit(_rows(lease.task), {})
        assert queue.counts()["completed_tasks"] == 1
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)


def test_manifest_change_cannot_resume(tmp_path):
    TaskQueue(tmp_path, _tasks())
    with pytest.raises(ValueError, match="manifest changed"):
        TaskQueue(tmp_path, _tasks(16))


def test_only_completely_free_4090_is_admitted():
    device = {"name": "NVIDIA GeForce RTX 4090", "pids": [], "memory_used_mib": 6,
              "memory_total_mib": 24564, "utilization": 0}
    assert free_device(device)
    for change in ({"pids": [123]}, {"memory_used_mib": 4096}, {"utilization": 99}, {"name": "different GPU"}):
        assert not free_device({**device, **change})


def test_authorized_gpu_list_must_be_explicit(monkeypatch):
    monkeypatch.delenv("ECG_IMAGE_GPU_ALLOWLIST", raising=False)
    with pytest.raises(ValueError):
        gpu_allowlist()
    monkeypatch.setenv("ECG_IMAGE_GPU_ALLOWLIST", "1,3,4")
    assert gpu_allowlist() == [1, 3, 4]
    for value in ("0,0", "8", "all", "", "1,-1"):
        monkeypatch.setenv("ECG_IMAGE_GPU_ALLOWLIST", value)
        with pytest.raises(ValueError):
            gpu_allowlist()


def test_elastic_configs_are_bounded_and_frozen():
    for mode in ("validation", "full"):
        config = yaml.safe_load((REPO / f"configs/eval/ecg_image_r1_elastic_{mode}.yaml").read_text())
        validate_config(config)
        for section, key, value in (("elastic", "max_workers", 8), ("runtime", "batch_size", 16),
                                    ("model", "precision", "float16"), ("elastic", "idle_seconds", 0)):
            bad = copy.deepcopy(config)
            bad[section][key] = value
            with pytest.raises(ValueError):
                validate_config(bad)


def test_gpu_processing_config_handoff_is_explicit_and_bounded():
    for name in ("validation", "full"):
        config = yaml.safe_load((REPO / f"configs/eval/ecg_image_r1_elastic_{name}_gpu_exact.yaml").read_text())
        validate_config(config)
        config["model"]["image_processing"] = "unverified_native_cuda"
        with pytest.raises(ValueError, match="preprocessing"):
            validate_config(config)


def test_equivalent_handoff_preserves_commits_and_rejects_drift(tmp_path, monkeypatch):
    from util.evaluation.ecg_image_elastic import import_equivalent_commits
    from util.evaluation.ecg_image_queue import digest_json
    from util.pn2021_artifact_contract import sha256_file
    import util.evaluation.ecg_image_artifact as artifact
    import util.run_record as run_record

    # Artifact schema/file-index verification has its own tests. This fixture
    # isolates the handoff gates, immutable queue commits and origin accounting.
    monkeypatch.setattr(artifact, "validate_result", lambda *args: None)
    monkeypatch.setattr(run_record, "verify_run_file_index", lambda *args: [])
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir(); destination.mkdir(); (source / "workers").mkdir()
    (source / "source_snapshot").mkdir()
    (source / "source_snapshot/frozen.py").write_text("unchanged inputs\n")
    source_hash = sha256_file(source / "source_snapshot/frozen.py")
    helper = "util/evaluation/ecg_image_processor.py"
    current_identity = {"implementation_sha256": {"frozen.py": source_hash, helper: "a" * 64}}
    rows = [{"sample_key": f"s{i}"} for i in range(32)]
    conditions = [f"condition{i}" for i in range(21)]
    old_tasks = fixed_tasks(rows, records_per_task=8, batch_size=8, condition_ids=conditions, protocol_identity="old")
    new_tasks = fixed_tasks(rows, records_per_task=8, batch_size=8, condition_ids=conditions, protocol_identity="new")
    old_queue, new_queue = TaskQueue(source / "queue", old_tasks), TaskQueue(destination / "queue", new_tasks)
    for _ in range(2):
        with old_queue.claim("old-worker") as lease:
            lease.commit(_rows(lease.task), {"seconds": 1})
    replay_rows = [{"sample_key": row["sample_key"], "condition_id": condition, "clean_waveform_sha256": "c",
                    "input_waveform_sha256": "i", "response": "NORM", "generated_tokens": 3,
                    "predicted_labels": ["NORM"]} for row in rows for condition in conditions]
    replay_paths = []
    for name in ("old-replay", "new-replay"):
        directory = tmp_path / name / "evaluation"; directory.mkdir(parents=True)
        atomic_json(directory.parent / "run_manifest.json", {"status": "complete"})
        (directory / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay_rows))
        path = directory / "evaluation_result.json"
        atomic_json(path, {"completed_predictions": 672, "protocol": {"stage": "smoke", **current_identity},
                           "prediction_artifact": {"path": "predictions.jsonl"}})
        replay_paths.append(path)
    old_config = {"model": {"model": "fixed"}, "runtime": {"batch_size": 8},
                  "elastic": {"validation_result": str(replay_paths[0])}}
    old_identity = {"config": old_config, "implementation_sha256": {"frozen.py": source_hash}}
    atomic_json(source / "identity.json", old_identity)
    prepared = {"cohort_sha256": "same", "conditions": conditions, "audit": {"baseline_protocol_sha256": "same"}}
    atomic_json(source / "prepared.json", {**prepared, "tasks": old_tasks})
    atomic_json(source / "control.json", {"paused": True})
    hashes = {p.name: sha256_file(p) for p in (source / "queue/results").glob("*.json")}
    pause_path, numeric_path = tmp_path / "pause.json", tmp_path / "numeric.json"
    atomic_json(pause_path, {"status": "safely_drained_for_user_authorized_optimization", "preserved_state": str(source),
                            "source_identity_sha256": sha256_file(source / "identity.json"),
                            "task_files_sha256": hashes, "completed_tasks": 2, "completed_predictions": 336})
    numeric = {"resize_mode": "cpu_equivalent_fixed_point", "all_float32_inputs_bitwise_equal": True,
               "all_bfloat16_inputs_bitwise_equal": True, "helper_sha256": "a" * 64,
               "checks": [{"images": 8, "float32_mismatched_elements": 0, "bfloat16_mismatched_elements": 0}] * 21}
    atomic_json(numeric_path, numeric)
    config = {**old_config, "elastic": {"validation_result": str(replay_paths[1]), "handoff": {
              "source_state": str(source), "pause_receipt": str(pause_path), "numeric_probe": str(numeric_path)}}}
    prepared = {**prepared, "tasks": new_tasks}
    for _ in range(2):
        import_equivalent_commits(destination, prepared, new_queue, config, current_identity)
        assert new_queue.counts()["completed_predictions"] == 336
    for task in new_tasks[:2]:
        payload = new_queue.validate_completed(task)
        assert payload["origin"]["state"] == str(source)
        assert hashes[Path(payload["origin"]["path"]).name] == payload["origin"]["sha256"]
    atomic_json(numeric_path, {**numeric, "all_bfloat16_inputs_bitwise_equal": False})
    with pytest.raises(ValueError, match="numeric equivalence"):
        import_equivalent_commits(destination, prepared, new_queue, config, current_identity)
    atomic_json(numeric_path, numeric)
    changed = old_queue.result_path(old_tasks[0]); payload = json.loads(changed.read_text())
    payload["predictions"][0]["extra"] = "tampered"
    payload["rows_sha256"] = digest_json(payload["predictions"])
    atomic_json(changed, payload)
    with pytest.raises(ValueError, match="changed after the pause"):
        import_equivalent_commits(destination, prepared, new_queue, config, current_identity)


def test_control_cannot_expand_authorization(tmp_path):
    atomic_json(tmp_path / "admission.json", {"max_workers": 4, "candidate_gpus": [1, 2, 3, 4]})
    control = {"paused": False, "max_workers": 2, "drain_gpus": [], "allowed_gpus": [1, 2]}
    atomic_json(tmp_path / "control.json", control)
    assert load_control(tmp_path, {})["max_workers"] == 2
    atomic_json(tmp_path / "control.json", {**control, "allowed_gpus": [0, 1]})
    with pytest.raises(ValueError, match="authorized"):
        load_control(tmp_path, {})


def test_coordinator_restart_counts_live_orphans(tmp_path, monkeypatch):
    from util.evaluation import ecg_image_elastic as module
    records = [{"worker_id": "old-running", "gpu": 2}, {"worker_id": "old-dead", "gpu": 3}]
    for record in records:
        atomic_json(tmp_path / "workers" / f"{record['worker_id']}.launch.json", record)
    monkeypatch.setattr(module, "owned_worker_alive", lambda record: record["worker_id"] == "old-running")
    assert live_workers(tmp_path) == records[:1]


def test_process_identity_is_incarnation_scoped():
    import os
    identity = process_identity(os.getpid())
    assert identity["pid"] == os.getpid()
    assert identity["start_ticks"].isdigit()
    assert identity["boot_id"]
    assert process_identity(999999999) is None


def test_drain_request_needs_owned_incarnation_and_no_pidfd_api(tmp_path, monkeypatch):
    from util.evaluation import ecg_image_elastic as module
    identity = {"pid": 123, "start_ticks": "456", "boot_id": "test-boot"}
    record = {"worker_id": "gpu2-test", "pid": 123, "process_identity": identity}
    monkeypatch.delattr(module.os, "pidfd_open", raising=False)
    monkeypatch.setattr(module, "owned_worker_alive", lambda value: False)
    module.drain_worker(tmp_path, record)
    assert not module.worker_drain_requested(tmp_path, "gpu2-test", identity)
    monkeypatch.setattr(module, "owned_worker_alive", lambda value: value == record)
    module.drain_worker(tmp_path, record)
    path = tmp_path / "workers/gpu2-test.drain.json"
    frozen = path.read_bytes()
    module.drain_worker(tmp_path, record)
    assert path.read_bytes() == frozen
    assert module.worker_drain_requested(tmp_path, "gpu2-test", identity)
    assert not module.worker_drain_requested(tmp_path, "gpu2-test", {**identity, "start_ticks": "reused-pid"})


def test_unequal_full_center_arrays_keep_equal_center_metric_weight():
    import numpy as np
    from util.evaluation.ecg_image_comparison import build_arrays
    from util.evaluation.ecg_image_metrics import paired_metrics, paired_bootstrap
    conditions = ["clean", *[f"noise{i}" for i in range(20)]]
    cohort = [{"sample_key": f"{center}{i}", "logical_center": center, "label": [1, 0, 0, 0, 0]}
              for n, center in enumerate(CENTERS) for i in range(n + 1)]
    rows = [{"sample_key": record["sample_key"], "condition_id": condition, "predicted_labels": ["CD"]}
            for record in cohort for condition in conditions]
    with pytest.raises(ValueError, match="balanced"):
        build_arrays(cohort, conditions, {"model": rows})
    truth, prediction = build_arrays(cohort, conditions, {"model": rows}, require_balanced=False)
    assert [len(truth[c]) for c in CENTERS] == [1, 2, 3, 4]
    metrics = paired_metrics(truth, prediction, ["model"], conditions)
    assert metrics["models"]["model"]["center_equal"]["clean"]["macro_f1"] == pytest.approx(0.2)
    intervals = paired_bootstrap(truth, prediction, ["model"], repeats=100)
    assert np.allclose(intervals["models"]["model"]["drop_pp_ci95"], [0, 0])


def test_exposure_sensitivity_excludes_whole_records_across_models_and_views():
    import numpy as np
    from util.evaluation.ecg_image_comparison import exclude_prior_records
    cohort = [{"sample_key": f"{c}{i}", "logical_center": c} for c in CENTERS for i in range(3)]
    truth = {c: np.eye(5, dtype=np.uint8)[:3] for c in CENTERS}
    prediction = {c: np.arange(2 * 3 * 21 * 5).reshape(2, 3, 21, 5) for c in CENTERS}
    exposed = [f"{c}1" for c in CENTERS]
    kept_y, kept_p, counts = exclude_prior_records(cohort, truth, prediction, exposed)
    for c in CENTERS:
        assert np.array_equal(kept_y[c], truth[c][[0, 2]])
        assert np.array_equal(kept_p[c], prediction[c][:, [0, 2]])
        assert counts[c] == {"full_records": 3, "excluded_records": 1, "retained_records": 2}
    for invalid in (exposed + exposed[:1], ["foreign"], [r["sample_key"] for r in cohort]):
        with pytest.raises(ValueError):
            exclude_prior_records(cohort, truth, prediction, invalid)


def _check_full_report_fixture(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from util.evaluation import ecg_image_report as report
    from util.evaluation.ecg_image_comparison import make_overview
    from util.evaluation.ecg_image_metrics import CLASS_ORDER, paired_metrics, paired_bootstrap

    # Synthetic fixture only: validate report wiring without publishing scientific results.
    names = ["PULSE-7B", "synthetic-R1"]
    conditions = ["clean", *[f"noise{i}" for i in range(20)]]
    truth = {c: np.eye(5, dtype=np.uint8)[:i + 1] for i, c in enumerate(CENTERS)}
    prediction = {c: np.tile(y[None, :, None, :], (2, 1, 21, 1)) for c, y in truth.items()}
    metrics = paired_metrics(truth, prediction, names, conditions)
    intervals = paired_bootstrap(truth, prediction, names, repeats=100)
    overview = make_overview(metrics, intervals)
    support = [{"center": c, "n_records": len(y), **dict(zip(CLASS_ORDER, map(int, y.sum(0))))} for c, y in truth.items()]
    quality = [{"model": m, "center": c, "n_predictions": len(y) * 21, "parse_failure_rate": 0,
                "non_list_outputs": 0, "common_parser_label_changes": 0, "normal_abnormal_conflicts": 0,
                "token_limit_hits": 0} for m in names for c, y in truth.items()]
    sensitivity = {"excluded_records": 0, "retained_records": 10, "overview": overview,
                   "center_counts": {c: {"full_records": len(y), "excluded_records": 0,
                                          "retained_records": len(y)} for c, y in truth.items()}}
    provenance = {"PULSE-7B": {}, "synthetic-R1": {"shards": [{"scheduler": "fixed_task_queue_v1",
                  "performance": {"worker_launches": 3, "images_per_second": 1.5, "gpu_hours": 2,
                                  "peak_allocated_bytes": 1024 ** 3}}]}}

    def fake_builder(*args, **kwargs):
        (tmp_path / "report.html").write_text("synthetic renderer placeholder, not a validated report")
        return SimpleNamespace(returncode=0, stdout="mocked builder", stderr="")

    monkeypatch.setattr(report.subprocess, "run", fake_builder)
    config = {"limitations": ["synthetic fixture only"]}
    report.build_report(tmp_path, metrics, intervals, overview, quality, support, provenance, config,
                        cohort_mode="full_pulse", sensitivity=sensitivity)
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    narrative = "\n".join(block.get("body", "") for block in artifact["manifest"]["blocks"])
    assert "全量 10 条" in narrative and "四中心各" not in narrative
    assert "3 次模型 worker 接入" in narrative and "3 个分片" not in narrative
    assert "敏感性分析" in narrative and "严格盲测" in narrative
    assert "sensitivity" in artifact["snapshot"]["datasets"]
    assert all("等量抽样" not in chart["subtitle"] for chart in artifact["manifest"]["charts"])
    with pytest.raises(ValueError, match="sensitivity"):
        report.build_report(tmp_path, metrics, intervals, overview, quality, support, provenance, config,
                            cohort_mode="full_pulse")


def test_full_report_uses_real_counts_and_labels_elastic_runtime(tmp_path):
    import ast
    import subprocess
    import sys
    module_source = Path(__file__).read_text()
    tree = ast.parse(module_source)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_check_full_report_fixture")
    helper_source = ast.get_source_segment(module_source, helper)

    # Match the real CPU-only report process. Test collection imports Torch;
    # its libstdc++ preload is incompatible with this host's SQLite/ICU build.
    # Do not alter either environment or weaken actual SQLite query provenance.
    source = ("import json, pytest\nfrom pathlib import Path\n"
              f"CENTERS = {CENTERS!r}\n" + helper_source +
              f"\n_check_full_report_fixture(Path({str(tmp_path)!r}), pytest.MonkeyPatch())\n")
    process = subprocess.run([sys.executable, "-c", source], cwd=REPO, text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.skipif(os.environ.get("ECG_IMAGE_ELASTIC_GPU_LIFECYCLE") != "1",
                    reason="explicit owned managed GPU lifecycle verification only")
def test_owned_gpu_lifecycle_gate():
    """Verify existing managed smoke workers; never launch/kill a GPU process here.

    Only the fixed goal's validation control is changed. Full remains paused
    unless an actual single-worker drain and two-GPU resume are both proved.
    This opt-in audit can wait in the background for free authorized resources.
    """
    from util.evaluation.ecg_image_artifact import validate_result
    from util.evaluation.ecg_image_elastic import resource_snapshot, verify_sources
    from util.evaluation.ecg_image_queue import exclusive_lock
    from util.pn2021_artifact_contract import sha256_file
    from util.run_record import verify_run_file_index

    root = Path("/home/linbinhao/ECG_adv_data/runs/ecg_r1_full_elastic_20260907")
    revision = os.environ.get("ECG_IMAGE_ELASTIC_REVISION", "original")
    assert revision in {"original", "retry1"}, "only declared managed retry variants are allowed"
    suffix = "" if revision == "original" else "_retry1"
    validation_name, full_name = f"validation{suffix}", f"full{suffix}"
    validation, full = root / "state" / validation_name, root / "state" / full_name
    receipt_path = root / f"gpu_lifecycle_validation{suffix}.json"
    read = lambda path: json.loads(path.read_text())
    manifest_path = root / validation_name / "run_manifest.json"
    result_path = root / validation_name / "evaluation/evaluation_result.json"
    prepared = read(validation / "prepared.json")
    assert len(prepared["rows"]) == 32 and len(prepared["tasks"]) == 4
    assert all(task["expected_predictions"] == 168 for task in prepared["tasks"])
    queue = TaskQueue(validation / "queue", prepared["tasks"])

    with exclusive_lock(root / f"gpu_lifecycle_validation{suffix}.lock"):
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt["status"] == "running", "existing terminal lifecycle audit requires explicit review"
        else:
            assert queue.counts()["completed_tasks"] == 0 and not live_workers(validation)
            full_control, validation_control = read(full / "control.json"), read(validation / "control.json")
            assert full_control["paused"] is True and not validation_control["paused"]
            receipt = {"schema_version": 1, "status": "running", "phase": "single_gpu",
                       "started_at": time.time(), "full_control_held": full_control,
                       "validation_control_expected": validation_control, "events": [],
                       "implementation_sha256": verify_sources(validation)["implementation_sha256"],
                       "single_worker_id": None, "single_gpu": None, "overlap_observed": False}
        assert verify_sources(full)["implementation_sha256"] == receipt["implementation_sha256"]

        def save_event(name, **fields):
            receipt["events"].append({"time": time.time(), "event": name, **fields})
            atomic_json(receipt_path, receipt)
            print(json.dumps({"phase": receipt["phase"], "event": name, **fields}), flush=True)

        def set_validation_control(**changes):
            current = read(validation / "control.json")
            assert current == receipt["validation_control_expected"], "validation control changed outside this audit"
            updated = {**current, **changes}
            atomic_json(validation / "control.json", updated)
            receipt["validation_control_expected"] = updated
            save_event("validation_control", control=updated)

        if receipt["phase"] == "single_gpu":
            set_validation_control(max_workers=1)
        receipt["audit_process_identity"] = process_identity(os.getpid())
        save_event("audit_started_or_resumed")
        try:
            while True:
                assert read(full / "control.json") == receipt["full_control_held"], "full hold changed outside this audit"
                manifest = read(manifest_path)
                assert manifest["status"] not in {"failed", "interrupted"}, "managed validation terminated unsuccessfully"
                # Verify a real process incarnation, not a heartbeat/lock file alone.
                events = [json.loads(line) for line in (validation / "events.jsonl").read_text().splitlines()]
                coordinator = next(e for e in reversed(events) if e["event"] == "coordinator_start")
                pid = coordinator["pid"]
                if manifest["status"] != "complete":
                    identity = process_identity(pid)
                    assert identity is not None, "validation coordinator is no longer live"
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                    assert str(root / validation_name / "evaluation") in command
                    assert any(value.endswith("/boot_scripts/evaluate_ecg_image.py") for value in command)
                else:
                    identity = None
                live = live_workers(validation)
                counts = queue.counts()
                if receipt["phase"] == "single_gpu":
                    assert len(live) <= 1
                    progress = [(record, validation / "workers" / f"{record['worker_id']}.progress.json") for record in live]
                    ready = [(record, read(path)) for record, path in progress if path.exists()]
                    if ready:
                        record, _ = ready[0]
                        assert read(validation / "workers" / f"{record['worker_id']}.replay.json")["status"] == "passed"
                        receipt.update(phase="draining", single_worker_id=record["worker_id"], single_gpu=record["gpu"])
                        set_validation_control(paused=True)
                        save_event("single_gpu_progress_observed_drain_requested", worker=record)
                elif receipt["phase"] == "draining" and not live:
                    final = read(validation / "workers" / f"{receipt['single_worker_id']}.final.json")
                    assert final["predictions"] > 0
                    assert 1 <= counts["completed_tasks"] <= 2, "too few remaining tasks to verify a two-GPU resume"
                    receipt["drained_task_hashes"] = {task["id"]: sha256_file(queue.result_path(task))
                                                     for task in queue.tasks if queue.result_path(task).exists()}
                    receipt["phase"] = "waiting_for_two_free_gpus"
                    save_event("single_gpu_drained", final=final, counts=counts)
                elif receipt["phase"] == "waiting_for_two_free_gpus":
                    snapshot = resource_snapshot()
                    candidates = read(validation / "admission.json")["candidate_gpus"]
                    available = [gpu for gpu in candidates if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu])]
                    first = receipt["single_gpu"]
                    preferred = ([first] if first in available else []) + [gpu for gpu in available if gpu != first]
                    if len(preferred) >= 2:
                        receipt["resume_gpus"] = sorted(preferred[:2])
                        receipt["phase"] = "two_gpu_resume"
                        set_validation_control(paused=False, max_workers=2, allowed_gpus=receipt["resume_gpus"])
                        save_event("two_free_gpus_resume_requested", resources=snapshot)
                elif receipt["phase"] == "two_gpu_resume":
                    # An unrelated job can take either device after admission.
                    # Our coordinator drains its own worker; the verification
                    # must follow other authorized free devices, not pin forever.
                    if not live and counts["tasks"] - counts["completed_tasks"] >= 2 and not receipt["overlap_observed"]:
                        snapshot = resource_snapshot()
                        candidates = read(validation / "admission.json")["candidate_gpus"]
                        available = [gpu for gpu in candidates if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu])]
                        if len(available) >= 2 and set(receipt["resume_gpus"]) != set(available[:2]):
                            receipt["resume_gpus"] = sorted(available[:2])
                            set_validation_control(allowed_gpus=receipt["resume_gpus"])
                            save_event("resume_reassigned_to_free_authorized_pair", resources=snapshot)
                    if len({record["gpu"] for record in live}) == 2 and not receipt["overlap_observed"]:
                        assert all(record["worker_id"] != receipt["single_worker_id"] for record in live)
                        receipt["overlap_observed"] = True
                        save_event("two_gpu_worker_overlap_observed", workers=live)
                    if manifest["status"] == "complete":
                        assert receipt["overlap_observed"], "smoke completion alone does not prove dynamic GPU addition"
                        assert counts["completed_predictions"] == 672 and not live
                        payloads = list(queue.completed_payloads())
                        resumed = [p for p in payloads if p["worker_id"] != receipt["single_worker_id"]]
                        assert {p["performance"]["gpu"] for p in resumed} == set(receipt["resume_gpus"])
                        for task_id, digest in receipt["drained_task_hashes"].items():
                            assert sha256_file(validation / "queue/results" / f"{task_id}.json") == digest
                        result = read(result_path)
                        validate_result(result, result_path)
                        assert not verify_run_file_index(manifest_path.parent)
                        assert result["protocol"]["implementation_sha256"] == receipt["implementation_sha256"]
                        for payload in payloads:
                            assert read(validation / "workers" / f"{payload['worker_id']}.replay.json")["status"] == "passed"
                        assert read(full / "control.json") == receipt["full_control_held"]
                        assert verify_sources(full)["implementation_sha256"] == receipt["implementation_sha256"]
                        receipt.update(status="passed", phase="complete", completed_at=time.time(),
                                       validation_result_sha256=sha256_file(result_path),
                                       checked_predictions=672, full_release_authorized=True)
                        save_event("gpu_lifecycle_validation_passed")
                        atomic_json(full / "control.json", {**receipt["full_control_held"], "paused": False})
                        return
                atomic_json(root / f"gpu_lifecycle_status{suffix}.json", {"status": "waiting_or_verifying", "time": time.time(),
                            "phase": receipt["phase"], "validation_coordinator_identity": identity,
                            "audit_process_identity": receipt["audit_process_identity"], "counts": counts,
                            "live_worker_ids": [record["worker_id"] for record in live]})
                time.sleep(10 if live else 30)
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_event("audit_failed_full_remains_held")
            raise
