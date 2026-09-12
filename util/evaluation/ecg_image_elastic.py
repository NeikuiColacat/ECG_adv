"""Finite YAML-managed R1 worker pool, with durable fixed-task scheduling.

The coordinator is CPU-only. Workers reuse the retained R1 backend, corruption
and renderer. Device count changes scheduling only, never task/batch identity.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

import yaml

from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, digest_json, exclusive_lock, fixed_tasks
from util.pn2021_artifact_contract import sha256_file

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/home/linbinhao/ECG_adv_data")
R1_PYTHON = DATA / "envs/ecg_image_r1_eval/bin/python"
HISTORICAL_PROBE = DATA / "runs/ecg_image_llm_10h_20260907/r1_b8_smoke/evaluation/evaluation_result.json"
SOURCE_FILES = (
    "util/evaluation/ecg_image_elastic.py", "util/evaluation/ecg_image_queue.py",
    "util/evaluation/ecg_image_llm.py", "util/evaluation/ecg_image_data.py",
    "util/evaluation/ecg_image_artifact.py", "util/ecg_image_renderer.py",
    "util/evaluation/ecg_image_processor.py",
    "boot_scripts/evaluate_ecg_image.py", "boot_scripts/run_experiment.py",
    "data_preprocess/PN2021_preprocess.py", "data_preprocess/preprocess_primitives.py",
    "util/augmentations/profile.py", "util/augmentations/torch_operators.py", "util/random_seed.py",
)


def validate_config(config: dict) -> None:
    if set(config) != {"schema_version", "stage", "references", "baseline", "cohort", "model", "runtime", "paths", "elastic"}:
        raise ValueError("invalid elastic config keys")
    if config["schema_version"] != 2 or config["stage"] not in {"smoke", "evaluate"}:
        raise ValueError("invalid elastic version or stage")
    model = dict(config["model"])
    processing = model.pop("image_processing", "pil_cpu")
    if processing not in {"pil_cpu", "cpu_uint8_equivalent_cuda_v1"}:
        raise ValueError("unsupported R1 preprocessing implementation")
    if model != {"backend": "ecg_r1_image", "repo_id": "PKUDigitalHealth/ECG-R1-8B-RL",
                 "revision": "f9257759e2e3d6b1864c2e1af4dfaa8e96eb84ea",
                 "directory": str(DATA / "models/ECG-R1-8B-RL_f925775"),
                 "precision": "bfloat16", "prompt_variant": "pulse_exact"}:
        raise ValueError("elastic evaluation requires the frozen image-only BF16 R1")
    if config["runtime"] != {"batch_size": 8, "loader_workers": 1, "max_new_tokens": 64}:
        raise ValueError("elastic batch, loader and generation budget are frozen")
    cohort = config["cohort"]
    if config["stage"] == "evaluate":
        if cohort != {"mode": "full_pulse"}:
            raise ValueError("formal elastic evaluation must use the exact full PULSE cohort")
    elif cohort != {"mode": "pilot_smoke", "per_center": 8}:
        raise ValueError("validation uses only the 32 reserved smoke records")
    elastic = config["elastic"]
    if set(elastic) - {"handoff"} != {"candidate_policy", "max_workers", "records_per_task", "poll_seconds", "idle_seconds",
                       "min_available_ram_gib", "max_load_fraction", "validation_result"}:
        raise ValueError("invalid elastic scheduling contract")
    if "handoff" in elastic:
        if config["stage"] != "evaluate" or processing != "cpu_uint8_equivalent_cuda_v1":
            raise ValueError("handoff is limited to the explicitly optimized full run")
        if set(elastic["handoff"]) != {"source_state", "pause_receipt", "numeric_probe"}:
            raise ValueError("invalid preprocessing handoff contract")
        for path in elastic["handoff"].values():
            Path(path).resolve().relative_to(DATA / "runs/ecg_r1_full_elastic_20260907")
    if elastic["candidate_policy"] != "explicit_environment_allowlist":
        raise ValueError("GPU indices belong in the launch environment, not tracked YAML")
    if type(elastic["max_workers"]) is not int or not 1 <= elastic["max_workers"] <= 4:
        raise ValueError("at most four GPU workers are authorized")
    if elastic["records_per_task"] not in {8, 16, 32}:
        raise ValueError("task size must preserve batch8 boundaries")
    if not 30 <= elastic["poll_seconds"] <= 60 or not 60 <= elastic["idle_seconds"] <= 300:
        raise ValueError("unsafe resource polling or idle qualification")
    if elastic["min_available_ram_gib"] < 48 or not 0.5 <= elastic["max_load_fraction"] <= 0.9:
        raise ValueError("shared host resource limits weakened")
    paths = config["paths"]
    if set(paths) != {"state_root", "working_root", "raw_root", "toolkit_dir"}:
        raise ValueError("invalid elastic paths")
    for key in ("state_root", "raw_root", "toolkit_dir"):
        Path(paths[key]).resolve().relative_to(DATA)
    state = Path(paths["state_root"]).resolve()
    if not state.is_relative_to(DATA / "runs/ecg_r1_full_elastic_20260907/state"):
        raise ValueError("state must belong to this authorized full evaluation")
    if Path(paths["working_root"]).resolve() != Path("/dev/shm/ecg_r1_full_elastic_20260907"):
        raise ValueError("unexpected task RAM directory")
    if config["stage"] == "evaluate":
        if not isinstance(elastic["validation_result"], str):
            raise ValueError("full inference needs a managed validation result")
        Path(elastic["validation_result"]).resolve().relative_to(DATA)
    elif elastic["validation_result"] is not None:
        raise ValueError("smoke validation cannot depend on formal results")


def verify_sources(state: Path) -> dict:
    identity = json.loads((state / "identity.json").read_text())
    for name, expected in identity["implementation_sha256"].items():
        if sha256_file(REPO / name) != expected or sha256_file(state / "source_snapshot" / name) != expected:
            raise ValueError(f"frozen execution source changed: {name}")
    for name, expected in identity.get("config_closure_sha256", {}).items():
        if (sha256_file(Path(identity["config_root"]) / name) != expected
                or sha256_file(state / "config_snapshot" / name) != expected):
            raise ValueError(f"frozen config closure changed: {name}")
    return identity


def gpu_allowlist() -> list[int]:
    raw = os.environ.get("ECG_IMAGE_GPU_ALLOWLIST", "")
    if not raw or any(not value.isdigit() for value in raw.split(",")):
        raise ValueError("ECG_IMAGE_GPU_ALLOWLIST must explicitly list candidate GPU indices")
    indices = [int(value) for value in raw.split(",")]
    if len(set(indices)) != len(indices) or not set(indices) <= set(range(8)):
        raise ValueError("invalid or duplicate candidate GPU indices")
    return indices


def prepare(config: dict, config_path: Path, config_root: Path) -> tuple[Path, dict, TaskQueue]:
    # Importing Torch is allowed in this CPU audit, but CUDA remains invisible.
    from util.evaluation.ecg_image_data import audit_inputs, select_cohort
    from util.evaluation.ecg_image_llm import audit_model
    from util.config_bundle import resolve_yaml_config_closure

    state = Path(config["paths"]["state_root"])
    state.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": sha256_file(config_path),
                "implementation_sha256": {name: sha256_file(REPO / name) for name in SOURCE_FILES},
                "config": config, "config_root": str(config_root),
                "config_closure_sha256": {str(p.relative_to(config_root)): sha256_file(p) for p in
                                           resolve_yaml_config_closure([config_path], config_root=config_root)}}
    with exclusive_lock(state / "prepare.lock"):
        if (state / "prepared.json").exists():
            previous_identity = json.loads((state / "identity.json").read_text())
            if ({k: v for k, v in previous_identity.items() if k != "config_root"}
                    != {k: v for k, v in identity.items() if k != "config_root"}):
                raise ValueError("cannot resume state under a different config/source identity")
            verify_sources(state)
            if json.loads((state / "admission.json").read_text())["candidate_gpus"] != gpu_allowlist():
                raise ValueError("resume cannot silently expand the admitted GPU candidate list")
            prepared = json.loads((state / "prepared.json").read_text())
            queue = TaskQueue(state / "queue", prepared["tasks"])
            import_equivalent_commits(state, prepared, queue, config, identity)
            return state, prepared, queue
        rows, conditions, profile, baseline, audit = audit_inputs(config, config_path, config_root)
        assets = audit_model(Path(config["model"]["directory"]), config["model"]["revision"], config["model"]["repo_id"])
        for name in identity["config_closure_sha256"]:
            target = state / "config_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config_root / name, target)
        for name in SOURCE_FILES:
            target = state / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        for name, expected in audit["common_implementation_sha256"].items():
            if sha256_file(REPO / name) != expected:
                raise ValueError("input code drift during preparation")
        if len(rows) != (39879 if config["stage"] == "evaluate" else 32):
            raise ValueError("unexpected frozen full/smoke cohort size")
        prior = Path(config["baseline"]["run_dir"]) / "cohort.jsonl"
        all_prior = [json.loads(line) for line in prior.read_text().splitlines()]
        exposed = {r["sample_key"] for r in select_cohort(all_prior, per_center=500, smoke=False)}
        exposed.update(r["sample_key"] for r in select_cohort(all_prior, per_center=8, smoke=True))
        atomic_json(state / "prior_exposed_records.json", sorted(exposed))
        cohort_path = state / "cohort.jsonl"
        with cohort_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Protocol hashing deliberately excludes current devices and worker count.
        protocol_identity = digest_json({"model": config["model"], "runtime": config["runtime"],
                                         "cohort_sha256": sha256_file(cohort_path),
                                         "baseline_protocol_sha256": audit["baseline_protocol_sha256"],
                                         "implementation_sha256": identity["implementation_sha256"]})
        tasks = fixed_tasks(rows, records_per_task=config["elastic"]["records_per_task"], batch_size=8,
                            condition_ids=[c["condition_id"] for c in conditions], protocol_identity=protocol_identity)
        prepared = {"rows": rows, "conditions": conditions, "baseline_protocol": baseline,
                    "audit": audit, "model_assets": assets, "tasks": tasks,
                    "protocol_identity": protocol_identity, "cohort_sha256": sha256_file(cohort_path),
                    "historical_r1_reuse": "not_imported_without_fixed_batch_replay; full fresh inference planned"}
        atomic_json(state / "identity.json", identity)
        atomic_json(state / "data_audit.json", audit)
        queue = TaskQueue(state / "queue", tasks)
        atomic_json(state / "admission.json", {"candidate_gpus": gpu_allowlist(), "max_workers": config["elastic"]["max_workers"]})
        atomic_json(state / "control.json", {"paused": False, "max_workers": config["elastic"]["max_workers"],
                                             "drain_gpus": [], "allowed_gpus": gpu_allowlist()})
        import_equivalent_commits(state, prepared, queue, config, identity)
        atomic_json(state / "prepared.json", prepared)  # publish readiness last
        return state, prepared, queue


def import_equivalent_commits(state: Path, prepared: dict, queue: TaskQueue, config: dict, identity: dict) -> None:
    """Explicit immutable handoff after numeric and real-model replay gates.

    Old results are copied with per-task origin hashes, never overwritten or
    silently relabelled as fresh GPU-processor inference. Old worker timing is
    retained so a resumed full report counts its already spent GPU-hours.
    """
    handoff = config["elastic"].get("handoff")
    if handoff is None:
        return
    from util.evaluation.ecg_image_artifact import validate_result
    from util.run_record import verify_run_file_index

    source = Path(handoff["source_state"]).resolve()
    if source == state.resolve():
        raise ValueError("handoff requires a distinct state, preserving the old attempt")
    pause_path, numeric_path = Path(handoff["pause_receipt"]), Path(handoff["numeric_probe"])
    pause, numeric = json.loads(pause_path.read_text()), json.loads(numeric_path.read_text())
    old_identity = json.loads((source / "identity.json").read_text())
    old = json.loads((source / "prepared.json").read_text())
    if (pause["status"] != "safely_drained_for_user_authorized_optimization"
            or Path(pause["preserved_state"]).resolve() != source
            or pause["source_identity_sha256"] != sha256_file(source / "identity.json")
            or live_workers(source)):
        raise ValueError("handoff source is not the audited drained incarnation")
    if (numeric.get("resize_mode") != "cpu_equivalent_fixed_point"
            or not numeric.get("all_float32_inputs_bitwise_equal")
            or not numeric.get("all_bfloat16_inputs_bitwise_equal")
            or numeric.get("helper_sha256") != identity["implementation_sha256"]["util/evaluation/ecg_image_processor.py"]
            or len(numeric["checks"]) != 21 or sum(c["images"] for c in numeric["checks"]) != 168
            or any(c["float32_mismatched_elements"] or c["bfloat16_mismatched_elements"] for c in numeric["checks"])):
        raise ValueError("GPU preprocessing numeric equivalence is not verified")
    strip_processing = lambda model: {k: v for k, v in model.items() if k != "image_processing"}
    if (strip_processing(old_identity["config"]["model"]) != strip_processing(config["model"])
            or old_identity["config"]["runtime"] != config["runtime"]
            or old["cohort_sha256"] != prepared["cohort_sha256"] or old["conditions"] != prepared["conditions"]
            or old["audit"]["baseline_protocol_sha256"] != prepared["audit"]["baseline_protocol_sha256"]):
        raise ValueError("handoff would change model, inputs, batch or scientific protocol")
    for relative, digest in old_identity["implementation_sha256"].items():
        if sha256_file(source / "source_snapshot" / relative) != digest:
            raise ValueError("old source snapshot changed")
        if (relative not in {"util/evaluation/ecg_image_elastic.py", "util/evaluation/ecg_image_llm.py"}
                and identity["implementation_sha256"].get(relative) != digest):
            raise ValueError("handoff changed frozen inputs, renderer, queue or other runtime code")
    # Both complete 672-prediction reserved-smoke results must agree, not scores.
    replay_paths = [Path(old_identity["config"]["elastic"]["validation_result"]),
                    Path(config["elastic"]["validation_result"])]
    replays = []
    for path in replay_paths:
        result = json.loads(path.read_text())
        validate_result(result, path)
        if (result["completed_predictions"] != 672 or result["protocol"]["stage"] != "smoke"
                or json.loads((path.parent.parent / "run_manifest.json").read_text())["status"] != "complete"
                or verify_run_file_index(path.parent.parent)):
            raise ValueError("complete managed replay required before importing results")
        replays.append({(r["sample_key"], r["condition_id"]): r for r in
                       map(json.loads, (path.parent / result["prediction_artifact"]["path"]).read_text().splitlines())})
    if result["protocol"]["implementation_sha256"] != identity["implementation_sha256"]:
        raise ValueError("new replay used different execution code")
    fields = ("clean_waveform_sha256", "input_waveform_sha256", "response", "generated_tokens", "predicted_labels")
    if replays[0].keys() != replays[1].keys() or any(
            replays[0][key][field] != replays[1][key][field] for key in replays[0] for field in fields):
        raise ValueError("old/new 672-prediction replay differs; preserve but do not import old results")
    with exclusive_lock(source / "coordinator.lock"):
        if not json.loads((source / "control.json").read_text())["paused"]:
            raise ValueError("source queue is not paused")
        old_queue = TaskQueue(source / "queue", old["tasks"])
        imported = []
        for old_task, new_task in zip(old["tasks"], prepared["tasks"], strict=True):
            if old_task["batches"] != new_task["batches"] or old_task["condition_ids"] != new_task["condition_ids"]:
                raise ValueError("handoff regrouped fixed batches")
            path = old_queue.result_path(old_task)
            if not path.exists():
                continue
            if pause["task_files_sha256"].get(path.name) != sha256_file(path):
                raise ValueError("preserved task changed after the pause audit")
            payload = old_queue.validate_completed(old_task)
            origin = {"state": str(source), "task_id": old_task["id"], "path": str(path), "sha256": sha256_file(path)}
            candidate = {**payload, "task_id": new_task["id"], "protocol_identity": new_task["protocol_identity"], "origin": origin}
            target = queue.result_path(new_task)
            if target.exists():
                if json.loads(target.read_text()) != candidate:
                    raise ValueError("import destination differs; refusing overwrite")
            else:
                atomic_json(target, candidate)
            imported.append({"target_task_id": new_task["id"], "predictions": len(payload["predictions"]), **origin})
        if len(imported) != pause["completed_tasks"] or sum(t["predictions"] for t in imported) != pause["completed_predictions"]:
            raise ValueError("pause receipt and imported commits disagree")
        (state / "workers").mkdir(exist_ok=True)
        for path in (source / "workers").glob("*.json"):
            target = state / "workers" / path.name
            if target.exists():
                if sha256_file(target) != sha256_file(path):
                    raise ValueError("imported worker accounting changed")
            else:
                shutil.copyfile(path, target)
        receipt = {"status": "passed", "source_state": str(source), "pause_receipt_sha256": sha256_file(pause_path),
                   "numeric_probe_sha256": sha256_file(numeric_path), "replay_result_sha256": [sha256_file(p) for p in replay_paths],
                   "replay_predictions_checked": 672, "imported_predictions": pause["completed_predictions"],
                   "imported_tasks": imported, "note": "old CPU-processor predictions preserved with explicit exact-equivalence provenance"}
        atomic_json(state / "handoff.json", receipt)
        prepared["historical_r1_reuse"] = f"current_{pause['completed_predictions']}_commits_imported_after_exact_processor_and_672_prediction_replay; old_pilot_not_imported"


def resource_snapshot() -> dict:
    def query(args):
        return list(csv.reader(subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                                       text=True, timeout=15).splitlines()))
    devices = query(["--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"])
    processes = query(["--query-compute-apps=gpu_uuid,pid"])
    gpu_processes = {}
    for row in processes:
        if len(row) != 2:
            raise ValueError("unparseable GPU process inventory")
        gpu_processes.setdefault(row[0].strip(), []).append(int(row[1].strip()))
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) / 1024**2
    load = float(Path("/proc/loadavg").read_text().split()[0])
    cpus = len(os.sched_getaffinity(0))
    result = {"time": time.time(), "available_ram_gib": available, "load1": load, "cpu_affinity_count": cpus, "gpus": {}}
    for raw in devices:
        index, device_uuid, name, used, total, utilization = [v.strip() for v in raw]
        result["gpus"][int(index)] = {"uuid": device_uuid, "name": name, "memory_used_mib": int(used),
                                       "memory_total_mib": int(total), "utilization": int(utilization),
                                       "pids": gpu_processes.get(device_uuid, [])}
    return result


def authorized_idle_contexts(gpu_uuid: str) -> set[int]:
    """Explicit run-scoped exception for one verified <=32 MiB CUDA context.

    Never hides processes in telemetry; PID reuse, memory growth, changed
    authorization, other devices and other processes remain fail-closed.
    """
    raw = os.environ.get("PULSE_IDLE_CONTEXT_PERMIT")
    if not raw:
        return set()
    path = Path(raw).resolve()
    path.relative_to(DATA / "runs")
    if path.stat().st_uid != os.getuid() or sha256_file(path) != os.environ.get("PULSE_IDLE_CONTEXT_PERMIT_SHA256"):
        raise ValueError("idle CUDA context permit identity changed")
    permit = json.loads(path.read_text())
    if (set(permit) != {"schema_version", "user_authorized", "gpu_uuids", "process", "max_context_mib"}
            or permit["schema_version"] != 1 or permit["user_authorized"] is not True
            or permit["max_context_mib"] != 32 or len(set(permit["gpu_uuids"])) != 4):
        raise ValueError("invalid bounded idle-context permit")
    if gpu_uuid not in permit["gpu_uuids"]:
        return set()
    identity = permit["process"]
    if set(identity) != {"pid", "uid", "start_ticks", "boot_id"} or type(identity["pid"]) is not int or identity["pid"] < 1:
        raise ValueError("invalid idle-context process identity")
    proc = Path("/proc") / str(identity["pid"])
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if (proc.stat().st_uid != identity["uid"] or fields[19] != identity["start_ticks"]
                or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]):
            return set()
    except (OSError, IndexError):
        return set()
    rows = csv.reader(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=15).splitlines())
    for row in rows:
        if len(row) == 3 and row[0].strip() == gpu_uuid and row[1].strip() == str(identity["pid"]):
            try:
                return {identity["pid"]} if 0 <= int(row[2].strip()) <= 32 else set()
            except ValueError:
                return set()
    return set()


def free_device(device: dict) -> bool:
    allowed = authorized_idle_contexts(device.get("uuid", "")) if device["pids"] else set()
    return (device["name"] == "NVIDIA GeForce RTX 4090" and not (set(device["pids"]) - allowed)
            and device["memory_used_mib"] <= 128 and device["utilization"] <= 5
            and device["memory_total_mib"] - device["memory_used_mib"] >= 23500)


def load_control(state: Path, config: dict) -> dict:
    value = json.loads((state / "control.json").read_text())
    limits = json.loads((state / "admission.json").read_text())
    if (set(value) != {"paused", "max_workers", "drain_gpus", "allowed_gpus"}
            or type(value["paused"]) is not bool or type(value["max_workers"]) is not int
            or not 0 <= value["max_workers"] <= limits["max_workers"]
            or not set(value["allowed_gpus"]) <= set(limits["candidate_gpus"])
            or not set(value["drain_gpus"]) <= set(limits["candidate_gpus"])):
        raise ValueError("control file exceeds authorized scheduling limits")
    return value


def event(state: Path, kind: str, **fields) -> None:
    with (state / "events.jsonl").open("a", buffering=1) as handle:
        handle.write(json.dumps({"time": time.time(), "event": kind, **fields}) + "\n")


def process_identity(pid: int) -> dict | None:
    """Scope adoption/signals to an exact same-user worker process incarnation."""
    try:
        directory = Path(f"/proc/{pid}")
        if directory.stat().st_uid != os.getuid():
            return None
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def owned_worker_alive(record: dict) -> bool:
    identity = record.get("process_identity")
    if not identity or process_identity(record["pid"]) != identity:
        return False
    try:
        arguments = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b"\0")
        return (b"util.evaluation.ecg_image_elastic" in arguments
                and record["worker_id"].encode() in arguments and b"--worker" in arguments)
    except OSError:
        return False


def live_workers(state: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in (state / "workers").glob("*.launch.json")]
    return [record for record in records if owned_worker_alive(record)]


def drain_worker(state: Path, record: dict) -> None:
    """Ask an exact owned worker to finish its current task, without pidfd APIs.

    The coordinator Python on this host lacks os.pidfd_open. A durable request
    also survives coordinator restart and never signals a recycled process id.
    """
    if not owned_worker_alive(record):
        return
    path = state / "workers" / f"{record['worker_id']}.drain.json"
    if not path.exists():
        atomic_json(path, {"worker_id": record["worker_id"], "process_identity": record["process_identity"],
                           "requested_at": time.time(), "mode": "finish_current_task"})


def worker_drain_requested(state: Path, worker_id: str, identity: dict) -> bool:
    path = state / "workers" / f"{worker_id}.drain.json"
    if not path.exists():
        return False
    request = json.loads(path.read_text())
    return (request.get("worker_id") == worker_id and request.get("process_identity") == identity
            and request.get("mode") == "finish_current_task")


def infer_batch(samples: list[dict], conditions: list[dict], config: dict, prepared: dict,
                backend, renderer, profile, pool) -> list[dict]:
    import numpy as np
    import torch
    from util.evaluation.ecg_image_data import corrupt_native500_one, native500_waveform
    from util.evaluation.ecg_image_artifact import parse_response

    raw_root = Path(config["paths"]["raw_root"])
    waveforms = list(pool.map(lambda s: native500_waveform(s, raw_root)[0], samples))
    hashes = [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in waveforms]
    clean = torch.from_numpy(np.stack(waveforms)).to("cuda")
    rows = []
    for condition in conditions:
        views = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=s["hash_id"],
                          condition=condition, profile=profile,
                          base_seed=prepared["baseline_protocol"]["corruption"]["seed_config"]["base_seed"])
                          for i, s in enumerate(samples)])
        view_hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in views.cpu().numpy()]
        rgb = renderer.render(views)
        responses, lengths = backend.generate(rgb, max_new_tokens=64)
        for sample, response, length, clean_hash, view_hash in zip(samples, responses, lengths, hashes, view_hashes, strict=True):
            rows.append({"schema_version": 1, "sample_key": sample["sample_key"],
                         "record_id": sample["record_id"], "hash_id": sample["hash_id"],
                         "logical_center": sample["logical_center"], **condition,
                         "true_labels": sample["label_names"], "response": response,
                         "clean_waveform_sha256": clean_hash, "input_waveform_sha256": view_hash,
                         "generated_tokens": int(length), "hit_token_limit": length >= 64,
                         **parse_response(response)})
        del rgb, views
    return rows


def verify_gpu_replay(state: Path, prepared: dict, config: dict, backend, renderer, profile, pool, worker_id: str) -> None:
    """Match 32 exact-input predictions from the pre-existing batch8 smoke.

These records were reserved for admission, not accuracy-based optimization.
Every new physical GPU/worker must reproduce the frozen input hashes and
responses before it can claim any full task. Differences stop admission.
"""
    from util.evaluation.ecg_image_artifact import validate_result
    historical = json.loads(HISTORICAL_PROBE.read_text())
    validate_result(historical, HISTORICAL_PROBE)
    if (historical["model"]["revision"] != config["model"]["revision"]
            or historical["model"]["prompt"] != backend.prompt
            or historical["model"]["generation"]["batch_size"] != 8
            or historical["model"]["precision"] != "bfloat16"):
        raise ValueError("historical GPU replay protocol mismatch")
    old_cohort = [json.loads(line) for line in (HISTORICAL_PROBE.parent / "cohort.jsonl").read_text().splitlines()]
    samples = old_cohort[:8]
    conditions = historical["protocol"]["conditions"]
    if len(conditions) != 4 or len(samples) != 8:
        raise ValueError("unexpected historical smoke grid")
    path = HISTORICAL_PROBE.parent / historical["prediction_artifact"]["path"]
    old_rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {(r["sample_key"], r["condition_id"]): r for r in old_rows}
    actual = infer_batch(samples, conditions, config, prepared, backend, renderer, profile, pool)
    fields = ("clean_waveform_sha256", "input_waveform_sha256", "response", "generated_tokens", "predicted_labels")
    mismatches = []
    for row in actual:
        old = expected[(row["sample_key"], row["condition_id"])]
        for field in fields:
            if row[field] != old[field]:
                mismatches.append({"sample_key": row["sample_key"], "condition_id": row["condition_id"], "field": field})
    evidence = {"status": "passed" if not mismatches else "failed", "worker_id": worker_id,
                "historical_result": str(HISTORICAL_PROBE), "historical_result_sha256": sha256_file(HISTORICAL_PROBE),
                "checked_predictions": len(actual), "compared_fields": list(fields), "mismatches": mismatches}
    atomic_json(state / "workers" / f"{worker_id}.replay.json", evidence)
    if mismatches:
        raise ValueError("GPU input/response replay failed; no full task may be claimed")


def worker(config_path: Path, config_root: Path, worker_id: str, gpu: int) -> None:
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    state = Path(config["paths"]["state_root"])
    identity = verify_sources(state)
    if sha256_file(config_path) != identity["config_sha256"]:
        raise ValueError("worker config drift")
    prepared = json.loads((state / "prepared.json").read_text())
    queue = TaskQueue(state / "queue", prepared["tasks"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("worker must bind exactly its admitted GPU")
    snapshot = resource_snapshot()
    if gpu not in snapshot["gpus"] or not free_device(snapshot["gpus"][gpu]):
        raise RuntimeError("GPU is no longer free at worker admission")
    start = time.monotonic()
    with exclusive_lock(DATA / "runs/ecg_r1_full_elastic_20260907/gpu_locks" / f"{snapshot['gpus'][gpu]['uuid']}.lock"):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from util.augmentations.profile import load_augmentation_profile
        from util.config_bundle import resolve_config_reference
        from util.ecg_image_renderer import PulseECGTensorRenderer
        from util.evaluation.ecg_image_llm import ECG_R1_ImageBackend

        if torch.__version__ != "2.6.0+cu118" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("R1 requires frozen Torch2.6 CUDA11.8 and exactly one admitted device")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260501)
        torch.backends.cuda.matmul.allow_tf32 = False
        stop = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *args: stop.__setitem__(0, True))
        backend = ECG_R1_ImageBackend(config["model"])
        renderer, render_audit = PulseECGTensorRenderer.from_ecg_image_kit(
            config["paths"]["toolkit_dir"], device="cuda:0", tmp_root=config["paths"]["working_root"])
        profile_path = resolve_config_reference(config["references"]["operators_config"], owner_config_path=config_path,
                                                config_root=config_root, description="operators", must_exist=True)
        profile = load_augmentation_profile(profile_path, config_root=config_root)
        if profile.describe()["config_sha256"] != prepared["baseline_protocol"]["corruption"]["profile"]["config_sha256"]:
            raise ValueError("worker corruption profile drift")
        records = {r["sample_key"]: r for r in prepared["rows"]}
        metadata = {"worker_id": worker_id, "pid": os.getpid(), "gpu": gpu,
                    "gpu_uuid": snapshot["gpus"][gpu]["uuid"], "prompt": backend.prompt,
                    "checkpoint_loading": backend.loading, "renderer": render_audit,
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "torchvision", "transformers", "numpy", "Pillow", "accelerate")}}
        atomic_json(state / "workers" / f"{worker_id}.identity.json", metadata)
        completed = 0
        worker_identity = process_identity(os.getpid())
        with ThreadPoolExecutor(max_workers=1) as pool:
            verify_gpu_replay(state, prepared, config, backend, renderer, profile, pool, worker_id)
            while not stop[0]:
                control = load_control(state, config)
                if (control["paused"] or gpu in control["drain_gpus"] or gpu not in control["allowed_gpus"]
                        or worker_drain_requested(state, worker_id, worker_identity)):
                    stop[0] = True
                    break
                lease = queue.claim(worker_id)
                if lease is None:
                    break
                with lease:
                    task_start = time.monotonic()
                    predictions = []
                    for batch in lease.task["batches"]:
                        # Do not shrink or regroup batches when tasks move between workers.
                        predictions.extend(infer_batch([records[key] for key in batch], prepared["conditions"], config,
                                                       prepared, backend, renderer, profile, pool))
                        atomic_json(state / "workers" / f"{worker_id}.progress.json",
                                    {"status": "running", "time": time.time(), "pid": os.getpid(), "gpu": gpu,
                                     "task_id": lease.task["id"], "task_predictions_in_ram": len(predictions),
                                     "committed_predictions_this_worker": completed})
                    elapsed = time.monotonic() - task_start
                    performance = {"gpu": gpu, "seconds": elapsed, "images_per_second": len(predictions) / elapsed,
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                    lease.commit(predictions, performance)
                    completed += len(predictions)
                    print(json.dumps({"worker": worker_id, "committed": completed, **performance}), flush=True)
        atomic_json(state / "workers" / f"{worker_id}.final.json",
                    {"status": "drained" if stop[0] else "finished", "worker_id": worker_id, "gpu": gpu,
                     "predictions": completed, "total_seconds": time.monotonic() - start,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()})


def finalize(state: Path, prepared: dict, queue: TaskQueue, config: dict, output: Path) -> None:
    identity = verify_sources(state)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    performance, task_index = [], []
    count = 0
    with prediction_path.open("x") as handle:
        for task, payload in zip(queue.tasks, queue.completed_payloads(), strict=True):
            performance.append(payload["performance"])
            path = queue.result_path(task)
            task_index.append({"task_id": task["id"], "path": str(path), "sha256": sha256_file(path),
                               "worker_id": payload["worker_id"], "n_predictions": len(payload["predictions"]),
                               **({"origin": payload["origin"]} if "origin" in payload else {})})
            for row in payload["predictions"]:
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                count += 1
        handle.flush()
        os.fsync(handle.fileno())
    for name in ("cohort.jsonl", "data_audit.json", "identity.json", "prior_exposed_records.json", "events.jsonl", "handoff.json"):
        if (state / name).exists():
            shutil.copyfile(state / name, output / name)
    shutil.copytree(state / "source_snapshot", output / "source_snapshot")
    shutil.copytree(state / "config_snapshot", output / "config_snapshot")
    shutil.copytree(state / "workers", output / "workers")
    atomic_json(output / "task_index.json", task_index)
    workers = [json.loads(p.read_text()) for p in (state / "workers").glob("*.identity.json")]
    if not workers or len({w["prompt"] for w in workers}) != 1 or len({digest_json(w["packages"]) for w in workers}) != 1:
        raise ValueError("workers used different or missing prompts")
    worker_times = [json.loads(p.read_text())["total_seconds"] for p in (state / "workers").glob("*.final.json")]
    active_seconds = sum(p["seconds"] for p in performance)
    result = {"schema_version": 2, "artifact_type": "ecg_image_evaluation_result", "status": "complete",
              "model": {**config["model"], "assets": prepared["model_assets"], "prompt": workers[0]["prompt"],
                        "generation": {"do_sample": False, "max_new_tokens": 64, "batch_size": 8,
                                       "padding_side": "left", "use_cache": True}, "precision_note": "unquantized_bfloat16"},
              "protocol": {"stage": config["stage"], "scheduler": "fixed_task_queue_v1",
                           "cohort_mode": config["cohort"]["mode"], "cohort_sha256": prepared["cohort_sha256"],
                           "baseline_protocol_sha256": prepared["audit"]["baseline_protocol_sha256"],
                           "mapping_hash": "555ec85d5b51", "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                           "metric_view": "drop_all_zero", "k500_ref_excluded": True,
                           "class_order": ["CD", "HYP", "MI", "NORM", "STTC"], "conditions": prepared["conditions"],
                           "source_snapshot": True, "implementation_sha256": identity["implementation_sha256"],
                           "inference_packages": workers[0]["packages"], "protocol_identity": prepared["protocol_identity"],
                           "task_index_sha256": sha256_file(output / "task_index.json"),
                           "evidence_level": "full_development_not_blind_not_strict_ood"},
              "expected_predictions": count, "completed_predictions": count,
              "prediction_artifact": {"path": "predictions.jsonl", "sha256": sha256_file(prediction_path)},
              "performance": {"completed": count, "images_per_second": count / active_seconds,
                              "inference_seconds": active_seconds, "total_seconds": sum(worker_times),
                              "gpu_hours": sum(worker_times) / 3600, "worker_launches": len(workers),
                              "peak_allocated_bytes": max(p["peak_allocated_bytes"] for p in performance)}}
    from util.evaluation.ecg_image_artifact import validate_result
    validate_result(result, output / "evaluation_result.json")
    atomic_json(output / "evaluation_result.json", result)


def run(config_path: Path, config_root: Path, output_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("coordinator must launch with CUDA_VISIBLE_DEVICES empty; it admits workers explicitly")
    state, prepared, queue = prepare(config, config_path, config_root)
    with exclusive_lock(state / "coordinator.lock"):
        for task in queue.tasks:
            if queue.result_path(task).exists():
                queue.validate_completed(task)
        children, idle_since, cooldown = {}, {}, {}
        rate_history = deque(maxlen=31)
        stopping = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
        event(state, "coordinator_start", pid=os.getpid(), output=str(output_dir), counts=queue.counts())
        # No workers are admitted if the independent managed smoke result is absent.
        gate_path = config["elastic"]["validation_result"]
        gate_verified_sha = None
        while True:
            control = load_control(state, config)
            for gpu, record in list(children.items()):
                code = record["process"].poll()
                if code is None:
                    continue
                record["log"].close()
                launch = json.loads((state / "workers" / f"{record['id']}.launch.json").read_text())
                atomic_json(state / "workers" / f"{record['id']}.exit.json",
                            {"worker_id": record["id"], "gpu": gpu, "returncode": code,
                             "process_wall_seconds": time.time() - launch["started_at"], "time": time.time()})
                log_path = Path(record["log"].name)
                if code and log_path.exists():
                    with (state / "workers" / f"{record['id']}.error.log").open("wb") as tail:
                        with log_path.open("rb") as source:
                            source.seek(max(0, source.seek(0, 2) - 65536))
                            shutil.copyfileobj(source, tail)
                event(state, "worker_exit", gpu=gpu, worker_id=record["id"], returncode=code)
                del children[gpu]
                cooldown[gpu] = time.time() + (600 if code else 60)
            live = live_workers(state)  # includes workers orphaned by a coordinator crash
            extra = {r["worker_id"] for r in sorted(live, key=lambda r: r["started_at"])[control["max_workers"]:]}
            for record in live:
                if (stopping[0] or control["paused"] or record["gpu"] in control["drain_gpus"]
                        or record["gpu"] not in control["allowed_gpus"] or record["worker_id"] in extra):
                    drain_worker(state, record)
            counts = queue.counts()
            if counts["completed_tasks"] == counts["tasks"] and not live and not children:
                finalize(state, prepared, queue, config, output_dir)
                event(state, "complete", counts=counts)
                return
            if stopping[0] and not live and not children:
                raise InterruptedError("coordinator drained; durable tasks retained for a new managed retry")
            gate = not gate_path
            if gate_path and Path(gate_path).exists() and not gate_verified_sha:
                from util.evaluation.ecg_image_artifact import validate_result
                from util.run_record import verify_run_file_index
                candidate = json.loads(Path(gate_path).read_text())
                validate_result(candidate, Path(gate_path))
                if (candidate["protocol"]["stage"] != "smoke"
                        or candidate["protocol"]["implementation_sha256"] != verify_sources(state)["implementation_sha256"]):
                    raise ValueError("GPU validation does not match this frozen code")
                manifest_path = Path(gate_path).parent.parent / "run_manifest.json"
                if manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete":
                    if verify_run_file_index(manifest_path.parent):
                        raise ValueError("managed GPU validation file index did not verify")
                    gate_verified_sha = sha256_file(Path(gate_path))
                    atomic_json(state / "validation_gate.json", {"result": gate_path, "sha256": gate_verified_sha})
            gate = gate or bool(gate_verified_sha)
            snapshot = resource_snapshot()
            limits = config["elastic"]
            pressure_ok = (snapshot["available_ram_gib"] >= limits["min_available_ram_gib"]
                           and snapshot["load1"] <= snapshot["cpu_affinity_count"] * limits["max_load_fraction"])
            # Never continue claiming work alongside a newly appeared unrelated GPU process.
            for record in live:
                device = snapshot["gpus"].get(record["gpu"])
                if device and set(device["pids"]) - {record["pid"]}:
                    drain_worker(state, record)
                    event(state, "external_gpu_process_appeared_drain_own_worker", gpu=record["gpu"])
            for gpu in gpu_allowlist():
                if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu]):
                    idle_since.setdefault(gpu, time.time())
                else:
                    idle_since.pop(gpu, None)
            admitted = None
            occupied = {r["gpu"] for r in live} | set(children)
            failures = [json.loads(p.read_text()) for p in (state / "workers").glob("*.exit.json")]
            failed_gpus = {gpu for gpu in gpu_allowlist() if sum(
                r["gpu"] == gpu and r["returncode"] != 0 for r in failures) >= 3}
            if (gate and pressure_ok and not control["paused"] and not stopping[0]
                    and len(occupied) < control["max_workers"]
                    and counts["tasks"] - counts["completed_tasks"] > len(occupied)):
                for gpu in control["allowed_gpus"]:
                    if (gpu in occupied or gpu in failed_gpus or gpu in control["drain_gpus"] or gpu not in idle_since
                            or time.time() < cooldown.get(gpu, 0)
                            or time.time() - idle_since[gpu] < limits["idle_seconds"]):
                        continue
                    verify_sources(state)
                    if not free_device(resource_snapshot()["gpus"][gpu]):
                        continue
                    worker_id = f"gpu{gpu}-{uuid.uuid4().hex[:12]}"
                    ram = Path(config["paths"]["working_root"])
                    (ram / "logs").mkdir(parents=True, exist_ok=True)
                    log = (ram / "logs" / f"{worker_id}.log").open("x")
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(ram),
                                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
                    command = [str(R1_PYTHON), "-m", "util.evaluation.ecg_image_elastic", "--worker",
                               "--config", str(config_path), "--config-root", str(config_root),
                               "--worker-id", worker_id, "--gpu", str(gpu)]
                    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    children[gpu] = {"process": process, "id": worker_id, "log": log}
                    atomic_json(state / "workers" / f"{worker_id}.launch.json",
                                {"worker_id": worker_id, "pid": process.pid, "gpu": gpu,
                                 "process_identity": process_identity(process.pid), "started_at": time.time(),
                                 "command": command, "log": str(log.name)})
                    event(state, "worker_start", gpu=gpu, worker_id=worker_id, pid=process.pid, command=command,
                          gpu_snapshot=snapshot["gpus"][gpu], log=str(log.name))
                    admitted = gpu
                    break  # stagger model loads; one admission per poll
            rate_history.append((time.time(), counts["completed_predictions"]))
            interval = rate_history[-1][0] - rate_history[0][0]
            speed = ((rate_history[-1][1] - rate_history[0][1]) / interval) if interval >= 60 else 0
            status = {"status": "running" if occupied or admitted is not None else
                      "waiting_for_free_gpu" if gate else "waiting_for_validation", "time": time.time(), **counts,
                      "active_gpus": sorted(occupied | ({admitted} if admitted is not None else set())), "resources": snapshot,
                      "admitted_gpu": admitted, "resource_pressure_ok": pressure_ok, "quarantined_gpus": sorted(failed_gpus),
                      "rolling_images_per_second": speed,
                      "eta_seconds": (counts["expected_predictions"] - counts["completed_predictions"]) / speed if speed > 0 else None}
            atomic_json(state / "status.json", status)
            print(json.dumps({k: v for k, v in status.items() if k != "resources"}), flush=True)
            time.sleep(limits["poll_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal worker invoked only by the YAML-managed coordinator")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    arguments = parser.parse_args()
    try:
        worker(arguments.config, arguments.config_root, arguments.worker_id, arguments.gpu)
    except Exception:
        traceback.print_exc()
        raise
