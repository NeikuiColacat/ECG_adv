"""Managed paired adapter benchmark: engineering gate, fixed pilot, full grid.

Training must finish before automatic inference admission, so the independent
training and evaluation schedulers cannot together exceed four GPU jobs. The
pilot is a predeclared subset of the exact full tasks; its predictions are
reused without retuning, regrouping batches or selecting checkpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from util.config_bundle import load_yaml_mapping, resolve_config_reference, resolve_yaml_config_closure
from util.evaluation.ecg_image_elastic import free_device, process_identity, resource_snapshot
from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, digest_json, exclusive_lock
from util.pn2021_artifact_contract import sha256_file
from util.pulse_benchmark_contract import ARMS, center_tasks, validate_config, validate_pair_row, validate_result
from util.pulse_training_contract import CENTERS, CLASS_ORDER
from util.pulse_training_queue import allowed_gpus

REPO = Path(__file__).resolve().parents[2]
PULSE_PYTHON = Path("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python")
SOURCES = ("util/evaluation/pulse_benchmark.py", "util/evaluation/pulse_adapters.py",
           "boot_scripts/evaluate_pulse_adapters.py", "util/pulse_training_contract.py",
           "util/evaluation/pulse_benchmark_metrics.py", "util/pulse_benchmark_contract.py",
           "util/evaluation/ecg_image_data.py", "util/evaluation/ecg_image_queue.py",
           "util/evaluation/ecg_image_artifact.py", "util/evaluation/ecg_image_metrics.py",
           "util/ecg_image_renderer.py", "core/pulse_finetune.py", "core/augmix.py",
           "util/augmentations/profile.py", "util/augmentations/torch_operators.py")


def source_identity():
    return {name: sha256_file(REPO / name) for name in SOURCES}


def verify_sources(expected):
    if source_identity() != expected:
        raise ValueError("paired inference source changed after protocol freeze")


def snapshot_sources(output, sources):
    for name, digest in sources.items():
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
        if sha256_file(destination) != digest:
            raise ValueError("source changed during snapshot")


def profile_and_baseline(config, path, root):
    from util.augmentations.profile import load_augmentation_profile
    baseline_path = Path(config["baseline"]["run_dir"]) / "protocol.json"
    if sha256_file(baseline_path) != config["baseline"]["protocol_sha256"]:
        raise ValueError("prior PULSE input protocol changed")
    baseline = json.loads(baseline_path.read_text())
    profile_path = resolve_config_reference(config["references"]["operators_config"],
        owner_config_path=path, config_root=root, description="operators", must_exist=True)
    return load_augmentation_profile(profile_path, config_root=root), baseline


def claim_gpu(config):
    import torch
    from util.evaluation.ecg_image_elastic import authorized_idle_contexts
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu or torch.cuda.device_count() != 1:
        raise ValueError("paired worker must bind one explicit GPU UUID")
    snapshot = resource_snapshot()
    device = next(d for d in snapshot["gpus"].values() if d["uuid"] == gpu)
    if set(device["pids"]) - {os.getpid(), *authorized_idle_contexts(gpu)} or torch.cuda.mem_get_info()[0] < 22 * 1024**3:
        raise RuntimeError("paired worker refuses an occupied GPU")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return [torch.empty(20 * 1024**3, dtype=torch.uint8, device="cuda")]


def make_renderer(config):
    import torch
    from util.ecg_image_renderer import PulseECGTensorRenderer
    temporary = Path(config["paths"]["temporary_root"])
    temporary.mkdir(parents=True, exist_ok=True)
    return PulseECGTensorRenderer.from_ecg_image_kit(config["paths"]["toolkit_dir"],
        device=torch.device("cuda:0"), tmp_root=str(temporary))


def infer_batch(samples, conditions, config, baseline, backend, renderer, profile, pool, *, verify_author=False):
    import numpy as np
    import torch
    from util.evaluation.ecg_image_artifact import parse_response
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    waveforms = list(pool.map(lambda s: native500_waveform(s, Path(config["paths"]["raw_root"]))[0], samples))
    clean_hashes = [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in waveforms]
    clean = torch.from_numpy(np.stack(waveforms)).to("cuda")
    rows = []
    for condition in conditions:
        views = torch.cat([corrupt_native500_one(clean[i:i + 1], source_hash=s["hash_id"], condition=condition,
            profile=profile, base_seed=baseline["corruption"]["seed_config"]["base_seed"]) for i, s in enumerate(samples)])
        view_hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in views.cpu().numpy()]
        pixels = renderer.preprocess_for_pulse(renderer.render(views)).clone()
        predictions = backend.generate_pair(pixels, verify_author=verify_author)
        for i, sample in enumerate(samples):
            row = {"schema_version": 1, "sample_key": sample["sample_key"], "record_id": sample["record_id"],
                   "hash_id": sample["hash_id"], "logical_center": sample["logical_center"], **condition,
                   "true_labels": sample["label_names"], "clean_waveform_sha256": clean_hashes[i],
                   "input_waveform_sha256": view_hashes[i], "arms": {}}
            for arm in ARMS:
                answer = predictions[arm][i]
                row["arms"][arm] = {**answer, **parse_response(answer["response"])}
            validate_pair_row(row, sample, condition)
            rows.append(row)
        del views, pixels
    return rows


def run_admission(config, path, root, output):
    import torch
    from util.evaluation.pulse_adapters import matched_training_evidence, PairedPulseBackend
    sources = source_identity()
    reservation = claim_gpu(config)
    started = time.time()
    evidence = matched_training_evidence(config["admission_training"], allow_smoke=True)
    backend = PairedPulseBackend(evidence, reservation=reservation)
    renderer, rendering = make_renderer(config)
    profile, baseline = profile_and_baseline(config, path, root)
    rows = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())[:4]
    conditions = baseline["corruption"]["conditions"]
    predictions = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in (0, 2):
            predictions.extend(infer_batch(rows[start:start + 2], conditions, config, baseline, backend,
                                           renderer, profile, pool, verify_author=True))
        # Admit the odd final batch as well, without changing full batching.
        single = infer_batch(rows[:1], conditions[:1], config, baseline, backend, renderer, profile, pool, verify_author=True)
        before = {k: v.clone() for k, v in backend.last_token_ids.items()}
        repeated = infer_batch(rows[:1], conditions[:1], config, baseline, backend, renderer, profile, pool)
    if single != repeated or any(not torch.equal(before[k], backend.last_token_ids[k]) for k in ARMS):
        raise ValueError("paired adapter switch failed deterministic token replay")
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output, sources)
    atomic_json(output / "engineering_predictions.json", predictions)
    atomic_json(output / "rendering.json", rendering)
    protocol = {"class_order": list(CLASS_ORDER), "mapping_hash": "555ec85d5b51", "partition": "k500_engineering_only",
                "source_identity": sources, "prompt": backend.prompt, "generation": backend.generation_kwargs,
                "training_result_sha256": {k: v["sha256"] for k, v in evidence.items()},
                "performance_evidence": False}
    result = {"artifact_type": "pulse_benchmark_result", "schema_version": 1, "stage": "admission", "status": "complete",
              "protocol": protocol, "admission": {"author_equivalence": backend.admissions,
              "repeated_adapter_switch_exact": True}, "runtime": {"seconds": time.time() - started,
              "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"]},
              "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}}
    validate_result(result, output / "benchmark_result.json")
    atomic_json(output / "benchmark_result.json", result)


def prepare(config, path, root, state, sources):
    from util.evaluation.ecg_image_data import audit_inputs
    from util.evaluation.pulse_adapters import matched_training_evidence
    from util.run_record import verify_run_file_index
    identity = {"config": config, "source_identity": sources,
                "config_closure": {str(p.relative_to(root)): sha256_file(p) for p in resolve_yaml_config_closure([path], config_root=root)}}
    if (state / "prepared.json").exists():
        previous = json.loads((state / "identity.json").read_text())
        if previous != identity:
            raise ValueError("paired benchmark state cannot resume under a different identity")
        prepared = json.loads((state / "prepared.json").read_text())
        return prepared, TaskQueue(state / "queue", prepared["tasks"])
    training = {}
    for center in CENTERS:
        evidence = matched_training_evidence(config["training_results"][center])
        for item in evidence.values():
            if item["result"]["protocol"]["center"] != center or verify_run_file_index(item["path"].parent.parent):
                raise ValueError("formal adapter center or managed evidence mismatch")
        training[center] = {arm: {"path": str(item["path"]), "sha256": item["sha256"],
                                 "protocol": item["result"]["protocol"]} for arm, item in evidence.items()}
    audit_config = {**config, "stage": "evaluate"}
    rows, conditions, _, baseline, audit = audit_inputs(audit_config, path, root)
    # Explicitly compare actual adapted identities, not only the split's name.
    overlap = {}
    for center in CENTERS:
        selected = {r["hash_id"] for r in rows if r["logical_center"] == center}
        k500 = json.loads((Path(training[center]["w1"]["path"]).parent / "k500_records.json").read_text())
        overlap[center] = len(selected & {r["hash_id"] for r in k500})
    if any(overlap.values()):
        raise ValueError("adapted K500 identities leaked into paired evaluation")
    protocol_identity = digest_json({"identity": identity, "training": training, "cohort_sha256": audit["baseline_cohort_sha256"]})
    tasks = center_tasks(rows, conditions, protocol_identity)
    prepared = {"rows": rows, "conditions": conditions, "baseline": baseline, "audit": audit,
                "training": training, "protocol_identity": protocol_identity, "tasks": tasks,
                "k500_overlap": overlap}
    snapshot_sources(state, sources)
    atomic_json(state / "identity.json", identity)
    atomic_json(state / "data_audit.json", audit)
    queue = TaskQueue(state / "queue", tasks)
    atomic_json(state / "prepared.json", prepared)
    return prepared, queue


def live_worker(record):
    if process_identity(record["pid"]) != record["process_identity"]:
        return False
    try:
        argv = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b"\0")
        if record.get("kind") == "admission":
            # The managed launcher reexecutes only its delegate, not this PID.
            return [a.decode() for a in argv if a] == record["command"]
        return b"util.evaluation.pulse_benchmark" in argv and record["worker_id"].encode() in argv and b"--worker" in argv
    except OSError:
        return False


def worker(config_path, root, state, worker_id):
    import torch
    from util.evaluation.pulse_adapters import matched_training_evidence, PairedPulseBackend
    config = load_yaml_mapping(config_path, description="paired benchmark")
    validate_config(config)
    identity = json.loads((state / "identity.json").read_text())
    verify_sources(identity["source_identity"])
    prepared = json.loads((state / "prepared.json").read_text())
    queue = TaskQueue(state / "queue", prepared["tasks"])
    by_key = {r["sample_key"]: r for r in prepared["rows"]}
    reservation = claim_gpu(config)
    started, backend, current_center = time.time(), None, None
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.__setitem__(0, True))
    profile, baseline = profile_and_baseline(config, config_path, root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        while not stop[0]:
            control = json.loads((state / "control.json").read_text())
            if control["paused"] or worker_id in control["drain_workers"]:
                break
            eligible = {t["id"] for t in queue.tasks if control["phase"] == "full" or t["development"]}
            preferred = {t["id"] for t in queue.tasks if t["id"] in eligible and t["center"] == current_center}
            lease = queue.claim(worker_id, allowed_task_ids=preferred) if preferred else None
            lease = lease or queue.claim(worker_id, allowed_task_ids=eligible)
            if lease is None:
                break
            with lease:
                center = lease.task["center"]
                if center != current_center:
                    evidence = matched_training_evidence(config["training_results"][center])
                    if any(evidence[arm]["sha256"] != prepared["training"][center][arm]["sha256"] for arm in ARMS):
                        raise ValueError("worker adapter differs from the frozen task identity")
                    if backend is None:
                        backend = PairedPulseBackend(evidence, reservation=reservation)
                        renderer, rendering = make_renderer(config)
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        backend.set_pair(evidence)
                    current_center = center
                task_started, predictions = time.time(), []
                for keys in lease.task["batches"]:
                    predictions.extend(infer_batch([by_key[k] for k in keys], prepared["conditions"], config,
                        baseline, backend, renderer, profile, pool))
                performance = {"seconds": time.time() - task_started,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "image_conditions": len(predictions), "paired_predictions": len(predictions) * 2}
                lease.commit(predictions, performance)
                atomic_json(state / "workers" / f"{worker_id}.progress.json", {"pid": os.getpid(),
                    "center": center, "task_id": lease.task["id"], "performance": performance, "time": time.time()})
    atomic_json(state / "workers" / f"{worker_id}.final.json", {"seconds": time.time() - started,
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "prompt": backend.prompt if backend else None})


def run_coordinator(config, path, root, output):
    from boot_scripts.run_experiment import load_experiment_plan
    from util.evaluation.pulse_benchmark_metrics import write_stage, finalize
    from util.run_record import verify_run_file_index
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("paired coordinator must keep CUDA invisible")
    sources, candidates = source_identity(), allowed_gpus()
    state = Path(config["paths"]["state_root"])
    state.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    admission_config = resolve_config_reference(config["references"]["admission_config"],
        owner_config_path=path, config_root=root, description="GPU admission", must_exist=True)
    admission_plan = load_experiment_plan(admission_config, config_root=root, allow_existing_run_dir=True)
    admission_path = admission_plan.delegate_output_dir / "benchmark_result.json"
    children, idle_since, failures = {}, {}, {}
    for receipt in (state / "workers").glob("*.exit.json"):
        previous = json.loads(receipt.read_text())
        if previous["returncode"]:
            failures[previous["gpu"]] = failures.get(previous["gpu"], 0) + 1
    prepared, queue, admission_ok = None, None, False
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    with exclusive_lock(state / "coordinator.lock"):
        if not (state / "control.json").exists():
            atomic_json(state / "control.json", {"paused": False, "max_workers": 4, "allowed_gpus": candidates,
                "phase": "development", "drain_workers": []})
        (state / "workers").mkdir(exist_ok=True)
        while True:
            verify_sources(sources)
            for worker_id, child in list(children.items()):
                code = child["process"].poll()
                if code is not None:
                    child["log"].close()
                    atomic_json(state / "workers" / f"{worker_id}.exit.json", {"returncode": code,
                        "wall_seconds": time.time() - child["started"], "gpu": child["gpu"], "time": time.time()})
                    if code:
                        failures[child["gpu"]] = failures.get(child["gpu"], 0) + 1
                    del children[worker_id]
            live = [json.loads(p.read_text()) for p in (state / "workers").glob("*.launch.json")]
            live = [record for record in live if live_worker(record)]
            control = json.loads((state / "control.json").read_text())
            if (set(control) != {"paused", "max_workers", "allowed_gpus", "phase", "drain_workers"}
                    or type(control["paused"]) is not bool or type(control["max_workers"]) is not int
                    or not 0 <= control["max_workers"] <= 4 or not set(control["allowed_gpus"]) <= set(candidates)
                    or control["phase"] not in {"development", "full"}):
                raise ValueError("paired control exceeds frozen resource/stage bounds")
            if stopping[0]:
                control["paused"] = True
                atomic_json(state / "control.json", control)
                raise InterruptedError("coordinator stopped; workers drain after committing their fixed task")
            training_ready = all(Path(raw).exists() and (Path(raw).parent.parent / "run_manifest.json").exists()
                and json.loads((Path(raw).parent.parent / "run_manifest.json").read_text()).get("status") == "complete"
                for pair in config["training_results"].values() for raw in pair.values())
            # No concurrent automatic inference while any required training is unfinished.
            admission_manifest = admission_plan.run_dir / "run_manifest.json"
            admission_complete = (admission_manifest.exists()
                and json.loads(admission_manifest.read_text()).get("status") == "complete")
            if training_ready and not admission_ok and admission_complete:
                result = json.loads(admission_path.read_text())
                validate_result(result, admission_path)
                if result["protocol"]["source_identity"] != sources or verify_run_file_index(admission_plan.run_dir):
                    raise ValueError("GPU engineering admission is stale or not managed")
                admission_ok = True
                atomic_json(state / "gpu_admission.json", {"path": str(admission_path), "sha256": sha256_file(admission_path)})
            if training_ready and admission_ok and prepared is None:
                prepared, queue = prepare(config, path, root, state, sources)
            counts = queue.counts() if queue else {}
            if queue:
                dev_tasks = [t for t in queue.tasks if t["development"]]
                dev_done = all(queue.result_path(t).exists() for t in dev_tasks)
                if control["phase"] == "full":
                    receipt_path = state / "development_freeze.json"
                    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                    if (not dev_done or receipt.get("protocol_identity") != prepared["protocol_identity"]
                            or receipt.get("task_ids") != [t["id"] for t in dev_tasks]
                            or receipt.get("metrics_sha256") != sha256_file(state / "development/metrics.json")):
                        raise ValueError("full stage requires the valid fixed development receipt")
                if dev_done and control["phase"] == "development" and not live and not children:
                    write_stage(state / "development", prepared, queue, config, development=True)
                    atomic_json(state / "development_freeze.json", {"protocol_identity": prepared["protocol_identity"],
                        "metrics_sha256": sha256_file(state / "development/metrics.json"),
                        "policy": "predeclared_full_extension_no_retuning_or_checkpoint_selection",
                        "task_ids": [t["id"] for t in dev_tasks]})
                    control["phase"] = "full"
                    atomic_json(state / "control.json", control)
                if counts.get("completed_tasks") == counts.get("tasks") and not live and not children:
                    finalize(output, state, prepared, queue, config, sources)
                    return
            snapshot = resource_snapshot()
            for gpu in candidates:
                if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu]):
                    idle_since.setdefault(gpu, time.time())
                else:
                    idle_since.pop(gpu, None)
            limits = config["runtime"]
            pressure = (snapshot["available_ram_gib"] >= limits["min_available_ram_gib"]
                and snapshot["load1"] <= snapshot["cpu_affinity_count"] * limits["max_load_fraction"]
                and shutil.disk_usage(state).free >= limits["min_disk_free_gib"] * 1024**3)
            occupied = {r["worker_id"] for r in live} | set(children)
            slots = control["max_workers"] if admission_ok else 1
            if training_ready and not control["paused"] and pressure and len(occupied) < slots:
                for gpu in control["allowed_gpus"]:
                    if (gpu not in idle_since or time.time() - idle_since[gpu] < limits["idle_seconds"]
                            or failures.get(gpu, 0) >= 3):
                        continue
                    device = resource_snapshot()["gpus"][gpu]
                    if not free_device(device):
                        continue
                    worker_id = f"gpu{gpu}-{uuid.uuid4().hex[:10]}"
                    if not admission_ok:
                        if admission_plan.run_dir.exists():
                            raise RuntimeError("preserved failed GPU admission requires a fresh managed run/config")
                        command = [sys.executable, str(REPO / "boot_scripts/run_experiment.py"), "--config",
                                   str(admission_config), "--config-root", str(root)]
                        subprocess.run([*command, "--dry-run"], check=True, capture_output=True)
                    else:
                        command = [str(PULSE_PYTHON), "-m", "util.evaluation.pulse_benchmark", "--worker",
                                   "--config", str(path), "--config-root", str(root), "--state", str(state), "--worker-id", worker_id]
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "HF_HOME": "/home/linbinhao/ECG_adv_data/hf_cache/pulse", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                    log = (state / "workers" / f"{worker_id}.log").open("x")
                    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    stamp = time.time()
                    children[worker_id] = {"process": process, "log": log, "gpu": gpu, "started": stamp}
                    atomic_json(state / "workers" / f"{worker_id}.launch.json", {"worker_id": worker_id,
                        "pid": process.pid, "process_identity": process_identity(process.pid), "gpu": gpu,
                        "command": command, "started": stamp, "kind": "inference" if admission_ok else "admission"})
                    break
            atomic_json(state / "status.json", {"time": time.time(), "training_ready": training_ready,
                "gpu_admission_ready": admission_ok, "phase": control["phase"], "active_workers": sorted(occupied),
                "counts": counts, "resources": snapshot, "failed_launches_by_gpu": failures})
            time.sleep(limits["poll_seconds"])


def run(path, root, output):
    config = load_yaml_mapping(path, description="paired PULSE benchmark")
    validate_config(config)
    if config["stage"] == "admission":
        run_admission(config, path, root, output)
    else:
        run_coordinator(config, path, root, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    worker(args.config, args.config_root, args.state, args.worker_id)
