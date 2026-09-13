"""Finite PULSE training queue; every child still uses the sole YAML launcher.

No model loading, GPU allocation, hyperparameter search or output overwrite in
this coordinator. A process is adopted only by its same-user live /proc identity
and exact training delegate/output arguments, never by a stale progress file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import deque
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.evaluation.ecg_image_elastic import free_device, process_identity, resource_snapshot
from util.evaluation.ecg_image_queue import atomic_json, exclusive_lock
from util.pn2021_artifact_contract import sha256_file
from util.pulse_training_contract import CENTERS, load_config, validate_result

REPO = Path(__file__).resolve().parents[1]
JOB_KEYS = tuple(f"w{width}_{center}" for center in CENTERS for width in (1, 2))
DUAL_SMOKES = tuple(f"pulse_visual_smoke_{arm}" for arm in ("clean", "single", "three", "resume"))
DUAL_PULSE = tuple(f"pulse_visual_{arm}_{center}" for center in CENTERS for arm in ("clean", "single", "three"))
DUAL_PULSE_EVAL = tuple(f"pulse_visual_eval_{center}" for center in CENTERS)
DUAL_FOUNDER = tuple(f"founder_objective_{arm}_{center}{suffix}" for center in CENTERS
                     for arm in ("simclr", "clean_bce", "jsd") for suffix in ("", "_eval"))
DUAL_KEYS = (*DUAL_SMOKES, *DUAL_PULSE, *DUAL_PULSE_EVAL, *DUAL_FOUNDER)


def validate_config(config):
    if set(config) != {"schema_version", "references", "admission_results", "runtime"} or config["schema_version"] != 1:
        raise ValueError("invalid PULSE training queue schema")
    if tuple(config["references"]) != JOB_KEYS:
        raise ValueError("queue must contain exactly the ordered four-center single/two pair")
    if set(config["admission_results"]) != {"single", "two", "resume"}:
        raise ValueError("all three engineering admissions are required")
    for path in config["admission_results"].values():
        Path(path).resolve().relative_to(Path("/home/linbinhao/ECG_adv_data"))
    if config["runtime"] != {"max_workers": 4, "poll_seconds": 30, "idle_seconds": 30,
                              "min_available_ram_gib": 48, "max_load_fraction": 0.9,
                              "min_disk_free_gib": 30}:
        raise ValueError("queue resource bounds are frozen")


def allowed_gpus():
    raw = os.environ.get("PULSE_GPU_ALLOWLIST", "")
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values or len(values) != len(set(values)) or not set(values) <= set(range(8)):
        raise ValueError("PULSE_GPU_ALLOWLIST must explicitly select unique physical GPUs 0..7")
    return values


def load_jobs(config, path, root):
    from boot_scripts.run_experiment import load_experiment_plan
    jobs, common = {}, None
    run_root = Path(config["admission_results"]["single"]).resolve().parents[2]
    for key, reference in config["references"].items():
        experiment = resolve_config_reference(reference, owner_config_path=path, config_root=root,
                                              description=key, must_exist=True)
        plan = load_experiment_plan(experiment, config_root=root, allow_existing_run_dir=True)
        if plan.entrypoint_name != "train_ecg_image" or plan.entry_arguments:
            raise ValueError("queue may only invoke the fixed PULSE training delegate")
        if plan.run_dir != run_root / f"pulse_augmix_{key}":
            raise ValueError("queue output is outside the admitted experiment family")
        payload, _ = load_config(plan.entry_config_path, root)
        if key != f"w{payload['width']}_{payload['center']}" or payload["mode"] != "train" or payload["resume_from"]:
            raise ValueError("queue member identity differs")
        paired = {k: v for k, v in payload.items() if k not in {"center", "width"}}
        common = paired if common is None else common
        if common != paired or (payload["training"]["optimizer_steps"], payload["training"]["effective_batch"]) != (100, 16):
            raise ValueError("queue pair changed optimizer/exposure budget or training recipe")
        jobs[key] = {"config": experiment, "run": plan.run_dir, "delegate": plan.delegate_output_dir,
                     "center": payload["center"], "width": payload["width"], "training": payload["training"]}
    if len({j["run"] for j in jobs.values()}) != 8:
        raise ValueError("queue output directories collide")
    return jobs


def admission(config):
    from util.run_record import verify_run_file_index
    hashes, sources = {}, None
    for kind, raw in config["admission_results"].items():
        path = Path(raw)
        result = json.loads(path.read_text())
        validate_result(result, path)
        if result["mode"] != "smoke" or verify_run_file_index(path.parent.parent):
            raise ValueError("invalid managed engineering admission")
        if result["protocol"]["width"] != (2 if kind == "two" else 1):
            raise ValueError("engineering admission has the wrong arm")
        current = result["protocol"]["implementation_sha256"]
        sources = current if sources is None else sources
        if sources != current:
            raise ValueError("engineering admissions did not use one implementation")
        audit = json.loads((path.parent / "optimization_audit.json").read_text())
        if audit["author_flat_packing_exact"] is not True or audit["augmented_images_persisted"] != 0:
            raise ValueError("visual caching admission failed")
        if kind == "resume":
            recovery = json.loads((path.parent / "recovery_audit.json").read_text())
            if recovery["process_resume_verified"] is not True or recovery["resume_max_trainable_delta"] != 0.0:
                raise ValueError("fresh-process exact resume admission failed")
        hashes[kind] = sha256_file(path)
    verify_sources(sources)
    return {"result_sha256": hashes, "training_sources": sources}


def verify_sources(sources):
    for relative, expected in sources.items():
        if sha256_file(REPO / relative) != expected:
            raise ValueError(f"admitted training source changed: {relative}")


def live_training_jobs(jobs):
    """Read-only process discovery; /proc identity is authoritative, not JSON."""
    outputs = {str(job["delegate"]).encode(): key for key, job in jobs.items()}
    matches = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        identity = process_identity(int(directory.name))
        if identity is None:
            continue
        try:
            argv = (directory / "cmdline").read_bytes().split(b"\0")
            if str(REPO / "boot_scripts/train_ecg_image.py").encode() not in argv or b"--output-dir" not in argv:
                continue
            key = outputs.get(argv[argv.index(b"--output-dir") + 1])
        except (OSError, IndexError):
            continue
        if key:
            if key in matches:
                raise ValueError(f"duplicate live training job: {key}")
            matches[key] = identity
    return matches


def job_state(job, *, live):
    manifest = job["run"] / "run_manifest.json"
    if live:
        return "running"
    if not job["run"].exists():
        return "pending"
    if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
        return "complete"
    return "stopped_incomplete_requires_explicit_resume"


def unclaimed_training_devices(candidates, job_devices, *, live, children):
    """Keep a CPU-loading child reserved before nvidia-smi can see its context."""
    active = set(live) | set(children)
    if active.difference(job_devices):
        # An adopted process can still be loading on CPU. Without its device
        # identity, no apparently idle GPU is safe to assign to another child.
        return []
    claimed = {job_devices[key] for key in active}
    return [gpu for gpu in candidates if gpu not in claimed]


def validate_queue_result(payload, path):
    if payload.get("schema_version") in (4, 5):
        from util.founder_width_queue import validate_result
        return validate_result(payload, path)
    if payload.get("schema_version") == 3:
        from util.run_record import verify_run_file_index
        if (payload.get("artifact_type") != "pulse_training_queue_result" or payload.get("status") != "complete"
                or set(payload.get("jobs", {})) != set(DUAL_KEYS)):
            raise ValueError("incomplete dual JSD training/evaluation workflow")
        for entry in payload["jobs"].values():
            member = Path(entry["result"])
            member.resolve().relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))
            if sha256_file(member) != entry["sha256"] or verify_run_file_index(member.parent.parent):
                raise ValueError("dual JSD result identity changed")
        comparison = payload.get("pulse_small_evaluation", {})
        if (set(comparison) != {"metrics", "sha256"}
                or Path(comparison["metrics"]).resolve() != (path.parent/"pulse_metrics.json").resolve()
                or sha256_file(Path(comparison["metrics"])) != comparison["sha256"]):
            raise ValueError("dual JSD matched visual comparison missing or changed")
        return
    if payload.get("artifact_type") != "pulse_training_queue_result" or payload.get("schema_version") not in (1, 2) or payload.get("status") != "complete":
        raise ValueError("invalid completed PULSE training queue")
    if set(payload["jobs"]) != set(JOB_KEYS):
        raise ValueError("queue completion is missing one of the eight training jobs")
    for key, item in payload["jobs"].items():
        result_path = Path(item["result"])
        result_path.resolve().relative_to(Path("/home/linbinhao/ECG_adv_data"))
        if sha256_file(result_path) != item["sha256"]:
            raise ValueError("queue training result changed")
        result = json.loads(result_path.read_text())
        validate_result(result, result_path)
        if result["mode"] != "train" or key != f"w{result['protocol']['width']}_{result['protocol']['center']}":
            raise ValueError("queue result identity mismatch")
    if payload["schema_version"] == 2:
        from util.evaluation.pulse_subset import validate_result as validate_subset
        item = payload["evaluation"]
        member = Path(item["result"])
        member.resolve().relative_to(Path("/home/linbinhao/ECG_adv_data"))
        if sha256_file(member) != item["sha256"]:
            raise ValueError("pixel pipeline evaluation fingerprint mismatch")
        validate_subset(json.loads(member.read_text()), member)


def pixel_pipeline_plans(config, path, root):
    """A fixed five-step workflow, not an arbitrary job graph or shell runner."""
    from boot_scripts.run_experiment import load_experiment_plan
    keys = ("single", "two", "resume", "training_queue", "evaluation")
    if (set(config) != {"schema_version", "references"} or config["schema_version"] != 2
            or tuple(config["references"]) != keys):
        raise ValueError("invalid finite pixel-mixing workflow")
    plans = {}
    for key, reference in config["references"].items():
        filename = resolve_config_reference(reference, owner_config_path=path, config_root=root,
            description=key, must_exist=True)
        plan = load_experiment_plan(filename, config_root=root, allow_existing_run_dir=True)
        expected = ("coordinate_pulse_training" if key == "training_queue" else
                    "evaluate_pulse_subset" if key == "evaluation" else "train_ecg_image")
        if plan.entrypoint_name != expected or plan.entry_arguments:
            raise ValueError("pixel workflow contains an unexpected delegate")
        if key in keys[:3]:
            child, _ = load_config(plan.entry_config_path, root)
            if (child.get("mixing_domain") != "rendered_rgb" or child["mode"] != "smoke"
                    or child["width"] != (2 if key == "two" else 1)):
                raise ValueError("pixel workflow admission has wrong domain/arm")
        plans[key] = (filename, plan)
    queue = load_yaml_mapping(plans["training_queue"][1].entry_config_path, description="pixel training queue")
    validate_config(queue)
    jobs = load_jobs(queue, plans["training_queue"][1].entry_config_path, root)
    for key in keys[:3]:
        if Path(queue["admission_results"][key]) != plans[key][1].delegate_output_dir / "train_result.json":
            raise ValueError("pixel admissions are not the queue's exact sources")
    for job in jobs.values():
        plan = load_experiment_plan(job["config"], config_root=root, allow_existing_run_dir=True)
        child, _ = load_config(plan.entry_config_path, root)
        if child.get("mixing_domain") != "rendered_rgb":
            raise ValueError("pixel workflow cannot train a waveform-mixing job")
    evaluation = load_yaml_mapping(plans["evaluation"][1].entry_config_path, description="pixel evaluation")
    if evaluation.get("schema_version") != 2:
        raise ValueError("pixel workflow must use matched completed-subset evaluation")
    from util.evaluation.pulse_subset import validate_config as validate_subset
    validate_subset(evaluation)
    paired_path = resolve_config_reference(evaluation["references"]["paired_config"],
        owner_config_path=plans["evaluation"][1].entry_config_path, config_root=root,
        description="pixel evaluation training lineage", must_exist=True)
    paired = load_yaml_mapping(paired_path, description="pixel paired evaluation")
    for job in jobs.values():
        if Path(paired["training_results"][job["center"]][f"w{job['width']}"]) != job["delegate"] / "train_result.json":
            raise ValueError("pixel evaluation is not linked to this training queue")
    return plans, jobs


def run_pixel_pipeline(config, config_path, config_root, output):
    from util.run_record import verify_run_file_index
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("pixel workflow coordinator must be CPU-only")
    candidates = allowed_gpus()
    if len(candidates) != 4:
        raise ValueError("pixel workflow requires exactly four explicit candidate GPUs")
    plans, jobs = pixel_pipeline_plans(config, config_path, config_root)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "control.json", {"paused": False})
    if os.environ.get("PULSE_IDLE_CONTEXT_PERMIT"):
        permit = Path(os.environ["PULSE_IDLE_CONTEXT_PERMIT"])
        if sha256_file(permit) != os.environ.get("PULSE_IDLE_CONTEXT_PERMIT_SHA256"):
            raise ValueError("GPU context authorization changed before launch")
        shutil.copyfile(permit, output / "idle_context_permit.json")
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    sources = {}
    for relative in ("util/pulse_training_queue.py", "boot_scripts/coordinate_pulse_training.py",
                     "core/augmix.py", "core/pulse_finetune.py", "util/pulse_training_contract.py",
                     "util/evaluation/pulse_subset.py", "util/evaluation/ecg_image_elastic.py",
                     "boot_scripts/run_experiment.py", "util/run_record.py"):
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        sources[relative] = sha256_file(destination)
    atomic_json(output / "source_identity.json", sources)
    completed = {}
    with exclusive_lock(output.parent / "pixel_pipeline.lock"):
        for key, (filename, plan) in plans.items():
            manifest = plan.run_dir / "run_manifest.json"
            if plan.run_dir.exists():
                if (not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete"
                        or verify_run_file_index(plan.run_dir)):
                    raise RuntimeError(f"preserved incomplete pixel workflow step requires explicit recovery: {key}")
                completed[key] = str(plan.run_dir)
                continue
            gpu, idle_since = None, {}
            while True:
                verify_sources(sources)
                control = json.loads((output / "control.json").read_text())
                if set(control) != {"paused"} or type(control["paused"]) is not bool:
                    raise ValueError("invalid pixel workflow control")
                if stopping[0]:
                    raise InterruptedError("pixel workflow stopped between steps; results preserved")
                snapshot = resource_snapshot()
                safe = (snapshot["available_ram_gib"] >= 48
                        and snapshot["load1"] <= snapshot["cpu_affinity_count"] * .9
                        and shutil.disk_usage(output).free >= 30 * 1024**3)
                for index in candidates:
                    if index in snapshot["gpus"] and free_device(snapshot["gpus"][index]):
                        idle_since.setdefault(index, time.time())
                    else:
                        idle_since.pop(index, None)
                available = [i for i in candidates if i in idle_since and time.time()-idle_since[i] >= 30]
                if not control["paused"] and safe and (key in ("training_queue", "evaluation") or available):
                    if available and key not in ("training_queue", "evaluation"):
                        gpu = resource_snapshot()["gpus"][available[0]]
                        if not free_device(gpu):
                            continue
                    break
                atomic_json(output / "status.json", {"stage": key, "status": "waiting_for_resources",
                    "completed": completed, "candidate_gpus": candidates, "resources": snapshot,
                    "time": time.time()})
                time.sleep(30)
            command = [sys.executable, str(REPO / "boot_scripts/run_experiment.py"),
                       "--config", str(filename), "--config-root", str(config_root)]
            dry = subprocess.run([*command, "--dry-run"], check=True, text=True, capture_output=True)
            atomic_json(output / f"{key}.dry_run.json", json.loads(dry.stdout))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"] if gpu else "",
                   "ECG_IMAGE_GPU_ALLOWLIST": ",".join(map(str, candidates)),
                   "PULSE_SUBSET_INITIAL_GPUS": ",".join(map(str, candidates))}
            with (output / f"{key}.launcher.log").open("x") as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(output / f"{key}.launch.json", {"command": command,
                    "process_identity": process_identity(child.pid), "gpu_uuid": gpu["uuid"] if gpu else None})
                while child.poll() is None:
                    atomic_json(output / "status.json", {"stage": key, "status": "running",
                        "pid": child.pid, "completed": completed, "child_run": str(plan.run_dir), "time": time.time()})
                    time.sleep(30)
                if child.returncode != 0 or verify_run_file_index(plan.run_dir):
                    raise RuntimeError(f"pixel workflow step failed; preserved without retry: {key}")
            completed[key] = str(plan.run_dir)
        queue_result = json.loads((plans["training_queue"][1].delegate_output_dir / "queue_result.json").read_text())
        evaluation = plans["evaluation"][1].delegate_output_dir / "subset_result.json"
        result = {**queue_result, "schema_version": 2,
            "evaluation": {"result": str(evaluation), "sha256": sha256_file(evaluation)}}
        validate_queue_result(result, output / "queue_result.json")
        atomic_json(output / "queue_result.json", result)
        atomic_json(output / "status.json", {"status": "complete", "completed": completed, "time": time.time()})


def dual_jsd_plans(config, config_path, config_root):
    from boot_scripts.run_experiment import load_experiment_plan
    if (set(config) != {"schema_version", "run_root", "references", "runtime"}
            or config["schema_version"] != 3 or tuple(config["references"]) != DUAL_KEYS):
        raise ValueError("dual JSD requires the exact finite two-lane experiment grid")
    if config["runtime"] != {"max_workers": 4, "poll_seconds": 30, "idle_seconds": 30,
            "min_available_ram_gib": 64, "max_load_fraction": 0.8, "min_disk_free_gib": 40}:
        raise ValueError("dual JSD resource bounds changed")
    run_root = Path(config["run_root"]).resolve()
    run_root.relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))
    plans = {}
    for key, raw in config["references"].items():
        path = resolve_config_reference(raw, owner_config_path=config_path, config_root=config_root,
                                        description=key, must_exist=True)
        plan = load_experiment_plan(path, config_root=config_root, allow_existing_run_dir=True)
        expected = ("evaluate_pulse_subset" if key in DUAL_PULSE_EVAL else "train_ecg_image"
                    if key.startswith("pulse_") else "evaluate_pn2021" if key.endswith("_eval") else "train_pn2021")
        if plan.run_dir != run_root / key or plan.entrypoint_name != expected:
            raise ValueError("dual JSD child escaped its finite run family or entrypoint")
        if key in DUAL_PULSE_EVAL:
            from util.evaluation.pulse_visual_subset import validate_config as validate_visual_subset
            child = load_yaml_mapping(plan.entry_config_path, description="visual subset")
            validate_visual_subset(child)
            if plan.entry_arguments or key != f"pulse_visual_eval_{child['center']}":
                raise ValueError("dual JSD PULSE evaluation center changed")
            if child["training_results"] != {arm:str(run_root/f"pulse_visual_{arm}_{child['center']}"/"training/train_result.json")
                                             for arm in ("clean", "single", "three")}:
                raise ValueError("dual JSD visual evaluation is not bound to its training")
        elif key.startswith("pulse_"):
            child, _ = load_config(plan.entry_config_path, config_root)
            arm = key.removeprefix("pulse_visual_").split("_")[0]
            if key in DUAL_SMOKES:
                arm = key.removeprefix("pulse_visual_smoke_")
            width = {"clean": 0, "single": 1, "three": 3, "resume": 3}[arm]
            if (child["schema_version"] != 3 or child["width"] != width or plan.entry_arguments
                    or child["mode"] != ("smoke" if key in DUAL_SMOKES else "train")):
                raise ValueError("dual JSD PULSE scope changed")
        else:
            args = dict(zip(plan.entry_arguments[::2], plan.entry_arguments[1::2]))
            if args.get("--model") != "ecgfounder" or args.get("--center") not in CENTERS:
                raise ValueError("dual JSD founder model/center mismatch")
            center = args["--center"]
            arm = key.removeprefix("founder_objective_").removesuffix("_eval").removesuffix("_"+center)
            method = {"simclr": "augmix_simclr_lhat", "clean_bce": "augmix_clean_bce_lhat",
                      "jsd": "augmix_clean_bce_jsd_lhat"}.get(arm)
            if args.get("--method-config") != f"train/methods/{method}.yaml":
                raise ValueError("dual JSD founder objective mismatch")
            if key.endswith("_eval") and args.get("--train-result") != str(run_root/key.removesuffix("_eval")/"training/train_result.json"):
                raise ValueError("founder evaluation is not linked to its matched training")
        plans[key] = (path, plan)
    return plans


class RecoveredManagedChild:
    """Observe a verified same-user launcher without signalling or restarting it."""
    def __init__(self, identity, run_dir):
        self.identity, self.run_dir = identity, run_dir
        self.pid, self.returncode = identity["pid"], None

    def poll(self):
        if process_identity(self.pid) == self.identity:
            return None
        manifest = self.run_dir/"run_manifest.json"
        self.returncode = 0 if manifest.is_file() and json.loads(manifest.read_text()).get("status") == "complete" else 75
        return self.returncode


def recover_dual_children(plans, run_root, candidates, config_root, output):
    """Adopt only exact live launch receipts in this one finite run family."""
    active = {}
    snapshot = resource_snapshot()
    for key, (_, plan) in plans.items():
        manifest = plan.run_dir/"run_manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "running":
            continue
        matches = []
        for receipt_path in run_root.glob(f"dual_jsd_workflow*/training/{key}.launch.json"):
            receipt = json.loads(receipt_path.read_text())
            identity = receipt["process_identity"]
            if process_identity(identity["pid"]) != identity:
                continue
            argv = Path(f"/proc/{identity['pid']}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            if argv != receipt["command"] or len(argv) < 4 or argv[1] != str(REPO/"boot_scripts/run_experiment.py"):
                raise ValueError("recovered launcher command differs from its owned receipt")
            for source, _ in plan.config_sources:
                previous = plan.run_dir/"configs"/source.relative_to(config_root)
                if not previous.is_file() or sha256_file(previous) != sha256_file(source):
                    raise ValueError("recovered child config closure differs from this workflow")
            gpu = next((i for i in candidates if snapshot["gpus"][i]["uuid"] == receipt["gpu_uuid"]), None)
            if gpu is None:
                raise ValueError("recovered child is outside the authorized four GPUs")
            matches.append((receipt_path, receipt, gpu))
        if len(matches) != 1:
            raise ValueError(f"incomplete child has no unique owned live launch receipt: {key}")
        previous, receipt, gpu = matches[0]
        if any(item["gpu"] == gpu for item in active.values()):
            raise ValueError("recovered live children overlap a GPU")
        active[key] = {"process": RecoveredManagedChild(receipt["process_identity"], plan.run_dir),
                       "log": None, "gpu": gpu, "lane": "pulse" if key.startswith("pulse_") else "founder"}
        atomic_json(output/f"{key}.recovery.json", {"previous_launch":str(previous),"receipt":receipt,
                    "checkpoint_restarted":False,"config_closure_exact":True})
    return active


def unclaimed_dual_devices(candidates, idle_since, active, now):
    claimed = {item["gpu"] for item in active.values()}
    return [i for i in candidates if i not in claimed and i in idle_since and now-idle_since[i] >= 30]


def run_dual_jsd(config, config_path, config_root, output):
    """Finite four-smoke/12-PULSE/4-subset/12-founder/12-full-eval grid."""
    from util.run_record import verify_run_file_index
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("dual JSD coordinator must be CPU-only")
    candidates = allowed_gpus()
    if len(candidates) != 4:
        raise ValueError("dual JSD uses exactly four explicit candidate GPUs")
    width_mode = config["schema_version"] in (4, 5)
    repeat_mode = config["schema_version"] == 5
    if width_mode:
        from util import founder_width_queue as width_queue
        plans = width_queue.plans(config, config_path, config_root)
        width_queue.admission(config)
    else:
        plans = dual_jsd_plans(config, config_path, config_root)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for relative in ("core/consistency.py", "core/pulse_visual.py", "core/pulse_finetune.py",
            "core/augmix.py", "core/online_trainer.py", "core/methods/registry.py",
            "util/pulse_training_contract.py", "util/pulse_training_queue.py",
            "util/evaluation/pulse_visual_subset.py", "util/evaluation/pulse_subset.py",
            "util/config_bundle.py", *(("util/founder_width_queue.py", "util/run_record.py",
                "core/methods/runtime.py", "core/corruption.py", "boot_scripts/run_experiment.py") if width_mode else ())):
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, dest)
        sources[relative] = sha256_file(dest)
    atomic_json(output/"source_identity.json", sources)
    atomic_json(output/"control.json", {"paused": False, "max_workers": 4})
    active, complete, idle_since = {}, {}, {}
    predecessor_finished = not width_mode
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    with exclusive_lock(Path(config["run_root"])/"dual_jsd.lock"):
        active.update(recover_dual_children(plans, Path(config["run_root"]), candidates, config_root, output))
        while True:
            verify_sources(sources)
            if not predecessor_finished:
                predecessor_finished = width_queue.predecessor_ready(config)
                if not predecessor_finished:
                    atomic_json(output/"status.json", {"status":"waiting_for_predecessor", "active":{},
                        "completed":{}, "gpu_allocations":0, "time":time.time(), "predecessor":config["predecessor"]})
                    if stopping[0]:
                        raise InterruptedError("width queue stopped while waiting; no children launched")
                    time.sleep(30)
                    continue
            for key, item in list(active.items()):
                if item["process"].poll() is None:
                    continue
                if item["log"] is not None:
                    if repeat_mode:
                        item["log"].seek(0)
                        (output/f"{key}.launcher.log").write_text("".join(deque(item["log"], maxlen=100)))
                    item["log"].close()
                if item["process"].returncode != 0:
                    raise RuntimeError(f"dual JSD child failed; preserved without automatic retry: {key}")
                del active[key]
                idle_since[item["gpu"]] = 0.0
            for key, (_, plan) in plans.items():
                if key in complete or key in active or not plan.run_dir.exists():
                    continue
                manifest = plan.run_dir/"run_manifest.json"
                if (not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete"
                        or verify_run_file_index(plan.run_dir)):
                    raise RuntimeError(f"preserved incomplete dual JSD child requires explicit recovery: {key}")
                member = plan.run_dir/plan.expected_result_relative_path
                complete[key] = {"result": str(member), "sha256": sha256_file(member)}
            if len(complete) == len(plans):
                if width_mode:
                    comparison = width_queue.finalize(config, complete, output)
                    result = {"artifact_type":"pulse_training_queue_result", "schema_version":config["schema_version"],
                        "status":"complete", "jobs":complete, "admission":{"sources":sources},
                        "width2_baselines":config.get("width2_baselines", {}), "width_comparison":comparison}
                    validate_queue_result(result, output/"queue_result.json")
                    atomic_json(output/"queue_result.json", result)
                    atomic_json(output/"status.json", {"status":"complete", "completed":complete})
                    return
                from util.evaluation.pulse_visual_subset import finalize_comparison
                comparison = finalize_comparison({c:complete[f"pulse_visual_eval_{c}"]["result"] for c in CENTERS}, output)
                result = {"artifact_type": "pulse_training_queue_result", "schema_version": 3,
                    "status": "complete", "jobs": complete, "admission": {"sources": sources},
                    "pulse_small_evaluation": comparison}
                validate_queue_result(result, output/"queue_result.json")
                atomic_json(output/"queue_result.json", result)
                atomic_json(output/"status.json", {"status": "complete", "completed": complete})
                return
            snapshot = resource_snapshot()
            control = json.loads((output/"control.json").read_text())
            if (set(control) != {"paused", "max_workers"} or type(control["paused"]) is not bool
                    or type(control["max_workers"]) is not int or not 0 <= control["max_workers"] <= 4):
                raise ValueError("dual JSD control exceeds permission")
            atomic_json(output/"status.json", {"status": "draining" if stopping[0] else "running",
                "completed": complete, "active": {k:{"pid":v["process"].pid,"gpu":v["gpu"]} for k,v in active.items()},
                "resources": snapshot, "time": time.time()})
            if stopping[0] and not active:
                raise InterruptedError("dual JSD stopped after running children; results preserved")
            for index in candidates:
                if index in snapshot["gpus"] and free_device(snapshot["gpus"][index]):
                    idle_since.setdefault(index, time.time())
                else:
                    idle_since.pop(index, None)
            available = unclaimed_dual_devices(candidates, idle_since, active, time.time())
            safe = (snapshot["available_ram_gib"] >= 64
                and snapshot["load1"] <= snapshot["cpu_affinity_count"] * .8
                and shutil.disk_usage(output).free >= 40*1024**3)
            pending = [k for k in plans if k not in complete and k not in active]
            if available and safe and not stopping[0] and not control["paused"] and len(active) < control["max_workers"]:
                for key in pending:
                    if key in DUAL_SMOKES and any(k not in complete for k in DUAL_SMOKES[:DUAL_SMOKES.index(key)]):
                        continue
                    if key in DUAL_PULSE:
                        if not all(k in complete for k in DUAL_SMOKES):
                            continue
                        recovery = Path(complete[DUAL_SMOKES[-1]]["result"]).parent/"recovery_audit.json"
                        if json.loads(recovery.read_text()).get("process_resume_verified") is not True:
                            raise ValueError("PULSE formal training requires fresh-process resume admission")
                    if key in DUAL_PULSE_EVAL and any(f"pulse_visual_{arm}_{key.removeprefix('pulse_visual_eval_')}" not in complete
                                                   for arm in ("clean", "single", "three")):
                        continue
                    if (key in DUAL_FOUNDER or width_mode) and key.endswith("_eval") and key.removesuffix("_eval") not in complete:
                        continue
                    lane = "pulse" if key.startswith("pulse_") else "founder"
                    other_lane_pending = any((k.startswith("pulse_")) != (lane == "pulse") for k in pending)
                    if other_lane_pending and sum(v["lane"] == lane for v in active.values()) >= 2:
                        continue
                    index = available[0]
                    device = resource_snapshot()["gpus"][index]
                    if not free_device(device):
                        break
                    path, plan = plans[key]
                    python = ("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python" if lane == "pulse"
                              else "/home/linbinhao/miniforge3/envs/ECGTwin/bin/python")
                    cmd = [python, str(REPO/"boot_scripts/run_experiment.py"), "--config", str(path), "--config-root", str(config_root)]
                    dry = subprocess.run([*cmd,"--dry-run"], check=True, capture_output=True, text=True)
                    atomic_json(output/f"{key}.dry_run.json", json.loads(dry.stdout))
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": device["uuid"], "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2"}
                    log = (tempfile.TemporaryFile(mode="w+", dir=os.environ["ECG_RUNTIME_LOG_DIR"])
                           if repeat_mode else (output/f"{key}.launcher.log").open("x"))
                    child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[key] = {"process":child,"log":log,"gpu":index,"lane":lane}
                    atomic_json(output/f"{key}.launch.json", {"command":cmd,"gpu_uuid":device["uuid"],"process_identity":process_identity(child.pid)})
                    if repeat_mode:
                        available.pop(0)
                        if available and len(active) < control["max_workers"]:
                            continue
                    break
            time.sleep(30)


def run(config_path, config_root, output):
    from util.run_record import verify_run_file_index
    config = load_yaml_mapping(config_path, description="PULSE training queue")
    if config.get("schema_version") in (3, 4, 5):
        return run_dual_jsd(config, config_path, config_root, output)
    if config.get("schema_version") == 2:
        return run_pixel_pipeline(config, config_path, config_root, output)
    validate_config(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU queue must launch with CUDA_VISIBLE_DEVICES empty")
    candidates = allowed_gpus()
    jobs = load_jobs(config, config_path, config_root)
    output.mkdir(parents=True, exist_ok=False)
    gate = admission(config)
    gate["coordinator_sources"] = {}
    for relative in ("util/pulse_training_queue.py", "boot_scripts/coordinate_pulse_training.py",
                     "util/config_bundle.py", "util/evaluation/ecg_image_elastic.py",
                     "util/evaluation/ecg_image_queue.py", "boot_scripts/run_experiment.py", "util/run_record.py"):
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        gate["coordinator_sources"][relative] = sha256_file(destination)
    atomic_json(output / "admission.json", gate)
    atomic_json(output / "control.json", {"paused": False, "max_workers": 4, "allowed_gpus": candidates})
    atomic_json(output / "jobs.json", {key: {k: str(v) if isinstance(v, Path) else v for k, v in job.items()}
                                       for key, job in jobs.items()})
    children, completed, idle_since, job_devices = {}, {}, {}, {}
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    # One lock per shared eight-job output set, including a restarted coordinator.
    root = jobs[JOB_KEYS[0]]["run"].parent
    with exclusive_lock(root / "training_queue.lock"):
        while True:
            for key, child in list(children.items()):
                code = child["process"].poll()
                if code is not None:
                    child["log"].close()
                    del children[key]
            live = live_training_jobs(jobs)
            snapshot = resource_snapshot()
            for key, identity in live.items():
                for gpu, device in snapshot["gpus"].items():
                    if identity["pid"] in device["pids"]:
                        job_devices[key] = gpu
            states = {key: job_state(job, live=key in live or key in children) for key, job in jobs.items()}
            for key, status in states.items():
                if status != "complete" or key in completed:
                    continue
                job = jobs[key]
                path = job["delegate"] / "train_result.json"
                result = json.loads(path.read_text())
                validate_result(result, path)
                if (verify_run_file_index(job["run"]) or result["mode"] != "train"
                        or result["protocol"]["center"] != job["center"] or result["protocol"]["width"] != job["width"]
                        or result["protocol"]["training"] != job["training"]
                        or result["protocol"]["implementation_sha256"] != gate["training_sources"]):
                    raise ValueError("completed training job failed queue lineage validation")
                completed[key] = {"result": str(path), "sha256": sha256_file(path)}
                if key in job_devices:  # immediate handoff of our own just-finished GPU
                    idle_since[job_devices[key]] = time.time() - config["runtime"]["idle_seconds"]
            atomic_json(output / "status.json", {"time": time.time(), "states": states,
                "live_processes": live, "job_devices": job_devices, "completed_jobs": len(completed),
                "resources": snapshot, "stop_after_running_jobs": stopping[0]})
            if len(completed) == 8:
                result = {"artifact_type": "pulse_training_queue_result", "schema_version": 1,
                          "status": "complete", "jobs": completed, "admission": gate}
                validate_queue_result(result, output / "queue_result.json")
                atomic_json(output / "queue_result.json", result)
                return
            failed = [key for key, state in states.items() if state.startswith("stopped_incomplete")]
            if failed:
                raise RuntimeError(f"queue preserved incomplete runs; explicit matched resume required: {failed}")
            if stopping[0]:
                raise InterruptedError("CPU coordinator stopped; active training jobs remain adoptable")
            control = json.loads((output / "control.json").read_text())
            if (set(control) != {"paused", "max_workers", "allowed_gpus"} or type(control["paused"]) is not bool
                    or type(control["max_workers"]) is not int or not 0 <= control["max_workers"] <= 4
                    or not set(control["allowed_gpus"]) <= set(candidates)):
                raise ValueError("queue control exceeds authorized bounds")
            limits = config["runtime"]
            pressure_ok = (snapshot["available_ram_gib"] >= limits["min_available_ram_gib"]
                           and snapshot["load1"] <= snapshot["cpu_affinity_count"] * limits["max_load_fraction"]
                           and shutil.disk_usage(root).free >= limits["min_disk_free_gib"] * 1024**3)
            for gpu in candidates:
                if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu]):
                    idle_since.setdefault(gpu, time.time())
                else:
                    idle_since.pop(gpu, None)
            pending = [key for key, value in states.items() if value == "pending"]
            if (pending and pressure_ok and not stopping[0] and not control["paused"]
                    and len(set(live) | set(children)) < control["max_workers"]):
                for gpu in unclaimed_training_devices(
                    control["allowed_gpus"], job_devices, live=live, children=children,
                ):
                    if gpu not in idle_since or time.time() - idle_since[gpu] < limits["idle_seconds"]:
                        continue
                    verify_sources(gate["training_sources"])
                    device = resource_snapshot()["gpus"][gpu]
                    if not free_device(device):
                        continue
                    key, job = pending[0], jobs[pending[0]]
                    command = [sys.executable, str(REPO / "boot_scripts/run_experiment.py"),
                               "--config", str(job["config"]), "--config-root", str(config_root)]
                    # Dry-run the exact frozen bundle, then execute without overrides.
                    dry = subprocess.run([*command, "--dry-run"], text=True, capture_output=True, check=True)
                    atomic_json(output / f"{key}.dry_run.json", json.loads(dry.stdout))
                    log = (output / f"{key}.launcher.log").open("x")
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device["uuid"],
                                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
                    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    children[key] = {"process": process, "log": log}
                    job_devices[key] = gpu
                    atomic_json(output / f"{key}.launch.json", {"command": command, "gpu_uuid": device["uuid"],
                        "launcher_identity": process_identity(process.pid), "started_at": time.time()})
                    break  # stagger CPU model loads and disk reads
            time.sleep(limits["poll_seconds"])
