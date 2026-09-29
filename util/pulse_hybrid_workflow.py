"""Finite study orchestration using the retained launcher/resource primitives.

No GPU model in the coordinator. Each child owns one free device. Scientific
trial values are materialized in a copied YAML bundle, never CLI overrides.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

from boot_scripts.run_experiment import load_experiment_plan
from core.pulse_hpo import open_study, suggest_recipe, training_cell, matched_trial_score
from util.evaluation.ecg_image_queue import atomic_json, digest_json, exclusive_lock
from util.pn2021_artifact_contract import sha256_file
from util.pulse_hybrid_contract import load_config, validate_config, validate_result, finalize
from util.pulse_training_contract import CENTERS, validate_result as validate_training

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/linbinhao/miniforge3/envs/ECGTwin/bin/python")


def write_yaml(path, payload):
    """Immutable generated scientific config; resumes must propose identical bytes."""
    text = yaml.safe_dump(payload, sort_keys=False)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"generated config drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


def make_jobs(config, root, template, *, phase, recipe, trial_number=None, resume_checkpoints=None, precompleted=None):
    """Return a finite dependency DAG; no data, model or GPU work here."""
    label = f"trial_{trial_number:02d}" if phase == "screen" else phase
    fixed = config["mode"] == "fixed"
    if precompleted and (not fixed or phase != "final"):
        raise ValueError("completed recovery entries require the fixed final phase")
    if fixed and phase not in ("smoke", "final"):
        raise ValueError("fixed workflow has no hyperparameter screening stage")
    durable = Path(config["paths"]["run_root"])
    training_root = Path(config["paths"]["temporary_root"]) / label if phase == "screen" else durable / label
    bundle = durable / "bundles" / label / "configs"
    # The parent's already-resolved bundle contains the template's complete data closure.
    if not bundle.exists():
        shutil.copytree(root, bundle)
    else:
        for source in root.rglob("*.yaml"):
            target = bundle / source.relative_to(root)
            if not target.is_file() or sha256_file(source) != sha256_file(target):
                raise ValueError("copied parent config closure changed")
    jobs = {}
    def add(name, entrypoint, child, run_dir, dependencies):
        section = "eval" if child.get("mode") == "evaluate" else "train"
        entry = bundle / section / f"{name}.yaml"
        experiment = bundle / "experiments" / f"{name}.yaml"
        write_yaml(entry, child)
        write_yaml(experiment, {"schema_version": 1,
            "experiment": {"name": name, "purpose": "Managed hybrid PN2021 development cell; no independent-test claim."},
            "entrypoint": {"name": entrypoint, "config": f"{section}/{name}.yaml", "arguments": []},
            "output": {"run_dir": str(run_dir), "if_exists": "error", "delegate_output_subdir": "evaluation" if section == "eval" else "training"}})
        plan = load_experiment_plan(experiment, config_root=bundle, allow_existing_run_dir=True)
        jobs[name] = {"experiment": experiment, "bundle": bundle, "plan": plan, "dependencies": tuple(dependencies)}
        return name, plan.delegate_output_dir
    c5 = fixed and config["image_suite"] == "image_c5_gpu_v1"
    centers = ("ningbo",) if phase == "smoke" and not c5 else CENTERS
    for center in centers:
        paths, dependencies = {}, []
        for arm, width in config["arms"].items():
            if phase == "smoke":
                child = copy.deepcopy(template)
                child.update(center=center, width=width)
                if fixed:
                    child.update(mode="smoke", resume_from=None)
                    child["training"].update(optimizer_steps=4 if c5 else 2, effective_batch=2, warmup_steps=0,
                                             save_every=1, smoke_records=4)
            elif fixed:
                child = copy.deepcopy(template)
                child.update(center=center, width=width)
                if resume_checkpoints and f"{center}_{arm}" in resume_checkpoints:
                    child["resume_from"] = resume_checkpoints[f"{center}_{arm}"]
            else:
                child = training_cell(template, center=center, width=width, recipe=recipe,
                    steps=config["screening_steps"] if phase == "screen" else config["final_steps"], space=config["search_space"])
            if fixed:
                child["paths"]["temporary_root"] = config["paths"]["temporary_root"]
            name, out = add(f"{label}_{center}_{arm}", "train_ecg_image", child, training_root / f"{center}_{arm}", [])
            dependencies.append(name)
            paths[arm] = (precompleted[name]["path"] if precompleted and name in precompleted
                          else str(out / "train_result.json"))
        if phase == "smoke":
            resumed = copy.deepcopy(child if fixed else template)
            resumed.update(center=center, width=3)
            resumed["resume_from"] = str(Path(paths["three"]).parent / "resume_probe.pt")
            name, _ = add(f"smoke_{center}_resume", "train_ecg_image", resumed, training_root / f"{center}_resume", [dependencies[-1]])
            dependencies.append(name)
        evaluation = {"schema_version": 1, "mode": "evaluate", "phase": phase,
            "references": {"training_config": config["references"]["training_config"]}, "center": center,
            "records_per_center": (4 if phase == "smoke" else 128 if phase == "screen"
                                    else config["records_per_center"]),
            "source_state": config["paths"]["source_state"], "training_results": paths,
            "waveform_indices": list(range(1, 21)) if phase == "final" else [1, 10, 11, 20],
            "image_strength": 1.0, "seed": config["seed"]}
        if fixed:
            evaluation.update(model_arms=["original", "single", "three"],
                image_suite=config["image_suite"], image_severity=config["image_severity"],
                waveform_indices=([] if config["image_suite"] == "image_c5_gpu_v1"
                                  else list(range(1, 6))))
            evaluation["references"] = {"training_config": f"train/{label}_{center}_three.yaml"}
            if c5:
                if "inference_batch_size" in config:
                    evaluation["inference_batch_size"] = config["inference_batch_size"]
                if "sampling_method" in config and phase != "smoke":
                    evaluation["sampling_method"] = config["sampling_method"]
                evaluation["original_baseline_mode"] = config["original_baseline_mode"]
                evaluation["execution_mode"] = "arm_major_fp16_adapters_v2"
                if phase == "smoke":
                    evaluation["admission_sample_keys"] = config["smoke_sample_keys"][center]
                    evaluation["performance_smoke"] = True
        validate_config(evaluation)
        if (phase == "final" and config.get("evaluation_block_size") == 512
                and f"final_{center}_eval" not in (precompleted or {})):
            for span in center_block_spans(center):
                child = {**evaluation, "phase": "block", "record_span": span}
                validate_config(child)
                suffix = f"{span[0]:05d}_{span[1]:05d}"
                name, _ = add(f"final_{center}_eval_{suffix}", "pulse_hybrid", child,
                    durable / label / f"{center}_eval_blocks" / suffix, dependencies)
                jobs[name].update(record_span=span, merge_config=evaluation)
        else:
            add(f"{label}_{center}_eval", "pulse_hybrid", evaluation, durable / label / f"{center}_eval", dependencies)
    if fixed and phase == "final":
        training = tuple(name for name, job in jobs.items() if job["plan"].entrypoint_name == "train_ecg_image")
        for job in jobs.values():
            if job["plan"].entrypoint_name == "pulse_hybrid":
                job["dependencies"] = training
    if phase == "final" and config.get("evaluation_block_size") == 512:
        # Stable order interleaves centers at each block offset; idle GPUs take the next ready block.
        jobs = dict(sorted(jobs.items(), key=lambda item: (
            item[1]["plan"].entrypoint_name != "train_ecg_image", item[1].get("record_span", [0])[0])))
    return jobs


def checked_completion(job):
    from util.run_record import verify_run_file_index
    plan = job["plan"]
    path = plan.run_dir / plan.expected_result_relative_path
    manifest = plan.run_dir / "run_manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete":
        return None
    if verify_run_file_index(plan.run_dir):
        raise ValueError(f"completed child file index changed: {plan.run_dir}")
    result = json.loads(path.read_text())
    if plan.entrypoint_name == "train_ecg_image":
        validate_training(result, path)
    else:
        validate_result(result, path)
    return {"path": str(path), "sha256": sha256_file(path), "result": result}


class LiveC5Metrics:
    """Summarize only atomically committed records while full C5 blocks run."""

    def __init__(self):
        self.cohorts = {}
        self.finalized_cohorts = set()
        self.expected_sample_owner = {}
        self.completed_sample_keys = {center: set() for center in CENTERS}
        self.conditions = {}
        self.counts = {}

    def _condition_stats(self, center, condition_id, arm, class_count):
        key = (center, condition_id, arm)
        if key not in self.counts:
            self.counts[key] = {"tp": [0] * class_count, "fp": [0] * class_count,
                "fn": [0] * class_count, "exact": 0, "hamming_errors": 0,
                "parse_failures": 0, "truncated": 0, "records": 0}
        return self.counts[key]

    def refresh(self, jobs, complete, active, config, state):
        from util.evaluation.pulse_hybrid_development import CLASS_ORDER, FULL_CENTER_RECORDS

        arms = ("original", "single", "three")
        batch_size = config.get("inference_batch_size", 1)
        for name, job in jobs.items():
            if job["plan"].entrypoint_name != "pulse_hybrid":
                continue
            if name in complete:
                output_dir = Path(complete[name]["path"]).parent
            elif name in active:
                output_dir = Path(job["plan"].delegate_output_dir)
            else:
                continue
            cohort_path = output_dir / "cohort.json"
            cohort_key = str(output_dir.resolve())
            if cohort_key in self.finalized_cohorts:
                continue
            if cohort_key not in self.cohorts:
                if not cohort_path.is_file():
                    continue
                cohort = json.loads(cohort_path.read_text())
                if job.get("merge_config"):
                    center = job["merge_config"]["center"]
                else:
                    child = yaml.safe_load(Path(job["plan"].entry_config_path).read_text())
                    center = child["center"]
                samples, conditions = cohort["samples"], cohort["conditions"]
            else:
                cohort = self.cohorts[cohort_key]
                center, samples, conditions = cohort["center"], cohort["samples"], cohort["conditions"]
            condition_ids = [item["condition_id"] for item in conditions]
            if not samples or len(condition_ids) != len(set(condition_ids)):
                raise ValueError(f"invalid live C5 cohort: {name}")

            if cohort_key not in self.cohorts:
                for sample in samples:
                    key = sample["sample_key"]
                    owner = self.expected_sample_owner.get((center, key))
                    if owner is not None and owner != cohort_key:
                        raise ValueError(f"overlapping live evaluation blocks: {center}/{key}")
                    self.expected_sample_owner[(center, key)] = cohort_key
                prior = self.conditions.get(center)
                if prior is not None and digest_json(prior) != digest_json(conditions):
                    raise ValueError(f"live evaluation conditions changed within {center}")
                self.conditions[center] = conditions
                self.cohorts[cohort_key] = {"center": center, "samples": samples,
                    "conditions": conditions, "next_offset": 0}

            progress_path = output_dir / "progress.json"
            if not progress_path.is_file():
                if name in complete:
                    raise ValueError(f"completed evaluation lacks progress receipt: {name}")
                continue
            progress = json.loads(progress_path.read_text())
            completed_records = progress.get("completed_records")
            if (type(completed_records) is not int or not 0 <= completed_records <= len(samples)
                    or progress.get("total_records") != len(samples)):
                raise ValueError(f"invalid live evaluation progress: {name}")
            batch_dir = output_dir / "batches"
            cohort_state = self.cohorts[cohort_key]
            while cohort_state["next_offset"] < completed_records:
                offset = cohort_state["next_offset"]
                batch_path = batch_dir / f"{offset:04d}.json"
                if not batch_path.is_file():
                    break
                batch_samples = samples[offset:offset + batch_size]
                rows = json.loads(batch_path.read_text())
                expected = {(sample["sample_key"], condition_id)
                    for sample in batch_samples for condition_id in condition_ids}
                identities = [(row["sample_key"], row["condition_id"]) for row in rows]
                if (not batch_samples or len(identities) != len(set(identities))
                        or set(identities) != expected):
                    raise ValueError(f"incomplete live prediction batch: {batch_path}")
                sample_by_key = {sample["sample_key"]: sample for sample in batch_samples}
                for row in rows:
                    sample = sample_by_key[row["sample_key"]]
                    if (row.get("true_labels") != sample["label_names"]
                            or set(row.get("arms", {})) != set(arms)):
                        raise ValueError(f"live prediction identity changed: {batch_path}")
                    truth = set(row["true_labels"])
                    for arm in arms:
                        answer = row["arms"][arm]
                        predicted = answer.get("predicted_labels")
                        if (not isinstance(predicted, list) or len(predicted) != len(set(predicted))
                                or set(predicted) - set(CLASS_ORDER)):
                            raise ValueError(f"invalid live predicted labels: {batch_path}")
                        prediction = set(predicted)
                        stats = self._condition_stats(center, row["condition_id"], arm, len(CLASS_ORDER))
                        for index, label in enumerate(CLASS_ORDER):
                            actual, guessed = label in truth, label in prediction
                            stats["tp"][index] += int(actual and guessed)
                            stats["fp"][index] += int(not actual and guessed)
                            stats["fn"][index] += int(actual and not guessed)
                            stats["hamming_errors"] += int(actual != guessed)
                        stats["exact"] += int(truth == prediction)
                        stats["parse_failures"] += int(not answer.get("parse_valid", False))
                        stats["truncated"] += int(answer.get("hit_max_new_tokens", False))
                        stats["records"] += 1
                for sample in batch_samples:
                    key = sample["sample_key"]
                    if key in self.completed_sample_keys[center]:
                        raise ValueError(f"duplicate live evaluated sample: {center}/{key}")
                    self.completed_sample_keys[center].add(key)
                cohort_state["next_offset"] += len(batch_samples)
            if name in complete and cohort_state["next_offset"] != len(samples):
                raise ValueError(f"completed evaluation has uncommitted live records: {name}")
            if name in complete:
                self.finalized_cohorts.add(cohort_key)

        per_center = {}
        for center in CENTERS:
            available = len(self.completed_sample_keys[center])
            families = {}
            condition_scores = {}
            condition_metrics = {}
            if available:
                conditions = self.conditions[center]
                families = {arm: {} for arm in arms}
                for condition in conditions:
                    condition_id = condition["condition_id"]
                    family = condition["family"]
                    for arm in arms:
                        stats = self.counts[(center, condition_id, arm)]
                        f1_values = []
                        for tp, fp, fn in zip(stats["tp"], stats["fp"], stats["fn"]):
                            denominator = 2 * tp + fp + fn
                            f1_values.append(2 * tp / denominator if denominator else 0.0)
                        macro_f1 = sum(f1_values) / len(f1_values)
                        condition_scores.setdefault(arm, {})[condition_id] = macro_f1
                        condition_metrics.setdefault(arm, {})[condition_id] = {
                            "macro_f1": macro_f1,
                            "exact_match": stats["exact"] / stats["records"],
                            "hamming_loss": stats["hamming_errors"] / (stats["records"] * len(CLASS_ORDER)),
                            "parse_failure_rate": stats["parse_failures"] / stats["records"],
                            "truncation_rate": stats["truncated"] / stats["records"],
                            "records": stats["records"]}
                        families[arm].setdefault(family, []).append(macro_f1)
                families = {arm: {family: sum(values) / len(values) for family, values in by_family.items()}
                    for arm, by_family in families.items()}
            deltas = ({family: 100 * (families["three"][family] - families["single"][family])
                       for family in families["single"]} if available else {})
            per_center[center] = {"completed_records": available,
                "total_records": FULL_CENTER_RECORDS[center], "arms": families,
                "three_minus_single_macro_f1_pp": deltas,
                "per_condition_macro_f1": condition_scores,
                "per_condition_metrics": condition_metrics}

        present = [center for center in CENTERS if per_center[center]["completed_records"]]
        mean_arms = {arm: {family: sum(per_center[c]["arms"][arm][family] for c in present) / len(present)
                           for family in ("clean", "image")}
                     for arm in arms} if present else {}
        totals = sum(FULL_CENTER_RECORDS.values())
        completed_records = sum(item["completed_records"] for item in per_center.values())
        atomic_json(state / "live_metrics.json", {
            "schema_version": 1,
            "status": "all_predictions_committed_final_merge_pending" if completed_records == totals else "in_progress",
            "provisional": True,
            "updated_at": time.time(),
            "metric": "macro_f1; image is the equal mean across five severity-5 image operators",
            "comparison_sign": "three_minus_single_macro_f1_pp; positive favors three-chain",
            "progress": {"completed_records": completed_records, "total_records": totals,
                "per_center": {center: {"completed_records": per_center[center]["completed_records"],
                    "total_records": per_center[center]["total_records"]} for center in CENTERS}},
            "metrics": {"per_center": per_center,
                "current_equal_center_mean": {"centers_included": present, "arms": mean_arms,
                    "three_minus_single_macro_f1_pp": ({family: 100 * (mean_arms["three"][family]
                        - mean_arms["single"][family]) for family in ("clean", "image")} if present else {})}}
        })


def stop_failed_children(active, state, *, grace_seconds=5):
    """Stop only verified launch-session members; never signal a reused PID."""
    from util.evaluation.ecg_image_elastic import process_identity
    members, skipped = {}, {}
    for name, child in active.items():
        receipt_path = state / "launches" / f"{name}.json"
        try:
            receipt = json.loads(receipt_path.read_text())
            identity = receipt["process"]
            pid = identity["pid"]
            if process_identity(pid) != identity:
                skipped[name] = "launcher exited or identity changed; no signal sent"
                continue
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            if (pid != child["process"].pid or argv != receipt["command"]
                    or os.getpgid(pid) != pid or os.getsid(pid) != pid):
                skipped[name] = "launch command/session ownership did not match"
                continue
            for directory in Path("/proc").iterdir():
                if not directory.name.isdigit():
                    continue
                try:
                    if directory.stat().st_uid != os.getuid():
                        continue
                    fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
                    if int(fields[2]) != pid or int(fields[3]) != pid:
                        continue
                    member = process_identity(int(directory.name))
                    if member is not None:
                        members[member["pid"]] = {"job": name, "identity": member}
                except (OSError, IndexError, ValueError):
                    continue
        except (OSError, KeyError, TypeError, ValueError) as error:
            skipped[name] = str(error)
    sent = []
    def signal_members(sig):
        for pid, member in sorted(members.items(), reverse=True):
            if process_identity(pid) == member["identity"]:
                try:
                    os.kill(pid, sig)
                    sent.append({"job": member["job"], "process": member["identity"], "signal": sig.name})
                except ProcessLookupError:
                    pass
    signal_members(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(process_identity(p) == m["identity"] for p, m in members.items()):
        time.sleep(0.1)
    signal_members(signal.SIGKILL)
    for child in active.values():
        process = child["process"]
        if callable(getattr(process, "wait", None)):
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    survivors = [m for p, m in members.items() if process_identity(p) == m["identity"]]
    result = {"signals": sent, "survivors": survivors, "skipped": skipped}
    atomic_json(state / "failure_cleanup.json", result)
    return result


def storage_admission(config, state, *, initial=False):
    """Keep RAM staging and durable-disk reserves separate and observable."""
    storage = config.get("storage")
    if not storage:
        return {"ready": shutil.disk_usage(state).free >= config["runtime"]["min_disk_gib"] * 1024**3
            and shutil.disk_usage(config["paths"]["temporary_root"]).free >= 8 * 1024**3}
    archive = Path(storage["archive_root"])
    ancestor = archive.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_uid != os.getuid() or archive.exists() or archive.with_name(archive.name + ".partial").exists():
        raise ValueError("archive target is unowned or already exists; inspect before recovery")
    ram_free = shutil.disk_usage(state).free / 1024**3
    disk_free = shutil.disk_usage(ancestor).free / 1024**3
    required_ram = storage["ram_reserve_gib"] + (storage["ram_budget_gib"] if initial else 0)
    # Account for inherited memory limits as well as host MemAvailable. Only
    # inactive file cache is considered reclaimable; shmem is never deducted.
    headroom = float("inf")
    cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    node = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    while node != Path("/sys/fs/cgroup"):
        if (node / "memory.current").exists():
            current = int((node / "memory.current").read_text())
            stats = dict(line.split() for line in (node / "memory.stat").read_text().splitlines())
            reclaimable = int(stats.get("inactive_file", 0))
            for name in ("memory.high", "memory.max"):
                value = (node / name).read_text().strip()
                if value != "max":
                    headroom = min(headroom, (int(value) - current + reclaimable) / 1024**3)
        node = node.parent
    used = sum(p.stat().st_size for p in Path(config["paths"]["run_root"]).rglob("*") if p.is_file()) / 1024**3
    return {"ready": ram_free >= required_ram and used < storage["ram_budget_gib"]
            and disk_free >= config["runtime"]["min_disk_gib"] + storage["ram_budget_gib"]
            and headroom >= 32,
        "ram_free_gib": ram_free, "ram_output_gib": used, "required_ram_free_gib": required_ram,
        "archive_free_gib": disk_free, "cgroup_reclaimable_headroom_gib": None if headroom == float("inf") else headroom,
        "archive_root": str(archive)}


def release_loaded_model_cache(job, state, name):
    """Advise away this task's clean disk cache after its model is loaded."""
    plan = job["plan"]
    if (plan.entrypoint_name != "train_ecg_image"
            or not (plan.delegate_output_dir / "model_audit.json").is_file()
            or not (plan.delegate_output_dir / "token_audit.json").is_file()):
        return False
    child = yaml.safe_load(plan.entry_config_path.read_text())
    directory = Path(child["model"]["directory"]).resolve()
    directory.relative_to(Path("/home/linbinhao/ECG_adv_data"))
    released = []
    for path in sorted(directory.glob("*.safetensors")):
        actual = path.resolve()
        actual.relative_to(Path("/home/linbinhao"))
        fd = os.open(actual, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid():
                raise ValueError("model cache belongs to another user")
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            released.append({"path": str(actual), "bytes": info.st_size})
        finally:
            os.close(fd)
    atomic_json(state / "cache_release" / f"{name}.json", {"time": time.time(),
        "action": "advisory_clean_file_cache_release", "files": released})
    return True


def run_jobs(jobs, config, state, source_hashes, *, smoke=False, precompleted=None):
    from util.evaluation.ecg_image_elastic import resource_snapshot, free_device, process_identity
    from util.pulse_training_queue import RecoveredManagedChild
    active, complete, idle_since = {}, dict(precompleted or {}), {}
    if not complete.keys() <= jobs.keys():
        raise ValueError("recovery entries are outside the managed job graph")
    for name, entry in complete.items():
        if (entry["result"].get("status") != "complete"
                or sha256_file(Path(entry["path"])) != entry["sha256"]):
            raise ValueError(f"reused completed result changed: {name}")
    live_metrics = LiveC5Metrics() if not smoke and config.get("evaluation_block_size") == 512 else None
    released_cache = set()
    stopping = [False]
    previous_handlers = {s: signal.signal(s, lambda *_: stopping.__setitem__(0, True)) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Adopt only exact same-user PID/start-time/boot identities from our receipts.
        for name, job in jobs.items():
            if name in complete:
                if job["plan"].run_dir.exists():
                    raise ValueError(f"reused task collides with a local output: {name}")
                mirror_metrics(name, complete[name], config, state)
                continue
            done = checked_completion(job)
            if done:
                complete[name] = done
                mirror_metrics(name, done, config, state)
                continue
            receipt = state / "launches" / f"{name}.json"
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if process_identity(old["process"]["pid"]) == old["process"]:
                    active[name] = {"process": RecoveredManagedChild(old["process"], job["plan"].run_dir), "gpu": old["gpu"]}
                    continue
            if job["plan"].run_dir.exists():
                raise RuntimeError(f"incomplete child requires inspected recovery, not overwrite: {job['plan'].run_dir}")
        if live_metrics:
            live_metrics.refresh(jobs, complete, active, config, state)
        while len(complete) < len(jobs):
            for name, current in list(active.items()):
                code = current["process"].poll()
                if code is None:
                    continue
                done = checked_completion(jobs[name]) if code == 0 else None
                if done is None:
                    raise RuntimeError(f"managed child failed ({code}): {name}; inspect its run/logs")
                complete[name] = done
                mirror_metrics(name, done, config, state)
                del active[name]
            if config.get("storage", {}).get("mode") == "ram_then_hdd":
                for name in active:
                    if name not in released_cache and release_loaded_model_cache(jobs[name], state, name):
                        released_cache.add(name)
            control = json.loads((state / "control.json").read_text())
            allowed = control["allowed_gpus"]
            if (not allowed or len(set(allowed)) != len(allowed) or any(type(i) is not int or i not in range(8) for i in allowed)
                    or type(control["max_workers"]) is not int or not 1 <= control["max_workers"] <= 4):
                raise ValueError("invalid live GPU control")
            maximum = min(control["max_workers"], config["max_gpu_workers"])
            if smoke and config.get("image_suite") != "image_c5_gpu_v1":
                maximum = 1
            if "storage" in config:
                maximum = min(maximum, config["storage"]["max_workers"])
            snapshot = resource_snapshot()
            now = time.time()
            for index in allowed:
                gpu = snapshot["gpus"].get(index)
                if gpu and not gpu["pids"] and free_device(gpu):
                    idle_since.setdefault(index, now)
                else:
                    idle_since.pop(index, None)
            if live_metrics:
                live_metrics.refresh(jobs, complete, active, config, state)
            atomic_json(state / "status.json", {"status": "draining" if stopping[0] else "paused" if control["paused"] else "running_or_waiting_for_free_gpu",
                "stage": "smoke" if smoke else "train_then_evaluate" if config["mode"] == "fixed" else "search_or_final",
                "active": {n: {"gpu": v["gpu"], "pid": v["process"].pid if hasattr(v["process"], "pid") else None} for n, v in active.items()},
                "completed": list(complete), "total": len(jobs), "time": now})
            if stopping[0] and not active:
                raise SystemExit(75)
            storage_status = storage_admission(config, state)
            if "storage" in config:
                atomic_json(state / "storage_status.json", storage_status)
            enough = (snapshot["available_ram_gib"] >= config["runtime"]["min_ram_gib"]
                and snapshot["load1"] <= 0.8 * snapshot["cpu_affinity_count"]
                and storage_status["ready"])
            if enough and not control["paused"] and not stopping[0]:
                candidates = [i for i in allowed if now - idle_since.get(i, now) >= config["runtime"]["idle_seconds"]
                              and i not in {a["gpu"] for a in active.values()}]
                for name, job in jobs.items():
                    if len(active) >= maximum or not candidates:
                        break
                    if name in complete or name in active or not set(job["dependencies"]) <= complete.keys():
                        continue
                    for source, digest in source_hashes.items():
                        if sha256_file(REPO / source) != digest:
                            raise RuntimeError(f"live workflow source changed: {source}")
                    gpu_index = candidates.pop(0)
                    gpu = resource_snapshot()["gpus"][gpu_index]
                    if gpu["pids"] or not free_device(gpu):
                        continue
                    command = [str(PYTHON), str(REPO / "boot_scripts/run_experiment.py"), "--config", str(job["experiment"]), "--config-root", str(job["bundle"])]
                    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, check=True)
                    atomic_json(state / "launches" / f"{name}.dry_run.json", json.loads(dry.stdout))
                    temporary = (config["paths"]["temporary_root"] if config["mode"] == "fixed"
                                 else "/dev/shm/linbinhao-pulse-hybrid")
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="1",
                        TMPDIR=temporary, PYTHONPYCACHEPREFIX=str(Path(temporary) / "pycache"))
                    for key in ("PULSE_IDLE_CONTEXT_PERMIT", "PULSE_IDLE_CONTEXT_PERMIT_SHA256", "LD_LIBRARY_PATH"):
                        environment.pop(key, None)
                    log_path = state / "launches" / f"{name}.log"
                    with log_path.open("a") as log:
                        child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[name] = {"process": child, "gpu": gpu_index}
                    atomic_json(state / "launches" / f"{name}.json", {"command": command, "gpu": gpu_index,
                        "gpu_uuid": gpu["uuid"], "process": process_identity(child.pid)})
                    # Stagger 7B model loading and recheck host/cgroup RAM
                    # before each additional worker; a single resource snapshot
                    # must not admit four simultaneous loading peaks.
                    break
            if len(complete) < len(jobs):
                time.sleep(config["runtime"]["poll_seconds"])
    except BaseException as error:
        try:
            cleanup = stop_failed_children(active, state)
        except Exception as cleanup_error:
            cleanup = {"signals": [], "survivors": [],
                "skipped": {n: f"cleanup failed: {cleanup_error}" for n in active}}
            atomic_json(state / "failure_cleanup.json", cleanup)
        remaining = {m["job"] for m in cleanup["survivors"]}
        # A skipped, still-live launcher is reported, never falsely cleared.
        remaining |= {n for n in cleanup["skipped"] if active[n]["process"].poll() is None}
        atomic_json(state / "status.json", {"status": "failed_or_interrupted",
            "error": f"{type(error).__name__}: {error}", "completed": list(complete),
            "total": len(jobs), "time": time.time(),
            "active": {n: {"gpu": active[n]["gpu"], "pid": active[n]["process"].pid} for n in remaining},
            "failure_cleanup": str(state / "failure_cleanup.json")})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return complete


def mirror_metrics(name, entry, config, state):
    """Tracking is optional telemetry: a failed mirror never discards the experiment."""
    receipt = state / "tracking" / f"{name}.json"
    # Preserve recipe/result hashes even when a screening checkpoint lives in RAM.
    atomic_json(state / "child_evidence" / f"{name}.json", {"path": entry["path"], "sha256": entry["sha256"],
        "result": entry["result"], "checkpoint_retention": "volatile_screening" if name.startswith("trial_") else "durable"})
    if config["mode"] == "fixed":
        return
    if receipt.exists() and json.loads(receipt.read_text()).get("source_sha256") == entry["sha256"]:
        return
    command = [config["tracking"]["python"], "-m", "util.pulse_tracking", "--result", entry["path"],
        "--storage", str(state.parent / "trackio"), "--name", name, "--project", config["tracking"]["project"]]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRACKIO_DIR=str(state.parent / "trackio"),
        HF_HUB_OFFLINE="1", DO_NOT_TRACK="1")
    environment.pop("LD_LIBRARY_PATH", None)
    try:
        result = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        atomic_json(receipt, {"source_sha256": None, "error": str(error)})
        return
    atomic_json(receipt, {"source_sha256": entry["sha256"] if result.returncode == 0 else None,
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})


def run(config, refs, config_path, root, output):
    from util.pulse_training_contract import load_config as load_training
    from util.evaluation.ecg_image_elastic import process_identity
    import optuna
    template, _ = load_training(refs["training_config"], root)
    durable = Path(config["paths"]["run_root"])
    state = durable / "state"
    output.mkdir(parents=True, exist_ok=False)
    state.mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["temporary_root"]).mkdir(parents=True, exist_ok=True)
    sources = {p.relative_to(REPO).as_posix(): sha256_file(p)
               for directory in ("core", "util", "boot_scripts", "data_preprocess", "models")
               for p in (REPO / directory).rglob("*.py") if "tests" not in p.parts}
    identity = {"config": digest_json(config), "sources": sources}
    with exclusive_lock(state / "coordinator.lock"):
        identity_path = state / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("workflow source/config changed; new study or re-admission required")
        atomic_json(identity_path, identity)
        for source, digest in sources.items():
            destination = state / "source_snapshot" / source
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / source, destination)
            if sha256_file(destination) != digest:
                raise RuntimeError("workflow source snapshot changed")
        if not (state / "control.json").exists():
            atomic_json(state / "control.json", {"paused": False, "max_workers": 4, "allowed_gpus": list(range(8))})
        atomic_json(state / "coordinator.json", {"process": process_identity(os.getpid()), "output": str(output)})
        smoke_jobs = make_jobs(config, root, template, phase="smoke", recipe=None)
        smoke = run_jobs(smoke_jobs, config, state, sources, smoke=True)
        recovery = Path(smoke["smoke_ningbo_resume"]["path"]).parent / "recovery_audit.json"
        if not json.loads(recovery.read_text())["process_resume_verified"]:
            raise RuntimeError("fresh-process training resume was not verified")
        study = open_study(state, seed=config["seed"])
        if not study.trials:
            study.enqueue_trial({"learning_rate": 1e-5, "projector_learning_rate": 1e-5,
                "jsd_weight": 12.0, "waveform_strength": 0.5, "image_strength": 0.5})
        trial_summaries = []
        for number in range(config["trials"]):
            # Per-trial sampler seed makes restart independent of in-memory RNG state.
            study = open_study(state, seed=config["seed"] + number)
            trials = study.get_trials()
            previous = trials[number] if len(trials) > number else None
            if previous and previous.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.COMPLETE):
                recipe = previous.params
                if len(recipe) != 5:
                    raise RuntimeError("partially proposed Optuna trial requires explicit recovery")
            else:
                trial = study.ask()
                if trial.number != number:
                    raise RuntimeError("unexpected Optuna trial numbering")
                recipe = suggest_recipe(trial, config["search_space"])
            jobs = make_jobs(config, root, template, phase="screen", recipe=recipe, trial_number=number)
            complete = run_jobs(jobs, config, state, sources)
            arms = {a: {} for a in config["arms"]}
            evaluations = {}
            for center in CENTERS:
                entry = complete[f"trial_{number:02d}_{center}_eval"]
                evaluations[center] = {k: entry[k] for k in ("path", "sha256")}
                for arm in arms:
                    arms[arm][center] = entry["result"]["details"]["families"][arm]
            score, arm_scores = matched_trial_score(arms)
            if previous and previous.state == optuna.trial.TrialState.COMPLETE:
                if previous.value != score:
                    raise RuntimeError("completed Optuna score no longer matches prediction evidence")
            else:
                study.tell(number, score)
            summary = {"number": number, "recipe": recipe, "score": score, "arm_scores": arm_scores,
                       "evaluations": evaluations, "development_only": True}
            atomic_json(state / f"trial_{number:02d}.json", summary)
            trial_summaries.append(summary)
        selected = study.best_trial
        atomic_json(state / "selected_recipe.json", {"number": selected.number, "params": selected.params,
            "value": selected.value, "selection": "PN2021-development", "not_independent_test": True})
        jobs = make_jobs(config, root, template, phase="final", recipe=selected.params)
        complete = run_jobs(jobs, config, state, sources)
        finals = {c: {k: complete[f"final_{c}_eval"][k] for k in ("path", "sha256")} for c in CENTERS}
        from util.evaluation.pulse_hybrid_development import summarize_four_centers
        summarize_four_centers(finals, output, seed=config["seed"])
        atomic_json(output / "trial_summaries.json", trial_summaries)
        atomic_json(output / "final_results.json", finals)
        atomic_json(output / "source_identity.json", identity)
        finalize(output, mode="search", details={"trials": trial_summaries, "final_results": finals,
            "selected_recipe": selected.params, "development_only": True})
        atomic_json(state / "status.json", {"status": "complete", "result": str(output / "hybrid_result.json")})


def prepare_fixed_recovery(config, template, source_hashes, state):
    """Reuse admissions from a pinned archive chain and copy current checkpoints."""
    recovery = config["recovery"]
    block_evaluation = config.get("evaluation_block_size") == 512
    revised_evaluation = block_evaluation or "inference_batch_size" in config or "sampling_method" in config
    evaluation_source = "util/evaluation/pulse_hybrid_development.py"
    inference_source = "util/evaluation/pulse_adapters.py"
    archive = Path(recovery["archive_root"]).resolve()

    def verified(relative, context):
        archive_path, receipt = context
        parent = (archive_path / "run").resolve()
        parent.relative_to(archive_path)
        path = (parent / relative).resolve()
        path.relative_to(parent)
        entry = receipt["files"].get(relative)
        if (not entry or not path.is_file() or path.stat().st_uid != os.getuid()
                or sha256_file(path) != entry["sha256"]):
            raise ValueError(f"recovery artifact changed: {relative}")
        return path

    def scientific(value):
        value = copy.deepcopy(value)
        value.pop("recovery", None)
        value.pop("evaluation_block_size", None)
        if revised_evaluation and not block_evaluation:
            for field in ("inference_batch_size", "sampling_method", "records_per_center"):
                value.pop(field, None)
        value["paths"].pop("run_root")
        value["paths"].pop("temporary_root")
        value["storage"].pop("archive_root")
        return value

    contexts, pins, visited, changed = [], [], set(), set()
    reference = recovery
    orchestration = {"util/pulse_hybrid_workflow.py", "util/pulse_hybrid_contract.py"}
    allowed_changes = orchestration | ({evaluation_source, inference_source} if revised_evaluation else set())
    while reference is not None:
        location = Path(reference["archive_root"]).resolve()
        if location.parent != archive.parent or location in visited:
            raise ValueError("recovery archive chain escapes its namespace or cycles")
        visited.add(location)
        receipt_path = location / "archive_receipt.json"
        if (not receipt_path.is_file() or receipt_path.stat().st_uid != os.getuid()
                or sha256_file(receipt_path) != reference["archive_receipt_sha256"]):
            raise ValueError("recovery archive receipt changed")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema_version") != 1 or receipt.get("status") != "verified"
                or receipt.get("source_status") != "failed"):
            raise ValueError("recovery requires a verified failed-run archive")
        context = (location, receipt)
        previous_config = "configs/train/pulse_full_lora32_pipeline.yaml"
        if "run_manifest.json" in receipt["files"]:
            manifest = json.loads(verified("run_manifest.json", context).read_text())
            snapshots = [entry.get("snapshot_path") for entry in manifest.get("config_snapshots", [])
                         if entry.get("role") == "delegate_entry_config"]
            if len(snapshots) != 1 or not isinstance(snapshots[0], str):
                raise ValueError("recovery archive lacks one delegate config snapshot")
            candidate = Path(snapshots[0])
            if (candidate.is_absolute() or ".." in candidate.parts
                    or candidate.parts[:2] != ("configs", "train")):
                raise ValueError("recovery delegate config snapshot is outside configs/train")
            previous_config = candidate.as_posix()
        previous = yaml.safe_load(verified(previous_config, context).read_text())
        if scientific(previous) != scientific(config):
            raise ValueError("recovery changed the scientific workflow")
        old_template = yaml.safe_load(verified("configs/train/pulse_full_lora32_template.yaml", context).read_text())
        old_template["paths"]["temporary_root"] = template["paths"]["temporary_root"]
        if old_template != template:
            raise ValueError("recovery changed the training recipe")
        old_sources = json.loads(verified("state/identity.json", context).read_text())["sources"]
        drift = {name for name in old_sources.keys() | source_hashes.keys()
                 if old_sources.get(name) != source_hashes.get(name)}
        if drift - allowed_changes:
            raise ValueError(f"recovery changed model/data/evaluation code: {sorted(drift - allowed_changes)}")
        changed.update(drift)
        contexts.append(context)
        pins.append({"archive_root": str(location),
                     "archive_receipt_sha256": reference["archive_receipt_sha256"]})
        reference = previous.get("recovery")

    primary = contexts[0]
    recorded = json.loads(verified("workflow/smoke_results.json", primary).read_text())
    expected = {f"smoke_{center}_{kind}" for center in CENTERS
                for kind in ("single", "three", "resume", "eval")}
    if set(recorded) != expected:
        raise ValueError("recovery lacks complete four-center smoke admissions")
    smoke, checkpoints, completed = {}, {}, {}
    for name, entry in recorded.items():
        original = Path(entry["path"])
        matched = None
        for context in contexts:
            location, receipt = context
            for anchor in (Path(receipt["source_run_root"]), location / "run"):
                try:
                    relative = original.relative_to(anchor).as_posix()
                except ValueError:
                    continue
                matched = (context, relative)
                break
            if matched is not None:
                break
        if matched is None:
            raise ValueError(f"smoke reference is outside the pinned archive chain: {name}")
        context, relative = matched
        path = verified(relative, context)
        result = json.loads(path.read_text())
        if sha256_file(path) != entry["sha256"] or result.get("status") != "complete":
            raise ValueError(f"recovery smoke is incomplete: {name}")
        if name.endswith("_resume"):
            verified(str(Path(relative).parent / "recovery_audit.json"), context)
        smoke[name] = {"path": str(path), "sha256": entry["sha256"], "result": result}
    # A changed evaluator never inherits old numerical admissions. Training
    # reuse still requires exact numerical source and scientific recipe identities.
    fresh_smoke_required = revised_evaluation and any(
        entry["result"].get("files", {}).get("source_snapshot/" + source)
        != source_hashes.get(source)
        for key, entry in smoke.items() if key.endswith("_eval")
        for source in (evaluation_source, inference_source))
    if "inference_batch_size" in config or "sampling_method" in config:
        fresh_smoke_required = True

    def completed_child(context, relative, entrypoint):
        from types import SimpleNamespace
        manifest_relative = f"{relative}/run_manifest.json"
        if manifest_relative not in context[1]["files"]:
            return None
        manifest = json.loads(verified(manifest_relative, context).read_text())
        if manifest.get("status") != "complete":
            return None
        result_relative = ("training/train_result.json" if entrypoint == "train_ecg_image"
                           else "evaluation/hybrid_result.json")
        verified(f"{relative}/run_file_index.json", context)
        path = verified(f"{relative}/{result_relative}", context)
        plan = SimpleNamespace(run_dir=context[0] / "run" / relative,
            expected_result_relative_path=result_relative, entrypoint_name=entrypoint)
        entry = checked_completion({"plan": plan})
        if entry is None or entry["sha256"] != sha256_file(path):
            raise ValueError(f"recovery completed child changed: {relative}")
        return entry

    for center in CENTERS:
        for arm in ("single", "three"):
            name = f"{center}_{arm}"
            for context in contexts:
                entry = completed_child(context, f"final/{name}", "train_ecg_image")
                if entry is None:
                    continue
                result = entry["result"]
                protocol = result["protocol"]
                if (result.get("mode") != "train" or protocol.get("center") != center
                        or protocol.get("width") != {"single": 1, "three": 3}[arm]
                        or result.get("optimizer_steps") != template["training"]["optimizer_steps"]
                        or protocol.get("training") != template["training"]
                        or protocol.get("model") != template["model"]
                        or protocol.get("training_scope") != template["training_scope"]
                        or protocol.get("image_augmentation") != template["image_augmentation"]
                        or any(source_hashes.get(k) != v
                               for k, v in protocol["implementation_sha256"].items())):
                    raise ValueError(f"recovery completed training identity changed: {name}")
                completed[f"final_{name}"] = entry
                break
            if f"final_{name}" in completed:
                continue
            relative = f"final/{name}/training/checkpoint.pt"
            if relative not in primary[1]["files"]:
                continue
            path = verified(relative, primary)
            protocol = json.loads(verified(f"final/{name}/training/protocol.json", primary).read_text())
            if any(source_hashes.get(k) != v for k, v in protocol["implementation_sha256"].items()):
                raise ValueError(f"checkpoint numerical implementation changed: {name}")
            destination = state.parent / "recovery" / name / "checkpoint.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 8 * 1024**2)
            if sha256_file(destination) != primary[1]["files"][relative]["sha256"]:
                raise ValueError(f"recovery checkpoint copy changed: {name}")
            checkpoints[name] = str(destination)
    skipped_evaluations = []
    for center in CENTERS:
        training_keys = [f"final_{center}_{arm}" for arm in ("single", "three")]
        if not all(key in completed for key in training_keys):
            continue
        expected_training = {arm: completed[f"final_{center}_{arm}"]["sha256"]
                             for arm in ("single", "three")}
        for context in contexts:
            entry = completed_child(context, f"final/{center}_eval", "pulse_hybrid")
            if entry is None:
                continue
            result = entry["result"]
            details = result["details"]
            if (result.get("mode") != "evaluate" or details.get("phase") != "final"
                    or details.get("center") != center
                    or details.get("image_suite") != config["image_suite"]
                    or details.get("image_severity") != config["image_severity"]
                    or details.get("evaluation_seed") != config["seed"]
                    or details.get("inference_batch_size", 1) != config.get("inference_batch_size", 1)
                    or details.get("sampling_method") != config.get("sampling_method")
                    or (config["records_per_center"] != "full" and details.get("records") != config["records_per_center"])
                    or details.get("model_arms") != ["original", "single", "three"]
                    or details.get("full_cohort") != (config["records_per_center"] == "full")):
                raise ValueError(f"recovery completed evaluation identity changed: {center}")
            if block_evaluation and (
                    result["files"].get("source_snapshot/" + evaluation_source) != source_hashes[evaluation_source]
                    or result["files"].get("source_snapshot/" + inference_source) != source_hashes[inference_source]):
                skipped_evaluations.append({"path": entry["path"],
                    "reason": "changed evaluator or inference adapter requires fresh admission and inference"})
                continue
            if details.get("training_results") != expected_training:
                skipped_evaluations.append({"path": entry["path"],
                    "reason": "different immutable training result identities"})
                continue
            completed[f"final_{center}_eval"] = entry
            break
    if block_evaluation:
        for center in CENTERS:
            if f"final_{center}_eval" in completed:
                continue
            training_keys = [f"final_{center}_{arm}" for arm in ("single", "three")]
            if not all(key in completed for key in training_keys):
                continue
            expected_training = {arm: completed[f"final_{center}_{arm}"]["sha256"]
                                 for arm in ("single", "three")}
            for span in center_block_spans(center):
                suffix = f"{span[0]:05d}_{span[1]:05d}"
                for context in contexts:
                    entry = completed_child(context, f"final/{center}_eval_blocks/{suffix}", "pulse_hybrid")
                    if entry is None:
                        continue
                    result, details = entry["result"], entry["result"]["details"]
                    if (result.get("mode") != "evaluate" or details.get("phase") != "block"
                            or details.get("center") != center or details.get("record_span") != span
                            or details.get("full_cohort") is not False
                            or details.get("image_suite") != config["image_suite"]
                            or details.get("image_severity") != config["image_severity"]
                            or details.get("evaluation_seed") != config["seed"]
                            or details.get("model_arms") != ["original", "single", "three"]):
                        raise ValueError(f"recovery completed block identity changed: {center}/{suffix}")
                    if (details.get("training_results") != expected_training
                            or result["files"].get("source_snapshot/" + evaluation_source)
                            != source_hashes[evaluation_source]
                            or result["files"].get("source_snapshot/" + inference_source)
                            != source_hashes[inference_source]):
                        skipped_evaluations.append({"path": entry["path"],
                            "reason": "block training, evaluator or inference adapter source identity differs"})
                        continue
                    completed[f"final_{center}_eval_{suffix}"] = entry
                    break
    atomic_json(state / "recovery_admission.json", {"archive": str(archive),
        "archive_receipt_sha256": recovery["archive_receipt_sha256"],
        "pinned_archive_chain": pins,
        "reused_smoke_count": 0 if fresh_smoke_required else len(smoke),
        "fresh_smoke_required": fresh_smoke_required, "resume_checkpoints": checkpoints,
        "reused_completed": {name: {k: entry[k] for k in ("path", "sha256")}
                              for name, entry in completed.items()},
        "skipped_evaluations": skipped_evaluations,
        "changed_orchestration_only": sorted(changed & orchestration),
        "changed_evaluation_sources": sorted(changed - orchestration),
        "training_sources_unchanged": True, "scientific_sources_unchanged": not bool(changed - orchestration)})
    return ({} if fresh_smoke_required else smoke), checkpoints, completed


def run_fixed(config, refs, config_path, root, output):
    """Finite smoke -> eight matched trains -> four C15 evaluations -> report."""
    from util.pulse_training_contract import load_config as load_training
    from util.pulse_training_queue import allowed_gpus
    from util.evaluation.ecg_image_elastic import process_identity
    from util.evaluation.pulse_hybrid_development import summarize_four_centers, merge_center_blocks

    candidates = allowed_gpus()
    full_lora = str(config["references"].get("training_config", "")).endswith("pulse_full_lora32_template.yaml")
    if (not candidates or len(candidates) > 8 or (len(candidates) != 4 and not full_lora)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("coordinator requires an explicit GPU allowlist and at most four workers")
    template, _ = load_training(refs["training_config"], root)
    durable = Path(config["paths"]["run_root"])
    state = durable / "state"
    state.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(state / "coordinator.lock"):
        output.mkdir(parents=True, exist_ok=False)
        Path(config["paths"]["temporary_root"]).mkdir(parents=True, exist_ok=True)
        if "storage" in config:
            admission = storage_admission(config, state, initial=True)
            atomic_json(state / "storage_admission.json", admission)
            if not admission["ready"]:
                raise RuntimeError(f"RAM/HDD resource admission failed: {admission}")
        sources = {p.relative_to(REPO).as_posix(): sha256_file(p)
                   for directory in ("core", "util", "boot_scripts", "data_preprocess", "models")
                   for p in (REPO / directory).rglob("*.py") if "tests" not in p.parts}
        identity = {"config": digest_json(config), "sources": sources}
        identity_path = state / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("fixed workflow source/config changed; new admission is required")
        atomic_json(identity_path, identity)
        for relative, digest in sources.items():
            destination = state / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(REPO / relative, destination)
            if sha256_file(destination) != digest:
                raise RuntimeError("fixed workflow source snapshot changed")
        if not (state / "control.json").exists():
            atomic_json(state / "control.json", {"paused": False,
                "max_workers": config.get("storage", {}).get("max_workers", 4), "allowed_gpus": candidates})
        atomic_json(state / "coordinator.json", {"process": process_identity(os.getpid()), "output": str(output)})
        try:
            resume_checkpoints, precompleted = None, {}
            smoke = {}
            if config.get("recovery"):
                smoke, resume_checkpoints, precompleted = prepare_fixed_recovery(config, template, sources, state)
            if not smoke:
                smoke = run_jobs(make_jobs(config, root, template, phase="smoke", recipe=None),
                                 config, state, sources, smoke=True)
            for name, entry in smoke.items():
                if not name.endswith("_resume"):
                    continue
                recovery = json.loads((Path(entry["path"]).parent / "recovery_audit.json").read_text())
                if not recovery["process_resume_verified"] or recovery["resume_max_trainable_delta"] > 1e-7:
                    raise RuntimeError(f"reference training did not pass exact fresh-process resume: {name}")
            atomic_json(output / "smoke_results.json", {n: {k: v[k] for k in ("path", "sha256")} for n, v in smoke.items()})
            jobs = make_jobs(config, root, template, phase="final", recipe=None,
                             resume_checkpoints=resume_checkpoints, precompleted=precompleted)
            complete = run_jobs(jobs, config, state, sources, precompleted=precompleted)
            if config.get("evaluation_block_size") == 512:
                finals = {}
                for center in CENTERS:
                    if f"final_{center}_eval" in complete:
                        finals[center] = {key: complete[f"final_{center}_eval"][key] for key in ("path", "sha256")}
                        continue
                    names = [name for name, job in jobs.items()
                             if job.get("merge_config", {}).get("center") == center]
                    entries = [{key: complete[name][key] for key in ("path", "sha256")} for name in names]
                    finals[center] = merge_center_blocks(entries, jobs[names[0]]["merge_config"],
                        output / "merged_evaluations" / center)
            else:
                finals = {c: {k: complete[f"final_{c}_eval"][k] for k in ("path", "sha256")} for c in CENTERS}
            training = {f"{c}_{a}": {k: complete[f"final_{c}_{a}"][k] for k in ("path", "sha256")}
                        for c in CENTERS for a in ("single", "three")}
            families = ("clean", "image") if config["image_suite"] == "image_c5_gpu_v1" else (
                "clean", "waveform", "image", "joint")
            summarize_four_centers(finals, output, seed=config["seed"],
                families=families, arms=("original", "single", "three"))
            live = None
            if config.get("evaluation_block_size") == 512:
                from util.evaluation.pulse_hybrid_development import FULL_CENTER_RECORDS
                live_path = state / "live_metrics.json"
                live = json.loads(live_path.read_text())
                if live["progress"]["completed_records"] != sum(FULL_CENTER_RECORDS.values()):
                    raise RuntimeError("live metrics do not cover the complete evaluation cohort")
                for center in CENTERS:
                    final_metrics = json.loads((Path(finals[center]["path"]).parent / "metrics.json").read_text())
                    observed = live["metrics"]["per_center"][center]["arms"]
                    for arm in ("original", "single", "three"):
                        for family in families:
                            if abs(observed[arm][family] - final_metrics["families"][arm][family]) > 1e-12:
                                raise RuntimeError(f"live metric differs from final recomputation: {center}/{arm}/{family}")
            atomic_json(output / "training_results.json", training)
            atomic_json(output / "final_results.json", finals)
            atomic_json(output / "source_identity.json", identity)
            finalize(output, mode="fixed", details={"final_results": finals,
                "model_arms": ["original", "single", "three"], "development_only": True,
                "selection": "fixed_recipe_last_checkpoint_no_hpo",
                "image_suite": config["image_suite"],
                "image_severity": config["image_severity"],
                **({k: config[k] for k in ("inference_batch_size", "sampling_method") if k in config}),
                "records_per_center": config["records_per_center"]})
            if live is not None:
                live.update(status="complete", provisional=False,
                    final_results=finals, final_comparison=str(output / "comparison.json"),
                    metric_recompute_matches_final=True, updated_at=time.time())
                atomic_json(state / "live_metrics.json", live)
            atomic_json(state / "status.json", {"status": "complete", "result": str(output / "hybrid_result.json")})
        except BaseException as error:
            previous = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
            atomic_json(state / "status.json", {**previous, "status": "failed_or_interrupted",
                "error": f"{type(error).__name__}: {error}", "time": time.time()})
            raise


def center_block_spans(center):
    from util.evaluation.pulse_hybrid_development import FULL_CENTER_RECORDS
    total = FULL_CENTER_RECORDS[center]
    return [[start, min(start + 512, total)] for start in range(0, total, 512)]
