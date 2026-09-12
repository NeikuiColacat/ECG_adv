"""Finite three-arm subset; preserve the original full queue and its sources.

Pair tasks reuse the frozen batch2 owner unchanged. Original-model tasks use
the author's generate with the original projector, never an adapter-disabled
fine-tuned projector. Both lanes share the same samples, corruption and image
owners; input fingerprints must agree before any comparative metrics publish.
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

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.evaluation.ecg_image_elastic import free_device, process_identity, resource_snapshot, gpu_allowlist
from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, digest_json, exclusive_lock
from util.evaluation.pulse_benchmark import (REPO, PULSE_PYTHON, claim_gpu, make_renderer,
    infer_batch, profile_and_baseline, source_identity, verify_sources, snapshot_sources)
from util.pn2021_artifact_contract import sha256_file
from util.pulse_benchmark_contract import ARMS, MODEL_NAMES, validate_pair_row
from util.pulse_training_contract import CENTERS, CLASS_ORDER

NAMES = ("PULSE-original", *MODEL_NAMES)
EXTRA_SOURCES = ("util/evaluation/pulse_subset.py", "boot_scripts/evaluate_pulse_subset.py",
                 "boot_scripts/run_experiment.py", "util/run_record.py",
                 "util/evaluation/ecg_image_elastic.py")


def validate_config(config):
    if set(config) != {"schema_version", "references", "source_state", "records_per_center", "runtime"}:
        raise ValueError("invalid subset config keys")
    if config["schema_version"] not in (1, 2) or set(config["references"]) != {"paired_config"}:
        raise ValueError("invalid subset schema/reference")
    if type(config["records_per_center"]) is not int or config["records_per_center"] not in (256, 384, 512):
        raise ValueError("subset count must be frozen before evaluation")
    pixel = config["schema_version"] == 2
    if pixel and config["records_per_center"] != 512:
        raise ValueError("pixel comparison uses the completed fixed 512 cohort")
    if config["runtime"] != {"initial_workers": 4 if pixel else 2, "max_workers": 4, "expand_after_seconds": 0 if pixel else 900,
            "idle_seconds": 60, "poll_seconds": 60, "max_wall_seconds": 21600}:
        raise ValueError("subset runtime exceeds approved bounds")
    Path(config["source_state"]).resolve().relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))


def selected_tasks(tasks, count):
    selected = []
    for center in CENTERS:
        group = [t for t in tasks if t["center"] == center][:count // 8]
        keys = [k for t in group for batch in t["batches"] for k in batch]
        if len(keys) != count or len(set(keys)) != count or any(len(b) != 2 for t in group for b in t["batches"]):
            raise ValueError("subset must preserve complete original batch2 tasks")
        selected.extend(group)
    return selected


def prepare(config, paired_config, output):
    if config["schema_version"] == 2:
        return prepare_pixel(config, paired_config, output)
    from util.evaluation.pulse_adapters import matched_training_evidence
    source = Path(config["source_state"])
    if not json.loads((source / "control.json").read_text())["paused"]:
        raise ValueError("full queue must remain paused")
    old_identity = json.loads((source / "identity.json").read_text())
    verify_sources(old_identity["source_identity"])
    if old_identity["config"] != paired_config:
        raise ValueError("subset references a different original evaluation config")
    original = json.loads((source / "prepared.json").read_text())
    tasks = selected_tasks(original["tasks"], config["records_per_center"])
    keys = {k for t in tasks for b in t["batches"] for k in b}
    rows = [r for r in original["rows"] if r["sample_key"] in keys]
    if len(rows) != 4 * config["records_per_center"]:
        raise ValueError("foreign or duplicated subset sample")
    for c in CENTERS:
        evidence = matched_training_evidence(paired_config["training_results"][c])
        for arm, item in evidence.items():
            if item["sha256"] != original["training"][c][arm]["sha256"]:
                raise ValueError("frozen training result changed")
        k500 = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())
        if {r["hash_id"] for r in rows if r["logical_center"] == c} & {r["hash_id"] for r in k500}:
            raise ValueError("K500 leakage")
    sources = {**source_identity(), **{n: sha256_file(REPO / n) for n in EXTRA_SOURCES}}
    identity = digest_json({"config": config, "sources": sources, "parent": original["protocol_identity"],
                            "sample_keys": sorted(keys), "baseline": "original_projector_author_generate_batch2"})
    base_tasks = []
    for t in tasks:
        task = {**t, "protocol_identity": identity}
        task["id"] = digest_json({"parent_task": t["id"], "protocol": identity, "arm": "original"})
        base_tasks.append(task)
    prepared = {**original, "rows": rows, "tasks": tasks, "base_tasks": base_tasks,
                "parent_identity": original["protocol_identity"], "subset_identity": identity,
                "sources": sources, "subset_config": config, "paired_config": paired_config}
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output, sources)
    atomic_json(output / "prepared.json", prepared)
    atomic_json(output / "cohort.json", rows)
    atomic_json(output / "data_audit.json", {**original["audit"], "full_cohort": False,
        "records_per_center": config["records_per_center"], "selection": "frozen_parent_prefix_no_label_selection"})
    pair_queue = TaskQueue(output / "pair_queue", tasks)
    TaskQueue(output / "base_queue", base_tasks)
    by_key = {r["sample_key"]: r for r in rows}
    conditions = {c["condition_id"]: c for c in original["conditions"]}
    reused = []
    for task in tasks:
        path = source / "queue/results" / f"{task['id']}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["task_id"] != task["id"] or payload["protocol_identity"] != task["protocol_identity"]:
            raise ValueError("reuse task identity mismatch")
        TaskQueue.validate_rows(task, payload["predictions"])
        if digest_json(payload["predictions"]) != payload["rows_sha256"]:
            raise ValueError("reuse content fingerprint mismatch")
        for row in payload["predictions"]:
            validate_pair_row(row, by_key[row["sample_key"]], conditions[row["condition_id"]])
        shutil.copyfile(path, pair_queue.result_path(task))
        reused.append({"task_id": task["id"], "path": str(path), "sha256": sha256_file(path)})
    atomic_json(output / "reuse.json", {"pair_tasks": reused, "historical_zero_shot_reused": False})
    return prepared


def validate_pixel_parent(original, paired_config, source_snapshot=None):
    """Only training weights/domain and output locations may change."""
    changed = {"core/augmix.py", "core/pulse_finetune.py", "util/pulse_training_contract.py",
               "util/evaluation/pulse_subset.py"}
    for name, sha in original["sources"].items():
        if name not in changed and sha256_file(REPO / name) != sha:
            # This approved handoff changes GPU admission only. Prove all
            # inference/data/metric AST nodes outside these helpers unchanged.
            import ast
            helpers = {"util/evaluation/ecg_image_elastic.py": {"free_device", "authorized_idle_contexts"},
                       "util/evaluation/pulse_benchmark.py": {"claim_gpu"}}
            if name not in helpers or source_snapshot is None:
                raise ValueError(f"reused baseline evaluation source changed: {name}")
            previous = Path(source_snapshot) / name
            if sha256_file(previous) != sha:
                raise ValueError("parent GPU-admission source snapshot changed")
            def non_admission_ast(filename):
                tree = ast.parse(filename.read_text())
                tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in helpers[name])]
                return ast.dump(tree, include_attributes=False)
            if non_admission_ast(previous) != non_admission_ast(REPO / name):
                raise ValueError(f"reused baseline changed beyond GPU admission: {name}")
    for name, sha in original["audit"]["common_implementation_sha256"].items():
        if sha256_file(REPO / name) != sha:
            raise ValueError(f"reused baseline data/render implementation changed: {name}")
    old, new = original["paired_config"], paired_config
    if set(old) != set(new):
        raise ValueError("pixel evaluation config shape changed")
    for key in old:
        if key not in {"training_results", "admission_training", "paths"} and old[key] != new[key]:
            raise ValueError(f"pixel evaluation changed frozen field: {key}")
    if ({k:v for k,v in old["paths"].items() if k not in {"state_root", "temporary_root"}}
            != {k:v for k,v in new["paths"].items() if k not in {"state_root", "temporary_root"}}):
        raise ValueError("pixel evaluation changed raw data/toolkit paths")


def matched_pixel_protocol(old, new):
    ignored = {"implementation_sha256", "mixing_domain"}
    if (new.get("mixing_domain") != "rendered_rgb" or "mixing_domain" in old
            or {k:v for k,v in old.items() if k not in ignored}
            != {k:v for k,v in new.items() if k not in ignored}):
        raise ValueError("pixel training changed more than the mixing domain")


def prepare_pixel(config, paired_config, output):
    from util.evaluation.pulse_adapters import matched_training_evidence
    source = Path(config["source_state"])
    parent_path = source / "subset_result.json"
    validate_result(json.loads(parent_path.read_text()), parent_path)
    if not json.loads((source / "control.json").read_text())["paused"]:
        raise ValueError("completed parent subset must be frozen")
    original = json.loads((source / "prepared.json").read_text())
    validate_pixel_parent(original, paired_config, source / "source_snapshot")
    if original["subset_config"]["records_per_center"] != config["records_per_center"]:
        raise ValueError("pixel evaluation must reuse the identical cohort")
    training, overlaps = {}, {}
    for center in CENTERS:
        training[center], overlaps[center] = {}, {}
        evidence = matched_training_evidence(paired_config["training_results"][center])
        test = [r for r in original["rows"] if r["logical_center"] == center]
        for arm, item in evidence.items():
            protocol = item["result"]["protocol"]
            matched_pixel_protocol(original["training"][center][arm]["protocol"], protocol)
            k500 = json.loads((item["path"].parent / "k500_records.json").read_text())
            for key in ("hash_id", "source_record"):
                if {r[key] for r in test} & {r[key] for r in k500}:
                    raise ValueError(f"pixel K500 leakage: {center}/{arm}/{key}")
            overlaps[center][arm] = 0
            training[center][arm] = {"path": str(item["path"]), "sha256": item["sha256"], "protocol": protocol}
    sources = {**source_identity(), **{n: sha256_file(REPO / n) for n in EXTRA_SOURCES}}
    parent = {"result": str(parent_path), "sha256": sha256_file(parent_path)}
    identity = digest_json({"config": config, "sources": sources, "parent": parent, "training": training})
    tasks = [{**t, "protocol_identity": identity,
              "id": digest_json({"parent_task": t["id"], "protocol": identity, "arm": "rendered_rgb_pair"})}
             for t in original["tasks"]]
    prepared = {**original, "tasks": tasks, "training": training, "k500_overlap": overlaps,
                "parent_identity": original["subset_identity"], "protocol_identity": identity,
                "subset_identity": identity, "sources": sources, "subset_config": config,
                "paired_config": paired_config, "pixel_parent": parent}
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output, sources)
    atomic_json(output / "prepared.json", prepared)
    atomic_json(output / "cohort.json", prepared["rows"])
    audit = {**original["audit"], "full_cohort": False, "records_per_center": 512,
        "selection": "identical_completed_512_development_subset", "k500_overlap": overlaps,
        "centers": {c: {"selected": 512, "ref_excluded": True,
            "selection_hash_id_set_sha256": digest_json(sorted(r["hash_id"] for r in prepared["rows"] if r["logical_center"] == c))}
            for c in CENTERS}}
    atomic_json(output / "data_audit.json", audit)
    TaskQueue(output / "pair_queue", tasks)
    base = TaskQueue(output / "base_queue", prepared["base_tasks"])
    old_base = TaskQueue(source / "base_queue", original["base_tasks"])
    reused = []
    for task in base.tasks:
        old_base.validate_completed(task)
        src = old_base.result_path(task)
        shutil.copyfile(src, base.result_path(task))
        reused.append({"task_id": task["id"], "path": str(src), "sha256": sha256_file(src)})
    admissions = [json.loads(p.read_text()) for p in (source / "workers").glob("*.admission.json")]
    if not admissions or any(a["status"] != "passed" for a in admissions) or not any(a["kind"] == "base" for a in admissions):
        raise ValueError("parent original-model generation admission missing")
    reference = {k: admissions[0]["audit"][k] for k in ("prompt", "generation")}
    if any({k:a["audit"][k] for k in reference} != reference for a in admissions):
        raise ValueError("parent generation admissions differ")
    atomic_json(output / "reference_generation.json", reference)
    atomic_json(output / "reuse.json", {"pair_tasks": [], "original_tasks": reused,
        "matched_completed_original_reused": True, "historical_zero_shot_reused": False, "parent": parent})
    return prepared


class OriginalBackend:
    """Pristine source model: remove zero-initialized adapters before GPU use."""
    def __init__(self, protocol, reservation):
        import torch
        from core.pulse_finetune import load_pulse_for_training
        from safetensors import safe_open
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from util.evaluation.ecg_image_data import QUESTION
        self.tokenizer, peft, loading = load_pulse_for_training(protocol)
        self.base = peft.unload()  # No fine-tuned checkpoint was loaded; no merge.
        if any("lora_" in n for n, _ in self.base.named_parameters()):
            raise ValueError("original model still has adapter branches")
        directory = Path(protocol["model"]["directory"])
        weights = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        projector_checks = {}
        for name, parameter in self.base.named_parameters():
            if ".mm_projector." in name:
                with safe_open(directory / weights[name], framework="pt", device="cpu") as handle:
                    expected = handle.get_tensor(name).to(torch.float16).float()
                if not torch.equal(parameter.detach().cpu(), expected):
                    raise ValueError("original projector differs from source checkpoint")
                projector_checks[name] = True
        if not projector_checks:
            raise ValueError("original projector audit missing")
        reservation.clear()
        torch.cuda.empty_cache()
        self.base.to("cuda").requires_grad_(False).eval()
        self.base.gradient_checkpointing_disable()
        self.base.disable_input_require_grads()
        self.base.config.use_cache = True
        conversation = conv_templates["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION)
        conversation.append_message(conversation.roles[1], None)
        self.prompt = conversation.get_prompt()
        self.ids = tokenizer_image_token(self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                         return_tensors="pt").unsqueeze(0).cuda()
        pad = self.tokenizer.pad_token_id
        self.kwargs = {"do_sample": False, "max_new_tokens": 32, "use_cache": True,
                       "pad_token_id": self.tokenizer.eos_token_id if pad is None else pad}
        self.audit = {"original_projector_exact": projector_checks, "lora_modules": 0,
                      "loading": loading, "prompt": self.prompt, "generation": self.kwargs}

    def generate(self, pixels, *, verify=False):
        import torch
        from util.evaluation.pulse_adapters import pack_shared_prompt_features, decode_generated
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        pixels = pixels.to(device="cuda", dtype=torch.float16)
        ids = self.ids.expand(len(pixels), -1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = self.base.generate(ids, images=pixels, image_sizes=[(2200, 1700)] * len(pixels), **self.kwargs)
            if verify:
                features = self.base.get_vision_tower()(pixels.flatten(0, 1))
                packed = pack_shared_prompt_features(self.base, ids, features)
                author = self.base.prepare_inputs_labels_for_multimodal(ids, None, None, None, None,
                    pixels, image_sizes=[(2200, 1700)] * len(pixels))[4]
                repeated = super(LlavaLlamaForCausalLM, self.base).generate(inputs_embeds=packed, **self.kwargs)
                if not torch.equal(packed, author) or not torch.equal(output, repeated):
                    raise ValueError("original model author/shared packing or generation mismatch")
        self.last_ids = output.cpu()
        return decode_generated(self.tokenizer, self.last_ids, max_new_tokens=32)


def infer_original(samples, conditions, config, baseline, backend, renderer, profile, pool, *, verify=False):
    import numpy as np
    import torch
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    from util.evaluation.ecg_image_artifact import parse_response
    waves = list(pool.map(lambda s: native500_waveform(s, Path(config["paths"]["raw_root"]))[0], samples))
    hashes = [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in waves]
    clean = torch.from_numpy(np.stack(waves)).cuda()
    rows = []
    for condition in conditions:
        views = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=s["hash_id"], condition=condition,
            profile=profile, base_seed=baseline["corruption"]["seed_config"]["base_seed"]) for i, s in enumerate(samples)])
        fingerprints = [hashlib.sha256(v.tobytes()).hexdigest() for v in views.cpu().numpy()]
        pixels = renderer.preprocess_for_pulse(renderer.render(views)).clone()
        answers = backend.generate(pixels, verify=verify)
        for i, sample in enumerate(samples):
            rows.append({"schema_version": 1, **condition,
                **{k: sample[k] for k in ("sample_key", "record_id", "hash_id", "logical_center")},
                "true_labels": sample["label_names"], "clean_waveform_sha256": hashes[i],
                "input_waveform_sha256": fingerprints[i], "answer": {**answers[i], **parse_response(answers[i]["response"])}})
    return rows


def verify_subset_sources(prepared):
    for name, sha in prepared["sources"].items():
        if sha256_file(REPO / name) != sha:
            raise ValueError(f"subset source changed: {name}")


def worker(path, root, state, worker_id, kind):
    import torch
    from util.evaluation.pulse_adapters import PairedPulseBackend, matched_training_evidence
    prepared = json.loads((state / "prepared.json").read_text())
    verify_subset_sources(prepared)
    config = prepared["paired_config"]
    queue = TaskQueue(state / f"{kind}_queue", prepared["tasks" if kind == "pair" else "base_tasks"])
    by_key = {r["sample_key"]: r for r in prepared["rows"]}
    reservation = claim_gpu(config)
    started = time.time()
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.__setitem__(0, True))
    profile, baseline = profile_and_baseline(config, path, root)
    backend, center = None, None
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=2) as pool:
        while not stop[0]:
            control = json.loads((state / "control.json").read_text())
            if control["paused"] or worker_id in control["drain_workers"]:
                break
            verify_subset_sources(prepared)
            preferred = {t["id"] for t in queue.tasks if t["center"] == center}
            lease = queue.claim(worker_id, allowed_task_ids=preferred) if preferred else None
            lease = lease or queue.claim(worker_id)
            if lease is None:
                break
            with lease:
                next_center = lease.task["center"]
                evidence = None
                if kind == "pair" and (backend is None or center != next_center):
                    evidence = matched_training_evidence(config["training_results"][next_center])
                    if any(item["sha256"] != prepared["training"][next_center][arm]["sha256"]
                           for arm, item in evidence.items()):
                        raise ValueError("subset frozen adapter identity changed")
                if backend is None:
                    if kind == "pair":
                        backend = PairedPulseBackend(evidence, reservation=reservation)
                    else:
                        backend = OriginalBackend(prepared["training"][next_center]["w1"]["protocol"], reservation)
                    renderer, rendering = make_renderer(config)
                    # Engineering uses training K500 only, not heldout selection.
                    train_path = Path(config["training_results"][next_center]["w1"])
                    engineering = json.loads((train_path.parent / "k500_records.json").read_text())[:2]
                    if kind == "base":
                        checks = infer_original(engineering, prepared["conditions"], config, baseline,
                            backend, renderer, profile, pool, verify=True)
                        audit = backend.audit
                    else:
                        checks = infer_batch(engineering, prepared["conditions"][:1], config, baseline,
                            backend, renderer, profile, pool, verify_author=True)
                        audit = {"author_equivalence": backend.admissions, "prompt": backend.prompt,
                                 "generation": backend.generation_kwargs}
                    atomic_json(state / "workers" / f"{worker_id}.admission.json",
                        {"status": "passed", "kind": kind, "audit": audit, "rendering": rendering,
                         "engineering_predictions": checks})
                elif kind == "pair" and center != next_center:
                    backend.set_pair(evidence)
                center = next_center
                before, predictions = time.time(), []
                for keys in lease.task["batches"]:
                    fn = infer_batch if kind == "pair" else infer_original
                    predictions.extend(fn([by_key[k] for k in keys], prepared["conditions"], config,
                        baseline, backend, renderer, profile, pool))
                elapsed = time.time() - before
                lease.commit(predictions, {"seconds": elapsed, "model_predictions": len(predictions) * (2 if kind == "pair" else 1),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                atomic_json(state / "workers" / f"{worker_id}.progress.json", {"time": time.time(),
                    "task_id": lease.task["id"], "kind": kind, "seconds": elapsed, "center": center})
    atomic_json(state / "workers" / f"{worker_id}.final.json", {"seconds": time.time()-started, "kind": kind})


def validate_original_row(row, sample, condition):
    # Reuse the exact parser, label, waveform hash and truncation contract.
    validate_pair_row({**row, "arms": {arm: row["answer"] for arm in ARMS}}, sample, condition)


def comparison_arrays(prepared, pair_queue, base_queue):
    import numpy as np
    samples = {r["sample_key"]: r for r in prepared["rows"]}
    conditions = {c["condition_id"]: c for c in prepared["conditions"]}
    if len(samples) != len(prepared["rows"]) or len(conditions) != 21:
        raise ValueError("duplicate cohort or incomplete conditions")
    indexed, truth, predictions = {}, {}, {}
    for c in CENTERS:
        selected = [r for r in prepared["rows"] if r["logical_center"] == c]
        if len(selected) != prepared["subset_config"]["records_per_center"]:
            raise ValueError("unbalanced or incomplete fixed subset")
        indexed[c] = {r["sample_key"]: i for i, r in enumerate(selected)}
        truth[c] = np.array([[label in r["label_names"] for label in CLASS_ORDER] for r in selected], dtype=np.uint8)
        predictions[c] = np.zeros((3, len(selected), 21, 5), dtype=np.uint8)
    pairs = {}
    for payload in pair_queue.completed_payloads():
        for row in payload["predictions"]:
            validate_pair_row(row, samples[row["sample_key"]], conditions[row["condition_id"]])
            key = (row["sample_key"], row["condition_id"])
            if key in pairs:
                raise ValueError("duplicate paired sample")
            pairs[key] = row
    quality = {c: {name: {"predictions": 0, "parse_invalid": 0, "truncated": 0} for name in NAMES} for c in CENTERS}
    seen = set()
    for payload in base_queue.completed_payloads():
        for row in payload["predictions"]:
            validate_original_row(row, samples[row["sample_key"]], conditions[row["condition_id"]])
            key = (row["sample_key"], row["condition_id"])
            if key in seen or key not in pairs:
                raise ValueError("duplicate or unpaired original result")
            seen.add(key)
            pair = pairs[key]
            if any(row[k] != pair[k] for k in ("clean_waveform_sha256", "input_waveform_sha256")):
                raise ValueError("baseline and fine-tuned input waveforms differ")
            c, i, j = row["logical_center"], indexed[row["logical_center"]][row["sample_key"]], row["condition_index"]
            if j != conditions[row["condition_id"]]["condition_index"]:
                raise ValueError("condition index mismatch")
            for a, answer in enumerate((row["answer"], pair["arms"]["w1"], pair["arms"]["w2"])):
                predictions[c][a, i, j] = [label in answer["predicted_labels"] for label in CLASS_ORDER]
                q = quality[c][NAMES[a]]
                q["predictions"] += 1
                q["parse_invalid"] += not answer["parse_valid"]
                q["truncated"] += answer["hit_max_new_tokens"]
    if len(seen) != len(samples) * 21 or len(pairs) != len(seen):
        raise ValueError("incomplete three-arm grid")
    return truth, predictions, quality


def finalize(output, prepared, pair_queue, base_queue):
    from util.evaluation.ecg_image_metrics import paired_metrics, paired_bootstrap
    from util.evaluation.pulse_benchmark_metrics import exact_bootstrap, write_csv
    from sklearn.metrics import f1_score, accuracy_score
    truth, predictions, quality = comparison_arrays(prepared, pair_queue, base_queue)
    conditions = [c["condition_id"] for c in prepared["conditions"]]
    metrics = paired_metrics(truth, predictions, list(NAMES), conditions)
    for row in metrics["condition_rows"]:
        y = truth[row["center"]]
        p = predictions[row["center"]][NAMES.index(row["model"]), :, conditions.index(row["condition"])]
        if abs(f1_score(y, p, average="macro", zero_division=0)-row["macro_f1"]) > 1e-12 or abs(accuracy_score(y,p)-row["exact_match_accuracy"]) > 1e-12:
            raise ValueError("independent sklearn metric mismatch")
    metrics["evidence"] = {"single_seed_development_only": True, "matched_direct_sft_control": False,
        "historical_zero_shot_reused": False, "subset_identity": prepared["subset_identity"],
        "records": len(prepared["rows"]), "input_fingerprints_equal": True}
    if "pixel_parent" in prepared:
        metrics["evidence"].update(mixing_domain="rendered_rgb", matched_completed_original_reused=True)
        finalize_pixel_comparison(output, prepared, truth, predictions)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "quality.json", quality)
    for a, b in ((0,1),(0,2),(1,2)):
        selected = {c: p[[a,b]] for c,p in predictions.items()}
        names = [NAMES[a], NAMES[b]]
        atomic_json(output / f"bootstrap_{a}_{b}.json", paired_bootstrap(truth, selected, names, repeats=1000, seed=20260909))
        exact = exact_bootstrap(truth, selected, repeats=1000, seed=20260909)
        for item in (exact["center_equal"], *exact["centers"].values()):
            item["models"] = {names[i]: item["models"][MODEL_NAMES[i]] for i in range(2)}
            item["candidate_minus_reference"] = item.pop("two_minus_single")
        atomic_json(output / f"exact_bootstrap_{a}_{b}.json", {"models": names, "intervals": exact})
    write_csv(output / "per_condition.csv", metrics["condition_rows"])
    write_csv(output / "per_class.csv", metrics["class_rows"])
    table = ["# PULSE three-arm fixed subset", "", "Single-seed development; no matched direct-SFT control.", "",
             "| Model | Clean F1 | Corrupt F1 | Clean exact | Corrupt exact |", "|---|---:|---:|---:|---:|"]
    for name in NAMES:
        v=metrics["models"][name]["center_equal"]
        vals=[v[s][m]*100 for m in ("macro_f1","exact_match_accuracy") for s in ("clean","corruption_view_mean")]
        table.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    (output / "comparison.md").write_text("\n".join(table)+"\n")
    index = [{"kind": kind, "task_id": t["id"], "path": str(q.result_path(t).relative_to(output)),
              "sha256": sha256_file(q.result_path(t))} for kind,q in (("pair",pair_queue),("base",base_queue)) for t in q.tasks]
    atomic_json(output / "task_index.json", index)
    runtime = {"new_worker_gpu_hours": sum(json.loads(p.read_text())["wall_seconds"] for p in (output/"workers").glob("*.exit.json"))/3600,
               "reused_pair_tasks": len(json.loads((output/"reuse.json").read_text())["pair_tasks"]),
               "reused_original_tasks": len(json.loads((output/"reuse.json").read_text()).get("original_tasks", []))}
    atomic_json(output / "runtime.json", runtime)
    result = {"artifact_type": "pulse_subset_result", "schema_version": 1, "status": "complete",
        "records_per_center": prepared["subset_config"]["records_per_center"], "models": list(NAMES),
        "subset_identity": prepared["subset_identity"],
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file() and not p.name.endswith(".lock")}}
    validate_result(result, output / "subset_result.json")
    atomic_json(output / "subset_result.json", result)


MIXING_NAMES = ("PULSE-original", "waveform-single", "waveform-two", "image-single", "image-two")


def pixel_comparison_arrays(prepared, truth, predictions):
    import numpy as np
    parent = prepared["pixel_parent"]
    path = Path(parent["result"])
    if sha256_file(path) != parent["sha256"]:
        raise ValueError("frozen waveform comparison result changed")
    validate_result(json.loads(path.read_text()), path)
    old = json.loads((path.parent / "prepared.json").read_text())
    if old["rows"] != prepared["rows"] or old["conditions"] != prepared["conditions"]:
        raise ValueError("image and waveform comparison grids differ")
    old_truth, old_predictions, _ = comparison_arrays(old, TaskQueue(path.parent / "pair_queue", old["tasks"]),
                                                     TaskQueue(path.parent / "base_queue", old["base_tasks"]))
    combined = {}
    for center in CENTERS:
        if not np.array_equal(old_truth[center], truth[center]) or not np.array_equal(old_predictions[center][0], predictions[center][0]):
            raise ValueError("frozen original predictions/labels differ")
        combined[center] = np.concatenate((old_predictions[center], predictions[center][1:]), axis=0)
    return combined


def finalize_pixel_comparison(output, prepared, truth, predictions):
    from util.evaluation.ecg_image_metrics import paired_metrics, paired_bootstrap
    combined = pixel_comparison_arrays(prepared, truth, predictions)
    metrics = paired_metrics(truth, combined, list(MIXING_NAMES), [c["condition_id"] for c in prepared["conditions"]])
    metrics["evidence"] = {"parent": prepared["pixel_parent"], "single_seed_development_only": True,
        "only_training_mixing_domain_changed": True, "matched_direct_sft_control": False,
        "old_predictions_frozen_and_validated": True}
    atomic_json(output / "mixing_comparison.json", metrics)
    for a, b in ((1,3), (2,4), (3,4)):
        atomic_json(output / f"mixing_bootstrap_{a}_{b}.json", paired_bootstrap(truth,
            {c:p[[a,b]] for c,p in combined.items()}, [MIXING_NAMES[a], MIXING_NAMES[b]], repeats=1000, seed=20260909))
    lines = ["# PULSE waveform versus rendered RGB mixing", "", "Fixed 512/center; K500 record-excluded; development only.", "",
        "| Model | Clean F1 | Corrupt F1 | Clean exact | Corrupt exact |", "|---|---:|---:|---:|---:|"]
    for name in MIXING_NAMES:
        value = metrics["models"][name]["center_equal"]
        values = [100*value[s][m] for m in ("macro_f1", "exact_match_accuracy") for s in ("clean", "corruption_view_mean")]
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    (output / "mixing_comparison.md").write_text("\n".join(lines) + "\n")


def validate_result(result, path):
    if result.get("schema_version") == 2:
        from util.evaluation.pulse_visual_subset import validate_result as validate_visual
        return validate_visual(result, path)
    if (result.get("artifact_type") != "pulse_subset_result" or result.get("schema_version") != 1
            or result.get("status") != "complete" or result.get("models") != list(NAMES)):
        raise ValueError("invalid subset result")
    required = {"prepared.json", "cohort.json", "data_audit.json", "metrics.json", "quality.json", "task_index.json", "reuse.json", "runtime.json",
                *{f"{prefix}_{a}_{b}.json" for prefix in ("bootstrap","exact_bootstrap") for a,b in ((0,1),(0,2),(1,2))}}
    if not required <= result["files"].keys():
        raise ValueError("subset evidence incomplete")
    for name, sha in result["files"].items():
        p=(path.parent/name).resolve(); p.relative_to(path.parent.resolve())
        if sha256_file(p) != sha:
            raise ValueError("subset file fingerprint mismatch")
    prepared=json.loads((path.parent/"prepared.json").read_text())
    if result["records_per_center"] != prepared["subset_config"]["records_per_center"] or result["subset_identity"] != prepared["subset_identity"]:
        raise ValueError("subset result identity mismatch")
    pair=TaskQueue(path.parent/"pair_queue",prepared["tasks"])
    base=TaskQueue(path.parent/"base_queue",prepared["base_tasks"])
    truth, predictions, _ = comparison_arrays(prepared,pair,base)
    from util.evaluation.ecg_image_metrics import paired_metrics
    expected=paired_metrics(truth,predictions,list(NAMES),[c["condition_id"] for c in prepared["conditions"]])
    observed=json.loads((path.parent/"metrics.json").read_text()); observed.pop("evidence")
    if expected != observed:
        raise ValueError("subset metrics differ from task recomputation")
    if "pixel_parent" in prepared:
        extra = {"mixing_comparison.json", "mixing_comparison.md", "reference_generation.json",
                 "mixing_bootstrap_1_3.json", "mixing_bootstrap_2_4.json", "mixing_bootstrap_3_4.json"}
        if not extra <= result["files"].keys():
            raise ValueError("pixel comparison evidence incomplete")
        combined = pixel_comparison_arrays(prepared, truth, predictions)
        expected = paired_metrics(truth, combined, list(MIXING_NAMES), [c["condition_id"] for c in prepared["conditions"]])
        observed = json.loads((path.parent / "mixing_comparison.json").read_text()); observed.pop("evidence")
        if expected != observed:
            raise ValueError("pixel comparison metrics differ from task recomputation")


def scheduling_limit(runtime, elapsed, *, both_lanes_verified):
    return runtime["max_workers"] if elapsed >= runtime["expand_after_seconds"] and both_lanes_verified else runtime["initial_workers"]


def run(path, root, output):
    config=load_yaml_mapping(path,description="PULSE subset")
    if config.get("schema_version") == 3:
        from util.evaluation.pulse_visual_subset import run as run_visual
        return run_visual(path, root, output)
    validate_config(config)
    paired_path=resolve_config_reference(config["references"]["paired_config"],owner_config_path=path,
        config_root=root,description="frozen paired config",must_exist=True)
    paired_config=load_yaml_mapping(paired_path,description="paired PULSE")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("subset coordinator must be CPU-only")
    candidates=gpu_allowlist()
    initial=[int(x) for x in os.environ.get("PULSE_SUBSET_INITIAL_GPUS","").split(",") if x]
    if len(initial)!=config["runtime"]["initial_workers"] or len(set(initial))!=len(initial) or not set(initial)<=set(candidates):
        raise ValueError("explicit initial GPUs must match the frozen worker count")
    prepared=prepare(config,paired_config,output)
    pair=TaskQueue(output/"pair_queue",prepared["tasks"]); base=TaskQueue(output/"base_queue",prepared["base_tasks"])
    (output/"workers").mkdir()
    atomic_json(output/"control.json",{"paused":False,"max_workers":4,"auto_expand":True,"drain_workers":[],"allowed_gpus":candidates})
    atomic_json(output/"resource_policy.json",{"initial_gpus":initial,"candidate_gpus":candidates,"runtime":config["runtime"]})
    children={}; idle={}; started=time.time(); stopping=[False]; failed=False
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *args:stopping.__setitem__(0,True))
    with exclusive_lock(output/"coordinator.lock"):
        try:
            while True:
                verify_subset_sources(prepared)
                for wid,item in list(children.items()):
                    code=item["process"].poll()
                    if code is not None:
                        item["log"].close()
                        atomic_json(output/"workers"/f"{wid}.exit.json",{"returncode":code,"wall_seconds":time.time()-item["started"],"gpu":item["gpu"]})
                        failed |= code!=0
                        del children[wid]
                control=json.loads((output/"control.json").read_text())
                if (set(control)!={"paused","max_workers","auto_expand","drain_workers","allowed_gpus"}
                        or type(control["max_workers"]) is not int or not 0<=control["max_workers"]<=4
                        or type(control["paused"]) is not bool or type(control["auto_expand"]) is not bool
                        or not set(control["allowed_gpus"])<=set(candidates)):
                    raise ValueError("invalid subset resource control")
                elapsed=time.time()-started
                if failed or stopping[0] or elapsed>=config["runtime"]["max_wall_seconds"]:
                    control["paused"]=True;atomic_json(output/"control.json",control)
                counts={k:q.counts() for k,q in (("pair",pair),("base",base))}
                if all(v["completed_tasks"]==v["tasks"] for v in counts.values()) and not children:
                    control["paused"]=True;atomic_json(output/"control.json",control)
                    atomic_json(output/"status.json", {"status":"complete", "time":time.time(),
                        "elapsed_seconds":elapsed, "counts":counts, "workers":{}, "paused":True})
                    finalize(output,prepared,pair,base);return
                if control["paused"] and not children:
                    raise RuntimeError("subset drained; partial predictions preserved; no automatic retry")
                admissions=[json.loads(p.read_text()) for p in (output/"workers").glob("*.admission.json")]
                if "pixel_parent" in prepared:
                    reference = json.loads((output/"reference_generation.json").read_text())
                    if any({k:a["audit"][k] for k in reference} != reference for a in admissions):
                        raise ValueError("pixel inference differs from reused original prompt/generation")
                verified={a["kind"] for a in admissions if a["status"]=="passed"}
                if admissions and any((a["audit"]["prompt"],a["audit"]["generation"]) !=
                        (admissions[0]["audit"]["prompt"],admissions[0]["audit"]["generation"]) for a in admissions):
                    raise ValueError("baseline and adapted prompt/generation mismatch")
                limit=min(control["max_workers"],scheduling_limit(config["runtime"],elapsed,both_lanes_verified=verified=={"pair","base"}) if control["auto_expand"] else 2)
                snapshot=resource_snapshot()
                for gpu in candidates:
                    if gpu in snapshot["gpus"] and free_device(snapshot["gpus"][gpu]):idle.setdefault(gpu,time.time())
                    else:idle.pop(gpu,None)
                pressure=(snapshot["available_ram_gib"]>=48 and snapshot["load1"]<=snapshot["cpu_affinity_count"]*.9
                          and shutil.disk_usage(output).free>=20*1024**3)
                if not control["paused"] and pressure and len(children)<limit:
                    usable=initial if limit<=2 else candidates
                    for gpu in usable:
                        if gpu not in control["allowed_gpus"] or gpu not in idle or time.time()-idle[gpu]<60:continue
                        device=resource_snapshot()["gpus"][gpu]
                        if not free_device(device) or any(i["gpu"]==gpu for i in children.values()):continue
                        active={k:sum(i["kind"]==k for i in children.values()) for k in counts}
                        remaining={k:v["tasks"]-v["completed_tasks"] for k,v in counts.items()}
                        options=[k for k in counts if remaining[k]>active[k]]
                        if not options:break
                        # Keep both lanes occupied, then balance remaining work.
                        kind=max(options,key=lambda k:(active[k]==0,remaining[k]/(active[k]+1)))
                        wid=f"{kind}-gpu{gpu}-{uuid.uuid4().hex[:8]}"
                        cmd=[str(PULSE_PYTHON),"-m","util.evaluation.pulse_subset","--worker","--config",str(paired_path),"--config-root",str(root),"--state",str(output),"--worker-id",wid,"--kind",kind]
                        env={**os.environ,"CUDA_VISIBLE_DEVICES":device["uuid"],"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","HF_HOME":"/home/linbinhao/ECG_adv_data/hf_cache/pulse","CUBLAS_WORKSPACE_CONFIG":":4096:8"}
                        log=(output/"workers"/f"{wid}.log").open("x")
                        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                        stamp=time.time()
                        children[wid]={"process":proc,"log":log,"gpu":gpu,"kind":kind,"started":stamp}
                        atomic_json(output/"workers"/f"{wid}.launch.json",{"pid":proc.pid,"process_identity":process_identity(proc.pid),"command":cmd,"gpu":gpu,"gpu_uuid":device["uuid"],"kind":kind,"started":stamp})
                        break
                atomic_json(output/"status.json",{"time":time.time(),"elapsed_seconds":elapsed,"counts":counts,"worker_limit":limit,
                    "workers":{k:{"pid":v["process"].pid,"gpu":v["gpu"],"kind":v["kind"]} for k,v in children.items()},"admitted_lanes":sorted(verified),"paused":control["paused"]})
                time.sleep(config["runtime"]["poll_seconds"])
        finally:
            # Cooperative drain only; never signal any unverified or foreign PID.
            control=json.loads((output/"control.json").read_text());control["paused"]=True
            atomic_json(output/"control.json",control)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true",required=True)
    for k in ("config","config-root","state"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--worker-id",required=True);p.add_argument("--kind",choices=("pair","base"),required=True)
    a=p.parse_args();worker(a.config,a.config_root,a.state,a.worker_id,a.kind)
