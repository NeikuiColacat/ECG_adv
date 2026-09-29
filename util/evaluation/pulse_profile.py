"""Bounded K500-only engineering timings; never mutates the formal benchmark."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

from util.pn2021_artifact_contract import sha256_file


def profile_generation(backend, views, output):
    """Trace one admitted view; profiled times are never throughput evidence."""
    import torch
    if len(views) != 1:
        raise ValueError("Torch profiling is bounded to one prepared view")
    modules = {backend.base.get_vision_tower(): "vision",
               backend.base.get_model().mm_projector: "projector",
               backend.base.get_model(): "llm", backend.base.lm_head: "lm_head"}
    handles, scopes = [], []

    def enter(module, args, kwargs):
        name = modules[module]
        if name == "llm":
            value = kwargs.get("inputs_embeds")
            if value is None:
                value = kwargs.get("input_ids")
            if value is None and args:
                value = args[0]
            name = "llm_prefill" if value is not None and value.shape[1] > 1 else "llm_decode"
        scope = torch.profiler.record_function("pulse/" + name)
        scope.__enter__()
        scopes.append(scope)

    def leave(module, args, kwargs, result):
        scopes.pop().__exit__(None, None, None)

    try:
        for module in modules:
            handles.append(module.register_forward_pre_hook(enter, with_kwargs=True))
            handles.append(module.register_forward_hook(leave, with_kwargs=True, always_call=True))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) as profile:
            answers = backend.generate_views(views)
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        while scopes:
            scopes.pop().__exit__(None, None, None)
    profile.export_chrome_trace(str(output / "torch_trace.json"))
    events = [{"name": e.key, "calls": e.count,
               "cpu_total_ms": e.cpu_time_total / 1000,
               "cpu_self_ms": e.self_cpu_time_total / 1000,
               "cuda_total_ms": e.cuda_time_total / 1000,
               "cuda_self_ms": e.self_cuda_time_total / 1000}
              for e in profile.key_averages()]
    summary = {"scope": "one_view_all_arms_after_admission_not_throughput",
               "cuda_trace_available": any("CUDA" in str(e.device_type) for e in profile.events()),
               "stages": [e for e in events if e["name"].startswith("pulse/")],
               "top_cpu": sorted(events, key=lambda e: e["cpu_self_ms"], reverse=True)[:20],
               "top_cuda": sorted(events, key=lambda e: e["cuda_self_ms"], reverse=True)[:20]}
    return answers, backend.last_view_token_ids, summary


@contextmanager
def merged_lora_probe(base):
    """Use PEFT's merge, but restore exact CPU backups instead of subtracting delta."""
    import torch
    from peft.tuners.lora.layer import Linear
    if base.training or torch.is_grad_enabled():
        raise ValueError("merge probe requires eval/no-grad")
    layers = [m for m in base.modules() if isinstance(m, Linear)]
    if not layers or any(m.merged for m in layers):
        raise ValueError("merge probe requires unmerged LoRA layers")
    backups = []
    try:
        for layer in layers:
            backups.append((layer, layer.get_base_layer().weight.detach().cpu().clone()))
            layer.merge(safe_merge=True)
        yield len(layers)
    finally:
        for layer, original in backups:
            layer.get_base_layer().weight.copy_(original)
            layer.merged_adapters.clear()


@contextmanager
def half_trainables_probe(parameters):
    """Isolated dtype experiment; retain exact originals and never save weights."""
    import torch
    if torch.is_grad_enabled() or any(p.requires_grad for p in parameters.values()):
        raise ValueError("dtype probe requires frozen/no-grad parameters")
    originals = [(p, p.data) for p in parameters.values()]
    try:
        for p, _ in originals:
            p.data = p.data.to(torch.float16)
        yield
    finally:
        for p, original in originals:
            p.data = original


def optimization_matrix(backend, pixels_by_condition, output):
    """Two arms, three conditions, eight records; no labels used for selection."""
    import gc
    import statistics
    import torch
    from contextlib import nullcontext
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_adapters import pack_shared_prompt_features, decode_generated
    from util.evaluation.pulse_generation_projection import last_token_projection

    rows, references = [], {}
    eos = backend.tokenizer.eos_token_id

    def answers(ids):
        result = []
        for sequence in ids.tolist():
            # Embedded-prompt dummy BOS stays; ignore only padding after first EOS.
            end = next((i + 1 for i, t in enumerate(sequence[1:], 1) if t == eos), len(sequence))
            result.append(sequence[:end])
        return result

    def generate(pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            features = backend.base.get_vision_tower()(pixels.flatten(0, 1))
            packed = pack_shared_prompt_features(backend.base,
                backend.prompt_ids.expand(len(pixels), -1), features)
            ids = backend._generate(packed)
        return ids.cpu()

    with torch.inference_mode():
        for arm in ("w1", "w2"):
            backend._select(arm)
            for mode in ("unmerged_fp32", "unmerged_fp16", "merged_fp32_delta"):
                context = (nullcontext() if mode == "unmerged_fp32" else
                           half_trainables_probe(backend.parameters) if mode == "unmerged_fp16" else
                           merged_lora_probe(backend.base))
                with context:
                    # Baseline without projection at batch 2; all larger batches
                    # also use last-position projection and are labeled accordingly.
                    for batch in ((2,) if mode == "unmerged_fp32" else (2, 4, 8)):
                        projection = batch > 2
                        failed = False
                        for condition, pixels in pixels_by_condition.items():
                            row = {"arm": arm, "mode": mode, "batch_size": batch,
                                   "condition_id": condition, "records": len(pixels),
                                   "last_position_projection": projection}
                            try:
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                                with last_token_projection(backend.base) if projection else nullcontext():
                                    generate(pixels[:batch])  # untimed warmup
                                    times, outputs, decoded = [], [], []
                                    for _ in range(3):
                                        torch.cuda.synchronize()
                                        start = time.perf_counter()
                                        ids = [generate(pixels[i:i + batch]) for i in range(0, len(pixels), batch)]
                                        torch.cuda.synchronize()
                                        times.append(time.perf_counter() - start)
                                        outputs.append([r for chunk in ids for r in answers(chunk)])
                                        decoded.append([r for chunk in ids for r in decode_generated(
                                            backend.tokenizer, chunk, max_new_tokens=backend.max_new_tokens)])
                                key = (arm, condition)
                                if mode == "unmerged_fp32":
                                    references[key] = (outputs[0], decoded[0])
                                expected, expected_decoded = references[key]
                                equal = [sum(a == b for a, b in zip(expected, actual)) for actual in outputs]
                                row.update(status="complete", seconds=times,
                                    median_seconds=statistics.median(times),
                                    predictions_per_second=len(pixels) / statistics.median(times),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    answer_token_rows_equal=equal,
                                    decoded_rows_equal=[sum(a == b for a, b in zip(expected_decoded, actual)) for actual in decoded],
                                    repeat_tokens_stable=all(v == outputs[0] for v in outputs),
                                    answer_token_ids=outputs[0], decoded_outputs=decoded[0])
                            except torch.cuda.OutOfMemoryError:
                                row.update(status="out_of_memory")
                                failed = True
                            rows.append(row)
                            atomic_json(output / "optimization_matrix.json", rows)
                            print(json.dumps({"optimization_matrix": {k: v for k, v in row.items()
                                if k not in ("answer_token_ids", "decoded_outputs")}}), flush=True)
                            if failed:
                                gc.collect()
                                torch.cuda.empty_cache()
                                break
                        if failed:
                            break  # do not retry larger batches after OOM
    return rows


def logit_distances(reference, candidate):
    """Per-position full-vocabulary distances, accumulated in CPU float64."""
    import torch
    if reference.ndim != 2 or reference.shape != candidate.shape or reference.shape[1] < 2:
        raise ValueError("logits must have equal nontrivial position/vocabulary shapes")
    a, b = reference.detach().cpu().double(), candidate.detach().cpu().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite logits")
    delta = b - a
    l2 = delta.norm(dim=-1)
    norm = a.norm(dim=-1)
    logp, logq = a.log_softmax(-1), b.log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    metrics = {
        "logit_l2": l2, "reference_logit_l2": norm,
        "relative_logit_l2": l2 / norm.clamp_min(1e-30),
        "logit_rmse": delta.square().mean(-1).sqrt(),
        "logit_max_abs": delta.abs().amax(-1),
        "centered_logit_l2": (delta - delta.mean(-1, keepdim=True)).norm(dim=-1),
        "probability_l2": (q - p).norm(dim=-1),
        "probability_max_abs": (q - p).abs().amax(-1),
        "probability_total_variation": (q - p).abs().sum(-1) / 2,
        "kl_reference_candidate": (p * (logp - logq)).sum(-1).clamp_min(0),
        "reference_top2_margin": a.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1),
        "argmax_equal": a.argmax(-1) == b.argmax(-1),
        "logits_bitwise_equal": (reference.cpu() == candidate.cpu()).all(-1),
    }
    return {k: v.tolist() for k, v in metrics.items()}


def capture_generation_logits(base, call):
    """Read model-forward logits before generation processors; restore hook always."""
    logits = []
    def capture(module, args, output):
        logits.append(output.logits[:, -1, :].detach().float().cpu())
    hook = base.register_forward_hook(capture)
    try:
        ids = call().detach().cpu()
    finally:
        hook.remove()
    if not logits or ids.shape[1] != len(logits) + 1:
        raise ValueError("unexpected embedded-prompt generation/logit alignment")
    return ids, logits


def summarize_logit_rows(rows):
    import torch
    keys = ("logit_l2", "reference_logit_l2", "relative_logit_l2", "logit_rmse",
            "logit_max_abs", "centered_logit_l2", "probability_l2", "probability_max_abs",
            "probability_total_variation", "kl_reference_candidate", "reference_top2_margin")
    def summarize(selected):
        out = {"positions": len(selected),
               "argmax_equal": sum(r["argmax_equal"] for r in selected),
               "logits_bitwise_equal": sum(r["logits_bitwise_equal"] for r in selected)}
        if selected:
            for key in keys:
                x = torch.tensor([r[key] for r in selected], dtype=torch.float64)
                out[key] = {"mean": x.mean().item(), "median": x.quantile(0.5).item(),
                            "p95": x.quantile(0.95).item(), "max": x.max().item()}
        return out
    return {"all": summarize(rows),
            "first_answer_token": summarize([r for r in rows if r["step"] == 0]),
            "later_answer_tokens": summarize([r for r in rows if r["step"] > 0]),
            **{arm: summarize([r for r in rows if r["arm"] == arm]) for arm in ("w1", "w2")}}


def logits_probe(backend, renderer, config, evidence, conditions, profile, baseline, output):
    import numpy as np
    import torch
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_adapters import pack_shared_prompt_features, decode_generated

    records = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())[:8]
    if len(records) != 8 or any(r["logical_center"] != "ningbo" for r in records):
        raise ValueError("logits probe must use eight Ningbo K500 records")
    pixels_by_condition = {}
    rows, checks, samples = [], [], []
    eos = backend.tokenizer.eos_token_id
    with torch.inference_mode():
        clean = torch.from_numpy(np.stack([native500_waveform(r, Path(config["paths"]["raw_root"]))[0]
                                          for r in records])).cuda()
        for condition in conditions:
            tiles = []
            for i in range(0, 8, 2):
                views = torch.cat([corrupt_native500_one(clean[j:j + 1], source_hash=records[j]["hash_id"],
                    condition=condition, profile=profile,
                    base_seed=baseline["corruption"]["seed_config"]["base_seed"]) for j in (i, i + 1)])
                tiles.append(renderer.preprocess_for_pulse(renderer.render(views)).half().clone())
            pixels_by_condition[condition["condition_id"]] = torch.cat(tiles)
        del clean, views, tiles
        backend.generate_pair(next(iter(pixels_by_condition.values()))[:2], verify_author=True)
        for arm in ("w1", "w2"):
            backend._select(arm)
            batches = []
            for condition_id, pixels in pixels_by_condition.items():
                for i in range(0, 8, 2):
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = backend.base.get_vision_tower()(pixels[i:i + 2].flatten(0, 1))
                        packed = pack_shared_prompt_features(backend.base, backend.prompt_ids.expand(2, -1), features)
                    def generate():
                        with torch.autocast("cuda", dtype=torch.float16):
                            return backend._generate(packed)
                    ids, scores = capture_generation_logits(backend.base, generate)
                    repeat_ids, repeat = capture_generation_logits(backend.base, generate)
                    exact = torch.equal(ids, repeat_ids) and len(scores) == len(repeat) and all(
                        torch.equal(a, b) for a, b in zip(scores, repeat))
                    checks.append({"arm": arm, "condition_id": condition_id, "record_offset": i,
                                   "unmerged_repeat_logits_and_tokens_exact": exact})
                    if not exact:
                        raise ValueError("unmerged repeat control is not bitwise exact")
                    batches.append((condition_id, i, packed.cpu(), ids, scores))
                    del repeat_ids, repeat, features, packed
                print(json.dumps({"logits_reference_complete": arm, "condition": condition_id}), flush=True)
            with merged_lora_probe(backend.base):
                for condition_id, i, packed_cpu, reference_ids, reference_scores in batches:
                    packed = packed_cpu.cuda()
                    ids, scores = capture_generation_logits(backend.base, generate)
                    reference_text = decode_generated(backend.tokenizer, reference_ids, max_new_tokens=32)
                    text = decode_generated(backend.tokenizer, ids, max_new_tokens=32)
                    for j in range(2):
                        ref_tokens = reference_ids[j, 1:].tolist()
                        length = ref_tokens.index(eos) + 1 if eos in ref_tokens else len(ref_tokens)
                        cand_tokens = ids[j, 1:].tolist()
                        cand_length = cand_tokens.index(eos) + 1 if eos in cand_tokens else len(cand_tokens)
                        compared = 0
                        for step in range(length):
                            if step >= len(scores) or not torch.equal(reference_ids[j, :step + 1], ids[j, :step + 1]):
                                continue
                            metrics = logit_distances(reference_scores[step][j:j + 1], scores[step][j:j + 1])
                            rows.append({"arm": arm, "condition_id": condition_id,
                                "record_hash": records[i + j]["hash_id"], "step": step,
                                "reference_token_id": ref_tokens[step],
                                **{key: value[0] for key, value in metrics.items()}})
                            compared += 1
                        samples.append({"arm": arm, "condition_id": condition_id,
                            "record_hash": records[i + j]["hash_id"], "reference_steps": length,
                            "compared_identical_prefix_steps": compared,
                            "skipped_divergent_prefix_steps": length - compared,
                            "reference_tokens": ref_tokens[:length], "candidate_tokens": cand_tokens[:cand_length],
                            "answer_tokens_equal": ref_tokens[:length] == cand_tokens[:cand_length],
                            "decoded_equal": reference_text[j] == text[j]})
                    atomic_json(output / "logits_metrics.json", {"positions": rows, "samples": samples,
                                                                 "repeat_controls": checks})
                    del packed, ids, scores
            del batches
            print(json.dumps({"logits_merge_complete": arm, "positions": len(rows)}), flush=True)
    summary = summarize_logit_rows(rows)
    summary.update(sample_model_condition_count=len(samples),
        free_generation_equal=sum(r["answer_tokens_equal"] for r in samples),
        skipped_divergent_prefix_steps=sum(r["skipped_divergent_prefix_steps"] for r in samples),
        unmerged_repeat_controls_exact=all(r["unmerged_repeat_logits_and_tokens_exact"] for r in checks))
    atomic_json(output / "logits_summary.json", summary)
    return {"records": [r["hash_id"] for r in records], "vocabulary_size": backend.base.config.vocab_size,
        "batch_size": 2, "comparison": "unmerged_fp32_trainables_vs_fp32_delta_merged_into_fp16_base",
        "logits": "raw_forward_last_position_before_generation_processors_not_super5_probabilities",
        "positions": "answer_prediction_steps_through_first_eos_only_with_identical_token_prefixes",
        "probabilities": "full_vocabulary_softmax_temperature1", "distance_accumulation": "cpu_float64",
        "aggregation": "uniform_over_valid_record_answer_positions_not_pooled_vector_norm",
        "relative_l2_formula": "norm(candidate-reference)/norm(reference)",
        "centered_l2_formula": "norm((candidate-reference)-mean(candidate-reference))",
        "raw_vectors_persisted": False, "weights_persisted": False}


def validate_result(payload, path):
    if (payload.get("artifact_type") != "pulse_profile_result"
            or payload.get("schema_version") != 1 or payload.get("status") != "complete"
            or payload.get("partition") != "k500_engineering_only"
            or payload.get("formal_optimization_enabled") is not False):
        raise ValueError("invalid engineering profile identity")
    suite = payload.get("suite", "optimization")
    if suite not in {"optimization", "logits", "deployment", "deployment_cache", "deployment_admission"}:
        raise ValueError("invalid engineering suite")
    required = ({"logits_metrics.json", "logits_summary.json", "protocol.json"} if suite == "logits"
                else {"timings.json", "cuda_operators.json", "protocol.json"})
    if suite in {"deployment", "deployment_cache", "deployment_admission"}:
        required = {"deployment_matrix.json", "deployment_summary.json", "deployment_logits.json", "protocol.json"}
    if suite == "deployment_admission":
        required.add("deployment_gate.json")
    if not required <= set(payload["files"]):
        raise ValueError("missing engineering profile products")
    for name, digest in payload["files"].items():
        member = (path.parent / name).resolve()
        member.relative_to(path.parent.resolve())
        if sha256_file(member) != digest:
            raise ValueError("engineering profile member changed")


@contextmanager
def timed_calls(targets, synchronize):
    """Restore even pre-existing instance overrides after timing or failure."""
    measurements, originals = {}, []
    try:
        for owner, name, label in targets:
            original = getattr(owner, name)
            previous = owner.__dict__.get(name)
            existed = name in owner.__dict__
            originals.append((owner, name, existed, previous))
            def measured(*args, _fn=original, _label=label, **kwargs):
                synchronize()
                started = time.perf_counter()
                result = _fn(*args, **kwargs)
                synchronize()
                measurements.setdefault(_label, []).append(time.perf_counter() - started)
                return result
            setattr(owner, name, measured)
        yield measurements
    finally:
        for owner, name, existed, previous in reversed(originals):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


def run(path, root, output, *, suite="optimization", center=None):
    import numpy as np
    import torch
    from util.config_bundle import load_yaml_mapping
    from util.evaluation import pulse_adapters
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_benchmark import (claim_gpu, make_renderer, profile_and_baseline,
                                               snapshot_sources, source_identity)
    from util.evaluation.pulse_generation_projection import last_token_projection
    from util.pulse_benchmark_contract import validate_config

    config = load_yaml_mapping(path, description="PULSE engineering profile input")
    if suite not in {"optimization", "logits", "deployment", "deployment_cache", "deployment_admission"}:
        raise ValueError("invalid engineering suite")
    if (suite == "deployment_admission") != (center is not None):
        raise ValueError("only deployment admission requires a center")
    center = center or "ningbo"
    if center not in config["training_results"]:
        raise ValueError("invalid engineering center")
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_identity()
    repo = Path(__file__).resolve().parents[2]
    for name in ("util/evaluation/pulse_profile.py", "util/evaluation/pulse_generation_projection.py",
                 "boot_scripts/profile_pulse_adapters.py"):
        sources[name] = sha256_file(repo / name)
    if suite in {"deployment", "deployment_cache", "deployment_admission"}:
        sources["util/evaluation/pulse_deployment.py"] = sha256_file(repo / "util/evaluation/pulse_deployment.py")
    snapshot_sources(output, sources)
    reservation = claim_gpu(config)
    started = time.time()
    evidence = pulse_adapters.matched_training_evidence(config["training_results"][center])
    backend = pulse_adapters.PairedPulseBackend(evidence, reservation=reservation)
    renderer, rendering = make_renderer(config)
    profile, baseline = profile_and_baseline(config, path, root)
    records = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())[:2]
    if len(records) != 2 or any(r["logical_center"] != center for r in records):
        raise ValueError("engineering records must be from the same trained center's K500")
    conditions = [next(c for c in baseline["corruption"]["conditions"] if c["depth"] == d) for d in (0, 2, 3)]
    if suite == "deployment_admission":
        conditions = baseline["corruption"]["conditions"]
    if suite in {"logits", "deployment", "deployment_cache", "deployment_admission"}:
        probe_function = logits_probe
        probe_kwargs = {}
        if suite in {"deployment", "deployment_cache", "deployment_admission"}:
            from util.evaluation.pulse_deployment import deployment_probe
            probe_function = deployment_probe
            probe_kwargs["cache_probe"] = suite == "deployment_cache"
            probe_kwargs.update(center=center, admission=suite == "deployment_admission")
        probe = probe_function(backend, renderer, config, evidence, conditions, profile, baseline, output, **probe_kwargs)
        atomic_json(output / "protocol.json", {"suite": suite, "source_identity": sources,
            "partition": "k500_engineering_only", "center": center, "mapping_hash": "555ec85d5b51",
            "config": str(path), "conditions": conditions, "rendering": rendering,
            "training_result_sha256": {a: v["sha256"] for a, v in evidence.items()},
            "author_equivalence": backend.admissions, f"{suite}_probe": probe,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
        atomic_json(output / "runtime.json", {"seconds": time.time() - started,
            "peak_allocated_bytes": max(torch.cuda.max_memory_allocated(), probe.get("peak_allocated_bytes", 0)),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
        result = {"artifact_type": "pulse_profile_result", "schema_version": 1, "status": "complete",
            "suite": suite, "partition": "k500_engineering_only", "formal_optimization_enabled": False,
            "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}}
        validate_result(result, output / "profile_result.json")
        atomic_json(output / "profile_result.json", result)
        return
    clean = torch.from_numpy(np.stack([native500_waveform(r, Path(config["paths"]["raw_root"]))[0]
                                      for r in records])).cuda()
    targets = [(renderer, "render", "render"), (renderer, "preprocess_for_pulse", "preprocess"),
               (backend.base.get_vision_tower(), "forward", "vision"),
               (backend, "_select", "adapter_copy"), (backend, "_generate", "generation"),
               (pulse_adapters, "pack_shared_prompt_features", "packing")]
    timings = []
    with torch.inference_mode():
        warm = renderer.preprocess_for_pulse(renderer.render(clean)).clone()
        backend.generate_pair(warm, verify_author=True)
        del warm
        for condition in conditions:
            views = torch.cat([corrupt_native500_one(clean[i:i + 1], source_hash=r["hash_id"],
                condition=condition, profile=profile, base_seed=baseline["corruption"]["seed_config"]["base_seed"])
                for i, r in enumerate(records)])
            with timed_calls(targets, torch.cuda.synchronize) as measured:
                pixels = renderer.preprocess_for_pulse(renderer.render(views)).clone()
                expected = backend.generate_pair(pixels)
            expected_ids = {k: v.clone() for k, v in backend.last_token_ids.items()}
            with timed_calls(targets, torch.cuda.synchronize) as reduced:
                with last_token_projection(backend.base) as projection_rows:
                    actual = backend.generate_pair(pixels)
            token_equal = {a: torch.equal(expected_ids[a], backend.last_token_ids[a]) for a in expected_ids}
            timings.append({"condition": condition, "baseline_seconds": measured,
                "last_position_seconds": reduced, "projection_rows": projection_rows,
                "complete_token_ids_equal": token_equal, "decoded_outputs_equal": expected == actual,
                "baseline_token_ids": {k: v.tolist() for k, v in expected_ids.items()},
                "candidate_token_ids": {k: v.tolist() for k, v in backend.last_token_ids.items()}})
            atomic_json(output / "timings.json", timings)
            print(json.dumps({"condition_id": condition["condition_id"], "timings": measured,
                              "candidate_timings": reduced, "token_equal": token_equal}), flush=True)
        # One unmodified paired generation only; no unbounded trace file.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            backend.generate_pair(pixels)
        torch.cuda.synchronize()
    operators = sorted(trace.key_averages(), key=lambda e: e.self_cuda_time_total, reverse=True)[:40]
    atomic_json(output / "cuda_operators.json", [{"operator": e.key, "calls": e.count,
        "self_cuda_time_us": e.self_cuda_time_total, "cuda_time_us": e.cuda_time_total,
        "self_cpu_time_us": e.self_cpu_time_total} for e in operators])
    initial_peak = torch.cuda.max_memory_allocated()
    del trace, operators
    # Keep only bounded model-ready tiles in GPU memory, not full-resolution images.
    matrix_records = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())[:8]
    matrix_pixels = {}
    with torch.inference_mode():
        matrix_clean = torch.from_numpy(np.stack([native500_waveform(r, Path(config["paths"]["raw_root"]))[0]
                                                 for r in matrix_records])).cuda()
        for condition in conditions:
            tiles = []
            for i in range(0, len(matrix_records), 2):
                views = torch.cat([corrupt_native500_one(matrix_clean[j:j + 1], source_hash=matrix_records[j]["hash_id"],
                    condition=condition, profile=profile, base_seed=baseline["corruption"]["seed_config"]["base_seed"])
                    for j in range(i, min(i + 2, len(matrix_records)))])
                tiles.append(renderer.preprocess_for_pulse(renderer.render(views)).to(torch.float16).clone())
            matrix_pixels[condition["condition_id"]] = torch.cat(tiles)
        del views, tiles, matrix_clean
        matrix_rows = optimization_matrix(backend, matrix_pixels, output)
    protocol = {"source_identity": sources, "config": str(path), "partition": "k500_engineering_only",
        "center": "ningbo", "records": [r["hash_id"] for r in records], "conditions": conditions,
        "training_result_sha256": {a: v["sha256"] for a, v in evidence.items()},
        "rendering": rendering, "mapping_hash": "555ec85d5b51", "batch_size": 2,
        "attention_classes": sorted({type(m).__name__ for n, m in backend.base.named_modules() if n.endswith("self_attn")}),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sdpa_flags": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                       "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                       "math": torch.backends.cuda.math_sdp_enabled()},
        "author_equivalence": backend.admissions, "scope": "bounded_diagnostic_not_full_gpu_admission"}
    protocol["optimization_matrix"] = {"records": [r["hash_id"] for r in matrix_records],
        "repeats": 3, "batch_candidates": [2, 4, 8], "both_arms": True,
        "timed_scope": "preprocessed_gpu_tiles_through_vision_packing_generation_and_token_cpu_copy",
        "excluded": ["waveform_io", "corruption", "rendering", "queue_io", "merge_setup"],
        "reference": "same_arm_unmerged_fp32_batch2", "weights_persisted": False,
        "merge_restoration": "exact_cpu_backup_not_subtractive_unmerge"}
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "runtime.json", {"seconds": time.time() - started,
        "peak_allocated_bytes": max(initial_peak, torch.cuda.max_memory_allocated(),
                                    *(r.get("peak_allocated_bytes", 0) for r in matrix_rows)),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__})
    result = {"artifact_type": "pulse_profile_result", "schema_version": 1, "status": "complete",
        "partition": "k500_engineering_only", "formal_optimization_enabled": False,
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}}
    validate_result(result, output / "profile_result.json")
    atomic_json(output / "profile_result.json", result)
