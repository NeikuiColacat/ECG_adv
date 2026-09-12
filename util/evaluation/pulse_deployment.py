"""Disposable single-arm merged deployment diagnostics, never a formal worker."""
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import statistics
import time


def merge_single_backend(backend):
    """Mutate only a disposable backend. Reload pristine source for another arm."""
    import torch
    if backend.model.training or torch.is_grad_enabled():
        raise ValueError("deployment merge requires eval/no-grad")
    count = sum(hasattr(m, "lora_A") for m in backend.base.modules())
    if not count:
        raise ValueError("expected unmerged LoRA modules")
    merged = backend.model.merge_and_unload(safe_merge=True)
    backend.base = merged
    backend.model = merged
    backend.parameters.clear()
    backend.states.clear()
    if any("lora_" in n for n, _ in merged.named_parameters()) or any(
            hasattr(m, "lora_A") for m in merged.modules()):
        raise ValueError("LoRA branches remain after unload")
    merged.requires_grad_(False)
    merged.eval()
    return count


def answer_tokens(ids, bos, eos):
    if ids.ndim != 2 or not bool((ids[:, 0] == bos).all()):
        raise ValueError("unexpected generation BOS")
    rows = []
    for seq in ids[:, 1:].tolist():
        rows.append(seq[:seq.index(eos) + 1] if eos in seq else seq)
    return rows


def admission_verdict(rows, condition_ids, *, repeats=2):
    """Strict engineering gate, independent of heldout labels or metric gains."""
    settings = (("unmerged_reference", 2, "legacy"), ("merged_unloaded", 3, "mutable"))
    expected = {(arm, condition, *setting) for arm in ("w1", "w2")
                for condition in condition_ids for setting in settings}
    keys = [(r["arm"], r["condition_id"], r["mode"], r["batch_size"], r["cache_variant"]) for r in rows]
    complete_grid = len(condition_ids) == len(set(condition_ids)) == 21 and len(keys) == len(set(keys)) and set(keys) == expected
    checks = []
    for r in rows:
        checks.append(r.get("status") == "complete" and r.get("records") == 8
            and r.get("answer_tokens_equal") == [8] * repeats and r.get("repeat_tokens_stable") is True
            and r.get("parsed_answers_equal") == 8 and r.get("eos_and_length_equal") == 8
            and r.get("skipped_divergent_prefix_positions") == 0
            and r.get("compared_identical_prefix_positions", 0) > 0
            and (r["mode"] != "unmerged_reference" or r.get("author_reference_verified") is True))
    return {"status": "passed" if complete_grid and all(checks) else "failed",
        "complete_grid": complete_grid, "expected_cells": len(expected), "observed_cells": len(rows),
        "passing_cells": sum(checks), "candidate_outputs": 8 * 2 * len(condition_ids),
        "criterion": "complete_grid_exact_tokens_parsing_eos_repeats_author_reference_no_oom",
        "formal_resume_authorized": False, "heldout_performance_evidence": False}


def deployment_probe(backend, renderer, config, evidence, conditions, profile, baseline, output, *,
                     cache_probe=False, center="ningbo", admission=False):
    import numpy as np
    import torch
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_adapters import PairedPulseBackend, pack_shared_prompt_features
    from util.evaluation.pulse_generation_projection import last_token_projection, generation_cache
    from util.evaluation.pulse_profile import capture_generation_logits, logit_distances, summarize_logit_rows
    from util.evaluation.ecg_image_artifact import parse_response

    records = json.loads((evidence["w1"]["path"].parent / "k500_records.json").read_text())[:8]
    if len(records) != 8 or any(r["logical_center"] != center for r in records):
        raise ValueError("deployment probe requires eight same-center K500 records")
    if admission and (len(conditions) != 21 or cache_probe or torch.cuda.get_allocator_backend() != "native"
                      or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")):
        raise ValueError("admission freezes 21 conditions and the native default allocator")
    timed_repeats = 2 if admission else 3
    rows, logit_rows, lifecycle = [], [], []
    references = {}
    allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    atomic_json(output / "deployment_environment.json", {"allocator_config": allocator_config,
        "allocator_backend": torch.cuda.get_allocator_backend()})
    with torch.inference_mode():
        clean = torch.from_numpy(np.stack([native500_waveform(r, Path(config["paths"]["raw_root"]))[0]
                                          for r in records])).cuda()
        pixels_by_condition = {}
        for condition in conditions:
            tiles = []
            for i in range(0, 8, 2):
                views = torch.cat([corrupt_native500_one(clean[j:j + 1], source_hash=records[j]["hash_id"],
                    condition=condition, profile=profile,
                    base_seed=baseline["corruption"]["seed_config"]["base_seed"]) for j in (i, i + 1)])
                tiles.append(renderer.preprocess_for_pulse(renderer.render(views)).half().clone())
            pixels_by_condition[condition["condition_id"]] = torch.cat(tiles)
        del clean, views, tiles

        for arm in ("w1", "w2"):
            if arm == "w2":
                backend = PairedPulseBackend(evidence)
            backend.generate_pair(next(iter(pixels_by_condition.values()))[:2], verify_author=True)
            backend._select(arm)
            bos, eos = backend.tokenizer.bos_token_id, backend.tokenizer.eos_token_id

            def generate(pixels, capture=False, author=False):
                with torch.autocast("cuda", dtype=torch.float16):
                    features = backend.base.get_vision_tower()(pixels.flatten(0, 1))
                    packed = pack_shared_prompt_features(backend.base,
                        backend.prompt_ids.expand(len(pixels), -1), features)
                    if author:
                        ids = backend.prompt_ids.expand(len(pixels), -1)
                        sizes = [(2200, 1700)] * len(pixels)
                        author_packed = backend.base.prepare_inputs_labels_for_multimodal(
                            ids, None, None, None, None, pixels, image_sizes=sizes)[4]
                        if not torch.equal(packed, author_packed):
                            raise ValueError("author prompt packing mismatch in admission")
                        return capture_generation_logits(backend.base, lambda: backend.base.generate(
                            ids, images=pixels, image_sizes=sizes, **backend.generation_kwargs))
                    if capture:
                        return capture_generation_logits(backend.base, lambda: backend._generate(packed))
                    return backend._generate(packed).cpu(), None

            def pass_once(pixels, batch, capture=False, author=False):
                tokens, vectors = [], []
                for i in range(0, len(pixels), batch):
                    ids, scores = generate(pixels[i:i + batch], capture, author)
                    sequences = answer_tokens(ids, bos, eos)
                    tokens.extend(sequences)
                    if capture:
                        stacked = torch.stack(scores, dim=1)
                        vectors.extend(stacked[j, :len(seq)].clone() for j, seq in enumerate(sequences))
                return tokens, vectors

            def measure(mode, batch, projection, cache_variant="legacy"):
                succeeded = True
                for condition_id, pixels in pixels_by_condition.items():
                    row = {"arm": arm, "mode": mode, "batch_size": batch, "condition_id": condition_id,
                           "records": len(pixels), "last_position_projection": projection,
                           "cache_variant": cache_variant}
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        with ExitStack() as contexts:
                            if projection:
                                contexts.enter_context(last_token_projection(backend.base))
                            if cache_variant != "legacy":
                                cache_counts = contexts.enter_context(generation_cache(backend.base,
                                    chunk_size=512 if cache_variant == "chunk512" else 0))
                            row["phase"] = "warmup"
                            generate(pixels[:batch])
                            times, repeats = [], []
                            row["phase"] = "timed_generation"
                            for _ in range(timed_repeats):
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                actual, _ = pass_once(pixels, batch)
                                torch.cuda.synchronize()
                                times.append(time.perf_counter() - start)
                                repeats.append(actual)
                            row["phase"] = "logits_capture"
                            author = admission and mode == "unmerged_reference"
                            actual, vectors = pass_once(pixels, batch, capture=True, author=author)
                            if cache_variant != "legacy":
                                row["cache_counts"] = dict(cache_counts)
                        key = (arm, condition_id)
                        if mode == "unmerged_reference":
                            references[key] = (actual, vectors)
                        expected, expected_vectors = references[key]
                        if admission:
                            def decoded(sequence):
                                response = backend.tokenizer.decode(sequence, skip_special_tokens=True).strip()
                                return {"response": response, **parse_response(response),
                                    "tokens": len(sequence), "eos": bool(sequence and sequence[-1] == eos),
                                    "hit_max_new_tokens": eos not in sequence and len(sequence) >= backend.max_new_tokens}
                            expected_answers, actual_answers = [decoded(s) for s in expected], [decoded(s) for s in actual]
                            row.update(author_reference_verified=author,
                                parsed_answers_equal=sum(a == b for a, b in zip(expected_answers, actual_answers)),
                                eos_and_length_equal=sum((a["eos"], a["tokens"]) == (b["eos"], b["tokens"])
                                    for a, b in zip(expected_answers, actual_answers)),
                                reference_answers=expected_answers, candidate_answers=actual_answers)
                        compared, skipped = 0, 0
                        for j, seq in enumerate(expected):
                            for step in range(len(seq)):
                                if step >= len(actual[j]) or seq[:step] != actual[j][:step]:
                                    skipped += 1
                                    continue
                                metrics = logit_distances(expected_vectors[j][step:step + 1], vectors[j][step:step + 1])
                                logit_rows.append({"arm": arm, "mode": mode, "batch_size": batch,
                                    "last_position_projection": projection, "cache_variant": cache_variant,
                                    "condition_id": condition_id,
                                    "record_hash": records[j]["hash_id"], "step": step,
                                    **{k: v[0] for k, v in metrics.items()}})
                                compared += 1
                        row.update(status="complete", phase="complete", seconds=times, median_seconds=statistics.median(times),
                            predictions_per_second=len(pixels) / statistics.median(times),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            answer_tokens_equal=[sum(a == b for a, b in zip(expected, r)) for r in repeats],
                            repeat_tokens_stable=all(r == actual for r in repeats),
                            compared_identical_prefix_positions=compared, skipped_divergent_prefix_positions=skipped,
                            answer_token_ids=actual)
                        del vectors
                    except torch.cuda.OutOfMemoryError as error:
                        memory = torch.cuda.memory_stats()
                        row.update(status="out_of_memory", error=str(error), allocator_memory={
                            k: memory[k] for k in ("allocated_bytes.all.current", "reserved_bytes.all.current",
                                "inactive_split_bytes.all.current", "num_alloc_retries", "num_ooms")})
                        succeeded = False
                    rows.append(row)
                    atomic_json(output / "deployment_matrix.json", rows)
                    atomic_json(output / "deployment_logits.json", logit_rows)
                    print(json.dumps({"deployment": {k: v for k, v in row.items() if k != "answer_token_ids"}}), flush=True)
                    if not succeeded:
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                return succeeded

            if not measure("unmerged_reference", 2, False):
                raise RuntimeError("baseline deployment profile failed")
            before = torch.cuda.memory_allocated()
            merged_layers = merge_single_backend(backend)
            gc.collect()
            torch.cuda.empty_cache()
            lifecycle.append({"arm": arm, "merged_and_removed_lora_modules": merged_layers,
                "remaining_adapter_states": len(backend.states), "allocated_before_merge": before,
                "allocated_after_unload": torch.cuda.memory_allocated(),
                "author_equivalence": dict(backend.admissions)})
            if admission:
                measure("merged_unloaded", 3, True, "mutable")
            elif cache_probe:
                measure("merged_unloaded", 2, True)
                mutable_complete = True
                for batch in (2, 3, 4):
                    if not measure("merged_unloaded", batch, True, "mutable"):
                        mutable_complete = False
                        break
                if not mutable_complete:
                    for batch in (3, 4):
                        if not measure("merged_unloaded", batch, True, "chunk512"):
                            break
            else:
                for batch, projection in ((2, False), (2, True), (3, True), (4, True)):
                    if not measure("merged_unloaded", batch, projection):
                        break
            # No rounded subtractive unmerge. Release this deployment and reload
            # original source + checkpoint for the other arm, without SSD export.
            backend.base = None
            backend.model = None
            backend.prompt_ids = None
            gc.collect()
            torch.cuda.empty_cache()
            lifecycle[-1]["allocated_after_model_release"] = torch.cuda.memory_allocated()
            atomic_json(output / "deployment_lifecycle.json", lifecycle)

    summary = {}
    settings = sorted({(r["mode"], r["batch_size"], r["last_position_projection"], r["cache_variant"]) for r in rows})
    for mode, batch, projection, cache_variant in settings:
        def matches(row):
            return (row["mode"], row["batch_size"], row["last_position_projection"], row["cache_variant"]) == (mode, batch, projection, cache_variant)
        cells = [r for r in rows if matches(r)]
        complete = [r for r in cells if r["status"] == "complete"]
        metrics = [r for r in logit_rows if matches(r)]
        suffix = f"_{cache_variant}" if cache_variant != "legacy" else ""
        summary[f"{mode}_b{batch}_projection{int(projection)}{suffix}"] = {
            "complete_cells": len(complete), "oom_cells": len(cells) - len(complete),
            "single_gpu_predictions_per_second": (sum(r["records"] for r in complete) /
                sum(r["median_seconds"] for r in complete)) if complete else None,
            "all_repeated_answers_equal": bool(complete) and all(r["answer_tokens_equal"] == [8] * timed_repeats for r in complete),
            "logits": summarize_logit_rows(metrics)}
    atomic_json(output / "deployment_summary.json", summary)
    if admission:
        atomic_json(output / "deployment_gate.json", admission_verdict(rows, [c["condition_id"] for c in conditions]))
    return {"records": [r["hash_id"] for r in records], "repeats": timed_repeats,
        "batch_candidates": [2, 3] if admission else [2, 3, 4], "center": center, "admission": admission,
        "allocator_config": allocator_config,
        "cache_probe": cache_probe, "chunk512_only_after_mutable_oom": True,
        "visual_and_text_tokens_preserved": True,
        "scope": "two_arms_all21_conditions_eight_k500_records" if admission else "two_arms_three_conditions_eight_records_not_full_admission",
        "timed_scope": "gpu_tiles_to_vision_packing_generation_and_cpu_tokens",
        "reference": "author_unmerged_batch2" if admission else "same_process_same_arm_unmerged_batch2", "weights_persisted": False,
        "quantization": False, "fresh_source_reload_between_arms": True,
        "batch3_partition": [3, 3, 2], "lifecycle": lifecycle,
        "peak_allocated_bytes": max(r.get("peak_allocated_bytes", 0) for r in rows)}
