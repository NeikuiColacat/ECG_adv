"""Matched PULSE development evaluation; frozen cohorts, online RGB effects.

No new split, test-set claim, prompt, processor, or class-score surrogate.
The original source and each adapted CLIP are encoded independently.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from util.evaluation.ecg_image_queue import atomic_json, digest_json
from util.pn2021_artifact_contract import sha256_file
from util.run_record import capture_source_snapshot
from util.pulse_training_contract import CENTERS, CLASS_ORDER, validate_result as validate_training
from util.pulse_hybrid_contract import (IMAGE_FP16_MODES, IMAGE_OPTIMIZED_MODES, ORIGINAL_BYPASS_MODE,
    GPU_C5_SUITE, PAPER_SUITE, GPU_IMAGE_SUITES)

REPO = Path(__file__).resolve().parents[2]
ARMS = ("original", "clean", "single", "three")
WIDTHS = {"clean": 0, "single": 1, "three": 3}
REFERENCE_C15_SUITE = "joint_c15_reference_v1"
C15_SUITES = ("joint_c15", REFERENCE_C15_SUITE)
PROCESSOR_INPUT_HASH_MODE = "view_identity_v1"
FULL_CENTER_RECORDS = dict(zip(CENTERS, (18727, 5322, 7619, 8211)))
FULL_PARENT_PROTOCOL_SHA256 = "75da63670290a3ddd5f9d0c3753710758940d633b7472d4564cd32447f0b24ea"


def validate_sample_labels(samples):
    """Reject coercion, duplicate identities, and label-name/binary disagreement."""
    if not samples or len({s["sample_key"] for s in samples}) != len(samples):
        raise ValueError("duplicate or empty cohort identity")
    for sample in samples:
        labels = sample.get("label", [])
        if (len(labels) != len(CLASS_ORDER) or any(type(v) not in (int, float) or v not in (0, 1) for v in labels)
                or not any(labels)
                or sample.get("label_names") != [name for name, value in zip(CLASS_ORDER, labels) if value]):
            raise ValueError("invalid binary Super5 labels or label-name disagreement")


def processor_input_identity(*, seed, sample, condition, image_suite, renderer_identity):
    """Bind the deterministic model-input recipe without copying CUDA pixels.

    The rendering, preprocessing, condition and source identities form the
    immutable input recipe. Hashing those identities avoids a host transfer
    for every materialized 5-tile model input.
    """
    return digest_json({
        "mode": PROCESSOR_INPUT_HASH_MODE,
        "image_suite": image_suite,
        "seed": int(seed),
        "sample_key": sample["sample_key"],
        "hash_id": sample["hash_id"],
        "condition": condition,
        "renderer_identity": renderer_identity,
        "processor": {
            "shape": [5, 3, 336, 336],
            "dtype": "float16",
            "normalization": "pulse_processor_v1",
        },
    })


def matched_evidence(config):
    from util.run_record import verify_run_file_index
    evidence, common, training_rows = {}, None, None
    ignored = {"width", "jsd", "jsd_weight", "images_per_record_exposure"}
    arms = tuple(config.get("model_arms", ARMS))
    for arm, width in WIDTHS.items():
        if arm not in arms:
            continue
        path = Path(config["training_results"][arm])
        result = json.loads(path.read_text())
        validate_training(result, path)
        p = result["protocol"]
        if "model_asset_sha256" not in p:
            raise ValueError("training result lacks the closed PULSE model asset manifest; retrain it")
        from core.pulse_finetune import MODEL_CONFIG_HASHES, MODEL_HASHES
        if p["model_asset_sha256"] != {**MODEL_HASHES, **MODEL_CONFIG_HASHES}:
            raise ValueError("training result model asset manifest is not the locked PULSE closure")
        if (p["center"] != config["center"] or p["width"] != width
                or p.get("augmentation_topology") not in ("shared_waveform_image_branches_v1", "image_only_gpu_branches_v1")
                or result["mode"] != ("smoke" if config["phase"] == "smoke" and (
                    not config.get("image_suite") or "model_arms" in config) else "train")
                or verify_run_file_index(path.parent.parent)):
            raise ValueError("invalid hybrid training lineage")
        scientific = {k: v for k, v in p.items() if k not in ignored}
        common = scientific if common is None else common
        if scientific != common:
            raise ValueError("hybrid arms differ beyond width/JSD-presence")
        for name, sha in p["implementation_sha256"].items():
            if sha256_file(REPO / name) != sha:
                raise ValueError(f"training source drift: {name}")
        rows = json.loads((path.parent / "k500_records.json").read_text())
        training_rows = rows if training_rows is None else training_rows
        if rows != training_rows:
            raise ValueError("hybrid arms have different K500 records")
        evidence[arm] = {"path": path, "sha256": sha256_file(path), "result": result}
    return evidence, training_rows


def select_development(parent_rows, training_rows, count):
    if count not in (4, 128, 512) or len(parent_rows) != 512:
        raise ValueError("development cohort must be a fixed 4/128/512 prefix")
    ids = [r["hash_id"] for r in parent_rows]
    if len(set(ids)) != 512 or set(ids) & {r["hash_id"] for r in training_rows}:
        raise ValueError("duplicated or adaptation-overlapping development cohort")
    return parent_rows[:count]


def checked_full_parent(config, training_rows=None):
    """Load the completed full original-PULSE cohort and its clean answers.

    The retained full run already contains the exact ref-excluded cohort and
    the original arm's clean predictions.  Adapted C5 views are generated in
    this run; waveform conditions from that historical run are not reused.
    """
    source = Path(config["source_state"]).resolve()
    protocol_path = source / "protocol.json"
    success_path = source / "_SUCCESS.json"
    cohort_path = source / "cohort.jsonl"
    for path in (protocol_path, success_path, cohort_path):
        if not path.is_file():
            raise ValueError(f"full original-PULSE source is incomplete: {path}")
    protocol = json.loads(protocol_path.read_text())
    success = json.loads(success_path.read_text())
    if sha256_file(protocol_path) != FULL_PARENT_PROTOCOL_SHA256:
        raise ValueError("full original-PULSE protocol SHA256 changed")
    if protocol.get("cohort", {}).get("sha256") != sha256_file(cohort_path):
        raise ValueError("full original-PULSE cohort SHA256 changed")
    if (success.get("status") != "complete" or success.get("prediction_count") != 837459
            or protocol.get("cohort", {}).get("record_count") != 39879
            or protocol.get("cohort", {}).get("k500_reference_records_excluded") is not True
            or protocol.get("cohort", {}).get("mapping_hash") != "555ec85d5b51"):
        raise ValueError("full original-PULSE source does not prove the locked cohort")
    for relative, expected in protocol.get("implementation_files", {}).items():
        candidate = REPO / relative
        if candidate.is_file() and sha256_file(candidate) != expected:
            raise ValueError(f"full original-PULSE source drift: {relative}")
    renderer_hash = protocol.get("rendering", {}).get("renderer_script_sha256")
    if renderer_hash and sha256_file(REPO / "util/ecg_image_renderer.py") != renderer_hash:
        raise ValueError("full original-PULSE renderer source drift")
    samples = [json.loads(line) for line in cohort_path.open()]
    validate_sample_labels(samples)
    if any(sum(r["logical_center"] == center for r in samples) != count
           for center, count in FULL_CENTER_RECORDS.items()):
        raise ValueError("full original-PULSE center population changed")
    if (len(samples) != 39879 or len({r["sample_key"] for r in samples}) != len(samples)
            or any(r.get("logical_center") not in CENTERS or not any(r.get("label", [])) for r in samples)):
        raise ValueError("full original-PULSE cohort is incomplete or has invalid labels")
    original = {}
    prediction_count = 0
    prediction_files = sorted((source / "predictions").glob("predictions.rank-*.jsonl"))
    if len(prediction_files) != 4:
        raise ValueError("full original-PULSE prediction shards are incomplete")
    for path in prediction_files:
        for line in path.open():
            row = json.loads(line)
            prediction_count += 1
            if row.get("condition_id") != "clean":
                continue
            key = (row.get("sample_key"), row.get("condition_id"))
            if key in original:
                raise ValueError("duplicate full original-PULSE clean prediction")
            original[key] = {"sample_key": row["sample_key"], "condition_id": "clean",
                "answer": {"response": row.get("response", ""),
                            "generated_tokens": row.get("generated_tokens"),
                            "hit_max_new_tokens": row.get("hit_max_new_tokens")}}
    center = config["center"]
    samples = [r for r in samples if r["logical_center"] == center]
    original = {key: value for key, value in original.items()
                if value["sample_key"].split(":", 1)[0] == center}
    sample_keys = {r["sample_key"] for r in samples}
    if prediction_count != 837459 or set(original) != {(key, "clean") for key in sample_keys}:
        raise ValueError("full original-PULSE prediction rectangle is incomplete")
    if training_rows is not None:
        training_ids = {r["hash_id"] for r in training_rows}
        if any(r["logical_center"] != center for r in training_rows):
            raise ValueError("training K500 manifest belongs to a different center")
        overlap = sorted(training_ids & {r["hash_id"] for r in samples})
        if overlap:
            raise ValueError(
                f"full parent cohort overlaps the adaptation K500 ({len(overlap)} records; first={overlap[0]})")
    # A prose handoff note is provenance, never permission to change a baseline.
    parent = {"conditions": [{"condition_id": "clean", "operators": []}],
              "baseline": {"protocol": protocol}}
    return parent, samples, list(original.values())


def conditions_for(parent_conditions, indices, image_strength):
    from core.image_corruption import IMAGE_OPERATORS
    if (not parent_conditions or parent_conditions[0]["condition_id"] != "clean"
            or not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 1 <= i <= 20 for i in indices)
            or not 0 < image_strength <= 1):
        raise ValueError("invalid frozen development conditions")
    conditions = [{**parent_conditions[0], "family": "clean"}]
    conditions += [{**parent_conditions[i], "family": "waveform"} for i in indices]
    conditions += [{"condition_id": f"image_v1__{op}", "family": "image", "operators": [],
                    "image_operator": op, "image_strength": image_strength} for op in IMAGE_OPERATORS]
    return conditions


def metrics_from_predictions(rows, samples, conditions, *, arms=ARMS):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss
    validate_sample_labels(samples)
    if len({c["condition_id"] for c in conditions}) != len(conditions):
        raise ValueError("duplicate condition identity")
    expected = {(s["sample_key"], c["condition_id"]) for s in samples for c in conditions}
    keyed = {(r["sample_key"], r["condition_id"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("incomplete or duplicate prediction rectangle")
    truth = np.array([s["label"] for s in samples], dtype=np.uint8)
    if truth.shape != (len(samples), 5) or not truth.any(axis=1).all():
        raise ValueError("expected drop-all-zero Super5 cohort")
    c5_identity = any(c.get("image_implementation") in ("image_c5_torch_gpu_v1", "paper_ecg_torch_v1") for c in conditions)
    per_condition, families = [], {a: {} for a in arms}
    for condition in conditions:
        selected = [keyed[s["sample_key"], condition["condition_id"]] for s in samples]
        for sample, row in zip(samples, selected):
            if row["true_labels"] != sample["label_names"] or set(row["arms"]) != set(arms):
                raise ValueError("prediction label/arm identity changed")
            if c5_identity:
                if row.get("hash_id") != sample.get("hash_id"):
                    raise ValueError("C5 prediction hash identity changed")
                for key in ("family", "image_operator", "image_severity", "image_implementation", "stress_group"):
                    if key in condition and row.get(key) != condition[key]:
                        raise ValueError("C5 prediction condition identity changed")
        for arm in arms:
            answers = [r["arms"][arm] for r in selected]
            if c5_identity:
                from util.evaluation.ecg_image_artifact import parse_response
                for answer in answers:
                    if not isinstance(answer.get("response"), str):
                        raise ValueError("C5 answer lacks its original response")
                    parsed = parse_response(answer["response"])
                    if any(answer.get(key) != value for key, value in parsed.items()):
                        raise ValueError("C5 parsed answer does not reproduce from response")
            if any(set(a["predicted_labels"]) - set(CLASS_ORDER) for a in answers):
                raise ValueError("unknown generated label")
            pred = np.array([[name in a["predicted_labels"] for name in CLASS_ORDER] for a in answers], dtype=np.uint8)
            per_condition.append({"arm": arm, "condition_id": condition["condition_id"], "family": condition["family"],
                "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                "exact_match": float(accuracy_score(truth, pred)), "hamming_loss": float(hamming_loss(truth, pred)),
                "parse_failure_rate": sum(not a["parse_valid"] for a in answers) / len(answers),
                "truncation_rate": sum(a["hit_max_new_tokens"] for a in answers) / len(answers),
                "records": len(samples)})
    if not any(c["family"] == "waveform" for c in conditions):
        present_families = tuple(dict.fromkeys(c["family"] for c in conditions))
        for arm in arms:
            for family in present_families:
                group = [r for r in per_condition if r["arm"] == arm and r["family"] == family]
                if not group:
                    raise ValueError("missing metric family")
                families[arm][family] = sum(r["macro_f1"] for r in group) / len(group)
        return {"per_condition": per_condition, "families": families,
                "undefined_class_f1": "zero", "class_order": list(CLASS_ORDER), "development_only": True}
    for arm in arms:
        for family in ("clean", "waveform", "image", *(["joint"] if any(c["family"] == "joint" for c in conditions) else [])):
            group = [r for r in per_condition if r["arm"] == arm and r["family"] == family]
            if not group:
                raise ValueError("missing metric family")
            families[arm][family] = sum(r["macro_f1"] for r in group) / len(group)
    return {"per_condition": per_condition, "families": families,
            "undefined_class_f1": "zero", "class_order": list(CLASS_ORDER), "development_only": True}


def joint_conditions_for(parent_conditions, indices):
    from core.image_stress import IMAGE_PROFILES
    base = conditions_for(parent_conditions, indices, 1.0)
    waves = [c for c in base if c["family"] == "waveform"]
    images = [{"condition_id": f"image_v2__{p}", "family": "image", "operators": [],
               "image_profile": p} for p in IMAGE_PROFILES]
    joint = [{**w, "condition_id": f"joint_v2__{w['condition_id']}__{p}", "family": "joint",
              "waveform_condition_id": w["condition_id"], "image_profile": p}
             for w in waves for p in IMAGE_PROFILES]
    return [base[0], *waves, *images, *joint]


def single_operator_joint_conditions_for(parent_conditions, waveform_operators):
    """Build the reduced suite: five single waveform ops x three image ops.

    Composite depth-2/3 chains are intentionally excluded.  The Cartesian
    joint views remain so the image effect is measured on each single-wave
    operator rather than on a mixed waveform chain.
    """
    from core.image_stress import IMAGE_PROFILES
    if (not parent_conditions or parent_conditions[0]["condition_id"] != "clean"
            or len(tuple(waveform_operators)) != 5
            or len(set(waveform_operators)) != 5
            or any(not isinstance(op, str) or not op for op in waveform_operators)):
        raise ValueError("invalid single-operator joint suite")
    clean = {**parent_conditions[0], "family": "clean"}
    waves = [{"condition_id": f"wave_v3__{op}", "family": "waveform",
              "operators": [op], "waveform_operator": op}
             for op in waveform_operators]
    images = [{"condition_id": f"image_v3__{profile}", "family": "image",
               "operators": [], "image_profile": profile}
              for profile in IMAGE_PROFILES]
    joint = [{"condition_id": f"joint_v3__{w['waveform_operator']}__{profile}",
              "family": "joint", "operators": list(w["operators"]),
              "waveform_condition_id": w["condition_id"],
              "image_profile": profile, "waveform_operator": w["waveform_operator"]}
             for w in waves for profile in IMAGE_PROFILES]
    return [clean, *waves, *images, *joint]


def c15_joint_conditions_for(parent_conditions, waveform_operators, *, severity=3, reference=False):
    """Build the 5-wave x 15 ImageNet-C/CIFAR-C-style image suite.

    The natural-image benchmark has five severities.  The first screen uses a
    fixed middle severity so that the 15 operator families can be compared
    without multiplying the PULSE generation cost by another factor of five.
    """
    from core.image_augmix_c import C_IMAGE_OPERATORS, C_IMPLEMENTATION
    from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
    if (not parent_conditions or parent_conditions[0]["condition_id"] != "clean"
            or parent_conditions[0].get("operators") != []
            or tuple(waveform_operators) != CANONICAL_OPERATOR_ORDER
            or type(severity) is not int or not 1 <= severity <= 5):
        raise ValueError("invalid C15 image suite")
    clean = {**parent_conditions[0], "family": "clean"}
    waves = [{"condition_id": f"wave_c15__{op}", "family": "waveform",
              "operators": [op], "waveform_operator": op} for op in waveform_operators]
    prefix = "c15ref_v1" if reference else "c15"
    identity = {"image_implementation": C_IMPLEMENTATION} if reference else {}
    images = [{"condition_id": f"image_{prefix}__{op}__s{severity}", "family": "image",
               "operators": [], "image_operator": op, "image_severity": severity}
              for op in C_IMAGE_OPERATORS]
    joint = [{"condition_id": f"joint_{prefix}__{w['waveform_operator']}__{op}__s{severity}",
              "family": "joint", "operators": list(w["operators"]),
              "waveform_condition_id": w["condition_id"], "waveform_operator": w["waveform_operator"],
              "image_operator": op, "image_severity": severity}
             for w in waves for op in C_IMAGE_OPERATORS]
    for condition in [*images, *joint]:
        condition.update(identity)
    return [clean, *waves, *images, *joint]


def c15_image_seed(seed, source_hash, condition):
    from core.image_augmix_c import C_IMPLEMENTATION
    from util.evaluation.ecg_image_data import derived_seed
    # Pair the very same image realization across clean-image and all five joint views.
    return derived_seed(seed, C_IMPLEMENTATION, source_hash,
                        condition["image_operator"], condition["image_severity"])


def c5_gpu_conditions_for(parent_conditions, *, severity=5):
    """Clean plus five independent GPU image corruptions, with waveforms off."""
    from core.image_augmix_gpu import C5_GPU_OPERATORS, GPU_C5_IMPLEMENTATION
    if (not parent_conditions or parent_conditions[0]["condition_id"] != "clean"
            or parent_conditions[0].get("operators") != []
            or type(severity) is not int or severity != 5):
        raise ValueError("invalid GPU C5 image suite")
    clean = {**parent_conditions[0], "family": "clean"}
    return [clean] + [
        {"condition_id": f"image_c5_gpu__{op}__s{severity}", "family": "image",
         "operators": [], "image_operator": op, "image_severity": severity,
         "image_implementation": GPU_C5_IMPLEMENTATION}
        for op in C5_GPU_OPERATORS
    ]


def gpu_image_conditions_for(parent_conditions, *, suite, severity):
    if suite == GPU_C5_SUITE:
        return c5_gpu_conditions_for(parent_conditions, severity=severity)
    from core.paper_ecg import paper_conditions, PAPER_IMPLEMENTATION
    if (suite != PAPER_SUITE or not parent_conditions or parent_conditions[0]["condition_id"] != "clean"
            or parent_conditions[0].get("operators") != []):
        raise ValueError("invalid paper ECG image suite")
    return [{**parent_conditions[0], "family": "clean"}] + [
        {"condition_id": c["condition_id"], "family": "image_" + c["stress_group"].removesuffix("_family"),
         "operators": [], "image_operator": c["image_operator"], "image_severity": c["image_severity"],
         "image_implementation": PAPER_IMPLEMENTATION, "stress_group": c["stress_group"]}
        for c in paper_conditions((severity,))]


def gpu_image_view(clean_rgb, condition, samples, seed):
    """Create one seeded image realization, shared by every model arm."""
    import torch
    if condition["family"] == "clean":
        return clean_rgb
    if condition["image_implementation"] == "paper_ecg_torch_v1":
        from core.paper_ecg import apply_paper_operator
        from util.evaluation.ecg_image_data import derived_seed
        def apply(image, sample):
            # Keep geometry/noise coupled across severity levels for this operator.
            key = derived_seed(seed, "paper_ecg_torch_v1", sample["hash_id"], condition["image_operator"])
            return apply_paper_operator(image, condition["image_operator"], severity=condition["image_severity"],
                rng=torch.Generator(device=image.device).manual_seed(key), validate=False)
    elif condition["image_implementation"] == "image_c5_torch_gpu_v1":
        from core.image_augmix_gpu import apply_gpu_c5_image_operator_prevalidated
        def apply(image, sample):
            return apply_gpu_c5_image_operator_prevalidated(image, condition["image_operator"], severity=condition["image_severity"],
                rng=torch.Generator(device=image.device).manual_seed(c5_gpu_image_seed(seed, sample["hash_id"], condition)))
    else:
        raise ValueError("unknown GPU image implementation")
    return torch.cat([apply(clean_rgb[i:i + 1], sample) for i, sample in enumerate(samples)])


def c5_gpu_image_seed(seed, source_hash, condition):
    from core.image_augmix_gpu import GPU_C5_IMPLEMENTATION
    from util.evaluation.ecg_image_data import derived_seed
    return derived_seed(seed, GPU_C5_IMPLEMENTATION, source_hash,
                        condition["image_operator"], condition["image_severity"])


def verify_original_answer(originals, sample_key, condition, answer, *, image_suite,
                           baseline_mode="frozen_parent_v1"):
    if baseline_mode not in ("frozen_parent_v1", "current_protocol_v1"):
        raise ValueError("unknown original baseline protocol")
    if baseline_mode == "current_protocol_v1" and image_suite not in GPU_IMAGE_SUITES:
        raise ValueError("current-protocol baseline is only admitted for GPU C5")
    if condition["family"] not in ("clean", "waveform"):
        return
    if condition["family"] == "waveform" and image_suite in ("joint_v3", *C15_SUITES):
        # The frozen parent contains composite chains, not these new single-op IDs.
        return
    previous = originals[sample_key, condition["condition_id"]]["answer"]
    keys = ("response", "generated_tokens", "hit_max_new_tokens")
    mismatches = {k: {"parent": previous.get(k), "current": answer.get(k)} for k in keys
                  if previous.get(k) is not None and answer.get(k) != previous[k]}
    if not mismatches:
        return {"exact": True, "canonical_equal": True, "mismatches": {}}
    from util.evaluation.ecg_image_artifact import parse_response
    parent_parsed = parse_response(previous.get("response", ""))
    current_parsed = parse_response(answer.get("response", ""))
    canonical_equal = (parent_parsed["predicted_labels"] == current_parsed["predicted_labels"]
                       if parent_parsed["parse_valid"] and current_parsed["parse_valid"] else None)
    if baseline_mode == "current_protocol_v1":
        # This is explicitly a new matched baseline. Historical clean answers
        # are audited, never substituted into this run's metric rectangle.
        return {"exact": False, "canonical_equal": canonical_equal, "mismatches": mismatches}
    if canonical_equal is True and not answer.get("hit_max_new_tokens") and not previous.get("hit_max_new_tokens"):
        return {"exact": False, "canonical_equal": True, "mismatches": mismatches}
    raise ValueError(
        "original PULSE no longer matches the frozen parent predictions: "
        f"sample_key={sample_key} condition_id={condition['condition_id']} "
        f"mismatches={json.dumps(mismatches, sort_keys=True)}")


def original_prediction_digest(rows):
    """Freeze the baseline over all record/views without using target labels."""
    return digest_json([{"sample_key": r["sample_key"], "hash_id": r["hash_id"],
        "condition_id": r["condition_id"],
        "processor_input_identity_sha256": r["processor_input_identity_sha256"],
        "answer": r["arms"]["original"]}
        for r in sorted(rows, key=lambda r: (r["sample_key"], r["condition_id"]))])


def select_image_samples(rows, count, admission_sample_keys=(), *, sampling_seed=None):
    """Keep the final cohort intact; include named regression records in smoke."""
    if count == "full":
        if admission_sample_keys:
            raise ValueError("admission samples must not select a final cohort")
        return rows
    if sampling_seed is not None:
        import hashlib
        if type(sampling_seed) is not int or admission_sample_keys or type(count) is not int or not 1 <= count <= len(rows):
            raise ValueError("invalid fixed random evaluation subset")
        rows = sorted(rows, key=lambda r: (hashlib.sha256(
            f"pulse-evaluation-subset-v1|{sampling_seed}|{r['sample_key']}".encode()).hexdigest(), r["sample_key"]))
    selected = list(rows[:count])
    if admission_sample_keys:
        keyed = {r["sample_key"]: r for r in rows}
        if len(set(admission_sample_keys)) != len(admission_sample_keys) or len(admission_sample_keys) > count:
            raise ValueError("invalid regression admission sample keys")
        if any(k not in keyed for k in admission_sample_keys):
            raise ValueError("regression admission sample is absent from the frozen cohort")
        selected = [keyed[k] for k in admission_sample_keys]
        selected += [r for r in rows if r["sample_key"] not in set(admission_sample_keys)][:count-len(selected)]
    return selected


def image_admission_indices(count, batch_size, *, every_batch=False):
    """Admit every encountered batch shape, including an incomplete tail."""
    if type(count) is not int or count < 1 or type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise ValueError("invalid C5 batch admission dimensions")
    indices, seen = [], set()
    for offset in range(0, count, batch_size):
        size = min(batch_size, count - offset)
        if every_batch or size not in seen:
            indices.extend(range(offset, offset + size))
        seen.add(size)
    return indices




def run(config, refs, root, output):
    import numpy as np
    import torch
    from core.image_corruption import apply_image_operator
    from core.image_augmix_c import apply_c_image_operator, c_reference_identity, AUGMIX_IMPLEMENTATION
    from core.image_augmix_gpu import (
        apply_gpu_c5_image_operator_prevalidated, GPU_C5_IMPLEMENTATION, validate_gpu_image,
    )
    from core.image_stress import apply_image_stress
    from util.augmentations.profile import load_augmentation_profile
    from util.ecg_image_renderer import PulseECGTensorRenderer
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one, derived_seed
    from util.evaluation.ecg_image_artifact import parse_response
    from util.evaluation.ecg_image_elastic import resource_snapshot, free_device
    from util.evaluation.pulse_adapters import HybridPulseBackend
    from util.evaluation.pulse_visual_subset import checked_parent as checked_subset_parent
    from util.pulse_training_contract import load_config
    from util.pulse_hybrid_contract import finalize

    torch.set_num_threads(2)
    # The paired baseline is used as an audit reference.  Make the evaluator
    # deterministic before loading the model so GPU kernels cannot introduce a
    # second, unrecorded source of answer drift.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.get("seed", 0))
    torch.cuda.manual_seed_all(config.get("seed", 0))
    image_suite = config.get("image_suite")
    arms = tuple(config.get("model_arms", ARMS))
    if image_suite == "joint_c15":
        raise ValueError("approximate C15 is historical only; use joint_c15_reference_v1 and new outputs")
    reference_identity = c_reference_identity() if image_suite == REFERENCE_C15_SUITE else None
    evidence, training = matched_evidence(config)
    # GPU C5 uses the retained full K500-excluded parent even for smoke; the
    # older 512-record subset parent is a different protocol and is not
    # present in the C5 source state.
    if config["records_per_center"] == "full" or image_suite in GPU_IMAGE_SUITES:
        parent, parent_rows, original_rows = checked_full_parent(config, training)
    else:
        parent, parent_rows, _, original_rows, _ = checked_subset_parent(config)
    if image_suite in GPU_IMAGE_SUITES:
        samples = select_image_samples(parent_rows, config["records_per_center"], config.get("admission_sample_keys", ()),
            sampling_seed=config["seed"] if config.get("sampling_method") == "seeded_hash_v1" else None)
    elif config["records_per_center"] == "full":
        samples = parent_rows
    else:
        samples = select_development(parent_rows, training, config["records_per_center"])
    if config["phase"] == "block":
        from util.pulse_hybrid_contract import validate_record_span
        start_record, stop_record = validate_record_span(config["center"], config["record_span"])
        samples = parent_rows[start_record:stop_record]
    template, train_refs = load_config(refs["training_config"], root)
    profile = load_augmentation_profile(train_refs["operators_config"], config_root=root)
    joint_suite = image_suite in ("joint_v2", "joint_v3", *C15_SUITES)
    if image_suite in GPU_IMAGE_SUITES:
        conditions = gpu_image_conditions_for(parent["conditions"], suite=image_suite, severity=config["image_severity"])
        if any(e["result"]["protocol"].get("image_gpu", {}).get("implementation") != (
                "paper_ecg_torch_v1" if image_suite == PAPER_SUITE else "augmix_torch_gpu_v2")
               for e in evidence.values()):
            raise ValueError("GPU C5 evaluation requires GPU v2 trained arms")
        trained = evidence["three"]["result"]["protocol"]
        if (template["center"] != config["center"]
                or template["image_augmentation"] != trained["image_augmentation"]
                or trained.get("augmentation_topology") != "image_only_gpu_branches_v1"
                or trained.get("mix_residual") != "clean_render"
                or trained["image_augmentation"].get("waveform_strength") != 0):
            raise ValueError("GPU C5 template/operator lineage differs from training")
    elif image_suite == REFERENCE_C15_SUITE:
        if any(e["result"]["protocol"]["image_augmentation"].get("implementation") != AUGMIX_IMPLEMENTATION
               for e in evidence.values()):
            raise ValueError("reference C15 comparison requires newly trained reference AugMix arms")
        trained = evidence["three"]["result"]["protocol"]
        profile_identity = {k: v for k, v in profile.describe().items() if not k.endswith("path")}
        if (template["center"] != config["center"]
                or template["image_augmentation"] != trained["image_augmentation"]
                or profile_identity != trained["operator_profile"]):
            raise ValueError("reference evaluation template/operator profile differs from training")
        conditions = c15_joint_conditions_for(parent["conditions"], profile.canonical_order,
                                              severity=config["image_severity"], reference=True)
    elif image_suite == "joint_v3":
        conditions = single_operator_joint_conditions_for(parent["conditions"], profile.canonical_order)
    else:
        conditions = (joint_conditions_for(parent["conditions"], config["waveform_indices"]) if joint_suite
                      else conditions_for(parent["conditions"], config["waveform_indices"], config["image_strength"]))
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    candidates = [g for g in resource_snapshot()["gpus"].values() if g["uuid"] == uuid]
    if len(candidates) != 1 or not free_device(candidates[0]) or candidates[0]["pids"]:
        raise RuntimeError("hybrid evaluation requires an exclusive free GPU UUID")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("one GPU per development evaluator")
    output.mkdir(parents=True, exist_ok=False)
    (output / "batches").mkdir()
    capture_source_snapshot(REPO, output / "source_snapshot",
        ("util/evaluation/pulse_hybrid_development.py", "util/evaluation/pulse_adapters.py", "core/pulse_finetune.py",
            "util/evaluation/pulse_visual_subset.py", "util/evaluation/pulse_profile.py", "core/image_corruption.py", "core/image_stress.py",
            "core/image_augmix_c.py", "core/image_augmix_gpu.py", "core/paper_ecg.py", "util/pulse_hybrid_contract.py",
            "util/pulse_training_contract.py", "util/augmentations/profile.py",
            "util/augmentations/torch_operators.py", "util/config_bundle.py",
            "util/evaluation/ecg_image_queue.py", "util/evaluation/pn2021.py",
            "util/evaluation/metrics.py", "util/pn2021_artifact_contract.py",
            "util/random_seed.py", "util/run_record.py",
            "util/ecg_image_renderer.py", "util/evaluation/ecg_image_data.py",
            "util/evaluation/ecg_image_artifact.py", "util/evaluation/ecg_image_elastic.py",
            "data_preprocess/load_cache.py", "data_preprocess/preprocess_primitives.py",
            "data_preprocess/PN2021_preprocess.py", "data_preprocess/pn2021_metadata.py",
            "data_preprocess/data_runtime.py", "models/checkpoints.py", "models/contracts.py",
            "models/input_adapter.py"))
    reservation = [torch.empty(20 * 1024**3, dtype=torch.uint8, device="cuda")]
    started = time.time()
    backend = HybridPulseBackend(evidence, reservation, arms=arms,
        fp16_adapters=config.get("execution_mode") in IMAGE_FP16_MODES,
        bypass_original_lora=config.get("execution_mode") == ORIGINAL_BYPASS_MODE)
    tmp = Path(template["paths"]["temporary_root"])
    tmp.mkdir(parents=True, exist_ok=True)
    renderer, rendering = PulseECGTensorRenderer.from_ecg_image_kit(template["paths"]["toolkit_dir"], device="cuda", tmp_root=tmp)
    renderer_identity = digest_json(rendering)
    atomic_json(output / "cohort.json", {"samples": samples, "conditions": conditions,
        "k500_overlap": 0, "patient_independence_verified": False, "renderer": rendering,
        "processor_input_hash_mode": PROCESSOR_INPUT_HASH_MODE,
        **({"image_reference": reference_identity} if reference_identity else {})})
    originals = {(r["sample_key"], r["condition_id"]): r for r in original_rows}
    original_parity = {"mode": "parent_response_audit",
        "baseline_mode": config.get("original_baseline_mode", "frozen_parent_v1"),
        "checked": 0, "exact": 0, "drift": 0, "canonical_equal": 0,
        "canonical_drift": 0, "canonical_unknown": 0, "examples": []}
    predictions = []
    performance = []
    performance_batches = []
    admitted_batch_sizes = set()
    optimized = image_suite in GPU_IMAGE_SUITES and config.get("execution_mode") in IMAGE_OPTIMIZED_MODES
    compare_baseline = config.get("execution_mode") == ORIGINAL_BYPASS_MODE and config.get("performance_smoke", False)
    torch.cuda.reset_peak_memory_stats()
    # C5 uses the explicitly admitted batch for all three arms. Each new
    # batch shape, including a tail, must pass same-batch author/token parity.
    batch_size = config.get("inference_batch_size", 1) if image_suite in GPU_IMAGE_SUITES else 2
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset:offset + batch_size]
        waves = np.stack([native500_waveform(s, Path(template["paths"]["raw_root"]))[0] for s in batch])
        clean = torch.from_numpy(waves).cuda()
        clean_rgb = renderer.render(clean)
        if image_suite in GPU_IMAGE_SUITES:
            validate_gpu_image(clean_rgb)
        batch_rows = []
        prepared_views = []
        if optimized:
            for condition in conditions:
                rgb = gpu_image_view(clean_rgb, condition, batch, config["seed"])
                prepared_views.append(renderer.preprocess_for_pulse(rgb).to(dtype=torch.float16).clone())
            new_batch_shape = len(batch) not in admitted_batch_sizes
            compare = new_batch_shape or config["phase"] == "smoke" or config.get("performance_smoke", False)
            if new_batch_shape:
                backend.generate_pair(prepared_views[0], verify_author=True)
                if compare_baseline:
                    backend.generate_views(prepared_views[:1], bypass_original_lora=False)
                backend.generate_views(prepared_views[:1])
                admitted_batch_sizes.add(len(batch))
            if config.get("performance_smoke", False) and offset == batch_size:
                torch.cuda.profiler.start()
            reference_tokens, reference_answers = [], []
            # Alternate measured order to avoid systematically favoring the
            # second path through thermal/cache warmup. Warmup is excluded.
            timings = {}
            order = ("reference", "optimized") if (offset // batch_size) % 2 == 0 else ("optimized", "reference")
            if compare_baseline:
                methods = ("reference", "baseline", "optimized")
                shift = (offset // batch_size) % len(methods)
                order = methods[shift:] + methods[:shift]
            for method in order:
                if method == "reference" and not compare:
                    continue
                torch.cuda.synchronize()
                measured = time.perf_counter()
                with torch.cuda.nvtx.range(method):
                    if method == "reference":
                        for pixels in prepared_views:
                            reference_answers.append(backend.generate_pair(pixels))
                            reference_tokens.append(backend.last_token_ids)
                    elif method == "baseline":
                        baseline_answers = backend.generate_views(prepared_views, bypass_original_lora=False)
                        baseline_tokens = backend.last_view_token_ids
                    else:
                        optimized_answers = backend.generate_views(prepared_views)
                        optimized_tokens = backend.last_view_token_ids
                torch.cuda.synchronize()
                timings[method] = time.perf_counter() - measured
            if compare and (reference_tokens != optimized_tokens or reference_answers != optimized_answers):
                raise ValueError("optimized C5 generation differs from reference tokens/answers")
            if compare_baseline and (baseline_tokens != optimized_tokens or baseline_answers != optimized_answers):
                raise ValueError("optimized C5 generation differs from current v2 tokens/answers")
            if config.get("torch_profile", False) and offset == 0:
                from util.evaluation.pulse_profile import profile_generation
                answers, tokens, summary = profile_generation(backend, prepared_views[:1], output)
                if answers != optimized_answers[:1] or tokens != optimized_tokens[:1]:
                    raise ValueError("Torch profiling changed admitted generation outputs")
                atomic_json(output / "torch_profile.json", {**summary, "tokens_exact": True})
            if compare:
                performance.extend({"sample_key": sample["sample_key"], "tokens_exact": True,
                    "views": len(conditions), "arms": len(arms), "seconds": timings,
                    "order": list(order), "batch_offset": offset, "actual_batch_size": len(batch)} for sample in batch)
                atomic_json(output / "performance_admission.json", {"execution_mode": config["execution_mode"],
                    "inference_trainable_dtype": ("float16" if config["execution_mode"] in IMAGE_FP16_MODES else "float32"),
                    **({"original_zero_lora_bypassed": True} if config["execution_mode"] == ORIGINAL_BYPASS_MODE else {}),
                    "reference_parameter_storage_restored": all(p.dtype == torch.float32 for p in backend.parameters.values()),
                    "batch_size": batch_size, "trainable_vision_outputs_cached": 0, "records": performance})
            performance_batches.append({"offset": offset, "sample_keys": [s["sample_key"] for s in batch],
                "actual_batch_size": len(batch), "seconds": timings, "reference_checked": compare,
                "warmup_excluded": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            atomic_json(output / "performance_batches.json", performance_batches)
        for condition in conditions:
            if optimized:
                pixels = prepared_views[conditions.index(condition)]
            elif image_suite in GPU_IMAGE_SUITES and condition["family"] != "clean":
                rgb = gpu_image_view(clean_rgb, condition, batch, config["seed"])
                viewed = clean
            elif image_suite == REFERENCE_C15_SUITE and condition["family"] in ("image", "joint"):
                viewed = clean
                if condition["family"] == "joint":
                    wave_condition = {**condition, "condition_id": condition["waveform_condition_id"]}
                    viewed = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=s["hash_id"], condition=wave_condition,
                        profile=profile, base_seed=parent["baseline"]["corruption"]["seed_config"]["base_seed"])
                        for i, s in enumerate(batch)])
                rgb = renderer.render(viewed) if condition["family"] == "joint" else clean_rgb
                rgb = torch.cat([apply_c_image_operator(rgb[i:i+1], condition["image_operator"],
                    severity=condition["image_severity"], rng=torch.Generator(device="cuda").manual_seed(
                        c15_image_seed(config["seed"], s["hash_id"], condition)))
                    for i, s in enumerate(batch)])
            elif joint_suite and condition["family"] in ("image", "joint"):
                viewed = clean
                if condition["family"] == "joint":
                    wave_condition = {**condition, "condition_id": condition["waveform_condition_id"]}
                    viewed = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=s["hash_id"], condition=wave_condition,
                        profile=profile, base_seed=parent["baseline"]["corruption"]["seed_config"]["base_seed"])
                        for i, s in enumerate(batch)])
                rgb = renderer.render(viewed) if condition["family"] == "joint" else clean_rgb
                rgb = torch.cat([apply_image_stress(rgb[i:i+1], condition["image_profile"],
                    rng=torch.Generator(device="cuda").manual_seed(derived_seed(config["seed"],
                        "hybrid-eval-image-v2", s["hash_id"], condition["image_profile"]))) for i,s in enumerate(batch)])
            elif condition["family"] == "image":
                rgb = torch.cat([apply_image_operator(clean_rgb[i:i+1], condition["image_operator"],
                    strength=condition["image_strength"], rng=torch.Generator(device="cuda").manual_seed(
                        derived_seed(config["seed"], "hybrid-eval-image-v1", s["hash_id"], condition["condition_id"])))
                    for i, s in enumerate(batch)])
                viewed = clean
            elif condition["family"] == "waveform":
                viewed = torch.cat([corrupt_native500_one(clean[i:i+1], source_hash=s["hash_id"], condition=condition,
                    profile=profile, base_seed=parent["baseline"]["corruption"]["seed_config"]["base_seed"])
                    for i, s in enumerate(batch)])
                rgb = renderer.render(viewed)
            else:
                viewed, rgb = clean, clean_rgb
            if not optimized:
                pixels = renderer.preprocess_for_pulse(rgb).clone()
            if joint_suite and config["phase"] == "smoke" and offset == 0 and condition["family"] in ("clean", "image"):
                from PIL import Image
                preview = (rgb[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
                Image.fromarray(preview).save(tmp / f"joint_v2_preview_{condition['condition_id']}.png")
            # Bind the deterministic rendered/preprocessed view recipe without
            # copying each CUDA tile to host memory in the inference loop.
            input_hashes = [processor_input_identity(
                seed=config["seed"], sample=sample, condition=condition,
                image_suite=image_suite, renderer_identity=renderer_identity,
            ) for sample in batch]
            answers = (optimized_answers[conditions.index(condition)] if optimized else
                       backend.generate_pair(pixels, verify_author=condition["family"] == "clean"
                           and (len(batch) not in admitted_batch_sizes if image_suite in GPU_IMAGE_SUITES else offset == 0)))
            if not optimized and condition["family"] == "clean":
                admitted_batch_sizes.add(len(batch))
            for i, sample in enumerate(batch):
                parity = verify_original_answer(
                    originals, sample["sample_key"], condition, answers["original"][i],
                    image_suite=image_suite,
                    baseline_mode=original_parity["baseline_mode"])
                if parity is not None:
                    original_parity["checked"] += 1
                    if parity["exact"]:
                        original_parity["exact"] += 1
                    else:
                        original_parity["drift"] += 1
                        if parity["canonical_equal"] is True:
                            original_parity["canonical_equal"] += 1
                        elif parity["canonical_equal"] is False:
                            original_parity["canonical_drift"] += 1
                        else:
                            original_parity["canonical_unknown"] += 1
                        if len(original_parity["examples"]) < 8:
                            original_parity["examples"].append({
                                "sample_key": sample["sample_key"],
                                "condition_id": condition["condition_id"],
                                "mismatches": parity["mismatches"],
                            })
                batch_rows.append({"sample_key": sample["sample_key"], "hash_id": sample["hash_id"],
                    "condition_id": condition["condition_id"],
                    **{k: condition[k] for k in ("family", "operators", "image_operator", "image_severity",
                                                  "image_implementation", "stress_group") if k in condition},
                    "true_labels": sample["label_names"],
                    # This is deliberately named as an identity digest: it is
                    # not a SHA-256 over CUDA tensor bytes.
                    "processor_input_identity_sha256": input_hashes[i],
                    "processor_input_hash_mode": PROCESSOR_INPUT_HASH_MODE,
                    "arms": {a: {**answers[a][i], **parse_response(answers[a][i]["response"])} for a in arms}})
        predictions.extend(batch_rows)
        atomic_json(output / f"batches/{offset:04d}.json", batch_rows)
        atomic_json(output / "progress.json", {"pid": os.getpid(), "gpu": uuid,
            "completed_records": offset + len(batch), "total_records": len(samples),
            "predictions": len(predictions) * len(arms), "seconds": time.time() - started})
        print(json.dumps({"event": "development_batch", "records": offset + len(batch), "total": len(samples)}), flush=True)
    if config.get("performance_smoke", False):
        torch.cuda.profiler.stop()
    baseline_details = {}
    if image_suite in GPU_IMAGE_SUITES:
        baseline = {"mode": config["original_baseline_mode"], "batch_size": batch_size,
            "seed": config["seed"], "precision": "float16",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "generation_kwargs": backend.generation_kwargs,
            "prompt_sha256": digest_json(backend.prompt),
            "model_asset_sha256": trained["model_asset_sha256"],
            "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "attention_implementation": backend.base.config._attn_implementation},
            "restore_admission": backend.original_restore_admission,
            "prediction_count": len(predictions),
            "predictions_sha256": original_prediction_digest(predictions),
            "historical_parent_role": "diagnostic_only_not_metric_baseline"}
        atomic_json(output / "original_baseline.json", baseline)
        baseline_details = {"original_baseline_mode": baseline["mode"],
                            "original_baseline_sha256": sha256_file(output / "original_baseline.json")}
    metrics = metrics_from_predictions(predictions, samples, conditions, arms=arms)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "original_parent_parity.json", original_parity)
    atomic_json(output / "generation_admission.json", backend.admissions)
    atomic_json(output / "runtime.json", {"seconds": time.time() - started, "gpu": uuid,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "augmented_images_persisted": 0})
    finalize(output, mode="evaluate", details={"center": config["center"], "phase": config["phase"],
        "records": len(samples), "conditions": len(conditions), "families": metrics["families"],
        "cohort_identity": digest_json(samples), "training_results": {a: e["sha256"] for a, e in evidence.items()},
        "full_cohort": config["records_per_center"] == "full" and config["phase"] != "block",
        **({"record_span": config["record_span"], "parent_cohort_identity": digest_json(parent_rows)}
           if config["phase"] == "block" else {}),
        **({"training_result_paths": {a: str(e["path"]) for a, e in evidence.items()}}
           if image_suite in GPU_IMAGE_SUITES else {}),
        "development_only": True, "processor_input_hash_mode": PROCESSOR_INPUT_HASH_MODE,
        **({"inference_batch_size": config["inference_batch_size"]} if "inference_batch_size" in config else {}),
        **({"performance_smoke": config["performance_smoke"]} if "performance_smoke" in config else {}),
        **({"sampling_method": config["sampling_method"], "sampling_source_state": config["source_state"],
            "sampling_population_identity": digest_json(parent_rows)} if "sampling_method" in config else {}),
        **({"execution_mode": config.get("execution_mode", "reference_v1")} if image_suite in GPU_IMAGE_SUITES else {}),
        "evaluation_seed": config["seed"],
        "original_parent_parity": original_parity,
        **baseline_details,
        **({"image_suite": image_suite} if image_suite else {}),
        **({"model_arms": list(arms)} if "model_arms" in config else {}),
        **({"image_severity": config["image_severity"]} if image_suite in (*GPU_IMAGE_SUITES, *C15_SUITES) else {}),
        **({"image_reference": reference_identity} if reference_identity else {}),
        **({"image_implementation": "paper_ecg_torch_v1" if image_suite == PAPER_SUITE else GPU_C5_IMPLEMENTATION}
           if image_suite in GPU_IMAGE_SUITES else {})})


def summarize_four_centers(entries, output, *, seed, phase="final", families=None, arms=ARMS):
    import numpy as np
    from util.pulse_hybrid_contract import validate_result
    from util.evaluation.pulse_benchmark_metrics import write_csv
    all_rows, draws, center_scores = [], [], {}
    rng = np.random.default_rng(seed)
    conditions, image_reference, suite_identity, baseline_identity = None, None, None, None
    selection_identity = None
    for center in CENTERS:
        path = Path(entries[center]["path"])
        if sha256_file(path) != entries[center]["sha256"]:
            raise ValueError("final evaluation result changed")
        result = json.loads(path.read_text())
        validate_result(result, path)
        if result["details"]["phase"] != phase or result["details"]["center"] != center:
            raise ValueError("summary requires four matched final development evaluations")
        if tuple(result["details"].get("model_arms", ARMS)) != tuple(arms):
            raise ValueError("summary model arms differ from evaluated arms")
        suite = result["details"].get("image_suite")
        selection = (result["details"].get("sampling_method"),
                     result["details"]["records"] if result["details"].get("sampling_method") else None)
        if selection_identity is not None and selection != selection_identity:
            raise ValueError("four-center sampling policies or subset budgets differ")
        selection_identity = selection
        if suite_identity is None:
            suite_identity = suite
            if families is None:
                families = ("clean", "image") if suite == GPU_C5_SUITE else ("clean", "waveform", "image")
        elif suite != suite_identity:
            raise ValueError("four-center image suites differ")
        if suite == GPU_C5_SUITE:
            receipt = json.loads((path.parent / "original_baseline.json").read_text())
            identity = {k: receipt[k] for k in ("mode", "batch_size", "precision", "seed",
                "deterministic_algorithms", "generation_kwargs", "prompt_sha256",
                "model_asset_sha256", "environment", "historical_parent_role")}
            if baseline_identity is not None and identity != baseline_identity:
                raise ValueError("four-center original baseline generation contracts differ")
            baseline_identity = identity
        cohort = json.loads((path.parent / "cohort.json").read_text())
        if result["details"].get("image_suite") == REFERENCE_C15_SUITE:
            image_reference = cohort["image_reference"] if image_reference is None else image_reference
            if image_reference != cohort["image_reference"]:
                raise ValueError("four-center C15 reference environments differ")
        conditions = cohort["conditions"] if conditions is None else conditions
        if conditions != cohort["conditions"]:
            raise ValueError("four-center corruption suites differ")
        available_families = {c["family"] for c in conditions}
        if any(family not in available_families for family in families):
            raise ValueError("summary requests a metric family absent from the suite")
        samples = cohort["samples"]
        records = {(r["sample_key"], r["condition_id"]): r for name in sorted(result["files"])
                   if name.startswith("batches/") for r in json.loads((path.parent / name).read_text())}
        truth = np.array([s["label"] for s in samples], dtype=float)
        prediction = np.array([[[[label in records[s["sample_key"], c["condition_id"]]["arms"][arm]["predicted_labels"]
            for label in CLASS_ORDER] for c in conditions] for s in samples] for arm in arms], dtype=float)
        count = len(samples)
        weights = rng.multinomial(count, np.full(count, 1/count), size=1000).astype(float)
        shape = (1000, len(arms), len(conditions), 5)
        tp = (weights @ (prediction * truth[None, :, None, :]).transpose(1, 0, 2, 3).reshape(count, -1)).reshape(shape)
        positive = (weights @ prediction.transpose(1, 0, 2, 3).reshape(count, -1)).reshape(shape)
        denominator = positive + (weights @ truth)[:, None, None, :]
        f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean(-1)
        draws.append(np.stack([f1[:, :, [i for i,c in enumerate(conditions) if c["family"] == family]].mean(-1)
                               for family in families], axis=-1))
        metrics = json.loads((path.parent / "metrics.json").read_text())
        center_scores[center] = metrics["families"]
        all_rows.extend({"center": center, **r} for r in metrics["per_condition"])
    means = np.mean(draws, axis=0)
    pairs = [(i, j) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    intervals = []
    for first, second in pairs:
        for family_index, family in enumerate(families):
            delta = 100 * (means[:, second, family_index] - means[:, first, family_index])
            point = 100 * np.mean([center_scores[c][arms[second]][family] - center_scores[c][arms[first]][family] for c in CENTERS])
            intervals.append({"comparison": f"{arms[second]}_minus_{arms[first]}", "family": family,
                "delta_pp": float(point), "ci95_pp": np.quantile(delta, [0.025, 0.975]).tolist()})
    # The same resampled records appear in both families and every model arm.
    # Joint is the balanced full wave x image cross product, so subtracting the
    # waveform mean measures the incremental effect of the image stresses.
    degradation = []
    if "joint" in families and "waveform" in families:
        j, w = families.index("joint"), families.index("waveform")
        for index, arm in enumerate(arms):
            delta = 100 * (means[:, index, j] - means[:, index, w])
            point = 100 * np.mean([center_scores[c][arm]["joint"] - center_scores[c][arm]["waveform"] for c in CENTERS])
            degradation.append({"arm": arm, "comparison": "joint_minus_waveform",
                "delta_pp": float(point), "ci95_pp": np.quantile(delta, [0.025, 0.975]).tolist()})
    atomic_json(output / "comparison.json", {"centers": center_scores, "paired_deltas": intervals,
        "incremental_image_effect": degradation,
        "evidence": "single_seed_development_selected_not_independent_test",
        "bootstrap": {"repeats": 1000, "seed": seed, "unit": "record", "strata": "center",
            "training_seed_uncertainty": False, "patient_cluster_adjustment": False, "selection_bias_corrected": False}})
    write_csv(output / "per_condition.csv", all_rows)


def checked_center_blocks(entries, samples, expected_details):
    """Validate complete coverage and a common scientific identity before merging."""
    from util.pulse_hybrid_contract import validate_result, validate_record_span, owned_path
    center = expected_details['center']
    full_identity = digest_json(samples)
    expected_spans = [[start, min(start + 512, len(samples))] for start in range(0, len(samples), 512)]
    if not isinstance(entries, list) or len(entries) != len(expected_spans):
        raise ValueError('missing or extra evaluation blocks')
    checked = []
    common_details, common_cohort, common_baseline, common_sources = None, None, None, None
    variable_details = {'phase', 'records', 'families', 'cohort_identity', 'full_cohort',
        'record_span', 'parent_cohort_identity', 'training_result_paths',
        'original_parent_parity', 'original_baseline_sha256'}
    for entry in entries:
        if set(entry) != {'path', 'sha256'}:
            raise ValueError('invalid immutable evaluation block reference')
        path = Path(entry['path'])
        owned_path(str(path))
        if sha256_file(path) != entry['sha256']:
            raise ValueError('evaluation block result changed')
        result = json.loads(path.read_text())
        details = result.get('details', {})
        if (result.get('mode') != 'evaluate' or details.get('phase') != 'block'
                or details.get('full_cohort') is not False or details.get('center') != center
                or details.get('parent_cohort_identity') != full_identity):
            raise ValueError('evaluation block has the wrong center, phase or parent cohort')
        start, stop = validate_record_span(center, details.get('record_span'))
        validate_result(result, path)
        if any(details.get(key) != value for key, value in expected_details.items()):
            raise ValueError('evaluation block scientific identity differs')
        cohort = json.loads((path.parent / 'cohort.json').read_text())
        if cohort['samples'] != samples[start:stop]:
            raise ValueError('evaluation block sample identities or ordering changed')
        metadata = {key: value for key, value in details.items() if key not in variable_details}
        cohort_metadata = {key: value for key, value in cohort.items() if key != 'samples'}
        baseline = json.loads((path.parent / 'original_baseline.json').read_text())
        baseline_metadata = {key: value for key, value in baseline.items()
                             if key not in {'prediction_count', 'predictions_sha256'}}
        sources = {name: sha for name, sha in result['files'].items() if name.startswith('source_snapshot/')}
        if common_details is not None and (metadata != common_details or cohort_metadata != common_cohort
                or baseline_metadata != common_baseline or sources != common_sources):
            raise ValueError('evaluation block sources, renderer or generation contract differ')
        common_details, common_cohort = metadata, cohort_metadata
        common_baseline, common_sources = baseline_metadata, sources
        rows = [row for name in sorted(result['files']) if name.startswith('batches/')
                for row in json.loads((path.parent / name).read_text())]
        # The retained validator already checks the prediction rectangle and reparses answers.
        checked.append({'start': start, 'stop': stop, 'path': path, 'result': result,
            'cohort': cohort, 'rows': rows, 'baseline': baseline})
    checked.sort(key=lambda block: block['start'])
    if [[block['start'], block['stop']] for block in checked] != expected_spans:
        raise ValueError('evaluation blocks overlap or leave holes')
    return checked



def merge_center_blocks(entries, config, output):
    """Build one full-center artifact by recomputing metrics over all predictions."""
    from util.pulse_hybrid_contract import validate_config, finalize
    validate_config(config)
    if (config['phase'] != 'final' or config['records_per_center'] != 'full'
            or config.get('image_suite') != GPU_C5_SUITE or 'record_span' in config):
        raise ValueError('block merge requires the full final C5 evaluation contract')
    evidence, training_rows = matched_evidence(config)
    _, samples, _ = checked_full_parent(config, training_rows)
    expected = {'center': config['center'], 'evaluation_seed': config['seed'],
        'image_suite': config['image_suite'], 'image_severity': config['image_severity'],
        'model_arms': config['model_arms'], 'execution_mode': config.get('execution_mode', 'reference_v1'),
        'original_baseline_mode': config['original_baseline_mode'],
        'training_results': {arm: item['sha256'] for arm, item in evidence.items()}}
    blocks = checked_center_blocks(entries, samples, expected)
    first = blocks[0]
    conditions = first['cohort']['conditions']
    predictions = [row for block in blocks for row in block['rows']]
    metrics = metrics_from_predictions(predictions, samples, conditions, arms=tuple(config['model_arms']))
    baseline = {**first['baseline'], 'prediction_count': len(predictions),
                'predictions_sha256': original_prediction_digest(predictions)}
    parity = {'mode': 'parent_response_audit', 'baseline_mode': config['original_baseline_mode'],
        **{key: 0 for key in ('checked', 'exact', 'drift', 'canonical_equal', 'canonical_drift', 'canonical_unknown')},
        'examples': []}
    for block in blocks:
        member = block['result']['details']['original_parent_parity']
        for key in ('checked', 'exact', 'drift', 'canonical_equal', 'canonical_drift', 'canonical_unknown'):
            parity[key] += member[key]
        parity['examples'].extend(member['examples'][:max(0, 8 - len(parity['examples']))])
    # Delay writes until every child is validated and all metric calculations succeed.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'batches').mkdir()
    for relative in first['result']['files']:
        if relative.startswith('source_snapshot/'):
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(first['path'].parent / relative, target)
    for block in blocks:
        atomic_json(output / f"batches/{block['start']:05d}.json", block['rows'])
    for name in ('generation_admission.json', 'performance_admission.json'):
        source = first['path'].parent / name
        if source.is_file():
            shutil.copyfile(source, output / name)
    atomic_json(output / 'cohort.json', {**first['cohort'], 'samples': samples})
    atomic_json(output / 'metrics.json', metrics)
    atomic_json(output / 'original_baseline.json', baseline)
    atomic_json(output / 'original_parent_parity.json', parity)
    references = [{'path': str(block['path']), 'sha256': sha256_file(block['path'])} for block in blocks]
    atomic_json(output / 'block_results.json', references)
    runtimes = [json.loads((block['path'].parent / 'runtime.json').read_text()) for block in blocks]
    atomic_json(output / 'runtime.json', {'execution': 'merged_record_blocks', 'blocks': len(blocks),
        'worker_seconds_sum': sum(value['seconds'] for value in runtimes),
        'peak_allocated_bytes': max(value['peak_allocated_bytes'] for value in runtimes),
        'augmented_images_persisted': 0})
    details = {key: value for key, value in first['result']['details'].items()
               if key not in {'record_span', 'parent_cohort_identity'}}
    details.update(phase='final', full_cohort=True, records=len(samples), families=metrics['families'],
        cohort_identity=digest_json(samples), original_parent_parity=parity,
        original_baseline_sha256=sha256_file(output / 'original_baseline.json'),
        training_result_paths={arm: str(item['path']) for arm, item in evidence.items()},
        merged_blocks=len(blocks))
    finalize(output, mode='evaluate', details=details)
    path = output / 'hybrid_result.json'
    return {'path': str(path), 'sha256': sha256_file(path)}
