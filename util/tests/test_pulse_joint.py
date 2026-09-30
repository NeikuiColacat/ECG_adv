"""Paired waveform/image acquisition probes: no model downloads or GPU."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from core.image_stress import IMAGE_PROFILES, apply_image_stress
from util.evaluation.pulse_hybrid_development import joint_conditions_for, metrics_from_predictions, single_operator_joint_conditions_for
from util.pulse_hybrid_contract import validate_config
from util.evaluation.ecg_image_artifact import parse_response
from util.evaluation.ecg_image_queue import digest_json
from util.evaluation.pulse_hybrid_development import processor_input_identity, PROCESSOR_INPUT_HASH_MODE

ROOT = Path(__file__).resolve().parents[2]


def answer_fixture(labels):
    response = ", ".join(labels)
    return {"response": response, **parse_response(response), "hit_max_new_tokens": False}


def test_gpu_c5_conditions_match_operator_owner_without_mutating_parent():
    from core.image_augmix_gpu import C5_GPU_OPERATORS, GPU_C5_IMPLEMENTATION
    from util.evaluation.pulse_hybrid_development import c5_gpu_conditions_for, c5_gpu_image_seed
    parent = [{"condition_id": "clean", "operators": []}]
    saved = copy.deepcopy(parent)
    conditions = c5_gpu_conditions_for(parent)
    assert parent == saved
    assert len(conditions) == len({c["condition_id"] for c in conditions}) == 6
    assert conditions[0]["family"] == "clean"
    assert tuple(c["image_operator"] for c in conditions[1:]) == C5_GPU_OPERATORS
    for c in conditions[1:]:
        assert c["operators"] == [] and c["family"] == "image"
        assert c["image_implementation"] == GPU_C5_IMPLEMENTATION and c["image_severity"] == 5
        assert c5_gpu_image_seed(7, "sample", c) == c5_gpu_image_seed(7, "sample", dict(c))
        assert c5_gpu_image_seed(7, "sample", c) != c5_gpu_image_seed(7, "other", c)
    assert len({c5_gpu_image_seed(7, "sample", c) for c in conditions[1:]}) == 5
    for severity in (5.0, True, "5", 0, 3, 6):
        with pytest.raises(ValueError):
            c5_gpu_conditions_for(parent, severity=severity)
    with pytest.raises(ValueError):
        c5_gpu_conditions_for([{"condition_id": "clean", "operators": ["noise"]}])


def test_gpu_c5_config_has_strict_severity_and_exact_three_arms():
    c = yaml.safe_load((ROOT / "configs/eval/pulse_image_c5_gpu_ningbo.yaml").read_text())
    validate_config(c)
    for key, value in (("image_severity", 5.0), ("image_severity", True),
                       ("image_severity", "5"), ("waveform_indices", [1]),
                       ("model_arms", ["original", "clean", "single", "three"])):
        with pytest.raises(ValueError):
            validate_config({**c, key: value})
    bad = copy.deepcopy(c)
    del bad["model_arms"]
    bad["training_results"]["clean"] = bad["training_results"]["single"]
    with pytest.raises(ValueError):
        validate_config(bad)


def test_gpu_c5_metrics_keep_only_clean_and_image_families_and_reject_identity_drift():
    from util.evaluation.pulse_hybrid_development import c5_gpu_conditions_for
    from util.tests.test_pulse_hybrid_workflow import fixture
    _, samples, _ = fixture()
    samples = [{**s, "hash_id": f"fixture-{s['sample_key']}"} for s in samples]
    conditions = c5_gpu_conditions_for([{"condition_id": "clean", "operators": []}])
    arms = ("original", "single", "three")
    rows = [{**c, "sample_key": s["sample_key"], "hash_id": s["hash_id"], "true_labels": s["label_names"],
             "arms": {a: answer_fixture(s["label_names"]) for a in arms}} for s in samples for c in conditions]
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    assert len(metrics["per_condition"]) == 18
    assert all(set(v) == {"clean", "image"} for v in metrics["families"].values())
    for key, value in (("hash_id", "other"), ("image_implementation", "reference"),
                       ("image_operator", "other"), ("image_severity", 3), ("family", "waveform")):
        bad = copy.deepcopy(rows)
        bad[1][key] = value
        with pytest.raises(ValueError, match="identity changed"):
            metrics_from_predictions(bad, samples, conditions, arms=arms)


@pytest.mark.parametrize("phase,random_subset,performance_smoke", [
    ("smoke", False, False), ("screen", False, False), ("screen", True, False),
    ("final", True, False), ("screen", True, True)])
@pytest.mark.parametrize("execution", ["reference_v1", "arm_major_fp16_adapters_v2", "arm_major_original_bypass_v3"])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("suite", ["image_c5_gpu_v1", "paper_ecg_gpu_v1", "paper_ecg_gpu_v2", "paper_ecg_gpu_v3"])
def test_gpu_image_result_roundtrip_and_tamper_rejection(tmp_path, monkeypatch, phase, execution, random_subset, batch_size, performance_smoke, suite):
    import json
    from util import pulse_hybrid_contract as contract
    from util.evaluation.pulse_hybrid_development import gpu_image_conditions_for
    from util.pn2021_artifact_contract import sha256_file
    from util.pulse_training_contract import CLASS_ORDER, validate_result as validate_training
    from util.tests.test_pulse_hybrid_workflow import fixture
    from core.pulse_finetune import MODEL_HASHES, MODEL_CONFIG_HASHES
    from util.evaluation.pulse_hybrid_development import original_prediction_digest

    paper = suite in contract.PAPER_IMPLEMENTATIONS
    implementation = contract.PAPER_IMPLEMENTATIONS.get(suite, "image_c5_torch_gpu_v1")
    monkeypatch.setattr(contract, "DATA", tmp_path)
    training_paths, training_hashes = {}, {}
    for arm, width in (("single", 1), ("three", 3)):
        directory = tmp_path / arm
        directory.mkdir()
        config_path = ("configs/train/pulse_full_lora32_template.yaml" if phase != "smoke"
                       else f"configs/train/pulse_augmix_gpu_{arm}_ningbo.yaml")
        if paper:
            config_path = f"configs/train/pulse_paper_{arm}_ningbo" + ("_smoke" if phase == "smoke" else "") + ".yaml"
            if suite == "paper_ecg_gpu_v2":
                config_path = config_path.replace("pulse_paper_", "pulse_paper_upstream_")
            elif suite == "paper_ecg_gpu_v3":
                config_path = config_path.replace("pulse_paper_", "pulse_paper_print_")
        config = yaml.safe_load((ROOT / config_path).read_text())
        config["width"] = width
        config["training"].update(optimizer_steps=200 if phase != "smoke" else 4, warmup_steps=0)
        records = [{"hash_id": f"k500-{i}", "logical_center": "ningbo"} for i in range(500)]
        (directory / "k500_records.json").write_text(json.dumps(records))
        (directory / "checkpoint.pt").write_bytes(b"synthetic contract fixture, not model weights")
        (directory / "visual_training_admission.json").write_text(json.dumps({
            "frozen_parameter_versions_unchanged": True, "trainable_vision_outputs_cached": 0,
            "vision_lora_updated": True, "projector_updated": True, "language_lora_updated": True}))
        protocol = {"center": "ningbo", "width": width, "mapping_hash": "555ec85d5b51",
            "model_asset_sha256": {**MODEL_HASHES, **MODEL_CONFIG_HASHES},
            "class_order": list(CLASS_ORDER), "adaptation_records": 500, "partition": "k500",
            "simclr": False, "vae_lhat": False, "jsd": True,
            "training_scope": config.get("training_scope", "clip_last4_qv_lora8_projector_frozen_llm"),
            "model": config["model"],
            "image_augmentation": config["image_augmentation"], "training": config["training"],
            "augmentation_topology": "image_only_gpu_branches_v1", "mix_residual": "clean_render",
            "dirichlet_alpha": 1., "clean_mix": "beta_1_1", "jsd_weight": config["image_augmentation"]["jsd_weight"],
            "image_gpu": {"implementation": implementation if paper else "augmix_torch_gpu_v2",
                "device_policy": "same_device_as_rendered_rgb", "waveform_corruption": "disabled",
                "parity": "procedural_appearance_not_author_pixel_equivalence" if paper else "visual_approximation_not_reference_pixel_equivalence",
                "host_tensor_transfer": "none_inside_prevalidated_operator" if paper else "none_inside_operator"}}
        if suite == "paper_ecg_gpu_v3":
            protocol["image_gpu"].update({
                "host_tensor_transfer": "small_parameter_uploads_and_one_time_texture_bank_no_image_readback",
                "randomness": "caller_owned_torch_generator_on_input_device",
                "augraphy_commit": "ed4dcbdaf7b1da6ef59ac60816f96ea253da4c9b",
                "upstream_parity": "v2_color_ink_fixtures_only_print_tolerances_reported_separately",
                "low_ink_lines": "row_only_lightening_variant", "sampling": "project_severity_and_private_torch_rng",
                "ecg_image_kit_commit": "27b90f56896c9fc78b05a83ca14844ea2637aa0b",
                "wrinkle_bank": "9b01be9b7f7c2d575ce57fab8d5551ee959a46a45a458d6615070fa8d63a0dfb",
                "print_parameter_sampling": "private_python_from_caller_seed_counter_v1",
                "texture_upload": "one_verified_bank_per_device_then_resident",
                "print_parity": "fixed_parameter_kernels_tested_sampling_and_interpolation_versioned"})
        path = directory / "train_result.json"
        trained = {"artifact_type": "pulse_train_result", "schema_version": 1, "status": "complete",
            "mode": "smoke" if phase == "smoke" else "train", "optimizer_steps": config["training"]["optimizer_steps"], "protocol": protocol,
            "files": {p.name: sha256_file(p) for p in directory.iterdir()}}
        path.write_text(json.dumps(trained))
        validate_training(trained, path)
        if suite == "paper_ecg_gpu_v3":
            for key in ("wrinkle_bank", "print_parameter_sampling", "augraphy_commit"):
                bad = copy.deepcopy(trained)
                bad["protocol"]["image_gpu"][key] = "changed"
                with pytest.raises(ValueError, match="implementation identity"):
                    validate_training(bad, path)
            bad = copy.deepcopy(trained)
            del bad["protocol"]["image_gpu"]["wrinkle_bank"]
            with pytest.raises(ValueError, match="implementation identity"):
                validate_training(bad, path)
        training_paths[arm], training_hashes[arm] = str(path), sha256_file(path)

    output = tmp_path / "evaluation"
    (output / "batches").mkdir(parents=True)
    # Independent minimal snapshot inventory required by the C5 runtime contract.
    sources = ("util/evaluation/pulse_hybrid_development.py", "util/evaluation/pulse_adapters.py", "core/pulse_finetune.py",
        "util/evaluation/ecg_image_data.py", "util/evaluation/ecg_image_queue.py", "util/evaluation/pn2021.py",
        "util/evaluation/metrics.py", "util/ecg_image_renderer.py", "util/augmentations/torch_operators.py",
        "util/config_bundle.py", "util/pn2021_artifact_contract.py", "util/random_seed.py", "util/run_record.py",
        "data_preprocess/load_cache.py", "data_preprocess/PN2021_preprocess.py", "data_preprocess/pn2021_metadata.py",
        "data_preprocess/data_runtime.py", "data_preprocess/preprocess_primitives.py", "core/image_augmix_gpu.py",
        "core/image_augmix_c.py", "core/paper_ecg.py", "core/image_corruption.py", "core/image_stress.py", "util/augmentations/profile.py",
        "util/evaluation/ecg_image_artifact.py", "util/evaluation/ecg_image_elastic.py",
        "util/evaluation/pulse_visual_subset.py", "util/pulse_hybrid_contract.py", "util/pulse_training_contract.py",
        "models/checkpoints.py", "models/contracts.py", "models/input_adapter.py")
    if suite in ("paper_ecg_gpu_v2", "paper_ecg_gpu_v3"):
        sources += ("core/paper_ecg_upstream.py",)
    if suite == "paper_ecg_gpu_v3":
        sources += ("core/paper_ecg_print.py", "core/paper_ecg_quilting.py")
    for source in sources:
        member = output / "source_snapshot" / source
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes((ROOT / source).read_bytes())
    _, samples, _ = fixture()
    samples = [{**s, "hash_id": f"heldout-{s['sample_key']}"} for s in samples[:4]]
    from util.evaluation import pulse_hybrid_development as evaluator
    sampling = {}
    if random_subset:
        population = [{**samples[i % 4], "sample_key": f"ningbo:heldout-{i}", "hash_id": f"heldout-{i}"} for i in range(64)]
        count = 16 if phase == "screen" else 32
        samples = evaluator.select_image_samples(population, count, sampling_seed=20260924)
        monkeypatch.setattr(evaluator, "checked_full_parent", lambda *args, **kwargs: ({}, population, []))
        sampling = {"sampling_method": "seeded_hash_v1", "sampling_source_state": str(tmp_path),
            "sampling_population_identity": digest_json(population)}
    count = len(samples)
    batch_shapes = {min(batch_size, count)} | ({count % batch_size} if count % batch_size else set())
    conditions = gpu_image_conditions_for([{"condition_id": "clean", "operators": []}], suite=suite, severity=5)
    arms = ["original", "single", "three"]
    rows = [{**c, "sample_key": s["sample_key"], "hash_id": s["hash_id"], "true_labels": s["label_names"],
             "processor_input_identity_sha256": processor_input_identity(seed=20260924, sample=s,
                 condition=c, image_suite=suite, renderer_identity=digest_json({})),
             "processor_input_hash_mode": PROCESSOR_INPUT_HASH_MODE,
             "arms": {a: answer_fixture(s["label_names"]) for a in arms}} for s in samples for c in conditions]
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    for name, value in {"cohort.json": {"samples": samples, "conditions": conditions, "k500_overlap": 0, "renderer": {}},
            "runtime.json": {},
            "batches/0000.json": rows, "metrics.json": metrics,
            "generation_admission.json": {f"{a}_batch{b}": {"packing_exact": True, "tokens_exact": True} for a in arms for b in batch_shapes}}.items():
        (output / name).write_text(json.dumps(value))
    baseline = {"mode": "current_protocol_v1", "batch_size": batch_size, "precision": "float16",
        "seed": 20260924, "deterministic_algorithms": True,
        "model_asset_sha256": {**MODEL_HASHES, **MODEL_CONFIG_HASHES},
        "generation_kwargs": {"do_sample": False, "max_new_tokens": 32, "use_cache": True, "pad_token_id": 2},
        "environment": {"torch": "synthetic", "cuda": "synthetic", "attention_implementation": "sdpa"},
        "prompt_sha256": "b" * 64, "restore_admission": {"tokens_exact_after_adapter_switches": True},
        "prediction_count": len(rows), "predictions_sha256": original_prediction_digest(rows),
        "historical_parent_role": "diagnostic_only_not_metric_baseline"}
    parity = {"mode": "parent_response_audit", "baseline_mode": "current_protocol_v1",
        "checked": count, "exact": count-1, "drift": 1, "canonical_equal": 0, "canonical_drift": 1, "canonical_unknown": 0}
    (output / "original_baseline.json").write_text(json.dumps(baseline))
    (output / "original_parent_parity.json").write_text(json.dumps(parity))
    if execution != "reference_v1":
        check_indices = evaluator.image_admission_indices(count, batch_size, every_batch=phase == "smoke" or performance_smoke)
        performance = {"execution_mode": execution, "batch_size": batch_size,
            "trainable_vision_outputs_cached": 0, "inference_trainable_dtype": "float16",
            "reference_parameter_storage_restored": True,
            **({"original_zero_lora_bypassed": True} if execution == "arm_major_original_bypass_v3" else {}),
            "records": [{"sample_key": s["sample_key"], "tokens_exact": True, "views": len(conditions), "arms": 3}
                        for s in [samples[i] for i in check_indices]]}
        (output / "performance_admission.json").write_text(json.dumps(performance))
        batches = [{"offset": start, "sample_keys": [s["sample_key"] for s in samples[start:start + batch_size]],
            "actual_batch_size": len(samples[start:start + batch_size]),
            "seconds": {"optimized": 1.0, **({"reference": 1.5} if start in check_indices else {}),
                **({"baseline": 1.2} if execution == "arm_major_original_bypass_v3" and performance_smoke else {})},
            "reference_checked": start in check_indices, "warmup_excluded": True,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
            for start in range(0, count, batch_size)]
        (output / "performance_batches.json").write_text(json.dumps(batches))
    contract.finalize(output, mode="evaluate", details={"development_only": True, "center": "ningbo",
        "original_baseline_mode": "current_protocol_v1", "evaluation_seed": 20260924,
        "original_baseline_sha256": sha256_file(output / "original_baseline.json"),
        "original_parent_parity": parity,
        "phase": phase, "execution_mode": execution, "inference_batch_size": batch_size,
        "performance_smoke": performance_smoke,
        "records": count, "conditions": len(conditions), "image_suite": suite, **sampling,
        "cohort_identity": digest_json(samples),
        "image_severity": 5, "image_implementation": implementation, "model_arms": arms,
        "training_result_paths": training_paths, "training_results": training_hashes, "families": metrics["families"]})
    path = output / "hybrid_result.json"
    result = json.loads(path.read_text())
    contract.validate_result(result, path)
    wrong_phase = copy.deepcopy(result)
    wrong_phase["details"]["phase"] = "screen" if phase == "smoke" else "smoke"
    with pytest.raises(ValueError):
        contract.validate_result(wrong_phase, path)
    if execution != "reference_v1":
        bad_fields = [("records", []), ("reference_parameter_storage_restored", False)]
        if execution == "arm_major_original_bypass_v3":
            bad_fields.append(("original_zero_lora_bypassed", False))
        for field, value in bad_fields:
            altered_admission = {**performance, field: value}
            member = output / "performance_admission.json"
            member.write_text(json.dumps(altered_admission))
            bad = copy.deepcopy(result)
            bad["files"][member.name] = sha256_file(member)
            with pytest.raises(ValueError):
                contract.validate_result(bad, path)
        member.write_text(json.dumps(performance))
    for key, value in (("batch_size", 5), ("batch_size", float(batch_size)), ("predictions_sha256", "0" * 64),
                       ("model_asset_sha256", {}), ("restore_admission", {}), ("prediction_count", 23)):
        with pytest.raises(ValueError):
            contract.validate_original_baseline({**baseline, key: value}, rows, result["details"])
    changed = copy.deepcopy(rows)
    changed[0]["arms"]["original"]["predicted_labels"] = ["MI"]
    with pytest.raises(ValueError, match="predictions changed"):
        contract.validate_original_baseline(baseline, changed, result["details"])
    for key, value in (("image_severity", 5.0), ("image_implementation", "reference"),
                       ("training_result_paths", {}), ("training_results", {"single": "0"*64, "three": "0"*64})):
        bad = copy.deepcopy(result)
        bad["details"][key] = value
        with pytest.raises(ValueError):
            contract.validate_result(bad, path)
    bad = copy.deepcopy(result)
    del bad["files"]["source_snapshot/core/image_augmix_gpu.py"]
    with pytest.raises(ValueError, match="snapshot is incomplete"):
        contract.validate_result(bad, path)
    bad = copy.deepcopy(result)
    bad["details"]["cohort_identity"] = "0" * 64
    with pytest.raises(ValueError, match="cohort identity"):
        contract.validate_result(bad, path)
    bad = copy.deepcopy(result)
    del bad["files"]["cohort.json"]
    with pytest.raises(ValueError, match="integrity coverage"):
        contract.validate_result(bad, path)
    altered = copy.deepcopy(rows)
    altered[0]["processor_input_identity_sha256"] = "0" * 64
    (output / "batches/0000.json").write_text(json.dumps(altered))
    bad = copy.deepcopy(result)
    bad["files"]["batches/0000.json"] = sha256_file(output / "batches/0000.json")
    with pytest.raises(ValueError, match="input recipe"):
        contract.validate_result(bad, path)
    (output / "batches/0000.json").write_text(json.dumps(rows))
    altered_samples = copy.deepcopy(samples)
    altered_samples[0]["hash_id"] = "k500-0"
    altered_cohort = json.loads((output / "cohort.json").read_text())
    altered_cohort["samples"] = altered_samples
    (output / "cohort.json").write_text(json.dumps(altered_cohort))
    bad = copy.deepcopy(result)
    bad["files"]["cohort.json"] = sha256_file(output / "cohort.json")
    bad["details"]["cohort_identity"] = digest_json(altered_samples)
    with pytest.raises(ValueError, match="overlaps adaptation K500"):
        contract.validate_result(bad, path)


def test_metric_truth_rejects_coercion_and_label_name_disagreement():
    from util.tests.test_pulse_hybrid_workflow import fixture
    rows, samples, conditions = fixture()
    for value in ([256, 0, 0, 0, 0], [1, 0, 0, 0, 0.5], [True, 0, 0, 0, 0]):
        bad = copy.deepcopy(samples)
        bad[0]["label"] = value
        with pytest.raises(ValueError, match="binary Super5"):
            metrics_from_predictions(rows, bad, conditions)
    samples[0]["label_names"] = ["MI"]
    with pytest.raises(ValueError, match="label-name"):
        metrics_from_predictions(rows, samples, conditions)


def test_c5_predictions_reparse_original_response():
    from util.tests.test_pulse_hybrid_workflow import fixture
    from util.evaluation.pulse_hybrid_development import c5_gpu_conditions_for
    _, samples, _ = fixture()
    samples = [{**s, "hash_id": s["sample_key"]} for s in samples]
    conditions = c5_gpu_conditions_for([{"condition_id": "clean", "operators": []}])
    arms = ("original", "single", "three")
    rows = [{**c, "sample_key": s["sample_key"], "hash_id": s["hash_id"],
        "true_labels": s["label_names"], "arms": {a: answer_fixture(s["label_names"]) for a in arms}}
        for s in samples for c in conditions]
    rows[0]["arms"]["single"]["response"] = "MI"
    with pytest.raises(ValueError, match="reproduce from response"):
        metrics_from_predictions(rows, samples, conditions, arms=arms)


def test_full_parent_rejects_changed_cohort_before_predictions(tmp_path, monkeypatch):
    import json
    from util.evaluation import pulse_hybrid_development as owner
    from util.pn2021_artifact_contract import sha256_file
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"cohort": {"sha256": "0" * 64}}))
    (tmp_path / "_SUCCESS.json").write_text("{}")
    (tmp_path / "cohort.jsonl").write_text("changed\n")
    monkeypatch.setattr(owner, "FULL_PARENT_PROTOCOL_SHA256", sha256_file(protocol))
    with pytest.raises(ValueError, match="cohort SHA256 changed"):
        owner.checked_full_parent({"source_state": str(tmp_path)})


@pytest.mark.parametrize("profile", IMAGE_PROFILES)
def test_image_stress_replay_range_and_private_rng(profile):
    image = torch.linspace(0, 1, 2*3*40*60).reshape(2, 3, 40, 60)
    before, state = image.clone(), torch.get_rng_state().clone()
    def apply():
        return apply_image_stress(image, profile, rng=torch.Generator().manual_seed(42))
    actual = apply()
    assert torch.equal(actual, apply()) and torch.equal(image, before)
    assert torch.equal(state, torch.get_rng_state())
    assert actual.shape == image.shape and torch.isfinite(actual).all()
    assert 0 <= actual.min() <= actual.max() <= 1 and not torch.equal(actual, image)


def test_perspective_retains_corner_landmarks():
    image = torch.ones(1, 3, 128, 128)
    for y,x in [(4,4), (4,120), (120,4), (120,120)]:
        image[:, :, y:y+4, x:x+4] = 0
    for seed in range(10):
        result = apply_image_stress(image, "perspective", rng=torch.Generator().manual_seed(seed))
        for y,x in [(0,0),(0,64),(64,0),(64,64)]:
            assert result[:, :, y:y+64, x:x+64].min() < 0.5


def test_joint_full_cross_product_keeps_wave_seed_identity():
    parent = [{"condition_id": "clean" if i == 0 else f"wave{i}", "operators": [] if i == 0 else [f"op{i}"]} for i in range(21)]
    conditions = joint_conditions_for(parent, list(range(1,21)))
    assert len(conditions) == 84 and len({c['condition_id'] for c in conditions}) == 84
    assert sum(c['family'] == 'joint' for c in conditions) == 60
    for c in conditions:
        if c['family'] == 'joint':
            original = next(w for w in parent if w['condition_id'] == c['waveform_condition_id'])
            assert c['operators'] == original['operators']
    assert len(joint_conditions_for(parent,[1,10,11,20])) == 20


def test_joint_config_is_frozen():
    c = yaml.safe_load((ROOT/'configs/eval/pulse_joint_ningbo.yaml').read_text())
    validate_config(c)
    for key,value in [('waveform_indices',[1]),('image_suite','unknown'),('records_per_center',127)]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError): validate_config(bad)


def test_single_operator_suite_has_five_plus_three_plus_fifteen_conditions():
    from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
    parent = [{"condition_id": "clean", "operators": []}]
    conditions = single_operator_joint_conditions_for(parent, CANONICAL_OPERATOR_ORDER)
    assert len(conditions) == 24
    assert sum(c["family"] == "clean" for c in conditions) == 1
    assert sum(c["family"] == "waveform" for c in conditions) == 5
    assert sum(c["family"] == "image" for c in conditions) == 3
    assert sum(c["family"] == "joint" for c in conditions) == 15
    assert all(len(c["operators"]) == 1 for c in conditions if c["family"] in ("waveform", "joint"))


def test_reference_c15_rectangle_and_image_realization_are_paired():
    from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
    from util.evaluation.pulse_hybrid_development import c15_joint_conditions_for, c15_image_seed
    parent = [{"condition_id": "clean", "operators": []}]
    conditions = c15_joint_conditions_for(parent, CANONICAL_OPERATOR_ORDER, severity=3, reference=True)
    assert len(conditions) == len({c["condition_id"] for c in conditions}) == 96
    assert {family: sum(c["family"] == family for c in conditions)
            for family in ("clean", "waveform", "image", "joint")} == {
                "clean": 1, "waveform": 5, "image": 15, "joint": 75}
    for joint in (c for c in conditions if c["family"] == "joint"):
        image = next(c for c in conditions if c["family"] == "image" and c["image_operator"] == joint["image_operator"])
        assert c15_image_seed(7, "record", joint) == c15_image_seed(7, "record", image)
        wave = next(c for c in conditions if c["condition_id"] == joint["waveform_condition_id"])
        assert joint["operators"] == wave["operators"] and len(wave["operators"]) == 1
    legacy = c15_joint_conditions_for(parent, CANONICAL_OPERATOR_ORDER)
    assert {c["condition_id"] for c in conditions if c["family"] == "image"}.isdisjoint(
        c["condition_id"] for c in legacy if c["family"] == "image")
    with pytest.raises(ValueError):
        c15_joint_conditions_for(parent, ["invented"] * 5, reference=True)


def test_new_single_wave_ids_do_not_lookup_nonexistent_frozen_predictions():
    from util.evaluation.pulse_hybrid_development import verify_original_answer
    answer = {"response": "NORM", "generated_tokens": 2, "hit_max_new_tokens": False}
    originals = {("record", "clean"): {"answer": answer}}
    for suite in ("joint_v3", "joint_c15", "joint_c15_reference_v1"):
        verify_original_answer(originals, "record", {"condition_id": "new_single_wave", "family": "waveform"},
                               answer, image_suite=suite)
        verify_original_answer(originals, "record", {"condition_id": "clean", "family": "clean"},
                               answer, image_suite=suite)
        with pytest.raises(ValueError, match="frozen parent"):
            verify_original_answer(originals, "record", {"condition_id": "clean", "family": "clean"},
                                   {**answer, "response": "MI"}, image_suite=suite)
        with pytest.raises(ValueError, match="only admitted"):
            verify_original_answer(originals, "record", {"condition_id": "clean", "family": "clean"},
                                   {**answer, "response": "MI"}, image_suite=suite,
                                   baseline_mode="current_protocol_v1")
    with pytest.raises(KeyError):
        verify_original_answer(originals, "record", {"condition_id": "missing", "family": "waveform"},
                               answer, image_suite="joint_v2")


def test_parent_parity_distinguishes_formatting_from_real_label_drift():
    from util.evaluation.pulse_hybrid_development import verify_original_answer
    previous = {"response": "CD; STTC", "generated_tokens": 6, "hit_max_new_tokens": False}
    originals = {("A3512", "clean"): {"answer": previous}}
    condition = {"condition_id": "clean", "family": "clean"}
    reordered = {**previous, "response": "STTC; CD", "generated_tokens": 7}
    result = verify_original_answer(originals, "A3512", condition, reordered, image_suite="image_c5_gpu_v1")
    assert result["canonical_equal"] is True and result["exact"] is False
    for response in ("STTC", "unparseable"):
        changed = {**previous, "response": response}
        with pytest.raises(ValueError, match="frozen parent"):
            verify_original_answer(originals, "A3512", condition, changed, image_suite="image_c5_gpu_v1")
        audit = verify_original_answer(originals, "A3512", condition, changed,
            image_suite="image_c5_gpu_v1", baseline_mode="current_protocol_v1")
        assert audit["canonical_equal"] is not True
    with pytest.raises(ValueError, match="frozen parent"):
        verify_original_answer(originals, "A3512", condition, {**reordered, "hit_max_new_tokens": True},
                               image_suite="image_c5_gpu_v1")


def test_legacy_c15_evaluation_fails_before_output_or_model_loading(tmp_path):
    from util.evaluation.pulse_hybrid_development import run
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="historical only"):
        run({"image_suite": "joint_c15"}, {}, ROOT / "configs", output)
    assert not output.exists()


@pytest.mark.parametrize("arms", [("original", "clean", "single", "three"), ("original", "single", "three")])
def test_reference_c15_live_config_and_result_contract(tmp_path, arms):
    import json
    from core.image_augmix_c import c_reference_identity
    from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
    from util.evaluation.pulse_hybrid_development import c15_joint_conditions_for
    from util.pulse_hybrid_contract import finalize, validate_result
    from util.pn2021_artifact_contract import sha256_file
    from util.tests.test_pulse_hybrid_workflow import fixture

    config = yaml.safe_load((ROOT / "configs/eval/pulse_c15_reference_ningbo.yaml").read_text())
    if "clean" not in arms:
        config["model_arms"] = list(arms)
        del config["training_results"]["clean"]
    validate_config(config)
    for value in (True, "3", 0, 6):
        with pytest.raises(ValueError):
            validate_config({**config, "image_severity": value})
    _, samples, _ = fixture()
    samples = samples[:4]
    conditions = c15_joint_conditions_for([{"condition_id": "clean", "operators": []}],
        CANONICAL_OPERATOR_ORDER, severity=3, reference=True)
    rows = [{"sample_key": s["sample_key"], "condition_id": c["condition_id"], "true_labels": s["label_names"],
        "arms": {a: {"predicted_labels": s["label_names"], "parse_valid": True, "hit_max_new_tokens": False}
                 for a in arms}} for s in samples for c in conditions]
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    reference = c_reference_identity()
    cohort = {"samples": samples, "conditions": conditions, "k500_overlap": 0, "image_reference": reference}
    (tmp_path / "batches").mkdir()
    for name, value in {"cohort.json": cohort, "batches/0000.json": rows, "metrics.json": metrics,
                       "generation_admission.json": {f"{a}_batch2": {"packing_exact": True, "tokens_exact": True}
                                                     for a in arms}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    finalize(tmp_path, mode="evaluate", details={"development_only": True, "center": "ningbo", "phase": "smoke",
        "records": 4, "conditions": 96, "image_suite": "joint_c15_reference_v1", "image_severity": 3,
        "image_reference": reference, "families": metrics["families"],
        **({"model_arms": list(arms)} if "clean" not in arms else {})})
    path = tmp_path / "hybrid_result.json"
    payload = json.loads(path.read_text())
    validate_result(payload, path)
    bad = copy.deepcopy(payload)
    bad["details"]["image_severity"] = "3"
    with pytest.raises(ValueError, match="invalid C15"):
        validate_result(bad, path)
    bad = copy.deepcopy(payload)
    bad["details"]["image_reference"] = {}
    with pytest.raises(ValueError, match="reference environment"):
        validate_result(bad, path)
    cohort["conditions"][6]["image_severity"] = 5
    (tmp_path / "cohort.json").write_text(json.dumps(cohort))
    payload["files"]["cohort.json"] = sha256_file(tmp_path / "cohort.json")
    with pytest.raises(ValueError, match="Cartesian product"):
        validate_result(payload, path)


def test_joint_metrics_have_four_separate_families():
    from util.tests.test_pulse_hybrid_workflow import fixture
    rows,samples,conditions=fixture()
    conditions.append({'condition_id':'joint','family':'joint'})
    rows += [{**copy.deepcopy(row), 'condition_id':'joint'} for row in rows if row['condition_id']=='image']
    metrics=metrics_from_predictions(rows,samples,conditions)
    assert all(set(v)=={'clean','waveform','image','joint'} for v in metrics['families'].values())
    assert all(v['joint']==1 for v in metrics['families'].values())


def test_joint_summary_reports_incremental_image_effect(tmp_path, monkeypatch):
    import json
    from util.evaluation.pulse_hybrid_development import summarize_four_centers
    from util.pn2021_artifact_contract import sha256_file
    from util.pulse_training_contract import CENTERS
    from util.tests.test_pulse_hybrid_workflow import fixture
    from util import pulse_hybrid_contract

    # Isolate aggregation arithmetic; result-integrity gates have separate tests.
    monkeypatch.setattr(pulse_hybrid_contract, 'validate_result', lambda *args: None)
    entries = {}
    for center in CENTERS:
        directory = tmp_path / center
        (directory / 'batches').mkdir(parents=True)
        rows, samples, conditions = fixture()
        conditions.append({'condition_id': 'joint', 'family': 'joint'})
        joint = [{**copy.deepcopy(row), 'condition_id': 'joint'} for row in rows if row['condition_id'] == 'image']
        for row in joint:
            row['arms']['single']['predicted_labels'] = []
        rows += joint
        (directory / 'cohort.json').write_text(json.dumps({'samples': samples, 'conditions': conditions}))
        (directory / 'batches/0000.json').write_text(json.dumps(rows))
        (directory / 'metrics.json').write_text(json.dumps(metrics_from_predictions(rows, samples, conditions)))
        path = directory / 'hybrid_result.json'
        path.write_text(json.dumps({'details': {'phase': 'screen', 'center': center}, 'files': {'batches/0000.json': 'fixture'}}))
        entries[center] = {'path': str(path), 'sha256': sha256_file(path)}
    output = tmp_path / 'summary'
    summarize_four_centers(entries, output, seed=42, phase='screen', families=('clean', 'waveform', 'image', 'joint'))
    report = json.loads((output / 'comparison.json').read_text())
    effects = {r['arm']: r for r in report['incremental_image_effect']}
    assert effects['single']['delta_pp'] == -100
    assert effects['three']['delta_pp'] == 0
    delta = next(r for r in report['paired_deltas'] if r['comparison'] == 'three_minus_single' and r['family'] == 'joint')
    assert delta['delta_pp'] == 100
    assert 0 <= delta['ci95_pp'][0] <= delta['ci95_pp'][1] <= 100
