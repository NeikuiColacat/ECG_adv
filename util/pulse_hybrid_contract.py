"""CPU configuration/result contracts for the finite hybrid workflow."""
from __future__ import annotations

import json
import math
from pathlib import Path

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.evaluation.ecg_image_queue import atomic_json
from util.pn2021_artifact_contract import sha256_file
from util.pulse_training_contract import CENTERS

DATA = Path("/home/linbinhao/ECG_adv_data")
RAM = Path("/dev/shm/linbinhao-pulse-hybrid")
RAM_PREFIX_ROOT = Path("/dev/shm")
RAM_PREFIX = "linbinhao-pulse-"
ARCHIVE_ROOT = Path("/data/linbinhao/ecg_llm_runs")
C5_ORIGINAL_BYPASS = "arm_major_original_bypass_v3"
C5_FP16_MODES = ("arm_major_fp16_adapters_v2", C5_ORIGINAL_BYPASS)
C5_OPTIMIZED_MODES = ("arm_major_cached_prompt_v1", *C5_FP16_MODES)


def is_scoped_ram_path(value: Path) -> bool:
    """Accept only this user's explicitly named shared-memory roots."""
    if value.is_relative_to(RAM):
        return True
    try:
        relative = value.relative_to(RAM_PREFIX_ROOT)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].startswith(RAM_PREFIX)


def owned_path(raw):
    value = Path(raw).resolve()
    if not value.is_relative_to(DATA) and not is_scoped_ram_path(value) and not value.is_relative_to(ARCHIVE_ROOT):
        raise ValueError("hybrid runtime paths require the owned ECG data root or scoped RAM")
    return value


def validate_config(c):
    if c.get("schema_version") != 1 or c.get("mode") not in ("search", "evaluate", "fixed"):
        raise ValueError("invalid hybrid workflow mode/schema")
    if c["mode"] == "evaluate":
        c15 = c.get("image_suite") in ("joint_c15", "joint_c15_reference_v1")
        c5 = c.get("image_suite") == "image_c5_gpu_v1"
        joint = c.get("image_suite") in ("joint_v2", "joint_v3") or c15
        reduced = c.get("image_suite") == "joint_v3"
        expected = {"schema_version", "mode", "references", "phase", "center", "records_per_center",
                    "source_state", "training_results", "waveform_indices", "image_strength", "seed"}
        if c.get("phase") == "block":
            expected.add("record_span")
            if not c5 or c.get("records_per_center") != "full":
                raise ValueError("record blocks require the full C5 cohort")
            validate_record_span(c.get("center"), c.get("record_span"))
        elif "record_span" in c:
            raise ValueError("record spans are reserved for block evaluations")
        if joint or c5:
            expected.add("image_suite")
        if c15 or c5:
            expected.add("image_severity")
        if c5:
            expected.add("original_baseline_mode")
            for key in ("execution_mode", "performance_smoke", "inference_batch_size", "sampling_method"):
                if key in c:
                    expected.add(key)
            if (c.get("execution_mode", "reference_v1") not in ("reference_v1", *C5_OPTIMIZED_MODES)
                    or type(c.get("performance_smoke", False)) is not bool
                    or (c.get("performance_smoke", False) and c["phase"] not in ("smoke", "screen"))):
                raise ValueError("invalid C5 execution/performance admission mode")
            if type(c.get("inference_batch_size", 1)) is not int or not 1 <= c.get("inference_batch_size", 1) <= 4:
                raise ValueError("C5 inference batch size must be an integer from one to four")
            if "sampling_method" in c and (c["sampling_method"] != "seeded_hash_v1"
                    or c["phase"] not in ("screen", "final") or type(c["records_per_center"]) is not int):
                raise ValueError("random sampling requires an explicit finite C5 development subset")
            if c["phase"] == "block" and c.get("inference_batch_size", 1) != 1:
                raise ValueError("legacy full-center blocks retain their batch-one grouping")
            if c.get("original_baseline_mode") != "current_protocol_v1":
                raise ValueError("GPU C5 requires an explicit current-protocol original baseline")
            if "admission_sample_keys" in c:
                expected.add("admission_sample_keys")
                keys = c["admission_sample_keys"]
                if (c["phase"] != "smoke" or not isinstance(keys, list) or len(keys) > 4
                        or any(not isinstance(k, str) or not k.startswith(c["center"] + ":") for k in keys)
                        or len(set(keys)) != len(keys)):
                    raise ValueError("regression samples are restricted to their center's smoke admission")
        if "model_arms" in c:
            expected.add("model_arms")
            if (c["model_arms"] != ["original", "single", "three"]
                    or c.get("image_suite") not in ("joint_c15_reference_v1", "image_c5_gpu_v1")):
                raise ValueError("three-arm evaluation requires the reference C15 suite")
        if set(c) != expected or c["phase"] not in ("smoke", "screen", "final", "block") or c["center"] not in CENTERS:
            raise ValueError("invalid hybrid development schema")
        expected_waveforms = ([] if c5 else
            list(range(1, 6)) if reduced or c15 else
            list(range(1, 21)) if c["phase"] == "final" or joint and c["phase"] == "screen" else [1, 10, 11, 20])
        from util.evaluation.pulse_hybrid_development import FULL_CENTER_RECORDS
        random_subset = (c5 and c.get("sampling_method") == "seeded_hash_v1"
            and type(c["records_per_center"]) is int and 16 <= c["records_per_center"] <= FULL_CENTER_RECORDS[c["center"]])
        expected_records = {"smoke": 4, "screen": (c["records_per_center"] if c5 else 128), "final": c["records_per_center"], "block": "full"}[c["phase"]]
        if ((c["records_per_center"] not in (4, 16, 128, 512, "full") and not random_subset)
                or (type(c["records_per_center"]) is not int and c["records_per_center"] != "full")
                or ("sampling_method" in c and not random_subset)
                or (c["phase"] == "screen" and c["records_per_center"] not in ((4, 16, 128) if c5 else (128,)) and not random_subset)
                or (c["phase"] == "final" and c["records_per_center"] not in (512, "full") and not random_subset)
                or c["phase"] not in ("final", "block") and c["records_per_center"] == "full"
                or c["records_per_center"] != expected_records
                or set(c["training_results"]) != ({"single", "three"} if "model_arms" in c else {"clean", "single", "three"})
                or c["waveform_indices"] != expected_waveforms
                or c["image_strength"] != 1.0 or type(c["seed"]) is not int):
            raise ValueError("development cohort or fixed corruption suite changed")
        if c15 and (type(c.get("image_severity")) is not int or not 1 <= c["image_severity"] <= 5):
            raise ValueError("invalid C15 image severity")
        if c5 and (type(c.get("image_severity")) is not int or c["image_severity"] != 5
                   or c.get("model_arms") != ["original", "single", "three"]):
            raise ValueError("GPU C5 evaluation requires integer severity 5 and three model arms")
        for raw in [c["source_state"], *c["training_results"].values()]:
            owned_path(raw)
    elif c["mode"] == "fixed":
        expected = {"schema_version", "mode", "references", "seed", "max_gpu_workers", "centers", "arms",
                    "evidence_role", "records_per_center", "image_suite", "image_severity", "paths", "runtime"}
        full_lora_pipeline = str(c["references"].get("training_config", "")).endswith("pulse_full_lora32_template.yaml")
        for key in ("inference_batch_size", "sampling_method"):
            if key in c:
                expected.add(key)
        if (type(c.get("inference_batch_size", 1)) is not int or not 1 <= c.get("inference_batch_size", 1) <= 4
                or ("sampling_method" in c and c["sampling_method"] != "seeded_hash_v1")
                or (("inference_batch_size" in c or "sampling_method" in c) and c["image_suite"] != "image_c5_gpu_v1")
                or ("evaluation_block_size" in c and (c.get("inference_batch_size", 1) != 1 or "sampling_method" in c))):
            raise ValueError("invalid fixed C5 batch/subset policy")
        random_subset = (c.get("sampling_method") == "seeded_hash_v1" and type(c["records_per_center"]) is int
            and 16 <= c["records_per_center"] <= 5322)
        if "sampling_method" in c and not random_subset:
            raise ValueError("fixed random subset must fit all four centers")
        if "evaluation_block_size" in c:
            expected.add("evaluation_block_size")
            if (not full_lora_pipeline or type(c["evaluation_block_size"]) is not int
                    or c["evaluation_block_size"] != 512 or c["records_per_center"] != "full"
                    or c["image_suite"] != "image_c5_gpu_v1"):
                raise ValueError("block scheduling requires fixed full-LoRA C5 at 512 records")
        if "recovery" in c:
            expected.add("recovery")
            recovery = c["recovery"]
            if (not full_lora_pipeline or not isinstance(recovery, dict)
                    or set(recovery) != {"archive_root", "archive_receipt_sha256"}):
                raise ValueError("recovery requires a pinned full-LoRA archive")
            archive = Path(recovery["archive_root"]).resolve()
            digest = recovery["archive_receipt_sha256"]
            if (archive.parent != ARCHIVE_ROOT or not archive.name.startswith("pulse_full_lora32_")
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError("invalid recovery archive identity")
        if "storage" in c:
            expected.add("storage")
            storage = c["storage"]
            if (not full_lora_pipeline or set(storage) != {"mode", "archive_root",
                    "ram_budget_gib", "ram_reserve_gib", "max_workers", "release_ram_after_archive"}
                    or storage["mode"] != "ram_then_hdd"
                    or storage["ram_budget_gib"] != 40 or storage["ram_reserve_gib"] != 8
                    or type(storage["max_workers"]) is not int or not 1 <= storage["max_workers"] <= 4
                    or storage["release_ram_after_archive"] is not True):
                raise ValueError("invalid bounded RAM-to-HDD storage contract")
            archive = Path(storage["archive_root"]).resolve()
            if archive.parent != ARCHIVE_ROOT or not archive.name.startswith("pulse_full_lora32_"):
                raise ValueError("archive must use the explicitly authorized user HDD namespace")
        if c["image_suite"] == "image_c5_gpu_v1":
            expected |= {"original_baseline_mode", "smoke_sample_keys"}
            if c.get("original_baseline_mode") != "current_protocol_v1":
                raise ValueError("GPU C5 requires an explicit current-protocol original baseline")
            probes = c.get("smoke_sample_keys", {})
            if (set(probes) != set(CENTERS) or any(not isinstance(keys, list) or len(keys) > 4
                    or any(not isinstance(k, str) or not k.startswith(center + ":") for k in keys)
                    or len(set(keys)) != len(keys) for center, keys in probes.items())):
                raise ValueError("fixed C5 requires explicit four-center regression admission samples")
        valid_suite = (c["image_suite"], c["image_severity"]) in {
            ("joint_c15_reference_v1", 3), ("image_c5_gpu_v1", 5)}
        if (set(c) != expected or c["centers"] != list(CENTERS) or c["arms"] != {"single": 1, "three": 3}
                or c["max_gpu_workers"] != 4 or (c["records_per_center"] not in (512, "full") and not random_subset)
                or not valid_suite or type(c["seed"]) is not int
                or c["evidence_role"] != "fixed_recipe_development_comparison"):
            raise ValueError("invalid fixed reference workflow contract")
        if set(c["paths"]) != {"run_root", "source_state", "temporary_root"}:
            raise ValueError("missing fixed workflow paths")
        run_root = owned_path(c["paths"]["run_root"])
        source_state = owned_path(c["paths"]["source_state"])
        temporary_root = owned_path(c["paths"]["temporary_root"])
        if not source_state.is_relative_to(DATA):
            raise ValueError("fixed source paths must remain under the home-owned data root")
        if "storage" in c:
            if (not is_scoped_ram_path(run_root) or not is_scoped_ram_path(temporary_root)
                    or run_root.parent != temporary_root.parent or run_root.name != "run"
                    or temporary_root.name != "tmp"):
                raise ValueError("RAM-to-HDD requires sibling task-scoped run and tmp roots")
        elif not run_root.is_relative_to(DATA):
            raise ValueError("fixed RAM outputs require an explicit archival contract")
        if full_lora_pipeline and not (is_scoped_ram_path(temporary_root) or temporary_root.is_relative_to(DATA)):
            raise ValueError("full-LoRA temporary_root must use owned data or scoped shared memory")
        expected_runtime = {"poll_seconds": 30, "idle_seconds": 60 if full_lora_pipeline else 30,
                            "min_ram_gib": 64, "min_disk_gib": 40}
        if c["runtime"] != expected_runtime:
            raise ValueError("resource floors changed")
    else:
        expected = {"schema_version", "mode", "references", "trials", "seed", "screening_steps", "final_steps",
                    "max_gpu_workers", "centers", "arms", "evidence_role", "search_space", "objective",
                    "tracking", "paths", "runtime"}
        if set(c) != expected or c["centers"] != list(CENTERS) or c["arms"] != {"clean": 0, "single": 1, "three": 3}:
            raise ValueError("invalid finite search schema/grid")
        if (c["trials"] != 8 or c["screening_steps"] != 25 or c["final_steps"] != 100
                or c["max_gpu_workers"] != 4 or c["evidence_role"] != "pn2021_development_validation"):
            raise ValueError("search budget or evidence contract changed")
        if c["objective"] != {"clean": 0.5, "waveform": 0.25, "image": 0.25, "centers": "equal", "arms": "equal"}:
            raise ValueError("search objective changed")
        space = c["search_space"]
        keys = {"learning_rate", "projector_learning_rate", "jsd_weight", "waveform_strength", "image_strength"}
        if set(space) != keys:
            raise ValueError("invalid search dimensions")
        for name, values in space.items():
            cap = 2e-5 if name.endswith("learning_rate") else 24 if name == "jsd_weight" else 1
            if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= cap for v in values):
                raise ValueError(f"unsafe search space: {name}")
            if name.endswith("learning_rate") and (len(values) != 2 or values[0] >= values[1]):
                raise ValueError("learning rate requires increasing bounds")
        if set(c["paths"]) != {"run_root", "source_state", "temporary_root"}:
            raise ValueError("missing workflow paths")
        for raw in c["paths"].values():
            owned_path(raw)
        if c["tracking"] != {"project": "pulse-hybrid-development", "local_only": True,
                "python": str(DATA / "envs/pulse_tracking/bin/python")}:
            raise ValueError("tracking must remain isolated and local")
        if c["runtime"] != {"poll_seconds": 30, "idle_seconds": 30, "min_ram_gib": 64, "min_disk_gib": 40}:
            raise ValueError("resource floors changed")
    if set(c["references"]) != {"training_config"}:
        raise ValueError("hybrid workflow must close over its training template")


def load_config(path, root):
    config = load_yaml_mapping(path, description="PULSE hybrid workflow")
    validate_config(config)
    refs = {name: resolve_config_reference(raw, owner_config_path=path, config_root=root,
        description=name, must_exist=True) for name, raw in config["references"].items()}
    if config["mode"] == "fixed":
        from util.pulse_training_contract import load_config as load_training
        from core.image_corruption import validate_image_config
        template, _ = load_training(refs["training_config"], root)
        validate_image_config(template["image_augmentation"], for_execution=True)
        if (template["image_augmentation"].get("implementation") not in ("augmix_pil_reference_v1", "augmix_torch_gpu_v2")
                or template["image_augmentation"]["severity"] != 3
                or template["training"]["seed"] != config["seed"]
                or template["mode"] != "train" or template["width"] != 3 or template["resume_from"] is not None):
            raise ValueError("fixed reference workflow has the wrong training template")
    return config, refs


def validate_original_baseline(baseline, rows, details):
    from core.pulse_finetune import MODEL_HASHES, MODEL_CONFIG_HASHES
    from util.evaluation.pulse_hybrid_development import original_prediction_digest
    if (baseline.get("mode") != "current_protocol_v1"
            or details.get("original_baseline_mode") != baseline["mode"]
            or type(details.get("inference_batch_size", 1)) is not int or not 1 <= details.get("inference_batch_size", 1) <= 4
            or type(baseline.get("batch_size")) is not int
            or baseline.get("batch_size") != details.get("inference_batch_size", 1) or baseline.get("precision") != "float16"
            or baseline.get("deterministic_algorithms") is not True
            or type(baseline.get("seed")) is not int or baseline["seed"] != details.get("evaluation_seed")
            or baseline.get("model_asset_sha256") != {**MODEL_HASHES, **MODEL_CONFIG_HASHES}
            or baseline.get("restore_admission") != {"tokens_exact_after_adapter_switches": True}
            or baseline.get("historical_parent_role") != "diagnostic_only_not_metric_baseline"):
        raise ValueError("current original baseline identity/admission changed")
    generation = baseline.get("generation_kwargs", {})
    environment = baseline.get("environment", {})
    if (generation.get("do_sample") is not False or generation.get("max_new_tokens") != 32
            or generation.get("use_cache") is not True or type(generation.get("pad_token_id")) is not int
            or not environment.get("torch") or not environment.get("cuda")
            or environment.get("attention_implementation") != "sdpa"
            or len(baseline.get("prompt_sha256", "")) != 64):
        raise ValueError("current original baseline generation contract changed")
    if (baseline.get("prediction_count") != details["records"] * details["conditions"]
            or baseline["prediction_count"] != len(rows)
            or baseline.get("predictions_sha256") != original_prediction_digest(rows)):
        raise ValueError("current original baseline predictions changed")


def validate_result(result, path):
    if (result.get("artifact_type") != "pulse_hybrid_result" or result.get("schema_version") != 1
            or result.get("status") != "complete" or result.get("mode") not in ("search", "evaluate", "fixed")
            or result["details"].get("development_only") is not True):
        raise ValueError("invalid hybrid result/evidence role")
    for relative, digest in result["files"].items():
        member = (path.parent / relative).resolve()
        member.relative_to(path.parent.resolve())
        if sha256_file(member) != digest:
            raise ValueError(f"hybrid result member changed: {relative}")
    if result["mode"] == "evaluate":
        from util.evaluation.pulse_hybrid_development import metrics_from_predictions
        details = result["details"]
        arms = tuple(details.get("model_arms", ("original", "clean", "single", "three")))
        if "model_arms" in details and (arms != ("original", "single", "three")
                                        or details.get("image_suite") not in ("joint_c15_reference_v1", "image_c5_gpu_v1")):
            raise ValueError("invalid reference result model arms")
        phase = details.get("phase")
        suite = details.get("image_suite")
        c15 = suite in ("joint_c15", "joint_c15_reference_v1")
        c5 = suite == "image_c5_gpu_v1"
        if type(details.get("records")) is not int:
            raise ValueError("result record count must be an integer")
        if c5 and (type(details.get("inference_batch_size", 1)) is not int
                or not 1 <= details.get("inference_batch_size", 1) <= 4
                or type(details.get("performance_smoke", False)) is not bool
                or (phase == "block" and details.get("inference_batch_size", 1) != 1)
                or (details.get("performance_smoke", False) and phase not in ("smoke", "screen"))):
            raise ValueError("invalid result batch/performance policy")
        joint = suite in ("joint_v2", "joint_v3") or c15
        condition_count = (6 if c5 else 96 if c15
                           else 24 if suite == "joint_v3"
                           else ((20 if phase == "smoke" else 84) if joint else (25 if phase == "final" else 9)))
        full_cohort = details.get("full_cohort") is True
        from util.evaluation.pulse_hybrid_development import FULL_CENTER_RECORDS
        random_subset = (c5 and details.get("sampling_method") == "seeded_hash_v1"
            and phase in ("screen", "final") and not full_cohort and type(details.get("records")) is int
            and 16 <= details["records"] <= FULL_CENTER_RECORDS.get(details.get("center"), 0))
        if "sampling_method" in details and not random_subset:
            raise ValueError("invalid random subset result scope")
        is_block = phase == "block"
        if is_block:
            start, stop = validate_record_span(details.get("center"), details.get("record_span"))
            parent_digest = details.get("parent_cohort_identity")
            if (not c5 or details.get("full_cohort") is not False
                    or details.get("records") != stop - start
                    or not isinstance(parent_digest, str) or len(parent_digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in parent_digest)):
                raise ValueError("invalid partial full-center evaluation scope")
        elif "record_span" in details or "parent_cohort_identity" in details:
            raise ValueError("non-block result carries a partial record scope")
        valid_records = (is_block or (phase == "smoke" and details.get("records") == 4)
                         or random_subset
                         or (phase == "screen" and type(details.get("records")) is int and details["records"] in ((4, 16, 128) if c5 else (128,)))
                         or (phase == "final" and ((full_cohort and details.get("records", 0) > 512)
                                                   or (not full_cohort and details.get("records") == 512))))
        if (phase not in ("smoke", "screen", "final", "block") or details.get("center") not in CENTERS
                or not valid_records
                or details.get("conditions") != condition_count):
            raise ValueError("hybrid result has an incomplete phase budget")
        cohort = json.loads((path.parent / "cohort.json").read_text())
        if c5:
            from util.evaluation.pulse_hybrid_development import c5_gpu_conditions_for, FULL_CENTER_RECORDS
            from util.evaluation.ecg_image_queue import digest_json
            if not {"cohort.json", "metrics.json", "generation_admission.json",
                    "original_parent_parity.json", "runtime.json"} <= set(result["files"]):
                raise ValueError("GPU C5 evidence files lack integrity coverage")
            if (details.get("cohort_identity") != digest_json(cohort["samples"])
                    or (full_cohort and details["records"] != FULL_CENTER_RECORDS[details["center"]])):
                raise ValueError("GPU C5 cohort identity or full population changed")
            if (type(details.get("image_severity")) is not int or details["image_severity"] != 5
                    or arms != ("original", "single", "three")
                    or details.get("image_implementation") != "image_c5_torch_gpu_v1"
                    or cohort["conditions"] != c5_gpu_conditions_for([cohort["conditions"][0]], severity=5)):
                raise ValueError("GPU C5 image suite identity changed")
            paths = details.get("training_result_paths", {})
            training_hashes = details.get("training_results", {})
            if set(training_hashes) != {"single", "three"} or set(paths) != set(training_hashes):
                raise ValueError("GPU C5 training result paths are incomplete")
            from util.pulse_training_contract import validate_result as validate_training
            from util.run_record import archived_reference
            training_records = None
            scientific_protocol = None
            for arm, raw in paths.items():
                owned_path(raw)
                member = archived_reference(raw, path)
                if sha256_file(member) != training_hashes[arm]:
                    raise ValueError("GPU C5 referenced training result changed")
                training = json.loads(member.read_text())
                validate_training(training, member)
                records = json.loads((member.parent / "k500_records.json").read_text())
                if training_records is not None and records != training_records:
                    raise ValueError("GPU C5 arms have different K500 records")
                training_records = records
                if {r["hash_id"] for r in records} & {r["hash_id"] for r in cohort["samples"]}:
                    raise ValueError("GPU C5 cohort overlaps adaptation K500")
                protocol = training["protocol"]
                scientific = {key: value for key, value in protocol.items()
                              if key not in {"width", "jsd", "jsd_weight", "images_per_record_exposure"}}
                if scientific_protocol is not None and scientific != scientific_protocol:
                    raise ValueError("GPU C5 training arms differ beyond their registered intervention")
                scientific_protocol = scientific
                if training["mode"] != ("smoke" if phase == "smoke" else "train"):
                    raise ValueError("GPU C5 training phase differs from evaluation")
                from core.pulse_finetune import MODEL_HASHES, MODEL_CONFIG_HASHES
                if protocol.get("model_asset_sha256") != {**MODEL_HASHES, **MODEL_CONFIG_HASHES}:
                    raise ValueError("GPU C5 training result lacks the closed model asset manifest; retrain it")
                if (protocol.get("center") != details.get("center")
                        or protocol.get("width") != {"single": 1, "three": 3}[arm]
                        or protocol.get("augmentation_topology") != "image_only_gpu_branches_v1"
                        or protocol.get("mix_residual") != "clean_render"
                        or protocol.get("image_gpu", {}).get("implementation") != "augmix_torch_gpu_v2"
                        or protocol.get("image_augmentation", {}).get("waveform_strength") != 0):
                    raise ValueError("GPU C5 referenced training lineage differs")
            if random_subset:
                from util.evaluation.pulse_hybrid_development import checked_full_parent, select_c5_samples
                _, population, _ = checked_full_parent({"source_state": details["sampling_source_state"],
                    "center": details["center"]}, training_records)
                expected_samples = select_c5_samples(population, details["records"], sampling_seed=details["evaluation_seed"])
                if (details.get("sampling_population_identity") != digest_json(population)
                        or cohort["samples"] != expected_samples):
                    raise ValueError("frozen random subset identity/order changed")
            required_sources = {
                "source_snapshot/core/pulse_finetune.py",
                "source_snapshot/util/evaluation/pulse_hybrid_development.py",
                "source_snapshot/util/evaluation/pulse_adapters.py",
                "source_snapshot/util/evaluation/ecg_image_data.py",
                "source_snapshot/util/evaluation/ecg_image_queue.py",
                "source_snapshot/util/evaluation/pn2021.py",
                "source_snapshot/util/evaluation/metrics.py",
                "source_snapshot/util/ecg_image_renderer.py",
                "source_snapshot/util/augmentations/torch_operators.py",
                "source_snapshot/util/config_bundle.py",
                "source_snapshot/util/pn2021_artifact_contract.py",
                "source_snapshot/util/random_seed.py",
                "source_snapshot/util/run_record.py",
                "source_snapshot/data_preprocess/load_cache.py",
                "source_snapshot/data_preprocess/PN2021_preprocess.py",
                "source_snapshot/data_preprocess/pn2021_metadata.py",
                "source_snapshot/data_preprocess/data_runtime.py",
                "source_snapshot/data_preprocess/preprocess_primitives.py",
                "source_snapshot/core/image_augmix_gpu.py",
                "source_snapshot/core/image_augmix_c.py",
                "source_snapshot/core/image_corruption.py",
                "source_snapshot/core/image_stress.py",
                "source_snapshot/util/augmentations/profile.py",
                "source_snapshot/util/evaluation/ecg_image_artifact.py",
                "source_snapshot/util/evaluation/ecg_image_elastic.py",
                "source_snapshot/util/evaluation/pulse_visual_subset.py",
                "source_snapshot/util/pulse_hybrid_contract.py",
                "source_snapshot/util/pulse_training_contract.py",
                "source_snapshot/models/checkpoints.py",
                "source_snapshot/models/contracts.py",
                "source_snapshot/models/input_adapter.py",
            }
            if not required_sources <= set(result["files"]):
                raise ValueError("GPU C5 source snapshot is incomplete")
        if joint:
            from util.evaluation.pulse_hybrid_development import joint_conditions_for
            if c15:
                from util.evaluation.pulse_hybrid_development import c15_joint_conditions_for
                from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
                conditions = cohort["conditions"]
                severity = details.get("image_severity") if suite == "joint_c15_reference_v1" else conditions[6]["image_severity"]
                if conditions != c15_joint_conditions_for([conditions[0]], CANONICAL_OPERATOR_ORDER,
                        severity=severity, reference=suite == "joint_c15_reference_v1"):
                    raise ValueError("C15 image suite is not the frozen Cartesian product")
                if suite == "joint_c15_reference_v1":
                    from core.image_augmix_c import C_IMPLEMENTATION
                    reference = cohort.get("image_reference", {})
                    if (reference.get("implementation") != C_IMPLEMENTATION
                            or reference != details.get("image_reference")
                            or not reference.get("versions") or not reference.get("files_sha256")):
                        raise ValueError("missing C15 reference environment identity")
            elif suite == "joint_v3":
                from util.evaluation.pulse_hybrid_development import single_operator_joint_conditions_for
                from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
                conditions = cohort["conditions"]
                if conditions != single_operator_joint_conditions_for([conditions[0]], CANONICAL_OPERATOR_ORDER):
                    raise ValueError("single-operator image suite is not the frozen Cartesian product")
            else:
                conditions = cohort["conditions"]
        if joint and suite != "joint_v3" and not c15:
            waves = [{k: v for k,v in c.items() if k != "family"} for c in conditions if c["family"] == "waveform"]
            # Rebuild the complete Cartesian product; no missing/easy-only views.
            parent = [{k: v for k,v in conditions[0].items() if k != "family"}, *waves]
            while len(parent) < 21:
                parent.append({"condition_id": f"unused_{len(parent)}", "operators": []})
            if conditions != joint_conditions_for(parent, list(range(1, len(waves) + 1))):
                raise ValueError("joint image suite is not the frozen full cross product")
        rows = [r for name in sorted(result["files"]) if name.startswith("batches/")
                for r in json.loads((path.parent / name).read_text())]
        if "merged_blocks" in details:
            if (not c5 or phase != "final" or not full_cohort
                    or "block_results.json" not in result["files"]):
                raise ValueError("merged blocks require a complete final C5 artifact")
            from util.run_record import archived_reference
            from util.evaluation.pulse_hybrid_development import checked_center_blocks
            entries = json.loads((path.parent / "block_results.json").read_text())
            entries = [{"path": str(archived_reference(entry["path"], path)), "sha256": entry["sha256"]}
                       for entry in entries]
            expected = {key: details[key] for key in ("center", "evaluation_seed", "image_suite",
                "image_severity", "model_arms", "execution_mode", "original_baseline_mode", "training_results")}
            blocks = checked_center_blocks(entries, cohort["samples"], expected)
            if (type(details["merged_blocks"]) is not int or details["merged_blocks"] != len(blocks)
                    or rows != [row for block in blocks for row in block["rows"]]):
                raise ValueError("merged predictions differ from immutable evaluation blocks")
        elif "block_results.json" in result["files"]:
            raise ValueError("block provenance lacks a declared full-center merge")
        if c5:
            from util.evaluation.pulse_hybrid_development import processor_input_identity, PROCESSOR_INPUT_HASH_MODE
            samples_by_key = {s["sample_key"]: s for s in cohort["samples"]}
            conditions_by_key = {c["condition_id"]: c for c in cohort["conditions"]}
            renderer_identity = digest_json(cohort["renderer"])
            for row in rows:
                expected_identity = processor_input_identity(seed=details["evaluation_seed"],
                    sample=samples_by_key[row["sample_key"]], condition=conditions_by_key[row["condition_id"]],
                    image_suite=suite, renderer_identity=renderer_identity)
                if (row.get("processor_input_hash_mode") != PROCESSOR_INPUT_HASH_MODE
                        or row.get("processor_input_identity_sha256") != expected_identity):
                    raise ValueError("GPU C5 processor input recipe identity changed")
        metrics = metrics_from_predictions(rows, cohort["samples"], cohort["conditions"], arms=arms)
        if metrics != json.loads((path.parent / "metrics.json").read_text()) or metrics["families"] != result["details"]["families"]:
            raise ValueError("hybrid metrics do not reproduce from predictions")
        if (cohort["k500_overlap"] != 0 or len(cohort["samples"]) != result["details"]["records"]
                or len(cohort["conditions"]) != result["details"]["conditions"]):
            raise ValueError("hybrid result cohort mismatch")
        admissions = json.loads((path.parent / "generation_admission.json").read_text())
        admission_batch = details.get("inference_batch_size", 1) if suite == "image_c5_gpu_v1" else 2
        batch_shapes = {min(admission_batch, len(cohort["samples"]))}
        if c5 and len(cohort["samples"]) % admission_batch:
            batch_shapes.add(len(cohort["samples"]) % admission_batch)
        if set(admissions) != {f"{a}_batch{b}" for a in arms for b in batch_shapes} or any(
                v != {"packing_exact": True, "tokens_exact": True} for v in admissions.values()):
            raise ValueError("all evaluated arms require author-generation admission")
        if "original_parent_parity" in details:
            parity_path = path.parent / "original_parent_parity.json"
            parity = json.loads(parity_path.read_text())
            if parity != details["original_parent_parity"]:
                raise ValueError("original parent parity audit changed")
            if not c5 and (parity.get("canonical_drift", 0) or parity.get("canonical_unknown", 0)):
                raise ValueError("unexplained original-parent answer drift")
        if c5:
            baseline_path = path.parent / "original_baseline.json"
            execution = details.get("execution_mode", "reference_v1")
            if execution not in ("reference_v1", *C5_OPTIMIZED_MODES):
                raise ValueError("unknown GPU C5 execution mode")
            if execution in C5_OPTIMIZED_MODES:
                if "performance_admission.json" not in result["files"]:
                    raise ValueError("optimized C5 lacks native token parity admission")
                admission = json.loads((path.parent / "performance_admission.json").read_text())
                if execution in C5_FP16_MODES and (admission.get("inference_trainable_dtype") != "float16"
                        or admission.get("reference_parameter_storage_restored") is not True):
                    raise ValueError("FP16 inference requires restored reference parameter storage")
                if execution == C5_ORIGINAL_BYPASS and admission.get("original_zero_lora_bypassed") is not True:
                    raise ValueError("original LoRA bypass lacks zero-update admission")
                checks = admission.get("records", [])
                from util.evaluation.pulse_hybrid_development import c5_admission_indices
                check_indices = c5_admission_indices(len(cohort["samples"]), admission_batch,
                    every_batch=phase == "smoke" or details.get("performance_smoke", False))
                if (admission.get("execution_mode") != execution or admission.get("batch_size") != admission_batch
                        or admission.get("trainable_vision_outputs_cached") != 0
                        or len(checks) != len(check_indices)
                        or [r["sample_key"] for r in checks] != [cohort["samples"][i]["sample_key"] for i in check_indices]
                        or any(r.get("tokens_exact") is not True or r.get("views") != 6 or r.get("arms") != 3 for r in checks)):
                    raise ValueError("optimized C5 token admission is incomplete")
                if "inference_batch_size" in details:
                    if "performance_batches.json" not in result["files"]:
                        raise ValueError("explicit C5 batching requires batch-level timing records")
                    timings = json.loads((path.parent / "performance_batches.json").read_text())
                    starts = list(range(0, len(cohort["samples"]), admission_batch))
                    if not isinstance(timings, list) or len(timings) != len(starts):
                        raise ValueError("C5 batch timing coverage changed")
                    for timing, start in zip(timings, starts):
                        keys = [s["sample_key"] for s in cohort["samples"][start:start + admission_batch]]
                        seconds = timing.get("seconds", {})
                        checked = start in check_indices
                        expected_seconds = {"reference", "optimized"} if checked else {"optimized"}
                        if execution == C5_ORIGINAL_BYPASS and details.get("performance_smoke", False):
                            expected_seconds.add("baseline")
                        if (type(timing.get("offset")) is not int or timing["offset"] != start
                                or timing.get("sample_keys") != keys
                                or type(timing.get("actual_batch_size")) is not int
                                or timing["actual_batch_size"] != len(keys)
                                or timing.get("reference_checked") is not checked
                                or timing.get("warmup_excluded") is not True
                                or set(seconds) != expected_seconds
                                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in seconds.values())
                                or any(type(timing.get(k)) is not int or timing[k] < 0 for k in
                                       ("peak_allocated_bytes", "peak_reserved_bytes"))):
                            raise ValueError("C5 batch timing identity or values changed")
            if ("original_baseline.json" not in result["files"] or "original_parent_parity" not in details
                    or result["files"]["original_baseline.json"] != details.get("original_baseline_sha256")):
                raise ValueError("GPU C5 requires an immutable current original baseline")
            baseline = json.loads(baseline_path.read_text())
            validate_original_baseline(baseline, rows, details)
            parity = details["original_parent_parity"]
            if (parity.get("baseline_mode") != baseline["mode"] or parity.get("checked") != len(cohort["samples"])
                    or parity["checked"] != parity.get("exact", 0) + parity.get("drift", 0)
                    or parity.get("drift", 0) != sum(parity.get(k, 0) for k in ("canonical_equal", "canonical_drift", "canonical_unknown"))):
                raise ValueError("historical parent diagnostic audit is incomplete")
    else:
        fixed = result["mode"] == "fixed"
        if ((not fixed and len(result["details"].get("trials", [])) != 8)
                or set(result["details"].get("final_results", {})) != set(CENTERS)):
            raise ValueError("incomplete hybrid search/final grid")
        required = {"comparison.json", "per_condition.csv", "source_identity.json"}
        required |= {"training_results.json", "smoke_results.json", "final_results.json"} if fixed else {"trial_summaries.json"}
        if not required <= result["files"].keys():
            raise ValueError("missing final comparison evidence")
        if fixed:
            if result["details"].get("model_arms") != ["original", "single", "three"] or result["details"].get("selection") != "fixed_recipe_last_checkpoint_no_hpo":
                raise ValueError("fixed workflow selection/arms changed")
            training = json.loads((path.parent / "training_results.json").read_text())
            if set(training) != {f"{center}_{arm}" for center in CENTERS for arm in ("single", "three")}:
                raise ValueError("fixed workflow requires all eight training results")
            from util.pulse_training_contract import validate_result as validate_training
            from util.run_record import archived_reference
            for key, entry in training.items():
                member = archived_reference(entry["path"], path)
                if sha256_file(member) != entry["sha256"]:
                    raise ValueError("referenced training result changed")
                trained = json.loads(member.read_text())
                validate_training(trained, member)
                protocol = trained["protocol"]
                arm = {1: "single", 3: "three"}.get(protocol["width"])
                if (trained["mode"] != "train"
                        or key != f"{protocol['center']}_{arm}"):
                    raise ValueError("fixed workflow training center/width differs from its key")
        expected_suite = result["details"].get("image_suite", "joint_c15_reference_v1")
        expected_severity = result["details"].get("image_severity", 3)
        for center, entry in result["details"]["final_results"].items():
            from util.run_record import archived_reference
            child_path = archived_reference(entry["path"], path)
            if sha256_file(child_path) != entry["sha256"]:
                raise ValueError("referenced final result changed")
            child = json.loads(child_path.read_text())
            if child.get("mode") != "evaluate" or child["details"].get("center") != center or child["details"].get("phase") != "final":
                raise ValueError("final result reference has the wrong center/phase")
            if fixed and (child["details"].get("model_arms") != ["original", "single", "three"]
                    or child["details"].get("image_suite") != expected_suite
                    or child["details"].get("image_severity") != expected_severity
                    or child["details"].get("training_results") != {
                        arm: training[f"{center}_{arm}"]["sha256"] for arm in ("single", "three")}):
                raise ValueError("fixed workflow final evaluation has the wrong protocol")
            validate_result(child, child_path)


def finalize(output, *, mode, details):
    result = {"schema_version": 1, "artifact_type": "pulse_hybrid_result", "status": "complete",
        "mode": mode, "details": details,
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    validate_result(result, output / "hybrid_result.json")
    atomic_json(output / "hybrid_result.json", result)


def validate_record_span(center, span):
    """Only fixed 512-record blocks of a complete locked center are valid."""
    from util.evaluation.pulse_hybrid_development import FULL_CENTER_RECORDS
    if (center not in FULL_CENTER_RECORDS or not isinstance(span, list) or len(span) != 2
            or any(type(value) is not int for value in span)):
        raise ValueError('invalid full-center record block')
    start, stop = span
    total = FULL_CENTER_RECORDS[center]
    if not (0 <= start < stop <= total and start % 512 == 0 and stop == min(start + 512, total)):
        raise ValueError('record block is not a fixed 512-record partition')
    return start, stop
