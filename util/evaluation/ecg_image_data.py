"""Frozen native500 input contract for the external image-LLM development probe.

This does not change the canonical raw100 classifier/adaptation interface.
The renderer and native500 seed/window semantics are promoted from the frozen
PULSE probe; no exploratory module is imported at runtime.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data_preprocess import preprocess_primitives as primitives
from data_preprocess.PN2021_preprocess import _resample_crop_pad
from data_preprocess.data_runtime import PN2021EvaluationLoaderPlan
from data_preprocess.load_cache import EXPECTED_LEADS
from util.augmentations.profile import load_augmentation_profile
from util.augmentations.torch_operators import apply_operator_batch_prevalidated
from util.config_bundle import resolve_config_reference
from util.evaluation.pn2021 import load_pn2021_eval_config
from util.pn2021_artifact_contract import sha256_file

CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
PROTOCOL_ID = "pulse_pn2021c_native500_gpu_renderer_s5_depth23_v1"
SEED_NAMESPACE = "pulse_pn2021c_native500_gpu_torch"
SELECTION_NAMESPACE = "ecg-image-llm-10h-20260907"
SMOKE_PER_CENTER = 8
QUESTION = (
    "You are provided with an ECG signal image. Please classify the ECG image "
    "into the following diagnostic statement classes:\n"
    "NORM: Normal ECG\n"
    "MI: Myocardial Infarction\n"
    "STTC: ST/T Change\n"
    "CD: Conduction Disturbance\n"
    "HYP: Hypertrophy. Directly output the class name and separate with a semicolon."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_cohort(rows: list[dict[str, Any]], *, per_center: int, smoke: bool) -> list[dict[str, Any]]:
    if per_center < 1 or (smoke and per_center > SMOKE_PER_CENTER):
        raise ValueError("invalid per-center cohort count")
    keys = [row["sample_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate source cohort identity")
    groups = []
    for center in CENTERS:
        ordered = sorted(
            (r for r in rows if r["logical_center"] == center),
            key=lambda r: hashlib.sha256(
                f"{SELECTION_NAMESPACE}|{r['sample_key']}".encode()
            ).hexdigest(),
        )
        start = 0 if smoke else SMOKE_PER_CENTER
        chosen = ordered[start:start + per_center]
        if len(chosen) != per_center:
            raise ValueError(f"insufficient eligible records for {center}")
        groups.append(chosen)
    # Round-robin centers: a time-limited run never silently becomes one-center.
    return [row for group in zip(*groups, strict=True) for row in group]


def select_full_cohort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every frozen eligible record, interleaving unequal centers safely."""
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("duplicate or empty full cohort")
    if set(row["logical_center"] for row in rows) != set(CENTERS):
        raise ValueError("full cohort requires exactly the four logical centers")
    groups = [sorted((r for r in rows if r["logical_center"] == center),
                     key=lambda r: hashlib.sha256(
                         f"{SELECTION_NAMESPACE}|{r['sample_key']}".encode()).hexdigest())
              for center in CENTERS]
    return [row for group in itertools.zip_longest(*groups) for row in group if row is not None]


def native500_waveform(sample: dict[str, Any], raw_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    record_path = (raw_root / str(sample["source_record"])).resolve()
    record_path.relative_to(raw_root.resolve())
    signal_tc, source_fs, source_leads = primitives._read_wfdb(record_path)
    if abs(source_fs - 500.0) > 1e-6:
        raise ValueError(f"native500 source is not 500 Hz: {record_path} fs={source_fs}")
    signal_tc = primitives._reorder_leads(signal_tc, source_leads, list(EXPECTED_LEADS))
    signal_tc, quality = primitives._repair_nonfinite_per_lead(
        signal_tc, max_record_fraction=0.01, max_lead_fraction=0.05
    )
    waveform, details = _resample_crop_pad(
        signal_tc, source_fs=source_fs, target_fs=500, duration_seconds=10,
        target_num_samples=5000, window_policy="center",
    )
    original_num_samples = int(sample["original_num_samples"])
    used_source_samples = min(original_num_samples, 5000)
    source_start_sample = max((original_num_samples - used_source_samples) // 2, 0)
    expected = {
        "original_num_samples": original_num_samples,
        "source_start_sample": source_start_sample,
        "source_end_sample": source_start_sample + used_source_samples,
        "was_padded": used_source_samples < 5000,
        "was_truncated": original_num_samples > 5000,
    }
    for name, value in expected.items():
        if details[name] != value:
            raise RuntimeError(f"raw preprocessing drift for {sample['sample_key']}: {name}")
    if quality["repaired_nonfinite_count"] != int(sample["repaired_nonfinite_count"]):
        raise RuntimeError(f"nonfinite repair drift for {sample['sample_key']}")
    return waveform, {**details, **quality, "source_fs": source_fs}


def derived_seed(base_seed: int, namespace: str, *identity: Any) -> int:
    payload = "|".join(str(v) for v in (base_seed, namespace, *identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@torch.inference_mode()
def corrupt_native500_one(
    waveform_btc: torch.Tensor, *, source_hash: str,
    condition: Mapping[str, Any], profile: Any, base_seed: int,
) -> torch.Tensor:
    if tuple(waveform_btc.shape) != (1, 5000, 12):
        raise ValueError("native-500 corruption expects one (1,5000,12) waveform")
    output = waveform_btc.to(dtype=torch.float32).contiguous()
    identity = (PROTOCOL_ID, profile.profile_name, profile.severity, 500,
                str(source_hash), str(condition["condition_id"]))
    for operator_index, operator in enumerate(condition["operators"]):
        seed = derived_seed(base_seed, SEED_NAMESPACE, *identity, operator_index, operator)
        generator = torch.Generator(device=output.device).manual_seed(seed)
        output = apply_operator_batch_prevalidated(
            str(operator), output, params=profile.parameters_for(str(operator)),
            sampling_rate_hz=500, rng=generator,
        )
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError(f"non-finite corruption: {source_hash}/{condition['condition_id']}")
    return output


def audit_inputs(config: dict[str, Any], config_path: Path, config_root: Path):
    """Validate prior PULSE identity and independently recheck selected split/labels."""
    baseline = Path(config["baseline"]["run_dir"]).resolve()
    protocol_path = baseline / "protocol.json"
    if sha256_file(protocol_path) != config["baseline"]["protocol_sha256"]:
        raise ValueError("PULSE protocol SHA256 mismatch")
    protocol = json.loads(protocol_path.read_text())
    cohort_path = baseline / "cohort.jsonl"
    if sha256_file(cohort_path) != protocol["cohort"]["sha256"]:
        raise ValueError("PULSE cohort SHA256 mismatch")
    repo = Path(__file__).resolve().parents[2]
    dependencies = {}
    for name, expected in protocol["implementation_files"].items():
        if name.startswith("agent_workspace/"):
            continue
        actual = sha256_file(repo / name)
        if actual != expected:
            raise ValueError(f"PULSE input implementation drift: {name}")
        dependencies[name] = actual
    renderer_sha = sha256_file(repo / "util/ecg_image_renderer.py")
    if renderer_sha != protocol["rendering"]["renderer_script_sha256"]:
        raise ValueError("renderer is not the byte-identical promoted PULSE renderer")
    dependencies["util/ecg_image_renderer.py"] = renderer_sha
    ref = config["references"]
    resolve = lambda key: resolve_config_reference(
        ref[key], owner_config_path=config_path, config_root=config_root,
        description=key, must_exist=True,
    )
    profile = load_augmentation_profile(resolve("operators_config"), config_root=config_root)
    if profile.describe()["config_sha256"] != protocol["corruption"]["profile"]["config_sha256"]:
        raise ValueError("PULSE operator profile drift")
    conditions = [{"condition_index": 0, "condition_id": "clean", "depth": 0, "operators": []}]
    for depth in (2, 3):
        for operators in itertools.combinations(profile.canonical_order, depth):
            conditions.append({"condition_index": len(conditions),
                               "condition_id": f"d{depth}__{'__'.join(operators)}",
                               "depth": depth, "operators": list(operators)})
    if conditions != protocol["corruption"]["conditions"]:
        raise ValueError("PULSE condition identity drift")
    full = config["cohort"].get("mode") == "full_pulse"
    rows = (select_full_cohort(read_jsonl(cohort_path)) if full else
            select_cohort(read_jsonl(cohort_path), per_center=config["cohort"]["per_center"],
                          smoke=config["stage"] == "smoke"))
    evaluation = load_pn2021_eval_config(resolve("evaluation_config"), config_root=config_root)
    views = evaluation.payload["protocol"]["canonical_views"]
    ep = evaluation.payload["protocol"]
    if ep["mapping_hash"] != "555ec85d5b51" or tuple(ep["class_order"]) != CLASS_ORDER:
        raise ValueError("Super5 mapping identity drift")
    plan = PN2021EvaluationLoaderPlan(
        clean_partition=views["clean_drop"], corrupted_partition=views["corrupted_drop"],
        batch_size=1, num_workers=0, pin_memory=False, persistent_workers=False,
        prefetch_factor=2, cache_mode="mmap", validate_values="sample",
        split_config_path=evaluation.split_config_path,
        data_load_config_path=evaluation.data_load_config_path,
        corruption_cache_config_path=evaluation.corruption_cache_config_path,
        seed_config_path=evaluation.random_seed_config_path,
    )
    split_audit = {}
    with plan.open_session() as session:
        for center in CENTERS:
            loader = session.open_clean(center)
            try:
                dataset = loader.dataset
                selection = dataset.selection
                cache = dataset._get_cache()
                positions = {str(h): int(i) for h, i in zip(selection.hash_ids, selection.indices, strict=True)}
                chosen = [row for row in rows if row["logical_center"] == center]
                if full and {row["hash_id"] for row in chosen} != set(positions):
                    raise ValueError("full cohort is not the exact ref-excluded dropzero split")
                for row in chosen:
                    if row["hash_id"] not in positions:
                        raise ValueError("selected record is not in the ref-excluded dropzero split")
                    index = positions[row["hash_id"]]
                    metadata = cache.records.iloc[index]
                    if str(metadata["record_id"]) != row["record_id"] or str(metadata["source_record"]) != row["source_record"]:
                        raise ValueError("selected record/source identity mismatch")
                    label = np.asarray(cache.labels[index], dtype=np.uint8).tolist()
                    if label != row["label"] or not any(label):
                        raise ValueError("selected label mismatch or all-zero label")
                split_audit[center] = {"selected": len(chosen), "ref_excluded": True,
                                       "selection_hash_id_set_sha256": selection.hash_id_set_sha256}
            finally:
                loader.close()
    audit = {"status": "passed", "mapping_hash": "555ec85d5b51",
             "mapping_version": ep["mapping_version"], "class_order": list(CLASS_ORDER),
             "metric_view": "drop_all_zero", "centers": split_audit,
             "baseline_protocol_sha256": sha256_file(protocol_path),
             "baseline_cohort_sha256": sha256_file(cohort_path),
             "common_implementation_sha256": dependencies,
             "selection_namespace": SELECTION_NAMESPACE,
             "smoke_per_center_reserved": SMOKE_PER_CENTER,
             "sampling_uses_labels": False, "input": "native500_raw_mV_center_crop_right_pad",
             "normalization_before_render": "none"}
    if full:
        if len(rows) != protocol["cohort"]["record_count"]:
            raise ValueError("full PULSE cohort count mismatch")
        audit.update(selection_namespace="full_pulse_all_eligible",
                     smoke_per_center_reserved=0, full_cohort=True)
    return rows, conditions, profile, protocol, audit
