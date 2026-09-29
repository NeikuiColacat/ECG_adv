"""Matched RGB-mixing regression and finite workflow tests; CPU only."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from core.augmix import native500_augmix_one
from core.pulse_finetune import FrozenVisionCache
from util.augmentations.profile import load_augmentation_profile
from util.evaluation.pulse_subset import matched_pixel_protocol, validate_config as validate_subset
from util.pulse_training_contract import CENTERS, validate_config
from util.pulse_training_queue import pixel_pipeline_plans

REPO = Path(__file__).resolve().parents[2]


class ProbeRenderer:
    def __init__(self):
        self.waves, self.images = [], []

    def render(self, wave):
        self.waves.append(wave.clone())
        # Nonlinear rendering surrogate: averaging before/after must differ.
        image = wave[:, :30].square().transpose(1, 2).unsqueeze(1)
        self.images.append(image.clone())
        return image


@pytest.mark.parametrize("width,sha", [
    (1, "582d2e35fcaba09798afc6b5c3c04a84a5f13512a794350b3919327426c48456"),
    (2, "7cb145b0e2536c8a28c94bd86048818f97f89f911ba620f902c0324f700c2b3a"),
])
def test_legacy_waveform_is_bitwise_unchanged(width, sha):
    profile = load_augmentation_profile(REPO / "configs/augmentation/operators.yaml")
    wave = torch.linspace(-1, 1, 60000).reshape(1, 5000, 12)
    actual, trace = native500_augmix_one(wave, width=width, profile=profile, seed=20260909, identity="pixel-regression")
    assert hashlib.sha256(actual.numpy().tobytes()).hexdigest() == sha
    assert "mixing_domain" not in trace and trace["clean_mix"] == "none"


@pytest.mark.parametrize("width", [1, 2])
def test_rgb_mixes_after_each_chain_render_and_preserves_rng(width):
    profile = load_augmentation_profile(REPO / "configs/augmentation/operators.yaml")
    wave = torch.linspace(-1, 1, 60000).reshape(1, 5000, 12)
    before, rng = wave.clone(), torch.get_rng_state().clone()
    renderer = ProbeRenderer()
    kwargs = dict(width=width, profile=profile, seed=20260909, identity="pixel-regression")
    image, trace = native500_augmix_one(wave, renderer=renderer, **kwargs)
    legacy, old_trace = native500_augmix_one(wave, **kwargs)
    assert torch.equal(before, wave) and torch.equal(rng, torch.get_rng_state())
    assert len(renderer.images) == width and trace.pop("mixing_domain") == "rendered_rgb"
    assert trace == old_trace
    weights = torch.tensor(trace["weights"])
    assert torch.equal(image, sum(w*i for w,i in zip(weights, renderer.images)))
    reference = ProbeRenderer().render(legacy)
    assert torch.equal(image, reference) if width == 1 else not torch.allclose(image, reference)


def test_rgb_cannot_enter_clean_feature_cache():
    cache = FrozenVisionCache(None, None, 500)
    with pytest.raises(ValueError, match="clean feature cache"):
        cache.features(torch.zeros(1), input_is_image=True, key=0)


def test_pixel_configs_are_matched_to_old_budget_and_only_change_domain():
    for center in CENTERS:
        for width in (1, 2):
            name = f"w{width}_{center}"
            old = yaml.safe_load((REPO / f"configs/train/pulse_augmix_{name}.yaml").read_text())
            new = yaml.safe_load((REPO / f"configs/train/pulse_pixel_{name}.yaml").read_text())
            validate_config(new)
            assert new.pop("mixing_domain") == "rendered_rgb" and new.pop("schema_version") == 2
            old.pop("schema_version")
            new["paths"]["temporary_root"] = old["paths"]["temporary_root"]
            assert new == old


def test_pixel_protocol_rejects_changed_hyperparameters_and_evaluation_budget():
    old = {"width": 2, "training": {"optimizer_steps":100}, "implementation_sha256": {"x":"a"}}
    new = {**copy.deepcopy(old), "mixing_domain":"rendered_rgb", "implementation_sha256":{"x":"b"}}
    matched_pixel_protocol(old, new)
    new["training"]["optimizer_steps"] = 101
    with pytest.raises(ValueError, match="more than"):
        matched_pixel_protocol(old, new)
    payload = yaml.safe_load((REPO / "configs/eval/pulse_pixel_subset.yaml").read_text())
    validate_subset(payload)
    payload["runtime"]["max_workers"] = 5
    with pytest.raises(ValueError):
        validate_subset(payload)


def test_fixed_workflow_has_three_admissions_eight_trains_and_one_evaluation(tmp_path):
    from uuid import uuid4
    from util.config_bundle import resolve_yaml_config_closure

    path = REPO / "configs/train/pulse_pixel_pipeline.yaml"
    wrong = REPO / "configs/experiments/pulse_augmix_single_smoke.yaml"
    old_root = "/home/linbinhao/ECG_adv_data/runs/pulse_pixel_mix_sft_20260910"
    run_root = Path("/home/linbinhao/ECG_adv_data/runs") / f"unit_pixel_{uuid4().hex}" / Path(old_root).name
    isolated = tmp_path / "configs"
    # Keep all admission, resume, training and evaluation links consistent while
    # removing this unit test's dependency on archived live experiment paths.
    closure = resolve_yaml_config_closure([path, wrong], config_root=REPO / "configs")
    for source in closure:
        destination = isolated / source.relative_to(REPO / "configs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text().replace(old_root, str(run_root)))
    path = isolated / path.relative_to(REPO / "configs")
    payload = yaml.safe_load(path.read_text())
    plans, jobs = pixel_pipeline_plans(payload, path, isolated)
    assert tuple(plans) == ("single", "two", "resume", "training_queue", "evaluation")
    assert len(jobs) == 8
    assert all(j["run"].is_relative_to(run_root) for j in jobs.values())
    assert not run_root.exists()
    child_path = plans["resume"][1].entry_config_path
    child = yaml.safe_load(child_path.read_text())
    assert Path(child["resume_from"]).is_relative_to(run_root)
    child["resume_from"] = "/data/archived_pixel_fixture/resume_probe.pt"
    with pytest.raises(ValueError, match="resume checkpoint"):
        validate_config(child)
    payload["references"]["single"] = "experiments/pulse_augmix_single_smoke.yaml"
    with pytest.raises(ValueError, match="domain/arm"):
        pixel_pipeline_plans(payload, path, isolated)


def test_reused_parent_rejects_inference_source_or_recipe_changes():
    from util.evaluation.pulse_subset import validate_pixel_parent
    from util.pn2021_artifact_contract import sha256_file
    name = "util/ecg_image_renderer.py"
    paired = {"runtime":{"batch_size":2}, "paths":{"raw_root":"raw", "temporary_root":"old"}}
    original = {"sources":{name:sha256_file(REPO / name)},
                "audit":{"common_implementation_sha256":{}}, "paired_config":paired}
    new = copy.deepcopy(paired); new["paths"]["temporary_root"] = "new"
    validate_pixel_parent(original, new)
    new["runtime"]["batch_size"] = 4
    with pytest.raises(ValueError, match="frozen field"):
        validate_pixel_parent(original, new)
    original["sources"][name] = "bad"
    with pytest.raises(ValueError, match="source changed"):
        validate_pixel_parent(original, paired)


def test_five_arm_report_recomputes_frozen_parent_and_new_pair(tmp_path):
    from util.evaluation.ecg_image_artifact import parse_response
    from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, fixed_tasks
    from util.evaluation.pulse_subset import finalize, validate_result, MIXING_NAMES
    from util.pn2021_artifact_contract import sha256_file
    samples = [{"sample_key":c, "record_id":c, "hash_id":c, "logical_center":c, "label_names":["CD"]} for c in CENTERS]
    conditions = [{"condition_id":"clean" if j==0 else f"c{j}", "condition_index":j, "operators":[], "depth":0} for j in range(21)]
    old_result = None
    for name in ("old", "new"):
        root = tmp_path/name
        tasks = fixed_tasks(samples, records_per_task=2, batch_size=2, condition_ids=[c["condition_id"] for c in conditions], protocol_identity=name)
        prepared = {"rows":samples, "conditions":conditions, "subset_config":{"records_per_center":1},
                    "subset_identity":name, "tasks":tasks, "base_tasks":tasks}
        if old_result:
            prepared["pixel_parent"] = {"result":str(old_result), "sha256":sha256_file(old_result)}
        pair, base = TaskQueue(root/"pair_queue", tasks), TaskQueue(root/"base_queue", tasks)
        for queue, kind in ((pair,"pair"), (base,"base")):
            while (lease := queue.claim("test")) is not None:
                with lease:
                    keys = {k for b in lease.task["batches"] for k in b}
                    rows = []
                    for sample in samples:
                        if sample["sample_key"] not in keys:
                            continue
                        for condition in conditions:
                            def answer(label):
                                return {"response":label, "generated_tokens":2, "hit_max_new_tokens":False, **parse_response(label)}
                            row = {**sample, **condition, "schema_version":1, "true_labels":["CD"],
                                   "clean_waveform_sha256":"a"*64, "input_waveform_sha256":"b"*64}
                            row.update({"answer":answer("CD")} if kind=="base" else
                                {"arms":{"w1":answer("CD"), "w2":answer("MI" if name=="new" else "CD")}})
                            rows.append(row)
                    lease.commit(rows, {"seconds":1})
        for key, value in (("prepared",prepared), ("cohort",samples), ("data_audit",{}), ("reuse",{"pair_tasks":[]}), ("reference_generation",{})):
            atomic_json(root/f"{key}.json", value)
        (root/"workers").mkdir()
        finalize(root, prepared, pair, base)
        result_path = root/"subset_result.json"
        validate_result(json.loads(result_path.read_text()), result_path)
        if name=="old":
            old_result = result_path
    comparison = json.loads((tmp_path/"new/mixing_comparison.json").read_text())
    assert set(comparison["models"]) == set(MIXING_NAMES)
    assert comparison["models"]["image-two"]["center_equal"]["clean"]["macro_f1"] == 0
    assert comparison["models"]["waveform-two"]["center_equal"]["clean"]["macro_f1"] > 0
