"""CPU-only scheduling/adoption contracts; never launches a training child."""
from pathlib import Path
import copy
import json
import os

import pytest
import yaml

from boot_scripts.run_experiment import load_experiment_plan
from util.pulse_training_queue import JOB_KEYS, allowed_gpus, job_state, live_training_jobs, load_jobs, validate_config

REPO = Path(__file__).resolve().parents[2]


def test_sampling_recovery_keeps_the_grid_and_requires_fresh_engineering(tmp_path):
    from util import simclr_joint_queue as q
    path = REPO/"configs/train/k500_sampling/workflow_recovery1.yaml"
    config = yaml.safe_load(path.read_text())
    members = q.sampling_plans(config, path, REPO/"configs")
    assert len(members) == 96
    assert config["recovery"] == q.SAMPLING_RECOVERY
    assert all("_recovery1/training/train_result.json" in p for p in config["admission_results"].values())
    config["recovery"] = {**config["recovery"], "previous_workflow": str(tmp_path)}
    with pytest.raises(ValueError, match="restricted"):
        q.sampling_plans(config, path, REPO/"configs")


def test_nvme_sampling_recovery_keeps_all_96_scientific_plans_and_exact_storage(tmp_path):
    from util import simclr_joint_queue as q
    path = REPO/"configs/train/k500_sampling/workflow_recovery2.yaml"
    config = yaml.safe_load(path.read_text())
    members = q.sampling_plans(config,path,REPO/"configs")
    assert len(members) == 96 and config["recovery"] == q.SAMPLING_NVME_RECOVERY
    assert all("_recovery2/training/train_result.json" in p for p in config["admission_results"].values())
    for key,(_,plan) in members.items():
        assert plan.run_dir == (q.SAMPLING_ROOT/key).resolve()
    config["recovery"] = {**config["recovery"], "storage_root":str(tmp_path)}
    with pytest.raises(ValueError,match="restricted"):
        q.sampling_plans(config,path,REPO/"configs")


def test_nvme_sampling_recovery_rejects_a_partial_draw0(tmp_path,monkeypatch):
    from util import simclr_joint_queue as q,run_record
    previous = tmp_path/"previous";(previous/"training").mkdir(parents=True)
    (previous/"run_manifest.json").write_text(json.dumps({"status":"failed"}))
    (previous/"training/status.json").write_text(json.dumps({"status":"draining","active":{},"completed":{}}))
    (previous/"training/source_identity.json").write_text("{}")
    policy = {**q.SAMPLING_NVME_RECOVERY,"previous_workflow":str(previous)}
    monkeypatch.setattr(q,"SAMPLING_NVME_RECOVERY",policy)
    monkeypatch.setattr(run_record,"verify_run_file_index",lambda path:[])
    with pytest.raises(ValueError,match="32 completed"):
        q.recover_sampling_completed({"recovery":policy},{},tmp_path,{})


@pytest.mark.parametrize("damage", [None, "source", "result", "config", "active", "training_link"])
def test_sampling_recovery_checks_exact_completed_artifacts(tmp_path, monkeypatch, damage):
    from types import SimpleNamespace
    from util import simclr_joint_queue as q, run_record, pn2021_artifact_contract as contract
    previous = tmp_path/"previous"
    (previous/"training/source_snapshot/core").mkdir(parents=True)
    source = previous/"training/source_snapshot/core/online_trainer.py"
    source.write_text("unchanged training")
    sources = {"core/online_trainer.py": q.sha256_file(source)}
    (previous/"training/source_identity.json").write_text(json.dumps(sources))
    (previous/"run_manifest.json").write_text(json.dumps({"status":"failed"}))
    root = tmp_path/"configs";root.mkdir()
    config = root/"fixture.yaml";config.write_text("locked: true\n")
    members, completed, repairs = {}, {}, {}
    for arm in ("full", "no_vae"):
        train_key = f"{arm}_d0_efficientnet1dv2_ningbo"
        for suffix in ("", "_eval"):
            key = train_key+suffix;run = tmp_path/key;(run/"training").mkdir(parents=True)
            path = run/"training/result.json"
            value = ({"subject":{"train_result":{"path":completed[train_key]["result"],
                    "sha256":completed[train_key]["sha256"]}},
                "protocol":{"corruption_cache":{"execution":{"mode":"online_torch_gpu_v1",
                    "metadata_verification":{"metadata_member_count":10}}}}} if suffix else {})
            if damage == "training_link" and suffix:
                value["subject"]["train_result"] = {"path":"other", "sha256":"0"*64}
            path.write_text(json.dumps(value))
            entry = {"result":str(path), "sha256":q.sha256_file(path)}
            if suffix: repairs[key] = entry
            else: completed[key] = entry
            name = "evaluate_pn2021" if suffix else "train_pn2021"
            (run/"run_manifest.json").write_text(json.dumps({"status":"complete", "entrypoint":{"name":name},
                "config_snapshots":[{"snapshot_path":"configs/fixture.yaml", "sha256":q.sha256_file(config)}]}))
            members[key] = (None, SimpleNamespace(config_sources=[(config, "fixture")],
                run_dir=run, expected_result_relative_path=Path("training/result.json"), entrypoint_name=name))
    (previous/"training/status.json").write_text(json.dumps({"status":"draining", "completed":completed,
        "active":{"job":123} if damage == "active" else {}}))
    monkeypatch.setattr(q, "SAMPLING_REPAIR_RESULTS", repairs)
    policy = {**q.SAMPLING_RECOVERY, "previous_workflow":str(previous)}
    monkeypatch.setattr(q, "SAMPLING_RECOVERY", policy)
    monkeypatch.setattr(run_record, "verify_run_file_index", lambda path: [])
    monkeypatch.setattr(contract, "validate_pn2021_train_result", lambda *a, **kw: None)
    monkeypatch.setattr(contract, "validate_pn2021_evaluation_result", lambda *a, **kw: None)
    if damage == "source": sources = {**sources, "core/online_trainer.py":"changed"}
    elif damage == "config": config.write_text("locked: false\n")
    elif damage == "result": Path(next(iter(completed.values()))["result"]).write_text("changed")
    if damage:
        with pytest.raises(ValueError):
            q.recover_sampling_completed({"recovery":policy},members,root,sources)
    else:
        assert len(q.recover_sampling_completed({"recovery":policy},members,root,sources)) == 4


@pytest.mark.parametrize("damage", [None, "key", "path", "hash"])
def test_sampling_result_allows_only_the_exact_external_repair(tmp_path, monkeypatch, damage):
    from util import simclr_joint_queue as q
    monkeypatch.setattr(q, "SAMPLING_ROOT", tmp_path/"sampling")
    jobs = {key:{"result":str(q.SAMPLING_ROOT/key/"training/result.json"), "sha256":"0"*64} for key in q.sampling_grid()}
    key = "full_d0_efficientnet1dv2_ningbo_eval"
    approved = {"result":str(tmp_path/"repair.json"), "sha256":"0"*64}
    monkeypatch.setattr(q, "SAMPLING_REPAIR_RESULTS", {key:approved})
    jobs[key] = dict(approved)
    if damage == "key": jobs["full_d1_efficientnet1dv2_ningbo_eval"] = dict(approved)
    elif damage == "path": jobs[key]["result"] = str(tmp_path/"other.json")
    elif damage == "hash": jobs[key]["sha256"] = "f"*64
    monkeypatch.setattr(q, "sha256_file", lambda path: "0"*64)
    value = {"schema_version":6, "artifact_type":"pulse_training_queue_result", "status":"complete", "jobs":jobs,
        "sampling_comparison":{"metrics":str(tmp_path/"sampling_comparison.json"), "sha256":"0"*64}}
    if damage:
        with pytest.raises(ValueError, match="escaped"):
            q.validate_sampling_result(value,tmp_path/"queue_result.json")
    else:
        q.validate_sampling_result(value,tmp_path/"queue_result.json")


def test_sampling_queue_resolves_all_children_and_keeps_training_seed_fixed(tmp_path):
    from util import simclr_joint_queue as q
    from util.config_bundle import load_yaml_mapping
    path=REPO/"configs/train/k500_sampling/workflow.yaml"
    payload=load_yaml_mapping(path,description="sampling workflow")
    members=q.sampling_plans(payload,path,REPO/"configs")
    assert len(members)==96 and len(q.sampling_grid())==96
    from boot_scripts.run_experiment import load_experiment_plan
    parent=load_experiment_plan(REPO/"configs/experiments/k500_sampling/workflow.yaml",run_dir=tmp_path/"workflow")
    assert parent.data_ledger.split_config_relative_path.as_posix()=="data/k500_sampling/draw0/splits.yaml"
    assert not parent.run_dir.exists()
    for model in q.MODELS:
        for center in q.CENTERS:
            seeds=[];splits=[]
            for draw in range(3):
                for arm in q.SAMPLING_METHODS:
                    plan=members[f"{arm}_d{draw}_{model}_{center}"][1]
                    p=load_yaml_mapping(plan.entry_config_path,description="sampling train")
                    seeds.append(p["random_seed"]);splits.append(p["references"]["split_config"])
                    assert plan.data_ledger.split_config_relative_path.as_posix()==p["references"]["split_config"]
            assert all(s==seeds[0] for s in seeds) and len(set(splits))==3
    payload["runtime"]={**payload["runtime"],"max_workers":3}
    with pytest.raises(ValueError,match="frozen"):
        q.sampling_plans(payload,path,REPO/"configs")


@pytest.mark.parametrize("mismatch", [None,"training_seed","evaluation_cohort","K500"])
def test_sampling_summary_requires_fixed_seed_distinct_draws_and_common_evaluation(tmp_path,mismatch):
    from util import simclr_joint_queue as q
    completed={}
    for key,(arm,draw,model,center,suffix) in q.sampling_grid().items():
        path=tmp_path/f"{key}.json"
        value={"optimizer_steps":100 if model=="efficientnet1dv2" else 240}
        if suffix:
            k500=f"{center}-{draw}"
            line={"adaptation_data":{"record_count":500,"hash_id_set_sha256":k500},
                "source_checkpoint":{"sha256":"a"*64},"seed":{"effective_seed":123},
                "selection":{"policy":"last","heldout_evaluation_used_for_selection":False},
                "method":{"recipe_id":q.SAMPLING_METHODS[arm]}}
            identity={"hash_id_set_sha256":center,"hash_label_set_sha256":center,"ordered_hash_ids_sha256":center,
                "reference_exclusion":{"common_sampling_exclusion":{"policy":"common"}}}
            if arm=="full" and draw==1 and model==q.MODELS[0] and center==q.CENTERS[0]:
                if mismatch=="training_seed":line["seed"]["effective_seed"]=456
                elif mismatch=="evaluation_cohort":identity["hash_id_set_sha256"]="changed"
                elif mismatch=="K500":line["adaptation_data"]["hash_id_set_sha256"]="changed"
            metric={"macro_auroc":.9,"macro_auprc":.7+draw*.01+(.02 if arm=="full" else 0)}
            value={"status":"complete","checkpoint":{"lineage":line,"sha256":"a"*64},
                "protocol":{"corruption_cache":{"view_count":20,"execution":{"mode":"online_torch_gpu_v1"}}},
                "clean":{"per_center":{center:{"identity":identity}},"evaluated_center_mean":{"pn2021_drop_all_zero_refexcluded":metric}},
                "corrupted":{"aggregates":{"depth23":{"evaluated_center_mean":{"pn2021c_drop_all_zero_corrupted_refexcluded":metric}}}}}
        path.write_text(json.dumps(value));completed[key]={"result":str(path),"sha256":"a"*64}
    if mismatch:
        with pytest.raises(ValueError):q.sampling_finalize(completed,tmp_path)
        assert not (tmp_path/"sampling_comparison.json").exists()
    else:
        q.sampling_finalize(completed,tmp_path)
        report=json.loads((tmp_path/"sampling_comparison.json").read_text())
        assert len(report["rows"])==48
        assert report["sampling_summary"][q.MODELS[0]]["full"]["corrupt_macro_auprc"]["sample_sd"]==pytest.approx(.01)
        assert next(c for c in report["paired_contrasts"] if c["metric"]=="corrupt_macro_auprc")["full_minus_no_vae_pp"]==pytest.approx(2)
CONFIG = REPO / "configs/train/pulse_training_queue.yaml"


def test_simclr_joint_grid_configs_and_side_effect_free_parent_plan(tmp_path):
    from util import simclr_joint_queue as joint
    path = REPO / "configs/train/simclr_joint/workflow.yaml"
    payload = yaml.safe_load(path.read_text())
    jobs = joint.plans(payload, path, REPO / "configs")
    assert len(jobs) == 96
    assert len({p.run_dir for _, p in jobs.values()}) == 96
    parent = load_experiment_plan(REPO / "configs/experiments/simclr_joint/workflow.yaml",
        run_dir=tmp_path / "dry_only")
    assert not parent.run_dir.exists()
    assert not parent.describe()["loads_model_or_gpu"]
    for seed in joint.SEEDS:
        a = yaml.safe_load((REPO/f"configs/train/simclr_joint/joint_s{seed}.yaml").read_text())
        b = yaml.safe_load((REPO/f"configs/train/simclr_joint/two_stage_s{seed}.yaml").read_text())
        assert a["random_seed"] == b["random_seed"]
        assert a["data"] == b["data"] and a["protocol"] == b["protocol"]


@pytest.mark.parametrize("change", ["extra_job", "gpu_workers", "output_root"])
def test_simclr_joint_queue_rejects_expanded_scope(change):
    from util import simclr_joint_queue as joint
    path = REPO / "configs/train/simclr_joint/workflow.yaml"
    payload = yaml.safe_load(path.read_text())
    if change == "extra_job":
        payload["references"]["extra_seed"] = next(iter(payload["references"].values()))
    elif change == "gpu_workers":
        payload["runtime"]["max_workers"] = 2
    else:
        payload["run_root"] += "_changed"
    with pytest.raises(ValueError):
        joint.plans(payload, path, REPO / "configs")


@pytest.mark.parametrize("mismatch", [None, "source", "seed", "evaluation"])
def test_simclr_joint_comparison_pairs_cohorts_before_writing(tmp_path, mismatch):
    from util import simclr_joint_queue as joint
    completed = {}
    for index, (key, (arm, seed, model, center, suffix)) in enumerate(joint.grid().items()):
        path = tmp_path/f"{key}.json"
        if suffix:
            lineage = {"adaptation_data":{"record_count":500,"center":center},
                "source_checkpoint":{"sha256":"a"*64}, "seed":{"effective_seed":seed+10},
                "selection":{"policy":"last","heldout_evaluation_used_for_selection":False},
                "method":{"recipe_id":joint.METHODS[arm]}}
            identity = {"hash_id_set_sha256":center,"hash_label_set_sha256":center,
                        "reference_exclusion":{"reference_count":500}}
            if arm == "joint" and seed == 0 and model == "efficientnet1dv2" and center == "ningbo":
                if mismatch == "source":lineage["source_checkpoint"]["sha256"] = "b"*64
                if mismatch == "seed":lineage["seed"]["effective_seed"] = 999
                if mismatch == "evaluation":identity["hash_id_set_sha256"] = "other"
            metric = {"macro_auroc":.9,"macro_auprc":.7 + (.001 if arm=="joint" else 0)}
            value = {"status":"complete","checkpoint":{"lineage":lineage,"sha256":"a"*64},
                "protocol":{"mapping_hash":"555ec85d5b51","logical_centers":[center],
                            "corruption_cache":{"view_count":20}},
                "clean":{"per_center":{center:{"identity":identity}},
                         "evaluated_center_mean":{"pn2021_drop_all_zero_refexcluded":metric}},
                "corrupted":{"aggregates":{"depth23":{"evaluated_center_mean":{
                    "pn2021c_drop_all_zero_corrupted_refexcluded":metric}}}}}
        else:
            value = {"optimizer_steps":100 if model=="efficientnet1dv2" else 240}
        path.write_text(json.dumps(value))
        completed[key] = {"result":str(path),"sha256":"a"*64}
    output = tmp_path / "report"
    output.mkdir()
    if mismatch:
        with pytest.raises(ValueError, match="unmatched"):
            joint.finalize(completed, output)
        assert not (output / "comparison.json").exists()
    else:
        joint.finalize(completed, output)
        report = json.loads((output / "comparison.json").read_text())
        assert len(report["rows"]) == 48
        for contrast in report["paired_contrasts"]:
            expected = .1 if "auprc" in contrast["metric"] else 0
            assert contrast["joint_minus_two_stage_pp"] == pytest.approx(expected)


@pytest.mark.parametrize("changed", [None, "util/simclr_joint_queue.py", "core/online_trainer.py"])
def test_joint_recovery_allows_only_scheduler_source_changes(changed):
    from util import simclr_joint_queue as joint
    before = {"util/simclr_joint_queue.py": "old", "core/online_trainer.py": "frozen"}
    after = dict(before)
    if changed:
        after[changed] = "new"
    if changed == "core/online_trainer.py":
        with pytest.raises(ValueError, match="source changed"):
            joint.admission_source_delta(before, after, recovery=True)
    else:
        assert set(joint.admission_source_delta(before, after, recovery=True)) == ({changed} if changed else set())
    if changed:
        with pytest.raises(ValueError):
            joint.admission_source_delta(before, after, recovery=False)


@pytest.mark.parametrize("corruption", [None, "config", "result", "incomplete"])
def test_joint_recovery_checks_completed_child_bytes_and_configs(tmp_path, monkeypatch, corruption):
    from types import SimpleNamespace
    from util import simclr_joint_queue as joint, run_record
    prior = tmp_path/"prior"
    (prior/"training").mkdir(parents=True)
    child = tmp_path/"child"
    child.mkdir()
    config_root = tmp_path/"configs"
    config_root.mkdir()
    config = config_root/"seed.yaml"
    config.write_text("replicate_id: 0\n")
    result = child/"train_result.json"
    result.write_text('{"status":"complete"}')
    entry = {"result": str(result), "sha256": joint.sha256_file(result)}
    (prior/"run_manifest.json").write_text(json.dumps({"status":"failed"}))
    (prior/"training/status.json").write_text(json.dumps({"status":"failed", "completed":{"job":entry}}))
    sources = {"core/online_trainer.py":"frozen", "util/simclr_joint_queue.py":"old"}
    (prior/"training/source_identity.json").write_text(json.dumps(sources))
    (child/"run_manifest.json").write_text(json.dumps({"status":"complete" if corruption!="incomplete" else "failed",
        "entrypoint":{"name":"train_pn2021"}, "config_snapshots":[{
            "snapshot_path":"configs/seed.yaml", "sha256":joint.sha256_file(config)}]}))
    if corruption == "config": config.write_text("replicate_id: 1\n")
    if corruption == "result": result.write_text('{"status":"changed"}')
    monkeypatch.setitem(joint.RECOVERY, "previous_workflow", str(prior))
    monkeypatch.setattr(run_record, "verify_run_file_index", lambda path: [])
    members = {"job": (None, SimpleNamespace(run_dir=child, expected_result_relative_path="train_result.json",
                                             entrypoint_name="train_pn2021"))}
    if corruption:
        with pytest.raises(ValueError):
            joint.recover_completed(members, config_root, sources)
    else:
        assert joint.recover_completed(members, config_root, sources) == {"job":entry}


def test_joint_recovery_waits_for_full_cache_verification(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from util import simclr_joint_queue as joint
    receipt = tmp_path/"receipt.json"
    cache = tmp_path/"cache"
    cache.mkdir()
    assert not joint.restored_cache_ready(receipt, cache)
    waveform = cache/"signals_100hz.npy"
    waveform.touch()
    fake = SimpleNamespace(st_dev=1, st_ino=2, st_size=63192000128, st_mtime_ns=3, st_ctime_ns=4)
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda path, *a, **kw: fake if path == waveform else original_stat(path,*a,**kw))
    value = {"status":"complete", "cache_dir":str(cache), "verification_mode":"full_content",
        "waveform_sha256":"8f416d86911e711dc3e359dd1284e36e473253bd28201d3b8eec1bb3a2972d6c",
        "ledger_sha256":"5991ac2123e9fe8774df7cb4a311fd0f4332554e51a77ea2a29ac2ce05ad5f13",
        "waveform_stat":vars(fake)}
    receipt.write_text(json.dumps(value))
    assert joint.restored_cache_ready(receipt, cache)
    fake.st_mtime_ns += 1
    with pytest.raises(ValueError, match="identity differs"):
        joint.restored_cache_ready(receipt, cache)


def test_gpu_online_joint_queue_preserves_training_and_finite_grid():
    from util import simclr_joint_queue as joint
    path = REPO/"configs/train/simclr_joint/workflow_online_gpu1.yaml"
    jobs = joint.plans(yaml.safe_load(path.read_text()), path, REPO/"configs")
    assert len(jobs) == 96
    for key, (_, plan) in jobs.items():
        if key.endswith("_eval"):
            assert plan.entry_config_path.name == "PN2021_online_gpu.yaml"
            assert "pn2021c_cache" not in plan.data_ledger.required_roots
    before = {"core/online_trainer.py":"frozen","core/corruption.py":"old"}
    after = {**before,"core/corruption.py":"new"}
    assert set(joint.admission_source_delta(before, after, recovery=True, online=True)) == {"core/corruption.py"}
    with pytest.raises(ValueError):
        joint.admission_source_delta(before, {**after,"core/online_trainer.py":"changed"}, recovery=True, online=True)



def config():
    return yaml.safe_load(CONFIG.read_text())


def test_queue_closure_and_exact_paired_jobs(tmp_path):
    payload = config()
    validate_config(payload)
    jobs = load_jobs(payload, CONFIG, REPO / "configs")
    assert tuple(jobs) == JOB_KEYS
    assert len({j["run"] for j in jobs.values()}) == 8
    plan = load_experiment_plan(REPO / "configs/experiments/pulse_training_queue.yaml", run_dir=tmp_path / "output")
    assert len(plan.config_sources) == 22
    assert plan.expected_result_type == "pulse_training_queue_result"
    assert plan.data_ledger.required_roots == ("pn2021_cache", "split_artifacts")
    assert not plan.describe()["loads_model_or_gpu"]
    assert not plan.run_dir.exists()


@pytest.mark.parametrize("change", ["extra_job", "max_workers", "missing_admission", "wrong_order"])
def test_queue_rejects_scope_and_admission_changes(change):
    payload = copy.deepcopy(config())
    if change == "extra_job":
        payload["references"]["w3_ningbo"] = "experiments/pulse_augmix_w2_ningbo.yaml"
    elif change == "max_workers":
        payload["runtime"]["max_workers"] = 8
    elif change == "missing_admission":
        del payload["admission_results"]["resume"]
    else:
        payload["references"] = dict(reversed(list(payload["references"].items())))
    with pytest.raises(ValueError):
        validate_config(payload)


@pytest.mark.parametrize("value", ["", "0,0", "8", "-1"])
def test_gpu_allowlist_rejects_implicit_or_invalid_devices(monkeypatch, value):
    monkeypatch.setenv("PULSE_GPU_ALLOWLIST", value)
    with pytest.raises(ValueError):
        allowed_gpus()


def test_gpu_allowlist_is_explicit(monkeypatch):
    monkeypatch.setenv("PULSE_GPU_ALLOWLIST", "6,5,4,0")
    assert allowed_gpus() == [6, 5, 4, 0]


def test_stale_progress_is_not_live_and_is_not_relaunched(tmp_path):
    job = {"run": tmp_path / "run", "delegate": tmp_path / "run/training"}
    assert job_state(job, live=False) == "pending"
    job["delegate"].mkdir(parents=True)
    (job["delegate"] / "progress.json").write_text(json.dumps({"pid": os.getpid(), "step": 7}))
    assert live_training_jobs({"test": job}) == {}
    assert job_state(job, live=False) == "stopped_incomplete_requires_explicit_resume"
    assert job_state(job, live=True) == "running"
    (job["run"] / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
    assert job_state(job, live=False) == "complete"


def test_cpu_loading_children_and_adopted_jobs_reserve_their_devices():
    from util.pulse_training_queue import unclaimed_training_devices

    devices = {"loading": 0, "adopted": 2, "completed": 1}
    # Loading launchers need not appear in the live training delegates yet.
    assert unclaimed_training_devices(
        [0, 1, 2, 3], devices, live={"adopted": {}}, children={"loading": {}},
    ) == [1, 3]
    # A delegate may be both discovered and owned; it still claims only one GPU.
    assert unclaimed_training_devices(
        [3, 2, 1, 0], devices, live={"loading": {}, "adopted": {}}, children={"loading": {}},
    ) == [3, 1]
    # Historical device assignments must not hold a completed job's GPU forever.
    assert unclaimed_training_devices([0, 1, 2, 3], devices, live={}, children={}) == [0, 1, 2, 3]


@pytest.mark.parametrize("live,children", [({"adopted": {}}, {}), ({}, {"loading": {}})])
def test_active_job_without_device_identity_blocks_new_gpu_assignments(live, children):
    from util.pulse_training_queue import unclaimed_training_devices

    assert unclaimed_training_devices([0, 1], {}, live=live, children=children) == []


@pytest.mark.parametrize("schema,expected_policy,count", [
    (1, "all_eight_matched_last_checkpoints", 8),
    (2, "all_eight_matched_last_checkpoints", 8),
    (3, "all_44_dual_jsd_training_and_evaluation_jobs", 44),
    (4, "width1_width3_four_centers_last_checkpoints_frozen_width2_baselines", 16),
    (5, "all_96_paired_three_seed_simclr_training_and_evaluation_jobs", 96),
    (6, "all_96_fixed_training_seed_three_K500_sampling_draw_jobs", 96),
])
def test_queue_summary_uses_validated_schema(monkeypatch, tmp_path, schema, expected_policy, count):
    from util import pulse_training_queue as queue
    from util.founder_width_queue import KEYS as WIDTH_KEYS
    from util.simclr_joint_queue import grid, sampling_grid
    from util.run_record import _result_summary

    keys = (sampling_grid() if schema == 6 else grid() if schema == 5 else queue.DUAL_KEYS if schema == 3
            else WIDTH_KEYS if schema == 4 else JOB_KEYS)
    assert len(keys) == count
    payload = {"artifact_type": "pulse_training_queue_result", "schema_version": schema,
               "status": "complete", "jobs": dict.fromkeys(keys, {}), "admission": {}}
    path = tmp_path / "queue_result.json"
    seen = []
    # Result validators own artifact checks; isolate only the summary projection.
    monkeypatch.setattr(queue, "validate_queue_result", lambda value, member: seen.append((value, member)))
    summary = _result_summary(payload, "pulse_training_queue_result", result_path=path)
    assert seen == [(payload, path)]
    assert summary == {"identity": payload["jobs"], "selection": {"policy": expected_policy}}

    def reject(value, member):
        raise ValueError("invalid queue evidence")

    monkeypatch.setattr(queue, "validate_queue_result", reject)
    with pytest.raises(ValueError, match="invalid queue evidence"):
        _result_summary(payload, "pulse_training_queue_result", result_path=path)
