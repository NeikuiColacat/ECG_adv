"""Subset identity, resource limits, baseline parsing and task completeness."""
import copy
from pathlib import Path

import pytest
import yaml

from util.evaluation.pulse_subset import selected_tasks, validate_config, scheduling_limit, validate_original_row
from util.evaluation.ecg_image_queue import fixed_tasks
from util.evaluation.ecg_image_artifact import parse_response
from util.pulse_training_contract import CENTERS

REPO = Path(__file__).resolve().parents[2]


def test_subset_config_and_dry_run(tmp_path):
    from boot_scripts.run_experiment import load_experiment_plan
    config = yaml.safe_load((REPO / "configs/eval/pulse_subset.yaml").read_text())
    validate_config(config)
    plan = load_experiment_plan(REPO / "configs/experiments/pulse_subset.yaml", run_dir=tmp_path / "dry")
    assert plan.entrypoint_name == "evaluate_pulse_subset"
    for key, value in (("max_workers", 5), ("initial_workers", 4), ("idle_seconds", 0)):
        bad = copy.deepcopy(config); bad["runtime"][key] = value
        with pytest.raises(ValueError): validate_config(bad)
    config["records_per_center"] = 513
    with pytest.raises(ValueError): validate_config(config)


def test_subset_preserves_batch_ids_prefix_and_balance():
    tasks=[]
    for c in CENTERS:
        rows=[{"sample_key": f"{c}-{i}"} for i in range(520)]
        group=fixed_tasks(rows,records_per_task=8,batch_size=2,condition_ids=["clean"],protocol_identity="old")
        tasks.extend({**t,"center":c} for t in group)
    subset=selected_tasks(tasks,512)
    assert len(subset)==256
    assert all(t in tasks for t in subset)
    assert all(sum(t["center"]==c for t in subset)==64 for c in CENTERS)
    with pytest.raises(ValueError): selected_tasks(tasks[:-4],512)


def test_subset_starts_two_and_expands_only_after_admission_and_delay():
    config=yaml.safe_load((REPO / "configs/eval/pulse_subset.yaml").read_text())
    runtime=config["runtime"]
    assert scheduling_limit(runtime,0,both_lanes_verified=True)==2
    assert scheduling_limit(runtime,1000,both_lanes_verified=False)==2
    assert scheduling_limit(runtime,900,both_lanes_verified=True)==4


def test_original_result_parser_and_fingerprint_contract():
    sample={"sample_key":"x","record_id":"x","hash_id":"x","logical_center":"ningbo","label_names":["CD"]}
    condition={"condition_id":"clean","operators":[],"depth":0}
    answer={"response":"CD","generated_tokens":2,"hit_max_new_tokens":False,**parse_response("CD")}
    row={**sample,**condition,"true_labels":["CD"],"answer":answer,
         "clean_waveform_sha256":"a"*64,"input_waveform_sha256":"b"*64}
    validate_original_row(row,sample,condition)
    bad=copy.deepcopy(row);bad["answer"]["predicted_labels"]=["MI"]
    with pytest.raises(ValueError):validate_original_row(bad,sample,condition)
    bad=copy.deepcopy(row);bad["input_waveform_sha256"]=""
    with pytest.raises(ValueError):validate_original_row(bad,sample,condition)


@pytest.mark.parametrize("schema,expected_policy", [
    (1, "fixed_subset_original_single_two_no_retuning"),
    (2, "fixed_subset_original_clean_single_three_no_retuning"),
])
def test_subset_summary_preserves_the_schema_model_arms(monkeypatch, tmp_path, schema, expected_policy):
    from util.evaluation import pulse_subset, pulse_visual_subset
    from util.run_record import _result_summary

    names = pulse_subset.NAMES if schema == 1 else pulse_visual_subset.NAMES
    assert len(names) == (3 if schema == 1 else 4)
    payload = {"artifact_type": "pulse_subset_result", "schema_version": schema,
               "status": "complete", "records_per_center": 512, "models": list(names),
               "subset_identity": "test", "files": {}}
    path = tmp_path / "subset_result.json"
    seen = []
    # Existing result validators own the prediction grid and fingerprint checks.
    monkeypatch.setattr(pulse_subset, "validate_result", lambda value, member: seen.append((value, member)))
    summary = _result_summary(payload, "pulse_subset_result", result_path=path)
    assert seen == [(payload, path)]
    assert summary == {"identity": {"subset_identity": "test"}, "selection": {"policy": expected_policy}}

    def reject(value, member):
        raise ValueError("invalid subset evidence")

    monkeypatch.setattr(pulse_subset, "validate_result", reject)
    with pytest.raises(ValueError, match="invalid subset evidence"):
        _result_summary(payload, "pulse_subset_result", result_path=path)


def test_complete_three_arm_grid_and_reject_changed_inputs(tmp_path):
    from util.evaluation.pulse_subset import comparison_arrays, finalize, validate_result
    from util.evaluation.ecg_image_queue import TaskQueue, atomic_json
    import json
    samples=[{"sample_key":c,"record_id":c,"hash_id":c,"logical_center":c,"label_names":["CD"]} for c in CENTERS]
    conditions=[{"condition_id":"clean" if j==0 else f"c{j}","condition_index":j,"operators":[],"depth":0} for j in range(21)]
    tasks=fixed_tasks(samples,records_per_task=2,batch_size=2,condition_ids=[c["condition_id"] for c in conditions],protocol_identity="test")
    prepared={"rows":samples,"conditions":conditions,"subset_config":{"records_per_center":1},"subset_identity":"test","tasks":tasks,"base_tasks":tasks}
    pair=TaskQueue(tmp_path/"pair_queue",tasks);base=TaskQueue(tmp_path/"base_queue",tasks)
    answer={"response":"CD","generated_tokens":2,"hit_max_new_tokens":False,**parse_response("CD")}
    rows=[{**s,**c,"schema_version":1,"true_labels":["CD"],"clean_waveform_sha256":"a"*64,"input_waveform_sha256":"b"*64} for s in samples for c in conditions]
    for queue,kind in ((pair,"pair"),(base,"base")):
        while (lease:=queue.claim("test")) is not None:
            with lease:
                keys={k for b in lease.task["batches"] for k in b}
                selected=[{**r,**({"answer":answer} if kind=="base" else {"arms":{"w1":answer,"w2":answer}})} for r in rows if r["sample_key"] in keys]
                lease.commit(selected,{"seconds":1})
    truth,predictions,quality=comparison_arrays(prepared,pair,base)
    assert all(p.shape==(3,1,21,5) and p[:,:,:,0].all() for p in predictions.values())
    bad=copy.deepcopy(prepared);bad["subset_config"]["records_per_center"]=2
    with pytest.raises(ValueError,match="unbalanced"):comparison_arrays(bad,pair,base)
    # Run final reporting and independent verification on the complete synthetic grid.
    for name,value in (("prepared",prepared),("cohort",samples),("data_audit",{}),("reuse",{"pair_tasks":[]})):
        atomic_json(tmp_path/f"{name}.json",value)
    (tmp_path/"workers").mkdir()
    finalize(tmp_path,prepared,pair,base)
    result=json.loads((tmp_path/"subset_result.json").read_text())
    validate_result(result,tmp_path/"subset_result.json")
    exact=json.loads((tmp_path/"exact_bootstrap_0_1.json").read_text())
    assert "PULSE-original" in exact["intervals"]["center_equal"]["models"]
    assert "candidate_minus_reference" in exact["intervals"]["center_equal"]
    payload=base.validate_completed(tasks[0]);payload["predictions"][0]["input_waveform_sha256"]="c"*64
    from util.evaluation.ecg_image_queue import digest_json
    payload["rows_sha256"]=digest_json(payload["predictions"])
    atomic_json(base.result_path(tasks[0]),payload)
    with pytest.raises(ValueError,match="waveforms differ"):comparison_arrays(prepared,pair,base)
