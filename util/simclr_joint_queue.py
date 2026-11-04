"""Single-GPU, finite three-seed comparison of joint and two-stage SimCLR.

The existing managed launcher owns every training/evaluation child. This queue
does not load models, search hyperparameters, overwrite or retry failed runs.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.evaluation.ecg_image_elastic import free_device, process_identity, resource_snapshot
from util.evaluation.ecg_image_queue import atomic_json, exclusive_lock
from util.pn2021_artifact_contract import sha256_file

REPO = Path(__file__).resolve().parents[1]
MODELS = ("efficientnet1dv2", "ecgfounder")
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
ARMS = ("joint", "two_stage")
SEEDS = (0, 1, 2)
METHODS = {"joint": "augmix_simclr_joint_no_vae", "two_stage": "augmix_simclr_matched_no_vae"}
GROUP = "simclr_joint_novae_20261104_r0"
ROOT = Path("/home/linbinhao/ECG_adv_data/runs") / GROUP
LIMITS = {"max_workers": 1, "poll_seconds": 30, "idle_seconds": 30,
          "min_available_ram_gib": 64, "max_load_fraction": .8, "min_disk_free_gib": 40}
METRICS = ("clean_macro_auroc", "clean_macro_auprc", "corrupt_macro_auroc", "corrupt_macro_auprc")
RECOVERY = {"previous_workflow": str(ROOT/"workflow"),
            "cache_receipt": str(ROOT/"cache_restore_result.json"),
            "schedule": "training_then_evaluation"}
ONLINE_RECOVERY = {"previous_workflow": str(ROOT/"workflow_recovery1"),
                   "evaluation_source": "online_torch_gpu_v1", "schedule": "interleaved"}
ONLINE_SOURCE_CHANGES = {"util/simclr_joint_queue.py", "core/corruption.py",
    "data_preprocess/data_ledger.py", "util/evaluation/pn2021.py", "boot_scripts/run_experiment.py"}
SAMPLING_GROUP = "k500_sampling_20261104_r0"
SAMPLING_ROOT = ROOT.parent/SAMPLING_GROUP
SAMPLING_METHODS = {"full": "augmix_simclr_lhat", "no_vae": "augmix_simclr_matched_no_vae"}
SAMPLING_LIMITS = {"max_workers": 2, "poll_seconds": 15, "idle_seconds": 30,
    "min_available_ram_gib": 64, "max_load_fraction": .8, "min_disk_free_gib": 25}
SAMPLING_REPAIR_RESULTS = {
    "full_d0_efficientnet1dv2_ningbo_eval": {
        "result": str(ROOT.parent/"traditional_simclr_fixcheck_full_20261104_r0/evaluation/evaluation_result.json"),
        "sha256": "cc5a0fb9c4cc91ef2b74bbb85db7f120f3090ef2e841f12454bdb9075adb398d"},
    "no_vae_d0_efficientnet1dv2_ningbo_eval": {
        "result": str(ROOT.parent/"traditional_simclr_fixcheck_20261104_r0/evaluation/evaluation_result.json"),
        "sha256": "7300869ccd4b3e4ee961b16edb8568e371e3fc6468a6133b23afc0aabe01ddce"},
}
SAMPLING_RECOVERY = {
    "previous_workflow": str(SAMPLING_ROOT/"workflow"),
    "engineering_admission": str(SAMPLING_ROOT/"engineering_admission_recovery1.json"),
    "reuse_policy": "two_original_trainings_and_two_verified_repair_evaluations_v1",
}
SAMPLING_NVME_ROOT = Path("/data_nvme/linbinhao/ECG_adv_data/runs/k500_sampling_20261104_r0")
SAMPLING_NVME_RECOVERY = {
    "previous_workflow": str(SAMPLING_ROOT/"workflow_recovery1"),
    "engineering_admission": str(SAMPLING_ROOT/"engineering_admission_recovery2.json"),
    "reuse_policy": "verified_32_completed_draw0_results_v1",
    "storage_root": str(SAMPLING_NVME_ROOT),
}


def sampling_grid():
    return {f"{arm}_d{draw}_{model}_{center}{suffix}": (arm, draw, model, center, suffix)
        for draw in (0,1,2) for center in CENTERS for model in MODELS
        for arm in SAMPLING_METHODS for suffix in ("", "_eval")}


def sampling_plans(config, path, root):
    from boot_scripts.run_experiment import load_experiment_plan
    required = {"schema_version", "run_root", "references", "runtime", "admission_results", "cohort_report"}
    recovery = "recovery" in config
    if (set(config) != required | ({"recovery"} if recovery else set())
            or config["schema_version"] != 6 or Path(config["run_root"]) != SAMPLING_ROOT
            or tuple(config["references"]) != (*sampling_grid(), "split_config", "data_load_config")
            or config["references"]["split_config"] != "data/k500_sampling/draw0/splits.yaml"
            or config["references"]["data_load_config"] != "data/k500_sampling/draw0/data_load.yaml"
            or config["runtime"] != SAMPLING_LIMITS
            or config["cohort_report"] != str(SAMPLING_ROOT/"sampling_cohorts.json")):
        raise ValueError("sampling queue requires frozen three draws, two methods/backbones, four centers")
    if recovery and config["recovery"] not in (SAMPLING_RECOVERY, SAMPLING_NVME_RECOVERY):
        raise ValueError("sampling recovery is restricted to the original stopped queue")
    suffix = "_recovery2" if config.get("recovery") == SAMPLING_NVME_RECOVERY else "_recovery1" if recovery else ""
    expected_admissions = {f"{arm}_{model}": str(SAMPLING_ROOT/f"engineering_{arm}_{model}{suffix}/training/train_result.json")
        for arm in SAMPLING_METHODS for model in MODELS}
    if config["admission_results"] != expected_admissions:
        raise ValueError("four real-model sampling engineering admissions are required")
    members = {}
    for key,(arm,draw,model,center,suffix) in sampling_grid().items():
        experiment = resolve_config_reference(config["references"][key], owner_config_path=path,
            config_root=root, description=key, must_exist=True)
        plan = load_experiment_plan(experiment, config_root=root, allow_existing_run_dir=True)
        method = f"train/methods/{SAMPLING_METHODS[arm]}.yaml"
        alias = "effnet" if model == "efficientnet1dv2" else model
        source = f"/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_{alias}_source_v1/training/checkpoints/best.pt"
        args = (["--model",model,"--train-result",str(SAMPLING_ROOT/key.removesuffix("_eval")/"training/train_result.json"),
                 "--center",center,"--method-config",method] if suffix else
                ["--model",model,"--method-config",method,"--center",center,"--source-checkpoint",source])
        expected_config = root/(f"eval/k500_sampling/draw{draw}.yaml" if suffix else f"train/k500_sampling/{arm}_d{draw}.yaml")
        if (plan.run_dir != (SAMPLING_ROOT/key).resolve() or list(plan.entry_arguments) != args
                or plan.entry_config_path != expected_config
                or plan.entrypoint_name != ("evaluate_pn2021" if suffix else "train_pn2021")):
            raise ValueError("sampling child escaped its frozen identity")
        payload = load_yaml_mapping(expected_config, description="sampling profile")
        if payload["references"]["split_config"] != f"data/k500_sampling/draw{draw}/splits.yaml":
            raise ValueError("sampling child uses a different draw")
        if not suffix:
            if payload["random_seed"] != {"stream_namespace": SAMPLING_GROUP, "comparison_group": SAMPLING_GROUP,
                    "replicate_id":0,"deterministic_algorithms":False}:
                raise ValueError("sampling repeats must keep the training seed fixed")
            for name,p in payload["training"]["model_profiles"].items():
                budget = (25,128,5e-5,1024) if name == "efficientnet1dv2" else (30,64,3e-5,256)
                if (p["epochs"],p["batch_size"],p["learning_rate"],p["stage1_steps"]) != budget:
                    raise ValueError("sampling changes only the K500 draw, not training budgets")
        members[key] = (experiment,plan)
    return members


def sampling_finalize(completed, output):
    rows, pairs, common, fixed_seeds, subsets = [], {}, {}, {}, {}
    for key,(arm,draw,model,center,suffix) in sampling_grid().items():
        if not suffix:
            continue
        value = json.loads(Path(completed[key]["result"]).read_text())
        train = json.loads(Path(completed[key.removesuffix("_eval")]["result"]).read_text())
        line = value["checkpoint"]["lineage"]
        identity = value["clean"]["per_center"][center]["identity"]
        reference = identity["reference_exclusion"]
        public = reference.get("common_sampling_exclusion")
        if (value["status"] != "complete" or not public
                or value["protocol"]["corruption_cache"]["execution"]["mode"] != "online_torch_gpu_v1"
                or value["protocol"]["corruption_cache"]["view_count"] != 20
                or line["method"]["recipe_id"] != SAMPLING_METHODS[arm]
                or line["adaptation_data"]["record_count"] != 500
                or line["selection"] != {"policy":"last","heldout_evaluation_used_for_selection":False}
                or train["optimizer_steps"] != (100 if model == "efficientnet1dv2" else 240)):
            raise ValueError("sampling result changed the frozen development contract")
        pair = {k:line[k] for k in ("adaptation_data","source_checkpoint","seed","selection")}
        if pairs.setdefault((draw,model,center),pair) != pair:
            raise ValueError("sampling methods do not share source/K500/training seed")
        if fixed_seeds.setdefault((model,center),line["seed"]) != line["seed"]:
            raise ValueError("training seed changed across sampling draws")
        cohort = {k:identity[k] for k in ("hash_id_set_sha256","hash_label_set_sha256","ordered_hash_ids_sha256")}
        cohort["common_sampling_exclusion"] = public
        if common.setdefault(center,cohort) != cohort:
            raise ValueError("common evaluation cohort changed across draws/models/methods")
        k500 = line["adaptation_data"]["hash_id_set_sha256"]
        if subsets.setdefault((draw,center),k500) != k500:
            raise ValueError("K500 draw changed between models or methods")
        clean = value["clean"]["evaluated_center_mean"]["pn2021_drop_all_zero_refexcluded"]
        corrupt = value["corrupted"]["aggregates"]["depth23"]["evaluated_center_mean"]["pn2021c_drop_all_zero_corrupted_refexcluded"]
        rows.append({"arm":arm,"draw":draw,"model":model,"center":center,"result":completed[key],
            "checkpoint_sha256":value["checkpoint"]["sha256"],"k500_sha256":k500,
            **{f"{kind}_{metric}":data[metric] for kind,data in (("clean",clean),("corrupt",corrupt))
               for metric in ("macro_auroc","macro_auprc")}})
    if len(rows) != 48 or any(len({subsets[(d,c)] for d in (0,1,2)}) != 3 for c in CENTERS):
        raise ValueError("all three distinct sampling draws must finish")
    summary, contrasts = {}, []
    for model in MODELS:
        summary[model] = {}
        for arm in SAMPLING_METHODS:
            summary[model][arm] = {}
            for metric in METRICS:
                means = [statistics.mean(r[metric] for r in rows if (r["arm"],r["draw"],r["model"]) == (arm,d,model)) for d in (0,1,2)]
                summary[model][arm][metric] = {"mean":statistics.mean(means),"sample_sd":statistics.stdev(means),
                    "sampling_draw_means":means,"per_center":{c:[next(r[metric] for r in rows if
                    (r["arm"],r["draw"],r["model"],r["center"])==(arm,d,model,c)) for d in (0,1,2)] for c in CENTERS}}
        for metric in METRICS:
            delta = [100*(a-b) for a,b in zip(summary[model]["full"][metric]["sampling_draw_means"],
                summary[model]["no_vae"][metric]["sampling_draw_means"])]
            contrasts.append({"model":model,"metric":metric,"full_minus_no_vae_pp":statistics.mean(delta),
                "sampling_draw_delta_pp":delta,"sample_sd_pp":statistics.stdev(delta)})
    report = {"status":"complete","evidence":"three_K500_sampling_draws_fixed_training_seed_development_not_paper_final",
        "aggregation":"equal_20_views_then_equal_four_centers_then_sampling_draw_mean",
        "corruption_backend":"online_torch_gpu_v1","rows":rows,"sampling_summary":summary,
        "paired_contrasts":contrasts,"common_cohort":common,
        "caveats":["variability_across_K500_sampling_draws_not_training_seed_variability",
            "three_draws_descriptive_fixed_development_centers_historically_tuned",
            "comparison_to_historical_cached_NumPy_metrics_not_matched"]}
    atomic_json(output/"sampling_comparison.json",report)
    with (output/"sampling_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=["arm","draw","model","center",*METRICS])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    return {"metrics":str(output/"sampling_comparison.json"),"sha256":sha256_file(output/"sampling_comparison.json")}


def validate_sampling_result(value, path):
    if (value.get("schema_version") != 6 or value.get("status") != "complete"
            or value.get("artifact_type") != "pulse_training_queue_result"
            or set(value.get("jobs",{})) != set(sampling_grid())):
        raise ValueError("incomplete K500 sampling queue")
    for key, entry in value["jobs"].items():
        member = Path(entry["result"])
        if not member.resolve().is_relative_to(SAMPLING_ROOT.resolve()):
            if entry != SAMPLING_REPAIR_RESULTS.get(key):
                raise ValueError("sampling result escaped its original root or exact repair identity")
        if sha256_file(member) != entry["sha256"]:
            raise ValueError("sampling result bytes changed")
    ref = value["sampling_comparison"]
    if Path(ref["metrics"]) != path.parent/"sampling_comparison.json" or sha256_file(ref["metrics"]) != ref["sha256"]:
        raise ValueError("sampling comparison changed")


def recover_sampling_completed(config, members, config_root, sources):
    """Reuse the policy's exact completed set with byte/config/lineage checks."""
    from util.run_record import verify_run_file_index
    from util.pn2021_artifact_contract import (
        validate_pn2021_train_result, validate_pn2021_evaluation_result,
    )
    policy = config.get("recovery")
    if policy not in (SAMPLING_RECOVERY, SAMPLING_NVME_RECOVERY):
        raise ValueError("sampling recovery identity differs from the frozen policy")
    previous = Path(policy["previous_workflow"])
    manifest = json.loads((previous/"run_manifest.json").read_text())
    status = json.loads((previous/"training/status.json").read_text())
    if (manifest["status"] != "failed" or status["status"] not in {"failed", "draining"}
            or status.get("active") or verify_run_file_index(previous)):
        raise ValueError("sampling recovery requires the intact stopped workflow")
    old = json.loads((previous/"training/source_identity.json").read_text())
    changed = {rel for rel in old.keys() | sources.keys() if old.get(rel) != sources.get(rel)}
    allowed = {"data_preprocess/data_ledger.py", "util/simclr_joint_queue.py"}
    if policy == SAMPLING_NVME_RECOVERY:
        allowed.add("util/run_record.py")  # Seal all declared per-draw ledgers.
    if not changed.issubset(allowed):
        raise ValueError("sampling recovery changed training/model/evaluation source")
    for rel, digest in old.items():
        if sha256_file(previous/"training/source_snapshot"/rel) != digest:
            raise ValueError("original sampling source snapshot changed")
    completed = dict(status["completed"])
    if policy == SAMPLING_NVME_RECOVERY:
        expected_keys = {key for key, (_, draw, _, _, _) in sampling_grid().items() if draw == 0}
        if set(completed) != expected_keys or any(completed.get(k) != v for k,v in SAMPLING_REPAIR_RESULTS.items()):
            raise ValueError("NVMe recovery requires exactly the verified 32 completed draw0 results")
    else:
        if set(completed) != {key.removesuffix("_eval") for key in SAMPLING_REPAIR_RESULTS}:
            raise ValueError("sampling recovery requires exactly the two original successful trainings")
        completed.update(SAMPLING_REPAIR_RESULTS)
    for key, entry in completed.items():
        plan = members[key][1]
        path = Path(entry["result"])
        run = path.parent.parent
        if sha256_file(path) != entry["sha256"] or verify_run_file_index(run):
            raise ValueError("recovered sampling artifact bytes changed")
        card = json.loads((run/"run_manifest.json").read_text())
        expected = {p.relative_to(config_root).as_posix(): sha256_file(p) for p, _ in plan.config_sources}
        snapshots = {Path(v["snapshot_path"]).relative_to("configs").as_posix(): v["sha256"]
            for v in card["config_snapshots"]}
        if (card["status"] != "complete" or card["entrypoint"]["name"] != plan.entrypoint_name
                or snapshots != expected):
            raise ValueError("recovered sampling configuration or status changed")
        value = json.loads(path.read_text())
        if key.endswith("_eval"):
            validate_pn2021_evaluation_result(value, result_path=path, prospective_only=True)
            train = completed[key.removesuffix("_eval")]
            if value["subject"]["train_result"] != {"path": train["result"], "sha256": train["sha256"]}:
                raise ValueError("recovered evaluation uses a different training artifact")
            execution = value["protocol"]["corruption_cache"]["execution"]
            if execution["mode"] != "online_torch_gpu_v1" or execution["metadata_verification"]["metadata_member_count"] != 10:
                raise ValueError("recovered evaluation omits repaired online metadata verification")
        else:
            if path.resolve() != (plan.run_dir/plan.expected_result_relative_path).resolve():
                raise ValueError("recovered training escaped its original output")
            validate_pn2021_train_result(value, result_path=path)
    return completed


def run_sampling(config, config_path, config_root, output):
    from util.run_record import verify_run_file_index
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("K500_SAMPLING_GPUS") != "0,1":
        raise ValueError("sampling coordinator must be CPU-only and explicitly authorize GPUs 0,1")
    members = sampling_plans(config,config_path,config_root)
    if config.get("recovery") == SAMPLING_NVME_RECOVERY and SAMPLING_ROOT.resolve() != SAMPLING_NVME_ROOT:
        raise ValueError("NVMe recovery requires the explicitly authorized verified storage binding")
    admission_path = (Path(config["recovery"]["engineering_admission"]) if "recovery" in config
        else SAMPLING_ROOT/"engineering_admission.json")
    admission = json.loads(admission_path.read_text())
    if admission.get("status") != "passed" or set(admission.get("jobs",{})) != set(config["admission_results"]):
        raise ValueError("four sampling engineering admissions have not passed")
    output.mkdir(parents=True,exist_ok=False)
    sources = {}
    for directory in ("core","models","data_preprocess","util","boot_scripts"):
        for source in sorted((REPO/directory).rglob("*.py")):
            if "tests" in source.relative_to(REPO).parts: continue
            rel = str(source.relative_to(REPO));dest=output/"source_snapshot"/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);sources[rel]=sha256_file(dest)
    if sources != admission["runtime_sources"]:
        raise ValueError("sampling runtime source differs from the engineering admission")
    for key,entry in admission["jobs"].items():
        if (entry["result"] != config["admission_results"][key]
                or sha256_file(entry["result"]) != entry["sha256"]
                or verify_run_file_index(Path(entry["result"]).parent.parent)):
            raise ValueError("sampling engineering artifact changed")
    atomic_json(output/"engineering_admission.json", admission)
    atomic_json(output/"source_identity.json",sources)
    atomic_json(output/"control.json",{"paused":False,"max_workers":2})
    config_hashes={p:sha256_file(p) for _,plan in members.values() for p,_ in plan.config_sources}
    config_hashes[config_path] = sha256_file(config_path)
    completed = recover_sampling_completed(config,members,config_root,sources) if "recovery" in config else {}
    atomic_json(output/"recovered_results.json", completed)
    active,idle = {},{}
    stopping=[False];failure=[None]
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *args:stopping.__setitem__(0,True))
    with exclusive_lock(SAMPLING_ROOT/"sampling.lock"):
        while True:
            if any(sha256_file(REPO/rel)!=digest for rel,digest in sources.items()) or any(sha256_file(p)!=h for p,h in config_hashes.items()):
                stopping[0]=True;failure[0]="frozen sampling source/config changed"
            for gpu,child in list(active.items()):
                if child["process"].poll() is None: continue
                child["log"].close();key=child["key"];plan=members[key][1]
                if child["process"].returncode or verify_run_file_index(plan.run_dir):
                    stopping[0]=True;failure[0]=f"failed child preserved without retry: {key}"
                else:
                    result=plan.run_dir/plan.expected_result_relative_path
                    completed[key]={"result":str(result),"sha256":sha256_file(result)}
                del active[gpu];idle.pop(gpu,None)
            if len(completed)==len(members):
                comparison=sampling_finalize(completed,output)
                result={"schema_version":6,"artifact_type":"pulse_training_queue_result","status":"complete",
                    "jobs":completed,"admission":{"sources":sources},"sampling_comparison":comparison}
                validate_sampling_result(result,output/"queue_result.json")
                atomic_json(output/"queue_result.json",result)
                atomic_json(output/"status.json",{"status":"complete","completed":completed});return
            snapshot=resource_snapshot();control=json.loads((output/"control.json").read_text())
            if set(control)!={"paused","max_workers"} or type(control["paused"]) is not bool or type(control["max_workers"]) is not int or not 0<=control["max_workers"]<=2:
                stopping[0]=True;failure[0]="sampling control exceeds two authorized GPUs"
                control={"paused":True,"max_workers":0}
            free_gib = shutil.disk_usage(output).free / 1024**3
            blockers = (["available_ram_below_64_GiB"] if snapshot["available_ram_gib"] < 64 else [])
            if snapshot["load1"] > snapshot["cpu_affinity_count"]*.8: blockers.append("host_load_above_limit")
            if free_gib < 25: blockers.append("disk_free_below_25_GiB")
            atomic_json(output/"status.json",{"status":"draining" if stopping[0] else "running","completed":completed,
                "active":{c["key"]:{"gpu":g,"pid":c["process"].pid} for g,c in active.items()},
                "resource_blockers":blockers,"disk_free_gib":free_gib,
                "time":time.time(),"failure":failure[0]})
            if stopping[0] and not active: raise RuntimeError(failure[0] or "sampling queue drained")
            safe_host=(snapshot["available_ram_gib"]>=64 and snapshot["load1"]<=snapshot["cpu_affinity_count"]*.8
                and free_gib>=25)
            for gpu in (0,1):
                device=snapshot["gpus"].get(gpu)
                safe=(safe_host and device is not None and not device["pids"] and free_device(device))
                idle[gpu]=(idle.get(gpu) or time.time()) if safe else None
                if stopping[0] or control["paused"] or gpu in active or len(active)>=control["max_workers"] or idle[gpu] is None or time.time()-idle[gpu]<30: continue
                active_keys={c["key"] for c in active.values()}
                key=next((k for k in members if k not in completed and k not in active_keys
                    and (not k.endswith("_eval") or k.removesuffix("_eval") in completed)),None)
                if key is None: continue
                experiment,plan=members[key]
                if plan.run_dir.exists(): raise FileExistsError(f"refusing existing sampling output: {plan.run_dir}")
                command=[sys.executable,str(REPO/"boot_scripts/run_experiment.py"),"--config",str(experiment),"--config-root",str(config_root)]
                dry=subprocess.run([*command,"--dry-run"],check=True,capture_output=True,text=True)
                atomic_json(output/f"{key}.dry_run.json",json.loads(dry.stdout))
                device=resource_snapshot()["gpus"][gpu]
                if device["pids"] or not free_device(device): idle[gpu]=None;continue
                env={**os.environ,"CUDA_VISIBLE_DEVICES":device["uuid"],"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
                    "OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","TMPDIR":str(SAMPLING_ROOT/"tmp")}
                log=(output/f"{key}.launcher.log").open("x")
                process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]={"key":key,"process":process,"log":log}
                atomic_json(output/f"{key}.launch.json",{"command":command,"gpu":gpu,"gpu_uuid":device["uuid"],"process_identity":process_identity(process.pid),"time":time.time()})
            time.sleep(15)


def grid():
    return {f"{arm}_s{seed}_{model}_{center}{suffix}": (arm, seed, model, center, suffix)
            for seed in SEEDS for center in CENTERS for model in MODELS
            for arm in ARMS for suffix in ("", "_eval")}


def plans(config, path, root):
    from boot_scripts.run_experiment import load_experiment_plan
    from core.methods.registry import load_recipe_spec
    base_keys = {"schema_version", "run_root", "references", "runtime"}
    if (set(config) not in (base_keys, base_keys | {"recovery"})
            or config["schema_version"] != 5 or Path(config["run_root"]) != ROOT
            or tuple(config["references"]) != tuple(grid()) or config["runtime"] != LIMITS):
        raise ValueError("joint SimCLR requires the frozen 2-arm/3-seed/2-model/4-center grid")
    if "recovery" in config and config["recovery"] not in (RECOVERY, ONLINE_RECOVERY):
        raise ValueError("recovery is restricted to the preserved first workflow and exact cache receipt")
    online = config.get("recovery") == ONLINE_RECOVERY
    result = {}
    for key, (arm, seed, model, center, suffix) in grid().items():
        exp = resolve_config_reference(config["references"][key], owner_config_path=path,
            config_root=root, description=key, must_exist=True)
        plan = load_experiment_plan(exp, config_root=root, allow_existing_run_dir=True)
        method = f"train/methods/{METHODS[arm]}.yaml"
        alias = "effnet" if model == "efficientnet1dv2" else model
        checkpoint = f"/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_{alias}_source_v1/training/checkpoints/best.pt"
        args = (["--model", model, "--train-result", str(ROOT/key.removesuffix("_eval")/"training/train_result.json"),
                 "--center", center, "--method-config", method] if suffix else
                ["--model", model, "--method-config", method, "--center", center, "--source-checkpoint", checkpoint])
        expected = root/(("eval/PN2021_online_gpu.yaml" if online else "eval/PN2021.yaml")
                         if suffix else f"train/simclr_joint/{arm}_s{seed}.yaml")
        if (plan.run_dir != ROOT/key or list(plan.entry_arguments) != args
                or plan.entrypoint_name != ("evaluate_pn2021" if suffix else "train_pn2021")
                or plan.entry_config_path != expected):
            raise ValueError(f"joint queue child escaped its identity: {key}")
        if not suffix:
            payload = load_yaml_mapping(expected, description="paired training")
            if payload["random_seed"] != {"stream_namespace": GROUP, "comparison_group": GROUP,
                    "replicate_id": seed, "deterministic_algorithms": False}:
                raise ValueError("training seeds are not the frozen independent identities")
            for name, profile in payload["training"]["model_profiles"].items():
                expected_budget = (25, 128, 5e-5) if name == "efficientnet1dv2" else (30, 64, 3e-5)
                if (profile["epochs"], profile["batch_size"], profile["learning_rate"]) != expected_budget:
                    raise ValueError("supervised budget changed")
                stage1 = 1024 if name == "efficientnet1dv2" else 256
                if profile.get("stage1_steps", 0) != (stage1 if arm == "two_stage" else 0):
                    raise ValueError("stage boundary/contrastive budget changed")
            if load_recipe_spec(root/method).requires_vae:
                raise ValueError("this comparison must not load a VAE")
        result[key] = (exp, plan)
    return result


def admission_source_delta(previous, current, *, recovery, online=False):
    """Recovery may change this scheduler; training/evaluation code stays frozen."""
    delta = {name: {"previous": previous.get(name), "current": current.get(name)}
             for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)}
    allowed = ONLINE_SOURCE_CHANGES if online else {"util/simclr_joint_queue.py"}
    if delta and (not recovery or not set(delta).issubset(allowed)):
        raise ValueError("engineering admission training/evaluation source changed")
    return delta


def recover_completed(members, config_root, sources, *, recovery_config=None):
    from util.run_record import verify_run_file_index
    recovery_config = RECOVERY if recovery_config is None else recovery_config
    online = recovery_config == ONLINE_RECOVERY
    previous = Path(recovery_config["previous_workflow"])
    manifest = json.loads((previous/"run_manifest.json").read_text())
    status = json.loads((previous/"training/status.json").read_text())
    if (manifest["status"] != "failed" or status["status"] not in {"failed", "draining"}
            or status.get("active")
            or verify_run_file_index(previous)):
        raise ValueError("recovery requires the intact stopped workflow")
    old_sources = json.loads((previous/"training/source_identity.json").read_text())
    admission_source_delta(old_sources, sources, recovery=True, online=online)
    completed = status["completed"]
    for key, entry in completed.items():
        if key not in members:
            raise ValueError("recovery result is outside the frozen grid")
        plan = members[key][1]
        result = plan.run_dir/plan.expected_result_relative_path
        if (Path(entry["result"]) != result or sha256_file(result) != entry["sha256"]
                or verify_run_file_index(plan.run_dir)):
            raise ValueError("recovery child bytes changed")
        card = json.loads((plan.run_dir/"run_manifest.json").read_text())
        if card["status"] != "complete" or card["entrypoint"]["name"] != plan.entrypoint_name:
            raise ValueError("recovery child is incomplete or has a different entrypoint")
        for member in card["config_snapshots"]:
            relative = Path(member["snapshot_path"]).relative_to("configs")
            if sha256_file(config_root/relative) != member["sha256"]:
                raise ValueError("recovery child configuration changed")
    return completed


def restored_cache_ready(receipt_path, cache_root):
    """A full-content restoration receipt gates evaluations, never training."""
    if not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text())
    expected = "8f416d86911e711dc3e359dd1284e36e473253bd28201d3b8eec1bb3a2972d6c"
    waveform = cache_root/"signals_100hz.npy"
    stat = waveform.stat()
    identity = {key: getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
    if (receipt.get("status") != "complete" or receipt.get("cache_dir") != str(cache_root)
            or receipt.get("waveform_sha256") != expected
            or receipt.get("ledger_sha256") != "5991ac2123e9fe8774df7cb4a311fd0f4332554e51a77ea2a29ac2ce05ad5f13"
            or receipt.get("verification_mode") != "full_content"
            or receipt.get("waveform_stat") != identity or stat.st_size != 63192000128):
        raise ValueError("restored corruption cache identity differs from the original ledger")
    return True


def finalize(completed, output):
    """Seeds are the repeat units; views/centers are not training repeats."""
    rows, paired, seed_ids, backends = [], {}, {}, set()
    for key, (arm, seed, model, center, suffix) in grid().items():
        if not suffix:
            continue
        entry = completed[key]
        value = json.loads(Path(entry["result"]).read_text())
        lineage = value["checkpoint"]["lineage"]
        protocol = value["protocol"]
        backends.add(protocol["corruption_cache"].get("execution", {}).get("mode", "historical_cached_numpy"))
        identity = {k: lineage[k] for k in ("adaptation_data", "source_checkpoint", "seed", "selection")}
        identity["protocol"] = protocol
        identity["evaluation"] = {k: value["clean"]["per_center"][center]["identity"][k]
            for k in ("hash_id_set_sha256", "hash_label_set_sha256", "reference_exclusion")}
        pair = (seed, model, center)
        if paired.setdefault(pair, identity) != identity:
            raise ValueError("unmatched source/K500/evaluation cohort/seed between arms")
        if (value["status"] != "complete" or protocol["mapping_hash"] != "555ec85d5b51"
                or protocol["logical_centers"] != [center]
                or protocol["corruption_cache"]["view_count"] != 20
                or lineage["adaptation_data"]["record_count"] != 500
                or lineage["method"]["recipe_id"] != METHODS[arm]
                or lineage["selection"] != {"policy": "last", "heldout_evaluation_used_for_selection": False}):
            raise ValueError("completed evaluation changed the frozen metric contract")
        seed_ids.setdefault((arm, model, center), set()).add(lineage["seed"]["effective_seed"])
        clean = value["clean"]["evaluated_center_mean"]["pn2021_drop_all_zero_refexcluded"]
        corrupt = value["corrupted"]["aggregates"]["depth23"]["evaluated_center_mean"]["pn2021c_drop_all_zero_corrupted_refexcluded"]
        train = json.loads(Path(completed[key.removesuffix("_eval")]["result"]).read_text())
        rows.append({"arm": arm, "seed": seed, "model": model, "center": center,
            "result": entry, "checkpoint_sha256": value["checkpoint"]["sha256"],
            "supervised_optimizer_steps": train["optimizer_steps"],
            "stage1_steps": (1024 if model == "efficientnet1dv2" else 256) if arm == "two_stage" else 0,
            **{f"{kind}_{metric}": data[metric] for kind, data in (("clean", clean), ("corrupt", corrupt))
               for metric in ("macro_auroc", "macro_auprc")}})
    if len(rows) != 48 or any(len(values) != 3 for values in seed_ids.values()) or len(backends) != 1:
        raise ValueError("all three distinct training seeds are required")
    summary, contrasts = {}, []
    for model in MODELS:
        summary[model] = {}
        for arm in ARMS:
            summary[model][arm] = {}
            for metric in METRICS:
                vals = [statistics.mean(r[metric] for r in rows if
                    (r["arm"], r["seed"], r["model"]) == (arm, seed, model)) for seed in SEEDS]
                summary[model][arm][metric] = {"mean": statistics.mean(vals),
                    "sample_sd": statistics.stdev(vals), "seed_means": vals}
        for metric in METRICS:
            differences = [100*(a-b) for a,b in zip(summary[model]["joint"][metric]["seed_means"],
                summary[model]["two_stage"][metric]["seed_means"])]
            sd = statistics.stdev(differences)
            from scipy.stats import t
            mean = statistics.mean(differences)
            half = float(t.ppf(.975, 2)) * sd / 3**.5
            contrasts.append({"model": model, "metric": metric, "joint_minus_two_stage_pp": mean,
                "paired_seed_delta_pp": differences, "sample_sd_pp": sd,
                "exploratory_t95_ci_pp": [mean-half, mean+half],
                "per_center_delta_pp": {c: statistics.mean(
                    next(r[metric] for r in rows if (r["arm"],r["seed"],r["model"],r["center"]) == ("joint",s,model,c))
                    - next(r[metric] for r in rows if (r["arm"],r["seed"],r["model"],r["center"]) == ("two_stage",s,model,c))
                    for s in SEEDS)*100 for c in CENTERS}})
    practical = {m: all(next(c["joint_minus_two_stage_pp"] for c in contrasts
        if c["model"] == m and c["metric"] == metric) >= -1.0
        for metric in ("clean_macro_auprc", "corrupt_macro_auprc")) for m in MODELS}
    report = {"status": "complete", "evidence": "frozen_three_seed_development_not_paper_final",
        "metric_definition": "macro_AUROC_and_sklearn_average_precision_raw_logits_drop_all_zero",
        "aggregation": "equal_20_corruption_views_then_equal_four_centers_then_seed_mean",
        "corruption_execution_backend": next(iter(backends)),
        "rows": rows, "seed_summary": summary, "paired_contrasts": contrasts,
        "practical_retention_within_1pp_clean_and_corrupt_AP": practical,
        "caveats": ["total_compute_and_contrastive_exposure_not_matched",
            "same_supervised_step_budget_new_joint_objective_and_optimizer_not_loss_only_causal",
            "three_seed_intervals_exploratory_fixed_development_centers_previously_tuned"]}
    if "online_torch_gpu_v1" in backends:
        report["caveats"].append("GPU_online_Torch_corruption_has_new_RNG_backend_not_byte_identical_to_retired_NumPy_cache")
    atomic_json(output/"comparison.json", report)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["arm","seed","model","center",*METRICS])
    writer.writeheader()
    writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    (output/"metrics.csv").write_text(stream.getvalue())
    return {"metrics": str(output/"comparison.json"), "sha256": sha256_file(output/"comparison.json")}


def validate_result(value, path):
    if (value.get("schema_version") != 5 or value.get("status") != "complete"
            or value.get("artifact_type") != "pulse_training_queue_result"
            or set(value.get("jobs", {})) != set(grid())):
        raise ValueError("incomplete joint SimCLR workflow")
    for entry in value["jobs"].values():
        member = Path(entry["result"])
        member.resolve().relative_to(ROOT)
        if sha256_file(member) != entry["sha256"]:
            raise ValueError("joint SimCLR result hash changed")
    comparison = value["comparison"]
    if (Path(comparison["metrics"]) != path.parent/"comparison.json"
            or sha256_file(comparison["metrics"]) != comparison["sha256"]):
        raise ValueError("joint SimCLR comparison changed")


def run(config, config_path, config_root, output):
    from util.run_record import verify_run_file_index
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("SIMCLR_JOINT_GPU") != "2":
        raise ValueError("CPU-only coordinator requires explicit GPU 2 authorization")
    members = plans(config, config_path, config_root)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for directory in ("core", "models", "data_preprocess", "util", "boot_scripts"):
        for source in sorted((REPO/directory).rglob("*.py")):
            if "tests" in source.relative_to(REPO).parts:
                continue
            relative = str(source.relative_to(REPO))
            dest = output/"source_snapshot"/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            sources[relative] = sha256_file(dest)
    atomic_json(output/"source_identity.json", sources)
    admission_path = ROOT/"engineering_admission.json"
    admission = json.loads(admission_path.read_text())
    recovery = "recovery" in config
    online = config.get("recovery") == ONLINE_RECOVERY
    source_delta = admission_source_delta(admission["runtime_sources"], sources, recovery=recovery, online=online)
    if online:
        evidence = json.loads((ROOT/"online_gpu_admission.json").read_text())
        if (evidence.get("status") != "passed" or evidence.get("runtime_sources") != sources
                or evidence.get("existing_training_functions_unchanged") is not True
                or set(evidence.get("gpu_model_checks", {})) != set(MODELS)):
            raise ValueError("frozen GPU-online validation and unchanged training functions are required")
        atomic_json(output/"online_gpu_admission.json", evidence)
    if (admission.get("status") != "passed"
            or set(admission.get("jobs", {})) != {f"{arm}_{model}" for arm in ARMS for model in MODELS}):
        raise ValueError("four real-model GPU engineering admissions with unchanged source are required")
    for entry in admission["jobs"].values():
        member = Path(entry["result"])
        if sha256_file(member) != entry["sha256"] or verify_run_file_index(member.parent.parent):
            raise ValueError("engineering admission artifact changed")
    atomic_json(output/"engineering_admission.json", admission)
    atomic_json(output/"admission_source_compatibility.json", {"source_delta": source_delta,
        "training_functions_unchanged": True, "evaluation_backend_changed_to_gpu_online": online})
    atomic_json(output/"control.json", {"paused": False, "max_workers": 1})
    completed = recover_completed(members, config_root, sources, recovery_config=config["recovery"]) if recovery else {}
    if recovery:
        atomic_json(output/"recovered_results.json", completed)
    active, idle_since = None, None
    order = sorted(members, key=lambda key: key.endswith("_eval")) if recovery and not online else list(members)
    cache_config = load_yaml_mapping(config_root/"augmentation/cache.yaml", description="restored cache")
    cache_root = Path(cache_config["output"]["cache_dir"])
    stopping = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *args: stopping.__setitem__(0, True))
    with exclusive_lock(ROOT/"simclr_joint.lock"):
        while True:
            if any(sha256_file(REPO/name) != expected for name,expected in sources.items()):
                raise RuntimeError("frozen execution source changed")
            if active is not None and active["process"].poll() is not None:
                active["log"].close()
                key = active["key"]
                plan = members[key][1]
                code = active["process"].returncode
                if code != 0 or verify_run_file_index(plan.run_dir):
                    atomic_json(output/"status.json", {"status":"failed", "job":key,
                        "returncode":code, "completed":completed, "time":time.time()})
                    raise RuntimeError(f"child failed; preserved without overwrite or automatic retry: {key}")
                member = plan.run_dir/plan.expected_result_relative_path
                completed[key] = {"result":str(member), "sha256":sha256_file(member)}
                active, idle_since = None, None
            if len(completed) == len(members):
                comparison = finalize(completed, output)
                result = {"schema_version":5, "artifact_type":"pulse_training_queue_result",
                    "status":"complete", "jobs":completed, "admission":{"sources":sources},
                    "comparison":comparison}
                validate_result(result, output/"queue_result.json")
                atomic_json(output/"queue_result.json", result)
                atomic_json(output/"status.json", {"status":"complete", "completed":completed})
                return
            control = json.loads((output/"control.json").read_text())
            if (set(control) != {"paused", "max_workers"} or type(control["paused"]) is not bool
                    or type(control["max_workers"]) is not int or not 0 <= control["max_workers"] <= 1):
                raise ValueError("workflow control exceeds one authorized GPU")
            snapshot = resource_snapshot()
            atomic_json(output/"status.json", {"status":"draining" if stopping[0] else "running",
                "completed":completed, "active":({active["key"]:{"pid":active["process"].pid,"gpu":2}}
                    if active else {}), "resources":snapshot, "time":time.time()})
            if stopping[0] and active is None:
                raise InterruptedError("workflow drained; all existing results preserved")
            if active is None and not stopping[0] and not control["paused"] and control["max_workers"] == 1:
                device = snapshot["gpus"].get(2)
                safe = (device is not None and not device["pids"] and free_device(device)
                    and snapshot["available_ram_gib"] >= 64
                    and snapshot["load1"] <= snapshot["cpu_affinity_count"]*.8
                    and shutil.disk_usage(output).free >= 40*1024**3)
                idle_since = (idle_since or time.time()) if safe else None
                if idle_since is not None and time.time()-idle_since >= 30:
                    key = next(k for k in order if k not in completed)
                    if recovery and not online and key.endswith("_eval") and not restored_cache_ready(
                            Path(RECOVERY["cache_receipt"]), cache_root):
                        atomic_json(output/"status.json", {"status": "waiting_for_corruption_cache",
                            "completed": completed, "active": {}, "time": time.time()})
                        time.sleep(30)
                        continue
                    exp, plan = members[key]
                    if plan.run_dir.exists():
                        raise FileExistsError(f"refusing existing output: {plan.run_dir}")
                    command = [sys.executable,str(REPO/"boot_scripts/run_experiment.py"),
                        "--config",str(exp),"--config-root",str(config_root)]
                    dry = subprocess.run([*command,"--dry-run"],check=True,capture_output=True,text=True)
                    atomic_json(output/f"{key}.dry_run.json",json.loads(dry.stdout))
                    device = resource_snapshot()["gpus"][2]
                    if device["pids"] or not free_device(device):
                        idle_since = None
                        continue
                    env = {**os.environ,"CUDA_VISIBLE_DEVICES":device["uuid"],
                        "CUDA_DEVICE_ORDER":"PCI_BUS_ID","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                        "TMPDIR":str(ROOT/"tmp")}
                    (ROOT/"tmp").mkdir(exist_ok=True)
                    log = (output/f"{key}.launcher.log").open("x")
                    process = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active = {"key":key,"process":process,"log":log}
                    atomic_json(output/f"{key}.launch.json", {"command":command,"gpu_uuid":device["uuid"],
                        "process_identity":process_identity(process.pid),"time":time.time()})
            time.sleep(30)
