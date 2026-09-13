"""Fixed 512/center visual-only ablation; each arm recomputes its trained CLIP.

Center jobs share the dual-workflow four-device budget. No heldout checkpoint
selection, cached trainable features, image files, or full model copies.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.evaluation.ecg_image_queue import TaskQueue, atomic_json, digest_json
from util.pn2021_artifact_contract import sha256_file
from util.pulse_training_contract import CENTERS, CLASS_ORDER, validate_result as validate_training

REPO = Path(__file__).resolve().parents[2]
ARMS = ("clean", "single", "three")
NAMES = ("PULSE-original", "visual-clean", "visual-single-JSD", "visual-three-JSD")
PARENT_SHA = "ad0be13c245b6fc62671bd79a64bda7a9aedc83a7a2aebbdb31b78e56bf9067d"


def validate_config(config):
    if (set(config) != {"schema_version", "references", "center", "records_per_center", "source_state", "training_results"}
            or config["schema_version"] != 3 or config["center"] not in CENTERS
            or config["records_per_center"] != 512 or set(config["references"]) != {"paired_config"}
            or tuple(config["training_results"]) != ARMS):
        raise ValueError("visual subset requires fixed center, cohort, and clean/single/three arms")
    for raw in [config["source_state"], *config["training_results"].values()]:
        Path(raw).resolve().relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))


def matched_evidence(config):
    from util.run_record import verify_run_file_index
    evidence, common, records = {}, None, None
    ignored = {"width", "jsd", "jsd_weight", "images_per_record_exposure"}
    for arm, raw in config["training_results"].items():
        path = Path(raw)
        result = json.loads(path.read_text())
        validate_training(result, path)
        protocol = result["protocol"]
        if (result["mode"] != "train" or protocol["center"] != config["center"]
                or protocol.get("training_scope") != "clip_last4_qv_lora8_projector_frozen_llm"
                or protocol["width"] != {"clean": 0, "single": 1, "three": 3}[arm]
                or verify_run_file_index(path.parent.parent)):
            raise ValueError("visual subset training lineage mismatch")
        value = {k:v for k,v in protocol.items() if k not in ignored}
        common = value if common is None else common
        if value != common:
            raise ValueError("visual arms differ outside the declared augmentation/JSD control")
        for name, sha in protocol["implementation_sha256"].items():
            if sha256_file(REPO/name) != sha:
                raise ValueError("training source changed before visual evaluation")
        selected = json.loads((path.parent/"k500_records.json").read_text())
        records = selected if records is None else records
        if selected != records:
            raise ValueError("visual arms adapted on different records")
        evidence[arm] = {"path": path, "result": result, "sha256": sha256_file(path)}
    return evidence, records


def checked_parent(config):
    """Read only the frozen cohort and original-model tasks, without editing parent state."""
    source = Path(config["source_state"])
    result_path = source/"subset_result.json"
    if sha256_file(result_path) != PARENT_SHA:
        raise ValueError("completed original-model subset result changed")
    result = json.loads(result_path.read_text())
    def member(relative):
        path = (source/relative).resolve(); path.relative_to(source.resolve())
        if sha256_file(path) != result["files"][relative]:
            raise ValueError("frozen original-model evidence file changed")
        return json.loads(path.read_text())
    parent = member("prepared.json")
    audit = parent["audit"]
    if (audit["mapping_hash"] != "555ec85d5b51" or audit["class_order"] != list(CLASS_ORDER)
            or audit["metric_view"] != "drop_all_zero" or audit["sampling_uses_labels"]):
        raise ValueError("frozen subset input contract mismatch")
    for name, sha in audit["common_implementation_sha256"].items():
        if sha256_file(REPO/name) != sha:
            raise ValueError("frozen corruption/render/data implementation changed")
    for name in ("util/evaluation/ecg_image_data.py", "util/evaluation/ecg_image_artifact.py"):
        if sha256_file(REPO/name) != parent["sources"][name]:
            raise ValueError("frozen waveform loading or answer parser changed")
    rows = [r for r in parent["rows"] if r["logical_center"] == config["center"]]
    if len(rows) != 512 or len({r["sample_key"] for r in rows}) != 512:
        raise ValueError("visual subset is not the fixed 512-record cohort")
    tasks = [t for t in parent["base_tasks"] if t["center"] == config["center"]]
    if (len(tasks) != 64 or {k for t in tasks for b in t["batches"] for k in b} != {r["sample_key"] for r in rows}
            or any(len(b) != 2 for t in tasks for b in t["batches"])):
        raise ValueError("original batch2 task topology changed")
    originals = [r for t in tasks for r in member(f"base_queue/results/{t['id']}.json")["predictions"]]
    admissions = [member(name) for name in result["files"] if name.startswith("workers/") and name.endswith(".admission.json")]
    bases = [a for a in admissions if a["kind"] == "base" and a["status"] == "passed"]
    if not bases:
        raise ValueError("original-model author-generation admission missing")
    generation = {k:bases[0]["audit"][k] for k in ("prompt", "generation")}
    if any({k:a["audit"][k] for k in generation} != generation for a in bases):
        raise ValueError("original generation admissions disagree")
    return parent, rows, tasks, originals, generation


def make_backend(evidence, reservation):
    import torch
    from util.evaluation.pulse_adapters import PairedPulseBackend, pack_shared_prompt_features, decode_generated
    class VisualBackend(PairedPulseBackend):
        def __init__(self):
            # Reuse source loading, checkpoint verification, prompt, and generation.
            # The legacy initializer reads its first protocol under the name w1.
            super().__init__({"w1": evidence["clean"]}, reservation=reservation)
            self.set_pair(evidence)

        @torch.inference_mode()
        def generate_pair(self, pixels, *, verify_author=False):
            if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (5,3,336,336):
                raise ValueError("visual PULSE expects Bx5x3x336x336")
            pixels = pixels.to(device="cuda", dtype=torch.float16)
            ids = self.prompt_ids.expand(len(pixels),-1)
            outputs, self.last_token_ids = {}, {}
            for arm in ARMS:
                self._select(arm)
                # CLIP is different for every arm; never share encoded features.
                with torch.autocast("cuda", dtype=torch.float16):
                    features = self.base.get_vision_tower()(pixels.flatten(0,1))
                    packed = pack_shared_prompt_features(self.base,ids,features)
                    generated = self._generate(packed)
                    if verify_author:
                        author = self.base.prepare_inputs_labels_for_multimodal(ids,None,None,None,None,
                            pixels,image_sizes=[(2200,1700)]*len(pixels))[4]
                        reference = self.base.generate(ids,images=pixels,image_sizes=[(2200,1700)]*len(pixels),**self.generation_kwargs)
                        if not torch.equal(author,packed) or not torch.equal(reference,generated):
                            raise ValueError("visual arm differs from author packing or generated tokens")
                        self.admissions[f"{arm}_batch{len(pixels)}"] = {"packing_exact":True,"generated_token_ids_exact":True}
                    self.last_token_ids[arm] = generated.cpu()
                    outputs[arm] = decode_generated(self.tokenizer,generated.cpu(),max_new_tokens=32)
            return outputs
    return VisualBackend()


def validate_row(row, sample, condition):
    from util.pulse_benchmark_contract import validate_prediction_row
    if set(row["arms"]) != set(ARMS) or row["condition_index"] != condition["condition_index"]:
        raise ValueError("visual prediction arm or condition-index mismatch")
    validate_prediction_row(row, sample, condition, answers=(row["arms"][arm] for arm in ARMS))


def infer(samples,conditions,config,baseline,backend,renderer,profile,pool,*,verify=False):
    import numpy as np
    import torch
    from util.evaluation.ecg_image_data import native500_waveform, corrupt_native500_one
    from util.evaluation.ecg_image_artifact import parse_response
    waves = list(pool.map(lambda r:native500_waveform(r,Path(config["paths"]["raw_root"]))[0],samples))
    hashes = [hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest() for w in waves]
    clean = torch.from_numpy(np.stack(waves)).cuda()
    rows=[]
    for condition in conditions:
        views=torch.cat([corrupt_native500_one(clean[i:i+1],source_hash=s["hash_id"],condition=condition,
            profile=profile,base_seed=baseline["corruption"]["seed_config"]["base_seed"]) for i,s in enumerate(samples)])
        fingerprints=[hashlib.sha256(v.tobytes()).hexdigest() for v in views.cpu().numpy()]
        pixels=renderer.preprocess_for_pulse(renderer.render(views)).clone()
        answers=backend.generate_pair(pixels,verify_author=verify)
        for i,sample in enumerate(samples):
            row={"schema_version":1,**condition,**{k:sample[k] for k in ("sample_key","record_id","hash_id","logical_center")},
                "true_labels":sample["label_names"],"clean_waveform_sha256":hashes[i],"input_waveform_sha256":fingerprints[i],
                "arms":{arm:{**answers[arm][i],**parse_response(answers[arm][i]["response"])} for arm in ARMS}}
            validate_row(row,sample,condition); rows.append(row)
    return rows


def comparison_arrays(prepared,predictions):
    import numpy as np
    from util.evaluation.pulse_subset import validate_original_row
    samples={r["sample_key"]:r for r in prepared["rows"]}
    positions={key:i for i,key in enumerate(samples)}
    conditions={c["condition_id"]:c for c in prepared["conditions"]}
    if len(samples)!=512 or len(conditions)!=21 or len(prepared["original_predictions"])!=512*21:
        raise ValueError("incomplete visual comparison inputs")
    originals={}
    for row in prepared["original_predictions"]:
        validate_original_row(row,samples[row["sample_key"]],conditions[row["condition_id"]])
        key=(row["sample_key"],row["condition_id"])
        if key in originals: raise ValueError("duplicate original prediction")
        originals[key]=row
    truth=np.array([r["label"] for r in samples.values()],dtype=np.uint8)
    values=np.zeros((4,512,21,5),dtype=np.uint8)
    seen=set()
    quality={name:{"predictions":0,"parse_invalid":0,"truncated":0} for name in NAMES}
    for row in predictions:
        key=(row["sample_key"],row["condition_id"])
        if key in seen or key not in originals: raise ValueError("duplicate or foreign visual prediction")
        seen.add(key); original=originals[key]
        validate_row(row,samples[key[0]],conditions[key[1]])
        if any(row[k]!=original[k] for k in ("clean_waveform_sha256","input_waveform_sha256")):
            raise ValueError("original and adapted model input fingerprints differ")
        for a,answer in enumerate((original["answer"],*[row["arms"][arm] for arm in ARMS])):
            values[a,positions[key[0]],row["condition_index"]]=[label in answer["predicted_labels"] for label in CLASS_ORDER]
            q=quality[NAMES[a]]; q["predictions"]+=1; q["parse_invalid"]+=not answer["parse_valid"]; q["truncated"]+=answer["hit_max_new_tokens"]
    if len(seen)!=512*21: raise ValueError("incomplete visual comparison grid")
    return truth,values,quality


def run(path,root,output):
    import torch
    from util.evaluation.pulse_benchmark import claim_gpu,make_renderer,profile_and_baseline,source_identity,snapshot_sources
    config=load_yaml_mapping(path,description="visual subset"); validate_config(config)
    paired_path=resolve_config_reference(config["references"]["paired_config"],owner_config_path=path,
        config_root=root,description="frozen inference inputs",must_exist=True)
    paired=load_yaml_mapping(paired_path,description="frozen paired settings")
    paired["paths"]["temporary_root"]="/home/linbinhao/ECG_adv_data/tmp/dual_jsd_20260911"
    evidence,k500=matched_evidence(config)
    parent,rows,tasks,originals,generation=checked_parent(config)
    for key in ("hash_id","source_record"):
        if {r[key] for r in rows}&{r[key] for r in k500}: raise ValueError("visual K500 leakage")
    if any(not any(r["label"]) for r in rows): raise ValueError("visual metric view must drop all-zero labels")
    sources={**source_identity(),"util/evaluation/pulse_visual_subset.py":sha256_file(Path(__file__))}
    lineage={arm:{"path":str(item["path"]),"sha256":item["sha256"]} for arm,item in evidence.items()}
    identity=digest_json({"config":config,"sources":sources,"training":lineage,"parent":PARENT_SHA})
    tasks=[{**t,"protocol_identity":identity,"id":digest_json({"parent_task":t["id"],"protocol":identity})} for t in tasks]
    prepared={"config":config,"rows":rows,"conditions":parent["conditions"],"tasks":tasks,"original_predictions":originals,
        "sources":sources,"training":lineage,"subset_identity":identity,"parent_result_sha256":PARENT_SHA,
        "mapping_hash":"555ec85d5b51","metric_view":"drop_all_zero","k500_overlap":0}
    output.mkdir(parents=True,exist_ok=False); snapshot_sources(output,sources)
    atomic_json(output/"prepared.json",prepared)
    queue=TaskQueue(output/"queue",tasks)
    reservation=claim_gpu(paired); started=time.time(); stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *a:stop.__setitem__(0,True))
    backend=make_backend(evidence,reservation)
    if {"prompt":backend.prompt,"generation":backend.generation_kwargs}!=generation:
        raise ValueError("visual evaluation changed original prompt or generation")
    renderer,rendering=make_renderer(paired); profile,baseline=profile_and_baseline(paired,paired_path,root)
    if parent["conditions"]!=baseline["corruption"]["conditions"]: raise ValueError("corruption grid changed")
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool:
        checks=infer(k500[:2],prepared["conditions"][:1],paired,baseline,backend,renderer,profile,pool,verify=True)
        single=infer(k500[:1],prepared["conditions"][:1],paired,baseline,backend,renderer,profile,pool,verify=True)
        before_tokens={arm:value.clone() for arm,value in backend.last_token_ids.items()}
        repeated=infer(k500[:1],prepared["conditions"][:1],paired,baseline,backend,renderer,profile,pool)
        if single!=repeated or any(not torch.equal(before_tokens[a],backend.last_token_ids[a]) for a in ARMS):
            raise ValueError("visual adapter switch is not deterministic")
        atomic_json(output/"admission.json",{"status":"passed","checks":backend.admissions,"repeated_switch_exact":True,
            "independent_clip_per_arm":True,"engineering_predictions":checks,"rendering":rendering})
        by_key={r["sample_key"]:r for r in rows}
        while not stop[0]:
            if any(sha256_file(REPO/n)!=sha for n,sha in sources.items()): raise ValueError("visual evaluation source changed")
            lease=queue.claim(f"visual-{os.getpid()}")
            if lease is None: break
            with lease:
                before=time.time(); predictions=[]
                for batch in lease.task["batches"]:
                    predictions.extend(infer([by_key[k] for k in batch],prepared["conditions"],paired,baseline,backend,renderer,profile,pool))
                lease.commit(predictions,{"seconds":time.time()-before,"model_predictions":3*len(predictions)})
                atomic_json(output/"progress.json",{"pid":os.getpid(),"counts":queue.counts(),"elapsed_seconds":time.time()-started})
    if queue.counts()["completed_tasks"]!=len(tasks): raise InterruptedError("visual evaluation drained; completed tasks preserved")
    predictions=[r for p in queue.completed_payloads() for r in p["predictions"]]
    truth,values,quality=comparison_arrays(prepared,predictions)
    atomic_json(output/"arrays.json",{"truth":truth.tolist(),"predictions":values.tolist()})
    atomic_json(output/"quality.json",quality)
    atomic_json(output/"runtime.json",{"seconds":time.time()-started,"gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"augmented_images_persisted":0})
    result={"artifact_type":"pulse_subset_result","schema_version":2,"status":"complete","center":config["center"],
        "models":list(NAMES),"records_per_center":512,"subset_identity":identity,
        "files":{p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*") if p.is_file() and not p.name.endswith(".lock")}}
    validate_result(result,output/"subset_result.json"); atomic_json(output/"subset_result.json",result)


def validate_result(result,path):
    if (result.get("artifact_type")!="pulse_subset_result" or result.get("schema_version")!=2
            or result.get("status")!="complete" or result.get("models")!=list(NAMES)
            or result.get("center") not in CENTERS or result.get("records_per_center")!=512):
        raise ValueError("invalid visual center result")
    if not {"prepared.json","arrays.json","quality.json","admission.json","runtime.json"}<=result["files"].keys():
        raise ValueError("visual center evidence missing")
    for relative,sha in result["files"].items():
        member=(path.parent/relative).resolve(); member.relative_to(path.parent.resolve())
        if sha256_file(member)!=sha: raise ValueError("visual result fingerprint changed")
    prepared=json.loads((path.parent/"prepared.json").read_text())
    if (prepared["subset_identity"]!=result["subset_identity"] or prepared["config"]["center"]!=result["center"]
            or prepared["k500_overlap"]!=0 or prepared["metric_view"]!="drop_all_zero"):
        raise ValueError("visual result protocol mismatch")
    queue=TaskQueue(path.parent/"queue",prepared["tasks"])
    truth,values,quality=comparison_arrays(prepared,[r for p in queue.completed_payloads() for r in p["predictions"]])
    if json.loads((path.parent/"arrays.json").read_text())!={"truth":truth.tolist(),"predictions":values.tolist()}:
        raise ValueError("visual prediction arrays differ from committed tasks")
    if json.loads((path.parent/"quality.json").read_text())!=quality: raise ValueError("visual quality audit mismatch")
    admission=json.loads((path.parent/"admission.json").read_text())
    if (admission["status"]!="passed" or not admission["repeated_switch_exact"] or not admission["independent_clip_per_arm"]
            or set(admission["checks"])!={f"{a}_batch{b}" for a in ARMS for b in (1,2)}
            or any(v!={"packing_exact":True,"generated_token_ids_exact":True} for v in admission["checks"].values())):
        raise ValueError("visual author-generation admission incomplete")


def finalize_comparison(paths,output):
    import numpy as np
    from sklearn.metrics import f1_score,accuracy_score
    from util.evaluation.ecg_image_metrics import paired_metrics,paired_bootstrap
    from util.evaluation.pulse_benchmark_metrics import write_csv
    if tuple(paths)!=CENTERS: raise ValueError("visual aggregate requires four ordered centers")
    truth,predictions,lineage={},{},{}
    for center,path in paths.items():
        path=Path(path); result=json.loads(path.read_text()); validate_result(result,path)
        if result["center"]!=center: raise ValueError("visual aggregate center mismatch")
        arrays=json.loads((path.parent/"arrays.json").read_text())
        truth[center]=np.array(arrays["truth"],dtype=np.uint8); predictions[center]=np.array(arrays["predictions"],dtype=np.uint8)
        prepared=json.loads((path.parent/"prepared.json").read_text())
        lineage[center]={"result":str(path),"sha256":sha256_file(path)}
    conditions=[c["condition_id"] for c in prepared["conditions"]]
    metrics=paired_metrics(truth,predictions,list(NAMES),conditions)
    for row in metrics["condition_rows"]:
        y=truth[row["center"]]; p=predictions[row["center"]][NAMES.index(row["model"]),:,conditions.index(row["condition"])]
        if abs(f1_score(y,p,average="macro",zero_division=0)-row["macro_f1"])>1e-12 or abs(accuracy_score(y,p)-row["exact_match_accuracy"])>1e-12:
            raise ValueError("visual metrics differ from independent sklearn")
    metrics["evidence"]={"lineage":lineage,"single_seed_development_only":True,"previously_observed_cohort":True,
        "matched_clean_supervised_control":True,"equal_optimizer_steps_not_equal_forward_compute":True,
        "k500_overlap":0,"parent_result_sha256":PARENT_SHA}
    atomic_json(output/"pulse_metrics.json",metrics)
    for a,b in ((1,2),(1,3),(2,3)):
        atomic_json(output/f"pulse_bootstrap_{a}_{b}.json",paired_bootstrap(truth,{c:p[[a,b]] for c,p in predictions.items()},
            [NAMES[a],NAMES[b]],repeats=1000,seed=20260909))
    write_csv(output/"pulse_per_condition.csv",metrics["condition_rows"])
    return {"metrics":str(output/"pulse_metrics.json"),"sha256":sha256_file(output/"pulse_metrics.json")}
