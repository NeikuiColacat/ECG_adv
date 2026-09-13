"""Finite width-only JSD extension; shares the managed queue, never PULSE code."""
from __future__ import annotations

import json
from pathlib import Path

from util.config_bundle import load_yaml_mapping, resolve_config_reference
from util.pn2021_artifact_contract import CLASS_ORDER, MAPPING_HASH, sha256_file
from util.pulse_training_contract import CENTERS

KEYS = tuple(f"founder_jsd_w{w}_{c}{s}" for c in CENTERS for w in (1, 3) for s in ("", "_eval"))
SOURCE = "/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_ecgfounder_source_v1/training/checkpoints/best.pt"


def plans(config, config_path, config_root):
    if config["schema_version"] == 5:
        return repeat_plans(config, config_path, config_root)
    from boot_scripts.run_experiment import load_experiment_plan
    if (set(config) != {"schema_version", "run_root", "references", "runtime", "predecessor", "width2_baselines"}
            or config["schema_version"] != 4 or tuple(config["references"]) != (*KEYS, "width2_method")
            or config["references"]["width2_method"] != "train/methods/augmix_clean_bce_jsd_lhat.yaml"):
        raise ValueError("width JSD requires the exact four-center width1/3 grid")
    if config["runtime"] != {"max_workers": 4, "poll_seconds": 30, "idle_seconds": 30,
            "min_available_ram_gib": 64, "max_load_fraction": .8, "min_disk_free_gib": 40}:
        raise ValueError("width JSD resource bounds changed")
    root = Path(config["run_root"]).resolve()
    root.relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))
    baseline_method = load_yaml_mapping(config_root/"train/methods/augmix_clean_bce_jsd_lhat.yaml", description="baseline method")
    baseline_augmix = load_yaml_mapping(config_root/"train/augmix.yaml", description="baseline AugMix")
    result = {}
    for c in CENTERS:
        for w in (1, 3):
            method_path = f"train/methods/augmix_clean_bce_jsd_width{w}_lhat.yaml"
            method = load_yaml_mapping(config_root/method_path, description="width method")
            augmix_path = method["resources"]["augmix_config"]["path"]
            method["resources"]["augmix_config"]["path"] = "train/augmix.yaml"
            augmix = load_yaml_mapping(config_root/augmix_path, description="width AugMix")
            if augmix["method"]["name"] != "jsd_width_augmix" or augmix["stage1_twochain"]["width"] != w:
                raise ValueError("wrong internal AugMix width")
            augmix["method"]["name"] = baseline_augmix["method"]["name"]
            augmix["stage1_twochain"]["width"] = 2
            if method != baseline_method or augmix != baseline_augmix:
                raise ValueError("width arm changed more than internal width and resource identity")
            for suffix in ("", "_eval"):
                key = f"founder_jsd_w{w}_{c}{suffix}"
                path = resolve_config_reference(config["references"][key], owner_config_path=config_path,
                    config_root=config_root, description=key, must_exist=True)
                plan = load_experiment_plan(path, config_root=config_root, allow_existing_run_dir=True)
                arguments = (["--model", "ecgfounder", "--train-result", str(root/f"founder_jsd_w{w}_{c}"/"training/train_result.json"),
                    "--center", c, "--method-config", method_path] if suffix else
                    ["--model", "ecgfounder", "--method-config", method_path, "--center", c, "--source-checkpoint", SOURCE])
                if (plan.run_dir != root/key or list(plan.entry_arguments) != arguments
                        or plan.entrypoint_name != ("evaluate_pn2021" if suffix else "train_pn2021")
                        or plan.entry_config_path != config_root/("eval/PN2021.yaml" if suffix else "train/PN2021.yaml")):
                    raise ValueError("width job escaped its matched identity")
                result[key] = (path, plan)
    if set(config["width2_baselines"]) != set(CENTERS):
        raise ValueError("four frozen width2 center baselines required")
    return {k: result[k] for k in KEYS}


def predecessor_ready(config):
    """Do not compete with the existing scheduler or launch over its failed jobs."""
    if config.get("schema_version") == 5:
        return True
    from util.evaluation.ecg_image_elastic import process_identity
    previous = config["predecessor"]
    root = Path(previous["run_dir"]).resolve()
    root.relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))
    manifest = json.loads((root/"run_manifest.json").read_text())
    if manifest["status"] not in {"running", "complete"}:
        raise RuntimeError("predecessor failed or stopped; width queue preserved without GPU launch")
    if any(process_identity(previous[k]) is not None for k in ("launcher_pid", "delegate_pid")):
        return False
    if manifest["status"] != "complete":
        raise RuntimeError("predecessor exited without completed managed manifest")
    status = json.loads((root/"training/status.json").read_text())
    if status["status"] != "complete" or status.get("active"):
        raise RuntimeError("predecessor queue did not finish all children")
    return True


def admission(config):
    if config["schema_version"] == 5:
        return  # Fresh paired runs; no historical checkpoint is reused.
    from util.run_record import verify_run_file_index
    config_root = Path(__file__).resolve().parents[1]/"configs"
    for c, entry in config["width2_baselines"].items():
        p = Path(entry["result"])
        if sha256_file(p) != entry["sha256"] or verify_run_file_index(p.parent.parent):
            raise ValueError("frozen width2 managed evidence changed")
        value = json.loads(p.read_text())
        protocol = value["protocol"]
        lineage = value["checkpoint"]["lineage"]
        # Preserve the full baseline config closure, including data and RNG files.
        for snapshot in (p.parent.parent/"configs").rglob("*"):
            if snapshot.is_file():
                current = config_root/snapshot.relative_to(p.parent.parent/"configs")
                if not current.is_file() or sha256_file(current) != sha256_file(snapshot):
                    raise ValueError(f"baseline config closure drifted: {current}")
        if (value["status"] != "complete" or protocol["mapping_hash"] != MAPPING_HASH
                or protocol["class_order"] != CLASS_ORDER or protocol["logical_centers"] != [c]
                or protocol["corruption_cache"]["view_count"] != 20
                or lineage["method"]["recipe_id"] != "augmix_clean_bce_jsd_lhat"
                or lineage["adaptation_data"]["record_count"] != 500
                or lineage["source_checkpoint"]["sha256"] != "c23ec4361b097b74a4efed427488c4dd6a3753adfc75f7d1fc281e4df0989948"
                or lineage["selection"] != {"policy":"last", "heldout_evaluation_used_for_selection":False}):
            raise ValueError("width2 baseline is not the matched full-evaluation JSD arm")


def finalize(config, complete, output):
    if config["schema_version"] == 5:
        return finalize_repeats(config, complete, output)
    from util.evaluation.ecg_image_queue import atomic_json
    admission(config)
    rows = []
    for c in CENTERS:
        baseline = json.loads(Path(config["width2_baselines"][c]["result"]).read_text())
        for w in (1, 2, 3):
            entry = config["width2_baselines"][c] if w == 2 else complete[f"founder_jsd_w{w}_{c}_eval"]
            value = json.loads(Path(entry["result"]).read_text())
            for field in ("adaptation_data", "seed", "source_checkpoint", "selection", "training_config_sha256", "method_resources"):
                if value["checkpoint"]["lineage"][field] != baseline["checkpoint"]["lineage"][field]:
                    raise ValueError(f"unmatched width comparison: {c} {field}")
            if value["protocol"] != baseline["protocol"]:
                raise ValueError("width evaluation protocol changed")
            clean = value["clean"]["evaluated_center_mean"]["pn2021_drop_all_zero_refexcluded"]
            corrupt = value["corrupted"]["aggregates"]["depth23"]["evaluated_center_mean"]["pn2021c_drop_all_zero_corrupted_refexcluded"]
            rows.append({"center": c, "width": w, "result": entry,
                **{f"{kind}_{metric}": data[metric] for kind,data in (("clean",clean),("corrupt",corrupt))
                   for metric in ("macro_auroc", "macro_auprc")}})
    metrics = ("clean_macro_auroc", "clean_macro_auprc", "corrupt_macro_auroc", "corrupt_macro_auprc")
    means = {str(w):{m:sum(r[m] for r in rows if r["width"]==w)/4 for m in metrics} for w in (1,2,3)}
    report = {"status":"complete", "evidence":"single_seed_development_not_paper_final",
        "aggregation":"equal_center_equal_20_corruption_views_drop_all_zero_K500_refexcluded",
        "rows": rows, "center_means":means,
        "delta_vs_width2_pp":{str(w):{m:100*(means[str(w)][m]-means["2"][m]) for m in metrics} for w in (1,3)}}
    path = output/"width_comparison.json"
    atomic_json(path, report)
    return {"metrics":str(path), "sha256":sha256_file(path)}


def validate_result(payload, path):
    from util.run_record import verify_run_file_index
    if (payload.get("artifact_type") != "pulse_training_queue_result" or payload.get("status") != "complete"
            or set(payload.get("jobs", {})) != (set(repeat_grid()) if payload.get("schema_version") == 5 else set(KEYS))):
        raise ValueError("incomplete Founder JSD width grid")
    for entry in payload["jobs"].values():
        p = Path(entry["result"])
        if sha256_file(p) != entry["sha256"] or verify_run_file_index(p.parent.parent):
            raise ValueError("width job artifact changed")
    comparison = payload["width_comparison"]
    p = Path(comparison["metrics"])
    if p != path.parent/"width_comparison.json" or sha256_file(p) != comparison["sha256"]:
        raise ValueError("width comparison missing or changed")


def repeat_grid():
    return {
        f"{arm}_seed{seed}_{short}_{center}{suffix}": (arm, seed, model, center, suffix)
        for seed in range(3) for center in CENTERS
        for arm in ("jsd_w3", "jsd_w1", "jsd_w2", "simclr")
        for short, model in (("ecgfounder", "ecgfounder"), ("effnet", "efficientnet1dv2"))
        for suffix in ("", "_eval")
    }


def repeat_plans(config, config_path, config_root):
    """Only the registered 96 paired jobs, with the existing managed delegates."""
    from boot_scripts.run_experiment import load_experiment_plan
    from core.methods.registry import load_recipe_spec
    grid = repeat_grid()
    baselines = {"frozen_jsd_training":"train/PN2021_a1_joint_long_budget_r13.yaml",
                 "frozen_simclr_training":"train/PN2021.yaml"}
    if (set(config["references"]) != set(grid) | set(baselines)
            or any(config["references"][k] != v for k,v in baselines.items())):
        raise ValueError("repeat study requires all 192 train/eval configurations")
    root = Path(config["run_root"]).resolve()
    root.relative_to(Path("/home/linbinhao/ECG_adv_data/runs"))
    result = {}
    for key, (arm, seed, model, center, suffix) in grid.items():
        path = resolve_config_reference(config["references"][key], owner_config_path=config_path,
            config_root=config_root, description=key, must_exist=True)
        plan = load_experiment_plan(path, config_root=config_root, allow_existing_run_dir=True)
        family = "simclr" if arm == "simclr" else "jsd"
        method = ("augmix_simclr_lhat" if arm == "simclr" else
                  "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2" if arm == "jsd_w2" else
                  f"a1_rot4_jsd_width{arm[-1]}_vae_lhat_replace0p2")
        method_path = f"train/methods/{method}.yaml"
        short = "effnet" if model == "efficientnet1dv2" else model
        source = f"/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_{short}_source_v1/training/checkpoints/best.pt"
        expected = (["--model", model, "--train-result", str(root/key.removesuffix("_eval")/"training/train_result.json"),
                     "--center", center, "--method-config", method_path] if suffix else
                    ["--model", model, "--method-config", method_path, "--center", center, "--source-checkpoint", source])
        if (plan.run_dir != root/key or list(plan.entry_arguments) != expected
                or plan.entrypoint_name != ("evaluate_pn2021" if suffix else "train_pn2021")
                or plan.entry_config_path != config_root/("eval/PN2021.yaml" if suffix else
                                                         f"train/jsd_width_repeat/{family}_seed{seed}.yaml")):
            raise ValueError(f"repeat job escaped its registered identity: {key}")
        result[key] = (path, plan)
    for family in ("jsd", "simclr"):
        baseline_path = "train/PN2021.yaml" if family == "simclr" else "train/PN2021_a1_joint_long_budget_r13.yaml"
        baseline = load_yaml_mapping(config_root/baseline_path, description="frozen training")
        for seed in range(3):
            child = load_yaml_mapping(config_root/f"train/jsd_width_repeat/{family}_seed{seed}.yaml", description="repeat training")
            identity = {"stream_namespace":"jsd_simclr_width_repeat_20260912_v1",
                        "comparison_group":"jsd_simclr_width_repeat_20260912_v1",
                        "replicate_id":seed, "deterministic_algorithms":False}
            if child["random_seed"] != identity:
                raise ValueError("repeat seed identity changed")
            child["output"]["root_dir"] = baseline["output"]["root_dir"]
            for field in ("profile_name", "random_seed"):
                child[field] = baseline[field]
            if child != baseline:
                raise ValueError("repeat changed the frozen training recipe")
    base = load_recipe_spec(config_root/"train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml")
    for width in (1, 3):
        arm = load_recipe_spec(config_root/f"train/methods/a1_rot4_jsd_width{width}_vae_lhat_replace0p2.yaml")
        if (arm.scientific_contract["lhat_auxiliary"] != base.scientific_contract["lhat_auxiliary"]
                or arm.comparison_rng_identity != base.comparison_rng_identity):
            raise ValueError("width ablation changed LHAT or its RNG identity")
    return result


def finalize_repeats(config, complete, output):
    """Seeds, not corruption views or individual ECGs, are repeat units."""
    import math
    import statistics
    from scipy.stats import t
    from util.evaluation.ecg_image_queue import atomic_json
    rows = []
    paired = {}
    jsd_training = {}
    for key, (arm, seed, model, center, suffix) in repeat_grid().items():
        if not suffix:
            continue
        entry = complete[key]
        value = json.loads(Path(entry["result"]).read_text())
        lineage = value["checkpoint"]["lineage"]
        if (value["status"] != "complete" or value["protocol"]["mapping_hash"] != MAPPING_HASH
                or value["protocol"]["logical_centers"] != [center]
                or value["protocol"]["corruption_cache"]["view_count"] != 20
                or lineage["adaptation_data"]["record_count"] != 500
                or lineage["selection"] != {"policy":"last", "heldout_evaluation_used_for_selection":False}):
            raise ValueError("repeat evaluation contract changed")
        pair = (seed, model, center)
        identity = {k:lineage[k] for k in ("adaptation_data", "seed", "source_checkpoint", "selection")}
        identity["protocol"] = value["protocol"]
        if paired.setdefault(pair, identity) != identity:
            raise ValueError("repeat arms differ in cohort, source, seed or evaluation")
        if arm.startswith("jsd_"):
            config_hash = lineage["training_config_sha256"]
            if jsd_training.setdefault(pair, config_hash) != config_hash:
                raise ValueError("JSD width arms changed training budget/config")
        clean = value["clean"]["evaluated_center_mean"]["pn2021_drop_all_zero_refexcluded"]
        corrupt = value["corrupted"]["aggregates"]["depth23"]["evaluated_center_mean"]["pn2021c_drop_all_zero_corrupted_refexcluded"]
        rows.append({"arm":arm, "seed":seed, "model":model, "center":center, "result":entry,
            **{f"{kind}_{metric}":data[metric] for kind,data in (("clean",clean),("corrupt",corrupt))
               for metric in ("macro_auroc","macro_auprc")}})
    metrics = ("clean_macro_auroc", "clean_macro_auprc", "corrupt_macro_auroc", "corrupt_macro_auprc")
    means, contrasts = {}, []
    for model in ("efficientnet1dv2", "ecgfounder"):
        means[model] = {}
        for arm in ("jsd_w1", "jsd_w2", "jsd_w3", "simclr"):
            seed_means = [{m:statistics.mean(r[m] for r in rows if
                (r["arm"],r["seed"],r["model"]) == (arm,seed,model)) for m in metrics} for seed in range(3)]
            means[model][arm] = {m:{"mean":statistics.mean(s[m] for s in seed_means),
                "sample_sd":statistics.stdev(s[m] for s in seed_means), "seed_means":[s[m] for s in seed_means]} for m in metrics}
        for arm in ("jsd_w2", "jsd_w3"):
            for metric in metrics:
                differences = [100*(a-b) for a,b in zip(means[model][arm][metric]["seed_means"],
                    means[model]["jsd_w1"][metric]["seed_means"])]
                mean, sd = statistics.mean(differences), statistics.stdev(differences)
                se = sd/math.sqrt(3)
                p = float(2*t.sf(abs(mean/se),2)) if se else (1.0 if mean == 0 else 0.0)
                half = float(t.ppf(.975,2))*se
                contrasts.append({"model":model, "contrast":arm+" - jsd_w1", "metric":metric,
                    "paired_seed_delta_pp":differences, "mean_pp":mean, "sample_sd_pp":sd,
                    "exploratory_t95_ci_pp":[mean-half,mean+half], "unadjusted_p":p,
                    "per_center_mean_delta_pp":{c:statistics.mean(
                        100*(next(r[metric] for r in rows if (r["arm"],r["seed"],r["model"],r["center"])==(arm,s,model,c))
                             -next(r[metric] for r in rows if (r["arm"],r["seed"],r["model"],r["center"])==("jsd_w1",s,model,c)))
                        for s in range(3)) for c in CENTERS}})
    primary = sorted((c for c in contrasts if c["metric"]=="corrupt_macro_auprc"), key=lambda c:c["unadjusted_p"])
    adjusted = 0.0
    for rank, c in enumerate(primary):
        adjusted = max(adjusted, min(1.0,(len(primary)-rank)*c["unadjusted_p"]))
        c["holm_p_four_primary_contrasts"] = adjusted
        c["exploratory_positive"] = adjusted < .05 and c["mean_pp"] > 0
        c["meets_practical_1pp_threshold"] = c["mean_pp"] >= 1.0
    report = {"status":"complete", "evidence":"frozen_three_seed_development_not_paper_final",
        "caveats":["SimCLR is a complete two-stage recipe reference, not an equal-budget loss-only control.",
                   "R18 width changes endpoint BCE/JSD view count; original AugMix uses fixed model-view count.",
                   "Three-seed t intervals are exploratory and conditional on these four development centers."],
        "rows":rows, "seed_summary":means, "width_contrasts":contrasts}
    path = output/"width_comparison.json"
    atomic_json(path,report)
    return {"metrics":str(path), "sha256":sha256_file(path)}
