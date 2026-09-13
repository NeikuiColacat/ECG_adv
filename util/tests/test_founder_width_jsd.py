"""Width-only mixing, unchanged width2 numerics, and successor safety."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest
import torch
import yaml

from core import augmix
from core.methods.registry import load_recipe_spec
from util import founder_width_queue as queue

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("seed", [2, 81, 20260501])
def test_width2_exact_original_rng_and_arithmetic(seed):
    config = augmix.load_augmix_config()
    raw = torch.randn(2, 1000, 12, generator=torch.Generator().manual_seed(23))
    g1, g2 = (torch.Generator().manual_seed(seed) for _ in range(2))
    a = augmix._generate_augmix_multiview(raw, sampling_rate_hz=100, config=config, generator=g1)
    b = augmix._generate_augmix_multiview(raw, sampling_rate_hz=100,
        config=replace(config, stage1_mode="jsd_width_augmix"), generator=g2)
    assert torch.equal(a.mixed_raw, b.mixed_raw) and torch.equal(g1.get_state(), g2.get_state())
    assert all(torch.equal(x,y) for x,y in zip(a.chain_raws,b.chain_raws))


@pytest.mark.parametrize("width", [1, 3])
def test_width_counts_and_clean_residual(monkeypatch, width):
    config = augmix.load_augmix_config(REPO/f"configs/train/augmix_jsd_width{width}.yaml")
    calls = []
    def corrupt(raw, **kwargs):
        calls.append(len(calls)+1)
        return SimpleNamespace(waveform_raw_100hz=raw+calls[-1])
    monkeypatch.setattr(augmix, "generate_canonical_corruption", corrupt)
    monkeypatch.setattr(augmix, "_dirichlet", lambda batch,width,**k:torch.full((batch,width),1/width))
    monkeypatch.setattr(augmix, "_symmetric_beta", lambda batch,**k:torch.full((batch,),.25))
    raw = torch.ones(2,1000,12)
    result = augmix._generate_augmix_multiview(raw, sampling_rate_hz=100, config=config,
        generator=torch.Generator().manual_seed(9))
    assert len(calls) == len(result.chain_raws) == width
    torch.testing.assert_close(result.mixed_raw, torch.full_like(raw,1+.25*(width+1)/2))
    assert torch.equal(raw,torch.ones_like(raw))


@pytest.mark.parametrize("width", [1, 3])
def test_real_width_replay_and_global_rng_isolation(width):
    config = augmix.load_augmix_config(REPO/f"configs/train/augmix_jsd_width{width}.yaml")
    raw = torch.randn(2,1000,12,generator=torch.Generator().manual_seed(64)); before=raw.clone()
    state = torch.get_rng_state()
    outputs = [augmix._generate_augmix_multiview(raw,sampling_rate_hz=100,config=config,
        generator=torch.Generator().manual_seed(75)) for _ in range(2)]
    assert torch.equal(outputs[0].mixed_raw,outputs[1].mixed_raw)
    assert torch.equal(torch.get_rng_state(),state) and torch.equal(raw,before)
    assert torch.isfinite(outputs[0].mixed_raw).all()


def test_finite_plans_config_closure_and_objective_budget(tmp_path):
    import shutil
    from boot_scripts.run_experiment import load_experiment_plan
    path = REPO/"configs/train/founder_jsd_width_workflow.yaml"
    config = yaml.safe_load(path.read_text())
    plans = queue.plans(config,path,REPO/"configs")
    assert len(plans)==16
    base=load_recipe_spec(REPO/"configs/train/methods/augmix_clean_bce_jsd_lhat.yaml")
    for width in (1,3):
        arm=load_recipe_spec(REPO/f"configs/train/methods/augmix_clean_bce_jsd_width{width}_lhat.yaml")
        assert arm.scientific_contract==base.scientific_contract
        assert arm.comparison_rng_identity==base.comparison_rng_identity
        for key,(_,plan) in plans.items():
            if f"_w{width}_" in key:
                assert REPO/f"configs/train/augmix_jsd_width{width}.yaml" in [p for p,_ in plan.config_sources]
    mutated=copy.deepcopy(config);mutated["references"].pop(next(iter(mutated["references"])))
    with pytest.raises(ValueError):queue.plans(mutated,path,REPO/"configs")
    parent=load_experiment_plan(REPO/'configs/experiments/founder_jsd_width_workflow.yaml',
        run_dir=tmp_path/'plan_only_output')
    for source,_ in parent.config_sources:
        target=tmp_path/source.relative_to(REPO/'configs')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    assert len(queue.plans(config,tmp_path/'train/founder_jsd_width_workflow.yaml',tmp_path))==16


def test_predecessor_requires_completed_manifest_and_no_live_process(monkeypatch,tmp_path):
    from util.evaluation import ecg_image_elastic
    # Keep the production path restriction, redirect only reads for these fixtures.
    root=Path('/home/linbinhao/ECG_adv_data/runs/test_predecessor')
    content={root/'run_manifest.json':{'status':'running'},root/'training/status.json':{'status':'complete'}}
    monkeypatch.setattr(Path,'read_text',lambda p,**k:json.dumps(content[p]))
    monkeypatch.setattr(ecg_image_elastic,'process_identity',lambda pid: {'pid':pid})
    config={'predecessor':{'run_dir':str(root),'launcher_pid':1,'delegate_pid':2}}
    assert not queue.predecessor_ready(config)
    content[root/'run_manifest.json']['status']='complete'
    assert not queue.predecessor_ready(config)
    monkeypatch.setattr(ecg_image_elastic,'process_identity',lambda pid:None)
    assert queue.predecessor_ready(config)
    content[root/'run_manifest.json']['status']='failed'
    with pytest.raises(RuntimeError):queue.predecessor_ready(config)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_r18_width_no_beta_replay_and_existing_width2_exact(width):
    original = augmix.load_augmix_config(REPO/"configs/train/augmix_two_chain_no_clean_mix.yaml")
    config = (replace(original, stage1_mode="jsd_width_no_clean_mix") if width == 2 else
              augmix.load_augmix_config(REPO/f"configs/train/augmix_jsd_width{width}_no_clean_mix.yaml"))
    raw = torch.randn(2,1000,12,generator=torch.Generator().manual_seed(61))
    before, state = raw.clone(), torch.get_rng_state()
    outputs = [augmix._generate_augmix_multiview(raw,sampling_rate_hz=100,config=config,
               generator=torch.Generator().manual_seed(71)) for _ in range(2)]
    result = outputs[0]
    assert len(result.chain_raws) == width and config.stage1_beta_alpha == 0
    assert torch.equal(result.mixed_raw,outputs[1].mixed_raw)
    assert torch.equal(raw,before) and torch.equal(torch.get_rng_state(),state)
    if width == 1:
        assert torch.equal(result.mixed_raw,result.chain_raws[0])
    if width == 2:
        old = augmix._generate_augmix_multiview(raw,sampling_rate_hz=100,config=original,
                                              generator=torch.Generator().manual_seed(71))
        assert torch.equal(old.mixed_raw,result.mixed_raw)
        assert all(torch.equal(a,b) for a,b in zip(old.chain_raws,result.chain_raws))


@pytest.mark.parametrize("width", [1, 3])
def test_r18_width_loss_is_mean_endpoint_bce_plus_jsd_with_finite_gradients(width):
    import core.online_trainer as trainer
    from core.methods.contracts import ViewBundle, WaveformView
    from models.contracts import EFFICIENTNET1DV2_SPEC
    from models.input_adapter import prepare_canonical_model_input
    recipe = load_recipe_spec(REPO/f"configs/train/methods/a1_rot4_jsd_width{width}_vae_lhat_replace0p2.yaml")
    base = load_recipe_spec(REPO/"configs/train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml")
    assert recipe.scientific_contract["stage_boundaries"] is False
    assert recipe.scientific_contract["lhat_auxiliary"] == base.scientific_contract["lhat_auxiliary"]
    assert recipe.comparison_rng_identity == base.comparison_rng_identity
    names = ("clean_view","augmix_view",*(f"augmix_chain{i}_view" for i in range(1,width+1)))
    labels = torch.tensor([[1,0,0,1,0],[0,1,1,0,0]],dtype=torch.float32)
    raws = {n:torch.randn(2,1000,12,generator=torch.Generator().manual_seed(i+30)) for i,n in enumerate(names)}
    bundle = ViewBundle(values={n:WaveformView(name=n,waveform=x,labels=labels,sample_ids=("a","b")) for n,x in raws.items()})
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(1.0))
        def forward(self,x):
            return self.scale*x[:,:5,0]
    model=Head()
    objective=trainer._compute_objective(recipe=recipe,bundle=bundle,model=model,spec=EFFICIENTNET1DV2_SPEC,
        normalization_epsilon=1e-6,pos_weight=None,
        objective_term_names=recipe.scientific_contract["augmix_auxiliary"]["objective_terms"])
    logits=[model(prepare_canonical_model_input(raws[n],EFFICIENTNET1DV2_SPEC,epsilon=1e-6)) for n in (names[0],*names[2:])]
    bce=torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(x,labels) for x in logits[1:]]).mean()
    p=torch.stack([x.sigmoid() for x in logits]).clamp(1e-6,1-1e-6); mean=p.mean(0)
    jsd=(p*(p.log()-mean.log())+(1-p)*((1-p).log()-(1-mean).log())).mean()
    torch.testing.assert_close(objective.total,bce+1.5*jsd)
    objective.total.backward()
    assert torch.isfinite(model.scale.grad)


def test_repeat_grid_is_complete_matched_and_dry_run_only(tmp_path):
    import shutil
    from boot_scripts.run_experiment import load_experiment_plan
    path=REPO/"configs/train/jsd_width_repeat/workflow.yaml"
    config=yaml.safe_load(path.read_text())
    jobs=queue.plans(config,path,REPO/"configs")
    assert len(jobs)==192 and len({p.run_dir for _,p in jobs.values()})==192
    assert queue.predecessor_ready(config)
    parent=load_experiment_plan(REPO/"configs/experiments/jsd_width_repeat/workflow.yaml",run_dir=tmp_path/"dry")
    assert not parent.describe()["loads_model_or_gpu"] and not (tmp_path/"dry").exists()
    snapshot=tmp_path/"snapshot"
    for source,_ in parent.config_sources:
        dest=snapshot/source.relative_to(REPO/"configs")
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    assert len(queue.plans(config,snapshot/"train/jsd_width_repeat/workflow.yaml",snapshot))==192
    changed=copy.deepcopy(config);changed["references"].pop(next(iter(changed["references"])))
    with pytest.raises(ValueError,match="192"):
        queue.plans(changed,path,REPO/"configs")


@pytest.fixture
def repeat_results(tmp_path):
    complete={}
    for key,(arm,seed,model,center,suffix) in queue.repeat_grid().items():
        if not suffix:
            continue
        gain={"jsd_w1":0.,"jsd_w2":.01,"jsd_w3":.02,"simclr":-.01}[arm]
        if arm != "jsd_w1":
            gain += (.0,.001,-.002)[seed]
        metrics={"macro_auroc":.8+gain,"macro_auprc":.6+gain}
        payload={"status":"complete","protocol":{"mapping_hash":queue.MAPPING_HASH,"class_order":queue.CLASS_ORDER,
            "logical_centers":[center],"corruption_cache":{"view_count":20}},
            "checkpoint":{"lineage":{"adaptation_data":{"record_count":500,"hash_id_set_sha256":"a"*64},
                "seed":{"replicate":seed,"stream_namespace":"repeat-test"},
                "source_checkpoint":{"sha256":model},"selection":{"policy":"last","heldout_evaluation_used_for_selection":False},
                "training_config_sha256":f"matched-{seed}"}},
            "clean":{"evaluated_center_mean":{"pn2021_drop_all_zero_refexcluded":metrics}},
            "corrupted":{"aggregates":{"depth23":{"evaluated_center_mean":{"pn2021c_drop_all_zero_corrupted_refexcluded":metrics}}}}}
        p=tmp_path/(key+".json");p.write_text(json.dumps(payload))
        complete[key]={"result":str(p),"sha256":queue.sha256_file(p)}
    return complete


def test_repeat_summary_pairs_seeds_not_corruption_views(tmp_path, repeat_results):
    complete = repeat_results
    artifact=queue.finalize_repeats({},complete,tmp_path)
    summary=json.loads(Path(artifact["metrics"]).read_text())
    assert len(summary["rows"])==96 and len(summary["width_contrasts"])==16
    for contrast in summary["width_contrasts"]:
        assert len(contrast["paired_seed_delta_pp"])==3
        assert len(contrast["per_center_mean_delta_pp"])==4
        if contrast["contrast"].startswith("jsd_w2"):
            assert contrast["mean_pp"]==pytest.approx((1+1.1+.8)/3)
        if contrast["metric"]=="corrupt_macro_auprc":
            assert contrast["holm_p_four_primary_contrasts"] >= contrast["unadjusted_p"]
    p=Path(next(iter(complete.values()))["result"])
    payload=json.loads(p.read_text());payload["checkpoint"]["lineage"]["training_config_sha256"]="unmatched"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="training budget"):
        queue.finalize_repeats({},complete,tmp_path)


@pytest.mark.parametrize("arm", ["jsd_w1", "simclr"])
@pytest.mark.parametrize(
    "section,field,replacement",
    [
        ("adaptation_data", "hash_id_set_sha256", "b" * 64),
        ("seed", "stream_namespace", "another-study"),
        ("source_checkpoint", "sha256", "different-source"),
        ("protocol", "class_order", list(reversed(queue.CLASS_ORDER))),
    ],
    ids=["cohort", "seed-namespace", "source-checkpoint", "evaluation-protocol"],
)
def test_repeat_summary_rejects_unmatched_pair(
    tmp_path, repeat_results, arm, section, field, replacement
):
    entry = repeat_results[f"{arm}_seed0_ecgfounder_ningbo_eval"]
    path = Path(entry["result"])
    payload = json.loads(path.read_text())
    identity = (
        payload["protocol"]
        if section == "protocol"
        else payload["checkpoint"]["lineage"][section]
    )
    identity[field] = replacement
    path.write_text(json.dumps(payload))
    entry["sha256"] = queue.sha256_file(path)

    with pytest.raises(ValueError, match="repeat arms differ in cohort, source, seed or evaluation"):
        queue.finalize_repeats({}, repeat_results, tmp_path)
    assert not (tmp_path / "width_comparison.json").exists()
