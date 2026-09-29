"""CPU contracts for objective-only Founder and visual-only PULSE JSD experiments."""
from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import torch
import torch.nn.functional as F
import yaml

from core.consistency import categorical_jsd, bernoulli_jsd
from core.augmix import native500_augmix_jsd_views
from core.methods.registry import load_recipe_spec, Stage1Objective
from core.pulse_visual import answer_jsd_loss
from util.augmentations.profile import load_augmentation_profile
from util.pulse_training_contract import validate_config
from util.pulse_training_queue import dual_jsd_plans, DUAL_KEYS

REPO = Path(__file__).resolve().parents[2]


def test_dual_gpu_is_claimed_during_cpu_model_loading():
    from util.pulse_training_queue import unclaimed_dual_devices
    # nvidia-smi can still report an idle device before the child enters CUDA.
    assert unclaimed_dual_devices([0,1,5,7],{0:0,1:0,5:0,7:0},{"loading":{"gpu":0}},100)==[1,5,7]
    assert unclaimed_dual_devices([0,1],{0:99,1:0},{},100)==[1]


def test_recovered_child_requires_live_identity_then_terminal_manifest(monkeypatch,tmp_path):
    import json
    from util import pulse_training_queue as queue
    identity={"pid":123,"start_ticks":"999","boot_id":"test"}
    child=queue.RecoveredManagedChild(identity,tmp_path)
    monkeypatch.setattr(queue,"process_identity",lambda pid:dict(identity))
    assert child.poll() is None
    monkeypatch.setattr(queue,"process_identity",lambda pid:None)
    assert child.poll()==75
    (tmp_path/"run_manifest.json").write_text(json.dumps({"status":"complete"}))
    assert child.poll()==0


@pytest.mark.parametrize("method,weight",[("augmix_clean_bce_lhat",0),("augmix_clean_bce_jsd_lhat",12)])
def test_founder_stage1_three_forwards_and_frozen_head(monkeypatch,method,weight):
    from core import online_trainer as trainer
    from core.methods.runtime import build_method_runtime
    recipe=load_recipe_spec(REPO/f"configs/train/methods/{method}.yaml")
    runtime=build_method_runtime(recipe,model_name="efficientnet1dv2",
        config_root=REPO/"configs",latent_pool=object(),decoder=torch.nn.Identity())
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=torch.nn.Linear(1,5); self.head=torch.nn.Linear(5,5); self.calls=0
        def forward(self,x):
            self.calls+=1; return self.head(self.backbone(x))
    model=Model()
    before={n:p.detach().clone() for n,p in model.head.named_parameters()}
    def forward(m,x,spec):
        value=m(x); return value,value
    patches={
        "_cache_k500_logits":lambda *a,**k:{},
        "_feature_width":lambda *a,**k:5,
        "_head_parameters":lambda m,spec:tuple(m.head.parameters()),
        "prepare_canonical_model_input":lambda raw,*a,**k:raw.mean((1,2)).unsqueeze(1),
        "generate_two_chain_augmix_strong_view":lambda raw,**k:SimpleNamespace(mixed_raw=raw*.5),
        "_forward_logits_and_features":forward,
        "_teacher_logits_for_hashes":lambda cache,hashes,**k:torch.zeros(len(hashes),5),
        "_simclr_nt_xent":lambda *a,**k:pytest.fail("objective ablation called SimCLR"),
        "_weighted_logit_anchor_loss":lambda logits,teacher,weights:logits.square().mean(),
    }
    for name,value in patches.items(): monkeypatch.setattr(trainer,name,value)
    loader=[{"waveform":torch.ones(2,1000,12),"label":torch.zeros(2,5),"hash_id":("a","b")}]
    summary=trainer._run_augmix_stage1(model,loader,spec=SimpleNamespace(name="efficientnet1dv2"),
        device=torch.device("cpu"),recipe=recipe,augmix_config=runtime.augmix_config,
        center="ningbo",base_seed=20260501,resolved={"stage1_steps":1,"stage1_learning_rate":1e-3,
        "stage1_weight_decay":0.,"stage1_gradient_clip_norm":1.},pos_weight=None,
        normalization_epsilon=1e-6,amp_enabled=False,amp_dtype=torch.bfloat16)
    assert model.calls==3 and summary["jsd_weight"]==weight and not summary["projector_used"]
    assert summary["mean_total_loss"]==pytest.approx(summary["mean_clean_bce"]
        +weight*summary["mean_bernoulli_jsd"]+5*summary["mean_logit_anchor_loss"],rel=1e-6)
    for n,p in model.head.named_parameters():
        assert torch.equal(p,before[n]) and p.requires_grad and p.grad is None
    assert model.backbone.weight.grad.abs().sum()>0


def test_jsd_matches_author_kl_and_all_three_gradients():
    torch.manual_seed(81)
    xs = [torch.randn(3, 17, requires_grad=True) for _ in range(3)]
    ps = [x.softmax(-1) for x in xs]
    mean = torch.stack(ps).mean(0)
    expected = sum(F.kl_div(mean.log(), p, reduction="batchmean") for p in ps)/3
    actual = categorical_jsd(xs)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    first = torch.autograd.grad(expected, xs, retain_graph=True)
    second = torch.autograd.grad(actual, xs)
    for a,b in zip(first,second):
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
        assert b.abs().sum() > 0
    assert abs(float(categorical_jsd([xs[0]]*3))) < 1e-7


def test_bernoulli_jsd_is_finite_at_extreme_multilabel_logits():
    xs = [torch.tensor([[1000.,-1000.,0.,2.,-3.]],requires_grad=True) * sign for sign in (1,-1,1)]
    loss = bernoulli_jsd(xs)
    assert torch.isfinite(loss) and 0 <= loss <= 1.1
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss,xs))


def test_pulse_clean_ce_and_jsd_recipe():
    xs = [torch.randn(6,13,requires_grad=True) for _ in range(3)]
    labels = torch.arange(6)
    loss,ce,jsd = answer_jsd_loss(xs,labels)
    torch.testing.assert_close(loss,ce+12*jsd)
    one,clean,zero = answer_jsd_loss(xs[:1],labels)
    assert zero == 0
    torch.testing.assert_close(one,F.cross_entropy(xs[0],labels))
    assert torch.autograd.grad(loss,xs)[2].abs().sum() > 0


def test_visual_inference_recomputes_clip_after_every_arm_switch(monkeypatch):
    from contextlib import nullcontext
    from util.evaluation import pulse_adapters as adapters
    from util.evaluation.pulse_visual_subset import make_backend, ARMS
    calls=[]
    class Base:
        def get_vision_tower(self):
            def forward(pixels):
                calls.append(self.arm); return torch.tensor([[ARMS.index(self.arm)]])
            return forward
    class Parent:
        def __init__(self,evidence,reservation):
            self.base=Base(); self.prompt_ids=torch.tensor([[1,2]]); self.tokenizer=None
        def set_pair(self,evidence): assert tuple(evidence)==ARMS
        def _select(self,arm): self.base.arm=arm
        def _generate(self,packed): return packed
    class Pixels:
        ndim=5; shape=(1,5,3,336,336)
        def __len__(self): return 1
        def to(self,**kwargs): return self
        def flatten(self,*args): return self
    monkeypatch.setattr(adapters,"PairedPulseBackend",Parent)
    monkeypatch.setattr(adapters,"pack_shared_prompt_features",lambda base,ids,features:features)
    monkeypatch.setattr(adapters,"decode_generated",lambda tokenizer,value,**k:value.tolist())
    monkeypatch.setattr(torch,"autocast",lambda *a,**k:nullcontext())
    backend=make_backend({a:{} for a in ARMS},[])
    assert backend.generate_pair(Pixels())=={a:[[i]] for i,a in enumerate(ARMS)}
    assert calls==list(ARMS)


def test_visual_predictions_reject_missing_arm_and_bad_condition_index():
    from util.evaluation.pulse_visual_subset import validate_row
    with pytest.raises(ValueError,match="arm or condition-index"):
        validate_row({"arms":{"clean":{}},"condition_index":0},{},{"condition_index":0})
    with pytest.raises(ValueError,match="arm or condition-index"):
        validate_row({"arms":dict.fromkeys(("clean","single","three")),"condition_index":1},{},{"condition_index":0})


@pytest.mark.parametrize("width",[0,1,3])
def test_original_mix_topology_and_rng(width):
    class Renderer:
        def __init__(self): self.images=[]
        def render(self,wave):
            image=wave[:,:9].square().transpose(1,2).unsqueeze(1)
            self.images.append(image.clone())
            return image
    wave=torch.linspace(-1,1,60000).reshape(1,5000,12)
    renderer=Renderer()
    profile=load_augmentation_profile(REPO/"configs/augmentation/operators.yaml")
    before=torch.get_rng_state().clone()
    views,trace=native500_augmix_jsd_views(wave,width=width,profile=profile,seed=6,identity="paired",renderer=renderer)
    assert torch.equal(before,torch.get_rng_state())
    assert len(views)==(3 if width else 1)
    assert len(renderer.images)==1+2*width
    for i,t in enumerate(trace["views"]):
        assert all(1<=len(c)<=3 for c in t["compositions"])
        assert abs(sum(t["weights"])-1)<1e-6 and 0<=t["m"]<=1
        mixed=sum(w*im for w,im in zip(t["weights"],renderer.images[1+i*width:1+(i+1)*width]))
        torch.testing.assert_close(views[i+1],(1-t["m"])*views[0]+t["m"]*mixed)


def test_founder_only_stage1_contract_changes():
    directory=REPO/"configs/train/methods"
    original=load_recipe_spec(directory/"augmix_simclr_lhat.yaml")
    for name,obj in [("augmix_clean_bce_lhat",Stage1Objective.CLEAN_BCE),
                     ("augmix_clean_bce_jsd_lhat",Stage1Objective.CLEAN_BCE_JSD)]:
        new=load_recipe_spec(directory/f"{name}.yaml")
        assert new.stage1_objective is obj
        assert new.comparison_rng_identity==original.comparison_rng_identity
        assert new.resources==original.resources
        a,b=dict(original.scientific_contract),dict(new.scientific_contract)
        for k in ("stages","stage1"): a.pop(k); b.pop(k)
        assert a==b


def test_visual_configs_and_finite_grid(tmp_path):
    path=REPO/"configs/train/dual_jsd_workflow.yaml"
    config=yaml.safe_load(path.read_text())
    # Historical output roots may now be symlinks to a cold archive. Exercise
    # this grid in a fresh home-owned namespace without weakening path guards.
    from uuid import uuid4
    import shutil
    from util.config_bundle import resolve_yaml_config_closure
    old_root=config["run_root"]
    run_root=Path("/home/linbinhao/ECG_adv_data/runs")/f"unit_dual_{uuid4().hex}"
    isolated=tmp_path/"configs"
    closure=resolve_yaml_config_closure([path],config_root=REPO/"configs")
    for source in closure:
        destination=isolated/source.relative_to(REPO/"configs")
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(source.read_text().replace(old_root,str(run_root)))
    local_path=isolated/path.relative_to(REPO/"configs")
    config=yaml.safe_load(local_path.read_text())
    plans=dual_jsd_plans(config,local_path,isolated)
    assert tuple(plans)==DUAL_KEYS and len(plans)==44
    assert all(plan.run_dir==run_root/key for key,(_,plan) in plans.items())
    assert not run_root.exists()
    # Prove the coordinator is portable with only its declared config closure.
    relocated=tmp_path/"relocated"
    shutil.copytree(isolated,relocated)
    copied=relocated/path.relative_to(REPO/"configs")
    assert tuple(dual_jsd_plans(yaml.safe_load(copied.read_text()),copied,relocated))==DUAL_KEYS
    bad_root=copy.deepcopy(config); bad_root["run_root"]="/data/archived_dual_fixture"
    with pytest.raises(ValueError):
        dual_jsd_plans(bad_root,local_path,isolated)
    for key,(_,plan) in plans.items():
        if plan.entrypoint_name == "train_ecg_image":
            data=yaml.safe_load(plan.entry_config_path.read_text())
            validate_config(data)
            bad=copy.deepcopy(data); bad["width"]=2
            with pytest.raises(ValueError): validate_config(bad)
