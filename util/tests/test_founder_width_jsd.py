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
