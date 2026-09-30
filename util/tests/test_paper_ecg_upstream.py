"""Fixed-parameter source oracles and device replay for paper version two."""
import json
import os

import numpy as np
import pytest
import torch

from core.paper_ecg_upstream import (PAPER_EVAL_OPERATORS, PORTED_OPERATORS,
    _color_paper_from_hs, _ink_bleed_from_mask, _low_ink_from_rows, apply_paper_operator)


def fixture_image(device='cpu', dtype=torch.float32):
    image=torch.ones(2,3,40,64,device=device,dtype=dtype)
    image[:,1:,::8,:]=.8
    image[:,1:,:,::8]=.8
    image[:,:,19:21,5:59]=.1
    return image


@pytest.mark.parametrize('operator',PAPER_EVAL_OPERATORS)
@pytest.mark.parametrize('severity',[0,2,5])
def test_private_rng_replay_finite_range_and_input_immutability(operator,severity):
    image=fixture_image();before=image.clone();global_rng=torch.random.get_rng_state().clone()
    rng=torch.Generator().manual_seed(19);initial=rng.get_state().clone()
    a=apply_paper_operator(image,operator,severity=severity,rng=rng)
    b=apply_paper_operator(image,operator,severity=severity,rng=torch.Generator().manual_seed(19))
    assert torch.equal(a,b) and torch.equal(image,before)
    assert torch.equal(torch.random.get_rng_state(),global_rng)
    assert a.shape==image.shape and a.dtype==image.dtype
    assert torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
    if severity==0:
        assert torch.equal(a,image) and torch.equal(rng.get_state(),initial)
    else:
        assert not torch.equal(a,image)


def test_color_paper_fixed_parameters_match_opencv_within_one_byte():
    cv2=pytest.importorskip('cv2');rng=np.random.default_rng(39)
    image=rng.integers(0,256,(31,47,3),dtype=np.uint8)
    hue=rng.integers(23,50,(31,47),dtype=np.uint8)
    saturation=rng.integers(5,46,(31,47),dtype=np.uint8)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);hsv[...,0]=hue;hsv[...,1]=saturation
    expected=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    actual=_color_paper_from_hs(torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()/255,
        torch.from_numpy(hue).reshape(1,1,31,47).float(),torch.from_numpy(saturation).reshape(1,1,31,47).float())
    error=np.abs(actual[0].permute(1,2,0).mul(255).round().numpy()-expected.astype(float))
    assert error.max()<=1


@pytest.mark.parametrize('kind',['random','ecg'])
def test_ink_bleed_fixed_mask_matches_opencv_within_one_byte(kind):
    cv2=pytest.importorskip('cv2');rng=np.random.default_rng(17)
    image=rng.integers(0,256,(40,64,3),dtype=np.uint8) if kind=='random' else fixture_image()[0].permute(1,2,0).mul(255).round().byte().numpy()
    mask=rng.random(image.shape[:2])<.35
    edge=cv2.convertScaleAbs(cv2.subtract(cv2.Sobel(image,cv2.CV_32F,1,0,ksize=-1),cv2.Sobel(image,cv2.CV_32F,0,1,ksize=-1)))
    edge=cv2.cvtColor(cv2.dilate(edge,np.ones((5,5),np.uint8)),cv2.COLOR_RGB2GRAY)>0
    eroded=cv2.erode(image,np.ones((5,5),np.uint8));updated=image.copy();updated[mask & edge]=eroded[mask & edge]
    expected=cv2.addWeighted(cv2.GaussianBlur(updated,(3,3),0),.55,image,.45,0)
    actual=_ink_bleed_from_mask(torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()/255,
        torch.from_numpy(mask).reshape(1,1,40,64),.55)
    error=np.abs(actual[0].permute(1,2,0).mul(255).round().numpy()-expected.astype(float))
    assert error.max()<=1


def test_low_ink_rows_never_modify_neighbors_or_darken():
    image=fixture_image();period=torch.tensor([8,7]).reshape(2,1,1,1);offset=torch.tensor([0,2]).reshape(2,1,1,1)
    image[:,:,::8,:]=.2
    out=_low_ink_from_rows(image,period,offset,.8)
    selected=((torch.arange(40).reshape(1,1,40,1)-offset)%period==0).expand_as(image)
    assert torch.equal(out[~selected],image[~selected]) and torch.all(out>=image)
    assert torch.equal(out[selected],image[selected].clamp_min(.8))
    assert not torch.equal(out,image)


def test_versioned_stress_dispatch_replays_and_preserves_clean():
    from core.paper_ecg_upstream import PAPER_TRAIN_OPERATORS, paper_conditions
    from core.image_corruption import IMAGE_IMPLEMENTATION_SOURCES, gpu_image_identity
    from util.evaluation.pulse_hybrid_development import gpu_image_conditions_for, gpu_image_view
    conditions=gpu_image_conditions_for([{'condition_id':'clean','operators':[]}],suite='paper_ecg_gpu_v2',severity=2)
    assert len(conditions)==17 and len(paper_conditions())==80
    assert all(c['image_implementation']=='paper_ecg_torch_v2' for c in conditions[1:])
    assert sum(c.get('stress_group')=='seen_family' for c in conditions)==len(PAPER_TRAIN_OPERATORS)
    assert 'core/paper_ecg_upstream.py' in IMAGE_IMPLEMENTATION_SOURCES['paper_ecg_torch_v2']
    assert gpu_image_identity('paper_ecg_torch_v2')['low_ink_lines']=='row_only_lightening_variant'
    image=fixture_image();before=image.clone();samples=[{'hash_id':'one'},{'hash_id':'two'}]
    for condition in conditions:
        a=gpu_image_view(image,condition,samples,42)
        b=gpu_image_view(image,condition,samples,42)
        assert torch.equal(a,b) and torch.equal(image,before)
    assert gpu_image_view(image,conditions[0],samples,42) is image


@pytest.mark.parametrize('level',[-1,6,True,float('nan')])
def test_invalid_severity_is_rejected_before_rng_consumption(level):
    rng=torch.Generator().manual_seed(4);before=rng.get_state().clone()
    with pytest.raises(ValueError):apply_paper_operator(fixture_image(),'ink_bleed',severity=level,rng=rng)
    assert torch.equal(before,rng.get_state())


@pytest.mark.skipif(os.environ.get('PULSE_GPU_OPS_NATIVE')!='1',reason='explicit admitted GPU required')
def test_native_cuda_replay_and_timing():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and torch.cuda.device_count()==1
    image=torch.ones(1,3,1700,2200,device='cuda',dtype=torch.float16)
    image[:,1:,::20,:]=.8;image[:,1:,:,::20]=.8;image[:,:,850:853,:]=.1
    rows=[]
    for op in PAPER_EVAL_OPERATORS:
        def apply():
            return apply_paper_operator(image,op,severity=2,rng=torch.Generator(device=image.device).manual_seed(41),validate=False)
        a=apply();assert torch.equal(a,apply())
        assert a.dtype==image.dtype and torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start.record()
        for _ in range(5):b=apply()
        end.record();end.synchronize();assert torch.equal(a,b)
        rows.append({'operator':op,'milliseconds':start.elapsed_time(end)/5,'peak_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps({'upstream_paper_gpu':rows,'shape':list(image.shape),'torch':str(torch.__version__)}))
