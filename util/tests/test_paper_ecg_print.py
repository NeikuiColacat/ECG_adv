"""CPU source oracles and explicit native-CUDA admission for printing v3."""
import itertools
import json
import os
import random

import numpy as np
import pytest
import torch

from core import paper_ecg_print as port
from core.paper_ecg_quilting import apply_wrinkle_texture, minimum_cut_path, quilt_texture


def paper(dtype=torch.float32, device='cpu'):
    value = torch.ones(2, 3, 40, 64, dtype=dtype, device=device)
    value[:, 1:, ::8, :] = .8
    value[:, 1:, :, ::8] = .8
    value[:, :, 19:21, 5:59] = .1
    return value


def textures():
    return torch.rand(3, 3, 25, 27, generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize('op', port.PRINT_OPERATORS)
@pytest.mark.parametrize('severity', [0, 2, 5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_operator_replay_input_global_rng_and_zero(op, severity, dtype):
    value = paper(dtype); saved = value.clone()
    global_state = torch.random.get_rng_state().clone()
    rng = torch.Generator().manual_seed(41); before = rng.get_state().clone()
    a = port.apply_paper_operator(value, op, severity=severity, rng=rng, texture_bank=textures())
    b = port.apply_paper_operator(value, op, severity=severity, rng=torch.Generator().manual_seed(41), texture_bank=textures())
    assert torch.equal(a, b) and torch.equal(value, saved)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert a.shape == value.shape and a.dtype == value.dtype
    assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    if severity == 0:
        assert torch.equal(a, value) and torch.equal(before, rng.get_state())


def _roller_reference(width, line_width, p):
    high, low = p.randint(86, 99), p.randint(70, 85)
    first = np.linspace(high, low, line_width)
    patterns = [np.r_[first, first[::-1]],
                np.r_[first, np.full(p.randint(1, 6), low), first[::-1]],
                np.r_[first, first[::-1], np.full(p.randint(1, 6), high)]]
    high += p.randint(-3, 3); low -= p.randint(5, 8)
    first = np.linspace(high, low, line_width)
    patterns += [np.r_[first, first[::-1]],
                 np.r_[first, np.full(p.randint(1, 6), low), first[::-1]],
                 np.r_[first, first[::-1], np.full(p.randint(1, 6), high)]]
    output = []
    while sum(map(len, output)) < width:
        output.append(patterns[p.randrange(6)])
    return np.concatenate(output)[:width]


@pytest.mark.parametrize('line_width', [1, 2, 7, 32])
def test_roller_six_patterns_match_source_formulas(line_width):
    expected = _roller_reference(193, line_width, random.Random(29))
    actual = port._roller_mask(193, line_width, random.Random(29), 'cpu').flatten().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize('dark', [False, True])
def test_roller_blend_matches_source_uint8(dark):
    value = paper()
    a, b = [port._roller_mask(64, w, random.Random(4), 'cpu') for w in (4, 32)]
    pixels = value.mul(255).round().numpy()
    mask = np.maximum(a.numpy()*b.numpy()/100, 0) if dark else np.minimum(a.numpy()*(2-b.numpy()/100), 99)
    expected = np.clip(pixels*(2-mask/100 if dark else mask/100), 0, 255).astype(np.uint8)
    actual = port.roller_blend(value, a, b, dark_background=dark).mul(255).round().numpy()
    assert np.max(np.abs(actual-expected)) <= 1


def test_scatter_duplicates_preserve_last_write_and_reject_out_of_bounds():
    x = torch.tensor([[1, 1, -1, 8, 2]]); y = torch.tensor([[2, 2, 0, 0, 1]])
    values = torch.tensor([[256+3, 512+27, 768, 1024, 1280+8]])
    out = port._scatter_last((1, 4, 5), x, y, values)
    expected = torch.full((1, 1, 4, 5), 255.)
    expected[0, 0, 2, 1] = 27; expected[0, 0, 1, 2] = 8
    assert torch.equal(out, expected)


def test_folding_matches_two_opencv_perspective_strips():
    cv2 = pytest.importorskip('cv2')
    raw = np.random.default_rng(5).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    expected = raw.copy(); height, center, width, shift = 48, 32, 12, 4
    source = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    for left, start in ((True, center-width), (False, center)):
        dest = source.copy();dest[[2, 3] if left else [0, 1], 1] += shift
        matrix = cv2.getPerspectiveTransform(source, dest)
        crop = (raw[:, start:start+width]*.995).astype(np.uint8)
        out = cv2.warpPerspective(crop.astype(float), matrix, (width, height+shift)).astype(np.uint8)
        mask = cv2.warpPerspective(np.full_like(crop, 255).astype(float), matrix, (width, height+shift)).astype(np.uint8)
        out[mask < 255] = 255; expected[:, start:start+width] = out[:height]
    actual = port.fold_image(torch.from_numpy(raw).permute(2, 0, 1)[None].float()/255,
        fold_x=center, fold_width=width, fold_shift=shift).mul(255).round()[0].permute(1, 2, 0).numpy()
    assert np.max(np.abs(actual-expected)) <= 1


@pytest.mark.parametrize('method', port.FAX_METHODS)
@pytest.mark.parametrize('kind', ['random', 'constant', 'bimodal'])
def test_fax_thresholds_match_skimage(method, kind):
    filters = pytest.importorskip('skimage.filters')
    raw = np.random.default_rng(7).integers(0, 256, (41, 57), dtype=np.uint8)
    if kind == 'constant':raw[:] = 93
    if kind == 'bimodal':raw = np.where(raw > 128, 220, 17).astype(np.uint8)
    expected = getattr(filters, 'threshold_'+method)(raw)
    actual = port.grayscale_threshold(torch.from_numpy(raw)[None, None].float(), method).numpy()[0, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=0)


@pytest.mark.parametrize('angle', [0, 30, 90])
def test_halftone_matches_opencv_block_reference(angle):
    cv2 = pytest.importorskip('cv2')
    raw = np.random.default_rng(8).random((40, 64)).astype(np.float32)
    def rotate(a, degrees):
        h, w = a.shape; m = cv2.getRotationMatrix2D((w/2, h/2), degrees, 1)
        nw, nh = int(h*abs(m[0, 1])+w*abs(m[0, 0])), int(h*abs(m[0, 0])+w*abs(m[0, 1]))
        m[0, 2] += nw/2-w/2; m[1, 2] += nh/2-h/2
        return cv2.warpAffine(a, m, (nw, nh))
    rotated = rotate(raw, angle); kernel = np.zeros((5, 5), dtype=np.float64);kernel[2, 2] = 1
    kernel = cv2.GaussianBlur(kernel, (5, 5), 2);kernel /= kernel.max()
    expected = np.zeros_like(rotated, dtype=np.float64)
    for y in range(0, rotated.shape[0]-4, 5):
        for x in range(0, rotated.shape[1]-4, 5):expected[y:y+5, x:x+5] = rotated[y:y+5, x:x+5].mean()*kernel
    expected = rotate(expected, -angle)
    y, x = (expected.shape[0]-40)//2, (expected.shape[1]-64)//2
    expected = expected[y:y+40, x:x+64]
    actual = port.halftone_image(torch.from_numpy(raw)[None, None], angle=angle).numpy()[0, 0]
    error = np.abs(actual-expected)
    assert error.max() <= .025 and error.mean() <= .002


@pytest.mark.parametrize('noise_type', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('side', port.NOISE_SIDES)
def test_photocopy_noise_modes_are_finite_replayable(noise_type, side):
    def make():return port.photocopy_noise((1, 32, 48), random.Random(18), torch.Generator().manual_seed(12), 'cpu', noise_type=noise_type, side=side)
    a = make()
    assert torch.equal(a, make()) and a.shape == (1, 1, 32, 48)
    assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 255


def test_photocopy_fixed_mask_matches_opencv_multiplication():
    cv2 = pytest.importorskip('cv2');p = np.random.default_rng(11)
    raw = p.integers(0, 256, (40, 64, 3), dtype=np.uint8);mask = p.integers(0, 256, (40, 64), dtype=np.uint8)
    expected = np.stack([cv2.multiply(mask, raw[..., c], scale=1/255) for c in range(3)], -1)
    actual = port.photocopy_from_mask(torch.from_numpy(raw).permute(2, 0, 1)[None].float()/255,
        torch.from_numpy(mask)[None, None].float()).mul(255).round()[0].permute(1, 2, 0).numpy()
    assert np.max(np.abs(actual-expected)) <= 1


def test_worley_chunking_matches_exhaustive_distances():
    points = torch.tensor([[[1., 2.], [3., 5.], [6., 1.]]])
    expected = np.array([[min(np.hypot(x-px, y-py) for px, py in points[0].numpy()) for x in range(8)] for y in range(7)])
    np.testing.assert_allclose(port.worley_noise(7, 8, points)[0, 0], expected, atol=1e-6)


def test_minimum_seam_matches_exhaustive_lexicographic_optimum():
    rng = np.random.default_rng(3)
    for costs in (rng.integers(0, 5, (6, 3)), np.zeros((6, 3), dtype=np.int64)):
        candidates = [(sum(costs[y, x] for y, x in enumerate(path)), path) for path in itertools.product(range(3), repeat=6)
                      if all(abs(a-b) <= 1 for a, b in zip(path, path[1:]))]
        expected = min(candidates)[1]
        assert tuple(minimum_cut_path(torch.from_numpy(costs)[None])[0].tolist()) == expected


@pytest.mark.parametrize('grid', [(1, 1), (2, 2), (2, 3)])
def test_quilting_matches_exhaustive_overlap_reference(grid):
    raw = np.random.default_rng(21).integers(0, 256, (17, 19, 3), dtype=np.int32)
    block, overlap = 12, 2;step = block-overlap
    result = np.zeros((grid[0]*step+overlap, grid[1]*step+overlap, 3), dtype=np.int32)
    def seam(errors):
        # Independent scalar dynamic program with full-path tuple tie breaking.
        paths = [(int(e), (x,)) for x, e in enumerate(errors[0])]
        for row in errors[1:]:
            paths = [min((cost+int(row[x]), path+(x,)) for cost, path in paths if abs(path[-1]-x) <= 1) for x in range(len(row))]
        return min(paths)[1]
    for i in range(grid[0]):
        for j in range(grid[1]):
            y, x = i*step, j*step;old = result[y:y+block, x:x+block]
            keep = np.zeros((block, block), dtype=bool)
            if j:keep[:, :overlap] = True
            if i:keep[:overlap] = True
            choices = [(int(((raw[a:a+block, b:b+block]-old)[keep]**2).sum()), a, b)
                       for a in range(raw.shape[0]-block) for b in range(raw.shape[1]-block)]
            _, a, b = min(choices);patch = raw[a:a+block, b:b+block].copy();keep[:] = False
            if j:
                for yy, xx in enumerate(seam(((patch[:, :overlap]-old[:, :overlap])**2).sum(2))):keep[yy, :xx] = True
            if i:
                for xx, yy in enumerate(seam(((patch[:overlap]-old[:overlap])**2).sum(2).T)):keep[:yy, xx] = True
            patch[keep] = old[keep];old[:] = patch
    actual = quilt_texture(torch.from_numpy(raw).permute(2, 0, 1)[None].float()/255, block, grid)
    assert np.array_equal(actual[0].permute(1, 2, 0).mul(255).round().numpy(), result)


def test_wrinkle_overlay_matches_author_fixed_texture():
    cv2 = pytest.importorskip('cv2')
    raw = np.random.default_rng(22).integers(0, 256, (40, 64, 3), dtype=np.uint8)
    texture = np.random.default_rng(23).integers(0, 256, (25, 27, 3), dtype=np.uint8)
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray = cv2.resize(gray, (64, 40));gray -= gray.mean()-.4
    value = raw.astype(np.float32)/255
    expected = np.where(gray[..., None] > .6, 1-2*(1-value)*(1-gray[..., None]), 2*value*gray[..., None])
    expected = (expected*255).clip(0, 255).astype(np.uint8)
    actual = apply_wrinkle_texture(torch.from_numpy(raw).permute(2, 0, 1)[None].float()/255,
        torch.from_numpy(texture).permute(2, 0, 1)[None].float()/255).mul(255).round()[0].permute(1, 2, 0).numpy()
    assert np.max(np.abs(actual-expected)) <= 2


def test_version_three_dispatch_has_23_conditions_and_private_replay(monkeypatch):
    from core.image_corruption import IMAGE_IMPLEMENTATION_SOURCES, gpu_image_identity
    from util.evaluation.pulse_hybrid_development import gpu_image_conditions_for, gpu_image_view
    from util.pulse_hybrid_contract import PAPER_CONDITION_COUNTS
    monkeypatch.setattr(port, 'load_original_texture_bank', lambda device: textures())
    conditions = gpu_image_conditions_for([{'condition_id': 'clean', 'operators': []}], suite='paper_ecg_gpu_v3', severity=2)
    assert len(conditions) == PAPER_CONDITION_COUNTS['paper_ecg_gpu_v3'] == 23
    assert len(port.paper_conditions()) == 110
    assert set(port.PAPER_TRAIN_OPERATORS).isdisjoint(port.PAPER_HELDOUT_OPERATORS)
    assert len(port.PAPER_TRAIN_OPERATORS) == 13 and len(port.PAPER_HELDOUT_OPERATORS) == 9
    assert 'core/paper_ecg_quilting.py' in IMAGE_IMPLEMENTATION_SOURCES['paper_ecg_torch_v3']
    assert len(gpu_image_identity('paper_ecg_torch_v3')['wrinkle_bank']) == 64
    value = paper();samples = [{'hash_id': 'one'}, {'hash_id': 'two'}]
    for condition in conditions:
        a = gpu_image_view(value, condition, samples, 42)
        assert torch.equal(a, gpu_image_view(value, condition, samples, 42))
    assert gpu_image_view(value, conditions[0], samples, 42) is value


def test_missing_texture_bank_is_rejected(tmp_path):
    from core.paper_ecg_quilting import load_original_texture_bank
    with pytest.raises(ValueError, match='pinned wrinkle texture'):
        load_original_texture_bank('cpu', tmp_path)


def test_texture_fetch_never_overwrites_mismatched_files(tmp_path, monkeypatch):
    from core.paper_ecg_quilting import fetch_original_textures, WRINKLE_TEXTURE_SHA256
    import urllib.request
    path = tmp_path/next(iter(WRINKLE_TEXTURE_SHA256))
    path.write_bytes(b'owned mismatched fixture')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('must reject before network'))
    with pytest.raises(ValueError, match='existing texture differs'):
        fetch_original_textures(tmp_path)
    assert path.read_bytes() == b'owned mismatched fixture'


@pytest.mark.parametrize('severity', [-1, 6, True, float('nan')])
def test_invalid_print_request_does_not_consume_rng(severity):
    rng = torch.Generator().manual_seed(3); before = rng.get_state().clone()
    with pytest.raises(ValueError):port.apply_paper_operator(paper(), 'faxify', severity=severity, rng=rng)
    assert torch.equal(before, rng.get_state())


@pytest.mark.skipif(os.environ.get('PULSE_GPU_OPS_NATIVE') != '1', reason='explicit free GPU admission required')
def test_native_print_pool_replay_and_timing():
    from core.paper_ecg_quilting import load_original_texture_bank
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-') and torch.cuda.device_count() == 1
    torch.use_deterministic_algorithms(True)
    image = torch.ones(1, 3, 1700, 2200, device='cuda', dtype=torch.float16)
    image[:, 1:, ::20, :] = .8;image[:, 1:, :, ::20] = .8;image[:, :, 850:853, :] = .1
    bank = load_original_texture_bank(str(image.device))
    source = image.clone();global_state = torch.cuda.get_rng_state().clone();rows = []
    for op in port.PAPER_EVAL_OPERATORS:
        def apply():return port.apply_paper_operator(image, op, severity=2, rng=torch.Generator(device=image.device).manual_seed(41), validate=False, texture_bank=bank)
        a = apply();assert torch.equal(a, apply()) and torch.equal(image, source)
        assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 and a.dtype == image.dtype
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats();start.record()
        for _ in range(3):b = apply()
        end.record();end.synchronize();assert torch.equal(a, b)
        rows.append({'operator': op, 'milliseconds': start.elapsed_time(end)/3, 'peak_bytes': torch.cuda.max_memory_allocated()})
    for method in port.FAX_METHODS:
        a = port.faxify(image, method=method)
        assert torch.equal(a, port.faxify(image, method=method)) and torch.isfinite(a).all()
    for kind in range(1, 6):
        def apply_noise():return port.bad_photocopy(image, random.Random(5), torch.Generator(device=image.device).manual_seed(6), noise_type=kind, side='all', blur=True, wave=True, edge=True)
        assert torch.equal(apply_noise(), apply_noise())
    tiny = bank[:1, :, :25, :27]
    assert torch.equal(quilt_texture(tiny, 12, (2, 2)), quilt_texture(tiny, 12, (2, 2)))
    assert torch.equal(global_state, torch.cuda.get_rng_state())
    print(json.dumps({'printing_gpu': rows, 'shape': list(image.shape), 'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__)}))
