"""GPU approximation contracts; CUDA tests require explicit single-device opt-in."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from core import image_augmix_gpu as ops


ROOT = Path(__file__).resolve().parents[2]
GPU = pytest.mark.skipif(os.environ.get("PULSE_GPU_OPS_TEST") != "1",
                         reason="requires PULSE_GPU_OPS_TEST=1 and an admitted free GPU")
NATIVE = pytest.mark.skipif(os.environ.get("PULSE_GPU_OPS_NATIVE") != "1",
                            reason="native-canvas tests require separate explicit opt-in")
CASES = [("augmix", op, level) for op in ops.AUGMIX_GPU_OPERATORS for level in (0.1, 3, 10)]
CASES += [("c5", op, level) for op in ops.C5_GPU_OPERATORS for level in range(1, 6)]
OPERATORS = [("augmix", op, 3) for op in ops.AUGMIX_GPU_OPERATORS]
OPERATORS += [("c5", op, 5) for op in ops.C5_GPU_OPERATORS]


def image_fixture(shape=(2, 3, 33, 49), *, device="cpu", dtype=torch.float32):
    # Different channel/sample histograms; non-multiple-of-eight JPEG edges.
    image = torch.arange(np.prod(shape), device=device, dtype=torch.float32).reshape(shape)
    return ((image.remainder(251) + 1) / 253).to(dtype)


def operator_function(family, *, trusted=False):
    if family == "augmix":
        return (ops.apply_gpu_augmix_image_operator_prevalidated if trusted
                else ops.apply_gpu_augmix_image_operator)
    return ops.apply_gpu_c5_image_operator_prevalidated if trusted else ops.apply_gpu_c5_image_operator


@pytest.fixture(scope="module")
def gpu_device():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible.startswith("GPU-") and "," not in visible, "select one admitted GPU UUID"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    return torch.device("cuda:0")


def test_pools_are_complete_disjoint_and_reference_identity_is_separate():
    from core.image_augmix_c import AUGMIX_IMPLEMENTATION, AUGMIX_TRAIN_OPERATORS, C_IMPLEMENTATION
    assert ops.AUGMIX_GPU_OPERATORS == AUGMIX_TRAIN_OPERATORS
    assert len(ops.AUGMIX_GPU_OPERATORS) == 9 and len(ops.C5_GPU_OPERATORS) == 5
    assert set(ops.AUGMIX_GPU_OPERATORS).isdisjoint(ops.C5_GPU_OPERATORS)
    assert ops.GPU_AUGMIX_IMPLEMENTATION != AUGMIX_IMPLEMENTATION
    assert ops.GPU_C5_IMPLEMENTATION != C_IMPLEMENTATION


@pytest.mark.parametrize("family", ["augmix", "c5"])
@pytest.mark.parametrize("trusted", [False, True])
def test_public_paths_refuse_cpu_tensors(family, trusted):
    op = ops.AUGMIX_GPU_OPERATORS[0] if family == "augmix" else ops.C5_GPU_OPERATORS[0]
    with pytest.raises(ValueError, match="CUDA"):
        operator_function(family, trusted=trusted)(image_fixture(), op, severity=3,
                                                  rng=torch.Generator())


@pytest.mark.parametrize("value", [0., 0.5, 1.])
@pytest.mark.parametrize("function", [ops._autocontrast, ops._equalize])
def test_constant_channels_are_preserved(function, value):
    image = torch.full((2, 3, 5, 7), value)
    assert torch.equal(function(image), image)


def test_histogram_and_contrast_have_independent_per_sample_per_channel_oracles():
    image = image_fixture((2, 3, 7, 11))
    equalized = ops._equalize(image).numpy()
    contrasted = ops._autocontrast(image).numpy()
    for b in range(2):
        for c in range(3):
            value = image[b, c].numpy()
            bins = np.rint(value * 255).astype(np.int64)
            hist = np.bincount(bins.ravel(), minlength=256)
            cdf = np.cumsum(hist)
            first = cdf[np.flatnonzero(hist)[0]]
            expected = (cdf[bins] - first) / (value.size - first)
            np.testing.assert_allclose(equalized[b, c], expected, atol=1e-7)
            np.testing.assert_allclose(contrasted[b, c],
                                       (value - value.min()) / (value.max() - value.min()), atol=1e-7)


def test_posterize_high_bits_and_solarize_byte_threshold():
    values = torch.arange(256).view(1, 1, 16, 16).expand(1, 3, -1, -1)
    image = values.float() / 255
    for level in (0.1, 3., 9.9):
        bits = max(1, 4 - int(level * 4 / 10))
        expected = (values >> (8 - bits)) << (8 - bits)
        actual = ops._posterize(image, torch.tensor([level]))
        torch.testing.assert_close(actual, expected.float() / 255, rtol=0, atol=0)
        threshold = 256 - int(level * 256 / 10)
        expected = torch.where(values >= threshold, 255 - values, values)
        actual = ops._solarize(image, torch.tensor([level]))
        torch.testing.assert_close(actual, expected.float() / 255, rtol=0, atol=0)


@pytest.mark.parametrize("operator", ["rotate", "shear_x", "shear_y", "translate_x", "translate_y"])
def test_rectangular_geometry_uses_pixel_space_not_stretched_normalized_space(monkeypatch, operator):
    import math
    image = image_fixture((1, 3, 41, 79))
    level = torch.tensor([4.])
    monkeypatch.setattr(ops, "_rand", lambda image, rng, *shape: torch.zeros(shape))
    recorded = []
    def capture(value, theta):
        recorded.append(theta)
        return value
    monkeypatch.setattr(ops, "_affine", capture)
    ops._geometric(image, operator, level, torch.Generator())
    theta = recorded[0][0]
    point = torch.tensor([10., 5.])
    size = torch.tensor([79., 41.])
    actual = (theta[:, :2] @ (2 * point / size) + theta[:, 2]) * size / 2
    x, y = point.tolist()
    angle = math.radians(12)
    expected = {
        "rotate": (math.cos(angle)*x - math.sin(angle)*y, math.sin(angle)*x + math.cos(angle)*y),
        "shear_x": (x + 0.12*y, y), "shear_y": (x, y + 0.12*x),
        "translate_x": (x + 4*79/30, y), "translate_y": (x, y + 4*41/30),
    }[operator]
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("level", [0.1, 0.5, 1., 10.])
def test_motion_blur_always_spreads_an_impulse_at_positive_strength(level):
    image = torch.zeros(1, 3, 65, 81)
    image[:, :, 32, 40] = 1
    actual = ops._motion_blur(image, torch.tensor([level]), torch.Generator().manual_seed(7))
    assert actual[0, 0, 32, 40] < 0.95
    assert (actual.sum() - actual[:, :, 32, 40].sum()) > 0.1
    assert torch.isfinite(actual).all() and 0 <= actual.min() <= actual.max() <= 1


def test_brightness_preserves_hue_and_brightens_black():
    image = torch.tensor([0.1, 0.2, 0.4]).view(1, 3, 1, 1).expand(1, 3, 3, 5)
    expected = image * 1.5
    torch.testing.assert_close(ops._brightness(image, torch.tensor([2.])), expected)
    actual = ops._brightness(torch.zeros_like(image), torch.tensor([2.]))
    torch.testing.assert_close(actual, torch.full_like(image, 0.2))


def test_jpeg_dct_basis_and_chroma_roundtrip():
    dct = ops._dct_matrix(torch.device("cpu"))
    torch.testing.assert_close(dct @ dct.t(), torch.eye(8), atol=1e-6, rtol=0)
    assert len(ops._QY) == len(ops._QC) == 64
    image = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1).expand(1, 3, 17, 29)
    result = ops._jpeg_compression(image, torch.tensor([1.]))
    assert result.shape == image.shape
    torch.testing.assert_close(result, image, atol=0.02, rtol=0)


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("family,operator,severity", CASES)
def test_cuda_all_operators_replay_rng_range_layout_and_immutability(gpu_device, dtype, family, operator, severity):
    image = image_fixture(device=gpu_device, dtype=dtype).transpose(-1, -2)
    before = image.clone()
    cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    rng = torch.Generator(device=gpu_device).manual_seed(37)
    replay = torch.Generator(device=gpu_device).manual_seed(37)
    result = operator_function(family)(image, operator, severity=severity, rng=rng)
    repeated = operator_function(family, trusted=True)(image, operator, severity=severity, rng=replay)
    assert torch.equal(result, repeated) and torch.equal(rng.get_state(), replay.get_state())
    assert torch.equal(image, before)
    assert torch.equal(torch.get_rng_state(), cpu_state) and torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert result.shape == image.shape and result.device == image.device and result.dtype == dtype
    assert torch.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1


@GPU
@pytest.mark.parametrize("family", ["augmix", "c5"])
@pytest.mark.parametrize("severity", [True, False, "3", None, float("nan"), float("inf"), 0, 11])
def test_cuda_rejects_invalid_severity_before_consuming_rng(gpu_device, family, severity):
    rng = torch.Generator(device=gpu_device).manual_seed(7)
    state = rng.get_state().clone()
    operator = "equalize" if family == "augmix" else "gaussian_noise"
    with pytest.raises(ValueError):
        operator_function(family)(image_fixture(device=gpu_device), operator, severity=severity, rng=rng)
    assert torch.equal(rng.get_state(), state)


@GPU
@pytest.mark.parametrize("family", ["augmix", "c5"])
def test_cuda_rejects_bad_images_unknown_operators_and_nonlocal_rng(gpu_device, family):
    fn = operator_function(family)
    image = image_fixture(device=gpu_device)
    operator = "equalize" if family == "augmix" else "brightness"
    for rng in (None, torch.Generator()):
        with pytest.raises(ValueError, match="generator"):
            fn(image, operator, severity=3, rng=rng)
    rng = torch.Generator(device=gpu_device)
    with pytest.raises(ValueError):
        fn(image, "missing", severity=3, rng=rng)
    for invalid in (image[0], image[:0], image[:, :1], image.byte(), image[..., :1],
                    image * float("nan"), image * float("inf"), image - 2, image + 2):
        with pytest.raises(ValueError):
            fn(invalid, operator, severity=3, rng=rng)


@GPU
@pytest.mark.parametrize("family,operator,severity", OPERATORS)
def test_cuda_autocast_and_smallest_supported_canvas(gpu_device, family, operator, severity):
    image = image_fixture((1, 3, 2, 2), device=gpu_device)
    with torch.autocast("cuda", dtype=torch.float16):
        out = operator_function(family)(image, operator, severity=severity,
                                       rng=torch.Generator(device=gpu_device).manual_seed(42))
    assert out.shape == image.shape and out.dtype == image.dtype
    assert torch.isfinite(out).all() and 0 <= out.min() <= out.max() <= 1


@GPU
@NATIVE
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("family,operator,severity", OPERATORS)
def test_cuda_native_canvas_batch_two(gpu_device, dtype, family, operator, severity):
    image = image_fixture((2, 3, 1700, 2200), device=gpu_device, dtype=dtype)
    saved = image.clone()
    fn = operator_function(family)
    out = fn(image, operator, severity=severity, rng=torch.Generator(device=gpu_device).manual_seed(101))
    repeated = fn(image, operator, severity=severity, rng=torch.Generator(device=gpu_device).manual_seed(101))
    assert torch.equal(out, repeated) and torch.equal(image, saved)
    assert out.shape == image.shape and out.dtype == dtype and out.device == gpu_device
    assert torch.isfinite(out).all() and 0 <= out.min() <= out.max() <= 1


@GPU
@pytest.mark.parametrize("width", [1, 3])
def test_cuda_training_views_are_image_only_replayable_and_use_clean_residual(gpu_device, width):
    from core.pulse_hybrid import hybrid_jsd_views
    from util.pulse_training_contract import load_config
    config, _ = load_config(ROOT / "configs/train/pulse_augmix_gpu_three_ningbo.yaml", ROOT / "configs")
    class Renderer:
        calls = 0
        def render(self, wave):
            self.calls += 1
            return image_fixture((1, 3, 33, 49), device=wave.device)
    wave = torch.zeros((1, 5000, 12), device=gpu_device)
    renderer = Renderer()
    kwargs = dict(width=width, profile=None, seed=42, identity="record", renderer=renderer,
                  image_config=config["image_augmentation"])
    cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    views, trace = hybrid_jsd_views(wave, **kwargs)
    assert renderer.calls == 1 and len(views) == 3
    repeated, repeat_trace = hybrid_jsd_views(wave, **kwargs)
    assert trace == repeat_trace and all(torch.equal(a, b) for a, b in zip(views, repeated))
    assert trace["topology"] == "image_only_gpu_branches_v1" and trace["residual"] == "clean_render"
    for view in trace["views"]:
        assert view["waveform_operators"] == [] and len(view["image_chains"]) == width
        assert sum(view["weights"]) == pytest.approx(1)
    assert torch.equal(torch.get_rng_state(), cpu_state) and torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert all(v.device == gpu_device and torch.isfinite(v).all() for v in views)


@GPU
@NATIVE
def test_cuda_synthetic_ecg_render_operator_processor_and_downstream_gradient(gpu_device):
    from util.ecg_image_renderer import PulseECGTensorRenderer
    from util.evaluation.pulse_hybrid_development import c5_gpu_conditions_for, c5_gpu_image_seed
    background = torch.ones((3, 1700, 2200), device=gpu_device)
    renderer = PulseECGTensorRenderer(background)
    time = torch.arange(5000, device=gpu_device).float() / 500
    wave = torch.sin(time * (2 * torch.pi * 1.2)).view(1, -1, 1).expand(1, -1, 12)
    rgb = renderer.render(wave)
    assert rgb.shape == (1, 3, 1700, 2200) and rgb.min() < 0.5
    conditions = c5_gpu_conditions_for([{"condition_id": "clean", "operators": []}])
    for condition in conditions:
        viewed = rgb
        if condition["family"] == "image":
            viewed = ops.apply_gpu_c5_image_operator(rgb, condition["image_operator"], severity=5,
                rng=torch.Generator(device=gpu_device).manual_seed(c5_gpu_image_seed(42, "synthetic", condition)))
        pixels = renderer.preprocess_for_pulse(viewed).clone().half()
        assert pixels.shape == (1, 5, 3, 336, 336) and torch.isfinite(pixels).all()
        # This checks tensor/autograd interoperability, not PULSE model admission.
        weight = torch.ones((3,), device=gpu_device, requires_grad=True)
        (pixels.float().square().mean((0, 1, 3, 4)) * weight).sum().backward()
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().min() > 0
