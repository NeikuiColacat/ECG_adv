"""Paper-image contracts; CPU fixtures do not establish GPU or clinical validity."""
import json
import os

import pytest
import torch

from core.paper_ecg import (PAPER_EVAL_OPERATORS, PAPER_TRAIN_OPERATORS,
    PAPER_HELDOUT_OPERATORS, apply_paper_operator, paper_conditions)


def paper(device="cpu", dtype=torch.float32):
    image = torch.ones(2, 3, 40, 64, device=device, dtype=dtype)
    image[:, 1:, ::8, :] = 0.75
    image[:, 1:, :, ::8] = 0.75
    image[:, :, 19:21, 5:59] = 0.1
    return image


@pytest.mark.parametrize("operator", PAPER_EVAL_OPERATORS)
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_paper_replay_and_range_preserve_source_and_global_rng(operator, severity):
    image = paper()
    source, global_rng = image.clone(), torch.random.get_rng_state().clone()
    def apply():
        return apply_paper_operator(image, operator, severity=severity, rng=torch.Generator().manual_seed(37))
    a, b = apply(), apply()
    assert torch.equal(a, b) and torch.equal(image, source)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert a.shape == image.shape and a.dtype == image.dtype and a.device == image.device
    assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert not torch.equal(a, image)


@pytest.mark.parametrize("operator", PAPER_EVAL_OPERATORS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_zero_is_exact_identity_without_rng_consumption(operator, dtype):
    image = paper(dtype=dtype)
    rng = torch.Generator().manual_seed(3)
    state = rng.get_state().clone()
    result = apply_paper_operator(image, operator, severity=0, rng=rng)
    assert torch.equal(result, image) and result.data_ptr() != image.data_ptr()
    assert torch.equal(state, rng.get_state())


@pytest.mark.parametrize("operator", PAPER_TRAIN_OPERATORS)
def test_paper_training_images_support_backward(operator):
    image = paper().requires_grad_()
    result = apply_paper_operator(image, operator, severity=2, rng=torch.Generator().manual_seed(19))
    result.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


def test_stress_conditions_keep_seen_and_heldout_families_separate():
    assert set(PAPER_TRAIN_OPERATORS).isdisjoint(PAPER_HELDOUT_OPERATORS)
    rows = paper_conditions()
    assert len(rows) == len({r["condition_id"] for r in rows}) == 80
    assert {r["image_severity"] for r in rows} == {1, 2, 3, 4, 5}
    assert all((r["stress_group"] == "seen_family") == (r["image_operator"] in PAPER_TRAIN_OPERATORS) for r in rows)
    for levels in ([], [0], [6], [True], [1, 1]):
        with pytest.raises(ValueError):
            paper_conditions(levels)


def test_grid_fade_keeps_achromatic_ink_and_yellowing_keeps_black():
    image = paper()
    faded = apply_paper_operator(image, "grid_fade", severity=5, rng=torch.Generator().manual_seed(1))
    gray = (image[:, :1] == image[:, 1:2]) & (image[:, :1] == image[:, 2:3])
    assert torch.equal(faded[gray.expand_as(image)], image[gray.expand_as(image)])
    black = torch.zeros_like(image)
    assert torch.equal(apply_paper_operator(black, "yellowing", severity=5, rng=torch.Generator()), black)


def test_jpeg_constants_are_reused_without_mutation():
    from core.image_augmix_gpu import _dct_matrix, _jpeg_quantization_tables
    device = torch.device("cpu")
    matrix, tables = _dct_matrix(device), _jpeg_quantization_tables(device)
    saved_matrix, saved_tables = matrix.clone(), tables.clone()
    assert _dct_matrix(device).data_ptr() == matrix.data_ptr()
    assert _jpeg_quantization_tables(device).data_ptr() == tables.data_ptr()
    apply_paper_operator(paper(), "jpeg_compression", severity=3, rng=torch.Generator().manual_seed(1))
    assert torch.equal(matrix, saved_matrix) and torch.equal(tables, saved_tables)


@pytest.mark.parametrize("severity", [-1, 6, float("nan"), True])
def test_paper_rejects_invalid_severity(severity):
    with pytest.raises(ValueError):
        apply_paper_operator(paper(), "crease", severity=severity, rng=torch.Generator())


def test_paper_rejects_nonfinite_pixels_and_missing_rng():
    image = paper()
    image[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        apply_paper_operator(image, "crease", severity=1, rng=torch.Generator())
    with pytest.raises(ValueError, match="generator"):
        apply_paper_operator(paper(), "crease", severity=1, rng=None)


@pytest.mark.parametrize("width", [0, 1, 3])
def test_paper_training_dispatch_renders_once_and_records_identity(monkeypatch, width):
    from core import image_augmix_gpu
    from core.image_corruption import image_augmentation_protocol, validate_image_config
    from core.paper_ecg import validate_paper_image
    from core.pulse_hybrid import hybrid_jsd_views
    monkeypatch.setattr(image_augmix_gpu, "validate_gpu_image", validate_paper_image)
    config = dict(implementation="paper_ecg_torch_v1", operators=list(PAPER_TRAIN_OPERATORS),
                  severity=2, waveform_strength=0.0, jsd_weight=3.0)
    validate_image_config(config, for_execution=True)
    class Renderer:
        calls = 0
        def render(self, wave):
            self.calls += 1
            return paper()[:1]
    renderer = Renderer()
    wave = torch.zeros(1, 5000, 12)
    views, trace = hybrid_jsd_views(wave, width=width, profile=None, seed=17, identity="fixture",
        renderer=renderer, image_config=config)
    assert renderer.calls == 1 and len(views) == (3 if width else 1)
    assert trace["topology"] == "image_only_gpu_branches_v1" and trace["residual"] == "clean_render"
    assert all(not view["waveform_operators"] for view in trace["views"])
    assert all(op in PAPER_TRAIN_OPERATORS for view in trace["views"] for chain in view["image_chains"] for op in chain)
    protocol = image_augmentation_protocol(config, width)
    assert protocol["image_gpu"]["implementation"] == "paper_ecg_torch_v1"
    assert protocol["jsd_weight"] == (3.0 if width else 0.0)
    for bad in ({**config, "waveform_strength": 0.1}, {**config, "operators": list(PAPER_HELDOUT_OPERATORS)}):
        with pytest.raises(ValueError):
            validate_image_config(bad, for_execution=True)


@pytest.mark.skipif(os.environ.get("PULSE_GPU_OPS_NATIVE") != "1", reason="explicit native GPU admission required")
def test_native_paper_gpu_replay_and_throughput():
    from core.paper_ecg import validate_paper_image
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
    image = torch.ones(1, 3, 2200, 1700, device="cuda", dtype=torch.float16)
    image[:, 1:, ::20, :] = 0.8
    image[:, 1:, :, ::20] = 0.8
    image[:, :, 1100:1103, 20:1680] = 0.1
    validate_paper_image(image)
    rows = []
    for operator in PAPER_EVAL_OPERATORS:
        def apply():
            return apply_paper_operator(image, operator, severity=3,
                rng=torch.Generator(device=image.device).manual_seed(71), validate=False)
        expected = apply()
        assert torch.equal(expected, apply())
        assert expected.dtype == image.dtype and expected.device == image.device
        assert torch.isfinite(expected).all() and expected.min() >= 0 and expected.max() <= 1
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        for _ in range(3):
            actual = apply()
        end.record()
        end.synchronize()
        rows.append({"operator": operator, "milliseconds": start.elapsed_time(end) / 3,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        assert torch.equal(actual, expected)
    print(json.dumps({"paper_gpu_admission": rows, "shape": list(image.shape),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "timing": "warm_operator_only_three_repeats"}))
