"""PIL/reference parity and regressions for the rejected September 18 proxies."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import pytest
import torch

from core.image_augmix_c import (
    AUGMIX_TRAIN_OPERATORS, C_IMAGE_OPERATORS, apply_augmix_image_operator,
    apply_c_image_operator, c_reference_identity,
)


def pixels():
    return torch.rand((1, 3, 32, 48), generator=torch.Generator().manual_seed(123))


def to_pil(image):
    return Image.fromarray((image[0].permute(1, 2, 0) * 255).round().byte().numpy())


def from_pil(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float() / 255


def numpy_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


@pytest.mark.parametrize("operator", AUGMIX_TRAIN_OPERATORS)
def test_augmix_matches_original_pil_formulas(operator):
    image, seed = pixels(), 17
    original = to_pil(image)
    rng = torch.Generator().manual_seed(seed)
    level = 0.1 + 2.9 * float(torch.rand((), generator=rng))
    sign = -1 if float(torch.rand((), generator=rng)) > 0.5 else 1
    matrices = {
        "shear_x": (1, sign * level * 0.3 / 10, 0, 0, 1, 0),
        "shear_y": (1, 0, 0, sign * level * 0.3 / 10, 1, 0),
        "translate_x": (1, 0, sign * int(level * 48 / 30), 0, 1, 0),
        "translate_y": (1, 0, 0, 0, 1, sign * int(level * 32 / 30)),
    }
    if operator == "autocontrast":
        expected = ImageOps.autocontrast(original)
    elif operator == "equalize":
        expected = ImageOps.equalize(original)
    elif operator == "posterize":
        expected = ImageOps.posterize(original, 4 - int(level * 4 / 10))
    elif operator == "solarize":
        expected = ImageOps.solarize(original, 256 - int(level * 256 / 10))
    elif operator == "rotate":
        expected = original.rotate(sign * int(level * 30 / 10), resample=Image.Resampling.BILINEAR)
    else:
        expected = original.transform(original.size, Image.Transform.AFFINE,
            matrices[operator], resample=Image.Resampling.BILINEAR)
    before, state = image.clone(), torch.get_rng_state().clone()
    actual = apply_augmix_image_operator(image, operator, severity=3, rng=torch.Generator().manual_seed(seed))
    assert torch.equal(actual, from_pil(expected))
    assert torch.equal(image, before) and torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("value", [0., 1.])
@pytest.mark.parametrize("operator", ["equalize", "autocontrast"])
def test_uniform_background_never_becomes_fake_spatial_texture(operator, value):
    image = torch.full((1, 3, 32, 48), value)
    actual = apply_augmix_image_operator(image, operator, severity=3, rng=torch.Generator().manual_seed(7))
    assert torch.equal(actual, image)


@pytest.mark.parametrize("operator", C_IMAGE_OPERATORS)
@pytest.mark.parametrize("severity", range(1, 6))
def test_c15_all_operators_replay_and_leave_global_rng_and_input_unchanged(operator, severity):
    c_reference_identity()
    image = pixels()
    before, state, numpy_state = image.clone(), torch.get_rng_state().clone(), np.random.get_state()
    def apply():
        return apply_c_image_operator(image, operator, severity=severity, rng=torch.Generator().manual_seed(7))
    actual = apply()
    assert torch.equal(actual, apply()) and torch.equal(before, image)
    assert torch.equal(state, torch.get_rng_state()) and numpy_state_equal(numpy_state, np.random.get_state())
    assert actual.shape == image.shape and actual.dtype == image.dtype
    assert torch.isfinite(actual).all() and 0 <= actual.min() <= actual.max() <= 1
    assert not torch.equal(actual, image)


@pytest.mark.parametrize("severity", range(1, 6))
def test_pixelate_is_real_downsampling_and_jpeg_is_real_codec(severity):
    image = pixels()
    original = to_pil(image)
    scale = (0.6, 0.5, 0.4, 0.3, 0.25)[severity - 1]
    expected = original.resize((int(48 * scale), int(32 * scale)), Image.Resampling.BOX)
    expected = expected.resize(original.size, Image.Resampling.NEAREST)
    actual = apply_c_image_operator(image, "pixelate", severity=severity, rng=torch.Generator().manual_seed(3))
    assert torch.equal(actual, from_pil(expected))
    data = BytesIO()
    original.save(data, "JPEG", quality=(25, 18, 15, 10, 7)[severity - 1])
    actual = apply_c_image_operator(image, "jpeg_compression", severity=severity, rng=torch.Generator().manual_seed(3))
    assert torch.equal(actual, from_pil(Image.open(data)))


def test_reference_failure_also_restores_numpy_rng_and_library(monkeypatch):
    c_reference_identity()
    from imagecorruptions import corruptions
    before, gaussian = np.random.get_state(), corruptions.gaussian
    def fail(*args, **kwargs):
        np.random.rand(3)
        raise RuntimeError("reference failure")
    monkeypatch.setattr(corruptions, "shot_noise", fail)
    with pytest.raises(RuntimeError, match="reference failure"):
        apply_c_image_operator(pixels(), "shot_noise", severity=3, rng=torch.Generator().manual_seed(7))
    assert numpy_state_equal(before, np.random.get_state())
    assert corruptions.gaussian is gaussian


@pytest.mark.parametrize("severity", range(1, 6))
@pytest.mark.parametrize("seed", [0, 7, 20260919])
def test_accelerated_glass_matches_pinned_reference_bytes_and_rng(severity, seed):
    from core.image_augmix_c import _glass_blur_exact, _reference_gaussian
    from imagecorruptions import corruptions
    original = to_pil(pixels())
    before, gaussian = np.random.get_state(), corruptions.gaussian
    try:
        corruptions.gaussian = _reference_gaussian
        np.random.seed(seed)
        expected = corruptions.glass_blur(original, severity=severity)
        expected_rng = np.random.get_state()
        np.random.seed(seed)
        actual = _glass_blur_exact(original, severity)
        assert np.array_equal(expected, actual)
        assert numpy_state_equal(expected_rng, np.random.get_state())
    finally:
        np.random.set_state(before)
        corruptions.gaussian = gaussian


def test_training_and_test_pools_do_not_overlap():
    assert len(AUGMIX_TRAIN_OPERATORS) == 9 and len(C_IMAGE_OPERATORS) == 15
    assert not set(AUGMIX_TRAIN_OPERATORS) & set(C_IMAGE_OPERATORS)


@pytest.mark.parametrize("severity", [True, 0, 11, float("nan"), float("inf")])
def test_reference_severity_rejects_invalid_values(severity):
    with pytest.raises(ValueError):
        apply_augmix_image_operator(pixels(), "equalize", severity=severity, rng=torch.Generator())
