"""Hybrid image AugMix invariants; no model downloads or GPU allocation."""
from pathlib import Path

import pytest
import torch

from core.image_corruption import IMAGE_OPERATORS, apply_image_operator, validate_image_config
from core.pulse_hybrid import hybrid_jsd_views
from core.pulse_visual import answer_jsd_loss
from util.augmentations.profile import load_augmentation_profile

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = dict(operators=list(IMAGE_OPERATORS), strength=0.5, waveform_strength=0.5, jsd_weight=12.0)


@pytest.mark.parametrize("operator", IMAGE_OPERATORS)
def test_image_operator_identity_range_replay_and_immutability(operator):
    image = torch.linspace(0, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
    saved, rng_state = image.clone(), torch.get_rng_state().clone()
    def apply(strength):
        return apply_image_operator(image, operator, strength=strength, rng=torch.Generator().manual_seed(7))
    assert torch.equal(apply(0), image)
    result = apply(1)
    assert torch.equal(result, apply(1)) and torch.equal(image, saved)
    assert torch.equal(rng_state, torch.get_rng_state())
    assert result.shape == image.shape and torch.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1


def test_grid_fade_preserves_achromatic_trace():
    image = torch.full((1, 3, 12, 16), 0.2)
    actual = apply_image_operator(image, "grid_fade", strength=1, rng=torch.Generator().manual_seed(8))
    assert torch.equal(actual, image)


class Renderer:
    def __init__(self):
        self.waves = []
    def render(self, wave):
        self.waves.append(wave.clone())
        return wave[:, :96, :3].transpose(1, 2).reshape(1, 3, 8, 12).sigmoid()


@pytest.mark.parametrize("width", [0, 1, 3])
def test_hybrid_renders_once_per_view_not_chain(width):
    wave = torch.linspace(-1, 1, 60000).reshape(1, 5000, 12)
    saved, rng_state = wave.clone(), torch.get_rng_state().clone()
    renderer = Renderer()
    views, trace = hybrid_jsd_views(wave, width=width,
        profile=load_augmentation_profile(ROOT / "configs/augmentation/operators.yaml"),
        seed=7, identity="record", renderer=renderer, image_config=OPTIONS)
    assert len(renderer.waves) == len(views) == (3 if width else 1)
    assert torch.equal(saved, wave) and torch.equal(rng_state, torch.get_rng_state())
    assert all(torch.isfinite(v).all() and 0 <= v.min() <= v.max() <= 1 for v in views)
    for view in trace["views"]:
        assert len(view["image_chains"]) == width
        assert sum(view["weights"]) == pytest.approx(1)
        assert all(1 <= len(chain) <= 3 for chain in view["image_chains"])


def test_widths_share_waveforms_and_first_image_chain():
    wave = torch.zeros(1, 5000, 12)
    rs, ts = [], []
    for width in (1, 3):
        renderer = Renderer()
        _, trace = hybrid_jsd_views(wave, width=width,
            profile=load_augmentation_profile(ROOT / "configs/augmentation/operators.yaml"),
            seed=4, identity="record", renderer=renderer, image_config=OPTIONS)
        rs.append(renderer); ts.append(trace)
    for a, b in zip(rs[0].waves, rs[1].waves):
        assert torch.equal(a, b)
    for a, b in zip(ts[0]["views"], ts[1]["views"]):
        assert a["image_chains"][0] == b["image_chains"][0]
        assert a["m"] == b["m"]


@pytest.mark.parametrize("key,value", [("strength", float("nan")), ("waveform_strength", -1), ("jsd_weight", float("inf"))])
def test_invalid_options(key, value):
    with pytest.raises(ValueError):
        validate_image_config({**OPTIONS, key: value})


def test_configurable_jsd_keeps_clean_ce_and_three_view_gradients():
    logits = [torch.randn(2, 7, requires_grad=True) for _ in range(3)]
    loss, ce, jsd = answer_jsd_loss(logits, torch.tensor([1, 2]), jsd_weight=3)
    torch.testing.assert_close(loss, ce + 3 * jsd)
    assert all(g.abs().sum() > 0 for g in torch.autograd.grad(loss, logits))


def test_approximate_augmix_is_readable_history_but_not_executable():
    from core.image_augmix_c import AUGMIX_TRAIN_OPERATORS
    legacy = {**OPTIONS, "operators": list(AUGMIX_TRAIN_OPERATORS)}
    validate_image_config(legacy)
    with pytest.raises(ValueError, match="historical only"):
        validate_image_config(legacy, for_execution=True)


@pytest.mark.parametrize("width", [0, 1, 3])
def test_reference_augmix_keeps_hybrid_topology_and_has_explicit_identity(width):
    from core.image_augmix_c import AUGMIX_TRAIN_OPERATORS, AUGMIX_IMPLEMENTATION
    options = {k: v for k, v in OPTIONS.items() if k != "strength"}
    options.update(operators=list(AUGMIX_TRAIN_OPERATORS), implementation=AUGMIX_IMPLEMENTATION, severity=3)
    validate_image_config(options, for_execution=True)
    def render():
        return hybrid_jsd_views(torch.zeros(1, 5000, 12), width=width,
            profile=load_augmentation_profile(ROOT / "configs/augmentation/operators.yaml"),
            seed=7, identity="record", renderer=Renderer(), image_config=options)
    views, trace = render()
    assert len(views) == (3 if width else 1)
    assert all(torch.equal(a, b) for a, b in zip(views, render()[0]))
    assert trace["residual"] == "corrupted_waveform_render"


def test_legacy_training_fails_before_output_or_model_loading(tmp_path):
    from core.pulse_finetune import run
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="historical only"):
        run(ROOT / "configs/train/pulse_c15_augmix_three_ningbo.yaml", ROOT / "configs", output)
    assert not output.exists()


def test_reference_training_configs_are_matched_and_close_over_the_same_inputs(tmp_path):
    from boot_scripts.run_experiment import load_experiment_plan
    from util.pulse_training_contract import load_config
    common = None
    for arm, width in (("clean", 0), ("single", 1), ("three", 3)):
        name = f"pulse_augmix_reference_{arm}_ningbo"
        config, refs = load_config(ROOT / f"configs/train/{name}.yaml", ROOT / "configs")
        validate_image_config(config["image_augmentation"], for_execution=True)
        assert config.pop("width") == width
        identity = (config, refs)
        common = identity if common is None else common
        assert identity == common
        plan = load_experiment_plan(ROOT / f"configs/experiments/{name}.yaml", run_dir=tmp_path / arm)
        assert not plan.run_dir.exists() and not plan.describe()["loads_model_or_gpu"]
