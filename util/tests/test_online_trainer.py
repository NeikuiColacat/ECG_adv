"""Numerical contracts for the retained optimizer/exposure owner."""

import copy
import hashlib
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import core.online_trainer as trainer
from core.methods.registry import load_recipe_spec
from core.methods.contracts import ViewBundle, WaveformView
from core.methods.runtime import build_method_runtime, _derive_seed
from models.contracts import EFFICIENTNET1DV2_SPEC
from util.random_seed import derive_seed, load_random_seed_config

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
RECIPES = CONFIG_ROOT / "train/methods"
ONLINE_CONFIG = trainer.DEFAULT_ONLINE_CONFIG_PATH
NAMES = ['a0_clean_v1', 'a1_corrupt_ft_rot4_v1', 'a1_rot4_two_chain_balanced_jsd1p5', 'a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2']
MAIN = 'a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2'
MATCHED = 'a1_rot4_two_chain_balanced_jsd1p5'

def _recipe(filename: str):
    return load_recipe_spec(RECIPES / filename)


@pytest.mark.parametrize("name", NAMES)
def test_five_epoch_rotation_and_family_balanced_bn(name):
    recipe = _recipe(name + ".yaml")
    plans = [trainer._method_exposure_steps(recipe, epoch=e) for e in range(1, 6)]
    if name == "a0_clean_v1":
        assert [(step.name, step.loss_scale) for step in plans[0]] == [("base", 1.0)]
        assert trainer._build_batch_norm_momentum_plan(
            torch.nn.BatchNorm1d(2), recipe, exposure_steps=plans[0]) is None
        return
    indices = [step.composition_index for plan in plans for step in plan
               if step.name.startswith("corruption_")]
    assert sorted(indices) == list(range(20))
    bn = torch.nn.BatchNorm1d(2, momentum=.1)
    plan = trainer._build_batch_norm_momentum_plan(bn, recipe, exposure_steps=plans[0])
    weights = [.5, .125, .125, .125, .125] + [0.] * (len(plans[0]) - 5)
    assert plan.exposure_weights == pytest.approx(weights)
    contributions = []
    for momentum in plan.schedules[0]:
        contributions = [x * (1 - momentum) for x in contributions] + [momentum]
    assert contributions == pytest.approx([.1 * value for value in weights])
    with pytest.raises(ValueError, match="positive integer"):
        trainer._method_exposure_steps(recipe, epoch=0)


@pytest.mark.parametrize("name", NAMES)
def test_online_parameter_overrides_preserve_scheduler_horizon(name):
    config = trainer.load_online_train_config(ONLINE_CONFIG)
    defaults, _ = trainer.resolve_online_training_parameters(config, "efficientnet1dv2")
    resolved, supplied = trainer.resolve_online_training_parameters(
        config, "efficientnet1dv2", {"epochs": 2})
    assert resolved["epochs"] == 2
    assert resolved["scheduler_horizon_epochs"] == defaults["scheduler_horizon_epochs"]
    assert supplied == {"epochs": 2}
    with pytest.raises(ValueError, match="unknown online training parameters"):
        trainer.resolve_online_training_parameters(config, "efficientnet1dv2", {"typo": 1})


def test_auxiliary_preserves_bn_rng_through_backward_but_accumulates_gradients():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.3), torch.nn.Linear(3, 5)).train()
    values = torch.randn(4, 3)
    model(values).sum().backward()
    before = {key: value.clone() for key, value in model.named_buffers()}
    rng = torch.get_rng_state().clone()
    previous = model[2].weight.grad.clone()
    with trainer._preserve_batch_norm_buffers(model), trainer._preserve_torch_rng(torch.device("cpu")):
        model(values).square().mean().backward()
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, before[key]) for key, value in model.named_buffers())
    assert not torch.equal(model[2].weight.grad, previous)


def test_empty_lhat_auxiliary_is_an_empty_gradient_sum() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD((parameter,), lr=.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    (parameter.square()).backward()
    accumulated = parameter.grad.clone()
    empty = trainer._ObjectiveBatch(
        total=torch.tensor(0.0), raw_terms={}, weighted_terms={},
        valid_counts={"lhat_direct_bce": 0})
    returned = trainer._backward_objective(
        empty, loss_scale=2.0, scaler=scaler, empty_lhat_auxiliary=True)
    assert returned.item() == 0.0 and torch.equal(parameter.grad, accumulated)
    optimizer.step()
    assert parameter.item() == pytest.approx(1.6)

    detached = trainer._ObjectiveBatch(
        total=torch.tensor(1.0), raw_terms={}, weighted_terms={},
        valid_counts={"lhat_direct_bce": 1})
    with pytest.raises(RuntimeError, match="detached"):
        trainer._backward_objective(
            detached, loss_scale=2.0, scaler=scaler,
            empty_lhat_auxiliary=False)


def test_lhat_clean_replacement_has_fixed_mass_and_clean_rejection_fallback() -> None:
    recipe = _recipe(
        "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml"
    )
    steps = trainer._method_exposure_steps(recipe, epoch=1)
    assert [step.loss_scale for step in steps] == pytest.approx(
        [0.05, 0.0625, 0.0625, 0.0625, 0.0625, 0.5, 0.2]
    )
    assert sum(step.loss_scale for step in steps) == pytest.approx(1.0)
    plan = trainer._build_batch_norm_momentum_plan(
        torch.nn.BatchNorm1d(2, momentum=0.1),
        recipe,
        exposure_steps=steps,
    )
    assert plan is not None
    assert plan.exposure_weights == pytest.approx(
        [0.5, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0]
    )

    clean = torch.randn(2, 1000, 12)
    hard = clean.roll(13, dims=1)
    labels = torch.tensor(
        [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32
    )
    sample_ids = ("accepted", "fallback")
    bundle = ViewBundle(
        values={
            "clean_view": WaveformView(
                name="clean_view",
                waveform=clean,
                labels=labels,
                sample_ids=sample_ids,
            ),
            "lhat_view": WaveformView(
                name="lhat_view",
                waveform=hard,
                labels=labels,
                sample_ids=sample_ids,
                valid_mask=torch.tensor([True, False]),
            ),
        }
    )

    class Head(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs[:, :5, 0]

    model = Head()
    objective = trainer._compute_objective(
        recipe=recipe,
        bundle=bundle,
        model=model,
        spec=EFFICIENTNET1DV2_SPEC,
        normalization_epsilon=1.0e-6,
        pos_weight=None,
        objective_term_names=("lhat_direct_bce",),
    )
    clean_logits = model(
        trainer.prepare_canonical_model_input(
            clean, EFFICIENTNET1DV2_SPEC, epsilon=1.0e-6
        )
    )
    hard_logits = model(
        trainer.prepare_canonical_model_input(
            hard[:1], EFFICIENTNET1DV2_SPEC, epsilon=1.0e-6
        )
    )
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            hard_logits, labels[:1], reduction="sum"
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            clean_logits[1:], labels[1:], reduction="sum"
        )
    ) / 10.0
    assert objective.valid_counts == {"lhat_direct_bce": 2}
    assert objective.total == pytest.approx(expected)


def test_jsd_chain_mean_and_clean_consistency_keep_mixed_forward():
    recipe = _recipe(MATCHED + ".yaml")
    clean = torch.randn(2, 1000, 12)
    labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    raw = {"clean_view": clean, "augmix_view": clean.roll(1, 1),
           "augmix_chain1_view": clean.roll(2, 1), "augmix_chain2_view": clean.roll(3, 1)}
    views = {name: WaveformView(name, value, labels, ("a", "b")) for name, value in raw.items()}
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
        def forward(self, inputs):
            self.seen.append(inputs)
            return inputs[:, :5, 0]
    model = Head()
    objective = trainer._compute_objective(
        recipe=recipe, bundle=ViewBundle(views), model=model,
        spec=EFFICIENTNET1DV2_SPEC, normalization_epsilon=1e-6,
        pos_weight=None, objective_term_names=recipe.scientific_contract["augmix_auxiliary"]["objective_terms"])
    assert len(model.seen) == 4
    clean_logits, mixed_logits, first, second = [value[:, :5, 0] for value in model.seen]
    supervised = torch.stack([F.binary_cross_entropy_with_logits(value, labels) for value in (first, second)]).mean()
    p = torch.stack([value.sigmoid().clamp(1e-6, 1 - 1e-6) for value in (clean_logits, first, second)])
    mean = p.mean(0).clamp(1e-6, 1 - 1e-6)
    jsd = (p * (p.log() - mean.log()) + (1-p) * ((1-p).log() - (1-mean).log())).mean()
    torch.testing.assert_close(objective.total, supervised + 1.5 * jsd)
