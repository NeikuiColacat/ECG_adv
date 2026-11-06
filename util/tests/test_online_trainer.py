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
NAMES = ['a0_clean_v1', 'a1_corrupt_ft_rot4_v1', 'augmix_simclr_lhat', 'augmix_simclr_matched_no_vae']
MAIN = 'augmix_simclr_lhat'
MATCHED = 'augmix_simclr_matched_no_vae'

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


def test_stage1_uses_resolved_augmix_constants_and_hash_free_legacy_seed(monkeypatch) -> None:
    recipe = _recipe("augmix_simclr_lhat.yaml")
    runtime = build_method_runtime(recipe, model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT, latent_pool=object(), decoder=torch.nn.Identity())
    augmix = runtime.augmix_config; assert augmix is not None
    captured = []
    def capture(device, namespace, *identity, config_path):
        seed = derive_seed(namespace, *identity, config_path=config_path)
        captured.append((seed, identity)); return torch.Generator(device=device).manual_seed(seed)
    patches = {
        "make_torch_generator": capture,
        "_cache_k500_logits": lambda *a, **k: {},
        "_feature_width": lambda *a, **k: 4,
        "_head_parameters": lambda *a, **k: (),
        "prepare_canonical_model_input": lambda raw, *a, **k: raw.mean((1, 2)).unsqueeze(1),
        "generate_two_chain_augmix_strong_view": lambda raw, **k: SimpleNamespace(mixed_raw=raw * .5),
        "_forward_logits_and_features": lambda model, x, spec: (model(x), model(x)),
        "_teacher_logits_for_hashes": lambda cache, hashes, **k: torch.zeros(len(hashes), 4),
        "_simclr_nt_xent": lambda clean, strong, **k: (clean - strong).square().mean(),
        "_weighted_logit_anchor_loss": lambda logits, teacher, weights: logits.square().mean(),
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(trainer, name, replacement)
    loader = [{"waveform": torch.full((2, 1000, 12), float(step + 1)),
               "hash_id": (f"batch-{step}-a", f"batch-{step}-b")} for step in range(2)]
    summary = trainer._run_augmix_stage1(torch.nn.Linear(1, 4), loader,
        spec=SimpleNamespace(name="efficientnet1dv2"), device=torch.device("cpu"),
        recipe=recipe, augmix_config=augmix, center="ningbo", base_seed=20260501,
        resolved={"stage1_steps": 2, "stage1_learning_rate": 1e-3,
                  "stage1_weight_decay": 0., "stage1_gradient_clip_norm": 1.},
        pos_weight=None,
        normalization_epsilon=1e-6, amp_enabled=False, amp_dtype=torch.bfloat16)
    assert [seed for seed, _ in captured] == [3967304348, 2995841993]
    assert [identity for _, identity in captured] == [
        ("augmix_simclr_lhat", "ningbo", "efficientnet1dv2", "base_seed=20260501", f"stage1_step={i}")
        for i in range(2)]
    assert (summary["simclr_temperature"], summary["augmix_internal_chains"],
            summary["augmix_dirichlet_alpha"], summary["augmix_beta_alpha"],
            summary["logit_anchor_weight"], summary["ptbxl_replay_weight"]) == (
                .5, 2, .5, .5, 5., 0.)


def test_simclr_mainline_and_matched_control_share_supervised_weights():
    for name in (MAIN, MATCHED):
        recipe = _recipe(name + ".yaml")
        assert recipe.scientific_contract["stage1"]["objective"] == "simclr"
        assert recipe.scientific_contract["stage1"]["classifier_head_trainable"] is False
        steps = trainer._method_exposure_steps(recipe)
        assert [step.loss_scale for step in steps[:5]] == [.5, .125, .125, .125, .125]
    assert trainer._method_exposure_steps(_recipe(MAIN + ".yaml"))[-1].loss_scale == 2.0
    assert len(trainer._method_exposure_steps(_recipe(MATCHED + ".yaml"))) == 5
