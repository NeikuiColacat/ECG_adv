"""Contracts for the four retained, code-owned adaptation recipes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import core.lhat as lhat
import core.methods.runtime as method_runtime
from core.lhat import AttackThenContractDiagnostics
from core.methods.registry import load_recipe_spec
from core.methods.runtime import build_method_runtime, _scoped_lhat_diagnostics
from core.online_trainer import _diagnostic_sample_summary

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
RECIPES = CONFIG_ROOT / "train/methods"
IDENTITY = ("center=ningbo", "model=efficientnet1dv2", "epoch=2")
RECIPE_HASHES = {'a0_clean_v1': '32296a2419575e66a4f702deaa5147b328b945b72b812e52019d8d3b9fdccf98', 'a1_corrupt_ft_rot4_v1': 'de0224de08ebf4658828f6ed0fa5ebd414d608fb7ab3430abaddfae8d33de44f', 'augmix_simclr_lhat': '6245e89f572b49bd24f5d688bda7d3f8b16fa4972051353afa311d007cd9f675', 'augmix_simclr_matched_no_vae': 'b2b140ef2e4d440769b14f35b02ca61ba3951502fd7699a276a821ee202ae0f0'}
DESCRIPTION_SHA256 = '4039f4f1915e284fb387caa4476753a89fad11e23efb486608fc5ef194b42fbd'

def _batch(size: int = 2):
    waveform = torch.linspace(-1.0, 1.0, size * 12000).reshape(size, 1000, 12)
    labels = torch.zeros(size, 5); labels[:, 3] = 1
    if size == 2:
        labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    return waveform, labels, tuple(f"g{i}" for i in range(size))


def _runtime(filename: str, **resources: object):
    return build_method_runtime(load_recipe_spec(RECIPES / filename),
        model_name="efficientnet1dv2", config_root=CONFIG_ROOT, **resources)


def test_exact_four_recipe_descriptions_and_scientific_hashes():
    assert {path.stem for path in RECIPES.glob("*.yaml")} == set(RECIPE_HASHES)
    descriptions = {}
    for name, expected in RECIPE_HASHES.items():
        path = RECIPES / (name + ".yaml")
        recipe = load_recipe_spec(path)
        assert recipe.recipe_sha256 == expected
        description = recipe.describe()
        description["source_path"] = path.relative_to(REPO).as_posix()
        descriptions[path.name] = description
    encoded = json.dumps(descriptions, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == DESCRIPTION_SHA256


@pytest.mark.parametrize("change", ["unknown", "dynamic", "missing_resource", "wrong_kind"])
def test_recipe_loader_rejects_unsupported_declarations(change):
    payload = yaml.safe_load((RECIPES / "a1_corrupt_ft_rot4_v1.yaml").read_text())
    if change == "unknown":
        payload["recipe"]["id"] = "retired_experiment"
    elif change == "dynamic":
        payload["resources"]["corruption_rng"]["module"] = "arbitrary.plugin"
    elif change == "missing_resource":
        del payload["resources"]["corruption_rng"]
    else:
        payload["recipe"]["kind"] = "two_stage_jsd"
    with pytest.raises(ValueError):
        load_recipe_spec(payload)


def test_rotating_corruption_is_deterministic_and_preserves_origin_binding():
    waveform, labels, hashes = _batch()
    runtime = _runtime("a1_corrupt_ft_rot4_v1.yaml")
    kwargs = dict(clean_raw=waveform, targets=labels, hash_ids=hashes,
                  classifier=object(), base_seed=20260501, rng_identity=IDENTITY,
                  composition_indices=torch.tensor([7, 7]), composition_index_hint=7,
                  objective_term_names=("corrupted_bce",))
    first = runtime.generate(**kwargs).bundle
    second = runtime.generate(**kwargs).bundle
    value = first.require("corrupted_view")
    assert torch.equal(value.waveform, second.require("corrupted_view").waveform)
    assert torch.equal(value.labels, labels) and value.sample_ids == hashes
    assert value.waveform.shape == waveform.shape


def test_canonical_nonfinite_waveform_is_masked_without_host_failfast() -> None:
    waveform, labels, hashes = _batch(); runtime = _runtime("a0_clean_v1.yaml")
    clean = waveform.index_put(
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
        torch.tensor(float("nan")),
    )
    generated = runtime.generate(clean_raw=clean, targets=labels, hash_ids=hashes,
        classifier=object(), base_seed=20260501, rng_identity=IDENTITY)
    assert generated.bundle.require("clean_view").valid_mask.tolist() == [False, True]

    invalid_labels = labels.index_put((torch.tensor([0]), torch.tensor([0])),
                                      torch.tensor(float("inf")))
    with pytest.raises(ValueError, match="labels must be finite"):
        runtime.generate(clean_raw=waveform, targets=invalid_labels, hash_ids=hashes,
            classifier=object(), base_seed=20260501, rng_identity=IDENTITY)


def test_lhat_diagnostics_keep_three_scopes_and_two_of_three_acceptance() -> None:
    accepted = torch.tensor([True, False, True])
    attack = SimpleNamespace(
        sample_tensor_dict=lambda: {"final_bce": torch.tensor([1., 5., 9.]),
            "sample_anyflip_eligible": torch.ones(3),
            "sample_anyflip_success": torch.tensor([0., 1., 1.])},
        mean_tensor_dict=lambda: {"final_bce": torch.tensor(5.),
            "decoded_invalid_rate": torch.tensor(1 / 3),
            "sample_anyflip_numerator": torch.tensor(2.),
            "sample_anyflip_denominator": torch.tensor(3.)})
    contract = AttackThenContractDiagnostics(
        accepted=accepted, selected_t=torch.tensor([.25, 0., .75]),
        raw_clean_bce=torch.tensor([.5, .6, .7]), selected_bce=torch.tensor([1., .6, 1.4]),
        bce_gain=torch.tensor([.5, 0., .7]), path_valid_count=torch.tensor([4, 0, 3]),
        path_preserving_count=torch.tensor([2, 0, 1]),
        clean_correct_class_count=torch.tensor([5, 4, 3]),
        training_anyflip_success=torch.tensor([False, False, True]))
    samples, means, weights = _scoped_lhat_diagnostics(attack, contract, accepted, 2)
    assert samples["raw_all_candidate_eligible/final_bce"].tolist() == [1., 5., 9.]
    assert means["contract_all_candidate_eligible/contract_acceptance_rate"] == pytest.approx(2 / 3)
    assert samples["contract_all_candidate_eligible/accepted"].numel() == 3
    assert samples["contract_training_accepted/selected_t"].tolist() == [.25, .75]
    assert weights["raw_all_candidate_eligible/final_bce"] == 3
    assert weights["contract_training_accepted/contract_selected_t"] == 2
    distributions, scalars, rates = _diagnostic_sample_summary(
        {f"lhat/{name}": [value] for name, value in samples.items()})
    rate = rates["lhat/raw_all_candidate_eligible/decoded_anchor_sample_anyflip_asr"]
    assert rate == {"numerator": 2., "denominator": 3., "rate": pytest.approx(2 / 3)}
    key = "lhat/contract_training_accepted/training_anyflip_success"
    assert (distributions[key]["count"], scalars[f"{key}_mean"]) == (2, pytest.approx(.5))


def test_locked_mainline_is_the_minimal_contracted_lhat_recipe() -> None:
    recipe = load_recipe_spec(RECIPES / "augmix_simclr_lhat.yaml")
    contract = recipe.scientific_contract
    assert contract["stage1"]["pretrain_logit_anchor_weight"] == 5.0
    assert contract["stage2_teacher"] == "disabled"
    assert contract["stage2_supervised_logit_anchor_weight_by_backbone"] == {
        "efficientnet1dv2": 0.0,
        "ecgfounder": 0.0,
    }
    config = lhat.load_lhat_config(CONFIG_ROOT / "train" / "lhat.yaml")
    assert (config.num_candidates, config.hull_lambda, config.pgd_epsilon) == (
        20,
        1.0,
        12.0,
    )
    assert (config.steps, config.learning_rate) == (1, 0.25)
    assert config.attack_objective == "maximize_multilabel_bce_with_logits"
    assert config.attack_then_contract.enabled is True
    assert config.attack_then_contract.endpoint_residual_correction == (
        "linear_clean_hard_endpoint"
    )


@pytest.mark.parametrize("model_name", ["efficientnet1dv2", "ecgfounder"])
def test_tiny_lhat_attack_keeps_model_gradients_and_contracts_without_heldout(model_name):
    from core.lhat import LatentStandardizer, generate_lhat_adversarial, contract_lhat_adversarial
    from models.vae import decode_to_ptbxl_waveform

    torch.manual_seed(19)
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(12), torch.nn.AdaptiveAvgPool1d(8),
        torch.nn.Flatten(), torch.nn.Linear(96, 5),
    ).train()

    class Decoder(torch.nn.Module):
        def forward(self, latent):
            return torch.nn.functional.interpolate(
                latent.repeat(1, 3, 1), size=1024,
                mode="linear", align_corners=True,
            ).transpose(1, 2)

    decoder = Decoder().eval()
    recipes = (load_recipe_spec(RECIPES / (name + ".yaml")) for name in RECIPE_HASHES)
    recipe = next(recipe for recipe in recipes if recipe.requires_vae)
    config = lhat.load_lhat_config(CONFIG_ROOT / recipe.resources["lhat_config"]["path"])
    standardizer = LatentStandardizer.fit(torch.randn(24, 4, 128))
    anchor = torch.randn(2, 4, 128)
    candidates = torch.randn(2, 20, 4, 128)
    clean = decode_to_ptbxl_waveform(decoder, standardizer.inverse_transform(anchor))
    labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    attack = generate_lhat_adversarial(
        classifier=model, decoder=decoder, anchor_standardized=anchor,
        candidates_standardized=candidates, raw_clean_waveform=clean,
        targets=labels, standardizer=standardizer, model_name=model_name, config=config,
    )
    assert model.training and not decoder.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(before, parameter.grad) for before, parameter in zip(gradients, model.parameters()))
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    assert not attack.waveform_raw.requires_grad
    torch.testing.assert_close(attack.weights.sum(1), torch.ones(2))
    assert bool(((attack.latent_standardized - anchor).flatten(1).norm(dim=1) <= config.pgd_epsilon + 1e-5).all())
    result = contract_lhat_adversarial(
        classifier=model, decoder=decoder, attack=attack,
        raw_clean_waveform=clean, targets=labels, standardizer=standardizer,
        model_name=model_name, minimum_std_mV=1e-4, maximum_abs_mV=20., config=config,
    )
    assert result.waveform_raw.shape == (2, 1000, 12)
    assert result.valid_mask.shape == (2,) and result.valid_mask.dtype == torch.bool
    assert bool(torch.isfinite(result.waveform_raw).all())
    assert config.attack_then_contract.uses_heldout_target is False
