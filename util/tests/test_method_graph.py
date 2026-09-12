"""Characterization goldens for the finite, code-owned ECG recipes."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import core
import core.augmix as augmix
import core.corruption as corruption
import core.latent_pool as latent_pool
import core.lhat as lhat
import core.methods as methods
import core.methods.contracts as method_contracts
import core.methods.runtime as method_runtime
import core.methods.registry as method_registry
import core.online_trainer as online_trainer
import core.train_PN2021 as train_adapter
from core.lhat import AttackThenContractDiagnostics
from core.methods.registry import (
    AuxiliaryVariant,
    RecipeKind,
    Stage1Objective,
    load_recipe_spec,
)
from core.methods.runtime import (
    _lhat_training_selection_mask,
    _scoped_lhat_diagnostics,
    build_method_runtime,
)
from core.online_trainer import _diagnostic_sample_summary


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
RECIPES = CONFIG_ROOT / "train" / "methods"
IDENTITY = (
    "comparison_group=aligned_targetonly_augmix_lhat_v1", "replicate_id=0",
    "center=ningbo", "model=efficientnet1dv2", "epoch=2",
    "ordered_batch_hash_sha256=x", "composition_index=7",
)


def _batch(size: int = 2):
    waveform = torch.linspace(-1.0, 1.0, size * 12000).reshape(size, 1000, 12)
    labels = torch.zeros(size, 5); labels[:, 3] = 1
    if size == 2:
        labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    return waveform, labels, tuple(f"g{i}" for i in range(size))


def _runtime(filename: str, **resources: object):
    return build_method_runtime(load_recipe_spec(RECIPES / filename),
        model_name="efficientnet1dv2", config_root=CONFIG_ROOT, **resources)


RECIPE_CASES = [
    ("a0_clean_v1.yaml", RecipeKind.CLEAN, AuxiliaryVariant.NOT_APPLICABLE, (),
     ("classifier",), (("clean_bce", "bce", 1.0),),
     "32296a2419575e66a4f702deaa5147b328b945b72b812e52019d8d3b9fdccf98"),
    ("a3c_depth23_v1.yaml", RecipeKind.RANDOM_DEPTH23, AuxiliaryVariant.NOT_APPLICABLE,
     ("operator_profile", "corruption_rng"), ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "3eb23fd6ec7de50ee78970a91e0f81d46359d7a5dabade06f20cfad22abe92c4"),
    ("a1_corrupt_ft_rot4_v1.yaml", RecipeKind.SUPERVISED_ROTATING_DEPTH23,
     AuxiliaryVariant.NOT_APPLICABLE,
     ("operator_profile", "corruption_rng"), ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "de0224de08ebf4658828f6ed0fa5ebd414d608fb7ab3430abaddfae8d33de44f"),
    ("direct_depth23_fixed20.yaml", RecipeKind.FIXED20, AuxiliaryVariant.NOT_APPLICABLE,
     ("operator_profile", "corruption_rng"), ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "fb4c2fd18cf59d44e5c358463867209b5dbcaf052c432c1e74f2910b8d7914b8"),
    ("augmix_simclr_lhat.yaml", RecipeKind.TWO_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "augmix_config", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0),
      ("corrupted_bce", "bce", 1.0)),
     "6245e89f572b49bd24f5d688bda7d3f8b16fa4972051353afa311d007cd9f675"),
    ("augmix_simclr_matched_no_vae.yaml", RecipeKind.TWO_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.MATCHED_NO_VAE,
     ("operator_profile", "augmix_config", "corruption_rng"),
     ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "b2b140ef2e4d440769b14f35b02ca61ba3951502fd7699a276a821ee202ae0f0"),
    ("augmix_supervised_lhat.yaml", RecipeKind.TWO_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "augmix_config", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0),
      ("corrupted_bce", "bce", 1.0)),
     "552b9324685d9465577969224d473bf3f02526de3d1655c39e04e028e5b94c1c"),
    ("augmix_supervised_matched_no_vae.yaml", RecipeKind.TWO_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.MATCHED_NO_VAE,
     ("operator_profile", "augmix_config", "corruption_rng"),
     ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "c6f6ecebcce3b3eabb03775663e1b4e5ec7e16ddb2cc979d6543b2eff8a6a525"),
    ("augmix_supervised_single_chain_matched_no_vae.yaml",
     RecipeKind.TWO_STAGE_AUGMIX_LHAT, AuxiliaryVariant.MATCHED_NO_VAE,
     ("operator_profile", "augmix_config", "corruption_rng"),
     ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0)),
     "17b977083a6b971a33c83b5f1b0fdd6ae15cd6fa27574db15f8a84f1f11698b2"),
    ("vae_lhat_only.yaml", RecipeKind.SUPERVISED_ROTATING_DEPTH23_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0),
     ("corrupted_bce", "bce", 1.0)),
     "2231090a62a1140adf1ff691f5f399b77aa93d0ed681c60df3e2126ac49e3b13"),
    ("a1_rot4_augmix_mild.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
     AuxiliaryVariant.NOT_APPLICABLE,
     ("operator_profile", "augmix_config", "corruption_rng"),
     ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("augmix_bce", "bce", 1.0)),
     "9629ddea01cf713e82c88b0c34601e40c9f85e5eebe92b93c512793f6fa7113f"),
    ("a1_rot4_single_chain_mild.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
     AuxiliaryVariant.NOT_APPLICABLE,
     ("operator_profile", "augmix_config", "corruption_rng"),
     ("classifier",),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("augmix_bce", "bce", 1.0)),
     "4ed8ad59d68eabce746710237354ad40f1df998bc95d6a4328b5f1f8520d6889"),
    ("a1_rot4_vae_lhat_mild.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("lhat_direct_bce", "bce", 1.0)),
     "bf8300107864cabcdc39e5103f687cfe98977f08f954b65e2a1a155a2333f2d1"),
    ("a1_rot4_augmix_vae_lhat_mild.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_MILD_AUX,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "augmix_config", "vae", "lhat_config",
      "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("augmix_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0)),
     "8ee620be2562fa996113d880e0799287ee057753c3eb950c98bb563fd9922a2b"),
    ("a1_rot4_vae_lhat_hardgain.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("lhat_direct_bce", "bce", 1.0)),
     "838bf5d1d5850181f26cf3f3984e36dfac95f40af7293ec674cfa928cfcc4cc6"),
    ("a1_rot4_augmix_vae_lhat_hardgain.yaml",
     RecipeKind.SUPERVISED_ROTATING_DEPTH23_HARDGAIN_AUX,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("operator_profile", "augmix_config", "vae", "lhat_config",
      "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("corrupted_bce", "bce", 1.0),
      ("augmix_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0)),
     "810a5b188e08562df9e592f5383b04089bbdb52d3209c010901508e6ed7ef5f1"),
    ("one_stage_augmix_supervised.yaml", RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
     AuxiliaryVariant.NOT_APPLICABLE,
     ("augmix_config", "corruption_rng"), ("classifier",),
     (("clean_bce", "bce", 1.0), ("augmix_bce", "bce", 1.0)),
     "2f95e038d5f3c3519922ceb6e183ccc00204f785fff30642b93cd1a122b706c4"),
    ("one_stage_single_chain_supervised.yaml",
     RecipeKind.ONE_STAGE_SUPERVISED_AUGMIX,
     AuxiliaryVariant.NOT_APPLICABLE,
     ("augmix_config", "corruption_rng"), ("classifier",),
     (("clean_bce", "bce", 1.0), ("augmix_bce", "bce", 1.0)),
     "702248e615f6a2dccb3edcafc25f17697382491a3d28c6954d883a2458059ffc"),
    ("one_stage_vae_lhat_mild.yaml", RecipeKind.ONE_STAGE_VAE_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("vae", "lhat_config", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("lhat_direct_bce", "bce", 1.0)),
     "dc863cf5181175f80eb5e47cf5817a39e5026e4712ed56c1dbc56406e405d81c"),
    ("one_stage_augmix_vae_lhat_mild.yaml", RecipeKind.ONE_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("augmix_config", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("augmix_bce", "bce", 1.0),
      ("lhat_direct_bce", "bce", 1.0)),
     "7e1795c7c281a64d7a3a45f0bfcbf9e462cafc7ee0c35f60787d031ad8212799"),
    ("one_stage_single_chain_vae_lhat_mild.yaml",
     RecipeKind.ONE_STAGE_AUGMIX_LHAT,
     AuxiliaryVariant.CONTRACTED_LHAT,
     ("augmix_config", "vae", "lhat_config", "corruption_rng", "lhat_rng"),
     ("classifier", "vae_decoder", "latent_pool"),
     (("clean_bce", "bce", 1.0), ("augmix_bce", "bce", 1.0),
      ("lhat_direct_bce", "bce", 1.0)),
     "b95f6a8a9f0e7f29f8ead86057414b76aeb49be073dae3637f4fa71c7507abae"),
]


@pytest.mark.parametrize("filename,kind,variant,resources,requirements,objective,sha", RECIPE_CASES)
def test_v2_recipe_files_are_finite_resource_closed_characterizations(
    filename, kind, variant, resources, requirements, objective, sha
) -> None:
    recipe = load_recipe_spec(RECIPES / filename)
    assert (recipe.schema_version, recipe.kind, recipe.auxiliary_variant) == (2, kind, variant)
    assert not hasattr(recipe, "executable")
    assert tuple(recipe.resources) == resources
    assert tuple(recipe.describe()["requirements"]) == requirements
    assert tuple(
        (term["name"], term["kind"], term["weight"])
        for term in recipe.describe()["objective_terms"]
    ) == objective
    assert tuple(view for _, view in recipe.objective_terms) == recipe.output_names
    assert recipe.requires_vae == ("lhat_view" in recipe.output_names)
    assert not hasattr(recipe, "objective") and not hasattr(recipe, "requirements")
    assert recipe.recipe_sha256 == sha


def test_all_recipe_descriptions_preserve_the_frozen_registry_characterization() -> None:
    # Includes every selector and its spec hash, including the width1/3 JSD arms.
    # Only the checkout prefix is normalized; scientific fields stay untouched.
    descriptions = {}
    for path in sorted(RECIPES.glob("*.yaml")):
        description = load_recipe_spec(path).describe()
        description["source_path"] = path.relative_to(REPO).as_posix()
        descriptions[path.name] = description
    encoded = json.dumps(
        descriptions, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert len(descriptions) == 74
    assert hashlib.sha256(encoded).hexdigest() == (
        "638f2d6d3386a3928db7db9c65f26a141714d31d7f87572215b8b70511fe99f1"
    )


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


@pytest.mark.parametrize(
    "filename,requires_vae",
    [
        ("augmix_supervised_lhat.yaml", True),
        ("augmix_supervised_matched_no_vae.yaml", False),
    ],
)
def test_supervised_augmix_recipes_remove_contrastive_learning_only(
    filename: str, requires_vae: bool
) -> None:
    recipe = load_recipe_spec(RECIPES / filename)
    contract = recipe.scientific_contract
    assert recipe.stage1_objective is Stage1Objective.SUPERVISED_AUGMIX
    assert recipe.comparison_rng_identity == "augmix_simclr_lhat"
    assert recipe.requires_vae is requires_vae
    assert contract["stages"] == ("augmix_supervised", "supervised_adaptation")
    assert contract["stage1"]["objective"] == "supervised_augmix"
    assert contract["stage1"]["objective_weights"] == {
        "clean_bce": 0.5,
        "strong_view_bce": 0.5,
    }
    assert contract["stage1"]["classifier_head_trainable"] is False
    assert contract["stage1"]["pretrain_logit_anchor_weight"] == 5.0
    assert contract["stage1"]["ptbxl_source_replay_weight"] == 0.0
    assert contract["stage1"]["vicreg_weight"] == 0.0
    assert contract["stage2_teacher"] == "disabled"


def test_supervised_single_chain_recipe_changes_only_the_strong_view_geometry(
) -> None:
    recipe = load_recipe_spec(
        RECIPES / "augmix_supervised_single_chain_matched_no_vae.yaml"
    )
    contract = recipe.scientific_contract
    assert recipe.stage1_objective is Stage1Objective.SUPERVISED_AUGMIX
    assert recipe.requires_vae is False
    assert contract["stage1"]["view"] == (
        "clean_vs_one_single_chain_corruption_view"
    )
    assert contract["stage1"]["objective_weights"] == {
        "clean_bce": 0.5,
        "strong_view_bce": 0.5,
    }
    assert contract["stage1"]["pretrain_logit_anchor_weight"] == 5.0
    assert contract["stage2_teacher"] == "disabled"


def test_one_stage_component_grid_has_no_stage_boundary_or_contrastive_objective(
) -> None:
    augmix_only = load_recipe_spec(RECIPES / "one_stage_augmix_supervised.yaml")
    single_augmix = load_recipe_spec(
        RECIPES / "one_stage_single_chain_supervised.yaml"
    )
    vae_only = load_recipe_spec(RECIPES / "one_stage_vae_lhat_mild.yaml")
    joint = load_recipe_spec(RECIPES / "one_stage_augmix_vae_lhat_mild.yaml")
    single_joint = load_recipe_spec(
        RECIPES / "one_stage_single_chain_vae_lhat_mild.yaml"
    )
    for recipe in (augmix_only, single_augmix, vae_only, joint, single_joint):
        contract = recipe.scientific_contract
        assert contract["stages"] == ("joint_supervised_adaptation",)
        assert contract["stage_boundaries"] is False
        assert contract["simclr"] == "disabled"
        assert contract["projector"] == "disabled"
        assert contract["source_logit_anchor"] == "disabled"
        assert recipe.stage1_objective is Stage1Objective.NOT_APPLICABLE
    for recipe in (augmix_only, single_augmix):
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": 0.5,
            "augmix": 0.5,
        }
    assert single_augmix.scientific_contract["exposure_policy"] == (
        "clean_once_plus_one_single_chain_corruption_view"
    )
    assert single_joint.scientific_contract["exposure_policy"] == (
        "clean_once_plus_one_single_chain_corruption_view_plus_lhat_auxiliary"
    )
    for recipe in (vae_only, joint, single_joint):
        assert recipe.scientific_contract["auxiliary"]["alpha_max"] == 0.25
        assert recipe.scientific_contract["auxiliary"]["linear_warmup_epochs"] == 5
        assert recipe.scientific_contract["auxiliary"][
            "attack_then_contract_version"
        ] == "nondecreasing_bce_grid_v2"


def test_one_stage_nondecreasing_lhat_config_keeps_raw_clean_endpoint() -> None:
    config = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_one_stage_nondecreasing.yaml"
    )
    contract = config.attack_then_contract
    assert contract.version == "nondecreasing_bce_grid_v2"
    assert contract.t_values == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert contract.selection == (
        "maximum_bce_not_below_clean_without_new_clean_correct_flip"
    )


def test_boundary_outside_lhat_selects_nearest_successful_valid_path() -> None:
    config = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_pure_delta_boundary_outside.yaml"
    )
    contract = config.attack_then_contract
    assert contract.version == "nearest_boundary_outside_grid_v4"
    assert contract.selection == "minimum_t_with_new_clean_correct_flip"
    assert config.training_selection_mode == "all_contract_accepted"

    t_values = torch.tensor(contract.t_values)
    allowed, accepted, selected = lhat._select_contract_path_indices(
        valid=torch.tensor(
            [
                [True, True, True, True, True],
                [True, False, True, True, True],
                [True, True, True, True, True],
            ]
        ),
        preserving=torch.ones((3, 5), dtype=torch.bool),
        attack_successful=torch.tensor(
            [
                [False, False, True, True, True],
                [False, True, False, False, True],
                [False, False, False, False, False],
            ]
        ),
        path_bce=torch.tensor(
            [[0.1, 0.2, 0.3, 9.0, 10.0]] * 3
        ),
        clean_bce=torch.tensor([0.1, 0.1, 0.1]),
        t_values=t_values,
        selection=contract.selection,
    )
    assert allowed.tolist() == [
        [False, False, True, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False],
    ]
    assert accepted.tolist() == [True, True, False]
    assert selected.tolist() == [2, 4, 0]

    recipe = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat_boundary.yaml"
    )
    auxiliary = recipe.scientific_contract["lhat_auxiliary"]
    assert auxiliary["attack_then_contract_version"] == contract.version
    assert auxiliary["alpha_max"] == pytest.approx(0.2)


def test_hardgain_lhat_config_and_recipe_lock_training_selection() -> None:
    config = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_hardgain_nondecreasing.yaml"
    )
    assert config.training_selection_mode == "minimum_bce_gain"
    assert config.minimum_training_bce_gain == pytest.approx(0.01)
    recipe = load_recipe_spec(RECIPES / "a1_rot4_vae_lhat_hardgain.yaml")
    auxiliary = recipe.scientific_contract["lhat_auxiliary"]
    assert auxiliary["alpha_max"] == pytest.approx(0.25)
    assert auxiliary["linear_warmup_epochs"] == 5
    assert auxiliary["training_selection"] == {
        "mode": "minimum_bce_gain",
        "minimum_bce_gain": 0.01,
    }
    runtime = build_method_runtime(
        recipe,
        model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT,
        latent_pool=object(),
        decoder=torch.nn.Identity(),
    )
    assert runtime.lhat_config is not None
    assert runtime.lhat_config.training_selection_mode == "minimum_bce_gain"


def test_raw_attack_success_recipe_locks_sparse_lhat_training_selection() -> None:
    config = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_pure_delta_raw_attack_success.yaml"
    )
    assert config.training_selection_mode == "raw_attack_success"
    assert config.minimum_training_bce_gain == 0.0
    recipe = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_rawsuccess.yaml"
    )
    auxiliary = recipe.scientific_contract["lhat_auxiliary"]
    assert auxiliary["alpha_max"] == pytest.approx(0.2)
    assert auxiliary["training_selection"] == {
        "mode": "raw_attack_success",
        "minimum_bce_gain": 0.0,
    }
    runtime = build_method_runtime(
        recipe,
        model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT,
        latent_pool=object(),
        decoder=torch.nn.Identity(),
    )
    assert runtime.lhat_config is not None
    assert runtime.lhat_config.training_selection_mode == "raw_attack_success"

    selected = _lhat_training_selection_mask(
        runtime.lhat_config,
        SimpleNamespace(
            positive_hide_numerator=torch.tensor([0, 1, 0, 2]),
            negative_add_numerator=torch.tensor([0, 0, 3, 1]),
        ),
        SimpleNamespace(
            accepted=torch.ones(4, dtype=torch.bool),
            bce_gain=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        ),
    )
    assert torch.equal(selected, torch.tensor([False, True, True, True]))

    equalpn_config = lhat.load_lhat_config(
        CONFIG_ROOT
        / "train"
        / "lhat_pure_delta_equal_positive_negative_raw_attack_success.yaml"
    )
    assert equalpn_config.attack_objective == (
        "maximize_equal_positive_negative_bce_with_logits"
    )
    equalpn_recipe = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_equalpn_rawsuccess.yaml"
    )
    assert equalpn_recipe.scientific_contract["lhat_auxiliary"] == auxiliary
    equalpn_runtime = build_method_runtime(
        equalpn_recipe,
        model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT,
        latent_pool=object(),
        decoder=torch.nn.Identity(),
    )
    assert equalpn_runtime.lhat_config is not None
    assert equalpn_runtime.lhat_config.attack_objective == (
        "maximize_equal_positive_negative_bce_with_logits"
    )


def test_repaired_auxiliary_recipes_keep_augmix_and_lhat_ablatable() -> None:
    single = load_recipe_spec(RECIPES / "a1_rot4_single_chain_jsd.yaml")
    two = load_recipe_spec(RECIPES / "a1_rot4_augmix_jsd.yaml")
    vae_only = load_recipe_spec(RECIPES / "a1_rot4_vae_lhat_puredelta.yaml")
    joint = load_recipe_spec(
        RECIPES / "a1_rot4_augmix_jsd_vae_lhat_puredelta.yaml"
    )
    assert single.scientific_contract["augmix_auxiliary"] == {
        "objective_terms": ("augmix_bce",),
        "weight": 0.125,
        "view_geometry": (
            "one_depth23_corruption_chain_with_clean_bernoulli_jsd"
        ),
        "bernoulli_jsd_weight": 12.0,
        "batch_norm_policy": "zero_momentum",
        "global_rng_policy": "snapshot_restore",
    }
    two_augmix = two.scientific_contract["augmix_auxiliary"]
    assert two_augmix["objective_terms"] == (
        "augmix_bce",
        "augmix_chain1_context",
        "augmix_chain2_context",
    )
    assert two_augmix["bernoulli_jsd_weight"] == pytest.approx(12.0)
    assert vae_only.scientific_contract["augmix_auxiliary"] is None
    for recipe in (vae_only, joint):
        assert recipe.scientific_contract["lhat_auxiliary"][
            "attack_then_contract_version"
        ] == "nondecreasing_pure_delta_grid_v3"
    assert joint.scientific_contract["augmix_auxiliary"] == two_augmix

    pure_delta = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_pure_delta_nondecreasing.yaml"
    )
    assert pure_delta.attack_then_contract.version == (
        "nondecreasing_pure_delta_grid_v3"
    )
    assert pure_delta.attack_then_contract.endpoint_residual_correction == (
        "constant_clean_anchor_residual"
    )


def test_r19_lhat_ablation_recipes_lock_only_candidate_source_or_contract() -> None:
    mixed = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_mixedm20.yaml"
    )
    raw = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2_nocontract.yaml"
    )
    mixed_aux = mixed.scientific_contract["lhat_auxiliary"]
    raw_aux = raw.scientific_contract["lhat_auxiliary"]
    assert mixed_aux["candidate_source_policy"] == (
        "clean10_corrupted10_same_neighbors_v1"
    )
    assert mixed_aux["attack_then_contract_version"] == (
        "nondecreasing_pure_delta_grid_v3"
    )
    assert raw.auxiliary_variant is AuxiliaryVariant.RAW_LHAT
    assert raw_aux["contract_enabled"] is False
    assert raw_aux["diagnostic_scopes"] == ("raw_all_candidate_eligible",)
    raw_config = lhat.load_lhat_config(
        CONFIG_ROOT / "train" / "lhat_pure_delta_no_contract.yaml"
    )
    assert raw_config.attack_then_contract.enabled is False
    assert raw_config.attack_then_contract.version == "raw_attack_no_contract_v1"
    assert raw_config.attack_then_contract.t_values == ()


def test_mixed_m20_uses_same_neighbor_ids_and_exactly_ten_corrupted_slots() -> None:
    count = 21
    clean = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1)
    corrupted = clean + 1000.0
    neighbor_rows = torch.stack(
        [
            torch.tensor([value for value in range(count) if value != anchor])
            for anchor in range(count)
        ]
    )
    standardizer = lhat.LatentStandardizer(
        mean=torch.zeros((1, 1, 1)),
        scale=torch.ones((1, 1, 1)),
        count=count,
        epsilon=1.0e-6,
        identity_sha256="1" * 64,
    )
    identity = latent_pool.LatentPoolIdentity(
        schema_version=3,
        encoder_identity="2" * 64,
        record_count=count,
        latent_shape=(1, 1),
        num_candidates=20,
        candidate_source_policy="clean10_corrupted10_same_neighbors_v1",
        ordered_hash_ids_sha256="3" * 64,
        labels_sha256="4" * 64,
        latents_sha256="5" * 64,
        candidate_corrupted_latents_sha256="6" * 64,
        eligibility_sha256="7" * 64,
        exact_neighbor_indices_sha256="8" * 64,
        standardizer_epsilon=1.0e-6,
        standardizer_sha256="1" * 64,
        identity_sha256="9" * 64,
    )
    pool = latent_pool.LatentPool(
        hash_ids=tuple(f"h{index}" for index in range(count)),
        selection_indices=torch.arange(count),
        cache_indices=torch.arange(count),
        labels=torch.ones((count, 5)),
        latents=clean,
        standardized_latents=clean,
        candidate_source_policy=identity.candidate_source_policy,
        candidate_corrupted_latents=corrupted,
        candidate_corrupted_standardized_latents=corrupted,
        candidate_counts=torch.full((count,), 20),
        exact_neighbor_indices=neighbor_rows,
        standardizer=standardizer,
        identity=identity,
    )
    attack = pool.get_attack_batch_by_pool_indices(
        [0],
        mode="nearest",
        local_pool_size=80,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(attack.candidate_pool_indices, neighbor_rows[:1])
    assert attack.candidate_corrupted_mask.sum().item() == 10
    expected = clean[neighbor_rows[:1]].clone()
    expected[:, 1::2] += 1000.0
    assert torch.equal(attack.candidates_standardized, expected)
    moved = pool.to("cpu").get_attack_batch_by_pool_indices(
        [0], mode="nearest", local_pool_size=80,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(moved.candidate_corrupted_mask, attack.candidate_corrupted_mask)


def test_strong_a1_pool_recipes_lock_endpoint_mean_and_component_ablation() -> None:
    single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_jsd_strong.yaml"
    )
    two = load_recipe_spec(
        RECIPES / "a1_rot4_augmix_jsd_endpoint_mean_strong.yaml"
    )
    joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_augmix_jsd_endpoint_mean_strong_vae_lhat_puredelta.yaml"
    )
    assert single.scientific_contract["stage_boundaries"] is False
    assert single.scientific_contract["simclr"] == "disabled"
    assert single.scientific_contract["augmix_auxiliary"]["weight"] == 0.5
    for recipe in (two, joint):
        augmix = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix["weight"] == 0.5
        assert augmix["supervised_view_policy"] == "mixed_plus_chains_mean"
        assert augmix["objective_terms"] == (
            "augmix_bce",
            "augmix_chain1_context",
            "augmix_chain2_context",
        )
    assert joint.scientific_contract["lhat_auxiliary"][
        "attack_then_contract_version"
    ] == "nondecreasing_pure_delta_grid_v3"

    hard = load_recipe_spec(
        RECIPES / "a1_rot4_augmix_jsd_hardview_strong.yaml"
    )
    hard_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_augmix_jsd_hardview_strong_vae_lhat_puredelta.yaml"
    )
    for recipe in (hard, hard_joint):
        assert recipe.scientific_contract["augmix_auxiliary"][
            "supervised_view_policy"
        ] == "per_sample_max_mixed_and_chains"
    assert hard_joint.scientific_contract["lhat_auxiliary"][
        "attack_then_contract_version"
    ] == "nondecreasing_pure_delta_grid_v3"

    complementary = load_recipe_spec(
        RECIPES / "a1_rot4_augmix_jsd_complementary_strong.yaml"
    )
    complementary_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_augmix_jsd_complementary_strong_vae_lhat_puredelta.yaml"
    )
    for recipe in (complementary, complementary_joint):
        assert recipe.scientific_contract["augmix_auxiliary"]["weight"] == 0.5
        runtime = build_method_runtime(
            recipe,
            model_name="efficientnet1dv2",
            config_root=CONFIG_ROOT,
            **(
                {"latent_pool": object(), "decoder": torch.nn.Identity()}
                if recipe.requires_vae
                else {}
            ),
        )
        assert runtime.augmix_config is not None
        assert runtime.augmix_config.stage1_mode == (
            "two_chain_complementary_no_clean_mix"
        )

    balanced_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_supervised_balanced.yaml"
    )
    balanced_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_supervised_balanced.yaml"
    )
    balanced_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_supervised_balanced_vae_lhat_puredelta.yaml"
    )
    for recipe in (balanced_single, balanced_two, balanced_joint):
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": 0.25,
            "corrupted_total": 0.25,
            "corrupted_per_composition": 0.0625,
        }
        assert recipe.scientific_contract["augmix_auxiliary"]["weight"] == 0.5
    for recipe in (balanced_two, balanced_joint):
        assert recipe.scientific_contract["augmix_auxiliary"][
            "supervised_view_policy"
        ] == "chains_mean"

    jsd3_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_balanced_jsd3.yaml"
    )
    jsd3_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_balanced_jsd3.yaml"
    )
    jsd3_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd3_vae_lhat_puredelta.yaml"
    )
    for recipe in (jsd3_single, jsd3_two, jsd3_joint):
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": 0.25,
            "corrupted_total": 0.25,
            "corrupted_per_composition": 0.0625,
        }
        augmix = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix["weight"] == 0.5
        assert augmix["bernoulli_jsd_weight"] == 3.0
    for recipe in (jsd3_two, jsd3_joint):
        assert recipe.scientific_contract["augmix_auxiliary"][
            "supervised_view_policy"
        ] == "chains_mean"
    assert jsd3_joint.scientific_contract["lhat_auxiliary"][
        "attack_then_contract_version"
    ] == "nondecreasing_pure_delta_grid_v3"

    jsd1p5_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_balanced_jsd1p5.yaml"
    )
    jsd1p5_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_balanced_jsd1p5.yaml"
    )
    jsd1p5_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_puredelta.yaml"
    )
    for recipe in (jsd1p5_single, jsd1p5_two, jsd1p5_joint):
        augmix = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix["weight"] == 0.5
        assert augmix["bernoulli_jsd_weight"] == 1.5
    for recipe in (jsd1p5_two, jsd1p5_joint):
        assert recipe.scientific_contract["augmix_auxiliary"][
            "supervised_view_policy"
        ] == "chains_mean"
    alpha0p25 = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p25.yaml"
    )
    alpha0p5 = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_alpha0p5.yaml"
    )
    assert alpha0p25.scientific_contract["lhat_auxiliary"][
        "alpha_max"
    ] == 0.25
    assert alpha0p5.scientific_contract["lhat_auxiliary"][
        "alpha_max"
    ] == 0.5
    for recipe in (alpha0p25, alpha0p5):
        assert recipe.scientific_contract["family_loss_weights"] == (
            jsd1p5_joint.scientific_contract["family_loss_weights"]
        )
        assert recipe.scientific_contract["augmix_auxiliary"] == (
            jsd1p5_joint.scientific_contract["augmix_auxiliary"]
        )

    for suffix, mass in (("0p05", 0.05), ("0p1", 0.1), ("0p2", 0.2)):
        replacement = load_recipe_spec(
            RECIPES
            / f"a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace{suffix}.yaml"
        )
        assert replacement.scientific_contract["family_loss_weights"] == (
            jsd1p5_joint.scientific_contract["family_loss_weights"]
        )
        lhat = replacement.scientific_contract["lhat_auxiliary"]
        assert lhat["alpha_max"] == mass
        assert lhat["linear_warmup_epochs"] == 1
        assert lhat["loss_integration"] == (
            "replace_clean_with_lhat_or_clean_fallback"
        )
        assert lhat["rejection_fallback"] == "clean_bce"

    robust_profiles = {
        "a1_rot4_two_chain_robust_c125_r375_a500_jsd1p5_vae_lhat.yaml": (
            0.125, 0.375, 0.5
        ),
        "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat.yaml": (
            0.125, 0.5, 0.375
        ),
        "a1_rot4_two_chain_robust_c0625_r4375_a500_jsd1p5_vae_lhat.yaml": (
            0.0625, 0.4375, 0.5
        ),
    }
    for filename, (clean, corrupted, augmix_weight) in robust_profiles.items():
        recipe = load_recipe_spec(RECIPES / filename)
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": clean,
            "corrupted_total": corrupted,
            "corrupted_per_composition": corrupted / 4.0,
        }
        assert recipe.scientific_contract["augmix_auxiliary"]["weight"] == (
            augmix_weight
        )
        assert recipe.scientific_contract["augmix_auxiliary"][
            "bernoulli_jsd_weight"
        ] == 1.5
        assert recipe.scientific_contract["lhat_auxiliary"]["alpha_max"] == 0.1
        assert clean + corrupted + augmix_weight == 1.0

    r8_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_robust_c125_r500_a375_jsd1p5.yaml"
    )
    r8_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5.yaml"
    )
    r8_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_jsd1p5_vae_lhat_rawsuccess.yaml"
    )
    for recipe in (r8_single, r8_two, r8_joint):
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": 0.125,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.125,
        }
        assert recipe.scientific_contract["augmix_auxiliary"]["weight"] == 0.375
        assert recipe.scientific_contract["augmix_auxiliary"][
            "bernoulli_jsd_weight"
        ] == 1.5
    assert len(r8_single.output_names) + 1 == len(r8_two.output_names)
    assert r8_two.scientific_contract["lhat_auxiliary"] is None
    assert r8_joint.scientific_contract["augmix_auxiliary"] == (
        r8_two.scientific_contract["augmix_auxiliary"]
    )

    r10_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_robust_c125_r500_a375_mixed_jsd1p5.yaml"
    )
    r10_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5.yaml"
    )
    r10_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_mixed_jsd1p5_vae_lhat.yaml"
    )
    for recipe in (r10_single, r10_two, r10_joint):
        augmix_contract = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix_contract["weight"] == 0.375
        assert augmix_contract["bernoulli_jsd_weight"] == 1.5
        assert "supervised_view_policy" not in augmix_contract
    assert r10_two.scientific_contract["lhat_auxiliary"] is None
    assert r10_joint.scientific_contract["lhat_auxiliary"]["alpha_max"] == 0.1
    assert r10_joint.scientific_contract["augmix_auxiliary"] == (
        r10_two.scientific_contract["augmix_auxiliary"]
    )

    r11_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_robust_c125_r500_a375_endpointmean_jsd1p5.yaml"
    )
    r11_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5.yaml"
    )
    r11_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_endpointmean_jsd1p5_vae_lhat.yaml"
    )
    for recipe in (r11_single, r11_two, r11_joint):
        augmix_contract = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix_contract["weight"] == 0.375
        assert augmix_contract["bernoulli_jsd_weight"] == 1.5
        assert augmix_contract["supervised_view_policy"] == (
            "mixed_plus_chains_mean"
        )
    assert r11_two.scientific_contract["lhat_auxiliary"] is None
    assert r11_joint.scientific_contract["lhat_auxiliary"]["alpha_max"] == 0.1
    assert r11_joint.scientific_contract["augmix_auxiliary"] == (
        r11_two.scientific_contract["augmix_auxiliary"]
    )

    r12_single = load_recipe_spec(
        RECIPES / "a1_rot4_single_chain_robust_c125_r500_a375_halfendpoint_jsd1p5.yaml"
    )
    r12_two = load_recipe_spec(
        RECIPES / "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5.yaml"
    )
    r12_joint = load_recipe_spec(
        RECIPES
        / "a1_rot4_two_chain_robust_c125_r500_a375_halfendpoint_jsd1p5_vae_lhat.yaml"
    )
    for recipe in (r12_single, r12_two, r12_joint):
        augmix_contract = recipe.scientific_contract["augmix_auxiliary"]
        assert augmix_contract["weight"] == 0.375
        assert augmix_contract["bernoulli_jsd_weight"] == 1.5
        assert augmix_contract["supervised_view_policy"] == (
            "mixed_plus_chains_half"
        )
    assert r12_two.scientific_contract["lhat_auxiliary"] is None
    assert r12_joint.scientific_contract["lhat_auxiliary"]["alpha_max"] == 0.1
    assert r12_joint.scientific_contract["augmix_auxiliary"] == (
        r12_two.scientific_contract["augmix_auxiliary"]
    )


def test_two_chain_consistency_runtime_exposes_both_chains(monkeypatch) -> None:
    waveform, labels, hashes = _batch()
    monkeypatch.setattr(
        method_runtime,
        "_generate_augmix_multiview",
        lambda clean, **kwargs: SimpleNamespace(
            mixed_raw=clean + 0.25,
            chain_raws=(clean + 0.5, clean + 0.75),
        ),
    )
    runtime = _runtime("a1_rot4_augmix_jsd.yaml")
    objective_terms = runtime.recipe.scientific_contract[
        "augmix_auxiliary"
    ]["objective_terms"]
    generated = runtime.generate(
        clean_raw=waveform,
        targets=labels,
        hash_ids=hashes,
        classifier=object(),
        base_seed=20260501,
        rng_identity=IDENTITY,
        objective_term_names=objective_terms,
    )
    assert tuple(generated.bundle.values) == (
        "clean_view",
        "augmix_view",
        "augmix_chain1_view",
        "augmix_chain2_view",
    )
    assert torch.equal(
        generated.bundle.require("augmix_view").waveform, waveform + 0.25
    )
    assert torch.equal(
        generated.bundle.require("augmix_chain2_view").waveform,
        waveform + 0.75,
    )


def test_a1_mild_auxiliary_grid_preserves_the_locked_rotating4_base() -> None:
    filenames = (
        "a1_rot4_augmix_mild.yaml",
        "a1_rot4_vae_lhat_mild.yaml",
        "a1_rot4_augmix_vae_lhat_mild.yaml",
    )
    for filename in filenames:
        recipe = load_recipe_spec(RECIPES / filename)
        contract = recipe.scientific_contract
        assert contract["stages"] == ("joint_supervised_adaptation",)
        assert contract["stage_boundaries"] is False
        assert contract["simclr"] == "disabled"
        assert contract["projector"] == "disabled"
        assert contract["family_loss_weights"] == {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.125,
        }
    augmix_recipe = load_recipe_spec(RECIPES / filenames[0])
    assert augmix_recipe.scientific_contract["augmix_auxiliary"]["weight"] == 0.125
    runtime = build_method_runtime(
        augmix_recipe,
        model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT,
    )
    assert runtime.augmix_config is not None
    assert runtime.augmix_config.stage1_mode == "two_chain_no_clean_mix"
    lhat_recipe = load_recipe_spec(RECIPES / filenames[1])
    assert lhat_recipe.scientific_contract["lhat_auxiliary"]["alpha_max"] == 0.1
    assert lhat_recipe.scientific_contract["lhat_auxiliary"][
        "attack_then_contract_version"
    ] == "nondecreasing_bce_grid_v2"


def test_a1_single_chain_is_matched_to_the_two_chain_auxiliary() -> None:
    single = load_recipe_spec(RECIPES / "a1_rot4_single_chain_mild.yaml")
    two = load_recipe_spec(RECIPES / "a1_rot4_augmix_mild.yaml")
    for recipe in (single, two):
        assert recipe.scientific_contract["family_loss_weights"] == {
            "clean": 0.5,
            "corrupted_total": 0.5,
            "corrupted_per_composition": 0.125,
        }
        assert recipe.scientific_contract["augmix_auxiliary"]["weight"] == 0.125
    assert single.comparison_rng_identity == two.comparison_rng_identity
    assert single.rng_namespaces == two.rng_namespaces
    single_runtime = build_method_runtime(
        single, model_name="efficientnet1dv2", config_root=CONFIG_ROOT
    )
    two_runtime = build_method_runtime(
        two, model_name="efficientnet1dv2", config_root=CONFIG_ROOT
    )
    assert single_runtime.augmix_config is not None
    assert two_runtime.augmix_config is not None
    assert single_runtime.augmix_config.random_namespace == (
        two_runtime.augmix_config.random_namespace
    )
    assert single_runtime.augmix_config.stage1_mode == "single_chain_no_mix"
    assert two_runtime.augmix_config.stage1_mode == "two_chain_no_clean_mix"
    assert single.scientific_contract["augmix_auxiliary"]["view_geometry"] == (
        "one_depth23_corruption_chain_without_mix"
    )


def test_one_stage_augmix_runtime_emits_supervised_strong_view(monkeypatch) -> None:
    waveform, labels, hashes = _batch()
    monkeypatch.setattr(
        method_runtime,
        "generate_two_chain_augmix_strong_view",
        lambda clean, **kwargs: SimpleNamespace(mixed_raw=clean + 0.25),
    )
    generated = _runtime("one_stage_augmix_supervised.yaml").generate(
        clean_raw=waveform,
        targets=labels,
        hash_ids=hashes,
        classifier=object(),
        base_seed=20260501,
        rng_identity=IDENTITY,
    )
    assert tuple(generated.bundle.values) == ("clean_view", "augmix_view")
    assert torch.equal(
        generated.bundle.require("augmix_view").waveform,
        waveform + 0.25,
    )
    assert generated.bundle.diagnostics["augmix/output_nonfinite_count"].item() == 0
    assert "augmix/rng/two_chain_and_mix/cpu" in generated.bundle.diagnostics


def test_one_stage_single_chain_runtime_resolves_no_mix_geometry() -> None:
    for filename in (
        "one_stage_single_chain_supervised.yaml",
        "one_stage_single_chain_vae_lhat_mild.yaml",
    ):
        recipe = load_recipe_spec(RECIPES / filename)
        runtime = build_method_runtime(
            recipe,
            model_name="efficientnet1dv2",
            config_root=CONFIG_ROOT,
            latent_pool=object() if recipe.requires_vae else None,
            decoder=torch.nn.Identity() if recipe.requires_vae else None,
        )
        assert runtime.augmix_config is not None
        assert runtime.augmix_config.stage1_mode == "single_chain_no_mix"
        assert runtime.augmix_config.stage1_width == 1


def test_loader_removes_dag_plugins_and_locks_component_ablation_slots() -> None:
    assert core.__all__ == []
    assert methods.__all__ == []
    assert tuple(tuple(module.__all__) for module in (
        augmix, corruption, lhat, latent_pool, method_contracts, method_registry,
        method_runtime, online_trainer, train_adapter,
    )) == (
        ("AugMixConfig", "generate_two_chain_augmix_strong_view", "load_augmix_config"),
        ("CANONICAL_OPERATORS", "INPUT_SAMPLING_RATE_HZ",
         "generate_canonical_corruption"),
        ("LHATConfig", "LatentStandardizer", "contract_lhat_adversarial",
         "generate_lhat_adversarial", "load_lhat_config"),
        ("LatentPool", "build_latent_pool"),
        ("BASE_VIEW_NAME", "ViewBundle", "WaveformView"),
        ("AuxiliaryVariant", "RecipeKind", "Stage1Objective", "load_recipe_spec"),
        ("build_method_runtime",),
        ("ALLOWED_CENTERS", "DEFAULT_ONLINE_CONFIG_PATH", "OnlineTrainingResult",
         "load_online_train_config", "resolve_online_training_parameters",
         "train_online_model"),
        ("build_pn2021_k500_loader_plan", "load_pn2021_recipe_spec", "train_pn2021"),
    )
    assert not hasattr(core, "train_online_model")
    assert not hasattr(methods, "load_recipe_spec")
    assert "mean_dict" not in vars(lhat.LHATDiagnostics)
    assert {"make_lhat_generator", "select_exact_label_candidates"}.isdisjoint(vars(lhat))
    pool_fields = latent_pool.LatentPool.__dataclass_fields__
    assert {"selection_indices", "cache_indices", "identity"} <= pool_fields.keys()
    assert {"_selection_to_pool", "_cache_to_pool", "_ineligible_hash_ids"}.isdisjoint(
        pool_fields
    )
    assert {
        "eligible_mask", "ineligible_hash_ids", "indices_for_selection_indices",
        "indices_for_cache_indices", "get_attack_batch_by_selection_indices",
        "get_attack_batch_by_cache_indices",
    }.isdisjoint(vars(latent_pool.LatentPool))
    assert {"_positions", "_RecipeContext"}.isdisjoint(vars(method_runtime))
    assert {"batch_size", "__post_init__"}.isdisjoint(
        vars(method_runtime.GeneratedMethodBatch)
    )
    assert "requires_latent_pool" not in vars(method_runtime.MethodViewRuntime)
    assert tuple(augmix.TwoChainAugMixBatch.__dataclass_fields__) == ("mixed_raw",)
    assert tuple(corruption.CorruptionDiagnostics.__dataclass_fields__) == (
        "composition_index", "depth", "operator_mask", "output_nonfinite_count")
    assert "return_reasons" not in inspect.signature(method_runtime._quality_mask).parameters
    assert "Provenance" not in set(vars(method_contracts)) | set(method_contracts.__all__)
    assert {"MethodRequirements", "ObjectivePlan", "ObjectiveTerm"}.isdisjoint(
        set(vars(method_contracts)) | set(method_contracts.__all__)
    )
    assert "RecipeSpec" not in method_registry.__all__
    assert tuple(method_registry.RecipeSpec.__dataclass_fields__) == (
        "profile_name", "resources", "rng_namespaces", "recipe_sha256",
        "source_path", "scientific_contract", "_definition",
    )
    assert tuple(method_contracts.WaveformView.__dataclass_fields__) == (
        "name", "waveform", "labels", "sample_ids", "valid_mask", "metadata",
        "sampling_rate_hz", "units", "layout")
    runtime_source = inspect.getsource(method_runtime.MethodViewRuntime)
    assert all(value not in runtime_source for value in (
        '"corruption_diagnostics"', '"exposure_kind"', '"diagnostic_means"',
        '"anchor_waveform_raw"', '"source_view"', "full_anchor_reconstruction",
        "node_type", "context.resource", "_generators"))
    assert "recipe and LHAT attack-then-contract versions differ" in runtime_source
    assert runtime_source.count("_torch_generator(") == 3
    assert "DEFAULT_METHOD_CONFIG_DIR" not in vars(online_trainer)
    assert "profile_name" not in vars(online_trainer.OnlineTrainConfig)
    assert {"model", "history"}.isdisjoint(
        online_trainer.OnlineTrainingResult.__dataclass_fields__)
    assert tuple(latent_pool.LatentAttackBatch.__dataclass_fields__) == (
        "labels", "anchor_standardized", "candidate_pool_indices",
        "candidates_standardized", "candidate_corrupted_mask")
    assert "LatentAttackBatch" not in latent_pool.__all__
    payload = yaml.safe_load((RECIPES / "a0_clean_v1.yaml").read_text())
    payload["recipe"]["module"] = "arbitrary.user.plugin"
    with pytest.raises(ValueError, match="may not select callables"):
        load_recipe_spec(payload)
    payload = yaml.safe_load((RECIPES / "a0_clean_v1.yaml").read_text())
    payload["nodes"] = {}
    with pytest.raises(ValueError, match="root keys must be exactly"):
        load_recipe_spec(payload)

    recipe = load_recipe_spec(RECIPES / "augmix_simclr_matched_no_vae.yaml")
    assert (recipe.kind, recipe.auxiliary_variant) == (
        RecipeKind.TWO_STAGE_AUGMIX_LHAT, AuxiliaryVariant.MATCHED_NO_VAE)
    assert tuple(recipe.resources) == ("operator_profile", "augmix_config", "corruption_rng")
    assert recipe.comparison_rng_identity == "augmix_simclr_lhat"
    assert recipe.recipe_sha256 == (
        "b2b140ef2e4d440769b14f35b02ca61ba3951502fd7699a276a821ee202ae0f0"
    )
    assert recipe.objective_terms == (
        ("clean_bce", "clean_view"),
        ("corrupted_bce", "corrupted_view"),
    )
    assert not recipe.requires_vae
    assert recipe.scientific_contract["stage2_teacher"] == "disabled"
    assert recipe.scientific_contract[
        "stage2_supervised_logit_anchor_weight_by_backbone"
    ] == {"efficientnet1dv2": 0.0, "ecgfounder": 0.0}
    vae_only = load_recipe_spec(RECIPES / "vae_lhat_only.yaml")
    assert (vae_only.kind, vae_only.auxiliary_variant) == (
        RecipeKind.SUPERVISED_ROTATING_DEPTH23_LHAT,
        AuxiliaryVariant.CONTRACTED_LHAT,
    )
    assert vae_only.scientific_contract["stages"] == ("supervised_adaptation",)
    assert vae_only.scientific_contract["stage2_teacher"] == "disabled"
    assert "stage1" not in vae_only.scientific_contract
    assert vae_only.comparison_rng_identity == "augmix_simclr_lhat"
    with pytest.raises(TypeError, match="loader-owned"):
        method_registry.RecipeSpec()
    with pytest.raises(TypeError, match="loader-owned"):
        replace(recipe, profile_name="a0_clean_v1")
    payload = yaml.safe_load(
        (RECIPES / "augmix_simclr_matched_no_vae.yaml").read_text()
    )
    payload["recipe"]["auxiliary_variant"] = "contracted_lhat"
    with pytest.raises(ValueError, match="requires auxiliary_variant='matched_no_vae'"):
        load_recipe_spec(payload)
    assert not (RECIPES / "exp_paired_augmix_latent_bridge_v1.yaml").exists()
    retired = yaml.safe_load((RECIPES / "a0_clean_v1.yaml").read_text())
    retired["recipe"].update(
        id="latent_threechain_augmix_residual_depth23_aug075",
        kind="latent_threechain",
        scientific_arm="latent_augmix",
        status="project_defined_candidate",
    )
    with pytest.raises(ValueError, match="unknown recipe.id"):
        load_recipe_spec(retired)


GENERATION_CASES = [
    ("a0_clean_v1.yaml", "clean_view", None,
     "2279cd3d724ea0732466b39d8a1ad7b3b1a7938a4c39ee7cb3d5c4028e1771ab", None),
    ("a3c_depth23_v1.yaml", "corrupted_view", None,
     "d02d094ff97cd583c677a476964298166a7d375e8c482c5a5dc71fd330649301",
     ([17, 12], [3, 3], [[False, True, True, False, True], [True, True, False, False, True]])),
    ("direct_depth23_fixed20.yaml", "corrupted_view", 7,
     "d2beb5d7c7dd240775b0d4e00370f2f73fec08552e97908b4691458e0414eaa0",
     ([7, 7], [2, 2], [[False, False, True, True, False]] * 2)),
    ("a1_corrupt_ft_rot4_v1.yaml", "corrupted_view", 7,
     "5edea709f09b66cf2f4da67f745f4c8ac5c115079ec8f45596c5d423f6f8a470",
     ([7, 7], [2, 2], [[False, False, True, True, False]] * 2)),
]


@pytest.mark.parametrize("filename,view_name,composition,waveform_sha,trace", GENERATION_CASES)
def test_cpu_finite_recipe_generation_goldens(
    filename, view_name, composition, waveform_sha, trace
) -> None:
    waveform, labels, hashes = _batch()
    kwargs = dict(clean_raw=waveform, targets=labels, hash_ids=hashes, classifier=object(),
                  base_seed=20260501, rng_identity=IDENTITY)
    if composition is not None:
        kwargs.update(composition_indices=torch.full((2,), composition),
                      composition_index_hint=composition)
    generated = _runtime(filename).generate(**kwargs)
    view = generated.bundle.require(view_name)
    assert hashlib.sha256(view.waveform.numpy().tobytes()).hexdigest() == waveform_sha
    assert torch.equal(generated.bundle.require("clean_view").waveform, waveform)
    assert generated.bundle.require("clean_view").metadata == {}
    assert view.sample_ids == hashes
    if trace is None:
        assert generated.stochastic_trace == {}
    else:
        prefix = "depth23_corruption/"
        assert tuple(view.metadata) == ("diagnostic_weight", "stochastic_trace")
        assert generated.stochastic_trace[prefix + "composition_index"].tolist() == trace[0]
        assert generated.stochastic_trace[prefix + "depth"].tolist() == trace[1]
        assert generated.stochastic_trace[prefix + "operator_mask"].tolist() == trace[2]
    if filename == "a3c_depth23_v1.yaml":
        runtime = _runtime(filename)
        clean_only = runtime.generate(
            **kwargs, objective_term_names=("clean_bce",)
        )
        reversed_terms = runtime.generate(
            **kwargs, objective_term_names=("corrupted_bce", "clean_bce")
        )
        assert tuple(clean_only.bundle.values) == ("clean_view",)
        assert tuple(reversed_terms.bundle.values) == (
            "clean_view", "corrupted_view"
        )


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
