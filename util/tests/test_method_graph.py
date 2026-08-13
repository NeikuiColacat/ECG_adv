"""Characterization goldens for the finite, code-owned ECG recipes."""

from __future__ import annotations

import hashlib
import inspect
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
from core.methods.registry import AuxiliaryVariant, RecipeKind, load_recipe_spec
from core.methods.runtime import build_method_runtime, _scoped_lhat_diagnostics
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
     "de3d6948a9efc974405b7382d45653f93032a34ac2fd88637172ed731d876602"),
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


def test_loader_removes_dag_plugins_and_allows_only_the_matched_no_vae_slot() -> None:
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
        ("AuxiliaryVariant", "RecipeKind", "load_recipe_spec"),
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
        "node_type", "attack_then_contract", "context.resource", "_generators"))
    assert runtime_source.count("_torch_generator(") == 2
    assert "DEFAULT_METHOD_CONFIG_DIR" not in vars(online_trainer)
    assert "profile_name" not in vars(online_trainer.OnlineTrainConfig)
    assert {"model", "history"}.isdisjoint(
        online_trainer.OnlineTrainingResult.__dataclass_fields__)
    assert tuple(latent_pool.LatentAttackBatch.__dataclass_fields__) == (
        "labels", "anchor_standardized", "candidate_pool_indices", "candidates_standardized")
    assert "LatentAttackBatch" not in latent_pool.__all__
    payload = yaml.safe_load((RECIPES / "a0_clean_v1.yaml").read_text())
    payload["recipe"]["module"] = "arbitrary.user.plugin"
    with pytest.raises(ValueError, match="may not select callables"):
        load_recipe_spec(payload)
    payload = yaml.safe_load((RECIPES / "a0_clean_v1.yaml").read_text())
    payload["nodes"] = {}
    with pytest.raises(ValueError, match="root keys must be exactly"):
        load_recipe_spec(payload)

    payload = yaml.safe_load((RECIPES / "augmix_simclr_lhat.yaml").read_text())
    payload["recipe"].update(id="augmix_simclr_matched_no_vae",
        auxiliary_variant="matched_no_vae", scientific_arm="augmix_simclr_matched_no_vae",
        status="prospective_matched_ablation")
    for name in ("vae", "lhat_config", "lhat_rng"):
        payload["resources"].pop(name)
    recipe = load_recipe_spec(payload)
    assert (recipe.kind, recipe.auxiliary_variant) == (
        RecipeKind.TWO_STAGE_AUGMIX_LHAT, AuxiliaryVariant.MATCHED_NO_VAE)
    assert tuple(recipe.resources) == ("operator_profile", "augmix_config", "corruption_rng")
    assert recipe.comparison_rng_identity == "augmix_simclr_lhat"
    assert recipe.recipe_sha256 == (
        "c831f5a20a517b1ac72e432f506ea1fa47ad3a686226f13dc9eda690748272dc"
    )
    assert recipe.objective_terms == (
        ("clean_bce", "clean_view"),
        ("corrupted_bce", "corrupted_view"),
    )
    assert not recipe.requires_vae
    assert recipe.scientific_contract["stage2_teacher"] == "post_stage1_pre_stage2_snapshot"
    with pytest.raises(TypeError, match="loader-owned"):
        method_registry.RecipeSpec()
    with pytest.raises(TypeError, match="loader-owned"):
        replace(recipe, profile_name="a0_clean_v1")
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
]


@pytest.mark.parametrize("filename,view_name,composition,waveform_sha,trace", GENERATION_CASES)
def test_cpu_a0_a3_and_direct_generation_goldens(
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
