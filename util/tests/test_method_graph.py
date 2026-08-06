from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from core.methods import (
    ExecutionResources,
    Provenance,
    ValueKind,
    WaveformView,
    build_method_runtime,
    compile_method_profile,
    execute_method,
    load_method_profile,
)


ROOT = Path(__file__).resolve().parents[2]
METHOD_PROFILES = ROOT / "configs" / "train" / "methods"

LOCKED_PROFILES = (
    (
        "a0_clean_v1.yaml",
        "a0_clean_v1",
        "A0",
        ("clean_identity",),
        {"clean_view"},
        ("clean_bce",),
        ("classifier",),
    ),
    (
        "a3c_depth23_v1.yaml",
        "a3c_depth23_v1",
        "A3c",
        ("clean_identity", "depth23_corruption"),
        {"clean_view", "corrupted_view"},
        ("clean_bce", "corrupted_bce"),
        ("classifier",),
    ),
    (
        "augmix_simclr_lhat.yaml",
        "augmix_simclr_lhat",
        "augmix_simclr_lhat",
        ("clean_identity", "depth23_corruption", "lhat"),
        {"clean_view", "corrupted_view", "lhat_view"},
        (
            "clean_bce",
            "lhat_direct_bce",
            "corrupted_bce",
        ),
        ("classifier", "vae_decoder", "latent_pool"),
    ),
)

EXPERIMENTAL_PROFILES = (
    "exp_augmix_guided_latent_simplex_v1.yaml",
    "exp_lhat_as_sixth_branch_v1.yaml",
    "exp_lhat_replay_pool_v1.yaml",
)

FORBIDDEN_DYNAMIC_KEYS = (
    "callable",
    "class",
    "class_path",
    "import",
    "import_path",
    "module",
)


def _clean_view(batch_size: int = 2) -> WaveformView:
    waveform = torch.linspace(-1.0, 1.0, 1000, dtype=torch.float32)
    waveform = waveform.view(1, 1000, 1).repeat(batch_size, 1, 12)
    labels = torch.zeros(batch_size, 5, dtype=torch.float32)
    labels[:, 3] = 1.0
    sample_ids = tuple(f"fixture-{index}" for index in range(batch_size))
    return WaveformView(
        name="fixture_clean_raw",
        waveform=waveform,
        labels=labels,
        sample_ids=sample_ids,
        provenance=Provenance(
            node_id="fixture_source",
            operation="cpu_test_fixture",
        ),
    )


@pytest.mark.parametrize(
    (
        "filename",
        "profile_name",
        "scientific_arm",
        "topological_nodes",
        "outputs",
        "objective_terms",
        "requirements",
    ),
    LOCKED_PROFILES,
)
def test_locked_method_profiles_compile(
    filename: str,
    profile_name: str,
    scientific_arm: str,
    topological_nodes: tuple[str, ...],
    outputs: set[str],
    objective_terms: tuple[str, ...],
    requirements: tuple[str, ...],
) -> None:
    compiled = compile_method_profile(METHOD_PROFILES / filename)

    assert compiled.profile_name == profile_name
    assert compiled.scientific_arm == scientific_arm
    assert compiled.executable is True
    assert tuple(node.profile.node_id for node in compiled.nodes) == topological_nodes
    assert set(compiled.outputs) == outputs
    assert all(kind is ValueKind.WAVEFORM for kind in compiled.output_kinds.values())
    assert tuple(term.name for term in compiled.objective.terms) == objective_terms
    assert compiled.requirements.names() == requirements
    assert len(compiled.profile_sha256) == 64


def test_mainline_locks_rotating_four_and_direct_lhat_gradient_sum() -> None:
    compiled = compile_method_profile(
        METHOD_PROFILES / "augmix_simclr_lhat.yaml"
    )

    assert compiled.contracts["batch_norm_running_stats_policy"] == (
        "family_loss_weighted_once_per_base_batch"
    )
    assert compiled.contracts["family_loss_weights"] == {
        "clean": pytest.approx(0.5),
        "corrupted_total": pytest.approx(0.5),
        "corrupted_per_composition": pytest.approx(0.125),
    }
    assert compiled.contracts["auxiliary_alpha"] == pytest.approx(2.0)
    assert compiled.contracts["auxiliary_gradient_merge"] == "direct_sum"
    assert compiled.contracts["auxiliary_batch_norm_policy"] == "snapshot_restore"
    assert compiled.contracts["auxiliary_rng_policy"] == (
        "snapshot_restore_global_rng"
    )
    assert compiled.contracts["attack_then_contract_version"] == (
        "preflip_maxloss_grid_v1"
    )


@pytest.mark.parametrize("filename", EXPERIMENTAL_PROFILES)
def test_experimental_profiles_compile_but_execution_is_rejected(filename: str) -> None:
    compiled = compile_method_profile(METHOD_PROFILES / filename)

    assert compiled.scientific_arm == "experimental"
    assert compiled.executable is False
    with pytest.raises(
        RuntimeError,
        match=r"audit-only: contracts\.executable=false",
    ):
        execute_method(compiled, ExecutionResources(sources={}))


def test_latent_threechain_profile_is_executable_and_uses_supervised_augmix_loss() -> None:
    compiled = compile_method_profile(
        METHOD_PROFILES / "exp_paired_augmix_latent_bridge_v1.yaml"
    )

    assert compiled.profile_name == "latent_threechain_augmix_residual_depth23_aug075"
    assert compiled.scientific_arm == "latent_augmix"
    assert compiled.executable is True
    assert tuple(node.profile.node_id for node in compiled.nodes) == (
        "clean_identity",
        "augmix_view_1",
        "augmix_view_2",
    )
    assert set(compiled.outputs) == {
        "clean_view",
        "augmix_view_1",
        "augmix_view_2",
    }
    assert compiled.requirements.names() == (
        "classifier",
        "vae_encoder",
        "vae_decoder",
    )
    assert tuple(term.name for term in compiled.objective.terms) == (
        "clean_bce",
        "clean_augmix_jsd",
        "augmix_view_1_bce",
        "augmix_view_2_bce",
    )
    assert compiled.objective.terms[0].weight == pytest.approx(1.0)
    assert compiled.objective.terms[1].weight == pytest.approx(3.0)
    assert compiled.objective.terms[2].weight == pytest.approx(0.75)
    assert compiled.objective.terms[3].weight == pytest.approx(0.75)
    assert compiled.contracts["batch_norm_running_stats_policy"] == (
        "objective_view_weighted_once_per_base_batch"
    )
    assert compiled.contracts["batch_norm_objective_view_weights"] == {
        "clean_view": pytest.approx(0.5),
        "augmix_view_1": pytest.approx(0.25),
        "augmix_view_2": pytest.approx(0.25),
    }
    assert compiled.contracts["chain_depth_sampling"] == "uniform_integer_2_to_3"
    assert compiled.contracts["reconstruction_residual_bypass"] == (
        "weighted_chain_residuals"
    )


def test_latent_threechain_profile_rejects_resource_binding_drift() -> None:
    payload = yaml.safe_load(
        (METHOD_PROFILES / "exp_paired_augmix_latent_bridge_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["nodes"]["augmix_view_1"]["inputs"]["encoder"] = "vae_decoder"

    with pytest.raises(ValueError, match=r"inputs\.encoder.*vae_encoder"):
        load_method_profile(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda resources: resources["latent_augmix_rng"].__setitem__(
                "type", "typo"
            ),
            r"latent_augmix_rng\.type.*isolated_torch_generator",
        ),
        (
            lambda resources: resources["latent_augmix_rng"].pop("seed_config"),
            r"latent_augmix_rng keys must be exactly",
        ),
    ],
)
def test_latent_threechain_profile_rejects_rng_resource_schema_drift(
    mutation, message
) -> None:
    payload = yaml.safe_load(
        (METHOD_PROFILES / "exp_paired_augmix_latent_bridge_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    mutation(payload["resources"])

    with pytest.raises(ValueError, match=message):
        load_method_profile(payload)


def test_latent_threechain_profile_uses_two_replayable_independent_node_streams() -> None:
    compiled = compile_method_profile(
        METHOD_PROFILES / "exp_paired_augmix_latent_bridge_v1.yaml"
    )
    source = _clean_view()

    def fake_augmix(context, inputs):
        clean = inputs[0]
        generator = context.torch_generator("fixture", device=clean.waveform.device)
        offset = torch.rand((), generator=generator)
        return WaveformView(
            name=context.node_id,
            waveform=clean.waveform + offset,
            labels=clean.labels,
            sample_ids=clean.sample_ids,
            provenance=Provenance(
                node_id=context.node_id,
                operation=context.node_type,
                parent_names=(clean.name,),
                rng_namespace=context.rng_namespace,
            ),
        )

    def run_once():
        return execute_method(
            compiled,
            ExecutionResources(
                sources={"clean_raw": source},
                adapters={"augmix": fake_augmix},
                classifier=object(),
                vae_encoder=object(),
                vae_decoder=object(),
                base_seed=20260717,
                rng_identity=("replicate=0", "epoch=1", "step=0"),
            ),
        )

    first = run_once()
    second = run_once()
    left = first.require("augmix_view_1", ValueKind.WAVEFORM)
    right = first.require("augmix_view_2", ValueKind.WAVEFORM)
    assert not torch.equal(left.waveform, right.waveform)
    torch.testing.assert_close(
        left.waveform,
        second.require("augmix_view_1", ValueKind.WAVEFORM).waveform,
    )
    torch.testing.assert_close(
        right.waveform,
        second.require("augmix_view_2", ValueKind.WAVEFORM).waveform,
    )


def test_a0_executes_as_a_typed_clean_waveform_graph() -> None:
    compiled = compile_method_profile(METHOD_PROFILES / "a0_clean_v1.yaml")
    source = _clean_view()

    bundle = execute_method(
        compiled,
        ExecutionResources(
            sources={"clean_raw": source},
            classifier=object(),
            base_seed=20260717,
            rng_identity=("replicate=0", "epoch=1", "step=0"),
        ),
    )

    clean = bundle.require("clean_view", ValueKind.WAVEFORM)
    assert isinstance(clean, WaveformView)
    assert clean.name == "clean_identity"
    assert clean.sample_ids == source.sample_ids
    assert torch.equal(clean.waveform, source.waveform)
    assert torch.equal(clean.labels, source.labels)
    assert bool(clean.valid_mask.all())
    assert clean.provenance.node_id == "clean_identity"
    assert clean.provenance.operation == "identity_raw100_view"
    assert clean.provenance.parent_names == ("fixture_clean_raw",)
    assert clean.provenance.parameters == {"source": "clean_raw"}
    assert bundle.diagnostics["method/profile_name"] == "a0_clean_v1"
    assert bundle.diagnostics["method/profile_sha256"] == compiled.profile_sha256


def test_executor_prunes_nodes_not_required_by_the_selected_objective_view() -> None:
    compiled = compile_method_profile(METHOD_PROFILES / "a3c_depth23_v1.yaml")
    source = _clean_view()

    def forbidden_corruption(context: Any, inputs: tuple[Any, ...]) -> WaveformView:
        del context, inputs
        raise AssertionError("unselected corruption node must not execute")

    bundle = execute_method(
        compiled,
        ExecutionResources(
            sources={"clean_raw": source},
            adapters={"canonical_corruption": forbidden_corruption},
            classifier=object(),
            base_seed=20260717,
            rng_identity=("replicate=0", "epoch=1", "step=0"),
        ),
        required_outputs=("clean_view",),
    )

    assert tuple(bundle.values) == ("clean_view",)
    assert tuple(bundle.node_values) == ("clean_identity",)
    assert torch.equal(
        bundle.require("clean_view", ValueKind.WAVEFORM).waveform,
        source.waveform,
    )


def test_executor_rejects_unknown_or_empty_required_outputs() -> None:
    compiled = compile_method_profile(METHOD_PROFILES / "a0_clean_v1.yaml")
    resources = ExecutionResources(
        sources={"clean_raw": _clean_view()},
        classifier=object(),
        base_seed=20260717,
        rng_identity=("replicate=0", "epoch=1", "step=0"),
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        execute_method(compiled, resources, required_outputs=())
    with pytest.raises(ValueError, match="unknown outputs"):
        execute_method(compiled, resources, required_outputs=("missing_view",))


def test_fixed20_direct_profile_declares_family_balanced_matched_base_exposure() -> None:
    compiled = compile_method_profile(
        METHOD_PROFILES / "direct_depth23_fixed20.yaml"
    )

    assert compiled.profile_name == "direct_depth23_fixed20"
    assert compiled.scientific_arm == "DirectDepth23Fixed20FamilyBalanced"
    assert tuple(term.name for term in compiled.objective.terms) == (
        "clean_bce",
        "corrupted_bce",
    )
    assert compiled.contracts["exposure_policy"] == (
        "clean_once_then_exhaustive_depth23"
    )
    assert compiled.contracts["composition_indices"] == list(range(20))
    assert compiled.contracts["composition_depth_counts"] == {
        "depth2": 10,
        "depth3": 10,
    }
    assert compiled.contracts["objective_term_routing"] == {
        "clean_exposure": "clean_bce",
        "corrupted_exposure": "corrupted_bce",
    }
    assert compiled.contracts["family_loss_weights"] == {
        "clean": 0.5,
        "corrupted_total": 0.5,
        "corrupted_per_composition": 0.025,
    }
    assert compiled.contracts["optimizer_step_policy"] == (
        "accumulate_family_balanced_once_per_base_batch"
    )
    assert compiled.contracts["batch_norm_running_stats_policy"] == (
        "family_loss_weighted_once_per_base_batch"
    )
    assert compiled.contracts["exposure_budget_match"] == (
        "matched_one_optimizer_step_per_base_batch"
    )
    assert compiled.contracts["materialized_dataset_expansion_allowed"] is False


@pytest.mark.parametrize(
    ("filename", "requirements"),
    (
        ("direct_depth23_fixed20_raw_aux.yaml", ("classifier",)),
        (
            "direct_depth23_fixed20_vae_reconstruction_aux.yaml",
            ("classifier", "vae_encoder", "vae_decoder"),
        ),
        (
            "direct_depth23_fixed20_lhat_aux.yaml",
            ("classifier", "vae_decoder", "latent_pool"),
        ),
    ),
)
def test_fixed20_aux_profiles_compile_with_one_normalized_auxiliary_exposure(
    filename: str,
    requirements: tuple[str, ...],
) -> None:
    compiled = compile_method_profile(METHOD_PROFILES / filename)

    assert compiled.executable is True
    assert compiled.comparison_rng_identity == "direct_depth23_fixed20"
    assert compiled.requirements.names() == requirements
    assert compiled.contracts["exposure_policy"] == (
        "clean_once_then_exhaustive_depth23_then_auxiliary"
    )
    assert compiled.contracts["auxiliary_objective_terms"] == ["auxiliary_bce"]
    assert compiled.contracts["family_loss_weights"] == {
        "clean": 0.45,
        "corrupted_total": 0.45,
        "corrupted_per_composition": 0.0225,
        "auxiliary": 0.1,
    }
    assert tuple(term.name for term in compiled.objective.terms) == (
        "clean_bce",
        "corrupted_bce",
        "auxiliary_bce",
    )


def _fixed20_runtime():
    compiled = compile_method_profile(
        METHOD_PROFILES / "direct_depth23_fixed20.yaml"
    )
    return build_method_runtime(
        compiled,
        model_name="efficientnet1dv2",
        config_root=ROOT / "configs",
    )


def _generate_fixed20_runtime_view(
    *,
    composition_indices: torch.Tensor | None,
    rng_identity: tuple[str, ...] = ("replicate=0", "epoch=1", "step=1"),
):
    source = _clean_view()
    result = _fixed20_runtime().generate(
        clean_raw=source.waveform,
        targets=source.labels,
        hash_ids=source.sample_ids,
        classifier=torch.nn.Identity(),
        base_seed=20260717,
        rng_identity=rng_identity,
        composition_indices=composition_indices,
    )
    view = result.bundle.require("corrupted_view", ValueKind.WAVEFORM)
    assert isinstance(view, WaveformView)
    return result, view, source


def test_runtime_forces_explicit_canonical_composition_indices() -> None:
    result, view, _ = _generate_fixed20_runtime_view(
        composition_indices=torch.tensor([0, 19], dtype=torch.int64)
    )

    diagnostics = view.metadata["corruption_diagnostics"]
    assert diagnostics.composition_index.tolist() == [0, 19]
    assert diagnostics.depth.tolist() == [2, 3]
    assert view.provenance.parameters["composition_selection"] == "forced"
    assert result.bundle.diagnostics["depth23_corruption/mean_depth"] == 2.5


def test_runtime_all_minus_one_is_identity_without_consuming_corruption_rng() -> None:
    result, view, source = _generate_fixed20_runtime_view(
        composition_indices=torch.full((2,), -1, dtype=torch.int64)
    )

    assert torch.equal(view.waveform, source.waveform)
    assert view.sample_ids == source.sample_ids
    assert view.provenance.parameters["composition_selection"] == (
        "clean_identity_sentinel"
    )
    assert view.metadata["corruption_diagnostics"] is None
    assert result.bundle.diagnostics["depth23_corruption/mean_depth"] == 0.0
    assert not any(
        key.startswith("depth23_corruption/rng/")
        for key in result.bundle.diagnostics
    )


def test_reconstruction_auxiliary_executes_only_the_frozen_codec_branch() -> None:
    class Identity:
        def describe(self) -> dict[str, str]:
            return {"sha256": "a" * 64}

    class Encoder(torch.nn.Module):
        checkpoint_identity = Identity()

        def forward(self, value, *, sample=False):
            assert value.shape[1:] == (1024, 12)
            assert sample is False
            latent = value.new_zeros((value.shape[0], 4, 128))
            return latent, latent, latent

    class Decoder(torch.nn.Module):
        checkpoint_identity = Identity()

        def forward(self, latent):
            time = torch.linspace(
                -1.0,
                1.0,
                1024,
                device=latent.device,
                dtype=latent.dtype,
            ).view(1, 1024, 1)
            leads = torch.arange(
                12, device=latent.device, dtype=latent.dtype
            ).view(1, 1, 12)
            return time.expand(latent.shape[0], -1, 12) + leads * 0.01

    compiled = compile_method_profile(
        METHOD_PROFILES / "direct_depth23_fixed20_vae_reconstruction_aux.yaml"
    )
    runtime = build_method_runtime(
        compiled,
        model_name="efficientnet1dv2",
        config_root=ROOT / "configs",
        encoder=Encoder(),
        decoder=Decoder(),
    )
    source = _clean_view()

    generated = runtime.generate(
        clean_raw=source.waveform,
        targets=source.labels,
        hash_ids=source.sample_ids,
        classifier=torch.nn.Identity(),
        base_seed=20260717,
        rng_identity=("replicate=0", "epoch=1", "exposure=auxiliary"),
        objective_term_names=("auxiliary_bce",),
    )

    assert tuple(generated.bundle.values) == ("clean_view", "auxiliary_view")
    assert "depth23_corruption" not in generated.bundle.node_values
    assert tuple(generated.bundle.node_values) == (
        "clean_identity",
        "reconstruction_encode",
        "reconstruction_decode",
    )
    auxiliary = generated.bundle.require("auxiliary_view", ValueKind.WAVEFORM)
    assert auxiliary.waveform.shape == source.waveform.shape
    assert bool(auxiliary.valid_mask.all())
    assert auxiliary.provenance.parameters["quality_rejection_policy"] == (
        "clean_loss_only"
    )


def test_runtime_rejects_mixed_identity_and_corruption_indices() -> None:
    with pytest.raises(ValueError, match="may be all -1"):
        _generate_fixed20_runtime_view(
            composition_indices=torch.tensor([-1, 0], dtype=torch.int64)
        )


def test_runtime_none_preserves_deterministic_random_composition_sampling() -> None:
    first, first_view, _ = _generate_fixed20_runtime_view(
        composition_indices=None
    )
    second, second_view, _ = _generate_fixed20_runtime_view(
        composition_indices=None
    )

    first_diagnostics = first_view.metadata["corruption_diagnostics"]
    second_diagnostics = second_view.metadata["corruption_diagnostics"]
    assert torch.equal(
        first_diagnostics.composition_index,
        second_diagnostics.composition_index,
    )
    assert torch.equal(first_view.waveform, second_view.waveform)
    assert first_view.provenance.parameters["composition_selection"] == "random"
    assert any(
        key.startswith("depth23_corruption/rng/composition_and_operators/")
        for key in first.bundle.diagnostics
    )


def _random_corruption_adapter(context: Any, inputs: tuple[Any, ...]) -> WaveformView:
    assert len(inputs) == 1
    source = inputs[0]
    assert isinstance(source, WaveformView)
    generator = context.torch_generator("cpu_test", device=source.waveform.device)
    noise = torch.rand(
        source.waveform.shape,
        dtype=source.waveform.dtype,
        device=source.waveform.device,
        generator=generator,
    )
    context.record_diagnostic("noise_checksum", float(noise.sum()))
    return WaveformView(
        name=context.node_id,
        waveform=source.waveform + 0.01 * noise,
        labels=source.labels,
        sample_ids=source.sample_ids,
        valid_mask=source.valid_mask,
        provenance=Provenance(
            node_id=context.node_id,
            operation=context.node_type,
            parent_names=(source.name,),
            rng_namespace=context.rng_namespace,
            parameters={"stream": "cpu_test"},
        ),
    )


def _execute_compiled_with_rng_identity(
    compiled: Any,
    rng_identity: tuple[str, ...],
):
    return execute_method(
        compiled,
        ExecutionResources(
            sources={"clean_raw": _clean_view()},
            adapters={"canonical_corruption": _random_corruption_adapter},
            classifier=object(),
            base_seed=20260717,
            rng_identity=rng_identity,
        ),
    )


def _execute_a3c_with_rng_identity(rng_identity: tuple[str, ...]):
    compiled = compile_method_profile(METHOD_PROFILES / "a3c_depth23_v1.yaml")
    return _execute_compiled_with_rng_identity(compiled, rng_identity)


def test_default_comparison_rng_identity_preserves_profile_scoped_stream() -> None:
    compiled = compile_method_profile(METHOD_PROFILES / "a3c_depth23_v1.yaml")
    identity = ("replicate=0", "center=ningbo", "epoch=1", "step=7")

    bundle = _execute_compiled_with_rng_identity(compiled, identity)

    assert compiled.comparison_rng_identity == compiled.profile_name
    assert compiled.describe()["comparison_rng_identity"] == compiled.profile_name
    assert bundle.diagnostics["method/comparison_rng_identity"] == compiled.profile_name
    rng_key = "depth23_corruption/rng/cpu_test/cpu"
    assert bundle.diagnostics[rng_key]["comparison_rng_identity"] == (
        compiled.profile_name
    )


def _matched_rng_profile(
    *,
    method_id: str,
    comparison_rng_identity: str,
) -> Any:
    path = METHOD_PROFILES / "a3c_depth23_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["method"]["id"] = method_id
    payload["method"]["scientific_arm"] = method_id
    payload["contracts"]["comparison_rng_identity"] = comparison_rng_identity
    return compile_method_profile(payload)


def test_explicit_comparison_rng_identity_pairs_random_views_across_profiles() -> None:
    identity = ("replicate=0", "center=ningbo", "epoch=1", "step=7")
    first_method = _matched_rng_profile(
        method_id="matched_arm_first",
        comparison_rng_identity="matched_fixed20_aux_v1",
    )
    second_method = _matched_rng_profile(
        method_id="matched_arm_second",
        comparison_rng_identity="matched_fixed20_aux_v1",
    )

    first = _execute_compiled_with_rng_identity(first_method, identity)
    second = _execute_compiled_with_rng_identity(second_method, identity)
    first_view = first.require("corrupted_view", ValueKind.WAVEFORM)
    second_view = second.require("corrupted_view", ValueKind.WAVEFORM)
    rng_key = "depth23_corruption/rng/cpu_test/cpu"

    assert first_method.profile_sha256 != second_method.profile_sha256
    assert torch.equal(first_view.waveform, second_view.waveform)
    assert first.diagnostics[rng_key]["seed"] == second.diagnostics[rng_key]["seed"]


def test_fixed20_baseline_and_aux_profile_share_corruption_random_numbers() -> None:
    identity = (
        "pn2021_direct_depth23_fixed20_family_balanced",
        "replicate=0",
        "ningbo",
        "efficientnet1dv2",
        "epoch=1",
        "view_execution_step=2",
        "exposure=corruption_00",
        "batch_hash_sha256=fixture",
    )
    baseline = compile_method_profile(
        METHOD_PROFILES / "direct_depth23_fixed20.yaml"
    )
    auxiliary = compile_method_profile(
        METHOD_PROFILES / "direct_depth23_fixed20_raw_aux.yaml"
    )

    first = _execute_compiled_with_rng_identity(baseline, identity)
    second = _execute_compiled_with_rng_identity(auxiliary, identity)
    first_view = first.require("corrupted_view", ValueKind.WAVEFORM)
    second_view = second.require("corrupted_view", ValueKind.WAVEFORM)
    rng_key = "depth23_corruption/rng/cpu_test/cpu"

    assert baseline.comparison_rng_identity == baseline.profile_name
    assert auxiliary.comparison_rng_identity == baseline.profile_name
    assert torch.equal(first_view.waveform, second_view.waveform)
    assert first.diagnostics[rng_key]["seed"] == second.diagnostics[rng_key]["seed"]


def test_comparison_rng_identity_separates_random_views_when_requested() -> None:
    identity = ("replicate=0", "center=ningbo", "epoch=1", "step=7")
    first_method = _matched_rng_profile(
        method_id="matched_arm_first",
        comparison_rng_identity="comparison_family_a",
    )
    second_method = _matched_rng_profile(
        method_id="matched_arm_second",
        comparison_rng_identity="comparison_family_b",
    )

    first = _execute_compiled_with_rng_identity(first_method, identity)
    second = _execute_compiled_with_rng_identity(second_method, identity)
    first_view = first.require("corrupted_view", ValueKind.WAVEFORM)
    second_view = second.require("corrupted_view", ValueKind.WAVEFORM)
    rng_key = "depth23_corruption/rng/cpu_test/cpu"

    assert not torch.equal(first_view.waveform, second_view.waveform)
    assert first.diagnostics[rng_key]["seed"] != second.diagnostics[rng_key]["seed"]


@pytest.mark.parametrize("invalid_identity", [None, "", [], 0])
def test_comparison_rng_identity_must_be_a_non_empty_string(
    invalid_identity: Any,
) -> None:
    path = METHOD_PROFILES / "a3c_depth23_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["contracts"]["comparison_rng_identity"] = invalid_identity

    with pytest.raises(ValueError, match="comparison_rng_identity must be"):
        load_method_profile(payload)


def test_same_rng_identity_reproduces_the_same_typed_view() -> None:
    identity = ("replicate=0", "center=ningbo", "epoch=1", "step=7")

    first = _execute_a3c_with_rng_identity(identity)
    second = _execute_a3c_with_rng_identity(identity)

    first_view = first.require("corrupted_view", ValueKind.WAVEFORM)
    second_view = second.require("corrupted_view", ValueKind.WAVEFORM)
    assert isinstance(first_view, WaveformView)
    assert isinstance(second_view, WaveformView)
    assert torch.equal(first_view.waveform, second_view.waveform)
    assert first.diagnostics == second.diagnostics


def test_rng_identity_separates_different_outer_steps() -> None:
    step_7 = _execute_a3c_with_rng_identity(
        ("replicate=0", "center=ningbo", "epoch=1", "step=7")
    )
    step_8 = _execute_a3c_with_rng_identity(
        ("replicate=0", "center=ningbo", "epoch=1", "step=8")
    )

    view_7 = step_7.require("corrupted_view", ValueKind.WAVEFORM)
    view_8 = step_8.require("corrupted_view", ValueKind.WAVEFORM)
    assert isinstance(view_7, WaveformView)
    assert isinstance(view_8, WaveformView)
    assert not torch.equal(view_7.waveform, view_8.waveform)
    rng_key = "depth23_corruption/rng/cpu_test/cpu"
    assert step_7.diagnostics[rng_key]["seed"] != step_8.diagnostics[rng_key]["seed"]
    assert step_7.diagnostics[rng_key]["execution_identity"][-1] == "step=7"
    assert step_8.diagnostics[rng_key]["execution_identity"][-1] == "step=8"


@pytest.mark.parametrize("forbidden_key", FORBIDDEN_DYNAMIC_KEYS)
def test_profile_rejects_dynamic_import_or_callable_keys(forbidden_key: str) -> None:
    path = METHOD_PROFILES / "a0_clean_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["contracts"][forbidden_key] = "untrusted.module:factory"

    with pytest.raises(
        ValueError,
        match=rf"profile\.contracts\.{forbidden_key} is forbidden; "
        r"profiles may not select callables",
    ):
        load_method_profile(payload)


def test_compiler_requires_the_canonical_clean_base_view() -> None:
    path = METHOD_PROFILES / "a3c_depth23_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["outputs"].pop("clean_view")
    payload["objective"]["terms"] = [
        term
        for term in payload["objective"]["terms"]
        if term["id"] != "clean_bce"
    ]

    with pytest.raises(ValueError, match="canonical base view 'clean_view'"):
        compile_method_profile(payload)
