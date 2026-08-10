"""Typed-graph contracts for the retained executable method profiles."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from core.methods import (
    ExecutionResources,
    Provenance,
    ValueKind,
    WaveformView,
    compile_method_profile,
    execute_method,
    load_method_profile,
)


REPO = Path(__file__).resolve().parents[2]
METHOD_PROFILES = REPO / "configs" / "train" / "methods"


def _clean_view() -> WaveformView:
    waveform = torch.linspace(-1.0, 1.0, 1000).view(1, 1000, 1).repeat(2, 1, 12)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 3] = 1.0
    return WaveformView(
        name="fixture_clean_raw",
        waveform=waveform,
        labels=labels,
        sample_ids=("fixture-0", "fixture-1"),
        provenance=Provenance(
            node_id="fixture_source",
            operation="cpu_test_fixture",
        ),
    )


@pytest.mark.parametrize(
    ("filename", "profile_name", "nodes", "requirements"),
    [
        (
            "a0_clean_v1.yaml",
            "a0_clean_v1",
            ("clean_identity",),
            ("classifier",),
        ),
        (
            "a3c_depth23_v1.yaml",
            "a3c_depth23_v1",
            ("clean_identity", "depth23_corruption"),
            ("classifier",),
        ),
        (
            "augmix_simclr_lhat.yaml",
            "augmix_simclr_lhat",
            ("clean_identity", "depth23_corruption", "lhat"),
            ("classifier", "vae_decoder", "latent_pool"),
        ),
    ],
)
def test_locked_profiles_compile_to_typed_resource_closed_graphs(
    filename: str,
    profile_name: str,
    nodes: tuple[str, ...],
    requirements: tuple[str, ...],
) -> None:
    compiled = compile_method_profile(METHOD_PROFILES / filename)

    assert compiled.profile_name == profile_name
    assert compiled.executable is True
    assert tuple(node.profile.node_id for node in compiled.nodes) == nodes
    assert all(kind is ValueKind.WAVEFORM for kind in compiled.output_kinds.values())
    assert compiled.requirements.names() == requirements
    assert len(compiled.profile_sha256) == 64


def test_mainline_profile_locks_the_attack_contract_and_family_balance() -> None:
    compiled = compile_method_profile(METHOD_PROFILES / "augmix_simclr_lhat.yaml")

    assert compiled.contracts["exposure_policy"] == (
        "clean_aux_once_then_rotating_depth23_2plus2"
    )
    assert compiled.contracts["family_loss_weights"] == {
        "clean": pytest.approx(0.5),
        "corrupted_total": pytest.approx(0.5),
        "corrupted_per_composition": pytest.approx(0.125),
    }
    assert compiled.contracts["auxiliary_alpha"] == pytest.approx(2.0)
    assert compiled.contracts["auxiliary_gradient_merge"] == "direct_sum"
    assert compiled.contracts["attack_then_contract_version"] == (
        "preflip_maxloss_grid_v1"
    )


def test_a0_executes_without_dynamic_imports_as_a_typed_identity_graph() -> None:
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
    assert torch.equal(clean.waveform, source.waveform)
    assert torch.equal(clean.labels, source.labels)
    assert clean.sample_ids == source.sample_ids
    assert clean.provenance.operation == "identity_raw100_view"
    assert bundle.diagnostics["method/profile_sha256"] == compiled.profile_sha256


@pytest.mark.parametrize(
    "filename",
    [
        "exp_augmix_guided_latent_simplex_v1.yaml",
        "exp_lhat_as_sixth_branch_v1.yaml",
        "exp_lhat_replay_pool_v1.yaml",
    ],
)
def test_audit_only_profiles_compile_for_review_but_cannot_execute(filename: str) -> None:
    compiled = compile_method_profile(METHOD_PROFILES / filename)

    assert compiled.executable is False
    with pytest.raises(RuntimeError, match=r"audit-only: contracts\.executable=false"):
        execute_method(compiled, ExecutionResources(sources={}))


def test_profile_schema_rejects_dynamic_import_keys() -> None:
    payload = yaml.safe_load(
        (METHOD_PROFILES / "a0_clean_v1.yaml").read_text(encoding="utf-8")
    )
    payload["nodes"]["clean_identity"]["module"] = "arbitrary.user.module"

    with pytest.raises(ValueError, match="forbidden; profiles may not select callables"):
        load_method_profile(payload)
