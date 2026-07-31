from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
HANDOFF_PATH = CONFIG_ROOT / "data" / "k500_handoff.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _handoff() -> dict:
    payload = yaml.safe_load(HANDOFF_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_required_config_bundle_is_complete_and_hash_locked() -> None:
    required = _handoff()["required_configs"]
    assert {
        "ptbxl_dataset",
        "pn2021_dataset",
        "pn2021_mapping",
        "splits",
        "data_load",
        "random_seed",
        "evaluation",
        "corruption_cache",
        "corruption_operators",
        "source_training",
        "vae",
        "source_weight_registry",
    } == set(required)

    for identity in required.values():
        path = CONFIG_ROOT / identity["path"]
        assert path.is_file(), path
        assert _sha256(path) == identity["sha256"]


def test_required_code_surface_exists_without_training_mainline() -> None:
    required_modules = _handoff()["required_modules"]
    declared = {
        relative
        for group in required_modules.values()
        for relative in group
    }
    for relative in declared:
        assert (REPO / relative).is_file(), relative

    for excluded in (
        "agent_workspace",
        "boot_scripts",
        "core",
        "ecg_adv_gen",
        "methods",
        "model",
        "scripts",
    ):
        assert not (REPO / excluded).exists(), excluded


def test_k500_protocol_locks_seed_mapping_and_four_centers() -> None:
    handoff = _handoff()
    protocol = handoff["protocol"]
    assert protocol["split_id"] == "pn2021_super5_k500_cpsc_combined_v1"
    assert protocol["base_seed"] == 20260501
    assert protocol["seed_namespace"] == "ecg_manual_refactor_split_v1"
    assert protocol["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert protocol["mapping_hash"] == "555ec85d5b51"
    assert protocol["class_order"] == ["CD", "HYP", "MI", "NORM", "STTC"]

    centers = handoff["split_artifacts"]["logical_centers"]
    assert set(centers) == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for identity in centers.values():
        assert identity["k500"]["count"] == 500
        assert identity["k500_tune_train"]["count"] == 400
        assert identity["k500_tune_validation"]["count"] == 100
        for partition in (
            "k500",
            "k500_tune_train",
            "k500_tune_validation",
            "evaluation_all_zero_kept",
            "evaluation_drop_all_zero",
        ):
            assert len(identity[partition]["hash_id_set_sha256"]) == 64


def test_external_bundle_declares_only_verified_split_and_weight_artifacts() -> None:
    handoff = _handoff()
    bundle = handoff["artifact_bundle"]
    assert bundle["git_tracked"] is False
    assert bundle["contains_raw_ecg"] is False
    assert bundle["contains_waveform_caches"] is False
    assert bundle["contains_frozen_splits"] is True
    assert bundle["contains_model_weights"] is True

    expected_paths = {
        "weights/ecgtwin/vae_model.pth",
        "weights/ecgfounder/12_lead_ECGFounder.pth",
        "weights/ptbxl_source/efficientnet1dv2_best.pt",
        "weights/ptbxl_source/ecgfounder_best.pt",
    }
    artifacts = handoff["model_artifacts"]
    assert {
        identity["packaged_checkpoint"] for identity in artifacts.values()
    } == expected_paths
    for identity in artifacts.values():
        assert identity["checkpoint_size_bytes"] > 0
        assert len(identity["checkpoint_sha256"]) == 64
        assert not (REPO / identity["packaged_checkpoint"]).exists()
