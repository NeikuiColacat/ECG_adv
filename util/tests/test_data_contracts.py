"""CPU-only contracts for the manually rebuilt data and model boundary."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    validate_model_input,
    validate_model_output,
)
from models.input_adapter import (
    CANONICAL_CHANNELS,
    CANONICAL_POINTS,
    CANONICAL_SAMPLING_RATE_HZ,
    ECGFOUNDER_TARGET_POINTS,
    INTERPOLATION_ALIGN_CORNERS,
    INTERPOLATION_MODE,
    prepare_canonical_model_input,
)
from util.config_bundle import resolve_yaml_config_closure


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
KEEP_MANIFEST = (
    REPO / "docs" / "refactor_cleanup" / "manual_refactor_keep_manifest.md"
)


def _manual_paths() -> set[Path]:
    text = KEEP_MANIFEST.read_text(encoding="utf-8")
    return {
        (REPO / value).resolve()
        for value in re.findall(r"`((?:configs|data_preprocess|models)/[^`]+)`", text)
        if not any(char in value for char in "*<>")
    }


def _manifest_section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_data_and_model_constants_share_one_super5_contract() -> None:
    assert CLASS_ORDER == EXPECTED_CLASS_ORDER
    assert CLASS_ORDER == ("CD", "HYP", "MI", "NORM", "STTC")
    assert EXPECTED_LEADS == (
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    assert EFFICIENTNET1DV2_SPEC.input_shape == (12, 1000)
    assert EFFICIENTNET1DV2_SPEC.sampling_rate_hz == 100
    assert ECGFOUNDER_SPEC.input_shape == (12, 5000)
    assert ECGFOUNDER_SPEC.sampling_rate_hz == 500


def test_canonical_input_adapter_preserves_the_locked_operation_order() -> None:
    raw = torch.arange(
        2 * CANONICAL_POINTS * CANONICAL_CHANNELS,
        dtype=torch.float32,
    ).reshape(2, CANONICAL_POINTS, CANONICAL_CHANNELS)

    effnet = prepare_canonical_model_input(raw, EFFICIENTNET1DV2_SPEC)
    founder = prepare_canonical_model_input(raw, ECGFOUNDER_SPEC)

    assert CANONICAL_SAMPLING_RATE_HZ == 100
    assert INTERPOLATION_MODE == "linear"
    assert INTERPOLATION_ALIGN_CORNERS is True
    assert ECGFOUNDER_TARGET_POINTS == 5000
    assert tuple(effnet.shape) == (2, 12, 1000)
    assert tuple(founder.shape) == (2, 12, 5000)
    assert effnet.is_contiguous() and founder.is_contiguous()
    torch.testing.assert_close(
        effnet.mean(dim=(1, 2)),
        torch.zeros(2),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        founder.std(dim=(1, 2), correction=0),
        torch.ones(2),
        atol=2e-6,
        rtol=0.0,
    )


def test_model_contract_rejects_layout_and_output_drift() -> None:
    with pytest.raises(ValueError, match="input shape"):
        validate_model_input(torch.zeros(2, 1000, 12), EFFICIENTNET1DV2_SPEC)
    with pytest.raises(ValueError, match="raw logits"):
        validate_model_output(
            torch.zeros(2, 4),
            EFFICIENTNET1DV2_SPEC,
            batch_size=2,
        )


def test_pn2021_preprocess_uses_aligned_corner_linear_interpolation() -> None:
    from data_preprocess.PN2021_preprocess import _resample_crop_pad

    signal = np.repeat(np.arange(5, dtype=np.float32)[:, None], 12, axis=1)
    resized, details = _resample_crop_pad(
        signal,
        source_fs=5.0,
        target_fs=2,
        duration_seconds=1,
        target_num_samples=2,
        window_policy="center",
    )

    np.testing.assert_allclose(resized[:, 0], [0.0, 4.0])
    assert details["valid_target_samples"] == 2
    assert details["was_padded"] is False

    short, short_details = _resample_crop_pad(
        signal[:4],
        source_fs=4.0,
        target_fs=2,
        duration_seconds=2,
        target_num_samples=4,
        window_policy="center",
    )
    np.testing.assert_allclose(short[:, 0], [0.0, 3.0, 0.0, 0.0])
    assert short_details["valid_target_samples"] == 2
    assert short_details["was_padded"] is True


def test_data_yaml_records_the_same_interpolation_and_layout_contract() -> None:
    pn2021 = yaml.safe_load((CONFIG_ROOT / "data" / "PN2021.yaml").read_text())
    ptbxl = yaml.safe_load((CONFIG_ROOT / "data" / "PTBXL.yaml").read_text())

    assert pn2021["target_sampling_rate_hz"] == 100
    assert pn2021["target_num_samples"] == 1000
    assert pn2021["target_interpolation"] == "linear_align_corners"
    assert pn2021["derived_sampling_rate_hz"] == 500
    assert pn2021["derived_num_samples"] == 5000
    assert pn2021["derived_interpolation"] == "linear_align_corners"
    assert ptbxl["derived_interpolation"] == "linear_align_corners"
    assert pn2021["lead_order"] == list(EXPECTED_LEADS)


def test_current_managed_config_closure_stays_inside_the_keep_manifest() -> None:
    entry = (
        CONFIG_ROOT
        / "experiments"
        / "manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml"
    )
    closure = resolve_yaml_config_closure([entry], config_root=CONFIG_ROOT)
    manual_paths = _manual_paths()

    assert entry.resolve() in closure
    assert closure
    assert set(closure) <= manual_paths


def test_active_index_exposes_only_the_manual_launcher_and_whitelist_configs() -> None:
    path = CONFIG_ROOT / "active_scripts.yaml"
    text = path.read_text(encoding="utf-8")
    index = yaml.safe_load(text)
    manual_paths = _manual_paths()

    assert index["launch_surface_policy"]["default_launcher"] == (
        "boot_scripts/run_experiment.py"
    )
    assert index["latest_mainline"]["launcher"] == "boot_scripts/run_experiment.py"
    assert index["launch_surface_policy"]["legacy_runtime_imports_allowed"] is False
    assert "\n  default_launcher: scripts/run_experiment.py" not in text
    assert "\n  launcher: scripts/run_experiment.py" not in text
    assert "ecg_adv_gen/" not in text
    for stage in index["latest_mainline"]["stages"]:
        config = (REPO / stage["config"]).resolve()
        assert config.is_file()
        assert config in manual_paths


def test_active_evidence_hashes_the_retained_lock_and_report() -> None:
    registry = yaml.safe_load(
        (CONFIG_ROOT / "active_evidence_registry.yaml").read_text(encoding="utf-8")
    )
    evidence = registry["active_development_mainline"]["evidence"]

    for key in ("machine_lock", "technical_report", "direct_registry", "random_seed"):
        identity = evidence[key]
        path = (REPO / identity["path"]).resolve()
        assert path.is_file()
        live_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if identity.get("sha256_scope") == "historical_run_snapshot":
            assert live_sha256 == identity["current_sha256"]
            snapshot = identity["snapshot_git"]
            historical = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{snapshot['commit']}:{snapshot['path']}",
                ],
                cwd=REPO,
            )
            assert hashlib.sha256(historical).hexdigest() == identity["sha256"]
        else:
            assert live_sha256 == identity["sha256"]
    assert registry["active_development_mainline"]["paper_claim_allowed"] is False
    assert registry["historical_trusted_snapshot"]["replay"][
        "live_legacy_code_required"
    ] is False


def test_active_evidence_quarantines_accepted_subset_lhat_diagnostics() -> None:
    registry = yaml.safe_load(
        (CONFIG_ROOT / "active_evidence_registry.yaml").read_text(encoding="utf-8")
    )
    active = registry["active_development_mainline"]
    integrity = active["diagnostic_integrity"]

    assert registry["schema_version"] == 3
    assert str(registry["updated"]) == "2026-08-12"
    assert integrity["status"] == "legacy_accepted_subset_quarantined"
    assert integrity["all_candidate_raw_diagnostics"] == {
        "availability": "unavailable",
        "backfill_from_existing_artifacts": False,
        "required_evidence": "post_fix_run_with_all_candidate_diagnostics",
    }

    replication = active["prospective_replication_20260806"]
    cases = (
        (active["diagnostics"]["efficientnet1dv2"], 43769, 40156),
        (active["diagnostics"]["ecgfounder"], 57090, 52892),
        (replication["efficientnet1dv2"]["diagnostics"], 43769, 39905),
        (replication["ecgfounder"]["diagnostics"], 57090, 52676),
    )
    legacy_unscoped = {
        "raw_search_sample_anyflip_asr",
        "decoded_invalid_rate_max",
    }
    for diagnostics, candidate_count, accepted_count in cases:
        assert legacy_unscoped.isdisjoint(diagnostics)
        candidate = diagnostics["candidate_eligible"]
        accepted = diagnostics["accepted_training_view"]
        all_candidate = diagnostics["all_candidate_raw_search"]
        legacy = diagnostics["legacy_accepted_subset"]

        assert candidate["sample_count"] == candidate_count
        assert candidate["accepted_training_view_count"] == accepted_count
        assert candidate["acceptance_rate"] == pytest.approx(
            accepted_count / candidate_count
        )
        assert accepted["sample_count"] == accepted_count
        assert accepted["contracted_sample_anyflip_asr"] == 0.0
        assert all_candidate == {
            "availability": "unavailable",
            "sample_anyflip_asr": None,
            "decoded_invalid_rate_max": None,
        }
        assert legacy["sample_count"] == accepted_count
        assert legacy["use"] == "audit_only_not_all_candidate_mechanism_evidence"


def test_manifest_test_inventory_matches_the_collected_clean_tree() -> None:
    text = KEEP_MANIFEST.read_text(encoding="utf-8")
    protected = _manifest_section(text, "### A9.", "## B.")
    historical = _manifest_section(text, "### B1.", "### B2.")
    pattern = r"`(util/tests/(?:__init__|test_[^`]+)\.py)`"
    protected_paths = {REPO / value for value in re.findall(pattern, protected)}
    historical_paths = {REPO / value for value in re.findall(pattern, historical)}
    live_paths = set((REPO / "util" / "tests").glob("*.py"))

    assert protected_paths == live_paths
    assert not protected_paths.intersection(historical_paths)
    assert all(not path.exists() for path in historical_paths)
