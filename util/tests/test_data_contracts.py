"""CPU-only contracts for the manually rebuilt data and model boundary."""

from __future__ import annotations

import hashlib
import json
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


def test_canonical_input_adapter_repairs_nonfinite_without_mutating_raw() -> None:
    raw = torch.arange(
        CANONICAL_POINTS * CANONICAL_CHANNELS,
        dtype=torch.float32,
    ).reshape(1, CANONICAL_POINTS, CANONICAL_CHANNELS)
    raw[0, 0, :3] = torch.tensor([float("nan"), float("inf"), float("-inf")])
    original = raw.clone()

    adapted = prepare_canonical_model_input(raw, EFFICIENTNET1DV2_SPEC)

    assert bool(torch.isfinite(adapted).all())
    torch.testing.assert_close(raw, original, equal_nan=True)


@pytest.mark.parametrize("spec", (EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC))
def test_canonical_input_adapter_maps_flat_samples_to_finite_zeros(spec) -> None:
    raw = torch.full(
        (2, CANONICAL_POINTS, CANONICAL_CHANNELS),
        3.0,
        dtype=torch.float32,
    )

    adapted = prepare_canonical_model_input(raw, spec)

    assert bool(torch.isfinite(adapted).all())
    assert torch.count_nonzero(adapted).item() == 0


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


def test_tracked_data_content_ledger_identity_is_consistent() -> None:
    ledger_path = CONFIG_ROOT / "data" / "data_content_ledger_v1.jsonl"
    raw = ledger_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    header, members = rows[0], rows[1:]
    digest = hashlib.sha256(raw).hexdigest()
    expected_sha = "d3bf1f18046a695b9c6089f2866cdb854042c3b528434018ecab8f56d36af665"
    expected_roots = {
        "ptbxl_cache": (8, 6287849438),
        "pn2021_cache": (8, 19152506828),
        "pn2021c_cache": (12, 379178277220),
        "split_artifacts": (93, 133085401),
    }
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert digest == expected_sha
    assert header == {
        "algorithm": "sha256", "artifact": "ecg_data_content_ledger",
        "member_count": 121, "roots": list(expected_roots),
        "schema_version": 1, "total_size_bytes": 404751718887,
    }
    assert len(members) == 121
    assert [(item["root"], item["path"]) for item in members] == sorted(
        (item["root"], item["path"]) for item in members
    )
    for root, (count, size) in expected_roots.items():
        selected = [item for item in members if item["root"] == root]
        assert (len(selected), sum(item["size_bytes"] for item in selected)) == (
            count, size
        )

    data_load = yaml.safe_load((CONFIG_ROOT / "data" / "data_load.yaml").read_text())
    handoff = yaml.safe_load((CONFIG_ROOT / "data" / "k500_handoff.yaml").read_text())
    scripts = yaml.safe_load((CONFIG_ROOT / "active_scripts.yaml").read_text())
    evidence = yaml.safe_load((CONFIG_ROOT / "active_evidence_registry.yaml").read_text())
    descriptor = data_load["content_ledger"]
    assert descriptor == {
        "path": "data/data_content_ledger_v1.jsonl", "sha256": expected_sha
    }
    assert handoff["required_configs"]["data_load"]["sha256"] == hashlib.sha256(
        (CONFIG_ROOT / "data" / "data_load.yaml").read_bytes()
    ).hexdigest()
    assert handoff["required_artifacts"]["data_content_ledger"]["sha256"] == expected_sha
    assert scripts["latest_mainline"]["verification"]["data_ledger_tool_contract"][
        "tracked_content_ledger"
    ]["sha256"] == expected_sha
    assert evidence["data_contract"]["prospective_managed_content_ledger"][
        "sha256"
    ] == expected_sha


def test_post_cp4_cuda_smoke_is_registered_as_diagnostic_only() -> None:
    scripts = yaml.safe_load((CONFIG_ROOT / "active_scripts.yaml").read_text())
    evidence = yaml.safe_load(
        (CONFIG_ROOT / "active_evidence_registry.yaml").read_text()
    )
    smoke = scripts["latest_mainline"]["verification"]["gpu_replication"][
        "post_cp4_diagnostic_smoke"
    ]

    assert smoke["source_git_sha"] == (
        "929c05eb9c9b21af892a93913aa8ebd1981758f5"
    )
    assert smoke["summary_sha256"] == (
        "54a98d2793a0547a07393e01c4316fb463bcabb2a096ba2aca38d87751bcd561"
    )
    assert smoke["claim_boundary"] == "migration_smoke_only_not_performance_evidence"
    assert smoke["common_contract"] == {
        "center": "ningbo",
        "repeats": 1,
        "epochs": 1,
        "mainline_stage1_steps": 2,
        "stage2_outer_optimizer_steps_per_job": 20,
        "adaptation_records": 500,
    }
    assert smoke["data_content"]["ledger_sha256"] == (
        "d3bf1f18046a695b9c6089f2866cdb854042c3b528434018ecab8f56d36af665"
    )
    assert smoke["managed_job_file_index_sha256"] == {
        "direct_effnet": "7d5f90f4bdfa8bc2392c234aab54fea0485262faaf23f882b9c740a35bd4a322",
        "mainline_effnet": "00957d123284f0c62fe064338fbbfb9914d52c4e1727b80370dda5af91509a02",
        "mainline_ecgfounder": "4467486d327344b657ca260b447d46207a24e9c579bacb36cdd9bb21c2b63fbb",
    }
    assert smoke["final_checkpoint_sha256"] == {
        "direct_effnet": "d566ed238eec69cddf7874089b70bae30fad411f9940737b7414cb7beb66cd46",
        "mainline_effnet": "9c3f3361e1f9b4c0eefc406855779b7ae11aa2cafb20ae88a0b191475be1aee6",
        "mainline_ecgfounder": "d633a79b4e24937f71ed260f8fb36e6b2b39cce2632c7f0bc0b52ce3332c9a64",
    }
    assert "throughput_not_comparable" in smoke["limitations"]
    assert evidence["active_development_mainline"]["managed_implementation"][
        "gpu_replication_status"
    ] == "diagnostic_smoke_passed_not_full_replication"


def test_k500_handoff_locks_finite_loader_openings() -> None:
    handoff = yaml.safe_load(
        (CONFIG_ROOT / "data" / "k500_handoff.yaml").read_text(encoding="utf-8")
    )
    plans = handoff["interface"]["public_loader_plans"]
    expected_openings = {
        "ptbxl_source": ["open_train", "open_validation", "open_test"],
        "pn2021_k500": ["open_ordered", "open_training"],
        "pn2021_evaluation": ["open_clean", "open_corrupted", "open_session"],
    }
    expected_shuffle = {
        "ptbxl_source": [True, False, False],
        "pn2021_k500": [False, True],
        "pn2021_evaluation": [False, False, False],
    }
    assert handoff["schema_version"] == 2
    assert handoff["profile_name"] == "pn2021_k500_data_interface_handoff"
    assert list(plans) == list(expected_openings)
    for name, openings in expected_openings.items():
        assert plans[name]["openings"] == openings
        assert list(plans[name]["shuffle_by_opening"]) == openings
        assert list(plans[name]["shuffle_by_opening"].values()) == expected_shuffle[name]
    assert "shuffle" not in handoff["interface"]["canonical_raw_request"]

    for identity in handoff["required_configs"].values():
        path = CONFIG_ROOT / identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]

    modules = handoff["required_modules"]
    assert set(modules["runtime"]).isdisjoint(modules["cache_builders"])
    assert [
        group for group, paths in modules.items()
        if "data_preprocess/split_cache.py" in paths
    ] == ["split_builders"]


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


def test_direct_historical_ledger_is_complete_and_immutable() -> None:
    direct = yaml.safe_load(
        (CONFIG_ROOT / "baselines" / "pn2021_direct_v1.yaml").read_text()
    )
    lock = direct["historical_managed_run_lock"]
    centers = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
    models = ("efficientnet1dv2", "ecgfounder")
    expected_ids = {
        *(f"{model}/tune/{center}" for model in models for center in centers),
        *(f"{model}/select/all4" for model in models),
        *(f"{model}/refit/{center}" for model in models for center in centers),
        *(f"{model}/eval/{center}" for model in models for center in centers),
    }
    runs = lock["managed_runs"]
    run_ids = [run["identity"]["id"] for run in runs]

    protocol = lock["protocol"]
    assert (
        protocol["adaptation_records_per_center"],
        protocol["selection_records"],
        protocol["tuning_partition"],
        protocol["refit_partition"],
    ) == (500, 400, "400_train_plus_100_validation_per_center", "all_k500_no_validation")
    assert protocol["outside_k500_model_access"] is False
    assert protocol["training_k500_evidence"] == {
        "run_embedded_hash": False, "evidence_level": "config_and_count"
    }
    assert protocol["evaluation_k500_exclusion_evidence"] == {
        "run_embedded_hash": True, "evidence_level": "exact_set_identity",
        "split_manifest_sha256": "58210d9b192bc18e9085f235453094814c9caed2cf9142998084ae7a004e9a42",
    }
    assert protocol["evaluation_aggregation"] == "within_center_then_equal_four_center_mean"
    assert lock["method_contract"]["loss"] == "0.5_clean_bce_plus_0.5_mean_twenty_corrupted_bce"
    method = lock["historical_method"]
    assert method["profile_path"] == "configs/train/methods/direct_depth23_fixed20.yaml"
    assert method["profile_file_sha256"] == "d41e77219726f983ad90a0b9db98cd99668c354dc903e83ecbbe24e6b53b5c9f"
    assert method["compiled_profile_identity_sha256"] == "336187af8edddaba54a07ae88a84561c5d6e87720a93d892ad115f648b71c2f3"
    assert set(run_ids) == expected_ids and len(run_ids) == len(set(run_ids)) == 26
    assert lock["run_matrix"] == {
        "eval": 8, "refit": 8, "select": 2, "total": 26, "tune": 8
    }
    assert {
        kind: sum(run["identity"]["kind"] == kind for run in runs)
        for kind in ("tune", "select", "refit", "eval")
    } == {"tune": 8, "select": 2, "refit": 8, "eval": 8}
    groups = {
        group["id"]: set(group["run_ids"])
        for group in lock["execution_provenance"]["dirty_groups"]
    }
    assert set().union(*groups.values()) == expected_ids
    assert sum(map(len, groups.values())) == 26
    assert all(
        run["identity"]["id"] in groups[run["provenance"]["dirty_group"]]
        for run in runs
    )
    assert sum(run["index"]["file_count"] for run in runs) == 1348
    assert sum(run["index"]["indexed_size_bytes"] for run in runs) == 3750461884

    verification = lock["verification"]
    assert verification["full_run_file_indexes_verified"] is True
    assert verification["source_checkpoint_bytes_verified"] is True
    catalog = lock["historical_config_snapshot_catalog"]
    catalog_ids = {
        (item["path"], item["identity"]["sha256"]) for item in catalog
    }
    assert len(catalog) == len(catalog_ids) == 42
    assert {
        (item["identity"]["sha256"], item["identity"]["occurrence_count"])
        for item in catalog
        if item["path"] == "configs/train/PN2021_fixed20.yaml"
    } == {
        ("5bab9e1e9508f22b810756b5c9e8ff4ab8c6e8c0976a21436442faa66d2dc95f", 5),
        ("3f482f9684d8a010a899243a9b3237bffbe3300f4c41430be2c4c590df715f86", 13),
    }
    for run in runs:
        for role in ("entry_config", "delegate_config"):
            identity = run[role]
            assert (identity["path"], identity["sha256"]) in catalog_ids

    boundary = lock["claim_boundary"]
    assert boundary["replay_level"] == "metric_level_not_bitwise"
    assert boundary["bitwise_replay_claimed"] is False
    assert lock["execution_provenance"]["dirty"] is True
    assert lock["execution_provenance"]["source_patch_available"] is False
    for name in ("training_config", "tuning_config"):
        identity = direct["references"][name]
        assert identity["sha256_scope"] == "historical_run_snapshot"
        assert identity["retained_live_path"] is False

    stored = verification["canonical_ledger_sha256"]
    canonical = {**lock, "verification": {
        key: value for key, value in verification.items()
        if key != "canonical_ledger_sha256"
    }}
    computed = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    assert stored == computed == "ab6999e48f8b684a61ea934de71ff646bdf8bbf305c11dd50a4a228f06e23288"
    active = yaml.safe_load((CONFIG_ROOT / "active_evidence_registry.yaml").read_text())
    identity = active["active_development_mainline"]["evidence"]["direct_registry"]
    direct_path = CONFIG_ROOT / "baselines" / "pn2021_direct_v1.yaml"
    assert identity["current_sha256"] == hashlib.sha256(direct_path.read_bytes()).hexdigest()
    assert identity["current_relation"] == (
        "historical_direct_metric_ledger_live_tune_select_refit_retired"
    )


def test_direct_historical_selection_and_eval_crosslinks_are_recomputable() -> None:
    direct = yaml.safe_load(
        (CONFIG_ROOT / "baselines" / "pn2021_direct_v1.yaml").read_text()
    )
    lock = direct["historical_managed_run_lock"]
    runs = {run["identity"]["id"]: run for run in lock["managed_runs"]}
    expected = {
        "efficientnet1dv2": (5e-5, 128, 30, 120, 23, 92, 6413827,
            "c930d19312bb8200c2043ee811fefc0865a18c4d79d769ef8350a58c660232c9"),
        "ecgfounder": (2e-5, 64, 20, 140, 20, 160, 30670389,
            "4a3ab6a9f5dbe544bb3d1f4ceb87415987ee70484363dccee129755b39ac2b12"),
    }
    for model, (lr, batch, horizon, tune_steps, epoch, refit_steps, params, identity) in expected.items():
        frozen = lock["models"][model]
        selection = frozen["selection"]
        composition = frozen["selected_composition_metrics"]
        optimization = frozen["optimization"]
        assert len(composition) == len({item["id"] for item in composition}) == 20
        assert np.mean([item["macro_auroc"] for item in composition]) == pytest.approx(
            selection["robust"]["macro_auroc"]
        )
        assert np.mean([item["macro_auprc"] for item in composition]) == pytest.approx(
            selection["robust"]["macro_auprc"]
        )
        assert selection["clean"]["floor"] == pytest.approx(
            selection["clean_reference"]["macro_auprc"] - 0.01
        )
        assert selection["clean"]["macro_auprc"] >= selection["clean"]["floor"]
        assert selection["score"] == pytest.approx(0.5 * (
            selection["clean"]["macro_auprc"]
            + selection["robust"]["macro_auprc"]
        ))
        assert selection["comparison_identity_sha256"] == identity
        assert selection["frozen_corruption_identity_sha256"] == (
            "ab38a829f1a7b9748c4348b38a4c893fc766555bdbc36c7a99d87657c0c2ce11"
        )
        assert len(selection["clean_reference"]["artifact_sha256"]) == 64
        training, schedule = optimization["training"], optimization["schedule"]
        assert isinstance(training["learning_rate"], float)
        assert training == {
            "optimizer": "adamw", "learning_rate": lr, "weight_decay": 1e-4,
            "batch_size": batch, "trainable_scope": "full",
        }
        assert schedule == {
            "scheduler": "cosine_annealing", "horizon_epochs": horizon,
            "tuning_optimizer_steps": tune_steps, "selected_refit_epoch": epoch,
            "refit_optimizer_steps": refit_steps,
        }
        assert optimization["numerics"] == {
            "gradient_clip_norm": 1.0, "amp_dtype": "bfloat16"
        }
        assert optimization["parameter_counts"] == {
            "trainable": params, "total": params
        }
        assert frozen["source_checkpoint"] == {
            "path": direct["models"][model]["source_checkpoint"],
            "sha256": direct["models"][model]["source_checkpoint_sha256"],
        }

        for center, chain in frozen["centers"].items():
            refit = runs[chain["run_ids"]["refit"]]
            evaluation = runs[chain["run_ids"]["evaluation"]]
            artifacts = chain["refit_artifacts"]
            assert refit["primary"]["sha256"] == artifacts["refit_contract_sha256"]
            assert evaluation["primary"]["sha256"] == chain["evaluation_result_sha256"]
            legacy = direct["models"][model]["centers"][center]
            for key in ("checkpoint_sha256", "train_result_sha256", "refit_contract_sha256"):
                assert artifacts[key] == legacy[key]
            assert chain["evaluation_result_sha256"] == legacy["evaluation_result_sha256"]
            assert chain["metrics"] == legacy["metrics"]
            entry = CONFIG_ROOT.parent / evaluation["entry_config"]["path"]
            config = yaml.safe_load(entry.read_text())
            assert hashlib.sha256(entry.read_bytes()).hexdigest() == evaluation["entry_config"]["sha256"]
            args = config["entrypoint"]["arguments"]
            value = lambda flag: args[args.index(flag) + 1]
            checkpoint = Path(lock["run_root"]) / refit["identity"]["basename"] / lock["artifact_paths"]["checkpoint"]
            assert config["entrypoint"]["name"] == "evaluate_pn2021"
            assert (value("--model"), value("--center"), value("--checkpoint")) == (model, center, str(checkpoint))
            assert len(artifacts["checkpoint_sha256"]) == len(chain["evaluation_result_sha256"]) == 64

        for view in ("clean_kept", "clean_drop", "corrupted_kept", "corrupted_drop"):
            aggregate = np.mean([chain["metrics"][view] for chain in frozen["centers"].values()], axis=0)
            np.testing.assert_allclose(aggregate, frozen["four_center_mean"][view])
            assert frozen["four_center_mean"][view] == direct["models"][model]["four_center_mean"][view]


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
