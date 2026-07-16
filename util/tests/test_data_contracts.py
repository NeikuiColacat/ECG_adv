"""CPU-only tests for data and preprocessing protocol contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ecg_adv_gen.config import ConfigError, load_experiment_config, validate_experiment_config
from ecg_adv_gen.data import (
    ECGTWIN_TO_PTBXL_INDICES,
    PN2021_EVAL_CENTERS_7,
    PN2021_TARGET_CENTERS_4,
    DataManifestError,
    build_data_path_manifest,
    get_data_preprocess_contract,
    validate_data_path_manifest,
    validate_data_preprocess_config,
    write_data_path_manifest,
)
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES as LEGACY_REORDER


REPO = Path(__file__).resolve().parents[2]
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _load(name: str) -> dict:
    return load_experiment_config(
        REPO / "configs" / "experiments" / name,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest_run"},
    )


def test_data_preprocess_contract_matches_current_pipeline_constants():
    contract = get_data_preprocess_contract()

    assert contract.source_dataset == "ptbxl"
    assert contract.target_dataset == "pn2021"
    assert contract.target_centers == PN2021_TARGET_CENTERS_4
    assert contract.eval_centers == PN2021_EVAL_CENTERS_7
    assert contract.classifier_fs == 100
    assert contract.classifier_len == 1000
    assert contract.crop_len == 1000
    assert contract.lead_order == "ptbxl"
    assert contract.ecgtwin_decode_input_len == 1024
    assert contract.ecgtwin_decode_output_len == 1000
    assert contract.ecgtwin_to_ptbxl_indices == tuple(LEGACY_REORDER)
    assert contract.ecgfounder_preprocess_policy == "official_ptbxl_eval"
    assert contract.ecgfounder_input_fs == 500
    assert contract.ecgfounder_input_len == 5000
    assert contract.ecgfounder_input_shape == (12, 5000)
    assert ECGTWIN_TO_PTBXL_INDICES == tuple(LEGACY_REORDER)


def test_ptbxl_dataset_normalizes_known_tc_and_ct_cache_layouts():
    from ecg_adv_gen.training import PTBXLDatasetScheme

    signals_tc = np.arange(2 * 1000 * 12, dtype=np.float32).reshape(2, 1000, 12)
    labels = np.zeros((2, 5), dtype=np.float32)
    sample_tc, _ = PTBXLDatasetScheme(signals_tc, labels, crop_len=1000, mode="eval")[0]
    sample_ct, _ = PTBXLDatasetScheme(
        signals_tc.transpose(0, 2, 1), labels, crop_len=1000, mode="eval"
    )[0]

    assert tuple(sample_tc.shape) == tuple(sample_ct.shape) == (12, 1000)
    np.testing.assert_array_equal(sample_ct.numpy(), sample_tc.numpy())
    with pytest.raises(ValueError, match="Expected classifier signals"):
        PTBXLDatasetScheme(np.zeros((2, 1000, 8), dtype=np.float32), labels)


def test_manual_pn2021_preprocess_uses_aligned_corner_linear_interpolation():
    from data_preprocess.PN2021_preprocess import _resample_crop_pad

    signal = np.arange(5, dtype=np.float32)[:, None]
    signal = np.repeat(signal, 12, axis=1)
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


def test_manual_data_configs_record_linear_interpolation_contract():
    pn2021 = yaml.safe_load((REPO / "configs" / "data" / "PN2021.yaml").read_text())
    ptbxl = yaml.safe_load((REPO / "configs" / "data" / "PTBXL.yaml").read_text())

    assert pn2021["target_interpolation"] == "linear_align_corners"
    assert pn2021["derived_interpolation"] == "linear_align_corners"
    assert ptbxl["derived_interpolation"] == "linear_align_corners"


def test_data_contract_tests_use_latest_mainline_or_active_fixtures():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    latest_config_names = {Path(stage["config"]).name for stage in index["latest_mainline"]["stages"]}
    public_config_names = {path.name for path in (REPO / "configs" / "experiments").glob("*.yaml")}

    assert "inactive_experiment_configs" not in index
    assert public_config_names == latest_config_names


def test_tracked_configs_follow_data_preprocess_contract():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    for stage in index["latest_mainline"]["stages"]:
        config = _load(Path(stage["config"]).name)
        validate_data_preprocess_config(config)
        validate_experiment_config(config, repo_root=REPO)


def test_validate_experiment_config_rejects_center_contract_drift():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["centers"]["target_4"] = ["ningbo", "georgia"]
    config["data"]["pn2021_centers"] = ["ningbo", "georgia"]

    with pytest.raises(ConfigError, match="paper_protocol.centers.target_4 mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_validate_experiment_config_rejects_preprocess_contract_drift():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["preprocess"]["classifier_fs"] = 500

    with pytest.raises(ConfigError, match="preprocess.classifier_fs mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_validate_experiment_config_rejects_ecgfounder_policy_drift():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    config = copy.deepcopy(config)
    config["model"]["preprocess_policy"] = "filtered_dataset"

    with pytest.raises(ConfigError, match="model.preprocess_policy mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_validate_experiment_config_rejects_ecgfounder_feature_shape_drift():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    config = copy.deepcopy(config)
    config["preprocess"]["ecgfounder"]["input_len"] = 1000

    with pytest.raises(ConfigError, match="preprocess.ecgfounder.input_len mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_validate_experiment_config_rejects_ecgfounder_policy_on_effnet():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    config = copy.deepcopy(config)
    config["model"]["preprocess_policy"] = "official_ptbxl_eval"

    with pytest.raises(ConfigError, match="reserved for ECGFounder"):
        validate_experiment_config(config, repo_root=REPO)


def _config_with_tmp_data_roots(tmp_path: Path) -> dict:
    config = copy.deepcopy(_load("effnet_direct_k500_v7_sjr_rgq.yaml"))
    data_root = tmp_path / "data"
    config["data"]["roots"] = {
        "ptbxl": str(data_root / "ptbxl"),
        "pn2021": str(data_root / "physionet2021"),
    }
    config["data"]["cache"] = {
        "pn2021_cache_dir": str(data_root / "triple_labels" / "pn2021_eval_cache"),
        "pn2021_mmap_cache_dir": str(data_root / "triple_labels" / "pn2021_eval_cache_mmap"),
        "prefer_mmap": True,
    }
    return config


def _materialize_manifest_dirs(config: dict) -> None:
    Path(config["data"]["roots"]["ptbxl"]).mkdir(parents=True, exist_ok=True)
    training = Path(config["data"]["roots"]["pn2021"]) / "training"
    for center in PN2021_EVAL_CENTERS_7:
        (training / center).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["cache"]["pn2021_cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["cache"]["pn2021_mmap_cache_dir"]).mkdir(parents=True, exist_ok=True)


def test_data_path_manifest_records_expected_roots_and_centers(tmp_path: Path):
    config = _config_with_tmp_data_roots(tmp_path)
    _materialize_manifest_dirs(config)

    manifest = build_data_path_manifest(config, local_paths={"write_boundary": str(tmp_path)})

    assert manifest["schema_version"] == 1
    assert manifest["summary"]["n_required_missing"] == 0
    assert manifest["summary"]["target_center_dirs_present"] == sorted(PN2021_TARGET_CENTERS_4)
    center_records = {r["center"]: r for r in manifest["pn2021_center_dirs"]}
    assert set(center_records) == set(PN2021_EVAL_CENTERS_7)
    assert center_records["ningbo"]["in_target_4"] is True
    assert center_records["ptb"]["in_target_4"] is False
    assert all(r["under_write_boundary"] is True for r in manifest["roots"])


def test_data_path_manifest_can_count_center_headers_without_recursive_scan(tmp_path: Path):
    config = _config_with_tmp_data_roots(tmp_path)
    _materialize_manifest_dirs(config)
    training = Path(config["data"]["roots"]["pn2021"]) / "training"
    ningbo = training / "ningbo"
    ptb = training / "ptb"
    for name in ["N001.hea", "N002.hea"]:
        (ningbo / name).write_text("mock header\n", encoding="utf-8")
    (ningbo / "N001.mat").write_text("not counted\n", encoding="utf-8")
    nested = ningbo / "g1"
    nested.mkdir()
    (nested / "N003.hea").write_text("counted from one-level subgroup\n", encoding="utf-8")
    deeper = nested / "deeper"
    deeper.mkdir()
    (deeper / "N004.hea").write_text("not counted recursively\n", encoding="utf-8")
    (ptb / "P001.hea").write_text("mock leak header\n", encoding="utf-8")

    manifest = build_data_path_manifest(
        config,
        local_paths={"write_boundary": str(tmp_path)},
        include_counts=True,
    )

    assert manifest["counting"] == {
        "include_counts": True,
        "loads_waveforms": False,
        "pn2021_header_count_pattern": "*.hea",
        "pn2021_header_count_scope": "center_dir_and_immediate_subdirs",
        "recursive_scan": False,
    }
    center_records = {r["center"]: r for r in manifest["pn2021_center_dirs"]}
    assert center_records["ningbo"]["record_count_available"] is True
    assert center_records["ningbo"]["n_header_files"] == 3
    assert center_records["chapman_shaoxing"]["n_header_files"] == 0
    assert center_records["ptb"]["n_header_files"] == 1
    assert manifest["summary"]["pn2021_center_header_counts"]["ningbo"] == 3
    assert manifest["summary"]["target_center_header_counts"]["ningbo"] == 3
    assert manifest["summary"]["target_center_header_count_total"] == 3
    assert manifest["summary"]["eval_center_header_count_total"] == 4


def test_write_data_path_manifest_and_require_existing(tmp_path: Path):
    config = _config_with_tmp_data_roots(tmp_path)
    _materialize_manifest_dirs(config)
    out = tmp_path / "out" / "data_manifest.json"

    manifest = write_data_path_manifest(
        config,
        out,
        local_paths={"write_boundary": str(tmp_path)},
        require_existing=True,
    )

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"] == manifest["summary"]
    validate_data_path_manifest(payload, require_existing=True)


def test_data_path_manifest_require_existing_rejects_missing_centers(tmp_path: Path):
    config = _config_with_tmp_data_roots(tmp_path)
    Path(config["data"]["roots"]["ptbxl"]).mkdir(parents=True)
    (Path(config["data"]["roots"]["pn2021"]) / "training").mkdir(parents=True)

    manifest = build_data_path_manifest(config, local_paths={"write_boundary": str(tmp_path)})
    assert manifest["summary"]["n_required_missing"] > 0
    with pytest.raises(DataManifestError, match="Missing required data path"):
        validate_data_path_manifest(manifest, require_existing=True)


def test_data_path_manifest_rejects_paths_outside_boundary(tmp_path: Path):
    config = _config_with_tmp_data_roots(tmp_path)
    config["data"]["roots"]["ptbxl"] = str(tmp_path.parent / "outside_ptbxl")

    with pytest.raises(DataManifestError, match="outside write boundary"):
        build_data_path_manifest(config, local_paths={"write_boundary": str(tmp_path)})
