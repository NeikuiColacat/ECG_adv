"""CPU-only contracts for the manually rebuilt data and model boundary."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data_preprocess import preprocess_primitives as primitives
from data_preprocess import split_cache
from data_preprocess import data_runtime, load_cache as cache_owner
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from data_preprocess.pn2021_metadata import parse_pn2021_header
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
import models.efficientnet1d as efficientnet1d
import models.ecgfounder as ecgfounder
from models.factory import build_model
from util import config_bundle
from util.config_bundle import resolve_yaml_config_closure


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"


@pytest.mark.parametrize("payload", [{}, {"record": "宁波", "nested": {"labels": [1, 0]}}])
def test_split_and_cache_json_mapping_share_the_same_reader(tmp_path: Path, payload: dict) -> None:
    assert data_runtime._read_json_mapping is cache_owner._read_json_mapping
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert data_runtime._read_json_mapping(path, description="split manifest") == payload


@pytest.mark.parametrize("content", [None, '{"record":', "[]"])
def test_shared_json_mapping_reader_preserves_errors(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "manifest.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    for description in ("cache manifest", "split manifest"):
        expected_type = FileNotFoundError if content is None else ValueError
        with pytest.raises(expected_type) as caught:
            data_runtime._read_json_mapping(path, description=description)
        error = caught.value
        if content is None:
            assert str(error) == f"{description} not found: {path}"
            assert error.__cause__ is None and error.__suppress_context__
        elif content == "[]":
            assert str(error) == f"{description} must be a JSON mapping: {path}"
            assert error.__cause__ is None
        else:
            assert isinstance(error.__cause__, json.JSONDecodeError)
            assert str(error) == f"invalid {description} JSON at {path}: {error.__cause__}"


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
    assert {"require_mapping", "load_yaml_mapping"}.issubset(config_bundle.__all__)
    assert split_cache._require_mapping is config_bundle.require_mapping
    assert split_cache._read_yaml_mapping is config_bundle.load_yaml_mapping
    assert EFFICIENTNET1DV2_SPEC.input_shape == (12, 1000)
    assert EFFICIENTNET1DV2_SPEC.sampling_rate_hz == 100
    assert ECGFOUNDER_SPEC.input_shape == (12, 5000)
    assert ECGFOUNDER_SPEC.sampling_rate_hz == 500
    assert tuple(inspect.signature(efficientnet1d.EfficientNet1DV2).parameters) == ()
    assert efficientnet1d.__all__ == [
        "EfficientNet1DV2", "build_efficientnet1dv2"]
    assert {"get_backbone_config", "get_activation"}.isdisjoint(vars(efficientnet1d))
    with pytest.raises(TypeError):
        build_model("efficientnet1dv2", variant="s_v2")
    model = efficientnet1d.EfficientNet1DV2()
    assert len(model.state_dict()) == 571
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_413_827
    assert len(list(model.named_modules(remove_duplicate=False))) == 687
    assert sum(isinstance(block, efficientnet1d.FusedMBConv1d)
               for block in model.features) == 16
    assert sum(isinstance(block, efficientnet1d.MBConv1d)
               for block in model.features) == 18
    assert tuple(inspect.signature(ecgfounder.ECGFounderNet1D).parameters) == ()
    assert ecgfounder.__all__ == ["ECGFounderNet1D", "build_ecgfounder"]
    with pytest.raises(TypeError):
        build_model("ecgfounder", base_filters=64)
    founder = ecgfounder.ECGFounderNet1D()
    assert len(founder.state_dict()) == 509
    assert sum(parameter.numel() for parameter in founder.parameters()) == 30_670_389
    assert len(list(founder.named_modules(remove_duplicate=False))) == 415
    assert [len(stage.block_list) for stage in founder.stage_list] == [2, 2, 2, 3, 3, 4, 4]


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


def test_pn2021_header_labels_and_demographics_share_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header_path = tmp_path / "record.hea"
    header_path.write_text(
        "record 12 500 5000\n# Age: 63\n# Sex: Female\n"
        "# Dx: 426783006, 22298006\n",
        encoding="utf-8",
    )
    original_open = Path.open
    open_count = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal open_count
        if path == header_path:
            open_count += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    snomed_codes, metadata = parse_pn2021_header(header_path)

    assert open_count == 1
    assert snomed_codes == [426783006, 22298006]
    assert metadata == {"age": 63.0, "sex": "F", "hr": None}


def test_cache_builders_share_exact_waveform_primitives(tmp_path: Path) -> None:
    shared_names = set(
        "WaveformQualityError _load_config _load_yaml_mapping "
        "_repair_nonfinite_per_lead _validate_nonfinite_policy _resolve_path "
        "_normalise_lead_name _reorder_leads _read_wfdb _record_hash "
        "_fixed_unicode_array _linear_interpolate_time_batch".split()
    )
    for name in ("PTBXL_preprocess.py", "PN2021_preprocess.py"):
        tree = ast.parse((REPO / "data_preprocess" / name).read_text())
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        assert shared_names.isdisjoint(definitions)

    signal = np.arange(8, dtype=np.float32).reshape(4, 2)
    signal[0, 0], signal[2, 1] = np.inf, np.nan
    repaired, quality = primitives._repair_nonfinite_per_lead(
        signal, max_record_fraction=0.3, max_lead_fraction=0.3
    )
    np.testing.assert_array_equal(repaired, [[2, 1], [2, 3], [4, 5], [6, 7]])
    assert quality["repaired_nonfinite_count"] == 2
    assert quality["repair_method"] == "linear_per_lead_nearest_edge"
    rejected = np.column_stack((np.arange(4), np.full(4, np.nan)))
    with pytest.raises(primitives.WaveformQualityError) as captured:
        primitives._repair_nonfinite_per_lead(
            rejected, max_record_fraction=1.0, max_lead_fraction=1.0
        )
    assert captured.value.reason == "all_nonfinite_lead"
    assert captured.value.details["nonfinite_count_by_lead"] == [0, 4]
    resized = primitives._linear_interpolate_time_batch(
        np.arange(6, dtype=np.float32).reshape(1, 3, 2), 5
    )
    np.testing.assert_array_equal(
        resized,
        np.array([[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]], dtype=np.float32),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    for dataset in ("PTB-XL", "PN2021"):
        mapping_error = (
            rf"^{dataset} config must be a YAML mapping: "
            rf"{re.escape(str(config_path))}$"
        )
        with pytest.raises(ValueError, match=mapping_error):
            primitives._load_yaml_mapping(dataset, config_path)
        with pytest.raises(
            ValueError, match=rf"^{dataset} config must define nonfinite_policy$"
        ):
            primitives._validate_nonfinite_policy(dataset, {})


def test_data_ledger_direct_cli_resolves_project_imports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO / "data_preprocess" / "data_ledger.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_data_yaml_persists_only_the_canonical_100hz_contract() -> None:
    pn2021 = yaml.safe_load((CONFIG_ROOT / "data" / "PN2021.yaml").read_text())
    ptbxl = yaml.safe_load((CONFIG_ROOT / "data" / "PTBXL.yaml").read_text())

    assert pn2021["target_sampling_rate_hz"] == 100
    assert pn2021["target_num_samples"] == 1000
    assert pn2021["target_interpolation"] == "linear_align_corners"
    derived_keys = {
        "derived_sampling_rate_hz",
        "derived_num_samples",
        "derived_interpolation",
        "derived_batch_size",
    }
    assert derived_keys.isdisjoint(pn2021)
    assert derived_keys.isdisjoint(ptbxl)
    assert ptbxl["sampling_rate_hz"] == 100
    assert ptbxl["num_samples"] == 1000
    assert pn2021["lead_order"] == list(EXPECTED_LEADS)


def test_all_current_config_references_resolve_inside_the_bundle() -> None:
    entries = sorted((CONFIG_ROOT / "experiments").glob("*.yaml"))
    selectors = sorted((CONFIG_ROOT / "train" / "methods").glob("*.yaml"))
    assert entries and selectors
    for entry in [*entries, *selectors]:
        closure = resolve_yaml_config_closure([entry], config_root=CONFIG_ROOT)
        assert entry.resolve() in closure
        assert all(path.is_file() and path.is_relative_to(CONFIG_ROOT) for path in closure)


def test_preprocessing_mapping_closure_matches_its_owner_directory() -> None:
    entry = CONFIG_ROOT / "data" / "PN2021.yaml"
    mapping = CONFIG_ROOT / "data" / "PN2021_super5_v7.yaml"
    assert resolve_yaml_config_closure([entry], config_root=CONFIG_ROOT) == (
        entry.resolve(), mapping.resolve(),
    )


def test_mapping_closure_is_portable_and_does_not_select_root_decoy(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    entry = data / "PN2021.yaml"
    mapping = data / "mapping.yaml"
    entry.write_text("label_mapping_file: mapping.yaml\n")
    mapping.write_text("mapping_version: owner_local\n")
    (tmp_path / "mapping.yaml").write_text("mapping_version: wrong_root\n")
    assert resolve_yaml_config_closure([entry], config_root=tmp_path) == (entry, mapping)


@pytest.mark.parametrize("reference", ["../../outside.yaml", "/outside.yaml", "~/outside.yaml"])
def test_owner_relative_mapping_cannot_escape_config_bundle(tmp_path: Path, reference: str) -> None:
    data = tmp_path / "data"
    data.mkdir()
    entry = data / "PN2021.yaml"
    entry.write_text(yaml.safe_dump({"label_mapping_file": reference}))
    with pytest.raises(ValueError, match="escapes config bundle|must be relative"):
        resolve_yaml_config_closure([entry], config_root=tmp_path)


def test_tracked_data_content_ledger_identity_is_consistent() -> None:
    ledger_path = CONFIG_ROOT / "data" / "data_content_ledger_v1.jsonl"
    raw = ledger_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    header, members = rows[0], rows[1:]
    digest = hashlib.sha256(raw).hexdigest()
    expected_sha = "5991ac2123e9fe8774df7cb4a311fd0f4332554e51a77ea2a29ac2ce05ad5f13"
    expected_roots = {
        "ptbxl_cache": (7, 1056089310),
        "pn2021_cache": (7, 3213386700),
        "pn2021c_cache": (11, 63218277092),
        "split_artifacts": (93, 133085401),
    }
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert digest == expected_sha
    assert header == {
        "algorithm": "sha256", "artifact": "ecg_data_content_ledger",
        "member_count": 118, "roots": list(expected_roots),
        "schema_version": 1, "total_size_bytes": 67620838503,
    }
    assert len(members) == 118
    assert [(item["root"], item["path"]) for item in members] == sorted(
        (item["root"], item["path"]) for item in members
    )
    for root, (count, size) in expected_roots.items():
        selected = [item for item in members if item["root"] == root]
        assert (len(selected), sum(item["size_bytes"] for item in selected)) == (
            count, size
        )

    data_load = yaml.safe_load((CONFIG_ROOT / "data" / "data_load.yaml").read_text())
    assert data_load["content_ledger"] == {
        "path": "data/data_content_ledger_v1.jsonl", "sha256": expected_sha
    }

def test_main_matrix_matches_training_recipe_seed_and_evaluation() -> None:
    from core.methods.registry import load_recipe_spec

    matrix = yaml.safe_load((CONFIG_ROOT / "eval/PN2021_matrix.yaml").read_text())
    split = yaml.safe_load((CONFIG_ROOT / "data/splits.yaml").read_text())
    centers = {item["name"]: item["source_centers"] for item in split["pn2021"]["logical_centers"]}
    for short, model in (("efficientnet", "efficientnet1dv2"), ("ecgfounder", "ecgfounder")):
        entry = yaml.safe_load((CONFIG_ROOT / f"experiments/{short}_train.yaml").read_text())
        args = entry["entrypoint"]["arguments"]
        selector = args[args.index("--method-config") + 1]
        recipe = load_recipe_spec(CONFIG_ROOT / selector)
        train_path = CONFIG_ROOT / entry["entrypoint"]["config"]
        train = yaml.safe_load(train_path.read_text())
        profile = train["training"]["model_profiles"][model]
        cohort = matrix["profiles"][model]["expected_cohort"]
        assert cohort["method"]["recipe_id"] == recipe.recipe_id
        assert cohort["method"]["recipe_spec_sha256"] == recipe.recipe_sha256
        assert cohort["training_config_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
        assert cohort["evaluation_config_sha256"] == hashlib.sha256(
            (CONFIG_ROOT / "eval/PN2021_online_gpu.yaml").read_bytes()
        ).hexdigest()
        assert cohort["comparison"] == {"group": train["random_seed"]["comparison_group"], "replicate_id": train["random_seed"]["replicate_id"]}
        assert cohort["seed"]["namespace"] == train["random_seed"]["stream_namespace"]
        assert cohort["epochs_completed"] == profile["epochs"]
        assert cohort["optimizer_steps"] == profile["epochs"] * ((500 + profile["batch_size"] - 1) // profile["batch_size"])
        assert {name: item["source_centers"] for name, item in cohort["center_adaptation"].items()} == centers


def test_training_and_evaluation_examples_bind_the_same_run_and_recipe() -> None:
    for short in ("efficientnet", "ecgfounder"):
        train = yaml.safe_load((CONFIG_ROOT / f"experiments/{short}_train.yaml").read_text())
        evaluation = yaml.safe_load((CONFIG_ROOT / f"experiments/{short}_eval.yaml").read_text())
        train_args = dict(zip(train["entrypoint"]["arguments"][::2], train["entrypoint"]["arguments"][1::2]))
        eval_args = dict(zip(evaluation["entrypoint"]["arguments"][::2], evaluation["entrypoint"]["arguments"][1::2]))
        for key in ("--model", "--center", "--method-config"):
            assert train_args[key] == eval_args[key]
        assert Path(eval_args["--train-result"]) == Path(train["output"]["run_dir"]) / "training/train_result.json"
        profile = yaml.safe_load((CONFIG_ROOT / evaluation["entrypoint"]["config"]).read_text())
        assert profile["runtime"]["corruption_source"] == "online_torch_gpu_v1"
