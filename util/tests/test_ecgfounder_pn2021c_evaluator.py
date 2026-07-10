from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ecg_adv_gen.data.waveform_datasets as waveform_datasets
import ecg_adv_gen.models.ecgfounder_inference as ecgfounder_inference
from ecg_adv_gen.data.waveform_datasets import (
    ECGFounderBottleneck5000CorruptedDataset,
)
from ecg_adv_gen.models.ecgfounder_heads import OperatorConditionedLogitAdapter
from ecg_adv_gen.models.ecgfounder_runtime import (
    OperatorConditionedFullFTModel,
    detect_eval_mode,
    extract_fullft_state_dict,
    fullft_model_path,
)
from ecg_adv_gen.models.ecgfounder_inference import (
    infer_ecgfounder_fullft,
)
from ecg_adv_gen.evaluation.pn2021c import stable_seed


REPO = Path(__file__).resolve().parents[2]
TARGET_POINTS = 5000

PUBLIC_OUTPUT_KEYS = [
    "scheme", "model_name", "eval_mode", "variant", "run_dir", "corruption_input",
    "pn2021_root", "clean_eval_json", "head_path", "model_path", "checkpoint",
    "center_from_run", "centers", "corruptions", "severities", "severity_profile",
    "severity_params_file", "severity_params_name", "severity_profile_metadata",
    "crop_len", "required_cache_version", "pn2021_c_cache_version",
    "min_target_ref_excluded", "input_stabilizer", "pn2021c_protocol",
    "corruption_order", "selected_ref_record_ids_count", "label_mapping",
    "class_names", "clean_metric_overrides", "per_center",
    "aggregate_by_corruption_severity",
]


def _seed_encoding_apply_sequence(
    ecg_ct,
    corruption,
    public_severity,
    severity_profile,
    *,
    base_seed,
    seed_parts,
    sample_rate_hz=None,
    severity_profile_params=None,
):
    del corruption, public_severity, severity_profile, sample_rate_hz, severity_profile_params
    sample_seed = stable_seed(base_seed, *seed_parts)
    encoded = torch.zeros_like(ecg_ct)
    encoded[0, 0] = sample_seed & 0xFFFF
    encoded[0, 1] = sample_seed >> 16
    encoded[0, 2] = int(seed_parts[-1])
    return encoded


def _resume_identity() -> dict[str, object]:
    input_order = ["raw", "bottleneck5000", "corruption", "zscore", "model"]
    protocol = {
        "input_order_id": "locked",
        "model_family": "ecgfounder",
        "waveform_order": input_order,
    }
    return {
        "resolved_evaluator_semantic_config": {
            "protocol": {"name": "official_s5_depth23_composite"},
            "pn2021c_protocol": protocol,
            "corruption_input": "bottleneck5000",
            "centers": ["ningbo"],
            "corruptions": [f"op{i}" for i in range(5)],
            "severities": [5],
        },
        "scored_checkpoint": {"path": "/tmp/last_model.pt", "sha256": "a" * 64},
        "seed": 20260501,
        "ref_record_ids_sha256_by_center": {"ningbo": "b" * 64},
        "label_mapping": {
            "pn2021_super5": {"mapping_version": "v7", "mapping_hash": "555ec85d5b51"}
        },
        "input_order": input_order,
    }


def _base_public_output(load_time_s: float = 1.0) -> dict[str, object]:
    values = {
        "scheme": "super5", "model_name": "ECGFounder", "eval_mode": "fullft_model",
        "variant": "test", "run_dir": "/tmp/run", "corruption_input": "bottleneck5000",
        "pn2021_root": "/tmp/pn2021", "clean_eval_json": "/tmp/run/eval_result.json",
        "head_path": None, "model_path": "/tmp/run/last_model.pt",
        "checkpoint": "/tmp/backbone.pt", "center_from_run": "ningbo",
        "centers": ["ningbo"], "corruptions": [f"op{i}" for i in range(5)],
        "severities": [5], "severity_profile": "standard", "severity_params_file": None,
        "severity_params_name": None, "severity_profile_metadata": None, "crop_len": 1000,
        "required_cache_version": "v7", "pn2021_c_cache_version": "v7",
        "min_target_ref_excluded": 500, "input_stabilizer": {"kwargs": {}},
        "pn2021c_protocol": copy.deepcopy(
            _resume_identity()["resolved_evaluator_semantic_config"]["pn2021c_protocol"]
        ),
        "corruption_order": ["raw", "bottleneck5000", "corruption", "zscore", "model"],
        "selected_ref_record_ids_count": 500,
        "label_mapping": _resume_identity()["label_mapping"],
        "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
        "clean_metric_overrides": {
            "ningbo": {
                "metadata": {"load_time_s": load_time_s},
                "macro_auroc": 0.9,
                "macro_auprc": 0.7,
            }
        },
        "per_center": {},
    }
    return values


def _public_row(corruption: str, severity: int, value: float, load_time_s: float) -> dict[str, object]:
    per_class = {name: {"auroc": value, "auprc": value} for name in ("CD", "HYP", "MI", "NORM", "STTC")}
    ref_hash = "b" * 64
    protocol = copy.deepcopy(
        _resume_identity()["resolved_evaluator_semantic_config"]["pn2021c_protocol"]
    )
    pn2021c = {
        "center": "ningbo",
        "corruption": corruption,
        "severity": severity,
        "public_severity": severity,
        "internal_severity": severity * 2,
        "seed": 20260501,
        "n_excluded_ref": 500,
        "ref_record_ids_sha256": ref_hash,
        "input_order_id": protocol["input_order_id"],
    }
    return {
        "cache_path": None,
        "cache_source": "stream:raw_first",
        "metadata": {
            "load_time_s": load_time_s,
            "pn2021c": pn2021c,
            "pn2021_c": dict(pn2021c),
            "pn2021c_protocol": protocol,
        },
        "metadata_compatibility": {
            "compatible": None,
            "input_order_id": protocol["input_order_id"],
            "ref_record_ids_sha256": ref_hash,
        },
        "n_excluded_ref_ids_for_center": 500,
        "ref_record_ids_sha256": ref_hash,
        "corruption": {
            "name": corruption,
            "public_severity": severity,
            "internal_severity": severity * 2,
            "seed": 20260501,
        },
        "n_records": 5,
        "n_all_zero_labels": 0,
        "macro_auroc": value,
        "macro_auprc": value,
        "n_classes_used": 5,
        "per_class": per_class,
        "drop_all_zero_n_records": 5,
        "drop_all_zero_macro_auroc": value,
        "drop_all_zero_macro_auprc": value,
        "drop_all_zero_n_classes_used": 5,
        "drop_all_zero_per_class": per_class,
        "clean_macro_auroc": 0.9,
        "clean_macro_auprc": 0.7,
        "clean_metric_source": "recomputed_bottleneck5000_clean",
        "auroc_drop_vs_clean": 0.9 - value,
        "auprc_drop_vs_clean": 0.7 - value,
    }


def _test_payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_detect_eval_mode_distinguishes_feature_head_and_fullft(tmp_path: Path):
    feature_run = tmp_path / "feature"
    feature_run.mkdir()
    (feature_run / "best_head.pt").write_bytes(b"head")

    fullft_run = tmp_path / "fullft"
    fullft_run.mkdir()
    (fullft_run / "best_model.pt").write_bytes(b"model")

    locked_last_run = tmp_path / "locked_last"
    locked_last_run.mkdir()
    (locked_last_run / "last_model.pt").write_bytes(b"model")

    assert detect_eval_mode(feature_run) == "feature_head"
    assert detect_eval_mode(fullft_run) == "fullft_model"
    assert detect_eval_mode(locked_last_run) == "fullft_model"
    assert fullft_model_path(locked_last_run) == locked_last_run / "last_model.pt"

    mixed_run = tmp_path / "mixed_fullft_with_stale_head"
    mixed_run.mkdir()
    (mixed_run / "best_head.pt").write_bytes(b"stale head")
    (mixed_run / "last_model.pt").write_bytes(b"model")
    assert detect_eval_mode(mixed_run) == "fullft_model"


def test_fullft_state_dict_extraction_strips_compile_prefixes(tmp_path: Path):
    path = tmp_path / "last_model.pt"
    payload = {
        "model_state_dict": {
            "_orig_mod.dense.weight": torch.ones((5, 3)),
            "dense.bias": torch.zeros(5),
        },
    }

    state = extract_fullft_state_dict(payload, path)

    assert sorted(state) == ["dense.bias", "dense.weight"]

    raw = extract_fullft_state_dict({"_orig_mod.head.weight": torch.ones((1, 1))}, path)
    assert sorted(raw) == ["head.weight"]


def test_ecgfounder_runtime_does_not_import_external_repo_at_module_import_time():
    runtime_tree = ast.parse((REPO / "ecg_adv_gen" / "models" / "ecgfounder_runtime.py").read_text())
    top_imports = [
        node
        for node in runtime_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots: set[str] = set()
    for node in top_imports:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        else:
            imported_roots.add((node.module or "").split(".")[0])

    assert {"net1d", "finetune_model", "physionet2021_dataset"}.isdisjoint(imported_roots)

    init_tree = ast.parse((REPO / "ecg_adv_gen" / "models" / "__init__.py").read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and "ecgfounder_runtime" in (node.module or "")
        for node in ast.walk(init_tree)
    )


def test_ecgfounder_pn2021c_evaluator_uses_package_label_mapping_payload():
    tree = ast.parse(
        (REPO / "ecg_adv_gen" / "runner" / "ecgfounder_pn2021c_eval.py").read_text()
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ecg_adv_gen.labels"
        and any(alias.name == "pn2021_super5_label_mapping_payload" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "pn2021_super5_label_mapping_payload"
        for node in ast.walk(tree)
    )


def test_ecgfounder_pn2021c_evaluator_uses_package_super5_scheme():
    tree = ast.parse(
        (REPO / "ecg_adv_gen" / "runner" / "ecgfounder_pn2021c_eval.py").read_text()
    )
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "scripts.triple_labels.label_schemes" not in modules
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ecg_adv_gen.labels"
        and any(alias.name == "get_super5_scheme" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_ecgfounder_pn2021c_requires_min_target_ref_exclusion_for_run_center():
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    with pytest.raises(ValueError, match="--exclude_ref_ids"):
        evaluator._excluded_ref_ids_for_center(
            {"center": "ningbo", "selected_ref_record_ids": ["r1"]},
            "ningbo",
            500,
        )

    assert evaluator._excluded_ref_ids_for_center(
        {"center": "ningbo", "selected_ref_record_ids": ["r1"]},
        "georgia",
        500,
    ) == set()


def test_ecgfounder_pn2021c_rejects_training_evaluation_exclusion_mismatch():
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    with pytest.raises(ValueError, match="training/evaluation K500 identity mismatch"):
        evaluator._excluded_ref_ids_for_center(
            {"center": "ningbo", "selected_ref_record_ids": ["r1", "r2"]},
            "ningbo",
            2,
            evaluation_ref_ids={"r2", "r3"},
        )


def test_ecgfounder_pn2021c_load_rejects_target_adapted_init_k500_mismatch(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    current_ids = [f"r{i}" for i in range(500)]
    init_ids = [*current_ids[:-1], "other"]
    init_dir = tmp_path / "init"
    current_dir = tmp_path / "current"
    init_dir.mkdir()
    current_dir.mkdir()
    init_checkpoint = init_dir / "last_model.pt"
    init_checkpoint.write_bytes(b"checkpoint")
    (init_dir / "eval_result.json").write_text(
        json.dumps(
            {
                "stage": "k500",
                "center": "ningbo",
                "K": 500,
                "target_train_K": 500,
                "selected_ref_record_ids": init_ids,
                "target_train_record_ids": init_ids,
                "config": {"stage": "k500", "seed": 20260601},
            }
        ),
        encoding="utf-8",
    )
    (current_dir / "eval_result.json").write_text(
        json.dumps(
            {
                "stage": "k500",
                "center": "ningbo",
                "K": 500,
                "target_train_K": 500,
                "selected_ref_record_ids": current_ids,
                "target_train_record_ids": current_ids,
                "config": {"stage": "k500", "seed": 20260531},
                "init_model": {"path": str(init_checkpoint)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-adapted initialization K500 identity mismatch"):
        evaluator._load_result(current_dir)


def test_ecgfounder_pn2021c_load_rejects_target_run_without_init_lineage(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    run_dir = tmp_path / "current"
    run_dir.mkdir()
    (run_dir / "eval_result.json").write_text(
        json.dumps(
            {
                "stage": "k500",
                "center": "ningbo",
                "K": 1,
                "target_train_K": 1,
                "selected_ref_record_ids": ["r1"],
                "target_train_record_ids": ["r1"],
                "config": {"stage": "k500", "seed": 20260531},
                "init_model": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="init_model.path"):
        evaluator._load_result(run_dir)


def test_ecgfounder_pn2021c_load_allows_strict_source_only_without_init(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    run_dir = tmp_path / "source"
    run_dir.mkdir()
    source = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
        "init_model": None,
    }
    (run_dir / "eval_result.json").write_text(json.dumps(source), encoding="utf-8")

    assert evaluator._load_result(run_dir) == source


def test_ecgfounder_pn2021c_load_rejects_incomplete_source_only_contract(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    run_dir = tmp_path / "source"
    run_dir.mkdir()
    source = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
        "init_model": None,
    }
    (run_dir / "eval_result.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError):
        evaluator._load_result(run_dir)


def test_locked_ecgfounder_eval_one_records_package_pn2021c_metadata(monkeypatch):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        evaluator,
        "_load_raw_first_center",
        lambda *args, **kwargs: (
            np.zeros((4, 1000, 12), dtype=np.float32),
            labels,
            np.array(["r0", "r1", "r2", "r3"]),
            {
                "label_mapping": {"version": "v7", "hash": "555ec85d5b51"},
                "preprocess": {
                    "contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1",
                    "preprocess_mode": "minimal_resample",
                    "norm_mode": "per_sample_global",
                    "crop_len": 1000,
                    "target_len": 1000,
                },
            },
            "raw_first",
        ),
    )

    class _Dataset(TensorDataset):
        def __init__(self, signals, labels, indices, *args, **kwargs):
            self.indices = np.asarray(indices, dtype=np.int64)
            super().__init__(
                torch.zeros((len(self.indices), 12, TARGET_POINTS)),
                torch.from_numpy(labels[self.indices]),
            )

    monkeypatch.setattr(evaluator, "ECGFounderBottleneck5000CorruptedDataset", _Dataset)

    def _infer(*args, **kwargs):
        assert kwargs["operator_name"] == "baseline_shift"
        y_true = labels[[0, 3]]
        y_score = np.array(
            [
                [0.9, 0.1, 0.2, 0.1, 0.1],
                [0.1, 0.2, 0.3, 0.8, 0.1],
            ],
            dtype=np.float32,
        )
        return y_true, y_score

    monkeypatch.setattr(evaluator, "infer_ecgfounder_fullft", _infer)
    args = argparse.Namespace(
        scheme="super5",
        corruption_input=evaluator.LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
        limit=None,
        seed=20260501,
        crop_len=1000,
        severity_profile="standard",
        severity_profile_params=None,
        severity_params_file=None,
        severity_params_name=None,
        batch_size=2,
        num_workers=0,
        min_pos=1,
        required_cache_version="v7_refexcluded_100hz1000",
        exclude_ref_ids_by_center={"ningbo": {"r1", "r2"}},
    )

    row = evaluator.eval_one(
        "fullft_model",
        nn.Identity(),
        None,
        {"center": "ningbo", "selected_ref_record_ids": ["r1", "r2"]},
        args,
        torch.device("cpu"),
        "ningbo",
        "baseline_shift",
        5,
        clean_metric_override={"macro_auroc": 1.0, "macro_auprc": 1.0},
        input_already_ecgfounder=True,
    )

    assert row["metadata"]["pn2021c"]["cache_version"] == "v7_refexcluded_100hz1000"
    assert row["metadata"]["pn2021c"]["input_order_id"] == "waveform_bottleneck_then_corrupt_then_zscore"
    assert row["metadata"]["pn2021c"]["ref_record_ids_sha256"] == evaluator.ref_ids_sha256({"r1", "r2"})
    assert row["metadata"]["pn2021c_protocol"]["input_order_id"] == row["metadata"]["pn2021c"]["input_order_id"]
    assert row["metadata_compatibility"]["cache_version"] == "v7_refexcluded_100hz1000"


def test_operator_conditioned_fullft_model_restores_return_features_and_rejects_unknown_op():
    class _Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.return_features = False

        def forward(self, x: torch.Tensor):
            logits = torch.full((x.shape[0], 5), 0.25, dtype=x.dtype, device=x.device)
            features = torch.ones((x.shape[0], 2), dtype=x.dtype, device=x.device)
            return (logits, features) if self.return_features else logits

    adapter = OperatorConditionedLogitAdapter(
        feature_dim=2,
        num_classes=5,
        op_names=["baseline_shift"],
    )
    model = OperatorConditionedFullFTModel(_Base(), adapter)
    x = torch.zeros((3, 12, TARGET_POINTS))

    out = model.forward_with_operator(x, "baseline_shift")

    assert out.shape == (3, 5)
    assert model.base_model.return_features is False
    try:
        model.forward_with_operator(x, "unknown")
    except ValueError as exc:
        assert "unknown operator" in str(exc)
    else:
        raise AssertionError("unknown operator must raise")


def test_fullft_inference_uses_operator_conditioned_forward(monkeypatch):
    class _OperatorAwareModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.operator_names: list[str] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("plain forward should not be used when operator_name is supplied")

        def forward_with_operator(self, x: torch.Tensor, operator_name: str) -> torch.Tensor:
            self.operator_names.append(operator_name)
            return torch.full((x.shape[0], 5), 0.25, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(ecgfounder_inference, "ecg1000_to_ecgfounder_input", lambda x, **kwargs: x)
    loader = DataLoader(
        TensorDataset(torch.zeros((3, 12, 1000)), torch.zeros((3, 5))),
        batch_size=2,
    )
    model = _OperatorAwareModel()

    y_true, y_score = infer_ecgfounder_fullft(
        model,
        loader,
        torch.device("cpu"),
        operator_name="baseline_shift",
    )

    assert y_true.shape == (3, 5)
    assert y_score.shape == (3, 5)
    assert model.operator_names == ["baseline_shift", "baseline_shift"]


def test_fullft_inference_passes_input_stabilizer_kwargs(monkeypatch):
    captured: list[dict[str, object]] = []

    def _convert(x: torch.Tensor, **kwargs):
        captured.append(dict(kwargs))
        return x

    class _PlainModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 5), 0.25, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(ecgfounder_inference, "ecg1000_to_ecgfounder_input", _convert)
    loader = DataLoader(
        TensorDataset(torch.zeros((3, 12, 1000)), torch.zeros((3, 5))),
        batch_size=2,
    )

    infer_ecgfounder_fullft(
        _PlainModel(),
        loader,
        torch.device("cpu"),
        input_stabilizer_kwargs={
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
        },
    )

    assert captured == [
        {
            "target_points": TARGET_POINTS,
            "apply_global_zscore": True,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
        },
        {
            "target_points": TARGET_POINTS,
            "apply_global_zscore": True,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
        },
    ]


def test_bottleneck5000_dataset_resamples_before_corruption(monkeypatch):
    seen_shapes: list[tuple[int, ...]] = []
    seen_sample_rates: list[float | None] = []

    def _apply_sequence(
        ecg_ct,
        corruption,
        public_severity,
        severity_profile,
        *,
        base_seed,
        seed_parts,
        sample_rate_hz=None,
        severity_profile_params=None,
    ):
        seen_shapes.append(tuple(ecg_ct.shape))
        seen_sample_rates.append(sample_rate_hz)
        return ecg_ct + 3.0

    monkeypatch.setattr(waveform_datasets, "apply_corruption_sequence", _apply_sequence)
    signals = torch.linspace(0, 1, steps=1000).view(1, 1000, 1).repeat(1, 1, 12).numpy()
    labels = torch.zeros((1, 5)).numpy()

    ds = ECGFounderBottleneck5000CorruptedDataset(
        signals,
        labels,
        indices=torch.tensor([0]).numpy(),
        corruption="baseline_shift",
        public_severity=5,
        seed=20260501,
        crop_len=1000,
        severity_profile="standard",
    )

    x, y = ds[0]

    assert seen_shapes == [(12, TARGET_POINTS)]
    assert seen_sample_rates == [500.0]
    assert x.shape == (12, TARGET_POINTS)
    assert y.shape == (5,)


def test_fullft_inference_zscores_bottleneck5000_without_reinterpolation(monkeypatch):
    def _convert_should_not_run(*args, **kwargs):
        raise AssertionError("bottleneck5000 inputs must not be passed through ecg1000_to_ecgfounder_input")

    class _ChecksZScored5000Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.shape[-1] == TARGET_POINTS
            flat = x.reshape(x.shape[0], -1)
            assert torch.allclose(flat.mean(dim=1), torch.zeros(x.shape[0]), atol=1e-5)
            assert torch.allclose(flat.std(dim=1), torch.ones(x.shape[0]), atol=1e-5)
            return torch.full((x.shape[0], 5), 0.25, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(ecgfounder_inference, "ecg1000_to_ecgfounder_input", _convert_should_not_run)
    ecg = torch.arange(2 * 12 * TARGET_POINTS, dtype=torch.float32).view(
        2,
        12,
        TARGET_POINTS,
    )
    loader = DataLoader(TensorDataset(ecg, torch.zeros((2, 5))), batch_size=2)

    y_true, y_score = infer_ecgfounder_fullft(
        _ChecksZScored5000Model(),
        loader,
        torch.device("cpu"),
        input_already_ecgfounder=True,
    )

    assert y_true.shape == (2, 5)
    assert y_score.shape == (2, 5)


def test_resume_after_third_committed_unit_reuses_only_finished_work(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    units = [("ningbo", f"op{i}", 5) for i in range(5)]
    interrupted_path = tmp_path / "interrupted.json"
    resumed_calls: list[tuple[str, str, int]] = []

    def evaluate(center, corruption, severity, base_output):
        load_time = base_output["clean_metric_overrides"][center]["metadata"]["load_time_s"]
        return _public_row(corruption, severity, 0.5 + int(corruption[-1]) / 100, load_time)

    def fail_after_third(unit, completed_count):
        del unit
        if completed_count == 3:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluator._run_resume_safe_evaluation(
            output_path=interrupted_path,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=lambda: _base_public_output(1.0),
            evaluate_unit=evaluate,
            after_commit=fail_after_third,
        )

    sidecar, _ = evaluator._resume_paths(interrupted_path)
    snapshot = json.loads(sidecar.read_text(encoding="utf-8"))
    assert not interrupted_path.exists()
    assert [entry["unit_key"] for entry in snapshot["completed_units"]] == [
        evaluator._unit_key(*unit) for unit in units[:3]
    ]

    resumed_base_calls = 0

    def changed_base_factory():
        nonlocal resumed_base_calls
        resumed_base_calls += 1
        return _base_public_output(999.0)

    def recording_evaluate(center, corruption, severity, base_output):
        resumed_calls.append((center, corruption, severity))
        return evaluate(center, corruption, severity, base_output)

    resumed = evaluator._run_resume_safe_evaluation(
        output_path=interrupted_path,
        ordered_units=units,
        run_identity=_resume_identity(),
        base_output_factory=changed_base_factory,
        evaluate_unit=recording_evaluate,
    )
    uninterrupted_path = tmp_path / "uninterrupted.json"
    uninterrupted = evaluator._run_resume_safe_evaluation(
        output_path=uninterrupted_path,
        ordered_units=units,
        run_identity=_resume_identity(),
        base_output_factory=lambda: _base_public_output(1.0),
        evaluate_unit=evaluate,
    )

    assert resumed_base_calls == 0
    assert resumed_calls == units[3:]
    assert resumed == uninterrupted
    assert interrupted_path.read_bytes() == uninterrupted_path.read_bytes()
    assert list(resumed) == PUBLIC_OUTPUT_KEYS
    assert not any("resume" in key or "failure" in key for key in resumed)
    assert {
        row["metadata"]["load_time_s"]
        for corruptions in resumed["per_center"].values()
        for severities in corruptions.values()
        for row in severities.values()
    } == {1.0}


@pytest.mark.parametrize("changed_field", ["clean", "config"])
def test_eval_result_drift_rejects_resume_before_eval_or_write(
    tmp_path: Path,
    changed_field: str,
):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"model")
    checkpoint = tmp_path / "backbone.pt"
    checkpoint.write_bytes(b"backbone")
    eval_result_path = run_dir / "eval_result.json"
    saved = {
        "clean": {"ningbo": {"macro_auroc": 0.9, "macro_auprc": 0.7}},
        "config": {"head_type": "linear", "adapter_hidden": 128},
    }
    eval_result_path.write_text(json.dumps(saved), encoding="utf-8")
    args = argparse.Namespace(
        batch_size=4,
        device="cpu",
        min_pos=1,
        seed=20260501,
        limit=None,
        clean_mmap_cache_dir=str(tmp_path / "mmap"),
        clean_cache_dir=str(tmp_path / "cache"),
        severity_profile_params=None,
        checkpoint=str(checkpoint),
    )
    base = _base_public_output()
    ref_hashes = {"ningbo": "b" * 64}
    first_identity = evaluator._build_run_identity(
        args, "fullft_model", run_dir, base, ref_hashes
    )
    output = tmp_path / f"eval-result-{changed_field}.json"
    units = [("ningbo", "op0", 5), ("ningbo", "op1", 5)]
    with pytest.raises(RuntimeError, match="stop"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=first_identity,
            base_output_factory=lambda: copy.deepcopy(base),
            evaluate_unit=lambda c, k, s, frozen: _public_row(k, s, 0.5, 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop")),
        )

    if changed_field == "clean":
        saved["clean"]["ningbo"]["macro_auroc"] = 0.1
    else:
        saved["config"]["head_type"] = "residual_adapter"
    eval_result_path.write_text(json.dumps(saved), encoding="utf-8")
    resumed_identity = evaluator._build_run_identity(
        args, "fullft_model", run_dir, base, ref_hashes
    )
    sidecar, _ = evaluator._resume_paths(output)
    before = sidecar.read_bytes()
    eval_calls: list[object] = []
    base_calls: list[object] = []

    with pytest.raises(ValueError, match="resume snapshot identity mismatch"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=resumed_identity,
            base_output_factory=lambda: base_calls.append(True),
            evaluate_unit=lambda *call_args: eval_calls.append(call_args),
        )

    assert not eval_calls
    assert not base_calls
    assert sidecar.read_bytes() == before
    assert not output.exists()


def test_completed_unit_uses_compact_run_bound_and_nan_compatible_digests(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    output = tmp_path / "compact.json"
    with pytest.raises(RuntimeError, match="stop"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=[("ningbo", "op0", 5)],
            run_identity=_resume_identity(),
            base_output_factory=_base_public_output,
            evaluate_unit=lambda c, k, s, base: _public_row(k, s, float("nan"), 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop")),
        )

    sidecar, _ = evaluator._resume_paths(output)
    snapshot = json.loads(sidecar.read_text(encoding="utf-8"))
    entry = snapshot["completed_units"][0]
    assert set(entry) == {
        "unit_key", "unit_identity_sha256", "result_sha256", "result"
    }
    assert entry["unit_identity_sha256"] == _test_payload_sha256(
        [snapshot["run_identity_sha256"], "ningbo", "op0", 5]
    )
    assert entry["result_sha256"] == _test_payload_sha256(entry["result"])
    assert "identity" not in entry


@pytest.mark.parametrize(
    "field",
    [
        "center", "corruption", "public_severity", "internal_severity",
        "seed", "ref_hash", "protocol", "input_order",
    ],
)
def test_result_nested_identity_is_cross_checked(field: str):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    identity = _resume_identity()
    row = _public_row("op0", 5, 0.5, 1.0)
    pn2021c = row["metadata"]["pn2021c"]
    if field == "center":
        pn2021c["center"] = "georgia"
    elif field == "corruption":
        pn2021c["corruption"] = "op1"
    elif field == "public_severity":
        pn2021c["public_severity"] = 4
    elif field == "internal_severity":
        pn2021c["internal_severity"] = 8
    elif field == "seed":
        pn2021c["seed"] = 7
    elif field == "ref_hash":
        pn2021c["ref_record_ids_sha256"] = "c" * 64
    elif field == "protocol":
        row["metadata"]["pn2021c_protocol"]["model_family"] = "efficientnet1dv2"
    else:
        pn2021c["input_order_id"] = "different"

    with pytest.raises(ValueError, match="resume unit"):
        evaluator._validate_result(row, ("ningbo", "op0", 5), identity)


@pytest.mark.parametrize("tamper", ["metric", "nested_center"])
def test_completed_result_tamper_rejects_before_eval_or_write(
    tmp_path: Path,
    tamper: str,
):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    output = tmp_path / f"tamper-{tamper}.json"
    units = [("ningbo", "op0", 5), ("ningbo", "op1", 5)]
    with pytest.raises(RuntimeError, match="stop"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=_base_public_output,
            evaluate_unit=lambda c, k, s, base: _public_row(k, s, 0.5, 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop")),
        )
    sidecar, _ = evaluator._resume_paths(output)
    snapshot = json.loads(sidecar.read_text(encoding="utf-8"))
    entry = snapshot["completed_units"][0]
    if tamper == "metric":
        entry["result"]["macro_auroc"] = 0.123
    else:
        entry["result"]["metadata"]["pn2021c"]["center"] = "georgia"
        if "result_sha256" in entry:
            entry["result_sha256"] = _test_payload_sha256(entry["result"])
    sidecar.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    before = sidecar.read_bytes()
    eval_calls: list[object] = []
    base_calls: list[object] = []

    with pytest.raises(ValueError, match=r"resume (snapshot|unit)"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=lambda: base_calls.append(True),
            evaluate_unit=lambda *call_args: eval_calls.append(call_args),
        )

    assert not eval_calls
    assert not base_calls
    assert sidecar.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize(
    "field",
    ["protocol", "checkpoint", "seed", "ref_hash", "mapping", "input_mode", "center", "corruption", "severity"],
)
def test_resume_identity_mismatch_rejects_whole_snapshot_before_eval_or_write(
    tmp_path: Path,
    field: str,
):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    output = tmp_path / f"{field}.json"
    units = [("ningbo", "op0", 5), ("ningbo", "op1", 5)]
    with pytest.raises(RuntimeError, match="stop after one"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=_base_public_output,
            evaluate_unit=lambda c, k, s, base: _public_row(k, s, 0.5, 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop after one")),
        )
    sidecar, _ = evaluator._resume_paths(output)
    snapshot = json.loads(sidecar.read_text(encoding="utf-8"))
    if field in {"center", "corruption", "severity"}:
        entry = snapshot["completed_units"][0]
        changed = {
            "center": ("georgia", "op0", 5),
            "corruption": ("ningbo", "other", 5),
            "severity": ("ningbo", "op0", 4),
        }[field]
        entry["unit_identity_sha256"] = _test_payload_sha256(
            [snapshot["run_identity_sha256"], *changed]
        )
    else:
        identity = snapshot["run_identity"]
        if field == "protocol":
            identity["resolved_evaluator_semantic_config"]["protocol"]["name"] = "different"
        elif field == "checkpoint":
            identity["scored_checkpoint"]["sha256"] = "c" * 64
        elif field == "seed":
            identity["seed"] = 7
        elif field == "ref_hash":
            identity["ref_record_ids_sha256_by_center"]["ningbo"] = "d" * 64
        elif field == "mapping":
            identity["label_mapping"]["pn2021_super5"]["mapping_hash"] = "changed"
        else:
            identity["resolved_evaluator_semantic_config"]["corruption_input"] = "raw_first"
        snapshot["run_identity_sha256"] = _test_payload_sha256(identity)
    sidecar.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    before = sidecar.read_bytes()
    eval_calls: list[object] = []
    base_calls: list[object] = []

    with pytest.raises(ValueError, match="resume snapshot"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=lambda: base_calls.append(True),
            evaluate_unit=lambda *args: eval_calls.append(args),
        )

    assert not eval_calls
    assert not base_calls
    assert sidecar.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["truncated", "duplicate", "unexpected"])
def test_invalid_resume_snapshot_is_rejected_without_salvage(tmp_path: Path, corruption: str):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    output = tmp_path / f"{corruption}.json"
    units = [("ningbo", "op0", 5), ("ningbo", "op1", 5)]
    with pytest.raises(RuntimeError):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=_base_public_output,
            evaluate_unit=lambda c, k, s, base: _public_row(k, s, 0.5, 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop")),
        )
    sidecar, _ = evaluator._resume_paths(output)
    if corruption == "truncated":
        sidecar.write_text('{"resume_schema_version": 1', encoding="utf-8")
    else:
        snapshot = json.loads(sidecar.read_text(encoding="utf-8"))
        entry = copy.deepcopy(snapshot["completed_units"][0])
        if corruption == "unexpected":
            entry["unit_key"] = evaluator._unit_key("georgia", "op0", 5)
        snapshot["completed_units"].append(entry)
        sidecar.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    before = sidecar.read_bytes()
    called: list[object] = []

    with pytest.raises(ValueError, match="resume snapshot"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=units,
            run_identity=_resume_identity(),
            base_output_factory=lambda: called.append("base"),
            evaluate_unit=lambda *args: called.append("eval"),
        )

    assert not called
    assert sidecar.read_bytes() == before
    assert not output.exists()


def test_identity_json_rejects_nonfinite_values_before_progress_write(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    identity = _resume_identity()
    identity["seed"] = float("nan")
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="finite JSON"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=[("ningbo", "op0", 5)],
            run_identity=identity,
            base_output_factory=lambda: pytest.fail("base output must not be built"),
            evaluate_unit=lambda *args: pytest.fail("unit must not be evaluated"),
        )
    sidecar, _ = evaluator._resume_paths(output)
    assert not sidecar.exists()
    assert not output.exists()


def test_resume_identity_normalizes_json_key_and_sequence_types(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    identity = _resume_identity()
    identity["resolved_evaluator_semantic_config"]["custom_profile"] = {5: ("x", 0.5)}
    output = tmp_path / "json-native-identity.json"
    with pytest.raises(RuntimeError, match="stop"):
        evaluator._run_resume_safe_evaluation(
            output_path=output,
            ordered_units=[("ningbo", "op0", 5)],
            run_identity=identity,
            base_output_factory=_base_public_output,
            evaluate_unit=lambda c, k, s, base: _public_row(k, s, 0.5, 1.0),
            after_commit=lambda unit, count: (_ for _ in ()).throw(RuntimeError("stop")),
        )

    evaluator._run_resume_safe_evaluation(
        output_path=output,
        ordered_units=[("ningbo", "op0", 5)],
        run_identity=identity,
        base_output_factory=lambda: pytest.fail("resume must reuse the frozen base"),
        evaluate_unit=lambda *args: pytest.fail("completed unit must not rerun"),
    )
    assert output.exists()


def test_atomic_json_write_fsyncs_file_and_directory_and_cleans_owned_temp(
    tmp_path: Path,
    monkeypatch,
):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    destination = tmp_path / "result.json"
    real_fsync = os.fsync
    synced_types: list[int] = []

    def recording_fsync(fd):
        synced_types.append(stat.S_IFMT(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(evaluator.os, "fsync", recording_fsync)
    evaluator._atomic_write_json(destination, {"metric": float("nan")})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IFREG in synced_types
    assert stat.S_IFDIR in synced_types
    assert b"NaN" in destination.read_bytes()
    original = destination.read_bytes()

    monkeypatch.setattr(evaluator.os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        evaluator._atomic_write_json(destination, {"metric": 0.5})
    assert destination.read_bytes() == original
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_output_lock_is_nonblocking(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    output = tmp_path / "locked.json"
    with evaluator._output_lock(output):
        with pytest.raises(RuntimeError, match="already locked"):
            evaluator._run_resume_safe_evaluation(
                output_path=output,
                ordered_units=[("ningbo", "op0", 5)],
                run_identity=_resume_identity(),
                base_output_factory=_base_public_output,
                evaluate_unit=lambda c, k, s, base: _public_row(k, s, 0.5, 1.0),
            )


def test_checkpoint_identity_hashes_actual_scored_artifacts(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"backbone")
    fullft_run = tmp_path / "fullft"
    fullft_run.mkdir()
    (fullft_run / "best_model.pt").write_bytes(b"best")
    (fullft_run / "last_model.pt").write_bytes(b"last")
    fullft = evaluator._checkpoint_identity("fullft_model", fullft_run, backbone)
    assert fullft["scored_checkpoint"]["path"] == str(fullft_run / "last_model.pt")
    assert fullft["scored_checkpoint"]["sha256"] == hashlib.sha256(b"last").hexdigest()
    assert set(fullft["scored_checkpoint"]["stat"]) == {
        "st_dev", "st_ino", "st_size", "st_mtime_ns"
    }

    feature_run = tmp_path / "feature"
    feature_run.mkdir()
    (feature_run / "best_head.pt").write_bytes(b"head")
    feature = evaluator._checkpoint_identity("feature_head", feature_run, backbone)
    assert feature["scored_head"]["sha256"] == hashlib.sha256(b"head").hexdigest()
    assert feature["scored_backbone"]["sha256"] == hashlib.sha256(b"backbone").hexdigest()


@pytest.mark.parametrize("mutation", ["last_appears", "best_replaced"])
def test_fullft_runtime_guard_rejects_path_or_content_drift_before_unit_eval(
    tmp_path: Path,
    mutation: str,
):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    guard = getattr(evaluator, "_build_runtime_with_checkpoint_guard", None)
    assert guard is not None, "runtime checkpoint guard is missing"
    run_dir = tmp_path / mutation
    run_dir.mkdir()
    best = run_dir / "best_model.pt"
    best.write_bytes(b"best-v1")
    backbone = tmp_path / f"{mutation}-backbone.pt"
    backbone.write_bytes(b"backbone")
    expected = evaluator._checkpoint_identity("fullft_model", run_dir, backbone)

    def construct_runtime():
        if mutation == "last_appears":
            (run_dir / "last_model.pt").write_bytes(b"last")
        else:
            best.write_bytes(b"best-v2-is-different")
        return object()

    unit_evals: list[object] = []

    def construct_then_evaluate():
        runtime = guard(
            expected,
            eval_mode="fullft_model",
            run_dir=run_dir,
            checkpoint=backbone,
            factory=construct_runtime,
        )
        unit_evals.append(runtime)

    with pytest.raises(RuntimeError, match="checkpoint identity changed"):
        construct_then_evaluate()
    assert not unit_evals


def test_feature_runtime_guard_rejects_backbone_replacement_before_unit_eval(tmp_path: Path):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    guard = getattr(evaluator, "_build_runtime_with_checkpoint_guard", None)
    assert guard is not None, "runtime checkpoint guard is missing"
    run_dir = tmp_path / "feature"
    run_dir.mkdir()
    (run_dir / "best_head.pt").write_bytes(b"head")
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"backbone-v1")
    expected = evaluator._checkpoint_identity("feature_head", run_dir, backbone)
    unit_evals: list[object] = []

    def construct_runtime():
        backbone.write_bytes(b"backbone-v2-is-different")
        return object()

    with pytest.raises(RuntimeError, match="checkpoint identity changed"):
        runtime = guard(
            expected,
            eval_mode="feature_head",
            run_dir=run_dir,
            checkpoint=backbone,
            factory=construct_runtime,
        )
        unit_evals.append(runtime)
    assert not unit_evals


def test_bottleneck5000_workers_preserve_order_labels_seeds_metadata_and_output(monkeypatch):
    import ecg_adv_gen.runner.ecgfounder_pn2021c_eval as evaluator

    monkeypatch.setattr(waveform_datasets, "apply_corruption_sequence", _seed_encoding_apply_sequence)
    signals = np.arange(6 * 1000 * 12, dtype=np.float32).reshape(6, 1000, 12)
    labels = np.eye(5, dtype=np.float32)[[0, 1, 2, 3, 4, 0]]
    labels[-1, 1] = 1.0
    indices = np.asarray([4, 1, 5, 0], dtype=np.int64)
    seed = 20260501

    def collect(num_workers):
        dataset = ECGFounderBottleneck5000CorruptedDataset(
            signals, labels, indices, "baseline_shift", 5,
            seed=seed, crop_len=1000, severity_profile="standard",
        )
        batches = list(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers))
        return torch.cat([batch[0] for batch in batches]), torch.cat([batch[1] for batch in batches])

    wave0, label0 = collect(0)
    wave2, label2 = collect(2)
    assert torch.equal(wave0, wave2)
    assert torch.equal(label0, label2)
    assert torch.equal(label0, torch.from_numpy(labels[indices]))
    expected = []
    for real_idx in indices:
        item_seed = stable_seed(seed, "bottleneck5000", "baseline_shift", 5, int(real_idx))
        expected.append([item_seed & 0xFFFF, item_seed >> 16, int(real_idx)])
    assert wave0[:, 0, :3].tolist() == expected

    live_metadata = {
        "load_time_s": 99.0,
        "label_mapping": {"version": "v7", "hash": "555ec85d5b51"},
        "preprocess": {
            "contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1",
            "preprocess_mode": "minimal_resample", "norm_mode": "per_sample_global",
            "crop_len": 1000, "target_len": 1000,
        },
    }
    monkeypatch.setattr(
        evaluator,
        "_load_raw_first_center",
        lambda *args, **kwargs: (
            signals, labels, np.asarray([f"r{i}" for i in range(len(labels))]), live_metadata, "raw_first"
        ),
    )

    def deterministic_infer(model, loader, device, **kwargs):
        del model, device, kwargs
        y_true = torch.cat([batch[1] for batch in loader]).numpy()
        return y_true, 0.1 + 0.8 * y_true

    monkeypatch.setattr(evaluator, "infer_ecgfounder_fullft", deterministic_infer)
    args = argparse.Namespace(
        scheme="super5", corruption_input=evaluator.LOCKED_ECGFOUNDER_CORRUPTION_INPUT,
        limit=None, seed=seed, crop_len=1000, severity_profile="standard",
        severity_profile_params=None, severity_params_file=None, severity_params_name=None,
        batch_size=2, num_workers=0, min_pos=1, required_cache_version="v7",
        exclude_ref_ids_by_center={},
    )
    frozen_clean = {
        "metadata": {**live_metadata, "load_time_s": 1.0},
        "macro_auroc": 1.0,
        "macro_auprc": 1.0,
    }

    row0 = evaluator.eval_one(
        "fullft_model", nn.Identity(), None, {"center": "ningbo", "selected_ref_record_ids": []},
        args, torch.device("cpu"), "ningbo", "baseline_shift", 5,
        clean_metric_override=frozen_clean, input_already_ecgfounder=True,
    )
    args.num_workers = 2
    row2 = evaluator.eval_one(
        "fullft_model", nn.Identity(), None, {"center": "ningbo", "selected_ref_record_ids": []},
        args, torch.device("cpu"), "ningbo", "baseline_shift", 5,
        clean_metric_override=frozen_clean, input_already_ecgfounder=True,
    )
    assert row0 == row2
    assert row0["metadata"]["load_time_s"] == 1.0
