from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

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


REPO = Path(__file__).resolve().parents[2]
TARGET_POINTS = 5000


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


def test_ecgfounder_pn2021c_evaluator_does_not_define_package_owned_runtime_helpers():
    tree = ast.parse(
        (REPO / "ecg_adv_gen" / "runner" / "ecgfounder_pn2021c_eval.py").read_text()
    )
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

    assert "ECGFounderStreamingCorruptedDataset" not in names
    assert "ECGFounderCleanDataset" not in names
    assert "ECGFounderBottleneck5000CleanDataset" not in names
    assert "ECGFounderBottleneck5000CorruptedDataset" not in names
    assert "infer_ecgfounder" not in names
    assert "infer_ecgfounder_fullft" not in names
    assert "_clean_metric_for_center" not in names
    assert "_ref_ids_sha256" not in names
    assert "build_ecgfounder_feature_model" not in names
    assert "_detect_eval_mode" not in names
    assert "_fullft_model_path" not in names
    assert "_infer_feature_dim" not in names
    assert "_build_head" not in names
    assert "_build_fullft_model" not in names
    assert "_forward_logits_features" not in names
    assert "OperatorConditionedFullFTModel" not in names


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
                "selected_ref_record_ids": ["r2", "r3"],
                "config": {"seed": 20260601},
            }
        ),
        encoding="utf-8",
    )
    (current_dir / "eval_result.json").write_text(
        json.dumps(
            {
                "stage": "k500",
                "center": "ningbo",
                "selected_ref_record_ids": ["r1", "r2"],
                "config": {"seed": 20260531},
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
