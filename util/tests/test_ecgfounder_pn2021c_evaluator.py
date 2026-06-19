from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import scripts.triple_labels.eval_ecgfounder_pn2021_corruptions as evaluator
from scripts.triple_labels.eval_ecgfounder_pn2021_corruptions import (
    ECGFounderBottleneck5000CorruptedDataset,
    _clean_metric_for_center,
    _detect_eval_mode,
)


def test_clean_metric_for_center_reads_fullft_target_excluding_ref():
    result = {
        "center": "cpsc_2018",
        "target_excluding_ref": {
            "macro_auroc": 0.91,
            "macro_auprc": 0.64,
        },
    }

    row = _clean_metric_for_center(result, "cpsc_2018")

    assert row["macro_auroc"] == 0.91
    assert row["macro_auprc"] == 0.64


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

    assert _detect_eval_mode(feature_run) == "feature_head"
    assert _detect_eval_mode(fullft_run) == "fullft_model"
    assert _detect_eval_mode(locked_last_run) == "fullft_model"

    mixed_run = tmp_path / "mixed_fullft_with_stale_head"
    mixed_run.mkdir()
    (mixed_run / "best_head.pt").write_bytes(b"stale head")
    (mixed_run / "last_model.pt").write_bytes(b"model")
    assert _detect_eval_mode(mixed_run) == "fullft_model"


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

    monkeypatch.setattr(evaluator, "ecg1000_to_ecgfounder_input", lambda x, **kwargs: x)
    loader = DataLoader(
        TensorDataset(torch.zeros((3, 12, 1000)), torch.zeros((3, 5))),
        batch_size=2,
    )
    model = _OperatorAwareModel()

    y_true, y_score = evaluator.infer_ecgfounder_fullft(
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

    monkeypatch.setattr(evaluator, "ecg1000_to_ecgfounder_input", _convert)
    loader = DataLoader(
        TensorDataset(torch.zeros((3, 12, 1000)), torch.zeros((3, 5))),
        batch_size=2,
    )

    evaluator.infer_ecgfounder_fullft(
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
            "target_points": evaluator.TARGET_POINTS,
            "apply_global_zscore": True,
                "bandpass_low_hz": 0.5,
                "bandpass_high_hz": 35.0,
                "repair_flat_leads": True,
            },
            {
                "target_points": evaluator.TARGET_POINTS,
                "apply_global_zscore": True,
                "bandpass_low_hz": 0.5,
                "bandpass_high_hz": 35.0,
                "repair_flat_leads": True,
        },
    ]


def test_bottleneck5000_dataset_resamples_before_corruption(monkeypatch):
    seen_shapes: list[tuple[int, ...]] = []
    seen_sample_rates: list[float | None] = []

    class _Op:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            seen_shapes.append(tuple(x.shape))
            return x + 3.0

    def _build_op(corruption, public_severity, severity_profile, *, sample_rate_hz=None, severity_profile_params=None):
        seen_sample_rates.append(sample_rate_hz)
        return _Op()

    monkeypatch.setattr(evaluator, "_build_corruption_op", _build_op)
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

    assert seen_shapes == [(12, evaluator.TARGET_POINTS)]
    assert seen_sample_rates == [500.0]
    assert x.shape == (12, evaluator.TARGET_POINTS)
    assert y.shape == (5,)


def test_fullft_inference_zscores_bottleneck5000_without_reinterpolation(monkeypatch):
    def _convert_should_not_run(*args, **kwargs):
        raise AssertionError("bottleneck5000 inputs must not be passed through ecg1000_to_ecgfounder_input")

    class _ChecksZScored5000Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.shape[-1] == evaluator.TARGET_POINTS
            flat = x.reshape(x.shape[0], -1)
            assert torch.allclose(flat.mean(dim=1), torch.zeros(x.shape[0]), atol=1e-5)
            assert torch.allclose(flat.std(dim=1), torch.ones(x.shape[0]), atol=1e-5)
            return torch.full((x.shape[0], 5), 0.25, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(evaluator, "ecg1000_to_ecgfounder_input", _convert_should_not_run)
    ecg = torch.arange(2 * 12 * evaluator.TARGET_POINTS, dtype=torch.float32).view(
        2,
        12,
        evaluator.TARGET_POINTS,
    )
    loader = DataLoader(TensorDataset(ecg, torch.zeros((2, 5))), batch_size=2)

    y_true, y_score = evaluator.infer_ecgfounder_fullft(
        _ChecksZScored5000Model(),
        loader,
        torch.device("cpu"),
        input_already_ecgfounder=True,
    )

    assert y_true.shape == (2, 5)
    assert y_score.shape == (2, 5)
