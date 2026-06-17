from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import scripts.triple_labels.eval_ecgfounder_pn2021_corruptions as evaluator
from scripts.triple_labels.eval_ecgfounder_pn2021_corruptions import (
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

    assert _detect_eval_mode(feature_run) == "feature_head"
    assert _detect_eval_mode(fullft_run) == "fullft_model"


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

    monkeypatch.setattr(evaluator, "ecg1000_to_ecgfounder_input", lambda x, target_points=None: x)
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
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
        },
        {
            "target_points": evaluator.TARGET_POINTS,
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
        },
    ]
