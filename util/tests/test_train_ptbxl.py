from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import torch
import yaml

import core.train_PTBXL as ptbxl_module
from core.train_PTBXL import build_ptbxl_dataloaders, train_ptbxl
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PTBXL.yaml"


class _Model(torch.nn.Module):
    def __init__(self, spec) -> None:
        super().__init__()
        self.model_spec = spec
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.zeros(waveform.shape[0], 5) + self.weight


class _Loader:
    def __init__(self, partition: str) -> None:
        self.partition = partition
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def describe(self):
        return {"partition": self.partition}


@pytest.mark.parametrize(
    ("spec", "expected_rate"),
    ((EFFICIENTNET1DV2_SPEC, 100), (ECGFOUNDER_SPEC, 500)),
)
def test_build_ptbxl_dataloaders_uses_model_sampling_rate_and_official_splits(
    monkeypatch, spec, expected_rate
):
    calls = []

    def fake_get_dataloader(**kwargs):
        calls.append(kwargs)
        return _Loader(kwargs["partition"])

    monkeypatch.setattr(ptbxl_module, "get_dataloader", fake_get_dataloader)
    loaders = build_ptbxl_dataloaders(
        _Model(spec),
        config_path=CONFIG,
        dataloader_parameters={"num_workers": 0, "train_batch_size": 7},
    )
    try:
        assert loaders.sampling_rate_hz == expected_rate
        assert [call["partition"] for call in calls] == [
            "train",
            "validation",
            "test",
        ]
        assert all(call["sampling_rate_hz"] == expected_rate for call in calls)
        assert calls[0]["batch_size"] == 7
        assert calls[0]["shuffle"] is True
        assert calls[1]["shuffle"] is False
        assert calls[2]["shuffle"] is False
        assert all(call["prepare_for_model"] is True for call in calls)
        assert all(call["global_zscore"] is True for call in calls)
        assert all(call["output_layout"] == "channel_time" for call in calls)
        assert all(call["persistent_workers"] is False for call in calls)
    finally:
        loaders.close()
    assert all(loader.closed for loader in (loaders.train, loaders.validation, loaders.test))


def test_train_ptbxl_passes_parameters_to_generic_trainer_and_closes_loaders(
    monkeypatch, tmp_path
):
    loaders = [_Loader(name) for name in ("train", "validation", "test")]
    captured = {}
    sentinel = object()

    def fake_get_dataloader(**kwargs):
        return loaders.pop(0)

    original_loaders = []

    def retaining_get_dataloader(**kwargs):
        loader = fake_get_dataloader(**kwargs)
        original_loaders.append(loader)
        return loader

    def fake_train_model(model, train_dataloader, **kwargs):
        captured["model"] = model
        captured["train"] = train_dataloader
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ptbxl_module, "get_dataloader", retaining_get_dataloader)
    monkeypatch.setattr(ptbxl_module, "train_model", fake_train_model)
    model = _Model(EFFICIENTNET1DV2_SPEC)
    result = train_ptbxl(
        model,
        config_path=CONFIG,
        output_dir=tmp_path / "run",
        device="cpu",
        pos_weight=[1, 2, 3, 4, 5],
        training_parameters={"epochs": 3, "learning_rate": 0.0002},
        dataloader_parameters={"num_workers": 0},
    )

    assert result is sentinel
    assert captured["model"] is model
    assert captured["training_parameters"] == {
        "epochs": 3,
        "learning_rate": 0.0002,
    }
    assert captured["device"] == "cpu"
    assert captured["pos_weight"] == [1, 2, 3, 4, 5]
    assert captured["validation_dataloader"].partition == "validation"
    assert captured["test_dataloader"].partition == "test"
    assert all(loader.closed for loader in original_loaders)


def test_ptbxl_adapter_rejects_sampling_rate_conflict(monkeypatch):
    with pytest.raises(ValueError, match="conflicts"):
        build_ptbxl_dataloaders(
            _Model(EFFICIENTNET1DV2_SPEC),
            config_path=CONFIG,
            sampling_rate_hz=500,
        )


def test_fold10_loader_is_not_constructed_when_final_test_is_disabled(
    monkeypatch, tmp_path
):
    config_root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", config_root)
    config_path = config_root / "train" / "PTBXL.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["selection"]["evaluate_test_at_end"] = False
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    calls = []

    def fake_get_dataloader(**kwargs):
        calls.append(kwargs)
        return _Loader(kwargs["partition"])

    monkeypatch.setattr(ptbxl_module, "get_dataloader", fake_get_dataloader)
    loaders = build_ptbxl_dataloaders(
        _Model(EFFICIENTNET1DV2_SPEC),
        config_path=config_path,
        dataloader_parameters={"num_workers": 0},
    )
    try:
        assert [call["partition"] for call in calls] == ["train", "validation"]
        assert loaders.test is None
        assert loaders.describe()["test"] is None
    finally:
        loaders.close()
