from __future__ import annotations

from pathlib import Path

import pytest
import torch

import core.train_PN2021 as pn_module
from core.train_PN2021 import build_pn2021_k500_dataloader


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021.yaml"


def test_k500_adapter_requests_raw_100hz_runtime_data(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    result = build_pn2021_k500_dataloader(
        center="cpsc_2018",
        model_name="ecgfounder",
        shuffle=True,
        config_path=CONFIG,
        training_parameters={"batch_size": 7},
        dataloader_parameters={"num_workers": 0},
    )

    assert result is sentinel
    assert captured["dataset"] == "pn2021"
    assert captured["partition"] == "k500"
    assert captured["logical_center"] == "cpsc_2018"
    assert captured["sampling_rate_hz"] == 100
    assert captured["batch_size"] == 7
    assert captured["prepare_for_model"] is False
    assert captured["sanitize"] is False
    assert captured["global_zscore"] is False
    assert captured["output_layout"] == "time_channel"
    assert captured["shuffle"] is True
    assert captured["persistent_workers"] is False
    assert captured["selection_resident"] is True
    assert captured["selection_resident_pin_memory"] is False


def test_default_online_config_uses_canonical_profile():
    assert pn_module.DEFAULT_ONLINE_CONFIG_PATH == CONFIG


def test_k500_matched_seed_namespace_does_not_contain_method_arm(monkeypatch):
    captured = {}

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    build_pn2021_k500_dataloader(
        center="ningbo",
        model_name="efficientnet1dv2",
        shuffle=True,
        config_path=CONFIG,
        dataloader_parameters={"num_workers": 0},
    )

    namespace = captured["seed_namespace"]
    assert "direct" not in namespace
    assert "lhat" not in namespace
    assert "augmix" not in namespace


def test_canonical_config_uses_single_process_selection_residency(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    result = build_pn2021_k500_dataloader(
        center="ningbo",
        model_name="efficientnet1dv2",
        shuffle=False,
        config_path=CONFIG,
    )

    assert result is sentinel
    assert captured["num_workers"] == 0
    assert captured["persistent_workers"] is False
    assert captured["cache_mode"] == "mmap"
    assert captured["selection_resident"] is True
    assert captured["selection_resident_pin_memory"] is False


def test_k500_adapter_rejects_drop_last(monkeypatch):
    monkeypatch.setattr(
        pn_module,
        "get_dataloader",
        lambda **kwargs: pytest.fail("runtime loader must not be built"),
    )
    with pytest.raises(ValueError, match="complete K500"):
        build_pn2021_k500_dataloader(
            center="ningbo",
            model_name="efficientnet1dv2",
            shuffle=True,
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 0, "drop_last": True},
        )


def test_k500_selection_residency_rejects_worker_processes(monkeypatch):
    monkeypatch.setattr(
        pn_module,
        "get_dataloader",
        lambda **kwargs: pytest.fail("runtime loader must not be built"),
    )
    with pytest.raises(ValueError, match="selection-resident.*num_workers=0"):
        build_pn2021_k500_dataloader(
            center="ningbo",
            model_name="efficientnet1dv2",
            shuffle=True,
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 1},
        )
