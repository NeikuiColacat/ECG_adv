"""CPU-only tests for direct latent-AugMix consistency training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

import scripts.pgd_cross_center.synth_online_at_super5 as stage3
from scripts.pgd_cross_center.synth_online_at_super5 import (
    train_latent_augmix_consistency_epoch,
    train_raw_corruption_consistency_epoch,
)


def test_train_latent_augmix_consistency_epoch_updates_on_generated_views():
    torch.manual_seed(7)
    clean = np.zeros((3, 12, 8), dtype=np.float32)
    views = np.concatenate([clean + 0.1, clean + 0.2], axis=0).astype(np.float32)
    labels = np.zeros((3, 5), dtype=np.float32)
    labels[:, 0] = 1.0

    model = nn.Sequential(nn.Flatten(), nn.Linear(12 * 8, 5))
    optimizer = AdamW(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    before = [p.detach().clone() for p in model.parameters()]

    stats = train_latent_augmix_consistency_epoch(
        model=model,
        clean_signals_ct=clean,
        augmix_signals_ct=views,
        labels_np=labels,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        copies=2,
        consistency_weight=0.5,
        bce_weight=1.0,
        consistency_loss="jsd",
        batch_size=2,
        crop_len=8,
        grad_clip=1.0,
        trainable_params=list(model.parameters()),
    )

    assert stats["enabled"] is True
    assert stats["n_batches"] == 2
    assert stats["n_generated"] == 6
    assert stats["copies"] == 2
    assert stats["consistency_objective"] == "jsd"
    assert stats["loss"] == stats["loss"]
    assert any(not torch.allclose(old, new) for old, new in zip(before, model.parameters()))


def test_train_raw_corruption_consistency_epoch_can_use_raw_augmix_views(monkeypatch):
    torch.manual_seed(11)
    signals = torch.zeros((2, 12, 8), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 0] = 1.0
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)
    calls: list[dict[str, object]] = []

    def fake_raw_augmix_views(clean_np, **kwargs):
        calls.append(dict(kwargs))
        views = np.concatenate([clean_np + 0.1, clean_np + 0.2], axis=0).astype(np.float32)
        stats = {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"powerline_noise": int(views.shape[0])},
            "copies": int(kwargs["copies"]),
            "severity": int(kwargs["severity"]),
            "severity_profile": str(kwargs["severity_profile"]),
            "width": int(kwargs["width"]),
            "depth": int(kwargs["depth"]),
            "alpha": float(kwargs["alpha"]),
            "ops": list(kwargs["ops"]),
            "renorm": bool(kwargs["renorm"]),
            "clip_abs": float(kwargs["clip_abs"]),
        }
        if kwargs.get("input_stabilizer_config"):
            stats["input_stabilizer"] = dict(kwargs["input_stabilizer_config"])
            stats["postprocess"] = "bandpass0.5-35.0+repairflat+renorm"
        return views, stats

    monkeypatch.setattr(stage3, "build_raw_augmix_views", fake_raw_augmix_views, raising=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(12 * 8, 5))
    optimizer = AdamW(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    stats = train_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise", "emg_noise"],
        prob=1.0,
        consistency_weight=0.5,
        bce_weight=1.0,
        consistency_loss="jsd",
        rng=np.random.default_rng(5),
        grad_clip=1.0,
        trainable_params=list(model.parameters()),
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=3,
        augmix_depth=2,
        augmix_alpha=1.0,
        input_stabilizer_config={
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 35.0,
            "repair_flat_leads": True,
            "renorm_after_stabilizer": True,
            "sample_rate_hz": 100.0,
        },
    )

    assert stats["enabled"] is True
    assert stats["view_mode"] == "augmix"
    assert stats["n_batches"] == 1
    assert stats["n_generated"] == 4
    assert calls[0]["width"] == 3
    assert calls[0]["depth"] == 2
    assert calls[0]["severity_profile"] == "calibrated_10to20pp"
    assert calls[0]["input_stabilizer_config"]["bandpass_high_hz"] == 35.0
    assert stats["input_stabilizer"]["bandpass_high_hz"] == 35.0
    assert stats["postprocess"] == "bandpass0.5-35.0+repairflat+renorm"
