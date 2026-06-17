"""CPU-only tests for ECGFounder raw-AugMix consistency plumbing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 as ecgfounder_lhat


class _TinyVictim:
    def __init__(self) -> None:
        self.feature_model = nn.Identity()
        self.head = nn.Linear(12 * 1000, 5)

    def features_from_ecg1000(self, ecg_ct_1000: torch.Tensor, grad: bool) -> torch.Tensor:
        return ecg_ct_1000.reshape(ecg_ct_1000.shape[0], -1)


def test_ecgfounder_profiled_raw_augmix_uses_profile_aware_op_callback():
    signals = np.zeros((2, 12, 32), dtype=np.float32)

    views, stats = ecgfounder_lhat.build_profiled_raw_corruption_views(
        signals,
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise"],
        prob=1.0,
        rng=np.random.default_rng(23),
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
    )

    assert views.shape == signals.shape
    assert stats["view_mode"] == "augmix"
    assert stats["severity_profile"] == "calibrated_10to20pp"


def test_ecgfounder_raw_corruption_epoch_can_use_raw_augmix_views(monkeypatch):
    torch.manual_seed(17)
    signals = torch.zeros((2, 12, 8), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 0] = 1.0
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)
    calls: list[dict[str, object]] = []

    def fake_raw_augmix_views(clean_np, **kwargs):
        calls.append(dict(kwargs))
        views = np.concatenate([clean_np + 0.1, clean_np + 0.2], axis=0).astype(np.float32)
        return views, {
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
            "mixture_mode": str(kwargs["mixture_mode"]),
            "mixture_prob": float(kwargs["mixture_prob"]),
            "ops": list(kwargs["ops"]),
            "renorm": bool(kwargs["renorm"]),
            "clip_abs": float(kwargs["clip_abs"]),
        }

    monkeypatch.setattr(ecgfounder_lhat, "build_raw_augmix_views", fake_raw_augmix_views, raising=False)

    victim = _TinyVictim()
    optimizer = AdamW(victim.head.parameters(), lr=1e-2)

    stats = ecgfounder_lhat.train_raw_corruption_feature_consistency_epoch(
        victim=victim,
        loader=loader,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise", "emg_noise"],
        prob=1.0,
        consistency_weight=0.5,
        bce_weight=1.0,
        consistency_loss="jsd",
        rng=np.random.default_rng(17),
        trainable_params=list(victim.head.parameters()),
        grad_clip=1.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
    )

    assert stats["enabled"] is True
    assert stats["view_mode"] == "augmix"
    assert stats["n_batches"] == 1
    assert stats["n_generated"] == 4
    assert calls[0]["width"] == 1
    assert calls[0]["depth"] == 1
    assert calls[0]["mixture_mode"] == "fixed"
    assert calls[0]["mixture_prob"] == 1.0
    assert calls[0]["severity_profile"] == "calibrated_10to20pp"
