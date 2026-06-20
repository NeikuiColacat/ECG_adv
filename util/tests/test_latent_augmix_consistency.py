"""CPU-only tests for direct latent-AugMix consistency training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

import scripts.pgd_cross_center.synth_online_at_super5 as stage3
from scripts.pgd_cross_center.synth_online_at_super5 import (
    select_target_real_augmix_anchors_ct,
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


def test_select_target_real_augmix_anchors_preserves_pick_order_and_raw_scale():
    signals_tc = np.arange(4 * 1000 * 12, dtype=np.float32).reshape(4, 1000, 12)
    picked = np.asarray([2, 0, 3], dtype=np.int64)

    selected_ct = select_target_real_augmix_anchors_ct(
        signals_tc,
        picked_indices=picked,
        expected_count=3,
    )

    assert selected_ct.shape == (3, 12, 1000)
    assert np.allclose(selected_ct[0], signals_tc[2].T)
    assert np.allclose(selected_ct[1], signals_tc[0].T)
    assert np.allclose(selected_ct[2], signals_tc[3].T)
    assert abs(float(selected_ct.mean())) > 1.0


def test_stage3_locked_three_chain_wrapper_forwards_official_profile_and_keeps_adv_uncorrupted():
    clean = np.zeros((2, 12, 8), dtype=np.float32)
    adv = np.full((2, 12, 8), 5.0, dtype=np.float32)
    calls: list[tuple[float, str, int, str]] = []

    def fake_apply_op(sig_t, op_name: str, severity: int):
        calls.append((float(sig_t.mean().item()), op_name, severity, "standard"))
        return sig_t + 1.0

    stage3_apply_orig = stage3._apply_op
    stage3._apply_op = fake_apply_op
    try:
        views, stats = stage3.build_three_chain_vae_lhat_augmix_views(
            clean,
            adv,
            copies=1,
            severity=5,
            severity_profile="standard",
            width=3,
            depth=1,
            alpha=1.0,
            ops=["powerline_noise"],
            rng=np.random.default_rng(23),
            renorm=False,
            clip_abs=0.0,
        )
    finally:
        stage3._apply_op = stage3_apply_orig

    assert views.shape == (2, 12, 8)
    assert stats["topology"] == "locked_three_chain_vae_lhat_augmix"
    assert stats["corruption_chain_count"] == 2
    assert stats["adversarial_chain_corrupted"] is False
    assert len(calls) == 2 * 2
    assert all(mean < 5.0 for mean, *_ in calls)


def test_stage3_locked_three_chain_wrapper_forwards_custom_profile_and_fixed_mix(monkeypatch):
    clean = np.zeros((2, 12, 8), dtype=np.float32)
    adv = np.full((2, 12, 8), 5.0, dtype=np.float32)
    profile = {"emg_noise": {5: {"max_amplitude": 0.6}}}
    calls: list[tuple[str, int, str, object]] = []

    class RecordingOp:
        def __call__(self, sig_t):
            return sig_t + 1.0

    def fake_build_op(op_name, severity, severity_profile, *, severity_profile_params=None, sample_rate_hz=None):
        calls.append((op_name, severity, severity_profile, severity_profile_params))
        return RecordingOp()

    monkeypatch.setattr(stage3, "_build_pn2021c_corruption_op", fake_build_op)

    views, stats = stage3.build_three_chain_vae_lhat_augmix_views(
        clean,
        adv,
        copies=1,
        severity=5,
        severity_profile="custom",
        severity_profile_params=profile,
        width=3,
        depth=1,
        alpha=1.0,
        mixture_mode="fixed",
        mixture_prob=0.75,
        ops=["emg_noise"],
        rng=np.random.default_rng(23),
        renorm=False,
        clip_abs=0.0,
    )

    assert views.shape == (2, 12, 8)
    assert calls
    assert all(call[2] == "custom" for call in calls)
    assert all(call[3] is profile for call in calls)
    assert stats["mixture_mode"] == "fixed"
    assert stats["mixture_prob"] == 0.75
    assert stats["beta_m_mean"] == 0.75


def test_stage3_locked_three_chain_can_cycle_per_operator_with_fixed_chain_weights(monkeypatch):
    clean = np.zeros((1, 12, 8), dtype=np.float32)
    adv = np.full((1, 12, 8), 9.0, dtype=np.float32)
    ops = [
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]
    calls: list[str] = []

    def fake_build_op(op_name, severity, severity_profile, *, severity_profile_params=None, sample_rate_hz=None):
        calls.append(op_name)

        class RecordingOp:
            def __call__(self, sig_t):
                return sig_t + float(ops.index(op_name) + 1)

        return RecordingOp()

    monkeypatch.setattr(stage3, "_build_pn2021c_corruption_op", fake_build_op)

    views, stats = stage3.build_three_chain_vae_lhat_augmix_views(
        clean,
        adv,
        copies=len(ops),
        severity=5,
        severity_profile="custom",
        severity_profile_params={"profiles": {}},
        width=3,
        depth=1,
        alpha=1.0,
        mixture_mode="fixed",
        mixture_prob=1.0,
        ops=ops,
        rng=np.random.default_rng(23),
        renorm=False,
        clip_abs=0.0,
        op_schedule="per_op",
        chain_weights=[0.45, 0.45, 0.10],
    )

    assert views.shape == (len(ops), 12, 8)
    assert stats["op_schedule"] == "per_op"
    assert stats["chain_weight_mode"] == "fixed"
    assert stats["chain_weights"] == [0.45, 0.45, 0.1]
    assert stats["view_ops"] == ops
    assert stats["op_counts"] == {op: 2 for op in ops}
    assert calls == [op for op in ops for _ in range(2)]
    assert np.allclose(views[:, 0, 0], np.asarray([1.8, 2.7, 3.6, 4.5, 5.4], dtype=np.float32))


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
