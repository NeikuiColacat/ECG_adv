"""CPU-only ECGFounder VAE-LHAT signal-ordering regressions."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_adv_gen.runner import ecgfounder_fullft
from ecg_adv_gen.training.resume_contract import validate_resume_contract


def test_locked_augmix_corrupts_raw5000_then_zscores_and_batches_match(monkeypatch, tmp_path):
    raw_anchor = torch.linspace(2.0, 4.0, 1000).view(1, 1, -1).repeat(1, 12, 1)
    raw_adv = torch.linspace(-4.0, -2.0, 1000).view(1, 1, -1).repeat(1, 12, 1)
    normalized_adv = torch.linspace(-1.0, 1.0, 1000).view(1, 1, -1).repeat(1, 12, 1)
    events: list[str] = []

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(5))

        def forward(self, x):
            return self.bias.unsqueeze(0).expand(x.shape[0], -1)

    class _Victim:
        num_classes = 5

        def __init__(self, model):
            self.model = model

        def eval(self):
            return self

        def forward_from_latent_to_logits(self, latent):
            return torch.zeros((latent.shape[0], 5), device=latent.device)

        def _ecgtwin_latent_to_ecg1000(self, latent):
            return raw_anchor.to(latent.device).expand(latent.shape[0], -1, -1)

    class _PGD:
        init_logit_gap = 4.0

        def attack_from_latent(self, z, y, candidate_latents):
            return normalized_adv.to(z.device), torch.ones_like(z)

        def _decode_to_ptbxl_1000_raw(self, z):
            raw = raw_adv if float(z.mean()) > 0.5 else raw_anchor
            return raw.to(z.device).expand(z.shape[0], -1, -1)

    class _Index:
        def candidates_for(self, batch_idx, hull_m):
            return np.zeros((len(batch_idx), hull_m, 4, 128), dtype=np.float32)

    def _capture_corruption_inputs(anchor, adv, **kwargs):
        events.append("corrupt_raw5000")
        assert anchor.shape == adv.shape == (1, 12, 5000)
        assert abs(float(anchor.mean())) > 1.0
        assert abs(float(adv.mean())) > 1.0
        assert not np.isclose(float(anchor.std()), 1.0, atol=0.05)
        assert not np.isclose(float(adv.std()), 1.0, atol=0.05)
        mixed = (0.25 * anchor + 0.75 * adv).astype(np.float32)
        return mixed, {"enabled": True, "n_generated": 1, "chain_base_mode": "clean_clean_third"}

    monkeypatch.setattr(ecgfounder_fullft, "sample_anchor_indices", lambda *args: (np.array([0]), {}))
    monkeypatch.setattr(ecgfounder_fullft, "initial_hull_latent", lambda z, *args, **kwargs: z)
    monkeypatch.setattr(ecgfounder_fullft, "fullft_adv_batch_diagnostics", lambda *args: {})
    monkeypatch.setattr(
        ecgfounder_fullft,
        "summarize_fullft_adv_epoch_diagnostics",
        lambda *args, **kwargs: {
            "n_adv": kwargs["n_adv"],
            "delta_mean": 0.0,
            "delta_max": 0.0,
        },
    )
    monkeypatch.setattr(
        ecgfounder_fullft,
        "build_locked_three_chain_latent_augmix_views",
        _capture_corruption_inputs,
    )

    args = Namespace(
        enable_latent_augmix_branch=True,
        latent_augmix_chain_base_mode="clean_clean_third",
        latent_augmix_third_chain_role="vae_lhat_adversarial_waveform",
        latent_augmix_copies=1,
        latent_augmix_severity=5,
        latent_augmix_severity_profile="standard",
        latent_augmix_width=3,
        latent_augmix_depth=-1,
        latent_augmix_alpha=1.0,
        latent_augmix_ops=["baseline_shift"],
        latent_augmix_adv_base_mix=1.0,
        vae_adv_consistency_weight=0.0,
        current_epoch=1,
        seed=7,
        pgd_batch=1,
        hull_m=1,
        hull_weight_mode="optimized",
        hull_lambda=0.05,
        adv_label_mode="multi_hot_hard",
        adv_teacher_mix=0.4,
        adv_soft_target_floor=0.0,
    )
    labels = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    anchor_pool = {"latents": np.zeros((1, 4, 128), dtype=np.float32), "labels": labels}
    model = _Model()
    result = ecgfounder_fullft.build_adv_epoch(
        model,
        _Victim(model),
        _PGD(),
        anchor_pool,
        object(),
        _Index(),
        torch.ones(5),
        args,
        torch.device("cpu"),
        restore_trainable_fn=lambda: None,
    )
    assert result["latent_augmix_stats"]["signal_space"] == "raw_pre_zscore"

    source_path = tmp_path / "source.npy"
    source_labels_path = tmp_path / "source_labels.npy"
    target_path = tmp_path / "target.npz"
    np.save(source_path, raw_anchor.numpy())
    np.save(source_labels_path, labels)
    np.savez(target_path, signals=raw_anchor.numpy(), labels=labels, record_ids=np.array(["r1"]))
    dataset_args = Namespace(
        source_raw1000_signal_cache=str(source_path),
        source_raw1000_label_cache=str(source_labels_path),
        target_raw1000_npz_override=str(target_path),
    )
    source_ds = ecgfounder_fullft.load_source_raw1000_dataset(dataset_args, np.array([0]), labels)
    target_ds = ecgfounder_fullft.load_target_raw_dataset("ningbo", dataset_args, ["r1"])
    assert source_ds[0][0].shape == target_ds[0][0].shape == result["signals"].shape[1:] == (12, 5000)

    class _CaptureModel(_Model):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x):
            events.append("model_after_mix")
            self.inputs.append(x.detach().clone())
            return super().forward(x)

    capture_model = _CaptureModel()
    optimizer = torch.optim.SGD(capture_model.parameters(), lr=0.01)
    ecgfounder_fullft.train_latent_augmix_consistency_epoch(
        capture_model,
        result["augmix_clean_signals"],
        result["augmix_view_signals"],
        result["augmix_clean_labels"],
        optimizer,
        torch.ones(5),
        torch.device("cpu"),
        list(capture_model.parameters()),
        copies=1,
        consistency_weight=0.0,
        bce_weight=1.0,
        consistency_loss="soft_bce",
        batch_size=1,
    )
    assert events[0] == "corrupt_raw5000"
    assert all(tuple(x.shape) == (1, 12, 5000) for x in capture_model.inputs)
    assert all(abs(float(x.mean())) < 1e-5 for x in capture_model.inputs)
    assert all(float(x.std()) == pytest.approx(1.0, abs=1e-4) for x in capture_model.inputs)


def test_ecgfounder_signal_space_contract_rejects_normalized_first_resume(tmp_path):
    locked = ecgfounder_fullft.latent_augmix_signal_space(
        "clean_clean_third",
        "vae_lhat_adversarial_waveform",
    )
    assert locked == "raw_pre_zscore"
    assert ecgfounder_fullft.latent_augmix_signal_space("all_clean", "clean_anchor_control") == "model_zscore"
    with pytest.raises(ValueError, match="latent_augmix_signal_space"):
        validate_resume_contract(
            {"latent_augmix_signal_space": "model_zscore"},
            {"latent_augmix_signal_space": locked},
            allow_drift=False,
        )
    old_result = tmp_path / "eval_result.json"
    old_result.write_text(
        json.dumps({"config": {"latent_augmix_signal_space": "model_zscore"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="latent_augmix_signal_space"):
        ecgfounder_fullft.validate_existing_run_signal_space(old_result, locked)
