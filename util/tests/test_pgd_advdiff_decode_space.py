"""CPU-only tests for ECGTwin latent decode spaces used by VAE-LH AugMix."""

from __future__ import annotations

import torch

from adversarial.pgd_advdiff import PGDAdvDiffGenerator


class _FakeVictim:
    def _decode_latent_differentiable(self, z: torch.Tensor) -> torch.Tensor:
        base = torch.linspace(-0.25, 0.75, 1024, dtype=torch.float32, device=z.device)
        lead_offsets = torch.arange(12, dtype=torch.float32, device=z.device) * 0.01
        return base.view(1, 1024, 1).repeat(z.shape[0], 1, 12) + lead_offsets.view(1, 1, 12)

    def _global_zscore(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + 1e-8)


def test_pgd_generator_can_decode_raw_pre_zscore_for_augmix_corruptions():
    gen = PGDAdvDiffGenerator.__new__(PGDAdvDiffGenerator)
    gen.victim = _FakeVictim()
    z = torch.zeros(2, 4, 128)

    raw = gen._decode_to_ptbxl_1000_raw(z)
    model = gen._decode_to_ptbxl_1000(z)

    assert raw.shape == (2, 12, 1000)
    assert model.shape == (2, 12, 1000)
    assert abs(float(raw.mean())) > 0.01
    assert abs(float(raw.std()) - 1.0) > 0.05
    assert abs(float(model.mean())) < 1e-5
    assert abs(float(model.std()) - 1.0) < 1e-3
