"""CPU-only shape tests for the repo-owned 500 Hz VAE."""

from __future__ import annotations

import torch

from ecg_adv_gen.vae import build_vae500, vae500_loss


def test_diffusets500_v1_dynamic_shapes_and_loss_are_finite():
    torch.manual_seed(7)
    model = build_vae500("diffusets500_v1_dynamic", base_channels=16, use_attention=False)
    x = torch.randn(2, 5000, 12)
    noise = torch.zeros(2, 4, 625)

    out = model(x, noise=noise)

    assert out.z.shape == (2, 4, 625)
    assert out.mu.shape == (2, 4, 625)
    assert out.log_var.shape == (2, 4, 625)
    assert out.recon.shape == (2, 5000, 12)
    loss = vae500_loss(out.recon, x, out.mu, out.log_var)
    assert torch.isfinite(loss.loss)
    assert torch.isfinite(loss.recon_huber)
    assert torch.isfinite(loss.kl)


def test_diffusets500_decode_is_deterministic_for_fixed_latent():
    model = build_vae500("diffusets500_v1_dynamic", base_channels=16, use_attention=False)
    z = torch.randn(2, 4, 625)

    recon_a = model.decode(z)
    recon_b = model.decode(z)

    assert recon_a.shape == (2, 5000, 12)
    torch.testing.assert_close(recon_a, recon_b)


def test_diffusets500_v2_compact_latent_shape():
    model = build_vae500("diffusets500_v2_compact", base_channels=16, use_attention=False)
    x = torch.randn(2, 5000, 12)
    noise = torch.zeros(2, 8, 313)

    out = model(x, noise=noise)

    assert out.z.shape == (2, 8, 313)
    assert out.recon.shape == (2, 5000, 12)
