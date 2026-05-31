"""Runtime helpers for loading repo-owned ECG VAEs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ecg_adv_gen.vae.diffusets_vae500 import build_vae500


class VAE500RuntimeWrapper(nn.Module):
    """Small module wrapper exposing encode/decode for the 500 Hz VAE."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        variant: str | None = None,
        device: str = "cuda",
        strict: bool = True,
    ):
        super().__init__()
        ckpt_path = Path(checkpoint).expanduser().resolve()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        meta = ckpt.get("model_metadata", {}) if isinstance(ckpt, dict) else {}
        model_variant = variant or meta.get("variant") or ckpt_args.get(
            "model_variant",
            "diffusets500_v1_dynamic",
        )
        self.model = build_vae500(
            model_variant,
            base_channels=int(ckpt_args.get("base_channels", 64)),
            use_attention=not bool(ckpt_args.get("no_attention", False)),
        )
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=strict)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.checkpoint = str(ckpt_path)
        self.variant = model_variant
        self.device = torch.device(device)
        self.input_length = int(self.model.input_length)
        self.in_channels = int(self.model.in_channels)
        self.latent_channels = int(self.model.latent_channels)
        self.latent_length = int(self.model.latent_length)
        self.latent_scale = float(self.model.latent_scale)

    def encode(self, x: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        noise = None
        if deterministic:
            noise = torch.zeros(
                (x.shape[0], self.latent_channels, self.latent_length),
                device=x.device,
                dtype=x.dtype,
            )
        z, _, _ = self.model.encode(x, noise=noise)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


__all__ = ["VAE500RuntimeWrapper"]
