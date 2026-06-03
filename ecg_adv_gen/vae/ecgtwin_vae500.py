"""ECGTwin-original VAE architecture adapted to 5000-sample ECG inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecg_adv_gen.vae.diffusets_vae500 import VAEForwardOutput

ECGTWIN_TO_PTBXL_INDICES = (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)
PTBXL_TO_ECGTWIN_INDICES = ECGTWIN_TO_PTBXL_INDICES


def _load_ecgtwin_vae_classes() -> tuple[type[nn.Module], type[nn.Module]]:
    repo_root = Path(__file__).resolve().parents[2]
    ecgtwin_root = repo_root / "model" / "ECGTwin"
    if not ecgtwin_root.exists():
        raise FileNotFoundError(f"ECGTwin repo not found at {ecgtwin_root}")
    if str(ecgtwin_root) not in sys.path:
        sys.path.insert(0, str(ecgtwin_root))
    from module.vae_model import VAE_Decoder, VAE_Encoder  # type: ignore

    return VAE_Encoder, VAE_Decoder


class ECGTwinVAE500(nn.Module):
    """Original ECGTwin VAE module family used with 500Hz/5000-sample ECGs.

    The original ECGTwin VAE is convolutional and has no learned positional
    embeddings, so the same encoder/decoder weights can be reused at a longer
    temporal grid. For a 5000-sample input the three stride-2 stages produce a
    latent length of 625 instead of the original 128.
    """

    def __init__(
        self,
        *,
        input_length: int = 5000,
        in_channels: int = 12,
        latent_scale: float = 0.18215,
        ptbxl_io: bool = True,
    ):
        super().__init__()
        if input_length % 8 != 0:
            raise ValueError("ECGTwinVAE500 requires input_length divisible by 8")
        if in_channels != 12:
            raise ValueError("ECGTwinVAE500 currently supports the original 12-lead ECGTwin layout only")
        VAE_Encoder, VAE_Decoder = _load_ecgtwin_vae_classes()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.input_length = int(input_length)
        self.in_channels = int(in_channels)
        self.latent_channels = 4
        self.latent_length = self.input_length // 8
        self.latent_scale = float(latent_scale)
        self.ptbxl_io = bool(ptbxl_io)
        self.register_buffer(
            "_ptbxl_to_ecgtwin",
            torch.tensor(PTBXL_TO_ECGTWIN_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_ecgtwin_to_ptbxl",
            torch.tensor(ECGTWIN_TO_PTBXL_INDICES, dtype=torch.long),
            persistent=False,
        )

    def load_ecgtwin_checkpoint(self, checkpoint: str | Path, *, strict: bool = True) -> dict[str, Any]:
        ckpt_path = Path(checkpoint).expanduser().resolve()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "encoder" not in ckpt or "decoder" not in ckpt:
            raise ValueError(f"{ckpt_path} is not an ECGTwin VAE checkpoint with encoder/decoder state dicts")
        enc_result = self.encoder.load_state_dict(ckpt["encoder"], strict=strict)
        dec_result = self.decoder.load_state_dict(ckpt["decoder"], strict=strict)
        return {
            "checkpoint": str(ckpt_path),
            "encoder": str(enc_result),
            "decoder": str(dec_result),
        }

    def encode(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected input shape (B, L, C), got {tuple(x.shape)}")
        if x.shape[1] != self.input_length or x.shape[2] != self.in_channels:
            raise ValueError(
                f"expected input shape (B, {self.input_length}, {self.in_channels}), got {tuple(x.shape)}"
            )
        if noise is None:
            noise = torch.randn(
                (x.shape[0], self.latent_channels, self.latent_length),
                device=x.device,
                dtype=x.dtype,
            )
        if noise.shape != (x.shape[0], self.latent_channels, self.latent_length):
            raise ValueError(
                "noise shape "
                f"{tuple(noise.shape)} does not match {(x.shape[0], self.latent_channels, self.latent_length)}"
            )
        if self.ptbxl_io:
            x = x.index_select(dim=2, index=self._ptbxl_to_ecgtwin.to(x.device))
        return self.encoder(x, noise=noise)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"expected latent shape (B, C, L), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_channels or z.shape[2] != self.latent_length:
            raise ValueError(
                f"expected latent shape (B, {self.latent_channels}, {self.latent_length}), got {tuple(z.shape)}"
            )
        out = self.decoder(z)
        if self.ptbxl_io:
            out = out.index_select(dim=2, index=self._ecgtwin_to_ptbxl.to(out.device))
        return out

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> VAEForwardOutput:
        z, mu, log_var = self.encode(x, noise=noise)
        recon = self.decode(z)
        return VAEForwardOutput(recon=recon, z=z, mu=mu, log_var=log_var)


__all__ = ["ECGTwinVAE500"]
