"""DiffuSETS-inspired 500 Hz ECG VAE.

The public DiffuSETS VAE operates on 10-second ECGs compressed to 1024 time
points. This module keeps the same practical interface for latent-hull online
AT, but owns the implementation inside this repository and accepts native
PTB-XL records500 tensors shaped ``(B, 5000, 12)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock1d(nn.Module):
    """Small 1D residual block matching the DiffuSETS VAE building pattern."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class BottleneckAttention1d(nn.Module):
    """Self-attention over the latent time axis, disabled for compact tests."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return residue + h.transpose(1, 2)


class DownBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock1d(in_channels, out_channels),
            ResidualBlock1d(out_channels, out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, target_length: int):
        super().__init__()
        self.target_length = int(target_length)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            ResidualBlock1d(out_channels, out_channels),
            ResidualBlock1d(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        return self.block(x)


@dataclass(frozen=True)
class VAEForwardOutput:
    """Forward output for the 500 Hz VAE."""

    recon: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


class DiffuSETSVAE500(nn.Module):
    """Native 500 Hz ECG VAE for 10-second 12-lead ECGs.

    Args:
        input_length: ECG time samples. The paper branch uses 5000.
        in_channels: ECG lead count. The paper branch uses 12.
        latent_channels: Number of latent channels; v1 uses 4.
        downsample_stages: v1 uses 3, giving latent length 625.
        base_channels: Width multiplier. Tests use a smaller value; training
            configs can raise it without changing the interface.
        latent_scale: DiffuSETS-style latent scaling constant.
    """

    def __init__(
        self,
        *,
        input_length: int = 5000,
        in_channels: int = 12,
        latent_channels: int = 4,
        downsample_stages: int = 3,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] | None = None,
        latent_scale: float = 0.18215,
        use_attention: bool = True,
    ):
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if downsample_stages <= 0:
            raise ValueError("downsample_stages must be positive")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if channel_mult is None:
            channel_mult = tuple(2**i for i in range(downsample_stages))
        if len(channel_mult) != downsample_stages:
            raise ValueError("channel_mult length must match downsample_stages")

        self.input_length = int(input_length)
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.downsample_stages = int(downsample_stages)
        self.latent_scale = float(latent_scale)
        self.lengths = self._length_schedule(self.input_length, self.downsample_stages)
        self.latent_length = self.lengths[-1]

        widths = [base_channels * mult for mult in channel_mult]
        self.input_conv = nn.Conv1d(in_channels, widths[0], kernel_size=3, padding=1)

        downs: list[nn.Module] = []
        in_ch = widths[0]
        for out_ch in widths:
            downs.append(DownBlock1d(in_ch, out_ch))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(downs)

        bottleneck_channels = widths[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock1d(bottleneck_channels, bottleneck_channels),
            BottleneckAttention1d(bottleneck_channels) if use_attention else nn.Identity(),
            ResidualBlock1d(bottleneck_channels, bottleneck_channels),
        )
        self.to_moments = nn.Conv1d(bottleneck_channels, latent_channels * 2, kernel_size=3, padding=1)

        self.from_latent = nn.Conv1d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)
        rev_widths = list(reversed(widths))
        rev_lengths = list(reversed(self.lengths[:-1]))
        ups: list[nn.Module] = []
        in_ch = bottleneck_channels
        for out_ch, target_length in zip(rev_widths, rev_lengths, strict=True):
            ups.append(UpBlock1d(in_ch, out_ch, target_length))
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(ups)
        self.output_norm = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.output_conv = nn.Conv1d(in_ch, in_channels, kernel_size=3, padding=1)

    @staticmethod
    def _length_schedule(input_length: int, stages: int) -> tuple[int, ...]:
        lengths = [int(input_length)]
        current = int(input_length)
        for _ in range(stages):
            current = ceil(current / 2)
            lengths.append(current)
        return tuple(lengths)

    def encode(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode channels-last ECG into latent sample, mean, and log variance."""
        if x.ndim != 3:
            raise ValueError(f"expected input shape (B, L, C), got {tuple(x.shape)}")
        if x.shape[1] != self.input_length or x.shape[2] != self.in_channels:
            raise ValueError(
                f"expected input shape (B, {self.input_length}, {self.in_channels}), got {tuple(x.shape)}"
            )
        h = self.input_conv(x.transpose(1, 2))
        for block in self.down_blocks:
            h = block(h)
        h = self.bottleneck(h)
        mu, log_var = self.to_moments(h).chunk(2, dim=1)
        log_var = log_var.clamp(-30.0, 20.0)
        if noise is None:
            noise = torch.randn_like(mu)
        if noise.shape != mu.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match latent shape {tuple(mu.shape)}")
        z = (mu + torch.exp(0.5 * log_var) * noise) * self.latent_scale
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor into channels-last ECG."""
        if z.ndim != 3:
            raise ValueError(f"expected latent shape (B, C, L), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_channels or z.shape[2] != self.latent_length:
            raise ValueError(
                f"expected latent shape (B, {self.latent_channels}, {self.latent_length}), got {tuple(z.shape)}"
            )
        h = self.from_latent(z / self.latent_scale)
        for block in self.up_blocks:
            h = block(h)
        out = self.output_conv(F.silu(self.output_norm(h)))
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> VAEForwardOutput:
        z, mu, log_var = self.encode(x, noise=noise)
        recon = self.decode(z)
        return VAEForwardOutput(recon=recon, z=z, mu=mu, log_var=log_var)


def build_vae500(variant: str, **kwargs) -> nn.Module:
    """Build a named 500 Hz VAE variant."""
    if variant == "diffusets500_v1_dynamic":
        defaults = {
            "input_length": 5000,
            "latent_channels": 4,
            "downsample_stages": 3,
            "base_channels": 64,
            "channel_mult": (1, 2, 4),
            "use_attention": True,
        }
    elif variant == "diffusets500_v2_compact":
        defaults = {
            "input_length": 5000,
            "latent_channels": 8,
            "downsample_stages": 4,
            "base_channels": 64,
            "channel_mult": (1, 2, 4, 4),
            "use_attention": True,
        }
    elif variant == "ecgtwin_init_vae500":
        from ecg_adv_gen.vae.ecgtwin_vae500 import ECGTwinVAE500

        allowed: dict[str, Any] = {
            key: value
            for key, value in kwargs.items()
            if key in {"input_length", "in_channels", "latent_scale", "ptbxl_io"}
        }
        return ECGTwinVAE500(**allowed)
    else:
        raise ValueError(f"unknown VAE500 variant: {variant}")
    defaults.update(kwargs)
    return DiffuSETSVAE500(**defaults)


__all__ = ["DiffuSETSVAE500", "VAEForwardOutput", "build_vae500"]
