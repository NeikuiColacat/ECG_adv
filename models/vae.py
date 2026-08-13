"""Self-contained ECGTwin VAE definition and strict component builder.

This is a local rewrite of the ECGTwin VAE architecture.  Runtime code imports
only this file and the manual checkpoint utilities; the external ECGTwin tree
is used solely as a user-owned checkpoint location and reference oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.checkpoints import (
    CheckpointIdentity,
    extract_state_dict,
    load_checkpoint_payload,
    sha256_file,
)
from util.config_bundle import require_mapping as _mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAE_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "vae.yaml"
LATENT_SCALE = 0.18215
ECGTWIN_TO_PTBXL_INDICES = (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class VAEConfig:
    checkpoint_path: Path
    input_points: int
    input_channels: int
    latent_channels: int
    latent_points: int
    latent_scale: float
    freeze: bool
    eval_mode: bool
    strict: bool
    expected_encoder_state_keys: int
    expected_decoder_state_keys: int

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.input_points, self.input_channels

    @property
    def latent_shape(self) -> tuple[int, int]:
        return self.latent_channels, self.latent_points


def _resolve_project_path(raw: Any, *, description: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{description} must be a non-empty path")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_vae_config(path: str | Path = DEFAULT_VAE_CONFIG_PATH) -> VAEConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "VAE config")
    if root.get("schema_version") != 1:
        raise ValueError("VAE config schema_version must be 1")
    model = _mapping(root.get("model"), "VAE config.model")
    contract = _mapping(
        root.get("checkpoint_contract"), "VAE config.checkpoint_contract"
    )
    latent_shape = tuple(int(value) for value in model.get("latent_shape", ()))
    if latent_shape != (4, 128):
        raise ValueError("ECGTwin VAE latent_shape must be [4, 128]")
    if model.get("input_layout") != "time_channel":
        raise ValueError("ECGTwin VAE input_layout must be time_channel")
    if model.get("input_units") != "mV":
        raise ValueError("ECGTwin VAE input_units must be mV")
    if float(model.get("latent_scale", -1.0)) != LATENT_SCALE:
        raise ValueError(f"ECGTwin VAE latent_scale must be {LATENT_SCALE}")
    expected_top = tuple(contract.get("expected_top_level_keys", ()))
    if expected_top != ("encoder", "decoder"):
        raise ValueError("VAE checkpoint top-level keys must be encoder, decoder")
    config = VAEConfig(
        checkpoint_path=_resolve_project_path(
            model.get("checkpoint_path"), description="model.checkpoint_path"
        ),
        input_points=int(model.get("input_points", 0)),
        input_channels=int(model.get("input_channels", 0)),
        latent_channels=latent_shape[0],
        latent_points=latent_shape[1],
        latent_scale=float(model["latent_scale"]),
        freeze=bool(model.get("freeze", True)),
        eval_mode=bool(model.get("eval_mode", True)),
        strict=bool(contract.get("strict", True)),
        expected_encoder_state_keys=int(
            contract.get("expected_encoder_state_keys", 0)
        ),
        expected_decoder_state_keys=int(
            contract.get("expected_decoder_state_keys", 0)
        ),
    )
    if config.input_shape != (1024, 12):
        raise ValueError("ECGTwin VAE input shape must be (1024, 12)")
    if not config.freeze or not config.eval_mode or not config.strict:
        raise ValueError("online AT requires a frozen eval-mode VAE with strict loading")
    return config


class SelfAttention(nn.Module):
    """Single-head self-attention with ECGTwin-compatible state names."""

    def __init__(self, n_heads: int, d_embed: int) -> None:
        super().__init__()
        if d_embed % n_heads:
            raise ValueError("d_embed must be divisible by n_heads")
        self.in_proj = nn.Linear(d_embed, 3 * d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed)
        self.n_heads = int(n_heads)
        self.d_head = d_embed // n_heads

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, embed = value.shape
        shape = (batch, length, self.n_heads, self.d_head)
        query, key, val = self.in_proj(value).chunk(3, dim=-1)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        val = val.view(shape).transpose(1, 2)
        weight = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_head)
        output = torch.softmax(weight, dim=-1) @ val
        output = output.transpose(1, 2).reshape(batch, length, embed)
        return self.out_proj(output)


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Kept for exact checkpoint compatibility.  The reference forward does
        # not apply this GroupNorm, so this rewrite intentionally does not either.
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.attention(value.transpose(-1, -2)).transpose(-1, -2)
        return value + residual


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv_1(F.silu(self.groupnorm_1(value)))
        value = self.conv_2(F.silu(self.groupnorm_2(value)))
        return value + self.residual_layer(residual)


class VAEEncoder(nn.Sequential):
    """ECGTwin encoder: raw ``(B,1024,12)`` mV to scaled ``(B,4,128)``."""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv1d(12, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2),
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),
            nn.Conv1d(256, 256, kernel_size=3, stride=2),
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv1d(512, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1),
        )

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError("VAE encoder input must be a rank-3 torch.Tensor")
        if tuple(value.shape[1:]) != (1024, 12):
            raise ValueError(
                f"VAE encoder input must have shape (B,1024,12), got {tuple(value.shape)}"
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("VAE encoder input must be finite floating-point mV")
        encoded = value.transpose(1, 2)
        for module in self:
            if getattr(module, "stride", None) == (2,):
                encoded = F.pad(encoded, (0, 1))
            encoded = module(encoded)
        mean, log_variance = torch.chunk(encoded, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30.0, 20.0)
        return mean * LATENT_SCALE, mean, log_variance


class VAEDecoder(nn.Sequential):
    """ECGTwin decoder: scaled ``(B,4,128)`` to raw ``(B,1024,12)`` mV."""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv1d(4, 4, kernel_size=1),
            nn.Conv1d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv1d(128, 12, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError("VAE decoder latent must be a rank-3 torch.Tensor")
        if tuple(value.shape[1:]) != (4, 128):
            raise ValueError(
                f"VAE decoder latent must have shape (B,4,128), got {tuple(value.shape)}"
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("VAE decoder latent must be finite floating-point")
        decoded = value / LATENT_SCALE
        for module in self:
            decoded = module(decoded)
        return decoded.transpose(1, 2).contiguous()


def prepare_ecgtwin_encoder_input(raw_ptbxl_btc: torch.Tensor) -> torch.Tensor:
    """Bridge raw PTB-XL-order 100 Hz ECGs to VAE encoder input.

    Input is strictly ``(B,1000,12)`` raw mV.  The aVL/aVF positions are
    swapped into ECGTwin order and the 100 Hz bottleneck is linearly expanded
    to the VAE's 1024-point grid with aligned endpoints.
    """

    if not isinstance(raw_ptbxl_btc, torch.Tensor):
        raise TypeError("raw_ptbxl_btc must be a torch.Tensor")
    if raw_ptbxl_btc.ndim != 3 or tuple(raw_ptbxl_btc.shape[1:]) != (1000, 12):
        raise ValueError(
            "VAE bridge input must have shape (B,1000,12) in PTB-XL order"
        )
    if not raw_ptbxl_btc.is_floating_point() or not bool(
        torch.isfinite(raw_ptbxl_btc).all()
    ):
        raise ValueError("VAE bridge input must be finite raw floating-point mV")
    indices = torch.tensor(
        ECGTWIN_TO_PTBXL_INDICES,
        device=raw_ptbxl_btc.device,
        dtype=torch.long,
    )
    ecgtwin_bct = raw_ptbxl_btc.index_select(2, indices).transpose(1, 2)
    return F.interpolate(
        ecgtwin_bct, size=1024, mode="linear", align_corners=True
    ).transpose(1, 2).contiguous()


def decode_to_ptbxl_waveform(
    decoder: nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Decode a scaled latent to canonical raw PTB-XL-order 100 Hz waveform."""

    decoded = decoder(latent)
    if decoded.ndim != 3 or tuple(decoded.shape[1:]) != (1024, 12):
        raise ValueError(
            "VAE decoder must return raw (B,1024,12) ECGTwin-order waveforms"
        )
    indices = torch.tensor(
        ECGTWIN_TO_PTBXL_INDICES, device=decoded.device, dtype=torch.long
    )
    canonical_bct = decoded.index_select(2, indices).transpose(1, 2)
    canonical = F.interpolate(
        canonical_bct, size=1000, mode="linear", align_corners=True
    )
    return canonical.transpose(1, 2).contiguous()


def _component_identity(
    path: Path,
    sha256: str,
    state: Mapping[str, torch.Tensor],
) -> CheckpointIdentity:
    return CheckpointIdentity(
        path=path,
        sha256=sha256,
        state_key_count=len(state),
        missing_keys=(),
        unexpected_keys=(),
    )


def build_ecgtwin_vae(
    *,
    config_path: str | Path = DEFAULT_VAE_CONFIG_PATH,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[VAEEncoder, VAEDecoder]:
    """Return a strictly loaded, frozen ``(encoder, decoder)`` pair."""

    config = load_vae_config(config_path)
    path = config.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
    resolved, payload = load_checkpoint_payload(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("ECGTwin VAE checkpoint must be a mapping")
    if set(payload) != {"encoder", "decoder"}:
        raise ValueError("ECGTwin VAE checkpoint must contain only encoder and decoder")
    encoder_state = extract_state_dict(payload["encoder"])
    decoder_state = extract_state_dict(payload["decoder"])
    if len(encoder_state) != config.expected_encoder_state_keys:
        raise ValueError("ECGTwin VAE encoder state-key count mismatch")
    if len(decoder_state) != config.expected_decoder_state_keys:
        raise ValueError("ECGTwin VAE decoder state-key count mismatch")

    encoder = VAEEncoder()
    decoder = VAEDecoder()
    encoder.load_state_dict(encoder_state, strict=config.strict)
    decoder.load_state_dict(decoder_state, strict=config.strict)
    digest = sha256_file(resolved)
    encoder.checkpoint_identity = _component_identity(
        resolved, digest, encoder_state
    )
    decoder.checkpoint_identity = _component_identity(
        resolved, digest, decoder_state
    )
    encoder.vae_config = config
    decoder.vae_config = config
    if config.freeze:
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
    if config.eval_mode:
        encoder.eval()
        decoder.eval()
    return encoder, decoder


__all__ = [
    "DEFAULT_VAE_CONFIG_PATH",
    "ECGTWIN_TO_PTBXL_INDICES",
    "LATENT_SCALE",
    "VAEConfig",
    "VAEDecoder",
    "VAEEncoder",
    "build_ecgtwin_vae",
    "decode_to_ptbxl_waveform",
    "load_vae_config",
    "prepare_ecgtwin_encoder_input",
]
