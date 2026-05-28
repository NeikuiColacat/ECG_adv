"""Small ECGFounder head modules shared by legacy runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ResidualAdapterHead(nn.Module):
    """Frozen or trainable source linear head plus a zero-init residual adapter."""

    def __init__(
        self,
        base_head: nn.Linear,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 1.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.scale = float(scale)
        in_dim = int(base_head.in_features)
        out_dim = int(base_head.out_features)
        hidden_dim = max(1, int(hidden_dim))
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if freeze_base:
            for param in self.base_head.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_head(x) + self.scale * self.adapter(x)


class FeatureAdapterHead(nn.Module):
    """Frozen or trainable source linear head after a zero-init feature adapter."""

    def __init__(
        self,
        base_head: nn.Linear,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 1.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.scale = float(scale)
        in_dim = int(base_head.in_features)
        hidden_dim = max(1, int(hidden_dim))
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, in_dim),
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if freeze_base:
            for param in self.base_head.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_head(x + self.scale * self.adapter(x))


def unwrap_linear_head(head: nn.Module) -> nn.Linear:
    """Return the linear base head from either a raw or adapter-wrapped head."""
    if isinstance(head, (ResidualAdapterHead, FeatureAdapterHead)):
        return head.base_head
    if isinstance(head, nn.Linear):
        return head
    raise TypeError(f"expected nn.Linear or ECGFounder adapter head, got {type(head).__name__}")


def clone_linear_head(
    head: nn.Module,
    feature_dim: int | None = None,
    device: torch.device | str | None = None,
) -> nn.Linear:
    """Freeze a linear copy of a raw or adapter-wrapped ECGFounder head."""
    base = unwrap_linear_head(head)
    in_dim = int(base.in_features if feature_dim is None else feature_dim)
    out = nn.Linear(in_dim, int(base.out_features))
    out.load_state_dict(base.state_dict())
    if device is not None:
        out = out.to(device)
    out.eval()
    for param in out.parameters():
        param.requires_grad_(False)
    return out


def init_dense_from_head_path(model: nn.Module, head_path: str | Path) -> dict[str, Any]:
    """Initialize ``model.dense`` from a simple linear-head checkpoint.

    ECGFounder full fine-tuning initializes the Super5 dense layer from the
    frozen-feature linear-probe head. The expected checkpoint is a plain
    ``{"weight": Tensor, "bias": Tensor}`` state dict.
    """
    path = Path(head_path)
    head_state = torch.load(path, map_location="cpu")
    if not isinstance(head_state, dict) or "weight" not in head_state or "bias" not in head_state:
        raise RuntimeError(f"{path} is not a simple linear-head state dict with weight/bias")
    if not hasattr(model, "dense"):
        raise RuntimeError("ECGFounder model has no dense attribute for Super5 head initialization")
    dense = getattr(model, "dense")
    if not isinstance(dense, nn.Linear):
        raise RuntimeError(f"model.dense is {type(dense)!r}, expected nn.Linear")
    weight = head_state["weight"].detach().to(dtype=dense.weight.dtype)
    bias = head_state["bias"].detach().to(dtype=dense.bias.dtype)
    if tuple(weight.shape) != tuple(dense.weight.shape) or tuple(bias.shape) != tuple(dense.bias.shape):
        raise RuntimeError(
            f"head shape mismatch: got weight={tuple(weight.shape)} bias={tuple(bias.shape)}, "
            f"expected weight={tuple(dense.weight.shape)} bias={tuple(dense.bias.shape)}"
        )
    with torch.no_grad():
        dense.weight.copy_(weight.to(device=dense.weight.device))
        dense.bias.copy_(bias.to(device=dense.bias.device))
    return {
        "path": str(path),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
    }


__all__ = [
    "FeatureAdapterHead",
    "ResidualAdapterHead",
    "clone_linear_head",
    "init_dense_from_head_path",
    "unwrap_linear_head",
]
