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


class OperatorConditionedLogitAdapter(nn.Module):
    """Per-operator residual logit adapters for explicit corruption diagnostics."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        op_names: list[str] | tuple[str, ...],
        hidden_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not op_names:
            raise ValueError("op_names must not be empty")
        self.op_names = [str(op) for op in op_names]
        self.op_to_idx = {op: i for i, op in enumerate(self.op_names)}
        self.scale = float(scale)
        feature_dim = int(feature_dim)
        num_classes = int(num_classes)
        hidden_dim = max(1, int(hidden_dim))
        self.adapters = nn.ModuleList()
        for _ in self.op_names:
            adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, num_classes),
            )
            final = adapter[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.adapters.append(adapter)

    def config(self) -> dict[str, Any]:
        first = self.adapters[0]
        assert isinstance(first, nn.Sequential)
        hidden = first[1]
        dropout = first[3]
        assert isinstance(hidden, nn.Linear)
        assert isinstance(dropout, nn.Dropout)
        return {
            "op_names": list(self.op_names),
            "feature_dim": int(hidden.in_features),
            "num_classes": int(self.adapters[0][-1].out_features),  # type: ignore[index, union-attr]
            "hidden_dim": int(hidden.out_features),
            "dropout": float(dropout.p),
            "scale": float(self.scale),
        }

    def op_ids_from_names(
        self,
        op_names: list[str] | tuple[str, ...],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        ids: list[int] = []
        unknown: list[str] = []
        for op in op_names:
            op = str(op)
            if op not in self.op_to_idx:
                unknown.append(op)
            else:
                ids.append(int(self.op_to_idx[op]))
        if unknown:
            raise ValueError(f"unknown operator ids for op-conditioned adapter: {sorted(set(unknown))}")
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, features: torch.Tensor, op_ids: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
        op_ids = op_ids.to(device=features.device, dtype=torch.long)
        if op_ids.ndim != 1 or op_ids.shape[0] != features.shape[0]:
            raise ValueError(
                f"op_ids must be 1D with batch size {features.shape[0]}, got {tuple(op_ids.shape)}"
            )
        out = features.new_zeros((features.shape[0], int(self.adapters[0][-1].out_features)))  # type: ignore[index, union-attr]
        for op_i, adapter in enumerate(self.adapters):
            mask = op_ids == int(op_i)
            if bool(mask.any()):
                out[mask] = adapter(features[mask])
        return self.scale * out


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
    "OperatorConditionedLogitAdapter",
    "ResidualAdapterHead",
    "clone_linear_head",
    "init_dense_from_head_path",
    "unwrap_linear_head",
]
