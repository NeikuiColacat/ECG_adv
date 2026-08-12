"""Checkpoint-compatible manual rebuild of the project's EfficientNet1DV2."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.checkpoints import CheckpointIdentity, load_model_checkpoint
from models.contracts import EFFICIENTNET1DV2_SPEC, validate_model_input


class CustomNorm(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value)


class SEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduced_dim: int,
        activation_func: nn.Module,
    ) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            activation_func,
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.se(value)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=value.dtype, device=value.device
            )
            random_tensor.floor_()
            return value.div(keep_prob) * random_tensor
        return value


class FusedMBConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = nn.LeakyReLU(0.01)
        self.fused_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            CustomNorm(out_channels),
            activation_func,
        )
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob)
            if self.use_residual
            else nn.Identity()
        )
        self.se = SEBlock(
            out_channels,
            max(1, int(out_channels // 4)),
            activation_func,
        )
        self.dropout = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value if self.use_residual else None
        value = self.fused_conv(value)
        value = self.se(value)
        value = self.dropout(value)
        if identity is not None:
            value = self.stochastic_depth(value) + identity
        return value


class MBConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = nn.LeakyReLU(0.01)
        mid_channels = in_channels * 6
        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            CustomNorm(mid_channels),
            activation_func,
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False,
            ),
            CustomNorm(mid_channels),
            activation_func,
        )
        self.se = SEBlock(
            mid_channels,
            max(1, int(mid_channels // 4)),
            activation_func,
        )
        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            CustomNorm(out_channels),
        )
        self.dropout = nn.Identity()
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob)
            if self.use_residual
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value if self.use_residual else None
        value = self.expand_conv(value)
        value = self.depthwise_conv(value)
        value = self.se(value)
        value = self.project_conv(value)
        value = self.dropout(value)
        if identity is not None:
            value = self.stochastic_depth(value) + identity
        return value


class EfficientNet1DV2(nn.Module):
    """The historical S-V2 architecture with a strict 100 Hz input contract."""

    model_spec = EFFICIENTNET1DV2_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                12, 12, kernel_size=3, stride=1, padding=1,
                bias=False,
            ),
            CustomNorm(12),
            nn.LeakyReLU(0.01),
        )
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.Linear(640, 5),
        )
        self.final_conv = nn.Conv1d(128, 640, kernel_size=1, stride=1, padding=0)
        self.final_norm = CustomNorm(640)
        self.checkpoint_identity: CheckpointIdentity | None = None

    @staticmethod
    def _make_layers() -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 12
        for index, (out_channels, num_blocks) in enumerate(
            zip(
                (12, 24, 32, 64, 80, 128),
                (2, 4, 4, 6, 8, 10),
                strict=True,
            )
        ):
            kernel_size = (3, 3, 5, 3, 5, 3)[index]
            stride = (1, 1, 2, 2, 2, 2)[index]
            for block_index in range(num_blocks):
                block_stride = stride if block_index == 0 else 1
                if index <= 3:
                    block: nn.Module = FusedMBConv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        block_stride,
                        0.304,
                    )
                else:
                    block = MBConv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        block_stride,
                        0.304,
                    )
                layers.append(block)
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone representation before dropout/head."""

        validate_model_input(value, self.model_spec, check_finite=False)
        value = self.initial_conv(value)
        value = self.features(value)
        value = self.final_conv(value)
        value = self.final_norm(value)
        value = self.classifier[0](value)
        return self.classifier[1](value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.forward_features(value)
        value = self.classifier[2](value)
        return self.classifier[3](value)


def build_efficientnet1dv2(
    *,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> EfficientNet1DV2:
    """Build the locked five-class EfficientNet and optionally load a checkpoint."""

    model = EfficientNet1DV2()
    if checkpoint_path is not None:
        model.checkpoint_identity = load_model_checkpoint(
            model,
            checkpoint_path,
            map_location=map_location,
            strict=strict,
        )
    return model


__all__ = [
    "EfficientNet1DV2",
    "build_efficientnet1dv2",
]
