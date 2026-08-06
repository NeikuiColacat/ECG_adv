"""Checkpoint-compatible manual rebuild of the project's EfficientNet1DV2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from models.checkpoints import CheckpointIdentity, load_model_checkpoint
from models.contracts import EFFICIENTNET1DV2_SPEC, validate_model_input


def get_backbone_config(variant: str) -> dict[str, float]:
    configs = {
        "b0_v1": (1.0, 1.0),
        "b1_v1": (1.0, 1.1),
        "b2_v1": (1.1, 1.2),
        "b3_v1": (1.2, 1.4),
        "b4_v1": (1.4, 1.8),
        "b5_v1": (1.6, 2.2),
        "b6_v1": (1.8, 2.6),
        "b7_v1": (2.0, 3.1),
        "b0_v2": (1.0, 1.0),
        "b1_v2": (1.0, 1.1),
        "b2_v2": (1.1, 1.2),
        "b3_v2": (1.2, 1.4),
        "b4_v2": (1.4, 1.8),
        "b5_v2": (1.6, 2.2),
        "b6_v2": (1.8, 2.6),
        "b7_v2": (2.0, 3.1),
        "s_v2": (1.0, 2.0),
        "m_v2": (1.1, 2.1),
        "l_v2": (1.2, 2.2),
    }
    try:
        width, depth = configs[str(variant)]
    except KeyError:
        raise ValueError(f"unsupported EfficientNet1DV2 variant: {variant!r}") from None
    return {"width_coefficient": width, "depth_coefficient": depth}


def get_activation(name: str = "relu") -> nn.Module:
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "swish": nn.SiLU(),
        "mish": nn.Mish(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.01),
    }
    try:
        return activations[str(name)]
    except KeyError:
        raise ValueError(f"unsupported activation: {name!r}") from None


def _find_optimal_num_groups(num_channels: int) -> int:
    target = int(num_channels) // 2
    divisors = [
        value
        for value in range(1, int(num_channels) + 1)
        if int(num_channels) % value == 0
    ]
    return min(divisors, key=lambda value: (abs(value - target), -value))


class CustomNorm(nn.Module):
    def __init__(self, num_features: int, norm_type: str = "batch") -> None:
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=_find_optimal_num_groups(num_features),
                num_channels=num_features,
            )
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=[num_features])
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(num_features)
        else:
            raise ValueError(f"unsupported norm_type: {norm_type!r}")

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
        activation: str = "relu",
        use_se: bool = False,
        se_ratio: int = 4,
        dropout_rate: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)
        self.fused_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            CustomNorm(out_channels, norm_type),
            activation_func,
        )
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob)
            if self.use_residual
            else nn.Identity()
        )
        self.se = (
            SEBlock(
                out_channels,
                max(1, int(out_channels // se_ratio)),
                activation_func,
            )
            if use_se
            else nn.Identity()
        )
        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

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
        expansion: int = 1,
        activation: str = "relu",
        use_se: bool = False,
        se_ratio: int = 4,
        dropout_rate: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)
        mid_channels = in_channels * expansion
        self.expand_conv = (
            nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, 1, bias=False),
                CustomNorm(mid_channels, norm_type),
                activation_func,
            )
            if expansion > 1
            else nn.Identity()
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
            CustomNorm(mid_channels, norm_type),
            activation_func,
        )
        self.se = (
            SEBlock(
                mid_channels,
                max(1, int(mid_channels // se_ratio)),
                activation_func,
            )
            if use_se
            else nn.Identity()
        )
        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            CustomNorm(out_channels, norm_type),
        )
        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
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

    def __init__(
        self,
        variant: str = "s_v2",
        input_channels: int = 12,
        num_classes: int = 5,
        activation: str = "leaky_relu",
        se_ratio: Sequence[int] = (4, 4, 4, 4, 4, 4, 4),
        base_depths: Sequence[int] = (1, 1, 2, 2, 3, 4, 5),
        base_channels: Sequence[int] = (12, 12, 24, 32, 64, 80, 128, 640),
        expansion_factors: Sequence[int] = (1, 6, 6, 6, 6, 6, 6),
        stochastic_depth_prob: float = 0.304,
        dropout_rate: float = 0.0,
        use_se: bool = True,
        kernel_sizes: Sequence[int] = (3, 3, 5, 3, 5, 3, 3, 3),
        strides: Sequence[int] = (1, 1, 2, 2, 2, 2, 2, 2),
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        if int(input_channels) != self.model_spec.input_channels:
            raise ValueError("EfficientNet1DV2 requires exactly 12 input channels")
        if int(num_classes) != self.model_spec.num_classes:
            raise ValueError("EfficientNet1DV2 manual mainline requires five classes")
        config = get_backbone_config(variant)
        width = config["width_coefficient"]
        depth = config["depth_coefficient"]
        channels = [max(1, int(value * width)) for value in base_channels]
        depths = [max(1, math.ceil(value * depth)) for value in base_depths]
        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                input_channels,
                channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
                padding=kernel_sizes[0] // 2,
                bias=False,
            ),
            CustomNorm(channels[0], norm_type),
            get_activation(activation),
        )
        self.features = self._make_layers(
            channels,
            depths,
            kernel_sizes,
            strides,
            expansion_factors,
            se_ratio,
            activation,
            stochastic_depth_prob,
            dropout_rate,
            use_se,
            norm_type,
            variant,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], num_classes),
        )
        self.final_conv = nn.Conv1d(
            channels[-2], channels[-1], kernel_size=1, stride=1, padding=0
        )
        self.final_norm = CustomNorm(channels[-1], norm_type)
        self.checkpoint_identity: CheckpointIdentity | None = None

    @staticmethod
    def _make_layers(
        channels: Sequence[int],
        depths: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        expansion_factors: Sequence[int],
        se_ratio: Sequence[int],
        activation: str,
        stochastic_depth_prob: float,
        dropout_rate: float,
        use_se: bool,
        norm_type: str,
        variant: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = channels[0]
        for index, (out_channels, num_blocks) in enumerate(
            zip(channels[1::], depths[1::])
        ):
            stride = strides[index]
            for block_index in range(num_blocks):
                block_stride = stride if block_index == 0 else 1
                if "v2" in variant and index <= 3:
                    block: nn.Module = FusedMBConv1d(
                        in_channels,
                        out_channels,
                        kernel_sizes[index],
                        block_stride,
                        activation,
                        use_se,
                        se_ratio[index],
                        dropout_rate,
                        stochastic_depth_prob,
                        norm_type,
                    )
                else:
                    block = MBConv1d(
                        in_channels,
                        out_channels,
                        kernel_sizes[index],
                        block_stride,
                        expansion_factors[index],
                        activation,
                        use_se,
                        se_ratio[index],
                        dropout_rate,
                        stochastic_depth_prob,
                        norm_type,
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
    "CustomNorm",
    "EfficientNet1DV2",
    "FusedMBConv1d",
    "MBConv1d",
    "SEBlock",
    "StochasticDepth",
    "build_efficientnet1dv2",
    "get_activation",
    "get_backbone_config",
]
