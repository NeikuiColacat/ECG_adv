"""Official-architecture ECGFounder model with strict Super5 loading rules."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.checkpoints import (
    CheckpointIdentity,
    extract_state_dict,
    load_checkpoint_payload,
    load_model_checkpoint,
    sha256_file,
)
from models.contracts import ECGFOUNDER_SPEC, validate_model_input


class MyConv1dPadSame(nn.Module):
    """Official ECGFounder Conv1d with dynamic SAME padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.groups = int(groups)
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_length = value.shape[-1]
        output_length = (input_length + self.stride - 1) // self.stride
        padding = max(
            0,
            (output_length - 1) * self.stride
            + self.kernel_size
            - input_length,
        )
        left = padding // 2
        right = padding - left
        return self.conv(F.pad(value, (left, right), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        padding = max(0, self.kernel_size - 1)
        left = padding // 2
        right = padding - left
        return self.max_pool(F.pad(value, (left, right), "constant", 0))


class Swish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class BasicBlock(nn.Module):
    """Official ECGFounder residual bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        is_first_block: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.is_first_block = bool(is_first_block)

        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            self.in_channels, self.out_channels, kernel_size=1, stride=1
        )
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            self.out_channels,
            self.out_channels,
            kernel_size=16,
            stride=stride,
            groups=self.out_channels // 16,
        )
        self.bn3 = nn.BatchNorm1d(self.out_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
        )
        reduction = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels // reduction)
        self.se_fc2 = nn.Linear(self.out_channels // reduction, self.out_channels)
        self.se_activation = Swish()
        if stride > 1:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=stride)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = value
        if not self.is_first_block:
            output = self.activation1(output)
        output = self.conv1(output)
        output = self.activation2(output)
        output = self.conv2(output)
        output = self.activation3(output)
        output = self.conv3(output)

        squeeze = output.mean(-1)
        squeeze = self.se_fc1(squeeze)
        squeeze = self.se_activation(squeeze)
        squeeze = self.se_fc2(squeeze)
        squeeze = torch.sigmoid(squeeze)
        output = torch.einsum("abc,ab->abc", output, squeeze)
        if hasattr(self, "max_pool"):
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            left = (self.out_channels - self.in_channels) // 2
            right = self.out_channels - self.in_channels - left
            identity = F.pad(identity, (left, right), "constant", 0)
            identity = identity.transpose(-1, -2)
        return output + identity


class BasicStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        i_stage: int,
        m_blocks: int,
    ) -> None:
        super().__init__()
        self.block_list = nn.ModuleList()
        for block_index in range(m_blocks):
            self.block_list.append(
                BasicBlock(
                    in_channels=in_channels if block_index == 0 else out_channels,
                    out_channels=out_channels,
                    stride=2 if block_index == 0 else 1,
                    is_first_block=i_stage == 0 and block_index == 0,
                )
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            value = block(value)
        return value


class ECGFounderNet1D(nn.Module):
    """Official 12-lead ECGFounder backbone with a five-logit task head."""

    model_spec = ECGFOUNDER_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.first_conv = MyConv1dPadSame(
            in_channels=12,
            out_channels=64,
            kernel_size=16,
            stride=2,
        )
        self.first_bn = nn.BatchNorm1d(64)
        self.first_activation = Swish()
        self.stage_list = nn.ModuleList()
        stage_in_channels = 64
        for stage_index, (out_channels, block_count) in enumerate(
            zip(
                (64, 160, 160, 400, 400, 1024, 1024),
                (2, 2, 2, 3, 3, 4, 4),
            )
        ):
            self.stage_list.append(
                BasicStage(
                    in_channels=stage_in_channels,
                    out_channels=out_channels,
                    i_stage=stage_index,
                    m_blocks=block_count,
                )
            )
            stage_in_channels = out_channels
        self.dense = nn.Linear(stage_in_channels, 5)
        self.pretrained_checkpoint_identity: CheckpointIdentity | None = None
        self.task_checkpoint_identity: CheckpointIdentity | None = None

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        validate_model_input(value, self.model_spec, check_finite=False)
        value = self.first_conv(value)
        value = self.first_activation(value)
        for stage in self.stage_list:
            value = stage(value)
        return value.mean(-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dense(self.forward_features(value))


def _load_official_pretrained_backbone(
    model: ECGFounderNet1D,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device,
) -> CheckpointIdentity:
    resolved, payload = load_checkpoint_payload(
        checkpoint_path,
        map_location=map_location,
        trusted=True,
    )
    full_state = extract_state_dict(payload)
    backbone_state = {
        key: value for key, value in full_state.items() if not key.startswith("dense.")
    }
    incompatible = model.load_state_dict(backbone_state, strict=False)
    expected_missing = {"dense.weight", "dense.bias"}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "official ECGFounder backbone checkpoint mismatch: "
            f"missing={sorted(actual_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return CheckpointIdentity(
        path=resolved,
        sha256=sha256_file(resolved),
        state_key_count=len(backbone_state),
        missing_keys=tuple(sorted(actual_missing)),
        unexpected_keys=(),
    )


def _set_trainable_scope(model: ECGFounderNet1D, scope: str) -> None:
    if scope not in {"full", "head"}:
        raise ValueError("ECGFounder trainable_scope must be 'full' or 'head'")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == "full" or name.startswith("dense."))


def build_ecgfounder(
    *,
    pretrained_checkpoint_path: str | Path | None = None,
    task_checkpoint_path: str | Path | None = None,
    trainable_scope: str = "full",
    map_location: str | torch.device = "cpu",
) -> ECGFounderNet1D:
    """Build the official architecture for Super5 and load requested weights."""

    model = ECGFounderNet1D()
    if pretrained_checkpoint_path is not None:
        model.pretrained_checkpoint_identity = _load_official_pretrained_backbone(
            model,
            pretrained_checkpoint_path,
            map_location=map_location,
        )
    if task_checkpoint_path is not None:
        model.task_checkpoint_identity = load_model_checkpoint(
            model,
            task_checkpoint_path,
            map_location=map_location,
            strict=True,
        )
    _set_trainable_scope(model, str(trainable_scope))
    return model


__all__ = [
    "ECGFounderNet1D",
    "build_ecgfounder",
]
