"""Shared input/output contracts for the manually rebuilt ECG models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from util.pn2021_artifact_contract import CLASS_ORDER as _PORTABLE_CLASS_ORDER


CLASS_ORDER = tuple(_PORTABLE_CLASS_ORDER)


@dataclass(frozen=True)
class ModelSpec:
    """Immutable model-facing waveform and classifier contract."""

    name: str
    input_channels: int
    input_points: int
    sampling_rate_hz: int
    num_classes: int = len(CLASS_ORDER)
    class_order: tuple[str, ...] = CLASS_ORDER
    input_layout: str = "channel_time"
    output_type: str = "raw_logits"

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.input_channels, self.input_points

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_shape": list(self.input_shape),
            "input_layout": self.input_layout,
            "sampling_rate_hz": self.sampling_rate_hz,
            "num_classes": self.num_classes,
            "class_order": list(self.class_order),
            "output_type": self.output_type,
        }


EFFICIENTNET1DV2_SPEC = ModelSpec(
    name="efficientnet1dv2",
    input_channels=12,
    input_points=1000,
    sampling_rate_hz=100,
)
ECGFOUNDER_SPEC = ModelSpec(
    name="ecgfounder",
    input_channels=12,
    input_points=5000,
    sampling_rate_hz=500,
)


def validate_model_input(
    value: torch.Tensor,
    spec: ModelSpec,
    *,
    check_finite: bool = True,
) -> torch.Tensor:
    """Validate a channel-first floating ECG batch without changing it."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("model input must be a torch.Tensor")
    if value.ndim != 3 or tuple(value.shape[1:]) != spec.input_shape:
        raise ValueError(
            f"{spec.name} input shape must be (B,{spec.input_channels},"
            f"{spec.input_points}), got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{spec.name} input must use a floating dtype")
    if check_finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{spec.name} input contains NaN or Inf")
    return value


def validate_model_output(
    value: torch.Tensor,
    spec: ModelSpec,
    *,
    batch_size: int,
    check_finite: bool = True,
) -> torch.Tensor:
    """Validate that a model returned five unactivated Super5 logits."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("model output must be a torch.Tensor")
    expected = (int(batch_size), spec.num_classes)
    if value.ndim != 2 or tuple(value.shape) != expected:
        raise ValueError(
            f"{spec.name} must return raw logits with shape {expected}, "
            f"got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{spec.name} raw logits must use a floating dtype")
    if check_finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{spec.name} raw logits contain NaN or Inf")
    return value


__all__ = [
    "CLASS_ORDER",
    "ECGFOUNDER_SPEC",
    "EFFICIENTNET1DV2_SPEC",
    "ModelSpec",
    "validate_model_input",
    "validate_model_output",
]
