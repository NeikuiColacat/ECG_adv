"""Small immutable contracts shared by the finite ECG recipes.

Every recipe view stays in the canonical raw-mV ``(B,1000,12)`` domain.
Normalization, model-specific resampling, objectives, and optimizer steps remain
trainer-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch


CANONICAL_SAMPLING_RATE_HZ = 100
CANONICAL_POINTS = 1000
CANONICAL_CHANNELS = 12
BASE_VIEW_NAME = "clean_view"
SUPER5_CLASSES = 5
EXPECTED_LATENT_SHAPE = (4, 128)


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _name(value: str, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _labels(labels: torch.Tensor, batch_size: int) -> None:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.shape != (batch_size, SUPER5_CLASSES):
        raise ValueError(
            f"labels must have shape ({batch_size},{SUPER5_CLASSES}), "
            f"got {tuple(labels.shape)}"
        )
    if not labels.is_floating_point():
        raise TypeError("labels must be floating point multi-hot targets")
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must be finite")


def _sample_ids(values: Sequence[str], batch_size: int) -> tuple[str, ...]:
    sample_ids = tuple(values)
    if len(sample_ids) != batch_size:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} does not match batch size "
            f"{batch_size}"
        )
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError("sample_ids must contain non-empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique within a batch")
    return sample_ids


@dataclass(frozen=True)
class WaveformView:
    """Finite canonical raw-mV ECG batch with shape ``(B,1000,12)``."""

    name: str
    waveform: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    valid_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sampling_rate_hz: int = CANONICAL_SAMPLING_RATE_HZ
    units: str = "mV"
    layout: str = "time_channel"

    def __post_init__(self) -> None:
        _name(self.name, "waveform view name")
        if not isinstance(self.waveform, torch.Tensor):
            raise TypeError("waveform must be a torch.Tensor")
        if self.waveform.ndim != 3 or tuple(self.waveform.shape[1:]) != (
            CANONICAL_POINTS,
            CANONICAL_CHANNELS,
        ):
            raise ValueError(
                "waveform must have canonical shape (B,1000,12), got "
                f"{tuple(self.waveform.shape)}"
            )
        if not self.waveform.is_floating_point():
            raise TypeError("waveform must be floating-point raw mV")
        if self.sampling_rate_hz != CANONICAL_SAMPLING_RATE_HZ:
            raise ValueError("WaveformView sampling_rate_hz must be 100")
        if self.units != "mV" or self.layout != "time_channel":
            raise ValueError("WaveformView must use raw mV time_channel layout")
        batch_size = int(self.waveform.shape[0])
        _labels(self.labels, batch_size)
        object.__setattr__(self, "sample_ids", _sample_ids(self.sample_ids, batch_size))
        finite = torch.isfinite(self.waveform).flatten(1).all(dim=1)
        mask = self.valid_mask
        if mask is None:
            mask = finite
        elif (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (batch_size,)
        ):
            raise ValueError("valid_mask must be a bool tensor with shape (B,)")
        else:
            mask = mask.to(device=self.waveform.device) & finite
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def batch_size(self) -> int:
        return int(self.waveform.shape[0])


@dataclass(frozen=True)
class ViewBundle:
    """Immutable named waveform outputs from one finite recipe execution."""

    values: Mapping[str, WaveformView]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.values)
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError("ViewBundle output names must be non-empty strings")
        if any(not isinstance(value, WaveformView) for value in values.values()):
            raise TypeError("ViewBundle may contain only WaveformView values")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    def require(self, name: str) -> WaveformView:
        try:
            return self.values[name]
        except KeyError:
            raise KeyError(f"named output is unavailable: {name}") from None


__all__ = [
    "BASE_VIEW_NAME",
    "CANONICAL_CHANNELS",
    "CANONICAL_POINTS",
    "CANONICAL_SAMPLING_RATE_HZ",
    "EXPECTED_LATENT_SHAPE",
    "SUPER5_CLASSES",
    "ViewBundle",
    "WaveformView",
]
