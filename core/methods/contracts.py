"""Typed values shared by the composable ECG method graph.

The graph deliberately has a small value algebra.  Waveforms are always raw
millivolts in the canonical 100 Hz, time-channel domain; normalization and
model-specific resampling remain outside this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeAlias

import torch


CANONICAL_SAMPLING_RATE_HZ = 100
CANONICAL_POINTS = 1000
CANONICAL_CHANNELS = 12
BASE_VIEW_NAME = "clean_view"
SUPER5_CLASSES = 5
EXPECTED_LATENT_SHAPE = (4, 128)


class ValueKind(str, Enum):
    """Closed set of values that may cross a graph edge."""

    WAVEFORM = "waveform"
    LATENT = "latent"
    PAIRED = "paired"
    CANDIDATES = "candidates"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _validate_name(value: str, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _validate_labels(labels: torch.Tensor, batch_size: int) -> None:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.shape != (batch_size, SUPER5_CLASSES):
        raise ValueError(
            "labels must have shape "
            f"({batch_size},{SUPER5_CLASSES}), got {tuple(labels.shape)}"
        )
    if not labels.is_floating_point():
        raise TypeError("labels must be floating point multi-hot targets")
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must be finite")


def _validate_sample_ids(sample_ids: Sequence[str], batch_size: int) -> tuple[str, ...]:
    values = tuple(sample_ids)
    if len(values) != batch_size:
        raise ValueError(
            f"sample_ids length {len(values)} does not match batch size {batch_size}"
        )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("sample_ids must contain non-empty strings")
    return values


@dataclass(frozen=True)
class Provenance:
    """Batch-level lineage for one named graph value."""

    node_id: str
    operation: str
    parent_names: tuple[str, ...] = ()
    rng_namespace: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.node_id, "provenance.node_id")
        _validate_name(self.operation, "provenance.operation")
        if any(not isinstance(value, str) or not value for value in self.parent_names):
            raise ValueError("provenance.parent_names must contain non-empty strings")
        if self.rng_namespace is not None:
            _validate_name(self.rng_namespace, "provenance.rng_namespace")
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))

    def describe(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "parent_names": list(self.parent_names),
            "rng_namespace": self.rng_namespace,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class WaveformView:
    """Canonical raw-mV ECG batch with shape ``(B,1000,12)``."""

    name: str
    waveform: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    provenance: Provenance
    valid_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sampling_rate_hz: int = CANONICAL_SAMPLING_RATE_HZ
    units: str = "mV"
    layout: str = "time_channel"

    def __post_init__(self) -> None:
        _validate_name(self.name, "waveform view name")
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
        _validate_labels(self.labels, batch_size)
        object.__setattr__(
            self, "sample_ids", _validate_sample_ids(self.sample_ids, batch_size)
        )
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
class LatentView:
    """ECGTwin latent batch aligned to labels and source record identities."""

    name: str
    latent: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...]
    provenance: Provenance
    encoder_identity: str
    latent_scale: float = 0.18215
    valid_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.name, "latent view name")
        _validate_name(self.encoder_identity, "encoder_identity")
        if not isinstance(self.latent, torch.Tensor):
            raise TypeError("latent must be a torch.Tensor")
        if self.latent.ndim != 3 or tuple(self.latent.shape[1:]) != EXPECTED_LATENT_SHAPE:
            raise ValueError(
                "latent must have shape (B,4,128), got "
                f"{tuple(self.latent.shape)}"
            )
        if not self.latent.is_floating_point():
            raise TypeError("latent must be floating point")
        if not isinstance(self.latent_scale, (int, float)) or self.latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        batch_size = int(self.latent.shape[0])
        _validate_labels(self.labels, batch_size)
        object.__setattr__(
            self, "sample_ids", _validate_sample_ids(self.sample_ids, batch_size)
        )
        finite = torch.isfinite(self.latent).flatten(1).all(dim=1)
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
            mask = mask.to(device=self.latent.device) & finite
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def batch_size(self) -> int:
        return int(self.latent.shape[0])


@dataclass(frozen=True)
class PairedView:
    """Strict one-to-one pair of canonical waveform views."""

    name: str
    left: WaveformView
    right: WaveformView
    provenance: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.name, "paired view name")
        if self.left.sample_ids != self.right.sample_ids:
            raise ValueError("paired views must have identical ordered sample_ids")
        if self.left.labels.shape != self.right.labels.shape or not torch.equal(
            self.left.labels.detach().cpu(), self.right.labels.detach().cpu()
        ):
            raise ValueError("paired views must have identical labels")
        if self.left.waveform.shape != self.right.waveform.shape:
            raise ValueError("paired views must have identical waveform shapes")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def batch_size(self) -> int:
        return self.left.batch_size


@dataclass(frozen=True)
class CandidateSet:
    """One latent anchor and an aligned candidate set for every record."""

    name: str
    anchor: LatentView
    candidates: torch.Tensor
    candidate_labels: torch.Tensor
    candidate_sample_ids: tuple[tuple[str, ...], ...]
    provenance: Provenance
    valid_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.name, "candidate set name")
        if not isinstance(self.candidates, torch.Tensor):
            raise TypeError("candidates must be a torch.Tensor")
        if self.candidates.ndim != 4 or tuple(self.candidates.shape[2:]) != EXPECTED_LATENT_SHAPE:
            raise ValueError("candidates must have shape (B,M,4,128)")
        batch_size, count = (int(self.candidates.shape[0]), int(self.candidates.shape[1]))
        if batch_size != self.anchor.batch_size or count <= 0:
            raise ValueError("candidate batch must align to anchor and have M > 0")
        if self.candidate_labels.shape != (batch_size, count, SUPER5_CLASSES):
            raise ValueError("candidate_labels must have shape (B,M,5)")
        if not self.candidate_labels.is_floating_point() or not bool(
            torch.isfinite(self.candidate_labels).all()
        ):
            raise ValueError("candidate_labels must be finite floating point")
        identifiers = tuple(tuple(row) for row in self.candidate_sample_ids)
        if len(identifiers) != batch_size or any(len(row) != count for row in identifiers):
            raise ValueError("candidate_sample_ids must have shape (B,M)")
        if any(
            not isinstance(value, str) or not value
            for row in identifiers
            for value in row
        ):
            raise ValueError("candidate_sample_ids must be non-empty strings")
        object.__setattr__(self, "candidate_sample_ids", identifiers)
        finite = torch.isfinite(self.candidates).flatten(1).all(dim=1)
        mask = self.valid_mask
        if mask is None:
            mask = self.anchor.valid_mask.to(finite.device) & finite
        elif (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (batch_size,)
        ):
            raise ValueError("valid_mask must be a bool tensor with shape (B,)")
        else:
            mask = (
                mask.to(device=self.candidates.device)
                & self.anchor.valid_mask.to(device=self.candidates.device)
                & finite
            )
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def batch_size(self) -> int:
        return self.anchor.batch_size


ViewValue: TypeAlias = WaveformView | LatentView | PairedView | CandidateSet


def value_kind(value: ViewValue) -> ValueKind:
    if isinstance(value, WaveformView):
        return ValueKind.WAVEFORM
    if isinstance(value, LatentView):
        return ValueKind.LATENT
    if isinstance(value, PairedView):
        return ValueKind.PAIRED
    if isinstance(value, CandidateSet):
        return ValueKind.CANDIDATES
    raise TypeError(f"unsupported graph value: {type(value).__name__}")


@dataclass(frozen=True)
class ViewBundle:
    """Immutable named outputs produced by one method execution."""

    values: Mapping[str, ViewValue]
    node_values: Mapping[str, ViewValue] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outputs = dict(self.values)
        if any(not isinstance(name, str) or not name for name in outputs):
            raise ValueError("ViewBundle output names must be non-empty strings")
        if any(not isinstance(value, (WaveformView, LatentView, PairedView, CandidateSet)) for value in outputs.values()):
            raise TypeError("ViewBundle contains an unsupported value")
        node_values = dict(self.node_values)
        if any(not isinstance(name, str) or not name for name in node_values):
            raise ValueError("ViewBundle node names must be non-empty strings")
        if any(
            not isinstance(value, (WaveformView, LatentView, PairedView, CandidateSet))
            for value in node_values.values()
        ):
            raise TypeError("ViewBundle node_values contains an unsupported value")
        object.__setattr__(self, "values", MappingProxyType(outputs))
        object.__setattr__(self, "node_values", MappingProxyType(node_values))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    def require(self, name: str, expected: ValueKind | None = None) -> ViewValue:
        try:
            value = self.values[name]
        except KeyError:
            raise KeyError(f"named output is unavailable: {name}") from None
        if expected is not None and value_kind(value) is not expected:
            raise TypeError(
                f"output {name!r} is {value_kind(value).value}, expected {expected.value}"
            )
        return value


@dataclass(frozen=True)
class MethodRequirements:
    classifier: bool = False
    vae_encoder: bool = False
    vae_decoder: bool = False
    latent_pool: bool = False
    replay_buffer: bool = False

    def union(self, other: "MethodRequirements") -> "MethodRequirements":
        return MethodRequirements(
            classifier=self.classifier or other.classifier,
            vae_encoder=self.vae_encoder or other.vae_encoder,
            vae_decoder=self.vae_decoder or other.vae_decoder,
            latent_pool=self.latent_pool or other.latent_pool,
            replay_buffer=self.replay_buffer or other.replay_buffer,
        )

    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "classifier",
                "vae_encoder",
                "vae_decoder",
                "latent_pool",
                "replay_buffer",
            )
            if bool(getattr(self, name))
        )


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    kind: str
    views: tuple[str, ...]
    weight: float
    mask_policy: str = "valid_intersection"

    def __post_init__(self) -> None:
        _validate_name(self.name, "objective term name")
        if self.kind not in {"bce", "bernoulli_jsd"}:
            raise ValueError("objective kind must be bce or bernoulli_jsd")
        if not self.views or any(not isinstance(value, str) or not value for value in self.views):
            raise ValueError("objective views must be non-empty named outputs")
        expected = 1 if self.kind == "bce" else 2
        if len(self.views) < expected:
            raise ValueError(f"{self.kind} requires at least {expected} view(s)")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or self.weight < 0:
            raise ValueError("objective weight must be finite and non-negative")
        if not torch.isfinite(torch.tensor(float(self.weight))):
            raise ValueError("objective weight must be finite")
        if self.mask_policy not in {"all", "valid_intersection"}:
            raise ValueError("unsupported objective mask_policy")


@dataclass(frozen=True)
class BufferRoute:
    view: str
    stream: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_name(self.view, "buffer route view")
        _validate_name(self.stream, "buffer route stream")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or self.weight < 0:
            raise ValueError("buffer route weight must be non-negative")
        if not torch.isfinite(torch.tensor(float(self.weight))):
            raise ValueError("buffer route weight must be finite")


@dataclass(frozen=True)
class ObjectivePlan:
    terms: tuple[ObjectiveTerm, ...] = ()
    buffer_routes: tuple[BufferRoute, ...] = ()

    def __post_init__(self) -> None:
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("objective term names must be unique")

    def referenced_outputs(self) -> frozenset[str]:
        return frozenset(
            value
            for term in self.terms
            for value in term.views
        ) | frozenset(route.view for route in self.buffer_routes)


class NodeContext(Protocol):
    """Narrow interface available to explicitly registered node callables."""

    node_id: str
    node_type: str
    params: Mapping[str, Any]
    rng_namespace: str

    def source(self, name: str) -> ViewValue:
        ...

    def call_adapter(
        self,
        name: str,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        ...

    def resource(self, name: str) -> Any:
        ...

    def record_diagnostic(self, name: str, value: Any) -> None:
        ...

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        ...


NodeCallable: TypeAlias = Callable[[NodeContext, tuple[ViewValue, ...]], ViewValue]


__all__ = [
    "BASE_VIEW_NAME",
    "BufferRoute",
    "CANONICAL_CHANNELS",
    "CANONICAL_POINTS",
    "CANONICAL_SAMPLING_RATE_HZ",
    "CandidateSet",
    "EXPECTED_LATENT_SHAPE",
    "LatentView",
    "MethodRequirements",
    "NodeCallable",
    "NodeContext",
    "ObjectivePlan",
    "ObjectiveTerm",
    "PairedView",
    "Provenance",
    "SUPER5_CLASSES",
    "ValueKind",
    "ViewBundle",
    "ViewValue",
    "WaveformView",
    "value_kind",
]
