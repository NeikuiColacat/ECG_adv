"""Deterministic executor for compiled typed method profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import torch

from core.methods.contracts import (
    NodeContext,
    ValueKind,
    ViewBundle,
    ViewValue,
    value_kind,
)
from core.methods.registry import CompiledMethod


AdapterCallable: TypeAlias = Callable[
    [NodeContext, tuple[ViewValue, ...]],
    ViewValue,
]


@dataclass(frozen=True)
class ExecutionResources:
    """Runner-owned sources, resources, and explicit adapter callables."""

    sources: Mapping[str, ViewValue]
    adapters: Mapping[str, AdapterCallable] = field(default_factory=dict)
    classifier: Any | None = None
    vae_encoder: Any | None = None
    vae_decoder: Any | None = None
    latent_pool: Any | None = None
    replay_buffer: Any | None = None
    base_seed: int = 0
    rng_identity: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sources = dict(self.sources)
        if any(not isinstance(name, str) or not name for name in sources):
            raise ValueError("source names must be non-empty strings")
        if any(
            value_kind(value) not in {
                ValueKind.WAVEFORM,
                ValueKind.LATENT,
                ValueKind.PAIRED,
                ValueKind.CANDIDATES,
            }
            for value in sources.values()
        ):
            raise TypeError("sources contains an unsupported graph value")
        adapters = dict(self.adapters)
        if any(not isinstance(name, str) or not name for name in adapters):
            raise ValueError("adapter names must be non-empty strings")
        if any(not callable(adapter) for adapter in adapters.values()):
            raise TypeError("every adapter must be callable")
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or not 0 <= self.base_seed < 2**32
        ):
            raise ValueError("base_seed must be a uint32 integer")
        identity = tuple(self.rng_identity)
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("rng_identity must contain non-empty strings")
        object.__setattr__(self, "sources", MappingProxyType(sources))
        object.__setattr__(self, "adapters", MappingProxyType(adapters))
        object.__setattr__(self, "rng_identity", identity)

    def validate_requirements(self, method: CompiledMethod) -> None:
        missing = [
            name
            for name in method.requirements.names()
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"method {method.profile_name!r} is missing resources: {missing}"
            )


def _derive_seed(
    base_seed: int,
    comparison_rng_identity: str,
    namespace: str,
    node_id: str,
    stream: str,
    execution_identity: tuple[str, ...],
) -> int:
    """Derive a seed from semantic identity, never topological position."""

    payload = "|".join(
        (
            str(base_seed),
            comparison_rng_identity,
            namespace,
            node_id,
            stream,
            *execution_identity,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class _ExecutionContext:
    def __init__(
        self,
        *,
        method: CompiledMethod,
        node_id: str,
        node_type: str,
        params: Mapping[str, Any],
        resources: ExecutionResources,
        diagnostics: dict[str, Any],
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.params = params
        configured_namespace = params.get("rng_namespace")
        self.rng_namespace = (
            str(configured_namespace)
            if isinstance(configured_namespace, str) and configured_namespace
            else f"{method.rng_namespace}/{node_id}"
        )
        self._method = method
        self._resources = resources
        self._diagnostics = diagnostics
        self._generators: dict[tuple[str, str], torch.Generator] = {}

    def source(self, name: str) -> ViewValue:
        try:
            return self._resources.sources[name]
        except KeyError:
            raise KeyError(
                f"node {self.node_id!r} requested unavailable source {name!r}"
            ) from None

    def call_adapter(
        self,
        name: str,
        inputs: tuple[ViewValue, ...],
    ) -> ViewValue:
        try:
            adapter = self._resources.adapters[name]
        except KeyError:
            raise KeyError(
                f"node {self.node_id!r} requires unregistered adapter {name!r}"
            ) from None
        return adapter(self, inputs)

    def resource(self, name: str) -> Any:
        if name not in {
            "classifier",
            "vae_encoder",
            "vae_decoder",
            "latent_pool",
            "replay_buffer",
        }:
            raise KeyError(f"unsupported execution resource: {name}")
        value = getattr(self._resources, name)
        if value is None:
            raise ValueError(
                f"node {self.node_id!r} requires unavailable resource {name!r}"
            )
        return value

    def record_diagnostic(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("diagnostic name must be non-empty")
        key = f"{self.node_id}/{name}"
        if key in self._diagnostics:
            raise ValueError(f"diagnostic already recorded: {key}")
        self._diagnostics[key] = value

    def torch_generator(
        self,
        stream: str = "default",
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        if not isinstance(stream, str) or not stream:
            raise ValueError("RNG stream must be non-empty")
        resolved_device = torch.device(device)
        key = (stream, str(resolved_device))
        generator = self._generators.get(key)
        if generator is None:
            seed = _derive_seed(
                self._resources.base_seed,
                self._method.comparison_rng_identity,
                self.rng_namespace,
                self.node_id,
                stream,
                self._resources.rng_identity,
            )
            generator = torch.Generator(device=resolved_device)
            generator.manual_seed(seed)
            self._generators[key] = generator
            self._diagnostics[f"{self.node_id}/rng/{stream}/{resolved_device}"] = {
                "seed": seed,
                "namespace": self.rng_namespace,
                "comparison_rng_identity": self._method.comparison_rng_identity,
                "execution_identity": list(self._resources.rng_identity),
            }
        return generator


def execute_method(
    method: CompiledMethod,
    resources: ExecutionResources,
    *,
    required_outputs: Sequence[str] | None = None,
) -> ViewBundle:
    """Execute a compiled graph and return only its declared named outputs.

    Objective computation and declarative replay routes remain trainer-owned.
    This keeps view generation independent from optimizer and exposure policy.
    """

    if not isinstance(method, CompiledMethod):
        raise TypeError("method must be a CompiledMethod")
    if not isinstance(resources, ExecutionResources):
        raise TypeError("resources must be ExecutionResources")
    if not method.executable:
        raise RuntimeError(
            f"method profile {method.profile_name!r} is audit-only: "
            "contracts.executable=false"
        )
    resources.validate_requirements(method)
    selected_outputs = (
        tuple(method.outputs)
        if required_outputs is None
        else tuple(str(value) for value in required_outputs)
    )
    if not selected_outputs or len(set(selected_outputs)) != len(selected_outputs):
        raise ValueError("required_outputs must be non-empty and unique")
    unknown_outputs = sorted(set(selected_outputs) - set(method.outputs))
    if unknown_outputs:
        raise ValueError(f"required_outputs contains unknown outputs: {unknown_outputs}")
    required_nodes = {method.outputs[name] for name in selected_outputs}
    by_id = {node.profile.node_id: node.profile for node in method.nodes}
    pending = list(required_nodes)
    while pending:
        node_id = pending.pop()
        for parent in by_id[node_id].inputs:
            if parent not in required_nodes:
                required_nodes.add(parent)
                pending.append(parent)
    results: dict[str, ViewValue] = {}
    diagnostics: dict[str, Any] = {
        "method/profile_name": method.profile_name,
        "method/profile_sha256": method.profile_sha256,
        "method/rng_namespace": method.rng_namespace,
        "method/comparison_rng_identity": method.comparison_rng_identity,
    }
    for compiled_node in method.nodes:
        node = compiled_node.profile
        if node.node_id not in required_nodes:
            continue
        definition = compiled_node.definition
        inputs = tuple(results[parent] for parent in node.inputs)
        actual_input_kinds = tuple(value_kind(value) for value in inputs)
        definition.validate_input_kinds(node.node_id, actual_input_kinds)
        context = _ExecutionContext(
            method=method,
            node_id=node.node_id,
            node_type=node.node_type,
            params=node.params,
            resources=resources,
            diagnostics=diagnostics,
        )
        value = definition.execute(context, inputs)
        actual_output = value_kind(value)
        if actual_output is not definition.output_kind:
            raise TypeError(
                f"node {node.node_id!r} returned {actual_output.value}, "
                f"expected {definition.output_kind.value}"
            )
        if value.name != node.node_id:
            raise ValueError(
                f"node {node.node_id!r} adapter returned value named {value.name!r}"
            )
        if value.provenance.node_id != node.node_id:
            raise ValueError(
                f"node {node.node_id!r} returned mismatched provenance node_id "
                f"{value.provenance.node_id!r}"
            )
        results[node.node_id] = value

    named = {
        output_name: results[node_id]
        for output_name, node_id in method.outputs.items()
        if output_name in selected_outputs
    }
    for output_name in selected_outputs:
        expected_kind = method.output_kinds[output_name]
        if value_kind(named[output_name]) is not expected_kind:
            raise RuntimeError(f"compiled output kind drifted for {output_name!r}")
    return ViewBundle(
        values=named,
        node_values=results,
        diagnostics=diagnostics,
    )


__all__ = ["AdapterCallable", "ExecutionResources", "execute_method"]
