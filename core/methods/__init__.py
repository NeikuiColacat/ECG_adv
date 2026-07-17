"""Typed, whitelist-only composition layer for ECG training views."""

from core.methods.contracts import (
    BASE_VIEW_NAME,
    BufferRoute,
    CandidateSet,
    LatentView,
    MethodRequirements,
    ObjectivePlan,
    ObjectiveTerm,
    PairedView,
    Provenance,
    ValueKind,
    ViewBundle,
    ViewValue,
    WaveformView,
    value_kind,
)
from core.methods.executor import (
    AdapterCallable,
    ExecutionResources,
    execute_method,
)
from core.methods.registry import (
    CompiledMethod,
    CompiledNode,
    DEFAULT_REGISTRY,
    MethodProfile,
    NodeDefinition,
    NodeProfile,
    NodeRegistry,
    compile_method_profile,
    load_method_profile,
)
from core.methods.runtime import (
    GeneratedMethodBatch,
    MethodViewRuntime,
    build_method_runtime,
)


__all__ = [
    "AdapterCallable",
    "BASE_VIEW_NAME",
    "BufferRoute",
    "CandidateSet",
    "CompiledMethod",
    "CompiledNode",
    "DEFAULT_REGISTRY",
    "ExecutionResources",
    "GeneratedMethodBatch",
    "LatentView",
    "MethodProfile",
    "MethodRequirements",
    "MethodViewRuntime",
    "NodeDefinition",
    "NodeProfile",
    "NodeRegistry",
    "ObjectivePlan",
    "ObjectiveTerm",
    "PairedView",
    "Provenance",
    "ValueKind",
    "ViewBundle",
    "ViewValue",
    "WaveformView",
    "compile_method_profile",
    "build_method_runtime",
    "execute_method",
    "load_method_profile",
    "value_kind",
]
