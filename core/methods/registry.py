"""Explicit node whitelist and compiler for tracked ECG method profiles.

The accepted YAML shape is the single project schema under
``configs/train/methods``.  A profile may reference only code-owned symbolic
node types from :data:`DEFAULT_REGISTRY`; import paths and arbitrary callables
are forbidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from core.methods.contracts import (
    BASE_VIEW_NAME,
    BufferRoute,
    MethodRequirements,
    NodeCallable,
    ObjectivePlan,
    ObjectiveTerm,
    ValueKind,
)
from core.methods.nodes import (
    augmix,
    canonical_corruption,
    clean_source,
    lhat_attack,
    paired_latent_bridge,
    replay_write,
    strict_pair,
    vae_decode,
    vae_encode,
)


PROFILE_SCHEMA_VERSION = 1
SUPER5_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
FORBIDDEN_DYNAMIC_KEYS = frozenset(
    {"callable", "class", "class_path", "import", "import_path", "module"}
)
OBJECTIVE_TYPE_MAP = {
    "multilabel_bce_with_logits": "bce",
    "multilabel_bernoulli_jsd": "bernoulli_jsd",
}
_RESOURCE_SCHEMAS_BY_NAME: Mapping[str, tuple[str, frozenset[str]]] = {
    "model": ("caller_owned_classifier", frozenset({"type"})),
    "method_rng": ("none", frozenset({"type"})),
    "operator_profile": (
        "config_reference",
        frozenset({"type", "path", "profile", "severity"}),
    ),
    "vae": ("config_reference", frozenset({"type", "path"})),
    "lhat_config": ("config_reference", frozenset({"type", "path"})),
    "augmix_config": ("config_reference", frozenset({"type", "path"})),
    "latent_pool": (
        "runtime_train_only_exact_label_pool",
        frozenset({"type"}),
    ),
    "vae_encoder": ("runtime_frozen_encoder", frozenset({"type"})),
    "vae_decoder": ("runtime_frozen_decoder", frozenset({"type"})),
    **{
        name: (
            "isolated_torch_generator",
            frozenset({"type", "seed_config", "namespace"}),
        )
        for name in (
            "augmix_rng",
            "branch_rng",
            "corruption_rng",
            "latent_augmix_rng",
            "lhat_rng",
            "replay_rng",
            "simplex_rng",
        )
    },
}


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _sequence(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list")
    return value


def _name(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _reject_dynamic_keys(value: Any, path: str = "profile") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_DYNAMIC_KEYS:
                raise ValueError(
                    f"{path}.{key} is forbidden; profiles may not select callables"
                )
            _reject_dynamic_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_dynamic_keys(nested, f"{path}[{index}]")


def _validate_resources(value: Any) -> dict[str, Any]:
    resources = _mapping(value, "resources")
    for raw_name, raw_resource in resources.items():
        name = _name(raw_name, "resource name")
        try:
            expected_type, expected_keys = _RESOURCE_SCHEMAS_BY_NAME[name]
        except KeyError:
            raise ValueError(f"unregistered method resource {name!r}") from None
        resource = _mapping(raw_resource, f"resources.{name}")
        if set(resource) != expected_keys:
            raise ValueError(
                f"resources.{name} keys must be exactly {sorted(expected_keys)}"
            )
        if resource.get("type") != expected_type:
            raise ValueError(
                f"resources.{name}.type must be {expected_type!r}"
            )
        for key in ("path", "profile", "seed_config", "namespace"):
            if key in resource:
                _name(resource[key], f"resources.{name}.{key}")
        if "severity" in resource:
            severity = resource["severity"]
            if isinstance(severity, bool) or not isinstance(severity, int) or severity <= 0:
                raise ValueError(f"resources.{name}.severity must be a positive integer")
    return resources


@dataclass(frozen=True)
class NodeDefinition:
    """One immutable, code-owned node definition."""

    type_name: str
    output_kind: ValueKind
    execute: NodeCallable
    input_kinds: tuple[ValueKind, ...] = ()
    variadic_input_kind: ValueKind | None = None
    minimum_inputs: int = 0
    maximum_inputs: int | None = 0
    requirements: MethodRequirements = MethodRequirements()

    def __post_init__(self) -> None:
        _name(self.type_name, "node type_name")
        if self.input_kinds and self.variadic_input_kind is not None:
            raise ValueError("node definition cannot be both fixed and variadic")
        if self.minimum_inputs < 0:
            raise ValueError("minimum_inputs must be non-negative")
        if self.maximum_inputs is not None and self.maximum_inputs < self.minimum_inputs:
            raise ValueError("maximum_inputs must be >= minimum_inputs")
        if self.input_kinds:
            object.__setattr__(self, "minimum_inputs", len(self.input_kinds))
            object.__setattr__(self, "maximum_inputs", len(self.input_kinds))

    def validate_input_kinds(
        self,
        node_id: str,
        actual: Sequence[ValueKind],
    ) -> None:
        count = len(actual)
        if count < self.minimum_inputs or (
            self.maximum_inputs is not None and count > self.maximum_inputs
        ):
            upper = "unbounded" if self.maximum_inputs is None else str(self.maximum_inputs)
            raise ValueError(
                f"node {node_id!r} expects {self.minimum_inputs}..{upper} "
                f"graph inputs, got {count}"
            )
        if self.input_kinds and tuple(actual) != self.input_kinds:
            raise TypeError(
                f"node {node_id!r} expects {[v.value for v in self.input_kinds]}, "
                f"got {[v.value for v in actual]}"
            )
        if self.variadic_input_kind is not None and any(
            value is not self.variadic_input_kind for value in actual
        ):
            raise TypeError(
                f"node {node_id!r} accepts only "
                f"{self.variadic_input_kind.value} graph inputs"
            )


class NodeRegistry:
    """Immutable explicit registry with no discovery/import-by-string path."""

    def __init__(self, definitions: Sequence[NodeDefinition]) -> None:
        values: dict[str, NodeDefinition] = {}
        for definition in definitions:
            if definition.type_name in values:
                raise ValueError(f"duplicate node type: {definition.type_name}")
            values[definition.type_name] = definition
        self._definitions = MappingProxyType(values)

    @property
    def type_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def resolve(self, type_name: str) -> NodeDefinition:
        try:
            return self._definitions[type_name]
        except KeyError:
            raise ValueError(
                f"unregistered node type {type_name!r}; allowed={list(self.type_names)}"
            ) from None


# These names match the tracked profile schema exactly.  Experimental node
# types are whitelisted for static compilation/audit, but their profiles carry
# contracts.executable=false and are rejected by the executor.
DEFAULT_REGISTRY = NodeRegistry(
    (
        NodeDefinition(
            type_name="identity_raw100_view",
            output_kind=ValueKind.WAVEFORM,
            execute=clean_source,
        ),
        NodeDefinition(
            type_name="canonical_depth23_corruption_view",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=canonical_corruption,
        ),
        NodeDefinition(
            type_name="vae_lhat_hard_view",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=lhat_attack,
            requirements=MethodRequirements(
                classifier=True,
                vae_decoder=True,
                latent_pool=True,
            ),
        ),
        NodeDefinition(
            type_name="vae_lhat_attack_then_contract_view",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=lhat_attack,
            requirements=MethodRequirements(
                classifier=True,
                vae_decoder=True,
                latent_pool=True,
            ),
        ),
        NodeDefinition(
            type_name="vae_lhat_threechain_augmix_view",
            input_kinds=(ValueKind.WAVEFORM, ValueKind.WAVEFORM),
            output_kind=ValueKind.WAVEFORM,
            execute=augmix,
        ),
        NodeDefinition(
            type_name="latent_threechain_augmix_view",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=augmix,
            requirements=MethodRequirements(
                vae_encoder=True,
                vae_decoder=True,
            ),
        ),
        NodeDefinition(
            type_name="paired_latent_bridge",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=paired_latent_bridge,
            requirements=MethodRequirements(vae_decoder=True, latent_pool=True),
        ),
        NodeDefinition(
            type_name="paired_threechain_augmix_views",
            variadic_input_kind=ValueKind.WAVEFORM,
            minimum_inputs=3,
            maximum_inputs=None,
            output_kind=ValueKind.WAVEFORM,
            execute=augmix,
        ),
        NodeDefinition(
            type_name="five_corruptions_plus_lhat_branch_selector",
            input_kinds=(ValueKind.WAVEFORM, ValueKind.WAVEFORM),
            output_kind=ValueKind.WAVEFORM,
            execute=canonical_corruption,
        ),
        NodeDefinition(
            type_name="run_scoped_lhat_replay_pool",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=replay_write,
            requirements=MethodRequirements(replay_buffer=True),
        ),
        NodeDefinition(
            type_name="canonical_augmix_guidance_probe",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.WAVEFORM,
            execute=augmix,
            requirements=MethodRequirements(classifier=True),
        ),
        NodeDefinition(
            type_name="augmix_guided_latent_simplex_search",
            variadic_input_kind=ValueKind.WAVEFORM,
            minimum_inputs=1,
            maximum_inputs=2,
            output_kind=ValueKind.WAVEFORM,
            execute=lhat_attack,
            requirements=MethodRequirements(
                classifier=True,
                vae_decoder=True,
                latent_pool=True,
            ),
        ),
        # Code-owned primitives retained for future profiles using the same
        # root schema; none permit a YAML-selected implementation path.
        NodeDefinition(
            type_name="strict_paired_waveform_view",
            input_kinds=(ValueKind.WAVEFORM, ValueKind.WAVEFORM),
            output_kind=ValueKind.PAIRED,
            execute=strict_pair,
        ),
        NodeDefinition(
            type_name="ecgtwin_vae_encode_view",
            input_kinds=(ValueKind.WAVEFORM,),
            output_kind=ValueKind.LATENT,
            execute=vae_encode,
            requirements=MethodRequirements(vae_encoder=True),
        ),
        NodeDefinition(
            type_name="ecgtwin_vae_decode_view",
            input_kinds=(ValueKind.LATENT,),
            output_kind=ValueKind.WAVEFORM,
            execute=vae_decode,
            requirements=MethodRequirements(vae_decoder=True),
        ),
    )
)


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    node_type: str
    inputs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.node_id, "node id")
        _name(self.node_type, "node type")
        if any(not isinstance(value, str) or not value for value in self.inputs):
            raise ValueError("node inputs must contain non-empty node ids")
        params = dict(self.params)
        _reject_dynamic_keys(params, f"nodes.{self.node_id}.params")
        object.__setattr__(self, "params", MappingProxyType(params))


@dataclass(frozen=True)
class MethodProfile:
    profile_name: str
    scientific_arm: str
    status: str
    executable: bool
    rng_namespace: str
    comparison_rng_identity: str
    nodes: tuple[NodeProfile, ...]
    outputs: Mapping[str, str]
    output_kinds: Mapping[str, ValueKind]
    objective: ObjectivePlan
    resources: Mapping[str, Any]
    contracts: Mapping[str, Any]
    raw_payload: Mapping[str, Any]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        _name(self.profile_name, "method.id")
        _name(self.scientific_arm, "method.scientific_arm")
        _name(self.status, "method.status")
        if not isinstance(self.executable, bool):
            raise TypeError("contracts.executable must be boolean")
        _name(self.rng_namespace, "rng_namespace")
        _name(self.comparison_rng_identity, "contracts.comparison_rng_identity")
        if not self.nodes:
            raise ValueError("method profile must contain at least one node")
        outputs = dict(self.outputs)
        output_kinds = dict(self.output_kinds)
        if not outputs:
            raise ValueError("method profile must expose at least one waveform output")
        if set(outputs) != set(output_kinds):
            raise ValueError("outputs and output_kinds must use identical names")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        object.__setattr__(self, "output_kinds", MappingProxyType(output_kinds))
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))
        object.__setattr__(self, "raw_payload", MappingProxyType(dict(self.raw_payload)))


@dataclass(frozen=True)
class CompiledNode:
    profile: NodeProfile
    definition: NodeDefinition


@dataclass(frozen=True)
class CompiledMethod:
    profile_name: str
    scientific_arm: str
    status: str
    executable: bool
    rng_namespace: str
    comparison_rng_identity: str
    nodes: tuple[CompiledNode, ...]
    outputs: Mapping[str, str]
    output_kinds: Mapping[str, ValueKind]
    objective: ObjectivePlan
    requirements: MethodRequirements
    resources: Mapping[str, Any]
    contracts: Mapping[str, Any]
    profile_sha256: str
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(
            self, "output_kinds", MappingProxyType(dict(self.output_kinds))
        )
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))

    def describe(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "scientific_arm": self.scientific_arm,
            "status": self.status,
            "executable": self.executable,
            "profile_sha256": self.profile_sha256,
            "source_path": None if self.source_path is None else str(self.source_path),
            "rng_namespace": self.rng_namespace,
            "comparison_rng_identity": self.comparison_rng_identity,
            "topological_nodes": [node.profile.node_id for node in self.nodes],
            "outputs": dict(self.outputs),
            "output_kinds": {
                name: kind.value for name, kind in self.output_kinds.items()
            },
            "requirements": list(self.requirements.names()),
            "objective_terms": [term.name for term in self.objective.terms],
            "resources": dict(self.resources),
            "contracts": dict(self.contracts),
        }


def _validate_external_inputs(value: Any) -> None:
    inputs = _mapping(value, "inputs")
    required = {"clean_raw", "targets", "hash_ids"}
    if set(inputs) != required:
        raise ValueError(f"inputs must contain exactly {sorted(required)}")
    clean = _mapping(inputs["clean_raw"], "inputs.clean_raw")
    expected_clean = {
        "type": "ecg_waveform",
        "domain": "raw_mV",
        "layout": "batch_time_lead",
        "sampling_rate_hz": 100,
        "points": 1000,
        "leads": 12,
    }
    if clean != expected_clean:
        raise ValueError("inputs.clean_raw must be canonical raw-mV (B,1000,12)")
    targets = _mapping(inputs["targets"], "inputs.targets")
    if targets.get("type") != "multilabel_targets" or tuple(
        targets.get("class_order", ())
    ) != SUPER5_ORDER:
        raise ValueError("inputs.targets must use Super5 CD/HYP/MI/NORM/STTC")
    hashes = _mapping(inputs["hash_ids"], "inputs.hash_ids")
    if hashes != {"type": "unique_record_ids"}:
        raise ValueError("inputs.hash_ids must be unique_record_ids")


def _parse_objective(value: Any) -> ObjectivePlan:
    raw = _mapping(value, "objective")
    unknown = set(raw) - {"eligibility_scaling", "terms", "buffer_routes"}
    if unknown:
        raise ValueError(f"objective contains unexpected keys: {sorted(unknown)}")
    terms: list[ObjectiveTerm] = []
    for index, item in enumerate(_sequence(raw.get("terms", []), "objective.terms")):
        payload = _mapping(item, f"objective.terms[{index}]")
        unknown_term = set(payload) - {
            "id",
            "type",
            "view",
            "views",
            "targets",
            "weight",
            "mask_policy",
        }
        if unknown_term:
            raise ValueError(
                f"objective term contains unexpected keys: {sorted(unknown_term)}"
            )
        raw_type = _name(payload.get("type"), "objective term type")
        try:
            kind = OBJECTIVE_TYPE_MAP[raw_type]
        except KeyError:
            raise ValueError(f"unsupported objective term type: {raw_type}") from None
        if kind == "bce" and payload.get("targets") != "targets":
            raise ValueError("BCE objective terms must use the declared targets input")
        if kind == "bernoulli_jsd" and payload.get("targets") not in {None, "targets"}:
            raise ValueError("JSD targets, when declared, must reference targets")
        if ("view" in payload) == ("views" in payload):
            raise ValueError("objective term must declare exactly one of view or views")
        raw_views = (
            [payload["view"]]
            if "view" in payload
            else _sequence(payload["views"], "objective term views")
        )
        terms.append(
            ObjectiveTerm(
                name=_name(payload.get("id"), "objective term id"),
                kind=kind,
                views=tuple(_name(item, "objective view") for item in raw_views),
                weight=payload.get("weight"),
                mask_policy=str(payload.get("mask_policy", "valid_intersection")),
            )
        )
    routes: list[BufferRoute] = []
    for index, item in enumerate(
        _sequence(raw.get("buffer_routes", []), "objective.buffer_routes")
    ):
        payload = _mapping(item, f"objective.buffer_routes[{index}]")
        if set(payload) - {"view", "stream", "weight"}:
            raise ValueError("buffer route has unexpected keys")
        routes.append(
            BufferRoute(
                view=_name(payload.get("view"), "buffer route view"),
                stream=_name(payload.get("stream"), "buffer route stream"),
                weight=payload.get("weight", 1.0),
            )
        )
    return ObjectivePlan(tuple(terms), tuple(routes))


def _profile_rng_namespace(
    method_id: str,
    input_bindings: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> str:
    rng_name = input_bindings.get("rng")
    if isinstance(rng_name, str):
        payload = resources.get(rng_name)
        if isinstance(payload, Mapping) and isinstance(payload.get("namespace"), str):
            return str(payload["namespace"])
    return method_id


_NODE_RESOURCE_BINDINGS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "canonical_depth23_corruption_view": {
            "rng": "corruption_rng",
            "operator_profile": "operator_profile",
        },
        "vae_lhat_hard_view": {
            "latent_pool": "latent_pool",
            "decoder": "vae_decoder",
            "classifier": "model",
            "rng": "lhat_rng",
            "config": "lhat_config",
        },
        "vae_lhat_attack_then_contract_view": {
            "latent_pool": "latent_pool",
            "decoder": "vae_decoder",
            "classifier": "model",
            "rng": "lhat_rng",
            "config": "lhat_config",
        },
        "vae_lhat_threechain_augmix_view": {
            "rng": "augmix_rng",
            "config": "augmix_config",
        },
        "latent_threechain_augmix_view": {
            "encoder": "vae_encoder",
            "decoder": "vae_decoder",
            "rng": "latent_augmix_rng",
            "config": "augmix_config",
        },
        "ecgtwin_vae_encode_view": {
            "encoder": "vae_encoder",
        },
        "ecgtwin_vae_decode_view": {
            "decoder": "vae_decoder",
        },
    }
)


def _validate_node_resource_bindings(
    node_id: str,
    node_type: str,
    input_bindings: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> None:
    expected = _NODE_RESOURCE_BINDINGS.get(node_type)
    if expected is None:
        return
    for port, resource_name in expected.items():
        if input_bindings.get(port) != resource_name:
            raise ValueError(
                f"nodes.{node_id}.inputs.{port} must bind {resource_name!r}"
            )
        if resource_name not in resources:
            raise ValueError(
                f"nodes.{node_id}.inputs.{port} references missing resource "
                f"{resource_name!r}"
            )


def load_method_profile(source: str | Path | Mapping[str, Any]) -> MethodProfile:
    """Load the tracked project method schema without importing user code."""

    source_path: Path | None = None
    if isinstance(source, Mapping):
        root = dict(source)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"method profile not found: {source_path}")
        root = _mapping(
            yaml.safe_load(source_path.read_text(encoding="utf-8")),
            "method profile",
        )
    expected_root = {
        "schema_version",
        "method",
        "nodes",
        "inputs",
        "outputs",
        "objective",
        "resources",
        "contracts",
    }
    if set(root) != expected_root:
        raise ValueError(
            "method profile keys are incomplete or unexpected: "
            f"expected={sorted(expected_root)}, got={sorted(root)}"
        )
    if root.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"method profile schema_version must be {PROFILE_SCHEMA_VERSION}")
    _reject_dynamic_keys(root)
    _validate_external_inputs(root["inputs"])

    method = _mapping(root["method"], "method")
    expected_method = {
        "id",
        "profile_version",
        "api_version",
        "scientific_arm",
        "status",
        "description",
    }
    if set(method) != expected_method:
        raise ValueError("method keys are incomplete or unexpected")
    if method.get("profile_version") != 1 or method.get("api_version") != 1:
        raise ValueError("method profile_version and api_version must be 1")
    method_id = _name(method.get("id"), "method.id")
    status = _name(method.get("status"), "method.status")

    resources = _validate_resources(root["resources"])
    contracts = _mapping(root["contracts"], "contracts")
    comparison_rng_identity = _name(
        contracts.get("comparison_rng_identity", method_id),
        "contracts.comparison_rng_identity",
    )
    executable = contracts.get("executable", True)
    if not isinstance(executable, bool):
        raise TypeError("contracts.executable must be boolean")
    if status.startswith("experimental_") and executable:
        raise ValueError("experimental profiles must declare contracts.executable=false")

    raw_nodes = _mapping(root["nodes"], "nodes")
    if not raw_nodes:
        raise ValueError("nodes must not be empty")
    parsed_nodes: dict[str, dict[str, Any]] = {}
    alias_to_node: dict[str, str] = {}
    alias_to_port: dict[str, tuple[str, str]] = {}
    for node_id, raw_node in raw_nodes.items():
        node_name = _name(node_id, "node id")
        payload = _mapping(raw_node, f"nodes.{node_name}")
        expected_node = {
            "type",
            "implementation",
            "implementation_version",
            "status",
            "inputs",
            "outputs",
        }
        if set(payload) != expected_node:
            raise ValueError(f"node {node_name!r} keys are incomplete or unexpected")
        if payload.get("implementation_version") != 1:
            raise ValueError(f"node {node_name!r} implementation_version must be 1")
        if executable and payload.get("implementation") not in {"builtin", "registered"}:
            raise ValueError(f"executable node {node_name!r} has no implementation")
        output_bindings = _mapping(payload["outputs"], f"nodes.{node_name}.outputs")
        for port, alias in output_bindings.items():
            port_name = _name(port, "node output port")
            alias_name = _name(alias, "node output alias")
            if alias_name in alias_to_node:
                raise ValueError(f"duplicate node output alias: {alias_name}")
            alias_to_node[alias_name] = node_name
            alias_to_port[alias_name] = (node_name, port_name)
        parsed_nodes[node_name] = payload

    nodes: list[NodeProfile] = []
    node_rng_namespaces: list[str] = []
    for node_id, payload in parsed_nodes.items():
        input_bindings = _mapping(payload["inputs"], f"nodes.{node_id}.inputs")
        _validate_node_resource_bindings(
            node_id,
            str(payload["type"]),
            input_bindings,
            resources,
        )
        graph_inputs = tuple(
            alias_to_node[value]
            for value in input_bindings.values()
            if isinstance(value, str) and value in alias_to_node
        )
        params: dict[str, Any]
        if payload["type"] == "identity_raw100_view":
            params = {"source": input_bindings.get("waveform")}
        else:
            params = {
                "input_bindings": dict(input_bindings),
                "output_bindings": dict(payload["outputs"]),
                "implementation": payload["implementation"],
                "implementation_version": payload["implementation_version"],
                "status": payload["status"],
            }
            rng_namespace = _profile_rng_namespace(method_id, input_bindings, resources)
            params["rng_namespace"] = rng_namespace
            node_rng_namespaces.append(rng_namespace)
        nodes.append(
            NodeProfile(
                node_id=node_id,
                node_type=_name(payload.get("type"), f"nodes.{node_id}.type"),
                inputs=graph_inputs,
                params=params,
            )
        )

    outputs: dict[str, str] = {}
    output_kinds: dict[str, ValueKind] = {}
    for output_name, raw_output in _mapping(root["outputs"], "outputs").items():
        name = _name(output_name, "output name")
        payload = _mapping(raw_output, f"outputs.{name}")
        if payload.get("type") != "ecg_waveform":
            continue
        source_value = _name(payload.get("source"), f"outputs.{name}.source")
        if "." not in source_value:
            raise ValueError(f"outputs.{name}.source must be node.port")
        source_node, source_port = source_value.split(".", 1)
        if alias_to_port.get(name) != (source_node, source_port):
            raise ValueError(
                f"outputs.{name} does not match node output binding {source_value}"
            )
        outputs[name] = source_node
        output_kinds[name] = ValueKind.WAVEFORM

    objective = _parse_objective(root["objective"])
    return MethodProfile(
        profile_name=method_id,
        scientific_arm=_name(method.get("scientific_arm"), "method.scientific_arm"),
        status=status,
        executable=executable,
        rng_namespace=(
            node_rng_namespaces[0]
            if len(set(node_rng_namespaces)) == 1 and node_rng_namespaces
            else method_id
        ),
        comparison_rng_identity=comparison_rng_identity,
        nodes=tuple(nodes),
        outputs=outputs,
        output_kinds=output_kinds,
        objective=objective,
        resources=resources,
        contracts=contracts,
        raw_payload=root,
        source_path=source_path,
    )


def _topological_order(nodes: Mapping[str, NodeProfile]) -> tuple[str, ...]:
    indegree = {node_id: 0 for node_id in nodes}
    children = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        for parent in node.inputs:
            if parent not in nodes:
                raise ValueError(f"node {node_id!r} references unknown input {parent!r}")
            indegree[node_id] += 1
            children[parent].append(node_id)
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(nodes):
        cyclic = sorted(node_id for node_id, count in indegree.items() if count > 0)
        raise ValueError(f"method graph contains a cycle involving: {cyclic}")
    return tuple(ordered)


def compile_method_profile(
    profile: MethodProfile | str | Path | Mapping[str, Any],
    *,
    registry: NodeRegistry = DEFAULT_REGISTRY,
) -> CompiledMethod:
    """Validate graph acyclicity, typed edges, outputs, and objectives."""

    resolved = profile if isinstance(profile, MethodProfile) else load_method_profile(profile)
    by_id: dict[str, NodeProfile] = {}
    for node in resolved.nodes:
        if node.node_id in by_id:
            raise ValueError(f"duplicate node id: {node.node_id}")
        registry.resolve(node.node_type)
        by_id[node.node_id] = node
    order = _topological_order(by_id)
    kinds: dict[str, ValueKind] = {}
    compiled: list[CompiledNode] = []
    requirements = MethodRequirements()
    for node_id in order:
        node = by_id[node_id]
        definition = registry.resolve(node.node_type)
        definition.validate_input_kinds(node_id, [kinds[parent] for parent in node.inputs])
        kinds[node_id] = definition.output_kind
        requirements = requirements.union(definition.requirements)
        compiled.append(CompiledNode(profile=node, definition=definition))

    for output_name, node_id in resolved.outputs.items():
        if node_id not in kinds:
            raise ValueError(
                f"named output {output_name!r} references unknown node {node_id!r}"
            )
        expected = resolved.output_kinds[output_name]
        if kinds[node_id] is not expected:
            raise TypeError(
                f"named output {output_name!r} declares {expected.value}, "
                f"producer returns {kinds[node_id].value}"
            )
    for referenced in resolved.objective.referenced_outputs():
        if referenced not in resolved.output_kinds:
            raise ValueError(f"objective references unknown waveform output {referenced!r}")
        if resolved.output_kinds[referenced] is not ValueKind.WAVEFORM:
            raise TypeError(f"objective output {referenced!r} must be waveform")
    if resolved.output_kinds.get(BASE_VIEW_NAME) is not ValueKind.WAVEFORM:
        raise ValueError(
            f"every method profile must expose canonical base view {BASE_VIEW_NAME!r}"
        )
    if resolved.objective.terms:
        requirements = requirements.union(MethodRequirements(classifier=True))
    if resolved.objective.buffer_routes:
        requirements = requirements.union(MethodRequirements(replay_buffer=True))

    try:
        identity = json.dumps(
            dict(resolved.raw_payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"method profile must be JSON-serializable: {exc}") from exc
    return CompiledMethod(
        profile_name=resolved.profile_name,
        scientific_arm=resolved.scientific_arm,
        status=resolved.status,
        executable=resolved.executable,
        rng_namespace=resolved.rng_namespace,
        comparison_rng_identity=resolved.comparison_rng_identity,
        nodes=tuple(compiled),
        outputs=resolved.outputs,
        output_kinds=resolved.output_kinds,
        objective=resolved.objective,
        requirements=requirements,
        resources=resolved.resources,
        contracts=resolved.contracts,
        profile_sha256=hashlib.sha256(identity).hexdigest(),
        source_path=resolved.source_path,
    )


__all__ = [
    "CompiledMethod",
    "CompiledNode",
    "DEFAULT_REGISTRY",
    "MethodProfile",
    "NodeDefinition",
    "NodeProfile",
    "NodeRegistry",
    "PROFILE_SCHEMA_VERSION",
    "compile_method_profile",
    "load_method_profile",
]
