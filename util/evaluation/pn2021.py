"""Managed PN2021/PN2021-C evaluation over the rebuilt runtime boundary."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from data_preprocess.data_runtime import (
    PN2021EvaluationLoaderPlan,
)
from data_preprocess.load_cache import EXPECTED_LEADS
from models.checkpoints import CheckpointIdentity
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
    validate_model_output,
)
from models.input_adapter import (
    CANONICAL_CHANNELS,
    CANONICAL_POINTS,
    CANONICAL_SAMPLING_RATE_HZ,
    ECGFOUNDER_TARGET_POINTS,
    INTERPOLATION_ALIGN_CORNERS,
    INTERPOLATION_MODE,
    prepare_canonical_model_input,
)
from util.config_bundle import (
    config_bundle_root,
    require_mapping as _mapping,
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.evaluation.metrics import (
    CLEAN_VIEW_ALIASES,
    CORRUPTED_VIEW_ALIASES,
    aggregate_corruption_views,
    compute_metric_views,
    mean_metric_views,
    validate_evaluation_aggregates,
)
from util.pn2021_artifact_contract import (
    CENTER_SOURCES,
    LOGICAL_CENTERS,
    MAPPING_HASH,
    MAPPING_VERSION,
    build_artifact_reference,
    load_json_mapping,
    sha256_file,
)
from util.random_seed import seed_process


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PN2021_EVAL_CONFIG = PROJECT_ROOT / "configs" / "eval" / "PN2021.yaml"
PN2021_MAPPING_VERSION = MAPPING_VERSION
PN2021_MAPPING_HASH = MAPPING_HASH
_LOCKED_INPUT_PIPELINE = {
    "canonical_domain": {
        "sampling_rate_hz": CANONICAL_SAMPLING_RATE_HZ, "duration_seconds": 10.0,
        "points": CANONICAL_POINTS, "channels": CANONICAL_CHANNELS,
        "layout": "time_channel", "physical_unit": "mV", "normalization": "none",
    },
    "sanitization": {"stage": "before_model_domain_adaptation",
                     "nonfinite_replacement": "zero"},
    "model_domains": {
        EFFICIENTNET1DV2_SPEC.name: {
            "sampling_rate_hz": EFFICIENTNET1DV2_SPEC.sampling_rate_hz,
            "points": EFFICIENTNET1DV2_SPEC.input_points, "adaptation": "identity",
        },
        ECGFOUNDER_SPEC.name: {
            "sampling_rate_hz": ECGFOUNDER_SPEC.sampling_rate_hz,
            "points": ECGFOUNDER_TARGET_POINTS, "adaptation": "interpolate",
            "mode": INTERPOLATION_MODE,
            "align_corners": INTERPOLATION_ALIGN_CORNERS,
        },
    },
    "normalization": {
        "stage": "after_model_domain_adaptation", "method": "per_sample_global_zscore",
        "epsilon": 1e-6, "variance_correction": 0,
    },
    "output": {"layout": "channel_time", "dtype": "float32"},
}


def _yaml_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} not found: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a YAML mapping")
    return payload


def _exact_keys(
    value: Mapping[str, Any], *, expected: set[str], description: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _positive_int(value: Any, *, description: str, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        comparator = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{description} must be a {comparator} integer")
    return int(value)


def _boolean(value: Any, *, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be boolean")
    return value


def _locked_sha256(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA256 digest")
    return value


def _validate_input_pipeline(protocol: Mapping[str, Any]) -> None:
    pipeline = _mapping(
        protocol["input_pipeline"], description="protocol.input_pipeline"
    )
    if pipeline != _LOCKED_INPUT_PIPELINE:
        raise ValueError("unsupported protocol.input_pipeline")
    ecgfounder = pipeline["model_domains"][ECGFOUNDER_SPEC.name]
    if not isinstance(ecgfounder["align_corners"], bool):
        raise ValueError("unsupported ECGFounder model-domain input contract")
    normalization = pipeline["normalization"]
    epsilon = normalization["epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not np.isfinite(float(epsilon))
        or float(epsilon) != 1e-6
        or isinstance(normalization["variance_correction"], bool)
        or not isinstance(normalization["variance_correction"], int)
        or normalization["variance_correction"] != 0
    ):
        raise ValueError("unsupported protocol.input_pipeline.normalization")


def _resolved_input_pipeline(
    config: "PN2021EvalConfig", spec: ModelSpec
) -> dict[str, Any]:
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError(
            "PN2021 evaluation supports only the canonical managed "
            "EfficientNet1DV2 and ECGFounder ModelSpecs"
        )
    pipeline = config.payload["protocol"]["input_pipeline"]
    model_domains = pipeline["model_domains"]
    if spec.name not in model_domains:
        raise ValueError(f"unsupported evaluation ModelSpec: {spec.name}")
    return {
        "canonical_domain": dict(pipeline["canonical_domain"]),
        "sanitization": dict(pipeline["sanitization"]),
        "model_domain": {
            "model": spec.name,
            **dict(model_domains[spec.name]),
        },
        "normalization": dict(pipeline["normalization"]),
        "output": dict(pipeline["output"]),
    }


@dataclass(frozen=True)
class CorruptionView:
    view_index: int
    depth: int
    operators: tuple[str, ...]
    composition_id: str

    def describe(self) -> dict[str, Any]:
        return {
            "view_index": self.view_index,
            "depth": self.depth,
            "operators": list(self.operators),
            "composition_id": self.composition_id,
        }


@dataclass(frozen=True)
class PN2021EvalConfig:
    path: Path
    sha256: str
    config_root: Path
    split_config_path: Path
    data_load_config_path: Path
    corruption_cache_config_path: Path
    random_seed_config_path: Path
    payload: dict[str, Any]

    @property
    def profile_name(self) -> str:
        return str(self.payload["profile_name"])

    @property
    def logical_centers(self) -> tuple[str, ...]:
        return tuple(self.payload["protocol"]["logical_centers"])

    def describe(self) -> dict[str, Any]:
        references = {
            "split_config": self.split_config_path,
            "data_load_config": self.data_load_config_path,
            "corruption_cache_config": self.corruption_cache_config_path,
            "random_seed_config": self.random_seed_config_path,
        }
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_root": str(self.config_root),
            "references": {
                name: build_artifact_reference(path)
                for name, path in references.items()
            },
            "resolved": self.payload,
        }


def _resolve_evaluation_centers(
    config: PN2021EvalConfig,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        return config.logical_centers
    centers = tuple(str(value) for value in requested)
    if not centers or len(set(centers)) != len(centers):
        raise ValueError("logical_centers override must be non-empty and unique")
    unknown = sorted(set(centers) - set(config.logical_centers))
    if unknown:
        raise ValueError(f"unknown logical center override: {unknown}")
    return centers


def load_pn2021_eval_config(
    path: str | Path = DEFAULT_PN2021_EVAL_CONFIG,
    *,
    config_root: str | Path | None = None,
) -> PN2021EvalConfig:
    """Load and strictly validate one configs-shaped PN2021 evaluation profile."""

    config_path = resolve_entry_config_path(path)
    payload = _yaml_mapping(config_path, description="PN2021 evaluation config")
    _exact_keys(
        payload,
        expected={
            "schema_version",
            "profile_name",
            "references",
            "protocol",
            "artifact_locks",
            "runtime",
            "output",
        },
        description="PN2021 evaluation config",
    )
    if payload["schema_version"] != 2:
        raise ValueError("PN2021 evaluation schema_version must be 2")
    if not isinstance(payload["profile_name"], str) or not payload["profile_name"]:
        raise ValueError("PN2021 evaluation profile_name must be non-empty")
    root = config_bundle_root(config_path, config_root=config_root)
    references = _mapping(payload["references"], description="references")
    _exact_keys(
        references,
        expected={
            "split_config",
            "data_load_config",
            "corruption_cache_config",
            "random_seed_config",
        },
        description="references",
    )

    def reference(name: str) -> Path:
        return resolve_config_reference(
            references[name],
            owner_config_path=config_path,
            config_root=root,
            description=f"references.{name}",
            must_exist=True,
        )

    protocol = _mapping(payload["protocol"], description="protocol")
    _exact_keys(
        protocol,
        expected={
            "mapping_version",
            "mapping_hash",
            "class_order",
            "logical_centers",
            "reference_count_per_center",
            "partitions",
            "canonical_views",
            "corruption",
            "input_pipeline",
        },
        description="protocol",
    )
    if (
        protocol["mapping_version"] != PN2021_MAPPING_VERSION
        or protocol["mapping_hash"] != PN2021_MAPPING_HASH
    ):
        raise ValueError("PN2021 evaluation mapping identity is not the locked v7 map")
    if tuple(protocol["class_order"]) != CLASS_ORDER:
        raise ValueError(f"protocol.class_order must be {CLASS_ORDER}")
    if tuple(protocol["logical_centers"]) != LOGICAL_CENTERS:
        raise ValueError(f"protocol.logical_centers must be {LOGICAL_CENTERS}")
    if _positive_int(
        protocol["reference_count_per_center"],
        description="protocol.reference_count_per_center",
    ) != 500:
        raise ValueError("PN2021 evaluation requires exactly K500 ref exclusion")
    partitions = _mapping(protocol["partitions"], description="protocol.partitions")
    _exact_keys(
        partitions,
        expected={"clean", "corrupted"},
        description="protocol.partitions",
    )
    if partitions != {
        "clean": "pn2021_all_zero_kept_refexcluded",
        "corrupted": "pn2021c_all_zero_kept_corrupted_refexcluded",
    }:
        raise ValueError("protocol partitions must use canonical ref-excluded aliases")
    aliases = _mapping(
        protocol["canonical_views"], description="protocol.canonical_views"
    )
    _exact_keys(
        aliases,
        expected={"clean_kept", "clean_drop", "corrupted_kept", "corrupted_drop"},
        description="protocol.canonical_views",
    )
    if tuple(aliases.values()) != CLEAN_VIEW_ALIASES + CORRUPTED_VIEW_ALIASES:
        raise ValueError("protocol canonical views do not match the locked aliases")
    corruption = _mapping(protocol["corruption"], description="protocol.corruption")
    _exact_keys(
        corruption,
        expected={
            "expected_profile",
            "expected_severity",
            "expected_domain_sampling_rate_hz",
            "expected_view_count",
            "depths",
            "combinations_per_depth",
            "aggregation",
        },
        description="protocol.corruption",
    )
    if (
        corruption["expected_profile"] != "pn2021c_paper_anchored_s5_v1"
        or _positive_int(
            corruption["expected_severity"],
            description="corruption.expected_severity",
        )
        != 5
        or _positive_int(
            corruption["expected_domain_sampling_rate_hz"],
            description="corruption.expected_domain_sampling_rate_hz",
        )
        != 500
    ):
        raise ValueError(
            "PN2021-C evaluation requires the locked paper-anchored severity-5 "
            "profile in the raw 500 Hz corruption domain"
        )
    if _positive_int(
        corruption["expected_view_count"], description="corruption.expected_view_count"
    ) != 20:
        raise ValueError("PN2021-C evaluation requires exactly 20 views")
    if tuple(corruption["depths"]) != (2, 3):
        raise ValueError("PN2021-C evaluation depths must be [2,3]")
    counts = _mapping(
        corruption["combinations_per_depth"],
        description="corruption.combinations_per_depth",
    )
    normalized_counts = {int(key): int(value) for key, value in counts.items()}
    if normalized_counts != {2: 10, 3: 10}:
        raise ValueError("PN2021-C evaluation requires 10 depth2 and 10 depth3 views")
    if corruption["aggregation"] != "equal_views_then_equal_centers":
        raise ValueError("unsupported corruption aggregation")
    _validate_input_pipeline(protocol)

    artifact_locks = _mapping(payload["artifact_locks"], description="artifact_locks")
    _exact_keys(
        artifact_locks,
        expected={
            "clean_cache_manifest_sha256",
            "split_manifest_sha256",
            "corruption_cache_manifest_sha256",
            "compositions_sha256",
        },
        description="artifact_locks",
    )
    for name, value in artifact_locks.items():
        _locked_sha256(value, description=f"artifact_locks.{name}")

    runtime = _mapping(payload["runtime"], description="runtime")
    _exact_keys(
        runtime,
        expected={
            "device",
            "batch_size_by_model",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "cache_mode",
            "validate_values",
            "amp",
        },
        description="runtime",
    )
    if runtime["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("runtime.device must be cpu, cuda or auto")
    batch_sizes = _mapping(
        runtime["batch_size_by_model"], description="runtime.batch_size_by_model"
    )
    if set(batch_sizes) != {"efficientnet1dv2", "ecgfounder"}:
        raise ValueError("runtime.batch_size_by_model must cover both model specs")
    for name, value in batch_sizes.items():
        _positive_int(value, description=f"runtime.batch_size_by_model.{name}")
    configured_workers = _positive_int(
        runtime["num_workers"],
        description="runtime.num_workers",
        allow_zero=True,
    )
    _boolean(runtime["pin_memory"], description="runtime.pin_memory")
    configured_persistent = _boolean(
        runtime["persistent_workers"], description="runtime.persistent_workers"
    )
    if configured_workers != 0 or configured_persistent:
        raise ValueError("PN2021 evaluation requires single-process workers=0")
    _positive_int(runtime["prefetch_factor"], description="runtime.prefetch_factor")
    if runtime["cache_mode"] != "mmap":
        raise ValueError("PN2021 evaluation requires runtime.cache_mode=mmap")
    if runtime["validate_values"] not in {"none", "sample", "full"}:
        raise ValueError("unsupported runtime.validate_values")
    amp = _mapping(runtime["amp"], description="runtime.amp")
    _exact_keys(amp, expected={"enabled", "dtype"}, description="runtime.amp")
    _boolean(amp["enabled"], description="runtime.amp.enabled")
    if amp["dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("runtime.amp.dtype must be bfloat16 or float16")

    output = _mapping(payload["output"], description="output")
    _exact_keys(
        output,
        expected={"run_dir", "if_exists", "result_file"},
        description="output",
    )
    if not isinstance(output["run_dir"], str) or not output["run_dir"]:
        raise ValueError("output.run_dir must be a non-empty path")
    if output["if_exists"] != "error":
        raise ValueError("output.if_exists must be error")
    result_file = Path(str(output["result_file"]))
    if result_file.is_absolute() or ".." in result_file.parts or result_file.suffix != ".json":
        raise ValueError("output.result_file must be a safe relative JSON path")

    return PN2021EvalConfig(
        path=config_path,
        sha256=sha256_file(config_path),
        config_root=root,
        split_config_path=reference("split_config"),
        data_load_config_path=reference("data_load_config"),
        corruption_cache_config_path=reference("corruption_cache_config"),
        random_seed_config_path=reference("random_seed_config"),
        payload=payload,
    )


def _cache_member(root: Path, value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{description} must stay inside the cache")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"{description} escapes the cache") from None
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_corruption_views(
    config: PN2021EvalConfig,
) -> tuple[tuple[CorruptionView, ...], dict[str, Any]]:
    """Read and validate the ordered PN2021-C view identity without waveform IO."""

    cache_config = _yaml_mapping(
        config.corruption_cache_config_path,
        description="corruption cache config",
    )
    cache_corruption = _mapping(
        cache_config.get("corruption"), description="cache corruption"
    )
    expected_corruption = _mapping(
        config.payload["protocol"]["corruption"],
        description="protocol.corruption",
    )
    expected_semantics = {
        "profile": str(expected_corruption["expected_profile"]),
        "severity": int(expected_corruption["expected_severity"]),
        "domain_sampling_rate_hz": int(
            expected_corruption["expected_domain_sampling_rate_hz"]
        ),
    }
    cache_semantics = {
        "profile": cache_corruption.get("profile"),
        "severity": cache_corruption.get("severity"),
        "domain_sampling_rate_hz": cache_corruption.get(
            "domain_sampling_rate_hz"
        ),
    }
    if cache_semantics != expected_semantics:
        raise ValueError(
            "PN2021-C cache config corruption semantics differ from the locked "
            f"evaluation protocol: expected={expected_semantics}, "
            f"cache_config={cache_semantics}"
        )
    output = _mapping(cache_config.get("output"), description="cache output")
    raw_root = Path(str(output.get("cache_dir", ""))).expanduser()
    if not raw_root.is_absolute():
        raw_root = config.config_root / raw_root
    cache_root = raw_root.resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = load_json_mapping(manifest_path, name="PN2021-C manifest")
    artifact_locks = config.payload["artifact_locks"]
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != artifact_locks["corruption_cache_manifest_sha256"]:
        raise ValueError("PN2021-C cache manifest differs from the locked artifact")
    if manifest.get("dataset") != "pn2021c" or manifest.get("schema_version") != 1:
        raise ValueError("corruption cache must be a schema_version=1 PN2021-C cache")
    if int(manifest.get("view_count", -1)) != 20:
        raise ValueError("corruption cache must contain exactly 20 views")
    source = _mapping(manifest.get("source"), description="manifest.source")
    if (
        source.get("mapping_version") != PN2021_MAPPING_VERSION
        or source.get("mapping_hash") != PN2021_MAPPING_HASH
        or tuple(source.get("class_order", ())) != CLASS_ORDER
    ):
        raise ValueError("corruption cache mapping/class identity mismatch")
    files = _mapping(manifest.get("files"), description="manifest.files")
    compositions_path = _cache_member(
        cache_root,
        files.get("compositions"),
        description="PN2021-C compositions",
    )
    compositions_sha256 = sha256_file(compositions_path)
    if compositions_sha256 != artifact_locks["compositions_sha256"]:
        raise ValueError("PN2021-C compositions differ from the locked artifact")
    try:
        raw_views = json.loads(compositions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid PN2021-C compositions JSON: {exc}") from exc
    if not isinstance(raw_views, list) or len(raw_views) != 20:
        raise ValueError("PN2021-C compositions must contain exactly 20 entries")
    views: list[CorruptionView] = []
    identifiers: set[str] = set()
    for expected_index, raw in enumerate(raw_views):
        item = _mapping(raw, description=f"composition[{expected_index}]")
        depth = int(item.get("depth", -1))
        operators = tuple(str(value) for value in item.get("operators", ()))
        identifier = str(item.get("composition_id", ""))
        if int(item.get("view_index", -1)) != expected_index:
            raise ValueError("PN2021-C composition indices must be contiguous")
        if depth not in {2, 3} or len(operators) != depth:
            raise ValueError("PN2021-C composition depth/operator mismatch")
        if not identifier or identifier in identifiers:
            raise ValueError("PN2021-C composition IDs must be unique")
        identifiers.add(identifier)
        views.append(CorruptionView(expected_index, depth, operators, identifier))
    if sum(view.depth == 2 for view in views) != 10 or sum(
        view.depth == 3 for view in views
    ) != 10:
        raise ValueError("PN2021-C cache must contain 10 depth2 and 10 depth3 views")
    corruption = _mapping(manifest.get("corruption"), description="manifest.corruption")
    manifest_semantics = {
        "profile": corruption.get("profile"),
        "severity": corruption.get("severity"),
        "domain_sampling_rate_hz": corruption.get("domain_sampling_rate_hz"),
    }
    if manifest_semantics != expected_semantics:
        raise ValueError(
            "PN2021-C cache manifest corruption semantics differ from the locked "
            f"evaluation protocol: expected={expected_semantics}, "
            f"manifest={manifest_semantics}"
        )
    identity = {
        "cache_dir": str(cache_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "compositions_path": str(compositions_path),
        "compositions_sha256": compositions_sha256,
        "view_count": len(views),
        "profile": corruption.get("profile"),
        "severity": corruption.get("severity"),
        "domain_sampling_rate_hz": corruption.get("domain_sampling_rate_hz"),
        "mapping_version": source.get("mapping_version"),
        "mapping_hash": source.get("mapping_hash"),
        "class_order": list(source.get("class_order", ())),
        "artifact_locks_verified": True,
    }
    return tuple(views), identity


def _resolved_output(config: PN2021EvalConfig, output_dir: str | Path | None) -> Path:
    raw = config.payload["output"]["run_dir"] if output_dir is None else output_dir
    output = Path(raw).expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("evaluation outputs must remain outside the Git worktree")
    return output


def _runtime_values(
    config: PN2021EvalConfig,
    spec: ModelSpec,
) -> dict[str, Any]:
    source = config.payload["runtime"]
    requested_device = str(source["device"])
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(requested_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("evaluation device must be CPU or CUDA")
    return {
        "device": str(resolved_device),
        "batch_size": int(source["batch_size_by_model"][spec.name]),
        "num_workers": int(source["num_workers"]),
        "pin_memory": bool(source["pin_memory"]),
        "persistent_workers": bool(source["persistent_workers"]),
        "prefetch_factor": int(source["prefetch_factor"]),
        "cache_mode": str(source["cache_mode"]),
        "validate_values": str(source["validate_values"]),
        "amp_enabled": bool(source["amp"]["enabled"]),
        "amp_dtype": str(source["amp"]["dtype"]),
    }


def build_evaluation_plan(
    *,
    model_spec: ModelSpec,
    checkpoint_path: str | Path,
    config: PN2021EvalConfig,
    output_dir: str | Path | None = None,
    logical_centers: Sequence[str] | None = None,
    subject_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only, no-loader evaluation plan suitable for CLI dry-run."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    views, corruption_identity = load_corruption_views(config)
    runtime = _runtime_values(config, model_spec)
    output = _resolved_output(config, output_dir)
    selected_centers = _resolve_evaluation_centers(config, logical_centers)
    return {
        "schema_version": 3,
        "artifact_type": "pn2021_evaluation_result",
        "profile_name": config.profile_name,
        "config": config.describe(),
        "model": model_spec.describe(),
        "checkpoint": build_artifact_reference(checkpoint),
        "subject": _resolved_subject_identity(
            subject_identity, model_spec=model_spec, centers=selected_centers
        ),
        "output_dir": str(output),
        "runtime": runtime,
        "protocol": {
            "logical_centers": list(selected_centers),
            "configured_logical_centers": list(config.logical_centers),
            "class_order": list(CLASS_ORDER),
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
            "input_pipeline": _resolved_input_pipeline(config, model_spec),
            "canonical_views": list(CLEAN_VIEW_ALIASES + CORRUPTED_VIEW_ALIASES),
            "corruption_cache": corruption_identity,
            "corruption_views": [view.describe() for view in views],
        },
    }


def _hash_id_set(values: Sequence[str]) -> str:
    ordered = sorted(str(value) for value in values)
    payload = "".join(f"{value}\n" for value in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_labels(values: Sequence[str], labels: np.ndarray) -> str:
    rows = sorted(
        (str(hash_id), "".join(str(int(value)) for value in label))
        for hash_id, label in zip(values, labels, strict=True)
    )
    payload = "".join(f"{hash_id}|{label}\n" for hash_id, label in rows).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _loader_identity(
    loader: Any,
    *,
    config: PN2021EvalConfig,
    center: str,
    corrupted: bool,
    view: CorruptionView | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    describe = getattr(loader, "describe", None)
    if not callable(describe):
        raise TypeError("evaluation dataloader must provide describe()")
    identity = describe()
    dataset = _mapping(identity.get("dataset"), description="loader.dataset")
    cache = _mapping(dataset.get("cache"), description="loader.dataset.cache")
    selection = _mapping(
        dataset.get("selection"), description="loader.dataset.selection"
    )
    expected_dataset = "pn2021c" if corrupted else "pn2021"
    if cache.get("dataset") != expected_dataset:
        raise ValueError(f"evaluation loader requires cache dataset {expected_dataset}")
    artifact_locks = config.payload["artifact_locks"]
    expected_cache_manifest_sha256 = artifact_locks[
        "corruption_cache_manifest_sha256"
        if corrupted
        else "clean_cache_manifest_sha256"
    ]
    if cache.get("manifest_sha256") != expected_cache_manifest_sha256:
        raise ValueError(
            f"evaluation {expected_dataset} cache manifest differs from the locked artifact"
        )
    if int(cache.get("sampling_rate_hz", -1)) != CANONICAL_SAMPLING_RATE_HZ:
        raise ValueError("evaluation cache must use the canonical 100 Hz domain")
    if float(cache.get("duration_seconds", -1.0)) != 10.0:
        raise ValueError("evaluation cache must contain ten-second ECGs")
    if tuple(cache.get("lead_order", ())) != EXPECTED_LEADS:
        raise ValueError("evaluation cache lead order mismatch")
    signal_shape = tuple(int(value) for value in cache.get("signal_shape", ()))
    if signal_shape[-2:] != (CANONICAL_POINTS, CANONICAL_CHANNELS):
        raise ValueError(
            "evaluation cache must use canonical (1000,12) time-channel waveforms"
        )
    if cache.get("physical_unit") != "mV" or cache.get("normalization") != "none":
        raise ValueError("evaluation cache must contain unnormalized raw mV signals")
    expected_view_count = 20 if corrupted else 1
    if int(cache.get("view_count", -1)) != expected_view_count:
        raise ValueError("evaluation cache view count mismatch")
    if tuple(cache.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("evaluation cache class order mismatch")
    if (
        cache.get("mapping_version") != PN2021_MAPPING_VERSION
        or cache.get("mapping_hash") != PN2021_MAPPING_HASH
    ):
        raise ValueError("evaluation cache mapping identity mismatch")
    if selection.get("logical_center") != center:
        raise ValueError("evaluation selection logical center mismatch")
    expected_sources = tuple(CENTER_SOURCES[center])
    if tuple(selection.get("source_centers", ())) != expected_sources:
        raise ValueError(
            "evaluation selection physical source centers mismatch: "
            f"center={center}, expected={expected_sources}, "
            f"actual={tuple(selection.get('source_centers', ()))}"
        )
    if selection.get("partition") != "evaluation_all_zero_kept":
        raise ValueError("evaluation must start from the all-zero-kept partition")
    if selection.get("ref_excluded_evaluation") is not True:
        raise ValueError("evaluation selection is not K500-ref-excluded")
    if tuple(selection.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("evaluation selection class order mismatch")
    if (
        selection.get("mapping_version") != PN2021_MAPPING_VERSION
        or selection.get("mapping_hash") != PN2021_MAPPING_HASH
    ):
        raise ValueError("evaluation selection mapping identity mismatch")
    if corrupted:
        if view is None or dataset.get("view") not in {
            view.view_index,
            view.composition_id,
        }:
            raise ValueError("PN2021-C loader did not preserve the explicit view")
    elif dataset.get("view") is not None:
        raise ValueError("clean PN2021 loader unexpectedly declares a corruption view")

    split_path = Path(str(selection.get("split_manifest_path", ""))).resolve()
    split = load_json_mapping(split_path, name="PN2021 split manifest")
    split_manifest_sha256 = sha256_file(split_path)
    if (
        selection.get("split_manifest_sha256") != split_manifest_sha256
        or split_manifest_sha256 != artifact_locks["split_manifest_sha256"]
    ):
        raise ValueError("evaluation split manifest SHA256 mismatch")
    if (
        split.get("schema_version") != 1
        or split.get("dataset") != "pn2021"
        or split.get("split_id") != selection.get("split_id")
        or split.get("source_manifest_sha256")
        != selection.get("source_manifest_sha256")
        or split.get("mapping_version") != PN2021_MAPPING_VERSION
        or split.get("mapping_hash") != PN2021_MAPPING_HASH
        or tuple(split.get("class_order", ())) != CLASS_ORDER
        or tuple(split.get("logical_center_order", ())) != LOGICAL_CENTERS
    ):
        raise ValueError("evaluation split identity is not the locked PN2021 split")
    if selection.get("source_manifest_sha256") != artifact_locks[
        "clean_cache_manifest_sha256"
    ]:
        raise ValueError("evaluation source cache manifest differs from the locked artifact")
    center_entry = _mapping(
        _mapping(split.get("logical_centers"), description="split.logical_centers").get(
            center
        ),
        description=f"split.logical_centers.{center}",
    )
    expected_k = int(config.payload["protocol"]["reference_count_per_center"])
    if int(center_entry.get("k500_count", -1)) != expected_k:
        raise ValueError("evaluation split does not declare the locked K500 exclusion")
    if int(center_entry.get("evaluation_all_zero_kept_count", -1)) != int(
        selection.get("record_count", -2)
    ):
        raise ValueError("loader selection count differs from split manifest")
    if (
        center_entry.get("evaluation_all_zero_kept_hash_id_set_sha256")
        != selection.get("hash_id_set_sha256")
    ):
        raise ValueError("loader selection hash differs from split manifest")
    reference_identity = {
        "excluded_reference_count": expected_k,
        "k500_hash_id_set_sha256": center_entry.get("k500_hash_id_set_sha256"),
        "all_zero_kept_count": center_entry.get("evaluation_all_zero_kept_count"),
        "all_zero_kept_hash_id_set_sha256": center_entry.get(
            "evaluation_all_zero_kept_hash_id_set_sha256"
        ),
        "drop_all_zero_count": center_entry.get("evaluation_drop_all_zero_count"),
        "drop_all_zero_hash_id_set_sha256": center_entry.get(
            "evaluation_drop_all_zero_hash_id_set_sha256"
        ),
    }
    return identity, reference_identity


def _strings(value: Any, *, batch_size: int, description: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{description} must contain one string per record")
    values = [str(item) for item in value]
    if len(values) != batch_size:
        raise ValueError(f"{description} length does not match batch size")
    return values


def _evaluate_loader(
    model: nn.Module,
    loader: Any,
    *,
    config: PN2021EvalConfig,
    center: str,
    spec: ModelSpec,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    corrupted: bool,
    view: CorruptionView | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    loader_description, reference_identity = _loader_identity(
        loader,
        config=config,
        center=center,
        corrupted=corrupted,
        view=view,
    )
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    hash_ids: list[str] = []
    record_ids: list[str] = []
    cache_indices: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("evaluation batches must be mappings")
            waveforms = batch.get("waveform")
            targets = batch.get("label")
            if not isinstance(waveforms, torch.Tensor) or not isinstance(
                targets, torch.Tensor
            ):
                raise TypeError("evaluation batch waveform/label must be tensors")
            if (
                waveforms.ndim != 3
                or waveforms.shape[0] < 1
                or tuple(waveforms.shape[1:])
                != (CANONICAL_POINTS, CANONICAL_CHANNELS)
            ):
                raise ValueError(
                    "evaluation waveform must have canonical raw shape "
                    "(B,1000,12) in time-channel layout"
                )
            if not waveforms.is_floating_point():
                raise TypeError("evaluation waveform must use a floating dtype")
            if targets.ndim != 2 or tuple(targets.shape) != (
                waveforms.shape[0],
                len(CLASS_ORDER),
            ):
                raise ValueError("evaluation labels must have shape (B,5)")
            if not torch.isfinite(targets).all() or not torch.all(
                (targets == 0) | (targets == 1)
            ):
                raise ValueError("evaluation labels must be finite binary values")
            batch_size = int(waveforms.shape[0])
            rates = batch.get("sampling_rate_hz")
            if not isinstance(rates, torch.Tensor) or not bool(
                torch.all(rates == CANONICAL_SAMPLING_RATE_HZ)
            ):
                raise ValueError("evaluation batch must declare canonical 100 Hz")
            batch_hashes = _strings(
                batch.get("hash_id"), batch_size=batch_size, description="hash_id"
            )
            batch_records = _strings(
                batch.get("record_id"), batch_size=batch_size, description="record_id"
            )
            if corrupted:
                composition_ids = _strings(
                    batch.get("composition_id"),
                    batch_size=batch_size,
                    description="composition_id",
                )
                assert view is not None
                if set(composition_ids) != {view.composition_id}:
                    raise ValueError("corrupted batch composition identity mismatch")
            raw_indices = batch.get("cache_index")
            if not isinstance(raw_indices, torch.Tensor) or raw_indices.numel() != batch_size:
                raise ValueError("cache_index must contain one value per record")
            raw_100hz_btc = waveforms.to(device, non_blocking=True)
            inputs = prepare_canonical_model_input(
                raw_100hz_btc,
                spec,
                epsilon=float(
                    config.payload["protocol"]["input_pipeline"]["normalization"][
                        "epsilon"
                    ]
                ),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled and device.type == "cuda",
            ):
                outputs = model(inputs)
            outputs = validate_model_output(outputs, spec, batch_size=batch_size)
            logits_parts.append(outputs.float().cpu().numpy())
            target_parts.append(targets.to(dtype=torch.uint8).cpu().numpy())
            hash_ids.extend(batch_hashes)
            record_ids.extend(batch_records)
            cache_indices.extend(int(value) for value in raw_indices.reshape(-1).tolist())
    if not logits_parts:
        raise ValueError("evaluation loader produced no records")
    logits = np.concatenate(logits_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    if len(set(hash_ids)) != len(hash_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("evaluation loader produced duplicate record identities")
    metrics = compute_metric_views(
        logits,
        targets,
        hash_ids,
        corrupted=corrupted,
    )
    aliases = CORRUPTED_VIEW_ALIASES if corrupted else CLEAN_VIEW_ALIASES
    expected_hashes = (
        reference_identity["all_zero_kept_hash_id_set_sha256"],
        reference_identity["drop_all_zero_hash_id_set_sha256"],
    )
    slice_name = f"depth{view.depth}" if corrupted and view is not None else "clean"
    for alias, expected_hash in zip(aliases, expected_hashes, strict=True):
        payload = metrics[alias]
        if payload["record_identity"]["hash_id_set_sha256"] != expected_hash:
            raise ValueError(f"computed {alias} record identity differs from split")
        payload["evaluation_slice"] = slice_name
        payload["excluded_reference_count"] = reference_identity[
            "excluded_reference_count"
        ]
    identity = {
        "loader": loader_description,
        "reference_exclusion": reference_identity,
        "evaluated_sample_count": int(targets.shape[0]),
        "ordered_hash_ids_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in hash_ids).encode("utf-8")
        ).hexdigest(),
        "hash_id_set_sha256": _hash_id_set(hash_ids),
        "hash_label_set_sha256": _hash_labels(hash_ids, targets),
        "ordered_record_ids_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in record_ids).encode("utf-8")
        ).hexdigest(),
        "cache_indices_sha256": hashlib.sha256(
            np.asarray(cache_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return metrics, identity


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _center_aggregation_fields(
    evaluated_center_mean: Mapping[str, Mapping[str, Any]],
    *,
    center_order: Sequence[str],
) -> dict[str, Any]:
    centers = tuple(str(value) for value in center_order)
    is_exact_four_center = centers == LOGICAL_CENTERS
    return {
        "center_count": len(centers),
        "center_order": list(centers),
        "evaluated_center_mean": evaluated_center_mean,
        "four_center_mean": (
            evaluated_center_mean if is_exact_four_center else None
        ),
    }


def _resolved_subject_identity(
    value: Mapping[str, Any] | None,
    *,
    model_spec: ModelSpec,
    centers: Sequence[str],
) -> dict[str, Any] | None:
    """Bind a managed subject to the model and exact evaluation-center order."""

    if value is None:  # Direct library calls remain usable for unit diagnostics.
        return None
    subject = _mapping(value, description="evaluation subject")
    _exact_keys(
        subject,
        expected={"mode", "train_result", "lineage"},
        description="evaluation subject",
    )
    mode = subject["mode"]
    if mode not in {
        "prospective_train_result", "legacy_center_adapted", "source_registry"
    }:
        raise ValueError("unsupported evaluation subject mode")
    lineage = _mapping(subject["lineage"], description="evaluation subject.lineage")
    if lineage.get("model") != {"name": model_spec.name, "spec": model_spec.describe()}:
        raise ValueError("evaluation subject model differs from evaluated model")
    selected = tuple(str(center) for center in centers)
    if mode == "source_registry":
        if subject["train_result"] is not None or selected != LOGICAL_CENTERS:
            raise ValueError("source registry subject requires the canonical four centers")
    else:
        reference = _mapping(
            subject["train_result"], description="evaluation subject.train_result"
        )
        _exact_keys(
            reference,
            expected={"path", "sha256"},
            description="evaluation subject.train_result",
        )
        if selected != (lineage.get("center"),):
            raise ValueError("adapted evaluation subject requires its single target center")
    return dict(subject)


def evaluate_pn2021(
    model: nn.Module,
    checkpoint_identity: CheckpointIdentity,
    *,
    config: PN2021EvalConfig | None = None,
    config_path: str | Path = DEFAULT_PN2021_EVAL_CONFIG,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    logical_centers: Sequence[str] | None = None,
    subject_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one strict task checkpoint on clean and 20 PN2021-C views."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(checkpoint_identity, CheckpointIdentity):
        raise TypeError("checkpoint_identity must be a CheckpointIdentity")
    if checkpoint_identity.missing_keys or checkpoint_identity.unexpected_keys:
        raise ValueError("evaluation requires a strictly loaded task checkpoint")
    if sha256_file(checkpoint_identity.path) != checkpoint_identity.sha256:
        raise ValueError("checkpoint file changed after it was loaded")
    resolved_config = (
        load_pn2021_eval_config(config_path, config_root=config_root)
        if config is None
        else config
    )
    spec = getattr(model, "model_spec", None)
    if not isinstance(spec, ModelSpec):
        raise ValueError("evaluation model must expose a canonical ModelSpec")
    views, corruption_identity = load_corruption_views(resolved_config)
    runtime = _runtime_values(resolved_config, spec)
    selected_centers = _resolve_evaluation_centers(
        resolved_config, logical_centers
    )
    subject = _resolved_subject_identity(
        subject_identity, model_spec=spec, centers=selected_centers
    )
    resolved_device = torch.device(runtime["device"])
    torch_amp_dtype = (
        torch.bfloat16 if runtime["amp_dtype"] == "bfloat16" else torch.float16
    )
    output = _resolved_output(resolved_config, output_dir)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / str(resolved_config.payload["output"]["result_file"])
    seed = seed_process(
        "pn2021_evaluation",
        resolved_config.profile_name,
        spec.name,
        checkpoint_identity.sha256,
        config_path=resolved_config.random_seed_config_path,
    )
    loader_plan = PN2021EvaluationLoaderPlan(
        clean_partition=resolved_config.payload["protocol"]["partitions"]["clean"],
        corrupted_partition=resolved_config.payload["protocol"]["partitions"][
            "corrupted"
        ],
        batch_size=runtime["batch_size"],
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        persistent_workers=runtime["persistent_workers"],
        prefetch_factor=runtime["prefetch_factor"],
        cache_mode=runtime["cache_mode"],
        validate_values=runtime["validate_values"],
        split_config_path=resolved_config.split_config_path,
        data_load_config_path=resolved_config.data_load_config_path,
        corruption_cache_config_path=resolved_config.corruption_cache_config_path,
        seed_config_path=resolved_config.random_seed_config_path,
    )
    was_training = model.training
    model.to(resolved_device)
    model.eval()
    clean_centers: dict[str, Any] = {}
    clean_record_identity: dict[str, dict[str, str]] = {}
    with ExitStack() as stack:
        stack.callback(model.train, was_training)
        data_session = stack.enter_context(loader_plan.open_session())
        for center in selected_centers:
            loader = data_session.open_clean(center)
            try:
                metrics, identity = _evaluate_loader(
                    model,
                    loader,
                    config=resolved_config,
                    center=center,
                    spec=spec,
                    device=resolved_device,
                    amp_enabled=runtime["amp_enabled"],
                    amp_dtype=torch_amp_dtype,
                    corrupted=False,
                    view=None,
                )
            finally:
                close = getattr(loader, "close", None)
                if callable(close):
                    close()
            clean_centers[center] = {"metrics": metrics, "identity": identity}
            clean_record_identity[center] = {
                "hash_id_set_sha256": identity["hash_id_set_sha256"],
                "hash_label_set_sha256": identity["hash_label_set_sha256"],
            }

        corrupted_views: list[dict[str, Any]] = []
        for view in views:
            per_center: dict[str, Any] = {}
            for center in selected_centers:
                loader = data_session.open_corrupted(center, view.composition_id)
                try:
                    metrics, identity = _evaluate_loader(
                        model,
                        loader,
                        config=resolved_config,
                        center=center,
                        spec=spec,
                        device=resolved_device,
                        amp_enabled=runtime["amp_enabled"],
                        amp_dtype=torch_amp_dtype,
                        corrupted=True,
                        view=view,
                    )
                finally:
                    close = getattr(loader, "close", None)
                    if callable(close):
                        close()
                expected = clean_record_identity[center]
                if (
                    identity["hash_id_set_sha256"] != expected["hash_id_set_sha256"]
                    or identity["hash_label_set_sha256"]
                    != expected["hash_label_set_sha256"]
                ):
                    raise ValueError(
                        "clean and PN2021-C evaluation records/labels are not identical"
                    )
                per_center[center] = {"metrics": metrics, "identity": identity}
            evaluated_center_mean = mean_metric_views(
                [per_center[center]["metrics"] for center in selected_centers]
            )
            for payload in evaluated_center_mean.values():
                payload["evaluation_slice"] = f"depth{view.depth}"
            corrupted_views.append(
                {
                    **view.describe(),
                    "evaluation_slice": f"depth{view.depth}",
                    "per_center": per_center,
                    **_center_aggregation_fields(
                        evaluated_center_mean,
                        center_order=selected_centers,
                    ),
                }
            )
    clean_evaluated_center_mean = mean_metric_views(
        [clean_centers[center]["metrics"] for center in selected_centers]
    )
    for payload in clean_evaluated_center_mean.values():
        payload["evaluation_slice"] = "clean"
    corruption_aggregates = aggregate_corruption_views(
        corrupted_views,
        center_order=selected_centers,
        canonical_four_center_order=LOGICAL_CENTERS,
    )
    result = {
        "schema_version": 3,
        "artifact_type": "pn2021_evaluation_result",
        "status": "complete",
        "evaluation_profile": resolved_config.profile_name,
        "model": spec.describe(),
        "checkpoint": checkpoint_identity.describe(),
        "subject": subject,
        "config": resolved_config.describe(),
        "seed": seed.describe(),
        "runtime": runtime,
        "protocol": {
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
            "class_order": list(CLASS_ORDER),
            "logical_centers": list(selected_centers),
            "configured_logical_centers": list(resolved_config.logical_centers),
            "reference_count_per_center": 500,
            "input_pipeline": _resolved_input_pipeline(resolved_config, spec),
            "canonical_views": list(CLEAN_VIEW_ALIASES + CORRUPTED_VIEW_ALIASES),
            "corruption_cache": corruption_identity,
            "center_aggregation": {
                "primary_field": "evaluated_center_mean",
                "compatibility_field": "four_center_mean",
                "compatibility_policy": (
                    "legacy schema-v1 four_center_mean is read only as a generic "
                    "evaluated-center mean unless four locked centers are verified"
                ),
            },
        },
        "clean": {
            "evaluation_slice": "clean",
            "per_center": clean_centers,
            **_center_aggregation_fields(
                clean_evaluated_center_mean,
                center_order=selected_centers,
            ),
        },
        "corrupted": {
            "per_view": corrupted_views,
            "aggregates": corruption_aggregates,
        },
        "output": {"directory": str(output), "result_file": str(result_path)},
    }
    validate_evaluation_aggregates(result)
    _atomic_json(result_path, result)
    return result


__all__ = [
    "DEFAULT_PN2021_EVAL_CONFIG",
    "LOGICAL_CENTERS",
    "PN2021_MAPPING_HASH",
    "PN2021_MAPPING_VERSION",
    "CorruptionView",
    "PN2021EvalConfig",
    "build_evaluation_plan",
    "evaluate_pn2021",
    "load_corruption_views",
    "load_pn2021_eval_config",
]
