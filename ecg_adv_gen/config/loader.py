"""YAML config loading, validation, and legacy-command dry runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import jsonschema
except Exception:  # pragma: no cover - jsonschema exists in the current env.
    jsonschema = None

from .paths import PathSafetyError, is_under, validate_local_paths
from .entrypoints import managed_runner_script_names, managed_script_profile
from .adapters.registry import build_runner_adapter_argv
from .artifact_trace import (
    append_k500_ref as _append_k500_ref,
    dedupe_path_records as _dedupe_path_records,
    path_record as _path_record,
    strip_known_suffix as _strip_known_suffix,
)
from .runner_audit import DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES, audit_runner_command
from ecg_adv_gen.data import DataContractError, validate_data_preprocess_config
from ecg_adv_gen.data.kshot_artifacts import canonical_kshot_base
from ecg_adv_gen.evaluation import (
    SelectionPolicyError,
    has_forbidden_selection_reference,
    validate_selection_policy,
)
from ecg_adv_gen.labels import Super5MetadataError, validate_super5_metadata
from ecg_adv_gen.run_naming import (
    build_ecgfounder_fullft_run_leaf,
    build_effnet_direct_run_leaf,
    build_effnet_vae_lhat_run_leaf,
)


class ConfigError(ValueError):
    """Raised when a run config is incomplete or paper-unsafe."""


_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_CLI_OVERRIDE_KEY_RE = re.compile(r"^[A-Za-z0-9_.]+$")

ALLOWED_CLI_OVERRIDE_KEYS = frozenset(
    {
        "resources.default_num_workers",
        "training.epochs",
        "training.batch_size",
        "training.eval_batch_size",
        "training.num_workers",
        "training.optimizer.lr",
        "training.optimizer.weight_decay",
        "training.grad_clip",
        "training.pos_weight_clip_max",
        "adaptation.hull.M",
        "adaptation.hull.lambda",
        "adaptation.hull.steps",
        "adaptation.hull.lr",
        "adaptation.hull.neighbor_pool_size",
        "adaptation.hull.neighbor_pool_multiplier",
        "adaptation.anchors.k_anchor",
        "adaptation.anchors.sample_power",
        "adaptation.anchors.sample_min_weight",
        "adaptation.loss.target_real_weight",
        "adaptation.loss.adv_weight",
        "adaptation.loss.adv_weight_warmup_epochs",
        "adaptation.loss.teacher_mix",
        "adaptation.latent_augmix.latent_weight_cap",
        "adaptation.latent_augmix.width",
        "adaptation.latent_augmix.depth",
        "adaptation.latent_augmix.alpha",
        "adaptation.latent_augmix.severity",
    }
)

ALLOWED_LOCAL_CONFIG_TOP_LEVEL_KEYS = frozenset(
    {
        "host",
        "paths",
        "python",
        "resources",
        "safety",
        "external_models",
    }
)

MANAGED_RUNNER_SCRIPT_NAMES = managed_runner_script_names()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


def _validate_local_config_overlay(local_raw: dict[str, Any]) -> None:
    disallowed = sorted(
        key
        for key in local_raw
        if not str(key).startswith("_") and key not in ALLOWED_LOCAL_CONFIG_TOP_LEVEL_KEYS
    )
    if disallowed:
        allowed = ", ".join(sorted(ALLOWED_LOCAL_CONFIG_TOP_LEVEL_KEYS))
        raise ConfigError(
            "Local config may only define host, paths, python, resources, and safety; "
            f"disallowed top-level keys: {disallowed}. Allowed keys: {allowed}"
        )


def normalize_pipeline_stages(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML-managed stage dependencies into a manifest contract."""

    raw_stages = config.get("stages") or []
    if not raw_stages:
        return {"schema_version": 1, "stages": []}
    if not isinstance(raw_stages, list):
        raise ConfigError("stages must be a list")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise ConfigError("stages[] entries must be mappings")
        name = raw_stage.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("stages[].name must be a non-empty string")
        name = name.strip()
        if name in seen:
            raise ConfigError(f"duplicate stage name: {name}")

        def normalize_string_list(key: str) -> list[str]:
            values = raw_stage.get(key, [])
            if values is None:
                return []
            if not isinstance(values, list):
                raise ConfigError(f"stages[].{key} must be a list")
            out: list[str] = []
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(f"stages[].{key} entries must be non-empty strings")
                out.append(value.strip())
            return out

        requires = normalize_string_list("requires")
        for dependency in requires:
            if dependency not in seen:
                raise ConfigError(f"stage {name} has unknown required stage: {dependency}")

        normalized.append(
            {
                "index": idx,
                "name": name,
                "requires": requires,
                "produces": normalize_string_list("produces"),
                "skip_if_exists": normalize_string_list("skip_if_exists"),
            }
        )
        seen.add(name)

    return {"schema_version": 1, "stages": normalized}


def _load_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ConfigError(f"Config extends cycle detected at {path}")
    seen.add(path)

    data = _read_yaml(path)
    extends = data.pop("extends", [])
    if isinstance(extends, (str, os.PathLike)):
        extends = [extends]
    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = (path.parent / str(parent)).resolve()
        merged = _deep_merge(merged, _load_with_extends(parent_path, seen))
    merged = _deep_merge(merged, data)
    merged.setdefault("_config_sources", []).append(str(path))
    return merged


def _lookup(config: dict[str, Any], key: str) -> Any:
    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Unknown config interpolation key: {key}")
        cur = cur[part]
    return cur


def _interpolate_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v, context) for v in value]
    if not isinstance(value, str):
        return value

    match = _VAR_RE.fullmatch(value)
    if match:
        key = match.group(1)
        try:
            return _lookup(context, key)
        except ConfigError:
            if key.startswith(("matrix.", "runtime.")):
                return value
            raise

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        try:
            return str(_lookup(context, key))
        except ConfigError:
            if key.startswith(("matrix.", "runtime.")):
                return m.group(0)
            raise

    return _VAR_RE.sub(repl, value)


def interpolate_config(config: dict[str, Any]) -> dict[str, Any]:
    previous = copy.deepcopy(config)
    for _ in range(8):
        current = _interpolate_value(previous, previous)
        if current == previous:
            return current
        previous = current
    raise ConfigError("Config interpolation did not converge")


def _load_schema(schema_path: Path) -> dict[str, Any] | None:
    if not schema_path.exists():
        return None
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_json_schema(config: dict[str, Any], schema_path: Path) -> None:
    if jsonschema is None:
        return
    schema = _load_schema(schema_path)
    if schema is not None:
        try:
            jsonschema.validate(config, schema)
        except jsonschema.exceptions.ValidationError as exc:
            raise ConfigError(f"Schema validation failed for {schema_path.name}: {exc.message}") from exc


def _configs_root_for(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if parent.name == "configs":
            return parent
    return config_path.parent.parent


def _scan_for_forbidden_local_paths(obj: Any, *, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("_"):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            hits.extend(_scan_for_forbidden_local_paths(value, prefix=child))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(_scan_for_forbidden_local_paths(value, prefix=f"{prefix}[{idx}]"))
    elif isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith("/root/") or stripped.startswith("/home/linbinhao/"):
            hits.append(f"{prefix}: {stripped}")
    return hits


def _parse_cli_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ConfigError(f"Bad --set override {raw!r}; expected key=value")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key or not _CLI_OVERRIDE_KEY_RE.fullmatch(key):
        raise ConfigError(f"Bad --set key {key!r}")
    if key not in ALLOWED_CLI_OVERRIDE_KEYS:
        raise ConfigError(
            f"--set override for {key!r} is not allowed; "
            "only whitelisted resource/training/method hyperparameters may be overridden"
        )
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Bad --set value for {key!r}: {raw_value!r}") from exc
    if value is None:
        raise ConfigError(f"--set value for {key!r} must not be null")
    return key, value


def _type_name(value: Any) -> str:
    return type(value).__name__


def _validate_override_type(key: str, old_value: Any, new_value: Any) -> None:
    if isinstance(old_value, str) and "${" in old_value:
        return
    if isinstance(old_value, bool):
        if not isinstance(new_value, bool):
            raise ConfigError(f"--set {key} expects bool, got {_type_name(new_value)}")
        return
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        if not isinstance(new_value, int) or isinstance(new_value, bool):
            raise ConfigError(f"--set {key} expects int, got {_type_name(new_value)}")
        return
    if isinstance(old_value, float):
        if not isinstance(new_value, (int, float)) or isinstance(new_value, bool):
            raise ConfigError(f"--set {key} expects float, got {_type_name(new_value)}")
        return
    if isinstance(old_value, str):
        if not isinstance(new_value, str):
            raise ConfigError(f"--set {key} expects string, got {_type_name(new_value)}")


def apply_cli_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply audited CLI overrides before interpolation.

    The whitelist intentionally excludes paper protocol, K-shot, centers, paths,
    checkpoints, and runner argv so ad-hoc launches cannot silently change the
    experimental contract.
    """
    if not overrides:
        return config
    out = copy.deepcopy(config)
    records: list[dict[str, Any]] = []
    for raw in overrides:
        key, value = _parse_cli_override(raw)
        cur: Any = out
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(cur, dict) or part not in cur:
                raise ConfigError(f"--set key does not exist before interpolation: {key}")
            cur = cur[part]
        last = parts[-1]
        if not isinstance(cur, dict) or last not in cur:
            raise ConfigError(f"--set key does not exist before interpolation: {key}")
        _validate_override_type(key, cur[last], value)
        cur[last] = value
        records.append({"key": key, "value": value, "raw": raw})
    out["_cli_overrides"] = records
    return out


def load_experiment_config(
    config_path: Path,
    local_config_path: Path,
    *,
    overrides: list[str] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    local_config_path = local_config_path.resolve()
    experiment_raw = _load_with_extends(config_path)
    local_raw = _load_with_extends(local_config_path)
    configs_root = _configs_root_for(config_path)
    _validate_local_config_overlay(local_raw)
    _validate_json_schema(local_raw, configs_root / "schemas" / "local_config.schema.json")

    forbidden_hits = _scan_for_forbidden_local_paths(experiment_raw)
    if forbidden_hits:
        joined = "\n".join(f"  - {hit}" for hit in forbidden_hits)
        raise ConfigError(
            "Tracked experiment config contains local/root absolute paths:\n" + joined
        )

    merged = _deep_merge(experiment_raw, local_raw)
    merged = apply_cli_overrides(merged, overrides)
    if runtime_context:
        runtime = copy.deepcopy(merged.get("runtime") or {})
        runtime.update(copy.deepcopy(runtime_context))
        merged["runtime"] = runtime
    merged = interpolate_config(merged)
    merged["_entry_config"] = str(config_path)
    merged["_local_config"] = str(local_config_path)
    return merged


def _require(config: dict[str, Any], key: str) -> Any:
    value = _lookup(config, key)
    if value in (None, "", []):
        raise ConfigError(f"Missing required config key: {key}")
    return value


def _validate_mapping(config: dict[str, Any]) -> None:
    mapping_version = _require(config, "paper_protocol.mapping_version")
    mapping_hash = _require(config, "paper_protocol.mapping_hash")
    class_order = _require(config, "paper_protocol.class_order")
    num_classes = _require(config, "paper_protocol.num_classes")

    try:
        validate_super5_metadata(
            mapping_version=str(mapping_version),
            mapping_hash=str(mapping_hash),
            class_order=class_order,
            num_classes=int(num_classes),
        )
    except Super5MetadataError as exc:
        raise ConfigError(str(exc)) from exc


def validate_experiment_config(config: dict[str, Any], *, repo_root: Path) -> dict[str, str]:
    schema_path = repo_root / "configs" / "schemas" / "experiment_config.schema.json"
    _validate_json_schema(config, schema_path)

    required = [
        "experiment.name",
        "runner.entrypoint",
        "paper_protocol.mapping_version",
        "paper_protocol.mapping_hash",
        "paper_protocol.class_order",
        "paper_protocol.num_classes",
        "paper_protocol.kshot.mode",
        "paper_protocol.kshot.k",
        "paper_protocol.kshot.seed",
        "paper_protocol.kshot.exclude_refs_from_eval",
        "paper_protocol.selection.policy",
        "paper_protocol.selection.forbid_heldout_target_labels",
        "paper_protocol.selection.forbid_full_target_distribution_tuning",
        "data.source_dataset",
        "data.target_dataset",
        "data.pn2021_cache_version",
        "preprocess.classifier_fs",
        "preprocess.classifier_len",
        "preprocess.lead_order",
        "model.backbone",
        "model.num_classes",
        "evaluation.views",
    ]
    for key in required:
        _require(config, key)

    if _lookup(config, "paper_protocol.kshot.exclude_refs_from_eval") is not True:
        raise ConfigError("paper_protocol.kshot.exclude_refs_from_eval must be true")
    try:
        validate_selection_policy(config["paper_protocol"]["selection"])
    except SelectionPolicyError as exc:
        raise ConfigError(str(exc)) from exc

    _validate_mapping(config)
    try:
        validate_data_preprocess_config(config)
    except DataContractError as exc:
        raise ConfigError(str(exc)) from exc
    normalize_pipeline_stages(config)
    return validate_local_paths(config)


def _matrix_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = (config.get("runner") or {}).get("matrix") or {}
    if not matrix:
        return [{"matrix": {}}]
    keys = list(matrix.keys())
    value_lists = []
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"runner.matrix.{key} must be a non-empty list")
        value_lists.append(values)
    contexts = []
    for combo in itertools.product(*value_lists):
        contexts.append({"matrix": dict(zip(keys, combo))})
    return contexts


def _flatten_argv(argv: list[Any], context: dict[str, Any]) -> list[str]:
    resolved = _interpolate_value(argv, context)
    out: list[str] = []
    for item in resolved:
        if item is None:
            continue
        if isinstance(item, list):
            out.extend(str(v) for v in item)
        elif isinstance(item, bool):
            if item:
                continue
        else:
            out.append(str(item))
    return out


def build_runner_commands(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = validate_local_paths(config)
    runner = config.get("runner") or {}
    adapter_name = runner.get("adapter")
    if not adapter_name:
        raise ConfigError("runner.adapter is required for managed YAML launches")
    if "argv" in runner:
        raise ConfigError("runner.adapter does not accept runner.argv; use typed YAML fields")
    entrypoint = Path(str(runner["entrypoint"]))
    if entrypoint.is_absolute():
        raise ConfigError("runner.entrypoint must be repo-relative")
    script_path = Path(paths["project_root"]) / entrypoint
    if not script_path.exists():
        raise ConfigError(f"runner.entrypoint does not exist: {script_path}")

    commands = []
    for matrix_ctx in _matrix_contexts(config):
        context = copy.deepcopy(config)
        context.update(matrix_ctx)
        try:
            adapter_argv = build_runner_adapter_argv(
                str(adapter_name),
                config=config,
                context=context,
            )
        except (KeyError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        argv = _flatten_argv(adapter_argv, context)
        command = [paths["python_executable"], str(script_path), *argv]
        env = _interpolate_value(runner.get("env") or {}, context)
        commands.append(
            {
                "name": _interpolate_value(runner.get("name", config["experiment"]["name"]), context),
                "matrix": matrix_ctx["matrix"],
                "cwd": paths["project_root"],
                "env": env,
                "argv": command,
            }
        )
    audit_runner_commands(config, commands)
    return commands


def _expected_artifact_records_from_spec(
    spec: dict[str, Any],
    *,
    context: dict[str, Any],
    boundary: Path,
) -> list[dict[str, Any]]:
    records = []
    for item in _interpolate_value(spec.get("expected_artifacts") or [], context):
        if not isinstance(item, dict):
            raise ConfigError("postprocess.commands[].expected_artifacts entries must be mappings")
        role = str(item.get("role") or "")
        path = str(item.get("path") or "")
        if not role or not path:
            raise ConfigError("postprocess expected artifact entries require role and path")
        resolved = Path(path).expanduser().resolve(strict=False)
        if resolved.is_absolute() and not is_under(resolved, boundary):
            raise ConfigError(f"postprocess expected artifact {role} is outside write boundary: {resolved}")
        records.append(_path_record(role, resolved, required=bool(item.get("required", True))))
    return records


def build_postprocess_commands(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build YAML-managed CPU postprocess commands."""
    specs = ((config.get("postprocess") or {}).get("commands") or [])
    if not specs:
        return []
    if not isinstance(specs, list):
        raise ConfigError("postprocess.commands must be a list")

    paths = validate_local_paths(config)
    boundary = Path(str(config.get("safety", {}).get("write_boundary", "/home/linbinhao"))).resolve()
    commands = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ConfigError("postprocess.commands entries must be mappings")
        entrypoint = Path(str(spec.get("entrypoint") or ""))
        if not entrypoint:
            raise ConfigError("postprocess.commands[].entrypoint is required")
        if entrypoint.is_absolute():
            raise ConfigError("postprocess.commands[].entrypoint must be repo-relative")
        script_path = Path(paths["project_root"]) / entrypoint
        if not script_path.exists():
            raise ConfigError(f"postprocess entrypoint does not exist: {script_path}")
        context = copy.deepcopy(config)
        argv = _flatten_argv(spec.get("argv") or [], context)
        env = _interpolate_value(spec.get("env") or {}, context)
        command = {
            "name": _interpolate_value(spec.get("name", f"postprocess_{idx:02d}"), context),
            "cwd": paths["project_root"],
            "env": env,
            "argv": [paths["python_executable"], str(script_path), *argv],
            "expected_artifacts": _expected_artifact_records_from_spec(
                spec,
                context=context,
                boundary=boundary,
            ),
        }
        commands.append(command)
    audit_postprocess_commands(config, commands)
    return commands


def _argv_option_map(argv: list[str]) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    idx = 2  # argv[0] is Python, argv[1] is the legacy entrypoint.
    while idx < len(argv):
        token = argv[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        key = token
        idx += 1
        values: list[str] = []
        while idx < len(argv) and not argv[idx].startswith("--"):
            values.append(argv[idx])
            idx += 1
        if not values:
            opts[key] = True
        elif len(values) == 1:
            opts[key] = values[0]
        else:
            opts[key] = values
    return opts


def _opt_first(opts: dict[str, Any], name: str, default: Any = None) -> Any:
    value = opts.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _opt_list(opts: dict[str, Any], name: str) -> list[str]:
    value = opts.get(name)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is True:
        return []
    return [str(value)]


def _audit_require_options(errors: list[str], script: str, opts: dict[str, Any], options: list[str]) -> None:
    for option in options:
        if option not in opts:
            errors.append(f"{script}: missing required option {option}")


def _audit_equals(errors: list[str], script: str, opts: dict[str, Any], option: str, expected: Any) -> None:
    value = _opt_first(opts, option)
    if str(value) != str(expected):
        errors.append(f"{script}: {option}={value!r}, expected {expected!r}")


def _matrix_case(command: dict[str, Any]) -> dict[str, Any]:
    case = (command.get("matrix") or {}).get("case")
    return case if isinstance(case, dict) else {}


def _audit_no_forbidden_substrings(errors: list[str], script: str, argv: list[str], forbidden: list[str]) -> None:
    joined = " ".join(str(x) for x in argv)
    for token in forbidden:
        if token.startswith("/"):
            if any(str(arg).startswith(token) for arg in argv):
                errors.append(f"{script}: command contains forbidden root-era path {token!r}")
            continue
        if token in joined:
            errors.append(f"{script}: command contains forbidden token {token!r}")


_PATH_OPTION_NAMES = {
    "--anchor_base",
    "--anchor_base_root",
    "--cache_path",
    "--cache_dir",
    "--class_trust",
    "--csv_path",
    "--clean_cache_dir",
    "--clean_eval_json",
    "--clean_mmap_cache_dir",
    "--data_dir",
    "--data_path",
    "--data_root",
    "--ecgtwin_config",
    "--input",
    "--input_dir",
    "--input_dirs",
    "--gated_dirs",
    "--ibe_path",
    "--exclude_ref_ids",
    "--checkpoint",
    "--init_ckpt",
    "--init_base_head_from_k500_root",
    "--init_head_path",
    "--linear_probe_dir",
    "--metrics-long",
    "--model_dir",
    "--out_dir",
    "--out_root",
    "--output_dir",
    "--output-dir",
    "--output_path",
    "--pn2021_cache_dir",
    "--pn2021_mmap_cache_dir",
    "--pn2021_root",
    "--ptbxl_cache",
    "--ptbxl_csv",
    "--ptbxl_prep",
    "--ptbxl_raw",
    "--ptbxl_vae_cache",
    "--python",
    "--ref_root",
    "--ref_meta_json",
    "--split_json",
    "--save_dir",
    "--semantic_ckpt",
    "--style_ckpt",
    "--synth_npz",
    "--target_logit_anchor_path",
    "--target_real_npz",
    "--token_bank",
    "--train_path",
    "--val_path",
    "--resume_path",
    "--victim_ckpt",
    "--prompt_bank",
}

_OUTPUT_PATH_OPTION_NAMES = {
    "--out_dir",
    "--out_root",
    "--output_dir",
    "--output-dir",
    "--output_path",
    "--save_dir",
}

_ENV_PATH_KEYS = {
    "DEEPECG_NOTEBOOKS",
    "ECG_ADV_GEN_DATA_ROOT",
    "ECGFOUNDER_ROOT",
    "TMPDIR",
    "XDG_CACHE_HOME",
}


def _audit_runtime_path(
    errors: list[str],
    *,
    script: str,
    label: str,
    value: Any,
    boundary: Path,
) -> None:
    if value in (None, "", True, False):
        return
    values = value if isinstance(value, list) else [value]
    for raw in values:
        text = str(raw).strip()
        if not text or not text.startswith("/"):
            continue
        if text.startswith("/root/"):
            errors.append(f"{script}: {label} points at root-era runtime path {text!r}")
            continue
        try:
            resolved = Path(os.path.abspath(os.path.expanduser(text)))
        except OSError:
            errors.append(f"{script}: {label} is not a valid path {text!r}")
            continue
        if not is_under(resolved, boundary):
            errors.append(f"{script}: {label}={resolved} is outside write boundary {boundary}")


def _audit_command_path_safety(
    errors: list[str],
    script: str,
    command: dict[str, Any],
    opts: dict[str, Any],
    *,
    boundary: Path,
) -> None:
    env = command.get("env") or {}
    if "CUDA_VISIBLE_DEVICES" in env:
        errors.append(f"{script}: runner env must not set CUDA_VISIBLE_DEVICES")

    _audit_runtime_path(errors, script=script, label="cwd", value=command.get("cwd"), boundary=boundary)
    argv = command.get("argv") or []
    if len(argv) > 0:
        _audit_runtime_path(errors, script=script, label="python", value=argv[0], boundary=boundary)
    if len(argv) > 1:
        _audit_runtime_path(errors, script=script, label="entrypoint", value=argv[1], boundary=boundary)

    for key in _ENV_PATH_KEYS:
        _audit_runtime_path(errors, script=script, label=f"env.{key}", value=env.get(key), boundary=boundary)
    for option in _PATH_OPTION_NAMES:
        if option in opts:
            _audit_runtime_path(errors, script=script, label=option, value=opts[option], boundary=boundary)


def _audit_output_paths_are_run_scoped(
    errors: list[str],
    script: str,
    opts: dict[str, Any],
    *,
    run_id: str,
) -> None:
    if not run_id:
        errors.append(f"{script}: runtime.run_id is required for managed output path scoping")
        return
    for option in sorted(_OUTPUT_PATH_OPTION_NAMES):
        for value in _opt_list(opts, option):
            if run_id not in value:
                errors.append(
                    f"{script}: {option}={value!r} must include runtime.run_id={run_id!r} "
                    "to avoid overwriting another managed run"
                )


def audit_runner_commands(config: dict[str, Any], commands: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate generated managed argv against paper-protocol invariants."""
    errors: list[str] = []
    warnings: list[str] = []
    managed_entrypoints: list[dict[str, str]] = []
    seen_managed_scripts: set[str] = set()
    boundary = Path(str(config.get("safety", {}).get("write_boundary", "/home/linbinhao"))).resolve()
    run_id = str((config.get("runtime") or {}).get("run_id") or "")

    for command in commands:
        argv = [str(x) for x in command["argv"]]
        script = Path(argv[1]).name
        opts = _argv_option_map(argv)
        if script not in MANAGED_RUNNER_SCRIPT_NAMES:
            errors.append(f"{script}: runner entrypoint is not in the managed runner allowlist")
            continue
        if script not in seen_managed_scripts:
            profile = managed_script_profile(script)
            managed_entrypoints.append(
                {
                    "script_name": profile.script_name,
                    "relative_path": profile.relative_path,
                    "runner_root": profile.runner_root,
                    "family": profile.family,
                }
            )
            seen_managed_scripts.add(script)
        _audit_no_forbidden_substrings(errors, script, argv, ["/root/", "seed42"])
        if has_forbidden_selection_reference(argv):
            errors.append(f"{script}: command argv references held-out target selection data")
        _audit_command_path_safety(errors, script, command, opts, boundary=boundary)
        _audit_output_paths_are_run_scoped(errors, script, opts, run_id=run_id)

        if script in DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES:
            report = audit_runner_command(command, config=config)
            errors.extend(report.get("errors", []))
            warnings.extend(report.get("warnings", []))
            continue

    if errors:
        joined = "\n".join(f"  - {error}" for error in errors)
        raise ConfigError("Generated runner commands failed protocol audit:\n" + joined)
    return {
        "passed": True,
        "errors": errors,
        "warnings": warnings,
        "managed_entrypoints": managed_entrypoints,
    }


def audit_postprocess_commands(config: dict[str, Any], commands: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate managed postprocess commands."""
    errors: list[str] = []
    warnings: list[str] = []
    boundary = Path(str(config.get("safety", {}).get("write_boundary", "/home/linbinhao"))).resolve()
    run_id = str((config.get("runtime") or {}).get("run_id") or "")
    allowed_scripts = {
        "export_data_manifest.py",
        "export_metrics_long.py",
        "merge_metrics_long.py",
        "export_paper_table.py",
    }

    for command in commands:
        argv = [str(x) for x in command["argv"]]
        script = Path(argv[1]).name if len(argv) > 1 else ""
        opts = _argv_option_map(argv)
        if script not in allowed_scripts:
            errors.append(f"{script}: postprocess entrypoint is not in the managed reporting allowlist")
            continue
        _audit_no_forbidden_substrings(errors, script, argv, ["/root/", "seed42"])
        _audit_command_path_safety(errors, script, command, opts, boundary=boundary)
        _audit_output_paths_are_run_scoped(errors, script, opts, run_id=run_id)

        if script == "export_metrics_long.py":
            _audit_require_options(
                errors,
                script,
                opts,
                ["--input", "--output-dir", "--expected-mapping-version", "--expected-mapping-hash"],
            )
            _audit_equals(
                errors,
                script,
                opts,
                "--expected-mapping-version",
                config["paper_protocol"]["mapping_version"],
            )
            _audit_equals(
                errors,
                script,
                opts,
                "--expected-mapping-hash",
                config["paper_protocol"]["mapping_hash"],
            )
        elif script == "export_paper_table.py":
            _audit_require_options(errors, script, opts, ["--metrics-long", "--output-dir", "--view", "--dataset"])
            view = str(_opt_first(opts, "--view", ""))
            if view not in set(config.get("evaluation", {}).get("views") or []):
                errors.append(f"{script}: --view={view!r} is not declared in evaluation.views")
        elif script == "merge_metrics_long.py":
            _audit_require_options(errors, script, opts, ["--input", "--output-dir"])
        elif script == "export_data_manifest.py":
            _audit_require_options(errors, script, opts, ["--config", "--local-config", "--output-dir"])

        if not command.get("expected_artifacts"):
            warnings.append(f"{script}: no expected postprocess artifacts declared")

    if errors:
        joined = "\n".join(f"  - {error}" for error in errors)
        raise ConfigError("Generated postprocess commands failed protocol audit:\n" + joined)
    return {"passed": True, "errors": errors, "warnings": warnings}


def _direct_ref_base(data_root: Path, center: str, k: int, seed: int) -> Path:
    return canonical_kshot_base(
        data_root / "paper_vae_only_latenthull_sweep_20260516" / "subsets",
        center,
        k=k,
        seed=seed,
    )


def _direct_ref_base_from_opts(opts: dict[str, Any], data_root: Path, center: str, k: int, seed: int) -> Path:
    subset_root = str(
        _opt_first(opts, "--subset_root", "")
        or _opt_first(opts, "--ref_root", "")
        or _opt_first(opts, "--anchor_base_root", "")
    )
    if subset_root:
        return canonical_kshot_base(subset_root, center, k=k, seed=seed)
    return _direct_ref_base(data_root, center, k, seed)


def _direct_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_root = Path(str(_opt_first(opts, "--out_root", "")))
    child_dir = out_root / "runs" / build_effnet_direct_run_leaf(
        {
            "center": center,
            "k": _opt_first(opts, "--k", "500"),
            "epochs": _opt_first(opts, "--epochs", "30"),
            "seed": _opt_first(opts, "--seed", "20260531"),
            "val_fraction": _opt_first(opts, "--val_fraction", "0.2"),
        }
    )
    return {
        "center": center,
        "output_root": str(out_root),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_model", child_dir / "best_model.pt"),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("train_result", child_dir / "train_result.json"),
            _path_record("run_config", child_dir / "run_config.json"),
            _path_record(
                "eval_result",
                child_dir / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json",
            ),
        ],
    }


def _vae_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_root = Path(str(_opt_first(opts, "--out_root", "")))
    checkpoint_policy = str(_opt_first(opts, "--checkpoint_policy", "best"))
    checkpoint_role = "last_model" if checkpoint_policy == "last" else "best_model"
    checkpoint_name = "last_model.pt" if checkpoint_policy == "last" else "best_model.pt"
    child_dir = out_root / build_effnet_vae_lhat_run_leaf(
        {
            "center": center,
            "hull_M": _opt_first(opts, "--hull_M", "20"),
            "hull_lambda": _opt_first(opts, "--hull_lambda", "0.15"),
            "latent_augmix_severity": _opt_first(opts, "--latent_augmix_severity", "2"),
            "latent_augmix_latent_weight_cap": _opt_first(
                opts,
                "--latent_augmix_latent_weight_cap",
                "0.3",
            ),
            "hull_steps": _opt_first(opts, "--hull_steps", "3"),
            "es_metric": _opt_first(opts, "--es_metric", "target_macro_auprc"),
            "classes_in_scope": _opt_list(opts, "--classes_in_scope")
            or ["CD", "HYP", "MI", "NORM", "STTC"],
            "hull_label_mode": _opt_first(opts, "--hull_label_mode", "primary"),
            "hull_mix_label_mode": _opt_first(opts, "--hull_mix_label_mode", "anchor"),
            "hull_neighbor_distance_space": _opt_first(
                opts,
                "--hull_neighbor_distance_space",
                "raw",
            ),
            "hull_neighbor_mode": _opt_first(opts, "--hull_neighbor_mode", "nearest"),
            "hull_neighbor_pool_size": _opt_first(opts, "--hull_neighbor_pool_size", "0"),
            "hull_neighbor_pool_multiplier": _opt_first(
                opts,
                "--hull_neighbor_pool_multiplier",
                "4",
            ),
            "run_tag_extra": _opt_first(opts, "--run_tag_extra", ""),
            "epochs": _opt_first(opts, "--epochs", "30"),
            "seed": _opt_first(opts, "--seed", "20260531"),
        }
    )
    return {
        "center": center,
        "output_root": str(out_root),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record(checkpoint_role, child_dir / checkpoint_name),
            _path_record("launch_config", child_dir / "launch_config.json"),
            _path_record("train_stdout", child_dir / "train_stdout.log"),
            _path_record("eval_log", child_dir / "eval_full.log"),
            _path_record("eval_result", child_dir / "eval_result_v7_exclrefs_crop1000.json"),
        ],
    }


def _ecgfounder_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_dir = Path(str(_opt_first(opts, "--out_dir", "")))
    params = {
        "center": center,
        "k": _opt_first(opts, "--k", "100"),
        "epochs": _opt_first(opts, "--epochs", "5"),
        "lr": _opt_first(opts, "--lr", "0.0001"),
        "source_weight": _opt_first(opts, "--source_weight", "1.0"),
        "target_real_weight": _opt_first(opts, "--target_real_weight", "40.0"),
        "enable_vae_adv_stream": bool(opts.get("--enable_vae_adv_stream", False)),
        "init_head_path": _opt_first(opts, "--init_head_path", ""),
        "vae_classes_in_scope": _opt_list(opts, "--vae_classes_in_scope") if "--vae_classes_in_scope" in opts else None,
        "vae_min_class_count": _opt_first(opts, "--vae_min_class_count", "1"),
        "adv_weight": _opt_first(opts, "--adv_weight", "20.0"),
        "k_anchor": _opt_first(opts, "--k_anchor", "100"),
        "hull_m": _opt_first(opts, "--hull_m", "20"),
        "hull_lambda": _opt_first(opts, "--hull_lambda", "0.15"),
        "hull_steps": _opt_first(opts, "--hull_steps", "3"),
        "hull_lr": _opt_first(opts, "--hull_lr", "0.25"),
        "hull_weight_mode": _opt_first(opts, "--hull_weight_mode", "optimized"),
        "hull_attack_pos_weight_source": _opt_first(opts, "--hull_attack_pos_weight_source", "none"),
        "hull_attack_pos_weight_clip": _opt_first(opts, "--hull_attack_pos_weight_clip", "50.0"),
        "hull_label_mode": _opt_first(opts, "--hull_label_mode", "primary"),
        "hull_include_anchor": bool(opts.get("--hull_include_anchor", False)),
        "hull_partner_pool": _opt_first(opts, "--hull_partner_pool", "target"),
        "source_partner_limit_per_class": _opt_first(opts, "--source_partner_limit_per_class", "0"),
        "anchor_sample_mode": _opt_first(opts, "--anchor_sample_mode", "stratified"),
        "anchor_sample_power": _opt_first(opts, "--anchor_sample_power", "1.0"),
        "anchor_class_sample_weights": _opt_first(opts, "--anchor_class_sample_weights", ""),
        "anchor_class_max_repeat": _opt_first(opts, "--anchor_class_max_repeat", "0"),
        "adv_weight_warmup_epochs": _opt_first(opts, "--adv_weight_warmup_epochs", "0"),
        "adv_bce_loss_weight": _opt_first(opts, "--adv_bce_loss_weight", "1.0"),
        "adv_clean_logit_anchor_weight": _opt_first(opts, "--adv_clean_logit_anchor_weight", "0.0"),
        "run_suffix": _opt_first(opts, "--run_suffix", ""),
        "target_val_count": _opt_first(opts, "--target_val_count", "0"),
        "selection_metric": _opt_first(opts, "--selection_metric", "source_auprc"),
        "target_val_split_mode": _opt_first(opts, "--target_val_split_mode", "random"),
        "target_val_seed": _opt_first(opts, "--target_val_seed", None),
        "run_name": _opt_first(opts, "--run_name", ""),
        "seed": _opt_first(opts, "--seed", "20260531"),
    }
    child_dir = out_dir / "runs" / build_ecgfounder_fullft_run_leaf(params)
    checkpoint_policy = str(_opt_first(opts, "--checkpoint_policy", "best"))
    checkpoint_role = "last_model" if checkpoint_policy == "last" else "best_model"
    checkpoint_name = "last_model.pt" if checkpoint_policy == "last" else "best_model.pt"
    return {
        "center": center,
        "output_root": str(out_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record(checkpoint_role, child_dir / checkpoint_name),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("eval_result", child_dir / "eval_result.json"),
        ],
    }


def _eval_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    output_path = Path(str(_opt_first(opts, "--output_path", "")))
    model_dir = Path(str(_opt_first(opts, "--model_dir", "")))
    return {
        "center": center,
        "output_root": str(output_path.parent),
        "child_run_dir": str(output_path.parent),
        "model_dir": str(model_dir),
        "expected_artifacts": [
            _path_record("eval_result", output_path),
        ],
    }


def build_artifact_trace(
    config: dict[str, Any],
    *,
    commands: list[dict[str, Any]],
    local_paths: dict[str, str],
    postprocess_commands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data_root = Path(local_paths["data_root"])
    kshot = config["paper_protocol"]["kshot"]
    k = int(kshot["k"])
    seed = int(kshot.get("subset_seed", kshot["seed"]))
    inputs = {
        "checkpoints": [],
        "init_heads": [],
        "k500_refs": [],
        "data_caches": [
            _path_record("pn2021_cache_dir", config["data"]["cache"]["pn2021_cache_dir"], required=False),
            _path_record("pn2021_mmap_cache_dir", config["data"]["cache"]["pn2021_mmap_cache_dir"], required=False),
        ],
    }
    child_runs: list[dict[str, Any]] = []
    postprocess_runs: list[dict[str, Any]] = []
    protocol_warnings: list[str] = []
    command_audit = audit_runner_commands(config, commands)
    postprocess_commands = postprocess_commands if postprocess_commands is not None else build_postprocess_commands(config)
    postprocess_audit = (
        audit_postprocess_commands(config, postprocess_commands)
        if postprocess_commands
        else {"passed": True, "errors": [], "warnings": []}
    )

    model_cfg = config.get("model") or {}
    if model_cfg.get("init_checkpoint"):
        inputs["checkpoints"].append(_path_record("model.init_checkpoint", model_cfg["init_checkpoint"]))
    if model_cfg.get("checkpoint"):
        inputs["checkpoints"].append(_path_record("model.checkpoint", model_cfg["checkpoint"]))

    for command_index, command in enumerate(commands):
        argv = [str(x) for x in command["argv"]]
        opts = _argv_option_map(argv)
        script = Path(argv[1]).name
        center = str(command.get("matrix", {}).get("center") or _opt_first(opts, "--center", ""))

        if _opt_first(opts, "--init_ckpt"):
            inputs["checkpoints"].append(_path_record("command.init_ckpt", _opt_first(opts, "--init_ckpt")))
        if _opt_first(opts, "--init_model_path"):
            inputs["checkpoints"].append(_path_record("command.init_model_path", _opt_first(opts, "--init_model_path")))
        if _opt_first(opts, "--model_dir"):
            checkpoint_name = str(_opt_first(opts, "--checkpoint_name", "best_model.pt"))
            checkpoint_stem = Path(checkpoint_name).stem
            inputs["checkpoints"].append(
                _path_record(
                    f"command.model_dir.{checkpoint_stem}",
                    Path(str(_opt_first(opts, "--model_dir"))) / checkpoint_name,
                )
            )
        if _opt_first(opts, "--init_head_path"):
            inputs["init_heads"].append(_path_record("command.init_head_path", _opt_first(opts, "--init_head_path")))
        if _opt_first(opts, "--checkpoint"):
            inputs["checkpoints"].append(_path_record("command.checkpoint", _opt_first(opts, "--checkpoint")))
        if _opt_first(opts, "--target_logit_anchor_path"):
            inputs["init_heads"].append(
                _path_record("command.target_logit_anchor_path", _opt_first(opts, "--target_logit_anchor_path"))
            )

        if script == "effnet_direct_finetune.py":
            for item_center in _opt_list(opts, "--centers"):
                command_k = int(_opt_first(opts, "--k", k))
                command_seed = int(_opt_first(opts, "--subset_seed", seed))
                base = _direct_ref_base_from_opts(opts, data_root, item_center, command_k, command_seed)
                _append_k500_ref(
                    inputs["k500_refs"],
                    center=item_center,
                    k=command_k,
                    seed=command_seed,
                    base=base,
                    include_latent=False,
                )
                child = _direct_child_run(opts, item_center)
                child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
                child_runs.append(child)
        elif script == "effnet_vae_lhat_augmix.py":
            anchor_base = Path(str(_opt_first(opts, "--anchor_base", "")))
            if "seed42" in str(anchor_base):
                protocol_warnings.append(f"{script}:{center} anchor_base contains seed42")
            matrix_case = _matrix_case(command)
            command_k = int(matrix_case.get("k", k))
            command_seed = int(_opt_first(opts, "--seed", seed))
            signal_override = str(_opt_first(opts, "--target_real_npz_override", ""))
            _append_k500_ref(
                inputs["k500_refs"],
                center=center,
                k=command_k,
                seed=command_seed,
                base=anchor_base,
                include_latent=True,
                signal_path=Path(signal_override) if signal_override else None,
            )
            child = _vae_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "ecgfounder_fullft.py":
            stage = str(_opt_first(opts, "--stage", "k500"))
            if stage != "ptbxl_source":
                ref_meta = str(_opt_first(opts, "--ref_meta_json", ""))
                base = Path(_strip_known_suffix(ref_meta, ".ref_meta.json"))
                signal_override = str(_opt_first(opts, "--target_raw1000_npz_override", ""))
                _append_k500_ref(
                    inputs["k500_refs"],
                    center=center,
                    k=k,
                    seed=seed,
                    base=base,
                    include_latent=False,
                    signal_path=Path(signal_override) if signal_override else None,
                )
            cache_dir = str(_opt_first(opts, "--cache_dir", ""))
            if cache_dir:
                inputs["data_caches"].append(
                    _path_record("ecgfounder.fullft.signal_cache_dir", cache_dir, required=False)
                )
            child = _ecgfounder_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "pn2021_clean_eval.py":
            for ref_meta in _opt_list(opts, "--exclude_ref_ids"):
                ref_name = Path(ref_meta).name
                ref_center = ref_name.split(f"_real_k{k}_seed{seed}.ref_meta.json")[0]
                if ref_center:
                    base = Path(_strip_known_suffix(ref_meta, ".ref_meta.json"))
                    _append_k500_ref(
                        inputs["k500_refs"],
                        center=ref_center,
                        k=k,
                        seed=seed,
                        base=base,
                        include_latent=False,
                    )
            child = _eval_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "pn2021c_eval.py":
            for ref_meta in _opt_list(opts, "--exclude_ref_ids"):
                ref_name = Path(ref_meta).name
                ref_center = ref_name.split(f"_real_k{k}_seed{seed}.ref_meta.json")[0]
                if ref_center:
                    base = Path(_strip_known_suffix(ref_meta, ".ref_meta.json"))
                    _append_k500_ref(
                        inputs["k500_refs"],
                        center=ref_center,
                        k=k,
                        seed=seed,
                        base=base,
                        include_latent=False,
                    )
            for option, role in [
                ("--clean_cache_dir", "pn2021_clean_cache_dir"),
                ("--clean_mmap_cache_dir", "pn2021_clean_mmap_cache_dir"),
            ]:
                value = str(_opt_first(opts, option, ""))
                if value:
                    inputs["data_caches"].append(_path_record(role, value, required=False))
            clean_eval = str(_opt_first(opts, "--clean_eval_json", ""))
            if clean_eval:
                inputs["checkpoints"].append(_path_record("command.clean_eval_json", clean_eval))
            child = _eval_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "ecgfounder_pn2021c_eval.py":
            run_dir = Path(str(_opt_first(opts, "--run_dir", "")))
            if run_dir:
                inputs["checkpoints"].append(_path_record("command.run_dir.last_model", run_dir / "last_model.pt"))
                inputs["data_caches"].append(_path_record("command.run_dir.eval_result", run_dir / "eval_result.json"))
            if center:
                ref_base = (
                    Path(str(config["data"]["kshot_subset_root"]))
                    / center
                    / f"k{k}_seed{seed}"
                    / f"{center}_real_k{k}_seed{seed}"
                )
                _append_k500_ref(
                    inputs["k500_refs"],
                    center=center,
                    k=k,
                    seed=seed,
                    base=ref_base,
                    include_latent=False,
                )
            for option, role in [
                ("--clean_cache_dir", "pn2021_clean_cache_dir"),
                ("--clean_mmap_cache_dir", "pn2021_clean_mmap_cache_dir"),
            ]:
                value = str(_opt_first(opts, option, ""))
                if value:
                    inputs["data_caches"].append(_path_record(role, value, required=False))
            child = _eval_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)

    for command_index, command in enumerate(postprocess_commands):
        expected_artifacts = command.get("expected_artifacts") or []
        postprocess_runs.append(
            {
                "command_index": command_index,
                "name": command.get("name"),
                "output_root": str(Path(str(expected_artifacts[0]["path"])).parent) if expected_artifacts else "",
                "expected_artifacts": expected_artifacts,
            }
        )

    inputs["checkpoints"] = _dedupe_path_records(inputs["checkpoints"])
    inputs["init_heads"] = _dedupe_path_records(inputs["init_heads"])
    inputs["data_caches"] = _dedupe_path_records(inputs["data_caches"])

    artifact_trace = {
        "schema_version": 1,
        "metrics": {
            "views": config.get("evaluation", {}).get("views", []),
            "metrics": config.get("evaluation", {}).get("metrics", []),
            "mapping_version": config["paper_protocol"]["mapping_version"],
            "mapping_hash": config["paper_protocol"]["mapping_hash"],
            "class_order": config["paper_protocol"]["class_order"],
        },
        "selection_policy": config["paper_protocol"]["selection"],
        "inputs": inputs,
        "expected_outputs": {
            "launch_artifacts_declared": config.get("logging", {}).get("launch_artifacts", []),
            "child_artifacts_declared": config.get("logging", {}).get("child_artifacts", []),
            "postprocess_artifacts_declared": config.get("logging", {}).get("postprocess_artifacts", []),
            "required_epoch_metrics": list(config.get("logging", {}).get("required_epoch_metrics", []) or []),
            "managed_required_artifacts": config.get("logging", {}).get("launch_artifacts", []),
            "child_runs": child_runs,
            "postprocess_runs": postprocess_runs,
        },
        "protocol_audit": {
            "kshot_seed": seed,
            "exclude_refs_from_eval": config["paper_protocol"]["kshot"]["exclude_refs_from_eval"],
            "warnings": protocol_warnings,
            "passed": not protocol_warnings,
            "command_audit": command_audit,
            "postprocess_audit": postprocess_audit,
        },
    }
    pipeline_stages = normalize_pipeline_stages(config)
    if pipeline_stages["stages"]:
        artifact_trace["pipeline_stages"] = pipeline_stages
    return artifact_trace


def _stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_summary(project_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(project_root),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def make_dry_run_manifest(
    config: dict[str, Any],
    *,
    commands: list[dict[str, Any]],
    local_paths: dict[str, str],
    run_id: str,
    cli_args: argparse.Namespace,
    postprocess_commands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project_root = Path(local_paths["project_root"])
    now = datetime.now(timezone.utc).isoformat()
    postprocess_commands = postprocess_commands if postprocess_commands is not None else build_postprocess_commands(config)
    manifest = {
        "manifest_schema_version": 2,
        "status": "dry_run",
        "created_at_utc": now,
        "run_id": run_id,
        "entry_config": config.get("_entry_config"),
        "local_config": config.get("_local_config"),
        "config_hash_sha256": _stable_config_hash(config),
        "git": _git_summary(project_root),
        "experiment": config.get("experiment"),
        "run_record": config.get("run_record", {}),
        "paper_protocol": config.get("paper_protocol"),
        "evaluation": config.get("evaluation"),
        "local_paths": local_paths,
        "commands": commands,
        "postprocess_commands": postprocess_commands,
        "cli_overrides": config.get("_cli_overrides", []),
        "artifact_trace": build_artifact_trace(
            config,
            commands=commands,
            local_paths=local_paths,
            postprocess_commands=postprocess_commands,
        ),
        "launcher": {
            "script": "scripts/run_experiment.py",
            "argv": sys.argv,
            "dry_run": bool(cli_args.dry_run),
            "write_plan": bool(cli_args.write_plan),
        },
        "safety": {
            "gpu_launch_requires_cuda_visible_devices": True,
            "managed_child_commands_invoked": False,
            "heldout_target_labels_for_selection": False,
        },
    }
    pipeline_stages = normalize_pipeline_stages(config)
    if pipeline_stages["stages"]:
        manifest["pipeline_stages"] = pipeline_stages
    return manifest
