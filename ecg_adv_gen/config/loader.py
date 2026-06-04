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
from .adapters.direct import audit_direct_finetune_command
from .adapters.source_training import audit_train_ptbxl_command
from ecg_adv_gen.data import DataContractError, validate_data_preprocess_config
from ecg_adv_gen.data.gated_pools import GatedPoolArtifactPaths
from ecg_adv_gen.data.kshot_artifacts import KShotArtifactGroup, canonical_kshot_base
from ecg_adv_gen.evaluation import (
    SelectionPolicyError,
    has_forbidden_selection_reference,
    validate_selection_policy,
)
from ecg_adv_gen.labels import Super5MetadataError, validate_super5_metadata
from ecg_adv_gen.models import (
    ecgfounder_feature_cache_path,
    ecgfounder_feature_cache_paths,
    ecgfounder_k500_head_path,
    ecgfounder_kshot_head_run_dir,
    ecgfounder_linear_probe_head_path,
    ecgfounder_lhat_run_dir,
)
from ecg_adv_gen.run_naming import (
    build_benchmark_direct_run_leaf,
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
        argv = _flatten_argv(runner.get("argv") or [], context)
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
            resolved = Path(text).expanduser().resolve(strict=False)
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
    """Validate generated legacy argv against paper-protocol invariants."""
    errors: list[str] = []
    warnings: list[str] = []
    managed_entrypoints: list[dict[str, str]] = []
    seen_managed_scripts: set[str] = set()
    kshot = config["paper_protocol"]["kshot"]
    expected_k = int(kshot["k"])
    expected_seed = int(kshot.get("subset_seed", kshot["seed"]))
    target_centers = set(config["paper_protocol"]["centers"]["target_4"])
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
                    "wrapper_root": profile.wrapper_root,
                    "family": profile.family,
                }
            )
            seen_managed_scripts.add(script)
        _audit_no_forbidden_substrings(errors, script, argv, ["/root/", "seed42"])
        if has_forbidden_selection_reference(argv):
            errors.append(f"{script}: command argv references held-out target selection data")
        _audit_command_path_safety(errors, script, command, opts, boundary=boundary)
        _audit_output_paths_are_run_scoped(errors, script, opts, run_id=run_id)

        if script == "run_direct_finetune_k500_20260516.py":
            errors.extend(
                audit_direct_finetune_command(
                    command,
                    expected_k=expected_k,
                    expected_seed=expected_seed,
                    target_centers=target_centers,
                )
            )
        elif script == "train_ptbxl.py":
            errors.extend(audit_train_ptbxl_command(command, expected_seed=expected_seed))
        elif script == "train_ibe_repro.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--output_dir",
                    "--train_path",
                    "--val_path",
                    "--epochs",
                    "--batch_size",
                    "--mini_batch_size",
                    "--val_batch_size",
                    "--lr",
                    "--weight_decay",
                    "--num_workers",
                    "--prefetch_factor",
                    "--amp_dtype",
                    "--matmul_precision",
                    "--device",
                    "--seed",
                    "--log_every",
                ],
            )
            output_dir = str(_opt_first(opts, "--output_dir", ""))
            if "ecgtwin_author_repro" not in output_dir or not output_dir.endswith("/ibe_stage1"):
                errors.append(f"{script}: output_dir must end with ecgtwin_author_repro/<run_id>/ibe_stage1")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if "--amp" not in opts:
                errors.append(f"{script}: managed author IBE repro should use AMP")
            if "--drop_last" not in opts:
                errors.append(f"{script}: managed author IBE repro should pass --drop_last")
            if "--persistent_workers" in opts:
                warnings.append(f"{script}: persistent workers are enabled despite historical worker cleanup issues")
            _audit_equals(errors, script, opts, "--batch_size", "65536")
            _audit_equals(errors, script, opts, "--mini_batch_size", "512")
            _audit_equals(errors, script, opts, "--amp_dtype", "bf16")
            train_path = str(_opt_first(opts, "--train_path", ""))
            val_path = str(_opt_first(opts, "--val_path", ""))
            if not train_path.endswith("/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt"):
                errors.append(f"{script}: train_path should point at paired_Mimic_vae_multi_nomic.pt")
            if not val_path.endswith("/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt"):
                errors.append(f"{script}: val_path should point at paired_Mimic_vae_multi_nomic_test.pt")
        elif script == "train_dit_repro.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--output_dir",
                    "--train_path",
                    "--val_path",
                    "--ibe_path",
                    "--epochs",
                    "--batch_size",
                    "--val_batch_size",
                    "--num_workers",
                    "--prefetch_factor",
                    "--amp_dtype",
                    "--matmul_precision",
                    "--device",
                    "--seed",
                    "--log_every",
                ],
            )
            output_dir = str(_opt_first(opts, "--output_dir", ""))
            if "ecgtwin_author_repro" not in output_dir or not output_dir.endswith("/dit_stage2"):
                errors.append(f"{script}: output_dir must end with ecgtwin_author_repro/<run_id>/dit_stage2")
            ibe_path = str(_opt_first(opts, "--ibe_path", ""))
            expected_ibe_suffix = f"/ecgtwin_author_repro/{run_id}/ibe_stage1/checkpoints/IBE_best.pth"
            if run_id and not ibe_path.endswith(expected_ibe_suffix):
                errors.append(f"{script}: --ibe_path must consume the managed IBE stage from the same runtime.run_id")
            if "--use_pretrained_author_ibe" in opts:
                errors.append(f"{script}: must consume the managed IBE stage instead of --use_pretrained_author_ibe")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if "--amp" not in opts:
                errors.append(f"{script}: managed author DiT repro should use AMP")
            if "--persistent_workers" in opts:
                warnings.append(f"{script}: persistent workers are enabled despite historical worker cleanup issues")
            _audit_equals(errors, script, opts, "--batch_size", "512")
            _audit_equals(errors, script, opts, "--val_batch_size", "512")
            _audit_equals(errors, script, opts, "--amp_dtype", "bf16")
            train_path = str(_opt_first(opts, "--train_path", ""))
            val_path = str(_opt_first(opts, "--val_path", ""))
            if not train_path.endswith("/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt"):
                errors.append(f"{script}: train_path should point at paired_Mimic_vae_multi_nomic.pt")
            if not val_path.endswith("/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt"):
                errors.append(f"{script}: val_path should point at paired_Mimic_vae_multi_nomic_test.pt")
        elif script == "train_center_prompt_tokens.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--centers",
                    "--cache_root",
                    "--prompt_bank",
                    "--save_dir",
                    "--ecgtwin_config",
                    "--device",
                    "--K",
                    "--seed",
                    "--total_steps",
                    "--batch_size",
                    "--num_workers",
                    "--lr",
                    "--weight_decay",
                    "--grad_clip",
                    "--amp_dtype",
                    "--token_repeat",
                    "--sample_strategy",
                    "--n_token_vectors",
                    "--token_mode",
                    "--ref_text_mode",
                    "--log_every",
                    "--save_every",
                ],
            )
            centers = set(_opt_list(opts, "--centers"))
            if centers != target_centers:
                errors.append(f"{script}: centers={sorted(centers)!r}, expected {sorted(target_centers)!r}")
            _audit_equals(errors, script, opts, "--K", expected_k)
            _audit_equals(errors, script, opts, "--seed", expected_seed)
            _audit_equals(errors, script, opts, "--total_steps", "2500")
            _audit_equals(errors, script, opts, "--batch_size", "16")
            _audit_equals(errors, script, opts, "--amp_dtype", "bf16")
            _audit_equals(errors, script, opts, "--sample_strategy", "center_class_balanced")
            _audit_equals(errors, script, opts, "--token_mode", "direct")
            _audit_equals(errors, script, opts, "--n_token_vectors", "4")
            _audit_equals(errors, script, opts, "--ref_text_mode", "actual_report")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            cache_root = str(_opt_first(opts, "--cache_root", ""))
            prompt_bank = str(_opt_first(opts, "--prompt_bank", ""))
            save_dir = str(_opt_first(opts, "--save_dir", ""))
            if not cache_root.endswith("/ecgtwin_prompt_token_super5/cache_v1"):
                errors.append(f"{script}: cache_root should point at ecgtwin_prompt_token_super5/cache_v1")
            if prompt_bank != str(Path(cache_root) / "text_prompt_bank.pt"):
                errors.append(f"{script}: prompt_bank must be cache_root/text_prompt_bank.pt")
            if "ecgtwin_prompt_token_minimal" not in save_dir or not save_dir.endswith("/prompt_token_train"):
                errors.append(f"{script}: save_dir must end with ecgtwin_prompt_token_minimal/<run_id>/prompt_token_train")
        elif script == "generate_center_prompt_token_synth.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--center",
                    "--classes",
                    "--cache_root",
                    "--token_bank",
                    "--arm",
                    "--prompt_bank",
                    "--out_dir",
                    "--K",
                    "--selection_seed",
                    "--n_per_class",
                    "--steps",
                    "--device",
                    "--seed",
                    "--victim_ckpt",
                    "--victim_crop_len",
                    "--token_repeat",
                    "--token_scale",
                ],
            )
            center = str(_opt_first(opts, "--center", ""))
            classes = _opt_list(opts, "--classes")
            if center not in target_centers:
                errors.append(f"{script}: unexpected center {center!r}")
            if classes != ["NORM", "MI", "STTC"]:
                errors.append(f"{script}: classes={classes!r}, expected ['NORM', 'MI', 'STTC']")
            _audit_equals(errors, script, opts, "--K", expected_k)
            _audit_equals(errors, script, opts, "--selection_seed", "42")
            _audit_equals(errors, script, opts, "--seed", expected_seed)
            _audit_equals(errors, script, opts, "--arm", "target_token")
            if "--no_token" in opts:
                errors.append(f"{script}: target-token managed config must not pass --no_token")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            cache_root = str(_opt_first(opts, "--cache_root", ""))
            prompt_bank = str(_opt_first(opts, "--prompt_bank", ""))
            token_bank = str(_opt_first(opts, "--token_bank", ""))
            out_dir = str(_opt_first(opts, "--out_dir", ""))
            expected_token_suffix = f"/ecgtwin_prompt_token_minimal/{run_id}/prompt_token_train/prompt_token_bank.pt"
            if not cache_root.endswith("/ecgtwin_prompt_token_super5/cache_v1"):
                errors.append(f"{script}: cache_root should point at ecgtwin_prompt_token_super5/cache_v1")
            if prompt_bank != str(Path(cache_root) / "text_prompt_bank.pt"):
                errors.append(f"{script}: prompt_bank must be cache_root/text_prompt_bank.pt")
            if run_id and not token_bank.endswith(expected_token_suffix):
                errors.append(f"{script}: token_bank must consume prompt_token_train from the same runtime.run_id")
            if f"/ecgtwin_prompt_token_minimal/{run_id}/generated/target_token" not in out_dir:
                errors.append(f"{script}: out_dir must be the run-id-scoped target_token generation directory")
            if not str(_opt_first(opts, "--victim_ckpt", "")).endswith("/triple_labels/super5/best_model.pt"):
                errors.append(f"{script}: victim_ckpt should point at the frozen Super5 baseline best_model.pt")
        elif script == "gate_prompt_token_synth.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--input_dir",
                    "--out_dir",
                    "--classes",
                    "--fs",
                    "--lead_order",
                    "--min_target_prob",
                    "--min_pass_per_class",
                    "--max_pass_per_class",
                    "--cache_root",
                    "--K",
                    "--selection_seed",
                ],
            )
            classes = _opt_list(opts, "--classes")
            input_dir = str(_opt_first(opts, "--input_dir", ""))
            out_dir = str(_opt_first(opts, "--out_dir", ""))
            center = Path(input_dir).name
            if center not in target_centers:
                errors.append(f"{script}: unexpected input center {center!r}")
            if classes != ["NORM", "MI", "STTC"]:
                errors.append(f"{script}: classes={classes!r}, expected ['NORM', 'MI', 'STTC']")
            _audit_equals(errors, script, opts, "--K", expected_k)
            _audit_equals(errors, script, opts, "--selection_seed", "42")
            _audit_equals(errors, script, opts, "--min_target_prob", "0.30")
            if run_id and f"/ecgtwin_prompt_token_minimal/{run_id}/generated/target_token/{center}" not in input_dir:
                errors.append(f"{script}: input_dir must consume generated target_token samples from the same runtime.run_id")
            if input_dir and out_dir != str(Path(input_dir) / "gated"):
                errors.append(f"{script}: out_dir must be <input_dir>/gated")
            cache_root = str(_opt_first(opts, "--cache_root", ""))
            if not cache_root.endswith("/ecgtwin_prompt_token_super5/cache_v1"):
                errors.append(f"{script}: cache_root should point at ecgtwin_prompt_token_super5/cache_v1")
        elif script == "synth_online_at_super5.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--center_name",
                    "--ref_meta_json",
                    "--synth_npz",
                    "--target_real_npz",
                    "--class_trust",
                    "--init_ckpt",
                    "--model_name",
                    "--output_dir",
                    "--data_dir",
                    "--quick_eval_source",
                    "--quick_eval_centers",
                    "--target_real_val_fraction",
                    "--target_real_val_seed",
                    "--ptbxl_raw",
                    "--ptbxl_csv",
                    "--ptbxl_prep",
                    "--attack_mode",
                    "--hull_M",
                    "--hull_lambda",
                    "--hull_steps",
                    "--hull_lr",
                    "--hull_label_mode",
                    "--hull_mix_label_mode",
                    "--hull_neighbor_distance_space",
                    "--hull_neighbor_mode",
                    "--hull_neighbor_pool_size",
                    "--K_anchor",
                    "--pgd_batch",
                    "--target_real_weight",
                    "--ptbxl_weight",
                    "--adv_weight",
                    "--adv_weight_warmup_epochs",
                    "--adv_label_mode",
                    "--adv_teacher_mix",
                    "--classes_in_scope",
                    "--n_epochs",
                    "--num_workers",
                    "--seed",
                    "--device",
                ],
            )
            center = str(_opt_first(opts, "--center_name", ""))
            if center not in target_centers:
                errors.append(f"{script}: unexpected center {center!r}")
            _audit_equals(errors, script, opts, "--seed", expected_seed)
            _audit_equals(errors, script, opts, "--target_real_val_seed", expected_seed)
            _audit_equals(errors, script, opts, "--quick_eval_source", "target_real_val")
            _audit_equals(errors, script, opts, "--attack_mode", "latent_hull")
            _audit_equals(errors, script, opts, "--hull_label_mode", "compatible")
            _audit_equals(errors, script, opts, "--hull_mix_label_mode", "anchor_soft")
            _audit_equals(errors, script, opts, "--adv_label_mode", "latent_mixed_teacher")
            _audit_equals(errors, script, opts, "--adv_teacher_mix", "0.4")
            if _opt_list(opts, "--quick_eval_centers") != [center]:
                errors.append(f"{script}: --quick_eval_centers must contain only the target center")
            if _opt_list(opts, "--classes_in_scope") != ["NORM", "MI", "STTC"]:
                errors.append(f"{script}: classes_in_scope must be ['NORM', 'MI', 'STTC']")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if "--build_class_trust" in opts:
                errors.append(f"{script}: managed online-AT config must consume a gated class_trust artifact")
            if "--disable_adv_stream" in opts:
                errors.append(f"{script}: prompt-token online-AT config must keep the adversarial stream enabled")
            model_name = str(_opt_first(opts, "--model_name", ""))
            expected_model = str((config.get("model") or {}).get("name") or "")
            if expected_model and model_name != expected_model:
                errors.append(f"{script}: --model_name={model_name!r}, expected {expected_model!r}")
            gated_dir = Path(str(_opt_first(opts, "--synth_npz", ""))).parent
            expected_gated_fragment = f"/ecgtwin_prompt_token_minimal/{run_id}/generated/target_token/{center}/gated"
            if run_id and expected_gated_fragment not in str(gated_dir):
                errors.append(f"{script}: synth_npz must consume same-run prompt-token gated latents")
            expected_files = {
                "--synth_npz": "gated_samples.latent.npz",
                "--class_trust": "gated_samples.class_trust.json",
                "--ref_meta_json": "gated_samples.ref_meta.json",
            }
            for option, expected_name in expected_files.items():
                value = Path(str(_opt_first(opts, option, "")))
                if value.name != expected_name:
                    errors.append(f"{script}: {option} must point at {expected_name}")
                if str(value.parent) != str(gated_dir):
                    errors.append(f"{script}: {option} must come from the same gated directory")
            target_real_npz = str(_opt_first(opts, "--target_real_npz", ""))
            expected_real_suffix = (
                f"/{center}/k{expected_k}_seed{expected_seed}/"
                f"{center}_real_k{expected_k}_seed{expected_seed}.signals.npz"
            )
            if not target_real_npz.endswith(expected_real_suffix):
                errors.append(f"{script}: target_real_npz must point at the v7 K-shot signals artifact")
            init_ckpt = str(_opt_first(opts, "--init_ckpt", ""))
            expected_init_suffix = (
                f"/effnet_direct_k500_v7_sjr_rgq/{run_id}/runs/"
                f"{center}_K{expected_k}_direct_ft_ep30_seed{expected_seed}_val0.2/best_model.pt"
            )
            if run_id and not init_ckpt.endswith(expected_init_suffix):
                errors.append(f"{script}: init_ckpt must consume same-run EfficientNet Direct K500 checkpoint")
            output_dir = str(_opt_first(opts, "--output_dir", ""))
            if run_id and not output_dir.endswith(f"/ecgtwin_prompt_token_online_at_minimal/{run_id}/{center}"):
                errors.append(f"{script}: output_dir must be run-id-scoped under ecgtwin_prompt_token_online_at_minimal")
        elif script == "run_benchmark_direct_finetune_v7_20260530.py":
            matrix = command.get("matrix") or {}
            matrix_case = _matrix_case(command)
            expected_command_k = str(matrix_case.get("k", expected_k))
            expected_command_seed = str(matrix_case.get("seed", expected_seed))
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--centers",
                    "--k",
                    "--subset_seed",
                    "--seed",
                    "--model_name",
                    "--init_ckpt",
                    "--epochs",
                    "--val_fraction",
                    "--out_root",
                    "--subset_root",
                    "--data_root",
                    "--python",
                ],
            )
            _audit_equals(errors, script, opts, "--k", expected_command_k)
            _audit_equals(errors, script, opts, "--subset_seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--seed", expected_command_seed)
            centers = set(_opt_list(opts, "--centers"))
            matrix_center = str(matrix.get("center") or matrix_case.get("center") or "")
            if matrix_center:
                if centers != {matrix_center}:
                    errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {sorted(centers)!r}")
                if matrix_center not in target_centers:
                    errors.append(f"{script}: unexpected matrix center {matrix_center!r}")
            elif centers != target_centers:
                errors.append(f"{script}: centers={sorted(centers)!r}, expected {sorted(target_centers)!r}")
            model_name = str(_opt_first(opts, "--model_name", ""))
            expected_model = str((config.get("model") or {}).get("name") or "")
            if expected_model and model_name != expected_model:
                errors.append(f"{script}: --model_name={model_name!r}, expected {expected_model!r}")
            init_ckpt = str(_opt_first(opts, "--init_ckpt", ""))
            if "benchmark_source_v7_sjr_rgq" not in init_ckpt:
                errors.append(f"{script}: init_ckpt must come from benchmark_source_v7_sjr_rgq")
            if model_name and model_name not in init_ckpt:
                errors.append(f"{script}: init_ckpt does not encode model_name {model_name!r}")
            if f"seed{expected_command_seed}" not in init_ckpt:
                errors.append(f"{script}: init_ckpt does not encode seed{expected_command_seed}")
            subset_root = str(_opt_first(opts, "--subset_root", ""))
            if subset_root:
                if f"k{expected_command_k}_seed{expected_command_seed}" in subset_root:
                    errors.append(f"{script}: subset_root should be the root directory, not one center-specific K-shot base")
                if "subsets" not in subset_root:
                    errors.append(f"{script}: subset_root does not point at a K-shot subsets directory")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if "--force" in opts:
                errors.append(f"{script}: benchmark direct managed config must not pass --force")
        elif script == "run_effnet_latent_augmix_stage3_20260524.py":
            matrix_case = _matrix_case(command)
            expected_command_k = str(matrix_case.get("k", expected_k))
            expected_command_seed = str(matrix_case.get("seed", expected_seed))
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--center",
                    "--seed",
                    "--data_root",
                    "--out_root",
                    "--init_ckpt",
                    "--anchor_base",
                    "--hull_lambda",
                    "--hull_neighbor_distance_space",
                    "--hull_neighbor_mode",
                    "--hull_neighbor_pool_size",
                    "--quick_eval_source",
                    "--target_real_val_fraction",
                    "--target_real_val_seed",
                ],
            )
            center = str(_opt_first(opts, "--center", ""))
            matrix_center = str(matrix_case.get("center") or "")
            if matrix_center and center != matrix_center:
                errors.append(f"{script}: matrix center {matrix_center!r} must match --center {center!r}")
            if center not in target_centers:
                errors.append(f"{script}: unexpected center {center!r}")
            _audit_equals(errors, script, opts, "--seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--quick_eval_source", "target_real_val")
            _audit_equals(errors, script, opts, "--target_real_val_seed", expected_command_seed)
            if _opt_first(opts, "--hull_neighbor_distance_space") not in {"raw", "standardized"}:
                errors.append(f"{script}: invalid --hull_neighbor_distance_space")
            if _opt_first(opts, "--hull_neighbor_mode") not in {"nearest", "local_random", "random"}:
                errors.append(f"{script}: invalid --hull_neighbor_mode")
            anchor_base = str(_opt_first(opts, "--anchor_base", ""))
            if f"k{expected_command_k}_seed{expected_command_seed}" not in anchor_base:
                errors.append(
                    f"{script}: anchor_base does not encode K{expected_command_k}/seed{expected_command_seed}"
                )
            init_ckpt = str(_opt_first(opts, "--init_ckpt", ""))
            protocol = str(matrix_case.get("protocol") or "")
            if protocol:
                if f"effnet_direct_{protocol}_v7_sjr_rgq" not in init_ckpt:
                    errors.append(f"{script}: init_ckpt does not match protocol {protocol!r}")
                if f"{center}_K{expected_command_k}_direct_ft_ep" not in init_ckpt:
                    errors.append(f"{script}: init_ckpt does not encode {center}/K{expected_command_k}")
            if "paper_direct_finetune_k500_20260516" not in init_ckpt:
                warnings.append(f"{script}: init_ckpt is not the historical direct-K500 run root")
        elif script == "run_ecgfounder_fullft_super5_pilot_20260523.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--out_dir",
                    "--center",
                    "--ref_meta_json",
                    "--k",
                    "--seed",
                    "--init_head_path",
                    "--target_val_count",
                    "--target_val_seed",
                    "--selection_metric",
                ],
            )
            center = str(_opt_first(opts, "--center", ""))
            if center not in target_centers:
                errors.append(f"{script}: unexpected center {center!r}")
            _audit_equals(errors, script, opts, "--k", expected_k)
            _audit_equals(errors, script, opts, "--seed", expected_seed)
            _audit_equals(errors, script, opts, "--target_val_seed", expected_seed)
            _audit_equals(errors, script, opts, "--selection_metric", "source_plus_target_val_auprc")
            ref_meta = str(_opt_first(opts, "--ref_meta_json", ""))
            if not ref_meta.endswith(f"_real_k{expected_k}_seed{expected_seed}.ref_meta.json"):
                errors.append(f"{script}: ref_meta_json does not encode K{expected_k}/seed{expected_seed}")
        elif script == "run_ecgfounder_kshot_head_ft_20260517.py":
            matrix = command.get("matrix") or {}
            matrix_case = _matrix_case(command)
            expected_command_k = str(matrix_case.get("k", expected_k))
            expected_command_seed = str(matrix_case.get("seed", expected_seed))
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--centers",
                    "--linear_probe_dir",
                    "--preprocess_policy",
                    "--out_dir",
                    "--k",
                    "--source_k",
                    "--subset_seed",
                    "--seed",
                    "--epochs",
                    "--lr",
                    "--weight_decay",
                    "--val_fraction",
                    "--batch_size",
                    "--eval_batch_size",
                    "--device",
                ],
            )
            centers = set(_opt_list(opts, "--centers"))
            matrix_center = str(matrix.get("center") or matrix_case.get("center") or "")
            if matrix_center:
                if centers != {matrix_center}:
                    errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {sorted(centers)!r}")
                if matrix_center not in target_centers:
                    errors.append(f"{script}: unexpected matrix center {matrix_center!r}")
            elif centers != target_centers:
                errors.append(f"{script}: centers={sorted(centers)!r}, expected {sorted(target_centers)!r}")
            _audit_equals(errors, script, opts, "--k", expected_command_k)
            _audit_equals(errors, script, opts, "--source_k", expected_command_k)
            _audit_equals(errors, script, opts, "--subset_seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--preprocess_policy", "official_ptbxl_eval")
            ref_root = str(_opt_first(opts, "--ref_root", ""))
            if matrix_center and not ref_root:
                errors.append(f"{script}: matrix command must pass --ref_root")
            if ref_root:
                if f"k{expected_command_k}_seed{expected_command_seed}" in ref_root:
                    errors.append(f"{script}: ref_root should be the root directory, not one center-specific K-shot base")
                if "subsets" not in ref_root:
                    errors.append(f"{script}: ref_root does not point at a K-shot subsets directory")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if "--reset_head" in opts:
                errors.append(f"{script}: managed direct-K500 head fine-tune must initialize from the PTB-XL linear-probe head")
            if "--force" in opts:
                errors.append(f"{script}: managed config must not pass --force")
            linear_dir = str(_opt_first(opts, "--linear_probe_dir", ""))
            if "ecgfounder_linear_probe_v6" not in linear_dir and "ecgfounder_linear_probe_v7" not in linear_dir:
                warnings.append(f"{script}: linear_probe_dir is not a managed ECGFounder linear-probe cache")
        elif script == "run_ecgfounder_vae_only_lhat_head_ft_20260523.py":
            matrix = command.get("matrix") or {}
            matrix_case = _matrix_case(command)
            expected_command_k = str(matrix_case.get("k", expected_k))
            expected_command_seed = str(matrix_case.get("seed", expected_seed))
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--centers",
                    "--out_dir",
                    "--linear_probe_dir",
                    "--checkpoint",
                    "--preprocess_policy",
                    "--k",
                    "--k_anchor",
                    "--anchor_sample_mode",
                    "--hull_m",
                    "--hull_lambda",
                    "--hull_steps",
                    "--hull_lr",
                    "--head_type",
                    "--init_base_head_from_k500_root",
                    "--selection_metric",
                    "--selection_source",
                    "--target_real_val_fraction",
                    "--target_real_val_seed",
                    "--anchor_base_root",
                    "--seed",
                    "--device",
                ],
            )
            centers = set(_opt_list(opts, "--centers"))
            if not centers:
                errors.append(f"{script}: --centers must name at least one target center")
            if not centers.issubset(target_centers):
                errors.append(f"{script}: centers={sorted(centers)!r}, expected subset of {sorted(target_centers)!r}")
            matrix_center = str(matrix.get("center") or matrix_case.get("center") or "")
            if matrix_center and centers != {matrix_center}:
                errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {sorted(centers)!r}")
            _audit_equals(errors, script, opts, "--k", expected_command_k)
            _audit_equals(errors, script, opts, "--seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--target_real_val_seed", expected_command_seed)
            _audit_equals(errors, script, opts, "--selection_source", "target_real_val")
            _audit_equals(errors, script, opts, "--preprocess_policy", "official_ptbxl_eval")
            if "--report_drop_all_zero_pn2021" not in opts:
                errors.append(f"{script}: missing --report_drop_all_zero_pn2021")
            if "--force" in opts:
                errors.append(f"{script}: managed config must not pass --force")
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            if str(_opt_first(opts, "--head_type", "")) != "residual_adapter":
                warnings.append(f"{script}: head_type is not residual_adapter")
            if "--freeze_base_head" not in opts:
                warnings.append(f"{script}: residual_adapter base head is not frozen")
            selection_metric = str(_opt_first(opts, "--selection_metric", ""))
            if selection_metric not in {"target_auprc", "target_auroc", "target_under_source_floor"}:
                errors.append(f"{script}: selection_metric={selection_metric!r} is not a managed paper-safe option")
            anchor_mode = str(_opt_first(opts, "--anchor_sample_mode", ""))
            if anchor_mode not in {"stratified", "hard_bce", "base_hard_bce", "target_hard_bce"}:
                warnings.append(f"{script}: anchor_sample_mode={anchor_mode!r} may be less stable for the mainline")
            anchor_root = str(_opt_first(opts, "--anchor_base_root", ""))
            if f"k{expected_command_k}_seed{expected_command_seed}" in anchor_root:
                errors.append(f"{script}: anchor_base_root should be the subset root, not one center-specific K-shot base")
            if "subsets" not in anchor_root:
                errors.append(f"{script}: anchor_base_root does not point at a K-shot subsets directory")
            ref_root = str(_opt_first(opts, "--ref_root", ""))
            init_root = str(_opt_first(opts, "--init_base_head_from_k500_root", ""))
            experiment_name = str((config.get("experiment") or {}).get("name") or "")
            requires_ref_root = (
                "ecgfounder_direct_k500_v7_sjr_rgq" in init_root
                or experiment_name.endswith("_v7_sjr_rgq")
            )
            if matrix_center and requires_ref_root and not ref_root:
                errors.append(f"{script}: matrix command must pass --ref_root")
            if ref_root:
                if f"k{expected_command_k}_seed{expected_command_seed}" in ref_root:
                    errors.append(f"{script}: ref_root should be the root directory, not one center-specific K-shot base")
                if "subsets" not in ref_root:
                    errors.append(f"{script}: ref_root does not point at a K-shot subsets directory")
            if (
                "ecgfounder_kshot_head_ft_v6" not in init_root
                and "ecgfounder_direct_k500_v7_sjr_rgq" not in init_root
            ):
                warnings.append(f"{script}: init_base_head_from_k500_root is not a managed ECGFounder direct-head root")
        elif script == "eval_crosscenter.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--scheme",
                    "--model_dir",
                    "--model_name",
                    "--device",
                    "--crop_len",
                    "--batch_size",
                    "--num_workers",
                    "--ptbxl_csv",
                    "--ptbxl_cache",
                    "--preprocess_mode",
                    "--norm_mode",
                    "--pn2021_root",
                    "--pn2021_cache_dir",
                    "--pn2021_mmap_cache_dir",
                    "--exclude_ref_ids",
                    "--output_path",
                ],
            )
            _audit_equals(errors, script, opts, "--scheme", "super5")
            _audit_equals(errors, script, opts, "--crop_len", config["preprocess"]["crop_len"])
            _audit_equals(errors, script, opts, "--preprocess_mode", config["preprocess"]["mode"])
            _audit_equals(errors, script, opts, "--norm_mode", config["preprocess"]["norm_mode"])
            if "--skip_mimic" not in opts:
                errors.append(f"{script}: missing --skip_mimic for managed PN2021 eval")
            if "--skip_pn2021" in opts:
                errors.append(f"{script}: managed PN2021 eval must not pass --skip_pn2021")
            if "--report_drop_all_zero_pn2021" not in opts:
                errors.append(f"{script}: missing --report_drop_all_zero_pn2021")
            ref_metas = _opt_list(opts, "--exclude_ref_ids")
            expected_suffixes = {
                f"{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
                for center in target_centers
            }
            found_suffixes = {Path(path).name for path in ref_metas}
            if expected_suffixes != found_suffixes:
                errors.append(
                    f"{script}: exclude_ref_ids={sorted(found_suffixes)!r}, "
                    f"expected {sorted(expected_suffixes)!r}"
                )
            matrix_center = str((command.get("matrix") or {}).get("center", ""))
            if matrix_center and matrix_center not in target_centers:
                errors.append(f"{script}: unexpected matrix center {matrix_center!r}")
            experiment_name = str((config.get("experiment") or {}).get("name") or "")
            if experiment_name == "ecgtwin_prompt_token_online_at_minimal_eval":
                center = matrix_center or str(_opt_first(opts, "--center", ""))
                model_dir = str(_opt_first(opts, "--model_dir", ""))
                output_path = str(_opt_first(opts, "--output_path", ""))
                expected_model_dir_suffix = f"/ecgtwin_prompt_token_online_at_minimal/{run_id}/{center}"
                expected_output_suffix = (
                    f"/ecgtwin_prompt_token_online_at_minimal_eval/{run_id}/{center}/"
                    "eval_result_v7_super5_sjr_rgq_refexcluded.json"
                )
                _audit_equals(errors, script, opts, "--model_name", "efficientnet1dv2")
                _audit_equals(errors, script, opts, "--device", "cuda")
                if run_id and not model_dir.endswith(expected_model_dir_suffix):
                    errors.append(
                        f"{script}: model_dir must point at same-run prompt-token online-AT output"
                    )
                if run_id and not output_path.endswith(expected_output_suffix):
                    errors.append(
                        f"{script}: output_path must be run-id-scoped under ecgtwin_prompt_token_online_at_minimal_eval"
                    )
        elif script == "eval_pn2021_corruptions.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--mode",
                    "--scheme",
                    "--model_dir",
                    "--clean_mmap_cache_dir",
                    "--clean_cache_dir",
                    "--clean_eval_json",
                    "--centers",
                    "--corruptions",
                    "--severities",
                    "--severity_profile",
                    "--device",
                    "--crop_len",
                    "--batch_size",
                    "--num_workers",
                    "--min_pos",
                    "--seed",
                    "--exclude_ref_ids",
                    "--output_path",
                ],
            )
            _audit_equals(errors, script, opts, "--mode", "stream")
            _audit_equals(errors, script, opts, "--scheme", "super5")
            _audit_equals(errors, script, opts, "--severity_profile", "standard")
            _audit_equals(errors, script, opts, "--crop_len", config["preprocess"]["crop_len"])
            _audit_equals(errors, script, opts, "--min_pos", config["evaluation"]["min_pos"])
            if str(_opt_first(opts, "--device", "")) != "cuda":
                errors.append(f"{script}: --device should be 'cuda' and GPU id must be selected by CUDA_VISIBLE_DEVICES")
            centers = _opt_list(opts, "--centers")
            matrix_center = str((command.get("matrix") or {}).get("center", ""))
            if len(centers) != 1:
                errors.append(f"{script}: managed PN2021-C eval must target exactly one center per command")
            center = centers[0] if centers else matrix_center
            if center not in target_centers:
                errors.append(f"{script}: unexpected center {center!r}")
            if matrix_center and center != matrix_center:
                errors.append(f"{script}: matrix center {matrix_center!r} must match --centers {centers!r}")
            ref_metas = _opt_list(opts, "--exclude_ref_ids")
            expected_ref_name = f"{center}_real_k{expected_k}_seed{expected_seed}.ref_meta.json"
            if [Path(path).name for path in ref_metas] != [expected_ref_name]:
                errors.append(
                    f"{script}: exclude_ref_ids={[Path(path).name for path in ref_metas]!r}, "
                    f"expected {[expected_ref_name]!r}"
                )
            clean_eval = str(_opt_first(opts, "--clean_eval_json", ""))
            model_dir = str(_opt_first(opts, "--model_dir", ""))
            if model_dir and clean_eval != str(Path(model_dir) / "eval_result_v7_exclrefs_crop1000.json"):
                errors.append(f"{script}: clean_eval_json must point inside --model_dir")
            if "--limit" in opts and str(_opt_first(opts, "--limit")) not in {"0", ""}:
                errors.append(f"{script}: managed main PN2021-C eval must not limit samples")
        elif script == "export_percent_kshot_v7_sjr_rgq_20260530.py":
            _audit_require_options(
                errors,
                script,
                opts,
                [
                    "--centers",
                    "--percents",
                    "--fixed-ks",
                    "--seed",
                    "--cache-root",
                    "--mmap-root",
                    "--output-root",
                    "--summary-path",
                ],
            )
            centers = set(_opt_list(opts, "--centers"))
            if centers != target_centers:
                errors.append(f"{script}: centers={sorted(centers)!r}, expected {sorted(target_centers)!r}")
            fixed_ks = set(_opt_list(opts, "--fixed-ks"))
            if fixed_ks != {str(expected_k)}:
                errors.append(f"{script}: fixed-ks={sorted(fixed_ks)!r}, expected {[str(expected_k)]!r}")
            _audit_equals(errors, script, opts, "--seed", expected_seed)
            percents = _opt_list(opts, "--percents")
            if not percents:
                errors.append(f"{script}: --percents must include percent-shot protocols")
            invalid_percents = []
            for value in percents:
                try:
                    number = float(value)
                except ValueError:
                    invalid_percents.append(value)
                    continue
                if number <= 0.0 or number > 1.0:
                    invalid_percents.append(value)
            if invalid_percents:
                errors.append(f"{script}: invalid --percents {invalid_percents!r}")
            if "--force" in opts:
                errors.append(f"{script}: managed subset export must not pass --force")
            output_root = str(_opt_first(opts, "--output-root", ""))
            if "subsets" not in output_root:
                errors.append(f"{script}: output-root does not point at a K-shot subsets directory")
            summary_path = str(_opt_first(opts, "--summary-path", ""))
            if run_id and run_id not in summary_path:
                errors.append(f"{script}: summary-path must be run-id scoped")

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


def _strip_known_suffix(path: str, suffix: str) -> str:
    return path[: -len(suffix)] if path.endswith(suffix) else path


def _path_record(role: str, path: str | Path, *, required: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    exists = resolved.exists()
    record: dict[str, Any] = {
        "role": role,
        "path": str(resolved),
        "required": required,
        "exists": exists,
    }
    if exists and resolved.is_file():
        try:
            record["size_bytes"] = resolved.stat().st_size
        except OSError:
            pass
    return record


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


def _append_k500_ref(
    refs: list[dict[str, Any]],
    *,
    center: str,
    k: int,
    seed: int,
    base: Path,
    include_latent: bool,
) -> None:
    group = KShotArtifactGroup.from_base(
        base,
        center=center,
        k=k,
        seed=seed,
        include_latent=include_latent,
    )
    refs.append(
        {
            "center": group.center,
            "k": group.k,
            "seed": group.seed,
            "anchor_base": str(group.base),
            "ref_meta_json": _path_record("kshot_ref_meta", group.ref_meta),
            "signals_npz": _path_record("kshot_signals", group.signals),
            "latent_npz": (
                _path_record("kshot_latents", group.latents)
                if include_latent
                else None
            ),
        }
    )


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
            _path_record("legacy_run_config", child_dir / "run_config.json"),
            _path_record(
                "eval_result",
                child_dir / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json",
            ),
        ],
    }


def _benchmark_direct_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_root = Path(str(_opt_first(opts, "--out_root", "")))
    child_dir = out_root / "runs" / build_benchmark_direct_run_leaf(
        {
            "center": center,
            "k": _opt_first(opts, "--k", "500"),
            "model_name": _opt_first(opts, "--model_name", "benchmark_resnet1d_wang"),
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
            _path_record("legacy_run_config", child_dir / "run_config.json"),
            _path_record(
                "eval_result",
                child_dir / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json",
            ),
        ],
    }


def _benchmark_source_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    child_dir = Path(str(_opt_first(opts, "--output_dir", "")))
    return {
        "model": str(_opt_first(opts, "--model_name", "")),
        "output_root": str(child_dir.parent),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_model", child_dir / "best_model.pt"),
            _path_record("best_model_auprc", child_dir / "best_model_auprc.pt"),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("train_result", child_dir / "train_result.json"),
        ],
    }


def _ecgtwin_author_ibe_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    child_dir = Path(str(_opt_first(opts, "--output_dir", "")))
    return {
        "stage": "ibe_stage1",
        "output_root": str(child_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("legacy_run_config", child_dir / "run_config.yaml"),
            _path_record("train_log", child_dir / "train.log"),
            _path_record("metrics_jsonl", child_dir / "metrics.jsonl"),
            _path_record("loss_curve_csv", child_dir / "loss_curve.csv"),
            _path_record("loss_curve_png", child_dir / "loss_curve.png"),
            _path_record("ecgtwin_author.latest_checkpoint", child_dir / "checkpoints" / "latest.pt"),
            _path_record("ecgtwin_author.best_checkpoint", child_dir / "checkpoints" / "best.pt"),
            _path_record("ecgtwin_author.ibe_best", child_dir / "checkpoints" / "IBE_best.pth"),
        ],
    }


def _ecgtwin_author_dit_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    child_dir = Path(str(_opt_first(opts, "--output_dir", "")))
    return {
        "stage": "dit_stage2",
        "output_root": str(child_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("legacy_run_config", child_dir / "run_config.yaml"),
            _path_record("train_log", child_dir / "train.log"),
            _path_record("metrics_jsonl", child_dir / "metrics.jsonl"),
            _path_record("loss_curve_csv", child_dir / "loss_curve.csv"),
            _path_record("loss_curve_png", child_dir / "loss_curve.png"),
            _path_record("ecgtwin_author.dit_latest", child_dir / "checkpoints" / "latest.pt"),
            _path_record("ecgtwin_author.dit_best_val", child_dir / "checkpoints" / "best_val.pt"),
            _path_record("ecgtwin_author.dit_best_train", child_dir / "checkpoints" / "best_train.pt"),
            _path_record(
                "ecgtwin_author.dit_latest_weights",
                child_dir / "checkpoints" / "DiT_ECGTwin_latest.pth",
            ),
            _path_record(
                "ecgtwin_author.dit_best_val_weights",
                child_dir / "checkpoints" / "DiT_ECGTwin_best_val.pth",
            ),
            _path_record(
                "ecgtwin_author.dit_best_train_weights",
                child_dir / "checkpoints" / "DiT_ECGTwin_best_train.pth",
            ),
        ],
    }


def _prompt_token_train_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    child_dir = Path(str(_opt_first(opts, "--save_dir", "")))
    return {
        "stage": "prompt_token_train",
        "output_root": str(child_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("prompt_token.run_config", child_dir / "run_config.json"),
            _path_record("prompt_token.metrics_jsonl", child_dir / "metrics.jsonl"),
            _path_record("prompt_token.prompt_token_bank", child_dir / "prompt_token_bank.pt"),
        ],
    }


def _prompt_token_generate_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    center = str(_opt_first(opts, "--center", ""))
    out_dir = Path(str(_opt_first(opts, "--out_dir", "")))
    child_dir = out_dir / center
    return {
        "stage": "prompt_token_generate",
        "center": center,
        "arm": str(_opt_first(opts, "--arm", "")),
        "output_root": str(out_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("prompt_token.samples_npz", child_dir / "samples.npz"),
            _path_record("prompt_token.summary_json", child_dir / "summary.json"),
        ],
    }


def _prompt_token_gate_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    input_dir = Path(str(_opt_first(opts, "--input_dir", "")))
    child_dir = Path(str(_opt_first(opts, "--out_dir", "")))
    gated_paths = GatedPoolArtifactPaths.from_dir(child_dir)
    return {
        "stage": "prompt_token_gate",
        "center": input_dir.name,
        "output_root": str(child_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("prompt_token.gated_samples_npz", gated_paths.samples),
            _path_record("prompt_token.gated_latent_npz", gated_paths.latents),
            _path_record("prompt_token.class_trust_json", gated_paths.class_trust),
            _path_record("prompt_token.ref_meta_json", gated_paths.ref_meta),
            _path_record("prompt_token.gate_report_json", gated_paths.gate_report),
        ],
    }


def _prompt_token_online_at_child_run(opts: dict[str, Any]) -> dict[str, Any]:
    child_dir = Path(str(_opt_first(opts, "--output_dir", "")))
    return {
        "stage": "prompt_token_online_at",
        "center": str(_opt_first(opts, "--center_name", "")),
        "output_root": str(child_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_model", child_dir / "best_model.pt"),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("train_result", child_dir / "train_result.json"),
            _path_record("checkpoint_latest", child_dir / "checkpoints" / "checkpoint_latest.pt"),
            _path_record("checkpoint_index", child_dir / "checkpoints" / "checkpoint_index.jsonl"),
        ],
    }


def _vae_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_root = Path(str(_opt_first(opts, "--out_root", "")))
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
            "freeze_backbone_classifier_only": bool(opts.get("--freeze_backbone_classifier_only")),
            "classifier_only_train_final_norm": bool(opts.get("--classifier_only_train_final_norm")),
            "classifier_adapter_type": _opt_first(opts, "--classifier_adapter_type", "none"),
            "classifier_lora_rank": _opt_first(opts, "--classifier_lora_rank", "16"),
            "unfreeze_last_n_features": _opt_first(opts, "--unfreeze_last_n_features", "0"),
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
            _path_record("best_model", child_dir / "best_model.pt"),
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
    return {
        "center": center,
        "output_root": str(out_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_model", child_dir / "best_model.pt"),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("eval_result", child_dir / "eval_result.json"),
        ],
    }


def _ecgfounder_lhat_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_dir = Path(str(_opt_first(opts, "--out_dir", "")))
    k = int(_opt_first(opts, "--k", "500"))
    hull_m = int(_opt_first(opts, "--hull_m", "20"))
    hull_lambda = str(_opt_first(opts, "--hull_lambda", "0.15"))
    epochs = int(_opt_first(opts, "--epochs", "20"))
    seed = int(_opt_first(opts, "--seed", "20260531"))
    child_dir = ecgfounder_lhat_run_dir(
        out_dir,
        center=center,
        k=k,
        hull_m=hull_m,
        hull_lambda=hull_lambda,
        epochs=epochs,
        seed=seed,
    )
    return {
        "center": center,
        "output_root": str(out_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_head", child_dir / "best_head.pt"),
            _path_record("initial_head", child_dir / "initial_head.pt"),
            _path_record("training_log", child_dir / "training_log.json"),
            _path_record("eval_result", child_dir / "eval_result.json"),
        ],
    }


def _ecgfounder_kshot_head_child_run(opts: dict[str, Any], center: str) -> dict[str, Any]:
    out_dir = Path(str(_opt_first(opts, "--out_dir", "")))
    k = int(_opt_first(opts, "--k", "500"))
    source_k = int(_opt_first(opts, "--source_k", "500"))
    epochs = int(_opt_first(opts, "--epochs", "50"))
    seed = int(_opt_first(opts, "--seed", "20260531"))
    child_dir = ecgfounder_kshot_head_run_dir(
        out_dir,
        center=center,
        k=k,
        source_k=source_k,
        epochs=epochs,
        seed=seed,
    )
    return {
        "center": center,
        "output_root": str(out_dir),
        "child_run_dir": str(child_dir),
        "expected_artifacts": [
            _path_record("best_head", child_dir / "best_head.pt"),
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


def _append_ecgfounder_linear_probe_inputs(
    inputs: dict[str, list[dict[str, Any]]],
    *,
    linear_probe_dir: Path,
    preprocess_policy: str,
) -> None:
    inputs["init_heads"].append(
        _path_record("ecgfounder.linear_probe.best_head", ecgfounder_linear_probe_head_path(linear_probe_dir))
    )
    for dataset, path in ecgfounder_feature_cache_paths(
        linear_probe_dir,
        preprocess_policy=preprocess_policy,
    ).items():
        inputs["data_caches"].append(
            _path_record(
                f"ecgfounder.linear_probe.{dataset}_features",
                path,
            )
        )


def _append_ecgfounder_pn_feature_input(
    inputs: dict[str, list[dict[str, Any]]],
    *,
    linear_probe_dir: Path,
    preprocess_policy: str,
) -> None:
    inputs["init_heads"].append(
        _path_record("ecgfounder.linear_probe.best_head", ecgfounder_linear_probe_head_path(linear_probe_dir))
    )
    inputs["data_caches"].append(
        _path_record(
            "ecgfounder.linear_probe.pn2021_features",
            ecgfounder_feature_cache_path(
                linear_probe_dir,
                dataset="pn2021",
                preprocess_policy=preprocess_policy,
            ),
        )
    )


def _append_ecgfounder_k500_base_head(
    inputs: dict[str, list[dict[str, Any]]],
    *,
    run_root: Path,
    center: str,
    k: int,
    seed: int,
) -> None:
    inputs["init_heads"].append(
        _path_record(
            "ecgfounder.k500_base_head.best_head",
            ecgfounder_k500_head_path(run_root, center=center, k=k, seed=seed),
        )
    )


def _dedupe_path_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("role", "")), str(record.get("path", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


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
        if _opt_first(opts, "--model_dir"):
            inputs["checkpoints"].append(
                _path_record(
                    "command.model_dir.best_model",
                    Path(str(_opt_first(opts, "--model_dir"))) / "best_model.pt",
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

        if script == "run_direct_finetune_k500_20260516.py":
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
        elif script == "train_ptbxl.py":
            for option, label in (
                ("--data_path", "command.data_path"),
                ("--csv_path", "command.csv_path"),
                ("--cache_path", "command.cache_path"),
                ("--split_json", "command.split_json"),
            ):
                if _opt_first(opts, option):
                    inputs["data_caches"].append(_path_record(label, _opt_first(opts, option)))
            child = _benchmark_source_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "train_ibe_repro.py":
            for option, label in (
                ("--train_path", "ecgtwin_author.train_pairs"),
                ("--val_path", "ecgtwin_author.val_pairs"),
            ):
                if _opt_first(opts, option):
                    inputs["data_caches"].append(_path_record(label, _opt_first(opts, option)))
            child = _ecgtwin_author_ibe_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "train_dit_repro.py":
            for option, label in (
                ("--train_path", "ecgtwin_author.train_pairs"),
                ("--val_path", "ecgtwin_author.val_pairs"),
            ):
                if _opt_first(opts, option):
                    inputs["data_caches"].append(_path_record(label, _opt_first(opts, option)))
            if _opt_first(opts, "--ibe_path"):
                inputs["checkpoints"].append(
                    _path_record("ecgtwin_author.ibe_stage1_best", _opt_first(opts, "--ibe_path"))
                )
            child = _ecgtwin_author_dit_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "train_center_prompt_tokens.py":
            for option, label in (
                ("--cache_root", "prompt_token.cache_root"),
                ("--prompt_bank", "prompt_token.text_prompt_bank"),
                ("--ecgtwin_config", "prompt_token.ecgtwin_config"),
                ("--style_ckpt", "prompt_token.style_ckpt"),
                ("--semantic_ckpt", "prompt_token.semantic_ckpt"),
            ):
                if _opt_first(opts, option):
                    inputs["data_caches"].append(
                        _path_record(label, _opt_first(opts, option), required=option != "--cache_root")
                    )
            child = _prompt_token_train_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "generate_center_prompt_token_synth.py":
            center = str(_opt_first(opts, "--center", ""))
            cache_root = Path(str(_opt_first(opts, "--cache_root", "")))
            if _opt_first(opts, "--token_bank"):
                inputs["init_heads"].append(
                    _path_record("prompt_token.prompt_token_bank", _opt_first(opts, "--token_bank"))
                )
            if _opt_first(opts, "--victim_ckpt"):
                inputs["checkpoints"].append(
                    _path_record("prompt_token.victim_ckpt", _opt_first(opts, "--victim_ckpt"))
                )
            for role, path in (
                ("prompt_token.text_prompt_bank", _opt_first(opts, "--prompt_bank")),
                ("prompt_token.center_full_latents", cache_root / "center_full_latents" / f"{center}.pt"),
                (
                    "prompt_token.ref_selection",
                    cache_root
                    / "ref_selection"
                    / f"{center}_k{_opt_first(opts, '--K', k)}_seed{_opt_first(opts, '--selection_seed', '42')}.json",
                ),
            ):
                if path:
                    inputs["data_caches"].append(_path_record(role, path))
            child = _prompt_token_generate_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "gate_prompt_token_synth.py":
            input_dir = Path(str(_opt_first(opts, "--input_dir", "")))
            center = input_dir.name
            cache_root = Path(str(_opt_first(opts, "--cache_root", "")))
            for role, path in (
                ("prompt_token.generated_samples", input_dir / "samples.npz"),
                ("prompt_token.generated_summary", input_dir / "summary.json"),
                ("prompt_token.center_full_latents", cache_root / "center_full_latents" / f"{center}.pt"),
                (
                    "prompt_token.ref_selection",
                    cache_root
                    / "ref_selection"
                    / f"{center}_k{_opt_first(opts, '--K', k)}_seed{_opt_first(opts, '--selection_seed', '42')}.json",
                ),
            ):
                inputs["data_caches"].append(_path_record(role, path))
            child = _prompt_token_gate_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "synth_online_at_super5.py":
            center = str(_opt_first(opts, "--center_name", ""))
            if _opt_first(opts, "--init_ckpt"):
                inputs["checkpoints"].append(
                    _path_record("efficientnet.direct_init_checkpoint", _opt_first(opts, "--init_ckpt"))
                )
            for role, option in (
                ("prompt_token.gated_latent_npz", "--synth_npz"),
                ("prompt_token.gated_class_trust_json", "--class_trust"),
                ("prompt_token.gated_ref_meta_json", "--ref_meta_json"),
                ("target_k500.signals_npz", "--target_real_npz"),
                ("pn2021.training_root", "--data_dir"),
                ("ptbxl.raw100", "--ptbxl_raw"),
                ("ptbxl.database_csv", "--ptbxl_csv"),
                ("ptbxl.preprocessed_cache", "--ptbxl_prep"),
            ):
                if _opt_first(opts, option):
                    inputs["data_caches"].append(_path_record(role, _opt_first(opts, option)))
            target_real_npz = str(_opt_first(opts, "--target_real_npz", ""))
            if target_real_npz.endswith(".signals.npz"):
                base = Path(_strip_known_suffix(target_real_npz, ".signals.npz"))
                _append_k500_ref(inputs["k500_refs"], center=center, k=k, seed=seed, base=base, include_latent=False)
            child = _prompt_token_online_at_child_run(opts)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "run_benchmark_direct_finetune_v7_20260530.py":
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
                child = _benchmark_direct_child_run(opts, item_center)
                child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
                child_runs.append(child)
        elif script == "run_effnet_latent_augmix_stage3_20260524.py":
            anchor_base = Path(str(_opt_first(opts, "--anchor_base", "")))
            if "seed42" in str(anchor_base):
                protocol_warnings.append(f"{script}:{center} anchor_base contains seed42")
            matrix_case = _matrix_case(command)
            command_k = int(matrix_case.get("k", k))
            command_seed = int(_opt_first(opts, "--seed", seed))
            _append_k500_ref(
                inputs["k500_refs"],
                center=center,
                k=command_k,
                seed=command_seed,
                base=anchor_base,
                include_latent=True,
            )
            child = _vae_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "run_ecgfounder_fullft_super5_pilot_20260523.py":
            ref_meta = str(_opt_first(opts, "--ref_meta_json", ""))
            base = Path(_strip_known_suffix(ref_meta, ".ref_meta.json"))
            _append_k500_ref(inputs["k500_refs"], center=center, k=k, seed=seed, base=base, include_latent=False)
            cache_dir = str(_opt_first(opts, "--cache_dir", ""))
            if cache_dir:
                inputs["data_caches"].append(_path_record("ecgfounder.fullft.signal_cache_dir", cache_dir))
            child = _ecgfounder_child_run(opts, center)
            child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
            child_runs.append(child)
        elif script == "run_ecgfounder_kshot_head_ft_20260517.py":
            linear_dir = Path(str(_opt_first(opts, "--linear_probe_dir", "")))
            preprocess_policy = str(_opt_first(opts, "--preprocess_policy", "official_ptbxl_eval"))
            _append_ecgfounder_pn_feature_input(
                inputs,
                linear_probe_dir=linear_dir,
                preprocess_policy=preprocess_policy,
            )
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
                child = _ecgfounder_kshot_head_child_run(opts, item_center)
                child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
                child_runs.append(child)
        elif script == "run_ecgfounder_vae_only_lhat_head_ft_20260523.py":
            linear_dir = Path(str(_opt_first(opts, "--linear_probe_dir", "")))
            preprocess_policy = str(_opt_first(opts, "--preprocess_policy", "official_ptbxl_eval"))
            _append_ecgfounder_linear_probe_inputs(
                inputs,
                linear_probe_dir=linear_dir,
                preprocess_policy=preprocess_policy,
            )
            init_root = str(_opt_first(opts, "--init_base_head_from_k500_root", ""))
            centers = _opt_list(opts, "--centers") or ([center] if center else [])
            for item_center in centers:
                if init_root:
                    _append_ecgfounder_k500_base_head(
                        inputs,
                        run_root=Path(init_root),
                        center=item_center,
                        k=k,
                        seed=seed,
                    )
                command_k = int(_opt_first(opts, "--k", k))
                command_seed = int(_opt_first(opts, "--seed", seed))
                base = _direct_ref_base_from_opts(opts, data_root, item_center, command_k, command_seed)
                _append_k500_ref(
                    inputs["k500_refs"],
                    center=item_center,
                    k=command_k,
                    seed=command_seed,
                    base=base,
                    include_latent=True,
                )
                child = _ecgfounder_lhat_child_run(opts, item_center)
                child.update({"command_index": command_index, "name": command["name"], "matrix": command["matrix"]})
                child_runs.append(child)
        elif script == "eval_crosscenter.py":
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
        elif script == "eval_pn2021_corruptions.py":
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

    return {
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
    return {
        "manifest_schema_version": 2,
        "status": "dry_run",
        "created_at_utc": now,
        "run_id": run_id,
        "entry_config": config.get("_entry_config"),
        "local_config": config.get("_local_config"),
        "config_hash_sha256": _stable_config_hash(config),
        "git": _git_summary(project_root),
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
            "legacy_child_scripts_invoked": False,
            "heldout_target_labels_for_selection": False,
        },
    }
