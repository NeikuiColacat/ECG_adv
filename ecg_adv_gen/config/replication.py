"""Non-canonical replication path, materialization, and audit contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ecg_adv_gen.data.kshot_artifacts import (
    canonical_kshot_base,
    read_ref_meta_record_ids,
)
from ecg_adv_gen.data.class_trust import real_all_present_trust_path
from ecg_adv_gen.matched_effnet import (
    F004_RHO_SWEEP_PROTOCOL,
    is_matched_effnet_arm,
    matched_effnet_arm,
)

from .adapters.common import argv_option_map, opt_first, opt_list
from .loader import (
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)


FIXED_K_SAMPLING_POLICY = "deterministic_random_from_v7_nonzero_eligible_pool"
SUPER5_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
_COMMAND_INPUT_SCOPE = {
    "effnet_vae_lhat_augmix.py": "train",
    "pn2021_clean_eval.py": "clean_eval",
    "pn2021c_eval.py": "center_eval",
}


def build_replication_k500_groups(
    *,
    subset_root: str | Path,
    centers: Sequence[str],
    k: int,
    seed: int,
    artifact_suffixes: Sequence[str],
) -> list[dict[str, Any]]:
    """Build the explicit materialization group checked before replication execution."""

    groups: list[dict[str, Any]] = []
    for center in centers:
        base = canonical_kshot_base(subset_root, str(center), k=int(k), seed=int(seed))
        groups.append(
            {
                "center": str(center),
                "k": int(k),
                "seed": int(seed),
                "base": str(base),
                "artifacts": {
                    str(suffix): {
                        "role": f"replication_k500.{suffix}",
                        "suffix": str(suffix),
                        "path": str(base.with_suffix(f".{suffix}")),
                        "required": True,
                    }
                    for suffix in artifact_suffixes
                },
            }
        )
    return groups


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata, got shape={array.shape}")
    return str(array.reshape(-1)[0])


def _validate_class_order(value: Any, source: str) -> None:
    observed = tuple(str(item) for item in np.asarray(value).reshape(-1).tolist())
    if observed != SUPER5_CLASS_ORDER:
        raise ValueError(
            f"{source} class order={observed!r}, expected {SUPER5_CLASS_ORDER!r}"
        )


def _require_matches(source: str, **checks: tuple[Any, Any]) -> None:
    drift = {key: values for key, values in checks.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"{source} metadata mismatch: {drift}")


def _validate_multihot(labels: Any, *, k: int, source: str) -> np.ndarray:
    array = np.asarray(labels, dtype=np.float32)
    if array.shape != (k, len(SUPER5_CLASS_ORDER)) or not np.isfinite(array).all():
        raise ValueError(f"{source} labels must be finite with shape ({k}, 5)")
    if not np.logical_or(array == 0.0, array == 1.0).all():
        raise ValueError(f"{source} labels must be binary multi-hot values")
    if np.any(array.sum(axis=1) <= 0):
        raise ValueError(f"{source} contains all-zero labels outside the v7 nonzero eligible pool")
    return array


def _validate_numeric_samples(
    values: Any, *, shape: tuple[int, ...], source: str, payload_key: str
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{source} {payload_key} shape={array.shape}, expected {shape}")
    rows = array.reshape(shape[0], -1)
    if not np.isfinite(rows).all(axis=1).all() or np.any(np.ptp(rows, axis=1) <= 1e-8):
        raise ValueError(f"{source} {payload_key} must be finite and non-flat per sample")
    return array


def _validate_ref_meta(
    path: Path,
    *,
    center: str,
    k: int,
    seed: int,
    mapping_version: str,
    mapping_hash: str,
) -> tuple[list[str], dict[str, Any]]:
    parsed = read_ref_meta_record_ids(
        path,
        expected_center=center,
        expected_k=k,
        expected_seed=seed,
    )
    meta = _json_object(path)
    expected = {
        "fixed_k": k,
        "protocol": "fixed_k",
        "protocol_tag": f"k{k}",
        "sampling_policy": FIXED_K_SAMPLING_POLICY,
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
    }
    _require_matches("ref_meta", **{key: (meta.get(key), value) for key, value in expected.items()})
    for key in ("raw_v7_nonzero_count", "eligible_v7_nonzero_with_latents"):
        if type(meta.get(key)) is not int or int(meta[key]) < k:
            raise ValueError(f"ref_meta {key} must be an integer >= K={k}")
    _validate_class_order(meta.get("class_names"), path.name)
    return parsed.record_ids, meta


def _validate_array_npz(
    path: Path,
    *,
    payload_key: str,
    center: str,
    k: int,
    record_ids: list[str],
    mapping_version: str,
    mapping_hash: str,
    expected_payload_shape: tuple[int, ...],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            payload_key, "labels", "record_ids", "center_name", "class_names",
            "mapping_version", "mapping_hash",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path.name} missing arrays {missing}")
        _validate_numeric_samples(
            payload[payload_key], shape=expected_payload_shape,
            source=path.name, payload_key=payload_key,
        )
        labels = _validate_multihot(payload["labels"], k=k, source=path.name)
        _validate_class_order(payload["class_names"], path.name)
        _require_matches(
            path.name,
            record_ids=(payload["record_ids"].astype(str).tolist() == record_ids, True),
            center_name=(_scalar_text(payload["center_name"]), center),
            mapping_version=(_scalar_text(payload["mapping_version"]), mapping_version),
            mapping_hash=(_scalar_text(payload["mapping_hash"]), mapping_hash),
        )
    return labels


def _validate_raw1000(
    path: Path,
    *,
    center: str,
    k: int,
    record_ids: list[str],
    ref_meta_path: Path,
) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        required = {"signals", "labels", "record_ids", "metadata"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path.name} missing arrays {missing}")
        _validate_numeric_samples(
            payload["signals"], shape=(k, 1000, 12),
            source=path.name, payload_key="signals",
        )
        labels = _validate_multihot(payload["labels"], k=k, source=path.name)
        ids = payload["record_ids"].astype(str).tolist()
        metadata_array = np.asarray(payload["metadata"], dtype=object).reshape(-1)
        metadata = metadata_array[0] if len(metadata_array) == 1 else None
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name} metadata must contain one mapping")
    _validate_class_order(metadata.get("class_names"), path.name)
    _require_matches(
        path.name,
        record_ids=(ids == record_ids, True),
        center=(str(metadata.get("center") or ""), center),
        n_records=(int(metadata.get("n_records", -1)), k),
        source_ref_meta_json=(
            Path(str(metadata.get("source_ref_meta_json") or "")).resolve(),
            ref_meta_path.resolve(),
        ),
        preprocess=(metadata.get("preprocess"), {
            "target_fs": 100, "target_len": 1000,
            "preprocess_mode": "minimal_resample", "norm_mode": "none",
        }),
        model_input_preprocess=(metadata.get("model_input_preprocess"), {
            "norm_mode": "per_sample_global", "crop_len": 1000,
        }),
        signals_are_pre_zscore=(metadata.get("signals_are_pre_zscore"), True),
    )
    return labels


def _validate_class_trust(
    path: Path,
    *,
    center: str,
    label_counts: Mapping[str, int],
    mapping_version: str,
    mapping_hash: str,
) -> None:
    payload = _json_object(path)
    _require_matches(
        path.name,
        center=(str(payload.get("center") or ""), center),
        mapping_version=(payload.get("mapping_version"), mapping_version),
        mapping_hash=(payload.get("mapping_hash"), mapping_hash),
    )
    observed_counts = {str(key): int(value) for key, value in (payload.get("label_counts") or {}).items()}
    if set(observed_counts) != set(SUPER5_CLASS_ORDER):
        raise ValueError(f"{path.name} label_counts keys do not match canonical class order")
    if observed_counts != dict(label_counts):
        raise ValueError(f"{path.name} label_counts do not match selected signal labels")
    if not isinstance(payload.get("class_trust"), dict):
        raise ValueError(f"{path.name} class_trust must be a mapping")
    expected_trust = {name: float(count > 0) for name, count in label_counts.items()}
    observed_trust = {str(key): float(value) for key, value in payload["class_trust"].items()}
    if set(observed_trust) != set(SUPER5_CLASS_ORDER):
        raise ValueError(f"{path.name} class_trust keys do not match canonical class order")
    if observed_trust != expected_trust:
        raise ValueError(f"{path.name} class_trust values do not match label_counts > 0")


def verify_replication_k500_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    mapping_version: str,
    mapping_hash: str,
) -> dict[str, Any]:
    """Fully validate each current seed's K-shot numeric artifact groups serially."""

    missing: list[dict[str, Any]] = []
    identity_errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    for raw_group in groups:
        group = dict(raw_group)
        center, k, seed = str(group["center"]), int(group["k"]), int(group["seed"])
        base = Path(str(group["base"]))
        expected_base = canonical_kshot_base(base.parents[2], center, k=k, seed=seed)
        artifacts = dict(group.get("artifacts") or {})
        group_missing: list[dict[str, Any]] = []
        if base != expected_base:
            identity_errors.append(
                {"center": center, "error": f"base path does not encode center/K/seed: {base}"}
            )
        for suffix, record in artifacts.items():
            path = Path(str(record.get("path") or ""))
            item = {"center": center, "suffix": suffix, "path": str(path), "exists": path.is_file()}
            if path.is_file():
                item["size_bytes"] = path.stat().st_size
                verified.append(item)
            else:
                missing.append(item)
                group_missing.append(item)
        if not group_missing and base == expected_base:
            try:
                ref_path = Path(artifacts["ref_meta.json"]["path"])
                record_ids, ref_meta = _validate_ref_meta(
                    ref_path,
                    center=center,
                    k=k,
                    seed=seed,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                )
                labels = _validate_array_npz(
                    Path(artifacts["signals.npz"]["path"]),
                    payload_key="signals",
                    center=center,
                    k=k,
                    record_ids=record_ids,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    expected_payload_shape=(k, 1000, 12),
                )
                latent_labels = _validate_array_npz(
                    Path(artifacts["latent.npz"]["path"]),
                    payload_key="latents",
                    center=center,
                    k=k,
                    record_ids=record_ids,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    expected_payload_shape=(k, 4, 128),
                )
                raw_labels = _validate_raw1000(
                    Path(artifacts["raw1000.npz"]["path"]),
                    center=center,
                    k=k,
                    record_ids=record_ids,
                    ref_meta_path=ref_path,
                )
                if not np.array_equal(labels, raw_labels):
                    raise ValueError("raw1000 labels do not match selected signals labels")
                if not np.array_equal(labels, latent_labels):
                    raise ValueError("latent labels do not match selected signals labels")
                class_names = list(SUPER5_CLASS_ORDER)
                label_counts = dict(zip(class_names, labels.sum(axis=0).astype(int).tolist()))
                ref_counts = {str(key): int(value) for key, value in (ref_meta.get("label_counts") or {}).items()}
                if set(ref_counts) != set(SUPER5_CLASS_ORDER):
                    raise ValueError("ref_meta label_counts keys do not match canonical class order")
                if ref_counts != label_counts:
                    raise ValueError("ref_meta label_counts do not match selected signals labels")
                _validate_class_trust(
                    Path(artifacts["class_trust.json"]["path"]),
                    center=center,
                    label_counts=label_counts,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                identity_errors.append({"center": center, "base": str(base), "error": str(exc)})
        group_reports.append(
            {
                "center": center,
                "k": k,
                "seed": seed,
                "base": str(base),
                "missing_count": len(group_missing),
            }
        )
    return {
        "passed": not missing and not identity_errors,
        "group_count": len(groups),
        "missing_count": len(missing),
        "identity_error_count": len(identity_errors),
        "missing": missing,
        "identity_errors": identity_errors,
        "verified_inputs": verified,
        "groups": group_reports,
    }


def _relative_config(repo_root: Path, value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _replication_input_contract(surface: Mapping[str, Any]) -> dict[str, Any]:
    replicates = [dict(item) for item in surface.get("replicates") or []]
    seeds = [int(item["seed"]) for item in replicates]
    declared_seeds = [int(seed) for seed in surface.get("seeds") or seeds]
    if not seeds or seeds != declared_seeds or len(seeds) != len(set(seeds)):
        raise ValueError("input contract seeds are empty, duplicate, or disagree with replicates")
    validation_seeds = [int(seed) for seed in surface.get("validation_seeds") or seeds]
    if (
        not validation_seeds
        or validation_seeds != sorted(set(validation_seeds))
        or not set(seeds).issubset(validation_seeds)
    ):
        raise ValueError(
            "input contract validation_seeds must be sorted, unique, and include replicate seeds"
        )
    families = {str(item.get("input_root_family") or "") for item in replicates}
    if len(families) != 1 or not next(iter(families)):
        raise ValueError("input contract must declare one non-empty root family")
    if not surface.get("materialization_artifact_suffixes"):
        raise ValueError("input contract must declare materialization artifact suffixes")
    report = dict(surface.get("validation_report") or {})
    if (
        not report.get("relative_to_input_root")
        or not report.get("sha256")
        or int(report.get("expected_group_count") or 0) <= 0
    ):
        raise ValueError("input contract validation report is incomplete")
    return {
        "source_surface": dict(surface),
        "root_family": next(iter(families)),
        "validation_seeds": validation_seeds,
    }


def _replication_surface_contracts(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for raw_surface in index.get("replication_surfaces") or []:
        surface = dict(raw_surface)
        name = str(surface.get("name") or "")
        if not name or name in contracts:
            raise ValueError(f"duplicate or empty replication surface name: {name!r}")
        contracts[name] = _replication_input_contract(surface)
    return contracts


def _normalized_input_seeds(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} input_seeds must be a non-empty list")
    seeds = [int(seed) for seed in value]
    if len(seeds) != len(set(seeds)) or seeds != sorted(seeds):
        raise ValueError(f"{label} input_seeds must be sorted and unique")
    return seeds


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _canonical_study_input_root(
    config: Mapping[str, Any], *, root_family: str
) -> Path:
    data_root = _resolved_path(str((config.get("paths") or {}).get("data_root") or ""))
    if not str((config.get("paths") or {}).get("data_root") or ""):
        raise ValueError("indexed auxiliary config has no paths.data_root")
    return _resolved_path(data_root / root_family / "subsets")


def _input_cell(
    *, center: str, k: int, seed: int, base: str | Path
) -> tuple[str, int, int, str]:
    return str(center), int(k), int(seed), str(_resolved_path(base))


def _binding_input_cell(item: "CommandInputBinding") -> tuple[str, int, int, str]:
    return _input_cell(center=item.artifact_center, k=item.k, seed=item.seed, base=item.base)


@dataclass(frozen=True)
class CommandInputBinding:
    """One command/input K-shot identity, retained before any de-duplication."""

    command_index: int
    input_index: int
    arm: str
    artifact_center: str
    k: int
    seed: int
    base: Path
    consume_latent: bool

    def path(self, suffix: str) -> Path:
        return self.base.with_suffix(suffix)

    ref_meta_json = property(lambda self: self.path(".ref_meta.json"))
    raw1000_npz = property(lambda self: self.path(".raw1000.npz"))
    signals_npz = property(lambda self: self.path(".signals.npz"))
    latent_npz = property(lambda self: self.path(".latent.npz"))

    def cell(self) -> tuple[Any, ...]:
        """Return the duplicate-preserving identity compared across four surfaces."""

        return (
            self.command_index, self.input_index, self.arm, self.artifact_center,
            self.k, self.seed, str(_resolved_path(self.base)), self.consume_latent,
        )

    def identity_record(self) -> dict[str, Any]:
        return {
            "command_index": self.command_index, "input_index": self.input_index,
            "arm": self.arm, "artifact_center": self.artifact_center,
            "k": self.k, "seed": self.seed,
            "anchor_base": str(_resolved_path(self.base)),
            "ref_meta_json": str(_resolved_path(self.ref_meta_json)),
            "consume_latent": self.consume_latent,
        }


def _command_argv_scope(
    command: Mapping[str, Any], command_index: int
) -> tuple[list[str], str]:
    argv = [str(item) for item in command.get("argv") or []]
    if len(argv) < 2:
        raise ValueError(f"indexed auxiliary manifest command[{command_index}] argv is incomplete")
    script = Path(argv[1]).name
    try:
        return argv, _COMMAND_INPUT_SCOPE[script]
    except KeyError:
        raise ValueError(
            f"indexed auxiliary command[{command_index}] uses unsupported runner {script!r}"
        ) from None


def _command_arm(
    config: Mapping[str, Any],
    matrix: Mapping[str, Any],
    opts: Mapping[str, Any],
) -> str:
    case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else {}
    protocol = str((config.get("paper_protocol") or {}).get("comparison_protocol") or "")
    if protocol == F004_RHO_SWEEP_PROTOCOL:
        return "a5"
    return str(
        case.get("arm")
        or matrix.get("arm")
        or opt_first(opts, "--comparison_arm", "")
    )


def _train_consumes_latent(
    paper: Mapping[str, Any], arm: str, command_index: int
) -> bool:
    if str(paper.get("comparison_protocol") or "") == F004_RHO_SWEEP_PROTOCOL:
        return True
    if not is_matched_effnet_arm(arm):
        raise ValueError(
            f"indexed auxiliary train command[{command_index}] has unsupported arm {arm!r}"
        )
    return bool(matched_effnet_arm(arm).vae_lhat)


def _expected_command_input_bindings(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_root: Path,
) -> list[CommandInputBinding]:
    commands = manifest.get("commands") or []
    if not isinstance(commands, list) or not commands:
        raise ValueError("indexed auxiliary manifest commands are missing")
    paper = config.get("paper_protocol") or {}
    kshot = paper.get("kshot") or {}
    default_k = int(kshot["k"])
    default_seed = int(kshot.get("subset_seed", kshot["seed"]))
    target_centers = [str(center) for center in (paper.get("centers") or {}).get("target_4") or []]
    bindings: list[CommandInputBinding] = []
    for command_index, raw_command in enumerate(commands):
        if not isinstance(raw_command, Mapping):
            raise ValueError(f"indexed auxiliary manifest command[{command_index}] is not a mapping")
        command = dict(raw_command)
        argv, scope = _command_argv_scope(command, command_index)
        opts = argv_option_map(argv)
        matrix = command.get("matrix") or {}
        case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else {}
        producer_center = str(matrix.get("center") or opt_first(opts, "--center", ""))
        if not producer_center:
            raise ValueError(f"indexed auxiliary manifest command[{command_index}] has no center cell")
        k = int(case.get("k", default_k))
        # Evaluation --seed is corruption RNG state. K-shot identity comes only
        # from the matrix case or the frozen paper subset seed.
        seed = int(case.get("seed", default_seed))
        arm = _command_arm(config, matrix, opts)
        if scope == "clean_eval":
            if not target_centers:
                raise ValueError("indexed clean evaluation has no target_4 center contract")
            artifact_centers = target_centers
        else:
            artifact_centers = [producer_center]
        consume_latent = (
            _train_consumes_latent(paper, arm, command_index)
            if scope == "train"
            else False
        )
        for input_index, artifact_center in enumerate(artifact_centers):
            base = canonical_kshot_base(expected_root, artifact_center, k=k, seed=seed)
            bindings.append(
                CommandInputBinding(
                    command_index=command_index, input_index=input_index, arm=arm,
                    artifact_center=artifact_center, k=k, seed=seed,
                    base=_resolved_path(base),
                    consume_latent=consume_latent,
                )
            )
    return bindings


def _require_exact_path(observed: Any, expected: Path, *, label: str) -> Path:
    if not observed:
        raise ValueError(f"{label} is required")
    actual = _resolved_path(str(observed))
    wanted = _resolved_path(expected)
    if actual != wanted:
        raise ValueError(f"{label} does not bind canonical input: {actual} != {wanted}")
    return actual


def _require_exact_paths(
    prefix: str,
    specs: Mapping[str, tuple[Any, Path]],
) -> dict[str, Path]:
    return {
        name: _require_exact_path(observed, expected, label=f"{prefix} {name}")
        for name, (observed, expected) in specs.items()
    }


def _binding_from_base(
    expected: CommandInputBinding,
    *,
    base: Path,
    consume_latent: bool,
) -> CommandInputBinding:
    return replace(
        expected,
        base=_resolved_path(base),
        consume_latent=bool(consume_latent),
    )


def _base_from_ref_meta(path: Path, *, label: str) -> Path:
    text = str(_resolved_path(path))
    suffix = ".ref_meta.json"
    if not text.endswith(suffix):
        raise ValueError(f"{label} must end with {suffix}: {text}")
    return _resolved_path(text[: -len(suffix)])


def _actual_input_record(
    binding: CommandInputBinding,
    *,
    ref_meta: Path,
    raw1000: Path | None = None,
    latent: Path | None = None,
    class_trust: Path | None = None,
) -> dict[str, Any]:
    trust_record = None if class_trust is None else {
        "path": str(class_trust), "provenance_only": True,
        "source_raw1000": str(raw1000),
    }
    return {
        **binding.identity_record(),
        "actual_consumed": {
            "ref_meta_json": str(ref_meta),
            "raw1000_npz": str(raw1000) if raw1000 else None,
            "latent_npz": str(latent) if latent else None, "class_trust": trust_record,
        },
    }


def _actual_command_input_bindings(
    manifest: Mapping[str, Any],
    expected_bindings: Sequence[CommandInputBinding],
) -> tuple[list[CommandInputBinding], list[dict[str, Any]]]:
    """Rebuild wrapper children and read the paths their final argv consumes."""

    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as effnet_wrapper
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )

    commands = manifest.get("commands") or []
    grouped: dict[int, list[CommandInputBinding]] = {}
    for item in expected_bindings:
        grouped.setdefault(item.command_index, []).append(item)
    observed: list[CommandInputBinding] = []
    records: list[dict[str, Any]] = []
    for command_index, command in enumerate(commands):
        cells = sorted(grouped.get(command_index, []), key=lambda item: item.input_index)
        if not cells:
            raise ValueError(f"expected input cells are missing for command[{command_index}]")
        argv, scope = _command_argv_scope(command, command_index)
        prefix = f"actual child command[{command_index}]"
        if scope == "train":
            if len(cells) != 1:
                raise ValueError(f"train command[{command_index}] must bind exactly one K500 input")
            try:
                args = effnet_wrapper.parse_args(argv[2:])
            except SystemExit as exc:
                raise ValueError(f"{prefix} wrapper argv is invalid") from exc
            cell = cells[0]
            if (str(args.center), int(args.seed)) != (cell.artifact_center, cell.seed):
                raise ValueError(f"{prefix} center/seed input identity drift")
            data_root = Path(args.data_root)
            out_root = Path(args.out_root) if args.out_root else data_root / "paper_effnet_latent_augmix_stage3_20260524"
            paths = resolve_effnet_vae_lhat_paths(args, data_root=data_root, out_root=out_root)
            wrapper = _require_exact_paths(prefix, {
                "anchor override": (paths.anchor_base, cell.base),
                "target override": (paths.signal_npz, cell.raw1000_npz),
                "synth override": (paths.latent_npz, cell.latent_npz),
            })
            trust = real_all_present_trust_path(wrapper["target override"], out_root / "config")
            child = build_effnet_vae_lhat_train_cmd(
                args, python=argv[0], data_root=data_root, paths=paths, class_trust=trust
            )
            opts = argv_option_map(child)
            final = _require_exact_paths(f"{prefix} final", {
                "ref_meta_json": (opt_first(opts, "--ref_meta_json", ""), cell.ref_meta_json),
                "target_real_npz": (opt_first(opts, "--target_real_npz", ""), cell.raw1000_npz),
                "generated class_trust": (opt_first(opts, "--class_trust", ""), trust),
            })
            raw_latent = opt_first(opts, "--synth_npz", "")
            latent = (
                _require_exact_path(raw_latent, cell.latent_npz, label=f"{prefix} final synth_npz")
                if cell.consume_latent else None
            )
            if not cell.consume_latent and raw_latent:
                raise ValueError(f"{prefix} non-VAE arm must not consume latent")
            ref_meta = final["ref_meta_json"]
            actual = _binding_from_base(
                cell, base=_base_from_ref_meta(ref_meta, label=f"{prefix} ref_meta_json"),
                consume_latent=latent is not None,
            )
            observed.append(actual)
            records.append(_actual_input_record(
                actual, ref_meta=ref_meta, raw1000=final["target_real_npz"],
                latent=latent, class_trust=final["generated class_trust"],
            ))
            continue

        opts = argv_option_map(argv)
        ref_values = opt_list(opts, "--exclude_ref_ids")
        if len(ref_values) != len(cells):
            raise ValueError(f"{prefix} exclude_ref_ids count {len(ref_values)} != {len(cells)}")
        forbidden = {"--target_real_npz", "--synth_npz", "--class_trust"}.intersection(opts)
        if forbidden:
            raise ValueError(f"{prefix} evaluation consumes {sorted(forbidden)}")
        for cell, value in zip(cells, ref_values):
            ref_meta = _require_exact_path(
                value, cell.ref_meta_json,
                label=f"{prefix} input[{cell.input_index}] exclude_ref_ids",
            )
            actual = _binding_from_base(
                cell, base=_base_from_ref_meta(ref_meta, label=f"{prefix} exclude_ref_ids"),
                consume_latent=False,
            )
            observed.append(actual)
            records.append(_actual_input_record(actual, ref_meta=ref_meta))
    return observed, records


def _manifest_path_record(
    ref: Mapping[str, Any], key: str, *, expected_path: Path,
    expected_role: str, ref_index: int,
) -> Path:
    record = ref.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest K500 refs[{ref_index}].{key} is required")
    if record.get("required") is not True:
        raise ValueError(f"manifest K500 refs[{ref_index}].{key}.required must be true")
    if str(record.get("role") or "") != expected_role:
        raise ValueError(f"manifest K500 refs[{ref_index}].{key}.role must be {expected_role!r}")
    return _require_exact_path(
        record.get("path"), expected_path,
        label=f"manifest K500 refs[{ref_index}].{key}.path",
    )


def _manifest_command_input_bindings(
    manifest: Mapping[str, Any],
    expected_bindings: Sequence[CommandInputBinding],
) -> tuple[list[CommandInputBinding], list[dict[str, Any]]]:
    refs = (((manifest.get("artifact_trace") or {}).get("inputs") or {})
            .get("k500_refs") or [])
    if not isinstance(refs, list) or not refs:
        raise ValueError("indexed auxiliary has no manifest K500 refs")
    indexed: dict[tuple[int, int], tuple[int, Mapping[str, Any]]] = {}
    for ref_index, ref in enumerate(refs):
        if not isinstance(ref, Mapping):
            raise ValueError(f"manifest K500 refs[{ref_index}] is not a mapping")
        try:
            key = int(ref["command_index"]), int(ref["input_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest K500 refs[{ref_index}] lacks command/input identity") from exc
        if key in indexed:
            raise ValueError(f"manifest K500 refs duplicate command/input identity {key}")
        indexed[key] = (ref_index, ref)

    commands = manifest.get("commands") or []
    expected_keys = {(item.command_index, item.input_index) for item in expected_bindings}
    if set(indexed) != expected_keys:
        raise ValueError(
            "manifest K500 refs/input cell mismatch: "
            f"observed={sorted(indexed)}, expected={sorted(expected_keys)}"
        )

    manifest_bindings: list[CommandInputBinding] = []
    public_records: list[dict[str, Any]] = []
    for cell in expected_bindings:
        ref_index, ref = indexed[(cell.command_index, cell.input_index)]
        try:
            identity = str(ref["center"]), int(ref["k"]), int(ref["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest K500 refs[{ref_index}] is incomplete: {exc}") from exc
        if identity != (cell.artifact_center, cell.k, cell.seed):
            raise ValueError(f"manifest K500 refs[{ref_index}] center/K/seed input identity drift")
        base = _require_exact_path(
            ref.get("anchor_base"),
            cell.base,
            label=f"manifest K500 refs[{ref_index}].anchor_base",
        )
        ref_path = _manifest_path_record(
            ref, "ref_meta_json", expected_path=cell.ref_meta_json,
            expected_role="kshot_ref_meta", ref_index=ref_index,
        )
        _, scope = _command_argv_scope(commands[cell.command_index], cell.command_index)
        is_train = scope == "train"
        signals_path = _manifest_path_record(
            ref, "signals_npz",
            expected_path=cell.raw1000_npz if is_train else cell.signals_npz,
            expected_role="kshot_raw1000_signals" if is_train else "kshot_signals",
            ref_index=ref_index,
        )
        if is_train and cell.consume_latent:
            latent_path = _manifest_path_record(
                ref, "latent_npz", expected_path=cell.latent_npz,
                expected_role="kshot_latents", ref_index=ref_index,
            )
        else:
            if ref.get("latent_npz") is not None:
                raise ValueError(
                    f"manifest K500 refs[{ref_index}].latent_npz must be absent for this command"
                )
            latent_path = None
        manifest_binding = _binding_from_base(
            cell, base=base, consume_latent=latent_path is not None,
        )
        manifest_bindings.append(manifest_binding)
        public_records.append(
            {
                **manifest_binding.identity_record(),
                "manifest_companions": {
                    "signals_npz": str(signals_path),
                    "latent_npz": str(latent_path) if latent_path else None,
                },
            }
        )
    return manifest_bindings, public_records


def _binding_counter(bindings: Sequence[CommandInputBinding]) -> Counter:
    return Counter(binding.cell() for binding in bindings)


def _assert_exact_binding_cells(
    expected: Sequence[CommandInputBinding],
    observed: Sequence[CommandInputBinding],
    *,
    label: str,
) -> None:
    wanted = _binding_counter(expected)
    actual = _binding_counter(observed)
    if actual != wanted:
        missing = list((wanted - actual).elements())
        unexpected = list((actual - wanted).elements())
        raise ValueError(
            f"{label} input cell mismatch: missing={missing}, unexpected={unexpected}"
        )


def _per_command_binding_cells_match(
    expected: Sequence[CommandInputBinding],
    *observed_surfaces: Sequence[CommandInputBinding],
) -> bool:
    for command_index in {item.command_index for item in expected}:
        select = lambda surface: [  # noqa: E731 - compact local projection
            item for item in surface if item.command_index == command_index
        ]
        wanted = _binding_counter(select(expected))
        if any(_binding_counter(select(surface)) != wanted for surface in observed_surfaces):
            return False
    return True


def _indexed_auxiliary_bindings(index: Mapping[str, Any]):
    contracts = _replication_surface_contracts(index)
    for raw_surface in index.get("replication_surfaces") or []:
        surface = dict(raw_surface)
        contract = contracts[str(surface["name"])]
        source_config = str(surface.get("source_config") or "")
        if source_config:
            yield {
                "surface": surface,
                "entry": {"canonical": False},
                "stage": "source",
                "config": source_config,
                "input_seeds": [],
                "expected_command_count": int(
                    (surface.get("source_stage") or {}).get("command_count") or 0
                ),
                "input_contract": contract,
                "source_only": True,
            }
        for raw_replicate in surface.get("replicates") or []:
            replicate = dict(raw_replicate)
            for stage, config in (replicate.get("stages") or {}).items():
                yield {
                    "surface": surface,
                    "entry": replicate,
                    "stage": str(stage),
                    "config": str(config),
                    "input_seeds": [int(replicate["seed"])],
                    "expected_command_count": None,
                    "input_contract": contract,
                }

    study_names: set[str] = set()
    for raw_surface in index.get("study_surfaces") or []:
        surface = dict(raw_surface)
        name = str(surface.get("name") or "")
        contract_ref = str(surface.get("input_contract_ref") or "")
        if not name or name in study_names or not contract_ref or contract_ref not in contracts:
            raise ValueError(
                f"study surface name {name!r} is duplicate/empty or references "
                f"unknown input contract {contract_ref!r}"
            )
        study_names.add(name)
        entries = surface.get("entries") or []
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"study surface {name!r} entries must be a non-empty list")
        roles: set[str] = set()
        for raw_entry in entries:
            entry = dict(raw_entry)
            role = str(entry.get("role") or "")
            config = str(entry.get("config") or "")
            if not role or role in roles or not config:
                raise ValueError(f"study surface {name!r} has duplicate/empty role or config")
            roles.add(role)
            expected_count = int(entry.get("expected_command_count") or 0)
            if expected_count <= 0:
                raise ValueError(f"study surface {name!r}/{role} command count must be positive")
            input_seeds = _normalized_input_seeds(
                entry.get("input_seeds"), label=f"study surface {name!r}/{role}"
            )
            if not set(input_seeds).issubset(contracts[contract_ref]["validation_seeds"]):
                raise ValueError(f"study surface {name!r}/{role} input seeds leave its contract")
            yield {
                "surface": surface,
                "entry": entry,
                "stage": role,
                "config": config,
                "input_seeds": input_seeds,
                "expected_command_count": expected_count,
                "input_contract": contracts[contract_ref],
            }


def _validate_replication_binding(
    index: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    entry = str(config.get("_entry_config") or "")
    target = _relative_config(repo_root, entry)
    records = list(_indexed_auxiliary_bindings(index))
    found = [record for record in records if record["config"] == target]
    if len(found) > 1:
        raise ValueError(f"duplicate config in auxiliary surfaces: {target}")
    binding = found[0] if found else None
    indexed_paths = {str(record["config"]) for record in records}
    families = {str(record["input_contract"]["root_family"]) for record in records}
    sources = {
        _relative_config(repo_root, source)
        for source in (config.get("_config_sources") or [])
    }
    subset_root = Path(str((config.get("data") or {}).get("kshot_subset_root") or ""))
    uses_indexed_family = subset_root.name == "subsets" and subset_root.parent.name in families
    derived_from_indexed = bool(sources.intersection(indexed_paths))
    under_auxiliary_root = target.startswith(("configs/replications/", "configs/studies/"))
    if binding is None:
        if uses_indexed_family or derived_from_indexed or under_auxiliary_root:
            raise ValueError(
                f"entry {target!r} must be indexed in an active replication/study surface"
            )
        return None

    if not _git_tracked(repo_root, target):
        raise ValueError(f"indexed replication entry must be git tracked: {target}")
    if target not in sources:
        raise ValueError(f"indexed auxiliary config sources do not contain entry {target}")
    if binding.get("source_only"):
        expected_count = int(binding.get("expected_command_count") or 0)
        observed_count = len((manifest or {}).get("commands") or [])
        if expected_count <= 0 or observed_count != expected_count:
            raise ValueError(
                f"indexed source command count drift: {observed_count} != {expected_count}"
            )
        return binding
    family = str(binding["input_contract"]["root_family"])
    expected_root = _canonical_study_input_root(config, root_family=family)
    observed_root = _resolved_path(subset_root)
    if observed_root != expected_root:
        raise ValueError(
            f"indexed auxiliary canonical input root drift for {binding['stage']}: "
            f"{observed_root} != {expected_root}"
        )

    expected_count = binding.get("expected_command_count")
    if expected_count is None:
        kshot = (config.get("paper_protocol") or {}).get("kshot") or {}
        config_seeds = sorted(
            {int(kshot.get(key, -1)) for key in ("seed", "subset_seed")}
        )
        if config_seeds != binding["input_seeds"]:
            raise ValueError(
                f"indexed replication input seed drift for {binding['stage']}: "
                f"{config_seeds} != {binding['input_seeds']}"
            )
    observed_count = len((manifest or {}).get("commands") or [])
    declared_count = (
        int(expected_count)
        if expected_count is not None
        else int(binding["surface"].get("command_count_per_stage") or 0)
    )
    if declared_count and observed_count != declared_count:
        label = "study" if expected_count is not None else "replication"
        raise ValueError(
            f"indexed {label} command count drift for {binding['stage']}: "
            f"{observed_count} != {declared_count}"
        )

    expected_bindings = _expected_command_input_bindings(
        config, manifest or {}, expected_root=expected_root
    )
    actual_bindings, actual_records = _actual_command_input_bindings(
        manifest or {}, expected_bindings
    )
    manifest_bindings, manifest_records = _manifest_command_input_bindings(
        manifest or {}, expected_bindings
    )
    _assert_exact_binding_cells(expected_bindings, actual_bindings, label="actual child argv")
    _assert_exact_binding_cells(expected_bindings, manifest_bindings, label="manifest K500 refs")
    observed_seeds = sorted({item.seed for item in manifest_bindings})
    if observed_seeds != binding["input_seeds"]:
        raise ValueError(
            f"indexed auxiliary input seeds drift for {binding['stage']}: "
            f"{observed_seeds} != {binding['input_seeds']}"
        )

    companions = {
        (item["command_index"], item["input_index"]): item["manifest_companions"]
        for item in manifest_records
    }
    command_records = [
        {**item, "manifest_companions": companions[(item["command_index"], item["input_index"])]}
        for item in actual_records
    ]
    return {
        **binding,
        "expected_command_input_bindings": expected_bindings,
        "actual_command_input_bindings": actual_bindings,
        "manifest_command_input_bindings": manifest_bindings,
        "command_input_records": command_records,
    }


def _surface_materialization_groups(
    surface: Mapping[str, Any], *, subset_root: Path, centers: Sequence[str], k: int
) -> list[dict[str, Any]]:
    contract = _replication_input_contract(surface)
    if contract["root_family"] != subset_root.parent.name:
        raise ValueError("replication surface input root families are inconsistent")
    return _materialization_groups_for_seeds(
        contract,
        subset_root=subset_root,
        centers=centers,
        k=k,
        seeds=contract["validation_seeds"],
    )


def _materialization_groups_for_seeds(
    contract: Mapping[str, Any],
    *,
    subset_root: Path,
    centers: Sequence[str],
    k: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        group
        for seed in seeds
        for group in build_replication_k500_groups(
            subset_root=subset_root,
            centers=centers,
            k=k,
            seed=int(seed),
            artifact_suffixes=contract["source_surface"][
                "materialization_artifact_suffixes"
            ],
        )
    ]


def _materialization_groups_for_cells(
    contract: Mapping[str, Any],
    cells: Sequence[tuple[str, int, int, str]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for center, k, seed, base in cells:
        built = build_replication_k500_groups(
            subset_root=_resolved_path(base).parents[2],
            centers=[center],
            k=k,
            seed=seed,
            artifact_suffixes=contract["source_surface"][
                "materialization_artifact_suffixes"
            ],
        )
        groups.extend(built)
    return groups


def attach_replication_preflight(
    manifest: dict[str, Any],
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    index_path: Path,
) -> dict[str, Any]:
    """Attach the indexed replication materialization gate to a launcher manifest."""

    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    binding = _validate_replication_binding(
        index, config, repo_root=repo_root, manifest=manifest
    )
    if binding is None:
        return manifest
    if binding.get("source_only"):
        return manifest
    surface = binding["surface"]
    entry = binding["entry"]
    stage = str(binding["stage"])
    contract = binding["input_contract"]
    input_surface = contract["source_surface"]
    input_seeds = list(binding["input_seeds"])
    paper = config["paper_protocol"]
    subset_root = Path(str(config["data"]["kshot_subset_root"]))
    centers = list(paper["centers"]["target_4"])
    k = int(paper["kshot"]["k"])
    expected_bindings = binding["expected_command_input_bindings"]
    actual_bindings = binding["actual_command_input_bindings"]
    manifest_bindings = binding["manifest_command_input_bindings"]
    unique_cells = list(dict.fromkeys(map(_binding_input_cell, expected_bindings)))
    groups = _materialization_groups_for_cells(contract, unique_cells)
    groups_by_cell = {
        _input_cell(center=group["center"], k=group["k"], seed=group["seed"], base=group["base"]): group
        for group in groups
    }
    if set(groups_by_cell) != set(unique_cells):
        raise ValueError(
            f"indexed auxiliary {stage} attached materialization group identities drift"
        )
    attached_bindings = [
        _binding_from_base(
            item, base=Path(str(groups_by_cell[_binding_input_cell(item)]["base"])),
            consume_latent=item.consume_latent,
        )
        for item in expected_bindings
    ]
    _assert_exact_binding_cells(
        expected_bindings,
        attached_bindings,
        label="attached replication K500 groups",
    )
    per_command_passed = _per_command_binding_cells_match(
        expected_bindings,
        actual_bindings,
        manifest_bindings,
        attached_bindings,
    )
    if not per_command_passed:
        raise ValueError(f"indexed auxiliary {stage} per-command input cell mismatch")
    validation_groups = _materialization_groups_for_seeds(
        contract,
        subset_root=subset_root,
        centers=centers,
        k=k,
        seeds=contract["validation_seeds"],
    )
    expected_group_count = int(
        (input_surface.get("validation_report") or {}).get("expected_group_count") or 0
    )
    if expected_group_count != len(validation_groups):
        raise ValueError(
            f"replication validation report scope={expected_group_count}, expected {len(validation_groups)}"
        )
    out = copy.deepcopy(manifest)
    trace = out.setdefault("artifact_trace", {})
    trace.setdefault("inputs", {})["replication_k500_groups"] = groups
    trace["inputs"]["replication_validation_groups"] = validation_groups
    trace["inputs"]["replication_command_input_bindings"] = binding[
        "command_input_records"
    ]
    trace["replication_preflight"] = {
        "surface": surface["name"],
        "stage": stage,
        "seeds": input_seeds,
        **({"seed": input_seeds[0]} if len(input_seeds) == 1 else {}),
        "canonical": bool(entry.get("canonical")),
        "declared_k500_input_status": entry.get("k500_input_status"),
        "materialization_artifact_suffixes": list(
            input_surface["materialization_artifact_suffixes"]
        ),
        "runtime_consumed_artifact_suffixes": list(
            input_surface.get("runtime_consumed_artifact_suffixes") or []
        ),
        "provenance_companion_suffixes": list(
            input_surface.get("provenance_companion_suffixes") or []
        ),
        "current_group_count": len(groups),
        "validation_group_count": len(validation_groups),
        "validation_artifact_count": sum(
            len(group.get("artifacts") or {}) for group in validation_groups
        ),
        "execute_gate": True,
        "command_input_contract": {
            "passed": True,
            "per_command_passed": per_command_passed,
            "expected_cell_count": len(expected_bindings),
            "actual_argv_cell_count": len(actual_bindings),
            "manifest_ref_cell_count": len(manifest_bindings),
            "attached_group_cell_count": len(attached_bindings),
        },
        "validation_report": {
            "path": str(
                Path(str(config["data"]["kshot_subset_root"])).parent
                / str(
                    (input_surface.get("validation_report") or {}).get(
                        "relative_to_input_root"
                    )
                )
            ),
            "expected_sha256": str(
                (input_surface.get("validation_report") or {}).get("sha256") or ""
            ),
            "expected_group_count": expected_group_count,
        },
    }
    return out


def _git_tracked(repo_root: Path, rel_path: str) -> bool:
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_replication_validation_report(
    contract: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind the surface report to every indexed seed/center artifact hash and size."""

    errors: list[str] = []
    path = Path(str(contract.get("path") or ""))
    report: dict[str, Any] = {}
    actual_sha = ""
    bound_artifact_count = 0
    try:
        actual_sha = _sha256_file(path)
        if actual_sha != str(contract.get("expected_sha256") or ""):
            errors.append(f"validation report sha256={actual_sha}, expected {contract.get('expected_sha256')}")
        report = _json_object(path)
        report_groups = report.get("groups") if isinstance(report.get("groups"), list) else []
        if report.get("passed") is not True or report.get("errors") not in ([], None):
            errors.append("validation report does not record passed=true with no errors")
        if len(report_groups) != int(contract.get("expected_group_count") or 0):
            errors.append(
                f"validation report group_count={len(report_groups)}, expected {contract.get('expected_group_count')}"
            )
        expected_pairs = {(int(group["seed"]), str(group["center"])) for group in groups}
        observed_pairs = {
            (int(group.get("seed")), str(group.get("center")))
            for group in report_groups
            if isinstance(group, Mapping)
        }
        if observed_pairs != expected_pairs:
            errors.append("validation report seed/center groups do not match the indexed surface")
        if any(
            not all(bool(value) for value in (group.get("checks") or {}).values())
            for group in report_groups
            if isinstance(group, Mapping)
        ):
            errors.append("validation report contains a failed group check")
        rows = {(int(row["seed"]), str(row["center"])): row for row in report_groups}
        key_map = {
            "ref_meta.json": "ref_meta", "signals.npz": "signals", "latent.npz": "latent",
            "raw1000.npz": "raw1000", "class_trust.json": "class_trust",
        }
        for group in groups:
            row = rows.get((int(group["seed"]), str(group["center"]))) or {}
            hashes, sizes = row.get("artifact_sha256") or {}, row.get("artifact_sizes") or {}
            for suffix, artifact in (group.get("artifacts") or {}).items():
                bound_artifact_count += 1
                report_key, artifact_path = key_map.get(str(suffix)), Path(str(artifact.get("path") or ""))
                if not report_key or not artifact_path.is_file():
                    errors.append(f"validation report cannot bind {group['center']}:{suffix}")
                elif hashes.get(report_key) != _sha256_file(artifact_path):
                    errors.append(f"artifact sha256 mismatch for {group['center']}:{suffix}")
                elif int(sizes.get(report_key, -1)) != artifact_path.stat().st_size:
                    errors.append(f"artifact size mismatch for {group['center']}:{suffix}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return {
        "passed": not errors,
        "path": str(path),
        "expected_sha256": str(contract.get("expected_sha256") or ""),
        "actual_sha256": actual_sha,
        "bound_group_count": len(groups),
        "bound_artifact_count": bound_artifact_count,
        "errors": errors,
    }


def _audit_validation_report(
    surface: Mapping[str, Any], *, input_roots: set[str], groups: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    spec = surface.get("validation_report") or {}
    if not spec or len(input_roots) != 1:
        return {"passed": False, "errors": ["replication validation report/root contract is invalid"]}
    result = verify_replication_validation_report(
        {
            "path": str(Path(next(iter(input_roots))) / str(spec.get("relative_to_input_root"))),
            "expected_sha256": str(spec.get("sha256") or ""),
            "expected_group_count": int(spec.get("expected_group_count") or 0),
        },
        groups,
    )
    result["input_roots"] = sorted(input_roots)
    return result


def _command_key(command: Mapping[str, Any]) -> tuple[str, str]:
    matrix = command.get("matrix") or {}
    case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else {}
    return str(matrix.get("center") or ""), str(case.get("arm") or matrix.get("arm") or "")


def audit_replication_command_grid(
    commands: Sequence[Mapping[str, Any]],
    *,
    centers: Sequence[str],
    arms: Sequence[str],
) -> dict[str, Any]:
    """Pure exact-grid audit shared by indexed auxiliary experiment surfaces."""

    expected = Counter((str(center), str(arm)) for center in centers for arm in arms)
    observed = Counter(_command_key(command) for command in commands)
    missing = list((expected - observed).elements())
    unexpected = list((observed - expected).elements())
    return {
        "passed": observed == expected,
        "expected_count": sum(expected.values()),
        "observed_count": sum(observed.values()),
        "missing": [{"center": center, "arm": arm} for center, arm in missing],
        "unexpected": [{"center": center, "arm": arm} for center, arm in unexpected],
    }


def audit_replication_producer_consumers(
    producer_children: Sequence[Mapping[str, Any]],
    consumer_commands: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Pure producer/consumer binding audit returning paths for isolation checks."""

    producers: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    for child in producer_children:
        matrix = child.get("matrix") or {}
        case = matrix.get("case") if isinstance(matrix.get("case"), Mapping) else {}
        arm = (child.get("protocol_claim") or {}).get("arm") or case.get("arm") or matrix.get("arm")
        key = (str(child.get("center") or ""), str(arm or ""))
        if key in producers:
            errors.append(f"duplicate producer key {key}")
        producers[key] = str(child.get("child_run_dir") or "")
    outputs: set[str] = set()
    for stage, commands in consumer_commands.items():
        for command in commands:
            options = argv_option_map([str(value) for value in command["argv"]])
            key = _command_key(command)
            model_dir = str(
                opt_first(options, "--model_dir", "")
                or opt_first(options, "--run_dir", "")
            )
            if producers.get(key) != model_dir:
                errors.append(
                    f"{stage}:{key} consumer_run_dir={model_dir!r}, "
                    f"producer={producers.get(key)!r}"
                )
            outputs.add(str(opt_first(options, "--output_path")))
    return {
        "passed": not errors,
        "producer_count": len(producers),
        "producer_dirs": sorted(producers.values()),
        "consumer_output_paths": sorted(outputs),
        "errors": errors,
    }


def audit_replication_source_checkpoint_binding(
    source_children: Sequence[Mapping[str, Any]],
    training_commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every matched training command to start from one source best model."""

    errors: list[str] = []
    source_best_model = ""
    if len(source_children) != 1:
        errors.append(
            f"source config must produce exactly one child run, observed {len(source_children)}"
        )
    else:
        artifacts = source_children[0].get("expected_artifacts") or []
        best_paths = [
            str(item.get("path") or "")
            for item in artifacts
            if str(item.get("role") or "") == "best_model"
        ]
        if len(best_paths) != 1 or not best_paths[0].endswith("/best_model.pt"):
            errors.append("source child must declare exactly one best_model.pt artifact")
        else:
            source_best_model = best_paths[0]

    bound_count = 0
    observed_paths: set[str] = set()
    for command in training_commands:
        options = argv_option_map([str(value) for value in command["argv"]])
        init_path = str(opt_first(options, "--init_model_path", ""))
        observed_paths.add(init_path)
        if source_best_model and init_path == source_best_model:
            bound_count += 1
        else:
            errors.append(
                f"training command {_command_key(command)} init_model_path={init_path!r}, "
                f"source_best_model={source_best_model!r}"
            )
    if not training_commands:
        errors.append("matched source binding requires at least one training command")

    return {
        "passed": not errors,
        "source_best_model": source_best_model,
        "bound_training_command_count": bound_count,
        "observed_init_model_paths": sorted(observed_paths),
        "errors": errors,
    }


def audit_replication_path_isolation(
    producer_dirs: Mapping[int, set[str]],
    consumer_outputs: Mapping[int, set[str]],
) -> dict[str, Any]:
    """Pure pairwise path-disjointness audit for same-run-id replications."""

    errors: list[str] = []
    seeds = sorted(producer_dirs)
    for index_left, left in enumerate(seeds):
        for right in seeds[index_left + 1 :]:
            producer_overlap = sorted(producer_dirs[left].intersection(producer_dirs[right]))
            consumer_overlap = sorted(consumer_outputs[left].intersection(consumer_outputs[right]))
            if producer_overlap:
                errors.append(f"producer paths overlap for seeds {left} and {right}: {producer_overlap}")
            if consumer_overlap:
                errors.append(f"consumer output paths overlap for seeds {left} and {right}: {consumer_overlap}")
    return {"passed": not errors, "errors": errors}


def _audit_replication_source_config(
    surface: Mapping[str, Any],
    *,
    repo_root: Path,
    local_config_path: Path,
    run_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load the optional source producer and expose its selected checkpoint contract."""

    config_rel = str(surface.get("source_config") or "")
    if not config_rel:
        return None, []
    source_stage = dict(surface.get("source_stage") or {})
    config_path = repo_root / config_rel
    tracked = _git_tracked(repo_root, config_rel)
    errors: list[str] = []
    children: list[dict[str, Any]] = []
    command_count = 0
    checkpoint = ""
    try:
        config = load_experiment_config(
            config_path,
            local_config_path,
            runtime_context={"run_id": run_id},
        )
        local_paths = validate_experiment_config(config, repo_root=repo_root)
        commands = build_runner_commands(config)
        command_count = len(commands)
        expected_count = int(source_stage.get("command_count") or 0)
        if command_count != expected_count or expected_count != 1:
            errors.append(
                f"source command_count={command_count}, declared={expected_count}; expected one"
            )
        for command in commands:
            options = argv_option_map([str(value) for value in command["argv"]])
            if str(opt_first(options, "--stage", "")) != "ptbxl_source":
                errors.append("source command must use --stage ptbxl_source")
        manifest = make_dry_run_manifest(
            config,
            commands=commands,
            local_paths=local_paths,
            run_id=run_id,
            cli_args=type("Args", (), {"dry_run": True, "write_plan": False})(),
        )
        children = list(
            manifest["artifact_trace"]["expected_outputs"]["child_runs"]
        )
        if len(children) == 1:
            artifacts = children[0].get("expected_artifacts") or []
            roles = {str(item.get("role") or "") for item in artifacts}
            if "selection" not in roles:
                errors.append("source child does not declare selection.json")
            checkpoint_paths = [
                Path(str(item.get("path") or ""))
                for item in artifacts
                if str(item.get("role") or "") == "best_model"
            ]
            if len(checkpoint_paths) == 1:
                checkpoint = checkpoint_paths[0].name
            else:
                errors.append("source child does not declare exactly one best_model artifact")
        else:
            errors.append(f"source config produced {len(children)} child runs, expected one")
        if checkpoint != str(source_stage.get("checkpoint") or ""):
            errors.append(
                f"source checkpoint={checkpoint!r}, declared={source_stage.get('checkpoint')!r}"
            )
        if source_stage.get("selection_data") != "ptbxl_fold9_only":
            errors.append("source selection_data must be ptbxl_fold9_only")
        if source_stage.get("report_only_data") != "ptbxl_fold10":
            errors.append("source report_only_data must be ptbxl_fold10")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    if not tracked:
        errors.append(f"source config is not tracked: {config_rel}")
    return (
        {
            "config": config_rel,
            "config_exists": config_path.is_file(),
            "config_tracked_by_git": tracked,
            "command_count": command_count,
            "selection_data": source_stage.get("selection_data"),
            "report_only_data": source_stage.get("report_only_data"),
            "checkpoint": checkpoint,
            "contract_passed": not errors,
            "errors": errors,
        },
        children,
    )


def _declared_ready(status: str) -> bool | None:
    if status.startswith("present"):
        return True
    if status == "materialization_required":
        return False
    return None


def _audit_study_binding(
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    index_path: Path,
    local_config_path: Path,
) -> dict[str, Any]:
    role, config_rel = str(binding["stage"]), str(binding["config"])
    base = {"role": role, "config": config_rel}
    try:
        run_id = f"audit_{binding['surface']['name']}_{role}"
        config = load_experiment_config(
            repo_root / config_rel,
            local_config_path,
            runtime_context={"run_id": run_id},
        )
        commands = build_runner_commands(config)
        manifest = make_dry_run_manifest(
            config,
            commands=commands,
            local_paths=validate_experiment_config(config, repo_root=repo_root),
            run_id=run_id,
            cli_args=type("Args", (), {"dry_run": True, "write_plan": False})(),
        )
        trace = attach_replication_preflight(
            manifest, config, repo_root=repo_root, index_path=index_path
        )["artifact_trace"]
        preflight = trace["replication_preflight"]
        return {
            **base,
            "command_count": len(commands),
            "observed_input_seeds": list(preflight["seeds"]),
            "current_group_count": int(preflight["current_group_count"]),
            "validation_group_count": int(preflight["validation_group_count"]),
            "passed": True,
            "errors": [],
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {**base, "passed": False, "errors": [str(exc)]}


def audit_study_surfaces(
    *, repo_root: Path, index_path: Path, local_config_path: Path
) -> dict[str, Any]:
    """Audit indexed studies through the same binding/preflight path as launch."""

    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    raw_surfaces = [dict(item) for item in index.get("study_surfaces") or []]
    declared = [
        str(entry.get("config") or "")
        for surface in raw_surfaces
        for entry in surface.get("entries") or []
        if isinstance(entry, Mapping)
    ]
    actual = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "configs/studies").glob("*.yaml")
    }
    duplicate_configs = sorted(
        path for path, count in Counter(declared).items() if path and count > 1
    )
    inventory_errors = []
    if set(declared) != actual:
        inventory_errors.append(
            f"study config inventory drift: missing={sorted(actual - set(declared))}, "
            f"unexpected={sorted(set(declared) - actual)}"
        )
    if duplicate_configs:
        inventory_errors.append(f"duplicate study configs: {duplicate_configs}")
    try:
        bindings = [
            item
            for item in _indexed_auxiliary_bindings(index)
            if item["expected_command_count"] is not None and not item.get("source_only")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        inventory_errors.append(str(exc))
        bindings = []

    entry_reports = [
        _audit_study_binding(
            binding,
            repo_root=repo_root,
            index_path=index_path,
            local_config_path=local_config_path,
        )
        for binding in bindings
    ]
    surface_reports = []
    for surface in raw_surfaces:
        entries = [
            report
            for binding, report in zip(bindings, entry_reports)
            if binding["surface"]["name"] == surface.get("name")
        ]
        errors = [
            f"{entry['role']}: {error}"
            for entry in entries
            for error in entry["errors"]
        ]
        surface_reports.append(
            {
                "name": surface.get("name"),
                "status": surface.get("status"),
                "input_contract_ref": surface.get("input_contract_ref"),
                "entry_count": len(entries),
                "contract_passed": len(entries) == len(surface.get("entries") or [])
                and not errors,
                "entries": entries,
                "errors": errors,
            }
        )
    passed = (
        not inventory_errors
        and bool(surface_reports)
        and all(item["contract_passed"] for item in surface_reports)
    )
    return {
        "schema_version": 1,
        "surface_count": len(surface_reports),
        "entry_count": sum(item["entry_count"] for item in surface_reports),
        "contract_passed": passed,
        "failed_count": len(inventory_errors)
        + sum(not item["contract_passed"] for item in surface_reports),
        "inventory_errors": inventory_errors,
        "surfaces": surface_reports,
    }


def audit_replication_surfaces(
    *,
    repo_root: Path,
    index_path: Path,
    local_config_path: Path,
) -> dict[str, Any]:
    """Audit non-canonical surfaces without expanding canonical managed experiments."""

    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    surface_reports: list[dict[str, Any]] = []
    for raw_surface in index.get("replication_surfaces") or []:
        surface = dict(raw_surface)
        pipeline = surface.get("pipeline_contract") or {}
        stage_order = tuple(str(stage) for stage in pipeline.get("stage_order") or [])
        producer_stage = str(pipeline.get("producer_stage") or "")
        consumer_stages = tuple(str(stage) for stage in pipeline.get("consumer_stages") or [])
        matrix_contract = pipeline.get("matrix") or {}
        centers = tuple(str(center) for center in matrix_contract.get("centers") or [])
        arms = tuple(str(arm) for arm in matrix_contract.get("arms") or [])
        surface_errors: list[str] = []
        shared_run_id = f"audit_{surface.get('name')}_shared_run_id"
        if not stage_order or producer_stage not in stage_order or not centers or not arms:
            surface_errors.append("pipeline_contract must declare stages, producer, centers, and arms")
        if set(consumer_stages) != set(stage_order).difference({producer_stage}):
            surface_errors.append("pipeline_contract consumer_stages must be every non-producer stage")
        if int(surface.get("command_count_per_stage") or 0) != len(centers) * len(arms):
            surface_errors.append("command_count_per_stage does not match centers x arms")
        source_report, source_children = _audit_replication_source_config(
            surface,
            repo_root=repo_root,
            local_config_path=local_config_path,
            run_id=shared_run_id,
        )
        if source_report is not None:
            surface_errors.extend(
                f"source_config: {error}" for error in source_report["errors"]
            )
        replicate_reports: list[dict[str, Any]] = []
        producers_by_seed: dict[int, set[str]] = {}
        outputs_by_seed: dict[int, set[str]] = {}
        input_roots: set[str] = set()
        materialization_groups: list[dict[str, Any]] = []
        validation_groups: list[dict[str, Any]] = []
        for raw_replicate in surface.get("replicates") or []:
            replicate = dict(raw_replicate)
            seed = int(replicate.get("seed"))
            run_id = shared_run_id
            stage_reports: list[dict[str, Any]] = []
            configs: dict[str, dict[str, Any]] = {}
            commands_by_stage: dict[str, list[dict[str, Any]]] = {}
            manifests_by_stage: dict[str, dict[str, Any]] = {}
            seed_errors: list[str] = []
            for stage in stage_order:
                config_rel = str((replicate.get("stages") or {}).get(stage) or "")
                config_path = repo_root / config_rel
                tracked = _git_tracked(repo_root, config_rel)
                try:
                    config = load_experiment_config(
                        config_path,
                        local_config_path,
                        runtime_context={"run_id": run_id},
                    )
                    validate_experiment_config(config, repo_root=repo_root)
                    commands = build_runner_commands(config)
                    grid_audit = audit_replication_command_grid(
                        commands,
                        centers=centers,
                        arms=arms,
                    )
                    grid_passed = bool(grid_audit["passed"])
                    config_seed = int(config["paper_protocol"]["kshot"]["subset_seed"])
                    if not grid_passed:
                        seed_errors.append(f"{stage}: command grid does not match the declared center x arm matrix")
                    if config_seed != seed:
                        seed_errors.append(f"{stage}: config subset_seed={config_seed}, expected {seed}")
                    if not tracked:
                        seed_errors.append(f"{stage}: config is not tracked: {config_rel}")
                    stage_manifest = make_dry_run_manifest(
                        config,
                        commands=commands,
                        local_paths=validate_experiment_config(config, repo_root=repo_root),
                        run_id=run_id,
                        cli_args=type(
                            "Args", (), {"dry_run": True, "write_plan": False}
                        )(),
                    )
                    stage_manifest = attach_replication_preflight(
                        stage_manifest,
                        config,
                        repo_root=repo_root,
                        index_path=index_path,
                    )
                    command_input_contract = stage_manifest["artifact_trace"][
                        "replication_preflight"
                    ]["command_input_contract"]
                    configs[stage], commands_by_stage[stage] = config, commands
                    manifests_by_stage[stage] = stage_manifest
                    stage_reports.append(
                        {
                            "stage": stage,
                            "config": config_rel,
                            "config_exists": config_path.is_file(),
                            "config_tracked_by_git": tracked,
                            "command_count": len(commands),
                            "exact_grid_passed": grid_passed,
                            "grid_audit": grid_audit,
                            "seed": config_seed,
                            "command_input_contract": command_input_contract,
                        }
                    )
                except (KeyError, OSError, ValueError) as exc:
                    seed_errors.append(f"{stage}: {exc}")
                    stage_reports.append(
                        {
                            "stage": stage,
                            "config": config_rel,
                            "config_exists": config_path.is_file(),
                            "config_tracked_by_git": tracked,
                            "command_count": 0,
                            "exact_grid_passed": False,
                            "command_input_contract": {
                                "passed": False,
                                "per_command_passed": False,
                            },
                            "error": str(exc),
                        }
                    )

            producer_consumer_match = False
            source_checkpoint_binding: dict[str, Any] | None = None
            preflight = {"passed": False, "missing": [], "identity_errors": []}
            if producer_stage and set(configs) == set(stage_order):
                train_manifest = manifests_by_stage[producer_stage]
                children = train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
                binding_audit = audit_replication_producer_consumers(
                    children,
                    {stage: commands_by_stage[stage] for stage in consumer_stages},
                )
                producers_by_seed[seed] = set(binding_audit["producer_dirs"])
                outputs_by_seed[seed] = set(binding_audit["consumer_output_paths"])
                producer_consumer_match = bool(binding_audit["passed"])
                seed_errors.extend(binding_audit["errors"])
                if source_report is not None:
                    source_checkpoint_binding = audit_replication_source_checkpoint_binding(
                        source_children,
                        commands_by_stage[producer_stage],
                    )
                    seed_errors.extend(source_checkpoint_binding["errors"])
                groups = train_manifest["artifact_trace"]["inputs"]["replication_k500_groups"]
                current_validation_groups = train_manifest["artifact_trace"]["inputs"][
                    "replication_validation_groups"
                ]
                if not validation_groups:
                    validation_groups = list(current_validation_groups)
                elif [group["base"] for group in validation_groups] != [
                    group["base"] for group in current_validation_groups
                ]:
                    seed_errors.append("replication validation group scope changed between seeds")
                if groups:
                    input_roots.add(str(Path(groups[0]["base"]).parents[3]))
                    materialization_groups.extend(groups)
                preflight = verify_replication_k500_groups(
                    groups,
                    mapping_version=str(configs[producer_stage]["paper_protocol"]["mapping_version"]),
                    mapping_hash=str(configs[producer_stage]["paper_protocol"]["mapping_hash"]),
                )
            if not producer_consumer_match:
                seed_errors.append("consumer model_dir paths do not match the seed's train producers")

            observed_status = "present_ready" if preflight["passed"] else "materialization_required"
            declared_status = str(replicate.get("k500_input_status") or "")
            declared_ready = _declared_ready(declared_status)
            status_consistent = declared_ready is not None and declared_ready == bool(preflight["passed"])
            status_errors = [] if status_consistent else [
                f"declared k500_input_status={declared_status!r}, observed={observed_status!r}"
            ]
            replicate_reports.append(
                {
                    "seed": seed,
                    "canonical": bool(replicate.get("canonical")),
                    "declared_k500_input_status": declared_status,
                    "observed_k500_input_status": observed_status,
                    "status_consistent": status_consistent,
                    "status_errors": status_errors,
                    "stages": stage_reports,
                    "producer_consumer_paths_match": producer_consumer_match,
                    "source_checkpoint_binding": source_checkpoint_binding,
                    "k500_preflight": preflight,
                    "contract_passed": not seed_errors and status_consistent,
                    "execution_ready": not seed_errors and bool(preflight["passed"]),
                    "errors": seed_errors,
                }
            )

        path_isolation = audit_replication_path_isolation(producers_by_seed, outputs_by_seed)
        path_isolation_passed = bool(path_isolation["passed"])
        surface_errors.extend(path_isolation["errors"])
        validation_report = _audit_validation_report(
            surface,
            input_roots=input_roots,
            groups=validation_groups or materialization_groups,
        )
        surface_errors.extend(validation_report["errors"])
        contract_passed = not surface_errors and all(item["contract_passed"] for item in replicate_reports)
        surface_reports.append(
            {
                "name": surface.get("name"),
                "status": surface.get("status"),
                "source_config": source_report,
                "path_isolation_passed": path_isolation_passed,
                "contract_passed": contract_passed,
                "execution_ready": contract_passed and all(item["execution_ready"] for item in replicate_reports),
                "validation_report": validation_report,
                "replicates": replicate_reports,
                "errors": surface_errors,
            }
        )
    return {
        "schema_version": 1,
        "surface_count": len(surface_reports),
        "contract_passed": all(item["contract_passed"] for item in surface_reports),
        "execution_ready": bool(surface_reports) and all(item["execution_ready"] for item in surface_reports),
        "failed_count": sum(1 for item in surface_reports if not item["contract_passed"]),
        "surfaces": surface_reports,
    }
