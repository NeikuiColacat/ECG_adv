"""Non-canonical replication path, materialization, and audit contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ecg_adv_gen.data.kshot_artifacts import (
    canonical_kshot_base,
    read_ref_meta_record_ids,
)

from .adapters.common import argv_option_map, opt_first
from .loader import (
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)


FIXED_K_SAMPLING_POLICY = "deterministic_random_from_v7_nonzero_eligible_pool"
SUPER5_CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")


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
        "validation_seeds": seeds,
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


def _indexed_auxiliary_bindings(index: Mapping[str, Any]):
    contracts = _replication_surface_contracts(index)
    for raw_surface in index.get("replication_surfaces") or []:
        surface = dict(raw_surface)
        contract = contracts[str(surface["name"])]
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
    family = str(binding["input_contract"]["root_family"])
    if subset_root.name != "subsets" or subset_root.parent.name != family:
        raise ValueError(
            f"indexed auxiliary input root drift for {binding['stage']}: "
            f"{subset_root} does not bind {family}"
        )

    trace_inputs = (((manifest or {}).get("artifact_trace") or {}).get("inputs") or {})
    refs = trace_inputs.get("k500_refs") or []
    expected_count = binding.get("expected_command_count")
    if expected_count is not None and not refs:
        raise ValueError(f"indexed study {binding['stage']} has no manifest K500 refs")
    observed_seeds = sorted(
        {int(ref["seed"]) for ref in refs if isinstance(ref, Mapping) and ref.get("seed") is not None}
    )
    if not observed_seeds:
        kshot = (config.get("paper_protocol") or {}).get("kshot") or {}
        observed_seeds = sorted(
            {int(kshot.get(key, -1)) for key in ("seed", "subset_seed")}
        )
    if observed_seeds != binding["input_seeds"]:
        raise ValueError(
            f"indexed auxiliary input seeds drift for {binding['stage']}: "
            f"{observed_seeds} != {binding['input_seeds']}"
        )
    if expected_count is not None:
        observed_count = len((manifest or {}).get("commands") or [])
        if observed_count != int(expected_count):
            raise ValueError(
                f"indexed study command count drift for {binding['stage']}: "
                f"{observed_count} != {expected_count}"
            )
    return binding


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
    groups = _materialization_groups_for_seeds(
        contract,
        subset_root=subset_root,
        centers=centers,
        k=k,
        seeds=input_seeds,
    )
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
            model_dir = str(opt_first(options, "--model_dir"))
            if producers.get(key) != model_dir:
                errors.append(
                    f"{stage}:{key} model_dir={model_dir!r}, producer={producers.get(key)!r}"
                )
            outputs.add(str(opt_first(options, "--output_path")))
    return {
        "passed": not errors,
        "producer_count": len(producers),
        "producer_dirs": sorted(producers.values()),
        "consumer_output_paths": sorted(outputs),
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
            if item["expected_command_count"] is not None
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
        if not stage_order or producer_stage not in stage_order or not centers or not arms:
            surface_errors.append("pipeline_contract must declare stages, producer, centers, and arms")
        if set(consumer_stages) != set(stage_order).difference({producer_stage}):
            surface_errors.append("pipeline_contract consumer_stages must be every non-producer stage")
        if int(surface.get("command_count_per_stage") or 0) != len(centers) * len(arms):
            surface_errors.append("command_count_per_stage does not match centers x arms")
        replicate_reports: list[dict[str, Any]] = []
        producers_by_seed: dict[int, set[str]] = {}
        outputs_by_seed: dict[int, set[str]] = {}
        input_roots: set[str] = set()
        materialization_groups: list[dict[str, Any]] = []
        for raw_replicate in surface.get("replicates") or []:
            replicate = dict(raw_replicate)
            seed = int(replicate.get("seed"))
            run_id = f"audit_{surface.get('name')}_shared_run_id"
            stage_reports: list[dict[str, Any]] = []
            configs: dict[str, dict[str, Any]] = {}
            commands_by_stage: dict[str, list[dict[str, Any]]] = {}
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
                    configs[stage], commands_by_stage[stage] = config, commands
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
                            "error": str(exc),
                        }
                    )

            producer_consumer_match = False
            preflight = {"passed": False, "missing": [], "identity_errors": []}
            if producer_stage and set(configs) == set(stage_order):
                train_manifest = make_dry_run_manifest(
                    configs[producer_stage],
                    commands=commands_by_stage[producer_stage],
                    local_paths=validate_experiment_config(configs[producer_stage], repo_root=repo_root),
                    run_id=run_id,
                    cli_args=type("Args", (), {"dry_run": True, "write_plan": False})(),
                )
                train_manifest = attach_replication_preflight(
                    train_manifest,
                    configs[producer_stage],
                    repo_root=repo_root,
                    index_path=index_path,
                )
                children = train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
                binding_audit = audit_replication_producer_consumers(
                    children,
                    {stage: commands_by_stage[stage] for stage in consumer_stages},
                )
                producers_by_seed[seed] = set(binding_audit["producer_dirs"])
                outputs_by_seed[seed] = set(binding_audit["consumer_output_paths"])
                producer_consumer_match = bool(binding_audit["passed"])
                seed_errors.extend(binding_audit["errors"])
                groups = train_manifest["artifact_trace"]["inputs"]["replication_k500_groups"]
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
            groups=materialization_groups,
        )
        surface_errors.extend(validation_report["errors"])
        contract_passed = not surface_errors and all(item["contract_passed"] for item in replicate_reports)
        surface_reports.append(
            {
                "name": surface.get("name"),
                "status": surface.get("status"),
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
