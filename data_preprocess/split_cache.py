"""Create immutable PTB-XL and PN2021 split artifacts from validated caches.

The split layer stores only indices and stable record/hash identities. It never
copies waveforms and never depends on a model-specific latent cache. CPSC 2018
and CPSC 2018 Extra remain distinct physical sources in metadata but are one
logical ``cpsc_2018`` center for K500 sampling and ref-excluded evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.load_cache import (  # noqa: E402
    EXPECTED_CLASS_ORDER,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    ECGCache,
    load_cache,
)
from util.config_bundle import (  # noqa: E402
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.random_seed import (  # noqa: E402
    derive_seed,
    load_random_seed_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "splits.yaml"
EXPECTED_LOGICAL_CENTERS = (
    ("ningbo", ("ningbo",)),
    ("chapman_shaoxing", ("chapman_shaoxing",)),
    ("cpsc_2018", ("cpsc_2018", "cpsc_2018_extra")),
    ("georgia", ("georgia",)),
)
EXPECTED_QUALITY_STATUSES = ("clean", "repaired")
TUNING_TRAIN_PARTITION = "k500_tune_train"
TUNING_VALIDATION_PARTITION = "k500_tune_validation"
TUNING_SPLIT_POLICY = "deterministic_multilabel_source_stratified"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_id_set_sha256(values: Iterable[str]) -> str:
    ordered = sorted(str(value) for value in values)
    payload = "".join(f"{value}\n" for value in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_yaml_mapping(path: str | Path, *, description: str) -> dict[str, Any]:
    resolved = resolve_entry_config_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a YAML mapping: {resolved}")
    return payload


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _validate_quality_statuses(value: Any, *, description: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list")
    statuses = [str(item) for item in value]
    if tuple(statuses) != EXPECTED_QUALITY_STATUSES:
        raise ValueError(
            f"{description} must be {list(EXPECTED_QUALITY_STATUSES)}, got {statuses}"
        )
    return statuses


def _validate_logical_centers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("pn2021.logical_centers must be a list")
    resolved: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in value:
        entry = _require_mapping(item, description="logical center")
        name = str(entry.get("name", ""))
        raw_sources = entry.get("source_centers")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"logical center {name!r} requires source_centers")
        sources = [str(source) for source in raw_sources]
        overlap = seen_sources.intersection(sources)
        if overlap:
            raise ValueError(
                f"physical source centers cannot belong to two logical centers: "
                f"{sorted(overlap)}"
            )
        seen_sources.update(sources)
        resolved.append({"name": name, "source_centers": sources})
    actual = tuple(
        (entry["name"], tuple(entry["source_centers"])) for entry in resolved
    )
    if actual != EXPECTED_LOGICAL_CENTERS:
        raise ValueError(
            "pn2021.logical_centers must preserve the four-center protocol with "
            "CPSC 2018 and CPSC 2018 Extra combined; "
            f"got {actual}"
        )
    return resolved


def _validate_tuning_split(value: Any, *, k: int) -> dict[str, Any]:
    tuning = _require_mapping(value, description="pn2021.tuning_split")
    expected_keys = {
        "parent_partition",
        "train_partition",
        "validation_partition",
        "train_count",
        "validation_count",
        "policy",
        "preserve_source_proportions",
        "singleton_positive_policy",
        "seed_namespace",
    }
    if set(tuning) != expected_keys:
        raise ValueError(
            "pn2021.tuning_split keys are incomplete or unexpected: "
            f"expected={sorted(expected_keys)}, actual={sorted(tuning)}"
        )
    if tuning["parent_partition"] != "k500":
        raise ValueError("pn2021.tuning_split parent_partition must be k500")
    if tuning["train_partition"] != TUNING_TRAIN_PARTITION:
        raise ValueError(
            f"pn2021.tuning_split train_partition must be {TUNING_TRAIN_PARTITION}"
        )
    if tuning["validation_partition"] != TUNING_VALIDATION_PARTITION:
        raise ValueError(
            "pn2021.tuning_split validation_partition must be "
            f"{TUNING_VALIDATION_PARTITION}"
        )
    counts: dict[str, int] = {}
    for key in ("train_count", "validation_count"):
        raw = tuning[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"pn2021.tuning_split {key} must be positive")
        counts[key] = int(raw)
    if counts["train_count"] + counts["validation_count"] != int(k):
        raise ValueError(
            "pn2021.tuning_split train_count + validation_count must equal k"
        )
    if tuning["policy"] != TUNING_SPLIT_POLICY:
        raise ValueError(
            f"pn2021.tuning_split policy must be {TUNING_SPLIT_POLICY}"
        )
    if tuning["preserve_source_proportions"] is not True:
        raise ValueError(
            "pn2021.tuning_split preserve_source_proportions must be true"
        )
    if tuning["singleton_positive_policy"] != "prefer_train":
        raise ValueError(
            "pn2021.tuning_split singleton_positive_policy must be prefer_train"
        )
    if not isinstance(tuning["seed_namespace"], str) or not tuning[
        "seed_namespace"
    ]:
        raise ValueError(
            "pn2021.tuning_split seed_namespace must be a non-empty string"
        )
    return dict(tuning)


def load_split_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked split policy."""

    config = _read_yaml_mapping(path, description="split config")
    if config.get("schema_version") != 1:
        raise ValueError("split config schema_version must be 1")
    random_seed = _require_mapping(
        config.get("random_seed"), description="random_seed"
    )
    output = _require_mapping(config.get("output"), description="output")
    ptbxl = _require_mapping(config.get("ptbxl"), description="ptbxl")
    pn2021 = _require_mapping(config.get("pn2021"), description="pn2021")

    if not str(random_seed.get("file", "")):
        raise ValueError("random_seed.file is required")
    if not str(random_seed.get("namespace", "")):
        raise ValueError("random_seed.namespace is required")
    if output.get("if_exists") != "error":
        raise ValueError("output.if_exists must be 'error'")
    if not str(output.get("root_dir", "")):
        raise ValueError("output.root_dir is required")

    if int(ptbxl.get("sampling_rate_hz", -1)) != 100:
        raise ValueError("ptbxl split source must be the 100 Hz cache")
    if ptbxl.get("all_zero_policy") != "exclude":
        raise ValueError("ptbxl.all_zero_policy must be 'exclude'")
    _validate_quality_statuses(
        ptbxl.get("eligible_quality_status"),
        description="ptbxl.eligible_quality_status",
    )
    folds = _require_mapping(ptbxl.get("folds"), description="ptbxl.folds")
    expected_folds = {
        "train": list(range(1, 9)),
        "validation": [9],
        "test": [10],
    }
    actual_folds = {
        name: [int(value) for value in folds.get(name, [])]
        for name in expected_folds
    }
    if actual_folds != expected_folds:
        raise ValueError(
            f"PTB-XL must use official folds 1-8/9/10, got {actual_folds}"
        )
    if not bool(ptbxl.get("require_patient_disjoint")):
        raise ValueError("ptbxl.require_patient_disjoint must be true")
    if not str(ptbxl.get("patient_id_column", "")):
        raise ValueError("ptbxl.patient_id_column is required")

    if int(pn2021.get("sampling_rate_hz", -1)) != 100:
        raise ValueError("pn2021 split source must be the 100 Hz cache")
    if pn2021.get("required_mapping_version") != PN2021_MAPPING_VERSION:
        raise ValueError("pn2021 required mapping version mismatch")
    if pn2021.get("required_mapping_hash") != PN2021_MAPPING_HASH:
        raise ValueError("pn2021 required mapping hash mismatch")
    _validate_quality_statuses(
        pn2021.get("eligible_quality_status"),
        description="pn2021.eligible_quality_status",
    )
    k = pn2021.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("pn2021.k must be a positive integer")
    if pn2021.get("k500_all_zero_policy") != "exclude":
        raise ValueError("pn2021.k500_all_zero_policy must be 'exclude'")
    sampling = _require_mapping(
        pn2021.get("sampling"), description="pn2021.sampling"
    )
    if sampling.get("policy") != "uniform_without_replacement":
        raise ValueError("PN2021 sampling must be uniform_without_replacement")
    if sampling.get("candidate_order") != "hash_id_lexicographic":
        raise ValueError("PN2021 candidates must be ordered by hash_id")
    if sampling.get("replacement") is not False:
        raise ValueError("PN2021 K sampling must not use replacement")
    pn2021["logical_centers"] = _validate_logical_centers(
        pn2021.get("logical_centers")
    )
    ignored = [str(value) for value in pn2021.get("ignored_source_centers", [])]
    if ignored != ["ptb", "st_petersburg_incart"]:
        raise ValueError(
            "pn2021.ignored_source_centers must be [ptb, st_petersburg_incart]"
        )
    hard_excluded = {
        str(value) for value in pn2021.get("hard_excluded_source_centers", [])
    }
    if hard_excluded != {"ptb-xl", "ptbxl"}:
        raise ValueError("PN2021 must hard-exclude ptb-xl and ptbxl")
    training = _require_mapping(
        pn2021.get("training"), description="pn2021.training"
    )
    if training != {
        "use_all_k500": True,
        "validation_split": False,
        "checkpoint_policy": "last_checkpoint_only",
    }:
        raise ValueError("PN2021 K500 must use all records and last checkpoint")
    pn2021["tuning_split"] = _validate_tuning_split(
        pn2021.get("tuning_split"), k=int(k)
    )
    evaluation = _require_mapping(
        pn2021.get("evaluation"), description="pn2021.evaluation"
    )
    if evaluation != {
        "exclude_k500_hashes": True,
        "emit_all_zero_kept": True,
        "emit_drop_all_zero": True,
    }:
        raise ValueError("PN2021 evaluation/ref-exclusion contract mismatch")
    return config


def _artifact(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    result = {
        "file": path.name,
        "sha256": _sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()
    return result


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    path = directory / name
    np.save(path, values, allow_pickle=False)
    return _artifact(path)


def _fixed_strings(values: Sequence[Any], *, width: int) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def _sorted_indices_by_hash(cache: ECGCache, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    hashes = np.asarray(cache.hash_ids[indices]).astype(str)
    order = np.argsort(hashes, kind="stable")
    return indices[order]


def _validate_cache_quality_status(cache: ECGCache, allowed: Sequence[str]) -> None:
    if "quality_status" not in cache.records.columns:
        raise ValueError(f"{cache.dataset} records lack quality_status")
    actual = set(cache.records["quality_status"].astype(str))
    unexpected = actual.difference(allowed)
    if unexpected:
        raise ValueError(
            f"{cache.dataset} cache contains non-eligible quality statuses: "
            f"{sorted(unexpected)}"
        )


def _write_identity_arrays(
    directory: Path,
    *,
    prefix: str,
    cache: ECGCache,
    indices: np.ndarray,
) -> dict[str, dict[str, Any]]:
    resolved = np.asarray(indices, dtype=np.int64)
    return {
        "indices": _save_array(directory, f"{prefix}_indices.npy", resolved),
        "hash_ids": _save_array(
            directory,
            f"{prefix}_hash_ids.npy",
            _fixed_strings(cache.hash_ids[resolved], width=64),
        ),
        "record_ids": _save_array(
            directory,
            f"{prefix}_record_ids.npy",
            _fixed_strings(cache.record_ids[resolved], width=128),
        ),
    }


def _build_ptbxl_split(
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    cache_dir = _resolve_project_path(config["cache_dir"])
    with load_cache(
        cache_dir,
        sampling_rate_hz=int(config["sampling_rate_hz"]),
        mode="mmap",
        validate_values="sample",
    ) as cache:
        if cache.dataset != "ptbxl":
            raise ValueError(f"expected PTB-XL cache, got {cache.dataset!r}")
        allowed = [str(value) for value in config["eligible_quality_status"]]
        _validate_cache_quality_status(cache, allowed)
        patient_column = str(config["patient_id_column"])
        required_columns = {"strat_fold", patient_column}
        missing = required_columns.difference(cache.records.columns)
        if missing:
            raise ValueError(f"PTB-XL records lack columns: {sorted(missing)}")
        folds = cache.records["strat_fold"].to_numpy()
        if not np.isfinite(folds.astype(float)).all():
            raise ValueError("PTB-XL strat_fold contains missing/non-finite values")
        folds = folds.astype(np.int64)
        if set(folds) != set(range(1, 11)):
            raise ValueError(
                f"PTB-XL cache must contain official folds 1..10, got {sorted(set(folds))}"
            )
        patients = cache.records[patient_column]
        if patients.isna().any() or (patients.astype(str).str.len() == 0).any():
            raise ValueError("PTB-XL patient identity is missing")
        nonzero = np.asarray(cache.labels).sum(axis=1) > 0
        fold_config = config["folds"]
        split_indices: dict[str, np.ndarray] = {}
        split_patients: dict[str, set[str]] = {}
        output_dir.mkdir()
        split_manifest: dict[str, Any] = {}
        for name in ("train", "validation", "test"):
            mask = np.isin(folds, np.asarray(fold_config[name], dtype=np.int64))
            indices = np.flatnonzero(mask & nonzero).astype(np.int64, copy=False)
            split_indices[name] = indices
            patient_set = set(patients.iloc[indices].astype(str))
            split_patients[name] = patient_set
            files = _write_identity_arrays(
                output_dir,
                prefix=name,
                cache=cache,
                indices=indices,
            )
            split_manifest[name] = {
                "folds": [int(value) for value in fold_config[name]],
                "record_count": int(indices.size),
                "patient_count": len(patient_set),
                "hash_id_set_sha256": _hash_id_set_sha256(
                    cache.hash_ids[indices].astype(str)
                ),
                "files": files,
            }
        assigned = np.concatenate(tuple(split_indices.values()))
        expected = np.flatnonzero(nonzero)
        if set(assigned.tolist()) != set(expected.tolist()):
            raise ValueError("PTB-XL eligible records were not assigned exactly once")
        overlap_counts = {
            "train__validation": len(
                split_patients["train"] & split_patients["validation"]
            ),
            "train__test": len(split_patients["train"] & split_patients["test"]),
            "validation__test": len(
                split_patients["validation"] & split_patients["test"]
            ),
        }
        if any(overlap_counts.values()):
            raise ValueError(f"PTB-XL patient leakage detected: {overlap_counts}")
        all_zero_indices = np.flatnonzero(~nonzero).astype(np.int64, copy=False)
        excluded_files = _write_identity_arrays(
            output_dir,
            prefix="excluded_all_zero",
            cache=cache,
            indices=all_zero_indices,
        )
        manifest = {
            "schema_version": 1,
            "dataset": "ptbxl",
            "split_id": str(config["split_id"]),
            "source_cache": cache.describe(),
            "source_manifest_sha256": cache.identity.manifest_sha256,
            "class_order": list(EXPECTED_CLASS_ORDER),
            "policy": {
                "folds": fold_config,
                "all_zero_policy": "exclude",
                "quality_status": allowed,
                "patient_id_column": patient_column,
                "patient_disjoint": True,
            },
            "source_record_count": len(cache),
            "eligible_record_count": int(nonzero.sum()),
            "excluded_all_zero_count": int((~nonzero).sum()),
            "excluded_all_zero_files": excluded_files,
            "patient_overlap_counts": overlap_counts,
            "splits": split_manifest,
        }
        manifest_path = output_dir / "split_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "split_id": manifest["split_id"],
            "manifest": str(manifest_path.relative_to(output_dir.parent)),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_manifest_sha256": cache.identity.manifest_sha256,
            "split_hashes": {
                name: split_manifest[name]["hash_id_set_sha256"]
                for name in split_manifest
            },
        }


def _source_counts(
    cache: ECGCache,
    indices: np.ndarray,
    source_centers: Sequence[str],
) -> dict[str, int]:
    centers = cache.records.iloc[np.asarray(indices, dtype=np.int64)][
        "center"
    ].astype(str)
    counts = centers.value_counts().to_dict()
    return {source: int(counts.get(source, 0)) for source in source_centers}


def _class_positive_counts(cache: ECGCache, indices: np.ndarray) -> dict[str, int]:
    labels = np.asarray(cache.labels[np.asarray(indices, dtype=np.int64)])
    positives = labels.astype(np.int64, copy=False).sum(axis=0)
    return {
        class_name: int(positives[position])
        for position, class_name in enumerate(EXPECTED_CLASS_ORDER)
    }


def _largest_remainder_quotas(
    counts: Mapping[str, int],
    *,
    selected_count: int,
    source_order: Sequence[str],
) -> dict[str, int]:
    total = sum(int(counts[source]) for source in source_order)
    if total <= 0 or not 0 <= int(selected_count) <= total:
        raise ValueError("invalid proportional source-quota request")
    raw = {
        source: int(counts[source]) * int(selected_count) / total
        for source in source_order
    }
    quotas = {source: int(np.floor(raw[source])) for source in source_order}
    remainder = int(selected_count) - sum(quotas.values())
    ranked = sorted(
        source_order,
        key=lambda source: (
            -(raw[source] - quotas[source]),
            source_order.index(source),
        ),
    )
    for source in ranked[:remainder]:
        quotas[source] += 1
    if sum(quotas.values()) != int(selected_count):
        raise RuntimeError("proportional source quotas do not sum to target")
    return quotas


def _multilabel_error(actual: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target.astype(np.float64, copy=False), 1.0)
    delta = (actual.astype(np.float64, copy=False) - target) / scale
    return float(np.dot(delta, delta))


def _build_k500_tuning_split(
    cache: ECGCache,
    *,
    parent_indices: np.ndarray,
    source_centers: Sequence[str],
    config: Mapping[str, Any],
    base_seed: int,
    split_id: str,
    logical_center: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive a deterministic train/validation split inside one fixed K set.

    Source-center validation counts use largest-remainder proportional quotas.
    Within those hard quotas, deterministic greedy selection plus same-source
    swaps minimizes marginal Super5 count error. Records carrying a class that
    appears only once in the parent K set are kept in training; this protects
    the single Georgia MI anchor without special-casing a center or class name.
    """

    parent = np.asarray(parent_indices, dtype=np.int64)
    train_count = int(config["train_count"])
    validation_count = int(config["validation_count"])
    if parent.ndim != 1 or parent.size != train_count + validation_count:
        raise ValueError("K500 tuning counts do not match the fixed parent selection")
    parent_hash = _hash_id_set_sha256(cache.hash_ids[parent].astype(str))
    effective_seed = derive_seed(
        str(config["seed_namespace"]),
        "pn2021",
        str(split_id),
        str(logical_center),
        parent_hash,
        f"train{train_count}",
        f"validation{validation_count}",
        base_seed=base_seed,
    )
    rng = np.random.RandomState(effective_seed)
    labels = np.asarray(cache.labels[parent], dtype=np.int64)
    if labels.shape != (parent.size, len(EXPECTED_CLASS_ORDER)):
        raise ValueError("K500 tuning labels violate the Super5 shape contract")
    if np.any(labels.sum(axis=1) == 0):
        raise ValueError("K500 tuning parent unexpectedly contains all-zero labels")
    parent_positive = labels.sum(axis=0)
    validation_target = np.floor(
        parent_positive * (validation_count / parent.size) + 0.5
    ).astype(np.int64)
    singleton_classes = np.flatnonzero(parent_positive == 1)
    protected = (
        labels[:, singleton_classes].any(axis=1)
        if singleton_classes.size
        else np.zeros(parent.size, dtype=bool)
    )

    source_values = (
        cache.records.iloc[parent]["center"].astype(str).to_numpy()
    )
    unexpected_sources = set(source_values).difference(source_centers)
    if unexpected_sources:
        raise ValueError(
            "K500 tuning parent contains unexpected physical sources: "
            f"{sorted(unexpected_sources)}"
        )
    parent_source_counts = {
        source: int(np.sum(source_values == source)) for source in source_centers
    }
    source_targets = _largest_remainder_quotas(
        parent_source_counts,
        selected_count=validation_count,
        source_order=source_centers,
    )
    for source in source_centers:
        eligible = int(np.sum((source_values == source) & ~protected))
        if eligible < source_targets[source]:
            raise ValueError(
                "singleton-positive train protection conflicts with the physical "
                f"source quota for {logical_center}/{source}: "
                f"eligible={eligible}, quota={source_targets[source]}"
            )

    tie_rank = np.empty(parent.size, dtype=np.int64)
    tie_rank[rng.permutation(parent.size)] = np.arange(parent.size)
    chosen = np.zeros(parent.size, dtype=bool)
    validation_positive = np.zeros(len(EXPECTED_CLASS_ORDER), dtype=np.int64)
    chosen_sources = {source: 0 for source in source_centers}
    for _ in range(validation_count):
        available_mask = ~chosen & ~protected
        forced_sources: list[str] = []
        for source in source_centers:
            remaining = source_targets[source] - chosen_sources[source]
            if remaining < 0:
                raise RuntimeError("physical source quota was exceeded")
            source_available = int(
                np.sum(available_mask & (source_values == source))
            )
            if source_available < remaining:
                raise RuntimeError("physical source quota became infeasible")
            if remaining and source_available == remaining:
                forced_sources.append(source)
        candidates = np.flatnonzero(
            available_mask
            & np.asarray(
                [
                    chosen_sources[source] < source_targets[source]
                    and (not forced_sources or source in forced_sources)
                    for source in source_values
                ],
                dtype=bool,
            )
        )
        if candidates.size == 0:
            raise RuntimeError("no eligible record remains for K500 validation")
        current_error = _multilabel_error(
            validation_positive, validation_target
        )
        best = min(
            candidates.tolist(),
            key=lambda position: (
                _multilabel_error(
                    validation_positive + labels[position], validation_target
                )
                - current_error,
                int(tie_rank[position]),
                str(cache.hash_ids[parent[position]]),
            ),
        )
        chosen[best] = True
        validation_positive += labels[best]
        chosen_sources[str(source_values[best])] += 1

    # Improve marginal label balance without changing any physical-source quota.
    for _ in range(parent.size):
        current_error = _multilabel_error(
            validation_positive, validation_target
        )
        best_swap: tuple[int, int] | None = None
        best_error = current_error
        best_tie: tuple[int, int, str, str] | None = None
        for outgoing in np.flatnonzero(chosen):
            incoming_candidates = np.flatnonzero(
                ~chosen
                & ~protected
                & (source_values == source_values[outgoing])
            )
            for incoming in incoming_candidates:
                candidate_positive = (
                    validation_positive - labels[outgoing] + labels[incoming]
                )
                candidate_error = _multilabel_error(
                    candidate_positive, validation_target
                )
                tie = (
                    int(tie_rank[incoming]),
                    int(tie_rank[outgoing]),
                    str(cache.hash_ids[parent[incoming]]),
                    str(cache.hash_ids[parent[outgoing]]),
                )
                if candidate_error < best_error - 1.0e-12 or (
                    abs(candidate_error - best_error) <= 1.0e-12
                    and candidate_error < current_error - 1.0e-12
                    and (best_tie is None or tie < best_tie)
                ):
                    best_error = candidate_error
                    best_swap = (int(outgoing), int(incoming))
                    best_tie = tie
        if best_swap is None:
            break
        outgoing, incoming = best_swap
        chosen[outgoing] = False
        chosen[incoming] = True
        validation_positive = (
            validation_positive - labels[outgoing] + labels[incoming]
        )

    validation = _sorted_indices_by_hash(cache, parent[chosen])
    train = _sorted_indices_by_hash(cache, parent[~chosen])
    train_hashes = set(cache.hash_ids[train].astype(str))
    validation_hashes = set(cache.hash_ids[validation].astype(str))
    parent_hashes = set(cache.hash_ids[parent].astype(str))
    overlap = train_hashes.intersection(validation_hashes)
    union = train_hashes.union(validation_hashes)
    if train.size != train_count or validation.size != validation_count:
        raise RuntimeError("K500 tuning split produced incorrect record counts")
    if overlap or union != parent_hashes:
        raise RuntimeError("K500 tuning split is not an exact parent partition")
    singleton_hashes = sorted(
        str(cache.hash_ids[parent[position]])
        for position in np.flatnonzero(protected)
    )
    if set(singleton_hashes).intersection(validation_hashes):
        raise RuntimeError("singleton-positive record leaked into validation")

    evidence = {
        "schema_version": 1,
        "policy": TUNING_SPLIT_POLICY,
        "parent_partition": "k500",
        "parent_count": int(parent.size),
        "parent_hash_id_set_sha256": parent_hash,
        "train_partition": str(config["train_partition"]),
        "train_count": int(train.size),
        "train_hash_id_set_sha256": _hash_id_set_sha256(
            cache.hash_ids[train].astype(str)
        ),
        "validation_partition": str(config["validation_partition"]),
        "validation_count": int(validation.size),
        "validation_hash_id_set_sha256": _hash_id_set_sha256(
            cache.hash_ids[validation].astype(str)
        ),
        "overlap_count": len(overlap),
        "union_count": len(union),
        "union_matches_parent": union == parent_hashes,
        "union_hash_id_set_sha256": _hash_id_set_sha256(union),
        "base_seed": int(base_seed),
        "effective_seed": int(effective_seed),
        "seed_namespace": str(config["seed_namespace"]),
        "seed_identity": [
            "pn2021",
            str(split_id),
            str(logical_center),
            parent_hash,
            f"train{train_count}",
            f"validation{validation_count}",
        ],
        "singleton_positive_policy": str(
            config["singleton_positive_policy"]
        ),
        "singleton_positive_classes": [
            EXPECTED_CLASS_ORDER[position] for position in singleton_classes
        ],
        "singleton_positive_hash_ids": singleton_hashes,
        "singleton_positive_all_kept_in_train": set(singleton_hashes).issubset(
            train_hashes
        ),
        "parent_class_positive_counts": _class_positive_counts(cache, parent),
        "train_class_positive_counts": _class_positive_counts(cache, train),
        "validation_class_positive_counts": _class_positive_counts(
            cache, validation
        ),
        "validation_class_target_counts": {
            class_name: int(validation_target[position])
            for position, class_name in enumerate(EXPECTED_CLASS_ORDER)
        },
        "parent_source_center_counts": parent_source_counts,
        "train_source_center_counts": _source_counts(
            cache, train, source_centers
        ),
        "validation_source_center_counts": _source_counts(
            cache, validation, source_centers
        ),
        "validation_source_target_counts": source_targets,
    }
    if evidence["validation_source_center_counts"] != source_targets:
        raise RuntimeError("K500 validation physical-source quotas were not preserved")
    return train, validation, evidence


def _build_pn2021_split(
    *,
    config: dict[str, Any],
    output_dir: Path,
    base_seed: int,
    seed_namespace: str,
) -> dict[str, Any]:
    cache_dir = _resolve_project_path(config["cache_dir"])
    with load_cache(
        cache_dir,
        sampling_rate_hz=int(config["sampling_rate_hz"]),
        mode="mmap",
        validate_values="sample",
    ) as cache:
        if cache.dataset != "pn2021":
            raise ValueError(f"expected PN2021 cache, got {cache.dataset!r}")
        if (
            cache.identity.mapping_version != config["required_mapping_version"]
            or cache.identity.mapping_hash != config["required_mapping_hash"]
        ):
            raise ValueError("PN2021 cache mapping identity does not match split config")
        allowed = [str(value) for value in config["eligible_quality_status"]]
        _validate_cache_quality_status(cache, allowed)
        available = set(cache.available_centers)
        hard_excluded = {
            str(value) for value in config["hard_excluded_source_centers"]
        }
        present_forbidden = available & hard_excluded
        if present_forbidden:
            raise ValueError(
                f"hard-excluded PN2021 centers are present: {sorted(present_forbidden)}"
            )
        configured_sources = {
            source
            for item in config["logical_centers"]
            for source in item["source_centers"]
        }
        ignored = {str(value) for value in config["ignored_source_centers"]}
        if available != configured_sources | ignored:
            raise ValueError(
                "PN2021 source-center accounting mismatch: "
                f"available={sorted(available)}, configured={sorted(configured_sources)}, "
                f"ignored={sorted(ignored)}"
            )
        output_dir.mkdir()
        all_centers = cache.records["center"].astype(str).to_numpy()
        nonzero = np.asarray(cache.labels).sum(axis=1) > 0
        k = int(config["k"])
        tuning_config = config["tuning_split"]
        center_manifests: dict[str, Any] = {}
        for item in config["logical_centers"]:
            logical_name = str(item["name"])
            source_centers = [str(value) for value in item["source_centers"]]
            center_dir = output_dir / logical_name
            center_dir.mkdir()
            center_mask = np.isin(all_centers, np.asarray(source_centers))
            all_center_indices = np.flatnonzero(center_mask).astype(
                np.int64, copy=False
            )
            candidate_indices = np.flatnonzero(center_mask & nonzero).astype(
                np.int64, copy=False
            )
            candidate_indices = _sorted_indices_by_hash(cache, candidate_indices)
            if candidate_indices.size < k:
                raise ValueError(
                    f"logical center {logical_name} has only "
                    f"{candidate_indices.size} nonzero candidates for K={k}"
                )
            effective_seed = derive_seed(
                seed_namespace,
                "pn2021",
                str(config["split_id"]),
                logical_name,
                f"k{k}",
                base_seed=base_seed,
            )
            rng = np.random.RandomState(effective_seed)
            positions = rng.choice(candidate_indices.size, size=k, replace=False)
            selected_indices = _sorted_indices_by_hash(
                cache, candidate_indices[positions]
            )
            tuning_train_indices, tuning_validation_indices, tuning_evidence = (
                _build_k500_tuning_split(
                    cache,
                    parent_indices=selected_indices,
                    source_centers=source_centers,
                    config=tuning_config,
                    base_seed=base_seed,
                    split_id=str(config["split_id"]),
                    logical_center=logical_name,
                )
            )
            selected_mask = np.zeros(len(cache), dtype=bool)
            selected_mask[selected_indices] = True
            eval_kept_indices = _sorted_indices_by_hash(
                cache,
                np.flatnonzero(center_mask & ~selected_mask).astype(
                    np.int64, copy=False
                ),
            )
            eval_drop_indices = _sorted_indices_by_hash(
                cache,
                np.flatnonzero(center_mask & nonzero & ~selected_mask).astype(
                    np.int64, copy=False
                ),
            )
            if np.intersect1d(selected_indices, eval_kept_indices).size:
                raise ValueError(f"K500 ref leakage in {logical_name} kept view")
            if np.intersect1d(selected_indices, eval_drop_indices).size:
                raise ValueError(f"K500 ref leakage in {logical_name} drop view")

            files = {
                "candidate_nonzero": _write_identity_arrays(
                    center_dir,
                    prefix="candidate_nonzero",
                    cache=cache,
                    indices=candidate_indices,
                ),
                "k500": _write_identity_arrays(
                    center_dir,
                    prefix="k500",
                    cache=cache,
                    indices=selected_indices,
                ),
                TUNING_TRAIN_PARTITION: _write_identity_arrays(
                    center_dir,
                    prefix=TUNING_TRAIN_PARTITION,
                    cache=cache,
                    indices=tuning_train_indices,
                ),
                TUNING_VALIDATION_PARTITION: _write_identity_arrays(
                    center_dir,
                    prefix=TUNING_VALIDATION_PARTITION,
                    cache=cache,
                    indices=tuning_validation_indices,
                ),
                "evaluation_all_zero_kept": _write_identity_arrays(
                    center_dir,
                    prefix="evaluation_all_zero_kept",
                    cache=cache,
                    indices=eval_kept_indices,
                ),
                "evaluation_drop_all_zero": _write_identity_arrays(
                    center_dir,
                    prefix="evaluation_drop_all_zero",
                    cache=cache,
                    indices=eval_drop_indices,
                ),
            }
            center_manifest = {
                "logical_center": logical_name,
                "source_centers": source_centers,
                "source_record_count": int(all_center_indices.size),
                "source_center_counts": _source_counts(
                    cache, all_center_indices, source_centers
                ),
                "all_zero_count": int((center_mask & ~nonzero).sum()),
                "candidate_nonzero_count": int(candidate_indices.size),
                "candidate_source_center_counts": _source_counts(
                    cache, candidate_indices, source_centers
                ),
                "candidate_hash_id_set_sha256": _hash_id_set_sha256(
                    cache.hash_ids[candidate_indices].astype(str)
                ),
                "base_seed": int(base_seed),
                "effective_seed": int(effective_seed),
                "seed_namespace": seed_namespace,
                "k500_count": int(selected_indices.size),
                "k500_source_center_counts": _source_counts(
                    cache, selected_indices, source_centers
                ),
                "k500_hash_id_set_sha256": _hash_id_set_sha256(
                    cache.hash_ids[selected_indices].astype(str)
                ),
                "k500_tune_train_count": int(tuning_train_indices.size),
                "k500_tune_train_hash_id_set_sha256": _hash_id_set_sha256(
                    cache.hash_ids[tuning_train_indices].astype(str)
                ),
                "k500_tune_validation_count": int(
                    tuning_validation_indices.size
                ),
                "k500_tune_validation_hash_id_set_sha256": _hash_id_set_sha256(
                    cache.hash_ids[tuning_validation_indices].astype(str)
                ),
                "k500_tuning_split": tuning_evidence,
                "evaluation_all_zero_kept_count": int(eval_kept_indices.size),
                "evaluation_all_zero_kept_hash_id_set_sha256": (
                    _hash_id_set_sha256(cache.hash_ids[eval_kept_indices].astype(str))
                ),
                "evaluation_drop_all_zero_count": int(eval_drop_indices.size),
                "evaluation_drop_all_zero_hash_id_set_sha256": (
                    _hash_id_set_sha256(cache.hash_ids[eval_drop_indices].astype(str))
                ),
                "files": files,
            }
            (center_dir / "split_manifest.json").write_text(
                json.dumps(center_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            center_manifest["manifest_sha256"] = _sha256_file(
                center_dir / "split_manifest.json"
            )
            center_manifests[logical_name] = center_manifest

        manifest = {
            "schema_version": 1,
            "dataset": "pn2021",
            "split_id": str(config["split_id"]),
            "source_cache": cache.describe(),
            "source_manifest_sha256": cache.identity.manifest_sha256,
            "mapping_version": cache.identity.mapping_version,
            "mapping_hash": cache.identity.mapping_hash,
            "class_order": list(EXPECTED_CLASS_ORDER),
            "policy": {
                "k": k,
                "quality_status": allowed,
                "k500_all_zero_policy": "exclude",
                "sampling": config["sampling"],
                "training": config["training"],
                "tuning_split": config["tuning_split"],
                "evaluation": config["evaluation"],
                "ignored_source_centers": config["ignored_source_centers"],
                "hard_excluded_source_centers": config[
                    "hard_excluded_source_centers"
                ],
                "logical_center_note": (
                    "cpsc_2018 combines physical sources cpsc_2018 and "
                    "cpsc_2018_extra; parent K sampling has no source quota, "
                    "while the internal validation split preserves the realized "
                    "parent source proportions"
                ),
            },
            "logical_center_order": [
                str(item["name"]) for item in config["logical_centers"]
            ],
            "logical_centers": center_manifests,
        }
        manifest_path = output_dir / "split_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "split_id": manifest["split_id"],
            "manifest": str(manifest_path.relative_to(output_dir.parent)),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_manifest_sha256": cache.identity.manifest_sha256,
            "k500_hashes": {
                name: entry["k500_hash_id_set_sha256"]
                for name, entry in center_manifests.items()
            },
            "k500_tuning_hashes": {
                name: {
                    "parent": entry["k500_hash_id_set_sha256"],
                    "train": entry["k500_tune_train_hash_id_set_sha256"],
                    "validation": entry[
                        "k500_tune_validation_hash_id_set_sha256"
                    ],
                }
                for name, entry in center_manifests.items()
            },
        }


def build_cache_splits(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Build the configured split artifacts atomically and return their root."""

    resolved_config_path = resolve_entry_config_path(config_path)
    config = load_split_config(resolved_config_path)
    seed_path = resolve_config_reference(
        config["random_seed"]["file"],
        owner_config_path=resolved_config_path,
        description="random_seed.file",
        must_exist=True,
    )
    base_seed = load_random_seed_config(seed_path).base_seed
    seed_namespace = str(config["random_seed"]["namespace"])
    output_root = _resolve_project_path(config["output"]["root_dir"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite split root: {output_root}")
    staging_root = output_root.with_name(f".{output_root.name}.building")
    if staging_root.exists():
        raise FileExistsError(f"split staging root already exists: {staging_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    try:
        config_snapshot = staging_root / "splits_config_snapshot.yaml"
        seed_snapshot = staging_root / "random_seed_snapshot.yaml"
        config_snapshot.write_text(
            resolved_config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        seed_snapshot.write_text(seed_path.read_text(encoding="utf-8"), encoding="utf-8")
        ptbxl_subdir = str(config["ptbxl"]["output_subdir"])
        pn2021_subdir = str(config["pn2021"]["output_subdir"])
        if not ptbxl_subdir or not pn2021_subdir or ptbxl_subdir == pn2021_subdir:
            raise ValueError("dataset output_subdir values must be unique and non-empty")
        ptbxl_summary = _build_ptbxl_split(
            config=config["ptbxl"], output_dir=staging_root / ptbxl_subdir
        )
        pn2021_summary = _build_pn2021_split(
            config=config["pn2021"],
            output_dir=staging_root / pn2021_subdir,
            base_seed=base_seed,
            seed_namespace=seed_namespace,
        )
        root_manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_type": "ecg_cache_splits",
            "config": {
                "source_path": str(resolved_config_path),
                "snapshot_file": config_snapshot.name,
                "sha256": _sha256_file(config_snapshot),
            },
            "random_seed": {
                "source_path": str(seed_path),
                "snapshot_file": seed_snapshot.name,
                "sha256": _sha256_file(seed_snapshot),
                "base_seed": int(base_seed),
                "namespace": seed_namespace,
                "derivation": "sha256_first_uint32_little_endian",
            },
            "datasets": {
                "ptbxl": ptbxl_summary,
                "pn2021": pn2021_summary,
            },
        }
        (staging_root / "split_manifest.json").write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_root.replace(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    print(f"[splits] artifacts written to {output_root}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PTB-XL official-fold and PN2021 K500 split artifacts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="tracked YAML containing every split parameter",
    )
    args = parser.parse_args()
    build_cache_splits(args.config)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "build_cache_splits", "load_split_config"]
