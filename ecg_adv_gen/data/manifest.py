"""Lightweight data path manifests for YAML-managed experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contracts import get_data_preprocess_contract
from .pn2021_index import (
    PN2021_HEADER_COUNT_PATTERN,
    PN2021_HEADER_COUNT_SCOPE,
    count_center_header_files,
)


class DataManifestError(ValueError):
    """Raised when a data path manifest violates safety or completeness rules."""


def _is_under(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
        return True
    except ValueError:
        return False


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _record_path(
    *,
    role: str,
    path: Path,
    boundary: Path | None,
    required: bool,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    exists = path.exists()
    record = {
        "role": role,
        "path": str(path),
        "required": bool(required),
        "kind": kind,
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "is_file": path.is_file() if exists else False,
        "under_write_boundary": None,
    }
    if boundary is not None:
        under = _is_under(path, boundary)
        record["under_write_boundary"] = under
        if not under:
            raise DataManifestError(f"{role} path is outside write boundary {boundary}: {path}")
    if metadata:
        record.update(metadata)
    return record


def _path_records(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ["roots", "cache_dirs", "pn2021_center_dirs", "pn2021_leak_excluded_dirs"]:
        value = manifest.get(key)
        if isinstance(value, list):
            yield from value


def _count_bounded_header_files(path: Path) -> dict[str, Any]:
    metadata = {
        "record_count_source": "bounded_depth_header_files",
        "record_count_pattern": PN2021_HEADER_COUNT_PATTERN,
        "record_count_scope": PN2021_HEADER_COUNT_SCOPE,
        "record_count_available": False,
        "n_header_files": None,
    }
    if not path.exists() or not path.is_dir():
        return metadata
    try:
        metadata["n_header_files"] = count_center_header_files(path)
        metadata["record_count_available"] = True
    except OSError as exc:
        metadata["record_count_error"] = str(exc)
    return metadata


def _center_metadata(
    *,
    center: str,
    target_centers: set[str],
    eval_centers: set[str],
    path: Path,
    include_counts: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "center": center,
        "in_target_4": center in target_centers,
        "in_eval_7": center in eval_centers,
    }
    if extra:
        metadata.update(extra)
    if include_counts:
        metadata.update(_count_bounded_header_files(path))
    return metadata


def _available_center_header_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(record["center"]): int(record["n_header_files"])
        for record in records
        if record.get("record_count_available") and record.get("n_header_files") is not None
    }


def build_data_path_manifest(
    config: dict[str, Any],
    *,
    local_paths: dict[str, str] | None = None,
    output_path: Path | None = None,
    include_counts: bool = False,
) -> dict[str, Any]:
    """Build a lightweight manifest without loading waveform data or scanning recursively."""
    contract = get_data_preprocess_contract()
    data = config.get("data") or {}
    roots = data.get("roots") or {}
    cache = data.get("cache") or {}
    preprocess = config.get("preprocess") or {}
    ecgtwin_decode = preprocess.get("ecgtwin_decode") or {}
    paper_protocol = config.get("paper_protocol") or {}
    centers = (paper_protocol.get("centers") or {})
    local_paths = local_paths or {}
    boundary = _as_path(local_paths["write_boundary"]) if local_paths.get("write_boundary") else None

    ptbxl_root = _as_path(roots.get("ptbxl"))
    pn2021_root = _as_path(roots.get("pn2021"))
    pn2021_training_root = pn2021_root / "training"
    pn2021_cache_dir = _as_path(cache.get("pn2021_cache_dir"))
    pn2021_mmap_cache_dir = _as_path(cache.get("pn2021_mmap_cache_dir"))

    target_centers = [str(c) for c in centers.get("target_4", contract.target_centers)]
    eval_centers = [str(c) for c in centers.get("eval_7", contract.eval_centers)]
    target_center_set = set(target_centers)
    eval_center_set = set(eval_centers)
    leak_excluded = [str(c) for c in data.get("exclude_centers", contract.leak_excluded_centers)]

    roots_records = [
        _record_path(role="ptbxl_root", path=ptbxl_root, boundary=boundary, required=True, kind="dir"),
        _record_path(role="pn2021_root", path=pn2021_root, boundary=boundary, required=True, kind="dir"),
        _record_path(
            role="pn2021_training_root",
            path=pn2021_training_root,
            boundary=boundary,
            required=True,
            kind="dir",
        ),
    ]
    cache_records = [
        _record_path(
            role="pn2021_cache_dir",
            path=pn2021_cache_dir,
            boundary=boundary,
            required=False,
            kind="dir",
        ),
        _record_path(
            role="pn2021_mmap_cache_dir",
            path=pn2021_mmap_cache_dir,
            boundary=boundary,
            required=False,
            kind="dir",
        ),
    ]
    center_records = []
    for center in eval_centers:
        center_path = pn2021_training_root / center
        center_records.append(
            _record_path(
                role="pn2021_center_dir",
                path=center_path,
                boundary=boundary,
                required=True,
                kind="dir",
                metadata=_center_metadata(
                    center=center,
                    target_centers=target_center_set,
                    eval_centers=eval_center_set,
                    path=center_path,
                    include_counts=include_counts,
                ),
            )
        )
    leak_records = []
    for center in leak_excluded:
        center_path = pn2021_training_root / center
        leak_records.append(
            _record_path(
                role="pn2021_leak_excluded_center_dir",
                path=center_path,
                boundary=boundary,
                required=False,
                kind="dir",
                metadata=_center_metadata(
                    center=center,
                    target_centers=target_center_set,
                    eval_centers=eval_center_set,
                    path=center_path,
                    include_counts=include_counts,
                    extra={"excluded_from_eval": True},
                ),
            )
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(output_path.resolve()) if output_path else "",
        "experiment_name": (config.get("experiment") or {}).get("name", ""),
        "entry_config": config.get("_entry_config", ""),
        "local_config": config.get("_local_config", ""),
        "contract": {
            "source_dataset": contract.source_dataset,
            "target_dataset": contract.target_dataset,
            "target_centers": list(contract.target_centers),
            "eval_centers": list(contract.eval_centers),
            "leak_excluded_centers": list(contract.leak_excluded_centers),
            "classifier_fs": contract.classifier_fs,
            "classifier_len": contract.classifier_len,
            "crop_len": contract.crop_len,
            "preprocess_mode": contract.preprocess_mode,
            "norm_mode": contract.norm_mode,
            "lead_order": contract.lead_order,
            "contract_id": contract.contract_id,
            "ecgtwin_decode_input_len": contract.ecgtwin_decode_input_len,
            "ecgtwin_decode_output_len": contract.ecgtwin_decode_output_len,
            "ecgtwin_to_ptbxl_indices": list(contract.ecgtwin_to_ptbxl_indices),
            "ecgfounder_preprocess_policy": contract.ecgfounder_preprocess_policy,
            "ecgfounder_input_fs": contract.ecgfounder_input_fs,
            "ecgfounder_input_len": contract.ecgfounder_input_len,
            "ecgfounder_input_shape": list(contract.ecgfounder_input_shape),
        },
        "config_values": {
            "source_dataset": data.get("source_dataset"),
            "target_dataset": data.get("target_dataset"),
            "pn2021_centers": data.get("pn2021_centers", []),
            "exclude_centers": leak_excluded,
            "target_centers": target_centers,
            "eval_centers": eval_centers,
            "classifier_fs": preprocess.get("classifier_fs"),
            "classifier_len": preprocess.get("classifier_len"),
            "crop_len": preprocess.get("crop_len"),
            "preprocess_mode": preprocess.get("mode"),
            "norm_mode": preprocess.get("norm_mode"),
            "lead_order": preprocess.get("lead_order"),
            "contract_id": preprocess.get("contract_id"),
            "ecgtwin_decode": {
                "input_len": ecgtwin_decode.get("input_len"),
                "output_len": ecgtwin_decode.get("output_len"),
                "reorder_indices": ecgtwin_decode.get("reorder_indices", []),
            },
            "ecgfounder": preprocess.get("ecgfounder", {}),
        },
        "counting": {
            "include_counts": bool(include_counts),
            "pn2021_header_count_pattern": PN2021_HEADER_COUNT_PATTERN,
            "pn2021_header_count_scope": PN2021_HEADER_COUNT_SCOPE,
            "loads_waveforms": False,
            "recursive_scan": False,
        },
        "roots": roots_records,
        "cache_dirs": cache_records,
        "pn2021_center_dirs": center_records,
        "pn2021_leak_excluded_dirs": leak_records,
    }
    required_missing = [
        r for r in _path_records(manifest)
        if r.get("required") and not r.get("exists")
    ]
    summary = {
        "n_required_paths": sum(1 for r in _path_records(manifest) if r.get("required")),
        "n_required_missing": len(required_missing),
        "required_missing_roles": [
            {"role": r["role"], "path": r["path"], "center": r.get("center", "")}
            for r in required_missing
        ],
        "target_center_dirs_present": sorted(
            r["center"] for r in center_records if r.get("in_target_4") and r.get("exists")
        ),
        "eval_center_dirs_present": sorted(
            r["center"] for r in center_records if r.get("exists")
        ),
    }
    if include_counts:
        eval_counts = _available_center_header_counts(center_records)
        target_counts = {
            center: count
            for center, count in eval_counts.items()
            if center in target_center_set
        }
        leak_counts = _available_center_header_counts(leak_records)
        summary.update({
            "pn2021_center_header_counts": eval_counts,
            "target_center_header_counts": target_counts,
            "leak_excluded_header_counts": leak_counts,
            "eval_center_header_count_total": sum(eval_counts.values()),
            "target_center_header_count_total": sum(target_counts.values()),
        })
    manifest["summary"] = summary
    return manifest


def validate_data_path_manifest(manifest: dict[str, Any], *, require_existing: bool = False) -> None:
    if require_existing:
        missing = [
            r for r in _path_records(manifest)
            if r.get("required") and not r.get("exists")
        ]
        if missing:
            first = missing[0]
            raise DataManifestError(
                f"Missing required data path: {first.get('role')} {first.get('path')}"
            )
    outside = [
        r for r in _path_records(manifest)
        if r.get("under_write_boundary") is False
    ]
    if outside:
        first = outside[0]
        raise DataManifestError(
            f"Data path outside write boundary: {first.get('role')} {first.get('path')}"
        )


def write_data_path_manifest(
    config: dict[str, Any],
    output_path: Path,
    *,
    local_paths: dict[str, str] | None = None,
    require_existing: bool = False,
    include_counts: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path).expanduser().resolve()
    manifest = build_data_path_manifest(
        config,
        local_paths=local_paths,
        output_path=output_path,
        include_counts=include_counts,
    )
    validate_data_path_manifest(manifest, require_existing=require_existing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
