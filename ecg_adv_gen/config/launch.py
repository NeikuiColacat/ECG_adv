"""Execution safety gates for YAML-managed experiment launches."""

from __future__ import annotations

import json
import os
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ecg_adv_gen.evaluation.selection import (
    ALLOWED_SELECTION_DATA,
    FORBIDDEN_SELECTION_REFERENCES,
    has_forbidden_selection_reference,
    validate_selection_policy,
)
from ecg_adv_gen.evaluation.pn2021c_metadata import validate_target_init_k500_identity

from .adapters.common import argv_option_map, opt_first
from .paths import PathSafetyError, is_under
from .replication import verify_replication_k500_groups, verify_replication_validation_report


class LaunchError(RuntimeError):
    """Raised when launching a managed experiment command would be unsafe."""


def require_cuda_visible_devices(env: dict[str, str] | None = None, *, allow_multi_gpu: bool = False) -> str:
    env = env or os.environ
    value = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        raise LaunchError(
            "CUDA_VISIBLE_DEVICES must be set explicitly before --execute. "
            "Check nvidia-smi and select intended free GPU(s)."
        )
    if value.lower() in {"all", "-1"}:
        raise LaunchError(f"CUDA_VISIBLE_DEVICES={value!r} is too broad for this shared server")
    if not allow_multi_gpu and "," in value:
        raise LaunchError(
            f"CUDA_VISIBLE_DEVICES={value!r} selects multiple GPUs; "
            "use a single GPU for managed shared-server runs"
        )
    return value


def check_nvidia_smi(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = runner(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise LaunchError("nvidia-smi is required before GPU launch but was not found") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise LaunchError(f"nvidia-smi failed before launch: {stderr or proc.returncode}")
    rows = []
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, mem_used, mem_total, util = parts
        rows.append(
            {
                "index": idx,
                "name": name,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "utilization_gpu_pct": util,
            }
        )
    if not rows:
        raise LaunchError("nvidia-smi returned no GPU rows")
    return {"command": cmd, "gpus": rows}


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchError(f"Expected existing manifest for resume/force: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"Existing manifest is not valid JSON: {path}") from exc


def _non_hidden_entries(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.iterdir() if p.name not in {".", ".."}]


def default_run_dir(local_paths: dict[str, str], run_id: str, *, dry_run: bool) -> Path:
    base = Path(local_paths["output_root"])
    if dry_run:
        return (base / "dry_runs" / run_id).resolve()
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    return (base / date_tag / run_id).resolve()


def prepare_output_dir(
    out_dir: Path,
    *,
    local_paths: dict[str, str],
    run_id: str,
    config_hash: str,
    resume: bool = False,
    force: bool = False,
) -> Path:
    out_dir = out_dir.expanduser().resolve()
    boundary = Path(local_paths["write_boundary"]).resolve()
    if not is_under(out_dir, boundary):
        raise PathSafetyError(f"output_dir={out_dir} is outside write boundary {boundary}")

    entries = _non_hidden_entries(out_dir)
    if entries:
        manifest_path = out_dir / "run_manifest.json"
        if not (resume or force):
            raise LaunchError(
                f"Refusing to use non-empty output directory without --resume or --force: {out_dir}"
            )
        existing = _load_manifest(manifest_path)
        if existing.get("run_id") != run_id:
            raise LaunchError(
                f"Existing manifest run_id={existing.get('run_id')!r} does not match {run_id!r}"
            )
        if existing.get("config_hash_sha256") != config_hash:
            raise LaunchError("Existing manifest config hash does not match current resolved config")
        if force and existing.get("status") not in {"dry_run", "launch_prepared", "failed"}:
            raise LaunchError(
                f"--force is only allowed for dry_run, launch_prepared, or failed runs; "
                f"existing status={existing.get('status')!r}"
            )
    return out_dir


def update_manifest_file(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_manifest(path)
    manifest.update(patch)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_seconds(started_at: str | None, finished_at: str) -> float | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(str(started_at))
        finish = datetime.fromisoformat(str(finished_at))
    except (TypeError, ValueError):
        return None
    return max(0.0, (finish - start).total_seconds())


def _final_success_patch(manifest: dict[str, Any], *, finished_at: str) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "status": "succeeded",
        "finished_at_utc": finished_at,
    }
    duration = _duration_seconds(manifest.get("execution_started_at_utc"), finished_at)
    if duration is not None:
        patch["duration_seconds"] = duration
    return patch


def _append_manifest_record(path: Path, key: str, record: dict[str, Any], patch: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = _load_manifest(path)
    records = list(manifest.get(key) or [])
    records.append(record)
    update = dict(patch or {})
    update[key] = records
    return update_manifest_file(path, update)


def _extract_mapping_metadata(result_json: dict[str, Any]) -> dict[str, Any] | None:
    label_mapping = result_json.get("label_mapping")
    if not isinstance(label_mapping, dict):
        return None
    mapping = label_mapping.get("pn2021_super5")
    return mapping if isinstance(mapping, dict) else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_size_sha_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256_file(path)
    return record


def _extract_ref_record_ids(meta: dict[str, Any], path: Path) -> list[str]:
    for key in ["ref_record_ids", "record_ids", "selected_ref_record_ids"]:
        value = meta.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    raise LaunchError(f"K500 ref meta has no ref_record_ids list: {path}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise LaunchError(f"{label} root must be a JSON object: {path}")
    return data


def _k500_ref_entries_by_center(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    trace = manifest.get("artifact_trace") or {}
    refs = ((trace.get("inputs") or {}).get("k500_refs") or [])
    out: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        center = str(ref.get("center", ""))
        if center:
            out.setdefault(center, []).append(ref)
    return out


def _k500_centers_for_manifest(manifest: dict[str, Any]) -> list[str]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    centers: list[str] = []
    for child in expected.get("child_runs") or []:
        center = str(child.get("center") or "")
        if center and center not in centers:
            centers.append(center)
    if centers:
        return centers
    paper = manifest.get("paper_protocol") or {}
    return list((paper.get("centers") or {}).get("target_4") or [])


def build_k500_ref_ids_artifact(manifest: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
    """Build a frozen K500 record-id artifact from traced ref-meta files."""
    paper = manifest.get("paper_protocol") or {}
    kshot = paper.get("kshot") or {}
    metrics = (manifest.get("artifact_trace") or {}).get("metrics") or {}
    expected_k = int(kshot.get("k", 500))
    expected_seed = int(kshot.get("subset_seed", kshot.get("seed", 0)))
    refs_by_center = _k500_ref_entries_by_center(manifest)
    target_centers = _k500_centers_for_manifest(manifest)
    centers: dict[str, Any] = {}

    for center in target_centers:
        entries = refs_by_center.get(center) or []
        if not entries:
            raise LaunchError(f"No K500 ref-meta entry traced for center={center}")
        first_ref = entries[0]
        meta_record = first_ref.get("ref_meta_json") or {}
        meta_path = Path(str(meta_record.get("path", "")))
        meta = _read_json_object(meta_path, label=f"K500 ref meta for {center}")
        ids_ordered = _extract_ref_record_ids(meta, meta_path)
        ids_sorted = sorted(ids_ordered)
        if len(ids_ordered) != expected_k:
            raise LaunchError(
                f"K500 ref meta for {center} has {len(ids_ordered)} ids, expected {expected_k}"
            )
        if len(set(ids_ordered)) != expected_k:
            raise LaunchError(f"K500 ref meta for {center} contains duplicate ref_record_ids")
        meta_k = meta.get("K", meta.get("k", expected_k))
        if int(meta_k) != expected_k:
            raise LaunchError(f"K500 ref meta for {center} has K={meta_k}, expected {expected_k}")
        meta_seed = meta.get("selection_seed", meta.get("subset_seed", meta.get("seed", expected_seed)))
        if int(meta_seed) != expected_seed:
            raise LaunchError(
                f"K500 ref meta for {center} has selection seed={meta_seed}, expected {expected_seed}"
            )

        per_consumer = []
        for entry in entries:
            per_consumer.append(
                {
                    "anchor_base": entry.get("anchor_base"),
                    "ref_meta_json": (entry.get("ref_meta_json") or {}).get("path"),
                    "signals_npz": (entry.get("signals_npz") or {}).get("path"),
                    "latent_npz": (
                        ((entry.get("latent_npz") or {}).get("path"))
                        if entry.get("latent_npz")
                        else None
                    ),
                }
            )
        signals_path = Path(str((first_ref.get("signals_npz") or {}).get("path", "")))
        latent_record = first_ref.get("latent_npz")
        latent_path = Path(str((latent_record or {}).get("path", ""))) if latent_record else None
        centers[center] = {
            "center": center,
            "k": expected_k,
            "selection_seed": expected_seed,
            "selection_policy": meta.get("policy") or (paper.get("selection") or {}).get("policy"),
            "ref_record_ids_ordered": ids_ordered,
            "ref_record_ids_sorted": ids_sorted,
            "ref_record_ids_sha256": _sha256_lines(ids_sorted),
            "source_indices": meta.get("source_indices"),
            "source_ref_meta_path": str(meta_path),
            "source_ref_meta_sha256": _sha256_file(meta_path),
            "signals_npz": _path_size_sha_record(signals_path),
            "latent_npz": _path_size_sha_record(latent_path),
            "parent_provenance": meta.get("parent"),
            "consumers": per_consumer,
        }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id"),
        "artifact_path": str(output_path),
        "paper_protocol": {
            "mapping_version": metrics.get("mapping_version"),
            "mapping_hash": metrics.get("mapping_hash"),
            "class_order": metrics.get("class_order"),
            "target_centers": target_centers,
            "k": expected_k,
            "seed": int(kshot.get("seed", expected_seed)),
            "subset_seed": expected_seed,
            "exclude_refs_from_eval": bool(kshot.get("exclude_refs_from_eval")),
        },
        "centers": centers,
        "ref_record_id_hashes": {
            center: payload["ref_record_ids_sha256"]
            for center, payload in centers.items()
        },
    }


def write_k500_ref_ids_artifact(manifest: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_k500_ref_ids_artifact(manifest, output_path=output_path)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def build_selection_record_artifact(manifest: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
    """Build the paper-safety record for model selection data boundaries."""
    trace = manifest.get("artifact_trace") or {}
    metrics = trace.get("metrics") or {}
    paper = manifest.get("paper_protocol") or {}
    selection = validate_selection_policy(trace.get("selection_policy") or paper.get("selection") or {})
    forbidden_scan_payload = {
        "selection": selection,
        "artifact_trace": {
            "selection_policy": trace.get("selection_policy") or {},
            "protocol_audit": trace.get("protocol_audit") or {},
        },
    }
    return {
        "schema_version": 1,
        "artifact_type": "selection_record",
        "generated_at_utc": _utc_now(),
        "artifact_path": str(output_path.resolve()),
        "run_id": manifest.get("run_id", ""),
        "experiment": manifest.get("experiment", {}),
        "config_hash_sha256": manifest.get("config_hash_sha256", ""),
        "paper_protocol": {
            "mapping_version": metrics.get("mapping_version") or paper.get("mapping_version", ""),
            "mapping_hash": metrics.get("mapping_hash") or paper.get("mapping_hash", ""),
            "class_order": metrics.get("class_order") or paper.get("class_order", []),
            "centers": paper.get("centers", {}),
            "kshot": paper.get("kshot", {}),
        },
        "selection_policy": selection,
        "selection_safety": {
            "allowed_data": list(ALLOWED_SELECTION_DATA),
            "forbidden_references": list(FORBIDDEN_SELECTION_REFERENCES),
            "forbid_heldout_target_labels": True,
            "forbid_full_target_distribution_tuning": True,
            "heldout_target_labels_used_for_selection": False,
            "full_target_distribution_used_for_tuning": False,
            "forbidden_reference_found": has_forbidden_selection_reference(forbidden_scan_payload),
        },
        "evidence": {
            "source": "run_manifest.artifact_trace.selection_policy",
            "selection_policy_validated": True,
            "command_audit_passed": bool(
                ((trace.get("protocol_audit") or {}).get("command_audit") or {}).get("passed", False)
            ),
            "postprocess_audit_passed": bool(
                ((trace.get("protocol_audit") or {}).get("postprocess_audit") or {}).get("passed", True)
            ),
        },
    }


def write_selection_record_artifact(manifest: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_selection_record_artifact(manifest, output_path=output_path)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def attach_launch_artifacts(manifest: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    """Attach concrete managed-run launch artifacts to the manifest."""
    manifest = dict(manifest)
    trace = dict(manifest.get("artifact_trace") or {})
    expected = dict(trace.get("expected_outputs") or {})
    launch_specs = {
        "run_config.resolved.yaml": ("run_config_yaml", run_dir / "run_config.resolved.yaml"),
        "run_config.resolved.json": ("run_config_json", run_dir / "run_config.resolved.json"),
        "run_manifest.json": ("run_manifest", run_dir / "run_manifest.json"),
        "command.sh": ("command_sh", run_dir / "command.sh"),
        "data_manifest.json": ("data_manifest", run_dir / "data_manifest.json"),
        "k500_ref_ids.json": ("k500_ref_ids", run_dir / "k500_ref_ids.json"),
        "selection.json": ("selection_record", run_dir / "selection.json"),
        "run_card.json": ("run_card", run_dir / "run_card.json"),
        "run_file_index.json": ("run_file_index", run_dir / "run_file_index.json"),
        "summary.md": ("run_summary", run_dir / "summary.md"),
    }
    declared = expected.get("launch_artifacts_declared") or list(launch_specs)
    launch_artifacts = []
    for name in declared:
        role, path = launch_specs.get(str(name), (str(name), run_dir / str(name)))
        launch_artifacts.append(
            {
                "role": role,
                "path": str(path),
                "required": True,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    expected["launch_artifacts"] = launch_artifacts
    trace["expected_outputs"] = expected
    manifest["artifact_trace"] = trace
    return manifest


def _validate_k500_ref_ids_artifact(
    path: Path,
    *,
    manifest: dict[str, Any],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    try:
        artifact = _read_json_object(path, label="k500_ref_ids artifact")
    except LaunchError as exc:
        bad = dict(record)
        bad["error"] = str(exc)
        return [bad]

    paper = manifest.get("paper_protocol") or {}
    kshot = paper.get("kshot") or {}
    metrics = (manifest.get("artifact_trace") or {}).get("metrics") or {}
    expected_k = int(kshot.get("k", 500))
    expected_seed = int(kshot.get("subset_seed", kshot.get("seed", 0)))
    expected_centers = _k500_centers_for_manifest(manifest)
    artifact_protocol = artifact.get("paper_protocol") or {}
    if artifact_protocol.get("mapping_version") != metrics.get("mapping_version"):
        bad = dict(record)
        bad["error"] = "k500_ref_ids mapping_version mismatch"
        errors.append(bad)
    if artifact_protocol.get("mapping_hash") != metrics.get("mapping_hash"):
        bad = dict(record)
        bad["error"] = "k500_ref_ids mapping_hash mismatch"
        errors.append(bad)

    centers = artifact.get("centers") or {}
    for center in expected_centers:
        payload = centers.get(center)
        if not isinstance(payload, dict):
            bad = dict(record)
            bad["center"] = center
            bad["error"] = "missing center in k500_ref_ids"
            errors.append(bad)
            continue
        ids_sorted = [str(v) for v in (payload.get("ref_record_ids_sorted") or [])]
        ids_ordered = [str(v) for v in (payload.get("ref_record_ids_ordered") or [])]
        if int(payload.get("k", -1)) != expected_k:
            bad = dict(record)
            bad["center"] = center
            bad["error"] = f"k={payload.get('k')} expected {expected_k}"
            errors.append(bad)
        if int(payload.get("selection_seed", -1)) != expected_seed:
            bad = dict(record)
            bad["center"] = center
            bad["error"] = f"selection_seed={payload.get('selection_seed')} expected {expected_seed}"
            errors.append(bad)
        if len(ids_ordered) != expected_k or len(ids_sorted) != expected_k or len(set(ids_ordered)) != expected_k:
            bad = dict(record)
            bad["center"] = center
            bad["error"] = "ref_record_ids count/uniqueness mismatch"
            errors.append(bad)
        if payload.get("ref_record_ids_sha256") != _sha256_lines(sorted(ids_ordered)):
            bad = dict(record)
            bad["center"] = center
            bad["error"] = "ref_record_ids_sha256 mismatch"
            errors.append(bad)
        source_meta = Path(str(payload.get("source_ref_meta_path", "")))
        if not source_meta.exists():
            bad = dict(record)
            bad["center"] = center
            bad["error"] = "source_ref_meta_path missing"
            errors.append(bad)
        elif payload.get("source_ref_meta_sha256") != _sha256_file(source_meta):
            bad = dict(record)
            bad["center"] = center
            bad["error"] = "source_ref_meta_sha256 mismatch"
            errors.append(bad)
    return errors


def _validate_selection_record_artifact(
    path: Path,
    *,
    manifest: dict[str, Any],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    try:
        artifact = _read_json_object(path, label="selection record")
        validate_selection_policy(artifact.get("selection_policy") or {})
    except Exception as exc:
        errors.append({**record, "error": str(exc)})
        return errors

    metrics = (manifest.get("artifact_trace") or {}).get("metrics") or {}
    protocol = artifact.get("paper_protocol") or {}
    safety = artifact.get("selection_safety") or {}
    checks = [
        (
            protocol.get("mapping_version") == metrics.get("mapping_version"),
            "selection mapping_version does not match manifest",
        ),
        (
            protocol.get("mapping_hash") == metrics.get("mapping_hash"),
            "selection mapping_hash does not match manifest",
        ),
        (
            safety.get("heldout_target_labels_used_for_selection") is False,
            "selection record must explicitly deny held-out target-label selection",
        ),
        (
            safety.get("full_target_distribution_used_for_tuning") is False,
            "selection record must explicitly deny full target-distribution tuning",
        ),
        (
            safety.get("forbidden_reference_found") is False,
            "selection record contains forbidden target-selection references",
        ),
    ]
    for passed, message in checks:
        if not passed:
            errors.append({**record, "error": message})
    return errors


def _epoch_entries_from_training_log(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("epochs", "history", "log"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "epoch" in data:
            return [data]
    return []


def _dotted_metric_present(entry: dict[str, Any], metric: str) -> bool:
    cur: Any = entry
    for part in str(metric).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return cur is not None


def _validate_training_log_metrics(
    path: Path,
    *,
    manifest: dict[str, Any],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    required_metrics = list(dict.fromkeys([
        *[str(item) for item in expected.get("required_epoch_metrics") or []],
        *[str(item) for item in record.get("required_epoch_metrics") or []],
    ]))
    if not required_metrics:
        return []
    errors: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [{**record, "error": f"training_log is not readable JSON: {exc}"}]

    entries = _epoch_entries_from_training_log(data)
    if not entries:
        return [{**record, "error": "training_log has no epoch entries"}]
    missing = [
        metric
        for metric in required_metrics
        if not any(_dotted_metric_present(entry, metric) for entry in entries)
    ]
    if missing:
        errors.append(
            {
                **record,
                "error": "training_log missing required epoch metrics",
                "missing_metrics": missing,
                "epoch_entry_count": len(entries),
            }
        )
    else:
        record["epoch_metric_verified"] = True
        record["required_epoch_metric_count"] = len(required_metrics)
        record["epoch_entry_count"] = len(entries)
    return errors


def verify_required_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify traced required input artifacts before executing child scripts."""
    trace = manifest.get("artifact_trace") or {}
    inputs = trace.get("inputs") or {}
    missing: list[dict[str, Any]] = []
    lineage_errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    initialization = trace.get("initialization") or {}
    initialization_lineage: dict[str, Any] = {}
    replication_preflight: dict[str, Any] | None = None

    def check_record(record: dict[str, Any] | None) -> None:
        if not record or record.get("required") is False:
            return
        path = Path(str(record.get("path", "")))
        out = {
            "role": record.get("role"),
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            missing.append(out)
        elif path.is_file():
            out["size_bytes"] = path.stat().st_size
        verified.append(out)

    for record in inputs.get("checkpoints") or []:
        check_record(record)
    for record in inputs.get("data_caches") or []:
        check_record(record)
    for ref in inputs.get("k500_refs") or []:
        check_record(ref.get("ref_meta_json"))
        check_record(ref.get("signals_npz"))
        check_record(ref.get("latent_npz"))
    replication_groups = inputs.get("replication_k500_groups") or []
    if replication_groups:
        metrics = trace.get("metrics") or {}
        replication_preflight = verify_replication_k500_groups(
            replication_groups,
            mapping_version=str(metrics.get("mapping_version") or ""),
            mapping_hash=str(metrics.get("mapping_hash") or ""),
        )
        missing.extend(replication_preflight["missing"])
        lineage_errors.extend(
            {
                "role": "replication_k500_identity",
                "path": str(item.get("base") or ""),
                "error": str(item.get("error") or ""),
                "center": item.get("center"),
            }
            for item in replication_preflight["identity_errors"]
        )
        verified.extend(replication_preflight["verified_inputs"])
        validation_contract = (trace.get("replication_preflight") or {}).get("validation_report") or {}
        if validation_contract:
            validation_groups = inputs.get("replication_validation_groups") or replication_groups
            validation_report = verify_replication_validation_report(
                validation_contract, validation_groups
            )
            replication_preflight["validation_report"] = validation_report
            lineage_errors.extend(
                {
                    "role": "replication_validation_report",
                    "path": str(validation_report.get("path") or ""),
                    "error": error,
                }
                for error in validation_report["errors"]
            )

    if initialization:
        try:
            if initialization.get("stage") != "ptbxl_source":
                raise ValueError("initialization.stage must be ptbxl_source")
            checkpoint = Path(str(initialization.get("checkpoint_path") or ""))
            expected_sha = str(initialization.get("checkpoint_sha256") or "")
            if not checkpoint.is_file() or not expected_sha:
                raise ValueError("ptbxl_source initialization requires checkpoint_path and checkpoint_sha256")
            actual_sha = _sha256_file(checkpoint)
            if actual_sha != expected_sha:
                raise ValueError("ptbxl_source checkpoint sha256 mismatch")
            evidence_path = Path(str(initialization.get("evidence_path") or ""))
            evidence = _read_json_object(evidence_path, label="ptbxl_source evidence")
            evidence_config = evidence.get("config") if isinstance(evidence.get("config"), dict) else {}
            binding = evidence.get("checkpoint") if isinstance(evidence.get("checkpoint"), dict) else {}
            bound_path = str(binding.get("path") or "")
            path_matches = Path(bound_path) == checkpoint or str(checkpoint).endswith(
                f"/{bound_path.lstrip('/')}"
            )
            if not path_matches or str(binding.get("sha256") or "") != actual_sha:
                raise ValueError("ptbxl_source evidence checkpoint binding mismatch")
            if str(evidence.get("scheme") or "") != "super5" or int(evidence.get("num_classes") or 0) != 5:
                raise ValueError("ptbxl_source evidence must be a Super5 source-training result")
            if "ptbxl" not in str(evidence_config.get("data_path") or "").lower() or any(
                evidence_config.get(key) not in (None, "") for key in ("synth_npz", "init_ckpt")
            ):
                raise ValueError("ptbxl_source evidence must be source-only PTB-XL")
            initialization_lineage = {
                "stage": "ptbxl_source",
                "source_only": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": actual_sha,
                "evidence_path": str(evidence_path),
            }
        except (LaunchError, OSError, ValueError) as exc:
            lineage_errors.append(
                {
                    "role": "checkpoint_source_lineage",
                    "path": str(initialization.get("checkpoint_path") or ""),
                    "error": str(exc),
                }
            )

    for command in manifest.get("commands") or []:
        argv = [str(item) for item in command.get("argv") or []]
        script = Path(argv[1]).name if len(argv) > 1 else ""
        opts = argv_option_map(argv)
        init_path = opt_first(opts, "--init_model_path" if script == "ecgfounder_fullft.py" else "--init_ckpt")
        if not init_path or script not in {"ecgfounder_fullft.py", "effnet_vae_lhat_augmix.py"}:
            continue
        if initialization_lineage.get("source_only") and Path(str(init_path)) == Path(
            initialization_lineage["checkpoint_path"]
        ):
            continue
        ref_meta_path = opt_first(opts, "--ref_meta_json")
        if not ref_meta_path and opt_first(opts, "--anchor_base"):
            ref_meta_path = f"{opt_first(opts, '--anchor_base')}.ref_meta.json"
        try:
            ref_meta = _read_json_object(Path(str(ref_meta_path)), label="current K500 ref meta")
            selected_ids = _extract_ref_record_ids(ref_meta, Path(str(ref_meta_path)))
            expected_k = int(opt_first(opts, "--k", "500"))
            current = {
                "stage": "k500",
                "center": ref_meta.get("center") or opt_first(opts, "--center"),
                "K": expected_k,
                "target_train_K": expected_k,
                "selected_ref_record_ids": selected_ids,
                "target_train_record_ids": list(selected_ids),
                "config": {
                    "stage": "k500",
                    "seed": ref_meta.get("selection_seed", ref_meta.get("seed")),
                },
            }
            init_dir = Path(str(init_path)).parent
            init_result_path = init_dir / "eval_result.json"
            if init_result_path.is_file():
                init_result = _read_json_object(init_result_path, label="initialization eval result")
            else:
                init_config = _read_json_object(init_dir / "run_config.json", label="initialization run config")
                train_ids = init_config.get("train_record_ids")
                val_ids = init_config.get("val_record_ids")
                if not isinstance(train_ids, list) or not isinstance(val_ids, list):
                    raise ValueError("initialization run config must expose train_record_ids and val_record_ids lists")
                all_ids = [str(item) for item in train_ids + val_ids]
                selected_ids = list(dict.fromkeys(all_ids))
                if len(selected_ids) != len(all_ids):
                    raise ValueError("initialization run config contains duplicate train/val record ids")
                init_result = {
                    "stage": "k500",
                    "center": init_config.get("center"),
                    "K": init_config.get("k"),
                    "target_train_K": expected_k,
                    "selected_ref_record_ids": selected_ids,
                    "target_train_record_ids": list(selected_ids),
                    "config": {**init_config, "stage": "k500"},
                }
            validate_target_init_k500_identity(current, init_result)
        except (LaunchError, OSError, ValueError) as exc:
            lineage_errors.append({"role": "checkpoint_k500_lineage", "path": str(init_path), "error": str(exc)})

    report = {
        "passed": not missing and not lineage_errors,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_checked": len(verified),
        "missing": missing,
        "lineage_errors": lineage_errors,
        "initialization_lineage": initialization_lineage,
        "verified_inputs": verified,
    }
    if replication_preflight is not None:
        report["replication_k500_preflight"] = replication_preflight
    return report


def verify_required_artifacts(manifest: dict[str, Any], *, include_postprocess: bool = True) -> dict[str, Any]:
    """Verify managed and child artifacts declared in a managed run manifest."""
    trace = manifest.get("artifact_trace") or {}
    metrics = trace.get("metrics") or {}
    expected_mapping_version = metrics.get("mapping_version")
    expected_mapping_hash = metrics.get("mapping_hash")
    expected_outputs = trace.get("expected_outputs") or {}
    launch_artifacts = expected_outputs.get("launch_artifacts") or []
    child_runs = expected_outputs.get("child_runs") or []
    postprocess_runs = expected_outputs.get("postprocess_runs") or []
    missing: list[dict[str, Any]] = []
    mapping_errors: list[dict[str, Any]] = []
    content_errors: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []

    if not child_runs and not launch_artifacts:
        return {
            "passed": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_checked": 0,
            "missing": [{"role": "artifact_trace.expected_outputs", "path": "", "exists": False}],
            "mapping_errors": [],
            "content_errors": [],
            "verified_artifacts": [],
        }

    for artifact in launch_artifacts:
        if artifact.get("required") is False:
            continue
        path = Path(str(artifact.get("path", "")))
        record = {
            "command_index": None,
            "center": None,
            "role": artifact.get("role"),
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            missing.append(record)
            verified.append(record)
            continue
        if path.is_file():
            record["size_bytes"] = path.stat().st_size
        if artifact.get("role") == "k500_ref_ids":
            errors = _validate_k500_ref_ids_artifact(path, manifest=manifest, record=record)
            content_errors.extend(errors)
            record["content_verified"] = not errors
        elif artifact.get("role") == "selection_record":
            errors = _validate_selection_record_artifact(path, manifest=manifest, record=record)
            content_errors.extend(errors)
            record["content_verified"] = not errors
        verified.append(record)

    for run in child_runs:
        for artifact in run.get("expected_artifacts") or []:
            if artifact.get("required") is False:
                continue
            path = Path(str(artifact.get("path", "")))
            record = {
                "command_index": run.get("command_index"),
                "center": run.get("center"),
                "role": artifact.get("role"),
                "path": str(path),
                "exists": path.exists(),
                "required_epoch_metrics": run.get("required_epoch_metrics") or [],
            }
            if not path.exists():
                missing.append(record)
                verified.append(record)
                continue
            if path.is_file():
                record["size_bytes"] = path.stat().st_size
            if artifact.get("role") == "eval_result":
                try:
                    result_json = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    record["mapping_verified"] = False
                    record["error"] = f"eval_result is not readable JSON: {exc}"
                    mapping_errors.append(record.copy())
                else:
                    mapping = _extract_mapping_metadata(result_json)
                    version = mapping.get("mapping_version") if mapping else None
                    mapping_hash = mapping.get("mapping_hash") if mapping else None
                    record["mapping_version"] = version
                    record["mapping_hash"] = mapping_hash
                    record["mapping_verified"] = (
                        version == expected_mapping_version
                        and mapping_hash == expected_mapping_hash
                    )
                    if not record["mapping_verified"]:
                        mapping_errors.append(record.copy())
            elif artifact.get("role") == "training_log":
                errors = _validate_training_log_metrics(path, manifest=manifest, record=record)
                content_errors.extend(errors)
                if errors:
                    record["epoch_metric_verified"] = False
            verified.append(record)

    if include_postprocess:
        for run in postprocess_runs:
            for artifact in run.get("expected_artifacts") or []:
                if artifact.get("required") is False:
                    continue
                path = Path(str(artifact.get("path", "")))
                record = {
                    "command_index": run.get("command_index"),
                    "center": None,
                    "role": artifact.get("role"),
                    "path": str(path),
                    "exists": path.exists(),
                    "postprocess": True,
                }
                if not path.exists():
                    missing.append(record)
                    verified.append(record)
                    continue
                if path.is_file():
                    record["size_bytes"] = path.stat().st_size
                verified.append(record)

    passed = not missing and not mapping_errors
    passed = passed and not content_errors
    return {
        "passed": passed,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_checked": len(verified),
        "missing": missing,
        "mapping_errors": mapping_errors,
        "content_errors": content_errors,
        "verified_artifacts": verified,
    }


def _prepare_expected_artifact_dirs(manifest: dict[str, Any], command_index: int) -> None:
    trace = manifest.get("artifact_trace") or {}
    expected_outputs = trace.get("expected_outputs") or {}
    for run in expected_outputs.get("child_runs") or []:
        if int(run.get("command_index", -1)) != int(command_index):
            continue
        for artifact in run.get("expected_artifacts") or []:
            if artifact.get("required") is False:
                continue
            raw_path = artifact.get("path")
            if raw_path:
                Path(str(raw_path)).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _prepare_expected_postprocess_artifact_dirs(manifest: dict[str, Any], command_index: int) -> None:
    trace = manifest.get("artifact_trace") or {}
    expected_outputs = trace.get("expected_outputs") or {}
    for run in expected_outputs.get("postprocess_runs") or []:
        if int(run.get("command_index", -1)) != int(command_index):
            continue
        for artifact in run.get("expected_artifacts") or []:
            if artifact.get("required") is False:
                continue
            raw_path = artifact.get("path")
            if raw_path:
                Path(str(raw_path)).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _append_log_file(src: Path, dst: Path, *, header: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8", errors="replace") as out:
        out.write(f"\n===== {header}: {src.name} =====\n")
        if not src.exists():
            out.write("[missing log]\n")
            return
        last_char = "\n"
        with src.open("r", encoding="utf-8", errors="replace") as inp:
            for chunk in iter(lambda: inp.read(1024 * 1024), ""):
                if not chunk:
                    break
                out.write(chunk)
                last_char = chunk[-1]
        if src.stat().st_size and last_char != "\n":
            out.write("\n")


def _write_combined_log(stdout_path: Path, stderr_path: Path, combined_path: Path, *, label: str) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8", errors="replace") as out:
        for stream_name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            out.write(f"===== {label} {stream_name}: {path.name} =====\n")
            if path.exists():
                with path.open("r", encoding="utf-8", errors="replace") as inp:
                    for chunk in iter(lambda: inp.read(1024 * 1024), ""):
                        if not chunk:
                            break
                        out.write(chunk)
            if stream_name == "stdout":
                out.write("\n")


def _run_logged_subprocess(
    command: dict[str, Any],
    *,
    env: dict[str, str],
    run_dir: Path,
    logs_dir: Path,
    prefix: str,
    idx: int,
) -> tuple[int, dict[str, str], str, str]:
    log_path = logs_dir / f"{prefix}_{idx:02d}.log"
    stdout_path = logs_dir / f"{prefix}_{idx:02d}.stdout.log"
    stderr_path = logs_dir / f"{prefix}_{idx:02d}.stderr.log"
    started_at = _utc_now()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_log, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_log:
        proc = subprocess.Popen(
            command["argv"],
            cwd=command["cwd"],
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
            text=True,
        )
        ret = proc.wait()
    finished_at = _utc_now()
    label = f"{prefix}_{idx:02d} {command.get('name') or ''}".strip()
    _write_combined_log(stdout_path, stderr_path, log_path, label=label)
    _append_log_file(stdout_path, run_dir / "stdout.log", header=label)
    _append_log_file(stderr_path, run_dir / "stderr.log", header=label)
    paths = {
        "log_path": str(log_path),
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
    }
    return ret, paths, started_at, finished_at


def run_managed_commands(
    commands: list[dict[str, Any]],
    *,
    run_dir: Path,
    manifest_path: Path,
    base_env: dict[str, str] | None = None,
) -> None:
    base_env = dict(base_env or os.environ)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    update_manifest_file(manifest_path, {"status": "running", "execution_started_at_utc": _utc_now()})

    for idx, command in enumerate(commands):
        manifest = _load_manifest(manifest_path)
        _prepare_expected_artifact_dirs(manifest, idx)
        env = dict(base_env)
        env.update({str(k): str(v) for k, v in (command.get("env") or {}).items()})
        for key in ("TMPDIR", "XDG_CACHE_HOME"):
            if env.get(key):
                Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
        ret, log_paths, started_at, finished_at = _run_logged_subprocess(
            command,
            env=env,
            run_dir=run_dir,
            logs_dir=logs_dir,
            prefix="command",
            idx=idx,
        )
        run_record = {
            "command_index": idx,
            "name": command.get("name"),
            "cwd": command.get("cwd"),
            **log_paths,
            "returncode": ret,
            "status": "succeeded" if ret == 0 else "failed",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        }
        if ret != 0:
            _append_manifest_record(
                manifest_path,
                "command_runs",
                run_record,
                {
                    "status": "failed",
                    "failed_command_index": idx,
                    "failed_command_name": command.get("name"),
                    "failed_returncode": ret,
                    "failed_log": log_paths["log_path"],
                    "failed_stdout_log": log_paths["stdout_log_path"],
                    "failed_stderr_log": log_paths["stderr_log_path"],
                },
            )
            raise LaunchError(f"Managed command {idx} failed with return code {ret}; see {log_paths['log_path']}")
        _append_manifest_record(manifest_path, "command_runs", run_record)

    manifest = _load_manifest(manifest_path)
    verification = verify_required_artifacts(manifest, include_postprocess=False)
    if not verification["passed"]:
        update_manifest_file(
            manifest_path,
            {
                "status": "failed",
                "artifact_verification": verification,
            },
        )
        raise LaunchError("Required artifact verification failed after managed commands")

    update_manifest_file(
        manifest_path,
        {
            "status": "succeeded",
            "artifact_verification": verification,
            "child_commands_finished_at_utc": _utc_now(),
        },
    )


def run_postprocess_commands(
    commands: list[dict[str, Any]],
    *,
    run_dir: Path,
    manifest_path: Path,
    base_env: dict[str, str] | None = None,
) -> None:
    """Run managed reporting/export commands after managed child commands."""
    if not commands:
        manifest = _load_manifest(manifest_path)
        update_manifest_file(manifest_path, _final_success_patch(manifest, finished_at=_utc_now()))
        return

    base_env = dict(base_env or os.environ)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    update_manifest_file(manifest_path, {"status": "postprocessing"})

    for idx, command in enumerate(commands):
        manifest = _load_manifest(manifest_path)
        _prepare_expected_postprocess_artifact_dirs(manifest, idx)
        env = dict(base_env)
        env.update({str(k): str(v) for k, v in (command.get("env") or {}).items()})
        for key in ("TMPDIR", "XDG_CACHE_HOME"):
            if env.get(key):
                Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
        ret, log_paths, started_at, finished_at = _run_logged_subprocess(
            command,
            env=env,
            run_dir=run_dir,
            logs_dir=logs_dir,
            prefix="postprocess",
            idx=idx,
        )
        run_record = {
            "command_index": idx,
            "name": command.get("name"),
            "cwd": command.get("cwd"),
            **log_paths,
            "returncode": ret,
            "status": "succeeded" if ret == 0 else "failed",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        }
        if ret != 0:
            _append_manifest_record(
                manifest_path,
                "postprocess_runs",
                run_record,
                {
                    "status": "failed",
                    "failed_postprocess_index": idx,
                    "failed_postprocess_name": command.get("name"),
                    "failed_returncode": ret,
                    "failed_log": log_paths["log_path"],
                    "failed_stdout_log": log_paths["stdout_log_path"],
                    "failed_stderr_log": log_paths["stderr_log_path"],
                },
            )
            raise LaunchError(f"Postprocess command {idx} failed with return code {ret}; see {log_paths['log_path']}")
        _append_manifest_record(manifest_path, "postprocess_runs", run_record)

    manifest = _load_manifest(manifest_path)
    verification = verify_required_artifacts(manifest, include_postprocess=True)
    if not verification["passed"]:
        update_manifest_file(
            manifest_path,
            {
                "status": "failed",
                "artifact_verification": verification,
            },
        )
        raise LaunchError("Required artifact verification failed after postprocess commands")

    manifest = _load_manifest(manifest_path)
    patch = _final_success_patch(manifest, finished_at=_utc_now())
    patch["artifact_verification"] = verification
    update_manifest_file(manifest_path, patch)
