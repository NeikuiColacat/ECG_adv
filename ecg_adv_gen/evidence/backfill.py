"""Backfill manifests for legacy evidence that predates managed launchers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import EvidenceAuditError, load_evidence_registry


_CENTER_REQUIRED_FILES = (
    "launch_config.json",
    "train_result.json",
    "early_stop_info.json",
    "eval_result_v7_exclrefs_crop1000.json",
    "best_model.pt",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    record: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
    }
    if hash_file:
        record["sha256"] = _sha256(path)
    return record


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceAuditError(f"Invalid JSON while backfilling manifest: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceAuditError(f"JSON root must be an object while backfilling manifest: {path}")
    return data


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    return {
        "backfill_git_sha": run(["rev-parse", "HEAD"]),
        "backfill_git_dirty_short": run(["status", "--short"]),
        "note": "This git state records the backfill operation, not the original legacy training launch.",
    }


def _center_from_launch_config(path: Path) -> str:
    data = _read_json(path)
    args = data.get("args") or {}
    center = args.get("center") or args.get("center_name")
    if not center:
        raise EvidenceAuditError(f"Cannot infer center from launch_config: {path}")
    return str(center)


def _collect_center_run(
    run_dir: Path,
    center: str,
    *,
    expected_mapping_version: str,
    expected_mapping_hash: str,
    hash_checkpoints: bool,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in _CENTER_REQUIRED_FILES:
        path = run_dir / name
        artifacts[name] = _file_record(
            path,
            hash_file=hash_checkpoints or name != "best_model.pt",
        )
        if not artifacts[name].get("exists"):
            raise EvidenceAuditError(f"Missing legacy center artifact for {center}: {path}")

    launch = _read_json(run_dir / "launch_config.json")
    train_result = _read_json(run_dir / "train_result.json")
    early_stop = _read_json(run_dir / "early_stop_info.json")
    eval_result = _read_json(run_dir / "eval_result_v7_exclrefs_crop1000.json")
    args = launch.get("args") or {}
    pn2021_eval = eval_result.get("pn2021") or {}
    label_mapping = eval_result.get("label_mapping") or {}
    mapping_info = label_mapping.get("pn2021_super5", {})
    observed_mapping_version = mapping_info.get("mapping_version")
    observed_mapping_hash = mapping_info.get("mapping_hash")
    if observed_mapping_version != expected_mapping_version or observed_mapping_hash != expected_mapping_hash:
        raise EvidenceAuditError(
            f"Eval mapping drift for {center}: "
            f"{observed_mapping_version}/{observed_mapping_hash} != "
            f"{expected_mapping_version}/{expected_mapping_hash}"
        )

    return {
        "center": center,
        "run_dir": str(run_dir.resolve()),
        "seed": args.get("seed"),
        "init_ckpt": args.get("init_ckpt"),
        "ref_meta_json": train_result.get("args", {}).get("ref_meta_json"),
        "target_real_npz": train_result.get("args", {}).get("target_real_npz"),
        "synth_npz": train_result.get("args", {}).get("synth_npz"),
        "selection": {
            "es_metric": early_stop.get("es_metric"),
            "best_epoch": early_stop.get("best_epoch"),
            "best_metric": early_stop.get("best_metric"),
            "stopped_epoch": early_stop.get("stopped_epoch"),
            "early_stopped": early_stop.get("early_stopped"),
        },
        "checkpoint_selection": {
            "center": center,
            "best_epoch": early_stop.get("best_epoch"),
            "metric": early_stop.get("es_metric"),
            "metric_value": early_stop.get("best_metric"),
            "selection_source": "target_real_val_internal",
        },
        "final_eval": {
            "macro_auroc": pn2021_eval.get("avg_macro_auroc"),
            "macro_auprc": pn2021_eval.get("avg_macro_auprc"),
            "drop_all_zero_macro_auroc": pn2021_eval.get("avg_drop_all_zero_macro_auroc"),
            "drop_all_zero_macro_auprc": pn2021_eval.get("avg_drop_all_zero_macro_auprc"),
            "mapping_version": observed_mapping_version,
            "mapping_hash": observed_mapping_hash,
        },
        "train_cmd": launch.get("train_cmd", []),
        "eval_cmd": launch.get("eval_cmd", []),
        "artifacts": artifacts,
    }


def build_legacy_vae_lhat_manifest(
    *,
    repo_root: Path,
    registry_path: Path,
    local_config_path: Path,
    claim_id: str = "effnet_v7_vae_lhat_improves_direct_k500",
    method_key: str = "vae_lhat",
    output_path: Path | None = None,
    write_boundary: Path | None = None,
    hash_checkpoints: bool = True,
) -> dict[str, Any]:
    """Create a run-level manifest for the legacy v7 VAE L-HAT result."""
    repo_root = Path(repo_root).expanduser().resolve()
    registry = load_evidence_registry(registry_path, local_config_path)
    claims = {claim["claim_id"]: claim for claim in registry.get("active_claims", [])}
    if claim_id not in claims:
        raise EvidenceAuditError(f"Unknown claim_id in registry: {claim_id}")
    claim = claims[claim_id]
    if method_key not in claim.get("methods", {}):
        raise EvidenceAuditError(f"Unknown method_key for claim {claim_id}: {method_key}")
    method = claim["methods"][method_key]
    run_root = Path(method["run_root"]).expanduser().resolve()
    method_manifest_path = method.get("manifest")
    if output_path is None:
        output_path = Path(method_manifest_path).expanduser().resolve() if method_manifest_path else run_root / "run_manifest.backfilled.json"
    else:
        output_path = Path(output_path).expanduser().resolve()
    if write_boundary is not None:
        boundary = Path(write_boundary).expanduser().resolve()
        if not _is_under(output_path, boundary):
            raise EvidenceAuditError(f"Backfilled manifest output is outside write boundary: {output_path}")
    if not run_root.exists():
        raise EvidenceAuditError(f"Legacy method run_root does not exist: {run_root}")

    center_launches: dict[str, Path] = {}
    for launch_path in sorted(run_root.glob("*/launch_config.json")):
        center = _center_from_launch_config(launch_path)
        center_launches[center] = launch_path

    expected_centers = list(claim["protocol"]["target_centers"])
    missing = [center for center in expected_centers if center not in center_launches]
    if missing:
        raise EvidenceAuditError(f"Legacy run is missing center launch_config files: {missing}")

    expected_mapping_version = str(claim["protocol"]["mapping_version"])
    expected_mapping_hash = str(claim["protocol"]["mapping_hash"])
    center_runs = [
        _collect_center_run(
            center_launches[center].parent,
            center,
            expected_mapping_version=expected_mapping_version,
            expected_mapping_hash=expected_mapping_hash,
            hash_checkpoints=hash_checkpoints,
        )
        for center in expected_centers
    ]

    source_artifacts = {
        "metrics_long": _file_record(Path(method["metrics_long"]), hash_file=True),
        "artifact_manifest": _file_record(Path(method["artifact_manifest"]), hash_file=True),
        "paper_tables": {
            view: _file_record(Path(path), hash_file=True)
            for view, path in (method.get("paper_tables") or {}).items()
        },
    }

    manifest = {
        "schema_version": 1,
        "manifest_kind": "legacy_backfilled_manifest",
        "status": "succeeded",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim_id,
        "method_key": method_key,
        "method_family": method.get("method_family"),
        "run_id": method.get("run_id"),
        "config": method.get("config"),
        "run_root": str(run_root),
        "protocol": {
            "mapping_version": claim["protocol"]["mapping_version"],
            "mapping_hash": claim["protocol"]["mapping_hash"],
            "class_order": list(claim["protocol"]["class_order"]),
            "target_centers": expected_centers,
            "kshot": dict(claim["protocol"]["kshot"]),
            "selection": dict(claim["protocol"]["selection"]),
        },
        "selection": {
            "policy": "target_real_val_internal",
            "selection_source": "target_real_val",
            "heldout_target_labels_used_for_selection": False,
            "full_target_distribution_used_for_selection": False,
            "source_floor_used": False,
            "notes": "Backfilled from legacy early_stop_info.json and command records.",
        },
        "run_record": {
            "registration_status": "provisional",
            "status": "succeeded",
            "purpose": "EffNet v7 VAE-LHAT K500 target-center adaptation evidence backfill.",
            "result_summary": "Legacy VAE-LHAT run backfilled for traceability; cite only after registry registration.",
        },
        "centers": center_runs,
        "source_artifacts": source_artifacts,
        "git": _git_snapshot(repo_root),
        "limitations": [
            "Backfilled after the original historical run; git state records the backfill, not the original launch.",
            "Use the per-center launch_config/train_result/eval artifacts as the replay source for this historical run.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest
