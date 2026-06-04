"""Agent-readable run cards, file indexes, and registry registration."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class RunRecordError(ValueError):
    """Raised when a run record cannot be finalized or registered safely."""


RUN_LAYOUT_DIRS = {
    "configs": "Resolved config, command line, and launch-time settings.",
    "manifests": "Run manifests, input manifests, file indexes, and run cards.",
    "logs": "Launcher stdout/stderr and child-process logs.",
    "checkpoints": "Model checkpoints and checkpoint indexes; do not commit.",
    "eval": "Evaluation metrics, paper tables, and per-center/per-class outputs.",
    "diagnostics": "Training diagnostics, attack diagnostics, and agent decisions.",
    "reports": "Human-readable summaries for handoff and paper work.",
    "artifacts": "Other generated small artifacts not covered by a narrower category.",
}

_SMALL_COPY_LIMIT_BYTES = 5 * 1024 * 1024
_GENERIC_RESULT_SUMMARIES = {
    "Run finalized with evaluation metrics; see metric_summary and eval artifacts.",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunRecordError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunRecordError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunRecordError(f"JSON root must be an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise RunRecordError(f"Missing YAML file: {path}") from exc
    if not isinstance(data, dict):
        raise RunRecordError(f"YAML root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _category_for(path: Path, run_dir: Path) -> str:
    rel = _relative(path, run_dir).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    if rel.startswith("logs/") or name.endswith(".log"):
        return "logs"
    if suffix in {".pt", ".pth", ".ckpt"} or "checkpoint" in rel or name in {"best_model.pt", "latest.pt"}:
        return "checkpoints"
    if rel.startswith("eval/") or name.startswith("eval_result") or name in {
        "metrics_long.csv",
        "per_center.csv",
        "per_class.csv",
        "paper_table.csv",
    } or "paper_table" in name:
        return "eval"
    if rel.startswith("diagnostics/") or name in {
        "diagnostics_epoch.jsonl",
        "agent_decision.json",
        "training_log.json",
        "train_result.json",
        "early_stop_info.json",
    }:
        return "diagnostics"
    if rel.startswith("configs/") or name in {
        "run_config.resolved.yaml",
        "run_config.resolved.json",
        "config.yaml",
        "resolved_config.yaml",
        "command.sh",
    }:
        return "configs"
    if rel.startswith("manifests/") or "manifest" in name or name in {
        "run_card.json",
        "run_file_index.json",
        "data_manifest.json",
        "k500_ref_ids.json",
        "selection.json",
    }:
        return "manifests"
    if rel.startswith("reports/") or suffix in {".md", ".html"}:
        return "reports"
    return "artifacts"


def ensure_run_layout(run_dir: Path) -> dict[str, str]:
    """Create the standard logical run layout directories."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for dirname, description in RUN_LAYOUT_DIRS.items():
        directory = run_dir / dirname
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(f"# {dirname}\n\n{description}\n", encoding="utf-8")
    return {name: str((run_dir / name).resolve()) for name in RUN_LAYOUT_DIRS}


def _copy_small_file(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_file():
        return
    if src.resolve() == dst.resolve():
        return
    if src.stat().st_size > _SMALL_COPY_LIMIT_BYTES:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_known_small_files(run_dir: Path) -> None:
    mirrors = {
        "run_config.resolved.yaml": "configs/run_config.resolved.yaml",
        "run_config.resolved.json": "configs/run_config.resolved.json",
        "command.sh": "configs/command.sh",
        "run_manifest.json": "manifests/run_manifest.snapshot.json",
        "data_manifest.json": "manifests/data_manifest.json",
        "k500_ref_ids.json": "manifests/k500_ref_ids.json",
        "selection.json": "manifests/selection.json",
        "run_card.json": "manifests/run_card.json",
        "run_file_index.json": "manifests/run_file_index.json",
        "summary.md": "reports/summary.md",
    }
    for src_name, dst_name in mirrors.items():
        _copy_small_file(run_dir / src_name, run_dir / dst_name)


def _iter_expected_artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    records: list[dict[str, Any]] = []
    for artifact in expected.get("launch_artifacts") or []:
        if artifact.get("path"):
            records.append({"path": Path(str(artifact["path"])), "role": artifact.get("role"), "source": "launch"})
    for group in ("child_runs", "postprocess_runs"):
        for run in expected.get(group) or []:
            for artifact in run.get("expected_artifacts") or []:
                if artifact.get("path"):
                    records.append(
                        {
                            "path": Path(str(artifact["path"])),
                            "role": artifact.get("role"),
                            "source": group,
                            "center": run.get("center"),
                            "command_index": run.get("command_index"),
                        }
                    )
    return records


def _iter_expected_artifact_paths(manifest: dict[str, Any]) -> list[Path]:
    return [record["path"] for record in _iter_expected_artifact_records(manifest)]


def _discover_metrics_paths(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for path in _iter_expected_artifact_paths(manifest):
        if path.name == "metrics_long.csv":
            candidates.append(path.expanduser())
    candidates.extend(run_dir.rglob("metrics_long.csv"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen and resolved.exists() and resolved.is_file():
            seen.add(key)
            out.append(resolved)
    return out


def _metric_summary_from_metrics_long(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]]
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("scope") != "center" or row.get("class_name"):
            continue
        metric = row.get("metric", "")
        if metric.endswith("macro_auroc"):
            canonical_metric = "macro_auroc"
        elif metric.endswith("macro_auprc"):
            canonical_metric = "macro_auprc"
        else:
            continue
        try:
            value = float(row.get("value", ""))
        except ValueError:
            continue
        view = row.get("canonical_view") or row.get("view") or "unknown_view"
        grouped.setdefault(view, {}).setdefault(canonical_metric, []).append(value)
    summary: dict[str, Any] = {}
    for view, metrics in grouped.items():
        summary[view] = {
            metric: round(sum(values) / len(values), 12)
            for metric, values in sorted(metrics.items())
            if values
        }
    return summary


def _build_metric_summary(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for path in _discover_metrics_paths(run_dir, manifest):
        for view, metrics in _metric_summary_from_metrics_long(path).items():
            summary[view] = metrics
    return summary


def _build_file_index(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in RUN_LAYOUT_DIRS}
    expected_paths = {str(path.expanduser().resolve()) for path in _iter_expected_artifact_paths(manifest)}
    seen_paths: set[str] = set()
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        resolved_key = str(path.resolve())
        seen_paths.add(resolved_key)
        category = _category_for(path, run_dir)
        record = {
            "relative_path": _relative(path, run_dir),
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "expected_artifact": str(path.resolve()) in expected_paths,
            "external": False,
        }
        categories.setdefault(category, []).append(record)
    for artifact in _iter_expected_artifact_records(manifest):
        path = artifact["path"].expanduser().resolve()
        resolved_key = str(path)
        if resolved_key in seen_paths:
            continue
        category = _category_for(path, run_dir)
        record = {
            "relative_path": _relative(path, run_dir),
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_artifact": True,
            "external": True,
            "role": artifact.get("role"),
            "source": artifact.get("source"),
            "center": artifact.get("center"),
            "command_index": artifact.get("command_index"),
        }
        categories.setdefault(category, []).append(record)
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "categories": categories,
    }


def _refresh_launch_artifact_status(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(manifest)
    trace = dict(manifest.get("artifact_trace") or {})
    expected = dict(trace.get("expected_outputs") or {})
    refreshed = []
    for artifact in expected.get("launch_artifacts") or []:
        item = dict(artifact)
        path = Path(str(item.get("path", "")))
        item["exists"] = path.exists()
        item["size_bytes"] = path.stat().st_size if path.is_file() else None
        refreshed.append(item)
    if refreshed:
        expected["launch_artifacts"] = refreshed
        trace["expected_outputs"] = expected
        manifest["artifact_trace"] = trace
    return manifest


def _default_result_summary(manifest: dict[str, Any], metric_summary: dict[str, Any]) -> str:
    status = str(manifest.get("status", "unknown"))
    if metric_summary:
        return "Run finalized with evaluation metrics; see metric_summary and eval artifacts."
    return f"Run finalized with status={status}; evaluation metrics were not discovered."


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _target_centers_from_protocol(protocol: dict[str, Any]) -> list[str]:
    centers = protocol.get("centers") or {}
    values = centers.get("target_4") or protocol.get("target_centers") or []
    return [str(item) for item in values if str(item)]


def _artifact_records_by_source(manifest: dict[str, Any], source: str) -> list[dict[str, Any]]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    if source == "launch":
        return list(expected.get("launch_artifacts") or [])
    records: list[dict[str, Any]] = []
    for run in expected.get(source) or []:
        for artifact in run.get("expected_artifacts") or []:
            item = dict(artifact)
            item.setdefault("command_index", run.get("command_index"))
            item.setdefault("center", run.get("center"))
            records.append(item)
    return records


def _expected_eval_artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    eval_roles = {
        "artifact_manifest",
        "eval_result",
        "metrics_long",
        "paper_table",
        "paper_table_manifest",
        "per_center_delta",
        "per_class_delta",
    }
    records: list[dict[str, Any]] = []
    for source in ("child_runs", "postprocess_runs"):
        for artifact in _artifact_records_by_source(manifest, source):
            role = str(artifact.get("role") or "")
            name = Path(str(artifact.get("path") or "")).name
            if role in eval_roles or name in {"metrics_long.csv", "eval_result.json"} or "paper_table" in name:
                records.append(artifact)
    return records


def _metric_centers_from_metrics_long(paths: list[Path]) -> set[str]:
    centers: set[str] = set()
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        for row in rows:
            if row.get("scope") == "center" and row.get("center"):
                metric = row.get("metric", "")
                if metric.endswith("macro_auroc") or metric.endswith("macro_auprc"):
                    centers.add(str(row["center"]))
    return centers


def _validate_command_records(commands: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(commands, list) or not commands:
        return ["run_manifest.commands must be a non-empty list"]
    for idx, command in enumerate(commands):
        if not isinstance(command, dict):
            errors.append(f"run_manifest.commands[{idx}] must be an object")
            continue
        argv = command.get("argv")
        if not isinstance(argv, list) or len(argv) < 2:
            errors.append(f"run_manifest.commands[{idx}].argv must record the python executable and entrypoint")
        if not _non_empty_text(command.get("cwd")):
            errors.append(f"run_manifest.commands[{idx}].cwd is required")
    return errors


def _validate_run_record_contract(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    purpose: str,
    result_summary: str,
    metric_summary: dict[str, Any],
    require_registration_ready: bool = False,
    registration_status: str | None = None,
) -> None:
    errors: list[str] = []
    if not _non_empty_text(purpose):
        errors.append("run_record.purpose must be non-empty")
    if not _non_empty_text(result_summary):
        errors.append("run_record.result_summary must be non-empty and explicit")
    if result_summary in _GENERIC_RESULT_SUMMARIES or result_summary.startswith("Run finalized with status="):
        errors.append("run_record.result_summary must not use a generic auto-generated summary")

    schema_version = manifest.get("manifest_schema_version")
    if schema_version != 2:
        errors.append("run_manifest.manifest_schema_version must be 2")
    if not _non_empty_text(manifest.get("run_id")):
        errors.append("run_manifest.run_id is required")
    if not _non_empty_text(manifest.get("status")):
        errors.append("run_manifest.status is required")
    if not _non_empty_text((manifest.get("git") or {}).get("commit")):
        errors.append("run_manifest.git.commit is required")
    if not _non_empty_text(manifest.get("config_hash_sha256")):
        errors.append("run_manifest.config_hash_sha256 is required")
    errors.extend(_validate_command_records(manifest.get("commands")))

    if not (run_dir / "run_config.resolved.yaml").exists() and not (run_dir / "run_config.resolved.json").exists():
        errors.append("resolved config artifact is required: run_config.resolved.yaml or run_config.resolved.json")
    if not (run_dir / "command.sh").exists():
        errors.append("command.sh is required")
    if not (run_dir / "selection.json").exists():
        errors.append("selection.json is required")

    paper = manifest.get("paper_protocol") or {}
    if not _non_empty_text(paper.get("mapping_version")):
        errors.append("paper_protocol.mapping_version is required")
    if not _non_empty_text(paper.get("mapping_hash")):
        errors.append("paper_protocol.mapping_hash is required")
    if not paper.get("class_order"):
        errors.append("paper_protocol.class_order is required")
    target_centers = _target_centers_from_protocol(paper)
    if not target_centers:
        errors.append("paper_protocol target centers are required")
    kshot = paper.get("kshot") or {}
    if not kshot.get("k") or not (kshot.get("subset_seed") or kshot.get("seed")):
        errors.append("paper_protocol.kshot must record k and seed/subset_seed")
    if not (paper.get("selection") or ((manifest.get("artifact_trace") or {}).get("selection_policy"))):
        errors.append("selection policy is required in paper_protocol or artifact_trace")

    trace = manifest.get("artifact_trace") or {}
    if trace.get("schema_version") != 1:
        errors.append("artifact_trace.schema_version must be 1")
    inputs = trace.get("inputs") or {}
    if not inputs.get("k500_refs"):
        errors.append("artifact_trace.inputs.k500_refs must be non-empty")
    expected_eval = _expected_eval_artifact_records(manifest)
    if not expected_eval:
        errors.append("expected eval artifacts must be declared in artifact_trace.expected_outputs")

    status = str(manifest.get("status") or "")
    if status == "succeeded":
        artifact_verification = manifest.get("artifact_verification") or {}
        if artifact_verification.get("passed") is not True:
            errors.append("run_manifest.artifact_verification.passed must be true for succeeded runs")
        metric_paths = _discover_metrics_paths(run_dir, manifest)
        if not metric_summary:
            errors.append("succeeded runs must include a non-empty metric_summary")
        observed_centers = _metric_centers_from_metrics_long(metric_paths)
        missing_centers = sorted(set(target_centers) - observed_centers)
        if missing_centers:
            errors.append(f"metric center coverage is missing target centers: {missing_centers}")

    if require_registration_ready:
        if registration_status in {"trusted", "provisional"} and status != "succeeded":
            errors.append("trusted/provisional registration requires run_manifest.status='succeeded'")
        for name in ("run_card.json", "run_file_index.json", "summary.md"):
            if not (run_dir / name).exists():
                errors.append(f"{name} is required before registry registration")

    if errors:
        raise RunRecordError("Run record contract failed:\n" + "\n".join(f"- {error}" for error in errors))


def _render_summary_md(card: dict[str, Any], file_index: dict[str, Any]) -> str:
    lines = [
        f"# Run Summary: {card['run_id']}",
        "",
        f"- Experiment: {card['experiment'].get('name', '')}",
        f"- Status: {card['result'].get('status', '')}",
        f"- Outcome: {card['result'].get('outcome', '')}",
        f"- Purpose: {card['experiment'].get('purpose', '')}",
        f"- Result: {card['result'].get('summary', '')}",
        "",
        "## Protocol",
        "",
        f"- Mapping: {card['protocol'].get('mapping_version', '')} / {card['protocol'].get('mapping_hash', '')}",
        f"- Class order: {'|'.join(card['protocol'].get('class_order') or [])}",
        f"- Target centers: {'|'.join(card['protocol'].get('target_centers') or [])}",
        f"- K-shot: {card['protocol'].get('kshot', {})}",
        "",
        "## Metric Summary",
        "",
    ]
    if card.get("metric_summary"):
        for view, metrics in card["metric_summary"].items():
            metric_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items()))
            lines.append(f"- {view}: {metric_text}")
    else:
        lines.append("- No metrics_long.csv summary discovered.")
    lines.extend(["", "## File Categories", ""])
    for category, items in file_index["categories"].items():
        lines.append(f"- {category}: {len(items)} files")
    lines.append("")
    return "\n".join(lines)


def finalize_run_record(
    run_dir: Path,
    *,
    purpose: str | None = None,
    result_summary: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Finalize a run directory with an agent-readable card, summary, and file index."""
    run_dir = Path(run_dir).expanduser().resolve()
    ensure_run_layout(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    metric_summary = _build_metric_summary(run_dir, manifest)
    experiment = manifest.get("experiment") or {}
    paper = manifest.get("paper_protocol") or {}
    centers = (paper.get("centers") or {}).get("target_4") or paper.get("target_centers") or []
    resolved_purpose = purpose or experiment.get("purpose") or experiment.get("description") or ""
    resolved_result = result_summary or (manifest.get("run_record") or {}).get("result_summary") or ""
    resolved_outcome = outcome or manifest.get("status") or "unknown"
    _validate_run_record_contract(
        run_dir,
        manifest,
        purpose=str(resolved_purpose),
        result_summary=str(resolved_result),
        metric_summary=metric_summary,
    )

    card = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "experiment": {
            "name": experiment.get("name", ""),
            "description": experiment.get("description", ""),
            "purpose": resolved_purpose,
        },
        "result": {
            "status": manifest.get("status", "unknown"),
            "outcome": resolved_outcome,
            "summary": resolved_result,
        },
        "protocol": {
            "mapping_version": paper.get("mapping_version", ""),
            "mapping_hash": paper.get("mapping_hash", ""),
            "class_order": list(paper.get("class_order") or []),
            "target_centers": list(centers),
            "kshot": paper.get("kshot", {}),
            "selection": paper.get("selection", {}),
        },
        "metric_summary": metric_summary,
        "artifacts": {
            "manifest": str(manifest_path),
            "file_index": str(run_dir / "run_file_index.json"),
            "summary": str(run_dir / "summary.md"),
        },
    }
    _write_json(run_dir / "run_card.json", card)
    manifest["run_record"] = {
        "finalized_at_utc": card["generated_at_utc"],
        "run_card": str(run_dir / "run_card.json"),
        "file_index": str(run_dir / "run_file_index.json"),
        "summary": str(run_dir / "summary.md"),
        "purpose": resolved_purpose,
        "result_summary": resolved_result,
        "outcome": resolved_outcome,
    }
    manifest["updated_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    _mirror_known_small_files(run_dir)
    file_index = _build_file_index(run_dir, manifest)
    _write_json(run_dir / "run_file_index.json", file_index)
    summary_md = _render_summary_md(card, file_index)
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    _mirror_known_small_files(run_dir)
    manifest = _refresh_launch_artifact_status(_read_json(manifest_path))
    _write_json(manifest_path, manifest)
    _mirror_known_small_files(run_dir)
    return card


def _path_ref(path: Path, *, output_root: Path) -> str:
    path = path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise RunRecordError(f"Run path is outside paths.output_root and cannot be registry-safe: {path}") from exc
    return "${paths.output_root}" if not rel else f"${{paths.output_root}}/{rel}"


def register_run_in_registry(
    *,
    registry_path: Path,
    local_config_path: Path,
    run_dir: Path,
    status: str = "provisional",
) -> dict[str, Any]:
    """Register a finalized run in the tracked active evidence registry."""
    registry_path = Path(registry_path).expanduser().resolve()
    local_config_path = Path(local_config_path).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    card_path = run_dir / "run_card.json"
    if not card_path.exists():
        raise RunRecordError(f"Missing run_card.json; finalize the run before registry registration: {card_path}")
    card = _read_json(card_path)
    manifest = _read_json(run_dir / "run_manifest.json")
    _validate_run_record_contract(
        run_dir,
        manifest,
        purpose=str((card.get("experiment") or {}).get("purpose") or ""),
        result_summary=str((card.get("result") or {}).get("summary") or ""),
        metric_summary=card.get("metric_summary") or {},
        require_registration_ready=True,
        registration_status=status,
    )
    registry = _read_yaml(registry_path)
    local_config = _read_yaml(local_config_path)
    output_root_raw = ((local_config.get("paths") or {}).get("output_root"))
    if not output_root_raw:
        raise RunRecordError(f"Local config has no paths.output_root: {local_config_path}")
    output_root = Path(str(output_root_raw)).expanduser().resolve()

    entry = {
        "run_id": card.get("run_id", run_dir.name),
        "experiment_name": (card.get("experiment") or {}).get("name", ""),
        "status": status,
        "outcome": (card.get("result") or {}).get("outcome", ""),
        "purpose": (card.get("experiment") or {}).get("purpose", ""),
        "result_summary": (card.get("result") or {}).get("summary", ""),
        "run_dir": _path_ref(run_dir, output_root=output_root),
        "run_card": _path_ref(card_path, output_root=output_root),
        "summary": _path_ref(run_dir / "summary.md", output_root=output_root),
        "mapping_version": (card.get("protocol") or {}).get("mapping_version", ""),
        "mapping_hash": (card.get("protocol") or {}).get("mapping_hash", ""),
        "updated": _utc_now().split("T", 1)[0],
    }
    managed_runs = list(registry.get("managed_runs") or [])
    key = (entry["run_id"], entry["experiment_name"], entry["run_dir"])
    next_runs = [
        item for item in managed_runs
        if (item.get("run_id"), item.get("experiment_name"), item.get("run_dir")) != key
    ]
    next_runs.append(entry)
    registry["managed_runs"] = next_runs
    run_catalog = dict(registry.get("run_catalog") or {})
    catalog_items = list(run_catalog.get(status) or [])
    catalog_ref = f"{entry['experiment_name']}/{entry['run_id']}"
    if catalog_ref not in catalog_items:
        catalog_items.append(catalog_ref)
    run_catalog[status] = catalog_items
    registry["run_catalog"] = run_catalog
    registry["updated"] = _utc_now().split("T", 1)[0]
    _write_yaml(registry_path, registry)
    return registry
