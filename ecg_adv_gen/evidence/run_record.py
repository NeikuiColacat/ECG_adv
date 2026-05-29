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
    resolved_result = result_summary or _default_result_summary(manifest, metric_summary)
    resolved_outcome = outcome or manifest.get("status") or "unknown"

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
        finalize_run_record(run_dir)
    card = _read_json(card_path)
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
