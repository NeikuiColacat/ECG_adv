"""Agent-readable run cards, file indexes, and registry registration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ecg_adv_gen.config.loader import interpolate_config


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
REGISTRATION_STATUSES = {"trusted", "provisional", "deprecated", "failed", "exploratory"}
_RUN_FILE_INDEX_INTEGRITY_SCOPE = (
    "required_manifest_outputs_required_file_inputs_and_env_excluding_self_index"
)


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


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    path = _absolute_no_resolve(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RunRecordError(f"Could not read stable JSON file: {path}: {exc}") from exc
    signatures = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        for item in (before, after, current)
    }
    if len(signatures) != 1 or stat.S_ISLNK(current.st_mode):
        raise RunRecordError(f"JSON file changed during read or is a symbolic link: {path}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunRecordError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunRecordError(f"JSON root must be an object: {path}")
    return data, hashlib.sha256(raw).hexdigest()


def _absolute_no_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_safe_write_path(path: Path, *, root: Path | None = None) -> None:
    path = _absolute_no_resolve(path)
    if path.is_symlink():
        raise RunRecordError(f"Refusing to write through symbolic link: {path}")
    if root is None:
        return
    root = Path(root).resolve()
    try:
        path.parent.relative_to(root)
    except ValueError as exc:
        raise RunRecordError(f"Refusing to write outside run_dir: {path}") from exc


@contextmanager
def _safe_parent_fd(path: Path, *, root: Path | None):
    path = _absolute_no_resolve(path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        if root is None:
            descriptors.append(os.open(path.parent, directory_flags))
        else:
            root = Path(root).resolve()
            try:
                relative_parent = path.parent.relative_to(root)
            except ValueError as exc:
                raise RunRecordError(f"Refusing to write outside run_dir: {path}") from exc
            descriptors.append(os.open(root, directory_flags))
            for part in relative_parent.parts:
                descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        yield descriptors[-1], path.name
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _reject_symlink_destination(parent_fd: int, name: str) -> None:
    try:
        target = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(target.st_mode):
        raise RunRecordError(f"Refusing to write through symbolic link: {name}")


def _atomic_write_text(path: Path, text: str, *, root: Path | None = None) -> None:
    path = _absolute_no_resolve(path)
    if root is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_write_path(path, root=root)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with _safe_parent_fd(path, root=root) as (parent_fd, name):
        _reject_symlink_destination(parent_fd, name)
        created = False
        try:
            fd = os.open(temporary, flags, 0o666, dir_fd=parent_fd)
            created = True
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            if created:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass


def _write_json(path: Path, data: dict[str, Any], *, root: Path | None = None) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        root=root,
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
    _atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=False))


def _relative(path: Path, root: Path) -> str:
    path = _absolute_no_resolve(path)
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _is_within_lexical(path: Path, root: Path) -> bool:
    try:
        _absolute_no_resolve(path).relative_to(_absolute_no_resolve(root))
    except ValueError:
        return False
    return True


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
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(run_dir, flags)
    try:
        for dirname, description in RUN_LAYOUT_DIRS.items():
            try:
                os.mkdir(dirname, dir_fd=root_fd)
            except FileExistsError:
                pass
            directory_fd = os.open(dirname, flags, dir_fd=root_fd)
            os.close(directory_fd)
            readme = run_dir / dirname / "README.md"
            if readme.is_symlink():
                raise RunRecordError(f"Refusing symbolic link run layout file: {readme}")
            if not readme.exists():
                _atomic_write_text(readme, f"# {dirname}\n\n{description}\n", root=run_dir)
    finally:
        os.close(root_fd)
    return {name: str((run_dir / name).resolve()) for name in RUN_LAYOUT_DIRS}


def _copy_small_file(src: Path, dst: Path, *, root: Path | None = None) -> None:
    src = _absolute_no_resolve(src)
    dst = _absolute_no_resolve(dst)
    _assert_safe_write_path(dst, root=root)
    if src == dst:
        return
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(src, source_flags)
    except FileNotFoundError:
        return
    with os.fdopen(source_fd, "rb") as source:
        source_stat = os.fstat(source.fileno())
        if not stat.S_ISREG(source_stat.st_mode):
            raise RunRecordError(f"Refusing non-regular mirror source: {src}")
        if source_stat.st_size > _SMALL_COPY_LIMIT_BYTES:
            return
        temporary = f".{dst.name}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with _safe_parent_fd(dst, root=root) as (parent_fd, name):
            _reject_symlink_destination(parent_fd, name)
            created = False
            try:
                destination_fd = os.open(temporary, flags, 0o666, dir_fd=parent_fd)
                created = True
                with os.fdopen(destination_fd, "wb") as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
                os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            finally:
                if created:
                    try:
                        os.unlink(temporary, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass


def _iter_expected_artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    records: list[dict[str, Any]] = []
    for artifact in expected.get("launch_artifacts") or []:
        if artifact.get("path"):
            records.append(
                {
                    "path": Path(str(artifact["path"])),
                    "role": artifact.get("role"),
                    "source": "launch",
                    "required": artifact.get("required", True),
                }
            )
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
                            "required": artifact.get("required", True),
                        }
                    )
    return records


def _iter_required_input_file_records(value: Any):
    if isinstance(value, dict):
        path_text = value.get("path")
        role = str(value.get("role") or "")
        if path_text and value.get("required") is True:
            path = Path(str(path_text)).expanduser()
            if not path.is_dir() and not role.endswith(("_dir", "_root")):
                yield {**value, "path": path, "source": "inputs"}
        for nested in value.values():
            yield from _iter_required_input_file_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_required_input_file_records(nested)


def _critical_records_by_path(manifest: dict[str, Any], run_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    self_index_paths = {
        str(_absolute_no_resolve(run_dir / "run_file_index.json")),
        str(_absolute_no_resolve(run_dir / "manifests/run_file_index.json")),
    }

    def add(record: dict[str, Any]) -> None:
        path = _absolute_no_resolve(Path(str(record["path"])))
        key = str(path)
        required = record.get("required") is not False
        if key in records:
            required = required or records[key].get("required") is not False
        records[key] = {**record, "path": path, "required": required}

    for record in _iter_expected_artifact_records(manifest):
        if record.get("required") is not False:
            declared = str(_absolute_no_resolve(Path(str(record["path"]))))
            if declared in self_index_paths:
                continue
            add(record)
    inputs = (manifest.get("artifact_trace") or {}).get("inputs") or {}
    for record in _iter_required_input_file_records(inputs):
        add(record)
    add({"path": run_dir / "env.json", "role": "environment", "source": "run_record", "required": True})
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


def _build_file_index(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    include_sha256: bool = True,
    content_overrides: dict[str, bytes] | None = None,
    generated_at_utc: str | None = None,
    sha256_cache: dict[tuple[Any, ...], str] | None = None,
) -> dict[str, Any]:
    overrides = {
        str(_absolute_no_resolve(Path(path))): payload
        for path, payload in (content_overrides or {}).items()
    }
    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in RUN_LAYOUT_DIRS}
    expected_records = _iter_expected_artifact_records(manifest)
    expected_by_path = {
        str(_absolute_no_resolve(record["path"])): record
        for record in expected_records
    }
    critical_by_path = _critical_records_by_path(manifest, run_dir)
    declared_by_path = {**critical_by_path, **expected_by_path}
    self_paths = {
        str(_absolute_no_resolve(run_dir / "run_file_index.json")),
        str(_absolute_no_resolve(run_dir / "manifests/run_file_index.json")),
    }
    seen_paths: set[str] = set()
    actual_paths = {p for p in run_dir.rglob("*") if p.is_file()}
    virtual_paths = {
        Path(path)
        for path in overrides
        if _is_within_lexical(Path(path), run_dir)
    }
    for path in sorted(actual_paths | virtual_paths):
        declared_key = str(_absolute_no_resolve(path))
        if declared_key in self_paths:
            continue
        seen_paths.add(declared_key)
        category = _category_for(path, run_dir)
        override = overrides.get(declared_key)
        record = {
            "relative_path": _relative(path, run_dir),
            "path": declared_key,
            "size_bytes": len(override) if override is not None else path.stat().st_size,
            "expected_artifact": declared_key in expected_by_path,
            "external": False,
        }
        if include_sha256 and declared_key in critical_by_path:
            if override is not None:
                record["sha256"] = hashlib.sha256(override).hexdigest()
            else:
                st = path.stat()
                cache_key = (
                    declared_key,
                    st.st_dev,
                    st.st_ino,
                    st.st_size,
                    st.st_mtime_ns,
                    st.st_ctime_ns,
                )
                if sha256_cache is not None and cache_key in sha256_cache:
                    record["sha256"] = sha256_cache[cache_key]
                else:
                    record["sha256"] = _sha256_file(path)
                    if sha256_cache is not None:
                        sha256_cache[cache_key] = record["sha256"]
        categories.setdefault(category, []).append(record)
    for declared_key, artifact in declared_by_path.items():
        if declared_key in seen_paths or declared_key in self_paths:
            continue
        path = Path(declared_key)
        seen_paths.add(declared_key)
        category = _category_for(path, run_dir)
        record = {
            "relative_path": _relative(path, run_dir),
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_artifact": declared_key in expected_by_path,
            "external": True,
            "role": artifact.get("role"),
            "source": artifact.get("source"),
            "center": artifact.get("center"),
            "command_index": artifact.get("command_index"),
        }
        if include_sha256 and declared_key in critical_by_path and path.is_file():
            record["sha256"] = _sha256_file(path)
        categories.setdefault(category, []).append(record)
    return {
        "schema_version": 2,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "run_dir": str(run_dir),
        "integrity": {
            "algorithm": "sha256",
            "scope": _RUN_FILE_INDEX_INTEGRITY_SCOPE,
        },
        "categories": categories,
    }


def verify_run_file_index(run_dir: Path) -> dict[str, Any]:
    """Verify manifest-declared reproduction-critical files against their index."""
    run_dir = Path(run_dir).expanduser().resolve()
    errors: list[str] = []
    index_sha256 = ""

    def report(
        status: str,
        *,
        checked_count: int = 0,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "passed": status == "verified",
            "status": status,
            "checked_count": checked_count,
            "index_sha256": index_sha256,
            "warnings": warnings or [],
            "errors": errors,
        }

    try:
        index, index_sha256 = _read_json_snapshot(run_dir / "run_file_index.json")
    except RunRecordError as exc:
        errors.append(str(exc))
        return report("failed")

    schema_version = index.get("schema_version")
    if schema_version == 1:
        return report(
            "unchecked",
            warnings=["run_file_index schema_version=1 has no required SHA-256 contract"],
        )
    if schema_version != 2:
        errors.append(f"unsupported run_file_index schema_version={schema_version!r}")
        return report("failed")
    integrity = index.get("integrity") or {}
    if integrity.get("algorithm") != "sha256" or integrity.get("scope") != _RUN_FILE_INDEX_INTEGRITY_SCOPE:
        errors.append("run_file_index integrity algorithm/scope mismatch")
        return report("failed")

    try:
        manifest = _read_json(run_dir / "run_manifest.json")
    except RunRecordError as exc:
        errors.append(str(exc))
        return report("failed")

    indexed_by_path: dict[str, dict[str, Any]] = {}
    categories = index.get("categories") or {}
    if not isinstance(categories, dict):
        errors.append("run_file_index.categories must be an object")
        return report("failed")
    records = [
        record
        for category_records in categories.values()
        if isinstance(category_records, list)
        for record in category_records
        if isinstance(record, dict) and record.get("path")
    ]
    for record in records:
        key = str(_absolute_no_resolve(Path(str(record["path"]))))
        if key not in indexed_by_path or record.get("sha256"):
            indexed_by_path[key] = record

    checked_count = 0
    for declared_key, artifact in _critical_records_by_path(manifest, run_dir).items():
        path = Path(declared_key)
        record = indexed_by_path.get(declared_key)
        if path.is_symlink():
            errors.append(f"reproduction-critical file is a symbolic link: {path}")
            continue
        if not path.is_file():
            if artifact.get("required") is not False:
                errors.append(f"reproduction-critical file is missing: {path}")
            continue
        if record is None:
            errors.append(f"reproduction-critical file is absent from run_file_index: {path}")
            continue
        if not record.get("sha256"):
            errors.append(f"reproduction-critical file has no sha256: {path}")
            continue
        label = str(record.get("relative_path") or path)
        checked_count += 1
        try:
            before = path.stat()
            actual_sha256 = _sha256_file(path)
            after = path.stat()
        except OSError as exc:
            errors.append(f"reproduction-critical file could not be verified: {path}: {exc}")
            continue
        actual_size = after.st_size
        if record.get("size_bytes") != actual_size:
            errors.append(
                f"size_bytes mismatch for {label}: indexed={record.get('size_bytes')!r}, actual={actual_size}"
            )
        before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_signature != after_signature:
            errors.append(f"reproduction-critical file changed during verification: {label}")
        if record.get("sha256") != actual_sha256:
            errors.append(f"sha256 mismatch for {label}")

    return report("failed" if errors else "verified", checked_count=checked_count)


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _target_centers_from_protocol(protocol: dict[str, Any]) -> list[str]:
    centers = protocol.get("centers") or {}
    values = centers.get("target_4") or protocol.get("target_centers") or []
    return [str(item) for item in values if str(item)]


def _k500_centers_from_manifest(manifest: dict[str, Any]) -> list[str]:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    centers: list[str] = []
    for child in expected.get("child_runs") or []:
        center = str(child.get("center") or "")
        if center and center not in centers:
            centers.append(center)
    if centers:
        return centers
    return _target_centers_from_protocol(manifest.get("paper_protocol") or {})


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


_REGISTRATION_REQUIRED_ROLE_ALIASES = {
    "run_manifest": {"run_manifest"},
    "run_card": {"run_card"},
    "run_file_index": {"run_file_index"},
    "run_summary": {"run_summary"},
    "resolved_config": {"resolved_config", "run_config_yaml", "run_config_json"},
    "command": {"command", "command_sh"},
    "data_manifest": {"data_manifest"},
    "selection": {"selection", "selection_record"},
    "k500_refs": {"k500_refs", "k500_ref_ids"},
    "metrics_source": {"metrics_source", "metrics_long"},
}


def _declared_artifact_roles(manifest: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for artifact in _artifact_records_by_source(manifest, "launch"):
        if artifact.get("role"):
            roles.add(str(artifact["role"]))
    for source in ("child_runs", "postprocess_runs"):
        for artifact in _artifact_records_by_source(manifest, source):
            if artifact.get("role"):
                roles.add(str(artifact["role"]))
    return roles


def _missing_registration_roles(manifest: dict[str, Any]) -> list[str]:
    declared = _declared_artifact_roles(manifest)
    missing: list[str] = []
    for canonical_role, aliases in _REGISTRATION_REQUIRED_ROLE_ALIASES.items():
        if not declared.intersection(aliases):
            missing.append(canonical_role)
    return missing


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


def _expected_metrics_long_artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_expected_artifact_records(manifest):
        role = str(record.get("role") or "")
        path = Path(str(record.get("path") or ""))
        if role == "metrics_long" or path.name == "metrics_long.csv":
            records.append(record)
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _expected_mapping_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    paper = manifest.get("paper_protocol") or {}
    trace_metrics = (manifest.get("artifact_trace") or {}).get("metrics") or {}
    return {
        "mapping_version": trace_metrics.get("mapping_version") or paper.get("mapping_version"),
        "mapping_hash": trace_metrics.get("mapping_hash") or paper.get("mapping_hash"),
        "class_order": trace_metrics.get("class_order") or paper.get("class_order") or [],
    }


def _validate_selection_content(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        selection = _read_json(run_dir / "selection.json")
    except RunRecordError as exc:
        return [str(exc)]
    expected_policy = (
        ((manifest.get("artifact_trace") or {}).get("selection_policy") or {}).get("policy")
        or ((manifest.get("paper_protocol") or {}).get("selection") or {}).get("policy")
    )
    actual_policy = (selection.get("selection_policy") or {}).get("policy")
    if expected_policy and actual_policy != expected_policy:
        errors.append(f"selection.json policy={actual_policy!r}, expected {expected_policy!r}")
    safety = selection.get("selection_safety") or {}
    if safety.get("heldout_target_labels_used_for_selection") is not False:
        errors.append("selection.json held-out target labels must be explicitly denied")
    if safety.get("full_target_distribution_used_for_tuning") is not False:
        errors.append("selection.json full target distribution tuning must be explicitly denied")
    if safety.get("forbidden_reference_found") is not False:
        errors.append("selection.json forbidden selection reference scan must be false")
    metadata = _expected_mapping_metadata(manifest)
    protocol = selection.get("paper_protocol") or {}
    if protocol:
        for key in ("mapping_version", "mapping_hash"):
            if metadata.get(key) and protocol.get(key) and protocol.get(key) != metadata[key]:
                errors.append(f"selection.json {key}={protocol.get(key)!r}, expected {metadata[key]!r}")
    return errors


def _validate_k500_ref_ids_content(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = run_dir / "k500_ref_ids.json"
    try:
        artifact = _read_json(path)
    except RunRecordError as exc:
        return [str(exc)]

    paper = manifest.get("paper_protocol") or {}
    kshot = paper.get("kshot") or {}
    expected_centers = _k500_centers_from_manifest(manifest)
    expected_k = int(kshot.get("k", 0) or 0)
    expected_seed = int(kshot.get("subset_seed", kshot.get("seed", 0)) or 0)
    metadata = _expected_mapping_metadata(manifest)
    artifact_protocol = artifact.get("paper_protocol") or {}
    for key in ("mapping_version", "mapping_hash"):
        if metadata.get(key) and artifact_protocol.get(key) != metadata[key]:
            errors.append(f"k500_ref_ids.json {key}={artifact_protocol.get(key)!r}, expected {metadata[key]!r}")
    expected_class_order = [str(item) for item in metadata.get("class_order") or []]
    artifact_class_order = [str(item) for item in artifact_protocol.get("class_order") or []]
    if expected_class_order:
        if not artifact_class_order:
            errors.append("k500_ref_ids.json class_order is required")
        elif artifact_class_order != expected_class_order:
            errors.append("k500_ref_ids.json class_order does not match run manifest")

    centers = artifact.get("centers") or {}
    if not isinstance(centers, dict) or not centers:
        errors.append("k500_ref_ids.json centers must be a non-empty object")
        return errors
    for center in expected_centers:
        payload = centers.get(center)
        if not isinstance(payload, dict):
            errors.append(f"k500_ref_ids.json missing center {center}")
            continue
        ids_ordered = [str(item) for item in payload.get("ref_record_ids_ordered") or []]
        ids_sorted = [str(item) for item in payload.get("ref_record_ids_sorted") or []]
        if int(payload.get("k", -1)) != expected_k:
            errors.append(f"k500_ref_ids.json {center} k={payload.get('k')!r}, expected {expected_k}")
        if int(payload.get("selection_seed", -1)) != expected_seed:
            errors.append(
                f"k500_ref_ids.json {center} selection_seed={payload.get('selection_seed')!r}, expected {expected_seed}"
            )
        if len(ids_ordered) != expected_k or len(ids_sorted) != expected_k or len(set(ids_ordered)) != expected_k:
            errors.append(f"k500_ref_ids.json {center} ref_record_ids count/uniqueness mismatch")
        expected_hash = _sha256_lines(sorted(ids_ordered))
        if payload.get("ref_record_ids_sha256") != expected_hash:
            errors.append(f"k500_ref_ids.json {center} ref_record_ids_sha256 mismatch")
        source_meta_text = _non_empty_text(payload.get("source_ref_meta_path"))
        if not source_meta_text:
            errors.append(f"k500_ref_ids.json {center} source_ref_meta_path is required")
            continue
        source_meta = Path(source_meta_text).expanduser()
        if not source_meta.exists():
            errors.append(f"k500_ref_ids.json {center} source_ref_meta_path is missing")
        elif payload.get("source_ref_meta_sha256") != _sha256_file(source_meta):
            errors.append(f"k500_ref_ids.json {center} source_ref_meta_sha256 mismatch")
    return errors


def _expects_launch_artifact_role(manifest: dict[str, Any], role: str) -> bool:
    expected = ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {})
    launch_artifacts = expected.get("launch_artifacts") or []
    if launch_artifacts:
        return any(
            artifact.get("role") == role and artifact.get("required") is not False
            for artifact in launch_artifacts
        )
    declared = expected.get("launch_artifacts_declared") or []
    if declared:
        role_by_name = {
            "k500_ref_ids.json": "k500_ref_ids",
            "selection.json": "selection_record",
        }
        return any(role_by_name.get(str(name)) == role for name in declared)
    return True


def _metrics_long_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _validate_metrics_long_content(paths: list[Path], manifest: dict[str, Any]) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    source_files: set[str] = set()
    metadata = _expected_mapping_metadata(manifest)
    expected_class_order = "|".join(str(item) for item in metadata.get("class_order") or [])
    for path in paths:
        try:
            rows = _metrics_long_rows(path)
        except OSError as exc:
            errors.append(f"metrics_long.csv could not be read: {path}: {exc}")
            continue
        if not rows:
            errors.append(f"metrics_long.csv is empty: {path}")
            continue
        for row_idx, row in enumerate(rows, start=2):
            for key in ("mapping_version", "mapping_hash"):
                if metadata.get(key):
                    actual = _non_empty_text(row.get(key))
                    if not actual:
                        errors.append(f"metrics_long.csv {path}:{row_idx} {key} is required")
                    elif actual != metadata[key]:
                        errors.append(f"metrics_long.csv {path}:{row_idx} {key}={actual!r}, expected {metadata[key]!r}")
            if expected_class_order:
                actual_class_order = _non_empty_text(row.get("class_order"))
                if not actual_class_order:
                    errors.append(f"metrics_long.csv {path}:{row_idx} class_order is required")
                elif actual_class_order != expected_class_order:
                    errors.append(f"metrics_long.csv {path}:{row_idx} class_order mismatch")
            if row.get("scope") == "center" and row.get("metric", "").endswith(("macro_auroc", "macro_auprc")):
                source_file = _non_empty_text(row.get("source_file"))
                if not source_file:
                    errors.append(f"metrics_long.csv {path}:{row_idx} source_file is required for center macro rows")
                else:
                    source_files.add(source_file)
                    if not Path(source_file).expanduser().exists():
                        errors.append(f"metrics_long.csv {path}:{row_idx} source_file does not exist: {source_file}")
    return errors, source_files


def _validate_paper_table_content(
    manifest: dict[str, Any],
    *,
    metrics_paths: list[Path],
    metrics_source_files: set[str],
) -> list[str]:
    errors: list[str] = []
    metrics_path_set = {str(path) for path in metrics_paths}
    for record in _expected_eval_artifact_records(manifest):
        role = str(record.get("role") or "")
        path = Path(str(record.get("path") or "")).expanduser()
        if role == "paper_table" and path.exists():
            try:
                with path.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
            except OSError as exc:
                errors.append(f"paper_table could not be read: {path}: {exc}")
                continue
            if not rows:
                errors.append(f"paper_table is empty: {path}")
                continue
            if "source_files" not in (rows[0].keys() if rows else []):
                errors.append(f"paper_table source_files column is required: {path}")
                continue
            for row_idx, row in enumerate(rows, start=2):
                source_files_text = _non_empty_text(row.get("source_files"))
                if not source_files_text:
                    errors.append(f"paper_table {path}:{row_idx} source_files is required")
                    continue
                for source_file in [item for item in source_files_text.split("|") if item]:
                    if source_file not in metrics_source_files or not Path(source_file).expanduser().exists():
                        errors.append(f"paper_table {path}:{row_idx} source_files contains untraced source: {source_file}")
        if role == "paper_table_manifest" and path.exists():
            try:
                manifest_payload = _read_json(path)
            except RunRecordError as exc:
                errors.append(str(exc))
                continue
            metrics_long = _non_empty_text(manifest_payload.get("metrics_long"))
            if not metrics_long:
                errors.append(f"paper_table_manifest metrics_long is required: {path}")
            elif metrics_long not in metrics_path_set:
                errors.append(f"paper_table_manifest metrics_long is not a discovered metrics_long.csv: {metrics_long}")
    return errors


def _validate_content_artifacts(run_dir: Path, manifest: dict[str, Any], metric_paths: list[Path]) -> list[str]:
    errors: list[str] = []
    if _expects_launch_artifact_role(manifest, "selection_record"):
        errors.extend(_validate_selection_content(run_dir, manifest))
    if _expects_launch_artifact_role(manifest, "k500_ref_ids"):
        errors.extend(_validate_k500_ref_ids_content(run_dir, manifest))
    metrics_errors, source_files = _validate_metrics_long_content(metric_paths, manifest)
    errors.extend(metrics_errors)
    errors.extend(
        _validate_paper_table_content(
            manifest,
            metrics_paths=metric_paths,
            metrics_source_files=source_files,
        )
    )
    return errors


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
    if not (run_dir / "env.json").exists():
        errors.append("env.json is required")
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
    if _expects_launch_artifact_role(manifest, "k500_ref_ids") and not inputs.get("k500_refs"):
        errors.append("artifact_trace.inputs.k500_refs must be non-empty")
    expected_artifacts = _iter_expected_artifact_records(manifest)
    if not expected_artifacts:
        errors.append("expected artifacts must be declared in artifact_trace.expected_outputs")
    expected_eval = _expected_eval_artifact_records(manifest)
    expected_metrics_long = _expected_metrics_long_artifact_records(manifest)
    metric_paths = _discover_metrics_paths(run_dir, manifest)

    status = str(manifest.get("status") or "")
    if status == "succeeded":
        artifact_verification = manifest.get("artifact_verification") or {}
        if artifact_verification.get("passed") is not True:
            errors.append("run_manifest.artifact_verification.passed must be true for succeeded runs")
        if _expects_launch_artifact_role(manifest, "k500_ref_ids") and not expected_eval:
            errors.append("expected eval artifacts must be declared for K500 target runs")
        if expected_metrics_long and not metric_summary:
            errors.append("succeeded runs must include a non-empty metric_summary")
        if expected_metrics_long:
            observed_centers = _metric_centers_from_metrics_long(metric_paths)
            missing_centers = sorted(set(target_centers) - observed_centers)
            if missing_centers:
                errors.append(f"metric center coverage is missing target centers: {missing_centers}")

    if require_registration_ready:
        if registration_status in {"trusted", "provisional"}:
            missing_roles = _missing_registration_roles(manifest)
            if missing_roles:
                errors.append(
                    "registry registration requires artifact roles: "
                    + ", ".join(missing_roles)
                )
        if registration_status in {"trusted", "provisional"} and status != "succeeded":
            errors.append("trusted/provisional registration requires run_manifest.status='succeeded'")
        for name in ("run_card.json", "run_file_index.json", "summary.md"):
            if not (run_dir / name).exists():
                errors.append(f"{name} is required before registry registration")

    if not errors:
        errors.extend(_validate_content_artifacts(run_dir, manifest, metric_paths))

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


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"
    ).encode("utf-8")


def _refresh_launch_artifacts_for_payloads(
    manifest: dict[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    out = dict(manifest)
    trace = dict(out.get("artifact_trace") or {})
    expected = dict(trace.get("expected_outputs") or {})
    refreshed: list[dict[str, Any]] = []
    for artifact in expected.get("launch_artifacts") or []:
        item = dict(artifact)
        path = _absolute_no_resolve(Path(str(item.get("path", ""))))
        override = payloads.get(str(path))
        item["exists"] = override is not None or path.exists()
        item["size_bytes"] = (
            len(override)
            if override is not None
            else path.stat().st_size if path.is_file() else None
        )
        refreshed.append(item)
    if refreshed:
        expected["launch_artifacts"] = refreshed
        trace["expected_outputs"] = expected
        out["artifact_trace"] = trace
    return out


def _run_card_for_manifest(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    purpose: str,
    result_summary: str,
    outcome: str,
    metric_summary: dict[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    experiment = manifest.get("experiment") or {}
    paper = manifest.get("paper_protocol") or {}
    centers = (paper.get("centers") or {}).get("target_4") or paper.get("target_centers") or []
    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "experiment": {
            "name": experiment.get("name", ""),
            "description": experiment.get("description", ""),
            "purpose": purpose,
        },
        "result": {
            "status": manifest.get("status", "unknown"),
            "outcome": outcome,
            "summary": result_summary,
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
            "manifest": str(run_dir / "run_manifest.json"),
            "file_index": str(run_dir / "run_file_index.json"),
            "summary": str(run_dir / "summary.md"),
        },
    }


def finalize_run_record(
    run_dir: Path,
    *,
    purpose: str | None = None,
    result_summary: str | None = None,
    outcome: str | None = None,
    final_manifest: Mapping[str, Any] | None = None,
    publish_hook: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Publish derivatives first and the final manifest as the sole commit marker."""
    run_dir = Path(run_dir).expanduser().resolve()
    ensure_run_layout(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    persisted = _read_json(manifest_path)
    manifest = json.loads(json.dumps(final_manifest if final_manifest is not None else persisted, default=str))
    if manifest.get("run_id") != persisted.get("run_id"):
        raise RunRecordError("final manifest run_id does not match the persisted run")
    metric_summary = _build_metric_summary(run_dir, manifest)
    experiment = manifest.get("experiment") or {}
    manifest_run_record = manifest.get("run_record") or {}
    resolved_purpose = purpose or manifest_run_record.get("purpose") or experiment.get("purpose") or experiment.get("description") or ""
    resolved_result = result_summary or manifest_run_record.get("result_summary") or ""
    resolved_outcome = outcome or manifest_run_record.get("outcome") or manifest_run_record.get("status") or manifest.get("status") or "unknown"
    _validate_run_record_contract(
        run_dir,
        manifest,
        purpose=str(resolved_purpose),
        result_summary=str(resolved_result),
        metric_summary=metric_summary,
    )

    finalized_at = _utc_now()
    manifest["run_record"] = {
        **manifest_run_record,
        "finalized_at_utc": finalized_at,
        "run_card": str(run_dir / "run_card.json"),
        "file_index": str(run_dir / "run_file_index.json"),
        "summary": str(run_dir / "summary.md"),
        "purpose": resolved_purpose,
        "result_summary": resolved_result,
        "outcome": resolved_outcome,
    }
    manifest["updated_at_utc"] = finalized_at
    card = _run_card_for_manifest(
        run_dir,
        manifest,
        purpose=str(resolved_purpose),
        result_summary=str(resolved_result),
        outcome=str(resolved_outcome),
        metric_summary=metric_summary,
        generated_at_utc=finalized_at,
    )

    def publish(stage: str, path: Path, payload: bytes) -> None:
        if publish_hook is not None:
            publish_hook(stage, path)
        _atomic_write_text(path, payload.decode("utf-8"), root=run_dir)

    # Static mirrors are also completed before the manifest commit marker.
    for source_name, destination_name in (
        ("run_config.resolved.yaml", "configs/run_config.resolved.yaml"),
        ("run_config.resolved.json", "configs/run_config.resolved.json"),
        ("command.sh", "configs/command.sh"),
        ("data_manifest.json", "manifests/data_manifest.json"),
        ("k500_ref_ids.json", "manifests/k500_ref_ids.json"),
        ("selection.json", "manifests/selection.json"),
    ):
        source = run_dir / source_name
        if source.is_file() and source.stat().st_size <= _SMALL_COPY_LIMIT_BYTES:
            publish(f"mirror:{destination_name}", run_dir / destination_name, source.read_bytes())

    card_bytes = _json_bytes(card)
    summary_bytes = b""
    index_bytes = b""
    sha256_cache: dict[tuple[Any, ...], str] = {}
    previous_signature: tuple[bytes, bytes, bytes] | None = None
    for _ in range(12):
        manifest_bytes = _json_bytes(manifest)
        virtual = {
            str(manifest_path): manifest_bytes,
            str(run_dir / "manifests/run_manifest.snapshot.json"): manifest_bytes,
            str(run_dir / "run_card.json"): card_bytes,
            str(run_dir / "manifests/run_card.json"): card_bytes,
            str(run_dir / "summary.md"): summary_bytes,
            str(run_dir / "reports/summary.md"): summary_bytes,
        }
        preliminary = _build_file_index(
            run_dir,
            manifest,
            content_overrides=virtual,
            generated_at_utc=finalized_at,
            sha256_cache=sha256_cache,
        )
        summary_bytes = _render_summary_md(card, preliminary).encode("utf-8")
        virtual[str(run_dir / "summary.md")] = summary_bytes
        virtual[str(run_dir / "reports/summary.md")] = summary_bytes
        file_index = _build_file_index(
            run_dir,
            manifest,
            content_overrides=virtual,
            generated_at_utc=finalized_at,
            sha256_cache=sha256_cache,
        )
        index_bytes = _json_bytes(file_index)
        virtual.update(
            {
                str(run_dir / "run_file_index.json"): index_bytes,
                str(run_dir / "manifests/run_file_index.json"): index_bytes,
            }
        )
        refreshed = _refresh_launch_artifacts_for_payloads(manifest, virtual)
        refreshed_bytes = _json_bytes(refreshed)
        signature = (refreshed_bytes, summary_bytes, index_bytes)
        manifest = refreshed
        if signature == previous_signature:
            break
        previous_signature = signature
    else:
        raise RunRecordError("finalizer derivative payloads did not converge")

    # Rebuild once with the converged manifest bytes, then validate the exact
    # payload that will be committed.
    manifest_bytes = _json_bytes(manifest)
    virtual = {
        str(manifest_path): manifest_bytes,
        str(run_dir / "manifests/run_manifest.snapshot.json"): manifest_bytes,
        str(run_dir / "run_card.json"): card_bytes,
        str(run_dir / "manifests/run_card.json"): card_bytes,
        str(run_dir / "summary.md"): summary_bytes,
        str(run_dir / "reports/summary.md"): summary_bytes,
    }
    file_index = _build_file_index(
        run_dir,
        manifest,
        content_overrides=virtual,
        generated_at_utc=finalized_at,
        sha256_cache=sha256_cache,
    )
    index_bytes = _json_bytes(file_index)
    manifest_record = next(
        record
        for records in file_index["categories"].values()
        for record in records
        if record.get("path") == str(manifest_path)
    )
    if manifest_record.get("sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise RunRecordError("prebuilt file index does not bind the final manifest payload")
    if card["result"]["status"] != manifest.get("status"):
        raise RunRecordError("prebuilt run card status does not match final manifest status")

    for stage, path, payload in (
        ("run_card", run_dir / "run_card.json", card_bytes),
        ("run_card_mirror", run_dir / "manifests/run_card.json", card_bytes),
        ("summary", run_dir / "summary.md", summary_bytes),
        ("summary_mirror", run_dir / "reports/summary.md", summary_bytes),
        ("manifest_snapshot", run_dir / "manifests/run_manifest.snapshot.json", manifest_bytes),
        ("file_index", run_dir / "run_file_index.json", index_bytes),
        ("file_index_mirror", run_dir / "manifests/run_file_index.json", index_bytes),
    ):
        publish(stage, path, payload)
    # No filesystem write is allowed after this final commit marker.
    publish("manifest_commit", manifest_path, manifest_bytes)
    return card


def _path_ref(path: Path, *, output_root: Path) -> str:
    path = path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    try:
        rel = path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise RunRecordError(f"Run path is outside paths.output_root and cannot be registry-safe: {path}") from exc
    return "${paths.output_root}" if not rel else f"${{paths.output_root}}/{rel}"


def _resolve_registration_status(
    *,
    requested_status: str | None,
    manifest: dict[str, Any],
    card: dict[str, Any],
) -> str:
    if requested_status and requested_status != "auto":
        if requested_status not in REGISTRATION_STATUSES:
            raise RunRecordError(f"Unsupported registration status: {requested_status!r}")
        return requested_status
    run_record = manifest.get("run_record") or {}
    experiment = manifest.get("experiment") or {}
    result = card.get("result") or {}
    candidates = [
        run_record.get("registration_status"),
        run_record.get("status"),
        result.get("outcome"),
        experiment.get("status"),
    ]
    for candidate in candidates:
        if str(candidate) in REGISTRATION_STATUSES:
            return str(candidate)
    return "provisional"


def _validate_run_card_matches_manifest(run_dir: Path, card: dict[str, Any], manifest: dict[str, Any]) -> None:
    experiment = manifest.get("experiment") if isinstance(manifest.get("experiment"), dict) else {}
    paper = manifest.get("paper_protocol") if isinstance(manifest.get("paper_protocol"), dict) else {}
    run_record = manifest.get("run_record") if isinstance(manifest.get("run_record"), dict) else {}
    card_experiment = card.get("experiment") if isinstance(card.get("experiment"), dict) else {}
    card_result = card.get("result") if isinstance(card.get("result"), dict) else {}
    card_protocol = card.get("protocol") if isinstance(card.get("protocol"), dict) else {}
    expected_centers = (paper.get("centers") or {}).get("target_4") or paper.get("target_centers") or []
    checks = {
        "run_id": (card.get("run_id"), manifest.get("run_id")),
        "run_dir": (str(Path(str(card.get("run_dir") or "")).expanduser().resolve()), str(run_dir)),
        "experiment.name": (card_experiment.get("name"), experiment.get("name")),
        "experiment.purpose": (card_experiment.get("purpose"), run_record.get("purpose")),
        "result.status": (card_result.get("status"), manifest.get("status")),
        "result.summary": (card_result.get("summary"), run_record.get("result_summary")),
        "protocol.mapping_version": (card_protocol.get("mapping_version"), paper.get("mapping_version")),
        "protocol.mapping_hash": (card_protocol.get("mapping_hash"), paper.get("mapping_hash")),
        "protocol.class_order": (list(card_protocol.get("class_order") or []), list(paper.get("class_order") or [])),
        "protocol.target_centers": (list(card_protocol.get("target_centers") or []), list(expected_centers)),
        "protocol.kshot": (card_protocol.get("kshot") or {}, paper.get("kshot") or {}),
    }
    mismatches = [field for field, (observed, expected) in checks.items() if observed != expected]
    if mismatches:
        raise RunRecordError(
            "run_card.json disagrees with run_manifest.json: " + ", ".join(mismatches)
        )


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
    summary_path = run_dir / "summary.md"
    index_path = run_dir / "run_file_index.json"
    registry = _read_yaml(registry_path)
    local_config = interpolate_config(_read_yaml(local_config_path))
    output_root_raw = ((local_config.get("paths") or {}).get("output_root"))
    if not output_root_raw:
        raise RunRecordError(f"Local config has no paths.output_root: {local_config_path}")
    output_root = Path(str(output_root_raw)).expanduser().resolve()
    run_dir_ref = _path_ref(run_dir, output_root=output_root)
    card_ref = _path_ref(card_path, output_root=output_root)
    summary_ref = _path_ref(summary_path, output_root=output_root)
    index_ref = _path_ref(index_path, output_root=output_root)
    if not card_path.exists():
        raise RunRecordError(f"Missing run_card.json; finalize the run before registry registration: {card_path}")
    card = _read_json(card_path)
    manifest = _read_json(run_dir / "run_manifest.json")
    resolved_status = _resolve_registration_status(
        requested_status=status,
        manifest=manifest,
        card=card,
    )
    _validate_run_card_matches_manifest(run_dir, card, manifest)
    _validate_run_record_contract(
        run_dir,
        manifest,
        purpose=str((card.get("experiment") or {}).get("purpose") or ""),
        result_summary=str((card.get("result") or {}).get("summary") or ""),
        metric_summary=card.get("metric_summary") or {},
        require_registration_ready=True,
        registration_status=resolved_status,
    )
    integrity = verify_run_file_index(run_dir)
    if resolved_status in {"trusted", "provisional"} and not integrity["passed"]:
        details = [*integrity["errors"], *integrity["warnings"]]
        raise RunRecordError(
            "Run file integrity verification failed:\n"
            + "\n".join(f"- {detail}" for detail in details)
        )
    index_sha256 = str(integrity.get("index_sha256") or "")
    if not index_sha256:
        raise RunRecordError("Run file index could not be read as a stable snapshot")

    entry = {
        "run_id": card.get("run_id", run_dir.name),
        "experiment_name": (card.get("experiment") or {}).get("name", ""),
        "status": resolved_status,
        "outcome": (card.get("result") or {}).get("outcome", ""),
        "purpose": (card.get("experiment") or {}).get("purpose", ""),
        "result_summary": (card.get("result") or {}).get("summary", ""),
        "run_dir": run_dir_ref,
        "run_card": card_ref,
        "summary": summary_ref,
        "run_file_index": index_ref,
        "run_file_index_sha256": index_sha256,
        "replay_status": (
            "replay_ready" if resolved_status in {"trusted", "provisional"}
            else "historical_evaluation_only"
        ),
        "mapping_version": (card.get("protocol") or {}).get("mapping_version", ""),
        "mapping_hash": (card.get("protocol") or {}).get("mapping_hash", ""),
        "updated": _utc_now().split("T", 1)[0],
    }
    managed_runs_raw = registry.get("managed_runs")
    if managed_runs_raw is None:
        managed_runs = []
    elif isinstance(managed_runs_raw, list):
        managed_runs = list(managed_runs_raw)
    else:
        raise RunRecordError("Registry managed_runs must be a list")
    identity = (entry["run_id"], entry["experiment_name"])
    for item in managed_runs:
        item_identity = (item.get("run_id"), item.get("experiment_name"))
        same_run_dir = item.get("run_dir") == entry["run_dir"]
        if same_run_dir and item_identity != identity:
            raise RunRecordError("Existing registry run_dir is already anchored to a different run identity")
        if item_identity == identity and not same_run_dir:
            raise RunRecordError("Existing registry run identity is already anchored to a different run_dir")
        anchored_sha = str(item.get("run_file_index_sha256") or "")
        if same_run_dir and anchored_sha and anchored_sha != index_sha256:
            raise RunRecordError(
                "Existing registry entry has a different anchored run_file_index SHA-256"
            )
    next_runs = [
        item for item in managed_runs
        if (item.get("run_id"), item.get("experiment_name")) != identity
    ]
    next_runs.append(entry)
    registry["managed_runs"] = next_runs
    run_catalog = dict(registry.get("run_catalog") or {})
    catalog_items = list(run_catalog.get(resolved_status) or [])
    catalog_ref = f"{entry['experiment_name']}/{entry['run_id']}"
    if catalog_ref not in catalog_items:
        catalog_items.append(catalog_ref)
    run_catalog[resolved_status] = catalog_items
    registry["run_catalog"] = run_catalog
    registry["updated"] = _utc_now().split("T", 1)[0]
    _write_yaml(registry_path, registry)
    return registry
