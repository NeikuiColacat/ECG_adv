"""Active evidence registry loading and CPU-only auditing."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ecg_adv_gen.config import (
    build_postprocess_commands,
    build_runner_commands,
    load_experiment_config,
    validate_experiment_config,
)
from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES


class EvidenceAuditError(ValueError):
    """Raised when the active evidence registry is structurally invalid."""


_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_FORBIDDEN_TRACKED_PREFIXES = ("/home/", "/root/")
_BLOCKED_TRACKED_EXTS = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".h5", ".pkl", ".mat"}
_BLOCKED_TRACKED_DIRS = (
    "datasets/",
    "runs/",
    "outputs/",
    "checkpoints/",
    "mlruns/",
    "wandb/",
    ".dvc/cache/",
)


def _guarded_staged_paths(staged_paths: list[str], do_not_commit_entries: list[str]) -> list[str]:
    guarded: list[str] = []
    seen: set[str] = set()

    for path in staged_paths:
        matched = False
        for raw in do_not_commit_entries:
            if raw.endswith("/"):
                matched = path.startswith(raw)
            elif any(ch in raw for ch in "*?[]"):
                matched = fnmatch.fnmatch(path, raw)
            else:
                matched = path == raw
            if matched:
                break
        if matched and path not in seen:
            guarded.append(path)
            seen.add(path)
    return guarded


def _dirty_status_path(line: str) -> str:
    """Return the handoff-relevant path from a git porcelain v1 status line."""
    raw = line[3:] if len(line) > 3 else line
    if " -> " in raw:
        return raw.rsplit(" -> ", 1)[1]
    return raw


def _dirty_status_layer(path: str) -> str:
    if path == "AGENTS.md" or path.endswith("/AGENTS.md"):
        return "agent_instructions"
    if path.startswith(".codex/skills/"):
        return "codex_skills"
    if path.startswith("model/"):
        return "external_models"
    if path.startswith("configs/"):
        return "configs"
    if path.startswith("docs/") or path == "README.md":
        return "docs"
    if path.startswith("ecg_adv_gen/"):
        return "package"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith("util/tests/") or path.startswith("tests/"):
        return "tests"
    if path.startswith("methods/"):
        return "methods"
    return "other"


def _empty_dirty_layer_summary() -> dict[str, Any]:
    return {
        "total_entries": 0,
        "staged_entries": 0,
        "unstaged_entries": 0,
        "untracked_entries": 0,
        "guarded_dirty_count": 0,
        "paths": [],
        "staged_paths": [],
        "unstaged_paths": [],
        "untracked_paths": [],
        "guarded_dirty_paths": [],
        "status_entries": [],
        "sample_paths": [],
    }


def _dirty_layer_handoff_action(layer: str) -> str:
    return {
        "agent_instructions": "review_first_100_lines_before_writes",
        "codex_skills": "sync_runtime_skills_intentionally",
        "configs": "verify_yaml_index_and_path_boundaries",
        "docs": "separate_pipeline_docs_from_archived_reports",
        "external_models": "keep_local_only_do_not_stage",
        "methods": "verify_method_compatibility_before_changes",
        "package": "run_focused_cpu_tests_before_handoff",
        "scripts": "keep_legacy_wrappers_thin_and_index_managed",
        "tests": "run_related_cpu_tests",
    }.get(layer, "classify_before_commit")


def _build_dirty_handoff_gate(
    *,
    by_layer: dict[str, dict[str, Any]],
    guarded_dirty_paths: list[str],
    total_entries: int,
) -> dict[str, Any]:
    return {
        "requires_attention": total_entries > 0,
        "dirty_layer_count": len(by_layer),
        "guarded_paths_require_local_only": guarded_dirty_paths,
        "actions": [
            {
                "layer": layer,
                "handoff_action": summary["handoff_action"],
                "dirty_entries": summary["total_entries"],
                "guarded_dirty_count": summary["guarded_dirty_count"],
            }
            for layer, summary in sorted(by_layer.items())
        ],
    }


def _summarize_dirty_workspace(
    status_lines: list[str],
    *,
    do_not_commit: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize git porcelain status into agent-handoff layers."""
    guarded = do_not_commit or set()
    by_layer: dict[str, dict[str, Any]] = {}
    total = staged = unstaged = untracked = 0
    guarded_dirty_paths: list[str] = []

    for line in status_lines:
        if not line:
            continue
        total += 1
        x = line[0] if len(line) > 0 else " "
        y = line[1] if len(line) > 1 else " "
        path = _dirty_status_path(line)
        raw_status = line[:2] if len(line) >= 2 else f"{x}{y}"
        git_status = raw_status.strip() or "modified"
        layer = _dirty_status_layer(path)
        layer_summary = by_layer.setdefault(layer, _empty_dirty_layer_summary())
        layer_summary["handoff_action"] = _dirty_layer_handoff_action(layer)
        layer_summary["total_entries"] += 1
        layer_summary["paths"].append(path)
        layer_summary["status_entries"].append(
            {
                "path": path,
                "git_status": git_status,
                "index_status": x,
                "worktree_status": y,
            }
        )
        if len(layer_summary["sample_paths"]) < 5:
            layer_summary["sample_paths"].append(path)

        if x == "?" and y == "?":
            untracked += 1
            layer_summary["untracked_entries"] += 1
            layer_summary["untracked_paths"].append(path)
        else:
            if x not in (" ", "?"):
                staged += 1
                layer_summary["staged_entries"] += 1
                layer_summary["staged_paths"].append(path)
            if y != " ":
                unstaged += 1
                layer_summary["unstaged_entries"] += 1
                layer_summary["unstaged_paths"].append(path)

        if path in guarded:
            guarded_dirty_paths.append(path)
            layer_summary["guarded_dirty_count"] += 1
            layer_summary["guarded_dirty_paths"].append(path)

    return {
        "total_entries": total,
        "staged_entries": staged,
        "unstaged_entries": unstaged,
        "untracked_entries": untracked,
        "guarded_dirty_count": len(guarded_dirty_paths),
        "guarded_dirty_paths": guarded_dirty_paths,
        "by_layer": {key: by_layer[key] for key in sorted(by_layer)},
        "handoff_gate": _build_dirty_handoff_gate(
            by_layer=by_layer,
            guarded_dirty_paths=guarded_dirty_paths,
            total_entries=total,
        ),
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise EvidenceAuditError(f"YAML root must be a mapping: {path}")
    return data


def _lookup(context: dict[str, Any], key: str) -> Any:
    cur: Any = context
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise EvidenceAuditError(f"Unknown registry interpolation key: {key}")
        cur = cur[part]
    return cur


def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, context) for v in value]
    if not isinstance(value, str):
        return value

    full = _VAR_RE.fullmatch(value)
    if full:
        return _lookup(context, full.group(1))

    def repl(match: re.Match[str]) -> str:
        return str(_lookup(context, match.group(1)))

    return _VAR_RE.sub(repl, value)


def _resolve_context(context: dict[str, Any]) -> dict[str, Any]:
    resolved = context
    for _ in range(8):
        current = _resolve_value(resolved, resolved)
        if current == resolved:
            return current
        resolved = current
    raise EvidenceAuditError("Local registry interpolation did not converge")


def _walk_strings(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)
    elif isinstance(obj, str):
        yield obj


def load_evidence_registry(
    registry_path: Path,
    local_config_path: Path | None = None,
    *,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the tracked registry and optionally resolve local placeholders."""
    registry_path = Path(registry_path).expanduser().resolve()
    registry = _read_yaml(registry_path)
    if local_config_path is None:
        return registry

    context = _resolve_context(_read_yaml(Path(local_config_path).expanduser().resolve()))
    if runtime_context:
        runtime = dict(context.get("runtime") or {})
        runtime.update(runtime_context)
        context["runtime"] = runtime
    resolved = _resolve_value(registry, context)
    if not isinstance(resolved, dict):
        raise EvidenceAuditError("Resolved registry root must be a mapping")
    resolved["_registry_path"] = str(registry_path)
    resolved["_local_config_path"] = str(Path(local_config_path).expanduser().resolve())
    return resolved


def _issue(level: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {"level": level, "code": code, "message": message}
    item.update(extra)
    return item


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_or_none(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path | None, issues: list[dict[str, Any]], *, code: str, label: str) -> bool:
    if path is None:
        issues.append(_issue("warning", code, f"{label} is not declared"))
        return False
    if not path.exists():
        issues.append(_issue("error", code, f"{label} does not exist: {path}", path=str(path)))
        return False
    if not path.is_file():
        issues.append(_issue("error", code, f"{label} is not a file: {path}", path=str(path)))
        return False
    return True


def _audit_raw_registry_boundaries(raw: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    hits = [
        value for value in _walk_strings(raw)
        if value.startswith(_FORBIDDEN_TRACKED_PREFIXES)
    ]
    for hit in hits:
        issues.append(
            _issue(
                "error",
                "tracked_registry_has_host_path",
                f"Tracked registry contains a host absolute path: {hit}",
            )
        )


def _audit_config(
    repo_root: Path,
    local_config_path: Path,
    claim: dict[str, Any],
    method_key: str,
    method: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    config_path = repo_root / method["config"]
    if not config_path.exists():
        issues.append(_issue("error", "config_missing", f"{method_key} config missing: {config_path}"))
        return None
    try:
        config = load_experiment_config(
            config_path,
            local_config_path,
            runtime_context={"run_id": str(method.get("run_id", "registry_audit"))},
        )
        validate_experiment_config(config, repo_root=repo_root)
        runner_commands = build_runner_commands(config)
        postprocess_commands = build_postprocess_commands(config)
    except Exception as exc:  # noqa: BLE001 - audit reports exact failure.
        issues.append(_issue("error", "config_audit_failed", f"{method_key} config audit failed: {exc}"))
        return None

    protocol = claim["protocol"]
    pp = config["paper_protocol"]
    checks = {
        "mapping_version": pp["mapping_version"] == protocol["mapping_version"],
        "mapping_hash": pp["mapping_hash"] == protocol["mapping_hash"],
        "class_order": list(pp["class_order"]) == list(protocol["class_order"]),
        "target_centers": list(pp["centers"]["target_4"]) == list(protocol["target_centers"]),
        "k": int(pp["kshot"]["k"]) == int(protocol["kshot"]["k"]),
        "seed": int(pp["kshot"]["seed"]) == int(protocol["kshot"]["seed"]),
        "ref_excluded": bool(pp["kshot"]["exclude_refs_from_eval"]) is bool(protocol["kshot"]["ref_excluded"]),
        "backbone": str(config["model"]["backbone"]) == str(protocol["model_backbone"]),
    }
    for key, ok in checks.items():
        if not ok:
            issues.append(
                _issue(
                    "error",
                    "config_protocol_mismatch",
                    f"{method_key} config disagrees with registry on {key}",
                    method=method_key,
                    key=key,
                )
            )
    return {
        "config": config,
        "runner_command_count": len(runner_commands),
        "postprocess_command_count": len(postprocess_commands),
    }


def _audit_k500_refs(claim: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = claim["protocol"]
    kshot = protocol["kshot"]
    out: dict[str, Any] = {}
    for center in protocol["target_centers"]:
        ref_path = _path_or_none(kshot["ref_meta_files"].get(center))
        if not _require_file(ref_path, issues, code="k500_ref_missing", label=f"K500 ref meta for {center}"):
            continue
        assert ref_path is not None
        try:
            meta = json.loads(ref_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(_issue("error", "k500_ref_bad_json", f"Invalid K500 ref JSON: {ref_path}: {exc}"))
            continue
        ref_ids = meta.get("ref_record_ids") or []
        checks = {
            "center": meta.get("center") == center,
            "K": int(meta.get("K", -1)) == int(kshot["k"]),
            "seed": int(meta.get("selection_seed", -1)) == int(kshot["seed"]),
            "mapping_version": meta.get("mapping_version") == protocol["mapping_version"],
            "mapping_hash": meta.get("mapping_hash") == protocol["mapping_hash"],
            "ref_record_ids": len(ref_ids) == int(kshot["k"]),
        }
        for key, ok in checks.items():
            if not ok:
                issues.append(
                    _issue(
                        "error",
                        "k500_ref_protocol_mismatch",
                        f"K500 ref meta for {center} disagrees on {key}",
                        center=center,
                        key=key,
                        path=str(ref_path),
                    )
                )
        out[center] = {
            "path": str(ref_path),
            "n_ref_record_ids": len(ref_ids),
            "label_counts": meta.get("label_counts", {}),
        }
    return out


def _audit_metrics(
    claim: dict[str, Any],
    method_key: str,
    method: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = claim["protocol"]
    metrics_path = _path_or_none(method.get("metrics_long"))
    out: dict[str, Any] = {"metrics_long": str(metrics_path) if metrics_path else ""}
    if not _require_file(metrics_path, issues, code="metrics_long_missing", label=f"{method_key} metrics_long"):
        return out
    assert metrics_path is not None
    rows = _read_csv(metrics_path)
    if not rows:
        issues.append(_issue("error", "metrics_long_empty", f"{method_key} metrics_long has no rows"))
        return out
    if list(rows[0].keys()) != METRICS_FIELDNAMES:
        issues.append(_issue("error", "metrics_long_bad_columns", f"{method_key} metrics_long columns changed"))
    mapping_pairs = {
        (r["mapping_version"], r["mapping_hash"], r["class_order"])
        for r in rows if r.get("mapping_version") or r.get("mapping_hash")
    }
    expected_pair = (
        protocol["mapping_version"],
        protocol["mapping_hash"],
        "|".join(protocol["class_order"]),
    )
    if mapping_pairs != {expected_pair}:
        issues.append(
            _issue(
                "error",
                "metrics_mapping_mismatch",
                f"{method_key} metrics mapping pairs are not the registry pair",
                observed=sorted("|".join(p) for p in mapping_pairs),
                expected="|".join(expected_pair),
            )
        )

    centers = set(protocol["target_centers"])
    views = set(protocol["evaluation_views"])
    for view in views:
        for metric in ("macro_auroc", "macro_auprc"):
            allowed_metric_names = {metric}
            if view == "pn2021_drop_all_zero_refexcluded":
                allowed_metric_names.add(f"drop_all_zero_{metric}")
            got_centers = {
                r["center"] for r in rows
                if r.get("canonical_view") == view
                and r.get("scope") == "center"
                and r.get("metric") in allowed_metric_names
            }
            if got_centers != centers:
                issues.append(
                    _issue(
                        "error",
                        "metrics_center_coverage_mismatch",
                        f"{method_key} metrics missing centers for {view}/{metric}",
                        observed=sorted(got_centers),
                        expected=sorted(centers),
                    )
                )
    out["n_rows"] = len(rows)
    return out


def _audit_paper_tables(
    claim: dict[str, Any],
    method_key: str,
    method: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = claim["protocol"]
    out: dict[str, Any] = {}
    for view in protocol["evaluation_views"]:
        table_path = _path_or_none(method.get("paper_tables", {}).get(view))
        if not _require_file(table_path, issues, code="paper_table_missing", label=f"{method_key} paper table {view}"):
            continue
        assert table_path is not None
        rows = _read_csv(table_path)
        center_mean = [r for r in rows if r.get("scope") == "center_mean"]
        if len(center_mean) != 1:
            issues.append(
                _issue(
                    "error",
                    "paper_table_center_mean_missing",
                    f"{method_key} paper table {view} should have exactly one center_mean row",
                )
            )
            continue
        row = center_mean[0]
        expected_center_group = "|".join(protocol["target_centers"])
        if row.get("center_group") != expected_center_group:
            issues.append(
                _issue(
                    "error",
                    "paper_table_center_group_mismatch",
                    f"{method_key} paper table {view} has wrong center group",
                    observed=row.get("center_group"),
                    expected=expected_center_group,
                )
            )
        for key, expected in {
            "mapping_version": protocol["mapping_version"],
            "mapping_hash": protocol["mapping_hash"],
            "class_order": "|".join(protocol["class_order"]),
        }.items():
            if row.get(key) != expected:
                issues.append(
                    _issue(
                        "error",
                        "paper_table_mapping_mismatch",
                        f"{method_key} paper table {view} disagrees on {key}",
                    )
                )
        summary = claim.get("summary_metrics", {}).get(view, {}).get(method_key, {})
        for metric in ("macro_auroc", "macro_auprc"):
            expected_value = _float(summary.get(metric))
            observed_value = _float(row.get(metric))
            if expected_value is not None and observed_value is not None:
                if abs(expected_value - observed_value) > 1e-9:
                    issues.append(
                        _issue(
                            "error",
                            "summary_metric_mismatch",
                            f"{method_key} summary {view}/{metric} differs from paper table",
                            observed=observed_value,
                            expected=expected_value,
                        )
                    )
        missing_sources = []
        for source in row.get("source_files", "").split("|"):
            if source and not Path(source).exists():
                missing_sources.append(source)
        if missing_sources:
            issues.append(
                _issue(
                    "error",
                    "paper_table_source_missing",
                    f"{method_key} paper table {view} references missing source files",
                    missing=missing_sources,
                )
            )
        out[view] = {
            "path": str(table_path),
            "macro_auroc": row.get("macro_auroc", ""),
            "macro_auprc": row.get("macro_auprc", ""),
        }
    return out


def _trusted_mainline_requires_manifest(claim: dict[str, Any], method: dict[str, Any]) -> bool:
    return (
        str(claim.get("status", "")) == "trusted"
        and str(claim.get("paper_use", "")) == "mainline"
        and str(method.get("status", "")) == "trusted"
    )


def _has_registered_run_for_method(registry: dict[str, Any], method_key: str, method: dict[str, Any]) -> bool:
    explicit = method.get("registered_run")
    if isinstance(explicit, dict) and explicit:
        return True
    expected_run_ids = {
        str(value)
        for value in (
            method.get("registered_run_id"),
            method.get("run_id"),
            method_key,
        )
        if value
    }
    config_name = Path(str(method.get("config") or "")).stem
    expected_experiments = {
        str(value)
        for value in (
            method.get("registered_experiment_name"),
            config_name,
            method_key,
        )
        if value
    }
    for item in registry.get("managed_runs") or []:
        if str(item.get("status") or "") not in {"trusted", "provisional"}:
            continue
        if str(item.get("run_id") or "") in expected_run_ids:
            return True
        if str(item.get("experiment_name") or "") in expected_experiments:
            return True
    return False


def _audit_legacy_backfill_registration(
    registry: dict[str, Any],
    claim: dict[str, Any],
    method_key: str,
    method: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if not _trusted_mainline_requires_manifest(claim, method):
        return
    if method.get("traceability") != "legacy_backfilled_manifest":
        return
    if _has_registered_run_for_method(registry, method_key, method):
        return
    issues.append(
        _issue(
            "error",
            "trusted_legacy_backfill_missing_registered_run",
            f"{method_key} is trusted mainline but only has a legacy backfilled manifest; register a run record first",
            method=method_key,
            run_id=method.get("run_id", ""),
        )
    )


def _audit_legacy_backfilled_manifest(
    method_key: str,
    method: dict[str, Any],
    claim: dict[str, Any],
    manifest: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    protocol = claim["protocol"]
    checks = {
        "claim_id": manifest.get("claim_id") == claim.get("claim_id"),
        "method_key": manifest.get("method_key") == method_key,
        "run_id": manifest.get("run_id") == method.get("run_id"),
        "mapping_version": manifest.get("protocol", {}).get("mapping_version") == protocol["mapping_version"],
        "mapping_hash": manifest.get("protocol", {}).get("mapping_hash") == protocol["mapping_hash"],
        "class_order": list(manifest.get("protocol", {}).get("class_order") or []) == list(protocol["class_order"]),
        "target_centers": list(manifest.get("protocol", {}).get("target_centers") or []) == list(protocol["target_centers"]),
    }
    for key, ok in checks.items():
        if not ok:
            issues.append(
                _issue(
                    "error",
                    "legacy_manifest_mismatch",
                    f"{method_key} legacy backfilled manifest disagrees on {key}",
                    key=key,
                )
            )

    centers = manifest.get("centers") or []
    if len(centers) != len(protocol["target_centers"]):
        issues.append(
            _issue(
                "error",
                "legacy_manifest_center_count_mismatch",
                f"{method_key} legacy backfilled manifest has the wrong number of centers",
                observed=len(centers),
                expected=len(protocol["target_centers"]),
            )
        )
    observed_centers = [str(center_run.get("center", "")) for center_run in centers]
    if observed_centers != list(protocol["target_centers"]):
        issues.append(
            _issue(
                "error",
                "legacy_manifest_center_order_mismatch",
                f"{method_key} legacy backfilled manifest center order differs from registry",
                observed=observed_centers,
                expected=list(protocol["target_centers"]),
            )
        )
    for center_run in centers:
        final_eval = center_run.get("final_eval") or {}
        for key in ("mapping_version", "mapping_hash"):
            if final_eval.get(key) != protocol[key]:
                issues.append(
                    _issue(
                        "error",
                        "legacy_manifest_eval_mapping_mismatch",
                        f"{method_key} legacy center eval disagrees on {key}",
                        center=center_run.get("center"),
                        observed=final_eval.get(key),
                        expected=protocol[key],
                    )
                )
        artifacts = center_run.get("artifacts") or {}
        for artifact_name in (
            "launch_config.json",
            "train_result.json",
            "early_stop_info.json",
            "eval_result_v7_exclrefs_crop1000.json",
            "best_model.pt",
        ):
            record = artifacts.get(artifact_name) or {}
            record_path = _path_or_none(record.get("path"))
            if not record.get("exists") or record_path is None or not record_path.exists():
                issues.append(
                    _issue(
                        "error",
                        "legacy_manifest_artifact_missing",
                        f"{method_key} legacy manifest missing {artifact_name}",
                        center=center_run.get("center"),
                    )
                )
                continue
            observed_sha = _sha256(record_path)
            if not record.get("sha256"):
                issues.append(
                    _issue(
                        "error",
                        "legacy_manifest_artifact_unhashed",
                        f"{method_key} legacy manifest has no sha256 for {artifact_name}",
                        center=center_run.get("center"),
                    )
                )
            elif record.get("sha256") != observed_sha:
                issues.append(
                    _issue(
                        "error",
                        "legacy_manifest_artifact_hash_mismatch",
                        f"{method_key} legacy manifest sha256 mismatch for {artifact_name}",
                        center=center_run.get("center"),
                        observed=observed_sha,
                        expected=record.get("sha256"),
                    )
                )


def _audit_manifest(
    method_key: str,
    method: dict[str, Any],
    claim: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = _path_or_none(method.get("manifest"))
    if manifest_path is None:
        level = "error" if _trusted_mainline_requires_manifest(claim, method) else "warning"
        issues.append(
            _issue(
                level,
                "manifest_not_declared",
                f"{method_key} has no managed run_manifest.json declared",
                method=method_key,
            )
        )
        return {"declared": False, "path": ""}
    if not manifest_path.exists():
        issues.append(_issue("error", "manifest_missing", f"{method_key} manifest missing: {manifest_path}"))
        return {"declared": True, "path": str(manifest_path), "exists": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(_issue("error", "manifest_bad_json", f"{method_key} manifest invalid: {exc}"))
        return {"declared": True, "path": str(manifest_path), "exists": True}
    status = manifest.get("status")
    if status not in {"succeeded", "dry_run"}:
        issues.append(
            _issue(
                "warning",
                "manifest_status_not_succeeded",
                f"{method_key} manifest status is {status!r}",
                method=method_key,
            )
        )
    manifest_kind = manifest.get("manifest_kind", "managed_launcher_manifest")
    if method.get("traceability") == "legacy_backfilled_manifest":
        if manifest_kind != "legacy_backfilled_manifest":
            issues.append(
                _issue(
                    "error",
                    "legacy_manifest_kind_mismatch",
                    f"{method_key} is marked legacy_backfilled_manifest but manifest_kind is {manifest_kind!r}",
                )
            )
        _audit_legacy_backfilled_manifest(method_key, method, claim, manifest, issues)
    return {
        "declared": True,
        "path": str(manifest_path),
        "exists": True,
        "status": status,
        "manifest_kind": manifest_kind,
    }


def _audit_comparison_bundle(claim: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = claim.get("comparison_bundle") or {}
    if not bundle:
        issues.append(_issue("warning", "comparison_bundle_missing", f"{claim['claim_id']} has no comparison bundle"))
        return {}
    output_dir = _path_or_none(bundle.get("output_dir"))
    if output_dir is None:
        issues.append(_issue("error", "comparison_bundle_output_missing", "Comparison bundle output_dir is empty"))
        return {}
    status = bundle.get("status", "")
    expected_artifacts = list(bundle.get("managed_artifacts") or [])
    summary = {"status": status, "output_dir": str(output_dir), "expected_artifacts": expected_artifacts}
    if status not in {"built", "trusted", "active"}:
        issues.append(
            _issue(
                "warning",
                "comparison_bundle_not_built",
                f"Comparison bundle is not marked built: status={status!r}",
                output_dir=str(output_dir),
            )
        )
        return summary
    if not output_dir.exists():
        issues.append(
            _issue(
                "error",
                "comparison_bundle_dir_missing",
                f"Built comparison bundle directory does not exist: {output_dir}",
            )
        )
        return summary
    missing = [name for name in expected_artifacts if not (output_dir / name).exists()]
    if missing:
        issues.append(
            _issue(
                "error",
                "comparison_bundle_artifact_missing",
                "Built comparison bundle is missing expected artifacts",
                output_dir=str(output_dir),
                missing=missing,
            )
        )
    manifest_path = output_dir / "comparison_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(_issue("error", "comparison_manifest_bad_json", f"Invalid comparison manifest: {exc}"))
        else:
            for key in ("comparison_id", "baseline_run_id", "candidate_run_id"):
                if manifest.get(key) != bundle.get(key):
                    issues.append(
                        _issue(
                            "error",
                            "comparison_manifest_mismatch",
                            f"Comparison manifest disagrees on {key}",
                            observed=manifest.get(key),
                            expected=bundle.get(key),
                        )
                    )
            summary["manifest_created_at_utc"] = manifest.get("created_at_utc", "")
    return summary


def _audit_git(repo_root: Path, registry: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    def run_git(args: list[str]) -> list[str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            issues.append(_issue("warning", "git_command_failed", f"git {' '.join(args)} failed: {proc.stderr.strip()}"))
            return []
        return [line for line in proc.stdout.splitlines() if line.strip()]

    tracked = run_git(["ls-files"])
    blocked_tracked = [
        path for path in tracked
        if Path(path).suffix.lower() in _BLOCKED_TRACKED_EXTS
        or path.startswith(_BLOCKED_TRACKED_DIRS)
    ]
    for path in blocked_tracked:
        issues.append(_issue("error", "blocked_artifact_tracked", f"Blocked artifact is tracked by git: {path}"))

    do_not_commit_entries = list(registry.get("artifact_policy", {}).get("do_not_commit", []))

    staged = run_git(["diff", "--cached", "--name-only"])
    blocked_staged = [
        path for path in staged
        if Path(path).suffix.lower() in _BLOCKED_TRACKED_EXTS
        or path.startswith(_BLOCKED_TRACKED_DIRS)
    ]
    for path in blocked_staged:
        issues.append(_issue("error", "blocked_artifact_staged", f"Blocked artifact is staged: {path}"))
    guarded_staged = _guarded_staged_paths(staged, do_not_commit_entries)
    for path in guarded_staged:
        issues.append(_issue("error", "guarded_path_staged", f"Guarded local-only path is staged: {path}"))

    dirty = run_git(["status", "--short"])
    do_not_commit = set()
    for raw in do_not_commit_entries:
        if not any(ch in raw for ch in "*?[]") and not raw.endswith("/"):
            do_not_commit.add(raw)
    dirty_summary = _summarize_dirty_workspace(dirty, do_not_commit=do_not_commit)
    dirty_guarded = dirty_summary["guarded_dirty_paths"]
    for path in dirty_guarded:
        issues.append(
            _issue(
                "warning",
                "guarded_path_dirty",
                f"Guarded external-model path is dirty and must not be staged: {path}",
            )
        )
    blocking_artifact_risks = [
        {
            "risk": "blocked_tracked_artifacts",
            "severity": "error",
            "count": len(blocked_tracked),
            "paths": blocked_tracked,
            "handoff_action": "remove_from_git_tracking_or_document_exception_before_handoff",
        },
        {
            "risk": "blocked_staged_artifacts",
            "severity": "error",
            "count": len(blocked_staged),
            "paths": blocked_staged,
            "handoff_action": "unstage_large_or_generated_artifacts",
        },
        {
            "risk": "guarded_staged_paths",
            "severity": "error",
            "count": len(guarded_staged),
            "paths": guarded_staged,
            "handoff_action": "unstage_local_only_guarded_paths",
        },
        {
            "risk": "dirty_guarded_local_only_paths",
            "severity": "warning",
            "count": len(dirty_guarded),
            "paths": dirty_guarded,
            "handoff_action": "keep_local_only_do_not_stage",
        },
    ]
    return {
        "tracked_count": len(tracked),
        "blocked_tracked_count": len(blocked_tracked),
        "staged_count": len(staged),
        "blocked_staged_count": len(blocked_staged),
        "guarded_staged_count": len(guarded_staged),
        "guarded_staged_paths": guarded_staged,
        "dirty_guarded_paths": dirty_guarded,
        "blocking_artifact_risks": blocking_artifact_risks,
        "dirty_summary": dirty_summary,
    }


def audit_active_evidence_registry(
    *,
    repo_root: Path,
    registry_path: Path,
    local_config_path: Path,
    require_existing_artifacts: bool = True,
    check_git: bool = True,
) -> dict[str, Any]:
    """Audit the active evidence registry without loading torch or using GPU."""
    repo_root = Path(repo_root).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    local_config_path = Path(local_config_path).expanduser().resolve()
    raw_registry = load_evidence_registry(registry_path)
    registry = load_evidence_registry(registry_path, local_config_path)
    issues: list[dict[str, Any]] = []
    _audit_raw_registry_boundaries(raw_registry, issues)

    claim_summaries: list[dict[str, Any]] = []
    for claim in registry.get("active_claims", []):
        claim_id = claim["claim_id"]
        k500_summary = _audit_k500_refs(claim, issues) if require_existing_artifacts else {}
        method_summaries: dict[str, Any] = {}
        config_summaries: dict[str, Any] = {}
        for method_key, method in claim["methods"].items():
            _audit_legacy_backfill_registration(registry, claim, method_key, method, issues)
            config_summary = _audit_config(repo_root, local_config_path, claim, method_key, method, issues)
            if config_summary is not None:
                config_summaries[method_key] = {
                    "runner_command_count": config_summary["runner_command_count"],
                    "postprocess_command_count": config_summary["postprocess_command_count"],
                }
            method_summary = {"status": method.get("status"), "role": method.get("role")}
            method_summary["manifest"] = (
                _audit_manifest(method_key, method, claim, issues)
                if require_existing_artifacts else {"skipped": True}
            )
            if require_existing_artifacts:
                method_summary["metrics"] = _audit_metrics(claim, method_key, method, issues)
                method_summary["paper_tables"] = _audit_paper_tables(claim, method_key, method, issues)
            method_summaries[method_key] = method_summary

        if len(config_summaries) >= 2:
            backbones = set()
            seeds = set()
            mappings = set()
            centers = set()
            for method_key in claim["methods"]:
                config = _audit_config(repo_root, local_config_path, claim, method_key, claim["methods"][method_key], [])
                if not config:
                    continue
                cfg = config["config"]
                backbones.add(cfg["model"]["backbone"])
                seeds.add(int(cfg["paper_protocol"]["kshot"]["seed"]))
                mappings.add((cfg["paper_protocol"]["mapping_version"], cfg["paper_protocol"]["mapping_hash"]))
                centers.add(tuple(cfg["paper_protocol"]["centers"]["target_4"]))
            if len(backbones) > 1 or len(seeds) > 1 or len(mappings) > 1 or len(centers) > 1:
                issues.append(
                    _issue(
                        "error",
                        "method_comparability_mismatch",
                        f"Direct/VAE configs are not comparable for {claim_id}",
                        backbones=sorted(backbones),
                        seeds=sorted(seeds),
                        mappings=sorted("|".join(x) for x in mappings),
                    )
                )

        claim_summaries.append(
            {
                "claim_id": claim_id,
                "status": claim.get("status"),
                "paper_use": claim.get("paper_use"),
                "evaluation_views": list(claim.get("protocol", {}).get("evaluation_views") or []),
                "required_reporting_views": list(
                    claim.get("protocol", {}).get("required_reporting_views")
                    or claim.get("protocol", {}).get("evaluation_views")
                    or []
                ),
                "k500_refs": k500_summary,
                "methods": method_summaries,
                "comparison_bundle": (
                    _audit_comparison_bundle(claim, issues)
                    if require_existing_artifacts else {"skipped": True}
                ),
            }
        )

    git_summary = _audit_git(repo_root, registry, issues) if check_git else {"skipped": True}
    error_count = sum(1 for i in issues if i["level"] == "error")
    warning_count = sum(1 for i in issues if i["level"] == "warning")
    return {
        "passed": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "registry_path": str(registry_path),
        "local_config_path": str(local_config_path),
        "evidence_surface_policy": raw_registry.get("evidence_surface_policy") or {},
        "claim_count": len(registry.get("active_claims", [])),
        "claims": claim_summaries,
        "git": git_summary,
    }
