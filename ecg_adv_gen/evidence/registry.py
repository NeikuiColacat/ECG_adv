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
from ecg_adv_gen.evidence.run_record import REGISTRATION_STATUSES, verify_run_file_index
from ecg_adv_gen.evaluation.pn2021c_metadata import (
    PN2021CMetadataError,
    evaluation_k500_identities,
)
from ecg_adv_gen.evaluation.pn2021_corruptions import ref_ids_sha256


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


def _read_csv(
    path: Path,
    issues: list[dict[str, Any]],
    *,
    code: str,
    label: str,
) -> list[dict[str, str | None]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        issues.append(_issue("error", code, f"{label} is invalid CSV: {exc}"))
        return None


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


def _add_error(issues: list[dict[str, Any]], code: str, message: str, **extra: Any) -> None:
    issues.append(_issue("error", code, message, **extra))


def _resolved_path(raw: Any, base: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _error_count(issues: list[dict[str, Any]]) -> int:
    return sum(issue["level"] == "error" for issue in issues)


def _require_path(path: Path | None, issues: list[dict[str, Any]], code: str, label: str, *, directory=False) -> bool:
    ok = path is not None and (path.is_dir() if directory else path.is_file())
    if not ok:
        _add_error(issues, code, f"{label} is missing: {path or '<not declared>'}", path=str(path or ""))
    return ok


def _read_json_object(path: Path, issues: list[dict[str, Any]], code: str, label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _add_error(issues, code, f"{label} is invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        _add_error(issues, code, f"{label} root must be an object")
        return None
    return value


_CLAIM_STATUSES = {"trusted", "provisional", "deprecated", "exploratory"}
_METHOD_STATUSES = {"trusted", "provisional", "supporting", "deprecated", "exploratory"}
_NON_READY_REPLAY_STATUSES = {"artifact_missing", "historical_evaluation_only"}


def _claim_shape_is_valid(claim: dict[str, Any], index: int, issues: list[dict[str, Any]]) -> bool:
    valid = True
    claim_status = str(claim.get("status") or "")
    claim_id = claim.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        _add_error(issues, "active_claim_id_missing", f"active_claims[{index}] needs claim_id")
        valid = False
    if claim_status not in _CLAIM_STATUSES:
        _add_error(issues, "active_claim_status_invalid", f"active_claims[{index}] has invalid status")
        valid = False
    methods = claim.get("methods")
    if not isinstance(methods, dict):
        _add_error(issues, "active_claim_methods_invalid", f"active_claims[{index}].methods must be a mapping")
        valid = False
    else:
        if claim_status in {"trusted", "provisional"} and not methods:
            _add_error(issues, "active_claim_methods_invalid", f"active_claims[{index}].methods must not be empty")
            valid = False
        for method_key, method in methods.items():
            if not isinstance(method_key, str) or not method_key:
                _add_error(issues, "active_claim_method_key_invalid", f"active_claims[{index}] method keys must be strings")
                valid = False
                continue
            if not isinstance(method, dict):
                _add_error(issues, "active_claim_method_invalid", f"active_claims[{index}].methods.{method_key} must be a mapping")
                valid = False
                continue
            if str(method.get("status") or "") not in _METHOD_STATUSES:
                _add_error(issues, "active_claim_method_status_invalid", f"{method_key} has invalid status")
                valid = False
            for field in ("run_id", "experiment_name", "config"):
                if not isinstance(method.get(field), str) or not method[field]:
                    _add_error(issues, "active_claim_method_field_missing", f"{method_key} needs {field}")
                    valid = False
            if "paper_tables" in method and not isinstance(method["paper_tables"], dict):
                _add_error(issues, "active_claim_method_field_invalid", f"{method_key}.paper_tables must be a mapping")
                valid = False
    protocol = claim.get("protocol")
    if not isinstance(protocol, dict):
        _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol must be a mapping")
        return False
    for field in ("mapping_version", "mapping_hash", "model_backbone"):
        if not isinstance(protocol.get(field), str) or not protocol[field]:
            _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol needs {field}")
            valid = False
    for field in ("class_order", "target_centers", "evaluation_views"):
        value = protocol.get(field)
        if (not isinstance(value, list) or not all(isinstance(item, str) and item for item in value)
                or (claim_status in {"trusted", "provisional"} and not value)):
            _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol.{field} must be a non-empty list")
            valid = False
    reporting_views = protocol.get("required_reporting_views")
    if reporting_views is not None and (
        not isinstance(reporting_views, list)
        or not all(isinstance(item, str) and item for item in reporting_views)
    ):
        _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol.required_reporting_views is invalid")
        valid = False
    kshot = protocol.get("kshot")
    if not isinstance(kshot, dict) or not isinstance(kshot.get("ref_meta_files"), dict):
        _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol.kshot is invalid")
        valid = False
    elif (not isinstance(kshot.get("k"), int) or isinstance(kshot.get("k"), bool) or kshot["k"] <= 0
          or not isinstance(kshot.get("seed"), int) or isinstance(kshot.get("seed"), bool)
          or not isinstance(kshot.get("ref_excluded"), bool)):
        _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}].protocol.kshot contract is incomplete")
        valid = False
    elif claim_status in {"trusted", "provisional"}:
        centers = protocol.get("target_centers") if isinstance(protocol.get("target_centers"), list) else []
        ref_files = kshot["ref_meta_files"]
        if any(not isinstance(center, str) or not isinstance(ref_files.get(center), str) or not ref_files[center]
               for center in centers):
            _add_error(issues, "active_claim_protocol_invalid", f"active_claims[{index}] needs one K500 ref path per center")
            valid = False
    if "comparison_bundle" in claim and not isinstance(claim["comparison_bundle"], dict):
        _add_error(issues, "active_claim_comparison_bundle_invalid", f"active_claims[{index}].comparison_bundle must be a mapping")
        valid = False
    elif isinstance(claim.get("comparison_bundle"), dict):
        bundle = claim["comparison_bundle"]
        if "status" in bundle and not isinstance(bundle["status"], str):
            _add_error(issues, "active_claim_comparison_bundle_invalid", f"active_claims[{index}].comparison_bundle.status is invalid")
            valid = False
        managed_artifacts = bundle.get("managed_artifacts")
        if managed_artifacts is not None and (
            not isinstance(managed_artifacts, list)
            or not all(isinstance(item, str) and item for item in managed_artifacts)
        ):
            _add_error(issues, "active_claim_comparison_bundle_invalid", f"active_claims[{index}].comparison_bundle.managed_artifacts is invalid")
            valid = False
    summaries = claim.get("summary_metrics")
    if summaries is not None and not isinstance(summaries, dict):
        _add_error(issues, "active_claim_summary_metrics_invalid", f"active_claims[{index}].summary_metrics must be a mapping")
        valid = False
    elif isinstance(summaries, dict):
        for view, methods_summary in summaries.items():
            if (not isinstance(view, str) or not isinstance(methods_summary, dict)
                    or not all(isinstance(name, str) and isinstance(value, dict)
                               for name, value in methods_summary.items())):
                _add_error(issues, "active_claim_summary_metrics_invalid", f"active_claims[{index}].summary_metrics is invalid")
                valid = False
                break
    return valid


def _managed_claim_pairs(registry: dict[str, Any]) -> dict[tuple[str, str], set[tuple[str, str]]]:
    pairs: dict[tuple[str, str], set[tuple[str, str]]] = {}
    claims = registry.get("active_claims")
    if not isinstance(claims, list):
        return pairs
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("methods"), dict):
            continue
        protocol = claim.get("protocol") if isinstance(claim.get("protocol"), dict) else {}
        pair = (str(protocol.get("mapping_version") or ""), str(protocol.get("mapping_hash") or ""))
        for method in claim["methods"].values():
            if (isinstance(method, dict) and method.get("status") in {"trusted", "provisional"}
                    and method.get("run_id") and method.get("experiment_name")):
                key = (str(method["run_id"]), str(method["experiment_name"]))
                pairs.setdefault(key, set()).add(pair)
    return pairs


_RUN_FILES = (("run_card.json", "run_card", "managed_run_card_missing"),
              ("summary.md", "summary", "managed_run_summary_missing"),
              ("run_file_index.json", "run_file_index", "managed_run_file_index_missing"))


def _audit_managed_runs(repo_root: Path, registry: dict[str, Any], issues, *, check_files: bool,
                        cache: dict[Path, str], stats: dict[str, int]) -> dict[str, Any]:
    raw_runs = registry.get("managed_runs")
    if raw_runs is None:
        runs = []
    elif not isinstance(raw_runs, list):
        _add_error(issues, "managed_runs_invalid", "managed_runs must be a list")
        runs = []
    else:
        runs = raw_runs
    entries, seen, valid, allowed = [], set(), 0, _managed_claim_pairs(registry)
    for index, item in enumerate(runs):
        before = _error_count(issues)
        if not isinstance(item, dict):
            _add_error(issues, "managed_run_invalid", f"managed_runs[{index}] must be a mapping")
            continue
        run_id, experiment = map(lambda key: str(item.get(key) or ""), ("run_id", "experiment_name"))
        pair = tuple(str(item.get(key) or "") for key in ("mapping_version", "mapping_hash"))
        key, replay, status = (run_id, experiment), str(item.get("replay_status") or ""), str(item.get("status") or "")
        if status not in REGISTRATION_STATUSES:
            _add_error(issues, "managed_run_status_invalid", f"{run_id} has invalid status={status!r}")
        if not all(key):
            _add_error(issues, "managed_run_key_not_declared", f"managed_runs[{index}] needs run_id/experiment_name")
        if key in seen:
            _add_error(issues, "managed_run_duplicate_key", f"Duplicate managed run key: {run_id}/{experiment}")
        seen.add(key)
        active = status in {"trusted", "provisional"}
        if (active and replay != "replay_ready") or (not active and replay not in _NON_READY_REPLAY_STATUSES):
            _add_error(issues, "managed_run_replay_status_invalid", f"{run_id} status/replay_status disagree")
        claim_pairs = allowed.get(key, set())
        if active and len(claim_pairs) > 1:
            _add_error(issues, "managed_run_claim_mapping_conflict", f"{run_id} has conflicting claim mappings")
        if not all(pair) or (active and claim_pairs != {pair}):
            _add_error(issues, "managed_run_claim_mapping_mismatch", f"{run_id} mapping is not the active mapping")

        run_dir = _resolved_path(item.get("run_dir"), repo_root)
        canonical = {name: run_dir / name if run_dir else None for name, _, _ in _RUN_FILES}
        for name, field, _ in _RUN_FILES:
            declared = _resolved_path(item.get(field), repo_root) or (canonical[name] if field == "run_file_index" else None)
            if declared != canonical[name]:
                _add_error(issues, "managed_run_path_mismatch", f"{run_id} {name} must be inside run_dir")
        required = card_matches = None
        integrity = {"passed": False, "status": "skipped", "checked_count": 0, "warnings": [], "errors": []}
        declared_index_sha = str(item.get("run_file_index_sha256") or "")
        if not _SHA256_RE.fullmatch(declared_index_sha):
            _add_error(issues, "managed_run_index_sha_invalid", f"{run_id} needs run_file_index_sha256")
        if check_files:
            checks = [_require_path(run_dir, issues, "managed_run_dir_missing", f"{run_id} run_dir", directory=True)]
            checks += [_require_path(canonical[name], issues, code, f"{run_id} {name}") for name, _, code in _RUN_FILES]
            required = all(checks)
            card = _read_json_object(canonical["run_card.json"], issues, "managed_run_card_invalid", run_id) if checks[1] else None
            experiment_obj, protocol = (card or {}).get("experiment", {}), (card or {}).get("protocol", {})
            observed = ((card or {}).get("run_id"), experiment_obj.get("name") if isinstance(experiment_obj, dict) else None,
                        protocol.get("mapping_version") if isinstance(protocol, dict) else None,
                        protocol.get("mapping_hash") if isinstance(protocol, dict) else None)
            card_matches = observed == (run_id, experiment, *pair)
            if not card_matches:
                _add_error(issues, "managed_run_card_mismatch", f"{run_id} run_card disagrees with registry")
            if checks[3] and _SHA256_RE.fullmatch(declared_index_sha):
                observed_index_sha = _cached_sha(canonical["run_file_index.json"], cache, stats)
                if observed_index_sha != declared_index_sha:
                    _add_error(issues, "managed_run_index_sha_mismatch", f"{run_id} run_file_index SHA256 mismatch")
            if required:
                integrity = verify_run_file_index(run_dir)
                if integrity.get("status") != "verified":
                    level, code = (("error", "managed_run_integrity_failed") if active
                                   else ("warning", "managed_run_integrity_non_ready"))
                    issues.append(_issue(level, code, f"{run_id} run_file_index status={integrity.get('status')}"))
        ok = _error_count(issues) == before
        valid += ok
        entries.append({"run_id": run_id, "experiment_name": experiment, "run_dir": str(run_dir or ""),
                        "required_files_present": required, "card_matches_registry": card_matches,
                        "replay_ready": ok and active and replay == "replay_ready"
                        and integrity.get("status") == "verified", "integrity": integrity})
    return {"count": len(entries), "valid_count": valid, "entries": entries}


_SUPPORT_FILES = ("evidence_doc", "method_config", "metrics_long", "data_manifest", "artifact_manifest", "summary_csv",
                  "center_csv", "summary_json", "provenance_gap", "artifact_integrity")
_SUPPORT_DIRS = ("training_root", "evaluation_root")
_SUPPORT_MAPS = {"method_artifact_manifests": True, "paper_tables": False, "paper_table_manifests": False}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SUPPORT_STATUSES = {
    "trusted", "provisional", "provisional_single_seed", "supporting_contrast_no_direct_delta",
    "deprecated", "exploratory",
}


def _cached_sha(path: Path, cache: dict[Path, str], stats: dict[str, int]) -> str:
    if path in cache:
        stats["cache_hits"] += 1
    else:
        cache[path], stats["computed"] = _sha256(path), stats["computed"] + 1
    return cache[path]


def _manifest_artifact_path(artifact: dict[str, Any], manifest: Path, context: dict[str, Any], issues, label) -> Path | None:
    path = _resolved_path(artifact.get("path"), manifest.parent)
    ref = artifact.get("path_ref")
    ref_path = None
    if ref is not None:
        try:
            resolved = _resolve_value(ref, context) if isinstance(ref, str) else None
        except EvidenceAuditError as exc:
            _add_error(issues, "supporting_manifest_path_ref_unresolvable", f"{label}: {exc}")
        else:
            ref_path = _resolved_path(resolved, manifest.parent)
            if ref_path is None:
                _add_error(issues, "supporting_manifest_path_ref_unresolvable", f"{label} path_ref is invalid")
    if path and path.is_file():
        return path
    return ref_path or path


def _audit_strict_manifest(path: Path, field: str, item: dict[str, Any], context: dict[str, Any],
                           issues, cache, stats) -> dict[str, int]:
    out = {"checked": 0, "missing": 0, "mismatch": 0}
    manifest = _read_json_object(path, issues, "supporting_manifest_invalid", field)
    artifacts = manifest.get("artifacts") if manifest else None
    if field == "artifact_integrity":
        declared = str(item.get("artifact_integrity_sha256") or "")
        observed = _cached_sha(path, cache, stats)
        if _SHA256_RE.fullmatch(declared) and observed != declared:
            _add_error(issues, "supporting_integrity_root_sha_mismatch", f"{field} root SHA256 mismatch")
    if not isinstance(artifacts, list) or not artifacts:
        _add_error(issues, "supporting_manifest_artifacts_invalid", f"{field}.artifacts must be a non-empty list")
        return out
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or not _SHA256_RE.fullmatch(str(artifact.get("sha256") or "")):
            _add_error(issues, "supporting_manifest_artifact_invalid", f"{field}.artifacts[{index}] needs path/path_ref and sha256")
            continue
        artifact_path = _manifest_artifact_path(artifact, path, context, issues, f"{field}.artifacts[{index}]")
        if artifact_path is None:
            _add_error(issues, "supporting_manifest_artifact_invalid", f"{field}.artifacts[{index}] has no usable path")
        elif not artifact_path.is_file():
            out["missing"] += 1
            _add_error(issues, "supporting_manifest_artifact_missing", f"{field} artifact missing: {artifact_path}")
        else:
            out["checked"] += 1
            if _cached_sha(artifact_path, cache, stats) != artifact["sha256"]:
                out["mismatch"] += 1
                _add_error(issues, "supporting_manifest_sha256_mismatch", f"{field} artifact SHA mismatch: {artifact_path}")
    return out


def _audit_supporting_reporting_artifacts(repo_root: Path, claim: dict[str, Any], issues, *, check_files: bool,
                                          check_git: bool, context, cache, stats) -> dict[str, Any]:
    raw = claim.get("supporting_reporting_artifacts")
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        _add_error(issues, "supporting_artifacts_invalid", "supporting_reporting_artifacts must be a mapping")
        return {"count": 0, "valid_count": 0, "entries": []}
    entries = []
    for artifact_id, item in raw.items():
        before = _error_count(issues)
        if not isinstance(item, dict):
            _add_error(issues, "supporting_artifact_invalid", f"{artifact_id} must be a mapping")
            continue
        commit, replay = str(item.get("implementation_commit") or ""), str(item.get("replay_status") or "")
        support_status = str(item.get("status") or "")
        if support_status not in _SUPPORT_STATUSES:
            _add_error(issues, "supporting_status_invalid", f"{artifact_id} has invalid status")
        if not item.get("evidence_doc"):
            _add_error(issues, "supporting_evidence_doc_not_declared", f"{artifact_id} needs evidence_doc")
        if not commit:
            _add_error(issues, "supporting_implementation_commit_not_declared", f"{artifact_id} needs implementation_commit")
        elif not _GIT_COMMIT_RE.fullmatch(commit):
            _add_error(issues, "supporting_implementation_commit_invalid", f"{artifact_id} implementation_commit must be a full SHA")
        if support_status in {"trusted", "provisional", "provisional_single_seed"}:
            for field in ("provenance_gap", "artifact_integrity", "artifact_integrity_sha256"):
                if not item.get(field):
                    _add_error(issues, f"supporting_{field}_not_declared", f"{artifact_id} needs {field}")
        if item.get("artifact_integrity") and not _SHA256_RE.fullmatch(str(item.get("artifact_integrity_sha256") or "")):
            _add_error(issues, "supporting_integrity_root_sha_mismatch", f"{artifact_id} artifact_integrity_sha256 is invalid")
        if check_git and _GIT_COMMIT_RE.fullmatch(commit) and subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo_root,
                                                   capture_output=True, check=False).returncode:
            _add_error(issues, "supporting_implementation_commit_unresolvable", f"{artifact_id} commit is unresolvable")

        refs: list[tuple[str, Path, bool]] = []
        for field in _SUPPORT_FILES:
            if field in item:
                path = _resolved_path(item[field], repo_root)
                if path is None:
                    _add_error(issues, "supporting_path_invalid", f"{artifact_id} {field} must be a path string")
                else:
                    refs.append((field, path, field in {"artifact_manifest", "artifact_integrity"}))
        for map_field, strict in _SUPPORT_MAPS.items():
            value = item.get(map_field)
            if value is not None and not isinstance(value, dict):
                _add_error(issues, "supporting_path_map_invalid", f"{artifact_id} {map_field} must be a mapping")
            elif isinstance(value, dict):
                for name, raw_path in value.items():
                    path = _resolved_path(raw_path, repo_root)
                    if path is None:
                        _add_error(issues, "supporting_path_invalid", f"{artifact_id} {map_field}.{name} is invalid")
                    else:
                        refs.append((f"{map_field}.{name}", path, strict))
        dirs = []
        for field in _SUPPORT_DIRS:
            if field in item:
                path = _resolved_path(item[field], repo_root)
                if path is None:
                    _add_error(issues, "supporting_path_invalid", f"{artifact_id} {field} must be a path string")
                else:
                    dirs.append((field, path))
        integrity = {"checked": 0, "missing": 0, "mismatch": 0}
        if check_files:
            for field, path, strict in refs:
                if not _require_path(path, issues, "supporting_file_missing", f"{artifact_id} {field}"):
                    continue
                if strict:
                    result = _audit_strict_manifest(path, field, item, context, issues, cache, stats)
                    for key in integrity:
                        integrity[key] += result[key]
            for field, path in dirs:
                _require_path(path, issues, "supporting_directory_missing", f"{artifact_id} {field}", directory=True)
        entries.append({"artifact_id": str(artifact_id), "status": support_status, "replay_status": replay,
                        "declared_file_count": len(refs), "integrity": integrity,
                        "valid": _error_count(issues) == before})
    return {"count": len(entries), "valid_count": sum(e["valid"] for e in entries), "entries": entries}


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

    configured_experiment = (config.get("experiment") or {}).get("name")
    if method.get("experiment_name") != configured_experiment:
        issues.append(
            _issue(
                "error",
                "config_experiment_mismatch",
                f"{method_key} experiment_name disagrees with resolved config",
                observed=method.get("experiment_name"),
                expected=configured_experiment,
            )
        )

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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "k500_ref_bad_json", f"Invalid K500 ref JSON: {ref_path}: {exc}"))
            continue
        if not isinstance(meta, dict):
            issues.append(_issue("error", "k500_ref_invalid", f"K500 ref JSON root must be an object: {ref_path}"))
            continue
        ref_ids = meta.get("ref_record_ids")
        if (not isinstance(meta.get("K"), int) or isinstance(meta.get("K"), bool)
                or not isinstance(meta.get("selection_seed"), int) or isinstance(meta.get("selection_seed"), bool)
                or not isinstance(ref_ids, list) or not all(isinstance(item, str) and item for item in ref_ids)
                or len(set(ref_ids)) != len(ref_ids)):
            issues.append(_issue("error", "k500_ref_invalid", f"K500 ref JSON field types are invalid: {ref_path}"))
            continue
        checks = {
            "center": meta.get("center") == center,
            "K": meta["K"] == kshot["k"],
            "seed": meta["selection_seed"] == kshot["seed"],
            "mapping_version": meta.get("mapping_version") == protocol["mapping_version"],
            "mapping_hash": meta.get("mapping_hash") == protocol["mapping_hash"],
            "ref_record_ids": len(ref_ids) == kshot["k"],
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
            "k": len(ref_ids),
            "selection_seed": meta["selection_seed"],
            "ref_record_ids_sha256": ref_ids_sha256(ref_ids),
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
    rows = _read_csv(
        metrics_path,
        issues,
        code="metrics_long_invalid",
        label=f"{method_key} metrics_long",
    )
    if rows is None:
        return out
    if not rows:
        issues.append(_issue("error", "metrics_long_empty", f"{method_key} metrics_long has no rows"))
        return out
    if list(rows[0].keys()) != METRICS_FIELDNAMES:
        issues.append(_issue("error", "metrics_long_bad_columns", f"{method_key} metrics_long columns changed"))
        out["n_rows"] = len(rows)
        return out
    if any(
        not isinstance(row.get(field), str)
        for row in rows
        for field in METRICS_FIELDNAMES
    ):
        issues.append(_issue("error", "metrics_long_bad_values", f"{method_key} metrics_long has truncated rows"))
        out["n_rows"] = len(rows)
        return out
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
        rows = _read_csv(
            table_path,
            issues,
            code="paper_table_invalid",
            label=f"{method_key} paper table {view}",
        )
        if rows is None:
            continue
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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(_issue("error", "manifest_bad_json", f"{method_key} manifest invalid: {exc}"))
        return {"declared": True, "path": str(manifest_path), "exists": True}
    if not isinstance(manifest, dict):
        issues.append(_issue("error", "manifest_invalid", f"{method_key} manifest root must be an object"))
        return {"declared": True, "path": str(manifest_path), "exists": True}
    status = manifest.get("status")
    if not isinstance(status, str):
        issues.append(
            _issue(
                "error",
                "manifest_status_invalid",
                f"{method_key} manifest status must be a string",
                method=method_key,
            )
        )
    elif status not in {"succeeded", "dry_run"}:
        issues.append(
            _issue(
                "warning",
                "manifest_status_not_succeeded",
                f"{method_key} manifest status is {status!r}",
                method=method_key,
            )
        )
    manifest_kind = manifest.get("manifest_kind", "managed_launcher_manifest")
    if manifest_kind != "managed_launcher_manifest":
        issues.append(
            _issue(
                "error",
                "manifest_kind_not_managed",
                f"{method_key} manifest_kind is {manifest_kind!r}, expected managed_launcher_manifest",
            )
        )
    return {
        "declared": True,
        "path": str(manifest_path),
        "exists": True,
        "status": status,
        "manifest_kind": manifest_kind,
    }


def _audit_evaluation_k500_identity(
    claim: dict[str, Any],
    method_key: str,
    method: dict[str, Any],
    claim_identities: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if claim.get("status") not in {"trusted", "provisional"} or method.get("status") not in {"trusted", "provisional"}:
        return {"skipped": True}
    code = "evaluation_k500_identity_missing"
    required = claim.get("required_reporting_artifacts")
    if not isinstance(required, dict):
        _add_error(issues, code, f"{method_key} required_reporting_artifacts must be a mapping")
        return {"skipped": False, "training": {}, "evaluation": {}}
    manifests = required.get("method_artifact_manifests")
    if not isinstance(manifests, dict):
        _add_error(issues, code, f"{method_key} method_artifact_manifests must be a mapping")
        return {"skipped": False, "training": {}, "evaluation": {}}
    eval_manifest_path = _path_or_none(manifests.get(method_key))
    if eval_manifest_path is None:
        _add_error(issues, code, f"{method_key} has no evaluation artifact manifest")
        return {"skipped": False, "training": {}, "evaluation": {}}

    def read(path: Path | None, label: str) -> dict[str, Any]:
        if not _require_path(path, issues, code, label):
            return {}
        assert path is not None
        return _read_json_object(path, issues, code, label) or {}

    method_manifest_path = _path_or_none(method.get("manifest"))
    method_manifest = read(method_manifest_path, f"{method_key} training manifest")
    launch_artifacts = (((method_manifest.get("artifact_trace") or {}).get("expected_outputs") or {}).get("launch_artifacts") or [])
    k500_record = next(
        (record for record in launch_artifacts if isinstance(record, dict) and record.get("role") == "k500_ref_ids"),
        {},
    )
    base = method_manifest_path.parent if method_manifest_path else Path()
    expected = read(
        _resolved_path(k500_record.get("path"), base),
        f"{method_key} k500_ref_ids artifact",
    ).get("centers") or {}
    expected_centers = {str(center) for center in (claim.get("protocol") or {}).get("target_centers") or []}
    if not isinstance(expected, dict) or not expected:
        _add_error(issues, code, f"{method_key} training K500 identities are empty")
        expected = {}
    elif set(expected) != expected_centers:
        _add_error(
            issues,
            "evaluation_k500_identity_mismatch",
            f"{method_key} training K500 centers do not match claim",
            expected=sorted(expected_centers),
            observed=sorted(expected),
        )
    claim_kshot = (claim.get("protocol") or {}).get("kshot") or {}
    for center in expected_centers:
        identity = expected.get(center)
        claimed = claim_identities.get(center)
        if not isinstance(identity, dict) or not isinstance(claimed, dict):
            _add_error(issues, code, f"{method_key} training identity missing for {center}")
            continue
        identity_hash = str(identity.get("ref_record_ids_sha256") or "")
        if (
            not isinstance(identity.get("k"), int)
            or isinstance(identity.get("k"), bool)
            or not isinstance(identity.get("selection_seed"), int)
            or isinstance(identity.get("selection_seed"), bool)
            or not _SHA256_RE.fullmatch(identity_hash)
        ):
            _add_error(issues, code, f"{method_key} training identity is invalid for {center}")
            continue
        claimed_identity = {
            "k": claimed.get("k"),
            "selection_seed": claimed.get("selection_seed"),
            "ref_record_ids_sha256": claimed.get("ref_record_ids_sha256"),
        }
        observed_identity = {
            "k": identity.get("k"),
            "selection_seed": identity.get("selection_seed"),
            "ref_record_ids_sha256": identity_hash,
        }
        registry_identity = {
            "k": claim_kshot.get("k"),
            "selection_seed": claim_kshot.get("seed"),
            "ref_record_ids_sha256": claimed_identity["ref_record_ids_sha256"],
        }
        if observed_identity != claimed_identity or observed_identity != registry_identity:
            _add_error(
                issues,
                "evaluation_k500_identity_mismatch",
                f"{method_key} training K500 identity disagrees with claim for {center}",
                expected=registry_identity,
                observed=observed_identity,
            )

    eval_manifest = read(eval_manifest_path, f"{method_key} evaluation artifact manifest")
    observed: dict[str, dict[str, Any]] = {}
    artifacts = eval_manifest.get("artifacts")
    eval_artifacts = [
        artifact
        for artifact in artifacts or []
        if isinstance(artifact, dict) and artifact.get("artifact_type") == "eval_result"
    ] if isinstance(artifacts, list) else []
    if not eval_artifacts:
        _add_error(issues, code, f"{method_key} evaluation artifact manifest has no eval_result artifacts")
    for artifact in eval_artifacts:
        eval_path = _resolved_path(artifact.get("path"), eval_manifest_path.parent)
        declared_sha = str(artifact.get("sha256") or "")
        if not _SHA256_RE.fullmatch(declared_sha):
            _add_error(issues, "evaluation_artifact_sha256_mismatch", f"{method_key} eval_result has no valid sha256")
            continue
        if not _require_path(eval_path, issues, code, f"{method_key} evaluation result"):
            continue
        assert eval_path is not None
        if _sha256(eval_path) != declared_sha:
            _add_error(
                issues,
                "evaluation_artifact_sha256_mismatch",
                f"{method_key} eval_result SHA mismatch: {eval_path}",
            )
            continue
        try:
            identities = evaluation_k500_identities(read(eval_path, f"{method_key} evaluation result"))
        except PN2021CMetadataError as exc:
            _add_error(issues, code, f"{method_key} evaluation identity is invalid: {exc}")
            continue
        for center, identity in identities.items():
            if center in observed and observed[center] != identity:
                _add_error(issues, "evaluation_k500_identity_mismatch", f"{method_key} has multiple identities for {center}")
            observed[center] = identity

    if set(observed) != expected_centers:
        _add_error(
            issues,
            "evaluation_k500_identity_mismatch",
            f"{method_key} evaluation K500 centers do not match claim",
            expected=sorted(expected_centers),
            observed=sorted(observed),
        )

    for center in expected_centers:
        training_identity = expected.get(center) or {}
        actual = observed.get(str(center))
        expected_pair = {
            "k": training_identity.get("k"),
            "ref_record_ids_sha256": training_identity.get("ref_record_ids_sha256"),
        }
        if actual is None:
            _add_error(issues, code, f"{method_key} evaluation identity missing for {center}")
        elif actual != expected_pair:
            _add_error(
                issues,
                "evaluation_k500_identity_mismatch",
                f"{method_key} training/evaluation K500 identity mismatch for {center}",
                method=method_key,
                center=center,
                expected=expected_pair,
                observed=actual,
            )
    return {"skipped": False, "training": expected, "evaluation": observed}


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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "comparison_manifest_bad_json", f"Invalid comparison manifest: {exc}"))
        else:
            if not isinstance(manifest, dict):
                issues.append(_issue("error", "comparison_manifest_invalid", "Comparison manifest root must be an object"))
                return summary
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
    local_context = _resolve_context(_read_yaml(local_config_path))
    sha_cache: dict[Path, str] = {}
    hash_stats = {"computed": 0, "cache_hits": 0}
    issues: list[dict[str, Any]] = []
    _audit_raw_registry_boundaries(raw_registry, issues)
    artifact_policy = registry.get("artifact_policy")
    if artifact_policy is None:
        registry["artifact_policy"] = {}
    elif not isinstance(artifact_policy, dict):
        _add_error(issues, "artifact_policy_invalid", "artifact_policy must be a mapping")
        registry["artifact_policy"] = {}
    else:
        do_not_commit = artifact_policy.get("do_not_commit")
        if do_not_commit is not None and (
            not isinstance(do_not_commit, list)
            or not all(isinstance(item, str) and item for item in do_not_commit)
        ):
            _add_error(issues, "artifact_policy_invalid", "artifact_policy.do_not_commit must be a string list")
            registry["artifact_policy"] = {**artifact_policy, "do_not_commit": []}
    raw_claims_value = registry.get("active_claims")
    if raw_claims_value is None:
        raw_claims = []
    elif not isinstance(raw_claims_value, list):
        _add_error(issues, "active_claims_invalid", "active_claims must be a list")
        raw_claims = []
    else:
        raw_claims = raw_claims_value
    claims = []
    for index, claim in enumerate(raw_claims):
        if not isinstance(claim, dict):
            _add_error(issues, "active_claim_invalid", f"active_claims[{index}] must be a mapping")
            continue
        if _claim_shape_is_valid(claim, index, issues):
            claims.append(claim)
    registry["active_claims"] = claims
    managed_run_summary = _audit_managed_runs(
        repo_root,
        registry,
        issues,
        check_files=require_existing_artifacts,
        cache=sha_cache,
        stats=hash_stats,
    )

    claim_summaries: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = claim["claim_id"]
        k500_summary = _audit_k500_refs(claim, issues) if require_existing_artifacts else {}
        method_summaries: dict[str, Any] = {}
        config_summaries: dict[str, Any] = {}
        for method_key, method in claim["methods"].items():
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
                method_summary["evaluation_k500_identity"] = _audit_evaluation_k500_identity(
                    claim, method_key, method, k500_summary, issues
                )
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
                "supporting_reporting_artifacts": _audit_supporting_reporting_artifacts(
                    repo_root,
                    claim,
                    issues,
                    check_files=require_existing_artifacts,
                    check_git=check_git,
                    context=local_context,
                    cache=sha_cache,
                    stats=hash_stats,
                ),
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
        "managed_runs": managed_run_summary,
        "artifact_hashing": hash_stats,
        "claims": claim_summaries,
        "git": git_summary,
    }
