"""Minimal integrity-tracked run records for the manual-refactor launcher."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_EXCLUDED_FILES = frozenset({"run_manifest.json", "run_file_index.json"})
DATA_LEDGER_SNAPSHOT = Path("manifests/data_content_ledger.jsonl")
RESULT_REQUIRED_KEYS = {
    "supervised_train_result": frozenset(
        {"config", "epochs_completed", "model", "selected_checkpoint", "selected_epoch"}
    ),
    "pn2021_train_result": frozenset(
        {
            "artifact_type",
            "center",
            "config",
            "epochs_completed",
            "last_checkpoint",
            "lineage",
            "method_id",
            "model",
            "optimizer_steps",
            "schema_version",
            "scientific_arm",
            "selection",
        }
    ),
    "evaluation_result": frozenset({"artifact_type", "checkpoint", "clean", "corrupted",
                                    "model", "protocol", "schema_version", "status", "subject"}),
    "pn2021_matrix_result": frozenset({"aggregation", "artifact_type", "centers", "clean",
                                       "cohort", "corrupted", "members", "model",
                                       "schema_version", "status"}),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout


def _git_text(repo_root: Path, *arguments: str) -> str:
    return _git_bytes(repo_root, *arguments).decode("utf-8", errors="replace").strip()


def capture_git_state(repo_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Capture commit identity plus a content hash of the complete dirty layer."""

    root = Path(repo_root).resolve()
    commit = _git_text(root, "rev-parse", "HEAD")
    branch = _git_text(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    tracked_diff = _git_bytes(root, "diff", "--binary", "HEAD", "--no-ext-diff")
    untracked = _git_bytes(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if candidate.is_symlink():
            digest.update(os.readlink(candidate).encode("utf-8", errors="surrogateescape"))
        elif candidate.is_file():
            digest.update(sha256_file(candidate).encode("ascii"))
        else:
            digest.update(b"missing-or-nonfile")
    status_text = status.decode("utf-8", errors="replace").replace("\x00", "\n")
    changed_paths = sorted(
        {
            line[3:]
            for line in status_text.splitlines()
            if len(line) >= 4 and line[3:]
        }
    )
    return {
        "worktree": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_diff_sha256": digest.hexdigest(),
        "changed_paths": changed_paths,
    }


def capture_environment() -> dict[str, Any]:
    """Capture process/runtime identity; Torch is imported only on execution."""

    try:
        import torch

        torch_version: str | None = str(torch.__version__)
        torch_cuda_version: str | None = torch.version.cuda
        cuda_available: bool | None = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - exercised only in broken envs
        torch_version = None
        torch_cuda_version = None
        cuda_available = None
        torch_error = f"{type(exc).__name__}: {exc}"
    else:
        torch_error = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "torch_cuda_available": cuda_available,
        "torch_import_error": torch_error,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def ensure_output_outside_worktree(
    output_dir: str | Path,
    *,
    worktree: str | Path = PROJECT_ROOT,
) -> Path:
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    root = Path(worktree).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        return output
    raise ValueError(f"run output must remain outside the Git worktree: {output}")


def _snapshot_destination(
    source: Path,
    *,
    config_root: Path,
) -> Path:
    try:
        relative = source.relative_to(config_root)
    except ValueError:
        relative = Path("external") / f"{sha256_file(source)[:12]}_{source.name}"
    return Path("configs") / relative


def _prepare_yaml_snapshots(
    sources: Sequence[tuple[Path, str]],
    *,
    config_root: Path,
) -> list[tuple[Path, bytes, dict[str, Any]]]:
    prepared: list[tuple[Path, bytes, dict[str, Any]]] = []
    seen: set[Path] = set()
    for raw_source, role in sources:
        source = raw_source.resolve()
        if source in seen:
            continue
        seen.add(source)
        if not source.is_file() or source.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"config snapshot source must be a YAML file: {source}")
        relative = _snapshot_destination(source, config_root=config_root)
        payload = source.read_bytes()
        prepared.append(
            (relative, payload, {
                "role": str(role),
                "source_path": str(source),
                "snapshot_path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            })
        )
    return prepared


def _write_yaml_snapshots(
    run_dir: Path, prepared: Sequence[tuple[Path, bytes, dict[str, Any]]]
) -> list[dict[str, Any]]:
    for relative, payload, _ in prepared:
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return [metadata for _, _, metadata in prepared]


def snapshot_yaml_files(
    run_dir: Path,
    sources: Sequence[tuple[Path, str]],
    *,
    config_root: Path,
) -> list[dict[str, Any]]:
    snapshots = _prepare_yaml_snapshots(sources, config_root=config_root)
    return _write_yaml_snapshots(run_dir, snapshots)


def _artifact_role(relative: Path) -> str:
    value = relative.as_posix()
    if relative == DATA_LEDGER_SNAPSHOT:
        return "data_content_ledger_snapshot"
    if value.startswith("configs/"):
        return "config_snapshot"
    if value.startswith("logs/"):
        return "log"
    if value.startswith("training/"):
        return "delegate_artifact"
    if value == "run_card.json":
        return "run_card"
    if value == "data_manifest.json":
        return "data_manifest"
    if value == "selection.json":
        return "selection_record"
    if value == "summary.md":
        return "summary"
    return "run_artifact"


def build_run_file_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run file index refuses symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in INDEX_EXCLUDED_FILES or path.name.endswith(".tmp"):
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "role": _artifact_role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "excluded_to_avoid_recursive_hashing": sorted(INDEX_EXCLUDED_FILES),
        "files": entries,
    }


def verify_run_file_index(run_dir: str | Path) -> list[str]:
    root = Path(run_dir).resolve()
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    index_path = root / "run_file_index.json"
    if not manifest_path.is_file() or not index_path.is_file():
        return ["run_manifest.json or run_file_index.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_index_hash = manifest.get("run_file_index_sha256")
    actual_index_hash = sha256_file(index_path)
    if expected_index_hash != actual_index_hash:
        errors.append("run_file_index.json SHA256 does not match run_manifest.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_paths: set[str] = set()
    for entry in index.get("files", []):
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe indexed path: {relative}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"indexed path escapes run directory: {relative}")
            continue
        indexed_paths.add(relative.as_posix())
        if path.is_symlink() or not path.is_file():
            errors.append(f"indexed file missing or is a symlink: {relative}")
            continue
        if path.stat().st_size != entry.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"SHA256 mismatch: {relative}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in INDEX_EXCLUDED_FILES
        and not path.name.endswith(".tmp")
    }
    for relative in sorted(actual_paths - indexed_paths):
        errors.append(f"unindexed file: {relative}")
    return errors


def _pick(payload: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys}


def _require_result_keys(payload: dict[str, Any], result_type: str) -> None:
    required = RESULT_REQUIRED_KEYS.get(result_type)
    if required is None or not required.issubset(payload):
        raise ValueError(f"expected {result_type} keys are missing")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verified_artifact(value: Any, description: str, *, parent: Path,
                       contained: bool = False) -> Path:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"},
             f"{description} must contain exactly path and sha256")
    path, digest = value["path"], value["sha256"]
    _require(isinstance(path, str) and bool(path) and isinstance(digest, str)
             and len(digest) == 64 and not set(digest) - set("0123456789abcdef"),
             f"{description} has an invalid path or SHA256")
    unresolved = Path(path).expanduser()
    candidate = unresolved if unresolved.is_absolute() else parent / unresolved
    resolved = candidate.resolve()
    if contained:
        try:
            resolved.relative_to(parent.resolve())
        except ValueError:
            raise ValueError(f"{description} must stay inside its output directory") from None
    _require(not candidate.is_symlink() and resolved.is_file(),
             f"{description} file is missing or is a symlink")
    _require(sha256_file(resolved) == digest,
             f"{description} SHA256 does not match the file")
    return resolved


def _train_result_summary(payload: dict[str, Any], result_type: str, *, result_path: Path) -> dict[str, Any]:
    _require_result_keys(payload, result_type)
    if result_type == "pn2021_train_result":
        from util.pn2021_artifact_contract import validate_pn2021_train_result
        validate_pn2021_train_result(payload, result_path=result_path)
        selection = payload["selection"]
        return {
            "identity": _pick(payload, ("model", "center", "method_id", "scientific_arm",
                                        "epochs_completed", "lineage")),
            "selection": _pick(selection, ("policy", "selected_epoch", "selected_checkpoint",
                                            "heldout_evaluation_used_for_selection")),
        }
    epochs = payload["epochs_completed"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("train_result epochs_completed must be positive")
    selection = payload.get("selection")
    if result_type == "supervised_train_result":
        if "selection" in payload and (
            not isinstance(selection, dict)
            or selection.get("heldout_evaluation_used_for_selection") is not False
        ):
            raise ValueError("explicit train_result selection must be heldout-free")
        selected_epoch = payload.get("selected_epoch")
        if (
            isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int)
            or not 1 <= selected_epoch <= epochs
            or not isinstance(payload.get("selected_checkpoint"), dict)
        ):
            raise ValueError("supervised train_result selection is incomplete")
    else:
        raise ValueError(f"unsupported train result type: {result_type}")
    selected = (
        _pick(
            selection,
            (
                "policy",
                "selected_epoch",
                "selected_checkpoint",
                "heldout_evaluation_used_for_selection",
            ),
        )
        if isinstance(selection, dict)
        else _pick(payload, ("selected_epoch", "selected_checkpoint"))
    )
    return {
        "identity": _pick(
            payload,
            ("model", "center", "method_id", "scientific_arm", "epochs_completed", "lineage"),
        ),
        "selection": selected,
    }


def _matrix_result_summary(payload: dict[str, Any], result_path: Path) -> dict[str, Any]:
    from util.evaluation.matrix import aggregate_pn2021_matrix

    _require_result_keys(payload, "pn2021_matrix_result")
    members = payload["members"]
    _require(isinstance(members, list) and len(members) == 4,
             "matrix_result must contain four members")
    paths: list[Path] = []
    for member in members:
        _require(isinstance(member, dict), "matrix_result member must be a mapping")
        paths.append(_verified_artifact(
            member.get("evaluation_result"), "matrix member evaluation_result",
            parent=result_path.parent,
        ))
    config_path = _verified_artifact(payload.get("config"), "matrix_result config", parent=result_path.parent)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = config.get("profiles") if isinstance(config, dict) else None
    matches = [item for item in profiles.values() if isinstance(item, dict)
               and item.get("profile_name") == payload.get("profile_name")] if isinstance(profiles, dict) else []
    _require(len(matches) == 1, "matrix_result config does not own its profile")
    expected = aggregate_pn2021_matrix(
        paths, profile_name=payload.get("profile_name"),
        expected_cohort=matches[0].get("expected_cohort"),
    )
    _require({key: value for key, value in payload.items()
              if key not in {"config", "output"}} == expected,
             "matrix_result differs from recomputed member evidence")
    _require(payload.get("output") == {"directory": str(result_path.parent),
                                       "result_file": str(result_path)},
             "matrix_result output identity is invalid")
    return {"identity": _pick(payload, ("model", "cohort", "centers", "members")),
            "selection": payload["aggregation"]}


def _result_summary(payload: dict[str, Any], result_type: str, *, result_path: Path) -> dict[str, Any]:
    if result_type in {"supervised_train_result", "pn2021_train_result"}:
        return _train_result_summary(payload, result_type, result_path=result_path)
    if result_type == "pn2021_matrix_result":
        return _matrix_result_summary(payload, result_path)
    from util.pn2021_artifact_contract import validate_pn2021_evaluation_result

    _require_result_keys(payload, result_type)
    validate_pn2021_evaluation_result(
        payload, result_path=result_path,
        registry_loader=lambda path: yaml.safe_load(path.read_text(encoding="utf-8")),
    )
    return {
        "identity": _pick(payload, ("model", "checkpoint", "protocol", "subject")),
        "selection": {"policy": "checkpoint_fixed_before_evaluation"},
    }


def _expected_result_evidence(
    run_dir: Path, relative_path: Path, result_type: str
) -> dict[str, Any]:
    result_path = run_dir / relative_path
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError(f"expected result is missing: {relative_path.as_posix()}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"expected result is not valid JSON: {relative_path.as_posix()}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected {result_type} result must be a JSON mapping")
    return {
        "path": relative_path.as_posix(), "type": result_type,
        "sha256": sha256_file(result_path), "size_bytes": result_path.stat().st_size,
        **_result_summary(payload, result_type, result_path=result_path),
    }


@dataclass(frozen=True)
class VerifiedDataContent:
    ledger_bytes: bytes = field(repr=False)
    expected_sha256: str
    required_roots: tuple[str, ...]
    config_relative_path: Path
    closure_sha256: tuple[tuple[Path, str], ...]
    verification: dict[str, Any]


def _validated_data_content(value: VerifiedDataContent) -> dict[str, Any]:
    digest = hashlib.sha256(value.ledger_bytes).hexdigest()
    verification = value.verification
    roots = verification.get("roots")
    if (
        digest != value.expected_sha256
        or digest != verification.get("ledger_sha256")
        or verification.get("verification_mode") != "inventory_only"
        or not isinstance(roots, dict)
        or tuple(roots) != value.required_roots
    ):
        raise ValueError("verified data content differs from the execution plan")
    try:
        header = json.loads(value.ledger_bytes.splitlines()[0])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("verified data ledger bytes lack a valid header") from exc
    if not isinstance(header, dict) or not all(
        type(header.get(key)) is int and header[key] >= 0
        for key in ("member_count", "total_size_bytes")
    ):
        raise ValueError("verified data ledger header seal is invalid")
    config_snapshot = Path("configs") / value.config_relative_path
    if value.config_relative_path.is_absolute() or ".." in value.config_relative_path.parts:
        raise ValueError("data ledger config-relative path is unsafe")
    return {
        "expected_ledger_sha256": value.expected_sha256,
        "verified_ledger_sha256": digest,
        "ledger_snapshot": DATA_LEDGER_SNAPSHOT.as_posix(),
        "config_ledger_snapshot": config_snapshot.as_posix(),
        "required_roots": list(value.required_roots),
        "config_closure_sha256": {
            str(path.resolve()): digest for path, digest in value.closure_sha256
        },
        "full_ledger_member_seal": {key: header[key] for key in
                                    ("member_count", "total_size_bytes")},
        "execution_verification": {
            "mode": "quick_exact_inventory_and_size",
            **{key: verification.get(key) for key in
               ("member_count", "total_size_bytes")}, "roots": roots,
        },
    }


def _validate_data_content_snapshot(run_dir: Path, record: Any) -> None:
    if record is None:
        return
    for relative in (record.get("ledger_snapshot"), record.get("config_ledger_snapshot")):
        snapshot = run_dir / str(relative)
        try:
            metadata = snapshot.lstat()
        except OSError as exc:
            raise ValueError("data content ledger snapshot is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("data content ledger snapshot must be a regular file")
        if sha256_file(snapshot) != record.get("expected_ledger_sha256"):
            raise ValueError("data content ledger snapshot SHA-256 mismatch")


@dataclass
class RunRecorder:
    run_dir: Path
    manifest: dict[str, Any]
    card: dict[str, Any]
    _started_monotonic: float = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        run_dir: str | Path,
        experiment_name: str,
        purpose: str,
        entrypoint_name: str,
        entrypoint_path: Path,
        config_root: Path,
        config_sources: Sequence[tuple[Path, str]],
        exact_launcher_argv: Sequence[str],
        exact_delegate_argv: Sequence[str],
        expected_result_relative_path: str | Path,
        expected_result_type: str,
        data_content: VerifiedDataContent | None = None,
        cwd: str | Path = PROJECT_ROOT,
    ) -> "RunRecorder":
        output = ensure_output_outside_worktree(run_dir)
        if output.exists():
            raise FileExistsError(f"run directory already exists: {output}")
        expected_result = Path(expected_result_relative_path)
        if (
            expected_result.is_absolute()
            or expected_result == Path(".")
            or ".." in expected_result.parts
        ):
            raise ValueError("expected result must be a safe run-relative path")
        if expected_result_type not in RESULT_REQUIRED_KEYS:
            raise ValueError(f"unsupported expected result type: {expected_result_type}")
        data_content_record = (None if data_content is None
                               else _validated_data_content(data_content))
        prepared_snapshots = _prepare_yaml_snapshots(
            config_sources, config_root=config_root.resolve()
        )
        if data_content is not None:
            actual = {
                item["source_path"]: item["sha256"]
                for _, _, item in prepared_snapshots
            }
            expected_data_configs = {
                str(path.resolve()): digest for path, digest in data_content.closure_sha256
            }
            if any(actual.get(path) != digest
                   for path, digest in expected_data_configs.items()):
                raise ValueError("data ledger root configuration changed before snapshot")
        output.mkdir(parents=True, exist_ok=False)
        if data_content is not None:
            snapshot = output / DATA_LEDGER_SNAPSHOT
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data_content.ledger_bytes)
            config_snapshot = output / "configs" / data_content.config_relative_path
            config_snapshot.parent.mkdir(parents=True, exist_ok=True)
            config_snapshot.write_bytes(data_content.ledger_bytes)
        snapshots = _write_yaml_snapshots(output, prepared_snapshots)
        started = utc_now()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "experiment_name": experiment_name,
            "purpose": purpose,
            "started_at_utc": started,
            "finished_at_utc": None,
            "duration_seconds": None,
            "exit_code": None,
            "delegate_exit_code": None,
            "cwd": str(Path(cwd).resolve()),
            "exact_launcher_argv": [str(value) for value in exact_launcher_argv],
            "exact_delegate_argv": [str(value) for value in exact_delegate_argv],
            "entrypoint": {
                "name": entrypoint_name,
                "path": str(entrypoint_path.resolve()),
                "sha256": sha256_file(entrypoint_path),
            },
            "config_root": str(config_root.resolve()),
            "config_snapshots": snapshots,
            "data_content": data_content_record,
            "expected_result": {
                "path": expected_result.as_posix(),
                "type": expected_result_type,
            },
            "delegate_result": None,
            "environment": capture_environment(),
            "git": capture_git_state(PROJECT_ROOT),
            "run_file_index_sha256": None,
            "error": None,
        }
        card = {
            "schema_version": 1,
            "experiment_name": experiment_name,
            "purpose": purpose,
            "status": "running",
            "entrypoint": entrypoint_name,
            "started_at_utc": started,
            "finished_at_utc": None,
            "exit_code": None,
            "replay_status": "running",
            "paper_eligibility": "not_assessed",
        }
        recorder = cls(
            run_dir=output,
            manifest=manifest,
            card=card,
            _started_monotonic=time.monotonic(),
        )
        recorder._write_live_files()
        return recorder

    def _write_live_files(self) -> None:
        _write_json_atomic(self.run_dir / "run_manifest.json", self.manifest)
        _write_json_atomic(self.run_dir / "run_card.json", self.card)
        (self.run_dir / "summary.md").write_text(
            "# Run summary\n\n"
            f"- Experiment: `{self.card['experiment_name']}`\n"
            f"- Status: `{self.card['status']}`\n"
            f"- Entrypoint: `{self.card['entrypoint']}`\n",
            encoding="utf-8",
        )

    def finalize(self, *, exit_code: int, error: str | None = None) -> int:
        finished = utc_now()
        delegate_exit_code = int(exit_code)
        effective_error = error
        result_evidence = None
        expected = self.manifest["expected_result"]
        if delegate_exit_code == 0 and effective_error is None:
            try:
                _validate_data_content_snapshot(
                    self.run_dir, self.manifest.get("data_content")
                )
                result_evidence = _expected_result_evidence(
                    self.run_dir, Path(expected["path"]), str(expected["type"])
                )
            except (OSError, ValueError) as exc:
                effective_error = str(exc)
        effective_exit_code = delegate_exit_code or int(effective_error is not None)
        status = "complete" if effective_exit_code == 0 else "failed"
        duration = max(0.0, time.monotonic() - self._started_monotonic)
        self.manifest.update(
            {
                "status": status,
                "finished_at_utc": finished,
                "duration_seconds": duration,
                "exit_code": effective_exit_code,
                "delegate_exit_code": delegate_exit_code,
                "delegate_result": result_evidence,
                "error": effective_error,
            }
        )
        self.card.update(
            {
                "status": status,
                "finished_at_utc": finished,
                "exit_code": effective_exit_code,
                "replay_status": (
                    "recorded_not_yet_audited" if status == "complete" else "failed"
                ),
            }
        )
        self._write_live_files()
        index = build_run_file_index(self.run_dir)
        index_path = self.run_dir / "run_file_index.json"
        _write_json_atomic(index_path, index)
        self.manifest["run_file_index_sha256"] = sha256_file(index_path)
        _write_json_atomic(self.run_dir / "run_manifest.json", self.manifest)
        return effective_exit_code


__all__ = [
    "PROJECT_ROOT",
    "RunRecorder",
    "VerifiedDataContent",
    "build_run_file_index",
    "capture_environment",
    "capture_git_state",
    "ensure_output_outside_worktree",
    "sha256_file",
    "snapshot_yaml_files",
    "utc_now",
    "verify_run_file_index",
]
