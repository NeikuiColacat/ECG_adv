"""Minimal integrity-tracked run records for the manual-refactor launcher."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_EXCLUDED_FILES = frozenset({"run_manifest.json", "run_file_index.json"})
RESULT_REQUIRED_KEYS = {
    "supervised_train_result": frozenset(
        {"config", "epochs_completed", "model", "selected_checkpoint", "selected_epoch"}
    ),
    "pn2021_train_result": frozenset(
        {
            "center",
            "config",
            "epochs_completed",
            "last_checkpoint",
            "method_id",
            "model",
            "optimizer_steps",
            "scientific_arm",
            "selection",
        }
    ),
    "evaluation_result": frozenset(
        {"schema_version", "checkpoint", "clean", "corrupted", "protocol", "status"}
    ),
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


def snapshot_yaml_files(
    run_dir: Path,
    sources: Sequence[tuple[Path, str]],
    *,
    config_root: Path,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_source, role in sources:
        source = raw_source.resolve()
        if source in seen:
            continue
        seen.add(source)
        if not source.is_file() or source.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"config snapshot source must be a YAML file: {source}")
        relative = _snapshot_destination(source, config_root=config_root)
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        snapshots.append(
            {
                "role": str(role),
                "source_path": str(source),
                "snapshot_path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return snapshots


def _artifact_role(relative: Path) -> str:
    value = relative.as_posix()
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


def _train_result_summary(
    payload: dict[str, Any], result_type: str
) -> dict[str, Any]:
    _require_result_keys(payload, result_type)
    epochs = payload["epochs_completed"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("train_result epochs_completed must be positive")
    selection = payload.get("selection")
    if result_type == "pn2021_train_result":
        method_id = payload["method_id"]
        optimizer_steps = payload.get("optimizer_steps")
        valid = (
            isinstance(method_id, str) and bool(method_id) and isinstance(selection, dict)
            and all(
                isinstance(payload.get(key), str) and payload[key]
                for key in ("center", "scientific_arm")
            )
            and isinstance(optimizer_steps, int)
            and not isinstance(optimizer_steps, bool)
            and optimizer_steps > 0
            and selection.get("policy") == "last"
            and selection.get("selected_epoch") == epochs
            and isinstance(payload.get("last_checkpoint"), dict)
            and selection.get("selected_checkpoint") == payload["last_checkpoint"]
            and selection.get("heldout_evaluation_used_for_selection") is False
        )
        if not valid:
            raise ValueError("typed train_result selection is not the heldout-free last checkpoint")
    elif result_type == "supervised_train_result":
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
            ("model", "center", "method_id", "scientific_arm", "epochs_completed"),
        ),
        "selection": selected,
    }


def _result_summary(payload: dict[str, Any], result_type: str) -> dict[str, Any]:
    if result_type in {"supervised_train_result", "pn2021_train_result"}:
        return _train_result_summary(payload, result_type)
    _require_result_keys(payload, result_type)
    protocol, checkpoint = payload["protocol"], payload["checkpoint"]
    clean, corrupted = payload["clean"], payload["corrupted"]
    valid = (
        payload["schema_version"] == 2 and payload["status"] == "complete"
        and isinstance(protocol, dict)
        and protocol.get("mapping_version") == "v7_super5_sjr_rgq_review_20260528"
        and protocol.get("mapping_hash") == "555ec85d5b51"
        and protocol.get("class_order") == ["CD", "HYP", "MI", "NORM", "STTC"]
        and isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("path"), str) and bool(checkpoint["path"])
        and isinstance(checkpoint.get("sha256"), str)
        and len(checkpoint["sha256"]) == 64
        and isinstance(clean, dict) and isinstance(clean.get("per_center"), dict)
        and bool(clean["per_center"])
        and isinstance(corrupted, dict)
        and isinstance(corrupted.get("per_view"), list)
        and len(corrupted["per_view"]) == 20
        and all(isinstance(view, dict) for view in corrupted["per_view"])
        and isinstance(corrupted.get("aggregates"), dict)
    )
    if not valid:
        raise ValueError("evaluation_result must be complete schema_version=2")
    return {
        "identity": _pick(payload, ("model", "checkpoint", "protocol")),
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
        **_result_summary(payload, result_type),
    }


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
        output.mkdir(parents=True, exist_ok=False)
        started = utc_now()
        snapshots = snapshot_yaml_files(
            output, config_sources, config_root=config_root.resolve()
        )
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
                result_evidence = _expected_result_evidence(
                    self.run_dir, Path(expected["path"]), str(expected["type"])
                )
            except ValueError as exc:
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
    "build_run_file_index",
    "capture_environment",
    "capture_git_state",
    "ensure_output_outside_worktree",
    "sha256_file",
    "snapshot_yaml_files",
    "utc_now",
    "verify_run_file_index",
]
