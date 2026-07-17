"""Minimal integrity-tracked run records for the manual-refactor launcher."""

from __future__ import annotations

import hashlib
import json
import math
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
        cwd: str | Path = PROJECT_ROOT,
    ) -> "RunRecorder":
        output = ensure_output_outside_worktree(run_dir)
        if output.exists():
            raise FileExistsError(f"run directory already exists: {output}")
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

    def _write_delegate_evidence(self) -> None:
        """Extract only evidence actually emitted by the delegated trainer."""

        result_path = self.run_dir / "training" / "train_result.json"
        evaluation_candidates = tuple(self.run_dir.glob("*/evaluation_result.json"))
        evaluation_path = (
            evaluation_candidates[0] if len(evaluation_candidates) == 1 else None
        )
        direct_selection_candidates = tuple(
            self.run_dir.glob("*/direct_baseline_selection.json")
        )
        direct_selection_path = (
            direct_selection_candidates[0]
            if len(direct_selection_candidates) == 1
            else None
        )
        if (
            not result_path.is_file()
            and evaluation_path is None
            and direct_selection_path is None
        ):
            data_manifest = {
                "schema_version": 1,
                "status": "delegate_did_not_emit_train_result",
                "source": None,
                "config_snapshots": self.manifest["config_snapshots"],
            }
            selection = {
                "schema_version": 1,
                "status": "delegate_did_not_emit_train_result",
                "policy": None,
                "selected_checkpoint": None,
            }
        elif result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                data_manifest = {
                    "schema_version": 1,
                    "status": "invalid_delegate_train_result",
                    "source": {
                        "path": "training/train_result.json",
                        "sha256": sha256_file(result_path),
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
                selection = {
                    "schema_version": 1,
                    "status": "invalid_delegate_train_result",
                    "policy": None,
                    "selected_checkpoint": None,
                }
            else:
                config = result.get("config")
                seed = result.get("seed")
                data_manifest = {
                    "schema_version": 1,
                    "status": "recorded_from_delegate_train_result",
                    "source": {
                        "path": "training/train_result.json",
                        "sha256": sha256_file(result_path),
                    },
                    "config": config,
                    "seed": seed,
                    "model": result.get("model"),
                    "latent_pool": result.get("latent_pool"),
                    "center": result.get("center"),
                    "method_id": result.get("method_id"),
                    "scientific_arm": result.get("scientific_arm"),
                }
                selected_checkpoint = result.get("selected_checkpoint")
                if selected_checkpoint is None:
                    selected_checkpoint = result.get("last_checkpoint")
                selected_epoch = result.get("selected_epoch")
                policy = "validation_metric"
                heldout_used = False
                explicit_selection = result.get("selection")
                if result.get("method_id") is not None:
                    if not isinstance(explicit_selection, dict):
                        raise ValueError(
                            "typed method train_result must record selection"
                        )
                    policy = explicit_selection.get("policy")
                    selected_epoch = explicit_selection.get("selected_epoch")
                    selected_checkpoint = explicit_selection.get(
                        "selected_checkpoint"
                    )
                    heldout_used = explicit_selection.get(
                        "heldout_evaluation_used_for_selection"
                    )
                    if (
                        policy != "last"
                        or selected_epoch != result.get("epochs_completed")
                        or not isinstance(selected_checkpoint, dict)
                        or heldout_used is not False
                    ):
                        raise ValueError(
                            "typed method selection must be last, match the final "
                            "epoch, include its checkpoint, and avoid heldout feedback"
                        )
                elif isinstance(explicit_selection, dict):
                    policy = explicit_selection.get("policy")
                    selected_epoch = explicit_selection.get("selected_epoch")
                    selected_checkpoint = explicit_selection.get(
                        "selected_checkpoint"
                    )
                    heldout_used = explicit_selection.get(
                        "heldout_evaluation_used_for_selection"
                    )
                    if heldout_used is not False:
                        raise ValueError(
                            "supervised selection must explicitly avoid heldout feedback"
                        )
                selection = {
                    "schema_version": 1,
                    "status": "recorded_from_delegate_train_result",
                    "policy": policy,
                    "selected_epoch": selected_epoch,
                    "selected_metric": result.get("selected_metric"),
                    "selected_checkpoint": selected_checkpoint,
                    "heldout_evaluation_used_for_selection": heldout_used,
                }
        elif direct_selection_path is not None:
            try:
                result = json.loads(
                    direct_selection_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as exc:
                data_manifest = {
                    "schema_version": 1,
                    "status": "invalid_direct_baseline_selection",
                    "source": {
                        "path": direct_selection_path.relative_to(
                            self.run_dir
                        ).as_posix(),
                        "sha256": sha256_file(direct_selection_path),
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
                selection = {
                    "schema_version": 1,
                    "status": "invalid_direct_baseline_selection",
                    "policy": None,
                    "selected_checkpoint": None,
                }
            else:
                rule = result.get("selection_rule")
                comparison = result.get("comparison_identity")
                selection_status = result.get("status")
                if (
                    result.get("artifact_type")
                    != "direct_k500_pooled_epoch_selection"
                    or selection_status not in {"selected", "failed_clean_floor"}
                    or not isinstance(rule, dict)
                    or not isinstance(comparison, dict)
                    or rule.get("heldout_evaluation_used") is not False
                    or rule.get("robust_aggregation")
                    != "mean_of_20_composition_macro_auprc"
                    or rule.get("score")
                    != "0.5_clean_macro_auprc_plus_0.5_robust_macro_auprc"
                    or comparison.get("protocol_id")
                    != "pn2021_direct_family_balanced_tuning"
                    or comparison.get("method_id") != "direct_depth23_fixed20"
                    or result.get("record_count") != 400
                    or result.get("composition_count") != 20
                ):
                    raise ValueError("invalid Direct pooled-selection evidence")
                selected_epoch = result.get("selected_epoch")
                selected_score = result.get("selected_score")
                selected_clean = result.get("selected_clean_macro_auprc")
                selected_robust = result.get("selected_robust_macro_auprc")
                if selection_status == "selected":
                    numeric = (selected_score, selected_clean, selected_robust)
                    if (
                        isinstance(selected_epoch, bool)
                        or not isinstance(selected_epoch, int)
                        or selected_epoch <= 0
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in numeric
                        )
                    ):
                        raise ValueError("selected Direct evidence is incomplete")
                elif any(
                    value is not None
                    for value in (
                        selected_epoch,
                        selected_score,
                        selected_clean,
                        selected_robust,
                    )
                ):
                    raise ValueError("failed clean-floor evidence must not select an epoch")
                source = {
                    "path": direct_selection_path.relative_to(
                        self.run_dir
                    ).as_posix(),
                    "sha256": sha256_file(direct_selection_path),
                }
                data_manifest = {
                    "schema_version": 1,
                    "status": "recorded_from_direct_pooled_selection",
                    "source": source,
                    "config": result.get("config"),
                    "comparison_identity": comparison,
                    "per_center_validation_identity": result.get(
                        "per_center_validation_identity"
                    ),
                    "record_count": result.get("record_count"),
                    "composition_count": result.get("composition_count"),
                    "composition_ids": result.get("composition_ids"),
                    "frozen_corruption_identity_sha256": result.get(
                        "frozen_corruption_identity_sha256"
                    ),
                    "centers": result.get("centers"),
                }
                selection = {
                    "schema_version": 1,
                    "status": (
                        "recorded_from_direct_pooled_selection"
                        if selection_status == "selected"
                        else "recorded_failed_clean_floor"
                    ),
                    "policy": "four_center_pooled_clean_robust_family_balanced",
                    "selected_epoch": selected_epoch,
                    "selected_metric": selected_score,
                    "selected_score": selected_score,
                    "selected_clean_macro_auprc": selected_clean,
                    "selected_robust_macro_auprc": selected_robust,
                    "clean_floor": result.get("clean_floor"),
                    "selected_checkpoint": None,
                    "heldout_evaluation_used_for_selection": False,
                    "source": source,
                }
        else:
            assert evaluation_path is not None
            try:
                result = json.loads(evaluation_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                data_manifest = {
                    "schema_version": 1,
                    "status": "invalid_delegate_evaluation_result",
                    "source": {
                        "path": evaluation_path.relative_to(self.run_dir).as_posix(),
                        "sha256": sha256_file(evaluation_path),
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                }
                selection = {
                    "schema_version": 1,
                    "status": "invalid_delegate_evaluation_result",
                    "policy": None,
                    "selected_checkpoint": None,
                }
            else:
                data_manifest = {
                    "schema_version": 1,
                    "status": "recorded_from_delegate_evaluation_result",
                    "source": {
                        "path": evaluation_path.relative_to(self.run_dir).as_posix(),
                        "sha256": sha256_file(evaluation_path),
                    },
                    "config": result.get("config"),
                    "seed": result.get("seed"),
                    "model": result.get("model"),
                    "protocol": result.get("protocol"),
                    "clean_record_identity": {
                        center: payload.get("identity")
                        for center, payload in result.get("clean", {})
                        .get("per_center", {})
                        .items()
                    },
                }
                selection = {
                    "schema_version": 1,
                    "status": "external_evaluation_only",
                    "policy": "checkpoint_fixed_before_pn2021_evaluation",
                    "selected_checkpoint": result.get("checkpoint"),
                    "heldout_evaluation_used_for_selection": False,
                }
        _write_json_atomic(self.run_dir / "data_manifest.json", data_manifest)
        _write_json_atomic(self.run_dir / "selection.json", selection)

    def finalize(self, *, exit_code: int, error: str | None = None) -> None:
        finished = utc_now()
        status = "complete" if int(exit_code) == 0 and error is None else "failed"
        duration = max(0.0, time.monotonic() - self._started_monotonic)
        self.manifest.update(
            {
                "status": status,
                "finished_at_utc": finished,
                "duration_seconds": duration,
                "exit_code": int(exit_code),
                "error": error,
            }
        )
        self.card.update(
            {
                "status": status,
                "finished_at_utc": finished,
                "exit_code": int(exit_code),
                "replay_status": (
                    "recorded_not_yet_audited" if status == "complete" else "failed"
                ),
            }
        )
        self._write_delegate_evidence()
        self._write_live_files()
        index = build_run_file_index(self.run_dir)
        index_path = self.run_dir / "run_file_index.json"
        _write_json_atomic(index_path, index)
        self.manifest["run_file_index_sha256"] = sha256_file(index_path)
        _write_json_atomic(self.run_dir / "run_manifest.json", self.manifest)


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
