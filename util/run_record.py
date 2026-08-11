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
RESULT_REQUIRED_KEYS = {
    "train_result": frozenset({"config", "epochs_completed", "model"}),
    "evaluation_result": frozenset(
        {"schema_version", "checkpoint", "clean", "corrupted", "protocol", "status"}
    ),
    "pooled_selection": frozenset({"schema_version", "artifact_type", "selection_rule", "status"}),
    "refit_contract": frozenset({"schema_version", "artifact_type", "selection", "result"}),
}
POOLED_VALUE_KEYS = (
    "selected_score",
    "selected_clean_macro_auprc",
    "selected_clean_macro_auroc",
    "selected_robust_macro_auprc",
    "selected_robust_macro_auroc",
)


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


def validate_managed_run_member(path: str | Path) -> dict[str, Any]:
    """Validate one immutable member of a complete indexed managed run."""

    member = Path(path).expanduser().resolve()
    if not member.is_file() or member.is_symlink():
        raise FileNotFoundError(f"managed run member not found: {member}")
    run_dir = next(
        (
            parent
            for parent in member.parents
            if (parent / "run_manifest.json").is_file()
            and (parent / "run_file_index.json").is_file()
            and (parent / "run_card.json").is_file()
        ),
        None,
    )
    if run_dir is None:
        raise ValueError("artifact must belong to a finalized managed run")
    errors = verify_run_file_index(run_dir)
    if errors:
        raise ValueError("managed run integrity check failed: " + "; ".join(errors))
    card = json.loads((run_dir / "run_card.json").read_text(encoding="utf-8"))
    if card.get("status") != "complete" or int(card.get("exit_code", -1)) != 0:
        raise ValueError("managed run is not complete with exit_code=0")
    index = json.loads((run_dir / "run_file_index.json").read_text(encoding="utf-8"))
    relative = member.relative_to(run_dir).as_posix()
    matches = [
        item
        for item in index.get("files", ())
        if isinstance(item, dict) and item.get("path") == relative
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(member):
        raise ValueError("artifact SHA256 differs from its managed run index")
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "run_dir": run_dir,
        "relative_path": relative,
        "manifest": manifest,
        "member_sha256": matches[0]["sha256"],
    }


def validate_run_config_snapshots(
    run_dir: str | Path,
    sources: Sequence[str | Path],
    *,
    config_root: str | Path,
) -> list[dict[str, Any]]:
    """Match current YAML bytes to the immutable snapshots of a managed run."""

    root = Path(run_dir).expanduser().resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    snapshots = manifest.get("config_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("managed run manifest has no config_snapshots list")
    by_destination: dict[str, list[dict[str, Any]]] = {}
    for item in snapshots:
        if isinstance(item, dict) and isinstance(item.get("snapshot_path"), str):
            by_destination.setdefault(item["snapshot_path"], []).append(item)
    evidence: list[dict[str, Any]] = []
    config_root_path = Path(config_root).expanduser().resolve()
    for raw_source in sources:
        source = Path(raw_source).expanduser().resolve()
        destination = _snapshot_destination(source, config_root=config_root_path)
        matches = by_destination.get(destination.as_posix(), [])
        if len(matches) != 1:
            raise ValueError(
                f"managed selection has no unique config snapshot for {source}"
            )
        expected = matches[0]
        current_sha256 = sha256_file(source)
        snapshot_path = (root / destination).resolve()
        try:
            snapshot_path.relative_to(root)
        except ValueError:
            raise ValueError("managed config snapshot escapes run directory") from None
        if not snapshot_path.is_file() or snapshot_path.is_symlink():
            raise ValueError(f"managed config snapshot is missing: {destination}")
        snapshot_sha256 = sha256_file(snapshot_path)
        if expected.get("sha256") != snapshot_sha256:
            raise ValueError(f"managed config snapshot SHA256 mismatch: {destination}")
        if current_sha256 != snapshot_sha256:
            raise ValueError(
                f"current config differs from pooled tuning snapshot: {source}"
            )
        evidence.append(
            {
                "path": str(source),
                "snapshot_path": destination.as_posix(),
                "sha256": current_sha256,
            }
        )
    return evidence


def resolve_run_config_snapshot_by_sha256(
    run_dir: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Resolve one immutable YAML snapshot by its recorded content digest."""

    root = Path(run_dir).expanduser().resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected config snapshot SHA256 must be a 64-character string")
    manifest_path = root / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"managed run manifest not found: {manifest_path}"
        ) from None
    snapshots = manifest.get("config_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("managed run manifest has no config_snapshots list")
    matches = [
        item
        for item in snapshots
        if isinstance(item, dict) and item.get("sha256") == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            "managed run must contain exactly one config snapshot with the "
            "selection online-training SHA256"
        )
    relative = Path(str(matches[0].get("snapshot_path", "")))
    if (
        not str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise ValueError("managed config snapshot path is unsafe or not YAML")
    snapshot = (root / relative).resolve()
    try:
        snapshot.relative_to(root)
    except ValueError:
        raise ValueError("managed config snapshot escapes run directory") from None
    if not snapshot.is_file() or snapshot.is_symlink():
        raise ValueError(f"managed config snapshot is missing: {relative}")
    if sha256_file(snapshot) != expected_sha256:
        raise ValueError("managed config snapshot SHA256 mismatch")
    return {
        "path": snapshot,
        "snapshot_path": relative.as_posix(),
        "sha256": expected_sha256,
    }


def _pick(payload: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys}


def _require_result_keys(payload: dict[str, Any], result_type: str) -> None:
    required = RESULT_REQUIRED_KEYS.get(result_type)
    if required is None or not required.issubset(payload):
        raise ValueError(f"expected {result_type} keys are missing")


def _train_result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    _require_result_keys(payload, "train_result")
    epochs, method_id, selection = (
        payload["epochs_completed"], payload.get("method_id"), payload.get("selection")
    )
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("train_result epochs_completed must be positive")
    if method_id is not None:
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
    else:
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


def _pooled_selection_summary(payload: dict[str, Any], exit_code: int) -> dict[str, Any]:
    from util.evaluation.direct_baseline_selection import locked_pooled_selection_rule

    _require_result_keys(payload, "pooled_selection")
    comparison, config = payload.get("comparison_identity"), payload.get("config")
    identity_ok = (
        payload["schema_version"] == 1
        and payload["artifact_type"] == "direct_k500_pooled_epoch_selection"
        and payload["selection_rule"] == locked_pooled_selection_rule()
        and isinstance(comparison, dict) and isinstance(config, dict)
        and comparison.get("method_id") == "direct_depth23_fixed20"
        and comparison.get("protocol_id") == config.get("protocol_id")
        and payload.get("record_count") == 400 and payload.get("composition_count") == 20
        and payload.get("centers") == ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
        and payload.get("class_order") == ["CD", "HYP", "MI", "NORM", "STTC"]
    )
    if not identity_ok:
        raise ValueError("invalid Direct pooled-selection identity")
    status, epoch = payload["status"], payload.get("selected_epoch")
    numbers = [payload.get(key) for key in POOLED_VALUE_KEYS]
    selected = (
        status == "selected" and exit_code == 0
        and isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in numbers
        )
    )
    failed = (
        status == "failed_clean_floor"
        and exit_code == 3
        and epoch is None
        and all(value is None for value in numbers)
    )
    if not (selected or failed):
        raise ValueError("pooled-selection status or selected values are inconsistent")
    return {
        "identity": _pick(comparison, ("protocol_id", "method_id", "model_family", "replicate_id")),
        "selection": {
            "status": status,
            "selected_epoch": epoch,
            **_pick(payload, POOLED_VALUE_KEYS),
        },
    }


def _result_summary(payload: dict[str, Any], result_type: str, exit_code: int) -> dict[str, Any]:
    if result_type == "train_result":
        return _train_result_summary(payload)
    if result_type == "pooled_selection":
        return _pooled_selection_summary(payload, exit_code)
    _require_result_keys(payload, result_type)
    if result_type == "evaluation_result":
        protocol = payload["protocol"]
        valid = (
            payload["schema_version"] == 2 and payload["status"] == "complete"
            and isinstance(protocol, dict)
            and protocol.get("mapping_version") == "v7_super5_sjr_rgq_review_20260528"
            and protocol.get("mapping_hash") == "555ec85d5b51"
            and protocol.get("class_order") == ["CD", "HYP", "MI", "NORM", "STTC"]
        )
        if not valid:
            raise ValueError("evaluation_result must be complete schema_version=2")
        return {
            "identity": _pick(payload, ("model", "checkpoint", "protocol")),
            "selection": {"policy": "checkpoint_fixed_before_evaluation"},
        }
    selection, result = payload["selection"], payload["result"]
    valid = (
        result_type == "refit_contract" and payload["schema_version"] == 1
        and payload["artifact_type"] == "pn2021_direct_refit_contract"
        and isinstance(selection, dict) and selection.get("heldout_evaluation_used") is False
        and isinstance(result, dict) and bool(result.get("method_id"))
    )
    if not valid:
        raise ValueError("invalid heldout-free Direct refit contract")
    nested = _train_result_summary(result)
    return {
        "identity": {**_pick(payload, ("model", "center", "method")), "result": nested["identity"]},
        "selection": _pick(
            selection,
            (
                "selected_epoch",
                "selected_score",
                "selected_clean_macro_auprc",
                "selected_robust_macro_auprc",
                "heldout_evaluation_used",
            ),
        ),
    }


def _expected_result_evidence(
    run_dir: Path, relative_path: Path, result_type: str, exit_code: int
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
        **_result_summary(payload, result_type, exit_code),
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
        validate_result = delegate_exit_code == 0 or (
            expected["type"] == "pooled_selection" and delegate_exit_code == 3
        )
        if validate_result and effective_error is None:
            try:
                result_evidence = _expected_result_evidence(
                    self.run_dir, Path(expected["path"]), str(expected["type"]), delegate_exit_code
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
    "resolve_run_config_snapshot_by_sha256",
    "sha256_file",
    "snapshot_yaml_files",
    "utc_now",
    "validate_managed_run_member",
    "validate_run_config_snapshots",
    "verify_run_file_index",
]
