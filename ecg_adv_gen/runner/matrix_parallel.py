"""Bounded single-GPU queue for YAML-managed matrix commands."""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ecg_adv_gen.config.launch import LaunchError, verify_required_artifacts
from ecg_adv_gen.runner.process import build_process_env


_PROGRESS_STATUSES = frozenset({"pending", "running", "failed", "interrupted", "succeeded"})
_COMMAND_STATUSES = _PROGRESS_STATUSES
_ATTEMPT_STATUSES = frozenset({"running", "stale", "failed", "interrupted", "succeeded"})
_RESUME_VOLATILE_KEYS = frozenset(
    {"checked_at_utc", "created_at_utc", "exists", "updated_at_utc"}
)


@dataclass(frozen=True)
class AssignedCommand:
    command_index: int
    command: Mapping[str, Any]
    gpu: str
    matrix_key: str
    matrix_value: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_list(raw: str) -> list[str]:
    raw_gpus = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not raw_gpus:
        raise ValueError("at least one GPU id is required")
    gpus: list[str] = []
    for raw_gpu in raw_gpus:
        if not raw_gpu.isdigit():
            raise ValueError(f"GPU ids must be numeric physical indices: {raw_gpu!r}")
        gpu = str(int(raw_gpu))
        if gpu in gpus:
            raise ValueError(f"duplicate GPU id in --gpus: {gpu}")
        gpus.append(gpu)
    return gpus


def validate_parallelism(max_parallel: int, gpus: Sequence[str]) -> int:
    value = int(max_parallel)
    if value < 1:
        raise ValueError("--max-parallel must be at least 1")
    if value > len(gpus):
        raise ValueError(f"--max-parallel={value} exceeds selected GPU count {len(gpus)}")
    return value


def validate_gpu_snapshot(
    gpus: Sequence[str],
    snapshot: Mapping[str, Any],
    *,
    min_free_memory_mb: int,
    max_utilization_pct: int,
) -> list[dict[str, Any]]:
    rows = {str(row.get("index")): dict(row) for row in snapshot.get("gpus") or []}
    selected: list[dict[str, Any]] = []
    for gpu in gpus:
        if gpu not in rows:
            raise LaunchError(f"selected GPU {gpu} is not present in nvidia-smi output")
        row = rows[gpu]
        try:
            used = int(row["memory_used_mb"])
            total = int(row["memory_total_mb"])
            utilization = int(row["utilization_gpu_pct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LaunchError(f"nvidia-smi row for GPU {gpu} is not numeric: {row}") from exc
        free = total - used
        if free < min_free_memory_mb:
            raise LaunchError(
                f"GPU {gpu} free memory {free} MiB is below required {min_free_memory_mb} MiB"
            )
        if utilization > max_utilization_pct:
            raise LaunchError(
                f"GPU {gpu} utilization {utilization}% exceeds allowed {max_utilization_pct}%"
            )
        selected.append({**row, "free_memory_mb": free})
    return selected


def assign_gpus_to_commands(
    commands: Sequence[Mapping[str, Any]],
    gpus: Sequence[str],
    *,
    matrix_key: str = "center",
) -> list[AssignedCommand]:
    """Build deterministic round-robin preview assignments; execution remains queued."""

    if not gpus:
        raise ValueError("at least one GPU is required")
    assigned: list[AssignedCommand] = []
    for index, command in enumerate(commands):
        value = str((command.get("matrix") or {}).get(matrix_key) or "")
        if not value:
            raise ValueError(f"command {index} is missing matrix.{matrix_key}")
        assigned.append(
            AssignedCommand(index, command, str(gpus[index % len(gpus)]), matrix_key, value)
        )
    return assigned


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchError(f"required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"JSON root must be an object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_resume_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_resume_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in _RESUME_VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_resume_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_expected_outputs(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_expected_outputs(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in {"exists", "size_bytes"}
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_expected_outputs(item) for item in value]
    return value


def build_matrix_resume_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized immutable execution surface used for resume."""

    artifact_trace = copy.deepcopy(dict(manifest.get("artifact_trace") or {}))
    if "expected_outputs" in artifact_trace:
        artifact_trace["expected_outputs"] = _normalize_expected_outputs(
            artifact_trace["expected_outputs"]
        )
    return _normalize_resume_value(
        {
            "schema_version": 1,
            "run_id": manifest.get("run_id"),
            "config_hash_sha256": manifest.get("config_hash_sha256"),
            "entry_config": manifest.get("entry_config"),
            "commands": manifest.get("commands") or [],
            "postprocess_commands": manifest.get("postprocess_commands") or [],
            "artifact_trace": artifact_trace,
        }
    )


def _resume_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_matrix_resume_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(manifest))
    contract = build_matrix_resume_contract(out)
    matrix = dict(out.get("matrix_parallel") or {})
    matrix["resume_contract"] = {
        "schema_version": 1,
        "sha256": _resume_contract_sha256(contract),
        "contract": contract,
    }
    out["matrix_parallel"] = matrix
    return out


def validate_matrix_resume_contract(
    existing: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    record = (existing.get("matrix_parallel") or {}).get("resume_contract")
    if not isinstance(record, Mapping) or record.get("schema_version") != 1:
        raise LaunchError("matrix resume contract is missing or has an unsupported schema")
    stored = record.get("contract")
    if not isinstance(stored, Mapping):
        raise LaunchError("matrix resume contract payload is missing")
    observed = build_matrix_resume_contract(existing)
    stored_normalized = _normalize_resume_value(stored)
    stored_sha = _resume_contract_sha256(stored_normalized)
    if record.get("sha256") != stored_sha or stored_normalized != observed:
        raise LaunchError("matrix resume contract does not match the persisted execution surface")
    expected_contract = build_matrix_resume_contract(expected)
    if stored_sha != _resume_contract_sha256(expected_contract) or stored_normalized != expected_contract:
        raise LaunchError("matrix resume contract drifted from the current execution surface")


def _patch_manifest(path: Path, **patch: Any) -> dict[str, Any]:
    manifest = read_json_object(path)
    manifest.update(patch)
    manifest["updated_at_utc"] = _utc_now()
    atomic_write_json(path, manifest)
    return manifest


@contextmanager
def execution_lock(run_dir: Path) -> Iterator[Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    path = run_dir.parent / f".{run_dir.name}.matrix_execute.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LaunchError(f"matrix output directory is already locked: {run_dir}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at_utc={_utc_now()}\n")
        handle.flush()
        try:
            yield path
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_command_artifacts(manifest: Mapping[str, Any], command_index: int) -> dict[str, Any]:
    filtered = copy.deepcopy(dict(manifest))
    expected = filtered.setdefault("artifact_trace", {}).setdefault("expected_outputs", {})
    expected["launch_artifacts"] = []
    expected["postprocess_runs"] = []
    expected["child_runs"] = [
        child
        for child in expected.get("child_runs") or []
        if int(child.get("command_index", -1)) == command_index
    ]
    return verify_required_artifacts(filtered, include_postprocess=False)


def _new_progress(commands: Sequence[Mapping[str, Any]], run_id: Any) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "pending",
        "created_at_utc": now,
        "updated_at_utc": now,
        "commands": [
            {
                "command_index": index,
                "name": command.get("name"),
                "matrix": dict(command.get("matrix") or {}),
                "status": "pending",
                "attempts": [],
            }
            for index, command in enumerate(commands)
        ],
    }


def _flatten_attempts(progress: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "executor": "matrix_parallel",
            "command_index": state["command_index"],
            "name": state.get("name"),
            "matrix": state.get("matrix"),
            **attempt,
        }
        for state in progress.get("commands") or []
        for attempt in state.get("attempts") or []
    ]


@dataclass
class _StateStore:
    progress: dict[str, Any]
    progress_path: Path
    manifest_path: Path
    gpus: list[str]
    max_parallel: int

    def save(self, status: str | None = None) -> None:
        if status:
            self.progress["status"] = status
        self.progress["updated_at_utc"] = _utc_now()
        atomic_write_json(self.progress_path, self.progress)
        manifest = read_json_object(self.manifest_path)
        matrix = dict(manifest.get("matrix_parallel") or {})
        matrix.update(
            {
                "enabled": True,
                "executor": "bounded_queue",
                "gpus": self.gpus,
                "max_parallel": self.max_parallel,
                "progress_path": str(self.progress_path),
                "status": self.progress["status"],
                "command_states": self.progress["commands"],
            }
        )
        old = [
            item
            for item in manifest.get("command_runs") or []
            if item.get("executor") != "matrix_parallel"
        ]
        manifest.update(
            {
                "matrix_parallel": matrix,
                "command_runs": [*old, *_flatten_attempts(self.progress)],
                "updated_at_utc": self.progress["updated_at_utc"],
            }
        )
        atomic_write_json(self.manifest_path, manifest)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_attempt(attempt: Any, *, command_index: int, position: int) -> None:
    prefix = f"matrix progress command {command_index} attempt {position}"
    if not isinstance(attempt, Mapping):
        raise LaunchError(f"{prefix} must be an object")
    if attempt.get("attempt") != position or not _positive_int(attempt.get("pid")):
        raise LaunchError(f"{prefix} has invalid attempt number or pid")
    if not isinstance(attempt.get("gpu"), str) or not str(attempt["gpu"]).isdigit():
        raise LaunchError(f"{prefix} has invalid gpu")
    status = attempt.get("status")
    if status not in _ATTEMPT_STATUSES:
        raise LaunchError(f"{prefix} has invalid status={status!r}")
    if not isinstance(attempt.get("started_at_utc"), str) or not attempt["started_at_utc"]:
        raise LaunchError(f"{prefix} has invalid started_at_utc")
    for key in ("stdout_log_path", "stderr_log_path"):
        if not isinstance(attempt.get(key), str) or not attempt[key]:
            raise LaunchError(f"{prefix} has invalid {key}")
    if attempt.get("resume") not in {None, "latest"}:
        raise LaunchError(f"{prefix} has invalid resume mode")
    finished, returncode = attempt.get("finished_at_utc"), attempt.get("returncode")
    if status == "running" and (finished is not None or returncode is not None):
        raise LaunchError(f"{prefix} running state must not be finished")
    if status in {"failed", "interrupted", "succeeded"} and (
        not isinstance(finished, str)
        or not finished
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
    ):
        raise LaunchError(f"{prefix} terminal state is incomplete")
    if status == "stale" and (not isinstance(finished, str) or not finished):
        raise LaunchError(f"{prefix} stale state needs finished_at_utc")
    if status == "stale" and (
        returncode is not None
        and (not isinstance(returncode, int) or isinstance(returncode, bool))
    ):
        raise LaunchError(f"{prefix} stale state has invalid returncode")


def _validate_progress(
    progress: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    *,
    run_id: Any,
) -> None:
    if progress.get("schema_version") != 1:
        raise LaunchError("matrix progress schema_version must be 1")
    if not isinstance(run_id, str) or not run_id or progress.get("run_id") != run_id:
        raise LaunchError("matrix progress run_id does not match the manifest")
    if progress.get("status") not in _PROGRESS_STATUSES:
        raise LaunchError(f"matrix progress has invalid status={progress.get('status')!r}")
    for key in ("created_at_utc", "updated_at_utc"):
        if not isinstance(progress.get(key), str) or not progress[key]:
            raise LaunchError(f"matrix progress has invalid {key}")
    states = progress.get("commands")
    if not isinstance(states, list) or len(states) != len(commands):
        raise LaunchError("matrix progress command count does not match the current config")
    for index, (state, command) in enumerate(zip(states, commands, strict=True)):
        if not isinstance(state, Mapping):
            raise LaunchError(f"matrix progress command {index} must be an object")
        if state.get("command_index") != index or state.get("name") != command.get("name"):
            raise LaunchError(f"matrix progress command {index} identity does not match")
        if not isinstance(state.get("matrix"), Mapping) or dict(state["matrix"]) != dict(
            command.get("matrix") or {}
        ):
            raise LaunchError(f"matrix progress command {index} matrix does not match")
        status = state.get("status")
        if status not in _COMMAND_STATUSES:
            raise LaunchError(f"matrix progress command {index} has invalid status={status!r}")
        attempts = state.get("attempts")
        if not isinstance(attempts, list):
            raise LaunchError(f"matrix progress command {index} attempts must be a list")
        for position, attempt in enumerate(attempts, start=1):
            _validate_attempt(attempt, command_index=index, position=position)
            if position < len(attempts) and attempt["status"] == "running":
                raise LaunchError(
                    f"matrix progress command {index} has a non-final running attempt"
                )
        if status == "running" and (not attempts or attempts[-1]["status"] != "running"):
            raise LaunchError(f"matrix progress command {index} running state has no live attempt")
        if status != "running" and attempts and attempts[-1]["status"] == "running":
            raise LaunchError(
                f"matrix progress command {index} hides a running attempt as {status}"
            )
        if status in {"failed", "interrupted", "succeeded"} and (
            not attempts or attempts[-1]["status"] != status
        ):
            raise LaunchError(f"matrix progress command {index} terminal state is inconsistent")
    if progress.get("status") == "succeeded" and any(
        state.get("status") != "succeeded" for state in states
    ):
        raise LaunchError("matrix progress succeeded state has incomplete commands")


def _expected_runs(manifest: Mapping[str, Any], bucket: str) -> list[Mapping[str, Any]]:
    return (
        ((manifest.get("artifact_trace") or {}).get("expected_outputs") or {}).get(bucket)
        or []
    )


def _prepare_artifact_dirs(manifest: Mapping[str, Any], bucket: str, command_index: int) -> None:
    for run in _expected_runs(manifest, bucket):
        if int(run.get("command_index", -1)) != command_index:
            continue
        for artifact in run.get("expected_artifacts") or []:
            if artifact.get("required") is not False and artifact.get("path"):
                Path(str(artifact["path"])).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _child_run_dir(manifest: Mapping[str, Any], command_index: int) -> Path | None:
    for child in _expected_runs(manifest, "child_runs"):
        if int(child.get("command_index", -1)) == command_index and child.get("child_run_dir"):
            return Path(str(child["child_run_dir"]))
    return None


def _resume_argv(
    command: Mapping[str, Any], manifest: Mapping[str, Any], command_index: int, resume: bool
) -> tuple[list[str], str | None]:
    argv = [str(item) for item in command.get("argv") or []]
    child_dir = _child_run_dir(manifest, command_index)
    latest = child_dir / "checkpoints" / "checkpoint_latest.pt" if child_dir else None
    if (
        resume
        and len(argv) > 1
        and Path(argv[1]).name == "effnet_vae_lhat_augmix.py"
        and latest is not None
        and latest.is_file()
        and "--resume" not in argv
    ):
        return [*argv, "--resume", "latest"], "latest"
    return argv, None


def _spawn(
    command: Mapping[str, Any],
    argv: Sequence[str],
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[Any]:
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        return subprocess.Popen(
            argv,
            cwd=str(command.get("cwd") or stdout_path.parent),
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _terminate_running(running: Mapping[int, dict[str, Any]], status: str) -> None:
    for item in running.values():
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for item in running.values():
        process = item["process"]
        _terminate_process(process)
        item["attempt"].update(
            {"status": status, "returncode": process.returncode, "finished_at_utc": _utc_now()}
        )
        item["state"]["status"] = status


def _require_stale_pid(pid: int, command_index: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise LaunchError(
            f"matrix progress command {command_index} pid={pid} cannot prove stale (EPERM)"
        ) from exc
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return
        raise LaunchError(
            f"matrix progress command {command_index} pid={pid} cannot prove stale: {exc}"
        ) from exc
    raise LaunchError(
        f"matrix progress command {command_index} pid={pid} is still alive; refusing resume"
    )


def _resume_states(progress: dict[str, Any], manifest: Mapping[str, Any], resume: bool) -> None:
    if not resume:
        return
    for state in progress["commands"]:
        index = int(state["command_index"])
        prior = state["status"]
        if prior == "succeeded":
            report = verify_command_artifacts(manifest, index)
            state["artifact_verification"] = report
            if report["passed"]:
                state["resume_action"] = "skipped_revalidated"
                continue
            state["resume_action"] = "retry_artifacts_invalid"
        elif prior == "running":
            attempt = state["attempts"][-1]
            _require_stale_pid(int(attempt["pid"]), index)
            attempt.update({"status": "stale", "finished_at_utc": _utc_now()})
            state["resume_action"] = "retry_stale"
        else:
            state["resume_action"] = f"retry_{prior}"
        state["status"] = "pending"


def run_matrix_queue(
    commands: Sequence[Mapping[str, Any]],
    *,
    gpus: Sequence[str],
    max_parallel: int,
    run_dir: Path,
    manifest_path: Path,
    resume: bool = False,
    base_env: Mapping[str, str] | None = None,
    poll_interval: float = 0.1,
    tick_hook: Callable[[], None] | None = None,
    acquire_lock: bool = True,
) -> dict[str, Any]:
    """Execute N commands over a bounded GPU pool with atomic per-command resume."""

    gpus = list(gpus)
    max_parallel = validate_parallelism(max_parallel, gpus)
    run_dir, manifest_path = Path(run_dir), Path(manifest_path)
    progress_path = run_dir / "matrix_progress.json"
    manifest = read_json_object(manifest_path)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    lock = execution_lock(run_dir) if acquire_lock else nullcontext()
    with lock:
        if progress_path.exists() and not resume:
            raise LaunchError(f"matrix progress already exists; use --resume: {progress_path}")
        progress = (
            read_json_object(progress_path)
            if progress_path.exists()
            else _new_progress(commands, manifest.get("run_id"))
        )
        _validate_progress(progress, commands, run_id=manifest.get("run_id"))
        _resume_states(progress, manifest, resume)
        store = _StateStore(progress, progress_path, manifest_path, gpus, max_parallel)
        store.save("running")
        pending = [s["command_index"] for s in progress["commands"] if s["status"] == "pending"]
        available, running = list(gpus), {}
        stop_dispatch = False
        first_failure: tuple[int, str] | None = None
        env_base = dict(base_env or os.environ)

        try:
            while pending or running:
                while not stop_dispatch and pending and available and len(running) < max_parallel:
                    index, gpu = int(pending.pop(0)), available.pop(0)
                    command, state = commands[index], progress["commands"][index]
                    number = len(state["attempts"]) + 1
                    argv, resume_mode = _resume_argv(command, manifest, index, resume)
                    _prepare_artifact_dirs(manifest, "child_runs", index)
                    stem = run_dir / "logs" / f"command_{index:03d}_attempt_{number:02d}"
                    stdout_path, stderr_path = Path(f"{stem}.stdout.log"), Path(f"{stem}.stderr.log")
                    env = build_process_env(
                        base=env_base,
                        updates={**(command.get("env") or {}), "CUDA_VISIBLE_DEVICES": gpu},
                    )
                    process = _spawn(command, argv, env, stdout_path, stderr_path)
                    attempt = {
                        "attempt": number,
                        "gpu": gpu,
                        "pid": process.pid,
                        "started_at_utc": _utc_now(),
                        "finished_at_utc": None,
                        "returncode": None,
                        "status": "running",
                        "stdout_log_path": str(stdout_path),
                        "stderr_log_path": str(stderr_path),
                        "resume": resume_mode,
                    }
                    state["status"] = "running"
                    state["attempts"].append(attempt)
                    running[index] = {
                        "process": process,
                        "gpu": gpu,
                        "state": state,
                        "attempt": attempt,
                    }
                    store.save()

                if tick_hook:
                    tick_hook()
                completed = [index for index, item in running.items() if item["process"].poll() is not None]
                for index in sorted(completed):
                    item = running.pop(index)
                    process, state, attempt = item["process"], item["state"], item["attempt"]
                    available.append(item["gpu"])
                    attempt.update({"returncode": process.returncode, "finished_at_utc": _utc_now()})
                    reason = f"return code {process.returncode}" if process.returncode else None
                    if reason is None:
                        report = verify_command_artifacts(read_json_object(manifest_path), index)
                        state["artifact_verification"] = report
                        if not report["passed"]:
                            reason = "required artifact verification failed"
                    state["status"] = attempt["status"] = "failed" if reason else "succeeded"
                    if reason:
                        first_failure = first_failure or (index, reason)
                        stop_dispatch = True
                    store.save()
                if running and not completed:
                    time.sleep(max(0.001, poll_interval))
                if stop_dispatch and not running:
                    break
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _terminate_running(running, status)
            store.save(status)
            raise

        if first_failure:
            store.save("failed")
            raise LaunchError(f"matrix command {first_failure[0]} failed: {first_failure[1]}")
        if not all(state["status"] == "succeeded" for state in progress["commands"]):
            store.save("failed")
            raise LaunchError("matrix queue finished without verified success for every command")
        store.save("succeeded")
        return progress


def run_postprocess_serial(
    commands: Sequence[Mapping[str, Any]],
    *,
    run_dir: Path,
    manifest_path: Path,
    base_env: Mapping[str, str] | None = None,
    acquire_lock: bool = True,
) -> dict[str, Any]:
    """Run reporting serially, then perform the full standard artifact check."""

    run_dir, manifest_path = Path(run_dir), Path(manifest_path)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock = execution_lock(run_dir) if acquire_lock else nullcontext()
    with lock:
        manifest = _patch_manifest(manifest_path, postprocess_status="running")
        records = list(manifest.get("postprocess_runs") or [])
        env_base = dict(base_env or os.environ)
        for index, command in enumerate(commands):
            _prepare_artifact_dirs(manifest, "postprocess_runs", index)
            stdout_path = logs_dir / f"postprocess_{index:03d}.stdout.log"
            stderr_path = logs_dir / f"postprocess_{index:03d}.stderr.log"
            env = build_process_env(base=env_base, updates=command.get("env") or {})
            started = _utc_now()
            process = _spawn(command, [str(x) for x in command.get("argv") or []], env, stdout_path, stderr_path)
            try:
                returncode = process.wait()
            except BaseException as exc:
                _terminate_process(process)
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                _patch_manifest(manifest_path, postprocess_status=status)
                raise
            record = {
                "command_index": index,
                "name": command.get("name"),
                "pid": process.pid,
                "started_at_utc": started,
                "finished_at_utc": _utc_now(),
                "returncode": returncode,
                "status": "succeeded" if returncode == 0 else "failed",
                "stdout_log_path": str(stdout_path),
                "stderr_log_path": str(stderr_path),
            }
            records.append(record)
            _patch_manifest(
                manifest_path,
                postprocess_status="running" if returncode == 0 else "failed",
                postprocess_runs=records,
            )
            if returncode:
                raise LaunchError(f"postprocess command {index} failed with return code {returncode}")

        verification = verify_required_artifacts(read_json_object(manifest_path), include_postprocess=True)
        _patch_manifest(
            manifest_path,
            postprocess_status="succeeded" if verification["passed"] else "failed",
            artifact_verification=verification,
        )
        if not verification["passed"]:
            raise LaunchError("Required artifact verification failed after postprocess commands")
        return verification
