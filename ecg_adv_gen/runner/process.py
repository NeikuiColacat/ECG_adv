"""Small process helpers shared by legacy runner scripts."""

from __future__ import annotations

import os
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PathValue = str | os.PathLike[str]


@dataclass(frozen=True)
class StreamRunResult:
    """Result metadata for a streamed command."""

    argv: tuple[str, ...]
    log_path: Path
    returncode: int
    dry_run: bool
    rendered: str


def _stringify_env(env: Mapping[str, PathValue] | None) -> dict[str, str]:
    if not env:
        return {}
    return {str(key): str(value) for key, value in env.items()}


def render_command(argv: list[PathValue] | tuple[PathValue, ...], *, env: Mapping[str, PathValue] | None = None) -> str:
    """Render a shell-readable command line for logs and dry-run plans."""

    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in sorted(_stringify_env(env).items())
    )
    rendered = " ".join(shlex.quote(str(part)) for part in argv)
    return f"{env_prefix} {rendered}".strip()


def build_process_env(
    *,
    base: Mapping[str, PathValue] | None = None,
    updates: Mapping[str, PathValue] | None = None,
    ensure_unbuffered: bool = True,
    mkdir_keys: tuple[str, ...] = ("TMPDIR", "XDG_CACHE_HOME"),
) -> dict[str, str]:
    """Merge process env values and create common runtime directories."""

    env = dict(os.environ if base is None else _stringify_env(base))
    env.update(_stringify_env(updates))
    if ensure_unbuffered:
        env.setdefault("PYTHONUNBUFFERED", "1")
    for key in mkdir_keys:
        value = env.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
    return env


def _transient_stream_log_path(log_path: Path, env: Mapping[str, str]) -> tuple[Path, Path | None]:
    root = env.get("ECG_ADV_GEN_STREAM_LOG_ROOT") or os.environ.get("ECG_ADV_GEN_STREAM_LOG_ROOT")
    if not root:
        return log_path, None
    digest = hashlib.sha1(str(log_path).encode("utf-8")).hexdigest()[:12]
    suffix = log_path.suffix or ".log"
    return Path(root) / f"{log_path.stem}.{digest}{suffix}", log_path


def run_stream(
    argv: list[PathValue] | tuple[PathValue, ...],
    *,
    log_path: Path,
    env: Mapping[str, PathValue] | None = None,
    cwd: Path | None = None,
    dry_run: bool = False,
    append: bool = False,
) -> StreamRunResult:
    """Run a command with stdout/stderr streamed to terminal and a log file.

    ``env`` is treated like ``subprocess.Popen(env=...)``: when provided, it is
    the child process environment, not an overlay on the parent environment.
    Dry-run logs intentionally render only argv so full runtime environments do
    not leak into durable artifacts.
    """

    cmd = [str(part) for part in argv]
    env_for_render = _stringify_env(env)
    rendered = render_command(cmd)
    log_target, pointer_path = _transient_stream_log_path(log_path, env_for_render)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(shlex.quote(part) for part in cmd)}", flush=True)

    mode = "a" if append else "w"
    if dry_run:
        with log_path.open(mode, encoding="utf-8") as log:
            log.write(rendered + "\n")
        return StreamRunResult(tuple(cmd), log_path, 0, True, rendered)

    proc_env = build_process_env(
        base={} if env is not None else os.environ,
        updates=env_for_render,
        mkdir_keys=(),
    )
    with log_target.open(mode, encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        ret = proc.wait()

    if pointer_path is not None:
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            json.dumps(
                {
                    "note": "Full stdout/stderr was written to transient storage to reduce disk IO.",
                    "requested_log_path": str(log_path),
                    "transient_log_path": str(log_target),
                    "returncode": int(ret),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    result = StreamRunResult(tuple(cmd), log_path, int(ret), False, rendered)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return result
