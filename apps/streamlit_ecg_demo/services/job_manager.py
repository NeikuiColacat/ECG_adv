from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobSpec:
    name: str
    command: list[str]
    cwd: str
    env: dict[str, str] | None = None


def _job_slug(name: str) -> str:
    keep = []
    for ch in name.lower().strip():
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    slug = "".join(keep).strip("._-")
    return slug or "job"


def start_job(project_root: Path, spec: JobSpec) -> Path:
    jobs_dir = project_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"{int(time.time())}_{_job_slug(spec.name)}"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "stdout.log"
    meta_path = job_dir / "job.json"
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    if spec.env:
        env.update(spec.env)
    log_f = open(log_path, "ab")
    proc = subprocess.Popen(
        spec.command,
        cwd=spec.cwd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_f.close()
    meta = {
        "job_id": job_id,
        "name": spec.name,
        "command": spec.command,
        "cwd": spec.cwd,
        "pid": proc.pid,
        "status": "running",
        "started_at": time.time(),
        "log_path": str(log_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return job_dir


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_job(job_dir: Path) -> dict:
    meta_path = job_dir / "job.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") == "running":
        pid = int(meta.get("pid", -1))
        try:
            done_pid, return_status = os.waitpid(pid, os.WNOHANG)
            if done_pid == pid:
                code = os.waitstatus_to_exitcode(return_status)
                meta["status"] = "finished" if code == 0 else "failed"
                meta["returncode"] = code
                meta["ended_at"] = time.time()
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except ChildProcessError:
            if not _pid_alive(pid):
                meta["status"] = "finished_or_failed"
                meta["ended_at"] = time.time()
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def list_jobs(project_root: Path) -> list[Path]:
    jobs_dir = project_root / "jobs"
    if not jobs_dir.exists():
        return []
    return sorted([p for p in jobs_dir.iterdir() if (p / "job.json").exists()], reverse=True)


def tail_log(job_dir: Path, max_chars: int = 6000) -> str:
    path = job_dir / "stdout.log"
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_chars:].decode("utf-8", errors="replace")


def stop_job(job_dir: Path) -> None:
    meta = read_job(job_dir)
    pid = int(meta.get("pid", -1))
    if pid > 0 and _pid_alive(pid):
        os.killpg(pid, signal.SIGTERM)
        meta["status"] = "stopping"
        meta["stopped_at"] = time.time()
        (job_dir / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
