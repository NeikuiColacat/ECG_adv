"""Single-host durable work queue: fixed tasks, kernel locks, atomic commits.

No Torch, SQLite, network service or TTL-based ownership stealing. A live
worker holds a kernel file lock until commit/release. Process death releases
the lock; the next worker can repeat only that uncommitted task. A completed
task is immutable. This is exactly-once acceptance, not exactly-once compute.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Durable same-filesystem rename; safe for process and machine restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fixed_tasks(rows: list[dict], *, records_per_task: int, batch_size: int,
                condition_ids: list[str], protocol_identity: str) -> list[dict]:
    if records_per_task < batch_size or records_per_task % batch_size or batch_size < 1:
        raise ValueError("task size must be a positive multiple of the frozen batch size")
    keys = [r["sample_key"] for r in rows]
    if len(keys) != len(set(keys)) or len(condition_ids) != len(set(condition_ids)):
        raise ValueError("duplicate task input identity")
    tasks = []
    for start in range(0, len(rows), records_per_task):
        selected = keys[start:start + records_per_task]
        batches = [selected[i:i + batch_size] for i in range(0, len(selected), batch_size)]
        body = {"protocol_identity": protocol_identity, "batches": batches,
                "condition_ids": condition_ids}
        tasks.append({"id": digest_json(body), "index": len(tasks), **body,
                      "expected_predictions": len(selected) * len(condition_ids)})
    return tasks


class TaskLease:
    def __init__(self, queue: "TaskQueue", task: dict, handle, worker_id: str):
        self.queue, self.task, self.handle, self.worker_id = queue, task, handle, worker_id
        self.started = time.time()

    def commit(self, rows: list[dict], performance: dict) -> None:
        if self.handle.closed:
            raise ValueError("cannot commit an expired task lease")
        self.queue.validate_rows(self.task, rows)
        destination = self.queue.result_path(self.task)
        if destination.exists():
            raise FileExistsError("completed task is immutable")
        atomic_json(destination, {"schema_version": 1, "task_id": self.task["id"],
                                 "protocol_identity": self.task["protocol_identity"],
                                 "worker_id": self.worker_id, "started_at": self.started,
                                 "completed_at": time.time(), "performance": performance,
                                 "rows_sha256": digest_json(rows), "predictions": rows})

    def close(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TaskQueue:
    def __init__(self, root: Path, tasks: list[dict]):
        self.root, self.tasks = root, tasks
        (root / "locks").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)
        manifest = root / "tasks.json"
        with (root / "manifest.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest.exists():
                if json.loads(manifest.read_text()) != tasks:
                    raise ValueError("task manifest changed; cannot resume")
            else:
                atomic_json(manifest, tasks)
        if len({t["id"] for t in tasks}) != len(tasks):
            raise ValueError("duplicate task id")

    def result_path(self, task: dict) -> Path:
        return self.root / "results" / f"{task['id']}.json"

    @staticmethod
    def validate_rows(task: dict, rows: list[dict]) -> None:
        expected = {(sample, condition) for batch in task["batches"] for sample in batch
                    for condition in task["condition_ids"]}
        keys = [(r["sample_key"], r["condition_id"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != task["expected_predictions"]:
            raise ValueError("incomplete, duplicate or foreign task predictions")

    def validate_completed(self, task: dict) -> dict:
        payload = json.loads(self.result_path(task).read_text())
        if (payload.get("schema_version") != 1 or payload.get("task_id") != task["id"]
                or payload.get("protocol_identity") != task["protocol_identity"]):
            raise ValueError("completed task identity mismatch")
        rows = payload["predictions"]
        self.validate_rows(task, rows)
        if digest_json(rows) != payload["rows_sha256"]:
            raise ValueError("completed task content changed")
        return payload

    def claim(self, worker_id: str, *, allowed_task_ids: set[str] | None = None) -> TaskLease | None:
        if allowed_task_ids is not None and not allowed_task_ids <= {t["id"] for t in self.tasks}:
            raise ValueError("task filter contains foreign task identities")
        for task in self.tasks:
            if allowed_task_ids is not None and task["id"] not in allowed_task_ids:
                continue
            if self.result_path(task).exists():
                continue
            handle = (self.root / "locks" / f"{task['id']}.lock").open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            if self.result_path(task).exists():
                handle.close()
                continue
            return TaskLease(self, task, handle, worker_id)
        return None

    def counts(self) -> dict:
        completed = [t for t in self.tasks if self.result_path(t).exists()]
        return {"tasks": len(self.tasks), "completed_tasks": len(completed),
                "expected_predictions": sum(t["expected_predictions"] for t in self.tasks),
                "completed_predictions": sum(t["expected_predictions"] for t in completed)}

    def completed_payloads(self) -> Iterator[dict]:
        for task in self.tasks:
            yield self.validate_completed(task)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
