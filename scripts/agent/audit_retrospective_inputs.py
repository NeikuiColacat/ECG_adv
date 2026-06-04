#!/usr/bin/env python3
"""Inventory project-local retrospective inputs for ECG_adv_Gen agents.

This script is intentionally read-only. It reports candidate files that a
retrospective agent may inspect before proposing small AGENTS.md or skill
updates. It does not read global Codex session content unless explicit session
files or directories are provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_HASH_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class FileRecord:
    category: str
    path: str
    size_bytes: int
    mtime_utc: str
    sha256: str | None
    line_count: int | None
    note: str


def utc_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / ".git").exists():
            return candidate
    raise SystemExit(f"Could not find repo root above {start}")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def iter_globs(root: Path, category: str, patterns: Iterable[str]) -> Iterable[tuple[str, Path]]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield category, path


def safe_hash_and_lines(path: Path, max_hash_bytes: int) -> tuple[str | None, int | None, str]:
    size = path.stat().st_size
    if size > max_hash_bytes:
        return None, None, f"skipped content hash: file exceeds {max_hash_bytes} bytes"
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return digest, None, "binary-or-non-utf8"
    return digest, text.count("\n") + (0 if text.endswith("\n") or not text else 1), ""


def make_record(root: Path, category: str, path: Path, max_hash_bytes: int, note: str = "") -> FileRecord:
    stat = path.stat()
    digest, line_count, hash_note = safe_hash_and_lines(path, max_hash_bytes)
    rel = path.resolve().relative_to(root.resolve()) if is_within(path, root) else path.resolve()
    notes = "; ".join(part for part in (note, hash_note) if part)
    return FileRecord(
        category=category,
        path=str(rel),
        size_bytes=int(stat.st_size),
        mtime_utc=utc_from_timestamp(stat.st_mtime),
        sha256=digest,
        line_count=line_count,
        note=notes,
    )


def collect_project_records(root: Path, max_hash_bytes: int) -> list[FileRecord]:
    specs: list[tuple[str, list[str]]] = [
        ("core_memory", ["AGENTS.md"]),
        ("skills", [".codex/skills/*/SKILL.md", ".codex/skills/*/references/*.md"]),
        ("handoffs", ["docs/codex-handoffs/*"]),
        ("pipeline_docs", ["docs/pipelines/*.md"]),
        ("labeling_docs", ["docs/labeling/*.md", "configs/label_mappings/*.jsonl"]),
        (
            "summary_docs",
            [
                "docs/reports/archive/**/*.md",
                "docs/reports/archive/**/*.html",
            ],
        ),
        (
            "active_config",
            [
                "configs/active_scripts.yaml",
                "configs/active_evidence_registry.yaml",
                "configs/evidence/*.yaml",
                "configs/evidence/*.json",
            ],
        ),
    ]
    records: list[FileRecord] = []
    for category, patterns in specs:
        for item_category, path in iter_globs(root, category, patterns):
            records.append(make_record(root, item_category, path, max_hash_bytes))
    return sorted(records, key=lambda r: (r.category, r.path))


def collect_session_records(
    root: Path,
    session_files: list[str],
    session_dirs: list[str],
    max_hash_bytes: int,
) -> list[FileRecord]:
    records: list[FileRecord] = []
    home = Path.home().resolve()
    explicit_paths = [Path(p).expanduser() for p in session_files]
    for directory in [Path(p).expanduser() for p in session_dirs]:
        resolved_dir = directory.resolve()
        if not is_within(resolved_dir, home):
            raise SystemExit(f"Refusing session dir outside home: {directory}")
        explicit_paths.extend(p for p in resolved_dir.glob("*") if p.is_file())

    for path in explicit_paths:
        resolved = path.resolve()
        if not is_within(resolved, home):
            raise SystemExit(f"Refusing session file outside home: {path}")
        if resolved.is_file():
            records.append(
                make_record(
                    root,
                    "explicit_session_input",
                    resolved,
                    max_hash_bytes,
                    note="explicit user-provided session input",
                )
            )
    return sorted(records, key=lambda r: r.path)


def render_markdown(records: list[FileRecord], root: Path) -> str:
    lines = [
        "# Retrospective Input Inventory",
        "",
        f"- repo_root: `{root}`",
        f"- generated_at_utc: `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`",
        f"- file_count: `{len(records)}`",
        "",
        "| category | path | size | lines | note |",
        "|---|---|---:|---:|---|",
    ]
    for record in records:
        line_count = "" if record.line_count is None else str(record.line_count)
        lines.append(
            f"| {record.category} | `{record.path}` | {record.size_bytes} | {line_count} | {record.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="", help="Repo root. Defaults to nearest parent with AGENTS.md and .git.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--session-file", action="append", default=[], help="Explicit session transcript/handoff file to inventory.")
    parser.add_argument("--session-dir", action="append", default=[], help="Explicit session directory to inventory, non-recursive.")
    parser.add_argument("--max-hash-bytes", type=int, default=DEFAULT_MAX_HASH_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else find_repo_root(Path.cwd())
    records = collect_project_records(root, args.max_hash_bytes)
    records.extend(collect_session_records(root, args.session_file, args.session_dir, args.max_hash_bytes))
    records = sorted(records, key=lambda r: (r.category, r.path))

    if args.format == "markdown":
        print(render_markdown(records, root))
    else:
        payload = {
            "repo_root": str(root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "records": [asdict(record) for record in records],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
