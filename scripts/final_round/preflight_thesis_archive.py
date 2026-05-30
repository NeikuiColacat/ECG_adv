#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_path(raw: str) -> Path:
    values = {
        "REPO_ROOT": str(REPO_ROOT),
        "ECG_ADV_DATA_ROOT": str(DATA_ROOT),
        "HOME": str(Path.home()),
    }
    values.update({k: v for k, v in os.environ.items() if k.startswith("ECG_ADV_")})

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        return values.get(name, match.group(0))

    expanded = re.sub(r"\$\{([^}]+)\}", repl, raw)
    return Path(expanded).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifact(item: dict[str, Any], *, verify_sha: bool) -> tuple[str, str]:
    path = expand_path(item["path"])
    required = bool(item.get("required", False))
    package = bool(item.get("package", True))
    label = "required" if required else "optional"
    if not package:
        label += ":nonpackage"
    if not path.exists():
        return ("missing-required" if required else "missing-optional", f"[missing:{label}] {item['id']} -> {path}")
    if not path.is_file():
        return ("bad", f"[bad] {item['id']} is not a file -> {path}")

    size = path.stat().st_size
    expected_size = item.get("size_bytes")
    if expected_size is not None and int(expected_size) != int(size):
        return (
            "bad",
            f"[bad] {item['id']} size mismatch: expected {expected_size}, got {size} -> {path}",
        )

    expected_sha = item.get("sha256")
    if verify_sha and expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            return (
                "bad",
                f"[bad] {item['id']} sha256 mismatch: expected {expected_sha}, got {actual_sha} -> {path}",
            )

    return ("ok", f"[ok:{label}] {item['id']} -> {path}")


def should_check_item(item: dict[str, Any], *, scope: str, include_optional: bool) -> bool:
    if not item.get("required", False) and not include_optional:
        return False
    if scope == "archive":
        return bool(item.get("package", True)) and bool(item.get("archive_required", True))
    if scope == "full":
        return True
    raise ValueError(f"unknown preflight scope: {scope}")


def check_manifest(
    manifest: dict[str, Any],
    *,
    verify_sha: bool,
    scope: str,
    include_optional: bool,
) -> dict[str, int]:
    statuses, _ = collect_manifest_statuses(
        manifest,
        verify_sha=verify_sha,
        scope=scope,
        include_optional=include_optional,
    )
    return statuses


def _status_record(item: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "status": status,
        "id": item["id"],
        "path": item["path"],
        "resolved_path": str(expand_path(item["path"])),
        "kind": item.get("kind"),
        "required": bool(item.get("required", False)),
        "package": bool(item.get("package", True)),
        "archive_required": bool(item.get("archive_required", True)),
        "used_by": item.get("used_by", []),
        "description": item.get("description", ""),
        "regenerate_command": item.get("regenerate_command"),
    }


def collect_manifest_statuses(
    manifest: dict[str, Any],
    *,
    verify_sha: bool,
    scope: str,
    include_optional: bool,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    statuses: dict[str, int] = {
        "ok": 0,
        "missing-required": 0,
        "missing-optional": 0,
        "bad": 0,
        "skipped": 0,
    }
    records: list[dict[str, Any]] = []
    for item in manifest.get("artifacts", []):
        if not should_check_item(item, scope=scope, include_optional=include_optional):
            statuses["skipped"] += 1
            records.append(_status_record(item, "skipped"))
            print(f"[skip:{scope}] {item['id']}")
            continue
        status, message = check_artifact(item, verify_sha=verify_sha)
        statuses[status] = statuses.get(status, 0) + 1
        records.append(_status_record(item, status))
        print(message)
    return statuses, records


def write_missing_reports(
    records: list[dict[str, Any]],
    *,
    json_path: Path | None,
    md_path: Path | None,
    scope: str,
) -> None:
    missing = [r for r in records if str(r.get("status", "")).startswith("missing") or r.get("status") == "bad"]
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"scope": scope, "missing_artifacts": missing}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        title = f"Missing {scope.title()} Preflight Artifacts"
        lines = [
            f"# {title}",
            "",
            "These files are required by the selected `preflight_thesis_archive.py` scope but were not present.",
            "",
        ]
        if not missing:
            lines.append("No missing artifacts.")
        for item in missing:
            used_by = ", ".join(item.get("used_by", [])) or "n/a"
            command = item.get("regenerate_command") or "n/a"
            lines.extend([
                f"## `{item['id']}`",
                "",
                f"- status: `{item.get('status')}`",
                f"- kind: `{item.get('kind', 'n/a')}`",
                f"- required: `{bool(item.get('required'))}`",
                f"- package: `{bool(item.get('package'))}`",
                f"- archive required: `{bool(item.get('archive_required'))}`",
                f"- source path: `{item.get('path', '')}`",
                f"- resolved path: `{item.get('resolved_path', '')}`",
                f"- used by: {used_by}",
                f"- description: {item.get('description', '')}",
                f"- regenerate: `{command}`",
                "",
            ])
        md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "docs" / "artifact_manifest.json"))
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--scope", choices=["archive", "full"], default="archive")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    manifest = load_manifest(manifest_path)
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        print(f"[error] no artifacts listed in {manifest_path}", file=sys.stderr)
        return 2

    statuses, records = collect_manifest_statuses(
        manifest,
        verify_sha=args.verify_sha,
        scope=args.scope,
        include_optional=args.include_optional,
    )
    write_missing_reports(
        records,
        json_path=Path(args.report_json).expanduser() if args.report_json else None,
        md_path=Path(args.report_md).expanduser() if args.report_md else None,
        scope=args.scope,
    )

    print(
        "[summary] ok={ok} missing_required={missing_required} "
        "missing_optional={missing_optional} bad={bad} skipped={skipped}".format(
            ok=statuses.get("ok", 0),
            missing_required=statuses.get("missing-required", 0),
            missing_optional=statuses.get("missing-optional", 0),
            bad=statuses.get("bad", 0),
            skipped=statuses.get("skipped", 0),
        )
    )
    if statuses.get("missing-required", 0) or statuses.get("bad", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
