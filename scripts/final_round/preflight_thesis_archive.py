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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "docs" / "artifact_manifest.json"))
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    manifest = load_manifest(manifest_path)
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        print(f"[error] no artifacts listed in {manifest_path}", file=sys.stderr)
        return 2

    statuses: dict[str, int] = {"ok": 0, "missing-required": 0, "missing-optional": 0, "bad": 0}
    for item in artifacts:
        if not item.get("required", False) and not args.include_optional:
            continue
        status, message = check_artifact(item, verify_sha=args.verify_sha)
        statuses[status] = statuses.get(status, 0) + 1
        print(message)

    print(
        "[summary] ok={ok} missing_required={missing_required} "
        "missing_optional={missing_optional} bad={bad}".format(
            ok=statuses.get("ok", 0),
            missing_required=statuses.get("missing-required", 0),
            missing_optional=statuses.get("missing-optional", 0),
            bad=statuses.get("bad", 0),
        )
    )
    if statuses.get("missing-required", 0) or statuses.get("bad", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
