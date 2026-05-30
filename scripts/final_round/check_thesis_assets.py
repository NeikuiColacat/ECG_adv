#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def iter_local_image_links(markdown_path: Path) -> list[tuple[int, str, Path]]:
    links: list[tuple[int, str, Path]] = []
    for line_no, line in enumerate(markdown_path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in IMAGE_RE.finditer(line):
            raw = match.group(1).strip()
            parsed = urlparse(raw)
            if parsed.scheme and parsed.scheme not in {"", "file"}:
                continue
            clean = unquote(parsed.path if parsed.scheme == "file" else raw)
            path = Path(clean)
            if not path.is_absolute():
                path = markdown_path.parent / path
            links.append((line_no, raw, path))
    return links


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thesis", default=str(REPO_ROOT / "thesis.md"))
    args = parser.parse_args()

    thesis_path = Path(args.thesis).expanduser()
    if not thesis_path.exists():
        print(f"[missing] thesis file -> {thesis_path}", file=sys.stderr)
        return 2

    missing = 0
    links = iter_local_image_links(thesis_path)
    for line_no, raw, path in links:
        rel = path.resolve().relative_to(REPO_ROOT.resolve()) if path.resolve().is_relative_to(REPO_ROOT.resolve()) else path
        if path.is_file():
            print(f"[ok] line {line_no}: {raw} -> {rel}")
        else:
            missing += 1
            print(f"[missing] line {line_no}: {raw} -> {path}")

    print(f"[summary] checked={len(links)} missing={missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
