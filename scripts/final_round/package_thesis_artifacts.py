#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()


def expand_path(raw: str) -> Path:
    values = {
        "REPO_ROOT": str(REPO_ROOT),
        "ECG_ADV_DATA_ROOT": str(DATA_ROOT),
        "HOME": str(Path.home()),
    }
    values.update({k: v for k, v in os.environ.items() if k.startswith("ECG_ADV_")})

    def repl(match: re.Match[str]) -> str:
        return values.get(match.group(1), match.group(0))

    return Path(re.sub(r"\$\{([^}]+)\}", repl, raw)).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_relative_path(path: Path) -> Path:
    resolved = path.resolve()
    for root, prefix in [(DATA_ROOT.resolve(), Path()), (REPO_ROOT.resolve(), Path("repo_required"))]:
        try:
            return prefix / resolved.relative_to(root)
        except ValueError:
            continue
    return Path("external") / path.name


def copy_artifacts(
    manifest: dict[str, Any],
    out_dir: Path,
    *,
    include_optional: bool,
    include_nonpackage: bool,
    skip_missing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied = []
    missing = []
    for item in manifest.get("artifacts", []):
        required = bool(item.get("required", False))
        package = bool(item.get("package", True))
        if not package and not include_nonpackage:
            print(f"[skip] non-package artifact: {item['id']}")
            continue
        if not required and not include_optional:
            continue
        src = expand_path(item["path"])
        if not src.exists():
            message = f"missing artifact: {item['id']} -> {src}"
            if required and not skip_missing:
                raise FileNotFoundError(message)
            missing.append({
                "id": item["id"],
                "kind": item.get("kind"),
                "required": required,
                "package": package,
                "source_path": str(src),
                "used_by": item.get("used_by", []),
                "description": item.get("description", ""),
                "regenerate_command": item.get("regenerate_command"),
            })
            print(f"[skip] {message}")
            continue
        rel = archive_relative_path(src)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        copied.append({
            "id": item["id"],
            "kind": item.get("kind"),
            "required": required,
            "package": package,
            "source_path": str(src),
            "archive_path": str(rel),
            "size_bytes": dst.stat().st_size,
            "sha256": digest,
            "used_by": item.get("used_by", []),
            "description": item.get("description", ""),
        })
        print(f"[copy] {item['id']} -> {rel}")
    return copied, missing


def write_checksums(out_dir: Path, copied: list[dict[str, Any]]) -> None:
    lines = [f"{item['sha256']}  {item['archive_path']}" for item in copied]
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_markdown(out_path: Path, missing: list[dict[str, Any]]) -> None:
    lines = [
        "# Missing Thesis Archive Artifacts",
        "",
        "These files were listed in `docs/artifact_manifest.json` but were not present when the artifact package was built.",
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
            f"- kind: `{item.get('kind', 'n/a')}`",
            f"- required: `{bool(item.get('required'))}`",
            f"- package: `{bool(item.get('package'))}`",
            f"- source path: `{item.get('source_path', '')}`",
            f"- used by: {used_by}",
            f"- description: {item.get('description', '')}",
            f"- regenerate: `{command}`",
            "",
        ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_tar(out_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "docs" / "artifact_manifest.json"))
    parser.add_argument("--out_dir", default=str(DATA_ROOT / "thesis_archive_artifacts"))
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--include-nonpackage", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--tar", default=None, help="Optional output .tar.gz path.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    copied, missing = copy_artifacts(
        manifest,
        out_dir,
        include_optional=args.include_optional,
        include_nonpackage=args.include_nonpackage,
        skip_missing=args.skip_missing,
    )
    write_checksums(out_dir, copied)
    (out_dir / "artifact_manifest.resolved.json").write_text(
        json.dumps({"artifacts": copied}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "missing_artifacts.json").write_text(
        json.dumps({"missing_artifacts": missing}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_missing_markdown(out_dir / "missing_artifacts.md", missing)
    if args.tar:
        make_tar(out_dir, Path(args.tar).expanduser())
    print(f"[done] packaged {len(copied)} artifacts under {out_dir}; missing={len(missing)}")


if __name__ == "__main__":
    main()
