#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
DEFAULT_ARCHIVE_PATTERNS = (
    "ecg_grad_min_runtime_artifacts_*.tar.gz",
    "ecg_grad_repro_addons_*.tar.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(migrate_dir: Path) -> dict[str, str]:
    checksum_files = sorted(migrate_dir.glob("SHA256SUMS*.txt"))
    checksums: dict[str, str] = {}
    for checksum_file in checksum_files:
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            digest, name = parts[0], parts[-1]
            checksums[Path(name).name] = digest
    return checksums


def discover_archives(migrate_dir: Path, patterns: tuple[str, ...] = DEFAULT_ARCHIVE_PATTERNS) -> list[Path]:
    archives: list[Path] = []
    for pattern in patterns:
        archives.extend(sorted(migrate_dir.glob(pattern)))
    return sorted(dict.fromkeys(archives))


def verify_checksum(path: Path, checksums: dict[str, str]) -> str | None:
    expected = checksums.get(path.name)
    if not expected:
        return None
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")
    return actual


def safe_members(archive: tarfile.TarFile, out_dir: Path) -> list[tarfile.TarInfo]:
    out_root = out_dir.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe tar member path: {member.name}")
        target = (out_dir / Path(*name.parts)).resolve()
        if target != out_root and out_root not in target.parents:
            raise ValueError(f"unsafe tar member target: {member.name}")
        members.append(member)
    return members


def restore_archives(
    *,
    migrate_dir: str | Path,
    out_dir: str | Path,
    archive_patterns: tuple[str, ...] = DEFAULT_ARCHIVE_PATTERNS,
    dry_run: bool = False,
    verify: bool = True,
) -> list[dict[str, Any]]:
    migrate_dir = Path(migrate_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    checksums = load_checksums(migrate_dir) if verify else {}
    archives = discover_archives(migrate_dir, archive_patterns)
    if not archives:
        raise FileNotFoundError(f"no migration artifact archives found under {migrate_dir}")

    restored: list[dict[str, Any]] = []
    for archive_path in archives:
        digest = verify_checksum(archive_path, checksums) if verify else None
        with tarfile.open(archive_path, "r:gz") as tar:
            members = safe_members(tar, out_dir)
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                tar.extractall(out_dir, members=members)
        restored.append({
            "archive": str(archive_path),
            "members": len(members),
            "dry_run": dry_run,
            "sha256": digest,
        })
    return restored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate_dir", default=str(REPO_ROOT / "migrate_files"))
    parser.add_argument("--out_dir", default=str(DATA_ROOT))
    parser.add_argument(
        "--archive-pattern",
        action="append",
        dest="archive_patterns",
        help="Glob pattern to select tar archives inside migrate_dir. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    restored = restore_archives(
        migrate_dir=args.migrate_dir,
        out_dir=args.out_dir,
        archive_patterns=tuple(args.archive_patterns) if args.archive_patterns else DEFAULT_ARCHIVE_PATTERNS,
        dry_run=args.dry_run,
        verify=not args.no_verify,
    )
    print(json.dumps({"out_dir": str(Path(args.out_dir).expanduser()), "archives": restored}, indent=2))


if __name__ == "__main__":
    main()
