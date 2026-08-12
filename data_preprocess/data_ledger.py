"""Deterministic, relocatable content ledger for retained ECG data roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml


ROOT_NAMES = ("ptbxl_cache", "pn2021_cache", "pn2021c_cache", "split_artifacts")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HEADER_KEYS = {"algorithm", "artifact", "member_count", "roots", "schema_version", "total_size_bytes"}
_MEMBER_KEYS = {"path", "root", "sha256", "size_bytes"}
_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _configured_path(value: Any, config: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.abspath(path))


def _required(names: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(names)
    if not all(isinstance(name, str) for name in requested):
        raise ValueError("required_roots must contain strings")
    if not requested:
        raise ValueError("required_roots must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("required_roots contains duplicates")
    unknown = set(requested).difference(ROOT_NAMES)
    if unknown:
        raise ValueError(f"unknown data roots: {sorted(unknown)}")
    return tuple(name for name in ROOT_NAMES if name in requested)


def load_roots(
    splits_config: str | Path,
    cache_config: str | Path | None = None,
    required_roots: Iterable[str] = ROOT_NAMES,
) -> dict[str, Path]:
    """Resolve fixed logical roots from the two runtime-owned YAML files."""

    splits_path = Path(splits_config).expanduser().absolute()
    cache_path = (
        Path(cache_config).expanduser().absolute()
        if cache_config is not None
        else splits_path.parent.parent / "augmentation" / "cache.yaml"
    )
    required = _required(required_roots)
    splits = _load_yaml(splits_path) if set(required).difference({"pn2021c_cache"}) else {}
    cache = _load_yaml(cache_path) if "pn2021c_cache" in required else {}
    ptbxl = _mapping(splits.get("ptbxl"), "splits.ptbxl") if "ptbxl_cache" in required else {}
    pn2021 = _mapping(splits.get("pn2021"), "splits.pn2021") if "pn2021_cache" in required else {}
    split_output = _mapping(splits.get("output"), "splits.output") if "split_artifacts" in required else {}
    cache_output = _mapping(cache.get("output"), "cache.output") if cache else {}
    values = {
        "ptbxl_cache": (ptbxl.get("cache_dir"), splits_path, "ptbxl.cache_dir"),
        "pn2021_cache": (pn2021.get("cache_dir"), splits_path, "pn2021.cache_dir"),
        "pn2021c_cache": (cache_output.get("cache_dir"), cache_path, "output.cache_dir"),
        "split_artifacts": (split_output.get("root_dir"), splits_path, "output.root_dir"),
    }
    return {name: _configured_path(*values[name]) for name in required}


def _selected_roots(
    roots: Mapping[str, str | Path], required_roots: Iterable[str]
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for name in _required(required_roots):
        if name not in roots:
            raise ValueError(f"missing configured data root: {name}")
        selected[name] = Path(os.path.abspath(Path(roots[name]).expanduser()))
    return selected


def _stat_row(relative: str, kind: str, value: os.stat_result) -> tuple[Any, ...]:
    return (
        relative,
        kind,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _kind(value: os.stat_result) -> str:
    if stat.S_ISREG(value.st_mode):
        return "file"
    if stat.S_ISDIR(value.st_mode):
        return "directory"
    raise ValueError("data roots may contain only regular files and directories")


def _inventory(root_name: str, root: Path) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing data root: {root_name}") from error
    if stat.S_ISLNK(root_stat.st_mode) or _kind(root_stat) != "directory":
        raise ValueError(f"data root must be a real directory: {root_name}")
    snapshot = [_stat_row(".", "directory", root_stat)]
    files: list[tuple[str, Path, tuple[Any, ...]]] = []
    def fail_walk(error: OSError) -> None:
        raise ValueError(f"cannot inventory data root {root_name}: {error}") from error

    for directory, names, filenames in os.walk(root, followlinks=False, onerror=fail_walk):
        names.sort()
        filenames.sort()
        parent = Path(directory)
        for name in [*names, *filenames]:
            path = parent / name
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(f"path escaped data root: {path}") from error
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode):
                raise ValueError(f"symlink is forbidden in data root: {root_name}/{relative}")
            kind = _kind(value)
            row = _stat_row(relative, kind, value)
            snapshot.append(row)
            if kind == "file":
                files.append((relative, path, row))
    snapshot.sort(key=lambda row: (row[0], row[1]))
    files.sort(key=lambda row: row[0])
    return tuple(snapshot), tuple(files)


def _hash_file(path: Path, expected: tuple[Any, ...]) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if _stat_row(expected[0], "file", before) != expected:
            raise ValueError(f"data file changed before hashing: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_row(expected[0], "file", after) != expected:
            raise ValueError(f"data file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _line(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _summary(
    digest: str, root_names: Iterable[str], members: Iterable[Mapping[str, Any]], mode: str
) -> dict[str, Any]:
    per_root = {
        name: {"member_count": 0, "total_size_bytes": 0} for name in root_names
    }
    for member in members:
        values = per_root[member["root"]]
        values["member_count"] += 1
        values["total_size_bytes"] += member["size_bytes"]
    return {
        "ledger_sha256": digest,
        "member_count": sum(item["member_count"] for item in per_root.values()),
        "roots": per_root,
        "total_size_bytes": sum(item["total_size_bytes"] for item in per_root.values()),
        "verification_mode": mode,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if linked:
            path.unlink(missing_ok=True)
        raise


def _assert_output_outside(path: Path, roots: Mapping[str, Path]) -> None:
    destination = path.parent.resolve(strict=False) / path.name
    for name, root in roots.items():
        try:
            destination.relative_to(root.resolve(strict=True))
        except ValueError:
            continue
        raise ValueError(f"ledger output must be outside data root: {name}")


def generate_ledger(
    roots: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash roots serially and atomically write a canonical JSONL ledger."""

    selected = _selected_roots(roots, ROOT_NAMES)
    output = Path(output_path).expanduser().absolute()
    inventories = {name: _inventory(name, root) for name, root in selected.items()}
    _assert_output_outside(output, selected)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"data ledger output already exists: {output}")
    members: list[dict[str, Any]] = []
    for root_name, root in selected.items():
        for relative, path, row in inventories[root_name][1]:
            members.append(
                {
                    "path": relative,
                    "root": root_name,
                    "sha256": _hash_file(path, row),
                    "size_bytes": row[4],
                }
            )
    for root_name, root in selected.items():
        if _inventory(root_name, root)[0] != inventories[root_name][0]:
            raise ValueError(f"data root changed during ledger generation: {root_name}")
    members.sort(key=lambda item: (item["root"], item["path"]))
    header = {
        "algorithm": "sha256",
        "artifact": "ecg_data_content_ledger",
        "member_count": len(members),
        "roots": list(selected),
        "schema_version": 1,
        "total_size_bytes": sum(item["size_bytes"] for item in members),
    }
    content = b"".join([_line(header), *(_line(item) for item in members)])
    _atomic_write(output, content)
    return _summary(hashlib.sha256(content).hexdigest(), selected, members, "full_content")


def _parse_ledger(
    path: Path, expected_sha256: str | None
) -> tuple[str, list[str], list[dict[str, Any]]]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("data ledger SHA-256 mismatch")
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("data ledger must use canonical LF-terminated JSONL")
    try:
        payloads = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("data ledger is not valid UTF-8 JSONL") from error
    if not payloads or not all(isinstance(item, dict) for item in payloads):
        raise ValueError("data ledger must contain a header and object members")
    if b"".join(_line(item) for item in payloads) != raw:
        raise ValueError("data ledger is not canonically serialized")
    header, members = payloads[0], payloads[1:]
    previous: tuple[str, str] | None = None
    for item in members:
        if set(item) != _MEMBER_KEYS:
            raise ValueError("invalid data ledger member fields")
        root, relative, size, sha = (
            item["root"], item["path"], item["size_bytes"], item["sha256"]
        )
        pure = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
        if root not in ROOT_NAMES or pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise ValueError("invalid or escaping data ledger member path")
        if pure.as_posix() != relative or relative in {"", "."}:
            raise ValueError("data ledger member path is not canonical POSIX")
        if type(size) is not int or size < 0:
            raise ValueError("invalid data ledger member size")
        if not isinstance(sha, str) or len(sha) != 64 or set(sha).difference(_HEX):
            raise ValueError("invalid data ledger member SHA-256")
        identity = (root, relative)
        if previous is not None and identity <= previous:
            raise ValueError("data ledger members are duplicated or unsorted")
        previous = identity
    roots = header.get("roots") if isinstance(header, dict) else None
    if roots != list(ROOT_NAMES):
        raise ValueError("invalid data ledger root order")
    for key in ("schema_version", "member_count", "total_size_bytes"):
        if type(header.get(key)) is not int:
            raise ValueError(f"data ledger header {key} must be an integer")
    expected_header = {
        "algorithm": "sha256", "artifact": "ecg_data_content_ledger",
        "member_count": len(members), "roots": roots, "schema_version": 1,
        "total_size_bytes": sum(item["size_bytes"] for item in members),
    }
    if set(header) != _HEADER_KEYS or header != expected_header:
        raise ValueError("invalid data ledger header")
    if any(item["root"] not in roots for item in members):
        raise ValueError("data ledger member uses an undeclared root")
    return digest, roots, members


def verify_ledger(
    ledger_path: str | Path,
    roots: Mapping[str, str | Path],
    required_roots: Iterable[str] = ROOT_NAMES,
    *,
    expected_sha256: str | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Verify canonical identity, then exact inventory or complete contents."""

    digest, ledger_roots, members = _parse_ledger(Path(ledger_path), expected_sha256)
    selected = _selected_roots(roots, required_roots)
    if not set(selected).issubset(ledger_roots):
        raise ValueError("required root is absent from data ledger")
    expected = {
        (item["root"], item["path"]): item
        for item in members
        if item["root"] in selected
    }
    before: dict[str, tuple[Any, ...]] = {}
    actual_files: list[tuple[str, str, Path, tuple[Any, ...]]] = []
    for root_name, root in selected.items():
        snapshot, files = _inventory(root_name, root)
        before[root_name] = snapshot
        actual_files.extend((root_name, relative, path, row) for relative, path, row in files)
    actual = {(root, relative): row[4] for root, relative, _, row in actual_files}
    ledger_sizes = {identity: item["size_bytes"] for identity, item in expected.items()}
    if actual != ledger_sizes:
        raise ValueError("data root inventory or file size differs from ledger")
    if full:
        for root_name, relative, path, row in actual_files:
            if _hash_file(path, row) != expected[(root_name, relative)]["sha256"]:
                raise ValueError(f"data content SHA-256 mismatch: {root_name}/{relative}")
    for root_name, root in selected.items():
        if _inventory(root_name, root)[0] != before[root_name]:
            raise ValueError(f"data root changed during ledger verification: {root_name}")
    return _summary(
        digest, selected, expected.values(), "full_content" if full else "inventory_only"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--splits-config", required=True, type=Path)
        subparser.add_argument("--cache-config", type=Path)
        if command == "generate":
            subparser.add_argument("--output", required=True, type=Path)
        else:
            subparser.add_argument("--ledger", required=True, type=Path)
            subparser.add_argument("--expected-sha256")
            subparser.add_argument("--full", action="store_true")
            subparser.add_argument("--required-root", action="append", choices=ROOT_NAMES)
    arguments = parser.parse_args()
    required = arguments.required_root or ROOT_NAMES if arguments.command == "verify" else ROOT_NAMES
    roots = load_roots(arguments.splits_config, arguments.cache_config, required)
    if arguments.command == "generate":
        summary = generate_ledger(roots, arguments.output)
    else:
        summary = verify_ledger(
            arguments.ledger, roots, required,
            expected_sha256=arguments.expected_sha256, full=arguments.full,
        )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
