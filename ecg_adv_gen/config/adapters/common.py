"""Shared helpers for YAML-managed command audit adapters."""

from __future__ import annotations

from typing import Any, Mapping


def argv_option_map(argv: list[str]) -> dict[str, Any]:
    """Parse simple ``--option value...`` argv lists into an option map."""

    opts: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        item = argv[i]
        if not item.startswith("--"):
            i += 1
            continue
        values: list[str] = []
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            values.append(argv[j])
            j += 1
        opts[item] = values or True
        i = j
    return opts


def opt_first(opts: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return the first value for an option or a default."""

    value = opts.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def opt_list(opts: Mapping[str, Any], key: str) -> list[str]:
    """Return option values as strings."""

    value = opts.get(key)
    if value is None or value is True:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def audit_require_options(errors: list[str], script: str, opts: Mapping[str, Any], names: list[str]) -> None:
    """Append an error for each missing required option."""

    for name in names:
        if name not in opts:
            errors.append(f"{script}: missing required option {name}")


def audit_equals(errors: list[str], script: str, opts: Mapping[str, Any], name: str, expected: str) -> None:
    """Append an error when an option does not match the expected value."""

    value = opt_first(opts, name)
    if str(value) != str(expected):
        errors.append(f"{script}: {name}={value!r}, expected {expected!r}")


def matrix_case(command: Mapping[str, Any]) -> dict[str, Any]:
    """Return command matrix case metadata."""

    matrix = command.get("matrix") or {}
    if isinstance(matrix.get("case"), dict):
        return matrix["case"]
    return matrix
