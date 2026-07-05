"""PN2021 metadata indexing helpers.

These helpers intentionally work only at the header/path metadata level. They
do not read WFDB waveform records, build caches, or apply preprocessing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import PN2021_EVAL_CENTERS_7, PN2021_LEAK_EXCLUDED_CENTERS

PN2021_HEADER_SUFFIX = ".hea"
PN2021_HEADER_COUNT_PATTERN = "*.hea"
PN2021_HEADER_COUNT_SCOPE = "center_dir_and_immediate_subdirs"
DX_LINE_RE = re.compile(r"^#\s*Dx\s*:\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class PN2021HeaderRecord:
    """A single PN2021 header-level record entry."""

    header_path: Path
    record_path: Path
    record_id: str
    snomeds: tuple[int, ...]


def record_id_from_path(path: str | Path) -> str:
    """Return the basename record id used by legacy PN2021 ref-exclusion."""
    p = Path(path)
    return p.stem if p.suffix == PN2021_HEADER_SUFFIX else p.name


def center_from_record_path(path: str | Path) -> str:
    """Return the PN2021 center from a WFDB record path."""
    parts = Path(path).parts
    if "training" in parts:
        idx = parts.index("training")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def parse_header_snomeds(header_path: str | Path) -> list[int]:
    """Parse SNOMED codes from a PN2021 ``#Dx:`` header line.

    This preserves the legacy behavior used by ``pn2021_clean_eval.py``:
    malformed code lists return an empty list instead of raising.
    """
    with Path(header_path).open("r", encoding="utf-8") as f:
        for line in f:
            match = DX_LINE_RE.match(line.strip())
            if not match:
                continue
            codes_str = match.group(1).strip()
            try:
                return [int(c.strip()) for c in codes_str.split(",") if c.strip()]
            except ValueError:
                return []
    return []


def scan_pn2021_center_records(center_dir: str | Path) -> list[PN2021HeaderRecord]:
    """Recursively scan one PN2021 center for ``*.hea`` records.

    The scan follows the legacy ``os.walk`` ordering and returns record paths
    without the ``.hea`` suffix, matching WFDB ``rdrecord`` input conventions.
    """
    records: list[PN2021HeaderRecord] = []
    for root, _, files in os.walk(center_dir):
        for filename in files:
            if not filename.endswith(PN2021_HEADER_SUFFIX):
                continue
            header_path = Path(root) / filename
            record_path = header_path.with_suffix("")
            records.append(
                PN2021HeaderRecord(
                    header_path=header_path,
                    record_path=record_path,
                    record_id=record_id_from_path(record_path),
                    snomeds=tuple(parse_header_snomeds(header_path)),
                )
            )
    return records


def _iter_bounded_header_candidates(center_dir: Path) -> Iterable[Path]:
    for child in center_dir.iterdir():
        if child.is_file():
            yield child
        elif child.is_dir():
            yield from (grandchild for grandchild in child.iterdir() if grandchild.is_file())


def count_center_header_files(center_dir: str | Path) -> int | None:
    """Count PN2021 headers at center level and one grouping level below it.

    Returns ``None`` when the center path is missing or not a directory. This is
    for lightweight manifest auditing, not for exact recursive eval indexing.
    """
    path = Path(center_dir)
    if not path.exists() or not path.is_dir():
        return None
    return sum(
        1
        for child in _iter_bounded_header_candidates(path)
        if child.match(PN2021_HEADER_COUNT_PATTERN)
    )


def assert_not_forbidden_center(center: str) -> None:
    """Raise if ``center`` is a PN2021 shard that leaks PTB-XL."""
    if center.lower() in set(PN2021_LEAK_EXCLUDED_CENTERS):
        raise ValueError(f"FORBIDDEN center {center} would leak PTB-XL data")


def default_eval_centers() -> tuple[str, ...]:
    """Return the current PN2021 7-center eval order."""
    return PN2021_EVAL_CENTERS_7
