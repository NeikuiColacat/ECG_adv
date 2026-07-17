"""Pure PN2021 ``.hea`` metadata parsing for the manual rebuild.

This module intentionally reads header text only.  It does not import the
legacy data package, read WFDB waveforms, map labels, resample signals, or
write caches.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


DX_LINE_RE = re.compile(r"^#\s*Dx\s*:\s*(.*)$", re.IGNORECASE)


def parse_header_snomeds(header_path: str | Path) -> list[int]:
    """Return integer SNOMED codes from the first PN2021 ``#Dx:`` line.

    A missing diagnosis line or a malformed code list returns an empty list,
    matching the cache-building behavior used before the dependency split.
    File access errors remain visible to the caller.
    """

    with Path(header_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            match = DX_LINE_RE.match(line.strip())
            if match is None:
                continue
            raw_codes = match.group(1).strip()
            try:
                return [
                    int(value.strip())
                    for value in raw_codes.split(",")
                    if value.strip()
                ]
            except ValueError:
                return []
    return []


def parse_pn2021_header_metadata(header_path: str | Path) -> dict[str, Any]:
    """Return PN2021 age/sex metadata without reading the waveform.

    ``hr`` is retained as ``None`` for compatibility with existing cache
    metadata.  Unreadable headers return the same all-missing demographic
    record used by the previous preprocessing path.
    """

    metadata: dict[str, Any] = {"age": None, "sex": None, "hr": None}
    try:
        with Path(header_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                body = stripped[1:].strip()
                if body.startswith("Age:"):
                    raw_age = body.split(":", 1)[1].strip()
                    try:
                        age = float(raw_age)
                    except ValueError:
                        continue
                    if math.isfinite(age):
                        metadata["age"] = age
                elif body.startswith("Sex:"):
                    raw_sex = body.split(":", 1)[1].strip().upper()
                    if raw_sex.startswith("M"):
                        metadata["sex"] = "M"
                    elif raw_sex.startswith("F"):
                        metadata["sex"] = "F"
                    else:
                        metadata["sex"] = "U"
    except OSError:
        pass
    return metadata


__all__ = ["parse_header_snomeds", "parse_pn2021_header_metadata"]
