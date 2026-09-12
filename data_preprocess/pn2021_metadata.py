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


def parse_pn2021_header(
    header_path: str | Path,
) -> tuple[list[int], dict[str, Any]]:
    """Read one PN2021 header and return SNOMED codes plus demographics."""

    snomed_codes: list[int] = []
    diagnosis_seen = False
    metadata: dict[str, Any] = {"age": None, "sex": None, "hr": None}
    with Path(header_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            diagnosis = DX_LINE_RE.match(stripped)
            if diagnosis is not None and not diagnosis_seen:
                diagnosis_seen = True
                try:
                    snomed_codes = [
                        int(value.strip())
                        for value in diagnosis.group(1).split(",")
                        if value.strip()
                    ]
                except ValueError:
                    snomed_codes = []
            if not stripped.startswith("#"):
                continue
            body = stripped[1:].strip()
            if body.startswith("Age:"):
                try:
                    age = float(body.split(":", 1)[1].strip())
                except ValueError:
                    continue
                if math.isfinite(age):
                    metadata["age"] = age
            elif body.startswith("Sex:"):
                raw_sex = body.split(":", 1)[1].strip().upper()
                metadata["sex"] = {"M": "M", "F": "F"}.get(raw_sex[:1], "U")
    return snomed_codes, metadata


__all__ = ["parse_pn2021_header"]
