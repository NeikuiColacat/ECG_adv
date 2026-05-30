"""Small helpers for PhysioNet/CinC 2021 WFDB header metadata."""

from __future__ import annotations

import re
from pathlib import Path


_DX_RE = re.compile(r"^#\s*Dx\s*:\s*(.*)$", re.IGNORECASE)


def parse_header_snomed(header_path: str | Path) -> list[int]:
    """Return SNOMED-CT diagnosis codes from a PN2021 `.hea` file."""
    with Path(header_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _DX_RE.match(line.strip())
            if not match:
                continue
            codes_str = match.group(1).strip()
            try:
                return [int(c.strip()) for c in codes_str.split(",") if c.strip()]
            except ValueError:
                return []
    return []
