"""Merge paper-safe ``metrics_long.csv`` files without changing semantics."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .metrics_export import METRICS_FIELDNAMES


class MetricsMergeError(ValueError):
    """Raised when metrics_long files cannot be safely merged."""


def _read_metrics(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != METRICS_FIELDNAMES:
            raise MetricsMergeError(
                f"Unexpected metrics_long columns in {path}: {reader.fieldnames}"
            )
        rows = list(reader)
    return rows, {"path": str(path), "n_rows": len(rows)}


def _mapping_pairs(rows: Iterable[dict[str, str]]) -> set[tuple[str, str, str]]:
    pairs = set()
    for row in rows:
        version = row.get("mapping_version", "")
        mapping_hash = row.get("mapping_hash", "")
        class_order = row.get("class_order", "")
        if version or mapping_hash or class_order:
            pairs.add((version, mapping_hash, class_order))
    return pairs


def _validate_duplicate_rows(rows: list[dict[str, str]]) -> None:
    seen: dict[tuple[str, str, str, str, str, str, str, str], str] = {}
    for row in rows:
        key = (
            row.get("run_id", ""),
            row.get("dataset", ""),
            row.get("canonical_view", "") or row.get("view", ""),
            row.get("scope", ""),
            row.get("center", ""),
            row.get("class_name", ""),
            row.get("metric", ""),
            row.get("mapping_hash", ""),
        )
        source = row.get("source_file", "")
        if key in seen:
            raise MetricsMergeError(
                "Duplicate metric row after merge for "
                f"key={key}; sources={seen[key]} and {source}"
            )
        seen[key] = source


def merge_metrics_long(
    inputs: Iterable[Path],
    output_dir: Path,
    *,
    allow_mixed_mapping: bool = False,
) -> dict[str, Any]:
    input_paths = [Path(p).expanduser().resolve() for p in inputs]
    if not input_paths:
        raise MetricsMergeError("No metrics_long inputs supplied")

    rows: list[dict[str, str]] = []
    inputs_manifest: list[dict[str, Any]] = []
    for path in input_paths:
        file_rows, record = _read_metrics(path)
        rows.extend(file_rows)
        inputs_manifest.append(record)
    if not rows:
        raise MetricsMergeError("No metric rows found in metrics_long inputs")

    pairs = _mapping_pairs(rows)
    if not allow_mixed_mapping and len(pairs) > 1:
        raise MetricsMergeError(f"Refusing mixed mapping metadata: {sorted(pairs)}")
    _validate_duplicate_rows(rows)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_long.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    view_counts = Counter(row["view"] for row in rows)
    canonical_view_counts = Counter(row["canonical_view"] for row in rows)
    run_counts = Counter(row["run_id"] for row in rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_long": str(metrics_path),
        "merge_manifest": str(output_dir / "merge_manifest.json"),
        "n_input_files": len(input_paths),
        "n_metric_rows": len(rows),
        "inputs": inputs_manifest,
        "run_ids": dict(sorted(run_counts.items())),
        "views": dict(sorted(view_counts.items())),
        "canonical_views": dict(sorted(canonical_view_counts.items())),
        "mapping_pairs": [
            f"{version}|{mapping_hash}|{class_order}"
            for version, mapping_hash, class_order in sorted(pairs)
        ],
    }
    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
