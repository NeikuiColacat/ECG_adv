"""Build compact paper tables from ``metrics_long.csv``."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ecg_adv_gen.evaluation import canonicalize_view, normalize_macro_metric


TABLE_FIELDNAMES = [
    "run_id",
    "dataset",
    "view",
    "raw_views",
    "scope",
    "center_group",
    "n_centers",
    "macro_auroc",
    "macro_auprc",
    "delta_macro_auroc_vs_baseline",
    "delta_macro_auprc_vs_baseline",
    "baseline_run_id",
    "mapping_version",
    "mapping_hash",
    "class_order",
    "source_files",
]


class PaperTableError(ValueError):
    """Raised when a paper table would mix incompatible metric semantics."""


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PaperTableError(f"No rows in metrics_long CSV: {path}")
    return rows


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _normalize_macro_metric(metric: str) -> str | None:
    return normalize_macro_metric(metric)


def _validate_single_mapping(rows: list[dict[str, str]], *, allow_mixed_mapping: bool) -> tuple[str, str, str]:
    pairs = {
        (row.get("mapping_version", ""), row.get("mapping_hash", ""), row.get("class_order", ""))
        for row in rows
        if row.get("mapping_version") or row.get("mapping_hash")
    }
    if not pairs:
        raise PaperTableError("No mapping metadata in selected metrics rows")
    if not allow_mixed_mapping and len(pairs) > 1:
        raise PaperTableError(f"Refusing mixed mapping metadata in one paper table: {sorted(pairs)}")
    version, mapping_hash, class_order = sorted(pairs)[0]
    return version, mapping_hash, class_order


def _filter_rows(
    rows: list[dict[str, str]],
    *,
    dataset: str | None,
    view: str,
) -> list[dict[str, str]]:
    requested_view = canonicalize_view(view)
    out = []
    for row in rows:
        raw_view = row.get("view", "")
        canonical_view = row.get("canonical_view") or canonicalize_view(raw_view)
        if canonical_view != requested_view:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if row.get("class_name"):
            continue
        if _normalize_macro_metric(row.get("metric", "")) is None:
            continue
        normalized = dict(row)
        normalized["raw_view"] = raw_view
        normalized["view"] = canonical_view
        normalized["canonical_view"] = canonical_view
        out.append(normalized)
    if not out:
        raise PaperTableError(f"No macro rows found for view={view!r} dataset={dataset!r}")
    return out


def _source_files(rows: Iterable[dict[str, str]]) -> str:
    return "|".join(sorted({row.get("source_file", "") for row in rows if row.get("source_file")}))


def _raw_views(rows: Iterable[dict[str, str]]) -> str:
    return "|".join(sorted({row.get("raw_view") or row.get("view", "") for row in rows if row.get("view")}))


def _build_dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[tuple[float, dict[str, str]]]]] = defaultdict(lambda: defaultdict(list))
    row_lookup: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("scope") != "dataset":
            continue
        metric = _normalize_macro_metric(row.get("metric", ""))
        value = _float_or_none(row.get("value"))
        if metric is None or value is None:
            continue
        key = (row["run_id"], row["dataset"], row["view"])
        grouped[key][metric].append((value, row))
        row_lookup[key].append(row)

    out = []
    for (run_id, dataset, view), metrics in sorted(grouped.items()):
        auroc = _mean(v for v, _ in metrics.get("macro_auroc", []))
        auprc = _mean(v for v, _ in metrics.get("macro_auprc", []))
        out.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "view": view,
                "raw_views": _raw_views(row_lookup[(run_id, dataset, view)]),
                "scope": "dataset",
                "center_group": "",
                "n_centers": "",
                "macro_auroc": auroc,
                "macro_auprc": auprc,
                "source_files": _source_files(row_lookup[(run_id, dataset, view)]),
            }
        )
    return out


def _build_center_mean_rows(rows: list[dict[str, str]], centers: list[str] | None) -> list[dict[str, Any]]:
    allowed = set(centers or [])
    grouped: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    row_lookup: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("scope") != "center":
            continue
        center = row.get("center", "")
        if centers is not None and center not in allowed:
            continue
        metric = _normalize_macro_metric(row.get("metric", ""))
        value = _float_or_none(row.get("value"))
        if metric is None or value is None:
            continue
        key = (row["run_id"], row["dataset"], row["view"])
        grouped[key][center][metric] = value
        row_lookup[key].append(row)

    out = []
    for (run_id, dataset, view), center_metrics in sorted(grouped.items()):
        if centers is not None:
            missing = [center for center in centers if center not in center_metrics]
            if missing:
                raise PaperTableError(f"Run {run_id!r} is missing centers for requested mean: {missing}")
            ordered_centers = centers
        else:
            ordered_centers = sorted(center_metrics)
        missing_metrics = {
            center: [
                metric
                for metric in ["macro_auroc", "macro_auprc"]
                if metric not in center_metrics[center]
            ]
            for center in ordered_centers
            if "macro_auroc" not in center_metrics[center]
            or "macro_auprc" not in center_metrics[center]
        }
        if missing_metrics:
            raise PaperTableError(
                f"Run {run_id!r} is missing center metrics for requested mean: {missing_metrics}"
            )
        auroc = _mean(
            center_metrics[c]["macro_auroc"]
            for c in ordered_centers
        )
        auprc = _mean(
            center_metrics[c]["macro_auprc"]
            for c in ordered_centers
        )
        out.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "view": view,
                "raw_views": _raw_views(row_lookup[(run_id, dataset, view)]),
                "scope": "center_mean",
                "center_group": "|".join(ordered_centers),
                "n_centers": len(ordered_centers),
                "macro_auroc": auroc,
                "macro_auprc": auprc,
                "source_files": _source_files(row_lookup[(run_id, dataset, view)]),
            }
        )
    return out


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _add_baseline_deltas(rows: list[dict[str, Any]], baseline_run_id: str | None) -> None:
    if not baseline_run_id:
        for row in rows:
            row["baseline_run_id"] = ""
            row["delta_macro_auroc_vs_baseline"] = ""
            row["delta_macro_auprc_vs_baseline"] = ""
        return

    baseline_by_key = {
        (row["dataset"], row["view"], row["scope"], row["center_group"]): row
        for row in rows
        if row["run_id"] == baseline_run_id
    }
    if not baseline_by_key:
        raise PaperTableError(f"baseline_run_id not found in table rows: {baseline_run_id}")

    for row in rows:
        key = (row["dataset"], row["view"], row["scope"], row["center_group"])
        base = baseline_by_key.get(key)
        if base is None:
            raise PaperTableError(f"No matching baseline row for {row['run_id']} key={key}")
        row["baseline_run_id"] = baseline_run_id
        row["delta_macro_auroc_vs_baseline"] = (
            None if row["macro_auroc"] is None or base["macro_auroc"] is None
            else row["macro_auroc"] - base["macro_auroc"]
        )
        row["delta_macro_auprc_vs_baseline"] = (
            None if row["macro_auprc"] is None or base["macro_auprc"] is None
            else row["macro_auprc"] - base["macro_auprc"]
        )


def export_paper_table(
    metrics_long: Path,
    output_dir: Path,
    *,
    view: str,
    dataset: str | None = None,
    centers: list[str] | None = None,
    include_dataset_rows: bool = True,
    include_center_mean: bool = True,
    baseline_run_id: str | None = None,
    allow_mixed_mapping: bool = False,
) -> dict[str, Any]:
    if not view or not str(view).strip():
        raise PaperTableError("view must be explicitly specified")
    rows = _read_metrics(metrics_long)
    filtered = _filter_rows(rows, dataset=dataset, view=view)
    mapping_version, mapping_hash, class_order = _validate_single_mapping(
        filtered,
        allow_mixed_mapping=allow_mixed_mapping,
    )

    table_rows: list[dict[str, Any]] = []
    if include_dataset_rows:
        table_rows.extend(_build_dataset_rows(filtered))
    if include_center_mean:
        table_rows.extend(_build_center_mean_rows(filtered, centers))
    if not table_rows:
        raise PaperTableError("No table rows produced")
    _add_baseline_deltas(table_rows, baseline_run_id)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "paper_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDNAMES)
        writer.writeheader()
        for row in table_rows:
            out = {key: "" for key in TABLE_FIELDNAMES}
            out.update(row)
            out["mapping_version"] = mapping_version
            out["mapping_hash"] = mapping_hash
            out["class_order"] = class_order
            for key, value in list(out.items()):
                out[key] = _format_value(value)
            writer.writerow(out)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_long": str(Path(metrics_long).expanduser().resolve()),
        "paper_table": str(table_path),
        "paper_table_manifest": str(output_dir / "paper_table_manifest.json"),
        "view": canonicalize_view(view),
        "dataset": dataset or "",
        "centers": centers or [],
        "include_dataset_rows": include_dataset_rows,
        "include_center_mean": include_center_mean,
        "baseline_run_id": baseline_run_id or "",
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
        "class_order": class_order,
        "n_input_metric_rows": len(filtered),
        "n_table_rows": len(table_rows),
    }
    manifest_path = output_dir / "paper_table_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
