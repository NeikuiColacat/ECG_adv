"""Export heterogeneous experiment JSON artifacts into ``metrics_long.csv``.

The exporter is intentionally conservative: it separates all-zero-kept and
drop-all-zero views, and refuses mixed PN2021 Super5 mapping versions unless the
caller explicitly opts out.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ecg_adv_gen.evaluation import (
    PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
    PN2021_DROP_ALL_ZERO_REFEXCLUDED,
    PN2021C_ALL_ZERO_KEPT_CORRUPTED_REFEXCLUDED,
    PN2021C_DROP_ALL_ZERO_CORRUPTED_REFEXCLUDED,
    PTBXL_FOLD10_SOURCE_FLOOR,
    TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
    TARGET_DROP_ALL_ZERO_REFEXCLUDED,
    TARGET_K500_INTERNAL_VAL,
    canonicalize_view,
)
from ecg_adv_gen.labels import default_class_order


METRICS_FIELDNAMES = [
    "run_id",
    "source_file",
    "artifact_type",
    "dataset",
    "view",
    "canonical_view",
    "scope",
    "center",
    "class_name",
    "metric",
    "value",
    "n_records",
    "n_excluded_ref",
    "n_all_zero_labels",
    "n_nonzero_labels",
    "n_classes_used",
    "n_pos",
    "n_valid",
    "mapping_version",
    "mapping_hash",
    "class_order",
]

ARTIFACT_PATTERNS = [
    "eval_result*.json",
    "eval_pn2021_c*.json",
    "eval_pn2021c*.json",
    "run_manifest.json",
    "train_result.json",
    "selection.json",
    "k500_ref_ids.json",
]


class MetricsExportError(ValueError):
    """Raised when metrics cannot be exported without paper-safety ambiguity."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetricsExportError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(data, dict):
        raise MetricsExportError(f"JSON artifact root must be an object: {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts(inputs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            candidates: list[Path] = []
            for pattern in ARTIFACT_PATTERNS:
                candidates.extend(path.rglob(pattern))
        else:
            candidates = [path]
        for candidate in sorted(c.resolve() for c in candidates if c.is_file()):
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return out


def _artifact_type(path: Path) -> str:
    name = path.name
    if name.startswith("eval_result") or name.startswith("eval_pn2021_c") or name.startswith("eval_pn2021c"):
        return "eval_result"
    if name == "run_manifest.json":
        return "run_manifest"
    if name == "train_result.json":
        return "train_result"
    if name == "training_log.json":
        return "training_log"
    if name == "selection.json":
        return "selection"
    if name == "k500_ref_ids.json":
        return "k500_ref_ids"
    return "json"


def _class_order(data: dict[str, Any]) -> list[str] | None:
    for key in ["class_names", "class_order"]:
        value = data.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    label_mapping = data.get("label_mapping")
    if isinstance(label_mapping, dict):
        pn = label_mapping.get("pn2021_super5")
        if isinstance(pn, dict) and isinstance(pn.get("class_names"), list):
            return [str(v) for v in pn["class_names"]]
    paper = data.get("paper_protocol")
    if isinstance(paper, dict) and isinstance(paper.get("class_order"), list):
        return [str(v) for v in paper["class_order"]]
    return None


def _default_class_order() -> list[str] | None:
    return default_class_order()


def _mapping_metadata(
    data: dict[str, Any],
    *,
    expected_mapping_version: str | None,
    expected_mapping_hash: str | None,
) -> tuple[str | None, str | None, str]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    label_mapping = data.get("label_mapping")
    if isinstance(label_mapping, dict) and isinstance(label_mapping.get("pn2021_super5"), dict):
        candidates.append(("label_mapping.pn2021_super5", label_mapping["pn2021_super5"]))
    for key in ["mapping", "mapping_metadata", "paper_protocol"]:
        if isinstance(data.get(key), dict):
            candidates.append((key, data[key]))

    for source, obj in candidates:
        version = obj.get("mapping_version")
        mapping_hash = obj.get("mapping_hash")
        if version or mapping_hash:
            return (
                None if version is None else str(version),
                None if mapping_hash is None else str(mapping_hash),
                source,
            )

    if expected_mapping_version or expected_mapping_hash:
        return expected_mapping_version, expected_mapping_hash, "expected_cli"
    return None, None, "missing"


def _run_id(path: Path, data: dict[str, Any]) -> str:
    for key in ["run_id", "experiment_name", "name"]:
        if data.get(key):
            return str(data[key])
    config = data.get("config")
    if isinstance(config, dict) and config.get("run_id"):
        return str(config["run_id"])
    return path.parent.name


def _infer_target_center(
    path: Path,
    data: dict[str, Any],
    *,
    target_centers: Iterable[str] | None,
) -> str | None:
    center = data.get("center")
    if center:
        return str(center)

    centers = sorted({str(c) for c in target_centers or [] if str(c)}, key=len, reverse=True)
    if not centers:
        return None
    for part in [path.parent.name, *reversed(path.parts)]:
        for candidate in centers:
            if part == candidate or part.startswith(f"{candidate}_"):
                return candidate
    return None


def _drop_all_zero_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    if stats.get("drop_all_zero_n_records") is not None:
        out["n_records"] = stats["drop_all_zero_n_records"]
        out["effective_n"] = stats["drop_all_zero_n_records"]
    if stats.get("drop_all_zero_n_classes_used") is not None:
        out["n_classes_used"] = stats["drop_all_zero_n_classes_used"]
    return out


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_metric(blocks: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value for block in blocks
        if (value := _float_or_none(block.get(key))) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _emit_metric(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    artifact_type: str,
    run_id: str,
    dataset: str,
    view: str,
    scope: str,
    metric: str,
    value: Any,
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
    center: str = "",
    class_name: str = "",
    stats: dict[str, Any] | None = None,
) -> None:
    if value is None:
        return
    stats = stats or {}
    row = {
        "run_id": run_id,
        "source_file": str(path),
        "artifact_type": artifact_type,
        "dataset": dataset,
        "view": view,
        "canonical_view": canonicalize_view(view),
        "scope": scope,
        "center": center,
        "class_name": class_name,
        "metric": metric,
        "value": value,
        "n_records": stats.get("n_records") or stats.get("effective_n") or "",
        "n_excluded_ref": stats.get("n_excluded_ref") or "",
        "n_all_zero_labels": stats.get("n_all_zero_labels") or "",
        "n_nonzero_labels": stats.get("n_nonzero_labels") or "",
        "n_classes_used": stats.get("n_classes_used") or stats.get("drop_all_zero_n_classes_used") or "",
        "n_pos": stats.get("n_pos") or "",
        "n_valid": stats.get("n_valid") or "",
        "mapping_version": mapping_version or "",
        "mapping_hash": mapping_hash or "",
        "class_order": "|".join(class_order or []),
    }
    rows.append(row)


def _emit_per_class(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    artifact_type: str,
    run_id: str,
    dataset: str,
    view: str,
    center: str,
    per_class: dict[str, Any],
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
) -> None:
    for class_name, metrics in sorted(per_class.items()):
        if not isinstance(metrics, dict):
            continue
        for metric in ["auroc", "auprc"]:
            _emit_metric(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset=dataset,
                view=view,
                scope="class",
                center=center,
                class_name=str(class_name),
                metric=metric,
                value=metrics.get(metric),
                stats=metrics,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )


def _extract_eval_crosscenter_rows(
    data: dict[str, Any],
    *,
    path: Path,
    artifact_type: str,
    run_id: str,
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ptbxl = data.get("ptbxl_test")
    if isinstance(ptbxl, dict):
        for metric in ["macro_auroc", "macro_auprc"]:
            _emit_metric(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset="ptbxl",
                view=PTBXL_FOLD10_SOURCE_FLOOR,
                scope="dataset",
                metric=metric,
                value=ptbxl.get(metric),
                stats=ptbxl,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        if isinstance(ptbxl.get("per_class"), dict):
            _emit_per_class(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset="ptbxl",
                view="ptbxl_fold10_source_floor",
                center="",
                per_class=ptbxl["per_class"],
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )

    pn = data.get("pn2021")
    if not isinstance(pn, dict):
        return rows

    for metric in ["avg_macro_auroc", "avg_macro_auprc"]:
        _emit_metric(
            rows,
            path=path,
            artifact_type=artifact_type,
            run_id=run_id,
            dataset="pn2021",
            view=PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
            scope="dataset",
            metric=metric,
            value=pn.get(metric),
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
            class_order=class_order,
        )
    for metric in ["avg_drop_all_zero_macro_auroc", "avg_drop_all_zero_macro_auprc"]:
        _emit_metric(
            rows,
            path=path,
            artifact_type=artifact_type,
            run_id=run_id,
            dataset="pn2021",
            view=PN2021_DROP_ALL_ZERO_REFEXCLUDED,
            scope="dataset",
            metric=metric,
            value=pn.get(metric),
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
            class_order=class_order,
        )

    per_center = pn.get("per_center")
    if isinstance(per_center, dict):
        for center, metrics in sorted(per_center.items()):
            if not isinstance(metrics, dict):
                continue
            for metric in ["macro_auroc", "macro_auprc"]:
                _emit_metric(
                    rows,
                    path=path,
                    artifact_type=artifact_type,
                    run_id=run_id,
                    dataset="pn2021",
                    view=PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
                    scope="center",
                    center=str(center),
                    metric=metric,
                    value=metrics.get(metric),
                    stats=metrics,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    class_order=class_order,
                )
            if isinstance(metrics.get("per_class"), dict):
                _emit_per_class(
                    rows,
                    path=path,
                    artifact_type=artifact_type,
                    run_id=run_id,
                    dataset="pn2021",
                    view=PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
                    center=str(center),
                    per_class=metrics["per_class"],
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    class_order=class_order,
                )
            for metric in ["drop_all_zero_macro_auroc", "drop_all_zero_macro_auprc"]:
                _emit_metric(
                    rows,
                    path=path,
                    artifact_type=artifact_type,
                    run_id=run_id,
                    dataset="pn2021",
                    view=PN2021_DROP_ALL_ZERO_REFEXCLUDED,
                    scope="center",
                    center=str(center),
                    metric=metric,
                    value=metrics.get(metric),
                    stats=_drop_all_zero_stats(metrics),
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    class_order=class_order,
                )
            if isinstance(metrics.get("drop_all_zero_per_class"), dict):
                _emit_per_class(
                    rows,
                    path=path,
                    artifact_type=artifact_type,
                    run_id=run_id,
                    dataset="pn2021",
                    view=PN2021_DROP_ALL_ZERO_REFEXCLUDED,
                    center=str(center),
                    per_class=metrics["drop_all_zero_per_class"],
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    class_order=class_order,
                )
    return rows


def _pn2021c_metric_blocks(center_payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for severity_map in center_payload.values():
        if not isinstance(severity_map, dict):
            continue
        for metrics in severity_map.values():
            if isinstance(metrics, dict) and (
                metrics.get("macro_auroc") is not None
                or metrics.get("macro_auprc") is not None
            ):
                blocks.append(metrics)
    return blocks


def _extract_pn2021c_rows(
    data: dict[str, Any],
    *,
    path: Path,
    artifact_type: str,
    run_id: str,
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
) -> list[dict[str, Any]]:
    per_center = data.get("per_center")
    if not isinstance(per_center, dict):
        return []

    rows: list[dict[str, Any]] = []
    for center, center_payload in sorted(per_center.items()):
        if not isinstance(center_payload, dict):
            continue
        blocks = _pn2021c_metric_blocks(center_payload)
        if not blocks:
            continue
        stats = blocks[0]
        for metric in ("macro_auroc", "macro_auprc"):
            _emit_metric(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset="pn2021c",
                view=PN2021C_ALL_ZERO_KEPT_CORRUPTED_REFEXCLUDED,
                scope="center",
                center=str(center),
                metric=metric,
                value=_mean_metric(blocks, metric),
                stats=stats,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        for metric in ("drop_all_zero_macro_auroc", "drop_all_zero_macro_auprc"):
            _emit_metric(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset="pn2021c",
                view=PN2021C_DROP_ALL_ZERO_CORRUPTED_REFEXCLUDED,
                scope="center",
                center=str(center),
                metric=metric,
                value=_mean_metric(blocks, metric),
                stats=_drop_all_zero_stats(stats),
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
    return rows


def _extract_named_metric_block(
    rows: list[dict[str, Any]],
    *,
    data: dict[str, Any],
    key: str,
    dataset: str,
    view: str,
    center: str,
    path: Path,
    artifact_type: str,
    run_id: str,
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
) -> None:
    block = data.get(key)
    if not isinstance(block, dict):
        return
    for metric in ["macro_auroc", "macro_auprc"]:
        _emit_metric(
            rows,
            path=path,
            artifact_type=artifact_type,
            run_id=run_id,
            dataset=dataset,
            view=view,
            scope="center" if center else "dataset",
            center=center,
            metric=metric,
            value=block.get(metric),
            stats=block,
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
            class_order=class_order,
        )
    if isinstance(block.get("per_class"), dict):
        _emit_per_class(
            rows,
            path=path,
            artifact_type=artifact_type,
            run_id=run_id,
            dataset=dataset,
            view=view,
            center=center,
            per_class=block["per_class"],
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
            class_order=class_order,
        )


def _extract_single_run_rows(
    data: dict[str, Any],
    *,
    path: Path,
    artifact_type: str,
    run_id: str,
    mapping_version: str | None,
    mapping_hash: str | None,
    class_order: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    center = str(data.get("center") or "")
    blocks = [
        ("ptbxl_fold10", "ptbxl", PTBXL_FOLD10_SOURCE_FLOOR, ""),
        ("final_ptbxl_fold10", "ptbxl", PTBXL_FOLD10_SOURCE_FLOOR, ""),
        ("target_val", "pn2021", TARGET_K500_INTERNAL_VAL, center),
        ("target_excluding_ref", "pn2021", TARGET_ALL_ZERO_KEPT_REFEXCLUDED, center),
        ("target_drop_all_zero_excluding_ref", "pn2021", TARGET_DROP_ALL_ZERO_REFEXCLUDED, center),
    ]
    for key, dataset, view, block_center in blocks:
        _extract_named_metric_block(
            rows,
            data=data,
            key=key,
            dataset=dataset,
            view=view,
            center=block_center,
            path=path,
            artifact_type=artifact_type,
            run_id=run_id,
            mapping_version=mapping_version,
            mapping_hash=mapping_hash,
            class_order=class_order,
        )
    target_view = data.get("target_view")
    if isinstance(target_view, dict):
        for metric in ["avg_macro_auroc", "avg_macro_auprc"]:
            _emit_metric(
                rows,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                dataset="pn2021",
                view=TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
                scope="dataset",
                metric=metric,
                value=target_view.get(metric),
                stats=target_view,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        per_center = target_view.get("per_center")
        if isinstance(per_center, dict):
            for center_name, metrics in sorted(per_center.items()):
                if not isinstance(metrics, dict):
                    continue
                for metric in ["macro_auroc", "macro_auprc"]:
                    _emit_metric(
                        rows,
                        path=path,
                        artifact_type=artifact_type,
                        run_id=run_id,
                        dataset="pn2021",
                        view=TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
                        scope="center",
                        center=str(center_name),
                        metric=metric,
                        value=metrics.get(metric),
                        stats=metrics,
                        mapping_version=mapping_version,
                        mapping_hash=mapping_hash,
                        class_order=class_order,
                    )
                if isinstance(metrics.get("per_class"), dict):
                    _emit_per_class(
                        rows,
                        path=path,
                        artifact_type=artifact_type,
                        run_id=run_id,
                        dataset="pn2021",
                        view=TARGET_ALL_ZERO_KEPT_REFEXCLUDED,
                        center=str(center_name),
                        per_class=metrics["per_class"],
                        mapping_version=mapping_version,
                        mapping_hash=mapping_hash,
                        class_order=class_order,
                    )

    final_views = data.get("final_pn2021_views")
    if isinstance(final_views, dict):
        view_payload = final_views.get(center) if center else None
        if not isinstance(view_payload, dict) and len(final_views) == 1:
            only_payload = next(iter(final_views.values()))
            if isinstance(only_payload, dict):
                view_payload = only_payload
        if isinstance(view_payload, dict):
            rows.extend(
                _extract_eval_crosscenter_rows(
                    {"pn2021": view_payload},
                    path=path,
                    artifact_type=artifact_type,
                    run_id=run_id,
                    mapping_version=mapping_version,
                    mapping_hash=mapping_hash,
                    class_order=class_order,
                )
            )
    return rows


def _filter_rows_to_target_center(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    target_center: str | None,
) -> list[dict[str, Any]]:
    if not target_center:
        raise MetricsExportError(f"Could not infer target center for artifact: {path}")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("center") == target_center:
            filtered.append(row)
        elif (
            row.get("dataset") == "ptbxl"
            and row.get("view") == PTBXL_FOLD10_SOURCE_FLOOR
            and not row.get("center")
        ):
            filtered.append({**row, "center": target_center, "scope": "center"})
    return filtered


def _extract_rows(
    path: Path,
    data: dict[str, Any],
    *,
    expected_mapping_version: str | None,
    expected_mapping_hash: str | None,
    run_id_override: str | None = None,
    filter_to_target_center: bool = False,
    target_centers: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact_type = _artifact_type(path)
    mapping_version, mapping_hash, mapping_source = _mapping_metadata(
        data,
        expected_mapping_version=expected_mapping_version,
        expected_mapping_hash=expected_mapping_hash,
    )
    class_order = _class_order(data)
    if class_order is None and (mapping_version or mapping_hash):
        class_order = _default_class_order()
    source_run_id = _run_id(path, data)
    run_id = str(run_id_override) if run_id_override else source_run_id
    target_center = _infer_target_center(path, data, target_centers=target_centers)

    rows: list[dict[str, Any]] = []
    if artifact_type == "eval_result":
        rows.extend(
            _extract_eval_crosscenter_rows(
                data,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        )
        rows.extend(
            _extract_pn2021c_rows(
                data,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        )
        rows.extend(
            _extract_single_run_rows(
                data,
                path=path,
                artifact_type=artifact_type,
                run_id=run_id,
                mapping_version=mapping_version,
                mapping_hash=mapping_hash,
                class_order=class_order,
            )
        )

    if filter_to_target_center:
        rows = _filter_rows_to_target_center(rows, path=path, target_center=target_center)

    artifact = {
        "path": str(path),
        "artifact_type": artifact_type,
        "sha256": _sha256_file(path),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "run_id_override": run_id_override or "",
        "target_center": target_center or "",
        "filter_to_target_center": bool(filter_to_target_center),
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
        "mapping_source": mapping_source,
        "class_order": class_order or [],
        "metrics_rows": len(rows),
    }
    return rows, artifact


def _validate_mapping_consistency(
    artifacts: list[dict[str, Any]],
    *,
    allow_mixed_mapping: bool,
    require_mapping: bool,
) -> None:
    pairs = {
        (a.get("mapping_version"), a.get("mapping_hash"))
        for a in artifacts
        if a.get("metrics_rows", 0) > 0
    }
    known_pairs = {p for p in pairs if p != (None, None)}
    if require_mapping:
        missing = [
            a["path"]
            for a in artifacts
            if a.get("metrics_rows", 0) > 0
            and (not a.get("mapping_version") or not a.get("mapping_hash") or not a.get("class_order"))
        ]
        if missing:
            raise MetricsExportError(f"Missing complete mapping metadata for metrics artifacts: {missing[:5]}")
    if not allow_mixed_mapping and len(known_pairs) > 1:
        raise MetricsExportError(f"Refusing to mix mapping versions/hashes: {sorted(known_pairs)}")


def export_metrics(
    inputs: Iterable[Path],
    output_dir: Path,
    *,
    expected_mapping_version: str | None = None,
    expected_mapping_hash: str | None = None,
    allow_mixed_mapping: bool = False,
    require_mapping: bool = True,
    run_id_override: str | None = None,
    filter_to_target_center: bool = False,
    target_centers: Iterable[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_paths = collect_artifacts(inputs)
    if not artifacts_paths:
        raise MetricsExportError("No artifact files found")

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for path in artifacts_paths:
        data = _load_json(path)
        extracted, artifact = _extract_rows(
            path,
            data,
            expected_mapping_version=expected_mapping_version,
            expected_mapping_hash=expected_mapping_hash,
            run_id_override=run_id_override,
            filter_to_target_center=filter_to_target_center,
            target_centers=target_centers,
        )
        rows.extend(extracted)
        artifacts.append(artifact)

    _validate_mapping_consistency(
        artifacts,
        allow_mixed_mapping=allow_mixed_mapping,
        require_mapping=require_mapping,
    )

    metrics_path = output_dir / "metrics_long.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    view_counts = Counter(row["view"] for row in rows)
    canonical_view_counts = Counter(row["canonical_view"] for row in rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_long": str(metrics_path),
        "artifact_manifest": str(output_dir / "artifact_manifest.json"),
        "n_artifacts": len(artifacts),
        "n_metric_rows": len(rows),
        "run_id_override": run_id_override or "",
        "filter_to_target_center": bool(filter_to_target_center),
        "target_centers": sorted({str(c) for c in target_centers or [] if str(c)}),
        "views": dict(sorted(view_counts.items())),
        "canonical_views": dict(sorted(canonical_view_counts.items())),
        "mapping_pairs": sorted(
            {
                f"{a.get('mapping_version') or ''}|{a.get('mapping_hash') or ''}"
                for a in artifacts
                if a.get("metrics_rows", 0) > 0
            }
        ),
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
