#!/usr/bin/env python3
"""Export paired PN2021-C recovery summaries for noAug vs VAE-LHAT runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EVAL_FILENAME = "eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json"
DEFAULT_BASELINE_METHOD = "vae_noaug"
DEFAULT_CANDIDATE_METHOD = "vae_lhat"
DEFAULT_SUCCESS_AUPRC_GAIN = 0.06


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _get_center_record(data: dict[str, Any], center: str, corruption: str, severity: int) -> dict[str, Any]:
    try:
        return data["per_center"][center][corruption][str(severity)]
    except KeyError as exc:
        raise KeyError(f"missing {center}/{corruption}/s{severity}") from exc


def _clean_drop_all_zero_metric(record: dict[str, Any], center: str, metric: str) -> float | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    center_meta = (
        metadata.get("pn2021", {})
        .get("per_center", {})
        .get(center, {})
    )
    return _safe_float(center_meta.get(f"drop_all_zero_macro_{metric}"))


def _clean_per_class(record: dict[str, Any], center: str, view: str) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    center_meta = (
        metadata.get("pn2021", {})
        .get("per_center", {})
        .get(center, {})
    )
    if view == "macro":
        value = center_meta.get("per_class", {})
    elif view == "drop_all_zero":
        value = center_meta.get("drop_all_zero_per_class", {})
    else:
        raise ValueError(f"unknown view: {view}")
    return value if isinstance(value, dict) else {}


def _corrupt_per_class(record: dict[str, Any], view: str) -> dict[str, Any]:
    if view == "macro":
        value = record.get("per_class", {})
    elif view == "drop_all_zero":
        value = record.get("drop_all_zero_per_class", {})
    else:
        raise ValueError(f"unknown view: {view}")
    return value if isinstance(value, dict) else {}


def _metrics_for_view(record: dict[str, Any], center: str, view: str) -> dict[str, float | None]:
    if view == "macro":
        clean_auroc = _safe_float(record.get("clean_macro_auroc"))
        clean_auprc = _safe_float(record.get("clean_macro_auprc"))
        corrupt_auroc = _safe_float(record.get("macro_auroc"))
        corrupt_auprc = _safe_float(record.get("macro_auprc"))
        drop_auroc = _safe_float(record.get("auroc_drop_vs_clean"))
        drop_auprc = _safe_float(record.get("auprc_drop_vs_clean"))
        return {
            "clean_auroc": clean_auroc,
            "clean_auprc": clean_auprc,
            "corrupt_auroc": corrupt_auroc,
            "corrupt_auprc": corrupt_auprc,
            "drop_auroc": drop_auroc,
            "drop_auprc": drop_auprc,
        }
    if view == "drop_all_zero":
        clean_auroc = _clean_drop_all_zero_metric(record, center, "auroc")
        clean_auprc = _clean_drop_all_zero_metric(record, center, "auprc")
        corrupt_auroc = _safe_float(record.get("drop_all_zero_macro_auroc"))
        corrupt_auprc = _safe_float(record.get("drop_all_zero_macro_auprc"))
        return {
            "clean_auroc": clean_auroc,
            "clean_auprc": clean_auprc,
            "corrupt_auroc": corrupt_auroc,
            "corrupt_auprc": corrupt_auprc,
            "drop_auroc": (
                clean_auroc - corrupt_auroc
                if clean_auroc is not None and corrupt_auroc is not None
                else None
            ),
            "drop_auprc": (
                clean_auprc - corrupt_auprc
                if clean_auprc is not None and corrupt_auprc is not None
                else None
            ),
        }
    raise ValueError(f"unknown view: {view}")


def _class_metrics_for_view(
    record: dict[str, Any],
    center: str,
    view: str,
    class_name: str,
) -> dict[str, float | None]:
    clean_row = _clean_per_class(record, center, view).get(class_name, {})
    corrupt_row = _corrupt_per_class(record, view).get(class_name, {})
    if not isinstance(clean_row, dict):
        clean_row = {}
    if not isinstance(corrupt_row, dict):
        corrupt_row = {}
    clean_auroc = _safe_float(clean_row.get("auroc"))
    clean_auprc = _safe_float(clean_row.get("auprc"))
    corrupt_auroc = _safe_float(corrupt_row.get("auroc"))
    corrupt_auprc = _safe_float(corrupt_row.get("auprc"))
    return {
        "clean_auroc": clean_auroc,
        "clean_auprc": clean_auprc,
        "corrupt_auroc": corrupt_auroc,
        "corrupt_auprc": corrupt_auprc,
        "drop_auroc": (
            clean_auroc - corrupt_auroc
            if clean_auroc is not None and corrupt_auroc is not None
            else None
        ),
        "drop_auprc": (
            clean_auprc - corrupt_auprc
            if clean_auprc is not None and corrupt_auprc is not None
            else None
        ),
        "n_pos": _safe_float(corrupt_row.get("n_pos")),
        "n_valid": _safe_float(corrupt_row.get("n_valid")),
    }


def _mapping_id(data: dict[str, Any]) -> tuple[Any, Any]:
    mapping = data.get("label_mapping") or {}
    return mapping.get("version"), mapping.get("hash")


def _collect_centers(run_root: Path, baseline_method: str, candidate_method: str, eval_filename: str) -> list[str]:
    centers: list[str] = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        baseline = path / baseline_method / eval_filename
        candidate = path / candidate_method / eval_filename
        if baseline.exists() and candidate.exists():
            centers.append(path.name)
    return centers


def build_recovery_rows(
    *,
    run_root: Path,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    eval_filename: str = DEFAULT_EVAL_FILENAME,
    centers: Iterable[str] | None = None,
    corruptions: Iterable[str] | None = None,
    severity: int = 5,
) -> list[dict[str, Any]]:
    centers_list = list(centers) if centers is not None else _collect_centers(
        run_root, baseline_method, candidate_method, eval_filename
    )
    if not centers_list:
        raise ValueError(f"no paired center outputs found under {run_root}")

    rows: list[dict[str, Any]] = []
    expected_mapping: tuple[Any, Any] | None = None
    for center in centers_list:
        baseline_path = run_root / center / baseline_method / eval_filename
        candidate_path = run_root / center / candidate_method / eval_filename
        if not baseline_path.exists():
            raise FileNotFoundError(baseline_path)
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        baseline_data = _load_json(baseline_path)
        candidate_data = _load_json(candidate_path)
        for data, path in ((baseline_data, baseline_path), (candidate_data, candidate_path)):
            mapping_id = _mapping_id(data)
            if expected_mapping is None:
                expected_mapping = mapping_id
            elif mapping_id != expected_mapping:
                raise ValueError(
                    f"mapping mismatch for {path}: expected={expected_mapping!r}, got={mapping_id!r}"
                )

        corruption_list = (
            list(corruptions)
            if corruptions is not None
            else sorted(set(baseline_data["per_center"][center]) & set(candidate_data["per_center"][center]))
        )
        for corruption in corruption_list:
            baseline_record = _get_center_record(baseline_data, center, corruption, severity)
            candidate_record = _get_center_record(candidate_data, center, corruption, severity)
            for view in ("macro", "drop_all_zero"):
                base = _metrics_for_view(baseline_record, center, view)
                cand = _metrics_for_view(candidate_record, center, view)
                corrupt_delta_auroc = (
                    cand["corrupt_auroc"] - base["corrupt_auroc"]
                    if cand["corrupt_auroc"] is not None and base["corrupt_auroc"] is not None
                    else None
                )
                corrupt_delta_auprc = (
                    cand["corrupt_auprc"] - base["corrupt_auprc"]
                    if cand["corrupt_auprc"] is not None and base["corrupt_auprc"] is not None
                    else None
                )
                clean_delta_auroc = (
                    cand["clean_auroc"] - base["clean_auroc"]
                    if cand["clean_auroc"] is not None and base["clean_auroc"] is not None
                    else None
                )
                clean_delta_auprc = (
                    cand["clean_auprc"] - base["clean_auprc"]
                    if cand["clean_auprc"] is not None and base["clean_auprc"] is not None
                    else None
                )
                drop_reduction_auroc = (
                    base["drop_auroc"] - cand["drop_auroc"]
                    if base["drop_auroc"] is not None and cand["drop_auroc"] is not None
                    else None
                )
                drop_reduction_auprc = (
                    base["drop_auprc"] - cand["drop_auprc"]
                    if base["drop_auprc"] is not None and cand["drop_auprc"] is not None
                    else None
                )
                rows.append({
                    "center": center,
                    "corruption": corruption,
                    "severity": int(severity),
                    "view": view,
                    "n_records": baseline_record.get("n_records"),
                    "clean_metric_source": baseline_record.get("clean_metric_source"),
                    "baseline_method": baseline_method,
                    "candidate_method": candidate_method,
                    "baseline_clean_auroc": base["clean_auroc"],
                    "baseline_clean_auprc": base["clean_auprc"],
                    "baseline_corrupt_auroc": base["corrupt_auroc"],
                    "baseline_corrupt_auprc": base["corrupt_auprc"],
                    "baseline_drop_auroc": base["drop_auroc"],
                    "baseline_drop_auprc": base["drop_auprc"],
                    "candidate_clean_auroc": cand["clean_auroc"],
                    "candidate_clean_auprc": cand["clean_auprc"],
                    "candidate_corrupt_auroc": cand["corrupt_auroc"],
                    "candidate_corrupt_auprc": cand["corrupt_auprc"],
                    "candidate_drop_auroc": cand["drop_auroc"],
                    "candidate_drop_auprc": cand["drop_auprc"],
                    "candidate_minus_baseline_clean_auroc": clean_delta_auroc,
                    "candidate_minus_baseline_clean_auprc": clean_delta_auprc,
                    "candidate_minus_baseline_corrupt_auroc": corrupt_delta_auroc,
                    "candidate_minus_baseline_corrupt_auprc": corrupt_delta_auprc,
                    "candidate_drop_reduction_auroc": drop_reduction_auroc,
                    "candidate_drop_reduction_auprc": drop_reduction_auprc,
                    "candidate_corrupt_auroc_recovery_fraction": _ratio(
                        corrupt_delta_auroc, base["drop_auroc"]
                    ),
                    "candidate_corrupt_auprc_recovery_fraction": _ratio(
                        corrupt_delta_auprc, base["drop_auprc"]
                    ),
                    "candidate_drop_reduction_auroc_fraction": _ratio(
                        drop_reduction_auroc, base["drop_auroc"]
                    ),
                    "candidate_drop_reduction_auprc_fraction": _ratio(
                        drop_reduction_auprc, base["drop_auprc"]
                    ),
                    "baseline_eval_json": baseline_path.as_posix(),
                    "candidate_eval_json": candidate_path.as_posix(),
                })
    return rows


def build_per_class_rows(
    *,
    run_root: Path,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    eval_filename: str = DEFAULT_EVAL_FILENAME,
    centers: Iterable[str] | None = None,
    corruptions: Iterable[str] | None = None,
    severity: int = 5,
) -> list[dict[str, Any]]:
    centers_list = list(centers) if centers is not None else _collect_centers(
        run_root, baseline_method, candidate_method, eval_filename
    )
    if not centers_list:
        raise ValueError(f"no paired center outputs found under {run_root}")

    rows: list[dict[str, Any]] = []
    for center in centers_list:
        baseline_path = run_root / center / baseline_method / eval_filename
        candidate_path = run_root / center / candidate_method / eval_filename
        baseline_data = _load_json(baseline_path)
        candidate_data = _load_json(candidate_path)
        corruption_list = (
            list(corruptions)
            if corruptions is not None
            else sorted(set(baseline_data["per_center"][center]) & set(candidate_data["per_center"][center]))
        )
        for corruption in corruption_list:
            baseline_record = _get_center_record(baseline_data, center, corruption, severity)
            candidate_record = _get_center_record(candidate_data, center, corruption, severity)
            for view in ("macro", "drop_all_zero"):
                class_names = sorted(
                    set(_clean_per_class(baseline_record, center, view))
                    | set(_corrupt_per_class(baseline_record, view))
                    | set(_clean_per_class(candidate_record, center, view))
                    | set(_corrupt_per_class(candidate_record, view))
                )
                for class_name in class_names:
                    base = _class_metrics_for_view(baseline_record, center, view, class_name)
                    cand = _class_metrics_for_view(candidate_record, center, view, class_name)
                    corrupt_delta_auroc = (
                        cand["corrupt_auroc"] - base["corrupt_auroc"]
                        if cand["corrupt_auroc"] is not None and base["corrupt_auroc"] is not None
                        else None
                    )
                    corrupt_delta_auprc = (
                        cand["corrupt_auprc"] - base["corrupt_auprc"]
                        if cand["corrupt_auprc"] is not None and base["corrupt_auprc"] is not None
                        else None
                    )
                    drop_reduction_auroc = (
                        base["drop_auroc"] - cand["drop_auroc"]
                        if base["drop_auroc"] is not None and cand["drop_auroc"] is not None
                        else None
                    )
                    drop_reduction_auprc = (
                        base["drop_auprc"] - cand["drop_auprc"]
                        if base["drop_auprc"] is not None and cand["drop_auprc"] is not None
                        else None
                    )
                    rows.append({
                        "center": center,
                        "corruption": corruption,
                        "severity": int(severity),
                        "view": view,
                        "class": class_name,
                        "baseline_clean_auroc": base["clean_auroc"],
                        "baseline_clean_auprc": base["clean_auprc"],
                        "baseline_corrupt_auroc": base["corrupt_auroc"],
                        "baseline_corrupt_auprc": base["corrupt_auprc"],
                        "baseline_drop_auroc": base["drop_auroc"],
                        "baseline_drop_auprc": base["drop_auprc"],
                        "candidate_clean_auroc": cand["clean_auroc"],
                        "candidate_clean_auprc": cand["clean_auprc"],
                        "candidate_corrupt_auroc": cand["corrupt_auroc"],
                        "candidate_corrupt_auprc": cand["corrupt_auprc"],
                        "candidate_drop_auroc": cand["drop_auroc"],
                        "candidate_drop_auprc": cand["drop_auprc"],
                        "candidate_minus_baseline_corrupt_auroc": corrupt_delta_auroc,
                        "candidate_minus_baseline_corrupt_auprc": corrupt_delta_auprc,
                        "candidate_drop_reduction_auroc": drop_reduction_auroc,
                        "candidate_drop_reduction_auprc": drop_reduction_auprc,
                        "candidate_corrupt_auroc_recovery_fraction": _ratio(
                            corrupt_delta_auroc, base["drop_auroc"]
                        ),
                        "candidate_corrupt_auprc_recovery_fraction": _ratio(
                            corrupt_delta_auprc, base["drop_auprc"]
                        ),
                        "baseline_n_pos": base["n_pos"],
                        "baseline_n_valid": base["n_valid"],
                        "candidate_n_pos": cand["n_pos"],
                        "candidate_n_valid": cand["n_valid"],
                    })
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    numeric_fields = [
        "baseline_clean_auroc",
        "baseline_clean_auprc",
        "baseline_corrupt_auroc",
        "baseline_corrupt_auprc",
        "baseline_drop_auroc",
        "baseline_drop_auprc",
        "candidate_clean_auroc",
        "candidate_clean_auprc",
        "candidate_corrupt_auroc",
        "candidate_corrupt_auprc",
        "candidate_drop_auroc",
        "candidate_drop_auprc",
        "candidate_minus_baseline_clean_auroc",
        "candidate_minus_baseline_clean_auprc",
        "candidate_minus_baseline_corrupt_auroc",
        "candidate_minus_baseline_corrupt_auprc",
        "candidate_drop_reduction_auroc",
        "candidate_drop_reduction_auprc",
        "candidate_corrupt_auroc_recovery_fraction",
        "candidate_corrupt_auprc_recovery_fraction",
        "candidate_drop_reduction_auroc_fraction",
        "candidate_drop_reduction_auprc_fraction",
        "baseline_n_pos",
        "baseline_n_valid",
        "candidate_n_pos",
        "candidate_n_valid",
    ]
    out: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["n_rows"] = len(group_rows)
        item["n_centers"] = len({row["center"] for row in group_rows})
        item["n_corruptions"] = len({row["corruption"] for row in group_rows})
        if "class" in group_rows[0]:
            item["n_classes"] = len({row["class"] for row in group_rows})
        for field in numeric_fields:
            item[field] = _mean(row.get(field) for row in group_rows)
        item["candidate_corrupt_auroc_recovery_fraction"] = _ratio(
            item["candidate_minus_baseline_corrupt_auroc"],
            item["baseline_drop_auroc"],
        )
        item["candidate_corrupt_auprc_recovery_fraction"] = _ratio(
            item["candidate_minus_baseline_corrupt_auprc"],
            item["baseline_drop_auprc"],
        )
        item["candidate_drop_reduction_auroc_fraction"] = _ratio(
            item["candidate_drop_reduction_auroc"],
            item["baseline_drop_auroc"],
        )
        item["candidate_drop_reduction_auprc_fraction"] = _ratio(
            item["candidate_drop_reduction_auprc"],
            item["baseline_drop_auprc"],
        )
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_float(row.get(field)) for field in fieldnames})


def _pp(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f} pp"


def _pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _metric(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("view") == "macro"]


def _extreme_row(
    rows: list[dict[str, Any]],
    key: str,
    *,
    largest: bool,
) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda row: float(row[key]), reverse=largest)[0]


def _write_markdown(
    path: Path,
    *,
    run_root: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    by_corruption: list[dict[str, Any]],
    by_class: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> None:
    macro = next(row for row in overall if row["view"] == "macro")
    daz = next(row for row in overall if row["view"] == "drop_all_zero")
    macro_gain_auprc = macro["candidate_minus_baseline_corrupt_auprc"]
    target_met = (
        macro_gain_auprc is not None
        and float(macro_gain_auprc) >= DEFAULT_SUCCESS_AUPRC_GAIN
    )
    threshold_pp = _pp(DEFAULT_SUCCESS_AUPRC_GAIN)
    best_corruption = _extreme_row(
        _macro_rows(by_corruption),
        "candidate_minus_baseline_corrupt_auprc",
        largest=True,
    )
    worst_corruption = _extreme_row(
        _macro_rows(by_corruption),
        "candidate_minus_baseline_corrupt_auprc",
        largest=False,
    )
    best_class = _extreme_row(
        _macro_rows(by_class),
        "candidate_minus_baseline_corrupt_auprc",
        largest=True,
    )
    worst_class = _extreme_row(
        _macro_rows(by_class),
        "candidate_minus_baseline_corrupt_auprc",
        largest=False,
    )
    lines = [
        "# PN2021-C Strong 10-20 pp Recovery Report",
        "",
        f"- Run root: `{run_root}`",
        f"- Output dir: `{output_dir}`",
        f"- Paired cells: {len(rows) // 2} center/corruption pairs, {len(rows)} rows after view split.",
        "- Mapping: v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51",
        "- Clean metric source: same filtered, ref-excluded subset computed inside PN2021-C evaluator.",
        "",
        "## Overall",
        "",
        "| view | noAug clean | noAug corrupt | noAug drop | LHAT clean | LHAT corrupt | LHAT drop | corrupt gain | drop reduction | corrupt recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (macro, daz):
        lines.append(
            "| {view} | {bc} / {bcp} | {bco} / {bcop} | {bd} / {bdp} | "
            "{cc} / {ccp} | {cco} / {ccop} | {cd} / {cdp} | "
            "{gain} / {gainp} | {red} / {redp} | {rec} / {recp} |".format(
                view=row["view"],
                bc=_metric(row["baseline_clean_auroc"]),
                bcp=_metric(row["baseline_clean_auprc"]),
                bco=_metric(row["baseline_corrupt_auroc"]),
                bcop=_metric(row["baseline_corrupt_auprc"]),
                bd=_pp(row["baseline_drop_auroc"]),
                bdp=_pp(row["baseline_drop_auprc"]),
                cc=_metric(row["candidate_clean_auroc"]),
                ccp=_metric(row["candidate_clean_auprc"]),
                cco=_metric(row["candidate_corrupt_auroc"]),
                ccop=_metric(row["candidate_corrupt_auprc"]),
                cd=_pp(row["candidate_drop_auroc"]),
                cdp=_pp(row["candidate_drop_auprc"]),
                gain=_pp(row["candidate_minus_baseline_corrupt_auroc"]),
                gainp=_pp(row["candidate_minus_baseline_corrupt_auprc"]),
                red=_pp(row["candidate_drop_reduction_auroc"]),
                redp=_pp(row["candidate_drop_reduction_auprc"]),
                rec=_pct(row["candidate_corrupt_auroc_recovery_fraction"]),
                recp=_pct(row["candidate_corrupt_auprc_recovery_fraction"]),
            )
        )
    if target_met:
        lines += [
            "",
            f"Interpretation: the calibrated profile produced the intended large drop, and this candidate meets the +{threshold_pp} corrupted macro AUPRC recovery target. "
            "In the main macro view, VAE-LHAT improves corrupted performance by "
            f"{_pp(macro['candidate_minus_baseline_corrupt_auroc'])} AUROC / {_pp(macro['candidate_minus_baseline_corrupt_auprc'])} AUPRC, which is "
            f"{_pct(macro['candidate_corrupt_auroc_recovery_fraction'])} / {_pct(macro['candidate_corrupt_auprc_recovery_fraction'])} of the noAug drop. "
            f"Clean macro AUPRC changes by {_pp(macro['candidate_minus_baseline_clean_auprc'])} versus noAug.",
        ]
    else:
        lines += [
            "",
            f"Interpretation: the calibrated profile produced the intended large drop, but this candidate does not meet the +{threshold_pp} corrupted macro AUPRC recovery target. "
            "In the main macro view, VAE-LHAT improves corrupted performance by "
            f"{_pp(macro['candidate_minus_baseline_corrupt_auroc'])} AUROC / {_pp(macro['candidate_minus_baseline_corrupt_auprc'])} AUPRC, which is "
            f"{_pct(macro['candidate_corrupt_auroc_recovery_fraction'])} / {_pct(macro['candidate_corrupt_auprc_recovery_fraction'])} of the noAug drop. "
            f"Clean macro AUPRC changes by {_pp(macro['candidate_minus_baseline_clean_auprc'])} versus noAug.",
        ]
    lines += [
        "",
        "## By Corruption",
        "",
        "| view | corruption | noAug drop | corrupt gain | drop reduction | corrupt recovery |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in by_corruption:
        if row["view"] != "macro":
            continue
        lines.append(
            "| {view} | {corr} | {bd} / {bdp} | {gain} / {gainp} | {red} / {redp} | {rec} / {recp} |".format(
                view=row["view"],
                corr=row["corruption"],
                bd=_pp(row["baseline_drop_auroc"]),
                bdp=_pp(row["baseline_drop_auprc"]),
                gain=_pp(row["candidate_minus_baseline_corrupt_auroc"]),
                gainp=_pp(row["candidate_minus_baseline_corrupt_auprc"]),
                red=_pp(row["candidate_drop_reduction_auroc"]),
                redp=_pp(row["candidate_drop_reduction_auprc"]),
                rec=_pct(row["candidate_corrupt_auroc_recovery_fraction"]),
                recp=_pct(row["candidate_corrupt_auprc_recovery_fraction"]),
            )
        )
    lines += [
        "",
        "## By Class",
        "",
        "| view | class | noAug drop | corrupt gain | drop reduction | corrupt recovery |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in by_class:
        if row["view"] != "macro":
            continue
        lines.append(
            "| {view} | {cls} | {bd} / {bdp} | {gain} / {gainp} | {red} / {redp} | {rec} / {recp} |".format(
                view=row["view"],
                cls=row["class"],
                bd=_pp(row["baseline_drop_auroc"]),
                bdp=_pp(row["baseline_drop_auprc"]),
                gain=_pp(row["candidate_minus_baseline_corrupt_auroc"]),
                gainp=_pp(row["candidate_minus_baseline_corrupt_auprc"]),
                red=_pp(row["candidate_drop_reduction_auroc"]),
                redp=_pp(row["candidate_drop_reduction_auprc"]),
                rec=_pct(row["candidate_corrupt_auroc_recovery_fraction"]),
                recp=_pct(row["candidate_corrupt_auprc_recovery_fraction"]),
            )
        )
    lines += [
        "",
        "## Mechanism-Level Readout",
        "",
    ]
    if target_met:
        lines.append(
            "- Target met for calibrated-profile source-target raw stressor training under strong PN2021-C; freeze this recipe for replication before broadening the method claim."
        )
    else:
        lines.append(
            "- Target not met; do not continue unstructured scalar sweeps of PGD strength or latent AugMix weight from this result."
        )
    if best_corruption is not None and worst_corruption is not None:
        lines.append(
            "- Corruption recovery is strongest for "
            f"{best_corruption['corruption']} ({_pp(best_corruption['candidate_minus_baseline_corrupt_auprc'])} AUPRC gain) "
            "and weakest for "
            f"{worst_corruption['corruption']} ({_pp(worst_corruption['candidate_minus_baseline_corrupt_auprc'])} AUPRC gain)."
        )
    if best_class is not None and worst_class is not None:
        lines.append(
            "- Class recovery is strongest for "
            f"{best_class['class']} ({_pp(best_class['candidate_minus_baseline_corrupt_auprc'])} AUPRC gain) "
            "and weakest for "
            f"{worst_class['class']} ({_pp(worst_class['candidate_minus_baseline_corrupt_auprc'])} AUPRC gain)."
        )
    lines += [
        "- Remaining follow-up should focus on replication and the weakest stressor/class cells, while preserving K500-internal validation plus PTB-XL/source-floor selection.",
        "",
        "## Artifacts",
        "",
        "- `pn2021c_recovery_by_center_corruption.csv`",
        "- `pn2021c_recovery_by_corruption.csv`",
        "- `pn2021c_recovery_by_center_corruption_class.csv`",
        "- `pn2021c_recovery_by_center_class.csv`",
        "- `pn2021c_recovery_by_class.csv`",
        "- `pn2021c_recovery_overall.csv`",
        "- `pn2021c_recovery_summary.md`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def export_report(
    *,
    run_root: Path,
    output_dir: Path,
    baseline_method: str = DEFAULT_BASELINE_METHOD,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    eval_filename: str = DEFAULT_EVAL_FILENAME,
    centers: Iterable[str] | None = None,
    corruptions: Iterable[str] | None = None,
    severity: int = 5,
) -> dict[str, str]:
    rows = build_recovery_rows(
        run_root=run_root,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        eval_filename=eval_filename,
        centers=centers,
        corruptions=corruptions,
        severity=severity,
    )
    per_class_rows = build_per_class_rows(
        run_root=run_root,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        eval_filename=eval_filename,
        centers=centers,
        corruptions=corruptions,
        severity=severity,
    )
    by_corruption = _aggregate_rows(rows, ("view", "corruption"))
    by_center_class = _aggregate_rows(per_class_rows, ("view", "center", "class"))
    by_class = _aggregate_rows(per_class_rows, ("view", "class"))
    overall = _aggregate_rows(rows, ("view",))

    detail_fields = [
        "center",
        "corruption",
        "severity",
        "view",
        "n_records",
        "clean_metric_source",
        "baseline_method",
        "candidate_method",
        "baseline_clean_auroc",
        "baseline_clean_auprc",
        "baseline_corrupt_auroc",
        "baseline_corrupt_auprc",
        "baseline_drop_auroc",
        "baseline_drop_auprc",
        "candidate_clean_auroc",
        "candidate_clean_auprc",
        "candidate_corrupt_auroc",
        "candidate_corrupt_auprc",
        "candidate_drop_auroc",
        "candidate_drop_auprc",
        "candidate_minus_baseline_clean_auroc",
        "candidate_minus_baseline_clean_auprc",
        "candidate_minus_baseline_corrupt_auroc",
        "candidate_minus_baseline_corrupt_auprc",
        "candidate_drop_reduction_auroc",
        "candidate_drop_reduction_auprc",
        "candidate_corrupt_auroc_recovery_fraction",
        "candidate_corrupt_auprc_recovery_fraction",
        "candidate_drop_reduction_auroc_fraction",
        "candidate_drop_reduction_auprc_fraction",
        "baseline_eval_json",
        "candidate_eval_json",
    ]
    aggregate_fields = [
        "view",
        "corruption",
        "n_rows",
        "n_centers",
        "n_corruptions",
        "baseline_clean_auroc",
        "baseline_clean_auprc",
        "baseline_corrupt_auroc",
        "baseline_corrupt_auprc",
        "baseline_drop_auroc",
        "baseline_drop_auprc",
        "candidate_clean_auroc",
        "candidate_clean_auprc",
        "candidate_corrupt_auroc",
        "candidate_corrupt_auprc",
        "candidate_drop_auroc",
        "candidate_drop_auprc",
        "candidate_minus_baseline_clean_auroc",
        "candidate_minus_baseline_clean_auprc",
        "candidate_minus_baseline_corrupt_auroc",
        "candidate_minus_baseline_corrupt_auprc",
        "candidate_drop_reduction_auroc",
        "candidate_drop_reduction_auprc",
        "candidate_corrupt_auroc_recovery_fraction",
        "candidate_corrupt_auprc_recovery_fraction",
        "candidate_drop_reduction_auroc_fraction",
        "candidate_drop_reduction_auprc_fraction",
    ]
    overall_fields = [field for field in aggregate_fields if field != "corruption"]
    class_detail_fields = [
        "center",
        "corruption",
        "severity",
        "view",
        "class",
        "baseline_clean_auroc",
        "baseline_clean_auprc",
        "baseline_corrupt_auroc",
        "baseline_corrupt_auprc",
        "baseline_drop_auroc",
        "baseline_drop_auprc",
        "candidate_clean_auroc",
        "candidate_clean_auprc",
        "candidate_corrupt_auroc",
        "candidate_corrupt_auprc",
        "candidate_drop_auroc",
        "candidate_drop_auprc",
        "candidate_minus_baseline_corrupt_auroc",
        "candidate_minus_baseline_corrupt_auprc",
        "candidate_drop_reduction_auroc",
        "candidate_drop_reduction_auprc",
        "candidate_corrupt_auroc_recovery_fraction",
        "candidate_corrupt_auprc_recovery_fraction",
        "baseline_n_pos",
        "baseline_n_valid",
        "candidate_n_pos",
        "candidate_n_valid",
    ]
    class_aggregate_fields = [
        "view",
        "center",
        "class",
        "n_rows",
        "n_centers",
        "n_corruptions",
        "n_classes",
        "baseline_clean_auroc",
        "baseline_clean_auprc",
        "baseline_corrupt_auroc",
        "baseline_corrupt_auprc",
        "baseline_drop_auroc",
        "baseline_drop_auprc",
        "candidate_clean_auroc",
        "candidate_clean_auprc",
        "candidate_corrupt_auroc",
        "candidate_corrupt_auprc",
        "candidate_drop_auroc",
        "candidate_drop_auprc",
        "candidate_minus_baseline_corrupt_auroc",
        "candidate_minus_baseline_corrupt_auprc",
        "candidate_drop_reduction_auroc",
        "candidate_drop_reduction_auprc",
        "candidate_corrupt_auroc_recovery_fraction",
        "candidate_corrupt_auprc_recovery_fraction",
        "baseline_n_pos",
        "baseline_n_valid",
        "candidate_n_pos",
        "candidate_n_valid",
    ]
    class_only_fields = [field for field in class_aggregate_fields if field != "center"]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_center_corruption": output_dir / "pn2021c_recovery_by_center_corruption.csv",
        "by_corruption": output_dir / "pn2021c_recovery_by_corruption.csv",
        "by_center_corruption_class": output_dir / "pn2021c_recovery_by_center_corruption_class.csv",
        "by_center_class": output_dir / "pn2021c_recovery_by_center_class.csv",
        "by_class": output_dir / "pn2021c_recovery_by_class.csv",
        "overall": output_dir / "pn2021c_recovery_overall.csv",
        "markdown": output_dir / "pn2021c_recovery_summary.md",
    }
    _write_csv(paths["by_center_corruption"], rows, detail_fields)
    _write_csv(paths["by_corruption"], by_corruption, aggregate_fields)
    _write_csv(paths["by_center_corruption_class"], per_class_rows, class_detail_fields)
    _write_csv(paths["by_center_class"], by_center_class, class_aggregate_fields)
    _write_csv(paths["by_class"], by_class, class_only_fields)
    _write_csv(paths["overall"], overall, overall_fields)
    _write_markdown(
        paths["markdown"],
        run_root=run_root,
        output_dir=output_dir,
        rows=rows,
        by_corruption=by_corruption,
        by_class=by_class,
        overall=overall,
    )
    return {key: path.as_posix() for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-method", default=DEFAULT_BASELINE_METHOD)
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--eval-filename", default=DEFAULT_EVAL_FILENAME)
    parser.add_argument("--centers", nargs="*", default=None)
    parser.add_argument("--corruptions", nargs="*", default=None)
    parser.add_argument("--severity", type=int, default=5)
    args = parser.parse_args()

    paths = export_report(
        run_root=Path(args.run_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        baseline_method=args.baseline_method,
        candidate_method=args.candidate_method,
        eval_filename=args.eval_filename,
        centers=args.centers,
        corruptions=args.corruptions,
        severity=args.severity,
    )
    for key, path in paths.items():
        print(f"[export] {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
