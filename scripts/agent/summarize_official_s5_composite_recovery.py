#!/usr/bin/env python3
"""Summarize official severity-5 composite PN2021-C recovery.

Inputs are paired direct and method evaluator JSON files, one JSON per center.
The output schema intentionally mirrors the existing c105 summary artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_center_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected CENTER=PATH, got {item!r}")
        center, raw_path = item.split("=", 1)
        center = center.strip()
        if not center:
            raise SystemExit(f"empty center in {item!r}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise SystemExit(f"missing JSON for {center}: {path}")
        out[center] = path
    return out


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


METRIC_ALIASES = {
    "mean_macro_auroc": ("mean_macro_auroc", "macro_auroc"),
    "mean_macro_auprc": ("mean_macro_auprc", "macro_auprc"),
    "mean_auroc_drop_vs_clean": ("mean_auroc_drop_vs_clean", "auroc_drop_vs_clean"),
    "mean_auprc_drop_vs_clean": ("mean_auprc_drop_vs_clean", "auprc_drop_vs_clean"),
    "clean_macro_auroc": ("clean_macro_auroc",),
    "clean_macro_auprc": ("clean_macro_auprc",),
}


def _leaf(payload: dict, center: str, combo: str) -> dict:
    try:
        return payload["per_center"][center][combo]["5"]
    except KeyError as exc:
        raise KeyError(f"{path_label(payload)} missing {center}/{combo}/5") from exc


def _metric(payload: dict, center: str, combo: str, key: str) -> float:
    leaf = _leaf(payload, center, combo)
    for alias in METRIC_ALIASES.get(key, (key,)):
        if alias in leaf:
            return float(leaf[alias])
    raise KeyError(
        f"{path_label(payload)} missing {center}/{combo}/5/{key}; "
        f"available={sorted(leaf)}"
    )


def path_label(payload: dict) -> str:
    return str(payload.get("model_dir") or payload.get("clean_eval_json") or "<payload>")


def _combo_depth(combo: str) -> int:
    return len(combo.split("+"))


def _row_mean(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return sum(values) / len(values) if values else float("nan")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    *,
    model: str,
    method: str,
    direct_paths: dict[str, Path],
    method_paths: dict[str, Path],
) -> list[dict]:
    centers = sorted(method_paths)
    missing_direct = sorted(set(centers) - set(direct_paths))
    if missing_direct:
        raise SystemExit(f"missing direct JSONs for centers: {missing_direct}")
    rows: list[dict] = []
    for center in centers:
        direct = _load_json(direct_paths[center])
        method_payload = _load_json(method_paths[center])
        combos = sorted(set(direct["per_center"][center]) & set(method_payload["per_center"][center]))
        if not combos:
            raise SystemExit(f"no overlapping combos for {center}")
        for combo in combos:
            direct_corr_auroc = _metric(direct, center, combo, "mean_macro_auroc")
            direct_corr_auprc = _metric(direct, center, combo, "mean_macro_auprc")
            direct_drop_auroc = _metric(direct, center, combo, "mean_auroc_drop_vs_clean") * 100.0
            direct_drop_auprc = _metric(direct, center, combo, "mean_auprc_drop_vs_clean") * 100.0
            method_corr_auroc = _metric(method_payload, center, combo, "mean_macro_auroc")
            method_corr_auprc = _metric(method_payload, center, combo, "mean_macro_auprc")
            method_drop_auroc = _metric(method_payload, center, combo, "mean_auroc_drop_vs_clean") * 100.0
            method_drop_auprc = _metric(method_payload, center, combo, "mean_auprc_drop_vs_clean") * 100.0
            direct_clean_auroc = _metric(direct, center, combo, "clean_macro_auroc")
            direct_clean_auprc = _metric(direct, center, combo, "clean_macro_auprc")
            method_clean_auroc = _metric(method_payload, center, combo, "clean_macro_auroc")
            method_clean_auprc = _metric(method_payload, center, combo, "clean_macro_auprc")
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "center": center,
                    "depth": _combo_depth(combo),
                    "combo": combo,
                    "direct_clean_auroc": direct_clean_auroc,
                    "direct_clean_auprc": direct_clean_auprc,
                    "direct_corr_auroc": direct_corr_auroc,
                    "direct_corr_auprc": direct_corr_auprc,
                    "direct_drop_auroc_pp": direct_drop_auroc,
                    "direct_drop_auprc_pp": direct_drop_auprc,
                    "method_clean_auroc": method_clean_auroc,
                    "method_clean_auprc": method_clean_auprc,
                    "method_corr_auroc": method_corr_auroc,
                    "method_corr_auprc": method_corr_auprc,
                    "method_drop_auroc_pp": method_drop_auroc,
                    "method_drop_auprc_pp": method_drop_auprc,
                    "recovery_auroc_pp": (method_corr_auroc - direct_corr_auroc) * 100.0,
                    "recovery_auprc_pp": (method_corr_auprc - direct_corr_auprc) * 100.0,
                    "method_clean_delta_vs_direct_pp_auroc": (
                        method_clean_auroc - direct_clean_auroc
                    )
                    * 100.0,
                    "method_clean_delta_vs_direct_pp_auprc": (
                        method_clean_auprc - direct_clean_auprc
                    )
                    * 100.0,
                }
            )
    return rows


LONG_FIELDS = [
    "model",
    "method",
    "center",
    "depth",
    "combo",
    "direct_clean_auroc",
    "direct_clean_auprc",
    "direct_corr_auroc",
    "direct_corr_auprc",
    "direct_drop_auroc_pp",
    "direct_drop_auprc_pp",
    "method_clean_auroc",
    "method_clean_auprc",
    "method_corr_auroc",
    "method_corr_auprc",
    "method_drop_auroc_pp",
    "method_drop_auprc_pp",
    "recovery_auroc_pp",
    "recovery_auprc_pp",
    "method_clean_delta_vs_direct_pp_auroc",
    "method_clean_delta_vs_direct_pp_auprc",
]


def summarize_by_depth(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for depth in sorted({int(r["depth"]) for r in rows}):
        selected = [r for r in rows if int(r["depth"]) == depth]
        out.append(
            {
                "model": selected[0]["model"],
                "method": selected[0]["method"],
                "depth": depth,
                "n_combos": len({r["combo"] for r in selected}),
                "n_centers": len({r["center"] for r in selected}),
                **{key: _row_mean(selected, key) for key in LONG_FIELDS[5:]},
            }
        )
    return out


def summarize_by_combo(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for combo in sorted({r["combo"] for r in rows}):
        selected = [r for r in rows if r["combo"] == combo]
        out.append(
            {
                "combo": combo,
                "depth": int(selected[0]["depth"]),
                "direct_corr_auroc": _row_mean(selected, "direct_corr_auroc"),
                "direct_corr_auprc": _row_mean(selected, "direct_corr_auprc"),
                "method_corr_auroc": _row_mean(selected, "method_corr_auroc"),
                "method_corr_auprc": _row_mean(selected, "method_corr_auprc"),
                "recovery_auroc_pp": _row_mean(selected, "recovery_auroc_pp"),
                "recovery_auprc_pp": _row_mean(selected, "recovery_auprc_pp"),
                "direct_drop_auroc_pp": _row_mean(selected, "direct_drop_auroc_pp"),
                "direct_drop_auprc_pp": _row_mean(selected, "direct_drop_auprc_pp"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--direct-json", action="append", default=[], metavar="CENTER=PATH")
    parser.add_argument("--method-json", action="append", default=[], metavar="CENTER=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    direct_paths = _parse_center_paths(args.direct_json)
    method_paths = _parse_center_paths(args.method_json)
    rows = build_rows(
        model=args.model,
        method=args.method,
        direct_paths=direct_paths,
        method_paths=method_paths,
    )
    summary = summarize_by_depth(rows)
    combo_summary = summarize_by_combo(rows)

    output_dir = Path(args.output_dir)
    long_path = output_dir / f"{args.prefix}_long.csv"
    summary_path = output_dir / f"{args.prefix}_summary.csv"
    combo_path = output_dir / f"{args.prefix}_combo_summary.csv"
    json_path = output_dir / f"{args.prefix}_summary.json"

    _write_csv(long_path, rows, LONG_FIELDS)
    _write_csv(summary_path, summary, list(summary[0].keys()) if summary else [])
    _write_csv(combo_path, combo_summary, list(combo_summary[0].keys()) if combo_summary else [])
    with json_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "method": args.method,
                "centers": sorted(method_paths),
                "summary": summary,
                "combo_summary": combo_summary,
                "long_csv": str(long_path),
                "summary_csv": str(summary_path),
                "combo_summary_csv": str(combo_path),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
