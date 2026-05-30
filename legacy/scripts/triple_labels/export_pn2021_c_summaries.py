"""Export PN2021-C corruption benchmark CSV summaries.

The PN2021-C evaluator writes one JSON per checkpoint. This helper flattens
those JSONs into CSVs for quick comparison across clean models, Latent-Hull
variants, and prompt-token pilots.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PATTERNS = [
    "/root/autodl-tmp/triple_labels/super5/eval_pn2021_c_stream_v1.json",
    "/root/autodl-tmp/latent_hull_super5_pilot/*/eval_pn2021_c_stream*.json",
    "/root/autodl-tmp/latent_hull_coeff_ablation/*/eval_pn2021_c_stream*.json",
    "/root/autodl-tmp/latent_hull_lambda_ablation/*/eval_pn2021_c_stream*.json",
    "/root/autodl-tmp/synth_anchored_super5_*/**/eval_pn2021_c_stream*.json",
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/**/eval_pn2021_c_stream*.json",
]


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] skip unreadable JSON {path}: {exc}")
        return None


def _is_pn2021_c(data: Dict[str, Any]) -> bool:
    return (
        data.get("scheme") == "super5"
        and bool(data.get("per_center"))
        and bool(data.get("aggregate_by_corruption_severity"))
    )


def _model_name(path: Path, root: Path) -> str:
    try:
        return path.parent.relative_to(root).as_posix()
    except ValueError:
        return path.parent.as_posix()


def _collect_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    by_dir: Dict[Path, Path] = {}
    for path in sorted(set(paths)):
        prev = by_dir.get(path.parent)
        if prev is None:
            by_dir[path.parent] = path
            continue
        prev_score = ("selfclean" in prev.name, prev.name)
        new_score = ("selfclean" in path.name, path.name)
        if new_score > prev_score:
            by_dir[path.parent] = path
    return sorted(by_dir.values())


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _clean_by_center(clean_json: Path | None) -> Dict[str, Dict[str, float]]:
    if clean_json is None or not clean_json.exists():
        return {}
    data = _load_json(clean_json)
    if not data:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for center, vals in data.get("pn2021", {}).get("per_center", {}).items():
        out[center] = {
            "macro_auroc": vals.get("macro_auroc"),
            "macro_auprc": vals.get("macro_auprc"),
        }
    return out


def _self_clean_path(data: Dict[str, Any]) -> Path | None:
    model_dir = data.get("model_dir")
    if not model_dir:
        return None
    path = Path(model_dir) / "eval_result_v3_super5_normsuppress.json"
    return path if path.exists() else None


def _passes_full_filter(data: Dict[str, Any], min_centers: int, min_corruptions: int, min_severities: int) -> bool:
    if len(data.get("centers", [])) < min_centers:
        return False
    if len(data.get("corruptions", [])) < min_corruptions:
        return False
    severities = {str(s) for s in data.get("severities", [])}
    if len(severities) < min_severities:
        return False
    return True


def export_summaries(
    paths: Iterable[Path],
    root: Path,
    out_dir: Path,
    min_centers: int,
    min_corruptions: int,
    min_severities: int,
) -> None:
    run_rows: List[Dict[str, Any]] = []
    corruption_rows: List[Dict[str, Any]] = []
    severity_rows: List[Dict[str, Any]] = []
    center_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    included = 0

    for path in paths:
        data = _load_json(path)
        if not data or not _is_pn2021_c(data):
            continue
        if not _passes_full_filter(data, min_centers, min_corruptions, min_severities):
            continue

        included += 1
        model = _model_name(path, root)
        eval_json = path.as_posix()
        agg = data["aggregate_by_corruption_severity"]
        json_clean_eval_json = data.get("clean_eval_json")
        self_clean_json = _self_clean_path(data)
        if self_clean_json is None and json_clean_eval_json:
            self_clean_json = Path(json_clean_eval_json)
        self_clean = _clean_by_center(self_clean_json)
        self_drop_by_key: Dict[Tuple[str, str], List[Tuple[float | None, float | None]]] = {}

        all_agg_values = [vals for sev_map in agg.values() for vals in sev_map.values()]
        run_mean_macro_auroc = _mean(v.get("mean_macro_auroc") for v in all_agg_values)
        run_mean_macro_auprc = _mean(v.get("mean_macro_auprc") for v in all_agg_values)
        json_run_auroc_drop = _mean(v.get("mean_auroc_drop_vs_clean") for v in all_agg_values)
        json_run_auprc_drop = _mean(v.get("mean_auprc_drop_vs_clean") for v in all_agg_values)

        for corruption, sev_map in sorted(agg.items()):
            sev_values = list(sev_map.values())
            for severity, vals in sorted(sev_map.items(), key=lambda item: int(item[0])):
                severity_rows.append({
                    "model": model,
                    "eval_json": eval_json,
                    "corruption": corruption,
                    "severity": int(severity),
                    "n_centers": vals.get("n_centers"),
                    "mean_macro_auroc": vals.get("mean_macro_auroc"),
                    "mean_macro_auprc": vals.get("mean_macro_auprc"),
                    "mean_auroc_drop_vs_clean": None,
                    "mean_auprc_drop_vs_clean": None,
                    "mean_auroc_drop_vs_json_clean": vals.get("mean_auroc_drop_vs_clean"),
                    "mean_auprc_drop_vs_json_clean": vals.get("mean_auprc_drop_vs_clean"),
                })

        for center, corruption_map in sorted(data["per_center"].items()):
            for corruption, sev_map in sorted(corruption_map.items()):
                for severity, vals in sorted(sev_map.items(), key=lambda item: int(item[0])):
                    self_clean_vals = self_clean.get(center, {})
                    self_clean_auroc = self_clean_vals.get("macro_auroc")
                    self_clean_auprc = self_clean_vals.get("macro_auprc")
                    self_auroc_drop = None
                    self_auprc_drop = None
                    if self_clean_auroc is not None and vals.get("macro_auroc") is not None:
                        self_auroc_drop = float(self_clean_auroc) - float(vals["macro_auroc"])
                    if self_clean_auprc is not None and vals.get("macro_auprc") is not None:
                        self_auprc_drop = float(self_clean_auprc) - float(vals["macro_auprc"])
                    key = (corruption, str(severity))
                    self_drop_by_key.setdefault(key, []).append((self_auroc_drop, self_auprc_drop))
                    center_rows.append({
                        "model": model,
                        "eval_json": eval_json,
                        "center": center,
                        "corruption": corruption,
                        "severity": int(severity),
                        "n_records": vals.get("n_records"),
                        "macro_auroc": vals.get("macro_auroc"),
                        "macro_auprc": vals.get("macro_auprc"),
                        "clean_macro_auroc": vals.get("clean_macro_auroc"),
                        "clean_macro_auprc": vals.get("clean_macro_auprc"),
                        "self_clean_macro_auroc": self_clean_auroc,
                        "self_clean_macro_auprc": self_clean_auprc,
                        "auroc_drop_vs_clean": self_auroc_drop,
                        "auprc_drop_vs_clean": self_auprc_drop,
                        "auroc_drop_vs_json_clean": vals.get("auroc_drop_vs_clean"),
                        "auprc_drop_vs_json_clean": vals.get("auprc_drop_vs_clean"),
                        "n_classes_used": vals.get("n_classes_used"),
                    })
                    for cls, pc in sorted(vals.get("per_class", {}).items()):
                        per_class_rows.append({
                            "model": model,
                            "eval_json": eval_json,
                            "center": center,
                            "corruption": corruption,
                            "severity": int(severity),
                            "class": cls,
                            "auroc": pc.get("auroc"),
                            "auprc": pc.get("auprc"),
                            "n_pos": pc.get("n_pos"),
                            "n_valid": pc.get("n_valid"),
                        })

        self_agg_by_key = {
            key: {
                "mean_auroc_drop_vs_clean": _mean(x for x, _ in vals),
                "mean_auprc_drop_vs_clean": _mean(y for _, y in vals),
            }
            for key, vals in self_drop_by_key.items()
        }
        for row in severity_rows:
            if row["eval_json"] != eval_json:
                continue
            key = (row["corruption"], str(row["severity"]))
            self_agg = self_agg_by_key.get(key, {})
            row["mean_auroc_drop_vs_clean"] = self_agg.get("mean_auroc_drop_vs_clean")
            row["mean_auprc_drop_vs_clean"] = self_agg.get("mean_auprc_drop_vs_clean")

        for corruption, sev_map in sorted(agg.items()):
            keys = [(corruption, str(severity)) for severity in sorted(sev_map.keys(), key=int)]
            corruption_rows.append({
                "model": model,
                "eval_json": eval_json,
                "corruption": corruption,
                "mean_macro_auroc": _mean(v.get("mean_macro_auroc") for v in sev_map.values()),
                "mean_macro_auprc": _mean(v.get("mean_macro_auprc") for v in sev_map.values()),
                "mean_auroc_drop_vs_clean": _mean(
                    self_agg_by_key.get(key, {}).get("mean_auroc_drop_vs_clean")
                    for key in keys
                ),
                "mean_auprc_drop_vs_clean": _mean(
                    self_agg_by_key.get(key, {}).get("mean_auprc_drop_vs_clean")
                    for key in keys
                ),
                "mean_auroc_drop_vs_json_clean": _mean(
                    v.get("mean_auroc_drop_vs_clean") for v in sev_map.values()
                ),
                "mean_auprc_drop_vs_json_clean": _mean(
                    v.get("mean_auprc_drop_vs_clean") for v in sev_map.values()
                ),
            })

        run_rows.append({
            "model": model,
            "eval_json": eval_json,
            "model_dir": data.get("model_dir"),
            "clean_eval_json": self_clean_json.as_posix() if self_clean_json else None,
            "json_clean_eval_json": json_clean_eval_json,
            "mode": data.get("mode"),
            "n_centers": len(data.get("centers", [])),
            "n_corruptions": len(data.get("corruptions", [])),
            "n_severities": len(data.get("severities", [])),
            "mean_macro_auroc": run_mean_macro_auroc,
            "mean_macro_auprc": run_mean_macro_auprc,
            "mean_auroc_drop_vs_clean": _mean(
                vals.get("mean_auroc_drop_vs_clean") for vals in self_agg_by_key.values()
            ),
            "mean_auprc_drop_vs_clean": _mean(
                vals.get("mean_auprc_drop_vs_clean") for vals in self_agg_by_key.values()
            ),
            "mean_auroc_drop_vs_json_clean": json_run_auroc_drop,
            "mean_auprc_drop_vs_json_clean": json_run_auprc_drop,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pn2021_c_run_summary.csv": run_rows,
        "pn2021_c_corruption_summary.csv": corruption_rows,
        "pn2021_c_severity_summary.csv": severity_rows,
        "pn2021_c_center_summary.csv": center_rows,
        "pn2021_c_per_class_summary.csv": per_class_rows,
    }
    fieldnames = {
        "pn2021_c_run_summary.csv": [
            "model", "eval_json", "model_dir", "clean_eval_json",
            "json_clean_eval_json", "mode",
            "n_centers", "n_corruptions", "n_severities", "mean_macro_auroc",
            "mean_macro_auprc", "mean_auroc_drop_vs_clean",
            "mean_auprc_drop_vs_clean", "mean_auroc_drop_vs_json_clean",
            "mean_auprc_drop_vs_json_clean",
        ],
        "pn2021_c_corruption_summary.csv": [
            "model", "eval_json", "corruption", "mean_macro_auroc",
            "mean_macro_auprc", "mean_auroc_drop_vs_clean",
            "mean_auprc_drop_vs_clean", "mean_auroc_drop_vs_json_clean",
            "mean_auprc_drop_vs_json_clean",
        ],
        "pn2021_c_severity_summary.csv": [
            "model", "eval_json", "corruption", "severity", "n_centers",
            "mean_macro_auroc", "mean_macro_auprc",
            "mean_auroc_drop_vs_clean", "mean_auprc_drop_vs_clean",
            "mean_auroc_drop_vs_json_clean", "mean_auprc_drop_vs_json_clean",
        ],
        "pn2021_c_center_summary.csv": [
            "model", "eval_json", "center", "corruption", "severity",
            "n_records", "macro_auroc", "macro_auprc", "clean_macro_auroc",
            "clean_macro_auprc", "self_clean_macro_auroc",
            "self_clean_macro_auprc", "auroc_drop_vs_clean",
            "auprc_drop_vs_clean", "auroc_drop_vs_json_clean",
            "auprc_drop_vs_json_clean", "n_classes_used",
        ],
        "pn2021_c_per_class_summary.csv": [
            "model", "eval_json", "center", "corruption", "severity",
            "class", "auroc", "auprc", "n_pos", "n_valid",
        ],
    }

    print(f"[export] included PN2021-C JSONs: {included}")
    for filename, rows in outputs.items():
        out_path = out_dir / filename
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames[filename])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[export] {filename}: rows={len(rows)} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/triple_labels")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    parser.add_argument("--min_centers", type=int, default=4)
    parser.add_argument("--min_corruptions", type=int, default=5)
    parser.add_argument("--min_severities", type=int, default=5)
    args = parser.parse_args()

    paths = _collect_paths(args.patterns)
    export_summaries(
        paths,
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        min_centers=args.min_centers,
        min_corruptions=args.min_corruptions,
        min_severities=args.min_severities,
    )


if __name__ == "__main__":
    main()
