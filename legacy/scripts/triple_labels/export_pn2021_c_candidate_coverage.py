"""Export coverage table for PN2021-C candidate checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.triple_labels.export_pn2021_v3_summaries import DEFAULT_PATTERNS as V3_PATTERNS


PN2021_C_PATTERNS = [
    "eval_pn2021_c_stream*.json",
    "eval_pn2021_c.json",
]


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _collect_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    by_dir: Dict[Path, Path] = {}
    for path in sorted(set(paths)):
        prev = by_dir.get(path.parent)
        if prev is None or path.name == "eval_result_v3_super5_normsuppress.json":
            by_dir[path.parent] = path
    return sorted(by_dir.values())


def _is_v3_super5(data: Dict[str, Any]) -> bool:
    return (
        data.get("scheme") == "super5"
        and data.get("cache_versions", {}).get("pn2021_eval") == "v3_super5_normsuppress"
        and data.get("label_mapping", {})
        .get("pn2021_super5", {})
        .get("mapping_version")
        == "v3_super5_normsuppress_20260501"
        and bool(data.get("pn2021", {}).get("per_center"))
    )


def _is_full_pn2021_c(path: Path) -> bool:
    data = _load_json(path)
    if not data:
        return False
    return (
        data.get("scheme") == "super5"
        and len(data.get("centers", [])) >= 4
        and len(data.get("corruptions", [])) >= 5
        and len(data.get("severities", [])) >= 5
        and bool(data.get("aggregate_by_corruption_severity"))
    )


def _model_name(path: Path, root: Path) -> str:
    try:
        return path.parent.relative_to(root).as_posix()
    except ValueError:
        return path.parent.as_posix()


def _pn2021_c_files(model_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pattern in PN2021_C_PATTERNS:
        out.extend(model_dir.glob(pattern))
    return sorted(set(out))


def _recommend(model: str, auprc: float | None, has_full_c: bool) -> str:
    if has_full_c:
        return "done"
    if auprc is None:
        return "skip_no_metric"
    if "latent_hull_super5_pilot" in model and auprc >= 0.5555:
        return "run_pn2021_c_high_priority"
    if "synth_anchored_super5" in model and auprc >= 0.5550:
        return "run_pn2021_c_optional"
    if "ecgtwin_prompt_token_super5" in model and auprc >= 0.5558:
        return "run_only_if_new_prompt_mechanism"
    return "skip_low_priority"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp")
    parser.add_argument("--out_path", default="/root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv")
    parser.add_argument("--patterns", nargs="*", default=V3_PATTERNS)
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for eval_path in _collect_paths(args.patterns):
        data = _load_json(eval_path)
        if not data or not _is_v3_super5(data):
            continue
        model_dir = eval_path.parent
        model = _model_name(eval_path, root)
        pn2021 = data.get("pn2021", {})
        c_files = _pn2021_c_files(model_dir)
        full_c = [p for p in c_files if _is_full_pn2021_c(p)]
        auroc = pn2021.get("avg_macro_auroc")
        auprc = pn2021.get("avg_macro_auprc")
        rows.append({
            "model": model,
            "model_dir": model_dir.as_posix(),
            "best_model_exists": int((model_dir / "best_model.pt").exists()),
            "eval_json": eval_path.as_posix(),
            "pn2021_macro_auroc": auroc,
            "pn2021_macro_auprc": auprc,
            "has_full_pn2021_c": int(bool(full_c)),
            "pn2021_c_json": full_c[-1].as_posix() if full_c else "",
            "n_pn2021_c_json": len(c_files),
            "recommendation": _recommend(model, auprc, bool(full_c)),
        })

    rows.sort(key=lambda r: (
        r["recommendation"] != "run_pn2021_c_high_priority",
        -(r["pn2021_macro_auprc"] or 0.0),
        r["model"],
    ))
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fields = [
            "model", "model_dir", "best_model_exists", "eval_json",
            "pn2021_macro_auroc", "pn2021_macro_auprc",
            "has_full_pn2021_c", "pn2021_c_json", "n_pn2021_c_json",
            "recommendation",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] rows={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
