"""Export PN2021 v3 super5 center/per-class CSV summaries.

The evaluator writes rich JSON per run. This script turns every v3-super5 JSON
it can find into two flat CSVs suitable for thesis tables and quick diffs.
Only JSONs with explicit v3 mapping metadata are included.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_PATTERNS = [
    "/root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/synth_anchored_super5_v13*/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/synth_anchored_super5_v14_ctv2/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/real_anchored_super5/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/latent_hull_super5_pilot/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/latent_hull_coeff_ablation/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/latent_hull_lambda_ablation/*/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/**/eval_result_v3_super5_normsuppress.json",
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/**/eval_result.json",
]


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] skip unreadable JSON {path}: {exc}")
        return None


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


def _model_name(path: Path, root: Path) -> str:
    try:
        parent = path.parent.relative_to(root)
        return parent.as_posix()
    except ValueError:
        return path.parent.as_posix()


def _collect_paths(patterns: Iterable[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))

    # If a directory contains both versioned and generic eval_result.json, keep
    # the versioned file to avoid duplicate rows for the same model.
    by_dir: Dict[Path, Path] = {}
    for path in sorted(set(paths)):
        prev = by_dir.get(path.parent)
        if prev is None:
            by_dir[path.parent] = path
            continue
        if path.name == "eval_result_v3_super5_normsuppress.json":
            by_dir[path.parent] = path
    return sorted(by_dir.values())


def export_summaries(paths: Iterable[Path], root: Path, out_dir: Path) -> None:
    center_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    included = 0

    for path in paths:
        data = _load_json(path)
        if not data or not _is_v3_super5(data):
            continue

        included += 1
        model = _model_name(path, root)
        eval_json = path.as_posix()
        pn2021 = data["pn2021"]
        for center, cdata in sorted(pn2021["per_center"].items()):
            center_rows.append({
                "model": model,
                "eval_json": eval_json,
                "center": center,
                "n_records": cdata.get("n_records"),
                "effective_n": cdata.get("effective_n"),
                "n_excluded_ref": cdata.get("n_excluded_ref"),
                "macro_auroc": cdata.get("macro_auroc"),
                "macro_auprc": cdata.get("macro_auprc"),
                "n_classes_used": cdata.get("n_classes_used"),
            })
            for cls, pc in sorted(cdata.get("per_class", {}).items()):
                per_class_rows.append({
                    "model": model,
                    "eval_json": eval_json,
                    "center": center,
                    "class": cls,
                    "auroc": pc.get("auroc"),
                    "auprc": pc.get("auprc"),
                    "n_pos": pc.get("n_pos"),
                    "n_valid": pc.get("n_valid"),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    center_path = out_dir / "pn2021_v3_center_summary.csv"
    per_class_path = out_dir / "pn2021_v3_per_class_summary.csv"

    with center_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "eval_json", "center", "n_records", "effective_n",
            "n_excluded_ref", "macro_auroc", "macro_auprc", "n_classes_used",
        ])
        writer.writeheader()
        writer.writerows(center_rows)

    with per_class_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "eval_json", "center", "class", "auroc", "auprc",
            "n_pos", "n_valid",
        ])
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(f"[export] included eval JSONs: {included}")
    print(f"[export] center rows: {len(center_rows)} -> {center_path}")
    print(f"[export] per-class rows: {len(per_class_rows)} -> {per_class_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/triple_labels")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=DEFAULT_PATTERNS,
        help="Glob patterns for eval JSONs. Defaults cover current v3 experiment roots.",
    )
    args = parser.parse_args()

    paths = _collect_paths(args.patterns)
    export_summaries(paths, root=Path(args.root), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
