"""Select paired target-token samples against no-token controls.

The selector assumes two synthetic pools were generated with the same center,
classes, reference policy, and seed. It keeps pairs where the target-token arm
has a higher target-center style score than the no-token arm, then writes two
matched gated pools:

  target/gated/gated_samples.*
  no_token/gated/gated_samples.*

This is for causal downstream tests: both arms have the same pair ids and
sample count, so downstream differences are less confounded by reference or
gate-count differences.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def norm_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def load_style_rows(path: Path) -> Dict[Tuple[str, int], float]:
    rows: Dict[Tuple[str, int], float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(norm_path(row["input_dir"]), int(row["sample_index"]))] = float(row["prob_expected"])
    if not rows:
        raise RuntimeError(f"no style rows in {path}")
    return rows


def load_gated(gated_dir: Path) -> Dict:
    samples_path = gated_dir / "gated_samples.npz"
    report_path = gated_dir / "gate_report.json"
    ref_meta_path = gated_dir / "gated_samples.ref_meta.json"
    if not samples_path.exists() or not report_path.exists():
        raise FileNotFoundError(f"missing gated outputs under {gated_dir}")
    with np.load(samples_path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    report = load_json(report_path)
    rows_by_original = {int(row["index"]): row for row in report["per_sample"]}
    source_indices = arrays["source_indices"].astype(int)
    by_original = {}
    for local_i, original_i in enumerate(source_indices.tolist()):
        row = dict(rows_by_original[int(original_i)])
        by_original[int(original_i)] = {
            "local_i": int(local_i),
            "original_i": int(original_i),
            "row": row,
        }
    ref_ids = set()
    if ref_meta_path.exists():
        ref_meta = load_json(ref_meta_path)
        ref_ids.update(str(x) for x in ref_meta.get("ref_record_ids", []))
    center_name = str(arrays["center_name"]) if "center_name" in arrays else gated_dir.parent.name
    return {
        "dir": gated_dir,
        "arrays": arrays,
        "by_original": by_original,
        "ref_ids": ref_ids,
        "center_name": center_name,
    }


def score_pair(
    target_row: Dict,
    no_row: Dict,
    target_style: float,
    no_style: float,
    args: argparse.Namespace,
) -> float:
    cls = str(target_row["class"]).upper()
    target_p = float(target_row.get("p_target", 0.0))
    no_p = float(no_row.get("p_target", 0.0))
    style_delta = target_style - no_style
    # Prefer style gain, but avoid purely saturated target probabilities.
    return (
        style_delta
        + args.target_prob_weight * (target_p - no_p)
        - args.boundary_weight * abs(target_p - args.boundary_target_p)
        - (args.saturation_penalty if target_p > args.saturation_p else 0.0)
        + (0.02 if cls == "STTC" else 0.0)
    )


def write_pool(
    selected: List[Dict],
    arm: str,
    blob: Dict,
    ref_ids: set[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = blob["arrays"]
    local_indices = np.asarray([item[f"{arm}_local_i"] for item in selected], dtype=np.int64)
    original_indices = np.asarray([item["original_i"] for item in selected], dtype=np.int64)
    pair_ids = np.asarray([item["pair_id"] for item in selected], dtype=np.int64)
    labels = arrays["labels"][local_indices].astype(np.float32)
    counts = Counter(CLASS_NAMES_SUPER5[int(i)] for i in labels.argmax(axis=1))
    class_trust = {cls: (1.0 if counts.get(cls, 0) > 0 else 0.0) for cls in CLASS_NAMES_SUPER5}
    class_trust["CD"] = 0.0
    class_trust["HYP"] = 0.0

    np.savez_compressed(
        out_dir / "gated_samples.npz",
        signals=arrays["signals"][local_indices].astype(np.float32),
        raw_signal_ct=arrays["raw_signal_ct"][local_indices].astype(np.float32),
        latents=arrays["latents"][local_indices].astype(np.float32),
        labels=labels,
        center_name=blob["center_name"],
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_indices=original_indices,
        pair_ids=pair_ids,
    )
    np.savez_compressed(
        out_dir / "gated_samples.latent.npz",
        latents=arrays["latents"][local_indices].astype(np.float32),
        labels=labels,
        center_name=blob["center_name"],
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_indices=original_indices,
        pair_ids=pair_ids,
    )
    with (out_dir / "gated_samples.class_trust.json").open("w") as f:
        json.dump({
            "tag": f"{args.tag}_{arm}",
            "center": blob["center_name"],
            "class_trust": class_trust,
            "counts": dict(counts),
            "policy": "paired token-delta selected pool; HYP/CD hardcoded 0",
        }, f, indent=2)
    with (out_dir / "gated_samples.ref_meta.json").open("w") as f:
        json.dump({
            "center": blob["center_name"],
            "ref_record_ids": sorted(ref_ids),
            "policy": "Union of paired source K-ref exclusions.",
        }, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_gated_dir", required=True)
    ap.add_argument("--no_token_gated_dir", required=True)
    ap.add_argument("--style_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--quota", type=int, default=80)
    ap.add_argument("--class_quota", nargs="*", default=[],
                    help="Optional CLASS=N overrides, e.g. STTC=30")
    ap.add_argument("--min_style_delta", type=float, default=0.05)
    ap.add_argument("--min_target_style", type=float, default=0.05)
    ap.add_argument("--max_target_style", type=float, default=0.95)
    ap.add_argument("--min_target_prob", type=float, default=0.45)
    ap.add_argument("--max_target_prob", type=float, default=0.995)
    ap.add_argument("--boundary_target_p", type=float, default=0.60)
    ap.add_argument("--boundary_weight", type=float, default=0.10)
    ap.add_argument("--target_prob_weight", type=float, default=0.10)
    ap.add_argument("--saturation_p", type=float, default=0.98)
    ap.add_argument("--saturation_penalty", type=float, default=0.03)
    ap.add_argument("--tag", default="paired_token_delta")
    args = ap.parse_args()

    allowed = {c.upper() for c in args.classes}
    quotas = {c: int(args.quota) for c in allowed}
    for item in args.class_quota:
        if "=" not in item:
            raise ValueError(f"bad --class_quota item {item!r}")
        cls, value = item.split("=", 1)
        cls = cls.upper()
        if cls not in allowed:
            raise ValueError(f"class_quota class {cls!r} not in --classes")
        quotas[cls] = int(value)

    target = load_gated(Path(args.target_gated_dir))
    no_token = load_gated(Path(args.no_token_gated_dir))
    if target["center_name"] != no_token["center_name"]:
        raise ValueError("target and no-token centers differ")
    style_rows = load_style_rows(Path(args.style_csv))
    target_root = norm_path(args.target_gated_dir)
    no_root = norm_path(args.no_token_gated_dir)

    candidates: Dict[str, List[Dict]] = defaultdict(list)
    shared = sorted(set(target["by_original"]) & set(no_token["by_original"]))
    pair_id = 0
    for original_i in shared:
        t_item = target["by_original"][original_i]
        n_item = no_token["by_original"][original_i]
        t_row = t_item["row"]
        n_row = n_item["row"]
        cls = str(t_row.get("class", "")).upper()
        if cls not in allowed or str(n_row.get("class", "")).upper() != cls:
            continue
        t_style = style_rows.get((target_root, t_item["local_i"]))
        n_style = style_rows.get((no_root, n_item["local_i"]))
        if t_style is None or n_style is None:
            continue
        target_p = float(t_row.get("p_target", 0.0))
        style_delta = float(t_style - n_style)
        if style_delta < args.min_style_delta:
            continue
        if not (args.min_target_style <= float(t_style) <= args.max_target_style):
            continue
        if not (args.min_target_prob <= target_p <= args.max_target_prob):
            continue
        candidates[cls].append({
            "pair_id": pair_id,
            "original_i": int(original_i),
            "target_local_i": t_item["local_i"],
            "no_token_local_i": n_item["local_i"],
            "class": cls,
            "target_style": float(t_style),
            "no_token_style": float(n_style),
            "style_delta": style_delta,
            "target_p": target_p,
            "no_token_p": float(n_row.get("p_target", 0.0)),
            "score": score_pair(t_row, n_row, float(t_style), float(n_style), args),
            "target_row": t_row,
            "no_token_row": n_row,
        })
        pair_id += 1

    selected: List[Dict] = []
    report_counts = {}
    for cls in [c.upper() for c in args.classes]:
        rows = sorted(candidates.get(cls, []), key=lambda x: x["score"], reverse=True)
        take = rows[:quotas.get(cls, args.quota)]
        selected.extend(take)
        report_counts[cls] = {"candidates": len(rows), "selected": len(take)}
        print(f"[select] {cls}: selected {len(take)} / candidates {len(rows)}")
    if not selected:
        raise RuntimeError("no paired samples selected")

    ref_ids = set(target["ref_ids"]) | set(no_token["ref_ids"])
    out_root = Path(args.out_dir)
    write_pool(selected, "target", target, ref_ids, out_root / "target_token" / "gated", args)
    write_pool(selected, "no_token", no_token, ref_ids, out_root / "no_token" / "gated", args)
    with (out_root / "selection_report.json").open("w") as f:
        json.dump({
            "tag": args.tag,
            "target_gated_dir": args.target_gated_dir,
            "no_token_gated_dir": args.no_token_gated_dir,
            "style_csv": args.style_csv,
            "config": vars(args),
            "counts": report_counts,
            "n_selected": len(selected),
            "selected": selected,
        }, f, indent=2)
    print(f"[select] wrote {out_root}")
    print(f"[select] total selected {len(selected)}")


if __name__ == "__main__":
    main()
