"""Build a quality-aware prompt-token gated pool from multiple gate outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ecg_adv_gen.data.gated_pools import load_gated_pool_npzs, write_selected_gated_pool_artifacts  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def quality_score(row: Dict, require_top1_bonus: float) -> float:
    p_target = float(row.get("p_target", 0.0))
    top1 = str(row.get("top1", ""))
    cls = str(row.get("class", ""))
    score = p_target
    if top1 == cls:
        score += require_top1_bonus
    metrics = row.get("digital_metrics", {})
    if metrics.get("unreliable_signal"):
        score -= 1.0
    # Mildly prefer confident but not pathologically saturated samples.
    if p_target > 0.98:
        score -= 0.02
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quota", type=int, default=40)
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--top1_bonus", type=float, default=0.25)
    ap.add_argument("--tag", default="quality_prompt_token_pool")
    args = ap.parse_args()

    allowed = {c.upper() for c in args.classes}
    candidates: Dict[str, List[Tuple[float, Dict, Dict[str, np.ndarray], int]]] = defaultdict(list)
    ref_ids = set()
    center_name = None

    for gated_dir_s in args.gated_dirs:
        gated_dir = Path(gated_dir_s)
        report_path = gated_dir / "gate_report.json"
        ref_meta_path = gated_dir / "gated_samples.ref_meta.json"
        if not report_path.exists():
            raise FileNotFoundError(f"missing gate_report.json under {gated_dir}")
        arrays = load_gated_pool_npzs(gated_dir)
        report = load_json(report_path)
        rows_by_index = {int(r["index"]): r for r in report["per_sample"]}
        source_indices = arrays.get("source_indices")
        if source_indices is None:
            raise ValueError(f"{gated_dir / 'gated_samples.npz'} missing source_indices")
        current_center = str(arrays["center_name"])
        if center_name is None:
            center_name = current_center
        elif current_center != center_name:
            raise ValueError(f"input centers differ: {[center_name, current_center]}")

        for local_i, src_i in enumerate(source_indices.astype(int).tolist()):
            row = rows_by_index[src_i]
            cls = str(row["class"]).upper()
            if cls not in allowed or not row.get("keep", False):
                continue
            sample = {
                "signals": arrays["signals"][local_i],
                "raw_signal_ct": arrays["raw_signal_ct"][local_i],
                "latents": arrays["latents"][local_i],
                "labels": arrays["labels"][local_i],
            }
            score = quality_score(row, args.top1_bonus)
            meta = dict(row)
            meta["source_gated_dir"] = str(gated_dir)
            meta["quality_score"] = score
            candidates[cls].append((score, meta, sample, local_i))

        if ref_meta_path.exists():
            ref_meta = load_json(ref_meta_path)
            ref_ids.update(str(x) for x in ref_meta.get("ref_record_ids", []))

    selected = []
    selected_meta = []
    for cls in args.classes:
        cls = cls.upper()
        rows = sorted(candidates.get(cls, []), key=lambda x: x[0], reverse=True)
        take = rows[:args.quota]
        selected.extend(take)
        selected_meta.extend([r[1] for r in take])
        print(f"[select] {cls}: selected {len(take)} / candidates {len(rows)}")
    if not selected:
        raise RuntimeError("no selected samples")

    signals = np.stack([x[2]["signals"] for x in selected]).astype(np.float32)
    raw_signal_ct = np.stack([x[2]["raw_signal_ct"] for x in selected]).astype(np.float32)
    latents = np.stack([x[2]["latents"] for x in selected]).astype(np.float32)
    labels = np.stack([x[2]["labels"] for x in selected]).astype(np.float32)

    out_dir = Path(args.out_dir)
    result = write_selected_gated_pool_artifacts(
        out_dir,
        tag=args.tag,
        center=center_name,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        ref_record_ids=sorted(ref_ids),
        report_name="quality_report.json",
        report_payload={
            "gated_dirs": list(args.gated_dirs),
            "quota": int(args.quota),
            "top1_bonus": float(args.top1_bonus),
            "selected": selected_meta,
        },
        trust_policy="quality-aware top quota by p_target + top1 bonus; HYP/CD hardcoded 0",
        ref_policy="Union of source K-ref exclusions.",
        class_names=CLASS_NAMES_SUPER5,
    )
    print(f"[select] wrote {result.paths.latents}")
    print(f"[select] counts: {result.counts}")
    print(f"[select] ref ids: {len(ref_ids)}")


if __name__ == "__main__":
    main()
