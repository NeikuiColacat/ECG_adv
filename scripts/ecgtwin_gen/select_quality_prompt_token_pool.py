"""Build a quality-aware prompt-token gated pool from multiple gate outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

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
        samples_path = gated_dir / "gated_samples.npz"
        report_path = gated_dir / "gate_report.json"
        ref_meta_path = gated_dir / "gated_samples.ref_meta.json"
        if not samples_path.exists() or not report_path.exists():
            raise FileNotFoundError(f"missing gated_samples.npz or gate_report.json under {gated_dir}")
        with np.load(samples_path, allow_pickle=True) as data:
            arrays = {k: data[k] for k in data.files}
        report = load_json(report_path)
        rows_by_index = {int(r["index"]): r for r in report["per_sample"]}
        source_indices = arrays.get("source_indices")
        if source_indices is None:
            raise ValueError(f"{samples_path} missing source_indices")
        if center_name is None:
            center_name = str(arrays["center_name"]) if "center_name" in arrays else "?"

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
    counts = Counter(CLASS_NAMES_SUPER5[int(i)] for i in labels.argmax(axis=1))
    class_trust = {cls: (1.0 if counts.get(cls, 0) > 0 else 0.0) for cls in CLASS_NAMES_SUPER5}
    class_trust["HYP"] = 0.0
    class_trust["CD"] = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "gated_samples.npz"
    latent_path = out_dir / "gated_samples.latent.npz"
    trust_path = out_dir / "gated_samples.class_trust.json"
    ref_meta_path = out_dir / "gated_samples.ref_meta.json"
    report_path = out_dir / "quality_report.json"

    np.savez_compressed(
        samples_path,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=center_name,
        class_names=np.asarray(CLASS_NAMES_SUPER5),
    )
    np.savez_compressed(
        latent_path,
        latents=latents,
        labels=labels,
        center_name=center_name,
        class_names=np.asarray(CLASS_NAMES_SUPER5),
    )
    with open(trust_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "center": center_name,
            "class_trust": class_trust,
            "counts": dict(counts),
            "policy": "quality-aware top quota by p_target + top1 bonus; HYP/CD hardcoded 0",
        }, f, indent=2)
    with open(ref_meta_path, "w") as f:
        json.dump({
            "center": center_name,
            "ref_record_ids": sorted(ref_ids),
            "policy": "Union of source K-ref exclusions.",
        }, f, indent=2)
    with open(report_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "center": center_name,
            "gated_dirs": list(args.gated_dirs),
            "quota": int(args.quota),
            "top1_bonus": float(args.top1_bonus),
            "n_samples": int(labels.shape[0]),
            "counts": dict(counts),
            "selected": selected_meta,
            "samples": str(samples_path),
            "latents": str(latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(ref_meta_path),
        }, f, indent=2)
    print(f"[select] wrote {latent_path}")
    print(f"[select] counts: {dict(counts)}")
    print(f"[select] ref ids: {len(ref_ids)}")


if __name__ == "__main__":
    main()
