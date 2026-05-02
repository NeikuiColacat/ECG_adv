"""Build a mixed confidence/boundary prompt-token gated pool.

The previous quality-only selector kept the most classifier-obvious generated
ECG and underperformed the simpler balanced gate-pass pool. This selector keeps
three slices per class:

  high       high target probability and agreement
  boundary   target is still top-1, but with a small target-vs-runner-up margin
  diverse    latent-space farthest-first fill from the remaining candidates

Inputs are raw gated output directories containing `gated_samples.npz` and
`gate_report.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def target_margin(row: Dict) -> float:
    cls = str(row["class"]).upper()
    probs = row.get("victim_probs", {})
    p_target = float(row.get("p_target", 0.0))
    other = [float(v) for k, v in probs.items() if str(k).upper() != cls]
    runner_up = max(other) if other else 0.0
    return p_target - runner_up


def high_score(row: Dict) -> float:
    p_target = float(row.get("p_target", 0.0))
    margin = target_margin(row)
    score = p_target + 0.25 * margin
    if p_target > 0.98:
        score -= 0.02
    if row.get("digital_metrics", {}).get("unreliable_signal"):
        score -= 1.0
    return float(score)


def boundary_score(row: Dict, target_p: float) -> float:
    p_target = float(row.get("p_target", 0.0))
    margin = target_margin(row)
    # Prefer nontrivial but not too easy samples. Margin must stay positive
    # because gate outputs already require target top-1.
    return float(abs(p_target - target_p) + 0.25 * max(margin, 0.0))


def greedy_diverse(indices: List[int], latents: np.ndarray, n_take: int) -> List[int]:
    if n_take <= 0 or not indices:
        return []
    flat = latents.reshape(latents.shape[0], -1)
    selected = [indices[0]]
    remaining = indices[1:]
    while remaining and len(selected) < n_take:
        sel_flat = flat[np.asarray(selected)]
        rem_flat = flat[np.asarray(remaining)]
        diff = rem_flat[:, None, :] - sel_flat[None, :, :]
        dist2 = np.einsum("nmd,nmd->nm", diff, diff)
        best_i = int(np.argmax(dist2.min(axis=1)))
        selected.append(remaining.pop(best_i))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quota", type=int, default=50)
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--high_frac", type=float, default=0.4)
    ap.add_argument("--boundary_frac", type=float, default=0.4)
    ap.add_argument("--boundary_target_p", type=float, default=0.6)
    ap.add_argument("--tag", default="mixed_prompt_token_pool")
    args = ap.parse_args()

    allowed = [c.upper() for c in args.classes]
    candidates: Dict[str, List[Dict]] = defaultdict(list)
    ref_ids = set()
    center_name = None

    for gated_dir_s in args.gated_dirs:
        gated_dir = Path(gated_dir_s)
        samples_path = gated_dir / "gated_samples.npz"
        report_path = gated_dir / "gate_report.json"
        ref_meta_path = gated_dir / "gated_samples.ref_meta.json"
        if not samples_path.exists() or not report_path.exists():
            raise FileNotFoundError(f"missing gated_samples.npz or gate_report.json under {gated_dir}")

        arrays = {k: v for k, v in np.load(samples_path, allow_pickle=True).items()}
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
            candidates[cls].append({
                "row": dict(row),
                "signals": arrays["signals"][local_i],
                "raw_signal_ct": arrays["raw_signal_ct"][local_i],
                "latents": arrays["latents"][local_i],
                "labels": arrays["labels"][local_i],
                "source_gated_dir": str(gated_dir),
                "source_index": int(src_i),
                "p_target": float(row.get("p_target", 0.0)),
                "margin": float(target_margin(row)),
            })

        if ref_meta_path.exists():
            ref_meta = load_json(ref_meta_path)
            ref_ids.update(str(x) for x in ref_meta.get("ref_record_ids", []))

    selected: List[Dict] = []
    report_selected: List[Dict] = []
    for cls in allowed:
        rows = list(candidates.get(cls, []))
        n_high = int(round(args.quota * args.high_frac))
        n_boundary = int(round(args.quota * args.boundary_frac))
        n_high = min(n_high, args.quota)
        n_boundary = min(n_boundary, args.quota - n_high)
        n_diverse = max(0, args.quota - n_high - n_boundary)

        chosen_keys = set()
        chosen: List[Dict] = []

        high_rows = sorted(rows, key=lambda x: high_score(x["row"]), reverse=True)
        for item in high_rows:
            key = (item["source_gated_dir"], item["source_index"])
            if key in chosen_keys:
                continue
            chosen.append(item)
            chosen_keys.add(key)
            if len(chosen) >= n_high:
                break

        boundary_rows = sorted(
            rows,
            key=lambda x: boundary_score(x["row"], args.boundary_target_p),
        )
        boundary_taken = 0
        for item in boundary_rows:
            key = (item["source_gated_dir"], item["source_index"])
            if key in chosen_keys:
                continue
            chosen.append(item)
            chosen_keys.add(key)
            boundary_taken += 1
            if boundary_taken >= n_boundary:
                break

        remaining = [
            item for item in rows
            if (item["source_gated_dir"], item["source_index"]) not in chosen_keys
        ]
        if remaining and n_diverse > 0:
            rem_latents = np.stack([x["latents"] for x in remaining]).astype(np.float32)
            order = sorted(range(len(remaining)), key=lambda i: remaining[i]["margin"])
            diverse_local = greedy_diverse(order, rem_latents, n_diverse)
            for local_i in diverse_local:
                item = remaining[local_i]
                key = (item["source_gated_dir"], item["source_index"])
                if key in chosen_keys:
                    continue
                chosen.append(item)
                chosen_keys.add(key)

        if len(chosen) < args.quota:
            for item in high_rows:
                key = (item["source_gated_dir"], item["source_index"])
                if key in chosen_keys:
                    continue
                chosen.append(item)
                chosen_keys.add(key)
                if len(chosen) >= args.quota:
                    break

        selected.extend(chosen[:args.quota])
        print(
            f"[select] {cls}: selected {len(chosen[:args.quota])} / candidates {len(rows)} "
            f"(high={n_high}, boundary={n_boundary}, diverse={n_diverse})"
        )
        for item in chosen[:args.quota]:
            meta = dict(item["row"])
            meta.update({
                "source_gated_dir": item["source_gated_dir"],
                "source_index": item["source_index"],
                "p_target": item["p_target"],
                "target_margin": item["margin"],
            })
            report_selected.append(meta)

    if not selected:
        raise RuntimeError("no selected samples")

    signals = np.stack([x["signals"] for x in selected]).astype(np.float32)
    raw_signal_ct = np.stack([x["raw_signal_ct"] for x in selected]).astype(np.float32)
    latents = np.stack([x["latents"] for x in selected]).astype(np.float32)
    labels = np.stack([x["labels"] for x in selected]).astype(np.float32)
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
    report_path = out_dir / "mixed_selection_report.json"

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
            "policy": "mixed high-confidence + boundary + latent-diverse selection; HYP/CD hardcoded 0",
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
            "high_frac": float(args.high_frac),
            "boundary_frac": float(args.boundary_frac),
            "boundary_target_p": float(args.boundary_target_p),
            "n_samples": int(labels.shape[0]),
            "counts": dict(counts),
            "selected": report_selected,
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
