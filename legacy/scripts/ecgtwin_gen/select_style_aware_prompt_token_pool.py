"""Select prompt-token samples with target-center style awareness.

This selector is intentionally a candidate-pool builder, not a proof of center
token validity. It uses the real-only center-style probe scores as a selection
signal, then writes the same `gated_samples.*` files consumed by Latent-Hull
online AT. The selected pools must still be validated by independent no-leak
same-label probes/C2ST before downstream training is interpreted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def norm_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def read_style_rows(style_csvs: Iterable[str]) -> Dict[Tuple[str, int], Dict]:
    rows: Dict[Tuple[str, int], Dict] = {}
    for csv_s in style_csvs:
        path = Path(csv_s)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (norm_path(row["input_dir"]), int(row["sample_index"]))
                rows[key] = row
    if not rows:
        raise RuntimeError("no style rows loaded")
    return rows


def target_margin(row: Dict) -> float:
    cls = str(row.get("class", "")).upper()
    probs = row.get("victim_probs", {})
    p_target = float(row.get("p_target", 0.0))
    other = [float(v) for k, v in probs.items() if str(k).upper() != cls]
    runner_up = max(other) if other else 0.0
    return float(p_target - runner_up)


def style_score(row: Dict, style_row: Dict, args: argparse.Namespace) -> float:
    p_style = float(style_row.get("prob_expected", 0.0))
    p_target = float(row.get("p_target", 0.0))
    pred_center = str(style_row.get("pred_center", ""))
    expected_center = str(style_row.get("center", ""))
    margin = target_margin(row)

    score = args.style_weight * p_style
    score += args.class_weight * p_target
    score += args.margin_weight * margin
    score -= args.boundary_weight * abs(p_target - args.boundary_target_p)
    if pred_center == expected_center:
        score += args.style_top1_bonus
    if row.get("digital_metrics", {}).get("unreliable_signal"):
        score -= 1.0
    # Very saturated classifier samples were repeatedly less useful downstream.
    if p_target > args.saturation_p:
        score -= args.saturation_penalty
    return float(score)


def boundary_rank(row: Dict, style_row: Dict, args: argparse.Namespace) -> float:
    p_style = float(style_row.get("prob_expected", 0.0))
    p_target = float(row.get("p_target", 0.0))
    # Lower is better: keep target-style samples near the adversarial boundary.
    return float(abs(p_target - args.boundary_target_p) - args.boundary_style_weight * p_style)


def greedy_diverse(indices: List[int], latents: np.ndarray, n_take: int, selected: List[int] | None = None) -> List[int]:
    if n_take <= 0 or not indices:
        return []
    flat = latents.reshape(latents.shape[0], -1)
    chosen = list(selected or [])
    out: List[int] = []
    remaining = list(indices)
    if not chosen and remaining:
        first = remaining.pop(0)
        chosen.append(first)
        out.append(first)
    while remaining and len(out) < n_take:
        sel_flat = flat[np.asarray(chosen)]
        rem_flat = flat[np.asarray(remaining)]
        diff = rem_flat[:, None, :] - sel_flat[None, :, :]
        dist2 = np.einsum("nmd,nmd->nm", diff, diff)
        best_i = int(np.argmax(dist2.min(axis=1)))
        picked = remaining.pop(best_i)
        chosen.append(picked)
        out.append(picked)
    return out


def load_candidates(
    gated_dirs: Iterable[str],
    style_rows: Dict[Tuple[str, int], Dict],
    allowed: set[str],
    min_style_prob: float,
) -> Tuple[Dict[str, List[Dict]], set[str], str | None, str | None]:
    candidates: Dict[str, List[Dict]] = defaultdict(list)
    ref_ids: set[str] = set()
    center_name: str | None = None
    arm_name: str | None = None

    for gated_dir_s in gated_dirs:
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
        local_center = str(arrays["center_name"]) if "center_name" in arrays else gated_dir.parent.name
        local_arm = gated_dir.parent.parent.name
        if center_name is None:
            center_name = local_center
        if arm_name is None:
            arm_name = local_arm
        if local_center != center_name:
            raise ValueError(f"mixed centers are not supported: {center_name} and {local_center}")

        gated_key_root = norm_path(gated_dir)
        for local_i, src_i in enumerate(source_indices.astype(int).tolist()):
            style_row = style_rows.get((gated_key_root, int(local_i)))
            if style_row is None:
                continue
            p_style = float(style_row.get("prob_expected", 0.0))
            if p_style < min_style_prob:
                continue
            row = dict(rows_by_index[src_i])
            cls = str(row["class"]).upper()
            if cls not in allowed or not row.get("keep", False):
                continue
            candidates[cls].append({
                "row": row,
                "style_row": dict(style_row),
                "signals": arrays["signals"][local_i],
                "raw_signal_ct": arrays["raw_signal_ct"][local_i],
                "latents": arrays["latents"][local_i],
                "labels": arrays["labels"][local_i],
                "source_gated_dir": str(gated_dir),
                "source_index": int(src_i),
                "sample_index": int(local_i),
                "p_style": p_style,
                "p_target": float(row.get("p_target", 0.0)),
                "target_margin": target_margin(row),
                "arm": local_arm,
                "center": local_center,
            })

        if ref_meta_path.exists():
            ref_meta = load_json(ref_meta_path)
            ref_ids.update(str(x) for x in ref_meta.get("ref_record_ids", []))

    return candidates, ref_ids, center_name, arm_name


def select_class_rows(rows: List[Dict], quota: int, args: argparse.Namespace) -> List[Dict]:
    if quota <= 0 or not rows:
        return []
    n_high = min(quota, int(round(quota * args.high_frac)))
    n_boundary = min(quota - n_high, int(round(quota * args.boundary_frac)))
    n_diverse = max(0, quota - n_high - n_boundary)

    selected: List[Dict] = []
    selected_keys = set()

    scored = sorted(
        rows,
        key=lambda x: style_score(x["row"], x["style_row"], args),
        reverse=True,
    )
    for item in scored:
        key = (item["source_gated_dir"], item["sample_index"])
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= n_high:
            break

    boundary = sorted(rows, key=lambda x: boundary_rank(x["row"], x["style_row"], args))
    boundary_taken = 0
    for item in boundary:
        key = (item["source_gated_dir"], item["sample_index"])
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        boundary_taken += 1
        if boundary_taken >= n_boundary:
            break

    remaining = [x for x in rows if (x["source_gated_dir"], x["sample_index"]) not in selected_keys]
    if remaining and n_diverse > 0:
        # Diverse fill starts from the most style-like remaining candidates.
        ordered = sorted(
            range(len(remaining)),
            key=lambda i: style_score(remaining[i]["row"], remaining[i]["style_row"], args),
            reverse=True,
        )
        latents = np.stack([x["latents"] for x in remaining]).astype(np.float32)
        diverse = greedy_diverse(ordered, latents, n_diverse)
        for local_i in diverse:
            item = remaining[local_i]
            key = (item["source_gated_dir"], item["sample_index"])
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)

    if len(selected) < quota:
        for item in scored:
            key = (item["source_gated_dir"], item["sample_index"])
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) >= quota:
                break

    return selected[:quota]


def write_pool(
    selected: List[Dict],
    ref_ids: set[str],
    center_name: str | None,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not selected:
        raise RuntimeError("no selected samples")
    signals = np.stack([x["signals"] for x in selected]).astype(np.float32)
    raw_signal_ct = np.stack([x["raw_signal_ct"] for x in selected]).astype(np.float32)
    latents = np.stack([x["latents"] for x in selected]).astype(np.float32)
    labels = np.stack([x["labels"] for x in selected]).astype(np.float32)
    counts = Counter(CLASS_NAMES_SUPER5[int(i)] for i in labels.argmax(axis=1))
    class_trust = {cls: (1.0 if counts.get(cls, 0) > 0 else 0.0) for cls in CLASS_NAMES_SUPER5}
    if not args.allow_hyp_cd_trust:
        class_trust["HYP"] = 0.0
        class_trust["CD"] = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "gated_samples.npz"
    latent_path = out_dir / "gated_samples.latent.npz"
    trust_path = out_dir / "gated_samples.class_trust.json"
    ref_meta_path = out_dir / "gated_samples.ref_meta.json"
    report_path = out_dir / "style_aware_selection_report.json"
    summary_path = out_dir.parent / "summary.json" if out_dir.name == "gated" else out_dir / "summary.json"

    source_indices = np.asarray([x["source_index"] for x in selected], dtype=np.int64)
    source_sample_indices = np.asarray([x["sample_index"] for x in selected], dtype=np.int64)
    source_gated_dirs = np.asarray([x["source_gated_dir"] for x in selected])

    np.savez_compressed(
        samples_path,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=center_name or "?",
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_indices=source_indices,
        source_sample_indices=source_sample_indices,
        source_gated_dirs=source_gated_dirs,
    )
    np.savez_compressed(
        latent_path,
        latents=latents,
        labels=labels,
        center_name=center_name or "?",
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_indices=source_indices,
        source_sample_indices=source_sample_indices,
        source_gated_dirs=source_gated_dirs,
    )
    with trust_path.open("w") as f:
        json.dump({
            "tag": args.tag,
            "center": center_name,
            "class_trust": class_trust,
            "counts": dict(counts),
            "policy": "style-aware selection with class semantics, boundary, and latent-diverse fill",
        }, f, indent=2)
    with ref_meta_path.open("w") as f:
        json.dump({
            "center": center_name,
            "ref_record_ids": sorted(ref_ids),
            "policy": "Union of source K-ref exclusions.",
        }, f, indent=2)

    selected_meta = []
    for item in selected:
        row = dict(item["row"])
        style_row = dict(item["style_row"])
        row.update({
            "source_gated_dir": item["source_gated_dir"],
            "source_index": item["source_index"],
            "sample_index": item["sample_index"],
            "p_style": item["p_style"],
            "style_pred_center": style_row.get("pred_center"),
            "style_prob_expected": style_row.get("prob_expected"),
            "target_margin": item["target_margin"],
            "style_aware_score": style_score(item["row"], item["style_row"], args),
        })
        selected_meta.append(row)
    with report_path.open("w") as f:
        json.dump({
            "tag": args.tag,
            "center": center_name,
            "gated_dirs": list(args.gated_dirs),
            "style_csvs": list(args.style_csv),
            "quota": int(args.quota),
            "classes": list(args.classes),
            "n_samples": int(labels.shape[0]),
            "counts": dict(counts),
            "config": vars(args),
            "selected": selected_meta,
            "samples": str(samples_path),
            "latents": str(latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(ref_meta_path),
        }, f, indent=2)
    with summary_path.open("w") as f:
        json.dump({
            "config": vars(args),
            "npz": str(samples_path),
            "n_samples": int(signals.shape[0]),
            "signal_shape": list(signals.shape),
            "raw_signal_shape": list(raw_signal_ct.shape),
            "latents_shape": list(latents.shape),
            "records": selected_meta,
            "selection_report": str(report_path),
        }, f, indent=2)

    print(f"[select] wrote {latent_path}")
    print(f"[select] wrote {summary_path}")
    print(f"[select] counts: {dict(counts)}")
    print(f"[select] ref ids: {len(ref_ids)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated_dirs", nargs="+", required=True)
    ap.add_argument("--style_csv", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--quota", type=int, default=40)
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--min_style_prob", type=float, default=0.0)
    ap.add_argument("--high_frac", type=float, default=0.55)
    ap.add_argument("--boundary_frac", type=float, default=0.25)
    ap.add_argument("--style_weight", type=float, default=1.0)
    ap.add_argument("--class_weight", type=float, default=0.25)
    ap.add_argument("--margin_weight", type=float, default=0.05)
    ap.add_argument("--boundary_weight", type=float, default=0.15)
    ap.add_argument("--boundary_style_weight", type=float, default=0.25)
    ap.add_argument("--boundary_target_p", type=float, default=0.60)
    ap.add_argument("--style_top1_bonus", type=float, default=0.10)
    ap.add_argument("--saturation_p", type=float, default=0.98)
    ap.add_argument("--saturation_penalty", type=float, default=0.02)
    ap.add_argument("--tag", default="style_aware_prompt_token_pool")
    ap.add_argument("--allow_hyp_cd_trust", action="store_true")
    args = ap.parse_args()

    allowed = {c.upper() for c in args.classes}
    style_rows = read_style_rows(args.style_csv)
    candidates, ref_ids, center_name, arm_name = load_candidates(
        args.gated_dirs,
        style_rows,
        allowed,
        args.min_style_prob,
    )

    selected: List[Dict] = []
    for cls in [c.upper() for c in args.classes]:
        rows = candidates.get(cls, [])
        take = select_class_rows(rows, args.quota, args)
        selected.extend(take)
        print(f"[select] {center_name} {arm_name} {cls}: selected {len(take)} / candidates {len(rows)}")

    write_pool(selected, ref_ids, center_name, Path(args.out_dir), args)


if __name__ == "__main__":
    main()
