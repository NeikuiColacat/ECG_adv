"""Prepare ECGTwin synthetic samples for the v2 self-distillation experiment.

The script implements the v2 plan consolidated in
docs/pipelines/graduate_project.md:

1. score ECGTwin synthetic candidates with one or more real-2000 teachers;
2. keep high-confidence and target-in-top2 hard samples;
3. build soft labels as gamma * condition + (1 - gamma) * teacher probability;
4. export a compact npz consumed by ptbxl_source_train_self_distill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ecg_adv_gen.runner.ptbxl_source_train_self_distill import (  # noqa: E402
    CLASS_NAMES,
    EvalDataset,
    build_model,
    load_state_dict_into,
    normalize_synth_signals,
)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_teacher_ckpts(raw: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw:
        out.extend([p for p in item.split(",") if p])
    return out


@torch.no_grad()
def predict_probs(
    ckpt: str,
    signals_tc: np.ndarray,
    crop_len: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model = build_model(device)
    load_state_dict_into(model, ckpt, device)
    model.eval()
    dummy = np.zeros((signals_tc.shape[0], len(CLASS_NAMES)), dtype=np.float32)
    loader = DataLoader(
        EvalDataset(signals_tc, dummy, crop_len=crop_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        persistent_workers=num_workers > 0,
    )
    probs: List[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs.append(p)
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()
    return np.concatenate(probs, axis=0)


def target_index_from_label(label: np.ndarray) -> int:
    positives = np.flatnonzero(label > 0.5)
    if positives.size == 0:
        return int(np.argmax(label))
    if positives.size == 1:
        return int(positives[0])
    return int(positives[np.argmax(label[positives])])


def classify_quality(
    probs: np.ndarray,
    labels: np.ndarray,
    tau_high: float,
    tau_low: float,
    hard_weight: float,
    norm_tau_high: float,
    norm_tau_low: float,
    norm_max_abnormal_high: float,
    norm_max_abnormal_hard: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = probs.shape[0]
    target_idx = np.zeros(n, dtype=np.int64)
    quality = np.zeros(n, dtype=np.int64)  # 0 reject, 1 hard, 2 high
    sample_weights = np.zeros(n, dtype=np.float32)
    target_conf = np.zeros(n, dtype=np.float32)
    norm_idx = CLASS_NAMES.index("NORM")

    for i in range(n):
        t = target_index_from_label(labels[i])
        target_idx[i] = t
        p = probs[i]
        top2 = set(np.argsort(p)[-2:].tolist())
        target_conf[i] = float(p[t])
        if t == norm_idx:
            abnormal = [j for j in range(len(CLASS_NAMES)) if j != norm_idx]
            max_abn = float(np.max(p[abnormal]))
            if p[norm_idx] >= norm_tau_high and max_abn <= norm_max_abnormal_high:
                quality[i] = 2
                sample_weights[i] = 1.0
            elif p[norm_idx] >= norm_tau_low and max_abn <= norm_max_abnormal_hard and norm_idx in top2:
                quality[i] = 1
                sample_weights[i] = float(hard_weight)
            continue
        if target_conf[i] >= tau_high and t in top2:
            quality[i] = 2
            sample_weights[i] = 1.0
        elif target_conf[i] >= tau_low and t in top2:
            quality[i] = 1
            sample_weights[i] = float(hard_weight)
    return quality, sample_weights, target_idx, target_conf


def select_balanced(
    keep_mask: np.ndarray,
    quality: np.ndarray,
    target_idx: np.ndarray,
    target_conf: np.ndarray,
    max_keep_total: int,
    per_class_cap: int,
    selection_order: str = "confidence",
    boundary_center: float = 0.55,
) -> np.ndarray:
    candidates = np.flatnonzero(keep_mask)
    if max_keep_total <= 0 and per_class_cap <= 0:
        return candidates

    selected: List[int] = []
    default_cap = int(np.ceil(max_keep_total / len(CLASS_NAMES))) if max_keep_total > 0 else 0
    for class_idx in range(len(CLASS_NAMES)):
        idx = candidates[target_idx[candidates] == class_idx]
        if idx.size == 0:
            continue
        if selection_order == "confidence":
            order = np.lexsort((-target_conf[idx], -quality[idx]))
        elif selection_order == "boundary":
            boundary_score = np.abs(target_conf[idx] - float(boundary_center))
            order = np.lexsort((boundary_score, -quality[idx]))
        else:
            raise ValueError(f"Unknown selection_order: {selection_order}")
        ranked = idx[order]
        cap = per_class_cap if per_class_cap > 0 else default_cap
        if cap > 0:
            ranked = ranked[:cap]
        selected.extend(ranked.tolist())

    selected_arr = np.asarray(selected, dtype=np.int64)
    if max_keep_total > 0 and selected_arr.size > max_keep_total:
        if selection_order == "confidence":
            order = np.lexsort((-target_conf[selected_arr], -quality[selected_arr]))
        elif selection_order == "boundary":
            boundary_score = np.abs(target_conf[selected_arr] - float(boundary_center))
            order = np.lexsort((boundary_score, -quality[selected_arr]))
        else:
            raise ValueError(f"Unknown selection_order: {selection_order}")
        selected_arr = selected_arr[order[:max_keep_total]]
    selected_arr.sort()
    return selected_arr


def summarize_selection(
    labels: np.ndarray,
    probs: np.ndarray,
    selected: np.ndarray,
    quality: np.ndarray,
    target_idx: np.ndarray,
    target_conf: np.ndarray,
) -> Dict:
    out: Dict[str, object] = {
        "n_candidates": int(labels.shape[0]),
        "n_kept": int(selected.size),
        "n_high": int((quality[selected] == 2).sum()),
        "n_hard": int((quality[selected] == 1).sum()),
        "target_conf_mean": float(target_conf[selected].mean()) if selected.size else None,
        "target_conf_median": float(np.median(target_conf[selected])) if selected.size else None,
        "per_class": {},
    }
    for i, name in enumerate(CLASS_NAMES):
        cand = np.flatnonzero(target_idx == i)
        kept = selected[target_idx[selected] == i]
        out["per_class"][name] = {
            "candidates": int(cand.size),
            "kept": int(kept.size),
            "high": int((quality[kept] == 2).sum()),
            "hard": int((quality[kept] == 1).sum()),
            "candidate_target_conf_mean": float(target_conf[cand].mean()) if cand.size else None,
            "kept_target_conf_mean": float(target_conf[kept].mean()) if kept.size else None,
            "kept_teacher_prob_mean": probs[kept].mean(axis=0).tolist() if kept.size else None,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synth_npz", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--summary_json", default="")
    p.add_argument("--teacher_ckpts", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--tau_high", type=float, default=0.6)
    p.add_argument("--tau_low", type=float, default=0.35)
    p.add_argument("--hard_weight", type=float, default=0.5)
    p.add_argument("--norm_tau_high", type=float, default=0.7)
    p.add_argument("--norm_tau_low", type=float, default=0.5)
    p.add_argument("--norm_max_abnormal_high", type=float, default=0.3)
    p.add_argument("--norm_max_abnormal_hard", type=float, default=0.45)
    p.add_argument("--target_conf_min", type=float, default=0.0)
    p.add_argument("--target_conf_max", type=float, default=1.0)
    p.add_argument("--selection_order", choices=["confidence", "boundary"], default="confidence")
    p.add_argument("--boundary_center", type=float, default=0.55)
    p.add_argument("--max_keep_total", type=int, default=2000)
    p.add_argument("--per_class_cap", type=int, default=400)
    args = p.parse_args()

    teacher_ckpts = parse_teacher_ckpts(args.teacher_ckpts)
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    if not args.summary_json:
        args.summary_json = os.path.splitext(args.output_npz)[0] + ".summary.json"

    data = np.load(args.synth_npz, allow_pickle=True)
    signals = normalize_synth_signals(data["signals"])
    label_key = "labels" if "labels" in data else "labels5"
    labels = np.asarray(data[label_key], dtype=np.float32)
    if labels.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected labels shape (N,5), got {labels.shape}")
    print(f"[data] candidates={signals.shape[0]} from {args.synth_npz}", flush=True)

    prob_sum = np.zeros_like(labels, dtype=np.float32)
    for ckpt in teacher_ckpts:
        print(f"[teacher] scoring with {ckpt}", flush=True)
        prob_sum += predict_probs(
            ckpt,
            signals,
            crop_len=args.crop_len,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    teacher_probs = prob_sum / float(len(teacher_ckpts))

    quality, sample_weights, target_idx, target_conf = classify_quality(
        teacher_probs,
        labels,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        hard_weight=args.hard_weight,
        norm_tau_high=args.norm_tau_high,
        norm_tau_low=args.norm_tau_low,
        norm_max_abnormal_high=args.norm_max_abnormal_high,
        norm_max_abnormal_hard=args.norm_max_abnormal_hard,
    )
    keep_mask = quality > 0
    keep_mask &= target_conf >= float(args.target_conf_min)
    keep_mask &= target_conf <= float(args.target_conf_max)
    selected = select_balanced(
        keep_mask,
        quality,
        target_idx,
        target_conf,
        max_keep_total=args.max_keep_total,
        per_class_cap=args.per_class_cap,
        selection_order=args.selection_order,
        boundary_center=args.boundary_center,
    )
    soft_labels = (
        float(args.gamma) * labels[selected]
        + (1.0 - float(args.gamma)) * teacher_probs[selected]
    )
    soft_labels = np.clip(soft_labels, 1e-4, 1.0 - 1e-4).astype(np.float32)
    summary = summarize_selection(labels, teacher_probs, selected, quality, target_idx, target_conf)
    summary["config"] = vars(args)
    summary["teacher_ckpts"] = teacher_ckpts
    summary["class_names"] = CLASS_NAMES

    np.savez_compressed(
        args.output_npz,
        signals=signals[selected].astype(np.float32, copy=False),
        labels=labels[selected].astype(np.float32, copy=False),
        soft_labels=soft_labels,
        teacher_probs=teacher_probs[selected].astype(np.float32, copy=False),
        sample_weights=sample_weights[selected].astype(np.float32, copy=False),
        quality=quality[selected].astype(np.int64, copy=False),
        target_class=target_idx[selected].astype(np.int64, copy=False),
        target_conf=target_conf[selected].astype(np.float32, copy=False),
        source_indices=selected.astype(np.int64, copy=False),
        class_names=np.asarray(CLASS_NAMES),
        metadata_json=np.asarray(json.dumps(summary, default=_json_default, sort_keys=True)),
    )
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(
        f"[done] kept={summary['n_kept']} high={summary['n_high']} hard={summary['n_hard']} "
        f"saved={args.output_npz}",
        flush=True,
    )
    for name, stats in summary["per_class"].items():
        print(
            f"  {name:<4} candidates={stats['candidates']:5d} kept={stats['kept']:4d} "
            f"high={stats['high']:4d} hard={stats['hard']:4d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
