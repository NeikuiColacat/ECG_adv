#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.classifier_backend import DEFAULT_CKPT, PyTorchClassifierBackend
from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, to_signal_ct
from apps.streamlit_ecg_demo.services.quality_gate import run_quality_gate


DEFAULT_INPUTS = [
    (
        "target_token_s05",
        "/root/autodl-tmp/ecgtwin_prompt_token_super5/"
        "effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/"
        "target_token_s05/ningbo/gated/gated_samples.npz",
    ),
    (
        "no_token",
        "/root/autodl-tmp/ecgtwin_prompt_token_super5/"
        "effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/"
        "no_token/ningbo/gated/gated_samples.npz",
    ),
    (
        "v4_balanced_token",
        "/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/"
        "ningbo/gated/gated_samples.npz",
    ),
]


def load_pool(path: str, cap: int):
    data = np.load(path, allow_pickle=True)
    signals = data["signals"]
    labels = data["labels"] if "labels" in data else np.full((len(signals), len(CLASS_NAMES)), np.nan)
    class_names = [str(x) for x in data["class_names"]] if "class_names" in data else CLASS_NAMES
    n = min(len(signals), cap)
    return signals[:n], labels[:n], class_names


def intended_label_names(label, class_names):
    return [class_names[i] for i, v in enumerate(label) if float(v) > 0.5]


def summarize_rows(rows):
    by_arm = {}
    for row in rows:
        arm = row["arm"]
        rec = by_arm.setdefault(arm, {
            "n": 0,
            "pass": 0,
            "warning": 0,
            "fail": 0,
            "semantic_top1": 0,
            "semantic_any_positive_ge_05": 0,
            "mean_target_prob": [],
            "mean_einthoven": [],
            "mean_avr": [],
        })
        rec["n"] += 1
        rec[row["quality_status"]] += 1
        rec["semantic_top1"] += int(row["semantic_top1"])
        rec["semantic_any_positive_ge_05"] += int(row["semantic_any_positive_ge_05"])
        rec["mean_target_prob"].append(float(row["target_prob_max"]))
        if np.isfinite(row["einthoven_residual"]):
            rec["mean_einthoven"].append(float(row["einthoven_residual"]))
        if np.isfinite(row["avR_residual"]):
            rec["mean_avr"].append(float(row["avR_residual"]))
    out = {}
    for arm, rec in by_arm.items():
        n = max(rec["n"], 1)
        out[arm] = {
            "n": rec["n"],
            "quality_pass_rate": rec["pass"] / n,
            "quality_warning_rate": rec["warning"] / n,
            "quality_fail_rate": rec["fail"] / n,
            "semantic_top1_rate": rec["semantic_top1"] / n,
            "semantic_any_positive_ge_05_rate": rec["semantic_any_positive_ge_05"] / n,
            "mean_target_prob": float(np.mean(rec["mean_target_prob"])) if rec["mean_target_prob"] else None,
            "mean_einthoven_residual": float(np.mean(rec["mean_einthoven"])) if rec["mean_einthoven"] else None,
            "mean_avR_residual": float(np.mean(rec["mean_avr"])) if rec["mean_avr"] else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", default=[],
                    help="Items like arm=/path/to/gated_samples.npz")
    ap.add_argument("--cap_per_arm", type=int, default=128)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/final_round_ablation_20260504/medical_validity")
    args = ap.parse_args()

    inputs = []
    if args.input:
        for item in args.input:
            arm, path = item.split("=", 1)
            inputs.append((arm, path))
    else:
        inputs = DEFAULT_INPUTS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = PyTorchClassifierBackend(ckpt_path=args.ckpt, device=args.device)
    rows = []
    for arm, path in inputs:
        if not Path(path).exists():
            continue
        signals, labels, class_names = load_pool(path, args.cap_per_arm)
        for i, (signal, label) in enumerate(zip(signals, labels)):
            sig_ct = to_signal_ct(signal)
            gate = run_quality_gate(sig_ct)
            pred = backend.predict(sig_ct)
            probs = np.asarray(pred["probabilities"], dtype=np.float32)
            positives = [j for j, v in enumerate(label) if float(v) > 0.5]
            target_prob_max = float(max([probs[j] for j in positives], default=np.nan))
            semantic_top1 = bool(positives and int(np.argmax(probs)) in positives)
            semantic_any = bool(positives and any(float(probs[j]) >= 0.5 for j in positives))
            rows.append({
                "arm": arm,
                "source_path": path,
                "index": i,
                "label_names": "|".join(intended_label_names(label, class_names)),
                "quality_status": gate["status"],
                "warnings": "|".join(gate["warnings"]),
                "einthoven_residual": gate["einthoven_residual"],
                "avR_residual": gate["avR_residual"],
                "hr_estimate_bpm": gate["hr_estimate_bpm"],
                "target_prob_max": target_prob_max,
                "semantic_top1": semantic_top1,
                "semantic_any_positive_ge_05": semantic_any,
                **{f"prob_{name}": float(probs[j]) for j, name in enumerate(CLASS_NAMES)},
            })

    csv_path = out_dir / "per_sample_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "inputs": [{"arm": arm, "path": path} for arm, path in inputs],
        "cap_per_arm": args.cap_per_arm,
        "checkpoint": args.ckpt,
        "summary_by_arm": summarize_rows(rows),
    }
    (out_dir / "medical_validity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Medical Validity Ablation Summary", ""]
    for arm, rec in summary["summary_by_arm"].items():
        lines.append(f"## {arm}")
        lines.append("")
        lines.append(f"- n: {rec['n']}")
        lines.append(f"- quality pass rate: {rec['quality_pass_rate']:.3f}")
        lines.append(f"- semantic top1 rate: {rec['semantic_top1_rate']:.3f}")
        lines.append(f"- semantic target prob>=0.5 rate: {rec['semantic_any_positive_ge_05_rate']:.3f}")
        lines.append(f"- mean target probability: {rec['mean_target_prob']:.3f}")
        lines.append("")
    (out_dir / "medical_validity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
