"""Lightweight quality report for Super5 ECGTwin synthetic pools.

The generated pool from ``generate_center_synth.py`` is already converted to
PTB-XL lead order, resampled to 1000 points, and per-sample z-scored. Because
it is classifier-scale rather than raw mV, this report intentionally avoids
claiming voltage-threshold medical validity. It reports engineering sanity,
Einthoven residuals, HR proxies, and frozen EfficientNet Super5 consistency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, NUM_SUPER5  # noqa: E402
from util.ecg_signal_sanity import _extract_clinical_features  # noqa: E402
from util.super5_victim import EfficientNetVictimTierM  # noqa: E402


DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def _finite_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


@torch.no_grad()
def victim_probs(signals_ct: np.ndarray, ckpt: str, device: str, batch_size: int) -> np.ndarray:
    victim = EfficientNetVictimTierM(
        weight_path=ckpt,
        device=device,
        ecgtwin_wrapper=None,
        num_classes=NUM_SUPER5,
        crop_len=250,
    )
    victim.eval()
    chunks = []
    for i in range(0, signals_ct.shape[0], batch_size):
        x = torch.from_numpy(signals_ct[i:i + batch_size]).float().to(device)
        logits = victim.compute_logits_from_ecg(x)
        chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def summarize(signals: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    target_idx = labels.argmax(axis=1)
    pred_idx = probs.argmax(axis=1)
    p_target = probs[np.arange(probs.shape[0]), target_idx]

    features = [_extract_clinical_features(sig) for sig in signals]
    p2p = np.ptp(signals, axis=2)
    global_p2p = p2p.max(axis=1)
    finite = np.isfinite(signals).reshape(signals.shape[0], -1).all(axis=1)
    flat = global_p2p < 1e-3
    saturated = np.abs(signals).max(axis=(1, 2)) > 8.0

    out: Dict[str, Any] = {
        "n": int(signals.shape[0]),
        "finite_rate": float(finite.mean()),
        "flatline_rate": float(flat.mean()),
        "saturation_rate_abs_gt8": float(saturated.mean()),
        "victim_top1_hit_rate": float((pred_idx == target_idx).mean()),
        "victim_target_prob": _finite_stats(p_target.tolist()),
        "global_p2p_zscore": _finite_stats(global_p2p.tolist()),
        "hr_bpm": _finite_stats([f["hr_bpm"] for f in features]),
        "einthoven_p95": _finite_stats([f["einthoven_p95"] for f in features]),
        "qrs_amp": _finite_stats([f["qrs_amp"] for f in features]),
        "per_class": {},
    }
    for j, cls in enumerate(CLASS_NAMES_SUPER5):
        mask = target_idx == j
        cls_features = [features[i] for i in np.where(mask)[0]]
        out["per_class"][cls] = {
            "n": int(mask.sum()),
            "finite_rate": float(finite[mask].mean()) if mask.any() else None,
            "victim_top1_hit_rate": float((pred_idx[mask] == j).mean()) if mask.any() else None,
            "victim_target_prob": _finite_stats(p_target[mask].tolist()),
            "global_p2p_zscore": _finite_stats(global_p2p[mask].tolist()),
            "hr_bpm": _finite_stats([f["hr_bpm"] for f in cls_features]),
            "einthoven_p95": _finite_stats([f["einthoven_p95"] for f in cls_features]),
            "qrs_amp": _finite_stats([f["qrs_amp"] for f in cls_features]),
        }
    return out


def write_md(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "# Super5 ECGTwin Synthetic Quality Report",
        "",
        "Input signals are PTB-XL order, 100 Hz, 1000 samples, classifier-scale z-score. "
        "Voltage-threshold medical criteria are not asserted in this report.",
        "",
        "## Overall",
        "",
        f"- N: {report['n']}",
        f"- finite rate: {report['finite_rate']:.4f}",
        f"- flatline rate: {report['flatline_rate']:.4f}",
        f"- abs(z) > 8 saturation rate: {report['saturation_rate_abs_gt8']:.4f}",
        f"- victim top1 hit rate: {report['victim_top1_hit_rate']:.4f}",
        f"- victim target prob mean/p50: {report['victim_target_prob']['mean']:.4f} / {report['victim_target_prob']['p50']:.4f}",
        f"- Einthoven p95 mean/p95: {report['einthoven_p95']['mean']:.4f} / {report['einthoven_p95']['p95']:.4f}",
        f"- HR proxy mean/p50: {report['hr_bpm']['mean']:.2f} / {report['hr_bpm']['p50']:.2f}",
        "",
        "## Per Class",
        "",
        "| class | n | finite | victim top1 | target prob mean | Einthoven p95 mean | HR mean | p2p zscore p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls, row in report["per_class"].items():
        lines.append(
            f"| {cls} | {row['n']} | {row['finite_rate']:.4f} | "
            f"{row['victim_top1_hit_rate']:.4f} | "
            f"{row['victim_target_prob']['mean']:.4f} | "
            f"{row['einthoven_p95']['mean']:.4f} | "
            f"{row['hr_bpm']['mean']:.2f} | "
            f"{row['global_p2p_zscore']['p50']:.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--ckpt", default=str(DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.synth_npz)
    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.float32)
    if signals.shape[1:] != (12, 1000):
        raise ValueError(f"Expected signals (N,12,1000), got {signals.shape}")
    if labels.shape[1] != NUM_SUPER5:
        raise ValueError(f"Expected labels (N,5), got {labels.shape}")

    if args.max_samples and signals.shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(np.arange(signals.shape[0]), size=args.max_samples, replace=False)
        pick.sort()
        signals = signals[pick]
        labels = labels[pick]

    probs = victim_probs(signals, args.ckpt, args.device, args.batch_size)
    report = summarize(signals, labels, probs)
    report["synth_npz"] = args.synth_npz
    report["ckpt"] = args.ckpt
    report["max_samples"] = int(args.max_samples)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    write_md(args.out_md, report)
    print(f"[quality] wrote {args.out_json}")
    print(f"[quality] wrote {args.out_md}")


if __name__ == "__main__":
    main()
