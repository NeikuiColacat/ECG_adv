"""Gate and export ECGTwin prompt-token synthetic samples.

Input is the output directory from `generate_center_prompt_token_synth.py`
containing:

  samples.npz
  summary.json

The script applies class-specific digital ECG gates plus a light classifier
confidence gate, then writes files directly consumable by
`scripts/pgd_cross_center/synth_online_at_super5.py`:

  gated_samples.npz
  gated_samples.latent.npz
  gated_samples.class_trust.json
  gated_samples.ref_meta.json
  gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.paths import DATA_ROOT  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from util.ecg_digital_features import (  # noqa: E402
    check_cd,
    check_hyp,
    check_mi,
    check_norm,
    check_sttc,
    extract_digital_features,
)


DEFAULT_CACHE_ROOT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1")

PASS_KEY = {
    "NORM": "NORM_any_pass",
    "MI": "MI_pass",
    "STTC": "STTC_pass",
    "HYP": "HYP_pass",
    "CD": "CD_pass",
}


def parse_thresholds(items: Iterable[str] | None, default: float) -> Dict[str, float]:
    thresholds = {cls: float(default) for cls in CLASS_NAMES_SUPER5}
    if not items:
        return thresholds
    for item in items:
        if "=" not in item:
            raise ValueError(f"bad threshold item {item!r}; expected CLASS=value")
        cls, value = item.split("=", 1)
        cls = cls.strip().upper()
        if cls not in thresholds:
            raise ValueError(f"unknown class {cls!r}; valid={CLASS_NAMES_SUPER5}")
        thresholds[cls] = float(value)
    return thresholds


def class_from_label(label: np.ndarray) -> str:
    if label.ndim != 1:
        raise ValueError(f"expected 1D label, got shape={label.shape}")
    return CLASS_NAMES_SUPER5[int(np.argmax(label))]


def digital_scores(sig_ct: np.ndarray, fs: float, lead_order: str) -> Tuple[Dict, Dict[str, Dict]]:
    feats = extract_digital_features(sig_ct, fs=fs, lead_order=lead_order)
    scores = {
        "NORM": check_norm(feats),
        "MI": check_mi(feats),
        "STTC": check_sttc(feats),
        "HYP": check_hyp(feats),
        "CD": check_cd(feats),
    }
    return feats, scores


def target_digital_pass(cls: str, scores: Dict[str, Dict]) -> bool:
    return bool(scores[cls].get(PASS_KEY[cls], False))


def compact_metrics(cls: str, feats: Dict, scores: Dict[str, Dict]) -> Dict:
    out = {
        "hr_bpm": feats.get("hr_bpm"),
        "qrs_duration_ms_avg": feats.get("qrs_duration_ms_avg"),
        "qrs_duration_ms_broadest": feats.get("qrs_duration_ms_broadest"),
        "unreliable_signal": bool(feats.get("unreliable_signal", False)),
    }
    if cls == "MI":
        out["mi_ste_2contig"] = bool(scores["MI"].get("MI1_ste_2contig", False))
    elif cls == "STTC":
        out["sttc_depression"] = bool(scores["STTC"].get("ST1_depression", False))
        out["sttc_t_inversion"] = bool(scores["STTC"].get("ST2_t_inversion", False))
    elif cls == "HYP":
        out["sokolow_mv"] = scores["HYP"].get("sokolow_mv")
        out["cornell_mv"] = scores["HYP"].get("cornell_mv")
    elif cls == "CD":
        out["lbbb_pass"] = bool(scores["CD"].get("LBBB_pass", False))
        out["rbbb_pass"] = bool(scores["CD"].get("RBBB_pass", False))
        out["avb_pass"] = bool(scores["CD"].get("AVB_pass", False))
    elif cls == "NORM":
        out["norm_strict_pass"] = bool(scores["NORM"].get("NORM_strict_pass", False))
        out["norm_ratevariant_pass"] = bool(scores["NORM"].get("NORM_ratevariant_pass", False))
        out["norm_afib_pass"] = bool(scores["NORM"].get("NORM_afib_pass", False))
    return out


def load_k_ref_ids(cache_root: Path, center: str, k: int, selection_seed: int) -> List[str]:
    cache_path = cache_root / "center_full_latents" / f"{center}.pt"
    selection_path = cache_root / "ref_selection" / f"{center}_k{k}_seed{selection_seed}.json"
    if not cache_path.exists() or not selection_path.exists():
        return []
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    with open(selection_path) as f:
        selection = json.load(f)
    selected = selection.get("selected_indices_in_full_cache", [])
    record_ids = cache.get("record_ids", [])
    return [str(record_ids[int(i)]) for i in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing samples.npz and summary.json")
    ap.add_argument("--out_dir", default=None, help="Default: <input_dir>/gated")
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--fs", type=float, default=102.4)
    ap.add_argument("--lead_order", choices=["mimic", "ptbxl"], default="mimic")
    ap.add_argument("--min_target_prob", type=float, default=0.30)
    ap.add_argument("--class_min_target_prob", nargs="*", default=None,
                    help="Optional overrides like NORM=0.50 MI=0.25 STTC=0.30")
    ap.add_argument("--require_top1", action="store_true")
    ap.add_argument("--min_pass_per_class", type=int, default=1)
    ap.add_argument("--max_pass_per_class", type=int, default=0,
                    help="If >0, keep at most this many passed samples per class.")
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--selection_seed", type=int, default=42)
    ap.add_argument("--allow_hyp_cd_trust", action="store_true",
                    help="Do not force HYP/CD trust to zero after digital/prob gates.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    samples_path = input_dir / "samples.npz"
    summary_path = input_dir / "summary.json"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir / "gated"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(samples_path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32)
        raw_signal_ct = data["raw_signal_ct"].astype(np.float32)
        latents = data["latents"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
        center_name = str(data["center_name"]) if "center_name" in data.files else input_dir.name
        class_names = data["class_names"] if "class_names" in data.files else np.asarray(CLASS_NAMES_SUPER5)
    with open(summary_path) as f:
        summary = json.load(f)
    records = summary.get("records", [])
    if len(records) != labels.shape[0]:
        raise ValueError(f"summary records ({len(records)}) != samples ({labels.shape[0]})")
    if raw_signal_ct.shape[0] != labels.shape[0] or latents.shape[0] != labels.shape[0]:
        raise ValueError("samples.npz arrays have inconsistent first dimension")

    allowed = {cls.upper() for cls in args.classes}
    thresholds = parse_thresholds(args.class_min_target_prob, args.min_target_prob)

    passed_indices: List[int] = []
    per_sample = []
    for i, rec in enumerate(records):
        cls = str(rec.get("class") or class_from_label(labels[i])).upper()
        p_target = float(rec.get("p_target", np.nan))
        top1 = str(rec.get("top1", ""))
        feats, scores = digital_scores(raw_signal_ct[i], fs=args.fs, lead_order=args.lead_order)
        digital_pass = target_digital_pass(cls, scores)
        prob_pass = bool(np.isfinite(p_target) and p_target >= thresholds[cls])
        top1_pass = bool((not args.require_top1) or top1 == cls)
        class_allowed = cls in allowed
        finite = bool(np.isfinite(signals[i]).all() and np.isfinite(raw_signal_ct[i]).all() and np.isfinite(latents[i]).all())
        keep = bool(class_allowed and finite and digital_pass and prob_pass and top1_pass)
        if keep:
            passed_indices.append(i)
        row = dict(rec)
        row.update({
            "index": i,
            "class": cls,
            "digital_pass": digital_pass,
            "prob_pass": prob_pass,
            "top1_pass": top1_pass,
            "class_allowed": class_allowed,
            "finite": finite,
            "keep": keep,
            "min_target_prob": thresholds[cls],
            "digital_metrics": compact_metrics(cls, feats, scores),
        })
        per_sample.append(row)

    if args.max_pass_per_class > 0:
        capped = []
        capped_counts = Counter()
        for i in passed_indices:
            cls = per_sample[i]["class"]
            if capped_counts[cls] >= args.max_pass_per_class:
                per_sample[i]["keep"] = False
                per_sample[i]["cap_pass"] = False
                continue
            capped.append(i)
            capped_counts[cls] += 1
            per_sample[i]["cap_pass"] = True
        passed_indices = capped

    if not passed_indices:
        raise RuntimeError("no samples passed gates")
    idx = np.asarray(passed_indices, dtype=np.int64)
    gated_samples_path = out_dir / "gated_samples.npz"
    gated_latent_path = out_dir / "gated_samples.latent.npz"
    np.savez_compressed(
        gated_samples_path,
        signals=signals[idx],
        raw_signal_ct=raw_signal_ct[idx],
        latents=latents[idx],
        labels=labels[idx],
        center_name=center_name,
        class_names=class_names,
        source_indices=idx,
    )
    np.savez_compressed(
        gated_latent_path,
        latents=latents[idx],
        labels=labels[idx],
        center_name=center_name,
        class_names=class_names,
        source_indices=idx,
    )

    passed_classes = [per_sample[i]["class"] for i in passed_indices]
    pass_counts = Counter(passed_classes)
    total_counts = Counter(row["class"] for row in per_sample)
    class_trust = {}
    for cls in CLASS_NAMES_SUPER5:
        class_trust[cls] = 1.0 if pass_counts.get(cls, 0) >= args.min_pass_per_class else 0.0
    # Keep the project-level HYP/CD caution by default. PTB-XL source-token
    # experiments can opt into all-5-class strict-gated trust.
    hardcoded_zero = []
    if not args.allow_hyp_cd_trust:
        class_trust["HYP"] = 0.0
        class_trust["CD"] = 0.0
        hardcoded_zero = ["HYP", "CD"]
    trust_path = out_dir / "gated_samples.class_trust.json"
    with open(trust_path, "w") as f:
        json.dump({
            "tag": center_name,
            "source_samples": str(samples_path),
            "gated_latent_pool": str(gated_latent_path),
            "class_trust": class_trust,
            "policy": {
                "digital_gate": "class-specific util.ecg_digital_features checks",
                "min_target_prob": thresholds,
                "require_top1": bool(args.require_top1),
                "min_pass_per_class": int(args.min_pass_per_class),
                "max_pass_per_class": int(args.max_pass_per_class),
                "hardcoded_zero": hardcoded_zero,
            },
            "counts_total": dict(total_counts),
            "counts_passed": dict(pass_counts),
        }, f, indent=2)

    ref_ids = load_k_ref_ids(Path(args.cache_root), center_name, args.K, args.selection_seed)
    if not ref_ids:
        ref_ids = sorted({str(r.get("ref_record_id")) for r in per_sample if r.get("ref_record_id")})
    ref_meta_path = out_dir / "gated_samples.ref_meta.json"
    with open(ref_meta_path, "w") as f:
        json.dump({
            "center": center_name,
            "ref_record_ids": ref_ids,
            "policy": "Exclude full prompt-token K-ref selection from downstream quick eval when available.",
            "K": int(args.K),
            "selection_seed": int(args.selection_seed),
        }, f, indent=2)

    by_class = defaultdict(lambda: {"total": 0, "passed": 0})
    for row in per_sample:
        by_class[row["class"]]["total"] += 1
        if row["keep"]:
            by_class[row["class"]]["passed"] += 1
    report_path = out_dir / "gate_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "source_samples": str(samples_path),
            "gated_samples": str(gated_samples_path),
            "gated_latent_pool": str(gated_latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(ref_meta_path),
            "n_total": int(labels.shape[0]),
            "n_passed": int(len(passed_indices)),
            "max_pass_per_class": int(args.max_pass_per_class),
            "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
            "per_sample": per_sample,
        }, f, indent=2)

    print(f"[gate] source: {samples_path}")
    print(f"[gate] kept {len(passed_indices)} / {labels.shape[0]}")
    for cls in CLASS_NAMES_SUPER5:
        if total_counts.get(cls, 0) or pass_counts.get(cls, 0):
            print(f"[gate] {cls:<4} {pass_counts.get(cls, 0):>3}/{total_counts.get(cls, 0):<3} trust={class_trust[cls]}")
    print(f"[gate] wrote {gated_samples_path}")
    print(f"[gate] wrote {gated_latent_path}")
    print(f"[gate] wrote {trust_path}")
    print(f"[gate] wrote {ref_meta_path}")
    print(f"[gate] wrote {report_path}")


if __name__ == "__main__":
    main()
