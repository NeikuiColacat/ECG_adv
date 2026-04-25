"""AugMix 快速静态 sanity：10 条 PTBXL fold-10 样本过 augmix(s=5, w=3)，
收集 sanity_check 指标 + 存 5 张 plot_comparison PNG。

判定（打印到终端）：
  - NaN/Inf 率 = 0
  - HR ∈ [30, 200] 的通过率 ≥ 8/10
  - einthoven_residual_p95 < 1.0
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.augmix.augmix import augmix  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import get_ptbxl_26_labels  # noqa: E402
from util.ecg_viz import plot_comparison, sanity_check  # noqa: E402


PTBXL_CACHE = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
PTBXL_CSV = "/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv"
OUT_DIR = Path("/root/ECG_adv_Gen/outputs/augmix_validation/quick")
SEED = 42
SEVERITY = 5
WIDTH = 3
N_SAMPLES = 10
N_PLOT = 5


def _summarize(reports, tag):
    n = len(reports)
    nan_n = sum(r["has_nan"] or r["has_inf"] for r in reports)
    hrs = [r["hr_estimate_bpm"] for r in reports if r["hr_estimate_bpm"] is not None]
    hr_in_range = sum(1 for h in hrs if 30 <= h <= 200)
    ei_res = np.array([r["einthoven_residual"] for r in reports])
    avr_res = np.array([r["avR_residual"] for r in reports])
    flat_n = sum(len(r["flatline_leads"]) > 0 for r in reports)
    sat_n = sum(len(r["saturated_leads"]) > 0 for r in reports)
    warn_n = sum(len(r["warnings"]) for r in reports)
    return {
        "tag": tag,
        "n": n,
        "nan_or_inf": nan_n,
        "hr_in_range": hr_in_range,
        "hr_missing": n - len(hrs),
        "hr_values": [round(h, 1) for h in hrs],
        "einthoven_residual_mean": float(ei_res.mean()),
        "einthoven_residual_p95": float(np.percentile(ei_res, 95)),
        "avR_residual_mean": float(avr_res.mean()),
        "avR_residual_p95": float(np.percentile(avr_res, 95)),
        "flatline_any_count": flat_n,
        "saturated_any_count": sat_n,
        "total_warnings": warn_n,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[sanity] loading PTBXL fold 10 labels + preprocessed cache ...")
    test_idx, _, _ = get_ptbxl_26_labels(PTBXL_CSV, folds=[10])
    cache = np.load(PTBXL_CACHE, mmap_mode="r")   # (N, 1000, 12) float32
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(test_idx), size=N_SAMPLES, replace=False)
    rec_idx = [int(test_idx[i]) for i in pick]
    print(f"[sanity] sampled {N_SAMPLES} test records: {rec_idx}")

    orig_reports = []
    aug_reports = []

    for rank, rec_i in enumerate(rec_idx):
        sig_tc = np.asarray(cache[rec_i], dtype=np.float32)       # (1000, 12)
        sig_ct = sig_tc.T                                         # (12, 1000)

        t_ct = torch.from_numpy(np.ascontiguousarray(sig_ct)).float()
        np.random.seed(SEED + rank)                               # op-level RNG (severity picks)
        aug_ct = augmix(t_ct, severity=SEVERITY, width=WIDTH).detach().cpu().numpy()

        r_orig = sanity_check(sig_ct, sample_rate=100, lead_order="ptbxl")
        r_aug = sanity_check(aug_ct, sample_rate=100, lead_order="ptbxl")
        orig_reports.append(r_orig)
        aug_reports.append(r_aug)

        if rank < N_PLOT:
            plot_comparison(
                [sig_ct, aug_ct],
                labels=["orig", f"augmix s{SEVERITY} w{WIDTH}"],
                sample_rate=100,
                save_path=OUT_DIR / f"cmp_{rank:02d}_idx{rec_i}.png",
                mode="overlay",
                lead_order="ptbxl",
            )

    summary = {
        "config": {"severity": SEVERITY, "width": WIDTH,
                   "n_samples": N_SAMPLES, "seed": SEED},
        "record_idx": rec_idx,
        "original": _summarize(orig_reports, "original"),
        "augmix": _summarize(aug_reports, "augmix"),
    }

    json_path = OUT_DIR.parent / "sanity_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sanity] saved json → {json_path}")
    print(f"[sanity] saved {N_PLOT} comparison PNGs → {OUT_DIR}")

    print("\n==================== SANITY SUMMARY ====================")
    for key in ("original", "augmix"):
        s = summary[key]
        print(f"\n[{s['tag']}] N={s['n']}  NaN/Inf={s['nan_or_inf']}  "
              f"HR_in_range={s['hr_in_range']}/{s['n']} (missing={s['hr_missing']})  "
              f"flatline_any={s['flatline_any_count']}  saturated_any={s['saturated_any_count']}  "
              f"warnings_total={s['total_warnings']}")
        print(f"  Einthoven residual: mean={s['einthoven_residual_mean']:.3f}  "
              f"p95={s['einthoven_residual_p95']:.3f}")
        print(f"  aVR residual:       mean={s['avR_residual_mean']:.3f}  "
              f"p95={s['avR_residual_p95']:.3f}")
        print(f"  HR values: {s['hr_values']}")

    s_a = summary["augmix"]
    pass_nan = s_a["nan_or_inf"] == 0
    pass_hr = s_a["hr_in_range"] >= 8
    pass_ei = s_a["einthoven_residual_p95"] < 1.0
    pass_sat = s_a["saturated_any_count"] == 0 and s_a["flatline_any_count"] == 0
    print("\n==================== VERDICT ====================")
    print(f"  [{'PASS' if pass_nan else 'FAIL'}] NaN/Inf = 0")
    print(f"  [{'PASS' if pass_hr else 'FAIL'}] HR in [30,200] ≥ 8/{N_SAMPLES}")
    print(f"  [{'PASS' if pass_ei else 'FAIL'}] Einthoven residual p95 < 1.0")
    print(f"  [{'PASS' if pass_sat else 'FAIL'}] No flatline/saturated lead")
    overall = pass_nan and pass_hr and pass_ei and pass_sat
    print(f"\n  OVERALL: {'PASS — safe to proceed to training' if overall else 'FAIL — investigate before training'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
