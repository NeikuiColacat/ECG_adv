"""Latent-space AugMix batched smoke test.

Loads ECGTwin VAE wrapper, applies latent_augmix_batch to 16 PTBXL samples,
checks output shape / NaN / finite, saves 3 PNG comparisons and an aggregate
sanity_check report.

Pass criteria:
  - Output shape (16, 12, 1000) float32
  - no NaN/Inf
  - HR ∈ [30, 200] on ≥ 10/16 augmented samples
  - Einthoven residual p95 < 2.0 (looser than time-domain: VAE reconstruct may add bias)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.augmix.latent_viz.latent_augmix import latent_augmix_batch  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import get_ptbxl_26_labels  # noqa: E402
from util.ecg_viz import plot_comparison, sanity_check  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


PTBXL_CACHE = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
PTBXL_CSV = "/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv"
OUT_DIR = Path("/root/ECG_adv_Gen/outputs/augmix_validation/latent_smoke")
SEED = 42
N = 16
N_PLOT = 3


def _summarize(reports, tag):
    n = len(reports)
    nan_n = sum(r["has_nan"] or r["has_inf"] for r in reports)
    hrs = [r["hr_estimate_bpm"] for r in reports if r["hr_estimate_bpm"] is not None]
    hr_in = sum(1 for h in hrs if 30 <= h <= 200)
    ei = np.array([r["einthoven_residual"] for r in reports])
    return {
        "tag": tag, "n": n, "nan_or_inf": nan_n,
        "hr_in_range": hr_in, "hr_missing": n - len(hrs),
        "hr_values": [round(h, 1) for h in hrs],
        "einthoven_p50": float(np.percentile(ei, 50)),
        "einthoven_p95": float(np.percentile(ei, 95)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[smoke] Loading ECGTwinWrapper (VAE encoder + decoder)...")
    wrapper = ECGTwinWrapper(device="cuda:0", load_encoder=True, load_text_model=False)
    print("[smoke] Wrapper loaded.\n")

    print("[smoke] Loading PTBXL fold-10 labels + preprocessed cache...")
    test_idx, _, _ = get_ptbxl_26_labels(PTBXL_CSV, folds=[10])
    cache = np.load(PTBXL_CACHE, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(test_idx), size=N, replace=False)
    rec_idx = [int(test_idx[i]) for i in pick]
    sig_tc = np.asarray(cache[rec_idx], dtype=np.float32)                   # (N, 1000, 12)
    sig_bct = np.transpose(sig_tc, (0, 2, 1))                               # (N, 12, 1000)
    sig_bct_t = torch.from_numpy(np.ascontiguousarray(sig_bct)).to("cuda:0")
    print(f"[smoke] Input batch: {tuple(sig_bct_t.shape)} {sig_bct_t.dtype} on {sig_bct_t.device}\n")

    print("[smoke] Running latent_augmix_batch(severity=5, width=3)...")
    out_bct_t = latent_augmix_batch(
        wrapper, sig_bct_t,
        severity=5, width=3, depth=-1, alpha=1.0,
    )
    print(f"[smoke] Output: {tuple(out_bct_t.shape)} {out_bct_t.dtype}")

    assert out_bct_t.shape == (N, 12, 1000), f"bad shape: {out_bct_t.shape}"
    assert torch.isfinite(out_bct_t).all().item(), "output has NaN/Inf"
    out_np = out_bct_t.detach().cpu().numpy()

    # sanity per sample
    orig_reports, aug_reports = [], []
    for b in range(N):
        orig_reports.append(sanity_check(sig_bct[b], 100, lead_order="ptbxl"))
        aug_reports.append(sanity_check(out_np[b], 100, lead_order="ptbxl"))

    # save N_PLOT comparisons
    for b in range(N_PLOT):
        plot_comparison(
            [sig_bct[b], out_np[b]],
            labels=["orig", "latent_augmix s5 w3"],
            sample_rate=100,
            save_path=OUT_DIR / f"cmp_{b:02d}_idx{rec_idx[b]}.png",
            mode="overlay", lead_order="ptbxl",
        )

    summary = {
        "config": {"severity": 5, "width": 3, "n_samples": N, "seed": SEED},
        "record_idx": rec_idx,
        "input_shape": list(sig_bct_t.shape),
        "output_shape": list(out_bct_t.shape),
        "original": _summarize(orig_reports, "original"),
        "latent_augmix": _summarize(aug_reports, "latent_augmix"),
    }
    json_path = OUT_DIR.parent / "latent_smoke_report.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[smoke] report → {json_path}")
    print(f"[smoke] PNGs → {OUT_DIR}\n")
    print("==================== LATENT AUGMIX SMOKE ====================")
    for k in ("original", "latent_augmix"):
        s = summary[k]
        print(f"[{s['tag']}] NaN/Inf={s['nan_or_inf']}  "
              f"HR_in_range={s['hr_in_range']}/{s['n']}  "
              f"einthoven(p50={s['einthoven_p50']:.3f} p95={s['einthoven_p95']:.3f})")
        print(f"  HRs: {s['hr_values']}")

    s_a = summary["latent_augmix"]
    pass_nan = s_a["nan_or_inf"] == 0
    pass_hr = s_a["hr_in_range"] >= 10
    pass_ei = s_a["einthoven_p95"] < 2.0
    print("\n==================== VERDICT ====================")
    print(f"  [{'PASS' if pass_nan else 'FAIL'}] NaN/Inf = 0")
    print(f"  [{'PASS' if pass_hr else 'FAIL'}] HR ∈ [30, 200] ≥ 10/{N}")
    print(f"  [{'PASS' if pass_ei else 'FAIL'}] Einthoven residual p95 < 2.0")
    overall = pass_nan and pass_hr and pass_ei
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
