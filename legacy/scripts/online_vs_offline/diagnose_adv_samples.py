"""诊断 offline 生成的对抗样本：
  1. 置信度分布（target_prob）
  2. sanity_check 生理合规指标（NaN/HR/Einthoven/flatline/saturation）
  3. plot_comparison 多张 PNG（orig vs augmix-adv，可视化对比）

跑 ~30 个样本 / 3 cells（chapman×LBBB 弱, cpsc×RBBB 最弱, georgia×NSR 中等）。
~5-8 min 完成。
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.ecg_viz import sanity_check, plot_comparison  # noqa: E402
from adversarial.adv_generate import BoundaryAdvDiffGenerator  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, DEFAULT_TIERM_CKPT,
)
from adversarial.tierM_labels import (  # noqa: E402
    TIER_M_TEXT_PROMPT, get_target_indices_tierM,
)
from methods.augmix.latent_viz.latent_augmix import latent_augmix_on_signal  # noqa: E402
from scripts.crosscenter_tierM.online_adv_train_tierM import (  # noqa: E402
    TIERM_GEN_HYPERPARAMS_DEFAULT, encode_refs_per_center,
)
from scripts.crosscenter_tierM.refs_per_center import (  # noqa: E402
    sample_refs_per_center, DEFAULT_PN2021_DIR,
)


DEVICE = "cuda"
OUT_DIR = Path("/root/ECG_adv_Gen/outputs/online_vs_offline/diagnose")
SEED = 42
ACCEPT_PROB = [0.5, 0.6]
# Three cells to sample from: largest/weakest targets
TARGETS = [
    ("chapman_shaoxing", "LBBB", 10),  # baseline AUROC 0.916, moderate difficulty
    ("cpsc_2018",        "RBBB", 10),  # AUROC 0.842, weakest cell
    ("georgia",          "NSR",  10),  # AUROC 0.878, large population
]
N_PLOT_PER_CELL = 2  # save 2 plot_comparison PNGs per cell


def _to_ct_canonical(ref_signal_ct_1000_canonical: np.ndarray, L_out: int = 250) -> np.ndarray:
    """Center crop (12, 1000) → (12, 250)."""
    T = ref_signal_ct_1000_canonical.shape[-1]
    start = (T - L_out) // 2
    return ref_signal_ct_1000_canonical[:, start:start + L_out]


def _aug_crop_250(aug_latent_ct_1000: np.ndarray, L_out: int = 250) -> np.ndarray:
    T = aug_latent_ct_1000.shape[-1]
    start = (T - L_out) // 2
    return aug_latent_ct_1000[:, start:start + L_out]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("[setup] ECGTwin + Tier-M baseline victim ...")
    ecgtwin = ECGTwinWrapper(device=DEVICE, load_encoder=True, load_text_model=True)
    victim = EfficientNetVictimTierM(
        weight_path=DEFAULT_TIERM_CKPT, device=DEVICE, ecgtwin_wrapper=ecgtwin,
    )
    victim.model.eval()

    print("[setup] Loading refs (cache hit expected) ...")
    refs = sample_refs_per_center(
        k_per_center=32,
        centers=[t[0] for t in TARGETS],
        data_dir=DEFAULT_PN2021_DIR,
        cache_path="/root/autodl-tmp/crosscenter_tierM_online/refs_cache_k32.npz",
        seed=SEED,
    )
    ref_items = encode_refs_per_center(refs, ecgtwin, DEVICE)

    # Generator with offline gen hyperparams
    gen_hp = dict(TIERM_GEN_HYPERPARAMS_DEFAULT)
    gen_hp["acceptance_range"] = ACCEPT_PROB
    gen_hp["batch_size"] = 4
    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, device=DEVICE, hyperparams=gen_hp,
    )

    all_records = []
    z_hist = {}

    for center, cls, target_count in TARGETS:
        print(f"\n{'=' * 60}")
        print(f"[gen] {center} × {cls}  target={target_count}")
        print('=' * 60)
        prompt = TIER_M_TEXT_PROMPT[cls]
        tgt_idx = get_target_indices_tierM(cls)[0]
        refs_center = ref_items[center]

        collected = 0
        attempts = 0
        t0 = time.time()
        while collected < target_count and attempts < 50:
            attempts += 1
            ref_item = refs_center[np.random.randint(0, len(refs_center))]
            conditions = ecgtwin.prepare_conditions(
                ref_latent=ref_item["data"],
                ref_label=ref_item["label"],
                batch_size=gen_hp["batch_size"],
                target_text=prompt,
            )
            conditions["ref_latent_raw"] = ref_item["data"]
            key = (center, cls)
            hist_lat = z_hist.get(key)

            batch_result = generator.generate_batch(
                conditions=conditions,
                target_indices=get_target_indices_tierM(cls),
                batch_size=gen_hp["batch_size"],
                historical_latent=hist_lat,
                clean_latent=ref_item["data"],
                best_of_k=5 if hist_lat is not None else 0,
            )
            n_new = batch_result["ecg"].shape[0]
            if n_new == 0:
                continue
            z_hist[key] = batch_result["latents"][-1].cpu()

            for i in range(n_new):
                if collected >= target_count:
                    break
                prob_t = float(batch_result["probs"][i][tgt_idx])
                z_adv = batch_result["latents"][i].unsqueeze(0)   # (1, 4, 128)

                # Run augmix injection to match offline training's actual input
                aug_res = latent_augmix_on_signal(
                    wrapper=ecgtwin,
                    signal_ct_100hz=ref_item["ref_signal_ct_1000_canonical"],
                    severity=5, width=3, depth=-1, alpha=1.0,
                    device=DEVICE,
                    inject_latents=[z_adv],
                )
                aug_ct_1000 = aug_res.augmix_latent_ct                      # (12, 1000)
                aug_ct_250 = _aug_crop_250(aug_ct_1000, 250)                # (12, 250) actual training input
                orig_ct_1000 = ref_item["ref_signal_ct_1000_canonical"]     # (12, 1000)

                # Sanity check on (12, 250) actual training input
                r_aug_250 = sanity_check(aug_ct_250, sample_rate=100, lead_order="ptbxl")
                # Sanity check on (12, 1000) full augmix for HR estimation (needs >=3s)
                r_aug_1000 = sanity_check(aug_ct_1000, sample_rate=100, lead_order="ptbxl")
                r_orig = sanity_check(orig_ct_1000, sample_rate=100, lead_order="ptbxl")

                rec = {
                    "center": center, "class": cls, "target_idx": tgt_idx,
                    "prob_target": prob_t,
                    "probs_6": batch_result["probs"][i].tolist(),
                    "sanity_orig_1000": r_orig,
                    "sanity_aug_1000": r_aug_1000,
                    "sanity_aug_250": r_aug_250,
                    "record_id": ref_item["record_id"],
                }
                all_records.append(rec)

                # Plot first N_PLOT_PER_CELL per cell
                if collected < N_PLOT_PER_CELL:
                    png_path = OUT_DIR / f"cmp_{center}_{cls}_{collected:02d}.png"
                    plot_comparison(
                        [orig_ct_1000, aug_ct_1000],
                        labels=["orig (ref)", f"augmix-adv prob={prob_t:.3f}"],
                        sample_rate=100,
                        save_path=png_path,
                        mode="overlay",
                        lead_order="ptbxl",
                    )
                collected += 1

        elapsed = time.time() - t0
        print(f"  → {collected}/{target_count} collected, {attempts} attempts, {elapsed:.0f}s")

    # ───────── Aggregate + print report ─────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    N = len(all_records)
    probs_t = np.array([r["prob_target"] for r in all_records])
    print(f"\nN_samples = {N}")
    print(f"target_prob:  min={probs_t.min():.3f}  max={probs_t.max():.3f}  "
          f"mean={probs_t.mean():.3f}  median={np.median(probs_t):.3f}")
    # Histogram bins on [0.5, 0.6]
    bins = np.arange(0.50, 0.61, 0.01)
    hist, _ = np.histogram(probs_t, bins=bins)
    print("  histogram on [0.50, 0.60] step 0.01:")
    for i, h in enumerate(hist):
        print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}) : {'#' * h} ({h})")

    # Sanity aggregate (on 1000-length augmix, which can estimate HR)
    def _agg(reports, tag):
        nan_n = sum(r["has_nan"] or r["has_inf"] for r in reports)
        hrs = [r["hr_estimate_bpm"] for r in reports if r["hr_estimate_bpm"] is not None]
        hr_in_range = sum(1 for h in hrs if 30 <= h <= 200)
        ei_res = np.array([r["einthoven_residual"] for r in reports])
        avr_res = np.array([r["avR_residual"] for r in reports])
        flat_n = sum(len(r["flatline_leads"]) > 0 for r in reports)
        sat_n = sum(len(r["saturated_leads"]) > 0 for r in reports)
        warn_n = sum(len(r["warnings"]) for r in reports)
        print(f"\n  [{tag}]  n={len(reports)}")
        print(f"    NaN/Inf           : {nan_n}/{len(reports)}")
        print(f"    HR ∈ [30,200]     : {hr_in_range}/{len(reports)}  (missing HR: {len(reports)-len(hrs)})")
        print(f"    HR values (first 10): {[round(h, 1) for h in hrs[:10]]}")
        print(f"    Einthoven residual: mean={ei_res.mean():.3f}  "
              f"p50={np.percentile(ei_res, 50):.3f}  p95={np.percentile(ei_res, 95):.3f}")
        print(f"    aVR residual      : mean={avr_res.mean():.3f}  "
              f"p95={np.percentile(avr_res, 95):.3f}")
        print(f"    Flatline leads    : {flat_n}/{len(reports)}")
        print(f"    Saturated leads   : {sat_n}/{len(reports)}")
        print(f"    Total warnings    : {warn_n}")

    _agg([r["sanity_orig_1000"] for r in all_records], "ORIG ref signals")
    _agg([r["sanity_aug_1000"] for r in all_records], "AUGMIX-ADV (12,1000)")
    _agg([r["sanity_aug_250"] for r in all_records], "AUGMIX-ADV (12,250) training input")

    # Save full json
    json_path = OUT_DIR.parent / "diagnose_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(
            {"config": {"accept_prob": ACCEPT_PROB, "seed": SEED, "targets": TARGETS},
             "n_samples": N,
             "target_prob_stats": {
                 "min": float(probs_t.min()), "max": float(probs_t.max()),
                 "mean": float(probs_t.mean()), "median": float(np.median(probs_t)),
                 "values": probs_t.tolist(),
             },
             "records": [
                 {"center": r["center"], "class": r["class"],
                  "prob_target": r["prob_target"], "probs_6": r["probs_6"],
                  "hr_orig":  r["sanity_orig_1000"].get("hr_estimate_bpm"),
                  "hr_aug":   r["sanity_aug_1000"].get("hr_estimate_bpm"),
                  "einth_res_orig": r["sanity_orig_1000"]["einthoven_residual"],
                  "einth_res_aug":  r["sanity_aug_1000"]["einthoven_residual"],
                  "warnings_aug":   r["sanity_aug_1000"]["warnings"],
                  } for r in all_records],
             },
            f, indent=2, default=str,
        )
    print(f"\n[diagnose] aggregate json → {json_path}")
    print(f"[diagnose] plots ({N_PLOT_PER_CELL*len(TARGETS)} PNGs) → {OUT_DIR}")


if __name__ == "__main__":
    main()
