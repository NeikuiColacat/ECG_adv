# R1+R2+R3 — Frozen IBE Ref-Source Ablation with Clean Control

**Question**: Does the pretrained frozen IBE implicitly encode center features via its `base_vector(ref)`? R1+R2 showed Δ_T signals, but had 56-90% patient overlap between refpool and eval set (potential leakage confound). **R3 uses hash-split + K=100 refs** to isolate center-level encoding from patient-level memorization.

## Three-Experiment Design

| Experiment | Ref pool size | Ref/eval overlap | Eval set | Purpose |
|---|---:|---:|---|---|
| **R1+R2** | 700~16K refs | 56-90% patient overlap | full eval | Initial H1 verification |
| **R3** | 100 refs (stratified) | **0%** (fold 1-5 refs, fold 6-10 eval) | eval_half only | Clean control + deployment-scale validation |

## Core Result — Δ_T Comparison R1+R2 vs R3

| Target | n | **R1+R2 Δ_T** (possible leakage) | **R3 Δ_T** (clean) | Diff R12−R3 |
|---|---:|---:|---:|---:|
| cpsc_2018_extra | 1296 | +1.96pp | **+0.77pp** | +1.19pp |
| chapman_shaoxing | 9709 | +0.78pp | **+1.15pp** | -0.38pp |
| cpsc_2018 | 5279 | +0.51pp | **-0.87pp** | +1.38pp |
| georgia | 9320 | +1.26pp | **-1.18pp** | +2.44pp |
| ningbo | 34470 | +0.20pp | **-0.16pp** | +0.36pp |

**Interpretation**:
- R1+R2 Δ_T is a mixed signal (center encoding + possible patient memorization)
- R3 Δ_T is pure center-encoding signal (zero patient overlap)
- If R3 Δ_T > 0 → H1 holds, frozen IBE truly encodes center features (not leakage)
- R12-R3 diff estimates upper-bound contribution of patient memorization

## R3 Verdict

**❌ H1 not clearly supported under clean conditions**: only 2/5 meet threshold.

- R3 Δ_cpsc_2018_extra = +0.77pp
- R3 Δ_chapman_shaoxing = +1.15pp
- R3 Δ_cpsc_2018 = -0.87pp
- R3 Δ_georgia = -1.18pp
- R3 Δ_ningbo = -0.16pp

## R3 Full Table (K=100 refs, eval_half only)

| Center | Baseline | R3_A_EXTRA | R3_B_EXTRA | R3_A_CHAP | R3_B_CHAP | R3_A_CPSC | R3_B_CPSC | R3_A_GEO | R3_B_GEO | R3_A_NIN | R3_B_NIN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.9575 | 0.9440 | 0.9505 | 0.9550 🎯 | 0.9434 🎯 | 0.9479 | 0.9425 | 0.9459 | 0.9541 | 0.9345 | 0.9383 |
| cpsc_2018 | 0.9166 | 0.9280 | 0.9308 | 0.9285 | 0.9237 | 0.9290 🎯 | 0.9377 🎯 | 0.9400 | 0.9372 | 0.9200 | 0.9184 |
| cpsc_2018_extra | 0.8538 | 0.8712 🎯 | 0.8635 🎯 | 0.8715 | 0.8741 | 0.8372 | 0.8743 | 0.8819 | 0.8702 | 0.8443 | 0.8156 |
| georgia | 0.9482 | 0.9367 | 0.9459 | 0.9485 | 0.9357 | 0.9401 | 0.9482 | 0.9363 🎯 | 0.9481 🎯 | 0.9319 | 0.9299 |
| ningbo | 0.9695 | 0.9588 | 0.9686 | 0.9696 | 0.9585 | 0.9642 | 0.9644 | 0.9607 | 0.9675 | 0.9539 🎯 | 0.9555 🎯 |
| ptb | 0.8543 | 0.9509 | 0.9062 | 0.9198 | 0.9224 | 0.8598 | 0.9042 | 0.9270 | 0.9353 | 0.9334 | 0.9056 |
| st_petersburg_incart | 0.9218 | 0.5800 | 0.7800 | 0.8000 | 0.7200 | 0.8000 | 0.7000 | 0.7200 | 0.7000 | 0.6000 | 0.5200 |

*Note: R3 AUROCs are NOT directly comparable to baseline; baseline uses full eval, R3 uses eval_half only. Compare R3_A vs R3_B within each column, not against baseline.*

## R1+R2 Reference Table (full refpools + full eval)

| Target | Baseline | A_T | B_T | Δ_T |
|---|---:|---:|---:|---:|
| cpsc_2018_extra | 0.8538 | 0.8715 | 0.8519 | +1.96pp |
| chapman_shaoxing | 0.9575 | 0.9517 | 0.9439 | +0.78pp |
| cpsc_2018 | 0.9166 | 0.9361 | 0.9310 | +0.51pp |
| georgia | 0.9482 | 0.9514 | 0.9388 | +1.26pp |
| ningbo | 0.9695 | 0.9658 | 0.9638 | +0.20pp |

## PTBXL Test AUROC Sanity (all variants)

| Variant | PTBXL Test | ≥ 0.96? |
|---|---:|:---:|
| R1_A_EXTRA | 0.9670 | ✅ |
| R1_B_EXTRA | 0.9744 | ✅ |
| R1_A_CHAP | 0.9687 | ✅ |
| R1_B_CHAP | 0.9732 | ✅ |
| R1_A_CPSC | 0.9687 | ✅ |
| R1_B_CPSC | 0.9666 | ✅ |
| R2_A_GEO | 0.9744 | ✅ |
| R2_B_GEO | 0.9680 | ✅ |
| R2_A_NIN | 0.9736 | ✅ |
| R2_B_NIN | 0.9746 | ✅ |
| R3_A_EXTRA | 0.9687 | ✅ |
| R3_B_EXTRA | 0.9742 | ✅ |
| R3_A_CHAP | 0.9737 | ✅ |
| R3_B_CHAP | 0.9688 | ✅ |
| R3_A_CPSC | 0.9719 | ✅ |
| R3_B_CPSC | 0.9744 | ✅ |
| R3_A_GEO | 0.9680 | ✅ |
| R3_B_GEO | 0.9733 | ✅ |
| R3_A_NIN | 0.9696 | ✅ |
| R3_B_NIN | 0.9658 | ✅ |

## Artifacts

- R1+R2 retrain dirs: `/root/autodl-tmp/center_aware_ibe/retrain/r{1,2}_{a,b}_*/`
- R3 retrain dirs: `/root/autodl-tmp/center_aware_ibe/retrain/r3_{a,b}_*/`
- Refpools: `/root/autodl-tmp/center_aware_ibe/datasets/refpools/`
- Runners: `/tmp/run_r{1,2,3}_ablation.sh`
- Aggregator: `/tmp/r123_aggregate.py`
- Baseline: `/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json`
