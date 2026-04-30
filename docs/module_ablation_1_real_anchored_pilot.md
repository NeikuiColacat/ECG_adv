# Synth-anchored Super5 pilot iter 4 — Plan Rev 13.2 results

Generation scope: NORM/MI/STTC (HYP/CD digital-GT 0/3 fail).
Decision gate: ≥2/3 cells pass `macro AUROC Δ > +0.30pp` AND `macro AUPRC Δ > +0.50pp` AND `PTBXL fold10 Δ ≥ -0.50pp`.

## Cell summary

| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |
|---|---|---|---|---|---|
| extra | cpsc_2018_extra | 39 | True | 18 | 0.8417 |
| nin | ningbo | 51 | True | 30 | 0.8457 |
| geo | georgia | 72 | True | 51 | 0.8461 |

## Table A — PN2021 per-center macro AUROC + AUPRC delta vs baseline

### AUROC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.8835 | +0.33pp | +0.32pp | +0.38pp |
| cpsc_2018 | 0.8052 | -0.65pp | -0.88pp | -1.06pp |
| cpsc_2018_extra | 0.8041 | -0.08pp | +0.15pp | +0.00pp |
| georgia | 0.8168 | +0.26pp | +0.32pp | +0.29pp |
| ningbo | 0.8820 | +0.33pp | +0.32pp | +0.40pp |
| ptb | 0.9009 | -0.07pp | +0.00pp | +0.02pp |
| st_petersburg_incart | 0.7785 | -0.71pp | -0.56pp | -0.57pp |
| **avg** | 0.8387 | -0.08pp | -0.05pp | -0.08pp |

### AUPRC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.5468 | +0.59pp | +0.34pp | +0.33pp |
| cpsc_2018 | 0.5633 | -0.16pp | -0.26pp | -0.38pp |
| cpsc_2018_extra | 0.6240 | -0.05pp | -0.00pp | -0.21pp |
| georgia | 0.6452 | +0.42pp | +0.45pp | +0.36pp |
| ningbo | 0.5899 | +0.55pp | +0.40pp | +0.59pp |
| ptb | 0.5950 | +0.08pp | +0.25pp | -0.04pp |
| st_petersburg_incart | 0.5575 | -0.28pp | -0.39pp | -0.46pp |
| **avg** | 0.5888 | +0.16pp | +0.11pp | +0.03pp |

## Table B — Per-cell PN2021 per-class AUROC delta

### Cell `extra` (training center = cpsc_2018_extra)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.15pp | +0.65pp | +0.27pp | +0.08pp | +0.48pp |
| cpsc_2018 | +0.18pp |   n/a |   n/a | -0.06pp | -2.06pp |
| cpsc_2018_extra | +0.15pp | -0.14pp | -0.13pp |   n/a | -0.20pp |
| georgia | +0.39pp | +0.42pp |   n/a | +0.07pp | +0.14pp |
| ningbo | +0.37pp | +0.39pp | +0.34pp | +0.14pp | +0.40pp |
| ptb | -0.08pp | -0.01pp | -0.27pp | +0.10pp |   n/a |
| st_petersburg_incart | -0.31pp | -0.78pp | +0.56pp |   n/a | -2.31pp |

### Cell `nin` (training center = ningbo)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.13pp | +0.67pp | +0.10pp | +0.10pp | +0.60pp |
| cpsc_2018 | +0.22pp |   n/a |   n/a | -0.10pp | -2.75pp |
| cpsc_2018_extra | +0.40pp | -0.19pp | +0.19pp |   n/a | +0.20pp |
| georgia | +0.47pp | +0.48pp |   n/a | +0.08pp | +0.24pp |
| ningbo | +0.57pp | +0.40pp | -0.05pp | +0.12pp | +0.57pp |
| ptb | +0.03pp | -0.11pp | -0.02pp | +0.11pp |   n/a |
| st_petersburg_incart | -0.31pp | +0.16pp | +0.51pp |   n/a | -2.60pp |

### Cell `geo` (training center = georgia)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.11pp | +0.80pp | +0.13pp | +0.10pp | +0.74pp |
| cpsc_2018 | +0.21pp |   n/a |   n/a | -0.13pp | -3.27pp |
| cpsc_2018_extra | +0.38pp | -0.16pp | +0.18pp |   n/a | -0.40pp |
| georgia | +0.46pp | +0.41pp |   n/a | +0.02pp | +0.29pp |
| ningbo | +0.47pp | +0.46pp | +0.23pp | +0.07pp | +0.76pp |
| ptb | -0.03pp | -0.02pp | -0.01pp | +0.13pp |   n/a |
| st_petersburg_incart | -0.23pp | -0.62pp | +0.45pp |   n/a | -1.88pp |

## Table C — MIMIC zero-shot delta

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.7857 | +0.20pp | +0.36pp | +0.33pp |
| macro_auprc | 0.6939 | +0.10pp | +0.35pp | +0.32pp |

## Table D — PTBXL fold10 in-domain delta (gate: ≥ -0.50pp)

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.9064 | -0.03pp | -0.01pp | -0.00pp |
| macro_auprc | 0.7754 | +0.04pp | +0.09pp | +0.11pp |

## Table E — PGD gates trace (last 5 logged epochs / cell)

### `extra`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 35 | 0.9949 | 0.2261 | True | 2048 | 0.5705 | 0.5295 |
| 36 | 0.9848 | 0.226 | True | 2048 | 0.5686 | 0.5241 |
| 37 | 0.9949 | 0.2247 | True | 2048 | 0.5713 | 0.5263 |
| 38 | 0.9848 | 0.2232 | True | 2048 | 0.5731 | 0.5229 |
| 39 | 0.9949 | 0.2252 | True | 2048 | 0.5717 | 0.5194 |

### `nin`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 47 | 0.984 | 0.2205 | True | 2048 | 0.5704 | 0.5334 |
| 48 | 1.0 | 0.2231 | True | 2048 | 0.5698 | 0.5285 |
| 49 | 1.0 | 0.2226 | True | 2048 | 0.5689 | 0.526 |
| 50 | 1.0 | 0.2176 | True | 2048 | 0.5728 | 0.5312 |
| 51 | 1.0 | 0.222 | True | 2048 | 0.5727 | 0.5284 |

### `geo`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 68 | 0.9888 | 0.2576 | True | 2048 | 0.5719 | 0.5221 |
| 69 | 1.0 | 0.2567 | True | 2048 | 0.5708 | 0.526 |
| 70 | 1.0 | 0.2664 | True | 2048 | 0.5698 | 0.5387 |
| 71 | 1.0 | 0.2666 | True | 2048 | 0.5726 | 0.5267 |
| 72 | 1.0 | 0.2549 | True | 2048 | 0.5737 | 0.5195 |

## Decision gate

- **extra**: AUROC Δ=-0.08pp | AUPRC Δ=+0.16pp | PTBXL Δ=-0.03pp → ❌ FAIL
- **nin**: AUROC Δ=-0.05pp | AUPRC Δ=+0.11pp | PTBXL Δ=-0.01pp → ❌ FAIL
- **geo**: AUROC Δ=-0.08pp | AUPRC Δ=+0.03pp | PTBXL Δ=-0.00pp → ❌ FAIL

**Summary**: 0/3 cells pass.
→ STOP-AND-ANALYZE: scale Stage 1 pool 300→1200, rerun once. If still fail, lock as ECGTwin vocab/manifold limitation.
