# Synth-anchored Super5 pilot iter 4 — Plan Rev 13.2 results

Generation scope: NORM/MI/STTC (HYP/CD digital-GT 0/3 fail).
Decision gate: ≥2/3 cells pass `macro AUROC Δ > +0.30pp` AND `macro AUPRC Δ > +0.50pp` AND `PTBXL fold10 Δ ≥ -0.50pp`.

## Cell summary

| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |
|---|---|---|---|---|---|
| extra | cpsc_2018_extra | 66 | True | 45 | 0.8432 |
| nin | ningbo | 42 | True | 21 | 0.8448 |
| geo | georgia | 42 | True | 21 | 0.8458 |

## Table A — PN2021 per-center macro AUROC + AUPRC delta vs baseline

### AUROC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.8835 | +0.49pp | +0.28pp | +0.40pp |
| cpsc_2018 | 0.8052 | -1.63pp | -0.83pp | -0.95pp |
| cpsc_2018_extra | 0.8041 | -0.20pp | +0.04pp | +0.01pp |
| georgia | 0.8168 | +0.28pp | +0.22pp | +0.24pp |
| ningbo | 0.8820 | +0.49pp | +0.25pp | +0.38pp |
| ptb | 0.9009 | +0.05pp | -0.03pp | -0.01pp |
| st_petersburg_incart | 0.7785 | -0.69pp | -0.75pp | -0.96pp |
| **avg** | 0.8387 | -0.17pp | -0.12pp | -0.13pp |

### AUPRC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.5468 | +0.67pp | +0.66pp | +0.83pp |
| cpsc_2018 | 0.5633 | -0.56pp | -0.27pp | -0.29pp |
| cpsc_2018_extra | 0.6240 | -0.13pp | -0.19pp | -0.20pp |
| georgia | 0.6452 | +0.38pp | +0.34pp | +0.30pp |
| ningbo | 0.5899 | +0.93pp | +0.43pp | +0.73pp |
| ptb | 0.5950 | -0.04pp | +0.73pp | +0.90pp |
| st_petersburg_incart | 0.5575 | -0.83pp | -0.36pp | -1.04pp |
| **avg** | 0.5888 | +0.06pp | +0.19pp | +0.18pp |

## Table B — Per-cell PN2021 per-class AUROC delta

### Cell `extra` (training center = cpsc_2018_extra)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.25pp | +0.90pp | +0.18pp | +0.09pp | +1.03pp |
| cpsc_2018 | +0.01pp |   n/a |   n/a | -0.09pp | -4.82pp |
| cpsc_2018_extra | +0.23pp | -0.08pp | +0.07pp |   n/a | -1.03pp |
| georgia | +0.54pp | +0.34pp |   n/a | -0.01pp | +0.25pp |
| ningbo | +0.51pp | +0.49pp | +0.35pp | +0.03pp | +1.08pp |
| ptb | -0.11pp | +0.01pp | +0.14pp | +0.18pp |   n/a |
| st_petersburg_incart | +0.16pp | -0.94pp | +0.34pp |   n/a | -2.31pp |

### Cell `nin` (training center = ningbo)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.12pp | +0.66pp | +0.21pp | -0.03pp | +0.44pp |
| cpsc_2018 | +0.07pp |   n/a |   n/a | -0.08pp | -2.49pp |
| cpsc_2018_extra | +0.25pp | -0.24pp | -0.04pp |   n/a | +0.20pp |
| georgia | +0.38pp | +0.39pp |   n/a | -0.04pp | +0.13pp |
| ningbo | +0.36pp | +0.41pp | +0.15pp | -0.09pp | +0.41pp |
| ptb | -0.10pp | +0.05pp | -0.22pp | +0.13pp |   n/a |
| st_petersburg_incart | -1.09pp | -0.16pp | +0.56pp |   n/a | -2.31pp |

### Cell `geo` (training center = georgia)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.15pp | +1.05pp | +0.19pp | -0.02pp | +0.64pp |
| cpsc_2018 | +0.14pp |   n/a |   n/a | -0.09pp | -2.91pp |
| cpsc_2018_extra | +0.32pp | -0.22pp | -0.07pp |   n/a | +0.01pp |
| georgia | +0.38pp | +0.44pp |   n/a | -0.07pp | +0.22pp |
| ningbo | +0.32pp | +0.62pp | +0.43pp | -0.07pp | +0.62pp |
| ptb | -0.14pp | +0.10pp | -0.19pp | +0.17pp |   n/a |
| st_petersburg_incart | -0.62pp | -0.94pp | +0.34pp |   n/a | -2.60pp |

## Table C — MIMIC zero-shot delta

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.7857 | +0.33pp | +0.23pp | +0.27pp |
| macro_auprc | 0.6939 | +0.49pp | +0.20pp | +0.25pp |

## Table D — PTBXL fold10 in-domain delta (gate: ≥ -0.50pp)

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.9064 | -0.03pp | -0.02pp | -0.00pp |
| macro_auprc | 0.7754 | +0.06pp | +0.09pp | +0.13pp |

## Table E — PGD gates trace (last 5 logged epochs / cell)

### `extra`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 62 | 1.0 | 0.0938 | True | 2048 | 0.5733 | 0.5256 |
| 63 | 1.0 | 0.0931 | True | 2048 | 0.5728 | 0.5251 |
| 64 | 1.0 | 0.094 | True | 2048 | 0.5767 | 0.5372 |
| 65 | 0.9967 | 0.0947 | True | 2048 | 0.5705 | 0.5195 |
| 66 | 1.0 | 0.0945 | True | 2048 | 0.574 | 0.5358 |

### `nin`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 38 | 1.0 | 0.1005 | True | 2048 | 0.5695 | 0.5283 |
| 39 | 1.0 | 0.102 | True | 2048 | 0.5705 | 0.521 |
| 40 | 0.9967 | 0.0994 | True | 2048 | 0.5708 | 0.5248 |
| 41 | 0.9967 | 0.0997 | True | 2048 | 0.5698 | 0.5271 |
| 42 | 0.9967 | 0.0994 | True | 2048 | 0.5732 | 0.5238 |

### `geo`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 38 | 1.0 | 0.1078 | True | 2048 | 0.5675 | 0.5352 |
| 39 | 1.0 | 0.1041 | True | 2048 | 0.5717 | 0.5228 |
| 40 | 1.0 | 0.1064 | True | 2048 | 0.5702 | 0.5303 |
| 41 | 1.0 | 0.105 | True | 2048 | 0.5708 | 0.5276 |
| 42 | 1.0 | 0.1041 | True | 2048 | 0.5698 | 0.5296 |

## Decision gate

- **extra**: AUROC Δ=-0.17pp | AUPRC Δ=+0.06pp | PTBXL Δ=-0.03pp → ❌ FAIL
- **nin**: AUROC Δ=-0.12pp | AUPRC Δ=+0.19pp | PTBXL Δ=-0.02pp → ❌ FAIL
- **geo**: AUROC Δ=-0.13pp | AUPRC Δ=+0.18pp | PTBXL Δ=-0.00pp → ❌ FAIL

**Summary**: 0/3 cells pass.
→ STOP-AND-ANALYZE: scale Stage 1 pool 300→1200, rerun once. If still fail, lock as ECGTwin vocab/manifold limitation.
