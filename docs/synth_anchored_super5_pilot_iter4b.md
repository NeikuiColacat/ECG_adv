# Synth-anchored Super5 pilot iter 4 — Plan Rev 13.2 results

Generation scope: NORM/MI/STTC (HYP/CD digital-GT 0/3 fail).
Decision gate: ≥2/3 cells pass `macro AUROC Δ > +0.30pp` AND `macro AUPRC Δ > +0.50pp` AND `PTBXL fold10 Δ ≥ -0.50pp`.

## Cell summary

| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |
|---|---|---|---|---|---|
| extra | cpsc_2018_extra | 48 | True | 27 | 0.8441 |
| nin | ningbo | 30 | True | 9 | 0.8448 |
| geo | georgia | 30 | True | 9 | 0.8457 |

## Table A — PN2021 per-center macro AUROC + AUPRC delta vs baseline

### AUROC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.8835 | +0.50pp | +0.35pp | +0.46pp |
| cpsc_2018 | 0.8052 | -1.72pp | -0.64pp | -0.79pp |
| cpsc_2018_extra | 0.8028 | -0.30pp | +0.18pp | +0.07pp |
| georgia | 0.8163 | +0.38pp | +0.38pp | +0.29pp |
| ningbo | 0.8819 | +0.49pp | +0.29pp | +0.49pp |
| ptb | 0.9009 | +0.04pp | -0.02pp | -0.09pp |
| st_petersburg_incart | 0.7785 | -0.75pp | -0.79pp | -0.72pp |
| **avg** | 0.8384 | -0.19pp | -0.04pp | -0.04pp |

### AUPRC
| center | baseline | extra | nin | geo |
|---|---|---|---|---|
| chapman_shaoxing | 0.5468 | +0.74pp | +0.82pp | +0.96pp |
| cpsc_2018 | 0.5633 | -0.68pp | -0.11pp | -0.19pp |
| cpsc_2018_extra | 0.6237 | -0.29pp | -0.10pp | -0.31pp |
| georgia | 0.6449 | +0.54pp | +0.57pp | +0.42pp |
| ningbo | 0.5898 | +0.84pp | +0.53pp | +0.83pp |
| ptb | 0.5950 | -0.11pp | +0.97pp | -0.03pp |
| st_petersburg_incart | 0.5575 | -0.45pp | -0.84pp | -0.52pp |
| **avg** | 0.5887 | +0.08pp | +0.26pp | +0.16pp |

## Table B — Per-cell PN2021 per-class AUROC delta

### Cell `extra` (training center = cpsc_2018_extra)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.17pp | +0.82pp | +0.34pp | +0.05pp | +1.11pp |
| cpsc_2018 | +0.17pp |   n/a |   n/a | -0.17pp | -5.15pp |
| cpsc_2018_extra | +0.17pp | -0.11pp | +0.04pp |   n/a | -1.29pp |
| georgia | +0.57pp | +0.56pp |   n/a | +0.05pp | +0.35pp |
| ningbo | +0.36pp | +0.51pp | +0.35pp | +0.06pp | +1.16pp |
| ptb | +0.20pp | -0.02pp | -0.07pp | +0.06pp |   n/a |
| st_petersburg_incart | -0.16pp | -1.41pp | +0.45pp |   n/a | -1.88pp |

### Cell `nin` (training center = ningbo)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.25pp | +0.63pp | +0.24pp | +0.16pp | +0.47pp |
| cpsc_2018 | +0.07pp |   n/a |   n/a | -0.01pp | -1.99pp |
| cpsc_2018_extra | +0.91pp | -0.29pp | -0.12pp |   n/a | +0.23pp |
| georgia | +0.60pp | +0.52pp |   n/a | +0.17pp | +0.24pp |
| ningbo | +0.33pp | +0.38pp | +0.17pp | +0.15pp | +0.42pp |
| ptb | -0.09pp | +0.03pp | -0.14pp | +0.12pp |   n/a |
| st_petersburg_incart | -0.47pp | -0.47pp | +0.23pp |   n/a | -2.45pp |

### Cell `geo` (training center = georgia)

| center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.29pp | +0.82pp | +0.34pp | +0.19pp | +0.64pp |
| cpsc_2018 | +0.15pp |   n/a |   n/a | +0.01pp | -2.53pp |
| cpsc_2018_extra | +0.88pp | -0.26pp | -0.25pp |   n/a | -0.11pp |
| georgia | +0.53pp | +0.32pp |   n/a | +0.11pp | +0.22pp |
| ningbo | +0.57pp | +0.57pp | +0.45pp | +0.25pp | +0.60pp |
| ptb | -0.23pp | +0.15pp | -0.31pp | +0.04pp |   n/a |
| st_petersburg_incart | -0.78pp | -0.16pp | +0.56pp |   n/a | -2.53pp |

## Table C — MIMIC zero-shot delta

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.7857 | +0.30pp | +0.27pp | +0.29pp |
| macro_auprc | 0.6939 | +0.34pp | +0.26pp | +0.21pp |

## Table D — PTBXL fold10 in-domain delta (gate: ≥ -0.50pp)

| metric | baseline | extra | nin | geo |
|---|---|---|---|---|
| macro_auroc | 0.9064 | -0.04pp | -0.02pp | -0.03pp |
| macro_auprc | 0.7754 | -0.02pp | +0.11pp | +0.06pp |

## Table E — PGD gates trace (last 5 logged epochs / cell)

### `extra`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 44 | 1.0 | 0.1132 | True | 2048 | 0.5746 | 0.5333 |
| 45 | 1.0 | 0.1134 | True | 2048 | 0.577 | 0.5379 |
| 46 | 1.0 | 0.113 | True | 2048 | 0.5736 | 0.5317 |
| 47 | 1.0 | 0.1145 | True | 2048 | 0.5774 | 0.5335 |
| 48 | 0.9983 | 0.1143 | True | 2048 | 0.5799 | 0.5384 |

### `nin`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 26 | 0.9983 | 0.106 | True | 2048 | 0.573 | 0.5284 |
| 27 | 1.0 | 0.1043 | True | 2048 | 0.57 | 0.5379 |
| 28 | 1.0 | 0.1043 | True | 2048 | 0.57 | 0.5309 |
| 29 | 0.9983 | 0.1045 | True | 2048 | 0.5711 | 0.5256 |
| 30 | 1.0 | 0.1044 | True | 2048 | 0.5696 | 0.5305 |

### `geo`

| ep | asr | einthoven_p95 | medical_pass | buf | val | train |
|---|---|---|---|---|---|---|
| 26 | 0.9983 | 0.099 | True | 2048 | 0.5717 | 0.5222 |
| 27 | 1.0 | 0.0983 | True | 2048 | 0.571 | 0.5253 |
| 28 | 1.0 | 0.0969 | True | 2048 | 0.573 | 0.532 |
| 29 | 1.0 | 0.0974 | True | 2048 | 0.5717 | 0.528 |
| 30 | 1.0 | 0.0974 | True | 2048 | 0.5692 | 0.5317 |

## Decision gate

- **extra**: AUROC Δ=-0.19pp | AUPRC Δ=+0.08pp | PTBXL Δ=-0.04pp → ❌ FAIL
- **nin**: AUROC Δ=-0.04pp | AUPRC Δ=+0.26pp | PTBXL Δ=-0.02pp → ❌ FAIL
- **geo**: AUROC Δ=-0.04pp | AUPRC Δ=+0.16pp | PTBXL Δ=-0.03pp → ❌ FAIL

**Summary**: 0/3 cells pass.
→ STOP-AND-ANALYZE: scale Stage 1 pool 300→1200, rerun once. If still fail, lock as ECGTwin vocab/manifold limitation.
