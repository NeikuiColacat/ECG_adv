# Synth-anchored Super5 pilot — 2026-04-26 08:10

Pilot of CenterToken-anchored VAE-latent PGD online AT against Super5 baseline. 2 cells × K=200, AugMix-off (Plan Rev 8).

## Decision gate

- Threshold: ≥1 cell with **macro AUROC Δ > +0.3pp** AND **macro AUPRC Δ > +0.5pp** (Plan Issue #34)
- Pass: **0/2**
- Decision: **STOP-AND-ANALYZE**

Failing / inconclusive cells:
- `extra_k200`: AUROC Δ=0.17pp / AUPRC Δ=0.22pp
- `nin_k200`: AUROC Δ=0.19pp / AUPRC Δ=0.30pp

## Cell: extra_k200

- best_avg_macro_auroc: **0.8456**
- baseline_avg_macro_auroc: **0.8439**
- avg macro AUROC Δ: **+0.17pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.70 | **+0.35pp** | 69.03 | 69.67 | **+0.64pp** |
| cpsc_2018_extra | 1000 | 81.01 | 81.01 | **+0.00pp** | 68.48 | 68.18 | **-0.30pp** |
| georgia | 1000 | 81.55 | 81.72 | **+0.17pp** | 68.22 | 68.50 | **+0.28pp** |
| ningbo | 1000 | 86.66 | 86.79 | **+0.13pp** | 72.49 | 72.73 | **+0.24pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.41pp | +0.56pp | +0.32pp | +0.24pp | +0.24pp |
| cpsc_2018_extra | +0.18pp | -0.16pp | -0.35pp | N/A | +0.32pp |
| georgia | +0.39pp | +0.23pp | N/A | -0.01pp | +0.09pp |
| ningbo | +0.35pp | +0.44pp | +0.27pp | -0.02pp | -0.39pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 1.0 | 0.0757 | 0.2979 | 0.9072 | 2048 | N | 0.5335 | 0.573 |
| 42 | 0.97 | 0.0799 | 0.1695 | 0.9071 | 2048 | N | 0.5329 | 0.5738 |
| 43 | 1.0 | 0.0792 | 0.0959 | 0.8955 | 2048 | N | 0.5319 | 0.5714 |
| 44 | 0.99 | 0.0757 | 0.5424 | 0.9002 | 2048 | N | 0.5383 | 0.5742 |
| 45 | 0.99 | 0.0701 | 0.0498 | 0.8837 | 2048 | N | 0.5398 | 0.5734 |
| 46 | 0.99 | 0.0715 | 0.0019 | 0.9307 | 2048 | N | 0.5317 | 0.5709 |
| 47 | 0.99 | 0.0772 | 0.1124 | 0.9164 | 2048 | N | 0.5358 | 0.5689 |
| 48 | 1.0 | 0.0646 | 0.3996 | 0.9046 | 2048 | N | 0.54 | 0.5736 |
| 49 | 0.97 | 0.0754 | 0.1155 | 0.909 | 2048 | N | 0.5309 | 0.5698 |
| 50 | 0.99 | 0.0705 | 0.0157 | 0.9039 | 2048 | N | 0.536 | 0.571 |

## Cell: nin_k200

- best_avg_macro_auroc: **0.8441**
- baseline_avg_macro_auroc: **0.8422**
- avg macro AUROC Δ: **+0.19pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.68 | **+0.33pp** | 69.03 | 69.49 | **+0.46pp** |
| cpsc_2018_extra | 1000 | 79.84 | 79.88 | **+0.04pp** | 67.59 | 67.64 | **+0.05pp** |
| georgia | 1000 | 82.16 | 82.30 | **+0.14pp** | 68.15 | 68.48 | **+0.33pp** |
| ningbo | 1000 | 86.55 | 86.77 | **+0.22pp** | 72.29 | 72.67 | **+0.38pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.22pp | +0.74pp | +0.30pp | +0.13pp | +0.28pp |
| cpsc_2018_extra | +0.17pp | -0.22pp | -0.14pp | N/A | +0.34pp |
| georgia | +0.39pp | +0.17pp | N/A | -0.08pp | +0.09pp |
| ningbo | +0.57pp | +0.48pp | +0.16pp | +0.05pp | -0.17pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 1.0 | 0.0703 | 0.3195 | 0.9404 | 2048 | N | 0.5283 | 0.5725 |
| 42 | 1.0 | 0.0749 | 0.1054 | 0.9224 | 2048 | N | 0.5267 | 0.5706 |
| 43 | 0.98 | 0.0729 | 0.1171 | 0.9455 | 2048 | N | 0.5268 | 0.5707 |
| 44 | 0.97 | 0.0664 | 0.1095 | 0.931 | 2048 | N | 0.5184 | 0.5769 |
| 45 | 0.98 | 0.0743 | 0.1324 | 0.9321 | 2048 | N | 0.5297 | 0.5718 |
| 46 | 0.99 | 0.0699 | 0.0458 | 0.9356 | 2048 | N | 0.5296 | 0.5697 |
| 47 | 0.98 | 0.0681 | 0.1241 | 0.9308 | 2048 | N | 0.5304 | 0.5709 |
| 48 | 0.99 | 0.0669 | 0.1149 | 0.9379 | 2048 | N | 0.5281 | 0.5721 |
| 49 | 0.98 | 0.0764 | 0.0461 | 0.9266 | 2048 | N | 0.5306 | 0.5722 |
| 50 | 1.0 | 0.0703 | 0.0033 | 0.9267 | 2048 | N | 0.5313 | 0.5734 |
