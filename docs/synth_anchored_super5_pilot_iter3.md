# Synth-anchored Super5 pilot — 2026-04-26 08:49

Pilot of CenterToken-anchored VAE-latent PGD online AT against Super5 baseline. 2 cells × K=200, AugMix-off (Plan Rev 8).

## Decision gate

- Threshold: ≥1 cell with **macro AUROC Δ > +0.3pp** AND **macro AUPRC Δ > +0.5pp** (Plan Issue #34)
- Pass: **0/2**
- Decision: **STOP-AND-ANALYZE**

Failing / inconclusive cells:
- `extra_k200`: AUROC Δ=0.20pp / AUPRC Δ=0.17pp
- `nin_k200`: AUROC Δ=0.17pp / AUPRC Δ=0.22pp

## Cell: extra_k200

- best_avg_macro_auroc: **0.8459**
- baseline_avg_macro_auroc: **0.8439**
- avg macro AUROC Δ: **+0.20pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.76 | **+0.41pp** | 69.03 | 69.38 | **+0.35pp** |
| cpsc_2018_extra | 1000 | 81.01 | 80.95 | **-0.06pp** | 68.48 | 68.22 | **-0.26pp** |
| georgia | 1000 | 81.55 | 81.80 | **+0.25pp** | 68.22 | 68.57 | **+0.35pp** |
| ningbo | 1000 | 86.66 | 86.86 | **+0.20pp** | 72.49 | 72.72 | **+0.23pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.45pp | +0.46pp | +0.23pp | +0.20pp | +0.73pp |
| cpsc_2018_extra | +0.32pp | -0.18pp | -0.58pp | N/A | +0.20pp |
| georgia | +0.37pp | +0.33pp | N/A | -0.02pp | +0.33pp |
| ningbo | +0.14pp | +0.36pp | +0.36pp | -0.01pp | +0.17pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 1.0 | 0.0862 | 0.1834 | 0.7298 | 2048 | N | 0.5353 | 0.5711 |
| 42 | 0.99 | 0.0892 | 0.0123 | 0.7381 | 2048 | N | 0.5337 | 0.5722 |
| 43 | 1.0 | 0.0883 | 0.1065 | 0.7489 | 2048 | N | 0.5302 | 0.5704 |
| 44 | 0.99 | 0.0864 | 0.247 | 0.7548 | 2048 | N | 0.5345 | 0.5711 |
| 45 | 0.99 | 0.0813 | 0.0823 | 0.7336 | 2048 | N | 0.5291 | 0.5706 |
| 46 | 1.0 | 0.0844 | 0.637 | 0.7967 | 2048 | N | 0.5224 | 0.572 |
| 47 | 0.98 | 0.0872 | 0.068 | 0.7766 | 2048 | N | 0.5337 | 0.5733 |
| 48 | 1.0 | 0.0714 | 0.1091 | 0.7564 | 2048 | N | 0.5399 | 0.5701 |
| 49 | 1.0 | 0.0838 | 0.1392 | 0.7614 | 2048 | N | 0.5333 | 0.5718 |
| 50 | 1.0 | 0.0819 | 0.0655 | 0.7449 | 2048 | N | 0.5407 | 0.5699 |

## Cell: nin_k200

- best_avg_macro_auroc: **0.8439**
- baseline_avg_macro_auroc: **0.8422**
- avg macro AUROC Δ: **+0.17pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.63 | **+0.28pp** | 69.03 | 69.39 | **+0.36pp** |
| cpsc_2018_extra | 1000 | 79.84 | 79.85 | **+0.01pp** | 67.59 | 67.53 | **-0.06pp** |
| georgia | 1000 | 82.16 | 82.28 | **+0.12pp** | 68.15 | 68.37 | **+0.22pp** |
| ningbo | 1000 | 86.55 | 86.79 | **+0.24pp** | 72.29 | 72.66 | **+0.37pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.13pp | +0.60pp | +0.19pp | +0.15pp | +0.33pp |
| cpsc_2018_extra | +0.08pp | -0.14pp | +0.02pp | N/A | +0.08pp |
| georgia | +0.26pp | +0.12pp | N/A | +0.00pp | +0.09pp |
| ningbo | +0.38pp | +0.49pp | +0.10pp | +0.18pp | +0.07pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 1.0 | 0.0725 | 0.0486 | 0.8218 | 2048 | N | 0.5369 | 0.5721 |
| 42 | 1.0 | 0.081 | 0.3513 | 0.7951 | 2048 | N | 0.5274 | 0.5731 |
| 43 | 0.99 | 0.0783 | 0.2864 | 0.8223 | 2048 | N | 0.5293 | 0.5719 |
| 44 | 0.99 | 0.0736 | 0.144 | 0.7987 | 2048 | N | 0.5305 | 0.5713 |
| 45 | 1.0 | 0.0808 | 0.2351 | 0.811 | 2048 | N | 0.5327 | 0.5718 |
| 46 | 1.0 | 0.0758 | 0.094 | 0.8078 | 2048 | N | 0.5182 | 0.5701 |
| 47 | 1.0 | 0.0757 | 0.3593 | 0.8217 | 2048 | N | 0.5232 | 0.5732 |
| 48 | 1.0 | 0.0709 | 0.3103 | 0.8139 | 2048 | N | 0.5297 | 0.5698 |
| 49 | 0.98 | 0.0842 | 0.2723 | 0.7933 | 2048 | N | 0.5294 | 0.5748 |
| 50 | 1.0 | 0.0744 | 0.6775 | 0.8183 | 2048 | N | 0.5324 | 0.5739 |
