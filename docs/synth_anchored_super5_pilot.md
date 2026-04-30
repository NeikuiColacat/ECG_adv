# Synth-anchored Super5 pilot — 2026-04-26 07:30

Pilot of CenterToken-anchored VAE-latent PGD online AT against Super5 baseline. 2 cells × K=200, AugMix-off (Plan Rev 8).

## Decision gate

- Threshold: ≥1 cell with **macro AUROC Δ > +0.3pp** AND **macro AUPRC Δ > +0.5pp** (Plan Issue #34)
- Pass: **0/2**
- Decision: **STOP-AND-ANALYZE**

Failing / inconclusive cells:
- `extra_k200`: AUROC Δ=0.15pp / AUPRC Δ=0.18pp
- `nin_k200`: AUROC Δ=0.28pp / AUPRC Δ=0.16pp

## Cell: extra_k200

- best_avg_macro_auroc: **0.8454**
- baseline_avg_macro_auroc: **0.8439**
- avg macro AUROC Δ: **+0.15pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.56 | **+0.21pp** | 69.03 | 69.34 | **+0.31pp** |
| cpsc_2018_extra | 1000 | 81.01 | 81.04 | **+0.03pp** | 68.48 | 68.28 | **-0.20pp** |
| georgia | 1000 | 81.55 | 81.74 | **+0.19pp** | 68.22 | 68.55 | **+0.33pp** |
| ningbo | 1000 | 86.66 | 86.80 | **+0.14pp** | 72.49 | 72.76 | **+0.27pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.20pp | +0.23pp | +0.26pp | +0.17pp | +0.19pp |
| cpsc_2018_extra | +0.24pp | -0.11pp | -0.22pp | N/A | +0.22pp |
| georgia | +0.37pp | +0.24pp | N/A | +0.08pp | +0.09pp |
| ningbo | +0.36pp | +0.27pp | +0.22pp | +0.03pp | -0.19pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 0.99 | 0.0745 | 0.2438 | 0.8932 | 2048 | N | 0.5696 | 0.5793 |
| 42 | 0.97 | 0.0805 | 0.6238 | 0.8785 | 2048 | N | 0.5692 | 0.5786 |
| 43 | 0.99 | 0.0799 | 0.168 | 0.8827 | 2048 | N | 0.5742 | 0.5772 |
| 44 | 0.96 | 0.0758 | 0.3557 | 0.8827 | 2048 | N | 0.5838 | 0.5807 |
| 45 | 0.97 | 0.0732 | 0.1688 | 0.8676 | 2048 | N | 0.5753 | 0.5795 |
| 46 | 0.98 | 0.0722 | 0.3557 | 0.9105 | 2048 | N | 0.5658 | 0.5818 |
| 47 | 0.95 | 0.0785 | 0.0975 | 0.9064 | 2048 | N | 0.568 | 0.5779 |
| 48 | 1.0 | 0.0624 | 0.1 | 0.8951 | 2048 | N | 0.5704 | 0.5823 |
| 49 | 0.96 | 0.0763 | 0.0588 | 0.8891 | 2048 | N | 0.575 | 0.5799 |
| 50 | 0.98 | 0.0722 | 0.2398 | 0.897 | 2048 | N | 0.5817 | 0.5835 |

## Cell: nin_k200

- best_avg_macro_auroc: **0.845**
- baseline_avg_macro_auroc: **0.8422**
- avg macro AUROC Δ: **+0.28pp**
- n_epochs_run: 50

### Per-center AUROC / AUPRC delta

| Center | n | baseline AUROC | best AUROC | Δ AUROC | baseline AUPRC | best AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|---|
| chapman_shaoxing | 1000 | 88.35 | 88.91 | **+0.56pp** | 69.03 | 69.42 | **+0.39pp** |
| cpsc_2018_extra | 1000 | 79.84 | 79.65 | **-0.19pp** | 67.59 | 67.33 | **-0.26pp** |
| georgia | 1000 | 82.16 | 82.37 | **+0.21pp** | 68.15 | 68.45 | **+0.30pp** |
| ningbo | 1000 | 86.55 | 87.07 | **+0.52pp** | 72.29 | 72.53 | **+0.24pp** |

### Per-class AUROC delta (best epoch, per center)

| Center | CD | HYP | MI | NORM | STTC |
|---|---|---|---|---|---|
| chapman_shaoxing | +0.36pp | +0.57pp | -0.10pp | +0.20pp | +1.78pp |
| cpsc_2018_extra | +0.13pp | -0.25pp | -0.63pp | N/A | -0.03pp |
| georgia | +0.51pp | +0.38pp | N/A | -0.22pp | +0.20pp |
| ningbo | +0.89pp | +0.15pp | +0.04pp | -0.18pp | +1.71pp |

### PGD gates time series (last 10 epochs)

| Epoch | ASR | Einthoven p95 | HR Δ | QRS ratio | Buffer | Skipped | Train | Val |
|---|---|---|---|---|---|---|---|---|
| 41 | 0.98 | 0.0682 | 0.4151 | 0.9302 | 2048 | N | 0.5503 | 0.5839 |
| 42 | 0.96 | 0.0755 | 0.2879 | 0.9056 | 2048 | N | 0.55 | 0.5806 |
| 43 | 0.98 | 0.0715 | 0.2471 | 0.9369 | 2048 | N | 0.5529 | 0.5827 |
| 44 | 0.98 | 0.0659 | 0.1088 | 0.9278 | 2048 | N | 0.5646 | 0.5849 |
| 45 | 0.97 | 0.0741 | 0.1972 | 0.9248 | 2048 | N | 0.5509 | 0.5876 |
| 46 | 0.97 | 0.0701 | 0.0098 | 0.9225 | 2048 | N | 0.5465 | 0.5801 |
| 47 | 0.95 | 0.0674 | 0.1035 | 0.9234 | 2048 | N | 0.5575 | 0.5788 |
| 48 | 0.99 | 0.0666 | 0.0268 | 0.9239 | 2048 | N | 0.5475 | 0.5829 |
| 49 | 0.95 | 0.0772 | 0.0314 | 0.9226 | 2048 | N | 0.5492 | 0.5823 |
| 50 | 0.96 | 0.0674 | 0.3601 | 0.9207 | 2048 | N | 0.5504 | 0.5799 |
