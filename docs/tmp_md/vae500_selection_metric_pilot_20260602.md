# VAE500 Online AT Checkpoint Selection Metric Pilot

Date: 2026-06-02

## Purpose

Test whether the Ningbo underperformance is caused by selecting checkpoints
with internal validation AUROC instead of AUPRC.

## Setup

```text
center:          ningbo
backbone:        EfficientNet1DV2 500Hz
VAE:             spectral-r001 DiffuSETSVAE500
acceptance:      adv_bce - clean_bce >= 0
selection A:     val_macro_auroc  (previous run)
selection B:     val_macro_auprc  (this pilot)
final eval:      PN2021 ref-excluded target-center held-out
```

Run:

```text
/root/autodl-tmp/vae500_spectral_r001_acceptlg_esauprc_k500_v7_20260602/ningbo_lg0_lam015_m10_k300_ep10_seed42
```

## Result

| method | selected checkpoint | ningbo AUROC / AUPRC | delta vs direct40 |
|---|---|---:|---:|
| direct40 | reference | 0.8810 / 0.5032 | reference |
| spectral-r001 + loss-gain, select AUROC | best val AUROC | 0.8807 / 0.5013 | -0.03pp / -0.19pp |
| spectral-r001 + loss-gain, select AUPRC | best val AUPRC | 0.8807 / 0.5018 | -0.03pp / -0.14pp |

PTB-XL source floor for AUPRC-selection:

```text
0.9071 / 0.7722
```

## Decision

Selecting by AUPRC slightly improves Ningbo AUPRC, but not enough to beat the
matched direct40 control. The main bottleneck is not simply the early-stopping
metric.

Do not expand selection-metric-only sweeps. Continue with a genuinely different
online-AT objective or sample weighting strategy.
