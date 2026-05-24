# Multi-Backbone VAE-Only L-HAT v5 Summary

Date: 2026-05-24

## Purpose

Re-evaluate the existing five `model/ecg_ptbxl_benchmarking` backbones under
the current PN2021 Super5 v5 mapping, without retraining source models and
without rerunning online AT.

Backbones:

```text
benchmark_fcn_wang
benchmark_resnet1d_wang
benchmark_inception1d
benchmark_lstm
benchmark_xresnet1d101
```

Evaluation protocol:

- PTB-XL Super5 source model, 100 Hz / 1000 samples.
- K=500 target-center real anchors excluded from that center's PN2021 eval.
- Compare source checkpoint vs existing VAE-only Latent-Hull online AT
  checkpoint.
- PN2021 mapping/cache: `v5_super5_strict_voltage_pacing_suppress`.

Summary files:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_multibackbone_vae_only_lhat_20260522/summaries/multibackbone_vae_only_lhat_v5_20260524.csv
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/paper_multibackbone_vae_only_lhat_20260522/summaries/multibackbone_vae_only_lhat_v5_20260524.md
```

## Parameter Counts

| model | parameters |
|---|---:|
| benchmark_fcn_wang | 0.277M |
| benchmark_resnet1d_wang | 0.441M |
| benchmark_inception1d | 0.475M |
| benchmark_lstm | 0.808M |
| benchmark_xresnet1d101 | 1.808M |

## Mean Effects Across 4 Target Centers

| model | mean target delta AUROC/AUPRC | target wins | mean PTB-XL delta AUROC/AUPRC | mean PN2021 avg delta AUROC/AUPRC |
|---|---:|---:|---:|---:|
| benchmark_fcn_wang | +1.86/+1.47pp | 4/4 | -0.81/-1.68pp | +0.28/-0.44pp |
| benchmark_resnet1d_wang | +2.27/+2.18pp | 4/4 | -0.87/-1.80pp | -0.18/-0.33pp |
| benchmark_inception1d | +2.52/+2.42pp | 4/4 | -0.47/-1.42pp | +0.37/-0.08pp |
| benchmark_lstm | +2.67/+2.93pp | 4/4 | -1.30/-2.65pp | +0.60/+0.22pp |
| benchmark_xresnet1d101 | +2.41/+2.35pp | 3/4 | -0.73/-1.66pp | -0.45/-0.33pp |

## Per Target-Center Delta

| model | center | target delta AUROC/AUPRC | PN2021 avg delta AUROC/AUPRC |
|---|---|---:|---:|
| benchmark_fcn_wang | ningbo | +0.74/+0.37pp | +0.28/-0.18pp |
| benchmark_fcn_wang | chapman_shaoxing | +0.79/+0.66pp | +0.32/-0.60pp |
| benchmark_fcn_wang | cpsc_2018 | +4.74/+3.39pp | +0.25/-1.03pp |
| benchmark_fcn_wang | georgia | +1.15/+1.44pp | +0.25/+0.06pp |
| benchmark_resnet1d_wang | ningbo | +0.64/+0.12pp | -0.19/-0.00pp |
| benchmark_resnet1d_wang | chapman_shaoxing | +1.07/+1.76pp | -0.10/-0.08pp |
| benchmark_resnet1d_wang | cpsc_2018 | +6.12/+4.88pp | -0.44/-1.28pp |
| benchmark_resnet1d_wang | georgia | +1.25/+1.94pp | -0.00/+0.03pp |
| benchmark_inception1d | ningbo | +0.80/+0.14pp | +0.57/+0.56pp |
| benchmark_inception1d | chapman_shaoxing | +1.17/+2.28pp | +0.44/+0.54pp |
| benchmark_inception1d | cpsc_2018 | +6.78/+4.67pp | +0.34/-1.05pp |
| benchmark_inception1d | georgia | +1.34/+2.58pp | +0.14/-0.37pp |
| benchmark_lstm | ningbo | +0.72/+0.35pp | +0.31/+0.38pp |
| benchmark_lstm | chapman_shaoxing | +1.61/+2.56pp | +0.53/+0.73pp |
| benchmark_lstm | cpsc_2018 | +7.02/+6.91pp | +1.22/+0.08pp |
| benchmark_lstm | georgia | +1.33/+1.89pp | +0.32/-0.30pp |
| benchmark_xresnet1d101 | ningbo | +0.84/-0.50pp | -0.96/-0.28pp |
| benchmark_xresnet1d101 | chapman_shaoxing | +1.47/+2.59pp | -0.76/-0.17pp |
| benchmark_xresnet1d101 | cpsc_2018 | +5.49/+4.55pp | +0.26/-0.74pp |
| benchmark_xresnet1d101 | georgia | +1.86/+2.77pp | -0.35/-0.14pp |

## Interpretation

The VAE-only online AT effect transfers across classic PTB-XL benchmark
backbones as a target-center adaptation method: 19 of 20 target-center
AUROC/AUPRC pairs improve, and every backbone has positive mean target-center
delta.

The effect is not yet a clean global PN2021 improvement story. Source-domain
PTB-XL performance consistently drops, and PN2021 7-center average AUPRC often
falls because non-target centers are not protected. This supports the current
research framing as target-center few-shot adaptation, but it also identifies
the next optimization target: keep the target-center gain while reducing source
and non-target-center forgetting.
