# 500Hz VAE V7 Mainline Report, 2026-06-01

## Summary

The 500Hz branch is technically viable. The PTB-XL-only VAE reconstructs
records500 ECG well enough for downstream latent-hull experiments, the 500Hz
EfficientNet1DV2 baseline trains cleanly, and VAE500 online adversarial training
improves all four PN2021 target centers after K500 ref exclusion.

Current EfficientNet1DV2 result:

| method | PN2021 4-center target K500-excluded AUROC / AUPRC |
|---|---:|
| 500Hz EfficientNet1DV2 baseline | 0.8336 / 0.5048 |
| 500Hz EfficientNet1DV2 + VAE500 online AT | 0.8578 / 0.5378 |
| delta | +2.42pp / +3.30pp |

## Input And Acceleration Policy

- Input: `minimal_resample`, `500Hz`, `5000 samples = 10s`.
- EfficientNet1DV2 normalization: per-sample global z-score.
- Benchmark-backbone normalization: dataset-level train mean/std, matching the
  external `ecg_ptbxl_benchmarking` protocol.
- Large arrays are kept under `/root/autodl-tmp`.
- PTB-XL training caches are loaded into RAM when practical.
- PN2021 500Hz evaluation keeps mmap caches on disk, then copies one center at a
  time into RAM for inference.
- Safe PyTorch acceleration defaults: bf16 AMP where the gradient path is not
  the VAE attack path, TF32, CuDNN benchmark, fused AdamW, pinned memory,
  persistent workers, and prefetching.

## VAE500 Audit

Accepted checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt
```

Audit summary:

| metric | value |
|---|---:|
| validation Pearson | 0.9976 |
| first-difference Pearson | 0.9494 |
| leads with Pearson >= 0.90 | 12 / 12 |
| invalid decoded samples | 0 |

## EfficientNet1DV2 Mainline

All values are macro `AUROC / AUPRC`, with target-center K500 records excluded
from that target center's test set.

| center | baseline all-zero-kept | baseline drop-all-zero | VAE500 AT all-zero-kept | VAE500 AT drop-all-zero |
|---|---:|---:|---:|---:|
| ningbo | 0.8598 / 0.4650 | 0.8776 / 0.6375 | 0.8774 / 0.4986 | 0.9013 / 0.6840 |
| chapman_shaoxing | 0.8602 / 0.4150 | 0.8787 / 0.5950 | 0.8728 / 0.4459 | 0.8965 / 0.6405 |
| cpsc_2018 | 0.7958 / 0.5427 | 0.8464 / 0.6702 | 0.8482 / 0.5932 | 0.9009 / 0.7674 |
| georgia | 0.8185 / 0.5963 | 0.8264 / 0.6850 | 0.8327 / 0.6135 | 0.8426 / 0.7075 |
| mean | 0.8336 / 0.5048 | 0.8573 / 0.6470 | 0.8578 / 0.5378 | 0.8853 / 0.6999 |

Source PTB-XL fold10 baseline is `0.9131 / 0.7867`; after online AT, source
metrics remain around `0.9105-0.9120 / 0.7792-0.7848`.

All-zero-kept delta is `+2.42pp / +3.30pp`; drop-all-zero delta is
`+2.80pp / +5.29pp`.

## Historical 100Hz Versus 500Hz

The old 100Hz results below are historical and not strictly apples-to-apples
with the current v7 mapping. They are useful only as a sanity check that the
VAE-only online AT effect size remains similar after moving to 500Hz.

| branch | mapping/input | baseline | VAE online AT | delta |
|---|---|---:|---:|---:|
| historical 100Hz EfficientNet1DV2 | older mapping, 100Hz/1000 | 0.8427 / 0.5134 | 0.8671 / 0.5464 | +2.45pp / +3.29pp |
| current 500Hz EfficientNet1DV2 | v7, 500Hz/5000 | 0.8336 / 0.5048 | 0.8578 / 0.5378 | +2.42pp / +3.30pp |

## ECGFounder Branch

| method | four target-center mean AUROC / AUPRC | interpretation |
|---|---:|---|
| frozen linear probe | 0.8610 / 0.5565 | strong frozen foundation baseline |
| direct K500 head fine-tune | 0.9086 / 0.6739 | strongest current K500 adaptation control |
| VAE500 online AT from direct K500 head | 0.9086 / 0.6739 | epoch-0 selected; no added gain in first recipe |

ECGFounder is a strong comparison branch. Its current improvement is mainly
from target K500 supervised head fine-tuning, not from VAE500 online AT.

## Benchmark Backbones

All PN2021 values are target-center K500-excluded macro `AUROC / AUPRC`.

| model | PTB-XL fold10 | PN2021 4-center mean |
|---|---:|---:|
| `fastai_inception1d` | 0.9247 / 0.8138 | 0.8452 / 0.4971 |
| `fastai_resnet1d_wang` | 0.9175 / 0.7984 | 0.8246 / 0.4833 |
| `fastai_xresnet1d50` | 0.9206 / 0.8079 | 0.8511 / 0.5120 |
| `fastai_fcn_wang` | 0.9084 / 0.7869 | 0.8245 / 0.4838 |
| `fastai_schirrmeister` | 0.9070 / 0.7651 | 0.8264 / 0.4671 |

The best direct benchmark backbone is `fastai_xresnet1d50`. It is competitive
with the EfficientNet1DV2 direct baseline but still below the EfficientNet1DV2 +
VAE500 online AT branch on four-center AUPRC.

## Current Interpretation

The most paper-safe claim is:

```text
Using a PTB-XL-only 500Hz VAE latent space as an on-manifold online adversarial
regularizer improves the 500Hz EfficientNet1DV2 Super5 classifier on PN2021
target centers under v7 label mapping, while preserving most PTB-XL source
performance.
```

The direct ECGTwin/DiT synthetic-sample story should remain an ablation, not the
main causal explanation. ECGFounder should be reported as a strong foundation
model comparison and direct K500 adaptation control.

## Detailed Reports

```text
docs/tmp_md/500hz_vae_training_audit_20260601.md
docs/tmp_md/500hz_effnet1dv2_baseline_20260601.md
docs/tmp_md/vae500_lhat_four_center_20260601.md
docs/tmp_md/ecgfounder_500hz_v7_phase6_20260601.md
docs/tmp_md/ptbxl_benchmarking_500hz_v7_phase7_20260601.md
```
