# ECGTwin-Teacher VAE500 + Loss-Gain Acceptance EfficientNet Pilot

Date: 2026-06-02

## Purpose

Test the user-proposed direction that the current 500Hz VAE may be a bottleneck
and that the original ECGTwin author's VAE weights may help train a stronger
500Hz VAE.

This pilot does not retrain the VAE again. It reuses the already trained
ECGTwin-teacher distilled VAE500 checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_teacher_r002_z001_b64_20260602/checkpoints/best.pt
```

Then it runs EfficientNet1DV2 online AT with the same loss-gain acceptance
filter that helped CPSC in the spectral-r001 branch.

## Setup

```text
backbone:        EfficientNet1DV2 500Hz PTB-XL source checkpoint
input protocol:  minimal_resample, per_sample_global, 500Hz, 5000 samples
mapping:         PN2021 Super5 v7
centers:         cpsc_2018, ningbo
K target:        500 real target ECG anchors
selection:       K500 internal validation, ref ids excluded from final eval
VAE backend:     ECGTwin-teacher DiffuSETSVAE500
attack:          latent_hull, M=10, lambda=0.15, steps=5
labels:          mixed_soft, teacher_mix=0.3
acceptance:      adv_bce - clean_bce >= 0
```

Run root:

```text
/root/autodl-tmp/vae500_teacher_acceptlg_effnet_k500_v7_20260602/
```

## Results

Direct40 and spectral references are from the matched 2026-06-02 runs in:

```text
docs/tmp_md/vae500_spectral_loss_pilot_20260602.md
```

| center | direct40 | spectral-r001 + loss-gain | ECGTwin-teacher VAE + loss-gain |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8686 / 0.6257 |
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5013 | 0.8807 / 0.5003 |

PTB-XL source floor:

| run | PTB-XL AUROC / AUPRC |
|---|---:|
| cpsc_2018 teacher-VAE + loss-gain | 0.9077 / 0.7710 |
| ningbo teacher-VAE + loss-gain | 0.9076 / 0.7727 |

## Interpretation

The ECGTwin-teacher VAE branch is technically valid, but it does not improve
the downstream result over spectral-r001 + loss-gain.

Key points:

- CPSC still benefits from the online AT recipe, but teacher-VAE is slightly
  below spectral-r001 + loss-gain.
- Ningbo remains below matched direct40 AUPRC, so this branch does not pass the
  two-center gate.
- The result supports the current diagnosis: VAE quality matters, but the main
  blocker is not solved by simply transferring ECGTwin's original VAE geometry.

## Decision

Do not expand ECGTwin-teacher VAE + loss-gain to four centers or other
backbones yet.

Next work should prioritize online-AT objective / sample weighting / selection
changes. VAE replacement should only be revisited if the new objective first
shows stable gains with the current or spectral-r001 VAE.
