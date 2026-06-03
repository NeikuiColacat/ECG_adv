# Soft Loss-Gain Buffer Weighting Pilot

Date: 2026-06-02

## Purpose

Test whether hard loss-gain acceptance is too coarse for Ningbo. Instead of
dropping adversarial samples with negative loss gain, this pilot keeps them but
softly reweights the adversarial buffer score.

## Code Change

Added two default-off CLI options to:

```text
scripts/pgd_cross_center/synth_online_at_super5.py
```

```text
--adv_loss_gain_score_scale
--adv_loss_gain_score_strength
```

When enabled:

```text
score *= 1 + strength * tanh(loss_gain / scale)
```

For this pilot:

```text
scale=0.05
strength=0.5
```

So zero loss gain leaves the score unchanged, positive gain upweights the
sample, and negative gain downweights it. Historical behavior is unchanged when
`strength=0`.

## Setup

```text
backbone:        EfficientNet1DV2 500Hz
VAE:             spectral-r001 DiffuSETSVAE500
mapping:         PN2021 Super5 v7
K target:        500
selection:       K500 internal validation AUPRC
final eval:      PN2021 ref-excluded held-out
```

Run root:

```text
/root/autodl-tmp/vae500_spectral_r001_softlg_esauprc_k500_v7_20260602/
```

## Results

| center | direct40 | hard loss-gain + AUPRC selection | soft loss-gain weighting |
|---|---:|---:|---:|
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | 0.8818 / 0.5063 |
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8670 / 0.6206 |

PTB-XL source floor:

| center-run | PTB-XL AUROC / AUPRC |
|---|---:|
| ningbo soft loss-gain | 0.9076 / 0.7750 |
| cpsc_2018 soft loss-gain | 0.9082 / 0.7746 |

## Interpretation

Soft loss-gain weighting is useful evidence but not a standalone global recipe.

- It fixes the small Ningbo AUPRC drop and becomes slightly better than direct40.
- It weakens the CPSC gain compared with hard loss-gain acceptance.
- This suggests the best policy may be center-adaptive or selector-based rather
  than one universal acceptance rule.

## Decision

Keep soft loss-gain weighting as a candidate for the selector pool or for
centers where hard filtering over-prunes useful adversarial samples.

Do not replace hard loss-gain acceptance globally yet.
