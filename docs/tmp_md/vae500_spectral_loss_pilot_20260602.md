# VAE500 Spectral-Loss Pilot

Date: 2026-06-02

## Purpose

Test whether adding a weak frequency-domain reconstruction constraint improves
the 500Hz VAE latent geometry used by VAE-only online adversarial training.

Motivation:

```text
SE-Diff uses simulator / beat / inter-lead constraints around an ECGTwin-style
VAE latent generator. Our current VAE500 already has first-difference and
lead-consistency losses, but no frequency-domain reconstruction term.
```

## Code Change

Added an optional spectral reconstruction term:

```text
spectral_logmag(x) = log(1 + abs(rfft(x, dim=time)))
spectral_huber    = SmoothL1(spectral_logmag(recon), spectral_logmag(target))
```

Files:

```text
ecg_adv_gen/vae/losses.py
ecg_adv_gen/vae/__init__.py
scripts/vae500/train_ptbxl_vae500.py
```

The new CLI flag is:

```text
--spectral_weight
```

Default is `0.0`, so historical VAE training commands are unchanged.

## VAE Training Pilots

Base checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt
```

Both pilots used:

```text
cache:       /root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1
init:        base checkpoint above
epochs:      4
batch_size:  64
lr:          1e-5
PTB-XL only: yes
```

| VAE | spectral_weight | hard gate | audit global Pearson | audit first-diff Pearson | audit MSE | audit lead residual ratio |
|---|---:|---|---:|---:|---:|---:|
| current VAE500 | 0.000 | pass | 0.9977 | 0.9505 | 0.00710 | 1.088 |
| spectral-r005 | 0.005 | fail | 0.9977 | 0.9453 | 0.00708 | 1.206 |
| spectral-r001 | 0.001 | pass | 0.9977 | 0.9507 | 0.00694 | 1.161 |

Interpretation:

```text
spectral_weight=0.005 is too strong because it worsens lead-consistency beyond
the hard gate threshold.

spectral_weight=0.001 is safe: it passes the reconstruction hard gate and
slightly improves MSE while preserving first-difference Pearson.
```

## Downstream Two-Center Online AT Pilot

New spectral-r001 K500 anchors were exported before downstream training:

```text
/root/autodl-tmp/vae500_lhat_v7/anchors_k500_spectral_r001/cpsc_2018/
/root/autodl-tmp/vae500_lhat_v7/anchors_k500_spectral_r001/ningbo/
```

Downstream recipe matched old VAE K500 old-recipe as closely as possible:

```text
backbone:           EfficientNet1DV2 500Hz source checkpoint
K target anchors:   500
K500 internal val:  20%
K_anchor:           300
hull_M:             10
hull_lambda:        0.15
hull_steps:         5
adv_label_mode:     mixed_soft
adv_teacher_mix:    0.3
adv_weight:         0.06
target_real_weight: 40
epochs:             10
classes_in_scope:   NORM, MI, STTC
quality gate:       enabled, same as old-recipe run
final eval:         target-center K500 ref ids excluded
```

| center | direct40 | old VAE500 | spectral-r001 VAE500 | spectral - direct40 | spectral - old VAE |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8671 / 0.6221 | 0.8674 / 0.6226 | +0.40pp / +0.71pp | +0.03pp / +0.05pp |
| ningbo | 0.8810 / 0.5032 | 0.8790 / 0.4964 | 0.8806 / 0.5024 | -0.04pp / -0.08pp | +0.16pp / +0.60pp |

PTB-XL source floor:

| run | PTB-XL AUROC / AUPRC |
|---|---:|
| CPSC direct40 | 0.9086 / 0.7749 |
| CPSC spectral-r001 | 0.9080 / 0.7735 |
| Ningbo direct40 | 0.9075 / 0.7723 |
| Ningbo spectral-r001 | 0.9081 / 0.7739 |

## Decision

Spectral-r001 is a safe VAE refinement but not a breakthrough.

It improves over the old VAE on both tested centers, but the margin is small.
It does not reach the goal of `+2pp` over matched direct target fine-tune.

Next direction should not be simply increasing spectral weight. The useful
follow-up is to change the online-AT objective or selection rule, for example:

```text
1. adversarial acceptance based on positive loss_gain;
2. uncertainty-window / boundary-only adversarial samples;
3. source-plus-target checkpoint selection that avoids K500-val overfitting;
4. SE-Diff-style beat/first-cycle auxiliary decoder only if it passes the same
   VAE reconstruction and latent-hull gates.
```

## Artifacts

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_spectral_r005_b64_20260602/
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_spectral_r001_b64_20260602/
/root/autodl-tmp/vae500_lhat_v7/anchors_k500_spectral_r001/
/root/autodl-tmp/vae500_spectral_r001_oldrecipe_k500_v7_20260602/
```

## Follow-up: Loss-Gain Acceptance Filter

Code addition:

```text
scripts/pgd_cross_center/synth_online_at_super5.py
--adv_accept_min_loss_gain
```

Default is `None`, so historical runs are unchanged. With
`--adv_accept_min_loss_gain 0.0`, adversarial samples are pushed into the buffer
only when:

```text
adv_bce - clean_bce >= 0
```

CPSC pilot:

```text
VAE:     spectral-r001
recipe:  same old-recipe K500 setting
filter:  --adv_accept_min_loss_gain 0.0
```

Filtering behavior:

| epoch | pushed | dropped by loss_gain | dropped by trust |
|---:|---:|---:|---:|
| 1 | 80 | 84 | 2 |
| 2 | 89 | 75 | 2 |
| 3 | 95 | 69 | 2 |
| 9 | 101 | 63 | 2 |
| 10 | 103 | 61 | 2 |

Held-out CPSC result:

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference |
| old VAE500 | 0.8671 / 0.6221 | +0.38pp / +0.66pp |
| spectral-r001 VAE500 | 0.8674 / 0.6226 | +0.40pp / +0.71pp |
| spectral-r001 + loss-gain acceptance | 0.8688 / 0.6270 | +0.55pp / +1.15pp |

Interpretation:

```text
Loss-gain acceptance is more useful than spectral VAE refinement alone on CPSC,
but still falls short of the +2pp goal. It is worth testing on Ningbo/Chapman
or combining with a boundary-probability window, but it is not yet sufficient
as the final method.
```

Additional artifact:

```text
/root/autodl-tmp/vae500_spectral_r001_acceptlg_k500_v7_20260602/
```

Ningbo stability check:

| method | ningbo AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8810 / 0.5032 | reference |
| old VAE500 | 0.8790 / 0.4964 | -0.20pp / -0.68pp |
| spectral-r001 VAE500 | 0.8806 / 0.5024 | -0.04pp / -0.08pp |
| spectral-r001 + loss-gain acceptance | 0.8807 / 0.5013 | -0.03pp / -0.19pp |

Interpretation:

```text
Loss-gain acceptance helps CPSC but does not solve Ningbo. On Ningbo it keeps
AUROC near direct40 but AUPRC remains below direct40. Do not claim the
acceptance filter as a global recipe yet.
```

## Follow-up: Boundary Window + Loss-Gain Acceptance

CPSC pilot:

```text
VAE:                    spectral-r001
filter 1:               adv_bce - clean_bce >= 0
filter 2:               target sigmoid probability in [0.15, 0.85]
other recipe settings:  same old-recipe K500 setting
```

Internal validation:

```text
baseline quick-val:      0.8642 / 0.7149
acceptance-only best:    0.9139 / 0.8310
boundary+accept best:    0.9136 / 0.8282
```

Held-out CPSC:

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference |
| spectral-r001 + loss-gain acceptance | 0.8688 / 0.6270 | +0.55pp / +1.15pp |
| spectral-r001 + loss-gain + prob[0.15,0.85] | 0.8670 / 0.6229 | +0.37pp / +0.74pp |

Decision:

```text
The [0.15,0.85] boundary window is too restrictive in this recipe. It reduces
buffer size and underperforms acceptance-only. Do not expand this exact window.
If boundary filtering is revisited, use a looser window or score-based sampling
instead of hard rejection.
```

Additional artifact:

```text
/root/autodl-tmp/vae500_spectral_r001_accept_boundary_k500_v7_20260602/
```
