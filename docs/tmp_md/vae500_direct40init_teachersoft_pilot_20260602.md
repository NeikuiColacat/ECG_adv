# VAE500 Direct40-Init Teacher-Soft Pilot

Date: 2026-06-02

## Purpose

This pilot tested whether the 500Hz VAE adversarial stream fails because hard or
mixed labels are too destructive. It starts from the matched `direct40/ep10`
checkpoint and removes supervised target-real confounding:

```text
init checkpoint: per-center direct40/ep10/no-adv best_model.pt
target_real_weight: 0
ptbxl_weight: 1
adv_weight: 0.30
adv_label_mode: teacher_soft
classes_in_scope: NORM, MI, STTC
hull_lambda: 0.15
hull_M: 10
hull_steps: 5
K_anchor: 300
epochs: 8
```

Run root:

```text
/root/autodl-tmp/vae500_direct40init_teachersoft_pilot_v7_20260602
```

## Results

Both centers selected epoch 0 by K500-internal validation. Therefore the final
held-out target result is identical to direct40.

| center | direct40/ep10 | direct40-init teacher-soft VAE | delta |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8633 / 0.6155 | +0.00pp / +0.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8810 / 0.5032 | +0.00pp / +0.00pp |

## Training Behavior

| center | direct40 K500-val baseline | best post-update K500-val | last K500-val | ASR mean | loss_gain mean |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.9128 / 0.8289 | 0.9043 / 0.7765 | 0.8890 / 0.7393 | 0.4126 | -0.0301 |
| ningbo | 0.8748 / 0.8012 | 0.8677 / 0.7857 | 0.8622 / 0.7703 | 0.3833 | 0.0465 |

The attack itself was active: ASR stayed in the practical 30-70% range and
decoded invalid rate stayed at 0. The problem is that the adversarial updates
move the model away from the direct40 K500 validation optimum.

## Interpretation

Teacher-soft labels do not fix the 500Hz VAE issue. This weakens the hypothesis
that the failure is only caused by hard-label noise in mixed latent samples.

Current evidence points to a deeper issue:

```text
The 500Hz VAE latent-hull branch creates valid adversarial pressure, but under
the current latent geometry and loss design it does not add held-out
generalization beyond a properly matched direct target fine-tune.
```

## Why 100Hz Looked Better Than 500Hz

The difference should not be attributed to sampling rate alone. Several factors
changed together:

1. The 500Hz direct baseline is stronger because it uses 5000 time samples and
   can learn more target-center morphology directly from K500 real ECGs.
2. The stricter comparison is now VAE AT versus matched direct target FT, not
   VAE AT versus PTB-XL source-only.
3. The 500Hz VAE latent is `(4, 625)`, much larger than ECGTwin's original
   `(4, 128)` latent. It reconstructs well, but may be less semantically
   compressed for diagnosis/domain interpolation.
4. The v7 label mapping changes K500 anchor distributions. MI is almost absent
   in several centers, so older `NORM/MI/STTC` recipes no longer match the
   target label geometry.

## Next Diagnostic

Before investing in more 500Hz VAE loss sweeps, run a cheap diagnostic using
the original ECGTwin VAE latent manifold with the 500Hz classifier:

```text
target ECG 500Hz
-> downsample / preprocess to ECGTwin 1024 format
-> original ECGTwin VAE encode / latent hull / decode
-> upsample or resample decoded ECG back to 5000 samples
-> evaluate with the 500Hz EfficientNet direct40 control
```

If this diagnostic works better than VAE500, the bottleneck is likely our
VAE500 latent geometry rather than 500Hz classification itself.

The more principled follow-up is a teacher-distilled VAE500:

```text
500Hz reconstruction loss
+ low-rate ECGTwin reconstruction consistency
+ ECGTwin latent / feature consistency
+ optional high-frequency residual branch
```

Do not directly load the original ECGTwin VAE checkpoint into VAE500 without
architecture changes. ECGTwin uses `1024 x 12 -> latent (4, 128)`, while VAE500
uses `5000 x 12 -> latent (4, 625)`.
