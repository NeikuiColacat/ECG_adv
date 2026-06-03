# VAE500 Hard + Soft Loss-Gain Gate

Date: 2026-06-02

## Purpose

Test whether combining two partially useful policies can improve 500Hz/v7
EfficientNet1DV2 VAE-online-AT:

```text
hard loss-gain acceptance:
  only accept adversarial decoded samples if adv_bce - clean_bce >= 0

soft loss-gain buffer weighting:
  among accepted samples, increase replay priority by
  1 + strength * tanh(loss_gain / scale)
```

This is different from the previous pilots:

```text
hard-only: strong on cpsc_2018, weak on ningbo
soft-only: safer on ningbo, weaker than hard-only on cpsc_2018
hard+soft: filter bad samples first, then rank the accepted samples
```

## Fixed Protocol

```text
mapping: v7_super5_sjr_rgq_review_20260528
backbone: EfficientNet1DV2 500Hz
preprocess: minimal_resample
norm: per_sample_global
input: 5000 samples = 10 seconds
VAE: spectral-r001 DiffuSETSVAE500
anchors: target-center K500 real ECG latent anchors
final eval: target-center K500 ref ids excluded
```

Shared training parameters:

```text
attack_mode=latent_hull
hull_M=10
hull_lambda=0.15
hull_steps=5
hull_lr=0.25
K_anchor=300
classes_in_scope=NORM,MI,STTC
adv_label_mode=mixed_soft
adv_teacher_mix=0.3
target_real_weight=40
adv_weight=0.06
adv_accept_min_loss_gain=0.0
adv_loss_gain_score_scale=0.05
adv_loss_gain_score_strength=0.5
selection=K500-internal val_macro_auprc
```

## Results

| center | direct40 | hard-only | soft-only | hard+soft | hard+soft vs direct40 |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8670 / 0.6206 | 0.8679 / 0.6255 | +0.46pp / +1.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | 0.8818 / 0.5063 | 0.8808 / 0.5024 | -0.02pp / -0.08pp |

Source floor:

| center-specific run | PTB-XL fold10 AUROC / AUPRC |
|---|---:|
| cpsc_2018 hard+soft | 0.9081 / 0.7737 |
| ningbo hard+soft | 0.9078 / 0.7735 |

Training behavior:

| center | internal best | ASR behavior | decision signal |
|---|---:|---|---|
| cpsc_2018 | 0.9138 / 0.8320 | around 0.29-0.47 | held-out improves over direct40 but stays below hard-only |
| ningbo | 0.8755 / 0.8032 | around 0.33-0.45 | held-out below direct40 and below soft-only |

Run dirs:

```text
/root/autodl-tmp/vae500_spectral_r001_hardsoftlg_k500_v7_20260602/cpsc_2018_hardlg0_softlg005s05_lam015_m10_k300_ep10_seed42
/root/autodl-tmp/vae500_spectral_r001_hardsoftlg_k500_v7_20260602/ningbo_hardlg0_softlg005s05_lam015_m10_k300_ep10_seed42
```

## Decision

The hard+soft combination is a useful diagnostic but not a solution:

```text
cpsc_2018: positive versus direct40, but weaker than hard-only.
ningbo: fails the gate because it is below direct40 and below soft-only.
```

Do not expand this exact hard+soft loss-gain recipe to Chapman, Georgia,
additional seeds, or benchmark backbones.

Current EfficientNet1DV2 500Hz/v7 conclusion:

```text
The available small recipe family can create local CPSC gains, but it does not
produce a global +2pp VAE-online-AT improvement over matched direct40 target
fine-tune. Internal K500 validation improvements frequently fail to transfer to
ref-excluded PN2021 held-out evaluation.
```

Recommended next work is non-cartesian:

```text
1. Stop adding minor EfficientNet VAE AT knobs unless they change the mechanism.
2. Consolidate the negative and weak-positive evidence into comparison tables.
3. If continuing experiments, move to a different mechanism with a pre-registered
   two-center gate, such as a new VAE objective that passes reconstruction and
   CPSC/Ningbo downstream gates, or a different paper-safe selection/adaptation
   framework.
```
