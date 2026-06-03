# VAE500 Positive-Hide / Negative-Add Pilot

Date: 2026-06-02

## Purpose

Test whether the 500Hz VAE online AT bottleneck comes from the old full-BCE
attack objective. The old objective maximizes BCE over all five labels at once,
so negative labels can dominate. This pilot splits the objective into:

```text
positive-hide: make true positive labels fall below threshold
negative-add: make false positive labels rise above threshold
```

## Code Change

Default behavior is unchanged.

```text
adversarial/pgd_advdiff.py
  --attack_loss_mode bce keeps old full-BCE attack.
  New modes: positive_hide, negative_add, pos_hide_neg_add.

adversarial/latent_hull_pgd.py
  latent-hull coefficient optimization now uses the same attack helper.

adversarial/adv_validation.py
  ASR now also reports negative-add metrics.

scripts/pgd_cross_center/synth_online_at_super5.py
  exposes attack objective arguments and --adv_accept_gain_mode bce|target.
```

Smoke checks:

```text
py_compile: passed
old BCE helper equals torch BCE: passed
new objective has finite gradients: passed
negative-add ASR metrics: passed
```

## Pilot Config

```text
run_dir:
  /root/autodl-tmp/vae500_poshide_k500_v7_20260602/cpsc_2018_poshide_negadd_ep10_seed42

backbone:
  EfficientNet1DV2 500Hz

VAE:
  /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_spectral_r001_b64_20260602/checkpoints/best.pt

target center:
  cpsc_2018, K=500, ref ids excluded in final eval

attack:
  attack_mode=latent_hull
  attack_loss_mode=pos_hide_neg_add
  attack_pos_hide_weight=1.0
  attack_neg_add_weight=0.25
  attack_negative_exclude_classes=NORM
  adv_accept_gain_mode=target
  hull_M=10
  hull_lambda=0.15
  hull_steps=5
  hull_lr=0.25

training labels:
  adv_label_mode=multi_hot_hard

matched real stream:
  target_real_weight=40
  adv_weight=0.06
  n_epochs=10
  es_metric=val_macro_auprc
```

The train split had no MI anchors after the internal K500 validation split, so
the actual latent attack used NORM and STTC anchors.

## Result

Training stopped at epoch 10 because ASR fell below 0.30 for three consecutive
epochs. The best checkpoint was saved at epoch 9.

| method | CPSC heldout AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 target-real control | 0.8633 / 0.6155 | reference |
| current hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp |
| poshide+negadd pilot | 0.8675 / 0.6234 | +0.42pp / +0.79pp |

PTB-XL fold10:

```text
direct40:        0.9086 / 0.7749
hard loss-gain:  0.9079 / 0.7733
poshide+negadd:  0.9081 / 0.7739
```

## Decision

The new objective is useful diagnostically and remains source-safe, but it does
not beat the current hard loss-gain CPSC result. Do not expand this exact recipe
to four centers.

Next higher-value branch:

```text
Run gated ECGTwin-init / ECGTwin-teacher VAE500 checks:
1. reconstruction audit against spectral-r001;
2. latent-hull decoded validity;
3. CPSC and Ningbo downstream smoke;
4. expand only if CPSC improves and Ningbo does not regress.
```
