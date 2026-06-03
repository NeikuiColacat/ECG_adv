# VAE500 Low-K K100 Direct vs Online AT Check

Date: 2026-06-02

## Purpose

Test whether VAE500 latent-hull online adversarial training becomes useful when
target adaptation data are scarce. This checks the hypothesis that direct
target fine-tune dominates at K500/K1000, while VAE may help at lower K.

## Protocol

Common setup:

```text
mapping:        v7_super5_sjr_rgq_review_20260528
source model:   EfficientNet1DV2 500Hz PTB-XL source checkpoint
input:          minimal_resample, per_sample_global, 500Hz, 5000 samples
centers:        cpsc_2018, chapman_shaoxing
K:              100 target-center records
validation:     20% of K100 target anchors
final eval:     PN2021 target center, K100 ref ids excluded
VAE checkpoint: /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt
```

Matched direct control:

```text
target_real_weight=40
disable_adv_stream=true
n_epochs=10
```

VAE online AT branch:

```text
target_real_weight=40
classes_in_scope=NORM,MI,STTC
hull_M=10
hull_lambda=0.15
hull_steps=5
K_anchor=80
adv_label_mode=mixed_soft
adv_teacher_mix=0.3
adv_weight=0.06
quality_gate=disabled
n_epochs=10
```

Low-K validation note:

```text
K100 with 20% validation leaves only 20 validation records. The default quick
eval min_pos=10 can leave only one scorable class and create falsely high
validation scores. The online-AT script now supports --quick_eval_min_pos, and
these K100 reruns used --quick_eval_min_pos 2.
```

## Results

| center | direct K100 | VAE500 online AT K100 | delta VAE - direct |
|---|---:|---:|---:|
| cpsc_2018 | 0.8288 / 0.5751 | 0.8281 / 0.5757 | -0.07pp / +0.06pp |
| chapman_shaoxing | 0.8668 / 0.4364 | 0.8671 / 0.4373 | +0.03pp / +0.09pp |

PTB-XL source-floor check:

| center run | direct PTB-XL | VAE PTB-XL |
|---|---:|---:|
| cpsc_2018 K100 | 0.9130 / 0.7873 | 0.9132 / 0.7871 |
| chapman_shaoxing K100 | 0.9131 / 0.7863 | 0.9129 / 0.7863 |

VAE attack diagnostics:

| center | ASR mean | ASR min/max |
|---|---:|---:|
| cpsc_2018 | 0.4727 | 0.4242 / 0.6364 |
| chapman_shaoxing | 0.3566 | 0.3333 / 0.3833 |

## Interpretation

K100 does not unlock the required VAE advantage. Compared with matched direct
fine-tune, VAE500 online AT is effectively neutral on both tested centers:
`-0.07pp/+0.06pp` for CPSC and `+0.03pp/+0.09pp` for Chapman.

This weakens the hypothesis that VAE only fails because K500/K1000 direct
fine-tune is too strong. The more likely blockers are:

```text
1. VAE500 latent geometry is not aligned with target-center discriminative
   boundaries.
2. Current online-AT objective creates valid adversarial samples but not useful
   held-out generalization pressure.
3. K-internal validation at very low K is noisy and must use low min_pos or
   a more stable source-plus-target selection rule.
```

## Decision

Do not expand this exact K100 old-recipe setup to all centers. The next useful
branch should change the VAE or the online adversarial objective, not simply
repeat K scaling.

Recommended next steps:

```text
1. Treat VAE quality/latent-manifold validation as a gate before downstream AT.
2. Continue the ECGTwin-teacher / ECGTwin-init VAE500 direction only if it beats
   the current VAE500 reconstruction and latent-hull validation gates.
3. Add SE-Diff-style VAE constraints if feasible: spectral loss, inter-lead
   physiological consistency, and beat/morphology-preserving decoder pressure.
4. Test adversarial acceptance / uncertainty-window selection from direct40
   initialization before more generic K sweeps.
```

## Artifacts

```text
/root/autodl-tmp/vae500_lhat_v7/anchors_k100/
/root/autodl-tmp/vae500_lowk_k100_direct40_v7_20260602/
/root/autodl-tmp/vae500_lowk_k100_oldrecipe_v7_20260602/
```
