# VAE500 AugMix Source-Floor Inv-Freq Pilot

Date: 2026-06-02

## Purpose

Test a non-trivial objective change instead of continuing small hard/soft
loss-gain sweeps.

Hypothesis:

```text
spectral-r001 VAE500
+ latent AugMix
+ inverse-frequency K-shot anchor quotas
+ light source-logit anchor
may improve over matched direct fine-tune by increasing local target-center
waveform diversity while preserving source performance.
```

## Protocol

Pilot center:

```text
cpsc_2018
```

Fixed inputs:

```text
backbone: EfficientNet1DV2 500Hz
mapping: v7_super5_sjr_rgq_review_20260528
target anchors: K=500, seed42, K500 ref ids excluded from final eval
VAE: spectral-r001 DiffuSETSVAE500
selection: K500 internal validation AUPRC
```

Core training knobs:

```text
--attack_mode latent_hull
--hull_M 10
--hull_lambda 0.15
--hull_steps 5
--hull_lr 0.25
--K_anchor 300
--classes_in_scope NORM MI STTC
--adv_label_mode mixed_soft
--adv_teacher_mix 0.3
--target_real_weight 40
--adv_weight 0.06
--enable_latent_augmix_branch
--latent_augmix_copies 1
--latent_augmix_severity 2
--latent_augmix_latent_weight_cap 0.10
--anchor_class_weight_mode inv_freq_kshot
--anchor_class_weight_gamma 0.35
--anchor_class_weight_cap 2.0
--anchor_class_weight_min 0.5
--source_logit_anchor_weight 0.02
--source_logit_anchor_batches 12
```

Run:

```text
/root/autodl-tmp/vae500_augmix_sourcefloor_invfreq_k500_v7_20260602/
  cpsc_2018_augmix_invfreq_sourceanchor_ep10_seed42
```

## Results

Held-out target-center metrics, K500 ref ids excluded:

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 | PTB-XL AUROC / AUPRC |
|---|---:|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference | 0.9086 / 0.7749 |
| spectral-r001 hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.9079 / 0.7733 |
| hard loss-gain, eval_every=1, select AUPRC | 0.8683 / 0.6255 | +0.50pp / +1.00pp | 0.9082 / 0.7738 |
| AugMix + invfreq + source anchor | 0.8633 / 0.6106 | -0.01pp / -0.49pp | 0.9082 / 0.7734 |
| accepted-only AugMix + hard loss-gain | 0.8669 / 0.6217 | +0.36pp / +0.62pp | 0.9082 / 0.7741 |

Training diagnostics for the AugMix pilot:

```text
best internal K500-val: epoch 3, 0.9135 / 0.8293
baseline internal K500-val: 0.8642 / 0.7149
ASR range before stop: about 0.47 -> 0.28
stop: ASR < 0.30 for 3 consecutive epochs at epoch 8
buffer: reached 2048
decoded invalid / Einthoven: no obvious gate failure
```

Interpretation:

```text
Internal K500 validation improved strongly, but the improvement did not transfer
to ref-excluded held-out CPSC. The AugMix branch overfit or diluted useful VAE
pressure compared with the simpler spectral-r001 hard loss-gain run.
```

## Source-Preserving Last-Block Control

An additional diagnostic direct control was run:

```text
unfreeze_last_n_features=1
source_logit_anchor_weight=0.05
disable_adv_stream
```

Result:

| method | cpsc_2018 AUROC / AUPRC | PTB-XL AUROC / AUPRC |
|---|---:|---:|
| direct40 | 0.8633 / 0.6155 | 0.9086 / 0.7749 |
| source-preserving last-block direct | 0.8218 / 0.5573 | 0.9127 / 0.7873 |

Interpretation:

```text
Source-preserving last-block adaptation improves PTB-XL/source metrics but
hurts target-center held-out CPSC badly. This is not a useful matched direct
control for the current goal.
```

## Decision

Do not expand this AugMix+invfreq recipe to Ningbo or all four centers.

The next implementation-level fix should address a concrete mismatch in the
current AugMix branch:

```text
The main adversarial push can be filtered by loss-gain or boundary gates, but
the latent-AugMix branch is currently built from all adversarial candidates.
That means AugMix can amplify samples that the main branch would reject.
```

Recommended next change:

```text
Make push_adv_to_buffer optionally return an accepted-sample mask, then build
latent-AugMix samples only from accepted adversarial candidates. Re-test on
cpsc_2018 first with:
  spectral-r001 VAE500
  hard loss-gain acceptance
  accepted-only latent AugMix
  same K500 split and ref-excluded eval
```

## Accepted-Only AugMix Follow-Up

Code change:

```text
push_adv_to_buffer now returns accepted_indices.
The latent-AugMix branch now uses only accepted adversarial candidates as
source samples.
```

Smoke validation:

```text
python -m py_compile scripts/pgd_cross_center/synth_online_at_super5.py
push_adv_to_buffer accepted_indices smoke test passed.
```

Run:

```text
/root/autodl-tmp/vae500_hardlg_accepted_augmix_k500_v7_20260602/
  cpsc_2018_hardlg_accepted_augmix_ep10_seed42
```

Training behavior:

```text
accepted-only AugMix generated roughly 78-102 samples per epoch, versus 166
per epoch in the unfiltered AugMix run.
best internal K500-val: epoch 9, 0.9133 / 0.8291
ASR stayed near 0.28-0.47 and did not abort.
```

Result:

```text
cpsc_2018 held-out ref-excluded: 0.8669 / 0.6217
delta vs direct40: +0.36pp / +0.62pp
delta vs spectral-r001 hard loss-gain: -0.19pp / -0.53pp
```

Decision:

```text
The accepted-only fix makes AugMix less harmful, but AugMix still does not beat
the simpler hard loss-gain branch. Keep the code fix because it is the correct
behavior for future AugMix experiments, but do not promote AugMix as the main
current method.
```

## Checkpoint-Selection Follow-Up

A final CPSC check reran the hard loss-gain recipe with every-epoch internal
AUPRC selection:

```text
/root/autodl-tmp/vae500_hardlg_esauprc_eval1_k500_v7_20260602/
  cpsc_2018_hardlg_esauprc_eval1_ep10_seed42
```

Result:

```text
cpsc_2018 held-out ref-excluded: 0.8683 / 0.6255
delta vs direct40: +0.50pp / +1.00pp
delta vs original hard loss-gain: -0.05pp / -0.15pp
```

Decision:

```text
Checkpoint-selection frequency/metric is not the missing mechanism on CPSC.
The best current CPSC EfficientNet1DV2 500Hz branch remains the original
spectral-r001 hard loss-gain run at 0.8688 / 0.6270.
```
