# Graduate Project Self-Distillation Plan

Status: active experiment, started 2026-05-03.

## Motivation

The current best route is:

```text
synthetic20k pretrain
-> real2000 clean fine-tune
-> strict real-anchor latent-hull AT
```

Best current C-series result:

```text
C3 strict custom seed42 test AUROC/AUPRC = 0.8654 / 0.6738
C3 strict PN2021 avg AUROC/AUPRC       = 0.7381 / 0.4230
```

Self-distillation tests whether the best C3 model can transfer its smoother
class boundary back into a student model without adding more low-quality
synthetic hard labels.

## Method

Teacher:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_real_latenthull_at_M10_n1000_seed42/best_model.pt
```

Student initialization:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42/best_model_auprc.pt
```

Training data:

```text
real supervised data = PTB-XL train2000 seed42 split
optional unlabeled synthetic data = ECGTwin synthetic20k candidate pool
validation/test = same PTB-XL seed42 split used by graduate_project.md
```

Loss:

```text
L = hard_weight * L_hard_real + distill_weight * L_soft_teacher

L_hard_real:
  masked BCEWithLogits(student_logits, y_real)

L_soft_teacher:
  BCEWithLogits(student_logits / T, sigmoid(teacher_logits / T)) * T^2

For synthetic samples:
  use teacher soft labels only by default.
```

Why teacher soft labels:

```text
1. C3 teacher has already seen synthetic pretraining + real fine-tune + strict
   latent-hull AT.
2. The teacher probabilities preserve uncertainty across CD/HYP/MI/NORM/STTC.
3. Synthetic hard labels are noisy, especially HYP/CD; teacher soft targets can
   down-weight implausible synthetic semantics.
```

## First Run

Run D1:

```text
teacher = C3 strict
student init = C0 synthetic-only AUPRC checkpoint
real_distill_alpha = 0.3
temperature = 2.0
synth_npz = full synthetic20k samples.npz
synth_ratio = 0.5
synth_distill_weight = 0.5
checkpoint_metric = auprc
```

Evaluation:

```text
custom seed42 test AUROC/AUPRC
official PTB-XL fold10 AUROC/AUPRC through eval_crosscenter.py
PN2021 7-center avg AUROC/AUPRC
```

Success threshold:

```text
D1 should beat C3 strict custom seed42 AUPRC 0.6738 or PN2021 AUPRC 0.4230.
If it only matches C1/C3, report self-distillation as neutral.
If official fold10 improves while custom/PN stay stable, report it as a
regularization route worth repeating across seeds.
```

## 2026-05-03 Results

Implemented entry:

```text
scripts/triple_labels/train_ptbxl_self_distill.py
```

The script trains the same EfficientNet1DV2 student with:

```text
real samples:
  hard masked BCE + teacher soft BCE

synthetic samples:
  teacher soft BCE only, if --synth_npz is supplied
```

Teacher for both runs:

```text
C3 strict teacher:
  /root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_real_latenthull_at_M10_n1000_seed42/best_model.pt
```

Student initialization:

```text
C0 synthetic-only AUPRC checkpoint:
  /root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42/best_model_auprc.pt
```

### Metrics

| run | training data for distillation | custom seed42 AUROC | custom seed42 AUPRC | official fold10 AUROC | official fold10 AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| C3 teacher baseline | synth20k -> real FT -> latent-hull AT | 0.8654 | 0.6738 | 0.7510 | 0.5118 | 0.7381 | 0.4230 |
| D1 | real2000 hard+soft + synthetic20k soft | 0.8670 | 0.6737 | 0.7631 | 0.5273 | 0.7407 | 0.4280 |
| D2 | real2000 hard+soft only | 0.8682 | 0.6836 | 0.7549 | 0.5180 | 0.7308 | 0.4215 |

### Interpretation

```text
D1:
  Best external/PN2021 self-distillation variant so far.
  It improves PN2021 AUPRC over C3 by +0.0050 and official fold10 AUPRC by
  +0.0155, while preserving custom seed42 AUPRC.

D2:
  Best custom seed42 test result so far, AUPRC 0.6836.
  However, PN2021 drops below C3/D1, so it may be over-adapting to the random
  train2000/test-rest split.

Current status:
  Self-distillation is useful.
  For thesis external generalization, D1 is the stronger candidate.
  For in-protocol custom seed42 score, D2 is the strongest candidate.
```

Recommended next ablation:

```text
D3 = D1 with weaker synthetic soft distillation
  real_distill_alpha = 0.3
  synth_distill_weight = 0.25
  temperature = 2.0

Goal:
  Keep D1's PN2021 gain while recovering part of D2's custom seed42 AUPRC.
```

## 2026-05-03 v2 E4 Result

The v2 plan in `graduate_project_self_disillationv2.md` was implemented as a
more faithful low-resource synthetic-sample self-distillation experiment.

Implementation updates:

```text
scripts/triple_labels/prepare_v2_self_distill_synth.py
scripts/triple_labels/train_ptbxl_self_distill.py
```

Protocol:

```text
Teacher:
  3-seed real2000 EfficientNet1DV2 ensemble
  seed42 = /root/autodl-tmp/graduate_project/method_a_real2000_seed42/best_model.pt
  seed43 = /root/autodl-tmp/graduate_project/method_a_real2000_seed43_v2teacher/best_model.pt
  seed44 = /root/autodl-tmp/graduate_project/method_a_real2000_seed44_v2teacher/best_model.pt

Synthetic candidates:
  /root/autodl-tmp/graduate_project/ref_target_mismatch_n20000_translated_randomany_mv4_seed42/ptbxl/samples.npz

Filtered synthetic export:
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_real2000_ens3_seed42/synth_v2_filtered_top2000_gamma03.npz

Student:
  scratch EfficientNet1DV2
  real2000 hard BCE
  synth2000 v2 soft-label BCE
  y_soft = 0.3 * y_condition + 0.7 * p_teacher_ensemble
  synth_distill_weight beta = 0.5
  real:synth sampler ratio = 1:1
  input = 10s, 100Hz, (1000,12)
```

Adaptation from the v2 document:

```text
The v2 document describes 4-crop teacher scoring for a 2.5s classifier.
The current graduate-project classifier uses 10s inputs, so this run used
full-10s teacher scoring instead of 4-crop scoring.
```

Synthetic filtering summary:

| class | candidates | kept | high | hard | kept target-conf mean |
|---|---:|---:|---:|---:|---:|
| CD | 4000 | 400 | 400 | 0 | 0.9071 |
| HYP | 4000 | 400 | 400 | 0 | 0.7308 |
| MI | 4000 | 400 | 400 | 0 | 0.8308 |
| NORM | 4000 | 400 | 400 | 0 | 0.9908 |
| STTC | 4000 | 400 | 400 | 0 | 0.9593 |

Overall kept:

```text
2000 / 20000 candidates
high = 2000
hard = 0
mean target confidence = 0.8837
median target confidence = 0.9054
```

Run output:

```text
/root/autodl-tmp/graduate_project/self_distill_v2_e4_real2000_ens3_filtered2000_gamma03_scratch_seed42
```

Metrics:

| run | custom seed42 AUROC | custom seed42 AUPRC | official fold10 AUROC | official fold10 AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| real2000 seed42 teacher | 0.8433 | 0.6234 | 0.8297 | 0.6190 | 0.7426 | 0.4123 |
| real2000 seed43 teacher | 0.8491 | 0.6379 | 0.8413 | 0.6347 | 0.7557 | 0.4400 |
| real2000 seed44 teacher | 0.8506 | 0.6469 | 0.8417 | 0.6482 | 0.7594 | 0.4339 |
| v2 E4 scratch student | 0.8500 | 0.6371 | 0.8405 | 0.6380 | 0.7616 | 0.4546 |
| previous C3 strict | 0.8654 | 0.6738 | 0.7510 | 0.5118 | 0.7381 | 0.4230 |
| previous D1 C3-soft+synth20k | 0.8670 | 0.6737 | 0.7631 | 0.5273 | 0.7407 | 0.4280 |

Interpretation:

```text
v2 E4 is not the best in custom seed42 PTB-XL AUPRC; the strongest custom
result remains the previous C3/D1 line.

v2 E4 is the best among the tested self-distillation/real2000-teacher variants
for PN2021 external generalization, improving PN2021 avg AUPRC over:
  real2000 seed42 by +0.0423
  real2000 seed43 by +0.0146
  real2000 seed44 by +0.0207
  previous D1 by +0.0267

This suggests v2's filtered soft-label synthetic samples are useful mainly as
cross-center regularization. They do not yet improve the in-protocol PTB-XL
custom split beyond the best teacher.
```

Recommended next step:

```text
Run v2 E4 with C0 synthetic-pretrained initialization, while keeping the same
filtered synth2000 soft labels. This tests whether v2 filtering can combine
with the previous C0->student route and recover the C3/D1 custom AUPRC while
preserving the new PN2021 gain.
```
