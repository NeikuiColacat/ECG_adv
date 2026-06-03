# VAE Online AT Refinement Goal Plan

Date: 2026-06-01

## Goal

探索并固化一套论文级 VAE latent online adversarial training 方法，使其在
matched direct K500 fine-tune 之上，尽量达到：

```text
VAE-online-AT - direct K500 fine-tune >= +2.0pp macro AUROC
or
VAE-online-AT - direct K500 fine-tune >= +2.0pp macro AUPRC

and the other metric must be non-negative.
```

User-confirmed success criterion, 2026-06-01:

```text
探索阶段和 goal 自动执行阶段采用：
AUROC 或 AUPRC 任意一个达到 +2pp，另一个不能下降。

最终论文强 claim 仍优先争取 AUROC 和 AUPRC 同时提升；
如果只满足单指标 +2pp，则必须在报告中说明另一个指标无下降。
```

目标 backbone：

```text
1. EfficientNet1DV2 500Hz
2. ECGFounder 500Hz
3. model/ecg_ptbxl_benchmarking 500Hz backbones
   - fastai_inception1d
   - fastai_resnet1d_wang
   - fastai_xresnet1d50
   - fastai_fcn_wang
   - fastai_schirrmeister
```

主评估中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

固定标签和评估协议：

```text
mapping: v7_super5_sjr_rgq_review_20260528
class order: CD, HYP, MI, NORM, STTC
K target anchors: 500 per center
final eval: target-center K500 ref ids excluded
metrics: macro AUROC / macro AUPRC
views: all-zero-kept and drop-all-zero
```

## Current Evidence Snapshot

### 2026-06-02 Added Diagnostic Branches

User-approved exploration update:

```text
If 500Hz VAE online AT keeps failing to beat matched direct FT, explicitly
test whether the bottleneck is the attack objective or the 500Hz VAE manifold.
```

Two branches are now part of the automatic execution plan:

| branch | purpose | first gate |
|---|---|---|
| multilabel positive-hide / negative-add attack objective | separate "hide true positive labels" from "add false positive labels" instead of relying only on full BCE | CPSC K500 pilot must beat direct40 or current hard loss-gain |
| ECGTwin-original VAE initialized 500Hz VAE | test whether current native DiffuSETSVAE500 manifold is weaker than the author's pretrained ECGTwin VAE family | reconstruction audit and CPSC/Ningbo smoke before four-center expansion |
| external 500Hz ECG VAE audit | check whether a public 500Hz / 10s / 12-lead decoder-capable ECG VAE can replace or initialize VAE500 | only DiffuSETS currently looks immediately auditable; all others require checkpoint confirmation or retraining |

### 500Hz VAE Bottleneck Branch

This branch is allowed because current 500Hz VAE-online-AT is mostly neutral
against matched direct fine-tuning. It must remain a gated branch, not an
automatic replacement of the active VAE.

Local ECGTwin-author options:

| option | role | current decision |
|---|---|---|
| ECGTwin-init VAE500 | strict-load original ECGTwin VAE encoder/decoder weights into the 5000-sample convolutional VAE | technically valid, but current reconstruction and downstream pilots do not beat spectral-r001 |
| ECGTwin-teacher DiffuSETSVAE500 | train the current VAE500 with frozen original ECGTwin VAE reconstruction/latent distillation at 1024 samples | gated candidate only; slight reconstruction gain has not translated to downstream gain |
| active spectral-r001 DiffuSETSVAE500 | current best owned PTB-XL-only VAE500 | keep as default unless a new VAE passes both reconstruction and downstream gates |

Public/external options:

| option | role | current decision |
|---|---|---|
| DiffuSETS / ECGTwin public family | closest released decoder-capable ECG VAE family | useful A/B and teacher reference |
| SE-Diff | ICLR 2026 ECG latent diffusion with simulator/experience constraints | borrow loss/constraint ideas; do not use as drop-in unless prerequisite weights are confirmed |
| ECGEN | open-source 12-lead 5000-sample VAE-style framework | possible native-5000 engineering baseline; requires self-training and integration |

Mandatory stop conditions before any four-center expansion:

```text
Gate 1: reconstruction
  finite decode rate = 100%
  Pearson >= active spectral-r001 reference
  MSE/MAE not worse than active spectral-r001 by more than 5%
  lead residual not worse than active spectral-r001 in a clinically obvious way

Gate 2: latent-hull diagnostics
  decoded_invalid_rate near 0
  loss_gain positive
  atk_init roughly 0.3-0.7
  anchor-local same-label interpolation does not create obvious waveform failure

Gate 3: downstream two-center smoke
  centers: cpsc_2018 and ningbo
  matched direct control: direct40 K500 fine-tune
  active VAE control: spectral-r001 + loss-gain policy
  continue only if CPSC improves and Ningbo does not regress in AUPRC
```

If a candidate VAE only improves reconstruction but fails Gate 3, treat that as
evidence that the bottleneck is online-AT objective/selection rather than VAE
pixel-level reconstruction.

Implementation status:

```text
adversarial/pgd_advdiff.py
  adds attack_loss_mode:
    bce                # historical behavior, default
    positive_hide      # maximize BCE only on true positive labels
    negative_add       # maximize BCE only on true negative labels
    pos_hide_neg_add   # weighted combination

adversarial/latent_hull_pgd.py
  reuses the same attack objective for latent-hull coefficient optimization.

scripts/pgd_cross_center/synth_online_at_super5.py
  exposes:
    --attack_loss_mode
    --attack_pos_hide_weight
    --attack_neg_add_weight
    --attack_negative_exclude_classes
    --attack_neg_topk
    --adv_accept_gain_mode bce|target
```

Default behavior remains the old full-BCE attack, so historical hard_lg and
AugMix runs remain comparable.

First CPSC pilot result:

```text
run:
  /root/autodl-tmp/vae500_poshide_k500_v7_20260602/cpsc_2018_poshide_negadd_ep10_seed42

config:
  spectral-r001 VAE500
  attack_loss_mode=pos_hide_neg_add
  attack_negative_exclude_classes=NORM
  adv_accept_gain_mode=target
  adv_label_mode=multi_hot_hard
  otherwise close to the old hard loss-gain CPSC recipe

training:
  best internal K500-val epoch=9, 0.9133 / 0.8295
  stopped at epoch 10 because ASR < 0.30 for three consecutive epochs

heldout CPSC ref-excluded:
  direct40       0.8633 / 0.6155
  hard loss-gain 0.8688 / 0.6270
  poshide+negadd 0.8675 / 0.6234
```

Decision:

```text
The explicit positive-hide / negative-add objective is technically valid and
beats direct40 on CPSC, but it is weaker than current hard loss-gain. Do not
expand this exact recipe to four centers. Keep the code because it improves
diagnostics and enables future targeted ASR experiments.
```

### 2026-06-01 Strict Matched Evidence Update

EfficientNet1DV2 500Hz matched direct FT controls have been rerun with the same
K500 pools, K500-internal validation split, final K500-ref-excluded evaluation,
and v7 500Hz preprocessing protocol.

Strict matched result:

| method | four-center mean AUROC | four-center mean AUPRC | delta vs direct |
|---|---:|---:|---:|
| direct K500 fine-tune | 0.8601 | 0.5416 | reference |
| plain VAE500 online AT | 0.8578 | 0.5378 | -0.23pp / -0.38pp |

Interpretation:

```text
Plain VAE500 online AT beats source-only PTB-XL baseline, but it does not beat
matched direct K500 fine-tune. The paper-safe target must therefore be measured
against direct K500, not source-only.
```

AugMix + VAE online AT screening:

| stage | centers | outcome |
|---|---|---|
| Stage A severity sweep | cpsc_2018, chapman_shaoxing | did not beat direct FT |
| Stage B low-pressure + soft-label sweep | cpsc_2018, chapman_shaoxing | CPSC below direct; Chapman only +0.10pp / +0.05pp best |

Stage B best concrete results:

| center | best Stage B method | AUROC / AUPRC | delta vs direct |
|---|---|---:|---:|
| cpsc_2018 | hard/latentmix/teacher all similar | 0.8499 / 0.5952 | -0.34pp / -0.60pp |
| chapman_shaoxing | teacher_soft | 0.8735 / 0.4453 | +0.10pp / +0.05pp |

Decision:

```text
Do not keep scaling generic AugMix severity or simple hard/soft label modes as
the main route. Move to boundary/uncertainty-targeted VAE AT, where adversarial
latents are generated only from direct-FT weak or low-margin target anchors.
```

Boundary/uncertainty-targeted Stage C/D update:

```text
Implemented anchor scoring in scripts/pgd_cross_center/synth_online_at_super5.py.
Scoring source: matched direct FT checkpoint on the K500 train split only.
Score: positive hard BCE + 0.5 * positive low-margin uncertainty.
Policy: keep each class's top 45% high-score anchors for latent-hull sampling.
```

Stage C/D results:

| stage | config | cpsc_2018 | chapman_shaoxing | outcome |
|---|---|---:|---:|---|
| C | direct-init, hard labels, adv_weight=0.20, target_real_weight=5 | 0.8533 / 0.6011 | 0.8725 / 0.4448 |回退 direct FT |
| D | direct-init, teacher_soft, adv_weight=0.05, target_real_weight=0 | 0.8533 / 0.6011 | 0.8725 / 0.4448 |回退 direct FT |

Interpretation:

```text
Boundary anchor selection worked mechanically and attack ASR was healthy, but
K500 internal validation dropped. This suggests the current EfficientNet1DV2
K500 direct-init VAE stream is not adding useful information beyond direct FT.
Do not keep making small hard/soft-label or adv-weight tweaks on the same setup.
Prioritize K/target-ratio and backbone-transfer experiments next.
```

Reports:

```text
docs/tmp_md/vae_online_at_refine_execution_audit_20260601.md
docs/tmp_md/vae_online_at_refine_augmix_stageA_20260601.md
docs/tmp_md/vae_online_at_refine_augmix_stageB_20260601.md
docs/tmp_md/vae_online_at_refine_boundary_stageC_D_20260601.md
docs/tmp_md/vae500_k1000_direct_vs_lhat_20260601.md
```

K1000 fixed-K update:

```text
K1000 anchors were exported for ningbo, chapman_shaoxing, cpsc_2018, georgia.
Matched K1000 direct FT and matched K1000 plain VAE500 LHAT were completed.
```

| center | K1000 direct | K1000 VAE LHAT | delta VAE - direct |
|---|---:|---:|---:|
| ningbo | 0.8849 / 0.5053 | 0.8872 / 0.5104 | +0.23pp / +0.51pp |
| chapman_shaoxing | 0.8758 / 0.4476 | 0.8748 / 0.4489 | -0.10pp / +0.13pp |
| cpsc_2018 | 0.8770 / 0.6328 | 0.8790 / 0.6364 | +0.19pp / +0.37pp |
| georgia | 0.8396 / 0.6215 | 0.8377 / 0.6179 | -0.19pp / -0.36pp |
| mean | 0.8693 / 0.5518 | 0.8697 / 0.5534 | +0.03pp / +0.16pp |

Interpretation:

```text
K1000 plain VAE is more stable than direct-init boundary C/D, but it is still far
from the +2pp target. K1000 direct FT itself improves over K500 direct by about
+0.93pp AUROC / +1.02pp AUPRC, mainly due to CPSC.
```

Target-ratio sizes based on nonzero Super5 records:

| center | nonzero Super5 records | 10% K | 20% K |
|---|---:|---:|---:|
| ningbo | 19228 | 1923 | 3846 |
| chapman_shaoxing | 5822 | 582 | 1164 |
| cpsc_2018 | 4746 | 475 | 949 |
| georgia | 8711 | 871 | 1742 |

Next ratio decision:

```text
Run 10% target-ratio first. It is the cleanest next experiment because it
normalizes adaptation sample count by center size while staying close to the
already-run K500/K1000 compute envelope for three of four centers.
```

10% target-ratio update:

```text
Matched 10% target-ratio direct FT and matched 10% plain VAE500 LHAT were
completed for ningbo, chapman_shaoxing, cpsc_2018, and georgia.
```

| center | K | 10% direct | 10% VAE LHAT | delta VAE - direct |
|---|---:|---:|---:|---:|
| ningbo | 1923 | 0.8912 / 0.5100 | 0.8911 / 0.5103 | -0.01pp / +0.03pp |
| chapman_shaoxing | 582 | 0.8740 / 0.4509 | 0.8745 / 0.4509 | +0.05pp / +0.00pp |
| cpsc_2018 | 475 | 0.8610 / 0.6068 | 0.8607 / 0.6084 | -0.03pp / +0.16pp |
| georgia | 871 | 0.8362 / 0.6190 | 0.8383 / 0.6235 | +0.21pp / +0.45pp |
| mean | - | 0.8656 / 0.5467 | 0.8662 / 0.5483 | +0.06pp / +0.16pp |

Interpretation:

```text
10% target-ratio does not meet the goal criterion. The current plain VAE500
latent-hull branch is safe and source performance is preserved, but it only
matches direct FT. All four centers show K-internal validation improvement;
that improvement does not meaningfully transfer to the held-out ref-excluded
target evaluation. Scaling K/ratio alone is therefore unlikely to be the
solution.
```

Report:

```text
docs/tmp_md/vae500_ratio10_direct_vs_lhat_20260601.md
```

20% target-ratio update:

```text
Matched 20% target-ratio direct FT has completed for all four centers.
Matched 20% VAE500 LHAT has completed for all four centers.
```

| center | K | 20% direct | 20% VAE LHAT | delta VAE - direct |
|---|---:|---:|---:|---:|
| ningbo | 3846 | 0.8952 / 0.5128 | 0.8949 / 0.5124 | -0.03pp / -0.04pp |
| chapman_shaoxing | 1164 | 0.8735 / 0.4436 | 0.8721 / 0.4414 | -0.14pp / -0.22pp |
| cpsc_2018 | 949 | 0.8796 / 0.6430 | 0.8785 / 0.6399 | -0.11pp / -0.31pp |
| georgia | 1742 | 0.8451 / 0.6302 | 0.8446 / 0.6293 | -0.05pp / -0.09pp |
| mean | - | 0.8733 / 0.5574 | 0.8726 / 0.5557 | -0.08pp / -0.17pp |

Interpretation:

```text
20% target-ratio direct FT is stronger than K1000 direct on average, but the
gain is uneven and mostly comes from cpsc_2018 and georgia. This raises the
matched baseline that VAE AT must beat.

The current plain VAE500 LHAT recipe does not beat matched direct FT at 20%.
It is slightly below direct on all four centers, with mean delta
-0.08pp AUROC / -0.17pp AUPRC. This exact recipe should not be transferred to
ECGFounder or other benchmark backbones as a main experiment before an
EfficientNet1DV2 version shows a stable positive delta over matched direct.
```

Report:

```text
docs/tmp_md/vae500_ratio20_direct_vs_lhat_20260601.md
```

Next EfficientNet-first decision:

```text
Do not expand the plain ratio20 VAE LHAT recipe to ECGFounder or
model/ecg_ptbxl_benchmarking backbones. It failed the EfficientNet1DV2 matched
direct gate at K500, K1000, 10%, and 20%.

The next useful EfficientNet diagnostic should change the adaptation mechanism,
not merely scale K or transfer backbone. Candidate:
  classifier/adapter-limited direct FT vs classifier/adapter-limited VAE LHAT
  with optional source-logit anchoring.

Rationale:
  full EfficientNet direct FT already absorbs most target-center signal.
  A low-capacity adapter or classifier-only setting tests whether VAE latent
  samples provide useful regularization when supervised target fitting capacity
  is constrained. It must still be compared against its own matched direct
  control before any backbone transfer.
```

Low-capacity LoRA diagnostic update:

```text
freeze EfficientNet1DV2 backbone
train only final classifier LoRA residual adapter
direct-LoRA vs VAE500 LoRA-LHAT
same K500 pools, same v7 500Hz ref-excluded evaluation
```

| center | direct-LoRA | VAE500 LoRA-LHAT | delta |
|---|---:|---:|---:|
| ningbo | 0.8591 / 0.4644 | 0.8589 / 0.4643 | -0.01pp / -0.02pp |
| chapman_shaoxing | 0.8613 / 0.4193 | 0.8612 / 0.4192 | -0.01pp / -0.01pp |
| cpsc_2018 | 0.8212 / 0.5537 | 0.8221 / 0.5541 | +0.09pp / +0.04pp |
| georgia | 0.8185 / 0.5965 | 0.8185 / 0.5965 | +0.00pp / -0.00pp |
| mean | 0.8400 / 0.5085 | 0.8402 / 0.5085 | +0.02pp / +0.00pp |

Interpretation:

```text
LoRA-limited capacity does not unlock VAE500 gain. This weakens the hypothesis
that the only blocker is full direct FT capacity. Current VAE500 LHAT is valid
but nearly neutral. Training diagnostics show decoded_invalid_rate=0 while
loss_gain is only about 3e-4 to 5e-4, so the latent-hull branch is not applying
meaningful adversarial pressure to the classifier.
```

500Hz old-recipe K500 update:

```text
Run root:
  /root/autodl-tmp/vae500_oldrecipe_k500_v7_20260601

Purpose:
  test whether VAE500 failed because the newer K500 recipe was too conservative.

Recipe:
  classes_in_scope=NORM,MI,STTC
  hull_M=10
  hull_lambda=0.15
  hull_steps=5
  K_anchor=300
  adv_label_mode=mixed_soft
  adv_teacher_mix=0.3
  target_real_weight=40
  adv_weight=0.06
  quality gate enabled
```

| center | direct K500 FT | old-recipe VAE500 | old - direct |
|---|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8790 / 0.4964 | -0.07pp / -0.59pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8749 / 0.4507 | +0.24pp / +0.59pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8671 / 0.6221 | +1.38pp / +2.09pp |
| georgia | 0.8347 / 0.6180 | 0.8360 / 0.6192 | +0.14pp / +0.12pp |
| mean | 0.8601 / 0.5416 | 0.8643 / 0.5471 | +0.42pp / +0.56pp |

Interpretation:

```text
Old-recipe VAE500 is clearly better than the plain K500 VAE500 recipe and gives
a real single-center win on CPSC, but it still misses the four-center global
goal. It also exposes a v7 label-scope mismatch: MI is almost absent in K500,
while CD/HYP are abundant but excluded by the old trusted scope.

Next global recipe to test:
  old-recipe pressure with v7-aware scope CD,HYP,NORM,STTC and
  --allow_hyp_cd_trust.
```

Report:

```text
docs/tmp_md/vae500_oldrecipe_k500_v7_20260601.md
```

V7-aware scope update:

```text
Run root:
  /root/autodl-tmp/vae500_oldrecipe_v7scope_k500_v7_20260601

Recipe:
  old-recipe pressure, but classes_in_scope=CD,HYP,NORM,STTC and
  --allow_hyp_cd_trust
```

| center | direct K500 FT | old NORM/MI/STTC | v7-scope CD/HYP/NORM/STTC | v7-scope - direct |
|---|---:|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8790 / 0.4964 | 0.8784 / 0.4955 | -0.13pp / -0.67pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8749 / 0.4507 | 0.8739 / 0.4511 | +0.14pp / +0.63pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8671 / 0.6221 | 0.8568 / 0.6017 | +0.35pp / +0.06pp |
| georgia | 0.8347 / 0.6180 | 0.8360 / 0.6192 | 0.8358 / 0.6191 | +0.11pp / +0.11pp |
| mean | 0.8601 / 0.5416 | 0.8643 / 0.5471 | 0.8612 / 0.5419 | +0.12pp / +0.03pp |

Interpretation:

```text
V7-aware CD/HYP inclusion is not the fix. It increases attack activity but does
not improve heldout generalization. CPSC loses the strong STTC-centered gain
seen with old NORM/MI/STTC scope.

Next direction should be class-aware selection, not a single center-global
scope. Evaluate paper-safe classwise selection over direct/plain/old/v7scope
using only K500-internal validation and a PTB-XL source floor.
```

Report:

```text
docs/tmp_md/vae500_oldrecipe_v7scope_k500_v7_20260601.md
```

K500-internal classwise selector update:

```text
Candidate pool:
  direct K500 FT
  plain VAE500 LHAT
  old NORM/MI/STTC VAE500
  v7-scope CD/HYP/NORM/STTC VAE500

Selection rule:
  for each center and each Super5 class, choose candidate by K500-internal
  validation AUPRC, breaking ties by AUROC. Then evaluate once on PN2021
  ref-excluded heldout.
```

| center | direct | selector | selector - direct |
|---|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8802 / 0.5014 | +0.04pp / -0.09pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8751 / 0.4494 | +0.26pp / +0.46pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8655 / 0.6178 | +1.22pp / +1.67pp |
| georgia | 0.8347 / 0.6180 | 0.8366 / 0.6209 | +0.19pp / +0.29pp |
| mean | 0.8601 / 0.5416 | 0.8643 / 0.5474 | +0.43pp / +0.58pp |

Interpretation:

```text
Classwise selection prevents some VAE harm and preserves part of the CPSC gain,
but it still does not approach the +2pp global target. The current candidate
pool is not strong enough; selection alone is not the missing mechanism.
```

Report:

```text
docs/tmp_md/vae500_candidate_selector_k500_v7_20260601.md
```

Matched direct40/ep10 control update:

```text
Run root:
  /root/autodl-tmp/vae500_direct40_ep10_v7_20260601

Purpose:
  separate VAE contribution from the heavier supervised target-real stream used
  by old-recipe runs.

Control:
  disable_adv_stream
  target_real_weight=40
  n_epochs=10
  same K500 anchors and validation split
```

| center | direct20/ep30 | direct40/ep10 | old-recipe VAE500 | old - direct40 |
|---|---:|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8810 / 0.5032 | 0.8790 / 0.4964 | -0.20pp / -0.68pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8757 / 0.4508 | 0.8749 / 0.4507 | -0.09pp / -0.01pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8633 / 0.6155 | 0.8671 / 0.6221 | +0.38pp / +0.66pp |
| georgia | 0.8347 / 0.6180 | 0.8364 / 0.6202 | 0.8360 / 0.6192 | -0.03pp / -0.09pp |
| mean | 0.8601 / 0.5416 | 0.8641 / 0.5474 | 0.8643 / 0.5471 | +0.02pp / -0.03pp |

Interpretation:

```text
This control largely explains the old-recipe gain. The VAE branch adds almost
nothing over a better matched target-real direct control. Future refinements
must compare against direct40/ep10 or another exactly matched direct recipe,
not only the older direct20/ep30 baseline.
```

Old 100Hz/v6 success route status:

```text
The 2026-05-25 EfficientNet result
  compatsoft / direct-init / latent_mixed_teacher
  mean 0.872177 / 0.514767
  gain +1.79pp AUROC / +2.68pp AUPRC over direct K500
was a real result under the older 100Hz/v6 protocol.

It should not be imported as evidence that the current 500Hz/v7 VAE AT already
beats direct fine-tuning. The closest 500Hz/v7 old-recipe reproduction gained
against direct20/ep30 but not against the matched direct40/ep10 control.

For current paper claims, every VAE-online-AT result must be compared against
the matched direct40 or an equivalently matched direct target fine-tune control.
```

Report:

```text
docs/tmp_md/vae500_direct40_control_20260601.md
```

Direct40-init pure VAE pilot:

```text
Run root:
  /root/autodl-tmp/vae500_direct40init_purevae_pilot_v7_20260601

Purpose:
  remove supervised target-real confounding by starting from direct40 and
  setting target_real_weight=0. Keep PTB-XL source floor plus VAE adversarial
  stream only.
```

| center | direct40/ep10 | direct40-init pure VAE | delta |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8633 / 0.6155 | +0.00pp / +0.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8810 / 0.5032 | +0.00pp / +0.00pp |

Interpretation:

```text
Pure VAE refinement from a strong direct checkpoint is rejected by K500-internal
validation; best checkpoints stay at the direct40 initialization. This confirms
that the current VAE loss/selection design does not add useful signal once the
target-real direct baseline is properly matched.
```

Report:

```text
docs/tmp_md/vae500_direct40init_purevae_pilot_20260601.md
```

Direct40-init teacher-soft VAE pilot:

```text
Run root:
  /root/autodl-tmp/vae500_direct40init_teachersoft_pilot_v7_20260602

Purpose:
  test whether the direct40-init VAE branch fails because hard/mixed labels are
  too destructive. Start from direct40, set target_real_weight=0, and use
  adv_label_mode=teacher_soft so adversarial decoded samples preserve the
  frozen direct40 teacher probabilities.
```

| center | direct40/ep10 | direct40-init teacher-soft VAE | delta |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8633 / 0.6155 | +0.00pp / +0.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8810 / 0.5032 | +0.00pp / +0.00pp |

Training behavior:

| center | direct40 K500-val baseline | best post-update K500-val | last K500-val | ASR mean | loss_gain mean |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.9128 / 0.8289 | 0.9043 / 0.7765 | 0.8890 / 0.7393 | 0.4126 | -0.0301 |
| ningbo | 0.8748 / 0.8012 | 0.8677 / 0.7857 | 0.8622 / 0.7703 | 0.3833 | 0.0465 |

Interpretation:

```text
Teacher-soft labels do not fix the 500Hz VAE issue. The attack is active and
decoded signals are valid, but K500-internal validation still rejects all VAE
updates and selects epoch 0. This suggests the current VAE500 latent geometry or
training objective is not adding useful held-out generalization beyond matched
direct40 fine-tune.
```

Report:

```text
docs/tmp_md/vae500_direct40init_teachersoft_pilot_20260602.md
```

Next diagnostic after this negative result:

```text
1. Cheap diagnostic:
   use the original ECGTwin VAE latent manifold with the 500Hz classifier by
   downsampling target ECGs to ECGTwin 1024 format, doing latent-hull AT there,
   decoding, then resampling decoded ECG back to 5000 samples for the 500Hz
   EfficientNet. This tests whether the original ECGTwin latent geometry is
   more useful than the repo-owned VAE500 latent.

2. If the diagnostic is positive:
   train a teacher-distilled VAE500 instead of directly continuing current
   VAE500 sweeps. The target design is:
     500Hz reconstruction
     + downsampled ECGTwin reconstruction consistency
     + ECGTwin latent / feature consistency
     + optional high-frequency residual branch.

3. Do not directly load the original ECGTwin VAE checkpoint into VAE500 without
   architecture changes. The original VAE uses 1024x12 input/output and latent
   (4,128); VAE500 uses 5000x12 and latent (4,625).
```

### 2026-06-02 VAE500 Replacement / Distillation Direction

Current working hypothesis:

```text
VAE500 reconstructs ECG very well, but its latent geometry may be too
waveform-detail-oriented and not semantic enough for latent-hull online AT.
This can explain why decoded samples are valid and attacks are active, while
matched direct40 K500 validation rejects the VAE updates.
```

Local ECGTwin VAE audit:

```text
Original ECGTwin VAE:
  input/output: 1024 x 12
  latent:       (4, 128)
  checkpoint:   {'encoder', 'decoder'}
  width:        128 / 256 / 512 style blocks

Current repo VAE500:
  input/output: 5000 x 12
  latent:       (4, 625)
  checkpoint:   model_state_dict
  width:        default 64 / 128 / 256 style blocks

Code audit refinement:
  original ECGTwin VAE has no learned positional embedding tied to length;
  convolution kernels and residual/attention weights are length-independent.
  If the original ECGTwin VAE architecture itself is reused with input length
  5000, the same three stride-2 stages naturally produce latent (4,625), and
  the original encoder/decoder checkpoint can be loaded into the same module
  family. The risk is sampling-rate semantics and bottleneck attention cost,
  not state-dict shape.
```

Decision:

```text
Do not directly load ECGTwin's original VAE checkpoint into the current
DiffuSETSVAE500 implementation. The checkpoint keys, channel widths, and layer
layout do not match. Shape-sliced partial init into DiffuSETSVAE500 is possible
but too brittle for the next mainline.

Do allow a separate backend:
  ecgtwin_init_vae500
which reuses ECGTwin's original VAE_Encoder/VAE_Decoder architecture at length
5000 and strictly loads the original ECGTwin encoder/decoder weights before
PTB-XL-only fine-tuning.
```

Recommended implementation route:

```text
Route A: ECGTwin-init VAE500.

  1. Add ECGTwinVAE500 wrapper around original ECGTwin VAE_Encoder/VAE_Decoder.
  2. Feed PTB-XL records500 as 5000 x 12.
  3. Strictly load model/ECGTwin/checkpoints/vae_model.pth {'encoder','decoder'}.
  4. Fine-tune on PTB-XL only with low LR and KL warmup.
  5. Save as a new backend, never overwriting diffusets500_v1.

Route B: frozen ECGTwin VAE teacher -> teacher-distilled VAE500 student.

For every PTB-XL records500 sample:
  x500_ptbxl:        (B, 5000, 12), PTB-XL lead order
  x1024_ecgtwin:     lead-reordered + resampled teacher view
  teacher z_t:       ECGTwin encoder(x1024), shape (B, 4, 128)
  teacher recon_t:   ECGTwin decoder(z_t), shape (B, 1024, 12)
  student z_s:       VAE500 encoder(x500), shape (B, 4, 625)
  student recon500:  VAE500 decoder(z_s), shape (B, 5000, 12)
  z_s_proj:          AdaptiveAvgPool1d(128) + 1x1 Conv, shape (B, 4, 128)

Loss:
  existing VAE500 reconstruction / KL / first-diff / lead-consistency
  + latent_distill_weight * SmoothL1(z_s_proj, z_t)
  + teacher_recon_weight * SmoothL1(resample(recon500 -> 1024), recon_t)

Initial weights:
  latent_distill_weight: 0.05, 0.10, 0.20 sweep
  teacher_recon_weight: 0.05
```

Route A/B selection:

```text
Try Route A first because it uses the original ECGTwin VAE module family and
therefore gives the cleanest use of the author's weights. If Route A is too
slow due to 625-token bottleneck attention or reconstructs poorly after a short
pilot, fall back to Route B teacher distillation into the lighter current
DiffuSETSVAE500 student.
```

Cheap diagnostic before full distillation:

```text
Use original ECGTwin VAE as-is with the 500Hz classifier:
  1. export K500 target anchors through original ECGTwin VAE at 1024 points;
  2. latent-hull decode at 1024;
  3. resample decoded signals to 5000 points;
  4. run direct40-init VAE branch against matched direct40.

If this beats VAE500 or preserves K500 validation better, then the bottleneck is
VAE500 latent geometry. If it also fails, the issue is more likely direct40
dominance / loss design rather than the VAE500 architecture alone.
```

Cheap diagnostic result, 2026-06-02:

```text
Implemented and ran:
  original ECGTwin VAE latent (4,128)
  -> decode 1024 points
  -> ECGTwin-to-PTBXL lead reorder
  -> interpolate to 5000 points
  -> 500Hz EfficientNet1DV2 direct40-init online AT

Anchor bundles:
  /root/autodl-tmp/ecgtwin1024_to5000_anchors_v7_20260602

Runs:
  /root/autodl-tmp/ecgtwin1024_to5000_direct40init_pilot_v7_20260602
```

| center | direct40 K500-val baseline | best post-update K500-val | last K500-val | selected |
|---|---:|---:|---:|---|
| cpsc_2018 | 0.9158 / 0.8556 | 0.8956 / 0.8037 | 0.8784 / 0.7782 | epoch 0 |
| ningbo | 0.8969 / 0.8526 | 0.8882 / 0.8438 | 0.8828 / 0.8366 | epoch 0 |

Diagnostics:

| center | ASR mean | atk_anchor mean | loss_gain mean | decoded invalid |
|---|---:|---:|---:|---:|
| cpsc_2018 | 0.4127 | 0.5188 | 0.0405 | 0.0000 |
| ningbo | 0.3523 | 0.4536 | 0.0019 | 0.0000 |

Interpretation:

```text
Original ECGTwin VAE latent geometry does not fix the 500Hz direct40-init
failure under the current online AT loss. The attack is active and decoded
signals are finite, but K500-internal validation rejects all post-update
checkpoints. The best saved weights are byte-identical to direct40 init.

Therefore do not immediately spend a long run on naive ECGTwin-weight VAE500
fine-tuning. If VAE500 is improved, do it as teacher-distilled VAE500 and pair
it with a changed online AT objective/selection rule.
```

Report:

```text
docs/tmp_md/ecgtwin1024_to5000_vae_diagnostic_20260602.md
```

External 500Hz ECG latent model candidates from web research:

| priority | candidate | role | decision |
|---|---|---|---|
| A | SE-Diff | 12-lead, 10s, 500Hz VAE + latent diffusion; reported VAE latent 4 x 128 | highest-priority architecture reference; verify whether pretrained VAE weights are actually downloadable before any drop-in evaluation |
| A- | DiffuSETS / ECGTwin | current closest available family | keep as main owned baseline; improve with better latent geometry constraints before replacing it |
| B | ECGEN VAE | 12-lead, 500Hz, 5000-sample convolutional VAE candidate | useful lightweight engineering baseline; do not use MIMIC-pretrained weights in the PTB-XL-only main claim |
| C+ | D-BETA / C-MELT / ECG-FM / ST-MEM | representation or MAE teachers | possible feature-consistency teachers; not decoder replacements for latent-hull online AT |
| C | WearECG / MCMA | 500Hz reconstruction / lead-completion models | useful reconstruction references, but not full 12-lead VAE decoders for this method |

Current external-model conclusion:

```text
No fully confirmed drop-in 12-lead / 10s / 500Hz / decoder-capable VAE with
publicly usable pretrained weights has been accepted yet.

SE-Diff is the strongest external reference because its VAE is already designed
for 12-lead 10s 500Hz ECG and uses a compact 4 x 128 latent. If weights are
available and license-compatible, evaluate it as an external VAE baseline. If
weights are unavailable, borrow its training ideas: beat/morphology decoder,
spectral loss, and inter-lead physiological constraints.

ECGEN is the cleanest lightweight 500Hz VAE code candidate, but current public
evidence is weaker than SE-Diff/DiffuSETS. Treat it as a baseline or teacher,
not as the main replacement unless it passes the same PTB-XL-only VAE gates and
downstream ref-excluded tests.
```

External source links checked:

```text
SE-Diff paper:  https://openreview.net/pdf?id=95ZV35sBDm
SE-Diff code:   https://github.com/ignite-abd/SE-Diff
DiffuSETS code: https://github.com/Raiiyf/DiffuSETS_Exp
DiffuSETS data: https://zenodo.org/records/15420698
ECGEN code:     https://github.com/vlbthambawita/ECGEN
```

SE-Diff local clone audit, 2026-06-02:

```text
Local path: model/SE-Diff
Git SHA:    dd0a476
Size:       4.8M

Findings:
  - repository contains code only; no pretrained weights or download script;
  - README requires pretrained weights, latent dataset, mini decoder, and
    simulator prior under ./prerequisites/;
  - VAE_Encoder/VAE_Decoder are effectively the same ECGTwin-style convolutional
    VAE family, with latent channels=4 and length input_len/8;
  - new useful pieces are VAE_Decoder_FirstCycle, MiniDecoderMSE, and
    simulator/inter-lead losses used during diffusion training.

Implication:
  SE-Diff should not be treated as an immediately available replacement VAE
  unless weights are separately obtained. Its highest value for this project is
  as a design reference for PTB-XL-only VAE500 training:
    compact 4-channel latent,
    first-cycle / beat-level decoder pressure,
    simulator-prior morphology loss,
    inter-lead consistency loss.
```

Report:

```text
docs/tmp_md/vae500_external_vae_candidates_20260602.md
```

Next concrete tasks:

```text
1. Check whether SE-Diff pretrained VAE weights are actually obtainable. If yes,
   evaluate only as an external-pretraining baseline first.
2. Add SE-Diff-style constraints to the owned PTB-XL-only VAE500 route before
   another expensive online-AT sweep:
     spectral reconstruction loss,
     inter-lead/Einthoven consistency,
     optional beat/morphology decoder or QRS-aware auxiliary loss.
3. Keep ECGTwin-teacher and ECGTwin-init VAE500 as the cleanest PTB-XL-only
   adaptation route because the teacher comes from the original ECGTwin
   framework already used by the project.
4. Evaluate ECGEN only after the SE-Diff/ECGTwin-teacher gates, unless a clean
   pretrained VAE checkpoint is found.
5. Consider D-BETA/ECG-FM/ST-MEM only as feature teachers or ablations.
```

User-added exploration branch, 2026-06-02:

```text
The current 500Hz VAE may itself be the bottleneck. During goal execution, allow
a controlled branch that uses the original ECGTwin VAE as teacher/initializer to
train a stronger 500Hz VAE.

Allowed:
  - create an ECGTwin original-architecture 5000-point VAE backend and strictly
    initialize it from the author's encoder/decoder weights;
  - freeze original ECGTwin VAE and distill its latent/reconstruction geometry
    into a 5000-sample student VAE trained only on PTB-XL;
  - initialize compatible convolutional blocks from the original ECGTwin VAE
    only when tensor shapes match exactly, and log every skipped key;
  - add a small high-frequency residual decoder branch if reconstruction
    metrics show the distilled student is too smooth;
  - evaluate SE-Diff as the highest-priority external 500Hz VAE reference if
    pretrained VAE weights are accessible and license-compatible;
  - evaluate ECGEN as a secondary external 500Hz VAE candidate or teacher;
  - borrow SE-Diff-style spectral, beat/morphology, and inter-lead consistency
    losses for PTB-XL-only VAE500 training even if SE-Diff weights are not
    available.

Not allowed for the main PTB-XL-only claim:
  - importing MIMIC-pretrained SE-Diff/ECGEN/other weights as the main VAE
    without reporting it as external pretraining;
  - direct strict loading of ECGTwin weights into the existing DiffuSETSVAE500
    module, because that is a different architecture;
  - continuing expensive VAE-online-AT sweeps if the new VAE fails the
    reconstruction/latent-manifold gates below.
```

Current decision after local + subagent audit:

```text
Do keep the "stronger 500Hz VAE" branch alive, but run it as a gated
diagnostic branch, not as an automatic replacement for the current online-AT
mainline.

Reason:
  - ECGTwin original VAE checkpoint exists and can be strictly loaded into an
    ECGTwin-original 5000-point backend;
  - frozen ECGTwin-teacher distillation into DiffuSETSVAE500 is implemented;
  - both branches are technically valid;
  - existing pilots have not yet shown downstream improvement over the active
    spectral-r001 VAE500 under matched direct40 controls.

Therefore the next useful VAE architecture work is not "train longer". It is:
  1. improve the latent geometry objective;
  2. prove the candidate VAE passes reconstruction and latent-hull gates;
  3. prove it beats spectral-r001 on cpsc_2018 + ningbo before any four-center
     expansion.
```

Minimal gate for ECGTwin-init / ECGTwin-teacher VAE500:

```text
Gate 1: reconstruction and physiology audit on PTB-XL validation
  - finite decode rate = 100%;
  - invalid/flatline/severe amplitude rate near 0;
  - Pearson >= active DiffuSETSVAE500 or not meaningfully lower;
  - MSE/MAE not worse than active DiffuSETSVAE500 by >5%;
  - lead residual not worse than active DiffuSETSVAE500;
  - visual spot-check for NORM, MI, STTC, CD, HYP.

Gate 2: latent-hull plausibility
  - same-label interpolation decodes finite ECGs;
  - small latent perturbations do not cause flatline/exploding amplitude;
  - CPSC/Ningbo attack diagnostics remain healthy:
      atk_init roughly 0.3-0.7,
      decoded_invalid_rate near 0,
      loss_gain positive.

Gate 3: two-center downstream pilot
  - centers: cpsc_2018 and ningbo;
  - matched control: direct40 K500 fine-tune;
  - comparator VAE: active spectral-r001 VAE500 + hard loss-gain recipe;
  - continue only if CPSC AUPRC improves and Ningbo AUPRC is non-decreasing
    versus matched direct40 / spectral-r001.
```

External 500Hz VAE decision:

```text
No public drop-in 12-lead / 10s / 500Hz decoder-capable VAE checkpoint has
been accepted for the mainline yet.

SE-Diff is the best top-conference architecture reference, but the cloned repo
requires prerequisite weights/data and does not provide a ready VAE checkpoint.
Use it to borrow constraints such as beat/first-cycle decoder pressure,
simulator or morphology losses, and inter-lead consistency.

ECGEN is a plausible 12-lead 5000-sample VAE engineering baseline, but it is
secondary until a clean checkpoint and license path are confirmed.

Any external-pretrained VAE must be reported as external pretraining and cannot
be mixed into the PTB-XL-only causal claim.
```

VAE replacement gates before downstream online AT:

```text
Minimum reconstruction gate on PTB-XL validation:
  - finite decode rate = 100%;
  - no flatline or severe amplitude explosion;
  - mean Pearson correlation should beat the current VAE500 checkpoint;
  - MSE/MAE should not be worse than the current VAE500 checkpoint by >5%;
  - QRS/ST/T morphology should pass visual spot-checks for at least NORM, MI,
    STTC, CD, and HYP examples.

Minimum latent-manifold gate:
  - same-record encode-decode is stable under small latent perturbations;
  - same-label nearest-neighbor latent hull decodes remain physiologically
    plausible;
  - latent-hull online AT pilot on cpsc_2018 and ningbo must improve K500
    internal validation over direct40 epoch-0, or at least avoid the current
    "all updates rejected" failure mode.
```

ECGTwin-init VAE500 implementation update, 2026-06-02:

```text
Implemented:
  ecg_adv_gen/vae/ecgtwin_vae500.py
  variant: ecgtwin_init_vae500

Validated:
  strict loading of original ECGTwin vae_model.pth encoder/decoder;
  5000-point forward/backward;
  batch-size memsmoke up to b64 on 4090D.

Critical fix:
  the wrapper must convert PTB-XL lead order -> ECGTwin lead order before
  encoding, and ECGTwin -> PTB-XL after decoding. Without this, limb-lead
  residuals are very poor.
```

Best ECGTwin-init pilot so far:

| metric | ECGTwin-init VAE500 | current DiffuSETSVAE500 |
|---|---:|---:|
| invalid decode rate | 0.0000 | 0.0000 |
| audit global Pearson | 0.9964 | 0.9977 |
| audit first-diff Pearson | 0.9478 | 0.9505 |
| audit MSE | 0.00977 | 0.00710 |
| audit lead residual ratio | 1.1255 | 1.0880 |

Decision:

```text
ECGTwin-init VAE500 is technically feasible and passes the basic hard sanity
gate, but it does not yet beat the current DiffuSETSVAE500 reconstruction gate.
Do not use it as the downstream VAE replacement unless a planned objective
change closes the reconstruction gap.

Next VAE improvement should be teacher-distilled or multi-resolution, not more
plain continuation epochs only.
```

Report:

```text
docs/tmp_md/ecgtwin_init_vae500_pilot_20260602.md
```

ECGTwin-teacher VAE500 update, 2026-06-02:

```text
Implemented:
  frozen original ECGTwin VAE teacher at 1024 samples
  + DiffuSETSVAE500 student at 5000 samples
  + teacher reconstruction and latent distillation losses

Code:
  scripts/vae500/train_ptbxl_vae500.py

Best checkpoint:
  /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_teacher_r002_z001_b64_20260602/checkpoints/best.pt
```

| metric | current DiffuSETSVAE500 | ECGTwin-init VAE500 | ECGTwin-teacher DiffuSETSVAE500 |
|---|---:|---:|---:|
| invalid decode rate | 0.0000 | 0.0000 | 0.0000 |
| audit global Pearson | 0.9977 | 0.9964 | 0.9977 |
| audit first-diff Pearson | 0.9505 | 0.9478 | 0.9506 |
| audit MSE | 0.00710 | 0.00977 | 0.00694 |
| audit lead residual ratio | 1.0883 | 1.1255 | 1.1592 |

Downstream xresnet1d50 two-center pilot:

| center | direct K500 | old VAE500 online AT | ECGTwin-teacher VAE online AT |
|---|---:|---:|---:|
| ningbo | 0.8885 / 0.5241 | 0.8886 / 0.5234 | 0.8881 / 0.5249 |
| cpsc_2018 | 0.8529 / 0.5840 | 0.8539 / 0.5856 | 0.8538 / 0.5856 |

Decision:

```text
Teacher-distilled VAE500 slightly improves reconstruction MSE and Pearson, but
the downstream gain is negligible under the old online-AT recipe. Do not scale
this branch to all centers/backbones until the adversarial objective or model
selection rule changes. The bottleneck is no longer only VAE reconstruction
quality.
```

Report:

```text
docs/tmp_md/vae500_teacher_distill_pilot_20260602.md
```

K100 low-target diagnostic, 2026-06-02:

```text
Purpose:
  test whether VAE500 online AT helps when direct target fine-tune has only
  K=100 target-center anchors.

Important fix:
  K100 with 20% validation leaves only 20 validation samples. The online-AT
  script now exposes --quick_eval_min_pos, and these reruns used min_pos=2 to
  avoid falsely using only one scorable class in quick validation.
```

| center | direct K100 | old-recipe VAE500 K100 | delta VAE - direct |
|---|---:|---:|---:|
| cpsc_2018 | 0.8288 / 0.5751 | 0.8281 / 0.5757 | -0.07pp / +0.06pp |
| chapman_shaoxing | 0.8668 / 0.4364 | 0.8671 / 0.4373 | +0.03pp / +0.09pp |

Interpretation:

```text
K100 does not unlock a VAE advantage. The exact old-recipe VAE500 branch is
neutral versus matched direct fine-tune on both tested centers. Do not expand
this exact K100 setup to all centers unless the objective or VAE changes first.
```

Report:

```text
docs/tmp_md/vae500_lowk_k100_direct_vs_lhat_20260602.md
```

### 2026-06-02 Spectral-Loss VAE500 Pilot

Implemented optional frequency-domain reconstruction loss:

```text
spectral_logmag(x) = log(1 + abs(rfft(x, dim=time)))
loss += spectral_weight * SmoothL1(spectral_logmag(recon), spectral_logmag(target))
```

Code:

```text
ecg_adv_gen/vae/losses.py
scripts/vae500/train_ptbxl_vae500.py
```

VAE reconstruction gate:

| VAE | spectral_weight | hard gate | audit Pearson | first-diff Pearson | MSE | lead residual ratio |
|---|---:|---|---:|---:|---:|---:|
| current VAE500 | 0.000 | pass | 0.9977 | 0.9505 | 0.00710 | 1.088 |
| spectral-r005 | 0.005 | fail | 0.9977 | 0.9453 | 0.00708 | 1.206 |
| spectral-r001 | 0.001 | pass | 0.9977 | 0.9507 | 0.00694 | 1.161 |

Two-center downstream old-recipe check:

| center | direct40 | old VAE500 | spectral-r001 VAE500 | spectral - direct40 | spectral - old VAE |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8671 / 0.6221 | 0.8674 / 0.6226 | +0.40pp / +0.71pp | +0.03pp / +0.05pp |
| ningbo | 0.8810 / 0.5032 | 0.8790 / 0.4964 | 0.8806 / 0.5024 | -0.04pp / -0.08pp | +0.16pp / +0.60pp |

Decision:

```text
spectral_weight=0.001 is a safe VAE refinement and slightly improves over the
old VAE in both CPSC and Ningbo, but it does not reach the +2pp goal versus
matched direct40. Do not scale spectral weight directly. Use spectral-r001 as
a safe VAE candidate only when testing a changed online-AT objective.
```

Loss-gain acceptance follow-up:

```text
Implemented:
  --adv_accept_min_loss_gain

Default:
  None, preserving historical runs.

When set to 0.0:
  only push adversarial samples with adv_bce - clean_bce >= 0.
```

| method | cpsc_2018 held-out AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference |
| old VAE500 | 0.8671 / 0.6221 | +0.38pp / +0.66pp |
| spectral-r001 VAE500 | 0.8674 / 0.6226 | +0.40pp / +0.71pp |
| spectral-r001 + loss-gain acceptance | 0.8688 / 0.6270 | +0.55pp / +1.15pp |

Ningbo stability check:

| method | ningbo held-out AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8810 / 0.5032 | reference |
| old VAE500 | 0.8790 / 0.4964 | -0.20pp / -0.68pp |
| spectral-r001 VAE500 | 0.8806 / 0.5024 | -0.04pp / -0.08pp |
| spectral-r001 + loss-gain acceptance | 0.8807 / 0.5013 | -0.03pp / -0.19pp |

Interpretation:

```text
Loss-gain acceptance gives a clearer CPSC improvement than spectral VAE
refinement alone, but it still misses the +2pp target. On Ningbo it remains
near direct40 AUROC but does not recover AUPRC, so it is not yet a global
recipe. Next useful variants:
  - test a softer acceptance weighting rather than hard rejection;
  - run Chapman only if the softer weighting branch looks safe;
  - consider selecting by source-plus-target validation instead of target-val
    AUROC only.

A hard boundary probability window [0.15, 0.85] was tested on CPSC and
underperformed acceptance-only:
  acceptance-only:       0.8688 / 0.6270
  + prob[0.15,0.85]:     0.8670 / 0.6229
Do not expand this exact hard boundary window.
```

Report:

```text
docs/tmp_md/vae500_spectral_loss_pilot_20260602.md
```

### EfficientNet1DV2

500Hz current reported result:

```text
PTB-XL-only baseline: 0.8336 / 0.5048
VAE500 online AT:     0.8578 / 0.5378
delta:                +2.42pp / +3.30pp
```

This is not yet the target claim, because the baseline is source-only PTB-XL,
not direct K500 fine-tune.

Historical matched v7 EfficientNet evidence:

```text
direct K500: 0.8522 / 0.5247
VAE LHAT:    0.8712 / 0.5544
delta:       +1.90pp / +2.97pp all-zero-kept
drop-all-zero delta: +2.33pp / +5.10pp
```

Interpretation: EfficientNet is close to the target. The first 500Hz task is to
run a strict matched direct K500 fine-tune control.

### ECGFounder

500Hz current matched result:

```text
direct K500 head fine-tune: 0.9086 / 0.6739
VAE500 online AT selected: 0.9086 / 0.6739
delta:                    +0.00pp / +0.00pp
```

Interpretation: ECGFounder direct K500 is strong. Simple VAE AT is currently
rejected by K500-internal selection. Improvements must target direct-FT weak
classes/uncertain samples instead of adding more generic adversarial pressure.

### Benchmark Backbones

Five direct 500Hz source-trained benchmarks have been trained/evaluated.
Best direct source model:

```text
fastai_xresnet1d50
PTB-XL fold10: 0.9206 / 0.8079
PN2021 four-center K500-excluded: 0.8511 / 0.5120
```

Interpretation: benchmark backbones do not yet have direct K500 fine-tune or
VAE AT matched pairs. They are the cleanest next place to test transferability.

Matched xresnet1d50 K500 update, 2026-06-02:

```text
Fresh fastai_xresnet1d50 source model was trained with the same 500Hz
minimal_resample/per_sample_global path used by the online-AT script.
Matched direct K500 and VAE-LHAT K500 runs completed for ningbo,
chapman_shaoxing, cpsc_2018, and georgia.
```

| center | direct K500 | VAE-LHAT K500 | delta |
|---|---:|---:|---:|
| ningbo | 0.8885 / 0.5241 | 0.8886 / 0.5234 | +0.01pp / -0.07pp |
| chapman_shaoxing | 0.8814 / 0.4718 | 0.8799 / 0.4688 | -0.14pp / -0.31pp |
| cpsc_2018 | 0.8529 / 0.5840 | 0.8539 / 0.5856 | +0.11pp / +0.15pp |
| georgia | 0.8286 / 0.6172 | 0.8268 / 0.6162 | -0.17pp / -0.10pp |
| mean | 0.8628 / 0.5493 | 0.8623 / 0.5485 | -0.05pp / -0.08pp |

Decision:

```text
The old VAE500 recipe does not transfer to xresnet1d50. K500 internal
validation improved, but held-out ref-excluded target performance was slightly
below direct K500 on mean. Do not keep scaling this recipe across additional
benchmark backbones before changing the VAE or the AT objective.
```

Report:

```text
docs/tmp_md/benchmark_xresnet50_matched_k500_v7_20260602.md
```

## Top-Conference Risk Register

1. **Wrong baseline risk**

   A claim against source-only PTB-XL baseline is not enough. The paper claim
   must compare against direct K500 fine-tune under the same K500 ids, optimizer,
   selection rule, source floor, and ref-excluded evaluation.

2. **Attribution risk**

   Online AT currently includes target-real anchors and supervised target
   streams. Ablations must separate:

   ```text
   source-only
   direct K500 fine-tune
   target-real supervised stream only
   random/uniform latent hull
   optimized VAE latent online AT
   ```

3. **Backbone-generalization risk**

   EfficientNet has signal, ECGFounder has 0 added gain, benchmark backbones are
   untested. Do not claim backbone-agnostic until matched benchmark results
   support it.

4. **Selection/tuning risk**

   Final recipe cannot be chosen using full target-center test labels or
   center-specific heldout tuning. Use K500-internal validation plus PTB-XL
   source floor.

5. **Label/evaluation risk**

   v7 mapping and all-zero handling must be frozen. Main tables should report
   both all-zero-kept and drop-all-zero.

6. **On-manifold claim risk**

   VAE reconstruction Pearson is strong, but on-manifold claim also needs
   decoded waveform sanity, invalid decode rate, lead consistency, latent
   distance, and teacher/class consistency.

## Method Hypotheses To Test

### H1: Matched Direct-FT Gap

Question:

```text
Does VAE500 online AT beat strict direct K500 fine-tune on EfficientNet1DV2?
```

Plan:

- Run direct K500 fine-tune for EfficientNet1DV2 500Hz on the same four K500
  ref pools already used by VAE500 AT.
- Use same optimizer family, epoch budget, K500-internal validation split,
  source floor, and final ref-excluded PN2021 evaluation.
- Report all-zero-kept/drop-all-zero, per-center/per-class, source PTB-XL floor.

Stop rule:

```text
If VAE500 AT <= direct K500 by more than 0.5pp mean AUPRC, do not scale
EfficientNet seeds yet. Diagnose direct-FT gap first.
```

### H2: Generic Backbone Transfer

Question:

```text
Does VAE latent online AT work beyond EfficientNet1DV2?
```

Plan:

- Add a generic `VictimAdapter` layer for:
  - EfficientNet1DV2
  - ECGFounder frozen/head and optional low-LR fullFT
  - benchmark backbones
- Start with `fastai_xresnet1d50`, because it is the strongest direct benchmark
  and close to EfficientNet source performance.
- For every backbone run matched pairs:

```text
direct_ft
vae_lhat
```

Minimum success:

```text
At least 2 non-EfficientNet backbones show positive mean AUPRC delta over direct
K500, and at least 1 reaches >= +2pp.
```

### H3: Boundary-Targeted VAE Regularizer

Question:

```text
Can we make VAE AT add value on strong direct-FT models by attacking only the
model's weak boundary samples?
```

Plan:

- Build candidate anchors from K500 samples with:
  - high BCE under direct head
  - positive classes below confidence threshold
  - rare classes HYP/MI/CD/STTC quotas
  - clean-correct but low-margin cases
- Generate latent hull samples only near these boundary anchors.
- Add adversarial samples with warmup and low stream weight.

Candidate settings:

```text
boundary_source = direct_head
anchor_score = hard_bce + rare_class_bonus + low_margin_bonus
classes_in_scope = CD,HYP,MI,STTC
adv_weight = 0.05,0.10,0.20,0.30 for EfficientNet-style waveform models
adv_weight = 2,4,8 for ECGFounder feature/head streams
warmup_epochs = 5 or 10
```

Success diagnostics:

```text
atk_init ~= 0.30-0.70
atk_anchor positive but not >0.80 early
loss_gain positive
source PTB-XL AUPRC drop <= 1.0pp
K500 internal validation improves over direct K500
```

### H4: Latent Partner Policy Matters

Question:

```text
Is optimized latent hull better than simple mixup/noise?
```

Ablations:

```text
direct K500 FT
input-space mixup
feature/manifold mixup
uniform same-label latent hull
Dirichlet same-label latent hull
optimized same-label latent hull
optimized exact-positive-set latent hull
optimized compatible-label latent hull with soft labels
SAM or SAM-like direct FT
```

Do not scale every ablation to all backbones initially. Run on:

```text
EfficientNet1DV2: all four centers
fastai_xresnet1d50: all four centers
ECGFounder: CPSC + one non-CPSC center first
```

### H5: K / Target-Ratio Scaling

Question:

```text
How does VAE AT behave as the amount of target-center labeled data increases?
```

Run both fixed-K and ratio-based sensitivity:

```text
fixed K:
  100, 200, 500, 1000

target-center ratios:
  10%, 20%
```

User-confirmed ratio denominator:

```text
10% and 20% are sampled from each center's effective Super5-positive pool:
records with at least one positive label among CD,HYP,MI,NORM,STTC.
All-zero mapped records are not counted in the ratio denominator.
```

Default fixed-K order:

```text
K=500 first, because it matches current mainline.
K=1000 second, because it tests whether more target anchors improve VAE AT
       or make direct FT too strong.
K=100/200 only if the method is promising or if K500 is too direct-FT dominated.
```

Ratio-based interpretation:

```text
10% and 20% are not the same sample count across centers. They should be
reported as target-ratio regimes, not directly mixed with fixed-K tables.
If 10% or 20% exceeds K=1000 by a large margin, it is still allowed to run.
The final target-center test view must exclude all sampled target records,
including internal validation and recipe-selection records.
```

For each fixed-K or ratio setting:

```text
direct target fine-tune
plain VAE-online-AT
AugMix + VAE-online-AT if promoted
same target ids within the setting
same internal validation fraction
same ref-exclusion from final evaluation
same source floor rule
same mapping version
```

Ref-exclusion rule:

```text
Every target record used for direct FT, VAE anchors, AugMix anchors, K-internal
validation, or recipe selection must be excluded from the final target-center
test view for that center.
```

Sampling rule:

```text
Use stratified multi-label sampling when possible:
  preserve rare positives for CD,HYP,MI,STTC;
  keep NORM-only records but do not let them dominate;
  write selected record ids to manifest.
For ratio-based runs, sample from the Super5-positive effective pool only.
```

Goal-mode budget:

```text
Do not run all K/ratio settings for all backbones immediately.
Stage 1: EfficientNet1DV2, cpsc_2018 + chapman_shaoxing, K=500 and K=1000.
Stage 2: EfficientNet1DV2 four centers, K=500 and best of K=1000/10%/20%.
Stage 3: best benchmark backbone, same selected K regimes.
```

Paper interpretation:

```text
If K=500 direct FT is too strong but lower-K shows robust VAE gains, the claim
becomes label-efficient target-center adaptation.

If K=1000 or 10%/20% improves VAE AT beyond direct FT, the claim becomes
scalable target-manifold regularization.
```

### H6: AugMix + VAE Online AT

Question:

```text
Can online ECG AugMix improve over plain VAE online AT by increasing local
waveform diversity around VAE latent adversarial samples?
```

Target claim:

```text
AugMix + VAE-online-AT > plain VAE-online-AT
```

This is a second-level improvement target. The strict comparison should be:

```text
direct K500 fine-tune
plain VAE-online-AT
AugMix + VAE-online-AT
```

All three arms must use the same:

```text
backbone
center
target ids for the current K/ratio setting
K/ratio-internal validation split
source floor rule
ref-excluded PN2021 final eval
mapping version
```

Existing codebase support:

```text
scripts/pgd_cross_center/synth_online_at_super5.py
  --enable_latent_augmix_branch
  --latent_augmix_copies
  --latent_augmix_width
  --latent_augmix_depth
  --latent_augmix_alpha
  --latent_augmix_severity
  --latent_augmix_latent_weight_cap
  --latent_augmix_ops

ecg_adv_gen/adaptation/lhat.py
  build_latent_augmix_branch_signals(...)

methods/augmix/severity.py
  powerline_noise
  emg_noise
  baseline_wander
  baseline_shift
  random_leads_masking
```

Use the five existing ECG operators:

```text
powerline_noise
emg_noise
baseline_wander
baseline_shift
random_leads_masking
```

Main design:

```text
x0    = decoded clean target-center anchor ECG
x_adv = decoded VAE latent-hull adversarial ECG

AugMix branch 0 = x_adv
AugMix branches 1..W-1 = ECG corruption chains applied to x0

x_augmix = (1 - m) * x0 + m * sum_j w_j * branch_j
```

The label is inherited from the anchor / VAE adversarial sample. This is only
valid if AugMix severity is mild enough that diagnostic semantics should not
change. Therefore severity must be treated as a controlled variable, not a
free robustness corruption benchmark.

Severity search:

```text
public severity: 1, 2, 3, 4, 5
internal severity: 2, 4, 6, 8, 10
```

Training-time severity exploration:

```text
safe:       public severity 1-2
medium:     public severity 3
aggressive: public severity 4-5
```

Recommended staged grid:

```text
Stage A smoke:
  centers: cpsc_2018, chapman_shaoxing
  backbone: EfficientNet1DV2 500Hz
  K regime: K=500
  severity: 1, 2, 3, 4, 5
  ops: first four ops only, excluding random_leads_masking
  epochs: 5

Stage B five-op test:
  centers: cpsc_2018, chapman_shaoxing
  backbone: EfficientNet1DV2 500Hz
  K regime: K=500
  severity: 1, 2, 3, 4, 5
  ops: all five ops
  random_leads_masking included at all severities for exploration, but it must
  be reported separately because high-severity lead masking can change the
  diagnostic signal more than other ops
  epochs: 10

Stage C K-scale screen:
  centers: cpsc_2018, chapman_shaoxing
  backbone: EfficientNet1DV2 500Hz
  K regimes: K=500, K=1000, 10%, 20%
  severity: best two severities from Stage A/B
  ops: best op set from Stage A/B
  epochs: 10

Stage D full comparison:
  centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
  backbones: EfficientNet1DV2 + fastai_xresnet1d50
  K regimes: K=500 and one best expanded-K/ratio setting
  severity: best from Stage A/B/C
  epochs: matched to plain VAE AT

Stage E seed scaling:
  seeds: 3
  only for the best frozen K/severity/AugMix recipe
```

Default AugMix hyperparameters:

```text
latent_augmix_width = 3
latent_augmix_depth = -1          # random chain depth 1..3
latent_augmix_alpha = 1.0
latent_augmix_copies = 1
latent_augmix_latent_weight_cap = 0.30
latent_augmix_renorm = true
latent_augmix_clip_abs = 6.0
```

Severity safety rules:

```text
1. If random_leads_masking at severity 3-5 improves PN2021-C but hurts clean
   PN2021, do not use it in the main clean-adaptation recipe.
2. If severity >=4 improves corrupted robustness but hurts clean PN2021, do not
   use it as the main clean-adaptation recipe.
3. If AugMix improves only drop-all-zero but hurts all-zero-kept, report it as
   sensitivity-only, not headline.
4. If AugMix + VAE AT improves over direct K500 but not over plain VAE AT, keep
   AugMix as an ablation, not as the main method.
5. If high severity changes label semantics according to teacher disagreement
   or sanity checks, demote it to robustness-only augmentation.
```

Required diagnostics:

```text
plain VAE AT vs AugMix+VAE AT delta
per-op and per-severity acceptance/survival rate
decoded invalid rate
flatline / amplitude / lead-consistency sanity
source PTB-XL floor
clean PN2021 and PN2021-C performance
attack diagnostics before and after AugMix branch
```

Success rule:

```text
AugMix + VAE AT is promoted only if it beats plain VAE AT by >= +0.5pp mean
AUPRC or AUROC without decreasing the other metric, and still beats direct K500
by the main +2pp rule.
```

Interpretation:

```text
If AugMix helps, claim that waveform-level stochastic neighborhoods around
VAE latent adversarial samples improve target-center adaptation.

If AugMix does not help, keep the clean story as VAE latent-hull online AT and
report AugMix as a negative robustness/augmentation ablation.
```

## Implementation Plan For Goal Mode

### Phase 0: Preflight

- Read `AGENTS.md` first 100 lines.
- Check `git status`.
- Check `/root/autodl-tmp` free disk.
- Check GPU with `nvidia-smi` before any training.
- Use `CUDA_VISIBLE_DEVICES=0` unless user says otherwise.
- Do not push/commit unless explicitly requested.

### Phase 1: Evidence Registry And Matched Pair Audit

Deliverable:

```text
docs/tmp_md/vae_online_at_refine_matched_audit_<date>.md
```

Actions:

- Parse existing EfficientNet, ECGFounder, benchmark run JSONs.
- Produce one table:

```text
backbone | source baseline | direct K500 | VAE AT | matched? | delta
```

- Identify missing direct K500 and missing VAE AT runs.

### Phase 2: EfficientNet1DV2 500Hz Direct K500 Control

Deliverable:

```text
/root/autodl-tmp/vae500_lhat_refine_v7/effnet_direct_k500_<center>_<date>/
docs/tmp_md/effnet500_direct_vs_vae_lhat_<date>.md
```

Actions:

- Reuse current VAE500 K500 ref pools.
- Run direct K500 fine-tune with latent/adv stream disabled.
- Evaluate source PTB-XL and PN2021 K500-excluded.
- Compare against existing VAE500 AT.

Decision:

```text
If EfficientNet VAE AT >= direct FT +2pp AUPRC and source floor holds,
scale to 3 seeds.
If not, move to boundary-targeted recipe before seed scaling.
```

### Phase 3: Generic Victim Adapter

Deliverable:

```text
ecg_adv_gen/adaptation/victim_adapters.py
scripts/pgd_cross_center/run_unified_vae_online_at_super5.py
util/tests/test_victim_adapters.py
```

Required interface:

```python
class VictimAdapter:
    num_classes: int
    class_names: list[str]

    def decode_latent_to_ecg_ct(self, z): ...
    def model_input_from_ecg_ct(self, ecg_ct): ...
    def logits_from_model_input(self, x): ...
    def forward_from_latent_to_logits(self, z): ...
    def trainable_parameters(self, policy: str): ...
    def save_checkpoint(self, path): ...
```

Adapters:

```text
EfficientNet500Adapter
Benchmark500Adapter
ECGFounderHeadAdapter
```

The first implementation target should be `Benchmark500Adapter` for
`fastai_xresnet1d50`, because it is strong enough to be meaningful and simpler
than ECGFounder fullFT.

### Phase 4: Boundary-Targeted Recipe

Deliverable:

```text
docs/tmp_md/vae_online_at_boundary_recipe_<date>.md
```

Add or expose config fields:

```text
anchor_score = hard_bce | low_margin | hard_bce_plus_low_margin
rare_class_quota = class weights for CD,HYP,MI,STTC
adv_acceptance = loss_gain_positive | uncertainty_window | no_gate
teacher_label = anchor_soft | direct_teacher_soft | latent_mixed_teacher
source_floor_metric = PTB-XL fold10 AUPRC
```

Search grid, first pass:

```text
hull_lambda: 0.03, 0.05, 0.10
hull_steps: 3, 5
M: 10, 20, 40
adv_weight: backbone-specific low/mid values
warmup_epochs: 5, 10
selection: K500-internal AUPRC + source floor
```

Do not run full cartesian grid. Use staged screening:

```text
Stage A: CPSC + Chapman, 5 epochs, 1 seed
Stage B: four centers, 10 epochs, 1 seed
Stage C: four centers, 3 seeds, frozen recipe
```

### Phase 5: AugMix + VAE Online AT Screening

Deliverable:

```text
docs/tmp_md/vae_augmix_online_at_screen_<date>.md
```

Actions:

- Reuse the existing `--enable_latent_augmix_branch` path for EfficientNet1DV2
  first.
- Expose the same AugMix path through the generic victim adapter if needed for
  benchmark backbones.
- Run three matched arms:

```text
direct target fine-tune
plain VAE-online-AT
AugMix + VAE-online-AT
```

Initial command shape:

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/ECGTwin/bin/python -u \
  scripts/pgd_cross_center/synth_online_at_super5.py \
  --vae_backend diffusets500_v1 \
  --vae500_ckpt /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt \
  --enable_latent_augmix_branch \
  --latent_augmix_width 3 \
  --latent_augmix_depth -1 \
  --latent_augmix_alpha 1.0 \
  --latent_augmix_copies 1 \
  --latent_augmix_latent_weight_cap 0.30 \
  --latent_augmix_severity <1|2|3|4|5> \
  --latent_augmix_ops powerline_noise emg_noise baseline_wander baseline_shift
```

Five-op test adds:

```bash
  --latent_augmix_ops powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking
```

Run five-op tests for severity 1-5, but report random lead masking separately
and do not promote high-severity random lead masking unless clean PN2021 does
not regress.

K/ratio expansion:

```text
fixed K: 500, 1000
ratio: 10%, 20%
optional if early result suggests low-K advantage: 100, 200
ratio denominator: per-center Super5-positive effective sample count
```

For every K/ratio setting, write a manifest:

```text
selected_record_ids.json
internal_val_record_ids.json
final_excluded_record_ids.json
sampling_summary.json
```

Promotion criteria:

```text
AugMix + VAE AT > plain VAE AT by >= +0.5pp on AUROC or AUPRC,
the other metric does not decrease,
source PTB-XL AUPRC drop <= 1.0pp,
invalid decode / augmentation sanity remains acceptable.
It must also satisfy the main rule versus direct target fine-tune:
AUROC or AUPRC >= +2.0pp and the other metric non-negative.
```

### Phase 6: Benchmark Backbone Matched Experiments

Order:

```text
1. fastai_xresnet1d50
2. fastai_inception1d
3. fastai_resnet1d_wang
4. fastai_fcn_wang
5. fastai_schirrmeister
```

For each:

```text
direct K500 FT
VAE-online-AT with frozen recipe
PN2021 K500-excluded eval
source PTB-XL eval
```

Stop rule:

```text
If xresnet1d50 and inception both show <=0 mean AUPRC delta, do not spend GPU
on all five until recipe is revised.
```

### Phase 7: ECGFounder Hard Case

Initial scope:

```text
cpsc_2018
chapman_shaoxing
```

Reason:

```text
CPSC is where VAE/rare-class recipes historically help.
Chapman is a non-CPSC stress test where direct K500 is strong.
```

Recipes:

```text
frozen head direct K500
frozen head + boundary-targeted VAE AT
low-LR partial/fullFT direct K500
low-LR partial/fullFT + boundary-targeted VAE AT
```

Do not claim ECGFounder success unless:

```text
VAE AT beats matched direct K500 on at least two centers
and source PTB-XL floor remains acceptable.
```

### Phase 8: Frozen Recipe Multi-Seed

Minimum:

```text
seeds = 3
centers = four main centers
backbones = EfficientNet1DV2 + best benchmark + ECGFounder if positive
K regimes = K500 + one best expanded-K/ratio setting
```

Optional K/ratio:

```text
K = 100, 200, 500, 1000
ratio = 10%, 20%
```

Report:

```text
mean +/- std
paired delta per center
bootstrap CI if cheap
positive centers count
source floor
```

### Phase 9: Final Top-Conference Report

Deliverables:

```text
docs/tmp_md/vae_online_at_refine_topconf_report_<date>.md
docs/tmp_html/vae_online_at_refine_topconf_report_<date>.html
configs/active_evidence_registry.yaml update only after results are accepted
```

Required tables:

```text
Table 1: direct K500 vs VAE AT by backbone and center
Table 2: ablations vs direct K500
Table 3: K sensitivity
Table 3b: target-ratio sensitivity
Table 4: source PTB-XL floor
Table 5: attack diagnostics and invalid decode rates
Table 6: per-class deltas
```

## External Method Context

Useful references for positioning:

- ECG foundation-model post-training reports large improvements over baseline
  fine-tuning and highlights that post-training strategy matters, not only
  pretraining scale: https://arxiv.org/abs/2509.12991
- CoTMix supports the idea that temporal mixup can reduce time-series domain
  shift through intermediate views and contrastive/domain adaptation:
  https://arxiv.org/abs/2212.01555
- Ad2Mix supports adaptive/adversarial mixup as a domain adaptation direction,
  especially curriculum-style intermediate domains:
  https://openaccess.thecvf.com/content/WACV2025/papers/Zhu_Ad2mix_Adversarial_and_Adaptive_Mixup_for_Unsupervised_Domain_Adaptation_WACV_2025_paper.pdf
- Recent latent-representation mixup work supports testing convex/masked latent
  interpolation as an adversarial robustness mechanism:
  https://ro.uow.edu.au/articles/journal_contribution/Boost_Off_On-Manifold_Adversarial_Robustness_for_Deep_Learning_with_Latent_Representation_Mixup/27807846

These are positioning references, not direct evidence for ECG VAE-LHAT. Final
claims must rely on this repo's matched ECG experiments.

## Confirmed Success Criterion

```text
AUROC 或 AUPRC 任意一个达到 +2pp，另一个不能下降。
```

Use this as the automatic goal-mode promotion rule. For the final paper claim,
also report both metrics and avoid hiding a trade-off.

## Execution Log

### 2026-06-01 Matched EfficientNet1DV2 Direct Control

Status: completed.

Purpose:

```text
Establish the strict 500Hz matched direct target fine-tune baseline for the
existing VAE500 online AT result.
```

Output root:

```text
/root/autodl-tmp/vae500_direct_ft_v7_20260601_r2/
```

Protocol:

```text
script: scripts/pgd_cross_center/synth_online_at_super5.py
control switch: --disable_adv_stream
backbone: EfficientNet1DV2 500Hz
init: /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601/best_model.pt
target anchors: /root/autodl-tmp/vae500_lhat_v7/anchors_k500/<center>/<center>_real_k500_seed42_vae500.*
target_real_weight: 20
ptbxl_weight: 1
selection: K500-internal validation, target_macro_auprc
final eval: PN2021 v7 500Hz ref-excluded, all-zero-kept and drop-all-zero
```

Hardware/acceleration:

```text
GPU: RTX 4090D, CUDA_VISIBLE_DEVICES=0
batch_size: 128
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 4
TF32: enabled
CuDNN benchmark: enabled
```

Implementation note:

```text
The first control run exposed a direct-control bug: with --disable_adv_stream,
attack diagnostics are expected to be NaN/None, but agent_attack_decision()
attempted float(None). The code now treats None as NaN and records the state as
no_attack_or_disabled. The failed partial run is kept under
/root/autodl-tmp/vae500_direct_ft_v7_20260601/ for traceability; the clean rerun
uses the _r2 output root above.
```

Result:

```text
matched direct FT mean: 0.8601 / 0.5416
plain VAE500 AT mean:   0.8578 / 0.5378
VAE - direct:          -0.23pp / -0.38pp
```

Decision:

```text
Plain VAE500 online AT does not satisfy the goal against matched direct FT.
Do not scale this plain recipe as the final claim. Continue with AugMix+VAE and
boundary-targeted online AT.
```

Detailed report:

```text
docs/tmp_md/vae_online_at_refine_execution_audit_20260601.md
```

### 2026-06-01 AugMix+VAE Stage A Smoke

Status: completed.

Reason:

```text
Strict matched direct FT erased the plain VAE500 AT advantage. The next
automatic branch tests whether ECG AugMix branches around VAE latent-hull
adversarial samples add useful diversity beyond plain VAE AT.
```

Output root:

```text
/root/autodl-tmp/vae500_augmix_stageA_v7_20260601/
```

Grid:

```text
centers: cpsc_2018, chapman_shaoxing
K: 500
epochs: 6 smoke
internal severity: 2, 4, 6, 8, 10
ops: powerline_noise, emg_noise, baseline_wander, baseline_shift
excluded op: random_leads_masking
```

Promotion rule:

```text
Continue only severities that improve over matched direct FT or at least beat
plain VAE500 AT without lowering the other metric.
```

Result:

```text
cpsc_2018 best AugMix severity 4: 0.8499 / 0.5954
cpsc_2018 direct FT:             0.8533 / 0.6011
delta:                          -0.34pp / -0.57pp

chapman best AugMix severity 10: 0.8733 / 0.4427
chapman direct FT:              0.8725 / 0.4448
delta:                          +0.08pp / -0.21pp
```

Decision:

```text
Default hard-label AugMix+VAE does not pass. It slightly repairs CPSC plain-VAE
loss but remains below direct FT, and on Chapman it trades small AUROC gains for
AUPRC loss. Do not scale this setting.
```

Detailed report:

```text
docs/tmp_md/vae_online_at_refine_augmix_stageA_20260601.md
```

Next branch:

```text
Stage B should reduce adversarial/AugMix pressure and use teacher/soft labels:
adv_weight 0.10 or 0.15, latent_augmix_latent_weight_cap 0.15,
latent_mixed_teacher or teacher_soft labels, severity 4, centers cpsc_2018 and
chapman_shaoxing.
```

### 2026-06-01 AugMix+VAE Stage B Soft/Low-Weight Smoke

Status: running.

Output root:

```text
/root/autodl-tmp/vae500_augmix_stageB_v7_20260601/
```

Grid:

```text
centers: cpsc_2018, chapman_shaoxing
K: 500
epochs: 10 smoke
internal severity: 4
latent_augmix_latent_weight_cap: 0.15
adv_weight: 0.10
label modes:
  - multi_hot_hard
  - latent_mixed_teacher
  - teacher_soft
```

Reason:

```text
Stage A suggested that hard-label AugMix/VAE can hurt AUPRC. Stage B tests
whether lower adversarial pressure plus teacher/soft labels preserve ranking
while retaining the small AUROC signal.
```

## Remaining Clarifications Before Goal Mode

All user-blocking clarification items below are now resolved as of 2026-06-01.
The next goal can start from this document.

1. **Primary comparison target**

   Recommendation:

   ```text
   plain VAE AT must beat direct K500 by the +2pp rule;
   AugMix + VAE AT must additionally beat plain VAE AT by >= +0.5pp
   on AUROC or AUPRC without decreasing the other metric.
   ```

   Status: confirmed.

2. **Expanded target data regimes**

   Recommendation:

   ```text
   include fixed K=500 and K=1000 plus target-ratio 10% and 20%.
   Optional K=100/200 only if early evidence suggests low-K is where VAE helps.
   ```

   Status: confirmed, with ratio denominator = per-center Super5-positive
   effective sample count. 10%/20% are allowed even when larger than K=1000.

3. **First backbone for AugMix**

   Recommendation:

   ```text
   start with EfficientNet1DV2 because the current online AT script already
   supports latent AugMix there; then port the winning recipe to xresnet1d50.
   ```

   Status: accepted by plan default.

4. **Main evaluation view**

   Recommendation:

   ```text
   all-zero-kept remains the main table;
   drop-all-zero is required sensitivity analysis.
   ```

   Status: accepted by plan default.

## 2026-06-02 User-Refined VAE Bottleneck Branch

User direction:

```text
During exploration, explicitly allow the possibility that the current 500Hz
VAE is the bottleneck. Explore a stronger 500Hz VAE by adapting the original
ECGTwin author's VAE weights, and keep checking whether there are credible
top-conference / top-journal 500Hz ECG VAE implementations on GitHub.
```

Status after local and subagent audit:

```text
1. ECGTwin original VAE weights are available at:
     model/ECGTwin/checkpoints/vae_model.pth
   with checkpoint keys {'encoder', 'decoder'}.

2. The original ECGTwin VAE is convolutional and length-flexible enough to be
   reused in an ECGTwinVAE500 wrapper:
     5000 x 12 -> latent (4,625) -> 5000 x 12
   and can strictly load the original encoder/decoder weights.

3. The current DiffuSETSVAE500 is not state-dict-compatible with ECGTwin's
   original VAE. It has different widths, module names, downsampling details,
   and attention implementation. Do not directly load the ECGTwin checkpoint
   into DiffuSETSVAE500.

4. A previous cheap diagnostic using the original ECGTwin 1024 latent manifold
   did not fix the direct40-init downstream failure. Therefore, a stronger VAE
   alone is not enough evidence; VAE changes must be paired with a changed
   online-AT objective / sample acceptance / checkpoint selection rule.
```

Controlled VAE options:

| option | what to do | role | stop condition |
|---|---|---|---|
| ECGTwin-init VAE500 | Same original ECGTwin VAE module family at 5000 samples, strict-load author weights, PTB-XL-only fine-tune | cleanest use of author weights | stop if reconstruction remains worse than current VAE500 by >5% MSE or if two-center pilot selects epoch0 |
| ECGTwin-teacher DiffuSETSVAE500 | Freeze original ECGTwin 1024 VAE, distill reconstruction/latent geometry into current 5000-sample student | safest way to transfer author geometry without incompatible partial load | stop if downstream delta stays within +/-0.2pp after objective change |
| spectral/inter-lead/morphology losses | Borrow SE-Diff-style constraints while training on PTB-XL only | improve latent geometry without external data leakage | stop if lead residual or first-diff Pearson worsens |
| SE-Diff external reference | Evaluate only if pretrained VAE weights are publicly obtainable and license-compatible | external 500Hz ECG latent reference | report as external pretraining, not PTB-XL-only main claim |
| ECGEN external reference | Secondary 12-lead 5000-sample VAE baseline or teacher | engineering baseline | do not use MIMIC-pretrained weights as the main causal claim |

External source check:

```text
SE-Diff GitHub: https://github.com/ignite-abd/SE-Diff
  - ICLR 2026 ECG generation work.
  - README requires pretrained weights, latent data, and simulator priors under
    ./prerequisites.
  - Local clone contains code but no bundled pretrained weights.

ECGEN GitHub: https://github.com/vlbthambawita/ECGEN
  - Provides a 12-lead VAE for 5000-sample ECG signals.
  - Trained primarily on MIMIC-IV-ECG; use as external baseline/teacher only.

DiffuSETS GitHub: https://github.com/Raiiyf/DiffuSETS_Exp
  - Closest ECGTwin/DiffuSETS family with VAE latent pipeline and public code.
  - Official assets are reported through GitHub/Hugging Face/Zenodo.
  - This is the only external 500Hz VAE-family candidate that currently looks
    immediately auditable as a pretrained checkpoint source.
```

External audit refinement:

```text
Do not claim "no external VAE exists" too broadly. The accurate statement is:
  - DiffuSETS is the only credible pretrained/public-asset candidate worth
    auditing immediately.
  - SE-Diff and ECGEN are useful architecture/loss references, but are not
    confirmed drop-in pretrained replacements for our VAE500.
  - Any MIMIC-pretrained external VAE must be labeled as external pretraining
    and cannot be mixed into the PTB-XL-only causal claim.
```

Minimum experiment matrix before any long run:

```text
centers: cpsc_2018 and ningbo only
control: matched direct40/direct K500, same K ids, ref-excluded eval
VAE backends:
  - current VAE500
  - spectral-r001 VAE500
  - ECGTwin-teacher VAE500
  - ECGTwin-init VAE500 only if reconstruction gate passes
AT objectives:
  - old recipe
  - loss-gain acceptance / soft weighting
selection:
  - K500 internal validation plus PTB-XL source floor
required logs:
  - ASR, loss_gain, atk_anchor, decoded invalid rate
  - K500-val selected epoch
  - held-out ref-excluded AUROC/AUPRC
```

Decision rule:

```text
Do not spend a multi-center/multi-backbone sweep on VAE replacement unless the
two-center pilot shows:
  - not selected back to epoch0 on both centers;
  - CPSC AUPRC improves by roughly >= +1pp over matched direct control;
  - Ningbo AUPRC is non-decreasing or within -0.1pp;
  - PTB-XL source floor remains acceptable.

If those conditions fail, freeze the VAE branch and continue on online-AT
objective design rather than VAE architecture.
```

Two-center gate result, 2026-06-02:

```text
Tested ECGTwin-teacher DiffuSETSVAE500 + loss-gain acceptance with
EfficientNet1DV2 500Hz on cpsc_2018 and ningbo.

Run root:
  /root/autodl-tmp/vae500_teacher_acceptlg_effnet_k500_v7_20260602/

Report:
  docs/tmp_md/vae500_teacher_acceptlg_effnet_pilot_20260602.md
```

| center | direct40 | spectral-r001 + loss-gain | ECGTwin-teacher VAE + loss-gain |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8686 / 0.6257 |
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5013 | 0.8807 / 0.5003 |

Decision:

```text
ECGTwin-teacher VAE + loss-gain is technically valid but does not beat the
spectral-r001 + loss-gain branch and still fails the Ningbo non-decreasing
AUPRC condition. Do not expand this VAE-teacher branch to four centers/backbones
under the current online-AT objective.

Continue with online-AT objective / sample weighting / checkpoint selection.
Only revisit VAE replacement after the objective change shows stable gains with
the current or spectral-r001 VAE.
```

Experiment-audit closure:

```text
As of the latest audit, ECGTwin-init and ECGTwin-teacher VAE500 are allowed only
as gated branches. They have passed feasibility checks, but they have not passed
the downstream CPSC/Ningbo gate. Do not launch a long four-center or
multi-backbone sweep for these VAE variants unless a new objective first creates
a clear two-center gain over both matched direct FT and the current VAE500.
```

Selection metric pilot, 2026-06-02:

```text
Tested whether Ningbo underperformance is caused by checkpoint selection using
internal AUROC rather than internal AUPRC.

Report:
  docs/tmp_md/vae500_selection_metric_pilot_20260602.md
```

| method | ningbo AUROC / AUPRC | delta vs direct40 |
|---|---:|---:|
| direct40 | 0.8810 / 0.5032 | reference |
| spectral-r001 + loss-gain, select AUROC | 0.8807 / 0.5013 | -0.03pp / -0.19pp |
| spectral-r001 + loss-gain, select AUPRC | 0.8807 / 0.5018 | -0.03pp / -0.14pp |

Decision:

```text
AUPRC-based selection slightly improves Ningbo AUPRC but still does not beat
matched direct40. Do not spend more time on selection-metric-only sweeps.
The next candidate must change the online-AT objective or sample weighting, not
just checkpoint selection.
```

Soft loss-gain weighting pilot, 2026-06-02:

```text
Implemented default-off buffer score reweighting:
  score *= 1 + strength * tanh(loss_gain / scale)

Pilot setting:
  scale=0.05
  strength=0.5
  no hard min_loss_gain rejection
  select by internal AUPRC

Report:
  docs/tmp_md/vae500_soft_loss_gain_weighting_pilot_20260602.md
```

| center | direct40 | hard loss-gain + AUPRC selection | soft loss-gain weighting |
|---|---:|---:|---:|
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | 0.8818 / 0.5063 |
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8670 / 0.6206 |

Decision:

```text
Soft loss-gain weighting fixes Ningbo's small AUPRC drop but weakens CPSC.
It is not a global replacement for hard loss-gain acceptance. Keep it as a
candidate in a selector pool or as a center-adaptive option for centers where
hard filtering over-prunes useful samples.
```

Four-center hard/soft loss-gain closure, 2026-06-02:

```text
Purpose:
  complete the hard-vs-soft loss-gain check on all four target centers before
  spending more compute on VAE replacement or small acceptance-threshold sweeps.

Fixed protocol:
  EfficientNet1DV2 500Hz
  v7_super5_sjr_rgq_review_20260528 mapping
  spectral-r001 VAE500 anchors
  K=500 target anchors per center
  target-center K500 reference ids excluded from final evaluation
```

| center | direct40 | hard loss-gain | hard delta | soft loss-gain | soft delta |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | -0.03pp / -0.14pp | 0.8818 / 0.5063 | +0.08pp / +0.32pp |
| chapman_shaoxing | 0.8757 / 0.4508 | 0.8742 / 0.4522 | -0.16pp / +0.14pp | 0.8727 / 0.4517 | -0.30pp / +0.09pp |
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.8670 / 0.6206 | +0.37pp / +0.51pp |
| georgia | 0.8364 / 0.6202 | 0.8340 / 0.6165 | -0.24pp / -0.36pp | 0.8349 / 0.6180 | -0.14pp / -0.21pp |
| mean | 0.8641 / 0.5474 | 0.8644 / 0.5494 | +0.03pp / +0.20pp | 0.8641 / 0.5492 | +0.00pp / +0.18pp |

Exploratory best-by-AUPRC selector among direct40/hard/soft:

```text
mean = 0.8653 / 0.5514
delta vs direct40 = +0.12pp AUROC / +0.40pp AUPRC
```

Decision:

```text
The hard/soft loss-gain family creates a small AUPRC lift, mainly from CPSC.
It still misses the +2pp goal and does not justify further small threshold or
selection-metric sweeps. Treat this as a limited negative result.

Next VAE work must be gated. ECGTwin author VAE initialization or
teacher-distillation is allowed, but only after reconstruction and two-center
downstream gates pass. Do not start a long four-center/multi-backbone VAE
replacement run solely because the current 500Hz VAE might be imperfect.
```

Report:

```text
docs/tmp_md/vae500_bottleneck_and_ecgtwin_teacher_plan_20260602.md
```

AugMix + inv-frequency pilot closure, 2026-06-02:

```text
Purpose:
  test a stronger objective-level change:
    spectral-r001 VAE500
    + latent AugMix
    + inverse-frequency K500 anchor quota
    + light source-logit anchor

Pilot center:
  cpsc_2018
```

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 | PTB-XL AUROC / AUPRC |
|---|---:|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference | 0.9086 / 0.7749 |
| spectral-r001 hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.9079 / 0.7733 |
| hard loss-gain, eval_every=1, select AUPRC | 0.8683 / 0.6255 | +0.50pp / +1.00pp | 0.9082 / 0.7738 |
| AugMix + invfreq + source anchor | 0.8633 / 0.6106 | -0.01pp / -0.49pp | 0.9082 / 0.7734 |
| accepted-only AugMix + hard loss-gain | 0.8669 / 0.6217 | +0.36pp / +0.62pp | 0.9082 / 0.7741 |

Decision:

```text
Do not expand this AugMix+invfreq recipe to Ningbo or four centers. It strongly
improved K500-internal validation but failed ref-excluded held-out CPSC, which
is exactly the validation-transfer failure this plan must avoid.

The next AugMix implementation fix should make latent-AugMix accepted-sample
aware. Today the main adversarial push can be filtered by loss-gain/boundary
gates, but the latent-AugMix branch is built from all adversarial candidates.
This can amplify samples that the main branch would reject.
```

Next concrete code task:

```text
Update push_adv_to_buffer to optionally return accepted indices/mask, and build
latent-AugMix only from accepted adversarial candidates. Then run a single
cpsc_2018 pilot:
  spectral-r001 VAE500
  hard loss-gain acceptance
  accepted-only latent AugMix
  same K500 split and ref-excluded eval
```

Accepted-only AugMix follow-up:

```text
Implemented and tested.
Run:
  /root/autodl-tmp/vae500_hardlg_accepted_augmix_k500_v7_20260602/
    cpsc_2018_hardlg_accepted_augmix_ep10_seed42

Result:
  0.8669 / 0.6217 on cpsc_2018 held-out ref-excluded.
  This is better than unfiltered AugMix but still below hard loss-gain
  without AugMix.
```

Decision:

```text
Keep the accepted-only AugMix code fix for correctness, but do not promote
AugMix as the main current method. The current best EfficientNet1DV2 CPSC
branch remains spectral-r001 hard loss-gain acceptance.
```

Checkpoint-selection follow-up:

```text
Reran CPSC hard loss-gain with eval_every=1 and val_macro_auprc selection.
Result:
  0.8683 / 0.6255
This is slightly below the original hard loss-gain result:
  0.8688 / 0.6270

Conclusion:
  checkpoint-selection frequency/metric is not the missing mechanism on CPSC.
```

Report:

```text
docs/tmp_md/vae500_augmix_sourcefloor_invfreq_pilot_20260602.md
```

Rank-aware objective gate, 2026-06-02:

```text
Purpose:
  test whether a rare-abnormal pairwise multilabel ranking objective can create
  a stronger VAE-online-AT signal than hard loss-gain alone.

Implementation:
  scripts/pgd_cross_center/synth_online_at_super5.py
  --rank_loss_weight
  --rank_loss_margin
  --rank_loss_positive_classes

Default behavior:
  rank_loss_weight=0.0, so historical runs are unchanged.
```

Pilot protocol:

```text
center: cpsc_2018
rank_loss_weight=0.05
rank_loss_positive_classes=CD,HYP,MI,STTC
matched direct control: direct40 + rank loss, no adv stream
VAE branch: spectral-r001 hard loss-gain + rank loss
selection: K500-internal AUPRC
final eval: cpsc_2018 held-out, K500 ref ids excluded
```

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 | PTB-XL AUROC / AUPRC |
|---|---:|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference | 0.9086 / 0.7749 |
| spectral-r001 hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.9079 / 0.7733 |
| direct40 + rank loss | 0.8652 / 0.6181 | +0.19pp / +0.26pp | 0.9086 / 0.7744 |
| VAE hard loss-gain + rank loss | 0.8687 / 0.6267 | +0.54pp / +1.12pp | 0.9083 / 0.7734 |

Decision:

```text
Rank-aware loss is technically valid, ASR remains healthy, and decoded_invalid
is 0, but the VAE-rank run does not beat the current hard loss-gain branch.
Because the pre-registered CPSC gate failed, do not expand this exact
rank_loss_weight=0.05 recipe to Ningbo or four centers.
```

Report:

```text
docs/tmp_md/vae500_rankaware_gate_pilot_20260602.md
```

Hard + soft loss-gain gate, 2026-06-02:

```text
Purpose:
  combine hard loss-gain acceptance with soft loss-gain replay weighting.

Rationale:
  hard-only helped CPSC but hurt Ningbo;
  soft-only helped Ningbo but weakened CPSC;
  hard+soft tests "filter bad samples first, then prioritize accepted samples".
```

| center | direct40 | hard-only | soft-only | hard+soft | hard+soft delta |
|---|---:|---:|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | 0.8670 / 0.6206 | 0.8679 / 0.6255 | +0.46pp / +1.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | 0.8818 / 0.5063 | 0.8808 / 0.5024 | -0.02pp / -0.08pp |

Decision:

```text
The hard+soft combination is not the missing mechanism. It keeps a positive
CPSC delta over direct40 but is weaker than hard-only, and it fails Ningbo by
falling below both direct40 and soft-only. Do not expand this recipe to Chapman,
Georgia, additional seeds, or benchmark backbones.

This closes the current minor loss-gain knob family. The next useful work is
either a materially different mechanism with a pre-registered CPSC/Ningbo gate,
or consolidation of the matched-direct negative evidence into paper-ready
comparison tables.
```

Report:

```text
docs/tmp_md/vae500_hardsoft_loss_gain_gate_20260602.md
```

EfficientNet1DV2 500Hz current closure, 2026-06-02:

```text
Report:
  docs/tmp_md/vae500_effnet500_current_closure_20260602.md

Conclusion:
  No tested EfficientNet1DV2 500Hz/v7 VAE-online-AT branch has satisfied the
  goal of +2pp over matched direct40 target fine-tuning. The available recipe
  family gives local CPSC gains but does not transfer globally, and the current
  minor knob family is closed.

Next work should be either:
  1. evidence consolidation / paper-ready comparison tables;
  2. a materially different mechanism with a pre-registered CPSC/Ningbo gate;
  3. an external few-shot/domain-adaptation comparison baseline.
```
