# AUROC +3pp / AUPRC +7pp Center-Token Ablation Audit

Date: 2026-05-04

## Question

The custom seed42 graduate-project line produced a large low-resource PTB-XL gain:

```text
 about 3 percentage points macro AUROC
+ about 7 percentage points macro AUPRC
```

This audit asks whether that gain is caused by the PTB-XL center prompt token, or by the broader ECGTwin synthetic-pretraining plus real-data fine-tuning/self-distillation recipe.

## Fixed Controls

The matched ablation keeps these variables fixed:

```text
split                 = /root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
real train samples    = same PTB-XL 2000 records
synthetic pool size   = 20000 ECGTwin samples
reference policy      = translated-report random_any reference from the same real2000 split
classifier            = EfficientNet1DV2 super5
preprocessing         = minimal_resample + per_sample_global, 100 Hz, 10 s
latent-hull AT        = same real-anchor latent pool and M10 recipe
self-distillation     = same D1 recipe and seeds
```

Only the prompt condition changes:

```text
token arm     = diagnosis prompt + PTB-XL MV4 class prompt token
no-token arm  = diagnosis prompt only
```

## Main Result

| stage | arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| A-fair | real2000 only | 0.8357 | 0.6040 | 0.8250 | 0.6011 | 0.7339 | 0.4074 |
| C0 | token synth-only pretrain | 0.6649 | 0.4107 |  |  |  |  |
| C0 | no-token synth-only pretrain | 0.6023 | 0.3851 |  |  |  |  |
| C1 | token synth-pretrain -> real2000 FT | 0.8645 | 0.6723 | 0.8551 | 0.6693 | 0.7364 | 0.4195 |
| C1 | no-token synth-pretrain -> real2000 FT | 0.8670 | 0.6846 | 0.8560 | 0.6695 | 0.7719 | 0.4750 |
| C3 | token C1 -> real-anchor latent-hull AT | 0.8654 | 0.6738 | 0.8565 | 0.6720 | 0.7381 | 0.4230 |
| C3 | no-token C1 -> real-anchor latent-hull AT | 0.8705 | 0.6904 | 0.8589 | 0.6745 | 0.7651 | 0.4670 |
| D1 | token C3-teacher self-distill | 0.8670 | 0.6737 | 0.8581 | 0.6711 | 0.7407 | 0.4280 |
| D1 | no-token C3-teacher self-distill | 0.8750 | 0.7007 | 0.8601 | 0.6813 | 0.7711 | 0.4687 |

## Token Minus No-Token

| stage | custom AUROC delta | custom AUPRC delta | PN2021 AUROC delta | PN2021 AUPRC delta |
|---|---:|---:|---:|---:|
| C1 token - no-token | -0.0025 | -0.0123 | -0.0355 | -0.0555 |
| C3 token - no-token | -0.0051 | -0.0166 | -0.0270 | -0.0440 |
| D1 token - no-token | -0.0081 | -0.0270 | -0.0304 | -0.0407 |

## Interpretation

The +3pp/+7pp gain is reproducible versus the A-fair real2000 baseline, but this ablation does not validate the center token as the causal factor.

The center-token arm is better only at the weak C0 synthetic-only pretraining stage. After real2000 fine-tuning, latent-hull AT, and self-distillation, the no-token ECGTwin control is consistently stronger on custom seed42, fold10 subset evaluation, and PN2021 external evaluation.

Therefore the thesis-safe interpretation for this line is:

```text
ECGTwin synthetic pretraining plus real-data fine-tuning/self-distillation improves the custom low-resource PTB-XL protocol.
The large +3pp/+7pp gain should not be attributed specifically to the PTB-XL center token.
```

## Remaining Positive Center-Token Evidence

The separate v2 filtered self-distillation ablation still shows a center-token advantage under the legacy PN2021 evaluation口径:

| arm | custom AUROC | custom AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|
| vanilla ECGTwin + v2 | 0.8333 | 0.6051 | 0.7528 | 0.4233 |
| PTB-XL center-token ECGTwin + v2 | 0.8500 | 0.6371 | 0.7616 | 0.4546 |

This is useful evidence that center-token conditioning can help a filtered/self-distilled synthetic pool, but it is not the same as proving that the original +3pp/+7pp line came from the center token.

## Same-Preprocessing Scale-Up Ablation

After the first audit, the filtered self-distillation v2 line was rerun under one
consistent external-evaluation口径:

```text
preprocess_mode = minimal_resample
norm_mode       = per_sample_global
crop_len        = 1000
PN2021 cache    = pn2021_eval_cache_mmap_minresample_perglobal
date            = 2026-05-04
```

This ablation asks a stricter question: if the no-token arm is allowed the same
synthetic scale and the same self-distillation recipe, does the center-token arm
still win?

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| real2000 baseline | 0.8433 | 0.6234 | 0.8297 | 0.6190 | 0.7085 | 0.3912 |
| token filtered top2000 | 0.8500 | 0.6371 | 0.8405 | 0.6380 | 0.7067 | 0.3937 |
| no-token filtered top2000 | 0.8333 | 0.6051 | 0.8212 | 0.6004 | 0.7020 | 0.3993 |
| token filtered top4000 | 0.8555 | 0.6528 | 0.8445 | 0.6463 | 0.7201 | 0.4168 |
| no-token filtered top3970 | 0.8620 | 0.6684 | 0.8532 | 0.6577 | 0.7288 | 0.4291 |
| HYP-token hybrid top4000 | 0.8530 | 0.6504 | 0.8448 | 0.6381 | 0.7253 | 0.4296 |

Key deltas:

| comparison | custom delta | fold10 delta | PN2021 delta |
|---|---:|---:|---:|
| token top2000 - real2000 | +0.0067 / +0.0137 | +0.0108 / +0.0190 | -0.0017 / +0.0024 |
| token top2000 - no-token top2000 | +0.0166 / +0.0320 | +0.0193 / +0.0375 | +0.0047 / -0.0057 |
| token top4000 - no-token top3970 | -0.0065 / -0.0157 | -0.0087 / -0.0114 | -0.0087 / -0.0123 |
| HYP-token hybrid - no-token top3970 | -0.0089 / -0.0180 | -0.0084 / -0.0197 | -0.0035 / +0.0006 |
| no-token top3970 - real2000 | +0.0187 / +0.0450 | +0.0235 / +0.0387 | +0.0203 / +0.0378 |

Interpretation:

```text
The center-token top2000 arm is better than the no-token top2000 arm on PTB-XL,
but the advantage disappears when no-token is scaled to the same candidate count.
The strongest same-preprocessing filtered-v2 model is the no-token top3970 arm,
not the center-token arm.

Therefore this line still supports ECGTwin synthetic/self-distillation utility,
but does not validate PTB-XL center token as the causal source of the AUROC/AUPRC gain.
```

## Generation-Side Token Evidence

A newer v45 contrastive token was trained for ningbo with paired token-vs-no-token
losses. This is not the PTB-XL +3pp/+7pp line, but it is useful for separating
generation-style effectiveness from downstream classifier effectiveness.

Large paired generation used the same reference policy and seed:

```text
target-token samples: 1200 = NORM 400 + MI 400 + STTC 400
no-token samples:     1200 = NORM 400 + MI 400 + STTC 400
style probe:          /root/autodl-tmp/center_style_classifier_noleak/full1000
```

| arm | class | mean P(ningbo) | top1 ningbo rate |
|---|---|---:|---:|
| no-token | MI | 0.0802 | 0.0800 |
| no-token | NORM | 0.1184 | 0.0725 |
| no-token | STTC | 0.0841 | 0.0875 |
| target-token | MI | 0.4643 | 0.5175 |
| target-token | NORM | 0.6247 | 0.7725 |
| target-token | STTC | 0.5190 | 0.6075 |

Quality gate counts:

| arm | total pass | MI | NORM | STTC |
|---|---:|---:|---:|---:|
| target-token, no top1 gate | 705 / 1200 | 304 | 390 | 11 |
| no-token, no top1 gate | 715 / 1200 | 283 | 271 | 161 |

Interpretation:

```text
v45 validates that a contrastively trained center token can push ECGTwin samples
toward the target center according to the no-leak style probe.

This is generation-side evidence only. It does not yet overturn the downstream
AUROC/AUPRC ablation, because the matched downstream token-vs-no-token test for
v45 has not beaten no-token.
```

## v46 PTB-XL Contrastive Token Matched Ablation

To directly target the PTB-XL low-resource self-distillation failure, a new
PTB-XL source center token was trained with paired contrastive reconstruction
and semantic objectives:

```text
token bank:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v46_contrast_recon_sem_steps4000_20260504

candidate generation:
  center        = ptbxl_source
  classes       = CD/HYP/MI/NORM/STTC
  candidates    = 1600/class, 8000 total per arm
  same refs      = yes
  same seeds     = yes
  difference     = learned PTB-XL class token vs no learned token
```

Generation/filter evidence:

| arm | candidates | v2 filtered kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v46 center-token | 8000 | 4000 | 800 | 800 | 800 | 800 | 800 |
| matched no-token | 8000 | 3734 | 800 | 534 | 800 | 800 | 800 |
| v46 token count-matched | token filtered subset | 3734 | 800 | 534 | 800 | 800 | 800 |

The center token clearly improves the v2 teacher-filter pass rate, especially
for HYP. However, downstream utility is the required proof.

Strict downstream result:

| arm | script test AUROC | script test AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| real2000 baseline | 0.8433 | 0.6234 | 0.8297 | 0.6190 | 0.7085 | 0.3912 |
| old token top4000 | 0.8555 | 0.6528 | 0.8445 | 0.6463 | 0.7201 | 0.4168 |
| old no-token top3970 | 0.8620 | 0.6684 | 0.8532 | 0.6577 | 0.7288 | 0.4291 |
| v46 token top4000 | 0.8508 | 0.6437 | 0.8430 | 0.6419 | 0.7187 | 0.4073 |
| v46 no-token 3734 | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| v46 token count-matched 3734 | 0.8537 | 0.6466 | 0.8426 | 0.6383 | 0.7078 | 0.4038 |

Token deltas versus matched no-token:

| comparison | script test delta | fold10 delta | PN2021 delta |
|---|---:|---:|---:|
| v46 token top4000 - no-token 3734 | -0.0022 / +0.0033 | -0.0003 / +0.0086 | -0.0063 / -0.0093 |
| v46 token count-matched - no-token 3734 | +0.0007 / +0.0062 | -0.0008 / +0.0050 | -0.0172 / -0.0129 |

Decision:

```text
v46 validates a generation/filtering effect but not a downstream AUROC/AUPRC
effect. PTB-XL AUPRC increases slightly in matched tests, but AUROC does not,
and PN2021 external performance is worse than no-token.

Therefore the +3pp/+7pp low-resource PTB-XL result still cannot be attributed
to the PTB-XL center token. The reproducible useful ingredient remains ECGTwin
synthetic/self-distillation itself, with no-token still the strongest strict
matched arm so far.
```

## v47 Feature-Contrast And Boundary-Confidence Ablation

After v46, a stronger PTB-XL source token was attempted with an additional
EfficientNet penultimate-feature reconstruction loss and paired feature-contrast
loss.

Token training:

```text
token bank:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v47_feature_contrast_steps4000_20260504

extra losses:
  feature_loss_weight = 0.05
  contrast_feature_weight = 0.10
  feature_crop_len = 1000
  semantic_loss_weight = 0.02
  contrast_recon_weight = 0.20
  contrast_semantic_delta_weight = 0.03
```

Small generation sanity check, 60 samples/class:

| arm | CD top1 | HYP top1 | MI top1 | NORM top1 | STTC top1 |
|---|---:|---:|---:|---:|---:|
| v46 token | 0.967 | 0.750 | 0.883 | 1.000 | 0.933 |
| v46 no-token | 0.833 | 0.583 | 0.300 | 0.900 | 0.650 |
| v47 feature-token | 0.800 | 0.750 | 0.550 | 0.917 | 0.567 |

Decision:

```text
v47 does not beat v46 on generation sanity. MI and STTC degrade sharply, so the
large-pool downstream run was not promoted.
```

Boundary-confidence ablation:

```text
source pools = v46 large token/no-token pools, 8000 candidates per arm
teacher ensemble = real2000 seed42/43/44
selection = target_conf in [0.35, 0.75], ranked near 0.55
purpose = test whether center token helps when selecting boundary-like
          synthetic samples instead of high-confidence easy samples
```

Filtered pools:

| arm | kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|
| v46 token boundary | 1580 | 348 | 400 | 400 | 36 | 396 |
| v46 no-token boundary | 1505 | 400 | 400 | 400 | 95 | 210 |

Downstream result:

| arm | script test AUROC | script test AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 token boundary | 0.8099 | 0.5484 | 0.7996 | 0.5521 | 0.6752 | 0.3525 |
| v46 no-token boundary | 0.8403 | 0.6266 | 0.8292 | 0.6182 | 0.7093 | 0.4076 |

Interpretation:

```text
Boundary-confidence selection is not a good route in this recipe; it weakens
both arms versus high-confidence filtered v2. More importantly for the causal
question, no-token still beats center-token by a large margin. This is another
negative ablation for attributing the +3pp/+7pp line to the PTB-XL center token.
```

## Added Test: v48 MMD Token And v46 Paired-Delta

v48 tried to repair the token objective by adding an EfficientNet
feature-distribution MMD term:

| arm | CD top1 | HYP top1 | MI top1 | NORM top1 | STTC top1 | overall top1 |
|---|---:|---:|---:|---:|---:|---:|
| v46 token | 0.967 | 0.750 | 0.883 | 1.000 | 0.933 | 0.907 |
| v48 MMD-token | 0.850 | 0.683 | 0.667 | 0.917 | 0.550 | 0.733 |

Decision:

```text
v48 was not promoted. It runs correctly, but it hurts generation sanity versus
v46, especially for MI and STTC.
```

The final v46 paired-delta test selected the exact pairs where token most
improved target-class probability over no-token:

```text
same ref ECG
same target class
same generation seed
top 400 pairs/class by token_p_target - no_token_p_target
token top1 must equal target class
token p_target >= 0.60
```

Filter result:

| arm | raw pairs | filtered kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v46 paired-delta token | 2000 | 1759 | 392 | 264 | 334 | 392 | 377 |
| v46 paired-delta no-token | 2000 | 1069 | 375 | 74 | 219 | 293 | 108 |

Downstream:

| arm | script test AUROC | script test AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 paired-delta token | 0.8472 | 0.6222 | 0.8241 | 0.5936 | 0.6886 | 0.3677 |
| v46 paired-delta no-token | 0.8597 | 0.6612 | 0.8260 | 0.6030 | 0.7092 | 0.4063 |

Interpretation:

```text
This is a strong negative causal result. The center token improves generation
confidence and filter acceptance on matched pairs, but those accepted samples
are worse for downstream EfficientNetV2 self-distillation than the matched
no-token samples.

Therefore the current evidence says:
  ECGTwin synthetic/self-distillation can produce the low-resource gain.
  The current PTB-XL center token does not explain that gain.
```

## Added Test: v46 Class-Oracle Hybrid

The paired-delta test showed that token utility is not uniform by class. In the
count-matched v46 downstream runs, center-token samples were better for
CD/HYP/MI, while no-token samples remained better for NORM/STTC.

To give center-token conditioning the most favorable downstream test without
changing the training recipe, a class-oracle hybrid pool was built:

```text
CD/HYP/MI  <- v46 center-token count-matched filtered pool
NORM/STTC  <- v46 no-token matched filtered pool
total      = 3734 synthetic ECGs
student    = same self-distillation v2 recipe
```

Result:

| arm | checkpoint metric | script test AUROC | script test AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| v46 no-token matched | AUROC | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| v46 class-oracle hybrid | AUROC | 0.8561 | 0.6511 | 0.8286 | 0.6045 | 0.7220 | 0.4043 |
| v46 class-oracle hybrid | AUPRC | 0.8517 | 0.6416 | not run | not run | not run | not run |

Interpretation:

```text
The class-oracle hybrid is the best token-aware custom-test variant so far:
  +0.31pp AUROC / +1.07pp AUPRC versus v46 no-token on the custom test.

It still misses the requested +2pp/+2pp center-token-over-no-token criterion.
It also degrades fold10 and PN2021 when using the AUROC-selected checkpoint.
The AUPRC-selected rerun did not improve the custom AUPRC enough to justify
formal fold10/PN2021 evaluation.
```

This makes the final causal conclusion unchanged:

```text
Current center-token conditioning changes generation and filtering behavior, and
can help selected classes, but it has not been validated as the cause of the
AUROC +3pp / AUPRC +7pp downstream gain.
```

## Added Test: Hard-Label Trust For Class-Oracle Token Samples

The next hypothesis was that center-token samples may be clean enough to use
their synthetic hard labels directly. This is stricter than the default v2
self-distillation setup, which deliberately does not trust synthetic hard labels.

Recipe:

```text
pool = v46 class-oracle hybrid
use_synth_hard_labels = true
synth_distill_weight = 0
real_distill_alpha = 0
soft_loss_mode = bce_soft
```

Result:

| arm | synth ratio | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-token hard-label, same seed | 1.00 | 0.8574 | 0.6633 | 0.8385 | 0.6394 | 0.7329 | 0.4345 |
| class-oracle token-hard | 1.00 | 0.8693 | 0.6955 | 0.8308 | 0.6320 | 0.7318 | 0.4349 |
| class-oracle token-hard | 0.75 | 0.8682 | 0.6961 | 0.8306 | 0.6443 | 0.7313 | 0.4302 |
| class-oracle token-hard | 1.25 | 0.8583 | 0.6676 | not run | not run | not run | not run |

Interpretation:

```text
The hard-label route is the strongest custom-test token result so far:
  +1.20pp AUROC / +3.22pp AUPRC versus matched no-token hard-label.

It still misses the requested +2pp AUROC criterion and does not generalize to
fold10 or PN2021. Increasing synth_ratio to 1.25 is harmful, while 0.75 preserves
custom AUPRC but does not repair external metrics.

Therefore hard-label token trust is not the final proof. It is a useful clue:
token samples carry class-specific signal, but the training objective needs an
external-alignment or validation-aware gate before thesis promotion.
```

## Added Test: Token-Hard Init Plus Real2000 Fine-Tune

The hard-label route was then repaired with a conservative real-only fine-tune:

```text
token route:
  init = e18 class-oracle token-hard
  fine-tune = real2000 only, lr=1e-4

matched no-token route:
  init = e21 no-token hard-label
  fine-tune = same real2000 only recipe
```

Result:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 no-token soft self-distill | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| token-hard -> real2000 FT | 0.8735 | 0.7013 | 0.8452 | 0.6455 | 0.7346 | 0.4281 |
| no-token-hard -> real2000 FT | 0.8669 | 0.6828 | 0.8479 | 0.6506 | 0.7370 | 0.4450 |

Interpretation:

```text
This is the first improved center-token route that clears +2pp AUROC and +2pp
AUPRC versus the naked v46 no-token soft self-distillation baseline:
  +2.05pp AUROC / +6.09pp AUPRC on custom seed42.

But it is not a strict matched-control win. Once no-token is allowed the same
hard-label pretrain plus real2000 fine-tune recipe, token improves custom by
only +0.66pp AUROC / +1.85pp AUPRC, and no-token remains better on fold10 and
PN2021.
```

## Artifact Paths

```text
A-fair:
  /root/autodl-tmp/graduate_project/method_a_real2000_seed42_ckpt_auprc_p50

Token C0/C1/C3/D1:
  /root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42
  /root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_auprc_real_finetune_seed42
  /root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_real_latenthull_at_M10_n1000_seed42
  /root/autodl-tmp/graduate_project/self_distill_d1_teacher_c3_init_c0_synth20k_seed42

No-token C0/C1/C3/D1:
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthpretrain_auprc_real_finetune_seed42
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthpretrain_real_latenthull_at_M10_n1000_seed42
  /root/autodl-tmp/graduate_project/self_distill_d1_teacher_no_token_c3_init_no_token_c0_synth20k_seed42

Filtered v2 same-preprocessing scale-up:
  /root/autodl-tmp/graduate_project/self_distill_v2_e4_real2000_ens3_filtered2000_gamma03_scratch_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_e4_vanilla_real2000_ens3_filtered2000_gamma03_scratch_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_e5_real2000_ens3_filtered4000_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e5_vanilla_real2000_ens3_filtered4000_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e6_hybrid_hyp_token_other_vanilla_filtered4000_gamma03_scratch_seed42_auroc

v45 generation-side style probe:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v45_contrast_large_ningbo_20260504/style_probe_full1000/style_probe_summary_samples_npz.json

v46 PTB-XL contrastive token:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v46_contrast_recon_sem_steps4000_20260504
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/v46_large_20260504/token/ptbxl_source/samples.npz
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/v46_large_20260504/no_token/ptbxl_source/samples.npz
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_ptbxl_contrast_seed42/synth_v2_filtered_top4000_gamma03.npz
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_no_token_matched_seed42/synth_v2_filtered_top4000_gamma03.npz
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_ptbxl_contrast_countmatched_seed42/synth_v2_filtered_countmatched3734_gamma03.npz
  /root/autodl-tmp/graduate_project/self_distill_v2_e7_v46_contrast_filtered4000_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e8_v46_no_token_matched_filtered3734_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e9_v46_token_countmatched3734_gamma03_scratch_seed42_auroc

v47 feature-contrast token:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v47_feature_contrast_steps4000_20260504
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/v47_sanity_20260504/token/ptbxl_source/summary.json

v46 boundary-confidence self-distillation:
  /root/autodl-tmp/graduate_project/self_distill_v2_boundary_v46_token_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_boundary_v46_no_token_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_e10_v46_token_boundary1580_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e11_v46_no_token_boundary1505_gamma03_scratch_seed42_auroc

v48 MMD token:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v48_feature_mmd_steps4000_20260504
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/v48_sanity_20260504/token/ptbxl_source/summary.json

v46 paired-delta self-distillation:
  /root/autodl-tmp/graduate_project/paired_delta_v46_top400x5_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_paired_delta_v46_token_top400x5_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_paired_delta_v46_no_token_top400x5_seed42
  /root/autodl-tmp/graduate_project/self_distill_v2_e12_v46_paired_delta_token1759_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e13_v46_paired_delta_no_token1069_gamma03_scratch_seed42_auroc

v46 class-oracle hybrid:
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_class_oracle_hybrid_seed42/synth_v2_filtered_class_oracle_cd_hyp_mi_token_norm_sttc_notoken_gamma03.npz
  /root/autodl-tmp/graduate_project/self_distill_v2_e15_v46_class_oracle_hybrid3734_gamma03_scratch_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e17_v46_class_oracle_hybrid3734_gamma03_scratch_seed42_auprc

v46 class-oracle hard-label trust:
  /root/autodl-tmp/graduate_project/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e20_v46_class_oracle_hardlabel_r125_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e21_v46_no_token_hardlabel_r10_seed8042_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e22_v46_class_oracle_hardlabel_r075_seed42_auroc

token-hard plus real2000 fine-tune:
  /root/autodl-tmp/graduate_project/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc
  /root/autodl-tmp/graduate_project/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc
```
