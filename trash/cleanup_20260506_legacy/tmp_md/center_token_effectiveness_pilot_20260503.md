# Center Token Effectiveness Pilot 2026-05-03

本报告记录 regenerated center prompt token 的第一轮有效性验证。

## Inputs

Token banks:

```text
PN2021 big4:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500/prompt_token_bank.pt

PTB-XL source:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503/prompt_token_bank.pt
```

Synthetic pilot:

```text
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503

centers:
  ningbo, chapman_shaoxing, cpsc_2018, georgia

classes:
  NORM, MI, STTC

n_per_class:
  10

arms:
  vanilla
  target_token
  wrong_center_token
  ptbxl_source_token
```

`cpsc_2018` has no selected MI references, so its pilot only contains NORM/STTC.

## Probe Upgrade

The old center style classifier used 2.5s crop:

```text
/root/autodl-tmp/per_center_style_classifier
test_acc = 0.3780
macro_f1 = 0.3624
ningbo recall = 0.1233
```

A stronger full-10s real-only style classifier was trained:

```text
/root/autodl-tmp/per_center_style_classifier_full10s_max3000
test_acc = 0.5093
macro_f1 = 0.5148
```

Per-center recall:

| center | recall |
|---|---:|
| chapman_shaoxing | 0.4800 |
| cpsc_2018 | 0.4967 |
| cpsc_2018_extra | 0.5600 |
| georgia | 0.6567 |
| ningbo | 0.3067 |
| ptb | 0.7170 |
| st_petersburg_incart | 0.8750 |

This full-10s probe is still not thesis-final because it does not yet store
record ids and therefore cannot prove K=500 anchor exclusion, but it is a better
sanity probe than the old 2.5s classifier.

## Digital / Semantic Gate

Passed samples:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| ningbo | 14/30 | 20/30 | 18/30 | 19/30 |
| chapman_shaoxing | 15/30 | 16/30 | 21/30 | 17/30 |
| cpsc_2018 | 13/20 | 14/20 | 12/20 | 13/20 |
| georgia | 19/30 | 17/30 | 17/30 | 20/30 |

Interpretation:

```text
ningbo:
  target token improves gate pass rate vs vanilla, but wrong/source controls
  are close, so this alone is weak evidence.

chapman_shaoxing:
  target token does not beat wrong-center token.

cpsc_2018:
  target token slightly improves pass rate over vanilla/wrong/source.

georgia:
  target token does not improve pass rate over vanilla/source.
```

## Full-10s Style Probe

Metric is mean `P(expected target center)` from the full-10s real-only center
style classifier.

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.0213 | 0.0121 | 0.0108 | 0.0081 | -0.0092 |
| cpsc_2018 | 0.0862 | 0.1883 | 0.1249 | 0.0999 | +0.0634 |
| georgia | 0.0313 | 0.0807 | 0.0814 | 0.0662 | -0.0007 |
| ningbo | 0.0061 | 0.0155 | 0.0098 | 0.0048 | +0.0056 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.0063 | 0.0139 | 0.0085 | 0.0102 | +0.0037 |
| cpsc_2018 | 0.0825 | 0.2138 | 0.1548 | 0.0919 | +0.0590 |
| georgia | 0.0427 | 0.0831 | 0.0570 | 0.0205 | +0.0262 |
| ningbo | 0.0046 | 0.0127 | 0.0112 | 0.0031 | +0.0015 |

## Verdict

Current status is **PARTIAL**, not fully supported.

Supported signals:

```text
cpsc_2018:
  target token clearly beats vanilla, wrong-center token, and PTB-XL source token
  on full-10s style score in both raw and gated subsets.

georgia:
  target token beats all controls after digital/semantic gating.

ningbo:
  target token is slightly higher than controls, but margin is too small and
  full-10s probe recall for real ningbo is still low.
```

Not supported:

```text
chapman_shaoxing:
  target token does not show a stable advantage.

all centers:
  this pilot is too small for a thesis-level claim.
  no-leak binary probe has not been completed yet.
```

## Proxy C2ST

A lightweight C2ST was run with downsampled waveform logistic regression:

```text
script:
  scripts/ecgtwin_gen/run_prompt_token_c2st_proxy.py

outputs:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/c2st_proxy/c2st_raw.csv
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/c2st_proxy/c2st_gated.csv
```

Metric is balanced accuracy for distinguishing real target-center ECG from
synthetic ECG. Lower is better; 0.5 means hard to distinguish.

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.670 | 0.720 | 0.762 | 0.630 |
| cpsc_2018 | 0.703 | 0.720 | 0.800 | 0.900 |
| georgia | 0.590 | 0.747 | 0.675 | 0.762 |
| ningbo | 0.757 | 0.765 | 0.731 | 0.787 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.730 | 0.615 | 0.689 | 0.707 |
| cpsc_2018 | 0.667 | 0.764 | 0.782 | 0.666 |
| georgia | 0.549 | 0.743 | 0.657 | 0.660 |
| ningbo | 0.664 | 0.660 | 0.725 | 0.811 |

C2ST interpretation:

```text
chapman_shaoxing:
  gated target-token is less distinguishable than vanilla, but style-probe
  evidence remains weak.

cpsc_2018:
  target-token improves center-style score, but C2ST gets worse than vanilla.
  This suggests target token changes style but may add synthetic artifacts.

georgia:
  target-token improves style score after gate, but C2ST is much worse than
  vanilla. This is not ready for downstream main evidence.

ningbo:
  gated target-token C2ST is similar to vanilla, but style score margin is tiny.
```

## No-Leak Binary Probe And Same-Label C2ST

The stricter no-leak validation was then run:

```text
script:
  scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py

raw output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/noleak_validation_raw

gated output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/noleak_validation_gated
```

Rules:

```text
1. load PN2021 mmap caches with record_ids;
2. exclude each center's K=500 prompt-token anchors;
3. train binary target-vs-other-center probes on real ECG only;
4. match positives and negatives by primary class in NORM/MI/STTC;
5. query synthetic arms after training;
6. run same-label real-vs-synth C2ST.
```

Anchor exclusion and usable class counts:

| center | excluded anchors | NORM | MI | STTC |
|---|---:|---:|---:|---:|
| ningbo | 500 | 4485 | 131 | 10054 |
| chapman_shaoxing | 500 | 1247 | 10 | 2687 |
| cpsc_2018 | 500 | 802 | 0 | 953 |
| georgia | 500 | 1628 | 0 | 4741 |

Real-only binary probe strength:

| center | test balanced accuracy | test AUROC |
|---|---:|---:|
| ningbo | 0.495 | 0.513 |
| chapman_shaoxing | 0.519 | 0.524 |
| cpsc_2018 | 0.555 | 0.607 |
| georgia | 0.586 | 0.617 |

This is much stricter than the 7-way full-10s probe and also much weaker. It
means that under same-label NORM/MI/STTC matching, this lightweight waveform
probe can barely separate the target center from the other big centers.

No-leak binary synthetic query, mean `P(target center)`:

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.461 | 0.559 | 0.631 | 0.605 | -0.072 |
| chapman_shaoxing | 0.477 | 0.525 | 0.406 | 0.539 | -0.014 |
| cpsc_2018 | 0.571 | 0.472 | 0.364 | 0.374 | -0.098 |
| georgia | 0.476 | 0.357 | 0.471 | 0.473 | -0.118 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.385 | 0.581 | 0.605 | 0.541 | -0.024 |
| chapman_shaoxing | 0.463 | 0.549 | 0.393 | 0.544 | +0.005 |
| cpsc_2018 | 0.558 | 0.509 | 0.560 | 0.413 | -0.051 |
| georgia | 0.437 | 0.383 | 0.461 | 0.495 | -0.112 |

No-leak same-label C2ST balanced accuracy. Lower is better:

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.630 | 0.725 | 0.712 | 0.744 |
| cpsc_2018 | 0.729 | 0.700 | 0.786 | 0.714 |
| georgia | 0.700 | 0.843 | 0.743 | 0.771 |
| ningbo | 0.723 | 0.742 | 0.780 | 0.786 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.793 | 0.580 | 0.550 | 0.633 |
| cpsc_2018 | 0.800 | 0.800 | 0.725 | 0.660 |
| georgia | 0.650 | 0.700 | 0.617 | 0.725 |
| ningbo | 0.760 | 0.729 | 0.703 | 0.800 |

Updated decision:

```text
Do not send regenerated v40 tokens into downstream AUROC/AUPRC claims yet.

The full-10s 7-way probe gives exploratory positive signals for cpsc_2018 and
gated georgia, but the no-leak same-label binary probe does not confirm a stable
target-token advantage.

The most conservative thesis statement is:
  regenerated center tokens are trainable and can shift generated ECG
  distributions under some probes, but current evidence does not prove reliable
  target-center style transfer.

Next valid step:
  run a stronger frozen EfficientNet feature binary probe, still with record-id
  anchor exclusion and same-label matching.
```

## Frozen EfficientNet Feature No-Leak Probe

The no-leak validation was repeated with frozen EfficientNet1DV2 super5
penultimate features:

```text
script:
  scripts/ecgtwin_gen/run_prompt_token_noleak_style_validation.py

feature backend:
  --feature_backend efficientnet
  --feature_ckpt /root/autodl-tmp/triple_labels/super5/best_model.pt

outputs:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/noleak_validation_raw_effnet
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503/noleak_validation_gated_effnet
```

Real-only binary probe strength improved compared with waveform features:

| center | test balanced accuracy | test AUROC |
|---|---:|---:|
| ningbo | 0.589 | 0.602 |
| chapman_shaoxing | 0.561 | 0.578 |
| cpsc_2018 | 0.663 | 0.714 |
| georgia | 0.627 | 0.646 |

EfficientNet feature binary query, mean `P(target center)`.

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.466 | 0.529 | 0.530 | 0.540 | -0.011 |
| cpsc_2018 | 0.474 | 0.414 | 0.361 | 0.323 | -0.061 |
| georgia | 0.525 | 0.496 | 0.488 | 0.491 | -0.029 |
| ningbo | 0.515 | 0.598 | 0.582 | 0.572 | +0.017 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token | target-best-control |
|---|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.491 | 0.535 | 0.524 | 0.537 | -0.002 |
| cpsc_2018 | 0.435 | 0.430 | 0.426 | 0.341 | -0.004 |
| georgia | 0.559 | 0.494 | 0.485 | 0.538 | -0.064 |
| ningbo | 0.491 | 0.579 | 0.584 | 0.577 | -0.005 |

EfficientNet feature same-label C2ST balanced accuracy. Lower is better:

Raw synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.583 | 0.732 | 0.679 | 0.822 |
| cpsc_2018 | 0.500 | 0.729 | 0.771 | 0.771 |
| georgia | 0.614 | 0.600 | 0.743 | 0.714 |
| ningbo | 0.637 | 0.591 | 0.771 | 0.600 |

Gated synthetic:

| center | vanilla | target_token | wrong_center_token | ptbxl_source_token |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.493 | 0.680 | 0.529 | 0.650 |
| cpsc_2018 | 0.640 | 0.780 | 0.710 | 0.800 |
| georgia | 0.342 | 0.700 | 0.633 | 0.725 |
| ningbo | 0.620 | 0.714 | 0.773 | 0.714 |

Final pilot verdict:

```text
NOT SUPPORTED as a reliable center-style transfer method yet.

The regenerated v40 center tokens are trainable and sometimes move simple
style scores, but under no-leak same-label validation they do not consistently
beat vanilla, wrong-center, or PTB-XL source-token controls.

The strongest no-leak feature probe, frozen EfficientNet, still fails the
acceptance criterion:
  P(target | target-token) > all controls
  and C2ST(real target, target-token) <= vanilla/wrong/source.

Do not scale regenerated v40 token generation or use it for downstream
AUROC/AUPRC claims.

Recommended next research direction:
  change the token training objective itself rather than only scaling samples.
  Candidate: add a differentiable feature-space center-style loss using a
  frozen real-only center probe, plus a semantic preservation loss from the
  frozen super5 classifier.
```
