# EfficientNet1DV2 500Hz VAE-online-AT Current Closure

Date: 2026-06-02

## Current State

The active 500Hz/v7 EfficientNet1DV2 VAE-online-AT search has not achieved the
goal of beating matched direct target fine-tuning by +2pp AUROC or AUPRC with
the other metric non-decreasing.

The strongest local signal remains CPSC, where loss-gain acceptance improves
AUPRC by about +1.15pp over matched direct40. The same recipe family does not
transfer to Ningbo, and the hard+soft combination also fails Ningbo.

## Verified JSON Results

All rows below are from `eval_result_v7_500hz_exclrefs.json` files and use:

```text
mapping: v7_super5_sjr_rgq_review_20260528
input: 500Hz, 5000 samples = 10 seconds
preprocess: minimal_resample
normalization: per_sample_global
final eval: target-center K500 ref ids excluded
```

| center | method | target AUROC / AUPRC | PTB-XL AUROC / AUPRC |
|---|---|---:|---:|
| cpsc_2018 | direct40 | 0.8633 / 0.6155 | 0.9086 / 0.7749 |
| cpsc_2018 | hard loss-gain | 0.8688 / 0.6270 | 0.9079 / 0.7733 |
| cpsc_2018 | soft loss-gain | 0.8670 / 0.6206 | 0.9082 / 0.7746 |
| cpsc_2018 | hard+soft loss-gain | 0.8679 / 0.6255 | 0.9081 / 0.7737 |
| ningbo | direct40 | 0.8810 / 0.5032 | 0.9075 / 0.7723 |
| ningbo | hard loss-gain | 0.8807 / 0.5018 | 0.9071 / 0.7722 |
| ningbo | soft loss-gain | 0.8818 / 0.5063 | 0.9076 / 0.7750 |
| ningbo | hard+soft loss-gain | 0.8808 / 0.5024 | 0.9078 / 0.7735 |
| chapman_shaoxing | direct40 | 0.8757 / 0.4508 | 0.9084 / 0.7744 |
| chapman_shaoxing | soft loss-gain | 0.8727 / 0.4517 | 0.9080 / 0.7735 |
| georgia | direct40 | 0.8364 / 0.6202 | 0.9057 / 0.7654 |
| georgia | soft loss-gain | 0.8349 / 0.6180 | 0.9028 / 0.7618 |

## Interpretation

Loss-gain policies are not enough:

```text
hard-only:
  best CPSC result, but Ningbo drops below direct40.

soft-only:
  improves Ningbo slightly, but weakens CPSC and hurts Georgia.

hard+soft:
  keeps a CPSC gain over direct40, but remains weaker than hard-only and fails
  Ningbo versus direct40 and soft-only.
```

The PTB-XL source floor is generally acceptable for the CPSC/Ningbo hard/soft
family, so the main blocker is not catastrophic source forgetting. The blocker
is target held-out transfer: K500-internal validation often improves while
ref-excluded PN2021 evaluation does not.

## Closed Branches

Do not expand these exact branches without a materially different mechanism:

```text
plain VAE500 LHAT
K100 / K200 / K1000 plain VAE scaling
10% / 20% target-ratio plain VAE scaling
old NORM/MI/STTC recipe as a global recipe
v7-scope CD/HYP/NORM/STTC recipe
hard/soft/hard+soft loss-gain minor knob family
accepted-only latent AugMix
rank-aware loss with weight 0.05
positive-hide / negative-add objective
direct40-init pure VAE or teacher-soft VAE
ECGTwin1024 latent -> 500Hz classifier diagnostic
ECGTwin-init / ECGTwin-teacher VAE500 as ungated long runs
xresnet1d50 old-recipe transfer
```

## Next Action

For the current goal, the next useful work should not be another small
EfficientNet knob. Use one of these routes:

```text
1. Evidence consolidation:
   build a paper-ready matched-direct table and mark all closed branches.

2. New mechanism only:
   introduce a pre-registered CPSC/Ningbo gate for a materially different
   VAE objective or adaptation/selection framework.

3. External comparison:
   run a fair few-shot/domain-adaptation baseline from docs/tmp_md/method_cmp.md
   if compute and disk allow.
```

The goal remains active because no current method has satisfied the +2pp over
matched direct fine-tuning criterion.
