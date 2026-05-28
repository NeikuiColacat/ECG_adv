---
name: data-prep-validator
description: Validate ECG_adv_Gen ECG data loading, preprocessing, split integrity, sampling rate, lead order, normalization, and label-map contracts.
---

# Data Prep Validator

Use this when touching or reviewing data loading, preprocessing, label mapping,
center splits, cache creation, resampling, normalization, or evaluation inputs.

## Contracts To Preserve

- Super5 class order is `CD, HYP, MI, NORM, STTC`.
- Label mapping source of truth is `scripts/triple_labels/label_schemes.py`.
- PN2021 PTB-XL shards must remain excluded from external-center evaluation.
- K500 reference samples used for adaptation must be excluded from downstream
  target-center evaluation.
- ECGTwin VAE decode output is `(B, 1024, 12)` channels-last and must be
  converted to classifier input `(N, 1000, 12)` PTB-XL lead order at 100 Hz.
- ECGTwin-to-PTB-XL lead index conversion is `[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]`.

## Workflow

1. Identify the exact input schema, expected sampling rate, lead order, and
   output shape.
2. Verify split rules and leakage guards.
3. Verify mapping version and cache version are recorded when label logic
   changes.
4. Run or specify the smallest smoke command that proves the changed path.
5. Report any behavior-critical drift explicitly.

## Stop Conditions

- A change silently alters sampling rate, lead order, normalization, split logic,
  label mapping, all-zero handling, or metric aggregation.
- The evidence cannot identify which dataset version, center list, mapping
  version, or K500 IDs were used.
