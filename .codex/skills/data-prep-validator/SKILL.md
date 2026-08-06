---
name: data-prep-validator
description: Validate ECG_manual_refactor ECG loading, preprocessing, split integrity, sampling rate, lead order, normalization, label mapping, and K500/ref-exclusion contracts.
---

# Data Prep Validator

Use this when touching or reviewing data loading, preprocessing, label mapping,
center splits, cache creation, resampling, normalization, or evaluation inputs.

## Contracts To Preserve

- Super5 class order is `CD, HYP, MI, NORM, STTC`.
- PN2021 mapping source of truth is
  `configs/data/PN2021_super5_v7.yaml`; report version
  `v7_super5_sjr_rgq_review_20260528` and hash `555ec85d5b51`.
- PTB-XL records must not enter PN2021 target-center evaluation.
- K500 reference samples used for adaptation must be excluded from downstream
  target-center evaluation.
- Canonical classifier-side ECG is raw mV `(B, 1000, 12)` at 100 Hz.
- ECGTwin VAE encode/decode uses `(B, 1024, 12)` channels-last; bridge it
  through `models/vae.py` and `core/lhat.py`, then return to canonical 100 Hz.
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
