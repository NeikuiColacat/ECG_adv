---
name: model-eval
description: Standardize ECG_adv_Gen model evaluation commands, metric naming, output reports, center aggregation, all-zero handling, and baseline comparisons.
---

# Model Eval

Use this when evaluating EfficientNet1DV2, ECGFounder, DeepECG, VAE-only
adaptation, direct fine-tuning, PN2021 cross-center results, or paper tables.

## Required Evaluation Metadata

- Model family and checkpoint path.
- Training/adaptation config and seed.
- Dataset and center list.
- PN2021 mapping version and cache version.
- Ref-exclusion rule, including K500 IDs when relevant.
- Metric view: all-zero kept vs drop all-zero.
- Macro AUROC and macro AUPRC, plus per-center and per-class metrics when
  available.

## Workflow

1. Find the canonical evaluation entry point or managed config.
2. Confirm class order, mapping version, center list, and all-zero handling.
3. Run or specify the smallest valid evaluation command.
4. Save outputs to a date/config-named directory.
5. Summarize deltas against direct fine-tuning and no-adaptation baselines in
   percentage points.

## Stop Conditions

- The evaluation includes target-center reference samples that should be
  excluded.
- The mapping version or all-zero handling is missing.
- The reported metric cannot be linked to a manifest, config, and checkpoint.
