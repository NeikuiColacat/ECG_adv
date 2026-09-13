---
name: model-eval
description: Interpret or evaluate traditional ECG classifiers and PULSE/ECG-R1 results with correct metric inputs, matched cohorts, aggregation and uncertainty.
---

# ECG Model Evaluation

Start from the relevant evidence-registry entry and actual result artifacts.
Requests for progress, explanations or existing results are read-only; run an
evaluation only when requested, through the managed launcher after dry-run.

## Select the Metric Family

- Traditional Super5: macro AUROC and sklearn average precision from raw logits;
  AP is not trapezoidal PR-AUC. Do not introduce sigmoid saturation into ranking.
- Generated-label PULSE/ECG-R1: use registered macro-F1, exact-match and Hamming
  metrics; report parsing failure/truncation handling. Hard labels discard
  fine-grained ranking; token-vocabulary logits are not automatically per-class
  decision scores. Do not treat label-derived AUROC/AP as equivalent to metrics
  from continuous class scores; probability calibration is not required.

## Verify the Comparison

- Identify model/checkpoint, method/training config, source, seed/namespace,
  selection rule, optimizer and record-exposure budgets.
- Match dataset/cache, Super5 class order/mapping hash, centers, K500 exclusions,
  record IDs and corruption views. Keep CPSC2018 + Extra one logical center.
- Use `drop_all_zero` as primary and `all_zero_kept` as audit; name undefined-class
  handling and macro denominators. Follow the registered metric contract.
- Report clean and PN2021-C, per-center/per-class where available; average views
  then centers equally, not by their unequal record counts.
- Name the actual baseline. Locked A1 includes corruption; it is not clean-only.
  Two-stage SimCLR vs single-stage JSD can be a whole-method comparison, not a
  loss-only causal estimate when stages/budgets differ.
- Report deltas in percentage points; a positive point estimate is not proof of
  a significant gain.

## Uncertainty and Evidence

Training repeats use independent seeds as the unit. Paired prediction bootstrap
resamples records jointly across arms/views and within the registered center
strata; it does not measure training-seed variability. Do not count twenty
corruption views as twenty independent experiments.

For new evaluation, preserve config/run/checkpoint hashes and external output
isolation. Missing identity, leaked K500 records or mismatched metric views
prevent a valid comparison; explain the gap instead of inventing a number.
Use `reproducibility-check` for artifact integrity and paper-promotion gates.
