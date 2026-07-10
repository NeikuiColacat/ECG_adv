# Mainline Protocol Integrity Repair Design

**Date:** 2026-07-10

**Status:** Approved by the user (方案 B)

## Goal

Make the managed EfficientNet and ECGFounder VAE-LHAT/AugMix mainlines safe to
launch and faithful to the locked three-chain protocol, while preventing K500
lineage mismatches from being presented as matched evidence.

This repair is CPU-only. It does not retrain models, run GPU inference, push the
branch, or touch existing `model/*` and unrelated dirty files.

## Root Causes

1. The EfficientNet adapter emits two arguments that its wrapper neither parses
   nor forwards to the inner runner.
2. `paths.project_root` can point at a sibling checkout, so a dry-run launched
   from this branch can execute code and record Git metadata from another branch.
3. The shared `clean_clean_third` implementation already matches the locked
   topology, but both mainline YAMLs select other modes and their callers use the
   wrong signal-space/order.
4. `latent_weight_cap` is advertised but unused; the existing explicit chain
   weights already provide the required control.
5. Evidence audits compare declared YAML/ref metadata but do not close the
   identity chain through the initialization checkpoint and actual evaluation
   exclusions. Historical results therefore passed audit despite K500 mismatch.

## Design

### Managed launch boundary

- Parse and forward the two existing EfficientNet arguments at both wrapper
  boundaries. Do not add another adapter or compatibility layer.
- Derive the project root from the checkout containing the experiment config.
  A local config that explicitly names another checkout is rejected instead of
  being silently followed.

### Locked topology

Reuse the existing `clean_clean_third` core:

```text
chain 1: clean anchor -> official corruption
chain 2: clean anchor -> official corruption
chain 3: raw VAE-LHAT adversarial waveform -> no extra corruption
mix -> global z-score -> classifier
```

- EfficientNet keeps normalized tensors for PGD optimization/diagnostics, but
  supplies the existing raw decoded anchor/adversarial tensors to AugMix. In the
  locked mode it trains the mixed view once and does not enqueue a second,
  independent VAE stream.
- ECGFounder converts raw 1000-point signals to raw 5000-point signals before
  corruption and mixing. Global z-score remains at the model boundary.
- Delete `latent_weight_cap` and its unused helper surface. Explicit
  `chain_weights` remains the only chain-mass control.
- Record the signal-space contract so checkpoints from the old normalized-first
  protocol cannot resume into the repaired protocol.

### Evidence and K500 identity

- Keep absolute historical metrics as descriptive observations.
- Mark the affected EfficientNet comparative claim and ECGFounder supporting
  entry `deprecated`; set paper use to `prohibited_protocol_invalid` and remove
  wording that calls the comparison matched.
- Reuse existing ref seed/hash metadata. For a paper-safe K500 mainline:
  - a target-adapted initialization checkpoint must expose the same K500 identity
    as the current run (or be explicitly source-only);
  - the actual evaluation exclusion identity must match the run's K500 identity;
  - active registry claims must agree with their recorded evaluation artifacts.
- Reject a different target K500 initialization rather than silently expanding
  the protocol beyond K=500.

## Compatibility

- Old normalized-first checkpoints/results remain historical artifacts but are
  not resumable or valid evidence for the repaired protocol.
- New experiments require a new run ID and later GPU retraining.
- No schema-wide `protocol_invalid` enum is added; existing `deprecated` plus an
  explicit paper-use reason is the smallest compatible representation.

## Verification

- Red/green CPU tests for wrapper parsing/forwarding and checkout binding.
- Deterministic topology tests for two corrupted clean chains plus one
  uncorrupted VAE chain.
- Runner tests proving raw-before-corruption and ECGFounder 5000-before-
  corruption ordering.
- K500 mismatch tests at evaluation and registry boundaries.
- Managed dry-runs, registry audit, targeted pytest, workspace audit, and Git
  artifact guard. No GPU command is part of this repair.
