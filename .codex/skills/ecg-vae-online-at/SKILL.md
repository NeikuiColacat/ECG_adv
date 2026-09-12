---
name: ecg-vae-online-at
description: Analyze or change ECGTwin VAE latent-hull online adversarial training in ECG_manual_refactor, including candidate geometry, attack strength, contraction, diagnostics, matched ablations, and paper-safe interpretation for EfficientNet1DV2 or ECGFounder.
---

# ECG VAE-LHAT Online AT

Use with `ecg-adv-gen`. This Skill owns VAE-LHAT decisions; it does not replace
the outer training, evaluation, reproducibility, or shared-server Skills.

## Resolve Current Truth First

Read:

1. `AGENTS.md` and the keep manifest.
2. `configs/train/lhat.yaml`, `configs/train/augmix.yaml`, and the selected file
   under `configs/train/methods/`.
3. `core/lhat.py` and its callers.
4. The active evidence registry before citing results.

The implementation and resolved YAML override this summary.

## Locked Development Recipe

```text
target-center K500 real anchor
-> nearest exact-positive-set non-self neighbors
-> optimized softmax hull, M=20, include_anchor=false
-> hull lambda=1.0, standardized L2 epsilon=12, one attack step
-> maximize_multilabel_bce_with_logits
-> t=[0.25,0.5,0.75,1.0] attack-then-contract
-> linear clean/hard endpoint residual correction
```

Stage 1 retains the frozen PTB-XL source-logit anchor. Stage 2 has no teacher
anchor. Any older include-anchor, compatible-label, low-lambda, multi-step,
raw-only, or teacher-enabled setting is an explicit ablation, not a silent
replacement.

## Experiment Discipline

- Change one mechanism at a time and keep backbone, K500 identities, seed,
  optimizer-step budget, checkpoint policy, mapping, and evaluation view matched.
- Pair a full VAE-LHAT run with the matched no-VAE control before attributing a
  gain to VAE-LHAT.
- Do not select geometry, checkpoints, or per-center settings from heldout target
  labels or the full target-center class distribution.
- Run through tracked YAML and the single launcher; dry-run before compute.
- Use `shared-gpu-server-discipline` before any GPU or long evaluation job.

## Required Diagnostics

Report attack generation and downstream performance separately.

- Baselines: raw-clean BCE, decoded-anchor BCE, initial uniform-hull BCE, and
  final hard BCE.
- Raw search: objective gain, `atk_init`, `atk_anchor`, standardized norm use,
  raw-search ASR, positive-hide, and negative-add.
- Contract: acceptance rate, selected `t`, BCE gain, contracted-view ASR, valid
  path count, and preserving-path count.
- Decode quality: invalid rate, flatline/low-variance rate, and maximum absolute
  amplitude.
- Outcomes: K500 validation, PTB-XL/source clean floor, PN2021 Clean and
  PN2021-C per-center/per-class AUROC and AP.

The four BCE baselines separate reconstruction damage, initial hull movement,
and optimized attack gain. Do not collapse them into one `loss_gain` number.

## Interpretation

- Healthy attack diagnostics prove that the attack mechanism moves the model;
  they do not prove downstream AP benefit.
- Contracted ASR near zero can be intentional when the contract forbids new
  flips. Judge it with acceptance, BCE gain, selected `t`, and downstream AP.
- A large raw-clean to decoded-anchor gap points to reconstruction/bridge error,
  not hull optimization.
- Little initial-to-final objective gain points to ineffective weight
  optimization or a saturated feasible set.
- Invalid decodes or implausible amplitudes invalidate performance
  interpretation until the bridge or filtering is repaired.
- Clean AP is not a substitute for PN2021-C AP; report both under the same
  ref-excluded contract.

Treat changes to hull projection, contraction, endpoint residual correction,
candidate policy, attack objective, or teacher anchors as matched ablations.
Do not remove a component merely because its diagnostics look redundant.

## Handoff

Use `model-eval` for result tables and `reproducibility-check` before calling a
run replayable or paper-ready. Read
`references/historical_optimization_evidence.md` only for legacy experiment
history, and `references/literature_and_repos.md` only for related-work sources.
