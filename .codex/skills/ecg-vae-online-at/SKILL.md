---
name: ecg-vae-online-at
description: Analyze or change ECGTwin VAE-LHAT geometry, contraction, gradients and diagnostics for traditional SimCLR or single-stage JSD experiments.
---

# ECG VAE-LHAT Online AT

Follow AGENTS and the selected method, not a universal LHAT default.

## Resolve the Actual Recipe

1. Start at the experiment's method selector under `configs/train/methods/`.
2. Follow its actual AugMix/VAE/LHAT resource references and inspect
   `core/lhat.py`, `core/methods/registry.py` and the relevant callers.
3. Read matching evidence-registry entries before citing results.

Both locked traditional directions use exact-label non-self M=20, hull lambda=1,
standardized L2 epsilon=12 and one attack step, but their contracts differ:

| Direction | LHAT resource | Contraction / stages |
|---|---|---|
| SimCLR | `configs/train/lhat.yaml` | preflip grid v1; linear clean/hard endpoint correction; source-logit anchor in Stage1, no Stage2 teacher |
| R18 JSD | `configs/train/lhat_pure_delta_nondecreasing.yaml` | pure-delta grid v3 including t=0; constant clean-anchor residual; single-stage, no source anchor |

R18 also selects no-clean-mix AugMix and replaces 0.20 of clean supervised loss;
do not substitute SimCLR resources or add a stage. The selected YAML/code owns
the exact grid, margins and weights. Other registered variants are explicit
ablations, not implicit fallbacks.

## Matched Experiment Discipline

- Match backbone, source, K500 identities, seed, optimizer/exposure budget,
  checkpoint policy, mapping and evaluation view; change one mechanism at a time.
- Attribute VAE gain only using a matched no-VAE control, not the full method's
  difference from direct fine-tuning.
- Never select geometry/checkpoints/per-center settings from heldout target
  labels or the full target-center class distribution.
- Use tracked YAML and the single launcher; dry-run and resource-check before
  authorized compute. An analysis request does not authorize a rerun.

## Diagnostics and Interpretation

Read available artifacts first. Report missing diagnostics rather than inventing
them or silently launching new validation.

- Four BCE baselines: raw clean, decoded anchor, initial uniform hull, final hard.
  They separate reconstruction error, initial movement and optimization gain.
- Raw search: objective gain, atk_init/atk_anchor, standardized norm use,
  raw-search ASR, positive-hide and negative-add, with denominators.
- Contract: acceptance, selected t, BCE gain, contracted-view ASR, valid and
  preserving path counts. ASR near zero may be intended by a no-new-flips contract.
- Decode quality: invalid/flatline/low-variance rates and maximum absolute mV.
- Outcomes: registered clean and PN2021-C per-center/per-class AUROC/AP;
  source floor or internal validation only when that protocol actually provides it.

Healthy attack diagnostics do not prove downstream AP benefit. A large
raw-clean/decoded-anchor gap implicates reconstruction or the bridge; weak
initial/final gain implicates optimization or the feasible set. Report invalid
decodes as a correctness issue; repair only within an authorized change.
Preserve finite checks, train-only standardization and explicit clean fallback.
Attack search must not populate decoder/classifier parameter gradients;
supervised classifier updates retain their gradients.

Use `model-eval` for tables and `reproducibility-check` for run identity.
Read [historical evidence](references/historical_optimization_evidence.md) only
for legacy experiments; read [literature](references/literature_and_repos.md)
only for related-work sources. Neither changes the active recipe.
