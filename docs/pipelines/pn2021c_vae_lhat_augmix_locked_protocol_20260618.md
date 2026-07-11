# PN2021-C VAE-LHAT AugMix Locked Protocol, 2026-06-18

This document records the clarified experimental protocol for the current
PN2021-C robustness exploration. It is the source of truth for the next
implementation and experiment runs unless the user explicitly changes it.

## Scope

The current phase is method exploration, not final blind-test reporting. We are
allowed to use the four PN2021 target centers as a development scoreboard to
judge whether the method direction works. Results from this phase must be
labeled exploratory if reported externally.

The mechanically supported claim is
`known_family_corruption_robustness`: training and evaluation use the same five
declared corruption families. Depth-2/3 rows are known-family compositions.
Describe results only as robustness to corruption families represented during
training; this protocol does not support held-out-family conclusions.

The target method remains VAE-LHAT plus AugMix: ECGTwin VAE latent hard-sample
search must be trained into the classifier parameters through an AugMix-style
multi-chain training topology. Gains from standalone preprocessing, post-hoc
selection, single-chain raw corruption, or raw-supervised auxiliary branches do
not count as the main method.

## Locked Data And Evaluation Protocol

- The matched EfficientNet matrix uses the same deterministic K500 internal
  train/validation split for A0/A2/A3/A4/A5.
- Select `best_model.pt` by K500-internal macro AUPRC subject to the declared
  PTB-XL source-floor constraint.
- Do not use PN2021 heldout labels, PN2021-C operator results, full target
  distributions, or oracle epoch selection.
- Exclude the K500 record ids from the corresponding PN2021 target-center
  evaluation set.
- Treat the four target centers as an exploratory development scoreboard:
  `ningbo`, `chapman_shaoxing`, `cpsc_2018`, and `georgia`.
- Do not use non-K500 target-center label distribution information for class
  weights, sampling rules, operator-specific tuning, or center-specific recipe
  changes.
- PTB-XL/source clean performance is the shared source-floor gate.
- The canonical EfficientNet VAE arms A3/A4/A5 use exact-label, non-self latent
  partners with `include_anchor=false`. Before epoch 1 they must emit the
  deterministic exact-partner eligibility manifest; A0/A2 remain non-VAE arms
  and do not load latent assets or create that manifest.
- The independent F-004 rho sweep uses seed `20260601` from the fresh
  `paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets` family. Its
  v3 topology freezes label/mix modes, hull learning rate, and neighbor search
  geometry; only `target_adv_fraction` varies across 0/0.25/0.5.

## Backbone Training Protocol

EfficientNet1DV2 and ECGFounder must be compared under aligned full fine-tuning
semantics.

EfficientNet1DV2:

- Use the PTB-XL Super5 source model as the source initialization.
- K500 direct and VAE-LHAT/AugMix variants use full fine-tuning.
- Do not introduce a separate preprocessing frontend for the main method.

ECGFounder:

- Use the official `12_lead_ECGFounder.pth` checkpoint.
- Add a Super5 five-class head and run PTB-XL Super5 full fine-tuning with all
  layers trainable.
- Run K500 full fine-tuning with all layers trainable.
- Run VAE-LHAT and VAE-LHAT plus AugMix with all layers trainable.
- Use `best_model.pt` style full-model checkpoints for this mainline.
- Old ECGFounder frozen-feature, head-only, two-head, `residual_adapter`,
  `base_head`, and `best_head.pt` routes are historical or ablation-only and
  must not appear as the main ECGFounder baseline or main method.

## Optional Backbone Replication

ECGFounder A0/A3/A5 is an `optional_backbone_replication`, not part of the
default command matrix. Activate it only after an ECGFounder-specific matched
initialization, split, selection, and budget contract exists. EfficientNet-only
results must remain scoped to the evaluated EfficientNet protocol.

The matched ECGFounder replication must keep the following fixed relative to
the EfficientNet1DV2 candidate unless a run is explicitly labeled as an
ablation:

- K500 seed and ref-exclusion policy.
- PN2021-C severity profile and corruption order.
- Three-chain AugMix topology, with two raw corruption chains and one VAE-LHAT
  adversarial waveform chain.
- No stabilizer, repair frontend, raw-supervised branch, operator oracle, or
  heldout selector.
- A backbone-specific matched checkpoint-selection policy declared before runs.

Required cross-backbone reporting rows:

- Direct K500 baseline for each backbone.
- VAE-LHAT online AT only, without AugMix.
- VAE-LHAT plus locked three-chain AugMix.
- ECGFounder trainable-scope ablations if full fine-tuning underperforms:
  fullFT, dense-only, last-N-stage partial unfreeze, and LoRA if implemented.

Partial-unfreeze and LoRA ECGFounder runs are trainable-scope ablations, not a
replacement for the fullFT mainline unless the user explicitly changes the
protocol. If they improve robustness, report them as evidence about
foundation-model adaptation sensitivity and then decide whether to promote a
new frozen recipe before final paper-style reruns.

## Input And Corruption Order

All corruptions and training augmentations are applied before z-score
normalization. Do not apply PN2021-C or AugMix operators to already z-scored
caches for the main protocol.

EfficientNet1DV2 input order:

```text
raw ECG
-> canonical PTB-XL lead order
-> 100 Hz / 1000 samples
-> optional corruption or AugMix/VAE-LHAT augmentation
-> per-sample global z-score
-> EfficientNet1DV2
```

ECGFounder input order:

```text
raw ECG
-> canonical PTB-XL lead order
-> 100 Hz / 1000 samples
-> 500 Hz / 5000 samples by interpolation
-> optional corruption or AugMix/VAE-LHAT augmentation
-> per-sample global z-score
-> ECGFounder
```

This order also applies to PN2021-C robustness evaluation. For the main
PN2021-C table, the corruption is applied after the 100 Hz bottleneck and, for
ECGFounder, after the 500 Hz interpolation, but before z-score.

Native-raw-first corruption may be kept as a diagnostic ablation, but it is not
the current main protocol unless the user changes this document.

For ECGTwin VAE decoded samples:

- Decode in ECGTwin order and raw waveform scale.
- Reorder leads with `ECGTWIN_TO_PTBXL_INDICES`.
- Resample to the backbone working waveform length before z-score:
  - EfficientNet1DV2: 100 Hz / 1000 samples.
  - ECGFounder: 500 Hz / 5000 samples after the 100 Hz bottleneck path.
- Do not apply z-score before AugMix chain mixing.

## PN2021-C Severity Protocol

First evaluate official severity control before custom stress profiles.

- Start with official severity `5` as the "max official" setting.
- Evaluate the five operators separately:
  - `powerline_noise`
  - `emg_noise`
  - `baseline_wander`
  - `baseline_shift`
  - `random_leads_masking`
- Use the same severity setting across models and centers.
- Do not tune severity per model, per center, per operator result, or per
  method variant during the main comparison.
- Accept official severity `5` if at least one of the two primary backbones
  reaches roughly 10 pp clean-to-corrupted degradation on the four-center,
  five-operator mean. In this protocol, "roughly 10 pp" means an 8-12 pp mean
  drop band in either macro AUROC or macro AUPRC. If one backbone reaches this
  band, do not create a custom profile just because the other backbone drops
  less.
- If neither EfficientNet1DV2 nor ECGFounder reaches the roughly 10 pp band
  under official severity `5`, design one global custom profile afterwards.
- A custom profile must be justified with ECG waveform visualization,
  amplitude/RMS/frequency or lead-mask statistics, and physiological sanity
  checks. It must not be tailored to one model, one center, or one method.

## AugMix Topology

The main method uses the classic three-chain AugMix topology. The selected
design is option A from the 2026-06-18 discussion:

```text
chain 1: official corruption chain
chain 2: official corruption chain
chain 3: VAE-LHAT adversarial waveform chain
```

Chain 3 is the VAE-LHAT adversarial waveform itself. Do not add another
official corruption chain on top of the VAE adversarial waveform in the main
method.

Mixing and loss:

- Mix the three chains with AugMix-style mixture weights.
- Generate augmented waveform views before z-score.
- Apply per-sample global z-score after chain mixing.
- Train with clean BCE plus AugMix-view supervision and consistency.
- JSD or equivalent consistency is allowed as part of AugMix.
- A small augmented-view BCE term is allowed inside the AugMix view training.
- Do not add an independent raw-supervised corruption branch outside AugMix for
  the main method.

The main method must not collapse into `width=1`, `depth=1`, or fixed
single-chain mixing. Such runs are diagnostics or ablations only.

## Agent-Controlled Exploration Knobs

During the exploratory phase, the agent may make bounded adjustments to the
training loss weights and VAE-LHAT adversarial strength without asking the user
for each scalar value, as long as the main protocol above is preserved.

Allowed self-directed exploration:

- AugMix view BCE weight.
- JSD or equivalent consistency weight.
- Relative clean BCE versus AugMix-view loss weight.
- VAE-LHAT PGD steps, latent epsilon, hull lambda, and adversarial branch
  generation frequency.
- AugMix mixture alpha and Dirichlet/Beta mixing parameters.

Boundaries:

- Keep the three-chain AugMix topology.
- Keep chain 3 as the VAE-LHAT adversarial waveform without extra corruption.
- Do not add raw-supervised or raw-corruption consistency branches outside
  AugMix.
- Do not introduce stabilizer35, flat-lead repair, bandpass frontends, or
  post-hoc oracle selectors into the main method.
- Record every explored scalar in resolved config, run card, and summary.
- Prefer small grids or staged adjustments over broad sweeps.

## Excluded Mainline Routes

The following are not the main method:

- ECGFounder frozen encoder or frozen feature extraction.
- ECGFounder head-only K500 tuning.
- ECGFounder residual adapter or two-head route.
- `best_head.pt` checkpoint comparisons.
- `stabilizer35`, flat-lead repair, bandpass frontend, or other matched input
  stabilizer frontends.
- Post-hoc operator-aware oracle selection.
- Last-epoch forcing or fixed last-checkpoint evaluation. Every matched
  EfficientNet arm uses `best_model.pt`, selected by K500-internal macro AUPRC
  subject to the declared PTB-XL source-floor constraint.
- Standalone raw-supervised or raw-corruption consistency branches outside the
  three-chain AugMix topology.

These routes may still be useful as historical evidence, negative controls,
or engineering diagnostics, but they must be labeled as such.

## Metrics And Reporting

For exploration tables:

- Report PN2021 clean ref-excluded metrics.
- Report PN2021-C corrupted absolute AUROC/AUPRC.
- Report clean-to-corrupted drops in pp.
- Report per-center, per-operator, and per-class breakdowns.
- Prefer corrupted absolute AUPRC as the primary robustness utility signal.
- Use clean-to-corrupted drop as a secondary diagnostic, because a small drop
  can be caused by a weak clean baseline.
- Mark clean PN2021 AUPRC drops greater than about 2 pp and PTB-XL source AUPRC
  drops greater than about 2 pp as warnings.

For final paper-style reporting after exploration:

- Freeze the recipe before rerunning.
- Run multiple seeds if compute allows.
- Record confidence intervals or at least seed variability.
- Record git SHA, dirty status, full command, resolved config, seed, K500 refs,
  mapping version/hash, checkpoint policy, and run artifacts.
- Do not present exploratory PN2021-dev choices as a fully blind benchmark.

## Implementation Gates

Future implementation should add or enforce these checks:

- Managed ECGFounder mainline configs must not use frozen feature, head-only,
  residual adapter, or `best_head.pt` paths.
- Managed ECGFounder mainline configs must use the locked full-FT staged path.
- Main PN2021-C configs must reject pre-z-scored corruption inputs.
- Main PN2021-C configs must record the corruption order.
- Main AugMix configs must record that the topology is three-chain and that
  chain 3 is VAE-LHAT adversarial waveform without additional corruption.
- Run records must explicitly state whether a result is exploratory or final.

## Still Open By Design

The following are not fixed yet and should be decided only after the official
severity and first three-chain AugMix exploratory runs:

- Whether an extra blind or semi-blind center set is needed for final paper
  reporting.
