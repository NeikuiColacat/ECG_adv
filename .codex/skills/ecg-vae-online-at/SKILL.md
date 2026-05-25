---
name: ecg-vae-online-at
description: Optimize ECGTwin VAE latent-hull / VAE-only online adversarial training for PTB-XL super5 and PN2021 cross-center K500 adaptation. Use when tuning latent mixup, PGD attack strength, attack success logging, ECGFounder or EfficientNet1DV2 VAE-only adaptation, AugMix-style latent branches, clean/adv loss scheduling, or paper-safe model selection.
---

# ECG VAE Online AT Skill

Use this skill together with `ecg-adv-gen`. It is a narrow guide for improving
the current mainline:

```text
PTB-XL super5 classifier
-> ECGTwin VAE latent space
-> target-center K500 real anchors
-> same-label / compatible latent-hull online adversarial training
-> PN2021 ref-excluded cross-center evaluation
```

Do not turn the project into a generic adversarial-robustness benchmark. The
goal is better external-center AUROC/AUPRC from realistic VAE latent
augmentation.

## Integrity Rules

- Final method selection must not use full target-center test labels,
  target-center class distribution, or per-center manual tuning.
- Exploration may inspect heldout feedback, but final runs must use a frozen
  global recipe selected from source data plus the random K500 target subset
  and its internal validation only.
- Keep PN2021 mapping version, random seed, K500 record IDs, attack config, and
  model-selection rule in every run artifact.
- Compare against direct K500 fine-tuning for the same backbone, mapping, seed
  policy, and ref-excluded evaluation split.

## Required Metrics

Log these per epoch and per selected checkpoint:

- Clean and adversarial macro AUROC/AUPRC.
- Clean BCE, adversarial BCE, `loss_gain = adv_bce - clean_bce`, plus median,
  p90, and normalized gain.
- `atk_init`: final adversarial latent versus deterministic initial hull point.
- `atk_anchor`: final adversarial latent versus original real anchor.
- Clean-correct conditioned multilabel ASR, split into positive-hide,
  negative-add, and sample-anyflip when feasible.
- Latent norm utilization, decoded waveform NaN/Inf, flatline, amplitude, and
  simple lead-consistency sanity rates.
- PTB-XL/source clean floor and PN2021 per-center/per-class deltas.
- `best_clean`, `best_robust_val`, and `last`, not only final epoch.

## Attack Strength Diagnosis

Use attack success as a control signal, not as the objective.

- `atk_init < 0.6` or non-positive `loss_gain`: PGD is too weak. Increase
  steps, step size, epsilon/hull radius, or add random start before increasing
  adversarial sample weight.
- `atk_init` high but `atk_anchor` low or negative: the latent-hull initializer
  is making easier samples than the original anchor. Reduce `hull_lambda`, make
  the mix more anchor-local, or include the anchor in the hull.
- `atk_anchor > 0.8` early and clean/source metrics drop: attack or adv stream
  is too strong. Lower `adv_weight`, warm it up, reduce epsilon/lambda, or stop
  earlier.
- Good attack success with worse K500 validation means the adv branch is likely
  overweighted or off-manifold. Add/strengthen clean anchor loss, source-logit
  anchor, or hard invalid-decode filtering before increasing PGD.
- Robust overfitting is expected. Pick checkpoints by validation robust/clean
  score with a source floor, not by last epoch.
- A practical training ASR target is roughly 30-70%. Do not optimize for the
  highest ASR.

Current 2026-05-24 evidence:

- Hard-BCE anchor sampling improved ECGFounder modestly.
- Including the anchor fixed `atk_anchor`, but the full 20-epoch run performed
  worse unless adversarial pressure is reduced.
- Therefore the next global experiment should be hard-BCE anchors plus
  include-anchor with lower or warmup adversarial weight, not a stronger PGD
  by default.

Current 2026-05-25 ECGFounder evidence:

- `warm20 + hull_lambda=0.05 + include-anchor` remains the best simple global
  recipe, but its four-center mean gain is only about `+0.57 AUROC pp /
  +1.20 AUPRC pp`; most of the gain comes from CPSC.
- Heldout-oracle checkpointing for the same recipe still does not reach the
  target, so the main blocker is not K500 validation checkpoint selection.
- `target_logit_anchor_weight=0.5` is too conservative; `0.1` is more balanced
  but still reduces the CPSC gain and does not improve the mean.
- Fixed direct-head anchor scoring (`base_hard_bce`, high power) behaves almost
  the same as current-head hard-BCE. It does not unlock non-CPSC centers.
- Pairwise ranking loss for rare classes and direct linear-head updates with
  L2 anchoring were tested; both were stable enough, but neither produced the
  required mean gain. Avoid repeating these exact sweeps unless changing another
  factor materially.
- Per-class diagnosis: CPSC gains mainly come from STTC AUPRC; Chapman/Ningbo
  macro AUPRC is bottlenecked by rare HYP/MI ranking. A useful next idea should
  directly address rare-class cross-center ranking without collapsing common
  classes.
- A K500-internal validation multi-candidate selector over direct, sourcebase,
  k500base, warm20, target-anchor, linear-L2, base-hard, and rankrare heads
  reached about `0.9131/0.6289` four-center mean, i.e. `+0.90 AUROC pp /
  +3.21 AUPRC pp` over ECGFounder direct K500. Logit-space selection raised
  AUPRC slightly to about `+3.30 pp` but lowered AUROC. This is a useful
  paper-safe selection module, but it is still short of the requested `+2/+5`.
- Adding a mild source-aware classwise selector (`val_auprc + 0.2 *
  source_fold10_auprc`) improved the selected mean to about `0.9138/0.6303`,
  i.e. `+0.96 AUROC pp / +3.35 AUPRC pp`. A pure validation/source harmonic
  mean preserved source better but hurt CPSC and dropped the target mean.
- Held-out oracle classwise selection over the same candidate pool is only about
  `+1.17 AUROC pp / +4.02 AUPRC pp`, so further progress requires stronger
  VAE-online training candidates, not just better selection among the current
  heads.
- Expanding the selector with older v6 global candidates did not improve the
  K500-val selected mean, and in CPSC it overfit to worse candidates. Avoid
  adding more historical candidates unless they are genuinely new global
  recipes.
- A sourcebase gentle recipe (`include-anchor + hull_lambda=0.05 + adv 0->20`)
  showed reasonable attack success (`atk_init` roughly 0.5-0.7) but did not add
  useful selector diversity; it mainly recovered from weak sourcebase baseline
  and stayed below direct K500 for several centers.
- ECGFounder K500-base residual-adapter with a mid-strength adv feature window
  (`adv_boundary_prob=[0.15,0.85]`) was tested on the hard-BCE
  include-anchor warm20/lam0.05 recipe. It kept attack success healthy but did
  not improve the four-center mean: `0.9103/0.6082`, about `+0.61/+1.14 pp`
  over direct, `+0.04/-0.06 pp` versus old hard-BCE, and `-0.03/-0.29 pp`
  versus rankrare. It helped Georgia (`+0.25/+0.29 pp` vs old hard-BCE) but
  hurt CPSC (`-0.07/-0.52 pp`; worse versus rankrare). Do not use it as a
  standalone ECGFounder breakthrough; at most keep it as a selector-diversity
  candidate. A source-base boundary run without `init_base_head_from_k500_root`
  is not comparable to the K500-base ECGFounder mainline.
- A larger ECGFounder residual adapter (`hidden=512`, dropout 0.10) with
  `source_target` pos_weight, fixed rare-class target/adv sampling
  (`HYP=2,MI=4`), and rank loss (`CD/HYP/MI/STTC`, weights `1/2/3/1`) was
  tested as a different rare-ranking candidate. It produced a strong CPSC
  single-center head (`0.8924/0.6922`, `+2.94/+7.03 pp` over direct) but
  K500-val selected the baseline for Ningbo/Chapman/Georgia, so the four-center
  standalone mean was only `0.9115/0.6144` (`+0.74/+1.76 pp`). Adding this
  candidate to the source-aware K500-val classwise selector did not change the
  selector ceiling: it stayed at `0.91375/0.63030` (`+0.96/+3.35 pp`). This
  confirms that stronger rare-ranking can solve CPSC, but the remaining gap is
  Ningbo/Chapman/Georgia generalization rather than CPSC-only capacity.
- A last-epoch diagnostic on the same big-adapter rankrare recipe confirmed
  that checkpoint selection is not the non-CPSC blocker. Last epoch improved
  CPSC to `0.8836/0.6718`, but hurt Ningbo, Chapman, and Georgia, with a
  four-center mean of only `0.8923/0.6017` (`-1.19/+0.49 pp` vs direct).
  Do not use fixed last-epoch training to bypass conservative K500-val
  selection.
- An adversarial-consistency variant was added for ECGFounder: stream-specific
  BCE weights (`--adv_bce_loss_weight`) plus `--target_logit_anchor_source
  initial_head` and `--target_logit_anchor_stream target_adv`, so online
  adversarial samples can be used as a direct-head distillation branch instead
  of hard-label retraining. The tested recipe
  (`adv_bce_loss_weight=0.15`, target-adv logit anchor `0.75`, mild rank loss
  on target-real) preserved non-CPSC by selecting baseline heads, but only
  improved CPSC to `0.8749/0.6333`; the four-center mean was
  `0.9071/0.5997` (`+0.30/+0.28 pp`). This is a useful safety mechanism, not a
  breakthrough candidate.
- Finer ECGFounder K500-val classwise selector alphas (`0.0..1.0` by `0.1`)
  plus the adv-distill candidate gave the current best AUROC selector:
  probability-space classwise mean `0.91466/0.63024`, i.e.
  `+1.05 AUROC pp / +3.34 AUPRC pp` vs direct K500. This improves AUROC
  slightly over the previous `0.91375/0.63030` selector but still misses the
  requested `+2/+5`; the next progress must come from new candidates for
  Ningbo/Chapman/Georgia, not selector-grid tweaks alone.
- ECGFounder feature-adapter heads were tested as a zero-init feature residual
  before the frozen Super5 head (`hidden=512`, K500-base, rankrare,
  warm20/lam0.05). K500-val selected the baseline for all four centers; the
  held-out mean was exactly direct K500 (`0.90416/0.59682`, `+0/+0 pp`).
  Attack success collapsed to about `0.10-0.19`, so this route does not create
  useful adversarial pressure. Avoid repeating feature adapters without a
  materially different encoder/update mechanism.
- Anchor-preserving soft-label ECGFounder candidates were tested to reduce
  hard-label noise in mixed latent-hull samples. A strong version
  (`lam0.10`, 8 PGD steps, `eps=3`, `adv_weight=15`) kept
  `atk_init~=0.84-0.89` and was too aggressive; held-out mean was
  `0.90404/0.59829` (`-0.01/+0.15 pp`). A gentler version
  (`lam0.05`, 5 PGD steps, `eps=2`, `adv_weight=8`,
  `target_logit_anchor=0.5`) kept healthy attack success
  (`atk_init~=0.61-0.70`, `atk_anchor~=0.50-0.64`) but still failed to
  improve held-out mean: `0.90495/0.59650` (`+0.08/-0.03 pp`). Adding both
  soft-label candidates to the fine-alpha source-aware classwise selector did
  not change the selector ceiling (`0.91466/0.63024`). The bottleneck is not
  simply hard labels or attack-strength calibration.
- Because frozen ECGFounder feature/head candidates now appear capped, an
  official-style full-fine-tuning control was run for Georgia on 2026-05-25.
  Direct full encoder fine-tuning improved PTB-XL source metrics
  (`0.9232/0.8063`) but hurt Georgia held-out target performance
  (`0.8567/0.6816`, drop-all-zero `0.8704/0.7650`) versus the frozen
  ECGFounder K500 direct baseline (`~0.8820/0.7226`). Do not pursue full
  encoder fine-tuning blindly as the main ECGFounder breakthrough; if revisited,
  use a lower-LR/partial-unfreeze recipe with a matched direct control.
- A matched four-center low-LR full-fine-tuning control was then run with
  `lr=2e-5`, `target_real_weight=40`, K500-internal random `100`-record
  validation, and source-plus-target-val AUPRC selection. Random Super5-head
  fullFT without VAE reached `0.90035/0.63008`; the matched VAE stream
  (`lam0.05`, `adv_weight=8`, `k_anchor=160`, exact labels) reached
  `0.90779/0.64125`. Thus VAE adds only `+0.74/+1.12 pp` over this fullFT
  control, while most of the AUPRC gain versus frozen headFT comes from fullFT
  itself. The held-out oracle/fixed-epoch ceiling for this family is only about
  `0.9137/0.6486`, so checkpoint selection is not the main blocker.
- `run_ecgfounder_fullft_super5_pilot_20260523.py` now supports
  `--init_head_path` to initialize `model.dense` from the trained Super5 K500
  head (`best_head.pt`). This was tested because random-head fullFT improved
  AUPRC but lost AUROC. Matched init-head direct fullFT reached
  `0.91354/0.62974`; init-head VAE with `adv_weight=8` reached
  `0.91374/0.63019`, and increasing VAE sampler pressure to `adv_weight=20`
  reached `0.91478/0.63012`. These are the best ECGFounder AUROC candidates so
  far (`+1.06 AUROC pp` versus frozen headFT direct), but VAE's net gain over
  the matched init-head direct control is only `+0.12/+0.04 pp` at
  `adv_weight=20`. Do not repeat init-head plus simple higher adv-weight
  sweeps as the main route; the missing mechanism is a VAE branch that improves
  Ningbo/Chapman/Georgia without being dominated by target-real fullFT.

Current 2026-05-25 EfficientNet1DV2 evidence:

- The best current four-center direct K500 baseline is about `0.8543/0.4880`.
- Local standardized same-label hull partners, include-anchor, rare-class
  K500-anchor quotas, AugMix latent branch, and low sampler adv weight
  (`adv_weight=0.3`, warmup 10) reached about `0.8709/0.5117`, i.e.
  `+1.66 AUROC pp / +2.38 AUPRC pp`. Gains are dominated by CPSC
  (`+5.39/+6.03 pp`) and Chapman AUPRC; Ningbo and Georgia remain near direct
  K500.
- Because `adv_weight` is a sampler weight, a true high-pressure run with
  `adv_weight=20`, warmup 20 was tested. It did not improve held-out mean
  (`0.8696/0.5096`, `+1.53/+2.17 pp`) and slightly hurt Ningbo/Chapman while
  preserving most CPSC gain. Do not assume simply increasing adv sampler weight
  solves the bottleneck.
- A hard-label variant (`adv_label_mode=multi_hot_hard`, `adv_weight=10`,
  warmup 20) improved Chapman AUPRC but reduced CPSC/Georgia, with mean about
  `0.8685/0.5096` (`+1.42/+2.16 pp`). Avoid repeating hard labels without a
  materially different regularizer or source/target selector.
- A paper-safe EfficientNet multi-candidate selector over old lam0.15,
  low-adv localstd, high-adv, hard-label, and direct candidates still selected
  only about `0.8707/0.5108` globally (`+1.64/+2.28 pp`) or
  `0.8701/0.5078` classwise (`+1.58/+1.98 pp`). This shows the current
  candidate pool ceiling is the blocker, not alpha/classwise selection.
- Practical next step: generate genuinely different candidates that address
  Ningbo/Georgia held-out generalization, not more variants of the same
  localstd include-anchor recipe. Candidates worth testing should add a
  source-aware or cross-center-invariant constraint, improve rare-class ranking
  without relying on K500-val overfit, or change the backbone adaptation mode.
- A mild source-logit anchor on the EfficientNet low-adv localstd recipe
  (`source_logit_anchor_weight=0.05`, 12 batches) was tested across the four
  K500 centers. It improved neither the mean nor selector ceiling: held-out
  mean was about `0.8677/0.5101` (`+1.34/+2.23 pp` over direct, but
  `-0.32/-0.16 pp` versus low-adv localstd). It also showed K500-val optimism:
  internal validation rose strongly while ref-excluded PN2021 did not.
- Last-2-feature-block EfficientNet adaptation was tested with the same
  low-adv localstd latent AugMix recipe. It trained only about 1.08M
  parameters and kept ASR healthy, but underfit the useful CPSC adaptation:
  held-out mean was about `0.8559/0.4902`, only `+0.16/+0.23 pp` over direct.
  Do not repeat last-block-only updates as a primary path unless another
  factor changes materially.
- Reducing target-real stream weight from 80 to 40 recovered PTB-XL source
  metrics (`~0.9033/0.7564`) but hurt target mean: `0.8678/0.5071`
  (`+1.35/+1.93 pp` over direct and `-0.31/-0.46 pp` versus low-adv localstd).
  It only slightly improved Ningbo AUPRC, and adding it to the K500-val
  selector did not raise the selector ceiling.
- A more anchor-local recipe (`hull_lambda=0.05`) with a mid-strength adv
  buffer window (`boundary_prob=[0.15,0.85]`) kept ASR in range for
  Ningbo/Chapman/Georgia but made CPSC attacks too weak after early epochs. Its
  held-out mean was `0.8703/0.5116` (`+1.60/+2.37 pp` over direct), essentially
  tied with but slightly below low-adv localstd. It gave small AUPRC gains on
  Ningbo/Chapman/Georgia, but lost CPSC (`-0.18/-0.35 pp` vs low-adv). Treat it
  as a selector-diversity/diagnostic candidate, not a standalone breakthrough;
  adding it to the K500-val selector also did not raise the ceiling.
- Continuing the best low-adv localstd recipe from each direct K500 checkpoint
  rather than from the PTB-XL source checkpoint gave the new best EfficientNet
  standalone mean, but only marginally: `0.8720/0.5139`, i.e.
  `+1.76 AUROC pp / +2.60 AUPRC pp` over direct and only
  `+0.10/+0.22 pp` over the previous low-adv localstd run. K500 internal
  validation improved strongly, but held-out PN2021 only moved a little; do not
  repeat direct-init alone as a breakthrough path. The next useful direction is
  label semantics or partner-pool diversity, especially because the current
  `mixed_soft` adv labels collapse multi-hot anchors to `argmax` before mixing
  with teacher probabilities.
- A direct-init `latent_mixed_teacher` label variant was tested to preserve
  multi-hot latent-mix semantics. It reached about `0.8721/0.5141`, only
  `+0.015/+0.017 pp` over direct-init localstd and `+1.78/+2.62 pp` over direct
  K500. This is a clean near-null result: label semantics alone is not the
  current EfficientNet bottleneck.
- Adding the direct-init candidates to the K500-validation multi-candidate
  selector also did not unlock the ceiling. The best global selected mean was
  about `0.8721/0.5142` (`+1.78/+2.63 pp` over direct), while classwise
  selection was lower. Further progress needs genuinely new candidates, not a
  larger pool of the same localstd/include-anchor family.
- A source-partner diversity run added a balanced PTB-XL folds1-8 latent pool
  (`250` primary-class samples per Super5 class) to each target K500 latent
  pool and sampled anchors with `real_anchor=1.0, ptbxl_source=0.35`. It
  produced healthy ASR but no breakthrough: target-center mean was about
  `0.8717/0.5138`, i.e. `+1.74/+2.59 pp` over direct K500, but
  `-0.026/-0.017 pp` versus direct-init localstd and `-0.040/-0.034 pp` versus
  direct-init `latent_mixed_teacher`. This argues that simply mixing source
  samples into the adversarial anchor pool is not the missing mechanism.
- A stricter source-neighbor-only run used the same merged latent pool but set
  `ptbxl_source=0.0` in anchor sampling, so adversarial anchors came only from
  the K500 target subset while PTB-XL could still appear as same-label latent
  hull neighbors. It tied direct-init localstd (`0.8719/0.5139`) but did not
  improve it. CPSC attack success fell to roughly `0.11-0.22` because the
  K500-train split had no real HYP/MI anchors; this confirms that source
  neighbors alone cannot compensate for missing target anchors. Adding both
  source-partner candidates to the K500-val selector left the ceiling at about
  `0.8721/0.5142`, essentially unchanged.
- An automatic K500-only inverse-frequency anchor quota policy was added:
  `anchor_class_weight_mode=inv_freq_kshot` computes weights from the K500-train
  real-anchor label counts after the internal K500-val split, with a capped
  inverse-frequency formula and a small missing-class source fallback. It
  fixed weak CPSC ASR (`~0.27-0.35` instead of `0.11-0.22`) but did not improve
  held-out target metrics: the four-center mean was about `0.8717/0.5128`,
  below direct-init localstd and `latent_mixed_teacher`. Use it as a
  paper-safe diagnostic mechanism, not a current best recipe.
- A new EfficientNet candidate was launched on 2026-05-25:
  `paper_effnet_sourcepartner_lmtm_directinit_seed20260531_v6_20260525`.
  It combines the source-partner latent pool with partner-only anchor sampling
  (`real_anchor=1.0,ptbxl_source=0.0`) and `adv_label_mode=latent_mixed_teacher`
  (`teacher_mix=0.4`) from direct K500 checkpoints. This is not a duplicate of
  the prior partner-only run, which used `mixed_soft`. The hypothesis is that
  PTB-XL latents can serve as same-label hull neighbors while multi-hot latent
  labels are preserved instead of collapsed to an argmax. Four centers
  (`ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`) are running on GPUs
  0-3; evaluate by `eval_result_v6_exclrefs_crop1000.json` and compare against
  direct K500 plus the current best `0.8721/0.5142` selector ceiling.
- The source-partner + `latent_mixed_teacher` candidate completed. Held-out
  v6 ref-excluded mean was `0.87191/0.51391`, slightly below direct-init
  `latent_mixed_teacher` (`0.87212/0.51410`) and essentially tied with
  partner-only `mixed_soft` (`0.87193/0.51395`). It improved over same-version
  v6 direct K500 (`0.85432/0.48790`) by only `+1.76/+2.60 pp`, still far from
  the requested `+4.5/+7`. Do not repeat source-partner plus label-mode-only
  combinations as the main route.
- The wrapper `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py` now
  exposes `--hull_label_mode`, `--hull_mix_label_mode`, and hull soft-label
  caps. The compatible-neighbor soft-label candidate
  `paper_effnet_compat_anchorsoft_lmtm_directinit_seed20260531_v6_20260525`.
  It uses `hull_label_mode=compatible`, `hull_mix_label_mode=anchor_soft`,
  `hull_label_lambda_y=0.35`, `hull_label_new_class_cap=0.35`,
  `source_weights=real_anchor=1.0,ptbxl_source=0.15`, and
  `source_floor_per_class=5`. The intended mechanism is abnormal-only
  cross-class local latent neighbors with capped weak labels, using only K500
  plus PTB-XL source latents. It completed at `0.87183/0.51418`, i.e.
  `+1.75/+2.63 pp` over same-version v6 direct K500 and effectively tied with
  direct-init `latent_mixed_teacher` (`0.87212/0.51410`). This rules out simple
  compatible-neighbor soft-label smoothing as the missing EfficientNet
  breakthrough while preserving paper-safe K500-only target information.

## Latent Mixup Rules

Treat ECGTwin VAE latent mixing as constrained VRM/manifold mixup.

- Mainline pairing: same-label or exact-positive-set first. Compatible-label
  soft BCE is an ablation.
- Do not mix NORM with abnormal labels unless the experiment is explicitly a
  hard-negative ablation.
- Prefer local partners: same-label kNN in normalized latent space, or distance
  cutoff inside the same-label pool. Avoid global far-pair random mixing as a
  main result.
- Use anchor-dominant convex coefficients: sample `lambda` from Beta-style
  mixup, set anchor weight to `max(lambda, 1-lambda)`, and clamp roughly
  `[0.6, 0.95]`.
- For same-label pairs, keep hard target `y_anchor`. For compatible-label
  ablations, use float multi-hot BCE targets and keep NORM suppression logic.
- Standardize VAE latents by aggregate posterior mean/std before distance
  sampling, mixup, or PGD when implementing new code.

## Loss And Schedule

Prefer mixed clean/adv training:

```text
loss = clean_anchor_bce
     + mix_weight * mixed_latent_bce
     + adv_weight * adversarial_bce_or_bernoulli_kl
     + source_anchor_weight * source_logit_or_clean_source_loss
```

Starting points:

- `adv_weight`: try lower values or warmup before increasing attack strength.
  If the current script implements `adv_weight` as a sampler weight rather than
  a direct loss coefficient, sweep a global `{10, 20, 40}` or add `0 -> target`
  warmup over the first 20-30% of epochs.
- `hull_lambda`: sweep `{0.05, 0.10, 0.15}` with include-anchor.
- `hull_steps`: train with 3-5 steps; validate with stronger 10-20 step
  audit. Final claims should use a fixed stronger protocol.
- Keep clean K500 anchors in the batch whenever latent mix or PGD is enabled.
- If using a TRADES-style loss for multilabel super5, use independent
  Bernoulli KL between clean and adversarial sigmoid probabilities, not
  softmax KL.

## Immediate Optimization Order

1. Confirm diagnostics are logged: `atk_init`, `atk_anchor`, loss gains,
   latent norm utilization, and validation source floor.
2. Add or use adversarial-weight warmup if the script only has a fixed high
   `--adv_weight`.
3. Run one global recipe across all target centers:
   hard-BCE anchors, include-anchor, lower or warmup `adv_weight`,
   `hull_lambda` in `{0.05, 0.10, 0.15}`.
4. Select by K500 internal validation with a source clean floor. Only then
   evaluate PN2021 heldout.
5. If average improvement is still small, test locality-aware same-label kNN
   partner selection before adding cross-label soft mixup.

Sanity checks are diagnostics, not the main quality gate. Only hard-reject
non-finite, flatline, or physically invalid decoded signals; do not revive
heavy victim-score quality gates as the main selection mechanism.

## Reference

For literature and repository priors behind these rules, read
`references/literature_and_repos.md` only when you need citations or rationale.
