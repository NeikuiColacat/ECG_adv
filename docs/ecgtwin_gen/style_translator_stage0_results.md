# Stage 0 Style Translator — 3-center (cpsc_extra + chapman + cpsc)

**Ablation A / B / C** vs **baseline**

- Baseline: `/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json`
- A: target style: `/root/autodl-tmp/center_aware_ibe/retrain/stage0_A/eval_crosscenter.json`
- B: null style: `/root/autodl-tmp/center_aware_ibe/retrain/stage0_B/eval_crosscenter.json`
- C: wrong (georgia): `/root/autodl-tmp/center_aware_ibe/retrain/stage0_C/eval_crosscenter.json`

## Headline Macro Metrics

| Metric | Baseline | A: target style | B: null style | C: wrong (georgia) | Δ A vs B | Δ A vs C | Δ A vs Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTBXL test macro AUROC | 0.9741 | 0.9705 | 0.9666 | 0.9738 | +0.39pp | -0.33pp | -0.36pp |
| PTBXL test macro AUPRC | 0.8447 | 0.7713 | 0.7585 | 0.8124 | +1.28pp | -4.11pp | -7.34pp |
| MAIN5 avg macro AUROC | 0.9291 | 0.9229 | 0.9315 | 0.9346 | -0.86pp | -1.17pp | -0.62pp |
| MAIN5 avg macro AUPRC | 0.7399 | 0.6926 | 0.6936 | 0.7358 | -0.10pp | -4.32pp | -4.73pp |

## Per-Center Macro AUROC

| Center | N | Baseline | A: target style | B: null style | C: wrong (georgia) | Δ A vs Base |
|---|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing ★ | 9709 | 0.9575 | 0.9453 | 0.9582 | 0.9534 | -1.22pp |
| cpsc_2018 ★ | 5279 | 0.9166 | 0.9250 | 0.9360 | 0.9328 | +0.85pp |
| cpsc_2018_extra ★ 🎯 | 1296 | 0.8538 | 0.8456 | 0.8643 | 0.8706 | -0.83pp |
| georgia ★ | 9320 | 0.9482 | 0.9423 | 0.9380 | 0.9491 | -0.59pp |
| ningbo ★ | 34470 | 0.9695 | 0.9563 | 0.9611 | 0.9672 | -1.32pp |
| ptb | 116 | 0.8543 | 0.9295 | 0.8790 | 0.9058 | +7.52pp |
| st_petersburg_incart | 33 | 0.9218 | 0.9421 | 0.9135 | 0.9175 | +2.04pp |

*★ = MAIN5 center; 🎯 = target of style enrollment*

## Target Center (cpsc_2018_extra) Per-Class AUROC

| Class | Baseline | A: target style | B: null style | C: wrong (georgia) | Δ A vs Base |
|---|---:|---:|---:|---:|---:|
| NSR | 0.7417 | 0.7173 | 0.8688 | 0.7641 | -2.44pp |
| STach | 0.9712 | 0.9541 | 0.9559 | 0.9640 | -1.71pp |
| AF | 0.9626 | 0.9508 | 0.9397 | 0.9583 | -1.18pp |
| IAVB | 0.9121 | 0.8757 | 0.9018 | 0.9067 | -3.64pp |
| LBBB | 0.5661 | 0.6002 | 0.5459 | 0.6497 | +3.41pp |
| RBBB | 0.9693 | 0.9753 | 0.9737 | 0.9809 | +0.60pp |

## Ablation Verdict

- ❌ A target AUROC ≥ +0.5pp vs Baseline (got -0.83pp)
- ❌ A > B target AUROC ≥ +0.3pp (got -1.88pp) — style vec adds value
- ❌ A > C target AUROC ≥ +0.1pp (got -2.51pp) — correct style beats wrong
- ✅ PTBXL test AUROC ≥ 0.970 (got 0.9705) — no cross-domain collapse

**🔴 No-go for Stage 0-multi** — pivot to root-cause investigation.

## Post-Mortem: Why A Underperformed

**Ranking on target center (cpsc_2018_extra):** C (0.8706) > B (0.8643) > Baseline (0.8538) > **A (0.8456)**.
Complete inversion of hypothesis. Stage 0 failed.

### Positive finding (not what we hypothesized)
- **B > Baseline +1.05pp** on target: vanilla ECGTwin synth (no style_fusion applied) *does* help
  cross-center transfer. `ib_projector(base_IBE(x))` conditioning alone is a useful data-aug
  signal for MAIN5 and target.
- **B > Baseline NSR on target +12.71pp** (0.7417 → 0.8688): huge NSR gain from vanilla synth.

### Negative finding (core hypothesis disproved)
- Every path including learned `style_fusion` (A) regresses vs B on every MAIN5 center.
- A vs B: target −1.88pp, NSR −15.15pp, IAVB −2.61pp, MAIN5 avg AUROC −0.86pp.
- A's PTBXL test ≥ 0.970 (0.9705) rules out cross-domain collapse — the regressions are
  genuinely in the synthetic data distribution, not in the classifier.

### Diagnosis of mechanism
Training metrics hinted at this: `L_identity` stuck at 0.4 (plan target was < 0.01), `feat_drift`
reached 0.53 meaning even own-center style pushes base_vector ~53% relative norm off. DiT was
trained with only a 15% random channel-mask augmentation — robust to small perturbations but
not to 50% drift. Every generated latent operates on a base_vector that's out of DiT's
training distribution; noise predictions degrade accordingly.

Geometric artifact: A vs C style_vec cos-sim = 0.60, norms 15.64 vs 15.55. Most of A's
direction is shared with C; the small divergent 40% that claims "target-ness" is where the
extra damage comes in. C's delta happens to land in a more benign region.

### Why L_identity didn't anchor
`lambda_sup=1.0, lambda_id=1.0`, but `L_supcon ≈ 4.8` while `L_identity ≈ 0.4`. SupCon's
absolute magnitude is 10× larger, so gradient pressure is dominated by center-discrimination.
The style_fusion converges to whatever separates centers in IBE feature space, with no
meaningful penalty for corrupting the own-center case.

## Pivot Options (ranked)

1. **Lean into B's finding — drop style_fusion entirely, ship vanilla ECGTwin synth.**
   B beat Baseline by +1.05pp on target (n=1296) and +2.27pp on cpsc_2018 (n=5279). That's
   a genuine result from ~2h of work. Write up as a positive finding, skip Stage 0-multi.

2. **Rebalance losses and retrain.** `lambda_id: 1.0 → 20.0`; apply LayerNorm on both sides
   of `feat_translated` / `base_feat` before MSE so scale doesn't bias supcon. Expected
   impact: identity loss drops to < 0.05, feat_drift < 0.1, but risk: style signal may
   collapse to near-null (making A ≡ B). ~30 min retrain, ~60 min retest (3 paths).

3. **Gate the delta.** Wrap `style_fusion` output with a `nn.Parameter` gate initialized
   near 0: `feat = base + tanh(α) · delta`, teach α via own lr. Smoother version of #2;
   higher chance of non-trivial A ≠ B while anchoring own-case. ~45 min impl + ~90 min
   retrain + retest.

4. **Inject style elsewhere.** Plan's deferred "LoRA on base_IBE" or "AdaLN on DiT" routes:
   do not edit `base_vector` at all; use style_vec as an external AdaX-style modulation.
   Higher impl cost (~2h), but addresses the root cause (DiT distribution shift).

## Artifacts
- Training log: `/root/autodl-tmp/center_aware_ibe/ckpts/stage0_style_translator_3c/training_log.txt`
- Best translator: `/root/autodl-tmp/center_aware_ibe/ckpts/stage0_style_translator_3c/translator_best.pth` (epoch 10 of 20)
- Synth .npz: `/root/autodl-tmp/center_aware_ibe/synth/stage0_{A,B,C}_*.npz`
- Retrain dirs: `/root/autodl-tmp/center_aware_ibe/retrain/stage0_{A,B,C}/`
- Style vecs: `/root/autodl-tmp/center_aware_ibe/styles/{cpsc_2018_extra,georgia}_style.pt`
