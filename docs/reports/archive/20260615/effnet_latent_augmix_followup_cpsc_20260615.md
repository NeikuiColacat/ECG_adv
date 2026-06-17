# EfficientNet1DV2 Latent AugMix Follow-up CPSC Check - 2026-06-15

## Scope

- Model family: EfficientNet1DV2 VAE-LHAT, CPSC 2018 K500, seed `20260601`.
- Evaluation: PN2021 Super5 v7 SJR/RGQ mapping, hash `555ec85d5b51`; K500 ref-excluded.
- PN2021-C setting: `cpsc_2018`, five calibrated severity-5 operators, `calibrated_10to20pp`.
- Baselines are the matched CPSC rows from `effnet_calibrated_latent_augmix_cpsc_20260615.csv`.

## Variants Tested

1. A1 `calibrated_latent_augmix_s5_norenorm`
   - Same calibrated latent AugMix chain as the previous s5 run.
   - Added `--no_latent_augmix_renorm` to test whether global renormalization was eating corruption strength.

2. A2 `calibrated_latent_augmix_s5_directloss`
   - Keeps VAE-LHAT + latent AugMix.
   - Uses `latent_augmix_copies=2`.
   - Adds direct training on latent-AugMix views with BCE plus JSD:
     `--enable_latent_augmix_consistency --latent_augmix_consistency_weight 2.0 --latent_augmix_consistency_loss jsd --latent_augmix_bce_weight 1.0`.
   - Does not add the separate raw ECG corruption branch.

## Main CPSC Comparison

| method | PN2021 7c AUPRC | CPSC clean AUPRC | CPSC PN2021-C AUPRC | CPSC AUPRC drop |
|---|---:|---:|---:|---:|
| VAE noAug | 0.4768 | 0.6123 | 0.4715 | 0.1408 |
| VAE + normal AugMix s2 | 0.4760 | 0.6150 | 0.4753 | 0.1397 |
| Calibrated raw-supervised | 0.4789 | 0.6016 | 0.5351 | 0.0664 |
| Calibrated latent AugMix s5 | 0.4741 | 0.6147 | 0.4776 | 0.1371 |
| A1 calibrated latent AugMix s5 no-renorm | 0.4741 | 0.6135 | 0.4738 | 0.1397 |
| A2 calibrated latent AugMix s5 direct-loss | 0.4741 | 0.6108 | 0.4821 | 0.1287 |

Delta versus VAE noAug:

| method | clean PN2021 7c AUPRC | CPSC clean AUPRC | CPSC PN2021-C AUPRC | AUPRC drop reduction |
|---|---:|---:|---:|---:|
| VAE + normal AugMix s2 | -0.08 pp | +0.28 pp | +0.38 pp | +0.10 pp |
| Calibrated raw-supervised | +0.21 pp | -1.07 pp | +6.37 pp | +7.44 pp |
| Calibrated latent AugMix s5 | -0.27 pp | +0.24 pp | +0.61 pp | +0.37 pp |
| A1 no-renorm | -0.27 pp | +0.12 pp | +0.23 pp | +0.11 pp |
| A2 direct-loss | -0.27 pp | -0.14 pp | +1.06 pp | +1.21 pp |

## A2 Per-Operator Results

| corruption | AUROC | AUPRC | AUROC drop | AUPRC drop |
|---|---:|---:|---:|---:|
| powerline_noise | 0.7680 | 0.4943 | 0.1077 | 0.1165 |
| emg_noise | 0.7973 | 0.5242 | 0.0784 | 0.0866 |
| baseline_wander | 0.7397 | 0.4658 | 0.1360 | 0.1451 |
| baseline_shift | 0.7522 | 0.4606 | 0.1235 | 0.1502 |
| random_leads_masking | 0.7713 | 0.4656 | 0.1044 | 0.1453 |

## Training Notes

- A2 direct consistency was active in the training log:
  `loss=jsd weights=(consistency=2.0, bce=1.0)`.
- A2 best K500 internal checkpoint was epoch 30, quick AUROC/AUPRC `0.9323/0.8491`.
- A2 final clean metrics:
  - PTB-XL: `0.8940/0.7408`
  - PN2021 7-center: `0.7821/0.4741`
  - CPSC clean: `0.8757/0.6108`
- A2 `agent_decision.json` still marked the latent-hull attack as `attack_too_weak` at epoch 30 with final ASR about `0.22`.

## Verdict

A1 rules out renormalization as the main blocker: disabling latent-AugMix renorm made the CPSC PN2021-C AUPRC worse than calibrated latent AugMix s5 (`0.4738` vs `0.4776`).

A2 partially supports the diagnosis that AugMix views need a direct supervised/JSD loss path, not just buffer insertion: it improves CPSC PN2021-C AUPRC to `0.4821`, or `+1.06 pp` over noAug. But it is still far below calibrated raw-supervised (`0.5351`, `+6.37 pp` over noAug).

Current conclusion: direct loss helps, but latent-AugMix views are still not equivalent to training on raw ECG corruptions. The useful signal appears to come from input-space raw corruptions with supervised/JSD recovery, not merely from putting stronger corruptions inside the latent AugMix chain.

Do not expand A1 or A2 to four centers as-is. The next meaningful test, if any, should move a calibrated raw ECG corruption view into the AugMix mixing graph itself and train it with direct BCE/JSD, while keeping the existing VAE-LHAT branch unchanged.
