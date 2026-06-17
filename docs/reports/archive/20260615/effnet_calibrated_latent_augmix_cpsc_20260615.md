# EfficientNet1DV2 Calibrated Latent AugMix CPSC Check - 2026-06-15

## Scope

- New method: EfficientNet1DV2 VAE-LHAT + latent AugMix.
- Change under test: latent AugMix non-latent chains use PN2021-C `calibrated_10to20pp`, severity `5`, all five ops.
- Explicitly disabled: extra raw ECG corruption supervised/JSD phase.
- Protocol: CPSC 2018 only, seed `20260601`, K500 ref-excluded, PN2021 Super5 v7 mapping `555ec85d5b51`.
- GPU: `CUDA_VISIBLE_DEVICES=4`.

Run directory:

```text
/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018/20260615_callatentaugmix_cpsc_gpu4/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s5_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_callatentaugmix_ep30_seed20260601
```

## Implementation Verification

- Added `latent_augmix_severity_profile` plumbing through:
  - `ecg_adv_gen/adaptation/lhat.py`
  - `scripts/pgd_cross_center/synth_online_at_super5.py`
  - `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`
  - `ecg_adv_gen/runner/effnet_vae_lhat.py`
  - `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- Added managed configs:
  - `configs/experiments/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq.yaml`
  - `configs/experiments/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018.yaml`
- Focused tests passed:
  - `util/tests/test_adaptation_lhat.py::test_latent_augmix_core_forwards_severity_profile_to_chain_ops`
  - `util/tests/test_effnet_vae_lhat_runner.py::test_build_effnet_vae_lhat_commands_preserve_wrapper_flags`
  - `util/tests/test_config_loader.py::test_effnet_vae_lhat_calibrated_latent_augmix_config_exposes_profile_flag`
  - full tracked config expansion test: `59 passed`

Dry-run confirmed the launcher command includes:

```text
--latent_augmix_severity 5
--latent_augmix_severity_profile calibrated_10to20pp
--latent_augmix_ops powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking
```

## Training Notes

- Best K500 internal validation checkpoint: epoch 30, AUROC/AUPRC `0.9361/0.8615`.
- Attack success weakened late: final ASR about `0.17`; `agent_decision.json` marked `attack_too_weak`.
- This means the calibrated chains are active, but the adversarial latent stream is not strong in the late epochs.

## Main CPSC Comparison

The table compares the CPSC-trained model against matched CPSC-trained references. PN2021-C is averaged over five calibrated severity-5 corruptions on `cpsc_2018`.

| method | PN2021 7c AUPRC | CPSC clean AUPRC | CPSC PN2021-C AUPRC | CPSC AUPRC drop |
|---|---:|---:|---:|---:|
| VAE noAug | 0.4768 | 0.6123 | 0.4715 | 0.1408 |
| VAE + normal AugMix s2 | 0.4760 | 0.6150 | 0.4753 | 0.1397 |
| Calibrated raw-supervised | 0.4789 | 0.6016 | 0.5351 | 0.0664 |
| Calibrated latent AugMix s5 | 0.4741 | 0.6147 | 0.4776 | 0.1371 |

Delta versus VAE noAug:

| method | clean PN2021 7c AUPRC | CPSC clean AUPRC | CPSC PN2021-C AUPRC | AUPRC drop reduction |
|---|---:|---:|---:|---:|
| VAE + normal AugMix s2 | -0.08 pp | +0.28 pp | +0.38 pp | +0.10 pp |
| Calibrated raw-supervised | +0.21 pp | -1.07 pp | +6.37 pp | +7.44 pp |
| Calibrated latent AugMix s5 | -0.27 pp | +0.24 pp | +0.61 pp | +0.37 pp |

Delta versus normal latent AugMix:

- CPSC clean AUPRC: `-0.04 pp`
- CPSC PN2021-C AUPRC: `+0.23 pp`
- CPSC AUPRC drop reduction: `+0.26 pp`

## Diagnostic Four-Center PN2021-C

The new CPSC-trained checkpoint was also evaluated on all four target centers as a diagnostic only. This should not be mixed with the main four-center comparison, because the reference rows use center-specific trained models.

Across four centers and five corruptions:

- Mean corrupted AUROC/AUPRC: `0.7365/0.3827`
- Mean drop vs clean: `10.93 pp AUROC / 12.89 pp AUPRC`
- Drop-all-zero corrupted AUROC/AUPRC: `0.7570/0.5274`

## Verdict

This exact "calibrated profile inside latent AugMix chains" variant is not enough.

It improves CPSC calibrated PN2021-C AUPRC only `+0.61 pp` over noAug and `+0.23 pp` over normal latent AugMix, while the calibrated raw-supervised branch gives `+6.37 pp` over noAug on the same CPSC comparison. The large recovery is still coming from the separate raw ECG corruption supervised/JSD training phase, not from merely increasing the AugMix chain strength.

Recommendation: do not expand this exact recipe to four centers. If continuing this direction, the next useful test is not another severity increase; it should either add calibrated corruption views as supervised/JSD training samples, or redesign latent AugMix so the calibrated branch contributes a direct consistency/supervised loss rather than only adding mixed buffer samples.
