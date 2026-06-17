# PN2021-C AugMix Gap Diagnostic, 2026-06-16

## Question

Can moving the calibrated PN2021-C 10-20 pp corruption profile into the
VAE-LHAT + AugMix mainline recover about 10 pp PN2021-C robustness for
EfficientNet1DV2 and ECGFounder?

## Protocol

- Mapping: `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`.
- Primary diagnostic center: `cpsc_2018`.
- Four-center follow-up: `ningbo`, `cpsc_2018`, `chapman_shaoxing`,
  `georgia`.
- PN2021-C profile: `calibrated_10to20pp`, severity `5`.
- Operators: `powerline_noise`, `emg_noise`, `baseline_wander`,
  `baseline_shift`, `random_leads_masking`.
- Metrics below are mean over the five operators on the same ref-excluded
  CPSC subset.

## Current Progress Audit

Authoritative compact audit:

`/home/linbinhao/ECG_adv_data/runs/pn2021c_progress_audit_20260616/summary.md`

This audit rebuilds the EfficientNet1DV2 rows from raw PN2021-C eval JSONs or
the paper-safe robust-selector bundle, because the broader candidate CSVs mix
diagnostic rows and are not safe for direct aggregation.

| Family | Current best credible view | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs anchor AUROC / AUPRC |
|---|---|---:|---:|---:|
| EfficientNet1DV2 | boundary raw-AugMix w1/d1/m1 + K500 robust selector | 0.8657 / 0.5508 | 0.8117 / 0.4722 | +6.41 / +6.49 pp vs VAE-LH noAug |
| EfficientNet1DV2 | boundary raw-AugMix w1/d1/m1 + K500 robust selector + post-hoc input stabilizer35 | 0.8668 / 0.5508 | 0.8362 / 0.5065 | +8.86 / +9.92 pp vs VAE-LH noAug |
| ECGFounder | current stabilizer reporting candidate | 0.8896 / 0.6289 | 0.8641 / 0.5807 | +4.72 / +6.74 pp vs original VAE+AugMix |
| ECGFounder | center-best candidate mix | 0.9019 / 0.6427 | 0.8735 / 0.5925 | +5.66 / +7.93 pp vs original VAE+AugMix |
| ECGFounder | per-operator oracle | 0.9007 / 0.6451 | 0.8771 / 0.5993 | +6.01 / +8.61 pp vs original VAE+AugMix |

Current state: EfficientNet1DV2 now reaches about `+8.9 / +9.9 pp` over
VAE-LH noAug after adding a post-hoc input stabilizer to the robust-selector
boundary raw-AugMix recipe. This effectively touches the `+10 pp` target on
AUPRC, but it is still an inference-time diagnostic stabilizer rather than a
fully training-integrated AugMix method. ECGFounder still does not have a
stable `+10 pp` four-center result; even the existing-pool per-operator oracle
is short of that target.

## PN2021-C Amplitude Check

The `calibrated_10to20pp` profile is applied in the preprocessed z-score
signal space, not raw mV. On the CPSC ref-excluded clean-cache subset used for
the ECGFounder PN2021-C eval (`n=6377`, `100 Hz`, `1000` samples), clean ECG
has RMS/std almost exactly `1.0`, median global peak-to-peak amplitude `16.73`,
median mean-per-lead peak-to-peak amplitude `7.34`, and median absolute p95
amplitude `1.78`.

| Operator | Main calibrated parameter | Median per-lead p2p | Per-lead p2p ratio | Median perturbation RMS / clean RMS |
|---|---:|---:|---:|---:|
| clean | n/a | 7.34 | 1.00x | n/a |
| powerline_noise | `max_amplitude=8.0` | 12.68 | 1.73x | 2.45x |
| emg_noise | `max_amplitude=2.3` | 11.67 | 1.59x | 1.32x |
| baseline_wander | `max_amplitude=2.5`, `k=6`, `max_freq=0.8` | 14.65 | 2.00x | 2.65x |
| baseline_shift | `max_amplitude=2.4`, `shift_ratio=0.9`, `num_segment=6` | 7.72 | 1.05x | 1.52x |
| random_leads_masking | `mask_leads_prob=0.57` | 3.07 | 0.42x | 0.76x |

`baseline_shift` mainly changes DC baseline position rather than increasing
peak-to-peak amplitude. `random_leads_masking` reduces amplitude because a
median of `7 / 12` leads are zeroed, but it removes lead information directly.

## CPSC Results

| Model / method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs reference |
|---|---:|---:|---:|---:|
| ECGFounder calibrated raw-supervised, selected epoch 0 | 0.863 / 0.622 | 0.804 / 0.532 | 5.90 / 9.01 pp | reference |
| ECGFounder calibrated raw-supervised, forced last epoch 20 | 0.869 / 0.631 | 0.805 / 0.544 | 6.38 / 8.68 pp | +0.09 / +1.20 pp corrupted |
| ECGFounder raw-AugMix width=1 depth=1 fixed m=1.0, forced last epoch 20 | 0.869 / 0.627 | 0.807 / 0.545 | 6.17 / 8.23 pp | +0.30 / +1.28 pp corrupted |
| ECGFounder fullFT direct init-head | 0.905 / 0.725 | 0.812 / 0.551 | 9.26 / 17.39 pp | +0.82 / +1.92 pp corrupted |
| ECGFounder fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | 9.36 / 17.71 pp | +1.14 / +2.90 pp corrupted |
| ECGFounder fullFT VAE + raw-AugMix width=1 depth=1 fixed m=1.0 | 0.901 / 0.719 | 0.836 / 0.589 | 6.49 / 13.00 pp | +3.23 / +5.74 pp corrupted |
| ECGFounder fullFT VAE + low-weight five-op raw-AugMix width=1 depth=1 fixed m=1.0 | 0.907 / 0.740 | 0.835 / 0.593 | 7.18 / 14.75 pp | +3.13 / +6.10 pp corrupted |
| ECGFounder fullFT VAE + noise/mask-only raw-AugMix width=1 depth=1 fixed m=1.0 | 0.904 / 0.728 | 0.843 / 0.598 | 6.08 / 13.02 pp | +3.88 / +6.61 pp corrupted |
| ECGFounder fullFT VAE + drift/shift-only raw-AugMix width=1 depth=1 fixed m=1.0 | 0.907 / 0.735 | 0.810 / 0.552 | 9.68 / 18.25 pp | +0.62 / +2.04 pp corrupted |
| ECGFounder fullFT VAE + two-branch raw-AugMix: noise/mask hard + drift/shift soft | 0.903 / 0.728 | 0.842 / 0.598 | 6.09 / 12.96 pp | +2.65 / +3.70 pp vs fullFT VAE |
| ECGFounder fullFT VAE + noise/mask hard + drift/shift low hard-BCE | 0.903 / 0.727 | 0.841 / 0.597 | 6.13 / 13.01 pp | +2.61 / +3.61 pp vs fullFT VAE |
| ECGFounder fullFT VAE + noise/mask hard + drift/shift fullFT-teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | 6.10 / 12.94 pp | +2.76 / +4.03 pp vs fullFT VAE |
| ECGFounder fullFT VAE + noise/mask hard + drift/shift EfficientNet-teacher distill | 0.903 / 0.723 | 0.843 / 0.595 | 5.96 / 12.87 pp | +2.78 / +3.37 pp vs fullFT VAE |
| ECGFounder fullFT VAE + teacher-distill + explicit operator-conditioned adapter | 0.901 / 0.720 | 0.842 / 0.596 | 5.95 / 12.40 pp | +2.67 / +3.51 pp vs fullFT VAE |
| ECGFounder fullFT VAE + noise/mask hard + drift/shift normalized feature consistency | 0.902 / 0.726 | 0.842 / 0.598 | 6.02 / 12.89 pp | +2.69 / +3.66 pp vs fullFT VAE |
| ECGFounder fullFT VAE + all-op normalized feature consistency w=300 | 0.907 / 0.740 | 0.837 / 0.595 | 6.96 / 14.41 pp | +2.22 / +3.45 pp vs fullFT VAE |
| ECGFounder fullFT VAE + teacher-distill + input bandpass35 + lead repair diagnostic | 0.899 / 0.715 | 0.864 / 0.651 | 3.47 / 6.45 pp | +4.86 / +8.98 pp vs fullFT VAE |
| ECGFounder fullFT VAE + noise/mask hard + all-op EfficientNet-teacher distill + stabilizer35 | 0.895 / 0.695 | 0.867 / 0.637 | 2.84 / 5.84 pp | +3.05 / +4.76 pp vs fullFT VAE raw-AugMix; -0.82 / -3.15 pp vs CPSC hybrid |
| EfficientNet1DV2 VAE-LH noAug | 0.877 / 0.612 | 0.754 / 0.473 | 12.25 / 13.93 pp | reference |
| EfficientNet1DV2 calibrated raw-supervised | 0.865 / 0.602 | 0.817 / 0.535 | 4.80 / 6.64 pp | +6.25 / +6.22 pp corrupted |
| EfficientNet1DV2 calibrated latent AugMix | 0.878 / 0.615 | 0.760 / 0.478 | 11.80 / 13.71 pp | +0.56 / +0.46 pp corrupted |
| EfficientNet1DV2 calibrated latent AugMix + direct BCE/JSD | 0.876 / 0.611 | 0.766 / 0.482 | 11.00 / 12.87 pp | +1.16 / +0.92 pp corrupted |
| EfficientNet1DV2 full-pool raw-AugMix + VAE-LH | 0.862 / 0.586 | 0.787 / 0.492 | 7.49 / 9.41 pp | +3.16 / +2.01 pp corrupted |
| EfficientNet1DV2 full-pool raw-AugMix depth=1 + VAE-LH | 0.872 / 0.611 | 0.806 / 0.521 | 6.67 / 9.02 pp | +5.00 / +4.92 pp corrupted |
| EfficientNet1DV2 full-pool raw-AugMix depth=1 no-clip + VAE-LH | 0.872 / 0.611 | 0.804 / 0.518 | 6.80 / 9.23 pp | +4.85 / +4.70 pp corrupted |
| EfficientNet1DV2 full-pool raw-AugMix depth=1 fixed m=0.75 + VAE-LH | 0.870 / 0.609 | 0.810 / 0.527 | 6.00 / 8.17 pp | +5.45 / +5.55 pp corrupted |
| EfficientNet1DV2 full-pool raw-AugMix width=1 depth=1 fixed m=1.0 + VAE-LH | 0.865 / 0.602 | 0.819 / 0.536 | 4.59 / 6.59 pp | +6.33 / +6.49 pp corrupted |

## Full-Pool Raw-AugMix Follow-up

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_k500_v7_sjr_rgq_cpsc_2018/smoke_execute_check_fullpool_rawaugmix_20260616/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_ep30_seed20260601`

Key setup:

- VAE-LH mainline kept enabled: latent branch severity `2`, width `3`.
- Raw consistency branch used the source+target full pool:
  `scope=source_target`, `n=17817`, `max_batches=64`.
- Raw views used AugMix-style chains:
  `view_mode=augmix`, `width=3`, `depth=-1`, `alpha=1.0`,
  `severity_profile=calibrated_10to20pp`, severity `5`,
  `raw_corrupt_no_renorm=true`.
- Epoch 1 raw branch generated `16384` raw-AugMix views with all five
  operators represented in `op_counts`.
- Best K500-internal checkpoint was epoch 28:
  `0.9254 / 0.8236`, versus baseline K500 internal `0.8616 / 0.7518`.

Clean held-out CPSC from the standard PN2021 eval:

- CPSC clean center: `0.8715 / 0.6090`.
- Seven-center mean: `0.7849 / 0.4784`.

Calibrated PN2021-C CPSC per operator:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.862 / 0.586 | 0.814 / 0.524 | 4.78 / 6.19 pp |
| emg_noise | 0.862 / 0.586 | 0.786 / 0.499 | 7.62 / 8.71 pp |
| baseline_wander | 0.862 / 0.586 | 0.788 / 0.497 | 7.43 / 8.88 pp |
| baseline_shift | 0.862 / 0.586 | 0.787 / 0.490 | 7.50 / 9.57 pp |
| random_leads_masking | 0.862 / 0.586 | 0.761 / 0.448 | 10.11 / 13.72 pp |

This confirms that full-pool raw-AugMix is real progress over latent AugMix
direct-loss (`+3.16 / +2.01 pp` over VAE noAug corrupted), but it still trails
calibrated raw-supervised (`+6.10 / +6.37 pp`). The main failure mode is not
the lack of raw corruption coverage anymore; it is clean held-out degradation
and distribution mismatch from multi-op AugMix chains. The AugMix views are
strong enough to reduce drops, but less clean-preserving than the single-op
raw-supervised branch.

## Depth=1 Raw-AugMix Follow-up

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_ep30_seed20260601`

Change versus the previous full-pool raw-AugMix run:

- raw AugMix `depth=1`, so each branch applies exactly one calibrated
  PN2021-C operator;
- source+target raw pool, weights, clip, no-renorm, latent branch, and
  VAE-LH online AT recipe otherwise unchanged.

Best K500-internal checkpoint was epoch 28: `0.9272 / 0.8281`.

Clean held-out CPSC from the standard PN2021 eval:

- CPSC clean center: `0.8724 / 0.6109`.

Calibrated PN2021-C CPSC per operator:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.872 / 0.611 | 0.820 / 0.543 | 5.27 / 6.84 pp |
| emg_noise | 0.872 / 0.611 | 0.822 / 0.556 | 5.06 / 5.50 pp |
| baseline_wander | 0.872 / 0.611 | 0.818 / 0.541 | 5.40 / 7.01 pp |
| baseline_shift | 0.872 / 0.611 | 0.800 / 0.507 | 7.25 / 10.42 pp |
| random_leads_masking | 0.872 / 0.611 | 0.769 / 0.458 | 10.39 / 15.34 pp |

Mean over five operators: clean `0.8724 / 0.6109`, corrupted
`0.8057 / 0.5207`, drop `6.67 / 9.02 pp`.

This is the best AugMix-shaped EfficientNet1DV2 result so far on this CPSC
diagnostic: `+5.00 / +4.92 pp` corrupted over matched VAE noAug. The gain over
the previous random-depth raw-AugMix run is `+1.84 / +2.91 pp` corrupted,
which confirms that PN2021-C single-operator alignment matters.

## Depth=1 No-Clip Check

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_noclip_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_noclip_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_noclip_ep30_seed20260601`

Change versus depth=1 clipped: `raw_corrupt_clip_abs=0.0` instead of `6.0`.

Best K500-internal checkpoint was epoch 28: `0.9264 / 0.8251`.

Mean calibrated PN2021-C CPSC: clean `0.8721 / 0.6108`, corrupted
`0.8041 / 0.5184`, drop `6.80 / 9.23 pp`.

Per-operator PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.872 / 0.611 | 0.818 / 0.539 | 5.40 / 7.17 pp |
| emg_noise | 0.872 / 0.611 | 0.821 / 0.554 | 5.11 / 5.70 pp |
| baseline_wander | 0.872 / 0.611 | 0.815 / 0.537 | 5.73 / 7.40 pp |
| baseline_shift | 0.872 / 0.611 | 0.799 / 0.506 | 7.33 / 10.49 pp |
| random_leads_masking | 0.872 / 0.611 | 0.768 / 0.457 | 10.44 / 15.40 pp |

No-clip is slightly worse than clipped depth=1 (`-0.15 / -0.22 pp` corrupted),
so clipping mismatch is not the missing mechanism.

## Depth=1 Fixed-Mixture Check

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_m075_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_m075_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_m075_ep30_seed20260601`

Change versus depth=1 clipped: raw AugMix mixture coefficient is fixed at
`m=0.75` instead of symmetric beta mixing around mean `0.5`.

Best K500-internal checkpoint was epoch 28: `0.9233 / 0.8234`.

Clean held-out CPSC from the standard PN2021 eval:

- CPSC clean center: `0.8701 / 0.6088`.

Mean calibrated PN2021-C CPSC: clean `0.8701 / 0.6088`, corrupted
`0.8101 / 0.5271`, drop `6.00 / 8.17 pp`.

Per-operator PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.870 / 0.609 | 0.823 / 0.548 | 4.72 / 6.07 pp |
| emg_noise | 0.870 / 0.609 | 0.825 / 0.560 | 4.52 / 4.84 pp |
| baseline_wander | 0.870 / 0.609 | 0.825 / 0.548 | 4.50 / 6.12 pp |
| baseline_shift | 0.870 / 0.609 | 0.808 / 0.519 | 6.23 / 9.00 pp |
| random_leads_masking | 0.870 / 0.609 | 0.770 / 0.460 | 10.05 / 14.83 pp |

Fixed `m=0.75` improves over depth=1 symmetric-beta raw-AugMix by
`+0.44 / +0.64 pp` corrupted, but it still trails calibrated raw-supervised by
about `0.66 / 0.80 pp` corrupted. The clean center score drops slightly versus
depth=1 beta (`-0.23 / -0.21 pp`), while corrupted performance improves, so
the mixture coefficient was a real but small bottleneck.

## Width=1 Depth=1 Fixed-Mixture Limit

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_w1_m100_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601`

Change versus fixed `m=0.75`: raw AugMix uses `width=1`, `depth=1`, and fixed
`m=1.0`, so each raw view is one full-strength PN2021-C calibrated operator
with no clean mixback and no multi-branch averaging. VAE-LH online AT and the
source+target raw pool are otherwise unchanged.

Best K500-internal checkpoint was epoch 28: `0.9188 / 0.8068`.

Clean held-out CPSC from the standard PN2021 eval:

- CPSC clean center: `0.8649 / 0.6022`.

Mean calibrated PN2021-C CPSC: clean `0.8649 / 0.6022`, corrupted
`0.8189 / 0.5364`, drop `4.59 / 6.59 pp`.

Per-operator PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.865 / 0.602 | 0.827 / 0.551 | 3.74 / 5.08 pp |
| emg_noise | 0.865 / 0.602 | 0.828 / 0.564 | 3.68 / 3.83 pp |
| baseline_wander | 0.865 / 0.602 | 0.832 / 0.551 | 3.24 / 5.12 pp |
| baseline_shift | 0.865 / 0.602 | 0.819 / 0.532 | 4.60 / 7.05 pp |
| random_leads_masking | 0.865 / 0.602 | 0.788 / 0.484 | 7.70 / 11.85 pp |

This is the first AugMix-shaped EfficientNet1DV2 run here that matches or
slightly exceeds the calibrated raw-supervised CPSC corrupted mean
(`0.817 / 0.535`). Relative to VAE-LH noAug, it recovers `+6.33 / +6.49 pp`
corrupted performance. The tradeoff is lower clean CPSC than the standard
depth=1 run (`0.8649 / 0.6022` versus `0.8724 / 0.6109`), so the current
boundary recipe buys robustness by moving close to direct corrupted-view
training rather than by preserving classical AugMix smoothing.

## EfficientNet Noise/Mask-only Raw-AugMix CPSC Pilot

Config:

`configs/experiments/effnet_vae_lhat_fullpool_raw_augmix_noisemask_w1_m100_k500_v7_sjr_rgq_cpsc_2018.yaml`

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_noisemask_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_effnet_noisemask_w1_m100_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_noisemask_w1_m100_ep30_seed20260601`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_noisemask_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_effnet_noisemask_w1_m100_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_noisemask_w1_m100_ep30_seed20260601/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json`

Change versus the all-operator `w1/d1/m1.0` run: raw AugMix keeps the same
calibrated severity-5 width/depth/mixture settings, but restricts the raw
corruption ops to `powerline_noise`, `emg_noise`, and
`random_leads_masking`.

Best K500-internal checkpoint was epoch 28: `0.9196 / 0.7957`.

Mean calibrated PN2021-C CPSC: clean `0.8651 / 0.5779`, corrupted
`0.7860 / 0.5013`, drop `7.91 / 7.66 pp`.

Per-operator PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.865 / 0.578 | 0.838 / 0.554 | 2.72 / 2.44 pp |
| emg_noise | 0.865 / 0.578 | 0.830 / 0.571 | 3.53 / 0.72 pp |
| baseline_wander | 0.865 / 0.578 | 0.738 / 0.459 | 12.73 / 11.85 pp |
| baseline_shift | 0.865 / 0.578 | 0.721 / 0.425 | 14.41 / 15.28 pp |
| random_leads_masking | 0.865 / 0.578 | 0.803 / 0.498 | 6.16 / 8.01 pp |

This confirms that the noise/mask-only branch learns stronger invariance to
the trained operators, especially `powerline_noise` and `emg_noise`, but it is
not a viable mean-score recipe by itself because the untrained low-frequency
operators collapse. Compared with all-operator `w1/d1/m1.0`, it improves
`powerline_noise`, `emg_noise`, and masking, but loses about
`9-10 pp` AUROC and `6-10 pp` AUPRC on `baseline_wander` and
`baseline_shift`.

## Width=1 Depth=1 Four-Center Check

Follow-up config:

`configs/experiments/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3.yaml`

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0`

This extends the same EfficientNet1DV2 boundary raw-AugMix recipe from CPSC
to the remaining three target centers. The CPSC result above is combined with
these new runs for the four-center summary.

Per-center mean over five calibrated PN2021-C operators:

| Center | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| cpsc_2018 | 0.865 / 0.602 | 0.819 / 0.536 | 4.59 / 6.59 pp |
| ningbo | 0.883 / 0.514 | 0.821 / 0.413 | 6.28 / 10.08 pp |
| chapman_shaoxing | 0.880 / 0.453 | 0.747 / 0.316 | 13.34 / 13.63 pp |
| georgia | 0.830 / 0.613 | 0.781 / 0.553 | 4.86 / 5.97 pp |
| mean | 0.864 / 0.545 | 0.792 / 0.455 | 7.27 / 9.07 pp |

Per-operator mean over four centers:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.864 / 0.545 | 0.812 / 0.476 | 5.27 / 6.92 pp |
| emg_noise | 0.864 / 0.545 | 0.788 / 0.447 | 7.66 / 9.79 pp |
| baseline_wander | 0.864 / 0.545 | 0.776 / 0.462 | 8.83 / 8.28 pp |
| baseline_shift | 0.864 / 0.545 | 0.806 / 0.462 | 5.88 / 8.28 pp |
| random_leads_masking | 0.864 / 0.545 | 0.777 / 0.425 | 8.71 / 12.06 pp |

Matched four-center references from the calibrated raw-supervised bundle:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| EfficientNet1DV2 VAE-LH noAug | 0.870 / 0.557 | 0.748 / 0.407 | 12.24 / 15.01 pp |
| EfficientNet1DV2 calibrated raw-supervised | 0.866 / 0.551 | 0.811 / 0.472 | 5.48 / 7.95 pp |
| EfficientNet1DV2 boundary raw-AugMix w1/d1/m1.0 | 0.864 / 0.545 | 0.792 / 0.455 | 7.27 / 9.07 pp |

Four-center verdict: the CPSC result does not fully generalize. The boundary
raw-AugMix recipe improves over VAE-LH noAug by `+4.42 / +4.74 pp` corrupted
AUROC/AUPRC, but still trails calibrated raw-supervised by
`-1.96 / -1.73 pp`. The weak center is `chapman_shaoxing`, where the selected
checkpoint stayed close to the direct baseline and baseline_wander still
caused a `23.22 / 16.41 pp` drop.

### Chapman Latest-Checkpoint Selector Diagnostic

Diagnostic model directory:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c`

This extracts `model_state_dict` from `checkpoints/checkpoint_latest.pt`
at epoch 30 into a diagnostic `best_model.pt`. It is not a paper-safe selected
checkpoint; it tests whether the four-center gap is caused by training or by
K500 target-val checkpoint selection.

Chapman selected checkpoint versus epoch-30 latest:

| Checkpoint | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| selected/direct checkpoint | 0.880 / 0.453 | 0.747 / 0.316 | 13.34 / 13.63 pp |
| epoch-30 latest diagnostic | 0.887 / 0.476 | 0.827 / 0.386 | 5.92 / 9.02 pp |

Per-operator Chapman epoch-30 latest:

| Operator | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.854 / 0.421 | 3.30 / 5.52 pp |
| emg_noise | 0.802 / 0.348 | 8.43 / 12.85 pp |
| baseline_wander | 0.841 / 0.416 | 4.57 / 6.01 pp |
| baseline_shift | 0.839 / 0.399 | 4.72 / 7.77 pp |
| random_leads_masking | 0.801 / 0.347 | 8.58 / 12.94 pp |

If CPSC, Ningbo, and Georgia keep their selected checkpoints while Chapman
uses this epoch-30 diagnostic checkpoint, four-center mean becomes clean
`0.8661 / 0.5513`, corrupted `0.8119 / 0.4721`, drop `5.41 / 7.91 pp`.
That is effectively tied with calibrated raw-supervised (`0.8114 / 0.4720`,
drop `5.48 / 7.95 pp`). Therefore the EfficientNet boundary raw-AugMix gap is
mostly a checkpoint-selection problem, not an operator-amplitude or training
coverage problem.

Georgia was also checked with the epoch-30 latest checkpoint to test whether
the latest-checkpoint direction only helps the weak Chapman center. Georgia
selected checkpoint mean was clean `0.8295 / 0.6127`, corrupted
`0.7809 / 0.5530`, drop `4.86 / 5.97 pp`. Georgia epoch-30 latest was clean
`0.8293 / 0.6128`, corrupted `0.7815 / 0.5538`, drop `4.78 / 5.90 pp`.
This confirms that adding a robust checkpoint candidate need not damage an
already stable center.

### K500 Internal Robust Selector Diagnostic

Selector helper:

`scripts/paper/select_effnet_robust_checkpoint.py`

Authoritative selector bundle:

`/home/linbinhao/ECG_adv_data/runs/effnet_robust_selector_w1_m100_k500_v7_authoritative_20260616/selector_bundle.json`

Rule:

- candidates: `best_model.pt` and `checkpoints/checkpoint_latest.pt`;
- validation data: only the target-center K500 internal split
  (`target_real_npz` under `k500_seed20260601`,
  `target_real_val_fraction=0.2`, split seed `20260601`);
- corruption view: the same five `calibrated_10to20pp` severity-5 operators;
- metric: select highest corrupted K500-val macro AUPRC;
- clean floor: reject candidates more than `0.02` AUPRC or `0.03` AUROC below
  the best clean K500-val candidate;
- held-out PN2021 / PN2021-C labels used for selection: `false`.

K500 internal selector results:

| Center | Selector choice | K500 clean AUROC / AUPRC | K500 corrupted AUROC / AUPRC | Corrupted AUPRC gain vs selected |
|---|---:|---:|---:|---:|
| cpsc_2018 | latest | 0.918 / 0.805 | 0.872 / 0.736 | +0.36 pp |
| ningbo | latest | 0.906 / 0.860 | 0.847 / 0.777 | +0.01 pp |
| chapman_shaoxing | latest | 0.899 / 0.781 | 0.835 / 0.698 | +6.96 pp |
| georgia | latest | 0.862 / 0.751 | 0.815 / 0.696 | +0.18 pp |

PN2021-C external diagnostic after applying the selector choice:

| Center | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| cpsc_2018 | 0.864 / 0.601 | 0.817 / 0.536 | 4.69 / 6.52 pp |
| ningbo | 0.883 / 0.513 | 0.821 / 0.413 | 6.24 / 10.01 pp |
| chapman_shaoxing | 0.887 / 0.476 | 0.827 / 0.386 | 5.92 / 9.02 pp |
| georgia | 0.829 / 0.613 | 0.782 / 0.554 | 4.78 / 5.90 pp |
| mean | 0.866 / 0.551 | 0.812 / 0.472 | 5.41 / 7.86 pp |

Matched four-center references:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| EfficientNet1DV2 VAE-LH noAug | 0.870 / 0.557 | 0.748 / 0.407 | 12.24 / 15.01 pp |
| EfficientNet1DV2 calibrated raw-supervised | 0.866 / 0.552 | 0.811 / 0.472 | 5.48 / 7.95 pp |
| EfficientNet1DV2 boundary raw-AugMix + robust selector | 0.866 / 0.551 | 0.812 / 0.472 | 5.41 / 7.86 pp |

The robust selector raises boundary raw-AugMix from selected-checkpoint
`0.792 / 0.455` corrupted mean to `0.812 / 0.472`. Relative to VAE-LH noAug,
the full EfficientNet recipe now recovers `+6.41 / +6.49 pp` corrupted
AUROC/AUPRC and reduces the drop by `6.83 / 7.14 pp`. The remaining gap to
calibrated raw-supervised is effectively closed on this four-center diagnostic
(`+0.02 / +0.02 pp` corrupted in favor of selector raw-AugMix).

The selector bundle consolidates the four original per-run diagnostic JSONs
and the selected external PN2021-C eval JSONs. It should be treated as the
source artifact for this EfficientNet1DV2 result. A separate audit rerun with
`k500_seed20260531` produced different K500-val scores for CPSC/Ningbo and is
not comparable to the trained `seed20260601` boundary raw-AugMix runs.

### Post-hoc Input-Stabilizer Follow-up

Four-center stabilizer summary:

`/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/effnet_selector_input_stabilizer_fourcenter_summary.csv`

Setup:

- same robust-selector model choices as above;
- PN2021-C `calibrated_10to20pp`, severity `5`;
- same ref-excluded `crop_len=1000` evaluator path;
- inference-time stabilizer: `0.5-35 Hz` bandpass, flat-lead repair, and
  per-sample global re-normalization after stabilizer;
- no retraining was performed for this diagnostic.

Per-center mean over five calibrated PN2021-C operators:

| Center | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs selector corrupted |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 0.887 / 0.473 | 0.852 / 0.422 | 3.53 / 5.14 pp | +2.44 / +3.57 pp |
| cpsc_2018 | 0.865 / 0.603 | 0.844 / 0.575 | 2.16 / 2.87 pp | +2.65 / +3.84 pp |
| georgia | 0.831 / 0.612 | 0.802 / 0.575 | 2.89 / 3.62 pp | +2.01 / +2.17 pp |
| ningbo | 0.884 / 0.515 | 0.847 / 0.454 | 3.67 / 6.07 pp | +2.68 / +4.16 pp |
| mean | 0.867 / 0.551 | 0.836 / 0.507 | 3.06 / 4.43 pp | +2.45 / +3.43 pp |

This moves EfficientNet1DV2 from the robust-selector row
`0.8117 / 0.4722` to `0.8362 / 0.5065` on the four-center PN2021-C mean.
Relative to VAE-LH noAug, the total gain is now `+8.86 / +9.92 pp`
corrupted AUROC/AUPRC. The gain mainly comes from removing low-frequency
drift/shift and masked-lead artifacts at inference time; EMG remains the
weakest operator and is slightly hurt by this stabilizer on several centers.

Because this is a post-hoc inference-time diagnostic, the method is not yet the
same as training the stabilizer behavior into the AugMix branch. It is,
however, the first EfficientNet1DV2 four-center result in this run family that
comes close to the requested `+10 pp` recovery target without held-out PN2021-C
label selection.

### EfficientNet Hard-op Weighted Chapman Pilot

Config:

`configs/experiments/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman.yaml`

Run:

`/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman/20260616_effnet_hardops_chapman_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_hardops_ep30_seed20260601`

Change versus boundary raw-AugMix: the raw corruption op list duplicates the
two weak Chapman operators, using
`powerline_noise, emg_noise, emg_noise, baseline_wander, baseline_wander,
baseline_shift, random_leads_masking` with the same width=1/depth=1/fixed
`m=1.0` calibrated severity-5 view geometry.

Training selected epoch 0 by clean K500 AUPRC, but the K500 internal robust
selector chose epoch-30 latest:

| Candidate | K500 clean AUROC / AUPRC | K500 corrupted AUROC / AUPRC |
|---|---:|---:|
| selected / epoch 0 | 0.904 / 0.785 | 0.766 / 0.629 |
| epoch-30 latest | 0.897 / 0.779 | 0.835 / 0.698 |

External Chapman PN2021-C for the epoch-30 latest diagnostic:

| Operator | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.849 / 0.413 | 3.25 / 4.77 pp |
| emg_noise | 0.815 / 0.362 | 6.74 / 9.91 pp |
| baseline_wander | 0.847 / 0.424 | 3.46 / 3.69 pp |
| baseline_shift | 0.840 / 0.410 | 4.20 / 5.04 pp |
| random_leads_masking | 0.789 / 0.328 | 9.32 / 13.22 pp |
| mean | 0.828 / 0.387 | 5.39 / 7.32 pp |

The hard-op weighting substantially improves the K500 internal corrupted
validation view, but it does not improve external Chapman PN2021-C in a
meaningful way. Versus the original Chapman robust-selector result
(`0.827 / 0.386` corrupted), the external gain is only about
`+0.10 / +0.14 pp`. This closes no additional four-center gap and should not
replace the current EfficientNet robust-selector result.

### EfficientNet Post-hoc Stabilizer Diagnostic

Partial Chapman-only output:

`/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/effnet_chapman_posthoc_stabilizer_partial.json`

Summary:

`/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/chapman_summary.md`

This checks whether the deterministic front-end that helped ECGFounder also
helps the EfficientNet weak center. It is deliberately marked partial: the
full four-center job was interrupted after Chapman because flat-lead repair is
slow, and Chapman is the center responsible for most remaining EfficientNet
selector headroom.

| Chapman variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Gain vs selector |
|---|---:|---:|---:|---:|
| selector baseline | 0.887 / 0.476 | 0.827 / 0.386 | 5.92 / 9.02 pp | - |
| renorm only | 0.887 / 0.476 | 0.823 / 0.381 | 6.35 / 9.53 pp | -0.43 / -0.51 pp |
| mask repair + renorm | 0.887 / 0.476 | 0.830 / 0.391 | 5.64 / 8.58 pp | +0.28 / +0.44 pp |
| bandpass 0.5-35 + mask repair + renorm | 0.887 / 0.474 | 0.852 / 0.422 | 3.53 / 5.16 pp | +2.45 / +3.58 pp |

Per-operator signal:

- bandpass solves `powerline_noise` and `baseline_shift` almost completely on
  Chapman;
- mask repair improves `random_leads_masking` from about `0.800 / 0.350` to
  `0.838 / 0.396`;
- `emg_noise` remains poor (`0.791 / 0.334`) and becomes the main remaining
  blocker.

This gives an additional weak-center signal, but not a global 10 pp solution
by itself. If the bandpass result generalized to all centers, the four-center
EfficientNet mean would move materially above the selector result, but the
remaining EMG drop means the next useful experiment should combine the
boundary raw-AugMix selector with an EMG-specific loss or filter path rather
than another global AugMix-strength increase.

### EfficientNet Chapman EMG Bandpass Sweep

Output:

`/home/linbinhao/ECG_adv_data/runs/effnet_emg_bandpass_sweep_20260616/chapman_clean_emg_bandpass_sweep.json`

Summary:

`/home/linbinhao/ECG_adv_data/runs/effnet_emg_bandpass_sweep_20260616/summary.md`

This isolates the remaining EffNet weak point after the post-hoc stabilizer
diagnostic: `emg_noise`. It evaluates only Chapman clean and EMG, using the
same selected checkpoint from the robust selector bundle. It does not use
held-out labels for selection; held-out labels are used only for this
diagnostic measurement.

| Chapman cutoff | Clean AUROC / AUPRC | EMG AUROC / AUPRC | EMG drop AUROC / AUPRC | EMG gain vs selector |
|---|---:|---:|---:|---:|
| selector baseline | 0.887 / 0.476 | 0.802 / 0.348 | 8.43 / 12.85 pp | - |
| renorm only, no bandpass | 0.887 / 0.476 | 0.783 / 0.324 | 10.32 / 15.19 pp | -1.89 / -2.34 pp |
| bandpass 0.5-45 | 0.887 / 0.472 | 0.789 / 0.328 | 9.84 / 14.39 pp | -1.36 / -1.94 pp |
| bandpass 0.5-40 | 0.887 / 0.473 | 0.790 / 0.331 | 9.73 / 14.20 pp | -1.24 / -1.69 pp |
| bandpass 0.5-35 | 0.887 / 0.474 | 0.791 / 0.334 | 9.59 / 13.94 pp | -1.10 / -1.36 pp |
| bandpass 0.5-30 | 0.887 / 0.472 | 0.794 / 0.336 | 9.22 / 13.66 pp | -0.80 / -1.20 pp |
| bandpass 0.5-25 | 0.883 / 0.470 | 0.795 / 0.338 | 8.76 / 13.13 pp | -0.70 / -0.94 pp |
| bandpass 0.5-20 | 0.872 / 0.451 | 0.793 / 0.337 | 7.96 / 11.41 pp | -0.98 / -1.10 pp |
| bandpass 0.5-15 | 0.835 / 0.408 | 0.778 / 0.324 | 5.76 / 8.46 pp | -2.47 / -2.41 pp |

The best cutoff by EMG AUPRC is `0.5-25 Hz`, but it still trails the selector
baseline by `0.70 / 0.94 pp` on EMG and loses clean `0.37 / 0.66 pp`.
Therefore EffNet's remaining EMG gap is not a post-hoc low-pass problem. The
boundary raw-AugMix model has already learned a stronger EMG decision boundary
than any tested inference-time frequency filter. The next aligned experiment
should be training-time EMG specialization, for example an EMG-only auxiliary
branch or EMG-weighted checkpoint objective, not a deterministic inference
filter.

The selected-vs-latest check confirms this is not just the wrong checkpoint:
Chapman selected checkpoint has EMG `0.766 / 0.312`, while the robust-selector
latest checkpoint has EMG `0.802 / 0.348`. The selector already chose the
better EMG checkpoint; the remaining gap needs a different training signal.

## ECGFounder Width=1 Depth=1 Fixed-Mixture Port

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_vae_lhat_fullpool_raw_augmix_w1_m100_last_epoch_k500_v7_sjr_rgq_cpsc_2018/20260616_ecgfounder_rawaugmix_w1_m100_last_cpsc_gpu0/cpsc_2018/runs/cpsc_2018_K500_M20_lam0p15_ep20_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_pn2021c_calibrated_10to20pp_20260616/rawaugmix_w1_m100_last_epoch/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus ECGFounder calibrated raw-supervised: the raw corruption branch
uses the same AugMix boundary condition that worked for EfficientNet:
`view_mode=augmix`, `width=1`, `depth=1`, fixed `m=1.0`,
`severity_profile=calibrated_10to20pp`, severity `5`, source+target raw pool,
`max_batches=64`, no-renorm, and clip `6.0`. The run is explicitly marked
`exploratory` because checkpoint selection is forced to `last_epoch`.

Final training checkpoint:

- best/selected epoch: `20`, selected by `selection_metric=last_epoch`;
- target K500 internal val: `0.8914 / 0.8070`;
- PTB-XL source fold10: `0.9011 / 0.7550`;
- epoch 1 raw branch generated `16384` views with all five operators
  represented in `op_counts`;
- checkpoint SHA256:
  `655e5658a2bdc0ab901febe407680a9d337f1a09416d468d288c788a43673ce2`.

Mean calibrated PN2021-C CPSC: clean `0.8687 / 0.6269`, corrupted
`0.8070 / 0.5446`, drop `6.17 / 8.23 pp`.

Per-operator PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.869 / 0.627 | 0.785 / 0.528 | 8.33 / 9.94 pp |
| emg_noise | 0.869 / 0.627 | 0.782 / 0.512 | 8.69 / 11.48 pp |
| baseline_wander | 0.869 / 0.627 | 0.814 / 0.556 | 5.43 / 7.06 pp |
| baseline_shift | 0.869 / 0.627 | 0.848 / 0.596 | 2.06 / 3.14 pp |
| random_leads_masking | 0.869 / 0.627 | 0.805 / 0.532 | 6.32 / 9.52 pp |

This port does not reproduce the EfficientNet gain. Relative to ECGFounder
direct/epoch-0 selected raw-supervised, it improves corrupted mean by only
`+0.30 / +1.28 pp`. Relative to ECGFounder calibrated raw-supervised forced
last epoch, it is essentially tied (`+0.21 / +0.08 pp` corrupted). The same
boundary AugMix graph that closes the EfficientNet CPSC gap is therefore not
the missing ECGFounder mechanism.

## ECGFounder FullFT Direct/VAE Check

FullFT direct init-head run:

`/home/linbinhao/ECG_adv_data/paper_ecgfounder_fullft_direct_inithead_lowlr_threads8_seed20260531_v6_20260525/runs/cpsc_2018_K500_fullft_ep10_lr2e-05_sw1p0_tw40p0_fullft_inithead_ihd_tv100_source_plus_target_val_auprc_tvseed20260531_seed20260531`

FullFT VAE init-head aw20 run:

`/home/linbinhao/ECG_adv_data/paper_ecgfounder_fullft_vae_inithead_aw20_lowlr_threads8_seed20260531_v6_20260525/runs/cpsc_2018_K500_fullft_ep10_lr2e-05_sw1p0_tw40p0_fullft_vae_inithead_aw20p0_ka160_M20_lam0p05_hs5_hlr0p25_labelexact_includeanchor_ihvaw20_tv100_source_plus_target_val_auprc_tvseed20260531_seed20260531`

PN2021-C eval paths:

- direct fullFT:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_pn2021c_calibrated_10to20pp_20260616/direct_inithead/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- VAE fullFT:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_pn2021c_calibrated_10to20pp_20260616/vae_inithead_aw20/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

The ECGFounder PN2021-C evaluator now detects fullFT model checkpoints
(`best_model.pt`) and evaluates the full fine-tuned model directly, instead of
assuming frozen ECGFounder features plus `best_head.pt`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs frozen direct |
|---|---:|---:|---:|---:|
| frozen direct | 0.863 / 0.622 | 0.804 / 0.532 | 5.90 / 9.01 pp | reference |
| fullFT direct init-head | 0.905 / 0.725 | 0.812 / 0.551 | 9.26 / 17.39 pp | +0.82 / +1.92 pp corrupted |
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | 9.36 / 17.71 pp | +1.14 / +2.90 pp corrupted |

Per-operator fullFT VAE PN2021-C:

| Operator | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| powerline_noise | 0.909 / 0.738 | 0.732 / 0.469 | 17.70 / 26.87 pp |
| emg_noise | 0.909 / 0.738 | 0.771 / 0.482 | 13.82 / 25.62 pp |
| baseline_wander | 0.909 / 0.738 | 0.853 / 0.599 | 5.56 / 13.85 pp |
| baseline_shift | 0.909 / 0.738 | 0.890 / 0.692 | 1.91 / 4.61 pp |
| random_leads_masking | 0.909 / 0.738 | 0.831 / 0.562 | 7.79 / 17.59 pp |

Full encoder fine-tuning raises clean CPSC substantially, but it does not
recover calibrated PN2021-C robustness. The corrupted mean improves only
`+0.82 / +1.92 pp` for direct fullFT and `+1.14 / +2.90 pp` for VAE fullFT
versus frozen direct. The drop grows because the clean score rises much more
than the corrupted score, especially for `powerline_noise` and `emg_noise`.
Within the fullFT family, adding the VAE branch contributes only
`+0.32 / +0.98 pp` corrupted over direct fullFT.

This rules out "unfreeze ECGFounder" and "VAE fullFT alone" as the missing
10 pp recovery mechanism. The untested ECGFounder combination is still
fullFT plus a calibrated raw-corruption/AugMix branch; the current ECGFounder
raw-AugMix run was a frozen-encoder feature/head path, while the current fullFT
runs had no raw-corruption branch.

## ECGFounder FullFT VAE + Calibrated Raw-AugMix Check

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_w1_m100_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Setup:

- ECGFounder full fine-tuning, K500 init-head, low LR `2e-5`.
- VAE latent-hull online AT kept enabled: `adv_weight=20`,
  `hull_lambda=0.05`, `hull_steps=5`, exact labels, include anchor.
- Calibrated raw branch added to the fullFT model:
  `scope=source_target`, `max_batches=64`, `copies=2`,
  `severity_profile=calibrated_10to20pp`, severity `5`,
  `view_mode=augmix`, `width=1`, `depth=1`, fixed mixture `m=1.0`,
  no post-corruption renorm, `clip_abs=6.0`.
- `methods/augmix/jsd_loss.py` was fixed to cast logits to fp32 before
  sigmoid/log terms; the earlier fp16 JSD path produced NaNs under extreme
  logits.

Training completed without non-finite guards firing. Best checkpoint by
`source_plus_target_val_auprc` was epoch 8:

- PTB-XL fold10: `0.9141 / 0.7876`.
- CPSC clean ref-excluded: `0.9012 / 0.7193`.
- CPSC drop-all-zero ref-excluded: `0.9463 / 0.8744`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta |
|---|---:|---:|---:|---:|
| frozen direct | 0.863 / 0.622 | 0.804 / 0.532 | 5.90 / 9.01 pp | reference |
| fullFT direct init-head | 0.905 / 0.725 | 0.812 / 0.551 | 9.26 / 17.39 pp | +0.82 / +1.92 pp vs frozen direct |
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | 9.36 / 17.71 pp | +1.14 / +2.90 pp vs frozen direct |
| fullFT VAE + raw-AugMix w1/d1/m1.0 | 0.901 / 0.719 | 0.836 / 0.589 | 6.49 / 13.00 pp | +3.23 / +5.74 pp vs frozen direct |

Per-operator comparison against fullFT VAE:

| Operator | fullFT VAE PN2021-C | fullFT VAE + raw-AugMix PN2021-C | Raw-AugMix gain |
|---|---:|---:|---:|
| powerline_noise | 0.732 / 0.469 | 0.831 / 0.584 | +9.93 / +11.43 pp |
| emg_noise | 0.771 / 0.482 | 0.800 / 0.545 | +2.97 / +6.28 pp |
| baseline_wander | 0.853 / 0.599 | 0.837 / 0.589 | -1.60 / -1.01 pp |
| baseline_shift | 0.890 / 0.692 | 0.874 / 0.649 | -1.56 / -4.29 pp |
| random_leads_masking | 0.831 / 0.562 | 0.838 / 0.580 | +0.69 / +1.78 pp |

This is the first ECGFounder-side result where calibrated raw-AugMix clearly
moves corrupted performance, but it still does not reach the requested
`~10 pp` recovery. It also exposes an operator conflict: raw-AugMix strongly
fixes `powerline_noise`, helps `emg_noise` and `random_leads_masking`, but
hurts the already-robust `baseline_wander` and `baseline_shift` directions.
That means a single mixed raw-AugMix objective is not yet the right
ECGFounder solution.

## ECGFounder FullFT VAE + Noise/Mask-only Raw-AugMix Check

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_noisemask_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_w1_m100_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_noisemask_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the mixed five-op raw-AugMix run:

- Same ECGFounder fullFT VAE recipe and same raw-AugMix boundary settings:
  `width=1`, `depth=1`, fixed mixture `m=1.0`, no post-corruption renorm,
  `clip_abs=6.0`.
- Raw corruption ops restricted to `powerline_noise`, `emg_noise`, and
  `random_leads_masking`.
- Goal: avoid over-training against `baseline_wander` and `baseline_shift`,
  which were already relatively robust and were hurt by the five-op mixed
  objective.

Training completed without non-finite guards firing. Best checkpoint by
`source_plus_target_val_auprc` was epoch 10:

- PTB-XL fold10: `0.9128 / 0.7865`.
- CPSC clean ref-excluded: `0.9035 / 0.7282`.
- CPSC drop-all-zero ref-excluded: `0.9474 / 0.8782`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs fullFT VAE |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | 9.36 / 17.71 pp | reference |
| fullFT VAE + five-op raw-AugMix w1/d1/m1.0 | 0.901 / 0.719 | 0.836 / 0.589 | 6.49 / 13.00 pp | +2.09 / +2.84 pp corrupted |
| fullFT VAE + noise/mask-only raw-AugMix w1/d1/m1.0 | 0.904 / 0.728 | 0.843 / 0.598 | 6.08 / 13.02 pp | +2.74 / +3.71 pp corrupted |

Per-operator comparison against fullFT VAE:

| Operator | fullFT VAE PN2021-C | five-op raw-AugMix gain | noise/mask-only gain |
|---|---:|---:|---:|
| powerline_noise | 0.732 / 0.469 | +9.93 / +11.43 pp | +10.64 / +12.28 pp |
| emg_noise | 0.771 / 0.482 | +2.97 / +6.28 pp | +3.44 / +7.25 pp |
| baseline_wander | 0.853 / 0.599 | -1.60 / -1.01 pp | -0.56 / -1.32 pp |
| baseline_shift | 0.890 / 0.692 | -1.56 / -4.29 pp | -0.83 / -2.17 pp |
| random_leads_masking | 0.831 / 0.562 | +0.69 / +1.78 pp | +1.00 / +2.53 pp |

Restricting the raw-AugMix branch to noise/masking reduces the operator
conflict and gives the best ECGFounder corrupted mean so far in this
diagnostic (`0.843 / 0.598`). It still does not achieve the requested
`~10 pp` global recovery. The pattern is now clearer: ECGFounder can learn
large invariance for `powerline_noise`, moderate invariance for `emg_noise`
and masking, but the shared objective still trades off against drift/shift
directions. The next ECGFounder variant should be operator-aware rather than
stronger globally.

## ECGFounder FullFT VAE + Drift/Shift-only Raw-AugMix Check

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_driftshift_matched_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_driftshift_matched_w1_m100_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_driftshift_matched_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the noise/mask-only run:

- Same matched ECGFounder fullFT VAE recipe:
  `k_anchor=160`, exact labels, include-anchor, `ptbxl_vae_cache` under
  `/home/linbinhao/ECG_adv_data/ptbxl/`.
- Same raw-AugMix boundary settings: `width=1`, `depth=1`, fixed mixture
  `m=1.0`, no post-corruption renorm, `clip_abs=6.0`.
- Raw corruption ops restricted to `baseline_wander` and `baseline_shift`.
- A first attempted drift/shift command omitted those VAE flags and is not used
  for comparison; the numbers below are from the matched rerun only.

Training completed without non-finite guards firing. Best checkpoint by
`source_plus_target_val_auprc` was epoch 10:

- PTB-XL fold10: `0.9190 / 0.7975`.
- CPSC clean ref-excluded: `0.9070 / 0.7348`.
- CPSC drop-all-zero ref-excluded: `0.9489 / 0.8805`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs fullFT VAE |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | 9.36 / 17.71 pp | reference |
| fullFT VAE + noise/mask-only raw-AugMix w1/d1/m1.0 | 0.904 / 0.728 | 0.843 / 0.598 | 6.08 / 13.02 pp | +2.74 / +3.71 pp corrupted |
| fullFT VAE + drift/shift-only raw-AugMix w1/d1/m1.0 | 0.907 / 0.735 | 0.810 / 0.552 | 9.68 / 18.25 pp | -0.52 / -0.86 pp corrupted |

Per-operator comparison against fullFT VAE:

| Operator | fullFT VAE PN2021-C | noise/mask-only gain | drift/shift-only gain |
|---|---:|---:|---:|
| powerline_noise | 0.732 / 0.469 | +10.64 / +12.28 pp | -0.06 / -0.05 pp |
| emg_noise | 0.771 / 0.482 | +3.44 / +7.25 pp | +0.50 / +0.67 pp |
| baseline_wander | 0.853 / 0.599 | -0.56 / -1.32 pp | -0.56 / +0.41 pp |
| baseline_shift | 0.890 / 0.692 | -0.83 / -2.17 pp | -1.72 / -3.91 pp |
| random_leads_masking | 0.831 / 0.562 | +1.00 / +2.53 pp | -0.78 / -1.41 pp |

This falsifies the simple two-branch hypothesis where drift/shift just needed
its own high-weight raw-AugMix branch. Drift/shift-only training preserves
clean CPSC well, but it does not improve the drift/shift stressors: it gives
only a tiny `baseline_wander` AUPRC increase and worsens `baseline_shift`.
It also loses the large noise robustness gained by the noise/mask-only branch.
For ECGFounder, the useful direction is therefore not a symmetric high-weight
operator split.

Diagnostic operator-aware selector upper bound using the existing ECGFounder
candidate pool (`fullFT VAE`, five-op raw-AugMix, noise/mask-only raw-AugMix,
and matched drift/shift-only raw-AugMix):

| Selection rule | PN2021-C AUROC / AUPRC | Gain vs noise/mask-only |
|---|---:|---:|
| Best existing candidate per operator by AUROC | 0.846 / 0.605 | +0.28 / +0.70 pp |
| Best existing candidate per operator by AUPRC | 0.844 / 0.606 | +0.16 / +0.78 pp |
| Pre-specified noise/mask -> noiseMask, drift/shift -> fullFT VAE | 0.846 / 0.605 | +0.28 / +0.70 pp |

This upper bound is diagnostic only because it uses PN2021-C held-out operator
metrics to choose candidates. It is still useful: even an oracle-like
operator selector over the current candidates cannot approach the desired
`~10 pp` recovery. The next ECGFounder step therefore needs a genuinely new
candidate or a training objective that preserves the noise/mask gains while
not degrading drift/shift, not just post-hoc selection among these branches.

Low-weight five-op diagnostic:

The existing fullFT raw-AugMix interface can only express one raw-corruption
branch, so it cannot directly encode "noise/mask main branch plus low-weight
drift/shift regularizer". As a no-code diagnostic, the same five-op
`w=1/d=1/m=1.0` run was repeated with the total raw branch weight reduced from
`raw_bce/jsd = 1.0 / 2.0` to `0.25 / 0.5`.

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs noise/mask-only |
|---|---:|---:|---:|
| five-op raw-AugMix high weight (`1.0 / 2.0`) | 0.901 / 0.719 | 0.836 / 0.589 | -0.65 / -0.87 pp |
| five-op raw-AugMix low weight (`0.25 / 0.5`) | 0.907 / 0.740 | 0.835 / 0.593 | -0.75 / -0.50 pp |
| noise/mask-only raw-AugMix high weight (`1.0 / 2.0`) | 0.904 / 0.728 | 0.843 / 0.598 | reference |

Per-operator low-weight five-op deltas versus noise/mask-only:

| Operator | Low-weight five-op PN2021-C | Delta vs noise/mask-only |
|---|---:|---:|
| powerline_noise | 0.824 / 0.584 | -1.48 / -0.82 pp |
| emg_noise | 0.794 / 0.532 | -1.15 / -2.19 pp |
| baseline_wander | 0.838 / 0.591 | -0.96 / +0.44 pp |
| baseline_shift | 0.882 / 0.676 | +0.06 / +0.61 pp |
| random_leads_masking | 0.839 / 0.582 | -0.20 / -0.54 pp |

Lowering the shared five-op raw branch preserves clean CPSC better but still
does not beat the noise/mask-only robustness result. This rules out "the
five-op branch was simply too strong" as the main explanation. This motivated
the two-branch diagnostic below: keep high-weight hard-label training for
noise/masking, and add a separate low-weight soft branch for drift/shift.

## ECGFounder Two-branch Raw-AugMix Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_twobranch_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_twobranch_noisehard_driftsoft_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_twobranch_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus noise/mask-only:

- Primary branch kept the best noise/mask recipe:
  `powerline_noise`, `emg_noise`, `random_leads_masking`,
  `raw_bce_weight=1.0`, `raw_consistency_loss=jsd`,
  `raw_consistency_weight=2.0`, `max_batches=64`.
- Auxiliary branch added drift/shift views:
  `baseline_wander`, `baseline_shift`, `raw_bce_weight=0.0`,
  `raw_consistency_loss=soft_bce`, `raw_consistency_weight=0.25`,
  `max_batches=32`.
- VAE-LH online AT, fullFT init-head recipe, raw view geometry
  (`width=1`, `depth=1`, fixed `m=1.0`), no-renorm, and clipping were kept
  matched to the previous best ECGFounder raw-AugMix runs.

Training completed and selected epoch 10 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9130 / 0.7868`.
- CPSC clean ref-excluded: `0.9028 / 0.7275`.
- CPSC drop-all-zero ref-excluded: `0.9470 / 0.8777`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs fullFT VAE | Delta vs noise/mask-only |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | reference | -2.74 / -3.71 pp |
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | +2.74 / +3.71 pp | reference |
| fullFT VAE + two-branch raw-AugMix | 0.903 / 0.728 | 0.842 / 0.598 | +2.65 / +3.70 pp | -0.09 / -0.01 pp |

Per-operator two-branch deltas versus noise/mask-only:

| Operator | Two-branch PN2021-C | Delta vs noise/mask-only |
|---|---:|---:|
| powerline_noise | 0.839 / 0.594 | +0.06 / +0.20 pp |
| emg_noise | 0.806 / 0.555 | +0.10 / +0.09 pp |
| baseline_wander | 0.845 / 0.587 | -0.29 / +0.07 pp |
| baseline_shift | 0.880 / 0.667 | -0.19 / -0.32 pp |
| random_leads_masking | 0.840 / 0.586 | -0.13 / -0.09 pp |

This falsifies the next simple operator-aware hypothesis. A separate soft
drift/shift branch does not add meaningful robustness on top of the
noise/mask-only branch; it reproduces the same corrupted mean within
`0.1 pp`. The remaining ECGFounder gap is therefore not solved by splitting
operators into hard and soft raw-AugMix branches with the current fullFT/VAE
training recipe. The next candidate needs a materially different mechanism,
for example teacher logits from a more robust raw-supervised model,
feature-space regularization tied to ECGFounder representations, or a
checkpoint/model-selection rule that is explicitly trained to preserve
clean CPSC while optimizing worst-operator robustness.

## ECGFounder Two-branch + Drift/Shift Low Hard-BCE Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_auxdriftbce025_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_auxdriftbce025_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_auxdriftbce025_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the previous two-branch run:

- Primary branch kept the best noise/mask recipe:
  `powerline_noise`, `emg_noise`, `random_leads_masking`,
  `raw_bce_weight=1.0`, `raw_consistency_loss=jsd`,
  `raw_consistency_weight=2.0`, `max_batches=64`.
- Auxiliary branch used drift/shift hard labels at low weight:
  `baseline_wander`, `baseline_shift`,
  `raw_corrupt_aux_bce_weight=0.25`,
  `raw_corrupt_aux_consistency_weight=0.0`,
  `raw_corrupt_aux_teacher_weight=0.0`,
  `raw_corrupt_aux_feature_consistency_weight=0.0`, `max_batches=32`.
- VAE-LH online AT, fullFT init-head recipe, raw view geometry
  (`width=1`, `depth=1`, fixed `m=1.0`), no-renorm, and clipping were kept
  matched to the previous best ECGFounder raw-AugMix runs.

Training completed and selected epoch 10 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9136 / 0.7878`.
- CPSC clean ref-excluded: `0.9028 / 0.7270`.
- CPSC drop-all-zero ref-excluded: `0.9469 / 0.8775`.
- The auxiliary drift/shift BCE fell from about `0.595` to `0.496`; with
  weight `0.25`, the auxiliary loss was about `0.124` at epoch 10.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs fullFT VAE | Delta vs noise/mask-only |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | reference | -2.74 / -3.71 pp |
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | +2.74 / +3.71 pp | reference |
| fullFT VAE + noise/mask hard + drift/shift low hard-BCE | 0.903 / 0.727 | 0.841 / 0.597 | +2.61 / +3.61 pp | -0.13 / -0.11 pp |
| fullFT VAE + noise/mask hard + drift/shift teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | +2.76 / +4.03 pp | +0.02 / +0.32 pp |

Per-operator low hard-BCE deltas versus noise/mask-only:

| Operator | Low hard-BCE PN2021-C | Delta vs noise/mask-only |
|---|---:|---:|
| powerline_noise | 0.839 / 0.594 | +0.02 / +0.16 pp |
| emg_noise | 0.806 / 0.555 | +0.05 / +0.07 pp |
| baseline_wander | 0.845 / 0.586 | -0.26 / -0.02 pp |
| baseline_shift | 0.879 / 0.665 | -0.28 / -0.53 pp |
| random_leads_masking | 0.839 / 0.585 | -0.19 / -0.21 pp |

This closes another gap in the ablation matrix. A low-weight hard-label
drift/shift auxiliary branch does not fix the conflict: it slightly preserves
the noise/mask gains but still fails to improve `baseline_wander` or
`baseline_shift`, and it is below both noise/mask-only and teacher-distill on
the five-operator mean. The current ECGFounder failure is therefore not just
that the drift/shift branch used soft labels or had too little total weight.

## ECGFounder Two-branch + FullFT VAE Teacher Distill Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the previous two-branch run:

- Primary branch kept the best noise/mask recipe:
  `powerline_noise`, `emg_noise`, `random_leads_masking`,
  `raw_bce_weight=1.0`, `raw_consistency_loss=jsd`,
  `raw_consistency_weight=2.0`, `max_batches=64`.
- Auxiliary drift/shift branch removed self-teacher consistency and hard BCE:
  `raw_corrupt_aux_bce_weight=0.0`,
  `raw_corrupt_aux_consistency_weight=0.0`.
- Auxiliary branch instead distilled a frozen fullFT VAE teacher on corrupted
  drift/shift views:
  `teacher_model_path` was the fullFT VAE init-head aw20 `best_model.pt`,
  `teacher_weight=1.0`, `teacher_loss=soft_bce`,
  `teacher_view=corrupt`, `max_batches=32`.
- VAE-LH online AT, fullFT init-head recipe, raw view geometry
  (`width=1`, `depth=1`, fixed `m=1.0`), no-renorm, and clipping were kept
  matched to the previous best ECGFounder raw-AugMix runs.

Training completed and selected epoch 10 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9138 / 0.7889`.
- CPSC clean ref-excluded: `0.9040 / 0.7306`.
- CPSC drop-all-zero ref-excluded: `0.9471 / 0.8773`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs fullFT VAE | Delta vs noise/mask-only |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | reference | -2.74 / -3.71 pp |
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | +2.74 / +3.71 pp | reference |
| fullFT VAE + two-branch raw-AugMix | 0.903 / 0.728 | 0.842 / 0.598 | +2.65 / +3.70 pp | -0.09 / -0.01 pp |
| fullFT VAE + noise/mask hard + drift/shift teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | +2.76 / +4.03 pp | +0.02 / +0.32 pp |

Per-operator teacher-distill deltas versus noise/mask-only:

| Operator | Teacher-distill PN2021-C | Delta vs noise/mask-only |
|---|---:|---:|
| powerline_noise | 0.840 / 0.595 | +0.16 / +0.27 pp |
| emg_noise | 0.806 / 0.557 | +0.07 / +0.25 pp |
| baseline_wander | 0.847 / 0.594 | -0.12 / +0.81 pp |
| baseline_shift | 0.881 / 0.672 | -0.05 / +0.17 pp |
| random_leads_masking | 0.842 / 0.588 | +0.05 / +0.10 pp |

The frozen fullFT VAE teacher gives a small AUPRC improvement over
noise/mask-only, mostly by raising `baseline_wander`, but it is still a
sub-`0.5 pp` change on the five-operator mean. This confirms that simply
adding a clean/fullFT VAE teacher to the drift/shift branch is not the missing
ECGFounder mechanism. A useful teacher would likely need to be explicitly
robust on corrupted drift/shift views, or the regularization must move to an
ECGFounder feature-space objective rather than only matching output logits.

## ECGFounder Two-branch + EfficientNet Robust Teacher Distill Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetteacher_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_effnetteacher_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetteacher_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Code change:

- ECGFounder fullFT raw-corruption teacher loading now supports
  `--raw_corrupt_aux_teacher_type efficientnet1dv2` in addition to the
  previous `ecgfounder_fullft` teacher.
- EfficientNet teachers are loaded through the Super5 model zoo and marked as
  `raw1000` teachers, so they consume the calibrated corrupted `(B, 12, 1000)`
  raw input directly instead of the ECGFounder 500 Hz input conversion.
- The selected EfficientNet teacher was the CPSC boundary raw-AugMix robust
  selector `latest` checkpoint. Its K500 internal corrupted validation score
  was `0.872 / 0.736`, and its four-center external PN2021-C result matched
  calibrated raw-supervised (`0.812 / 0.472` mean).

Matched training setup:

- Primary branch kept the best ECGFounder noise/mask recipe:
  `powerline_noise`, `emg_noise`, `random_leads_masking`,
  hard BCE + JSD, `max_batches=64`.
- Auxiliary branch kept only `baseline_wander` and `baseline_shift`, with
  `teacher_weight=1.0`, `teacher_loss=soft_bce`, `teacher_view=corrupt`,
  and `max_batches=32`.
- VAE-LH online AT, fullFT init-head, raw view geometry (`width=1`, `depth=1`,
  fixed `m=1.0`), no-renorm, and clipping were kept matched to the
  ECGFounder fullFT-teacher distill run.

Training completed and selected epoch 8 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9120 / 0.7852`.
- CPSC clean ref-excluded: `0.9025 / 0.7232`.
- CPSC drop-all-zero ref-excluded: `0.9471 / 0.8753`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs fullFT-teacher distill | Delta vs noise/mask-only |
|---|---:|---:|---:|---:|
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | -0.02 / -0.32 pp | reference |
| fullFT VAE + noise/mask hard + drift/shift fullFT-teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | reference | +0.02 / +0.32 pp |
| fullFT VAE + noise/mask hard + drift/shift EfficientNet-teacher distill | 0.903 / 0.723 | 0.843 / 0.595 | -0.01 / -0.67 pp | +0.01 / -0.35 pp |

Per-operator EfficientNet-teacher PN2021-C:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.838 / 0.591 | 6.42 / 13.19 pp |
| emg_noise | 0.805 / 0.554 | 9.77 / 16.93 pp |
| baseline_wander | 0.849 / 0.580 | 5.31 / 14.32 pp |
| baseline_shift | 0.882 / 0.665 | 2.04 / 5.82 pp |
| random_leads_masking | 0.840 / 0.582 | 6.29 / 14.08 pp |

This is a stronger negative result than the previous ECGFounder-teacher run:
the teacher was explicitly robust under the same calibrated corruption family,
but output-level soft distillation still did not transfer robustness into
ECGFounder. The issue is therefore not only that the ECGFounder teacher was
weak on corrupted views. A useful cross-backbone teacher likely needs either
feature/representation matching, intermediate pseudo-label filtering, or a
student objective that keeps ECGFounder representations invariant rather than
only matching final logits on two drift/shift operators.

## ECGFounder Two-branch + Drift/Shift Feature Consistency Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_featdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_auxfeatnorm100_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_featdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the previous two-branch and teacher-distill runs:

- Primary branch kept the best noise/mask recipe:
  `powerline_noise`, `emg_noise`, `random_leads_masking`,
  `raw_bce_weight=1.0`, `raw_consistency_loss=jsd`,
  `raw_consistency_weight=2.0`, `max_batches=64`.
- Auxiliary branch kept drift/shift views but removed hard/logit objectives:
  `baseline_wander`, `baseline_shift`,
  `raw_corrupt_aux_bce_weight=0.0`,
  `raw_corrupt_aux_consistency_weight=0.0`,
  `raw_corrupt_aux_teacher_weight=0.0`.
- Auxiliary branch instead matched ECGFounder penultimate features between
  clean and corrupted drift/shift views:
  `raw_corrupt_aux_feature_consistency_weight=100.0`,
  `raw_corrupt_feature_consistency_normalize=true`, `max_batches=32`.
- VAE-LH online AT, fullFT init-head recipe, raw view geometry
  (`width=1`, `depth=1`, fixed `m=1.0`), no-renorm, and clipping were kept
  matched to the previous best ECGFounder raw-AugMix runs.

Training completed and selected epoch 9 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9129 / 0.7869`.
- CPSC clean ref-excluded: `0.9025 / 0.7264`.
- CPSC drop-all-zero ref-excluded: `0.9469 / 0.8771`.
- Auxiliary normalized feature loss was tiny: about `0.00029`, so the
  weighted auxiliary loss was only about `0.029` even with weight `100`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs fullFT VAE | Delta vs noise/mask-only |
|---|---:|---:|---:|---:|
| fullFT VAE init-head aw20 | 0.909 / 0.738 | 0.815 / 0.561 | reference | -2.74 / -3.71 pp |
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | +2.74 / +3.71 pp | reference |
| fullFT VAE + noise/mask hard + drift/shift teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | +2.76 / +4.03 pp | +0.02 / +0.32 pp |
| fullFT VAE + noise/mask hard + drift/shift feature consistency | 0.902 / 0.726 | 0.842 / 0.598 | +2.69 / +3.66 pp | -0.05 / -0.05 pp |

Per-operator feature-consistency deltas versus noise/mask-only:

| Operator | Feature-consistency PN2021-C | Delta vs noise/mask-only |
|---|---:|---:|
| powerline_noise | 0.839 / 0.594 | +0.04 / +0.20 pp |
| emg_noise | 0.806 / 0.555 | +0.08 / +0.07 pp |
| baseline_wander | 0.848 / 0.586 | -0.01 / +0.03 pp |
| baseline_shift | 0.879 / 0.666 | -0.23 / -0.42 pp |
| random_leads_masking | 0.840 / 0.586 | -0.12 / -0.13 pp |

This rules out the first simple feature-space variant. Normalized penultimate
feature MSE between clean and corrupted drift/shift views is too small and too
weakly aligned with PN2021-C robustness to add useful headroom. The best
ECGFounder CPSC AUPRC among these diagnostics remains the teacher-distill
variant (`0.601`), and the feature-consistency variant is slightly below both
teacher-distill and noise/mask-only on the five-operator mean.

## ECGFounder All-op Feature Consistency Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_allop_featnorm300_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_allop_featnorm300_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_allop_featnorm300_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Change versus the drift/shift-only feature-consistency run:

- The raw-AugMix primary branch used all five calibrated PN2021-C operators:
  `powerline_noise`, `emg_noise`, `baseline_wander`, `baseline_shift`,
  `random_leads_masking`.
- The branch was deliberately low-logit-pressure:
  `raw_corrupt_bce_weight=0.25`,
  `raw_corrupt_consistency_weight=0.5`, `loss=jsd`.
- Normalized ECGFounder penultimate feature consistency was increased to
  `raw_corrupt_feature_consistency_weight=300.0`.
- View geometry stayed at the EfficientNet boundary setting:
  `width=1`, `depth=1`, fixed `m=1.0`, no-renorm, clip_abs `6.0`.
- VAE-LH online AT and fullFT init-head recipe stayed matched to the previous
  fullFT VAE diagnostics.

Training completed and selected epoch 10 by `source_plus_target_val_auprc`:

- PTB-XL fold10: `0.9158 / 0.7910`.
- CPSC clean ref-excluded: `0.9068 / 0.7396`.
- CPSC drop-all-zero ref-excluded: `0.9490 / 0.8835`.
- VAE latent attack ASR remained `0.000`, so the run improved clean CPSC but
  did not create visible online adversarial pressure.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs noise/mask-only | Delta vs teacher-distill |
|---|---:|---:|---:|---:|
| fullFT VAE + noise/mask-only raw-AugMix | 0.904 / 0.728 | 0.843 / 0.598 | reference | -0.02 / -0.32 pp |
| fullFT VAE + noise/mask hard + drift/shift teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | +0.02 / +0.32 pp | reference |
| fullFT VAE + all-op normalized feature consistency w=300 | 0.907 / 0.740 | 0.837 / 0.595 | -0.56 / -0.25 pp | -0.58 / -0.57 pp |

Per-operator PN2021-C:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.827 / 0.589 | 7.99 / 15.11 pp |
| emg_noise | 0.798 / 0.536 | 10.86 / 20.32 pp |
| baseline_wander | 0.841 / 0.595 | 6.63 / 14.45 pp |
| baseline_shift | 0.879 / 0.671 | 2.74 / 6.86 pp |
| random_leads_masking | 0.841 / 0.586 | 6.59 / 15.33 pp |

This stronger all-operator feature-invariance variant is also negative. It
preserves or improves clean CPSC relative to the best raw-AugMix diagnostics,
but corrupted mean falls below both noise/mask-only and fullFT-teacher
distill. The failure is concentrated in `emg_noise` and `powerline_noise`, so
forcing a single ECGFounder feature space to be invariant to every calibrated
operator appears to trade away the noise/mask gains rather than adding
drift/shift robustness.

## ECGFounder Input Stabilizer Diagnostic

Diagnostic output:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_fft_filter_diag.json`

Lead-repair output:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_bandpass35_leadrepair_diag.json`

Formal evaluator output:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_input_stabilizer_official_eval_20260616/cpsc_2018/teacherdrift_bandpass35_repair_eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Matched train/eval output:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

This diagnostic does not train a new model. It reloads the current best
ECGFounder AUPRC candidate, `fullFT VAE + noise/mask hard + drift/shift
teacher distill`, and changes only the ECGFounder inference input before the
standard `ecg1000_to_ecgfounder_input` upsample + global z-score step.

Two deterministic stabilizers were tested:

- FFT bandpass at `0.5-35 Hz` on the 100 Hz ECG before upsampling. This removes
  50 Hz powerline and the 60 Hz alias at 40 Hz, while preserving more clean
  CPSC AUPRC than lower cutoffs.
- Flat-lead repair for `random_leads_masking`: reconstruct missing limb leads
  from Einthoven/Goldberger relations when at least two independent limb leads
  remain, and fill missing precordial leads by nearest-neighbor chest-lead
  interpolation. This leaves clean records unchanged because no leads are flat.

Cutoff sweep:

| Input filter | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| none | 0.898 / 0.720 | 0.843 / 0.601 | 5.54 / 11.84 pp |
| lowpass35 | 0.898 / 0.712 | 0.856 / 0.624 | 4.19 / 8.79 pp |
| bandpass 0.5-25 | 0.856 / 0.606 | 0.834 / 0.577 | 2.26 / 2.91 pp |
| bandpass 0.5-30 | 0.884 / 0.667 | 0.851 / 0.611 | 3.35 / 5.62 pp |
| bandpass 0.5-35 | 0.899 / 0.715 | 0.860 / 0.637 | 3.83 / 7.85 pp |
| bandpass 0.5-40 | 0.902 / 0.728 | 0.855 / 0.630 | 4.71 / 9.81 pp |
| bandpass 0.5-45 | 0.902 / 0.731 | 0.854 / 0.631 | 4.75 / 10.01 pp |

`0.5-35 Hz` is the best tradeoff in this diagnostic. Lower cutoffs suppress
EMG drop but destroy clean CPSC AUPRC; higher cutoffs preserve clean better but
let the 60 Hz alias leak back into `powerline_noise`.

Best combined deterministic input stabilizer, rerun through the formal
ECGFounder PN2021-C evaluator with clean recomputed under the same input
stabilizer:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise, bandpass 0.5-35 | 0.899 / 0.715 | 0.00 / 0.00 pp |
| emg_noise, bandpass 0.5-35 | 0.809 / 0.559 | 8.98 / 15.67 pp |
| baseline_wander, bandpass 0.5-35 | 0.853 / 0.605 | 4.58 / 10.99 pp |
| baseline_shift, bandpass 0.5-35 | 0.898 / 0.715 | 0.02 / -0.02 pp |
| random_leads_masking, bandpass 0.5-35 + lead repair | 0.861 / 0.659 | 3.75 / 5.61 pp |

Mean over five operators: clean `0.8986 / 0.7153`, corrupted
`0.8639 / 0.6508`, drop `3.46 / 6.45 pp`.

This is the first ECGFounder-side diagnostic in the current sweep that reaches
the target scale: it improves corrupted mean by `+2.09 / +4.95 pp` over the
previous best teacher-distill candidate and by `+4.86 / +8.98 pp` over the
fullFT VAE baseline. Against the original ECGFounder calibrated raw-supervised
selected-epoch reference (`0.804 / 0.532` corrupted), the gain is about
`+6.0 / +11.9 pp`.

The matched training rerun implements the same `0.5-35 Hz` bandpass and
flat-lead repair path inside the ECGFounder fullFT VAE + raw-AugMix
teacher-distill recipe, then evaluates through the same formal PN2021-C
evaluator with clean recomputed under the stabilizer:

| Variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|---:|
| post-hoc stabilizer on teacher-distill checkpoint | 0.8986 / 0.7153 | 0.8639 / 0.6508 | 3.46 / 6.45 pp |
| matched stabilizer train/eval | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp |
| EMG-only primary raw-AugMix + stabilizer train/eval | 0.8941 / 0.7039 | 0.8633 / 0.6452 | 3.08 / 5.87 pp |

Per-operator matched train/eval result:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.894 / 0.699 | -0.00 / 0.00 pp |
| emg_noise | 0.813 / 0.564 | 8.11 / 13.50 pp |
| baseline_wander | 0.858 / 0.608 | 3.67 / 9.08 pp |
| baseline_shift | 0.894 / 0.699 | 0.02 / 0.01 pp |
| random_leads_masking | 0.861 / 0.631 | 3.35 / 6.79 pp |

Matched training does not improve beyond the post-hoc stabilizer. It slightly
raises the corrupted AUROC mean by `+0.02 pp`, but lowers corrupted AUPRC by
`-1.03 pp` because clean/STTC AUPRC is lower. Relative to the original
ECGFounder calibrated raw-supervised selected-epoch reference (`0.804 / 0.532`
corrupted), it still gives about `+6.0 / +10.8 pp`, so the stabilizer remains
the only ECGFounder-side change that reaches target-scale AUPRC improvement.
The current training implementation applies the stabilizer to already-cached
500 Hz ECGFounder training tensors for source/target clean batches, while the
formal evaluator applies it to raw 100 Hz PN2021-C inputs before upsampling.
That is a useful matched approximation, but not yet a fully rebuilt 100 Hz raw
training cache.

The follow-up EMG-only primary raw-AugMix run uses the same stabilizer but makes
`emg_noise` the only hard raw-corruption branch, with `baseline_wander` and
`baseline_shift` kept only as frozen-teacher auxiliary views. It was intended
to test whether the remaining ECGFounder gap is mostly an underweighted EMG
training signal. It does not beat the post-hoc stabilizer: corrupted mean is
`0.8633 / 0.6452`, about `-0.06 / -0.56 pp` versus post-hoc stabilizer.

Per-operator EMG-only train/eval result:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.894 / 0.704 | -0.00 / 0.00 pp |
| emg_noise | 0.815 / 0.569 | 7.91 / 13.50 pp |
| baseline_wander | 0.858 / 0.613 | 3.59 / 9.12 pp |
| baseline_shift | 0.894 / 0.705 | 0.01 / -0.08 pp |
| random_leads_masking | 0.855 / 0.636 | 3.88 / 6.82 pp |

This marginally improves EMG AUPRC over matched stabilizer (`0.569` versus
`0.564`) but loses on masking and mean AUPRC, so EMG-only reweighting is not a
useful replacement for the post-hoc stabilizer candidate.

The result changes the ECGFounder diagnosis. The missing mechanism is not
stronger AugMix amplitude; it is a front-end stability mismatch. ECGFounder
uses 100 Hz to 500 Hz interpolation plus global z-score with no bandpass,
notch, baseline removal, or flat-lead handling, so PN2021-C noise can dominate
the normalization and missing leads remain semantically impossible. The next
mainline candidate should therefore be a frequency-stabilized ECGFounder
VAE+AugMix run: train and evaluate with the same deterministic input
stabilizer, keep the successful noise/mask + teacher-distill raw-AugMix branch,
and treat EMG as the remaining unsolved operator.

## ECGFounder Stabilizer Cross-center Follow-up

Additional outputs:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/georgia/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/ningbo/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

The same teacher-distill + deterministic input-stabilizer recipe was extended
from CPSC to Chapman-Shaoxing, Georgia, and Ningbo. The first Ningbo parallel
attempt was intentionally interrupted because it created too much shared-server
CPU pressure; the completed run below was rerun as a single low-load GPU job
with `num_workers=0` and thread limits.

| Variant | Center | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---|---:|---:|---:|
| CPSC post-hoc stabilizer | cpsc_2018 | 0.8986 / 0.7153 | 0.8639 / 0.6508 | 3.46 / 6.45 pp |
| CPSC matched train/eval | cpsc_2018 | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp |
| CPSC EMG-only train/eval | cpsc_2018 | 0.8941 / 0.7039 | 0.8633 / 0.6452 | 3.08 / 5.87 pp |
| Chapman teacherdrift + stabilizer | chapman_shaoxing | 0.8901 / 0.5408 | 0.8714 / 0.4979 | 1.87 / 4.28 pp |
| Georgia teacherdrift + stabilizer | georgia | 0.8782 / 0.7057 | 0.8482 / 0.6570 | 3.00 / 4.86 pp |
| Ningbo teacherdrift + stabilizer | ningbo | 0.8955 / 0.5698 | 0.8728 / 0.5272 | 2.27 / 4.27 pp |
| four-center reporting mean | cpsc+chapman+georgia+ningbo | 0.8906 / 0.6329 | 0.8641 / 0.5832 | 2.65 / 4.97 pp |

Four-center per-operator mean for the reporting candidate:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.891 / 0.633 | -0.00 / -0.00 pp |
| emg_noise | 0.835 / 0.524 | 5.55 / 10.85 pp |
| baseline_wander | 0.850 / 0.551 | 4.06 / 8.20 pp |
| baseline_shift | 0.890 / 0.630 | 0.10 / 0.26 pp |
| random_leads_masking | 0.855 / 0.578 | 3.55 / 5.53 pp |

The cross-center follow-up supports the main stabilizer diagnosis: bandpass
and flat-lead repair make `powerline_noise` and `baseline_shift` essentially
non-issues across all four centers, while `emg_noise` and `baseline_wander`
remain the largest residual drops. The result is not a universal `+10 pp`
corrupted-mean gain, because the clean baseline can move down under this
recipe; it is instead a large drop-reduction mechanism. The next ECGFounder
gain still has to come from cleaner 100 Hz train/eval parity or a separate
EMG/baseline-wander-specific front-end/loss design.

Four-center gain against the original ECGFounder PN2021-C baselines:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Stabilizer gain over method, corrupted AUROC / AUPRC |
|---|---:|---:|---:|---:|
| ECGFounder direct, 20260615 | 0.8973 / 0.6325 | 0.8135 / 0.5064 | 8.37 / 12.61 pp | +5.06 / +7.68 pp |
| ECGFounder VAE noAug, 20260615 | 0.9031 / 0.6470 | 0.8169 / 0.5132 | 8.62 / 13.38 pp | +4.72 / +7.00 pp |
| ECGFounder VAE+AugMix, 20260615 | 0.9037 / 0.6455 | 0.8169 / 0.5132 | 8.68 / 13.23 pp | +4.72 / +7.00 pp |
| ECGFounder teacherdrift + stabilizer | 0.8906 / 0.6329 | 0.8641 / 0.5832 | 2.65 / 4.97 pp | reference |

This means the deterministic stabilizer plus teacher-distill raw-AugMix path is
already a `~7 pp` four-center AUPRC improvement over the original ECGFounder
VAE+AugMix baseline, but it is still short of a robust `~10 pp` four-center
claim. The residual gap is concentrated in `emg_noise` and
`baseline_wander`, not in powerline, baseline shift, or lead masking.

### ECGFounder Stabilizer Cutoff Sweep

Output root:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_input_stabilizer_sweep_fourcenter_20260616/`

This diagnostic reuses the same ECGFounder teacher-distill checkpoints and runs
only the two remaining hard operators, `emg_noise` and `baseline_wander`, under
three deterministic input-stabilizer high-cut settings. All rows recompute the
clean metric through the same stabilizer before reporting drops. The purpose is
to test whether a narrower low-pass cutoff can recover the remaining PN2021-C
AUPRC gap without changing the training objective.

Four-center mean over `emg_noise` and `baseline_wander`:

| Stabilizer high cut | Clean AUROC / AUPRC | Corrupted AUROC / AUPRC | Drop AUROC / AUPRC |
|---:|---:|---:|---:|
| 25 Hz | 0.8790 / 0.5920 | 0.8379 / 0.5280 | 4.11 / 6.39 pp |
| 30 Hz | 0.8877 / 0.6185 | 0.8424 / 0.5359 | 4.53 / 8.26 pp |
| 35 Hz | 0.8896 / 0.6289 | 0.8437 / 0.5388 | 4.58 / 9.01 pp |

Per-operator four-center mean:

| Stabilizer high cut | Operator | Corrupted AUROC / AUPRC | Drop AUROC / AUPRC |
|---:|---|---:|---:|
| 25 Hz | emg_noise | 0.8360 / 0.5256 | 4.30 / 6.64 pp |
| 25 Hz | baseline_wander | 0.8398 / 0.5305 | 3.93 / 6.15 pp |
| 30 Hz | emg_noise | 0.8365 / 0.5264 | 5.12 / 9.21 pp |
| 30 Hz | baseline_wander | 0.8483 / 0.5454 | 3.94 / 7.31 pp |
| 35 Hz | emg_noise | 0.8363 / 0.5258 | 5.33 / 10.31 pp |
| 35 Hz | baseline_wander | 0.8512 / 0.5517 | 3.84 / 7.72 pp |

The sweep rules out "make the front-end low-pass more aggressive" as the next
large-gain path. A 25 Hz cutoff reduces the reported drop, but it does so by
lowering the clean score and also lowers the absolute corrupted AUPRC
(`0.5280` versus `0.5388` at 35 Hz). The 35 Hz setting remains the best
absolute corrupted-mean choice among this small sweep. Therefore the remaining
ECGFounder gap is not mainly a cutoff-selection problem; it likely needs
training-time 100 Hz train/eval parity or an operator-specific EMG/wander loss
or front-end branch.

### ECGFounder 100 Hz Train/Eval Parity Audit

The current "matched input-stabilizer" ECGFounder fullFT path is matched at the
flag level, but not fully matched at the signal-chain level.

Current code path:

- PN2021-C formal evaluator: loads PN2021 clean/corrupted samples as
  `12 x 1000` tensors, applies the deterministic stabilizer at `100 Hz` inside
  `ecg1000_to_ecgfounder_input(...)`, then upsamples to ECGFounder's
  `12 x 5000` input and applies global z-score.
- Raw-corruption branch during training: takes a raw ECG batch, downsamples to
  `1000` if necessary, builds PN2021-C profiled corruption views, then applies
  `prepare_raw_ecgfounder_input(...)`, which also stabilizes at `100 Hz` before
  upsampling and z-score.
- Main supervised fullFT BCE branch: uses `build_signal_cache(...)` with
  `TARGET_POINTS=5000`, so source PTB-XL and target K500 batches are already
  ECGFounder-style `12 x 5000` tensors. With stabilizer flags enabled, the
  branch calls `prepare_cached_ecgfounder_input(...)`, which filters at
  `500 Hz` and then global-zscores again.

This means the current training setup still mixes two input distributions:
supervised source/target BCE is stabilized in 500 Hz cache space, while
raw-AugMix corruptions and PN2021-C evaluation are stabilized in 100 Hz
corruption/eval space. The available code does not yet expose a pure parameter
switch that makes all training streams use the PN2021-C evaluator's 100 Hz
chain.

The next ECGFounder experiment should therefore add an explicit supervised
input-mode switch:

| Option | Meaning | Expected use |
|---|---|---|
| `cached5000` | Current default: supervised BCE consumes ECGFounder `12 x 5000` caches and optional 500 Hz stabilizer. | Backward-compatible baseline. |
| `raw1000` | Source/target supervised BCE consumes `12 x 1000` tensors and calls `ecg1000_to_ecgfounder_input(...)` for clean, corrupt, and eval paths. | True train/eval parity test for PN2021-C. |

This is the most targeted remaining implementation hypothesis for ECGFounder:
keep the current VAE-LH online AT + raw-AugMix teacher-distill recipe, keep
35 Hz + flat-lead repair, but make the supervised clean stream and corrupted
stream enter ECGFounder through the same 100 Hz stabilizer path before judging
whether EMG/wander need a separate operator-specific loss.

Implementation design for this next test:

1. Add a CLI option to the fullFT pilot script:
   `--supervised_input_mode {cached5000,raw1000}`, defaulting to `cached5000`
   for backward compatibility.
2. For `cached5000`, preserve the current path exactly:
   `build_signal_cache(...) -> build_weighted_signal_stream_loader(...) ->
   prepare_cached_ecgfounder_input(...)`.
3. For `raw1000`, build source/target supervised datasets from 1000-point ECG:
   source PTB-XL can reuse the existing
   `/home/linbinhao/ECG_adv_data/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy`
   plus the matching Super5 label cache; target K500 can reuse the current
   anchor `.signals.npz` files under
   `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets`.
4. Keep the same stream ids and `WeightedRandomSampler` semantics, but call
   `prepare_raw_ecgfounder_input(...)` in the supervised BCE forward when the
   batch is raw1000. This makes source, target, VAE decoded adversarial ECG,
   raw-AugMix corruption views, and PN2021-C evaluator all enter ECGFounder
   through the same `100 Hz -> stabilizer -> 5000 -> z-score` path.
5. Add CPU tests for:
   parser default/backward compatibility; raw1000 mode dispatches the
   supervised forward through `ecg1000_to_ecgfounder_input(...)`; existing
   cached5000 mode still calls `prepare_cached_ecgfounder_input(...)`; and
   the run record stores `supervised_input_mode`.

Recommended first experiment after implementation:

Detailed implementation plan:

`/home/linbinhao/ECG_adv_Gen/docs/superpowers/plans/2026-06-16-ecgfounder-raw1000-parity.md`

| Center | Purpose | Run length |
|---|---|---:|
| `cpsc_2018` | Fast smoke and direct comparison against the existing matched stabilizer CPSC run. | 10 epochs |

Only if CPSC improves the five-operator PN2021-C AUPRC over the current
matched-stabilizer run (`0.6405` CPSC five-op mean, `0.8641 / 0.6405`) should
the raw1000 parity branch be expanded to Chapman, Georgia, and Ningbo. If CPSC
does not improve, the remaining path should move to an explicit
EMG/wander-specific loss/front-end branch instead of more input-chain tuning.

### ECGFounder Raw1000 Parity CPSC Result

Implementation and verification:

- Raw1000 supervised mode was implemented in
  `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py`, with the
  dataset-aware stream-loader helper in `ecg_adv_gen/training/signal_streams.py`.
- CPU plumbing tests passed:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_signal_streams.py util/tests/test_ecgfounder_fullft_raw_augmix.py util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
  -> `33 passed`.
- Training artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_raw1000_stabilizer35_ep10_seed20260531`.
- Formal PN2021-C artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.

Matched CPSC comparison:

| Variant | Supervised input | PTB-XL fold10 AUROC / AUPRC | CPSC clean AUROC / AUPRC | CPSC PN2021-C AUROC / AUPRC | CPSC drop AUROC / AUPRC | Delta vs matched-stabilizer corrupted AUROC / AUPRC |
|---|---|---:|---:|---:|---:|---:|
| teacherdrift + stabilizer reference | cached5000 | 0.9118 / 0.7844 | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp | reference |
| teacherdrift + stabilizer + raw1000 parity | raw1000 | 0.8811 / 0.7365 | 0.9059 / 0.7382 | 0.8743 / 0.6678 | 3.16 / 7.04 pp | +1.02 / +2.73 pp |

Raw1000 parity passes the planned CPSC expansion threshold
(`0.6678 > 0.6405` five-op AUPRC), and it improves the corrupted CPSC mean by
`+2.73 pp` AUPRC over the matched-stabilizer reference. It also recovers a
large part of the earlier no-stabilizer gap: no-stabilizer teacherdrift was
`0.8430 / 0.6012`, so raw1000 reaches `+3.13 / +6.66 pp` corrupted
AUROC/AUPRC over that starting point.

Per-operator raw1000 CPSC result:

| Operator | Corrupted AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.9059 / 0.7382 | 0.00 / 0.00 pp |
| emg_noise | 0.8285 / 0.5796 | 7.75 / 15.86 pp |
| baseline_wander | 0.8616 / 0.6207 | 4.43 / 11.75 pp |
| baseline_shift | 0.9055 / 0.7373 | 0.05 / 0.08 pp |
| random_leads_masking | 0.8701 / 0.6632 | 3.58 / 7.50 pp |

Decision: raw1000 parity is a real target-robustness improvement, but it is
not yet a paper-safe replacement for the cached5000 supervised stream because
PTB-XL fold10 falls from `0.9118 / 0.7844` to `0.8811 / 0.7365`. The next
method should therefore be a hybrid rather than full replacement: keep
cached5000 for source-supervised stability, but route target K500, VAE-decoded
adversarial samples, and raw-AugMix corruption views through the raw1000
100 Hz stabilizer path. If that preserves the PTB-XL source floor while keeping
the `+2-3 pp` CPSC corrupted-AUPRC gain, then expand to the other three
centers. EMG and baseline_wander remain the residual operators; powerline and
baseline_shift are already controlled by the stabilizer.

### ECGFounder Source-Cached Target-Raw1000 Hybrid CPSC Result

Hybrid implementation:

- Added supervised mode `source_cached_target_raw1000`.
- Source PTB-XL supervised BCE remains on cached ECGFounder `12 x 5000`
  tensors.
- Target K500 supervised BCE and VAE decoded adversarial samples are prepared
  through the same raw1000 `100 Hz -> stabilizer -> 5000 -> z-score` path used
  by raw-AugMix and PN2021-C evaluation.
- The supervised forward is stream-aware: stream `0` source batches still get
  cached5000 stabilizer handling, while target/adv rows are already prepared
  and are not stabilized a second time.

Verification:

- CPU tests:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_signal_streams.py util/tests/test_ecgfounder_fullft_raw_augmix.py util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
  -> `36 passed`.
- Training artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_hybrid_sourcecached_targetraw1000_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_hybrid_sourcecached_targetraw1000_stabilizer35_ep10_seed20260531`.
- Formal PN2021-C artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_hybrid_sourcecached_targetraw1000_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.

Matched CPSC comparison after the hybrid test:

| Variant | Supervised input | PTB-XL fold10 AUROC / AUPRC | CPSC clean AUROC / AUPRC | CPSC PN2021-C AUROC / AUPRC | CPSC drop AUROC / AUPRC | Delta vs matched-stabilizer corrupted AUROC / AUPRC |
|---|---|---:|---:|---:|---:|---:|
| teacherdrift + stabilizer reference | cached5000 | 0.9118 / 0.7844 | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp | reference |
| teacherdrift + stabilizer + raw1000 parity | raw1000 | 0.8811 / 0.7365 | 0.9059 / 0.7382 | 0.8743 / 0.6678 | 3.16 / 7.04 pp | +1.02 / +2.73 pp |
| teacherdrift + stabilizer + source-cached target-raw1000 hybrid | source cached, target/adv raw1000-prepared | 0.8839 / 0.7413 | 0.9064 / 0.7388 | 0.8750 / 0.6684 | 3.15 / 7.04 pp | +1.09 / +2.79 pp |

Hybrid gives only a negligible CPSC PN2021-C gain over full raw1000 parity:
`+0.06 pp AUROC / +0.06 pp AUPRC`. It also improves PTB-XL fold10 only
slightly over raw1000 (`+0.28 pp AUROC / +0.48 pp AUPRC`) and remains far below
the cached5000 matched-stabilizer source floor (`0.8839 / 0.7413` versus
`0.9118 / 0.7844`). Therefore the source-floor problem is not fixed by merely
keeping source samples cached while target/adv samples are raw1000-prepared.

Current CPSC best under this family is the hybrid by a small margin:

| Reference | CPSC PN2021-C AUROC / AUPRC | Gain AUROC / AUPRC |
|---|---:|---:|
| no-stabilizer teacherdrift | 0.8430 / 0.6012 | reference |
| matched stabilizer reference | 0.8641 / 0.6405 | +2.11 / +3.93 pp |
| raw1000 parity | 0.8743 / 0.6678 | +3.13 / +6.66 pp |
| source-cached target-raw1000 hybrid | 0.8750 / 0.6684 | +3.19 / +6.72 pp |

Decision: do not expand this hybrid to four centers as a source-preserving
solution. It is useful evidence that the 100 Hz target/corruption path is
helping CPSC robustness, but it does not recover source performance and barely
beats raw1000. The next ECGFounder branch should attack the residual
`emg_noise` and `baseline_wander` drops directly, and should include an explicit
source-floor constraint or stronger source loss/selection rule rather than
another input-chain-only variant.

### ECGFounder Source-BCE2 Hybrid CPSC Diagnostic

Follow-up intent:

- Keep the same `source_cached_target_raw1000` hybrid input chain.
- Increase source supervised BCE from `1.0` to `2.0`.
- Bias model selection toward the source term by lowering
  `target_val_score_weight` from `1.0` to `0.25`.

Training artifact:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_hybrid_sourcecached_targetraw1000_sourcebce2_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_hybrid_sourcecached_targetraw1000_sourcebce2_stabilizer35_ep10_seed20260531`

Formal PN2021-C artifact:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_hybrid_sourcecached_targetraw1000_sourcebce2_cpsc_pn2021c_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Matched CPSC comparison with the source-weighted follow-up:

| Variant | Source BCE weight | Target-val score weight | PTB-XL fold10 AUROC / AUPRC | CPSC clean AUROC / AUPRC | CPSC PN2021-C AUROC / AUPRC | CPSC drop AUROC / AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| matched-stabilizer reference | 1.0 | 1.0 | 0.9118 / 0.7844 | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp |
| source-cached target-raw1000 hybrid | 1.0 | 1.0 | 0.8839 / 0.7413 | 0.9064 / 0.7388 | 0.8750 / 0.6684 | 3.15 / 7.04 pp |
| source-cached target-raw1000 + source-BCE2 | 2.0 | 0.25 | 0.8856 / 0.7444 | 0.8849 / 0.6934 | 0.8599 / 0.6389 | 2.49 / 5.45 pp |

Per-operator source-BCE2 CPSC drops:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.8849 / 0.6934 | 0.00 / 0.00 pp |
| emg_noise | 0.8173 / 0.5608 | 6.76 / 13.26 pp |
| baseline_wander | 0.8552 / 0.6115 | 2.97 / 8.19 pp |
| baseline_shift | 0.8850 / 0.6929 | -0.01 / 0.05 pp |
| random_leads_masking | 0.8575 / 0.6357 | 2.74 / 5.77 pp |

The source-weighted follow-up is negative. It recovers only
`+0.17 pp AUROC / +0.31 pp AUPRC` on PTB-XL fold10 relative to the previous
hybrid, while still remaining far below the cached5000 matched-stabilizer
source floor. More importantly, CPSC PN2021-C falls by
`-1.50 pp AUROC / -2.95 pp AUPRC` versus the previous hybrid. The smaller
drop is therefore not a robustness improvement; it is mostly caused by the
lower clean CPSC baseline.

### ECGFounder All-op EfficientNet-teacher + Stabilizer CPSC Diagnostic

Follow-up intent:

- Keep the current best ECGFounder stabilizer/noise-mask primary branch:
  `powerline_noise`, `emg_noise`, and `random_leads_masking` with hard BCE and
  JSD.
- Add an auxiliary EfficientNet1DV2 robust-teacher branch over all five
  calibrated operators, including the previously missing `emg_noise`.
- Keep the teacher branch soft-only with weight `0.5`, so the run tests whether
  the EfficientNet robust model can transfer operator coverage into ECGFounder.

Training artifact:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetallopw05_stabilizer35_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_effnetallopw05_stabilizer35_ep10_seed20260531`

Formal PN2021-C artifact:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetallopw05_stabilizer35_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Training selection:

- Best epoch: `8`.
- PTB-XL fold10: `0.9109 / 0.7828`.
- CPSC clean held-out from training eval: `0.8936 / 0.6877`.
- Formal PN2021-C clean, recomputed with stabilizer: `0.8952 / 0.6953`.

Matched CPSC comparison:

| Variant | PTB-XL fold10 AUROC / AUPRC | CPSC clean AUROC / AUPRC | CPSC PN2021-C AUROC / AUPRC | CPSC drop AUROC / AUPRC |
|---|---:|---:|---:|---:|
| matched-stabilizer reference | 0.9118 / 0.7844 | 0.8944 / 0.6992 | 0.8641 / 0.6405 | 3.03 / 5.88 pp |
| source-cached target-raw1000 hybrid | 0.8839 / 0.7413 | 0.9064 / 0.7388 | 0.8750 / 0.6684 | 3.15 / 7.04 pp |
| noise/mask hard + all-op EfficientNet-teacher w=0.5 + stabilizer | 0.9109 / 0.7828 | 0.8952 / 0.6953 | 0.8667 / 0.6369 | 2.84 / 5.84 pp |

Per-operator CPSC result:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.8952 / 0.6953 | 0.00 / -0.00 pp |
| emg_noise | 0.8178 / 0.5596 | 7.74 / 13.57 pp |
| baseline_wander | 0.8632 / 0.6038 | 3.20 / 9.15 pp |
| baseline_shift | 0.8949 / 0.6948 | 0.03 / 0.06 pp |
| random_leads_masking | 0.8626 / 0.6311 | 3.25 / 6.42 pp |

Decision: do not expand this all-op EfficientNet-teacher branch to four
centers. Compared with the matched-stabilizer reference, it changes CPSC
corrupted mean by only `+0.26 pp AUROC / -0.36 pp AUPRC`. Compared with the
current CPSC hybrid best, it is worse by `-0.82 pp AUROC / -3.15 pp AUPRC`.
It preserves the PTB-XL source floor much better than raw1000/hybrid variants,
but it does not transfer the EfficientNet robust-teacher advantage into
ECGFounder's residual `emg_noise` and `baseline_wander` operators.

### ECGFounder Existing-pool Operator Oracle Diagnostic

This diagnostic asks whether the remaining gap is mainly a selection/routing
problem among the checkpoints already trained. It does not train a new model.
It scans all available CPSC ECGFounder calibrated-10to20pp JSONs and also runs
a four-center oracle over the candidates available per center. For each
corruption operator, it picks the candidate with the best AUPRC.

Artifacts:

- CPSC candidate summary:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/cpsc_2018/candidate_summary.csv`
- CPSC operator oracle:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/cpsc_2018/per_operator_oracle.csv`
- Four-center oracle summary:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/fourcenter/oracle_summary.json`
- Four-center operator oracle:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/fourcenter/per_center_operator_oracle.csv`

CPSC result over 24 existing candidates:

| Candidate view | PN2021-C AUROC / AUPRC | Delta vs CPSC hybrid AUROC / AUPRC |
|---|---:|---:|
| best single candidate: source-cached target-raw1000 hybrid | 0.8750 / 0.6684 | reference |
| per-operator oracle over existing pool | 0.8774 / 0.6694 | +0.25 / +0.10 pp |

CPSC per-operator oracle selections:

| Operator | Best existing candidate | AUROC / AUPRC |
|---|---|---:|
| powerline_noise | source-cached target-raw1000 hybrid | 0.9064 / 0.7388 |
| emg_noise | source-cached target-raw1000 hybrid | 0.8290 / 0.5801 |
| baseline_wander | original VAE-LH noAug | 0.8756 / 0.6265 |
| baseline_shift | source-cached target-raw1000 hybrid | 0.9060 / 0.7385 |
| random_leads_masking | raw1000 parity | 0.8701 / 0.6632 |

Four-center oracle over currently available candidates:

| Candidate view | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs original VAE+AugMix AUROC / AUPRC |
|---|---:|---:|---:|---:|
| original ECGFounder VAE+AugMix | 0.9037 / 0.6455 | 0.8169 / 0.5132 | 8.68 / 13.23 pp | reference |
| current stabilizer reporting candidate | 0.8896 / 0.6289 | 0.8641 / 0.5807 | 2.54 / 4.82 pp | +4.72 / +6.75 pp |
| per-center/per-operator oracle over current pool | 0.8978 / 0.6456 | 0.8739 / 0.5973 | 2.39 / 4.83 pp | +5.70 / +8.41 pp |

The oracle is informative but not a solution. On CPSC, the existing pool has
only `+0.25 / +0.10 pp` hidden headroom over the source-cached target-raw1000
hybrid, so CPSC is not limited by model selection. Four-center oracle improves
the reporting candidate by only `+0.98 / +1.67 pp`, and still reaches only
`+5.70 / +8.41 pp` over original ECGFounder VAE+AugMix. This is close on AUPRC
but still short of a stable `+10 pp` claim.

The oracle also identifies the next useful training test: CPSC's raw1000/hybrid
mechanism is not available for Chapman-Shaoxing, Georgia, or Ningbo. If the
CPSC `+2-3 pp` raw1000 AUPRC gain over the matched stabilizer generalizes to
the other centers, the four-center ECGFounder result could move close to
`+10 pp` AUPRC over original VAE+AugMix. The lowest-cost next run is therefore
Chapman-Shaoxing raw1000 parity, because Chapman has the weakest current
stabilized AUPRC (`0.4979`) and the largest room to recover target clean/corrupt
performance.

Prepared next-run inputs for Chapman raw1000 parity:

- Ref metadata:
  `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/chapman_shaoxing/k500_seed20260531/chapman_shaoxing_real_k500_seed20260531.ref_meta.json`
- Direct K500 init head:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_direct_k500_v7_sjr_rgq/mainline_v7_k500_4gpu_20260605/chapman_shaoxing/runs/chapman_shaoxing_K500_fromK500_headft_ep50_seed20260531/best_head.pt`
- FullFT VAE aw20 teacher for drift/shift distillation:
  `/home/linbinhao/ECG_adv_data/paper_ecgfounder_fullft_vae_inithead_aw20_lowlr_threads8_seed20260531_v6_20260525/runs/chapman_shaoxing_K500_fullft_ep10_lr2e-05_sw1p0_tw40p0_fullft_vae_inithead_aw20p0_ka160_M20_lam0p05_hs5_hlr0p25_labelexact_includeanchor_ihvaw20_tv100_source_plus_target_val_auprc_tvseed20260531_seed20260531/best_model.pt`
- Shared source raw1000 caches:
  `/home/linbinhao/ECG_adv_data/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy`
  and
  `/home/linbinhao/ECG_adv_data/triple_labels/super5_minresample_full10_perglobal_20260503/ptbxl_labels.C5.all.npy`.

Chapman raw1000 parity has now been run and formally evaluated:

- Training artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_chapman_20260616/runs/chapman_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_raw1000_stabilizer35_ep10_seed20260531`
- Formal PN2021-C artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_chapman_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Chapman raw1000 parity result:

| Variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs stabilizer AUROC / AUPRC |
|---|---:|---:|---:|---:|
| matched stabilizer reference | 0.8901 / 0.5408 | 0.8714 / 0.4979 | 1.87 / 4.28 pp | reference |
| raw1000 parity | 0.8907 / 0.5469 | 0.8721 / 0.4950 | 1.86 / 5.19 pp | +0.07 / -0.29 pp |

Per-operator raw1000 deltas versus the stabilizer reference:

| Operator | Raw1000 AUROC / AUPRC | Delta AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.8907 / 0.5469 | +0.06 / +0.62 pp |
| emg_noise | 0.8807 / 0.4819 | +1.84 / +2.82 pp |
| baseline_wander | 0.8454 / 0.4319 | -0.85 / -3.00 pp |
| baseline_shift | 0.8884 / 0.5393 | -0.03 / +0.50 pp |
| random_leads_masking | 0.8553 / 0.4750 | -0.66 / -2.40 pp |

Interpretation: raw1000 parity does not reproduce the CPSC mean gain on
Chapman. It improves the hard `emg_noise` operator, but gives back the gain on
`baseline_wander` and `random_leads_masking`; the five-op mean AUPRC is
`0.29 pp` below the matched stabilizer reference. This weakens the hypothesis
that simply expanding raw1000 parity to the remaining centers will yield a
four-center `+10 pp` ECGFounder improvement.

Chapman direct feature-head teacher has also now been run and formally
evaluated. This diagnostic keeps the main VAE-LH online AT + noise/mask
raw-AugMix recipe and the `0.5-35 Hz` ECGFounder input stabilizer, but replaces
the drift/shift auxiliary teacher with the frozen direct K500 ECGFounder
feature-head model.

- Training artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_chapman_20260616/runs/chapman_fullft_vae_aw20_rawaugmix_noisemask_directheadteacher_stabilizer35_ep10_seed20260531`
- Formal PN2021-C artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_chapman_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- Refreshed candidate-oracle artifact including this run:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_candidate_oracle_diagnostics_20260616_directheadteacher_refresh/oracle_summary.json`

Chapman direct-head teacher result:

| Variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs stabilizer AUROC / AUPRC | Delta vs original VAE+AugMix AUROC / AUPRC |
|---|---:|---:|---:|---:|---:|
| matched stabilizer reference | 0.8901 / 0.5408 | 0.8714 / 0.4979 | 1.87 / 4.28 pp | reference | +3.45 / +5.23 pp |
| raw1000 parity | 0.8907 / 0.5469 | 0.8721 / 0.4950 | 1.86 / 5.19 pp | +0.07 / -0.29 pp | +3.52 / +4.94 pp |
| direct feature-head teacher | 0.9154 / 0.5573 | 0.8866 / 0.5110 | 2.87 / 4.63 pp | +1.52 / +1.31 pp | +4.98 / +6.54 pp |

Per-operator direct-head teacher deltas versus the stabilizer reference:

| Operator | Direct-head AUROC / AUPRC | Delta AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.9154 / 0.5573 | +2.53 / +1.65 pp |
| emg_noise | 0.8520 / 0.4569 | -1.03 / +0.32 pp |
| baseline_wander | 0.8759 / 0.4861 | +2.20 / +2.42 pp |
| baseline_shift | 0.9142 / 0.5544 | +2.55 / +2.01 pp |
| random_leads_masking | 0.8757 / 0.5004 | +1.37 / +0.13 pp |

Interpretation: direct feature-head teacher is better than both the matched
stabilizer reference and raw1000 parity on Chapman five-op mean, and it
recovers `+4.98 / +6.54 pp` over original ECGFounder VAE+AugMix. However, it
still does not reach a `+10 pp` single-center gain and it mainly opens a
Chapman-specific tradeoff: powerline, baseline-wander, baseline-shift, and mask
improve, while EMG AUROC falls behind raw1000. The refreshed four-center
operator oracle barely changes, from roughly `0.8738 / 0.5976` to
`0.8757 / 0.5981`. The hidden oracle headroom added by this run is therefore
only about `+0.19 / +0.06 pp`, so this is not the missing mechanism for a
stable four-center `+10 pp` ECGFounder improvement.

Ningbo direct feature-head teacher was then run with the same recipe to test
whether the Chapman gain generalizes to another bottleneck center.

- Training artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_ningbo_20260616/runs/ningbo_fullft_vae_aw20_rawaugmix_noisemask_directheadteacher_stabilizer35_ep10_seed20260531`
- Formal PN2021-C artifact:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_ningbo_pn2021c_20260616/ningbo/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- Refreshed candidate-oracle artifact including both Chapman and Ningbo
  direct-head runs:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_candidate_oracle_diagnostics_20260616_directheadteacher_chapman_ningbo_refresh/oracle_summary.json`

Ningbo direct-head teacher result:

| Variant | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop AUROC / AUPRC | Delta vs stabilizer AUROC / AUPRC | Delta vs original VAE+AugMix AUROC / AUPRC |
|---|---:|---:|---:|---:|---:|
| matched stabilizer reference | 0.8955 / 0.5698 | 0.8728 / 0.5272 | 2.27 / 4.27 pp | reference | +4.99 / +6.91 pp |
| direct feature-head teacher | 0.9077 / 0.5691 | 0.8841 / 0.5337 | 2.36 / 3.54 pp | +1.14 / +0.65 pp | +6.12 / +7.56 pp |

Per-operator direct-head teacher deltas versus the Ningbo stabilizer reference:

| Operator | Direct-head AUROC / AUPRC | Delta AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.9077 / 0.5691 | +1.22 / -0.07 pp |
| emg_noise | 0.8674 / 0.5087 | +0.92 / +1.78 pp |
| baseline_wander | 0.8626 / 0.4981 | +0.75 / +0.47 pp |
| baseline_shift | 0.9067 / 0.5699 | +1.30 / +0.33 pp |
| random_leads_masking | 0.8763 / 0.5226 | +1.50 / +0.76 pp |

Interpretation: the Ningbo direct-head teacher is a real improvement over the
matched stabilizer reference and recovers `+6.12 / +7.56 pp` over original
ECGFounder VAE+AugMix. It is still not a `+10 pp` solution. In the four-center
candidate pool, adding Ningbo changes the operator oracle from `0.8757 /
0.5981` to `0.8771 / 0.5993`, only `+0.13 / +0.12 pp`. The current
center-level best-candidate mix reaches `0.8735 / 0.5925`, or `+5.66 /
+7.93 pp` over original ECGFounder VAE+AugMix. The stricter per-operator oracle
reaches `0.8771 / 0.5993`, or `+6.01 / +8.61 pp`. Both views remain below a
stable four-center `+10 pp` claim.

The progress audit also clarifies the remaining bottlenecks. For
EfficientNet1DV2, the robust-selector boundary raw-AugMix recipe gains only
about `+3.12 pp` AUPRC on `random_leads_masking` and `+4.62 pp` on
`emg_noise`, while `baseline_wander` already gains about `+9.94 pp`; another
global raw-AugMix strength increase is therefore poorly targeted. For
ECGFounder, the per-operator oracle already recovers large AUPRC on
`powerline_noise` (`+22.32 pp`), `emg_noise` (`+11.37 pp`), and masking
(`+8.09 pp`), but gains almost nothing on `baseline_wander` (`+0.21 pp`) or
`baseline_shift` (`+1.05 pp`) because the original direct/VAE+AugMix heads are
already the best candidates there. The ECGFounder gap is now mostly a
Chapman/Ningbo clean-vs-robust tradeoff, not missing corruption amplitude.

## ECGFounder Model-blend Upper-bound Diagnostic

Output:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_ensemble_diagnostics_20260616/cpsc_2018/ecgfounder_fullft_vae_noisemask_teacher_blend_sweep.json`

This diagnostic does not train a new model. It reloads the existing CPSC
fullFT checkpoints, runs the same ref-excluded PN2021-C CPSC samples through
each model, and sweeps probability/logit blending weights. It is meant to
answer whether the current ECGFounder gap is just a single-head averaging
problem.

Models in the sweep:

- `fullft_vae`: fullFT VAE init-head aw20.
- `noisemask`: fullFT VAE + noise/mask-only raw-AugMix.
- `teacherdrift`: fullFT VAE + noise/mask hard + drift/shift teacher distill.

Best global five-operator blends:

| Blend | Mode | Weight for second model | Mean PN2021-C AUROC / AUPRC | Streaming-clean AUROC / AUPRC |
|---|---|---:|---:|---:|
| fullFT VAE + noise/mask | probability | 0.9 | 0.844 / 0.599 | 0.898 / 0.718 |
| fullFT VAE + noise/mask | logit | 0.8 | 0.844 / 0.600 | 0.899 / 0.720 |
| fullFT VAE + teacher-distill | probability | 0.9 | 0.844 / 0.602 | 0.899 / 0.721 |
| fullFT VAE + teacher-distill | logit | 0.8 | 0.845 / 0.603 | 0.900 / 0.722 |
| noise/mask + teacher-distill | probability | 1.0 | 0.843 / 0.601 | 0.898 / 0.720 |
| noise/mask + teacher-distill | logit | 1.0 | 0.843 / 0.601 | 0.898 / 0.720 |

The best global blend is `fullft_vae + teacherdrift` with logit blending and
weight `0.8` on the teacher-distill model: `0.8447 / 0.6028`. That is only
about `+0.17 / +0.16 pp` above the teacher-distill single model
(`0.8430 / 0.6012`) on the five-operator mean.

Per-operator AUPRC-optimal weights for the best blend show the same conflict:

| Operator | Best weight for teacher-distill | AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.8 | 0.841 / 0.600 |
| emg_noise | 1.0 | 0.806 / 0.557 |
| baseline_wander | 0.0 | 0.853 / 0.599 |
| baseline_shift | 0.0 | 0.890 / 0.692 |
| random_leads_masking | 0.9 | 0.843 / 0.588 |

The blend diagnostic confirms that the current pool has little hidden
headroom. Noise/mask robustness and drift/shift preservation sit in different
models, but simple probability/logit interpolation cannot combine them into a
near-`10 pp` ECGFounder recovery. The next useful ECGFounder experiment needs
new supervision or capacity outside the current checkpoint pool, not post-hoc
averaging of the current checkpoints.

## ECGFounder Explicit Operator-conditioned Adapter Diagnostic

Training run:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_opadapter_teacherdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_opadapter_teacherdrift_ep10_seed20260531`

PN2021-C eval:

`/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_opadapter_teacherdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`

Code change:

- Raw AugMix builders now record `view_ops` for each generated view. For the
  boundary recipe (`width=1`, `depth=1`, fixed `m=1.0`), each training view has
  one explicit operator id.
- ECGFounder fullFT can optionally train a zero-initialized
  `OperatorConditionedLogitAdapter`: one residual feature-to-logit adapter per
  PN2021-C operator, added only for corrupted views.
- The ECGFounder PN2021-C evaluator can reload that adapter checkpoint and,
  as a diagnostic, pass the current corruption name to `forward_with_operator`.
  This is an oracle-style diagnostic because the evaluator knows the active
  corruption operator.

Matched training setup:

- Base recipe kept the best CPSC ECGFounder teacher-distill setup:
  fullFT VAE init-head aw20, VAE-LH online AT, primary
  `powerline_noise` / `emg_noise` / `random_leads_masking` raw-AugMix with
  hard BCE + JSD, and auxiliary `baseline_wander` / `baseline_shift`
  fullFT-teacher distillation.
- Added `--raw_corrupt_op_conditioning explicit_adapter` with
  `hidden=128`, `dropout=0.0`, and `scale=1.0`.
- Best checkpoint selected epoch 8 by `source_plus_target_val_auprc`.
  The final epoch reached clean CPSC `0.904 / 0.728`, but the saved
  paper-style best checkpoint is epoch 8 with clean CPSC `0.901 / 0.720`.

Mean calibrated PN2021-C CPSC:

| Method | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Delta vs teacher-distill |
|---|---:|---:|---:|
| fullFT VAE + noise/mask hard + drift/shift teacher distill | 0.904 / 0.731 | 0.843 / 0.601 | reference |
| + explicit operator-conditioned adapter | 0.901 / 0.720 | 0.842 / 0.596 | -0.14 / -0.53 pp |

Per-operator PN2021-C:

| Operator | AUROC / AUPRC | Drop AUROC / AUPRC |
|---|---:|---:|
| powerline_noise | 0.839 / 0.593 | 6.19 / 12.70 pp |
| emg_noise | 0.806 / 0.555 | 9.53 / 16.47 pp |
| baseline_wander | 0.846 / 0.589 | 5.51 / 13.11 pp |
| baseline_shift | 0.878 / 0.660 | 2.30 / 5.95 pp |
| random_leads_masking | 0.839 / 0.582 | 6.20 / 13.75 pp |

Training diagnostics confirm that the adapter branch was active, not silently
ignored: epoch 8 recorded `16384` op-conditioned primary views and `8192`
op-conditioned auxiliary views. The result is still slightly below the
teacher-distill baseline. Therefore the missing ECGFounder mechanism is not
just "know which operator this is and add a small per-operator residual head".
Either the adapter capacity/supervision is too weak, or the needed robust
teacher signal must be stronger than the current ECGFounder fullFT teacher.

## Evidence Paths

- ECGFounder last-epoch training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_vae_lhat_calibrated_rawsupervised_last_epoch_cpsc_20260616`
- ECGFounder last-epoch PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_pn2021c_calibrated_10to20pp_20260616/calrawsupervised_last_epoch/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder raw-AugMix width=1 depth=1 fixed m=1.0 last-epoch training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_vae_lhat_fullpool_raw_augmix_w1_m100_last_epoch_k500_v7_sjr_rgq_cpsc_2018/20260616_ecgfounder_rawaugmix_w1_m100_last_cpsc_gpu0/cpsc_2018/runs/cpsc_2018_K500_M20_lam0p15_ep20_seed20260531`
- ECGFounder raw-AugMix width=1 depth=1 fixed m=1.0 PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_pn2021c_calibrated_10to20pp_20260616/rawaugmix_w1_m100_last_epoch/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT direct init-head PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_pn2021c_calibrated_10to20pp_20260616/direct_inithead/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE init-head aw20 PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_pn2021c_calibrated_10to20pp_20260616/vae_inithead_aw20/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE raw-AugMix training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_w1_m100_ep10_seed20260531`
- ECGFounder fullFT VAE raw-AugMix PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE low-weight five-op raw-AugMix training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_lowweight_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_w1_m100_lowraw_bce025_jsd050_ep10_seed20260531`
- ECGFounder fullFT VAE low-weight five-op raw-AugMix PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_lowweight_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE noise/mask-only raw-AugMix training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_noisemask_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_w1_m100_ep10_seed20260531`
- ECGFounder fullFT VAE noise/mask-only raw-AugMix PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_noisemask_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE drift/shift-only raw-AugMix matched training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_driftshift_matched_w1_m100_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_driftshift_matched_w1_m100_ep10_seed20260531`
- ECGFounder fullFT VAE drift/shift-only raw-AugMix matched PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_driftshift_matched_w1_m100_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE two-branch raw-AugMix training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_twobranch_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_twobranch_noisehard_driftsoft_ep10_seed20260531`
- ECGFounder fullFT VAE two-branch raw-AugMix PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_twobranch_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE noise/mask + drift/shift low hard-BCE training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_auxdriftbce025_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_auxdriftbce025_ep10_seed20260531`
- ECGFounder fullFT VAE noise/mask + drift/shift low hard-BCE PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_auxdriftbce025_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE noise/mask + drift/shift teacher-distill training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_ep10_seed20260531`
- ECGFounder fullFT VAE noise/mask + drift/shift teacher-distill PN2021-C
  eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE noise/mask + drift/shift EfficientNet-teacher
  distill training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetteacher_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_effnetteacher_ep10_seed20260531`
- ECGFounder fullFT VAE noise/mask + drift/shift EfficientNet-teacher
  distill PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetteacher_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE noise/mask + drift/shift feature-consistency training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_featdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_auxfeatnorm100_ep10_seed20260531`
- ECGFounder fullFT VAE noise/mask + drift/shift feature-consistency PN2021-C
  eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_featdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder fullFT VAE all-op feature-consistency training:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_allop_featnorm300_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_allop_featnorm300_ep10_seed20260531`
- ECGFounder fullFT VAE all-op feature-consistency PN2021-C eval:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_allop_featnorm300_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`
- ECGFounder CPSC model-blend upper-bound diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_ensemble_diagnostics_20260616/cpsc_2018/ecgfounder_fullft_vae_noisemask_teacher_blend_sweep.json`
- ECGFounder CPSC input-filter cutoff diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_fft_filter_diag.json`
- ECGFounder CPSC bandpass35 + lead-repair diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_bandpass35_leadrepair_diag.json`
- EfficientNet calibrated latent AugMix eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018/20260615_callatentaugmix_cpsc_gpu4`
- EfficientNet calibrated latent AugMix direct-loss eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_calibrated_latent_augmix_directloss_k500_v7_sjr_rgq_cpsc_2018/20260615_callatentaugmix_directloss_cpsc_gpu4`
- EfficientNet calibrated raw-supervised eval:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_effnet_v7_strong_calibrated_rawsupervised/pn2021c_calrawsupervised_fourcenter_20260614_gpu0`
- EfficientNet full-pool raw-AugMix eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_k500_v7_sjr_rgq_cpsc_2018/smoke_execute_check_fullpool_rawaugmix_20260616`
- EfficientNet full-pool raw-AugMix depth=1 eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_cpsc_gpu0`
- EfficientNet full-pool raw-AugMix depth=1 no-clip eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_noclip_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_noclip_cpsc_gpu0`
- EfficientNet full-pool raw-AugMix depth=1 fixed m=0.75 eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_m075_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_m075_cpsc_gpu0`
- EfficientNet full-pool raw-AugMix width=1 depth=1 fixed m=1.0 eval:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_w1_m100_cpsc_gpu0`
- EfficientNet full-pool raw-AugMix width=1 depth=1 fixed m=1.0 remaining
  three-center run:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0`
- EfficientNet CPSC K500 internal robust selector:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_w1_m100_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/k500_internal_robust_selector.json`
- EfficientNet Ningbo K500 internal robust selector:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/ningbo_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/k500_internal_robust_selector.json`
- EfficientNet Chapman K500 internal robust selector:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/k500_internal_robust_selector.json`
- EfficientNet Georgia K500 internal robust selector:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/georgia_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/k500_internal_robust_selector.json`
- EfficientNet CPSC epoch-30 latest checkpoint diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_cpsc_2018/20260616_fullpool_rawaugmix_depth1_w1_m100_cpsc_gpu0/cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json`
- EfficientNet Ningbo epoch-30 latest checkpoint diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/ningbo_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json`
- EfficientNet Chapman epoch-30 latest checkpoint diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json`
- EfficientNet Georgia epoch-30 latest checkpoint diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3/20260616_fullpool_rawaugmix_w1_m100_remaining3_gpu0/georgia_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json`

## Interpretation

The current calibrated latent-AugMix route does not explain the raw-supervised
gain. The operator strength is present in the training chain, but the training
coverage is much smaller: latent AugMix generated about 300 views per epoch
(600 for the direct-loss variant), while raw-supervised used a source+target
raw pool of 17,816 samples with 64 corruption-consistency batches per epoch.

The full-pool raw-AugMix follow-ups close the EfficientNet CPSC gap, but the
four-center selected-checkpoint check initially looked incomplete. Matching
raw-supervised sample coverage helped, forcing AugMix branch `depth=1` helped
again, and increasing the mixture coefficient helped. Removing training-time
clipping did not help. The final `width=1`, `depth=1`, fixed `m=1.0` boundary
recipe plus K500 internal robust checkpoint selection now matches calibrated
raw-supervised on the four-center corrupted mean. Adding the post-hoc
input-stabilizer diagnostic on top of that selector raises the four-center
corrupted mean further to `0.8362 / 0.5065`, or `+8.86 / +9.92 pp` over
VAE-LH noAug.

The main EfficientNet bottleneck was therefore not operator amplitude. The
original AugMix mixing graph diluted the PN2021-C stressor, and once that was
fixed, checkpoint selection became the decisive issue. The paper-safe K500
internal robust selector chooses the epoch-30 checkpoint for all four centers,
with the material gain concentrated in Chapman. Applying that selector closes
the raw-supervised gap without using held-out PN2021 labels for selection.
A follow-up that duplicated the weak Chapman operators (`emg_noise` and
`baseline_wander`) in the raw-AugMix op sampler improved K500 internal
corrupted validation but externally changed Chapman PN2021-C by only
`+0.10 / +0.14 pp`, so operator reweighting is not the remaining EfficientNet
lever. The remaining EfficientNet lever appears to be input-side
stabilization: bandpass35 + flat-lead repair improves all four centers by
`+2.45 / +3.43 pp` over the robust selector, but it is currently only an
inference-time diagnostic and still needs to be folded into the
training/evaluation recipe cleanly.

For ECGFounder, forcing last-epoch selection recovered only +1.20 pp corrupted
AUPRC over the epoch-0 selected checkpoint. This rules out checkpoint selection
as the main reason the ECGFounder raw-supervised result is still below the
desired 10 pp recovery. Porting the EfficientNet boundary raw-AugMix recipe to
ECGFounder also did not unlock the gap: the result was only `+0.30 / +1.28 pp`
corrupted over direct and essentially tied with ECGFounder calibrated
raw-supervised last epoch. FullFT direct/VAE improves clean CPSC but still
recovers only `+0.82 / +1.92 pp` and `+1.14 / +2.90 pp` corrupted over frozen
direct. Adding calibrated raw-AugMix to fullFT VAE finally moves ECGFounder
to `+3.23 / +5.74 pp` corrupted over frozen direct. Restricting that branch
to noise/masking improves it further to `+3.88 / +6.61 pp`, but that is still
short of the desired `~10 pp`. A matched drift/shift-only branch is worse
than fullFT VAE on the five-op corrupted mean (`0.810 / 0.552`) and does not
solve `baseline_shift`. A diagnostic operator-aware oracle over the existing
ECGFounder candidates only reaches about `0.846 / 0.606`, improving the best
single noise/mask-only candidate by less than `1 pp`. Lowering the shared
five-op raw branch to `raw_bce/jsd = 0.25 / 0.5` preserves clean CPSC
(`0.907 / 0.740`) but still underperforms noise/mask-only on the corrupted
mean (`0.835 / 0.593` vs `0.843 / 0.598`). The bottleneck is therefore
ECGFounder-specific and not solved by AugMix chain strength, last-epoch
selection, full encoder fine-tuning, VAE fullFT alone, a single mixed
raw-AugMix objective, a symmetric high-weight operator split, post-hoc
selection among the current candidate pool, lowering the shared five-op raw
branch weight, a true two-branch hard/soft operator split, distilling
drift/shift views from the frozen fullFT VAE teacher, or adding normalized
clean/corrupt feature consistency on the drift/shift branch, or using a
low-weight hard-label drift/shift auxiliary branch, adding an explicit
per-operator residual adapter to the teacher-distill recipe, or replacing the
ECGFounder fullFT teacher with a robust EfficientNet boundary raw-AugMix
teacher, or making all five operators share a high-weight normalized
feature-consistency objective. The fullFT-teacher distill variant is still the
best ECGFounder AUPRC among these diagnostics (`0.601`), but its gain over
noise/mask-only is only `+0.02 / +0.32 pp`. The drift/shift
feature-consistency variant lands at `0.842 / 0.598`, and the low hard-BCE
variant lands at `0.841 / 0.597`, both below noise/mask-only and
teacher-distill. The all-op feature-consistency variant improves clean CPSC to
`0.907 / 0.740` but drops corrupted mean to `0.837 / 0.595`, also below
noise/mask-only and teacher-distill. The explicit operator-conditioned adapter
lands at `0.842 / 0.596`, and the EfficientNet-teacher distill variant lands
at `0.843 / 0.595`, both below fullFT-teacher distill. A model-blend diagnostic over
fullFT VAE, noise/mask, and teacher-distill finds only `0.8447 / 0.6028`
at the best global logit
blend, so the current ECGFounder pool has only about `+0.17 / +0.16 pp`
hidden mean-corrupted headroom beyond teacher-distill. The first materially
different positive diagnostic is deterministic input stabilization: applying
`0.5-35 Hz` bandpass before ECGFounder upsampling plus flat-lead repair for
masking raises the teacher-distill candidate to `0.8639 / 0.6507` corrupted
mean, a `+2.09 / +4.95 pp` gain over teacher-distill without the stabilizer.
The formal evaluator rerun records the same result at `0.8639 / 0.6508` with
clean recomputed under the same stabilizer.

## Next Step

The next useful experiment is no longer another AugMix amplitude sweep.
For EfficientNet1DV2, the path is to promote the K500 internal robust selector
from diagnostic helper into the managed model-selection/finalization flow and
then rerun or finalize the selected checkpoint artifacts under that rule.

For ECGFounder, stop treating this as an AugMix-strength problem. The useful
next ECGFounder axis is a frequency-stabilized input path, not a stronger mixed
chain or selection among the existing heads. The fullFT raw-AugMix result shows
the model can learn invariance to `powerline_noise` and `emg_noise`, but the
same objective damages `baseline_wander` and `baseline_shift`. The noise/mask
and teacher-distill checks partly reduce the conflict but still leave a large
gap. The matched drift/shift-only, low-weight five-op, two-branch,
feature-consistency, operator-adapter, robust-teacher, and model-blend
diagnostics all fail to create target-scale headroom. The input-stabilizer
diagnostic finally changes that: bandpass `0.5-35 Hz` eliminates most
powerline and baseline-shift damage, and flat-lead repair turns
`random_leads_masking` from `0.843 / 0.589` to `0.861 / 0.659` without
changing clean metrics.

The next run should implement this stabilizer as a shared train/eval
ECGFounder front-end option and train the current best fullFT VAE +
noise/mask hard + drift/shift teacher-distill recipe through that same front
end. That keeps the VAE+AugMix mainline intact but moves the missing mechanism
from "stronger raw loss" to "make ECGFounder see a stable frequency/lead
representation before the pretrained encoder".

That run has now been completed for CPSC. It confirms the stabilizer mechanism
but not the need to train through it: matched stabilizer training lands at
`0.8641 / 0.6405`, versus `0.8639 / 0.6508` for applying the same stabilizer
post-hoc to the teacher-distill checkpoint. A later Chapman raw1000 parity
test also weakens the full raw-100Hz parity hypothesis: it improves
`emg_noise` AUPRC by `+2.82 pp` but loses `-3.00 pp` on `baseline_wander`
and `-2.40 pp` on `random_leads_masking`, so the five-op mean is
`0.29 pp` below the matched stabilizer reference. The next ECGFounder step
should therefore freeze the deterministic stabilizer as the reporting
front-end and solve only the remaining EMG/wander conflict with an
operator-specific front-end or loss, not with a global raw1000 conversion.

The follow-up on Chapman-Shaoxing, Georgia, and Ningbo gives the same
qualitative answer. The stabilizer reduces mean PN2021-C drops to
`1.87 / 4.28 pp` on Chapman, `3.00 / 4.86 pp` on Georgia, and
`2.27 / 4.27 pp` on Ningbo, with powerline and baseline-shift nearly
eliminated. Across the four-center reporting candidate, clean is
`0.8906 / 0.6329`, corrupted mean is `0.8641 / 0.5832`, and the remaining
drop is `2.65 / 4.97 pp`. It does not by itself guarantee a `+10 pp`
corrupted-mean gain because the clean model basis shifts, but it is the first
ECGFounder direction that consistently reduces the robustness drop to the low
single digits.

The EMG-only primary raw-AugMix follow-up also completed for CPSC and does not
change this recommendation. It lands at `0.8633 / 0.6452`; EMG AUPRC improves
only slightly (`0.569`), while mask/mean AUPRC fall below the post-hoc
stabilizer. Chapman raw1000 parity confirms the same tradeoff at another
center: EMG can be improved, but the improvement is paid for by
baseline-wander and masking. The remaining EMG problem likely needs a
different front-end or loss design, not simply assigning the hard raw branch
to EMG or converting the whole supervised stream to raw1000.

The EfficientNet result says the boundary raw-AugMix recipe can match
raw-supervised when the backbone is receptive to raw corruption training; the
ECGFounder result says one shared raw-AugMix objective, symmetric high-weight
split, post-hoc candidate selection, and shared low-weight full-pool training
are all too blunt, and neither the hard/soft two-branch split nor a fullFT VAE
teacher for drift/shift nor normalized feature consistency, even when applied
to all five operators with larger weight, changes that materially. The
input-stabilizer diagnostic is now the strongest ECGFounder direction and
should be promoted from diagnostic to a matched training/evaluation candidate.

## Verification

- Focused tests after adapter fix:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_adaptation_lhat.py util/tests/test_latent_augmix_consistency.py util/tests/test_effnet_vae_lhat_runner.py util/tests/test_config_loader.py::test_effnet_vae_lhat_fullpool_raw_augmix_config_exposes_flags -q`
- Result: `21 passed`.
- ECGFounder raw-AugMix focused tests:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_config_loader.py::test_ecgfounder_vae_lhat_diagnostic_last_epoch_config_is_explicitly_allowed util/tests/test_config_loader.py::test_ecgfounder_vae_lhat_raw_augmix_config_exposes_mixture_flags util/tests/test_ecgfounder_raw_augmix.py -q`
- Result: `4 passed`.
- Remaining three-center config tests:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_config_loader.py::test_effnet_vae_lhat_raw_augmix_w1_m100_remaining3_config_exposes_matrix util/tests/test_config_loader.py::test_effnet_vae_lhat_raw_augmix_config_exposes_mixture_flags -q`
- Result: `2 passed`.
- Robust selector tests after selector implementation and report update:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_effnet_robust_checkpoint_selector.py util/tests/test_config_loader.py::test_effnet_vae_lhat_raw_augmix_w1_m100_remaining3_config_exposes_matrix util/tests/test_config_loader.py::test_effnet_vae_lhat_raw_augmix_config_exposes_mixture_flags -q`
- Result: `4 passed`.
- ECGFounder fullFT PN2021-C evaluator test:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
- Result: `2 passed`.
- ECGFounder fullFT raw-AugMix and JSD finite-guard tests after the fullFT
  raw-AugMix run:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest methods/augmix/tests/test_jsd.py util/tests/test_ecgfounder_fullft_raw_augmix.py util/tests/test_ecgfounder_raw_augmix.py util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
- Result: `15 passed`.
- ECGFounder fullFT VAE low-weight five-op raw-AugMix training and PN2021-C
  eval completed on 2026-06-16. This was a parameter-only run using existing
  raw-AugMix code paths, so no additional unit tests were required.
- ECGFounder fullFT VAE two-branch raw-AugMix training and PN2021-C eval
  completed on 2026-06-16. The branch interface was covered by
  `util/tests/test_ecgfounder_fullft_raw_augmix.py` in the `15 passed` test
  command above.
- ECGFounder fullFT VAE noise/mask + drift/shift low hard-BCE training and
  PN2021-C eval completed on 2026-06-16. This used the existing auxiliary
  branch interface, so no new code path was added.
- ECGFounder fullFT VAE teacher-distill branch parser/training tests:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest methods/augmix/tests/test_jsd.py util/tests/test_ecgfounder_fullft_raw_augmix.py util/tests/test_ecgfounder_raw_augmix.py util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
- Result: `17 passed`.
- ECGFounder fullFT VAE noise/mask + drift/shift teacher-distill training and
  PN2021-C eval completed on 2026-06-16.
- ECGFounder fullFT VAE feature-consistency branch parser/training tests:
  `/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest methods/augmix/tests/test_jsd.py util/tests/test_ecgfounder_fullft_raw_augmix.py util/tests/test_ecgfounder_raw_augmix.py util/tests/test_ecgfounder_pn2021c_evaluator.py -q`
- Result: `19 passed`.
- ECGFounder fullFT VAE noise/mask + drift/shift feature-consistency training
  and PN2021-C eval completed on 2026-06-16.
- ECGFounder fullFT VAE all-op feature-consistency training and PN2021-C eval
  completed on 2026-06-16. This was a parameter-only run over existing
  all-op raw-AugMix and normalized feature-consistency code paths.
- ECGFounder CPSC input-stabilizer diagnostics completed on GPU 0 on
  2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_fft_filter_diag.json`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_filter_diagnostics_20260616/cpsc_2018/teacherdrift_bandpass35_leadrepair_diag.json`.
- ECGFounder CPSC input-stabilizer formal evaluator rerun completed on GPU 0 on
  2026-06-16 with `--recompute_clean_with_input_stabilizer`:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_input_stabilizer_official_eval_20260616/cpsc_2018/teacherdrift_bandpass35_repair_eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder matched input-stabilized fullFT VAE + raw-AugMix teacher-distill
  training and formal PN2021-C eval completed on GPU 0 on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder EMG-only primary raw-AugMix + stabilizer training and formal
  PN2021-C eval completed on GPU 0 on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_emgonly_stabilizer35_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_emgonly_teacherdrift_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_emgonly_stabilizer35_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder Chapman, Georgia, and Ningbo teacher-distill + post-hoc
  stabilizer formal PN2021-C eval completed on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`,
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/georgia/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`,
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_stabilizer35_fourcenter_pn2021c_20260616/ningbo/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- The Ningbo checkpoint was trained as a single low-load GPU job after the
  first parallel attempt was interrupted for shared-server CPU pressure:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_fourcenter_20260616/runs/ningbo_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_ep10_seed20260531`.
- EfficientNet1DV2 boundary raw-AugMix robust selector bundle generated from
  the original `k500_seed20260601` selector diagnostics on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/effnet_robust_selector_w1_m100_k500_v7_authoritative_20260616/selector_bundle.json`
  with compact outputs
  `/home/linbinhao/ECG_adv_data/runs/effnet_robust_selector_w1_m100_k500_v7_authoritative_20260616/selector_summary.csv`
  and
  `/home/linbinhao/ECG_adv_data/runs/effnet_robust_selector_w1_m100_k500_v7_authoritative_20260616/summary.md`.
- PN2021-C progress audit rebuilt the current EffNet rows from raw eval JSONs
  or the authoritative robust-selector bundle and corrected the ECGFounder
  four-center stabilizer reporting row:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_progress_audit_20260616/summary.md`,
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_progress_audit_20260616/progress_summary.csv`,
  and
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_progress_audit_20260616/effnet_operator_rows.csv`.
- EfficientNet1DV2 Chapman-only post-hoc input-stabilizer diagnostic completed
  and then the full four-center job was intentionally interrupted because
  flat-lead repair was too slow for a low-value complete sweep:
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/effnet_chapman_posthoc_stabilizer_partial.json`.
- EfficientNet1DV2 robust-selector post-hoc input-stabilizer four-center
  formal PN2021-C eval completed on GPU 0 on 2026-06-16 using the aligned
  `crop_len=1000` evaluator path:
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/chapman_bandpass35_maskrepair_renorm_crop1000.json`,
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/cpsc_2018_bandpass35_maskrepair_renorm_crop1000.json`,
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/georgia_bandpass35_maskrepair_renorm_crop1000.json`,
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/ningbo_bandpass35_maskrepair_renorm_crop1000.json`,
  and summary CSV
  `/home/linbinhao/ECG_adv_data/runs/effnet_posthoc_input_stabilizer_diagnostics_20260616/evaluator_full_20260616/effnet_selector_input_stabilizer_fourcenter_summary.csv`.
- EfficientNet1DV2 Chapman clean+EMG bandpass cutoff sweep completed on GPU 0
  on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/effnet_emg_bandpass_sweep_20260616/chapman_clean_emg_bandpass_sweep.json`.
- EfficientNet1DV2 Chapman hard-op weighted raw-AugMix pilot completed on GPU
  0 on 2026-06-16. K500 internal selector:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman/20260616_effnet_hardops_chapman_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_hardops_ep30_seed20260601/diagnostics/k500_internal_robust_selector_hardops.json`;
  clean single-center diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman/20260616_effnet_hardops_chapman_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_hardops_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c_hardops/eval_result_chapman_only_v7_exclrefs_crop1000.json`;
  PN2021-C diagnostic:
  `/home/linbinhao/ECG_adv_data/runs/effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman/20260616_effnet_hardops_chapman_gpu0/chapman_shaoxing_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_fullpool_rawaugmix_d1_w1_m100_hardops_ep30_seed20260601/diagnostics/epoch30_latest_model_for_pn2021c_hardops/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_hardops.json`.
- ECGFounder CPSC model-blend upper-bound diagnostic completed on GPU 0 on
  2026-06-16. It is a no-training inference diagnostic and wrote:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_ensemble_diagnostics_20260616/cpsc_2018/ecgfounder_fullft_vae_noisemask_teacher_blend_sweep.json`.
- ECGFounder explicit operator-conditioned adapter training and PN2021-C eval
  completed on GPU 0 on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_opadapter_teacherdrift_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_opadapter_teacherdrift_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_opadapter_teacherdrift_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder four-center input-stabilizer high-cut sweep completed on GPU 0 on
  2026-06-16 for `emg_noise` and `baseline_wander` only, using
  `25/30/35 Hz` high cuts and recomputed clean metrics:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_input_stabilizer_sweep_fourcenter_20260616/summary.csv`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_input_stabilizer_sweep_fourcenter_20260616/sweep_rows.csv`.
- ECGFounder existing-pool operator oracle diagnostic completed on 2026-06-16.
  It is a no-training CSV/JSON aggregation over already evaluated candidates:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/cpsc_2018/candidate_summary.csv`,
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/cpsc_2018/per_operator_oracle.csv`,
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/fourcenter/oracle_summary.json`,
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_operator_oracle_diagnostics_20260616/fourcenter/per_center_operator_oracle.csv`.
- ECGFounder Chapman raw1000 parity training and formal PN2021-C eval
  completed on GPU 0 on 2026-06-16 using existing raw1000/aux-teacher code
  paths:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_chapman_20260616/runs/chapman_fullft_vae_aw20_rawaugmix_noisemask_teacherdrift_raw1000_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_teacherdrift_raw1000_chapman_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder CPSC all-op EfficientNet-teacher w=0.5 + stabilizer training and
  formal PN2021-C eval completed on GPU 0 on 2026-06-16 using existing
  raw-AugMix aux-teacher and input-stabilizer paths:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetallopw05_stabilizer35_cpsc_20260616/runs/cpsc_fullft_vae_aw20_rawaugmix_noisemask_effnetallopw05_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_effnetallopw05_stabilizer35_pn2021c_calibrated_10to20pp_20260616/cpsc_2018/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder Chapman direct feature-head teacher + stabilizer training and
  formal PN2021-C eval completed on GPU 0 on 2026-06-16. The auxiliary
  drift/shift teacher used `ecgfounder_feature_head` with the direct K500
  `best_head.pt`:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_chapman_20260616/runs/chapman_fullft_vae_aw20_rawaugmix_noisemask_directheadteacher_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_chapman_pn2021c_20260616/chapman_shaoxing/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder candidate oracle was refreshed with the Chapman direct feature-head
  teacher result:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_candidate_oracle_diagnostics_20260616_directheadteacher_refresh/oracle_summary.json`.
- ECGFounder Ningbo direct feature-head teacher + stabilizer training and
  formal PN2021-C eval completed on GPU 0 on 2026-06-16:
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_ningbo_20260616/runs/ningbo_fullft_vae_aw20_rawaugmix_noisemask_directheadteacher_stabilizer35_ep10_seed20260531`
  and
  `/home/linbinhao/ECG_adv_data/runs/ecgfounder_fullft_vae_rawaugmix_directheadteacher_stabilizer35_ningbo_pn2021c_20260616/ningbo/eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp_s5.json`.
- ECGFounder candidate oracle was refreshed with both Chapman and Ningbo
  direct feature-head teacher results:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_candidate_oracle_diagnostics_20260616_directheadteacher_chapman_ningbo_refresh/oracle_summary.json`.
