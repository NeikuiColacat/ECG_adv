# ECGFounder Calibrated 10-20pp PN2021-C Eval - 2026-06-15

Scope: ECGFounder direct, VAE noAug, and VAE+AugMix on four v7 SJR/RGQ K500 target centers. Evaluation uses the `calibrated_10to20pp` profile and public severity 5 only, matching the EfficientNet1DV2 calibration profile.

Run root: `/home/linbinhao/ECG_adv_data/runs/ecgfounder_pn2021c_calibrated_10to20pp_20260615`

## Four-Center x Five-Operator Mean

| Variant | Clean AUROC/AUPRC | Corrupted AUROC/AUPRC | Drop AUROC/AUPRC | Delta corrupted vs direct | Delta drop vs direct |
|---|---:|---:|---:|---:|---:|
| direct | 0.897254/0.632539 | 0.813528/0.506445 | 8.37/12.61 pp | +0.00/+0.00 pp | +0.00/+0.00 pp |
| vae_noaug | 0.903117/0.646974 | 0.816879/0.513212 | 8.62/13.38 pp | +0.34/+0.68 pp | -0.25/-0.77 pp |
| vae_augmix | 0.903672/0.645512 | 0.816920/0.513218 | 8.68/13.23 pp | +0.34/+0.68 pp | -0.30/-0.62 pp |

VAE+AugMix vs VAE noAug:
- Corrupted metric delta: +0.00 pp AUROC / +0.00 pp AUPRC.
- Drop reduction: -0.05 pp AUROC / +0.15 pp AUPRC.

## Operator Drops

| Variant | Operator | Drop AUROC/AUPRC | Corrupted AUROC/AUPRC |
|---|---|---:|---:|
| direct | powerline_noise | 16.95/20.76 pp | 0.727765/0.424906 |
| direct | emg_noise | 13.00/19.71 pp | 0.767218/0.435403 |
| direct | baseline_wander | 3.31/6.67 pp | 0.864189/0.565887 |
| direct | baseline_shift | 0.47/0.97 pp | 0.892535/0.622830 |
| direct | random_leads_masking | 8.13/14.93 pp | 0.815934/0.483199 |
| vae_noaug | powerline_noise | 17.70/22.19 pp | 0.726144/0.425067 |
| vae_noaug | emg_noise | 13.64/21.24 pp | 0.766763/0.434547 |
| vae_noaug | baseline_wander | 3.33/7.09 pp | 0.869828/0.576028 |
| vae_noaug | baseline_shift | 0.32/0.85 pp | 0.899913/0.638446 |
| vae_noaug | random_leads_masking | 8.14/15.50 pp | 0.821747/0.491974 |
| vae_augmix | powerline_noise | 17.92/22.58 pp | 0.724467/0.419670 |
| vae_augmix | emg_noise | 14.09/21.74 pp | 0.762748/0.428114 |
| vae_augmix | baseline_wander | 3.37/7.03 pp | 0.869959/0.575177 |
| vae_augmix | baseline_shift | 0.15/0.24 pp | 0.902177/0.643158 |
| vae_augmix | random_leads_masking | 7.84/14.55 pp | 0.825248/0.499970 |

## Artifacts

- Summary CSV: `docs/reports/archive/20260615/ecgfounder_calibrated_10to20pp_pn2021c_summary_20260615.csv`
- Per-operator CSV: `docs/reports/archive/20260615/ecgfounder_calibrated_10to20pp_pn2021c_per_op_20260615.csv`
- Per-unit CSV: `docs/reports/archive/20260615/ecgfounder_calibrated_10to20pp_pn2021c_per_unit_20260615.csv`
