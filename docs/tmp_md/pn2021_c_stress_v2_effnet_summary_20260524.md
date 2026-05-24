# PN2021-C Stress-v2 EfficientNet1DV2 Summary

Date: 2026-05-24

## Purpose

Validate whether stronger severity settings for the five ECG augmentation /
corruption operators can create an ImageNet-C-style stress benchmark where the
PTB-XL-trained EfficientNet1DV2 drops by several percentage points on major
PN2021 centers.

This is evaluation-time corruption only. It does not change the training-time
AugMix defaults in `methods/augmix/severity.py`.

## Model And Clean Baseline

- Model: `super5_minresample_full10_perglobal_20260503/best_model.pt`
- Input protocol: 100 Hz, 1000 samples, `minimal_resample`,
  `per_sample_global`
- PN2021 mapping: `v5_super5_strict_voltage_pacing_suppress`
- Clean eval JSON:
  `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_result_v5_super5_strict_voltage_pacing_suppress_20260524.json`

Clean metrics:

| split | AUROC | AUPRC |
|---|---:|---:|
| PTB-XL fold10 | 0.9071 | 0.7744 |
| PN2021 7-center avg | 0.7785 | 0.4636 |
| PN2021 drop-all-zero avg | 0.7978 | 0.5722 |

## Stress-v2 Sweep

Output JSON:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_pn2021_c_stress_v2_stream_v5_20260524.json
```

Flattened CSV directory:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/pn2021_c_stress_v2_summary/
```

Centers:

```text
ningbo, chapman_shaoxing, cpsc_2018, georgia
```

Mean drop across the 4 centers:

| corruption | s1 drop AUROC/AUPRC | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| powerline_noise | +0.00/+0.02pp | +0.02/+0.05pp | +0.11/+0.17pp | +0.29/+0.46pp | +0.54/+0.91pp |
| emg_noise | +0.02/+0.09pp | +0.27/+0.64pp | +1.00/+2.23pp | +2.39/+4.01pp | +4.20/+6.03pp |
| baseline_wander | +0.03/-0.01pp | +0.15/+0.06pp | +0.59/+0.56pp | +1.54/+1.79pp | +3.69/+3.61pp |
| baseline_shift | +0.05/+0.07pp | +0.23/+0.42pp | +1.84/+2.61pp | +2.94/+4.05pp | +4.92/+6.60pp |
| random_leads_masking | +1.64/+2.64pp | +3.58/+5.37pp | +8.36/+10.28pp | +13.79/+15.54pp | +21.87/+20.99pp |

Overall mean across 5 corruptions x 5 severities x 4 centers:

| metric | corrupted mean | drop vs clean |
|---|---:|---:|
| macro AUROC | 0.8141 | +2.96pp |
| macro AUPRC | 0.4538 | +3.57pp |

## Interpretation

The stronger profile works for four of the five operators:

- EMG noise, baseline wander, baseline shift, and random lead masking all
  create multi-point PN2021 degradation at high severity.
- Powerline noise remains weak even at high amplitude in the current 100 Hz
  protocol; severity 5 drops only 0.54pp AUROC and 0.91pp AUPRC on average.
  This is likely because 50/60 Hz corruption is poorly represented after the
  100 Hz pipeline and can alias into a pattern the model tolerates.

For paper use, treat `stress_v2` as a robustness stress benchmark, not as a
physiologically faithful acquisition simulator. The training-time AugMix branch
should keep the milder default severity table unless a separate ablation proves
the stronger table improves generalization.
