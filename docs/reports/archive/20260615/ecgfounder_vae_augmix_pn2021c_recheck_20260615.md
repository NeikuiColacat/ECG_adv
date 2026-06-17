# ECGFounder VAE+AugMix PN2021-C Recheck - 2026-06-15

Scope: ECGFounder v7 SJR/RGQ K500, centers `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`, ref-excluded PN2021/PN2021-C. No GPU rerun was needed because current artifacts already covered every required center/variant.

Mapping: `v7_super5_sjr_rgq_review_20260528`, hash `555ec85d5b51`. PN2021-C standard profile is five corruptions by public severity 1..5, 25 units per center.

## Validation

- all referenced ECGFounder standard and PN2021-C drop-all-zero source files exist.
- PN2021-C drop-all-zero JSONs contain drop_all_zero_macro_auroc/drop_all_zero_macro_auprc/n_all_zero_labels.
- all ECGFounder variants cover four target centers for clean, PN2021-C standard, and PN2021-C drop-all-zero.
- Backfill manifest: `docs/reports/archive/20260605/pn2021c_dropallzero_eval_manifest_20260605.json` with `n_jobs_total=68`.
- Plan: `docs/superpowers/plans/2026-06-15-ecgfounder-vae-augmix-pn2021c-eval.md`.

## Four-Center Means

| View | Direct | VAE noAug | VAE+AugMix | AugMix - direct | AugMix - noAug |
|---|---:|---:|---:|---:|---:|
| PN2021 clean AUROC | 0.897254 | 0.903117 | 0.903672 | +0.642 pp | +0.055 pp |
| PN2021 clean AUPRC | 0.632539 | 0.646974 | 0.645512 | +1.297 pp | -0.146 pp |
| PN2021-C AUROC | 0.890041 | 0.896905 | 0.899531 | +0.949 pp | +0.263 pp |
| PN2021-C AUPRC | 0.616535 | 0.631439 | 0.637371 | +2.084 pp | +0.593 pp |
| PN2021-C drop-all-zero AUROC | 0.909594 | 0.914840 | 0.917660 | +0.807 pp | +0.282 pp |
| PN2021-C drop-all-zero AUPRC | 0.728681 | 0.743770 | 0.750777 | +2.210 pp | +0.701 pp |

## PN2021-C Per-Center Deltas

| Center | Standard AugMix - direct | Standard AugMix - noAug | Drop-all-zero AugMix - direct | Drop-all-zero AugMix - noAug |
|---|---:|---:|---:|---:|
| ningbo | +0.000/+0.000 pp | +0.000/+0.000 pp | +0.000/+0.000 pp | +0.000/+0.000 pp |
| chapman_shaoxing | +0.000/+0.000 pp | +0.000/+0.000 pp | +0.000/+0.000 pp | +0.000/+0.000 pp |
| cpsc_2018 | +3.787/+8.324 pp | +1.007/+2.259 pp | +3.264/+8.871 pp | +1.100/+2.741 pp |
| georgia | +0.009/+0.010 pp | +0.044/+0.114 pp | -0.037/-0.032 pp | +0.028/+0.062 pp |

## Interpretation

- Against matched ECGFounder direct K500, VAE+AugMix improves PN2021-C standard by +0.949 AUROC pp and +2.084 AUPRC pp.
- Against VAE noAug, the AugMix branch still improves PN2021-C standard by +0.263 AUROC pp and +0.593 AUPRC pp.
- Under PN2021-C drop-all-zero, VAE+AugMix improves over direct by +0.807 AUROC pp and +2.210 AUPRC pp, and over noAug by +0.282 AUROC pp and +0.701 AUPRC pp.
- Clean PN2021 all-zero-kept shows a small AugMix AUROC gain over noAug (+0.055 pp) but AUPRC drops (-0.146 pp), so the clean-view story is not uniformly better; the PN2021-C result is stronger.

## Source Tables

- Standard means: `docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_mean_20260605.csv`
- Standard per-center: `docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv`
- PN2021-C drop-all-zero means: `docs/reports/archive/20260605/pn2021c_drop_all_zero_direct_vae_augmix_mean_20260605.csv`
- PN2021-C drop-all-zero per-center: `docs/reports/archive/20260605/pn2021c_drop_all_zero_direct_vae_augmix_per_center_20260605.csv`
- Recheck CSV: `docs/reports/archive/20260615/ecgfounder_vae_augmix_pn2021c_recheck_20260615.csv`
