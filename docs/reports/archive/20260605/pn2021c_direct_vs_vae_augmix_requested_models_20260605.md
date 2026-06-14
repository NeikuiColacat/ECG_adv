# PN2021-C Direct vs VAE+AugMix Requested Models - 2026-06-05

## Scope

Protocol:

- PN2021 v7 Super5 SJR/RGQ mapping, hash `555ec85d5b51`.
- Four target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- K=500 target reference records excluded from the matching target-center eval.
- PN2021-C standard corruption profile: 5 corruptions x public severity 1..5.
- Compared saved direct K500 heads/checkpoints against matched VAE-LHAT + AugMix saved heads/checkpoints.

Requested model coverage:

- EfficientNet1DV2.
- PTB-XL benchmarking backbones: FCN-Wang, Inception1D, ResNet1D-Wang.
- ECGFounder frozen encoder with saved direct or residual-adapter Super5 head.

Before this run, EfficientNet1DV2 had partial PN2021-C VAE noAug vs VAE+AugMix coverage, and ECGFounder had clean PN2021 AugMix coverage. The requested direct/original vs VAE+AugMix PN2021-C comparison across all three model families was not complete.

## Artifacts

Run root:

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/pn2021c_direct_vs_vae_augmix_mainline_v7_k500_20260605
```

Summary tables:

```text
summary/pn2021c_model_zoo_direct_vs_vae_augmix_mean.csv
summary/pn2021c_ecgfounder_direct_vs_vae_augmix_mean.csv
summary/pn2021c_all_requested_models_direct_vs_vae_augmix_mean.csv
```

ECGFounder evaluator added:

```text
scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py
```

## Mean Results

Values below are four-center means. Delta rows are VAE+AugMix minus direct in percentage points.

| Model | Clean AUROC delta | Clean AUPRC delta | PN2021-C AUROC delta | PN2021-C AUPRC delta | Drop AUROC delta | Drop AUPRC delta |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet1DV2 | +1.91 | +3.02 | +1.77 | +2.71 | +0.14 | +0.31 |
| FCN-Wang | +0.54 | +1.12 | +0.39 | +0.81 | +0.15 | +0.30 |
| Inception1D | +0.61 | +1.24 | +0.48 | +0.98 | +0.14 | +0.26 |
| ResNet1D-Wang | +0.88 | +1.96 | +0.71 | +1.69 | +0.17 | +0.28 |
| ECGFounder | +0.64 | +1.30 | +0.95 | +2.08 | -0.31 | -0.79 |

Interpretation:

- VAE+AugMix improves absolute PN2021-C corrupted AUROC/AUPRC for every requested backbone family.
- For the CNN/model-zoo backbones, the clean model improves more than the corrupted model, so the clean-to-corrupt drop rises slightly. Report these as absolute corrupted-performance gains, not as relative robustness/drop improvements.
- ECGFounder is stronger here: VAE+AugMix improves absolute corrupted performance and reduces the clean-to-corrupt drop.

## Verification

- Full PN2021-C JSON count under the run root: 40.
- CNN/model-zoo jobs: 32 JSONs.
- ECGFounder jobs: 8 JSONs.
- Regression tests: `23 passed in 5.79s`.
