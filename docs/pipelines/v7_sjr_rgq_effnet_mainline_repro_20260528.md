# V7 SJR/RGQ EfficientNet Mainline Reproduction, 2026-05-28

This run reran the EfficientNet1DV2 K500 mainline with the PN2021 Super5
`v7_super5_sjr_rgq_review_20260528` label mapping.

## Protocol

- Mapping: `v7_super5_sjr_rgq_review_20260528`
- Mapping hash: `555ec85d5b51`
- Class order: `CD, HYP, MI, NORM, STTC`
- Target centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`
- K-shot protocol: fixed `K=500`, seed `20260531`
- K500 records were relabeled under v7 and excluded from final target-center
  PN2021 evaluation.
- Direct run config: `configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml`
- VAE L-HAT run config: `configs/experiments/effnet_vae_lhat_k500_v7_sjr_rgq.yaml`

## Artifact Roots

```text
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/effnet_direct_k500_v7_sjr_rgq/v7_sjr_rgq_main_20260528
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/effnet_vae_lhat_k500_v7_sjr_rgq/v7_sjr_rgq_main_20260528
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/v7_sjr_rgq_main_20260528_effnet_direct_k500_v7_sjr_rgq
/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/runs/metrics_export/v7_sjr_rgq_main_20260528_effnet_vae_lhat_k500_v7_sjr_rgq
```

## Four-Center Target Mean

| View | Direct K500 AUROC | Direct K500 AUPRC | VAE L-HAT AUROC | VAE L-HAT AUPRC | VAE - Direct AUROC | VAE - Direct AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| PN2021 all-zero kept, ref-excluded | 0.852181 | 0.524691 | 0.871177 | 0.554395 | +1.90 pp | +2.97 pp |
| PN2021 drop-all-zero, ref-excluded | 0.876307 | 0.666638 | 0.899637 | 0.717650 | +2.33 pp | +5.10 pp |

## Per-Center All-Zero-Kept Metrics

| Center | Direct AUROC | Direct AUPRC | VAE AUROC | VAE AUPRC | VAE - Direct AUROC | VAE - Direct AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.885276 | 0.445553 | 0.897209 | 0.482611 | +1.19 pp | +3.71 pp |
| cpsc_2018 | 0.821217 | 0.559345 | 0.876214 | 0.621216 | +5.50 pp | +6.19 pp |
| georgia | 0.820684 | 0.598200 | 0.824654 | 0.601870 | +0.40 pp | +0.37 pp |
| ningbo | 0.881546 | 0.495668 | 0.886633 | 0.511884 | +0.51 pp | +1.62 pp |

## Per-Center Drop-All-Zero Metrics

| Center | Direct AUROC | Direct AUPRC | VAE AUROC | VAE AUPRC | VAE - Direct AUROC | VAE - Direct AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing | 0.903551 | 0.615505 | 0.919921 | 0.662489 | +1.64 pp | +4.70 pp |
| cpsc_2018 | 0.873591 | 0.708750 | 0.930770 | 0.823366 | +5.72 pp | +11.46 pp |
| georgia | 0.827686 | 0.678131 | 0.835276 | 0.691329 | +0.76 pp | +1.32 pp |
| ningbo | 0.900398 | 0.664165 | 0.912582 | 0.693415 | +1.22 pp | +2.93 pp |

## Diagnostics

- Four VAE center runs were launched across four GPUs after direct checkpoints
  became available.
- Attack success was in the intended 30-70% band for `ningbo`,
  `chapman_shaoxing`, and `georgia`.
- `cpsc_2018` ASR decayed below the intended band to roughly 0.16-0.21 late in
  training, but the same global recipe was preserved to avoid center-specific
  tuning.
- `git diff --check` passed after the run.
