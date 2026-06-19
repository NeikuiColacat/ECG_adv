# PN2021-C Dual-Model Calibration Baseline Inventory

Date: 2026-06-18

Purpose: freeze the direct-K500/fullFT baseline paths used for the
`dual_model_10to15pp_v1` corruption-parameter calibration. These are calibration
baselines only; VAE-LH, AugMix, stabilizer, and raw-supervised methods are not
part of this baseline inventory.

## Protocol

- Class order: `CD, HYP, MI, NORM, STTC`.
- PN2021 mapping: `v7_super5_sjr_rgq_review_20260528`.
- Mapping hash: `555ec85d5b51`.
- K-shot protocol: K=500, seed `20260531`.
- Calibration centers: `cpsc_2018`, `georgia`.
- Confirmation centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- EfficientNet checkpoint policy: `best_model.pt` from direct K500 supervised fine-tuning.
- ECGFounder checkpoint policy: locked full fine-tuning `last_model.pt`; no K500 validation split.
- Clean drop baseline: same-center ref-excluded clean metrics from the listed clean eval JSONs.

## EfficientNet1DV2 Direct-K500/FullFT

Run root:

```text
/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605
```

Clean metric source: `pn2021.per_center[center]` inside
`eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json`.

| Center | Clean AUROC | Clean AUPRC | Checkpoint | Clean eval JSON | K500 ref-meta |
|---|---:|---:|---|---|---|
| `ningbo` | 0.881419 | 0.495540 | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/ningbo/runs/ningbo_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt` | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/ningbo/runs/ningbo_K500_direct_ft_ep30_seed20260531_val0.2/eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json` |
| `chapman_shaoxing` | 0.885275 | 0.445541 | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/chapman_shaoxing/runs/chapman_shaoxing_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt` | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/chapman_shaoxing/runs/chapman_shaoxing_K500_direct_ft_ep30_seed20260531_val0.2/eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/chapman_shaoxing/k500_seed20260531/chapman_shaoxing_real_k500_seed20260531.ref_meta.json` |
| `cpsc_2018` | 0.821216 | 0.559339 | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/cpsc_2018/runs/cpsc_2018_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt` | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/cpsc_2018/runs/cpsc_2018_K500_direct_ft_ep30_seed20260531_val0.2/eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531.ref_meta.json` |
| `georgia` | 0.820685 | 0.598213 | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/georgia/runs/georgia_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt` | `/home/linbinhao/ECG_adv_data/runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605/georgia/runs/georgia_K500_direct_ft_ep30_seed20260531_val0.2/eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/georgia/k500_seed20260531/georgia_real_k500_seed20260531.ref_meta.json` |

## ECGFounder Direct-K500/FullFT

Run root:

```text
/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs
```

Clean metric source: `target_excluding_ref` inside `eval_result.json`.

| Center | Clean AUROC | Clean AUPRC | Checkpoint | Clean eval JSON | K500 ref-meta |
|---|---:|---:|---|---|---|
| `ningbo` | 0.901246 | 0.541138 | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/ningbo_k500_fullft_locked/last_model.pt` | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/ningbo_k500_fullft_locked/eval_result.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json` |
| `chapman_shaoxing` | 0.912462 | 0.528082 | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/chapman_shaoxing_k500_fullft_locked/last_model.pt` | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/chapman_shaoxing_k500_fullft_locked/eval_result.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/chapman_shaoxing/k500_seed20260531/chapman_shaoxing_real_k500_seed20260531.ref_meta.json` |
| `cpsc_2018` | 0.881184 | 0.700885 | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/cpsc_2018_k500_fullft_locked/last_model.pt` | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/cpsc_2018_k500_fullft_locked/eval_result.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531.ref_meta.json` |
| `georgia` | 0.853363 | 0.657152 | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/georgia_k500_fullft_locked/last_model.pt` | `/home/linbinhao/ECG_adv_data/runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/georgia_k500_fullft_locked/eval_result.json` | `/home/linbinhao/ECG_adv_data/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/georgia/k500_seed20260531/georgia_real_k500_seed20260531.ref_meta.json` |

## Readiness Verdict

All required direct-K500/fullFT baseline checkpoints, clean eval JSONs, and K500
ref-meta files are present for the four target centers. No baseline retraining is
required before the custom-profile smoke sweep.

Important command detail: EfficientNet clean eval filenames are
`eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json`, not
`eval_result_v7_exclrefs_crop1000.json`.
