# Thesis-Selected Synthetic ECG Examples

Source NPZ: `/root/autodl-tmp/graduate_project/self_distill_v2_filtered_v46_ptbxl_contrast_seed42/synth_v2_filtered_top4000_gamma03.npz`

| class | source index | quality | score | target conf | HR | target digital pass | figure |
|---|---:|---|---:|---:|---:|---|---|
| CD | 308 | pass | 329.7 | 0.9652 | 101.7 | True | `/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_CD_12lead.png` |
| HYP | 1225 | pass | 370.9 | 0.8884 | 89.6 | True | `/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_HYP_12lead.png` |
| MI | 1864 | pass | 340.7 | 0.8422 | 80.0 | True | `/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_MI_12lead.png` |
| NORM | 3129 | pass | 119.9 | 0.9978 | 61.9 | False | `/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_NORM_12lead.png` |
| STTC | 3462 | pass | 374.4 | 0.9865 | 82.2 | True | `/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_STTC_12lead.png` |

Vector sample NPZ:
`/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/thesis_selected_samples.npz`

Saved arrays: `signals` (5,1000,12), `labels` (5,5), `class_names`, `target_class`, `source_indices`, and metadata JSON.
