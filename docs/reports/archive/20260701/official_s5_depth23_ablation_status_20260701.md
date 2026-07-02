# Official S5 Depth2+3 PN2021-C Ablation Status

Date: 2026-07-01

Scope: official severity=5, depth2+depth3 composite PN2021-C, four centers
(`cpsc_2018`, `chapman_shaoxing`, `georgia`, `ningbo`), seed `20260601`,
all-zero-kept macro AUROC/AUPRC. Corruption is applied before z-score.

## Composite Corruption Drop

Drop is Direct K500 fullFT clean minus Direct K500 fullFT PN2021-C.

| Backbone | Depth | Clean AUROC/AUPRC | Corrupted AUROC/AUPRC | Drop vs clean |
|---|---:|---:|---:|---:|
| EffNet | 2 | 0.8492 / 0.5271 | 0.7749 / 0.4317 | 7.42 / 9.54 pp |
| EffNet | 3 | 0.8492 / 0.5271 | 0.7448 / 0.3965 | 10.44 / 13.06 pp |
| ECGFounder | 2 | 0.8898 / 0.6142 | 0.7935 / 0.4631 | 9.63 / 15.12 pp |
| ECGFounder | 3 | 0.8898 / 0.6142 | 0.7611 / 0.4182 | 12.87 / 19.61 pp |

Depth=3 is harder than depth=2 for both backbones. Combined depth=2+3
corrupted metrics are 0.7599 / 0.4141 for EffNet Direct K500 fullFT and
0.7773 / 0.4406 for ECGFounder Direct K500 fullFT.

## Ours vs Corrupted-K500 Supervised Baseline

| Backbone | K500 clean direct FT (clean) | K500 clean direct FT PN2021-C | Ours vs clean direct PN2021-C | Corrupted-K500 supervised PN2021-C | Ours PN2021-C | Ours vs corrupted baseline | Ours clean | Ours clean vs K500 clean direct FT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EffNet c117 VAE-LH + locked three-chain AugMix | 0.8492 / 0.5271 | 0.7599 / 0.4141 | +4.78 / +6.56 pp | 0.7907 / 0.4602 | 0.8077 / 0.4797 | +1.70 / +1.95 pp | 0.8742 / 0.5683 | +2.51 / +4.12 pp |
| ECGFounder c110 VAE-LH + locked three-chain AugMix | 0.8898 / 0.6142 | 0.7773 / 0.4406 | +0.68 / +4.27 pp | 0.7717 / 0.4420 | 0.7841 / 0.4833 | +1.23 / +4.14 pp | 0.8867 / 0.6366 | -0.31 / +2.24 pp |

Per-depth view:

| Backbone | Depth | K500 clean direct FT PN2021-C | Corrupted-K500 supervised | Ours | Ours vs clean direct FT | Ours vs corrupted baseline |
|---|---:|---:|---:|---:|---:|---:|
| EffNet | 2 | 0.7749 / 0.4317 | 0.8053 / 0.4810 | 0.8197 / 0.4952 | +4.47 / +6.35 pp | +1.43 / +1.42 pp |
| EffNet | 3 | 0.7448 / 0.3965 | 0.7761 / 0.4394 | 0.7957 / 0.4642 | +5.09 / +6.77 pp | +1.96 / +2.48 pp |
| ECGFounder | 2 | 0.7935 / 0.4631 | 0.7886 / 0.4655 | 0.7999 / 0.5067 | +0.64 / +4.36 pp | +1.13 / +4.12 pp |
| ECGFounder | 3 | 0.7611 / 0.4182 | 0.7548 / 0.4184 | 0.7682 / 0.4600 | +0.71 / +4.18 pp | +1.34 / +4.16 pp |

Evidence:

- Direct fullFT composite drop summary:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_official_s5_composite_direct_drop_20260622/official_s5_composite_direct_drop_summary.csv`
- EffNet corrupted-K500 supervised summary:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_official_s5_composite_k500_direct_supervised_20260624/summary/fourcenter_method_compare_summary.json`
- EffNet c117 summary:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_goal_effnet_officials5_c117_fourcenter_20260624/eval_c117_officials5_fourcenter/summary/c117_fourcenter_official_s5_composite_method_summary.csv`
- ECGFounder c110 summary:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_goal_ecgfounder_officials5_c110_20260624/eval_c110_officials5_fourcenter/summary/ecgfounder_c110_fourcenter_official_s5_composite_method_summary.csv`
- ECGFounder direct corrupted-K500 supervised PN2021-C eval JSONs:
  `/home/linbinhao/ECG_adv_data/runs/pn2021c_ecgfounder_direct_corrupted_k500_supervised_official_s5_depth23_*_20260701/ecgf_direct_corrs5sup_eval_*_20260701_gpu*/<center>/ecgfounder_direct_corrupted_k500_supervised_officials5_depth23_ep10/eval_pn2021_c_ecgfounder_v7_refexcluded_stream_standard_official_s5_depth23_composite.json`

Corrupted-K500 supervised means direct fine-tuning on K500 clean plus all 10
pair and 10 triple official severity=5 composite corrupted views. It is a strong
supervised baseline, not our VAE-LH + AugMix main method. Ours clean has been
evaluated: EffNet c117 improves clean AUROC/AUPRC over K500 clean direct FT,
while ECGFounder c110 keeps clean AUROC roughly flat (-0.31 pp) and improves
clean AUPRC (+2.24 pp). ECGFounder c110 is still exploratory lineage until
rerun with a self-contained managed run card, resolved config, command, git SHA,
and env record.

## Remaining Required Work

1. ECGFounder matched official depth2+3 ablations: noAug, raw noVAE, and
   trainable-scope/LoRA if fullFT remains weak.
2. Random/no-hard-search latent-hull negative control under the same official
   evaluator.
3. Paper-grade replay records for the final promoted recipe: run card, resolved
   config, command, git SHA/dirty status, env, seed, K500 IDs, checkpoint policy,
   mapping/hash, and metric summaries.
