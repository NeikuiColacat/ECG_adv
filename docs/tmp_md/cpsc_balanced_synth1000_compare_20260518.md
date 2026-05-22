# CPSC 2018 Balanced Synth1000 Pool 对比

## 为什么旧实验每个中心合成样本数不一样

旧实验没有强制每个中心、每个类别固定数量，而是按每个中心可用 reference、生成候选、筛选/合并后的实际保留样本数进入 pool。不同中心的可用类别和候选通过率不同，所以最终 no-token / target-token 的合成样本数不一致。

本轮按你的要求在 `cpsc_2018` 上固定生成：5 个 super5 类别各 200 条，总计 1000 条合成 ECG；训练时加入 pool，而不是单独评估合成 ECG。

## 统一设置

- baseline: PTB-XL full train baseline，`minimal_resample + per_sample_global + 100Hz + 10s`。
- target-center refs: CPSC 2018 K=500，评估时从 CPSC 测试集中排除这 500 条，避免污染。
- 在线对抗训练: `hull_M=10`, `lambda=0.15`, `hull_steps=5`, `adv_weight=0.06`, 10 epochs, `--disable_quality_gate`。
- no-token / target-token pool: `500 real anchors + 1000 synth`，source weights `real=1.0, synth=1.0`。
- CPSC 2018 原始 K=500 中只有 CD/NORM/STTC，没有 HYP/MI；因此 HYP/MI 在合成 arm 中只来自 ECGTwin 合成样本。

## 合成 pool 数量

| arm | synth pool | synth class counts | merged pool | merged class counts |
|---|---:|---|---:|---|
| no-token | 1000 | {'CD': 200, 'HYP': 200, 'MI': 200, 'NORM': 200, 'STTC': 200} | 1500 | {'CD': 459, 'HYP': 200, 'MI': 200, 'NORM': 316, 'STTC': 334} |
| target-token | 1000 | {'CD': 200, 'HYP': 200, 'MI': 200, 'NORM': 200, 'STTC': 200} | 1500 | {'CD': 459, 'HYP': 200, 'MI': 200, 'NORM': 316, 'STTC': 334} |

## 生成样本目标类置信度均值

| arm | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|
| no-token | 0.764 | 0.369 | 0.473 | 0.914 | 0.655 |
| target-token | 0.958 | 0.439 | 0.443 | 0.886 | 0.863 |

## CPSC 2018 全量测试结果

| method | CPSC AUROC | CPSC AUPRC | delta AUROC | delta AUPRC |
|---|---:|---:|---:|---:|
| baseline | 0.8115 | 0.5586 | +0.00pp | +0.00pp |
| VAE-only real-anchor 在线对抗训练 | 0.8539 | 0.5960 | +4.25pp | +3.74pp |
| no-token synth1000 + 在线对抗训练 | 0.8539 | 0.5959 | +4.24pp | +3.73pp |
| target-token synth1000 + 在线对抗训练 | 0.8539 | 0.5959 | +4.25pp | +3.73pp |

## 结论

固定每类 200、总计 1000 条合成样本后，no-token 和 target-token 在 CPSC 2018 上仍然与 VAE-only real-anchor 基本打平，没有显示出明显额外收益。target-token 在生成阶段提高了 CD/STTC 的目标类置信度，但 downstream 在线对抗训练指标没有随之拉开差距。

## Artifacts

- root: `/root/autodl-tmp/cpsc_balanced_synth1000_20260518`
- baseline: `/root/autodl-tmp/cpsc_balanced_synth1000_20260518/baseline_eval/cpsc_2018_baseline_exclrefs.json`
- vae_only: `/root/autodl-tmp/cpsc_balanced_synth1000_20260518/runs/vae_only_M10_lam0p15_ep10/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json`
- no_token_synth1000: `/root/autodl-tmp/cpsc_balanced_synth1000_20260518/runs/no_token_synth1000_M10_lam0p15_ep10/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json`
- target_token_synth1000: `/root/autodl-tmp/cpsc_balanced_synth1000_20260518/runs/target_token_synth1000_M10_lam0p15_ep10/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json`
