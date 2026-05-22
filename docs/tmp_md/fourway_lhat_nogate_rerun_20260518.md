# Four-Way Latent-Hull AT No-Gate 复跑报告

日期：2026-05-18

## 实验目的

复跑下面四类方法，并明确关闭 online AT 里的 hard quality gate：

1. `baseline`：只用 PTB-XL 训练得到的 EfficientNet1DV2。
2. `VAE-only / real-anchor LH-AT`：只用目标中心 K=500 真实样本的 ECGTwin VAE latent 做 Latent-Hull AT。
3. `no-token + LH-AT`：目标中心 K=500 真实样本 + ECGTwin no-token 生成候选 latent。
4. `target-token + LH-AT`：目标中心 K=500 真实样本 + ECGTwin target-center prompt token 生成候选 latent。

## 关键设置

所有方法使用同一套评估口径：

```text
target refs       = K=500, full eval 时排除这 500 条目标中心样本
preprocess_mode   = minimal_resample
norm_mode         = per_sample_global
sampling rate     = 100 Hz
input length      = 1000 samples = 10 seconds
attack_mode       = latent_hull
hull_M            = 10
hull_lambda       = 0.15
hull_steps        = 5
hull_lr           = 0.25
n_epochs          = 10
adv_weight        = 0.06
PTB-XL stream     = enabled
VAE roundtrip     = enabled
quality gate      = disabled by --disable_quality_gate
```

注意：`--disable_quality_gate` 的含义是仍然计算和记录 gate 指标，但 medical/semantic gate
失败不会跳过 adversarial buffer push。

运行入口：

```text
scripts/paper/run_fourway_no_gate_lhat_20260518.py
```

输出目录：

```text
/root/autodl-tmp/paper_fourway_lhat_nogate_20260518/
```

## Target-Center 结果

表中括号内为相对同一 excluded-ref baseline 的提升，单位为 percentage points。

| center | baseline | VAE-only / real-anchor LH-AT | no-token + LH-AT | target-token + LH-AT |
|---|---:|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8918 / 0.5231 (+2.45pp / +4.48pp) | 0.8884 / 0.5167 (+2.12pp / +3.83pp) | 0.8885 / 0.5165 (+2.13pp / +3.81pp) |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4634 (+2.03pp / +3.82pp) | 0.8972 / 0.4653 (+2.09pp / +4.02pp) | 0.8972 / 0.4647 (+2.09pp / +3.95pp) |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8552 / 0.5973 (+4.38pp / +3.87pp) | 0.8541 / 0.5958 (+4.27pp / +3.72pp) | 0.8543 / 0.5959 (+4.28pp / +3.73pp) |
| georgia | 0.8157 / 0.5916 | 0.8259 / 0.6026 (+1.02pp / +1.10pp) | 0.8255 / 0.6019 (+0.98pp / +1.03pp) | 0.8256 / 0.6021 (+0.99pp / +1.05pp) |

四中心平均：

| method | avg target AUROC/AUPRC | avg delta |
|---|---:|---:|
| baseline | 0.8427 / 0.5134 | 0.00pp / 0.00pp |
| VAE-only / real-anchor LH-AT | 0.8674 / 0.5466 | +2.47pp / +3.32pp |
| no-token + LH-AT | 0.8663 / 0.5449 | +2.37pp / +3.15pp |
| target-token + LH-AT | 0.8664 / 0.5448 | +2.37pp / +3.14pp |

## PN2021 7-Center Average

这里是完整 PN2021 7-center 平均，不只看当前 target center。

| method | avg PN2021 AUROC/AUPRC |
|---|---:|
| baseline | 0.7780 / 0.4817 |
| VAE-only / real-anchor LH-AT | 0.7798 / 0.4850 |
| no-token + LH-AT | 0.7794 / 0.4843 |
| target-token + LH-AT | 0.7796 / 0.4844 |

## 结论

1. 关闭 hard quality gate 后，主线收益仍然稳定存在。四个目标中心上，VAE-only real-anchor LH-AT 平均提升约 `+2.47pp AUROC / +3.32pp AUPRC`。
2. `no-token + LH-AT` 和 `target-token + LH-AT` 与 VAE-only 都处在同一收益带，但没有明显超过 VAE-only。
3. target-token 相对 no-token 的平均差异约为 `+0.007pp AUROC / -0.014pp AUPRC`，基本可以视为噪声级。当前复跑不支持“center token 是主要性能增益来源”的强因果结论。
4. 从性能与复杂度看，论文主线仍建议写成：少量目标中心真实样本作为 real anchors，在 ECGTwin VAE latent space 做 Latent-Hull online AT。center token 保留为可控生成和候选池消融模块。
5. 质量 gate 在这组实验中不是性能增益的必要条件；关闭 gate 没有导致明显崩坏。但 gate 仍可作为医学合法性分析和样本可解释性报告的一部分，而不是主训练路径的必要过滤器。

## 产物

```text
summary md:
  /root/autodl-tmp/paper_fourway_lhat_nogate_20260518/summaries/fourway_lhat_nogate.md

summary csv:
  /root/autodl-tmp/paper_fourway_lhat_nogate_20260518/summaries/fourway_lhat_nogate.csv

run script:
  scripts/paper/run_fourway_no_gate_lhat_20260518.py
```
