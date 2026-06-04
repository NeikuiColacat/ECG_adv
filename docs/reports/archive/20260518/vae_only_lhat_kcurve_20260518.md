# VAE-only real-anchor LH-AT K 曲线结果

实验日期：2026-05-18

## 实验设置

本轮测试的是当前主线的 VAE-only 方法：

- 不使用 ECGTwin DiT 合成样本。
- 不使用 center token。
- 只使用目标中心真实 ECG 的 ECGTwin VAE latent 作为 anchor。
- 关闭 quality gate。
- `M=20`，`lambda=0.15`，`epochs=10`，`hull_steps=5`。
- `target_real_weight=40.0`，`adv_weight=0.06`，`roundtrip_anchor_n=1500`。
- 每个 K 的目标中心测试指标都排除了参与适配的 K 条 ref ECG，避免测试污染。
- baseline 是 PTB-XL full train baseline：`minimal_resample + 100Hz + 10s + per_sample_global`。

输出目录：

```text
/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518/
```

自动汇总文件：

```text
/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518/summaries/vae_only_lhat_kcurve.csv
/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518/summaries/vae_only_lhat_kcurve.md
```

## 结论

整体趋势符合预期：K 越大，目标中心 AUROC/AUPRC 通常越好，尤其是 `ningbo`、`chapman_shaoxing`、`cpsc_2018` 三个中心。`georgia` 的增益较小，K=300 左右基本达到峰值。

K=500 时四个中心的目标中心提升如下：

| center | baseline | VAE-only LH-AT | delta |
|---|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8917 / 0.5223 | +2.44pp / +4.39pp |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4638 | +2.03pp / +3.87pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp |
| georgia | 0.8157 / 0.5916 | 0.8256 / 0.6020 | +1.00pp / +1.04pp |

注意：LH-AT 对目标中心提升明显，但 PTB-XL fold10 指标会随 K 增大略微下降，说明它是目标中心适配方法，不是无代价的全局提升方法。

## Ningbo

| K | baseline | VAE-only LH-AT | delta |
|---:|---:|---:|---:|
| 20 | 0.8656 / 0.4841 | 0.8707 / 0.4932 | +0.51pp / +0.91pp |
| 50 | 0.8665 / 0.4841 | 0.8754 / 0.5004 | +0.89pp / +1.63pp |
| 100 | 0.8663 / 0.4831 | 0.8793 / 0.5072 | +1.30pp / +2.41pp |
| 200 | 0.8665 / 0.4811 | 0.8826 / 0.5118 | +1.61pp / +3.08pp |
| 300 | 0.8672 / 0.4806 | 0.8880 / 0.5202 | +2.09pp / +3.95pp |
| 400 | 0.8670 / 0.4793 | 0.8887 / 0.5191 | +2.17pp / +3.97pp |
| 500 | 0.8672 / 0.4784 | 0.8917 / 0.5223 | +2.44pp / +4.39pp |

Ningbo 是最干净的 K 增长曲线之一，K=20 就有正向提升，K=500 达到本轮最好结果。

## Chapman-Shaoxing

| K | baseline | VAE-only LH-AT | delta |
|---:|---:|---:|---:|
| 20 | 0.8732 / 0.4422 | 0.8753 / 0.4449 | +0.21pp / +0.27pp |
| 50 | 0.8726 / 0.4407 | 0.8762 / 0.4442 | +0.36pp / +0.35pp |
| 100 | 0.8725 / 0.4365 | 0.8815 / 0.4498 | +0.89pp / +1.33pp |
| 200 | 0.8737 / 0.4381 | 0.8885 / 0.4641 | +1.47pp / +2.60pp |
| 300 | 0.8736 / 0.4374 | 0.8916 / 0.4700 | +1.80pp / +3.26pp |
| 400 | 0.8760 / 0.4318 | 0.8950 / 0.4695 | +1.90pp / +3.78pp |
| 500 | 0.8763 / 0.4251 | 0.8967 / 0.4638 | +2.03pp / +3.87pp |

Chapman-Shaoxing 在 K>=200 后提升明显，K=400 到 K=500 基本进入平台期。

## CPSC 2018

| K | baseline | VAE-only LH-AT | delta |
|---:|---:|---:|---:|
| 20 | 0.8153 / 0.5707 | 0.8144 / 0.5751 | -0.09pp / +0.44pp |
| 50 | 0.8150 / 0.5703 | 0.8216 / 0.5790 | +0.66pp / +0.87pp |
| 100 | 0.8143 / 0.5687 | 0.8240 / 0.5789 | +0.97pp / +1.03pp |
| 200 | 0.8139 / 0.5662 | 0.8403 / 0.5862 | +2.64pp / +2.00pp |
| 300 | 0.8131 / 0.5636 | 0.8461 / 0.5896 | +3.30pp / +2.60pp |
| 400 | 0.8127 / 0.5614 | 0.8520 / 0.5943 | +3.93pp / +3.30pp |
| 500 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp |

CPSC 2018 的 AUROC 提升最大，但本轮 `classes_in_scope=NORM/MI/STTC` 下，CPSC 2018 的 MI walker anchor 为 0，提升主要来自 NORM/STTC latent-hull、目标真实样本监督流和 VAE roundtrip anchor。

## Georgia

| K | baseline | VAE-only LH-AT | delta |
|---:|---:|---:|---:|
| 20 | 0.8168 / 0.5965 | 0.8198 / 0.5970 | +0.30pp / +0.06pp |
| 50 | 0.8167 / 0.5959 | 0.8201 / 0.5984 | +0.34pp / +0.24pp |
| 100 | 0.8162 / 0.5948 | 0.8220 / 0.5988 | +0.58pp / +0.40pp |
| 200 | 0.8156 / 0.5930 | 0.8246 / 0.6035 | +0.90pp / +1.05pp |
| 300 | 0.8160 / 0.5931 | 0.8266 / 0.6059 | +1.06pp / +1.28pp |
| 400 | 0.8159 / 0.5925 | 0.8262 / 0.6039 | +1.03pp / +1.15pp |
| 500 | 0.8157 / 0.5916 | 0.8256 / 0.6020 | +1.00pp / +1.04pp |

Georgia 的提升幅度最小，K=300 是本轮目标中心最优点。K=400/500 没有继续上升，可能和 MI anchor 极少、中心标签分布和当前 `NORM/MI/STTC` trusted scope 有关。

## 解释和限制

1. 这轮是单 seed/subset seed：`20260531`。如果论文里要强调稳定性，需要补多 seed 或 K-fold 复现实验。
2. K 越小，类别覆盖越不完整。低 K 下某些中心缺少 MI anchor，因此不应把 K=20/50 的曲线解释成完整五类适配能力。
3. 本轮方法只在 `NORM/MI/STTC` 上做 latent-hull trusted attack，`CD/HYP` 没进入主攻击类别。后续若要覆盖完整 super5，需要单独处理 CD/HYP 的医学合法性和标签可信度。
4. 目标中心提升与 PTB-XL source performance 存在 trade-off。K=500 时 PTB-XL fold10 从接近 `0.9098 / 0.7800` 降到约 `0.9038-0.9052 / 0.7660-0.7718`，属于目标中心适配带来的轻微 source forgetting。
5. `/root/autodl-tmp` 在实验结束后剩余约 15GB，可继续读结果和小规模复现；继续大规模 sweep 前建议清理旧 checkpoints 或迁出结果。
