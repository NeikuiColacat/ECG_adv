# ECGFounder 线性头与 VAE-only 路线 review

日期：2026-05-23

## 任务 1：ECGFounder 微调分类头策略

当前强基线是：

```text
ECGFounder frozen encoder
-> PTB-XL fold 1-8 训练 Super5 Linear(1024, 5)
-> fold 9 按 macro AUPRC 选择 best epoch
-> fold 10 与 PN2021 v5 ref-excluded 评估
```

seed42 已完成：

| split / center | AUROC | AUPRC |
|---|---:|---:|
| PTB-XL fold10 | 0.9224 | 0.8016 |
| ningbo | 0.8850 | 0.4342 |
| chapman_shaoxing | 0.8946 | 0.3612 |
| cpsc_2018 | 0.8219 | 0.5725 |
| georgia | 0.8525 | 0.6551 |

2026-05-23 已补 seed2025/seed3407：

| seed | PTB-XL fold10 |
|---:|---:|
| 42 | 0.9224 / 0.8016 |
| 2025 | 0.9214 / 0.8012 |
| 3407 | 0.9217 / 0.8011 |

PN2021 4-center target view，3 seed mean：

| center | AUROC | AUPRC |
|---|---:|---:|
| ningbo | 0.8850 | 0.4347 |
| chapman_shaoxing | 0.8943 | 0.3622 |
| cpsc_2018 | 0.8218 | 0.5743 |
| georgia | 0.8527 | 0.6553 |
| 4-center avg | 0.8635 | 0.5067 |

Review 结论：

1. 协议合理：encoder 冻结、PTB-XL split 正确、PN2021 使用 v5 映射、target K=500 评估排除已经实现。
2. 多 seed 稳定：PTB-XL fold10 和 PN2021 target view 的随机性很小。
3. 需要补强：fold9 阈值/校准、drop-all-zero sensitivity、per-class PN2021 csv。
4. 已修补：线性头脚本新增 `--feature_cache_dir`；K-shot head FT 脚本默认读取 2026-05-22 v5 feature cache，并支持 `preprocess_policy` 后缀。
5. 重要边界：这不是 ECGFounder 官方 Super5 head，而是我们基于 ECGFounder frozen encoder 训练的 Super5 线性探针。

## 任务 2：VAE-only 路线 refinement

当前 v5 主结果：

| center | baseline | VAE-only | delta |
|---|---:|---:|---:|
| ningbo | 0.8718 / 0.4222 | 0.8880 / 0.4393 | +1.62pp / +1.71pp |
| chapman_shaoxing | 0.8805 / 0.3458 | 0.8949 / 0.3720 | +1.44pp / +2.63pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp |
| georgia | 0.8168 / 0.5913 | 0.8266 / 0.6012 | +0.98pp / +0.99pp |

最优先改进方向：

```text
legacy:
  classes_in_scope = NORM, MI, STTC
  CD/HYP trust = 0

new:
  trust_policy = real_all_present
  classes_in_scope = CD, HYP, MI, NORM, STTC
```

原因：

历史 CD/HYP 关闭是为了避免 ECGTwin synthetic HYP/CD 医学质量不足；但 VAE-only real-anchor 使用的是目标中心真实 ECG latent。真实 CD/HYP anchor 不应被 synthetic 质量 gate 误伤。

第一轮实验：

```text
center = ningbo, chapman_shaoxing, cpsc_2018, georgia
K = 500
variant = lambda015
M = 20
epochs = 30
trust_policy = real_all_present
classes_in_scope = CD HYP MI NORM STTC
quality gate = disabled
```

已完成结果：

| center | legacy VAE-only | all-class VAE-only | all-class vs legacy | baseline -> all-class |
|---|---:|---:|---:|---:|
| ningbo | 0.8880 / 0.4393 | 0.8894 / 0.4419 | +0.15pp / +0.26pp | +1.77pp / +1.97pp |
| chapman_shaoxing | 0.8949 / 0.3720 | 0.8964 / 0.3750 | +0.15pp / +0.30pp | +1.59pp / +2.93pp |
| cpsc_2018 | 0.8545 / 0.5962 | 0.8684 / 0.6151 | +1.40pp / +1.89pp | +5.70pp / +5.65pp |
| georgia | 0.8266 / 0.6012 | 0.8280 / 0.6033 | +0.14pp / +0.21pp | +1.12pp / +1.20pp |
| 4-center avg | 0.8660 / 0.5022 | 0.8706 / 0.5088 | +0.46pp / +0.66pp | - |

结论：

1. `real_all_present` 是正向 refinement，四个中心全部相对 legacy VAE-only 提升。
2. `cpsc_2018` 提升最大，因为 legacy 关闭 CD，而该中心 K=500 真实 anchor 主要是 CD/NORM/STTC。
3. per-class 看，`cpsc_2018` 的 CD/NORM/STTC 都提升；ningbo/chapman 的 HYP 或 MI 因正样本少反而略降。
4. 该实验是 target-center 强适配；7-center average 没同步提高，说明下一步应做 per-center/per-class 权重，而不是盲目继续放大 epoch。
5. PTB-XL fold10 保持在约 `0.901-0.904 / 0.761-0.769`，没有严重源域崩坏，但 AUPRC 仍低于原始 PTB-XL source baseline。

## 离 PTB-XL 同源性能的差距

PTB-XL fold10：

```text
EfficientNet1DV2 baseline roughly 0.906 / 0.775
ECGFounder linear probe seed42 0.9224 / 0.8016
```

PN2021 外部中心目前 AUPRC 明显低于 PTB-XL，尤其 ningbo/chapman。这个差距来自跨中心域移、标签体系不完全一致、all-zero 样本大量存在和类别分布差异。不能把“达到 PTB-XL 同源绝对指标”当作已经完成；需要通过 all-class trust、per-class loss/采样、target real 权重、ECGFounder head-only AT 等逐步逼近。
