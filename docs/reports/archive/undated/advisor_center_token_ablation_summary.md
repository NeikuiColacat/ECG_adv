# Center Token / No Token / Real-Anchor LH-AT 对比简表

说明：

- 指标格式均为 `AUROC / AUPRC`。
- 括号内为相对同一测试子集 `baseline` 的变化量，即 `method - baseline`。
- delta 单位为百分点 `pp`，例如 `+2.46pp` 表示指标绝对值提升 `0.0246`。
- 本表使用 K=500 目标中心样本，评估时排除这 500 条 adaptation refs，避免测试污染。
- 训练与评估口径固定为 `minimal_resample + per_sample_global + 100Hz + 10s`。
- LH-AT 使用主线设置 `hull_lambda=0.15`。

## 目标中心测试集表现

| center | Baseline | 仅目标中心真实样本对抗训练 | 不带训练的提示向量 + 对抗训练 | 带训练的提示向量 + 对抗训练 |
|---|---:|---:|---:|---:|
| ningbo | 0.87 / 0.48 | 0.89 / 0.52 (+2.46pp / +4.50pp) | 0.89 / 0.52 (+2.06pp / +3.73pp) | 0.89 / 0.52 (+2.09pp / +3.71pp) |
| chapman_shaoxing | 0.88 / 0.43 | 0.90 / 0.46 (+2.03pp / +3.87pp) | 0.90 / 0.47 (+2.09pp / +4.02pp) | 0.90 / 0.46 (+2.09pp / +3.95pp) |
| cpsc_2018 | 0.81 / 0.56 | 0.85 / 0.60 (+4.30pp / +3.76pp) | 0.85 / 0.60 (+4.27pp / +3.72pp) | 0.85 / 0.60 (+4.28pp / +3.73pp) |
| georgia | 0.82 / 0.59 | 0.83 / 0.60 (+1.00pp / +1.04pp) | 0.81 / 0.59 (-0.30pp / -0.55pp) | 0.83 / 0.60 (+1.06pp / +1.13pp) |

## PN2021-C：原始数据集 vs ImageNet-C 风格增强/扰动副本

这张表使用当前论文主线 EfficientNet1DV2 baseline：

```text
checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

输入口径:
  minimal_resample + per_sample_global + 100Hz + 10s

PN2021-C:
  4 centers x 5 corruption operators x 5 severity levels
```

### 按中心汇总

| center | 原始 PN2021 clean | PN2021-C 加扰动算子 | delta |
|---|---:|---:|---:|
| ningbo | 0.8657 / 0.4842 | 0.8547 / 0.4663 | -1.10pp / -1.79pp |
| chapman_shaoxing | 0.8733 / 0.4427 | 0.8616 / 0.4277 | -1.16pp / -1.50pp |
| cpsc_2018 | 0.8154 / 0.5712 | 0.8092 / 0.5595 | -0.62pp / -1.17pp |
| georgia | 0.8167 / 0.5966 | 0.8074 / 0.5846 | -0.93pp / -1.20pp |
| 4-center mean | 0.8428 / 0.5237 | 0.8332 / 0.5095 | -0.95pp / -1.41pp |

### 按 corruption 算子汇总

| corruption operator | PN2021- 加扰动算子 | delta |
|---|---:|---:|
| powerline_noise | 0.8427 / 0.5234 | -0.01pp / -0.03pp |
| emg_noise | 0.8412 / 0.5204 | -0.15pp / -0.32pp |
| baseline_wander | 0.8421 / 0.5233 | -0.06pp / -0.04pp |
| baseline_shift | 0.8393 / 0.5189 | -0.34pp / -0.48pp |
| random_leads_masking | 0.8008 / 0.4617 | -4.20pp / -6.20pp |

## 简要结论

1. 在目标中心测试集上，K=500 的 LH-AT 基本都能提升 AUROC/AUPRC。
2. `仅目标中心真实 anchor LH-AT`、`No-token + LH-AT` 和 `Target-token + LH-AT` 三者整体接近。
3. `Target-token` 相比 `No-token` 没有稳定显著优势；只有 `georgia` 上 token 明显优于 no-token。
4. 当前更稳妥的结论是：主要收益来自目标中心真实样本构成的 ECGTwin VAE latent manifold，以及基于它的 latent-hull 对抗训练；center token 更适合作为生成控制与机制消融，而不是主要增益来源。
5. PN2021-C 上，5 个 ECG corruption 算子的整体平均影响不大：4-center mean 只下降约 `0.95pp AUROC / 1.41pp AUPRC`。
6. 下降主要来自 `random_leads_masking`；`powerline_noise`、`emg_noise`、`baseline_wander` 和 `baseline_shift` 在当前 severity 设置下影响较小。


老师最新一版的对抗训练方法大概有 2-3 个百分点的 AUROC/AUPRC 提升。

我们训练目标中心 prompt token 方案：使用 500 条目标中心样本训练代表目标中心特征的 prompt 向量。每个疾病类别对应
一组 4 x 768 维的 prompt vectors，一共训练 5 组，分别对应 5 个疾病类别。之后将这些 prompt token 加入 ECGTwin 的生成
条件中，用于合成带有目标中心风格的 ECG 样本。

对比了三种方案：不用diffusion的合成样本仅用500个目标中心样本做对抗训练、使用无目标中心 prompt token 的ECGTwin 合成样本、以及使用带目标中心 prompt token 的 ECGTwin 合成样本。结果发现，三种方法的性能提升整体比较接近，模型的性能提升应该主要来自于对抗训练。

另外，我还构建了类似 ImageNet-C 的多中心 ECG 鲁棒性测试集，对多个 PN2021 中心数据分别施加5 类 ECG 扰动算子。实验结
果显示，模型在这些ECG数据上的 AUROC/AUPRC 平均下降大约只有 1 个百分点左右
