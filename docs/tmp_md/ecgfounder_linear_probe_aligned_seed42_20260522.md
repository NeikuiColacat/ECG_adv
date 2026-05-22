# ECGFounder Frozen Linear Probe 对齐实验 seed42

日期：2026-05-22

## 方法

本实验按当前对齐策略运行 ECGFounder 强基线：

```text
ECGFounder frozen encoder
-> PTB-XL fold 1-8 训练 Super5 线性头
-> PTB-XL fold 9 选择 best epoch
-> PTB-XL fold 10 与 PN2021 外部中心评估
```

输入协议：

```text
preprocess_policy = official_ptbxl_eval
PTB-XL source     = records500 / filename_hr
input shape       = 12 x 5000
duration          = 10 seconds
normalization     = per-sample global z-score
extra filtering   = no notch / no bandpass / no median baseline removal
```

标签协议：

```text
Super5 order = CD, HYP, MI, NORM, STTC
PN2021 mapping = v5_super5_strict_voltage_pacing_suppress_20260522
mapping hash = 1141e0a9f94b
```

运行目录：

```text
/root/autodl-tmp/paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_seed42_official
```

## 训练结果

| split | AUROC | AUPRC |
|---|---:|---:|
| PTB-XL fold 9 best epoch | 0.9227 | 0.7949 |
| PTB-XL fold 10 test | 0.9224 | 0.8016 |

best epoch：

```text
epoch = 49
selection metric = fold9 macro AUPRC
```

样本量：

| split | n |
|---|---:|
| train fold 1-8 | 17418 |
| val fold 9 | 2183 |
| test fold 10 | 2198 |

## PN2021 外部中心结果

target-center view 默认排除对应中心 K=500 anchor ids，和 VAE-only 在线对抗训练评估 denominator 对齐。

| target center | AUROC | AUPRC | effective n | excluded |
|---|---:|---:|---:|---:|
| ningbo | 0.8850 | 0.4342 | 34405 | 500 |
| chapman_shaoxing | 0.8946 | 0.3612 | 9747 | 500 |
| cpsc_2018 | 0.8219 | 0.5725 | 6377 | 500 |
| georgia | 0.8525 | 0.6551 | 9844 | 500 |

对应 7-center average：

| target view | avg AUROC | avg AUPRC |
|---|---:|---:|
| ningbo view | 0.8460 | 0.5502 |
| chapman_shaoxing view | 0.8479 | 0.5478 |
| cpsc_2018 view | 0.8455 | 0.5494 |
| georgia view | 0.8458 | 0.5502 |

## 当前结论

这版 ECGFounder baseline 已经和 EfficientNet1DV2 在实验协议层面对齐：

```text
same Super5 label space
same PTB-XL fold 1-8 / 9 / 10 split
same PN2021 v5 mapping
same AUROC/AUPRC metric
same target-anchor exclusion policy
```

不同点只保留在模型合理输入协议上：

```text
EfficientNet1DV2: 100Hz, 1000 samples
ECGFounder: 500Hz, 5000 samples
```

seed42 结果显示，ECGFounder frozen linear probe 在 PTB-XL fold10 上很强，PN2021 上也明显强于弱 zero-shot keyword mapping 基线。下一步应补：

1. seed2025 和 seed3407，确认线性头随机性。
2. `filtered_dataset` 消融，确认 ECGFounder README 中滤波链对结果的影响。
3. 汇总表：EfficientNet1DV2 baseline、VAE-only 在线对抗训练、ECGFounder frozen linear probe。
