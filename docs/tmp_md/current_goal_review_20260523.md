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

## 任务 2 追加：ECGFounder + VAE-only head AT

新增脚本：

```text
scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py
```

设计：

```text
冻结 ECGFounder encoder
加载 PTB-XL Super5 Linear(1024,5) head
每个 epoch 用当前 head 做 VAE latent-hull 在线对抗搜索
decode 后经过 ECGFounder encoder 抽 feature
只更新 Super5 线性头
```

20 epoch，K=500，M=20，lambda=0.15，hull_steps=3，K_anchor=150：

| center | ECGFounder baseline | ECGFounder + VAE-only head AT | delta |
|---|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9322 / 0.5601 | +4.72pp / +12.59pp |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9386 / 0.4762 | +4.40pp / +11.50pp |
| cpsc_2018 | 0.8219 / 0.5725 | 0.8852 / 0.6590 | +6.33pp / +8.64pp |
| georgia | 0.8525 / 0.6551 | 0.8878 / 0.7250 | +3.53pp / +6.99pp |
| mean | 0.8635 / 0.5058 | 0.9110 / 0.6051 | +4.75pp / +9.93pp |

PTB-XL fold10 源域保持：

| method | PTB-XL fold10 |
|---|---:|
| ECGFounder linear probe | 0.9224 / 0.8016 |
| ECGFounder + VAE-only head AT mean after target adaptation | 0.9120 / 0.7783 |

当前判断：

1. ECGFounder 版本的 VAE-only 路线比 EfficientNet1DV2 版本更强，尤其是 cpsc_2018：`0.8852 / 0.6590` vs EfficientNet all-class VAE-only `0.8684 / 0.6151`。
2. 四个中心 AUROC 已经明显接近或超过 PTB-XL 源域 AUROC，但 AUPRC 还没有达到 PTB-XL 源域 `0.80` 左右。
3. 这说明目标“做到和 PTB-XL 同样 AUROC/AUPRC”还没完成，但 ECGFounder head-only AT 是目前最有希望的路线。
4. 下一步优先减少源域遗忘并继续推高 AUPRC：head L2/EWA anchor、source_weight 提高、per-center best epoch、seed 稳定性。

### CPSC source_weight 消融

为了平衡目标中心适配和 PTB-XL 源域保持，先在 `cpsc_2018` 上试了 `source_weight`：

| source_weight | target center | target delta | PTB-XL fold10 after AT |
|---:|---:|---:|---:|
| 1.0 | 0.8852 / 0.6590 | +6.33pp / +8.64pp | 0.9112 / 0.7789 |
| 2.0 | 0.8755 / 0.6336 | +5.36pp / +6.10pp | 0.9157 / 0.7882 |
| 3.0 | 0.8681 / 0.6216 | +4.62pp / +4.91pp | 0.9177 / 0.7921 |

解释：

```text
source_weight 越大，PTB-XL 源域 AUPRC 保持越好，但目标中心 AUPRC 增益下降。
```

论文主实验如果强调 target adaptation，可以用 `source_weight=1.0`；如果导师更关心不牺牲源域，则 `source_weight=2.0` 是更稳的折中。

### Target-heavy 上限实验

为了测试目标中心性能上限，使用：

```text
source_weight      = 1.0
target_real_weight = 80.0
adv_weight         = 40.0
epochs             = 20
```

四中心结果：

| center | ECGFounder baseline | target-heavy VAE-only head AT | delta |
|---|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9394 / 0.5962 | +5.44pp / +16.20pp |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9408 / 0.4927 | +4.61pp / +13.15pp |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9011 / 0.7086 | +7.92pp / +13.60pp |
| georgia | 0.8525 / 0.6551 | 0.8960 / 0.7459 | +4.35pp / +9.08pp |
| mean | 0.8635 / 0.5058 | 0.9193 / 0.6358 | +5.58pp / +13.01pp |

对比默认权重：

| config | target mean | PTB-XL fold10 mean after AT |
|---|---:|---:|
| default `source/target/adv = 1/20/10` | 0.9110 / 0.6051 | 0.9120 / 0.7783 |
| target-heavy `1/80/40` | 0.9193 / 0.6358 | 0.9000 / 0.7555 |

当前最好目标中心结果来自 target-heavy；当前最好源域保持来自更高 source_weight 或 source-floor checkpoint selection。两者还没有同时达到 PTB-XL 源域 AUPRC。

### Relative head-anchor 正则

脚本已改成支持相对 head-anchor：

```text
relative_head_anchor = ||head - source_head||^2 / ||source_head||^2
```

在 `cpsc_2018` + target-heavy `1/80/40` 下：

| head_l2_anchor | target center | PTB-XL fold10 after AT |
|---:|---:|---:|
| 0.0 | 0.9011 / 0.7086 | 0.8998 / 0.7559 |
| 0.25 | 0.8962 / 0.6933 | 0.9044 / 0.7652 |
| 1.0 | 0.8859 / 0.6640 | 0.9102 / 0.7776 |

这个正则能有效控制 head 偏移，但越保源，目标中心 AUPRC 越低。后续应在 `0.1-0.5` 做 per-center sweep，而不是全中心固定同一个值。

### Class-aware refinement

新增脚本参数：

```text
--class_loss_weights
--target_class_sample_weights
--adv_class_sample_weights
```

稀有类定向实验：

| center | target-heavy baseline | class-aware result | delta |
|---|---:|---:|---:|
| chapman_shaoxing | 0.9408 / 0.4927 | 0.9401 / 0.4993 | -0.07pp / +0.66pp |
| ningbo | 0.9394 / 0.5962 | 0.9374 / 0.6009 | -0.20pp / +0.47pp |
| cpsc_2018 STTC boost | 0.9011 / 0.7086 | 0.8936 / 0.6781 | -0.75pp / -3.05pp |

有效改善的 per-class：

```text
chapman HYP AUPRC: 0.0276 -> 0.0369
chapman MI  AUPRC: 0.0865 -> 0.1135
ningbo  MI  AUPRC: 0.2376 -> 0.2653
```

判断：class-aware 能小幅改善稀有类，但不是决定性突破。ningbo/chapman 的 AUPRC 上限受稀有阳性数量和标签域差异限制很明显。

### EfficientNet1DV2 target-heavy check

为了验证同样思路能不能拉高 EfficientNet1DV2，跑了 CPSC：

| model/config | CPSC target | PTB-XL fold10 |
|---|---:|---:|
| EfficientNet all-class default | 0.8684 / 0.6151 | 0.9012 / 0.7629 |
| EfficientNet target-heavy | 0.8758 / 0.6258 | 0.8950 / 0.7492 |
| ECGFounder head target-heavy | 0.9011 / 0.7086 | 0.8998 / 0.7559 |

结论：EfficientNet1DV2 也能被 target-heavy 推高一点，但远不如 ECGFounder head AT。目标里“EfficientNet1DV2 也达到 PTB-XL 源域指标”目前还没有实现。

### EfficientNet1DV2 source-consistency pilot

给 EfficientNet 在线 AT 脚本新增：

```text
--source_logit_anchor_weight
--source_logit_anchor_batches
```

做法：训练开始时复制 frozen initial EfficientNet teacher，每个 mixed target/adv epoch 后，用 PTB-XL source batch 做 logits MSE 约束。

`cpsc_2018` full-set 结果：

| config | CPSC target | PTB-XL fold10 | PN2021 avg |
|---|---:|---:|---:|
| target-heavy no anchor | 0.8758 / 0.6258 | 0.8950 / 0.7492 | 0.7868 / 0.4593 |
| source-logit anchor 0.1 | 0.8731 / 0.6208 | 0.8999 / 0.7604 | 0.7847 / 0.4593 |
| source-logit anchor 0.5 | 0.8571 / 0.6004 | 0.9055 / 0.7724 | 0.7840 / 0.4623 |

判断：

1. EfficientNet source-consistency 能保源：0.5 把 PTB-XL AUPRC 从 `0.7492` 拉到 `0.7724`。
2. 但它牺牲目标中心：CPSC AUPRC 从 `0.6258` 降到 `0.6004`。
3. `0.1` 只是轻量折中，没有超过 no-anchor 的目标性能，也没有恢复到 PTB-XL source baseline。
4. 这说明 EfficientNet 全模型微调和 source consistency 梯度冲突明显；后续更应该试 backbone-freeze/head-adapter，而不是继续加大 consistency。

EfficientNet1DV2 结构：

```text
initial_conv -> features -> final_conv/final_norm -> classifier
classifier = AdaptiveAvgPool1d -> Flatten -> Dropout -> Linear
```

注意：head-only EfficientNet 不能直接在现有 full-model trainer 中只关掉 `requires_grad`，因为 `model.train()` 会继续更新 frozen BatchNorm running stats。需要单独 runner，让 frozen backbone 始终 `eval()`，只训练 classifier Linear 或 residual adapter。

### 2026-05-23 Residual Adapter 追加实验

为了改善 ECGFounder head AT 的目标中心/源域折中，脚本新增 residual adapter 分类头：

```text
logits = frozen PTB-XL source linear head + zero-init residual adapter
```

CPSC pilot：

| method | checkpoint selection | CPSC target | PTB-XL fold10 |
|---|---|---:|---:|
| linear target-heavy | target AUPRC | 0.9011 / 0.7086 | 0.8998 / 0.7559 |
| residual adapter | target AUPRC | 0.9112 / 0.7385 | 0.8941 / 0.7435 |
| residual adapter, source_weight=2 | target/source hmean | 0.9088 / 0.7290 | 0.9085 / 0.7719 |
| residual adapter, source_weight=3 | target/source hmean | 0.9083 / 0.7278 | 0.9128 / 0.7817 |
| residual adapter, head_l2_anchor=0.25 | target/source hmean | 0.9111 / 0.7365 | 0.8967 / 0.7506 |

判断：

1. residual adapter 是目前 CPSC 目标中心最强路线，AUPRC 从 baseline `0.5725` 提升到 `0.7385`。
2. `source_weight=2 + target/source hmean selection` 是当前更接近双目标的折中：目标 AUPRC `0.7290`，PTB-XL AUPRC `0.7719`。
3. `source_weight=3` 能把 PTB-XL AUPRC 提到 `0.7817`，但目标 AUPRC 回落到 `0.7278`。
4. 这仍没有达到 PTB-XL source AUPRC `0.8016`，所以 active goal 还未完成。
5. 下一步应该围绕 residual adapter 做小网格：`source_weight=3/4`、`head_l2_anchor=0.1/0.25/0.5`、更小 `adapter_hidden` 或 `adapter_scale`。

实现修正：

```text
scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py
```

已修复 `--freeze_base_head` 路径，确保只冻结 base linear head，不冻结 residual adapter。

### Source-logit anchor：当前最有效保源改进

新增参数：

```text
--source_logit_anchor_weight
```

在 PTB-XL source batch 上约束 adapted logits 接近 frozen source head logits。CPSC 结果：

| config | CPSC target | PTB-XL fold10 |
|---|---:|---:|
| residual sw2 hmean，无 logit anchor | 0.9088 / 0.7290 | 0.9085 / 0.7719 |
| residual sw2 hmean，logit anchor 0.1 | 0.9089 / 0.7304 | 0.9190 / 0.7943 |
| residual sw2 hmean，logit anchor 0.2 | 0.9082 / 0.7295 | 0.9204 / 0.7975 |
| residual sw2 hmean，logit anchor 0.5 | 0.9062 / 0.7232 | 0.9216 / 0.8005 |

判断：

1. source-logit anchor 比参数 L2 更有效；0.1/0.2 几乎不牺牲 CPSC target，却显著恢复 PTB-XL AUPRC。
2. `0.5` 已经把 PTB-XL AUPRC 拉回 `0.8005`，基本等于 ECGFounder source baseline `0.8016`。
3. 代价是 CPSC target AUPRC 从最强的 `0.7385` 回落到 `0.7232`，仍比 baseline `0.5725` 高很多。
4. 当前还只在 CPSC 验证；需要扩展到 ningbo/chapman/georgia 才能证明这是稳定主线。

### Four-center source-logit anchor 0.2 / 0.5

`source_logit_anchor_weight=0.5` 已扩展到四个主中心：

| center | baseline target | residual adapter + logit anchor 0.5 | PTB-XL fold10 after AT |
|---|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9210 / 0.5748 | 0.9211 / 0.8000 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9013 / 0.4850 | 0.9215 / 0.7978 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9056 / 0.7218 | 0.9217 / 0.8007 |
| georgia | 0.8525 / 0.6551 | 0.8909 / 0.7418 | 0.9216 / 0.7992 |
| mean | 0.8635 / 0.5058 | 0.9047 / 0.6309 | 0.9214 / 0.7994 |

这轮是目前最接近目标的 ECGFounder 版本：

```text
target-center mean: +4.12pp AUROC / +12.51pp AUPRC
PTB-XL source retained: 0.9214 / 0.7994
ECGFounder source baseline: 0.9224 / 0.8016
```

`source_logit_anchor_weight=0.2` 四中心：

| center | baseline target | residual adapter + logit anchor 0.2 | PTB-XL fold10 after AT |
|---|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9229 / 0.5816 | 0.9201 / 0.7977 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9086 / 0.4862 | 0.9201 / 0.7934 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9082 / 0.7274 | 0.9205 / 0.7980 |
| georgia | 0.8525 / 0.6551 | 0.8929 / 0.7466 | 0.9201 / 0.7959 |
| mean | 0.8635 / 0.5058 | 0.9082 / 0.6354 | 0.9202 / 0.7962 |

Pareto 判断：

```text
0.2: 目标中心更强，target mean 0.9082 / 0.6354，PTB-XL 0.9202 / 0.7962。
0.5: 源域保持更强，target mean 0.9047 / 0.6309，PTB-XL 0.9214 / 0.7994。
```

重要边界：

```text
源域指标已经基本保住，但外部目标中心 AUPRC 仍低于 PTB-XL source AUPRC。
所以不能说 target-center 已达到 PTB-XL 同源性能；
只能说在保住 PTB-XL 源域性能的同时显著提升了目标中心性能。
```

下一步：

```text
1. ECGFounder 0.2/0.5 版本做多 seed 复现。
2. EfficientNet1DV2 若继续追，应优先尝试 backbone-freeze/head-adapter；
   单纯 source-consistency 已验证为保源但压制目标适配。
```
