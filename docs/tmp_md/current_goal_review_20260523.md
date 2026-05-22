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

## 2026-05-23 追加事实核查：ECGFounder 官方策略、all-zero、real-only

### ECGFounder 官方微调策略

本地官方仓库 `model/ecgfounder` 的 `finetune_ECGFounder.ipynb` 默认使用：

```text
linear_prob=False
```

官方代码含义：

```text
linear_prob=True  -> freeze backbone, train linear head
linear_prob=False -> full fine-tuning
```

所以我们现在的 ECGFounder frozen encoder + Super5 linear/residual adapter
不是作者默认最强 fine-tuning 策略，而是更保守、更可解释的 frozen-feature
对比基线。论文叙述时不能把它写成 ECGFounder 官方推荐的最强微调。

### All-zero 评估已补

新增：

```text
scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py
scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py
scripts/paper/reevaluate_ecgfounder_head_run_20260523.py
```

现在 ECGFounder PN2021 view 同时输出：

```text
all-zero-kept main metrics
drop-all-zero sensitivity metrics
```

K=500 VAE-only residual adapter + source-logit anchor 0.2 复评：

| center | all-zero-kept | drop-all-zero |
|---|---:|---:|
| ningbo | 0.9229 / 0.5816 | 0.9338 / 0.6830 |
| chapman_shaoxing | 0.9086 / 0.4862 | 0.9138 / 0.5808 |
| cpsc_2018 | 0.9082 / 0.7274 | 0.9465 / 0.8708 |
| georgia | 0.8929 / 0.7466 | 0.9036 / 0.8125 |

结论：这组 ECGFounder 结果不是 all-zero 样本把 AUROC/AUPRC 虚高；排除 all-zero 后指标更高。

### Real-only adapter 消融

新增参数：

```text
--disable_adv_stream
```

该设置只用：

```text
PTB-XL source features
+ target-center K real ECGFounder features
+ residual adapter
+ source-logit anchor
```

不使用 VAE latent-hull adversarial samples。

K=500 四中心对比：

| center | ECGFounder baseline | real-only adapter | VAE-only adapter |
|---|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9239 / 0.5850 | 0.9229 / 0.5816 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9254 / 0.4860 | 0.9086 / 0.4862 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9059 / 0.7238 | 0.9082 / 0.7274 |
| georgia | 0.8525 / 0.6551 | 0.8928 / 0.7479 | 0.8929 / 0.7466 |
| mean | 0.8635 / 0.5058 | 0.9120 / 0.6357 | 0.9082 / 0.6354 |

这说明 ECGFounder 的大提升主要来自目标中心真实样本 head/adapter adaptation。
VAE-only latent-hull 当前和 real-only 持平，没有形成稳定额外优势。

### K=100 四中心小样本 pilot

| center | baseline | real-only adapter | VAE-only adapter | VAE-only - real-only |
|---|---:|---:|---:|---:|
| ningbo | 0.8848 / 0.4382 | 0.9066 / 0.5197 | 0.9050 / 0.5146 | -0.16pp / -0.51pp |
| chapman_shaoxing | 0.8913 / 0.3724 | 0.9200 / 0.4471 | 0.9171 / 0.4442 | -0.29pp / -0.28pp |
| cpsc_2018 | 0.8247 / 0.5819 | 0.8755 / 0.6540 | 0.8748 / 0.6541 | -0.07pp / +0.01pp |
| georgia | 0.8536 / 0.6593 | 0.8745 / 0.7086 | 0.8700 / 0.6984 | -0.45pp / -1.02pp |
| mean | - | 0.8941 / 0.5824 | 0.8917 / 0.5779 | -0.24pp / -0.45pp |

K=100 下，VAE-only 也没有稳定优于 real-only adapter。CPSC 的 AUPRC 基本打平，
其他三个中心都是 real-only 更强。

drop-all-zero：

| center | real-only drop-all-zero | VAE-only drop-all-zero |
|---|---:|---:|
| ningbo | 0.9182 / 0.6547 | 0.9158 / 0.6479 |
| chapman_shaoxing | 0.9333 / 0.5709 | 0.9287 / 0.5708 |
| cpsc_2018 | 0.9186 / 0.8095 | 0.9187 / 0.8087 |
| georgia | 0.8869 / 0.7833 | 0.8815 / 0.7735 |

这轮结果把结论进一步收紧：

```text
ECGFounder 上的主要性能来源是 target-center real adapter。
VAE-only latent-hull 目前只是一个可匹配 real-only 的 on-manifold adversarial variant，
并没有稳定额外增益。
```

### 当前目标状态

已做到：

```text
ECGFounder 源域 PTB-XL 性能基本保持：约 0.920-0.922 / 0.796-0.802
目标中心 AUROC 显著提高
drop-all-zero 后指标不崩，且通常更高
```

尚未做到：

```text
外部目标中心 all-zero-kept AUPRC 达到 PTB-XL source AUPRC 0.8016
证明 VAE-only latent-hull 明显优于 real-only target adapter
EfficientNet1DV2 达到 ECGFounder 这一级别的源域/目标域双保持
```

更合理的下一步：

```text
1. 如果论文必须保留 VAE-only 主张：做 K=20/50 多 seed，验证极小样本下是否优于 real-only。
2. 如果追求最强性能：ECGFounder 应主打 real-only residual adapter + source-logit anchor。
3. EfficientNet1DV2 不应继续 full-model target-heavy 微调；应实现 frozen backbone + residual classifier adapter，
   否则 source consistency 与目标适配继续冲突。
```

## 2026-05-23 追加 K=50 小样本审计

目标：继续寻找 VAE-only 在线对抗训练相对 real-only adapter 的独特优势。

K=50 四中心结果：

| center | baseline | real-only adapter | VAE-only adapter | VAE-only - real-only |
|---|---:|---:|---:|---:|
| ningbo | 0.8847 / 0.4388 | 0.9055 / 0.4948 | 0.9008 / 0.4848 | -0.46pp / -1.00pp |
| chapman_shaoxing | 0.8871 / 0.3823 | 0.9160 / 0.4458 | 0.9077 / 0.4488 | -0.83pp / +0.30pp |
| cpsc_2018 | 0.8252 / 0.5834 | 0.8406 / 0.6061 | 0.8462 / 0.6146 | +0.56pp / +0.84pp |
| georgia | 0.8538 / 0.6602 | 0.8679 / 0.6925 | 0.8628 / 0.6786 | -0.51pp / -1.39pp |
| mean | - | 0.8825 / 0.5598 | 0.8794 / 0.5567 | -0.31pp / -0.31pp |

drop-all-zero 后：

| center | real-only | VAE-only |
|---|---:|---:|
| ningbo | 0.9170 / 0.6447 | 0.9140 / 0.6407 |
| chapman_shaoxing | 0.9262 / 0.5754 | 0.9178 / 0.5822 |
| cpsc_2018 | 0.8863 / 0.7480 | 0.8901 / 0.7571 |
| georgia | 0.8790 / 0.7675 | 0.8731 / 0.7560 |

唯一正向窗口是 `cpsc_2018`。进一步 refinement：

| variant | CPSC target |
|---|---:|
| standard primary-label, lambda=0.15 | 0.8462 / 0.6146 |
| high-adv / low-target-real | 0.8426 / 0.6021 |
| exact multi-hot candidate | 0.8440 / 0.6092 |
| lambda=0.35 | 0.8439 / 0.6088 |
| STTC-only latent-hull | 0.8452 / 0.6124 |
| CD+STTC latent-hull | 0.8448 / 0.6039 |

per-class：

| method | CD AUPRC | NORM AUPRC | STTC AUPRC |
|---|---:|---:|---:|
| real-only | 0.9237 | 0.6634 | 0.2313 |
| standard VAE-only | 0.9264 | 0.6623 | 0.2551 |
| STTC-only VAE | 0.9241 | 0.6586 | 0.2544 |
| CD+STTC VAE | 0.9244 | 0.6306 | 0.2567 |

判断：

```text
高 adv 权重、exact-label candidate、更大 lambda、STTC-only、CD+STTC 都没有改进 CPSC。
因此当前最好的 VAE-only 设置仍是 standard primary-label lambda=0.15。
但是它只在 CPSC K=50 小幅强于 real-only，不能支撑“VAE-only 明显强于其他方法”。
```

下一步只值得做一种更聚焦的检查：

```text
ECGFounder head-adapter 路线继续扫 VAE-only 的收益很低；
下一个有价值方向是 EfficientNet1DV2 frozen backbone + residual classifier adapter，
看 VAE latent-hull 是否能在非 foundation frozen representation 上形成独立贡献。
```

## 2026-05-23 追加：EfficientNet1DV2 frozen-backbone adapter pilot

为了避免全模型微调导致源域遗忘，`synth_online_at_super5.py` 新增：

```text
--freeze_backbone_classifier_only
--classifier_only_train_final_norm
--disable_adv_stream
```

协议：

```text
head:
  冻结 EfficientNet1DV2 卷积主干，只训练 classifier。

head+final_norm:
  额外训练 final_norm affine 参数，但 BatchNorm running statistics 保持 eval/frozen。

real-only:
  使用 --disable_adv_stream，只用 PTB-XL source + K 个目标中心真实样本更新 adapter。

VAE-only:
  同样 adapter，同时把目标中心真实 ECG 的 ECGTwin VAE latent-hull 在线对抗样本加入训练流。
```

CPSC K=50/100 pilot，目标中心 ref ids 已从评估集排除：

| setting | method | PTB-XL fold10 | CPSC all-zero-kept | 7-center avg |
|---|---|---:|---:|---:|
| K=50 head | real-only | 0.9083 / 0.7771 | 0.8145 / 0.5702 | 0.7800 / 0.4629 |
| K=50 head | VAE-only | 0.9085 / 0.7777 | 0.8168 / 0.5713 | 0.7809 / 0.4625 |
| K=100 head | real-only | 0.9084 / 0.7775 | 0.8154 / 0.5694 | 0.7805 / 0.4622 |
| K=100 head | VAE-only | 0.9083 / 0.7773 | 0.8193 / 0.5716 | 0.7812 / 0.4621 |
| K=100 head+final_norm | real-only | 0.9084 / 0.7774 | 0.8155 / 0.5695 | 0.7803 / 0.4620 |
| K=100 head+final_norm | VAE-only | 0.9082 / 0.7770 | 0.8202 / 0.5721 | 0.7811 / 0.4618 |

K=100 drop-all-zero sensitivity：

| setting | method | CPSC drop-all-zero | 7-center drop-all-zero avg |
|---|---|---:|---:|
| head | real-only | 0.8654 / 0.7004 | 0.8000 / 0.5704 |
| head | VAE-only | 0.8696 / 0.7056 | 0.8005 / 0.5712 |
| head+final_norm | real-only | 0.8655 / 0.7006 | 0.7997 / 0.5703 |
| head+final_norm | VAE-only | 0.8707 / 0.7072 | 0.8007 / 0.5712 |

判断：

```text
1. 这条 EfficientNet adapter 路线可以保持 PTB-XL 源域性能：
   fold10 约 0.908 / 0.777，基本等于原始 source baseline。
2. VAE-only 相对 real-only 有稳定但很小的 CPSC 正增益。
   K=100 head+final_norm 下，all-zero-kept 约 +0.47pp / +0.26pp；
   drop-all-zero 后约 +0.52pp / +0.66pp。
3. 这说明小幅收益不是 all-zero 样本虚高，但还远不到“明显强于其他方法”。
4. head+final_norm 比纯 head 略好，是目前更值得继续扩展到多中心/K sweep 的 EfficientNet 适配协议。
5. 目标中心 CPSC AUPRC 仍远低于 PTB-XL source AUPRC 0.777；当前目标“外部中心达到 PTB-XL 同源水平”未完成。
```

下一步：

```text
1. 不再优先提高 lambda/hull_steps；lambda=0.35,hull_steps=20 提高 ASR，但 quick AUPRC 未超过 lambda=0.15。
2. 用 head+final_norm + lambda=0.15 + M=20 + K=100 扩到 ningbo/chapman/georgia。
3. 如果多中心仍只有 CPSC 正向，VAE-only 主张应收窄为“特定中心/类别结构有效”，而不是全局方法优势。
```

### K=100 四中心补跑结果

head+final_norm adapter 已扩展到 `ningbo`、`chapman_shaoxing`、`cpsc_2018`、`georgia`。
所有 run 都排除了目标中心 K=100 ref ids，训练协议保持一致：

```text
EfficientNet1DV2 source checkpoint
freeze backbone
train classifier + final_norm affine
source_logit_anchor_weight = 0.2
real-only: PTB-XL source + target real K
VAE-only: real-only + target real latent-hull online adversarial samples
```

主评估 all-zero-kept：

| center | real-only adapter | VAE-only adapter | VAE-only - real-only |
|---|---:|---:|---:|
| ningbo | 0.8741 / 0.4310 | 0.8723 / 0.4285 | -0.18pp / -0.25pp |
| chapman_shaoxing | 0.8810 / 0.3588 | 0.8822 / 0.3609 | +0.12pp / +0.21pp |
| cpsc_2018 | 0.8155 / 0.5695 | 0.8202 / 0.5721 | +0.47pp / +0.26pp |
| georgia | 0.8196 / 0.5958 | 0.8195 / 0.5958 | -0.01pp / +0.00pp |
| 4-center mean | 0.8476 / 0.4888 | 0.8486 / 0.4893 | +0.10pp / +0.06pp |

drop-all-zero sensitivity：

| center | real-only adapter | VAE-only adapter | VAE-only - real-only |
|---|---:|---:|---:|
| ningbo | 0.9016 / 0.6130 | 0.9001 / 0.6112 | -0.15pp / -0.18pp |
| chapman_shaoxing | 0.9074 / 0.5437 | 0.9079 / 0.5444 | +0.05pp / +0.07pp |
| cpsc_2018 | 0.8655 / 0.7006 | 0.8707 / 0.7072 | +0.52pp / +0.66pp |
| georgia | 0.8276 / 0.6764 | 0.8274 / 0.6764 | -0.02pp / +0.00pp |
| 4-center mean | 0.8740 / 0.6334 | 0.8765 / 0.6348 | +0.25pp / +0.14pp |

PTB-XL fold10 保持在 `0.9074-0.9084 / 0.7749-0.7774`，说明 frozen-backbone
adapter 的源域保持是有效的。

当前结论进一步收紧：

```text
EfficientNet frozen-backbone adapter 能保住 PTB-XL source。
VAE-only latent-hull 在 CPSC 最有用，在 Chapman 有轻微正向，在 Ningbo/Georgia 没有优势。
四中心均值只有约 +0.10pp / +0.06pp，不能作为强主张。
如果继续论文主线，应把 VAE-only 定位为 on-manifold adversarial regularization，
而不是稳定超越 real-only target adaptation 的独立增强模块。
```

## 2026-05-23 追加：ECGFounder 纯 VAE-only 独立性测试

为了区分“目标中心真实样本 adapter 适配”和“VAE-only latent-hull 独立贡献”，
给 ECGFounder runner 新增：

```text
--init_head_path
```

这个参数允许第二阶段从已有 head 初始化，例如：

```text
real-only residual adapter
-> source + VAE-only adversarial stream refinement
```

### Stage-2 after real-only

先在 `cpsc_2018` 测试：

```text
init = K100 real-only residual adapter best_head.pt
target_real_weight = 0
adv_weight = 40
source_weight = 2
source_logit_anchor_weight = 0.2
epochs = 15
```

结果：

| setting | CPSC target | CPSC drop-all-zero | PTB-XL fold10 |
|---|---:|---:|---:|
| real-only init | 0.8755 / 0.6540 | 0.9186 / 0.8095 | 0.9224 / 0.8019 |
| best after VAE-only stage-2 | 0.8755 / 0.6540 | 0.9186 / 0.8095 | 0.9224 / 0.8019 |

训练过程里 VAE-only stage-2 epoch 1-15 的 target AUPRC 下降到约 `0.636-0.646`，
selection 保留 epoch 0。说明在 real-only adapter 已经适配后，继续只加 VAE
latent-hull adversarial stream 没有独立增量。

### Pure VAE-only without target-real stream

再测试一个更直接的问题：从 ECGFounder source head 出发，不喂目标中心真实特征，
只用 PTB-XL source + K=100 目标中心 VAE latent-hull adversarial samples。

配置：

```text
K = 100
k_anchor = 100 per epoch
target_real_weight = 0
adv_weight = 40
source_weight = 2
source_logit_anchor_weight = 0.2
head_type = residual_adapter
freeze_base_head = true
```

结果：

| center | source baseline | pure VAE-only | delta |
|---|---:|---:|---:|
| ningbo | 0.8848 / 0.4382 | 0.8845 / 0.4401 | -0.03pp / +0.19pp |
| chapman_shaoxing | 0.8913 / 0.3724 | 0.8933 / 0.3773 | +0.20pp / +0.49pp |
| cpsc_2018 | 0.8247 / 0.5819 | 0.8317 / 0.5879 | +0.70pp / +0.60pp |
| georgia | 0.8536 / 0.6593 | 0.8559 / 0.6614 | +0.23pp / +0.22pp |
| 4-center mean | 0.8636 / 0.5129 | 0.8663 / 0.5167 | +0.27pp / +0.38pp |

drop-all-zero：

| view | baseline mean | pure VAE-only mean | delta |
|---|---:|---:|---:|
| drop-all-zero | 0.8920 / 0.6569 | 0.8950 / 0.6628 | +0.30pp / +0.59pp |

PTB-XL fold10 after pure VAE-only 平均约 `0.9229 / 0.8031`，源域保持很好。

解释：

```text
Pure VAE-only 有独立适配能力，但幅度很小。
它不能解释 ECGFounder residual adapter 的大幅提升；
大提升主要来自 target-real adapter adaptation。
这条证据不满足“VAE-only 明显强于其他方法”的目标。
```

下一步不建议继续把 ECGFounder residual adapter 作为证明 VAE-only 独特优势的主路线。
如果继续追 VAE-only，优先方向应转向：

```text
1. 低 K/稀有类场景，只把 VAE-only 定位为 target-real 不足时的弱正则。
2. EfficientNet frozen-backbone adapter 或更轻量 residual classifier adapter，
   因为非 foundation frozen features 更可能暴露 VAE latent-hull 的独立价值。
3. 明确报告 negative evidence：real-only adapter 是当前主增益来源。
```

## 2026-05-23 追加：EfficientNet pure VAE-only K=100 四中心

ECGFounder pure VAE-only 只有小幅独立增益后，转到 EfficientNet1DV2
frozen-backbone adapter，测试同样的纯 VAE-only 设定：

```text
source checkpoint = super5_minresample_full10_perglobal_20260503
adapter           = classifier + final_norm affine
target_real_weight = 0
adv_weight         = 20
source_logit_anchor_weight = 0.2
K = 100
M = 20
lambda = 0.15
hull_steps = 10
quality gate = disabled
```

即只用：

```text
PTB-XL source stream + target-center real ECGTwin VAE latent-hull adversarial stream
```

不使用目标中心真实 ECG 的监督 stream。

四中心同 ref-exclusion baseline 对比：

| center | source baseline | pure VAE-only | delta |
|---|---:|---:|---:|
| ningbo | 0.8707 / 0.4266 | 0.8736 / 0.4296 | +0.29pp / +0.31pp |
| chapman_shaoxing | 0.8770 / 0.3567 | 0.8820 / 0.3606 | +0.50pp / +0.39pp |
| cpsc_2018 | 0.8143 / 0.5687 | 0.8170 / 0.5705 | +0.27pp / +0.18pp |
| georgia | 0.8173 / 0.5944 | 0.8193 / 0.5952 | +0.20pp / +0.08pp |
| 4-center mean | 0.8448 / 0.4866 | 0.8480 / 0.4890 | +0.32pp / +0.24pp |

drop-all-zero：

| center | source baseline | pure VAE-only | delta |
|---|---:|---:|---:|
| ningbo | 0.8987 / 0.6107 | 0.9010 / 0.6122 | +0.23pp / +0.15pp |
| chapman_shaoxing | 0.9046 / 0.5429 | 0.9080 / 0.5443 | +0.34pp / +0.13pp |
| cpsc_2018 | 0.8631 / 0.6979 | 0.8665 / 0.7021 | +0.34pp / +0.42pp |
| georgia | 0.8258 / 0.6757 | 0.8271 / 0.6757 | +0.13pp / -0.00pp |
| 4-center mean | 0.8730 / 0.6318 | 0.8756 / 0.6336 | +0.26pp / +0.18pp |

PTB-XL fold10 after pure VAE-only 约 `0.908 / 0.776`，源域保持良好。

判断：

```text
EfficientNet pure VAE-only 比 ECGFounder pure VAE-only 更一致：四中心都是弱正向。
但提升幅度仍非常小，离“明显强于其他方法”很远。
drop-all-zero 后仍是弱正向，说明不是 all-zero 虚高，但也没有达到 PTB-XL 源域性能。
```

下一步最合理的技术路线：

```text
EfficientNet frozen backbone
-> residual classifier adapter, preferably small MLP or low-rank residual head
-> source-logit anchor keeps PTB-XL
-> compare pure VAE-only / real-only / real+VAE
```

原因：

```text
当前 classifier+final_norm affine 容量太低；
full-model target-heavy 会损伤 source；
residual classifier adapter 是两者之间最干净的折中。
```

### Low-rank residual classifier adapter pilot

实现了一个可折叠 low-rank residual head：

```text
logits = source_linear(x) + alpha / rank * up(down(x))
```

训练时只更新 `down/up` 和可选 `final_norm` affine；保存 checkpoint 时把
`up @ down` 折叠回原 EfficientNet `classifier` 的最后 Linear，保证
`eval_crosscenter.py` 仍能按 vanilla EfficientNet 加载。

新增参数：

```text
--classifier_adapter_type lora
--classifier_lora_rank
--classifier_lora_alpha
```

CPSC K=100 pure VAE-only 结果：

| adapter | lr | CPSC target | CPSC drop-all-zero | PTB-XL fold10 |
|---|---:|---:|---:|---:|
| source baseline | - | 0.8143 / 0.5687 | 0.8631 / 0.6979 | 0.9072 / 0.7744 |
| linear head+final_norm | 5e-5 | 0.8170 / 0.5705 | 0.8665 / 0.7021 | 0.9082 / 0.7770 |
| low-rank r16 | 5e-5 | 0.8142 / 0.5686 | 0.8629 / 0.6977 | 0.9072 / 0.7746 |
| low-rank r16 | 5e-4 | 0.8143 / 0.5687 | 0.8630 / 0.6978 | 0.9073 / 0.7747 |

结论：

```text
低秩 residual head 没有改善 pure VAE-only，反而低于简单线性 head。
这说明当前瓶颈不只是 classifier adapter 容量不足；
VAE latent-hull adversarial stream 本身提供的监督信号太弱，无法替代目标真实样本监督。
```

下一步不建议继续扫 `rank/alpha/lr`。如果还要推进 VAE-only 独立性，应换问题设定：

```text
1. VAE-only 作为 real-only adapter 的稳定性/校准正则，而不是主增益来源；
2. 或者改进 latent-hull 标签/候选构造，让 adversarial stream 的监督信号更接近真实 target labels；
3. 继续保留 negative evidence：当前大幅提升主要来自 target-real adapter。
```

### Soft-label regularizer and K=20 check

CPSC K=100 real+VAE 对照：

| setting | CPSC target | CPSC drop-all-zero | PTB-XL fold10 |
|---|---:|---:|---:|
| real-only K100 | 0.8155 / 0.5695 | 0.8655 / 0.7006 | 0.9072 / 0.7744 |
| hard multi-hot VAE | 0.8202 / 0.5721 | 0.8707 / 0.7072 | 0.9082 / 0.7770 |
| teacher-soft VAE | 0.8161 / 0.5699 | 0.8660 / 0.7012 | 0.9082 / 0.7771 |
| mixed-soft VAE | 0.8198 / 0.5718 | 0.8702 / 0.7064 | 0.9082 / 0.7770 |

结论：

```text
teacher-soft 太保守；
mixed-soft 接近 hard multi-hot，但仍略弱；
当前 CPSC K=100 最好的版本仍是 hard multi-hot VAE。
```

CPSC K=20 极小目标样本检查：

| setting | CPSC target | CPSC drop-all-zero | PTB-XL fold10 |
|---|---:|---:|---:|
| source / real-only best | 0.8153 / 0.5707 | 0.8636 / 0.6988 | 0.9072 / 0.7744 |
| VAE hard multi-hot | 0.8153 / 0.5707 | 0.8636 / 0.6988 | 0.9072 / 0.7744 |

备注：

```text
K=20 时，本轮 real-only 和 VAE 分支的 best checkpoint 都停在 epoch0。
当前协议下，VAE-only 没有补足极小样本 target supervision 的不足。
```

### EfficientNet last-block adaptation pilot

为了验证“只改分类头太弱”这个假设，新增了 EfficientNet last-block adaptation：

```text
trainable = classifier + final_conv + final_norm affine + last 4 feature blocks
frozen    = earlier backbone blocks and BatchNorm running statistics
lr        = 5e-6
K         = 100
center    = cpsc_2018
```

结果：

| setting | target real weight | VAE adv weight | CPSC target | CPSC drop-all-zero | PTB-XL fold10 |
|---|---:|---:|---:|---:|---:|
| real-only last4 | 40 | 0 | 0.8143 / 0.5688 | 0.8634 / 0.6982 | 0.9077 / 0.7759 |
| real+VAE last4 | 40 | 20 | 0.8163 / 0.5700 | 0.8657 / 0.7010 | 0.9078 / 0.7760 |
| pure VAE last4 | 0 | 20 | 0.8147 / 0.5691 | 0.8636 / 0.6986 | 0.9077 / 0.7757 |

判断：

```text
轻微解冻不会破坏 PTB-XL source performance；
VAE stream 仍只提供弱增益；
pure VAE last4 没有明显超过 source/real-only；
当前瓶颈不是“只训练分类头太弱”这么简单。
```

### ECGFounder official fine-tuning check

本地 ECGFounder 官方仓库证据：

```text
model/ecgfounder/finetune_ECGFounder.ipynb:
  ft_12lead_ECGFounder(..., linear_prob=False)
  "linear classificaion -> linear_prob=True"
  "full fine-tuning -> linear_prob=False"

model/ecgfounder/README.md:
  fine-tuning/validation must strictly follow dataset.py preprocessing.
```

因此当前 ECGFounder frozen-encoder 5-class head/adapter 实验是保守
linear-probe 设定，不是作者推荐的最强 fine-tuning 设定。ECGFounder
较大 target-real gain 不能直接归因于 VAE-only；需要用官方 full fine-tuning
作为更公平的 ECGFounder 上限对照。

### ECGFounder official-style full fine-tuning control

新增脚本：

```text
scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py
```

目的：确认 ECGFounder 在作者推荐 full fine-tuning 设置下，本身能达到什么水平。
该脚本不使用 ECGTwin VAE 对抗样本，只使用 PTB-XL source 和可选的目标中心 K 条真实 ECG。

实现依据：

```text
ft_12lead_ECGFounder(..., linear_prob=False)
lr=1e-4, weight_decay=1e-5, epochs=5
preprocess=official_ptbxl_eval, input=(12, 5000)
fold9 macro AUPRC 选择 best checkpoint
```

CPSC K=100 ref-excluded 结果：

| setting | PTB-XL fold10 | CPSC target | CPSC drop-all-zero |
|---|---:|---:|---:|
| source-only full FT | 0.9300 / 0.8229 | 0.8187 / 0.5878 | 0.8696 / 0.7109 |
| source + K100 target-real full FT | 0.9244 / 0.8134 | 0.8684 / 0.6895 | 0.9154 / 0.8198 |

备注：target-real run 按 fold9 选择的是 epoch 3；如果只看目标中心最佳 epoch 4，
CPSC target 为 `0.8723 / 0.7025`，drop-all-zero 为 `0.9157 / 0.8253`。

结论：

```text
ECGFounder 官方 full fine-tuning + K=100 target-real，在不使用 VAE 的情况下已经显著强于当前 VAE-only head/adapter 尝试。
因此 ECGFounder 上的大幅 target gain 不能归因于 VAE-only 本身；后续必须把官方 full fine-tuning 作为公平强基线。
VAE-only 若要成为论文主贡献，需要证明它能在这个 full fine-tuning 基线上继续带来增益，或在更小 K 下稳定胜出。
```

### Full fine-tuning + VAE-only online AT pilot

脚本 `scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py` 已扩展：

```text
--enable_vae_adv_stream
```

训练协议：

```text
ECGFounder official full fine-tuning
PTB-XL source stream + K=100 target-real stream
每个 epoch 用当前 full model 作为 victim
从同一批 K=100 目标中心真实 ECGTwin VAE latent 做 same-label latent-hull online AT
VAE adv ECG: ECGTwin latent -> 12 x 1000 -> interpolate to 12 x 5000
```

CPSC K=100 结果：

| setting | PTB-XL fold10 | CPSC target | CPSC drop-all-zero |
|---|---:|---:|---:|
| source-only full FT | 0.9300 / 0.8229 | 0.8187 / 0.5878 | 0.8696 / 0.7109 |
| source + K100 target-real full FT | 0.9244 / 0.8134 | 0.8684 / 0.6895 | 0.9154 / 0.8198 |
| source + K100 target-real + VAE-only online AT | 0.9247 / 0.8157 | 0.8754 / 0.7148 | 0.9168 / 0.8261 |

VAE run 关键配置：

```text
k_anchor=100 requested, actually generated 84/epoch because K=100 pool only has CD/NORM/STTC positives
M=20, lambda=0.15, hull_steps=3, hull_lr=0.25
adv_weight=20, target_real_weight=40
best checkpoint selected by PTB-XL fold9 macro AUPRC
```

阶段性判断：

```text
这是目前最重要的正向证据：VAE-only 在 ECGFounder 官方 full fine-tuning 强基线上仍有增益。
CPSC target AUPRC 比 no-VAE full FT 高 +2.53pp；drop-all-zero AUPRC 高 +0.62pp。
因此该增益不是只靠 all-zero 样本虚高，也不是仅由 frozen-head 弱 baseline 造成。
但目前只完成 CPSC K=100 单中心 pilot，还不能宣称全中心稳定达到 PTB-XL 源域水平。
下一步需要在 ningbo、chapman_shaoxing、georgia 复跑，并做 K=20/50/100 sensitivity。
```

### Full-FT + VAE-only two-center update

已补 `ningbo` K=100 同协议复现。当前结果均为 PTB-XL fold9 macro AUPRC 选择 best checkpoint。

| center | setting | PTB-XL fold10 | target | drop-all-zero |
|---|---|---:|---:|---:|
| cpsc_2018 | no-VAE full FT | 0.9244 / 0.8134 | 0.8684 / 0.6895 | 0.9154 / 0.8198 |
| cpsc_2018 | full FT + VAE-only online AT | 0.9247 / 0.8157 | 0.8754 / 0.7148 | 0.9168 / 0.8261 |
| ningbo | no-VAE full FT | 0.9292 / 0.8223 | 0.8919 / 0.4942 | 0.9087 / 0.6429 |
| ningbo | full FT + VAE-only online AT | 0.9271 / 0.8188 | 0.8999 / 0.5162 | 0.9150 / 0.6509 |

增量：

| center | target delta | drop-all-zero delta | source delta |
|---|---:|---:|---:|
| cpsc_2018 | +0.70pp / +2.53pp | +0.15pp / +0.62pp | +0.03pp / +0.23pp |
| ningbo | +0.80pp / +2.20pp | +0.63pp / +0.80pp | -0.21pp / -0.35pp |

关键边界：

```text
Ningbo no-VAE 的 epoch1 target AUPRC 曾到 0.5317，超过 VAE source-selected checkpoint 的 0.5162。
因此 Ningbo 不能解释成 VAE 在 target-oracle 下无条件更强；
更准确的说法是：在不使用目标测试集选 checkpoint、只按 PTB-XL fold9 选择时，VAE stream 提高了 target 指标。
后续必须同时报告 source-selected 和 target-oracle 曲线，并设计不泄漏的 target-val selection。
```
