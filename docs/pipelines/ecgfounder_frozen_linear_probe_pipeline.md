# ECGFounder Frozen Encoder Super5 Linear Probe Pipeline

本文档规划一个更公平、可解释的 ECGFounder 对比基线：

```text
冻结 ECGFounder encoder
-> PTB-XL fold 1-8 训练 5 类 Super5 线性头
-> PTB-XL fold 9 做模型选择、阈值选择和校准
-> PTB-XL fold 10 与 PN2021 外部中心评估
```

这个方案替代“ECGFounder 150-class head 通过 keyword/max pooling 映射到 Super5”的弱基线。旧映射只能作为辅助零样本参考，主对比应优先使用本线性探针方案。

## 目标

1. 使用 ECGFounder 官方 encoder 作为冻结特征提取器，不更新 encoder 权重。
2. 使用 PTB-XL 官方 `diagnostic_class` Super5 标签训练 `Linear(1024, 5)` 多标签分类头。
3. 使用 fold 9 做验证集，只允许在 fold 9 上选择 best epoch、阈值和校准参数。
4. 在 fold 10 报告 PTB-XL 内部分布测试性能。
5. 在 PN2021 外部中心报告跨中心 AUROC/AUPRC，并使用当前 v5 Super5 标签映射。
6. 与 EfficientNet1DV2 baseline、VAE-only 在线对抗训练、ECGFounder zero-shot strict mapping 做表格对比。

## 数据划分

### PTB-XL

| split | 用途 |
|---|---|
| fold 1-8 | 训练线性头 |
| fold 9 | best epoch、per-class threshold、校准参数 |
| fold 10 | 最终 PTB-XL test，只评估不调参 |

标签来源：

```text
scripts/triple_labels/label_schemes.py
CLASS_NAMES_SUPER5 = CD, HYP, MI, NORM, STTC
```

PTB-XL Super5 使用官方 `scp_statements.csv` 里的 `diagnostic_class` 聚合规则。PTB-XL 中没有 diagnostic superclass 的 all-zero 样本默认保留，作为“5 个 diagnostic superclass 均不阳性”的负样本；如果过滤 all-zero，必须作为 sensitivity analysis 单独报告。

### PN2021

使用当前严格标签映射：

```text
SUPER5_PN2021_MAPPING_VERSION = v5_super5_strict_voltage_pacing_suppress_20260522
mapping hash = 1141e0a9f94b
```

主评估中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

完整外部评估仍保留 7-center 输出：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

必须 hard-exclude `ptb-xl/ptbxl` shards，避免 PTB-XL 训练数据泄漏。

为了和 K=500 target-anchor 方法公平对齐，PN2021 target-center view 默认排除对应中心被抽作 K=500 anchor 的 record ids。虽然 ECGFounder 线性探针本身没有用这些目标中心样本训练，但比较表应使用同一个评估 denominator。

## 信号输入

本方案使用 ECGFounder encoder，因此输入预处理应匹配 ECGFounder 官方/复现代码，而不是 EfficientNet1DV2 的 `minimal_resample` 主线。

当前已有脚本入口：

```text
scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py
```

主线 `--preprocess_policy official_ptbxl_eval` 使用：

```text
wfdb 读取原始 12-lead ECG
lead reorder 到 ECGFounder EXPECTED_LEADS
截取前 10 秒
不足 10 秒则 zero padding
resample 到 ECGFounder TARGET_POINTS = 5000
per-sample global z-score
Net1D(return_features=True) 提取 1024-d feature
```

该主线对齐 ECGFounder 官方 `ptbxl_eval.py` 的 PTB-XL evaluation style：使用
PTB-XL `records500/filename_hr`，输入 `12 x 5000`，不额外加
notch/bandpass/median baseline removal。

消融 `--preprocess_policy filtered_dataset` 额外加入 ECGFounder `util.py` 中的
50Hz notch、0.67-40Hz bandpass 和 0.4 秒 median baseline removal，用于回应
README 对 filtering 的要求。

后续报告中应写清楚：这是 ECGFounder baseline 的模型专属输入协议；不要把它和 EfficientNet1DV2 的 `minimal_resample + per_sample_global + 100Hz + 1000 samples` 混成同一个预处理。

## 模型

### 冻结 encoder

```text
checkpoint: /root/autodl-tmp/ecgfounder/checkpoint/12_lead_ECGFounder.pth
encoder: ECGFounder Net1D
feature dim: 1024
requires_grad: False for all encoder parameters
```

需要在输出 metadata 中记录：

```text
checkpoint path
checkpoint sha256 if practical
encoder parameter count
trainable parameter count
feature dimension
preprocess config
label mapping version/hash
```

### 线性头

```text
head = Linear(1024, 5)
loss = masked BCEWithLogitsLoss
pos_weight = n_negative / n_positive, clip max = 50
optimizer = AdamW
model selection = fold 9 macro AUPRC
```

默认先跑 3 个 seed：

```text
seed = 42, 2025, 3407
epochs = 80
lr = 1e-3
weight_decay = 1e-4
head_batch_size = 1024
```

如果 3 个 seed 的 fold 9 AUPRC 方差很小，论文主表可报告 seed 42，附录报告 mean/std；如果方差明显，则主表应报告 mean/std。

## Fold 9 阈值与校准

AUROC/AUPRC 使用 sigmoid 概率连续分数，不依赖阈值。

fold 9 阈值/校准只用于二值化指标和模型可解释性，不能影响 fold 10/PN2021 的 AUROC/AUPRC 计算。

需要补充到脚本的输出：

1. `threshold_0.5`: 固定 0.5 阈值。
2. `threshold_f1_per_class`: 每类在 fold 9 上最大化 F1 的阈值。
3. `threshold_youden_per_class`: 每类在 fold 9 上最大化 `TPR - FPR` 的阈值。
4. `temperature_per_class` 或 `platt_per_class`: 可选校准参数，只在 fold 9 拟合。
5. fold 10 和 PN2021 上同时保存 raw sigmoid scores 与 calibrated scores。

主论文表仍以 raw sigmoid 的 macro AUROC/AUPRC 为主，避免阈值选择影响跨中心结论。

## 评估输出

每个 seed 至少保存：

```text
run_config.json
ptbxl_ecgfounder_features.npz
pn2021_ecgfounder_features.npz
best_head.pt
head_training_log.json
train_result.json
calibration_fold9.json
ptbxl_fold10_metrics.json
pn2021_ref_excluded_metrics.json
summary.csv
```

PTB-XL fold 10 输出：

```text
macro AUROC
macro AUPRC
per-class AUROC/AUPRC
thresholded F1/sensitivity/specificity as secondary metrics
```

PN2021 输出：

```text
per-center macro AUROC/AUPRC
4-center average
7-center average
target-view ref exclusion count
all-zero-kept main metrics
drop-all-zero sensitivity metrics
```

all-zero-kept 是主结果，因为真实外部中心中会出现不属于 Super5 或证据不足的样本；drop-all-zero 只作为标签映射敏感性分析。

## 需要先补的代码

当前脚本已经支持：

```text
冻结 ECGFounder encoder
提取 1024-d feature cache
fold 1-8 训练 linear head
fold 9 best epoch selection
fold 10 test
PN2021 ref-excluded views
```

2026-05-23 review 后已补：

1. 固定并记录随机种子。
2. 输出 ECGFounder checkpoint hash、标签映射 version/hash。
3. `--feature_cache_dir`：多 seed 线性头实验可复用 PTB-XL/PN2021 ECGFounder feature cache，避免每个 seed 重新读取 WFDB。
4. `run_ecgfounder_kshot_head_ft_20260517.py` 默认指向 2026-05-22 v5 线性探针目录，并按 `preprocess_policy` 查找 feature cache。
5. `run_ecgfounder_vae_only_lhat_head_ft_20260523.py`：新增 ECGFounder 专用 VAE-only latent-hull 在线对抗训练 runner，冻结 ECGFounder encoder，只更新 Super5 线性头。

仍需要补：

1. feature cache 内写入 mapping/preprocess metadata，并在读取旧 cache 时校验。
2. fold 9 per-class threshold selection。
3. 可选 per-class calibration。
4. PN2021 drop-all-zero sensitivity view。
5. per-class PN2021 metrics csv，方便定位 CD/HYP/MI/NORM/STTC 哪类拉低。
6. 统一生成和 EfficientNet/VAE-only 对比的 markdown/html report。

## 2026-05-23 Review 结论

当前 ECGFounder 线性头策略本身是合理强基线：

```text
ECGFounder frozen encoder
-> PTB-XL fold 1-8 训练 Linear(1024, 5)
-> fold 9 按 macro AUPRC 选 best epoch
-> fold 10 和 PN2021 v5 ref-excluded 评估
```

需要注意的边界：

1. 它不是“ECGFounder 官方 Super5 头”，而是我们在官方 encoder 上重新训练的 Super5 线性探针。
2. 它的输入协议是 ECGFounder 专属的 `12 x 5000`，不应写成和 EfficientNet1DV2 完全相同。
3. seed42/2025/3407 已完成，线性头随机性很小：

| seed | PTB-XL fold10 | PN2021 4-center target mean |
|---:|---:|---:|
| 42 | 0.9224 / 0.8016 | 0.8635 / 0.5058 |
| 2025 | 0.9214 / 0.8012 | 0.8636 / 0.5069 |
| 3407 | 0.9217 / 0.8011 | 0.8633 / 0.5073 |

4. ECGFounder + VAE-only online AT 不能直接复用 EfficientNet 的 AT runner；需要封装 `waveform -> frozen ECGFounder encoder -> Super5 head` 的可微 victim，使 latent-hull 内层搜索穿过 frozen encoder，但外层只更新线性头。

## 2026-05-23 ECGFounder + VAE-only Head AT

新增脚本：

```text
scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py
```

方法：

```text
ECGTwin VAE latent real anchors from target center
-> same-label latent-hull online search
-> ECGTwin VAE decode to 10s ECG
-> interpolate to ECGFounder 12 x 5000 input
-> frozen ECGFounder encoder
-> update only Linear(1024, 5) Super5 head
```

这条路线不使用 DiT 合成样本，也不使用 center token。和 EfficientNet1DV2 版的关键差异是：

```text
EfficientNet1DV2 AT: 更新整套 EfficientNet1DV2
ECGFounder AT: 冻结 ECGFounder encoder，只更新 Super5 线性头
```

20 epoch 四中心结果：

| center | ECGFounder linear baseline | ECGFounder + VAE-only head AT | delta | PTB-XL fold10 after AT |
|---|---:|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9322 / 0.5601 | +4.72pp / +12.59pp | 0.9122 / 0.7806 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9386 / 0.4762 | +4.40pp / +11.50pp | 0.9102 / 0.7708 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.8852 / 0.6590 | +6.33pp / +8.64pp | 0.9112 / 0.7789 |
| georgia | 0.8525 / 0.6551 | 0.8878 / 0.7250 | +3.53pp / +6.99pp | 0.9143 / 0.7829 |
| mean | 0.8635 / 0.5058 | 0.9110 / 0.6051 | +4.75pp / +9.93pp | 0.9120 / 0.7783 |

输出：

```text
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_headft_ep20_20260523/
```

解读：

1. ECGFounder head-only AT 在四个目标中心全部明显优于 frozen linear baseline。
2. `cpsc_2018` 上已经超过 EfficientNet1DV2 all-class VAE-only 结果：`0.8852 / 0.6590` vs `0.8684 / 0.6151`。
3. 20 epoch 相比 10 epoch 继续提升，但 PTB-XL fold10 有轻度遗忘：从 `0.9224 / 0.8016` 降到四中心适配后的约 `0.9120 / 0.7783`。
4. 目前 AUROC 在 ningbo/chapman 已达到或超过 PTB-XL fold10 AUROC；AUPRC 仍明显低于 PTB-XL 源域，不能声称已经完全达到源域性能。

推荐下一步：

```text
1. 在 ECGFounder head AT 上做 source_weight / target_real_weight / adv_weight 网格。
2. 加 EWA/L2-to-source-head 正则，降低 PTB-XL AUPRC 遗忘。
3. 对每个中心按 best epoch 早停，而不是固定 20 epoch。
4. 做 seed 稳定性：seed42, 20260531, 20260601。
```

### Source-Weight Tradeoff Pilot

`cpsc_2018` 上的第一轮 source stream 权重消融：

| source_weight | target center | target delta | PTB-XL fold10 after AT |
|---:|---:|---:|---:|
| 1.0 | 0.8852 / 0.6590 | +6.33pp / +8.64pp | 0.9112 / 0.7789 |
| 2.0 | 0.8755 / 0.6336 | +5.36pp / +6.10pp | 0.9157 / 0.7882 |
| 3.0 | 0.8681 / 0.6216 | +4.62pp / +4.91pp | 0.9177 / 0.7921 |

结论：

```text
source_weight=1.0: 最强目标中心适配
source_weight=2.0: 目标提升仍明显，PTB-XL AUPRC 遗忘减少约一半
source_weight=3.0: 更接近 PTB-XL 源域，但目标中心增益明显变小
```

`head_l2_anchor=0.01/0.1` 在旧 mean-squared 实现下影响很小。2026-05-23 已把脚本改成支持 relative head-anchor：

```text
relative_head_anchor = ||head - source_head||^2 / ||source_head||^2
```

### Target-Heavy Pilot

为了测试“外部目标中心性能最大化”上限，把 target real stream 和 online adversarial stream 加重：

```text
source_weight      = 1.0
target_real_weight = 80.0
adv_weight         = 40.0
epochs             = 20
K_anchor           = 150
M                  = 20
lambda             = 0.15
hull_steps         = 3
```

四中心结果：

| center | ECGFounder baseline | target-heavy VAE-only head AT | delta | PTB-XL fold10 after AT |
|---|---:|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9394 / 0.5962 | +5.44pp / +16.20pp | 0.9000 / 0.7610 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9408 / 0.4927 | +4.61pp / +13.15pp | 0.8961 / 0.7424 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9011 / 0.7086 | +7.92pp / +13.60pp | 0.8998 / 0.7559 |
| georgia | 0.8525 / 0.6551 | 0.8960 / 0.7459 | +4.35pp / +9.08pp | 0.9041 / 0.7626 |
| mean | 0.8635 / 0.5058 | 0.9193 / 0.6358 | +5.58pp / +13.01pp | 0.9000 / 0.7555 |

和默认 20 epoch 配置相比：

| config | target mean | PTB-XL fold10 mean after AT |
|---|---:|---:|
| default weights `1/20/10` | 0.9110 / 0.6051 | 0.9120 / 0.7783 |
| target-heavy `1/80/40` | 0.9193 / 0.6358 | 0.9000 / 0.7555 |

结论：

```text
target-heavy 明显提升外部目标中心 AUPRC，但会进一步牺牲 PTB-XL 源域。
如果论文主张是“少量目标中心样本适配能提升目标中心性能”，target-heavy 是最强结果。
如果论文主张需要同时保持源域性能，source_weight=2.0 或 source-floor checkpoint selection 更稳。
```

这说明当前还没有同时达到：

```text
target PN2021 AUPRC 接近 PTB-XL 0.80
PTB-XL fold10 AUPRC 仍保持 0.80
```

下一步应做多目标训练而不是单纯加大 target 权重：

```text
1. source-aware early stopping：保存 source floor 下 target 最优 checkpoint。
2. head EWA / stronger anchor loss：比当前 mean-squared L2 更强。
3. class-wise target weighting：优先增强目标中心低 AUPRC 类，而不是整体加权。
4. per-center config：georgia/cpsc 可用 target-heavy，chapman/ningbo 可能需要更强 source regularization。
```

### Relative Head-Anchor Pilot

在 target-heavy `1/80/40` 配置上，用 `cpsc_2018` 测 relative head-anchor：

| head_l2_anchor | target center | PTB-XL fold10 after AT | interpretation |
|---:|---:|---:|---|
| 0.0 | 0.9011 / 0.7086 | 0.8998 / 0.7559 | 目标中心最强，源域遗忘较大 |
| 0.25 | 0.8962 / 0.6933 | 0.9044 / 0.7652 | 折中，保住大部分目标增益 |
| 1.0 | 0.8859 / 0.6640 | 0.9102 / 0.7776 | 保源更强，但目标增益回落 |

结论：

```text
relative head-anchor 正则有效，但不能单独解决“目标中心接近 PTB-XL AUPRC 且源域不掉”的双目标。
它适合作为 target-heavy 的保源旋钮，推荐后续在 0.1-0.5 范围做 per-center sweep。
```

### Class-Aware Pilot

Per-class 审计显示，target-heavy 的 macro AUPRC 主要被稀有类拖低：

```text
chapman_shaoxing: HYP n_pos=19, MI n_pos=10
ningbo: MI n_pos=131
cpsc_2018: STTC AUPRC 低于 CD/NORM
```

脚本新增：

```text
--class_loss_weights CLASS=w,...
--target_class_sample_weights CLASS=w,...
--adv_class_sample_weights CLASS=w,...
```

这些参数分别作用于：

```text
BCE class dimension weight
target-real stream positive-class sampling multiplier
online-adv stream positive-class sampling multiplier
```

试验结果：

| center | class-aware setting | target-heavy baseline | class-aware result | interpretation |
|---|---|---:|---:|---|
| chapman_shaoxing | HYP/MI loss + sampling boost | 0.9408 / 0.4927 | 0.9401 / 0.4993 | HYP/MI AUPRC 小幅改善，macro +0.66pp |
| ningbo | HYP/MI loss + sampling boost | 0.9394 / 0.5962 | 0.9374 / 0.6009 | MI AUPRC 改善，macro +0.47pp，AUROC 小降 |
| cpsc_2018 | STTC boost | 0.9011 / 0.7086 | 0.8936 / 0.6781 | 负向，简单 STTC boost 不适合 CPSC |

Per-class 变化：

```text
chapman HYP AUPRC: 0.0276 -> 0.0369
chapman MI  AUPRC: 0.0865 -> 0.1135
ningbo  MI  AUPRC: 0.2376 -> 0.2653
```

结论：

```text
class-aware weighting 是有效但有限的 refinement。
它能改善稀有类一点，但不能把 ningbo/chapman macro AUPRC 推到 PTB-XL 同源水平。
后续不要盲目加大类权重；更合理的是 per-center、per-class 小范围 sweep，
并报告稀有类 n_pos 对 AUPRC 上限和方差的影响。
```

### Residual Adapter Head Pilot

2026-05-23 新增 `residual_adapter` 分类头：

```text
logits = frozen_source_linear_head(features) + adapter(features)
adapter = LayerNorm -> Linear(hidden) -> GELU -> Dropout -> Linear(5)
```

最后一层 adapter 使用 zero-init，所以初始输出严格等于 PTB-XL source linear head。这个版本的动机是让目标中心适配只通过残差分支发生，保留原始 PTB-XL Super5 head 作为锚点。

脚本修正：

```text
--head_type residual_adapter
--freeze_base_head
```

会显式冻结 base head、只训练 adapter 参数。

`cpsc_2018` pilot：

| method | selection | target center | target delta | PTB-XL fold10 after AT |
|---|---|---:|---:|---:|
| linear target-heavy | target AUPRC | 0.9011 / 0.7086 | +7.92pp / +13.60pp | 0.8998 / 0.7559 |
| residual adapter | target AUPRC | 0.9112 / 0.7385 | +8.93pp / +16.60pp | 0.8941 / 0.7435 |
| residual adapter, source_weight=2 | target/source hmean | 0.9088 / 0.7290 | +8.69pp / +15.65pp | 0.9085 / 0.7719 |
| residual adapter, source_weight=3 | target/source hmean | 0.9083 / 0.7278 | +8.84pp / +15.52pp | 0.9128 / 0.7817 |
| residual adapter, head_l2_anchor=0.25 | target/source hmean | 0.9111 / 0.7365 | +8.92pp / +16.39pp | 0.8967 / 0.7506 |

输出：

```text
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_20260523/
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_sw2_hmean_20260523/
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_sw3_hmean_20260523/
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_anchor025_hmean_20260523/
```

结论：

```text
residual adapter 能显著提高 CPSC 目标中心上限，target AUPRC 达到 0.7385，
比 linear target-heavy 高约 +2.99pp AUPRC。
source_weight=2 + hmean selection 是目前更好的双目标折中：
target 0.9088 / 0.7290，PTB-XL 0.9085 / 0.7719。
source_weight=3 能把 PTB-XL AUPRC 进一步拉到 0.7817，但目标 AUPRC 小幅回落到 0.7278。
但它仍没有达到 PTB-XL source AUPRC 0.8016，说明目标还未完成。
```

下一步优先级：

```text
1. 在 residual adapter 上做 source_weight=3/4 与 head_l2_anchor=0.1/0.25/0.5 的小网格。
2. 把 target/source hmean selection 作为保源默认，target-only selection 只作为上限报告。
3. 先在 CPSC 稳定后再扩展到 ningbo/chapman/georgia，避免四中心盲跑。
4. 如果 PTB-XL AUPRC 仍低于 0.79，尝试更小 adapter_hidden 或 adapter_scale，并加入 source calibration loss。
```

### Source-Logit Anchor Pilot

参数：

```text
--source_logit_anchor_weight
```

含义：在 PTB-XL source stream batch 上，用 frozen PTB-XL source head 作为 teacher，惩罚适配后 logits 偏离 source logits：

```text
L_source_logit = MSE(adapted_logits_source, frozen_source_logits)
```

这比单纯 `head_l2_anchor` 更直接，因为它约束的是源域输入上的输出行为，而不是参数距离。

`cpsc_2018`，`source_weight=2`，target/source hmean selection：

| config | CPSC target | PTB-XL fold10 |
|---|---:|---:|
| no source-logit anchor | 0.9088 / 0.7290 | 0.9085 / 0.7719 |
| source-logit anchor 0.1 | 0.9089 / 0.7304 | 0.9190 / 0.7943 |
| source-logit anchor 0.2 | 0.9082 / 0.7295 | 0.9204 / 0.7975 |
| source-logit anchor 0.5 | 0.9062 / 0.7232 | 0.9216 / 0.8005 |

输出：

```text
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_sw2_logit01_hmean_20260523/
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_sw2_logit02_hmean_20260523/
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_cpsc_sw2_logit05_hmean_20260523/
```

结论：

```text
source-logit anchor 是目前最有效的保源机制。
0.1/0.2 基本不牺牲 CPSC target AUPRC，同时把 PTB-XL AUPRC 从 0.7719 拉到 0.7943/0.7975。
0.5 可以把 PTB-XL AUPRC 拉回 0.8005，几乎等于 source baseline 0.8016，
但 CPSC target AUPRC 从 0.7290 小降到 0.7232。
```

当前最接近 active goal 的 CPSC 配置：

```text
residual_adapter + source_weight=2 + target/source hmean selection
+ source_logit_anchor_weight=0.2 or 0.5
```

其中 `0.2` 更适合目标中心适配，`0.5` 更适合“不损失 PTB-XL 源域”的论证。

### Four-Center Source-Logit Anchor 0.2 / 0.5

把 `source_logit_anchor_weight=0.2` 和最保源的 `0.5` 扩展到四个主中心：

```text
head_type                  = residual_adapter
source_weight              = 2.0
target_real_weight         = 80.0
adv_weight                 = 40.0
source_logit_anchor_weight = 0.2 or 0.5
selection_metric           = target_source_hmean_auprc
```

`source_logit_anchor_weight=0.5` 结果：

| center | baseline target | VAE-only residual adapter | delta | PTB-XL fold10 after AT |
|---|---:|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9210 / 0.5748 | +3.60pp / +14.06pp | 0.9211 / 0.8000 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9013 / 0.4850 | +0.67pp / +12.38pp | 0.9215 / 0.7978 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9056 / 0.7218 | +8.38pp / +14.93pp | 0.9217 / 0.8007 |
| georgia | 0.8525 / 0.6551 | 0.8909 / 0.7418 | +3.83pp / +8.67pp | 0.9216 / 0.7992 |
| mean | 0.8635 / 0.5058 | 0.9047 / 0.6309 | +4.12pp / +12.51pp | 0.9214 / 0.7994 |

输出：

```text
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_big4_sw2_logit05_hmean_20260523/
```

`source_logit_anchor_weight=0.2` 结果：

| center | baseline target | VAE-only residual adapter | delta | PTB-XL fold10 after AT |
|---|---:|---:|---:|---:|
| ningbo | 0.8850 / 0.4342 | 0.9229 / 0.5816 | +3.79pp / +14.74pp | 0.9201 / 0.7977 |
| chapman_shaoxing | 0.8946 / 0.3612 | 0.9086 / 0.4862 | +1.40pp / +12.50pp | 0.9201 / 0.7934 |
| cpsc_2018 | 0.8219 / 0.5725 | 0.9082 / 0.7274 | +8.63pp / +15.49pp | 0.9205 / 0.7980 |
| georgia | 0.8525 / 0.6551 | 0.8929 / 0.7466 | +4.04pp / +9.15pp | 0.9201 / 0.7959 |
| mean | 0.8635 / 0.5058 | 0.9082 / 0.6354 | +4.47pp / +12.97pp | 0.9202 / 0.7962 |

输出：

```text
/root/autodl-tmp/paper_ecgfounder_vae_only_lhat_resadapter_big4_sw2_logit02_hmean_20260523/
```

关键结论：

```text
这是一条新的强主线。
0.2: 目标中心更强，mean target 0.9082 / 0.6354，PTB-XL 0.9202 / 0.7962。
0.5: 源域保持更强，mean target 0.9047 / 0.6309，PTB-XL 0.9214 / 0.7994。
两者都显著优于 ECGFounder linear baseline 的 target mean 0.8635 / 0.5058。
```

但它仍不能声称目标中心指标已经达到 PTB-XL 源域同等水平，因为四中心 target mean AUPRC 是 `0.6309`，明显低于 PTB-XL source AUPRC `0.8016`。现在可以更准确地表述为：

```text
VAE-only residual-adapter online AT can substantially improve target-center PN2021 performance
while preserving PTB-XL source performance.
```

下一步：

```text
1. 对 0.2/0.5 两个版本做 3 seed 复现，确认四中心稳定性。
2. 对 EfficientNet1DV2 尝试类似 source-logit distillation / source-consistency loss；
   但 EfficientNet 没有 frozen source head，可能需要 EMA teacher 或 baseline teacher logits。
3. 如果要继续追 target AUPRC，上限实验保留 target-only residual adapter；
   如果要强调源域不掉，主线优先使用 source-logit anchor 0.5。
```

## 推荐执行命令

Smoke test：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py \
  --out_dir /root/autodl-tmp/paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_smoke \
  --limit_ptbxl 256 \
  --limit_per_center 16 \
  --batch_size 64 \
  --num_workers 2 \
  --epochs 2 \
  --preprocess_policy official_ptbxl_eval
```

Full run：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py \
  --out_dir /root/autodl-tmp/paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_seed42 \
  --batch_size 96 \
  --num_workers 4 \
  --head_batch_size 1024 \
  --epochs 80 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 42 \
  --preprocess_policy official_ptbxl_eval
```

多 seed 建议写 wrapper 顺序跑：

```text
seed42
seed2025
seed3407
```

不要同时开多个 ECGFounder feature extraction 进程读取 PN2021 原始 WFDB，否则 IO 会成为瓶颈。

## 预期耗时

如果 feature cache 不存在：

```text
PTB-XL feature extraction: 约 10-25 min
PN2021 feature extraction: 约 40-120 min，主要受 WFDB IO 影响
linear head training: 通常 < 10 min
report aggregation: < 5 min
```

如果 feature cache 已存在：

```text
单 seed 线性头训练 + 评估通常 < 10 min
3 seed 通常 < 30 min
```

## 论文使用原则

这个方法可以作为 ECGFounder 的强基线：

```text
Frozen ECGFounder encoder + PTB-XL Super5 linear probe
```

它比 150-class zero-shot keyword mapping 更公平，因为分类头直接在 PTB-XL Super5 上训练；也比 full fine-tuning 更保守，因为没有更新 ECGFounder encoder。

最终对比时必须分清：

| 方法 | 是否使用 PTB-XL 监督 | 是否使用目标中心 K 样本 | 是否更新大模型 encoder |
|---|---:|---:|---:|
| ECGFounder zero-shot strict mapping | 否 | 否 | 否 |
| ECGFounder frozen linear probe | 是，fold 1-8 | 否 | 否 |
| EfficientNet1DV2 PTB-XL baseline | 是，fold 1-8 | 否 | 训练 from scratch |
| VAE-only 在线对抗训练 | 是，PTB-XL baseline | 是，K target anchors | 不涉及 ECGFounder |

如果 ECGFounder frozen linear probe 在 PN2021 上超过我们的 VAE-only 方法，应如实报告，并把我们的贡献定位为“小样本目标中心适配方法”；如果它低于或接近我们的 VAE-only 方法，则可说明目标中心 latent-anchor 在线对抗训练在跨中心适配上仍有优势。
