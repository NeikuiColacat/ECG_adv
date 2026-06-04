# 近期实验汇报：PN2021 跨中心适配、PN2021-C 扰动与 VAE-only 在线对抗训练探索

日期：2026-05-18

## 1. 汇报摘要

本轮实验主要围绕三个问题：

1. 对比三种目标中心增强路线：`VAE-only / real-anchor 在线对抗训练`、`no-token + 在线对抗训练`、`target-token + 在线对抗训练`。
2. 构建 ImageNet-C 风格的 `PN2021-C` 扰动测试集，观察 ECG corruption 对模型 AUROC/AUPRC 的影响。
3. 针对当前最稳的 `VAE-only real-anchor latent-hull 在线对抗训练` 做 K、lambda、组合样本数 M、训练轮数和标签组合方式的探索。

当前最重要的结论是：

- 对四个目标中心 `ningbo`、`chapman_shaoxing`、`cpsc_2018`、`georgia`，只使用目标中心 K=500 真实样本的 ECGTwin VAE latent 做 latent-hull 对抗训练，平均提升约 `+2.47pp AUROC / +3.32pp AUPRC`。
- `no-token + 在线对抗训练` 和 `target-token + 在线对抗训练` 也能提升，但和 VAE-only 的结果非常接近。当前数据不支持“center token 是主要性能提升来源”的强结论。
- PN2021-C 上，5 类 ECG 扰动整体让 baseline 平均下降约 `0.95pp AUROC / 1.41pp AUPRC`。
- VAE-only 在线对抗训练的关键收益来自“目标中心真实样本的 ECGTwin VAE latent manifold”。

## 2. 统一实验协议

### 2.1 源模型

源模型为 PTB-XL 训练得到的 EfficientNet1DV2 super5 分类器：

```text
checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

preprocess_mode:
  minimal_resample

norm_mode:
  per_sample_global

sampling rate:
  100 Hz

input length:
  1000 samples = 10 seconds
```

super5 类别顺序：

```text
CD, HYP, MI, NORM, STTC
```

### 2.2 输入预处理解释

`minimal_resample`：只做必要格式对齐，包括 NaN/Inf 清理、导联重排、重采样到 `100Hz`、pad/truncate 到 `1000 samples = 10 seconds`。

`per_sample_global`：每条 ECG 单独在完整 `12 x 1000` 输入上算一个全局 mean/std 做 z-score：

```text
x_norm = (x - mean(x_12x1000)) / (std(x_12x1000) + 1e-8)
```

它不是 dataset-level 标准化，也不是 per-lead 标准化。优点是缓解跨设备幅值差异；限制是绝对电压信息会被削弱，HYP/CD 仍需 raw/digital ECG 分支辅助验证。

### 2.3 目标中心与测试污染控制

主要目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

每个目标中心抽取 K 条真实 ECG 作为适配 anchor。做目标中心评估时，这 K 条 ECG 会从测试集中排除，避免把参与适配的数据重新用于测试。

### 2.4 当前 PN2021 标签映射

PN2021 使用项目定义的 `v3_super5_normsuppress` 映射到 PTB-XL super5。该映射将 PN2021 的 SNOMED 标签投影到 `CD/HYP/MI/NORM/STTC`，并使用 NORM suppression 逻辑避免 abnormal 样本被错误标成 NORM。

## 3. 三种增强方法对比

### 3.1 方法定义

`baseline`：只用 PTB-XL 训练的 EfficientNet1DV2，不使用目标中心样本适配。

`VAE-only / real-anchor 在线对抗训练`：只使用目标中心 K=500 真实 ECG，经 ECGTwin VAE encoder 得到 latent，在 VAE latent space 内做 latent-hull 在线对抗训练。不使用 ECGTwin DiT 合成样本，不使用 center token。

`no-token + 在线对抗训练`：在目标中心 `K=500` 真实 anchor 基础上，加入 ECGTwin no-token 生成候选 latent，再做在线对抗训练。

`target-token + 在线对抗训练`：在目标中心 `K=500` 真实 anchor 基础上，加入 ECGTwin target-center prompt token 生成候选 latent，再做在线对抗训练。center token 是添加到 ECGTwin DiT denoise text prompt embedding path 的 learnable soft prompt token。

本轮四中心候选池数量如下，括号内为 ECGTwin 生成候选数：

| center | no-token pool | target-token pool |
|---|---:|---:|
| ningbo | 846 = 500 real + 346 generated | 846 = 500 real + 346 generated |
| chapman_shaoxing | 598 = 500 real + 98 generated | 617 = 500 real + 117 generated |
| cpsc_2018 | 560 = 500 real + 60 generated | 604 = 500 real + 104 generated |
| georgia | 605 = 500 real + 105 generated | 600 = 500 real + 100 generated |

### 3.2 在线对抗训练核心公式

当前主线使用：

```text
z_adv = (1 - lambda) * z0 + lambda * sum_i softmax(a_i) * z_i
```

含义：

- `z0`：一个目标中心真实 ECG 的 ECGTwin VAE latent。
- `z_i`：同标签或兼容标签集合里的其他目标中心 ECG latent，默认来自真实目标中心样本。
- `a_i`：可优化的组合权重。训练时对 `a_i` 做几步梯度上升，让 VAE decode 后的 ECG 更靠近当前分类器的错误或高损失区域。
- `softmax(a_i)`：保证多个候选样本权重非负且总和为 1。
- `lambda`：控制从 anchor `z0` 向 latent hull 内部移动的幅度。`lambda=0` 时完全是原始 anchor；`lambda` 越大，对抗样本越靠近候选 latent 组合；`lambda=1.0` 时显式的 `(1-lambda) * z0` anchor 保留项消失。

训练时，先在 latent space 中构造 `z_adv`，再通过 ECGTwin VAE decoder 还原为 ECG 波形，最后用这些样本训练 EfficientNet1DV2，使模型对目标中心边界附近样本更稳。

### 3.3 关键训练设置

```text
K                 = 500 target-center real anchors
quality gate      = disabled
hull_M            = 10
hull_lambda       = 0.15
hull_lr           = 0.25
adv_weight        = 0.06
PTB-XL stream     = enabled
VAE roundtrip     = enabled
```

本轮质量 gate 关闭的含义是：仍然记录医学/语义 gate 指标，但 gate 失败不会阻止样本进入 adversarial buffer。

其中：

- `adv_weight=0.06`：训练总 loss 里对抗样本 loss 的权重。数值越大，对抗样本影响越强，但也更可能牺牲原始 PTB-XL/source 性能。

运行入口：

```text
scripts/paper/run_fourway_no_gate_lhat_20260518.py
```

### 3.4 四中心目标测试结果

表中括号为相对同一 excluded-ref baseline 的提升，单位为 percentage points。

| center | baseline | VAE-only / real-anchor 在线对抗训练 | no-token + 在线对抗训练 | target-token + 在线对抗训练 |
|---|---:|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8918 / 0.5231 (+2.45pp / +4.48pp) | 0.8884 / 0.5167 (+2.12pp / +3.83pp) | 0.8885 / 0.5165 (+2.13pp / +3.81pp) |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4634 (+2.03pp / +3.82pp) | 0.8972 / 0.4653 (+2.09pp / +4.02pp) | 0.8972 / 0.4647 (+2.09pp / +3.95pp) |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8552 / 0.5973 (+4.38pp / +3.87pp) | 0.8541 / 0.5958 (+4.27pp / +3.72pp) | 0.8543 / 0.5959 (+4.28pp / +3.73pp) |
| georgia | 0.8157 / 0.5916 | 0.8259 / 0.6026 (+1.02pp / +1.10pp) | 0.8255 / 0.6019 (+0.98pp / +1.03pp) | 0.8256 / 0.6021 (+0.99pp / +1.05pp) |

四中心平均：

| method | avg AUROC/AUPRC | avg delta |
|---|---:|---:|
| baseline | 0.8427 / 0.5134 | 0.00pp / 0.00pp |
| VAE-only / real-anchor 在线对抗训练 | 0.8674 / 0.5466 | +2.47pp / +3.32pp |
| no-token + 在线对抗训练 | 0.8663 / 0.5449 | +2.37pp / +3.15pp |
| target-token + 在线对抗训练 | 0.8664 / 0.5448 | +2.37pp / +3.14pp |

### 3.5 解读

当前性能最稳的是 `VAE-only / real-anchor 在线对抗训练`。这说明少量目标中心真实 ECG 的 VAE latent manifold 本身就已经提供了足够有效的目标中心适配信号。

`target-token + 在线对抗训练` 没有明显超过 `no-token + 在线对抗训练`。因此当前不能把性能提升主要归因于 center token。更合理的表述是：center token 可以作为 ECGTwin 可控生成和候选池构造模块，但当前主性能增益来自真实目标中心样本的 VAE latent-hull 对抗训练。

## 4. PN2021-C 风格扰动测试

### 4.1 构建方式

PN2021-C 参考 ImageNet-C 的思想，不用于训练，只作为鲁棒性测试集。

```text
centers:
  ningbo, chapman_shaoxing, cpsc_2018, georgia

corruptions:
  powerline_noise
  emg_noise
  baseline_wander
  baseline_shift
  random_leads_masking

severities:
  1, 2, 3, 4, 5
```

构建与评估入口：

```text
scripts/triple_labels/build_pn2021_corruptions.py
scripts/triple_labels/eval_pn2021_corruptions.py
```

### 4.2 baseline 在 PN2021-C 上的下降

| metric | clean mean | corrupted mean | drop |
|---|---:|---:|---:|
| AUROC | 0.8428 | 0.8332 | -0.95pp |
| AUPRC | 0.5237 | 0.5095 | -1.41pp |

按中心：

| center | clean AUROC/AUPRC | corrupted mean AUROC/AUPRC | drop |
|---|---:|---:|---:|
| ningbo | 0.8657 / 0.4842 | 0.8547 / 0.4663 | -1.10pp / -1.79pp |
| chapman_shaoxing | 0.8733 / 0.4427 | 0.8616 / 0.4277 | -1.16pp / -1.50pp |
| cpsc_2018 | 0.8154 / 0.5712 | 0.8092 / 0.5595 | -0.62pp / -1.17pp |
| georgia | 0.8167 / 0.5966 | 0.8074 / 0.5846 | -0.93pp / -1.20pp |

按 corruption：

| corruption | corrupted mean AUROC/AUPRC | drop |
|---|---:|---:|
| powerline_noise | 0.8427 / 0.5234 | -0.01pp / -0.03pp |
| emg_noise | 0.8412 / 0.5204 | -0.15pp / -0.32pp |
| baseline_wander | 0.8421 / 0.5233 | -0.06pp / -0.04pp |
| baseline_shift | 0.8393 / 0.5189 | -0.34pp / -0.48pp |
| random_leads_masking | 0.8008 / 0.4617 | -4.20pp / -6.20pp |

按 severity：

| severity | corrupted mean AUROC/AUPRC | drop |
|---:|---:|---:|
| 1 | 0.8403 / 0.5196 | -0.24pp / -0.40pp |
| 2 | 0.8373 / 0.5151 | -0.55pp / -0.86pp |
| 3 | 0.8342 / 0.5098 | -0.85pp / -1.39pp |
| 4 | 0.8296 / 0.5051 | -1.31pp / -1.86pp |
| 5 | 0.8246 / 0.4981 | -1.81pp / -2.56pp |

## 5. VAE-only 在线对抗训练的进一步探索

### 5.1 K 值曲线

本轮 K 曲线使用：

```text
method             = VAE-only / real-anchor 在线对抗训练
quality gate       = disabled
M                  = 20
lambda             = 0.15
target_real_weight = 40.0
adv_weight         = 0.06
roundtrip_anchor_n = 1500
```

完整 K=500 结果：

| center | baseline | VAE-only 在线对抗训练 | delta |
|---|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8917 / 0.5223 | +2.44pp / +4.39pp |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4638 | +2.03pp / +3.87pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp |
| georgia | 0.8157 / 0.5916 | 0.8256 / 0.6020 | +1.00pp / +1.04pp |

简化 K 曲线，表中为 delta：

| center | K=20 | K=100 | K=300 | K=500 |
|---|---:|---:|---:|---:|
| ningbo | +0.51pp / +0.91pp | +1.30pp / +2.41pp | +2.09pp / +3.95pp | +2.44pp / +4.39pp |
| chapman_shaoxing | +0.21pp / +0.27pp | +0.89pp / +1.33pp | +1.80pp / +3.26pp | +2.03pp / +3.87pp |
| cpsc_2018 | -0.09pp / +0.44pp | +0.97pp / +1.03pp | +3.30pp / +2.60pp | +4.30pp / +3.76pp |
| georgia | +0.30pp / +0.06pp | +0.58pp / +0.40pp | +1.06pp / +1.28pp | +1.00pp / +1.04pp |

结论：K 越大，目标中心性能通常越好。`ningbo`、`chapman_shaoxing`、`cpsc_2018` 的趋势明显；`georgia` 在 K=300 左右基本进入平台期。

### 5.2 lambda 敏感性

这里的常规 lambda sweep 使用的是 `VAE-only / real-anchor 在线对抗训练`：只使用目标中心 K=500 真实 ECG 的 ECGTwin VAE latent，不使用 DiT synthetic，不使用 center token；target-center 评估排除对应 K=500 ref ECG。固定设置为 `M=10`。

| method | center | lambda | target AUROC/AUPRC | 备注 |
|---|---|---:|---:|---|
| VAE-only real-anchor | chapman_shaoxing | 0.05 | 0.8967 / 0.4641 | 常规 sweep |
| VAE-only real-anchor | chapman_shaoxing | 0.15 | 0.8967 / 0.4638 | 主线默认 |
| VAE-only real-anchor | chapman_shaoxing | 0.35 | 0.8969 / 0.4641 | 基本持平 |
| VAE-only real-anchor | cpsc_2018 | 0.05 | 0.8542 / 0.5961 | 常规 sweep |
| VAE-only real-anchor | cpsc_2018 | 0.15 | 0.8544 / 0.5962 | 主线默认 |
| VAE-only real-anchor | cpsc_2018 | 0.35 | 0.8545 / 0.5968 | 小幅提升 |
| VAE-only real-anchor | cpsc_2018 | 0.50 | 0.8546 / 0.5975 | 高 lambda ablation |
| VAE-only real-anchor | cpsc_2018 | 0.70 | 0.8547 / 0.5976 | 高 lambda ablation |
| VAE-only real-anchor | cpsc_2018 | 0.80 | 0.8547 / 0.5977 | 高 lambda ablation |
| VAE-only real-anchor | georgia | 0.05 | 0.8257 / 0.6021 | 常规 sweep |
| VAE-only real-anchor | georgia | 0.15 | 0.8256 / 0.6020 | 主线默认 |
| VAE-only real-anchor | georgia | 0.35 | 0.8255 / 0.6017 | 基本持平 |

`lambda=1.0` 时显式 anchor 插值项消失，公式从：

```text
z_adv = (1 - lambda) * z0 + lambda * sum_i softmax(a_i) * z_i
```

变成：

```text
z_adv = sum_i softmax(a_i) * z_i
```

因此必须区分候选集合里是否还包含原始 anchor `z0`，以及是否混入 ECGTwin prompt-token synthetic pool：

| method / variant | pool | z0 是否仍在候选集合 | center / setting | lambda | target AUROC/AUPRC | 结论 |
|---|---|---|---|---:|---:|---|
| VAE-only real-anchor | real K500 only | 显式插值保留 z0 | cpsc_2018, M=10, ep=10 | 0.15 | 0.8544 / 0.5962 | 主线默认 |
| mixed target-token | real K500 + prompt-token synth | 显式插值保留 z0 | cpsc_2018, M=10, ep=10 | 0.15 | 0.8543 / 0.5959 | 与 real-anchor 主线接近 |
| mixed target-token no-lambda | real K500 + prompt-token synth | 否，激进 no-anchor mixed pool | cpsc_2018, M=10, ep=10 | 1.00 | 0.8447 / 0.5739 | 明显变差，不适合作为主线 |

结论：`lambda=0.05-0.35` 对 VAE-only real-anchor 主线不敏感，CPSC 上提高到 `0.50-0.80` 只有小幅提升。混入 prompt-token synthetic 后再使用激进 no-anchor `lambda=1.0` 会明显变差。因此主线保留 `lambda=0.15`。

### 5.3 组合样本数 M 与训练轮数

在更长训练 `epochs=30` 下，M 的影响整体不大，但 Ningbo 上 M=80 有小幅最好结果。

| center | M=10 | M=20 | M=40 | M=80 |
|---|---:|---:|---:|---:|
| ningbo | 0.8946 / 0.5300 | 0.8946 / 0.5300 | 0.8946 / 0.5301 | 0.8953 / 0.5314 |
| cpsc_2018 | 0.8680 / 0.6146 | 0.8682 / 0.6147 | 0.8681 / 0.6147 | 0.8681 / 0.6146 |

这说明 M 从 10 增加到 80 并不会稳定带来大幅收益。更合理的工程设置是：

- 快速复现实验：`M=10` 或 `M=20`。
- 论文主结果或最终表：`M=20` 或 `M=80`，`epochs=30`，看目标中心是否值得加算力。

## 6. 对比普通 K-shot 适配方法

在相同 EfficientNet backbone 下，普通目标中心 K-shot fine-tuning、Mixup、Manifold Mixup、input-space FGSM/PGD-AT、MixStyle、GroupDRO proxy、AdaBN、TENT、EATA-lite、CoTTA-lite、SHOT-style、Deep CORAL、DANN、CDAN 都没有追上 VAE real-anchor 在线对抗训练。

外部大模型维度下，ECGFounder K-shot head fine-tuning 可以超过当前方法，但那是更换 backbone 的强表征 baseline，不是同一 EfficientNet 源模型上的轻量目标中心适配。

## 7. 相关产物

```text
三方法 no-gate 复跑：
  docs/reports/archive/20260518/fourway_lhat_nogate_rerun_20260518.md
  /root/autodl-tmp/paper_fourway_lhat_nogate_20260518/

VAE-only K 曲线：
  docs/reports/archive/20260518/vae_only_lhat_kcurve_20260518.md
  /root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518/

PN2021-C pipeline：
  docs/pipelines/pn2021_c_corruption_benchmark_pipeline.md
  /root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
```
