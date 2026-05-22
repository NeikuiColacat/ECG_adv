# 当前论文主线 Pipeline：Center Token + VAE Latent 在线对抗训练

日期：2026-05-12

本文档定义当前论文工作的主实验路线。毕业设计演示系统已经完成，后续重点是论文方法验证：

```text
训练目标中心 ECGTwin center-class prompt token
-> 生成并筛选目标中心 ECGTwin VAE latent 候选样本
-> 结合目标中心真实 K=500 anchor 和 ECGTwin synthetic latent
-> 在 VAE latent space 中做 same-label Latent-Hull 在线对抗训练
-> 在 PN2021 跨中心 held-out 数据上评测 AUROC / AUPRC
```

当前论文要回答的核心问题：

```text
1. center token 是否能学习目标中心 ECG 风格？
2. center-token ECGTwin latent candidates 是否比 no-token candidates 更有帮助？
3. VAE latent space 在线对抗训练是否能稳定提升跨中心 AUROC / AUPRC？
```

## 总体方法

论文主线分为六步：

```text
1. 训练 source classifier：
   用 PTB-XL folds 1-8 训练 EfficientNet1DV2 super5 分类器。

2. 构造目标中心 few-shot 数据：
   对每个 PN2021 目标中心采样 K=500 条真实 ECG。

3. 训练 center token：
   冻结 ECGTwin，只训练目标中心、目标类别对应的 soft prompt token。

4. 生成候选 latent：
   分别用 no-token、target-token、wrong-center token 生成 ECGTwin candidates，
   并保存它们的 ECGTwin VAE latent。

5. 在线对抗训练：
   以目标中心真实 ECG latent 为 anchor，在 same-label candidate latent 邻域里
   构造 latent-hull 对抗样本，然后解码成 ECG 训练 EfficientNet1DV2。

6. 跨中心评测：
   在 PN2021 held-out target-center 数据上评测 AUROC / AUPRC。
   用于 token training / target adaptation 的 K=500 ref ECG 必须从 eval 中排除。
```

## 固定标签体系

任务固定为 PTB-XL diagnostic super class 5：

```text
class order = CD, HYP, MI, NORM, STTC
```

源码依据：

```text
scripts/triple_labels/label_schemes.py
```

PN2021 使用项目定义的 v3 super5 semantic projection：

```text
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
PN2021_EVAL_CACHE_VERSION     = v3_super5_normsuppress
```

注意：

```text
PN2021 没有官方 SNOMED -> PTB-XL super5 crosswalk。
本项目使用的是论文中需要明确说明的语义投影规则。
```

目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

主要成功中心目前是：

```text
ningbo
chapman_shaoxing
cpsc_2018
```

`georgia` 当前提升较小，论文中应作为困难中心或限制分析。

## Source EfficientNet1DV2 输入协议

当前论文主线使用新输入协议：

```text
preprocess_mode = minimal_resample
norm_mode       = per_sample_global
sampling rate   = 100Hz
input length    = 1000 samples = 10 seconds
crop_len        = 1000
lead order      = PTB-XL canonical order
```

含义：

```text
minimal_resample:
  只做必要清理、导联顺序对齐、重采样、pad/truncate。
  不默认使用 legacy ECGFounder-style 完整滤波链。

per_sample_global:
  每条 ECG 在 12 x 1000 全部点上计算一个 mean/std 做 z-score。
```

当前 baseline：

```text
/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503
```

已记录指标：

```text
PTB-XL fold10 AUROC/AUPRC ~= 0.9072 / 0.7744
PN2021 avg AUROC/AUPRC    ~= 0.7780 / 0.4831
```

该 baseline 是后续所有 online AT 的 source classifier。

### 2026-05-12 baseline 事实核查

已核查当前 baseline 训练产物：

```text
/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503
```

事实结论：

```text
preprocess_mode = minimal_resample
norm_mode       = per_sample_global
sampling rate   = 100Hz
input length    = 1000 samples = 10 seconds
crop_len        = 1000
lead order      = PTB-XL canonical order
```

证据：

```text
train_result.json:
  config.preprocess_mode = minimal_resample
  config.norm_mode       = per_sample_global
  config.crop_len        = 1000
  config.epochs          = 80
  epochs_trained         = 27
  checkpoint_metric      = auprc

train_stdout.log:
  cache = /root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
  Train/Val/Test = 17418 / 2183 / 2198
  Early stopping @ ep27
  Test macro AUROC/AUPRC = 0.9072 / 0.7745

eval_result_v3_super5_normsuppress.json:
  config.preprocess_mode = minimal_resample
  config.norm_mode       = per_sample_global
  config.crop_len        = 1000
  PN2021 avg AUROC/AUPRC = 0.7780 / 0.4831
```

PTB-XL 原始缓存和训练缓存形状：

```text
/root/autodl-tmp/ptbxl/raw100.npy
  shape = (21799, 1000, 12)

/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
  shape = (21799, 1000, 12)
  dtype = float32
  first sample mean/std ~= 0 / 1
```

因此不需要为了确认输入协议重训 baseline。后续若要研究 seed 稳定性，可以新增
baseline seed ablation，但不能把它和 center-token/Latent-Hull AT 消融混在一个结论里。

## Center Token 训练方案

当前 center token 是 textual-inversion-style soft prompt，不是修改 ECGTwin tokenizer。

实现位置：

```text
methods/ecgtwin_gen/prompt_token/
scripts/ecgtwin_gen/train_center_prompt_tokens.py
```

ECGTwin 原始条件路径：

```text
普通诊断文本 -> nomic text embedding -> text_embed
reference ECG / report / patient info -> IBE -> base_vector
DiT 同时接收 text_embed、text_embed_mask、pat_info、base_vector
```

我们的 center token 做法：

```text
普通诊断 prompt 仍然走 nomic embedding。
<center_CLASS> 不走 tokenizer。
它由我们自己的 prompt compiler 替换成可训练的 768-d soft prompt 向量。
这些向量被追加到 ECGTwin 的 text_embed 序列末尾，并设置 mask=1。
```

推荐 token schema：

```text
token_mode      = direct
n_token_vectors = 4
embedding shape = embeddings[center, class, vector, 768]
```

也就是说，每个中心、每个类别有 4 个 768 维向量：

```text
<ningbo_MI> = 4 x 768
```

生成 ningbo 风格 MI ECG 时：

```text
diagnosis prompt text_embed + <ningbo_MI> MV4 soft prompt
reference ECG/text/patient info -> IBE base_vector
DiT denoising -> VAE decoder -> ECG
```

边界：

```text
不要把目标中心风格写进 base_vector。
base_vector 保持 ECGTwin 的 reference-personalization 条件。
center style 只放在 text prompt-token path。
```

## ECGTwin 候选池

论文必须保留 matched controls，不能只报告 target-token：

```text
T0: no-token ECGTwin candidates
T1: target-center token ECGTwin candidates
T2: wrong-center token ECGTwin candidates
T3: PTB-XL/source token candidates
T4: target-center real K=500 anchors only
```

当前有用产物路径：

```text
target-token scale=0.50 large ningbo:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/target_token_s05/ningbo/gated

matched no-token large ningbo:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/no_token/ningbo/gated

v4 balanced prompt-token pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated

multi-center online AT historical runs:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2
```

候选样本筛选 gate：

```text
基础信号 gate:
  NaN/Inf
  flatline
  amplitude
  lead consistency
  Einthoven residual

语义 gate:
  frozen EfficientNet1DV2 target-class probability

NORM gate:
  abnormal class suppression

Latent-Hull gate:
  only same-label candidates can be used for label-preserving latent mixing
```

## VAE Latent Space 在线对抗训练

ECGTwin 数据格式：

```text
ECGTwin VAE ECG format = (B, 1024, 12), ECGTwin/MIMIC lead order
VAE latent             = (B, 4, 128)
classifier ECG format  = (B, 12, 1000), PTB-XL lead order
```

ECGTwin -> PTB-XL 导联转换：

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

核心公式：

```text
z_adv = (1 - lambda) z0 + lambda * sum_i softmax(a_i) z_i
```

### 这个公式具体怎么做

变量含义：

```text
z0:
  一个目标中心真实 ECG 的 VAE latent。
  它是 anchor，也就是我们最信任的目标中心真实样本。

z_i:
  与 z0 同标签的候选 latent。
  可以来自目标中心真实 K=500 anchors，也可以来自 ECGTwin no-token /
  target-token synthetic candidates。

a_i:
  每个候选 latent 的可优化权重参数。
  它不是手工固定的权重，而是在 online AT 过程中根据分类器梯度更新。

softmax(a_i):
  把 a_i 转成非负且和为 1 的权重。
  这样 sum_i softmax(a_i) z_i 是候选 latent 的凸组合。

lambda:
  控制 anchor z0 和候选组合之间的混合强度。
  lambda 越大，对抗样本越靠近候选池；
  lambda 越小，对抗样本越接近真实 anchor。
```

执行流程：

```text
1. 取一个目标中心真实 ECG anchor。
   例如 ningbo 的一条 MI ECG。

2. 用 ECGTwin VAE encoder 得到 anchor latent:
   z0 = VAE.encode(real_target_ecg)

3. 在候选池里找同标签 latent。
   例如同样是 MI 的 no-token / target-token ECGTwin generated latents。

4. 选 M 个候选 latent:
   z_1, z_2, ..., z_M

5. 初始化权重参数 a:
   可以 one-hot，也可以全 0。
   如果 a 全 0，那么 softmax(a) 是均匀平均。
   如果 one-hot 初始，相当于一开始更接近某一个候选样本。

6. 构造候选组合:
   z_mix = sum_i softmax(a_i) z_i

7. 构造对抗 latent:
   z_adv = (1 - lambda) z0 + lambda z_mix

8. 用 ECGTwin VAE decoder 解码:
   ecg_adv = VAE.decode(z_adv)

9. 把 ecg_adv 转成 EfficientNet1DV2 输入格式:
   ECGTwin lead order -> PTB-XL lead order
   1024 samples -> 1000 samples
   per-sample global z-score
   shape -> (B, 12, 1000)

10. 送入当前 EfficientNet1DV2:
    logits = classifier(ecg_adv)

11. 根据目标 loss 反向传播，更新 a_i。
    目标是找到仍在同类 latent manifold 内、但能让分类器更困难的样本。

12. 用这些 online 生成的困难样本训练 classifier。
```

为什么用 `softmax(a_i)`：

```text
softmax 保证所有权重 >= 0，且总和 = 1。
因此 z_mix 是 convex combination，不是任意线性外推。
这比自由 PGD perturbation 更不容易跑出 ECGTwin VAE 的真实数据流形。
```

为什么标签通常不变：

```text
1. z0 是真实目标中心 ECG anchor，标签可信。
2. z_i 只选 same-label candidates。
3. 混合方式是 convex hull，不做跨类插值。
4. NORM 最严格，不能和 abnormal candidates 混。
5. semantic gate 会检查 target-class probability 和 NORM suppression。
```

因此论文里可以说：

```text
在 same-label 约束和 semantic gate 下，该方法构造的是
label-preserving latent-hull adversarial candidates。
```

但不能说：

```text
任意 latent 线性组合都医学合法。
```

### 训练 loss 设计

在线训练时通常有两路数据：

```text
clean target-real stream:
  使用 K=500 目标中心真实 ECG 做普通 supervised BCE loss。

latent-hull adversarial stream:
  使用 z_adv 解码后的 ECG 做 adversarial supervised loss 或 teacher soft-label loss。
```

总 loss 可以写成：

```text
L_total =
  L_source_or_target_real
  + adv_weight * L_adv(z_adv)
  + anchor_regularization
  + optional soft-label / consistency loss
```

当前默认参数：

```text
K target real anchors = 500
M candidates          = 10
lambda                = 0.15
adv_weight            = 0.06
softmix               = 0.30
target_real_weight    = 40
epochs                = 10
```

实现位置：

```text
adversarial/latent_hull_pgd.py
scripts/pgd_cross_center/synth_online_at_super5.py
```

## 必要消融实验

论文不能只报告 target-token 成功，需要以下因果对照：

```text
A0: PTB-XL full EfficientNet1DV2 baseline
A1: target-center real K=500 supervised stream only
A2: target-center real anchors + real-anchor Latent-Hull AT
A3: target-center real anchors + no-token ECGTwin candidates + Latent-Hull AT
A4: target-center real anchors + target-token ECGTwin candidates + Latent-Hull AT
A5: target-center real anchors + wrong-center token candidates + Latent-Hull AT
```

解释规则：

```text
A3 > A1/A2:
  说明 ECGTwin synthetic latent candidates 有帮助。

A4 > A3:
  说明 center token 在 no-token ECGTwin candidates 之外有额外价值。

A4 ~= A3:
  说明 online AT 的主要增益来自 target-real anchors + ECGTwin latent candidates
  + latent-hull training，而不是 center token 本身。

A5 < A4:
  支持 center-specific token control。
```

## 当前已有跨中心结果

### K=500 Token 对照总表

本表使用每个方法实际使用的 `merged.ref_meta.json` 排除对应 500 条目标中心
adaptation refs。评估口径固定为 `minimal_resample + per_sample_global +
10s/100Hz`。`baseline` 是同一 excluded-ref 测试子集上的 bare PTB-XL
EfficientNet1DV2。

| center | baseline | no-token + LH-AT | target-token + LH-AT | no-token delta | token delta | token - no-token |
|---|---:|---:|---:|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8879 / 0.5156 | 0.8881 / 0.5155 | +0.0206 / +0.0373 | +0.0209 / +0.0371 | +0.0003 / -0.0001 |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8972 / 0.4653 | 0.8972 / 0.4647 | +0.0209 / +0.0402 | +0.0209 / +0.0395 | -0.0001 / -0.0007 |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8541 / 0.5958 | 0.8543 / 0.5959 | +0.0427 / +0.0372 | +0.0428 / +0.0373 | +0.0002 / +0.0001 |
| georgia | 0.8157 / 0.5916 | 0.8126 / 0.5861 | 0.8262 / 0.6029 | -0.0031 / -0.0055 | +0.0106 / +0.0113 | +0.0136 / +0.0168 |

对应产物：

```text
baseline exact excluded-ref:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_exclrefs/token_comparison_20260512/

ningbo no-token / target-token strict:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/repro_seed_checks/

chapman/cpsc/georgia v42 token controls:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/

georgia no-token补跑:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/merged_real_token_v2/georgia_real500_v42_vanilla/
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/georgia_v42_vanilla_w40_src04_M10_lam015_adv006_softmix03_ep10/
```

当前诚实结论：

```text
1. 相对同一 K=500 excluded-ref baseline，real-anchor 或 target-token
   Latent-Hull AT 在四个中心都有提升。
2. ningbo / chapman / cpsc 的 no-token 与 target-token 几乎持平。
3. georgia 是例外：no-token 低于 baseline，而 target-token 高于 baseline，
   token - no-token 为 +0.0136 / +0.0168。
4. 因此当前 full-data cross-center 平均增益仍不能主要归因于 center token；
   但 georgia 个例支持 center token 可能在部分中心有帮助。
5. 更稳妥的论文主贡献应写作：
   target-center real-anchor + ECGTwin VAE latent candidates + Latent-Hull online AT。
6. center token 目前可以作为候选生成模块和可控生成方向，不能单独声称是主要增益来源。
```

### K=100 表

本表是 real-anchor Latent-Hull K sweep，固定 `lambda=0.15`。no-token /
target-token 的 K sweep 尚未运行，因此这里不填 token 值。

| center | baseline | real-anchor LH-AT | delta | no-token | target-token |
|---|---:|---:|---:|---:|---:|
| ningbo | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| chapman_shaoxing | 0.8725 / 0.4365 | 0.8814 / 0.4494 | +0.0088 / +0.0129 | 未跑 | 未跑 |
| cpsc_2018 | 0.8143 / 0.5687 | 0.8240 / 0.5789 | +0.0097 / +0.0102 | 未跑 | 未跑 |
| georgia | 0.8162 / 0.5948 | 0.8219 / 0.5988 | +0.0058 / +0.0040 | 未跑 | 未跑 |

### K=200 表

| center | baseline | real-anchor LH-AT | delta | no-token | target-token |
|---|---:|---:|---:|---:|---:|
| ningbo | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| chapman_shaoxing | 0.8737 / 0.4381 | 0.8884 / 0.4641 | +0.0147 / +0.0260 | 未跑 | 未跑 |
| cpsc_2018 | 0.8139 / 0.5662 | 0.8403 / 0.5862 | +0.0264 / +0.0200 | 未跑 | 未跑 |
| georgia | 0.8156 / 0.5930 | 0.8246 / 0.6036 | +0.0090 / +0.0105 | 未跑 | 未跑 |

### K=300 表

| center | baseline | real-anchor LH-AT | delta | no-token | target-token |
|---|---:|---:|---:|---:|---:|
| ningbo | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| chapman_shaoxing | 0.8736 / 0.4374 | 0.8911 / 0.4702 | +0.0175 / +0.0328 | 未跑 | 未跑 |
| cpsc_2018 | 0.8131 / 0.5636 | 0.8461 / 0.5896 | +0.0329 / +0.0260 | 未跑 | 未跑 |
| georgia | 0.8160 / 0.5931 | 0.8265 / 0.6059 | +0.0105 / +0.0128 | 未跑 | 未跑 |

### K=400 表

| center | baseline | real-anchor LH-AT | delta | no-token | target-token |
|---|---:|---:|---:|---:|---:|
| ningbo | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| chapman_shaoxing | 0.8760 / 0.4318 | 0.8950 / 0.4695 | +0.0190 / +0.0378 | 未跑 | 未跑 |
| cpsc_2018 | 0.8127 / 0.5614 | 0.8519 / 0.5943 | +0.0393 / +0.0330 | 未跑 | 未跑 |
| georgia | 0.8159 / 0.5925 | 0.8262 / 0.6039 | +0.0103 / +0.0115 | 未跑 | 未跑 |

### K=500 表

| center | baseline | real-anchor LH-AT | delta | no-token + LH-AT | target-token + LH-AT |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8918 / 0.5234 | +0.0246 / +0.0450 | 0.8879 / 0.5156 | 0.8881 / 0.5155 |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4638 | +0.0203 / +0.0387 | 0.8972 / 0.4653 | 0.8972 / 0.4647 |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8544 / 0.5962 | +0.0430 / +0.0376 | 0.8541 / 0.5958 | 0.8543 / 0.5959 |
| georgia | 0.8157 / 0.5916 | 0.8256 / 0.6020 | +0.0100 / +0.0104 | 0.8126 / 0.5861 | 0.8262 / 0.6029 |

K sweep 产物：

```text
/root/autodl-tmp/paper_latenthull_grid_20260512/summaries/real_anchor_k_lambda_grid_with_baseline.csv
/root/autodl-tmp/paper_latenthull_grid_20260512/summaries/real_anchor_k_lambda_grid_with_baseline.md
```

补跑产物：

```text
ningbo real-anchor K500 lambda=0.15:
  /root/autodl-tmp/paper_latenthull_grid_20260512/runs/ningbo_realK500_lambda0p15_seed20260531/

ningbo baseline with same excluded refs:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_exclrefs/baseline_ningbo_realanchor_v1_k500_exclrefs.json
```

注意：`ningbo` 当前只有 `real_anchor_selected_v1` 的 K=500 artifact，本轮补跑明确
使用这个 v1 ref set；完整 eval 已排除这 500 条 ref ids。

### CPSC Lambda 加大和 No-Lambda 混合池

本表固定目标中心 `cpsc_2018`，均排除对应 500 条 target refs。

| method | pool | hull_lambda | target cpsc_2018 AUROC/AUPRC | PN2021 avg AUROC/AUPRC | PTB-XL AUROC/AUPRC |
|---|---|---:|---:|---:|---:|
| baseline | bare PTB-XL | - | 0.8115 / 0.5586 | 0.7774 / 0.4813 | 0.9072 / 0.7744 |
| real-anchor LH-AT | real K500 | 0.15 | 0.8544 / 0.5962 | 0.7833 / 0.4812 | 0.9050 / 0.7708 |
| real-anchor LH-AT | real K500 | 0.50 | 0.8546 / 0.5975 | 0.7824 / 0.4803 | 0.9050 / 0.7707 |
| real-anchor LH-AT | real K500 | 0.70 | 0.8547 / 0.5976 | 0.7825 / 0.4804 | 0.9050 / 0.7707 |
| real-anchor LH-AT | real K500 | 0.80 | 0.8547 / 0.5977 | 0.7825 / 0.4803 | 0.9050 / 0.7707 |
| mixed target-token LH-AT | real K500 + prompt-token synth | 0.15 | 0.8543 / 0.5959 | 0.7824 / 0.4804 | 0.9052 / 0.7711 |
| mixed target-token no-lambda | real K500 + prompt-token synth | 1.00 | 0.8447 / 0.5739 | 0.7702 / 0.4564 | 0.8809 / 0.7181 |

解释：

```text
hull_lambda=1.0 等价于去掉 z0 插值项：
  z_adv = sum_i softmax(a_i) z_i

该版本确实混合 real_anchor 和 prompt_token generated ECG，并优化 softmax(a_i)
组合系数，但 full eval 明显低于 lambda=0.15 主线。
```

当前结论：

```text
1. 在 CPSC 上把 lambda 从 0.15 加到 0.5/0.7/0.8，target AUPRC 有极小提升
   0.5962 -> 0.5977，但 PN2021 avg 没有提升。
2. lambda=1.0 的 no-lambda 混合池会明显伤害 PTB-XL 保持性和 PN2021 平均性能。
3. 因此主线仍推荐保留小 lambda 约束，例如 0.15；高 lambda 只能作为 CPSC
   target-center 小幅 ablation，不宜作为默认方案。
```

### Four-Center Lambda 0.8 复跑

本轮把 real-anchor LH-AT 的 `hull_lambda` 从主线 `0.15` 调到 `0.8`，
其余训练参数不变。所有 target-center 评估均排除对应 K=500 target refs。

| center | baseline target | lambda=0.15 target | lambda=0.8 target | delta 0.8 - baseline | delta 0.8 - 0.15 | lambda=0.8 PN2021 avg | lambda=0.8 PTB-XL |
|---|---:|---:|---:|---:|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8918 / 0.5234 | 0.8919 / 0.5233 | +0.0247 / +0.0449 | +0.0001 / -0.0001 | 0.7804 / 0.4915 | 0.9038 / 0.7668 |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4638 | 0.8970 / 0.4645 | +0.0207 / +0.0394 | +0.0004 / +0.0007 | 0.7795 / 0.4898 | 0.9035 / 0.7653 |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8544 / 0.5962 | 0.8547 / 0.5977 | +0.0432 / +0.0391 | +0.0003 / +0.0014 | 0.7825 / 0.4803 | 0.9050 / 0.7707 |
| georgia | 0.8157 / 0.5916 | 0.8256 / 0.6020 | 0.8257 / 0.6020 | +0.0100 / +0.0105 | +0.0001 / +0.0000 | 0.7766 / 0.4776 | 0.9050 / 0.7715 |

产物：

```text
/root/autodl-tmp/paper_latenthull_grid_20260512/runs/ningbo_realK500_lambda0p8_seed20260531/
/root/autodl-tmp/paper_latenthull_grid_20260512/runs/chapman_shaoxing_realK500_lambda0p8_seed20260531/
/root/autodl-tmp/paper_latenthull_grid_20260512/runs/cpsc_2018_realK500_lambda0p8_seed20260531/
/root/autodl-tmp/paper_latenthull_grid_20260512/runs/georgia_realK500_lambda0p8_seed20260531/
```

结论：

```text
lambda=0.8 相比 lambda=0.15 几乎持平。

target-center:
  ningbo:  -0.0001 AUPRC
  chapman: +0.0007 AUPRC
  cpsc:    +0.0014 AUPRC
  georgia: +0.0000 AUPRC

因此 lambda=0.8 可以作为稳健性 ablation，说明方法对 lambda 不敏感；
但它没有带来足够明确的新收益，不建议替换主线 lambda=0.15。
```

### Center-Token Ablation With Lambda 0.8

为回答 `no-token + LH-AT` vs `target-token + LH-AT`，本轮固定：

```text
hull_lambda = 0.8
K target real refs = 500
attack = latent_hull
M = 10
same-label latent mixing
strict eval = minimal_resample + per_sample_global + 100Hz + 10s
target eval excludes each center's K=500 refs
```

注意：这一节不是 real-anchor-only LH-AT，而是把 no-token 或 target-token
ECGTwin latent candidates 加入 anchor pool 后做 LH-AT，因此它专门用于检验
center token 是否优于 no-token synthetic candidates。

| center | no-token target | target-token target | target-token - no-token | no-token PN2021 avg | target-token PN2021 avg | PTB-XL no-token -> target-token |
|---|---:|---:|---:|---:|---:|---:|
| ningbo | 0.8720 / 0.4879 | 0.8721 / 0.4884 | +0.0000 / +0.0005 | 0.7688 / 0.4681 | 0.7696 / 0.4694 | 0.8848 / 0.7261 -> 0.8848 / 0.7262 |
| chapman_shaoxing | 0.8818 / 0.4289 | 0.8816 / 0.4287 | -0.0001 / -0.0002 | 0.7673 / 0.4657 | 0.7680 / 0.4655 | 0.8806 / 0.7167 -> 0.8806 / 0.7166 |
| cpsc_2018 | 0.8448 / 0.5743 | 0.8445 / 0.5738 | -0.0003 / -0.0005 | 0.7700 / 0.4564 | 0.7698 / 0.4562 | 0.8807 / 0.7176 -> 0.8809 / 0.7181 |
| georgia | 0.8132 / 0.5865 | 0.8131 / 0.5864 | -0.0001 / -0.0001 | 0.7651 / 0.4567 | 0.7648 / 0.4563 | 0.8818 / 0.7181 -> 0.8819 / 0.7184 |

平均差值：

```text
target-center macro delta:
  AUROC -0.000131
  AUPRC -0.000080

PN2021 7-center avg delta:
  AUROC +0.000270
  AUPRC +0.000133
```

产物：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_lam08_no_vs_target_summary.json
/root/autodl-tmp/ecgtwin_prompt_token_super5/repro_seed_checks/ningbo_v45_no_token_seed20260531_w20_src10_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/repro_seed_checks/ningbo_v45_target_seed20260531_w20_src10_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/chapman_v42_vanilla_w40_src04_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/chapman_v42_target_w40_src04_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/cpsc_v42_vanilla_w40_src04_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/cpsc_v42_target_w40_src04_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/georgia_v42_vanilla_w40_src04_M10_lam08_adv006_softmix03_ep10/
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v2/georgia_v42_target_w40_src04_M10_lam08_adv006_softmix03_ep10/
```

结论：

```text
lambda=0.8 下，target-token 相对 no-token 没有形成可解释的稳定收益。
四个 target-center 的平均差值约等于 0，PN2021 7-center 平均差值也只有
约 +0.0001 AUPRC，属于噪声级。

更重要的是，mixed token pool 在 lambda=0.8 下的绝对性能明显低于
lambda=0.15 的 matched token runs，也低于 real-anchor-only LH-AT。
这说明高 lambda 会让对抗样本过度靠近候选 latent pool，削弱真实 target
anchor 的约束；它不适合作为 center-token 主线设置。

当前推荐：
  1. real-anchor-only LH-AT 仍是最稳定主线。
  2. center-token 只保留为 synthetic candidate / mechanism ablation。
  3. mixed no-token vs target-token 的默认 lambda 仍用 0.15，不推荐 0.8。
```

### No-Gate Target-Only 30-Epoch Pilot

用户要求尝试更激进的 ablation：

```text
center = ningbo
target real refs = K500
n_epochs = 30
quality / semantic gate = disabled
PTB-XL stream = disabled
PTB-XL roundtrip stream = disabled
training streams = target real K500 + online adv buffer only
hull_M = 10
hull_lambda = 0.15
K_anchor = 300 / epoch
crop_len = 1000
```

为支持该 ablation，`scripts/pgd_cross_center/synth_online_at_super5.py`
新增 `--disable_quality_gate`，并修复 `ptbxl_weight=0` 时单 stream fallback
错误回到 PTB-XL `train_ds` 的问题。现在 `ptbxl_weight=0` 和
`roundtrip_weight=0` 会真正从训练 sampler 中移除 PTB-XL / roundtrip stream。

结果：

| model | PTB-XL fold10 | PN2021 7-center avg | held-out ningbo |
|---|---:|---:|---:|
| baseline | 0.9072 / 0.7744 | 0.7782 / 0.4823 | 0.8672 / 0.4784 |
| main real-anchor LH-AT, gate on, PTB-XL mixed, 10 ep | 0.9039 / 0.7669 | 0.7803 / 0.4908 | 0.8918 / 0.5234 |
| no-gate target-only, no PTB-XL, 30 ep | 0.8826 / 0.7206 | 0.7661 / 0.4669 | 0.8687 / 0.4816 |

产物：

```text
/root/autodl-tmp/paper_latenthull_ablation_no_gate_targetonly_20260512/ningbo_realK500_nogate_noPTBXL_ep30_crop1000_seed20260531/
```

结论：

```text
这个设置不值得扩展到其他中心。

held-out ningbo 只比 baseline 高 +0.0015 / +0.0032，
远低于主线 real-anchor LH-AT 的 +0.0246 / +0.0450。

同时 PTB-XL fold10 从 0.9072 / 0.7744 掉到 0.8826 / 0.7206，
说明去掉 PTB-XL stream 和 roundtrip stream 后出现明显 source forgetting。

质量 gate 在本 run 中没有成为主要瓶颈，因为 Einthoven p95 仍然较低；
主要负面因素是 target-only 训练和 30 epoch 过度适配。
主线应继续保留 PTB-XL mixed stream、roundtrip stream 和 gate。
```

## 下一轮复跑计划

第一阶段：复核 source baseline。

```text
R0:
  复用 /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503
  重新跑 PN2021 clean eval，确认 minimal_resample / per_sample_global / crop1000 口径。
```

第二阶段：复跑 ningbo matched control。

```text
R1:
  target-center real K=500 + no-token ECGTwin latent candidates + Latent-Hull AT

R2:
  target-center real K=500 + target-token ECGTwin latent candidates + Latent-Hull AT

目标:
  确认二者都 beat baseline，同时量化 target-token - no-token 差值。
```

第三阶段：复跑多中心 target-token。

```text
R3:
  chapman_shaoxing target-token + Latent-Hull AT

R4:
  cpsc_2018 target-token + Latent-Hull AT

R5:
  georgia target-token + Latent-Hull AT，作为困难中心。
```

第四阶段：如果 center-token 因果性仍不明显，再改 token / selection。

```text
可尝试:
  token_scale = 0.25 / 0.50 / 0.75
  wrong-center negative selection
  no-leak style C2ST gate
  feature MMD/FID target-likeness gate
  diversity-preserving candidate selection
```

## 新复跑输出路径

不要覆盖历史实验，后续统一写到：

```text
/root/autodl-tmp/paper_center_token_latenthull_20260512/
  baseline_eval/
  center_token/
  candidate_pools/
  online_at/
    ningbo_no_token/
    ningbo_target_token/
    chapman_target_token/
    cpsc_target_token/
    georgia_target_token/
  summaries/
    per_center_delta.csv
    token_vs_notoken.csv
    paper_pipeline_report.md
```

## 论文级验收标准

最低要求：

```text
1. R0 确认 baseline eval 口径正确。
2. R1/R2 都排除 K=500 target refs。
3. R1/R2 都报告 AUROC 和 AUPRC。
4. 至少 ningbo 复现相对 bare PTB-XL baseline 的 AUPRC >= +2pp。
5. target-token vs no-token 结果必须如实报告。
```

更强的 center-token 论文贡献要求：

```text
1. target-token AT 在至少两个中心明显超过 no-token AT。
2. wrong-center token 明显弱于 target-center token。
3. center-style metrics 支持 target-token synthetic ECG 更接近目标中心真实 ECG。
```

如果强 center-token claim 不成立：

```text
论文主方法应表述为 target-center real-anchor ECGTwin VAE latent online AT。
center token 作为可控候选生成模块，而不是唯一因果来源。
```

## 2026-05-12 Real-Anchor K/Lambda 稳定性复跑

本轮针对论文主线里更稳的 real-anchor Latent-Hull AT 做复跑，目标是回答：

```text
1. chapman_shaoxing / cpsc_2018 / georgia 三个中心是否仍有提升？
2. 目标中心真实 anchor 数量 K = 100/200/300/400/500 对结果有什么影响？
3. Latent-Hull lambda = 0.05/0.10/0.15/0.25/0.35 是否敏感？
4. adaptation 使用的目标中心 ref ids 是否从测试集中排除？
```

固定设置：

```text
source checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

source classifier eval protocol:
  preprocess_mode = minimal_resample
  norm_mode       = per_sample_global
  ptbxl_cache     = /root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
  pn2021 mmap     = /root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal

online AT:
  attack_mode          = latent_hull
  hull_M               = 10
  hull_steps           = 5
  hull_lr              = 0.25
  hull_weight_mode     = optimized
  adv_label_mode       = mixed_soft
  adv_teacher_mix      = 0.30
  adv_weight           = 0.06
  target_real_weight   = 40
  roundtrip_anchor_n   = 1500
  epochs               = 10
  classes_in_scope     = NORM, MI, STTC
```

实现和产物：

```text
script:
  scripts/paper/run_latenthull_real_anchor_grid_20260512.py

run root:
  /root/autodl-tmp/paper_latenthull_grid_20260512/

summary:
  /root/autodl-tmp/paper_latenthull_grid_20260512/summaries/real_anchor_k_lambda_grid_with_baseline.csv
  /root/autodl-tmp/paper_latenthull_grid_20260512/summaries/real_anchor_k_lambda_grid_with_baseline.md

baseline excluded-ref eval:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_exclrefs/paper_grid_20260512/
```

注意：本轮所有 target-center 指标都使用对应 K 的 `ref_meta.json` 做
`--exclude_ref_ids`。因此 K 条 adaptation anchor 不会进入该中心测试集。

### K sweep 结果

下表为 `lambda=0.15`。`delta` 是相对同一中心、同一 K、同一 excluded-ref
测试子集上的 bare PTB-XL baseline。

| center | K | target AUROC/AUPRC | baseline target AUROC/AUPRC | delta |
|---|---:|---:|---:|---:|
| chapman_shaoxing | 100 | 0.8814 / 0.4494 | 0.8725 / 0.4365 | +0.0088 / +0.0129 |
| chapman_shaoxing | 200 | 0.8884 / 0.4641 | 0.8737 / 0.4381 | +0.0147 / +0.0260 |
| chapman_shaoxing | 300 | 0.8911 / 0.4702 | 0.8736 / 0.4374 | +0.0175 / +0.0328 |
| chapman_shaoxing | 400 | 0.8950 / 0.4695 | 0.8760 / 0.4318 | +0.0190 / +0.0378 |
| chapman_shaoxing | 500 | 0.8967 / 0.4638 | 0.8763 / 0.4251 | +0.0203 / +0.0387 |
| cpsc_2018 | 100 | 0.8240 / 0.5789 | 0.8143 / 0.5687 | +0.0097 / +0.0102 |
| cpsc_2018 | 200 | 0.8403 / 0.5862 | 0.8139 / 0.5662 | +0.0264 / +0.0200 |
| cpsc_2018 | 300 | 0.8461 / 0.5896 | 0.8131 / 0.5636 | +0.0329 / +0.0260 |
| cpsc_2018 | 400 | 0.8519 / 0.5943 | 0.8127 / 0.5614 | +0.0393 / +0.0330 |
| cpsc_2018 | 500 | 0.8544 / 0.5962 | 0.8115 / 0.5586 | +0.0430 / +0.0376 |
| georgia | 100 | 0.8219 / 0.5988 | 0.8162 / 0.5948 | +0.0058 / +0.0040 |
| georgia | 200 | 0.8246 / 0.6036 | 0.8156 / 0.5930 | +0.0090 / +0.0105 |
| georgia | 300 | 0.8265 / 0.6059 | 0.8160 / 0.5931 | +0.0105 / +0.0128 |
| georgia | 400 | 0.8262 / 0.6039 | 0.8159 / 0.5925 | +0.0103 / +0.0115 |
| georgia | 500 | 0.8256 / 0.6020 | 0.8157 / 0.5916 | +0.0100 / +0.0104 |

结论：

```text
chapman_shaoxing:
  AUROC 随 K 增加基本单调提升。
  AUPRC 在 K=400/500 区间较好，K=500 的 delta 最大，但绝对 AUPRC 比 K=300/400 略低。

cpsc_2018:
  K 增加带来最清楚的收益。
  K=500 达到 target +4.30pp AUROC / +3.76pp AUPRC。

georgia:
  有稳定小幅提升，但明显弱于 chapman/cpsc。
  K=300 左右最好，继续到 K=500 没有额外收益。
```

### Lambda sweep 结果

下表固定 `K=500`。

| center | lambda | target AUROC/AUPRC | delta |
|---|---:|---:|---:|
| chapman_shaoxing | 0.05 | 0.8967 / 0.4641 | +0.0204 / +0.0389 |
| chapman_shaoxing | 0.10 | 0.8967 / 0.4639 | +0.0203 / +0.0387 |
| chapman_shaoxing | 0.15 | 0.8967 / 0.4638 | +0.0203 / +0.0387 |
| chapman_shaoxing | 0.25 | 0.8967 / 0.4639 | +0.0203 / +0.0388 |
| chapman_shaoxing | 0.35 | 0.8969 / 0.4641 | +0.0206 / +0.0390 |
| cpsc_2018 | 0.05 | 0.8542 / 0.5961 | +0.0427 / +0.0375 |
| cpsc_2018 | 0.10 | 0.8543 / 0.5960 | +0.0428 / +0.0374 |
| cpsc_2018 | 0.15 | 0.8544 / 0.5962 | +0.0430 / +0.0376 |
| cpsc_2018 | 0.25 | 0.8545 / 0.5962 | +0.0430 / +0.0376 |
| cpsc_2018 | 0.35 | 0.8545 / 0.5968 | +0.0430 / +0.0382 |
| georgia | 0.05 | 0.8257 / 0.6021 | +0.0100 / +0.0105 |
| georgia | 0.10 | 0.8256 / 0.6020 | +0.0100 / +0.0104 |
| georgia | 0.15 | 0.8256 / 0.6020 | +0.0100 / +0.0104 |
| georgia | 0.25 | 0.8257 / 0.6021 | +0.0100 / +0.0105 |
| georgia | 0.35 | 0.8255 / 0.6017 | +0.0098 / +0.0101 |

结论：

```text
lambda 在 0.05-0.35 范围内不敏感。
当前主线可以继续固定 lambda=0.15，把调参重点放到：
  K / target_real_weight / class coverage / token-vs-no-token candidate quality。
```

### 重要限制

```text
1. 本轮是 real-anchor Latent-Hull AT，主要验证 target-center anchor 数量和 lambda，
   不是 center-token 因果性验证。

2. cpsc_2018 和 georgia 的 real anchors 中 MI 极少。
   cpsc_2018 K=500 的 latent-hull 实际主要由 NORM/STTC 参与。
   georgia K=500 也只有很少 MI anchor。

3. 源域 PTB-XL fold10 指标随 K 增大略降：
   这是目标中心适配和源域保持之间的 trade-off，不应忽略。
```

## 2026-05-12 标签映射复核

当前标签策略可以继续用于论文主线，但论文表述必须精确：

```text
PTB-XL super5:
  official diagnostic superclass labels from scp_statements.csv diagnostic_class.

PN2021 super5:
  project-defined SNOMED-to-PTB-XL-super5 semantic projection.
  不是 PN2021 官方 crosswalk。
```

固定类顺序：

```text
CD, HYP, MI, NORM, STTC
```

当前 PN2021 映射版本：

```text
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
PN2021_EVAL_CACHE_VERSION     = v3_super5_normsuppress
mapping_hash                  = 544ed42dee6d
```

当前 v3 规则重点：

```text
1. SNOMED_TO_SUPER5_POSITIVE 只放明确异常类 CD/HYP/MI/STTC。
2. NORM_POSITIVE_SNOMEDS 只把 strict sinus rhythm 作为 NORM 候选。
3. NORM_SUPPRESS_SNOMEDS 把 AF/AFL/PAC/PVC/axis/low-voltage/boundary 等
   不应视作 PTB-XL NORM 的 code 设为 suppress-only。
4. 如果存在异常阳性类或 suppress-only code，则 NORM=0。
```

本轮 spot check：

```text
sinus rhythm only        -> NORM=1
sinus rhythm + MI        -> MI=1, NORM=0
sinus rhythm + AF        -> all-zero super5, NORM=0
Q wave abnormal only     -> all-zero super5, NORM=0
early repolarization only-> all-zero super5, NORM=0
LBBB                     -> CD=1
LVH                      -> HYP=1
ST depression            -> STTC=1
```

PTB-XL 全量 super5 计数：

```text
N = 21799
CD   = 4898
HYP  = 2649
MI   = 5469
NORM = 9514
STTC = 5235
all-zero diagnostic-superclass records = 411
```

fold 10 计数：

```text
N = 2198
CD   = 496
HYP  = 262
MI   = 550
NORM = 963
STTC = 521
all-zero diagnostic-superclass records = 40
```

结论：

```text
标签映射当前没有发现会阻塞论文主线的问题。
PTB-XL 标签可称为 official diagnostic superclass。
PN2021 标签必须称为 custom semantic projection，并报告 mapping version/hash。
```

## 2026-05-12 Ningbo seed 复现和 no-token 消融

本轮复现目标：

```text
确认 Latent-Hull online AT 是否在换 seed 后仍有增益；
确认 target-center token 是否明显强于 matched no-token。
```

共同设置：

```text
center            = ningbo
seed              = 20260531
init_ckpt         = super5_minresample_full10_perglobal_20260503/best_model.pt
attack_mode       = latent_hull
hull_M            = 10
hull_lambda       = 0.15
hull_steps        = 5
hull_lr           = 0.25
hull_weight_mode  = optimized
hull_label_mode   = primary
K_anchor          = 300 per epoch
n_epochs          = 10
adv_weight        = 0.06
target_real_weight= 20.0
ptbxl_weight      = 1.0
roundtrip_weight  = 0.5
crop_len          = 1000
classes_in_scope  = NORM, MI, STTC
target refs       = K=500, excluded from full eval
```

Latent-Hull 公式：

```text
z_mix = sum_i softmax(a_i) * z_i, i = 1..M
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

这里 `hull_M=10`，所以每个 anchor `z0` 的对抗样本由 10 个同主类候选 latent
参与线性组合。`a_i` 是在线优化的混合系数 logits；优化 5 step 后用 softmax 得到
非负且和为 1 的 convex weights。

复现输出：

```text
target-token:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/repro_seed_checks/
    ningbo_v45_target_seed20260531_w20_src10_M10_lam015_adv006_softmix03_ep10

no-token:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/repro_seed_checks/
    ningbo_v45_no_token_seed20260531_w20_src10_M10_lam015_adv006_softmix03_ep10
```

完整 PN2021 held-out 评测结果：

| model | PTB-XL AUROC | PTB-XL AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC | Ningbo AUROC | Ningbo AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| bare PTB-XL baseline | 0.9072 | 0.7744 | 0.7780 | 0.4831 | 0.8657 | 0.4842 |
| target-token + Latent-Hull AT | 0.8959 | 0.7380 | 0.7800 | 0.4881 | 0.8881 | 0.5155 |
| no-token + Latent-Hull AT | 0.8958 | 0.7377 | 0.7812 | 0.4892 | 0.8879 | 0.5156 |

本轮结论：

```text
1. Latent-Hull online AT 相对 bare PTB-XL baseline 仍有跨中心增益：
   target-token: PN2021 avg +0.20pp AUROC / +0.50pp AUPRC；
   no-token:     PN2021 avg +0.32pp AUROC / +0.61pp AUPRC。

2. 对目标中心 Ningbo 的提升明显：
   target-token: +2.24pp AUROC / +3.13pp AUPRC；
   no-token:     +2.22pp AUROC / +3.14pp AUPRC。

3. 本轮 target-token 没有超过 no-token：
   PN2021 avg target-token 比 no-token 低 0.12pp AUROC / 0.11pp AUPRC；
   Ningbo target-token 与 no-token 基本持平。

4. 因此当前证据支持：
   real-anchor + ECGTwin VAE Latent-Hull online AT 是有效主线；
   center token 在这个 Ningbo v45 配置中还不能作为独立因果增益主张。
```

## 2026-05-12 Ningbo Real-Anchor LH-AT 的 epoch / M grid

本轮 grid 用来回答两个问题：

```text
1. 在线对抗训练 epoch 从 10 增加到 20/30 是否继续提升；
2. 每个 anchor 参与 latent-hull 线性组合的候选样本数 M 从 10 扩到
   20/30/40/50/100 是否继续提升。
```

修正后的主实验必须满足：

```text
train crop_len = 1000
eval  crop_len = 1000
```

如果 eval 不显式传入 `--crop_len 1000`，`eval_crosscenter.py` 会回退到
250 sample 默认值，导致 PTB-XL 与 PN2021 指标被错误低估。因此旧的
crop250 结果不得用于论文表格。

共同设置：

```text
center             = ningbo
target K           = 500, strict held-out eval excludes these refs
init_ckpt          = super5_minresample_full10_perglobal_20260503/best_model.pt
attack_mode        = latent_hull
hull_lambda        = 0.15
hull_steps         = 5
hull_lr            = 0.25
hull_weight_mode   = optimized
hull_label_mode    = primary
K_anchor           = 300 per epoch
source_strategy    = source_weighted
target_real_weight = 40.0
ptbxl_weight       = 1.0
roundtrip_anchor_n = 1500
roundtrip_weight   = 0.5
adv_weight         = 0.06
adv_label_mode     = mixed_soft
adv_teacher_mix    = 0.3
classes_in_scope   = NORM, MI, STTC
seed               = 20260531
```

结果目录：

```text
/root/autodl-tmp/paper_latenthull_mscale_exact_20260512/
  ningbo_realK500_lambda015_sourceweighted/
```

汇总文件：

```text
summary_crop1000.md
summary_crop1000.csv
summary_crop1000.json
```

baseline：

```text
PTB-XL fold10      = 0.9072 / 0.7744
PN2021 avg         = 0.7782 / 0.4823
held-out Ningbo    = 0.8672 / 0.4784
```

主要结果：

| rank | setting | held-out Ningbo AUROC/AUPRC | Ningbo delta | PN2021 avg AUROC/AUPRC | PTB-XL AUROC/AUPRC |
|---:|---|---:|---:|---:|---:|
| 1 | M20, ep30 | 0.8945 / 0.5303 | +2.73pp / +5.20pp | 0.7826 / 0.4917 | 0.9024 / 0.7633 |
| 2 | M10, ep30 | 0.8946 / 0.5303 | +2.73pp / +5.20pp | 0.7826 / 0.4918 | 0.9023 / 0.7633 |
| 3 | M20, ep20 | 0.8942 / 0.5295 | +2.70pp / +5.11pp | 0.7821 / 0.4923 | 0.9026 / 0.7638 |
| 4 | M30, ep20 | 0.8942 / 0.5294 | +2.70pp / +5.11pp | 0.7822 / 0.4926 | 0.9026 / 0.7638 |
| 5 | M100, ep20 | 0.8944 / 0.5294 | +2.72pp / +5.10pp | 0.7813 / 0.4916 | 0.9024 / 0.7634 |
| 6 | M40, ep20 | 0.8944 / 0.5294 | +2.71pp / +5.10pp | 0.7813 / 0.4916 | 0.9024 / 0.7633 |
| 7 | M50, ep20 | 0.8944 / 0.5293 | +2.71pp / +5.09pp | 0.7813 / 0.4916 | 0.9024 / 0.7634 |
| 8 | M10, ep20 | 0.8944 / 0.5293 | +2.71pp / +5.09pp | 0.7812 / 0.4916 | 0.9024 / 0.7634 |
| 9 | M30, ep10 | 0.8918 / 0.5234 | +2.46pp / +4.50pp | 0.7803 / 0.4909 | 0.9039 / 0.7670 |
| 10 | M10, ep10 | 0.8918 / 0.5234 | +2.46pp / +4.50pp | 0.7803 / 0.4909 | 0.9039 / 0.7669 |

结论：

```text
1. 旧的 M10/ep10 主结果可以在 crop_len=1000 的严格 eval 下复现：
   held-out Ningbo 约 0.8918 / 0.5234。

2. 从 ep10 增加到 ep20/ep30 是主要收益来源：
   held-out Ningbo AUPRC 约从 0.5234 提升到 0.529-0.530。

3. M 从 10 扩到 20/30/40/50/100 没有稳定额外收益：
   同一 epoch 下，各 M 指标几乎重合。

4. 推荐论文默认配置：
   M=20，max_epochs=30，patience=10；
   实际 best 通常出现在 epoch 16-20 附近。

5. 如果优先节省算力：
   M=10, ep20 已经非常接近最优；
   M=50/100 不建议作为默认，因为收益不明显但显存/耗时更高。
```
