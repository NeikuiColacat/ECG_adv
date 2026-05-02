# Center Style Classifier Pipeline

本文档定义如何构造、训练和测试中心数据风格分类器，用于验证 ECGTwin center prompt token
生成的 ECG 是否更接近目标 PN2021 中心风格。

该分类器不是最终诊断模型。它的作用是做 domain probe：

```text
真实目标中心 ECG 是否和其他中心 ECG 在风格上可分？
加入 center token 后的 synthetic ECG 是否更像目标中心？
```

## 1. 小白版：什么是中心风格分类器

ECG 不同中心之间可能有很多差异：

```text
设备厂商
采样率
滤波设置
电极贴附习惯
病人群体
标签体系
噪声和基线漂移
导联幅值分布
```

这些差异不一定是疾病本身，但会影响模型泛化。中心风格分类器就是让模型回答：

```text
这条 ECG 看起来更像来自哪个中心？
```

如果一个只用真实 ECG 训练的中心判别器认为：

```text
target-center-token synthetic 比 vanilla synthetic 更像目标中心，
```

那么说明 center token 至少在可观测特征空间里产生了目标中心风格迁移。

## 2. 旧版现状

已有脚本：

```text
scripts/ecgtwin_gen/train_per_center_style_classifier.py
scripts/ecgtwin_gen/eval_style_classifier_on_synth.py
```

旧版设置：

```text
task: 7-way PN2021 center classification
centers:
  chapman_shaoxing
  cpsc_2018
  cpsc_2018_extra
  georgia
  ningbo
  ptb
  st_petersburg_incart

input: (B,12,250)
backbone: EfficientNet1DV2 s_v2
training data: real PN2021 ECG only
preprocess: unified_preprocess_to_1000 + crop
```

旧版结果：

```text
checkpoint: /root/autodl-tmp/per_center_style_classifier/best_model.pt
test accuracy: 0.3780
macro F1:      0.3624
chance:        1/7 = 0.1429
```

这个结果说明 PN2021 center style 可被部分识别，但旧版不能作为正式结论：

- 没有保存 `record_ids`；
- 没有排除 center-token K=500 anchors；
- 没有 same-label 控制；
- 只用 250 点 crop，可能漏掉全局中心风格；
- `ptb` 和 `st_petersburg_incart` 样本太少，7-way 分类不平衡。

## 3. 正式验证目标

正式目标不是刷高 7-way accuracy，而是构造一个 no-leak、same-label、可解释的风格验证栈。

需要回答：

```text
1. 目标中心真实 ECG 与非目标中心真实 ECG 是否可分？
2. target-center-token synthetic 是否比 vanilla synthetic 更像目标中心？
3. target-center-token synthetic 是否更接近 held-out target real 分布？
4. 这种风格接近是否能转化为下游 AUROC/AUPRC 收益？
```

## 4. 数据来源

主训练数据：

```text
PN2021 clean v3 cache
/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap/
```

优先目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

优先类别：

```text
NORM
MI
STTC
```

`CD` 和 `HYP` 先作为 exploratory，因为当前 synthetic digital gate 更不稳定。

生成样本：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_*/<center>/gated/gated_samples.npz
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_*/<center>/gated/gated_samples.latent.npz
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_*/<center>/gated/gated_samples.ref_meta.json
```

K=500 ref selection：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/ref_selection/<center>_k500_seed42.json
```

## 5. 防泄漏 split

对每个目标中心 `c`：

```text
A_c_anchor = K=500 center-token 训练/生成 reference ECG
D_c_real   = PN2021 中除 A_c_anchor 外的真实 ECG
G_c        = center-token synthetic ECG
```

强制规则：

1. `A_c_anchor` 不进入判别器 train/val/test。
2. `G_c` 不进入判别器训练和阈值选择，只能做 query/eval。
3. 判别器 split 保存 `record_ids`，便于审计。
4. 同中心下游 eval 也要排除 `A_c_anchor`。

推荐 split：

```text
target center remaining real:
  train 60%
  val   20%
  test  20%

non-target centers:
  train/val/test 按同样规则
```

如果未来能可靠获得 patient id，则优先 patient-level split；当前 PN2021 多数中心只能按 record id 固定划分。

## 6. Same-Label 采样

判别器不能直接把所有样本混起来训练，否则它可能学到“某中心 MI 更多”这类标签分布差异。

训练 batch 应尽量保证：

```text
target positive : non-target negative = 1 : 1
same-label positive : same-label negative = matched
negative centers = center-balanced
```

优先匹配：

1. 精确 super5 multi-hot label；
2. 不足时用 primary class；
3. NORM 只和纯 NORM 比；
4. 每个 center-class test positives 少于 30 时只报告 exploratory。

## 7. 模型构造

### 7.1 Probe A：EfficientNet Feature Classifier

这是推荐主结果。

流程：

```text
ECG -> EfficientNet1DV2 baseline frozen backbone -> penultimate feature -> small MLP/logistic head
```

输入信号处理：

```text
raw ECG
  -> classifier preprocessing
  -> (1000,12)
  -> full 10s or multi-crop
  -> frozen EfficientNet feature
```

分类头：

```text
LayerNorm(d)
Linear d -> 256
GELU
Dropout 0.3
Linear 256 -> 64
GELU
Dropout 0.2
Linear 64 -> target logits
```

优点：

- 和下游 EfficientNetV2 AUROC/AUPRC 空间一致；
- 比 raw waveform 端到端更不容易只学到低级噪声；
- 训练快，适合多中心多种子。

### 7.2 Probe B：ECGTwin VAE Latent Classifier

流程：

```text
ECG -> ECGTwin VAE encode -> latent (4,128) -> flatten 512 -> MLP
```

输入信号处理：

```text
raw mV
1024 points
ECGTwin/MIMIC lead order
no classifier z-score
```

分类头：

```text
LayerNorm(512)
Linear 512 -> 256
GELU
Dropout 0.2
Linear 256 -> 64
GELU
Dropout 0.2
Linear 64 -> target logits
```

优点：

- 直接验证生成模型的 VAE 表征空间；
- 对 center prompt token 和 Latent-Hull 解释更贴合。

### 7.3 Probe C：Digital Feature Classifier

流程：

```text
ECG -> digital feature extractor -> logistic regression / shallow MLP
```

feature 示例：

```text
HR
RR irregularity
PR interval
QRS duration
ST deviation
lead amplitude
lead covariance
Einthoven residual
aVR residual
baseline/noise proxy
```

优点：

- 可解释；
- 能说明模型是否只靠医学/信号统计特征区分中心；
- 可发现 synthetic 是否在 HR/QRS/STT/幅值上偏离目标中心。

### 7.4 Probe D：Raw Waveform Classifier

可作为 sensitivity analysis，不作为主结论。

推荐设置：

```text
input: (B,12,1000)
backbone: SE-ResNet / EfficientNet1DV2 full 10s
metric: balanced accuracy, macro F1, AUROC
```

注意：raw waveform classifier 很容易学到滤波、增益、噪声、padding 等伪中心痕迹。
如果 raw probe 很强但 feature/latent probe 不支持，不能直接说 synthetic 医学风格更像目标中心。

## 8. 训练任务设计

### 8.1 任务一：7-Way Center Sanity Classifier

用途：

```text
确认 PN2021 7 centers 是否整体可分
输出 confusion matrix
找出天然混淆中心
```

推荐改进旧版：

```text
input: full 10s or multi-crop
early stopping: macro F1 / balanced accuracy
sampler: center-balanced
epochs: 80
patience: 15
```

该任务只做 sanity，不做中心 token 主证据。

### 8.2 任务二：Per-Target Binary Classifier

这是主任务。

对每个 target center `c` 单独训练：

```text
positive = real ECG from center c
negative = real ECG from all other centers
```

输出：

```text
P(center = c | x)
```

用这个分数评估：

```text
real_target
real_non_target
vanilla_ecgtwin
target_center_token
wrong_center_token
```

理想结果：

```text
P(target | target_center_token) > P(target | vanilla_ecgtwin)
P(target | target_center_token) 接近 P(target | real_target)
P(target | wrong_center_token) 低于 target_center_token
```

### 8.3 任务三：Real-vs-Synth C2ST

对每个 target center 和 class：

```text
class 0 = held-out target real
class 1 = synthetic target-token
```

如果生成分布接近真实目标中心，C2ST 应难以区分：

```text
C2ST balanced accuracy -> 0.5
C2ST AUROC -> 0.5
Proxy A-distance -> 0
```

同时比较：

```text
C2ST(real_target, vanilla_ecgtwin)
C2ST(real_target, target_center_token)
C2ST(real_target, wrong_center_token)
```

主结论看 target-center-token 是否比 vanilla 更难区分。

## 9. 训练超参数

Feature/latent MLP：

```text
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
batch_size: 256
epochs: 50
patience: 8
loss: BCEWithLogitsLoss for binary, CrossEntropyLoss for 7-way
seeds: 42, 1042, 2026
```

Raw waveform classifier：

```text
optimizer: AdamW
lr: 3e-4 to 1e-3
weight_decay: 1e-2
batch_size: 64-128
epochs: 80
patience: 15
crop_len: 1000 for full 10s, or multi-crop inference
```

DataLoader on current hardware:

```text
num_workers: 4-8
pin_memory: true
persistent_workers: true if num_workers > 0
outputs/cache: /root/autodl-tmp/domain_discriminator/
```

## 10. 评估指标

判别器自身：

```text
balanced accuracy
macro F1
AUROC
AUPRC
per-center recall
confusion matrix
calibration: Brier / ECE
```

Synthetic target-likeness：

```text
mean P(target)
median P(target)
target-likeness percentile
% synthetic above real-target median
bootstrap 95% CI
```

C2ST：

```text
balanced accuracy
AUROC
Proxy A-distance = 4 * balanced_accuracy - 2
permutation/binomial p-value
```

Feature distribution：

```text
MMD
KID
FID/rFID
energy distance
precision/recall/coverage
```

Downstream：

```text
PN2021 target center AUROC/AUPRC
PN2021 7-center macro AUROC/AUPRC
PN2021-C corrupted absolute AUROC/AUPRC
```

## 11. 判定规则

可以说 center token 让 ECGTwin 更符合目标中心风格，至少需要：

1. target-center-token 的 `P(target)` 高于 vanilla；
2. target-center-token 的 feature/latent distance 更接近 held-out real target；
3. target-center-token 的 C2ST 不比 vanilla 更容易被区分；
4. digital ECG sanity 不恶化；
5. 下游 target center 或 PN2021 avg AUPRC 不下降，最好上升。

如果只满足前两条，只能说：

```text
center token shifts generated ECGs toward target-center style,
but downstream utility remains unproven.
```

如果 target-likeness 上升但 C2ST 也高：

```text
token learned some target-center attributes, but synthetic artifacts remain.
```

如果 C2ST 低但 target-likeness 不升：

```text
synthetic ECGs look realistic under this probe, but not specifically target-center-like.
```

## 12. 建议脚本

新增：

```text
scripts/ecgtwin_gen/train_prompt_token_domain_discriminator.py
scripts/ecgtwin_gen/eval_prompt_token_domain_discriminator.py
scripts/ecgtwin_gen/compare_prompt_token_target_style.py
```

输出：

```text
/root/autodl-tmp/domain_discriminator/center_style_v1/
```

必须保存：

```text
config.json
split_record_ids.json
train_result.json
training_log.json
best_model.pt
real_probe_scores.csv
synth_target_likeness.csv
c2st_results.csv
feature_distance_results.csv
digital_style_results.csv
validation_report.md
```

## 13. 第一版执行顺序

1. 从 PN2021 v3 mmap cache 读取 clean real ECG 和 record ids。
2. 读取 target center 的 K=500 ref ids 并从 real split 中排除。
3. 为 `ningbo/chapman_shaoxing/cpsc_2018/georgia` 构造 per-target binary split。
4. 提取 EfficientNet frozen feature 和 ECGTwin VAE latent。
5. 训练 feature probe 和 latent probe。
6. 评估 vanilla vs target-center-token vs wrong-center-token synthetic。
7. 生成 target-likeness、C2ST、feature distance 和 digital feature 表。
8. 与下游 PN2021 AUROC/AUPRC delta 合并成最终报告。

## 14. 不要做的事

- 不要用 synthetic 样本训练 real-only center classifier；
- 不要把 K=500 anchors 放进判别器 train/val/test；
- 不要只看 raw 7-way accuracy；
- 不要混合所有 class 后声称 center style；
- 不要把 C2ST 接近 0.5 解释成医学等价；
- 不要把 PTB-XL `ptb-xl` shard 放回 PN2021 center eval。
