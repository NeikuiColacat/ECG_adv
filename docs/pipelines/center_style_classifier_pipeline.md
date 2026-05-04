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

## 3.0 2026-05-03 Strengthened Validation Plan

本轮 center-style classifier 的目标升级为：

```text
证明或否定：
  ECGTwin + target center token 生成的数据，
  是否比 vanilla/wrong-token 生成数据更接近目标中心真实 ECG 分布。
```

不能只看一个 7-way style classifier 的 top-1 accuracy。正式验证分成四层：

```text
Layer A: real-center separability
Layer B: synthetic target-style score
Layer C: real-vs-synth distribution gap
Layer D: downstream utility through Latent-Hull online AT
```

### Layer A: Real-Center Separability

训练数据只能是真实 ECG：

```text
positive = held-out real ECG from target center
negative = held-out real ECG from other centers
synthetic = never used for train/val/threshold selection
```

强制 split：

```text
1. 保存 record_ids。
2. 排除每个 center-token 的 K=500 anchors。
3. same-label matching，优先 exact multi-hot，其次 primary class。
4. NORM 只能匹配 pure NORM。
5. 每个 target center 单独训练 binary probe。
```

优先 feature backend：

| backend | role |
|---|---|
| frozen new full10 EfficientNet feature | 主验证 |
| ECGTwin VAE latent | 检查生成模型 latent manifold 是否接近 |
| digital ECG feature | 检查 HR/QRS/ST/voltage/noise 等可解释风格 |
| downsampled waveform logistic | 快速 smoke，不作为主结论 |

### Layer B: Synthetic Query Arms

每个 synthetic query 必须 paired control：

| arm | token | purpose |
|---|---|---|
| vanilla | none | ECGTwin 原始 prompt |
| target-token | `<target_center_CLASS>` | 主实验 |
| wrong-center | `<other_center_CLASS>` | 目标中心特异性 |
| wrong-class | `<target_center_OTHERCLASS>` | 类别语义是否混乱 |
| PTB-XL source | `<ptbxl_source_CLASS>` | source-style negative control |

判定：

```text
mean P(target center | target-token) > all controls
bootstrap CI of target-token - best-control > 0
```

如果 target-token 只赢 vanilla，但输给 wrong-center 或 PTB-XL-source，则不能声称
目标中心风格有效。

### Layer C: C2ST And Feature Distance

style score 上升可能是 artifact，所以必须同时跑 C2ST：

```text
C2ST(real target, synthetic target-token) lower is better
0.5 means real/synth hard to distinguish
```

Acceptance：

```text
C2ST(target-token) <= C2ST(vanilla)
C2ST(target-token) <= C2ST(wrong-center)
C2ST(target-token) <= C2ST(PTB-XL-source)
```

同时输出 feature distance：

```text
MMD or FID-like distance in frozen EfficientNet feature space
VAE latent mean/cov distance
digital feature KS distance
```

### Layer D: Latent-Hull Online AT As Utility Validation

center-style classifier 只能证明“像不像目标中心”，不能证明“对 EfficientNetV2 有用”。
因此把通过 gate 的 synthetic latents 接入 Latent-Hull online AT 做 Layer D。

比较矩阵：

| pool | purpose |
|---|---|
| real-anchor only | 当前最稳主线 |
| vanilla synthetic | ECGTwin 原始生成控制 |
| target-token synthetic | center-token 主实验 |
| wrong-center synthetic | 伪 style 控制 |
| target-token + real-anchor source-aware | 混合上限探索 |

Latent-Hull 形式：

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

如果 center token 有效，应该至少满足：

```text
target-token synthetic hull AT >= vanilla synthetic hull AT
target-token synthetic hull AT >= wrong-center synthetic hull AT
target-center AUPRC improves or PN2021 macro AUPRC non-decreasing
```

注意：

```text
Layer D 是 utility 证据，不是 style 证据。
即使 downstream AUPRC 提升，也不能单独说明样本符合目标中心风格。
```

## 3.1 2026-05-03 Formal Verification Plan For Regenerated Tokens

本轮验证对象：

```text
PN2021 prompt-token bank:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500/prompt_token_bank.pt

PTB-XL source prompt-token bank:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503/prompt_token_bank.pt
```

验证中心：

```text
PN2021:
  ningbo
  chapman_shaoxing
  cpsc_2018
  georgia

PTB-XL:
  ptbxl_source
```

执行状态：

```text
PN2021 big4 token bank: done
PTB-XL source token bank: done
token diagnostics:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics/regen_20260503/

注意：
  cpsc_2018_HYP and cpsc_2018_MI have no K500 anchors and remain init-like.
  They must be excluded from main target-style claims.
```

First pilot result:

```text
report:
  docs/tmp_md/center_token_effectiveness_pilot_20260503.md

synthetic output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503

full-10s real-only style probe:
  /root/autodl-tmp/per_center_style_classifier_full10s_max3000
  test_acc = 0.5093
  macro_f1 = 0.5148

verdict:
  NOT SUPPORTED as reliable center-style transfer yet.
  Full-10s 7-way probe shows exploratory positives for cpsc_2018/gated georgia,
  but no-leak same-label waveform and frozen-EfficientNet feature probes do not
  confirm a stable target-token advantage.
  Same-label C2ST shows target-token samples remain distinguishable from real ECG.
  Do not use regenerated v40 tokens as downstream main evidence yet.
```

Follow-up corrective run:

```text
script:
  scripts/ecgtwin_gen/run_center_token_v41_style_semantic_20260503.sh

change:
  keep denoising MSE as the main ECGTwin objective, then add frozen center-style
  CE and frozen super5 semantic BCE on low-noise one-step x0 predictions.

decision rule:
  v41 must pass the same no-leak same-label EfficientNet feature probe and C2ST
  controls before any downstream AUROC/AUPRC claim.
```

2026-05-03 v42 no-leak style aux update:

```text
new training script:
  scripts/ecgtwin_gen/train_noleak_center_style_aux_classifier.py

checkpoint:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/style_aux_noleak_full10_task1_v1/best_model.pt

training data:
  real PN2021 mmap only
  centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
  classes = NORM, MI, STTC
  K=500 prompt-token anchors excluded per center
  full 10s minimal_resample + per_sample_global

held-out test:
  acc = 0.6279
  macro F1 = 0.6555
```

这个 checkpoint 只用于 v42 token 训练的可微 style loss。正式证据仍然来自
`run_prompt_token_noleak_style_validation.py` 的独立 held-out query：

```text
target-token must beat vanilla, wrong-center, and PTB-XL-source controls
under Task-1 full10 EfficientNet feature probe and same-label C2ST.
```

V42-A0 validation result:

```text
report:
  docs/tmp_md/center_token_v42_validation_20260503.md

verdict:
  failed promotion gate.
  target-token improves style score only for ningbo and partly raw cpsc_2018.
  same-label C2ST is worse than controls on all four centers under the Task-1
  full10 EfficientNet feature check.
```

Next style-validation action:

```text
run checkpoint selection for v42 step00500 and step01000 before trying larger
generation or downstream AT.
```

本轮不把 synthetic ECG 用来训练风格分类器。风格分类器必须只看真实 ECG：

```text
train/val/test = real ECG only
query/eval    = vanilla synthetic / target-token synthetic / wrong-token synthetic
```

核心对照组：

| arm | target prompt | token | 作用 |
|---|---|---|---|
| A vanilla | class prompt | none | ECGTwin 原始生成基线 |
| B target-token | class prompt | `<target_center_CLASS>` | 主实验 |
| C wrong-center | class prompt | `<other_center_CLASS>` | 证明不是任意 token 都有效 |
| D wrong-class | class prompt | `<target_center_OTHERCLASS>` | 证明 token 类别语义没有乱用 |
| E source-token | class prompt | `<ptbxl_source_CLASS>` | PN2021 的 source-style negative control |

PTB-XL 作为 target 时，E 反过来变成 target-token，PN2021 token 作为 wrong-center
negative control。

## 3.2 需要证明的三层结论

### Layer 1: Token 确实被模型使用

轻量诊断：

```text
token_delta_norm
cos_to_init
within-token cosine
between-center same-class cosine
between-class same-center cosine
held-out denoise MSE: with-token vs no-token vs wrong-token
```

判定：

```text
with-token denoise MSE < no-token
target-token denoise MSE < wrong-center token
```

这一层只说明 DiT text path 对 soft prompt 有响应，不能单独证明中心风格。

### Layer 2: 生成 ECG 更像目标中心

正式主证据来自 real-only domain probes：

```text
Probe A: EfficientNet frozen penultimate feature -> target-center binary MLP
Probe B: ECGTwin VAE latent -> target-center binary MLP
Probe C: digital ECG feature -> logistic regression / shallow MLP
```

对每个 target center 独立训练二分类器：

```text
positive = held-out real ECG from target center
negative = held-out real ECG from other centers, same-label matched
```

评估 synthetic query：

```text
mean P(target | target-token)
median P(target | target-token)
bootstrap CI of P(target-token) - P(vanilla)
bootstrap CI of P(target-token) - P(wrong-center)
```

主判定：

```text
P(target | target-token) > P(target | vanilla)
P(target | target-token) > P(target | wrong-center)
P(target | target-token) does not exceed real-target range by an artifact-like margin
```

### Layer 3: 风格迁移对下游有效

把通过 gate 的 synthetic latents 接入 Latent-Hull online AT，做下游验证：

```text
downstream target center AUROC/AUPRC
PN2021 7-center macro AUROC/AUPRC
per-class AUROC/AUPRC
PN2021-C absolute corrupted AUROC/AUPRC
```

主判定：

```text
target-token downstream >= vanilla downstream
target-token downstream >= wrong-center downstream
target center AUPRC improves without harming 7-center macro AUPRC
```

如果 Layer 2 成立但 Layer 3 不成立，结论写成：

```text
center token creates measurable target-center style shift,
but this style shift is not yet useful for EfficientNet1DV2 adaptation.
```

## 3.3 No-Leak Split For PN2021 + PTB-XL

PN2021 K=500 anchors：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/ref_selection/<center>_k500_seed42.json
```

PTB-XL K=500 anchors：

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1/ref_selection/ptbxl_source_k500_seed42.json
```

强制排除：

```text
1. K anchors 不进入 style classifier train/val/test；
2. K anchors 不进入 downstream quick eval/full eval；
3. synthetic ECG 不进入 style classifier train；
4. PTB-XL fold9/fold10 不进入 PTB-XL source-token training；
5. PN2021 的 ptb-xl/ptbxl shard 仍然硬排除，避免 PTB-XL 泄漏。
```

PTB-XL source 风格验证时，PTB-XL split：

```text
token train anchors: folds 1-8 selected K=500
style probe real train: folds 1-8 minus anchors
style probe real val: fold 9
style probe real test: fold 10
```

PN2021 style probe split：

```text
target center remaining real:
  train 60%
  val   20%
  test  20%

non-target centers:
  same split policy, center-balanced negatives
```

## 3.4 Same-Label Matching

中心判别器不能靠“某中心 MI 更多”这种标签比例作弊。
训练和评估都优先 same-label matching：

```text
priority 1: exact super5 multi-hot
priority 2: primary class
priority 3: NORM only matches pure NORM
```

主报告分层：

```text
NORM
MI
STTC
ALL_MAIN3 = NORM + MI + STTC
ALL5 exploratory = CD + HYP + MI + NORM + STTC
```

`cpsc_2018` 没有可靠 HYP/MI anchors，`georgia` MI 很少；这些 center-class 只能报告
exploratory，不作为 thesis 主证据。

## 3.5 Ablation Matrix

必跑最小矩阵：

| ablation | 比较 | 证明点 |
|---|---|---|
| token vs no-token | B vs A | center token 是否有方向性 |
| target vs wrong-center | B vs C | 是否是目标中心特异，而不是任意 token 扰动 |
| target vs wrong-class | B vs D | token 是否保留疾病类别语义 |
| PN2021 vs PTB-XL token | B vs E | source-style token 是否与 target-style token 可分 |
| direct MV4 vs single-vector | regenerated MV4 vs v1/v4 | 多向量 soft prompt 是否更好 |
| actual-report ref text vs class-fallback | `--ref_text_mode actual_report` vs `class_fallback` | 旧增益是否来自 fallback text |
| seed42 vs seed1042 anchors | old ref selection vs quality-ref | 是否依赖偶然 reference selection |
| with digital/semantic gate vs no gate | gated vs ungated | 风格接近是否伴随医学质量 |

主报告只接受同一 reference、同一 class prompt、同一 DDPM seed/steps 下只改变 token 的 paired comparison。

## 3.6 Recommended Outputs

输出根目录：

```text
/root/autodl-tmp/domain_discriminator/center_style_v2_regen_20260503/
```

必须保存：

```text
config.json
split_record_ids.json
anchor_exclusion_report.json
probe_train_result.json
probe_training_log.json
real_probe_scores.csv
synth_query_scores.csv
c2st_results.csv
feature_distance_results.csv
digital_style_results.csv
downstream_joined_metrics.csv
validation_report.md
```

最终 `validation_report.md` 要给出一句清晰结论：

```text
SUPPORTED:
  target-token generated ECG is measurably closer to target-center style and
  downstream AUROC/AUPRC is non-decreasing or improved.

PARTIAL:
  target-token shifts style probes but downstream utility is not proven.

NOT SUPPORTED:
  target-token does not beat vanilla/wrong-token controls.
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

## 15. 2026-05-03 No-Leak Gate Result

The first Task-1-gated ningbo pilot used the new full10 EfficientNet checkpoint
for both generation-time p_target and independent feature validation:

```text
Task-1 checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt
candidate root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_n80_20260503
token-scale root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_20260503
```

Validation stack:

```text
K=500 ningbo anchors excluded from real probe/eval
same-label matching for NORM/MI/STTC
feature backend = Task-1 EfficientNet full10 penultimate feature
controls = vanilla, wrong-center token, PTB-XL source token
```

Key result:

| arm | mean ningbo style prob | no-leak same-label C2ST bacc | C2ST AUROC |
|---|---:|---:|---:|
| vanilla | 0.0823 | 0.7725 | 0.8299 |
| wrong-center token | 0.1617 | 0.8222 | 0.8988 |
| PTB-XL source token | 0.0826 | 0.7867 | 0.8385 |
| target token scale 1.0 | 0.4038 | 0.8113 | 0.8619 |
| target token scale 0.50 | 0.2867 | 0.7702 | 0.8292 |
| target token scale 0.25 | 0.1793 | 0.8295 | 0.8990 |

Interpretation:

```text
Full-strength target token makes samples more ningbo-like to the style probe
but also more distinguishable from real ECG. This is a style-artifact failure.

Target token with scale=0.50 is the current valid setting: it has much higher
ningbo style probability than controls while matching or slightly improving the
same-label C2ST realism proxy versus vanilla.
```

Downstream utility check:

```text
scale=0.50 target-token pool was promoted to Latent-Hull online AT.
The best target-AUPRC checkpoint improved ningbo clean AUROC/AUPRC:
  Task-1 baseline: 0.8657 / 0.4842
  scale=0.50 AT:   0.8679 / 0.4873
  delta:          +0.0022 / +0.0031
```

This satisfies a narrow target-center validity claim for ningbo:

```text
center token, when strength-scaled, can generate ECGTwin latent candidates that
are more target-center-like under a no-leak style probe and can improve the
target center's downstream AUROC/AUPRC after conservative Latent-Hull AT.
```

It does not yet satisfy the stronger all-center or PN2021-average claim because
PN2021 7-center AUPRC still falls slightly.

## 16. 2026-05-03 Other-Center Gate Result

The Task-1-gated `token_scale=0.50` pilot was repeated for
`chapman_shaoxing`, `georgia`, and `cpsc_2018`:

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_othercenters_scale05_n60_20260503
```

Strict promotion requires both:

```text
1. target-center style score above controls;
2. same-label no-leak EfficientNet C2ST no worse than vanilla/control.
```

Summary:

| center | target-token style vs vanilla | target-token C2ST vs vanilla | decision |
|---|---:|---:|---|
| chapman_shaoxing | 0.1492 vs 0.0675 | 0.7622 vs 0.7236 | fail: style up but artifacts higher |
| cpsc_2018 | 0.5657 vs 0.1716 | 0.7749 vs 0.6571 | fail as full pool: no MI refs |
| georgia | 0.6096 vs 0.6554 | 0.8087 vs 0.7286 | fail |

Additional chapman lower-strength test:

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_chapman_scale035_n60_20260503
```

`scale=0.35` kept the gate count high but did not pass no-leak EfficientNet C2ST:

| arm | mean style prob | C2ST bacc |
|---|---:|---:|
| target token scale 0.35 | 0.1370 | 0.8000 |
| vanilla | 0.0675 | 0.7904 |
| wrong-center token | 0.1339 | 0.7577 |
| PTB-XL source token | 0.0645 | 0.7464 |

Current validity boundary:

```text
The center-token evidence is strong for ningbo only. Other centers are useful
negative controls showing that higher gate pass count or higher style score is
not sufficient: the no-leak C2ST realism gate must also pass before downstream
AT promotion.
```

## 17. 2026-05-04 Target-Real Stream Utility Follow-Up

The strongest downstream utility result so far does not come from direct
synthetic self-distillation. It comes from using the validated ningbo
scale=0.50 target-token synthetic pool as a bounded latent candidate source,
while also adding the real K=500 ningbo ECG as a supervised target-real stream.

Inputs:

```text
target-token pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/target_token_s05/ningbo/gated
  kept 757/1200 = MI 296, NORM 366, STTC 95

real K=500 signals:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/real_anchor_selected_v1/ningbo/ningbo_real_k500_seed42.signals.npz

merged latent pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/merged_real_token_v1/ningbo_real500_s05large/merged.latent.npz
```

Best formal result after excluding the K=500 ref ids from ningbo eval:

| model | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|
| Task-1 baseline | 0.7780 / 0.4831 | 0.8657 / 0.4842 |
| real+token Latent-Hull AT, target_real_weight=20 | 0.7798 / 0.4885 | 0.8829 / 0.5049 |
| delta | +0.0018 / +0.0053 | +0.0172 / +0.0207 |

Validation meaning:

```text
This supports a partial target-center utility claim: with real K=500 anchors,
the token-derived latent candidates can help held-out ningbo AUPRC by +2.07pp.

It is not a pure center-token causal proof. The next required style-classifier
ablation is paired real-only / no-token / target-token / wrong-token under the
same target_real_weight=20 and the same ref-id exclusion.
```

Matched control result:

| model | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|
| real K500 only, abort-best | 0.7787 / 0.4856 | 0.8767 / 0.4936 |
| real K500 + no-token synthetic | 0.7797 / 0.4883 | 0.8830 / 0.5050 |
| real K500 + target-token synthetic | 0.7798 / 0.4885 | 0.8829 / 0.5049 |

Updated validation meaning:

```text
The downstream utility layer no longer supports a center-token-specific claim.
The no-token synthetic pool matches the target-token pool under the same
target_real_weight=20 recipe. The useful factor is synthetic latent candidates
plus target-real adaptation; the current center token does not provide an
additional downstream gain.

Future style validation should therefore use paired sample selection:
keep only candidates where target-token beats no-token on moderate target-center
style while maintaining equal or lower C2ST distance. High style score alone is
not enough and has already produced downstream-negative samples.
```

## 18. 2026-05-04 Multi-Center Target-Real + Token Utility

The target-real stream was extended from ningbo to three additional large
PN2021 centers using the existing v42 target-token generated pools.

Shared recipe:

```text
base model = /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt
K target real = 500 per center
synthetic source = v42 target-token gated latent pool
merge = real_anchor + prompt_token
source_weights = real_anchor=1.0,prompt_token=0.4
target_real_weight = 40
attack = latent_hull, M=10, lambda=0.15, steps=5
adv_weight = 0.06
adv labels = mixed_soft, teacher_mix=0.3
boundary gate = target probability in [0.45, 0.70]
eval = PN2021 full mmap, same K=500 ref ids excluded from target center
```

Formal target-center deltas:

| center | baseline target AUROC/AUPRC | token+real AT target AUROC/AUPRC | delta |
|---|---:|---:|---:|
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8972 / 0.4647 | +2.09pp / +3.96pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8543 / 0.5959 | +4.28pp / +3.73pp |
| georgia | 0.8157 / 0.5916 | 0.8262 / 0.6029 | +1.05pp / +1.13pp |

Matched no-token controls:

| center | target-token AUROC/AUPRC | no-token AUROC/AUPRC | token - no-token |
|---|---:|---:|---:|
| chapman_shaoxing | 0.8972 / 0.4647 | 0.8972 / 0.4653 | +0.00pp / -0.06pp |
| cpsc_2018 | 0.8543 / 0.5959 | 0.8541 / 0.5958 | +0.02pp / +0.01pp |

Interpretation:

```text
This is positive target-center adaptation evidence for chapman_shaoxing and
cpsc_2018, and a weaker positive trend for georgia. It supports the practical
route "small target-real ECG + center-token ECGTwin latent candidates +
latent-hull online AT" versus the bare PTB-XL baseline.

It does not prove center-token causality. The no-token controls for chapman and
cpsc are tied with the target-token runs. The next useful controls are real-only
and wrong-token under the same K=500 ref exclusion and target_real_weight=40
recipe, but current evidence already says v42 token is not the unique cause of
the downstream gain.
```

Pairwise style-delta test result:

```text
selector:
  scripts/ecgtwin_gen/select_paired_token_delta_pool.py

selected pairs:
  NORM=80, MI=80, STTC=30

criterion:
  target-token style score > no-token style score by at least 0.05
```

| model | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|
| paired target-token selected | 0.7794 / 0.4883 | 0.8834 / 0.5050 |
| paired no-token selected | 0.7802 / 0.4902 | 0.8837 / 0.5063 |

Conclusion:

```text
The style classifier can identify target-token samples that look more like
ningbo, but this style delta still does not improve downstream utility. The
style classifier must be treated as a filter/probe, not as the training target
alone. A successful next token must improve style while also improving or at
least preserving C2ST/feature-distance realism against real target ECG.
```

## 19. 2026-05-04 PTB-XL Token-Delta Downstream Check

The same lesson was reproduced on the PTB-XL low-resource self-distillation
line. A paired-delta selector used identical references and seeds, then kept
the top 400 pairs/class where the PTB-XL center token most increased target
class probability over no-token.

Filter-side evidence:

| arm | raw pairs | v2-filter kept | HYP kept | MI kept | STTC kept |
|---|---:|---:|---:|---:|---:|
| v46 paired-delta token | 2000 | 1759 | 264 | 334 | 377 |
| v46 paired-delta no-token | 2000 | 1069 | 74 | 219 | 108 |

Downstream:

| arm | custom AUROC/AUPRC | fold10 AUROC/AUPRC | PN2021 AUROC/AUPRC |
|---|---:|---:|---:|
| v46 paired-delta token | 0.8472 / 0.6222 | 0.8241 / 0.5936 | 0.6886 / 0.3677 |
| v46 paired-delta no-token | 0.8597 / 0.6612 | 0.8260 / 0.6030 | 0.7092 / 0.4063 |

Conclusion:

```text
Generation-side style or semantic-token gains are not sufficient validation.
The center-style classifier and target-class confidence can both favor token
samples while the downstream classifier still performs worse. The validation
pipeline must therefore report style, C2ST/feature realism, and downstream
AUROC/AUPRC separately, with downstream matched no-token controls as the final
promotion gate.
```
