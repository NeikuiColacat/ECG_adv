# Graduate Project Pipeline

本文档记录当前毕设主实验计划：用低样本 PTB-XL 训练集验证 ECGTwin prompt-token
合成数据是否能提升 EfficientNet1DV2 的 super5 分类性能。

## 目标

比较两条方法在同一 PTB-XL split、同一预处理、同一 EfficientNet1DV2 训练配置下的
AUROC 和 AUPRC：

| 方法 | 训练数据 | 目的 |
|---|---|---|
| Method A | PTB-XL 随机 2000 条真实 ECG | 低样本真实数据 baseline |
| Method B | 同一 2000 条真实 ECG + ECGTwin prompt-token 合成 ECG | 测试 ECGTwin 合成数据增强是否带来增益 |

固定分类任务：

```text
super5 class order = CD, HYP, MI, NORM, STTC
metric = macro AUROC, macro AUPRC, per-class AUROC/AUPRC
model = EfficientNet1DV2, 5-class multi-label BCE classifier
```

核心约束：

- Method A 和 Method B 必须使用完全相同的真实 PTB-XL 2000 条训练样本。
- Method B 的 ECGTwin reference ECG 也只能来自这同一批 2000 条训练样本。
- validation/test 中不得出现 Method B 用于 token training 或 reference selection 的这 2000 条样本。
- 两个方法都从随机初始化训练 EfficientNet1DV2，不从已有 super5 checkpoint 微调。
- 训练和评测都使用 `per_sample_global + minimal_resample`，不使用 legacy full filter。

## 数据划分

推荐第一版 split：

```text
seed = 42
train_real = 2000 PTB-XL records
val_real   = 2000 PTB-XL records from the remaining records
test_real  = all other remaining PTB-XL records
```

优先使用 patient-aware 或 record-aware multilabel stratified sampling，目标是让
`CD/HYP/MI/NORM/STTC` 在 train/val/test 中都有尽量接近全量 PTB-XL 的分布。

保存 split 文件：

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
```

split 文件至少记录：

```json
{
  "seed": 42,
  "scheme": "super5",
  "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
  "train_indices": [],
  "val_indices": [],
  "test_indices": [],
  "train_label_counts": {},
  "val_label_counts": {},
  "test_label_counts": {}
}
```

如果 train=2000 中某类过少，先不重抽整个 split，而是记录 class count，并在 ECGTwin
token training 和 synthetic generation 中对少数类做 oversampling 与质量 gate。

## PTB-XL Super5 标签组合

当前项目的 PTB-XL super5 标签来自 `scripts/triple_labels/label_schemes.py`：

```text
class order = CD, HYP, MI, NORM, STTC
source      = PTB-XL scp_codes -> scp_statements.csv diagnostic_class
threshold   = confidence >= 0.0
```

PTB-XL 是多标签任务，不是五类互斥单标签任务。一条 ECG 可以同时属于多个 super5
类别，例如 `CD+MI` 或 `HYP+STTC`。按当前项目规则，全量 `21799` 条 PTB-XL 记录中
出现过 23 种组合：

| 组合 | multi-hot CD/HYP/MI/NORM/STTC | 数量 | 比例 |
|---|---|---:|---:|
| NORM | `[0,0,0,1,0]` | 9069 | 41.60% |
| MI | `[0,0,1,0,0]` | 2532 | 11.62% |
| STTC | `[0,0,0,0,1]` | 2400 | 11.01% |
| CD | `[1,0,0,0,0]` | 1708 | 7.84% |
| CD+MI | `[1,0,1,0,0]` | 1297 | 5.95% |
| HYP+STTC | `[0,1,0,0,1]` | 781 | 3.58% |
| MI+STTC | `[0,0,1,0,1]` | 599 | 2.75% |
| HYP | `[0,1,0,0,0]` | 535 | 2.45% |
| CD+STTC | `[1,0,0,0,1]` | 471 | 2.16% |
| NONE | `[0,0,0,0,0]` | 411 | 1.89% |
| CD+NORM | `[1,0,0,1,0]` | 407 | 1.87% |
| HYP+MI+STTC | `[0,1,1,0,1]` | 361 | 1.66% |
| CD+HYP | `[1,1,0,0,0]` | 300 | 1.38% |
| CD+MI+STTC | `[1,0,1,0,1]` | 223 | 1.02% |
| CD+HYP+STTC | `[1,1,0,0,1]` | 211 | 0.97% |
| HYP+MI | `[0,1,1,0,0]` | 183 | 0.84% |
| CD+HYP+MI+STTC | `[1,1,1,0,1]` | 156 | 0.72% |
| CD+HYP+MI | `[1,1,1,0,0]` | 117 | 0.54% |
| NORM+STTC | `[0,0,0,1,1]` | 28 | 0.13% |
| CD+NORM+STTC | `[1,0,0,1,1]` | 5 | 0.02% |
| CD+HYP+NORM | `[1,1,0,1,0]` | 2 | 0.01% |
| HYP+NORM | `[0,1,0,1,0]` | 2 | 0.01% |
| CD+HYP+MI+NORM | `[1,1,1,1,0]` | 1 | 0.005% |

标签数分布：

| 阳性标签数 | 数量 | 比例 |
|---:|---:|---:|
| 0 | 411 | 1.89% |
| 1 | 16244 | 74.52% |
| 2 | 4068 | 18.66% |
| 3 | 919 | 4.22% |
| 4 | 157 | 0.72% |

全量类别阳性数：

```text
CD   = 4898
HYP  = 2649
MI   = 5469
NORM = 9514
STTC = 5235
```

对本毕设实验的影响：

- 随机 2000 条训练样本通常不会缺少五个单类；按 fold 1-8 训练分布估算，2000 条期望
  约有 `CD 449`、`HYP 243`、`MI 503`、`NORM 872`、`STTC 481` 个阳性。
- 稀少的是组合语义，特别是 `NORM+异常`、三类共现和四类共现。
- Method B 第一版合成建议只做五个单类 one-hot；不要把极稀少组合作为主生成目标。
- 第二阶段可补常见组合 ablation：`CD+MI`、`HYP+STTC`、`MI+STTC`、`CD+STTC`。
- `NONE` 不是一个医学诊断类别；它只表示当前 super5 规则没有命中任何目标类，不作为
  ECGTwin 合成目标。

## 统一预处理

本实验指定：

```text
preprocess_mode = minimal_resample
norm_mode       = per_sample_global
target_fs       = 100Hz
target_len      = 1000 samples = 10s
lead_order      = PTB-XL order
```

处理流程：

```text
native PTB-XL ECG
-> NaN/Inf 清理
-> lead reorder to PTB-XL canonical order
-> native fs -> 100Hz resample
-> pad/truncate to 10s = 1000 samples
-> per-sample global z-score
-> EfficientNet1DV2 input
```

明确不使用：

- `50Hz notch + 0.67-40Hz bandpass + 0.4s median baseline removal`
- DeepECG spectral scaling / FFT peak cleanup / `mhi_factor`
- train-set StandardScaler

原因：

- 用户指定本实验使用 `per_sample_global + minimal_resample`。
- 本实验重点是比较合成增强的效果，先避免滤波和 dataset-level scaling 额外改变变量。
- HYP/CD 等类别可能受电压尺度影响，后续可单独做 `train_standardizer` 或
  `legacy_ecgfounder_filter` ablation，但不混入本主实验。

需要补齐的代码能力：

```text
scripts/crosscenter_v2/preprocess_utils.py:
  add preprocess_mode=minimal_resample
  add norm_mode=per_sample_global

scripts/triple_labels/train_ptbxl.py:
  add --split_json
  add --preprocess_mode
  add --norm_mode
  allow train/val/test indices from custom split
```

## Method A

Method A 是低样本真实数据 baseline。

输入：

```text
train = 2000 real PTB-XL ECG
val   = 2000 real PTB-XL ECG
test  = remaining real PTB-XL ECG
```

训练：

```text
model          = EfficientNet1DV2 s_v2
num_classes    = 5
loss           = masked BCEWithLogitsLoss
pos_weight     = n_neg / n_pos, clipped to max 50
epochs         = 50
patience       = 10
batch_size     = 96, reduce if 10s input OOM
num_workers    = 4-8 on 4090D host
amp/bf16       = enable if current training script supports it safely
crop_len       = 1000 for full 10s first pass
```

输出：

```text
/root/autodl-tmp/graduate_project/method_a_real2000_seed42/
  best_model.pt
  training_log.json
  train_result.json
  split.json or split symlink
```

记录指标：

```text
val macro AUROC/AUPRC
test macro AUROC/AUPRC
test per-class AUROC/AUPRC
```

## Method B

Method B 使用 ECGTwin 和 5 个 PTB-XL class prompt tokens 生成额外训练样本，然后从零
训练同结构 EfficientNet1DV2。

### Token 定义

把 PTB-XL 当作当前实验的 source/style center，训练 5 个 class-specific prompt token：

```text
<ptbxl_CD>
<ptbxl_HYP>
<ptbxl_MI>
<ptbxl_NORM>
<ptbxl_STTC>
```

生成某一类 ECG 时：

```text
class prompt + <ptbxl_CLASS> token
```

示例：

```text
MI generation prompt:
  "stemi|st elevation myocardial infarction|acute" + <ptbxl_MI>

NORM generation prompt:
  "sinus rhythm|normal ecg." + <ptbxl_NORM>
```

工程实现不是修改 BERT/nomic tokenizer，而是在 prompt compiler 中把 `<ptbxl_MI>` 映射为
一个可训练的 768-d embedding，并 append 到 ECGTwin 的 `text_embed` 序列中：

```text
text_embed_aug  = concat(nomic_embed(class_prompt), learnable_center_class_token)
text_embed_mask = concat(class_prompt_mask, 1)
```

### ECGTwin Prompt 格式

ECGTwin 作者推理配置和示例使用 `|` 分隔诊断短语：

```text
primary diagnosis|supplementary diagnosis|supplementary diagnosis
```

当前 `config/DiT_ECGTwin.yaml` 使用 `mix: False`。代码路径是：

```text
target text
-> text.split('|')
-> bert-base-uncased tokenizer
-> nomic-ai/nomic-embed-text-v1.5
-> mean pooling + layer norm + L2 normalize
-> text_embed: (num_diagnosis_fragments, 768)
```

因此 prompt 的第一段应放最重要的诊断，后续片段作为补充条件。作者默认和示例覆盖：

| super5 | 本实验推荐 prompt | 证据等级 | 说明 |
|---|---|---|---|
| NORM | `sinus rhythm|normal ecg.` | 高 | 作者默认配置和 `normal_ecg` 示例 |
| CD | `left bundle branch block|lbbb` | 高 | 作者直接示例；可另测 `right bundle branch block|rbbb` |
| HYP | `left ventricular hypertrophy|high voltage` | 高 | 作者直接示例 |
| MI | `stemi|st elevation myocardial infarction|acute` | 高 | 作者直接示例；可另测 `myocardial infarction|st elevation|anterior wall` |
| STTC | `st-t change|st segment abnormality|t wave abnormality` | 中 | 作者有 ST/T 相关示例，但没有原生 `STTC` 标签 |

注意：

- ECGTwin 不是 PTB-XL super5 条件模型；它支持开放文本 cardiac condition，我们需要把
  `CD/HYP/MI/NORM/STTC` 映射成诊断 prompt。
- `nstemi|non st elevation|t wave inversion` 是作者示例，但 `NSTEMI` 同时带 MI 和 ST/T
  改变语义；若目标是纯 STTC，首轮不要把 `nstemi` 放在第一片段。
- 多标签组合 prompt 第二阶段再做，例如：

```text
CD+MI:
  left bundle branch block|lbbb|stemi|st elevation myocardial infarction|acute

HYP+STTC:
  left ventricular hypertrophy|high voltage|st segment abnormality|t wave abnormality
```

多标签 prompt 的 synthetic label 应使用对应 multi-hot，而不是 one-hot。

### Reference ECG / base vector

Reference ECG 只从 Method A 的同一 2000 条 train_real 中抽取。

原则：

- 生成 `CLASS` 时优先使用同类 reference ECG。
- 多标签样本可以作为所有阳性类的候选 reference，但生成标签第一版按目标 class one-hot 记录。
- 如果某类 reference 太少，允许重复采样 reference，但必须在 run config 中记录。
- `base_vector` 仍由 ECGTwin 原始 `ref_latent + ref_text_embed + pat_info -> IBE` 路径得到；
  PTB-XL style/class token 只进入 text condition，不写入 `base_vector`。

### Token 训练

冻结：

```text
ECGTwin VAE
ECGTwin IBE
ECGTwin DiT
nomic text encoder
```

只训练：

```text
5 x 4 x 768 prompt-token embeddings
```

#### 技术细节

中心 token 的目的不是让 tokenizer 学会一个新词，而是在 ECGTwin 已有文本条件序列后面
追加一组可学习的连续向量。原始 Textual Inversion 默认一个概念用 1 个向量，但本实验
主方案使用每个 super5 类别 4 个向量，以增强 MI/STTC/CD/HYP 这类多形态医学类别的表达能力。

以 PTB-XL MI 为例：

```text
plain prompt:
  stemi|st elevation myocardial infarction|acute

compiled prompt:
  nomic_embed("stemi")
  nomic_embed("st elevation myocardial infarction")
  nomic_embed("acute")
  learnable_embed("<ptbxl_MI_1>")
  learnable_embed("<ptbxl_MI_2>")
  learnable_embed("<ptbxl_MI_3>")
  learnable_embed("<ptbxl_MI_4>")
```

训练对象：

```text
TokenBank:
  shape = (5, 4, 768)
  classes = <ptbxl_CD>, <ptbxl_HYP>, <ptbxl_MI>, <ptbxl_NORM>, <ptbxl_STTC>
  vectors_per_class = 4
```

初始化：

```text
token[class, k] = class_init + small_noise[k]
class_init      = mean(nomic embeddings of that class prompt fragments)
```

例如：

```text
<ptbxl_MI_k> init = mean(
  embed("stemi"),
  embed("st elevation myocardial infarction"),
  embed("acute")
) + small_noise[k]
```

每个训练 step 的数据：

```text
ref ECG latent z_ref:    ECGTwin VAE latent, shape (4,128)
ref text_embed:          原始/类别 prompt embedding, shape (L,768)
target text_embed_base:  类别 prompt embedding, shape (L,768)
target token sequence:   TokenBank[class], shape (4,768)
pat_info:                hr, age, sex
label:                   PTB-XL super5 multi-hot
primary_class:           当前采样的目标 class
```

前向路径：

```text
1. 用 reference ECG/text/patient info 提取 base_vector:

   base_vector = IBE(z_ref, ref_text_embed, pat_info_ref)

2. 拼接目标类别 prompt 和可学习 token:

   text_embed_aug  = concat(target_text_embed_base, token_seq[primary_class])
   text_embed_mask = concat(ones(L), ones(4))

3. 对真实 ECG latent 加 diffusion noise:

   z_t = q_sample(z_real, t, epsilon)

4. 冻结 ECGTwin DiT，预测噪声:

   epsilon_hat = DiT(z_t, t, text_embed_aug, text_embed_mask, pat_info_tar, base_vector)

5. 只反传到 token:

   L_eps = MSE(epsilon_hat, epsilon)
```

推荐损失：

```text
L = L_eps
  + lambda_init * ||token_seq - token_seq_init||_2^2
  + lambda_norm * max(0, ||token_seq|| - norm_ceiling)^2
  + lambda_sep * token_class_separation_penalty
  + lambda_intra * same_class_vector_diversity_penalty
```

其中：

- `L_eps` 让 token 学会让 ECGTwin 在该类别 PTB-XL 样本上更容易去噪。
- `L_init` 防止 token 远离原始诊断语义太多，避免变成不可解释的任意向量。
- `L_norm` 防止 token norm 爆炸，避免 cross-attention 被 token 过度支配。
- `L_sep` 防止五个 class token 彼此塌缩成相同向量。
- `L_intra` 防止同一类别的 4 个向量完全塌缩成同一个方向；如果实际实现里未启用，
  至少在 diagnostics 中记录 same-class cosine。

采样策略：

```text
sample_strategy = class_balanced
```

原因：

- PTB-XL 中 NORM 最多，HYP 最少；普通 shuffle 会让 `<ptbxl_HYP>` 梯度太少。
- 多标签样本可以按每个阳性类展开为多个训练 item，例如 `CD+MI` 样本既可训练
  `<ptbxl_CD>`，也可训练 `<ptbxl_MI>`。
- 第一版不训练组合 token；组合样本只为其阳性单类 token 提供梯度。

多标签处理规则：

```text
if label = CD+MI:
  training item A: prompt CD + <ptbxl_CD>, target ECG = this record
  training item B: prompt MI + <ptbxl_MI>, target ECG = this record
```

暂不使用：

```text
<ptbxl_CD_MI>
<ptbxl_HYP_STTC>
```

原因是 23 种组合里很多非常稀少，第一版训练组合 token 容易过拟合。组合 token 只作为
第二阶段 ablation。

生成时：

```text
target class = MI
prompt       = stemi|st elevation myocardial infarction|acute
tokens       = <ptbxl_MI_1> + <ptbxl_MI_2> + <ptbxl_MI_3> + <ptbxl_MI_4>
ref ECG      = Method A 2000 条中的 MI-positive reference
base_vector  = IBE(ref ECG, ref text, ref pat_info)
output       = ECGTwin decoded ECG -> PTB-XL lead order -> 1000 samples
label        = [0,0,1,0,0] first-pass one-hot
```

如果生成多标签组合：

```text
target combo = CD+MI
prompt       = left bundle branch block|lbbb|stemi|st elevation myocardial infarction|acute
tokens       = <ptbxl_CD_1..4> + <ptbxl_MI_1..4>
label        = [1,0,1,0,0]
```

但多标签组合生成只在 Method B 主结果之后作为 ablation。

推荐设置：

```text
total_steps = 2000 first pass
batch_size  = 16
num_workers = 4
amp_dtype   = bf16
sample_strategy = class_balanced
n_token_vectors = 4
token_repeat = 1
```

这里 `n_token_vectors=4` 表示每类训练 4 个不同向量；`token_repeat=1` 表示不重复同一个向量。
不要把两者混淆。

保留 ablation：

```text
5 x 1 x 768  # 原始 Textual Inversion 风格最小版
5 x 2 x 768  # 保守 multi-vector
5 x 8 x 768  # 更强表达力，但更容易过拟合或过度控制
```

如果 HYP/CD gate 过低，再单独尝试：

```text
n_token_vectors = 4, token_repeat = 2
class-specific checkpoint selection
quality-aware reference selection
```

输出：

```text
/root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42/
  prompt_token_bank.pt
  metrics.jsonl
  run_config.json
  token_summary.csv
```

### 合成数量推荐

首轮推荐目标是每类通过 gate 的 synthetic ECG 为 1000 条：

| class | accepted target | candidate over-generate target |
|---|---:|---:|
| CD | 1000 | 4000 |
| HYP | 1000 | 4000 |
| MI | 1000 | 2500 |
| NORM | 1000 | 2500 |
| STTC | 1000 | 2500 |

最终 Method B 首轮训练池：

```text
real PTB-XL train = 2000
synthetic ECG     = up to 5000 accepted, balanced 1000/class
```

理由：

- 5000 synthetic 是真实训练集的 2.5 倍，足够测试低样本扩充是否有价值。
- 10,000 synthetic 也可做第二阶段，但首轮过大容易让合成数据噪声主导训练。
- CD/HYP 历史上生成质量更不稳定，所以候选生成数设得更高，但最终必须通过 gate。

如果 CD/HYP 无法达到 1000 条 accepted：

```text
1. 不用低质量样本硬凑数量；
2. 记录每类实际 accepted count；
3. Method B 主结果使用 gated accepted pool；
4. 另做 B-trusted-classes ablation，只使用 NORM/MI/STTC。
```

### 合成样本 gate

每条 synthetic ECG 至少经过：

```text
NaN/Inf gate
flatline/saturation gate
amplitude p2p gate
Einthoven / aVR residual gate
HR/QRS sanity gate
class-specific digital ECG gate when available
frozen EfficientNet1DV2 teacher target-class score gate
```

第一版 teacher/class gate：

```text
NORM: target p_NORM high, abnormal class probabilities low
MI/STTC/CD/HYP: target p_CLASS reaches class-specific threshold
```

注意：teacher gate 只能过滤明显不一致样本，不能作为临床合法性的唯一证据。

输出 synthetic pool：

```text
/root/autodl-tmp/graduate_project/method_b_synth_seed42/
  samples_all_candidates.npz
  gated_samples.npz
  gated_quality.csv
  gated_summary.json
```

`gated_samples.npz` 格式：

```text
signals: (N, 1000, 12) float32, PTB-XL lead order, 100Hz
labels:  (N, 5) float32, class order CD/HYP/MI/NORM/STTC
```

### Classifier 训练

Method B 从零训练 EfficientNet1DV2。

训练数据：

```text
train = same 2000 real PTB-XL + gated synthetic pool
val   = same Method A val split
test  = same Method A test split
```

首轮推荐采样：

```text
real:synth sampling ratio = 1:1
epoch sample budget       = comparable to Method A unless explicitly running expansion ablation
```

如果使用现有 `--synth_npz --synth_ratio` 路线：

```text
--synth_ratio 1.0
```

含义：

- 每个训练 epoch 中，synthetic 与 real 的抽样权重接近 1:1。
- 不是把波形拼成更长 ECG。
- 是把 synthetic 作为单独训练样本加入训练池。

输出：

```text
/root/autodl-tmp/graduate_project/method_b_real2000_plus_synth5000_seed42/
  best_model.pt
  training_log.json
  train_result.json
  synth_summary.json
```

## 对比报告

生成最终报告：

```text
/root/autodl-tmp/graduate_project/reports/method_a_vs_b_seed42.md
docs/tmp_md/graduate_project_method_a_vs_b_seed42.md
```

报告表格：

| run | train real | train synth | test AUROC | test AUPRC | delta AUROC | delta AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Method A real-only seed42 | 2000 | 0 | 0.8433 | 0.6234 | baseline | baseline |
| Method B class-fallback ref text seed42 | 2000 | 3364 | 0.8477 | 0.6419 | +0.0044 | +0.0185 |
| Method B actual-report ref text seed42 | 2000 | 3313 | 0.8412 | 0.6239 | -0.0021 | +0.0005 |
| Method B class-fallback new-generator seed42 | 2000 | 3391 | 0.8432 | 0.6065 | -0.0001 | -0.0169 |

必须额外报告：

- per-class AUROC/AUPRC，特别是 HYP/CD 是否被合成数据伤害。
- synthetic accepted count per class。
- synthetic quality gate pass rate per class。
- Method B 是否只是提高训练集拟合而没有提高 test AUPRC。
- 如果只提升 AUROC 不提升 AUPRC，需要单独解释正负样本排序和少数类 precision 的差异。

建议统计：

```text
primary = seed42 single split result
confirm = seeds 42/43/44 repeat if Method B shows positive delta or close-to-positive trend
```

判定标准：

```text
strong positive:
  Method B test macro AUPRC >= Method A + 0.01
  and no more than one class AUPRC has meaningful regression

weak positive:
  Method B test macro AUPRC improves 0.003-0.01
  or PN2021/robustness follow-up improves despite PTB-XL small gain

negative:
  Method B macro AUPRC <= Method A
  or CD/HYP/MI/STTC key classes are harmed by noisy synthetic samples
```

## Scale-Up Plan After Seed42 Positive Result

Seed42 first run result:

```text
Method A real2000:
  test AUROC/AUPRC = 0.8433 / 0.6234

Method B real2000 + 3364 gated synthetic, 5 x 4 x 768 token,
class-fallback ref text:
  test AUROC/AUPRC = 0.8477 / 0.6419

Delta:
  AUROC +0.0044
  AUPRC +0.0185
```

Per-class outcome:

| class | delta AUROC | delta AUPRC | interpretation |
|---|---:|---:|---|
| CD | +0.0004 | +0.0349 | useful despite only 118 accepted samples |
| HYP | +0.0124 | +0.0209 | useful despite only 246 accepted samples |
| MI | +0.0252 | +0.0745 | strongest gain |
| NORM | -0.0008 | +0.0082 | stable |
| STTC | -0.0148 | -0.0464 | main regression risk |

Important correction:

```text
This first positive Method B run did not use the per-record clinical report
embedding in ECGTwin's IBE/base_vector path.

It used:
  base_vector text = class fallback prompt
  target text      = class fallback prompt + 4 learned class token vectors
```

Keep this run as a useful ablation, not as the most faithful ECGTwin-author
usage.

## Actual-Report Fix And Hyperparameter Probe 2026-05-03

Implemented code fixes:

```text
methods/ecgtwin_gen/prompt_token/trainer.py
  --ref_text_mode class_fallback|actual_report|normal
  IBE/base_vector now receives ref_text_embed separately from target_text_embed.

scripts/ecgtwin_gen/train_center_prompt_tokens.py
  exposes --ref_text_mode.

scripts/ecgtwin_gen/generate_center_prompt_token_synth_batched.py
  exposes --ref_text_mode, --ref_class_policy, --per_ref_cap.
  generation records now save ref_text_mode and ref_class_policy.

scripts/triple_labels/train_ptbxl.py
  saves best_model_auroc.pt and best_model_auprc.pt.
  best_model.pt follows --checkpoint_metric, default auroc for backward compatibility.
```

PTB-XL cache check:

```text
cache = /root/autodl-tmp/graduate_project/method_b_ptbxl_prompt_cache_v1/center_full_latents/ptbxl.pt
records = 1971
text coverage = 1971 / 1971 non-empty
text_embed item shape = (1, 768)
latent shape = (1971, 4, 128)
```

Actual-report token runs:

| token run | ref text | ref policy | vectors/class | steps | lr | final delta norm | note |
|---|---|---|---:|---:|---:|---:|---|
| `mv4_actual_report_steps4000` | actual report | same-class probe | 4 | 4000 | 8e-4 | 1.7308 | stable but not better than step2000 by gate |
| `mv8_actual_report_steps2000` | actual report | same-class probe | 8 | 2000 | 6e-4 | 1.3505 | stable but worse gate than MV4 |

Small generation probe, 200/class candidates, same digital/teacher gate:

| probe | kept/1000 | NORM | MI | STTC | HYP | CD | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| MV4 step2000, same-class ref | 413 | 139 | 128 | 122 | 20 | 4 | best overall probe |
| MV4 step4000, same-class ref | 404 | 103 | 142 | 132 | 26 | 1 | helps MI/STTC/HYP, hurts NORM/CD |
| MV4 step4000, normal ref | 258 | 103 | 70 | 83 | 1 | 1 | too much abnormal-class semantic loss |
| MV4 step4000, random-any ref | 273 | 74 | 89 | 104 | 5 | 1 | more diverse but weaker controllability |
| MV8 step2000, same-class ref | 357 | 110 | 151 | 80 | 12 | 4 | MI improves, STTC/HYP/NORM worse |

Current hyperparameter recommendation:

```text
ref_text_mode      = actual_report for faithful ECGTwin reproduction
ref_class_policy   = same_class for controllability
n_token_vectors    = 4
checkpoint         = step2000, not final step4000
lr                 = 8e-4 for MV4 long run, 1e-3 also remains valid for old class-fallback run
token_orth_weight  = 0.01
token_repeat       = 1
per_ref_cap        = 20 for full generation
num_inference_steps= 25 for fast candidate search
```

Full actual-report run using the recommended probe choice:

```text
token_bank = /root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4_actual_report_steps4000/prompt_token_bank_step02000.pt
candidate pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_actual_report_mv4_step2000_seed42/
gated pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_actual_report_mv4_step2000_seed42/ptbxl/gated_max1000/gated_samples.npz
accepted = 3313 / 12500
accepted by class:
  NORM = 1000 / 2500
  MI   = 1000 / 2500
  STTC = 1000 / 2500
  HYP  = 255  / 2500
  CD   = 58   / 2500
```

Downstream actual-report Method B:

```text
run = /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3313_actual_report_mv4_step2000_seed42
train = real2000 + synth3313
synth_ratio = 1.0
checkpoint_metric = auroc
test AUROC/AUPRC = 0.8412 / 0.6239
```

Per-class test result:

| class | AUROC | AUPRC |
|---|---:|---:|
| CD | 0.8320 | 0.6550 |
| HYP | 0.7583 | 0.3288 |
| MI | 0.8160 | 0.5647 |
| NORM | 0.9192 | 0.8861 |
| STTC | 0.8805 | 0.6850 |

Conclusion:

```text
actual_report fixes ECGTwin-author fidelity but does not improve downstream
EfficientNet1DV2 on this seed42 split.

Compared with Method A:
  AUROC -0.0021
  AUPRC +0.0005

Compared with old class-fallback Method B:
  AUROC -0.0065
  AUPRC -0.0180

Therefore, use actual_report as the thesis-faithful ECGTwin path, but keep the
class-fallback run as a strong empirical ablation. Do not claim actual_report
improved downstream performance until another seed/pool variant beats Method A.
```

## Class-Fallback New-Generator Control 2026-05-03

Purpose:

```text
Isolate whether the old class-fallback positive result came from:
  A. class-fallback text itself, or
  B. the old random candidate pool / sampling / gate composition.
```

Control setup:

```text
token_bank = /root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4/prompt_token_bank.pt
ref_text_mode = class_fallback
ref_class_policy = same_class
per_ref_cap = 20
n_per_class = 2500
steps = 25
seed = 42
```

Generated and gated pool:

```text
candidate pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/
gated pool = /root/autodl-tmp/graduate_project/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/ptbxl/gated_max1000/gated_samples.npz
accepted = 3391 / 12500
accepted by class:
  NORM = 1000 / 2500
  MI   = 1000 / 2500
  STTC = 1000 / 2500
  HYP  = 259  / 2500
  CD   = 132  / 2500
```

This accepted distribution is very close to the old class-fallback pool:

| pool | total accepted | NORM | MI | STTC | HYP | CD |
|---|---:|---:|---:|---:|---:|---:|
| old class-fallback | 3364 | 1000 | 1000 | 1000 | 246 | 118 |
| new class-fallback | 3391 | 1000 | 1000 | 1000 | 259 | 132 |
| actual-report | 3313 | 1000 | 1000 | 1000 | 255 | 58 |

Downstream run:

```text
run = /root/autodl-tmp/graduate_project/method_b_real2000_plus_synth3391_classfallback_newgen_sameclass_cap20_seed42
train = PTB-XL real2000 + synthetic3391
synth_ratio = 1.0
checkpoint_metric = auroc
test AUROC/AUPRC = 0.8432 / 0.6065
```

Best AUPRC checkpoint check:

```text
best_model_auroc.pt: test AUROC/AUPRC = 0.8432 / 0.6065
best_model_auprc.pt: test AUROC/AUPRC = 0.8414 / 0.6042
```

Per-class AUPRC for the AUROC checkpoint:

| class | AUPRC |
|---|---:|
| CD | 0.5975 |
| HYP | 0.3331 |
| MI | 0.5142 |
| NORM | 0.8864 |
| STTC | 0.7015 |

Conclusion:

```text
The new-generator class-fallback control did not reproduce the old positive
Method B result.

Compared with Method A:
  AUROC -0.0001
  AUPRC -0.0169

Compared with old class-fallback Method B:
  AUROC -0.0045
  AUPRC -0.0354

Therefore the old 0.6419 AUPRC result should be treated as a pool-specific
positive ablation, not as proof that class-fallback text reliably improves the
classifier. The next step should inspect pool diversity/reference reuse and
sample-level quality differences between the old and new pools before scaling.
```

### 当前 center token 是否足够收敛

当前 token run:

```text
run = /root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4
token shape = 5 x 4 x 768
steps = 2000
batch_size = 16
usable token-training records = 1971
effective epochs ~= 2000 / ceil(1971/16) ~= 16
elapsed ~= 91s
```

最后几个 logged windows：

```text
step 1920: recon_loss=0.03630, delta_norm=1.4451
step 1940: recon_loss=0.03321, delta_norm=1.4612
step 1960: recon_loss=0.03364, delta_norm=1.4521
step 1980: recon_loss=0.03593, delta_norm=1.4543
step 2000: recon_loss=0.03085, delta_norm=1.4585
```

结论：

- 2000 steps 已经足够形成有效 token，因为 downstream AUPRC 已经提升。
- 但不能宣称严格收敛；`recon_loss` 仍有 stochastic 波动，`delta_norm` 到最后仍在缓慢上升。
- 当前没有 held-out denoise MSE、checkpoint-gate curve 或 checkpoint downstream pilot，因此不应只相信最后一个 checkpoint。

推荐 scale-up token 训练：

| token run | steps | lr | save_every | 用途 |
|---|---:|---:|---:|---|
| `mv4_steps2000` | 2000 | 1e-3 | 0 | 已完成 baseline |
| `mv4_steps4000` | 4000 | 8e-4 | 500 | 主 scale-up |
| `mv4_steps6000` | 6000 | 5e-4 | 500 | 检查长训练是否过拟合 |

checkpoint 选择不要只看最后一步。每个 checkpoint 用小样本 probe：

```text
per checkpoint:
  generate 200/class candidates
  gate with same digital + teacher rules
  record pass rate, p_target, class distribution, visual examples

choose checkpoint by:
  1. MI/CD/HYP accepted count and target score improve;
  2. STTC 不继续恶化；
  3. token norm/delta norm 不爆炸；
  4. small downstream pilot AUPRC 不低于 current Method B。
```

如果要加快执行，第一轮只跑：

```text
mv4_steps4000, save_every=500
```

暂时不把 `5 x 8 x 768` 作为主 scale-up。当前正增益来自 `5 x 4 x 768`，先把训练步数和合成池规模做清楚；8-vector 会引入新的变量，适合作为后续 ablation。

### Synthetic Scale-Up

当前 synthetic gate：

```text
candidates = 2500/class = 12500 total
accepted:
  NORM = 1000 / 2500  # hit cap
  MI   = 1000 / 2500  # hit cap
  STTC = 1000 / 2500  # hit cap, but downstream regressed
  HYP  = 246  / 2500
  CD   = 118  / 2500
```

因此不要简单均匀扩大所有类。更合理的策略：

1. 对 NORM/MI 已经足够，最多从 `1000/class` 扩到 `1500/class`。
2. STTC 虽然 gate 过了 1000 条，但 downstream 明显下降；不要盲目扩大 STTC。
3. 重点补 HYP/CD 的随机候选，因为它们通过数少但 downstream 是正增益。
4. 增加随机性时只改变 generation seed/reference draw，不改变 validation/test，不引入新真实样本。

推荐生成计划：

| pool | 新候选 | gate cap | 目标 |
|---|---|---|---|
| `pool_v1_current` | 已完成 2500/class | NORM/MI/STTC 1000, HYP 246, CD 118 | 当前正结果 |
| `pool_v2_hyp_cd_boost` | HYP +8000, CD +8000 | HYP 500-800, CD 500-800 | 补少数类 |
| `pool_v3_mi_norm_plus` | NORM +2500, MI +2500 | NORM 1500, MI 1500 | 测试安全扩大 |
| `pool_v4_sttc_reprompt` | STTC +5000 with new prompt | STTC 500-1000 | 修复 STTC，而不是直接扩大旧 STTC |

STTC 新 prompt 候选：

```text
st segment depression|t wave inversion|st-t abnormality
st-t change|t wave abnormality|non specific st abnormality
```

避免把 `nstemi` 放在第一片段，因为它容易带入 MI 语义。

### Classifier Scale-Up Runs

不要只跑一个更大合成池。建议固定同一 Method A split，跑以下对照：

| run | synthetic pool | synth_ratio | 目的 |
|---|---|---:|---|
| `B1_current` | 3364 current gated | 1.0 | 已完成正结果 |
| `B2_hyp_cd_boost_r1` | current + HYP/CD boost | 1.0 | 看少数类补强是否继续提升 |
| `B3_hyp_cd_boost_r05` | same as B2 | 0.5 | 防止 synthetic 过强 |
| `B4_no_sttc_extra_r1` | NORM/MI/HYP/CD expanded, STTC 不扩 | 1.0 | 验证 STTC 是不是负迁移源 |
| `B5_all_expand_r05` | NORM/MI/HYP/CD expanded + STTC reprompt | 0.5 | 最终大池但降低 synthetic 权重 |

优先级：

```text
1. train mv4_steps4000 token checkpoint.
2. generate HYP/CD extra random candidates with new seeds.
3. gate into pool_v2_hyp_cd_boost.
4. run B2_hyp_cd_boost_r1 and B3_hyp_cd_boost_r05.
5. 如果 STTC 仍下降，再做 STTC reprompt，而不是继续扩大旧 STTC。
```

成功标准：

```text
primary:
  macro AUPRC > 0.6419

secondary:
  STTC AUPRC drop shrinks from -0.0464 to within -0.02
  MI AUPRC remains above Method A by at least +0.04
  CD/HYP gains remain positive
```

### Critical Hyperparameters To Clarify Before More Scale-Up

不要在没分清变量的情况下继续盲目扩大 synthetic pool。下一轮必须把以下超参数和设计点拆开：

#### 1. `ref_text_mode`: base_vector 是否使用真实 clinical report

ECGTwin 官方 base_vector 路径需要：

```text
ref ECG latent + ref clinical report text_embed + ref HR/age/sex -> IBE -> base_vector
```

当前 Method B seed42 实现情况：

```text
cache 中有 PTB-XL actual report text 和 text_embed
但 token training / generation 实际使用 class fallback prompt embedding
而不是每条 ref ECG 自己的 actual clinical report embedding
```

也就是说，当前结果是：

```text
base_vector = IBE(ref ECG latent, class fallback prompt, ref HR/age/sex)
target text = class prompt + <ptbxl_CLASS_1..4>
```

这不是最贴近 ECGTwin 作者设计的用法。下一轮必须加入：

| mode | base_vector text | 用途 |
|---|---|---|
| `class_fallback` | 当前类别 fallback prompt | 复现已完成正结果 |
| `actual_report` | PTB-XL cache 中每条 ref ECG 的真实 report text_embed | 推荐主线 |
| `null_or_normal` | `sinus rhythm|normal ecg.` 或 null text | ablation，测试 report 贡献 |

优先级：

```text
1. patch trainer/generator to support --ref_text_mode actual_report
2. rerun token mv4_steps4000 with actual_report base_vector
3. compare small probe gate and downstream pilot against class_fallback
```

#### 2. `ref_class_policy`: ref ECG 和目标 ECG 是否必须同类

ECGTwin 的目标是 personalized ECG generation。作者示例支持：

```text
ref ECG/report = Sinus rhythm|Normal ECG
target report  = myocardial infarction / LBBB / LVH / etc.
```

因此 ref ECG 和 target ECG 不必是同一种诊断类型。当前 Method B 使用的是：

```text
same-class reference:
  generate MI -> choose MI-positive ref ECG
  generate HYP -> choose HYP-positive ref ECG
```

下一轮应明确测试：

| policy | 例子 | 目的 |
|---|---|---|
| `same_class` | MI ref -> MI target | 当前策略，形态一致，容易生成目标类 |
| `normal_ref` | NORM ref -> MI/HYP/CD/STTC target | 作者风格，测试能否从 normal 个体生成目标病种 |
| `random_any` | any train2000 ref -> target class | 增加个体多样性 |
| `cross_non_target` | non-MI ref -> MI target | 测试 target prompt/token 控制力 |
| `mixed_ref_bank` | same_class + normal_ref + random_any | 最终扩池候选 |

判断指标：

```text
target p_CLASS
digital gate pass rate
feature diversity / duplicate rate
downstream AUROC/AUPRC
```

如果 `normal_ref` 或 `random_any` 也能通过 gate，synthetic pool 的多样性会明显增加，尤其 HYP/CD 不再受 117/313 个 primary-class reference 限制。

#### 3. Randomness and diversity

ECGTwin diffusion 生成是随机的，但“随机”来自初始噪声 `x_init` 和采样过程。

规则：

```text
same model + same ref + same prompt + same seed/x_init
  -> same generated ECG

same model + same ref + same prompt + different x_init seed
  -> different generated ECG

same model + same prompt + different ref ECG/base_vector
  -> different personalized ECG
```

当前 batched generator 已经使用：

```text
seed_base = seed + class_idx * 1_000_000 + offset
x_init = torch.randn(..., generator=manual_seed(seed_base))
```

因此大量合成是有意义的，前提是：

```text
1. seed/offset 不重复；
2. ref ECG 不只是少数记录反复复制；
3. gate 后检查 duplicate / near-duplicate；
4. 不把同一个 ref + same seed 的重复样本当成新增数据。
```

下一轮必须新增 diversity report：

```text
per-class pairwise distance in EfficientNet feature space
nearest-neighbor duplicate rate
per-reference accepted count
per-reference cap, e.g. max 20 accepted samples/ref
```

#### 4. Token training hyperparameters

优先调：

| hyperparameter | candidates | reason |
|---|---|---|
| `n_token_vectors` | 1, 2, 4, 8 | 4 当前有效；1 是 Textual-Inversion baseline；8 看是否过强 |
| `total_steps` | 2000, 4000, 6000 | 2000 有效但未证明收敛 |
| `lr` | 5e-4, 8e-4, 1e-3 | 长训练应降 LR |
| `reg_init_weight` | 1e-4, 1e-3, 1e-2 | 防止 token 远离诊断语义 |
| `token_orth_weight` | 0, 0.001, 0.01 | 防止 4 个向量塌缩 |
| `init_noise_std` | 0, 0.005, 0.01 | 控制 multi-vector 初始多样性 |
| `sample_strategy` | class_balanced, primary_distribution | class-balanced 当前有效，但可能改变真实分布 |
| `ref_text_mode` | class_fallback, actual_report | 必须补齐 |

#### 5. Generation hyperparameters

优先调：

| hyperparameter | candidates | reason |
|---|---|---|
| `num_inference_steps` | 25, 50, 100 | 25 很快但可能质量不够 |
| `ref_class_policy` | same_class, normal_ref, random_any, mixed | 影响多样性和 target controllability |
| `target_hr_age_mode` | ref, class_median, random_train_distribution | HR/age 也是 ECGTwin 条件 |
| `candidate_per_class` | 2500, 5000, 10000 | 先补 HYP/CD，不均匀扩 |
| `per_ref_cap` | 10, 20, 50 | 防止少数 ref 生成过多重复样本 |
| `seed_policy` | fixed_reproducible, random_bank | 保证可复现且不重复 |

#### 6. Gate and downstream training hyperparameters

优先调：

| hyperparameter | candidates | reason |
|---|---|---|
| `min_target_prob` | class-specific | HYP/CD/STTC 难度不同 |
| `digital_gate_required` | strict, teacher_only, hybrid | STTC strict gate 可能不等于 downstream utility |
| `max_pass_per_class` | 500, 1000, 1500 | 控制合成类分布 |
| `synth_ratio` | 0.25, 0.5, 1.0 | 当前 1.0 有效，但 STTC 负迁移 |
| `epoch_sample_budget` | real_len, real+synth_len | 当前每 epoch 只采样 2000，不是完整遍历合成池 |
| `checkpoint_metric` | val_auroc, val_auprc, composite | 当前保存 best AUROC；主目标 AUPRC 应保存 best AUPRC 或双 checkpoint |
| `label_mode` | hard_onehot, teacher_soft, ref_multihot | 合成标签噪声会影响 STTC |

Code-level TODO status after 2026-05-03 run:

```text
DONE 1. add --ref_text_mode to token trainer and batched generator.
DONE 2. add --ref_class_policy and --per_ref_cap to batched generator.
DONE 3. save both best_val_auroc.pt and best_val_auprc.pt in train_ptbxl.py.
TODO 4. add diversity/duplicate report for synthetic pools.
WAIT 5. run B2/B3 only after actual_report and ref policy probe.
```

Decision on item 5:

```text
Do not immediately scale actual_report synthetic pools. The actual-report full
run did not beat Method A or old class-fallback Method B. The next useful
experiment is a targeted ablation, not blind scale-up:

  A. rerun class-fallback with same new same-class/per_ref_cap code to isolate
     whether the improvement came from class-fallback text or from pool sampling;
  B. add diversity/duplicate report before generating more candidates;
  C. if continuing actual_report, focus on HYP/CD/CD digital gates and STTC
     negative transfer rather than increasing all classes.
```

## Ref-Target Mismatch Synthetic Pretrain + Real AT Fine-Tune

### Motivation

ECGTwin officially supports different reference and target conditions:

```text
ref condition:
  ref ECG + ref report + ref HR/age/sex -> IBE -> base_vector

target condition:
  target report + target HR/age/sex -> DiT generation condition
```

Therefore, generation does not need:

```text
generate MI -> MI reference ECG
```

It can instead test:

```text
generate MI -> any/ref NORM/other PTB-XL ECG + target MI prompt
```

This experiment separates the role of:

```text
reference ECG = morphology / patient / center style anchor
target prompt = diagnostic target
```

### Experiment Design

Use the same seed42 graduate-project split:

```text
train real ECG = 2000 PTB-XL records
validation = 2000 PTB-XL records
test = remaining 17799 PTB-XL records
preprocess = minimal_resample + per_sample_global
classifier input = 100 Hz, 10 s, (N, 1000, 12)
```

New synthetic generation:

```text
ref ECG pool = the same random train2000 PTB-XL ECGs
ref report = translated/normalized ECGTwin prompt from:
  /root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompts_seed42.jsonl
target report = target super5 prompt + optional <ptbxl_CLASS_1..4> token
ref_class_policy = random_any / mixed_ref_bank
target classes = CD, HYP, MI, NORM, STTC
synthetic target = 20000 accepted ECG total
```

Recommended class allocation for the first 20000 accepted pool:

| class | accepted target | reason |
|---|---:|---|
| NORM | 3000 | common baseline morphology, do not dominate |
| MI | 5000 | prior synthetic MI was the most useful class |
| STTC | 3000 | previous STTC caused negative transfer, cap it |
| HYP | 4500 | sparse real class, needs more candidates |
| CD | 4500 | sparse/weak generated class, needs more candidates |

If CD/HYP cannot pass gate, do not lower quality thresholds blindly. Generate
more candidates or report lower accepted counts.

### Training Schedule

Stage 1: synthetic-only pretraining.

```text
input = 20000 gated synthetic ECGs
label = target class one-hot for first version
loss = masked BCEWithLogitsLoss + pos_weight
epochs = 30-50
checkpoint_metric = val AUPRC on real validation split
```

Do not evaluate success on synthetic validation only. The pretraining checkpoint
must be selected by real PTB-XL validation AUROC/AUPRC.

Stage 2: real-only adversarial fine-tuning.

```text
init = best synthetic-pretrained checkpoint
clean data = 2000 real PTB-XL train records
fine-tune epochs = 20-30
lr = 1e-4 to 5e-4
main metric = test macro AUROC/AUPRC
```

### Recommended Adversarial Training Scheme

Use **real-anchor latent-hull PGD / TA-OMAT**, not synthetic-anchor free PGD as
the first choice.

Reason:

```text
1. Direct synthetic augmentation has been unstable.
2. Raw input-space PGD is cheap but can create non-physiological waveforms.
3. Free VAE-latent PGD can move off the medical manifold.
4. Same-label real-anchor latent hull stays closer to real PTB-XL morphology:

   z_adv = (1 - lambda) * z0 + lambda * sum_i softmax(a_i) * z_i

   where z0 and z_i are ECGTwin VAE latents from real train2000 ECGs.
```

First-version AT constraints:

```text
anchor source = real train2000 only
label group = exact same super5 multi-hot when group size is sufficient
fallback = same primary class only with teacher-soft label, not hard one-hot
lambda = 0.25
hull_M = 10
hull_steps = 5
hull_lr = 0.3
pgd_eps_l2 = 2.0 in ECGTwin latent space
adv_weight = 0.25 first, then 0.5 if stable
clean_weight = 1.0
```

Loss design:

```text
L = L_clean + adv_weight * L_adv + consistency_weight * L_consistency

L_clean:
  BCEWithLogits(clean_logits, y_real)

L_adv, exact-label hull:
  BCEWithLogits(adv_logits, y_real)

L_adv, primary-class fallback hull:
  BCEWithLogits(adv_logits, y_soft)

y_soft:
  teacher probabilities from the synthetic-pretrained model on decoded adv ECG,
  optionally mixed with the original real label:
    y_soft = 0.7 * y_real + 0.3 * teacher_prob

L_consistency:
  KL/BCE consistency between clean prediction and adv prediction, detached
  clean prediction as teacher.
```

Gate adversarial samples during fine-tuning:

```text
NaN/Inf check
amplitude range check
Einthoven/aVR residual check
teacher confidence sanity
skip samples whose teacher target semantics collapse completely
```

Do not use cross-class hard-label latent mixing in the first version. If latent
mixing changes the diagnosis, hard labels become invalid and the experiment is
hard to interpret.

### Baselines For This Experiment

Required comparison table:

| run | stage 1 | stage 2 | purpose |
|---|---|---|---|
| A | none | real-only clean train | existing Method A baseline |
| C1 | synthetic20k pretrain | real clean fine-tune | separates pretraining effect |
| C2 | synthetic20k pretrain | real input-space PGD fine-tune | cheap AT baseline |
| C3 | synthetic20k pretrain | real latent-hull PGD fine-tune | recommended main run |
| C4 | no synthetic pretrain | real latent-hull PGD fine-tune | tests whether synthetic pretrain matters |

Success criteria:

```text
primary:
  test macro AUPRC > Method A 0.6234

strong:
  test macro AUPRC >= 0.6419
  and at least 3/5 per-class AUPRC improve vs Method A

failure:
  AUPRC improves only on NORM while MI/HYP/CD/STTC regress
```

## 执行顺序

1. 实现 custom PTB-XL split 脚本，保存 train2000/val2000/test-rest JSON。
2. 给 EfficientNetV2 训练入口补齐 `minimal_resample + per_sample_global` 配置。
3. 跑 Method A，拿到 seed42 baseline。
4. 用同一 2000 条训练 ECG 构建 PTB-XL prompt-token reference cache。
5. 训练 5 个 `<ptbxl_CLASS>` prompt tokens。
6. 每类生成候选 synthetic ECG，做 quality/digital/teacher gate。
7. 跑 Method B，从零训练 EfficientNet1DV2。
8. 对 Method A/B 做同一 test split 的 AUROC/AUPRC 对比报告。
9. 如果 Method B 有希望，再跑 seeds 43/44 或扩大到每类 2000 accepted synthetic。

## 当前不确定点

默认计划已经可以执行，但以下选择会影响实验解释：

1. `val=2000, test=remaining` 是否接受；如果想让 val/test 各占剩余一半，需要改 split。
2. Method B synthetic label 第一版建议用 one-hot 目标 class；如果希望更贴近 PTB-XL 多标签分布，可以改成 teacher soft label 或 reference-label multi-hot。
3. 首轮 synthetic accepted target 建议 `1000/class`；如果目标是最大化样本扩张，可以直接改成 `2000/class`，但训练更容易被低质量 CD/HYP 样本影响。
4. 首轮 Method B 建议 real:synth = 1:1；如果目标是“把合成样本和 PTB-XL 全部拼起来完整训练”，需要另跑 expansion ablation。

## 2026-05-03 自动执行记录：Ref-Target Mismatch C 系列

已完成工程准备：

```text
translated report prompt JSONL:
  /root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompts_seed42.jsonl

translated report text-embedding cache:
  /root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompt_embeds_seed42.pt

real-anchor latent pool:
  /root/autodl-tmp/graduate_project/ptbxl_real_train2000_seed42.latent.npz
  N = 1971 usable train2000 ECGTwin latents
```

生成配置：

```text
token bank = method_b_ptbxl_tokens_seed42_mv4
prompt token vectors = 4 per class
reference ECG = random_any from train2000 cache
reference condition = translated/report-normalized English ECGTwin text embedding
target condition = super5 class prompt + <ptbxl_CLASS> token
candidate size = 4000/class = 20000 total
output:
  /root/autodl-tmp/graduate_project/ref_target_mismatch_n20000_translated_randomany_mv4_seed42/
```

Smoke test 100 candidates:

```text
gate kept 33 / 100
NORM 11/20
MI   12/20
STTC 10/20
HYP   0/20
CD    0/20
```

Interpretation:

```text
NORM/MI/STTC remain the reliable direct-synthesis classes.
HYP/CD can often raise the victim target probability, but the strict digital
criteria still reject them. Therefore C-series experiments must keep two pools:
  1. full candidate pool for broad synthetic-only pretraining;
  2. gated trusted pool for quality-sensitive AT/ablation.
```

Active runs:

```text
C0 gate:
  input  = ref_target_mismatch_n20000.../ptbxl/samples.npz
  output = ref_target_mismatch_n20000.../ptbxl/gated/

C1 synthetic-only pretrain:
  output = /root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42
  train  = 20000 synthetic candidates only
  val/test = real PTB-XL seed42 split
  follow-up = initialize real-only fine-tune and then latent-hull AT from this checkpoint
```

Completed metrics:

| run | training route | custom seed42 test AUROC | custom seed42 test AUPRC | official fold10 AUROC | official fold10 AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| A-old | real2000 from scratch, original selection | 0.8433 | 0.6234 | not rerun | not rerun | not rerun | not rerun |
| A-fair | real2000 from scratch, AUPRC checkpoint, no early-stop bias | 0.8357 | 0.6040 | 0.8224 | 0.5922 | 0.7339 | 0.4074 |
| C0 | synthetic20k only | 0.6649 | 0.4107 | not evaluated | not evaluated | not evaluated | not evaluated |
| C1 | synthetic20k pretrain -> real2000 clean fine-tune | 0.8645 | 0.6723 | 0.7490 | 0.5106 | 0.7364 | 0.4195 |
| C3 | C1 -> real-anchor latent-hull AT, M10, boundary 0.5-0.6 | 0.8654 | 0.6738 | 0.7510 | 0.5118 | 0.7381 | 0.4230 |
| C3-wide | C1 -> real-anchor latent-hull AT, M10, boundary 0.45-0.65 | 0.8655 | 0.6733 | 0.7512 | 0.5095 | 0.7373 | 0.4198 |
| C4 | A-fair -> real-anchor latent-hull AT, M10, boundary 0.5-0.6 | 0.8356 | 0.6046 | 0.8214 | 0.5925 | 0.7314 | 0.4063 |
| D1 | C3 teacher -> student self-distill, real2000 + synthetic20k soft | 0.8670 | 0.6737 | 0.7631 | 0.5273 | 0.7407 | 0.4280 |
| D2 | C3 teacher -> student self-distill, real2000 only | 0.8682 | 0.6836 | 0.7549 | 0.5180 | 0.7308 | 0.4215 |

Important interpretation:

```text
custom seed42 test:
  C1/C3 are clearly better than A-fair and A-old.

PN2021:
  C1/C3 give small but consistent average AUPRC gains over A-fair
  (+0.0121 for C1, +0.0156 for C3).

official PTB-XL fold10:
  C1/C3 are worse than A-fair. This is a warning that the synthetic-pretrain
  benefit may be split-dependent or may over-adapt to the random train2000/test-rest
  protocol.

latent-hull AT:
  The strict 0.5-0.6 target-probability gate kept only 75/1000 adversarial
  samples. It still improved custom seed42 and PN2021 slightly, but the next
  ablation should try wider boundary gates such as 0.45-0.65 and larger
  n_adv_total before making a final claim.

follow-up result:
  The wider 0.45-0.65 gate produced 427/3000 adversarial samples but did not
  beat the strict gate on AUPRC. The no-synthetic-pretrain C4 AT run also did
  not improve PN2021 over A-fair. Current evidence says the main gain comes from
  synthetic20k pretraining followed by real fine-tuning; latent-hull AT is a
  small stabilizing add-on, not the primary source of gain.

self-distillation:
  D1 improves the external PN2021 average to 0.7407 / 0.4280 and official
  fold10 AUPRC to 0.5273. This is the best current external-generalization
  student.
  D2 improves the custom seed42 test to 0.8682 / 0.6836, but PN2021 falls to
  0.7308 / 0.4215. Treat D2 as split-specialized until repeated across seeds.
```

Executed C3 latent-hull AT implementation details:

```text
base checkpoint:
  /root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_auprc_real_finetune_seed42/best_model.pt

real latent pool:
  /root/autodl-tmp/graduate_project/ptbxl_real_train2000_seed42.latent.npz
  N = 1971 usable real train2000 ECGTwin VAE latents

attack mode:
  real-anchor latent-hull PGD
  anchor z0 = real train2000 ECGTwin latent
  candidates z_i = nearest/same-label real train2000 ECGTwin latents
  label grouping = exact same super5 multi-hot
  hull_M = 10
  hull_lambda = 0.25
  hull_steps = 5
  hull_lr = 0.3
  pgd_eps_l2 = 2.0

optimized variable:
  mixture logits a_i only
  w_i = softmax(a_i)
  z_mix = sum_i w_i z_i
  z_adv = (1 - hull_lambda) z0 + hull_lambda z_mix

semantic/boundary gate:
  strict run: target sigmoid probability in [0.50, 0.60]
  accepted buffer = 75 / 1000 attacked anchors
  wider run: target sigmoid probability in [0.45, 0.65]
  accepted buffer = 427 / 3000 attacked anchors

training label for adv buffer:
  target-only masked BCE label
  target class = argmax positive class of the anchor label
  target dimension = 1.0
  non-target dimensions = -1.0 and ignored by masked BCE

fine-tune mix:
  real train2000 clean samples + latent-hull adv buffer
  ptbxl_weight = 1.0
  adv_weight = 0.25
  lr = 5e-5
  weight_decay = 1e-4
  max epochs = 12
```

Method status:

```text
Keep C3 strict as the current best AT variant.
Do not claim latent-hull AT is the main source of gain.
The main effective route is:
  ECGTwin prompt-token synthetic20k pretrain
  -> real2000 clean fine-tune
  -> strict real-anchor latent-hull AT as a small stabilizer.
```

## Canonical Reproducibility Record: Custom Seed42 +3pp AUROC / +7pp AUPRC

This is the canonical record for the large PTB-XL custom-seed42 improvement
observed on 2026-05-03. It consolidates the previously separate
`graduate_project_self_disillation*.md` notes into this main graduation-project
document.

### Claim Boundary

The reproducible positive claim is:

```text
On the project-defined low-resource PTB-XL custom seed42 protocol,
ECGTwin synthetic pretraining followed by real2000 fine-tune and optional
self-distillation improves macro AUROC by about +3pp and macro AUPRC by about
+7pp versus the A-fair real2000 baseline.
```

This claim is specific to the custom seed42 split. It should not be written as
an official PTB-XL fold10 improvement, because the C/D line did not reproduce
the same gain on fold10.

Important 2026-05-04 center-token ablation update:

```text
The +3pp/+7pp custom-seed42 gain should not be attributed causally to the
PTBXL center token. A no-token vanilla ECGTwin synthetic pretrain control
matched or exceeded the original center-token C1 result after real2000
fine-tuning. The supported claim is "ECGTwin synthetic pretraining helps this
custom protocol"; the center-token-specific claim must use the separate v2
self-distillation ablation, where center-token beats vanilla under that v2 setup.
```

### Fixed Data Protocol

Split:

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json

train = 2000 randomly selected PTB-XL records
val   = 2000 randomly selected PTB-XL records
test  = remaining 17799 PTB-XL records
```

Labels:

```text
scheme = super5
class order = CD, HYP, MI, NORM, STTC
source = PTB-XL diagnostic_class mapping in scripts/triple_labels/label_schemes.py
```

Classifier preprocessing:

```text
preprocess_mode = minimal_resample
norm_mode       = per_sample_global
target_fs       = 100 Hz
target_len      = 1000 samples = 10 seconds
lead order      = PTB-XL order: I, II, III, aVR, aVL, aVF, V1-V6
```

Classifier architecture:

```text
EfficientNet1DV2
variant = s_v2
input_channels = 12
num_classes = 5
activation = leaky_relu
stochastic_depth_prob = 0.304
dropout_rate = 0.0
use_se = True
norm_type = batch
```

Hardware/runtime expectation:

```text
RTX 4090D 24GB
80GB RAM
15 CPU cores
large caches/checkpoints under /root/autodl-tmp
```

### Baseline A-Fair

Purpose:

```text
Use only the 2000 real PTB-XL training ECGs.
Select checkpoint by AUPRC to avoid comparing against an AUROC-biased or
early-stop-biased baseline.
```

Artifact:

```text
/root/autodl-tmp/graduate_project/method_a_real2000_seed42_ckpt_auprc_p50
```

Metrics:

| metric | value |
|---|---:|
| custom seed42 AUROC | 0.8357 |
| custom seed42 AUPRC | 0.6040 |
| official fold10 AUROC | 0.8224 |
| official fold10 AUPRC | 0.5922 |
| PN2021 avg AUROC | 0.7339 |
| PN2021 avg AUPRC | 0.4074 |

### PTBXL Center-Token Synthetic Candidate Pool

The large-gain C/D line uses synthetic ECG generated with the PTB-XL source
center prompt-token bank.

Token setup:

```text
token bank = /root/autodl-tmp/graduate_project/method_b_ptbxl_tokens_seed42_mv4/prompt_token_bank.pt
token schema = direct MV4
token vectors = 4 x 768 per class token
token examples = <ptbxl_NORM>, <ptbxl_MI>, <ptbxl_STTC>, <ptbxl_HYP>, <ptbxl_CD>
```

Reference and target condition:

```text
reference ECG = random_any from the same real2000 train split
reference text = translated/report-normalized English ECGTwin text embedding
target condition = super5 diagnosis prompt + <ptbxl_CLASS> token
ECGTwin base_vector = official IBE reference path
center/style information = target prompt-token path
```

Candidate pool:

```text
N = 20000 synthetic ECG
per class = 4000 candidates
output:
  /root/autodl-tmp/graduate_project/ref_target_mismatch_n20000_translated_randomany_mv4_seed42/ptbxl/samples.npz

signals shape = (20000, 12, 1000)
labels shape  = (20000, 5)
raw ECGTwin decode shape before classifier postprocess = (20000, 12, 1024)
```

Important limitation:

```text
NORM/MI/STTC are the most reliable direct-synthesis classes.
HYP/CD can be used for broad synthetic pretraining, but strict digital gates
often reject them. Do not claim all generated HYP/CD are medically validated.
```

### Stage C0: Synthetic-Only Pretraining

Purpose:

```text
Initialize EfficientNet1DV2 on the broad PTBXL center-token synthetic pool.
This stage alone is not useful as a final model; it provides a representation
initialization for later real-data fine-tuning.
```

Artifact:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42
```

Key config from `train_result.json`:

```text
synth_npz = /root/autodl-tmp/graduate_project/ref_target_mismatch_n20000_translated_randomany_mv4_seed42/ptbxl/samples.npz
synth_ratio = 0.25
batch_size = 64
epochs = 30
patience = 8
lr = 0.01
weight_decay = 0.01
cosine_tmax = 15
checkpoint_metric = auroc
seed = 5042
```

Result:

| run | custom seed42 AUROC | custom seed42 AUPRC |
|---|---:|---:|
| C0 synthetic20k only | 0.6649 | 0.4107 |

Interpretation:

```text
Synthetic-only training is weak. The gain appears only after real2000 fine-tune.
```

### Stage C1: Synthetic Pretrain -> Real2000 Fine-Tune

Purpose:

```text
Fine-tune the C0 synthetic-pretrained checkpoint on the 2000 real PTB-XL ECGs.
This is the main source of the custom seed42 gain.
```

Artifact:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_auprc_real_finetune_seed42
```

Key config:

```text
init_ckpt = /root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42/best_model_auprc.pt
synth_npz = none
train data = real2000 only
batch_size = 64
epochs = 50
patience = 10
lr = 0.003
weight_decay = 0.01
cosine_tmax = 15
checkpoint_metric = auprc
seed = 6042
```

Result:

| run | custom seed42 AUROC | custom seed42 AUPRC | delta vs A-fair |
|---|---:|---:|---:|
| A-fair | 0.8357 | 0.6040 | baseline |
| C1 | 0.8645 | 0.6723 | +2.88pp / +6.83pp |

### Stage C3: C1 -> Real-Anchor Latent-Hull AT

Purpose:

```text
Use ECGTwin VAE latent space to create on-manifold real-anchor adversarial
samples. This stage is a small stabilizer on top of C1, not the primary source
of the large custom-seed42 gain.
```

Artifact:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_real_latenthull_at_M10_n1000_seed42
```

Key config:

```text
init_ckpt = /root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_auprc_real_finetune_seed42/best_model.pt
real latent pool = /root/autodl-tmp/graduate_project/ptbxl_real_train2000_seed42.latent.npz
usable real ECGTwin latents = 1971
attack mode = real-anchor latent-hull PGD
same-label grouping = exact same super5 multi-hot
hull_M = 10
hull_lambda = 0.25
hull_steps = 5
hull_lr = 0.3
pgd_eps_l2 = 2.0
boundary gate = target sigmoid probability in [0.50, 0.60]
accepted adv buffer = 75 / 1000 attacked anchors
fine-tune lr = 5e-5
weight_decay = 1e-4
batch_size = 64
eval_batch_size = 256
patience = 5
seed = 42
```

Latent-hull definition:

```text
w_i = softmax(a_i)
z_mix = sum_i w_i z_i
z_adv = (1 - lambda) z0 + lambda z_mix

z0 = anchor real ECGTwin VAE latent
z_i = same-label real ECGTwin VAE latents
optimized variable = mixture logits a_i only
```

Result:

| run | custom seed42 AUROC | custom seed42 AUPRC | delta vs A-fair |
|---|---:|---:|---:|
| A-fair | 0.8357 | 0.6040 | baseline |
| C3 strict | 0.8654 | 0.6738 | +2.97pp / +6.98pp |

### Stage D1/D2: C3-Teacher Self-Distillation

Purpose:

```text
Use the C3 strict model as a teacher to transfer smoother class boundaries into
students initialized from C0 synthetic pretraining.
```

Teacher:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthpretrain_real_latenthull_at_M10_n1000_seed42/best_model.pt
```

Student initialization:

```text
/root/autodl-tmp/graduate_project/ref_mismatch_synthonly_pretrain_n20000_mv4_seed42/best_model_auprc.pt
```

Distillation loss:

```text
L = hard_weight * L_hard_real + L_soft_teacher

real samples:
  hard masked BCE + teacher soft BCE

synthetic samples, D1 only:
  teacher soft BCE only

temperature = 2.0
real_distill_alpha = 0.3
synth_distill_weight = 0.5
checkpoint_metric = auprc
```

D1 artifact:

```text
/root/autodl-tmp/graduate_project/self_distill_d1_teacher_c3_init_c0_synth20k_seed42

synth_npz = /root/autodl-tmp/graduate_project/ref_target_mismatch_n20000_translated_randomany_mv4_seed42/ptbxl/samples.npz
synth_ratio = 0.5
batch_size = 64
eval_batch_size = 256
epochs = 50
patience = 10
lr = 0.003
weight_decay = 0.01
seed = 7042
```

D2 artifact:

```text
/root/autodl-tmp/graduate_project/self_distill_d2_teacher_c3_init_c0_realonly_seed42

synth_npz = none
batch_size = 64
eval_batch_size = 256
epochs = 50
patience = 10
lr = 0.003
weight_decay = 0.01
seed = 7043
```

Results:

| run | custom seed42 AUROC | custom seed42 AUPRC | delta vs A-fair |
|---|---:|---:|---:|
| A-fair | 0.8357 | 0.6040 | baseline |
| D1 C3-teacher + synth20k soft | 0.8670 | 0.6737 | +3.13pp / +6.97pp |
| D2 C3-teacher real-only | 0.8682 | 0.6836 | +3.25pp / +7.96pp |

D1 vs D2 interpretation:

```text
D1 is better for external-generalization style reporting because it improves
PN2021 over C3 and keeps synthetic soft data in the method.

D2 is the best custom seed42 PTB-XL score, but it uses no synthetic samples
during distillation and may be split-specialized.
```

### Complete Metrics Table For The Large-Gain Line

| run | route | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| A-fair | real2000 from scratch | 0.8357 | 0.6040 | 0.8224 | 0.5922 | 0.7339 | 0.4074 |
| C1 | synth20k pretrain -> real2000 FT | 0.8645 | 0.6723 | 0.7490 | 0.5106 | 0.7364 | 0.4195 |
| C3 | C1 -> latent-hull AT | 0.8654 | 0.6738 | 0.7510 | 0.5118 | 0.7381 | 0.4230 |
| D1 | C3 teacher -> real2000 + synth20k soft | 0.8670 | 0.6737 | 0.7631 | 0.5273 | 0.7407 | 0.4280 |
| D2 | C3 teacher -> real2000 soft only | 0.8682 | 0.6836 | 0.7549 | 0.5180 | 0.7308 | 0.4215 |

### 2026-05-04 Center-Token Causal Ablation For The Large-Gain Line

Question:

```text
Is the +3pp/+7pp custom seed42 gain caused by the PTBXL center token, or by the
broader ECGTwin synthetic-pretrain -> real-fine-tune route?
```

Control design:

```text
Keep the same train2000 split, same translated-report random_any reference
policy, same 20000 synthetic candidate size, same C0 synthetic-only pretrain,
and same C1 real2000 fine-tune.

Only change:
  target prompt = super5 diagnosis prompt only
  no learned <ptbxl_CLASS> token
```

No-token vanilla synthetic pool:

```text
/root/autodl-tmp/graduate_project/vanilla_ecgtwin_synth_candidates_n20000_translated_randomany_seed42/ptbxl/samples.npz

N = 20000
per class = 4000
signals shape = (20000, 12, 1000)
labels shape = (20000, 5)
```

Vanilla C0 synthetic-only pretrain:

```text
artifact:
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42

config:
  synth_npz = vanilla no-token 20k pool
  synthetic_only = true
  synth_ratio = 0.25
  batch_size = 64
  epochs = 30
  patience = 8
  lr = 0.01
  checkpoint_metric = auroc
  seed = 5042

custom test = 0.6023 / 0.3851
```

Vanilla C1 real2000 fine-tune:

```text
artifact:
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthpretrain_auprc_real_finetune_seed42

init_ckpt:
  /root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42/best_model_auprc.pt

config:
  train data = real2000 only
  batch_size = 64
  epochs = 50
  patience = 10
  lr = 0.003
  checkpoint_metric = auprc
  seed = 6042
```

Repro commands:

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
SPLIT=/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
CACHE=/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
SYNTH=/root/autodl-tmp/graduate_project/vanilla_ecgtwin_synth_candidates_n20000_translated_randomany_seed42/ptbxl/samples.npz
C0=/root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42
C1=/root/autodl-tmp/graduate_project/ablation_vanilla_no_token_synthpretrain_auprc_real_finetune_seed42

$PY scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --split_json "$SPLIT" \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --cache_path "$CACHE" \
  --output_dir "$C0" \
  --crop_len 1000 \
  --batch_size 64 \
  --epochs 30 \
  --patience 8 \
  --lr 0.01 \
  --weight_decay 0.01 \
  --cosine_tmax 15 \
  --num_workers 4 \
  --seed 5042 \
  --checkpoint_metric auroc \
  --synth_npz "$SYNTH" \
  --synth_ratio 0.25 \
  --synthetic_only

$PY scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --split_json "$SPLIT" \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --cache_path "$CACHE" \
  --output_dir "$C1" \
  --init_ckpt "$C0/best_model_auprc.pt" \
  --crop_len 1000 \
  --batch_size 64 \
  --epochs 50 \
  --patience 10 \
  --lr 0.003 \
  --weight_decay 0.01 \
  --cosine_tmax 15 \
  --num_workers 4 \
  --seed 6042 \
  --checkpoint_metric auprc

$PY scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir "$C1" \
  --crop_len 1000 \
  --ptbxl_cache "$CACHE" \
  --pn2021_mmap_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap \
  --skip_mimic
```

Result:

| run | token | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| A-fair | none | 0.8357 | 0.6040 | 0.8224 | 0.5922 | 0.7339 | 0.4074 |
| C1 original | PTBXL MV4 token | 0.8645 | 0.6723 | 0.7490 | 0.5106 | 0.7364 | 0.4195 |
| C1 vanilla control | no token | 0.8670 | 0.6846 | 0.8560 | 0.6695 | 0.7719 | 0.4750 |

Interpretation:

```text
This ablation is negative for the center-token causal story in the +3pp/+7pp
pipeline. The no-token vanilla ECGTwin control is not worse; after real2000
fine-tuning it exceeds the original center-token C1 on custom seed42, official
fold10, and PN2021.

Therefore the +3pp/+7pp result should be attributed to ECGTwin synthetic
pretraining plus real2000 fine-tuning, not specifically to center-token
conditioning.
```

Conservative conclusion:

```text
Center token is useful in the separate v2 self-distillation setting where it
rescues a vanilla ECGTwin synthetic pool, but it is not validated as the cause
of the large C0->C1 custom-seed42 gain.
```

### Thesis-Safe Wording

Use this wording:

```text
Under our low-resource PTB-XL custom seed42 protocol, ECGTwin synthetic
pretraining followed by real2000 fine-tuning improves macro AUROC by about
+3 percentage points and macro AUPRC by about +7 percentage points over the
A-fair real2000 baseline. The strongest custom result is obtained after
C3-teacher self-distillation. A 2026-05-04 no-token ablation shows this large
custom gain is not caused specifically by the PTBXL center token.
```

Do not use this wording:

```text
The method improves official PTB-XL fold10 performance by +3pp/+7pp.
The +3pp/+7pp gain proves the center-token design is effective.
```

Reason:

```text
The large C/D custom gain did not transfer to official fold10. The official
fold10-safe center-token/self-distillation evidence is the separate v2 E4 line.
The large custom C0->C1 gain now has a no-token control that is stronger than
the original center-token C1.
```

## Canonical v2 Center-Token Self-Distillation Ablation

This section keeps the useful content from the former
`graduate_project_self_disillationv2.md` file.

Question:

```text
Does self-distillation v2 benefit from ECGTwin itself, or specifically from
PTBXL center-token conditioning?
```

Three-arm ablation:

| ID | method | real data | synthetic data | center token |
|---|---|---:|---:|---:|
| A | real2000 EfficientNet1DV2 | 2000 | none | no |
| B | vanilla ECGTwin + self-distill v2 | 2000 | 20000 candidates -> 2000 filtered | no |
| C | PTBXL center-token ECGTwin + self-distill v2 | 2000 | 20000 candidates -> 2000 filtered | yes |

Shared v2 filter:

```text
teacher = 3-seed real2000 EfficientNet1DV2 ensemble
teacher checkpoints:
  /root/autodl-tmp/graduate_project/method_a_real2000_seed42/best_model.pt
  /root/autodl-tmp/graduate_project/method_a_real2000_seed43_v2teacher/best_model.pt
  /root/autodl-tmp/graduate_project/method_a_real2000_seed44_v2teacher/best_model.pt

gamma = 0.3
y_soft = 0.3 * y_condition + 0.7 * p_teacher_ensemble
tau_high = 0.6
tau_low = 0.35
NORM gate = p_NORM >= 0.7 and max abnormal <= 0.3
per_class_cap = 400
max_keep_total = 2000
```

Vanilla ECGTwin generation support:

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth_batched.py
  --no_token
```

Vanilla artifacts:

```text
candidate pool:
  /root/autodl-tmp/graduate_project/vanilla_ecgtwin_synth_candidates_n20000_translated_randomany_seed42/ptbxl/samples.npz

filtered pool:
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_vanilla_real2000_ens3_seed42/synth_v2_filtered_top2000_gamma03.npz

student:
  /root/autodl-tmp/graduate_project/self_distill_v2_e4_vanilla_real2000_ens3_filtered2000_gamma03_scratch_seed42
```

Center-token v2 artifacts:

```text
filtered pool:
  /root/autodl-tmp/graduate_project/self_distill_v2_filtered_real2000_ens3_seed42/synth_v2_filtered_top2000_gamma03.npz

student:
  /root/autodl-tmp/graduate_project/self_distill_v2_e4_real2000_ens3_filtered2000_gamma03_scratch_seed42
```

Main result under the legacy-default PN2021 evaluation口径 used by the original
v2 record:

| ID | method | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| A | real2000 best teacher seed44 | 0.8506 | 0.6469 | 0.8417 | 0.6482 | 0.7594 | 0.4339 |
| B | vanilla ECGTwin + v2 | 0.8333 | 0.6051 | 0.8212 | 0.6005 | 0.7528 | 0.4233 |
| C | PTBXL center-token ECGTwin + v2 | 0.8500 | 0.6371 | 0.8405 | 0.6380 | 0.7616 | 0.4546 |

Interpretation:

```text
Vanilla ECGTwin synthetic data is harmful even after v2 filtering.
PTBXL center-token conditioning is what makes the v2 synthetic pool useful.
Compared with vanilla ECGTwin + v2, center-token + v2 improves custom AUPRC by
+0.0320, fold10 AUPRC by +0.0375, and PN2021 AUPRC by +0.0313.
Compared with the strongest real2000 seed44 teacher, center-token + v2 does not
improve PTB-XL in-domain AUPRC, but it improves PN2021 AUPRC by +0.0207.
```

Sensitivity:

```text
If PN2021 evaluation is explicitly changed to
  preprocess_mode = minimal_resample
  norm_mode = per_sample_global
the three-arm PN2021 result becomes:
  A = 0.7077 / 0.4027
  B = 0.7020 / 0.3993
  C = 0.7067 / 0.3937

Therefore the v2 PN2021 improvement should be reported with the exact
legacy-default evaluation口径, and re-optimized if the project standard switches
to minimal_resample PN2021 evaluation.
```

## Center-Token vNext Plan: Matched No-Token +2pp Target

### Goal

这一节专门服务后续 center-token 改进，不再把 2026-05-03 C0/C1 大增益线
解释成 center-token 因果结果。

目标改成更严格的 matched ablation：

```text
在同一 split、同一 reference ECG、同一 reference text、同一 target diagnosis、
同一 sampling seed、同一过滤器、同一下游训练 recipe 下，
center-token arm 的 macro AUPRC 至少比 no-token arm 高 2 个百分点。
```

Primary acceptance:

```text
target-token AUPRC - matched no-token AUPRC >= +0.0200
```

Secondary acceptance:

```text
target-token AUROC 不下降超过 0.005
official fold10 或 target-center PN2021 不出现明显回退
bootstrap 95% CI 的 AUPRC delta 下界 > 0 时才能写成稳定增益
```

### Why The Previous C0/C1 Evidence Is Not Enough

已有 no-token 消融说明：

```text
C1 no-token vanilla control:
  custom seed42 = 0.8670 / 0.6846
  fold10        = 0.8560 / 0.6695
  PN2021        = 0.7719 / 0.4750

C1 original PTBXL MV4 center-token:
  custom seed42 = 0.8645 / 0.6723
  fold10        = 0.7490 / 0.5106
  PN2021        = 0.7364 / 0.4195
```

因此旧 C0/C1 路线的失败点不是 ECGTwin synthetic pretraining 没用，而是：

```text
1. token 只用 diffusion reconstruction loss 学习，没有强制目标中心风格变强；
2. no-token prompt + translated report 已经很强，center token 没有额外信息优势；
3. candidate pool 不是 paired control，随机采样差异和 gate 差异会混进结论；
4. HYP/CD 低质量样本可能稀释 token 的有效类信号；
5. broad synthetic pretrain 会把 token 影响冲淡，真正能体现 token 的位置应该是
   filtered synthetic pool、self-distillation 或 latent-hull AT。
```

### Existing Positive But Limited Evidence

v2 self-distillation 里已有 center-token 相对 no-token 的正向证据：

```text
vanilla ECGTwin + v2:
  custom = 0.8333 / 0.6051
  fold10 = 0.8212 / 0.6005
  PN2021 = 0.7528 / 0.4233

PTBXL center-token ECGTwin + v2:
  custom = 0.8500 / 0.6371
  fold10 = 0.8405 / 0.6380
  PN2021 = 0.7616 / 0.4546
```

这个结果已经满足 legacy-default 口径下的 `>2pp vs vanilla`，但还不能作为最终
center-token 主结论，原因是：

```text
1. PN2021 minimal_resample/per_sample_global 口径下该优势没有复现；
2. 当前 v2 过滤不是严格 same-reference/same-seed paired filtering；
3. style validity 主要来自 downstream utility，还需要 no-leak center-style probe、
   C2ST 和 feature-distance 共同支撑。
```

所以 vNext 的目标不是重复 v2，而是把 v2 的正向信号做成更干净的因果实验。

### vNext Core Idea

center token 必须同时满足三件事：

```text
1. 语义不变：生成的 ECG 仍然像目标 super5 类别；
2. 风格增强：比 no-token/wrong-token 更像目标中心；
3. 下游有用：在同一 filtered pool 和同一训练 recipe 下比 no-token 多至少 2pp AUPRC。
```

因此 vNext 不再只训练一个 diffusion reconstruction token，而是训练
style-aware + semantic-preserving prompt token。

### Token Training Loss

当前代码已经支持的可执行第一版：

```text
scripts/ecgtwin_gen/train_center_prompt_tokens.py
methods/ecgtwin_gen/prompt_token/trainer.py

available losses:
  L_denoise      = ECGTwin diffusion noise MSE
  L_init         = token stays near class text embedding init
  L_orth         = multi-vector token diversity
  L_style        = frozen center-style classifier CE on decoded predicted x0
  L_semantic     = frozen EfficientNet super5 BCE on decoded predicted x0
```

vNext-A loss:

```text
L = L_denoise
  + lambda_init * L_init
  + lambda_orth * L_orth
  + lambda_style * L_style
  + lambda_semantic * L_semantic
```

Recommended first sweep on 4090D:

| run | token schema | style weight | semantic weight | aux timestep max | steps | reason |
|---|---|---:|---:|---:|---:|---|
| vNext-A | direct MV4 | 0.05 | 0.05 | 50 | 3000 | smallest change from best existing MV4 |
| vNext-B | direct MV8 | 0.05 | 0.05 | 50 | 3000 | more token capacity, still simple |
| vNext-C | factorized C2+K2+R4 | 0.05 | 0.05 | 50 | 3000 | separate center style, class semantics, residual |
| vNext-D | direct MV8 | 0.10 | 0.05 | 50 | 3000 | stronger style pressure |

Do not start with very large `lambda_style`; the decoded `pred_x0` at diffusion
training time is only an approximation. If style loss dominates, the token may
learn classifier artifacts instead of physiologic center style.

Future vNext-Beyond code change if vNext-A/D is insufficient:

```text
Add prototype/contrastive style loss:
  f = frozen EfficientNet or ECGTwin latent feature
  mu_target(center, class) = held-out target-center real prototype
  mu_neg(other centers, same class) = negative style prototypes

  L_proto = ||f(x0) - mu_target||_2^2
          + max(0, margin + sim(f, mu_neg) - sim(f, mu_target))

This is preferred over only using a center classifier if the classifier's
accuracy remains weak or if it overfits center-specific artifacts.
```

### Reference Policy

For target-center style tokens:

```text
reference ECG = target center K=500 anchor when possible
reference class = same primary super5 class when possible
reference text = actual_report / SNOMED-derived report text
target text = super5 diagnosis prompt + <target_center_CLASS>
```

For no-token controls:

```text
Use the exact same reference ECG, same reference text, same patient info,
same target diagnosis prompt, and same sampling seed.
Only omit the learned token.
```

This is required because otherwise the experiment measures random reference
selection rather than center-token effect.

### Paired Generation Arms

Every generated candidate should have paired arms:

| arm | prompt condition | purpose |
|---|---|---|
| no-token | diagnosis prompt only | strict baseline |
| target-token | diagnosis prompt + `<target_center_CLASS>` | main method |
| wrong-center | diagnosis prompt + `<other_center_CLASS>` | center-specificity control |
| PTBXL-source-token | diagnosis prompt + `<ptbxl_CLASS>` | source-style negative control |
| wrong-class-token | diagnosis prompt + `<target_center_OTHERCLASS>` | semantic leakage check |

The candidate manifest must store:

```text
pair_id
ref_record_id
ref_center
ref_primary_class
target_class
sampling_seed
arm
token_bank
quality gate results
style score
semantic scores
selected_for_downstream
```

### Gate And Selection

The selector should be pairwise, not independent per arm.

Hard gates:

```text
NaN/Inf absent
amplitude and flatline sanity pass
Einthoven/aVR residual pass
semantic target class probability passes class-specific threshold
NORM: p_NORM high and max abnormal low
HYP/CD: digital criteria are report-only until gates become reliable
```

Style gates:

```text
style_score_delta = P(target center | target-token)
                  - P(target center | no-token)

target-token kept only if:
  style_score_delta >= delta_min
  P(target center | target-token) > P(target center | wrong-center)
  C2ST distance to held-out target real <= no-token C2ST distance
```

Initial thresholds:

```text
delta_min = 0.05 for screening
delta_min = 0.10 for high-confidence pool
per_class_cap = 400 for v2 self-distill
max_keep_total = 2000
NORM/MI/STTC are first-class production classes
HYP/CD start as ablation-only until digital gates pass
```

Diversity selection:

```text
After hard/style gates, run farthest-point or k-center selection in frozen
EfficientNet feature space within each class. This avoids keeping 400 nearly
duplicate high-style-score samples.
```

### Downstream Ablation Matrix

Primary matrix:

| ID | synthetic source | token | filtering | student recipe | primary metric |
|---|---|---|---|---|---|
| A | none | no | none | real2000 from scratch | reference baseline |
| B | paired vanilla ECGTwin | no | v2 + style/digital gates | self-distill v2 | no-token control |
| C | paired target-token ECGTwin | yes | same v2 + style/digital gates | self-distill v2 | must beat B by >=2pp AUPRC |
| D | paired wrong-center token | wrong | same gates | self-distill v2 | should not beat C |
| E | target-token accepted latents | yes | same gates | Latent-Hull online AT | utility add-on |

Primary target for PTB-XL seed42 v2:

```text
matched no-token v2 custom AUPRC = 0.6051
target-token vNext custom AUPRC target >= 0.6251

matched no-token v2 fold10 AUPRC = 0.6005
target-token vNext fold10 AUPRC target >= 0.6205

matched no-token v2 PN2021 AUPRC = 0.4233
target-token vNext PN2021 AUPRC target >= 0.4433
```

Stricter C0/C1 route target:

```text
matched no-token C1 custom AUPRC = 0.6846
center-token C1 would need custom AUPRC >= 0.7046

This is much harder. Do not use C0/C1 as the first center-token target.
Use v2 filtering and/or Latent-Hull AT first, where center-token effects are
less diluted by broad synthetic pretraining.
```

### Recommended Execution Order

1. Train or select no-leak style probes with full 1000-sample ECG input, K=500
   anchors excluded, and same-label binary target-vs-other splits.
2. Train vNext-A and vNext-D first because they use already implemented
   `style_loss_weight` and `semantic_loss_weight`.
3. Generate paired no-token/target-token/wrong-token candidates with identical
   references and seeds.
4. Run pairwise style + semantic + digital + diversity selection.
5. Train v2 self-distillation students for B/C/D.
6. If C beats B by at least 2pp AUPRC, run fold10, PN2021, and bootstrap CI.
7. Only after v2 passes, try Latent-Hull online AT with selected token latents.
8. If v2 fails, inspect whether failure is style-score, semantic-score,
   diversity, or downstream-training related before changing token capacity.

### First Concrete Commands To Materialize vNext-A

Use existing code paths first. The first executable target should be a PN2021
center whose label is present in the no-leak style classifier. PTB-XL source
style-token training can use the same recipe only after the style probe includes
`ptbxl_source` as one of its labels.

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python

$PY -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --cache_root /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1 \
  --prompt_bank /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt \
  --save_dir /root/autodl-tmp/graduate_project/center_token_vnext/pn2021_big4_mv4_style005_sem005_steps3000 \
  --K 500 \
  --seed 42 \
  --total_steps 3000 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy center_class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 4 \
  --token_orth_weight 1e-4 \
  --ref_text_mode actual_report \
  --style_ckpt /root/autodl-tmp/center_style_classifier_noleak/full1000/best_model.pt \
  --style_loss_weight 0.05 \
  --style_crop_len 1000 \
  --semantic_ckpt /root/autodl-tmp/graduate_project/method_a_real2000_seed44_v2teacher/best_model.pt \
  --semantic_loss_weight 0.05 \
  --semantic_crop_len 1000 \
  --aux_timestep_max 50 \
  --log_every 50 \
  --save_every 500
```

If the no-leak style checkpoint above does not exist, create it first from
`docs/pipelines/center_style_classifier_pipeline.md`; do not fall back to the old
250-point 7-way classifier for final evidence.

PTB-XL source-token variant:

```text
Use --centers ptbxl_source only with a style classifier whose center_names
contains ptbxl_source. Otherwise set --style_loss_weight 0 and treat the run as
semantic-preserving only, not style-aware.
```

### 2026-05-04 vNext-A Execution Record

No-leak full-1000 PN2021 big4 style auxiliary classifier:

```text
artifact:
  /root/autodl-tmp/center_style_classifier_noleak/full1000

script:
  scripts/ecgtwin_gen/train_noleak_center_style_aux_classifier.py

centers:
  ningbo, chapman_shaoxing, cpsc_2018, georgia

classes:
  NORM, MI, STTC

input:
  PN2021 mmap minimal_resample/per_sample_global, 100Hz, 1000 samples

leak control:
  K=500 center-token anchors excluded per center

split:
  train = 11517
  val   = 2466
  test  = 2475

test accuracy = 0.6279
test macro F1 = 0.6555
best val macro F1 = 0.6713
```

Important caveat:

```text
This style classifier is an auxiliary differentiable training signal and a
fast probe. It is stronger than the old 250-point/7-way probe, but it is still
not final proof of clinical style validity. Final evidence still needs paired
no-token controls, C2ST/feature distance, and downstream AUROC/AUPRC.
```

vNext-A PN2021 big4 token training:

```text
artifact:
  /root/autodl-tmp/graduate_project/center_token_vnext/pn2021_big4_mv4_style005_sem005_steps3000

token bank:
  prompt_token_bank.pt

schema:
  direct MV4
  centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
  classes = CD, HYP, MI, NORM, STTC

loss:
  L_denoise + 1e-3 L_init + 1e-4 L_orth
  + 0.05 L_style + 0.05 L_semantic

runtime:
  3000 steps, batch 16, bf16, 4 workers
  elapsed ~= 339 seconds on RTX 4090D
```

Final training metrics:

```text
step = 3000
loss = 0.10625
recon_loss = 0.05615
style_loss = 0.49121
semantic_loss = 0.51043
orth_loss = 0.11677
avg token norm = 1.9972
avg token delta norm = 1.8033
```

Token diagnostic artifact:

```text
/root/autodl-tmp/graduate_project/center_token_vnext/diagnostics/token_summary.csv
/root/autodl-tmp/graduate_project/center_token_vnext/diagnostics/token_cosine.csv

avg token norm = 1.8923
avg token delta norm = 1.6347
avg cosine to init = 0.4866
```

Per-center delta summary:

| center | avg token delta norm | note |
|---|---:|---|
| ningbo | 1.8576 | active NORM/MI/STTC token movement |
| chapman_shaoxing | 1.9294 | active token movement despite few MI anchors |
| cpsc_2018 | 1.0931 | HYP/MI stayed near init because selected anchors had no HYP/MI |
| georgia | 1.6588 | active NORM/STTC token movement |

Generator compatibility fix:

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth_batched.py

PN2021 center caches do not always contain source_indices. The batched
generator now falls back to the cache index when source_indices is absent, so
PN2021 target-center smoke generation can write summary.json records.
```

Paired smoke: georgia target-token vs no-token:

```text
center = georgia
classes = NORM, STTC
n_per_class = 50
steps = 25
reference policy = same_class
ref_text_mode = actual_report
seed = 4242

token pool:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_georgia_token/georgia/samples.npz

no-token pool:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_georgia_no_token/georgia/samples.npz

style probe output:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_georgia_style_probe/

style probe script:
  scripts/ecgtwin_gen/score_prompt_token_style_probe.py
```

Semantic target-probability smoke:

| arm | class | n | mean target probability |
|---|---|---:|---:|
| no-token | NORM | 50 | 0.7737 |
| target-token | NORM | 50 | 0.9954 |
| no-token | STTC | 50 | 0.6571 |
| target-token | STTC | 50 | 0.8138 |

No-leak style probe smoke:

| arm | class | n | mean P(georgia) | top1 georgia |
|---|---|---:|---:|---:|
| no-token | NORM | 50 | 0.6175 | 0.72 |
| target-token | NORM | 50 | 0.9948 | 1.00 |
| no-token | STTC | 50 | 0.6022 | 0.60 |
| target-token | STTC | 50 | 0.9829 | 0.98 |

Expanded big4 paired smoke:

```text
token pools:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_big4_token/<center>/samples.npz

no-token pools:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_big4_no_token/<center>/samples.npz

style probe output:
  /root/autodl-tmp/graduate_project/center_token_vnext/smoke_big4_style_probe/
```

Expanded big4 smoke target-class probability:

| center | class | no-token p_target | target-token p_target | delta |
|---|---|---:|---:|---:|
| ningbo | NORM | 0.8324 | 0.9583 | +0.1259 |
| ningbo | STTC | 0.6303 | 0.6904 | +0.0601 |
| chapman_shaoxing | NORM | 0.8487 | 0.8598 | +0.0111 |
| chapman_shaoxing | STTC | 0.7047 | 0.9564 | +0.2517 |
| cpsc_2018 | NORM | 0.9300 | 0.9923 | +0.0623 |
| cpsc_2018 | STTC | 0.6505 | 0.8436 | +0.1931 |
| georgia | NORM | 0.7737 | 0.9954 | +0.2217 |
| georgia | STTC | 0.6571 | 0.8138 | +0.1567 |

Expanded big4 smoke target-center style probe:

| center | class | no-token P(center) | token P(center) | no-token top1 | token top1 |
|---|---|---:|---:|---:|---:|
| ningbo | NORM | 0.1100 | 0.8365 | 0.02 | 0.90 |
| ningbo | STTC | 0.0362 | 0.7781 | 0.00 | 0.88 |
| chapman_shaoxing | NORM | 0.2238 | 0.5072 | 0.20 | 0.60 |
| chapman_shaoxing | STTC | 0.0407 | 0.5937 | 0.04 | 0.92 |
| cpsc_2018 | NORM | 0.0957 | 0.8999 | 0.08 | 0.94 |
| cpsc_2018 | STTC | 0.4567 | 0.9448 | 0.48 | 0.98 |
| georgia | NORM | 0.6175 | 0.9948 | 0.72 | 1.00 |
| georgia | STTC | 0.6022 | 0.9829 | 0.60 | 0.98 |

Expanded smoke interpretation:

```text
vNext-A has a consistent positive style-probe effect across all four target
centers for NORM/STTC. This is materially stronger than the old prompt-token
evidence because every target-token pool has a same-center, same-reference,
same-seed no-token control.

However, the smoke still only validates the direction of the generative
conditioning. It does not prove the requested +2pp downstream AUPRC gain.
Proceed to a larger paired pool and a matched no-token vs target-token
self-distillation or Latent-Hull AT ablation.
```

### 2026-05-04 Matched Downstream Ablation: Negative Result

This block records the first real downstream test after the positive vNext-A
style smoke.

Candidate generation:

```text
centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
classes = NORM, STTC
n_per_class_per_center = 250
per arm total = 2000 synthetic ECG
same reference policy = same_class
same seed = 5252
same target text = super5 diagnosis prompt
only difference = with or without vNext-A center token
```

Candidate pools:

```text
target-token:
  /root/autodl-tmp/graduate_project/center_token_vnext/downstream_n250_token/merged/samples.npz

no-token:
  /root/autodl-tmp/graduate_project/center_token_vnext/downstream_n250_no_token/merged/samples.npz
```

v2 semantic filter:

```text
teachers:
  method_a_real2000_seed42
  method_a_real2000_seed43_v2teacher
  method_a_real2000_seed44_v2teacher

gamma = 0.3
tau_high = 0.6
tau_low = 0.35
norm_tau_high = 0.7
norm_tau_low = 0.5
per_class_cap = 1000
```

Filter result:

| arm | candidates | kept | high | hard | NORM kept | STTC kept |
|---|---:|---:|---:|---:|---:|---:|
| no-token | 2000 | 1572 | 1391 | 181 | 916 | 656 |
| target-token | 2000 | 1814 | 1679 | 135 | 980 | 834 |

Student recipe:

```text
script = scripts/triple_labels/train_ptbxl_self_distill.py
real data = PTB-XL seed42 train2000
teacher = method_a_real2000_seed42
synth_ratio = 1.0
synth_distill_weight = 0.5
soft_loss_mode = bce_soft
epochs = 50
checkpoint_metric = auprc
```

Result:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| no-token | 0.8511 | 0.6330 | 0.8433 | 0.6309 | 0.7494 | 0.4242 |
| target-token | 0.8400 | 0.6125 | 0.8306 | 0.6130 | 0.7471 | 0.4160 |
| token - no-token | -0.0111 | -0.0205 | -0.0127 | -0.0179 | -0.0023 | -0.0082 |

Interpretation:

```text
The center token strongly improves synthetic target-center style and increases
semantic filter pass rate, but direct self-distillation with the resulting
NORM/STTC target-center-style synthetic pool hurts downstream performance.

This means style transfer alone is not sufficient. Strong target-center style
can become a domain artifact or over-specialized shortcut when mixed into a
PTB-XL source-supervised diagnosis learner.
```

### 2026-05-04 Style-Top1000 Ablation: Also Negative

To check whether the failure came from unequal keep counts or weak no-token
style, a second subset selected the top target-center style samples after the
v2 semantic filter:

```text
selection = top 500 NORM + top 500 STTC by no-leak target-center style score
same count for target-token and no-token
```

Selected style score:

| arm | selected | mean style score |
|---|---:|---:|
| no-token | 1000 | 0.4324 |
| target-token | 1000 | 0.9916 |

Student result:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| no-token top-style | 0.8381 | 0.6059 | 0.8266 | 0.5887 | 0.7513 | 0.4208 |
| target-token top-style | 0.8283 | 0.5757 | 0.8137 | 0.5724 | 0.7305 | 0.3980 |
| token - no-token | -0.0098 | -0.0302 | -0.0129 | -0.0163 | -0.0208 | -0.0228 |

Interpretation:

```text
The strongest target-center-looking token samples are the worst downstream
choice. Therefore the next version must not optimize for maximum style score.
It should target bounded, on-manifold style perturbations that keep diagnosis
semantics and source/target compatibility.
```

### vNext-B Revision After Negative Downstream

The next center-token strategy should change from direct synthetic expansion to
bounded target-style latent augmentation.

Design changes:

```text
1. Do not use style score as a monotonic objective.
   Use a band such as 0.45 <= P(target center) <= 0.85, and reject samples above
   0.95 unless they are only used as style prototypes.

2. Do not use NORM/STTC-only synthetic expansion as the main student data.
   Missing CD/HYP/MI coverage hurts macro metrics and encourages class-specific
   shortcuts.

3. Use target-token samples as latent candidates inside real-anchor Latent-Hull
   AT, not as standalone training data:

     z_adv = (1 - lambda) * z_real + lambda * sum_i softmax(a_i) * z_token_i

   Recommended lambda = 0.05, 0.10, 0.20 rather than 0.25 first.

4. Use teacher boundary gate on generated/decoded samples:
   target class probability in 0.45-0.70, not near 0.99.

5. Add per-class and per-center caps before downstream:
   NORM/STTC max 150 each per center;
   MI only where same-class anchors exist;
   CD/HYP ablation-only until digital gates pass.

6. Compare against no-token under the same latent-hull/gating process.
   The only intended difference should be whether the synthetic candidate
   latents came from target-token or no-token generation.
```

vNext-B acceptance:

```text
Target-token Latent-Hull AT must beat no-token Latent-Hull AT by >= 2pp AUPRC
on either:
  1. the four target centers averaged, or
  2. the full PN2021 seven-center average,
without losing more than 0.5pp PTB-XL custom AUROC.
```

Current status:

```text
Not achieved. vNext-A proves controllable target-center style generation, but
the first two downstream matched ablations are negative. The next productive
route is bounded style + real-anchor latent-hull AT, not stronger direct
synthetic self-distillation.
```

Earlier smoke interpretation:

```text
The georgia-only smoke was the first positive vNext-A style signal. With matched
references and the same generation seed, the style-aware center token increases
both target-class probability and target-center style probability versus
no-token.

The later downstream matched ablations show that this style signal does not
automatically convert to AUROC/AUPRC gain.
```

### Claim Rules

Allowed claim if the matched ablation passes:

```text
With paired references and identical downstream training, the style-aware
center-token synthetic pool improves macro AUPRC by at least 2pp over no-token
ECGTwin synthetic data.
```

Not allowed:

```text
The C0/C1 +3pp/+7pp gain proves center token works.
The style classifier alone proves medical validity.
The target-token pool is better if it only wins after using a different filter,
different reference distribution, or different synthetic count.
```
