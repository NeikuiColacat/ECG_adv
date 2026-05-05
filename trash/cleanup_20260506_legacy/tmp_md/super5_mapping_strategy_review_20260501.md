# Super5 标签映射策略临时说明

日期：2026-05-01

本文是为了可读性整理的临时报告。PN2021 v3 修复已在代码中实施。正式长期维护版本见：

```text
docs/pipelines/super5_label_mapping_pipeline.md
scripts/triple_labels/label_schemes.py
```

## 1. 当前 super5 是什么

我们当前使用的五类任务是：

```text
CD, HYP, MI, NORM, STTC
```

含义：

| 类别 | 含义 | 通俗解释 |
|---|---|---|
| CD | Conduction Disturbance | 传导异常，例如束支传导阻滞、房室传导阻滞 |
| HYP | Hypertrophy | 心室/心房肥大或高电压相关表现 |
| MI | Myocardial Infarction | 心肌梗死相关诊断 |
| NORM | Normal ECG | 正常心电图 |
| STTC | ST/T Change | ST 段、T 波、缺血/复极异常 |

这五类来自 PTB-XL 官方 `diagnostic_class`。因此：

```text
PTB-XL super5 = 官方 diagnostic superclass
PN2021 super5 = 我们自定义的 SNOMED -> PTB-XL super5 语义投影
MIMIC super5 = 我们用 report regex 得到的弱伪标签
```

## 2. PTB-XL 当前策略

PTB-XL 使用：

```text
ptbxl_database.csv: scp_codes
scp_statements.csv: diagnostic_class
```

当前代码逻辑：

1. 解析每条 ECG 的 `scp_codes`。
2. 只保留 `scp_statements.csv` 中 `diagnostic == 1.0` 的 SCP code。
3. 读取对应的 `diagnostic_class`。
4. 转成固定顺序 `CD/HYP/MI/NORM/STTC` 的 multi-hot label。

当前 `confidence_threshold=0.0` 是合理的。PTB-XL 官方 example 也是按 `scp_codes`
的 key 是否存在来聚合 `diagnostic_class`，没有按 likelihood value 过滤。

当前全量 PTB-XL 计数：

```text
CD=4898
HYP=2649
MI=5469
NORM=9514
STTC=5235
```

这个计数与 PhysioNet 页面公布的 diagnostic superclass distribution 一致。

注意点：

- PTB-XL 原始 `scp_codes` 里的 value 是 likelihood，`0` 表示 unknown likelihood。
- 我们训练 label 里的 `0.0` 是 multi-hot negative。
- 这两个 `0` 不是一个概念。
- 当前 trainer 保留 411 条没有 diagnostic superclass 的 all-zero 样本；一些 benchmark 会过滤这些样本。

## 3. PN2021 当前映射策略

PN2021 官方标签是 SNOMED-CT code。官方没有提供到 PTB-XL super5 的映射，所以现在是我们项目自定义映射。

当前代码位置：

```text
scripts/triple_labels/label_schemes.py
SNOMED_TO_SUPER5
```

### NORM

v3 直接 NORM 候选只保留：

| SNOMED | 当前注释 |
|---:|---|
| 426783006 | sinus rhythm |

问题：

`sinus bradycardia`、`sinus tachycardia`、`sinus arrhythmia` 更像“节律描述”，不一定等于 PTB-XL 里的“正常心电图”。例如一个人可以同时有 sinus rhythm 和其他异常诊断。v3 已将这些节律变体放入 `NORM_SUPPRESS_SNOMEDS`，不再直接产生 `NORM=1`。

v3 同时把 AF/AFL/PAC/PVC/LAD/RAD/low voltage/PRWP 这类“不正常但不属于 super5 阳性类”的标签放入 `NORM_SUPPRESS_SNOMEDS`，用于取消 NORM。

### MI

当前映射为 MI：

| SNOMED | 当前注释 |
|---:|---|
| 164865005 | myocardial infarction |
| 164867002 | old myocardial infarction |
| 57054005 | acute myocardial infarction |
| 54329005 | anterior MI |
| 22298006 | subacute MI |
| 401303003 | acute anterior MI alt |
| 233843008 | inferior MI alt |

主要风险：

- MI 这一组医学语义相对清楚。
- 但 `22298006`、`401303003`、`233843008` 没在已核查的 Challenge 2021 官方 scored/unscored mapping 文件里出现。若保留，需要注明是扩展兼容 SNOMED，不是 PN2021 官方词表核心项。

### STTC

v3 当前映射为 STTC：

| SNOMED | 当前注释 |
|---:|---|
| 164934002 | T wave abnormal |
| 111975006 | prolonged QT |
| 164931005 | ST elevation |
| 429622005 | ST depression |
| 164930006 | ST interval abnormal |
| 59931005 | T wave inversion |
| 164861001 | myocardial ischemia |
| 428750005 | nonspecific ST-T abnormality |
| 55930002 | ST changes |
| 425623009 | lateral ischemia |
| 425419005 | inferior ischemia |
| 426434006 | anterior ischemia |

主要风险：

- ST/T 改变、ST elevation/depression、T wave abnormal、ischemia 放进 STTC 基本合理。
- `Q wave abnormal` 不一定是 STTC，可能更接近 MI 证据或 form-only 描述。v3 已改为 suppress-only。
- `early repolarization` 可能是良性变异，不应默认当作 STTC 强阳性。v3 已改为 suppress-only。

如果要把 `Q wave abnormal` 和 `early repolarization` 保留为 STTC，应作为 broad mapping ablation。

### CD

当前映射为 CD：

| SNOMED | 当前注释 |
|---:|---|
| 270492004 | 1st degree AV block |
| 195042002 | 2nd degree AV block |
| 54016002 | Mobitz I |
| 426183003 | Mobitz II |
| 27885002 | 3rd degree AV block |
| 233917008 | AV block generic |
| 164947007 | prolonged PR interval |
| 164909002 | LBBB |
| 733534002 | complete LBBB |
| 59118001 | RBBB |
| 713427006 | complete RBBB |
| 251120003 | incomplete LBBB |
| 713426002 | incomplete RBBB |
| 6374002 | bundle branch block |
| 445118002 | left anterior fascicular block |
| 445211001 | left posterior fascicular block |
| 698252002 | nonspecific IV conduction block |
| 10370003 | pacing rhythm |
| 251268003 | atrial pacing pattern |
| 251266004 | ventricular pacing pattern |
| 74390002 | WPW |
| 26749005 | WPW alternate code |
| 195060002 | ventricular pre-excitation |

主要风险：

- 束支阻滞、房室传导阻滞、室内传导阻滞、WPW 放进 CD 基本合理。
- pacing rhythm 在 PTB-XL 里更像 rhythm/form 信息，不一定是 diagnostic `CD`。可保守改成 suppress-only 或保留为 CD ablation。
- `26749005` 没在已核查的 Challenge 2021 官方 scored/unscored mapping 文件里出现，若保留需要说明来源。

### HYP

当前映射为 HYP：

| SNOMED | 当前注释 |
|---:|---|
| 164873001 | left ventricular hypertrophy |
| 55827005 | left ventricular high voltage |
| 89792004 | right ventricular hypertrophy |
| 266249003 | ventricular hypertrophy generic |
| 446358003 | right atrial hypertrophy / RAE |
| 446813000 | left atrial hypertrophy / LAE |
| 67741000119109 | left atrial enlargement alt |
| 67751000119106 | right atrial high voltage |
| 195126007 | atrial hypertrophy |
| 164828000 | atrial hypertrophy alt |

主要风险：

- LVH/RVH/ventricular hypertrophy 放进 HYP 基本合理。
- high voltage、atrial enlargement/hypertrophy 与 PTB-XL HYP 的边界可能不完全一致。
- `164828000` 没在已核查的 Challenge 2021 官方 scored/unscored mapping 文件里出现。

## 4. 当前策略最大的问题

最大问题不是“有没有五类”，而是“跨数据集五类是不是同一个医学语义”。

PTB-XL 的 super5 是官方定义，比较可靠。

PN2021 的 SNOMED 标签来自不同数据源和不同标注体系。我们把它们压缩成 PTB-XL 五类，会丢掉很多信息：

- AF/AFL 是心律失常，但 super5 没有 rhythm 类。
- LAD/RAD 是电轴异常，但 super5 没有 axis 类。
- PAC/PVC 是早搏，但 super5 没有 ectopy 类。
- low QRS voltage、PRWP、abnormal QRS 也没有干净的 super5 对应类。

如果这些标签不映射到任何 super5 类，但记录里又有 sinus rhythm，那么当前代码可能把它当作 `NORM=1`。这会污染 PN2021 的 NORM 评测。

## 5. 推荐解决方案

PN2021 v3 映射已拆成三层：

```python
SNOMED_TO_SUPER5_POSITIVE = {
    # 只放明确属于 CD/HYP/MI/STTC 的异常 code
}

NORM_POSITIVE_SNOMEDS = {
    426783006,  # sinus rhythm
}

NORM_SUPPRESS_SNOMEDS = {
    # 不直接产生 super5 阳性，但说明这条 ECG 不应算 NORM
}
```

决策逻辑：

```python
label = zeros(5)

for code in snomed_codes:
    if code in SNOMED_TO_SUPER5_POSITIVE:
        label[mapped_super5_class] = 1
    if code in NORM_POSITIVE_SNOMEDS:
        norm_candidate = True
    if code in NORM_SUPPRESS_SNOMEDS:
        norm_suppress = True

if norm_candidate and no abnormal super5 class and not norm_suppress:
    label[NORM] = 1
else:
    label[NORM] = 0
```

首批 `NORM_SUPPRESS_SNOMEDS` 应覆盖：

```text
AF/AFL
PAC/SVPB
PVC/VPB
LAD/RAD
LQRSV
PRWP
Brady
abnormal QRS
atrial/junctional/idioventricular rhythm abnormalities
Q wave abnormal
early repolarization
pacing rhythm, if not treated as direct CD positive
```

v3 已完成：

1. bump PN2021 cache version 为 `v3_super5_normsuppress`。
2. cache metadata 保存 mapping version/hash、class order、cache version 和 preprocess config。
3. 在论文中写清楚 PN2021 是 custom semantic projection，不是官方 super5。

仍需要实验侧执行：

1. 重建 `/root/autodl-tmp/triple_labels/pn2021_eval_cache` 中旧 v2 cache 对应的新 v3 cache。
2. 重新跑 PN2021 7-center eval。
3. 后续可继续增加 unmapped SNOMED 统计。

## 6. 如果 MIMIC regex 标签不可信，ECGTwin 作者是怎么用 MIMIC 的

关键区别：

```text
我们的 MIMIC super5 regex = 把 MIMIC report 硬映射成五类分类标签
ECGTwin 作者 = 不把 MIMIC report regex 成 super5，而是直接使用报告文本作为生成条件
```

ECGTwin 作者的数据构建流程：

1. 从 MIMIC-IV-ECG 读取 ECG 波形、`record_list.csv` 和 `machine_measurements.csv`。
2. 把 `report_0..report_17` 中非空文本用 `|` 拼起来，作为 `label['text']`。
3. 同时保留 `subject_id`、`ecg_time`、heart rate、age、sex。
4. ECG 被 VAE 编码成 latent。
5. 报告文本用 `nomic-ai/nomic-embed-text-v1.5` 编码成 768 维 text embedding。
6. 同一个病人有多条 ECG 时，按时间组成 reference-target pairs。

代码依据：

```text
model/ECGTwin/data/mimic_iv_ecg_dataset.py
model/ECGTwin/data/store_embedding_nomic.py
model/ECGTwin/data/dataset_construction.py
model/ECGTwin/trainer/IBETrainer.py
model/ECGTwin/trainer/ECGTwinTrainer.py
model/ECGTwin/utils/training_utils.py
```

ECGTwin 的 IBE 阶段：

- 输入同一病人的两条 ECG latent。
- 输入两条 ECG 各自的 report text embedding、HR、age、sex。
- 用 contrastive loss 训练 Individual Base Extractor，让同一个人的两次记录提取出一致的 base vector。
- 训练时有 0.15 概率把 text embedding 置空，用来增强鲁棒性。

ECGTwin 的 diffusion 阶段：

- 参考 ECG 经过 IBE 得到 `base_vector`。
- 目标 ECG 的 report text embedding + HR/age/sex 作为条件。
- DiT/Unet 学习预测目标 latent 加噪后的 noise。
- loss 是扩散噪声预测 MSE，不是 super5 分类 BCE。

所以，如果 MIMIC regex super5 不可信，并不直接否定 ECGTwin 作者做法。作者没有用 regex 生成五类 one-hot 作为监督标签，而是把原始机器报告文本当作条件描述，配合同一病人的纵向 ECG pair 来训练 personalized generation。

但这也有局限：

- MIMIC machine report 文本仍然是弱文本条件，里面可能有错误、否定、历史诊断、不确定诊断。
- ECGTwin 依赖大规模数据和文本编码器的语义鲁棒性，不等于每条 report 都是干净标签。
- 因此我们不能把 MIMIC regex-super5 当成主监督，但可以借鉴 ECGTwin：用 MIMIC report text 做生成条件或预训练条件，再用 PTB-XL/PN2021 做更清晰的分类验证。

## 7. 对我们毕设的建议表述

推荐答辩/论文口径：

```text
本研究的主监督标签采用 PTB-XL 官方 diagnostic superclass 五分类。
PN2021 用作跨中心外部评测，其 SNOMED 标签被投影到 PTB-XL super5 空间，
该投影是本研究定义的语义映射，并非官方映射。
MIMIC-IV-ECG 的 report-derived 标签仅作为弱伪标签或生成条件，不作为主监督证据。
```

如果后续修代码，优先顺序：

1. 重跑 PN2021 7-center eval，生成 v3 结果。
2. 做 P1/P2 sensitivity analysis。
3. 再考虑是否改 MIMIC regex 标签缓存。
