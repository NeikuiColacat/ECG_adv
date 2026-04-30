# Super5 Label Mapping Pipeline

本文档定义 PTB-XL、PN2021、MIMIC 在 `super5` 标签空间中的标准映射规则。

## 类顺序

所有 super5 相关模型、缓存和合成数据必须使用固定顺序：

```text
CD, HYP, MI, NORM, STTC
```

源码事实依据：

```text
scripts/triple_labels/label_schemes.py
```

## 标签值语义

```text
1.0  positive
0.0  known negative
-1.0 unknown / uncovered, masked in loss and metric
```

当前 super5 的 PTB-XL、PN2021、MIMIC 映射都应输出 0/1。只有更细粒度 scheme 在部分数据集可能输出 -1。

## PTB-XL -> Super5

来源：

```text
/root/autodl-tmp/ptbxl/ptbxl_database.csv: scp_codes
/root/autodl-tmp/ptbxl/scp_statements.csv: diagnostic_class
```

标准规则：

1. 解析 `scp_codes` 字典。
2. 只使用 `scp_statements.csv` 中 `diagnostic == 1.0` 的诊断 SCP。
3. 读取每个 SCP 的 `diagnostic_class`。
4. 映射到 `CD/HYP/MI/NORM/STTC`。
5. 输出 `(5,) float32` multi-hot。

PTB-XL 是 super5 训练标签的最高优先级来源，因为 super5 本身来自 PTB-XL 官方 `diagnostic_class`。

当前实现细节：

- `ptbxl_scp_to_super5()` 默认使用 `confidence_threshold=0.0` 且条件是 `conf >= threshold`。
- 这意味着只要 SCP key 存在并且在 `scp_statements.csv` 中是 diagnostic code，就会计入 super5；包括 likelihood/confidence 等于 0 的诊断 key。
- 不要在未记录新实验版本的情况下改成 `conf > 0`、`conf >= 1` 或其他阈值。阈值改变会改变 STTC 等类别阳性数，必须重建标签 cache 并重新报告 per-class label distribution。

缓存失效规则：

```text
If PTB-XL mapping rule, confidence threshold, scp_statements.csv, or class order changes:
    delete/regenerate ptbxl_labels.C5.all
    rerun label_schemes.py --sanity
    record new per-class positive counts
```

## PN2021 -> Super5

来源：

```text
PN2021 .hea header: #Dx SNOMED code list
```

标准规则：

1. 从 `.hea` 中读取 `#Dx:` SNOMED 列表。
2. 使用 `SNOMED_TO_SUPER5` 映射到五类。
3. 输出 `(5,) float32` multi-hot。
4. 强制 NORM exclusivity guard：

```text
if any(CD, HYP, MI, STTC) == 1:
    NORM = 0
```

原因：PN2021 经常同时标注 sinus rhythm 和异常诊断；PTB-XL 训练语义中 `NORM` 更接近“未发现异常”。不加 guard 会导致 NORM 的跨中心语义漂移。

当前已知限制：

- 现有 `snomed_list_to_super5()` 只在 SNOMED 已映射到 `CD/HYP/MI/STTC` 时压制 `NORM`。
- 对 AF/AFL/PAC/PVC/LAD/RAD/low voltage 等未映射到 super5 但明显非正常的 PN2021 诊断，当前不会压制 `NORM`。
- 下一版标签策略应把“映射到 super5 阳性类”和“取消 NORM 的异常证据”拆开，引入 `NORM_SUPPRESS_SNOMEDS`。
- `Q wave abnormal`、`early repolarization` 等边界 SNOMED 不应随意归入 STTC；优先作为 NORM-suppress 或单独 ablation，并在代码表中注明和 PTB-XL `diagnostic_class` 的对齐理由。

PN2021 cache 注意事项：

```text
If SNOMED_TO_SUPER5, NORM guard, NORM_SUPPRESS_SNOMEDS, or preprocessing changes:
    bump PN2021_EVAL_CACHE_VERSION
    rebuild /root/autodl-tmp/triple_labels/pn2021_eval_cache
    rerun 7-center eval and compare per-center/per-class deltas
```

PN2021 评测中心固定为：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

硬排除：

```text
ptb-xl
ptbxl
```

## MIMIC -> Super5

来源：

```text
/root/autodl-tmp/MIMIC/record_list.csv
/root/autodl-tmp/MIMIC/machine_measurements.csv report_0..report_17
```

标准规则：

1. 用 `study_id` 合并 record list 和 machine measurements。
2. 拼接 `report_0..report_17` 为 `report_text`。
3. 使用正则规则匹配五类：
   - `NORM`: sinus rhythm, NSR, normal ECG, normal sinus 等。
   - `MI`: myocardial infarct, STEMI/NSTEMI, old/recent/acute infarct 等。
   - `STTC`: ST elevation/depression, ST-T change, T wave abnormal, ischemia, long QT 等。
   - `CD`: BBB, LBBB/RBBB, AV block, fascicular block, IVCD, WPW, paced rhythm 等。
   - `HYP`: ventricular/atrial hypertrophy, LVH/RVH, atrial enlargement 等。
4. 输出 `(5,) float32` multi-hot。
5. 采用和 PN2021 一致的 NORM exclusivity guard：

```text
if any(CD, HYP, MI, STTC) == 1:
    NORM = 0
```

当前实现：`mimic_report_to_super5()` 位于 `scripts/triple_labels/label_schemes.py`，正则匹配后会应用上述 NORM exclusivity guard。

MIMIC 定位和风险：

- MIMIC report-super5 是 weak-label regex，不是 PTB-XL/PN2021 级别的人工诊断监督。
- `possible/cannot rule out/rule out/history of` 等语境可能把不确定或既往诊断标成阳性，尤其影响 MI。
- `NORM` guard 会放大异常 regex 误报的影响：一旦 MI/STTC/CD/HYP 被误报，`NORM` 会被强制置 0。
- 因此 MIMIC super5 应定位为 noisy external reference、可选预训练或消融，不作为 thesis 主监督训练证据。
- `/root/autodl-tmp/mimic_tierM/mimic_index.npz` 中的 `labels_6` 是历史 Tier-M 标签；super5 评测只复用其中的 `valid_mask/split/subject_ids`，不复用 `labels_6`。

建议后续增加版本化 MIMIC super5 标签缓存：

```text
/root/autodl-tmp/mimic_super5/mimic_super5_labels_v<regex_version>_normguard.npz
keys: labels5, study_id, subject_id, split, valid_mask, regex_version, created_at
```

## 合成数据标签

ECGTwin 生成的 super5 synthetic `.npz` 必须保存：

```text
signals: (N, 1000, 12), float32, PTB-XL lead order, 100 Hz, 10 s
labels:  (N, 5), float32, class order = CD/HYP/MI/NORM/STTC
```

单类生成推荐 one-hot。多标签生成按 multi-hot 保存，并且 `NORM` 不应与异常类共同为 1。

## 代码检查点

每次修改标签逻辑后至少检查：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/label_schemes.py --sanity --mimic_n 2000
```

并重新跑一个小规模训练 smoke test，确认：

- 标签维度和模型输出维度一致。
- 每类阳性数不为 0。
- `NORM` 与异常类共现率符合预期。
