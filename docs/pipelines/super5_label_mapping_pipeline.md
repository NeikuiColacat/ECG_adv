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
