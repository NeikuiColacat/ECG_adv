# Super5 Label Mapping Pipeline

本文档定义 PTB-XL、PN2021、MIMIC 在 `super5` 标签空间中的标准映射规则。

## 事实核查结论

当前标签策略的核心问题不是 PTB-XL，而是把 PN2021/MIMIC 的标签投到
PTB-XL `super5` 空间时存在语义降级：

| 数据源 | 当前结论 | 风险等级 | 处理原则 |
|---|---|---:|---|
| PTB-XL | 官方提供 `diagnostic_class`，天然支持 `NORM/MI/STTC/CD/HYP` 五个 diagnostic superclass | 低 | 作为 super5 主监督来源 |
| PN2021 | 官方提供 SNOMED-CT scored/unscored labels，不提供官方 `PN2021 -> PTB-XL super5` 映射 | 高 | 只能称为本项目自定义语义映射，用于外部中心评测 |
| MIMIC | 当前是 report regex weak-label，不是人工诊断强监督 | 中 | 只能作为 noisy pretraining、外部参考或消融 |
| synthetic `.npz` | loader 只转 shape 和 crop，不做预处理/导联修正/标签修正 | 中 | 导出前必须保证信号和标签已经 classifier-ready |

论文表述必须区分：

```text
PTB-XL super5 = official diagnostic superclass labels.
PN2021 super5 = custom SNOMED-to-PTB-XL-super5 semantic projection.
MIMIC super5 = weak report-regex pseudo-labels.
```

事实来源：

- PTB-XL PhysioNet 页面说明 `scp_codes` 覆盖 diagnostic/form/rhythm statements，并为
  diagnostic statements 提供 `diagnostic_class` / `diagnostic_subclass`。
- PTB-XL 官方 `example_physionet.py` 使用 `scp_statements.csv` 中
  `diagnostic == 1` 的条目聚合 `diagnostic_class`。
- PhysioNet/CinC 2021 官方页面说明 Challenge 标签是 SNOMED-CT codes，并提供
  scored/unscored label lists；官方评测目标不是 PTB-XL super5。
- 官方 Challenge 2021 `dx_mapping_scored.csv` 和 `dx_mapping_unscored.csv`
  是 PN2021 诊断词表来源，但不是 super5 crosswalk。

参考链接：

```text
https://physionet.org/content/ptb-xl/1.0.3/
https://physionet.org/files/ptb-xl/1.0.3/example_physionet.py
https://physionet.org/content/challenge-2021/1.0.3/
https://github.com/physionetchallenges/evaluation-2021
https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv
https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_unscored.csv
```

本地核查命令：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/label_schemes.py --sanity --mimic_n 2000
```

本轮任务的标签相关结论：

- 刚完成的 PN2021 v3 super5 重新映射会改变外部评测标签，因此所有要写进论文的
  PN2021 AUROC/AUPRC 都必须重跑。
- 重新评测计划见 `docs/pipelines/pn2021_v3_reevaluation_pipeline.md`。
- ECGTwin center prompt-token 的 ref pool、target-center sample pool、latent anchor pool 和
  PN2021-C corruption cache 都必须记录并继承该 mapping version/hash。
- 旧 `eval_result.json`、`eval_result_NORMguard.json`、`v2_normguard` cache 只能作为历史结果，
  不能继续作为当前主结果。

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

注意不要把这里的 `0.0` 和 PTB-XL 原始 `scp_codes` 中的 likelihood `0`
混淆。PTB-XL 文档中 `scp_codes` 的 value 是 statement likelihood，`0`
表示 unknown likelihood；官方 diagnostic superclass 聚合示例仍按 key 是否存在来聚合。
本项目输出 label 里的 `0.0` 才表示该 class 在当前映射下为 negative。

当前 super5 的 PTB-XL、PN2021、MIMIC 映射都应输出 0/1。只有更细粒度 scheme 在部分数据集可能输出 -1。

## PTB-XL -> Super5

来源：

```text
/root/autodl-tmp/ptbxl/ptbxl_database.csv: scp_codes
/root/autodl-tmp/ptbxl/scp_statements.csv: diagnostic_class
```

官方/事实状态：

- PTB-XL 的 `scp_statements.csv` 里存在五个 diagnostic superclass：
  `CD/HYP/MI/NORM/STTC`。
- `super5` 是本项目对 PTB-XL official diagnostic superclass 的固定向量化命名，
  不是 PTB-XL metadata 中原生存在的 `super5` 列。
- 官方 example 的聚合逻辑是：遍历 `scp_codes` 的 key，只要该 key 在
  `scp_statements.csv` 中属于 `diagnostic == 1`，就取其 `diagnostic_class`。
- `scp_codes` 的 value 是 likelihood；PTB-XL 文档说明 likelihood 为 0 时代表
  unknown。官方 example 没有按 likelihood value 过滤。
- 因此当前 `confidence_threshold=0.0` 和 `conf >= threshold` 的实现更接近官方
  example 的“key 存在即参与聚合”语义。若改成 `conf > 0`，必须作为新标签版本。

标准规则：

1. 解析 `scp_codes` 字典。
2. 只使用 `scp_statements.csv` 中 `diagnostic == 1.0` 的诊断 SCP。
3. 读取每个 SCP 的 `diagnostic_class`。
4. 映射到 `CD/HYP/MI/NORM/STTC`。
5. 输出 `(5,) float32` multi-hot。

PTB-XL 是 super5 训练标签的最高优先级来源，因为 super5 本身来自 PTB-XL 官方 `diagnostic_class`。
论文中可直接表述为 “PTB-XL official diagnostic superclass labels”。

当前实现细节：

- `ptbxl_scp_to_super5()` 默认使用 `confidence_threshold=0.0` 且条件是 `conf >= threshold`。
- 这意味着只要 SCP key 存在并且在 `scp_statements.csv` 中是 diagnostic code，就会计入 super5；包括 likelihood/confidence 等于 0 的诊断 key。
- 本地核查显示默认规则得到的全量 PTB-XL 计数为：

```text
CD=4898, HYP=2649, MI=5469, NORM=9514, STTC=5235
```

这些计数与 PhysioNet 页面公布的 diagnostic superclass distribution 一致。
把阈值改成 `conf > 0` 会改变这些计数，例如 STTC 会下降，因此不能静默修改。
- 不要在未记录新实验版本的情况下改成 `conf > 0`、`conf >= 1` 或其他阈值。阈值改变会改变 STTC 等类别阳性数，必须重建标签 cache 并重新报告 per-class label distribution。

训练策略差异：

- PTB-XL 中存在没有 diagnostic superclass 的记录。当前 trainer 保留这些记录，
  对 super5 输出 all-zero label。
- 本地核查数量：全量 411 条；fold 1-8 为 334 条，fold 9 为 37 条，fold 10 为 40 条。
- 一些官方/benchmark 流程会过滤掉没有 superclass label 的记录。因此论文或复现实验
  需要写明：本项目当前 EfficientNet1DV2 baseline 保留 all-zero diagnostic-superclass
  样本。如果后续改为过滤，必须作为新训练版本重跑 baseline 和 PN2021 eval。

缓存失效规则：

```text
If PTB-XL mapping rule, confidence threshold, scp_statements.csv, or class order changes:
    delete/regenerate ptbxl_labels.C5.all.npy
    rerun label_schemes.py --sanity
    record new per-class positive counts
```

当前代码注意事项：

- `train_ptbxl.py::get_ptbxl_labels_for_scheme()` 的 cache key 是
  `ptbxl_labels.C5.all.npy`，可以正常复用 `np.save` 写出的标签缓存。
- 后续若要进一步增强可追踪性，可加入 scheme/mapping/confidence 版本，例如
  `ptbxl_labels.super5.official_diagclass_confge0.v1.npy`。

## PN2021 -> Super5

来源：

```text
PN2021 .hea header: #Dx SNOMED code list
PhysioNet/CinC 2021 dx_mapping_scored.csv
PhysioNet/CinC 2021 dx_mapping_unscored.csv
```

官方/事实状态：

- PN2021/CinC2021 官方标签空间是 SNOMED-CT code list。
- 官方提供 scored/unscored diagnosis mapping 和 Challenge weight matrix。
- 官方没有提供 `PN2021 SNOMED -> PTB-XL diagnostic_class/super5` 的标准映射。
- 因此本项目的 `SNOMED_TO_SUPER5` 只能称为 “custom semantic projection”，
  不能写成官方映射。

当前 v3 标准规则：

1. 从 `.hea` 中读取 `#Dx:` SNOMED 列表。
2. 使用 `SNOMED_TO_SUPER5_POSITIVE` 映射明确异常类 `CD/HYP/MI/STTC`。
3. 只把 `NORM_POSITIVE_SNOMEDS` 中的 strict sinus rhythm 作为 NORM 候选。
4. 使用 `NORM_SUPPRESS_SNOMEDS` 压制 rhythm/axis/ectopy/low-voltage/boundary 等
   不应算 normal、但没有直接 super5 阳性类的 code。
5. 输出 `(5,) float32` multi-hot。
6. 强制 NORM exclusivity guard：

```text
if any(CD, HYP, MI, STTC) == 1 or any(code in NORM_SUPPRESS_SNOMEDS):
    NORM = 0
elif sinus rhythm is present:
    NORM = 1
else:
    NORM = 0
```

当前实现：

```text
scripts/triple_labels/label_schemes.py
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
SNOMED_TO_SUPER5_POSITIVE
NORM_POSITIVE_SNOMEDS
NORM_SUPPRESS_SNOMEDS
```

原因：PN2021 经常同时标注 sinus rhythm 和异常诊断；PTB-XL 训练语义中 `NORM` 更接近“未发现异常”。不加 suppress guard 会导致 NORM 的跨中心语义漂移。

历史 v2 规则：

1. 从 `.hea` 中读取 `#Dx:` SNOMED 列表。
2. 使用 `SNOMED_TO_SUPER5` 映射到五类。
3. 输出 `(5,) float32` multi-hot。
4. 强制 NORM exclusivity guard：

```text
if any(CD, HYP, MI, STTC) == 1:
    NORM = 0
```

原因：PN2021 经常同时标注 sinus rhythm 和异常诊断；PTB-XL 训练语义中 `NORM` 更接近“未发现异常”。不加 guard 会导致 NORM 的跨中心语义漂移。

v2 已知限制，v3 已修复：

- v2 `snomed_list_to_super5()` 只在 SNOMED 已映射到 `CD/HYP/MI/STTC` 时压制 `NORM`。
- v2 对 AF/AFL/PAC/PVC/LAD/RAD/low voltage 等未映射到 super5 但明显非正常的 PN2021 诊断，不会压制 `NORM`。
- v3 已把“映射到 super5 阳性类”和“取消 NORM 的异常证据”拆开，引入 `NORM_SUPPRESS_SNOMEDS`。
- v3 已将 `Q wave abnormal`、`early repolarization` 从直接 STTC 阳性改为 NORM-suppress-only。若后续要作为 STTC，应做 broad-mapping ablation。

仍需记录的 PN2021 策略风险：

1. `NORM` 语义仍是投影策略，不是官方定义：
   PN2021 中 `sinus rhythm`、`sinus bradycardia`、`sinus tachycardia`、`sinus arrhythmia`
   等节律描述可以和其他异常诊断共存。v3 已用 `NORM_SUPPRESS_SNOMEDS` 避免
   AF/AFL/PAC/PVC/LAD/RAD/LQRSV/PRWP 等明显非正常证据保留 `NORM=1`。
   但 suppress set 仍是本项目规则，不能写成 PN2021 官方 normal 定义。
2. 边界码归类过强：
   `Q wave abnormal` 可能指向 MI 相关证据，不等价于 PTB-XL 的 STTC；
   `early repolarization` 可能是良性变异，也不应默认当作 STTC 强阳性。
3. PN2021 官方词表覆盖问题：
   当前 `SNOMED_TO_SUPER5_POSITIVE` 中有少数 code 没在已核查的 Challenge 2021 官方
   `dx_mapping_scored.csv` / `dx_mapping_unscored.csv` 中出现，例如
   `22298006`、`401303003`、`233843008`、`26749005`、`164828000`。
   如果保留这些 code，必须注明它们来自其他 SNOMED/数据源兼容需求；否则应从
   PN2021 主映射中移除或放入扩展映射。
4. cache 版本：
   v3 已使用 `PN2021_EVAL_CACHE_VERSION = "v3_super5_normsuppress"`，并在 cache
   中写入 metadata。后续任何映射、suppress set、parser、预处理变化仍必须 bump 版本。
5. header parser：
   v3 `parse_header_snomed()` 使用严格 `^#\s*Dx\s*:` 匹配。后续仍建议记录
   empty/unmapped SNOMED 统计，便于审计。

当前 v3 代码策略：

```python
SNOMED_TO_SUPER5_POSITIVE = {...}  # 只放能明确投到 CD/HYP/MI/STTC 的 abnormal code
NORM_POSITIVE_SNOMEDS = {...}      # 默认只放 strict normal，例如 sinus rhythm
NORM_SUPPRESS_SNOMEDS = {...}      # rhythm/axis/ectopy/voltage/boundary abnormalities

if any(super5 abnormal positive) or any(code in NORM_SUPPRESS_SNOMEDS):
    label[NORM] = 0.0
```

`NORM_SUPPRESS_SNOMEDS` 首批覆盖：

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
pacing rhythm, if not treated as a direct CD positive
```

其中 `Q wave abnormal`、`early repolarization` 当前作为 `NORM_SUPPRESS_SNOMEDS`，
不直接产生 STTC 阳性。若后续改回 STTC，需要作为 broad mapping ablation 并重跑 PN2021。

PN2021 cache v3：

```text
PN2021_EVAL_CACHE_VERSION = "v3_super5_normsuppress"
cache metadata:
  scheme
  class_names
  cache_version
  pn2021_mapping.mapping_version
  pn2021_mapping.mapping_hash
  preprocess_config
```

加载 cache 时必须校验 metadata；如果 metadata 不匹配，应拒绝加载并重建，而不是
静默复用旧 labels。

PN2021 cache 注意事项：

```text
If SNOMED_TO_SUPER5_POSITIVE, NORM_POSITIVE_SNOMEDS, NORM_SUPPRESS_SNOMEDS, parser, or preprocessing changes:
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

## Center Token / Latent Pool 标签规则

ECGTwin center prompt-token 和 Latent-Hull TA-OMAT 都使用同一个 super5 标签空间：

```text
CD, HYP, MI, NORM, STTC
```

token bank 可以为每个中心保留 5 个 token：

```text
<center_CD>, <center_HYP>, <center_MI>, <center_NORM>, <center_STTC>
```

但第一版下游 synthetic augmentation 只把通过数字 ECG gate 和 teacher/classifier gate 的类别纳入主线。
当前建议是 NORM/MI/STTC 先进入主线；HYP/CD 先作为 token 训练、验证和消融保留。

latent convex combination 的标签继承规则：

- 只有所有 anchor latent 共享同一个 primary class 或完全相同 multi-hot label 时，才允许沿用 hard GT label。
- 若未来做跨类别 latent mixing，必须单独定义 union label 或 soft label，并把该实验从主线中分离。
- PN2021 样本若作为目标中心少样本池，必须来自 v3 mapping cache；不能混用 v2 label。

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

论文表述建议：

```text
MIMIC-IV ECG report-derived labels are used only as weak pseudo-labels.
They are not treated as authoritative diagnostic labels for the main super5 task.
```

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

重要限制：

- `scripts/triple_labels/train_ptbxl.py::SynthNPZDataset` 只会接受 `(N,1000,12)`
  或 `(N,12,1000)`，并执行 shape normalization 和 crop。
- 它不会执行 filter、z-score、resample、lead reorder、NaN/Inf 清理或标签修复。
- 因此 synthetic `.npz` 导出脚本必须在写盘前完成：
  1. ECGTwin/MIMIC lead order -> PTB-XL lead order；
  2. 1024 点 -> 1000 点；
  3. 与 PTB-XL classifier cache 一致的预处理尺度；
  4. `labels.shape == (N,5)`；
  5. class order 固定为 `CD/HYP/MI/NORM/STTC`；
  6. `NORM` 不与 `CD/HYP/MI/STTC` 共存。

如果导出 raw mV synthetic signals，必须把该实验标记为 raw-mV ablation，不能和
classifier-ready synthetic augmentation 混为同一个实验版本。

## 代码检查点

每次修改标签逻辑后至少检查：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/label_schemes.py --sanity --mimic_n 2000
```

并重新跑一个小规模训练 smoke test，确认：

- 标签维度和模型输出维度一致。
- 每类阳性数不为 0。
- `NORM` 与异常类共现率符合预期。
- PTB-XL label cache 文件名或目录能反映 mapping version/confidence threshold。
- PN2021 cache version 能反映 `SNOMED_TO_SUPER5_POSITIVE`、`NORM_POSITIVE_SNOMEDS`、`NORM_SUPPRESS_SNOMEDS` 和 parser 版本。
