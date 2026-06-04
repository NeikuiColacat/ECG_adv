# Super5 Label Mapping Deep Research 2026-05-22

本报告审核两类映射：

1. PN2021 / PhysioNet Challenge 2021 SNOMED-CT labels -> PTB-XL Super5。
2. ECGFounder official 150-class head -> PTB-XL Super5。

结论先行：两者都没有官方一一对应 crosswalk。论文中应避免写成 official Super5 label，改写为 **task-specific semantic projection** 或 **label-space compatibility baseline**，并提供 strict/main 与 sensitive/ablation 两套规则。

## 主要来源

- PhysioNet/CinC Challenge 2021 data: <https://physionet.org/content/challenge-2021/1.0.3/>
- PhysioNet Challenge 2021 evaluation repo: <https://github.com/physionetchallenges/evaluation-2021>
- Official scored labels: <https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv>
- Official unscored labels: <https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_unscored.csv>
- PN2021 paper: <https://moody-challenge.physionet.org/2021/papers/2021ChallengePaperCinC.pdf>
- PN2021 FAQ: <https://moody-challenge.physionet.org/2021/faq/>
- PTB-XL PhysioNet: <https://physionet.org/content/ptb-xl/1.0.3/>
- PTB-XL `scp_statements.csv`: <https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv>
- PTB-XL paper: <https://www.nature.com/articles/s41597-020-0495-6>
- ECGFounder repo: <https://github.com/PKUDigitalHealth/ECGFounder>
- ECGFounder official tasks: <https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/tasks.txt>
- ECGFounder paper/PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/>
- SNOMED ischemia editorial guide: <https://docs.snomed.org/education/snomed-ct-education-editorial-guide/readme/authoring/domain-specific-modeling/clinical-finding-and-disorder/clinical-finding-and-disorder-modeling/specific-clinical-finding-and-disorder-modeling/ischemia>
- AHA/ACCF/HRS ECG interpretation: conduction, hypertrophy, ischemia/infarction:
  <https://pubmed.ncbi.nlm.nih.gov/19281930/>,
  <https://pubmed.ncbi.nlm.nih.gov/19228820/>,
  <https://pubmed.ncbi.nlm.nih.gov/19281933/>
- Fourth Universal Definition of MI: <https://pubmed.ncbi.nlm.nih.gov/30154043/>

## PN2021 -> PTB-XL Super5

PTB-XL Super5 的来源是 `scp_statements.csv` 的 `diagnostic_class`：

```text
CD, HYP, MI, NORM, STTC
```

PN2021 官方只提供 SNOMED-CT 诊断码、30 个 scored labels 和 challenge scoring 权重，并没有把 SNOMED-CT 标签映射到 PTB-XL Super5 的官方表。因此当前项目的 v3 映射必须被描述为：

```text
project-defined semantic projection from PN2021 SNOMED-CT labels to PTB-XL diagnostic superclasses
```

### 推荐 main/strict 规则

| Super5 | strict/main |
|---|---|
| `NORM` | 只把 explicit sinus rhythm 当作 normal proxy，并要求同条 ECG 没有任何 Super5 阳性或 suppress-only abnormal code。 |
| `CD` | BBB/LBBB/RBBB、complete/incomplete BBB、AV block、prolonged PR、fascicular block、NSIVCB、WPW/pre-excitation。 |
| `HYP` | explicit LVH/RVH/ventricular hypertrophy、LAE/RAE/atrial overload-enlargement-hypertrophy。 |
| `MI` | explicit myocardial infarction、old MI、acute MI、anterior/inferior/lateral MI 等 infarction code。 |
| `STTC` | T-wave abnormal/inversion、ST depression/elevation/ST interval abnormal/ST changes、prolonged QT、nonspecific ST-T abnormality、ECG myocardial/territorial ischemia。 |

### 推荐 sensitive/ablation 规则

| 风险项 | main 建议 | sensitive/ablation |
|---|---|---|
| pacing rhythm / atrial pacing / ventricular pacing | 不作为 CD 阳性，只 suppress NORM 或 exclude | 放入 CD-sensitive |
| left ventricular high voltage / right atrial high voltage | 不作为 HYP 阳性，只 suppress NORM | 放入 HYP-sensitive |
| Q wave abnormal | suppress-only | 放入 MI-sensitive |
| early repolarization | suppress-only | 只在 STTC-sensitive 中测试，默认不建议 |
| sinus brady/tach/arrhythmia | suppress NORM | 宽松 NORM ablation，但不作为主结果 |
| ST elevation | STTC，不映射 MI | 保持 STTC；不要仅凭 ST elevation 宣称 MI |

### 当前 v3 的保留与调整

当前 v3 已经比较保守，值得保留：

- `NORM` 有 abnormal/suppress guard。
- `Q wave abnormal` 和 `early repolarization` 默认 suppress-only。
- `sinus bradycardia/tachycardia/arrhythmia` 不当作 NORM。
- ischemia 放入 `STTC` 比放入 `MI` 更稳。

建议调整：

- `pacing rhythm / atrial pacing / ventricular pacing -> CD` 从 main 移到 CD-sensitive。
- `LV high voltage / RA high voltage -> HYP` 从 main 移到 HYP-sensitive。
- 论文中明确说明 PN2021 labels 本身跨数据源、跨 ontology，存在异质性和错误/不完整标签，PN2021-Super5 是外部评估标签投影，不是临床 gold-standard Super5。

## ECGFounder 150-class -> PTB-XL Super5

ECGFounder official head 输出 150 个报告式标签，包含诊断、节律、形态、设备状态、轴偏移和报告比较语句。它不是 PTB-XL Super5 head。

当前项目的 `keyword + max pooling` 逻辑应只作为弱基线：

```text
ECGFounder 150-label out-of-box scores heuristically pooled into PTB-XL Super5
```

不能写成：

```text
ECGFounder official Super5 zero-shot
```

### keyword + max pooling 风险

- 150-label ontology 与 PTB-XL diagnostic_class 不一致。
- `max` 对大池类别有极值偏置，CD/STTC 池越大越容易被抬高。
- `SINUS RHYTHM` 不等价于 PTB-XL `NORM`。
- STTC 池容易混入 `NOW EVIDENT`、`NO LONGER EVIDENT`、`LESS/MORE` 这类报告比较语句。
- `EARLY REPOLARIZATION`、`ACUTE PERICARDITIS`、`QT HAS SHORTENED` 不应直接等价 PTB-XL STTC。
- `WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY` 同时落入 CD/STTC，会污染解释。
- 150-label 概率没有按 Super5 校准，AUPRC 对类先验和池大小敏感。

### 推荐 ECGFounder 对比层级

| 层级 | 用途 | 推荐程度 |
|---|---|---|
| Frozen ECGFounder backbone + PTB-XL Super5 linear head | 最公平主表。fold 1-8 训练 head，fold 9 调参/校准，fold 10 + PN2021 外评。 | 主结果 |
| ECGFounder K-shot Super5 head fine-tune | 强 target baseline，回答少量目标中心数据下 foundation encoder 能做到什么。 | 主/附表 |
| Logistic bridge: 150 probs -> 5 Super5 | 保留官方 150 head，但用 PTB-XL 学习 5 类映射。 | 附表 |
| strict zero-shot manual pooling | 不训练，只做开箱弱基线。 | 附录 |
| keyword + max pooling | 只作历史弱基线，不作为严肃主对比。 | 不推荐主表 |

### 推荐 strict zero-shot pooling

- `NORM`: 只允许 `NORMAL ECG` / `OTHERWISE NORMAL ECG`，不要把 `SINUS RHYTHM` 单独当 NORM。
- `CD`: BBB、AV block、fascicular block、IVCD、WPW；device/pacemaker-only 移出。
- `HYP`: explicit hypertrophy/enlargement；high voltage 做 sensitive。
- `MI`: explicit infarct/acute MI/STEMI；injury pattern 做 sensitive。
- `STTC`: direct ST/T/QT abnormality；去掉 temporal comparison、early repolarization、pericarditis、short QT。

## 论文写法建议

建议主文写：

```text
Because PhysioNet/CinC 2021 and PTB-XL use different diagnostic taxonomies,
we define a conservative, versioned semantic projection from PN2021 SNOMED-CT
diagnoses to PTB-XL Super5 classes. Ambiguous rhythm, device, voltage-only,
and boundary morphology codes are not treated as positive Super5 labels in the
main analysis; they either suppress NORM or are reserved for sensitivity
analyses.
```

ECGFounder 对比写：

```text
For ECGFounder, the official 150-label head is not a PTB-XL Super5 classifier.
Therefore, our main comparison trains a PTB-XL Super5 head on frozen ECGFounder
features. Heuristic 150-label pooling is reported only as an auxiliary
label-space compatibility baseline.
```
