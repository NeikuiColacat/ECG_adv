# Super5 映射策略审核记录

日期：2026-05-21
范围：PN2021 SNOMED -> PTB-XL Super5；ECGFounder 150-class head -> PTB-XL Super5

## 一句话结论

PTB-XL 的 Super5 标签本身有官方背书；PN2021 -> Super5 没有官方 crosswalk，我们当前 `v3_super5_normsuppress` 是可用但必须明确声明的项目自定义语义投影；ECGFounder 150 分类头 -> Super5 的 zero-shot 结果风险更高，只能作为启发式弱基线，不应写成官方 Super5 分类头。

## 已下载资料

本次审核资料保存到：

```text
/root/autodl-tmp/mapping_audit_sources_20260521/
```

包含：

```text
pn2021_dx_mapping_scored.csv
pn2021_dx_mapping_unscored.csv
ptbxl_scp_statements.csv
ecgfounder_tasks.txt
ecgfounder_readme.md
ecgfounder_arxiv_2410.04133.pdf
physionet_challenge_2021_paper.pdf
```

主要来源：

- PhysioNet/CinC Challenge 2021 evaluation repo: https://github.com/physionetchallenges/evaluation-2021
- `dx_mapping_scored.csv`: https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv
- `dx_mapping_unscored.csv`: https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_unscored.csv
- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- PTB-XL `scp_statements.csv`: https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
- PTB-XL paper: https://www.nature.com/articles/s41597-020-0495-6
- ECGFounder paper: https://arxiv.org/abs/2410.04133
- ECGFounder official repo: https://github.com/PKUDigitalHealth/ECGFounder
- ECGFounder official tasks: https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/tasks.txt

## 代码依据

- PTB-XL / PN2021 label source: `scripts/triple_labels/label_schemes.py`
- 当前 PN2021 映射版本：

```text
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
PN2021_EVAL_CACHE_VERSION     = v3_super5_normsuppress
mapping_hash                  = 544ed42dee6d
```

- ECGFounder zero-shot Super5 pooling: `scripts/paper/eval_ecgfounder_super5_zero_shot_20260517.py`

## PN2021 -> Super5 审核

### 强背书部分

PTB-XL Super5 本身是官方定义，来自 `scp_statements.csv` 的 `diagnostic_class`：

```text
CD, HYP, MI, NORM, STTC
```

PN2021 中以下映射方向医学上比较稳：

- `LBBB/RBBB/AV block/fascicular block/IVCD/WPW` -> `CD`
- 明确 `myocardial infarction/old MI/acute MI/anterior MI/inferior MI` -> `MI`
- `T wave abnormal/inversion`、`prolonged QT`、`ST changes/depression/elevation`、`ischemia` -> `STTC`
- `LVH/RVH/LAE/RAE` -> `HYP`

### 必须声明为项目自定义的部分

PN2021 官方只给 SNOMED-CT 诊断和 Challenge scored/unscored label；官方没有 `PN2021 SNOMED -> PTB-XL Super5` crosswalk。因此当前映射应该表述为：

```text
project-defined semantic projection from PN2021 SNOMED-CT labels to PTB-XL diagnostic superclasses
```

当前 v3 策略是：

```text
1. 明确异常 SNOMED 才映射为 CD/HYP/MI/STTC 阳性。
2. NORM 只接受 explicit sinus rhythm。
3. 如果存在 abnormal positive 或 suppress-only code，则 NORM=0。
4. rhythm/axis/ectopy/low-voltage/boundary codes suppress NORM，但不强行变成 Super5 阳性。
```

### PN2021 当前标签分布

| center | N | CD | HYP | MI | NORM | STTC | all-zero | multi-label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing | 10247 | 1226 | 1322 | 40 | 1371 | 2951 | 4499 / 43.9% | 1057 / 10.3% |
| cpsc_2018 | 6877 | 2797 | 0 | 0 | 918 | 1087 | 2131 / 31.0% | 56 / 0.8% |
| cpsc_2018_extra | 3453 | 381 | 225 | 1515 | 0 | 2212 | 89 / 2.6% | 855 / 24.8% |
| georgia | 10344 | 2307 | 2145 | 7 | 1752 | 5011 | 1625 / 15.7% | 2261 / 21.9% |
| ningbo | 34905 | 4936 | 4867 | 165 | 4606 | 10407 | 15109 / 43.3% | 4625 / 13.3% |
| ptb | 516 | 22 | 13 | 368 | 80 | 0 | 36 / 7.0% | 3 / 0.6% |
| st_petersburg_incart | 74 | 10 | 10 | 15 | 0 | 11 | 33 / 44.6% | 5 / 6.8% |
| total | 66416 | 11679 | 8582 | 2110 | 8727 | 21679 | 23522 / 35.4% | 8862 / 13.3% |

重要解释：

- `all-zero` 不是“健康正常”，大多是 rhythm/axis/ectopy/suppress-only 或 uncovered code 导致没有 Super5 阳性。
- `cpsc_2018` 当前没有 HYP/MI 阳性，`cpsc_2018_extra` 没有 NORM，`ptb` 没有 STTC；per-center macro 的有效类别数会变化。

### 高风险映射点

| code | diagnosis | current policy | risk |
|---:|---|---|---|
| 164917005 | Q wave abnormal | suppress-only | Q wave 可提示既往 MI，审稿人可能要求 `QAb -> MI` 或消融。 |
| 10370003 | pacing rhythm | CD positive | PTB-XL CD 子类没有明确 pacemaker 类；建议做 `pacing->CD` vs `pacing suppress-only`。 |
| 251266004 / 251268003 | ventricular/atrial pacing pattern | CD positive | 同上。 |
| 55827005 | left ventricular high voltage | HYP positive | high voltage 不等于结构性 hypertrophy，但样本数大，会影响 HYP。 |
| 67751000119106 | right atrial high voltage | HYP positive | high voltage 类证据弱于明确 hypertrophy/enlargement。 |
| 426177001 / 427084000 / 427393009 | sinus brady/tachy/arrhythmia | suppress NORM | 可能是生理变异，但不是 PTB-XL normal ECG 的官方等价项。 |
| 164931005 / 429622005 / 55930002 | ST elevation/depression/changes | STTC positive | 方向合理，但 ST 改变可能来自 MI、LVH、药物、电解质等。 |
| 164861001 / 425623009 / 425419005 / 426434006 | ischemia | STTC positive | PTB-XL 有 ischemic STTC 子类，但 SNOMED ischemia 不是单纯 ECG 形态。 |

### 代码卫生点

以下 alias 在当前映射表中，但不在 PN2021 官方 scored/unscored 表，也未出现在当前本地 PN2021 header 中：

```text
22298006, 401303003, 233843008, 26749005, 164828000
```

它们不影响当前实验结果；论文里不要把它们当作 PN2021 映射依据。下次 bump cache 时可移除或单列为 legacy aliases。

## ECGFounder 150-class -> Super5 审核

ECGFounder 官方模型输出是 150 个 HEEDB/报告式诊断标签，不是 PTB-XL Super5 头。当前代码用 keyword 建池，再对池内 sigmoid 概率取 `max`：

```text
Super5 score = max(sigmoid(ECGFounder logits over keyword-matched tasks))
```

当前 pool size：

| Super5 | ECGFounder task pool size |
|---|---:|
| CD | 37 |
| HYP | 7 |
| MI | 11 |
| NORM | 4 |
| STTC | 31 |

风险：

- 不是官方映射，是启发式 keyword/max pooling。
- `max` 对大池类别有天然偏置，CD/STTC 池明显大于 HYP/NORM。
- `NORM` 里包含 `SINUS RHYTHM`，但 sinus rhythm 不等价于 PTB-XL normal ECG。
- STTC 池包含 `NOW EVIDENT`、`NO LONGER`、`LESS/MORE` 这类报告比较语句。
- STTC 池还包含 `EARLY REPOLARIZATION`、`ACUTE PERICARDITIS`、`QT HAS SHORTENED` 等边界或非 PTB-XL STTC 严格等价项。
- MI 池没有系统性纳入所有 injury pattern，而 PTB-XL MI 中包含部分 injury 相关 SCP code。
- 任务 `WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY` 同时进入 CD 和 STTC。

因此 ECGFounder zero-shot 结果建议表述为：

```text
ECGFounder 150-label out-of-box scores heuristically pooled into PTB-XL Super5 by keyword OR/max.
This is a non-calibrated label-space compatibility baseline, not an official Super5 head.
```

更公平的 ECGFounder 对比应为：

```text
ECGFounder backbone + PTB-XL fold 1-8 Super5 linear head
-> fold 9 calibration/model selection
-> fold 10 and PN2021 external-center evaluation
```

当前 ECGFounder zero-shot keyword pooling 可保留在附录或辅助 sanity check，不建议作为主公平对比结论。

## 建议消融实验

PN2021 映射敏感性：

1. `QAb suppress-only` vs `QAb -> MI` vs `QAb -> STTC`
2. `pacing -> CD` vs `pacing suppress-only`
3. `high voltage -> HYP` vs `high voltage suppress-only`
4. strict NORM vs 允许 `sinus bradycardia/sinus arrhythmia` 作为 NORM-compatible
5. all labels vs official scored-only vs scored+unscored Super5
6. 保留 all-zero 样本 vs 排除 outside-super5-only 样本

ECGFounder 映射敏感性：

1. 当前 broad keyword/max pooling
2. strict manual pool：去掉 temporal-comparison、device-only、early repol、pericarditis、short QT
3. NORM suppression version：若任何 abnormal pool score 高于阈值，则抑制 NORM
4. ECGFounder frozen feature + PTB-XL Super5 linear head

## 论文写法建议

推荐写法：

```text
Because PN2021 does not provide PTB-XL diagnostic superclass labels, we define a conservative SNOMED-to-Super5 semantic projection for external-center evaluation. Only unambiguous SNOMED codes are mapped to CD/HYP/MI/STTC. Rhythm-only, ectopy, axis-deviation, low-voltage, and boundary diagnoses suppress NORM but are not forced into positive Super5 labels.
```

不推荐写法：

```text
PN2021 official Super5 labels
ECGFounder official Super5 zero-shot classifier
```

更准确的说法：

```text
PN2021 project-defined Super5 semantic projection
ECGFounder heuristic 150-label-to-Super5 pooling baseline
```

## 最终判断

当前 PN2021 v3 映射不是“严重错误”，但它不是官方标签，需要靠透明说明和 sensitivity ablation 防审稿质疑。ECGFounder zero-shot 映射风险比 PN2021 映射更高，最好只作为辅助基线；主论文公平对比应优先使用 ECGFounder backbone 上重新训练的 PTB-XL Super5 head。
