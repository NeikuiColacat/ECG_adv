# PN2021 多标签与争议标签丢弃影响统计

日期：2026-05-01

统计范围：

```text
/root/autodl-tmp/physionet2021/training/
包含 7 个评测中心
排除 ptb-xl
按 .hea header 的 #Dx SNOMED code 统计
```

中心：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

注意：这是 header-level 统计，未执行 `wfdb.rdrecord` 信号读取过滤；实际 eval cache 可能因为坏文件、缺导联等略少。

## 1. PN2021 是否一个样本可能有多个标签

是。PN2021 一个 ECG header 的 `#Dx:` 可以包含多个 SNOMED-CT code。

7 个中心总样本数：

```text
66416
```

多标签统计：

| 范围 | 样本数 | 多 SNOMED 样本 | 比例 | 映射后多 super5 样本 | 比例 | 当前 zero-super5 样本 | 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 66416 | 31619 | 47.6% | 9007 | 13.6% | 6015 | 9.1% |
| chapman_shaoxing | 10247 | 5106 | 49.8% | 1083 | 10.6% | 1120 | 10.9% |
| cpsc_2018 | 6877 | 476 | 6.9% | 56 | 0.8% | 2131 | 31.0% |
| cpsc_2018_extra | 3453 | 1756 | 50.9% | 855 | 24.8% | 82 | 2.4% |
| georgia | 10344 | 6284 | 60.8% | 2290 | 22.1% | 456 | 4.4% |
| ningbo | 34905 | 17868 | 51.2% | 4715 | 13.5% | 2164 | 6.2% |
| ptb | 516 | 63 | 12.2% | 3 | 0.6% | 35 | 6.8% |
| st_petersburg_incart | 74 | 66 | 89.2% | 5 | 6.8% | 27 | 36.5% |

结论：

```text
PN2021 是明显的 multi-label 数据集。
如果用 PTB-XL super5 去评测，它也应该按 multi-hot 处理，而不是互斥单标签分类。
```

## 2. 当前 super5 阳性数量

当前类顺序：

```text
CD, HYP, MI, NORM, STTC
```

当前映射后阳性计数：

| 中心 | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|
| ALL | 11679 | 8582 | 2110 | 25882 | 22198 |
| chapman_shaoxing | 1226 | 1322 | 40 | 4686 | 3044 |
| cpsc_2018 | 2797 | 0 | 0 | 918 | 1087 |
| cpsc_2018_extra | 381 | 225 | 1515 | 7 | 2213 |
| georgia | 2307 | 2145 | 7 | 2857 | 5107 |
| ningbo | 4936 | 4867 | 165 | 17327 | 10736 |
| ptb | 22 | 13 | 368 | 81 | 0 |
| st_petersburg_incart | 10 | 10 | 15 | 6 | 11 |

## 3. “丢弃争议样本”会损失多少

这里的“争议样本”不是官方定义，而是按当前项目风险拆成几档。

### P1：只丢最核心 STTC 边界码

定义：

```text
Q wave abnormal
early repolarization
```

损失：

```text
drop = 2031 / 66416 = 3.1%
keep = 64385
```

分中心：

| 中心 | 丢弃数 | 比例 |
|---|---:|---:|
| chapman_shaoxing | 257 / 10247 | 2.5% |
| cpsc_2018 | 0 / 6877 | 0.0% |
| cpsc_2018_extra | 1 / 3453 | 0.0% |
| georgia | 604 / 10344 | 5.8% |
| ningbo | 1169 / 34905 | 3.3% |
| ptb | 0 / 516 | 0.0% |
| st_petersburg_incart | 0 / 74 | 0.0% |

### P2：P1 + pacing 相关边界码

定义：

```text
Q wave abnormal
early repolarization
pacing rhythm
atrial pacing pattern
ventricular pacing pattern
```

损失：

```text
drop = 3293 / 66416 = 5.0%
keep = 63123
```

### P3：P2 + HYP 边界码

定义：

```text
P2
left ventricular high voltage
right atrial high voltage
atrial hypertrophy/enlargement
left/right atrial enlargement variants
```

损失：

```text
drop = 9414 / 66416 = 14.2%
keep = 57002
```

这一档主要会影响 HYP，因为 high voltage / atrial enlargement 是否等价于 PTB-XL HYP 有边界风险。

### P4：丢掉任何 NORM-suppress 候选异常码

定义：

不直接映射到 super5 阳性，但说明这条 ECG 不应该算纯 NORM 的标签，例如：

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
rotation abnormalities
```

损失：

```text
drop = 19715 / 66416 = 29.7%
keep = 46701
```

### P5：P1 + NORM-suppress 候选异常码

定义：

```text
核心边界码 QAb/ERe
加所有 NORM-suppress 候选异常码
```

损失：

```text
drop = 21017 / 66416 = 31.6%
keep = 45399
```

### P6：严格丢弃所有争议项

定义：

```text
P1 core boundary
P2 pacing boundary
P3 HYP boundary
P4 NORM-suppress candidate
```

损失：

```text
drop = 25264 / 66416 = 38.0%
keep = 41152
```

分中心：

| 中心 | 丢弃数 | 比例 |
|---|---:|---:|
| chapman_shaoxing | 4166 / 10247 | 40.7% |
| cpsc_2018 | 1833 / 6877 | 26.7% |
| cpsc_2018_extra | 817 / 3453 | 23.7% |
| georgia | 4093 / 10344 | 39.6% |
| ningbo | 14313 / 34905 | 41.0% |
| ptb | 23 / 516 | 4.5% |
| st_petersburg_incart | 19 / 74 | 25.7% |

## 4. 当前最实际的问题：false NORM risk

当前只标成 NORM，但同时带有 NORM-suppress 候选异常码的样本：

```text
2046 / 66416 = 3.1% of all
2046 / 25882 = 7.9% of current NORM positives
```

这批样本是当前策略最应该修的地方：它们不是要直接丢弃，而是应该把 `NORM=1` 改成 `NORM=0`，避免把 rhythm/axis/ectopy 异常样本误当正常。

分中心：

| 中心 | false NORM risk | 比例 |
|---|---:|---:|
| chapman_shaoxing | 429 / 10247 | 4.2% |
| cpsc_2018 | 0 / 6877 | 0.0% |
| cpsc_2018_extra | 5 / 3453 | 0.1% |
| georgia | 222 / 10344 | 2.1% |
| ningbo | 1390 / 34905 | 4.0% |
| ptb | 0 / 516 | 0.0% |
| st_petersburg_incart | 0 / 74 | 0.0% |

最常见 false NORM risk code：

| SNOMED | 数量 | 含义 |
|---:|---:|---|
| 39732003 | 522 | LAD |
| 284470004 | 470 | PAC |
| 251146004 | 453 | low QRS voltage |
| 47665007 | 296 | RAD |
| 427172004 | 190 | PVC |
| 365413008 | 83 | PRWP |
| 17338001 | 67 | VPB |
| 251199005 | 67 | counterclockwise rotation |
| 713422000 | 39 | atrial tachycardia |
| 164890007 | 14 | AFL |
| 164889003 | 5 | AF |

## 5. 最常见争议码

| SNOMED | 数量 | 含义 |
|---:|---:|---|
| 164890007 | 8301 | AFL |
| 55827005 | 5401 | left ventricular high voltage |
| 164889003 | 3741 | AF |
| 284470004 | 2643 | PAC |
| 39732003 | 2485 | LAD |
| 164917005 | 1528 | Q wave abnormal |
| 251146004 | 1417 | low QRS voltage |
| 427172004 | 1279 | PVC |
| 10370003 | 1185 | pacing rhythm |
| 47665007 | 937 | RAD |
| 67741000119109 | 872 | left atrial enlargement alt |
| 17338001 | 659 | VPB |
| 365413008 | 638 | PRWP |
| 428417006 | 506 | early repolarization |

## 6. 建议

不建议直接把所有争议样本都丢掉作为主评测。

原因：

```text
P6 严格丢弃会损失 38.0% 样本，且对 ningbo/chapman/georgia 影响特别大。
这会改变 PN2021 的中心分布和疾病分布，让外部评测变得不代表真实跨中心数据。
```

更推荐：

1. 主评测保留样本，但修正标签映射。
2. 加 `NORM_SUPPRESS_SNOMEDS`，把 2046 条 false NORM risk 样本从 NORM 中移除。
3. `Q wave abnormal`、`early repolarization` 默认改为 suppress-only，不直接算 STTC。
4. P1 或 P2 可作为 sensitivity analysis：报告“去除边界码后结果是否稳定”。
5. 不建议用 P6 作为主结果，只能作为极端保守分析。

推荐主线：

```text
Main eval:
  PN2021 v3 super5 mapping with NORM_SUPPRESS_SNOMEDS

Sensitivity eval:
  exclude P1 core boundary samples
  optional exclude P2 boundary+pacing samples

Do not use:
  strict P6 all disputed removal as main evaluation
```
