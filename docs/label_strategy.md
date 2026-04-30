---
title: 跨中心 EfficientNetV2 分类头标签选择调研
audience: ECG-adv-Gen 项目内部
date: 2026-04-19
status: draft / 待 pipeline 落地后补实测
---

> 当前状态：历史标签调研文档。本文保留 2026-04-19 对 Tier-M/Tier-X/AdvDiff/AugMix 路线的论证过程，但不再作为当前 super5 主线的事实来源。
>
> 当前主线标签事实以 `docs/pipelines/super5_label_mapping_pipeline.md` 和 `scripts/triple_labels/label_schemes.py` 为准；当前训练/评测入口是 `scripts/triple_labels/train_ptbxl.py` 和 `scripts/triple_labels/eval_crosscenter.py`。

# 跨中心 EfficientNetV2 分类头标签选择调研

## 0. 结论前置（TL;DR）

**若严格要求"ECGTwin 友好 ∩ PTBXL ∩ PN2021 ∩ MIMIC"四方全部官方支持，最终推荐为 7 类分类头**：

> **严格交集（7 类）**：**NSR / SB / STach / AF / IAVB / LBBB / RBBB**
>
> 即：窦性心律 / 窦性心动过缓 / 窦性心动过速 / 房颤 / 一度房室传导阻滞 / 左束支传导阻滞 / 右束支传导阻滞

这 7 类同时满足 4 道筛子：
1. **ECGTwin 官方 demo** 已验证生成质量（`model/ECGTwin/generation_result_by_disease/`）
2. **PTBXL** 有原生 SCP 编码（`NORM/SR, SBRAD, STACH, AFIB, 1AVB, CLBBB, CRBBB`），**无 -1 mask**
3. **PN2021 官方 scored 26 类**之内（最高质量标签，可直接对标公开 benchmark）
4. **MIMIC** 有稳定的 regex 关键词抽取规则

被淘汰的 5 类（LVH / MI / NSTEMI / STEMI / Pericarditis）详见 §7.5 —— 这些要么跨库没有可信标签，要么样本量过小。

备用方案：若可接受标签质量略降（PN2021 用非 scored SNOMED），可扩为 **9 类**（+ LVH, + MI merged）；见 §7.5 方案 B。

> ⚠️ **2026-04-19 实测核查后的精化**（见 §7.6）：**严格 5 中心跨中心可评的只有 3 类 (IAVB/LBBB/RBBB)**。
> SB 在 cpsc_2018 = 0、STach 在 cpsc_2018 = 0、AF 在 ningbo = 0 (编码错位)、NSR 在 cpsc_2018_extra = 4 条。
> 实用推荐：**训练用 Tier-M 6 类**（7 类去掉 SB，并修 AFL→AF 的 mapping），**AdvDiff 生成用 Tier-X 7 类**。

---

## 1. 背景与研究线

- 研究目标：提升自研 EfficientNetV2 ECG 模型的**跨中心泛化**
- 当前路线：ECGTwin（latent 生成） + AdvDiff（梯度引导的对抗样本） + 已实现的 ECG 时域增强算子，
  合入 **AugMix 多链路数据增强**框架，用生成样本微调 EfficientNetV2
- 数据资源：PTBXL（本地 `datasets/PTBXL/`）+ PhysioNet 2021（`/root/autodl-tmp/physionet2021/`，7 中心）+ MIMIC-IV-ECG
- 待定问题：最终分类头应该用哪些标签 / 多少类 / 什么标签体系？

选择标签体系约束来自三方：
1. **三数据集可对齐**（否则无法做跨中心评测）
2. **ECGTwin 能可控生成**（否则 AdvDiff 引导的梯度就是"对着不稳定生成器找对抗方向"，噪声大到没意义）
3. **临床/ECG 形态学上意义明确**（否则跨中心性能提升无法解释）

---

## 2. 三数据集标签体系现状

### 2.1 PTBXL（`adversarial/label_mapping.py` + `scripts/crosscenter_v2/label_alignment_v2.py`）

两套标签并存于仓库：
- **历史 77-class**（`adversarial/label_mapping.py:53-143`，`SCP_TO_EFFICIENTNET`）：
  把 PTBXL SCP codes 映射到 77 个**形态学 pattern**（Q 波下壁、ST 段前壁抬高…），
  来自 `model/DeepECG`，是 JIT 冻结的 legacy victim model 的分类头。
  不是语义诊断类，是"ECG 视觉 pattern"。**当前仅用于 AdvDiff victim，不适合作跨中心训练头**。
- **26-class SNOMED**（`scripts/crosscenter_v2/label_alignment_v2.py:79-152`，`PTBXL_SCP_TO_CLASSES` + `ptbxl_scp_to_26`）：
  via PTBXL `scp_statements.csv` → SNOMED 映射，presence-based（置信度≥0），
  其中 **Brady / PRWP / RAD** 三个类在 PTBXL 的 SCP 体系里没有对应编码，用 -1 mask 掉不参与 loss。
  **这是 v2 基线训练实际用的头**（`scripts/crosscenter_v2/train_ptbxl_v2.py:37-79`，`MaskedFocalLoss` 忽略 -1）。

实测跨中心 gap（`docs/training/gap_report.md`）：
- Tier-1（5 类）PTBXL test AUROC **0.9685**，PN2021 均值 **0.9283**（+4.02pp gap）
- Tier-2（15 类）PTBXL test AUROC **0.9238**，PN2021 均值 **0.8742**（+4.96pp gap）

### 2.2 PhysioNet 2021（`scripts/crosscenter_v2/eval_crosscenter_v2.py:33-73`）

- 来源：每份 wfdb record 的 `.hea` header 包含 `# Dx:` 行，逗号分隔多个 SNOMED CT 编码
- 使用：`label_alignment_v2.py:22-49` 的 **SCORED_26** 就是 PN2021 官方评分用的 27 类里去掉 Bundle Branch Block（非特异）/ 合并同义项后剩下的 26 类
- 覆盖：**26/26 全覆盖**，且是临床专家编码（最可靠的标签源）

26 类清单（按 SNOMED name）：
```
IAVB, AF, AFL, BBB, Brady, LBBB, RBBB, ILBBB, IRBBB,
LAD, LAnFB, LQRSV, NSIVCB, PR, PRWP, PAC, PVC, LPR,
LQT, QAb, RAD, SA, SB, NSR, STach, TAb
```

### 2.3 MIMIC-IV-ECG（`scripts/crosscenter_v2/eval_mimic_zeroshot.py:33-121`）

- 标签来源：`machine_measurements.csv` 有 `report_0 ... report_17` 共 18 列机器报告文本
- 使用：`label_alignment_v2.py:199-276` 的 **MIMIC_KEYWORD_PATTERNS** —— 每类一个 regex 关键字列表
- 覆盖：**26/26 名义全覆盖**，但是**机器报告文本 + regex 匹配**，噪声显著高于 PN2021 的临床 SNOMED
- 典型：AF → `r'atrial fibrillation'`，LBBB → `r'left bundle branch block'`，
  但遇到 "possible AF"、"cannot rule out"、"borderline" 这类模糊表述时 regex 召回偏高，准确率下降

### 2.4 已统一的跨中心头（v2 active）

结论：仓库里现有的 **26-class SNOMED** 就是目前唯一三数据集对齐的头：
- 训练：`train_ptbxl_v2.py`（PTBXL 26-class, confidence_threshold=0）
- 评测：`eval_crosscenter_v2.py`（PN2021 SNOMED 直读）、`eval_mimic_zeroshot.py`（MIMIC regex）
- Loss：`MaskedFocalLoss`，忽略 PTBXL 的 3 个 -1 类

**强烈建议以此为基础**，不要再造一套新标签体系。

---

## 3. ECGTwin 的生成可控范围

调研自 `model/ECGTwin/` 源码 + `util/ecgtwin_utils.py:179-293`。

### 3.1 条件字段

```python
ref_label = {
    "text": "sinus rhythm|normal ecg",  # 自由文本, "|" 分隔多诊断
    "hr":   72.0,                         # 心率
    "age":  65,                           # 年龄
    "sex":  "M",                          # 'M'/'F'
    "text_embed": torch.Tensor,           # 可选: 预计算 (num_diag, 768) nomic embed
}
```

- Text encoder：**nomic-ai/nomic-embed-text-v1.5**（768 维），**不是 SNOMED / ICD 的封闭词表**
- Prompt 模板：`"Most importantly, the 1st diagnosis is {d1}. As a supplementary condition, the 2nd diagnosis is {d2}."`
- hr/age 有 MIMIC 统计量归一化（hr mean=77.95 / std=20.37；age mean=64.25 / std=17.13）

### 3.2 训练数据

- MIMIC-IV-ECG 的**自由文本诊断报告**（非 SNOMED 标签）
- 通过 nomic-embed 得到 768 维向量，作为 cross-attention key/value
- 原文件：`model/ECGTwin/data/paired_Mimic_vae_multi_nomic.pt`

### 3.3 实际验证过的生成类别

`model/ECGTwin/generation_result_by_disease/` 下作者提供了 **12 个 demo 类别**的生成样例：

```
normal_ecg, sinus_bradycardia, sinus_tachycardia,
atrial_fibrillation, atrioventricular_block,
left_bundle_branch_block, right_bundle_branch_block,
left_ventricular_hypertrophy,
myocardial_infarction, nstemi, st_elevation_mi,
acute_pericarditis
```

这**不是封闭词表**（自由文本生成理论上支持任意诊断），但 demo 里给出的 12 类 **可以认为是作者自己验证过生成质量的 subset**。
对于 AdvDiff 这种对生成稳定性敏感的下游任务，**优先锁在这 12 类范围内**更保险。

---

## 4. 三方约束的交集分析

把 ECGTwin 的 12 类 demo 与 PN2021 26 类求交集，逐项判定 PTBXL 覆盖情况：

| PN2021 26-class | ECGTwin demo 对齐 | PTBXL 可学? | 三数据集交集? | 临床语义 |
|---|---|---|---|---|
| NSR          | normal_ecg             | ✅ | ✅ | 窦性心律 |
| SB           | sinus_bradycardia      | ✅ | ✅ | 窦缓 |
| STach        | sinus_tachycardia      | ✅ | ✅ | 窦速 |
| AF           | atrial_fibrillation    | ✅ | ✅ | 房颤 |
| IAVB         | atrioventricular_block | ✅ | ✅ | 一度 AVB |
| LBBB         | left_bundle_branch_block | ✅ | ✅ | 左束支传导阻滞 |
| RBBB         | right_bundle_branch_block | ✅ | ✅ | 右束支传导阻滞 |
| PVC          | （demo 未单列，但 MIMIC 报告中高频） | ✅ | ✅ | 室早 |
| AFL          | —（ECGTwin 未 demo） | ✅ | ✅ | 房扑 |
| (others 17 类) | —                  | 部分 | 部分 | — |

不在 26 类但在 ECGTwin demo 里的：
- **LVH**：PTBXL 有 LVH SCP（LVH 编码存在于 PTBXL），但 26 类把 LVH 合入了其他形态学类别之外，目前被 -1 / 忽略
- **MI / NSTEMI / STEMI**：26 类里没有通用 MI 类，只有 QAb、TAb 等形态学衍生指标；PTBXL 有丰富的 MI SCP code（IMI / AMI / LMI / PMI）
- **Acute pericarditis**：三个数据集都罕见，不建议

---

## 5. 标签选择方案（三层）

### 第 1 层（分类头 / 监督信号）：26 类 SNOMED 全量

- 继续用 `label_alignment_v2.py` 的 26 维 multi-label 头
- 训练时 `MaskedFocalLoss` 自动忽略 PTBXL 的 Brady/PRWP/RAD 三个 -1 类
- 跨中心评测指标全覆盖三数据集

### 第 2 层（主评指标 / 论文 headline）：Tier-1 / Tier-2

沿用 `label_alignment_v2.py:60-68` 的分层：
- **Tier-1 (5 类)**：AF, LBBB, RBBB, IAVB, NSR
  - 所有中心 ≥5 正样本，**最稳**，写论文主表优先
- **Tier-2 (15 类)**：Tier-1 + AFL, BBB, IRBBB, LAD, LAnFB, LQT, NSIVCB, PVC, PR, TAb
  - 覆盖更广，多数中心样本量充足，适合做 ablation

### 第 3 层（AdvDiff 目标生成子集）：**8 类 ECGTwin-friendly**

这是本调研的**新增建议**，用于 AdvDiff 对抗生成的目标类别（不是分类头，是生成目标）：

| 代号 | SNOMED | ECGTwin prompt 建议 | 在三数据集中的样本密度 |
|---|---|---|---|
| NSR   | 426783006 | `normal sinus rhythm` | 高（基线） |
| SB    | 426177001 | `sinus bradycardia`   | 中 |
| STach | 427084000 | `sinus tachycardia`   | 中 |
| AF    | 164889003 | `atrial fibrillation` | 高 |
| IAVB  | 270492004 | `first degree atrioventricular block` | 中 |
| LBBB  | 164909002 | `left bundle branch block`  | 中 |
| RBBB  | 59118001  | `right bundle branch block` | 中 |
| PVC   | 427172004 | `premature ventricular contraction` | 中 |

选取理由：
- 全部都在 ECGTwin 官方 demo 或 MIMIC 训练集高频文本里，**生成质量有 baseline 保证**
- PTBXL 三数据集全覆盖、**无 -1 mask**，可以 unbias 地计算 per-class AUPRC
- 覆盖四大 ECG 语义类别：**节律（NSR/SB/STach/AF）+ 传导阻滞（IAVB/LBBB/RBBB）+ 异位搏动（PVC）**，
  对抗微调后 EfficientNetV2 的跨中心泛化提升能**逐类归因**

### 第 4 层（不建议作为主要目标）

- **LVH / MI 系列**：ECGTwin demo 有，但 26 类头 -1 / 语义错配，需要单独改分类头才能评，**作为扩展实验，不进主 pipeline**
- **Acute pericarditis / LAD / RAD / PRWP / LQRSV / LPR / Brady / ILBBB / SA / QAb / TAb / NSIVCB / LAnFB / LQT / PR / PAC / BBB**：
  要么在 ECGTwin 生成稳定性未验证、要么某数据集上过稀，**只作 Tier-2 的评测范围，不作 AdvDiff 生成目标**

---

## 6. 落地操作建议

### 训练头
```python
# 照搬 scripts/crosscenter_v2/label_alignment_v2.py:22-49 的 SCORED_26
# train_ptbxl_v2.py 已经跑通，直接继承即可
NUM_CLASSES = 26
CRITERION = MaskedFocalLoss(alpha=0.25, gamma=2.0)  # 忽略 -1
```

### AdvDiff 目标生成
```python
ADVDIFF_TARGET_CLASSES = [
    ("NSR",   "normal sinus rhythm",             "426783006"),
    ("SB",    "sinus bradycardia",               "426177001"),
    ("STach", "sinus tachycardia",               "427084000"),
    ("AF",    "atrial fibrillation",             "164889003"),
    ("IAVB",  "first degree atrioventricular block", "270492004"),
    ("LBBB",  "left bundle branch block",        "164909002"),
    ("RBBB",  "right bundle branch block",       "59118001"),
    ("PVC",   "premature ventricular contraction", "427172004"),
]
# 在 adversarial/adv_generate.py 里作为 target 标签循环，
# ref_label["text"] 直接塞自由文本形式
```

### 评测
- **主表**：Tier-1 5 类 AUROC/AUPRC，三中心（PTBXL hold-out / PN2021 each center / MIMIC subset）
- **Ablation**：Tier-2 15 类 + 8 类 AdvDiff 目标子集的 per-class 对比
- **Sanity**：26 类整体 AUROC，确认 AdvDiff finetune 没在非目标类上造成灾难性遗忘

---

## 7. 未竟问题 / 待后续验证

1. **ECGTwin 对 26 类里除 12 demo 外的自由文本生成质量如何？**
   → 需要一个小实验：各采 10 个 prompt，人肉 / sanity_check 评判。建议
   在 `sub_experiment/tsne_clustering_v1/` 完成后顺带测（同基础设施）。
2. **MIMIC regex 标签噪声到底多严重？**
   → `docs/training/gap_report.md` 已有初步讨论，若要引入 MIMIC 参与训练（而不只是评测）
   需要 label-denoising 或 confident-learning 类方法。
3. **LVH / MI 要不要单独扩 headed？** 临床意义大且 ECGTwin 能生成，
   但代价是重训分类头；留到主 pipeline 跑通后再议。

---

## 7.5 ECGTwin-first 标签方案（新增：若以 ECGTwin 生成类别为主导）

### 问题改写

前文以 PN2021 26-class 为主头、ECGTwin-friendly 8 类为 AdvDiff 目标。
若反过来 —— **把 ECGTwin 官方 demo 的 12 类当主分类头**，需要回答：

> 这 12 类里有哪些能在 PTBXL / PN2021 / MIMIC 三数据集都拿到可靠标签？

### 三数据集 × ECGTwin 12 类对齐表

| ECGTwin demo | PTBXL (SCP) | PN2021 (SNOMED) | MIMIC (regex) | 三数据集交集 |
|---|---|---|---|---|
| normal_ecg (NSR)      | ✅ `NORM`, `SR`                           | ✅ 426783006 (scored) | ✅ | ★ |
| sinus_bradycardia (SB) | ✅ `SBRAD`                                | ✅ 426177001 (scored) | ✅ | ★ |
| sinus_tachycardia     | ✅ `STACH`                                | ✅ 427084000 (scored) | ✅ | ★ |
| atrial_fibrillation   | ✅ `AFIB`                                 | ✅ 164889003 (scored) | ✅ | ★ |
| av_block (IAVB)       | ✅ `1AVB` (+ `2AVB`,`3AVB` 可并入 high-AVB) | ✅ 270492004 (1°, scored) | ✅ | ★ |
| LBBB                  | ✅ `CLBBB`                                | ✅ 164909002 (scored) | ✅ | ★ |
| RBBB                  | ✅ `CRBBB`                                | ✅ 59118001  (scored) | ✅ | ★ |
| LVH                   | ✅ `LVH`（PTBXL 原生 SCP）                | ⚠️ 164873001 **存在但非 scored 26** | ✅ (regex) | △ 需扩 mapping |
| MI (IMI/AMI/…)        | ✅ `IMI`,`AMI`,`LMI`,`PMI`,`ASMI`,`IPMI`… | ⚠️ 57054005/164861001 **存在但非 scored 26** | ✅ (regex) | △ 需扩 mapping |
| NSTEMI                | ⚠️ 无专属 SCP（`NDT`,`NST_` 仅形态学代理） | ⚠️ 非 scored | ✅ (regex) | △ 弱 |
| STEMI                 | ⚠️ `INJAS`,`INJAL`（ST 段损伤作代理）     | ⚠️ 非 scored | ✅ (regex) | △ 弱 |
| acute_pericarditis    | ⚠️ 极少（<30 条）                          | ⚠️ 基本无 scored 编码 | ✅ (regex) | ✗ 样本量不够 |

**横竖全绿的只有 7 类**：NSR / SB / STach / AF / IAVB / LBBB / RBBB。

### ★ 严格交集 = 7 类（最终推荐）

这是"ECGTwin 友好 ∩ PTBXL ∩ PN2021 ∩ MIMIC"四方全部官方支持的唯一集合：

| # | 代号 | 全称 | SNOMED | PTBXL SCP | PN2021 scored | MIMIC regex | ECGTwin prompt |
|---|---|---|---|---|---|---|---|
| 1 | **NSR**   | Normal Sinus Rhythm           | 426783006 | `NORM`, `SR` | ✅ | ✅ | `normal sinus rhythm` |
| 2 | **SB**    | Sinus Bradycardia             | 426177001 | `SBRAD`      | ✅ | ✅ | `sinus bradycardia` |
| 3 | **STach** | Sinus Tachycardia             | 427084000 | `STACH`      | ✅ | ✅ | `sinus tachycardia` |
| 4 | **AF**    | Atrial Fibrillation           | 164889003 | `AFIB`       | ✅ | ✅ | `atrial fibrillation` |
| 5 | **IAVB**  | First-degree AV Block         | 270492004 | `1AVB`       | ✅ | ✅ | `first degree atrioventricular block` |
| 6 | **LBBB**  | Left Bundle Branch Block      | 164909002 | `CLBBB`      | ✅ | ✅ | `left bundle branch block` |
| 7 | **RBBB**  | Right Bundle Branch Block     | 59118001  | `CRBBB`      | ✅ | ✅ | `right bundle branch block` |

**临床谱系**：节律 4（NSR / SB / STach / AF）+ 传导 3（IAVB / LBBB / RBBB）—— 完整覆盖"节律异常"和"传导异常"两大 ECG 核心类别。

**落地代码（可直接复用）**：

```python
# 从 scripts/crosscenter_v2/label_alignment_v2.py:SCORED_26 切子集
ECGTWIN_STRICT_7 = ["NSR", "SB", "STach", "AF", "IAVB", "LBBB", "RBBB"]
ECGTWIN_STRICT_7_SNOMED = {
    "NSR":   "426783006",
    "SB":    "426177001",
    "STach": "427084000",
    "AF":    "164889003",
    "IAVB":  "270492004",
    "LBBB":  "164909002",
    "RBBB":  "59118001",
}
ECGTWIN_STRICT_7_PROMPTS = {
    "NSR":   "normal sinus rhythm",
    "SB":    "sinus bradycardia",
    "STach": "sinus tachycardia",
    "AF":    "atrial fibrillation",
    "IAVB":  "first degree atrioventricular block",
    "LBBB":  "left bundle branch block",
    "RBBB":  "right bundle branch block",
}
# PTBXL SCP → 7 类索引：复用 label_alignment_v2.py:PTBXL_SCP_TO_CLASSES 并切片
# PN2021 SNOMED → 7 类索引：复用 snomed_to_26() 并切片
# MIMIC regex → 7 类索引：复用 MIMIC_KEYWORD_PATTERNS 并切片
```

**优点**：
- 零改造，已有的 `label_alignment_v2.py` 全部 mapping 直接可用（切 7 维子集即可）
- PTBXL 无 -1 mask，loss 不需要 MaskedFocalLoss 的 mask 逻辑（可用普通 BCE + class-balanced weight）
- PN2021 scored 标签是临床 gold standard，可直接对标已发表的 PN2021 评分 benchmark
- ECGTwin 这 7 类 demo 全部验证过，AdvDiff 梯度引导有稳定生成器做 backbone
- 三数据集均可作训练 + 评测（不只是评测）

**局限**：
- 不覆盖结构性异常（LVH）和缺血（MI）
- 类别数偏少，论文 headline 的临床广度受限
- 若想扩展 → 参考下文方案 B（9 类，需扩 mapping）

### 两套可行的 ECGTwin-first 头（A = 上表 7 类；B = 扩展 9 类）

#### 方案 A（快速路径，零改造，7 类 = 严格交集）

- 直接从 `label_alignment_v2.py:SCORED_26` 里切 7 类的子集 = {NSR, SB, STach, AF, IAVB, LBBB, RBBB}
- 所有映射函数（PTBXL SCP、PN2021 SNOMED、MIMIC regex）**全部已就绪**，只需把 26 维 label 向量切片
- 三数据集都是"scored / 官方编码"级别的标签质量，无 -1 mask
- 代价：覆盖的 ECG 临床谱系偏窄，只有**节律 + 传导**两大类，缺 LVH / MI / 异位搏动

#### 方案 B（扩展路径，9 类，~1 天工作量）

在方案 A 基础上 **+ LVH + MI (merged)**：

| 扩展类 | 扩展工作 |
|---|---|
| **LVH** | `PTBXL_SCP_TO_CLASSES` 加 `'LVH': ['LVH']`；`_SNOMED_TO_IDX` 加 `164873001 → LVH`；`MIMIC_KEYWORD_PATTERNS` 加 `['left ventricular hypertrophy','\\blvh\\b']` |
| **MI** (merged, anterior+inferior+lateral+STEMI+NSTEMI 全部归 MI) | `PTBXL_SCP_TO_CLASSES` 加 `IMI/AMI/LMI/PMI/ASMI/IPMI/ALMI/IPLMI/INJAS/INJAL: ['MI']`；`_SNOMED_TO_IDX` 加 `57054005, 164861001, 413444003, 413439005 → MI`；MIMIC regex `['myocardial infarction','\\bmi\\b','\\bstemi\\b','\\bnstemi\\b','infarct']` |

- 代价：**MI 合并**牺牲了 ST 定位特异性（前壁/下壁/侧壁不再区分），但换来三数据集都可比、且 ECGTwin 能条件生成"Myocardial infarction"
- **不建议拆开**成 STEMI/NSTEMI/老陈旧 MI 子类：PN2021 不区分，PTBXL 的 `INJAS/INJAL` 只是 ST 段损伤代理不等同 STEMI，会引入 label 噪声
- **不建议单列 acute_pericarditis**：PTBXL/PN2021 样本量都极少（<30 条），AUPRC 估计方差会吞掉信号

### 方案 B（扩展路径）

```python
ECGTWIN_FIRST_9 = [
    "NSR",    # normal_ecg
    "SB",     # sinus_bradycardia
    "STach",  # sinus_tachycardia
    "AF",     # atrial_fibrillation
    "IAVB",   # atrioventricular_block（1°; 2°/3° 若需要可合并进同类）
    "LBBB",   # left_bundle_branch_block
    "RBBB",   # right_bundle_branch_block
    "LVH",    # left_ventricular_hypertrophy   ← 扩展
    "MI",     # myocardial_infarction (merged) ← 扩展
]
```

理由：
- **ECGTwin 这 9 类全部有官方 demo 生成验证**（覆盖 12 demo 的 9 个，剩 3 个因数据集支持不足被淘汰）
- **PTBXL**：9 类全部有原生 SCP，**无 -1 mask**（比 26 类头更干净）
- **PN2021**：7 类是 scored、2 类（LVH/MI）用非 scored SNOMED 补；非 scored 不参与官方评分但仍是临床编码，可信度高于 MIMIC regex
- **MIMIC**：9 类 regex 覆盖，作为零样本评测集
- **临床谱系**：节律 4 + 传导 3 + 结构（LVH）1 + 缺血（MI）1，四大 ECG 异常类别齐全

### 代价与风险

- **需要扩 `label_alignment_v2.py` 两处**（LVH + MI mapping），总代码量 < 30 行
- **MI 合并后无法再拆 STEMI vs NSTEMI** —— 但本身这俩在 PTBXL/PN2021 上就没有可信独立标签，保持合并反而**避免虚假性能**
- **PN2021 的 LVH / MI 走非 scored SNOMED**：需要在 `label_alignment_v2.py` 里加一份 `UNSCORED_BUT_USED_SNOMED = {...}` 显式维护，避免与官方 26-class 混淆。实际评测时这两类**不能拿 PN2021 官方 metric 直接对比**其他人论文，得自建 baseline
- **LVH 在 MIMIC 机器报告里经常是"borderline LVH"、"cannot exclude LVH"** → regex 召回会偏高，precision 低。建议 MIMIC 评测只作辅助指标，主结论锁 PTBXL + PN2021

### 与"方案 B 之上再扩"的边界

若后续想再加类：
- **PVC**（异位搏动）：PTBXL ✓ `PVC`、PN2021 ✓ scored、MIMIC ✓；ECGTwin demo 无但 MIMIC 训练文本里高频，生成稳定性**需要小实验验证**。若稳定则 +1 → **10 类**
- **AFL**（房扑）：同 PVC 情况；PN2021 有 scored、PTBXL 有 `AFLT`，ECGTwin demo 未覆盖
- **不建议**：pericarditis（样本太少）、STEMI/NSTEMI 独立拆（跨库无可靠标签）

---

## 7.6 实测核查（2026-04-19 补充）

对 §7.5 推荐的 7 类做了实证核查（PTBXL 21,799 条 + PN2021 五主中心 65,826 条）。
结果与原推荐有**两处需要修正**。

### 7.6.1 ECGTwin demo 核查 ✅ 通过

`model/ECGTwin/generation_result_by_disease/` 实际存在 12 个子目录，
其中 7 类 ECGTwin-friendly 全部有 demo 输出：
```
normal_ecg/, sinus_bradycardia-窦性心动过缓/, sinus_tachycardia-窦性心动过速/,
atrial_fibrillation-心房颤动/, atrioventricular_block-房室传导阻滞/,
left_bundle_branch_block-左束支阻滞/, right_bundle_branch_block-右束支阻滞/
```

### 7.6.2 PTBXL 核查 ✅ 通过

PTBXL (N=21,799) 对 7 类的 SCP 阳性计数（presence-based, conf≥0）：

| 类 | SCP 编码 | 正样本 | 占比 |
|---|---|---|---|
| NSR   | `NORM` or `SR` | 18,058 | 82.8% |
| SB    | `SBRAD`        |    637 |  2.9% |
| STach | `STACH`        |    826 |  3.8% |
| AF    | `AFIB`         |  1,514 |  6.9% |
| IAVB  | `1AVB`         |    793 |  3.6% |
| LBBB  | `CLBBB`        |    536 |  2.5% |
| RBBB  | `CRBBB`        |    541 |  2.5% |

**结论**：7 类在 PTBXL 全部 >500 正样本，训练信号充足。

### 7.6.3 PN2021 五主中心核查 ⚠️ **发现两处需修正**

五主中心 (chapman_shaoxing / cpsc_2018 / cpsc_2018_extra / georgia / ningbo) 阳性计数：

| 类 | chapman | cpsc_2018 | cpsc_2018_extra | georgia | ningbo | 最小中心 | 可全 5 中心评测? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| NSR   | 1,826 |   918 | **4**   | 1,752 | 6,299 |   **4**   | ⚠️ 警戒 |
| SB    | 3,889 | **0** |    45   | 1,677 |12,670 |   **0**   | ✗ 不可 |
| STach | 1,568 | **0** |   303   | 1,261 | 5,687 |   **0**   | ✗ 不可 |
| AF    | 1,780 | 1,221 |   153   |   570 | **0** |   **0**   | ⚠️ 需修 mapping |
| IAVB  |   247 |   722 |   106   |   769 |   893 |  106      | ✅ 可 |
| LBBB  |   205 |   236 |    38   |   231 |   248 |   38      | ✅ 可 |
| RBBB  |   454 | 1,857 |   114   |   556 | 1,291 |  114      | ✅ 可 |

**根因诊断**（进一步挖 top-SNOMED 查证）：

1. **AF 在 ningbo = 0 的真相**：ningbo 中用 SNOMED `164890007` 编码了 **7,615** 条，
   而 `label_alignment_v2.py:25` 把 `164890007` 归到 AFL（房扑），没归到 AF。
   考虑到 ningbo 这一编码的高频度（占全中心 22%，远超房扑真实流行率 ~2%），
   几乎可以肯定 ningbo 把 AF + AFL 合并编码到 `164890007` 下了（或其标注惯例与 PN2021 官方解释相左）。
   **可通过在 `_SNOMED_TO_IDX` 里加 `164890007 → [AFL, AF]` 同时双射解决**，代价是
   AFL 单独 metric 会掺入少量 AF 误判，但 AF 的跨中心可评性得到拯救。

2. **SB / STach 在 cpsc_2018 = 0 的真相**：cpsc_2018 是 2018 China Physiological Signal Challenge 的原始数据集，
   原任务只有 9 类（Normal / AF / 1AVB / LBBB / RBBB / PAC / PVC / STD / STE），
   **SB / STach 不是标注目标**，其中的窦缓/窦速样本都归到 "Normal"。
   这是**数据集结构性缺失**，无法通过修 mapping 补救。

3. **NSR 在 cpsc_2018_extra = 4 的真相**：cpsc_2018_extra 是 CPSC 2018 的**异常样本补充集**，
   设计上就是供训练期补异常样本，NSR 本来就不应该多。
   无解但可接受（仍有 4 个样本，不是结构性 0）。

### 7.6.4 修正后的推荐：三层置信

基于实测证据，把 7 类切成三个置信度层次：

#### Tier-S（严格跨中心可评，3 类）

> **IAVB / LBBB / RBBB**

- PN2021 五主中心全覆盖（最小 38 条 @ cpsc_2018_extra 的 LBBB）
- PTBXL 均 >500 条；MIMIC regex 稳定
- ECGTwin demo 验证
- **用途**：论文主表 / "最干净"的跨中心泛化 headline 指标

#### Tier-M（加修 AF mapping + 接受 cpsc_2018_extra 弱 NSR，6 类）

> **IAVB / LBBB / RBBB / NSR / STach / AF**

- 需在 `label_alignment_v2.py:29` 把 `AFL` 条目改成 `164890007 → [AFL, AF]` 双射
- 需接受 NSR 在 cpsc_2018_extra 只有 4 条（该中心被设计为异常样本集，合理）
- STach 只丢 cpsc_2018 一个中心（同样是 CPSC 2018 原任务不标 STach 的结构性缺失）
- **用途**：训练头 / 一般跨中心 metric，覆盖四大语义类（节律 3 + 传导 3）

#### Tier-X（训练可用但 SB 慎评，7 类 = 原 §7.5 全集）

> **+ SB**

- SB 在 cpsc_2018 = 0 是结构性，无法修
- 训练信号完全够（PTBXL 637、ningbo 12,670、chapman 3,889）
- 但**跨中心评测时需显式 mask cpsc_2018 的 SB 结果**（per-class per-center 报表把这格标 N/A）
- **用途**：AdvDiff 目标生成（尽可能覆盖 ECGTwin 全部 rhythm 类 demo）

### 7.6.5 最终建议（分场景）

| 场景 | 推荐标签集 | 理由 |
|---|---|---|
| EfficientNetV2 **主训练头** | **Tier-M（6 类）** | 覆盖四大语义 + 跨中心可评 + 只需 1 行 mapping 修改 |
| 论文**主表 metric** | **Tier-S（3 类）** + Tier-M per-class | 严格 all-center metric 保主 headline，Tier-M 给细节 |
| **AdvDiff 对抗样本生成** | **Tier-X（7 类）** | 生成覆盖 ECGTwin 全部 rhythm demo，SB 参与训练但评测时 mask cpsc_2018 |
| **消融/扩展**（后续实验） | Tier-X + LVH + MI（9 类）| 若愿意走 §7.5 方案 B 的 mapping 扩展工作 |

### 7.6.6 落地 patch（Tier-M 6 类即可运行）

```python
# scripts/crosscenter_v2/label_alignment_v2.py 的最小修改
# (1) 放宽 AFL → 同时命中 AF 和 AFL（修 ningbo AF=0 的问题）
SCORED_26[2] = ("AFL", [164890007])   # 原样保留
# 额外在 _SNOMED_TO_IDX 里加：
_SNOMED_TO_IDX.setdefault(164890007, []).append(CLASS_TO_IDX["AF"])
# 即 164890007 同时映射到 AFL 和 AF（双射）

# (2) 从 26 类里切 Tier-M 6 类子集
ECGTWIN_TIER_M_6 = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]
ECGTWIN_TIER_M_6_IDX = [CLASS_TO_IDX[c] for c in ECGTWIN_TIER_M_6]

# (3) 训练时 label 取 6 维子集；MaskedFocalLoss 可退化为普通 BCE
#     （Tier-M 6 类在 PTBXL 无 -1 mask）
labels_6 = labels_26[:, ECGTWIN_TIER_M_6_IDX]
```

---

## 8. 附：关键文件索引

| 角色 | 路径 |
|---|---|
| **主力标签对齐** | `scripts/crosscenter_v2/label_alignment_v2.py` |
| 现役训练入口 | `scripts/crosscenter_v2/train_ptbxl_v2.py` |
| 跨中心评测 | `scripts/crosscenter_v2/eval_crosscenter_v2.py` |
| MIMIC 零样本评测 | `scripts/crosscenter_v2/eval_mimic_zeroshot.py` |
| Gap 报告 | `docs/training/gap_report.md` |
| AdvDiff victim head (77 类, legacy) | `adversarial/label_mapping.py` |
| ECGTwin 包装器 | `util/ecgtwin_utils.py` |
| ECGTwin 训练数据说明 | `model/ECGTwin/data/README_Data.md` |
| ECGTwin 官方 demo 类别 | `model/ECGTwin/generation_result_by_disease/` |
