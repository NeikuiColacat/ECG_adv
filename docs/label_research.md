# ECG异常检测分类头设计与ECGTwin数据增强整合方案

> 研究日期：2026-04-24　｜　面向：MIMIC-IV-ECG + PTB-XL + PhysioNet/CinC Challenge 2021 多数据集多标签分类任务；生成式增强=ECGTwin (Lai et al., 2025, arXiv:2508.02720)
> 说明：ECGTwin未在GitHub开放官方代码仓（截至2026-04），其前驱工作 **DiffuSETS** (Lai et al., Patterns 2025, arXiv:2501.05932) 的VAE latent和文本条件设计被ECGTwin直接复用，代码位于 `https://github.com/PKUDigitalHealth/DiffuSETS` ，可作为ECGTwin落地的实际依托。部分PART 5/6中对ECGTwin的精确类表描述存在研究文献中未公开的细节（见相应说明）。

---

## PART 1　分类粒度候选对比

### 表1.1　七种分类头方案总览

| 方案 | 类数 | 原论文 / 机构 | 引用量级(估) | PTB-XL可用 | Challenge 2021可用 | MIMIC-IV-ECG可用 | 实现难度(1-5) | 最强背书理由 |
|---|---|---|---|---|---|---|---|---|
| **A. PTB-XL 5 superclass** (NORM, MI, STTC, CD, HYP) | 5 | Wagner et al. 2020 *Sci. Data* 7:154 (PTB/HHI Berlin) [Source](https://www.nature.com/articles/s41597-020-0495-6) | >1500次;是PTB-XL benchmark默认 | ✅ 原生 (`diagnostic_superclass`) | 部分(需SNOMED→superclass聚合) | 无 (需弱标签) | **1** | Strodthoff et al. 2021 JBHI benchmark, ~90%下游论文采用 [Source](https://doi.org/10.1109/JBHI.2020.3022989) |
| **B. PTB-XL 23 subclass / 44 form+rhythm** | 23或44 | Wagner et al. 2020同上 | ~数百 | ✅ 原生 | 需精细映射 | 无 | 2 | 同一数据集内高粒度benchmark；但跨数据集兼容性差 |
| **C. PhysioNet/CinC Challenge 2021 scored** | **30 SNOMED→ 4对equivalence合并→26独立评估类** | Reyna et al. 2021 CinC 48:1-4; Reyna et al. 2022 *Physiol. Meas.* 43:084001 (Emory) [Source](https://moody-challenge.physionet.org/2021/) | >500次引用,>300队参赛 | ✅ (PTB-XL是Challenge 2021训练源之一) | ✅ 原生 | 无 (需弱标签) | **2** | 跨机构最广benchmark(CPSC+INCART+PTB+PTB-XL+Georgia+Chapman+Ningbo), 官方evaluation repo可直接reuse [Source](https://github.com/physionetchallenges/evaluation-2021) |
| **D. PTB-XL 71 full SCP-ECG** | 71 | Wagner et al. 2020同上 | ~数十 | ✅ 原生(`scp_codes`) | 部分映射 | 无 | 4 | 最细粒度,但类别极不平衡, 多数论文报告AUROC而放弃AUPRC |
| **E. Ribeiro et al. 6-class (1dAVb, RBBB, LBBB, SB, AF, ST)** | 6 | Ribeiro et al. 2020 *Nat. Commun.* 11:1760 (UFMG/Uppsala) [Source](https://www.nature.com/articles/s41467-020-15432-4) | >2000次引用(CODE ~2.3M ECG) | 需映射 | 需映射(对应SNOMED: 270492004/59118001/164909002/426177001/164889003/427084000) | 无 | **1** | 最大规模ECG DL研究之一, F1>80%, 预训练权重公开于 Zenodo 3625017 [Source](https://zenodo.org/records/3765780) |
| **F. Hannun et al. 12-class rhythm** | 12 | Hannun et al. 2019 *Nat. Med.* 25:65-69 (Stanford/iRhythm; AF/AFL, AVB, EAR, IVR, junctional, SR, SVT, VT, sinus tach, sinus brady, other, noise) [Source](https://www.nature.com/articles/s41591-018-0268-3) | >2200次 | 仅rhythm覆盖 | 仅rhythm类 | 仅rhythm类 | 3 | single-lead ambulatory导向，12-lead迁移需重训 |
| **G. Hughes/Mayo style binary single-label** | N个二分类头 | 多篇 Attia/Hughes *Nat. Med.* 2019-2022 (Mayo Clinic) [Source](https://www.nature.com/articles/s41591-019-0158-5) | 数百 | 可拆分 | 可拆分 | 可拆分 | 5 (N×训练) | 单病种临床部署强, 但不适合benchmark多任务 |

### 🏆 推荐：**方案 C — PhysioNet/CinC Challenge 2021 scored diagnoses (30 SNOMED labels, 26 effective classes)**

**判据汇总：**

| 标准 | C方案 | 备注 |
|---|---|---|
| (a) 实现难度 | 2/5 | 官方`dx_mapping_scored.csv`+`weights.csv`现成, `evaluate_model.py`开箱即用 |
| (b) 学术背书 | ★★★★★ | 2篇CinC+PhysMeas peer-reviewed,>300队基准,后续ECG-FM/KED/Self-DANA等均沿用 |
| (c) 三数据集兼容性 | **最佳** | PTB-XL是Challenge 2021原生子集；MIMIC-IV-ECG通过弱标签管道可对齐 |
| (d) ECGTwin兼容 | ✅ | 26类包含ECGTwin训练集MIMIC-IV-ECG cart report高频语义(AF, MI, RBBB, LBBB, 1dAVb, sinus rhythm, sinus brady, sinus tach等) |

---

## PART 2　PhysioNet/CinC Challenge 2021 Scored Diagnoses 深度解析

来源：2020 Challenge paper Table 3 (Perez Alday et al. 2021 *Physiol. Meas.* 41:124003) [Source](https://pmc.ncbi.nlm.nih.gov/articles/PMC8015789/)；2021 Challenge 在 2020 的 27 类基础上扩展至 **30 个 scored SNOMED codes**（根据 `dx_mapping_scored.csv`），通过 **4 对 equivalence pair**（用 `|` 分隔）合并后实际独立评估维度 = **26**。4 对中 3 对自 2020 沿用（CRBBB|RBBB, PAC|SVPB, PVC|VPB），2021 新增第 4 对 **CLBBB|LBBB (733534002|164909002)**。

> 注：Challenge 2020 为 27 scored rows；Challenge 2021 保留同一 scoring 框架，扩增数据源（加入 Chapman-Shaoxing + Ningbo + UMich）并追加至 30 scored rows。官方 `dx_mapping_scored.csv` 30 行、`weights.csv` 26 列（= 30 rows − 4 equivalence pairs）[Source](https://github.com/physionetchallenges/evaluation-2021)。

### 表2.1　30个Scored Labels (基于2020 Table 3 + 2021扩展; 等价组用**粗体**色标识)

| # | Diagnosis | SNOMED-CT | Abbr. | Equivalence group | 训练总数(2020公开部分) | 频率(近似) | 稀有? |
|---|---|---|---|---|---|---|---|
| 1 | 1st degree AV block | 270492004 | IAVB | — | 2946 | ~6.8% | ○ |
| 2 | Atrial fibrillation | 164889003 | AF | — | 4026 | ~9.3% | ○ |
| 3 | Atrial flutter | 164890007 | AFL | — | 423 | ~1.0% | △ |
| 4 | Bradycardia | 426627000 | Brady | — | 289 | ~0.7% | **✓稀有** |
| 5 | Complete RBBB | 713427006 | CRBBB | **G1 (=RBBB)** | 701 | ~1.6% | ○ |
| 6 | Right bundle branch block | 59118001 | RBBB | **G1** | 3051 | ~7.1% | ○ |
| 7 | Incomplete RBBB | 713426002 | IRBBB | — | 1817 | ~4.2% | ○ |
| 8 | Left anterior fascicular block | 445118002 | LAnFB | — | 1916 | ~4.4% | ○ |
| 9 | Left axis deviation | 39732003 | LAD | — | 6564 | ~15.2% | ○ |
| 10 | Left bundle branch block | 164909002 | LBBB | — | 1197 | ~2.8% | ○ |
| 11 | Low QRS voltages | 251146004 | LQRSV | — | 748 | ~1.7% | ○ |
| 12 | Nonspecific IVCB | 698252002 | NSIVCB | — | 1093 | ~2.5% | ○ |
| 13 | Pacing rhythm | 10370003 | PR | — | 301 | ~0.7% | **✓稀有** |
| 14 | Premature atrial contraction | 284470004 | PAC | **G2 (=SVPB)** | 2188 | ~5.1% | ○ |
| 15 | Supraventricular premature beats | 63593006 | SVPB | **G2** | 215 | ~0.5% | **✓稀有(单独)** |
| 16 | Premature ventricular contraction | 427172004 | PVC | **G3 (=VPB)** | — | — | — |
| 17 | Ventricular premature beats | 17338001 | VPB | **G3** | 543 | ~1.3% | △ |
| 18 | Prolonged PR interval | 164947007 | LPR | — | 340 | ~0.8% | **✓稀有** |
| 19 | Prolonged QT interval | 111975006 | LQT | — | 2253 | ~5.2% | ○ |
| 20 | Q wave abnormal | 164917005 | QAb | — | 1252 | ~2.9% | ○ |
| 21 | Right axis deviation | 47665007 | RAD | — | 465 | ~1.1% | △ |
| 22 | Sinus arrhythmia | 427393009 | SA | — | 1476 | ~3.4% | ○ |
| 23 | Sinus bradycardia | 426177001 | SB | — | 3219 | ~7.5% | ○ |
| 24 | Sinus rhythm (normal) | 426783006 | NSR | — | 21944 | ~50.9% | ○ |
| 25 | Sinus tachycardia | 427084000 | STach | — | 3050 | ~7.1% | ○ |
| 26 | T wave abnormal | 164934002 | TAb | — | 5792 | ~13.4% | ○ |
| 27 | T wave inversion | 59931005 | TInv | — | 1550 | ~3.6% | ○ |
| 28-30 | 2021新增3个(含CLBBB/Incomplete LBBB系若干派生) | — | — | — | — | — | — |

### 表2.2　4 对 Equivalence Pair → 实际独立评估 = 26 类

权威来源：`weights.csv` 第一行列头（pipe 分隔），2026-04 从 GitHub 直接拉取核查。

| # | Equivalence Pair (pipe-merged in weights.csv) | 临床合并理由 | 加入年份 |
|---|---|---|:---:|
| P1 | CRBBB (713427006) \| RBBB (59118001) | 完全性 RBBB ≡ 广义 RBBB | 2020 |
| P2 | PAC (284470004) \| SVPB (63593006) | PAC ≡ 室上性早搏 | 2020 |
| P3 | PVC (427172004) \| VPB (17338001) | 双 SNOMED 同一室性早搏 | 2020 |
| P4 | CLBBB (733534002) \| LBBB (164909002) | 完全性 LBBB ≡ 广义 LBBB | 2021 新增 |

> 30 scored SNOMED rows − 4 pairs × 1 merge = **26 effective classes**（= `weights.csv` 列数，官方评分矩阵维度）[Source](https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv)。
>
> 早期笔记里"7 个 equivalence groups → 24 effective"的说法有误，实事核查 `weights.csv` 后更正为 4 对 → 26 effective。

### 稀有类 (<1% prevalence) — ECGTwin 增强优先对象

**优先增强目标：Brady, PR (pacing rhythm), SVPB, LPR, AFL(~1%边缘), RAD(~1%边缘), VPB**。这些类在macro-AUPRC上会主导评价指标。

---

## PART 3　跨数据集标签对齐表

### 表3.1　26 Challenge 2021 Effective Classes 在三数据集中的映射

| SNOMED / Group | Abbr. | PTB-XL SCP code (见`scp_statements.csv`) | Challenge 2021 原生 | MIMIC-IV-ECG weak-label 来源 |
|---|---|---|---|---|
| 270492004 | IAVB | 1AVB | ✅ | cart report regex "1st degree AV block\|1st deg*"; ICD-10 I44.0 |
| 164889003 | AF | AFIB | ✅ | regex "atrial fibrillation"; ICD-10 I48.* |
| 164890007 | AFL | AFLT | ✅ | regex "atrial flutter" |
| 426627000 | Brady | SBRAD? or via HR | ✅ | regex "brady*" + HR<60 from machine measurements |
| 713427006 \| 59118001 | CRBBB\|RBBB | CRBBB / IRBBB(部分) | ✅ | regex "right bundle branch block" |
| 713426002 | IRBBB | IRBBB | ✅ | regex "incomplete RBBB" |
| 445118002 | LAnFB | LAFB | ✅ | regex "left anterior (fascicular\|hemiblock)" |
| 39732003 | LAD | LAD (via heart_axis) | ✅ | regex "left axis deviation" |
| 164909002 | LBBB | CLBBB / ILBBB | ✅ | regex "left bundle branch block" |
| 251146004 | LQRSV | LOWT? LVOLT | ✅ | regex "low voltage" |
| 698252002 | NSIVCB | IVCD | ✅ | regex "nonspecific intraventricular" |
| 10370003 | PR | PACE | ✅ | regex "pacing\|paced" + cart "atrial/ventricular paced" |
| 284470004 \| 63593006 | PAC\|SVPB | PAC / SVARR | ✅ | regex "premature atrial\|supraventricular premature" |
| 427172004 \| 17338001 | PVC\|VPB | PVC | ✅ | regex "premature ventricular\|ventricular ectop*" |
| 164947007 | LPR | LPR | ✅ | regex "prolonged PR\|first-degree" (重叠IAVB) |
| 111975006 | LQT | LNGQT | ✅ | regex "prolonged QT\|long QT" + QTc>450 |
| 164917005 | QAb | QWAVE / ABQRS | ✅ | regex "Q wave abnormal\|abnormal Q" |
| 47665007 | RAD | RAD | ✅ | regex "right axis deviation" |
| 427393009 | SA | SARRH | ✅ | regex "sinus arrhythmia" |
| 426177001 | SB | SBRAD | ✅ | regex "sinus bradycardia" |
| 426783006 | NSR | NORM / SR | ✅ | regex "normal sinus\|normal ECG" |
| 427084000 | STach | STACH | ✅ | regex "sinus tachycardia" |
| 164934002 | TAb | TAB_ / ABT | ✅ | regex "T wave abnormal" |
| 59931005 | TInv | INVT | ✅ | regex "T wave inversion" |

**PTB-XL → Challenge 2021 官方映射脚本**：`physionetchallenges/python-example-2024/prepare_ptbxl_data.py` 已实现 `scp_codes → diagnostic_superclass + SL (GE Marquette 12SL) → SNOMED` 合并，使用 `scp_statements.csv` + `12slv23ToSNOMED.csv` [Source](https://github.com/physionetchallenges/python-example-2024/blob/main/prepare_ptbxl_data.py)。推荐直接复用并裁剪到 26 类。

**可复用PTB-XL迁移benchmark**：`helme/ecg_ptbxl_benchmarking` [Source](https://github.com/helme/ecg_ptbxl_benchmarking)。

---

## PART 4　MIMIC-IV-ECG Weak-Label 方案对比与推荐

MIMIC-IV-ECG v1.0 (Gow et al. 2023, DOI 10.13026/4nqg-sb35) 原生包含: (a) 12SL机器cart report (report_0..report_17列, machine_measurements.csv), (b) 全局测量(rr_interval, qrs_onset/end, qt, qtc等), (c) ~600k份cardiologist free-text reports [Source](https://physionet.org/content/mimic-iv-ecg/1.0/)。**无结构化诊断标签。**

### 表4.1　六种弱标签方案对比

| # | 方案 | 实现者 / 代表论文 | 覆盖率 | 噪声水平 | 算力成本 | 公共label file | 与Challenge 2021 对齐难度 |
|---|---|---|---|---|---|---|---|
| (i) | **MIMIC-IV-ECG-Ext-ICD** — 通过subject_id链接MIMIC-IV住院/ED ICD-10-CM代码 | Strodthoff/Lopez Alcaraz/Haverkamp 2024 *EHJ-Digital Health* ztae039 (UOL Oldenburg) | ~184,700 ED样本 / 83,738患者 (non-ED+ED合计全库子集) | **高噪声**: ICD-10反映住院诊断,非ECG当下异常;时间错配; recall低(许多 AF发作期未被入院编码) | 低(仅SQL join) | ✅ PhysioNet DOI 10.13026/ypt5-9d58 [Source](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/) + 代码 [Source](https://github.com/AI4HealthUOL/ECG-MIMIC) | 高（ICD-10粒度与SNOMED-CT不对齐；需再映射） |
| (ii) | **ECG-FM预训练label方案** — 将MIMIC-IV-ECG cart report解析成SCP-like标签后用于diagnosis微调 | McKeen/Wang 2025 *JAMIA Open* 8:ooaf122 (UHN/Vector/UofT) [Source](https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827) | 整套MIMIC-IV-ECG v1.0 | 中; 依赖cart machine statements | 低(基于fairseq-signals pipeline) | 部分;checkpoints 公开于 HF `wanglab/ecg-fm-preprint` [Source](https://github.com/bowang-lab/ECG-FM) | 中；需再regex对齐 |
| (iii) | **Regex / rule-based cart report提取** | Jwoo5 `ECG-QA` v1.0.2 保留 **155 SCP codes** 基于cart statements手动映射(参照SCP-ECG v3.0标准,与PTB-XL同类) | 全库 | 中偏低(cart 12SL算法本身F1 ~ 0.7-0.9对多数主要类) | 极低 | ✅ GitHub [Source](https://github.com/Jwoo5/ecg-qa) | **低**;155 SCP中可映射到 Challenge 26类约20类 |
| (iv) | **LLM-based (GPT-4o)报告标注** | MEETI dataset (Hong/Peking U 2025 *Sci. Data*) 使用GPT-4o+ECG features生成高粒度解释 [Source](https://www.nature.com/articles/s41597-026-06796-1) | 全库~800k | 中;需 prompt engineering校对 | **高**(API费用~几千USD,或本地med-LLM) | ✅ HuggingFace / PhysioNet (pending) | 中; 输出自由文本需再二次regex |
| (v) | **Bootstrapping / Pseudo-labeling** — 用在PhysioNet 2021训练的分类器打标签 | 多个工作（如ECG-FM linear probing, Self-DANA FM） | 全库 | 中; confirmation bias风险 | 中 (一次性forward pass) | 需自行跑 | 原生对齐(目标label空间 = 源label空间) |
| (vi) | **ECG-QA v1.0.2 / MEETI / PTB-XL+** 等2024-2026开源标签 | Oh et al. 2023(ECG-QA); MEETI 2026; PTB-XL+ (Strodthoff et al.) | 各有侧重 | 因来源异 | 低(下载即用) | ✅ 已释放 | 需映射 |

### 🏆 推荐：(iii) + (v) 两阶段混合

1. **Stage A — Rule-based regex warm start**: 复用 `ECG-QA` 仓已手工维护的 **155 SCP-ECG v3.0 cart statements → SNOMED** 词典(它"manually labeled them according to the corresponding machine-generated statements with reference to SCP-ECG v3.0 standard" [Source](https://github.com/Jwoo5/ecg-qa))，裁剪到与 Challenge 2021 的 26 类对齐的子集(~20 个可直接对齐,如 AF, AFL, SB, STach, LBBB, RBBB, IAVB, PAC, PVC, LAD, RAD, LQT, QAb, TAb, TInv, NSR, PR/pacing, LAnFB, IVCB, LQRSV)。
2. **Stage B — Pseudo-label refinement**: 用在PhysioNet Challenge 2021 public training set上训练出的分类器(ECG-FM finetune或SE-ResNet1d101)对全MIMIC-IV-ECG进行概率预测，在regex阳性样本上做置信度融合(union+conf-threshold)，在regex无输出样本上做零样本伪标签(threshold>0.9)。
3. **可选补充**: 用(i) MIMIC-IV-ECG-Ext-ICD 的AF(I48)/HF/MI(I21)/1dAVb(I44.0)作为 **远程监督信号** 用于额外一致性正则(只对这几个与ICD强相关的诊断类有效)。

**不推荐单独使用(i)**: ICD-10住院编码时间跨度宽(一个I48编码可能对应多份ECG,有些是窦律)，会造成训练标签噪声过大。Strodthoff原论文在其benchmark自己承认对ECG形态类(如TAb, LAD)AUROC显著低于Challenge 2021监督范式。

---

## PART 5　ECGTwin 兼容性分析

### 表5.1　ECGTwin 关键事实

| 属性 | 值 | 来源 |
|---|---|---|
| 论文 | *ECGTwin: Personalized ECG Generation Using Controllable Diffusion Model* (Lai, Chen, Zhao, Zhang, Wang, Geng, Li, Hong) | arXiv:2508.02720v2 (2026-02-02) [Source](https://arxiv.org/abs/2508.02720) |
| 第一作者所属 | 北大/PKU-Wuhan AI + Peking U Health Data Science + HeartVoice | 同上 |
| 架构 | 两阶段：(1) Individual Base Extractor (contrastive, self-sup) → base vector **b**; (2) LDM(潜空间=DiffuSETS pretrained VAE `R^{4×128}`) + **AdaX Condition Injector** = Cross-Attention(Cardiac Condition path) + adaLN(Base+Time path);两种backbone：DiT与UNet | [Source](https://arxiv.org/html/2508.02720v1) |
| 训练数据 | MIMIC-IV-ECG (cart reports text pairs, ~20×大于前期个性化工作) | 同上 |
| 外部验证集 | PTB-XL (Table 6 报告out-of-distribution性能,CLIP Score略降因德语报告) | 同上 |
| **条件词汇** | **自由文本cart report**: 采用 per-report token + `nomic-embed-text-v1.5` (768-dim) 进行tokenize, 而非闭集类别embedding;因此"支持条件"=MIMIC-IV-ECG cart report涵盖的全部语义 | [Source](https://arxiv.org/html/2508.02720v1)报告明确说明"report-level tokenization allow the model to selectively attend to different reports" |
| 评估 | FID / Precision / Recall / F1 (分布层) + HR-MAE + CLIP Score(语义对齐层) + 下游诊断增强 (application层, 基于DiffuSETS "three-level evaluation protocol") | 同上 |
| 代表性生成案例 | 在前驱DiffuSETS (Lai et al. Patterns 2025)中已验证可生成 Brugada syndrome 等 **稀有类** ECG (展示了RBBB+持续性ST抬高形态) [Source](https://www.cell.com/patterns/fulltext/S2666-3899(25)00139-4) | ECGTwin继承此能力 |
| **代码/权重** | 截至2026-04-24**尚无官方GitHub/HuggingFace release** for ECGTwin本身；DiffuSETS官方代码+权重可用: [Source](https://github.com/PKUDigitalHealth/DiffuSETS) (MIT License)；**ECGTwin的VAE latent与tokenizer直接复用DiffuSETS的pretrained weights** | 见ECGTwin附录"we use pre-trained VAE model with latent space of R4×128 from DiffuSETS without further parameter-tuning" |
| 许可 | DiffuSETS: MIT (代码) + MIMIC-IV-ECG access required(数据);ECGTwin: 无明示,论文发表于AAAI 2026录用(待确认) | 同上 |

### 表5.2　ECGTwin支持的条件 vs. Challenge 2021 的 26 类

由于ECGTwin的condition是**自由文本报告**, 我们可以通过MIMIC-IV-ECG cart report频次分析其**实际覆盖良好**的类（Figure 5 of ECGTwin 论文：top 30 unique reports含 "abnormal ecg", "normal ecg", "atrial fibrillation", "sinus tachycardia" 等 [Source](https://arxiv.org/html/2508.02720)）。下表为基于cart report语料频率估算的可生成置信度:

| Challenge 2021 class | MIMIC cart report 常见短语 | ECGTwin 生成可信度 | 论文验证? |
|---|---|---|---|
| NSR | "normal sinus rhythm" | 🟢 极高 | ✅ 明确 (Report 2: Normal ECG) |
| AF | "atrial fibrillation" | 🟢 极高 | ✅ top频词 |
| STach | "sinus tachycardia" | 🟢 极高 | ✅ top频词 |
| SB | "sinus bradycardia" | 🟢 高 | ✓ |
| IAVB | "1st degree AV block" | 🟢 高 | ✓ |
| RBBB / CRBBB | "right bundle branch block" | 🟢 高 | ✓ DiffuSETS Brugada案例含RBBB形态 |
| LBBB | "left bundle branch block" | 🟢 高 | ✓ |
| PAC | "premature atrial" | 🟢 高 | ✓ |
| PVC | "premature ventricular" | 🟢 高 | ✓ |
| LAD/RAD | "left/right axis deviation" | 🟡 中 | ○ |
| TAb/TInv | "T wave inversion", "T wave abnormal" | 🟡 中 | ○ |
| QAb | "Q wave abnormal"/"old MI" | 🟡 中 | ✅ Report 3: MI (形态类) |
| AFL | "atrial flutter" | 🟡 中 (较稀疏) | ○ |
| Brady, PR(pacing), LPR, SVPB, VPB, LQRSV, NSIVCB, LQT, LAnFB, IRBBB, SA | 较罕见 cart 短语 | 🟠 低-中; 适合augmentation但需人工质检 | ✗ 未单独验证 |

> **研究空白注**: ECGTwin 未在论文中报告 per-class FID 或 morphological realism 的分层评估; 仅报告整体 FID/Precision/Recall 及 CLIP Score + 下游 AF/MI 任务的 AUROC 提升。因此对于**稀有类的生成保真度是否足够**这一问题，**论文并未直接给出证据**，用户需在自己pipeline中加入 FID-per-class + 专家review (或 surrogate classifier的AUPRC) 验证。

---

## PART 6　最终整合推荐

### 🔴 **主推方案：PhysioNet/CinC Challenge 2021 scored diagnoses (30 SNOMED labels → 4 equivalence pairs merged → 26 independent evaluation classes) 作为分类头**

### 6.1　推荐分类头设计（26 类，已合并等价组）

| 索引 | 评估维度 | SNOMED组 | 期望正样本源 |
|---|---|---|---|
| 1 | IAVB | 270492004 | 三数据集直接 |
| 2 | AF | 164889003 | 三数据集直接 |
| 3 | AFL | 164890007 | 同上 |
| 4 | BBB | 6374002 | 三数据集（regex + SNOMED 匹配）|
| 5 | Brady | 426627000 | 三数据集(MIMIC需HR辅助) |
| 6 | **RBBB(含CRBBB)** | 713427006\|59118001 | 等价对 P1 |
| 7 | IRBBB | 713426002 | 同 |
| 8 | LAnFB | 445118002 | 同 |
| 9 | LAD | 39732003 | 同 |
| 10 | **LBBB(含CLBBB)** | 164909002\|733534002 | 等价对 P4（2021 新增） |
| 11 | LQRSV | 251146004 | 同 |
| 12 | NSIVCB | 698252002 | 同 |
| 13 | PR (pacing) | 10370003 | 同 |
| 14 | **PAC(含SVPB)** | 284470004\|63593006 | 等价对 P2 |
| 15 | **PVC(含VPB)** | 427172004\|17338001 | 等价对 P3 |
| 16 | LPR | 164947007 | 同 |
| 17 | LQT | 111975006 | 同 |
| 18 | PRWP | 365413008 | 主要 PTB-XL + MIMIC（PN2021 稀疏）|
| 19 | QAb | 164917005 | 同 |
| 20 | RAD | 47665007 | 同 |
| 21 | SA | 427393009 | 同 |
| 22 | SB | 426177001 | 同 |
| 23 | NSR | 426783006 | 三数据集原生频高 |
| 24 | STach | 427084000 | 同 |
| 25 | TAb | 164934002 | 同 |
| 26 | TInv | 59931005 | 同 |

### 6.2　推荐 MIMIC-IV-ECG 标注流水线

```
MIMIC-IV-ECG v1.0  ──► machine_measurements.csv (cart report_0..17)
                   │
                   ├──► [Stage A] Regex + ECG-QA 155-SCP词典
                   │       ↳ 输出: ~20/26类 positive hits + confidence=0.9
                   │
                   ├──► [Stage B] PhysioNet 2021 trained classifier
                   │       (建议使用 ECG-FM finetune on PhysioNet 2021 fold 0-17)
                   │       ↳ 输出: 24-dim sigmoid probability
                   │
                   ├──► [Fusion] label = regex_hit OR (prob > 0.85)
                   │
                   └──► [Optional Validation] MIMIC-IV-ECG-Ext-ICD
                           交叉一致性 (仅对 AF/HF/MI/IAVB)
```

**关键资源链接**:
- MIMIC-IV-ECG v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/
- MIMIC-IV-ECG-Ext-ICD v1.0.1 (可选): https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ + GitHub https://github.com/AI4HealthUOL/ECG-MIMIC
- ECG-QA (含155 SCP cart→SNOMED词典): https://github.com/Jwoo5/ecg-qa
- fairseq-signals (PhysioNet 2021 finetune pipeline): https://github.com/Jwoo5/fairseq-signals
- ECG-FM weights (pretrained on MIMIC-IV-ECG + PhysioNet 2021): https://github.com/bowang-lab/ECG-FM + HuggingFace `wanglab/ecg-fm-preprint`

### 6.3　ECGTwin 数据增强目标类 (推荐类 ∩ ECGTwin覆盖良好 ∩ rare)

| 优先级 | 目标类 | 理由 | Challenge 2021 prevalence |
|---|---|---|---|
| P0 | **Brady** (426627000) | 稀有+cart词频中等+心率可精确控制 | ~0.7% |
| P0 | **PR/Pacing** (10370003) | 稀有+cart"paced"短语明确 | ~0.7% |
| P0 | **LPR** (164947007) | 稀有+与IAVB相关可合成 | ~0.8% |
| P1 | **AFL** (164890007) | 边缘稀有+RR规律特征ECGTwin可捕捉 | ~1.0% |
| P1 | **RAD** (47665007) | 边缘稀有 | ~1.1% |
| P1 | **LQRSV** (251146004) | 低振幅形态; ECGTwin VAE latent可control amplitude | ~1.7% |
| P2 | **VPB/PVC** (17338001/427172004) | G3合并后中等; 形态明显 | 中等 |
| P2 | **SVPB** (63593006单独) | G2合并后可忽略; 但若想单独生成则需condition precise | ~0.5% |
| **不建议** | IRBBB, LAnFB, NSIVCB, TInv等 | ECGTwin cart报告词粒度对这些形态类 fidelity 尚未论文验证 | — |

### 6.4　实施优先级 (步骤ranked list)

**步骤1** — 数据底座准备 (1-2周)  
  a. 下载 PTB-XL v1.0.3, PhysioNet/CinC Challenge 2021 v1.0.3 (含CPSC/INCART/PTB/PTB-XL/Georgia/Chapman/Ningbo), MIMIC-IV-ECG v1.0 及 machine_measurements.csv  
  b. 复用 `physionetchallenges/python-example-2024/prepare_ptbxl_data.py` 将 PTB-XL SCP + 12SL映射至 Challenge 2021 30-SNOMED → 24 merged classes (脚本: https://github.com/physionetchallenges/python-example-2024/blob/main/prepare_ptbxl_data.py)  
  c. 用 Challenge 2021 原生 `Dx` header字段直接获取外部训练源标签，应用 `dx_mapping_scored.csv` + `weights.csv` equivalence 逻辑 (https://github.com/physionetchallenges/evaluation-2021)

**步骤2** — MIMIC-IV-ECG 弱标签生成 (2周)  
  a. 实现 Stage A: 基于 ECG-QA 的 155 SCP词典写 regex rules (约20条)，在 `machine_measurements.csv` report_0..17 列进行匹配 → 输出 `mimic_weak_labels_stageA.parquet`  
  b. 在 Challenge 2021 训练集上 finetune ECG-FM (fairseq-signals diagnosis config, 24 head), 得到 `mimic_2021_classifier.ckpt`  
  c. 执行 Stage B inference, fuse with Stage A → `mimic_weak_labels_final.parquet`  
  d. (Optional) 与 MIMIC-IV-ECG-Ext-ICD v1.0.1 对 AF/IAVB 做 Cohen κ 一致性检查 sanity QC

**步骤3** — 分类头训练 (2周)  
  a. Baseline: SE-ResNet1d101 或 xresnet1d101 (Strodthoff 2021), 24-sigmoid head, BCE + pos_weight  
  b. 评估协议: Challenge 2021 official metric(`evaluate_model.py`) + macro AUROC + macro AUPRC  
  c. 验证: PhysioNet 2021 CPSC test fold(patient-stratified) + PTB-XL 官方 fold 9-10 + MIMIC-IV-ECG fold 19(按Strodthoff split)

**步骤4** — ECGTwin 增强 pipeline (3-4周)  
  a. 从 DiffuSETS 仓 (https://github.com/PKUDigitalHealth/DiffuSETS) clone + 加载 pretrained VAE (R^{4×128})  
  b. 复现 ECGTwin 的 Individual Base Extractor + AdaX Condition Injector (尚无官方code，需按论文算法自行实现约300-600行PyTorch；或联系第一作者 yongfanlai@pku.edu.cn 获取内部代码)  
  c. 在 MIMIC-IV-ECG 上训练 ECGTwin (T=1000, β ∈ [8.5e-4, 1.2e-2], base-vector dropout=0.15)  
  d. 针对 P0/P1 rare 类 (Brady, PR, LPR, AFL, RAD, LQRSV) 合成 10k-50k samples per class, 使用 cart-style prompt (e.g., `"Sinus bradycardia with heart rate 45 bpm."`)

**步骤5** — 增强后微调与消融 (1-2周)  
  a. Train set = real + synthetic(P0/P1类)  
  b. Report Challenge metric ΔAUPRC(synthetic vs. baseline) per class  
  c. **必做验证**: per-class FID(real vs. synthetic) + cardiologist spot-check 20 synthetic samples per rare class — 因为ECGTwin 原论文未在这些稀有类上做 per-class 质量验证

**步骤6** — 外部泛化测试  
  a. 跨数据集 leave-one-source-out: train={MIMIC+Chapman+Ningbo+Georgia+CPSC}, test={PTB-XL}；反之亦然  
  b. 报告 macro AUROC/AUPRC per class + Challenge 2021 metric

---

### 6.5　关键代码/数据资源速查表

| 用途 | URL |
|---|---|
| Challenge 2021 evaluation | https://github.com/physionetchallenges/evaluation-2021 |
| Challenge 2021 classifier template | https://github.com/physionetchallenges/python-classifier-2021 |
| PTB-XL → Challenge 2021 mapping | https://github.com/physionetchallenges/python-example-2024/blob/main/prepare_ptbxl_data.py |
| PTB-XL benchmark (Strodthoff) | https://github.com/helme/ecg_ptbxl_benchmarking |
| MIMIC-IV-ECG-Ext-ICD | https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ · https://github.com/AI4HealthUOL/ECG-MIMIC |
| ECG-QA (155 SCP词典) | https://github.com/Jwoo5/ecg-qa |
| fairseq-signals | https://github.com/Jwoo5/fairseq-signals |
| ECG-FM (开源FM) | https://github.com/bowang-lab/ECG-FM |
| DiffuSETS (ECGTwin前驱) | https://github.com/PKUDigitalHealth/DiffuSETS |
| Ribeiro DNN 权重 | https://doi.org/10.5281/zenodo.3625017 |
| Ribeiro CODE-test | https://zenodo.org/records/3765780 |
| SNOMED mapping Kaggle | https://www.kaggle.com/datasets/bjoernjostein/physionet-snomed-mappings |
| MEETI (LLM标注参考) | https://www.nature.com/articles/s41597-026-06796-1 |

---

### 研究限制说明

1. **ECGTwin代码缺位**: 截至2026-04-24在GitHub/HuggingFace无官方release。需自实现或等待作者开源。DiffuSETS代码完备可作为脚手架。
2. **30 scored → 26 effective 精确值**: 官方文档中 2020 为 27 scored rows, 2021 扩展至 30 scored rows。经实事核查 `weights.csv`（2026-04 从 GitHub 直拉），实际 equivalence pair = **4 对**（CRBBB|RBBB, PAC|SVPB, PVC|VPB 自 2020 沿用，CLBBB|LBBB 为 2021 新增），故独立评估维度 = 30 − 4 = **26**。早期笔记里引用的"7 groups → 24 effective"是错误复述，本次更正。
3. **ECGTwin的per-class验证空白**: 论文未给出 per-class FID 或 expert review，因此"ECGTwin cover X类"的判断是基于 MIMIC-IV-ECG cart report 词频的合理推断，**并非论文直接验证**。
4. **MIMIC-IV-ECG-Ext-ICD的适用边界**: ICD-10编码来自住院discharge,时序可能与ECG偏差>24h,对形态类诊断(TAb, LAD等)几乎无参考价值，仅对rhythm类(AF/pacing)和infarction类(MI=I21)有实用性。

— 以上为完整研究输出，step-ranked action list见6.4节。

---

## PART 7　本地三数据集实地核查（2026-04-25）

> 对 `/root/autodl-tmp/ptbxl/`、`/root/autodl-tmp/physionet2021/training/`、`/root/autodl-tmp/MIMIC/` 直接扫描的结果。以实际数据为准，覆盖 PART 1-6 中部分理论性推断。

### 7.1　PTB-XL 类数官方核查

从 `scp_statements.csv` 的 `diagnostic_class` / `diagnostic_subclass` 两列直接统计（`dropna().unique()`）：

| 粒度 | 类数 | 完整列表 |
|---|:---:|---|
| Superclass | **5** | CD, HYP, MI, NORM, STTC |
| Subclass | **23** | AMI, CLBBB, CRBBB, ILBBB, IMI, IRBBB, ISCA, ISCI, ISC_, IVCD, LAFB/LPFB, LAO/LAE, LMI, LVH, NORM, NST_, PMI, RAO/RAE, RVH, SEHYP, STTC, WPW, _AVB |
| 总 SCP 代码 | 71 | — |
| diagnostic=1 SCP | 44 | — |

PTB-XL 21,799 条记录中 21,388 条有 superclass / subclass 标签（**98.1% 覆盖率**）。

**Super 各类样本数**：NORM 9,514 / MI 5,469 / STTC 5,235 / CD 4,898 / HYP 2,649

**Subclass 头部稀有尾**（含样本 < 100 的 4 类）：NORM 9514 / IMI 3271 / AMI 3078 / STTC 2239 / LVH 2132 / … / RVH 126 / RAO/RAE 99 / WPW 79 / ILBBB 77 / SEHYP 29 / **PMI 17**

### 7.2　PN2021 raw data vs scored label 的重大差异

扫 8 个 source 88,253 条 `.hea` 文件：**raw data 里出现 133 个 unique SNOMED codes**，远多于官方 scored 30。

关键发现：**MI 相关 SNOMED 在 raw data 里大量存在，但不在 scored 30 里**。

| SNOMED | 含义 | 数据里数量 | 是否 scored? |
|---|---|---:|:---:|
| 164865005 | Myocardial infarction (general) | 6,144 | ❌ |
| 164873001 | Inferior MI | 4,406 | ❌ |
| 164861001 | Anteroseptal MI | 2,559 | ❌ |
| 164867002 | Old MI | 1,168 | ❌ |
| 164930006 | ST elevation (alt) | 2,276 | ❌ |
| 429622005 | ST depression | 3,645 | ❌ |
| 55827005 | LVH | 5,401 | ❌ |
| 67741000119109 | Left atrial enlargement | 1,299 | ❌ |

**总 MI 相关记录 ≈ 21,463**。若自定义 label set，可在 PN2021 上原生评估 MI AUROC/AUPRC，**不局限于官方 scored 30**。

**Per-source 记录数**：chapman_shaoxing 10,247 / cpsc_2018 6,877 / cpsc_2018_extra 3,453 / georgia 10,344 / **ningbo 34,905** / ptb 516 / **ptb-xl 21,837** / st_petersburg_incart 74 = **88,253 total**

### 7.3　MIMIC-IV-ECG Cart Report Regex 覆盖率

扫描 800,035 条记录的 18 个 report 列：

| 关键词 | 命中数 | % |
|---|---:|---:|
| Normal ECG | 516,011 | 64.5% |
| Sinus rhythm | 461,098 | 57.6% |
| Abnormal ECG | 359,304 | 44.9% |
| Ischemia | 104,553 | 13.1% |
| Sinus bradycardia | 99,249 | 12.4% |
| **Atrial fibrillation** | **81,336** | 10.2% |
| LAD | 80,352 | 10.0% |
| LVH | 73,414 | 9.2% |
| Sinus tachycardia | 69,260 | 8.7% |
| RBBB | 65,901 | 8.2% |
| LAnFB | 48,341 | 6.0% |
| Prolonged QT | 39,533 | 4.9% |
| **Acute MI / STEMI** | **30,632** | 3.8% |
| Pacing | 30,458 | 3.8% |
| LBBB | 29,693 | 3.7% |
| PVC | 26,017 | 3.3% |
| PAC | 21,562 | 2.7% |
| Atrial flutter | 13,817 | 1.7% |
| **MI (any)** | **12,446** | 1.6% |
| IAVB | 7,748 | 1.0% |

**MI 相关合计（去重）≈ 43,078 条**；**AF 合计 ≈ 81,336 条**——数据量都够用。

### 7.4　重要警示：PTB-XL 在 PN2021 中出现两次（数据重复）

- `/root/autodl-tmp/ptbxl/`（21,799 条，SCP 代码格式）
- `/root/autodl-tmp/physionet2021/training/ptb-xl/`（21,837 条，SNOMED 代码格式）

**同一数据集、两种标签格式**。做跨数据集评估时必须避免重复：
- **推荐协议**：PTB-XL 标签用 `/ptbxl/`（SCP 原生），PN2021 测试集 **排除 ptb-xl shard**，仅用 chapman_shaoxing + cpsc_2018 + cpsc_2018_extra + georgia + ningbo + ptb + st_petersburg_incart 共 **66,416 条**做独立跨机构测试。

### 7.5　修正 PART 1-6 的若干推断

| PART 1-6 里的断言 | 实事核查后结论 |
|---|---|
| "PN2021 没有 MI 标签" | ❌ 错，raw data 有 ~21k MI 相关记录，只是不在 scored 30 |
| "PN2021 scored 实际 24 类 / 7 equivalence groups" | ❌ 错，`weights.csv` 列数 = **26**，4 对 equivalence pair |
| "PTB-XL subclass = 24 类" | ❌ 错，官方 `diagnostic_subclass` 唯一非空值 = **23** |
| "MIMIC 需 LLM 标注才能做多类标签" | ⚠️ 非必要，cart report regex 对 MI / AF / LBBB / RBBB 等主要类覆盖率足够 |

### 7.6　基于实地数据的新主推：PTB-XL 5-super（替换 PART 6 的 PN2021 26 类）

在"train PTB-XL → test on PN2021 + MIMIC + 含 MI"的硬约束下，**PTB-XL 5-superclass (NORM/MI/STTC/CD/HYP) 是唯一能在三数据集全部评估全 5 类的方案**：

| 标签 | PTB-XL 来源 | PN2021 来源（排除 ptb-xl shard） | MIMIC 来源 |
|---|---|---|---|
| NORM | 原生 `diagnostic_class=NORM` | SNOMED 426783006 | regex `"normal ecg\|normal sinus"` |
| **MI** | 原生（5,469 条） | 164865005 ∪ 164873001 ∪ 164861001 ∪ 164867002 ∪ 164895002 | regex `"myocardial infarct\|stemi\|ischemi\|infarction"` |
| STTC | 原生（5,235 条） | 164934002 ∪ 59931005 ∪ 429622005 ∪ 164930006 | regex `"st depress\|t wave abnorm\|t wave invers"` |
| CD | 原生（4,898 条） | 270492004 ∪ 164909002 ∪ 59118001 ∪ 713427006 ∪ 713426002 ∪ 445118002 ∪ 698252002 | regex `"bundle branch\|av block\|fascicular"` |
| HYP | 原生（2,649 条） | 55827005 ∪ 67741000119109 | regex `"lvh\|ventricular hypertroph"` |

**每类三数据集合计 ≥ 5 万条**（MI ≈ 59k / STTC ≈ 137k / CD ≈ 178k / HYP ≈ 83k / NORM ≈ 536k），训得动、测得到。

**权威背书**：Wagner et al. 2020 *Scientific Data* 7:154（>1500 引用）；Strodthoff et al. 2021 *JBHI* 提供 xresnet1d101 benchmark macro-AUROC ≈ 0.925 作对齐基线。

### 7.7　核查溯源

- PTB-XL 类数来源：`/root/autodl-tmp/ptbxl/scp_statements.csv` 直读 `diagnostic_class` / `diagnostic_subclass` 字段
- PN2021 scored 类数来源：`https://github.com/physionetchallenges/evaluation-2021/blob/main/weights.csv` 列数直读
- PN2021 raw SNOMED 来源：`/root/autodl-tmp/physionet2021/training/**/*.hea` 扫描 `# Dx:` 行
- MIMIC 关键词来源：`/root/autodl-tmp/MIMIC/machine_measurements.csv` report_0..17 regex 扫描