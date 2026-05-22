# 面向 ECG 跨中心 / 跨数据集泛化能力增强的开源方法调研报告

> **版本**：2026-05 截止版
> **筛选规则**：仅保留有公开代码仓库、模型权重或作者明确给出可复现实验代码入口的项目。未能核验官方/作者代码的论文不进入主 baseline 表，只在“剔除/待核验”中说明。
> **目标任务**：12-lead ECG super 5 class 分类：CD / HYP / MI / NORM / STTC。
> **目标实验设定**：PTB-XL 训练 EfficientNet1DV2，在 PhysioNet/CinC 2021 多中心数据集上跨中心测试，例如 Ningbo、Chapman-Shaoxing、CPSC、Georgia 等。
> **主要指标**：macro AUROC、macro AUPRC。

---

## 0. 执行摘要

本报告的核心结论是：如果你们的 **target-center real-anchor latent-hull adversarial training** 能在 PhysioNet/CinC 2021 跨中心测试中稳定带来 **AUROC +2–4 pp、AUPRC +3–6 pp**，并且该提升在 Ningbo / Chapman-Shaoxing / CPSC / Georgia 等多个中心均成立，那么在 ECG 跨中心泛化增强方向上属于**较强且有论文贡献度的结果**。

原因如下：

1. **ECG 专用跨中心泛化方法中，真正开源且可直接跑 AUROC/AUPRC 对比的项目并不多。** 很多 PhysioNet/CinC 参赛方案或 ECG domain adaptation 论文只报告 challenge score、F1、accuracy，或者没有官方代码。
2. **通用 DG / UDA / TTA 方法虽然开源成熟，但大多不是 ECG-specific。** 它们在 ECG 跨中心 macro AUROC / macro AUPRC 上的提升需要在你们协议下重跑；文献中的提升不能直接等价迁移。
3. **ECG foundation model 很重要，但它们通常解决“预训练表征 + 下游微调”，不直接解决 target-center small-K hard sample adaptation。** 因此 ECG-FM、ECGFounder、ST-MEM、MERL 应作为强基线，但不是与你们方法完全同类的竞争方法。
4. **ECG 生成式增强方法大多是类别增强、文本条件生成或隐私友好合成，不是目标中心 small-K anchor-driven adaptation。** 你们的 novelty 在于：使用真实目标中心少量样本作为同标签 anchor，在 ECG VAE / ECGTwin latent manifold 内搜索 hard-but-on-manifold adversarial samples。
5. **最公平的主对比不应堆很多不相关论文，而应围绕协议划分：source-only、target-unlabeled adaptation、target-labeled K-shot adaptation、foundation model fine-tuning、synthetic ECG augmentation、latent mixup / manifold mixup。**

---

## 1. 你们方法的定位

### 1.1 当前方法

你们的方法可以概括为：

```text
Source training:
    PTB-XL -> EfficientNet1DV2

Target small-K adaptation:
    For each target center c and class y:
        collect K labeled real ECG samples from target center
        encode ECG into ECGTwin / ECG VAE latent space
        construct same-label target-center latent hull
        optimize convex coefficients to find hard but on-manifold latent samples
        decode / use latent adversarial samples for online adversarial training

Evaluation:
    target center held-out test set
    macro AUROC / macro AUPRC
```

核心公式：

```math
z_{adv} = (1 - \lambda) z_0 + \lambda \sum_i \mathrm{softmax}(a_i) z_i
```

其中：

- `z0` 是当前样本 latent；
- `zi` 是同标签、目标中心真实样本的 anchor latent；
- `a_i` 是可优化组合权重；
- `lambda` 控制从原样本向目标中心 latent hull 的移动强度；
- 优化目标不是随机插值，而是让样本变成 **hard example**，同时仍保持在目标中心 ECG manifold 附近。

### 1.2 与已有方法的核心差异

| 对比对象 | 已有方法通常做什么 | 你们方法的差异 |
|---|---|---|
| 普通 K-shot fine-tuning | 直接用 K 个目标中心有标签样本微调 | 你们用 K 个目标样本构造 latent hull，并主动搜索 hard-but-on-manifold 样本 |
| Mixup / Manifold Mixup | 随机线性插值输入、标签或隐藏层特征 | 你们限定同标签 + 目标中心 anchor + VAE latent manifold + adversarial optimization |
| DG 方法 | 不使用目标中心数据，学习 domain-invariant 或 domain-robust 表征 | 你们明确使用 target labeled K-shot 数据，属于 target-aware adaptation |
| UDA / TTA | 使用目标中心无标签数据，对齐分布或测试时自适应 | 你们使用少量目标中心有标签数据，并保持类别条件约束 |
| ECG foundation model | 大规模预训练后下游 fine-tune / zero-shot | 你们是一个可叠加在任意 backbone / foundation representation 上的 target-center adversarial augmentation 策略 |
| ECG synthetic generation | 根据类别、文本或患者信息生成 ECG | 你们不是一般生成，而是目标中心真实样本 anchor 驱动的 hard sample search |
| PGD / TRADES | 在输入空间或参数邻域做 worst-case perturbation | 你们在 ECG VAE latent manifold 中做可解释的 target-center constrained adversarial search |

---

## 2. 开源项目总表

> 说明：表中的“提升幅度”只记录论文中明确报告或可合理归纳的信息。若论文没有报告 ECG 跨中心 macro AUROC / macro AUPRC，则标为“未直接报告”，不强行编造数值。

| 类别 | 方法 / 项目 | 论文与 venue | ECG 专门 | 目标中心数据需求 | 输入数据类型 | 论文报告的主要指标 / 提升 | 开源状态 | 是否适合直接对比 |
|---|---|---|---:|---|---|---|---|---|
| ECG FM | ECG-FM | *ECG-FM: An Open Electrocardiogram Foundation Model* | 是 | 下游 fine-tune 可用 target labeled K-shot | 12-lead ECG | 预训练 2.5M samples，评估 ECG interpretation、LVEF、troponin 等；不是直接 PTB-XL→PN2021 five-class 协议 | GitHub: `bowang-lab/ECG-FM` | **高**：foundation model fine-tuning 强基线 |
| ECG FM | ECGFounder | *An Electrocardiogram Foundation Model Built on over 10 Million Recordings...* | 是 | 下游 fine-tune 可用 target labeled K-shot | 12-lead / reduced-lead ECG | 内部验证 80 个诊断 AUROC > 0.95；外部验证与 fine-tuning 均较强；需按你们任务重跑 | GitHub: `bdsp-core/ECGFounder` | **高**：大规模监督 ECG FM 强基线 |
| ECG SSL | ST-MEM | *Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of ECG* | 是 | 不需要 target；下游可 target K-shot fine-tune | 12-lead ECG | arrhythmia classification 多设置优于 SSL baseline；未直接报告你们协议下 AUPRC | GitHub: `bakqui/ST-MEM` | **中高**：SSL pretraining baseline |
| ECG multimodal | MERL | *Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement* | 是 | zero-shot 不需要 target label；fine-tune 可用 target K-shot | ECG + clinical report / prompt | 六个公开 ECG 数据集 zero-shot 平均 AUC 75.2%，比 10% label eSSL linear probe 高 3.2 pp | GitHub: `cheliu-computation/MERL` | **中高**：zero-shot / multimodal baseline |
| ECG SSL | PCLR | *Patient Contrastive Learning of Representations* | 是 | 不需要 target；下游可 target K-shot fine-tune | 12-lead ECG | 3.2M ECG 预训练；少于 5000 labels 的任务有明显收益；需按你们任务重跑 AUROC/AUPRC | GitHub: `broadinstitute/ml4h/model_zoo/PCLR` | **中**：older but useful SSL baseline |
| ECG SSL | CLOCS | *Contrastive Learning of Cardiac Signals Across Space, Time, and Patients* | 是 | 不需要 target；下游可 fine-tune | ECG segments / leads / patients | 25% labeled data 下仍有较强泛化；未直接报告你们协议 AUPRC | GitHub: `danikiyasseh/CLOCS` | **中**：SSL baseline，可放补充 |
| ECG generation | Auto-TTE / Text-to-ECG | *Text-to-ECG: 12-Lead ECG Synthesis conditioned on Clinical Text Reports* | 是 | 不针对 target-center；需要文本条件 | 12-lead ECG + report | 主要评价生成保真度、语义对齐和医生用户研究；不是 small-K adaptation | GitHub: `TClife/text_to_ecg` | **低-中**：synthetic augmentation 补充 |
| Generic DG | MixStyle | ICLR 2021, *Domain Generalization with MixStyle* | 否 | 不需要 target | 任意 CNN feature | 视觉 DG 多任务有效；ECG AUROC/AUPRC 需重跑 | GitHub: `KaiyangZhou/mixstyle-release` | **高**：source-only DG baseline |
| Generic DG | GroupDRO | ICLR 2020, *Distributionally Robust Neural Networks for Group Shifts* | 否 | 不需要 target；需要 source group/center 标签 | source domains / groups | worst-group robustness；ECG AUROC/AUPRC 需重跑 | GitHub: `kohpangwei/group_DRO` | **高**：source-center robust baseline |
| UDA | Deep CORAL | ECCV / CORAL 系列 | 否 | 需要 target unlabeled | source labeled + target unlabeled | 对齐二阶统计；原文为视觉/标准 DA，不是 ECG | GitHub: `VisionLearningGroup/CORAL` | **高**：target-unlabeled baseline |
| UDA | DANN | JMLR 2016, *Domain-Adversarial Training of Neural Networks* | 否 | 需要 target unlabeled | source labeled + target unlabeled | 梯度反转学习 domain-invariant feature；ECG 指标需重跑 | 多个实现；建议用 Transfer-Learning-Library | **中**：可选，不如 CDAN 稳 |
| UDA | CDAN | NeurIPS 2018, *Conditional Adversarial Domain Adaptation* | 否 | 需要 target unlabeled | source labeled + target unlabeled | 条件 adversarial alignment，适合多类条件分布 shift；ECG 指标需重跑 | THUML / Transfer-Learning-Library | **中高**：比 DANN 更建议 |
| Source-free UDA | SHOT / SHOT++ | ICML 2020 系列 | 否 | 需要 target unlabeled；不需要 source data | source model + target unlabeled | source-free UDA；医学隐私场景合理；ECG 指标需重跑 | GitHub: `tim-learn/SHOT-plus` | **高**：source-free target adaptation baseline |
| TTA | TENT | ICLR 2021, *Fully Test-Time Adaptation by Entropy Minimization* | 否 | 测试时 target unlabeled stream | source model + test batch | ImageNet-C 等 corruption 下有效；ECG 指标需重跑 | GitHub: `DequanWang/tent` | **中高**：部署型 TTA baseline |
| TTA | EATA | ICML 2022, *Efficient Test-Time Model Adaptation without Forgetting* | 否 | 测试时 target unlabeled stream | source model + test batch | 相比 TENT 处理遗忘与低效；ECG 指标需重跑 | GitHub: `mr-eggplant/EATA` | **中高**：TTA 表中优先 |
| TTA | CoTTA | CVPR 2022, *Continual Test-Time Domain Adaptation* | 否 | 持续 target unlabeled stream | source model + sequential target stream | 适合连续变化 domain；ECG center 一次性测试不一定需要 | GitHub: `qinenergy/cotta` | **中**：若做连续中心流，可放 |
| Augmentation | Mixup | ICLR 2018, *mixup: Beyond Empirical Risk Minimization* | 否 | 可用于 source 或 target K-shot | input / label interpolation | 原文显示改善泛化、噪声标签、对抗鲁棒性；ECG 跨中心需重跑 | 官方/常用实现很多 | **高**：必须作为 ablation |
| Augmentation | Manifold Mixup | ICML 2019, *Better Representations by Interpolating Hidden States* | 否 | 可用于 source 或 target K-shot | hidden representations | 平滑隐藏层决策边界；ECG 指标需重跑 | 作者实现/常用实现可用 | **高**：与你们 latent-hull 最直接对照 |
| Optimization | SAM | ICLR 2021, *Sharpness-Aware Minimization* | 否 | 不需要 target；也可用于 K-shot FT | model parameters | 多视觉任务泛化提升；ECG 指标需重跑 | GitHub: `google-research/sam` | **中高**：鲁棒优化 ablation |

---

## 3. 最推荐作为论文对比 baseline 的排序

### 3.1 主论文必须放的 baseline

| 优先级 | Baseline | 协议 | 目标中心数据 | 为什么必须放 |
|---:|---|---|---|---|
| 1 | Source-only EfficientNet1DV2 | PTB-XL train → PN2021 centers test | 无 | 所有提升的绝对参照 |
| 2 | Target K-shot plain fine-tuning | source model + target K labeled | 有标签 K | 你们也用 target K labeled，必须证明不只是普通微调 |
| 3 | Target K-shot FT + Mixup | target K labeled | 有标签 K | 检验是否只是输入空间线性插值 |
| 4 | Target K-shot FT + Manifold Mixup | target K labeled | 有标签 K | 检验是否只是 hidden representation interpolation |
| 5 | Target K-shot FT + SAM | target K labeled | 有标签 K | 检验提升是否来自 flatter minima / robust optimization |
| 6 | ECG-FM fine-tuning | same K-shot | 有标签 K | 回答“开放 ECG foundation model 是否已经解决” |
| 7 | ECGFounder fine-tuning | same K-shot | 有标签 K | 回答“大规模 ECG foundation model 是否碾压” |
| 8 | Ours: real-anchor latent-hull AT | same K-shot | 有标签 K | 主方法 |

### 3.2 建议作为补充实验的 baseline

| 优先级 | Baseline | 协议 | 目标中心数据 | 价值 |
|---:|---|---|---|---|
| 9 | MixStyle | source-only DG | 无 | 检验 source-domain style augmentation 是否足够 |
| 10 | GroupDRO | source-center group robust training | 无 | 检验 worst-source-center robustness |
| 11 | Deep CORAL | target-unlabeled UDA | 无标签 target | 低成本 target distribution alignment |
| 12 | SHOT / SHOT++ | source-free target-unlabeled UDA | 无标签 target | 医疗隐私场景合理 |
| 13 | TENT / EATA | test-time adaptation | 测试流无标签 | 部署型 baseline；协议需单独说明 |
| 14 | Auto-TTE synthetic ECG augmentation | synthetic augmentation | 文本条件或标签条件 | 证明普通合成增强不同于 target-anchor hard sample search |
| 15 | ST-MEM / MERL / PCLR / CLOCS | SSL / multimodal representation | 视协议而定 | 可作为 foundation / representation 补充 |

---

## 4. 各类方法的 AUROC / AUPRC 提升预期

> 注意：下面的 pp 范围不是所有论文直接报告的统一结果，而是基于方法类型、已有文献趋势和你们 ECG 跨中心任务的合理预期。最终必须按同一数据切分、同一 EfficientNet1DV2 或同一 backbone 重新跑。

### A. ECG 专门跨中心 / 跨数据集泛化方法

严格意义上的 ECG domain generalization / leave-one-center-out 方法数量少，而且常见问题是：

- 使用 F1、challenge score 或 accuracy，不报告 macro AUROC / macro AUPRC；
- 不公开代码；
- 数据集组合与 PTB-XL → PN2021 center transfer 不一致；
- 不使用 target K-shot 标签数据，因此协议与你们不同。

**预计提升**：

| 方法类型 | AUROC 提升 | AUPRC 提升 | 可靠性 |
|---|---:|---:|---|
| ECG source-only DG | +0.5–2.5 pp | +1–4 pp | 中等，依赖 source domains 是否足够多 |
| ECG multi-center robust training | +1–3 pp | +1–5 pp | 中等，center label 质量重要 |
| ECG feature-level DG | +0–2 pp | +0–3 pp | 任务特定，容易不稳定 |

**结论**：如果你们使用 target K-shot 后稳定拿到 AUROC +2–4 pp、AUPRC +3–6 pp，通常会强于大多数 source-only DG 的可期待收益。

---

### B. 通用 DG / DA / TTA 方法

#### B1. Domain Generalization

| 方法 | 是否需要 target | 适合程度 | 预期 |
|---|---|---|---|
| MixStyle | 否 | 高 | 对 center style shift 有一定解释性；ECG 是 1D signal，需改成 1D feature statistic mixing |
| GroupDRO | 否 | 高 | 如果 source 里有多个 center/domain，可优化 worst-domain performance |
| IRM | 否 | 低-中 | 理论漂亮，但医疗多标签/不平衡任务常难调 |

**预计提升**：AUROC +0.5–2.5 pp；AUPRC +1–4 pp。
**风险**：如果 source domains 太少，DG 方法无法学到足够 domain variation。

#### B2. Unsupervised Domain Adaptation

| 方法 | 是否需要 target unlabeled | 适合程度 | 预期 |
|---|---|---|---|
| Deep CORAL | 是 | 高 | 简单、稳定、低成本；适合作强 baseline |
| DANN | 是 | 中 | 对抗对齐可能损失类别判别性 |
| CDAN | 是 | 中高 | 条件对齐更适合多类别任务 |
| SHOT | 是 | 高 | source-free 医疗隐私场景有说服力 |

**预计提升**：AUROC +0–3 pp；AUPRC +0–5 pp。
**风险**：目标中心类别分布不均衡时，无标签 UDA 可能把少数类对齐坏。

#### B3. Test-Time Adaptation

| 方法 | 是否需要 target labels | 适合程度 | 预期 |
|---|---|---|---|
| TENT | 不需要 | 中高 | 简单，但 batch size 小或类别偏斜时可能 collapse |
| EATA | 不需要 | 中高 | 比 TENT 稳，筛选可靠样本并加 anti-forgetting |
| CoTTA | 不需要 | 中 | 适合连续 drift，不一定适合一次性 center test |

**预计提升**：AUROC +0–2 pp；AUPRC +0–3 pp。
**风险**：TTA 使用测试分布自适应，必须单独说明 protocol，避免被审稿人认为测试集泄漏。

---

### C. ECG foundation model / self-supervised / multimodal pretraining

| 方法 | 主要价值 | 与你们任务关系 | 预期 |
|---|---|---|---|
| ECG-FM | 开放 ECG foundation model，2.5M samples 预训练 | 适合作 K-shot fine-tuning backbone | 若只换 backbone，可能提升 source-only 或 K-shot FT；是否超过你们方法需实测 |
| ECGFounder | 10M+ ECG、150 labels 的大规模 ECG FM | 强 baseline；需确认权重可用性 | 可能是最强 open ECG FM baseline |
| ST-MEM | 12-lead masked ECG modeling | 适合与 EfficientNet1DV2 表征比较 | 在低标注下可能明显优于随机初始化 |
| MERL | ECG-report multimodal zero-shot | 适合作 zero-shot / prompt baseline | 原文六数据集 zero-shot 平均 AUC 75.2%，比 10% label eSSL linear probe 高 3.2 pp |
| PCLR / CLOCS | ECG contrastive pretraining | older SSL baseline | 低标签任务有价值，但可能不如 ECG-FM / ECGFounder |

**预计提升**：

| 设置 | AUROC 提升 | AUPRC 提升 | 可靠性 |
|---|---:|---:|---|
| source-only 使用 FM 表征 | +1–5 pp | +2–8 pp | 中高，但依赖任务匹配 |
| target K-shot fine-tuning | +1–6 pp | +2–10 pp | 高，尤其 K 小时 |
| zero-shot prompt | 不稳定 | 不稳定 | 类别文本能否映射 CD/HYP/MI/NORM/STTC 很关键 |

**结论**：foundation model 是强对手，但如果你们在同一 foundation backbone 上继续叠加 latent-hull AT 仍有 +2–4 / +3–6 pp，贡献度会非常强。

---

### D. ECG 生成式增强 / synthetic ECG augmentation

| 方法 | 主要目标 | 是否解决 target-center small-K adaptation | 适合作什么 baseline |
|---|---|---:|---|
| Auto-TTE / Text-to-ECG | 根据临床文本生成 12-lead ECG | 否 | synthetic augmentation baseline |
| DiffuSETS | 文本/患者信息条件 ECG 生成 | 否或弱相关 | synthetic augmentation baseline，若代码和权重可用 |
| ECG-GAN / TimeGAN / VAE | 类别或无条件 ECG 生成 | 通常否 | 低优先级补充 |
| latent interpolation | latent 空间插值增强 | 部分相关 | 你们必须做 ablation |
| same-label random latent hull | 同标签目标中心随机 hull | 部分相关 | 最关键消融之一 |
| adversarial latent hull | 同标签目标中心 + hard sample search | 是 | 你们主方法 |

**预计提升**：

| 类型 | AUROC 提升 | AUPRC 提升 | 可靠性 |
|---|---:|---:|---|
| 普通 synthetic augmentation | -1–3 pp | -2–5 pp | 不稳定，可能伤害真实分布泛化 |
| same-label latent interpolation | +0.5–2 pp | +1–3 pp | 中等 |
| target-anchor latent adversarial augmentation | +2–4 pp | +3–6 pp | 若多中心稳定，则很有竞争力 |

**结论**：你们应重点证明“不是因为用了生成模型”，而是因为 **目标中心真实 anchor + 同标签 latent hull + adversarial hard search**。

---

### E. 鲁棒训练 / 对抗训练 / 数据增强

| 方法 | 是否建议放 | 原因 |
|---|---:|---|
| PGD adversarial training | 可放小消融 | 输入空间扰动未必符合 ECG 生理 manifold，可能损害波形 |
| TRADES / MART | 可选 | 实现成本高，且 ECG 跨中心收益不一定强 |
| SAM | 建议放 | 与泛化能力直接相关，开源成熟 |
| SWA | 可选 | 简单、低成本，但 novelty 对照不如 SAM |
| Mixup | 必须放 | 你们方法最直接对照之一 |
| Manifold Mixup | 必须放 | 与 latent-hull 的差异需要实验证明 |
| RandAugment / AugMix | 可选 | ECG 增强操作需谨慎定义，不能直接套图像增强 |
| ECG corruption training | 可放补充 | 更偏 robustness，不等同 center shift |

**预计提升**：

| 方法 | AUROC 提升 | AUPRC 提升 | 备注 |
|---|---:|---:|---|
| Mixup | +0–2 pp | +0–3 pp | 小 K 下可能有帮助，但类别不平衡时可能稀释少数类 |
| Manifold Mixup | +0–2.5 pp | +0–4 pp | 是你们最需要打赢的消融 |
| SAM | +0.5–2 pp | +0.5–3 pp | 稳定但不一定解决 center style shift |
| PGD / TRADES | -1–2 pp | -2–3 pp | 如果 perturbation 不生理，可能负收益 |

---

## 5. 最公平的实验设计

### 5.1 数据划分

建议按中心独立报告：

```text
Source train:
    PTB-XL train folds

Source validation:
    PTB-XL validation fold

Target centers:
    Ningbo
    Chapman-Shaoxing
    CPSC
    Georgia
    optionally PTB / INCART / other PN2021 subsets

For each target center:
    labeled adaptation set: K = 100 / 200 / 300 / 400 / 500
    held-out target test set: never used for tuning
```

关键要求：

1. K-shot 样本必须按类别尽量 stratified sampling；
2. 每个 K 至少跑 3–5 个 random seeds；
3. 报告 mean ± std；
4. 所有方法使用相同 K-shot 样本；
5. 目标中心 test set 不参与 early stopping；
6. 对 TTA / UDA 单独声明是否使用 target test unlabeled stream。

---

### 5.2 Baseline 组别

#### Group 1: Source-only baseline

| 方法 | 训练 | 目标数据 |
|---|---|---|
| EfficientNet1DV2 source-only | PTB-XL only | 无 |
| EfficientNet1DV2 + stronger augmentation | PTB-XL only | 无 |
| MixStyle | PTB-XL / source centers | 无 |
| GroupDRO | PTB-XL source groups | 无 |

#### Group 2: Target-unlabeled adaptation baseline

| 方法 | 训练 | 目标数据 |
|---|---|---|
| Deep CORAL | source labeled + target unlabeled | target unlabeled |
| CDAN | source labeled + target unlabeled | target unlabeled |
| SHOT | source model + target unlabeled | target unlabeled |
| TENT / EATA | source model + test stream | target test unlabeled stream |

#### Group 3: Target-labeled K-shot adaptation baseline

| 方法 | 训练 | 目标数据 |
|---|---|---|
| K-shot fine-tuning | source model + target K labeled | K labeled |
| K-shot FT + class-balanced sampling | source model + target K labeled | K labeled |
| K-shot FT + Mixup | source model + target K labeled | K labeled |
| K-shot FT + Manifold Mixup | source model + target K labeled | K labeled |
| K-shot FT + SAM | source model + target K labeled | K labeled |
| Ours | source model + target K labeled + latent-hull AT | K labeled |

#### Group 4: Foundation model baseline

| 方法 | 训练 | 目标数据 |
|---|---|---|
| ECG-FM linear probe | frozen ECG-FM + target K labels | K labeled |
| ECG-FM fine-tuning | ECG-FM + target K labels | K labeled |
| ECGFounder linear probe | frozen ECGFounder + target K labels | K labeled |
| ECGFounder fine-tuning | ECGFounder + target K labels | K labeled |
| ST-MEM fine-tuning | ST-MEM + target K labels | K labeled |
| MERL zero-shot / prompt | prompt-based | 0 label or K label |

#### Group 5: Synthetic ECG augmentation baseline

| 方法 | 训练 | 目标数据 |
|---|---|---|
| Auto-TTE generated ECG augmentation | synthetic ECG + target K | text/report condition required |
| Random VAE latent interpolation | same-label source/target latent interpolation | K labeled |
| Target same-label latent hull without adversarial optimization | random convex hull | K labeled |
| Target same-label latent hull with adversarial optimization | your method | K labeled |

---

## 6. 你们方法必须做的 ablation

| Ablation | 目的 | 预期解释 |
|---|---|---|
| w/o target anchor | 只用 source latent 或随机 latent | 证明目标中心真实 anchor 重要 |
| w/o same-label constraint | 跨标签混合 | 证明类别条件 hull 重要 |
| w/o adversarial optimization | 随机 convex combination | 证明 hard sample search 重要 |
| input-space Mixup | 普通输入插值 | 证明不是普通 mixup |
| feature-space Manifold Mixup | hidden feature 插值 | 证明不是普通 hidden mixup |
| latent interpolation but no decoder | 直接 latent feature training | 区分 decoded waveform vs latent-level training |
| center token only | soft prompt center conditioning | 证明真实 target-anchor 更稳定 |
| different K | K=100/200/300/400/500 | 证明 sample efficiency |
| different lambda | λ sweep | 证明 hull movement 强度影响 |
| different hull size | number of anchors sweep | 证明 anchor coverage 影响 |

---

## 7. 推荐的主结果表格式

### 7.1 Center-wise macro AUROC

| Method | Target data | Ningbo | Chapman-Shaoxing | CPSC | Georgia | Avg. |
|---|---|---:|---:|---:|---:|---:|
| Source-only | none |  |  |  |  |  |
| K-shot FT | K labels |  |  |  |  |  |
| K-shot FT + Mixup | K labels |  |  |  |  |  |
| K-shot FT + Manifold Mixup | K labels |  |  |  |  |  |
| K-shot FT + SAM | K labels |  |  |  |  |  |
| ECG-FM FT | K labels |  |  |  |  |  |
| ECGFounder FT | K labels |  |  |  |  |  |
| Ours | K labels |  |  |  |  |  |
| Δ over K-shot FT | - |  |  |  |  |  |

### 7.2 Center-wise macro AUPRC

| Method | Target data | Ningbo | Chapman-Shaoxing | CPSC | Georgia | Avg. |
|---|---|---:|---:|---:|---:|---:|
| Source-only | none |  |  |  |  |  |
| K-shot FT | K labels |  |  |  |  |  |
| K-shot FT + Mixup | K labels |  |  |  |  |  |
| K-shot FT + Manifold Mixup | K labels |  |  |  |  |  |
| K-shot FT + SAM | K labels |  |  |  |  |  |
| ECG-FM FT | K labels |  |  |  |  |  |
| ECGFounder FT | K labels |  |  |  |  |  |
| Ours | K labels |  |  |  |  |  |
| Δ over K-shot FT | - |  |  |  |  |  |

### 7.3 K-shot scaling table

| K | K-shot FT AUROC | Mixup AUROC | Manifold Mixup AUROC | SAM AUROC | Ours AUROC | Ours Δ | K-shot FT AUPRC | Ours AUPRC | Ours Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 |  |  |  |  |  |  |  |  |  |
| 200 |  |  |  |  |  |  |  |  |  |
| 300 |  |  |  |  |  |  |  |  |  |
| 400 |  |  |  |  |  |  |  |  |  |
| 500 |  |  |  |  |  |  |  |  |  |

---

## 8. 推荐写进论文的方法贡献表述

可以这样写：

```text
Existing ECG foundation models and self-supervised methods mainly improve transferability by learning generic ECG representations from large-scale unlabeled or weakly labeled corpora. In contrast, our method addresses a different but clinically important setting: target-center small-K adaptation. Given a small number of labeled ECGs from a new center, we construct class-conditioned real-anchor convex hulls in an ECG generative latent space and adversarially search hard but target-manifold-preserving samples. This makes the adaptation target-aware, class-preserving, and physiologically constrained, unlike input-space adversarial training, generic mixup, or unconditional synthetic ECG augmentation.
```

中文表述：

```text
现有 ECG foundation model 与自监督预训练方法主要通过大规模预训练提升通用 ECG 表征能力，而本文关注的是更具体的目标中心 small-K adaptation 场景。给定新中心少量有标签 ECG 样本，我们在 ECG 生成式 latent space 中构造类别条件的真实目标中心 convex hull，并在该 hull 内对抗性搜索“更难分类但仍位于目标中心流形附近”的样本。相比普通输入空间对抗训练、Mixup、Manifold Mixup 或无目标中心约束的合成 ECG 增强，本文方法同时具备 target-aware、class-preserving 和 manifold-constrained 三个特征。
```

---

## 9. 剔除 / 待核验项目

以下方法与主题相关，但不建议放入“开源 baseline 主表”：

| 方法 / 论文 | 处理 | 原因 |
|---|---|---|
| 若干 PhysioNet/CinC 2020/2021 adversarial feature learning 参赛方案 | related work 里提，不做主 baseline | 多数只报告 challenge score / F1，且官方代码不稳定或未核验 |
| BioDG | 暂不放主 baseline，除非你能拿到官方 repo | 论文声称 open-source ECG/EEG DG benchmark，但本轮未稳定定位到官方仓库 URL |
| IRM | 可选，不建议主表 | 理论方法，医疗多标签不平衡任务难调，且不是最直接对手 |
| DSBN | 可选，不建议主表 | 需要 domain-specific BN 和 domain id，协议容易不公平 |
| DiffECG / SSSD-ECG / BioDiffusion | related work 或补充 | 与 target-center small-K adaptation 不完全同类，且部分代码/权重可用性需核验 |
| ECGTwin 原始方法 | 作为你们方法组件，不作为外部 baseline | 你们已经使用 ECGTwin / VAE latent space，作为外部 baseline 容易混淆贡献 |
| PGD / TRADES / MART | 小消融即可 | 不是 ECG cross-center 专门方法，且输入空间扰动可能不符合生理 manifold |

---

## 10. 最终结论

如果你们在 PN2021 多中心测试上得到如下稳定结果：

```text
AUROC: +2–4 pp over K-shot fine-tuning / Mixup / Manifold Mixup
AUPRC: +3–6 pp over K-shot fine-tuning / Mixup / Manifold Mixup
```

并且满足：

1. 多中心平均有效；
2. 各中心大多数为正收益，而不是单一中心拉高；
3. K=100/200/300/400/500 均有趋势；
4. 对 macro AUPRC 的提升明显，因为 AUPRC 更敏感于类别不平衡；
5. 打赢 K-shot FT、Mixup、Manifold Mixup、SAM、ECG-FM FT / ECGFounder FT 中至少一部分强基线；
6. ablation 证明 target anchor、same-label hull、adversarial optimization 都有独立贡献；

那么该结果在 ECG 跨中心泛化增强方向上可以评价为：

> **中高强度贡献，足以支撑一篇以方法为主的 ECG domain adaptation / robust adaptation 论文。**

最强的论文叙事不是“我们生成了 ECG”，而是：

> **我们提出了一种 target-center real-anchor, class-conditioned, manifold-constrained adversarial adaptation 方法，专门解决 ECG 跨中心 small-K 泛化问题。**

---

## 11. 参考文献与开源链接

1. McKeen et al. *ECG-FM: An Open Electrocardiogram Foundation Model*. arXiv:2408.05178. GitHub: `https://github.com/bowang-lab/ECG-FM/`
2. Li et al. *An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains*. arXiv:2410.04133. GitHub: `https://github.com/bdsp-core/ECGFounder`
3. Na et al. *Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram*. arXiv:2402.09450. GitHub: `https://github.com/bakqui/ST-MEM`
4. Liu et al. *Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement*. arXiv:2403.06659. GitHub: `https://github.com/cheliu-computation/MERL`
5. Diamant et al. *Patient Contrastive Learning: a Performant, Expressive, and Practical Approach to ECG Modeling*. arXiv:2104.04569. GitHub: `https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR`
6. Kiyasseh et al. *CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients*. arXiv:2005.13249. GitHub: `https://github.com/danikiyasseh/CLOCS`
7. Chung et al. *Text-to-ECG: 12-Lead Electrocardiogram Synthesis conditioned on Clinical Text Reports*. arXiv:2303.09395. GitHub: `https://github.com/TClife/text_to_ecg`
8. Zhou et al. *Domain Generalization with MixStyle*. ICLR 2021. GitHub: `https://github.com/KaiyangZhou/mixstyle-release`
9. Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*. ICLR 2020. GitHub: `https://github.com/kohpangwei/group_DRO`
10. Sun et al. *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. ECCV Workshops / arXiv. GitHub: `https://github.com/VisionLearningGroup/CORAL`
11. Ganin et al. *Domain-Adversarial Training of Neural Networks*. JMLR 2016. arXiv:1505.07818.
12. Long et al. *Conditional Adversarial Domain Adaptation*. NeurIPS 2018. arXiv:1705.10667.
13. Liang et al. *Source Data-absent Unsupervised Domain Adaptation through Hypothesis Transfer and Labeling Transfer*. ICML 2020. GitHub: `https://github.com/tim-learn/SHOT-plus`
14. Wang et al. *Tent: Fully Test-Time Adaptation by Entropy Minimization*. ICLR 2021. GitHub: `https://github.com/DequanWang/tent`
15. Niu et al. *Efficient Test-Time Model Adaptation without Forgetting*. ICML 2022. GitHub: `https://github.com/mr-eggplant/EATA`
16. Wang et al. *Continual Test-Time Domain Adaptation*. CVPR 2022. GitHub: `https://github.com/qinenergy/cotta`
17. Zhang et al. *mixup: Beyond Empirical Risk Minimization*. ICLR 2018. arXiv:1710.09412.
18. Verma et al. *Manifold Mixup: Better Representations by Interpolating Hidden States*. ICML 2019. arXiv:1806.05236.
19. Foret et al. *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021. GitHub: `https://github.com/google-research/sam`
20. Strodthoff et al. *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*. arXiv:2004.13701.
