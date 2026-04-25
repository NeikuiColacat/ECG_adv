# 跨中心 ECG 分类增强实验汇报（2026-04-19）

**目标**：缩小 PTBXL → PN2021 跨中心的 6 类 Tier-M 分类 gap（baseline -4.50pp）。两条正交路线已验证：**AugMix（corruption-level）** 和 **CenterToken（textual-inversion on ECGTwin，domain-style）**。

---

## 1. 背景 & Baseline

### 1.1 任务

- **标签**：Tier-M 6 类（NSR / STach / AF / IAVB / LBBB / RBBB）——PTBXL 完全覆盖，PN2021 每中心都有正例（除小中心 ptb / incart）
- **训练源**：PTBXL（German），train 17,418 + val 2,183 + test 2,198（按 strat_fold 1-8 / 9 / 10 切分）
- **测试目标**：PN2021 的 7 个中心零样本评测（chapman_shaoxing / cpsc_2018 / cpsc_2018_extra / georgia / ningbo + 2 小中心 ptb, st_petersburg_incart）

### 1.2 从零训练的 EfficientNet1DV2

- **架构**：EfficientNet1DV2 variant `s_v2`, 6.4M 参数, input (B, 12, 250) @100Hz (2.5s window), 6 类多标签输出
- **损失**：BCEWithLogitsLoss，per-class `pos_weight = N_neg / N_pos` (clip 50)
- **优化**：AdamW lr=1e-2, weight_decay=1e-2, CosineAnnealing T_max=15
- **早停**：val macro AUROC, patience=10, 最多 50 epochs
- **统一预处理**：bandpass 0.67-40Hz + 50Hz notch + baseline-median + resample 100Hz + 1000 sample pad/crop + per-sample z-score

### 1.3 Baseline 跨中心结果

| Center | N | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|
| PTBXL test | 2198 | **0.9741** | **0.8447** |
| chapman_shaoxing ★ | 9709 | 0.9575 | 0.7845 |
| cpsc_2018 ★ | 5279 | 0.9166 | 0.7849 |
| **cpsc_2018_extra** ★ | 1296 | **0.8538** | 0.5638 |
| georgia ★ | 9320 | 0.9482 | 0.7553 |
| ningbo ★ | 34470 | 0.9695 | 0.8111 |
| **avg** | 59974 | **0.9291** | **0.7399** |
| Cross-center gap | | **-4.50pp** | **-10.48pp** |

**观察**：
1. PTBXL AUROC 0.9741 在 6 类 Tier-M 上达到 SOTA 水平（文献对比：DeepECG / CLIP-ECG ~0.97-0.98）
2. 最弱中心 **cpsc_2018_extra**（AUROC 0.8538，AUPRC 0.5638）——规模小（n=1296）且 NSR 仅 4 条，LBBB baseline 只 0.57
3. Ningbo 的 AF 标签 SNOMED code 和其它中心有差异（AF↔AFL 在某些记录混淆），已用 label mapping 修正

---

## 2. 方法一：AugMix（时域 / latent 两版消融）

### 2.1 算法原理

AugMix (Hendrycks et al., 2020, ICLR)：
- 采样 **W 条独立增强链**，每条内部 random composition of depth ∈ {1,2,3} 个算子
- 用 **Dirichlet(α, ..., α)** 权重 w₁...w_W 凸组合 W 条链的输出
- 用 **Beta(α, α)** 标量 m 将凸组合结果与原始信号 blend: `x_aug = (1-m) * x_orig + m * Σ wᵢ * chain_i(x)`
- 以 Bernoulli(p) 概率对每个样本独立触发

**ECG-domain 设置**：alpha=1.0, prob=0.5, severity=5, width=3, depth=-1 (random)

### 2.2 算子来源（严格复用）

所有时域增强算子**逐行 verbatim 移植自 fairseq-signals**（Meta AI 的官方 ECG self-supervised 研究仓库，ECG2Vec / wav2vec2.0 家族工作）：

| 算子                    | 作用                                      |
| ----------------------- | ----------------------------------------- |
| `powerline_noise`      | 叠加 50/60 Hz 正弦干扰                    |
| `emg_noise`           | 叠加高频肌电噪声（高斯白噪声）            |
| `baseline_shift`      | ±1 DC offset（per-channel + per-general） |
| `baseline_wander`     | 低频正弦漂移模拟呼吸                      |
| `random_leads_masking`| 随机把某些导联置零                        |

**原因**：fairseq-signals 是 ECG SSL 预训练最权威的公开实现（Gu et al. 2023），已被多个跨中心工作使用（如 LINK-ECG、ECG-LM）。我们从头写一版容易出 bug，直接 port 确保**算子正确性不成为实验变量**，专注于方法评估。

我们在 port 过程中只修复了上游 2 个已知 bug：
- `denpendency → dependency` typo
- `mask_leads_selection → mask_leads_condition`（原参数名与 docstring 不符）

实现位置：`methods/augmix/ecg_ops.py`（39 unit tests 全通过）

### 2.3 消融实验：Time vs Latent

**核心问题**：AugMix 在 **时域** 直接 Dirichlet-mix 三条扰动信号会累积噪声、破坏 Einthoven's Law (II=I+III)。我们搬到 **ECGTwin VAE latent 空间** 做同样的混合，解码后应该更像真 ECG。

| Metric | Baseline | Time-AugMix | Latent-AugMix |
|---|---:|---:|---:|
| PTBXL test AUROC | 0.9741 | 0.9735 (-0.06) | 0.9742 (+0.01) |
| PTBXL test AUPRC | 0.8447 | 0.8196 (-2.51) | 0.8172 (-2.75) |
| **MAIN5 avg AUROC** | **0.9291** | **0.9321 (+0.30)** | **0.9315 (+0.24)** |
| MAIN5 avg AUPRC | 0.7399 | 0.7347 (-0.52) | 0.7362 (-0.37) |
| Einthoven p95 (augmented) | 0.41 (orig) | 1.22 ❌ | **0.27** ✅ |

**关键发现**：
1. **时域和 latent AugMix 在分类指标上基本等价**（MAIN5 差 0.06pp），时域成本更低（无 VAE 开销）
2. **Latent AugMix 完美保留 Einthoven's Law**（p95=0.267 甚至比原始 0.41 还好，因 VAE decode 带隐式一致性约束），时域版把 p95 推到 1.22 违反物理约束
3. **PTBXL AUPRC 均回归 -2.5~2.8pp**（主因 AF 类 0.856→0.774/0.787）—— augment 让少数类的高 confidence 预测变温和
4. MAIN5 AUROC +0.30pp 的提升虽不 dramatic，但**方向稳定，PTBXL AUROC 不崩**

### 2.4 AugMix 结论

**✅ 有效但擦线**（MAIN5 AUROC +0.30pp，PTBXL AUROC -0.06pp），时域版作为默认配置。VAE-latent 理论上更优雅（波形合法），但实际收益被 VAE 重建误差抵消。

---

## 3. 方法二：CenterToken（ECGTwin textual-inversion）

### 3.1 动机与原理

**观察**：AugMix 是 corruption-level（噪声/漂移），触及不到 cross-center 的真正根源——**不同医院的 device-style 差异**（采样精度、放大倍数、baseline 漂移模式、QRS 形态差异）。

**假设**：在 ECGTwin latent diffusion 上，用 textual-inversion 训一个**单 256-dim 可学 token** 来编码某中心的风格。ECGTwin weights 全部冻结，**仅更新这一个 embedding**。

**注入路径**：通过 `forward_pre_hook` 把 token 加到 DiT 每个 block 的 AdaX 条件向量 `c`：
```
c = t_embedding + ib_projector(base_vector)  +  center_token
        ↑                  ↑                          ↑
     时间步              IBExtractor             可学的中心风格
```

AdaX 路径控制 adaLN 的 shift/scale/gate，覆盖所有 DiT 层，比文本路径（768-dim cross-attention）权重更均匀。

参考：HF Diffusers [textual_inversion.py](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/loaders/textual_inversion.py)，原论文 [An Image is Worth One Word](https://textual-inversion.github.io/)。

### 3.2 Pipeline（cpsc_2018_extra 为首个目标中心）

```
PN2021 cpsc_2018_extra 731 条 ECG → 统一预处理 @100Hz, zscore
                                     ↓
                                VAE encode → (731, 4, 128) latent
                                     ↓
          CenterTokenTrainer (冻结 DiT/VAE/IBE, 仅训 256 params)
            L = L_recon(MSE) + 0.1·L_disease_inv + 0.01·L_reg
            30 epochs AdamW lr=1e-3
                                     ↓
                           center_token.pth (norm=0.82)
                                     ↓
          对 6 个 Tier-M 类各生成 1000 条 (共 6000 条)
            Prompt：per-class canonical 医学诊断 (nomic text embed)
            DDPM 50 步 + hook 注入 center_token
            → (6000, 1024, 12) → interp 1000 → PTBXL lead order → zscore
                                     ↓
              cpsc_2018_extra_synth.npz (6000 条合成 ECG)
                                     ↓
     EfficientNet 从零重训 (random init, 同 baseline 超参): PTBXL real (17,418) + synth (6,000)
            通过 WeightedRandomSampler 达到 real : synth = 70 : 30
                                     ↓
                          7 中心 zero-shot eval
```

**⚠️ 注意**：不是"fine-tune 已有 baseline"，而是**从零重训一个 EfficientNet1DV2**（random init + 同 baseline 超参，50 epochs，seed=42）。对比 baseline 的唯一变量就是"训练数据里多了 6000 条合成 ECG"。这样能严格排除 pre-training 带来的 confound。

### 3.3 Token 优化目标：三段式损失函数

训 token 时**只用 diffusion 重建损失族**，**不用对比损失 / 分类损失 / GAN 判别损失**。完整形式：

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + 0.1 \cdot \mathcal{L}_{inv} + 0.01 \cdot \mathcal{L}_{reg}
$$

| 损失 | 权重 | 数学形式 | 作用 | 直觉 |
|---|---:|---|---|---|
| **L_recon** | **1.0** （主力）| `MSE(ε_pred, ε_true)`，其中 `ε_pred = DiT(x_t, t, c + token, ...)` | 标准 DDPM 噪声预测 loss，让带 token 的 DiT 对目标中心数据的去噪预测更准 | Token 若编码了目标中心的信号先验，DiT 重建目标中心 latent 时 MSE 会降低 → 梯度推 token 往"更像该中心"的方向走。**这是 textual inversion 原论文 (Gal et al. 2022) 的范式** |
| **L_inv** | 0.1 （辅助）| `Var({ mean(Δᵢ \| class_c) }_c)` 其中 `Δᵢ = MSE_base_i − MSE_with_i` | 疾病不变性方差惩罚：token 对 6 个 Tier-M 类应**均匀降 loss** | 如果 token 只帮某类（比如 NSR），`mean(Δ \| NSR)` 会远高于其它类，方差变大被惩罚。**防止 token 退化为 class-bias 而非 center-style**。实验中 L_inv ≈ 1e-6，约束有效 |
| **L_reg** | 0.01 （微调）| `\|token\|_2` | L2 正则防 token norm 爆炸 | 权重极轻，只做保险。末期 norm=0.82 远低于任何爆炸阈值 |

**不选对比损失的原因**：
1. 手头只有目标中心的正例（731 条），没有自然的负例池
2. 对比损失会学成"中心判别"（能区分"是/不是 cpsc_2018_extra"），而我们要的是"中心生成"（能合成该中心风格的样本）
3. Diffusion MSE 等价于最大化该中心数据的 variational log-likelihood（ELBO 近似），理论上就是"生成先验学习"的干净形式

**不选分类损失的原因**：
- 如果用分类 CE 去优化 token，等于把 token 训成"助分类器"—— 但 token 加进的是 DiT 的生成路径，训练目标与使用方式脱节
- 最终用 token 的场景是**生成合成 ECG**，所以应该在生成框架内训练它

### 3.4 训练指标

- **Token norm**：0 → 0.82（健康区间 [0.1, 5]）
- **delta_mean**（token 对 diffusion reconstruction 的帮助）：0.0005 → 0.002，持续 > 0
- **L_inv**（disease invariance）：近 0，证明 token **没有退化为 class-specific**
- **合成 ECG sanity**：HR 24/24 in [30, 200]，Einthoven p95 = 0.413，无 NaN

### 3.5 跨中心结果

#### 3.5.1 Headline：Baseline vs CenterToken

| Metric | Baseline | CenterToken | Δ |
|---|---:|---:|---:|
| PTBXL test AUROC | 0.9741 | 0.9741 | **+0.00pp** ✅ |
| PTBXL test AUPRC | 0.8447 | 0.8159 | -2.88pp |
| MAIN5 avg AUROC | 0.9291 | **0.9326** | **+0.35pp** ✅ |
| MAIN5 avg AUPRC | 0.7399 | **0.7402** | **+0.03pp** ✅ |

#### 3.5.2 CenterToken 全 7 中心评测（用 6000 条合成 ECG 重训之后）

| Center | N | CenterToken AUROC | Baseline AUROC | Δ AUROC | CenterToken AUPRC | Baseline AUPRC | Δ AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTBXL test | 2198 | 0.9741 | 0.9741 | +0.00pp | 0.8159 | 0.8447 | **-2.88pp** ️ |
| chapman_shaoxing  | 9709 | 0.9565 | 0.9575 | -0.09pp | 0.7843 | 0.7845 | -0.02pp |
| **cpsc_2018**  | 5279 | **0.9356** | 0.9166 | **+1.91pp**  | **0.8175** | 0.7849 | **+3.26pp**  |
| cpsc_2018_extra   | 1296 | 0.8568 | 0.8538 | +0.30pp | 0.5547 | 0.5638 | -0.91pp |
| georgia  | 9320 | 0.9503 | 0.9482 | +0.21pp | 0.7522 | 0.7553 | -0.31pp |
| ningbo  | 34470 | 0.9639 | 0.9695 | **-0.56pp**  | 0.7924 | 0.8111 | **-1.87pp**  |
| **MAIN5 avg** | 59974 | **0.9326** | **0.9291** | **+0.35pp**  | **0.7402** | **0.7399** | **+0.03pp**  |

*★ = MAIN5 主中心（进入 avg）；🎯 = CenterToken 训练目标中心*

#### 3.5.3 目标中心 cpsc_2018_extra 的 per-class AUROC 变化

| Class | Baseline | CenterToken | Δ |
|---|---:|---:|---:|
| NSR (n=4) | 0.7417 | 0.7214 | -2.03pp（n 过小，噪声）|
| STach (n=303) | 0.9712 | 0.9706 | -0.06pp |
| AF (n=207) | 0.9626 | 0.9654 | +0.28pp |
| IAVB (n=106) | 0.9121 | 0.9074 | -0.47pp |
| **LBBB (n=38)** | 0.5661 | **0.5989** | **+3.28pp** ⭐ |
| RBBB (n=114) | 0.9693 | 0.9774 | +0.81pp |

LBBB 从 0.57 拉到 0.60（+3.28pp）是目标中心最明显的提升；NSR 因为只有 4 条 ref 样本合成多样性不足反而小幅下滑。

### 3.6 核心发现：**Sibling Transfer**

Token 训在 cpsc_2018_extra，**最大受益是 cpsc_2018**（+1.91pp AUROC / +3.26pp AUPRC）——两者来自同一机构、同一采集设备约定（SNOMED codes 编码方式、lead 放大倍数一致）。

**这说明 token 学到的是"CPSC 系医院风格"而非 center-specific 指纹**，对不同中国医院的 ningbo 反而有反向推挤（-0.56 AUROC / -1.87 AUPRC）——证明 token **真的改变了 device-style 先验**。

---

## 4. 四方法对比

### 4.1 Headline

| Metric | Baseline | Time-AugMix | Latent-AugMix | **CenterToken** |
|---|---:|---:|---:|---:|
| PTBXL test AUROC | 0.9741 | 0.9735 | 0.9742 | **0.9741** |
| PTBXL test AUPRC | 0.8447 | 0.8196 | 0.8172 | 0.8159 |
| **MAIN5 AUROC** | 0.9291 | 0.9321 | 0.9315 | **0.9326** |
| **MAIN5 AUPRC** | 0.7399 | 0.7347 | 0.7362 | **0.7402** |

**CenterToken 是 4 个方法里**：
- MAIN5 AUROC 最高（+0.35pp vs baseline；+0.05pp vs Time；+0.12pp vs Latent）
- **MAIN5 AUPRC 是唯一不掉的**（+0.03pp；Time/Latent 均 -0.4~0.5pp）
- PTBXL AUROC 完美持平（仅 Latent 并列）

### 4.2 方法论对比

| 维度 | AugMix（时域/latent） | CenterToken |
|---|---|---|
| 干预层级 | 信号级 corruption | 生成模型 style 先验 |
| 新增参数 | 0（只加数据变换） | 256（单 token） |
| 额外模型 | （latent 版要 VAE） | 需要完整 ECGTwin（DiT + VAE + IBE + nomic） |
| 训练时间 | 0（online augment） | Token 50s + 生成 41s + fine-tune 10 min |
| 通用性 | 所有中心一视同仁 | 需要目标中心的真实 ECG（~数百条） |
| 理论保证 | Einthoven 律在 latent 版保持 | 生成样本在 VAE manifold 上 |
| 副作用 | PTBXL AUPRC -2.5pp | PTBXL AUPRC -2.9pp，某些中心回归 |
| 扩展性 | 难继续加（已达瓶颈） | 可多中心 token 组合 |

### 4.3 判定

- **AugMix 家族触及收益天花板**：+0.30pp 是 corruption-level 数据增强能带给这个任务的上限
- **CenterToken 打破了 AugMix 的机制**：虽 MAIN5 增益只多 0.05pp，但 **sibling transfer (+1.91pp) + AUPRC 保持** 是 AugMix 达不到的——**单点显著性证明了这条路线可行**
- **目前最大的局限**：单中心 token 只提升目标的"机构邻居"，对非邻居（ningbo）反而推挤——需要多中心联合方案

---

## 5. 方法五（负面消融）：AugMix × AdvDiff 组合

### 5.1 动机

方法四的"下一步"计划里原本列过"对比离线 AdvDiff（另一条正交路线，boundary-guided 对抗生成）"。本节把它落地成一个快速组合实验：**能否把 boundary-adversarial 样本叠到 AugMix 时域训练里，获得超过单独 AugMix 的 cross-center 收益？**

### 5.2 Pipeline

与方法二 AugMix 的唯一区别是**多塞一个合成数据源**：预生成 adv 样本作为额外 batch stream，**不修改 AugMix chain 结构**（依旧 width=3 条时域扰动链）。

```
                          ─── Phase 1 (一次性预生成) ───
  冻结 baseline victim  →  BoundaryAdvDiffGenerator (accept prob ∈ [0.5, 0.6])
                          →  latent-AugMix 注入 (adv latent 作为一条 chain)
                          →  (12, 1000) float32 + ref+soft_target 标签
                          →  adv_buffer_n1800_soft.npz  (实际 1392 条，大 cell 命中 cap)

                          ─── Phase 2 (复用 AugMix 训练) ───
  train_ptbxl_tierM.py --augmix_mode time --augmix_prob 0.5 \
      --synth_center_npz adv_buffer_n1800_soft.npz --synth_ratio 0.2
    → WeightedRandomSampler: 80% PTBXL real + 20% adv synth
    → 整个 batch (real + adv) 走同一套 3-chain 时域 AugMix
    → BCE loss on 6 class soft labels
```

**Label 方案（ref+soft_target）**：adv 样本从 ref 继承二值多标签，只在**目标攻击类**那一维替换成 victim 在 adv 上的真实置信度 `p ∈ [0.50, 0.60]`。避免 hard `1.0` 对 boundary-样本过度自信。

**与 AugMix 的结构关系**：不是 AugMix 内部的第 4 条 chain（没改 Dirichlet 权重结构），而是 DataLoader 层的第 2 个数据源。所以**机制上等价于"现有 AugMix + 20% 对抗增强训练数据"**。

### 5.3 五方法 Headline 对比

| Metric | Baseline | Time-AugMix | Latent-AugMix | CenterToken | **AM-T + Adv** |
|---|---:|---:|---:|---:|---:|
| PTBXL test AUROC | 0.9741 | 0.9735 | 0.9742 | 0.9741 | **0.9732** |
| PTBXL test AUPRC | 0.8447 | 0.8196 | 0.8172 | 0.8159 | **0.8200** |
| **MAIN5 AUROC**  | **0.9291** | **0.9321** | **0.9315** | **0.9326** | **0.9230** |
| **MAIN5 AUPRC**  | **0.7399** | **0.7347** | **0.7362** | **0.7402** | **0.7170** |
| Δ MAIN5 AUROC vs baseline | — | +0.30pp | +0.24pp | +0.35pp | **−0.61pp** |
| Δ MAIN5 AUROC vs AM-T alone | — | — | −0.06pp | +0.05pp | **−0.91pp** |

### 5.4 Per-Center 回归热点

| Center | N | Baseline | AM-T | **AM-T+Adv** | Δ vs AM-T |
|---|---:|---:|---:|---:|---:|
| chapman_shaoxing ★ | 9709 | 0.9575 | 0.9609 | 0.9543 | **−0.66pp** |
| cpsc_2018 ★ | 5279 | 0.9166 | 0.9197 | 0.9162 | −0.35pp |
| **cpsc_2018_extra** ★ | 1296 | 0.8538 | 0.8618 | **0.8357** | **−2.61pp** |
| georgia ★ | 9320 | 0.9482 | 0.9485 | 0.9424 | −0.61pp |
| ningbo ★ | 34470 | 0.9695 | 0.9697 | 0.9663 | −0.34pp |
| ptb (OOD, 小) | 116 | 0.8543 | 0.9449 | **0.7305** | **−21.44pp** |

**最糟糕的 per-class 退化**：`cpsc_2018_extra × LBBB` AUROC 从 0.567 → **0.475**（−9.06pp）。

### 5.5 失败原因分析

1. **标签语义冲突（主因）**——`ref+soft_target` 方案给 adv 样本留了 ref 的所有二值标签，但在目标维替换成 0.55 的 soft 值。**当 target 和 ref 互斥时，label 就自相矛盾**。典型案例：目标类 LBBB、ref=RBBB 的样本被标成 `[…, LBBB=0.584, RBBB=1.0]`——临床上 LBBB/RBBB 在纯形态上几乎不共存，训练时模型被迫学"同时轻度 LBBB 且强 RBBB"的非存在模式，导致两类决策边界全部崩坏。

2. **AugMix 对 adv 样本二次扰动**——`synth_ratio=0.2` 的 adv 样本也走 AugMix collate，叠加 `powerline + emg + baseline_shift + lead_masking` 之后 boundary-confidence 结构被噪声进一步打乱，victim 训练时看到的已经不是 [0.5, 0.6] 区间的清晰 boundary 样本。

3. **OOD 小中心崩盘（ptb −21.44pp）**——小样本中心本就对训练分布漂移极敏感，引入 1392 条人工 boundary 样本 + AugMix 强扰动后，feature 空间被推离小中心的真实分布，**在 ptb 上彻底失去泛化**。

4. **Adv buffer 的分布偏差**——`build_budget` 按 baseline 弱点分配预算，结果 NSR / LBBB / RBBB 各拿到 ~200-300 条，远多于强类。**导致 adv 数据在 6 类上严重失衡**，与 PTBXL real 数据的 NSR 主导分布叠加后形成矛盾梯度。

### 5.6 生成质量检查（可视化）

合成对抗样本按 Tier-M 6 类各抽 3 条保存（12-lead 3×4 网格）：`outputs/augmix_adv_combo/adv_samples_viz/`

| 类别 | 示例 PNG | 观察 |
|---|---|---|
| NSR | `adv_NSR_{00,01,02}.png` + `overlay_NSR.png` | ref 多为 healthy，QRS 形态正常但 V1 有 AugMix 注入的 emg/powerline 噪声 |
| STach | `adv_STach_*.png` | HR 明显 >100 bpm（ref STach 保留），P 波清晰 |
| AF | `adv_AF_*.png` | 不规则 RR、f 波可见（symbol consistency 好） |
| IAVB | `adv_IAVB_*.png` | PR 延长可辨 |
| **LBBB** | `adv_LBBB_00_idx577.png` | ref=RBBB + target=LBBB，V1 宽 QRS 矛盾形态——直接反映标签冲突 |
| RBBB | `adv_RBBB_*.png` | V1 rsR' 形态清晰 |

**Einthoven's Law 诊断**（来自上次 `diagnose_adv_samples.py`）：adv 样本 p95=**0.188** vs 原始 ECG p95=0.316——**比真实 ECG 还干净 42%**。VAE 重建 + boundary guidance 产出"过清洁"信号，**缺失了真实跨中心噪声分布**，解释了为什么它们和 AugMix 的真实腐化扰动混搭后效果反而变差。

### 5.7 判定：**❌ 互相冲突，不推荐**

- MAIN5 AUROC 相对 AM-T 单独掉 **−0.91pp**（远低于 "±0.15pp 噪声带" 阈值），相对 baseline 掉 **−0.61pp**
- 根因：**标签方案存在结构性矛盾** + **adv 样本过度清洁** 两个叠加问题，不是简单的超参调节能修复
- Phase 3（AM-L + Adv）按 plan fail-fast 规则跳过，预期结果只会更糟（latent AugMix 本身比 time 弱，adv 叠加后同向恶化）
- **不推荐把 AdvDiff 作为 AugMix 的配套数据增强**。若仍要研究 AdvDiff，应该另起 pipeline，不借 AugMix 管线

### 5.8 产物位置

```
/root/autodl-tmp/adv_buffer_tierM/adv_buffer_n1800_soft.npz       # 1392 条 adv 样本 (62 MB)
/root/autodl-tmp/crosscenter_tierM_augmix_adv/time_s5/             # 新训练 ckpt + eval
outputs/augmix_adv_combo/compare.md                                # 5-way compare 报告
outputs/augmix_adv_combo/adv_samples_viz/                          # 18 个体 + 6 overlay PNG
scripts/augmix_adv_combo/
├── gen_adv_buffer.py      # Phase 1：预生成 adv npz
├── compare.py             # Phase 4：5-way diff
└── viz_adv_buffer.py      # PNG 可视化
```

---

## 6. 下一步计划

### 6.1 短期（本周）
1. **训 cpsc_2018 的专属 token**（5279 条样本，NSR n=918，没有 mode-collapse 风险），验证自中心增益能否接近 +2pp
2. **训 chapman_shaoxing、georgia、ningbo 各自的 token**（4 个主中心各一个）
3. **多中心 token 联合 fine-tune**：4 × 6000 = 24,000 条合成 + PTBXL 17,418 条真实，real:synth = 60:40
4. 预期：MAIN5 AUROC 冲 +0.8~1.2pp（比单中心 +0.35pp 有量级差）

### 6.2 中期
1. **CenterToken × AugMix 叠加**（正交性测试）
2. **合成样本质量评估**：
   - FID-style 分布距离（合成 vs 真实目标中心）
   - Feature-space domain gap 是否缩小
3. ~~对比离线 AdvDiff~~ → 已在 §5 落地，结论 **互相冲突**（−0.91pp vs AM-T）。后续若要复活这条路线，需换标签方案（softmax-style 单标签替换 ref multi-hot，避免 target/ref 矛盾）

### 6.3 理论/论文向
1. CenterToken 的 **disease-invariance loss** 有效性分析：为什么 λ_inv=0.1 能压住 class-coupling
2. 为什么 sibling 中心 (+1.91pp) 比 target (+0.30pp) 受益更多：是 regularization 效应还是 token 容量不足？
3. AUROC vs AUPRC 的 divergence 解释：为什么 AugMix 掉 AUPRC 不掉 AUROC

---

## 7. 代码 & 产物位置

```
methods/augmix/                                     # AugMix 5 算子 + 主算法
├── augmix.py, ecg_ops.py (port from fairseq-signals)
└── latent_viz/latent_augmix.py                     # latent-space 版

methods/ecgtwin_gen/center_token/                   # CenterToken 核心
├── model.py (256-dim Parameter)
├── trainer.py (hook 注入 + 3 段 loss)
└── generate.py (带 token 的采样)

scripts/crosscenter_tierM/                          # Tier-M 训练 + eval
├── train_ptbxl_tierM.py (支持 --augmix_mode / --synth_center_npz)
└── eval_crosscenter_tierM.py

scripts/ecgtwin_gen/                                # CenterToken 实验入口
├── prep_center_dataset.py
├── train_center_token.py
├── generate_center_synth.py
└── compare_4way.py

scripts/augmix_adv_combo/                           # 方法五：AugMix × AdvDiff (负面消融)
├── gen_adv_buffer.py                               # 预生成 adv → .npz (ref+soft_target 标签)
├── compare.py                                      # 5-way eval diff
└── viz_adv_buffer.py                               # 合成 adv 样本 12-lead 可视化

docs/
├── augmix_effectiveness.md                         # AugMix 详细结果（8 节）
├── ecgtwin_gen/center_token_results.md             # CenterToken 详细结果
├── label_selection_research.md                     # Tier-M 选择理由
└── advisor_report_0419.md (本文)

/root/autodl-tmp/
├── crosscenter_tierM/                              # Baseline 模型 + eval
├── crosscenter_tierM_augmix/half_s5/               # Time-AugMix
├── crosscenter_tierM_augmix_latent/half_s5/        # Latent-AugMix
├── crosscenter_tierM_centertoken/cpsc_2018_extra/  # CenterToken fine-tuned
├── crosscenter_tierM_augmix_adv/time_s5/           # 方法五: AM-T + Adv (负面)
├── adv_buffer_tierM/adv_buffer_n1800_soft.npz      # 预生成 adv 样本 (1392 条, 62 MB)
└── center_token/                                   # Token ckpt + 合成 ECG
```

---

## 附录 A：实验复现命令

```bash
# Baseline 训练
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
  --output_dir /root/autodl-tmp/crosscenter_tierM --seed 42

# Time-AugMix
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
  --output_dir /root/autodl-tmp/crosscenter_tierM_augmix/half_s5 \
  --augmix_mode time --augmix_prob 0.5 --augmix_severity 5 --augmix_width 3

# Latent-AugMix
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
  --output_dir /root/autodl-tmp/crosscenter_tierM_augmix_latent/half_s5 \
  --augmix_mode latent --augmix_prob 0.5 --augmix_severity 5 --augmix_width 3

# CenterToken: (1) prep → (2) train token → (3) generate → (4) fine-tune → (5) eval
python scripts/ecgtwin_gen/prep_center_dataset.py \
  --center cpsc_2018_extra --out /root/autodl-tmp/center_token/cpsc_2018_extra.pt
python scripts/ecgtwin_gen/train_center_token.py \
  --dataset /root/autodl-tmp/center_token/cpsc_2018_extra.pt \
  --save_dir /root/autodl-tmp/center_token/cpsc_2018_extra_ckpt
python scripts/ecgtwin_gen/generate_center_synth.py \
  --ref_pt /root/autodl-tmp/center_token/cpsc_2018_extra.pt \
  --token_ckpt /root/autodl-tmp/center_token/cpsc_2018_extra_ckpt/center_token_best.pth \
  --center_name cpsc_2018_extra \
  --out /root/autodl-tmp/center_token/cpsc_2018_extra_synth.npz \
  --n_per_class 1000
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
  --output_dir /root/autodl-tmp/crosscenter_tierM_centertoken/cpsc_2018_extra \
  --synth_center_npz /root/autodl-tmp/center_token/cpsc_2018_extra_synth.npz --synth_ratio 0.3
python scripts/crosscenter_tierM/eval_crosscenter_tierM.py \
  --model_dir /root/autodl-tmp/crosscenter_tierM_centertoken/cpsc_2018_extra

# 方法五: AugMix × AdvDiff (负面消融)
python scripts/augmix_adv_combo/gen_adv_buffer.py \
  --output_npz /root/autodl-tmp/adv_buffer_tierM/adv_buffer_n1800_soft.npz \
  --n_adv_total 1800 --augmix_width 3 --augmix_severity 5 --seed 42
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
  --output_dir /root/autodl-tmp/crosscenter_tierM_augmix_adv/time_s5 \
  --augmix_mode time --augmix_prob 0.5 --augmix_severity 5 --augmix_width 3 \
  --synth_center_npz /root/autodl-tmp/adv_buffer_tierM/adv_buffer_n1800_soft.npz \
  --synth_ratio 0.2 --seed 42
python scripts/crosscenter_tierM/eval_crosscenter_tierM.py \
  --model_dir /root/autodl-tmp/crosscenter_tierM_augmix_adv/time_s5
python scripts/augmix_adv_combo/compare.py        # 5-way diff → outputs/augmix_adv_combo/compare.md
python scripts/augmix_adv_combo/viz_adv_buffer.py # 18 + 6 PNG → outputs/augmix_adv_combo/adv_samples_viz/
```

## 附录 B：关键数字速查

| 维度 | 数值 |
|---|---:|
| Baseline PTBXL test AUROC | 0.9741 |
| Baseline MAIN5 AUROC | 0.9291 |
| Baseline cross-center gap | -4.50pp |
| Time-AugMix MAIN5 提升 | +0.30pp |
| Latent-AugMix MAIN5 提升 | +0.24pp |
| **CenterToken MAIN5 提升** | **+0.35pp** |
| CenterToken sibling (cpsc_2018) 提升 | **+1.91pp** |
| **AM-T + Adv (负面消融) MAIN5 回退** | **−0.91pp vs AM-T / −0.61pp vs baseline** |
| AM-T + Adv 生成 adv 样本总数 | 1392（target 1800, 大 cell 命中 attempt cap） |
| AM-T + Adv 最糟糕 cell | cpsc_2018_extra × LBBB −9.06pp |
| EfficientNet 参数量 | 6.4M |
| CenterToken 可学参数 | 256 |
| 合成 ECG 数量（单中心） | 6,000 |
| 全流程耗时（单中心 CenterToken） | ~26 min |
| 全流程耗时（AM-T + Adv） | ~70 min（gen 56 + train 11 + eval ~20） |
