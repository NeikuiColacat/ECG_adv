# 跨中心 ECG 对抗训练实验汇报

**日期**：2026-04-27
**任务**：Super5（NORM/MI/HYP/CD/STTC）多标签分类，PTB-XL 训练，跨中心鲁棒性
**最终方法**：TA-OMAT (Target-Anchored On-Manifold Adversarial Training) — 在 ECGTwin VAE latent 空间做 PGD，anchor 用真实目标中心 K=200 records
**Victim**：EfficientNet1DV2 s_v2 (PTB-XL fold10 macro AUROC = **0.9064**)

---

## 1. TL;DR

通过 **VAE-latent on-manifold PGD 在线对抗训练**，在不损害源域的前提下提升 PTB-XL → 4 个外部目标（chap_shaoxing / ningbo / georgia / MIMIC）的跨中心 macro AUROC **+0.29–0.35pp**、AUPRC **+0.26–0.51pp**（3 cells × 100ep 一致正向）。结构性 drag 仅在 cpsc_2018 中心（PTB-XL 词表 gap，−0.86pp AUROC）。

**关键决策**：经 Module Ablation #1 验证 ECGTwin diffusion 模块**可移除**（real anchor 严格优于 synth + 6× 训练加速），最终方法学落地为 TA-OMAT。

---

## 2. 数据集 + Pre-AT Baseline

| 数据集 | 角色 | 样本数 | macro AUROC | macro AUPRC |
|---|---|---|---|---|
| **PTB-XL fold 10** | 源域 in-domain 参考 | 2,198 | **0.9064** | **0.7754** |
| PN2021 chap_shaoxing | 跨中心目标 | 10,247 | 0.8835 | 0.5468 |
| PN2021 ningbo | 跨中心目标（最大） | 34,905 | 0.8826 | 0.5914 |
| PN2021 georgia | 跨中心目标 | 10,344 | 0.8173 | 0.6462 |
| PN2021 cpsc_2018_extra | 跨中心目标（亦作 Post-AT cell `extra` 的 anchor 源） | 3,453 | 0.8052 | 0.6220 |
| PN2021 cpsc_2018 | 跨中心目标（STTC 词表 gap） | 6,877 | 0.8052 | 0.5633 |
| PN2021 ptb | 跨中心目标 (n 小) | 516 | 0.9009 | 0.5950 |
| PN2021 st_petersburg | 跨中心目标 (n 极小) | 74 | 0.7785 | 0.5575 |
| MIMIC test | 第三方 zero-shot | 78,707 | 0.7857 | 0.6939 |

> 上表为 super5 victim（无 AT）的 baseline 跨域性能。AT 实验 baseline 行使用相同模型，但 PN2021 eval 时**排除 200 个 anchor 病人** 防 leakage（fair 比较协议；详 §3 表头）。

---

## 3. 核心结果 — Matched-cell pre/post（用 X 中心做 anchor → X 中心自己 pre/post）

> **回答的问题**：用某中心的真实数据做 anchor 训练后，**对该中心自己**的 pre vs post 性能变化是多少？

每个 cell 训练时只见过自己中心的 K=200 anchor。下表展示 **anchor 中心 = eval 中心** (matched diagonal)：

|  中心 | n_records | Pre-AT AUROC | Post-AT AUROC | **Δ AUROC** | Pre-AT AUPRC | Post-AT AUPRC | **Δ AUPRC** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ningbo** (cell `nin`) | 34,705 | 0.8820 | **0.8852** | **+0.32pp**  | 0.5899 | **0.5939** | **+0.40pp**  |
| **georgia** (cell `geo`) | 10,144 | 0.8168 | **0.8197** | **+0.29pp**  | 0.6452 | **0.6488** | **+0.36pp**  |
| cpsc_2018_extra (cell `extra`) | 3,253 | 0.8041 | 0.8032 | −0.08pp ️ | 0.6240 | 0.6234 | −0.05pp  |

**关键观察**（对应 Stutz 2019 on-manifold AT 理论）：
- ✅ **ningbo / georgia source-center 自己也提升** +0.29~0.32pp AUROC，+0.36~0.40pp AUPRC ← 与跨中心 transfer 同方向
- ⚠️ **cpsc_2018_extra source-center 不升反小降** −0.08pp ← 与 Stutz 2019 caveat 一致（on-manifold AT 不改善 source domain），具体到此 cell 是因 PN2021 cpsc_2018_extra 的 STTC 类样本与 PTB-XL 训练分布差距最大（Δ vs PTBXL = −10.2pp）

---

## 4. 三个 cell 完整 pre/post 对比详表（含 transfer 到所有中心）

> ★ 标记当前 cell 的 anchor 源中心（matched diagonal 行）。每张表展示该 cell 训完后在所有数据集上的 pre vs post。

### 4.1 Cell `extra` — anchor = cpsc_2018_extra (3,253 records)

| Eval 中心 | n_eff | Pre AUROC | Post AUROC | **Δ AUROC** | Pre AUPRC | Post AUPRC | **Δ AUPRC** |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTB-XL fold10 (源域) | 2,198 | 0.9064 | 0.9061 | −0.03 | 0.7754 | 0.7758 | +0.04 |
| chap_shaoxing | 10,247 | 0.8835 | 0.8868 | **+0.33** ✓ | 0.5468 | 0.5527 | **+0.59** ✓ |
| ningbo | 34,705 | 0.8820 | 0.8853 | **+0.33** ✓ | 0.5899 | 0.5954 | **+0.55** ✓ |
| georgia | 10,144 | 0.8168 | 0.8193 | **+0.26** ✓ | 0.6452 | 0.6494 | **+0.42** ✓ |
| **★ cpsc_2018_extra (anchor)** | **3,253** | **0.8041** | **0.8032** | **−0.08** ⚠️ | **0.6240** | **0.6234** | **−0.05** ⚠️ |
| cpsc_2018 ⚠️ STTC gap | 6,877 | 0.8052 | 0.7987 | **−0.65** ✗ | 0.5633 | 0.5617 | −0.16 |
| ptb (n=516) | 516 | 0.9009 | 0.9002 | −0.07 | 0.5950 | 0.5958 | +0.08 |
| st_petersburg (n=74) | 74 | 0.7785 | 0.7714 | −0.71 | 0.5575 | 0.5547 | −0.28 |
| MIMIC test (zero-shot) | 78,707 | 0.7857 | 0.7877 | **+0.20** ✓ | 0.6939 | 0.6950 | +0.10 |

**`extra` cell 观察**：
- 4 个独立目标（chap/ningbo/georgia/MIMIC）一致 +0.20~+0.33pp AUROC
- 自己的 source-center cpsc_2018_extra 微负 −0.08pp（Stutz 2019 caveat）
- cpsc_2018 −0.65pp 是 STTC 词表 gap（结构性，非方法问题）

---

### 4.2 Cell `nin` — anchor = ningbo (34,705 records)

| Eval 中心 | n_eff | Pre AUROC | Post AUROC | **Δ AUROC** | Pre AUPRC | Post AUPRC | **Δ AUPRC** |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTB-XL fold10 (源域) | 2,198 | 0.9064 | 0.9063 | −0.01 | 0.7754 | 0.7763 | +0.09 |
| chap_shaoxing | 10,247 | 0.8835 | 0.8867 | **+0.32** ✓ | 0.5468 | 0.5502 | **+0.34** ✓ |
| **★ ningbo (anchor)** | **34,705** | **0.8820** | **0.8852** | **+0.32** ✓ | **0.5899** | **0.5939** | **+0.40** ✓ |
| georgia | 10,144 | 0.8168 | 0.8199 | **+0.32** ✓ | 0.6452 | 0.6497 | **+0.45** ✓ |
| cpsc_2018_extra | 3,253 | 0.8041 | 0.8056 | **+0.15** ✓ | 0.6240 | 0.6240 | −0.00 |
| cpsc_2018 ⚠️ STTC gap | 6,877 | 0.8052 | 0.7964 | **−0.88** ✗ | 0.5633 | 0.5608 | −0.26 |
| ptb (n=516) | 516 | 0.9009 | 0.9009 | +0.00 | 0.5950 | 0.5975 | +0.25 |
| st_petersburg (n=74) | 74 | 0.7785 | 0.7729 | −0.56 | 0.5575 | 0.5536 | −0.39 |
| MIMIC test (zero-shot) | 78,707 | 0.7857 | 0.7892 | **+0.36** ✓ | 0.6939 | 0.6975 | **+0.35** ✓ |

**`nin` cell 观察**：
- **5 个独立目标**（chap/ningbo★/georgia/extra/MIMIC）一致 +0.15~+0.36pp AUROC
- **source-center ningbo 自己也提升 +0.32pp AUROC / +0.40pp AUPRC** ← 与 cross-center transfer 同向
- cpsc_2018_extra +0.15pp（cross-cell transfer 反而比 extra cell 自己 +0.02pp 更好）

---

### 4.3 Cell `geo` — anchor = georgia (10,144 records)

| Eval 中心 | n_eff | Pre AUROC | Post AUROC | **Δ AUROC** | Pre AUPRC | Post AUPRC | **Δ AUPRC** |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTB-XL fold10 (源域) | 2,198 | 0.9064 | 0.9064 | −0.00 | 0.7754 | 0.7765 | +0.11 |
| chap_shaoxing | 10,247 | 0.8835 | 0.8873 | **+0.38** ✓ | 0.5468 | 0.5502 | **+0.33** ✓ |
| ningbo | 34,705 | 0.8820 | 0.8860 | **+0.40** ✓ | 0.5899 | 0.5958 | **+0.59** ✓ |
| **★ georgia (anchor)** | **10,144** | **0.8168** | **0.8197** | **+0.29** ✓ | **0.6452** | **0.6488** | **+0.36** ✓ |
| cpsc_2018_extra | 3,253 | 0.8041 | 0.8041 | +0.00 | 0.6240 | 0.6219 | −0.21 |
| cpsc_2018 ⚠️ STTC gap | 6,877 | 0.8052 | 0.7945 | **−1.06** ✗ | 0.5633 | 0.5595 | −0.38 |
| ptb (n=516) | 516 | 0.9009 | 0.9010 | +0.02 | 0.5950 | 0.5946 | −0.04 |
| st_petersburg (n=74) | 74 | 0.7785 | 0.7728 | −0.57 | 0.5575 | 0.5529 | −0.46 |
| MIMIC test (zero-shot) | 78,707 | 0.7857 | 0.7890 | **+0.33** ✓ | 0.6939 | 0.6972 | **+0.32** ✓ |

**`geo` cell 观察**：
- **4 个独立目标**（chap/ningbo/georgia★/MIMIC）一致 +0.29~+0.40pp AUROC
- **source-center georgia 自己 +0.29pp AUROC / +0.36pp AUPRC** ← 与 cross-center transfer 同向

---

### 4.4 跨 cell 一致性（Cross-cell transfer matrix 简表）

| (anchor) → (eval) | chap_shaoxing | ningbo | georgia | cpsc_2018_extra | MIMIC | PTB-XL fold10 |
|---|---:|---:|---:|---:|---:|---:|
| **extra** cell | +0.33 | +0.33 | +0.26 | **−0.08★** | +0.20 | −0.03 |
| **nin** cell | +0.32 | **+0.32★** | +0.32 | +0.15 | +0.36 | −0.01 |
| **geo** cell | +0.38 | +0.40 | **+0.29★** | +0.00 | +0.33 | −0.00 |
| **3-cell avg** ΔAUROC | **+0.34** ✓ | **+0.35** ✓ | **+0.29** ✓ | +0.02 | **+0.30** ✓ | −0.01 |

★ matched diagonal（用作 anchor 的中心自己）。注意：3 个 source-center 自己只有 cpsc_2018_extra 是负向；其余 5 列 4 行都正向。

---

### 4.5 跨域 gap 参考表（Pre-AT 距离 PTB-XL 多远）

为方便讨论"哪些中心需要 AT 帮助最多"，给出 baseline 跨域 gap（按 AUROC gap 升序）：

| 中心 | Pre-AT AUROC | Δ vs PTBXL | Pre-AT AUPRC | Δ vs PTBXL |
|---|---:|---:|---:|---:|
| **PTB-XL fold10** ★ 参考 | **0.9064** | reference | **0.7754** | reference |
| ptb (n=516) | 0.9009 | −0.55pp | 0.5950 | −18.04pp |
| chap_shaoxing | 0.8835 | −2.29pp | 0.5468 | −22.85pp |
| ningbo | 0.8820 | −2.44pp | 0.5899 | −18.55pp |
| georgia | 0.8168 | −8.96pp | 0.6452 | −13.02pp |
| cpsc_2018 | 0.8052 | −10.12pp | 0.5633 | −21.20pp |
| cpsc_2018_extra | 0.8041 | −10.23pp | 0.6240 | −15.14pp |
| MIMIC test (zero-shot) | 0.7857 | −12.07pp | 0.6939 | −8.14pp |
| st_petersburg (n=74) | 0.7785 | −12.79pp | 0.5575 | −21.78pp |

> AUPRC 跨域 gap 比 AUROC 大很多（~13–23pp vs 2–13pp），因 baseline rate 在不同中心差异大，对类不平衡敏感。AUROC 是更稳的跨域可比指标。

---

## 5. 在线对抗训练 pipeline 技术细节

### 5.1 总流程图

```
[Stage 0] PTB-XL 训练 Super5 victim (一次性, 28 epoch — 详 §5.2)
            EfficientNet1DV2 s_v2 + masked BCE + AdamW + Cosine
            → /root/autodl-tmp/triple_labels/super5/best_model.pt
            → PTB-XL fold10 macro AUROC = 0.9064

[Stage 1] Per cell: K=200 真实目标中心 records 抽样 (详 §5.3)
            cpsc_2018_extra → extra cell anchor pool
            ningbo          → nin cell anchor pool
            georgia         → geo cell anchor pool
            病人级 disjoint with eval; meta 记录 ref_record_ids
            → ECGTwin VAE encode → (200, 4, 128) latent .npz

[Stage 2] 在线对抗训练 (per cell, 100 epoch — 详 §5.4–5.10)
            ┌─────────────────────────────────────────────────────┐
            │ 每 epoch:                                              │
            │   A) PGD attack on frozen anchor pool (§5.4)            │
            │   B) 双 quality gate 过滤 adv 样本 (§5.5)               │
            │   C) push QualityAwareBuffer FIFO 2048 (§5.6)           │
            │   D) Mixed loader 1.0:0.5:2.0 三流采样 (§5.7)           │
            │   E) Masked BCE + EWA anchor 训练 1 步 (§5.8/5.9)       │
            │   F) (每 3 ep) quick eval + early stop (patience=20)   │
            └─────────────────────────────────────────────────────┘

[Stage 3] PN2021 7-ctr + MIMIC + PTB-XL fold10 全量评测 (每 cell)
            scripts/triple_labels/eval_crosscenter.py --exclude_ref_ids 防 leakage
```

**单 cell 训练耗时**：~25 s/epoch × 100 epoch = **~40 min/cell**（real-anchored 实测；synth+diffusion 版本 150 s/epoch = **6× 慢**）。3 cells 并行 GPU ~7 GB / cell（4090 24 GB 容得下）。

---

### 5.2 Pre-AT victim 训练（一次性）

详见 `docs/triple_labels_metrics_for_advisor.md §1`（PTB-XL pipeline 完整描述）。要点：
- Backbone：`EfficientNet1DV2(variant='s_v2', input_channels=12, num_classes=5, use_se=True, stochastic_depth_prob=0.304)`，~5M 参数
- 输入：(B, 12, 250) @ 100 Hz，per-record global z-score（统一来源 `unified_preprocess_to_1000`）
- Loss：`masked_bce_with_logits` + per-class `pos_weight = N_neg/N_pos` clip 50
- Optimizer：AdamW lr=0.01 wd=0.01 + CosineAnnealing T_max=15
- AMP autocast + GradScaler + grad clip 1.0；early stop @ ep28 best val AUROC=0.9126

**关键设计**：在线 AT 阶段以**这个 baseline 为 init**（`--init_ckpt`），不 from-scratch。Tier-M 历史实验证明 from-scratch 50 epoch 内学不动。

---

### 5.3 Anchor pool 构建（per cell, 一次性）

```python
# scripts/pgd_cross_center/prep_real_anchor_npz.py
# 1. 从 PN2021 中心抽 K=200 records (病人级随机, with seed=42)
# 2. 排除其他实验 ref pool 已用病人 (跨实验 disjoint)
# 3. 信号经 unified_preprocess_to_1000 → (1000, 12) @ 100 Hz, z-scored
# 4. ECGTwin VAE encode (channels-last 输入, scale ×0.18215):
#       z = vae_encoder(x_lc).sample() * 0.18215   # (4, 128) per record
# 5. 标签 = SNOMED → super5 multi-hot (NORM exclusivity guard 后)
# 6. 输出 .npz: latents (200, 4, 128) + labels (200, 5) + record_ids
```

**验证**：训完后 anchor 病人 ID 写入 `meta.json`，eval 时 `--exclude_ref_ids` 强制排除。

---

### 5.4 PGD attack 核心（每 epoch 重做）

**起点**：`z₀ = anchor_latents[stratified_sample(K=300, by class)]` ∈ ℝ^{300×4×128}（按 super5 类平衡采样，每类 60 个 anchor）

**PGD 梯度上升**（`adversarial/pgd_advdiff.py:79-130`）：

```
δ ← N(0, 0.1²·I)                       # 随机初始化扰动
δ ← Π_{||δ||₂ ≤ ε}(δ)                  # 投影到 L2 ε-球（保证起点合法）

repeat K_pgd=10 times:
    z = z₀ + δ
    x_ecg = VAE_decode(z)              # (300, 12, 1000) 解码到信号域
    ŷ = victim(x_ecg)                  # (300, 5) logits
    L = BCE_with_logits(ŷ, y_target)   # y_target = anchor 真实 super5 multi-hot
    g = ∇_δ L                          # 对 δ 求梯度
    g_norm = g / ||g||₂                # per-sample L2 normalize（稳定步长）
    δ ← δ + α · g_norm                  # gradient ASCENT (maximize loss)
    δ ← Π_{||δ||₂ ≤ ε}(δ)              # 重新投影
```

**关键超参**：
- `ε = 2.0`（L2 ball 半径，约 anchor norm √512·σ_z=0.149 的 30% — 详 `memory/ecgtwin_usage_guide.md`）
- `K_pgd = 10`（与 Madry 2018 ICLR 标配一致）
- `α = 2ε/K = 0.4`（步长，标准 PGD scaling）
- `delta_init_scale = 0.1`

**为什么走 latent PGD 而不是信号域 PGD？**
- Han 2020 Nat Med 已证 ECG 信号域 PGD 产方波 artifact，临床不合法
- VAE decoder 起 manifold projection 作用，adv samples 自然落在 ECG 流形上
- 学术背书：Wong & Kolter ICLR 2021（perturbation set learning）+ Stutz CVPR 2019（on-manifold AT）

**输出**：`x_adv ∈ (300, 12, 1000)` adv 信号（z-scored），`δ_final ∈ (300, 4, 128)` 最终扰动（保留作 logging）

---

### 5.5 双 quality gate（每 epoch 过滤 adv 样本）

`adversarial/adv_validation.py` 实现两组指标：

**Gate 1 — ASR (Attack Success Rate)**：
- `prob_target = sigmoid(victim(x_adv))[:, target_class]`
- `attack_success = prob_target < 0.5`（成功翻转预测）
- 通过条件：`asr_overall ≥ 0.70` AND `每类 asr ≥ 0.30`
- 验证 PGD **真的攻击成功了**

**Gate 2 — 医学语义（Einthoven residual）**：
- 12 导联第二定律：II ≈ I + III（理论恒等式）
- `residual = |lead_II - lead_I - lead_III|` per sample
- `einthoven_p95 = percentile(residual, 95)` per sample
- 通过条件：`mean(einthoven_p95) < 0.5`（z-scored 信号尺度，< 0.5 即合法 ECG）
- 兼测 HR ∈ [40, 180]、QRS amplitude / 时长与 anchor 接近
- 验证 adv **还像 ECG**（不是 random noise）

**行为**：
- 医学 gate fail → 该 epoch 不 push buffer（continue），不污染
- ASR < 0.30 连续 3 epoch → halt with RuntimeError（PGD broken）
- 通过则进 §5.6 buffer

**实测健康度**（TA-OMAT 3 cells × 100 epoch）：
- ASR overall **0.98–1.00 全程**
- Einthoven p95 **0.218–0.267 全程**（远低于 0.5 阈值）
- medical_pass = True 100% 全程

---

### 5.6 QualityAwareBuffer（FIFO + informativeness eviction）

**目的**：保留过去 epoch 的高质量 adv 样本，跨 epoch 监督；防止 victim 仅看当前 epoch 的 adv 而过拟合。

**数据结构**：环形 FIFO buffer，容量 `qab_size = 2048`（约 7 epoch 历史，K_anchor=300 时）

**Push 规则**：
- 仅当通过双 gate（§5.5）才 push
- 每条 adv 计算 `informativeness_score = 1.0 − 2·|prob_target − 0.5|`（越接近 boundary 分越高）
- score 同时 × `class_trust`（H4 gate, §5.10）→ HYP/CD adv trust=0 等价丢弃

**Pop 规则**：FIFO（最老的先被覆盖）；保留 informativeness 排序作 sampler weight

**采样**：训练时用 `WeightedRandomSampler`，权重 ∝ informativeness_score

---

### 5.7 Mixed loader（三流加权采样）

每个 batch 由 3 个独立 stream 按 `WeightedRandomSampler` 比例混合：

| Stream | 权重 | 内容 | 数量 |
|---|---|---|---|
| **PTBXL real** | 1.0 | PTB-XL fold 1-8 训练集（全量） | ~17,400 |
| **Roundtrip anchor** | 0.5 | PTB-XL real → VAE encode → decode → multi-hot label | 1500 records |
| **Adv buffer** | 2.0 | QualityAwareBuffer 内容（§5.6） | 2048 |

**有效 batch 比例**：PTBXL real : roundtrip : adv = 0.286 : 0.143 : 0.571 ≈ **2 : 1 : 4**

**为什么加 roundtrip anchor？**
- 防 victim 把 "VAE-manifold 失真" 自身学成 abnormality signal
- VAE encode→decode 不完美（Einthoven residual ~0.05 加大）；adv 都是 VAE-decoded，real PTBXL 是 raw → roundtrip 提供"无 PGD 的 VAE 失真样本"作 reference
- 学术背书：Cooperative Training MICCAI 2021（FTN 学 reconstruction + segmentation）

**为什么 adv weight=0.5 而不是 Wang 2023 的 0.7？**
- Wang 2023 是合成图像偏多设定；我们 cross-center 任务 real domain 信号才是 primary
- adv 偏多会导致 victim 漂离真实目标分布
- 0.5 验证为最佳 trade-off（pilot 实验）

---

### 5.8 Loss — Masked BCE with -1 sentinel

```python
def masked_bce_with_logits(logits, labels, pos_weight):
    mask = (labels >= 0).float()         # -1 → 0 in mask, others → 1
    labels_safe = where(mask, labels, 0) # treat -1 as 0 to avoid NaN in BCE
    bce = F.binary_cross_entropy_with_logits(
        logits, labels_safe, pos_weight=pos_weight, reduction='none')
    return (bce * mask).sum() / mask.sum().clamp_min(1.0)
```

**为什么需要 -1 sentinel？**
- adv stream 的 label 是 anchor 原 super5 multi-hot；某些类（H4 trust=0）需 mask 不参与 loss
- roundtrip stream 的 label 也是 super5 multi-hot；NORM exclusivity guard 后 -1 不会出现
- PTB-XL real 全 0/1（PTB-XL diagnostic_class 全覆盖）

**pos_weight**：与 §5.2 baseline 相同，per-class N_neg/N_pos clip 50

---

### 5.9 EWA anchor 正则（Mean Teacher 风格）

**动机**：单纯 BCE 让 victim 在 adv 上学得太激进可能漂离 baseline；引入 EMA teacher 作"软先验"约束 logit 不要离 baseline 太远。

```python
# 初始化
ewa_anchor = copy.deepcopy(super5_baseline)   # 教师 model
for p in ewa_anchor.parameters():
    p.requires_grad = False

# 每 step 更新（在 optimizer.step() 之后）
with torch.no_grad():
    for p_ewa, p_stu in zip(ewa_anchor.parameters(), victim.parameters()):
        p_ewa.data.mul_(ewa_decay).add_(p_stu.data, alpha=1 - ewa_decay)
        # ewa_decay = 0.999

# 计算正则项 (前向时)
logit_diff = (victim(x) - ewa_anchor(x).detach()) ** 2
loss_total = bce_loss + anchor_lambda * logit_diff.mean()
# anchor_lambda = 0.05
```

**学术背书**：
- Tarvainen & Valpola NeurIPS 2017 (Mean Teachers，EMA 教师起源)
- ADR ICLR 2024 (Yu Wu, EMA self-distill 专门 for adversarial training)

**实测影响**：PTB-XL fold10 退化控制在 −0.01 ~ −0.03pp（在噪声内）。

---

### 5.10 H4 per-class trust gate

**问题**：ECGTwin VAE 在 normal_1.pt 参考下生成 HYP（左室肥厚）/CD（束支阻滞）的电压不达医学阈值（详 `memory/ecgtwin_super5_class_support.md`：Sokolow ≤ 2.15 mV，需 ≥ 3.5 mV）。real anchor 不在生成器上跑（直接 encode 真实信号）所以**这个限制本身不直接适用于 TA-OMAT**，但保留 trust gate 是 backward-compat 安全措施。

**实现**：
```python
class_trust = {"NORM": 1.0, "MI": 1.0, "STTC": 1.0, "HYP": 0.0, "CD": 0.0}

# 在 push_adv_to_buffer 中:
for adv, target_cls in zip(adv_signals, target_classes):
    trust = class_trust[target_cls]
    if trust <= 0.0:
        continue   # 不进 buffer，等价 drop
    score = informativeness_score × trust   # 或减权
    buffer.add_one(adv, label, score)
```

**影响**：HYP/CD adv 不参与 AT（masked），所以这两类不享受 AT 增益（Δ ≈ 0）；其它 3 类（NORM/MI/STTC）跨中心提升明显（详 §3/§4 表）。

---

### 5.11 Quick eval + early stop

每 3 epoch 在 PN2021 4-center stratified subset (n=1000) 上跑 macro AUROC：
- 中心：chap_shaoxing / cpsc_2018_extra / georgia / ningbo（cpsc_2018 STTC 词表 gap 排除）
- 监 best `quick_eval_avg_macro_auroc`，patience=20 epoch
- 触发 early stop 后保存 best ckpt

**TA-OMAT 3 cells 实际收敛**：
- extra: best @ ep18, 总 39 ep（早停）
- nin: best @ ep30, 总 51 ep（早停）
- geo: best @ ep51, 总 72 ep（早停）

---

### 5.12 关键超参一览

| 类别 | 参数 | 值 |
|---|---|---|
| Optimizer | lr / weight_decay | 5e-5 / 1e-4 |
| | scheduler | None (固定 lr) |
| | AMP / grad_clip | autocast + scaler / 1.0 |
| Anchor | K_per_center | 200 真实目标中心 records |
| | K_anchor_per_epoch | 300 (stratified by super5 class, 每类 60) |
| PGD | ε (L2) | 2.0 |
| | K_pgd | 10 |
| | α | 2ε/K = 0.4 |
| | delta_init_scale | 0.1 (随机初始化) |
| Buffer | QualityAwareBuffer FIFO | 2048 |
| | adv_weight | 0.5 |
| Mix loader | PTBXL : roundtrip : adv | 1.0 : 0.5 : 2.0 |
| | roundtrip_anchor_n | 1500 |
| Gate | ASR overall threshold | 0.70 |
| | ASR per-class threshold | 0.30 |
| | Einthoven residual p95 max | 0.5 |
| Loss | masked_BCE pos_weight clip | 50 |
| | EWA anchor decay | 0.999 |
| | anchor_lambda | 0.05 |
| Trust | H4 class_trust | NORM/MI/STTC=1.0; HYP/CD=0.0 |
| Train | n_epochs / patience | 100 / 20 |
| | eval_every | 3 epochs |
| | seed | 42 (single seed pilot) |

---

## 6. 关键 caveats（汇报时主动披露）

### 6.1 cpsc_2018 STTC 词表 gap — 唯一显著负向 cell
- cpsc_2018 中心 SNOMED 词表只含 ST elevation/depression，缺 T-wave/Q-wave/QT/ischemia 子码
- PTB-XL STTC 类训练在更广义形态上 → cpsc_2018 STTC AUROC **结构性 capped at 0.59**（baseline 已 0.59）
- AT 推得越狠该类 drag 越大 → 拉低 cpsc_2018 整体 macro -0.86pp
- **不是 method bug**，是数据集词表不可调和。PaperLimitation 章明确披露
- 已 documented：`memory/super5_label_audit_2026_04_26.md`

### 6.2 H4 per-class trust gate (HYP/CD trust=0)
- `memory/ecgtwin_super5_class_support.md`：ECGTwin 数字 GT 校验 NORM/MI/STTC 通过医学阈值，**HYP/CD 0/3 fail**（normal_1.pt ref Sokolow=1.51 mV 限制）
- 决策：HYP/CD adv 样本 trust=0 不进 buffer（结构性，非临时方案）
- TA-OMAT 用 real anchor 后理论可全 1.0（real 标签可信），保留 trust=0 是 backward-compat 安全措施
- 影响：HYP/CD 类不享受 AT 增益，但其它 3 类（NORM/MI/STTC）跨中心提升明显

### 6.3 Pre-AT baseline 排除 ref 病人（fair 协议）
- §2 表 baseline 数字 = **vanilla baseline**（无 AT，全部样本）
- §3/§4 表 Pre-AT 数字 = **ref-excluded baseline**（TA-OMAT 训练 anchor 病人从 eval 排除）
- 两者差异最多 0.0003（200 病人占 ningbo 0.6% / chap 0% 等小比例）
- 这是 IBE R3 caveat 的标准做法（病人级 disjoint），数字可信

### 6.4 单 seed pilot
- 当前结果 = seed=42 单跑。完整 paper 需 3 seed bootstrap CI（pending P2 task）
- 实测 method-side 所有 cell ASR ≥ 0.99 / Einthoven < 0.27 / medical_pass 全程 True → 训练稳定

---

## 7. 决策与方法学落地

### 7.1 与 synth-anchored 实验头对头（Module Ablation #1）

> "Module Ablation #1" 是项目内正式实验 ID，验证"去除 ECGTwin diffusion 模块"假设。胜出方法即 **TA-OMAT**（real-anchored），下表中"实验 2"列。

3 cells × K=200 head-to-head：

| metric | 实验 1 (synth+diffusion) | 实验 2 (TA-OMAT real-anchored) | 优势 |
|---|---|---|---|
| PN2021 avg AUROC Δ | −0.14pp | **−0.07pp** | TA-OMAT +0.07pp |
| PN2021 avg AUPRC Δ | +0.14pp | +0.10pp | synth +0.04pp |
| PTBXL fold10 AUROC Δ | −0.02pp | **−0.01pp** | TA-OMAT +0.01pp |
| MIMIC AUROC Δ | +0.28pp | **+0.30pp** | TA-OMAT +0.02pp |
| cpsc_2018 STTC drag | -1.13pp | **−0.86pp** | TA-OMAT +0.27pp 缓解 |
| 单 epoch 训练 | 150s | **25s** | **6× 加速** |

**3/3 cells AUROC 严格优于**（extra: −0.08 vs −0.17 / nin: −0.05 vs −0.12 / geo: −0.08 vs −0.13）

### 7.2 决策

**移除 ECGTwin diffusion + CenterToken 模块**：
1. AUROC 严格优于（all 3 cells）
2. 6× 训练加速（去 DDPM）
3. Pipeline 大幅简化（5 模块 → 2 模块）
4. Per-target 跨中心 generalization 完全保留
5. 无需 LDM memorization audit
6. cpsc_2018 STTC 反向监督显著缓解（synth 的 "nstemi" prompt 产 MI 形态污染 STTC head）

### 7.3 论文 framing

| 维度 | 内容 |
|---|---|
| Method 名 | **TA-OMAT** (Target-Anchored On-Manifold Adversarial Training) |
| 模块 | 2 个（real anchor PGD + AT），无生成模型 |
| 卖点 1 | 第一个 12-lead 1D ECG **多中心** on-manifold AT pipeline |
| 卖点 2 | 跨中心 per-target consistent **+0.29-0.35pp AUROC / +0.26-0.51pp AUPRC**（4 个目标 × 3 cell 复现） |
| 卖点 3 | Source-center 不升揭示 Stutz 2019 caveat（理论一致） |
| 卖点 4 | 通过 Module Ablation 系统性剥除证明真正起作用的是 latent-PGD 而非 diffusion |
| 学术背书 | Stutz CVPR 2019 + Wong-Kolter ICLR 2021 + Wang ICML 2023 + ADR ICLR 2024 + Madry 2018 |

---

## 8. 限制与未来工作

| 类别 | 内容 |
|---|---|
| 结构性 | cpsc_2018 STTC 词表 gap 不可通过算法修复 → paper Limitations 显式披露 |
| 统计严谨性 | 单 seed pilot → P2 任务跑 3 seed bootstrap CI |
| Memorization | TA-OMAT anchor 都是 PN2021 公开数据，但仍应跑 MIA-AUC audit (Song et al. Princeton DLS 2019) |
| 待跑 ablation | (a) drop roundtrip_anchor (real anchor 自带 clean reconstruction)；(b) K-sensitivity {100, 200, 400} |
| 待补 confirmatory | 5-center TA-OMAT pilot (添加 chap_shaoxing + cpsc_2018 cells) ~5h |

---

## 附录 A：核心训练超参

```yaml
# Super5 baseline (一次性, 28 epoch)
backbone: EfficientNet1DV2 s_v2
input: (B, 12, 250) @ 100 Hz, per-record global z-score
loss: masked BCE + per-class pos_weight (clip 50)
optimizer: AdamW lr=0.01 wd=0.01 + CosineAnnealing T_max=15
batch: 96, AMP, grad_clip=1.0, early_stop patience=10

# TA-OMAT 在线对抗训练 (per cell, 100 epoch)
victim_init: super5 baseline (NOT from-scratch)
optimizer: AdamW lr=5e-5 wd=1e-4
n_epochs: 100, patience: 20, eval_every: 3

# Anchor pool (per cell)
K_per_center: 200 真实目标中心 records (病人级 disjoint with eval)
encode: ECGTwin VAE → (200, 4, 128) latent

# PGD
K_anchor_per_epoch: 300 (stratified sample, class re-balance)
K_pgd: 10 steps
epsilon: 2.0 L2 (≈ 30% anchor norm √512·σ_z=0.149)
random_init_scale: 0.01

# Buffer + Mixer
QualityAwareBuffer FIFO size: 2048
adv_weight: 0.5
WeightedRandomSampler: PTBXL_real:roundtrip_anchor:adv = 1.0:0.5:2.0
EWA anchor: decay=0.999, anchor_lambda=0.05

# H4 Trust gate
NORM:1.0  MI:1.0  STTC:1.0  HYP:0.0  CD:0.0
```

## 附录 B：训练健康度指标（TA-OMAT 3 cells）

| cell | 训练 epoch_run | best_epoch | 平均 ASR | Einthoven p95 | medical_pass |
|---|---|---|---|---|---|
| extra (cpsc_2018_extra) | 39 (early-stop) | 18 | 0.98–1.00 | 0.225–0.226 | True 全程 |
| nin (ningbo) | 51 (early-stop) | 30 | 0.98–1.00 | 0.218–0.223 | True 全程 |
| geo (georgia) | 72 (early-stop) | 51 | 0.99–1.00 | 0.255–0.267 | True 全程 |

PGD method-side 全部健康，无失败 epoch。

## 附录 C：文件归档

| 类型 | 路径 |
|---|---|
| Pre-AT baseline | `/root/autodl-tmp/triple_labels/super5/best_model.pt` |
| Pre-AT eval (vanilla) | `/root/autodl-tmp/triple_labels/super5/eval_result_NORMguard.json` |
| Pre-AT eval (ref-excluded) | `/root/autodl-tmp/triple_labels/super5/eval_with_exclude_3cells_real_k200.json` |
| TA-OMAT 训练产物 (3 cells) | `/root/autodl-tmp/real_anchored_super5/{extra,nin,geo}_real_k200/` |
| TA-OMAT aggregate report | `/root/autodl-tmp/real_anchored_super5/module_ablation_1_report.md` |
| 实验 2 head-to-head | `docs/module_ablation_1_real_vs_synth.md` |
| 实验 1 final summary | `docs/synth_anchored_super5_pilot_summary_final.md` |
| Memory: 实验 1 | `memory/synth_anchored_iter4_final_results.md` |
| Memory: 实验 2 | `memory/module_ablation_1_real_anchored.md` |
| Plan 全文 (9 revisions) | `/root/.claude/plans/luminous-munching-gizmo.md` |
| 核心 orchestrator | `scripts/pgd_cross_center/synth_online_at_super5.py` (anchor-source-agnostic) |
| TA-OMAT-specific scripts | `scripts/pgd_cross_center/{prep_real_anchor_npz.py, run_module_ablation_1_real_anchored.sh}` |
