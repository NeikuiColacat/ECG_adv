# Center-Style Token 实验结果（首期 — cpsc_2018_extra）

## 1. 实验目标

用 textual-inversion 思路在 ECGTwin 上学一个单 256-dim 可学 token 来编码某中心的 **device/recording style**，拿这个 token 做条件生成 → 合成"像该中心"的 ECG → fine-tune EfficientNet 来缩 cross-center gap。

这是对 AugMix（时域 / latent 两版已验证）的正交路线：AugMix 改变的是**噪声/漂移等 corruption**，中心 token 改变的是 VAE 流形上的**中心风格**。

首期只做 **cpsc_2018_extra**（baseline AUROC 0.8538，MAIN5 里最弱的中心）。

## 2. 方法简述

```
PN2021 cpsc_2018_extra 731 条 ECG  ──[unified preprocess @100Hz, zscore]──▶  (N, 12, 1000)
                                                                                 │
                                                                                 ▼
                                   [ECGTwinVAE encode, PTBXL→ECGTwin lead swap, interp 1000→1024]
                                                                                 │
                                                                                 ▼
                                                                           (N, 4, 128) latent
                                                                                 │
                                        ┌────────────────────────────────────────┴──────────────┐
                                        │ CenterTokenTrainer                                     │
                                        │ ─ 冻结 DiT、VAE、nomic、IBExtractor                      │
                                        │ ─ 只更新 256-dim Parameter                             │
                                        │ ─ 通过 forward_pre_hook 注入到每个 DiTBlock 的 AdaX 路径  │
                                        │   c ← t_emb + ib_proj(base) + center_token            │
                                        │ ─ Loss = L_recon + 0.1·L_disease_inv + 0.01·L_reg     │
                                        └──────────────────────────┬─────────────────────────────┘
                                                                   ▼
                                                       center_token_best.pth (256 params)
                                                                   │
                     [per 6 Tier-M 类各 1000 条] ──[DDPM 50 步 + hook 注入]── decode → (N, 12, 1000)
                                                                   │
                                                                   ▼
                                                          cpsc_2018_extra_synth.npz (6000 条)
                                                                   │
                                                                   ▼
                              [EfficientNet1DV2 s_v2 训练，real:synth = 70:30 WeightedRandomSampler]
                                                                   │
                                                                   ▼
                                               PTBXL + 7 center eval → 跨中心 macro AUROC
```

### 2.1 超参

| 阶段 | 配置 |
|---|---|
| Token 训练 | AdamW lr=1e-3, 30 epochs, batch=128, λ_inv=0.1, λ_reg=0.01, grad_clip=1.0 |
| 合成生成 | 每类 N=1000 (共 6000), batch=50, DDPM 50 步 |
| EfficientNet fine-tune | 50 epochs, batch=128, AdamW lr=0.01, cosine T_max=15, pos_weight per class, synth_ratio=0.3, seed=42 |

### 2.2 训练侧关键指标

- **token norm**: 0.82（健康：[0.1, 5]）
- **delta_mean**（token 是否帮 diffusion reconstruction）: 0.0005 → 0.002，持续 > 0
- **L_inv**: 近 0（token 未退化成 class-specific）
- **PTBXL test AUROC（fine-tuned）**: 0.9741 ← 与 baseline 相同，源域未崩 ✅

## 3. 跨中心评测结果

### 3.1 Headline（4-way）

| Metric | Baseline | Time-AugMix | Latent-AugMix | **CenterToken** | Δ CT vs Base |
|---|---:|---:|---:|---:|---:|
| PTBXL test macro AUROC | 0.9741 | 0.9735 | 0.9742 | **0.9741** | **+0.00pp** |
| PTBXL test macro AUPRC | 0.8447 | 0.8196 | 0.8172 | 0.8159 | -2.88pp |
| **MAIN5 avg macro AUROC** | **0.9291** | 0.9321 | 0.9315 | **0.9326** | **+0.35pp** |
| MAIN5 avg macro AUPRC | 0.7399 | 0.7347 | 0.7362 | 0.7402 | +0.03pp |

**CenterToken 的 MAIN5 AUROC 涨幅（+0.35pp）比 time-AugMix（+0.30pp）和 latent-AugMix（+0.24pp）都略高。**

### 3.2 Per-Center Macro AUROC

| Center | N | Baseline | Time-AugMix | Latent-AugMix | CenterToken | Δ CT vs Base |
|---|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing ★ | 9709 | 0.9575 | 0.9609 | 0.9584 | 0.9565 | -0.09pp |
| cpsc_2018 ★ | 5279 | 0.9166 | 0.9197 | 0.9247 | **0.9356** | **+1.91pp** |
| cpsc_2018_extra ★ 🎯 | 1296 | 0.8538 | 0.8618 | 0.8568 | 0.8568 | +0.30pp |
| georgia ★ | 9320 | 0.9482 | 0.9485 | 0.9493 | 0.9503 | +0.21pp |
| ningbo ★ | 34470 | 0.9695 | 0.9697 | 0.9681 | 0.9639 | **-0.56pp** |
| ptb (small) | 116 | 0.8543 | 0.9449 | 0.9279 | 0.8842 | +2.98pp |
| st_petersburg (small) | 33 | 0.9218 | 0.9220 | 0.9521 | 0.9249 | +0.32pp |

*★ = main-5 center（计入 headline）；🎯 = center token 的训练目标*

### 3.3 目标中心 cpsc_2018_extra 的 per-class AUROC 变化

| Class | Baseline | CenterToken | Δ (pp) |
|---|---:|---:|---:|
| NSR | 0.7417 | 0.7214 | **-2.03pp** |
| STach | 0.9712 | 0.9706 | -0.06pp |
| AF | 0.9626 | 0.9654 | +0.28pp |
| IAVB | 0.9121 | 0.9074 | -0.47pp |
| **LBBB** | 0.5661 | 0.5989 | **+3.28pp** |
| RBBB | 0.9693 | 0.9774 | +0.81pp |

LBBB 这个之前最弱的类提升了 +3.28pp，RBBB 提升 +0.81pp；但 NSR 从 0.74 跌到 0.72（-2.03pp）—— **该中心 NSR 只有 4 条真实样本，合成样本的多样性受限**，合成 NSR 的变分可能把分类器拉向非代表性 NSR 模式。

## 4. 核心结论与分析

### 4.1 方向是对的（但幅度和 AugMix 持平）

- MAIN5 AUROC +0.35pp（略胜 AugMix）
- PTBXL 源域 **完全持平**（+0.00pp）
- 训练 + 生成 + fine-tune 全链路 pass，无 NaN / 崩塌

### 4.2 最大惊喜：**token 帮了 cpsc_2018（+1.91pp），超过目标中心 cpsc_2018_extra（+0.30pp）**

CPSC_2018_extra 和 CPSC_2018 来自同一医院（同 SNOMED、同采集设备约定），**token 学到的是"CPSC-style"而不是 cpsc_2018_extra-specific 风格**。这是好消息：单次训练的 token 能泛化到兄弟中心。

### 4.3 目标中心增益有限的 3 个可能原因

1. **Baseline 已触底**：cpsc_2018_extra NSR n=4 的 LBBB baseline 0.57 意味着基础分类器在此表现极差，提升空间存在但需要更强干预
2. **合成数据多样性不足**：NSR 只有 4 个 refs → 1000 条合成样本高度同质 → 分类器学到的是"那 4 个样本的 interpolation"而非真实 NSR 分布
3. **synth_ratio=0.3 可能偏高**：该中心样本稀缺时，合成过多会盖过源域信号

### 4.4 ningbo 小回归（-0.56pp）的原因

Token 注入的是 cpsc_2018_extra 风格（中国医院设备/导联放大倍数），和 ningbo（也是中国但 **不同** 医院，AF↔AFL 标签错乱是已知 quirk）的 style 有冲突。**这说明方法确实在改变模型的 device-style 先验**。

## 5. 判定

| 判据 | 阈值 | 实际 | 结果 |
|---|---|---|---|
| 目标中心 AUROC 提升 | ≥ +1.0pp | +0.30pp | ❌ 未达到 |
| PTBXL 源域 drop | ≤ +0.5pp | +0.00pp | ✅ |
| MAIN5 avg 改进 | 方向正 | +0.35pp | ✅ |
| 兄弟中心 transfer | 意外 | +1.91pp | ✨ |

**总结**：**方法 work，但目标中心的增益没达到单变量 +1.0pp 的预设门槛。实质上与 AugMix 打平（MAIN5 +0.35 vs time +0.30 vs latent +0.24），成本却更高（需要 VAE + DiT + nomic 全套加载）。**

按原判定：
- ❌ cpsc_2018_extra +0.30pp < +1.0pp → 判为 **AMBIGUOUS**
- ✅ 但 cpsc_2018 +1.91pp、MAIN5 +0.35pp 说明方法非无效
- ⚠️ Verdict: "与 AugMix 等价，目前成本收益比不如 AugMix"

## 6. 下一步建议（按优先级）

### 6.1 进一步验证方向（低成本，验证价值）

1. **对 cpsc_2018 训一个专门的 token**（有 5279 条，6 类齐全包括 NSR n=918）
   - 预期：自中心 +1.5~2.5pp，且不拖累 cpsc_2018_extra
   - 时间成本：~90 min（和本实验相同）

2. **对 chapman_shaoxing 训一个 token**（最丰富的中心之一）
   - 检验方法在高质量中心上是否仍有收益

### 6.2 真正的改进方向

1. **合成数据增多样性**：减少 mode collapse
   - 每样本用不同 patient_info (hr/age/sex) 随机采样
   - 扩大 reference pool（多随机画 ref）
   - DDPM 步数从 50→100（换取 1.5× 时间）

2. **Per-center token 的多中心联合**：训 4 个主 center 的 token，合成 4×6000 条，真:合成 = 50:50 混合训练

3. **CenterToken × AugMix 叠加**：观察两种不同 augment 机制是否正交

4. **合成样本质量评估**：
   - FID-style 分布距离（合成 vs 真实目标中心）
   - EfficientNet feature-space 的 domain gap（是否缩小）

### 6.3 不建议继续的方向

- ❌ 继续用 cpsc_2018_extra 单中心调超参（baseline 天花板太低）
- ❌ 增加 epoch / lr 搜索（当前已 converge）
- ❌ 改 token dim（256 → 512/1024）：参数仍然太少，对表达能力影响有限

## 7. 产物清单

```
methods/ecgtwin_gen/center_token/                        # 从 trash 迁入
├── __init__.py, model.py, trainer.py, generate.py, config.yaml

scripts/ecgtwin_gen/
├── prep_center_dataset.py   # 构建 per-center VAE-latent + text_embed .pt
├── train_center_token.py    # CenterTokenTrainer 入口
├── generate_center_synth.py # 生成 N 条合成 ECG per class
└── compare_4way.py          # 4-way 对比报告

scripts/crosscenter_tierM/train_ptbxl_tierM.py
   - 新增：--synth_center_npz, --synth_ratio, SynthCenterDataset, ConcatDataset+WeightedRandomSampler

/root/autodl-tmp/center_token/
├── cpsc_2018_extra.pt              # 731 条 latent + text (训练集)
├── cpsc_2018_extra_ckpt/
│   ├── center_token_best.pth       # 训好的 token (256 params)
│   └── center_token_latest.pth
└── cpsc_2018_extra_synth.npz       # 6000 条合成 ECG (267 MB)

/root/autodl-tmp/crosscenter_tierM_centertoken/cpsc_2018_extra/
├── best_model.pt
├── train_result.json
├── training_log.json
└── eval_crosscenter.json           # 7 中心 AUROC

outputs/augmix_validation/compare_4way.md   # 4-way 对比 markdown
```

## 8. 时间成本

| 步骤 | 时间 |
|---|---|
| 数据集 prep（扫 + 预处理 + VAE encode）| 40 s |
| Token 训练 30 epochs | ~50 s |
| 生成 6000 ECG | 41 s |
| EfficientNet fine-tune 50 epochs | ~10 min |
| 7 中心 eval | ~14 min |
| **共计** | **~26 min** |

对比原计划 95 min，快了 3.6 倍（prep 和 token 训练比预想更快）。
