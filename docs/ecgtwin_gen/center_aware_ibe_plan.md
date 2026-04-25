# Plan B v2 — Style Translator IBE + Prototype Encoder

> **角色转变**：IBE 从"捕获个体身份"升级为 **"回答 counterfactual: 如果这条 ECG 来自目标中心 X，它的 base_vector 应该长什么样？"**
>
> **部署友好**：用 PrototypeEncoder 代替固定 `nn.Embedding(N_centers, 256)` 查表。新医院 50 条未标注 ECG → 1 个 256-d style vec → 立刻可用，**不需要再训练**。

---

## Context

### 背景回顾

- **Baseline**（Tier-M EfficientNet1DV2 s_v2 on PTBXL）：PTBXL test AUROC **0.9741**, MAIN5 avg AUROC **0.9291**（cross-center gap -4.50pp），弱中心 cpsc_2018_extra **0.8538**
- **CenterToken**（256-dim 可学向量加到 DiT AdaLN `c` 路径）：cpsc_2018_extra +0.30pp，和 AugMix 持平。**失败假设**：256 维 additive token 被 680K 参数 IBExtractor 的通用 disease 特征淹没

### Plan B v1（已废弃）
曾打算做 `feat_final = feat_old + alpha * center_bias[cid]` 的 pooled additive bias。**问题**：
1. 不是真正的 "角色转变"——老 feat 仍占主导，只加了偏移
2. `nn.Embedding(N_centers)` 是死表——**新医院部署时 center_id 不存在，方案不可迁移**

### Plan B v2（本文档）
核心升级两处：

1. **IBE 改造为 Counterfactual Style Translator**：接受 `style_vec` 作为 conditioning，输出"这条 ECG 如果来自 target center 的 base_vector"
2. **`style_vec` 来自 PrototypeEncoder(K 支撑样本)**，不再是查表：任何医院（训练见过或没见过）都只需要 50 条 ECG 跑一次前向

### ECGTwin 关键限制回顾（已验证）

- DiT conditioning：`c = t_emb + ib_projector(base_vector)`（`DiT_ECGTwin.py:145-146`），AdaLN 调制
- DiT 训练时对 base_vector 有 **15% 随机 channel mask**（`training_utils.py:41`）——对 base_vector 扰动有鲁棒性
- 系统盘 5.4G free，**一切大件必须落 `/root/autodl-tmp/`**；数据盘 60G free

### 判定（Stage 0，针对 cpsc_2018_extra）

| 结果 | 解读 |
|---|---|
| AUROC ≥ +1.0pp 且 PTBXL drop ≤ +0.5pp | Style Translator 成立，Stage 1 扩 5 中心 |
| +0.5pp ~ +1.0pp | 方向对，但不比 CenterToken 显著强，优化 λ/K |
| < +0.3pp | 角色转变假设失败 → pivot |

---

## 架构

### 1. PrototypeEncoder（~50K 参数）

K 条 VAE-latent → 1 个 256-d style vector。**置换不变**（mean pool），支持任意 K。

```python
# methods/ecgtwin_gen/style_translator/prototype_encoder.py
class PrototypeEncoder(nn.Module):
    """
    K support ECG latents (K, 4, 128) → (256,) permutation-invariant style vector.
    """
    def __init__(self, in_channels=4, hidden=128, out_dim=256):
        super().__init__()
        self.per_sample = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, hidden, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),              # (K, hidden, 1)
        )
        self.aggregate = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),                # stabilize style vec scale
        )

    def forward(self, support_latents):           # (K, 4, 128)
        h = self.per_sample(support_latents).squeeze(-1)  # (K, 128)
        style = h.mean(dim=0)                              # (128,)  perm-inv
        return self.aggregate(style)                       # (256,)
```

**Why this design**:
- Conv1d on latent 利用序列结构（P/QRS/T 波的位置信息）
- Mean pool 而不是 attention：enrollment 时 K=50 太少不值得用 self-attn
- LayerNorm 在最后：保证不同医院的 style_vec 尺度一致

### 2. StyleTranslatorIBE（~200K 可训参数 + 冻结 base IBE 680K）

```python
# methods/ecgtwin_gen/style_translator/model.py
class StyleTranslatorIBE(nn.Module):
    """
    Wraps a frozen IBExtractor. Adds style-conditioned residual.
    
    IBE(x, style=None) == base_IBE(x)        # backward compat
    IBE(x, style=s)    == base_IBE(x) + style_fusion([base_IBE(x), s])
    """
    def __init__(self, base_ibe: IBExtractor, prototype_encoder: PrototypeEncoder):
        super().__init__()
        self.base_ibe = base_ibe
        for p in self.base_ibe.parameters():
            p.requires_grad = False           # freeze
        self.prototype_encoder = prototype_encoder
        self.style_fusion = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 256),
        )
        # Zero-init last: initial behavior ≡ base IBE, δ=0
        nn.init.zeros_(self.style_fusion[-1].weight)
        nn.init.zeros_(self.style_fusion[-1].bias)

    def extract_features(self, x, text_embed, mask, p, style_vec=None, reduce=True):
        with torch.no_grad():
            base_feat = self.base_ibe.extract_features(x, text_embed, mask, p, reduce=reduce)
        if style_vec is None:
            return base_feat
        # style_vec: (256,) -> (B, 256)
        B = base_feat.shape[0]
        style = style_vec.unsqueeze(0).expand(B, -1)
        delta = self.style_fusion(torch.cat([base_feat, style], dim=-1))
        return base_feat + delta
```

**Trainable 参数总计**：~250K（PrototypeEncoder 50K + style_fusion 200K）

**冻结**：base IBE 680K + DiT 12.8M + VAE 2M

---

## 训练损失

三个目标互补：

### L_identity（锚点：不偏离 base IBE）

When the support set comes from **the same center as the query ECG**: behavior should ≡ base IBE（保证冷启动安全、保留已训练的患者-disease 编码能力）。

```
L_identity = || SIBE(x, style=proto(same_center)) − base_IBE(x) ||_2^2
```
λ_id = 1.0

### L_style_transfer（核心：风格翻译）

For `x_A` from center A with Tier-M class D, sample `x_B` from center B with **same class D**:
```
L_style = || SIBE(x_A, style=proto_B) − SIBE(x_B, style=proto_B) ||_2^2
```

即：feeding (x_A, proto_B) 应该给出 (x_B, proto_B) 一致的表征——把 A 的内容翻译到 B 的风格。
λ_style = 0.5

**对齐策略**：batch 内按 Tier-M 类 hard-match；若无 cross-class pair 则跳过此 loss 那一步。

### L_supcon（中心判别）

SupCon（Khosla 2020）on query outputs, 正负样本按 "用了哪个 proto" 分组：
```
L_supcon = SupConLoss(
    {SIBE(x_i, proto_{c_i})}, 
    labels=[c_i]
)
```
迫使 style 信号真的注入到 feat，使 feat 按 style 分簇。
λ_sup = 1.0

### 总损失

```
L = λ_sup · L_supcon + λ_id · L_identity + λ_style · L_style_transfer
  = 1.0 · SupCon   + 1.0 · Identity   + 0.5 · StyleTransfer
```

---

## 数据

### Stage 0（**3 中心**，为 SupCon 提供 3-way 判别信号）

- `cpsc_2018_extra`（目标，~800 Tier-M 正例）
- `chapman_shaoxing`（大样本对照）
- `cpsc_2018`（**cpsc_2018_extra 的兄弟中心**，CenterToken 实验意外 +1.91pp 的那个；同时作为 SupCon 的第 3 类提供更强判别信号）

### Per-iteration 采样

- K = 50 support per center
- Query: **256 samples**（4090 24GB 显存富余，batch size 128 → 256，训练快 ~40%）
- Per batch 总 VAE-latent 张量：3×50 + 256 = 406 条 → 可容纳
- DataLoader `num_workers=8`（15 核 CPU 充分利用）

### 存储

`/root/autodl-tmp/center_aware_ibe/datasets/3center_styled.pt`（~75 MB）

---

## 部署 Playbook（新医院）

### Step 1 — Style Enrollment（< 1 min）

```bash
python scripts/ecgtwin_gen/enroll_new_center.py \
    --ecg_dir /path/to/new_hospital_ecgs \
    --n_samples 50 \
    --prototype_ckpt /root/autodl-tmp/.../stage0_style_translator/prototype_best.pth \
    --out /root/autodl-tmp/.../styles/new_hospital_style.pt
```

内部：读 50 条 ECG → unified_preprocess_to_1000 → VAE encode → PrototypeEncoder → `(256,) style_vec` → save

**要求**：仅 **50 条原始 ECG（无 label）**。

### Step 2 — 风格化合成（~40 sec）

```bash
python scripts/ecgtwin_gen/generate_with_style_translator.py \
    --style_vec /root/autodl-tmp/.../styles/new_hospital_style.pt \
    --translator_ckpt /root/autodl-tmp/.../stage0_style_translator/translator_best.pth \
    --n_per_class 1000 \
    --ref_source /root/autodl-tmp/.../datasets/2center_styled.pt \
    --out /root/autodl-tmp/.../synth/new_hospital_synth.npz
```

### Step 3 — EfficientNet 再训练

```bash
python scripts/crosscenter_tierM/train_ptbxl_tierM.py \
    --synth_center_npz .../new_hospital_synth.npz \
    --synth_ratio 0.3 \
    --output_dir .../eval/new_hospital/
```

### 隐私友好选项

| 方案 | 流程 |
|---|---|
| Raw upload | 新医院上传 50 条 ECG → 我方跑 enrollment |
| **Federated enrollment** | 在医院本地部署 `VAE_Encoder + PrototypeEncoder`（~3MB）→ 仅回传 `(256,) style_vec`。原始 ECG 永不出院。 |

---

## Reuse vs 新增

| 模块 | 路径 | 动作 |
|---|---|---|
| IBExtractor 基类（frozen） | `model/ECGTwin/module/IBExtractor.py` | 复用，不改 |
| DiT / VAE / ECGTwinWrapper | `util/ecgtwin_utils.py`, `model/ECGTwin/module/*` | 复用 |
| Tier-M label 扫描 | `scripts/crosscenter_v2/label_alignment_v2.py` + `scripts/crosscenter_tierM/eval_crosscenter_tierM.py::{scan_center_records, parse_header_snomed}` | 复用 |
| Per-center prep | `scripts/ecgtwin_gen/prep_center_dataset.py::{_encode_batch, TIER_M_CANONICAL_PROMPT, _parse_header_meta, _fold_from_hash}` | 复用 |
| 生成后处理 | `scripts/ecgtwin_gen/generate_center_synth.py::_postprocess` | 复用 |
| EfficientNet retrain | `scripts/crosscenter_tierM/train_ptbxl_tierM.py --synth_center_npz ... --synth_ratio 0.3` | 直接复用 |
| Tier-M eval | `scripts/crosscenter_tierM/eval_crosscenter_tierM.py` | 复用 |
| 4-way 对比 | `scripts/ecgtwin_gen/compare_4way.py` | 复制改名 compare_5way.py |

### 新增（Stage 0）

| 文件 | 功能 | 大小 |
|---|---|---|
| `methods/ecgtwin_gen/style_translator/__init__.py` | export | 小 |
| `methods/ecgtwin_gen/style_translator/prototype_encoder.py` | PrototypeEncoder | ~40 LOC |
| `methods/ecgtwin_gen/style_translator/model.py` | StyleTranslatorIBE | ~60 LOC |
| `methods/ecgtwin_gen/style_translator/losses.py` | SupConLoss + helpers | ~60 LOC |
| `methods/ecgtwin_gen/style_translator/trainer.py` | 三损失训练 loop + K=50 support sampling | ~280 LOC |
| `methods/ecgtwin_gen/style_translator/config_stage0.yaml` | 超参 | ~30 LOC |
| `scripts/ecgtwin_gen/prep_multicenter_dataset.py` | 扫 2+ 中心 → 统一 .pt | ~120 LOC |
| `scripts/ecgtwin_gen/train_style_translator.py` | CLI 薄壳 | ~50 LOC |
| `scripts/ecgtwin_gen/enroll_new_center.py` | 新医院 50 条 ECG → style_vec | ~80 LOC |
| `scripts/ecgtwin_gen/generate_with_style_translator.py` | style_vec + ref → synth .npz | ~180 LOC |
| `scripts/ecgtwin_gen/compare_5way.py` | baseline/time/latent/centertoken/**style_translator** 对比 | ~200 LOC |
| `docs/ecgtwin_gen/style_translator_results.md` | 结果 + go/no-go | ~300 LOC |

### 不改

- `model/ECGTwin/**`
- `scripts/crosscenter_tierM/train_ptbxl_tierM.py`（`--synth_center_npz` 已支持）
- `scripts/crosscenter_tierM/eval_crosscenter_tierM.py`

---

## 存储布局（严守系统盘 5.4G 上限）

```
/root/autodl-tmp/center_aware_ibe/
├── datasets/
│   ├── 3center_cpsc_chapman_cpsc_extra_styled.pt  # ~75MB (Stage 0)
│   └── 5center_main_styled.pt                      # ~250MB (Stage 0-multi, 条件)
├── ckpts/
│   ├── stage0_style_translator_3c/                 # Stage 0 产物
│   │   ├── translator_best.pth                     # ~1MB
│   │   ├── prototype_best.pth                      # ~200KB
│   │   ├── training_log.json
│   │   └── config_used.yaml
│   └── stage0multi_style_translator_5c/            # Stage 0-multi 产物
├── styles/                                         # 计算出的 style vectors
│   ├── cpsc_2018_extra_style.pt                    # ~2KB (256 float)
│   ├── chapman_shaoxing_style.pt                   # ~2KB
│   ├── cpsc_2018_style.pt                          # ~2KB (兄弟中心)
│   ├── georgia_style.pt                            # ~2KB (ablation C 用)
│   └── new_hospital_style.pt                       # 部署后产生
├── synth/
│   ├── stage0_A_cpsc_extra_style.npz               # ~267MB 主实验
│   ├── stage0_B_null_style.npz                     # ~267MB ablation: 不传 style
│   ├── stage0_C_georgia_style.npz                  # ~267MB ablation: 陌生中心错 style
│   └── stage0multi_5center_combined_synth.npz      # ~1.3GB (5×6000=30K, 条件)
├── eval/
│   ├── stage0_A_main/                              # 25MB
│   ├── stage0_B_null/                              # 25MB
│   ├── stage0_C_wrong/                             # 25MB
│   └── stage0multi_main/                           # 25MB (条件)
└── logs/
    ├── stage0_training.log
    ├── stage0_generation.log
    └── stage0multi_training.log
```

**预估总占用**：Stage 0 ~900 MB；Stage 0-multi（若触发）+1.6GB = ~2.5 GB 总计。远在 60 GB free 内。

**代码全在系统盘 `/root/ECG_adv_Gen/`（< 100 KB 新增）**。

---

## 实施步骤（Stage 0：3 中心）

| # | 动作 | 时间 | 产物 |
|---|---|---:|---|
| 1 | PrototypeEncoder + StyleTranslatorIBE + SupConLoss 模块 + smoke test | 45 min | `methods/.../{prototype_encoder,model,losses}.py` |
| 2 | `prep_multicenter_dataset.py` **3 中心**扫描（复用现有 utils） | 20 min | `scripts/ecgtwin_gen/prep_multicenter_dataset.py` |
| 3 | 跑 prep（~2400 条，VAE encode + text embed） | 12 min | `3center_styled.pt` |
| 4 | `trainer.py`：load frozen base IBE，三损失，K=50 support sampling，**batch=256 / num_workers=8** | 60 min | `trainer.py` |
| 5 | 跑 20 epochs 训练 | **~15 min**（4090 加速） | `translator_best.pth`, `prototype_best.pth` |
| 5.5 | 训练指标验证（见下方 Verification） | — | stdout |
| 6 | `enroll_new_center.py`（仅 50 条 ECG → style_vec） | 15 min | 脚本 |
| 7 | `generate_with_style_translator.py` | 35 min | 脚本 |
| 8 | 跑 3 路 synth 生成（A: target style, B: null, C: **georgia 错 style**） | **~10 min**（4090 每路 ~3 min） | 3 个 .npz |
| 9 | 3 路 retrain EfficientNet（reuse `train_ptbxl_tierM.py`） | **~21 min**（4090 每路 ~7 min） | 3 × best_model.pt |
| 10 | 3 路 eval_crosscenter_tierM | ~42 min | 3 × eval_crosscenter.json |
| 11 | `compare_5way.py` + ablation table | 20 min | markdown |
| 12 | `docs/ecgtwin_gen/style_translator_results.md`（结果 + go/no-go） | 20 min | 结果文档 |

**Stage 0 总时间**：impl ~3h + run ~2h = **~5h**

## 实施步骤（Stage 0-multi：5 中心，仅在 Stage 0 通过后）

**复用 Stage 0 所有架构，不加 LoRA。** 只扩数据到全 MAIN5 → 拿 MAIN5 avg headline 指标。

| # | 动作 | 时间 |
|---|---|---:|
| 13 | 扩 prep 至 5 中心（+georgia +ningbo subsample 3K） | 15 min |
| 14 | 重训 StyleTranslator（同架构，n_centers=5）20 epochs | ~25 min |
| 15 | 生成 5 × 6000 = **30K synth**（全 MAIN5 风格，combined） | ~15 min |
| 16 | EffNet retrain 单次（PTBXL + 30K synth，synth_ratio=0.3）| ~8 min |
| 17 | 7 中心 eval + 写入 compare_5way.md 补列 | ~15 min |

**Stage 0-multi 总时间**：**~1.5h**

---

## Verification（每节点的确认性检查）

### 训练阶段（Step 5.5）

- **L_identity** epoch 10 后收敛到 < 0.01（feat 在同 center 支持下几乎不变）
- **L_style_transfer** 从 init 下降 ≥ 50%
- **L_supcon** 从 ~ln(B_per_class) 下降到 < 1.0
- **style_fusion 最后一层权重** ||W|| > 0（不是 trivially zero）
- **PrototypeEncoder 稳定性**：同一 center 重复采 K=50 两次，两个 style_vec 的 cos-sim > 0.9
- **centers 可分性**：训练集所有样本的 `SIBE(x, proto_{true_cid})` 在 TSNE 上按 cid 分簇

### 生成阶段（Step 8 后）

- 3 个 .npz 每个 `signals: (6000, 12, 1000)`, finite，HR ∈ [30, 200] 占比 ≥ 85%
- Einthoven p95（II − I − III 残差）< 1.0（DiT 没崩）
- 肉眼对比 (A vs B) 波形：A 应带 cpsc_extra 风格（可能是基线漂移 / 噪声 profile）

### Retrain 阶段（Step 9-10）

- **A（目标 style）**：PTBXL ≥ 0.970；cpsc_2018_extra AUROC ≥ 0.859（baseline + 0.5pp）
- **A > B**：证明 style_vec 承载了有效信号
- **A > C**：证明 style_vec 的 center-specific 性（错 style 不如对 style）

### 部署可行性（额外做一次）

- 用 PTBXL 当 "新医院" enroll → 得到 PTBXL_style_vec
- PTBXL_style_vec 与 cpsc_extra / chapman 的 style_vec cos-sim < 0.8（应明显不同）
- 用 PTBXL_style_vec 生成 synth，人眼看波形是否像 PTBXL（v.s. 2 个训练中心的生成结果）

### Go / No-Go（主 gate）

| 判据 | 门槛 | 结果 |
|---|---|---|
| target A AUROC vs baseline | ≥ +0.5pp | go |
| A > B delta | ≥ +0.3pp | go (style works) |
| A > C delta | ≥ +0.1pp | go (correct style > wrong) |
| PTBXL test AUROC | ≥ 0.970 | go (source domain 不崩) |

全 go → 写 "Proceed to Stage 1"（5 中心 + 可能解冻 base IBE LoRA）
任一 fail → pivot 分析 / 调 λ

---

## 关键设计决策

| 问题 | 选择 | 为什么 |
|---|---|---|
| IBE 角色 | **Counterfactual Style Translator** | 从 "我是谁" 转变为 "我能变成谁"，对 cross-center 生成语义最贴合 |
| Center ID 表达 | **PrototypeEncoder(K ECGs) → 256-d vec** | 新医院零训练部署；支持连续 style 空间 |
| Base IBE | **冻结** | 保留已训练的 disease/patient 编码能力；降低 DiT OOD 风险 |
| 可训模块规模 | ~250K（PrototypeEncoder 50K + style_fusion 200K） | 比 Plan v1 的 ~1K 大得多，但仍小于 base IBE 本身 |
| style_fusion 初始化 | 最后一层 **zero-init** | 初始行为 ≡ base IBE，训练安全启动 |
| 中心数 | 2（cpsc_extra + chapman） | Plan agent 建议；2 中心足够 debug + SupCon |
| 每 center K | **50** | 足够稳定 prototype；不会爆内存 |
| 主损失 | **SupCon + L_identity + L_style_transfer** | 分别覆盖：center 判别、安全锚点、核心翻译目标 |
| L_disease | **drop** | text_embed 已输入 IBE，disease 走捷径，没教到 center |
| ref_latent | Stage 0 保持目标中心 ref（和 CenterToken 实验对齐） | 可比较 |
| synth_ratio | **0.3** | 和上一期一致 |
| DDPM 步数 | **50** | 和上一期一致 |

---

## 风险 & 回滚

| 风险 | 征兆 | 应对 |
|---|---|---|
| PrototypeEncoder mode collapse | 不同 center 的 style_vec cos-sim > 0.95；L_supcon 不降 | K 50→100；加 batch-level SupCon 直接在 proto 上；LayerNorm 加 affine |
| style_fusion 永不激活（zero-init 困境）| L_style_transfer 不降；feat == base_feat | 改 std=0.01 init；加 warmup：先只训 PrototypeEncoder + L_supcon |
| Style 过强，DiT OOD | HR 分布失败率 > 20%；Einthoven p95 > 1.0 | 加 learnable gate `α`（类似 Plan v1）；降 λ_sup；加 L_reg 约束 delta |
| Enrollment 不稳定 | 同中心 3 次 enrollment style cos-sim < 0.9 | K↑；PrototypeEncoder 最后层 LayerNorm 要有 `affine=True` 并训稳定 |
| Style 过弱（= baseline） | A ≈ B AUROC | 扩展 style_fusion 容量；改把 style 注入 base_ibe 内部而非 post-hoc 融合（Stage 1） |
| 磁盘爆 | /root/autodl-tmp 使用 > 90% | 清旧 `crosscenter_tierM_augmix*`；Stage 1 前清 Stage 0 的 B/C synth |

---

## 非目标 / 延后

- ❌ 首期不 > 2 个中心（Stage 1 才扩 5 中心）
- ❌ 首期不做 few-shot labeled fine-tune（enrollment 仅用 50 条 **未标注** ECG）
- ❌ 不动 DiT / VAE 权重
- ❌ Stage 0 暂不 unfreeze base IBE（Stage 1 可能加 LoRA）
- ❌ 首期不实现 federated VAE_Encoder 部署包（架构已支持，延后）
- ❌ 首期不做 StyleTranslator + CenterToken ensemble
- ❌ 首期不做 seed sweep / token 可视化

---

## Stage 0-multi（条件触发，Stage 0 通过后）

**复用 Stage 0 的 StyleTranslator 同架构**（不加 LoRA），只扩数据到 MAIN5：

- 扩 5 中心（Stage 0 的 3 个 + georgia + ningbo）
- `ningbo` subsample 到 3K（避免失衡）
- 重训 StyleTranslator 20 epochs（n_centers=5）
- 5 个中心各生成 6000 条 synth → combined 30K
- EffNet 单次 retrain（PTBXL + 30K synth，synth_ratio=0.3）
- 目标：**MAIN5 avg AUROC 增益 ≥ +0.5pp**（presentation-ready headline）

**未纳入本预算的未来选项**（如果 Stage 0-multi 仍然不够）：
- LoRA rank=8 挂到 `IBEncoderLayer.cross_attn.Q/K/V`（~200K 新增）
- DiT 的 `ib_projector` Linear(256,256) 解冻（65K）
- 重训时间 +1h，impl +2h，延后到未来实验

---

## 时间预算（9h 总预算，4090 + 15 核 CPU）

| 阶段 | 时间 |
|---|---:|
| Impl（3 模块 + 5 脚本） | ~3h |
| Stage 0 run（3 中心 + A/B/C ablation） | ~2h |
| Stage 0-multi run（条件：5 中心 scale-up） | ~1.5h |
| Buffer / debug / 写 docs | ~2.5h |
| **总计** | **~9h** ✓ |

**两种 scenario**：
- Stage 0 通过 → Stage 0-multi → 总 **~9h**（吃满预算）
- Stage 0 失败 → 跳过 multi，写 pivot 分析 → 总 **~6h**（留 3h 调研后续方向）

### 4090 加速对比

| 操作 | 老卡估计 | 4090 估计 | 加速 |
|---|---:|---:|---:|
| DDPM 50 步生成 6000 条 | 41 s | ~25 s | 1.6× |
| EffNet retrain 50 epochs | ~10 min | ~7 min | 1.4× |
| StyleTranslator 20 epochs | ~20 min | ~15 min | 1.3× |
| PTBXL retrain batch=128 → 256 | — | 吞吐 +40% | — |

### 原 Stage 1（LoRA + DiT 解冻）延后

砍掉 LoRA 不是因为没用，而是：
1. 2h 的 impl 成本对 9h 预算太贵
2. Stage 0-multi 如果能拿到 +0.8pp MAIN5，已经是 presentation-ready 结果
3. LoRA 作为"如果 Stage 0-multi 仍然不够"的备用方案，留给未来
