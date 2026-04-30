# 基于 ECGTwin 的 Class-conditional ECG 增强器 — 毕设方案

> 面向毕设："基于 Latent Diffusion Model 的心电图数据生成与异常检测系统设计与实现"
> 目标：把 ECGTwin (patient-level personalization 工具) 改造为 super5 class-conditional ECG 增强器
> 文档日期：2026-04-30 | 作者：Claude (与开发者讨论后整理)

---

## 概述

### 毕设核心目标

按任务书 + 中期报告，毕设需要交付：
1. 一套基于 LDM 的 ECG 数据生成系统（用于增强分类训练集）
2. 一套基于 CNN 的 super5 异常检测系统（NORM / MI / STTC / HYP / CD 五分类）
3. 一套 Streamlit 可视化前端（含 TensorRT 量化加速）

### ECGTwin 当前架构 vs 毕设目标的 mismatch

ECGTwin 是 **patient-level personalization** 工具（Lai et al., arXiv 2508.02720），不是 class-conditional generator。其 conditioning 三路：

| 路径 | 来源 | 作用 |
|---|---|---|
| `c = t + ib_projector(base_vector)` | timestep + IBE patient identity | AdaLN modulation (DiT block) |
| `c2 = text_projector([text_embed; pat_info])` | nomic-embed prompt + (hr, age, sex) | cross-attention KV |
| `(无)` | — | **没有 class label 显式监督** |

**已知失败的 evidence**：
- HYP/CD 数字 GT 校验 0/3 通过（`docs/ecgtwin_super5_digital_gt_validation.md`）—— 因 `normal_1.pt` 参考 ECG 自身 Sokolow=1.51 mV，而 prompt 不能控制绝对电压
- STTC reverse-supervision: synth-victim AUROC 0.115/0.205（`memory/ecgtwin_prompt_vs_basevector.md`）—— 因 nomic embedding 把 STTC↔MI 在训练共现里学纠缠
- CenterToken (256-d AdaLN 偏移) 实测仅占 AdaLN driver 5-10% magnitude，class control 能力弱（`memory/centertoken_actual_mechanism.md`）

**结论**：必须显式加 class label conditioning + CFG，并精简 patient-level 路径。

### 方案三轴

```
┌────────────────────────────────────────────────────────────┐
│  轴 1: 魔改 (加 class embedder + CFG + DDIM, 删 IBE)        │
│  轴 2: 训练记录 (loss / 临床 / class control / 下游 AUROC) │
│  轴 3: 架构精简 (砍 patient personalization, 留 DiT + VAE) │
└────────────────────────────────────────────────────────────┘
```

---

## 一、ECGTwin 必须做的魔改

### 1.1 [P0 必做] 加显式 super5 class label 注入路径

**位置**：`model/ECGTwin/module/DiT_ECGTwin.py:74-169`

**改法**（class label 进 AdaLN driver）：

```python
# DiT_ECGTwin.__init__ 加:
self.class_embedder = nn.Embedding(num_classes + 1, hidden_size)  # +1 = null class (CFG)
# 或多 hot:
# self.class_embedder = nn.Linear(num_classes, hidden_size)
nn.init.zeros_(self.class_embedder.weight)   # zero-init 让首轮等价无 class 信号

# forward signature 加 class_label:
def forward(self, x, t, text_embed, text_embed_mask, p, base_vector, class_label):
    ...
    t_emb = self.t_embedder(t)
    b = self.ib_projector(base_vector)
    cls_e = self.class_embedder(class_label)   # (B, hidden_size)
    c = t_emb + b + cls_e   # 三路相加进 AdaLN
    ...
```

**为什么进 AdaLN driver 不进 cross-attn**：
1. AdaLN 是 DiT 全局 modulation（per-block shift / scale / gate），CenterToken v2 实测 256-d 偏移就能产 6-16% probit shift（`memory/centertoken_v2_results.md`）—— class signal 走同一通道天然合理
2. cross-attn 已被 nomic text_embed 占住，且 STTC↔MI 在 nomic 空间 cos > 0.69 学纠缠（`memory/ecgtwin_usage_guide.md`）—— class signal 进 cross-attn 反被纠缠权重稀释
3. **TokenVerse SIGGRAPH 2025 Best Paper 直接背书 modulation-space > text-token-space**（Fig.10a ablation；`memory/centertoken_tokenverse_backing.md`）

### 1.2 [P0 必做] Classifier-Free Guidance + DDIM

**Train CFG drop**（`utils/training_utils.py:8-92`）：

```python
# 训练 forward 前随机 drop class
drop_mask = torch.rand(B, device=device) < 0.1   # 10% 替换为 null class
class_label_in = torch.where(drop_mask, num_classes, class_label)  # null = idx num_classes
noise_pred = noise_predictor(xt, t, ..., class_label=class_label_in)
loss = F.mse_loss(noise_pred, noise)
```

**Inference DDIM + CFG**（`ECGTwin_inference.py`）：

```python
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120)
scheduler.set_timesteps(50)   # 50 step → 比 DDPM 1000 step 快 20×

# CFG sampling
for t in scheduler.timesteps:
    noise_cond = model(x, t, ..., class_label=target_class)
    noise_uncond = model(x, t, ..., class_label=null_class)
    guidance_scale = 4.0   # 起手 3-7 区间
    noise = (1 + guidance_scale) * noise_cond - guidance_scale * noise_uncond
    x = scheduler.step(noise, t, x).prev_sample
```

**收益**：
- 中期报告第 2 痛点（DDPM 1000 步 ≈ 数分钟/批）直接消除 —— 50 step ≈ 5-10 s/批
- CFG 让 inference 时可调"类强度"（高 guidance → 类纯净但 diversity 低；低反之），毕设答辩可演示这个 trade-off

### 1.3 [P1 推荐] IBE 路径处理（三选一）

| 方案 | 改法 | 优点 | 缺点 | 学术叙事 |
|---|---|---|---|---|
| **(A) 完全砍** | 删 `IBExtractor` 整个；DiT forward 只剩 `c = t_emb + cls_e` | 最简化；省 ~4M params；省 IBE 训练阶段 | 失去 patient morphology 多样性 | "drop patient-level identity branch" |
| **(B) 50% drop** | 训练时 `if rand < 0.5: base_vector = 0`，推理时 0 base_vector | 保留 ECGTwin pretrained 能力 | 多 4M params 空跑 | "optional patient conditioning" |
| **(C) Class prototype** | 训 5 个 256-d learnable class prototype, `base_vector = class_proto[class_label]` | 直接绕过 IBE 的 patient CLIP 训练；class 控制力最强 | 工程量稍多 | "替换 patient prototype 为 class prototype，将 ECGTwin 从 personalized generator 改造为 class-conditional augmenter" |

**毕设建议走 (C)** —— 学术 framing 最强，工程量也只多 5 × 256 = 1280 个参数。

```python
# 方案 C 实现
self.class_prototypes = nn.Parameter(torch.randn(num_classes, base_vector_dim) * 0.02)
# forward:
proto = self.class_prototypes[class_label]   # (B, 256)
b = self.ib_projector(proto)                  # 取代 IBE 输出
```

### 1.4 [P1] 训练数据切换：MIMIC → PTB-XL

中期报告核心痛点 "跨中心域偏移" 的根因是 ECGTwin 训在 MIMIC 上。

**现成数据**：`/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`（已 VAE-encode + 已 nomic text_embed）

**数据 schema 改造**：
- 复用 PTB-XL fold 1-8 train / fold 9 val / fold 10 test（官方 split）
- super5 multi-hot label 用 `scripts/triple_labels/label_schemes.py::get_scheme("super5")`
- **Lead order**：ECGTwin VAE 期待 MIMIC order（aVL ↔ aVF 与 PTB-XL 互换），用 `util/lead_utils.py::ECGTWIN_TO_PTBXL_INDICES` 转换（自反，apply twice == no-op）
- **采样率**：1024 @ 102.4 Hz（ECGTwin） vs 1000 @ 100 Hz（PTB-XL）→ 训练前 linear interp 1000→1024（RMSE 0.02 mV）
- **新写 dataset**：`data/ptbxl_class_dataset.py` 替代 `mimic_iv_ecg_dataset.py`，每条 sample yield `(latent (4,128), class_label (5,) multi-hot, [可选 text_embed], [可选 pat_info])`

### 1.5 [P2 可选] Text / nomic 路径处理

| 派别 | 改法 | 取舍 |
|---|---|---|
| **简化派 (推荐毕设)** | 把 cross-attn(text_embed) 完全删掉，class label 单一控制源 | pipeline 极简；剔除 STTC↔MI 在 nomic 空间纠缠的根因；论文叙事最干净 |
| **保留派** | 自己写 super5 → "pipe-separated phrases" mapping，避开 ECGTwin 已知 STTC reverse-supervision landmine（不要用 "nstemi" / "myocardial"，用 `"t wave inversion\|repolarization abnormality"`，cos vs MI ≈ 0.634，已是当前最干净）；保留 nomic-embed-text-v1.5 推理依赖 | 保留自由文本控制能力（毕设不需要）；多一个调参变量 |

**毕设走简化派** —— 删 text path，class embedding 单一控制源。

### 1.6 [P2] VAE 层面 minor fix

| Fix | 位置 | 必要性 |
|---|---|---|
| Decoder in-place division 破坏 autograd | `vae_model.py:181` `x /= 0.18215` → `x = x / 0.18215` | 训练时不影响（只跑 encoder），但若未来加 perceptual loss / adversarial latent regularization 必需 |
| `paired_ecg_collate_fn` 要求 paired (ref, target) | `utils/data_utils.py` | 毕设不需要 patient pair，写新的 `class_collate_fn` 只 yield 单 (target, class_label) |
| 复用 frozen VAE checkpoint | `checkpoints/vae_model.pth` | 不重训，毕设创新点在 DiT class conditioning，不在 VAE |

---

## 二、训练时必须保存的图表/记录

按毕设论文 Chap "实验结果" 章必须有的内容分组。

### A. 优化曲线（基本必备）

| 文件 | 内容 |
|---|---|
| `train_loss.csv` | 每 epoch 一行：(epoch, train_mse, lr, epoch_time, vram_peak) |
| `val_loss.csv` | PTB-XL fold 9 hold-out 上的 noise-prediction MSE（**当前 ECGTwinTrainer 没 val loop，必须加**） |
| `loss_components.csv` | 若加多分量 loss：重建 / class CE / CFG drop 各分量曲线 — 直接对应中期报告第 3 痛点（多目标 loss 调权） |
| `ema_loss.csv` | EMA model loss vs 原始 model（如果用 EMA） |

### B. 生成质量定量评估（毕设主图 #1）

| 指标 | 计算 | 报哪 |
|---|---|---|
| **FID-1D** (Fréchet ECG distance) | 用 super5 victim 倒数第二层 (d=1024) 算 FID(生成集 vs PTB-XL test) | 越小越好 |
| **MMD** on raw signal | RBF kernel | distribution distance 第二参考 |
| **Per-lead 信号统计** | mean / std / p5 / p95，12 leads × {real, synth} | 表格对照 |

### C. 临床有效性（毕设导师建议第 1 条强调，**毕设论文核心创新点之一**）

直接复用 `util/ecg_digital_features.py`（已经验证 NORM/MI/STTC 3/3 通过；HYP/CD 0/3 —— 这正是要训练改进的目标）。

每 N=10 epoch 后跑 50/类 × 5 类 = 250 样本：

| 指标 | 计算 | 验收阈值 |
|---|---|---|
| HR (bpm) | RR interval median × 60 | ∈ [40, 180]; class-specific NORM 60-100 |
| QRS duration (ms) | Q→S width | NORM/MI/STTC < 110ms; LBBB/RBBB > 120ms |
| PR interval (ms) | P-onset → QRS-onset | NORM < 200ms; AVB > 200ms |
| ST 振幅 (µV) | J-point + 60ms | STEMI > +200; STD < -50 |
| Sokolow voltage (mV) | SV1 + max(RV5, RV6) | HYP > 3.5（当前 0/3 —— 必须改进） |
| Lateral R amp (mV) | I/aVL R 峰 | LBBB > 0.5（当前 0/3 —— 必须改进） |
| Einthoven residual | ‖II - I - III‖ p95 | < 0.5（lead consistency） |

输出格式 `eval_metrics_epoch{N}.json`，含每类每个指标的 [mean, p5, p95, pass_rate@threshold]。

### D. Class control 能力（**毕设最核心证据**）

| 指标 | 计算 | 阈值 |
|---|---|---|
| **Synth-victim AUROC (per-class)** | 用 `/root/autodl-tmp/triple_labels/super5/best_model.pt` 在生成 ECG (synth=ground-truth pair) 上算 | 每类 > 0.55 才算 class control 起作用（当前 STTC 0.115/0.205 完全失败） |
| **Confusion matrix** | target_class vs victim_predicted_class，5×5 | 看哪两类纠缠 |
| **t-SNE / UMAP** | generated VAE latents colored by target class，每 5 epoch 一张 PNG | 看 class separation 演化 |

### E. 下游分类增强效果（**毕设主图 #2**）

每 10 epoch 出一次：

| 实验 | 训练数据 | 评测 |
|---|---|---|
| Baseline AUROC | PTB-XL fold 1-8 only | PTB-XL fold10 + PN2021 7 中心 + MIMIC test |
| **+Synth AUROC** | real + synth 1:1 mix | 同 baseline |
| **AUROC Δ per class** | 哪类受益、哪类受损 | STTC/HYP/CD 三个弱类的 Δ 是论文 hero number |
| **AUPRC** | 同 AUROC | 中期报告说 "AUPRC 出现较大下滑"，必须报这个 |
| Cross-center | 复用 `scripts/triple_labels/eval_crosscenter.py --scheme super5` | PN2021 7-ctr macro AUROC + per-center delta |

### F. 效率指标（中期报告第 2 痛点直接对应）

| 指标 | 报哪 |
|---|---|
| **DDIM 50 vs DDPM 1000 wall-clock** | 单 batch 50 ECG 的生成时间（DDIM target ~5s vs DDPM ~100s） |
| **VRAM peak** | 训练 + 推理 |
| **Throughput** | samples/min |
| **TensorRT 量化对比** | FP32 / FP16 / INT8 三档 inference latency × accuracy |

### G. 定性可视化（答辩 PPT 用）

| 项 | 频率 | 工具 |
|---|---|---|
| 12-lead PNG | 每 5 epoch × 每类 5 条 | `util/ecg_viz.py engine='ecgplot'` |
| **DDIM trajectory** | 同一 (class, seed)，t=1000 / 800 / 600 / 400 / 200 / 0 各保存一帧 | 看 noise → ECG 演化 |
| **CFG 强度扫** | 同一 class，guidance ∈ {0, 1, 3, 5, 7, 10} | 看类纯净度 vs diversity trade-off |
| **Class interpolation** | latent space 5 类中心点连线 | 看形态过渡是否平滑 |
| **Cross-attention 热图** (若保留 text path) | — | 哪些 prompt token 影响哪个 lead/timestep |

### H. 训练日志结构（推荐目录布局）

```
checkpoints/DiT_class_run_001/
├── config.yaml                          # snapshot
├── train.log                            # text log
├── train_loss.csv                       # epoch-level
├── val_loss.csv                         # epoch-level
├── eval_metrics/
│   ├── epoch_010.json                   # FID + 临床 + synth-victim AUROC + downstream Δ
│   ├── epoch_020.json
│   └── ...
├── samples/
│   ├── epoch_010/
│   │   ├── class_NORM/{0..4}.png
│   │   ├── class_MI/{0..4}.png
│   │   └── ...
│   └── ddim_trajectory_epoch_030.png
├── tsne/
│   ├── epoch_010.png
│   └── ...
├── ckpt/
│   ├── best.pth                         # by val_loss
│   ├── ema.pth
│   └── epoch_{N}.pth
└── final_report.md                      # 自动 aggregate 上述指标 → markdown 表格
```

---

## 三、ECGTwin 架构精简建议

### 3.1 必须删（与毕设目标无关）

| 模块 | 路径 | 理由 |
|---|---|---|
| `IBExtractor` (若选 1.3 方案 A 或 C) | `module/IBExtractor.py` | patient-level 个性化非毕设需求（~4M params） |
| `IBETrainer` | `trainer/IBETrainer.py` | 同上，省独立训练阶段 |
| `CLIPTrainer` | `trainer/CLIPTrainer.py` | patient ID contrastive 不需要 |
| `ECGTwin_Edit` 子类 | `module/DiT_ECGTwin.py:171-253` | report swap/add 编辑功能毕设不需要 |
| `DiT_adaLN.py` / `DiT_ATTN.py` | `module/` 下两文件 | ECGTwin 论文的 ablation backbone, 主线只留 `DiT_ECGTwin.py` |
| `Unet_ECGTwin.py` / `Unet_adaLN.py` / `Unet_ATTN.py` | `module/` 下三文件 | 同上 |
| `config/{DiT_ATTN.yaml,DiT_adaLN.yaml,Unet_ECGTwin.yaml}` | `config/` | 只保留 `DiT_ECGTwin.yaml`（改名 `DiT_class.yaml`） |
| `pECGMonitor/` | 整个目录 | ECGTwin 作者自带的 Streamlit 前端，毕设要从头写自己的 |
| `figure/` | 整个目录 | ECGTwin paper figure 资源 |
| `test_scripts/{batch_generation,ecg_edit,attention_map_visualization,ib_analysis,hr_test}.py` | 5 个 demo 脚本 | 毕设不复用，`evaluation.py` 也要重写 |
| `generation_result_by_disease/` | 整个目录 | 12 类 PNG 演示画廊（750+ 张图）；不参与 pipeline |

### 3.2 可选删

| 模块 | 路径 | 取舍 |
|---|---|---|
| `process_pat_info` (hr/age/sex 注入) | `utils/data_utils.py` | 3-d 信息影响很小；如果毕设不强调 demographic conditioning 可砍 |
| `dataset_analysis.ipynb` | `data/` | 是 ECGTwin 作者 EDA notebook，毕设要写自己的 PTB-XL EDA |

### 3.3 不要砍

- DiT depth=6/7 — 浅了会丢失时序长程依赖
- hidden_size=256 — 与 latent (4, 128) 维度匹配
- DDPM 1000 train steps — 训练时多 step 反而稳定，inference 时切 DDIM 50 步即可
- VAE checkpoint — 已 pretrained 良好，11M 参数从头训成本巨大，不动

### 3.4 推荐最终架构

```
ECGTwin_minimal/                    # 重命名体现毕设原创工作
├── module/
│   ├── vae_model.py                # 复用，frozen
│   ├── DiT_class.py                # 改写自 DiT_ECGTwin.py，class label conditioning
│   ├── Embedder.py                 # 复用 (TimestepEmbedder/RoPE/PositionalEmbedder)
│   └── Attention.py                # 复用 (CrossAttention/SelfAttention)
├── trainer/
│   └── DiT_class_trainer.py        # 改写：class label sampling + CFG drop + EMA + val loop
├── data/
│   └── ptbxl_class_dataset.py      # 新写：PTB-XL super5 + VAE latent + 1024 重采样 + lead reorder
├── utils/
│   ├── training_utils.py           # 改：加 class loss / EMA / val loop / 监控 metric
│   ├── inference_utils.py          # 改：DDIM scheduler + CFG sampling
│   └── eval_utils.py               # 新写：FID, 临床, synth-victim AUROC, downstream
├── config/
│   └── DiT_class.yaml              # num_classes=5, drop_class_prob=0.1, ddim_step=50, ema_decay=0.9999
├── train.py                        # 改自 ECGTwinTrainer.py
└── inference.py                    # 改自 ECGTwin_inference.py
```

**参数量预估**：ECGTwin 原 ~25M（含 IBE 4M + Unet/DiT ablation 一堆）→ 精简版 ~15M（DiT 5M + VAE 11M frozen）

---

## 四、训练时间预算（4090D + 15 核 CPU）

### 4.1 单次训练 (PTB-XL super5, ~17k records, batch=256, AMP fp16)

| 项 | 每 iter | iter/epoch | 每 epoch |
|---|---|---|---|
| DiT forward + backward (batch=256) | ~80 ms (4090D) | 17470 / 256 ≈ 68 | **~5.5 s** |
| DataLoader (latent 已预 encode) | ~5 ms | — | < 1 s |
| Optimizer step + AMP scaler | ~10 ms | — | < 1 s |
| **训练 per epoch 总计** | | | **~7 s** |

**Eval (每 5 epoch 一次)**：

| 项 | 时间 |
|---|---|
| 生成 250 样本 (DDIM 50 step × 5 batch=50) | ~15 s |
| Synth-victim AUROC (super5 victim 推理 250 条) | ~5 s |
| 7 个临床指标 | ~10 s |
| FID + MMD on signal embedding | ~15 s |
| t-SNE + 25 张 12-lead viz PNG | ~30 s |
| **Eval per 5 epoch 总计** | **~75 s** |

**单次主训练 (100 epoch + eval per 5 = 20 次 eval)**：
- 训练: 100 × 7 s = 12 min
- Eval: 20 × 75 s = 25 min
- ckpt 保存 + log: ~3 min
- **主训练单跑 ≈ 40 min**

200 epoch (更稳的收敛预算): **70-80 min/跑**

加 MIMIC 预训练 → PTB-XL fine-tune 两段：
- MIMIC 800k @ batch=512 一 epoch ≈ 156 s × 30 epoch ≈ 80 min
- 加 PTB-XL 100 epoch fine-tune (40 min)
- **两段联合一跑 ≈ 2.2 - 2.5 h**

### 4.2 毕设全周期 GPU 预算

| 实验阶段 | 跑数 | 单跑 | 累计 |
|---|---|---|---|
| Smoke test + debug | 5-8 | 10 min | ~1 h |
| **主训练**: PTB-XL only baseline | 1 | 40 min | 40 min |
| **主训练**: + MIMIC 预训练 + PTB-XL FT | 1 | 2.5 h | 2.5 h |
| **Loss 权重调权** (重建 + class CE + CFG drop_prob × 3 取值 = 9 跑，缩到 50 epoch) | 9 | 25 min | ~4 h |
| **架构消融** (IBE 三方案 / text path on-off / class embedder dim 变化) | 6 | 40 min | 4 h |
| **CFG guidance scale 扫描** (inference-only) | 1 | 10 min | 10 min |
| **DDIM step 数 vs FID 对照** (inference-only) | 1 | 15 min | 15 min |
| **下游分类器增强评测** (real / real+synth, 3 种 mix ratio × 5 generator ckpt) | 15 | 30 min | 7.5 h |
| **Cross-center eval** (5 ckpt × PN2021 7-ctr + MIMIC) | 5 | 10 min | 1 h |
| **TensorRT 量化对比** (FP32/FP16/INT8) | 3 | 5 min | 15 min |
| **GPU 总用时** | | | **~21 h** |

**关键结论**：
- 纯 GPU 时间 **~1.5 天** 连续跑完所有训练（24h 排队，但不现实）
- 实际开发不会连续跑，加 debug / 改代码 / 等结果 / 看图调整：实际占用 **3-5 个工作日**
- 现实开发日历：**3-4 周**（W3-W6 阶段），每天平均 GPU 占用 1-3 h

### 4.3 关键加速 tips（节省 30-50% GPU 时间）

| Tip | 收益 | 是否必开 |
|---|---|---|
| **AMP fp16 训练** | 4090D Tensor Core，单 iter 时间 -40% | 必开 |
| **`torch.compile(model)`** | 4090D + Pytorch 2.x 再省 10-20%（首次编译开销 ~30s 摊销快） | 推荐 |
| **batch=512 不要 256** | latent 已预 encode VRAM 富余；throughput 翻倍 | 推荐 |
| **`num_workers=12 + persistent_workers=True + pin_memory=True`** | DataLoader 永远不是瓶颈（latent 已 cache） | 必开 |
| **Eval 异步化** | eval 期间 GPU 不训，t-SNE / 临床指标移到 CPU side thread | 进阶 |
| **Mixed precision 推理** | DDIM sampling fp16 速度再翻倍 | 推荐 |
| **CenterToken-style hook 复用 ECGTwin pretrained ckpt (LoRA-style)** | from-scratch 30 epoch → fine-tune 5 epoch，单跑 40 min → 8 min | 时间紧时用，但学术贡献减弱 |

---

## 五、毕设全周期时间表（与中期报告 W6-12 对齐）

| 周次 | GPU 时间 | 主要工作 | 累计 GPU |
|---|---|---|---|
| W1 (代码改造) | ~1 h | DiT class embedder 加入 + DDIM 切换 + smoke test | 1 h |
| W2 (数据准备) | ~0.5 h | PTB-XL super5 dataset + lead reorder + 1024 重采样 | 1.5 h |
| W3-4 (主训练) | ~5 h | baseline + MIMIC 预训练 + 调权 5 跑 | 6.5 h |
| W5 (消融) | ~5 h | 架构消融 + 调权剩余 + CFG/DDIM 扫描 | 11.5 h |
| W6 (下游评估) | ~9 h | super5 增强评测 + cross-center | 20.5 h |
| W7 (TensorRT) | ~1 h | 量化 + 部署 | 21.5 h |
| W8-15 (撰写 + 答辩) | ~5 h | 补实验 + 答辩 demo 重跑 | ~26 h |

**风险点**：如果 50 epoch 内 class control 不收敛（synth-victim AUROC 仍有类 < 0.55），就要加长到 200-500 epoch + 调权迭代，预算可能膨胀 2-3×（特别是 STTC/HYP/CD 三个已知难类）。建议 W3 第一跑就设 100 epoch 看曲线趋势。

---

## 六、毕设论文写作 framing 建议

### 6.1 不要 claim

- **"first ECG LDM"** —— Alcaraz SSSD-ECG (CBM 2023) / DiffECG (arxiv 2306.01875) / ECGTwin 自己 (arxiv 2508.02720) 都已发
- **"first manifold AT for ECG"** —— Yang/La Cava et al. 2025 (arxiv 2509.19564) 已发
- **"first medical TI augmenter"** —— Wilde MIDL 2024 已发

### 6.2 三个明确创新点

1. **Class label 进 AdaLN modulation**（区别于 prompt-only conditioning，cite TokenVerse SIGGRAPH 2025 + DiT ICCV 2023）
2. **CFG + DDIM 解决采样效率**（中期报告第 2 痛点的工程层创新）
3. **跨中心 augmentation：PTB-XL 训生成器 + 评测 PN2021 / MIMIC**（cite Stutz CVPR 2019 on-manifold AT framing；区别 Yang 2025 单中心 pediatric STFT 路线）

### 6.3 限制必写

- **HYP/CD 数字 GT 不通过的 root cause**：参考样本电压分布限制，不是模型 capacity 限制（来自 `docs/ecgtwin_super5_digital_gt_validation.md`）—— 这是真诚也保护毕设的发现
- **DDIM 50 step 的 FID degradation vs DDPM 1000 step**（如果有）
- **跨域 (PTB-XL → MIMIC) AUPRC 下滑**（中期报告原话）
- **Class label 是 multi-hot 还是 one-hot 的 trade-off**（multi-hot 更贴合 PTB-XL 共病，但 CFG 数学不严格；one-hot 简化但失去多标签真实性）

### 6.4 答辩 demo 必有的画面

1. 同一 class 五条不同 ECG 12-lead 网格（diversity）
2. CFG 强度从 0 → 10 滑块（可控性）
3. Synth-victim AUROC per-class bar chart（class control 证据）
4. Baseline vs +Synth AUROC bar chart on PTB-XL / PN2021 / MIMIC（增强器效果证据）
5. DDIM 50 step vs DDPM 1000 step latency 对比（效率证据）

---

## 附录 A：Memory & papers 引用速查

### 关键 memory（已有）

| 文件 | 内容 |
|---|---|
| `memory/ecgtwin_usage_guide.md` | ECGTwin VAE I/O / nomic / DiT cross-attn / 9 个 landmine |
| `memory/ecgtwin_super5_class_support.md` | NORM/MI/STTC 3 类官方支持，HYP/CD 0/3 通过的 root cause |
| `memory/ecgtwin_prompt_vs_basevector.md` | prompt 一阶 / base_vector 二阶 / CenterToken 三阶噪声实测 |
| `memory/centertoken_actual_mechanism.md` | CenterToken 5 trainer bug + 多层验证证伪 |
| `memory/centertoken_v2_results.md` | per-block per-center sphere=5 部分 style transfer 成功 |
| `memory/centertoken_tokenverse_backing.md` | TokenVerse SIGGRAPH 2025 直接背书 modulation-space |
| `memory/triple_labels_training.md` | super5/sub23/pn26 三个 victim 的 baseline AUROC |
| `memory/super5_label_audit_2026_04_26.md` | NORM exclusivity guard，5-ctr 0.755 → 0.839 |

### 必引 paper

| Paper | arxiv | 用处 |
|---|---|---|
| Rombach et al. 2022 LDM | 2112.10752 | 整体 LDM 范式 |
| Peebles & Xie 2023 DiT | 2212.09748 | DiT 架构 + AdaLN-Zero |
| Ho et al. 2020 DDPM | 2006.11239 | DDPM 训练目标 |
| Song et al. 2020 DDIM | 2010.02502 | DDIM 加速采样 |
| Ho & Salimans 2021 CFG | 2207.12598 | Classifier-Free Guidance |
| Geyer et al. 2025 TokenVerse | 2501.12224 | modulation-space token 直接先例 |
| Lai et al. 2025 ECGTwin | 2508.02720 | base model |
| Alcaraz & Strodthoff 2023 SSSD-ECG | 2301.08227 | ECG diffusion baseline |
| Strodthoff et al. 2020 PTB-XL benchmarks | 2004.13701 | super5 标签官方协议 |
| Stutz et al. 2019 On-manifold AT | 1812.00740 | cross-center generalization framing |

---

## 附录 B：关键 file:line 索引

| 任务 | 文件 | 行 |
|---|---|---|
| DiT class embedder 加入 | `model/ECGTwin/module/DiT_ECGTwin.py` | 78-108 (`__init__`), 128-169 (`forward`) |
| CFG drop 训练逻辑 | `model/ECGTwin/utils/training_utils.py` | 8-92 (`train_epoch_channels`) |
| DDIM scheduler 切换 | `model/ECGTwin/ECGTwin_inference.py` | (整体 inference loop) |
| VAE in-place division fix | `model/ECGTwin/module/vae_model.py` | 181 |
| Lead order 转换 | `util/lead_utils.py::ECGTWIN_TO_PTBXL_INDICES` | — |
| Super5 label scheme | `scripts/triple_labels/label_schemes.py::get_scheme("super5")` | — |
| Super5 victim ckpt | `/root/autodl-tmp/triple_labels/super5/best_model.pt` | — |
| Cross-center eval | `scripts/triple_labels/eval_crosscenter.py --scheme super5` | — |
| 临床指标 extractor | `util/ecg_digital_features.py` | — |
| 12-lead 可视化 | `util/ecg_viz.py` | — |
| ECGTwin wrapper | `util/ecgtwin_utils.py::ECGTwinWrapper` | — |

---

## 附录 C：决策树速查

```
开始
├── Step 1: 选 IBE 路径方案（1.3 节）
│   ├── (A) 完全砍 → 工程最简，论文 framing 一般
│   ├── (B) 50% drop → 折中，保留 future work 空间
│   └── (C) Class prototype → 推荐毕设 ★
│
├── Step 2: 选 text path（1.5 节）
│   ├── 删 → 推荐 ★（避开 STTC↔MI 纠缠）
│   └── 保留 → 需要自己写 super5 prompts 表
│
├── Step 3: 选预训练策略
│   ├── PTB-XL only from-scratch → ~1 h/跑，单一域
│   ├── MIMIC 预训练 + PTB-XL fine-tune → ~2.5 h/跑，跨域更稳
│   └── 复用 ECGTwin pretrained + LoRA-style fine-tune → ~10 min/跑，但学术贡献减弱
│
├── Step 4: smoke test（W1-2）
│   ├── 100 epoch → 看曲线
│   ├── 若 W3 第一跑 synth-victim AUROC 全类 > 0.55 → 推完整实验
│   └── 若有类 < 0.55 → 加长 epoch + 调权 + 加 H4 trust gate fallback
│
├── Step 5: 主实验（W3-6）
│   └── 21 h GPU 跑完 11 个核心实验
│
└── Step 6: 论文 + 答辩（W7+）
    └── 重点 framing: 三个创新点 + 限制坦诚 + 5 个答辩 demo 画面
```

---

**最后注**：本方案是基于 2026-04-30 时间点的 ECGTwin repo + 已有 memory + 已发表 paper 综合给出。如果在 W1-W2 实施时发现新的 ECGTwin 行为或新的 SOTA paper，欢迎根据实际情况调整。

实施代码层面如需 trial 实现（例如 class embedder 代码骨架、CFG 训练 loop、临床指标 eval pipeline），可以单独开 trial branch 跑出最小 working example 再展开。
