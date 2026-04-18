# ECG_adv_Gen 项目结构文档

## 项目概述

**目标**：通过 ECGTwin（条件扩散模型）合成带有 PTBXL 中心特征的 ECG 样本，结合 AdvDiff 对抗梯度引导生成决策边界困难样本，微调 DeepECG EfficientNet 分类器，提升其在跨中心数据集（MIMIC-IV-ECG）上的诊断性能。

**核心问题**：DeepECG 在 MHI 训练集上 Infarction/Ischemia AUPRC = 0.77，跨中心到 MIMIC 后 AUPRC 降至 0.089。

---

## 顶层目录结构

```
ECG_adv_Gen/
├── adversarial/           # 🔑 核心：对抗生成 + 微调 + 评估 pipeline
├── model/                 # 第三方模型仓库（git submodule）
│   ├── ECGTwin/           #   条件扩散生成模型（DiT / U-Net + VAE + IBExtractor）
│   ├── DeepECG/           #   77类 ECG 分类器（EfficientNetV2，JIT frozen）
│   ├── advdiff/           #   AdvDiff 对抗扩散攻击框架（参考实现）
│   └── ecg_ptbxl_benchmarking/  # PTB-XL 基准评估工具
├── scripts/               # 实验运行脚本（pipeline 各阶段入口）
├── util/                  # 工具函数（ECGTwin 封装、数据加载、攻击工具）
├── data/                  # 数据预处理脚本
├── datasets/              # 原始数据集
│   ├── PTBXL/             #   PTB-XL ECG 数据（21799条，12导联）
│   └── MIMIC/             #   MIMIC-IV-ECG 数据
├── center_token/          # Center Token 模块（中心特征注入方案）
├── outputs/               # 实验产出（模型权重、日志、评估结果、ECG图片）
├── result/                # 早期实验结果 & ECG 可视化图片
├── roadmap.md             # 技术路线 & 实验计划
├── pyrightconfig.json     # Python 类型检查配置
├── 工作簿1.csv             # 数据分析表格
└── 附录相关技术细节.pdf     # 论文附录技术细节
```

---

## 核心模块详解

### 1. `adversarial/` — 对抗生成 Pipeline（~2000行）

本项目的核心自研代码，实现从对抗样本生成到微调再到评估的完整 pipeline。

| 文件 | 行数 | 功能 |
|---|---|---|
| `label_mapping.py` | 227 | PTB-XL SCP 诊断码 ↔ EfficientNet 77类 pattern 的双向映射；定义 `SCP_TO_EFFICIENTNET` 映射表、`MIMIC_HARD_TARGETS`（MI子类型到目标pattern索引）、prompt-to-target 聚合关系 |
| `efficientnet_victim.py` | 244 | **可微分推理链路**：VAE latent → ECG decode → 导联重排 → resample → EfficientNet logits。关键修复：(1) VAE decoder in-place `/=` 改为 out-of-place；(2) ECG ±3mV clamp 防止 OOD 梯度爆炸 |
| `adv_generate.py` | 629 | **对抗生成核心**：`BoundaryAdvDiffGenerator` 类，在 DDPM 反向去噪过程中注入对抗梯度，将 EfficientNet 预测概率推向 0.5 决策边界。关键：latent-relative gradient step sizing（归一化梯度方向，按 `guidance_scale × ‖latent‖` 缩放） |
| `efficientnet_adapter.py` | 146 | Residual Adapter：frozen EfficientNet logits → LayerNorm → FC(77→128) → GELU → Dropout → FC(128→77) → 残差连接。参数量 ~20K，fc2 零初始化保证初始行为 = 原模型 |
| `finetune.py` | 310 | 微调训练循环：只训练 adapter 参数，使用 BCEWithLogitsLoss + per-class pos_weight 处理类别不平衡，CosineAnnealing 学习率调度 |
| `evaluate.py` | 421 | AUPRC/AUROC 评估：按 6 大临床类别（ECG_CATEGORIES）分组计算 macro-average AUPRC，输出 baseline vs finetuned 对比表 |
| `__init__.py` | 11 | 模块导出 |

**对抗生成核心算法（`adv_generate.py`）**：

```
输入：ECGTwin conditions（text_embed, pat_info, ref_latent）
      EfficientNet victim（frozen JIT model）
      目标 pattern indices

for t in reverse_timesteps:
    # 1. DDPM 正常去噪一步
    x_prev = ddpm_step(x_t, t, conditions)

    # 2. 如果在 guidance 窗口内（后50% timestep）：
    if t < guidance_start:
        # 计算 boundary loss = (sigmoid(logits[target]) - 0.5)²
        loss = boundary_loss(victim(x_prev), target_indices)
        grad = -∇loss / ‖∇loss‖  # 归一化梯度方向
        # Latent-relative step sizing（3% of latent norm per step）
        x_prev += 0.03 × ‖x_prev‖ × grad

输出：latent → VAE decode → ECG (1024, 12)
筛选：target pattern 均值概率 ∈ [0.4, 0.65]
```

### 2. `model/` — 第三方模型仓库

#### `model/ECGTwin/` — 条件扩散 ECG 生成模型

```
ECGTwin/
├── module/                    # 模型组件
│   ├── DiT_ECGTwin.py         #   DiT 架构主模型（Diffusion Transformer）
│   ├── IBExtractor.py         #   Information Bottleneck Extractor
│   │                          #   输入 ref_latent(128,4) + text_embed + pat_info
│   │                          #   self-attn + cross-attn → base_vector(256)
│   ├── vae_model.py           #   VAE 编码器/解码器（ECG ↔ latent (4,128)）
│   ├── clip_model.py          #   CLIP 文本编码器（诊断文本 → text_embed）
│   ├── Embedder.py            #   嵌入层
│   ├── Attention.py           #   注意力模块
│   ├── DiT_adaLN.py           #   AdaLN（Adaptive Layer Normalization）
│   └── DiT_ATTN.py / Unet_*  #   替代架构
├── trainer/                   # 训练器
│   ├── ECGTwinTrainer.py      #   主训练循环
│   ├── IBETrainer.py          #   IBExtractor 训练
│   └── CLIPTrainer.py         #   CLIP 微调
├── utils/                     # 工具函数
│   ├── data_utils.py          #   数据加载 & pat_info 构造
│   │                          #   pat_info = [hr_norm, age_norm, sex_binary]
│   ├── inference_utils.py     #   推理工具
│   └── model_utils.py         #   模型加载
├── config/                    # 模型配置 YAML
│   ├── DiT_ECGTwin.yaml       #   DiT 主配置
│   └── IBEConfig.yaml         #   IBExtractor 配置
├── checkpoints/               # 预训练权重
│   ├── ECGTwin_DiT.pth        #   DiT 模型权重
│   ├── vae_model.pth          #   VAE 权重
│   └── ibe_model.pth          #   IBExtractor 权重
├── pECGMonitor/               # ECG 可视化/监控工具
│   └── pECG_generation.py     #   ECG 图片生成（ecg_plot）
└── ECGTwin_inference.py       # 推理入口
```

**ECGTwin 生成流程**：
```
text_prompt → CLIP → text_embed (768-dim)
ref_ecg → VAE encode → ref_latent (128, 4)
pat_info = [hr_norm, age_norm, sex_binary]

IBExtractor(ref_latent, text_embed + pat_info) → base_vector (256-dim)

DiT 去噪（100 steps）:
  timestep → t_embed
  c = t_embed + base_vector    ← base_vector 通过 adaLN 全局调制
  x_T (noise) → x_0 (latent)

VAE decode(x_0) → ECG (1024, 12) @ 500Hz
```

#### `model/DeepECG/` — 77类 ECG 分类器

```
DeepECG/
├── weights/
│   └── efficientnetv2_77_classes/  # JIT-compiled EfficientNetV2 权重
├── utils/
│   └── constants.py                # ECG_PATTERNS (77类名称列表)
│                                   # ECG_CATEGORIES (6大临床类别分组)
│                                   # BERT_THRESHOLDS, PTBXL_POWER_RATIO
├── models/                         # 模型定义
├── main.py                         # 训练入口
└── compute_optimal_thresholds.py   # 阈值计算
```

**EfficientNet 规格**：
- 输入：`(B, 12, 2500)` @ 250Hz，需乘 `mhi_factor = 1/0.0048 ≈ 208.33`
- 输出：`(B, 77)` raw logits → sigmoid → 独立概率（多标签分类）
- JIT frozen，不可修改权重，但 autograd 可以流过

#### `model/advdiff/` — AdvDiff 参考实现

原版 AdvDiff 论文的对抗扩散攻击实现，本项目从中借鉴了：
- DDPM 采样框架
- VAE decoder autograd 绕过 in-place 操作的技巧
- 梯度引导注入的基本模式

### 3. `scripts/` — 实验运行脚本（~1260行）

Pipeline 各阶段的执行入口，按执行顺序排列：

| 脚本 | 行数 | 功能 | 执行顺序 |
|---|---|---|---|
| `run_prepare_ptbxl.py` | 39 | PTB-XL 数据预处理：raw ECG → VAE encode → ref_latent | Step 0 |
| `run_train_center_token.py` | 64 | 训练 Center Token 模块 | (可选) |
| `run_generate_with_center.py` | 17 | 用 Center Token 生成样本 | (可选) |
| `run_evaluate_baseline.py` | 61 | 评估 EfficientNet 在 PTB-XL 上的基线 AUPRC | Step 1 |
| `run_adv_generate.py` | 115 | 对抗生成困难样本（调用 adv_generate.py） | Step 2 |
| `run_finetune.py` | 132 | 微调 adapter（调用 finetune.py） | Step 3 |
| `run_evaluate_finetuned.py` | 89 | 评估微调后性能 | Step 4 |
| `run_mi_experiment.py` | 283 | **一键运行完整 MI 实验 pipeline**（生成→微调→评估→画图） | All-in-one |
| `prepare_hard_samples.py` | 160 | 准备 inferolateral MI 困难样本 | 辅助 |
| `test_fix_quality.py` | 224 | 验证 bug 修复效果（latent norm ratio、ECG 振幅检查） | 测试 |
| `plot_comparison.py` | 76 | 生成真实 MI vs 合成样本对比图 | 可视化 |

**完整 Pipeline 执行顺序**：
```
Step 0: run_prepare_ptbxl.py      → 准备 PTB-XL VAE 编码数据
Step 1: run_evaluate_baseline.py  → 得到 baseline AUPRC
Step 2: run_adv_generate.py       → 生成困难样本 (~2-4h GPU)
Step 3: run_finetune.py           → 微调 adapter (~1h)
Step 4: run_evaluate_finetuned.py → 评估提升效果
(或直接: run_mi_experiment.py     → 一键执行 Step 2-4)
```

### 4. `util/` — 工具函数（~2200行）

| 文件 | 行数 | 功能 |
|---|---|---|
| `ecgtwin_utils.py` | 502 | **ECGTwin 封装类** `ECGTwinWrapper`：加载所有组件（VAE, DiT, IBExtractor, CLIP），提供 `prepare_conditions()`, `ddpm_sample()`, `decode_latent()`, `encode_ecg()` 接口 |
| `advdiff_attack.py` | 648 | **AdvDiff 攻击封装** `AdvDiffAttacker`：原版 AdvDiff 的对抗采样实现，VAE decode autograd workaround |
| `pgd_atk.py` | 422 | PGD 攻击实现（对比实验用） |
| `save_tool.py` | 333 | 结果保存/加载工具 |
| `get_PTBXL.py` | 124 | PTB-XL 数据加载：`get_ecg_dataset()` → (train, val, test, mean, std) |
| `lead_utils.py` | 156 | ECG 导联处理：`ECGTWIN_TO_PTBXL_INDICES`（导联重排映射），`prepare_ecg_for_classifier()` |
| `test_third_party_repos.py` | 17 | 第三方仓库功能验证测试 |

### 5. `center_token/` — Center Token 模块（~790行）

独立的中心特征注入方案（ECGTwin 之外的另一种思路）：

| 文件 | 行数 | 功能 |
|---|---|---|
| `model.py` | 24 | Center Token 模型定义 |
| `trainer.py` | 505 | 训练器：学习各中心的特征 token |
| `generate.py` | 258 | 用 center token 条件生成 ECG |
| `config.yaml` | - | 训练配置 |

### 6. `data/` — 数据预处理

| 文件 | 行数 | 功能 |
|---|---|---|
| `prepare_ptbxl_for_ecgtwin.py` | 382 | 将 PTB-XL raw ECG 转换为 ECGTwin 可用的格式：VAE 编码得到 ref_latent，提取 pat_info，构造诊断文本 prompt |

### 7. `datasets/` — 原始数据集

```
datasets/
├── PTBXL/
│   ├── ptbxl_database.csv         # 21799条记录的元数据
│   ├── scp_statements.csv         # SCP 诊断码定义
│   ├── records500/                # 500Hz 原始波形 (.dat/.hea)
│   ├── records100/                # 100Hz 降采样波形
│   ├── raw100.npy                 # 预处理后的 100Hz numpy 数据
│   ├── PTBXL_vae_500MI.pt         # VAE 编码后的 MI 样本
│   └── PTBXL_vae_multi_nomic.pt   # VAE 编码后的多类别样本
└── MIMIC/
    ├── record_list.csv            # 记录列表
    ├── files/                     # 波形文件
    └── machine_measurements.csv   # 机器测量数据
```

---

## 实验产出 `outputs/`

```
outputs/
├── eval_baseline.json             # Baseline AUPRC 评估结果
├── eval_finetuned.json            # 微调后 AUPRC 评估结果
├── hard_samples/                  # 早期对抗生成困难样本
│   ├── MI_inferior.pt             #   下壁心梗困难样本
│   ├── MI_anterior.pt             #   前壁心梗困难样本
│   ├── MI_lateral.pt              #   侧壁心梗困难样本
│   ├── MI_acute.pt                #   急性心梗困难样本
│   ├── NORM.pt / LBBB.pt / ...    #   其他类别
│   └── all_hard_samples.pt        #   合并后的全部困难样本
├── adapter_best.pt                # 最优 adapter 权重
├── finetune_history.pt            # 微调训练历史
├── mi_experiment/                 # MI 实验完整产出
│   ├── MIMIC_MI_inferior.pt       #   下壁心梗对抗样本（修复后）
│   ├── MIMIC_MI_acute.pt          #   急性心梗对抗样本
│   ├── MIMIC_MI_inferolateral.pt  #   下侧壁心梗对抗样本
│   ├── all_hard_samples.pt        #   合并样本
│   ├── adapter_best.pt            #   最优 adapter
│   ├── eval_results.json          #   评估结果（含 per-pattern AUPRC）
│   ├── ecg_plots/                 #   ECG 可视化（按置信度排序）
│   ├── ecg_plots_fixed/           #   Bug 修复后的 ECG 可视化
│   ├── fix_test/                  #   修复验证图片
│   ├── old_buggy_samples/         #   修复前的有 bug 样本（备份）
│   ├── confidence_distribution.png #  置信度分布图
│   ├── real_vs_synth_comparison.png # 真实 vs 合成 ECG 对比图
│   └── pipeline_log_fixed.txt     #   修复后的 pipeline 日志
└── *.log                          # 各阶段运行日志
```

---

## 关键数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                              │
│                                                                  │
│  PTB-XL raw ECG ──→ VAE encode ──→ ref_latent (128, 4)         │
│  PTB-XL metadata ──→ pat_info [hr, age, sex]                    │
│  SCP codes ──→ text prompt (e.g. "inferior MI|pathological Q")  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      对抗样本生成阶段                            │
│                                                                  │
│  ref_latent ─┐                                                   │
│  text_embed ─┤→ IBExtractor → base_vector (256-dim)             │
│  pat_info ───┘                                                   │
│                                                                  │
│  base_vector + t_embed → DiT 去噪 (100 steps)                   │
│       ↓                                                          │
│  在后50% timestep 注入对抗梯度：                                  │
│       latent → VAE decode → ECG → EfficientNet → boundary_loss  │
│       grad = -∇loss, normalized, scaled by 3% × ‖latent‖        │
│       ↓                                                          │
│  筛选：target pattern 均值概率 ∈ [0.4, 0.65]                     │
│       ↓                                                          │
│  输出：困难样本 ECG (12, 2500) + 77-dim 标签                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        微调阶段                                  │
│                                                                  │
│  困难样本 + PTB-XL 真实数据                                      │
│       ↓                                                          │
│  ECG → EfficientNet (frozen) → logits → Adapter → adapted_logits│
│       ↓                                                          │
│  BCEWithLogitsLoss + per-class pos_weight                        │
│  只优化 adapter ~20K 参数                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        评估阶段                                  │
│                                                                  │
│  PTB-XL test fold (fold 10)                                     │
│       ↓                                                          │
│  EfficientNet + Adapter → per-pattern AUPRC                     │
│       ↓                                                          │
│  按 6 大临床类别聚合 → macro-average AUPRC                      │
│  输出：baseline vs finetuned 对比表                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 已修复的关键 Bug

| Bug | 原因 | 影响 | 修复 |
|---|---|---|---|
| 梯度步长过大（3766%/step） | raw gradient L2 norm ~45，guidance_scale=2.0，步长=90 >> latent norm ~3 | ECG 完全失真，振幅达 14.7mV | Latent-relative step sizing：归一化方向，scale by `0.03 × ‖latent‖` |
| ECG 振幅无限制 | VAE decode 后无 clamp | OOD 值被 ×208 mhi_factor 放大后导致 EfficientNet 梯度爆炸 | `torch.clamp(ecg, -3.0, 3.0)` |
| VAE decoder in-place 操作 | `x /= 0.18215` 破坏计算图 | autograd 无法回传梯度 | 改为 `x = x / 0.18215`（out-of-place） |

**修复后效果**：
- Latent norm ratio: 2.58× → 0.91~1.04×
- ECG max amplitude: 14.7mV → 0.7~1.1mV
- Infarction/Ischemia AUPRC: 0.329 → 0.385 (+17%)

---

## 代码量统计

| 模块 | 行数 | 说明 |
|---|---|---|
| `adversarial/` | ~2,000 | 核心对抗 pipeline |
| `scripts/` | ~1,260 | 实验运行脚本 |
| `util/` | ~2,200 | 工具函数 |
| `center_token/` | ~790 | Center Token 模块 |
| `data/` | ~380 | 数据预处理 |
| **自研代码合计** | **~6,630** | |
| `model/ECGTwin/` | ~数千 | 第三方 submodule |
| `model/DeepECG/` | ~数千 | 第三方 submodule |
| `model/advdiff/` | ~数千 | 第三方 submodule |
