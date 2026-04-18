# ECG Cross-Center v2 Baseline — 训练技术细节

> PTBXL → PN2021/MIMIC 跨中心 gap 量化实验的完整训练技术栈。本文档为下游方法（AugMix / 对抗训练 / 域泛化）提供可复现的技术基线。

---

## 1. 数据与预处理

### 1.1 三数据集规模
| 数据集 | 记录数 | 原始 fs | 用途 |
|---|---:|---:|---|
| PTBXL | 21,799 | 100 Hz | 训练+val+源域测试 |
| PhysioNet 2021 | 61,223 (7 centers) | 500/1000 Hz | 主 OOD |
| MIMIC-IV ECG | 800,035 | 500 Hz | 次 OOD (keyword labels) |

PTBXL 按 `strat_fold`：folds 1-8 训练 (17,418)，fold 9 验证 (2,183)，fold 10 测试 (2,198)。

### 1.2 统一预处理 pipeline（**公平性核心**）

所有三个数据集走**完全一致**的预处理链，使残余 gap 反映真实 domain shift 而非预处理不一致：

```
原始 WFDB/npy  (native fs, 12 导)
       ↓
 [1] NaN 保护 (np.nan_to_num)
       ↓
 [2] 导联重排 → [I, II, III, aVR, aVL, aVF, V1-V6]
     (大小写不敏感，匹配失败则 skip)
       ↓
 [3] filter_bandpass_safe(signal, fs=native_fs)
     - fs > 150Hz: 50Hz notch (iirnotch Q=30) + Butterworth N=4 [0.67, 40]Hz bandpass
     - fs ≤ 150Hz (PTBXL 100Hz): skip notch (Nyquist=50Hz 不稳定), 保留 bandpass [0.67, 47.5]Hz
     - medfilt baseline removal, kernel = int(0.4*fs)+1 (奇数)
     - 零相位 filtfilt, 保证 causality 无畸变
       ↓
 [4] resample → 100Hz (scipy.signal.resample)
       ↓
 [5] pad/truncate → 1000 samples (10s @ 100Hz)
       ↓
 [6] per-sample global z-score
     signal = (signal - signal.mean()) / (signal.std() + 1e-8)
     # 12 导 × time 展平计算单一 mean/std，不是 fit-on-train scaler
       ↓
 [7] crop → 250 samples (2.5s)
     - train: random crop
     - eval:  center crop
       ↓
 输出: (12, 250) float32
```

### 1.3 关键设计决策（避坑）

| 决策 | 理由 |
|---|---|
| **先滤波再下采样** | 滤波器参数按 native fs 设计最稳；500Hz 下 50Hz 工频 notch 才有效，resample 后就失去高频信息 |
| **100Hz 跳过 notch** | PTBXL 100Hz 时 Nyquist=50Hz，iirnotch(50Hz) 极点落在单位圆上不稳定，会产生 NaN |
| **per-sample z-score（不是 StandardScaler）** | fit-on-train 把 PTBXL 均值注入 OOD，让 gap 里混入 scaler-domain-bias；per-sample 自归一化才能让 gap 反映真实分布差异 |
| **crop 250 samples** | 与旧基线对齐；stride 累计 32× 下采样后特征图 ≈ 8 长度，足够 conv 学全局 morphology |
| **filtfilt 容错** | kernel_size > signal length 时自动降级到 skip baseline，防止短信号 crash |

代码：`scripts/crosscenter_v2/preprocess_utils.py`

---

## 2. 标签空间设计（26 类 SNOMED 头）

### 2.1 类别定义

**26 类 = PhysioNet 2021 Challenge 官方 scored classes**，合并 4 组等价对（CLBBB≡LBBB 等），保持与 PN2021 leaderboard 完全兼容。

**3 层汇报**：
- **Tier-1**（5 类）：AF, LBBB, RBBB, IAVB, NSR — 主报告指标，5 大中心都有 ≥5 正样本
- **Tier-2**（15 类）：包含旧基线的 15 类，供直接对比
- **ALL**（26 类）：完整汇报

### 2.2 跨数据集标签映射

| 数据集 | 标签来源 | 覆盖 26 类中 |
|---|---|---:|
| PTBXL | SCP codes（presence-based） | **23/26** (Brady/PRWP/RAD 不覆盖) |
| PN2021 | SNOMED Dx (hea header) | **26/26** |
| MIMIC | machine report 正则关键词 | **26/26** (有噪声) |

### 2.3 Masked 标签约定

PTBXL 不覆盖的 3 类（Brady/PRWP/RAD）标为 **-1**，由 `MaskedFocalLoss` 跳过——既不当正样本也不当负样本。如果当 0 处理会让模型在训练集强学"这 3 类永远是负"，污染 OOD 推理。

### 2.4 关键 bug 修复：PTBXL confidence threshold = 0（不是 50）

**旧基线 bug**：用 `confidence_threshold=50.0`，导致 AFIB 从 1211 → 37（丢 97%）。

**原因**：PTBXL 用 confidence 0 标记"自动诊断系统给出的标签"，仍然是有效正样本。strict threshold=50 只保留人工复核的标签，大量真阳性被丢弃。

**修复**：改用 `confidence_threshold=0`（presence-based）。结果：
- AFIB: 37 → 1,211 (+3178%)
- STACH: 4 → 661
- 匹配 PN2021 (SNOMED binary) 和 MIMIC (keyword binary) 的语义

代码：`scripts/crosscenter_v2/label_alignment_v2.py`

---

## 3. 模型架构

**EfficientNet1DV2 s_v2** (`model/DeepECG/notebooks/EfficientNetv2.py`)

```python
model = EfficientNet1DV2(
    variant='s_v2',          # width=1.0, depth=2.0
    input_channels=12,
    num_classes=26,          # SNOMED scored
    activation='leaky_relu',
    stochastic_depth_prob=0.304,
    dropout_rate=0.0,        # 关掉 dropout，仅用 stochastic_depth
    use_se=True,
    norm_type='batch',
)
```

**参数量**：6,427,288（约 6.4M）

**架构要点**：
- `FusedMBConv1d` 前 4 stage（浅层强 conv）
- `MBConv1d` 后 3 stage（带 expansion + SE + depthwise）
- SE ratio=4，激活 LeakyReLU(0.01)
- 输入 (12, 250)，累计下采样 32× → 特征图长度 ~8
- AdaptiveAvgPool1d(1) → Flatten → Linear(640, 26)

### 3.1 权重初始化（DeepECG `sl_e2e_training.ipynb` cell-8 共识）

```python
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
model.apply(init_weights)
```

**为什么 fan_out**：conv1d 输出通道数远小于 kernel × in_channels，fan_out 让初始尺度更保守，避免深层激活爆炸。

---

## 4. 损失函数

### 4.1 MaskedFocalLoss

```python
class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_value=-1.0):
        ...
    def forward(self, logits, targets):
        mask = (targets != -1).float()
        targets_safe = torch.where(mask.bool(), targets, 0)
        bce = F.binary_cross_entropy_with_logits(logits, targets_safe, reduction='none')
        p_t = sigmoid(logits)*targets_safe + (1-sigmoid(logits))*(1-targets_safe)
        focal = (1 - p_t) ** gamma
        alpha_t = alpha*targets_safe + (1-alpha)*(1-targets_safe)
        loss = alpha_t * focal * bce * mask
        return loss.sum() / mask.sum().clamp(min=1)
```

**超参**：α=0.25, γ=2.0（DeepECG + ECGFounder 两 repo 共识）

**作用**：
- **α=0.25**：负样本权重 0.75，正样本权重 0.25——ECG 大多数类高度不平衡（NSR ~83%），α 把梯度注意力从 dominant 类拉到 hard examples
- **γ=2.0**：focal factor `(1-p_t)^2` 让 easy examples 的梯度衰减 10-100×，模型集中学 hard boundary
- **masked**：-1 标签不参与 loss 和 mean 分母，避免污染

---

## 5. 优化器与调度器

| 组件 | 配置 | 来源 |
|---|---|---|
| Optimizer | AdamW, lr=0.01, weight_decay=0.01 | 旧基线沿用 |
| Scheduler | CosineAnnealingLR(T_max=15, eta_min=1e-4) | **per-epoch** (替换旧 OneCycleLR per-batch) |
| Grad clip | `clip_grad_norm_(max_norm=1.0)` | 防止 focal loss 早期大梯度 |
| AMP | `torch.cuda.amp.autocast + GradScaler` | 显存+速度 |

### 5.1 为什么 CosineAnnealingLR（不是 OneCycleLR）

- DeepECG 两个监督 notebook 和 ECGFounder 都用 Cosine
- OneCycleLR per-batch 对 PTBXL 17k × 50 epoch = 850k steps 过于激进，lr 早期 warmup 顶到 0.01 就掉
- Cosine `T_max=15`：前 15 epoch 从 0.01 平滑降到 eta_min，之后 cosine warm restart（每 15 epoch 一个周期）

**实测 lr 曲线**：
```
Ep 1:  0.00989   (首次下降)
Ep 5:  0.00752
Ep 10: 0.00258
Ep 15: 0.00010   (最低点，即 eta_min)
Ep 16: 0.00021   (开始上升，warm restart)
Ep 20: 0.00257
Ep 25: 0.00753   (回到高 lr)
```

**观察**：最佳 val Tier-1 AUROC 在 **ep 15**（lr 最低点），之后 warm restart 未再改进 → patience=10 在 ep 25 触发 early stop。

### 5.2 超参汇总

| 超参 | 值 | 说明 |
|---|---:|---|
| batch_size | 128 | 4090 24GB 显存充足 |
| epochs | 50 (max) | 实际 ep 25 早停 |
| patience | 10 | 监控 val Tier-1 AUROC (不是 val loss) |
| seed | 42 | 可复现 |
| num_workers | 4 | DataLoader 并行 |

---

## 6. 训练循环关键点

### 6.1 Val 监控指标

**val loss 而是 val Tier-1 AUROC**——loss 在 focal 下很平，AUROC 对最佳 checkpoint 更有判断力。

### 6.2 预处理缓存

PTBXL 全量预处理结果缓存到 `/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy` (1GB)：
- 首次运行 ~2 min 处理 21,799 条
- 之后 reload 只需秒级

代码：`preprocess_ptbxl_all()` in `train_ptbxl_v2.py`

### 6.3 训练曲线（关键 checkpoint）

| Epoch | train_loss | val Tier-1 AUROC | val Tier-1 AUPRC | 备注 |
|------:|---:|---:|---:|---|
| 1  | 0.0327 | 0.5990 | 0.2487 | 初始化 |
| 5  | 0.0131 | 0.8979 | 0.5541 | 快速收敛 |
| 10 | 0.0103 | 0.9655 | 0.7461 | |
| **15** | **0.0091** | **0.9724** | **0.8152** | **best** (lr 最低) |
| 20 | 0.0093 | 0.9683 | 0.8014 | warm restart 后抖动 |
| 25 | 0.0095 | 0.9710 | 0.8022 | early stop |

**训练速度**：~12s/epoch on RTX 4090，总训练时间 ~5 min。

---

## 7. 评测方案

### 7.1 指标

| 指标 | 原因 |
|---|---|
| Macro AUROC | 排序能力，对不平衡鲁棒 |
| **Macro AUPRC** | 精度-召回曲线下面积，对不平衡**更敏感**——本问题的主要 headline 指标 |
| Bootstrap n=1000 95% CI | PN2021 每中心；MIMIC n=100（大 N 开销） |

### 7.2 Bootstrap 实现

```python
def bootstrap_tiered(y_true, y_score, indices, n_bootstrap=1000):
    point = compute_tiered(y_true, y_score, indices)
    rng = np.random.RandomState(42)
    boot = [compute_tiered(y_true[rng.randint(0,n,n)], ...) for _ in range(n_bootstrap)]
    point['auroc_ci_low']  = np.percentile(boot_auroc, 2.5)
    point['auroc_ci_high'] = np.percentile(boot_auroc, 97.5)
    return point
```

**注意**：对大 N（MIMIC 800k），AUROC/AUPRC 各约 500ms，`n_bootstrap=100` 已经 ~77 min。1000 次会到 12 小时。**大 N 下 CI 本来也非常窄**，100 次已经精确到 ±0.002。

### 7.3 Masked 类处理

eval 时对 label==-1 的类**逐 sample 过滤**后计算 AUROC：
```python
col_t = y_true[:, i]
valid = col_t != -1.0
auc = roc_auc_score(col_t[valid], y_score[valid, i])
```

PTBXL 测试集对 Brady/PRWP/RAD 三类输出 `N/A`，不进入 macro 平均。

---

## 8. 工程细节

### 8.1 路径约定
- 代码：`/root/ECG_adv_Gen/scripts/crosscenter_v2/`
- 产出：`/root/autodl-tmp/crosscenter_v2/` (28GB free，model + cache + jsons)
- 报告：`/root/ECG_adv_Gen/outputs/gap_report_v2.md`

### 8.2 Python unbuffered 输出
后台跑 WFDB 读档时 stdout 会被 Python 行缓冲。用：
```bash
PYTHONUNBUFFERED=1 nohup python script.py > log.log 2>&1 &
```
否则 `tail -f log.log` 会看不到实时进度（scanning 阶段可能静默 2 分钟）。

### 8.3 WFDB 流式读取
MIMIC 800k 不能全量加载内存，用 `MIMICStreamDataset.__getitem__` 逐条 `wfdb.rdrecord`：
- DataLoader `num_workers=8` 并行
- failed 记录返回 valid=0 标志位，inference 后过滤
- 800k 记录 @ 203 rec/s = 66 min 推理

---

## 9. 复现命令

```bash
# 1. 训练 (生成 best_model.pt, ~5-10 min)
/root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_v2/train_ptbxl_v2.py \
  --epochs 50 --patience 10 \
  --output_dir /root/autodl-tmp/crosscenter_v2

# 2. PN2021 eval (~20 min)
/root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_v2/eval_crosscenter_v2.py \
  --model_dir /root/autodl-tmp/crosscenter_v2 \
  --data_dir datasets/physionet2021

# 3. MIMIC eval (~80 min)
/root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_v2/eval_mimic_zeroshot.py \
  --model_dir /root/autodl-tmp/crosscenter_v2 \
  --data_dir datasets/MIMIC --n_bootstrap 100

# 4. 生成 gap report
/root/miniforge3/envs/ECGTwin/bin/python scripts/crosscenter_v2/build_gap_report.py \
  --model_dir /root/autodl-tmp/crosscenter_v2
```

---

## 10. 关键技术对比（v2 vs 旧基线）

| 方面 | 旧基线 (`train_ptbxl.py`) | v2 (`train_ptbxl_v2.py`) |
|---|---|---|
| 类数 | 15 | **26** (PN2021 官方 scored) |
| 预处理 | fit-on-train StandardScaler（注入 bias） | **per-sample z-score** (无 bias) |
| 滤波 | 无 | **bandpass + notch + baseline** |
| Loss | BCEWithLogitsLoss | **MaskedFocalLoss(α=0.25, γ=2)** |
| Scheduler | OneCycleLR (per-batch) | **CosineAnnealingLR(T_max=15)** (per-epoch) |
| Dropout | 0.5 | **0.0** (仅 stochastic_depth) |
| Init | PyTorch 默认 | **Kaiming fan_out + Xavier + BN 1/0** |
| SCP 阈值 | confidence≥50 | **presence-based (≥0)** |
| 最佳 val AUROC (15-class 对比) | 0.9079 | **0.9238** (+1.6 pp) |

---

## 11. 后续扩展点

此 baseline 已冻结为 `best_model.pt`，所有下游方法**必须保持预处理一致**才能公平对比：

1. **AugMix 变种**：在 PTBXL 训练时加入混合增强（保守策略，避免破坏 ECG 波形生理意义）
2. **在线对抗训练**：PGD/FGM 攻击在 PN2021 validation split 上，强化 boundary-hard 样本
3. **域泛化方法**：DRO, IRM, CORAL（架构不变）
4. **ECGTwin 合成样本**：条件生成跨中心 ECG，扩充训练集

每种方法的成功指标：**gap 缩小幅度**（相对本 baseline 的 4.02pp Tier-1 AUROC / 13.72pp AUPRC 减少多少）。
