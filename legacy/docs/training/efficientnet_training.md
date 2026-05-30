# EfficientNet1DV2 ECG 分类：模型架构 + 训练 + 评测 技术文档

## 源码链接

| 文件 | 作用 |
|------|------|
| [`EfficientNetv2.py`](../../model/DeepECG/notebooks/EfficientNetv2.py) | 模型架构定义（6 个类：CustomNorm、SEBlock、StochasticDepth、FusedMBConv1d、MBConv1d、EfficientNet1DV2） |
| [`train_ptbxl.py`](./train_ptbxl.py) | 基线训练脚本（无增强，仅随机裁剪） |
| [`train_ptbxl_augmix.py`](./train_ptbxl_augmix.py) | AugMix 消融训练脚本 |
| [`eval_crosscenter.py`](./eval_crosscenter.py) | PhysioNet2021 跨中心评测脚本 |
| [`label_alignment.py`](./label_alignment.py) | 15 类标签定义 + PTBXL SCP / PhysioNet2021 SNOMED 映射 |
| [`ecg_augmix.py`](./ecg_augmix.py) | ECG-AugMix 增强模块（8 个算子 + JSD 损失） |

---

## 1. EfficientNet1DV2 模型架构

### 1.1 设计来源

EfficientNet1DV2 是 EfficientNetV2（Tan & Le, ICML 2021）从 2D 图像到 1D ECG 信号的完整适配。原始 EfficientNetV2 的核心改进（相对于 V1）是在网络早期阶段用 **FusedMBConv**（融合卷积）替换 MBConv（深度可分离卷积），减少了低分辨率特征图上深度卷积的内存访问瓶颈。

### 1.2 1D 适配：什么被改了

所有 2D 操作被替换为 1D 等价物：

| 2D 原版 | 1D 适配 |
|---------|---------|
| `Conv2d` | `Conv1d` |
| `BatchNorm2d` | `BatchNorm1d` / `GroupNorm` / `LayerNorm` / `InstanceNorm1d`（通过 `CustomNorm` 统一包装） |
| `AdaptiveAvgPool2d(1)` | `AdaptiveAvgPool1d(1)` |
| 输入 `(B, 3, H, W)` | 输入 `(B, 12, T)`（12 导联 ECG，T=250 采样点） |

模型支持 4 种归一化方式（通过 `norm_type` 参数切换），当前使用 `'batch'`（BatchNorm1d）。

### 1.3 网络结构总览

```
输入 (B, 12, 250)
  │
  ├── Stage 0-3: FusedMBConv1d 块（早期阶段，融合卷积）
  │     单次 Conv1d → Norm → 激活 → SE → Dropout → 残差连接 + StochasticDepth
  │
  ├── Stage 4-6: MBConv1d 块（后期阶段，深度可分离卷积）
  │     扩展 Conv1d → Norm → 激活 → 深度卷积 → SE → 投影 Conv1d → 残差 + StochasticDepth
  │
  ├── Final Conv1d → Norm → 激活
  │
  └── AdaptiveAvgPool1d(1) → Flatten → Dropout(0.5) → Linear(640, 15)
        │
        └── 输出 (B, 15) raw logits
```

### 1.4 FusedMBConv1d 块（Stage 0-3）

**核心思想**：把 MBConv 的"1×1 扩展 + 深度卷积"合并成一个标准卷积，在低分辨率（通道数少）的早期阶段更高效。

数据流：
1. `Conv1d(in → out, kernel, stride, padding)` — 融合的扩展+空间卷积
2. `CustomNorm(out)` — 归一化
3. 激活函数（LeakyReLU）
4. `SEBlock(out)` — Squeeze-and-Excitation 通道注意力（可选）
5. `Dropout`（可选）
6. 残差连接 + StochasticDepth（仅当 in==out 且 stride==1 时）

### 1.5 MBConv1d 块（Stage 4-6）

**核心思想**：Mobile Inverted Bottleneck——先用 1×1 卷积扩展通道，再用深度卷积处理空间信息，最后用 1×1 投影回目标通道。深度卷积 `groups=channels` 使得每个通道独立卷积，大幅减少参数。

数据流：
1. `Conv1d(in → in*expansion, 1)` — 1×1 扩展（当 expansion > 1）
2. `Norm → 激活`
3. `Conv1d(mid, mid, kernel, stride, groups=mid)` — **深度卷积**（核心）
4. `Norm → 激活`
5. `SEBlock(mid)` — 通道注意力
6. `Conv1d(mid → out, 1)` — 1×1 投影
7. `Norm`
8. `Dropout` + 残差 + StochasticDepth

### 1.6 SE（Squeeze-and-Excitation）模块

通道级注意力机制：学习每个特征通道的重要性权重。

数据流：
1. `AdaptiveAvgPool1d(1)` — 全局平均池化，压缩时间维度 `(B, C, T) → (B, C, 1)`
2. `Conv1d(C → C/ratio, 1)` — 压缩到 1/4 通道（ratio=4）
3. `ReLU`
4. `Conv1d(C/ratio → C, 1)` — 恢复通道数
5. `Sigmoid` — 生成 [0,1] 通道门控权重
6. 逐元素乘：`output = input × gate`

**作用**：让网络对 ECG 中不同频率/形态特征的通道做自适应加权——例如对 QRS 相关通道赋高权重、对噪声通道赋低权重。

### 1.7 Stochastic Depth（随机深度）

训练时以概率 `p=0.304`（s_v2 默认）**随机丢弃整个残差分支**，保留 identity。

效果：
- 隐式集成——训练了指数级多个"子网络"
- 正则化——防止对深层残差路径过拟合
- 推理时关闭（确定性前向）

实现公式：`output = identity + residual × mask / keep_prob`（其中 `mask ∈ {0, 1}`，`keep_prob = 1 - p`）

### 1.8 变体配置（Variant Configs）

通过 `get_backbone_config(variant)` 获取宽度/深度系数：

| 变体 | 宽度系数 | 深度系数 | 适用场景 |
|------|---------|---------|---------|
| **`s_v2`** | 1.0 | **2.0** | **当前使用**（Small，块数翻倍） |
| `m_v2` | 1.1 | 2.1 | Medium |
| `l_v2` | 1.2 | 2.2 | Large |
| `b0_v2` ~ `b7_v2` | 1.0~2.0 | 1.0~3.1 | EfficientNet 系列 |

- **宽度系数**：缩放每个 stage 的通道数（`channels = int(base_channels × width_coef)`）
- **深度系数**：缩放每个 stage 的块重复次数（`depths = ceil(base_depths × depth_coef)`）

`s_v2` 的意思是：通道数不变（1.0x），但每个 stage 的块数**翻倍**（2.0x），从而加深网络。

### 1.9 各 Stage 输入/输出维度（s_v2 变体）

| Stage | 块类型 | 输入通道 | 输出通道 | Stride | 块数(1x) | 块数(2x) | 输出形状 |
|-------|--------|---------|---------|--------|---------|---------|---------|
| Init | Conv1d | 12 | 12 | 1 | - | - | (B, 12, 250) |
| 0 | FusedMBConv | 12 | 12 | 1 | 1 | 2 | (B, 12, 250) |
| 1 | FusedMBConv | 12 | 24 | 1 | 1 | 2 | (B, 24, 250) |
| 2 | FusedMBConv | 24 | 32 | 2 | 2 | 4 | (B, 32, 125) |
| 3 | FusedMBConv | 32 | 64 | 2 | 2 | 4 | (B, 64, 62) |
| 4 | MBConv | 64 | 80 | 2 | 3 | 6 | (B, 80, 31) |
| 5 | MBConv | 80 | 128 | 2 | 4 | 8 | (B, 128, 15) |
| 6 | MBConv | 128 | 640 | 2 | 5 | 10 | (B, 640, 7) |
| Final | Conv1d | 640 | 640 | 1 | - | - | (B, 640, 7) |
| Pool | AvgPool1d | 640 | 640 | - | - | - | (B, 640, 1) |
| Head | Linear | 640 | 15 | - | - | - | (B, 15) |

**总参数量**：约 6,420,237（s_v2，15 类头）

### 1.10 分类头

```
AdaptiveAvgPool1d(1)  — 全局平均池化 (B, 640, 7) → (B, 640, 1)
Flatten              — 去时间维 (B, 640, 1) → (B, 640)
Dropout(0.5)         — 正则化
Linear(640, 15)      — 分类层 → (B, 15) raw logits
```

输出是 **raw logits**，不经过 sigmoid/softmax。训练时 `BCEWithLogitsLoss` 内部做 sigmoid；推理时手动 `sigmoid(logits)` 得到每类概率。

---

## 2. 15 类标签体系

### 2.1 标签定义

从 PTBXL SCP 编码体系和 PhysioNet2021 SNOMED 编码体系中筛选出 **15 个跨数据集共有的心电诊断类别**：

| 索引 | 缩写 | 全称 | PTBXL SCP 码 | SNOMED 码 |
|------|------|------|-------------|-----------|
| 0 | AF | 房颤 | AFIB | 164889003 |
| 1 | AFL | 房扑 | AFLT | 164890007 |
| 2 | BBB | 束支传导阻滞 | CLBBB/CRBBB/ILBBB | 6374002 |
| 3 | LBBB | 左束支传导阻滞 | CLBBB | 733534002, 164909002 |
| 4 | RBBB | 右束支传导阻滞 | CRBBB | 713427006, 59118001 |
| 5 | IAVB | 一度房室传导阻滞 | 1AVB | 270492004 |
| 6 | IRBBB | 不完全性右束支阻滞 | IRBBB | 713426002 |
| 7 | LAD | 左轴偏移 | LAFB | 39732003 |
| 8 | LAnFB | 左前分支阻滞 | LAFB | 445118002 |
| 9 | LQT | 长 QT 综合征 | LNGQT | 111975006 |
| 10 | NSIVCB | 非特异性室内传导阻滞 | IVCD | 698252002 |
| 11 | NSR | 正常窦性心律 | NORM/SR | 426783006 |
| 12 | PVC | 室性早搏 | PVC | 427172004, 17338001 |
| 13 | PR | 起搏心律 | PACE | 10370003 |
| 14 | TAb | T 波异常 | NDT/NST_ | 164934002 |

### 2.2 标签转换

- **PTBXL → 15 类**：`ptbxl_scp_to_labels(scp_codes, confidence_threshold=50.0)` — 读取 SCP 编码字典，置信度 ≥ 50% 的映射到对应类别
- **PhysioNet2021 → 15 类**：`snomed_to_labels(snomed_codes)` — 读取 SNOMED 编码列表，直接映射

两者输出都是 `(15,)` 的 float32 二值向量。**多标签**：一条 ECG 记录可以同时属于多个类（例如 LBBB + IAVB）。

---

## 3. 数据 Pipeline

### 3.1 PTBXL 数据加载

| 步骤 | 操作 | 输出形状 |
|------|------|---------|
| 原始数据 | `np.load('raw100.npy')` — 21,799 条记录 @ 100Hz, 10s | (21799, 1000, 12) |
| Fold 切分 | `ptbxl_database.csv` 的 `strat_fold` 列（1-10） | train: folds 1-8, val: fold 9, test: fold 10 |
| 标准化 | `StandardScaler().fit(train.reshape(-1, 12))` — **仅在 train 上 fit** | 每导联零均值单位方差 |
| 保存 Scaler | `pickle.dump(scaler, 'scaler.pkl')` — 跨中心评测复用 | — |

### 3.2 数据裁剪策略

每条记录是 10 秒 1000 采样。训练/评测时裁剪到 2.5 秒 250 采样：

- **训练**：从 [0, 750] 中**随机采样**起始位置 → 每 epoch 看到不同窗口，相当于**免费的数据增强**
- **评测**：固定**居中裁剪**（起始 = 375）→ 保证可复现

裁剪后转置为 `(12, 250)`（通道优先，Conv1d 需要）。

### 3.3 PhysioNet2021 数据加载（跨中心评测）

跨中心评测的信号来源完全不同（WFDB 格式、不同采样率、不同导联顺序），需要额外的预处理：

| 步骤 | 操作 |
|------|------|
| 读取 | `wfdb.rdrecord(path)` — 读取 .mat + .hea 文件 |
| NaN 处理 | `nan_to_num(signal, nan=0.0)` |
| 截断 | 取前 10 秒（`signal[:10*fs]`） |
| 重采样 | `scipy.signal.resample()` 到 100Hz |
| 填充/截断 | 统一到 1000 采样 |
| 导联重排 | 按标准 12 导联顺序重排（I, II, III, aVR, aVL, aVF, V1-V6） |
| 标准化 | 用**训练时保存的** `scaler.pkl` 做 transform |
| 裁剪 | 居中裁剪到 250 采样 |

### 3.4 评测的 7 个 PhysioNet2021 中心

| 中心名 | 地区 | 典型记录数 |
|--------|------|-----------|
| chapman_shaoxing | 中国绍兴 | ~5,648 |
| cpsc_2018 | 中国 CPSC 竞赛 | ~4,735 |
| cpsc_2018_extra | CPSC 额外数据 | ~663 |
| georgia | 美国乔治亚 | ~7,234 |
| ningbo | 中国宁波 | ~19,182 |
| ptb | 德国 PTB 数据库 | ~115 |
| st_petersburg_incart | 俄罗斯圣彼得堡 | ~6 |

---

## 4. 训练策略

### 4.1 基线训练（`train_ptbxl.py`）

| 项目 | 值 | 说明 |
|------|---|------|
| 优化器 | **AdamW** | lr=0.01, weight_decay=0.01 |
| 调度器 | **OneCycleLR** | max_lr=0.01, 每 batch 更新 |
| 损失函数 | **BCEWithLogitsLoss** | 多标签二元交叉熵 |
| 混合精度 | **AMP** (`torch.cuda.amp`) | fp16 前向 + fp32 反传 |
| 梯度裁剪 | max_norm=**1.0** | 防梯度爆炸 |
| Batch size | **128** | |
| 最大 Epochs | **50** | |
| 早停 | patience=**10** | 监控 val_loss |
| 数据增强 | **仅随机裁剪** | 无噪声/缩放/遮蔽等 |

**训练循环每步**：
1. 取 batch `(signals, labels)` → GPU
2. `optimizer.zero_grad()`
3. AMP autocast 下前向 `logits = model(signals)`
4. `loss = BCEWithLogitsLoss(logits, labels)`
5. `scaler.scale(loss).backward()`
6. `scaler.unscale_(optimizer)` + `clip_grad_norm_(1.0)`
7. `scaler.step(optimizer)` + `scaler.update()`
8. `scheduler.step()`

### 4.2 AugMix 训练变体（`train_ptbxl_augmix.py`）

**仅有三处改动**，其余与基线字节级一致：

| 改动 | 基线 | AugMix |
|------|------|--------|
| Dataset 输出 | `(clean, label)` 2-tuple | `(clean, aug1, aug2, label)` 4-tuple |
| 前向 | `model(signals)` 单视图 | `model(cat([clean, aug1, aug2]))` 三视图 concat |
| 损失 | `BCE(logits, labels)` | `BCE(logits_clean, labels) + λ × JSD(clean, aug1, aug2)` |

`ecg_augmix` 的调用发生在 DataLoader worker 中（`PTBXLDataset.__getitem__`），由 `num_workers=4` 并行化。

AugMix 专属超参数：

| 参数 | 默认 | 作用 |
|------|------|------|
| `augmix_severity` | 3 | 增强强度（1-10） |
| `augmix_width` | 3 | Dirichlet 混合的链数 |
| `jsd_weight` | 6.0（脚本默认） | JSD 一致性损失权重 λ |

---

## 5. 评测方法

### 5.1 同分布评测（PTBXL fold 10）

- 加载 `best_model.pt`（val_loss 最优 checkpoint）
- 在 fold 10 test set 上跑推理
- 对每类计算 `roc_auc_score`（跳过正例 < 2 的类）
- 报告 **macro-AUROC**（所有有效类的 AUROC 均值）

### 5.2 跨中心评测（PhysioNet2021）

- 加载同一个 `best_model.pt` + `scaler.pkl`
- 对 7 个 PhysioNet2021 子库分别：
  1. 扫描 `.hea` 文件获取 SNOMED 编码
  2. 加载并预处理信号（重采样 + 标准化 + 裁剪）
  3. 模型推理
  4. 计算 **macro-AUROC + 95% bootstrap 置信区间**（100 次重采样）
  5. 计算 **per-class AUROC**（正例 ≥ 5 的类）
- 输出汇总表 + 详细 per-class 矩阵
- 保存到 `eval_crosscenter.json`

### 5.3 Bootstrap 置信区间

对每个中心的 AUROC 计算 95% CI：
1. 从 N 条记录中有放回采样 N 条（100 次）
2. 每次采样计算 macro-AUROC
3. 取 2.5th 和 97.5th 百分位作为 CI 下界和上界

---

## 6. 模型实例化参数

当前跨中心实验使用的精确配置：

```
EfficientNet1DV2(
    variant='s_v2',                 # 宽度 1.0x + 深度 2.0x
    input_channels=12,              # 12 导联 ECG
    num_classes=15,                 # 15 类多标签
    activation='leaky_relu',        # LeakyReLU(α=0.01)
    stochastic_depth_prob=0.304,    # 随机深度 drop rate
    dropout_rate=0.5,               # 分类头 dropout
    use_se=True,                    # 启用 SE 通道注意力
    norm_type='batch',              # BatchNorm1d
)
```

总参数量：**6,420,237**

---

## 7. 产物文件

训练完成后 `output_dir/` 内容：

| 文件 | 大小 | 说明 |
|------|------|------|
| `best_model.pt` | ~25 MB | 模型权重（`state_dict`） |
| `scaler.pkl` | ~700 B | StandardScaler（跨中心评测必需） |
| `training_log.json` | ~6 KB | 每 epoch 的 train_loss / val_loss / val_auroc / lr / time |
| `train_result.json` | ~400 B | 最终 test 指标 + 训练配置 |
| `eval_crosscenter.json` | ~8 KB | 7 个中心的 macro-AUROC + CI + per-class 详情 |

---

## 8. 基线实验结果

| 指标 | 值 |
|------|---|
| PTBXL fold 10 test macro-AUROC | **0.9079** |
| 训练 epochs | 49（patience=10 触发早停） |
| 每 epoch 耗时 | ~4 秒 |
| 跨中心平均 macro-AUROC (6 中心, excl incart) | **0.7655** |
| 最强跨中心 | ningbo: 0.8196 |
| 最弱跨中心（有统计意义） | cpsc_2018_extra: 0.6716 |

完整跨中心结果详见 `/root/autodl-tmp/crosscenter/eval_crosscenter.json`。
