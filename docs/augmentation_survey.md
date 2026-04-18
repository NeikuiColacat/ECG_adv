# ECG-AugMix 调研报告：将 AugMix 框架迁移到 12 导联 ECG 多标签疾病分类任务

## 概述

**目前尚无已发表的工作直接将 AugMix 的"多链混合 + JSD 一致性损失"框架迁移到一维 ECG 信号领域**，这是一个明确的研究机会。最接近的工作包括 TaskAug（Raghu 等, CHIL 2022）、RandECG（Nonaka & Seita, 2022）和 Non-Uniform-Mix（ECG-Mamba, 2025），但均未实现完整的 AugMix 流程。

本报告整理了 **40+ 种 ECG 数据增强操作**，分为七大类别，评估了各操作对 77 类疾病多标签分类的安全性，并提出了完整的 ECG-AugMix 架构设计和 PyTorch 实现。

**核心发现：** 增强效果高度依赖任务——同一操作可能提升某些疾病的检测性能，却降低另一些疾病的检测性能，这使得 AugMix 的"通过混合实现多样性"策略尤其适合 77 类多标签 ECG 分类。

---

## 一、40+ 种增强操作按类别整理

### 1.1 时域增强

| 操作名称 | 说明 | 参数范围 |
|---------|------|---------|
| 时间扭曲（Time Warping） | 用三次样条插值变形时间轴，保留波形形态同时引入速率变化 | σ ≈ 0.2，4个结点 |
| 窗口扭曲（Window Warping） | 选取随机 10% 片段局部加速(2×)或减速(0.5×)，再缩放回原始长度 | Iwana 等发现该方法效果突出 |
| 时间平移（Temporal Shifting） | 将整个信号平移随机偏移量 dt | dt ~ Uniform(-s, s) |
| 窗口裁剪（Window Slicing） | 截取 50-90% 的连续窗口并插值回原始长度 | 窗口比例 50-90% |
| STAR（正弦时间-幅度重采样） | 按心拍对R-R段进行正弦扭曲+幅度缩放，保留P-QRS-T顺序。**micro-AUROC 从 0.86 提升到 0.95** | Nemati 等, 2025 |
| 时间遮蔽（1D Cutout） | 将信号的连续片段置零，模拟信号缺失 | 遮蔽比例 10-50% |
| 重采样（Resampling） | 模拟不同设备的采样率差异 | 采样率变化范围 |

### 1.2 幅度域增强

| 操作名称 | 说明 | 参数范围 |
|---------|------|---------|
| 幅度缩放（Magnitude Scaling） | 将信号乘以随机系数 α，模拟不同设备的增益差异 | α ~ Uniform(0.8, 1.2) |
| 幅度抖动（Amplitude Jittering） | 添加独立同分布噪声：x'(t) = x(t) + ε | ε ~ N(0, σ²) |
| 幅度扭曲（Magnitude Warping） | 用三次样条插值生成平滑的时变缩放曲线 | 结点 ~ N(1, 0.2²) |
| Sigmoid 压缩 | 将信号通过 sigmoid 非线性函数进行幅度压缩 | Hatamian 等报告 F1 +2.04% |
| 心拍级幅度改变 | 对每个心拍独立随机缩放，保留整体时间线 | Lee 等, 2021 |

### 1.3 噪声注入增强

| 操作名称 | 说明 | 参数范围 |
|---------|------|---------|
| 高斯噪声 | 在受控 SNR 下添加高斯噪声，模拟传感器噪声 | SNR 15-40 dB |
| 基线漂移（Baseline Wander） | 添加低频正弦分量（0.05-0.5 Hz），模拟呼吸伪影 | 幅度达满偏转的 15% |
| 工频干扰（Powerline） | 注入 50/60 Hz 正弦波及其谐波 | QRS 幅值的 1-10% |
| 肌电噪声（EMG Noise） | 添加宽带骨骼肌伪影（>10 Hz） | 可控幅度 |
| 电极运动伪影 | 模拟皮肤-电极界面中断引起的瞬态高幅度干扰 | 短暂突发式 |

### 1.4 频域增强

- **SpecAugment**（来自语音领域）：对 STFT 的时频带独立遮蔽，然后反变换。Raghu 等发现它提升了右心室肥大（RVH）的检测，但**降低了房颤（AFib）的检测**——充分说明了任务依赖性
- **频率扭曲**：类似时间扭曲但作用于频率轴
- **带通扰动**：随机变化预处理的滤波器截止频率
- **频率遮蔽**：在傅里叶域中将特定频段置零

### 1.5 导联特异性增强（12导联专用）

| 操作名称 | 说明 | 效果 |
|---------|------|-----|
| 随机导联遮蔽（RLM） | 以概率 p 将整个导联置零，提高对部分导联配置的鲁棒性 | p = 0.1-0.5/导联 |
| 肢体电极偏移模拟 | 用导联向量模型模拟轻微电极放置偏差 | PTB-XL 上 AUC +3.5% |
| 胸导联电极偏移 | 用相邻导联的线性插值计算新导联：V'₁ = (1-β)V₁ + βV₂ | 模拟电极放置变异 |
| 3KG 心电向量变换 | 转换为 3D 心电向量空间，施加旋转/缩放，再反投影回 12 导联 | 1% 标记数据下 AUC +9.1% |
| SIMVEA | 利用胸导联角度关系生成增强的胸导联 | Lim 等, 2024 |

### 1.6 混合类增强（对你的 AugMix 设计最相关）

| 操作名称 | 说明 | 注意事项 |
|---------|------|---------|
| 1D Mixup | 线性插值信号和标签：x̃ = λxᵢ + (1-λ)xⱼ | PTB-XL 前 28 名均使用混合方法 |
| 1D CutMix | 将一个信号的连续时间段替换为另一个信号的，按比例混合标签 | 类似 CutMix 但在时间轴上操作 |
| Non-Uniform-Mix | 渐进应用 MixUp：第1个epoch 20% 参与，第5个epoch增加到 80% | **均匀混合反而会损害性能！** |
| 流形混合（Manifold Mixup） | 在网络中间层插值，而非输入级别 | 更平滑的表示混合 |
| 随机擦除（Random Erasing） | 将连续区域替换为随机噪声或零 | 类似图像的 Random Erasing |

> ⚠️ **重要发现：** ECG-Mamba 研究表明，均匀 MixUp 实际上会降低 ECG 分类性能（AUPRC 从 0.6100 降至 0.6042），必须使用渐进式应用策略。

### 1.7 生成式增强

包括基于 GAN 的方法（DC-GAN, WGAN-GP, AC-WGAN-GP）和基于扩散模型的方法（DiffECG, DSAT-ECG, BioDiffusion）。GAN 方法通常能提升 6-7% 的准确率，少数类 F1 超过 98%。扩散模型作为更新的替代方案正在崭露头角。另外 SMOTE 通过插值少数类样本也是常用手段。

---

## 二、哪些增强保留 vs 破坏诊断标签

### 2.1 普遍安全的操作

- **轻度高斯噪声**（σ ≤ 0.01-0.05，SNR ≥ 25 dB）：模拟真实传感器变异，不改变形态
- **小幅时间平移和循环滚动**：扰动起始对齐，不截断心拍
- **适度基线漂移**（<0.2 mV, 频率 <1 Hz）：模拟呼吸伪影，不改变心拍内形态

这三类操作在合理参数范围内保留所有诊断标签，建议作为默认基础操作。

### 2.2 有条件安全的操作（需要标签感知应用）

**幅度缩放：** 保留相对形态但改变绝对电压阈值。对左心室肥大（LVH）、右心室肥大（RVH）、低电压 ECG 和 STEMI 危险。安全范围：±3-5%。

**时间扭曲：** 最微妙的操作。拉伸 10% 会将 400ms 的 QT 间期变为 440ms，从正常变成长 QT。拉伸 3% 会将 195ms 的 PR 间期变为 201ms，越过一度房室传导阻滞的 200ms 阈值。QRS 宽度阈值（120ms，用于 LBBB/RBBB 诊断）同样脆弱。最大安全范围：±5%，对间期依赖性诊断使用标签条件排除。

**导联丢弃：** 破坏区域特异性心梗信息（II/III/aVF 对应下壁心梗，V1-V4 对应前壁心梗）。安全概率：p ≤ 0.1/导联，最多丢 2-3 个导联。

### 2.3 危险操作——避免或极度谨慎

- **极性翻转（垂直翻转）：** 反转心脏电轴，破坏轴偏移诊断、Q波心梗模式、束支传导阻滞形态和 P 波极性——实际上会破坏 77 个类别中的大多数
- **时间反转（水平翻转）：** 反转 P-QRS-T 顺序。没有任何心脏病理会产生反转顺序——产生的是生理学上不可能的信号
- **段置换和导联间洗牌：** 前者破坏节律模式和心拍内形态，后者破坏导联间的空间关系。两者均一致有害，应从增强池中排除

### 2.4 疾病特异性排除规则

| 疾病类别 | 排除的增强操作 | 原因 |
|---------|-------------|------|
| 长/短 QT 综合征 | 时间扭曲 >5% | QT 间期跨越诊断阈值 |
| 房室传导阻滞 (1°/2°/3°) | 时间扭曲 >3% | PR 间期阈值为 200ms |
| LBBB/RBBB/IVCD | 时间扭曲 >5% | QRS 宽度阈值为 120ms |
| STEMI/NSTEMI | 幅度缩放 >5%、大基线漂移 | ST 抬高使用绝对电压阈值 |
| LVH/RVH | 幅度缩放 >3%、极性翻转 | 电压标准有方向性 |
| 房颤/房扑 | SpecAugment、置换、强时间遮蔽 | 节律规律性是关键特征 |
| Brugada 模式 | 导联丢弃(V1-V2) | V1-V2 的穹顶型 ST 是诊断特征 |
| 窦性心动过缓/过速 | 时间扭曲 >5% | 心率阈值 60/100 bpm |
| 电轴偏移 (LAD/RAD) | 极性翻转、导联重排 | 电轴判断需要正确导联极性 |
| 低电压 ECG | 任何幅度缩放 | 定义特征就是幅度 |

> 对于多标签样本，排除列表取所有单独疾病排除的**并集**。

---

## 三、ECG-AugMix 架构设计

### 3.1 AugMix 框架适配 1D ECG 的三个关键修改

1. **用 ECG 专用操作池替换图像操作池**
2. **调整链深度以保护脆弱的形态特征**——ECG 信号比图像更脆弱
3. **将 JSD 重新公式化为多标签 sigmoid 输出**

**操作池分层：**

- **第一层（始终包含）：** 高斯噪声、幅度缩放、基线漂移、时间平移
- **第二层（谨慎包含）：** 工频噪声、随机遮蔽、幅度扭曲、导联丢弃
- **第三层（任务特定）：** STAR、SpecAugment、频率丢弃

**链结构：** k=3 条链（与 AugMix 默认一致），但深度 d 从 [1,2] 中采样而非 [1,3]——ECG 信号比图像更脆弱，深链复合会导致形态失真。混合权重服从 Dirichlet(α=1.0)。跳连接权重 m ~ Beta(2, 2)，偏向保留更多原始信号。

### 3.2 多标签 JSD 一致性损失

原始 AugMix 的 JSD 作用于 softmax 分布。对于多标签 ECG，将每个标签视为独立的 Bernoulli 分布：

```
JSD_j = JSD(σ(z_orig_j) || σ(z_aug1_j) || σ(z_aug2_j))
L_consistency = (1/C) Σ_j JSD_j
L_total = L_BCE(z_orig, y) + λ · L_consistency

建议 λ = 6-12，从 6 开始调参
```

### 3.3 严重度映射表

| 严重度 | 高斯噪声 σ | 缩放范围 | 时间扭曲 % | 基线漂移幅度 | 遮蔽宽度(秒) |
|-------|-----------|---------|----------|-----------|------------|
| 1 | 0.002 | 0.98-1.02 | ±1% | 1% | 0.02 |
| 3 | 0.010 | 0.93-1.07 | ±3% | 3% | 0.06 |
| 5 | 0.020 | 0.88-1.12 | ±6% | 6% | 0.10 |
| 7 | 0.040 | 0.82-1.18 | ±10% | 10% | 0.14 |
| 10 | 0.080 | 0.75-1.25 | ±15% | 15% | 0.20 |

> 📌 **推荐操作范围：严重度 3-5**，足够提供有意义的正则化，同时对大多数疾病类别保持安全。

---

## 四、性能基准与最佳实践

### 4.1 单一增强对 12 导联 ECG 分类的影响

STAR 论文（2025）提供了最清晰的消融研究（14 类多标签任务）：

| 增强方法 | Micro AUROC | Macro AUROC | 相对基线提升 |
|---------|------------|------------|------------|
| 无增强（基线） | 0.86 | 0.85 | — |
| 噪声 + 滚动 | 0.90 | 0.88 | +4% / +3% |
| 噪声 + 滚动 + Notch | 0.89 | 0.89 | +3% / +4% |
| Multiply-Triangle | 0.92 | 0.88 | +6% / +3% |
| **STAR（心拍级别）** | **0.95** | **0.91** | **+9% / +6%** |
| 水平/垂直翻转 | 0.86 | 0.86 | 0% / +1% |

### 4.2 互补 vs 冲突的增强组合

**互补组合（正交域）：**

- 高斯噪声 + 时间平移（加性噪声 + 相位变化，操作在独立域）
- 基线漂移 + 工频噪声（不重叠的频段：<0.5 Hz vs 50/60 Hz）
- 缩放 + 时间遮蔽（幅度 vs 时间域）
- STAR + 轻度高斯噪声（形态保持性扭曲 + 传感器真实性）

**冲突组合（避免链接）：**

- 多个激进时间扭曲复合会产生非生理形态
- 强噪声 + 重度遮蔽同时降级信号，组合消除太多信息
- 多个重叠幅度变换（抖动 + 缩放 + 幅度扭曲）收益递减
- 朴素均匀 MixUp 会损害 ECG 分类性能——必须渐进式应用

### 4.3 推荐的 8 操作增强池

基于跨研究验证，最优的 8 操作池平衡了多样性、安全性和实验效果：

| 操作 | 参数范围 | 核心依据 |
|-----|---------|---------|
| 高斯噪声 | SNR 15-40 dB | 所有研究均验证有效 |
| 幅度缩放 | 0.85-1.15× | 大多数架构受益 |
| 基线漂移 | 0.05-0.5 Hz | Hatamian 等报告 F1 显著提升 |
| 工频噪声 | 50/60 Hz 注入 | 真实域变异 |
| 幅度扭曲 | 4结点三次样条 | Iwana & Uchida 验证有效 |
| 时间扭曲 | 保守的 ±5% 最大 | 需要标签排除 |
| 随机遮蔽 | 80-180ms 窗口 | TaskAug, CLOCS, 3KG 验证 |
| 导联丢弃 | p ≤ 0.1 | 提升部分导联鲁棒性 |

> ❌ **排除操作：** 极性翻转、时间反转、段置换、导联洗牌
>
> ✅ **可扩展操作：** 加入 STAR 心拍级重采样和电极位移模拟

---

## 五、完整 PyTorch 实现

### 5.1 单个增强操作

```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import CubicSpline

def gaussian_noise(signal, severity):
    """添加高斯噪声。severity 1→SNR 37dB, 10→SNR 10dB"""
    snr_db = 40 - severity * 3
    power = torch.mean(signal ** 2, dim=-1, keepdim=True)
    noise_std = torch.sqrt(power / (10 ** (snr_db / 10)))
    return signal + torch.randn_like(signal) * noise_std

def amplitude_scale(signal, severity):
    """全局缩放。severity 1→±2%, 10→±25%"""
    s = severity * 0.025
    factor = 1.0 + np.random.uniform(-s, s)
    return signal * factor

def baseline_wander(signal, severity, fs=500):
    """添加低频漂移。severity 1→1%, 10→15%"""
    T = signal.shape[-1]
    t = torch.arange(T, dtype=torch.float32) / fs
    amplitude = severity * 0.015 * signal.std()
    n_components = np.random.randint(1, 4)
    wander = torch.zeros(T)
    for _ in range(n_components):
        freq = np.random.uniform(0.05, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        wander += amplitude * torch.sin(2 * np.pi * freq * t + phase)
    return signal + wander.unsqueeze(0)

def powerline_noise(signal, severity, fs=500):
    """添加 50/60 Hz 干扰"""
    T = signal.shape[-1]
    t = torch.arange(T, dtype=torch.float32) / fs
    amp = severity * 0.003 * signal.std()
    freq = np.random.choice([50, 60])
    phase = np.random.uniform(0, 2 * np.pi)
    noise = amp * torch.sin(2 * np.pi * freq * t + phase)
    return signal + noise.unsqueeze(0)

def time_warp(signal, severity, num_knots=4):
    """平滑时间扭曲。severity 1→±1%, 10→±15%"""
    sigma = severity * 0.015
    T = signal.shape[-1]
    knot_pos = np.linspace(0, T - 1, num_knots + 2)
    knot_vals = np.random.normal(1.0, sigma, num_knots + 2)
    knot_vals[0] = knot_vals[-1] = 1.0
    spline = CubicSpline(knot_pos, knot_vals)
    warp = np.cumsum(spline(np.arange(T)))
    warp = warp / warp[-1] * (T - 1)
    result = torch.zeros_like(signal)
    for lead in range(signal.shape[0]):
        result[lead] = torch.from_numpy(
            np.interp(np.arange(T), warp, signal[lead].numpy()))
    return result

def random_mask(signal, severity, fs=500):
    """置零连续片段。severity 控制宽度和数量"""
    sig = signal.clone()
    T = sig.shape[-1]
    n_masks = max(1, severity // 3)
    width = int((0.02 + severity * 0.018) * fs)
    for _ in range(n_masks):
        start = np.random.randint(0, max(1, T - width))
        sig[:, start:start + width] = 0.0
    return sig

def magnitude_warp(signal, severity, num_knots=4):
    """平滑时变幅度缩放"""
    sigma = severity * 0.03
    T = signal.shape[-1]
    knot_pos = np.linspace(0, T - 1, num_knots + 2)
    knot_vals = np.random.normal(1.0, sigma, num_knots + 2)
    spline = CubicSpline(knot_pos, knot_vals)
    warp_curve = torch.from_numpy(spline(np.arange(T))).float()
    return signal * warp_curve.unsqueeze(0)

def lead_dropout(signal, severity):
    """随机置零导联。severity 控制丢弃概率"""
    sig = signal.clone()
    p = min(0.05 * severity, 0.3)
    mask = torch.rand(sig.shape[0]) > p
    sig[~mask] = 0.0
    return sig
```

### 5.2 ECG-AugMix 核心流程

```python
ECG_OPS = [gaussian_noise, amplitude_scale, baseline_wander,
           powerline_noise, time_warp, random_mask,
           magnitude_warp, lead_dropout]

def ecg_augmix(signal, severity=3, width=3, depth=-1, alpha=1.0):
    """AugMix for 12-lead ECG. signal shape: (12, T)"""
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(2, 2))  # 偏向保留原始信号
    mix = torch.zeros_like(signal)
    for i in range(width):
        aug = signal.clone()
        d = np.random.randint(1, 3) if depth < 0 else depth  # 最大深度 2
        for _ in range(d):
            op = ECG_OPS[np.random.randint(len(ECG_OPS))]
            aug = op(aug, severity)
        mix += ws[i] * aug
    return (1 - m) * signal + m * mix

class ECGAugMixDataset(torch.utils.data.Dataset):
    """包装 ECG 数据集，产生 (原始, aug1, aug2) 三元组"""
    def __init__(self, base_dataset, severity=3, width=3):
        self.base = base_dataset
        self.severity = severity
        self.width = width

    def __getitem__(self, idx):
        signal, label = self.base[idx]
        aug1 = ecg_augmix(signal, self.severity, self.width)
        aug2 = ecg_augmix(signal, self.severity, self.width)
        return signal, aug1, aug2, label

    def __len__(self):
        return len(self.base)
```

### 5.3 多标签 JSD 一致性损失与训练步骤

```python
def jsd_multi_label(logits_clean, logits_aug1, logits_aug2):
    """多标签 sigmoid 输出的 JSD 一致性损失"""
    p0 = torch.sigmoid(logits_clean)
    p1 = torch.sigmoid(logits_aug1)
    p2 = torch.sigmoid(logits_aug2)
    m = torch.clamp((p0 + p1 + p2) / 3.0, 1e-7, 1 - 1e-7)
    jsd = (F.binary_cross_entropy(m, p0, reduction='mean') +
           F.binary_cross_entropy(m, p1, reduction='mean') +
           F.binary_cross_entropy(m, p2, reduction='mean')) / 3.0
    return jsd

def train_step(model, batch, optimizer, jsd_weight=12):
    signal, aug1, aug2, targets = batch
    all_in = torch.cat([signal, aug1, aug2], dim=0).cuda()
    targets = targets.cuda()
    logits = model(all_in)
    lc, la1, la2 = torch.split(logits, signal.size(0))
    loss = (F.binary_cross_entropy_with_logits(lc, targets)
            + jsd_weight * jsd_multi_label(lc, la1, la2))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## 六、代码库与工具资源

| 库名 | 地址 | 核心特性 |
|-----|------|---------|
| torch_ecg | github.com/DeepPSP/torch_ecg | AugmenterManager：BaselineWander, Mixup, RandomMasking, StretchCompress |
| ecg-augmentations | github.com/klean2050/ecg-augmentations | PyTorch nn.Module：RandomCrop, PRMask, QRSMask, Scale, GaussianNoise |
| ecglib | pypi.org/project/ecglib | Albumentations风格：SomeOf/OneOf/Compose；SumAug, RandomConvexAug |
| tsaug | github.com/arundo/tsaug | 15种时间序列增强，算符组合（+, @, *） |
| TaskAug | github.com/aniruddhraghu/ecg_aug | 双层优化学习的增强策略 |
| STAR | github.com/NaderNemati/ecg-multilabel-classifier | 心拍级正弦重采样 + SE-ResNet-18 |

---

## 七、关键论文参考

- **TaskAug:** Raghu 等, CHIL/PMLR 2022 — 学习每个任务、每个类别的增强强度
- **RandECG:** Nonaka & Seita, Springer 2022 — 将 RandAugment 适配到 ECG
- **3KG:** Gopal 等, ML4H/PMLR 2021 — VCG 空间 3D 旋转用于对比学习
- **STAR:** Nemati 等, 2025 — 心拍级正弦重采样，micro-AUROC +9%
- **ECG-Mamba:** 2025 — Non-Uniform-Mix 增强策略
- **Rahman 等**, Sensors 2023 — ECG 增强方法的系统综述
- **Iwana & Uchida**, PLOS ONE 2021 — 128 个数据集上的时间序列增强基准
- **Hong 等**, Frontiers in Physiology 2022 — PhysioNet Challenge 2020 所有 41 个方案的元分析

---

## 八、结论与建议

构建 ECG-AugMix 需要在增强多样性与诊断标签保留之间取得平衡——这是图像领域中不存在的约束。以下是三个与图像 AugMix 直觉不同的关键发现：

**第一，** ECG 增强效果高度依赖任务——SpecAugment 提升 RVH 检测但降低 AFib 检测——因此 AugMix 的混合策略必须配合标签条件操作排除来保证安全性。

**第二，** 心拍感知的增强显著优于全局变换——STAR 的 +9% AUROC 对比噪声/平移组合的 +3-4%——建议操作池中至少包含一个心动周期级别的操作。

**第三，** 朴素混合会损害 ECG 分类性能——必须使用渐进/非均匀应用策略，且跳连接应比图像 AugMix 保留更多原始信号。

最优设计使用 **8 个 ECG 专用操作、严重度 3-5**，组织在 **3 条深度 1-2 的链**中，配合 **Beta(2,2) 跳连接**偏向保留原始形态，以及**每标签 Bernoulli JSD 一致性损失**（λ = 6-12）。