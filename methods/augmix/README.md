# ECG AugMix

ECG 版本的 AugMix 数据增强，配合 JSD 一致性损失用于跨中心泛化训练。

**设计准则**：**尽可能复用上游代码**。5 个算子逐字复制自 DeepECG（fairseq-signals），meta 算法逐字复制自 Google Research 官方 AugMix。仅修 2 个上游已有 bug，不加任何未经验证的启发式规则。

---

## 目录

```
methods/augmix/
├── ecg_ops.py       # 5 个算子 class (verbatim + bugfix)
├── severity.py      # severity[1..10] → 参数映射
├── augmix.py        # Dirichlet + Beta meta 算法（torch tensor 版）
├── jsd_loss.py      # 多标签 JSD 一致性损失
├── dataset.py       # AugMixECGDataset 包装器
├── tests/           # 39 个单元测试
└── README.md        # 本文
```

---

## 快速使用

```python
import torch
from methods.augmix.augmix import augmix
from methods.augmix.dataset import AugMixECGDataset
from methods.augmix.jsd_loss import jsd_multilabel

# 1. 单样本增强（severity ∈ [1, 10]，越大扰动越强）
signal = torch.randn(12, 250)           # PTBXL crop
mixed = augmix(signal, severity=3, width=3, depth=-1, alpha=1.0)

# 2. Dataset 包装（用于训练循环）
aug_ds = AugMixECGDataset(
    base_dataset=my_ptbxl_dataset,      # 必须返回 (signal[12,L], label)
    severity=3,
)
for clean, aug1, aug2, label in DataLoader(aug_ds, batch_size=32):
    logits_c = model(clean)
    logits_1 = model(aug1)
    logits_2 = model(aug2)
    ce_loss = focal_loss(logits_c, label)              # 只在 clean 上监督
    jsd = jsd_multilabel(logits_c, logits_1, logits_2) # 三路一致性
    loss = ce_loss + 12.0 * jsd                        # λ 参考 AugMix 论文
    loss.backward()
```

---

## 算子池（5 个）

来源：`model/DeepECG/fairseq-signals/fairseq_signals/data/ecg/augmentations.py`

| 名称 | 作用 | severity=1 | severity=10 |
|---|---|---|---|
| `powerline_noise` | 50/60Hz 工频正弦叠加（随机相位） | max_amp=0.03 | max_amp=0.30 |
| `emg_noise` | 逐导联高斯白噪（肌电） | max_amp=0.02 | max_amp=0.20 |
| `baseline_shift` | 阶梯型 DC 偏移（覆盖信号一段时间） | shift_ratio=0.05, max_amp=0.05 | shift_ratio=0.40, max_amp=0.40 |
| `baseline_wander` | 3 个低频余弦叠加（0.01–0.2Hz，模拟呼吸/体动） | max_amp=0.05 | max_amp=0.40 |
| `random_leads_masking` | 随机导联清零 | mask_prob=0.05 | mask_prob=0.50 |

**校准基准**：PTBXL 100Hz、per-sample z-score 后 std ≈ 1。

**所有算子共享**的固定参数（不随 severity 变化）：`freq=100`、`p=1.0`、`dependency=False`。

---

## Meta 算法

来源：`model/augmix/augment_and_mix.py:augment_and_mix()`（Google Research, 2019）

伪代码：
```
ws ~ Dirichlet([α]*width)       # width 条 chain 的混合权重
m  ~ Beta(α, α)                 # 原信号 vs mix 的比重

mix = 0
for i in range(width):
    signal_aug = signal.clone()
    d = depth if depth > 0 else rand(1, 4)
    for _ in range(d):
        op = random.choice(ops)
        signal_aug = op(signal_aug)
    mix += ws[i] * signal_aug

mixed = (1-m) * signal + m * mix
```

**与上游差异**：输入从 `PIL.Image / np.ndarray` 改为 `torch.Tensor[12, L]`；去掉 `normalize()`（ECG 预处理在 Dataset 里做完）。其余逐字保留。

---

## 与 `trash/scripts_crosscenter_v1/ecg_augmix.py` 的差异

v1 手搓版本已归档。本版**明确剔除**了 v1 中未经上游验证的部分：

| v1 的做法 | 本版的决定 | 理由 |
|---|---|---|
| 自写 `baseline_wander` 单正弦波 | 用 DeepECG 的 k=3 低频余弦 + 导联相关性 | 更接近真实生理 |
| 自写 `powerline_noise` 固定相位 | 用 DeepECG 的随机相位 + 导联相关性 | 更真实 |
| 自加 `time_warp / magnitude_warp / amplitude_scale` | **不纳入** | 未在 DeepECG 上游验证，首版不冒险 |
| `_TIME_WARP_EXCLUDE / _LEAD_DROP_EXCLUDE / ...` label-aware 黑名单 | **不纳入** | 医学合理性未经审稿，属作者个人启发式 |
| `depth ∈ {1, 2}` 硬编码 | 用 AugMix 原论文默认 `depth=-1` → 随机 {1,2,3} | 尊重原算法 |
| `Beta(2, 2)` 混合系数 | 用 AugMix 原论文默认 `α=1.0`（即 Beta(1,1) 均匀） | 尊重原算法 |

如果后续实验发现 `depth=1~2` 或 `α=2.0` 更适合 ECG，可通过 `augmix(... depth=2, alpha=2.0)` 参数传入，不需要改源码。

---

## 修的上游 bug

| 文件:行 | 上游 bug | 修正 |
|---|---|---|
| `ecg_ops.py::PowerlineNoise.__init__` | `self.denpendency = dependency` (typo) | `self.dependency = dependency` |
| `ecg_ops.py::RandomLeadsMask.__call__` conditional 分支 | `(n1, n2) = self.mask_leads_selection`（字符串 "random"/"conditional" 解包会报错） | `(n1, n2) = self.mask_leads_condition`（正确的 tuple 字段） |

---

## 测试

```bash
cd /root/ECG_adv_Gen
/root/miniforge3/envs/ECGTwin/bin/python -m pytest methods/augmix/tests/ -v
```

**39 条断言**覆盖：
- 每个算子：shape/dtype/no-NaN/severity=1 mild/severity monotonic（5×5=25）
- AugMix meta：shape/no-NaN/perturbation range/depth-width 组合/op 子集/未知 op 拒绝/DEFAULT_OPS 完整（7）
- JSD loss：identical→0/不同→正/梯度流/reduction 模式/非负（7）

---

## 后续

**不包含**（留给下一个 task）：
- 接入 `scripts/crosscenter_v2/train_ptbxl_v2.py`：替换 dataloader + 加 JSD 到 total loss
- λ（JSD 权重）消融
- 消融实验：w/o AugMix vs w/ AugMix 的跨中心 gap 对比
