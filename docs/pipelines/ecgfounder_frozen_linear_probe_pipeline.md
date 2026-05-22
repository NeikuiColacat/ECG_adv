# ECGFounder Frozen Encoder Super5 Linear Probe Pipeline

本文档规划一个更公平、可解释的 ECGFounder 对比基线：

```text
冻结 ECGFounder encoder
-> PTB-XL fold 1-8 训练 5 类 Super5 线性头
-> PTB-XL fold 9 做模型选择、阈值选择和校准
-> PTB-XL fold 10 与 PN2021 外部中心评估
```

这个方案替代“ECGFounder 150-class head 通过 keyword/max pooling 映射到 Super5”的弱基线。旧映射只能作为辅助零样本参考，主对比应优先使用本线性探针方案。

## 目标

1. 使用 ECGFounder 官方 encoder 作为冻结特征提取器，不更新 encoder 权重。
2. 使用 PTB-XL 官方 `diagnostic_class` Super5 标签训练 `Linear(1024, 5)` 多标签分类头。
3. 使用 fold 9 做验证集，只允许在 fold 9 上选择 best epoch、阈值和校准参数。
4. 在 fold 10 报告 PTB-XL 内部分布测试性能。
5. 在 PN2021 外部中心报告跨中心 AUROC/AUPRC，并使用当前 v5 Super5 标签映射。
6. 与 EfficientNet1DV2 baseline、VAE-only 在线对抗训练、ECGFounder zero-shot strict mapping 做表格对比。

## 数据划分

### PTB-XL

| split | 用途 |
|---|---|
| fold 1-8 | 训练线性头 |
| fold 9 | best epoch、per-class threshold、校准参数 |
| fold 10 | 最终 PTB-XL test，只评估不调参 |

标签来源：

```text
scripts/triple_labels/label_schemes.py
CLASS_NAMES_SUPER5 = CD, HYP, MI, NORM, STTC
```

PTB-XL Super5 使用官方 `scp_statements.csv` 里的 `diagnostic_class` 聚合规则。PTB-XL 中没有 diagnostic superclass 的 all-zero 样本默认保留，作为“5 个 diagnostic superclass 均不阳性”的负样本；如果过滤 all-zero，必须作为 sensitivity analysis 单独报告。

### PN2021

使用当前严格标签映射：

```text
SUPER5_PN2021_MAPPING_VERSION = v5_super5_strict_voltage_pacing_suppress_20260522
mapping hash = 1141e0a9f94b
```

主评估中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

完整外部评估仍保留 7-center 输出：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

必须 hard-exclude `ptb-xl/ptbxl` shards，避免 PTB-XL 训练数据泄漏。

为了和 K=500 target-anchor 方法公平对齐，PN2021 target-center view 默认排除对应中心被抽作 K=500 anchor 的 record ids。虽然 ECGFounder 线性探针本身没有用这些目标中心样本训练，但比较表应使用同一个评估 denominator。

## 信号输入

本方案使用 ECGFounder encoder，因此输入预处理应匹配 ECGFounder 官方/复现代码，而不是 EfficientNet1DV2 的 `minimal_resample` 主线。

当前已有脚本入口：

```text
scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py
```

主线 `--preprocess_policy official_ptbxl_eval` 使用：

```text
wfdb 读取原始 12-lead ECG
lead reorder 到 ECGFounder EXPECTED_LEADS
截取前 10 秒
不足 10 秒则 zero padding
resample 到 ECGFounder TARGET_POINTS = 5000
per-sample global z-score
Net1D(return_features=True) 提取 1024-d feature
```

该主线对齐 ECGFounder 官方 `ptbxl_eval.py` 的 PTB-XL evaluation style：使用
PTB-XL `records500/filename_hr`，输入 `12 x 5000`，不额外加
notch/bandpass/median baseline removal。

消融 `--preprocess_policy filtered_dataset` 额外加入 ECGFounder `util.py` 中的
50Hz notch、0.67-40Hz bandpass 和 0.4 秒 median baseline removal，用于回应
README 对 filtering 的要求。

后续报告中应写清楚：这是 ECGFounder baseline 的模型专属输入协议；不要把它和 EfficientNet1DV2 的 `minimal_resample + per_sample_global + 100Hz + 1000 samples` 混成同一个预处理。

## 模型

### 冻结 encoder

```text
checkpoint: /root/autodl-tmp/ecgfounder/checkpoint/12_lead_ECGFounder.pth
encoder: ECGFounder Net1D
feature dim: 1024
requires_grad: False for all encoder parameters
```

需要在输出 metadata 中记录：

```text
checkpoint path
checkpoint sha256 if practical
encoder parameter count
trainable parameter count
feature dimension
preprocess config
label mapping version/hash
```

### 线性头

```text
head = Linear(1024, 5)
loss = masked BCEWithLogitsLoss
pos_weight = n_negative / n_positive, clip max = 50
optimizer = AdamW
model selection = fold 9 macro AUPRC
```

默认先跑 3 个 seed：

```text
seed = 42, 2025, 3407
epochs = 80
lr = 1e-3
weight_decay = 1e-4
head_batch_size = 1024
```

如果 3 个 seed 的 fold 9 AUPRC 方差很小，论文主表可报告 seed 42，附录报告 mean/std；如果方差明显，则主表应报告 mean/std。

## Fold 9 阈值与校准

AUROC/AUPRC 使用 sigmoid 概率连续分数，不依赖阈值。

fold 9 阈值/校准只用于二值化指标和模型可解释性，不能影响 fold 10/PN2021 的 AUROC/AUPRC 计算。

需要补充到脚本的输出：

1. `threshold_0.5`: 固定 0.5 阈值。
2. `threshold_f1_per_class`: 每类在 fold 9 上最大化 F1 的阈值。
3. `threshold_youden_per_class`: 每类在 fold 9 上最大化 `TPR - FPR` 的阈值。
4. `temperature_per_class` 或 `platt_per_class`: 可选校准参数，只在 fold 9 拟合。
5. fold 10 和 PN2021 上同时保存 raw sigmoid scores 与 calibrated scores。

主论文表仍以 raw sigmoid 的 macro AUROC/AUPRC 为主，避免阈值选择影响跨中心结论。

## 评估输出

每个 seed 至少保存：

```text
run_config.json
ptbxl_ecgfounder_features.npz
pn2021_ecgfounder_features.npz
best_head.pt
head_training_log.json
train_result.json
calibration_fold9.json
ptbxl_fold10_metrics.json
pn2021_ref_excluded_metrics.json
summary.csv
```

PTB-XL fold 10 输出：

```text
macro AUROC
macro AUPRC
per-class AUROC/AUPRC
thresholded F1/sensitivity/specificity as secondary metrics
```

PN2021 输出：

```text
per-center macro AUROC/AUPRC
4-center average
7-center average
target-view ref exclusion count
all-zero-kept main metrics
drop-all-zero sensitivity metrics
```

all-zero-kept 是主结果，因为真实外部中心中会出现不属于 Super5 或证据不足的样本；drop-all-zero 只作为标签映射敏感性分析。

## 需要先补的代码

当前脚本已经支持：

```text
冻结 ECGFounder encoder
提取 1024-d feature cache
fold 1-8 训练 linear head
fold 9 best epoch selection
fold 10 test
PN2021 ref-excluded views
```

还需要补：

1. 固定并记录随机种子。
2. 输出 ECGFounder checkpoint hash、标签映射 version/hash。
3. feature cache 文件名纳入 mapping/preprocess version，避免旧标签缓存混用。
4. fold 9 per-class threshold selection。
5. 可选 per-class calibration。
6. PN2021 drop-all-zero sensitivity view。
7. per-class PN2021 metrics csv，方便定位 CD/HYP/MI/NORM/STTC 哪类拉低。
8. 统一生成和 EfficientNet/VAE-only 对比的 markdown/html report。

## 推荐执行命令

Smoke test：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py \
  --out_dir /root/autodl-tmp/paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_smoke \
  --limit_ptbxl 256 \
  --limit_per_center 16 \
  --batch_size 64 \
  --num_workers 2 \
  --epochs 2 \
  --preprocess_policy official_ptbxl_eval
```

Full run：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_ecgfounder_linear_probe_super5_20260517.py \
  --out_dir /root/autodl-tmp/paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_seed42 \
  --batch_size 96 \
  --num_workers 4 \
  --head_batch_size 1024 \
  --epochs 80 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 42 \
  --preprocess_policy official_ptbxl_eval
```

多 seed 建议写 wrapper 顺序跑：

```text
seed42
seed2025
seed3407
```

不要同时开多个 ECGFounder feature extraction 进程读取 PN2021 原始 WFDB，否则 IO 会成为瓶颈。

## 预期耗时

如果 feature cache 不存在：

```text
PTB-XL feature extraction: 约 10-25 min
PN2021 feature extraction: 约 40-120 min，主要受 WFDB IO 影响
linear head training: 通常 < 10 min
report aggregation: < 5 min
```

如果 feature cache 已存在：

```text
单 seed 线性头训练 + 评估通常 < 10 min
3 seed 通常 < 30 min
```

## 论文使用原则

这个方法可以作为 ECGFounder 的强基线：

```text
Frozen ECGFounder encoder + PTB-XL Super5 linear probe
```

它比 150-class zero-shot keyword mapping 更公平，因为分类头直接在 PTB-XL Super5 上训练；也比 full fine-tuning 更保守，因为没有更新 ECGFounder encoder。

最终对比时必须分清：

| 方法 | 是否使用 PTB-XL 监督 | 是否使用目标中心 K 样本 | 是否更新大模型 encoder |
|---|---:|---:|---:|
| ECGFounder zero-shot strict mapping | 否 | 否 | 否 |
| ECGFounder frozen linear probe | 是，fold 1-8 | 否 | 否 |
| EfficientNet1DV2 PTB-XL baseline | 是，fold 1-8 | 否 | 训练 from scratch |
| VAE-only 在线对抗训练 | 是，PTB-XL baseline | 是，K target anchors | 不涉及 ECGFounder |

如果 ECGFounder frozen linear probe 在 PN2021 上超过我们的 VAE-only 方法，应如实报告，并把我们的贡献定位为“小样本目标中心适配方法”；如果它低于或接近我们的 VAE-only 方法，则可说明目标中心 latent-anchor 在线对抗训练在跨中心适配上仍有优势。
