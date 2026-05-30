# Cross-Center v2 Pipeline

PTBXL → PhysioNet 2021 → MIMIC-IV ECG 跨中心泛化性评测管线。
26 类 SNOMED 标签空间、统一预处理、EfficientNet1DV2 + MaskedFocalLoss。

## 目录

```
crosscenter_v2/
├── preprocess_utils.py       # 统一预处理：bandpass → resample 100Hz → per-sample z-score → crop
├── label_alignment_v2.py     # 26 类标签对齐（PTBXL SCP ↔ PN2021 SNOMED ↔ MIMIC 关键词）
├── train_ptbxl_v2.py         # PTBXL 训练入口（MaskedFocalLoss + CosineLR）
├── eval_crosscenter_v2.py    # PhysioNet 2021 零样本评测（5 + 2 中心）
├── eval_mimic_zeroshot.py    # MIMIC-IV ECG 800k 流式评测
└── build_gap_report.py       # 汇总生成 gap_report.md

# 训练/评测技术文档见 ../../docs/training/:
#   - training_technical_details.md
#   - efficientnet_training.md
#   - gap_report.md
```

## 快速复现

```bash
cd /root/ECG_adv_Gen
PY=/root/miniforge3/envs/ECGTwin/bin/python

# 1. 训练
$PY scripts/crosscenter_v2/train_ptbxl_v2.py \
    --epochs 50 --batch_size 64 --lr 1e-3

# 2. PhysioNet 2021 跨中心评测
$PY scripts/crosscenter_v2/eval_crosscenter_v2.py \
    --ckpt outputs/crosscenter_v2/best.pt --n_bootstrap 1000

# 3. MIMIC-IV 800k 零样本评测
PYTHONUNBUFFERED=1 $PY scripts/crosscenter_v2/eval_mimic_zeroshot.py \
    --ckpt outputs/crosscenter_v2/best.pt --n_bootstrap 100

# 4. 汇总报告
$PY scripts/crosscenter_v2/build_gap_report.py \
    --output outputs/gap_report_v2.md
```

## 关键设计

| 项目 | 选择 | 为什么 |
|---|---|---|
| 预处理 | per-sample z-score | 避免 fit-on-train 把源域偏差注入 OOD |
| 标签阈值 | `confidence≥0`（presence-based） | PTBXL 机器标签 conf=0 也是有效阳性；原 50 阈值丢掉 97% AFIB |
| 损失 | MaskedFocalLoss (α=0.25, γ=2.0, ignore=-1) | 未覆盖类置 -1 不污染训练 |
| 调度 | CosineAnnealingLR(T_max=15, η_min=1e-4) | 实测 ep15 达最优 |
| 早停 | 监控 val Tier-1 AUROC | 不监控 loss（与指标脱钩） |

详见 `../../docs/training/training_technical_details.md`。

## 核心结论（见 ../../docs/training/gap_report.md）

- PTBXL Tier-1 AUROC **0.9685**
- PN2021 5 中心平均 **0.9283**（gap **+4.02 pp**）
- PN2021 AUPRC gap **+13.72 pp**（稀有类差距更大）
- MIMIC 800k Tier-1 AUROC **0.8976**（gap **+7.09 pp**）

## 与 legacy 的关系

- 15 类 legacy 代码仍在 `scripts/crosscenter/`（`train_ptbxl.py` / `label_alignment.py` / `eval_crosscenter*.py`），与本目录完全解耦，不再维护。
- AugMix / combined 实验线也在 legacy 目录，后续若要复用会再拆一层。
