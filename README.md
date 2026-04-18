# ECG_adv_Gen

**目标**：用 AugMix / ECGTwin / AdvDiff 生成对抗样本微调 ECG 模型，提升跨中心泛化。

## 目录导航（3 分钟）

| 目录 | 用途 |
|---|---|
| `scripts/crosscenter_v2/` | **基线训练 + 评测 pipeline**（26 类 SNOMED, EfficientNet1DV2）|
| `adversarial/` | **AdvDiff 对抗生成 + Adapter fine-tune + 评估** |
| `center_token/` | ECGTwin 中心 token embedding 模块 |
| `scripts/` | 入口脚本（`run_*.py`）|
| `util/` | 核心工具（ECGTwin 加载 / 导联对齐 / 可视化） |
| `data/` | PTBXL → ECGTwin 格式预处理脚本 |
| `datasets/` | 原始数据（PTBXL npy + csv） |
| `model/` | 第三方模型代码（DeepECG / ECGTwin / ecgfounder / advdiff / augmix / SA-AET / ptbxl_benchmarking） |
| `methods/` | **未来研究方向入口**（augmix / ecgtwin_gen / advdiff）|
| `docs/` | 所有文档和参考论文 |
| `outputs/` | 当前实验产出（hard_samples 等）|
| `trash/` | 已归档的 legacy 代码和旧产出 |

## 快速复现（v2 基线）

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
cd /root/ECG_adv_Gen

# 训练 PTBXL baseline
$PY scripts/crosscenter_v2/train_ptbxl_v2.py --epochs 50 --batch_size 64 --lr 1e-3

# PhysioNet 2021 跨中心 zero-shot 评测
$PY scripts/crosscenter_v2/eval_crosscenter_v2.py \
    --ckpt /root/autodl-tmp/crosscenter_v2/best_model.pt --n_bootstrap 1000

# MIMIC-IV 800k 零样本评测
PYTHONUNBUFFERED=1 $PY scripts/crosscenter_v2/eval_mimic_zeroshot.py \
    --ckpt /root/autodl-tmp/crosscenter_v2/best_model.pt --n_bootstrap 100
```

## 关键产物位置

| 产物 | 位置 |
|---|---|
| v2 baseline 模型 | `/root/autodl-tmp/crosscenter_v2/best_model.pt` |
| 评测 JSON | `/root/autodl-tmp/crosscenter_v2/eval_*.json` |
| 对抗困难样本 | `outputs/hard_samples/all_hard_samples.pt` |

## 进一步阅读

- 项目结构详解：[`docs/project_structure.md`](docs/project_structure.md)
- 代码阅读路线：[`docs/reading_guide.md`](docs/reading_guide.md)
- 训练技术细节：[`docs/training/training_technical_details.md`](docs/training/training_technical_details.md)
- 跨中心 gap 报告：[`docs/training/gap_report.md`](docs/training/gap_report.md)
- 方法研究入口：[`methods/README.md`](methods/README.md)

## 环境

```
Python: /root/miniforge3/envs/ECGTwin/bin/python
Conda env: ECGTwin
GPU: RTX 4090 24GB
大件产物: /root/autodl-tmp/（根盘只有 4GB 空闲）
```

更多配置见 [`CLAUDE.md`](CLAUDE.md)。
