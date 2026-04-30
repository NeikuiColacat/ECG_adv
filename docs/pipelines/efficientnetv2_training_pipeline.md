# EfficientNet1DV2 Training Pipeline

本文档记录 EfficientNet1DV2 分类器的标准训练、增强训练和跨数据集评测流程。

## 目标

训练一个 1D EfficientNetV2 ECG 多标签分类器，用于：

- PTB-XL fold 10 in-domain 测试。
- PN2021 7-center 跨中心外部评测。
- 可选 MIMIC-IV ECG test split 外部评测。
- 作为 ECGTwin synthetic augmentation 的 downstream utility 验证器。

## 代码入口

| 文件 | 作用 |
|---|---|
| `scripts/triple_labels/train_ptbxl.py` | PTB-XL 训练入口，支持 `super5/sub23/pn26` |
| `scripts/triple_labels/eval_crosscenter.py` | PTB-XL fold10、PN2021、MIMIC 评测入口 |
| `scripts/triple_labels/label_schemes.py` | 标签体系和跨数据集映射 |
| `scripts/crosscenter_v2/preprocess_utils.py` | 统一信号预处理 |
| `model/DeepECG/notebooks/EfficientNetv2.py` | EfficientNet1DV2 模型定义 |

## 数据输入

PTB-XL 输入：

```text
/root/autodl-tmp/ptbxl/raw100.npy
/root/autodl-tmp/ptbxl/ptbxl_database.csv
```

默认预处理缓存：

```text
/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy
```

输出目录必须放在大盘：

```text
/root/autodl-tmp/triple_labels/<run_name>/
```

## 数据切分

PTB-XL 使用 `ptbxl_database.csv` 的 `strat_fold`：

| split | folds |
|---|---|
| train | 1-8 |
| val | 9 |
| test | 10 |

不要把 fold 10 用于模型选择。模型选择监控 fold 9 validation AUROC。

## 信号形状

统一信号缓存：

```text
(N, 1000, 12), float32, 100 Hz, 10 s, lead order = I II III aVR aVL aVF V1-V6
```

训练和评测输入模型前裁剪：

```text
(1000, 12) -> crop_len=250 -> transpose -> (12, 250)
```

训练随机裁剪，评测中心裁剪。

## 模型配置

标准模型：

```text
EfficientNet1DV2(
  variant='s_v2',
  input_channels=12,
  num_classes=<scheme classes>,
  activation='leaky_relu',
  stochastic_depth_prob=0.304,
  dropout_rate=0.0,
  use_se=True,
  norm_type='batch',
)
```

标签方案决定输出维度：

| scheme | classes |
|---|---:|
| `super5` | 5 |
| `sub23` | 23 |
| `pn26` | 26 |

## 损失和指标

损失：

```text
masked BCEWithLogitsLoss
```

规则：

- 标签值 `1.0` 是阳性，`0.0` 是阴性，`-1.0` 是 unknown。
- `-1.0` 不参与 loss 和 AUROC/AUPRC。
- `pos_weight = n_neg / n_pos`，按训练集每类统计，默认 clip 到 50。

主指标：

```text
macro AUROC
macro AUPRC
per-class AUROC/AUPRC
per-center AUROC/AUPRC for PN2021
```

## 标准训练命令

super5 baseline：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/super5 \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 50 \
  --lr 0.01 \
  --weight_decay 0.01 \
  --patience 10 \
  --device cuda
```

synthetic augmentation：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/<run_name> \
  --synth_npz /root/autodl-tmp/<synthetic_run>/synth_waveforms.npz \
  --synth_ratio 0.25 \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 50 \
  --device cuda
```

## 标准评测命令

PN2021 7-center，不跑 MIMIC：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/<run_name> \
  --batch_size 256 \
  --num_workers 8 \
  --skip_mimic
```

完整评测：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/<run_name> \
  --batch_size 256 \
  --num_workers 8
```

## 必须保存的产物

每个训练 run 至少保存：

```text
best_model.pt
training_log.json
train_result.json
```

训练后按标准评测命令运行 `scripts/triple_labels/eval_crosscenter.py`，至少保存：

```text
eval_result.json
```

论文汇总时额外导出：

```text
baseline_vs_synth_metrics.csv
pn2021_per_center_delta.csv
per_class_delta.csv
loss_curve.png
val_curve.png
```

如果当前脚本没有直接生成曲线图，需要从 `training_log.json` 生成，不要只保存终点指标。
