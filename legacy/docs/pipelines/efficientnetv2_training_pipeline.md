# EfficientNet1DV2 Training Pipeline

本文档记录 EfficientNet1DV2 分类器的标准训练、预处理依据、推荐重构方案、增强训练和跨数据集评测流程。

## 目标

训练一个 1D EfficientNetV2 ECG 多标签分类器，用于：

- PTB-XL fold 10 in-domain 测试。
- PN2021 7-center 跨中心外部评测。
- PN2021 v3 super5 重新映射后的 AUROC/AUPRC 复评。
- PN2021-C 中心腐蚀副本上的鲁棒性下降评测。
- 作为 ECGTwin center prompt-token、Latent-Hull TA-OMAT 和 synthetic augmentation 的 downstream utility 验证器。

当前专项文档：

| 任务 | 执行细则 |
|---|---|
| PN2021 v3 clean 复评 | `docs/pipelines/pn2021_v3_reevaluation_pipeline.md` |
| ECGTwin center prompt-token | `docs/pipelines/ecgtwin_center_prompt_token_pipeline.md` |
| Latent-Hull 在线对抗训练 | `docs/pipelines/latent_hull_online_at_pipeline.md` |
| PN2021-C 鲁棒性评测 | `docs/pipelines/pn2021_c_corruption_benchmark_pipeline.md` |

## 2026-05-03 New Baseline Rerun: Minimal Resample + Full 10s

本轮重新训练一个新的 EfficientNet1DV2 super5 主基线。目的不是继续复用旧
`legacy_ecgfounder_filter + crop_len=250` baseline，而是测试更符合当前推荐的
`recommended A` 输入协议：

```text
native ECG
-> NaN/Inf guard
-> lead reorder to PTB-XL canonical order
-> minimal_resample to 100Hz
-> pad/truncate to 1000 samples = 10s
-> per_sample_global z-score
-> EfficientNet1DV2 full-10s input
```

### Experiment Name

```text
super5_minresample_full10_perglobal_20260503
```

输出目录：

```text
/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503
```

PTB-XL cache：

```text
/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
```

### Training Command

当前 `scripts/triple_labels/train_ptbxl.py` 已经支持这组参数：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503 \
  --cache_path /root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --crop_len 1000 \
  --batch_size 48 \
  --epochs 80 \
  --lr 0.003 \
  --weight_decay 0.01 \
  --cosine_tmax 30 \
  --patience 12 \
  --num_workers 6 \
  --checkpoint_metric auprc \
  --device cuda
```

选择 `checkpoint_metric=auprc` 的原因：

```text
super5 是类别不平衡多标签任务；
后续跨中心和增强实验更关心 AUPRC；
脚本仍会同时保存 best_model_auroc.pt 和 best_model_auprc.pt。
```

第一轮以 `best_model.pt` 即 AUPRC-selected checkpoint 为主。如果 fold10 AUROC 明显下降，
再额外评估 `best_model_auroc.pt`，不要在同一轮里改动预处理或 loss。

### PN2021 Clean Evaluation Requirement

当前 `scripts/triple_labels/eval_crosscenter.py` 的 PN2021 cache metadata 仍硬编码：

```text
apply_filter=True
apply_zscore=True
```

因此在评估本模型前必须先补一个小实现改动：

```text
1. 给 eval_crosscenter.py 增加 --preprocess_mode 和 --norm_mode。
2. 写入 PN2021 compressed/mmap cache metadata。
3. 调用 unified_preprocess_to_1000 时传入 preprocess_mode/norm_mode。
4. 为 minimal_resample/per_sample_global 使用独立 PN2021 cache 目录，避免污染旧 baseline cache。
```

目标评测命令：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503 \
  --crop_len 1000 \
  --batch_size 192 \
  --num_workers 6 \
  --skip_mimic \
  --ptbxl_cache /root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy \
  --preprocess_mode minimal_resample \
  --norm_mode per_sample_global \
  --pn2021_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache_minresample_perglobal \
  --pn2021_mmap_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal \
  --output_path /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/eval_result_v3_super5_normsuppress.json
```

### Metrics To Report

必须记录：

```text
PTB-XL fold10 macro AUROC/AUPRC
PTB-XL per-class AUROC/AUPRC: CD, HYP, MI, NORM, STTC
PN2021 7-center macro AUROC/AUPRC
PN2021 per-center AUROC/AUPRC
PN2021 per-class AUROC/AUPRC
delta vs old baseline /root/autodl-tmp/triple_labels/super5
```

旧 baseline 对照：

```text
old PTB-XL fold10 ~= 0.9064 / 0.7754
old PN2021 7-center ~= 0.8344 / 0.5526
```

Acceptance：

```text
1. PTB-XL fold10 AUPRC 不明显低于旧 baseline。
2. PN2021 AUPRC 持平或提升。
3. 如果 AUROC 小幅下降但 AUPRC 提升，优先保留为后续 AT/PN2021-C 主基线。
4. 如果 full10 训练显存不稳，先降 batch_size 到 32，不改变其他超参数。
```

## 代码入口

| 文件 | 作用 |
|---|---|
| `scripts/triple_labels/train_ptbxl.py` | PTB-XL 训练入口，支持 `super5/sub23/pn26` |
| `scripts/triple_labels/eval_crosscenter.py` | PTB-XL fold10、PN2021、MIMIC 评测入口 |
| `scripts/triple_labels/label_schemes.py` | 标签体系和跨数据集映射 |
| `scripts/crosscenter_v2/preprocess_utils.py` | 统一信号预处理 |
| `model/DeepECG/notebooks/EfficientNetv2.py` | EfficientNet1DV2 模型定义 |

## 参考仓库调研结论

### `model/ecg_ptbxl_benchmarking`

这个仓库是 PTB-XL 训练/评测范式最直接的参考。

- 默认 `sampling_frequency=100`，使用 PTB-XL 的 `raw100.npy` 或 `records100`；也支持 `500Hz`。证据：`model/ecg_ptbxl_benchmarking/code/experiments/scp_experiment.py:14`，`model/ecg_ptbxl_benchmarking/code/utils/utils.py:154`。
- 标准 fold 规则是 train `1-8`、val `9`、test `10`。证据：`model/ecg_ptbxl_benchmarking/code/experiments/scp_experiment.py:48`。
- 信号标准化是 dataset-level `StandardScaler`，只在训练集 fit，然后应用到 train/val/test。证据：`model/ecg_ptbxl_benchmarking/code/utils/utils.py:316`。
- fastai 1D 模型默认 `input_size=2.5` 秒，所以在 `100Hz` 下输入长度是 `250`。证据：`model/ecg_ptbxl_benchmarking/code/models/fastai_model.py:160` 和 `model/ecg_ptbxl_benchmarking/code/models/fastai_model.py:168`。
- 它的 Dataset 支持随机裁剪和中心裁剪；默认 valid/test 会切成多个 2.5s chunk，stride 为 1.25s，最后聚合预测。证据：`model/ecg_ptbxl_benchmarking/code/models/timeseries_utils.py:223`，`model/ecg_ptbxl_benchmarking/code/models/fastai_model.py:174`。
- 默认预测聚合是 `np.amax`，只有显式 `aggregate_fn="mean"` 时才用均值。证据：`model/ecg_ptbxl_benchmarking/code/models/fastai_model.py:288`。
- deep 模型训练使用普通 `binary_cross_entropy_with_logits`，没有训练期 `pos_weight` 或 weighted sampler；类别不平衡主要没有专门处理。证据：`model/ecg_ptbxl_benchmarking/code/models/fastai_model.py:313`。
- fine-tuning 示例明确建议自定义数据为 `100Hz`、`[N,L,12]`、mV；PTB-XL 固定 `L=1000`，但模型只要求最短样本长于 `input_size`。证据：`model/ecg_ptbxl_benchmarking/code/Finetuning-Example.ipynb`。

结论：`2.5s crop` 有背书，但完整做法不是“评测只取中心 2.5s”，而是 train 随机 crop、valid/test 多窗口聚合，benchmark 默认偏向 `max` 聚合。当前 pipeline 只继承了随机 crop，缺了多窗口评测。

### `model/ecgfounder`

这个仓库对滤波、10s 输入和严格 preprocessing 约束最有参考价值。

- README 明确要求使用预训练/微调权重时必须严格遵守 `dataset.py` 的 preprocessing，包括 filtering 和 z-score。证据：`model/ecgfounder/README.md:17`。
- ECGFounder 的 12-lead 输入目标是 `500Hz x 10s = 5000` 点。证据：`model/ecgfounder/physionet2021_dataset.py:27`。
- PhysioNet2021 处理流程：读取 mV 物理单位、截断前 10 秒、按 native fs 滤波、重采样到 5000 点、z-score。证据：`model/ecgfounder/physionet2021_dataset.py:4`。
- 滤波为 50Hz notch、0.67-40Hz Butterworth bandpass、0.4s median baseline removal。证据：`model/ecgfounder/util.py:28`。
- z-score 是 per-record 全局均值/方差。证据：`model/ecgfounder/physionet2021_dataset.py:85`。
- 模型 `Net1D` 用全局平均池化，所以结构上可以吃不同长度，但官方验证/预训练约定是 `(B,12,5000)`。证据：`model/ecgfounder/net1d.py:373`。
- 需要注意，仓库里的 `ptbxl_eval.py` 对 PTB-XL 只做 z-score 和 resample，没有显式复用 README 强调的完整 filtering；因此 ECGFounder 的 README/PN2021 dataset 更适合作为预处理背书，`ptbxl_eval.py` 只能作为 eval 脚本参考。

结论：ECGFounder 支持我们保留滤波链、10s 记录和 per-record z-score 作为合理选择。但它的 `500Hz/5000` 是为了 ECGFounder 自己的权重；我们的 EfficientNetV2 不必直接切到 500Hz。

### `model/DeepECG`

这个仓库主要是部署/推理 pipeline，不是 PTB-XL 从零训练配方。

- DeepECG 提供 EfficientNetV2/WCR/BERT 多模型部署能力。证据：`model/DeepECG/README.md:23`。
- NPY/XML processor 的 expected shape 是 `(2500,12)`，短于 2500 会拒绝，长于 2500 用整数步长下采样。证据：`model/DeepECG/utils/files_handler.py:201` 和 `model/DeepECG/utils/files_handler.py:356`。
- `ECGSignalProcessor` 默认 `fs=250`，因此 `(2500,12)` 对应 250Hz 10s。证据：`model/DeepECG/utils/ecg_signal_processor.py:18`。
- 它会把信号频谱功率缩放到 PTB-XL 参考比例，并做 FFT peak cleanup。证据：`model/DeepECG/utils/analysis_pipeline.py:275` 和 `model/DeepECG/utils/ecg_signal_processor.py:46`。
- EfficientNet wrapper 推理前还会乘 `mhi_factor = 1/0.0048`。证据：`model/DeepECG/models/efficientnet_wrapper.py:32`。

结论：DeepECG 支持“10s 输入”这个大方向，但它的 250Hz、频谱缩放和 MHI factor 是面向其部署模型和数据源的兼容处理，不应直接搬进我们的 PTB-XL/PN2021 super5 主线，除非单独做 ablation。

## 推荐主线

推荐把 EfficientNetV2 classifier 重构为：

```text
native ECG
-> lead reorder to PTB-XL order
-> NaN/Inf guard and unit sanity
-> optional native-fs filter branch
-> resample to 100Hz
-> pad/truncate to 10s = 1000 samples
-> classifier scaling
-> EfficientNet1DV2 full-10s input or 2.5s multi-crop input
```

### 采样率推荐

主线推荐继续使用 `100Hz`。

理由：

- PTB-XL benchmark 默认 100Hz，PTB-XL 官方也提供 `records100/raw100`。
- ECGTwin classifier-ready synthetic output已经被转成 `(N,1000,12)`，即 100Hz 10s。
- PN2021 多中心评测统一到 100Hz，可以显著降低 IO 和缓存压力。
- 我们的 super5 任务主要是 `CD/HYP/MI/NORM/STTC`，`0.67-40Hz` 频段在 100Hz Nyquist 50Hz 下仍可保留主要诊断波形信息。

不推荐现在把主线切到 `500Hz`：

- ECGFounder 的 500Hz 是为了它自己的预训练权重，不是 EfficientNetV2 必需条件。
- PN2021 7-center 和 PN2021-C 会大幅增加缓存与 IO。
- ECGTwin 输出天然更接近 `1024 samples / 10s`，转 100Hz 最干净。

保留 `500Hz/10s` 作为未来 ECGFounder-feature 或数字 ECG 精细测量分支，而不是当前 EfficientNetV2 mainline。

### 输入时长推荐

推荐把主线从当前单中心 `2.5s crop` 升级为 `10s-aware`：

| 方案 | 输入 | 训练 | 评测 | 用途 |
|---|---:|---|---|---|
| legacy | 250 点，2.5s | 随机 crop | 中心 crop | 复现当前 baseline |
| recommended A | 1000 点，10s | 全记录 | 全记录 | 推荐主线 |
| recommended B | 250 点，2.5s | 随机 crop | 多 crop 聚合 | PTB-XL benchmark 风格对照 |

最终推荐优先级：

1. `crop_len=1000` 全 10s 训练/评测作为新主线。
2. `crop_len=250 + multi-crop eval` 作为对照，补齐 PTB-XL benchmark 的完整 2.5s 做法；同时比较 `max` 与 `mean` 聚合。
3. 保留当前 `crop_len=250 + center eval` 只作为历史 baseline。

原因：

- 10s 输入和 ECGFounder/DeepECG/ECGTwin 的记录级处理更一致。
- 2.5s 单 crop 可能错过局部 MI/STTC 形态、低频漂移后的稳定形态、罕见异常 beat，以及跨中心噪声特征。
- 4090D 24GB VRAM 足够支撑 EfficientNet1DV2 的 1000 点输入，必要时降低 batch size。

### 标准化推荐

当前实现是 per-sample global z-score。它有 ECGFounder 风格背书，但会抹掉绝对电压尺度，因此对 HYP、电压相关中心风格、synthetic-real amplitude 对齐不友好。

推荐新增两套标准化配置：

| 名称 | 做法 | 用途 |
|---|---|---|
| `per_sample_global` | 每条 ECG 自己算一个全局 mean/std | 兼容当前 baseline；跨设备幅值更稳 |
| `train_standardizer` | 只在 PTB-XL train fold 1-8 fit 一个全局 scaler，再应用到 val/test/PN2021/synth | 推荐主线；保留样本间幅值差异 |

第一轮实验不要删除 legacy。应同时跑：

```text
legacy_250_per_sample_legacyfilter
full10_1000_per_sample_legacyfilter
full10_1000_trainstandardizer_minimal
crop250_multicrop_trainstandardizer_minimal
optional: full10_1000_trainstandardizer_legacyfilter
```

如果 `train_standardizer` 在 PN2021 上因跨设备幅值差异明显掉点，再退回 `per_sample_global` 作为分类器主线；但论文中要明确 HYP/CD 数字 ECG 验证不能只依赖 per-sample z-score 后的分类器。

### 滤波推荐

调研三个参考仓库后，不建议把当前 full filter 当成唯一主线。更稳的修改是把滤波拆成可配置分支：

| 分支 | 默认用途 | 操作 |
|---|---|---|
| `minimal_resample` | 新主线优先实验 | NaN/Inf 清理、lead reorder、native fs -> 100Hz 重采样、pad/truncate、scaling |
| `legacy_ecgfounder_filter` | 当前结果复现和 ablation | 当前 `filter_bandpass_safe`：notch、0.67-40Hz bandpass、0.4s median baseline removal |
| `raw_for_generation_or_digital` | ECGTwin VAE、center style、数字 ECG 验证 | 保留 raw mV 尺度，只做必要的 lead/order/length 对齐；不做分类器 z-score 和 aggressive filter |

推荐从主线默认删除或移出到 ablation 的操作：

- 不把 `0.4s median baseline removal` 作为 EfficientNetV2 默认主线。它有 ECGFounder 背书，但对 MI/STTC 的 ST/T 低频形态、HYP 的电压尺度、center style 的采集差异都可能过强；适合保留为 `legacy_ecgfounder_filter` 对照。
- 不把 `0.67Hz high-pass` 强制用于所有主线实验。它有 ECGFounder 背书，但 PTB-XL benchmark 主线并没有默认滤波；如果用它，应作为 `legacy_ecgfounder_filter` 或 `bandpass_only` 实验。
- 不引入 DeepECG 的频谱功率缩放、FFT peak cleanup、`mhi_factor = 1/0.0048`。这些是 DeepECG 部署模型的兼容处理，不是我们从零训练 PTB-XL super5 EfficientNetV2 的通用预处理。
- 不把 ECGFounder 的 `500Hz/5000` 当成 EfficientNetV2 主线约束。它服务于 ECGFounder 权重；我们的 PTB-XL/PN2021/MIMIC/synthetic 闭环更适合统一到 `100Hz/1000`。

推荐保留的操作：

- 保留 NaN/Inf 清理、12 lead canonical reorder、10s pad/truncate。
- 保留 `100Hz/1000` 统一长度。PTB-XL benchmark 默认 100Hz，ECGTwin classifier-ready 输出也是 100Hz 10s，PN2021/MIMIC 统一后 IO 压力可控。
- 如果需要从高采样率 PN2021/MIMIC 降到 100Hz，保留 native-fs 下的 anti-alias/低通思想；第一版可继续用 `resample`，后续可以单独 ablation `resample_poly`。
- `50Hz/60Hz notch` 只作为可选 native-fs 操作：当 `fs` 明显高于 notch 频率且频谱中确有工频尖峰时启用。不要在 `fs=100` 上做 50Hz notch，因为 50Hz 是 Nyquist；当前代码跳过 `fs <= 150` 是正确的保护。

如果要复现当前 baseline 或做 ECGFounder-style 对照，保留当前滤波链：

```text
native fs 处理 NaN
-> 50Hz notch only if fs > 150
-> 0.67-40Hz Butterworth bandpass, zero-phase
-> 0.4s median baseline removal
-> resample to 100Hz
```

理由：

- ECGFounder 使用同类 notch + 0.67-40Hz + median baseline removal，因此它可以作为有背书的 ablation。
- PTB-XL benchmark 使用 raw100 + train-fitted StandardScaler，而不是默认 full filter，因此新主线应补一个 `minimal_resample + train_standardizer` 版本。
- DeepECG 支持 10s 输入和频谱清理思路，但它的频谱缩放/MHI 缩放不是通用训练预处理，不能直接搬到本项目主线。
- 滤波如果启用，必须在 native fs 上做，再降采样，避免高频干扰 alias 到低频。

注意：

- 对 super5 来说，HYP 依赖电压尺度，MI/STTC 依赖 ST/T 形态，CD 依赖 QRS/传导形态；因此 aggressive filter 和 per-sample z-score 都不能作为医学合法性的唯一证据。
- 训练/评测必须同一预处理分支。PTB-XL 用 `minimal_resample` 训练时，PN2021/MIMIC/synthetic eval 也必须走同一分支；PTB-XL 用 `legacy_ecgfounder_filter` 训练时，外部评测也用同一分支。
- ECGTwin VAE 输入和数字 ECG 验证保留 raw/decoded 分支；不要把分类器 z-score 后的 ECG 喂回 VAE 或用于电压/QRS/ST 的医学测量。

## 当前实现

当前 unified cache 形状：

```text
(N, 1000, 12), float32, 100Hz, 10s
lead order = I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
```

当前模型输入：

```text
(1000, 12)
-> crop_len=250
-> transpose
-> (12, 250)
```

训练随机裁剪，评测中心裁剪。

当前代码位置：

- PTB-XL 预处理：`scripts/triple_labels/train_ptbxl.py`
- PN2021 预处理：`scripts/triple_labels/eval_crosscenter.py`
- 统一预处理：`scripts/crosscenter_v2/preprocess_utils.py`
- 模型定义：`model/DeepECG/notebooks/EfficientNetv2.py`

## 数据切分

PTB-XL 使用 `ptbxl_database.csv` 的 `strat_fold`：

| split | folds |
|---|---|
| train | 1-8 |
| val | 9 |
| test | 10 |

不要把 fold 10 用于模型选择。模型选择优先看 fold 9 macro AUROC，同时记录 macro AUPRC。

## 类别不平衡处理

当前使用：

```text
masked BCEWithLogitsLoss
pos_weight = n_neg / n_pos
clip max = 50
```

规则：

- 标签值 `1.0` 是阳性，`0.0` 是阴性，`-1.0` 是 unknown。
- `-1.0` 不参与 loss 和 AUROC/AUPRC。
- real-only 训练默认不使用 sampler，靠 `pos_weight` 处理类别不平衡。
- synthetic augmentation 使用 `WeightedRandomSampler` 控制 synthetic:real ratio。

推荐后续只在 ablation 中尝试 sampler 或 focal/asymmetric loss，不要和输入时长、标准化同时改变，否则难以解释增益来源。

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

## 指标

主指标：

```text
macro AUROC
macro AUPRC
per-class AUROC/AUPRC
per-center AUROC/AUPRC for PN2021
```

解释限制：

- AUROC/AUPRC 是分类性能，不等价于生成 ECG 的医学合法性。
- HYP/CD 的医学解释必须结合 raw/decoded ECG 的数字心电标准。
- PN2021 的 super5 标签是项目定义映射，不是官方 SNOMED 到 PTB-XL super5 crosswalk。

## 推荐实验矩阵

### A. 立即可跑

当前代码已经支持 `crop_len=1000`，但它仍然使用当前 `legacy_ecgfounder_filter + per_sample_global` 预处理缓存。它适合回答“10s 是否比中心 2.5s 好”，还不能回答“minimal/no-filter 是否更好”：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/super5_full10_per_sample_legacyfilter \
  --crop_len 1000 \
  --batch_size 64 \
  --num_workers 6 \
  --epochs 50 \
  --lr 0.01 \
  --weight_decay 0.01 \
  --patience 10 \
  --device cuda
```

评测：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5_full10_per_sample_legacyfilter \
  --crop_len 1000 \
  --batch_size 128 \
  --num_workers 6 \
  --skip_mimic \
  --output_path /root/autodl-tmp/triple_labels/super5_full10_per_sample_legacyfilter/eval_result_v3_super5_normsuppress.json
```

### B. 需要小改代码

新增 `--preprocess_mode`：

```text
minimal_resample
legacy_ecgfounder_filter
raw_for_generation_or_digital
```

新增 `--norm_mode`：

```text
per_sample_global
train_standardizer
none
```

新增 `--eval_crop_mode`：

```text
center
full
multicrop_mean
multicrop_max
```

新增推荐 run：

```text
/root/autodl-tmp/triple_labels/super5_full10_trainstd_minimal
/root/autodl-tmp/triple_labels/super5_crop250_multicrop_trainstd_minimal
/root/autodl-tmp/triple_labels/super5_full10_trainstd_legacyfilter
```

`train_standardizer` 实现要求：

- scaler 只 fit PTB-XL train folds `1-8`。
- 保存到 run 目录：`standardizer.json` 或 `standardizer.pkl`。
- val/test/PN2021/synth 必须用同一个 scaler。
- scaler 只能使用 real PTB-XL train，不应混入 synthetic 或 PN2021。
- 每个 `preprocess_mode + norm_mode` 必须有独立 cache version，不能复用旧 `/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy`。

`multicrop` 实现要求：

- 对 10s/100Hz 信号取多个 2.5s crop。
- 建议第一版固定 start：

```text
0, 125, 250, 375, 500, 625, 750
```

- 对每个 crop 得到 logits/probs；`max` 是 PTB-XL benchmark 默认聚合，`mean` 是校准更稳的对照。
- 第一轮同时保存两套结果：`multicrop_max` 和 `multicrop_mean`。如果 `max` 明显提升 MI/STTC/CD 但损害 NORM/AUPRC，应优先报告 `mean` 或按 fold9 AUPRC 选聚合策略。
- 训练仍可用随机 crop，评测不要只用中心 crop。

## 标准训练命令

legacy super5 baseline：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/super5 \
  --crop_len 250 \
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
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/triple_labels/<run_name> \
  --synth_npz /root/autodl-tmp/<synthetic_run>/synth_waveforms.npz \
  --synth_ratio 0.25 \
  --crop_len 250 \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 50 \
  --device cuda
```

synthetic `.npz` 注意事项：

- `train_ptbxl.py` 对 synthetic signals 只做 `(N,12,1000)`/`(N,1000,12)` shape normalization 和 crop。
- 它不会对 synthetic signals 重新滤波、z-score 或 lead reorder。
- 因此 `--synth_npz` 必须已经使用 PTB-XL lead order，并与分类器预处理尺度一致。
- 如果后续启用 `train_standardizer`，synthetic 也必须进入同一 scaler，不允许 synthetic 自己 per-sample z-score 后混入 train-standardized real ECG。

## 标准评测命令

PN2021 7-center，不跑 MIMIC：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/<run_name> \
  --batch_size 256 \
  --num_workers 8 \
  --skip_mimic \
  --output_path /root/autodl-tmp/triple_labels/<run_name>/eval_result_v3_super5_normsuppress.json
```

完整评测：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/<run_name> \
  --batch_size 256 \
  --num_workers 8
```

## Online AT 接入原则

在线对抗训练入口应在 `scripts/pgd_cross_center/` 下扩展，不要把 adversarial 逻辑塞进标准 `train_ptbxl.py` baseline 路径。当前主线使用 Latent-Hull TA-OMAT：

```text
ECGTwin center prompt-token 生成目标中心候选样本
-> VAE encoder 得到 same-label latent pool
-> 在线优化 simplex 权重 w，得到 z_mix
-> decoder 生成 ECG
-> 数字 ECG gate + frozen classifier/teacher gate
-> masked BCE outer training
```

center prompt-token 使用每中心 K=500 ref 样本；评测 fine-tune/增强效果时，必须把这 K=500 条记录从目标中心 validation/eval 中排除。

第一版只允许同 super5 标签或同 primary class 的 latent convex combination：

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

在该约束下，GT 标签可以继承原 class。跨类别混合不进入第一版；若后续要做，必须单独设计 union/soft-label 规则和 teacher-consistency loss，不能默认沿用单一 hard label。

## PN2021-C 鲁棒性评测

PN2021-C 是评测 benchmark，不是训练增强。它应从 clean PN2021 v3 cache 派生 corruption 副本，并保持原 labels、record_ids 和 metadata 可追踪。第一版固定中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

第一版 corruption 算子来自 `methods/augmix/ecg_ops.py`：

```text
powerline_noise
emg_noise
baseline_wander
baseline_shift
random_leads_masking
```

公开 severity 使用 1-5；内部 ECG ops severity 推荐映射到 `[2, 4, 6, 8, 10]`。

## 执行顺序

推荐下一步：

1. 跑 `super5_full10_per_sample`，确认 10s 输入是否直接改善 PTB-XL fold10 和 PN2021 clean。
2. 实现 `multicrop_max/multicrop_mean` eval，复核当前 2.5s baseline 是否因为 center crop 单点评测低估。
3. 实现 `train_standardizer`，跑 full10 和 multicrop 两条线。
4. 再把最好的 preprocessing/input-length 作为后续 ECGTwin synthetic、center-token 和 TA-OMAT 的统一 downstream classifier。

不要同时改采样率、标准化、输入时长、loss 和 augmentation。每次只改一个轴，避免结果不可解释。
