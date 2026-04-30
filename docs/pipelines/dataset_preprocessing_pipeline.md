# Dataset Preprocessing Pipeline

本文档定义 PTB-XL、PN2021、MIMIC 在分类器训练和跨中心评测中的标准信号预处理流程。

## 统一目标格式

分类器输入前的统一缓存格式：

```text
(N, 1000, 12)
float32 or float16 cache
100 Hz
10 seconds
lead order = I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

模型实际输入：

```text
(N, 12, 250)
100 Hz
2.5 seconds crop
```

训练随机裁剪，验证/测试中心裁剪。

## 通用预处理函数

源码：

```text
scripts/crosscenter_v2/preprocess_utils.py
```

核心函数：

```text
unified_preprocess_to_1000()
crop_signal_tc()
```

通用步骤：

1. 输入转为 `(time, channels)` float32。
2. NaN 替换为 0。
3. 如果提供 source lead names，重排到标准 12 导联顺序。
4. 在原始采样率下滤波。
5. 重采样到 100 Hz。
6. pad/truncate 到 1000 点。
7. 每条样本 global z-score。
8. 训练/评测时裁剪到 250 点。

## 滤波规则

`filter_bandpass_safe()`：

```text
notch 50 Hz: only if fs > 150
bandpass: 0.67 - 40 Hz Butterworth, zero-phase
baseline removal: median filter, kernel approximately 0.4 s
```

不要在 100 Hz 信号上做 50 Hz notch，因为 Nyquist 处不稳定。

## PTB-XL

原始输入：

```text
/root/autodl-tmp/ptbxl/raw100.npy
/root/autodl-tmp/ptbxl/ptbxl_database.csv
```

已是：

```text
100 Hz
10 seconds
(N, 1000, 12)
```

标准处理：

```text
fs = 100
source_leads = None
target_fs = 100
target_len = 1000
apply_filter = True
apply_zscore = True
```

注意：

- 该缓存是分类器输入缓存，不是 raw mV 信号。
- 当前训练路径会执行 bandpass/baseline removal 和 per-sample global z-score。
- global z-score 有利于跨中心稳定，但会削弱绝对电压信息；HYP/CD 的医学电压标准不能只靠该分类器特征来证明。

推荐缓存：

```text
/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy
```

fold 切分：

```text
train: strat_fold 1-8
val:   strat_fold 9
test:  strat_fold 10
```

## PN2021

原始输入：

```text
/root/autodl-tmp/physionet2021/training/<center>/*.hea
/root/autodl-tmp/physionet2021/training/<center>/*.mat
```

读取：

```text
wfdb.rdrecord(path)
signal = rec.p_signal
fs = rec.fs
source_leads = rec.sig_name
```

标准处理：

```text
reorder leads by rec.sig_name
filter at native fs
resample to 100 Hz
pad/truncate to 1000
global z-score
center crop to 250 for eval
```

评测缓存：

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache/<scheme>_<center>_100hz1000_v3_super5_normsuppress.npz
```

super5 标签版本：

```text
source of truth: scripts/triple_labels/label_schemes.py
SUPER5_PN2021_MAPPING_VERSION = v3_super5_normsuppress_20260501
SUPER5_PN2021_MAPPING_HASH    = 544ed42dee6d
```

语义约束：

- PN2021 官方标签是 SNOMED-CT code list，不存在官方 `PN2021 -> PTB-XL super5` crosswalk。
- 当前 `super5` 是本项目自定义语义投影，只能按该版本号引用。
- v3 只把明确异常 code 映射到 `CD/HYP/MI/STTC`。
- `NORM` 只允许 explicit sinus rhythm 且没有任何直接异常或 suppress-only 异常时为阳性。
- AF/AFL/PAC/PVC/LAD/RAD/low voltage/Q wave abnormal/early repolarization 等 code 会压制 `NORM`，但不直接产生 super5 阳性类。

缓存失效规则：

- 若 `SNOMED_TO_SUPER5_POSITIVE`、`NORM_POSITIVE_SNOMEDS`、`NORM_SUPPRESS_SNOMEDS`、PN2021 header parser、滤波、z-score、lead reorder 或 crop 规则变化，必须 bump cache version 并重建该目录下对应 cache。
- 评测输出应记录实际 evaluated centers；`ptb-xl` / `ptbxl` 目录即使存在也只能被记录为 excluded，不能进入结果平均。
- v3 cache 已保存并校验基础 metadata：`scheme`、`center`、`class_names`、`cache_version`、`preprocess_config`，super5 还保存 `pn2021_mapping.mapping_version` 和 `pn2021_mapping.mapping_hash`。
- 当前机器若只存在 unversioned 或 `v2_normguard` PN2021 super5 cache，应视为历史缓存；第一次按当前脚本评测会自动生成 v3 cache。
- `--exclude_ref_ids` 只在加载/构建完整 center cache 后过滤测试记录，不要把 ref-pool 排除逻辑写进共享预处理 cache。
- 后续建议继续增加 `missing_lead_count`、`lead_order_counter` 和 `unmapped_snomed_counter`。

固定评测中心：

```text
chapman_shaoxing
cpsc_2018
cpsc_2018_extra
georgia
ningbo
ptb
st_petersburg_incart
```

排除 `ptb-xl` / `ptbxl`，避免 PTB-XL 训练数据泄漏。

标签映射改变后的重建范围：

- 必须重建 PN2021 super5 eval cache，并重跑所有仍要引用的 PN2021 super5 指标。
- 必须重建基于 PN2021 label 抽样的 center-token ref pool、TA-OMAT quick-eval cache、synth-anchor/real-anchor latent pool，尤其是包含 `labels5`、`super5_multi_hot` 或 `ref_record_ids` 的文件。
- 不需要因为 PN2021 标签映射变化而重建 PTB-XL signal cache、PTB-XL `ptbxl_labels.C5.all.npy`、MIMIC signal cache 或 PN2021 raw `.hea/.mat`。
- 若只是用已有 PTB-XL baseline 权重重新评测 PN2021，模型权重可复用；改变的是外部评测标签，不是 PTB-XL 训练监督。

## MIMIC-IV ECG

分类器评测使用预处理缓存：

```text
/root/autodl-tmp/mimic_tierM/mimic_preprocessed_f16.npy
/root/autodl-tmp/mimic_tierM/mimic_index.npz
```

元数据：

```text
/root/autodl-tmp/MIMIC/record_list.csv
/root/autodl-tmp/MIMIC/machine_measurements.csv
```

标准 split：

```text
split == 2 and valid_mask == True
```

MIMIC test dataset 从 f16 mmap 读取后转成 float32，再 center crop 到 `(12, 250)`。

注意：

- `mimic_tierM` 目录名来自历史 Tier-M cache。当前 super5 评测只复用其中的 signal cache、patient-level split 和 valid mask。
- `mimic_index.npz` 内的 `labels_6` 不是 super5 标签，不应被用于 super5 指标。
- super5 labels 当前由 `record_list.csv` + `machine_measurements.csv` report regex 即时生成；若要保证长期复现，应另建版本化 `mimic_super5_labels` cache。

如果重新构建 MIMIC cache，应使用与 PN2021 相同的通用预处理：

```text
lead reorder -> filter at native fs -> resample 100 Hz -> 1000 samples -> global z-score
```

## ECGTwin VAE / 生成数据的特殊规则

ECGTwin VAE 原始约定：

```text
input/output = (B, 1024, 12)
latent = (B, 4, 128)
ECGTwin/MIMIC lead order
raw mV scale
```

分类器需要：

```text
(N, 1000, 12)
PTB-XL lead order
100 Hz
10 seconds
```

ECGTwin 解码后进入分类器前必须：

1. 从 ECGTwin/MIMIC lead order 转成 PTB-XL order。
2. 从 1024 点重采样/截断到 1000 点。
3. 进入 `scripts/triple_labels/train_ptbxl.py --synth_npz` 前，必须匹配分类器训练尺度。
4. 保存为 `(N, 1000, 12)`。

`SynthNPZDataset` 只做 shape normalization 和 crop，不会再次执行 `unified_preprocess_to_1000()`。因此 synthetic `.npz` 必须已经是与 PTB-XL classifier cache 可比的尺度；如果保存 raw mV，应明确作为 raw-mV ablation 并单独记录。

导联转换：

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

## 最低质量检查

每个新 cache 至少检查：

- shape 是否是 `(N, 1000, 12)`。
- 是否含 NaN/Inf。
- 每导联幅值是否明显异常。
- PN2021/MIMIC 的导联是否完整并按标准顺序。
- 训练随机 crop 和评测 center crop 是否一致。
