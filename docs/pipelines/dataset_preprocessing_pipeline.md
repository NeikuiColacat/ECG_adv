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
/root/autodl-tmp/triple_labels/pn2021_eval_cache/<scheme>_<center>_100hz1000_v2_normguard.npz
```

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
3. 保存为 `(N, 1000, 12)`。

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
