# Dataset Preprocessing Pipeline

本文档定义 PTB-XL、PN2021、MIMIC 在 ECGTwin、EfficientNet1DV2 和中心风格判别器中的标准信号预处理流程。

这三个数据源在原始格式上差异很大：

```text
PTB-XL:  规整 10 秒 12 导联源域数据，官方提供 100 Hz / 500 Hz
MIMIC:   ECGTwin 原生大规模 10 秒 12 导联数据，500 Hz，有报告文本
PN2021:  多中心目标域数据，时长/采样率/标签体系/设备风格差异最大
```

因此本项目不是“读进来直接训练”，而是按模型需求走不同入口：

```text
ECGTwin 入口:
  raw mV -> 1024 points -> ECGTwin/MIMIC lead order -> VAE latent/text condition

EfficientNet1DV2 入口:
  lead reorder -> filter -> 100 Hz -> 1000 points -> global z-score -> crop

中心风格判别器入口:
  旧版复用 EfficientNet1DV2 输入；
  新版建议同时使用 EfficientNet feature、ECGTwin latent、digital feature。
```

## 统一目标格式

EfficientNet1DV2 和旧版中心风格分类器输入前的统一缓存格式：

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

ECGTwin 的目标格式不同：

```text
(N, 1024, 12)
raw mV scale
lead order = ECGTwin/MIMIC order
VAE latent = (N, 4, 128)
```

不要把 z-score 后的分类器信号直接送进 ECGTwin VAE。ECGTwin 的 VAE 学到的是 raw mV
波形分布，分类器的 global z-score 会抹掉绝对幅值和部分采集风格。

## 小白版：这些预处理到底在做什么

### Lead Reorder

12 导联 ECG 的 12 个通道必须按同一个顺序排列。不同数据库可能把 `aVL` 和 `aVF`
放在不同位置，或者 header 里的导联顺序不同。

如果导联顺序错了，模型会把一个导联当成另一个导联看。例如把 `aVL` 当成 `aVF`，
等价于把身体不同方向的电信号混起来，诊断和生成都会出问题。

本项目有两个常用顺序：

```text
PTB-XL / classifier order:
  I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

ECGTwin / MIMIC order:
  I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
```

二者主要差异是 `aVL` 和 `aVF` 对调：

```python
ECGTWIN_TO_PTBXL_INDICES = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
```

### Resample

不同数据集采样率不同，例如 PTB-XL 可用 100/500 Hz，PN2021 里有 257/500/1000 Hz。
采样率就是每秒记录多少个点。

EfficientNet1DV2 要统一成：

```text
100 Hz x 10 seconds = 1000 points
```

ECGTwin VAE 要统一成：

```text
102.4 Hz x 10 seconds = 1024 points
```

重采样的作用是让不同来源的 ECG 变成同样长度，方便同一个模型处理。

### Pad / Truncate

有些 ECG 原始时长不足 10 秒，有些长达几十秒甚至 30 分钟。

```text
pad:      太短就在后面补 0
truncate: 太长就截取前 10 秒
```

这让所有样本都能变成固定长度。分类器使用 `(1000,12)`，ECGTwin 使用 `(1024,12)`。

### Filter

滤波是为了去掉 ECG 里常见的非诊断噪声：

```text
50 Hz notch:     去电源工频干扰
0.67-40 Hz bandpass: 保留主要 ECG 频段，去低频漂移和高频肌电噪声
median baseline removal: 去基线漂移
```

分类器和旧版域判别器会做滤波。ECGTwin VAE cache 不做这套滤波，因为 ECGTwin
原生训练分布是 raw mV 波形，过度滤波会改变生成模型看到的风格。

### Global Z-Score

对每一条 ECG，把 12 导联和时间点全部展平成一组数，计算一个 mean/std：

```text
x_norm = (x - mean) / std
```

好处：

- 降低不同设备增益、患者体型、电极贴附造成的幅值差异；
- 让分类器训练更稳定。

坏处：

- 绝对电压信息被削弱；
- HYP/CD 这类依赖电压或传导细节的医学解释不能只靠 z-score 后分类器来证明；
- 做 center style classifier 时，z-score 可能会抹掉一部分中心风格。

### Crop

EfficientNet1DV2 默认不吃完整 10 秒，而是从 `(1000,12)` 里裁剪 `(250,12)`：

```text
train: random crop
eval:  center crop
```

训练随机 crop 可以增加数据变化；评测 center crop 保持确定性。中心风格分类器如果要更强，
应考虑 full 10s 或 multi-crop，因为中心风格可能体现在全局噪声、基线和频谱上。

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

### PTB-XL 进入 ECGTwin

用途：

```text
PTB-XL source-style prompt token
PTB-XL reference latent cache
ECGTwin synthetic generation / offline PGD AT
```

处理流程：

```text
/root/autodl-tmp/ptbxl/raw100.npy
  -> (N,1000,12), 100 Hz, PTB-XL lead order
  -> resample 1000 -> 1024
  -> PTB-XL lead order -> ECGTwin/MIMIC lead order
  -> VAE encode
  -> latent (N,4,128)
  -> SCP code -> diagnostic text
  -> text embedding (L,768)
  -> save list-of-dicts .pt cache
```

实现：

```text
data/prepare_ptbxl_for_ecgtwin.py
scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py
```

重要区别：

- ECGTwin cache 使用 VAE latent 和文本条件；
- 不使用分类器 crop；
- PTB-XL latent cache 已按 ECGTwin 约定保存，后续不要再次 z-score latent。

### PTB-XL 进入 EfficientNet1DV2

用途：

```text
source-domain super5 training
fold9 validation
fold10 in-domain test
synthetic augmentation 对照
```

处理流程：

```text
raw100.npy
  -> unified_preprocess_to_1000(fs=100, target_fs=100, target_len=1000)
  -> filter + global z-score
  -> cache (N,1000,12)
  -> Dataset random/center crop
  -> model input (B,12,250)
```

实现：

```text
scripts/triple_labels/train_ptbxl.py
```

### PTB-XL 进入中心风格判别器

PTB-XL 不应该进入 PN2021 center-id 分类器的主训练，因为目标是区分 PN2021 目标中心。
它可以作为 source/background 参照：

```text
target-likeness:  看 PTB-XL 是否明显低于 target PN2021 center
C2ST:             对比 target-token synth 是否比 PTB-XL/source-token 更接近 target real
feature distance: 作为 source-domain baseline
```

如果使用 PTB-XL 作为判别器输入，应走和 EfficientNet1DV2 相同的 classifier preprocessing；
如果用 ECGTwin latent probe，则走 ECGTwin 1024/raw-mV/VAE encode 路径。

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

### PN2021 进入 ECGTwin

用途：

```text
target-center K=500 reference cache
center prompt-token 训练
target-center synthetic generation
Latent-Hull / real-anchor TA-OMAT
```

处理流程：

```text
WFDB raw PN2021 record
  -> read rec.p_signal and rec.sig_name
  -> reorder to canonical PTB-XL 12-lead order
  -> resample native fs -> 102.4 Hz equivalent
  -> pad/truncate -> 1024 points
  -> PTB-XL lead order -> ECGTwin/MIMIC lead order
  -> no filter, no global z-score
  -> ECGTwin VAE encode -> latent (4,128)
  -> SNOMED -> super5 multi-hot / primary class
  -> save center_full_latents/<center>.pt
  -> select K=500 refs and save ref_selection JSON
```

实现：

```text
scripts/ecgtwin_gen/build_prompt_token_latent_cache.py
```

注意：

- 这里故意不复用旧 `center_token_ablation` 文件，因为旧文件来自分类器风格 z-score 信号；
- `ref_selection/<center>_k500_seed42.json` 是后续所有 no-leak eval 的硬排除依据；
- 目标中心样本 K=500 只用于 center-token/生成，不进入下游同中心评测。

### PN2021 进入 EfficientNet1DV2

用途：

```text
PTB-XL-trained classifier 的 cross-center clean eval
PN2021-C corruption benchmark
fine-tuned/enhanced model 的 external AUROC/AUPRC
```

处理流程：

```text
WFDB raw
  -> lead reorder by rec.sig_name
  -> filter at native fs
  -> resample to 100 Hz
  -> pad/truncate to 1000
  -> global z-score
  -> save clean PN2021 v3 mmap/npz cache
  -> eval center crop (B,12,250)
```

实现：

```text
scripts/triple_labels/eval_crosscenter.py
```

### PN2021 进入中心风格判别器

PN2021 是中心风格判别器的主数据源。

旧版：

```text
PN2021 raw
  -> unified_preprocess_to_1000(apply_filter=True, apply_zscore=True)
  -> (N,1000,12)
  -> crop (12,250)
  -> EfficientNet1DV2 7-way center classifier
```

新版建议：

```text
real-only target-vs-rest binary probe:
  positive = target center held-out real ECG
  negative = other centers held-out real ECG
  query    = target-center-token synthetic ECG

feature probes:
  EfficientNet penultimate feature
  ECGTwin VAE latent
  digital ECG feature vector
```

关键规则：

- 排除 K=500 `ref_record_ids`；
- same-label 匹配，优先 NORM/MI/STTC；
- 生成样本只做 query，不参与判别器训练或阈值选择。

## PN2021-C 腐蚀副本

PN2021-C 是从 clean PN2021 v3 cache 派生的鲁棒性评测副本，目标类似 ImageNet-C：
在不改变原始 GT 标签的前提下，对信号施加可控 severity 的单一 corruption，然后测量
EfficientNet1DV2 AUROC/AUPRC 的下降。

专项计划：

```text
docs/pipelines/pn2021_c_corruption_benchmark_pipeline.md
```

第一版中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

第一版算子来自：

```text
methods/augmix/ecg_ops.py
methods/augmix/severity.py
```

corruption 列表：

```text
powerline_noise
emg_noise
baseline_wander
baseline_shift
random_leads_masking
```

cache 命名：

```text
/root/autodl-tmp/triple_labels/pn2021_c_cache/super5_<center>_<corruption>_s<severity>_100hz1000_v1.npz
```

PN2021-C 规则：

- 输入必须来自 clean PN2021 v3 cache，不能从 raw `.hea/.mat` 另走一套标签逻辑。
- 每个 corruption cache 保留 clean `labels`、`record_ids`、`center`、`class_names`、
  `pn2021_mapping` 和 `preprocess_config`。
- corruption 后必须检查 shape、NaN/Inf、每导联幅值、全零导联比例和 class label 一致性。
- PN2021-C 是评测 benchmark；不要把它混进 clean PN2021 平均指标。

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

### MIMIC 进入 ECGTwin

用途：

```text
ECGTwin 作者原始训练/复现
IBE base-vector reference-target training
DiT text-conditioned generation
optional MIMIC pretraining / ablation
```

处理逻辑：

```text
MIMIC raw diagnostic ECG
  -> 10s, 500 Hz, 12-lead
  -> ECGTwin/MIMIC lead order
  -> resample/pad to 1024 when building ECGTwin cache
  -> raw mV scale
  -> VAE encode -> latent (4,128)
  -> machine/cardiologist report -> text embedding
  -> hr/age/sex -> patient info
```

主要缓存：

```text
/root/autodl-tmp/ECGTwin_Data/Mimic_vae.pt
/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt
```

注意：

- MIMIC 是 ECGTwin 原生风格来源；
- MIMIC 报告标签来自文本抽取，适合生成/预训练，但作为 super5 监督时比 PTB-XL noisy；
- MIMIC 不作为当前论文主线必需数据源。

### MIMIC 进入 EfficientNet1DV2

用途：

```text
zero-shot / external eval
optional MIMIC pretraining ablation
```

处理流程：

```text
record_list.csv + machine_measurements.csv
  -> join report text
  -> report regex -> labels
  -> WFDB raw record
  -> unified_preprocess_to_1000
  -> save mimic_preprocessed_f16.npy (N,1000,12)
  -> patient-level split 8:1:1 by subject_id
  -> test split center crop -> model input (B,12,250)
```

实现：

```text
scripts/crosscenter_tierM/build_mimic_cache.py
scripts/triple_labels/eval_crosscenter.py
```

### MIMIC 进入中心风格判别器

MIMIC 不进入 PN2021 center classifier 的主训练。它可作为：

```text
source-domain/background reference
ECGTwin native-style reference
real-vs-synth C2ST 的外部对照
```

推荐使用方式：

- raw waveform probe：走 EfficientNet classifier preprocessing；
- ECGTwin latent probe：走 ECGTwin raw-mV/VAE encode 路径；
- feature distance：作为 “ECGTwin native source style” 与 PN2021 target style 的距离参照。

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

当前 prompt-token 生成脚本的后处理：

```text
decode latent -> (B,1024,12), ECGTwin order
transpose -> (B,12,1024)
interpolate -> (B,12,1000)
lead reorder -> PTB-XL order
per-sample global z-score
save signals -> .npz
```

实现：

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth.py::postprocess_for_classifier()
```

注意：该 `.npz` 已经是分类器尺度。`SynthNPZDataset` 不会再滤波，只做 shape normalization 和 crop。

## ECGTwin Center Token / Latent-Hull 数据产物

center prompt-token 训练数据应同时保留 ECGTwin 生成链路和分类器链路需要的字段：

```text
latent:            (4, 128), ECGTwin VAE scale
decoded_signal:    optional, (1024, 12), ECGTwin/MIMIC lead order, raw mV
classifier_signal: optional, (1000, 12), PTB-XL lead order, classifier scale
super5_multi_hot:  (5,), CD/HYP/MI/NORM/STTC
primary_class:     one of CD/HYP/MI/NORM/STTC
center:            target center name
record_id:         source record id
text_embed_base:   diagnostic prompt embedding before center token insertion
token_name:        <center_CLASS>
```

Latent-Hull TA-OMAT 的 `z_mix` 只在 same-label pool 内构造。若 mixed latent 解码后的 ECG
未通过数字 ECG gate、teacher consistency gate 或 classifier confidence gate，该样本不能进入
online adversarial training buffer。

当前确认：每个目标中心使用 K=500 ref 样本。包含这些 ref 样本 record ids 的 meta 文件是
后续 fine-tune validation/eval 的硬排除输入，避免把少样本目标中心适配集混入评测集。

## 最低质量检查

每个新 cache 至少检查：

- shape 是否是 `(N, 1000, 12)`。
- 是否含 NaN/Inf。
- 每导联幅值是否明显异常。
- PN2021/MIMIC 的导联是否完整并按标准顺序。
- 训练随机 crop 和评测 center crop 是否一致。

## 三类数据进入三类模型的速查表

| 数据集 | ECGTwin 输入 | EfficientNet1DV2 输入 | 中心风格判别器输入 |
|---|---|---|---|
| PTB-XL | `raw100 -> 1024 -> ECGTwin order -> VAE latent/text embed` | `raw100 -> filter/z-score -> (1000,12) -> crop` | source/background；按 probe 选择 classifier feature 或 ECGTwin latent |
| PN2021 | `raw WFDB -> 1024 -> ECGTwin order -> VAE latent -> K=500 ref` | `raw WFDB -> lead reorder/filter/100Hz/1000/z-score -> eval cache` | 主数据源；target-vs-rest real-only probe，排除 K=500 |
| MIMIC | `raw/report -> ECGTwin native VAE/text/patient cache` | `raw/report -> mimic_preprocessed_f16.npy -> patient split` | background/source-style 参照，不作为 PN2021 center-id 主训练 |
| ECGTwin synthetic | `latent/generated waveform 原生为 ECGTwin order/raw scale` | `decode -> PTB-XL order -> 1000 -> z-score -> .npz` | query-only；不能参与判别器训练 |

## 代码事实来源

| 功能 | 文件 |
|---|---|
| 通用 classifier preprocessing | `scripts/crosscenter_v2/preprocess_utils.py` |
| PTB-XL classifier training | `scripts/triple_labels/train_ptbxl.py` |
| PN2021 clean eval cache | `scripts/triple_labels/eval_crosscenter.py` |
| MIMIC classifier cache | `scripts/crosscenter_tierM/build_mimic_cache.py` |
| PTB-XL -> ECGTwin cache | `data/prepare_ptbxl_for_ecgtwin.py` |
| PN2021 -> ECGTwin center latent cache | `scripts/ecgtwin_gen/build_prompt_token_latent_cache.py` |
| ECGTwin lead conversion | `util/lead_utils.py` |
| prompt-token generation postprocess | `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py` |
