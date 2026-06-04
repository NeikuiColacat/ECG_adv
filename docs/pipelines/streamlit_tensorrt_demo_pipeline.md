# Streamlit And TensorRT Demo Pipeline

Date: 2026-05-04.

本文档根据 `docs/final_round/` 中对“物联网可视化管理系统”和“TensorRT 推理加速”的要求，
规划最终答辩演示系统。目标是搭建一个可运行的 Web 原型，把 ECG 生成、异常检测、
结果可视化和推理加速串成闭环。

## Positioning

本系统服务于毕业设计答辩演示，不替代训练 pipeline。

```text
训练与实验:
  docs/pipelines/efficientnetv2_training_pipeline.md
  docs/pipelines/ecgtwin_center_prompt_token_pipeline.md
  docs/pipelines/latent_hull_online_at_pipeline.md

演示与部署:
  Streamlit frontend
  PyTorch / TensorRT inference backend
  ECG waveform visualization
  super5 abnormal detection report
  optional ECGTwin synthetic generation demo
```

优先级：

```text
P0: Streamlit 可视化 + EfficientNet1DV2 检测闭环
P1: EfficientNet1DV2 ONNX/TensorRT FP16 加速
P2: ECGTwin 生成样本展示与参数控制
P3: ECGTwin 生成后自动送入检测器，形成 generate -> detect -> visualize 闭环
P4: TensorRT 加速 ECGTwin 子模块，仅作为可选增强，不作为答辩必要项
```

## Demo Story

答辩时建议讲成三段：

```text
1. 上传或选择一条 12-lead ECG。
2. 系统完成统一预处理，并用 EfficientNet1DV2 输出 CD/HYP/MI/NORM/STTC 概率。
3. 用户选择疾病类别和中心风格，ECGTwin 生成对应 ECG，系统再检测生成样本并展示医学合法性 gate。
```

这能对应 final_round 中的三个功能要求：

```text
数据接入:
  ECG 文件上传 / 示例样本选择

智能诊断:
  EfficientNet1DV2 super5 multi-label classification

可视化展示:
  12-lead waveform, class probability bar chart, generated-vs-real comparison
```

## System Architecture

推荐目录：

```text
apps/streamlit_ecg_demo/
  app.py
  pages/
    1_detection.py
    2_generation.py
    3_robustness.py
    4_benchmark.py
  components/
    ecg_plot.py
    metric_cards.py
    report_panel.py
  services/
    preprocessing.py
    classifier_backend.py
    generator_backend.py
    quality_gate.py
    tensorrt_backend.py
  assets/
    sample_records.json
    demo_config.yaml
```

模型与缓存不放进 repo：

```text
/root/autodl-tmp/streamlit_ecg_demo/
  models/
    efficientnetv2_super5.pt
    efficientnetv2_super5.onnx
    efficientnetv2_super5_fp16.engine
  samples/
    real_examples/
    generated_examples/
  reports/
    inference_benchmark.json
    demo_run_log.jsonl
```

## Data Input

支持三种输入：

```text
1. 内置示例 ECG:
   从 PTB-XL fold10 或 PN2021 demo subset 中预先挑选少量样本。

2. 文件上传:
   .npy / .npz first; WFDB .hea/.dat optional.

3. 生成样本:
   ECGTwin 输出的 `(1024,12)` ECGTwin order raw mV，
   经 lead reorder + resample 后进入分类器。
```

统一内部格式：

```text
raw_display_signal:
  shape = (1000, 12) or (12, 1000)
  sampling_rate = 100Hz
  unit = mV or normalized unit with clear label

classifier_input:
  shape = (1, 12, 1000)
  lead order = PTB-XL canonical order
  norm = same as selected EfficientNet1DV2 training pipeline
```

必须显示预处理说明：

```text
sampling rate
signal length
lead order
normalization mode
model checkpoint
backend = pytorch / tensorrt
```

## Streamlit Pages

### Page 1: ECG Detection

功能：

```text
upload ECG or choose demo sample
show 12-lead waveform
run super5 classifier
show class probabilities for CD/HYP/MI/NORM/STTC
show thresholded labels
show simple interpretation text
save inference report
```

展示重点：

```text
NORM 与 abnormal class 不互相混淆。
MI/STTC/CD/HYP 是多标签概率，不是互斥 softmax。
未知标签和 PN2021 映射差异不在演示界面里过度解释，只在论文中说明。
```

### Page 2: ECGTwin Generation

功能：

```text
select target class: NORM / MI / STTC / HYP / CD
select center style: PTB-XL source / ningbo / chapman_shaoxing / cpsc_2018 / georgia
select prompt token mode: no-token / target-token / wrong-token demo
select reference ECG from safe demo refs
run ECGTwin generation or load precomputed generated sample
show generated 12-lead ECG
run classifier on generated ECG
show medical validity gates
```

推荐实现策略：

```text
默认使用 precomputed generated pool，保证答辩现场稳定。
保留 live generation 按钮，但标记为 slower path。
live generation 默认 25 sampling steps。
```

原因：

```text
ECGTwin 包含 text embedding、IBE/base_vector、DiT sampling、VAE decode。
现场实时生成比分类器推理慢很多，且依赖 GPU 显存状态。
答辩演示应优先稳定，而不是现场跑大规模采样。
```

### Page 3: Robustness Demo

功能：

```text
select corruption: powerline_noise / emg_noise / baseline_wander / baseline_shift / random_leads_masking
select severity 1..5
apply corruption to current ECG
compare clean vs corrupted waveform
compare clean vs corrupted class probabilities
```

对应论文：

```text
PN2021-C corruption benchmark
ImageNet-C style robustness evaluation
```

### Page 4: Inference Benchmark

功能：

```text
compare PyTorch FP32 / PyTorch AMP / ONNX Runtime if available / TensorRT FP16
batch size = 1, 8, 32
measure latency p50/p95
measure throughput samples/s
measure GPU memory
compare probability drift versus PyTorch reference
```

报告：

```text
latency_ms
throughput_ecg_per_second
max_abs_prob_diff
mean_abs_prob_diff
backend_available
engine_build_config
```

## TensorRT Scope

第一阶段只加速 EfficientNet1DV2 分类器。

理由：

```text
EfficientNet1DV2 输入固定 `(B,12,1000)`，最适合 ONNX -> TensorRT。
ECGTwin DiT 生成链路包含动态 text condition、diffusion loop、VAE decode，
整体 TensorRT 化收益和工程风险不成比例。
答辩中 TensorRT 加速分类器已经能覆盖“实时智能诊断”要求。
```

ECGTwin TensorRT 可选拆分：

```text
VAE decoder only:
  latent `(B,4,128)` -> ECG `(B,1024,12)`
  可作为 P4 尝试。

DiT denoiser:
  暂不作为必要项。
  diffusion loop 多步调用、condition 输入复杂，先保留 PyTorch/bf16。
```

## EfficientNet1DV2 Export Plan

输入：

```text
checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

dummy input:
  shape = (1, 12, 1000)
  dtype = float32
```

导出阶段：

```text
1. load PyTorch EfficientNet1DV2
2. wrap sigmoid outside or inside model consistently
3. export ONNX with dynamic batch axis
4. validate ONNX output against PyTorch logits
5. build TensorRT FP16 engine
6. validate TensorRT logits/probabilities against PyTorch
7. save benchmark report
```

建议脚本：

```text
scripts/deploy/export_efficientnetv2_onnx.py
scripts/deploy/build_tensorrt_engine.py
scripts/deploy/benchmark_inference_backends.py
```

输出：

```text
/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5.onnx
/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5_fp16.engine
/root/autodl-tmp/streamlit_ecg_demo/reports/tensorrt_validation.json
/root/autodl-tmp/streamlit_ecg_demo/reports/inference_benchmark.json
```

通过标准：

```text
max_abs_logit_diff <= 1e-3 to 1e-2 for FP16 TensorRT
max_abs_prob_diff <= 1e-3 to 1e-2 for FP16 TensorRT
batch=1 latency lower than PyTorch eager
batch=32 throughput higher than PyTorch eager
```

如果 TensorRT 环境缺失：

```text
保留 ONNX export + PyTorch AMP benchmark。
Streamlit 后端自动降级到 PyTorch。
论文中写“完成 TensorRT 加速接口设计，实际环境不满足时可回退 PyTorch/AMP”。
```

## Backend Abstraction

分类器后端统一接口：

```python
class ClassifierBackend:
    def predict(self, signal_12x1000) -> dict:
        return {
            "logits": ...,
            "probabilities": ...,
            "labels": ["CD", "HYP", "MI", "NORM", "STTC"],
            "backend": "pytorch" | "onnxruntime" | "tensorrt",
            "latency_ms": ...
        }
```

后端选择：

```text
1. TensorRT engine exists and import succeeds -> TensorRT
2. ONNX Runtime exists and onnx exists -> ONNX Runtime
3. otherwise -> PyTorch
```

生成器后端统一接口：

```python
class GeneratorBackend:
    def generate(self, target_class, center, token_mode, ref_id, steps) -> dict:
        return {
            "signal": ...,
            "metadata": ...,
            "quality_gate": ...,
            "latency_ms": ...
        }
```

默认策略：

```text
load_precomputed=True for final defense demo
live_generation=True only when CUDA memory is sufficient
```

## Quality Gate In Demo

Streamlit 中只展示轻量 gate，不做长时间离线统计：

```text
NaN/Inf check
flatline check
amplitude range
Einthoven residual
HR estimate if available
classifier semantic consistency
```

界面表述：

```text
Signal sanity: pass / warning / fail
Lead consistency: pass / warning / fail
Semantic consistency: intended class probability
```

不要写成：

```text
clinically diagnosed as ...
```

建议写成：

```text
model-predicted probability
basic ECG sanity check
semantic consistency proxy
```

## UI Design

页面结构：

```text
left sidebar:
  backend selector
  model checkpoint selector
  sample selector
  generation controls

main area:
  waveform plot
  probability chart
  quality gate panel
  inference metadata
```

必须能在无 GPU 或 TensorRT 不可用时启动：

```text
app starts
classification backend says unavailable and falls back
generation page can load precomputed samples
benchmark page marks TensorRT unavailable
```

## Security And Demo Stability

文件上传限制：

```text
allowed extensions = .npy, .npz, optional .csv
max upload size = small demo limit, e.g. 20MB
do not execute uploaded files
do not allow arbitrary filesystem path input
```

答辩稳定性：

```text
prepare 20-50 demo samples locally
prepare 5-10 generated examples per class for NORM/MI/STTC
HYP/CD show only if gate pass examples exist
cache model in st.cache_resource
cache sample arrays in st.cache_data
```

## Implementation Checklist

Phase 1: Streamlit minimum viable demo

```text
1. create apps/streamlit_ecg_demo/app.py
2. implement sample selection and file upload
3. reuse util/ecg_viz.py or build Streamlit matplotlib plot wrapper
4. load EfficientNet1DV2 PyTorch checkpoint
5. run preprocessing identical to training/eval pipeline
6. display super5 probabilities
```

Phase 2: TensorRT classifier acceleration

```text
7. write ONNX export script
8. write TensorRT engine build script
9. write backend validation script
10. write benchmark page and JSON report
```

Phase 3: ECGTwin generation demo

```text
11. prepare precomputed ECGTwin generated sample pool
12. implement generation page with no-token / target-token / wrong-token toggle
13. run generated sample through classifier and quality gate
14. add real-vs-generated waveform comparison
```

Phase 4: robustness and final packaging

```text
15. implement lightweight corruption controls
16. add clean-vs-corrupted probability comparison
17. generate final demo screenshots for thesis
18. write user manual under docs/final_round or docs/reports/archive/YYYYMMDD
```

## Evaluation Metrics

系统功能指标：

```text
can launch app
can load demo ECG
can upload ECG file
can run classifier
can render 12-lead waveform
can show probabilities
can load generated sample
can run robustness perturbation
```

加速指标：

```text
PyTorch latency p50/p95
TensorRT latency p50/p95
speedup ratio
probability drift
engine build time
GPU memory peak
```

论文建议表格：

```text
Table: PyTorch vs TensorRT inference latency and probability consistency
Table: demo module functional test checklist
Figure: Streamlit ECG detection page
Figure: ECGTwin generation and detection comparison page
Figure: robustness corruption comparison page
```

## Risks And Fallbacks

TensorRT engine build fails:

```text
Use PyTorch AMP + ONNX export as fallback.
Keep TensorRT code path and document environment limitation.
```

ECGTwin live generation too slow:

```text
Use precomputed generated samples in Streamlit.
Keep live generation as optional advanced mode.
```

Generated HYP/CD samples fail medical gate:

```text
Demo NORM/MI/STTC as stable classes.
Report HYP/CD as limitations and keep their probabilities in classifier output.
```

Uploaded ECG shape incompatible:

```text
Reject with clear message.
Do not silently reshape unknown layouts.
```

## Acceptance Criteria

最终答辩最低标准：

```text
1. Streamlit app can run locally on AutoDL or laptop.
2. App can show 12-lead ECG and super5 prediction probabilities.
3. App can compare real ECG and generated ECG.
4. App can show at least one ECGTwin generated sample for NORM/MI/STTC.
5. App records backend latency and model metadata.
6. TensorRT path either works with validation report, or has ONNX/PyTorch fallback with clear explanation.
```

理想标准：

```text
EfficientNet1DV2 TensorRT FP16 backend works.
batch=1 latency is lower than PyTorch eager.
probability drift remains <= 1e-2.
Streamlit includes robustness corruption demo.
final screenshots are ready for thesis and defense slides.
```

## 2026-05-04 MVP Execution Status

已落地的最小演示闭环：

```text
apps/streamlit_ecg_demo/app.py
apps/streamlit_ecg_demo/components/ecg_plot.py
apps/streamlit_ecg_demo/services/preprocessing.py
apps/streamlit_ecg_demo/services/classifier_backend.py
apps/streamlit_ecg_demo/services/quality_gate.py
```

当前 Streamlit MVP 包含：

```text
Detection tab:
  upload .npy/.npz or choose demo generated ECG
  12-lead waveform visualization
  EfficientNet1DV2 super5 probability output
  basic quality gate

Generation tab:
  load precomputed ECGTwin generated samples
  show generator metadata
  run generated ECG through classifier and quality gate

Robustness tab:
  apply lightweight PN2021-C-style corruptions
  compare clean/corrupted waveform and class probabilities

Benchmark tab:
  show deployment commands and read benchmark JSON if present
```

已落地部署脚本：

```text
scripts/deploy/export_efficientnetv2_onnx.py
scripts/deploy/build_tensorrt_engine.py
scripts/deploy/benchmark_inference_backends.py
```

PyTorch benchmark 已完成：

```text
report:
  /root/autodl-tmp/streamlit_ecg_demo/reports/inference_benchmark.json

checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

batch=1:
  latency ~= 8.27 ms/batch
  throughput ~= 120.86 ECG/s

batch=8:
  latency ~= 8.51 ms/batch
  throughput ~= 940.28 ECG/s

batch=32:
  latency ~= 8.61 ms/batch
  throughput ~= 3716.67 ECG/s
```

ONNXRuntime benchmark 已完成：

```text
report:
  /root/autodl-tmp/streamlit_ecg_demo/reports/inference_benchmark.json

PyTorch CUDA:
  batch=1  latency ~= 9.34 ms, throughput ~= 107.05 ECG/s
  batch=8  latency ~= 9.46 ms, throughput ~= 845.26 ECG/s
  batch=32 latency ~= 9.79 ms, throughput ~= 3269.41 ECG/s

ONNXRuntime:
  provider = CPUExecutionProvider
  batch=1  latency ~= 112.12 ms, throughput ~= 8.92 ECG/s
  batch=8  latency ~= 240.07 ms, throughput ~= 33.32 ECG/s
  batch=32 latency ~= 341.99 ms, throughput ~= 93.57 ECG/s
```

当前部署结论：

```text
PyTorch CUDA is the fastest available backend in this AutoDL environment.
ONNXRuntime works but only has CPUExecutionProvider, so it is a functional
fallback rather than an acceleration backend.
```

当前 TensorRT 阻塞：

```text
TensorRT engine build blocked because `trtexec` is not found.
```

因此答辩表述应为：

```text
已完成 Streamlit 演示系统与 PyTorch 推理基线；
已完成 ONNX/TensorRT 导出与构建脚本；
当前 AutoDL 环境缺少 trtexec，ONNXRuntime 只有 CPU provider，
因此系统默认使用 PyTorch CUDA，ONNXRuntime 作为功能 fallback。
```

若继续完善 TensorRT：

```text
1. install onnx / onnxruntime in ECGTwin env;
2. install TensorRT runtime/toolkit with trtexec;
3. rerun export_efficientnetv2_onnx.py;
4. rerun build_tensorrt_engine.py;
5. add TensorRT backend runtime class to Streamlit service.
```
