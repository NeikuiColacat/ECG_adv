# Graduate Project Pipeline Walkthrough

本文档用于从头到尾说明当前毕业设计主线做了什么、每个模块为什么这样设计、关键技术细节是什么，以及论文中应该如何解释实验结果。

## 1. 一句话概括

本项目围绕“真实 ECG 低样本条件下，ECGTwin 潜在扩散模型生成的合成 ECG 能否提升异常检测模型性能”展开。完整链路是：

```text
PTB-XL 12 导联 ECG
-> super5 多标签映射和统一预处理
-> 抽取 2000 条真实 ECG 构造低样本训练集
-> 训练 EfficientNet1DV2 real2000 baseline
-> 使用 ECGTwin 生成 no-token / center-token synthetic ECG
-> synthetic hard-label pretrain 或 self-distillation
-> 回到真实 2000 条 ECG clean fine-tune
-> 在固定 custom test 上比较 AUROC/AUPRC
-> 用质量门控、特征分布、可视化和推理部署实验支撑论文
```

论文主结论需要保持克制：ECGTwin 合成数据预训练再用真实 2000 条 ECG 微调，能提升低样本 ECG 异常检测性能；center prompt token 在当前 custom split 上取得最佳结果，但 no-token hard-label 对照也很强，因此不能把所有收益都归因于 center token。

## 2. 数据和任务定义

主任务使用 PTB-XL 的 super5 多标签分类协议。五个类别顺序固定为：

```text
CD, HYP, MI, NORM, STTC
```

含义分别是传导异常、心肌肥厚、心肌梗死、正常心电和 ST-T 改变。PTB-XL 在本项目中是多标签任务，不是五类互斥单分类。一条 ECG 可以同时属于 `CD+MI`、`HYP+STTC` 等组合。

低样本主实验使用固定 seed42 custom split：

| split | 数量 | 用途 |
|---|---:|---|
| train | 2000 | 真实低样本训练集，也是 ECGTwin reference selection / token training 的来源边界 |
| val | 2000 | 训练过程选择 checkpoint |
| test | 17799 | 最终 custom test，对比三条路线 |

split 文件：

```text
/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json
```

这个 test 是内部低样本 custom test，不是官方 PTB-XL fold10。论文里需要明确写成“固定随机种子低样本划分的 custom test”，避免和官方 fold10 混淆。

## 3. 统一 ECG 预处理

分类器侧输入统一为 10 秒、100 Hz、12 导联 ECG：

```text
shape      = (N, 1000, 12)
fs         = 100 Hz
duration   = 10 s
lead order = PTB-XL canonical order
norm       = per_sample_global
preprocess = minimal_resample
```

处理流程：

```text
raw PTB-XL ECG
-> NaN/Inf 清理
-> 重采样到 100 Hz
-> pad/truncate 到 1000 点
-> per-sample global z-score
-> EfficientNet1DV2
```

这里刻意不用复杂滤波和 dataset-level 标准化，原因是本实验关注合成数据对低样本训练的增益，需要尽量减少额外预处理变量。HYP/CD 这类和电压、传导时长相关的类别在 per-sample normalization 下要谨慎解释。

## 4. ECGTwin 生成模块

ECGTwin 的作用是提供一个可控 ECG 潜在扩散生成框架。它不是直接在 1000 点分类器输入空间中生成，而是先在 VAE 潜空间中建模，再 decode 回 ECG。

关键张量和转换规则：

| 项目 | 形状/规则 |
|---|---|
| ECGTwin VAE input/output | `(B, 1024, 12)`，raw mV，ECGTwin/MIMIC lead order |
| VAE latent | `(B, 4, 128)` |
| 分类器输入 | `(N, 1000, 12)`，PTB-XL lead order |
| lead reorder | `ECGTWIN_TO_PTBXL_INDICES = [0,1,2,3,5,4,6,7,8,9,10,11]` |

因此 ECGTwin synthetic ECG 进入分类器前必须做：

```text
ECGTwin decode output, 1024 x 12
-> reorder ECGTwin leads to PTB-XL leads
-> resample/crop to 1000 x 12
-> classifier preprocessing
```

ECGTwin 条件路径由三部分组成：

| 条件 | 作用 |
|---|---|
| reference ECG / text / patient info | 通过 IBE/base_vector 保留个体基础形态 |
| diagnosis text embedding | 给出目标疾病语义 |
| center-class soft prompt token | 在 text path 中注入目标中心/类别风格 |

当前主线使用 textual-inversion-style prompt token，而不是旧的 256-d AdaLN hook。推荐 schema 是 direct MV4：

```text
embeddings[center, class, vector, 768]
```

也就是每个 `center-class` 有 4 个 768 维 soft prompt 向量。生成某个类别时，基础诊断文本 embedding 和 `<center_CLASS>` soft token 一起进入 ECGTwin 的 text path；reference ECG/text/patient info 仍通过 IBE/base_vector 提供个体条件。论文中要强调：目标中心或类别风格放在 text token path，不污染 base_vector。

## 5. 三条低样本实验路线

最终低样本主线比较三条路线。

| 方法 | 训练流程 | 目的 |
|---|---|---|
| real2000 baseline | 只用 2000 条真实 PTB-XL ECG 从零训练 EfficientNet1DV2 | 低样本真实数据基线 |
| no-token hard -> real2000 FT | ECGTwin no-token synthetic hard-label pretrain，再用真实 2000 条 clean fine-tune | 验证 ECGTwin 合成数据本身是否有用 |
| center-token hard -> real2000 FT | ECGTwin center-token synthetic hard-label pretrain，再用真实 2000 条 clean fine-tune | 验证 center-token 版本是否进一步提升 |

这里选择“synthetic pretrain -> real fine-tune”，而不是简单把 synthetic 和 real 混在一起训练，是因为合成 ECG 与真实 ECG 之间仍存在分布差异。先让模型从合成样本中学习类别相关特征，再回到真实 2000 条 ECG 做 clean fine-tune，能让最终决策边界重新贴近真实数据。

分类器是 EfficientNet1DV2 一维 CNN，多标签输出 5 个 sigmoid 概率，训练损失使用 BCE 类多标签目标，主指标为 macro AUROC 和 macro AUPRC。AUPRC 对低样本和类别不平衡更敏感，因此论文中应同时报告 AUROC/AUPRC，不能只看 AUROC。

## 6. 主实验结果

当前论文主表建议使用以下 low-sample custom test 结果：

| method | AUROC | AUPRC | vs real2000 AUROC | vs real2000 AUPRC |
|---|---:|---:|---:|---:|
| real2000 baseline | 0.8433 | 0.6234 | 0.0000 | 0.0000 |
| no-token hard pretrain -> real fine-tune | 0.8669 | 0.6828 | +0.0236 | +0.0594 |
| center-token hard pretrain -> real fine-tune | 0.8735 | 0.7013 | +0.0302 | +0.0779 |

对应 run：

```text
/root/autodl-tmp/graduate_project/method_a_real2000_seed42
/root/autodl-tmp/graduate_project/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc
/root/autodl-tmp/graduate_project/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc
```

可以这样解释：

1. 真实 2000 条 ECG baseline 已经具备一定分类能力，但低样本限制明显。
2. no-token synthetic pretrain 后再真实微调，AUROC/AUPRC 均提升，说明 ECGTwin 合成样本对低样本分类有辅助作用。
3. center-token 版本在当前 custom test 上取得最高 AUROC/AUPRC，说明 prompt-token 条件生成路线是有效候选。
4. 由于 no-token hard-label 对照也很强，论文不能写成“提升完全来自 center token”，更稳妥的说法是“ECGTwin 合成预训练与真实数据微调带来主要收益，center-token 在当前对照范围内进一步取得最佳结果”。

## 7. 合成 ECG 质量和语义验证

生成 ECG 不能只凭“分类器概率高”证明医学有效性。本项目用三层证据支撑生成样本的可用性。

第一层是基础信号质量门控：

```text
NaN / Inf
flatline
amplitude p2p
DC offset
Einthoven residual
aVR residual
HR range
```

第二层是语义 proxy：

```text
frozen EfficientNet1DV2 target-class probability
target class top-1 rate
target probability >= 0.5 rate
NORM abnormal suppression
```

第三层是可视化审计：

```text
每个 super5 类别选 1 张 12 导联 ECG 图
保存对应 synthetic waveform vector
用于论文 qualitative visualization
```

当前 medical validity MVP 结果：

| arm | n | quality pass | warning | fail | semantic top1 | target prob >= 0.5 | mean target prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_token_s05 | 64 | 0.0312 | 0.9688 | 0.0000 | 0.9375 | 1.0000 | 0.9071 |
| no_token | 64 | 0.0938 | 0.9062 | 0.0000 | 0.9062 | 0.9375 | 0.8811 |
| v4_balanced_token | 64 | 0.5312 | 0.4688 | 0.0000 | 0.7969 | 0.9062 | 0.8381 |

解释边界：

```text
这些结果支持 basic plausibility + semantic proxy，
不能宣称合成 ECG 已达到临床级医学有效性。
```

warnings 主要来自导联一致性等数字规则，不是 hard fail。论文中可以写“生成 ECG 未出现明显数值崩溃，类别语义 proxy 较强，但部分导联一致性 warning 表明仍需更严格医学验证”。

## 8. 特征分布分析

为了判断合成 ECG 是否已经和真实 ECG 分布一致，本项目使用 EfficientNet1DV2 penultimate feature 做 real-vs-synth 特征分布分析。

使用的 feature extractor：

```text
/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt
```

这是全量 PTB-XL super5 EfficientNet1DV2 模型，用于提取特征，不是低样本三方法中的任一模型。

C2ST balanced accuracy 的解释：

```text
约 0.5: 两组特征难以区分
越高: 两组分布差异越明显
没有跨领域通用的“有效阈值”，必须和 real-vs-real baseline 对比
```

当前结果中，`real_train2000_vs_real_test` 的 C2ST 基本在 0.50 附近，而 `synth_vs_real_test` 约在 0.84 到 0.95。这说明 synthetic ECG 在 feature space 中仍明显可被区分，不能说它已经完全等价于真实 ECG。这个结果并不否定合成数据的下游价值，反而解释了为什么最终需要真实 2000 条 ECG clean fine-tune。

论文建议表述：

```text
合成 ECG 与真实 ECG 仍存在可测的特征分布差异；
但作为预训练/自蒸馏信号，合成样本能够提供有用的类别相关变化；
最终真实数据微调是控制分布偏移的关键步骤。
```

## 9. 系统演示和推理部署

毕设系统不只包含离线实验，还实现了 Streamlit ECG 演示和推理后端。

Streamlit app：

```text
apps/streamlit_ecg_demo/app.py
```

核心功能：

| 页面 | 功能 |
|---|---|
| Detection | 上传 `.npy/.npz` 或选择 demo ECG，显示 12 导联，输出 super5 概率和 quality gate |
| Generation | 载入预生成 ECGTwin synthetic pool，展示生成 ECG，并送入分类器检测 |
| Robustness | 添加 powerline noise、EMG noise、baseline wander、baseline shift、random lead masking，比较 clean/corrupted |
| Benchmark | 展示 PyTorch、ONNXRuntime、TensorRT benchmark |

部署/加速脚本：

```text
scripts/deploy/export_efficientnetv2_onnx.py
scripts/deploy/build_tensorrt_engine.py
scripts/deploy/benchmark_inference_backends.py
```

当前 TensorRT 已跑通，关键结果：

| backend | device | batch | p50 latency | throughput |
|---|---|---:|---:|---:|
| PyTorch | CUDA | 1 | 8.6646 ms | 113.85 ECG/s |
| TensorRT | CUDA | 1 | 1.2511 ms | 647.64 ECG/s |
| PyTorch | CUDA | 32 | 8.8414 ms | 3492.38 ECG/s |
| TensorRT | CUDA | 32 | 2.0610 ms | 12597.97 ECG/s |

ONNXRuntime 当前主要作为 CPU fallback。TensorRT 用于说明模型推理部署优化已经可运行，但论文不要把 TensorRT 写成生成模型加速，它加速的是 EfficientNet1DV2 异常检测推理后端。

## 10. 论文证据包

写论文时优先引用整理后的 evidence pack：

```text
final/artifacts/evidence_pack/
```

主要内容：

| 路径 | 用途 |
|---|---|
| `tables/low_sample_main_results.md` | 低样本三方法主结果 |
| `tables/per_class_main_results.md` | 分类别 AUROC/AUPRC |
| `tables/medical_validity_summary.md` | 合成 ECG 基础质量和语义 proxy |
| `feature_distribution/feature_distribution_report.md` | real-vs-synth 特征分布分析 |
| `tables/inference_benchmark.md` | PyTorch/ONNX/TensorRT 推理速度 |
| `figures/` | 技术路线图、系统架构图、主结果柱状图、五类 ECG 可视化等 |
| `raw/` | JSON/CSV 原始摘要，便于追溯 |

五类生成 ECG 论文图和对应 waveform vector 保存在：

```text
/root/autodl-tmp/final_round_ablation_20260504/thesis_paper_selected_v1/
```

其中 CD、HYP、MI、STTC 的 target digital pass 较好；NORM 图可作为 representative generated NORM-label sample，但不要写成医生确认正常心电。

## 11. 当前最稳妥的论文叙事

推荐论文主线按下面逻辑展开：

1. 医学 ECG 标注成本高，低样本分类容易受数据不足和类别不平衡影响。
2. 先构建 PTB-XL super5 custom low-sample 协议，确保 baseline、no-token、center-token 都在同一数据划分上比较。
3. 使用 ECGTwin 潜在扩散模型生成 synthetic ECG，并通过质量门控和语义 proxy 过滤/分析生成结果。
4. 直接混合 synthetic 和 real 可能受分布差异影响，因此采用 synthetic pretrain -> real2000 clean fine-tune。
5. 实验显示 no-token 和 center-token 路线均明显优于 real2000 baseline，center-token 在当前 custom test 上最高。
6. 特征分布实验显示 synthetic 与 real 仍可区分，说明真实数据微调不可省略，也限制了“生成 ECG 等价真实 ECG”的说法。
7. Streamlit + TensorRT 展示了系统工程闭环：输入 ECG、显示 12 导联、输出异常检测概率、展示生成样本和鲁棒性扰动，并提供加速推理后端。

## 12. 需要避免的过度表述

不要写：

```text
center token 是所有性能提升的唯一原因。
生成 ECG 已经达到临床诊断级别。
synthetic ECG 与真实 ECG 分布完全一致。
TensorRT 加速了 ECGTwin 扩散采样。
custom test 等同于 PTB-XL 官方 fold10。
```

建议写：

```text
ECGTwin 合成数据预训练结合真实低样本微调，在固定 custom test 上提升了 EfficientNet1DV2 的 AUROC/AUPRC。
center-token 方法在当前三方法对照中取得最佳 AUPRC，但 no-token hard-label 对照同样有效，说明主要收益来自合成预训练与真实微调的组合。
生成 ECG 通过了基础数值检查和语义 proxy 验证，但仍存在导联一致性 warning 和 real-vs-synth 特征分布差异。
TensorRT 用于 EfficientNet1DV2 异常检测模型的推理加速，Streamlit 用于系统交互展示。
```

## 13. 最小复现入口

三方法低样本复现实验入口：

```bash
cd /root/autodl-tmp/ECG_adv_Gen_graduate
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
RUN_TAG=20260506_final \
bash scripts/final_round/run_low_sample_three_methods.sh
```

Streamlit 演示入口：

```bash
cd /root/autodl-tmp/ECG_adv_Gen_graduate
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -m streamlit run apps/streamlit_ecg_demo/app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

构建/更新论文证据包入口：

```bash
cd /root/autodl-tmp/ECG_adv_Gen_graduate
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python scripts/final_round/build_thesis_evidence_pack.py
```

## 14. 模块源码索引

| 模块 | 主要路径 |
|---|---|
| super5 标签映射 | `scripts/triple_labels/label_schemes.py` |
| EfficientNet1DV2 训练/评测 | `scripts/triple_labels/` |
| ECGTwin prompt-token 实现 | `methods/ecgtwin_gen/prompt_token/` |
| prompt-token 训练/生成/gate | `scripts/ecgtwin_gen/train_center_prompt_tokens.py`, `scripts/ecgtwin_gen/generate_center_prompt_token_synth.py`, `scripts/ecgtwin_gen/gate_prompt_token_synth.py` |
| 医学质量验证 | `scripts/final_round/run_medical_validity_ablation.py` |
| 特征分布分析 | `scripts/final_round/run_feature_distribution_analysis.py` |
| 五类 ECG 图挑选 | `scripts/final_round/curate_thesis_ecg_examples.py`, `scripts/final_round/export_five_class_ecg_figures.py` |
| 证据包整理 | `scripts/final_round/build_thesis_evidence_pack.py` |
| Streamlit demo | `apps/streamlit_ecg_demo/` |
| TensorRT/ONNX 部署 | `scripts/deploy/` |

