# 毕业论文代码归档整理标准

本文档定义 `ECG_adv` 毕业设计代码仓库在刻录光盘归档前应达到的整理标准。目标不是保留所有历史探索代码，而是交付一份能够支撑 `thesis.md` 中实验、系统演示和结果追溯的干净仓库。

## 1. 总目标

最终归档应满足：

- 论文主线可复现：`thesis.md` 中出现的表格、图和实验结论，都能在代码、文档和 artifacts 中找到对应来源。
- 答辩演示可运行：Streamlit 系统能在目标机器启动，默认样本、默认权重和默认路径可用。
- 历史探索不干扰：论文未采用的旧实验线不混在主线入口中，避免误导后续检查者。
- 大文件不进 git，但随光盘 artifacts 交付，并有清单和校验值。

一句话验收标准：

> 一个没有参与过项目的人，只看 `README.md` 和 `docs/thesis_reproduction.md`，能知道论文每个结果怎么来的；运行 preflight 能知道缺什么；运行指定 stage 能复现对应表图；论文外旧代码不会误导他。

## 2. 主线范围

归档主线只覆盖以下内容：

- PTB-XL super5 五标签任务：`CD`、`HYP`、`MI`、`NORM`、`STTC`。
- EfficientNetV2 super5 异常检测模型。
- ECGTwin 作者式 IBE + DiT 两阶段复现。
- ECGTwin VAE/DiT 生成路径、ECGTwin 到 PTB-XL 导联顺序转换、100 Hz/1000 点输入统一。
- 中心提示向量或 prompt token ECG 生成。
- 合成 ECG 质量门控和可视化。
- 低样本分类实验：真实 2000 baseline、无提示预训练后真实微调、中心提示预训练后真实微调。
- 表 6.8 训练策略和生成条件消融。
- Streamlit demo。
- ONNX/TensorRT 推理导出和 benchmark。

不属于论文主线的内容应迁入 `legacy/` 或从严格归档版移除，例如：

- PN2021-C 鲁棒性实验。
- TA-OMAT、Latent-Hull、real-anchor、synth-anchor 对抗训练。
- AdvDiff、PGD、online/offline adversarial training。
- AugMix。
- Tier-M 或旧 26 类跨中心实验。
- 旧自蒸馏实验线。
- 论文未引用的历史 pipeline 文档和临时脚本。

## 3. 代码目录标准

建议最终主线目录如下：

```text
ECG_adv/
  README.md
  thesis.md
  pyproject.toml
  uv.lock
  docs/
    thesis_reproduction.md
    thesis_repro_manifest.json
    artifact_manifest.json
  apps/streamlit_ecg_demo/
  scripts/final_round/
  scripts/triple_labels/
  scripts/ecgtwin_author_repro/
  scripts/ecgtwin_gen/
  scripts/deploy/
  scripts/streamlit_demo/
  methods/ecgtwin_gen/prompt_token/
  util/
  legacy/                    # 可选，只放论文外历史代码
```

整理要求：

- 主线入口不应 import `legacy/`。
- `scripts/ecgtwin_gen/` 只保留论文采用的 prompt-token 训练、生成、筛选、质量评估脚本。
- `scripts/triple_labels/` 只保留 super5 训练、划分、必要评估和结果整理脚本。
- `scripts/final_round/` 是论文复现总入口，不应默认触发论文外 PN2021、MIMIC、fold10 或旧实验评估。
- `model/ECGTwin` 和 `model/DeepECG` 可以保留为外部模型依赖；其他非主线模型仓库应移除或标注为 optional legacy。
- 如果旧目录仍被主线依赖，应先把必要函数迁移到 `util/` 或主线模块，再移动旧目录。

## 4. 复现入口标准

`scripts/final_round/run_thesis_reproduction.sh` 至少应提供以下 stage：

```text
env              打印路径和 Python 命令
preflight        检查必需数据、权重、合成池、ONNX/TensorRT 文件
split            生成 PTB-XL super5 固定少样本划分
author_repro     ECGTwin IBE + DiT 作者式复现
low_sample       复现表 6.5-6.7
ablation_6_8     逐项复现表 6.8
medical_validity 复现合成 ECG 质量代理统计
figures          导出论文 ECG 可视化样例
feature_dist     真实/合成特征分布分析
export_onnx      导出 EfficientNetV2 ONNX
build_trt        构建 TensorRT FP16 engine
benchmark        运行推理性能 benchmark
evidence         汇总论文证据包
streamlit        启动 Streamlit 演示
```

每个 stage 应满足：

- 默认路径基于 `ECG_ADV_DATA_ROOT`，不能硬编码某台机器的绝对路径。
- 缺文件时明确输出缺失路径和用途。
- 输出目录稳定，文档路径和脚本实际输出一致。
- 表 6.5-6.8 的主指标只来自 PTB-XL 固定 custom split。
- PN2021、MIMIC、fold10 只能作为 optional extra，不得混入论文主表复现入口。

## 5. 论文映射标准

`docs/thesis_reproduction.md` 和 `docs/thesis_repro_manifest.json` 必须覆盖：

- 表 6.2 / 图 6.1：PTB-XL super5 固定划分。
- 表 6.3 / 图 6.2 / 图 6.3：ECGTwin IBE + DiT 复现。
- 表 6.4：合成 ECG 质量代理统计。
- 图 3.3：五类 12 导联合成 ECG 样例。
- 表 6.5：EfficientNetV2 真实 2000 baseline。
- 表 6.6 / 表 6.7 / 图 6.4：无提示/中心提示合成预训练后真实微调。
- 表 6.8：五个消融项的完整命令、输入池、输出目录和训练参数。
- 表 5.2：PyTorch/ONNX/TensorRT 推理性能。
- 图 5.1-5.3：Streamlit 系统截图。

要求：

- 每个论文条目有明确代码入口。
- 每个论文条目有明确输入文件和输出产物。
- 表 6.8 不能只写泛化命令，必须列出每一行实验。
- 结果 JSON、Markdown 汇总和论文数值应能互相追溯。

## 6. 必需 artifacts 标准

必要权重和大文件不进入 git，但必须随光盘打包。

必须打包：

- EfficientNetV2 super5 默认分类器 `best_model.pt`。
- 表 6.5-6.7 使用的无提示预训练初始化权重。
- 表 6.5-6.7 使用的中心提示预训练初始化权重。
- ECGTwin 生成必需权重，包括 VAE、DiT/noise predictor、IBE 相关 checkpoint。
- center token 或 prompt token bank，例如 `prompt_token_bank.pt`。
- 默认 demo ECG 数据和标签文件。
- 默认生成样本池，例如 `gated_samples.npz`。
- ONNX 模型，例如 `efficientnetv2_super5.onnx`。
- 论文结果汇总 JSON、训练日志、重要曲线和截图。

建议打包：

- TensorRT engine，例如 `efficientnetv2_super5_fp16.engine`。需注明只保证当前 CUDA/TensorRT/GPU 环境可用，换机器建议重建。
- `training_log.json`、`train_result.json`、`medical_validity_summary.json`、`inference_benchmark.json`。
- ECGTwin IBE/DiT 训练曲线和配置文件。

不建议打包到最小主线 artifacts；这类条目仍应保留在
`docs/artifact_manifest.json` 里，并用 `package: false` 标注为外部依赖：

- 全量 MIMIC 原始数据。
- 全量 PTB-XL 原始压缩包，除非授权和体积允许。
- PTB-XL 全量 raw/cache 和 MIMIC 派生训练 cache，除非归档介质和授权都允许。
- 论文未采用的历史 checkpoint。
- PN2021-C、TA-OMAT、AdvDiff、AugMix 相关大产物。

## 7. Artifact 清单标准

必须提供 `docs/artifact_manifest.json` 或同等清单。每个条目应包含：

```json
{
  "relative_path": "classifier/super5/best_model.pt",
  "kind": "checkpoint",
  "required": true,
  "package": true,
  "used_by": ["Table 5.2", "Table 6.4", "Streamlit demo"],
  "description": "EfficientNetV2 super5 classifier checkpoint",
  "sha256": "...",
  "size_bytes": 0,
  "can_regenerate": true,
  "regenerate_command": "..."
}
```

同时提供：

- `checksums.sha256`
- artifacts 根目录说明
- `preflight` 检查逻辑
- `missing_artifacts.json` 和 `missing_artifacts.md`，分别给机器和人工检查缺失项
- `package: false` 条目不进入默认光盘 artifact 包，但 `preflight` 仍会检查它们是否满足完整复现环境。

## 8. Streamlit 演示标准

答辩演示需要达到：

- `uv run streamlit run apps/streamlit_ecg_demo/app.py` 能启动。
- 默认打开后加载 demo 样本和默认权重。
- 异常检测、ECG 生成、目标医院训练三个模块均有可展示默认路径。
- 缺少权重或数据时给出清楚 UI 提示，不直接 traceback。
- 生成样本保存路径和右侧预览路径一致。
- TensorRT 不可用时能退回 PyTorch 或 ONNX，并明确提示。
- 长训练任务不应被误触发；演示模式和真实训练模式要能区分。

## 9. 环境标准

最低要求：

- `uv sync --all-groups` 可完成基础环境安装。
- Python 版本在 `pyproject.toml` 和文档中一致。
- CUDA、ONNX Runtime、TensorRT 依赖说明清楚。
- TensorRT 最好拆成 optional deploy 依赖；如保留在主依赖中，文档必须说明 CUDA/TensorRT wheel 约束。
- 所有缓存、checkpoint、生成样本和日志默认写入 `ECG_ADV_DATA_ROOT`，不写入 repo。

## 10. 验收命令

整理完成前至少运行：

```bash
git status --short
uv lock --check
bash -n scripts/final_round/*.sh

uv run python -m compileall -q \
  apps \
  scripts/final_round \
  scripts/triple_labels \
  scripts/ecgtwin_author_repro \
  scripts/ecgtwin_gen \
  scripts/deploy \
  scripts/streamlit_demo \
  methods/ecgtwin_gen \
  util

uv run pytest util/tests -q
uv run ruff check --select F apps scripts methods util
uv run python -m json.tool docs/thesis_repro_manifest.json
uv run python -m json.tool docs/artifact_manifest.json
```

Streamlit 还应至少完成一次：

- AppTest 无异常启动。
- 实际浏览器或截图验证三个模块能打开。
- 默认 demo 样本能展示。
- 默认分类器能完成一次推理。

## 11. 最终完成判定

只有同时满足以下条件，才认为归档整理完成：

- `README.md` 能说明如何安装、如何放置 artifacts、如何启动 demo、如何复现实验。
- `docs/thesis_reproduction.md` 覆盖论文全部实验表图。
- `docs/thesis_repro_manifest.json` 是机器可读的复现索引。
- `docs/artifact_manifest.json` 覆盖所有必需大文件。
- `run_thesis_reproduction.sh preflight` 能明确检查必需文件。
- 表 6.5-6.8 没有混入 PN2021、MIMIC 或 fold10 结果。
- 论文外实验代码已经删除、迁入 `legacy/`，或在文档中明确标注为非主线。
- git 跟踪文件中没有大型权重、数据集、ONNX、TensorRT engine 或本机缓存。
- 工作区干净，host-specific symlink diff 不出现在最终提交中。
- 在目标答辩机器上，Streamlit demo 能启动并完成默认推理/预览流程。
