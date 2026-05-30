# 论文实验代码复现与光盘归档说明

本文件是 `thesis.md` 中实验对应的代码复现入口。归档分支只跟踪源码、配置、说明文档和轻量截图；PTB-XL、MIMIC、ECGTwin 权重、训练 checkpoint、ONNX、TensorRT engine、生成样本池等大文件统一放在 `ECG_ADV_DATA_ROOT` 下，随光盘以 `artifacts/` 或 `migrate_files/` 形式单独交付。

## 1. 环境

推荐在仓库根目录执行：

```bash
uv sync --all-groups
```

默认数据根目录是：

```bash
export ECG_ADV_DATA_ROOT="${HOME}/autodl-tmp"
```

如光盘或新机器的数据盘不同，只需要改这个变量。常用派生路径如下：

```bash
export ECG_ADV_PTBXL_ROOT="${ECG_ADV_DATA_ROOT}/ptbxl"
export ECG_ADV_GRAD_ROOT="${ECG_ADV_DATA_ROOT}/graduate_project"
export ECG_ADV_TRIPLE_ROOT="${ECG_ADV_DATA_ROOT}/triple_labels"
export ECG_ADV_APP_DATA_ROOT="${ECG_ADV_DATA_ROOT}/streamlit_ecg_demo"
export ECG_ADV_FINAL_ROUND_ROOT="${ECG_ADV_DATA_ROOT}/final_round_ablation_20260504"
```

外部模型仓库用脚本恢复软链接：

```bash
bash scripts/bootstrap_model_repos.sh
```

## 2. 一键入口

统一入口脚本：

```bash
bash scripts/final_round/run_thesis_reproduction.sh env
```

可用 stage：

```text
split            创建 PTB-XL super5 固定划分
low_sample       复现真实2000、无提示预训练微调、中心提示预训练微调
medical_validity 复现合成 ECG 质量代理统计
figures          导出五类合成 ECG 可视化样例
feature_dist     真实/合成特征分布分析
export_onnx      导出 EfficientNetV2 ONNX
build_trt        构建 TensorRT FP16 engine
benchmark        运行推理性能 benchmark
evidence         汇总论文结果 JSON/Markdown
streamlit        启动 Streamlit 演示系统
all              运行主要离线复现流程
```

示例：

```bash
bash scripts/final_round/run_thesis_reproduction.sh split
bash scripts/final_round/run_thesis_reproduction.sh low_sample
bash scripts/final_round/run_thesis_reproduction.sh benchmark
bash scripts/final_round/run_thesis_reproduction.sh streamlit
```

## 3. 论文实验对应关系

| 论文位置 | 内容 | 主要代码入口 | 主要产物 |
|---|---|---|---|
| 表 6.2 / 图 6.1 | PTB-XL super5 train=2000、val=2000、test=17799 固定划分 | `scripts/triple_labels/create_ptbxl_super5_split.py` | `${ECG_ADV_GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json` |
| 表 6.3 / 图 6.2 / 图 6.3 | ECGTwin IBE + DiT 两阶段复现 | `scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh` | `${ECG_ADV_DATA_ROOT}/ecgtwin_author_repro/<run>/` |
| 表 6.4 | 中心提示向量/无提示向量生成质量代理统计 | `scripts/final_round/run_medical_validity_ablation.py` | `${ECG_ADV_FINAL_ROUND_ROOT}/medical_validity/` |
| 图 3.3 | 五类 12 导联合成 ECG 样例 | `scripts/final_round/curate_thesis_ecg_examples.py` | `${ECG_ADV_FINAL_ROUND_ROOT}/thesis_selected_ecg_examples/` |
| 表 6.5 | EfficientNetV2 真实 2000 baseline | `scripts/triple_labels/train_ptbxl.py` | `${ECG_ADV_GRAD_ROOT}/method_a_real2000_seed42*/train_result.json` |
| 表 6.6 / 表 6.7 / 图 6.4 | 无提示/中心提示合成预训练后真实微调 | `scripts/final_round/run_low_sample_three_methods.sh` | `${ECG_ADV_GRAD_ROOT}/self_distill_v2_*_realfine_*/train_result.json` |
| 表 6.8 | 训练策略与生成条件消融 | `scripts/triple_labels/train_ptbxl.py` 的 `--synth_npz`、`--synthetic_only`、`--init_ckpt` | `${ECG_ADV_GRAD_ROOT}/method_b_*` 与 `ablation_*` run |
| 表 5.2 | PyTorch / TensorRT 推理性能 | `scripts/deploy/export_efficientnetv2_onnx.py`、`build_tensorrt_engine.py`、`benchmark_inference_backends.py` | `${ECG_ADV_APP_DATA_ROOT}/reports/inference_benchmark.json` |
| 图 5.1-5.3 | Streamlit 原型展示 | `apps/streamlit_ecg_demo/app.py` | `docs/figures/streamlit_demo/*.png` |

## 4. 必需数据与权重

最小复现需要以下类别的文件，文件体积大，不进入 git：

```text
${ECG_ADV_PTBXL_ROOT}/raw100.npy
${ECG_ADV_PTBXL_ROOT}/ptbxl_database.csv
${ECG_ADV_PTBXL_ROOT}/scp_statements.csv
${ECG_ADV_GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json
${ECG_ADV_TRIPLE_ROOT}/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
${ECG_ADV_TRIPLE_ROOT}/super5_minresample_full10_perglobal_20260503/best_model.pt
${ECG_ADV_GRAD_ROOT}/self_distill_v2_*/best_model.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/**/prompt_token_bank.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/**/gated_samples.npz
${ECG_ADV_APP_DATA_ROOT}/models/efficientnetv2_super5.onnx
${ECG_ADV_APP_DATA_ROOT}/models/efficientnetv2_super5_fp16.engine
```

ECGTwin 作者复现还需要：

```text
${ECG_ADV_DATA_ROOT}/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt
${ECG_ADV_DATA_ROOT}/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt
model/ECGTwin/
model/DeepECG/
```

## 5. 归档边界

进入 git：

- `apps/streamlit_ecg_demo/`
- `scripts/final_round/`
- `scripts/triple_labels/`
- `scripts/ecgtwin_author_repro/`
- `scripts/ecgtwin_gen/`
- `scripts/deploy/`
- `methods/ecgtwin_gen/prompt_token/`
- `util/`
- `pyproject.toml`、`uv.lock`
- `README.md`、`AGENTS.md`、本文档和 `thesis.md`

不进入 git：

- `.venv/`
- `migrate_files/`
- `__pycache__/`
- `*.pyc`
- 大型数据集、权重、ONNX、TensorRT engine、生成样本池
- 本机绝对路径 symlink diff，例如 `model/DeepECG -> /home/...`

## 6. 快速健康检查

不跑长训练时，可先做这些检查：

```bash
uv run python -m compileall -q \
  apps scripts/final_round scripts/triple_labels scripts/deploy methods/ecgtwin_gen util adversarial

uv run pytest methods/augmix/tests util/tests -q

uv run streamlit run apps/streamlit_ecg_demo/app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

TensorRT 只在有 CUDA、可用 TensorRT wheel/engine 且当前 GPU 架构匹配时保证可运行；如果换机器，优先保留 ONNX 和 PyTorch checkpoint，然后在目标机器上重新执行 `build_trt`。
