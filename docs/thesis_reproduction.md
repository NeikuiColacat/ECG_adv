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

如果当前仓库带有 `migrate_files/ecg_grad_*_artifacts_*.tar.gz`，先用下列命令把
可随光盘交付的运行权重、demo 数据、ONNX/TensorRT 文件和轻量复现实验结果恢复到
`ECG_ADV_DATA_ROOT`：

```bash
bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts
```

恢复脚本会读取 `migrate_files/SHA256SUMS_*.txt` 校验 tar 包，并拒绝带路径穿越的
tar member。可用 `RESTORE_DRY_RUN=1 bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts`
只查看会解压哪些归档。

## 2. 一键入口

统一入口脚本：

```bash
bash scripts/final_round/run_thesis_reproduction.sh env
```

可用 stage：

```text
preflight        检查 docs/artifact_manifest.json 中默认光盘归档必需的数据、权重和结果文件
preflight_full   检查完整重跑所需的全部数据、权重和结果文件
restore_artifacts 从 migrate_files/*.tar.gz 恢复已打包 artifacts 到 ECG_ADV_DATA_ROOT
thesis_assets    检查 thesis.md 引用的本地图片是否都存在
split            创建 PTB-XL super5 固定划分
author_repro     复现 ECGTwin IBE + DiT 两阶段训练
low_sample       汇总表 6.5-6.7 已归档的 custom split 结果
low_sample_rerun 完整重跑表 6.5-6.7 训练，需要额外初始化 checkpoint
ablation_6_8     逐项运行表 6.8 消融命令
medical_validity 复现合成 ECG 质量代理统计
figures          导出五类合成 ECG 可视化样例
feature_dist     可选的真实/合成特征分布分析
export_onnx      导出 EfficientNetV2 ONNX
build_trt        构建 TensorRT FP16 engine
benchmark        运行推理性能 benchmark
evidence         汇总论文结果 JSON/Markdown
package_artifacts 根据 artifact_manifest 打包可随光盘交付的权重、demo 和结果 artifacts
streamlit        启动 Streamlit 演示系统
all              运行主要离线复现流程
```

`feature_dist` 依赖额外的 DeepECG 特征提取代码，当前论文正文没有单独表图引用，
因此保留为可选审计 stage，不作为 `all` 和必需 preflight 的主线门槛。

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
| 表 3.1 | 合成 ECG 质量门控策略 | `scripts/ecgtwin_gen/gate_prompt_token_synth.py`、`scripts/final_round/run_medical_validity_ablation.py`、`util/ecg_digital_features.py` | 表格写入 `thesis.md`；统计结果见 `artifacts/evidence_pack/raw/medical_validity_summary.json` |
| 表 4.1 | PTB-XL 诊断超类五标签任务定义 | `scripts/triple_labels/label_schemes.py`、`${ECG_ADV_PTBXL_ROOT}/scp_statements.csv` | 表格写入 `thesis.md`；类别顺序由代码常量 `CLASS_NAMES_SUPER5` 固定 |
| 表 4.2 | EfficientNetV2 训练实现要点 | `scripts/triple_labels/train_ptbxl.py`、`pyproject.toml`、`uv.lock` | 表格写入 `thesis.md`；训练参数保存在各 `train_result.json` 的 `config` 字段 |
| 表 5.1 | Streamlit 原型功能模块 | `apps/streamlit_ecg_demo/app.py`、`apps/streamlit_ecg_demo/services/` | 表格写入 `thesis.md`；截图见 `artifacts/figures/streamlit_demo/` |
| 表 6.1 | 实验环境配置 | `pyproject.toml`、`uv.lock`、`scripts/final_round/run_thesis_reproduction.sh env` | 表格写入 `thesis.md`；运行 `env` stage 查看当前路径、Python 命令和 PyTorch/CUDA/TensorRT 版本快照 |
| 图 3.1 / 图 3.2 | IBE 与 DiT 结构示意图 | 静态论文图；用 `thesis_assets` 检查断链 | `artifacts/evidence_pack/figures/architecture/*.png` |
| 表 6.2 / 图 6.1 | PTB-XL super5 train=2000、val=2000、test=17799 固定划分 | `scripts/triple_labels/create_ptbxl_super5_split.py` | `${ECG_ADV_GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json` |
| 表 6.3 / 图 6.2 / 图 6.3 | ECGTwin IBE + DiT 两阶段复现 | `scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh` | `${ECG_ADV_DATA_ROOT}/ecgtwin_author_repro/<run>/` |
| 表 6.4 | 中心提示向量/无提示向量生成质量代理统计 | `scripts/final_round/run_medical_validity_ablation.py` | `${ECG_ADV_FINAL_ROUND_ROOT}/medical_validity/` |
| 图 3.3 | 五类 12 导联合成 ECG 样例 | `scripts/final_round/curate_thesis_ecg_examples.py` | 默认读取 `artifacts/samples/generated_ecg_examples/thesis_selected_samples.npz`；输出到 `${ECG_ADV_FINAL_ROUND_ROOT}/thesis_selected_ecg_examples/`，并同步更新论文引用图 `artifacts/figures/generated_ecg_examples/thesis_synthetic_12lead.png` |
| 表 6.5 | EfficientNetV2 真实 2000 baseline | `scripts/final_round/summarize_low_sample_results.py` 或 `scripts/triple_labels/train_ptbxl.py` | `${ECG_ADV_GRAD_ROOT}/method_a_real2000_seed42/train_result.json` |
| 表 6.6 / 表 6.7 / 图 6.4 | 无提示/中心提示合成预训练后真实微调 | `scripts/final_round/summarize_low_sample_results.py`；完整重跑用 `run_low_sample_three_methods.sh` | `${ECG_ADV_GRAD_ROOT}/self_distill_v2_e23.../train_result.json` 与 `${ECG_ADV_GRAD_ROOT}/self_distill_v2_e24.../train_result.json` |
| 表 6.8 | 训练策略与生成条件消融 | `scripts/final_round/run_table_6_8_ablations.sh` | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/` |
| 表 5.2 | PyTorch / TensorRT 推理性能 | `scripts/deploy/export_efficientnetv2_onnx.py`、`build_tensorrt_engine.py`、`benchmark_inference_backends.py` | `${ECG_ADV_APP_DATA_ROOT}/reports/inference_benchmark.json` |
| 图 5.1-5.3 | Streamlit 原型展示 | `apps/streamlit_ecg_demo/app.py` | `artifacts/figures/streamlit_demo/*.png` |

`thesis.md` 直接引用的轻量图表证据已放在 `artifacts/evidence_pack/` 与
`artifacts/figures/` 下。可用下面命令检查论文图片是否全部存在：

```bash
bash scripts/final_round/run_thesis_reproduction.sh thesis_assets
```

`low_sample` 和 `evidence` stage 会优先读取 `${ECG_ADV_GRAD_ROOT}` 下的原始
`train_result.json`，如果新机器没有这些历史路径，则回退到仓库内
`artifacts/evidence_pack/raw/train_results/` 的轻量结果证据。

### 表 6.8 消融行

`ablation_6_8` stage 固定使用 `scripts/final_round/run_table_6_8_ablations.sh`。基础训练参数为
`--scheme super5 --preprocess_mode minimal_resample --norm_mode per_sample_global --crop_len 1000 --checkpoint_metric auroc --lr 0.01 --weight_decay 0.01 --cosine_tmax 15 --patience 10 --epochs ${EPOCHS:-50}`。
五个训练目录都会写出 `train_result.json`、`training_log.json` 和 best checkpoint；全部完成后，
脚本会额外在 `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/` 写出
`table_6_8_summary.csv` 和 `table_6_8_summary.json`。`evidence` stage 默认优先读取最新
`table_6_8_ablation_*/table_6_8_summary.csv`，没有 rerun 结果时再回退到仓库内归档 CSV。

| 论文方法 | 输入 | 初始化 | 输出目录 | 额外训练参数 |
|---|---|---|---|---|
| 提示向量联合训练 | `${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_mv4_seed42/ptbxl/gated_max1000/gated_samples.npz` | random init | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/center_token_joint_seed42/` | `--synth_ratio 1.0 --seed 42` |
| 报告文本联合训练 | `${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_actual_report_mv4_step2000_seed42/ptbxl/gated_max1000/gated_samples.npz` | random init | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/report_text_joint_seed42/` | `--synth_ratio 1.0 --seed 42` |
| 默认文本联合训练 | `${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/ptbxl/gated_max1000/gated_samples.npz` | random init | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/default_text_joint_seed42/` | `--synth_ratio 1.0 --seed 42` |
| 无提示向量仅合成训练 | `${ECG_ADV_GRAD_ROOT}/vanilla_ecgtwin_synth_candidates_n20000_translated_randomany_seed42/ptbxl/samples.npz` | random init | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/no_token_synthetic_only_seed5042/` | `--synthetic_only --epochs 30 --patience 8 --seed 5042` |
| 中心提示向量预训练 | 真实 2000 固定划分 | `${ECG_ADV_GRAD_ROOT}/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc/best_model.pt` | `${ECG_ADV_GRAD_ROOT}/table_6_8_ablation_<tag>/center_token_pretrain_realfine_seed42/` | `--init_ckpt <ckpt> --lr 0.0001 --cosine_tmax 10 --patience 8 --epochs 25 --seed 9042` |

## 4. 必需数据与权重

最小复现需要以下类别的文件，文件体积大，不进入 git：

```text
${ECG_ADV_PTBXL_ROOT}/raw100.npy
${ECG_ADV_PTBXL_ROOT}/ptbxl_database.csv
${ECG_ADV_PTBXL_ROOT}/scp_statements.csv
${ECG_ADV_GRAD_ROOT}/splits/ptbxl_super5_seed42_train2000_val2000.json
${ECG_ADV_TRIPLE_ROOT}/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy
${ECG_ADV_TRIPLE_ROOT}/super5_minresample_full10_perglobal_20260503/best_model.pt
${ECG_ADV_GRAD_ROOT}/self_distill_v2_e21_v46_no_token_hardlabel_r10_seed8042_auroc/best_model.pt
${ECG_ADV_GRAD_ROOT}/self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc/best_model.pt
${ECG_ADV_GRAD_ROOT}/self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc/best_model.pt
${ECG_ADV_GRAD_ROOT}/self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc/best_model.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/**/prompt_token_bank.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/cache_v1/center_full_latents/ningbo.pt
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/cache_v1/ref_selection/ningbo_k500_seed42.json
${ECG_ADV_DATA_ROOT}/ecgtwin_prompt_token_super5/**/gated_samples.npz
${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_mv4_seed42/ptbxl/gated_max1000/gated_samples.npz
${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_actual_report_mv4_step2000_seed42/ptbxl/gated_max1000/gated_samples.npz
${ECG_ADV_GRAD_ROOT}/method_b_synth_candidates_classfallback_newgen_sameclass_cap20_seed42/ptbxl/gated_max1000/gated_samples.npz
${ECG_ADV_GRAD_ROOT}/vanilla_ecgtwin_synth_candidates_n20000_translated_randomany_seed42/ptbxl/samples.npz
${ECG_ADV_APP_DATA_ROOT}/models/efficientnetv2_super5.onnx
${ECG_ADV_APP_DATA_ROOT}/models/efficientnetv2_super5_fp16.engine
```

完整 artifact 清单见 `docs/artifact_manifest.json`。可用以下命令检查默认光盘归档环境：

```bash
bash scripts/final_round/run_thesis_reproduction.sh preflight
```

需要从原始数据和中间 checkpoint 完整重跑所有实验时，使用 full scope：

```bash
bash scripts/final_round/run_thesis_reproduction.sh preflight_full
```

如果 full scope 仍有缺项，该命令会同步写出：

```text
${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.json
${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/full_preflight_missing_artifacts.md
```

可用以下命令将当前机器上存在的可交付文件打包到
`${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/`：

```bash
bash scripts/final_round/run_thesis_reproduction.sh package_artifacts
```

默认打包只复制 `docs/artifact_manifest.json` 中允许随光盘交付且属于 archive scope
的条目。PTB-XL 原始数据、PTB-XL 全量预处理 cache、MIMIC/ECGTwin 作者训练 cache
等授权或体积敏感数据保留为 `package: false` 外部依赖；若需要判断完整重跑环境是否齐备，
使用 `preflight_full`。精确复现表 6.8 的部分生成池和表 6.6/6.7 完整训练重跑所需的
synthetic-pretrain 初始化 checkpoint 在 manifest 中标记为 `archive_required: false`；
它们仍是 full scope 的必需项，但默认光盘包会跳过，避免把“完整重跑缺失件”误报为
“毕业设计演示归档缺失件”。
打包目录会额外写出 `artifact_manifest.resolved.json`、`checksums.sha256`、
`missing_artifacts.json` 和 `missing_artifacts.md`，用于记录 archive scope 中未能打包的
可交付文件。完整重跑仍需补齐的外部数据和 `archive_required: false` 产物以
`preflight_full` 输出为准；如确实需要把 full-rerun-only 文件也纳入打包检查，可直接运行
`package_thesis_artifacts.py --include-rerun`。

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
  apps scripts/final_round scripts/triple_labels scripts/deploy scripts/ecgtwin_author_repro scripts/ecgtwin_gen methods/ecgtwin_gen util adversarial

uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q

uv run streamlit run apps/streamlit_ecg_demo/app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

TensorRT 只在有 CUDA、可用 TensorRT wheel/engine 且当前 GPU 架构匹配时保证可运行；如果换机器，优先保留 ONNX 和 PyTorch checkpoint，然后在目标机器上重新执行 `build_trt`。
