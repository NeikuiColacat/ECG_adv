# 2026 毕业归档光盘交付说明

本文档把学院 `00-毕业设计提交材料清单与要求-发布 -2026新.docx` 中“光盘”条目落实到本仓库。光盘要求除 1-12 项毕业设计文档外，还包含：

1. 作品的源代码、软件系统等；
2. 软、硬件展示材料（视频或照片、图片）；
3. 答辩 PPT。

本仓库对应“作品源代码、软件系统”。答辩 PPT 和论文文档由上级毕业归档目录统一放置。

## 交付件

建议光盘中保留以下文件或目录：

| 交付内容 | 建议位置 | 说明 |
|---|---|---|
| 源码仓库 | `ECG_adv/` | 当前清理后的 thesis archive 分支源码，不包含 `.venv/`、缓存、原始数据集、训练级 checkpoint。 |
| 干净源码包 | `${ECG_ADV_DATA_ROOT}/thesis_archive_artifacts/source_code/ECG_adv_source_*.tar.gz` | 由 `package_source` stage 生成，只包含 git 跟踪或未忽略源码文件，适合直接放入光盘。 |
| 大文件 artifact 包 | `migrate_files/*.tar.gz` 或上级 `ecg_grad_repro_no_pn2021_*.tar.gz` | 权重、demo 数据、生成池、ONNX/TensorRT 文件和轻量复现实验结果。 |
| artifact 校验清单 | `migrate_files/SHA256SUMS_*.txt`、`docs/artifact_manifest.json` | 用于检查大文件是否完整。 |
| 论文表图证据 | `artifacts/evidence_pack/`、`artifacts/figures/` | 小型 JSON/CSV/PNG 证据，已随源码跟踪。 |
| 系统演示入口 | `apps/streamlit_ecg_demo/app.py` | 通过统一 stage 启动，不建议直接手动拼命令。 |
| 复现总入口 | `scripts/final_round/run_thesis_reproduction.sh` | 统一执行环境检查、artifact 恢复、preflight、demo、表图复现。 |
| 答辩 PPT | 上级 `defense_ppt/` | HTML 版 `index.html` 与已转换的 PPTX 文件。 |

## 新机器恢复流程

在源码仓库根目录执行：

```bash
uv sync --all-groups
export ECG_ADV_DATA_ROOT="${HOME}/autodl-tmp"
bash scripts/final_round/run_thesis_reproduction.sh restore_artifacts
bash scripts/final_round/run_thesis_reproduction.sh env
bash scripts/final_round/run_thesis_reproduction.sh preflight
bash scripts/final_round/run_thesis_reproduction.sh package_source
```

`preflight` 是毕业归档/答辩演示默认门禁。它检查可随光盘交付的 archive/demo scope。若需要完整从原始数据重跑所有长训练任务，再执行：

```bash
bash scripts/final_round/run_thesis_reproduction.sh preflight_full
```

`preflight_full` 允许报告缺少 PTB-XL/MIMIC 原始或派生数据、完整重跑缓存、历史中间生成池等外部依赖。这些缺口不代表毕业归档演示失败；它们属于 full rerun scope，需在授权数据和足够存储空间存在时补齐。

## 与论文的对应关系

毕业论文 `毕业论文.docx` / `thesis.md` 的代码主线对应以下模块：

- ECGTwin IBE + DiT 两阶段复现：`scripts/ecgtwin_author_repro/`；
- ECGTwin prompt-token/no-token 合成和质量门控：`scripts/ecgtwin_gen/`、`methods/ecgtwin_gen/prompt_token/`、`util/ecg_digital_features.py`；
- PTB-XL super5 少样本异常检测：`scripts/triple_labels/`；
- 论文最终复现和归档打包：`scripts/final_round/`；
- Streamlit/TensorRT 原型系统：`apps/streamlit_ecg_demo/`、`scripts/deploy/`；
- 论文证据包：`artifacts/evidence_pack/`、`docs/thesis_reproduction.md`、`docs/thesis_repro_manifest.json`、`docs/artifact_manifest.json`。

不属于论文主线的 PN2021、Latent-Hull、PGD、TA-OMAT、自蒸馏等探索内容已集中到 `legacy/`。严格论文路线不应从 `legacy/` 导入代码，也不应把 `legacy/` 中的结果写成正文主实验结论。

## 不应放入源码仓库的内容

以下内容应作为外部 artifact 或本地环境保留，不进入 git 源码提交：

- `.venv/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/`；
- PTB-XL/MIMIC 原始数据和大规模预处理缓存；
- 训练级 `.pt`、`.pth`、`.onnx`、`.engine`、`.npy`、`.npz`；
- 大规模生成池、临时训练日志、wandb/runs/checkpoints；
- 主机相关模型仓库软链接目标；
- Windows 下载附属 `Zone.Identifier` 文件。

例外：`artifacts/samples/generated_ecg_examples/thesis_selected_samples.npz` 是论文图 3.3 的小型可视化样本，保留在 git 中。

## 验收命令

归档前至少运行：

```bash
git status --short --branch
uv lock --check
bash -n scripts/final_round/*.sh
uv run pytest apps/streamlit_ecg_demo/tests scripts/final_round/tests util/tests -q
uv run python -m json.tool docs/thesis_repro_manifest.json >/dev/null
uv run python -m json.tool docs/artifact_manifest.json >/dev/null
bash scripts/final_round/run_thesis_reproduction.sh thesis_assets
bash scripts/final_round/run_thesis_reproduction.sh preflight
bash scripts/final_round/run_thesis_reproduction.sh package_source
bash scripts/final_round/run_thesis_reproduction.sh package_artifacts
```

若目标机器要现场演示，还需启动：

```bash
bash scripts/final_round/run_thesis_reproduction.sh streamlit
```
