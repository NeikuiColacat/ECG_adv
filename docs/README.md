# Docs

当前 `docs/` 只保留论文归档主线、系统演示和最终交付需要直接引用的文档。历史方案、旧实验记录和被新版替代的笔记统一迁入 `legacy/docs/`。

## 论文归档主线

| 文件 | 用途 |
|---|---|
| `thesis_reproduction.md` | `thesis.md` 表格、图和代码入口的人工可读映射 |
| `thesis_repro_manifest.json` | 论文复现实验的机器可读索引 |
| `artifact_manifest.json` | 必需数据、权重、生成池、ONNX/TensorRT 和结果文件清单 |
| `thesis_archive_cleanup_standard.md` | 本归档分支的整理验收标准 |

## ECG 质量验证

| 文件 | 用途 |
|---|---|
| `ecgtwin_super5_digital_gt_validation.md` | ECGTwin super5 生成样本的数字心电规则验证报告 |
| `ecgtwin_super5_digital_gt_validation.json` | 上述验证的结构化结果 |
| `ecg_digital_thresholds.md` | 数字心电质量阈值与判定标准 |

## 论文材料

| 路径 | 用途 |
|---|---|
| `../artifacts/figures/streamlit_demo/` | Streamlit 演示截图 |
| `../artifacts/figures/generated_ecg_examples/` | 论文图 3.3 合成 ECG 示例图 |
| `final_round/` | 任务书、开题报告、中期报告、指导记录的 Markdown 版本 |
| `pipelines/README.md` | 说明旧 pipeline 文档已迁入 legacy |

## 历史材料

`legacy/docs/` 保存 PN2021-C、TA-OMAT、AdvDiff、AugMix、Tier-M、旧 center-style prompt token、自蒸馏、早期导师汇报和本地论文索引等非主线材料。主线复现文档不依赖这些文件。
