# Docs

当前 `docs/` 只保留毕设主线和最终写作需要直接引用的文档。历史方案、旧试验记录和被新版替代的笔记已归档到 `trash/docs_cleanup_20260501/`，没有直接删除。

## 主线结论

| 文件 | 用途 |
|---|---|
| `experiment_summary_for_advisor.md` | 面向导师汇报的整体实验故事和结论 |
| `module_ablation_1_real_vs_synth.md` | TA-OMAT / real-anchor / synthetic-anchor 对比与模块消融 |
| `triple_labels_metrics_for_advisor.md` | EfficientNet1DV2 在 super5/sub23/pn26 等设置下的指标汇总 |
| `label_strategy.md` | 历史标签调研；当前 super5 映射以 `pipelines/super5_label_mapping_pipeline.md` 和代码为准 |
| `synth_anchored_super5_pilot_summary_final.md` | synthetic-anchor super5 最终 pilot 总结 |
| `center_token_v2_results.md` | center token v2 扩展实验和消融结果 |

## ECG 质量验证

| 文件 | 用途 |
|---|---|
| `ecgtwin_super5_digital_gt_validation.md` | ECGTwin super5 生成样本的数字心电规则验证报告 |
| `ecgtwin_super5_digital_gt_validation.json` | 上述验证的结构化结果 |
| `ecg_digital_thresholds.md` | 数字心电质量阈值与判定标准 |

## 训练细节

| 文件 | 用途 |
|---|---|
| `training/efficientnet_training.md` | EfficientNet1DV2 架构、训练配置和日志要求 |
| `pipelines/` | 长期维护的标准化 pipeline 记忆文档 |

## 论文材料

| 路径 | 用途 |
|---|---|
| `final_round/` | 任务书、开题报告、中期报告、指导记录的 Markdown 版本 |
| `papers/INDEX.md` | 本地论文索引 |
| `references/` | 本地参考 PDF，通常不纳入 git |

## 归档策略

- 旧 no-IBE diffusion 方案、早期 synth pilot、旧 center-token/IBE 风格迁移记录、旧 Tier-M/AugMix 文档、旧标签审计和旧项目结构说明，统一放入 `trash/docs_cleanup_20260501/`。
- `trash/` 默认不跟踪，适合保留临时归档材料；需要恢复时直接从该目录移回。
