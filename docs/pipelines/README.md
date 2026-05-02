# Pipeline Memory

本目录保存项目长期维护的标准化流程文档。它们是论文实验、复现实验和后续代码改动的事实依据。

## 文档定位

| 文件 | 范围 |
|---|---|
| `efficientnetv2_training_pipeline.md` | EfficientNet1DV2 从 PTB-XL 训练到 PN2021/MIMIC 评测的全过程 |
| `super5_label_mapping_pipeline.md` | PTB-XL、MIMIC、PN2021 的 super5 标签映射标准 |
| `ecgtwin_author_repro_pipeline.md` | 复现 ECGTwin 作者 IBE + DiT 两阶段训练的技术细节 |
| `dataset_preprocessing_pipeline.md` | PTB-XL、MIMIC、PN2021 的统一信号预处理、采样率、长度和导联规则 |
| `center_style_classifier_pipeline.md` | 中心数据风格分类器 / target-likeness probe / C2ST 的构造、训练、测试和防泄漏规则 |
| `pn2021_v3_reevaluation_pipeline.md` | PN2021 super5 v3 重新映射后的 clean 7-center AUROC/AUPRC 重评测 |
| `ecgtwin_center_prompt_token_pipeline.md` | 基于 textual inversion / dataset interfaces 思想的 ECGTwin 目标中心 prompt-token 架构 |
| `latent_hull_online_at_pipeline.md` | 参考 SA-AET 的 Latent-Hull TA-OMAT 在线对抗训练计划 |
| `pn2021_c_corruption_benchmark_pipeline.md` | 参考 ImageNet-C 的 PN2021-C 中心数据增强/腐蚀鲁棒性评测 |
| `ptbxl_ecgtwin_offline_pgd_at_pipeline.md` | PTB-XL 抽 1000 refs，经 ECGTwin 生成 10000 super5 synthetic，再做离线 VAE latent-PGD 对抗训练的计划 |
| `ptbxl_prompt_token_boundary_at_pipeline.md` | 训练 5 个 PTB-XL source-style prompt token，并只接收 target probability 0.50-0.60 的 Latent-Hull 边界对抗样本 |
| `next_auto_execution_plan.md` | 当前四项任务的自动执行顺序、产物和阻塞条件 |

## Docs 还是 Skill

这些完整流程应该放在 `docs/pipelines/`，原因是它们需要给人读、给论文引用、给实验记录追溯。Skill 只适合保存 agent 执行时必须快速遵守的少量硬规则，例如数据路径、super5 类顺序、NORM guard、输出日志要求。

维护原则：

- 改训练脚本、评测脚本或标签逻辑时，同步更新对应 pipeline 文档。
- pipeline 文档优先写稳定规则；临时实验参数写到 run config 或实验记录里。
- 如果某条规则已经进入代码实现，文档要写明源码位置。
- 如果某条规则还没有代码实现，文档要明确写成 TODO/缺口，不要把计划写成已完成事实。
