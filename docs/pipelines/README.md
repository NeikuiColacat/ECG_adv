# Pipeline Memory

本目录保存项目长期维护的标准化流程文档。它们是论文实验、复现实验和后续代码改动的事实依据。

## 文档定位

| 文件 | 范围 |
|---|---|
| `efficientnetv2_training_pipeline.md` | EfficientNet1DV2 从 PTB-XL 训练到 PN2021/MIMIC 评测的全过程 |
| `super5_label_mapping_pipeline.md` | PTB-XL、MIMIC、PN2021 的 super5 标签映射标准 |
| `ecgtwin_author_repro_pipeline.md` | 复现 ECGTwin 作者 IBE + DiT 两阶段训练的技术细节 |
| `dataset_preprocessing_pipeline.md` | PTB-XL、MIMIC、PN2021 的统一信号预处理、采样率、长度和导联规则 |

## Docs 还是 Skill

这些完整流程应该放在 `docs/pipelines/`，原因是它们需要给人读、给论文引用、给实验记录追溯。Skill 只适合保存 agent 执行时必须快速遵守的少量硬规则，例如数据路径、super5 类顺序、NORM guard、输出日志要求。

维护原则：

- 改训练脚本、评测脚本或标签逻辑时，同步更新对应 pipeline 文档。
- pipeline 文档优先写稳定规则；临时实验参数写到 run config 或实验记录里。
- 如果某条规则已经进入代码实现，文档要写明源码位置。
