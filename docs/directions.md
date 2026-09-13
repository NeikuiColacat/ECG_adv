# 三个工作方向

这是一份导航，不是新的实验配置或执行授权。机器可读目录见
[configs/directions.yaml](../configs/directions.yaml)。目录只列现有代表入口；
Ningbo 示例不等于全四中心快捷启动，完整网格仍以注册表为准。
本导航不授权移动或删除 worktree／配置／归档，不改 canonical 路径、配方或历史结果身份。
已单独获批完成三树整理，目录与恢复记录见
[整理记录](refactor_cleanup/three_direction_consolidation.md)。

| 方向 | 当前工作主方法 | 主要对照 |
|---|---|---|
| ECG LLM | PULSE-7B、ECG-R1 图片输入评估；PULSE visual JSD 微调 | 原始模型、clean、单链、三链；旧 LoRA 波形／图像混合 |
| 传统两阶段 SimCLR | 两链 AugMix＋SimCLR → rotating4＋VAE-LHAT | 无 VAE、VAE-only、A0、锁定 A1 |
| 传统单阶段 JSD | 已选 R18：JSD1.5＋LHAT 监督替换权重 0.20 | 单链、两链无 VAE、同长预算 A1、R19 机制消融 |

三个方向均保留开发证据边界。用户选择第三方向的既有 R18 配方，不代表
重新按成绩选优，也不把它升级为论文最终方法。

## 1. ECG LLM：PULSE / ECG-R1

- R1 接入检查：[GPU-exact validation](../configs/experiments/ecg_image_r1_elastic_validation_gpu_exact.yaml)。
- R1 全量入口：[GPU-exact full](../configs/experiments/ecg_image_r1_elastic_full_gpu_exact.yaml)。
- 两模型已完成结果的配对分析：[full comparison](../configs/experiments/ecg_image_r1_pulse_full_comparison.yaml)。
- PULSE visual 宁波训练：[clean](../configs/experiments/pulse_visual_clean_ningbo.yaml)、
  [单链 JSD](../configs/experiments/pulse_visual_single_ningbo.yaml)、
  [三链 JSD](../configs/experiments/pulse_visual_three_ningbo.yaml)；
  [固定 512 条评估](../configs/experiments/pulse_visual_eval_ningbo.yaml)。
- 既有 LoRA 对照：[波形混合](../configs/experiments/pulse_subset.yaml)、
  [图像混合](../configs/experiments/pulse_pixel_subset.yaml)。

R1 当前验证的是 image-only，不冒称作者的波形＋图片联合路径。
PULSE 原始全量基线是 comparison 配置引用的冻结外部产物；报告入口不重新推理。
`dual_jsd_workflow.yaml` 同时启动 Founder，不能当 LLM-only 默认入口。
四中心 visual 网格定位 [active_scripts.yaml](../configs/active_scripts.yaml)
的 `dual_jsd_20260911`；全量证据定位
[active_evidence_registry.yaml](../configs/active_evidence_registry.yaml)
的 `image_llm_full_elastic_20260907`。

## 2. 传统两阶段 SimCLR

主方法：[augmix_simclr_lhat](../configs/train/methods/augmix_simclr_lhat.yaml)；
训练配置：[PN2021.yaml](../configs/train/PN2021.yaml)。

| 骨干 | 宁波训练 | 宁波评估 | 四中心汇总 |
|---|---|---|---|
| EfficientNet | [train](../configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_ningbo.yaml) | [eval](../configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_eval_ningbo.yaml) | [matrix](../configs/experiments/manual_refactor_pn2021_effnet_augmix_simclr_lhat_matrix.yaml) |
| ECGFounder | [train](../configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_ningbo.yaml) | [eval](../configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_eval_ningbo.yaml) | [matrix](../configs/experiments/manual_refactor_pn2021_ecgfounder_augmix_simclr_lhat_matrix.yaml) |

必要消融为 `augmix_simclr_matched_no_vae`、`vae_lhat_only`、`a0_clean_v1`、
`a1_corrupt_ft_rot4_v1`。双骨干代表入口已列于导航 YAML；完整网格定位
`latest_mainline`、`prospective_component_ablation_2x2_r0`、
`prospective_matched_base_a0_a1`。
锁定 A1 已包含腐蚀增强，不是 clean-only；它与 SimCLR 的总计算量不相等。

## 3. 传统单阶段 JSD：R18

主方法：[R18 replace0p2](../configs/train/methods/a1_rot4_two_chain_balanced_jsd1p5_vae_lhat_replace0p2.yaml)；
训练配置：[R13 长预算](../configs/train/PN2021_a1_joint_long_budget_r13.yaml)。
无 SimCLR、无阶段切换；JSD 权重 1.5。LHAT 的 0.20 是替换 clean 监督损失的
权重份额，不是额外叠加 20% 总损失：clean／rotating／AugMix／LHAT 监督权重为
0.05／0.25／0.50／0.20，JSD 项另计；拒绝的 LHAT 样本回退到 clean BCE。

| 骨干 | 宁波训练 | 宁波评估 |
|---|---|---|
| EfficientNet | [train](../configs/experiments/manual_refactor_pn2021_effnet_a1_vae_replace0p2_joint_ningbo.yaml) | [eval](../configs/experiments/manual_refactor_pn2021_effnet_a1_vae_replace0p2_joint_eval_ningbo.yaml) |
| ECGFounder | [train](../configs/experiments/manual_refactor_pn2021_ecgfounder_a1_vae_replace0p2_joint_ningbo.yaml) | [eval](../configs/experiments/manual_refactor_pn2021_ecgfounder_a1_vae_replace0p2_joint_eval_ningbo.yaml) |

必要对照：R17 单链／两链无 VAE、R13 同长预算 A1，以及原锁定短预算 A1。
完整网格定位 `a1_vae_clean_replacement_r18_20260902`、
`a1_balanced_jsd1p5_long_r17_20260902`、`a1_joint_long_budget_r13_20260902`。
EffNet 使用 50 epochs／200 steps，Founder 使用 60 epochs／480 steps；
不能把相对短预算 A1 的全部差异归因于组件。

R19 保留 baseline／mixed-M20／no-contract 和完整 seed 身份；R20 是同 seed
重跑，不是新独立种子。R18/R13 的 replicate 0 与真正 R19 seed0 的 namespace
不同，不能按数字 0 合并去重。

9/11 的 clean-BCE／clean-BCE＋12JSD 及
[宽度消融](../configs/experiments/founder_jsd_width_workflow.yaml) 是两阶段附属实验，
不是此单阶段主方法；这些新目标／宽度正式实验目前只覆盖 Founder。
更早的 sweep、未选参数、失败／重试和旧实现保留为历史来源，不在整理中删除。

## 传统训练代码阅读顺序

两条传统方向共用以下调用链，不需要各复制一个 trainer：

```text
boot_scripts/run_experiment.py → boot_scripts/train_pn2021.py
→ core/train_PN2021.py → core/online_trainer.py
```

| 职责 | 当前唯一入口／位置 | 修改时的边界 |
|---|---|---|
| 启动与产物登记 | `run_experiment.py`、`util/run_record.py` | 配置闭包、外部输出目录、run card 和文件 hash |
| K500 数据与可选 latent pool | `core/train_PN2021.py::train_pn2021` | 不在 trainer 内重新选样或读取 heldout |
| 训练配置与参数覆盖 | `online_trainer.py::load_online_train_config`、`resolve_online_training_parameters` | 骨干／全局默认值 → 显式覆盖 → 校验；已配置的 scheduler horizon 不随 epochs 自动缩短 |
| 配方、视图与损失 | `core/methods/registry.py`、`runtime.py`；`online_trainer.py::_compute_objective` | 保留 recipe 身份、视图顺序、loss 权重与梯度边界 |
| 阶段与 optimizer | `online_trainer.py::_run_augmix_stage1`、`train_online_model` | SimCLR 有 Stage 1；单阶段 R18 不走该阶段；每个 base batch 一次 outer step |
| 训练记录 | `_training_lineage`、`OnlineTrainingResult.describe`、epoch 末写入 | lineage、history、checkpoint 与最终结果各自保留原 schema |

第一轮仅简化参数覆盖：先构造默认字典，再统一应用显式覆盖，保留参数顺序、
类型转换、Stage-1 缺省值与错误语义。未改 YAML、配方、数值训练或日志字节格式。
日志 writer 暂不合并：训练器使用 `.文件名.tmp`，run recorder 使用 `文件名.tmp`；
字节内容相似不代表失败路径和临时文件契约完全相同。后续先验证这些边界，
再决定是否复用，不新增通用训练框架或仅用于转发的模块。

## 共用边界与启动

传统分类器共用固定 PTB-XL source、K500、Super5 mapping `555ec85d5b51`、
100 Hz canonical 输入和 K500-ref-excluded Clean／PN2021-C 评估。
图片模型使用自己明确的 native500 图像协议，不能混成同一个模型输入契约。
源模型身份见 [ptbxl_source_v1.yaml](../configs/baselines/ptbxl_source_v1.yaml)。

仅对选定的真实 experiment YAML 使用既有 launcher，先 `--dry-run`；
不要把 `configs/directions.yaml` 传给 launcher。真实执行前仍须资源检查，
已有输出目录不能覆盖。部分历史权重在登记位置缺失：只记录待定位，不自动
搜索全盘、删除、重训或改写旧 hash；元数据完整不等于 checkpoint 可直接加载。
