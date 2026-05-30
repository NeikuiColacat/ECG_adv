# methods/advdiff/

**研究方向**：AdvDiff（latent-space 对抗扩散）生成决策边界困难样本，fine-tune adapter，改善模型鲁棒性。

**状态**：✅ 已实现。本目录仅作文档入口，代码位于 `adversarial/`。

## 代码位置

| 模块 | 路径 |
|---|---|
| 对抗生成主逻辑 | `adversarial/adv_generate.py` |
| 可微分 EfficientNet 封装 | `adversarial/efficientnet_victim.py` |
| Residual adapter（~20K 参数） | `adversarial/efficientnet_adapter.py` |
| Fine-tune 循环 | `adversarial/finetune.py` |
| 评估 | `adversarial/evaluate.py` |
| PTBXL SCP ↔ EfficientNet 77 类映射 | `adversarial/label_mapping.py` |

## 入口脚本

| 步骤 | 脚本 |
|---|---|
| Step 0: PTBXL VAE 编码 | `scripts/run_prepare_ptbxl.py` |
| Step 3: 评估 baseline | `scripts/run_evaluate_baseline.py` |
| Step 5: AdvDiff 生成困难样本 | `scripts/run_adv_generate.py` |
| Step 6: 合并困难样本 | `scripts/prepare_hard_samples.py` |
| Step 7: Fine-tune adapter | `scripts/run_finetune.py` |
| Step 8: 评估 fine-tuned | `scripts/run_evaluate_finetuned.py` |
| MI 专题实验 | `scripts/run_mi_experiment.py` |
| 在线对抗训练 v4（SA-AET 启发） | `scripts/run_online_training_v4.py` |

## 参考实现

- 原论文实现：`model/advdiff/`
- SA-AET（v4 online 借鉴的 triangular 插值 + EWA anchor）：`model/SA-AET/`
