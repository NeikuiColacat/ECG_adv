# methods/ecgtwin_gen/

**研究方向**：用 ECGTwin（DiT + VAE 条件扩散）生成少量目标域 OOD 样本，用于 fine-tune baseline，改善跨中心性能。

**状态**：🚧 待填充（center token 已实现，还需目标域 anchor 采样策略）

## 起点参考

- **ECGTwin 封装**：`util/ecgtwin_utils.py`（加载模型 / 条件编码 / 可微分采样）
- **中心 token 嵌入**：`center_token/`（已实现，训练脚本 `scripts/run_train_center_token.py`，生成 `scripts/run_generate_with_center.py`）
- **AdvDiff 边界生成**：`adversarial/adv_generate.py`（latent-space DDPM 引导，同时可作为 ECGTwin 生成的参考）
- **ECGTwin 模型代码**：`model/ECGTwin/`

## 研究设计建议

1. 采样策略：从 PN2021 单一中心（如 Chapman）取 N 个 anchor ECG → ECGTwin 条件生成 K 个变体
2. Fine-tune：baseline + 生成样本混合训练
3. 评测：在其他中心 zero-shot 看 gap 变化
