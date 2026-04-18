# ECG Adversarial Generation Project

## Overview
AdvDiff-based adversarial training pipeline for improving DeepECG EfficientNet cross-center generalization.
Pipeline: ECGTwin generates boundary-hard ECG samples -> finetune EfficientNet via residual adapter -> evaluate AUPRC.

## Environment
- Python: `/root/miniforge3/envs/ECGTwin/bin/python`
- Conda env: `ECGTwin` (activate not needed, use full python path)
- GPU: RTX 4090 24GB
- Root disk: ~4GB free -> store large files in `/root/autodl-tmp/`
- `/root/autodl-tmp/` has ~28GB free

## Key Paths
- PTBXL data: `/root/ECG_adv_Gen/datasets/PTBXL/`
- PTBXL encoded (VAE): `/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt`
- EfficientNet weights: `/root/ECG_adv_Gen/model/DeepECG/weights/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt`
- ECGTwin model: loaded via `util/ecgtwin_utils.py` -> `ECGTwinWrapper`
- v2 baseline model: `/root/autodl-tmp/crosscenter_v2/best_model.pt`
- Experiment outputs: `/root/ECG_adv_Gen/outputs/` (hard_samples 等；大件在 `/root/autodl-tmp/`)
- Docs: `/root/ECG_adv_Gen/docs/`（训练细节、gap 报告、项目结构等）
- Legacy 归档: `/root/ECG_adv_Gen/trash/`

## Running Scripts
Always use the full Python path:
```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/run_mi_experiment.py
```

## Architecture
- `scripts/crosscenter_v2/` - **v2 baseline 训练+评测 pipeline**（26 类 SNOMED, EfficientNet1DV2 s_v2）
- `adversarial/` - AdvDiff 对抗生成 + Adapter fine-tune（77 类）
  - `efficientnet_victim.py` - Differentiable EfficientNet wrapper (latent -> 77-class logits)
  - `efficientnet_adapter.py` - Residual adapter on frozen backbone (~20K params)
  - `adv_generate.py` - Boundary-guided AdvDiff sample generation
  - `finetune.py` - Adapter training loop
  - `evaluate.py` - AUPRC/AUROC evaluation
  - `label_mapping.py` - PTBXL SCP <-> EfficientNet 77-class mapping
- `center_token/` - ECGTwin 中心 token embedding
- `util/` - Shared utilities (ECGTwin wrapper, lead mapping, etc.)
- `scripts/` - Runnable experiment scripts (`run_*.py` 入口)
- `methods/` - 未来方法研究入口（augmix / ecgtwin_gen / advdiff，空壳 + README）
- `docs/` - 所有技术文档 + PDF 参考
- `model/ECGTwin/` - ECGTwin model code (DiT + VAE)
- `model/DeepECG/` - DeepECG EfficientNet (JIT frozen)
- `trash/` - 归档区（legacy 代码、旧实验产物，不参与当前 pipeline）

## Important Notes
- EfficientNet is JIT-compiled: `torch.jit.load()`, supports autograd but weights are frozen
- EfficientNet input: (B, 12, 2500) @ 250Hz, scaled by MHI_FACTOR = 1/0.0048
- VAE decoder has in-place `/= 0.18215` that breaks autograd -> use out-of-place division
- ECGTwin lead order differs from PTBXL -> use `ECGTWIN_TO_PTBXL_INDICES` for reordering
