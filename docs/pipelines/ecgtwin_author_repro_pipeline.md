# ECGTwin Author Reproduction Pipeline

本文档记录复现 ECGTwin 作者训练流程的标准技术细节。目标是为毕设论文提供可追溯的 loss 曲线、val 曲线、配置和 checkpoint。

## 总体结构

ECGTwin 作者流程分两阶段：

```text
Stage 1: Individual Base Extractor (IBE)
Stage 2: Latent Diffusion noise predictor (DiT by default, Unet optional)
```

本项目复现入口：

```text
scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh
scripts/ecgtwin_author_repro/train_ibe_repro.py
scripts/ecgtwin_author_repro/train_dit_repro.py
```

原作者代码保留在：

```text
model/ECGTwin/
```

## 数据输入

默认使用 ECGTwin paired MIMIC latent cache：

```text
/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt
/root/autodl-tmp/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt
```

说明：

- paired cache 用于原作者 IBE 的 reference-target 训练。
- 当前 thesis mainline 不应默认依赖 paired MIMIC；作者复现可以使用它。
- 大输出目录必须放在 `/root/autodl-tmp/`。

## Stage 1 IBE

作者论文配置：

```text
Transformer encoder layers = 3
model/base vector dim = 256
effective batch size = 65,536
micro batch size = 512
gradient accumulation = 128
optimizer = AdamW
lr = 1e-3
epochs = 40
cref text/reference substitution probability = 0.15
```

当前复现脚本默认：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_ibe_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/full_dit_main/ibe_stage1 \
  --epochs 40 \
  --batch_size 65536 \
  --mini_batch_size 512 \
  --val_batch_size 256 \
  --num_workers 0 \
  --max_val_batches 64 \
  --log_every 1 \
  --amp \
  --amp_dtype bf16 \
  --matmul_precision high \
  --device cuda
```

4090D 推荐加速参数：

```text
--amp --amp_dtype bf16 --matmul_precision high
--pin_memory --persistent_workers --prefetch_factor 2
num_workers 从 0/4/8/12 试，不要默认拉满
```

注意：`batch_size=65536` 是有效 batch，不是一次性放进显存的真实 batch；通过 `mini_batch_size=512` 累积 128 步实现。

## Stage 2 DiT Diffusion

作者论文配置：

```text
VAE latent = R^{4 x 128}
T = 1000 diffusion training timesteps
beta schedule = linear [8.5e-4, 1.2e-2]
base vector dropout = 0.15
DiT hidden size = 256
DiT depth = 7
DiT heads = 8
lr is most sensitive hyperparameter
```

当前复现脚本默认：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_author_repro/train_dit_repro.py \
  --output_dir /root/autodl-tmp/ecgtwin_author_repro/full_dit_main/dit_stage2 \
  --ibe_path /root/autodl-tmp/ecgtwin_author_repro/full_dit_main/ibe_stage1/checkpoints/IBE_best.pth \
  --epochs 30 \
  --batch_size 512 \
  --val_batch_size 512 \
  --num_workers 0 \
  --max_val_batches 64 \
  --log_every 20 \
  --amp \
  --amp_dtype bf16 \
  --matmul_precision high \
  --device cuda
```

## 一键复现入口

```bash
ROOT=/root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512 \
NUM_WORKERS=8 \
AMP_ARGS="--amp --amp_dtype bf16 --matmul_precision high" \
DATALOADER_ARGS="--pin_memory --persistent_workers --prefetch_factor 2" \
bash scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh
```

## 必须保存的日志和图

每个 stage 必须保存：

```text
run_config.yaml
train.log
metrics.jsonl
loss_curve.csv
loss_curve.png
checkpoints/latest.pt or latest.pth
checkpoints/best.pt or best.pth
```

pipeline 根目录保存：

```text
pipeline.log
```

论文记录至少提取：

- IBE train loss 曲线。
- IBE eval score 曲线。
- DiT train diffusion MSE 曲线。
- DiT val diffusion MSE 曲线。
- 每阶段训练耗时。
- 最佳 checkpoint 路径。

## 与毕设主线的关系

作者复现的作用是：

- 证明我们理解并能复现 ECGTwin 架构和训练细节。
- 为论文提供对照背景。
- 作为 thesis mainline 的架构复现与实验背景，不直接等同于最终增强方法。

毕设主线仍然是：

```text
PTB-XL super5 real ECG
-> EfficientNet1DV2 super5 baseline
-> ECGTwin author IBE + DiT reproduction
-> ECGTwin VAE latent manifold / target-center anchors
-> TA-OMAT and synth-anchor ablations
-> PN2021 7-center downstream utility
```

不要把作者 IBE paired MIMIC 复现流程写成最终增强方法本身；它主要提供 ECGTwin 架构复现、VAE latent manifold 和对照背景。
