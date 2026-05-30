# ECGTwin Author Reproduction Pipeline

本文档记录复现 ECGTwin 作者训练流程的标准技术细节。目标是为毕设论文提供可追溯的 loss 曲线、val 曲线、配置和 checkpoint。

## 总体结构

ECGTwin 作者流程分两阶段：

```text
Stage 1: Individual Base Extractor (IBE)
Stage 2: Latent Diffusion noise predictor (DiT by default, Unet optional)
```

作者推理链路：

```text
reference ECG/text/patient info
-> VAE encoder + IBE
-> base_vector

target diagnostic prompt
-> BERT tokenizer + nomic text encoder
-> text_embed/text_embed_mask

DDPM/DiT noise predictor:
noise_predictor(x_t, t, text_embed, text_embed_mask, pat_info, base_vector)
-> VAE decoder
-> ECGTwin/MIMIC lead-order ECG, length 1024
```

关键约束：

- `base_vector` 由 reference ECG/text/patient info 产生，主要走 AdaLN/timestep 条件路径。
- 目标诊断文本走 `text_embed` cross-attention 条件路径。
- `text_embed_mask` 不能是全 0；如果使用 null text，也要给一个有效 token mask。
- 官方代码支持 open-vocabulary prompt embedding，但没有原生 textual inversion/token training 流程。

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

### 2026-05-01 partial run status

当前最长一次 4090D 复现运行：

```text
root: /root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512
stage: IBE stage1
config: effective batch 65536, micro batch 512, num_workers 8,
        pin_memory + persistent_workers + prefetch_factor 2, bf16 AMP
status: saved partial run through epoch 9; full author pipeline did not complete
```

已保存 IBE artifacts：

```text
ibe_stage1/run_config.yaml
ibe_stage1/train.log
ibe_stage1/metrics.jsonl
ibe_stage1/loss_curve.csv
ibe_stage1/loss_curve.png
ibe_stage1/checkpoints/IBE_best.pth
ibe_stage1/checkpoints/best.pt
ibe_stage1/checkpoints/latest.pt
```

指标：

| epoch | train_loss | eval_score | epoch_time_sec |
|---:|---:|---:|---:|
| 1 | 3.052135 | 0.718906 | 1039.9 |
| 2 | 1.694837 | 0.657278 | 1071.8 |
| 3 | 1.213244 | 0.627823 | 1064.6 |
| 4 | 0.919663 | 0.625451 | 1062.4 |
| 5 | 0.751632 | 0.628861 | 1056.3 |
| 6 | 0.634500 | 0.611151 | 1071.6 |
| 7 | 0.557742 | 0.611722 | 1067.1 |
| 8 | 0.498903 | 0.616669 | 1070.9 |
| 9 | 0.455224 | 0.611132 | 1070.0 |

重要解释：

- `eval_score` 是 IBE paired feature diagonal similarity，当前脚本按 higher is better 保存 best。
- 因此 `best.pt` / `IBE_best.pth` 仍来自 epoch 1，`latest.pt` 来自 epoch 9。
- epoch 9 后主训练进程被系统 kill，随后 orphan DataLoader workers 持有 stdout pipe 和 GPU memory；这些 worker 已手动终止。
- 该运行不能作为完整 ECGTwin author reproduction；只能作为 partial IBE training log/checkpoint。

后续建议：

- 不要用同样 `num_workers=8 + persistent_workers` 配置无脑重跑 10 小时。
- 若继续复现作者流程，先给 `train_ibe_repro.py` 增加 resume/latest checkpoint 支持，或使用 `num_workers=0/4` 且关闭 `persistent_workers` 做稳定性优先重跑。
- 在当前自动执行队列中，GPU 已切换到 PN2021-C missing-candidate 补评测。

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

## Center Prompt-Token 扩展

本项目接下来的增强主线不是重新训练 tokenizer，而是在 ECGTwin 的 `text_embed` 序列里插入
可学习的 center-class embedding。专项设计见：

```text
docs/pipelines/ecgtwin_center_prompt_token_pipeline.md
```

计划中的 token bank：

```text
CenterClassTokenBank[center] = 5 x 768
tokens = <center_CD>, <center_HYP>, <center_MI>, <center_NORM>, <center_STTC>
```

实现原则：

- 普通诊断文本仍用官方 `bert-base-uncased` + `nomic-ai/nomic-embed-text-v1.5` 得到。
- `<center_CLASS>` 由本项目 prompt compiler 替换成 learnable 768-d embedding 并追加到
  `text_embed` 序列，不依赖 BERT tokenizer 真的认识这个字符串。
- 每个目标中心直接使用 K=500 条 ECG 训练/测试 token；这些 ref ids 后续必须从
  fine-tune validation/eval 中排除。
- `base_vector` 使用同类 reference ECG 生成，不用 center token 改写 base-vector 本身。
- 已归档的 `trash/cleanup_20260506_legacy/methods_ecgtwin_gen/center_token/`
  是 256-d AdaLN/base-vector hook，不是 prompt token；
  后续只能作为 ablation 或旧方案参考。

当前建议 prompt fragment：

| class | prompt fragment | 第一版用途 |
|---|---|---|
| NORM | `sinus rhythm|normal ecg.` | 主线生成 |
| MI | `stemi|st elevation myocardial infarction|acute` | 主线生成 |
| STTC | `nstemi|non st elevation|t wave inversion` | 主线生成 |
| HYP | `left ventricular hypertrophy|high voltage` | token 训练/验证/消融，等 voltage gate 通过后再进主线 |
| CD | `left bundle branch block|lbbb` | token 训练/验证/消融，等 conduction gate 通过后再进主线 |

原因：当前数字 ECG quick validation 显示 NORM/MI/STTC 更稳定；HYP 和 CD 可以学习
center-class token，但不能在第一版下游增强中默认当作医学语义可靠的合成样本。

## 一键复现入口

```bash
ROOT=/root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512 \
NUM_WORKERS=8 \
AMP_ARGS="--amp --amp_dtype bf16 --matmul_precision high" \
DATALOADER_ARGS="--pin_memory --persistent_workers --prefetch_factor 2" \
bash scripts/ecgtwin_author_repro/run_author_repro_pipeline.sh
```

## Current Run Status

Started on 2026-05-01:

```text
root: /root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512
stage: IBE stage1 running
settings:
  effective batch_size=65536
  mini_batch_size=512
  num_workers=8
  AMP bf16
  pin_memory + persistent_workers + prefetch_factor=2
observed startup:
  train_size=6408782
  val_size=399499
  GPU memory ~= 5.9GB
  first optimizer steps ~= 10.5s/step
epoch 1:
  train_loss=3.052135
  eval_score=0.718906
  epoch_time=1039.9s
  artifacts written:
    metrics.jsonl
    loss_curve.csv
    loss_curve.png
    checkpoints/IBE_best.pth
    checkpoints/best.pt
    checkpoints/latest.pt
epoch 2:
  train_loss=1.694837
  eval_score=0.657278
  epoch_time=1071.8s
  latest.pt updated; best.pt/IBE_best.pth still from epoch 1 because current
  script treats higher eval_score as better.
epoch 3:
  train_loss=1.213244
  eval_score=0.627823
  epoch_time=1064.6s
  training continuing into epoch 4
epoch 4:
  train_loss=0.919663
  eval_score=0.625451
  epoch_time=1062.4s
epoch 5:
  train_loss=0.751632
  eval_score=0.628861
  epoch_time=1056.3s
epoch 6:
  train_loss=0.634500
  eval_score=0.611151
  epoch_time=1071.6s
  training continuing into epoch 7
note:
  eval_score is mean paired feature diagonal similarity from IBExtractor
  validation, so higher is better in the current script. This explains why
  best.pt/IBE_best.pth can remain from epoch 1 even while train_loss decreases.
```

Monitor:

```bash
tail -f /root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512/pipeline.log
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
-> ECGTwin center-class prompt tokens
-> Latent-Hull TA-OMAT and synth-anchor ablations
-> PN2021 7-center downstream utility
```

不要把作者 IBE paired MIMIC 复现流程写成最终增强方法本身；它主要提供 ECGTwin 架构复现、VAE latent manifold 和对照背景。
