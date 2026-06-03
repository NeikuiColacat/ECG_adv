# ECGTwin1024 VAE -> 500Hz Classifier Diagnostic

Date: 2026-06-02

## Question

当前 VAE500 在 matched direct40 fine-tune 之后没有带来额外增益。一个可能原因是：

```text
VAE500 reconstruction 很强，但 latent geometry 太偏 waveform detail，
不如 ECGTwin 原版 (4,128) latent 那样有语义压缩。
```

因此先做低成本诊断：不重训 VAE，直接把原版 ECGTwin VAE latent 接入 500Hz EfficientNet1DV2。

## Implementation

新增兼容路径：

```text
scripts/pgd_cross_center/synth_online_at_super5.py
  --vae_backend ecgtwin1024
  --input_len 5000
```

脚本现在会让原版 ECGTwin VAE：

```text
latent (B,4,128)
-> ECGTwin decoder
-> 1024 x 12 waveform
-> ECGTwin lead order -> PTB-XL lead order
-> interpolate 1024 -> 5000
-> per-sample global z-score
-> 500Hz EfficientNet1DV2
```

新增 anchor bundle 工具：

```text
scripts/vae500/rebuild_ecgtwin1024_anchor_bundle_for_v7_500hz.py
```

它保留旧 ECGTwin `(4,128)` latent，但重新生成：

```text
v7 Super5 labels
500Hz / 5000 samples real target signals
ref_meta.json for exclusion
class_trust.json
```

Pilot anchor root:

```text
/root/autodl-tmp/ecgtwin1024_to5000_anchors_v7_20260602
```

Training root:

```text
/root/autodl-tmp/ecgtwin1024_to5000_direct40init_pilot_v7_20260602
```

## Pilot Config

Matched with previous direct40-init VAE-only probes:

```text
init checkpoint: per-center direct40/ep10/no-adv best_model.pt
target_real_weight: 0
ptbxl_weight: 1
adv_weight: 0.30
classes_in_scope: NORM, MI, STTC
adv_label_mode: mixed_soft
adv_teacher_mix: 0.3
hull_M: 10
hull_lambda: 0.15
hull_steps: 5
hull_lr: 0.25
K_anchor: 300
epochs: 8
selection: K500-internal validation AUROC
```

## Results

The selected best checkpoint stayed at epoch 0 for both centers. SHA256 also
matches the direct40 initialization exactly.

| center | direct40 K500-val baseline | best post-update K500-val | last K500-val | selected |
|---|---:|---:|---:|---|
| cpsc_2018 | 0.9158 / 0.8556 | 0.8956 / 0.8037 | 0.8784 / 0.7782 | epoch 0 |
| ningbo | 0.8969 / 0.8526 | 0.8882 / 0.8438 | 0.8828 / 0.8366 | epoch 0 |

Training diagnostics:

| center | ASR mean | atk_anchor mean | loss_gain mean | decoded invalid |
|---|---:|---:|---:|---:|
| cpsc_2018 | 0.4127 | 0.5188 | 0.0405 | 0.0000 |
| ningbo | 0.3523 | 0.4536 | 0.0019 | 0.0000 |

Interpretation:

```text
The original ECGTwin latent branch is active and produces finite decoded ECGs,
but the updates still move away from the direct40 K500-validation optimum.
This weakens the hypothesis that VAE500 failure is only caused by its larger
(4,625) latent geometry.
```

## External 500Hz VAE Search

No drop-in model was found that is simultaneously:

```text
12-lead
10s / 500Hz / 5000 samples
VAE or decoder-capable latent model
open weights
paper-grade
easy to integrate into latent-hull online AT
```

Most relevant candidates:

| candidate | status | project decision |
|---|---|---|
| ECGEN VAE | closest 5000-sample 12-lead VAE candidate; MIT; code available | evaluate as external baseline/teacher, not immediate replacement |
| DiffuSETS / ECGTwin VAE | existing 1024-point `(4,128)` VAE with known weights | useful teacher; not native 500Hz |
| D-BETA / C-MELT | 5000-sample ECG-text representation model | feature teacher only, not decoder replacement |
| ECG-FM | ECG foundation encoder | feature teacher/source floor only |
| ST-MEM | ICLR ECG masked autoencoder | possible representation teacher, not VAE decoder |

Sources:

```text
ECGEN: https://pypi.org/project/ecgen/ , https://github.com/vlbthambawita/ECGEN
DiffuSETS: https://github.com/Raiiyf/DiffuSETS_Exp
ST-MEM: https://github.com/bakqui/ST-MEM
D-BETA: https://huggingface.co/Manhph2211/D-BETA
ECG-FM: https://github.com/bowang-lab/ECG-FM
```

## Decision

Do not spend a long run immediately on naive ECGTwin-original-weight fine-tuning
into VAE500. Direct initialization is structurally invalid:

```text
original ECGTwin VAE: 1024 x 12 -> latent (4,128)
current VAE500:       5000 x 12 -> latent (4,625)
```

If we still want to improve the VAE branch, use teacher distillation rather than
checkpoint loading:

```text
freeze ECGTwin original VAE
x500 -> VAE500 student
x500 -> lead reorder/resample -> x1024 -> ECGTwin teacher
loss =
  VAE500 reconstruction / KL / first-diff / lead-consistency
  + latent_distill_weight * SmoothL1(pool_or_project(z_s), z_t)
  + teacher_recon_weight * SmoothL1(resample(recon500 -> 1024), recon_t)
```

Recommended only if the next method revision changes the online AT objective or
selection rule. The current evidence says the bottleneck is not solved by simply
switching from VAE500 latent to original ECGTwin latent.
