# 500Hz EfficientNet1DV2 Super5 Baseline, 2026-06-01

## Run

```text
/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601
```

Protocol:

```text
PTB-XL records500
minimal_resample
per_sample_global
500Hz / 5000 samples / 10s
Super5 class order = CD, HYP, MI, NORM, STTC
checkpoint metric = fold9 macro AUPRC
```

Training acceleration:

```text
batch_size      = 128
AMP             = bf16
TF32            = enabled
optimizer       = fused AdamW
num_workers     = 8
prefetch_factor = 4
```

The accepted run used about 12.5GB GPU memory and sampled at 100% GPU SM
utilization during steady-state training. Early stopping triggered at epoch 47.

## PTB-XL Fold10

```text
macro AUROC = 0.9131
macro AUPRC = 0.7867
```

## PN2021 V7

| center | n | all-zero-kept AUROC/AUPRC | drop-all-zero AUROC/AUPRC |
|---|---:|---:|---:|
| ningbo | 34905 | 0.8602 / 0.4675 | 0.8779 / 0.6376 |
| chapman_shaoxing | 10247 | 0.8605 / 0.4261 | 0.8784 / 0.5977 |
| cpsc_2018 | 6877 | 0.7995 / 0.5498 | 0.8471 / 0.6684 |
| georgia | 10344 | 0.8187 / 0.5989 | 0.8264 / 0.6835 |
| four-center mean | - | 0.8347 / 0.5106 | 0.8574 / 0.6468 |
| seven-center mean | - | 0.7812 / 0.4764 | 0.8003 / 0.5804 |

Primary artifact:

```text
/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601/eval_result_v7_500hz.json
```
