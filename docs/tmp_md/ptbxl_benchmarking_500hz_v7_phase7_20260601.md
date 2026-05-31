# PTB-XL Benchmarking 500Hz V7 Phase 7 Result, 2026-06-01

## Protocol

- Label mapping: `v7_super5_sjr_rgq_review_20260528`.
- Data: PTB-XL records500 cache, `500Hz`, `5000 samples = 10s`.
- Training labels: PTB-XL Super5, class order `CD, HYP, MI, NORM, STTC`.
- External evaluation: PN2021 v7, target-center K500 reference records excluded
  for the four main target centers.
- Benchmark protocol difference: these models use dataset-level train
  `StandardScaler` mean/std, matching `model/ecg_ptbxl_benchmarking`; our
  EfficientNet1DV2 branch uses per-sample global z-score.

## Runtime Acceleration

Training used RAM-loaded PTB-XL arrays, `batch_size=128`, `eval_batch_size=256`,
`num_workers=8`, pinned memory, persistent workers, `prefetch_factor=4`,
bf16 AMP, TF32, CuDNN benchmark, and fused AdamW.

PN2021 evaluation used mmap caches on disk, then copied each center into RAM for
inference. Evaluation used `batch_size=512`, `num_workers=12`, pinned memory,
persistent workers, bf16 AMP, TF32, and CuDNN benchmark. The redundant compressed
`.npz` PN2021 benchmark cache was removed; only mmap cache is kept.

## PTB-XL Fold10 And PN2021 Direct Results

All PN2021 values below are target-center K500-excluded macro `AUROC / AUPRC`.

| model | PTB-XL fold10 | PN2021 4-center mean | ningbo | chapman_shaoxing | cpsc_2018 | georgia |
|---|---:|---:|---:|---:|---:|---:|
| `fastai_inception1d` | 0.9247 / 0.8138 | 0.8452 / 0.4971 | 0.8697 / 0.4873 | 0.8788 / 0.4402 | 0.8042 / 0.5551 | 0.8280 / 0.5057 |
| `fastai_resnet1d_wang` | 0.9175 / 0.7984 | 0.8246 / 0.4833 | 0.8593 / 0.4802 | 0.8650 / 0.4422 | 0.7736 / 0.5189 | 0.8004 / 0.4921 |
| `fastai_xresnet1d50` | 0.9206 / 0.8079 | 0.8511 / 0.5120 | 0.8791 / 0.5101 | 0.8854 / 0.4643 | 0.8092 / 0.5580 | 0.8306 / 0.5157 |
| `fastai_fcn_wang` | 0.9084 / 0.7869 | 0.8245 / 0.4838 | 0.8560 / 0.4809 | 0.8663 / 0.4549 | 0.7657 / 0.5131 | 0.8097 / 0.4863 |
| `fastai_schirrmeister` | 0.9070 / 0.7651 | 0.8264 / 0.4671 | 0.8571 / 0.4537 | 0.8582 / 0.4040 | 0.7917 / 0.5390 | 0.7987 / 0.4716 |

The strongest benchmark backbone is `fastai_xresnet1d50` on both PTB-XL fold10
and PN2021 four-center direct evaluation among this five-model set.

## Comparison To Current Main Branches

| branch | 4-center target K500-excluded AUROC / AUPRC | note |
|---|---:|---|
| EfficientNet1DV2 500Hz direct baseline | 0.8336 / 0.5048 | per-sample z-score |
| EfficientNet1DV2 + VAE500 online AT | 0.8578 / 0.5378 | current VAE-only mainline |
| best benchmark direct, `xresnet1d50` | 0.8511 / 0.5120 | no target adaptation |
| ECGFounder frozen linear probe | 0.8610 / 0.5565 | frozen encoder, no target K500 fine-tune |
| ECGFounder direct K500 head fine-tune | 0.9086 / 0.6739 | strong foundation-model adaptation control |

## Interpretation

The benchmark backbones validate that 500Hz PTB-XL training is healthy: several
independent 1D architectures reach PTB-XL fold10 performance comparable to or
above EfficientNet1DV2. On PN2021, `xresnet1d50` is the strongest direct
benchmark baseline but remains below the EfficientNet1DV2 + VAE500 online AT
mainline on four-center AUPRC.

This phase does not yet prove VAE500 online AT transfers to every benchmark
backbone. The current online AT trainer is written around the project
EfficientNet1DV2 victim path. Running VAE500 online AT on these five benchmark
models requires a small generic victim/model-builder adapter before it becomes a
fair matched comparison.

## Artifacts

Training and evaluation root:

```text
/root/autodl-tmp/ptbxl_benchmarking_500hz_super5_v7/five_backbones_full30_20260601
```

PN2021 mmap cache:

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache_v7_500hz_benchmark_none_mmap
```
