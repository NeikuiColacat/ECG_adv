# 500Hz VAE-Only Online AT Four-Center Result, 2026-06-01

## Protocol

- Label mapping: `v7_super5_sjr_rgq_review_20260528`.
- Backbone: project `EfficientNet1DV2`, trained on PTB-XL records500.
- Input: `minimal_resample`, `per_sample_global`, `500Hz`, `5000 samples = 10s`.
- VAE: repo-owned PTB-XL-only 500Hz VAE,
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt`.
- Target adaptation: `K=500` real target-center anchors, encoded to VAE latent
  shape `(4, 625)`.
- Final target evaluation excludes the same K500 record ids.
- Main online AT recipe:
  `M=20`, `hull_lambda=0.05`, `hull_steps=3`, `hull_lr=0.25`,
  `K_anchor=128`, `pgd_batch=16`, `target_real_weight=20`,
  `adv_weight=0.3`, `adv_weight_warmup_epochs=3`.
- Quality gate: disabled as a learned/victim-score filter; hard invalid signal
  diagnostics are still logged.

## Runtime Acceleration

The online AT script now exposes and records:

- `--allow_tf32`
- `--matmul_precision high`
- `--cudnn_benchmark`
- `--pin_memory`
- `--persistent_workers`
- `--prefetch_factor 4`

For the four-center runs, `batch_size=128`, `num_workers=8` was stable. GPU
utilization was typically about `88-100%`, and memory use was about `22.5GB`,
so this is close to the safe single-4090D ceiling.

AMP was not enabled for this online AT path because latent-hull PGD optimizes
through the VAE decoder and classifier. Keeping that gradient path in fp32 is
safer for this first 500Hz validation run.

## Ref-Excluded Target-Center Results

All values are macro `AUROC / AUPRC`, all-zero-kept PN2021 evaluation.

| center | 500Hz baseline, K500 excluded | VAE500 online AT, K500 excluded | delta |
|---|---:|---:|---:|
| ningbo | 0.8598 / 0.4650 | 0.8774 / 0.4986 | +1.75pp / +3.35pp |
| chapman_shaoxing | 0.8602 / 0.4150 | 0.8728 / 0.4459 | +1.26pp / +3.09pp |
| cpsc_2018 | 0.7958 / 0.5427 | 0.8482 / 0.5932 | +5.24pp / +5.05pp |
| georgia | 0.8185 / 0.5963 | 0.8327 / 0.6135 | +1.42pp / +1.72pp |
| mean | 0.8336 / 0.5048 | 0.8578 / 0.5378 | +2.42pp / +3.30pp |

## Source PTB-XL Floor

Baseline PTB-XL fold10 is `0.9131 / 0.7867`.

| adapted center | PTB-XL fold10 after AT | source delta |
|---|---:|---:|
| ningbo | 0.9109 / 0.7807 | -0.22pp / -0.60pp |
| chapman_shaoxing | 0.9109 / 0.7804 | -0.22pp / -0.62pp |
| cpsc_2018 | 0.9120 / 0.7848 | -0.11pp / -0.19pp |
| georgia | 0.9105 / 0.7792 | -0.26pp / -0.75pp |

## Run Artifacts

Baseline:

```text
/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601
```

Adapted models:

```text
/root/autodl-tmp/vae500_lhat_v7/ningbo_k500_ep30_20260601
/root/autodl-tmp/vae500_lhat_v7/chapman_shaoxing_k500_ep30_20260601
/root/autodl-tmp/vae500_lhat_v7/cpsc_2018_k500_ep30_20260601
/root/autodl-tmp/vae500_lhat_v7/georgia_k500_ep30_20260601
```

Anchor exports:

```text
/root/autodl-tmp/vae500_lhat_v7/anchors_k500/<center>/<center>_real_k500_seed42_vae500.*
```

## Interpretation

The PTB-XL-only 500Hz VAE branch successfully preserves the core VAE-only
online AT story under the stricter v7 mapping and 500Hz input protocol. The
four target centers all improve after K500 ref exclusion, with modest source
PTB-XL degradation.

The largest gain is still CPSC. Ningbo and Chapman show useful AUPRC gains,
while Georgia is positive but smaller. This supports keeping the 500Hz VAE-only
online AT branch as the current EfficientNet1DV2 paper mainline, with follow-up
work needed for multi-seed stability and direct K500 fine-tune controls.
