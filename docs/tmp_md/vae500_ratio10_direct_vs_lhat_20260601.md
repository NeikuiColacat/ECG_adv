# VAE500 Ratio10 Direct FT vs Online AT

Date: 2026-06-01

## Protocol

This run tests whether VAE500 latent-hull online adversarial training adds
value over a matched direct target-center fine-tune when the target anchor set
is selected by ratio instead of fixed K.

Target anchors are sampled as 10% of each center's effective nonzero Super5
records. Ref ids are excluded from final target-center evaluation.

Fixed input protocol:

```text
mapping: v7_super5_sjr_rgq_review_20260528
backbone: EfficientNet1DV2 500Hz
sampling rate: 500Hz
input length: 5000 samples = 10 seconds
preprocess_mode: minimal_resample
norm_mode: per_sample_global
source init: /root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601/best_model.pt
VAE: /root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt
```

Online AT recipe:

```text
attack_mode: latent_hull
hull_M: 20
hull_lambda: 0.05
hull_steps: 3
hull_lr: 0.25
hull_include_anchor: true
hull_neighbor_mode: local_random
hull_neighbor_distance_space: standardized
target_real_weight: 20.0
adv_weight: 0.3
quality_gate: disabled
selection: K-internal target validation AUPRC
```

## Result

| center | K | direct FT | VAE500 online AT | delta |
|---|---:|---:|---:|---:|
| ningbo | 1923 | 0.8912 / 0.5100 | 0.8911 / 0.5103 | -0.01pp / +0.03pp |
| chapman_shaoxing | 582 | 0.8740 / 0.4509 | 0.8745 / 0.4509 | +0.05pp / +0.00pp |
| cpsc_2018 | 475 | 0.8610 / 0.6068 | 0.8607 / 0.6084 | -0.03pp / +0.16pp |
| georgia | 871 | 0.8362 / 0.6190 | 0.8383 / 0.6235 | +0.21pp / +0.45pp |
| mean | - | 0.8656 / 0.5467 | 0.8662 / 0.5483 | +0.06pp / +0.16pp |

PTB-XL source performance after VAE AT:

| center-adapted model | PTB-XL AUROC / AUPRC |
|---|---:|
| ningbo ratio10 VAE | 0.9022 / 0.7597 |
| chapman_shaoxing ratio10 VAE | 0.9098 / 0.7787 |
| cpsc_2018 ratio10 VAE | 0.9095 / 0.7804 |
| georgia ratio10 VAE | 0.9077 / 0.7681 |

## Diagnostics

All four centers showed clear K-internal target validation improvement, but
that improvement mostly did not transfer to final held-out ref-excluded PN2021
evaluation.

Internal validation best snapshots:

| center | internal baseline | best internal VAE |
|---|---:|---:|
| ningbo | 0.8750 / 0.7920 | 0.9173 / 0.8588 |
| chapman_shaoxing | 0.8736 / 0.7762 | 0.9085 / 0.8393 |
| cpsc_2018 | 0.8412 / 0.6886 | 0.8836 / 0.7577 |
| georgia | 0.8075 / 0.6912 | 0.8369 / 0.7269 |

Attack strength:

```text
ningbo: ASR mostly 0.41-0.51
chapman_shaoxing: ASR mostly 0.38-0.51
cpsc_2018: ASR often 0.17-0.30, relatively weak
georgia: ASR mostly 0.38-0.52
```

## Interpretation

The ratio10 experiment does not meet the goal criterion. VAE500 online AT is
safe and source performance is preserved, but the matched direct FT control is
already strong after adding more target-center data. The current plain
latent-hull branch adds almost no external-center gain.

For the next search step, scaling K/ratio alone is unlikely to unlock +2pp over
direct FT. More promising changes are:

1. CPSC-specific attack strengthening, because ASR is below the desired range.
2. A method that selects adversarial anchors by expected held-out
   generalization risk rather than K-internal target validation improvement.
3. Backbone-transfer experiments, because the EfficientNet1DV2 direct FT branch
   appears close to saturation under ratio10.
