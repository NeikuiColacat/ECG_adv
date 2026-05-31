# VAE500 Latent-Hull Online AT CPSC Smoke, 2026-06-01

## Purpose

Validate that the accepted PTB-XL-only 500Hz VAE can replace the old ECGTwin
1024-point VAE in the online adversarial training path.

## Inputs

```text
baseline model:
/root/autodl-tmp/triple_labels/super5_v7_effnet1dv2_500hz_full10_b128_20260601/best_model.pt

VAE500:
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt

CPSC anchors:
/root/autodl-tmp/vae500_lhat_v7/anchors_k500/cpsc_2018/cpsc_2018_real_k500_seed42_vae500.*
```

## Smoke Protocol

```text
center = cpsc_2018
K = 500
vae_backend = diffusets500_v1
latent shape = (4, 625)
input = 500Hz / 5000 samples / 10s
hull_M = 20
hull_lambda = 0.05
hull_steps = 3
K_anchor per epoch = 128
epochs = 3
quality gate = disabled, but diagnostics logged
```

## Result

| model | PTB-XL fold10 | CPSC ref-excluded | PN2021 7-center |
|---|---:|---:|---:|
| 500Hz baseline | 0.9131 / 0.7867 | 0.7958 / 0.5427 | 0.7807 / 0.4754 |
| VAE500 LHAT ep3 | 0.9120 / 0.7843 | 0.8444 / 0.5896 | 0.7906 / 0.4791 |
| delta | -0.0011 / -0.0024 | +0.0486 / +0.0469 | +0.0099 / +0.0037 |

This is not yet the final four-center run, but it verifies that the VAE500
backend works at K500 scale and gives a positive CPSC signal.
