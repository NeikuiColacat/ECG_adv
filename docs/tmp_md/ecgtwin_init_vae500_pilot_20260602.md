# ECGTwin-Init VAE500 Pilot

Date: 2026-06-02

## Purpose

Test whether the original ECGTwin author VAE weights can initialize a stronger
500Hz / 5000-sample VAE for the VAE-only online AT mainline.

## Implementation

Added backend:

```text
ecg_adv_gen/vae/ecgtwin_vae500.py
variant: ecgtwin_init_vae500
```

Design:

- Reuses original ECGTwin `VAE_Encoder` and `VAE_Decoder`.
- Keeps 5000-sample input/output; latent becomes `(B, 4, 625)`.
- Strictly loads original `model/ECGTwin/checkpoints/vae_model.pth`
  `encoder` and `decoder` state dicts.
- Fixes lead-order mismatch:
  - training/eval API uses PTB-XL order;
  - wrapper converts PTB-XL -> ECGTwin before encoder;
  - wrapper converts ECGTwin -> PTB-XL after decoder.

Smoke checks:

```text
py_compile: passed
strict checkpoint load: all keys matched
5000-point forward: z=(1,4,625), recon=(1,5000,12), finite=True
batch size memsmoke: b8, b32, b64 all passed on 4090D
```

## Pilot Runs

### Lead-order bug smoke

Before lead-order correction, reconstruction was poor:

| run | val loss | val global Pearson | audit MSE | audit lead residual ratio |
|---|---:|---:|---:|---:|
| no lead fix, 1 epoch smoke | 0.1300 | 0.9173 | 0.1262 | 6.27 |

This confirmed that ECGTwin original weights must not be trained/evaluated in
PTB-XL lead order directly.

### Lead-fixed smoke

| run | val loss | val global Pearson | audit MSE | audit lead residual ratio |
|---|---:|---:|---:|---:|
| lead-fixed, 1 epoch smoke | 0.0367 | 0.9652 | 0.0681 | 2.12 |

Lead-order correction immediately improved reconstruction.

### Full PTB-XL pilot

Training data:

```text
cache: /root/autodl-tmp/vae500/ptbxl_records500_v7/cache_v1
train: PTB-XL folds 1-8
val:   fold 9
audit: fold 10
```

Runs:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc025_20260602
/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc025_cont20_20260602
/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc100_cont10_20260602
```

Best current ECGTwin-init checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc100_cont10_20260602/checkpoints/best.pt
```

Audit result:

| metric | ECGTwin-init VAE500 | current DiffuSETSVAE500 best |
|---|---:|---:|
| invalid decode rate | 0.0000 | 0.0000 |
| audit global Pearson | 0.9964 | 0.9977 |
| audit first-diff Pearson | 0.9478 | 0.9505 |
| audit MSE | 0.00977 | 0.00710 |
| audit lead residual ratio | 1.1255 | 1.0880 |

## Decision

ECGTwin-init VAE500 is technically feasible and now passes the basic hard
sanity gate, but it does not yet beat the current DiffuSETSVAE500 reconstruction
quality. In particular, audit MSE remains about 37% worse than the current
best VAE500 checkpoint.

Do not use this checkpoint for downstream VAE-online AT as a replacement yet.

Recommended next VAE actions:

1. Stop short ad-hoc continuation unless explicitly testing a better objective.
2. If continuing this branch, use a planned objective change rather than more
   epochs only:
   - keep `lead_consistency_weight=1.0`;
   - add teacher-distill loss from original 1024-point ECGTwin recon/latent;
   - optionally add multi-resolution spectral/STFT loss.
3. Treat current DiffuSETSVAE500 as the active downstream VAE until a new VAE
   matches or beats its reconstruction gate.
