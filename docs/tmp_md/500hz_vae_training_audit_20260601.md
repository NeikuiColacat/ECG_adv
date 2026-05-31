# 500Hz PTB-XL VAE Training Audit, 2026-06-01

## Accepted Checkpoint

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt
```

This is the first accepted PTB-XL-only 500Hz VAE checkpoint for the v7 branch.
It uses PTB-XL records500 only: folds 1-8 for train, fold 9 for validation, and
fold 10 for reconstruction audit.

## Training Acceleration

The accepted run uses the high-throughput path requested for this branch:

```text
cache_in_memory = true
batch_size      = 64
num_workers     = 8
prefetch_factor = 4
AMP             = bf16
TF32            = enabled
optimizer       = fused AdamW
```

`torch.compile(mode=reduce-overhead)` was smoke-tested but not used for the
accepted VAE run because PyTorch 2.1.1 failed on the model's dynamic
interpolation path. During steady-state VAE training, observed GPU utilization
was about 99-100%.

## Numeric Audit

| split | invalid | global Pearson | first-diff Pearson | leads >=0.90 | p2p median | lead residual ratio |
|---|---:|---:|---:|---:|---:|---:|
| fold9 val | 0.0 | 0.9976 | 0.9494 | 12/12 | 0.9925 | 1.0482 |
| fold10 audit | 0.0 | 0.9977 | 0.9505 | 12/12 | 0.9929 | 1.0883 |

The strict numeric gate passed.

## Visual Audit

32 random fold10 original-vs-reconstruction overlays were reviewed. No flatline,
QRS collapse, gross smoothing, or obvious lead-order corruption was observed.

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/audit/visual_audit_32/visual_audit_manifest.json
```

## Next Use

Use this checkpoint as the 500Hz VAE backend for the next EfficientNet1DV2
500Hz baseline and later VAE-only real-anchor online adversarial training.
