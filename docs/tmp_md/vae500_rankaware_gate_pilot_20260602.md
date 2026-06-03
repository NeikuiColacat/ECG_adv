# VAE500 Rank-Aware Objective Gate Pilot

Date: 2026-06-02

## Purpose

Test whether adding a rare-abnormal multilabel ranking objective can make
EfficientNet1DV2 500Hz VAE online AT beat the matched direct target fine-tune
control by the requested margin.

This branch was run because previous evidence showed that generic VAE500 LHAT,
AugMix, positive-hide/negative-add, and ECGTwin-init/teacher VAE replacement did
not beat matched direct fine-tuning.

## Code Change

Added default-off rank-aware training to:

```text
scripts/pgd_cross_center/synth_online_at_super5.py
```

New CLI flags:

```text
--rank_loss_weight
--rank_loss_margin
--rank_loss_positive_classes
```

When enabled, the training loss is:

```text
masked_BCE + rank_loss_weight * pairwise_multilabel_rank_loss + EWA_anchor
```

The pairwise rank term enforces selected positive-class logits to rank above
valid negative-class logits. Unknown labels `-1` remain masked out. Default
`rank_loss_weight=0.0` preserves historical behavior.

Pilot setting:

```text
rank_loss_weight=0.05
rank_loss_margin=1.0
rank_loss_positive_classes=CD,HYP,MI,STTC
```

## Protocol

```text
backbone: EfficientNet1DV2 500Hz
mapping: v7_super5_sjr_rgq_review_20260528
target center: cpsc_2018
K target anchors: 500, seed42
eval: cpsc_2018 held-out, K500 ref ids excluded
VAE: spectral-r001 DiffuSETSVAE500
selection: K500-internal validation AUPRC
```

Runs:

```text
/root/autodl-tmp/vae500_rankaware_objective_gate_k500_v7_20260602/
  cpsc_2018_direct40_rank005_noadv_ep10_seed42
  cpsc_2018_vae_rank005_hardlg_ep10_seed42
```

## Results

Held-out cpsc_2018, K500 ref ids excluded:

| method | cpsc_2018 AUROC / AUPRC | delta vs direct40 | PTB-XL fold10 AUROC / AUPRC |
|---|---:|---:|---:|
| direct40 | 0.8633 / 0.6155 | reference | 0.9086 / 0.7749 |
| spectral-r001 hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.9079 / 0.7733 |
| direct40 + rank loss | 0.8652 / 0.6181 | +0.19pp / +0.26pp | 0.9086 / 0.7744 |
| VAE hard loss-gain + rank loss | 0.8687 / 0.6267 | +0.54pp / +1.12pp | 0.9083 / 0.7734 |

K500-internal validation:

| method | best internal AUROC / AUPRC | selected epoch |
|---|---:|---:|
| direct40 + rank loss | 0.9122 / 0.8324 | 5 |
| VAE hard loss-gain + rank loss | 0.9145 / 0.8307 | 10 |

VAE-rank diagnostics:

```text
ASR: about 0.30-0.47 across epochs
decoded_invalid_rate: 0.0
buffer size at epoch 10: 947
attack state: healthy at epoch 10
```

## Decision

Rank-aware objective is technically valid and does not break training, but it
does not improve over the current spectral-r001 hard loss-gain VAE branch.

The pre-registered CPSC gate was:

```text
CPSC AUPRC must exceed current hard loss-gain, otherwise do not expand to
Ningbo/four centers.
```

It failed that gate:

```text
hard loss-gain:     0.8688 / 0.6270
VAE-rank hard-gain: 0.8687 / 0.6267
```

Therefore, do not run the Ningbo rank-aware expansion under this exact recipe.
Keep the code because the objective is useful for future rare-positive
experiments, but freeze this specific `rank_loss_weight=0.05` branch as a
limited negative result.
