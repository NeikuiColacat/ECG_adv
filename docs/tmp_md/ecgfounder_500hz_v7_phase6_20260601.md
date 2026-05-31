# ECGFounder 500Hz V7 Phase 6 Result, 2026-06-01

## Protocol

- Label mapping: `v7_super5_sjr_rgq_review_20260528`.
- ECGFounder input: official-style `12 x 5000`, 500Hz, 10 seconds,
  per-record z-score, no additional filter in the main branch.
- Feature cache: reused old ECGFounder features and rebuilt PN2021 labels to
  v7 without rereading WFDB.
- Current K500 anchors:
  `/root/autodl-tmp/vae500_lhat_v7/anchors_k500`.
- All target-center metrics below exclude the same K500 record ids.

## Cache Relabel

Output:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/feature_cache_official_v7_from_v5
```

PN2021 cache rows: `66,416`; missing manifest keys: `0`.
Rows whose Super5 label changed under v7: `6,003`.

## Frozen Linear Probe

Run:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/linear_probe_seed42_official_v7
```

PTB-XL fold10: `0.9191 / 0.7952`.

| target view | target-center AUROC / AUPRC |
|---|---:|
| ningbo | 0.8846 / 0.5134 |
| chapman_shaoxing | 0.8871 / 0.4794 |
| cpsc_2018 | 0.8157 / 0.5690 |
| georgia | 0.8564 / 0.6640 |

## Direct K500 Head Fine-Tune

Run:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/kshot_head_ft_seed42_official_v7
```

Only the Super5 head is fine-tuned on the target-center K500 features; the
ECGFounder encoder stays frozen.

| center | target-center AUROC / AUPRC | 7-center avg AUROC / AUPRC |
|---|---:|---:|
| ningbo | 0.9301 / 0.6444 | 0.8627 / 0.6212 |
| chapman_shaoxing | 0.9311 / 0.6367 | 0.8615 / 0.6180 |
| cpsc_2018 | 0.8796 / 0.6618 | 0.8599 / 0.6155 |
| georgia | 0.8937 / 0.7527 | 0.8479 / 0.5977 |

## VAE500 Online AT Head Fine-Tune

Run:

```text
/root/autodl-tmp/paper_ecgfounder_500hz_v7/vae500_lhat_headft_ep10_seed42_official_v7
```

Configuration: initialize from the corresponding direct K500 head, use VAE500
latent-hull online adversarial samples with `M=20`, `hull_lambda=0.05`,
`hull_steps=3`, `K_anchor=128`, and select checkpoints from the K500-internal
target validation split.

| center | direct K500 baseline | VAE500 online AT selected | delta |
|---|---:|---:|---:|
| ningbo | 0.9301 / 0.6444 | 0.9301 / 0.6444 | +0.00pp / +0.00pp |
| chapman_shaoxing | 0.9311 / 0.6367 | 0.9311 / 0.6367 | +0.00pp / +0.00pp |
| cpsc_2018 | 0.8796 / 0.6618 | 0.8796 / 0.6618 | +0.00pp / +0.00pp |
| georgia | 0.8937 / 0.7527 | 0.8937 / 0.7527 | +0.00pp / +0.00pp |

In all four centers, the K500-internal validation criterion kept the epoch-0
direct K500 head. The online adversarial attack was active, with roughly
`0.4-0.6` attack success in logged epochs, but it did not improve the selected
ECGFounder head.

## Interpretation

ECGFounder is a strong comparison branch under the v7 mapping. Its largest gain
comes from direct K500 supervised head fine-tuning. The first VAE500 online-AT
configuration does not add measurable benefit on top of direct K500 for
ECGFounder, unlike the EfficientNet1DV2 branch where VAE500 online AT improved
all four target centers.

For the paper, ECGFounder should be reported as a strong frozen-foundation
baseline and direct K500 adaptation control. Do not attribute ECGFounder gains
to VAE500 unless a later recipe beats the direct K500 selected heads.
