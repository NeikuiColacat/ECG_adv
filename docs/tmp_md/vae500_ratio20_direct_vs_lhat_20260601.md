# VAE500 Ratio20 Direct vs LHAT

Date: 2026-06-01

## Scope

This report tracks the 20% target-ratio branch for the VAE online adversarial
training refine goal.

Protocol:

- backbone: EfficientNet1DV2 500Hz
- label mapping: `v7_super5_sjr_rgq_review_20260528`
- input: `minimal_resample`, `per_sample_global`, 500Hz, 5000 samples
- centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`
- K rule: 20% of each center's nonzero Super5 records
- final evaluation: target-center ref ids excluded

## Target Anchor Sizes

| center | K |
|---|---:|
| ningbo | 3846 |
| chapman_shaoxing | 1164 |
| cpsc_2018 | 949 |
| georgia | 1742 |

Anchor root:

```text
/root/autodl-tmp/vae500_lhat_v7/anchors_ratio20/
```

## Matched Direct FT Control

Output root:

```text
/root/autodl-tmp/vae500_direct_ft_ratio20_v7_20260601/
```

Configuration:

- initialized from PTB-XL 500Hz EfficientNet1DV2 source checkpoint
- target real stream enabled
- adversarial stream disabled
- `target_real_weight=20.0`
- K-internal validation split: 20%, seed `20260531`
- checkpoint selection: K-internal `target_macro_auprc`

Results:

| center | K | ratio20 direct | PTB-XL fold10 | excluded refs | delta vs 10% direct | delta vs K1000 direct |
|---|---:|---:|---:|---:|---:|---:|
| ningbo | 3846 | 0.8952 / 0.5128 | 0.8955 / 0.7451 | 3846 | +0.40pp / +0.28pp | +1.03pp / +0.75pp |
| chapman_shaoxing | 1164 | 0.8735 / 0.4436 | 0.9059 / 0.7661 | 1164 | -0.05pp / -0.73pp | -0.23pp / -0.40pp |
| cpsc_2018 | 949 | 0.8796 / 0.6430 | 0.9072 / 0.7732 | 949 | +1.86pp / +3.62pp | +0.26pp / +1.02pp |
| georgia | 1742 | 0.8451 / 0.6302 | 0.9014 / 0.7503 | 1742 | +0.89pp / +1.12pp | +0.55pp / +0.87pp |
| mean | - | 0.8733 / 0.5574 | - | - | +0.77pp / +1.07pp | +0.40pp / +0.56pp |

## Matched VAE500 LHAT

Output root:

```text
/root/autodl-tmp/vae500_lhat_ratio20_v7_20260601/
```

Configuration:

- initialized from PTB-XL 500Hz EfficientNet1DV2 source checkpoint
- target real stream enabled
- latent-hull VAE online AT enabled
- `target_real_weight=20.0`
- `adv_weight=0.3`
- `hull_M=20`
- `hull_lambda=0.05`
- quality gate disabled
- K-internal validation split: 20%, seed `20260531`
- checkpoint selection: K-internal `target_macro_auprc`

Results:

| center | K | ratio20 direct | ratio20 VAE LHAT | delta VAE - direct | PTB-XL fold10 VAE |
|---|---:|---:|---:|---:|---:|
| ningbo | 3846 | 0.8952 / 0.5128 | 0.8949 / 0.5124 | -0.03pp / -0.04pp | 0.8964 / 0.7454 |
| chapman_shaoxing | 1164 | 0.8735 / 0.4436 | 0.8721 / 0.4414 | -0.14pp / -0.22pp | 0.9057 / 0.7664 |
| cpsc_2018 | 949 | 0.8796 / 0.6430 | 0.8785 / 0.6399 | -0.11pp / -0.31pp | 0.9071 / 0.7733 |
| georgia | 1742 | 0.8451 / 0.6302 | 0.8446 / 0.6293 | -0.05pp / -0.09pp | 0.9016 / 0.7498 |
| mean | - | 0.8733 / 0.5574 | 0.8726 / 0.5557 | -0.08pp / -0.17pp | - |

## Interpretation

Increasing target-center data from K1000 or 10% to 20% improves the matched
direct FT mean, but the gain is uneven. It is mainly driven by `cpsc_2018` and
`georgia`; `chapman_shaoxing` drops.

The matched VAE500 latent-hull branch does not beat direct FT at 20% target
ratio. All four centers are slightly below direct, and the four-center mean is
lower by `-0.08pp AUROC / -0.17pp AUPRC`.

This supports a diminishing-return conclusion for the current EfficientNet1DV2
plain VAE recipe: extra target real samples help supervised adaptation, while
the current VAE branch adds little beyond a strong matched direct control.
Do not transfer this exact recipe to ECGFounder or other backbones as a main
experiment unless it is explicitly treated as a negative-control ablation.
