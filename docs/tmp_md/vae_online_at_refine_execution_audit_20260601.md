# VAE Online AT Refine Execution Audit, 2026-06-01

## Fixed Protocol

- Main VAE: repo-owned PTB-XL-only 500Hz VAE.
- VAE checkpoint:
  `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt`.
- Backbone first target: EfficientNet1DV2 500Hz.
- Input: `minimal_resample`, `per_sample_global`, `500Hz`, `5000 samples = 10s`.
- Mapping: `v7_super5_sjr_rgq_review_20260528`.
- Classes: `CD, HYP, MI, NORM, STTC`.
- Centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- Target anchors: K500 per center from
  `/root/autodl-tmp/vae500_lhat_v7/anchors_k500/<center>/`.
- Final held-out evaluation: exclude each center's K500 ref ids.

## Hardware Policy

- Current visible GPU: one idle RTX 4090D 24GB.
- Current CPU quota: 15 cores.
- Current RAM is sufficient for cached PTB-XL / center-level PN2021 inference.
- Default single-run acceleration: `batch_size=128`, `num_workers=8`,
  pinned memory, persistent workers, prefetching, TF32, and CuDNN benchmark.
- Large outputs stay under `/root/autodl-tmp`.

## Current Evidence Before New Runs

The existing 500Hz VAE AT result is positive against the PTB-XL source-only
baseline, but this is not yet the strict target claim because it is not compared
against matched direct target fine-tuning.

| center | source baseline | VAE500 online AT | delta |
|---|---:|---:|---:|
| ningbo | 0.8598 / 0.4650 | 0.8774 / 0.4986 | +1.75pp / +3.35pp |
| chapman_shaoxing | 0.8602 / 0.4150 | 0.8728 / 0.4459 | +1.26pp / +3.09pp |
| cpsc_2018 | 0.7958 / 0.5427 | 0.8482 / 0.5932 | +5.24pp / +5.05pp |
| georgia | 0.8185 / 0.5963 | 0.8327 / 0.6135 | +1.42pp / +1.72pp |
| mean | 0.8336 / 0.5048 | 0.8578 / 0.5378 | +2.42pp / +3.30pp |

## Immediate Missing Evidence

Run strict matched direct target fine-tune controls for EfficientNet1DV2 500Hz:

```text
direct control = same script, same PTB-XL source stream, same target K500 split,
same K500-internal validation, same checkpoint selection, same final
ref-excluded evaluation, but with --disable_adv_stream.
```

This isolates the added value of VAE latent online adversarial samples beyond
ordinary target-center supervised adaptation.

## First Execution Batch

Output root:

```text
/root/autodl-tmp/vae500_direct_ft_v7_20260601/
```

Runs:

```text
<center>_k500_matched_noadv_ep30_seed42_20260601
```

Success check after completion:

```text
VAE500 online AT - matched direct FT >= +2pp AUROC
or
VAE500 online AT - matched direct FT >= +2pp AUPRC
and the other metric must not decrease.
```

If the strict direct control erases the VAE advantage, the next step is not
seed scaling. The next step is AugMix+VAE, boundary-targeted anchors, or
lower-K / larger-ratio analysis depending on which centers/classes fail.

## Matched Direct FT Result

Completed output root:

```text
/root/autodl-tmp/vae500_direct_ft_v7_20260601_r2/
```

The first failed partial run under
`/root/autodl-tmp/vae500_direct_ft_v7_20260601/` is retained only as bug
evidence. The valid run is `_r2`.

All values below are target-center K500 ref-excluded macro `AUROC / AUPRC`.

| center | matched direct FT | VAE500 online AT | VAE - direct |
|---|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8774 / 0.4986 | -0.24pp / -0.37pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8728 / 0.4459 | +0.03pp / +0.11pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8482 / 0.5932 | -0.51pp / -0.80pp |
| georgia | 0.8347 / 0.6180 | 0.8327 / 0.6135 | -0.20pp / -0.45pp |
| mean | 0.8601 / 0.5416 | 0.8578 / 0.5378 | -0.23pp / -0.38pp |

Drop-all-zero sensitivity shows the same pattern:

| center | matched direct FT | VAE500 online AT | VAE - direct |
|---|---:|---:|---:|
| ningbo | 0.9031 / 0.6860 | 0.9013 / 0.6840 | -0.18pp / -0.20pp |
| chapman_shaoxing | 0.8951 / 0.6368 | 0.8965 / 0.6405 | +0.14pp / +0.37pp |
| cpsc_2018 | 0.9062 / 0.7790 | 0.9009 / 0.7674 | -0.53pp / -1.16pp |
| georgia | 0.8449 / 0.7115 | 0.8426 / 0.7075 | -0.24pp / -0.40pp |

Interpretation:

```text
Plain VAE500 online AT improves over the source-only PTB-XL model, but it does
not beat matched direct target fine-tuning under this strict K500 protocol.
The next experiment should test whether AugMix+VAE or boundary-targeted VAE
regularization can add value beyond direct FT.
```
