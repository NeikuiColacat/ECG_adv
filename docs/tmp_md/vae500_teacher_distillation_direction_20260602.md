# VAE500 Teacher-Distillation Direction

Date: 2026-06-02

## Why This Direction

The latest 500Hz experiments show a consistent pattern:

```text
VAE500 reconstructs well.
Decoded adversarial samples are valid.
Attack ASR is active.
But matched direct40 K500 validation rejects the VAE updates.
```

This suggests the current VAE500 latent space is probably too focused on
waveform reconstruction and not semantic enough for online adversarial training.

## Original ECGTwin VAE Compatibility

Original ECGTwin VAE:

```text
input/output: 1024 x 12
latent:       (4, 128)
checkpoint:   {'encoder', 'decoder'}
lead order:   ECGTwin / MIMIC order
```

Current VAE500:

```text
input/output: 5000 x 12
latent:       (4, 625)
checkpoint:   model_state_dict
lead order:   PTB-XL canonical order
```

Therefore ECGTwin's original VAE checkpoint cannot be directly loaded into the
current VAE500. Partial weight transfer would require a new compatible variant
and careful state-dict mapping; it is not the safest next step.

## Recommended Main Route

Train a teacher-distilled VAE500:

```text
freeze original ECGTwin VAE
PTB-XL records500 x500 -> student VAE500
same x500 -> lead reorder + resample to 1024 -> ECGTwin teacher VAE
```

Loss design:

```text
VAE500 reconstruction / KL / first-diff / lead-consistency
+ latent_distill_weight * SmoothL1(project(z_s_4x625), z_t_4x128)
+ teacher_recon_weight * SmoothL1(resample(recon500 -> 1024), recon_t_1024)
```

Suggested first sweep:

```text
latent_distill_weight: 0.05, 0.10, 0.20
teacher_recon_weight: 0.05
train data: PTB-XL records500 only
eval: reconstruction audit + K500 direct40-init VAE pilot on cpsc_2018/ningbo
```

## Cheap Diagnostic Before Full Training

Before training a new VAE, run a cheaper check:

```text
target ECG 500Hz
-> ECGTwin-compatible 1024 view
-> original ECGTwin VAE encode / latent hull / decode
-> resample decoded ECG to 5000
-> feed 500Hz EfficientNet
```

If this works better than VAE500 latent hull, then the problem is likely VAE500
latent geometry. If it also fails, the issue is more likely the direct40 control
or online-AT loss design.

## External Candidate Models

| candidate | role | priority |
|---|---|---|
| ECGEN VAE | closest public 12-lead, 500Hz, 5000-sample VAE candidate | A |
| D-BETA / C-MELT | 5000-sample ECG-text feature teacher | A- |
| ECG-FM | ECG foundation embedding teacher | B+ |
| ST-MEM | high-quality masked autoencoder teacher, but not native 500Hz | B |
| MCMA | reduced-lead reconstruction reference | B- |

No currently identified public model is a mature drop-in replacement for
ECGTwin's original VAE in our 500Hz latent-hull AT pipeline.
