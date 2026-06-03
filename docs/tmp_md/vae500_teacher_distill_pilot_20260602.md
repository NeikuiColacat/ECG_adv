# 500Hz VAE With ECGTwin Teacher Pilot

Date: 2026-06-02

## Question

The 500Hz VAE may be one bottleneck for VAE-only online adversarial training.
This pilot tests whether the original ECGTwin VAE weights can help build a
stronger PTB-XL-only 500Hz VAE.

## Implemented Branches

### Route A: ECGTwin-init VAE500

Code:

```text
ecg_adv_gen/vae/ecgtwin_vae500.py
ecg_adv_gen/vae/diffusets_vae500.py
scripts/vae500/train_ptbxl_vae500.py
```

Implementation:

- Reuse original ECGTwin `VAE_Encoder` and `VAE_Decoder`.
- Run them at 5000 samples, so the latent shape becomes `(B, 4, 625)`.
- Strict-load original `model/ECGTwin/checkpoints/vae_model.pth`.
- Convert PTB-XL lead order to ECGTwin order before encoding, and convert back
  after decoding.

Best checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc100_cont10_20260602/checkpoints/best.pt
```

### Route B: DiffuSETSVAE500 + frozen ECGTwin teacher

Code:

```text
scripts/vae500/train_ptbxl_vae500.py
```

Implementation:

- Student: current `diffusets500_v1_dynamic` VAE500.
- Teacher: frozen original ECGTwin VAE at 1024 samples.
- Student loss adds:

```text
teacher_recon_weight * SmoothL1(resample(recon500 -> 1024), recon_teacher)
+ teacher_latent_weight * SmoothL1(pool(mu_student -> 128), mu_teacher)
```

Best checkpoint:

```text
/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_teacher_r002_z001_b64_20260602/checkpoints/best.pt
```

## Reconstruction Gate

All numbers below are PTB-XL fold10 audit metrics.

| VAE | invalid | global Pearson | first-diff Pearson | MSE | lead residual |
|---|---:|---:|---:|---:|---:|
| current DiffuSETSVAE500 | 0.0000 | 0.997656 | 0.950460 | 0.007098 | 1.0883 |
| ECGTwin-init VAE500 | 0.0000 | 0.996431 | 0.947812 | 0.009768 | 1.1255 |
| ECGTwin-teacher DiffuSETSVAE500 | 0.0000 | 0.997734 | 0.950630 | 0.006939 | 1.1592 |

Decision:

- ECGTwin-init VAE500 is feasible but not better than current DiffuSETSVAE500.
- Teacher-distilled DiffuSETSVAE500 slightly improves MSE and Pearson, but lead
  residual is worse. It passes the hard reconstruction gate, but the margin is
  small.

## Downstream Pilot

Backbone:

```text
fastai_xresnet1d50
```

Matched direct K500 and old VAE-LHAT numbers come from:

```text
/root/autodl-tmp/vae500_benchmark_xresnet50_matched_k500_v7_20260602
```

Teacher-VAE pilot numbers come from:

```text
/root/autodl-tmp/vae500_teacher_benchmark_xresnet50_k500_v7_20260602_sourceweighted
```

| center | direct K500 | old VAE500 online AT | teacher-VAE online AT |
|---|---:|---:|---:|
| ningbo | 0.8885 / 0.5241 | 0.8886 / 0.5234 | 0.8881 / 0.5249 |
| cpsc_2018 | 0.8529 / 0.5840 | 0.8539 / 0.5856 | 0.8538 / 0.5856 |

Notes:

- CPSC teacher-VAE training stopped by the ASR stop condition after saving a
  valid `best_model.pt`; evaluation used that best checkpoint.
- The teacher-VAE downstream effect is tiny: Ningbo gains only AUPRC, CPSC is
  essentially tied with old VAE500 online AT.

## External 500Hz VAE Candidates

Web check summary:

- DiffuSETS is still the most relevant public family. The paper describes
  10-second, 12-lead, 500Hz ECGs and a VAE compression/reconstruction path, and
  the official implementation is public.
- ECGEN is the closest independent VAE candidate: PyPI documents a 12-lead
  5000-sample VAE with residual 1D conv blocks and a decoder. It is useful as an
  external VAE baseline or teacher, but public pretrained VAE weights are not
  yet as clearly usable as our current local VAE assets.
- ECG masked autoencoders / foundation encoders are usually not drop-in
  replacements because our method needs a differentiable decoder back to
  10-second 12-lead waveforms.

Sources:

- DiffuSETS paper / public code: https://pmc.ncbi.nlm.nih.gov/articles/PMC12546759/
- DiffuSETS GitHub: https://github.com/PKUDigitalHealth/DiffuSETS
- ECGEN PyPI: https://pypi.org/project/ecgen/
- ECGEN GitHub: https://github.com/vlbthambawita/ECGEN

## Current Decision

Do not spend long GPU time on plain ECGTwin-init VAE500 continuation. The better
engineering direction is:

1. keep current DiffuSETSVAE500 as the active downstream VAE;
2. keep ECGTwin-teacher DiffuSETSVAE500 as a valid candidate, but do not scale it
   unless the online AT objective changes;
3. explore ECGEN only as an external baseline/teacher, not as a main-claim
   pretrained replacement unless clearly labeled as external pretraining;
4. focus the next online-AT effort on objective/selection changes, because small
   VAE reconstruction improvements did not translate into the requested +2pp
   gain over direct K500 fine-tuning.
