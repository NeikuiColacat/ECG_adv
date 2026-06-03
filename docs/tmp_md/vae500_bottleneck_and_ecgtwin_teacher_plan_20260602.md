# 500Hz VAE Bottleneck And ECGTwin-Teacher Plan

Date: 2026-06-02

## Current Diagnosis

The current 500Hz VAE branch is valid but not yet strong enough to beat the
matched direct target fine-tune baseline by the requested margin.

Key matched evidence:

| experiment | conclusion |
|---|---|
| K500 / K1000 / 10% / 20% plain VAE500 LHAT | mostly neutral versus matched direct fine-tune |
| direct40-init pure VAE | K500-internal validation selected epoch 0 |
| direct40-init teacher-soft VAE | K500-internal validation selected epoch 0 |
| original ECGTwin 1024 latent diagnostic | also selected epoch 0 after adapting decoded output to 500Hz classifier input |
| ECGTwin-teacher VAE500 + loss-gain | technically valid, but weaker than spectral-r001 + loss-gain |

Interpretation:

```text
500Hz VAE quality may still be a bottleneck, but the evidence does not support
blindly training a larger VAE as the next main route. The stronger diagnosis is
that the current online adversarial objective and acceptance/selection policy
do not create enough useful signal once direct K500/K40 fine-tune is matched.
```

## Hard Vs Soft Loss-Gain Four-Center Matrix

Protocol:

```text
backbone: EfficientNet1DV2 500Hz
mapping: v7_super5_sjr_rgq_review_20260528
input: 500Hz, 5000 samples, per-sample global normalization
target anchors: K=500 per center
eval: target-center K500 reference ids excluded
VAE backend: spectral-r001 VAE500
selection: internal AUPRC for latest pilot runs where available
```

| center | direct40 | hard loss-gain | hard delta | soft loss-gain | soft delta |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8810 / 0.5032 | 0.8807 / 0.5018 | -0.03pp / -0.14pp | 0.8818 / 0.5063 | +0.08pp / +0.32pp |
| chapman_shaoxing | 0.8757 / 0.4508 | 0.8742 / 0.4522 | -0.16pp / +0.14pp | 0.8727 / 0.4517 | -0.30pp / +0.09pp |
| cpsc_2018 | 0.8633 / 0.6155 | 0.8688 / 0.6270 | +0.55pp / +1.15pp | 0.8670 / 0.6206 | +0.37pp / +0.51pp |
| georgia | 0.8364 / 0.6202 | 0.8340 / 0.6165 | -0.24pp / -0.36pp | 0.8349 / 0.6180 | -0.14pp / -0.21pp |
| mean | 0.8641 / 0.5474 | 0.8644 / 0.5494 | +0.03pp / +0.20pp | 0.8641 / 0.5492 | +0.00pp / +0.18pp |

Best-by-AUPRC exploratory selector among direct40, hard, and soft:

| center | selected | AUROC / AUPRC | delta vs direct40 |
|---|---|---:|---:|
| ningbo | soft loss-gain | 0.8818 / 0.5063 | +0.08pp / +0.32pp |
| chapman_shaoxing | hard loss-gain | 0.8742 / 0.4522 | -0.16pp / +0.14pp |
| cpsc_2018 | hard loss-gain | 0.8688 / 0.6270 | +0.55pp / +1.15pp |
| georgia | direct40 | 0.8364 / 0.6202 | +0.00pp / +0.00pp |
| mean | exploratory selector | 0.8653 / 0.5514 | +0.12pp / +0.40pp |

Decision:

```text
Hard/soft loss-gain is a small AUPRC improvement, not a breakthrough. It does
not meet the +2pp goal. Do not continue small hard/soft loss-gain sweeps as the
main line unless the objective changes materially.
```

## ECGTwin Author VAE Transfer Route

Local source checkpoint:

```text
model/ECGTwin/checkpoints/vae_model.pth
checkpoint keys: encoder, decoder
original input/output: 1024 x 12
original latent: (4, 128)
```

Important architecture fact:

```text
The original ECGTwin VAE is convolutional and mostly length-flexible. Reusing
the same VAE module family at 5000 samples naturally gives latent (4,625).
This supports a clean ECGTwinVAE500 wrapper with strict author-weight loading.
```

Local audit result, 2026-06-02:

```text
The original ECGTwin checkpoint can be strictly loaded into ECGTwinVAE500.
Conv1d, GroupNorm, residual blocks, and self-attention weights do not depend on
the temporal length. The input length changes activations from 1024 -> 5000 and
latents from 128 -> 625, but not parameter shapes.
```

Existing implementation:

```text
ecg_adv_gen/vae/ecgtwin_vae500.py
  ECGTwinVAE500 wrapper
  input:  (B,5000,12)
  latent: (B,4,625)
  output: (B,5000,12)
  handles PTB-XL <-> ECGTwin lead-order conversion internally

scripts/vae500/train_ptbxl_vae500.py
  --model_variant ecgtwin_init_vae500
  --init_ecgtwin_vae_ckpt model/ECGTwin/checkpoints/vae_model.pth
```

Do not do this:

```text
Do not partial-load the original ECGTwin checkpoint into the current
DiffuSETSVAE500. Module names, widths, attention blocks, and downsampling
details differ too much. It would create an unclear hybrid with weak
reproducibility.
```

Recommended controlled routes:

| route | description | when to continue |
|---|---|---|
| ECGTwin-init VAE500 | same original ECGTwin encoder/decoder family, input_len=5000, strict-load author weights, PTB-XL-only fine-tune | only if reconstruction audit is within 5% of current VAE500 and latent-hull smoke passes |
| ECGTwin-teacher DiffuSETSVAE500 | freeze original ECGTwin 1024 VAE as teacher and distill reconstruction/latent geometry into 5000-sample student | only if two-center downstream pilot beats current spectral-r001 or avoids epoch0 selection |
| spectral / morphology / inter-lead losses | keep owned VAE500, add losses inspired by SE-Diff style constraints | continue only if it improves both reconstruction gate and downstream two-center gate |

Execution priority after the positive-hide pilot:

```text
1. Run/refresh reconstruction audit for current spectral-r001 VAE500,
   ECGTwin-init VAE500, and teacher-distilled VAE500 on the same PTB-XL fold.
2. If ECGTwin-init is not worse than spectral-r001 by more than 5%, export
   CPSC and Ningbo K500 anchors with that VAE.
3. Run the same direct40-init / hard loss-gain pilot on CPSC and Ningbo.
4. Expand to four centers only if CPSC AUPRC improves and Ningbo does not
   regress beyond -0.1pp AUPRC versus direct40.
```

Minimum gates before a long downstream sweep:

```text
Gate A: strict load / finite shape check
  input (B,5000,12), latent (B,4,625), recon (B,5000,12), all finite

Gate B: reconstruction audit
  invalid decode rate < 1%
  MSE/MAE not worse than current spectral-r001 VAE500 by more than 5%
  global Pearson and first-diff Pearson not worse
  lead residual ratio not worse

Gate C: latent-hull plausibility
  same-label latent interpolation has no flatline/exploding amplitude
  decoded invalid rate near zero

Gate D: two-center downstream pilot
  centers: cpsc_2018 and ningbo
  matched control: direct40/direct K500
  must not select epoch0 on both centers
  cpsc_2018 AUPRC should improve by about +1pp
  ningbo AUPRC should be non-decreasing or within -0.1pp
```

## External 500Hz VAE Candidate Audit

Subagent/web audit result:

```text
No non-DiffuSETS drop-in native-500Hz / 10s / 12-lead ECG VAE or VQ-VAE
checkpoint was confirmed. DiffuSETS is the only public-asset candidate that
looks immediately auditable as a pretrained VAE source.
```

| candidate | source | role | decision |
|---|---|---|---|
| DiffuSETS | https://github.com/Raiiyf/DiffuSETS_Exp; https://huggingface.co/Laiyf/DiffuSETS; https://zenodo.org/records/15420698 | native 10s / 12-lead / 500Hz VAE-family assets | highest-priority external checkpoint audit; report as external pretraining if used |
| ECGEN | https://github.com/vlbthambawita/ECGEN | 12-lead 5000-sample VAE code path | useful engineering baseline, public reusable VAE weights unclear |
| SE-Diff | https://github.com/ignite-abd/SE-Diff | ECG latent diffusion with VAE, beat decoder, simulator constraints | best method reference, but repo requires prerequisite weights/data |
| TimeVQVAE | https://github.com/ML4ITS/TimeVQVAE | generic time-series VQ-VAE/tokenizer | longer-term tokenizer route, not a drop-in ECG VAE |
| SSSD-ECG | https://github.com/AI4HealthUOL/SSSD-ECG | raw ECG diffusion baseline | not a VAE latent manifold replacement |

Recommendation:

```text
Do not pause the paper line waiting for a new external 500Hz VAE checkpoint.
If external assets are audited, start with DiffuSETS encoder/decoder only and
compare reconstruction, lead order/scaling, latent interpolation validity, and
CPSC/Ningbo latent-hull AT behavior against the current project VAE500.

Keep the causal claim PTB-XL-only unless an external pretrained model is
explicitly reported as external pretraining.
```

External audit interpretation:

```text
ECGEN is the closest native 5000-sample/12-lead VAE-style reference, but it is
not currently a drop-in replacement for ECGTwin-compatible latent-hull AT.
SE-Diff is a stronger modern method reference for ECG latent diffusion and
simulator/morphology constraints, but it should inform losses and validation
rather than replace the PTB-XL-only VAE without a separate external-pretraining
claim.

DiffuSETS is the practical external checkpoint route, but it is not a free
paper-mainline replacement. If its pretrained assets use MIMIC-scale data, the
run must be labeled as an external-pretrained VAE ablation.
```

## Next Execution Decision

Priority order:

1. Freeze the small hard/soft loss-gain and positive-hide/negative-add sweeps
   as negative/limited results.
2. Use ECGTwin author VAE only through a gated ECGTwin-init or teacher-distill
   pilot, not an open-ended long training run.
3. Put new compute into objective-level changes that can create stronger
   adversarial signal:
   - class/rare-label ranking branch;
   - source-preserving target adapter;
   - K500-internal paper-safe selector;
   - stronger positive-hide / negative-add multilabel attack diagnostics.
4. Revisit VAE architecture only if the new objective shows stable two-center
   gain with current or spectral-r001 VAE.

Positive-hide / negative-add follow-up, 2026-06-02:

```text
CPSC poshide+negadd pilot:
  direct40       0.8633 / 0.6155
  hard loss-gain 0.8688 / 0.6270
  poshide+negadd 0.8675 / 0.6234

Decision:
  keep the implementation and diagnostics;
  do not expand this exact objective to four centers;
  prioritize gated ECGTwin-init/teacher VAE500 tests next.
```

Latest gated-VAE decision after experiment audit:

```text
ECGTwin-init VAE500 and ECGTwin-teacher DiffuSETSVAE500 are technically valid,
but current downstream pilots do not beat the active spectral-r001 VAE500
branch. Treat them as stopped under the current online-AT objective.

Only resume this branch if either:
  1. a new online-AT objective first improves CPSC and Ningbo with current VAE,
     then needs a better latent space; or
  2. the official DiffuSETS pretrained VAE assets pass a strict checkpoint audit
     and are reported as external pretraining.
```

Read-only experiment audit closure:

```text
Plain continuation of ECGTwin-init VAE500 is not justified by current evidence.
The branch passes load/shape/finite checks, but its reconstruction quality is
worse than the active DiffuSETSVAE500 and it has no positive downstream gate.
```

Key audit numbers:

| VAE branch | checkpoint / run | Pearson | first-diff Pearson | MSE | lead residual | decision |
|---|---|---:|---:|---:|---:|---|
| active DiffuSETSVAE500 | `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_v1_fast_b64_lc025_20260601/checkpoints/best.pt` | 0.997656 | 0.950460 | 0.007098 | 1.0883 | active baseline |
| ECGTwin-init VAE500 | `/root/autodl-tmp/vae500/ptbxl_records500_v7/ecgtwin_init_vae500_b64_lc100_cont10_20260602` | 0.996431 | 0.947812 | 0.009768 | 1.1255 | stop under current objective |
| ECGTwin-teacher DiffuSETSVAE500 | `/root/autodl-tmp/vae500/ptbxl_records500_v7/diffusets500_teacher_r002_z001_b64_20260602` | 0.997734 | 0.950630 | 0.006939 | 1.1592 | keep as gated candidate only |

Matched-control reminder:

```text
The old VAE recipe looked helpful versus source-only, but not versus the matched
direct40/ep10 target fine-tune control:

direct40 mean:        0.8641 / 0.5474
old-recipe VAE mean:  0.8643 / 0.5471
delta:                +0.02pp / -0.03pp

Therefore the current bottleneck is not simply "more ECGTwin VAE fine-tuning";
it is the online-AT signal, sample policy, and paper-safe selection rule.
```
