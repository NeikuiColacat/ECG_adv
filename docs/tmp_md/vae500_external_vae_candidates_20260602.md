# 500Hz ECG VAE Candidate Audit

Date: 2026-06-02

## Question

Can we replace or strengthen the current PTB-XL-only VAE500 used by VAE-only
online adversarial training?

Current problem:

```text
The current VAE500 reconstructs ECG well, but downstream online AT is mostly
neutral versus matched direct fine-tune. This suggests the bottleneck may be
latent geometry or the online-AT objective, not only reconstruction quality.
```

## Candidate Summary

| candidate | decoder-capable VAE? | 500Hz / 12-lead fit | weights status | decision |
|---|---|---|---|---|
| SE-Diff | yes, ECGTwin-style VAE plus latent diffusion | strong fit: 10s, 12-lead, 500Hz, compact 4-channel latent | code found; pretrained weights not found in repo | highest-priority architecture reference; verify weights separately |
| DiffuSETS / ECGTwin | yes | current project family | weights already available locally | keep as owned baseline and teacher |
| ECGEN | yes, 1D conv VAE | strong engineering fit: 12 x 5000 | code found; mature VAE weights not confirmed | secondary baseline, not main replacement |
| D-BETA / C-MELT / ECG-FM / ST-MEM | no direct VAE decoder for this pipeline | feature/model dependent | varies | feature teacher only |
| WearECG / MCMA | reconstruction/lead-completion, not VAE manifold decoder | partial fit | varies | not suitable for latent-hull online AT decoder |

## SE-Diff Local Audit

Local clone:

```text
path: model/SE-Diff
sha:  dd0a476
size: 4.8M
```

Repository facts:

```text
README says required pretrained weights, latent data, mini decoder, and
simulator priors must be placed under ./prerequisites/.

No release, download script, or actual pretrained weight file was found in the
cloned repository.
```

Code facts:

```text
model/SE-Diff/vae/vae_model.py contains an ECGTwin-style VAE_Encoder and
VAE_Decoder. The main VAE family is not a fundamentally new architecture versus
the original ECGTwin VAE; it is the same compact 4-channel latent family.

The useful additions are:
  VAE_Decoder_FirstCycle
  MiniDecoderMSE
  simulator/inter-lead losses in utils/simulator_trainer.py
```

Practical interpretation:

```text
SE-Diff is not an immediate drop-in replacement unless the pretrained VAE
weights can be obtained. Its best short-term value is to guide a stronger
PTB-XL-only VAE500 objective.
```

## Recommended VAE500 Refinement

Keep the main claim PTB-XL-only. Use ECGTwin original VAE weights as the teacher
or initializer, and borrow SE-Diff's training constraints:

```text
1. ECGTwin-init VAE500:
   original ECGTwin VAE architecture at input_len=5000
   strict ECGTwin encoder/decoder initialization
   PTB-XL-only fine-tuning

2. Teacher-distilled DiffuSETSVAE500:
   current lighter VAE500 student
   frozen original ECGTwin VAE teacher at 1024 samples
   teacher reconstruction + latent distillation

3. SE-Diff-style additions:
   spectral reconstruction loss
   first-cycle / beat-level auxiliary decoder
   inter-lead/Einthoven consistency loss
   hard invalid-decode rejection
```

Stop condition before downstream online AT:

```text
Do not run expensive PN2021 online-AT sweeps unless the candidate VAE beats or
matches the current VAE500 reconstruction gate and improves latent-hull
validation behavior on a two-center pilot.
```

## Source Links

```text
SE-Diff paper:  https://openreview.net/pdf?id=95ZV35sBDm
SE-Diff code:   https://github.com/ignite-abd/SE-Diff
DiffuSETS code: https://github.com/Raiiyf/DiffuSETS_Exp
DiffuSETS data: https://zenodo.org/records/15420698
ECGEN code:     https://github.com/vlbthambawita/ECGEN
```

## Follow-up External Audit Update

Subagent web audit conclusion:

```text
As of the public material checked on 2026-06-02, no credible drop-in
native-500Hz / 10s / 12-lead ECG VAE or VQ-VAE checkpoint was found that can
directly replace the current project VAE.
```

Additional candidates and decisions:

| candidate | role | decision |
|---|---|---|
| ECGEN | 12-lead 5000-sample VAE code path | best short-term external architecture baseline; public reusable VAE weights remain unclear |
| SE-Diff | ICLR 2026 ECG latent diffusion with VAE, beat decoder, simulator constraints | best method reference; repo requires prerequisite weights/data and is not a drop-in checkpoint |
| TimeVQVAE | generic time-series VQ-VAE/tokenizer | possible long-term 500Hz ECG tokenizer, but requires full retraining and reworking latent-hull logic |
| CardioRadar hierarchical VQ-VAE | strong representation/pretraining idea | useful literature reference, but no direct public drop-in weights for this project |
| SSSD-ECG | raw ECG diffusion | relevant generation baseline, not a VAE latent manifold replacement |

Updated recommendation:

```text
Do not rely on finding an off-the-shelf 500Hz ECG VAE checkpoint. For the
current paper timeline, keep using the owned PTB-XL-only VAE500 family and
focus on online-AT objective/selection. If VAE architecture work resumes,
prefer:
  1. ECGEN-style 5000-sample 12-lead VAE as an engineering baseline;
  2. SE-Diff-style beat/simulator/inter-lead losses as training constraints;
  3. VQ-VAE/tokenizer routes only as a longer-term extension.
```

## Subagent Web Audit Update

Follow-up web/subagent audit reached the same practical conclusion:

```text
The only public families that are already close to our latent-hull online-AT
loop are ECGTwin / DiffuSETS-style VAE models. Public native-500Hz, 10-second,
12-lead, decoder-capable VAE checkpoints are not currently confirmed as a
drop-in replacement.
```

| model | code / weights | input protocol | decoder latent | online-AT usability | decision |
|---|---|---|---|---|---|
| ECGTwin | GitHub/HF weights available locally | MIMIC-IV-ECG raw is 500Hz, but the released VAE path uses 1024 x 12 | VAE latent 4 x 128 | directly compatible with current project loop | keep as original trusted VAE family |
| DiffuSETS | GitHub, Zenodo, HF assets reported public | original ECG may be 500Hz/5000, released latent path uses 1024-style VAE | VAE latent 4 x 128 | useful external/public-weight A/B | evaluate only as external-pretraining or public-weight baseline |
| SE-Diff | ICLR 2026 code found, weights not found in repo | paper targets 10s / 12-lead / 500Hz | ECGTwin-style compact latent | not drop-in without prerequisites | use as method reference for losses/constraints |
| ECGEN | code-level 12 x 5000 VAE path | native 500Hz / 5000 samples | VAE code path exists | requires training and integration | secondary native-5000 engineering baseline |
| WearECG | reconstruction / lead-completion VAE | 500Hz but short-window lead reconstruction | not a full ECG latent generator | not suitable for our latent-hull decoder loop | structure reference only |
| ECG-Diffusion-DiffWave | raw diffusion | MIT-BIH style, not 12-lead PTB-XL super5 | no VAE decoder | not suitable | reject |
| CR-VAE / medical time-series VAE | generic medical time series | not ECG waveform-specific | generic VAE | not suitable without major redesign | method reference only |
| DCDM-ECG | reports PTB-XL VAE-like latent route | no accepted public code/weights confirmed | frozen VAE in paper | cannot run now | monitor only |

Practical decision:

```text
Do not replace the current VAE in the active goal. If the paper needs a native
500Hz VAE experiment, run it as a separate branch:
  ECGTwin-init or ECGEN-style 5000-point VAE
  -> PTB-XL-only training
  -> reconstruction/latent gate
  -> CPSC + Ningbo downstream gate
  -> only then four-center expansion.
```

## ECGEN Local Clone Audit

Local shallow clone:

```text
path: model/ECGEN
size: about 668K
```

Useful facts:

```text
model/ECGEN/src/ecgen/models/vae.py
  VAE1D with Encoder1D / Decoder1D
  input protocol: (B, 12, 5000)
  default latent_channels: 8
  channel_multipliers: (1, 2, 4, 4)
  downsampling stages: 3 stride-2 convs, so 5000 -> about 625 latent length
  loss: MSE reconstruction + kl_weight * KL

model/ECGEN/configs/experiments/vae_mimic.yaml
  default data target is MIMIC-IV-ECG, not PTB-XL
  default max_epochs=100, batch_size=32, seq_length=5000
```

Decision:

```text
ECGEN is the most concrete native-5000 engineering reference found so far, but
it is not a drop-in replacement for our current loop:
  - default latent shape is 8 x 625, not ECGTwin/DiffuSETS-style 4 x 625;
  - default training recipe is MIMIC, while our main claim must stay PTB-XL-only;
  - no mature public VAE checkpoint was confirmed in the local clone.

If used, implement it as a separate PTB-XL-only self-trained VAE baseline:
  ECGEN-style VAE500 -> reconstruction audit -> latent-hull adapter -> CPSC/Ningbo gate.
```
