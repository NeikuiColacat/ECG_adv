# VAE500 K200 Gate And VAE Bottleneck Update

Date: 2026-06-02

## Why This Was Run

当前 500Hz VAE-online-AT 相对 matched direct fine-tune 基本中性。为确认问题是否来自
K=500 样本量过大导致边际收益消失，补了 K=200 的小规模 gate。同时，把“ECGTwin 原版
VAE 权重初始化/蒸馏出更强 500Hz VAE”和“外部 500Hz ECG VAE”纳入后续受控分支。

固定协议：

```text
mapping: v7_super5_sjr_rgq_review_20260528
preprocess: minimal_resample
normalization: per_sample_global
sampling: 500Hz, 5000 samples = 10s
backbone: EfficientNet1DV2
final eval: target-center K ref ids excluded
```

## K200 Direct Baseline

| center | direct K200 AUROC / AUPRC | note |
|---|---:|---|
| cpsc_2018 | 0.8476 / 0.5924 | 200 ref ids excluded |
| chapman_shaoxing | 0.8689 / 0.4436 | 200 ref ids excluded |

Run dirs:

```text
/root/autodl-tmp/vae500_lowk_k200_direct40_v7_20260602/cpsc_2018_k200_direct40_noadv_ep10_seed42_r2
/root/autodl-tmp/vae500_lowk_k200_direct40_v7_20260602/chapman_shaoxing_k200_direct40_noadv_ep10_seed42_r2
```

## K200 VAE Policy Gate

Only CPSC was expanded to VAE policy because it is the fastest hard gate.

Config:

```text
VAE: spectral-r001 DiffuSETSVAE500
attack: latent_hull
hull_M: 10
hull_lambda: 0.15
hull_steps: 5
hull_lr: 0.25
K_anchor: 160
policy: hard loss-gain acceptance, adv_accept_min_loss_gain=0.0
classes_in_scope: NORM, MI, STTC
trusted classes in this K200 pool: NORM, STTC
```

Internal K200-val looked strong:

```text
baseline internal val: 0.8497 / 0.6700
best internal val:     0.9232 / 0.8142
best epoch:            8
ASR range:             about 0.42-0.57
```

But final ref-excluded target result regressed:

| center | direct K200 | VAE policy K200 | delta |
|---|---:|---:|---:|
| cpsc_2018 | 0.8476 / 0.5924 | 0.8467 / 0.5905 | -0.09pp / -0.20pp |

Run dir:

```text
/root/autodl-tmp/vae500_lowk_k200_spectral_policy_v7_20260602/cpsc_2018_k200_hardlg_lam015_m10_k160_ep10_seed42
```

## Decision

K200 does not unlock the 500Hz VAE-online-AT advantage. The internal K-val
improvement did not transfer to ref-excluded target evaluation, so this branch
should not be expanded to Chapman/four-center unless the objective or VAE
manifold changes materially.

## VAE Bottleneck Branch

The next VAE-quality exploration is allowed, but only as a gated branch:

| branch | role | current status |
|---|---|---|
| ECGTwin-init VAE500 | strict-load original ECGTwin VAE weights into 5000-sample architecture | technically valid; existing pilots worse than spectral-r001 |
| ECGTwin-teacher VAE500 | current student VAE with frozen original ECGTwin VAE reconstruction/latent distillation | reconstruction slightly better in MSE, downstream not better |
| ECGEN native-5000 VAE | external open-source 12-lead 5000-sample VAE-style baseline | requires self-training and integration |
| SE-Diff-style losses | beat/first-cycle and simulator/inter-lead constraints | method reference only unless prerequisite weights are obtained |

Mandatory gate before expensive downstream sweeps:

```text
1. Reconstruction gate:
   candidate must match/beat active spectral-r001 on Pearson and MSE,
   and must not worsen lead residual materially.

2. Latent-hull gate:
   decoded_invalid_rate near 0,
   positive loss_gain,
   ASR roughly 0.3-0.7,
   no obvious waveform failure from anchor-local same-label interpolation.

3. Two-center downstream gate:
   centers: cpsc_2018 and ningbo
   controls: direct40 K500 and spectral-r001 loss-gain policy
   continue only if CPSC improves and Ningbo AUPRC is non-decreasing.
```

If a new VAE only improves reconstruction but fails the downstream gate, the
main bottleneck should be treated as online-AT objective/selection rather than
pixel-level VAE reconstruction.
