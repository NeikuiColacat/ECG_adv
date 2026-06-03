# VAE Online AT Refine: AugMix Stage A, 2026-06-01

## Purpose

Strict matched direct fine-tuning erased the plain VAE500 online AT advantage.
Stage A tests whether adding ECG AugMix branches around VAE latent-hull
adversarial samples can recover useful gain beyond direct FT.

## Protocol

- Backbone: EfficientNet1DV2 500Hz.
- VAE: PTB-XL-only VAE500.
- Mapping: `v7_super5_sjr_rgq_review_20260528`.
- K: 500 target-center real anchors.
- Final eval: target-center K500 refs excluded.
- Centers: `cpsc_2018`, `chapman_shaoxing`.
- Epochs: 6 smoke.
- Online AT: latent hull with `M=20`, `lambda=0.05`, `hull_steps=3`.
- AugMix branch: `x_adv` plus ECG corruption chains from clean anchor.
- Ops: `powerline_noise`, `emg_noise`, `baseline_wander`, `baseline_shift`.
- Excluded in Stage A: `random_leads_masking`.
- Internal severity values: `2, 4, 6, 8, 10`.
- Labels: current hard multi-hot anchor labels.
- Large output root:
  `/root/autodl-tmp/vae500_augmix_stageA_v7_20260601/`.

## Results

All target results are K500 ref-excluded macro `AUROC / AUPRC`.

| center | method | target AUROC/AUPRC | delta vs direct | PTB-XL |
|---|---|---:|---:|---:|
| cpsc_2018 | direct | 0.8533 / 0.6011 | +0.00pp / +0.00pp | 0.9116 / 0.7826 |
| cpsc_2018 | plain_vae | 0.8482 / 0.5932 | -0.51pp / -0.80pp | 0.9120 / 0.7848 |
| cpsc_2018 | augmix_s2 | 0.8493 / 0.5949 | -0.40pp / -0.62pp | 0.9118 / 0.7835 |
| cpsc_2018 | augmix_s4 | 0.8499 / 0.5954 | -0.34pp / -0.57pp | 0.9118 / 0.7835 |
| cpsc_2018 | augmix_s6 | 0.8494 / 0.5949 | -0.39pp / -0.63pp | 0.9118 / 0.7833 |
| cpsc_2018 | augmix_s8 | 0.8493 / 0.5949 | -0.40pp / -0.62pp | 0.9118 / 0.7834 |
| cpsc_2018 | augmix_s10 | 0.8492 / 0.5946 | -0.41pp / -0.65pp | 0.9119 / 0.7837 |
| chapman_shaoxing | direct | 0.8725 / 0.4448 | +0.00pp / +0.00pp | 0.9111 / 0.7807 |
| chapman_shaoxing | plain_vae | 0.8728 / 0.4459 | +0.03pp / +0.11pp | 0.9109 / 0.7804 |
| chapman_shaoxing | augmix_s2 | 0.8732 / 0.4429 | +0.07pp / -0.19pp | 0.9115 / 0.7832 |
| chapman_shaoxing | augmix_s4 | 0.8728 / 0.4413 | +0.03pp / -0.35pp | 0.9118 / 0.7820 |
| chapman_shaoxing | augmix_s6 | 0.8727 / 0.4420 | +0.02pp / -0.28pp | 0.9117 / 0.7818 |
| chapman_shaoxing | augmix_s8 | 0.8727 / 0.4413 | +0.02pp / -0.35pp | 0.9118 / 0.7820 |
| chapman_shaoxing | augmix_s10 | 0.8733 / 0.4427 | +0.08pp / -0.21pp | 0.9115 / 0.7832 |

## Interpretation

Stage A does not satisfy the success criterion.

- CPSC: AugMix improves plain VAE slightly, but remains below direct FT.
- Chapman: AugMix slightly improves AUROC, but consistently lowers AUPRC.
- PTB-XL source performance is stable, so the failure is not source collapse.
- The likely issue is not severity alone. The current hard-label AugMix/VAE
  branch is adding samples that do not improve target-center ranking.

## Next Action

Do not scale Stage A severity settings.

Next smoke should test safer adversarial supervision:

```text
1. lower adv_weight: 0.10 or 0.15
2. lower latent_augmix_latent_weight_cap: 0.15
3. use soft labels: latent_mixed_teacher or teacher_soft
4. keep best mild/medium severity: internal severity 4
5. run only cpsc_2018 and chapman_shaoxing first
```

Promotion rule remains:

```text
AUROC or AUPRC >= +2pp over matched direct FT, and the other metric must not
decrease.
```
