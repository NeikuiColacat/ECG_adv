# Center Token v42 Validation Notes

Date: 2026-05-03

## Artifacts

```text
style aux:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/style_aux_noleak_full10_task1_v1
  test acc / macro F1 = 0.6279 / 0.6555

v42 token bank:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v42_noleak_style_semantic_direct_mv4_actual_steps1500_20260503/prompt_token_bank.pt

v42 validation:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_noleak_style_semantic_20260503
```

## What Changed

v42 uses a no-leak real-only full10 style aux checkpoint instead of the older
7-way full10 style classifier:

```text
real PN2021 only
K=500 center-token anchors excluded per center
centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
classes = NORM, MI, STTC
preprocess = minimal_resample + per_sample_global + full 10s
```

Token training used:

```text
direct MV4
steps = 1500
style_loss_weight = 0.003
semantic_loss_weight = 0.02
aux_timestep_max = 250
semantic ckpt = Task-1 full10 EfficientNet1DV2
```

## Result

v42-A0 does not pass the promotion gate.

Task-1 full10 no-leak EfficientNet feature style score:

| center | raw target score | best raw control | gated target score | best gated control |
|---|---:|---:|---:|---:|
| ningbo | 0.5766 | 0.4729 | 0.5268 | 0.4827 |
| chapman_shaoxing | 0.4276 | 0.4872 | 0.3928 | 0.5134 |
| cpsc_2018 | 0.6027 | 0.5716 | 0.5894 | 0.5970 |
| georgia | 0.4146 | 0.5696 | 0.4064 | 0.6230 |

Same-label C2ST balanced accuracy, lower is better:

| center | raw target C2ST | best raw control | gated target C2ST | best gated control |
|---|---:|---:|---:|---:|
| ningbo | 0.7347 | 0.6097 | 0.5500 | 0.3500 |
| chapman_shaoxing | 0.6417 | 0.5347 | 0.6905 | 0.3667 |
| cpsc_2018 | 0.7833 | 0.5167 | 0.7000 | 0.4417 |
| georgia | 0.9167 | 0.4833 | 1.0000 | 0.5500 |

Gate pass rates:

| center | target | vanilla | wrong-center | PTB-XL-source |
|---|---:|---:|---:|---:|
| ningbo | 11/24 | 15/24 | 15/24 | 16/24 |
| chapman_shaoxing | 18/24 | 14/24 | 9/24 | 15/24 |
| cpsc_2018 | 13/16 | 9/16 | 9/16 | 10/16 |
| georgia | 12/24 | 12/24 | 11/24 | 19/24 |

## Interpretation

v42-A0 improves target-center style score only for `ningbo` and partly raw
`cpsc_2018`. It does not improve all centers, and C2ST becomes worse than
controls for every center in the raw EfficientNet feature check. Therefore it
must not enter downstream Latent-Hull online AT as positive center-token evidence.

The likely issue is not token capacity alone. The low-weight style aux moves
some center-probe scores, but the generated feature distribution becomes easier
to distinguish from real ECG. This is a sign of token-induced artifacts or
semantic/physiology drift rather than reliable target-center style transfer.

## Next Action

Do not expand final v42 samples. First run checkpoint selection on v42 step 500
and step 1000, because final-step token delta reached about 0.79 and may be too
large. If earlier checkpoints improve C2ST while preserving style score, use the
best checkpoint for a larger pool. If not, back off to a weaker style loss or
use style score only for post-hoc candidate selection instead of direct training
loss.
