# VAE500 Direct40-Init Pure VAE Pilot

Date: 2026-06-01

## Purpose

After the direct40 control showed that old-recipe gains mostly come from the
supervised target-real stream, this pilot tested a cleaner VAE-only refinement:

```text
init checkpoint: per-center direct40/ep10/no-adv best_model.pt
target_real_weight: 0
ptbxl_weight: 1
adv_weight: 0.30
classes_in_scope: NORM, MI, STTC
adv_label_mode: mixed_soft
teacher_mix: 0.3
hull_lambda: 0.15
K_anchor: 300
epochs: 8
```

Run root:

```text
/root/autodl-tmp/vae500_direct40init_purevae_pilot_v7_20260601
```

## Results

| center | direct40/ep10 | direct40-init pure VAE | delta |
|---|---:|---:|---:|
| cpsc_2018 | 0.8633 / 0.6155 | 0.8633 / 0.6155 | +0.00pp / +0.00pp |
| ningbo | 0.8810 / 0.5032 | 0.8810 / 0.5032 | +0.00pp / +0.00pp |

## Training Behavior

For both centers, K500-internal validation dropped immediately after VAE-only
updates. The selected best checkpoint stayed at the direct40 initialization.

Example:

```text
CPSC K500-val baseline: 0.9128 / 0.8289
after VAE epochs:       fell to about 0.89 / 0.74-0.78

Ningbo K500-val baseline: 0.8748 / 0.8012
after VAE epochs:         fell to about 0.86 / 0.77-0.79
```

## Interpretation

Starting from a strong direct target-adapted checkpoint and adding pure VAE
adversarial refinement does not improve the model. It is actively rejected by
K500-internal validation.

This supports the current working conclusion:

```text
VAE500 latent-hull samples can create attack pressure, but under the current
loss/selection design they do not provide useful extra generalization beyond a
properly matched direct target fine-tune.
```

The next useful step should be a different mechanism, not another minor
scope/weight tweak:

```text
1. adversarial consistency/distillation instead of hard or mixed-soft BCE;
2. explicit per-class STTC-focused branch for CPSC-like centers;
3. backbone transfer only after an EfficientNet variant beats direct40;
4. stronger paper-safe selector only if the candidate pool contains genuinely
   positive VAE candidates.
```
