# VAE500 Direct40 Control

Date: 2026-06-01

## Purpose

The 500Hz old-recipe run changed more than the VAE branch. It used:

```text
target_real_weight = 40
n_epochs = 10
```

while the earlier direct K500 control used:

```text
target_real_weight = 20
n_epochs = 30
```

This control disables the adversarial stream but keeps the old-recipe
supervised target-real setting:

```text
direct40/ep10/no-adv
target_real_weight = 40
n_epochs = 10
same K500 anchors
same K500-internal validation split
same v7 500Hz ref-excluded PN2021 evaluation
```

Run root:

```text
/root/autodl-tmp/vae500_direct40_ep10_v7_20260601
```

## Results

| center | direct20/ep30 | direct40/ep10 | old-recipe VAE500 | old - direct40 |
|---|---:|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8810 / 0.5032 | 0.8790 / 0.4964 | -0.20pp / -0.68pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8757 / 0.4508 | 0.8749 / 0.4507 | -0.09pp / -0.01pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8633 / 0.6155 | 0.8671 / 0.6221 | +0.38pp / +0.66pp |
| georgia | 0.8347 / 0.6180 | 0.8364 / 0.6202 | 0.8360 / 0.6192 | -0.03pp / -0.09pp |
| mean | 0.8601 / 0.5416 | 0.8641 / 0.5474 | 0.8643 / 0.5471 | +0.02pp / -0.03pp |

## Interpretation

This is the most important control from the 2026-06-01 refinement round.

The apparent old-recipe gain over direct20/ep30:

```text
+0.42pp AUROC / +0.55pp AUPRC
```

almost entirely disappears against the better matched direct40/ep10 control:

```text
+0.02pp AUROC / -0.03pp AUPRC
```

Therefore the old-recipe improvement should not be attributed to VAE online AT.
It is mostly explained by the supervised target-real stream weight and
checkpoint-selection dynamics.

The only center where VAE keeps a non-trivial net positive delta is CPSC:

```text
+0.38pp AUROC / +0.66pp AUPRC over direct40/ep10
```

This is useful mechanistic evidence but far below the global +2pp target.

## Consequence For The Goal

The current candidate pool:

```text
plain VAE500 LHAT
old NORM/MI/STTC recipe
v7-scope CD/HYP/NORM/STTC recipe
K500-internal classwise selector
```

does not beat a properly matched direct target fine-tune baseline.

Next experiments should avoid adding more supervised target-real confounding.
They should start from a strong direct checkpoint and add only a small VAE
regularization/refinement branch, or move to a different mechanism.
