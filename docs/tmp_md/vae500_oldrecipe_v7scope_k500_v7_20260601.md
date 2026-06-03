# VAE500 Old-Recipe V7-Scope K500 Check

Date: 2026-06-01

## Purpose

Follow-up to `vae500_oldrecipe_k500_v7_20260601.md`.

Old 100Hz recipe used `NORM/MI/STTC`, but v7 K500 target anchors contain very
few MI samples and many CD/HYP samples. This run keeps the old-recipe attack
strength but changes adversarial anchor scope to match v7-positive classes:

```text
classes_in_scope: CD, HYP, NORM, STTC
allow_hyp_cd_trust: true
hull_M: 10
hull_lambda: 0.15
hull_steps: 5
K_anchor: 300
adv_label_mode: mixed_soft
adv_teacher_mix: 0.3
target_real_weight: 40
adv_weight: 0.06
quality gate: enabled
```

Run root:

```text
/root/autodl-tmp/vae500_oldrecipe_v7scope_k500_v7_20260601
```

## Heldout Results

| center | direct K500 FT | old NORM/MI/STTC | v7-scope CD/HYP/NORM/STTC | v7-scope - direct |
|---|---:|---:|---:|---:|
| ningbo | 0.8798 / 0.5023 | 0.8790 / 0.4964 | 0.8784 / 0.4955 | -0.13pp / -0.67pp |
| chapman_shaoxing | 0.8725 / 0.4448 | 0.8749 / 0.4507 | 0.8739 / 0.4511 | +0.14pp / +0.63pp |
| cpsc_2018 | 0.8533 / 0.6011 | 0.8671 / 0.6221 | 0.8568 / 0.6017 | +0.35pp / +0.06pp |
| georgia | 0.8347 / 0.6180 | 0.8360 / 0.6192 | 0.8358 / 0.6191 | +0.11pp / +0.11pp |
| mean | 0.8601 / 0.5416 | 0.8643 / 0.5471 | 0.8612 / 0.5419 | +0.12pp / +0.03pp |

## Diagnostics

| center | epochs | ASR mean | loss_gain mean | decoded invalid | note |
|---|---:|---:|---:|---:|---|
| ningbo | 10 | 0.482 | 0.0539 | 0.000 | stronger attack, worse heldout |
| chapman_shaoxing | 10 | 0.432 | 0.0009 | 0.000 | slight AUPRC gain |
| cpsc_2018 | 3 | 0.292 | -0.0253 | 0.000 | stopped by ASR-low guard; best checkpoint evaluated |
| georgia | 10 | 0.457 | 0.0270 | 0.000 | tiny heldout gain |

## Interpretation

Adding CD/HYP increases attack activity on Ningbo and Georgia but does not
improve heldout generalization. On CPSC it removes the strong STTC-centered
gain from the old `NORM/MI/STTC` recipe:

```text
CPSC old scope:      +1.38pp AUROC / +2.09pp AUPRC
CPSC v7-aware scope: +0.35pp AUROC / +0.06pp AUPRC
```

This suggests the useful VAE500 signal in the current setup is not "use every
frequent v7 class." It is more class-specific: CPSC benefits mainly from STTC
boundary shaping, while CD/HYP anchors dilute or destabilize the effect.

## Decision

Do not use the v7-scope CD/HYP/NORM/STTC recipe as the next mainline.

Next useful direction should be class-aware rather than center-global:

```text
1. keep direct K500 as the default head for classes where VAE hurts;
2. use old-recipe VAE candidate only for classes where K500-internal validation
   supports it, especially STTC-like gains;
3. evaluate a paper-safe classwise selector over direct/plain/old/v7scope
   candidates using only K500-internal validation and PTB-XL source floor.
```
