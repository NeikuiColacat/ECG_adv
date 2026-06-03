# Why VAE500 Did Not Beat Direct Fine-Tune

Date: 2026-06-01

## Question

旧 ECGTwin VAE / 100Hz 路线有明显提升，为什么切到 500Hz VAE 后，
VAE online AT 相对 matched direct fine-tune 没有提升？

## Main Finding

当前证据不支持“500Hz VAE 完全不行”这个结论。

500Hz VAE online AT 相对 PTB-XL source-only baseline 仍然有接近旧路线的
提升；失效的是更严格的 comparison：

```text
VAE online AT vs matched direct target fine-tune
```

也就是说，当前问题更像是：

```text
direct fine-tune 已经吸收了少量目标中心样本的大部分收益，
VAE latent branch 没有在 direct FT 之上提供额外泛化收益。
```

## Source-Only Baseline Comparison

### Old 100Hz / original ECGTwin VAE / v3 mapping

From `docs/tmp_md/vae_only_lhat_kcurve_20260518.md`, K=500:

| center | source-only baseline | VAE-only LH-AT | delta |
|---|---:|---:|---:|
| ningbo | 0.8672 / 0.4784 | 0.8917 / 0.5223 | +2.44pp / +4.39pp |
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8967 / 0.4638 | +2.03pp / +3.87pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp |
| georgia | 0.8157 / 0.5916 | 0.8256 / 0.6020 | +1.00pp / +1.04pp |
| mean | 0.8427 / 0.5134 | 0.8671 / 0.5461 | +2.44pp / +3.27pp |

### New 500Hz / VAE500 / v7 mapping

Strict K500 source-only vs VAE500:

| center | source-only baseline | VAE500 LHAT | delta |
|---|---:|---:|---:|
| ningbo | 0.8598 / 0.4650 | 0.8774 / 0.4986 | +1.76pp / +3.36pp |
| chapman_shaoxing | 0.8602 / 0.4150 | 0.8728 / 0.4459 | +1.26pp / +3.09pp |
| cpsc_2018 | 0.7958 / 0.5427 | 0.8482 / 0.5932 | +5.24pp / +5.05pp |
| georgia | 0.8185 / 0.5963 | 0.8327 / 0.6135 | +1.42pp / +1.72pp |
| mean | 0.8336 / 0.5048 | 0.8578 / 0.5378 | +2.42pp / +3.30pp |

This is almost the same gain scale as the old source-only comparison.

## Matched Direct Fine-Tune Comparison

The strict 500Hz matched direct comparison is different:

| method | four-center mean AUROC | four-center mean AUPRC | delta |
|---|---:|---:|---:|
| direct K500 fine-tune | 0.8601 | 0.5416 | reference |
| VAE500 LHAT | 0.8578 | 0.5378 | -0.23pp / -0.38pp |

Scaling target-center samples also did not make VAE beat direct:

| setting | VAE - direct |
|---|---:|
| K1000 | +0.03pp / +0.16pp |
| 10% target-ratio | +0.06pp / +0.16pp |
| 20% target-ratio | -0.08pp / -0.17pp |

## Low-Capacity LoRA Diagnostic

为了排除“full direct fine-tune 容量太强，所以盖住了 VAE 分支”的解释，
又跑了一组低容量诊断：

```text
freeze EfficientNet1DV2 backbone
只训练 final classifier 的 LoRA residual adapter
direct-LoRA vs VAE500 LoRA-LHAT
同一批 K500 anchors，同一套 v7 500Hz ref-excluded evaluation
```

结果：

| center | direct-LoRA | VAE500 LoRA-LHAT | delta |
|---|---:|---:|---:|
| ningbo | 0.8591 / 0.4644 | 0.8589 / 0.4643 | -0.01pp / -0.02pp |
| chapman_shaoxing | 0.8613 / 0.4193 | 0.8612 / 0.4192 | -0.01pp / -0.01pp |
| cpsc_2018 | 0.8212 / 0.5537 | 0.8221 / 0.5541 | +0.09pp / +0.04pp |
| georgia | 0.8185 / 0.5965 | 0.8185 / 0.5965 | +0.00pp / -0.00pp |
| mean | 0.8400 / 0.5085 | 0.8402 / 0.5085 | +0.02pp / +0.00pp |

这个结果说明：即使把直接微调容量压低，当前 VAE500 latent-hull 分支也几乎
没有额外贡献。因此问题不只是 full direct FT 太强，也包括当前 VAE500
online-AT 机制没有给分类器提供有用的新边界信息。

## Non-Comparable Changes Between Old And New Runs

The old and new runs changed several variables at once.

| factor | old positive route | new 500Hz route |
|---|---|---|
| VAE | original ECGTwin VAE | newly trained PTB-XL-only VAE500 |
| latent shape | `(N, 4, 128)` | `(N, 4, 625)` |
| classifier input | 100Hz, 1000 samples | 500Hz, 5000 samples |
| label mapping | v3 super5 | v7 clinician-review mapping |
| comparison often reported | source-only baseline | matched direct target FT |
| target real weight | 40.0 | 20.0 in main 500Hz K500 |
| roundtrip anchor | 1500 samples, weight 0.5 | disabled |
| trusted attack classes | mostly NORM/MI/STTC | all five classes |
| quality gate | enabled in old strong run | disabled in new main matched runs |
| `hull_lambda` | 0.15 | 0.05 |
| `K_anchor` | 300 | 128 |
| adv label | mixed soft, teacher mix 0.3 | multi-hot hard |

Therefore the observed difference cannot be attributed to sampling rate alone.

## VAE500 Quality Check

The accepted VAE500 checkpoint passed reconstruction audit:

| split | invalid | global Pearson | first-diff Pearson | leads >=0.90 | p2p median |
|---|---:|---:|---:|---:|---:|
| fold9 val | 0.0 | 0.9976 | 0.9494 | 12/12 | 0.9925 |
| fold10 audit | 0.0 | 0.9977 | 0.9505 | 12/12 | 0.9929 |

This weakens the hypothesis that VAE500 simply cannot reconstruct ECG.
The more likely issue is latent-space utility for adversarial training:
the 500Hz latent has good reconstruction quality but may not expose a stronger
semantic decision-boundary direction than direct target fine-tuning already
learns.

训练日志也支持这个判断。K500 plain VAE500 和 LoRA-VAE500 的 decoded invalid
rate 都是 0；`atk_anchor` 大多在 0.7-0.8 左右，说明攻击流程不是完全失效。
但 `loss_gain` 只有大约 `3e-4` 到 `5e-4`，幅度太小，分类器训练几乎感受不到
有效的 adversarial pressure。这更像是 latent-hull 方向太保守/太局部，或者
500Hz latent 的几何结构更偏重重建而不是可迁移的诊断边界。

## Anchor Label Distribution Drift

The v7 mapping changes the target anchor label distribution. Example K500:

| center | VAE500 CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|
| ningbo | 102 | 130 | 5 | 120 | 278 |
| chapman_shaoxing | 105 | 120 | 4 | 119 | 253 |
| cpsc_2018 | 295 | 0 | 0 | 97 | 110 |
| georgia | 128 | 121 | 1 | 101 | 300 |

In several centers MI is almost absent after v7 mapping. Old v3 Ningbo K500 had
34 MI-positive anchors. This makes five-class same-label latent search much less
balanced and can reduce the value of the adversarial branch, especially for rare
class macro AUPRC.

## Current Hypotheses

1. **Baseline changed.**
   VAE500 still improves over source-only by about `+2.42pp / +3.30pp`.
   It fails because matched direct K500 fine-tune already captures most
   target-center signal.

2. **Current VAE500 adversarial pressure is too weak.**
   Low-capacity LoRA diagnostic still shows only `+0.02pp / +0.00pp` mean gain,
   and training logs show tiny `loss_gain`. The VAE branch is valid but not
   producing useful additional boundary movement.

3. **Recipe mismatch.**
   The strong old route used heavier target-real supervision, roundtrip anchors,
   mixed-soft labels, larger `hull_lambda`, larger `K_anchor`, and trusted
   NORM/MI/STTC scope. The 500Hz route is more conservative and different.

4. **Latent geometry changed.**
   VAE500 reconstructs well, but its `(4,625)` latent is much higher-dimensional.
   The old `(4,128)` ECGTwin latent may have been smoother and more useful for
   local semantic interpolation. Reconstruction quality alone does not prove
   adversarial latent utility.

5. **v7 label distribution changed the problem.**
   MI anchors are extremely sparse in v7 K500 for several centers. This hurts
   same-label latent-hull diversity and macro metric gains.

## Immediate Next Checks

1. Keep the current EfficientNet-first strategy.
2. Run one 500Hz recipe closer to the old strong route before blaming 500Hz:
   `target_real_weight=40`, `hull_lambda=0.15`, `K_anchor=300`,
   `mixed_soft`, `teacher_mix=0.3`, trusted `NORM/MI/STTC`, and optional
   roundtrip anchor if VAE500 support is correct.
3. If that still fails over matched direct, the likely conclusion is that
   VAE500's latent manifold is reconstructive but not better than direct
   supervised target adaptation for this setting.
