# VAE500 + AugMix Stage B 结果记录

日期：2026-06-01

## 目的

在严格 matched direct K500 fine-tune 之上，测试低强度 VAE latent online AT
和 AugMix-style latent branch 是否能带来额外 AUROC/AUPRC 增益。

固定协议：

```text
backbone: EfficientNet1DV2 500Hz
mapping: v7_super5_sjr_rgq_review_20260528
input: minimal_resample + per_sample_global + 500Hz + 5000 samples
VAE: repo-owned PTB-XL-only VAE500
K: 500 target-center anchors
final eval: K500 ref ids excluded
selection: K500 internal validation target_macro_auprc
```

## Stage B 设置

相对 Stage A，Stage B 降低对抗压力，并比较三种标签策略：

```text
adv_weight = 0.10
hull_lambda = 0.05
hull_M = 20
hull_steps = 3
latent_augmix_severity = 4
latent_augmix_latent_weight_cap = 0.15
latent_augmix_ops = powerline_noise, emg_noise, baseline_wander, baseline_shift
label modes = multi_hot_hard, latent_mixed_teacher, teacher_soft
```

## 结果

| center | method | AUROC | AUPRC | delta AUROC vs direct | delta AUPRC vs direct | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpsc_2018 | hard | 0.8499 | 0.5952 | -0.34pp | -0.60pp | 0.9118 | 0.7829 |
| cpsc_2018 | latent_mixed_teacher | 0.8499 | 0.5950 | -0.34pp | -0.61pp | 0.9118 | 0.7830 |
| cpsc_2018 | teacher_soft | 0.8499 | 0.5950 | -0.34pp | -0.62pp | 0.9118 | 0.7830 |
| chapman_shaoxing | hard | 0.8731 | 0.4450 | +0.06pp | +0.02pp | 0.9116 | 0.7820 |
| chapman_shaoxing | latent_mixed_teacher | 0.8733 | 0.4450 | +0.08pp | +0.02pp | 0.9116 | 0.7820 |
| chapman_shaoxing | teacher_soft | 0.8735 | 0.4453 | +0.10pp | +0.05pp | 0.9115 | 0.7819 |

Matched direct FT references:

```text
cpsc_2018:          0.8533 / 0.6011
chapman_shaoxing:   0.8725 / 0.4448
```

## 结论

Stage B 没有达到 goal 标准。CPSC 三组均低于 direct FT；Chapman 三组略高于
direct FT，但最大只有 `+0.10pp AUROC / +0.05pp AUPRC`，不足以支持
“VAE/AugMix 相对 direct FT 额外 +2pp”的主张。

当前证据说明：

```text
继续扩大 AugMix severity 或简单切换 hard/soft label，不太可能解决问题。
下一步应转向 boundary/uncertainty-targeted VAE AT：
只对 direct FT 下高 BCE、低 margin、接近决策边界的目标中心 anchors 施加 VAE latent 对抗训练。
```

实验根目录：

```text
/root/autodl-tmp/vae500_augmix_stageB_v7_20260601_r2/
```
