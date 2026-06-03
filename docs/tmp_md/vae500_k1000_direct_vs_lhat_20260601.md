# VAE500 K1000 Direct FT vs Plain LHAT 结果记录

日期：2026-06-01

## 目的

在 K=500 之外测试 K=1000 目标中心样本规模：

```text
PTB-XL 500Hz EfficientNet1DV2 source checkpoint
-> K1000 target-center matched direct fine-tune
-> K1000 target-center VAE500 latent-hull online AT
-> PN2021 target-center held-out evaluation with the K1000 ref ids excluded
```

固定协议：

```text
mapping: v7_super5_sjr_rgq_review_20260528
input: minimal_resample + per_sample_global + 500Hz + 5000 samples
VAE: PTB-XL-only VAE500, diffusets500_v1
selection: K1000 internal validation target_macro_auprc
```

## K1000 Anchor 缓存

已导出四中心 K1000 real-anchor 缓存：

```text
/root/autodl-tmp/vae500_lhat_v7/anchors_k1000/<center>/<center>_real_k1000_seed42_vae500.*
```

验证：

| center | K requested | K kept | latent shape |
|---|---:|---:|---|
| ningbo | 1000 | 1000 | 4 x 625 |
| chapman_shaoxing | 1000 | 1000 | 4 x 625 |
| cpsc_2018 | 1000 | 1000 | 4 x 625 |
| georgia | 1000 | 1000 | 4 x 625 |

总缓存大小约 322MB。

## K1000 Direct FT

输出：

```text
/root/autodl-tmp/vae500_direct_ft_k1000_v7_20260601/
```

| center | K1000 direct AUROC / AUPRC | delta vs K500 direct | PTB-XL fold10 | excluded refs |
|---|---:|---:|---:|---:|
| ningbo | 0.8849 / 0.5053 | +0.51pp / +0.30pp | 0.9066 / 0.7698 | 1000 |
| chapman_shaoxing | 0.8758 / 0.4476 | +0.33pp / +0.28pp | 0.9080 / 0.7705 | 1000 |
| cpsc_2018 | 0.8770 / 0.6328 | +2.37pp / +3.17pp | 0.9065 / 0.7738 | 1000 |
| georgia | 0.8396 / 0.6215 | +0.49pp / +0.35pp | 0.9062 / 0.7655 | 1000 |
| mean | 0.8693 / 0.5518 | +0.93pp / +1.02pp | - | - |

结论：

```text
K 从 500 增加到 1000 后，direct FT 平均有小到中等提升，主要来自 CPSC。
因此 K1000 下 VAE AT 的有效性必须与更强的 K1000 direct FT 对照比较。
```

## K1000 Plain VAE500 LHAT

输出：

```text
/root/autodl-tmp/vae500_lhat_k1000_v7_20260601/
```

VAE 配方与 K500 plain VAE 保持一致：

```text
target_real_weight = 20
adv_weight = 0.30
hull_lambda = 0.05
hull_M = 20
hull_steps = 3
adv_label_mode = multi_hot_hard
classes_in_scope = CD,HYP,MI,NORM,STTC
```

结果：

| center | K1000 direct | K1000 VAE LHAT | delta VAE - direct | PTB-XL fold10 after VAE |
|---|---:|---:|---:|---:|
| ningbo | 0.8849 / 0.5053 | 0.8872 / 0.5104 | +0.23pp / +0.51pp | 0.9057 / 0.7710 |
| chapman_shaoxing | 0.8758 / 0.4476 | 0.8748 / 0.4489 | -0.10pp / +0.13pp | 0.9077 / 0.7697 |
| cpsc_2018 | 0.8770 / 0.6328 | 0.8790 / 0.6364 | +0.19pp / +0.37pp | 0.9059 / 0.7722 |
| georgia | 0.8396 / 0.6215 | 0.8377 / 0.6179 | -0.19pp / -0.36pp | 0.9072 / 0.7667 |
| mean | 0.8693 / 0.5518 | 0.8697 / 0.5534 | +0.03pp / +0.16pp | - |

结论：

```text
K1000 plain VAE500 LHAT 比 K1000 direct FT 平均只多 +0.03pp AUROC / +0.16pp AUPRC。
Ningbo 和 CPSC 小幅正向，Chapman AUPRC 小涨但 AUROC 小降，Georgia 负迁移。
这说明 K1000 下 plain LHAT 不是突破配方，但它比 K500 direct-init boundary C/D 更稳定。
```

## 10% / 20% Target-Ratio 规模

按每个中心 Super5 至少一类阳性的有效样本数计算：

| center | total records | nonzero Super5 records | 10% K | 20% K |
|---|---:|---:|---:|---:|
| ningbo | 34905 | 19228 | 1923 | 3846 |
| chapman_shaoxing | 10247 | 5822 | 582 | 1164 |
| cpsc_2018 | 6877 | 4746 | 475 | 949 |
| georgia | 10344 | 8711 | 871 | 1742 |

建议下一步：

```text
优先跑 10% ratio，因为它覆盖 CPSC≈K500、Chapman≈K500、Georgia≈K1000、
Ningbo≈K2000，可以测试“按中心规模比例取样”是否比固定 K 更合理。
20% ratio 训练成本和 ref-exclusion 影响更大，应在 10% 有信号后再跑。
```
