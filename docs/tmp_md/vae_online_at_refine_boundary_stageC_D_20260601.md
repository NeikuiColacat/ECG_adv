# VAE500 Boundary-Targeted Online AT Stage C/D 结果记录

日期：2026-06-01

## 目的

Stage A/B 显示 generic VAE/AugMix online AT 不能超过 matched direct K500
fine-tune。因此本轮测试 boundary/uncertainty-targeted anchor 选择：

```text
用 matched direct FT checkpoint 给 K500 train anchors 打分
score = positive hard BCE + 0.5 * positive low-margin uncertainty
每个类别只保留 top 45% 高分 anchor 进入 latent-hull 抽样
最终 PN2021 评估仍排除 K500 ref ids
```

本轮同时给训练脚本新增了可复用参数：

```text
--anchor_score_mode {hard_bce, positive_hard_bce, low_margin, hard_bce_plus_low_margin}
--anchor_score_ckpt
--anchor_score_low_margin_weight
--anchor_score_top_frac
--anchor_score_min_per_class
```

实现文件：

```text
scripts/pgd_cross_center/synth_online_at_super5.py
```

验证：

```text
/root/miniforge3/envs/ECGTwin/bin/python -m py_compile scripts/pgd_cross_center/synth_online_at_super5.py
```

## Stage C

配置：

```text
init_ckpt = matched direct FT checkpoint
anchor_score_ckpt = same matched direct FT checkpoint
adv_label_mode = multi_hot_hard
adv_weight = 0.20
target_real_weight = 5
hull_lambda = 0.05
hull_M = 20
hull_steps = 5
lr = 2e-5
```

结果：

| center | AUROC | AUPRC | delta AUROC vs direct | delta AUPRC vs direct | note |
|---|---:|---:|---:|---:|---|
| cpsc_2018 | 0.8533 | 0.6011 | -0.00pp | +0.00pp | K500-val 下降，best 回退 initial direct |
| chapman_shaoxing | 0.8725 | 0.4448 | +0.00pp | -0.00pp | K500-val 下降，best 回退 initial direct |

诊断：

```text
CPSC ASR roughly 0.38-0.54; Chapman ASR roughly 0.51-0.65.
攻击强度健康，但 hard-label adversarial stream 对 K500 internal validation 是负迁移。
```

## Stage D

配置：

```text
init_ckpt = matched direct FT checkpoint
anchor_score_ckpt = same matched direct FT checkpoint
adv_label_mode = teacher_soft
adv_weight = 0.05
target_real_weight = 0
source_logit_anchor_weight = 0.05
source_logit_anchor_batches = 20
hull_lambda = 0.05
hull_M = 20
hull_steps = 5
lr = 1e-5
```

结果：

| center | AUROC | AUPRC | delta AUROC vs direct | delta AUPRC vs direct | note |
|---|---:|---:|---:|---:|---|
| cpsc_2018 | 0.8533 | 0.6011 | -0.00pp | +0.00pp | K500-val 下降，best 回退 initial direct |
| chapman_shaoxing | 0.8725 | 0.4448 | +0.00pp | -0.00pp | K500-val 下降，best 回退 initial direct |

诊断：

```text
更温和的 teacher-soft / low-adv-weight / no-target-real-stream 仍不能提升 K500-val。
这说明问题不是 hard label 或 adv_weight 单独造成，而是当前 direct-init 后的 VAE adversarial stream
没有给 CPSC/Chapman 的 direct K500 模型提供额外泛化信息。
```

## 当前结论

Boundary-targeted anchor 选择机制已经实现并生效，anchor pool 能按 direct FT
高 BCE / 低 margin 样本排序和截断；但在 EfficientNet1DV2 500Hz、K500、
direct-init 条件下，Stage C/D 均没有超过 direct FT。

下一步不建议继续围绕同一个 EfficientNet K500 direct-init 配方做小幅 hard/soft
label 或 adv_weight 微调。更有价值的方向：

```text
1. 做 K / target-ratio 实验：K=1000, 10%, 20%，确认 direct FT 是否更强或 VAE 是否在不同数据量下更有价值。
2. 做 benchmark backbone transfer：优先 fastai_xresnet1d50，因为它是当前 500Hz source benchmark 最强模型。
3. 若继续做 EfficientNet VAE，改成 candidate/selector 或 classwise refinement，而不是单一全模型 checkpoint。
```

实验根目录：

```text
/root/autodl-tmp/vae500_boundary_stageC_v7_20260601/
/root/autodl-tmp/vae500_boundary_stageD_v7_20260601/
```
