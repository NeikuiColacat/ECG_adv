# ECGFounder 官方权重 zero-shot vs VAE-only 在线对抗训练

日期：2026-05-21

## 评测口径

ECGFounder 使用官方 12-lead checkpoint：

```text
/root/autodl-tmp/ecgfounder/checkpoint/12_lead_ECGFounder.pth
```

本次没有微调 ECGFounder，也没有训练新的 Super5 分类头。评测脚本直接使用 ECGFounder 官方 150 类输出，通过关键词池映射到本项目 Super5：

```text
CD, HYP, MI, NORM, STTC
```

运行命令：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/eval_ecgfounder_super5_zero_shot_20260517.py \
  --out_dir /root/autodl-tmp/paper_foundation_baselines_20260521/ecgfounder_super5_zeroshot_refcheck \
  --batch_size 128 \
  --num_workers 4
```

输出文件：

```text
/root/autodl-tmp/paper_foundation_baselines_20260521/ecgfounder_super5_zeroshot_refcheck/ecgfounder_super5_zeroshot_ref_excluded.json
/root/autodl-tmp/paper_foundation_baselines_20260521/ecgfounder_super5_zeroshot_refcheck/ecgfounder_super5_zeroshot_ref_excluded.csv
```

所有目标中心评估均排除 K=500 target refs，和我们的 VAE-only 在线对抗训练保持同一测试口径。

## Target-Center 对比

表中 `ECGFounder - VAE-only` 是 ECGFounder 官方权重 zero-shot 相对我们 VAE-only 在线对抗训练的差值，单位为 percentage points。

| center | effective_n | PTB-XL EfficientNet baseline | VAE-only 在线对抗训练 | ECGFounder zero-shot | ECGFounder - VAE-only |
|---|---:|---:|---:|---:|---:|
| ningbo | 34405 | 0.8672 / 0.4784 | 0.8918 / 0.5231 | 0.8731 / 0.4992 | -1.87pp / -2.39pp |
| chapman_shaoxing | 9747 | 0.8763 / 0.4251 | 0.8967 / 0.4634 | 0.8717 / 0.4781 | -2.50pp / +1.47pp |
| cpsc_2018 | 6377 | 0.8115 / 0.5586 | 0.8552 / 0.5973 | 0.8121 / 0.5802 | -4.31pp / -1.71pp |
| georgia | 9844 | 0.8157 / 0.5916 | 0.8259 / 0.6026 | 0.8434 / 0.6604 | +1.75pp / +5.78pp |

四中心平均：

| method | avg AUROC / AUPRC | vs baseline |
|---|---:|---:|
| PTB-XL EfficientNet baseline | 0.8427 / 0.5134 | 0.00pp / 0.00pp |
| VAE-only 在线对抗训练 | 0.8674 / 0.5466 | +2.47pp / +3.32pp |
| ECGFounder zero-shot | 0.8501 / 0.5545 | +0.74pp / +4.10pp |

## 结论

ECGFounder 官方原始权重 zero-shot 在四中心平均 AUPRC 上略高于我们的 VAE-only 在线对抗训练，但 AUROC 平均值低于我们：

```text
ECGFounder zero-shot - VAE-only 在线对抗训练
= -1.73pp AUROC / +0.79pp AUPRC
```

分中心看，ECGFounder 在 georgia 上明显更强，chapman_shaoxing 的 AUPRC 也更高；但在 ningbo 和 cpsc_2018 上，我们的 VAE-only 在线对抗训练同时取得更高 AUROC 和 AUPRC。

需要注意：ECGFounder zero-shot 是官方 150 类任务输出到 Super5 的关键词映射，不是为本项目 Super5 重新训练的分类头。因此它适合作为“官方权重开箱即用”对比，不应和 ECGFounder K-shot head fine-tuning 混为同一种实验。
