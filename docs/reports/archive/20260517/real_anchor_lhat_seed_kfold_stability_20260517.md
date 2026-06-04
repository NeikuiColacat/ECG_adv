# Real-Anchor LH-AT 多 Seed K-Fold 稳定性复现实验

日期：2026-05-17

## 目的

验证当前论文主线方法 `real-anchor VAE latent-hull online adversarial training`
在 PN2021 多个目标中心上是否能稳定复现增益，而不是依赖某一次目标中心
K=500 抽样或随机种子。

## 方法配置

- baseline：PTB-XL super5 EfficientNet1DV2，`minimal_resample + 100Hz + 10s + per_sample_global`。
- 目标中心：`ningbo`、`chapman_shaoxing`、`cpsc_2018`、`georgia`。
- 每个中心使用已有 K=500 target-anchor pool 做 5-fold。
- 随机种子：`20260531`、`20260541`、`20260551`。
- 每个中心共 `3 seeds x 5 folds = 15` 次；四个中心共 60 次。
- 每次训练只使用该 fold 的 train target anchors，held-out fold 只用于目标中心评测。
- 评测通过 `--include_record_ids` 限定 held-out fold，同时通过 `--exclude_ref_ids` 排除训练 anchor IDs。
- LH-AT 参数：`M=20`、`lambda=0.15`、`hull_steps=5`、`hull_lr=0.25`、`epochs=10`。
- 质量 gate：关闭。
- 合成样本：不使用 DiT synthetic；这是纯 real-anchor ECGTwin VAE latent manifold 方法。

## 核心公式

```text
z_adv = (1 - lambda) * z0 + lambda * sum_i softmax(a_i) * z_i
```

其中 `z0` 是一个目标中心真实 ECG 的 ECGTwin VAE latent，`z_i` 是同标签集合内
其他目标中心真实 ECG 的 latent，`a_i` 是在线优化的组合权重。训练时优化 `a_i`
让组合后的 ECG 更接近模型决策边界，然后用该样本做对抗训练。

## 汇总结果

单位为绝对指标差值，括号内为百分点 pp。

| center | baseline AUROC/AUPRC | LH-AT AUROC/AUPRC | delta AUROC mean +/- std | delta AUPRC mean +/- std | positive folds |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8653 / 0.7724 | 0.8975 / 0.8157 | +0.0322 +/- 0.0067 (+3.22pp) | +0.0433 +/- 0.0128 (+4.33pp) | 15/15 AUROC, 15/15 AUPRC |
| chapman_shaoxing | 0.8765 / 0.7534 | 0.8970 / 0.7914 | +0.0205 +/- 0.0064 (+2.05pp) | +0.0380 +/- 0.0095 (+3.80pp) | 15/15 AUROC, 15/15 AUPRC |
| cpsc_2018 | 0.8694 / 0.7473 | 0.9085 / 0.8082 | +0.0391 +/- 0.0085 (+3.91pp) | +0.0609 +/- 0.0170 (+6.09pp) | 15/15 AUROC, 15/15 AUPRC |
| georgia | 0.8370 / 0.7154 | 0.8486 / 0.7236 | +0.0116 +/- 0.0054 (+1.16pp) | +0.0083 +/- 0.0107 (+0.83pp) | 14/15 AUROC, 13/15 AUPRC |

## 解读

1. `ningbo`、`chapman_shaoxing`、`cpsc_2018` 的增益非常稳定，所有 fold 在 AUROC
   和 AUPRC 上均为正。
2. `georgia` 平均仍为正，但稳定性明显弱一些。原因很可能是该 K=500 anchor pool
   中 MI 等类别非常稀疏，某些 fold 的 held-out 集只有约 100 条，单个类别分布波动
   会放大 AUROC/AUPRC 方差。
3. 这轮实验支持论文主张：使用少量目标中心真实 ECG 的 ECGTwin VAE latent manifold
   做 real-anchor latent-hull AT，在多个 PN2021 中心上可以稳定提升跨中心性能。
4. 对 georgia 的表述应保守：平均提升为正，但不是每个 fold 都提升，后续可考虑
   class-balanced anchor 选择或提高 K 来降低稀有类波动。

## 输出文件

- runner：`scripts/paper/run_real_anchor_lhat_seed_kfold_20260517.py`
- CSV：`/root/autodl-tmp/paper_real_anchor_lhat_seed_kfold_20260517/summaries/real_anchor_lhat_seed_kfold.csv`
- Markdown：`/root/autodl-tmp/paper_real_anchor_lhat_seed_kfold_20260517/summaries/real_anchor_lhat_seed_kfold.md`
