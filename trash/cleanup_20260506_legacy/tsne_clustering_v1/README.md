# ECGTwin 中心条件生成 — t-SNE 聚类验证实验

## 假设

AugMix + ECGTwin + AdvDiff 跨中心泛化 pipeline 的前提是：

> **当 ECGTwin 以目标中心的 ECG 作为参考（`ref_latent`）时，生成的 ECG 在特征空间中会聚集到该目标中心的簇上。**

本实验用 PhysioNet 2021 多中心数据 + 独立第三方特征提取器（ECGFounder）来验证这条假设。如果假设不成立，整条 finetune pipeline 的前提就崩了。

## 实验设计

- **数据**：PhysioNet 2021，5 主中心 (chapman_shaoxing / cpsc_2018 / cpsc_2018_extra / georgia / ningbo)
- **真实样本**：每中心 100 条，unified preprocess 到 (12, 1000) @ 100Hz
- **生成样本**：每中心挑 50 条作为 reference latent，用 ECGTwin 生成
- **Embedder**：ECGFounder (10M-ECG 预训练 Net1D) 的 1024-d 池化特征；**独立于** ECGTwin 和 victim EfficientNet
- **可视化**：t-SNE (perplexity=30)
- **Metrics**：
  - **k-NN purity (k=10)** — 每个 gen 在 real 空间中的 k 近邻，多大比例来自其 target 中心（随机基线 = 1/5 = 0.2）
  - **Silhouette** — reals 单独 + reals+gens 合并
  - **MMD (RBF)** — per 中心的 gen(C) vs real(C) 距离，对比 gen(C) vs real(other)

## 结论判定

- `knn_purity_overall > 0.5` 且每个中心 `mmd_same < mmd_other_mean` → **H1 成立**，ref 条件迁移中心风格
- `knn_purity_overall < 0.3` → **H1 拒绝**，pipeline 前提需要重新讨论
- 中间区 → 换 embedder / 扩样本复测

**前提自检**：`silhouette_real > 0.3` 说明 embedder 本身就能区分中心；否则实验无结论，需换 embedder。

## 目录

```
sub_experiment/tsne_clustering_v1/
├── README.md            # 本文
├── config.yaml          # 实验超参
├── data_loader.py       # PN2021 多中心加载
├── embedders.py         # ECGFounder 特征提取
├── generation.py        # ECGTwin 批量生成（含重采样/导联对齐/zscore）
├── analyze.py           # t-SNE + purity/silhouette/MMD + 绘图
├── run_experiment.py    # Stage1→2→3 端到端
└── tests/test_smoke.py  # 小规模 smoke
```

输出（大文件，在 `/root/autodl-tmp/sub_experiment/tsne_clustering_v1/`）：

```
real_signals.pt       # (N_real, 12, 1000) + metadata
real_features.pt      # (N_real, 1024) ECGFounder
gen_signals.pt        # (N_gen, 12, 1000) + target_center_ids
gen_features.pt       # (N_gen, 1024)
tsne_coords.npy       # (N_real+N_gen, 2)
metrics.json          # knn_purity / silhouette / mmd
plots/
  tsne_all.png        # 所有点，中心着色 + real圆/gen三角
  tsne_per_center_*   # 5 张 per-center 高亮图
  purity_bar.png      # k-NN purity 柱状图 + 随机基线
```

## 运行

### smoke（开发期，约 2 分钟）

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
cd /root/ECG_adv_Gen
$PY -m pytest sub_experiment/tsne_clustering_v1/tests/ -v
```

### 正式运行（GPU 30-40 分钟）

```bash
$PY sub_experiment/tsne_clustering_v1/run_experiment.py \
    --config sub_experiment/tsne_clustering_v1/config.yaml
```

中间 `.pt` 会缓存；重跑时若需要强制重建，删掉 `/root/autodl-tmp/sub_experiment/tsne_clustering_v1/*.pt`。

## 结果（待填）

运行后根据 `metrics.json` 填写：

| 中心 | k-NN purity | MMD same | MMD other (mean) |
|---|---|---|---|
| chapman_shaoxing | _ | _ | _ |
| cpsc_2018 | _ | _ | _ |
| cpsc_2018_extra | _ | _ | _ |
| georgia | _ | _ | _ |
| ningbo | _ | _ | _ |

- `knn_purity_overall = _`
- `silhouette_real   = _`
- `silhouette_all    = _`

**结论**： _待填：H1 成立 / H1 拒绝 / 中间区，是否需要复测_

## 非目标

- 不测诊断标签迁移（target_text 效应留后续实验）
- 不与 AugMix / AdvDiff 对比（那是完整 pipeline 消融）
- 不 finetune ECGFounder（零样本使用）
- 不测小中心 (ptb / st_petersburg_incart)
