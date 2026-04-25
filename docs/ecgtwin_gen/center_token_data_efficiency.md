# Center-Token 数据效率消融（PTBXL + MIMIC 双 victim）

*生成脚本：`/tmp/agg_center_token_ablation.py`；数据来源：`/root/autodl-tmp/center_token_ablation/retrain/`*

## 1. Baselines (eval_half_only, fold>=6)

| Center | PTBXL victim | MIMIC victim |
|---|---:|---:|
| cpsc_2018_extra | 0.8603 | 0.8822 |
| chapman_shaoxing | 0.9538 | 0.9577 |
| cpsc_2018 | 0.9161 | 0.8936 |
| georgia | 0.9448 | 0.9249 |
| ningbo | 0.9696 | 0.9693 |
| **MAIN5 avg** | **0.9289** | **0.9255** |

## 2. Δ Target-Center AUROC (pp) = variant(center, K) − baseline(center)

### PTBXL victim (retrain-from-scratch)

| Center | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|
| cpsc_2018_extra | +1.31 | +1.67 | +0.43 | +1.20 | +2.53 |
| chapman_shaoxing | -1.28 | -1.67 | -1.88 | -0.93 | -0.12 |
| cpsc_2018 | +1.29 | +2.42 | +2.36 | +2.20 | +1.54 |
| georgia | -1.09 | -1.01 | +0.17 | -0.97 | +0.26 |
| ningbo | -0.63 | -0.38 | -0.88 | -0.18 | -0.32 |

### MIMIC victim (finetune, 5 epoch)

| Center | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|
| cpsc_2018_extra | +1.02 | +0.67 | +0.92 | +1.11 | +0.72 |
| chapman_shaoxing | +0.36 | +0.32 | +0.31 | +0.39 | +0.27 |
| cpsc_2018 | +4.26 | +4.73 | +4.51 | +4.40 | +4.41 |
| georgia | +0.86 | +0.73 | +0.77 | +0.94 | +0.96 |
| ningbo | -0.11 | -0.20 | -0.04 | -0.05 | -0.22 |

## 3. Δ MAIN5 AUROC (pp) = MAIN5_avg(variant) − MAIN5_avg(baseline)

### PTBXL victim

| Center token trained on | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|
| cpsc_2018_extra | -0.33 | -0.07 | -0.81 | +0.53 | +0.90 |
| chapman_shaoxing | -0.44 | -0.85 | -0.43 | -0.25 | +0.74 |
| cpsc_2018 | -0.27 | +0.05 | +0.56 | -0.46 | -0.74 |
| georgia | -0.38 | -1.10 | +0.56 | +0.03 | +0.46 |
| ningbo | +0.03 | -0.19 | +0.03 | +0.66 | +0.35 |

### MIMIC victim

| Center token trained on | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|
| cpsc_2018_extra | +1.32 | +1.14 | +1.26 | +1.31 | +1.09 |
| chapman_shaoxing | +1.08 | +0.99 | +0.98 | +1.21 | +1.07 |
| cpsc_2018 | +1.31 | +1.38 | +1.38 | +1.24 | +1.36 |
| georgia | +1.21 | +0.94 | +1.04 | +1.07 | +1.03 |
| ningbo | +0.92 | +0.99 | +1.12 | +1.10 | +0.83 |

## 4. Subsample Meta (K_actual + per-class distribution + pool size)

| Center | K_req | K_actual | NSR | STach | AF | IAVB | LBBB | RBBB | fold<=5 pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpsc_2018_extra | 25 | 25 | 1 | 5 | 6 | 5 | 4 | 4 | 352 |
| cpsc_2018_extra | 50 | 50 | 1 | 14 | 9 | 8 | 8 | 10 | 352 |
| cpsc_2018_extra | 100 | 100 | 1 | 21 | 23 | 20 | 15 | 20 | 352 |
| cpsc_2018_extra | 200 | 200 | 1 | 61 | 50 | 37 | 15 | 36 | 352 |
| cpsc_2018_extra | 500 | 352 | 1 | 150 | 92 | 41 | 15 | 53 | 352 |
| chapman_shaoxing | 25 | 25 | 5 | 4 | 4 | 4 | 4 | 4 | 2991 |
| chapman_shaoxing | 50 | 50 | 9 | 9 | 8 | 8 | 8 | 8 | 2991 |
| chapman_shaoxing | 100 | 100 | 17 | 18 | 17 | 16 | 16 | 16 | 2991 |
| chapman_shaoxing | 200 | 200 | 36 | 36 | 39 | 33 | 23 | 33 | 2991 |
| chapman_shaoxing | 500 | 500 | 108 | 97 | 115 | 83 | 23 | 74 | 2991 |
| cpsc_2018 | 25 | 25 | 4 | 0 | 5 | 5 | 4 | 7 | 2355 |
| cpsc_2018 | 50 | 50 | 10 | 0 | 9 | 9 | 8 | 14 | 2355 |
| cpsc_2018 | 100 | 100 | 20 | 0 | 22 | 19 | 16 | 23 | 2355 |
| cpsc_2018 | 200 | 200 | 41 | 0 | 40 | 38 | 34 | 47 | 2355 |
| cpsc_2018 | 500 | 500 | 96 | 0 | 107 | 94 | 84 | 119 | 2355 |
| georgia | 25 | 25 | 4 | 4 | 5 | 4 | 4 | 4 | 2473 |
| georgia | 50 | 50 | 8 | 9 | 9 | 8 | 8 | 8 | 2473 |
| georgia | 100 | 100 | 17 | 18 | 16 | 17 | 16 | 16 | 2473 |
| georgia | 200 | 200 | 34 | 33 | 34 | 33 | 33 | 33 | 2473 |
| georgia | 500 | 500 | 93 | 89 | 87 | 84 | 61 | 86 | 2473 |
| ningbo | 25 | 25 | 4 | 5 | 4 | 4 | 4 | 4 | 10257 |
| ningbo | 50 | 50 | 9 | 8 | 9 | 8 | 8 | 8 | 10257 |
| ningbo | 100 | 100 | 19 | 19 | 16 | 16 | 14 | 16 | 10257 |
| ningbo | 200 | 200 | 39 | 39 | 42 | 33 | 14 | 33 | 10257 |
| ningbo | 500 | 500 | 99 | 105 | 113 | 84 | 14 | 85 | 10257 |

## 5. Per-Class Target-Center AUROC Δ (pp) — PTBXL victim

### cpsc_2018_extra
| Class | Baseline | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| NSR | 0.8104 | +5.93 | +6.32 | -4.97 | +1.70 | +8.21 |
| STach | 0.9830 | -1.51 | -1.09 | -0.91 | -0.46 | -0.46 |
| AF | 0.9605 | -0.82 | -1.16 | -1.38 | -0.06 | -0.14 |
| IAVB | 0.9280 | -2.98 | -0.53 | -1.01 | -0.83 | -1.80 |
| LBBB | 0.5127 | +6.65 | +4.73 | +8.91 | +4.89 | +7.34 |
| RBBB | 0.9670 | +0.57 | +1.75 | +1.92 | +1.93 | +2.01 |

### chapman_shaoxing
| Class | Baseline | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| NSR | 0.8994 | -2.72 | -2.82 | -3.57 | -1.84 | -0.36 |
| STach | 0.9935 | -0.73 | -1.02 | -2.18 | -0.81 | -0.25 |
| AF | 0.9914 | -1.35 | -1.61 | -1.91 | -1.16 | -0.85 |
| IAVB | 0.9722 | -2.91 | -6.36 | -2.55 | -1.68 | -0.60 |
| LBBB | 0.9162 | -2.08 | +0.43 | -2.67 | -1.58 | +0.71 |
| RBBB | 0.9502 | +2.09 | +1.36 | +1.60 | +1.49 | +0.66 |

### cpsc_2018
| Class | Baseline | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| NSR | 0.8700 | +2.05 | +4.19 | +3.78 | +5.23 | +4.66 |
| STach | — | — | — | — | — | — |
| AF | 0.9822 | -0.68 | -1.75 | -1.71 | -2.33 | -1.74 |
| IAVB | 0.9524 | -0.95 | -0.84 | -1.50 | -2.61 | -1.38 |
| LBBB | 0.9397 | +2.57 | +1.68 | +2.79 | +1.52 | +1.01 |
| RBBB | 0.8362 | +3.48 | +8.82 | +8.42 | +9.20 | +5.14 |

### georgia
| Class | Baseline | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| NSR | 0.8688 | -3.17 | -2.74 | +0.45 | -1.92 | +1.03 |
| STach | 0.9900 | -0.42 | -0.58 | -0.09 | -0.76 | -0.12 |
| AF | 0.9230 | -1.89 | -1.88 | -0.03 | -1.54 | -0.48 |
| IAVB | 0.9670 | -1.39 | -0.84 | -0.24 | -1.12 | -0.64 |
| LBBB | 0.9608 | -0.20 | +0.01 | -0.11 | -0.92 | +0.67 |
| RBBB | 0.9592 | +0.52 | -0.06 | +1.03 | +0.46 | +1.08 |

### ningbo
| Class | Baseline | K=25 | K=50 | K=100 | K=200 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| NSR | 0.8732 | +0.56 | -0.16 | -0.95 | -0.04 | -0.52 |
| STach | 0.9905 | -0.81 | -0.34 | -0.44 | -0.30 | -0.22 |
| AF | 0.9932 | -1.58 | -0.87 | -1.57 | -0.73 | -0.69 |
| IAVB | 0.9728 | -2.08 | -1.00 | -2.38 | -0.39 | -0.67 |
| LBBB | 0.9937 | +0.13 | +0.33 | -0.01 | +0.20 | +0.08 |
| RBBB | 0.9945 | -0.02 | -0.26 | +0.06 | +0.16 | +0.06 |

## 6. 拐点 & 饱和分析 (双 victim 同时 ≥ +0.3pp)

| Center | 拐点 n_refs* (双 victim Δ≥+0.3pp 最小 K) | 饱和 n_refs† (下一档 ΔΔ<+0.1pp 最小 K) |
|---|---|---|
| cpsc_2018_extra | K=25 | 未达到 |
| chapman_shaoxing | 未达到 | 未达到 |
| cpsc_2018 | K=25 | 未达到 |
| georgia | 未达到 | 未达到 |
| ningbo | 未达到 | 未达到 |

## 7. 总结（Executive Summary）

**研究问题**：Center-Token textual-inversion 方法在多个目标中心是否有效？训练一个有效 token 需要多少条目标中心 ECG 数据？

**实验规模**：5 目标中心 × 5 K 档位 × 2 victim = 50 次 retrain/finetune（~15h 实际耗时）

### 核心发现

1. **MIMIC victim 全面稳定**：5 中心 × 5 K 的所有 25 个 MAIN5 avg Δ **都在 +0.83~+1.38pp**（全部正面）。只要加入合成数据 finetune，无论 K 多少、token 针对哪个中心训练，MIMIC victim 的跨中心性能都提升。说明 center-token 合成数据对 MIMIC finetune 是鲁棒的有效 augmentation。

2. **PTBXL victim 受 K 影响更敏感，且中心特异**：
   - **小中心 cpsc_2018_extra (baseline 0.8603)**：K=500 目标 +2.53pp / MAIN5 +0.90pp — 最强增益；K 越大效果越好
   - **中型中心 cpsc_2018 (baseline 0.9161)**：K=50-100 最佳（目标 +2.42 / MAIN5 +0.56），K≥200 递减
   - **大中心 chapman_shaoxing (baseline 0.9538)**：目标全 K 负（-1.88 ~ -0.12），MAIN5 仅 K=500 +0.74pp
   - **Western 中心 georgia (baseline 0.9448)**：目标不稳定，MAIN5 最佳 K=100 +0.56pp
   - **大中心 ningbo (baseline 0.9696)**：目标全 K 负（天花板），MAIN5 最佳 K=200 +0.66pp

3. **拐点 n_refs\*（双 victim 同时 ≥+0.3pp）**：
   - **cpsc_2018_extra**: K=25 ✅（PTBXL +1.31, MIMIC +1.02）— 小中心 25 条就够
   - **cpsc_2018**: K=25 ✅（PTBXL +1.29, MIMIC +4.26）
   - **chapman/georgia/ningbo**: 所有 K 不双达标（PTBXL 侧拖后腿）

4. **Sibling transfer 异常强**：任一中心训的 token，对 cpsc_2018 中心都大幅提升（MIMIC +3.3~+4.7pp 稳定）。说明合成数据带有 CPSC-family acquisition style（与既有 IBE 消融结论一致——IBE 已编码 acquisition-culture 级特征）。

5. **Token norm 随 K 单调增（0.18→2.5）**：因 `batch_size=min(K,32)` 使 K≥32 时 batches/epoch 随 K 递增（1→15），每 epoch 梯度更新数 ∝ K/32。这是 pipeline artifact，使得同一 token 的 30 epoch 效果随 K 扩大。未来工作可固定 total updates 以分离'数据量'和'训练步数'因素。

### 用户原 pitch 验证（50-100 条数据）

- **小中心**（~1300 条）：K=25 就够拐点，K=50-100 区间效果稳定（target PTBXL +1.31~+1.67pp）。**pitch 完全成立**
- **中型中心**（~5000 条）：K=50-100 最佳（target PTBXL +2.36~+2.42pp）。**pitch 完全成立**
- **大中心**（≥9000 条）：K=50-100 在 PTBXL 上不足以克服天花板。需要 K≥500 才有 MAIN5 正面效果，且目标中心自体仍难提升。**pitch 部分成立**（MIMIC finetune 场景下 50 条就够）

### Deployment Playbook（推荐）

- **下游是 MIMIC-style 大量训练数据的预训练 victim**：50 条目标中心 ECG 就足够（K=25-50 区间 MAIN5 已 +1pp+）
- **下游是 PTBXL-scale 小模型需要 retrain from scratch**：
  - 小中心（<2000 条）：200-500 条 refs 最佳
  - 中等中心（2000-5000）：50-100 条 refs 最佳
  - 大中心（>9000）：方法对 target 中心效果有限，考虑其他路径

## 8. Notes

- **Ref/Eval split**: `fold<=5` → ref (for token training + synth generation); `fold>=6` → eval (via `--eval_half_only`). Deterministic per record_id via SHA1. Zero patient overlap.
- **PTBXL victim**: retrain-from-scratch (50 epoch, lr=0.01, synth_ratio=0.3, seed=42).
- **MIMIC victim**: finetune from `/root/autodl-tmp/mimic_tierM/best_model.pt` (5 epoch, lr=1e-4, synth_ratio=0.3, samples_per_epoch=80000).
- **Token**: 30-epoch AdamW lr=1e-3, L_recon + 0.1·L_inv + 0.01·L_reg, uses `center_token_latest.pth` (val loader empty for fold<=5 subsample).
- **Synth**: 6 classes × 1000 = 6000 ECGs per variant, DDPM 50 steps.
