# 当前方法尝试与 AUROC/AUPRC 提升报告

日期：2026-05-02

本文总结当前 ECG cross-center 增强实验中已经尝试过的方法，以及它们在 clean PN2021 v3 super5 评测和 PN2021-C corrupted 评测上的 AUROC/AUPRC 变化。核心结论是：当前方法主要稳定提升 AUPRC，AUROC 基本保持在同一水平；clean PN2021 最强结果来自 real-anchor Latent-Hull，PN2021-C corrupted absolute AUPRC 最强结果来自 prompt-token v14。

## 评测设置

主评测：

```text
任务：PTB-XL super5 -> PN2021 cross-center generalization
类别：CD, HYP, MI, NORM, STTC
PN2021 label mapping：v3_super5_normsuppress_20260501
clean 评测：PN2021 7-center average，排除 ptb-xl/ptbxl 防止 PTB-XL 泄漏
robustness 评测：PN2021-C 4 centers
  centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
  corruptions = powerline_noise, emg_noise, baseline_wander, baseline_shift, random_leads_masking
  severities = 1..5
```

主要结果文件：

```text
/root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv
/root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
/root/autodl-tmp/triple_labels/pn2021_v3_center_summary.csv
/root/autodl-tmp/triple_labels/pn2021_v3_per_class_summary.csv
```

Baseline：

| model | PTB-XL AUROC | PTB-XL AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|
| EfficientNet1DV2 super5 baseline | 0.9064 | 0.7754 | 0.8344 | 0.5526 |

后文的 `Δ` 均相对该 baseline。

## Clean PN2021 结果总览

| 方法 | 代表 run | PN2021 AUROC | ΔAUROC | PN2021 AUPRC | ΔAUPRC |
|---|---|---:|---:|---:|---:|
| Baseline EfficientNet1DV2 | `triple_labels/super5` | 0.8344 | +0.0000 | 0.5526 | +0.0000 |
| Latent-Hull real-anchor main | `latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8341 | -0.0003 | 0.5560 | +0.0034 |
| Prompt-token v14 | `online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03` | 0.8342 | -0.0001 | 0.5558 | +0.0033 |
| Prompt-token v4 balanced pool | `online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20` | 0.8344 | +0.0001 | 0.5558 | +0.0032 |
| Latent-Hull nin M5 | `latent_hull_super5_pilot/nin_M5_lambda025_ep20` | 0.8343 | -0.0001 | 0.5557 | +0.0031 |
| Latent-Hull extra M5 | `latent_hull_super5_pilot/extra_M5_lambda025_ep20` | 0.8346 | +0.0002 | 0.5557 | +0.0031 |
| Latent-Hull geo M20 | `latent_hull_super5_pilot/geo_M20_lambda025_ep20` | 0.8345 | +0.0002 | 0.5556 | +0.0030 |
| Synth-anchor K400 nin | `synth_anchored_super5_v13b_K400/nin_k400` | 0.8334 | -0.0009 | 0.5555 | +0.0029 |
| Multi-center prompt-token v36 chapman | `online_at_multicenter_v36/chapman_v36_M10_ep20_seed03` | 0.8344 | +0.0000 | 0.5554 | +0.0028 |
| Multi-center prompt-token v36 cpsc | `online_at_multicenter_v36/cpsc_v36_M10_ep20_seed03` | 0.8340 | -0.0003 | 0.5553 | +0.0028 |
| Multi-center prompt-token v36 georgia | `online_at_multicenter_v36/georgia_v36_M10_ep20_seed03` | 0.8343 | -0.0001 | 0.5541 | +0.0015 |

结论：

- clean PN2021 上，最稳定、最高的 AUPRC 提升来自 Latent-Hull real-anchor 主实验，AUPRC 从 0.5526 提升到 0.5560，提升约 +0.0034。
- AUROC 整体变化很小，最好和 baseline 只差约 0.0002 量级；我们的方法主要改善 PR 曲线，而不是明显提高 ROC 曲线。
- Prompt-token 方法已经能接近 Latent-Hull 主结果，但当前没有超过 real-anchor Latent-Hull。

## 方法一：Baseline 与 PN2021 v3 重评测

我们首先把 PN2021 标签映射统一到 `v3_super5_normsuppress_20260501`，并重新评测 baseline 和历史可比 checkpoint。这个步骤的意义是建立干净、可复现的比较基线。

结果：

```text
baseline PN2021 avg = AUROC 0.8344, AUPRC 0.5526
```

该 baseline 的 AUROC 已经较高，后续增强方法很难显著提高 AUROC；但 AUPRC 对少数类和跨中心分布变化更敏感，因此成为主要观察指标。

## 方法二：Synth-anchor / Real-anchor 历史增强

历史 synth-anchor 和 real-anchor 方法把少量目标中心样本或生成样本作为 anchor，训练 EfficientNet1DV2 做跨中心适应。

代表结果：

| 方法 | run | PN2021 AUROC | PN2021 AUPRC | ΔAUPRC |
|---|---|---:|---:|---:|
| Synth-anchor K400 | `synth_anchored_super5_v13b_K400/nin_k400` | 0.8334 | 0.5555 | +0.0029 |
| Synth-anchor v14 CTV2 geo k200 | `synth_anchored_super5_v14_ctv2/geo_k200` | 0.8340 | 0.5550 | +0.0024 |
| Real-anchor extra k200 | `real_anchored_super5/extra_real_k200` | 0.8334 | 0.5542 | +0.0016 |

结论：

- anchor 类方法是有效的，能把 AUPRC 从 0.5526 提高到约 0.5542-0.5555。
- Synth-anchor K400 在 corrupted PN2021-C 上表现很强，说明“更多样本 + 合理筛选”的简单策略仍然有价值。
- 但 clean PN2021 最优结果仍不是 synth-anchor，而是后续的 Latent-Hull real-anchor。

## 方法三：Latent-Hull Online Adversarial Training

Latent-Hull 是目前 clean PN2021 主结果。核心思想是在 ECGTwin VAE latent space 中做 same-label convex combination，而不是直接做自由 PGD perturbation：

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

当前主设定：

```text
same-label combination only
M = 10
lambda = 0.25
hull_steps = 5
coefficient mode = optimized softmax weights
```

主要 ablation：

| ablation | run | PN2021 AUROC | PN2021 AUPRC | 结论 |
|---|---|---:|---:|---|
| C3 optimized, M10, lambda 0.25 | `latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8341 | 0.5560 | clean AUPRC 最优 |
| M5, lambda 0.25 | `latent_hull_super5_pilot/nin_M5_lambda025_ep20` | 0.8343 | 0.5557 | 接近主结果 |
| extra M5, lambda 0.25 | `latent_hull_super5_pilot/extra_M5_lambda025_ep20` | 0.8346 | 0.5557 | AUROC 略高，AUPRC 略低 |
| geo M20, lambda 0.25 | `latent_hull_super5_pilot/geo_M20_lambda025_ep20` | 0.8345 | 0.5556 | self-drop 较低但 absolute AUPRC 不最高 |
| C2 Dirichlet | `latent_hull_coeff_ablation/nin_M10_lambda025_C2_dirichlet_noabort_ep20` | 0.8344 | 0.5550 | 固定随机系数不如 optimized |
| lambda 0.40 | `latent_hull_lambda_ablation/nin_M10_lambda040_ep20` | 0.8339 | 0.5552 | 不如 lambda 0.25 |

结论：

- Latent-Hull 是 clean PN2021 当前最强主线，AUPRC 提升约 +0.0034。
- optimized softmax coefficient 比 one-hot/uniform/Dirichlet 更合理。
- `lambda=0.25` 是目前最稳的折中；`lambda=0.40` 没有带来额外收益。
- 这个方法对 AUROC 的提升不明显，但在 AUPRC 上稳定优于 baseline。

## 方法四：ECGTwin Center Prompt Token

center prompt-token 的目标是用少量目标中心样本训练 `<center_CLASS>` 风格 token，并把它作为 learnable 768-d embedding 插入 ECGTwin 的 text condition，而不是改 tokenizer vocabulary。

当前实现方式：

```text
diagnosis text embedding + learnable <center_CLASS> token
token shape = 768-d
每个 center 5 个 class token：CD, HYP, MI, NORM, STTC
ECGTwin base_vector 仍来自 reference ECG/text/patient info，不被 center token 污染
```

代表结果：

| 方法 | run | PN2021 AUROC | PN2021 AUPRC | ΔAUPRC |
|---|---|---:|---:|---:|
| Prompt-token v14 direct MV4 | `online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03` | 0.8342 | 0.5558 | +0.0033 |
| Prompt-token v4 balanced pool | `online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20` | 0.8344 | 0.5558 | +0.0032 |
| Prompt-token repeat2 v3 | `online_at_pilot_v3/ningbo_tokenv3_repeat2_gated153_M10_ep20` | 0.8345 | 0.5556 | +0.0030 |
| Prompt-token v31 mixed-diverse | `online_at_pilot_v16/ningbo_v31_mixed_diverse_q50_M10_ep20_seed03` | 0.8340 | 0.5550 | +0.0025 |

探索过但没有继续作为主线的方向：

- 单纯增加 `token_repeat`：容易过强，不稳定。
- target-center-only token training：没有明显优于 seed42 direct MV4。
- quality-ref seed1042：提升了部分 gate count，但 downstream 不如 old seed42。
- high-confidence-only / boundary mixing / source-aware hybrid：能形成对照，但没有超过 v4/v14。
- class-specific checkpoint composition：提高了一些 gate 数量，但 downstream 没有显著收益。

结论：

- center-token 是可行的，能生成目标中心风格的 gated ECG 样本，并带来约 +0.0032 到 +0.0033 的 clean AUPRC 提升。
- 目前 prompt-token 最强结果非常接近 Latent-Hull real-anchor，但还没超过。
- 下一步不应继续盲目扩大 pool 或调 downstream 参数，而应做 gate-aware token checkpoint/probe selection、held-out denoise MSE 和更强 reference selection。

## 方法五：Multi-center Prompt-token v36

我们进一步把 prompt-token 从 `ningbo` 扩展到 `chapman_shaoxing`、`cpsc_2018`、`georgia`。经过 v32 生成、v33/v35 targeted boost 后，形成 v36 multi-center gated pool。

v36 pool：

| center | gated count | class counts | 备注 |
|---|---:|---|---|
| `chapman_shaoxing` | 115 | NORM=38, MI=33, STTC=44 | 类数达标 |
| `cpsc_2018` | 97 | NORM=38, STTC=59, MI=0 | 当前 v3 primary cache 无 MI |
| `georgia` | 149 | NORM=38, MI=74, STTC=37 | 类数达标 |

v36 downstream：

| run | PN2021 AUROC | PN2021 AUPRC | ΔAUPRC | 结论 |
|---|---:|---:|---:|---|
| `chapman_v36_M10_ep20_seed03` | 0.8344 | 0.5554 | +0.0028 | 低于 prompt-token v4/v14 |
| `cpsc_v36_M10_ep20_seed03` | 0.8340 | 0.5553 | +0.0028 | MI 缺失限制明显 |
| `georgia_v36_M10_ep20_seed03` | 0.8343 | 0.5541 | +0.0015 | 不适合作为主结果 |

结论：

- multi-center prompt-token 生成链路已经跑通，且可以通过 targeted boost 补齐部分低通过类别。
- downstream 没有超过 single-center `ningbo` prompt-token 结果。
- `cpsc_2018` 暴露了当前 primary-label reference selection 的问题：该中心 v3 primary cache 没有 MI reference，因此 MI token 需要 multi-label/SNOMED fallback，而不是继续换随机种子。

## PN2021-C Robustness 结果

PN2021-C 衡量 corrupted target-center 数据上的 absolute performance 和相对 clean drop。当前 12 个完整 PN2021-C JSON 已纳入汇总。

| model | corrupted AUROC | ΔAUROC vs baseline | corrupted AUPRC | ΔAUPRC vs baseline | self-clean AUPRC drop |
|---|---:|---:|---:|---:|---:|
| `online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03` | 0.8170 | +0.0013 | 0.4928 | +0.0039 | 0.0349 |
| `synth_anchored_super5_v13b_K400/nin_k400` | 0.8166 | +0.0009 | 0.4927 | +0.0038 | 0.0348 |
| `latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8170 | +0.0013 | 0.4927 | +0.0038 | 0.0351 |
| `online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20` | 0.8171 | +0.0013 | 0.4921 | +0.0032 | 0.0352 |
| `latent_hull_super5_pilot/extra_M5_lambda025_ep20` | 0.8171 | +0.0014 | 0.4921 | +0.0032 | 0.0350 |
| `latent_hull_lambda_ablation/nin_M10_lambda040_ep20` | 0.8168 | +0.0011 | 0.4919 | +0.0031 | 0.0352 |
| `latent_hull_super5_pilot/nin_M5_lambda025_ep20` | 0.8168 | +0.0011 | 0.4919 | +0.0030 | 0.0352 |
| `online_at_pilot_v16/ningbo_v31_mixed_diverse_q50_M10_ep20_seed03` | 0.8168 | +0.0011 | 0.4914 | +0.0025 | 0.0347 |
| `online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20` | 0.8164 | +0.0006 | 0.4914 | +0.0025 | 0.0350 |
| `latent_hull_coeff_ablation/nin_M10_lambda025_C2_dirichlet_noabort_ep20` | 0.8168 | +0.0011 | 0.4913 | +0.0024 | 0.0347 |
| `latent_hull_super5_pilot/geo_M20_lambda025_ep20` | 0.8169 | +0.0011 | 0.4912 | +0.0023 | 0.0346 |
| `triple_labels/super5` | 0.8157 | +0.0000 | 0.4889 | +0.0000 | 0.0341 |

Robustness 结论：

- corrupted absolute AUPRC 最好的是 prompt-token v14，达到 0.4928，比 baseline 0.4889 高约 +0.0039。
- synth-anchor K400 和 Latent-Hull nin_M10 的 corrupted AUPRC 也很接近，分别是 0.4927 和 0.4927。
- 但 baseline 的 self-clean AUPRC drop 仍然最低，说明增强方法不是降低 corruption 相对跌幅，而是在 clean 和 corrupted 两端都抬高了 absolute performance。
- 因此论文里应表述为：增强策略提高 corrupted ECG 上的绝对 AUROC/AUPRC，尤其 AUPRC；不能表述为显著提升 corruption invariance。

## ECGTwin Author Reproduction 状态

ECGTwin 作者流程 IBE stage 已完成 partial run 到 epoch 9：

```text
root: /root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512
epoch 1 train_loss=3.052135 eval_score=0.718906
epoch 9 train_loss=0.455224 eval_score=0.611132
```

该运行保存了 `IBE_best.pth`、`best.pt`、`latest.pt`、`metrics.jsonl` 和 loss curve，但 epoch 9 后主进程被系统 kill，full author pipeline 未完成。因此它目前只能作为 ECGTwin 架构复现和训练日志证据，不能作为 AUROC/AUPRC downstream 结果引用。

## 总体判断

当前最适合写进论文主结果的 clean PN2021 方法：

```text
Latent-Hull real-anchor nin_M10_lambda025
PN2021 AUROC/AUPRC = 0.8341 / 0.5560
相对 baseline = -0.0003 / +0.0034
```

当前最适合写进 robustness 对照的 PN2021-C 方法：

```text
Prompt-token v14
PN2021-C corrupted AUROC/AUPRC = 0.8170 / 0.4928
相对 baseline corrupted = +0.0013 / +0.0039
```

总体规律：

- AUPRC 提升比 AUROC 更明显，说明方法主要改善少数类或跨中心 PR 表现。
- Real-anchor Latent-Hull 是当前 clean generalization 主线。
- Prompt-token 已经接近主线，并在 corrupted absolute AUPRC 上最强，但还需要更好的 token/ref selection 才能超过 real-anchor。
- Multi-center prompt-token 当前证明了可行性，但 downstream 暂未超过 single-center `ningbo` prompt-token。
- 后续最值得做的是改进 center-token 上游训练与 reference selection，而不是继续大量 downstream sweep。

## 下一步建议

1. 给 `cpsc_2018` 增加 multi-label/SNOMED fallback reference selection，解决 MI primary reference 缺失。
2. 对 prompt-token 做 held-out generation probe 和 denoise MSE，先筛 token checkpoint，再进入 online-AT。
3. 保留 Latent-Hull `M=10, lambda=0.25, optimized weights` 作为 clean PN2021 主方法。
4. 报告 PN2021-C 时同时给 corrupted absolute metric 和 self-clean drop，避免把 absolute performance 提升误写成 robustness drop 改善。
