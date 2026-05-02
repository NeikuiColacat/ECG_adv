# PTB-XL Prompt-Token Boundary AT Pipeline

本文档定义一个新的 ablation：

```text
PTB-XL folds 1-8
-> train five PTB-XL source-style prompt tokens
-> generate ECGTwin samples with class prompt + <ptbxl_CLASS>
-> gate generated samples
-> Latent-Hull online AT with boundary-confidence acceptance
-> PTB-XL fold10 + PN2021 v3 evaluation
```

## Motivation

之前的 direct synth augmentation 已经失败：

```text
baseline:
  PTB-XL  0.9064 / 0.7754
  PN2021  0.8344 / 0.5526

direct ECGTwin synth r0.25:
  PTB-XL  0.9010 / 0.7605
  PN2021  0.8277 / 0.5401
```

所以本实验不再把 ECGTwin 合成样本当作普通扩增数据同权训练，而是测试：

```text
1. PTB-XL source-style prompt token 是否改善 source-domain generation quality；
2. 对抗样本只接受 target sigmoid probability in [0.50, 0.60] 的边界样本；
3. 是否能作为小权重 regularizer 改善 EfficientNet1DV2 的 AUROC/AUPRC。
```

## Token Definition

训练 5 个 PTB-XL source-style center-class prompt token：

```text
<ptbxl_source_CD>
<ptbxl_source_HYP>
<ptbxl_source_MI>
<ptbxl_source_NORM>
<ptbxl_source_STTC>
```

实现仍然是 textual-inversion-style soft prompt：

```text
base diagnosis text_embed + learnable 768-d <ptbxl_source_CLASS>
```

不是 tokenizer vocabulary hack。ECGTwin 的 DiT、VAE、IBE、nomic text encoder 全部冻结。

`base_vector` 仍来自同类 PTB-XL reference ECG/text/patient info：

```text
ref latent + base diagnosis text_embed + pat_info -> IBE -> base_vector
```

center/source style 放在 text token path，不污染 IBE 的个体 base_vector。

## Data Split

PTB-XL split 固定：

```text
token train / generation refs: folds 1-8 only
model selection: fold 9 only
final in-domain test: fold 10 only
external eval: PN2021 v3 7 centers
```

新增 PTB-XL prompt-token cache：

```text
script:
  scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py

cache:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1/
    center_full_latents/ptbxl_source.pt
    ref_selection/ptbxl_source_k1000_seed42.json
```

Selection policy:

```text
folds = 1-8
per_class = 200
K = 1000
primary class = PTB-XL diagnostic_class
super5 multi-hot = scripts/triple_labels/label_schemes.py source of truth
```

## Generation

第一版生成五类：

```text
CD, HYP, MI, NORM, STTC
```

命令入口：

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth.py
```

输出：

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/generated_pool_v1/ptbxl_source/
  samples.npz
  summary.json
```

第一轮计划：

```text
n_per_class = 80
DDPM steps = 25
```

如果 gate 后 NORM/MI/STTC 数量不足，再单类 boost；不要盲目扩大整池。

## Gate

生成池 gate：

```text
script:
  scripts/ecgtwin_gen/gate_prompt_token_synth.py

classes:
  CD/HYP/MI/NORM/STTC

min_target_prob:
  0.30

digital gate:
  util.ecg_digital_features class-specific checks
```

本实验允许 HYP/CD 经过严格 gate 后保留 trust：

```text
--allow_hyp_cd_trust
```

但结论解释要保守：

- 如果 HYP/CD gate 不通过，下游不会强行使用。
- 如果 HYP/CD 通过，也只作为 source-style all5 ablation，不直接推翻之前 HYP/CD 风险结论。

## Boundary-Confidence Online AT

核心约束：

```text
Only push adversarial samples when:
  0.50 <= sigmoid(logit_target) <= 0.60
```

目的：

```text
避免 latent-hull 内层攻击过强；
让样本保持在“模型边界附近”而不是明显改变语义；
减少 GT label drift 风险。
```

实现：

```text
scripts/pgd_cross_center/synth_online_at_super5.py
  --boundary_prob_min 0.50
  --boundary_prob_max 0.60
```

攻击设置：

```text
attack_mode = latent_hull
classes_in_scope = CD/HYP/MI/NORM/STTC
M = 10
lambda = 0.25
hull_steps = 5
K_anchor = 150
adv_weight = 0.10
anchor_lambda = 0.10
lr = 2e-5
epochs = 20
```

注意：这个实验不要求高 ASR。传统 ASR 是“让模型错”的指标，但本实验目标是边界样本，
因此 runner 中将 `--asr_low_threshold 0.0`，以免边界约束和旧 ASR guard 冲突。

## Runner

一键入口：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/ecgtwin_gen/run_ptbxl_prompt_token_boundary_at_v1.sh
```

主要输出：

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/
  cache_v1/
  prompt_token_runs/ptbxl_source_5class_steps2000/
  generated_pool_v1/ptbxl_source/
  online_at/ptbxl_source_all5_boundary_p050_060_M10_ep20/
  logs/
```

Full eval:

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/
  online_at/ptbxl_source_all5_boundary_p050_060_M10_ep20/
    eval_result_v3_super5_normsuppress.json
```

## Acceptance Criteria

主判定：

```text
PN2021 v3 avg macro AUPRC > 0.5560
```

最低可接受信号：

```text
PN2021 AUPRC >= prompt-token v4 band around 0.5558
PTB-XL fold10 AUPRC does not drop more than ~0.001
```

失败判定：

```text
PTB-XL 或 PN2021 明显低于 baseline；
boundary gate 几乎没有样本进入 buffer；
提升只出现在 fold9 quick eval，full PN2021 不复现。
```

## Result 2026-05-02

Run completed:

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/
```

PTB-XL source-token training:

```text
steps = 2000
final token_delta_norm ~= 1.2298
```

Generation / gate:

| class | generated | gated | trust |
|---|---:|---:|---:|
| CD | 80 | 7 | 1.0 |
| HYP | 80 | 8 | 1.0 |
| MI | 80 | 66 | 1.0 |
| NORM | 80 | 36 | 1.0 |
| STTC | 80 | 34 | 1.0 |
| total | 400 | 151 | - |

Observation:

```text
MI/STTC/NORM remain the strongest generated classes.
CD/HYP passed strict gate only sparsely; keep their interpretation as all5
ablation evidence, not as proof that HYP/CD generation is solved.
```

Boundary AT:

```text
boundary window = target sigmoid p in [0.50, 0.60]
final buffer size = 159
most candidate samples were rejected by boundary gate each epoch
best quick subset = epoch 12, AUROC/AUPRC 0.8454 / 0.6650
```

Full evaluation:

| model | PTB-XL AUROC | PTB-XL AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---:|---:|---:|---:|
| baseline `/root/autodl-tmp/triple_labels/super5` | 0.9064 | 0.7754 | 0.8344 | 0.5526 |
| PTB-XL source-token boundary AT v1 | 0.9067 | 0.7765 | 0.8345 | 0.5549 |
| best real-anchor latent-hull reference | 0.9064 | 0.7765 | 0.8341 | 0.5560 |

Interpretation:

```text
This is a positive ablation versus baseline and clearly better than direct
synthetic mixing. It does not beat the best real-anchor AUPRC route.

The source-style PTB-XL prompt token likely improves in-domain/source-style
regularization and gives a modest external AUPRC lift, but target-center
real-anchor Latent-Hull remains the stronger mainline.
```

Useful next ablations:

```text
1. trusted3 boundary AT: use only NORM/MI/STTC gated pool, same p in [0.50,0.60].
2. boundary window sweep: [0.45,0.60] vs [0.50,0.65] if buffer is too sparse.
3. teacher-soft labels for synthetic/boundary samples to reduce HYP/CD/MI-STTC label noise.
4. multi-vector PTB-XL source token only if trusted3 shows a further positive signal.
```

## Status

2026-05-02 planned, launched, and completed.

Code changes:

```text
NEW: scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py
NEW: scripts/ecgtwin_gen/run_ptbxl_prompt_token_boundary_at_v1.sh
MOD: scripts/ecgtwin_gen/gate_prompt_token_synth.py
MOD: scripts/pgd_cross_center/synth_online_at_super5.py
```

Current caveat:

```text
This is a source-style DiT/prompt-token ablation, not the highest-confidence
target-center real-anchor mainline. If it improves only PTB-XL but not PN2021,
report it as source-domain regularization rather than cross-center enhancement.
```
