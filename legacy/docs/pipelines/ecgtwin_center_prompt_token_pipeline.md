# ECGTwin Center Prompt Token Pipeline

本文档定义目标中心 few-shot prompt-token 方案。目标是用每个目标中心 K=500 条 ECG，
学习可写进 prompt 的 center-class token，使 ECGTwin 在生成时更接近目标中心风格。

## 关键结论

ECGTwin 官方支持 open-vocabulary prompt，但不是离散 tokenizer token 训练框架。

官方生成链路：

```text
reference ECG latent + reference text_embed + reference pat_info
  -> IBE -> base_vector

target prompt text
  -> bert-base-uncased tokenizer
  -> nomic-ai/nomic-embed-text-v1.5
  -> text_embed sequence

DiT(x_t, t, text_embed, text_embed_mask, pat_info, base_vector)
  -> VAE decoder -> ECG
```

源码：

```text
model/ECGTwin/ECGTwin_inference.py
model/ECGTwin/utils/data_utils.py
model/ECGTwin/utils/inference_utils.py
model/ECGTwin/module/DiT_ECGTwin.py
util/ecgtwin_utils.py
```

因此 `<ningbo_MI>` 这类 token 不能直接依赖 nomic tokenizer 理解。工程上需要一个 prompt compiler：普通诊断短语仍走 nomic embedding，center token 由我们自己的可学习 768-d 向量追加到 ECGTwin 的 `text_embed` 序列中。

使用语义示例：

```text
Generate center A MI ECG:
  user prompt       = "stemi|st elevation myocardial infarction|acute <A_MI>"
  prompt compiler  = nomic_embed("stemi|st elevation myocardial infarction|acute")
                     + learnable_embed("<A_MI>")
  ECGTwin condition = text_embed_with_center_token + reference-derived base_vector
```

也就是说，方法确实是“生成中心 A 的 MI 类 ECG 时，把训练好的 `<A_MI>` token 加到
MI prompt 里”。区别在于 `<A_MI>` 由本项目 prompt compiler 替换成可训练 embedding，
而不是由 ECGTwin 官方 tokenizer 自动学习。

## 2026-05-03 最新状态摘要

当前文档已经覆盖 v1/v2/v3/v4/v31/v32-v36 的主要 prompt-token 实验。
最新可执行结论如下：

```text
Active implementation:
  methods/ecgtwin_gen/prompt_token/
  scripts/ecgtwin_gen/train_center_prompt_tokens.py
  scripts/ecgtwin_gen/generate_center_prompt_token_synth.py
  scripts/ecgtwin_gen/gate_prompt_token_synth.py

Recommended token schema:
  direct multi-vector prompt token, MV4
  embeddings[center, class, vector, 768]

Current best prompt-token downstream result:
  ningbo v4 balanced gated pool + Latent-Hull online AT
  PN2021 v3 = 0.8344 / 0.5558

Current best overall AUPRC comparison point:
  real-anchor Latent-Hull nin_M10
  PN2021 v3 = 0.8341 / 0.5560
```

Multi-center v36 已完成：

| run | pool | PN2021 AUROC/AUPRC | decision |
|---|---|---:|---|
| `chapman_v36_M10_ep20_seed03` | NORM=38, MI=33, STTC=44 | 0.8344 / 0.5554 | below v4/real-anchor |
| `cpsc_v36_M10_ep20_seed03` | NORM=38, STTC=59, MI=0 | 0.8340 / 0.5553 | below v4/real-anchor; MI cache limitation |
| `georgia_v36_M10_ep20_seed03` | NORM=38, MI=74, STTC=37 | 0.8343 / 0.5541 | below v4/real-anchor |

PN2021-C 当前记录：

```text
Best prompt-token absolute corrupted mean AUPRC:
  prompt-token v14 ~= 0.492808

But baseline still has the lowest self-clean relative AUPRC drop.
因此增强模型应描述为 improving absolute corrupted performance，
不要描述为已经改善 relative robustness drop。
```

下一步不建议继续做的事情：

```text
1. 不再盲目增加 token_repeat。
2. 不再盲目扩大同一个 generator distribution 的样本池。
3. 不再只做 MI-only prompt supplementation。
4. 不把 quick-eval 提升当成 acceptance criterion。
```

下一步推荐：

```text
gate-aware token checkpoint/probe selection
diversity-preserving candidate selection
multi-label/SNOMED fallback for centers such as cpsc_2018 where primary MI refs are absent
full PN2021 seven-center evaluation as acceptance criterion
```

## 2026-05-03 Token Regeneration Plan: PN2021 Big Centers + PTB-XL Source

本轮重新生成的目标不是继续随机扩大旧合成池，而是重新训练一组可审计的
`direct MV4` prompt-token bank，并配套做 target-style 验证。主问题是：

```text
加入 <center_CLASS> soft prompt token 后，
ECGTwin 生成的 ECG 是否比 vanilla prompt 更接近目标中心真实 ECG？
```

### Scope

中心集合：

```text
PN2021 big centers:
  ningbo
  chapman_shaoxing
  cpsc_2018
  georgia

PTB-XL source center:
  ptbxl_source
```

类别集合：

```text
CD, HYP, MI, NORM, STTC
```

主结论仍优先看 `NORM/MI/STTC`，`CD/HYP` 继续作为 smoke/gated-only 类别。
原因没有变：HYP/CD 能生成，但数字 ECG gate 和 frozen super5 victim gate 仍不如
NORM/MI/STTC 稳定。

### Token Schema

本轮推荐使用 direct multi-vector token：

```text
embeddings[center, class, vector, 768]
vector = 0..3
```

也就是每个 `center-class` 有 4 个 768-d soft-prompt 向量：

```text
<ningbo_MI> = 4 x 768
<ptbxl_source_STTC> = 4 x 768
```

生成某一类样本时只追加目标类别 token。例如生成 ningbo 风格 MI：

```text
target text_embed = nomic_embed("stemi|st elevation myocardial infarction|acute")
                    + learned_soft_prompt("<ningbo_MI>", 4 x 768)
```

当前执行版仍使用 primary-class token；多标签样本不同时追加 5 个 token。
多标签 token 组合单独作为后续 ablation，避免本轮把 token 数、标签语义和中心风格混在一起。

### Reference And Text Policy

`base_vector` 仍来自 ECGTwin 官方 IBE/reference path：

```text
ref_latent + ref_text_embed + ref_pat_info -> IBE -> base_vector
```

本轮重训使用：

```text
--ref_text_mode actual_report
```

含义：

- PN2021：优先使用 SNOMED/report cache 中的实际诊断文本 embedding；
- PTB-XL：优先使用 `PTBXL_vae_multi_nomic.pt` 中已有的文本 embedding；
- 缺失时才回退到 class fallback prompt；
- center style 不写进 `base_vector`，只写进 target prompt token path。

这样可以避免旧实验里 `class_fallback` 文本过强或过弱导致的混淆：

```text
base_vector: 个体/reference 条件
prompt token: 目标中心风格偏移
diagnosis text: 目标疾病语义
```

### Anchor Selection

PN2021 使用已经存在的 K=500 selection：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/ref_selection/<center>_k500_seed42.json
```

PTB-XL 重新补一个 K=500 balanced selection：

```text
/root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1/ref_selection/ptbxl_source_k500_seed42.json
```

PTB-XL selection 规则：

```text
folds: 1-8 only
per class: 100
total K: 500
fold9/fold10: never used for token training
```

所有 K anchors 后续都必须从 center-style classifier 的 train/val/test 和下游
target-center eval 中排除。

### New Token Runs

可复现一键脚本：

```bash
bash scripts/ecgtwin_gen/run_center_token_regen_20260503.sh
```

PN2021 big4 token bank：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --cache_root /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1 \
  --prompt_bank /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt \
  --save_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500 \
  --K 500 \
  --seed 42 \
  --total_steps 2500 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy center_class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 4 \
  --token_orth_weight 1e-4 \
  --ref_text_mode actual_report \
  --log_every 50 \
  --save_every 500
```

PTB-XL source token bank：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/build_ptbxl_prompt_token_cache.py \
  --out_root /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1 \
  --center ptbxl_source \
  --folds 1 2 3 4 5 6 7 8 \
  --per_class 100 \
  --seed 42

/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ptbxl_source \
  --cache_root /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/cache_v1 \
  --prompt_bank /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt \
  --save_dir /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503 \
  --K 500 \
  --seed 42 \
  --total_steps 2500 \
  --batch_size 16 \
  --num_workers 4 \
  --sample_strategy class_balanced \
  --amp_dtype bf16 \
  --token_mode direct \
  --n_token_vectors 4 \
  --token_orth_weight 1e-4 \
  --ref_text_mode actual_report \
  --log_every 50 \
  --save_every 500
```

## 2026-05-03 v41 Corrective Objective: Style + Semantic Prompt Tokens

v40 的严格验证结论是：

```text
direct MV4 token 可训练，也会移动部分 style score；
但在 no-leak、same-label、EfficientNet feature probe 和 C2ST 下，
target-token 没有稳定优于 vanilla / wrong-center / PTB-XL source controls。
```

因此下一步不再盲目扩大 v40 样本池，而是修改 token 训练目标本身。v41 保留原始
ECGTwin denoising objective，并额外加入两个冻结模型约束：

```text
loss =
  denoising_mse_full_timestep
  + 1e-3 * init_regularization
  + 1e-4 * multi_vector_orthogonality
  + style_loss_weight * frozen_center_style_CE
  + semantic_loss_weight * frozen_super5_BCE
```

关键设计：

```text
1. 主 denoising MSE 仍在完整 DDPM timestep 范围采样，避免破坏 ECGTwin 原始训练目标。
2. style/semantic loss 单独使用低噪声 one-step x0 prediction：
     z_t_aux = add_noise(z0, t_aux), t_aux < 250
     x0_hat = predict_x0(z_t_aux, eps_pred, t_aux)
     ECG_hat = VAE_decode(x0_hat)
3. center style loss:
     frozen full-10s PN2021 center classifier
     target = 当前 anchor 的 center id
4. semantic preservation loss:
     frozen EfficientNet1DV2 super5 classifier
     target = anchor 的 super5 multi-hot label
5. 只有 prompt token bank 可训练；VAE、IBE、DiT、style classifier、super5 classifier 全部冻结。
```

可复现脚本：

```bash
bash scripts/ecgtwin_gen/run_center_token_v41_style_semantic_20260503.sh
```

默认配置：

```text
token schema: direct MV4, embeddings[center, class, 4, 768]
centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
K: 500 per center
ref_text_mode: actual_report
steps: 1500
batch_size: 2
style_loss_weight: 0.01
semantic_loss_weight: 0.05
aux_timestep_max: 250
style checkpoint:
  /root/autodl-tmp/per_center_style_classifier_full10s_max3000/best_model.pt
semantic checkpoint:
  /root/autodl-tmp/triple_labels/super5/best_model.pt
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v41_style_semantic_direct_mv4_actual_steps1500_20260503
validation:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v41_style_semantic_20260503
```

Acceptance criterion 不变：

```text
1. target-token digital/semantic gate pass rate 不低于 vanilla；
2. no-leak same-label EfficientNet feature probe:
     P(target center | target-token) > vanilla/wrong-center/PTB-XL source;
3. same-label C2ST:
     C2ST(real target, target-token) <= vanilla/wrong-center/PTB-XL source;
4. 只有满足前 3 条，才进入 downstream AUROC/AUPRC 训练。
```

## 2026-05-03 v42 Improvement Plan Before Long Run

v41 smoke 已经证明可微分链路可跑通，但正式长跑前要避免一个关键风险：

```text
不能用 weak/leaky center-style classifier 直接驱动 token，然后再用同一个 probe 证明 token 有效。
```

因此 v42 的顺序是：

```text
1. 先完成新的 EfficientNet1DV2 full10 minimal_resample baseline。
2. 用新 baseline 的 frozen features 训练 no-leak same-label center-style probe。
3. 再把这个 no-leak probe 作为 style auxiliary loss 训练 center token。
4. 用独立 held-out split 和 C2ST 验证 target-token 是否优于 controls。
```

### Candidate Objectives

保留 ECGTwin 原始 denoising objective：

```text
L = L_denoise_full_t
  + w_init * L_init
  + w_orth * L_orth
  + w_style * L_center_style
  + w_sem * L_super5_semantic
  + optional w_div * L_token_diversity
```

定义：

```text
L_denoise_full_t:
  原始 ECGTwin DiT noise prediction MSE，t 从完整 DDPM timestep 采样。

L_center_style:
  使用 no-leak real-only center probe，对低噪声 one-step x0 decode 后的 ECG 做
  target-center CE。

L_super5_semantic:
  使用 frozen super5 EfficientNet，约束 decode 后 ECG 仍保留 anchor 的 super5 label。

L_token_diversity:
  只作为小权重正则，避免 4/8 个 token vectors collapse 到同一个方向。
```

style/semantic loss 使用低噪声 one-step x0，而不是完整 DDPM 采样：

```text
z_t_aux = add_noise(z0, eps, t_aux), t_aux < aux_timestep_max
eps_pred = DiT(z_t_aux, t_aux, prompt + token, base_vector)
x0_hat = predict_x0(z_t_aux, eps_pred, t_aux)
ecg_hat = VAE_decode(x0_hat)
```

这样训练成本可控，也避免把完整 25/50-step sampling 放进每个 token update。

### Token Architecture Search

先跑小网格，不做盲目大规模生成：

| schema | vectors | purpose |
|---|---:|---|
| direct MV4 | 4 x 768 | 当前最强 baseline |
| direct MV8 | 8 x 768 | 测试 text-inversion capacity 是否不足 |
| factorized C4+Y4+R4 | 12 x 768 | 分离 center style、class semantics、center-class residual |
| single vector | 1 x 768 | 低容量负对照 |

生成时仍然只追加目标 class token。例如生成 ningbo MI：

```text
nomic_embed("stemi|st elevation myocardial infarction|acute")
  + learned_soft_prompt("<ningbo_MI>")
```

不会同时追加 5 个 class token。多标签 token 组合单独作为 ablation：

```text
exact multi-hot sample:
  append tokens for all positive classes, each with lower repeat/scale
```

### Hyperparameter Grid

第一轮只跑 small grid，每个候选先做 20/class generation probe：

| group | values |
|---|---|
| learning rate | `3e-4`, `1e-3` |
| `n_token_vectors` | `4`, `8` |
| `reg_init_weight` | `1e-4`, `1e-3`, `1e-2` |
| `token_orth_weight` | `1e-5`, `1e-4` |
| `style_loss_weight` | `0.003`, `0.01`, `0.03` |
| `semantic_loss_weight` | `0.02`, `0.05`, `0.10` |
| `aux_timestep_max` | `100`, `250`, `500` |
| `ref_text_mode` | `actual_report`, `class_fallback` ablation |

推荐先跑 4 个候选，不做全网格。当前已落地的第一条是保守版 V42-A0，
先降低辅助 loss，避免 style classifier 直接压过 ECGTwin denoising MSE：

```text
V42-A0 direct MV4, lr=1e-3, style=0.003, sem=0.02, aux_t=250
       script: scripts/ecgtwin_gen/run_center_token_v42_noleak_style_semantic_20260503.sh
       style aux: /root/autodl-tmp/ecgtwin_prompt_token_super5/style_aux_noleak_full10_task1_v1/best_model.pt
       style aux test acc / macro F1 = 0.6279 / 0.6555
       semantic ckpt: /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt
       status: failed promotion gate
       report: docs/tmp_md/center_token_v42_validation_20260503.md
V42-B direct MV8, lr=3e-4, style=0.01, sem=0.05, aux_t=250
V42-C direct MV4, lr=1e-3, style=0.03, sem=0.10, aux_t=100
V42-D factorized C4+Y4+R4, lr=3e-4, style=0.01, sem=0.05, aux_t=250
```

V42-A0 失败原因：

```text
target-token style score only improves ningbo and partly raw cpsc_2018;
chapman/georgia lose to controls;
same-label C2ST is worse than controls on all centers under the Task-1 full10
EfficientNet feature probe.
```

因此下一步先不扩大 v42 final samples，而是做 checkpoint selection：

```text
probe v42 step00500 and step01000
accept only if earlier checkpoint improves C2ST while keeping target-style score
otherwise reduce style loss or move style signal to post-hoc sample selection
```

### Reference Policy

主线仍使用 K=500：

```text
PN2021 center K=500 anchors
PTB-XL source K=500 anchors
fold9/fold10 never used for PTB-XL source token training
```

但要补两个 ablation：

```text
same-class reference:
  ref ECG primary class == target generated class

cross-class reference:
  ref ECG primary class != target class, target prompt decides diagnosis
```

ECGTwin 官方链路允许 reference condition 和 target condition 不同，但医学语义更难保证。
因此 cross-class 只作为 ablation，必须通过 semantic gate 和 digital gate 后才能进入下游。

### Promotion Rule

一个 token bank 只有满足以下条件才进入大规模生成和 Latent-Hull online AT：

```text
1. no-leak same-label style probe:
     target-token P(target center) > vanilla/wrong-center/PTB-XL-source
2. same-label C2ST:
     target-token real-vs-synth balanced accuracy <= controls
3. digital/semantic gate:
     target-token pass rate >= vanilla
4. class balance:
     NORM/MI/STTC each >= 40 accepted samples for ningbo/chapman/georgia;
     cpsc_2018 can report MI missing-primary limitation.
```

如果只满足 style probe、不满足 C2ST，则记录为：

```text
token changes center-probe score but may introduce synthetic artifacts.
```

不进入 downstream 主证据。

### 2026-05-03 Execution Status

本轮 token 已完成重训：

```text
PN2021 big4:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v40_pn2021_big4_direct_mv4_actual_steps2500/prompt_token_bank.pt
  final step: loss=0.04169, recon=0.04166, orth=0.25472, mean_delta=1.2367

PTB-XL source:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_k500_direct_mv4_actual_steps2500_20260503/prompt_token_bank.pt
  final step: loss=0.03234, recon=0.03232, orth=0.17076, mean_delta=1.3629

Diagnostics:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics/regen_20260503/token_summary.csv
  /root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics/regen_20260503/token_cosine.csv
```

诊断摘要：

```text
PN2021 final bank:
  tokens = 80
  delta_norm mean = 1.1217
  delta_norm min/max = 0.0002 / 1.6469
  cos_to_init mean = 0.5511

PTB-XL final bank:
  tokens = 20
  delta_norm mean = 1.3605
  delta_norm min/max = 1.1188 / 1.8568
  cos_to_init mean = 0.4861
```

PN2021 的 `cpsc_2018_HYP` 和 `cpsc_2018_MI` 四向量 token 基本没训练：

```text
cpsc_2018 HYP: selected anchors = 0
cpsc_2018 MI:  selected anchors = 0
delta_norm ~= 0.0002, cos_to_init ~= 1.0
```

因此后续 cpsc_2018 的主验证只看 `NORM/STTC/CD`，其中 `CD` 仍是 exploratory；
不能声称 cpsc_2018 的 MI/HYP token 已经学到目标中心风格。

### 2026-05-03 Effectiveness Pilot Status

第一轮有效性 pilot 已完成：

```text
script:
  scripts/ecgtwin_gen/run_center_token_effectiveness_pilot_20260503.sh

output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_20260503

report:
  docs/tmp_md/center_token_effectiveness_pilot_20260503.md
```

结论：

```text
NOT SUPPORTED as reliable center-style transfer yet.

cpsc_2018:
  target token shows the clearest full-10s 7-way style-probe advantage,
  but no-leak waveform and frozen-EfficientNet binary probes do not confirm it.

georgia:
  target token is positive after gate in full-10s 7-way style probe,
  but no-leak waveform and frozen-EfficientNet binary probes do not confirm it.

ningbo:
  weak positive but too small.

chapman_shaoxing:
  not supported.
```

因此 regenerated v40 token 不能作为 downstream AUROC/AUPRC 主证据，也不建议继续
盲目放大当前生成池。下一步如果继续 center-token 方向，应改训练目标本身：

```text
1. 在 token 训练时加入 frozen real-only center feature probe 的 style loss；
2. 同时加入 frozen super5 classifier semantic preservation loss；
3. 保留 current denoise MSE / init regularization / MV4 orthogonality；
4. 仍然用 K500 anchor exclusion + same-label validation 做验收。
```

### Generation For Validation

每个 center-class 用相同 reference、相同 disease prompt、相同 seed/DDPM steps 生成对照组：

```text
A. vanilla: diagnosis prompt only, no learned token
B. target token: diagnosis prompt + <target_center_CLASS>
C. wrong center: diagnosis prompt + <wrong_center_CLASS>
D. wrong class: diagnosis prompt + <target_center_OTHERCLASS>
E. PTB-XL source token: diagnosis prompt + <ptbxl_source_CLASS>
```

PN2021 中 `ptbxl_source` 是 source-style negative control；
PTB-XL 中 PN2021 token 是 target-shift negative control。

推荐第一轮生成规模：

```text
NORM/MI/STTC:
  n_per_class = 80 raw generations per center per ablation arm

CD/HYP:
  n_per_class = 20 smoke generations per center per ablation arm
```

第一轮只需要证明 token 的方向性，不需要马上把所有样本都放进 downstream augmentation。

### Acceptance Before Downstream Use

一个 token bank 可以进入 downstream Latent-Hull / online AT，必须满足：

```text
1. token_delta_norm 非零，cos_to_init 不塌缩到 1.0；
2. target-token digital gate pass rate 不低于 vanilla；
3. target-token frozen victim class consistency 不低于 vanilla；
4. target-token P(target center) > vanilla and wrong-center token；
5. target-token feature distance to held-out target real < vanilla；
6. C2ST(real_target, target-token) 不比 C2ST(real_target, vanilla) 更容易区分；
7. 下游 PN2021 7-center macro AUPRC 不下降，target center AUPRC 最好上升。
```

如果只满足 1-5，只能声称：

```text
prompt token shifts ECGTwin generation toward target-center style.
```

如果同时满足 6-7，才可以声称：

```text
prompt token improves target-center useful generation for EfficientNet1DV2.
```

## 与现有 CenterToken 的区别

现有实现：

```text
trash/cleanup_20260506_legacy/methods_ecgtwin_gen/center_token/model.py
trash/cleanup_20260506_legacy/methods_ecgtwin_gen/center_token/trainer_v2.py
trash/cleanup_20260506_legacy/scripts_ecgtwin_gen/train_center_token_v2.py
trash/cleanup_20260506_legacy/scripts_ecgtwin_gen/generate_center_synth.py
```

当前 token 是 256-d AdaLN hook：

```text
c = timestep_embed + ib_projector(base_vector) + center_token_256
```

它不进入 prompt，也不能写成自然语言 prompt token。新方案改为：

```text
text_embed_aug = concat(text_embed_diagnosis, center_class_token_768)
text_embed_mask_aug = concat(mask_diagnosis, 1)
```

`base_vector` 仍由 reference ECG 和原始诊断文本提取，不加入 center token，避免 center token 污染个体身份/形态基线。

## Token 设计

每个中心一个 token bank：

```text
CenterClassTokenBank(center):
  <center_CD>   -> 768-d
  <center_HYP>  -> 768-d
  <center_MI>   -> 768-d
  <center_NORM> -> 768-d
  <center_STTC> -> 768-d
```

Super5 类顺序固定：

```text
CD, HYP, MI, NORM, STTC
```

Token 初始化：

```text
token[class] = mean(nomic_embed(super5 fallback prompt for class))
```

推荐 fallback prompt：

| class | prompt | 生成可靠性 |
|---|---|---|
| NORM | `sinus rhythm|normal ecg.` | 数字验证 3/3，通过 |
| MI | `stemi|st elevation myocardial infarction|acute` | 数字验证 3/3，但多为 Q-wave path |
| STTC | `nstemi|non st elevation|t wave inversion` | 数字验证 3/3，通过 |
| HYP | `left ventricular hypertrophy|high voltage` | victim/视觉强，数字 LVH 电压门失败 |
| CD | `left bundle branch block|lbbb` | 当前数字验证 0/3，不作为主增强类 |

按当前确认，必须进一步核查 ECGTwin 作者官方 prompt/推理链路是否能跑通 `CD/HYP` 生成。
如果官方链路支持并能正常生成样本，则 `CD/HYP` token 保留；是否进入下游增强仍由
digital ECG gate 和 teacher/classifier gate 决定。

2026-05-01 smoke 结论：

```text
official author gallery includes:
  left_ventricular_hypertrophy
  left_bundle_branch_block
  right_bundle_branch_block
  atrioventricular_block

smoke output:
  /root/autodl-tmp/ecgtwin_smoke_authorprompt_hyp_cd_20260501/summary.json
  /root/autodl-tmp/ecgtwin_smoke_authorprompt_tensors_20260501/digital_gt_validation.md
```

HYP prompt `left ventricular hypertrophy|high voltage` 能生成并被 frozen super5 victim 判为
HYP (`p=0.976`)，但数字 LVH 电压 gate 未过，Sokolow 约 1.53 mV，低于 >3.5 mV。
CD prompts 能生成波形；`right bundle branch block|rbbb` 过了一次数字 RBBB gate，
但 three CD author prompts 的 frozen super5 victim CD 概率仍很低。因此 HYP/CD 可保留为
token 训练目标和 gated generation 目标，但不能不经 gate 直接进入 downstream augmentation。

## Reference / base_vector 要求

`base_vector` 是 ECGTwin 的 AdaLN 主调制条件，来自：

```text
ref_latent + ref_text_embed + ref_pat_info -> IBExtractor -> base_vector
```

要求：

- 目标类别生成优先使用同类或相近形态 reference。
- 固定 normal reference 只适合 prompt sanity，不适合最终生成。
- HYP/CD 失败主要来自 VAE 解码绝对电压偏低，不是简单 prompt 文本问题。
- `text_embed_mask` 不得全 0；如果 append token，token mask 必须为 1。

## 数据准备

每个目标中心准备一个 K=500 `.pt`：

```python
{
  "data": Tensor(4, 128),
  "label": {
    "center": "ningbo",
    "record_id": "...",
    "hr": float,
    "age": float,
    "sex": "M/F/U",
    "text": "diagnosis prompt without center token",
    "text_embed_base": Tensor(L, 768),
    "super5_multi_hot": Tensor(5),
    "primary_class": "MI",
    "token_names": ["<ningbo_MI>"]
  }
}
```

K=500 时采用混合采样：

```text
per-class floor = min(30, available)
remaining slots = natural distribution fill
```

如果某类样本不足：

- token 仍可初始化并保存；
- 不足类不进入主增强训练；
- 文档和 run config 记录 class_count gate。

固定第一批目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

这 K=500 条 ref ids 后续必须传给 fine-tune/validation/eval 的 exclusion 逻辑；
评测增强效果时不能让模型在 validation set 上看到自己的目标中心 ref 样本。

确认的 cache 方案：

```text
旧 /root/autodl-tmp/center_token_ablation/datasets/*_k500.pt
  -> 只作为历史 ablation，不作为新 prompt-token 主线输入。

新主线：
  /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/
    text_prompt_bank.pt
    center_full_latents/{center}.pt
    ref_selection/{center}_k500_seed42.json
```

原因：

- 旧 K500 prep 走 classifier 风格 filter + global z-score 后再 VAE encode；
  新主线需要更接近 ECGTwin VAE 训练语义的 raw-mV / ECGTwin lead-order latent cache。
- K=500 selection 应从 full latent cache 抽样，避免每次改 seed/floor 都重复 WFDB IO、预处理和 VAE encode。
- `ref_selection/*.json` 是后续 fine-tune validation/eval 的硬排除输入。

当前 cache 已完成：

```text
text_prompt_bank.pt: 58 SNOMED prompts, 5 class fallback prompts
center_full_latents/cpsc_2018.pt: 4746 latents
center_full_latents/ningbo.pt: 19796 latents
center_full_latents/chapman_shaoxing.pt: 5748 latents
center_full_latents/georgia.pt: 8719 latents
```

K=500 selection class counts:

| center | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|
| ningbo | 107 | 111 | 34 | 121 | 127 |
| chapman_shaoxing | 95 | 97 | 30 | 124 | 154 |
| cpsc_2018 | 259 | 0 | 0 | 116 | 125 |
| georgia | 111 | 110 | 7 | 124 | 148 |

Implication: `cpsc_2018` lacks HYP/MI in the selected/found super5-positive pool,
and `georgia` has only 7 MI refs. Token training must log these shortages and
avoid overclaiming those center-class tokens.

## 训练目标

冻结：

```text
ECGTwin VAE encoder/decoder
IBExtractor
DiT noise predictor
nomic text encoder
```

只训练：

```text
CenterClassTokenBank: 5 x 768 parameters per center
```

主损失：

```text
L_eps = MSE(noise_pred_with_center_token, noise)
```

约束：

```text
L_init = ||token_c - init_embed_c||^2
L_iso  = MSE(pred_with_token_on_PTBXL, pred_base_on_PTBXL)
L_sep  = mean(max(0, cos(token_i, token_j) - margin))
L_norm = norm/token-distribution projection to nomic embedding scale
```

多标签样本默认 append 所有阳性类 token；如果 class confusion 明显，退回 primary-only。

## 当前实现状态

新增实现：

```text
methods/ecgtwin_gen/prompt_token/model.py
methods/ecgtwin_gen/prompt_token/trainer.py
scripts/ecgtwin_gen/train_center_prompt_tokens.py
scripts/ecgtwin_gen/build_prompt_token_latent_cache.py
scripts/ecgtwin_gen/gate_prompt_token_synth.py
```

当前第一版训练语义：

```text
base diagnosis text_embed = SNOMED prompt embedding or class fallback embedding
center token            = CenterClassPromptTokenBank[center, primary_class]
text_embed_aug          = concat(base diagnosis text_embed, center token)
text_embed_mask_aug     = concat(base mask, 1)
base_vector             = IBExtractor(ref_latent, base diagnosis text_embed, pat_info)
trainable params        = centers x 5 x 768 only
```

也就是说，生成中心 A 的 MI 类 ECG 时，工程上等价于把 `<A_MI>` 追加到 prompt，
但 `<A_MI>` 由 prompt compiler 映射为一个可学习 768-d embedding，而不是让
`bert-base-uncased` / `nomic-ai/nomic-embed-text-v1.5` 学新词。

## Center Token v2 Improvement Plan

目标不是只让 ECGTwin “能生成”，而是让生成池经过 gate 后能稳定提升
EfficientNet1DV2 的 PN2021 AUROC/AUPRC。当前 v1 的主要问题：

```text
1. token 训练是普通 shuffle，中心/类别不均衡会稀释少数类 token。
2. 单个 768-d token 在 ECGTwin cross-attention 中权重偏弱。
3. 生成池 gate 后类别不平衡，STTC 尤其容易成为瓶颈。
4. clean AUPRC 有小幅收益，但 PN2021-C robustness drop 未稳定改善。
```

v2 改进：

```text
token_repeat:
  参考 multi-vector textual inversion，把同一个 <center_CLASS> embedding
  重复追加到 text_embed 多个位置；默认试 token_repeat=4。

sample_strategy:
  train_center_prompt_tokens.py 新增 center_class_balanced sampler。
  对每个非空 center-class cell 做 inverse-count 采样，避免 NORM/STTC
  大类或某个中心主导 token 梯度。

gate-driven pool building:
  不固定每类只生成 50 条；按 gate pass 缺口 over-generate。
  第一轮目标至少 NORM/MI/STTC 每类 >= 40 gated samples。

downstream acceptance:
  先跑 20 epoch pilot。
  如果 clean PN2021 AUPRC >= 0.5560 或 PN2021-C drop 明显优于 v1，
  再扩大到更多中心/更长训练。
```

当前已新增 CLI：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --sample_strategy center_class_balanced \
  --token_repeat 4

/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
  --token_repeat 4
```

注意：`token_repeat` 只是 prompt compiler 中重复同一个 learnable embedding；
不会改 tokenizer，不会改 ECGTwin checkpoint，也不会污染 IBE `base_vector`。

v2/v3 执行结果：

| token bank | train setting | generation setting | gated result | downstream PN2021 |
|---|---|---|---|---:|
| v2 | balanced + repeat4, 3000 steps | repeat4 | 85/150: NORM 31, MI 35, STTC 19 | not run |
| v2 | balanced + repeat4, 3000 steps | repeat2 smoke | 38/60: NORM 20, MI 10, STTC 8 | not run |
| v3 | balanced + repeat2, 2500 steps | repeat2 + boost/merge | 153 total: NORM 69, MI 41, STTC 43 | 0.8345 / 0.5556 |

Interpretation:

- `center_class_balanced` training is useful for rare center-class coverage, but
  larger `token_repeat` can over-steer samples toward STTC-like morphology.
- `token_repeat=2` is a better matched setting than `repeat4` for current ECGTwin.
- The best pure prompt-token downstream result remains the v4 balanced pool from
  v1 tokens, not the stronger repeat-token banks.
- Do not blindly increase token strength; future token improvements should add
  quality-aware generation/reference selection instead of only increasing
  cross-attention weight.

已完成 smoke：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/smoke_ningbo_steps3/
/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/smoke_all4_bf16_steps5/
```

结论：四中心 K=500 cache、bf16 autocast、DataLoader workers=2 的最小训练闭环已跑通；
token delta norm 非零，说明 ECGTwin text path 到 token bank 的梯度可用。

正式四中心训练已完成：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4_steps2000/
steps=2000, batch_size=16, workers=4, amp=bf16
elapsed ~= 87s on RTX 4090D
final logged loss=0.04253
final mean token_delta_norm=1.1961
```

保存：

```text
prompt_token_bank.pt
metrics.jsonl
run_config.json
```

重要观测：

- `cpsc_2018` 的 HYP/MI 选中样本数为 0，因此 `<cpsc_2018_HYP>` 和
  `<cpsc_2018_MI>` 基本保持初始化，delta norm 约 0。
- `georgia` MI 只有 7 条，`<georgia_MI>` 可训练但必须在生成阶段单独 gate。
- NORM/MI/STTC 仍是第一轮 downstream augmentation 主类；HYP/CD 只走 gated path。

推荐正式训练入口，按 4090D/80G/15 核优化：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --total_steps 2000 \
  --batch_size 16 \
  --num_workers 4 \
  --amp_dtype bf16 \
  --save_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4
```

## 验证

最低验证集：

1. Token norms、cosine matrix、与初始化 class embedding 的 cosine。
2. held-out target-center denoise MSE：with-token vs no-token。
3. PTB-XL isolation delta：center token 不应破坏 generic ECGTwin 行为。
4. 生成质量：NaN/Inf、flatline、幅值、Einthoven residual、HR/QRS。
5. 语义一致性：冻结 EfficientNet super5 victim，目标类概率和 NORM abnormal suppression。
6. 下游：目标中心 AUROC/AUPRC delta、7-center macro delta、per-class delta。

Current diagnostic export:

```text
script: scripts/ecgtwin_gen/export_prompt_token_diagnostics.py
out_dir: /root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics
combined:
  token_summary.csv  # 530 token rows across 14 banks
  token_cosine.csv   # 15276 pairwise cosine rows
per-bank:
  *.token_summary.csv
  *.token_cosine.csv
```

This closes the lightweight token norm/cosine TODO. Held-out denoise and
generation-sanity JSON remain separate validation tasks because they require
ECGTwin forward/generation rather than just reading token banks.

### Prompt-Token Generation Smoke

新增生成入口：

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth.py
```

该脚本执行：

```text
base diagnosis text_embed + learned <center_CLASS> token -> ECGTwin target text path
ref latent + base diagnosis text_embed + pat_info -> IBE base_vector
DDPM latent -> VAE decode -> classifier-ready PTBXL-order 1000 samples
```

已完成 `ningbo` 5-class smoke：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/generate_center_prompt_token_synth.py \
  --center ningbo \
  --classes NORM MI STTC HYP CD \
  --n_per_class 1 \
  --steps 25 \
  --out_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_smoke_v1
```

输出：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_smoke_v1/ningbo/samples.npz
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_smoke_v1/ningbo/summary.json
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_smoke_v1/ningbo/digital_gate_summary.json
```

Smoke 结果：

| class | victim top1 | p_target | digital pass | notes |
|---|---|---:|:---:|---|
| NORM | NORM | 0.901 | yes | generation path works |
| MI | STTC | 0.385 | yes | digital MI criteria pass, victim class confusion remains |
| STTC | CD | 0.790 | yes | digital STTC criteria pass, victim top1 mismatch |
| HYP | STTC | 0.589 | no | still fails voltage gate |
| CD | CD | 0.945 | no | QRS broadest about 117 ms, below 120 ms gate |

Interpretation:

- The prompt-token compiler and ECGTwin generation path are functional.
- NORM/MI/STTC can enter the next gated generation pool after multi-seed digital
  and teacher/victim filtering.
- HYP/CD remain gated-only and should not enter downstream augmentation without
  class-specific digital pass.

### Gated Pool And Downstream Pilot

新增 gated export 入口：

```text
scripts/ecgtwin_gen/gate_prompt_token_synth.py
```

该脚本读取 `generate_center_prompt_token_synth.py` 产生的 `samples.npz` 和
`summary.json`，执行：

```text
class-specific digital ECG gate
frozen victim p_target gate
optional top1 gate
K=500 ref-id exclusion metadata export
```

并输出 online AT 可直接读取的文件：

```text
gated_samples.npz
gated_samples.latent.npz
gated_samples.class_trust.json
gated_samples.ref_meta.json
gate_report.json
```

`ningbo` prompt-token pool v2 已完成：

```text
source: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/
generated: 50 NORM + 50 MI + 50 STTC
gate: min_target_prob=0.30, digital gate required
passed: 97 / 150
  NORM: 41 / 50
  MI:   41 / 50
  STTC: 15 / 50
```

下游 Latent-Hull online AT pilot：

```text
output: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20/
input pool: gated_samples.latent.npz, n=97
M=10, lambda=0.25, hull_steps=5, K_anchor=45, n_epochs=20
K=500 ningbo ref ids excluded from quick eval
```

结果：

| model | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | delta vs baseline |
|---|---:|---:|---:|
| baseline | 0.9064 / 0.7754 | 0.8344 / 0.5526 | - |
| prompt-token gated pilot | 0.9068 / 0.7772 | 0.8342 / 0.5555 | -0.0002 / +0.0029 |
| real-anchor Latent-Hull best AUPRC | 0.9064 / 0.7765 | 0.8341 / 0.5560 | -0.0003 / +0.0034 |

Interpretation:

- Prompt-token 生成样本已接通：generation -> digital/victim gate -> `.latent.npz`
  export -> Latent-Hull online AT -> PN2021 clean eval。
- clean AUPRC 有小幅正向信号，但 STTC gate pass 只有 15/50，是当前样本池主要瓶颈。
- PN2021-C robustness 未显著改善；要作为主结果需要扩大/均衡 gated pool，尤其补 STTC。

Balanced v4 pool 已构建：

```text
extra STTC source:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v3_sttc_boost/ningbo/
  generated STTC=100, passed STTC=34
merged pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated/
counts:
  NORM=41, MI=41, STTC=49, total=131
```

v4 downstream pilot 已完成：

```text
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20/
settings:
  M=10, lambda=0.25, hull_steps=5, K_anchor=90, n_epochs=20
PN2021 v3:
  0.8344 / 0.5558
```

Additional downstream attempts:

| run | pool | setting | PN2021 AUROC/AUPRC | interpretation |
|---|---|---|---:|---|
| v3 repeat2 | token-v3 gated153 | M=10, K_anchor=120 | 0.8345 / 0.5556 | better AUROC, lower AUPRC |
| v4 M5 adv025 | token-v1 v4 gated131 | M=5, adv_weight=0.25 | 0.8343 / 0.5555 | quick eval improved, full eval did not |
| hybrid real+prompt | nin real200 + prompt v4 131 | M=10, K_anchor=150 | 0.8345 / 0.5549 | stable gates, but full AUPRC lower |
| hybrid real+prompt MI-only | nin real200 + prompt MI40 | M=10, K_anchor=150 | 0.8343 / 0.5540 | MI-only prompt supplementation is worse than balanced prompt pool |
| hybrid source-aware | nin real200 + prompt v4 131 | M=10, K_anchor=150, real=1.0/prompt=0.35, MI prompt=1.0 | 0.8342 / 0.5552 | better than naive hybrid, still below pure prompt v4 |
| quality top40 | mixed prompt candidates, 40/class | M=10, K_anchor=120 | 0.8343 / 0.5551 | high-confidence-only selection loses useful boundary samples |
| mixed high/boundary v11 | candidates from v2/v3/v5/v6/v7, 50/class | high 40%, boundary 40%, diverse 20% | 0.8345 / 0.5547 | too many boundary samples; AUROC ok, AUPRC lower |
| mixed high/boundary v12 | candidates from v2/v3/v5/v6/v7, 50/class | high 60%, boundary 20%, diverse 20% | 0.8342 / 0.5542 | lowering boundary did not recover AUPRC |
| v4 seed rerun | v4 balanced | M=10, K_anchor=90, seed=20260513 | 0.8343 / 0.5546 | v4 best is not a stable seed-only effect |
| v4 adv035 | v4 balanced | M=10, K_anchor=90, adv_weight=0.35 | 0.8342 / 0.5554 | closer, but below original v4 |
| v4 adv045 | v4 balanced | M=10, K_anchor=90, adv_weight=0.45 | 0.8340 / 0.5535 | worse than original v4 |
| fresh v1 quality50 | regenerated v1-token candidates, 50/class | NORM58/MI85/STTC50 candidates -> top50/class | 0.8343 / 0.5548 | fresh random expansion does not beat v4 |
| v20 quality-ref factorized/direct selected | K500 seed1042 quality refs, factorized/direct MV4, top50/class selected | M=10, K_anchor=90, seed=20260505 | 0.8335 / 0.5543 | gate pool grew, but full 7-center generalization fell |
| v21 quality-ref direct all-gated | K500 seed1042 quality refs, direct MV4, all gated 202 | M=10, K_anchor=90, seed=20260503 | 0.8340 / 0.5546 | more diverse than v20, still below v4 |
| v27 seed42 direct MV4 balanced | old K500 seed42 refs, direct MV4, targeted MI/STTC boost, 50/class | M=10, K_anchor=90, seed=20260503 | 0.8342 / 0.5558 | matches v4 AUPRC, slightly lower AUROC; best new MV4 route so far |
| v31 seed42 MV4 mixed diverse | selected across v24/v25/v26/v28/v29, 50/class | M=10, K_anchor=90, seed=20260503 | 0.8340 / 0.5550 | diversity-preserving selector did not beat v4/v27; PN2021-C mean drop 0.022597/0.034711 |

Current conclusion:

- Best pure prompt-token result is `v4 balanced M10`: PN2021 `0.8344 / 0.5558`.
- Best overall AUPRC remains real-anchor Latent-Hull `nin_M10`: `0.8341 / 0.5560`.
- Center-token samples are useful enough to reach the same gain band, but the
  current improvement ceiling is limited by gate-selected sample quality and
  downstream distribution shift.
- High-confidence-only selection is not enough; online AT appears to need some
  boundary-informative samples, not only the most classifier-obvious generated ECG.
- MI-only prompt supplementation is also not enough; it reduces full PN2021 AUPRC
  despite acceptable quick-eval behavior, so the next attempt should not keep
  adding a single disease class without controlling source/class sampling.
- Source-aware hybrid sampling partially fixes naive concatenation (`0.5549` to
  `0.5552`) but remains below the pure balanced prompt-token pool. Keep it as a
  reusable ablation/tool, not as the current best center-token route.
- Mixed high-confidence/boundary selection from the existing candidate pools did
  not improve over v4. More boundary samples increased ASR but lowered full
  PN2021 AUPRC.
- `adv_weight` and seed-only reruns also did not beat the original v4 balanced
  run. The current bottleneck is upstream sample generation/reference selection,
  not downstream Latent-Hull hyperparameters.
- Fresh random expansion with the same v1 token bank produced enough NORM/MI/STTC
  gated samples after targeted MI/STTC boosts, but the downstream result still
  stayed below v4. More samples from the same generator distribution are not
  sufficient by themselves.
- The first quality-ref attempt (`seed1042`) improved some gated counts, but
  its downstream full PN2021 result underperformed. The old seed42 reference
  selection remains a stronger anchor for the current ECGTwin generator.
- Direct multi-vector tokens can recover the v4 AUPRC band when combined with
  class-targeted over-generation (`v27`), but this is not yet a clear win over
  the simpler v1 token bank.
- The v31 diversity-preserving selector was evaluated on PN2021-C at user
  request. It slightly improves self-clean mean corruption drop versus
  prompt-token v4, but its clean PN2021 AUPRC is lower. Real-anchor Latent-Hull
  remains better on absolute corrupted AUPRC, while not reducing self-clean
  relative drop.
- Target-center-only token training overfits the prompt token: `ningbo`-only
  direct MV4 at 3000 steps gated only 71/240, and the 1500-step version gated
  91/240 with STTC only 14/80. Training all four centers is still preferred.
- Next useful center-token improvement should not be another blind pool expansion.
  Add gate-aware token checkpointing/selection: periodically export token banks,
  generate a small held-out NORM/MI/STTC probe, score by digital+victim gates and
  diversity, then only run downstream for token checkpoints that beat the v4
  gated profile. Keep `v4 balanced M10` and `v27 seed42 direct MV4 balanced`
  as comparison baselines.

### 2026-05-01 Multi-Vector / Quality-Ref Experiments

New implementation supports two prompt-token schemas:

```text
direct:
  embeddings[center, class, vector, 768]

factorized:
  center_embeddings[center, center_vector, 768]
  class_embeddings[class, class_vector, 768]
  residual_embeddings[center, class, residual_vector, 768]
```

Generation supports legacy v1 banks `(center,class,768)`, direct MV4 banks, and
factorized banks. Compatibility smoke passed for all three formats.

Reference mining script:

```text
scripts/ecgtwin_gen/quality_ref_selection.py
```

Quality K500 selection `seed1042` counts:

| center | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|
| ningbo | 82 | 80 | 120 | 80 | 138 |
| chapman_shaoxing | 111 | 89 | 40 | 80 | 180 |
| cpsc_2018 | 261 | 0 | 0 | 80 | 159 |
| georgia | 107 | 101 | 7 | 80 | 205 |

Token/gate results:

| token run | ref selection | gated result |
|---|---|---|
| v4 quality-ref factorized c1/class1/res2 | seed1042 quality | 98/240: NORM 45, MI 25, STTC 28 |
| v5 quality-ref direct MV4 | seed1042 quality | 105/240: NORM 37, MI 22, STTC 46 |
| v19 direct MV4 NORM/MI boost | seed1042 quality | 97/240: NORM 54, MI 43 |
| v21 all-gated merge | seed1042 quality | 202 total: NORM 91, MI 65, STTC 46 |
| v6 ningbo-only direct MV4 3000 | seed1042 quality | 71/240: NORM 32, MI 19, STTC 20 |
| v7 ningbo-only direct MV4 1500 | seed1042 quality | 91/240: NORM 47, MI 30, STTC 14 |
| v8 seed42 direct MV4 | old seed42 refs | 108/240: NORM 52, MI 32, STTC 24 |
| v25 seed42 STTC boost | old seed42 refs | 56/160 STTC |
| v26 seed42 MI boost | old seed42 refs | 75/160 MI |
| v27 seed42 direct MV4 balanced | old seed42 refs | selected 150: NORM 50, MI 50, STTC 50 |
| v28 class-checkpoint hybrid | old seed42 refs | 119/240: NORM 43, MI 42, STTC 34 |
| v30 class-checkpoint hybrid balanced | old seed42 refs | selected 150: NORM 50, MI 50, STTC 50 |
| v31 mixed diverse q50 | v24/v25/v26/v28/v29 candidates | selected 150: NORM 50, MI 50, STTC 50 |

Important negative result: quick-eval improvements are not sufficient. Both v20
and v21 had encouraging quick-eval curves, but full PN2021 did not beat v4.
Full seven-center eval remains the acceptance criterion.

Gate-aware checkpoint probe:

```text
run:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v9_seed42_direct_mv4_ckptprobe_steps2500
new trainer arg:
  --save_every 500
probe:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/probe_v9_seed42_direct_mv4_ckpts
```

Probe counts from 20 samples/class:

| checkpoint | total | NORM | MI | STTC |
|---|---:|---:|---:|---:|
| step500 | 25/60 | 12 | 4 | 9 |
| step1000 | 25/60 | 13 | 8 | 4 |
| step1500 | 24/60 | 13 | 6 | 5 |
| step2000 | 19/60 | 7 | 8 | 4 |
| final | 24/60 | 14 | 6 | 4 |

Class-specific checkpoint composition:

```text
script:
  scripts/ecgtwin_gen/compose_prompt_token_bank_by_class.py
hybrid bank:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v10_seed42_direct_mv4_class_ckpt_hybrid/prompt_token_bank.pt
policy:
  NORM = final
  MI   = step1000
  STTC = step500
```

This improved one-shot gate to `119/240`, but the balanced downstream run
`v30` still evaluated at `0.8342 / 0.5546`. Therefore gate-aware token checkpoint
selection is useful for generation diagnostics, but the current top50/class
selection policy still over-selects samples that do not improve full PN2021.
The next checkpoint-aware step should preserve more diversity/boundary coverage
or use full validation probes before committing to downstream training.

Current active downstream test for that recommendation:

```text
pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v31_seed42_mv4_mixed_diverse_q50/ningbo/gated
selector:
  high_frac=0.30, boundary_frac=0.25, diverse remainder=0.45
downstream:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v16/ningbo_v31_mixed_diverse_q50_M10_ep20_seed03
status:
  complete; 500 ningbo K-ref ids excluded from quick eval
result:
  quick AUROC/AUPRC = 0.8285 / 0.6448
  full PN2021 v3 = 0.8340 / 0.5550
decision:
  do not run PN2021-C for v31; move compute to Latent-Hull coefficient ablations
```

## 输出

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/<run_name>/
  run_config.json
  metrics.jsonl
  prompt_token_bank.pt
  token_cosine.csv
  heldout_denoise.json
  generation_sanity.json
  samples/*.npz
  figures/*.png
```

## 实现 TODO

- DONE: 新增 `CenterClassTokenBank(centers,5,768)`。
- DONE: 新增训练侧 prompt compiler：把 `<center_CLASS>` 语义映射为可学习 embedding，并 append 到 `text_embed`。
- DONE: 新增 raw-mV full latent cache + K=500 selection，保留 `CD/HYP` token 训练记录，不在 prep 阶段直接过滤。
- DONE: 新增训练脚本，冻结 ECGTwin，仅优化 token bank。
- DONE: 新增生成脚本，支持 `prompt + <center_CLASS>`，并输出 classifier-ready `.npz` 和 raw latent `.npz`。
- DONE: 新增 gated downstream export，并完成 `ningbo` NORM/MI/STTC pilot。
- DONE: token norm/cosine diagnostics export。
- RUNNING: multi-center prompt-token generation/gating pilot v32 for
  `chapman_shaoxing`, `cpsc_2018`, and `georgia`, NORM/MI/STTC each 50.
- TODO: held-out denoise MSE、larger balanced gated pool downstream pilot。

### Multi-Center Prompt-Token v32 Pilot

目的：补齐 `ningbo` 之外三个目标中心的 prompt-token 生成和 gate 结果，判断
center-class token 是否能跨中心稳定产生 NORM/MI/STTC 可用样本。

脚本：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/ecgtwin_gen/run_multicenter_prompt_token_v32.sh
```

配置：

```text
token bank:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v8_seed42_direct_mv4_steps2500/prompt_token_bank.pt
centers:
  chapman_shaoxing, cpsc_2018, georgia
classes:
  NORM, MI, STTC
n_per_class:
  50
DDPM steps:
  25
gate:
  digital ECG gate + victim p_target >= 0.30, no top1 hard requirement
output root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v32_seed42_direct_mv4_multicenter_q50/
logs:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/logs/v32_<center>_generate.log
  /root/autodl-tmp/ecgtwin_prompt_token_super5/logs/v32_<center>_gate.log
```

完成后读取每个中心的：

```text
gated/gate_report.json
gated/gated_samples.latent.npz
```

判定：

- 如果每个中心 NORM/MI/STTC 至少各 30 条通过 gate，可进入 multi-center
  prompt-token Latent-Hull downstream pilot。
- 如果某中心或某类 gate collapse，先回到 reference/token checkpoint probe，
  不直接做下游训练。

实际结果：

| center | generated | gated total | NORM | MI | STTC | decision |
|---|---:|---:|---:|---:|---:|---|
| `chapman_shaoxing` | 150 | 90 | 38/50 | 33/50 | 19/50 | STTC boost needed |
| `cpsc_2018` | 100 | 59 | 38/50 | 0/0 | 21/50 | no MI primary refs in v3 cache; STTC boost only |
| `georgia` | 150 | 78 | 38/50 | 28/50 | 12/50 | MI/STTC boost needed |

`cpsc_2018` 当前 v3 primary-class cache 没有 MI：

```text
full cache primary counts: CD=2797, STTC=1031, NORM=918
seed42 K500 counts: CD=259, STTC=125, NORM=116
seed1042 K500 counts: CD=261, STTC=159, NORM=80
```

因此 cpsc 的 MI prompt-token 不能靠当前 primary-class reference selection
生成；后续需要引入 multi-label/SNOMED fallback，或承认该中心 MI 类不进入
first multi-center prompt-token downstream。

### Active Multi-Center Prompt-Token v33 Boost

目的：只补 v32 中不足的类，再合并成 v34 multi-center gated pool。

脚本：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/ecgtwin_gen/run_multicenter_prompt_token_v33_boost.sh
```

补生成：

| center | classes | n_per_class | reason |
|---|---|---:|---|
| `chapman_shaoxing` | STTC | 80 | v32 STTC only 19/50 |
| `cpsc_2018` | STTC | 80 | v32 STTC only 21/50; no MI refs |
| `georgia` | MI, STTC | 80 | v32 MI 28/50 and STTC 12/50 |

输出：

```text
boost root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v33_seed42_direct_mv4_multicenter_boost/
merged root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged/
```

进入 downstream 的条件：

- `chapman_shaoxing`: NORM/MI/STTC each >= 30 after v34 merge.
- `georgia`: NORM/MI/STTC each >= 30 after v34 merge.
- `cpsc_2018`: NORM/STTC each >= 30; MI 缺失单独记录为 label/cache limitation。

v33/v34 result:

| center | merged gated counts | status |
|---|---|---|
| `chapman_shaoxing` | NORM=38, MI=33, STTC=44 | ready |
| `cpsc_2018` | NORM=38, STTC=59, MI=0 | ready except MI primary-label limitation |
| `georgia` | NORM=38, MI=74, STTC=25 | needs one more STTC boost |

Active v35:

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/ecgtwin_gen/run_georgia_prompt_token_v35_sttc_boost.sh
```

Purpose: generate another 80 `georgia` STTC samples with a different seed and
merge v34+v35 into:

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v36_seed42_direct_mv4_multicenter_merged2/
```

v36 result:

| center | gated count | class counts | downstream eligibility |
|---|---:|---|---|
| `chapman_shaoxing` | 115 | NORM=38, MI=33, STTC=44 | yes |
| `cpsc_2018` | 97 | NORM=38, STTC=59, MI=0 | partial; MI absent in v3 primary labels |
| `georgia` | 149 | NORM=38, MI=74, STTC=37 | yes |

### Active Multi-Center Prompt-Token v36 Online-AT Pilot

脚本：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/ecgtwin_gen/run_multicenter_prompt_token_v36_online_at.sh
```

配置：

```text
pool root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v36_seed42_direct_mv4_multicenter_merged2/
output root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_multicenter_v36/
centers:
  chapman_shaoxing, cpsc_2018, georgia
attack:
  latent_hull, M=10, lambda=0.25, hull_steps=5
training:
  20 epochs, batch_size=128, num_workers=8, seed=20260503
eval:
  full PN2021 v3 after each run
```

Success criterion:

- Any one center-specific v36 pilot beats prompt-token v4 or real-anchor main
  on PN2021 v3 AUPRC.
- If clean PN2021 AUPRC is competitive, run PN2021-C only for the winner.

Actual result:

| run | v36 pool counts | best quick AUROC/AUPRC | full PN2021 AUROC/AUPRC | decision |
|---|---|---:|---:|---|
| `chapman_v36_M10_ep20_seed03` | NORM=38, MI=33, STTC=44 | 0.8398 / 0.6426 | 0.8344 / 0.5554 | below prompt-v4/real-anchor; no PN2021-C |
| `cpsc_v36_M10_ep20_seed03` | NORM=38, STTC=59, MI=0 | 0.8510 / 0.6667 | 0.8340 / 0.5553 | below prompt-v4/real-anchor; no PN2021-C |
| `georgia_v36_M10_ep20_seed03` | NORM=38, MI=74, STTC=37 | 0.8462 / 0.6741 | 0.8343 / 0.5541 | below prompt-v4/real-anchor; no PN2021-C |

Interpretation:

- Multi-center prompt-token generation works mechanically and can create gated
  pools for `chapman_shaoxing`, `cpsc_2018`, and `georgia`.
- Downstream utility did not improve beyond the best `ningbo` prompt-token pool
  or real-anchor Latent-Hull.
- `cpsc_2018` exposes a label/cache limitation: the current v3 primary-label
  cache has no MI references, so MI prompt-token generation requires a
  multi-label/SNOMED fallback, not another sampling seed.
- Next improvement should target reference selection/token objective and
  held-out generation probes before more downstream online-AT sweeps.

### 2026-05-02 Boundary-Confidence Target-Center Test

After the PTB-XL source-style prompt-token boundary AT ablation produced a
small positive result, the same boundary acceptance idea was tested on the
large PN2021 target-center prompt-token pools:

```text
boundary rule:
  push adversarial sample only if target sigmoid probability is in [0.50, 0.60]

shared settings:
  attack_mode = latent_hull
  M = 10
  lambda = 0.25
  hull_steps = 5
  adv_weight = 0.10
  anchor_lambda = 0.10
  seed = 20260503
  full eval excludes the center's K=500 ref ids via --exclude_ref_ids

runner:
  scripts/ecgtwin_gen/run_bigcenter_prompt_token_boundary_at_v1.sh

output root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/boundary_at_bigcenters_v1/
```

Pools:

| run | pool | classes in scope | gated counts |
|---|---|---|---|
| `ningbo_v4_boundary` | v4 balanced | NORM/MI/STTC | NORM=41, MI=41, STTC=49 |
| `chapman_v34_boundary` | v34 merged | NORM/MI/STTC | NORM=38, MI=33, STTC=44 |
| `cpsc_v34_boundary` | v34 merged | NORM/STTC | NORM=38, STTC=59 |
| `georgia_v36_boundary` | v36 merged2 | NORM/MI/STTC | NORM=38, MI=74, STTC=37 |

Full PN2021 v3 result:

| model | PTB-XL AUROC | PTB-XL AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|
| baseline | 0.9064 | 0.7754 | 0.8344 | 0.5526 |
| PTB-XL source-token boundary AT | 0.9067 | 0.7765 | 0.8345 | 0.5549 |
| `ningbo_v4_boundary` | 0.9066 | 0.7761 | 0.8344 | 0.5540 |
| `chapman_v34_boundary` | 0.9069 | 0.7763 | 0.8351 | 0.5517 |
| `cpsc_v34_boundary` | 0.9066 | 0.7761 | 0.8336 | 0.5523 |
| `georgia_v36_boundary` | 0.9068 | 0.7767 | 0.8339 | 0.5542 |
| `ningbo_v4_nonboundary` | 0.9068 | 0.7769 | 0.8344 | 0.5558 |
| `chapman_v36_nonboundary` | 0.9068 | 0.7764 | 0.8344 | 0.5554 |
| `cpsc_v36_nonboundary` | 0.9066 | 0.7765 | 0.8340 | 0.5553 |
| `georgia_v36_nonboundary` | 0.9069 | 0.7769 | 0.8343 | 0.5541 |

Interpretation:

```text
Boundary-confidence acceptance works as a conservative source-style ablation,
but it does not improve the large PN2021 target-center prompt-token runs.
The target-center boundary runs are generally below their non-boundary versions.

The likely issue is that p in [0.50, 0.60] over-filters useful prompt-token
anchors and changes the target-center sample mix; for the target center itself,
K=500 ref-id exclusion also makes the center-specific delta harder to interpret,
but the full 7-center average is still not competitive.
```

Decision:

```text
Do not make boundary-confidence prompt-token AT the target-center mainline.
Keep it as a PTB-XL source-style positive ablation. For PN2021 target centers,
prefer the earlier non-boundary prompt-token v4/v36 runs or real-anchor
Latent-Hull TA-OMAT.
```

### 2026-05-03 Task-1-Gated Token-Scale Test

Problem found:

```text
The large v42 pilot originally used the old 250-sample/crop victim inside the
generation-time teacher gate. That made p_target/gated samples inconsistent with
the new Task-1 full10 EfficientNet baseline.
```

Code changes:

```text
scripts/ecgtwin_gen/generate_center_prompt_token_synth.py:
  add --victim_crop_len
  add --token_scale

scripts/ecgtwin_gen/run_center_token_effectiveness_pilot_20260503.sh:
  add GEN_VICTIM_CKPT
  add GEN_VICTIM_CROP_LEN
  add GEN_TOKEN_SCALE
  add CENTERS override
```

`token_scale` is defined as:

```text
token_scaled = token_init + scale * (token_learned - token_init)
```

This is important: it weakens the learned center-style delta without destroying
the class-text initialization.

Task-1-gated ningbo pilot:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_n80_20260503
victim gate:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt
  crop_len = 1000
classes:
  NORM, MI, STTC
n_per_class:
  80
```

Full-strength target token (`scale=1.0`) had high target-center style score but
worse realism/C2ST after gating:

| arm | gated count | mean ningbo style prob | no-leak EffNet C2ST bacc | interpretation |
|---|---:|---:|---:|---|
| vanilla | 130 | 0.0823 | 0.7725 | realistic control |
| wrong-center token | 132 | 0.1617 | 0.8222 | worse C2ST |
| PTB-XL source token | 138 | 0.0826 | 0.7867 | worse than vanilla |
| target token scale 1.0 | 127 | 0.4038 | 0.8113 | style shift but synthetic artifacts |

Token-strength sweep:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_20260503
```

| arm | gated count | NORM/MI/STTC | mean ningbo style prob | no-leak EffNet C2ST bacc | decision |
|---|---:|---:|---:|---:|---|
| target token scale 0.50 | 163 | 77/56/30 | 0.2867 | 0.7702 | promote to conservative AT |
| target token scale 0.25 | 137 | 55/57/25 | 0.1793 | 0.8295 | too weak style and worse C2ST |

Interpretation:

```text
scale=0.50 is the first Task-1-gated target-token setting that both:
  1. raises the independent no-leak ningbo style score well above controls, and
  2. does not worsen no-leak same-label EfficientNet C2ST versus vanilla.

This is evidence that center token can improve target-center style when the
token strength is moderated. Full-strength token over-steers and creates
detectable synthetic artifacts.
```

Next center-token search should use:

```text
token_scale = 0.35, 0.50, 0.65
target center first: ningbo
promotion gate:
  style score > vanilla/wrong-center/PTB-XL-source
  C2ST bacc <= best control + 0.01
  class balance has at least 25 usable NORM/MI/STTC samples
```

## 2026-05-03 Other-Center Task-1-Gated Token-Scale Pilot

After the ningbo `scale=0.50` success, repeated the Task-1-gated center-token
effectiveness pilot on additional PN2021 centers:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_othercenters_scale05_n60_20260503
centers:
  chapman_shaoxing, georgia, cpsc_2018
token_scale:
  0.50
n_per_class:
  60
victim gate:
  Task-1 full10 EfficientNet, crop_len=1000
```

Gated counts:

| center | arm | passed/total | class counts |
|---|---|---:|---|
| chapman_shaoxing | target token | 117/180 | MI=43, NORM=52, STTC=22 |
| chapman_shaoxing | vanilla | 98/180 | MI=42, NORM=37, STTC=19 |
| cpsc_2018 | target token | 104/120 | NORM=55, STTC=49 |
| cpsc_2018 | vanilla | 60/120 | NORM=33, STTC=27 |
| georgia | target token | 100/180 | MI=35, NORM=46, STTC=19 |
| georgia | vanilla | 105/180 | MI=46, NORM=36, STTC=23 |

No-leak EfficientNet same-label C2ST and full10 style summary:

| center | arm | mean style prob | EffNet C2ST bacc | decision |
|---|---|---:|---:|---|
| chapman_shaoxing | target token | 0.1492 | 0.7622 | style up, but C2ST worse than vanilla |
| chapman_shaoxing | vanilla | 0.0675 | 0.7236 | more realistic control |
| cpsc_2018 | target token | 0.5657 | 0.7749 | style/gate up, but no MI refs; not full super5 pool |
| cpsc_2018 | vanilla | 0.1716 | 0.6571 | more realistic control |
| georgia | target token | 0.6096 | 0.8087 | fails C2ST; do not promote |
| georgia | vanilla | 0.6554 | 0.7286 | target token does not improve style |

Because chapman was closest, a lower-strength `token_scale=0.35` pilot was run:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_chapman_scale035_n60_20260503
center:
  chapman_shaoxing
```

| arm | passed/total | mean style prob | EffNet C2ST bacc | decision |
|---|---:|---:|---:|---|
| target token scale 0.35 | 117/180 | 0.1370 | 0.8000 | fails no-leak EffNet C2ST |
| vanilla | 98/180 | 0.0675 | 0.7904 | control |
| wrong-center token | 104/180 | 0.1339 | 0.7577 | better C2ST than target |
| PTB-XL source token | 109/180 | 0.0645 | 0.7464 | best C2ST control |

Decision:

```text
Only ningbo currently passes the strict Task-1-gated target-token promotion
gate. chapman/georgia/cpsc_2018 should not enter downstream online AT yet.

cpsc_2018 shows strong target style and gate-count gains for NORM/STTC, but the
K=500 seed42 ref selection has no MI references, so it is not a complete
NORM/MI/STTC synthetic pool.

For non-ningbo centers, the next center-token improvement should not be another
blind token_scale sweep. Use class-specific token scale or class-specific
checkpoint selection, and fix center/class reference coverage before downstream
AT.
```

## 2026-05-04 Large Ningbo Pool + Target-Real Stream Result

The next test scaled the only promoted setting, ningbo `token_scale=0.50`, to a
larger Task-1-gated pool and used it inside source-aware Latent-Hull online AT.

Generation/gate:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504

arm:
  target_token_s05

kept:
  757 / 1200
  MI=296, NORM=366, STTC=95

style probe:
  NORM mean P(ningbo)=0.4366
  MI   mean P(ningbo)=0.2774
  STTC mean P(ningbo)=0.3402

same-label no-leak C2ST:
  balanced accuracy = 0.8076
  AUROC = 0.8775
```

Downstream route:

```text
real target anchors:
  K=500 ningbo ECG

merged latent pool:
  real_anchor + prompt_token_s05_large

online AT:
  attack_mode = latent_hull
  hull_M = 10
  hull_lambda = 0.15
  adv_weight = 0.06
  source_weights = real_anchor=1.0,prompt_token_s05_large=0.4
  target_real_weight = 20

checkpoint:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v1/ningbo_real500_s05large_targetrealw20_srcw04_M10_lam015_adv006_softmix03_ep10
```

Formal eval, with the K=500 ningbo ref ids excluded:

| model | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|---:|
| Task-1 baseline | 0.9072 / 0.7744 | 0.7780 / 0.4831 | 0.8657 / 0.4842 |
| large token + real K500 AT | 0.9073 / 0.7747 | 0.7798 / 0.4885 | 0.8829 / 0.5049 |
| delta | +0.0002 / +0.0004 | +0.0018 / +0.0053 | +0.0172 / +0.0207 |

Decision:

```text
This is the strongest target-center downstream result so far and reaches the
+2pp held-out ningbo AUPRC target. It does not yet reach +2pp AUROC and does
not prove the center-token causal effect because the recipe also uses a
supervised real K=500 target stream.

Next center-token proof must be a matched control under this same recipe:
  real-only
  no-token synthetic
  target-token synthetic
  wrong-token synthetic
```

### Matched No-Token Control

The matched no-token pool used:

```text
same center = ningbo
same K=500 refs
same classes = NORM, MI, STTC
same n_per_class = 400
same sampling seed = 20260504
same ref_text_mode = class_fallback
same ref_class_policy = same_class
same per_ref_cap = 6
same Task-1 victim gate
only difference = --no_token
```

No-token generation/gate:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504

kept:
  694 / 1200
  MI=272, NORM=267, STTC=155
```

Downstream matched result:

| model | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|---:|
| real K500 + no-token synthetic | 0.9073 / 0.7747 | 0.7797 / 0.4883 | 0.8830 / 0.5050 |
| real K500 + target-token synthetic | 0.9073 / 0.7747 | 0.7798 / 0.4885 | 0.8829 / 0.5049 |
| target-token minus no-token | +0.0000 / -0.0000 | +0.0000 / +0.0002 | -0.0001 / -0.0001 |

Real-only reference:

```text
artifact:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_real_token_v1/ningbo_real500_only_targetrealw20_M10_lam015_adv006_softmix03_ep10

status:
  aborted after low ASR, but saved best ep2 checkpoint.

abort-best formal eval:
  PTB-XL      = 0.9077 / 0.7762
  PN2021 avg = 0.7787 / 0.4856
  ningbo     = 0.8767 / 0.4936
```

Updated decision:

```text
The large-pool v42 scale=0.50 center token is not downstream-positive against
matched no-token. ECGTwin synthetic latent candidates help relative to the weak
real-only abort-best, but the current learned token does not add utility.

Do not keep scaling this token pool. The next attempt must change the token
training/selection objective, especially pairwise target-token vs no-token
selection and bounded style rather than maximum target-center style score.
```

### Pairwise Style-Delta Selection Test

To test whether the problem was simply noisy sample selection, a paired selector
was added:

```text
script:
  scripts/ecgtwin_gen/select_paired_token_delta_pool.py

style CSV:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_pairwise_large_20260504/style_probe_full1000_gated/style_probe_samples_gated_samples_npz.csv

output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/paired_token_delta_v1_ningbo_20260504
```

Selection:

```text
same original generated sample index exists in target-token and no-token arms
target-token style score - no-token style score >= 0.05
target-token style score in [0.05, 0.95]
target-token p_target in [0.45, 0.995]
quota = NORM 80, MI 80, STTC 30
```

Downstream matched result:

| model | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | held-out ningbo AUROC/AUPRC |
|---|---:|---:|---:|
| paired target-token selected | 0.9067 / 0.7740 | 0.7794 / 0.4883 | 0.8834 / 0.5050 |
| paired no-token selected | 0.9071 / 0.7744 | 0.7802 / 0.4902 | 0.8837 / 0.5063 |

Decision:

```text
Pairwise style-delta selection is also negative. The no-token selected pool
still wins, so the current target-token samples are not useful simply because
they are more center-style-like.

Next token training must optimize for downstream-compatible realism:
  target-token should reduce same-label real-vs-synth C2ST or feature distance,
  not just increase P(target center).
```

### 2026-05-04 Contrastive Token-Training Patch

Post-hoc fixes were exhausted:

```text
v42 scale=0.50 large pool        -> no-token matched control ties/wins
pairwise positive style-delta    -> no-token selected control wins
v44 bounded-style/composed token -> no-token actual-report control wins
PTB-XL E5 top4000 token pool     -> beats naked no-token top2000, but loses to
                                   scaled no-token top3970
HYP-token class-wise hybrid      -> loses to no-token top3970
```

The next token-training route is now implemented in code as paired
token-vs-no-token contrastive supervision:

```text
script:
  scripts/ecgtwin_gen/train_center_prompt_tokens.py

trainer:
  methods/ecgtwin_gen/prompt_token/trainer.py

new losses:
  contrast_recon:
    same noisy latent, same ref ECG/text, same class prompt.
    token denoise MSE must be no worse than no-token denoise MSE.

  contrast_style_delta:
    P(target center | token) - P(target center | no-token) >= margin.

  contrast_semantic_delta:
    P(primary super5 class | token) - P(primary super5 class | no-token) >= margin.
```

Recommended first v45 PN2021 command:

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo \
  --cache_root /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1 \
  --prompt_bank /root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt \
  --save_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v45_ningbo_contrast_recon_style_sem_20260504 \
  --K 500 \
  --seed 42 \
  --total_steps 3000 \
  --batch_size 16 \
  --num_workers 4 \
  --token_mode direct \
  --n_token_vectors 4 \
  --sample_strategy class_balanced \
  --ref_text_mode actual_report \
  --style_ckpt /root/autodl-tmp/center_style_classifier_noleak/full1000/best_model.pt \
  --style_loss_weight 0.03 \
  --style_loss_mode target_prob \
  --style_target_prob 0.65 \
  --semantic_ckpt /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt \
  --semantic_loss_weight 0.03 \
  --contrast_recon_weight 0.20 \
  --contrast_recon_margin 0.0 \
  --contrast_style_delta_weight 0.05 \
  --contrast_style_delta_margin 0.05 \
  --contrast_semantic_delta_weight 0.05 \
  --contrast_semantic_delta_margin 0.02 \
  --aux_timestep_max 50 \
  --amp_dtype bf16
```

Promotion gate:

```text
Do not promote v45 unless paired target-token generation beats no-token on:
  1. semantic target-class pass rate,
  2. no-leak style/C2ST,
  3. downstream matched no-token AUROC/AUPRC.
```

Executed v45 generation-side result:

```text
token bank:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v45_ningbo_contrast_recon_style_sem_20260504/prompt_token_bank.pt

paired large generation:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v45_contrast_large_ningbo_20260504/

style probe:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v45_contrast_large_ningbo_20260504/style_probe_full1000/style_probe_summary_samples_npz.json
```

No-leak style probe, mean target-center probability and top1 rate:

| arm | class | mean P(ningbo) | top1 ningbo rate |
|---|---|---:|---:|
| no-token | MI | 0.0802 | 0.0800 |
| no-token | NORM | 0.1184 | 0.0725 |
| no-token | STTC | 0.0841 | 0.0875 |
| target-token | MI | 0.4643 | 0.5175 |
| target-token | NORM | 0.6247 | 0.7725 |
| target-token | STTC | 0.5190 | 0.6075 |

Quality gate counts:

| arm | total pass | MI | NORM | STTC |
|---|---:|---:|---:|---:|
| target-token, no top1 gate | 705 / 1200 | 304 | 390 | 11 |
| no-token, no top1 gate | 715 / 1200 | 283 | 271 | 161 |

Decision:

```text
v45 passes the generation-side center-style test. It does not yet pass the
downstream utility gate because matched v45 online-AT token-vs-no-token
AUROC/AUPRC has not produced a token win.

For downstream use, avoid relying on v45 STTC synthetic from the strict digital
gate because only 11/400 STTC samples passed; use v45 mainly for NORM/MI unless
a separate STTC token/gate strategy is introduced.
```

Matched downstream online-AT follow-up:

```text
paired selector:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/paired_token_delta_v45_norm_mi_ningbo_20260504

paired selected counts:
  NORM=223
  MI=123

merged target-token pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/merged_v45_norm_mi_real500_ningbo_20260504/target_token/merged.latent.npz

merged no-token pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/merged_v45_norm_mi_real500_ningbo_20260504/no_token/merged.latent.npz
```

Formal ref-excluded results:

| run | source weight | PN2021 avg AUROC | PN2021 avg AUPRC | held-out ningbo AUROC | held-out ningbo AUPRC |
|---|---:|---:|---:|---:|---:|
| Task-1 baseline | n/a | 0.7780 | 0.4831 | 0.8657 | 0.4842 |
| v45 target-token | 0.4 | 0.7813 | 0.4893 | 0.8882 | 0.5159 |
| v45 no-token | 0.4 | 0.7806 | 0.4890 | 0.8884 | 0.5165 |
| v45 target-token | 1.0 | 0.7810 | 0.4896 | 0.8882 | 0.5168 |
| v45 no-token | 1.0 | 0.7808 | 0.4887 | 0.8882 | 0.5158 |

Decision:

```text
v45 target-token reaches the target-center +2pp criterion versus the PTB-XL-only
Task-1 baseline on held-out ningbo. However, no-token remains essentially tied,
so this is not yet a center-token causal win.
```

### 2026-05-04 Multi-Center Target-Real v2 Check

The same target-center adaptation idea was extended with v42 target-token pools
for chapman_shaoxing, cpsc_2018, and georgia.

Shared recipe:

```text
K target real = 500
synthetic = v42 target-token gated latents
merge = real_anchor + prompt_token
target_real_weight = 40
source_weights = real_anchor=1.0,prompt_token=0.4
attack_mode = latent_hull
hull_M = 10
hull_lambda = 0.15
adv_weight = 0.06
adv_label_mode = mixed_soft
boundary probability window = [0.45, 0.70]
eval = full PN2021 v3 with target-center K=500 ref ids excluded
```

Formal results:

| center | baseline target AUROC/AUPRC | token+real AT target AUROC/AUPRC | delta |
|---|---:|---:|---:|
| chapman_shaoxing | 0.8763 / 0.4251 | 0.8972 / 0.4647 | +2.09pp / +3.96pp |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8543 / 0.5959 | +4.28pp / +3.73pp |
| georgia | 0.8157 / 0.5916 | 0.8262 / 0.6029 | +1.05pp / +1.13pp |

No-token controls:

| center | target-token AUROC/AUPRC | no-token AUROC/AUPRC | token - no-token |
|---|---:|---:|---:|
| chapman_shaoxing | 0.8972 / 0.4647 | 0.8972 / 0.4653 | +0.00pp / -0.06pp |
| cpsc_2018 | 0.8543 / 0.5959 | 0.8541 / 0.5958 | +0.02pp / +0.01pp |

Decision:

```text
The target-center adaptation route is now useful for at least two large
centers versus the bare PTB-XL baseline. The result should be presented as a
practical adaptation win, not as center-token causality. No-token controls are
effectively tied with target-token on the successful chapman/cpsc centers.
```

### 2026-05-04 v46 PTB-XL Contrastive Token Check

v46 applied the paired contrastive token-vs-no-token idea to the PTB-XL source
center, because the graduate-project self-distillation line still needed a
strict center-token causal test.

Training:

```text
save_dir:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v46_contrast_recon_sem_steps4000_20260504

center:
  ptbxl_source

token schema:
  direct MV4, 4 learned vectors/class, 5 classes

loss:
  reconstruction + semantic + paired contrastive reconstruction/semantic delta
```

Matched generation and v2 filtering:

| arm | candidates | kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v46 center-token | 8000 | 4000 | 800 | 800 | 800 | 800 | 800 |
| v46 no-token | 8000 | 3734 | 800 | 534 | 800 | 800 | 800 |
| v46 token count-matched | subset | 3734 | 800 | 534 | 800 | 800 | 800 |

Downstream self-distillation:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| real2000 baseline | 0.8433 | 0.6234 | 0.8297 | 0.6190 | 0.7085 | 0.3912 |
| v46 center-token top4000 | 0.8508 | 0.6437 | 0.8430 | 0.6419 | 0.7187 | 0.4073 |
| v46 no-token 3734 | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| v46 token count-matched 3734 | 0.8537 | 0.6466 | 0.8426 | 0.6383 | 0.7078 | 0.4038 |

Decision:

```text
v46 center token improves synthetic acceptance/filter count, but it still does
not pass the downstream causal gate.

The best v46 PTB-XL AUPRC is the token count-matched arm, but AUROC does not
improve and PN2021 external performance is worse than no-token. This means the
current prompt-token pathway is useful as a generation-control mechanism, not
yet as a proven EfficientNetV2 performance enhancer.
```

### 2026-05-04 v47 Feature-Contrast Token And Boundary Selection

v47 added a frozen EfficientNet1DV2 penultimate-feature objective to token
training. The intent was to make the token alter ECGTwin output in a classifier
feature space, not only in latent reconstruction and semantic probability.

Training:

```text
save_dir:
  /root/autodl-tmp/ecgtwin_ptbxl_prompt_token_boundary_at_v1/prompt_token_runs/ptbxl_source_v47_feature_contrast_steps4000_20260504

feature_loss_weight = 0.05
contrast_feature_weight = 0.10
feature_crop_len = 1000
semantic_loss_weight = 0.02
contrast_recon_weight = 0.20
contrast_semantic_delta_weight = 0.03
```

Generation sanity, 60 samples/class:

| arm | CD top1 | HYP top1 | MI top1 | NORM top1 | STTC top1 |
|---|---:|---:|---:|---:|---:|
| v46 token | 0.967 | 0.750 | 0.883 | 1.000 | 0.933 |
| v46 no-token | 0.833 | 0.583 | 0.300 | 0.900 | 0.650 |
| v47 feature-token | 0.800 | 0.750 | 0.550 | 0.917 | 0.567 |

Decision:

```text
Do not promote v47 as the next large-pool token. It degraded MI and STTC
generation sanity compared with v46.
```

Boundary-confidence v2 filtering was then tried on the existing v46 large
token/no-token pools:

```text
teacher ensemble = real2000 seed42/43/44
target confidence window = [0.35, 0.75]
rank = closest to 0.55
goal = select decision-boundary-like synthetic samples instead of high-confidence
       easy samples
```

| arm | kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|
| v46 token boundary | 1580 | 348 | 400 | 400 | 36 | 396 |
| v46 no-token boundary | 1505 | 400 | 400 | 400 | 95 | 210 |

Downstream:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 token boundary | 0.8099 | 0.5484 | 0.7996 | 0.5521 | 0.6752 | 0.3525 |
| v46 no-token boundary | 0.8403 | 0.6266 | 0.8292 | 0.6182 | 0.7093 | 0.4076 |

Decision:

```text
Boundary-confidence selection is negative for the center-token causal story.
The no-token arm remains stronger even when both arms are forced into the same
teacher-confidence band. Future token promotion should require paired
no-token/wrong-token controls and downstream wins before large-pool training.
```

### 2026-05-04 v48 MMD Token And v46 Paired-Delta Result

v48 added a frozen EfficientNet feature-distribution MMD loss to the PTB-XL
source center token:

```text
feature_mmd_weight = 0.10
contrast_feature_mmd_weight = 0.20
feature_mmd_sigma = 1.0
semantic_loss_weight = 0.01
4000 steps, direct MV4 token
```

Sanity generation did not beat v46:

| arm | CD top1 | HYP top1 | MI top1 | NORM top1 | STTC top1 | overall top1 |
|---|---:|---:|---:|---:|---:|---:|
| v46 token | 0.967 | 0.750 | 0.883 | 1.000 | 0.933 | 0.907 |
| v48 MMD-token | 0.850 | 0.683 | 0.667 | 0.917 | 0.550 | 0.733 |

Decision:

```text
Do not promote v48. The MMD objective is implemented and runs, but this
configuration damages MI/STTC generation sanity relative to v46.
```

The strongest v46 token was then tested with a paired-delta downstream
selection:

```text
same ref ECG, same seed, same target class, token vs no-token
select top 400/class where token raises target probability most
filter both arms with the same 3-teacher v2 self-distillation gate
```

Filter result:

| arm | raw pairs | filtered kept | CD | HYP | MI | NORM | STTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v46 paired-delta token | 2000 | 1759 | 392 | 264 | 334 | 392 | 377 |
| v46 paired-delta no-token | 2000 | 1069 | 375 | 74 | 219 | 293 | 108 |

Downstream:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 paired-delta token | 0.8472 | 0.6222 | 0.8241 | 0.5936 | 0.6886 | 0.3677 |
| v46 paired-delta no-token | 0.8597 | 0.6612 | 0.8260 | 0.6030 | 0.7092 | 0.4063 |

Interpretation:

```text
The current center token is effective at generation control: it increases
target-class confidence and v2 filter acceptance on the exact same pairs.
It is not yet an effective downstream augmentation token. Even a token-favored
paired-delta selection loses to no-token after EfficientNet self-distillation.
Future token work needs a downstream-aware objective, not only stronger
semantic/style confidence.
```

Follow-up combo test:

```text
v46 no-token filtered pool + v46 token filtered pool
token synthetic sample_weights *= 0.35
same self-distill v2 student recipe
```

| arm | custom AUROC | custom AUPRC |
|---|---:|---:|
| v46 no-token 3734 | 0.8530 | 0.6404 |
| v46 token top4000 | 0.8508 | 0.6437 |
| combo no-token + token w0.35 | 0.8497 | 0.6388 |

Decision:

```text
Token samples do not help as a simple low-weight add-on to the no-token pool.
The next token revision needs a new objective or selection policy; simple
pool concatenation and token sample-weight sweeps are closed.
```

Class-oracle hybrid follow-up:

```text
CD/HYP/MI from v46 token count-matched pool
NORM/STTC from v46 no-token matched pool
```

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 no-token 3734 | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| class-oracle hybrid | 0.8561 | 0.6511 | 0.8286 | 0.6045 | 0.7220 | 0.4043 |
| class-oracle hybrid, AUPRC checkpoint | 0.8517 | 0.6416 | not run | not run | not run | not run |

Interpretation:

```text
Token utility is class-specific. The hybrid beats no-token on the custom test by
+0.31pp AUROC and +1.07pp AUPRC, but it still misses the +2pp/+2pp target and
does not generalize to fold10 or PN2021. The next token objective should learn
per-class/domain reliability directly instead of using one shared token pool.

The AUPRC-checkpoint rerun did not rescue the hybrid: custom AUPRC was only
0.6416, so it was not promoted to fold10/PN2021 evaluation.
```

Hard-label trust follow-up:

```text
Hypothesis:
  center-token samples may be semantically clean enough to use synthetic hard
  labels directly, while no-token samples should remain weaker.

Recipe:
  use_synth_hard_labels = true
  synth_distill_weight = 0
  real_distill_alpha = 0
  synth_ratio = 1.0, plus 0.75 and 1.25 token-only probes
```

| arm | synth ratio | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-token hard-label, same seed | 1.00 | 0.8574 | 0.6633 | 0.8385 | 0.6394 | 0.7329 | 0.4345 |
| class-oracle token-hard | 1.00 | 0.8693 | 0.6955 | 0.8308 | 0.6320 | 0.7318 | 0.4349 |
| class-oracle token-hard | 0.75 | 0.8682 | 0.6961 | 0.8306 | 0.6443 | 0.7313 | 0.4302 |
| class-oracle token-hard | 1.25 | 0.8583 | 0.6676 | not run | not run | not run | not run |

Conclusion:

```text
This is custom-positive but external-negative. Token-hard beats no-token-hard
on custom AUPRC by +3.22pp, but custom AUROC improves only +1.20pp and fold10 /
PN2021 do not improve. This supports class-specific token signal, not a robust
center-token downstream claim.
```

Real-only fine-tune repair:

| arm | custom AUROC | custom AUPRC | fold10 AUROC | fold10 AUPRC | PN2021 AUROC | PN2021 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| v46 no-token soft self-distill | 0.8530 | 0.6404 | 0.8433 | 0.6333 | 0.7250 | 0.4166 |
| token-hard -> real2000 FT | 0.8735 | 0.7013 | 0.8452 | 0.6455 | 0.7346 | 0.4281 |
| no-token-hard -> real2000 FT | 0.8669 | 0.6828 | 0.8479 | 0.6506 | 0.7370 | 0.4450 |

Conclusion:

```text
The improved center-token route meets +2pp AUROC / +2pp AUPRC only versus the
naked v46 no-token soft self-distillation baseline. It does not beat the
same-recipe no-token-hard -> real2000 fine-tune control by +2pp, and external
fold10/PN2021 favor the no-token fine-tuned control.
```
