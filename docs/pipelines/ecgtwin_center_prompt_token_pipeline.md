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

## 与现有 CenterToken 的区别

现有实现：

```text
methods/ecgtwin_gen/center_token/model.py
methods/ecgtwin_gen/center_token/trainer_v2.py
scripts/ecgtwin_gen/train_center_token_v2.py
scripts/ecgtwin_gen/generate_center_synth.py
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
