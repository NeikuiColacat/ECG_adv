# Latent-Hull TA-OMAT Online Adversarial Training Pipeline

本文档定义参考 SA-AET 思想的新在线对抗训练方案。核心是把旧 TA-OMAT 的自由 latent PGD `z0 + delta` 改成 same-label latent convex hull。

## 动机

旧路线：

```text
z_adv = z0 + delta
```

自由 `delta` 攻击强，但医学语义容易漂移。新路线：

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

其中：

- `z0` 是真实目标中心 anchor latent。
- `z_i` 来自同一 super5 类或同 exact-label pattern；第一版不做跨类别 mixing。
- `softmax(a)` 保证 convex combination。
- `lambda` 限制在 `[0, lambda_max]`，第一版建议 `0.25` 或 `<= 0.4`。
- 第一轮候选池规模数组固定为 `M=[5,10,20]`；默认主线用 `M=10`。
- `M=3` 只作为 very-small sanity ablation。它更接近少数近邻插值，
  多样性不足，优先级低于 `M=5/10/20`。
- `M=100` 暂不进入第一轮，因为大候选池容易均值化 latent、降低医学证据强度，并显著增加 decode/victim backward 成本。

推荐名称：

```text
Latent-Hull TA-OMAT
```

## 2026-05-03 Mainline Update

本项目后续在线对抗训练主线改为 Latent-Hull online AT，而不是自由 latent PGD。
自由 `z0 + delta` 仍保留为历史对照，但不作为毕业设计主方法。

新的主线输入模型优先使用：

```text
/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503
```

也就是新训练的：

```text
PTB-XL super5
minimal_resample
per_sample_global
100Hz / 1000 samples / full 10s
```

如果新 baseline 尚未完成，则只允许用旧 baseline 做 smoke，不写最终结论。

### Main Algorithm

对每个 anchor latent `z0`，从 same-label candidate pool 中取 `M` 个候选：

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

其中 `a_i` 是在线优化变量，不是固定手工权重。

推荐默认：

```text
hull_weight_mode = optimized
M = 10
lambda = 0.25
hull_steps = 5
hull_lr = 0.3
pgd_eps = 2.0
same-label mode = primary first, exact multi-hot as strict ablation
```

### Candidate Pool Priority

候选池分三层：

```text
P0 real-anchor:
  目标中心真实 ECG 的 VAE latent。
  当前最稳，作为主方法基础。

P1 target-token synthetic:
  通过 no-leak style validation 和 digital/semantic gate 的 ECGTwin center-token synthetic latent。
  只有通过 center-style pipeline acceptance 后进入主实验。

P2 controls:
  vanilla synthetic, wrong-center token synthetic, PTB-XL-source token synthetic。
  只作为证明 target-token 是否有用的对照。
```

### M And Lambda Search

已有结果支持 `M=10, lambda=0.25` 是当前最稳 AUPRC 主设置。下一轮不做无目的大网格，
只做解释性 ablation：

| parameter | main | ablation | note |
|---|---:|---|---|
| M | 10 | 5, 20 | 已有结果覆盖；M=10 当前 AUPRC 最好 |
| M very small | - | 3 | sanity check only；更像少数近邻插值，可能多样性不足 |
| M large | - | 50 | 只在 cluster-balanced candidate pool 上跑 |
| M very large | - | 100 | 默认不跑，容易 latent 均值化并增加 decode/backward 成本 |
| lambda | 0.25 | 0.10, 0.40 | 已有结果显示 0.25 最稳 |
| hull_steps | 5 | 10 | 只在 best pool 上做 |
| weight mode | optimized | one_hot, uniform, Dirichlet | 已有 C3 optimized AUPRC 最好 |

## 2026-05-23 v5 Label Refinement: Real-Anchor All-Class Trust

当前 v5 PN2021 标签映射变得更严格：

```text
pacing/device rhythm 不再算 CD
voltage-only 不再算 HYP
all-zero 保留为主评估负样本
```

在这个口径下，VAE-only real-anchor 方案需要重新审视旧的 class trust。历史主线沿用了
ECGTwin synthetic 质量判断：

```text
classes_in_scope = NORM, MI, STTC
CD/HYP class_trust = 0
```

这对 ECGTwin 直接合成样本是合理的，因为 HYP/CD 合成样本的数字 ECG 质量曾经较弱；但对
`real-anchor` VAE-only 方法不一定合理。这里的 anchor latent 来自目标中心真实 ECG，
CD/HYP 标签也来自真实中心标签映射，不是 DiT 合成标签。因此新增优先 refinement：

```text
trust_policy = real_all_present
classes_in_scope = CD, HYP, MI, NORM, STTC
```

实现细节：

1. `run_vae_only_latenthull_sweep_20260516.py` 新增 `--trust_policy real_all_present`。
2. 对每个 K=500 真实目标中心子集，凡是该类真实阳性数 > 0，就设 `class_trust[class] = 1.0`。
3. 下游调用 `synth_online_at_super5.py` 时传入 `--allow_hyp_cd_trust`，避免训练入口再次把 CD/HYP 硬置 0。
4. 仍然禁用 hard quality gate；gate 指标只作为监控。
5. 第一轮只跑 `lambda015, M=20, epochs=30`，与当前主线最接近，避免同时改变太多变量。

优先对比表：

| arm | classes | trust | purpose |
|---|---|---|---|
| legacy VAE-only | NORM/MI/STTC | synthetic-style trust | 当前主线 |
| all-class VAE-only | CD/HYP/MI/NORM/STTC | real_all_present | 检验 v5 下是否需要把真实 CD/HYP anchor 纳入 AT |

判断标准：

```text
target-center AUROC/AUPRC
4-center average AUROC/AUPRC
PTB-XL fold10 是否明显下降
CD/HYP per-class 是否改善
```

第一轮完成结果：

| center | legacy VAE-only | all-class VAE-only | all-class vs legacy |
|---|---:|---:|---:|
| ningbo | 0.8880 / 0.4393 | 0.8894 / 0.4419 | +0.15pp / +0.26pp |
| chapman_shaoxing | 0.8949 / 0.3720 | 0.8964 / 0.3750 | +0.15pp / +0.30pp |
| cpsc_2018 | 0.8545 / 0.5962 | 0.8684 / 0.6151 | +1.40pp / +1.89pp |
| georgia | 0.8266 / 0.6012 | 0.8280 / 0.6033 | +0.14pp / +0.21pp |

结论：`real_all_present` 应作为 v5 mapping 下的新主线候选。它证明历史 CD/HYP gate 对真实 anchor 过保守，尤其在 `cpsc_2018` 这种 CD 占比高的中心。下一轮优先做 per-center/per-class 权重，而不是单纯增加 epoch。

### Boundary Confidence Constraint

用户希望对抗样本不要过度扰动导致 GT 标签漂移。因此新增 acceptance gate：

```text
preferred target probability:
  p_target in [0.50, 0.60]

acceptable boundary window:
  p_target in [0.45, 0.65]
```

实现方式：

```text
inner objective still maximizes victim loss;
after decode, samples outside the confidence window are down-ranked or rejected;
QualityAwareBuffer score = 1 - 2 * abs(p_target - 0.5).
```

如果内层攻击太强：

```text
reduce lambda to 0.10
reduce hull_steps to 3
increase teacher consistency weight
reject samples with semantic drift
```

### Label Policy

第一版只允许 same-label mixing：

```text
same primary super5 class -> keep original GT label
exact same multi-hot -> keep original multi-hot label
NORM -> only pure NORM neighbors
```

不允许跨类 latent mixing 后继续使用原标签。若未来做跨类：

```text
y_adv = convex/union soft label
non-target classes with uncertainty use -1 masked sentinel
```

这不是当前主线。

### Source-Aware Hybrid Rule

如果把 real-anchor 和 target-token synthetic 放在同一个 candidate pool，必须记录 source：

```text
source_id = real_anchor | target_token | vanilla | wrong_center | ptbxl_source
```

并使用 source-aware sampling/weights：

```text
real_anchor weight = 1.0
target_token weight = 0.35 to 0.70 initial sweep
wrong/vanilla controls = matched ratio only, not mixed into main model
```

Acceptance：

```text
target-token + real-anchor source-aware
  must beat real-anchor only or at least improve a target-center metric
  without lowering PN2021 macro AUPRC.
```

### Target-Real Supervised Stream Update

2026-05-04 后，`synth_online_at_super5.py` 支持把目标中心 K 条真实 ECG 作为一个
额外监督训练流加入外层训练：

```text
--target_real_npz
--target_real_weight
```

该训练流不同于 Latent-Hull 内层攻击：

```text
Latent-Hull candidate pool:
  real_anchor latent + target-token synthetic latent
  used to create online adversarial/boundary samples

target_real stream:
  K=500 target-center real ECG signals and labels
  used as supervised mini-batch stream during outer training
```

这样做的动机：

```text
1. 单独 synthetic target-token 下游 self-distillation 已出现负结果；
2. 目标中心真实 K=500 ECG 是最可信的 center-style anchor；
3. target-token latent 只作为 bounded style/boundary candidate，不直接主导训练；
4. 外层训练用 target_real_weight 控制目标中心监督信号强度。
```

已完成 ningbo sweep：

| run | target_real_weight | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | ningbo AUROC/AUPRC, K=500 refs excluded |
|---|---:|---:|---:|---:|
| Task-1 baseline | 0 | 0.9072 / 0.7744 | 0.7780 / 0.4831 | 0.8657 / 0.4842 |
| real+token AT w4 | 4 | 0.9089 / 0.7780 | 0.7780 / 0.4825 | 0.8749 / 0.4903 |
| real+token AT w12 | 12 | 0.9079 / 0.7761 | 0.7796 / 0.4876 | 0.8811 / 0.5019 |
| real+token AT w20 | 20 | 0.9073 / 0.7747 | 0.7798 / 0.4885 | 0.8829 / 0.5049 |
| real+no-token AT w20 | 20 | 0.9073 / 0.7747 | 0.7797 / 0.4883 | 0.8830 / 0.5050 |
| real-only w20 abort-best | 20 | 0.9077 / 0.7762 | 0.7787 / 0.4856 | 0.8767 / 0.4936 |

Current interpretation:

```text
w20 gives the best held-out ningbo AUPRC gain:
  +1.72pp AUROC
  +2.07pp AUPRC

This is close to the target-center +2pp AUROC/AUPRC acceptance target, but not
fully achieved because AUROC is still short. It is also not yet a center-token
causal proof until matched real-only, no-token, and wrong-token controls are
run under the same target_real_weight.

2026-05-04 matched no-token update:
  no-token w20 matches/slightly exceeds target-token w20 on held-out ningbo
  (0.8830 / 0.5050 vs 0.8829 / 0.5049). Thus the current benefit comes from
  adding ECGTwin synthetic latent candidates plus K=500 target-real adaptation,
  not from the learned center token. Real-only abort-best is weaker, so a
  synthetic pool is useful; it is just not yet center-token-specific.
```

Next required controls:

```text
R0: target_real stream only, no synthetic latent pool
R1: target_real + real_anchor latent pool only
R2: target_real + no-token synthetic latent pool
R3: target_real + target-token synthetic latent pool
R4: target_real + wrong-center token latent pool

All controls must use:
  same K=500 ref ids
  same ref-id exclusion in eval
  same target_real_weight, preferably 20 first
  same source weights and hull parameters
```

Current control status:

```text
R1 real-only was attempted and produced an abort-best checkpoint, but ASR stayed
below 0.3 for three epochs because the K=500 real pool has too few MI anchors.
It is useful as a weak reference, not a completed AT run.

R2 no-token and R3 target-token are complete and are effectively tied.
R4 wrong-token remains pending and should be run only after a new token/selection
strategy has a plausible chance to beat R2.
```

Pairwise style-delta Latent-Hull control:

| run | synthetic selection | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | ningbo AUROC/AUPRC |
|---|---|---:|---:|---:|
| target-token selected | token style - no-token style >= 0.05 | 0.9067 / 0.7740 | 0.7794 / 0.4883 | 0.8834 / 0.5050 |
| no-token selected | same pair ids | 0.9071 / 0.7744 | 0.7802 / 0.4902 | 0.8837 / 0.5063 |

Interpretation:

```text
Latent-Hull online AT is sensitive to candidate quality, but target-center
style-score gains are not the right quality signal by themselves. For future
center-token AT, the token-selected pool must improve real-vs-synth realism
metrics or C2ST against the matched no-token pool before spending GPU on full
online AT.
```

### Outputs For New Main Runs

每个 online AT run 必须保存：

```text
run_config.json
training_log.json
train_result.json
eval_result_v3_super5_normsuppress.json
pn2021_per_center_delta.csv
hull_stats.jsonl
buffer_quality_summary.json
accepted_adv_samples.npz or indices-only manifest
```

PN2021-C 只在 clean PN2021 AUPRC 不低于 baseline 或目标中心明显提升时再跑。

## 可复用代码

```text
adversarial/pgd_advdiff.py
scripts/pgd_cross_center/synth_online_at_super5.py
scripts/crosscenter_tierM/online_adv_train_tierM.py
docs/module_ablation_1_real_vs_synth.md
model/SA-AET/SA_AET.py
```

旧代码已有：

- current victim online PGD。
- ASR gate 和 semantic gate。
- QualityAwareBuffer。
- PTB-XL real / roundtrip / adv buffer 混合训练。
- masked BCE 和 EWA anchor regularizer。

## 医学语义规则

不能把任意 latent 线性组合说成 label-preserving。只在以下约束下复用 GT：

1. 同 super5 类或同 exact multi-hot label pattern。
2. NORM 最严格，只允许 NORM 邻居，且 abnormal probability gate 必须通过。
3. MI/STTC 可做同类 convex hull，但只能表述为“约束插值下的 label-preserving adversarial candidate”。
4. CD/HYP 只有在 ECGTwin 官方 prompt smoke generation、数字 ECG gate 和 teacher/classifier gate
   都通过时才进入下游增强；否则只报告 token/生成验证。
5. 不允许跨病理混合后继续使用原单标签。若未来允许跨类混合，GT 必须变为 union/soft label，并另设实验版本。

## 内层攻击目标

第一版：

```text
maximize_a BCEWithLogits(f_theta(decode(z_adv(a))), y)
```

约束：

```text
w = softmax(a)
z_mix = sum_i w_i * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
||z_adv - z0||_2 <= eps
```

完整版本可加入：

```text
L_inner =
  BCE(f_theta(x_adv), y)
  - beta_dist * ||z_adv - z0||_2^2
  - beta_proto * d_mahalanobis(z_adv, class_proto_y)
  - beta_teacher * BCE(frozen_teacher(x_adv), y)
  + beta_entropy * H(w)
```

注意：内层是最大化分类器损失；正则项符号要按实现方向检查。

## 系数确定方式

第一版不是手工固定所有系数，而是把系数作为内层攻击变量：

```text
w_i = softmax(a_i)
z_mix = sum_i w_i * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

这样保证：

```text
w_i >= 0
sum_i w_i = 1
z_mix stays in same-label convex hull
```

初始化：

```text
a starts one-hot-like:
  nearest same-label candidate gets the largest logit
  other candidates start low but nonzero after softmax
```

优化：

```text
inner objective = maximize victim BCE/margin loss
optimizer = Adam or projected gradient on a
steps = 5 in A1, 10 in A2-light
lambda = 0.25 fixed in A1
```

对照消融：

```text
C0 one-hot nearest neighbor, no optimization
C1 uniform weights, no optimization
C2 Dirichlet(alpha=1) random weights, no optimization
C3 optimized softmax weights from one-hot init  # main setting
```

`lambda` 消融：

```text
lambda in [0.1, 0.25, 0.4]
```

医学语义解释：

- 同标签 convex hull + 小 `lambda` 是 label-preserving 的必要条件之一，不是充分条件。
- 最终是否接收样本由 digital ECG gate、teacher/classifier gate 和 ref-id exclusion 决定。

## 外层训练目标

```text
min_theta
  BCE_masked(f_theta(x_real), y_real)
  + adv_weight * BCE_masked(f_theta(x_adv), y_adv)
  + anchor_lambda * EWA_regularizer
```

`y_adv` 第一版沿用旧方案：

```text
target class = known 0/1
non-target classes = -1 sentinel masked
```

如果 anchor 是可信 multi-hot，则保留 multi-hot；如果来自 single-target synth pool，则使用 target-only sentinel 更稳。

## Gates

进入 adv buffer 前必须通过：

- ASR gate。
- Einthoven residual p95 gate。
- HR delta gate。
- QRS ratio gate。
- amplitude / flatline / NaN / Inf gate。
- frozen teacher consistency gate。
- NORM abnormal suppression gate。
- K=500 ref ids 排除，避免同中心 ref pool 泄漏到 fine-tune validation/eval。

Buffer 评分优先 boundary-informative：

```text
score = 1 - 2 * abs(prob_target - 0.5)
```

## 第一版实现步骤

1. DONE: 新增 `LatentHullPGDGenerator`，接口对齐 `PGDAdvDiffGenerator.attack_from_latent`。
2. 输入 `.latent.npz`：

```text
latents: (N, 4, 128)
labels:  (N, 5)
record_ids optional
center_name
```

3. DONE: 为每个 anchor 找同类 top-M candidate latent，M in `[5,10,20]`；候选池来自 K=500 ref pool
   以外的训练/生成样本，评测 split 必须排除这 K=500 ref ids。
4. DONE: 内层优化 `a` 5 步，先固定 `lambda=0.25`。
5. DONE: decode 后转 `(B,12,1000)` PTB-XL lead order。
6. DONE: 复用旧 gates。
7. DONE: push 到 QualityAwareBuffer。
8. DONE: 外层训练沿用 `synth_online_at_super5.py` 的 stream mixing。

当前源码：

```text
adversarial/latent_hull_pgd.py
scripts/pgd_cross_center/synth_online_at_super5.py
```

CLI：

```text
--attack_mode latent_hull
--hull_M 5|10|20
--hull_lambda 0.25
--hull_steps 5
--hull_lr 0.3
--hull_label_mode primary|exact
```

实现细节：

- `SameLabelLatentIndex` 先按 `primary` 或 `exact multi-hot` 建池。
- 每个 anchor 选 same-label nearest top-M candidate，默认排除自身；不足 M 时重复最近邻补齐。
- `LatentHullPGDGenerator` 优化 `a`，`w=softmax(a)`，并用 `--pgd_eps` 做最终 L2 cap。
- L2 cap 是沿 `z0 -> z_adv` 方向缩回，因此仍在同标签凸包线段内。
- 日志记录 `delta_mean/max`、`hull_weight_entropy_mean`、`hull_weight_top1_mean`。

已完成 smoke：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -m adversarial.latent_hull_pgd \
  --smoke --M 5 --hull_lambda 0.25 --hull_steps 1 \
  --device cuda:0 --ckpt /root/autodl-tmp/triple_labels/super5/best_model.pt \
  --num_classes 5 --epsilon 2.0
```

在线训练 smoke：

```text
/root/autodl-tmp/latent_hull_smoke/extra_M5_steps1_cap/
```

结果：

```text
K_anchor=6, M=5, lambda=0.25, hull_steps=1
delta_mean=1.9772, delta_max=2.0000
hull_weight_entropy_mean=0.0164, top1_mean=0.9981
ASR=1.0, Einthoven p95=0.1512, buffer_size=6
```

`cpsc_2018_extra` real-anchor M-grid pilot 已完成：

```text
input pool:
  /root/autodl-tmp/real_anchored_super5/synth_latents/extra_real_k200.latent.npz
  class counts: MI=90, NORM=7, STTC=103, CD/HYP=0
quick eval:
  4 centers x 300 records, cpsc_2018_extra ref ids excluded
training:
  n_epochs=20, eval_every=2, patience=10, K_anchor=300, pgd_batch=16
  lambda=0.25, hull_steps=5, pgd_eps=2.0
```

Results:

| run | best quick AUROC | epochs | PTB-XL AUROC/AUPRC | PN2021 avg AUROC/AUPRC | delta vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | - | - | 0.9064 / 0.7754 | 0.8344 / 0.5526 | - |
| M=5 | 0.8608 | 16 | 0.9067 / 0.7766 | 0.8346 / 0.5557 | +0.0002 / +0.0031 |
| M=10 | 0.8609 | 16 | 0.9067 / 0.7765 | 0.8345 / 0.5553 | +0.0002 / +0.0027 |
| M=20 | 0.8604 | 20 | 0.9066 / 0.7767 | 0.8349 / 0.5550 | +0.0005 / +0.0024 |

First-epoch hull stats:

| M | delta mean/max | entropy mean | top1 weight mean | ASR | Einthoven p95 |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.9744 / 2.0000 | 0.0875 | 0.9860 | 0.3909 | 0.1418 |
| 10 | 1.9736 / 2.0000 | 0.1902 | 0.9697 | 0.4010 | 0.1410 |
| 20 | 1.9718 / 2.0000 | 0.3792 | 0.9398 | 0.4010 | 0.1394 |

Interpretation:

- All M values are stable and pass medical gates.
- M=5 gives the best AUPRC lift in this pilot; M=20 gives the best AUROC lift.
- Because `extra_real_k200` has only 7 NORM anchors, M=10/20 repeat NORM nearest candidates;
  the next comparison should include `ningbo` and `georgia` before choosing the thesis default.

Three-center real-anchor M-grid completed:

```text
summary: /root/autodl-tmp/latent_hull_super5_pilot/summary_mgrid_v1.json
baseline PN2021 v3: AUROC=0.8344, AUPRC=0.5526
settings: lambda=0.25, hull_steps=5, hull_lr=0.3, pgd_eps=2.0,
          K_anchor=300, pgd_batch=16, n_epochs=20, patience=10
```

| center | M | epochs | best quick AUROC | PTB-XL AUROC/AUPRC | PN2021 AUROC/AUPRC | delta vs baseline | first ASR | first entropy/top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra | 5 | 16 | 0.8608 | 0.9067/0.7766 | 0.8346/0.5557 | +0.0002/+0.0031 | 0.3909 | 0.0875/0.9860 |
| extra | 10 | 16 | 0.8609 | 0.9067/0.7765 | 0.8345/0.5553 | +0.0002/+0.0027 | 0.4010 | 0.1902/0.9697 |
| extra | 20 | 20 | 0.8604 | 0.9066/0.7767 | 0.8349/0.5550 | +0.0005/+0.0024 | 0.4010 | 0.3792/0.9398 |
| geo | 5 | 14 | 0.8371 | 0.9066/0.7766 | 0.8340/0.5544 | -0.0004/+0.0018 | 0.4134 | 0.0659/0.9895 |
| geo | 10 | 14 | 0.8371 | 0.9066/0.7765 | 0.8341/0.5554 | -0.0003/+0.0028 | 0.4134 | 0.1399/0.9778 |
| geo | 20 | 12 | 0.8367 | 0.9070/0.7769 | 0.8345/0.5556 | +0.0002/+0.0030 | 0.4190 | 0.2842/0.9550 |
| nin | 5 | 20 | 0.8420 | 0.9065/0.7767 | 0.8343/0.5557 | -0.0001/+0.0031 | 0.3422 | 0.0518/0.9918 |
| nin | 10 | 20 | 0.8423 | 0.9064/0.7765 | 0.8341/0.5560 | -0.0003/+0.0034 | 0.3422 | 0.1098/0.9827 |
| nin | 20 | 20 | 0.8419 | 0.9066/0.7764 | 0.8339/0.5553 | -0.0005/+0.0027 | 0.3422 | 0.2247/0.9648 |

Interpretation after three centers:

- All 9 pilots improve PN2021 AUPRC over baseline by +0.0018 to +0.0034.
- AUROC moves are small and mixed; the largest AUROC gain is `extra M=20` (+0.0005).
- The best AUPRC run is `nin M=10` (PN2021 AUPRC 0.5560, +0.0034).
- The default main setting can remain M=10 because it is the best AUPRC run and avoids
  the more single-neighbor-like M=5 and broader M=20 averaging. For robustness reporting,
  keep M=5 and M=20 as completed ablations.

Prompt-token gated synthetic pool pilot completed:

```text
pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/gated/gated_samples.latent.npz
gate report:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/gated/gate_report.json
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20/
settings:
  M=10, lambda=0.25, hull_steps=5, K_anchor=45, n_epochs=20
```

Gated pool class counts:

| class | passed / generated |
|---|---:|
| NORM | 41 / 50 |
| MI | 41 / 50 |
| STTC | 15 / 50 |

Training/eval:

| run | best quick AUROC | best quick AUPRC | PTB-XL AUROC/AUPRC | PN2021 AUROC/AUPRC | delta vs baseline |
|---|---:|---:|---:|---:|---:|
| prompt-token gated, ningbo pool150->97 | 0.8423 | 0.6697 | 0.9068/0.7772 | 0.8342/0.5555 | -0.0002/+0.0029 |

Interpretation:

- The prompt-token pool is technically usable by Latent-Hull TA-OMAT.
- The clean PN2021 AUPRC gain is close to the real-anchor pilots, but below the
  current `nin M=10` best AUPRC.
- STTC scarcity after gates likely limits this pilot; next prompt-token run should
  over-generate STTC or tune prompt/reference selection before increasing epochs.

Follow-up prompt-token/hybrid attempts:

| run | pool | setting | PN2021 AUROC/AUPRC | delta vs baseline |
|---|---|---|---:|---:|
| prompt v4 balanced | NORM 41, MI 41, STTC 49 | M=10, K_anchor=90 | 0.8344 / 0.5558 | +0.0001 / +0.0032 |
| prompt v3 repeat2 | NORM 69, MI 41, STTC 43 | M=10, K_anchor=120 | 0.8345 / 0.5556 | +0.0002 / +0.0030 |
| prompt v4 M5 adv025 | NORM 41, MI 41, STTC 49 | M=5, adv_weight=0.25 | 0.8343 / 0.5555 | -0.0001 / +0.0029 |
| hybrid real+prompt | real nin 200 + prompt v4 131 | M=10, K_anchor=150 | 0.8345 / 0.5549 | +0.0001 / +0.0023 |
| hybrid real+prompt MI-only | real nin 200 + prompt MI 40 | M=10, K_anchor=150 | 0.8343 / 0.5540 | -0.0001 / +0.0014 |
| hybrid source-aware | real nin 200 + prompt v4 131 | M=10, source-weighted | 0.8342 / 0.5552 | -0.0002 / +0.0026 |
| quality top40 | selected 40/class from prompt candidates | M=10, K_anchor=120 | 0.8343 / 0.5551 | -0.0001 / +0.0025 |
| mixed h40/b40 q50 | prompt candidate mix, 50/class | M=10, K_anchor=120 | 0.8345 / 0.5547 | +0.0001 / +0.0021 |
| mixed h60/b20 q50 | prompt candidate mix, 50/class | M=10, K_anchor=120 | 0.8342 / 0.5542 | -0.0002 / +0.0016 |
| v4 seed13 rerun | v4 balanced | M=10, K_anchor=90 | 0.8343 / 0.5546 | -0.0001 / +0.0020 |
| v4 adv035 | v4 balanced | M=10, K_anchor=90, adv_weight=0.35 | 0.8342 / 0.5554 | -0.0002 / +0.0028 |
| v4 adv045 | v4 balanced | M=10, K_anchor=90, adv_weight=0.45 | 0.8340 / 0.5535 | -0.0004 / +0.0009 |
| fresh v1 quality50 | regenerated v1 token candidates | M=10, K_anchor=120 | 0.8343 / 0.5548 | -0.0001 / +0.0022 |

Interpretation:

- Prompt-token samples can match most of the real-anchor AUPRC lift, but simple
  repeat-token strengthening or real+prompt concatenation did not exceed the
  current real-anchor best.
- The hybrid pool made medical gates easier (`Einthoven p95` about 0.23-0.25)
  but lowered full PN2021 AUPRC; this suggests the combined pool changes the
  training distribution rather than simply improving anchor quality.
- Adding prompt-token samples only to MI lowered full PN2021 AUPRC further. Use
  balanced class/source control for the next hybrid attempt instead of one-class
  supplementation.
- Source-aware hybrid sampling improves over naive hybrid and MI-only hybrid,
  but does not beat pure prompt-token v4. The implementation is still useful for
  controlled ablations where real and prompt pools must coexist.
- Existing prompt-token candidate-pool mixing and `adv_weight` sweeps did not
  beat the original v4 balanced run. Do not keep sweeping downstream online-AT
  knobs blindly.
- Fresh random expansion of the same v1 token generator also did not beat v4;
  stronger upstream reference/token selection is needed before another downstream run.
- Source-aware sampling is now available for controlled hybrid ablations, but the
  next meaningful improvement should happen before online AT: cleaner center-token
  generation, better reference selection, or a larger medically gated v4-style pool.

正式第一轮建议在 real-anchor pool 上跑：

```bash
for M in 5 10 20; do
  /root/miniforge3/envs/ECGTwin/bin/python -u scripts/pgd_cross_center/synth_online_at_super5.py \
    --center_name cpsc_2018_extra \
    --ref_meta_json /root/autodl-tmp/center_token_super5/extra_real_k200.meta.json \
    --synth_npz /root/autodl-tmp/real_anchored_super5/synth_latents/extra_real_k200.latent.npz \
    --class_trust /root/autodl-tmp/real_anchored_super5/synth_latents/extra_real_k200.class_trust.json \
    --output_dir /root/autodl-tmp/latent_hull_super5/extra_M${M}_lambda025 \
    --attack_mode latent_hull \
    --hull_M ${M} \
    --hull_lambda 0.25 \
    --hull_steps 5 \
    --hull_lr 0.3 \
    --K_anchor 300 \
    --pgd_batch 16 \
    --n_epochs 100 \
    --batch_size 128 \
    --num_workers 8
done
```

## Ablation

```text
A0: current real-anchor z0 + delta PGD
A1-small: z0 + same-label convex hull, M=5, fixed lambda=0.25, optimized w
A1-main:  z0 + same-label convex hull, M=10, fixed lambda=0.25, optimized w
A1-wide:  z0 + same-label convex hull, M=20, fixed lambda=0.25, optimized w
A2-light: M=10, lambda in [0.1,0.25,0.4], optimized w
A3: SA-AET-style triangle: z_clean / z_adv_current / z_adv_ema
```

Coefficient-policy implementation status:

```text
DONE scripts/pgd_cross_center/synth_online_at_super5.py --hull_weight_mode
DONE adversarial/latent_hull_pgd.py fixed-weight modes

C0 one_hot:
  --hull_weight_mode one_hot
C1 uniform:
  --hull_weight_mode uniform
C2 random Dirichlet(alpha=1):
  --hull_weight_mode dirichlet --hull_dirichlet_alpha 1.0
C3 optimized softmax weights:
  --hull_weight_mode optimized  # current main/default
```

Completed downstream coefficient-policy tests on `ningbo`, `M=10`,
`lambda=0.25`, `n_epochs=20`:

| mode | run | PN2021 AUROC/AUPRC | interpretation |
|---|---|---:|---|
| C0 one-hot | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C0_onehot_ep20` | 0.8341 / 0.5547 | weaker than optimized |
| C1 uniform | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C1_uniform_ep20` | 0.8340 / 0.5533 | aborted by ASR gate, weak best checkpoint |
| C1 uniform noabort | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C1_uniform_noabort_ep20` | 0.8341 / 0.5549 | still weaker than optimized |
| C2 Dirichlet noabort | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C2_dirichlet_noabort_ep20` | 0.8344 / 0.5550 | best fixed-weight control, below C3 AUPRC |
| C3 optimized | `/root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8341 / 0.5560 | current main/best AUPRC |

Conclusion: optimizing the softmax coefficients inside the same-label hull is
still justified; fixed or random coefficient policies underperform on AUPRC.

A2-light lambda sweep on `ningbo`, `M=10`, optimized weights:

| lambda | run | PN2021 AUROC/AUPRC | interpretation |
|---:|---|---:|---|
| 0.10 | `/root/autodl-tmp/latent_hull_lambda_ablation/nin_M10_lambda010_ep20` | 0.8340 / 0.5546 | too weak |
| 0.25 | `/root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8341 / 0.5560 | current default |
| 0.40 | `/root/autodl-tmp/latent_hull_lambda_ablation/nin_M10_lambda040_ep20` | 0.8339 / 0.5552 | stronger perturbation but lower AUPRC |

PN2021-C follow-up was run for C2 Dirichlet and lambda=0.40. After correcting
all drops against each model's own clean JSON, C3/lambda=0.25 remains the best
absolute corrupted-AUPRC run (`0.492684`) but does not improve self-clean mean
AUPRC drop versus the clean baseline (`0.035135` vs `0.034103`). Report the
PN2021-C result as improved corrupted performance, not reduced relative
corruption sensitivity.

主比较指标：

```text
target-center AUROC/AUPRC delta
PN2021 7-center macro AUROC/AUPRC delta
per-class delta
ASR
gate pass rate
buffer diversity
training time
```

## 风险

- Convex hull 可能比自由 PGD 攻击弱，需要 ASR 和 downstream delta 同时判断。
- Label-preserving 只在约束和 gates 下成立，不能无条件声明。
- NORM 污染风险最高。
- CD/HYP 不适合第一版主结果。
- Teacher gate 不能替代数字心电规则；最终论文需要同时报告 digital sanity 和 downstream utility。

## 2026-05-03 Task-1-Gated Center-Token Latent-Hull Result

Candidate pool:

```text
root:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_20260503/target_token_s05/ningbo/gated
token:
  v42 no-leak style/semantic direct MV4
token_scale:
  0.50
victim gate:
  Task-1 full10 EfficientNet checkpoint, crop_len=1000
gated counts:
  NORM=77, MI=56, STTC=30, total=163
validation:
  no-leak same-label EfficientNet C2ST bacc = 0.7702
  vanilla C2ST bacc = 0.7725
  mean ningbo style prob = 0.2867
  vanilla mean ningbo style prob = 0.0823
```

AT settings:

```text
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_task1gate_scale05/ningbo_s05_M10_lam015_adv003_targetauprc_ep8
attack_mode:
  latent_hull
M:
  10
lambda:
  0.15
hull_steps:
  5
boundary gate:
  p_target in [0.45, 0.65]
adv_weight:
  0.03
anchor_lambda:
  0.20
checkpoint metric:
  target_macro_auprc
```

Formal clean eval, compared with Task-1 full10 baseline:

| metric | Task-1 baseline | scale=0.50 AT | delta |
|---|---:|---:|---:|
| PTB-XL AUROC | 0.9072 | 0.9090 | +0.0018 |
| PTB-XL AUPRC | 0.7744 | 0.7788 | +0.0044 |
| PN2021 avg AUROC | 0.7780 | 0.7782 | +0.0003 |
| PN2021 avg AUPRC | 0.4831 | 0.4825 | -0.0006 |
| ningbo AUROC | 0.8657 | 0.8679 | +0.0022 |
| ningbo AUPRC | 0.4842 | 0.4873 | +0.0031 |

PN2021-C cache eval, 4 centers x 5 corruptions x 5 severities:

| metric | Task-1 baseline | scale=0.50 AT | delta |
|---|---:|---:|---:|
| mean corrupted AUROC | 0.8332 | 0.8335 | +0.0003 |
| mean corrupted AUPRC | 0.5095 | 0.5110 | +0.0015 |
| mean AUROC drop | 0.0095 | 0.0092 | -0.0004 |
| mean AUPRC drop | 0.0141 | 0.0138 | -0.0004 |
| ningbo corrupted AUROC | 0.8547 | 0.8573 | +0.0026 |
| ningbo corrupted AUPRC | 0.4663 | 0.4700 | +0.0037 |

Decision:

```text
This is a narrow positive result:
  center-token scale=0.50 + conservative Latent-Hull AT improves the target
  center ningbo on both clean and corrupted AUROC/AUPRC, and improves PTB-XL.

It is not yet a full PN2021-average success:
  PN2021 7-center AUPRC remains slightly below the Task-1 baseline because
  cpsc_2018 and st_petersburg_incart regress.
```

Implementation update:

```text
scripts/pgd_cross_center/synth_online_at_super5.py now supports:
  --es_metric val_macro_auroc
  --es_metric val_macro_auprc
  --es_metric target_macro_auroc
  --es_metric target_macro_auprc
```

Next AT improvement:

```text
Use target_macro_auprc or val_macro_auprc for checkpointing when the claim is
AUPRC utility. The old AUROC-only checkpoint can select an epoch that is worse
for AUPRC.

Add a soft-label/KL self-distillation variant before another hard-label sweep:
  real PTB-XL: masked BCE
  generated/adv latent-hull ECG: alpha * masked BCE + beta * KL(student, frozen Task-1 teacher)
  start with beta >= alpha for synthetic samples
```

## 2026-05-03 Soft-Label Self-Distillation Pilot

Implemented a first soft-label proxy in:

```text
scripts/pgd_cross_center/synth_online_at_super5.py
```

New arguments:

```text
--adv_label_mode hard|mixed_soft|teacher_soft
--adv_teacher_mix
--adv_soft_target_floor
```

`mixed_soft` uses a frozen copy of the initial Task-1 model as teacher. For
each generated/adv latent-hull ECG, the buffer label is:

```text
y_adv = mix * sigmoid(teacher(x_adv)) + (1 - mix) * one_hot(target_class)
y_adv[target_class] = max(y_adv[target_class], soft_target_floor)
```

First pilot:

```text
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_task1gate_scale05/ningbo_s05_M10_lam015_adv003_softmix07_valauprc_ep8
source pool:
  ningbo target-token scale=0.50 gated pool
attack:
  latent_hull, M=10, lambda=0.15, hull_steps=5
adv labels:
  mixed_soft, teacher_mix=0.7, target_floor=0.55
checkpoint metric:
  val_macro_auprc
```

Quick eval looked positive at epoch 8:

| metric | baseline quick | softmix ep8 |
|---|---:|---:|
| avg AUROC | 0.8305 | 0.8347 |
| avg AUPRC | 0.6508 | 0.6529 |
| ningbo AUROC | 0.8466 | 0.8561 |
| ningbo AUPRC | 0.6496 | 0.6561 |

Formal clean eval against Task-1 baseline:

| metric | Task-1 baseline | hard target-AUPRC AT | softmix07 val-AUPRC AT |
|---|---:|---:|---:|
| PTB-XL AUROC | 0.9072 | 0.9090 | 0.9085 |
| PTB-XL AUPRC | 0.7744 | 0.7788 | 0.7779 |
| PN2021 avg AUROC | 0.7780 | 0.7782 | 0.7767 |
| PN2021 avg AUPRC | 0.4831 | 0.4825 | 0.4817 |
| ningbo AUROC | 0.8657 | 0.8679 | 0.8681 |
| ningbo AUPRC | 0.4842 | 0.4873 | 0.4875 |

Decision:

```text
softmix07 is useful evidence that target-center gains remain, but it does not
replace the current main hard-label checkpoint because PN2021 average AUROC/AUPRC
regresses more. Keep hard target-AUPRC AT as the current main result.

If soft labels are revisited, test a lighter teacher mix such as 0.3 or implement
a true separate KL term with lower synthetic weight instead of putting full soft
labels through pos_weighted BCE.
```

Follow-up `teacher_mix=0.3`:

```text
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_task1gate_scale05/ningbo_s05_M10_lam015_adv003_softmix03_targetauprc_ep8
checkpoint metric:
  target_macro_auprc
formal clean eval:
  PTB-XL:     0.9085 / 0.7778
  PN2021 avg: 0.7767 / 0.4816
  ningbo:     0.8681 / 0.4875
```

This is effectively the same as `teacher_mix=0.7` and still worse than the
hard-label target-AUPRC main checkpoint on PN2021 average. Under the current
`adv_weight=0.03` setting, changing the soft-label mixing ratio is not a
meaningful lever.

## 2026-05-04 v45 Real-Anchor + NORM/MI Synthetic Follow-Up

The v45 contrastive center token was tested in the real-anchor online-AT route
with paired target-token/no-token synthetic samples:

```text
paired pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/paired_token_delta_v45_norm_mi_ningbo_20260504

paired counts:
  NORM=223
  MI=123

real anchors:
  ningbo K=500

online AT:
  target_real_weight=20
  attack_mode=latent_hull
  hull_M=10
  hull_lambda=0.15
  adv_weight=0.06
  adv_label_mode=mixed_soft
  adv_teacher_mix=0.3
```

Formal ref-excluded eval:

| run | source weight | PN2021 avg AUROC | PN2021 avg AUPRC | held-out ningbo AUROC | held-out ningbo AUPRC |
|---|---:|---:|---:|---:|---:|
| Task-1 baseline | n/a | 0.7780 | 0.4831 | 0.8657 | 0.4842 |
| v45 target-token | 0.4 | 0.7813 | 0.4893 | 0.8882 | 0.5159 |
| v45 no-token | 0.4 | 0.7806 | 0.4890 | 0.8884 | 0.5165 |
| v45 target-token | 1.0 | 0.7810 | 0.4896 | 0.8882 | 0.5168 |
| v45 no-token | 1.0 | 0.7808 | 0.4887 | 0.8882 | 0.5158 |

Decision:

```text
This route now meets the +2pp target-center adaptation criterion versus the
PTB-XL-only Task-1 baseline on held-out ningbo. The effect is not yet center-token
causal because matched no-token remains tied within ~0.1pp AUPRC.

The next latent-hull change should not be another source-weight sweep. It should
make the center-style signal enter the adversarial objective or sample selection
more directly, for example by adding style-probe/C2ST-aware anchor scoring or a
feature-distance regularizer, then rerunning matched token/no-token controls.
```

## 2026-05-04 Georgia Follow-Up Sweeps

Georgia remained below the +2pp/+2pp target after the first multi-center
target-real run:

| run | georgia AUROC | georgia AUPRC | delta vs baseline |
|---|---:|---:|---:|
| PTB-XL-only baseline, K=500 refs excluded | 0.8157 | 0.5916 | n/a |
| v42 target-token + real-anchor AT, w40 src0.4 adv0.06 | 0.8262 | 0.6029 | +1.05pp / +1.13pp |

Follow-up sweeps:

| run | change | georgia AUROC | georgia AUPRC |
|---|---|---:|---:|
| `georgia_v42_target_w80_src04_M10_lam015_adv006_softmix03_ep10` | target_real_weight 40 -> 80 | 0.8118 | 0.5848 |
| `georgia_v42_target_w40_src20_M10_lam015_adv006_softmix03_ep10` | prompt_token source weight 0.4 -> 2.0 | 0.8113 | 0.5847 |
| `georgia_v36_target_w40_src04_M10_lam015_adv006_softmix03_ep10` | larger v36 MI/STTC prompt-token pool | 0.8116 | 0.5848 |
| `georgia_v42_target_w40_src04_M10_lam015_adv003_softmix03_ep10` | adv_weight 0.06 -> 0.03 | 0.8119 | 0.5843 |
| `georgia_v42_target_w40_src04_M10_lam015_adv006_softmix03_hypcd_ep10` | enable CD/HYP trust, exclude MI from scope | 0.8113 | 0.5846 |

Decision:

```text
Do not continue scalar sweeps for georgia. Stronger target-real supervision,
more prompt-token source sampling, more MI/STTC prompt-token anchors, lower
adv_weight, and enabling CD/HYP anchors all underperform the old
w40/src0.4/adv0.06 run on full-center eval.

Label audit:
  full georgia has only 7 MI positives under the project super5 mapping;
  the K=500 selected refs contain all 7;
  after ref exclusion, formal georgia eval has no MI positives and uses
  CD/HYP/NORM/STTC only.

Next georgia work should inspect the target sample selection, PN2021 super5
mapping, and center-specific label noise before more online-AT sweeps.
```

## 2026-05-16 论文主线收缩：VAE-Only Real-Anchor Latent-Hull AT

当前论文路线先收缩到最干净的问题：

```text
不使用 ECGTwin DiT 合成样本
不使用 center token
只使用目标中心真实 ECG 的 ECGTwin VAE latent
在 same-label VAE latent hull 内做在线对抗训练
```

这样做的原因：

```text
1. real-anchor LH-AT 已经在 ningbo / chapman_shaoxing / cpsc_2018 上稳定提升。
2. center token 与 no-token 在若干 matched control 中差异很小，因果性不稳定。
3. VAE latent manifold 本身是更稳的论文贡献点：用目标中心 K 条真实 ECG 构造
   医学语义更可信的局部 latent hull，然后在 hull 内搜索决策边界样本。
```

### 固定输入协议

所有实验继续使用 Task-1 source classifier：

```text
checkpoint:
  /root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt

preprocess_mode = minimal_resample
norm_mode       = per_sample_global
sampling rate   = 100Hz
input length    = 1000 samples = 10 seconds
crop_len        = 1000
lead order      = PTB-XL canonical order
```

目标中心 anchor：

```text
K = 500 first
centers = ningbo, cpsc_2018 first-stage
then expand to chapman_shaoxing, georgia
eval must exclude the K target-center ref ids
```

### 变体定义

主方法 A：anchored lambda hull。

```text
z_mix = sum_i softmax(a_i) * z_i
z_adv = (1 - lambda) * z0 + lambda * z_mix
```

其中：

```text
z0 = 当前目标中心真实 ECG 的 VAE latent
z_i = same-primary-label 近邻目标中心真实 ECG latent
a_i = 每个 batch 在线优化的组合权重 logits
M = 参与组合的 same-label 近邻数量
```

推荐主线：

```text
lambda = 0.15
M = 10 or 20
epochs = 30
```

消融 B：no-lambda neighbor convex hull。

```text
z_adv = sum_i softmax(a_i) * z_i
z_i 不包含 z0
```

这对应 `hull_lambda=1.0` 且不加 `--hull_include_anchor`。它是更激进的
no-lambda 版本，可能更强也更容易离开原始 ECG 局部邻域。

消融 C：no-lambda anchor-included convex hull。

```text
z_adv = sum_i softmax(a_i) * z_i
z_i 包含 z0，且 z0 是 candidate 0
```

这对应：

```text
--hull_lambda 1.0
--hull_include_anchor
```

它是更稳的 no-lambda 版本，因为 softmax 权重可以选择留在原始 anchor 附近。

### M Sweep

完整 M 数组：

```text
M = 10, 20, 30, 40, 50, 60, 70, 80, 100
```

执行顺序：

```text
phase 1:
  M = 10, 20, 40, 80
  centers = ningbo, cpsc_2018
  variants = lambda015, convex_anchor
  epochs = 30

phase 2:
  around best region add M = 30, 50, 60, 70, 100
  add convex_neighbors only if convex_anchor is competitive

phase 3:
  expand best 2-3 configs to chapman_shaoxing and georgia
```

预期：

```text
M=10/20:
  局部同类 manifold 最干净，预期最稳。

M=30/40/50:
  增加多样性，可能略增 AUPRC，但也可能稀释中心局部风格。

M=60/80/100:
  更像同类全局原型混合，风险是 latent 均值化和医学证据变弱。
  只作为 scale-up ablation，不作为默认主线。
```

已有 ningbo exact M grid 提示：

```text
lambda=0.15, K=500, crop_len=1000:
  M10 ep30 ~= 0.8946 / 0.5303
  M20 ep30 ~= 0.8945 / 0.5303
  M20 ep20 ~= 0.8942 / 0.5295
  M100 ep20 ~= 0.8944 / 0.5294

结论：
  M 从 10/20 扩到 100 没有明显继续提升；
  epoch 从 10 增到 20/30 的收益更明确。
```

### Epoch Scale-Up

推荐 epoch 数组：

```text
epochs = 10, 20, 30, 40, 60
```

执行策略：

```text
M=10/20:
  跑 20, 30, 40 epoch。

M=30/50:
  跑 20, 30 epoch。

M=80/100:
  先跑 10, 20 epoch。
  若 20 epoch 无优势，不继续拉长。
```

当前判断：

```text
epoch=30 是主线默认。
epoch=40 可作为上限确认。
epoch=60 只在最优 M/variant 上做一次，不做全网格。
```

### 新实现

新增代码：

```text
scripts/pgd_cross_center/synth_online_at_super5.py
  --hull_include_anchor
  --disable_quality_gate

scripts/paper/run_vae_only_latenthull_sweep_20260516.py
  default: hard quality gate disabled for VAE-only real-anchor LH-AT
```

`--hull_include_anchor` 的作用：

```text
SameLabelLatentIndex 默认排除 anchor 自身。
打开该开关后，候选集合 candidate 0 = z0，后面再接 same-label nearest neighbors。
这让 no-lambda convex hull 可以表达“保持在原始 ECG 附近”。
```

### Quality Gate 策略

论文主方法默认关闭 hard quality gate：

```text
scripts/paper/run_vae_only_latenthull_sweep_20260516.py
  default behavior: pass --disable_quality_gate to synth_online_at_super5.py
```

原因：

```text
2026-05-16 no-gate ablation 显示，在 real-anchor VAE-only LH-AT 主线中，
gate-on 和 no-gate 的 AUROC/AUPRC 完全一致；原 gate-on 日志没有
medical gate FAIL 或 skip=True。

这说明当前 real-anchor latent-hull 样本本身通过 semantic gate，
性能收益不是由 gate-based sample selection 带来的。
```

保留 gate 指标，但只作为 monitoring / safety audit：

```text
Einthoven residual
HR / QRS sanity
NaN / Inf
amplitude / flatline
ASR trace
```

如果需要复现 gate-on 消融，使用：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_vae_only_latenthull_sweep_20260516.py \
  --enable_quality_gate
```

已完成 no-gate matched ablation：

```text
output:
  /root/autodl-tmp/paper_vae_only_latenthull_nogate_ablation_20260516/

matched cells:
  ningbo, M=20/80
  cpsc_2018, M=20/80
  K=500, lambda=0.15, epochs=30, seed=20260531

result:
  gate-on vs no-gate target AUROC/AUPRC delta = 0.000 / 0.000 pp
  gate-on vs no-gate PN2021 avg delta        = 0.000 / 0.000 pp
```

### 第一轮命令

建议先跑小网格：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_vae_only_latenthull_sweep_20260516.py \
  --centers ningbo cpsc_2018 \
  --K 500 \
  --variants lambda015 convex_anchor \
  --Ms 10 20 40 80 \
  --epochs 30 \
  --num_workers 6
```

输出：

```text
/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/
```

如果 phase 1 结果显示 `convex_anchor` 不输给 `lambda015`，再补：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/paper/run_vae_only_latenthull_sweep_20260516.py \
  --centers ningbo cpsc_2018 \
  --K 500 \
  --variants lambda015 convex_anchor convex_neighbors \
  --Ms 10 20 30 40 50 60 70 80 100 \
  --epochs 20 30 \
  --num_workers 6
```

### 验收标准

最小论文可用标准：

```text
1. 至少两个中心相对 bare PTB-XL baseline 提升：
   target AUROC >= +2pp 或 target AUPRC >= +3pp。

2. 目标中心 ref ids 全部从 eval 中排除。

3. PTB-XL fold10 AUPRC 不下降超过 2pp；若下降，需要解释为
   target adaptation vs source retention trade-off。

4. no-lambda 版本必须和 lambda015 对比：
   如果 no-lambda 不稳定，论文主方法保留 lambda015；
   如果 no-lambda 接近或更好，可作为更简洁的主公式候选。
```
