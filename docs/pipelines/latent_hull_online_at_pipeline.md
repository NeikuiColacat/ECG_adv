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
- `M=100` 暂不进入第一轮，因为大候选池容易均值化 latent、降低医学证据强度，并显著增加 decode/victim backward 成本。

推荐名称：

```text
Latent-Hull TA-OMAT
```

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
