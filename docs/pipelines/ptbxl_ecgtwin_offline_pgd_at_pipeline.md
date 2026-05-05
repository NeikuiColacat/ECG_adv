# PTB-XL ECGTwin 10k Synthetic Offline PGD-AT Pipeline

本文档定义一个新的 PTB-XL in-domain 实验计划：

```text
PTB-XL train folds 1-8
-> 抽取约 1000 条 ECGTwin-compatible reference ECG
-> ECGTwin 生成 10000 条 super5 synthetic ECG
-> 在 synthetic VAE latent 上一次性离线 PGD
-> 用固定 adversarial buffer 微调 EfficientNet1DV2
-> 检查 PTB-XL fold10 AUROC/AUPRC 是否提升
```

这是一个 **PTB-XL source-domain downstream utility** 实验。它不替代当前
PN2021 跨中心主线；它回答一个更窄的问题：

```text
ECGTwin + VAE latent PGD 是否能改善 EfficientNet1DV2 在 PTB-XL 自身测试集上的
macro AUROC / macro AUPRC？
```

## 2026-05-02 执行记录

已完成：

```text
25-ref / 10-synth smoke                       PASS
1000-ref all5 balanced source pool            PASS
10000-synth all5 balanced ECGTwin pool         PASS
10-sample offline PGD-AT smoke                PASS
10000-sample free-PGD eps=2.0,K=10 main        negative
10000-sample free-PGD eps=0.5,K=5 conservative close / slight AUPRC gain
direct synthetic augmentation control              negative
trusted3 PGD buffer control                        near-flat, below all5 aw010
latent-hull M5 n=2000 pilot                        near-flat, most stable AUROC
synth quality report                               PASS
PN2021 v3 clean 7-center eval for best run     PASS
```

关键产物：

```text
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.pt
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.meta.json
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.latent.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_buffers/freepgd_eps2_k10_adv10000_seed42.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_buffers/freepgd_eps05_k5_adv10000_seed42.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_buffers/freepgd_eps05_k5_trusted3_adv_seed42.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_buffers/latent_hull_M5_lam025_steps3_adv2000_seed42.npz
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.quality.json
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.quality.md
```

同口径 baseline：

| model | PTB-XL AUROC | PTB-XL AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC |
|---|---:|---:|---:|---:|
| `/root/autodl-tmp/triple_labels/super5` | 0.9064 | 0.7754 | 0.8344 | 0.5526 |

已跑离线 PGD-AT 结果：

| run | PGD | adv weight | best epoch | PTB-XL AUROC | PTB-XL AUPRC | PN2021 avg AUROC | PN2021 avg AUPRC | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `freepgd_eps2_k10_ep20_seed42` | eps=2.0,K=10 | 0.50 | 1 | 0.8925 | 0.7498 | not run | not run | too strong, harms source-domain utility |
| `freepgd_eps05_k5_ep15_aw025_seed42` | eps=0.5,K=5 | 0.25 | 1 | 0.9050 | 0.7736 | not run | not run | close but still below baseline |
| `freepgd_eps05_k5_ep15_aw010_seed42` | eps=0.5,K=5 | 0.10 | 1 | 0.9062 | 0.7756 | 0.8341 | 0.5535 | AUROC near-flat/slightly lower, AUPRC slight gain |
| `freepgd_eps05_k5_trusted3_aw010_seed42` | eps=0.5,K=5, CD/HYP trust=0 | 0.10 | 1 | 0.9062 | 0.7751 | 0.8337 | 0.5533 | removing CD/HYP did not beat all5 aw010 |
| `latent_hull_M5_lam025_steps3_n2000_aw010_seed42` | same-label hull, M=5,n=2000 | 0.10 | 1 | 0.9064 | 0.7754 | 0.8343 | 0.5533 | most stable AUROC, but gain is within noise |
| `direct_synth_r025_seed42` | no PGD, train from scratch | synth ratio 0.25 | 25 | 0.9010 | 0.7605 | 0.8277 | 0.5401 | direct waveform augmentation is clearly negative |

当前结论：

```text
VAE latent PGD can create a useful weak regularizer only when perturbation and
sampling weight are conservative. Strong free-PGD is destructive. The best
current signal is small. Conservative free-PGD gives the best AUPRC deltas
(PTB-XL +0.0002, PN2021 +0.0009) but slightly lowers AUROC. The 2000-sample
latent-hull pilot is the most stable AUROC-wise but its gain is within noise.
Direct synthetic augmentation is clearly negative, so any weak positive signal
comes from conservative latent-space regularization, not from adding ECGTwin
waveform samples directly. Do not claim a strong improvement yet.
```

仍未完成：

```text
per-class raw-mV digital gates, if a raw ECGTwin decode export is added
larger latent-hull comparison, if we decide the near-flat pilot is worth scaling
```

Quality summary for the 10k synthetic pool:

| class | n | finite | victim top1 hit | target prob mean | Einthoven p95 mean | HR mean |
|---|---:|---:|---:|---:|---:|---:|
| CD | 2000 | 1.0000 | 0.7245 | 0.8098 | 0.1053 | 78.19 |
| HYP | 2000 | 1.0000 | 0.0265 | 0.5094 | 0.0950 | 77.82 |
| MI | 2000 | 1.0000 | 0.2780 | 0.6420 | 0.0923 | 85.09 |
| NORM | 2000 | 1.0000 | 0.8415 | 0.8334 | 0.1427 | 74.54 |
| STTC | 2000 | 1.0000 | 0.6440 | 0.7468 | 0.0915 | 81.95 |

Interpretation:

```text
All samples are finite and Einthoven residual is low. HYP and MI have weak
victim top1 consistency, especially HYP. The current .npz is z-scored
classifier-scale, so voltage-threshold medical validity cannot be asserted
from this export; raw decode export is needed for true mV digital gates.
```

## 当前事实

已有 PTB-XL ECGTwin cache：

```text
/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt
/root/autodl-tmp/ptbxl/PTBXL_vae_multi_nomic.pt
```

格式：

```text
list length = 21799
sample["data"]              -> torch.float32 (4,128)
sample["label"]["text"]     -> diagnostic text
sample["label"]["text_embed"] -> (L,768)
sample["label"]["hr"]
sample["label"]["age"]
sample["label"]["sex"]
sample["label"]["diagnostic_class"]
sample["label"]["strat_fold"]
```

该 cache 来自 `data/prepare_ptbxl_for_ecgtwin.py`：

```text
raw100.npy (100Hz, 1000,12)
-> resample to 1024
-> PTB-XL lead order 转 ECGTwin lead order
-> VAE encode
-> nomic text embedding
```

注意：这不是作者论文里最理想的 500Hz 原始 PTB-XL 下采样路径，而是从
`raw100.npy` 插值到 1024。工程上可用；论文表述时要说明来源。

现有可复用代码：

| 功能 | 文件 |
|---|---|
| ECGTwin PTB-XL cache 构建 | `data/prepare_ptbxl_for_ecgtwin.py` |
| ECGTwin 批量生成 synthetic signals/latents | `trash/cleanup_20260506_legacy/scripts_ecgtwin_gen/generate_center_synth.py`（旧入口） |
| EfficientNet1DV2 super5 baseline/real+synth 训练 | `scripts/triple_labels/train_ptbxl.py` |
| Super5 online latent-PGD/Latent-Hull AT 组件 | `scripts/pgd_cross_center/synth_online_at_super5.py` |
| VAE latent PGD generator | `adversarial/pgd_advdiff.py` |
| Same-label latent hull generator | `adversarial/latent_hull_pgd.py` |
| Tier-M offline AT 骨架 | `scripts/crosscenter_tierM/offline_adv_train_tierM.py` |

缺口：

```text
没有专门的 super5 + PTB-XL + ECGTwin-generated pool 的 offline PGD-AT 入口。
```

## 数据规则

PTB-XL split 固定：

| split | folds |
|---|---|
| train/ref/generation source | 1-8 |
| validation/model selection | 9 |
| final test | 10 |

严格禁止从 fold 9/10 抽 ECGTwin reference。否则 generated/PGD 数据会把验证或测试分布泄漏回训练。

super5 class order 固定：

```text
CD, HYP, MI, NORM, STTC
```

第一版 reference selection：

```text
total_refs = 1000
per_class_refs = 200
folds = 1-8 only
seed = 42
```

默认按 `diagnostic_class` 分层抽样。后续更严谨版本应从
`scripts/triple_labels/label_schemes.py` 重新生成 `super5_multi_hot`，避免把 PTB-XL 多标签样本误压成单标签。

第一版 synthetic generation：

```text
total_synth = 10000
per_class_synth = 2000
synthetic_per_ref = 10
classes = CD HYP MI NORM STTC
```

HYP/CD 必须生成并记录质量结果，但若数字 ECG gate 或 victim consistency 明显失败，下游训练主结果应同时报告：

```text
A. all5: CD/HYP/MI/NORM/STTC 全部使用
B. trusted3: 仅 NORM/MI/STTC 使用
```

这样既满足 all-super5 生成需求，也保留医学质量风险的可解释消融。

## ECGTwin 生成设计

每个 synthetic 样本使用同类 PTB-XL ref：

```text
ref ECG latent        -> IBE base_vector
ref text_embed        -> IBE reference text condition
ref HR / age / sex    -> patient info condition
target text_embed     -> 默认同 ref text_embed
DDPM random noise     -> 产生多样性
```

例如生成 MI：

```text
PTB-XL train fold 1-8 中 diagnostic_class=MI 的 ref ECG
-> ref latent + ref text_embed + HR/age/sex 抽 base_vector
-> target text_embed 仍为该 MI ref 的 text condition
-> ECGTwin sample
```

第一版不使用 center prompt-token，因为本实验目标是 PTB-XL source-domain synthetic utility，不是目标中心迁移。center token 可作为后续版本。

ECGTwin 输出到 EfficientNet1DV2 的转换固定：

```text
ECGTwin decoder output: (B,1024,12), ECGTwin/MIMIC lead order
-> transpose to (B,12,1024)
-> resample 1024 -> 1000
-> ECGTwin lead order -> PTB-XL lead order
-> per-sample global z-score
-> save signals as (N,12,1000)
```

生成产物：

```text
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/
  refs/ptbxl_train1000_balanced_seed42.pt
  refs/ptbxl_train1000_balanced_seed42.meta.json
  synth/ptbxl_train1000_to_synth10000.npz
  synth/ptbxl_train1000_to_synth10000.latent.npz
  synth/ptbxl_train1000_to_synth10000.ref_trace.json
  synth/ptbxl_train1000_to_synth10000.quality.json
  synth/ptbxl_train1000_to_synth10000.quality.md
```

`.npz` schema：

```text
signals:     (10000,12,1000) float32, PTB-XL order, classifier scale
labels:      (10000,5) float32, one-hot target class
center_name: "ptbxl_source"
```

`.latent.npz` schema：

```text
latents:     (10000,4,128) float32, generated ECGTwin latent
labels:      (10000,5) float32
center_name: "ptbxl_source"
source_ids/source_names optional
```

## 需要新增或修改的最小代码

### 1. 新增 PTB-XL ref subset 脚本

新增：

```text
scripts/ecgtwin_gen/prep_ptbxl_super5_ref_subset.py
```

功能：

```text
--ptbxl_cache datasets/PTBXL/PTBXL_vae_multi_nomic.pt
--out_pt /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.pt
--out_meta /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.meta.json
--folds 1 2 3 4 5 6 7 8
--total_refs 1000
--per_class 200
--classes CD HYP MI NORM STTC
--seed 42
```

输出仍保持 `generate_center_synth.py` 可读的 list-of-dict schema。

增强项：

```text
label["super5_multi_hot"] = true PTB-XL super5 multi-hot
label["record_id"] 或 ecg_id/patient_id
label["source_fold"]
```

### 2. 修改 `generate_center_synth.py`

现状：

```text
--scheme super5 时硬编码只生成 NORM/MI/STTC
```

需要改成显式参数：

```text
--classes CD HYP MI NORM STTC
--trusted_default false
--save_ref_trace
```

第一版命令：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u trash/cleanup_20260506_legacy/scripts_ecgtwin_gen/generate_center_synth.py \
  --scheme super5 \
  --ref_pt /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/refs/ptbxl_train1000_balanced_seed42.pt \
  --center_name ptbxl_source \
  --out /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.npz \
  --classes CD HYP MI NORM STTC \
  --n_per_class 2000 \
  --batch_size 50 \
  --num_steps 50 \
  --seed 42 \
  --device cuda:0 \
  --save_latent
```

若 `--classes` 尚未实现，先实现再运行，不要手动改硬编码常量来跑一次性实验。

### 3. 新增 Super5 offline PGD-AT 入口

新增：

```text
scripts/pgd_cross_center/offline_synth_pgd_at_super5.py
```

设计来自：

```text
scripts/pgd_cross_center/synth_online_at_super5.py
scripts/crosscenter_tierM/offline_adv_train_tierM.py
```

Phase 0：一次性离线 PGD，冻结 baseline EfficientNet1DV2。

```text
input synthetic latent pool: (10000,4,128)
for each synthetic latent z0:
  z_adv = z0 + PGD_delta
  decode z_adv
  convert to PTB-XL classifier format
  gate
  push into fixed buffer
save offline_adv_buffer.npz
```

Phase 1：固定 buffer 训练，不再每 epoch 重生成。

```text
PTB-XL real train fold 1-8
+ optional VAE roundtrip anchor
+ fixed offline PGD buffer
-> EfficientNet1DV2 fine-tune
-> select by fold9 macro AUROC
-> final fold10 AUROC/AUPRC
```

主参数：

```text
--synth_npz /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.latent.npz
--init_ckpt /root/autodl-tmp/triple_labels/super5/best_model.pt
--output_dir /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_at/freepgd_eps2_k10_ep20
--offline_adv_npz /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/offline_buffers/freepgd_eps2_k10_adv10000.npz
--attack_mode pgd
--pgd_eps 2.0
--pgd_K 10
--pgd_batch 32
--n_adv_total 10000
--ptbxl_weight 1.0
--adv_weight 0.5
--roundtrip_weight 0.5
--roundtrip_anchor_n 1500
--n_epochs 20
--batch_size 128
--num_workers 8
--lr 5e-5
--weight_decay 1e-4
--anchor_lambda 0.05
--seed 42
```

Label policy：

```text
offline PGD buffer 训练时优先使用 target-only masked labels:
  target class = 1
  non-target classes = -1
```

原因：ECGTwin synthetic 由单一 class prompt/ref 生成，非目标类阴性并不一定可靠。masked label 能减少错误负标签污染。

对照实验可使用 hard one-hot：

```text
target class = 1
non-target classes = 0
```

但不作为第一主结果。

## 实验阶段

### Stage 0: Baseline 锁定

确认 baseline：

```text
/root/autodl-tmp/triple_labels/super5/best_model.pt
/root/autodl-tmp/triple_labels/super5/train_result.json
```

当前记忆基线：

```text
PTB-XL fold10 macro AUROC/AUPRC ~= 0.9064 / 0.7754
```

如果 `train_result.json` 与该记忆不一致，以文件为准。

### Stage 1: Ref Subset Smoke

先抽小集确认 schema：

```text
per_class_refs = 5
total_refs = 25
```

检查：

```text
class counts
folds only 1-8
text_embed shapes finite
latent shape (4,128)
```

### Stage 2: ECGTwin Generation Smoke

先生成：

```text
classes = CD HYP MI NORM STTC
n_per_class = 2
total = 10
num_steps = 10 or 25
```

检查：

```text
signals finite
latents shape (N,4,128)
labels shape (N,5)
lead conversion works
EfficientNet forward works
```

### Stage 3: Full 10k Synthetic Pool

生成主 pool：

```text
2000 samples/class
10000 total
num_steps = 50
batch_size = 50
```

质量报告：

```text
finite / NaN / Inf
amplitude p2p
flatline / saturation
Einthoven/aVR residual
HR range
victim class consistency
per-class pass rate
```

HYP/CD 单独报告，不要因为它们进入 all5 训练就宣称医学有效。

### Stage 4: Direct Synthetic Augmentation Control

在 PGD-AT 前先跑一个直接增强对照：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/triple_labels/train_ptbxl.py \
  --scheme super5 \
  --output_dir /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/direct_synth_r025 \
  --synth_npz /root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/synth/ptbxl_train1000_to_synth10000.npz \
  --synth_ratio 0.25 \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 50 \
  --device cuda
```

目的：区分“ECGTwin synthetic 本身”与“latent-PGD 对抗训练”的贡献。

### Stage 5: Offline PGD Buffer Smoke

先对 50-100 个 synthetic latent 做 PGD：

```text
n_adv_total = 100
pgd_eps = 2.0
pgd_K = 3
pgd_batch = 16
```

检查：

```text
decode finite
ASR not near 0
medical gate not catastrophic
buffer labels are masked correctly
training one epoch runs
```

### Stage 6: Full Offline PGD-AT

主实验：

```text
n_adv_total = 10000
pgd_eps = 2.0
pgd_K = 10
n_epochs = 20
adv_weight = 0.5
```

执行结果：`eps=2.0,K=10,adv_weight=0.5` 过强，fold10 降到
`0.8925 / 0.7498`。后续主线改用保守设置：

```text
pgd_eps = 0.5
pgd_K = 5
adv_weight = 0.10
lr = 2e-5
anchor_lambda = 0.10
```

候选消融：

| run | purpose |
|---|---|
| `freepgd_eps1_k5_ep20` | conservative |
| `freepgd_eps2_k10_ep20` | main |
| `freepgd_eps2_k10_trusted3_ep20` | only NORM/MI/STTC |
| `latent_hull_M10_lam025_ep20` | optional manifold-constrained comparison |

第一轮已跑 main、两个 conservative 消融、direct synthetic control、trusted3
和 2000-sample latent-hull pilot。下一步若继续本路线，优先做 raw decode
quality export 或扩大 latent-hull 到 10k。

### Stage 7: Evaluation

主指标：

```text
PTB-XL fold10 macro AUROC
PTB-XL fold10 macro AUPRC
per-class AUROC/AUPRC
fold9 best epoch metric
```

辅助指标：

```text
PN2021 v3 clean 7-center AUROC/AUPRC
PN2021-C absolute corrupted AUPRC
```

辅助指标不是本计划的主判定条件，但可以确认 PTB-XL in-domain 提升是否牺牲外部泛化。

## 成功标准

最低有效信号：

```text
PTB-XL fold10 macro AUPRC >= baseline + 0.002
且 macro AUROC 不下降超过 0.001
```

较强结果：

```text
PTB-XL fold10 macro AUROC 和 AUPRC 同时提升
per-class 中 HYP/CD 至少一个有改善，且 NORM/MI/STTC 不明显下降
```

如果 PTB-XL 提升但 PN2021 明显下降，则报告为 source-domain regularization，不作为跨中心增强主结论。

## 预计资源

数据大小粗估：

```text
10000 signals (12,1000,float32) ~= 480 MB raw, npz 约 300-600 MB
10000 latents (4,128,float32)  ~= 20 MB raw
offline adv buffer ~= 300-600 MB
```

所有产物放在：

```text
/root/autodl-tmp/ptbxl_ecgtwin_offline_pgd/
```

不要写入系统盘或 repo。

4090D 24GB 推荐：

```text
ECGTwin generation batch_size = 50
PGD batch = 16-32
EfficientNet batch_size = 128
num_workers = 4-8
AMP/bf16 where safe
```

预计时长需要用 smoke 实测。第一轮不要直接开 10k 全流程；先完成 10-sample generation smoke 和 100-sample PGD smoke。

## 风险与处理

1. **HYP/CD 生成质量风险**

   当前项目经验显示 HYP/CD 的数字 ECG gate 较弱。计划中仍生成 all5，但训练主结论必须同时报告 trusted3 对照。

2. **PTB-XL cache 来源风险**

   现有 cache 从 `raw100.npy` 插值到 1024，不是 500Hz 原始 PTB-XL。若本实验出现明显正向信号，后续应重建 500Hz -> 1024 的高保真 ECGTwin cache。

3. **多标签压缩风险**

   现有 `diagnostic_class` 是 primary/single class。第一版可跑通；正式版本应补 `super5_multi_hot` 并记录 ref 的完整标签。

4. **离线 PGD 过强导致医学语义漂移**

   第一主线使用 gate + target-only masked labels。若 free PGD gate 失败，切换到 `latent_hull` same-label convex hull。

5. **只提升 PTB-XL 不提升 PN2021**

   这是可接受结果，但论文表述只能说 in-domain utility/regularization，不说跨中心泛化增强。

## 需要确认的问题

默认假设如下，若用户不修改就按这个执行：

1. 10000 synthetic 使用 **all5 balanced**：每类 2000。
2. 1000 PTB-XL refs 只从 folds 1-8 抽：每类 200。
3. HYP/CD 先生成并评估；如果 gate 太差，训练主结果补跑 trusted3。
4. Offline PGD 默认是一条 synthetic latent 生成一条 adversarial ECG，即 `n_adv_total=10000`。
5. 模型选择只看 fold9；fold10 只做最终测试。

## 执行清单

- DONE: 新增 `prep_ptbxl_super5_ref_subset.py`。
- DONE: 修改 `generate_center_synth.py`，支持 `--classes` all5。
- DONE: 生成 25-ref / 10-synth smoke。
- DONE: 生成 1000-ref / 10000-synth full pool。
- DONE: 生成 synth quality report。
- DONE: 跑 direct synthetic augmentation control。
- DONE: 新增 `offline_synth_pgd_at_super5.py`。
- DONE: 跑 10-sample offline PGD smoke。
- DONE: 跑 10000-sample offline PGD-AT main。
- DONE: 跑 trusted3 PGD buffer control。
- DONE: 汇总 baseline/direct-synth/offline-PGD 的 PTB-XL AUROC/AUPRC。
- DONE: 跑 latent-hull 小规模对照。
- TODO: raw decode mV export + stricter digital quality gates。
