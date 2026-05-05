# Final Round Ablation Pipeline

Date: 2026-05-04.

本文档根据 `docs/final_round/` 中的任务书、开题报告、中期报告和指导记录，
补齐毕业设计最终答辩需要的消融实验计划。目标不是继续无限扩大训练，而是回答论文中
最容易被追问的几个问题：

```text
1. ECGTwin 生成的 ECG 是否具有医学语义合法性？
2. center prompt token 是否真的改变目标中心风格？
3. 下游 AUROC/AUPRC 提升来自 ECGTwin/center token/在线对抗训练的哪一部分？
4. 方法是否提升跨中心鲁棒性，而不是只提升单个验证集？
5. 扩散采样与训练策略是否满足系统设计中的效率要求？
```

## Thesis Requirement Mapping

`docs/final_round` 给出的关键要求可以映射为以下实验模块：

| 毕设材料中的要求 | 需要补齐的实验 |
|---|---|
| 生成数据要保持医学语义一致 | ECG 数字指标、导联一致性、类别语义一致性、人工可视化抽样 |
| 解决真实医疗数据稀缺和类别不平衡 | 低样本 real2000 vs ECGTwin 合成增强消融 |
| 面向跨中心域偏移提升鲁棒性 | PN2021 7-center 和 target-center K=500 在线 AT 消融 |
| 噪声干扰和个体差异下保持性能 | PN2021-C corruption benchmark |
| 采样速度和系统可用性 | DDPM/DDIM 或不同采样步数的速度-质量消融 |
| 多目标损失权重需要控制变量 | token loss、Latent-Hull AT 参数和 label strategy 消融 |

## Scope

主任务仍然是 PTB-XL super5：

```text
classes = CD, HYP, MI, NORM, STTC
classifier input = (N, 1000, 12), 100Hz, 10s, PTB-XL lead order
ECGTwin VAE latent = (N, 4, 128)
large outputs = /root/autodl-tmp/final_round_ablation_20260504/
```

评测数据：

```text
PTB-XL fold10 / custom low-resource test
PN2021 clean 7 centers
PN2021-C: ningbo, chapman_shaoxing, cpsc_2018, georgia
```

默认对照组：

```text
real-only EfficientNet1DV2
ECGTwin no-token synthetic
ECGTwin target center-token synthetic
wrong-center token synthetic
wrong-class prompt synthetic
real-anchor Latent-Hull online AT
target-real + no-token synthetic online AT
target-real + target-token synthetic online AT
```

## Ablation 1: Generated ECG Medical Semantic Legality

目的：证明生成 ECG 不是纯粹让分类器分数变高的伪影，而是在基础 ECG 物理和医学规则上合理。

输入样本：

```text
R0: PTB-XL real same-class ECG
R1: PN2021 target-center real same-class ECG
S0: ECGTwin no-token synthetic
S1: ECGTwin target center-token synthetic
S2: wrong-center token synthetic
S3: wrong-class prompt synthetic
S4: Latent-Hull decoded adversarial samples
```

第一层信号合法性 gate：

```text
NaN / Inf rate
flatline rate
amplitude p2p range
DC offset range
lead order sanity
Einthoven residual: lead I + lead III ~= lead II
aVR residual consistency
estimated HR range
RR variability outlier rate
```

第二层类别医学语义 gate：

```text
NORM:
  abnormal class probabilities should stay low
  HR and QRS should stay in plausible normal range

MI:
  MI classifier probability should increase versus no-token/wrong-class controls
  ST/Q-wave related digital features should be reported when available

STTC:
  ST/T abnormality score or proxy features should be higher than NORM controls

HYP:
  voltage criteria should be checked, but do not overclaim if per-sample z-score removes amplitude semantics

CD:
  QRS duration / bundle-branch-block proxy should be checked; keep as limitation if weak
```

第三层可视化审计：

```text
per class sample 16 real + 16 no-token + 16 target-token + 16 wrong-token ECG figures
12-lead gallery with consistent scale
save abnormal examples and failed gate examples, not only successful examples
```

输出：

```text
/root/autodl-tmp/final_round_ablation_20260504/medical_validity/
  per_sample_quality.csv
  per_class_quality.csv
  lead_consistency.json
  digital_feature_summary.json
  semantic_gate_summary.json
  figures/
  medical_validity_report.md
```

通过标准：

```text
target-token synthetic pass rate should be close to no-token and not much worse than real ECG.
wrong-class prompt should fail semantic consistency more often than same-class prompt.
HYP/CD can be reported as weaker generated classes if voltage/QRS gates are unstable.
```

论文表述边界：

```text
这些 gate 只能证明 basic medical plausibility / semantic consistency，
不能宣称合成 ECG 已达到临床诊断级有效性。
```

## Ablation 2: Prompt/Class Semantic Consistency

目的：验证 ECGTwin 对 super5 prompt 的条件控制是否有效。

对照：

```text
C0: same-class prompt, no token
C1: same-class prompt, target-center token
C2: same-class prompt, wrong-center token
C3: wrong-class prompt, target-center token
C4: shuffled prompt embedding
```

指标：

```text
frozen EfficientNet1DV2 target-class probability
macro class consistency rate
target class top-1 / top-2 rate
multi-label BCE against intended super5 label
NORM abnormal suppression rate
```

注意：

```text
同一个 frozen classifier 不能作为唯一医学证据。
该实验只回答 prompt 是否控制了分类语义，必须和 Ablation 1 的数字 ECG gate 一起报告。
```

## Ablation 3: Center Token Target-Style Validity

目的：回答 `<center_CLASS>` 是否真的让 ECGTwin 生成更像目标中心，而不是只改变随机种子。

对照：

```text
T0: no-token
T1: target-center token
T2: wrong-center token
T3: PTB-XL-source token
T4: token_scale = 0.25 / 0.50 / 1.00
T5: MV1 / MV4 / MV8 token length
```

验证器：

```text
no-leak center-style classifier
same-label C2ST balanced accuracy
EfficientNet penultimate feature MMD/FID
real-vs-synth precision/recall in feature space
per-center target-likeness probability
```

防泄漏规则：

```text
style classifier train/val/test split must be record-disjoint.
K=500 target refs used for token training must be excluded from final target-center eval.
C2ST must compare same-label real vs synthetic; otherwise class signal can masquerade as center style.
```

通过标准：

```text
target-center token should beat no-token and wrong-center token on at least two independent style metrics.
If downstream AT improves but target-token does not beat no-token, the paper should attribute gains to
target-real/Latent-Hull adaptation rather than center token itself.
```

## Ablation 4: Downstream Utility And Causal Attribution

目的：把 AUROC/AUPRC 提升拆开，避免把所有增益都归因到 center token。

低样本路线：

| arm | train data | purpose |
|---|---|---|
| L0 | real2000 only | low-resource baseline |
| L1 | real2000 + no-token ECGTwin self-distill | ECGTwin synthetic utility |
| L2 | real2000 + target-token ECGTwin self-distill | center-token incremental utility |
| L3 | real2000 + wrong-token ECGTwin self-distill | negative control |
| L4 | L2 synthetic pretrain -> real2000 clean fine-tune | current strongest low-resource recipe |

全量跨中心路线：

| arm | method | purpose |
|---|---|---|
| F0 | PTB-XL folds 1-8 real-only EfficientNet1DV2 | full baseline |
| F1 | F0 + target-real K=500 supervised stream | target real adaptation |
| F2 | F0 + real-anchor Latent-Hull AT | VAE latent hull value |
| F3 | F0 + target-real + no-token synthetic latent pool | ECGTwin latent candidates |
| F4 | F0 + target-real + target-token synthetic latent pool | center-token incremental value |
| F5 | F0 + target-real + wrong-center token pool | negative control |

主指标：

```text
PTB-XL fold10 macro AUROC/AUPRC
PN2021 7-center macro AUROC/AUPRC
PN2021 per-center AUROC/AUPRC
PN2021 per-class AUROC/AUPRC
target-center eval with K=500 ref ids excluded
```

结论规则：

```text
L2/L4 > L1 supports center token in low-resource setting.
F4 > F3 supports center token in full cross-center AT setting.
F3 > F1/F2 supports ECGTwin synthetic latent candidates.
F1/F2/F3/F4 > F0 supports target-center adaptation over bare PTB-XL training.
```

## Ablation 5: Latent-Hull Online AT Hyperparameters

目的：验证在线对抗样本不是任意 latent 混合，而是有约束的 label-preserving boundary sample。

参数矩阵：

```text
M candidates: 5, 10, 20
lambda: 0.10, 0.15, 0.25, 0.40
adv_weight: 0.03, 0.06, 0.10
target_real_weight: 0, 4, 12, 20, 40
hull weight mode: onehot, uniform, dirichlet, optimized
soft label mode: hard label, teacher soft label, anchor-target mix
```

医学语义约束：

```text
same-label convex hull only.
NORM cannot mix with abnormal labels.
If class probability moves outside accepted band or abnormal suppression fails, reject the adversarial sample.
For multi-label ECG, preserve the source multi-hot label unless the semantic gate indicates label drift.
```

推荐第一轮：

```text
M = 10
lambda = 0.15
adv_weight = 0.06
target_real_weight = 20
hull weight mode = optimized
teacher soft label = on for a controlled follow-up only
```

## Ablation 6: PN2021-C Robustness

目的：呼应中期报告中“噪声干扰和个体差异”的要求。

固定腐蚀：

```text
powerline_noise
emg_noise
baseline_wander
baseline_shift
random_leads_masking
severity = 1..5
centers = ningbo, chapman_shaoxing, cpsc_2018, georgia
```

对比模型：

```text
F0 full PTB-XL baseline
best low-resource ECGTwin route if applicable
best real-anchor Latent-Hull AT
best target-real + no-token synthetic AT
best target-real + target-token synthetic AT
```

报告方式：

```text
absolute corrupted AUROC/AUPRC
self-clean AUROC/AUPRC drop
per-corruption severity curve
per-center drop
```

结论边界：

```text
如果 enhanced model absolute corrupted AUPRC 更高但 self-clean drop 更大，
只能说提升了 corrupted absolute performance，不能说相对鲁棒性已经全面改善。
```

## Ablation 7: Sampling Speed Versus Quality

目的：回应任务书/中期报告中扩散采样效率问题。

对照：

```text
sampling steps = 10, 25, 50, 100
sampler = current DDPM path; DDIM only if existing wrapper supports it cleanly
batch size = 16 / 32 / max safe under 4090D 24GB
AMP = bf16 where numerically safe
```

指标：

```text
seconds per 100 ECG
GPU memory peak
medical validity pass rate
semantic consistency rate
feature FID/MMD
downstream gate pass count
```

通过标准：

```text
25 or 50 steps should be the thesis default if quality loss is small.
Do not claim DDIM acceleration unless the exact sampler path is implemented and logged.
```

## Ablation 8: Loss And Token Design

目的：把 center token 训练的可调项记录成控制变量，而不是经验调参。

token design:

```text
single-vector vs MV4 vs MV8
direct vs factorized center+class+residual
token_scale 0.25 / 0.50 / 1.00
init from class prompt embedding vs random small noise
actual report ref text vs class-fallback ref text
same-class ref vs random-any ref
```

loss terms:

```text
reconstruction denoise loss
init regularization
style CE / target probability loss
semantic BCE preservation loss
contrast token-vs-no-token reconstruction loss
EfficientNet feature alignment loss
feature MMD loss
orthogonality penalty for multiple prompt vectors
```

推荐答辩表述：

```text
MV4 direct token 是当前最稳妥主线。
MV8 和 factorized token 可作为容量增大但不稳定的消融。
token_scale=0.50 已经有窄正向证据，scale=1.00 容易过强。
```

## Execution Order

第一阶段：不训练长模型，先补生成质量证据。

```text
1. 选取每类 real/no-token/target-token/wrong-token/wrong-class 样本。
2. 跑 signal sanity、lead consistency、semantic classifier gate。
3. 输出医学合法性表格和 12-lead gallery。
```

第二阶段：补 center-token 因果验证。

```text
4. 在同一候选池上跑 no-leak style classifier、same-label C2ST、feature MMD。
5. 对比 token_scale 和 wrong-center controls。
```

第三阶段：补下游因果 attribution。

```text
6. 低样本路线复核 L0/L1/L2/L3/L4。
7. 全量路线复核 F0/F1/F2/F3/F4/F5。
8. 所有 target-center eval 排除 K=500 ref ids。
```

第四阶段：鲁棒性和效率。

```text
9. 对最有代表性的 3-5 个模型跑 PN2021-C。
10. 对生成器跑 steps/throughput/quality 消融。
```

## Acceptance Criteria For Final Thesis

最低可答辩证据：

```text
1. 有医学合法性 gate 表，且生成 ECG 的基础合法性不明显崩坏。
2. 有 same-class vs wrong-class prompt 语义一致性对照。
3. 有 no-token / target-token / wrong-token 的 center-style 对照。
4. 有 real-only / no-token / target-token 的下游 AUROC/AUPRC 对照。
5. 有 PN2021 clean 和 PN2021-C 的跨中心鲁棒性结果。
6. 有采样步数或吞吐量记录，说明系统效率取舍。
```

理想论文结论：

```text
ECGTwin 生成样本在基础 ECG 物理约束和 super5 语义一致性上通过多数 gate；
在低样本 PTB-XL 场景中，ECGTwin 合成数据结合自蒸馏能够提升 AUROC/AUPRC；
在跨中心场景中，目标中心真实 anchor 与 VAE Latent-Hull 在线对抗训练是主要增益来源；
center token 可以作为目标风格控制机制，但必须用 no-token/wrong-token 对照限定其贡献。
```

不能过度表述：

```text
不能把 frozen classifier 分数等同于临床有效性。
不能把 target-real K=500 带来的提升全部归因于 center token。
不能在 HYP/CD 数字 gate 不稳定时宣称五类生成都同等可靠。
不能只报告 PN2021 平均值而忽略 per-center/per-class 退化。
```

## Expected Final Artifacts

```text
/root/autodl-tmp/final_round_ablation_20260504/
  medical_validity/
  prompt_semantic_consistency/
  center_style_validity/
  downstream_attribution/
  pn2021_c_robustness/
  sampling_speed_quality/
  final_round_ablation_summary.md
```

论文中建议产出表格：

```text
Table A: generated ECG medical validity pass rate by class and method
Table B: prompt semantic consistency, same-class vs wrong-class
Table C: center-style validation, target-token vs no-token vs wrong-token
Table D: low-resource downstream AUROC/AUPRC attribution
Table E: full PTB-XL cross-center online AT attribution
Table F: PN2021-C corruption robustness
Table G: sampling steps vs quality vs speed
```

## 2026-05-04 MVP Execution Status

已落地脚本：

```text
scripts/final_round/run_medical_validity_ablation.py
```

本轮先执行不需要重新训练的必要消融：

```text
target_token_s05:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_token_scale_large_20260504/target_token_s05/ningbo/gated/gated_samples.npz

no_token:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/effectiveness_pilot_v42_task1gate_ningbo_no_token_large_20260504/no_token/ningbo/gated/gated_samples.npz

v4_balanced_token:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated/gated_samples.npz
```

输出：

```text
/root/autodl-tmp/final_round_ablation_20260504/medical_validity/
  per_sample_quality.csv
  medical_validity_summary.json
  medical_validity_report.md
```

cap=64 初步结果：

| arm | n | quality pass | fail | semantic top1 | target prob >=0.5 | mean target prob |
|---|---:|---:|---:|---:|---:|---:|
| target_token_s05 | 64 | 0.031 | 0.000 | 0.938 | 1.000 | 0.907 |
| no_token | 64 | 0.094 | 0.000 | 0.906 | 0.938 | 0.881 |
| v4_balanced_token | 64 | 0.531 | 0.000 | 0.797 | 0.906 | 0.838 |

解释：

```text
1. 三组样本没有出现 hard fail，说明基础数值合法性没有崩坏。
2. target_token_s05 的语义一致性最高，但导联一致性 warning 较多。
3. v4_balanced_token 的基础 ECG gate 更稳，但 target-class probability 较低。
4. 论文中应将该实验表述为 basic plausibility + semantic proxy，
   不应表述为临床诊断级医学有效性证明。
```
