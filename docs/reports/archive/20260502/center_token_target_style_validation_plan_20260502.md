# Center Token 目标中心风格验证方案

日期：2026-05-02

本文档定义如何验证：在 ECGTwin prompt 中加入目标中心 `center token`
之后，生成 ECG 是否比 vanilla ECGTwin 更符合目标 PN2021 中心风格。

这不是医学诊断有效性的唯一证明，而是 center-token 机制证据。最终仍需结合
digital ECG sanity、super5 语义一致性和 PN2021 下游 AUROC/AUPRC。

## 1. 核心问题

我们要回答的问题是：

```text
给定同一个 super5 类别和相近 reference ECG，
target-center-token ECGTwin 生成样本是否比 vanilla/source-token ECGTwin
更接近目标中心真实 ECG 分布？
```

不能只看 7-way center classifier accuracy。旧版 center style classifier
测试准确率约 0.378，高于随机 1/7=0.143，但它只是 smoke test：

- 没有排除 center-token 训练使用的 K=500 anchors；
- 没有按 super5 label 做 same-label 匹配；
- 输入只用 250 点 crop，容易漏掉 10 秒全局中心风格；
- 7-way 分类会混合中心相似性、类别分布差异和样本量不平衡。

正式验证应使用多指标证据链。

## 2. 对照方法

每个目标中心至少比较 4 组：

| method | 说明 | 作用 |
|---|---|---|
| `real_target` | held-out 目标中心真实 ECG | 上界/参照分布 |
| `real_non_target` | 其他中心同类别真实 ECG | 负对照 |
| `vanilla_ecgtwin` | 不加 center token 的 ECGTwin 生成样本 | 生成器默认风格 |
| `target_center_token` | 加目标中心 token 的 ECGTwin 生成样本 | 待验证方法 |

可选增加：

| method | 说明 |
|---|---|
| `ptbxl_source_token` | 加 PTB-XL/source 风格 token |
| `wrong_center_token` | 加非目标中心 token，作为强负对照 |

优先目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

优先类别：

```text
NORM, MI, STTC
```

`CD` 和 `HYP` 先作为 exploratory，因为当前生成质量和 digital gate 更不稳定。

## 3. 数据划分与泄漏控制

对每个目标中心 `c`：

```text
A_c_anchor = K=500 center-token 训练/生成 reference ECG
D_c_real   = target center 剩余真实 ECG
G_c        = target-center-token 生成 ECG
```

强制规则：

1. `A_c_anchor` 只允许用于 center token 训练、reference selection 和生成。
2. `A_c_anchor` 禁止进入 domain discriminator 的 train/val/test。
3. `A_c_anchor` 禁止进入下游同中心 validation/eval。
4. `G_c` 只用于最终 query/eval，不能参与 discriminator 训练或阈值选择。
5. split 文件必须保存 `record_ids`，方便审计。

推荐 split：

```text
target center:
  anchors: K=500, fixed by ref_selection JSON
  train:   remaining real ECG 60%
  val:     remaining real ECG 20%
  test:    remaining real ECG 20%

non-target centers:
  train/val/test 按同样比例固定划分
```

K=500 ref id 来源：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1/ref_selection/<center>_k500_seed42.json
```

生成池 metadata：

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/.../<center>/gated/gated_samples.ref_meta.json
```

## 4. Same-Label 控制

所有域判别和 two-sample test 都必须按 super5 label 分层。

原因：PN2021 各中心疾病分布差异很大。如果不控制 label，判别器可能学到
“哪个中心 MI/STTC/NORM 更多”，而不是中心采集风格。

优先匹配策略：

1. 精确 multi-hot super5 label set 匹配；
2. 样本不足时退化为 primary class 匹配；
3. NORM 只和纯 NORM 比；
4. 每个 center-class 测试样本少于 30 时，只报告 exploratory。

报告维度：

```text
center x class x method
```

## 5. 验证指标一：Target-Likeness Probe

### 5.1 目标

训练只使用真实 ECG 的目标中心判别器：

```text
positive = target center real ECG
negative = other centers real ECG
query    = synthetic ECG
```

然后比较：

```text
P(target | target_center_token)
P(target | vanilla_ecgtwin)
P(target | wrong_center_token)
P(target | real_target)
P(target | real_non_target)
```

理想结果：

```text
real_target 高
real_non_target 低
target_center_token > vanilla_ecgtwin
target_center_token 接近 real_target
wrong_center_token 低于 target_center_token
```

### 5.2 模型选择

不要只依赖 raw ECG 端到端分类器。建议三路 probe：

| probe | 输入 | 模型 | 解释 |
|---|---|---|---|
| EfficientNet feature probe | 冻结 PTB-XL super5 EfficientNet1DV2 penultimate feature | logistic / 2-layer MLP | 最贴近下游分类空间 |
| ECGTwin latent probe | ECGTwin VAE encode 后 `(4,128)` latent flatten | logistic / 2-layer MLP | 生成模型空间证据 |
| Raw waveform probe | `(12,1000)` full 10s ECG | SE-ResNet/EfficientNet full-length | 信号级敏感分析 |

主报告优先使用 EfficientNet feature probe 和 ECGTwin latent probe。
Raw waveform probe 可能学到滤波、幅值、噪声或设备伪迹，只作辅助。

### 5.3 指标

判别器自身：

```text
target-vs-rest AUROC
target-vs-rest AUPRC
balanced accuracy
macro F1
per-center confusion
calibration: Brier / ECE
```

synthetic target-likeness：

```text
mean P(target)
median P(target)
% samples above target real test median
target-likeness percentile
bootstrap 95% CI
```

`target-likeness percentile` 定义：

```text
synthetic score 在 held-out real target same-label score 分布中的百分位
```

示例解释：

```text
ningbo-token MI synthetic 位于 held-out ningbo MI score 的第 62 百分位。
```

## 6. 验证指标二：Real-vs-Synth C2ST

### 6.1 目标

对每个 target center 和每个 class 做 classifier two-sample test：

```text
class 0 = held-out real target ECG
class 1 = synthetic target ECG
```

如果 synthetic 接近 target real，C2ST 应难以区分两者。

### 6.2 对照

必须比较：

```text
C2ST(real_target, target_center_token)
C2ST(real_target, vanilla_ecgtwin)
C2ST(real_target, wrong_center_token)
```

理想结果：

```text
C2ST AUROC(target_center_token) < C2ST AUROC(vanilla_ecgtwin)
C2ST balanced accuracy 越接近 0.5 越好
Proxy A-distance 越低越好
```

Proxy A-distance：

```text
PAD = 4 * balanced_accuracy - 2
```

`balanced_accuracy = 0.5` 时 `PAD = 0`，表示当前 probe 下两分布不可分。

### 6.3 注意

C2ST 接近 0.5 不是医学有效性证明，可能只是 probe 不够强。因此必须配合
target-likeness、feature distance、digital ECG 和 downstream utility。

## 7. 验证指标三：Feature Distribution Distance

在固定 feature space 中比较分布距离：

```text
distance(target_center_token, real_target)
distance(vanilla_ecgtwin, real_target)
distance(wrong_center_token, real_target)
distance(real_non_target, real_target)
```

推荐 feature space：

1. EfficientNet1DV2 penultimate feature；
2. ECGTwin VAE latent；
3. ECG digital feature vector。

推荐指标：

```text
MMD
KID
FID/rFID
energy distance
precision / recall / coverage
```

理想结果：

```text
target_center_token 与 real_target 的距离最低，
或者明显低于 vanilla_ecgtwin / wrong_center_token。
```

## 8. 验证指标四：Digital ECG Style Features

这部分验证生成 ECG 是否保留医学和信号层面的合理性，同时检查是否靠近目标中心
统计风格。

可比较的 feature：

```text
HR distribution
RR irregularity
PR interval
QRS duration
ST deviation
T wave amplitude
lead-wise amplitude / p2p
lead covariance
Einthoven residual
aVR residual
baseline wander proxy
high-frequency noise proxy
```

推荐统计检验：

```text
KS test
Wasserstein distance
Jensen-Shannon divergence
per-feature z-distance to target real
```

注意：digital feature 不能代替波形级验证，但能防止 domain probe 被非医学伪迹欺骗。

## 9. 验证指标五：Downstream Utility

最终仍要看下游实用性：

```text
EfficientNetV2 + target-center-token synthetic augmentation
vs
EfficientNetV2 baseline / vanilla ECGTwin augmentation
```

主要指标：

```text
PN2021 7-center macro AUROC/AUPRC
target center AUROC/AUPRC
per-class AUROC/AUPRC
PN2021-C corrupted absolute performance
```

判定逻辑：

| 情况 | 解释 |
|---|---|
| target-likeness 上升，C2ST 降低，下游 AUPRC 上升 | 最理想，说明更像目标中心且有用 |
| target-likeness 上升，C2ST 仍高，下游不升 | token 学到部分中心特征，但 synthetic artifact 仍强 |
| C2ST 降低，target-likeness 不升 | synthetic 更真实，但不一定像目标中心 |
| target-likeness 上升，下游下降 | 可能学到目标中心伪迹或标签语义被污染 |

## 10. 推荐第一版实验

### 10.1 数据

```text
target centers:
  ningbo
  chapman_shaoxing
  cpsc_2018
  georgia

classes:
  NORM
  MI
  STTC
```

### 10.2 Probe

第一版不追求强 raw classifier，先做 feature-space probe：

```text
1. EfficientNet1DV2 penultimate feature + 2-layer MLP
2. ECGTwin VAE latent + 2-layer MLP
3. digital feature + logistic regression
```

### 10.3 主表

```text
center | class | method | target_probe_score ↑ | target_percentile ↑ |
C2ST_AUROC ↓ | PAD ↓ | MMD ↓ | digital_style_distance ↓ |
downstream_AUPRC_delta ↑
```

### 10.4 判定标准

一个 center-token 方法可被认为“生成数据更符合目标中心风格”，至少应满足：

1. `target_center_token` 的 target-probe score 高于 `vanilla_ecgtwin`；
2. `target_center_token` 到 `real_target` 的 feature distance 低于 vanilla；
3. same-label C2ST 不比 vanilla 更容易区分；
4. digital ECG sanity 不明显恶化；
5. 下游 target center 或 PN2021 avg AUPRC 不下降，最好上升。

## 11. 建议脚本落点

新增脚本：

```text
scripts/ecgtwin_gen/train_prompt_token_domain_discriminator.py
scripts/ecgtwin_gen/eval_prompt_token_domain_discriminator.py
scripts/ecgtwin_gen/compare_prompt_token_target_style.py
```

输出目录：

```text
/root/autodl-tmp/domain_discriminator/center_token_style_v1/
```

建议输出文件：

```text
split_record_ids.json
probe_train_result.json
target_likeness_scores.csv
c2st_results.csv
feature_distance_results.csv
digital_style_results.csv
center_token_style_validation_report.md
```

## 12. 论文表述建议

谨慎表述：

```text
The target-center prompt token shifts generated ECGs toward the target-center
distribution under real-only target-likeness probes and feature-space
two-sample tests.
```

不要表述为：

```text
Generated ECGs are clinically identical to real target-center ECGs.
```

更稳妥的中文表述：

```text
加入目标中心 token 后，生成 ECG 在冻结 EfficientNet 特征空间、ECGTwin VAE
潜空间和数字 ECG 统计特征上均更接近目标中心 held-out 真实 ECG；同时，
real-only 目标中心判别器给出的 target-likeness score 高于 vanilla ECGTwin。
因此，center token 对目标中心风格具有可观测的调制作用。
```

如果下游 AUPRC 没有提升，应补充：

```text
这种目标中心风格迁移尚未稳定转化为下游诊断性能提升，说明生成分布对齐和
诊断有效增强之间仍存在差距。
```
