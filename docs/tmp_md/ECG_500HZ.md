# ECG VAE 与 500Hz 输入协议调研备忘

日期：2026-06-01
项目：ECG_adv_Gen / ECGTwin VAE Latent-Hull Online AT
关注问题：当前 ECGTwin VAE 是否适合 500Hz 主线、是否需要自训练 VAE、PTB-XL/PN2021/SPH/ECGFounder/DeepECG 的采样率如何对齐，以及这件事对论文归因有什么影响。

## 结论先行

1. **当前 ECGTwin VAE 不是 500Hz VAE。**
   当前项目记忆中，ECGTwin VAE 输入/输出是 `(B, 1024, 12)`，raw mV，ECGTwin/MIMIC lead order，latent 是 `(B, 4, 128)`。当前 EfficientNet1DV2 主线把 decode 后的 1024 点重排导联并 resample 到 1000 点，服务的是 `100Hz / 10s / 1000 samples` 分类器协议。

2. **它仍然适合当前 EfficientNet 主线，但定位要收紧。**
   对 EfficientNet1DV2 来说，ECGTwin VAE 最合理的角色是“on-manifold adversarial regularizer / latent hard-example search space”，不是高保真 500Hz ECG 生成器。论文里不要说它保留了 500Hz 级别的细节。

3. **如果后续主线切到 ECGFounder 或 500Hz classifier，最好重训一个 500Hz PTB-XL VAE。**
   ECGFounder 输入协议是 `12 x 5000`，也就是 10 秒 500Hz。把 ECGTwin VAE 的 1024 点 decode 插值到 5000 点只能满足 shape，不能恢复已经不在 VAE latent 里的高频形态。

4. **PTB-XL 的 2 万条足够训练“项目内 500Hz VAE / latent regularizer”，但不足以声称通用临床生成器。**
   PTB-XL 有约 2.18 万条 10 秒 12 导联记录，包含 500Hz 版本。用它训练一个专门服务 PTB-XL Super5 与跨中心适配的 VAE 是可行的；但若要声称泛化到所有中心、罕见病种和临床高保真生成，则需要更多数据和更强验证。

5. **论文归因的关键不是“VAE 是不是外部数据增强”，而是控制变量。**
   如果 VAE 只用 PTB-XL source 训练，再在 PN2021 target-center K500 上做 latent adversarial search，闭环最干净。若 VAE 用了 PN2021 target/test 或其他目标域大数据，就会变成外部/目标域数据增强，必须如实声明，并用 matched direct、random latent mixup、input-space PGD 等对照证明 adversarial search 的独立贡献。

## 采样率与输入协议核对

| 对象 | 采样率 / 输入长度 | 结论 | 对本项目的影响 |
|---|---:|---|---|
| PTB-XL | public `records500` 为 500Hz，另有 `records100` 为 100Hz | 可作为训练 500Hz VAE 的 source 数据 | 建议用 fold 1-8 train、fold 9 val、fold 10 test；不要把 fold10 用进 VAE model selection |
| PN2021 / Challenge 2021 | 非统一 500Hz。CPSC、Georgia、Chapman-Shaoxing、Ningbo 是 500Hz；PTB/PTB-XL 源有 500/1000Hz；INCART 是 257Hz；UMich 是 250/500Hz | 我们关注的四个 target centers 大多可对齐到 500Hz，但 PN2021 总体不能假设全是 500Hz | 所有 PN2021 cache 必须记录 native fs、resample policy、lead order、length |
| SPH | 500Hz，10-60 秒，12 导联 | 是很适合 500Hz ECG 表示学习的外部数据候选 | 若用于 VAE 预训练，论文里必须声明外部数据；不能把收益完全归因为跨中心对抗训练 |
| ECGFounder | `12 x 5000`，10 秒 500Hz | 500Hz 兼容性最强的 foundation backbone | 如果走 ECGFounder 主线，500Hz VAE 更自然 |
| DeepECG / WCR | 本项目适配文档记录为 `12 x 2500`，10 秒 250Hz | 不是 500Hz 模型 | 不要为了 DeepECG 把全项目改成 500Hz；只在 DeepECG adapter 转成 2500 点 |
| 当前 EfficientNet1DV2 主线 | `100Hz / 1000 samples / 10s` | 当前最稳定主线 | 现有 ECGTwin VAE decode 1024 -> 1000 是合理工程折中 |
| 当前 ECGTwin VAE | `(B,1024,12)` decode，latent `(B,4,128)` | 不是 500Hz 保真 VAE | 用作 latent regularizer 可以；若主张 500Hz morphology fidelity，需要新 VAE |

## 当前 ECGTwin VAE 的适配性判断

### 适合的场景

- EfficientNet1DV2 `100Hz / 1000 samples` 主线。
- target-center K500 real anchors 的 latent-hull online AT。
- 同标签 / exact multi-hot latent convex hull。
- 证明“在真实目标中心 ECG 的 VAE latent 附近搜索困难样本，可以提升目标中心泛化”。

### 不适合强行承担的场景

- 直接作为 500Hz ECGFounder 的高保真生成器。
- 证明 ECGTwin VAE 保留了高频 QRS、细粒度 pacing spike、极短时程形态。
- 用插值后的 `1024 -> 5000` 波形声称“500Hz 生成”。
- 在没有 ECG sanity、digital feature、lead consistency、attack diagnostics 的情况下声称临床有效。

### 当前最稳表述

> We use the ECGTwin VAE as a learned latent manifold for target-anchor constrained adversarial regularization, not as evidence that the decoder alone generates clinically faithful 500Hz ECGs.

中文表述：

> 我们把 ECGTwin VAE 作为目标中心真实 ECG 附近的隐空间正则器和困难样本搜索空间，而不是把它作为高保真 500Hz ECG 生成器来主张。

## 是否需要重新训练 500Hz VAE

### 什么时候不急

如果最终主论文仍以 EfficientNet1DV2 为主，且 classifier 输入固定为 `100Hz / 1000 samples`，当前 VAE 可以继续使用。此时要做的是补强证据链：

- matched direct baseline；
- random latent mixup 对照；
- input-space PGD 对照；
- no-PGD latent mix 对照；
- attack diagnostics：`loss_gain`、`atk_init`、`atk_anchor`、positive-hide / negative-add ASR、invalid decode rate；
- VAE/AugMix stream 拆开。

### 什么时候应该重训

如果后续要把 ECGFounder 或 500Hz EfficientNet 作为主线，就建议训练一个新的 500Hz VAE：

```text
PTB-XL records500
-> lead order canonicalization
-> 10s crop/pad = 5000 samples
-> mV unit sanity
-> optional ECGFounder-style filtering branch
-> per-record or dataset-level normalization, frozen in metadata
-> 12-lead 5000-point VAE
```

目标不是追求最大生成多样性，而是得到一个和 500Hz classifier 输入协议一致的可微 latent manifold。

## 2 万条 PTB-XL 是否足够

结论：**足够训练项目专用 VAE，不足以单独支撑通用临床生成器叙事。**

理由：

- PTB-XL 是大规模公开 12 导联数据集，提供约 2.18 万条 10 秒记录，并有标准 fold。
- 外部 VAE 先例中，有工作用约 2,033 条 10 秒 500Hz ECG 切成约 25 万个 cardiac cycles 来训练 cycle-level VAE，说明 ECG VAE 不一定要求百万级样本才能收敛。
- 但是完整 `12 x 5000` 的 10 秒 12 导联 VAE 比单周期、单导联或 lead-II VAE 难很多，罕见类覆盖也会不足。
- 因此，PTB-XL-only 500Hz VAE 最适合做 source-domain latent regularizer，而不是作为通用生成模型。

推荐定位：

```text
PTB-XL-only 500Hz VAE
= source-domain ECG latent manifold
= fair source-only regularizer
!= universal ECG generator
```

## 论文归因风险

### 干净方案

```text
VAE training data: PTB-XL train folds only
classifier source: PTB-XL train folds
target adaptation: PN2021 target-center K=500 real anchors
final evaluation: target-center ref-excluded PN2021 held-out
```

这种情况下，VAE 没有提前见过 PN2021 held-out，论文可以说：

> latent adversarial training improves cross-center adaptation over matched K500 direct fine-tuning.

### 风险方案

```text
VAE training data includes PN2021 full target center
or includes PN2021 test/held-out style data
or includes broad external ECG data without matched controls
```

这不一定错误，但论文主张会变成：

> external-data-pretrained generative manifold plus target adaptation improves performance.

这比“对抗训练本身起作用”弱很多。此时必须加入：

- same external VAE + random latent mixup；
- same external VAE + no adversarial search；
- input-space PGD；
- direct K500 matched protocol；
- source-only VAE 与 external-data VAE 的对照；
- VAE 训练数据来源和病人/中心泄漏审计。

## 推荐的 500Hz VAE 训练方案

### 数据

主线只用 PTB-XL：

```text
train: fold 1-8
val: fold 9
test/reconstruction audit: fold 10 only for final report
input: records500 / filename_hr
shape: (12, 5000)
unit: mV
labels: Super5 only for stratified audit, VAE 本身可无监督
```

可选外部预训练只放 future work：

```text
MIMIC-IV-ECG / SPH / Chapman-Shaoxing / PN2021 source-only
```

如果使用外部数据，必须在标题和实验命名中写清楚。

### 模型

建议先做 1D convolutional VAE，不要直接上很复杂的 diffusion VAE：

```text
Encoder:
  Conv1D blocks over time
  lead-aware channel mixing
  downsample 5000 -> latent temporal map

Latent:
  z shape could be (Cz, Tz), e.g. (8, 156) or (16, 80)
  use KL warmup / beta schedule

Decoder:
  upsample temporal map back to 5000
  reconstruct 12 leads jointly
```

关键是 latent 要能做局部插值和 PGD，而不是只追 reconstruction loss。

### Loss

基础：

```text
L = L_recon + beta * KL
```

推荐加入：

```text
L_recon = Huber(x_hat, x)
        + alpha * multi_resolution_STFT_loss
        + gamma * lead_consistency_loss
```

ECG lead consistency 可以检查：

```text
I + III ~= II
aVR ~= -(I + II) / 2
aVL ~= I - II / 2
aVF ~= II - I / 2
```

不要让模型只学会平滑平均波形。QRS、ST-T、T wave morphology 需要专门审计。

### 验证

不要只看 MSE。建议保存：

- reconstruction MAE/MSE by lead；
- QRS / ST / T morphology error；
- amplitude p2p 分布；
- lead consistency residual；
- latent interpolation decode sanity；
- same-label latent hull invalid rate；
- classifier teacher consistency；
- downstream direct vs random mixup vs adversarial search。

## 500Hz VAE 与当前 Latent-Hull 方法的关系

当前方法：

```text
real target ECG
-> ECGTwin VAE latent z0
-> same-label local latent hull
-> maximize classifier BCE / targeted multilabel loss
-> decode to ECG
-> train classifier on hard on-manifold variants
```

如果换成 500Hz VAE，方法本身不变，只是 `decode` 后的信号协议更干净：

```text
real target ECG at 500Hz
-> 500Hz VAE latent z0
-> same-label local latent hull
-> APGD-lite in latent space
-> decode to (12,5000)
-> ECGFounder / 500Hz classifier
```

这会让 “on-manifold” 叙事更自然，但也会增加训练成本和工程复杂度。

## 建议实验顺序

1. **先不动当前 EfficientNet 主线。**
   把 matched baseline、random latent mixup、input-space PGD 和 attack diagnostics 补齐。这个最直接保护论文主结论。

2. **做 500Hz PTB-XL VAE smoke。**
   用 fold 1-8 训练小型 12-lead VAE，先看 reconstruction、latent interpolation、sanity gate，不直接上大规模 online AT。

3. **做 ECGFounder 500Hz wrapper 对照。**
   比较：
   - ECGFounder direct K500；
   - ECGFounder + random 500Hz VAE latent mix；
   - ECGFounder + 500Hz VAE latent APGD；
   - ECGFounder + current ECGTwin VAE 1024->5000 interpolation。

4. **只有 500Hz VAE 明显改善再升级成主线。**
   否则把它放入 future work，当前主论文继续以 EfficientNet VAE-LHAT 的 matched protocol 为主。

## 可引用与可借鉴资料

| Item | Year | Type | Source/Repo | What it does | Borrow for ECG_adv_Gen | Risk/Limit | Thesis use |
|---|---:|---|---|---|---|---|---|
| PTB-XL dataset | 2020 | benchmark/data | https://www.nature.com/articles/s41597-020-0495-6 | 12-lead 10s ECG, records500 and records100, suggested folds | 500Hz VAE source data and fold protocol | public 500Hz version is resampled from original acquisition chain; still official dataset release | Core citation |
| PhysioNet/CinC Challenge 2021 | 2021 | benchmark/data | https://physionet.org/content/challenge-2021/1.0.2/ | Multi-center ECG source with mixed sampling rates | PN2021 sampling-rate audit and center-specific preprocessing | not uniformly 500Hz; labels are heterogeneous | Core citation |
| SPH / Shandong Provincial Hospital ECG DB | 2022 | benchmark/data | https://torch-ecg.readthedocs.io/en/latest/api/generated/torch_ecg.databases.SPH.html | 25,770 records, 500Hz, 10-60s | Candidate external 500Hz pretraining data | external-data use weakens pure PTB-XL -> PN2021 causality unless controlled | Future work / optional |
| ECGFounder | 2025 | foundation model | https://github.com/PKUDigitalHealth/ECGFounder | Large ECG foundation model; project-side protocol uses `12 x 5000` | Strong 500Hz backbone branch | VAE gains must be separated from K500/fullFT/head-init gains | Core baseline / extension |
| ECGFounder transfer-learning example | 2025 | implementation reference | https://cinc.org/2025/Program/accepted/332_Preprint.pdf | Confirms `12 x 5000` model input in a downstream transfer setup | Supports our ECGFounder input statement | not the original ECGFounder paper; use as implementation support, not primary method citation | Implementation reference |
| Interpretable Feature Generation in ECG Using a VAE | 2021 | VAE prior art | https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.638191/full | Cycle-level ECG VAE using 500Hz ECG source, 25-d latent features | Shows ECG VAE can work with far fewer full recordings when segmented into cycles | single-cycle / not full 12-lead 10s latent-hull AT | Background citation |
| Current project EfficientNet pipeline | 2026 | local evidence | `docs/pipelines/efficientnetv2_training_pipeline.md` | Defines current 100Hz/1000-sample EfficientNet protocol | Prevents mixing 100Hz classifier protocol with 500Hz foundation-model protocol | local protocol, not external citation | Internal source of truth |
| Current project ECGFounder pipeline | 2026 | local evidence | `docs/pipelines/ecgfounder_frozen_linear_probe_pipeline.md` | Defines ECGFounder `12 x 5000` adapter and input policy | Provides current implementation boundary | must align with final code before thesis table | Internal source of truth |
| Current project VAE-LHAT lessons | 2026 | local evidence | `AGENTS.md`, `docs/tmp_md/vae_online_at_closing_summary_20260525.md` | Summarizes VAE-LHAT gains and attribution boundaries | Keeps paper claim conservative | historical mappings v5/v6/v7 must not be mixed | Internal source of truth |

## Final recommendation

当前不要为了“500Hz 更高级”就直接推翻主线。更稳的路线是：

```text
Main paper:
  EfficientNet1DV2 100Hz
  + current ECGTwin VAE latent-hull
  + matched direct/random/input-PGD controls
  + strict attack diagnostics

Extension:
  PTB-XL-only 500Hz VAE
  + ECGFounder/500Hz classifier
  + same matched controls
```

如果时间有限，优先保证主论文归因干净。500Hz VAE 是有价值的升级方向，但它应该作为“协议更一致的后续增强”，而不是在没有 matched baseline 和诊断证据前替代当前主线。
