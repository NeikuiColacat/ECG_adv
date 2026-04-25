# AugMix over ECGTwin VAE Latent — Visualization & Validity Check

## 目的

验证一个假设：**如果把 AugMix 的多链路叠加搬到 ECGTwin 的 VAE latent 空间（而不是 ECG 时域），最终解码出来的 ECG 仍然保持生理合法性。**

动机：AugMix 的标准做法在时域混合增强版本，而 ECG 时域线性混合是有风险的
（比如两个 QRS 不对齐直接叠加会造出"双波"这种非生理产物）。VAE latent
空间理论上是连续流形，凸组合理应仍落在数据流形附近。本实验用
`util/ecg_viz.py::sanity_check` 的客观指标（Einthoven 定律、aVR 定律、HR
估计、flatline/饱和等）量化验证。

## 实验设计

- **数据**：PhysioNet 2021 真实 ECG，每个中心抽 N 条，用
  `scripts/crosscenter_v2/preprocess_utils.unified_preprocess_to_1000` 统一到
  (12, 1000) @ 100 Hz + per-sample z-score。
- **增强算子库**：复用 `methods/augmix/ecg_ops.py` 的 5 个时域算子 +
  `methods/augmix/severity.build_op` 的 severity 1–10 参数标定。
- **AugMix 配方**：width=3 链，每链 depth∈{1,2,3}，Dirichlet(α=1.0) 混合权重
  + Beta(1.0, 1.0) 原始信号融合比例。`alpha` 等参数与
  `methods/augmix/augmix.py` 保持一致。
- **Pipeline**（单样本）：
  1. 原始信号 `x` → `encode_ecg` → `z0 (1,4,128)`
  2. 对每条链：时域施加 d 个随机算子得 `x_i`，再 encode 得 `z_i`
  3. Dirichlet 权 `ws` 线性组合：`z_mix = Σ ws_i · z_i`
  4. Beta 比例 `m` 做 VAE 空间的 "clean-mix" 融合：`z* = (1-m)·z0 + m·z_mix`
  5. `decode_latent(z*)` → 最终 ECG
  6. 另外对每条链的时域信号、VAE 直接 round-trip（`decode(z0)`）都做可视化
     + sanity_check，方便归因。
- **评价**：
  - 绘图：`original / vae_roundtrip / 每条链 / augmix_output / overlay 对比`
  - Sanity：`has_nan/has_inf, flatline_leads, saturated_leads, dc_offset,
    einthoven_residual, avR_residual, hr_estimate, warnings[]`
  - 判定"合法"：warnings 为空 + Einthoven<0.2 + aVR<0.3 + HR∈[30,200]

## 目录

```
methods/augmix/latent_viz/
├── README.md           # 本文
├── config.yaml         # 实验超参（中心列表、样本数、augmix 参数）
├── latent_augmix.py    # 核心：latent 空间 AugMix 实现
├── run_viz.py          # 端到端入口
└── tests/test_smoke.py # 小规模 smoke（1 center × 1 sample）
```

大件输出在 `/root/autodl-tmp/methods/augmix/latent_viz/`：

```
<output_root>/
├── config.snapshot.yaml
├── aggregate.json                  # 跨样本的 warning / HR 统计
└── <center>/<record_id>/
    ├── report.json                 # 每个版本的完整 sanity + augmix 元数据
    └── plots/
        ├── original.png
        ├── vae_roundtrip.png
        ├── chain_{i}_{ops}.png
        ├── augmix_output.png
        └── compare_overlay.png     # 原始/roundtrip/augmix 三者叠加
```

## 运行

### smoke（约 20 秒）

```bash
PY=/root/miniforge3/envs/ECGTwin/bin/python
cd /root/ECG_adv_Gen
$PY -m pytest methods/augmix/latent_viz/tests/ -v
```

### 正式运行（N_centers × N_samples × ~5 秒/样本）

```bash
$PY methods/augmix/latent_viz/run_viz.py \
    --config methods/augmix/latent_viz/config.yaml
```

## 观察（当前配置：2 中心 × 3 样本, severity=5, width=3）

跨 6 个样本的 `aggregate.json`：

| 版本           | 平均 warning 数 | 有 warning 的样本 | HR 估计均值 (bpm) |
|----------------|----------------:|-------------------:|-------------------:|
| original       | 0.83            | 3 / 6              | 126.0              |
| vae_roundtrip  | 0.33            | 2 / 6              | 124.8              |
| augmix_output  | 0.33            | 2 / 6              | 126.0              |

**关键发现（例：chapman_shaoxing/JS01730）**

原始 → Einthoven=0.09, aVR=0.09, HR≈148 bpm, 零 warning，是合法 ECG。

三条时域链单独看都是**不合法**的：
- chain_0 (`random_leads_masking + baseline_shift`)：Einthoven=**1.59**, aVR=**0.86**, HR 失效
- chain_1 (`random_leads_masking + baseline_wander`)：Einthoven=**0.68**（导联被遮蔽）
- chain_2 (`baseline_shift + baseline_shift`)：Einthoven=**1.28**, aVR=**0.63**, V1 大 DC 偏置

但**混合 latent 再解码**后 (`augmix_output`)：
- Einthoven=**0.08**, aVR=**0.13**, HR=**148.3 bpm**, **零 warning**

即：**即便单条增强链已经破坏了 Einthoven/aVR 定律、把导联整条归零、叠了大
DC 偏置，通过 VAE 解码器后输出仍然重新落在生理流形上。** VAE decoder 起到
了一个隐式"投影回合法 ECG"的作用，AugMix 在 latent 空间做凸组合不会让输出
离开流形。

**跨样本的 `beta_m` 敏感性**（m 越大表示最终 latent 越偏向 chain mix，越
少保留原始 latent）：

| sample            | m     | augmix warnings | Einthoven | aVR   |
|-------------------|------:|----------------:|----------:|------:|
| JS01730           | 0.020 | 0               | 0.076     | 0.134 |
| JS04324           | 0.133 | 1               | 0.167     | 0.458 |
| JS01494           | 0.424 | 1               | 0.159     | 0.323 |
| **E08082**        | **0.841** | **0**       | **0.092** | **0.086** |
| **E00415**        | **0.861** | **0**       | **0.102** | **0.294** |

m ≈ 0.85（原始信号贡献几乎被替换）的两个样本都是 **零 warning + 所有定律
满足**。m 与失败率**无明显正相关**，甚至略反相关；这说明 latent 凸组合本
身不是破坏源——剩下的 warning 多半来自原始样本本身高 HR（~140 bpm）导致
的边界 sanity 残差，而不是 AugMix 引入。

## 结论（初步）

- ✅ **latent 空间 AugMix 的输出在 sanity_check 层面与 VAE round-trip 同等
  合法**，没有额外退化。说明 Dirichlet+Beta 的凸组合是在流形上做的。
- ⚠️ 时域算子单独看对 ECG 的破坏很大（`random_leads_masking` 尤其），这在
  直接时域 AugMix（非 latent）下会直接进入训练。latent 版本相当于多了一道
  VAE 投影滤波器。
- 📌 当前 sanity_check 只覆盖**物理/代数层面**（Einthoven、aVR、DC、HR），
  不涉及波形形态（QRS 宽度、ST 段抬高等诊断层面）。要更严格判定"生理合
  法"，下一步需要：
  - 形态对比（原始 vs augmix 的 QRS peak 相关、ST 段 ΔmV）
  - ECGFounder 或 EfficientNet 的诊断预测是否保持一致（语义保真）
  - Beta `m` 扫频（m→0 即 roundtrip，m→1 即纯 latent 混合；看合法性曲线）

## 非目标

- 不训练任何 downstream 模型；纯可视化+客观指标验证
- 不改 augmix / ecg_ops / ecgtwin 代码，只组合
- 不做与时域 AugMix 的对比消融（那是下一步 "latent vs time-domain AugMix"
  实验的任务）
