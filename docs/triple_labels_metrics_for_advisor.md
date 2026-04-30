# Triple-label EfficientNetV2 — 三套标签体系性能对比

**汇报日期**：2026-04-27
**Backbone**：EfficientNet1DV2 (`variant='s_v2'`)，单一 PTB-XL 训练源域
**Eval 协议**：PTB-XL fold 10 + PN2021 7 centers + MIMIC test，per-class AUROC/AUPRC
              `compute_macro_auroc_auprc(min_pos=10)` 过滤 n_pos<10 的类，bootstrap 不开（单 seed）

三个 head 共用同一 backbone 训练 pipeline，**仅 label 空间与映射函数不同**：

| 体系 | 类数 | label 来源 | 特点 |
|---|---|---|---|
| **super5** | 5 | PTB-XL `diagnostic_class` (CD / HYP / MI / NORM / STTC) | PTB-XL 全覆盖；MI 类毕设医学卖点；2026-04-26 已加 NORM exclusivity guard |
| **sub23** | 23 | PTB-XL `diagnostic_subclass` (AMI / IMI / LMI / PMI / NORM / STTC / LVH / CLBBB / CRBBB / ...) | PTB-XL 全覆盖；MI 子型细分；PN2021 上 10/23 类 SNOMED 无法可靠区分（强制 -1）；2026-04-27 已加 NORM exclusivity guard |
| **pn26** | 26 | PhysioNet 2021 官方 scored SNOMED (NSR / AF / IAVB / RBBB / LBBB / Brady / RAD / PRWP / ...) | PN2021 全覆盖；3 个类 (Brady / RAD / PRWP) PTB-XL 训练时 mask=-1 → eval 时拖累 macro |

---

## 1. PTB-XL 训练 pipeline（EfficientNet1DV2 / 三套体系共享）

代码：`scripts/triple_labels/{train_ptbxl.py, label_schemes.py, run_parallel_training.sh}`

### 1.1 数据准备

```
PTB-XL raw100.npy (21,799 × 1000 × 12 @ 100 Hz)
        │
        ▼
unified_preprocess_to_1000  (scripts/crosscenter_v2/preprocess_utils.py:115)
  ├ NaN guard (np.nan_to_num)
  ├ reorder_leads_tc       (PTB-XL native = canonical, skip)
  ├ filter_bandpass_safe   (Butterworth 0.67-40 Hz + median baseline +
  │                         50 Hz notch 当 fs > 150 Hz; PTB-XL fs=100 → 仅 BP)
  ├ resample_tc 至 target_fs=100 (PTB-XL 已是 100，no-op)
  ├ pad/truncate 至 target_len=1000 (= 10 秒)
  └ per-sample global z-score (整记录所有导联同 mean/std；非 per-lead)
        │
        ▼
缓存到 /root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy (3 scheme 共用)
```

**Fold split** (PTB-XL 官方 strat_fold)：
- **Train**：fold 1–8（~17,419 记录）
- **Val**：fold 9（~2,182 记录，用于 early-stop & best ckpt 选）
- **Test**：fold 10（~2,198 记录，最终汇报指标）

**Label 生成**（per scheme）：
- PTB-XL 元数据 `ptbxl_database.csv` 的 `scp_codes` 字段（dict-string），通过
  `scheme['ptbxl_fn']` 转 (C,) float32 multi-hot
- 缓存键带类数后缀（`ptbxl_labels.C5.all.npy` / `.C23.all.npy` / `.C26.all.npy`）→ 三个 scheme 互不干扰
- Confidence 用 presence-based（threshold=0），而非 confidence ≥ 50
  （memory `ptbxl_confidence_semantics.md`）

### 1.2 训练时数据增强

```
PTBXLDatasetScheme.__getitem__:
  signal_1000 (1000, 12) float32
        │
        ▼
crop_signal_tc(crop_len=250, mode='random')   ← 训练时随机 2.5 秒窗
                                              ← val/eval 时 mode='center'
        │
        ▼
.T → (12, 250) channels-first
        │
        ▼
torch.tensor float32 → 喂入 model
```

无其他 augmentation（AugMix / cutout / mixup 都关闭，作独立 ablation）。

### 1.3 Backbone

```python
EfficientNet1DV2(
    variant='s_v2',           # ~5M params
    input_channels=12,
    num_classes=N,            # 5 / 23 / 26
    activation='leaky_relu',  # 不是 default ReLU/SiLU
    use_se=True,              # Squeeze-and-Excitation block
    norm_type='batch',
    stochastic_depth_prob=0.304,
    dropout_rate=0.0,
)
model.apply(init_weights)     # Kaiming-He fan_out for Conv1d, Xavier for Linear
```

输入：`(B, 12, 250)` z-scored；输出：`(B, N)` raw logits（loss 内 sigmoid）。

### 1.4 Loss / Optimizer

**masked BCE-with-logits**（`-1 sentinel` 的类被 mask 出 loss + macro，pn26 Brady/PRWP/RAD 永远不参与 loss）：

```python
def masked_bce_with_logits(logits, labels_with_minus1, pos_weight):
    mask = (labels_with_minus1 >= 0).float()
    labels_safe = where(mask, labels_with_minus1, 0)
    bce = F.binary_cross_entropy_with_logits(logits, labels_safe,
                                              pos_weight=pos_weight, reduction='none')
    return (bce * mask).sum() / mask.sum().clamp_min(1.0)
```

**Per-class pos_weight = N_neg / N_pos**（仅在有效 0/1 上算），clip 到 [1, 50]：

| scheme | pos_weight 例 |
|---|---|
| super5 | CD=3.5, HYP=7.2, MI=3.0, NORM=1.3, STTC=3.2 |
| sub23 | NORM=1.3, AMI=6.1, IMI=5.7, LVH=9.2, CLBBB=39.7, ... 12 个类被 clip 到 50（罕见类） |
| pn26 | NSR=1.0, TAb=5.6, AF=13.4, RBBB=39.7, ... 多个 clip 到 50 |

**Optimizer**：AdamW
- lr=0.01, weight_decay=0.01
- CosineAnnealingLR (T_max=15, eta_min=lr × 0.01)
- AMP autocast + GradScaler
- grad clip norm=1.0
- batch_size=96
- num_workers=4，persistent_workers，pin_memory

**Early stop**：监 val_macro_auroc，patience=10，max 50 epochs；保存 `best_model.pt` 时去掉 `_orig_mod.` 前缀（兼容 torch.compile）。

### 1.5 实际训练统计

| Scheme | epochs trained | best val AUROC | wall time |
|---|---|---|---|
| super5 | **28** (early-stop @ ep28) | 0.9126 | ~7 min |
| sub23 | 50 (max) | 0.9255 | ~8 min |
| pn26 | 50 (max) | 0.9147 | ~8 min |

GPU 占用 ~7 GB / scheme（RTX 4090 24 GB），3 scheme 并行训练总 wall time ~13 min。

启动：`bash scripts/triple_labels/run_parallel_training.sh parallel`

### 1.6 Eval pipeline

`scripts/triple_labels/eval_crosscenter.py --scheme {super5|sub23|pn26}`

三套数据集**共用同一 preprocessing**（`unified_preprocess_to_1000` + center crop 250），仅 label 函数不同：

| 数据集 | label 函数 | 备注 |
|---|---|---|
| PTB-XL fold10 | `scheme['ptbxl_fn']` | 与训练 label 一致 |
| PN2021 7 centers | `scheme['pn2021_fn']` (SNOMED → C-vec) | `parse_header_snomed` 处理 `# Dx:` 空格 |
| MIMIC test | `scheme['mimic_fn']` (regex on report 文本) | machine_measurements.csv 的 18 个 report_X 字段拼接 |

**关键守门**：PN2021 7 centers 中 `ptb-xl` shard 硬 assert 排除（`ptb-xl/` 目录下文件就是 PTB-XL 训练数据本身，泄露）。

---

## 2. 总览：跨域 macro 性能对比

| Domain | super5 AUROC / AUPRC | sub23 AUROC / AUPRC | pn26 AUROC / AUPRC |
|---|---|---|---|
| **PTB-XL fold10** (in-domain, baseline) | **0.9064 / 0.7754** | **0.9158 / 0.4893** | **0.9117 / 0.5293** |
| PN2021 7-ctr avg | 0.8390 / 0.5889 | 0.8929 ✱ / 0.4572 | 0.8119 / 0.4108 |
| MIMIC test (zero-shot) | 0.7857 / 0.6939 | 0.7828 / 0.2698 | 0.7871 / 0.3520 |
| **ΔAUROC vs PTB-XL** | | | |
| → PN2021 avg | **−0.067** | **−0.023** ✱ | **−0.100** |
| → MIMIC | **−0.121** | **−0.133** | **−0.125** |
| **ΔAUPRC vs PTB-XL** | | | |
| → PN2021 avg | **−0.187** | **−0.032** | **−0.119** |
| → MIMIC | **−0.082** | **−0.220** | **−0.177** |

✱ sub23 的 PN2021 数字是 2026-04-27 加 NORM-guard 后的（修前 0.8892 → 修后 0.8929，+0.36pp）。

> 横向解读：sub23 的 PN2021 ΔAUROC 最小（−0.023）— 看似最稳，但因为 10/23 类强制 -1 自动 mask，**实际只用 6–11 个类算 macro**，与 PTB-XL fold10 的 19 类不可比。super5/pn26 是更公平的跨域指标。

---

## 3. 详表 — super5 (5 classes)

**PTB-XL fold10 baseline**：macro AUROC=0.9064 / AUPRC=0.7754

> **n_eff** = 实际参与 macro AUROC/AUPRC 计算的样本数（即每个 used 类的 `n_valid`，等价于 dataset 中成功 load + 未被 ref 排除的记录数；详见 §6.6）

| Center | n_eff | AUROC | ΔAUROC | AUPRC | ΔAUPRC | n_classes used | 备注 |
|---|---|---|---|---|---|---|---|
| **PTB-XL fold10** ★ | 2198 | **0.9064** | — | **0.7754** | — | 5/5 | in-domain baseline |
| chapman_shaoxing | 10247 | 0.8835 | **−0.023** | 0.5468 | −0.229 | 5/5 | 表现最好 cross-center |
| ningbo | 34905 | 0.8826 | **−0.024** | 0.5914 | −0.184 | 5/5 | n 最大，CI 最稳 |
| ptb | 516 | 0.9009 | **−0.006** | 0.5950 | −0.180 | 4/5 | n 小但接近 PTB-XL |
| georgia | 10344 | 0.8173 | −0.089 | 0.6462 | −0.129 | 4/5 | 无 MI (n_pos=7<10) |
| cpsc_2018_extra | 3453 | 0.8052 | −0.101 | 0.6220 | −0.153 | 4/5 | 无 NORM (n_pos=7<10) |
| cpsc_2018 | 6877 | 0.8052 | −0.101 | 0.5633 | −0.212 | 3/5 | 无 HYP/MI；STTC 结构性 cap 0.59 |
| st_petersburg_incart | 74 | 0.7785 | −0.128 | 0.5575 | −0.218 | 4/5 | n 太小 (CI 太宽) |
| MIMIC test (zero-shot) | 78707 | 0.7857 | **−0.121** | 0.6939 | −0.082 | 5/5 | NORM AUROC=0.612 (regex drift) |

**Per-class PTB-XL → cross-center 关键观察**：
- **MI 类**：PTB-XL 0.920 → PN2021 5-center avg 0.923 → 已在 ceiling，几乎无 cross-center gap。**毕设 MI 卖点**。
- **NORM 类**：PTB-XL 0.939 → chap 0.906 / ningbo 0.884（健康，post-guard）；MIMIC 0.612（drift，未加 regex guard）
- **STTC**：cpsc_2018 结构性 cap 0.59（vocab gap，AT 工作时排除该 cell）

---

## 4. 详表 — sub23 (23 classes, 含 NORM-guard 修后)

**PTB-XL fold10 baseline**：macro AUROC=0.9158 / AUPRC=0.4893

> **n_eff** = 实际参与 macro AUROC/AUPRC 计算的样本数；对 sub23 而言**已 mask 掉 10 类强制 -1 的影响**（那 10 类不进 macro，剩下 used 类的 n_valid 都 = dataset size）。`st_petersburg_incart` n_eff=0（74 records 全部因 0 类满足 min_pos=10 而不参与 macro）。

| Center | n_eff | AUROC | ΔAUROC | AUPRC | ΔAUPRC | n_classes used | 备注 |
|---|---|---|---|---|---|---|---|
| **PTB-XL fold10** ★ | 2198 | **0.9158** | — | **0.4893** | — | 19/23 | in-domain baseline |
| ptb | 516 | 0.9392 | **+0.023** | 0.6688 | **+0.180** | 1/23 | 仅 NORM (其它 n_pos<10) |
| cpsc_2018 | 6877 | 0.9283 | **+0.013** | 0.7353 | **+0.246** | 4/23 | conduction blocks 强 |
| **ningbo** ✱ | 34905 | **0.8924** | **−0.023** | 0.3181 | −0.171 | 11/23 | NORM AUROC pre→post: 0.707→0.790 |
| georgia | 10344 | 0.8838 | −0.032 | 0.3555 | −0.134 | 10/23 | — |
| **chapman_shaoxing** ✱ | 10247 | **0.8607** | −0.055 | 0.4482 | −0.041 | 6/23 | NORM AUROC pre→post: 0.725→0.811 |
| cpsc_2018_extra | 3453 | 0.8530 | −0.063 | 0.2175 | **−0.272** | 8/23 | MI subtypes 全 mask；AUPRC 受细类拖累 |
| st_petersburg_incart | **0** ⚠️ | NaN | — | NaN | — | 0/23 | 74 records 全部因 0 类满足 min_pos=10 而不参与 macro |
| MIMIC test (zero-shot) | 78707 | 0.7828 | **−0.133** | 0.2698 | **−0.220** | 20/23 | NORM AUROC=0.477 (regex drift) |

✱ chap/ningbo 是 2026-04-27 NORM-guard 修复的两个直接受益中心。
> 注：sub23 PN2021 的 ΔAUROC 看似很小或为正（+0.013 / +0.023）是因为 MI subtypes（AMI/IMI/LMI/PMI）+ 4 个 ischemia subtypes + STTC + NST_ + SEHYP 共 10 个类在 PN2021 强制 -1，自动 mask；实际只用 1–11 个易类（NORM + conduction blocks + LVH/RVH）算 macro，**与 PTB-XL fold10 的 19 类不可同框比较**。建议看 super5 / pn26 的 ΔAUROC 作为跨域 gap 真实参考。

---

## 5. 详表 — pn26 (26 classes)

**PTB-XL fold10 baseline**：macro AUROC=0.9117 / AUPRC=0.5293

> **n_eff** = 实际参与 macro AUROC/AUPRC 计算的样本数（pn26 的 PN2021/MIMIC 26 类全 0/1 → n_eff = dataset size）。注意 used 类里包含 PTB-XL 未训练的 Brady/RAD/PRWP，详见 §6.1。

| Center | n_eff | AUROC | ΔAUROC | AUPRC | ΔAUPRC | n_classes used | 已知 drag 类 |
|---|---|---|---|---|---|---|---|
| **PTB-XL fold10** ★ | 2198 | **0.9117** | — | **0.5293** | — | 21/26 | in-domain baseline |
| cpsc_2018 | 6877 | 0.9015 | **−0.010** | 0.6810 | **+0.152** | 6/26 | — |
| ptb | 516 | 0.8883 | −0.023 | 0.4234 | −0.106 | 3/26 | — |
| chapman_shaoxing | 10247 | 0.8402 | −0.072 | 0.3997 | −0.130 | 18/26 | RAD AUROC=0.341 (n=215) |
| ningbo | 34905 | 0.8368 | −0.075 | 0.3597 | −0.170 | 24/26 | RAD=0.351 (n=638), PRWP=0.667 (n=10) |
| georgia | 10344 | 0.8321 | −0.080 | 0.3712 | −0.158 | 22/26 | RAD=0.371 (n=83) |
| cpsc_2018_extra | 3453 | 0.7939 | **−0.118** | 0.2880 | **−0.241** | 14/26 | **Brady AUROC=0.190 (n=271)** ⚠️ |
| st_petersburg_incart | 74 | 0.5905 | −0.321 | 0.3522 | −0.177 | 2/26 | n 太小 |
| MIMIC test (zero-shot) | 78707 | 0.7871 | **−0.125** | 0.3520 | −0.177 | 25/26 | Brady/PRWP 同样 drift |

**若把 PTB-XL 未训练类 (Brady, RAD, PRWP) 排除出 PN2021 macro**（可一行代码改）：
- chap_shaoxing 0.840 → ~0.85（+1pp）
- cpsc_2018_extra 0.794 → ~0.83（**+3.5pp**，主要砍掉 Brady=0.190）
- ningbo 0.837 → ~0.85（+1.5pp）
- 7-ctr avg 0.812 → ~0.84（+2.5pp，gap 从 −0.100 缩到 ~−0.07，向 super5 靠近）

---

## 6. 关键 caveat（汇报时主动披露）

### 6.1 pn26 PTB-XL 未训练类持续拉低 PN2021/MIMIC macro
`label_alignment_v2.py:131-137` 注释明文 `"Expected uncovered: {Brady, PRWP, RAD}"`。这 3 类在 PTB-XL 训练 label=-1（masked from loss），但 PN2021/MIMIC eval 时给出真实 0/1 标签，**仍被算入 macro**。drag 量化在第 5 节末。建议 future work：在 `compute_macro_auroc_auprc` 加 `trained_indices` 参数。

### 6.2 super5 cpsc_2018 STTC 结构性 cap 0.59
cpsc_2018 中心 SNOMED 词表只有 ST elevation / ST depression，缺 T-wave inversion / Q-wave / prolonged QT / ischemia 编码（PTB-XL STTC 类下都覆盖）→ AT 工作里把该 cell 的 STTC 排除。已 documented (`memory/super5_label_audit_2026_04_26.md`)。

### 6.3 sub23 PN2021 上 10/23 类强制 -1
AMI/IMI/LMI/PMI/STTC/NST_/ISC_/ISCA/ISCI/SEHYP 在 PN2021 SNOMED 词表里无法可靠区分（generic MI 164865005 不切前/下/侧/后壁；ischemia 子型也无特异 SNOMED）。这 10 类 PN2021 eval 自动 mask（不参与 macro）。**导致 sub23 PN2021 ΔAUROC −0.023 看似最小，但实际只用 6–11 个易类算，不可与 PTB-XL fold10 19 类同框对比**。

### 6.4 NORM-guard 覆盖局限（MIMIC regex 路径未加）

| 路径 | NORM-guard | NORM AUROC 状态 |
|---|---|---|
| super5 PN2021 (SNOMED) | ✅ 2026-04-26 | chap 0.906 / ningbo 0.884 健康 |
| super5 MIMIC (regex) | ❌ 未加 | MIMIC NORM=0.612 ⚠️ |
| sub23 PN2021 (SNOMED) | ✅ 2026-04-27 | chap 0.811 / ningbo 0.790 健康 |
| sub23 MIMIC (regex) | ❌ 未加 | MIMIC NORM=0.477 ⚠️ |
| pn26 PN2021 (SNOMED) | n/a | NSR 语义不同（PTB-XL 用 'NORM' + 'SR' 双 SCP 训），无 drift |
| pn26 MIMIC (regex) | n/a | NSR 同样无 drift |

> sub23/super5 在 MIMIC regex 上的 NORM-guard 是已知 follow-up；其它 4-5 个类不受影响，所以 MIMIC 整体 macro AUROC 仍 0.78 量级（NORM 类拖累约 −0.02pp on macro）。

### 6.5 AUPRC 跨中心对比偏不稳
sub23/pn26 的细类 baseline rate < 1% → AUPRC 对类不平衡极敏感，单 cell 抖动可达 ±0.05。**AUROC 是更稳的跨域 gap 指标**；AUPRC 仅作辅助。

### 6.6 表中 `n_eff` 字段含义（实测验证）

**`n_eff` = 实际参与 macro AUROC/AUPRC 计算的样本数**。机制：

1. **per-class -1 mask** (`compute_macro_auroc_auprc:108`)：`valid = t >= 0` → 把 -1 entries 从该类 AUROC 计算中排除（**不会被错算成阴性**）
2. **min_pos=10 过滤** (`compute_macro_auroc_auprc:113`)：n_pos < 10 的类直接不进 macro（避免小样本噪声 dominate）
3. **n_classes_used 字段**：经上面两层过滤后实际进入 macro 的类数

**实测三 scheme 所有 cell 的 used 类 n_valid 分布**（取每 cell 的 used 类 n_valid 集合）：

| Scheme / Cell | min(n_valid) over used classes | max(n_valid) | n_eff = n_records? |
|---|---|---|---|
| **super5** 全部 cell (PTB-XL/PN2021/MIMIC) | = dataset size | = dataset size | ✅ Yes |
| **pn26** 全部 cell | = dataset size | = dataset size | ✅ Yes |
| **sub23** 大部分 cell (PTB-XL/PN2021 6 个/MIMIC) | = dataset size | = dataset size | ✅ Yes |
| **sub23 st_petersburg_incart** | — (0 used 类) | — | ⚠️ n_eff=0, dataset size=74 |

**结论**：表中数字 = 真实 effective participation count。**唯一 edge case 是 sub23 st_petersburg_incart**（n=74 全部因 0 类满足 min_pos=10 而不进 macro，本质 NaN cell），第 4 节表里已显式标 n_eff=0。

**为什么 sub23 strict-mask 的 10 类不影响 n_eff**：那 10 类（AMI/IMI/LMI/PMI/STTC/NST_/ISC_/ISCA/ISCI/SEHYP）的 n_valid=0，自动被两层过滤剔除（步骤 1 → n_valid=0；步骤 2 → n_pos=0<10）。没有任何 n_valid<dataset_size 的类进入 macro，所以 used 类 n_valid 一致 = dataset size。

**仍需注意**：sub23 PN2021 实际只用 6–11 个易类算 macro，**与 PTB-XL fold10 用 19 类不同维度对比**。建议看 super5/pn26 的 ΔAUROC 作为跨域 gap 真实参考。

---

## 7. 选型小结

| 用途 | 推荐 | 主要理由 |
|---|---|---|
| MI 毕设医学卖点 | **super5** | MI 类 PTB-XL 0.920 / PN2021 5-ctr 0.923 已在 ceiling；MI 跨中心几乎无 gap |
| 跨中心鲁棒性研究 (AT / synth-aug) | **super5** | 全覆盖 + NORM-guard 完成 + MI 稳定；类粒度合适 |
| MI 子型细分研究 (PTB-XL only) | **sub23** | AMI/IMI/LMI/PMI 分得清；PN2021 上 4 子型自动 mask |
| 与 Tier-M / PN2021 challenge 直接对位 | **pn26** | label 空间与 PhysioNet 2021 challenge 完全一致 |

**避开**：
- super5 cpsc_2018 STTC（vocab gap，cap 0.59）
- super5 georgia MI（n_pos=7 < 10 自动过滤）
- pn26 PN2021 上的 Brady / RAD / PRWP（结构性拖累，建议 mask）
- 任何 scheme 的 st_petersburg_incart（n=74，CI 太宽）

---

## 附录 A：复现命令

```bash
# 训练（3 scheme 并行，single 4090，~13 min）
bash scripts/triple_labels/run_parallel_training.sh parallel

# 单 scheme 训练
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/train_ptbxl.py \
    --scheme {super5|sub23|pn26} --batch_size 96 --epochs 50 \
    --output_dir /root/autodl-tmp/triple_labels/{scheme}/

# Eval (PTB-XL fold10 + PN2021 + MIMIC)
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
    --scheme {super5|sub23|pn26} \
    --model_dir /root/autodl-tmp/triple_labels/{scheme}/ \
    --output_path /root/autodl-tmp/triple_labels/{scheme}/eval_result.json
```

## 附录 B：审计文档

- 全量审计报告：`docs/triple_labels_audit_2026_04_27.md`
- super5 NORM 审计（2026-04-26）：`memory/super5_label_audit_2026_04_26.md`
- sub23 NORM-guard 修复（2026-04-27）：`memory/triple_labels_audit_2026_04_27.md`
- 三套体系训练记录：`memory/triple_labels_training.md`

## 附录 C：数据来源

- super5 PTB-XL/PN2021：`/root/autodl-tmp/triple_labels/super5/eval_result_NORMguard.json`
- super5 MIMIC：`/root/autodl-tmp/triple_labels/super5/eval_result.json` (pre-guard, MIMIC 路径不受 PN2021 guard 影响)
- sub23 PTB-XL/PN2021：`/root/autodl-tmp/triple_labels/sub23/eval_result_NORMguard.json`
- sub23 MIMIC：`/root/autodl-tmp/triple_labels/sub23/eval_result_pre_normguard.json` (同上理由)
- pn26：`/root/autodl-tmp/triple_labels/pn26/eval_result.json`
- 训练统计：`/root/autodl-tmp/triple_labels/{scheme}/train_result.json` + `training_log.json`
