# Synth-Anchored Online AT Plan — Rev 13 (Streamlined, 2026-04-27)

> **本文档**：取代 Rev 1-12 全部内容。聚焦 Stage 1-3（synth pool → online AT → eval）。
> Rev 1-12 历史决策已固化为下述 Pre-conditions / Design 部分，原始流水账见 git log。
> **关联 memory**: `synth_anchored_super5_pilot1_findings.md`, `centertoken_actual_mechanism.md`,
> `ecgtwin_super5_class_support.md`, `ecgtwin_prompt_vs_basevector.md`, `online_at_academic_backing.md`
> **关联 skill**: `/root/.claude/skills/synth-anchored-online-at/SKILL.md`

---

## TL;DR

- **Pipeline = 3 components**: PGD adversarial training + ECGTwin synth (3-class only) + H4 per-class trust gate
- **Drop CenterToken** (Plan Rev 12 — 双 subagent 实证 NOOP)
- **Generate only NORM/MI/STTC** (Plan Rev 11 — 数字 GT 校验 HYP/CD 0/3 通过)
- **Scale online AT data per Wang 2023 ICML**: K_pgd=10, pool 100/class × 3 = 300/cell (DataDream-aligned 1.5×, scalable to 6× = 1200/cell), K_anchor stratified no-revisit walker
- **Train budget**: 100 epoch + early-stop `patience=20` on val macro AUROC (Rev 13.1) — 4090 算力允许 + 与 Stutz 2019 / Rebuffi 2021 100-ep canonical 对齐
- **3 中心 pilot (Rev 13.2)**: cpsc_2018_extra + ningbo + georgia × K=200 ref. georgia 选作第 3 cell 因 baseline AUROC 0.8173 (MAIN5 第二低, lift 空间大) + AUPRC 0.6462 (MAIN5 最高, 决策门 +0.50pp 更易触发). MI n_pos=7 < min_pos=10 → georgia source-eval 算 4 类 macro (CD/HYP/NORM/STTC), MI synth 只作 cross-center 贡献
- **病人级 ref-eval 隔离 (Rev 13.2)**: prep 端 `{tag}_k200.meta.json::ref_record_ids` 已实施 ✓；eval 端必须新增 `--exclude_ref_ids` flag（当前 missing）→ Phase 1.5 强制实施
- **Pilot iter 4 决策门**: 3 cells × ≤100 epoch (early-stop), ≥2 cell pass macro AUROC Δ > +0.30pp **AND** macro AUPRC Δ > +0.50pp → 推 5-center confirmatory；否则锁定瓶颈在 ECGTwin 词表 / synth 流形质量

---

## Pre-conditions (DONE — 不重做)

✅ **Super5 victim** `/root/autodl-tmp/triple_labels/super5/best_model.pt`
   PN2021 5-ctr macro AUROC baseline = **0.8388**, MI cross-center = 0.923

✅ **2/3 ref pool 已生成 (super5 scheme, K=200, multiprocessing prep)**:
   - `/root/autodl-tmp/center_token_super5/extra_k200.pt` + `extra_k200.meta.json` (含 ref_record_ids 200 项)
   - `/root/autodl-tmp/center_token_super5/nin_k200.pt`   + `nin_k200.meta.json`
   - **缺**: `geo_k200.pt` — Phase 1 一并生成 (~10 min @ multiprocessing×12)

⚠️ **Ref-eval 隔离半成品**:
   - prep 端记录 ref_record_ids ✓
   - eval 端 `eval_crosscenter.py` 没有 `--exclude_ref_ids` flag (argparse 缺失) — Phase 1.5 必须实施

✅ **ECGTwin generation digital-GT 校验** (CLAUDE.md "ECGTwin super5 generation scope")
   - NORM / MI(stemi) / STTC(NSTEMI): **3/3 best-cell** 数字阈值通过
   - HYP / CD: **0/3** (HYP Sokolow 0.97-2.15 mV < 3.5 mV; CD lateral R-amp 不足 / PR ≤ 88ms)
   - 根因：`normal_1.pt` ref Sokolow=1.51 mV，prompt 控形态不控电压 → 结构性，非可修
   - 决策：production scope 写死为 **NORM / MI / STTC 3 类**
   - 数字 extractor `util/ecg_digital_features.py`，报告 `docs/ecgtwin_super5_digital_gt_validation.md`

✅ **CenterToken 已废弃** (Plan Rev 12, `memory/centertoken_actual_mechanism.md`)
   - 5 trainer bug (detach 杀 L_inv / 1 batch/epoch / best-vs-latest 14× / L_inv var 不奖励改善 / L_reg unit-vec)
   - 4-condition 多层验证: trained CT 方向无法与 random Gaussian noise 区分
   - `scripts/ecgtwin_gen/generate_center_synth.py --token_ckpt` 已改为 `default=None`
   - production pipeline 用 vanilla ECGTwin + IBE base_vector path

✅ **Pilot 3-iter 结果** (`docs/synth_anchored_super5_pilot_summary.md`)
   - 6/6 cell-runs avg-gate STOP，但 **chap_shaoxing 跨中心 +0.40pp 一致** (per-center pass)
   - STTC 单类峰值 +1.78pp (chap_shaoxing iter1) — 信号真实
   - Source-center self-eval 平 (Stutz 2019: on-manifold AT 不改善源域)
   - Best config: **adv_weight=0.5, ε=2.0** (iter3)

---

## Strategic shift (Rev 13 vs Rev 8 pilot)

| 维度 | Rev 8 pilot (iter3 best) | **Rev 13 iter 4** | 出发点 |
|---|---|---|---|
| **中心数** | 2 (extra, nin) | **3 (extra, nin, georgia)** | 用户决策 — georgia baseline AUROC 0.8173 (lift 空间大) + AUPRC 0.6462 (MAIN5 最高) + 4 类全足 n_pos≥10 |
| **Ref-eval 隔离** | 无 (eval 撞 ref) | **--exclude_ref_ids** 实施 | 用户提问 + Plan Rev 7 Issue #39 |
| Generation scope | 5 类 (CD/HYP/MI/NORM/STTC) | **3 类 (NORM/MI/STTC)** | Plan Rev 11 数字 GT 校验 |
| CenterToken hook | extra/nin K=200 ckpt | **drop (no hook)** | Plan Rev 12 NOOP 实证 |
| H4 per-class trust gate | 不实施 | **实施**, HYP/CD trust=0 | pilot finding + ECGTwin scope |
| Synth pool | 100 latent / cell (5 cls × 20) | **300 latent / cell** (3 cls × 100) | DataDream 6.25× → 我们 1.5× baseline |
| K_pgd | 5 | **10** | Wang 2023 / Madry 2018 (≥7) |
| K_anchor / epoch | 100 random with-replacement | **300 stratified no-revisit walker** | Wang 2023 + 减少 pool starvation |
| ε | 2.0 | 2.0 (保留) | iter3 信号 + Einthoven p95=0.09 通过 |
| adv_weight | 0.5 | 0.5 (保留) | iter3 best；real-domain bias 防 cross-center drift |
| Epochs | 50 | **100 + early-stop patience=20** on val macro AUROC | Wang 2023 等效 ~295 ep；100 ep 与 Stutz/Rebuffi canonical 对齐；patience 自动 fit cell 收敛速度 |

---

## Reference: Online AT data scale 数据规模设计

四个直接 antecedent paper 的实测 config（已 clone repo 全文）：

| Param | **Wang 2023 ICML**<br>DM-Improves-AT 144⭐ | Wilde MIDL 2024<br>Medical TI K-shot | DataDream ECCV 2024<br>K-shot synth | Stutz 2019 CVPR<br>On-Manifold AT 414 cites | **Rev 13 ours** |
|---|---|---|---|---|---|
| Domain | CIFAR-10 image (32×32 RGB) | Medical 2D image (TI personalize) | Few-shot 2D classification | SVHN/CelebA VAE-GAN θ ∈ ℝ¹⁰ | ECGTwin VAE latent (4, 128) |
| Reference set | 50K train + 1M EDM | K=200 personalize set | K=16 per class | 1K SVHN | **K=200 PN2021 ref** |
| Synth pool | **1M** | 400 (= 2× K) | **100/class** (= 6.25× per shot) | 1K | **300/cell** (3 cls × 100; option to scale 1200/cell) |
| Synth:K ratio | 20× | 2× | 6.25× | 1× | **1.5× baseline → 6× option** |
| K_pgd (`attack-iter` default) | **10** | n/a (no AT) | n/a (no AT) | varies (paper 7) | **10** ⭐ Rev 13 改 |
| ε | 8/255 (L_∞) | n/a | n/a | 0.1 + ‖θ‖∞ ≤ 2 prior | **2.0 in unscaled VAE latent ≈ 30% anchor norm** |
| Mix ratio | unsup-fraction = **0.7** (synth:real = 7:3) | n/a | n/a | n/a | **adv:real ≈ 1:1** (real-dominated, novel finding) |
| Epochs | **400** (CIFAR 50K records) | n/a | n/a | 50 | **100 + early-stop patience=20** (PTB-XL 17K records, ≈ Wang 295 ep equivalent) |
| Batch size | 512 | n/a | n/a | minibatch | 128 (PTBXL real) + 64 (adv pool sample) |
| **Total adv exposure** | 67M (1M pool × 67% touch over 400 ep) | 0 | 0 | minibatch × 50 ep | **30K (300 pool × 100 ep × 1× revisit/ep)** — 仍 2200× 少于 Wang，但比 Rev 13.0 翻倍 |

实测 config 来源 (`grep -n attack-iter /root/autodl-tmp/external_repos/DM-Improves-AT/core/utils/parser.py`):
```
34: parser.add_argument('-na', '--num-adv-epochs', type=int, default=400, ...)
47: parser.add_argument('--attack-step', type=str2float, default=2/255, ...)
48: parser.add_argument('--attack-iter', type=int, default=10, ...)
55: parser.add_argument('--unsup-fraction', type=float, default=0.7, ...)
```

### 数据规模 trade-off (Rev 13 关键决策)

**为什么不全盘 copy Wang 2023 1M pool / 400 epoch?**

1. **Pool 1M 不可行**: ECGTwin DDPM 50-step batch=100 生成单 latent ~0.6s → 1M latent = 167h；4× DataDream ratio (1200/cell × 2 cell = 2400 total) 才 ~24min Stage 1，是 sweet spot
2. **100 epoch + early-stop 是 sweet spot**: Wang 50K/512 × 400 ep = 39.2K gradient steps；我们 17K/128 × 100 ep = 13.3K steps（仍 Wang 的 34%，但比 Rev 13.0 50ep 翻倍）。等效到 Wang 步数需要 295 ep，过激进且 4090 时间紧；100 ep + patience=20 既匹配 Stutz/Rebuffi canonical，又能让快收敛 cell 自动停在 30-50 ep 不浪费、慢收敛 cell 跑满 100 ep 榨信号。pilot iter1 cell 1 的"晚期 AUROC decay" 在 patience 20 下会被自动捕获
3. **unsup-fraction 0.7 反向**: pilot iter1 (adv_weight=2.0 ≈ adv:real 1:0.5 即 synth-fraction 67%) 出现晚期 AUROC decay；iter3 best (adv_weight=0.5 即 synth-fraction 33%) 反而最稳。**Cross-center setting 与 CIFAR 不同：real distribution 漂移到 OOD center 比 adv robustness 更要紧** → 我们采用 **real-dominated mix**，与 Wang 2023 反向，作为 paper 的 novel finding 之一
4. **K_pgd 5 → 10 必改**: Madry 2018 §5 警告 K_pgd<7 风险 gradient obfuscation；pilot ASR=0.95-1.00 全程是攻击太弱的征兆（victim 还没收紧），K=10 保留 safety margin
5. **K_anchor no-revisit walker 必改**: pilot K=100 random with-replacement on pool 100 → 完全自循环，victim 第 5 epoch 已记住所有 anchor；no-revisit walker 强制 epoch 内每 anchor 唯一，跨 epoch 才重看（pool=300 / K=300 / 1 epoch cycle = 50 cycle of replay；pool=1200 / K=300 / 4 epoch cycle = 12.5 replay — 后者接近 Wang 2023 effective per-anchor exposure）

### Pool size 决策树 (Rev 13)

```
Default: 100/class × 3 = 300/cell  (~6 min Stage 1, 1.5× DataDream)
    │
    ├── iter4 pass +0.30/+0.50 双 gate
    │       → 推 5-center confirmatory，pool 不变
    │
    └── iter4 fail
            → Scale to 400/class × 3 = 1200/cell (~24 min Stage 1, 6× DataDream baseline)
            → 仍 fail 锁定瓶颈在 ECGTwin 词表 / synth 流形质量 (而非 adv 算力)
```

---

## Stage 1 — Synth pool generation (vanilla ECGTwin, 3 classes)

**输入**：`scripts/ecgtwin_gen/prep_center_dataset_super5.py` 生成的 K=200 ref pool
- `/root/autodl-tmp/center_token_super5/extra_k200.pt` (cpsc_2018_extra) ✓
- `/root/autodl-tmp/center_token_super5/nin_k200.pt` (ningbo) ✓
- `/root/autodl-tmp/center_token_super5/geo_k200.pt` (georgia) — **Phase 1 待生成**

**操作**：
```bash
# Phase 1 先生成 geo_k200 ref pool (~10 min, multiprocessing×12)
/root/miniforge3/envs/ECGTwin/bin/python scripts/ecgtwin_gen/prep_center_dataset_super5.py \
  --center georgia --K 200 --seed 42 \
  --text_embeds /root/autodl-tmp/center_token_super5/super5_text_embeds.pt \
  --out /root/autodl-tmp/center_token_super5/geo_k200.pt
# 验 meta.json 含 ref_record_ids 200 项
# ⚠️ georgia MI n_pos=7 < floor=10 → MI 类只能取全部 7 个 ref；synth MI 来自小样本基底，质量风险

# 3 cells × 100/class × 3 classes = 300 latent / cell, 一次性生成
for tag in extra nin geo; do
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/generate_center_synth.py \
      --scheme super5 \
      --ref_pt /root/autodl-tmp/center_token_super5/${tag}_k200.pt \
      --center_name ${tag} \
      --out /root/autodl-tmp/synth_anchored_super5_v13/synth_latents/${tag}_k200_pool300.npz \
      --n_per_class 100 --batch_size 100 --num_steps 50 --save_latent
  # --token_ckpt 不传 → 默认 None (Plan Rev 12 vanilla path)
done
```

**关键改动 (Rev 13)**:
1. `prep_center_dataset_super5.py::pick_primary_super5` priority list 缩到 `["MI", "STTC", "NORM"]` (drop HYP/CD)
2. `super5_text_embeds.py::SNOMED_TO_PROMPT` 缩到 NORM/MI/STTC 子码 (drop HYP 1+ codes / CD 17 codes)
3. `generate_center_synth.py` 已默认 `--token_ckpt None` (✅ DONE)
4. `_index_by_class` 仍按 `super5_multi_hot` 检索 (HYP/CD ref 仍可作 NORM/MI/STTC 的 ref pool 候选，因为 multi-hot)

**Sanity 0.4 (强制门)**:
对 3 cells × 100/类 = 900 synth ECG 跑 super5 victim 推理：
- 输出 per-class AUROC 写入 `{tag}_k200_pool300_sanity.json`
- **gate**: NORM/MI/STTC 各类 AUROC > 0.55 → 进 Stage 2；否则停下查 prompt
- **trust 派生**: `class_trust = {c: 1.0 if auroc[c]>0.7 else (0.5 if auroc[c]>0.55 else 0.0) for c in {NORM,MI,STTC}}`，HYP/CD 直接写 0.0
- 写入 `{tag}_k200_pool300_class_trust.json`，Stage 2 加载

**预算**: 6 min × 3 cell = 18 min (含 geo prep 10 min 共 ~28 min)

---

## Stage 2 — Online AT pilot iter 4 (Wang 2023-aligned + H4 trust gate)

**Fork base**: `scripts/crosscenter_tierM/online_adv_train_tierM.py` → 新写
`scripts/pgd_cross_center/synth_online_at_super5.py`

### Single epoch 流程 (Rev 13)

```
A) Stratified no-revisit walker 从 frozen synth pool (300) 取 K_anchor=300 stratified by NORM/MI/STTC
   - 1 epoch cycle = 1 pass over pool (≤100 ep × 1 cycle/ep = ≤100 cycle of replay)

B) PGD on synth latent (新 K_pgd=10 vs pilot 5):
   - random init perturbation δ ~ N(0, 0.01²) (Madry standard)
   - 10 step × batch=32 = ~100s/epoch (vs pilot 50s)
   - clean anchor restart per epoch (Wang 2023 标准；不 continual)

C) Decode + 双 gate (复用 adversarial/adv_validation.py):
   - ASR gate: compute_asr(adv_logits, target_labels, num_classes=5)
   - 医学 gate: compute_semantic_gate(adv_signals, anc_signals, einthoven_p95_max=0.5)
   - **新增 H4 trust 加权**: buffer.add_one(...) weight × class_trust[primary_class]
     - HYP/CD trust=0 → effective drop (虽然 Stage 1 已不生成，buffer 入口再加一层保险)
     - 类 AUROC<0.55 → trust=0.5

D) Buffer push:
   - QualityAwareBuffer (FIFO + informativeness, cap=2048)
   - buffer.add_one(adv_signal_ct_250, lbl, score=class_trust × (1 - 2|p_t - 0.5|))

E) Mixed loader:
   - PTBXL real (w=1.0) + roundtrip anchor (w=0.5) + adv buffer (w=0.5)
   - WeightedRandomSampler total = 2.0; effective synth-fraction = 0.5/2.0 = 25% (real-dominated)
   - **Rev 13 mix ratio 与 Wang 2023 0.7 反向**, paper Discussion novel finding

F) Train 1 epoch:
   - masked BCE on -1 sentinel (synth one-hot vs PTBXL multi-hot 异构)
   - EWA anchor logit 正则 (decay=0.999, anchor_lambda=0.05)
   - lr=5e-5, batch=128 (continue init from super5 baseline ckpt)

G) Val + (每 3 epoch) quick_eval (PN2021 4-ctr stratified subset):
   - 写 `training_log.json`: {epoch, asr_overall, asr_per_class, einthoven_p95, mix_share, val_loss, val_macro_auroc}
   - Best ckpt save by **val_macro_auroc** (不是 val_loss — AT setting 下 BCE loss 不单调反映 generalization)
   - Early stopping: patience=20 epochs without val_macro_auroc improvement → halt
   - Hard halt: ASR<0.30 连续 3 epoch → RuntimeError (PGD broken)
   - Max epochs cap: 100 (即使 patience 没触发也停)
```

### Hard-coded sentinels

```python
# Top of synth_online_at_super5.py
SUPER5_SCOPE_GEN = {"NORM", "MI", "STTC"}     # Plan Rev 11 — 不生成 HYP/CD
DEFAULT_TRUST_HARDCODE = {"HYP": 0.0, "CD": 0.0}   # 即使 Stage 0.4 sanity 写了别的也覆盖

assert all(c in {"NORM","MI","STTC","HYP","CD"} for c in class_trust)
class_trust.update(DEFAULT_TRUST_HARDCODE)
```

### Smoke test (必跑，先于完整 ≤100 ep)

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/pgd_cross_center/synth_online_at_super5.py \
  --tag extra --K 200 \
  --synth_pool /root/autodl-tmp/synth_anchored_super5_v13/synth_latents/extra_k200_pool300.npz \
  --class_trust /root/autodl-tmp/synth_anchored_super5_v13/synth_latents/extra_k200_pool300_class_trust.json \
  --init_ckpt /root/autodl-tmp/triple_labels/super5/best_model.pt \
  --K_anchor 300 --K_pgd 10 --epsilon 2.0 --adv_weight 0.5 \
  --n_epochs 3 --patience 20 \
  --out_dir /tmp/smoke_super5_v13_extra
```

注：smoke 用 `--n_epochs 3` 跑通 forward/PGD/buffer/loss; patience 在 3 epoch 内不会触发，正式 pilot 设 `--n_epochs 100 --patience 20`。

期望:
- 显存峰值 < 22GB (`nvidia-smi -l 1` 监控；如爆 fallback K_anchor=200 + batch=24)
- ASR ≥ 0.5 但不应 1.0 全程 (K_pgd=10 应给攻击多一点点头会让 victim 的 BCE 在前期下降而非平)
- Einthoven p95 < 0.5 (medical gate 一致通过)
- val_loss 单调下降或平稳，不爆

### 完整 pilot iter 4

3 cells (extra, nin, geo) × ≤100 epoch (early-stop patience=20) 串行:

```bash
bash scripts/pgd_cross_center/run_pilot_iter4.sh
# 内部: --n_epochs 100 --patience 20 --es_metric val_macro_auroc
# 输出:
#   /root/autodl-tmp/synth_anchored_super5_v13/extra_k200/best_model.pt
#                                                        /training_log.json
#                                                        /eval_result.json
#                                                        /early_stop_info.json (epoch_stopped, best_auroc, best_epoch)
#   /root/autodl-tmp/synth_anchored_super5_v13/nin_k200/...
```

**预算** (3 cells):
- 最坏 (patience 不触发，跑满 100 ep): 100 ep × ~1.5 min/epoch × 3 cell = **~7.5h**
- 最好 (patience 在 ep 30 触发): 50 ep × ~1.5 min/epoch × 3 cell = ~3.75h
- 预期 (cell 在 ep 50-70 收敛): **~5.25h**

---

## Stage 3 — Eval + 决策

复用 `scripts/triple_labels/eval_crosscenter.py --scheme super5`:
- PTBXL fold 10 (in-domain) — **PTBXL 不掉太多**的检查面
- PN2021 7 中心 (PN2021_FORBIDDEN={'ptb-xl','ptbxl'} 已硬排除)
- MIMIC test split (zero-shot)
- Per-class AUROC + macro, bootstrap CI n=1000
- **`--exclude_ref_ids /path/to/{tag}_k200.meta.json` 排除 K=200 ref 的 record_ids** (Phase 1.5 实施;
  当前 argparse 缺失，必须新增。从 meta.json 读 `ref_record_ids` list, eval 同中心 PN2021 时 skip
  `record_id ∈ ref_record_ids` 的 records, 报告 `effective_n` = `n_records - n_excluded`)

**Aggregate** (`scripts/pgd_cross_center/aggregate_pilot_results.py`):
- Table A: 3 cells × 7 centers macro AUROC + per-center delta vs super5 baseline 0.8388
- Table B: 3 cells × 5 classes per-class AUROC (per-class trust gate 触发用 *)
- Table C: MIMIC zero-shot delta
- Table D: PTBXL fold10 in-domain delta (确认 AT 不严重伤 in-domain — Δ ≥ -0.5pp 接受)
- Table E: 同中心 with-ref vs without-ref (`--exclude_ref_ids` on/off) AUROC delta — 量化 ref 泄漏校正幅度

### 决策门 (Rev 13.2 — 3 cells)

```
Per cell pass iff (使用 --exclude_ref_ids 后的同中心 + 跨中心 macro):
  macro AUROC delta > +0.30pp  AND  macro AUPRC delta > +0.50pp
  AND
  PTBXL fold10 in-domain AUROC delta ≥ -0.50pp   ← 你的"PTBXL 不掉太多"约束
  (per Issue #34 双 metric 阈值;
   AUROC 噪声窄 ~0.005, AUPRC 噪声宽 ~0.01, 各 +1.5σ-2σ over noise)

≥2/3 cell pass:
  → 推 5-center confirmatory phase (额外 2 cell × ~1.75h ≈ 3.5h)
  → 然后写论文，附 Issue #24 Ablation A+B (mix ratio + buffer eviction) + 完整 audit

0-1/3 cell pass:
  Decision tree:
  (a) chap_shaoxing per-cell 仍 +0.40pp 一致 (pilot 1-3 历史信号) → 改写 paper framing 为
      "cross-center per-target generalization with semantic-aware trust gating"
      (per pilot summary recommendation);
  (b) chap_shaoxing 信号也消失 → scale Stage 1 pool 300→1200/cell, rerun once;
  (c) 仍 fail → 锁定瓶颈在 ECGTwin 词表 (CLAUDE.md "ECGTwin landmine: STTC reverse-supervision")
      或 base_vector class control (per ecgtwin_prompt_vs_basevector.md 推荐 #2)
      → paper limitations 章节明确披露, future work 留 per-class base_vector token 训练
```

### 报告

`docs/synth_anchored_super5_pilot_iter4.md`:
- Pre-conditions check ✓
- Stage 0.4 per-class sanity AUROC + class_trust map
- Stage 2 ASR / Einthoven 时序图 (per-cell)
- Stage 3 aggregate 4 table
- 决策门通过/未通过的归因
- Memory update: `synth_anchored_super5_pilot1_findings.md` 加 iter4 行

---

## Compute budget (Rev 13)

| Stage | 单 cell | × cell | 累计 |
|---|---|---|---|
| Phase 1: georgia K=200 ref pool prep (multiprocessing×12) + 重生 extra/nin (3-class priority) | 30 min | × 1 | 30 min |
| Phase 1.5: 实施 `eval_crosscenter.py --exclude_ref_ids` + smoke test | 20 min | × 1 | 20 min |
| Stage 0.4 sanity (Stage 1 后置 gate) | 5 min | × 3 | 15 min |
| Stage 1 synth pool 300/cell (vanilla, no CT) | 6 min | × 3 | 18 min |
| Stage 2 online AT ≤100 ep + patience=20 (K_pgd=10, K_anchor=300) | 预期 ~1.75h, 最坏 ~2.5h | × 3 | **~5.25h 预期 / ~7.5h 最坏** |
| Stage 3 eval (3 集 bootstrap, exclude_ref_ids on) | 12 min | × 3 | 36 min |
| 报告 + memory + 图 | 30 min | × 1 | 30 min |
| **Pilot iter 4 总计 (3 cells)** | | | **~7.5h 预期 / ~10h 最坏** |
| 决策门通过 → 5-center confirmatory (额外 2 cell, 同 protocol) | | | +3.5h 预期 |
| Issue #24 Ablation A+B (mix ratio + buffer eviction, post-pilot) | | | +3h |
| Audit (post-pilot, optional, per Plan Rev 5) | | | +1.5h |
| **完整 ablation (如通过决策门)** | | | **~16h 预期** |

vs Pilot Rev 8 实际 ~3h (50 ep, 2 cells, K_pgd=5)；vs Rev 13.1 ~5h (2 cells, 100 ep)；本 Rev 13.2 加第 3 cell + ref 隔离实施约 +2.5h 预期。4090 一晚仍可跑完 pilot iter 4。

---

## Risks (Rev 13)

| 风险 | 征兆 | 应对 |
|---|---|---|
| K_pgd 5→10 ASR 仍饱和 0.95+ | Stage 2 log 全程 ASR≥0.95 | 加 ε 2.0→2.5；如仍饱和说明 victim 已学到 synth manifold 的弱点表示，paper 报告 "PGD on synth saturates after K_pgd=10" 作为 method limitation |
| Pool 300 不够 → cross-center 信号消失 | iter4 chap_shaoxing 跨中心 transfer 不再 +0.4pp | Scale 300→1200/cell rerun (~24 min Stage 1 + 不变 Stage 2)；如仍 fail 是流形质量问题不是算力 |
| **georgia MI 源端弱 (n_pos=7)** | Stage 0.4 sanity georgia MI AUROC < 0.55 | H4 trust gate 自动写 MI=0.0 in georgia ckpt; MI synth 仍由 extra/nin cell 贡献 cross-center 信号; paper Table 显式披露 georgia MI source-eval n_pos<10 → macro skip MI |
| H4 trust gate 让 buffer 失活 | buffer 有效大小 < 100 / epoch | 监控 `buffer.size()`；如 < 100 调整 trust threshold 0.7→0.6 (放宽中级类) |
| K_anchor=300 walker 显存爆 | nvidia-smi > 22GB | Fallback K_anchor=200 + batch=24 (PGD 100s → 80s); 进一步 fallback K_pgd=8 |
| Real-dominated mix 仍跑出 source-center plain | extra cell 自身 eval 仍 -0.0 | Stutz 2019 caveat 自洽现象，paper framing 应直接 cross-center; per-cell gate 仍可 pass |
| ECGTwin STTC 倒置导致 buffer 标 STTC 但内容像 MI | Stage 0.4 STTC AUROC 仍 < 0.5 | H4 trust 自动写 0 → 该类 adv contrib 全零，paper 显式披露 trust map (transparency) |
| Patient leakage from K=200 ref into eval | 任何 cell macro 异常高 (>baseline+1pp) | `--exclude_ref_ids` 已实施 (Plan Rev 7 Issue #39); eval 报告标 effective n |
| Early-stop 在 noisy AUROC 上误触发 | val_macro_auroc 单 epoch 抖动超过 +0.001 但实际未真正改善 | patience=20 已经留 buffer; 监控 best_epoch + plateau detection (val AUROC EMA decay=0.9 平滑后再判 patience) |
| Patience 跑满 100 ep 仍无 plateau | val AUROC 单调上升到 ep 100 未稳 | log 显示 val AUROC 仍线性升 → 该 cell 单独续训到 150 ep (单次手动 rerun, 不在 default protocol) |

---

## Critical files

| 角色 | 路径 | 状态 |
|---|---|---|
| Super5 victim ckpt | `/root/autodl-tmp/triple_labels/super5/best_model.pt` | ✅ 复用 |
| Super5 scheme | `scripts/triple_labels/label_schemes.py::get_scheme("super5")` | ✅ 复用 |
| Super5 dataset | `scripts/triple_labels/train_ptbxl.py::PTBXLDatasetScheme` | ✅ 复用 |
| Cross-center eval | `scripts/triple_labels/eval_crosscenter.py --scheme super5` | ✅ 复用 base, ⚠️ Phase 1.5 加 `--exclude_ref_ids` flag |
| Mode A PGD core | `adversarial/pgd_advdiff.py` | ✅ 复用 (assert num_classes 已 self.victim.num_classes) |
| ASR + 医学 gate | `adversarial/adv_validation.py` | ✅ 复用 (class-agnostic) |
| Online AT base | `scripts/crosscenter_tierM/online_adv_train_tierM.py` | ✅ fork base |
| Super5 ref pool prep | `scripts/ecgtwin_gen/prep_center_dataset_super5.py` | ⚠️ 需改 priority list 到 ["MI","STTC","NORM"] |
| Super5 SNOMED prompts | `scripts/ecgtwin_gen/super5_text_embeds.py` | ⚠️ 需缩到 NORM/MI/STTC 子码 |
| Synth gen (no CT default) | `scripts/ecgtwin_gen/generate_center_synth.py` | ✅ DONE (Rev 12) |
| **新** online AT iter4 | `scripts/pgd_cross_center/synth_online_at_super5.py` | 🆕 写 (~400 行 fork from online_adv_train_tierM.py + class_trust + K_pgd=10 + walker) |
| **新** iter4 runner | `scripts/pgd_cross_center/run_pilot_iter4.sh` | 🆕 写 ~30 行 |
| **新** aggregator | `scripts/pgd_cross_center/aggregate_pilot_results.py` | 🆕 写 ~120 行 (含 class_trust 列 + per-class table) |
| Synth pool 输出根 | `/root/autodl-tmp/synth_anchored_super5_v13/synth_latents/` | 🆕 |
| AT 训练产物根 | `/root/autodl-tmp/synth_anchored_super5_v13/{tag}_k200/` | 🆕 |
| Pilot iter 4 报告 | `docs/synth_anchored_super5_pilot_iter4.md` | 🆕 |

---

## Execution order (Rev 13)

```
Phase 1   (~40 min):
  (a) 改 prep_center_dataset_super5.py priority list 缩到 ["MI","STTC","NORM"]
  (b) 改 super5_text_embeds.py SNOMED_TO_PROMPT 缩到 NORM/MI/STTC 子码 + 重生 super5_text_embeds.pt
  (c) 生成 georgia K=200 ref pool (新, --center georgia)
  (d) 重生 extra/nin K=200 ref pool（用 3 类 priority + 新 text_embed）
Phase 1.5 (~20 min): 实施 eval_crosscenter.py --exclude_ref_ids:
  - argparse 加 `--exclude_ref_ids /path/to/meta.json [/path/to/meta2.json ...]`
  - load `ref_record_ids` from each meta.json → set
  - PN2021 record loop 内 `if record_id in excluded_ids: continue`
  - 报告 effective_n + n_excluded per center
  - smoke test: extra cell with vs without flag, 验证 n_excluded ≈ 200, AUROC 略降
Phase 2   (~18 min): Stage 1 vanilla 生成 3 cell × 300/cell pool
Phase 3   (~15 min): Stage 0.4 sanity (3 cells) → 写 class_trust.json
Phase 4   (~30 min): 写 synth_online_at_super5.py (fork online_adv_train_tierM.py，加 class_trust + walker + K_pgd=10 + early-stop patience=20 on val_macro_auroc)
Phase 5   (~10 min): Smoke test --n_epochs 3 (extra cell)
Phase 6   (~5.25h 预期 / ~7.5h 最坏): run_pilot_iter4.sh — 3 cells × ≤100 ep early-stop
Phase 7   (~36 min): aggregate_pilot_results.py + Stage 3 eval (--exclude_ref_ids on)
Phase 8   (~30 min): docs/synth_anchored_super5_pilot_iter4.md + memory update
                     决策: ≥2/3 cell pass +0.30/+0.50 双 gate AND PTBXL fold10 Δ ≥ -0.5pp?
```

---

## Non-goals (Rev 13 显式)

- ❌ CenterToken retrain / 修 5 trainer bug — drop CT 是定调 (Plan Rev 12)
- ❌ HYP / CD synth 生成 — Plan Rev 11 数字 GT 决策
- ❌ Pool 1M / epoch 400 / unsup-fraction 0.7 全盘 copy Wang 2023 — 算力 + 设定差异
- ❌ Tier-M scheme 回退 — Super5 victim 已是 production 选择
- ❌ Offline cadence — Plan Rev 9 已废弃
- ❌ AugMix-inject 在 buffer push 阶段 — Plan Rev 8 (Issue #25) 关闭，pilot 不变
- ❌ Mode B (DiT denoising PGD) / LoRA on DiT — 当前不在范围
- ❌ Memorization audit 在 pilot 阶段跑 — Plan Rev 5 改为 post-pilot optional
- ❌ Issue #24 Ablation A/B 在 pilot 阶段跑 — 完整 ablation 阶段才补
- ❌ Per-class learnable base_vector token (per ecgtwin_prompt_vs_basevector.md 推荐 #2) — future work，独立项目

---

## Paper framing (Rev 13)

3-component pipeline cite 学术 backing:

1. **PGD adversarial training** (Madry 2018, Wang 2023 ICML 144⭐ DM-Improves-AT)
   - Latent-space variant: Stutz 2019 (CVPR, 414 cites) + Wong-Kolter ICLR 2020
   - Cross-center novelty: "first cross-center on-manifold AT pipeline for 12-lead 1D ECG with semantic-aware trust gating"
   - 区分 Yang/La Cava 2025 (arxiv 2509.19564, 2D STFT pediatric 单中心) 不 claim "first manifold AT for ECG"

2. **ECGTwin synth (vanilla, prompt-driven)** (复用 ECGTwin author trained ckpt)
   - 3-class scope justified by digital GT validation (`docs/ecgtwin_super5_digital_gt_validation.md`)
   - prompt > base_vector > CenterToken hierarchy 实证 (`memory/ecgtwin_prompt_vs_basevector.md`)
   - Backing: Wilde MIDL 2024 medical TI K-shot (我们 K=200 ref, 1.5× synth ratio)

3. **H4 per-class trust gate** (semantic-aware, transparency component)
   - Phase 0.4 sanity → class_trust map → buffer push weight
   - Backing: dataset-conditional disagreement weighting in distributed AT (Liu AAAI 2023 memorization-weighted reweighting), audit transparency (Pizzi CVPR 2022 SSCD spirit)

**Discussion 必比对照**:
- Wang 2023 ICML (frozen pool + per-epoch PGD): 我们 = latent-space cross-center variant
- Stutz 2019 CVPR: on-manifold AT 提升 generalization, 我们 confirmed in 1D medical setting (chap_shaoxing transfer +0.40pp from pilot 1-3)
- Yang/La Cava 2025: 单中心 pediatric 数据效率, 我们 cross-center clinical robustness

**Limitations 必披露**:
- ECGTwin 词表对 STTC 与 MI 训练共现耦合, 子句 prompt 微调无法解纠缠 (CLAUDE.md "ECGTwin landmine")
- HYP/CD 数字 GT 不通过 → trust=0 → 这两类未受 AT 增益 (paper Table 显式)
- Source-center self-eval flat (Stutz 2019 caveat: on-manifold AT 不改善源域)
- 不 claim Lp robustness, 只 claim cross-center generalization (Liu 2020 Dual Manifold AT 警示)

---

## 历史 revision (Rev 1-12 摘要，详见 git log)

| Rev | 时间 | 主决策 |
|---|---|---|
| 1-7 | 2026-04-26 | scheme tierM→super5, K-grid 路径修订, online cadence 切换, 7-aspect 工程严谨性 |
| 8 | 2026-04-26 | compute optimization (multiprocessing prep + DDPM batch 100 + PGD batch 32 + workers 12), pilot ~3h |
| 9 | 2026-04-27 | (proposed but **降级**) pool 500→2000 + K_anchor 100→500 + walker — 被 Rev 10 H4 优先级覆盖 |
| 10 | 2026-04-27 | STTC prompt 修复尝试失败, 改 H4 per-class trust gate 为结构性 fallback |
| 11 | 2026-04-27 | Generation scope 缩到 NORM/MI/STTC (HYP/CD 数字 GT 0/3 通过, 结构性) |
| 12 | 2026-04-27 | Drop CenterToken (双 subagent 调研 5 trainer bug + 多层验证 NOOP) |
| **13** | **2026-04-27** | **Streamlined: 聚焦 Stage 1-3, Wang 2023 ICML 数据规模对齐 (K_pgd=10, no-revisit walker), pool 300/cell 起步可 scale 1200, real-dominated mix 反向 Wang** |
| **13.1** | **2026-04-27** | **Epochs 50 → 100 + early-stop patience=20 on val_macro_auroc**: Wang 295 ep equivalent target, Stutz/Rebuffi 100-ep canonical, 4090 算力翻倍 epoch 是免费 lever (Stage 1 不重生)。最坏 ~5h pilot, 预期 ~3.5h |
| **13.2** | **2026-04-27** | **3 cells (+georgia) + 真正实施 ref-eval 隔离**: 用户决策第 3 cell = georgia (baseline 0.8173 lift 空间 + AUPRC 0.6462 最高 + 4 类全足); MI n_pos=7 源端弱但 cross-center 仍贡献。用户提问发现 prep 端 ref_record_ids ✓ 但 eval 端 `--exclude_ref_ids` flag 缺失 → Phase 1.5 实施。决策门改 ≥2/3 cell pass + 加 PTBXL fold10 Δ ≥ -0.5pp 约束。预算 ~7.5h 预期 |
