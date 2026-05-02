# Next Auto Execution Plan

Date: 2026-05-01.

本文档把当前四项任务排序成可执行计划。原则：先建立 clean v3 baseline，再做生成与对抗训练，再做 corruption robustness。

## Phase 0: Runtime 和事实源

已同步 runtime skill：

```text
/root/.codex/skills/ecg-adv-gen/SKILL.md
```

事实源：

```text
docs/pipelines/
.codex/skills/ecg-adv-gen/SKILL.md
AGENTS.md
CLAUDE.md
```

Current long run:

```text
ECGTwin author reproduction:
  root: /root/autodl-tmp/ecgtwin_author_repro/full_dit_main_accel_bf16_mbs512
  status: partial IBE run saved; full author pipeline did not complete
  config: effective batch 65536, micro batch 512, workers 8, bf16 AMP
  observed: train_size=6408782, val_size=399499, ~=10.5s/optimizer step
  latest IBE metrics:
    epoch 1 train_loss=3.052135 eval_score=0.718906
    epoch 2 train_loss=1.694837 eval_score=0.657278
    epoch 3 train_loss=1.213244 eval_score=0.627823
    epoch 4 train_loss=0.919663 eval_score=0.625451
    epoch 5 train_loss=0.751632 eval_score=0.628861
    epoch 6 train_loss=0.634500 eval_score=0.611151
    epoch 7 train_loss=0.557742 eval_score=0.611722
    epoch 8 train_loss=0.498903 eval_score=0.616669
    epoch 9 train_loss=0.455224 eval_score=0.611132
  failure mode:
    after epoch 9 the main train process was killed; orphan DataLoader workers
    held the pipeline pipe/GPU memory until manually terminated
  saved artifacts:
    ibe_stage1/metrics.jsonl, loss_curve.csv/png, run_config.yaml,
    checkpoints/best.pt, checkpoints/latest.pt, checkpoints/IBE_best.pth
  retry recommendation:
    retry only after adding resume support or run with fewer workers and no
    persistent_workers; do not spend another 10h blindly rerunning the same config
```

Completed 1-2h GPU task:

```text
script: scripts/triple_labels/run_selected_pn2021_c_queue.sh
status: completed after author-repro failure cleanup
purpose: fill high-priority missing PN2021-C self-clean results
models:
  /root/autodl-tmp/latent_hull_super5_pilot/nin_M5_lambda025_ep20
  /root/autodl-tmp/latent_hull_super5_pilot/extra_M5_lambda025_ep20
  /root/autodl-tmp/latent_hull_super5_pilot/geo_M20_lambda025_ep20
outputs:
  eval_pn2021_c_stream_v1_selfclean_all4.json in each model dir
  refreshed /root/autodl-tmp/triple_labels/pn2021_c_*_summary.csv
result:
  included PN2021-C JSONs increased from 7 to 12 after optional K400 and v14
  prompt-token v14 is best absolute corrupted mean AUPRC = 0.492808
  synth-anchor K400 is second = 0.492707
  nin_M10_lambda025 is third = 0.492684
  geo_M20_lambda025 has the lowest enhanced self-clean AUPRC drop = 0.034599,
  but its absolute corrupted mean AUPRC = 0.491239 is not competitive enough
```

## Phase 1: PN2021 v3 clean re-evaluation

目的：用新的 PN2021 super5 v3 mapping 重新计算 baseline 每中心 AUROC/AUPRC。

确认的 IO/cache 决策：

```text
保留旧 compressed .npz clean cache，不覆盖旧结果。
新增 mmap-friendly PN2021 clean cache:
  signals.npy
  labels.npy
  record_ids.npy
  metadata.json
eval_crosscenter.py cache hit 时优先读取 mmap cache，旧 .npz 作为兼容 fallback。
PN2021 DataLoader 使用 CLI --num_workers，不再硬编码 2。
```

当前 baseline 已完成：

```text
output: /root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json
PTB-XL fold10: AUROC=0.9064, AUPRC=0.7754
PN2021 7-center average: AUROC=0.8344, AUPRC=0.5526
```

执行：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/label_schemes.py \
  --sanity --scheme super5 --mimic_n 2000

/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_crosscenter.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5 \
  --batch_size 256 \
  --num_workers 8 \
  --skip_mimic \
  --output_path /root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json
```

阻塞条件：

- `/root/autodl-tmp/triple_labels/super5/best_model.pt` 不存在。
- PN2021 raw 或 cache 目录不存在。

完成条件：

```text
eval_result_v3_super5_normsuppress.json exists
JSON metadata contains v3 mapping/hash
7 centers all evaluated
```

### Phase 1b: Historical Model Re-eval Candidate Scan

自动扫描：

```bash
find /root/autodl-tmp -path '*/best_model.pt'
```

扫描结果：

```text
total best_model.pt dirs: 129
```

优先级：

```text
P0 already done:
  /root/autodl-tmp/triple_labels/super5

P1 likely thesis-comparable super5 runs:
  /root/autodl-tmp/synth_anchored_super5_v14_ctv2/{extra,geo,nin}_k200
  /root/autodl-tmp/synth_anchored_super5_v13b_K400/{extra,geo,nin}_k400
  /root/autodl-tmp/synth_anchored_super5_v13/{extra,geo,nin}_k200
  /root/autodl-tmp/real_anchored_super5/{extra_real,geo_real,nin_real}_k200
  /root/autodl-tmp/pgd_adv_ablation/finetune/{chap,cpsc,extra,geo,nin}_k200
  /root/autodl-tmp/center_token_ablation/retrain/{ptbxl,mimic}/{chap,cpsc,extra,geo,nin}_k500

P2 historical/diagnostic:
  center_aware_ibe/retrain/*
  crosscenter*/ and crosscenter_tierM*/ old runs
  pgd_adv_ablation k50/k100 runs
  center_token_ablation k25/k50/k100/k200 runs

Excluded from super5 v3 re-eval unless a separate scheme is requested:
  /root/autodl-tmp/triple_labels/sub23
  /root/autodl-tmp/triple_labels/pn26
```

`chap/cpsc/extra/geo/nin` aliases must be mapped to canonical PN2021 centers in reports:

```text
chap  -> chapman_shaoxing
cpsc  -> cpsc_2018
extra -> cpsc_2018_extra
geo   -> georgia
nin   -> ningbo
```

P1 batch re-eval completed:

```text
summary: /root/autodl-tmp/triple_labels/p1_v3_eval_summary_20260501.jsonl
ok: 12 super5 checkpoints
failed: 15 incompatible old 6-class checkpoints
best P1 AUPRC:
  synth_anchored_super5_v13b_K400/nin_k400 -> PN2021 0.8334 / 0.5555
current Latent-Hull best AUPRC:
  latent_hull_super5_pilot/nin_M10_lambda025_ep20 -> PN2021 0.8341 / 0.5560
```

## Phase 2: ECGTwin Center Prompt Token

目的：实现 textual-inversion-style center-class prompt token。

确认的 cache 决策：

```text
旧 /root/autodl-tmp/center_token_ablation/datasets/*_k500.pt 只作为历史 ablation。
新 prompt-token 主线重建 ECGTwin-compatible raw-mV VAE latent cache。
新增两层 cache:
  center_full_latents/{center}.pt
  ref_selection/{center}_k500_seed42.json
```

先实现最小闭环：

1. `CenterClassTokenBank(5,768)`。
2. prompt compiler：`diagnosis prompt + <center_CLASS>` -> augmented `text_embed`。
3. per-center dataset 保留 5 类 token 训练记录。
4. 冻结 ECGTwin，只训练 token bank。
5. 先核查 ECGTwin 作者官方 prompt/推理链路是否能正常生成 CD/HYP。
6. 生成样本进入增强前必须通过 digital ECG gate 和 teacher/classifier gate。

HYP/CD smoke 已完成：

```text
/root/autodl-tmp/ecgtwin_smoke_authorprompt_hyp_cd_20260501/summary.json
/root/autodl-tmp/ecgtwin_smoke_authorprompt_tensors_20260501/digital_gt_validation.md
```

结论：作者官方 gallery 支持 HYP/CD 相关 prompt；当前环境能重新生成样本。HYP victim
响应强但 LVH 电压 gate 未过；RBBB prompt 有一次数字 CD gate 通过但 victim CD 概率低。
后续 HYP/CD 只可走 gated path，不能无条件当作可靠增强样本。

输出：

```text
/root/autodl-tmp/ecgtwin_center_prompt_tokens/<center>/
```

首批中心固定：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

K 策略：

```text
K=500 per center
K500 ref ids must be excluded from fine-tune validation/eval
```

阻塞条件：

- ECGTwin checkpoints 或 nomic text encoder 不可用。
- 目标中心可用样本不足，尤其某类 count < gate。

当前实现状态：

```text
DONE scripts/ecgtwin_gen/build_prompt_token_latent_cache.py
DONE methods/ecgtwin_gen/prompt_token/model.py
DONE methods/ecgtwin_gen/prompt_token/trainer.py
DONE scripts/ecgtwin_gen/train_center_prompt_tokens.py
DONE scripts/ecgtwin_gen/quality_ref_selection.py
DONE smoke:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/smoke_all4_bf16_steps5/
DONE formal token training:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4_steps2000/
  final loss=0.04253, mean token_delta_norm=1.1961, elapsed ~=87s
DONE prompt-token generation smoke:
  script: scripts/ecgtwin_gen/generate_center_prompt_token_synth.py
  output: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_smoke_v1/ningbo/
  result: NORM/MI/STTC digital pass in 1-seed smoke; HYP/CD still gate-fail
DONE gated prompt-token export:
  script: scripts/ecgtwin_gen/gate_prompt_token_synth.py
  source: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/
  generated: NORM=50, MI=50, STTC=50
  passed gates: NORM=41, MI=41, STTC=15, total=97/150
  exported:
    gated/gated_samples.npz
    gated/gated_samples.latent.npz
    gated/gated_samples.class_trust.json
    gated/gated_samples.ref_meta.json
DONE prompt-token downstream pilot:
  output: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20/
  settings: M=10, lambda=0.25, hull_steps=5, K_anchor=45, n_epochs=20
  PN2021 v3: AUROC=0.8342, AUPRC=0.5555
  delta vs baseline: -0.0002 / +0.0029
  conclusion: usable pipeline, clean AUPRC signal, STTC gate scarcity remains bottleneck
DONE balanced gated pool v4:
  extra STTC generated: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v3_sttc_boost/ningbo/
  extra STTC passed: 34/100
  merged pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v4_merged_balanced/ningbo/gated/
  counts: NORM=41, MI=41, STTC=49, total=131
RUNNING prompt-token balanced downstream pilot:
  output: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20/
  settings: M=10, lambda=0.25, hull_steps=5, K_anchor=90, n_epochs=20
NEXT center-token improvement:
  stop blind token_repeat/pool expansion sweeps
  add gate-aware token checkpoint/probe selection before downstream training
```

Executed follow-ups:

```text
prompt v4 balanced M10:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20
  PN2021 = 0.8344 / 0.5558

prompt token v2 repeat4:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v2_all4_balanced_repeat4_steps3000
  gate repeat4 = 85/150; over-steers NORM/MI toward STTC, not used downstream

prompt token v3 repeat2:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v3_all4_balanced_repeat2_steps2500
  merged gated pool = NORM 69, MI 41, STTC 43
  downstream PN2021 = 0.8345 / 0.5556

prompt v4 M5 adv025:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v4/ningbo_v4pool_M5_adv025_ep20
  PN2021 = 0.8343 / 0.5555

hybrid real+prompt:
  pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/hybrid_pools/nin_real200_promptv4_131/merged.latent.npz
  downstream: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_hybrid_v1/nin_real200_promptv4_131_M10_ep20
  PN2021 = 0.8345 / 0.5549

hybrid real+prompt MI-only:
  pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/hybrid_pools/nin_real200_prompt_mi40/merged.latent.npz
  downstream: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_hybrid_v2/nin_real200_prompt_mi40_M10_ep20
  PN2021 = 0.8343 / 0.5540
  conclusion: MI-only prompt supplementation is worse than balanced prompt pool;
              do not continue one-class prompt additions without source/class control.

hybrid source-aware sampler:
  code:
    scripts/ecgtwin_gen/merge_latent_pools.py now writes source_ids/source_names
    scripts/pgd_cross_center/synth_online_at_super5.py supports:
      --source_sampling_strategy source_weighted
      --source_weights real_anchor=1.0,prompt_token=0.35
      --source_class_weights MI:prompt_token=1.0
  pool:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/hybrid_pools/nin_real200_promptv4_131_srcaware/merged.latent.npz
  downstream:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_hybrid_v3/nin_real200_promptv4_131_srcaware_M10_ep20
  quick eval best:
    0.8455 / 0.6941
  full PN2021:
    0.8342 / 0.5552
  conclusion:
    source-aware sampling improves naive hybrid but still does not beat pure
    prompt-token v4 or real-anchor nin_M10.

quality-aware top40:
  selector: scripts/ecgtwin_gen/select_quality_prompt_token_pool.py
  pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v9_quality_top40/ningbo/gated/gated_samples.latent.npz
  downstream: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v5/ningbo_quality_top40_M10_ep20
  PN2021 = 0.8343 / 0.5551
  conclusion: high-confidence-only selection is worse than v4 balanced pool;
              keep boundary-informative generated samples.

mixed confidence/boundary selectors:
  selector: scripts/ecgtwin_gen/select_mixed_prompt_token_pool.py
  v11:
    pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v11_mixed_h40_b40_q50/ningbo/gated/gated_samples.latent.npz
    downstream: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v6/ningbo_mixed_h40_b40_q50_M10_ep20
    policy: 50/class, high 40%, boundary 40%, diverse 20%
    PN2021 = 0.8345 / 0.5547
  v12:
    pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v12_mixed_h60_b20_q50/ningbo/gated/gated_samples.latent.npz
    downstream: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v7/ningbo_mixed_h60_b20_q50_M10_ep20
    policy: 50/class, high 60%, boundary 20%, diverse 20%
    PN2021 = 0.8342 / 0.5542
  conclusion:
    mixing older boundary-heavy candidate pools increases ASR but hurts full
    PN2021 AUPRC; do not continue this selector family without better upstream
    medical/reference filtering.

v4 balanced downstream sweeps:
  seed rerun:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v8/ningbo_v4_balanced_M10_ep20_seed13
    PN2021 = 0.8343 / 0.5546
  adv_weight 0.35:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v9/ningbo_v4_balanced_M10_adv035_ep20
    PN2021 = 0.8342 / 0.5554
  adv_weight 0.45, original seed:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v10/ningbo_v4_balanced_M10_adv045_seed030_ep20
    PN2021 = 0.8340 / 0.5535
  conclusion:
    original v4 balanced M10 remains the best pure prompt-token run. Downstream
    weight/seed sweeps are not the current bottleneck.

fresh v1 generation expansion:
  generated:
    v13: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v13_v1_fresh240/ningbo
      raw NORM/MI/STTC = 80/80/80
      gated = NORM 58, MI 32, STTC 14
    v14: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v14_v1_sttc_boost160/ningbo
      gated STTC = 36/160
    v15: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v15_v1_mi_boost120/ningbo
      gated MI = 53/120
  selected:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v16_fresh_v1_quality50/ningbo/gated/gated_samples.latent.npz
    policy: top-quality quota 50/class from v13+v14+v15 candidates
  downstream:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v11/ningbo_fresh_v1_quality50_M10_ep20
    quick best = 0.8421 / 0.6939
    full PN2021 = 0.8343 / 0.5548
  conclusion:
    fresh expansion with the same v1 token bank is not enough. The next
    center-token attempt should change reference selection or token training,
    not just generate more random samples.

multi-vector / quality-reference execution:
  quality-ref selector:
    script: scripts/ecgtwin_gen/quality_ref_selection.py
    output seed1042:
      ningbo = CD82/HYP80/MI120/NORM80/STTC138
      chapman_shaoxing = CD111/HYP89/MI40/NORM80/STTC180
      cpsc_2018 = CD261/NORM80/STTC159
      georgia = CD107/HYP101/MI7/NORM80/STTC205
  factorized MV4:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v4_qualityref_factorized_c1_cls1_res2_steps2500_seed1042
    gate = 98/240 (NORM45, MI25, STTC28)
  direct MV4 quality-ref:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v5_qualityref_direct_mv4_steps2500_seed1042
    first gate = 105/240 (NORM37, MI22, STTC46)
    boost/select v20 -> full PN2021 = 0.8335 / 0.5543
    all-gated v21 -> full PN2021 = 0.8340 / 0.5546
  ningbo-only direct MV4:
    3000-step gate = 71/240 (NORM32, MI19, STTC20)
    1500-step gate = 91/240 (NORM47, MI30, STTC14)
    conclusion: target-only training overfits and hurts STTC.
  seed42 direct MV4:
    /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v8_seed42_direct_mv4_steps2500
    first gate = 108/240 (NORM52, MI32, STTC24)
    STTC boost = 56/160; MI boost = 75/160
    v27 selected pool = NORM50/MI50/STTC50
    downstream:
      /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03
      full PN2021 = 0.8342 / 0.5558
    conclusion: direct MV4 with old seed42 references matches v4 AUPRC, but
    does not clearly exceed the simpler v1 token bank.
  checkpoint-probe direct MV4:
    trainer now supports --save_every.
    /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v9_seed42_direct_mv4_ckptprobe_steps2500
    probe result from 20/class:
      step500 = 25/60 (NORM12, MI4, STTC9)
      step1000 = 25/60 (NORM13, MI8, STTC4)
      final = 24/60 (NORM14, MI6, STTC4)
    composed hybrid:
      script: scripts/ecgtwin_gen/compose_prompt_token_bank_by_class.py
      bank: /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v10_seed42_direct_mv4_class_ckpt_hybrid/prompt_token_bank.pt
      NORM=final, MI=step1000, STTC=step500
    one-shot gate v28 = 119/240 (NORM43, MI42, STTC34)
    selected v30 pool = NORM50/MI50/STTC50
    downstream:
      /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v15/ningbo_v30_classckpt_balanced_q50_M10_ep20_seed03
      full PN2021 = 0.8342 / 0.5546
    conclusion: checkpoint-aware class composition improves gate diagnostics but
    top50/class selection still loses full PN2021 AUPRC.
```

Current best prompt-token result:

```text
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20
PN2021 v3 = 0.8344 / 0.5558

New MV4 baseline:
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03
PN2021 v3 = 0.8342 / 0.5558
```

Current best overall AUPRC result remains:

```text
/root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20
PN2021 v3 = 0.8341 / 0.5560
```

Latest prompt-token v31 run:

```text
pool:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v31_seed42_mv4_mixed_diverse_q50/ningbo/gated
selection:
  NORM=50, MI=50, STTC=50
downstream:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v16/ningbo_v31_mixed_diverse_q50_M10_ep20_seed03
result:
  best quick AUROC/AUPRC = 0.8285 / 0.6448
  full PN2021 v3 = 0.8340 / 0.5550
  PN2021-C mean drop = 0.022597 AUROC / 0.034711 AUPRC
interpretation:
  diversity-preserving v31 did not beat v4 prompt-token (0.8344 / 0.5558) or
  real-anchor nin_M10 (0.8341 / 0.5560). PN2021-C was still run at user request;
  it is slightly better than prompt v4 on mean AUPRC drop but worse than
  real-anchor Latent-Hull.
```

Prompt-token diagnostics completed:

```text
script: scripts/ecgtwin_gen/export_prompt_token_diagnostics.py
out_dir: /root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics
token_summary.csv: 530 token rows across 14 banks
token_cosine.csv: 15276 pairwise cosine rows
```

Active multi-center prompt-token v32 pilot:

```text
script: scripts/ecgtwin_gen/run_multicenter_prompt_token_v32.sh
status: queued/running after PN2021-C fill-in queue
token bank: v8_seed42_direct_mv4_steps2500
centers: chapman_shaoxing, cpsc_2018, georgia
classes: NORM, MI, STTC
n_per_class: 50
steps: 25
gate: digital ECG gate + victim p_target >= 0.30
output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v32_seed42_direct_mv4_multicenter_q50/
purpose:
  close the multi-center prompt-token pilot gap before running any more
  downstream online-AT sweeps
result:
  chapman_shaoxing kept 90/150: NORM 38, MI 33, STTC 19
  cpsc_2018 kept 59/100: NORM 38, STTC 21; no MI primary refs in v3 cache
  georgia kept 78/150: NORM 38, MI 28, STTC 12
  not ready for downstream; run class-targeted v33 boost first
```

Active multi-center prompt-token v33 boost:

```text
script: scripts/ecgtwin_gen/run_multicenter_prompt_token_v33_boost.sh
status: queued/running after v32
boost:
  chapman_shaoxing STTC x80
  cpsc_2018 STTC x80
  georgia MI x80 + STTC x80
merged output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v34_seed42_direct_mv4_multicenter_merged/
downstream gate:
  require chapman and georgia NORM/MI/STTC >=30 after merge;
  require cpsc_2018 NORM/STTC >=30 and record MI as missing-primary-label limitation
result:
  chapman_shaoxing v34 counts: NORM=38, MI=33, STTC=44 -> ready
  cpsc_2018 v34 counts: NORM=38, STTC=59, MI=0 -> ready except MI limitation
  georgia v34 counts: NORM=38, MI=74, STTC=25 -> needs STTC boost2
```

Active georgia STTC v35 boost:

```text
script: scripts/ecgtwin_gen/run_georgia_prompt_token_v35_sttc_boost.sh
status: queued/running after v33
boost: georgia STTC x80, seed=2042
merged output:
  /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v36_seed42_direct_mv4_multicenter_merged2/
result:
  chapman_shaoxing v36 counts: NORM=38, MI=33, STTC=44
  cpsc_2018 v36 counts: NORM=38, STTC=59, MI=0
  georgia v36 counts: NORM=38, MI=74, STTC=37
```

Active v36 downstream online-AT pilots:

```text
script: scripts/ecgtwin_gen/run_multicenter_prompt_token_v36_online_at.sh
status: queued/running after v35
runs:
  chapman_shaoxing -> online_at_multicenter_v36/chapman_v36_M10_ep20_seed03
  cpsc_2018        -> online_at_multicenter_v36/cpsc_v36_M10_ep20_seed03
  georgia          -> online_at_multicenter_v36/georgia_v36_M10_ep20_seed03
after each run:
  eval_crosscenter.py -> eval_result_v3_super5_normsuppress.json
decision:
  run PN2021-C only if one v36 pilot reaches competitive clean PN2021 AUPRC
result:
  chapman_v36_M10_ep20_seed03 -> PN2021 0.8344 / 0.5554
  cpsc_v36_M10_ep20_seed03    -> PN2021 0.8340 / 0.5553
  georgia_v36_M10_ep20_seed03 -> PN2021 0.8343 / 0.5541
  none beats prompt-token v4 (0.8344 / 0.5558) or real-anchor nin_M10
  (0.8341 / 0.5560), so PN2021-C is skipped for these v36 pilots.
```

Next useful center-token improvement:

```text
gate-aware token checkpoint/probe selection:
  during token training, save token banks at multiple steps/delta norms.
  For each checkpoint, run a small held-out generation probe for NORM/MI/STTC.
  Score by digital pass, victim top1/p_target, per-class balance, latent
  diversity, and reference coverage before doing expensive downstream training.
  Downstream only if a probe beats both:
    v4 gate profile: NORM41/MI41/STTC49 from 250 generated
    v27 gate profile: NORM50/MI50/STTC50 after class boost
  Current v30 finding: better probe/gate is not enough; preserve diversity or
  validate with a stronger held-out PN2021 subset before full downstream runs.

reference policy:
  seed1042 quality refs improved some per-class counts but underperformed full
  PN2021; keep old seed42 refs as the current stronger baseline.

Avoid next:
  blind token_repeat increase,
  blind M/adv_weight sweeps,
  naive real+prompt pool concatenation without per-source weighting.
  target-center-only token training without held-out gate probes.
```

Implemented code-level improvement:

```text
Source-aware sampler/weights in synth_online_at_super5.py:
  source=real_anchor or prompt_token
  per-class quota preserved
  prompt-token anchors receive lower outer adv_weight or lower sampling ratio
  boundary samples retained with QualityAwareBuffer score instead of discarded

Result:
  source-aware hybrid recovered some AUPRC vs naive hybrid, but did not beat
  pure prompt-token v4. Keep this as an ablation/control mechanism.
```

正式训练推荐命令：

```bash
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --total_steps 2000 \
  --batch_size 16 \
  --num_workers 4 \
  --amp_dtype bf16 \
  --save_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4
```

## Phase 3: Latent-Hull TA-OMAT

目的：把 online AT 的 latent attack 从自由 `delta` 改成 same-label convex hull。

第一版：

```text
M candidates = [5, 10, 20] ablation
main default: M=10, lambda=0.25, inner_steps=5
z_adv = (1 - lambda) * z0 + lambda * sum_i softmax(a_i) * z_i
```

第一版只做 same-label combination，不做跨类别 soft-label mixing：

```text
z_i same primary class or exact same multi-hot label as z0
```

系数策略：

```text
w_i = softmax(a_i), so w_i >= 0 and sum_i w_i = 1.
Initialize a as one-hot-like logits:
  one candidate starts with most weight, usually nearest same-label neighbor or z0-compatible anchor.
During inner attack:
  optimize a by gradient ascent on victim BCE/margin loss for 5 steps.
Keep lambda fixed at 0.25 in A1; A2-light tries lambda in [0.1, 0.25, 0.4].
Accepted samples must pass digital/teacher/class gates.
```

Coefficient ablations:

```text
C0 one-hot nearest neighbor, no optimization
C1 uniform weights, no optimization
C2 random Dirichlet(alpha=1), no optimization
C3 optimized softmax weights from one-hot init  # main
```

Implementation update:

```text
DONE adversarial/latent_hull_pgd.py --weight_mode
DONE scripts/pgd_cross_center/synth_online_at_super5.py --hull_weight_mode

Executable modes:
  C0: --hull_weight_mode one_hot
  C1: --hull_weight_mode uniform
  C2: --hull_weight_mode dirichlet --hull_dirichlet_alpha 1.0
  C3: --hull_weight_mode optimized
```

Coefficient/lambda ablation results:

| ablation | run | PN2021 AUROC/AUPRC | status |
|---|---|---:|---|
| C0 one-hot | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C0_onehot_ep20` | 0.8341 / 0.5547 | complete |
| C1 uniform | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C1_uniform_noabort_ep20` | 0.8341 / 0.5549 | complete |
| C2 Dirichlet | `/root/autodl-tmp/latent_hull_coeff_ablation/nin_M10_lambda025_C2_dirichlet_noabort_ep20` | 0.8344 / 0.5550 | complete |
| C3 optimized | `/root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.8341 / 0.5560 | current main |
| lambda 0.10 | `/root/autodl-tmp/latent_hull_lambda_ablation/nin_M10_lambda010_ep20` | 0.8340 / 0.5546 | complete |
| lambda 0.40 | `/root/autodl-tmp/latent_hull_lambda_ablation/nin_M10_lambda040_ep20` | 0.8339 / 0.5552 | complete |

Decision: keep optimized weights and `lambda=0.25` as the default. Fixed/random
coefficients and stronger/weaker lambda do not beat the current AUPRC result.

`M=100` 不进入第一轮；仅在 M=20 gate pass rate 和 downstream delta 都稳定后作为诊断实验。

执行前需要：

- Phase 1 clean baseline。
- Phase 2 生成的 `.latent.npz` 或 real-anchor latent pool。
- K=500 ref ids 排除 meta。

完成条件：

```text
online AT checkpoint
train_result.json
eval_result_v3_super5_normsuppress.json
gate_pass_rate / ASR / buffer stats
```

当前实现状态：

```text
DONE adversarial/latent_hull_pgd.py
DONE scripts/pgd_cross_center/synth_online_at_super5.py --attack_mode latent_hull
DONE generator smoke:
  python -m adversarial.latent_hull_pgd --smoke --M 5 --hull_lambda 0.25 --hull_steps 1
DONE online smoke:
  /root/autodl-tmp/latent_hull_smoke/extra_M5_steps1_cap/
```

Smoke 关键指标：

```text
delta_mean=1.9772, delta_max=2.0000
hull_weight_entropy_mean=0.0164, hull_weight_top1_mean=0.9981
ASR=1.0, Einthoven p95=0.1512, buffer_size=6
```

下一步自动执行：

```text
M=5,10,20 on real-anchor pools:
  extra_real_k200.latent.npz
  geo_real_k200.latent.npz
  nin_real_k200.latent.npz
main lambda=0.25, hull_steps=5, hull_lr=0.3, pgd_eps=2.0
```

Completed pilot on `extra_real_k200`:

| run | PN2021 avg AUROC | PN2021 avg AUPRC | delta vs baseline |
|---|---:|---:|---:|
| baseline | 0.8344 | 0.5526 | - |
| M=5 | 0.8346 | 0.5557 | +0.0002 / +0.0031 |
| M=10 | 0.8345 | 0.5553 | +0.0002 / +0.0027 |
| M=20 | 0.8349 | 0.5550 | +0.0005 / +0.0024 |

Next pending pilots:

```text
geo_real_k200: MI=7, NORM=72, STTC=121
nin_real_k200: MI=11, NORM=113, STTC=76
```

Three-center M-grid is complete:

```text
summary JSON:
  /root/autodl-tmp/latent_hull_super5_pilot/summary_mgrid_v1.json
best AUPRC:
  nin M=10 -> PN2021 AUROC/AUPRC = 0.8341 / 0.5560
best AUROC:
  extra M=20 -> PN2021 AUROC/AUPRC = 0.8349 / 0.5550
recommended main default:
  M=10, lambda=0.25, hull_steps=5
```

Prompt-token gated Latent-Hull pilot is complete:

```text
pool: /root/autodl-tmp/ecgtwin_prompt_token_super5/generated_pool_v2/ningbo/gated/gated_samples.latent.npz
run:  /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20
clean PN2021 v3: 0.8342 / 0.5555
PN2021-C stream eval: eval_pn2021_c_stream_v1.json
interpretation:
  clean AUPRC improves over baseline, but PN2021-C drop does not improve consistently.
  real-anchor nin_M10 remains stronger on absolute corrupted AUPRC, not on
  self-clean relative drop.
```

## Phase 4: PN2021-C corruption benchmark

目的：仿 ImageNet-C，测 clean baseline 和增强模型在 corrupted target-center 数据上的 AUROC/AUPRC drop。

脚本已新增并 smoke tested：

```text
scripts/triple_labels/build_pn2021_corruptions.py
scripts/triple_labels/eval_pn2021_corruptions.py
```

确认的磁盘策略：

```text
/root/autodl-tmp currently has about 66GB free.
PN2021-C 默认 streaming/batch 构建+评测，只保存 metrics JSON。
不强制保留 4 centers x 5 corruptions x 5 severities 的全部物化副本 cache。
如需保存 cache，只保存 selected corruption/severity 或 smoke cache。
```

Completed PN2021-C eval/export:

```text
script: scripts/triple_labels/export_pn2021_c_summaries.py
outputs:
  /root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_corruption_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_severity_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_center_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_per_class_summary.csv
included: 7 full PN2021-C JSONs
```

Current PN2021-C self-clean mean drop ranking, lower AUPRC drop is better:

| rank | model | mean corrupted AUPRC | mean AUROC drop | mean AUPRC drop |
|---:|---|---:|---:|---:|
| 1 | `triple_labels/super5` | 0.488896 | 0.022576 | 0.034103 |
| 2 | `latent_hull_super5_pilot/geo_M20_lambda025_ep20` | 0.491239 | 0.022772 | 0.034599 |
| 3 | `ecgtwin_prompt_token_super5/online_at_pilot_v16/ningbo_v31_mixed_diverse_q50_M10_ep20_seed03` | 0.491427 | 0.022597 | 0.034711 |
| 4 | `latent_hull_coeff_ablation/nin_M10_lambda025_C2_dirichlet_noabort_ep20` | 0.491276 | 0.022652 | 0.034737 |
| 5 | `synth_anchored_super5_v13b_K400/nin_k400` | 0.492707 | 0.022284 | 0.034823 |
| 6 | `ecgtwin_prompt_token_super5/online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03` | 0.492808 | 0.022626 | 0.034933 |
| 7 | `ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20` | 0.491379 | 0.022716 | 0.035044 |
| 8 | `latent_hull_super5_pilot/extra_M5_lambda025_ep20` | 0.492100 | 0.022882 | 0.035049 |
| 9 | `latent_hull_super5_pilot/nin_M10_lambda025_ep20` | 0.492684 | 0.022671 | 0.035135 |
| 10 | `ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20` | 0.492127 | 0.022780 | 0.035157 |
| 11 | `latent_hull_lambda_ablation/nin_M10_lambda040_ep20` | 0.491948 | 0.022677 | 0.035201 |
| 12 | `latent_hull_super5_pilot/nin_M5_lambda025_ep20` | 0.491931 | 0.022672 | 0.035207 |

Interpretation: use self-clean corrected drop for robustness sensitivity.
Prompt-token v14 is now the best absolute corrupted-AUPRC checkpoint, but it
does not reduce relative self-clean corruption drop. `geo_M20_lambda025` reduces
enhanced-model self-drop the most, but its absolute corrupted AUPRC is too low
to become the main result.

Disk hygiene update:

```text
System disk was 3.6GB free; conda/rattler caches and Python bytecode caches were cleaned.
Current / free: about 9.7GB.
Keep all future samples/checkpoints/eval caches under /root/autodl-tmp.
Use streaming PN2021-C unless selected corrupted caches are explicitly needed.
```

第一版 corruption：

```text
powerline_noise
emg_noise
baseline_wander
baseline_shift
random_leads_masking
```

public severity：

```text
1 2 3 4 5
```

内部映射：

```text
2 4 6 8 10
```

完成条件：

```text
eval_pn2021_c.json
relative_drop_vs_clean
mean_by_corruption
mean_by_severity
```

## 已确认决策

1. 第一批目标中心固定为 `ningbo chapman_shaoxing cpsc_2018 georgia`。
2. 每中心 token 训练直接使用 K=500，不再做 K=200 smoke。
3. K=500 ref ids 后续测试 fine-tune/增强效果时必须从 validation/eval 中排除。
4. HYP/CD 需要先核查 ECGTwin 作者官方是否支持生成，并跑 smoke generation。
5. Latent-Hull 第一版做 same-label combination。
6. 历史模型重评测名单由脚本自动扫描后给候选列表。
7. 新增 PN2021 mmap cache，保留旧 `.npz`。
8. 新 prompt-token 主线重建 ECGTwin-compatible raw-mV VAE latent cache；旧 K500 只做历史 ablation。
9. PN2021-C 默认 streaming/batch metrics，不物化保存全部 corruption 副本。
10. Latent-Hull 第一轮 M 数组为 `[5, 10, 20]`，默认 `M=10, lambda=0.25`，暂不跑 `M=100`。
11. 5 类 token 全部训练；HYP/CD 只走 gated path，不无条件进入增强主结果。

## Next Center Token v2 Auto Plan

目的：让 center token 对 EfficientNet1DV2 的增益从“可用小信号”变成更稳定的
clean AUPRC / corruption robustness 增益。

实现已准备：

```text
scripts/ecgtwin_gen/train_center_prompt_tokens.py:
  --sample_strategy center_class_balanced
  --token_repeat 4

scripts/ecgtwin_gen/generate_center_prompt_token_synth.py:
  --token_repeat 4
```

自动执行顺序：

1. 等当前 balanced pool v4 online AT pilot 完成并跑完整 PN2021 v3 eval。
2. 训练 prompt-token bank v2：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
/root/miniforge3/envs/ECGTwin/bin/python -u scripts/ecgtwin_gen/train_center_prompt_tokens.py \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --total_steps 3000 \
  --batch_size 16 \
  --num_workers 4 \
  --amp_dtype bf16 \
  --sample_strategy center_class_balanced \
  --token_repeat 4 \
  --save_dir /root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v2_all4_balanced_repeat4_steps3000
```

3. 用 v2 token bank 先在 `ningbo` 生成 NORM/MI/STTC each 50，gate 后检查 pass rate。
4. 如果任一主类 gated count < 40，则只补生成该类直到 >=40 或达到额外 200 条上限。
5. 合并 gated pool 后跑 Latent-Hull online AT：

```text
M=10, lambda=0.25, hull_steps=5, K_anchor=min(120, 3 * min_class_count)
n_epochs=20, quick_eval_n_per_center=300
```

6. 满足以下任一条件就跑完整 PN2021-C：

```text
clean PN2021 AUPRC >= 0.5560
or quick_eval AUPRC improves over v4 by >= 0.002
or AUROC improves while AUPRC remains within 0.001
```

否则记录失败原因，并优先调 prompt/reference/gate，而不是盲目加 epoch。
