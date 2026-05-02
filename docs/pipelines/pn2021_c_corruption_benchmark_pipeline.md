# PN2021-C Corruption Benchmark Pipeline

本文档定义参考 ImageNet-C 的 PN2021 corruption benchmark，用于评估 EfficientNet1DV2 在目标中心增强/腐蚀数据集上的 AUROC/AUPRC 下降。

## 目标

构建离线 `PN2021-C`：

```text
clean PN2021 v3 cache
  -> center x corruption x severity corrupted cache
  -> same EfficientNet1DV2 checkpoint eval
  -> AUROC/AUPRC drop vs clean
```

当前确认的执行方式：默认采用 streaming/batch corruption eval，只保存 metrics JSON；
不强制物化保存全部 corrupted cache。原因是 `/root/autodl-tmp` 剩余空间约 73GB，
4 centers x 5 corruptions x 5 severities 的全量副本会显著放大磁盘压力。

第一版只覆盖 4 个目标中心：

```text
ningbo
chapman_shaoxing
cpsc_2018
georgia
```

## 已有算子

当前 ECG corruption 算子来自 DeepECG/fairseq-signals 复制实现：

```text
methods/augmix/ecg_ops.py
methods/augmix/severity.py
methods/augmix/augmix.py
```

第一版使用 5 个单算子 corruption，不使用 AugMix meta mixing：

| corruption | 目的 |
|---|---|
| `powerline_noise` | 50/60 Hz 工频噪声 |
| `emg_noise` | 肌电/白噪声 |
| `baseline_wander` | 低频基线漂移 |
| `baseline_shift` | 阶梯型 DC 偏移 |
| `random_leads_masking` | missing-lead robustness |

历史归档 `trash/scripts_crosscenter_v1/ecg_augmix.py` 中的 `amplitude_scale`、`time_warp`、`lead_dropout` 等作为二期候选，不进入第一版主表。

## Severity

ImageNet-C 使用 5 档 severity。当前 ECG severity 是 1-10，映射为：

```text
public severity 1..5 -> internal severity [2, 4, 6, 8, 10]
```

注意：

- `powerline_noise` 在 100 Hz z-score 信号上加入 50/60 Hz，会发生 Nyquist/aliasing；作为 stress benchmark 可接受，不解释为真实采集链路。
- `baseline_shift` 偏强，应单独解读。
- `random_leads_masking` severity 5 平均 50% 导联清零，非常强，应单独报告。

## Cache 格式

输入 clean cache：

```text
/root/autodl-tmp/triple_labels/pn2021_eval_cache/
  super5_<center>_100hz1000_v3_super5_normsuppress.npz
```

字段：

```text
signals:    (N, 1000, 12) float32
labels:     (N, 5) float32
record_ids: (N,) str
metadata_json
```

可选输出 corrupted cache：

```text
/root/autodl-tmp/triple_labels/pn2021_c_cache/
  super5_<center>_<corruption>_s<severity>_100hz1000_v1.npz
```

字段：

```text
signals:    (N, 1000, 12) float32
labels:     (N, 5) float32
record_ids: (N,) str
metadata_json:
  scheme
  center
  class_names
  source_clean_cache_path
  corruption
  severity_public
  severity_internal
  op_params
  seed
  cache_version = pn2021_c_v1
```

## 推荐命令形态

脚本：

```text
scripts/triple_labels/build_pn2021_corruptions.py
scripts/triple_labels/eval_pn2021_corruptions.py
```

构建 corrupted cache：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/build_pn2021_corruptions.py \
  --scheme super5 \
  --clean_cache_dir /root/autodl-tmp/triple_labels/pn2021_eval_cache \
  --out_dir /root/autodl-tmp/triple_labels/pn2021_c_cache \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --corruptions powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking \
  --severities 1 2 3 4 5 \
  --seed 42
```

评测：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_pn2021_corruptions.py \
  --scheme super5 \
  --model_dir /root/autodl-tmp/triple_labels/super5 \
  --cache_dir /root/autodl-tmp/triple_labels/pn2021_c_cache \
  --centers ningbo chapman_shaoxing cpsc_2018 georgia \
  --corruptions powerline_noise emg_noise baseline_wander baseline_shift random_leads_masking \
  --severities 1 2 3 4 5 \
  --output_path /root/autodl-tmp/triple_labels/super5/eval_pn2021_c.json
```

## 输出指标

```text
clean_per_center
per_center_corruption_severity
mean_by_corruption
mean_by_severity
mean_corruption_auroc
mean_corruption_auprc
relative_drop_vs_clean
```

主表：

```text
center x corruption x severity -> AUROC/AUPRC
corruption mean drop
severity mean drop
overall PN2021-C score
```

CSV export:

```text
script: scripts/triple_labels/export_pn2021_c_summaries.py
outputs:
  /root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_corruption_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_severity_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_center_summary.csv
  /root/autodl-tmp/triple_labels/pn2021_c_per_class_summary.csv
```

Candidate coverage export:

```text
script: scripts/triple_labels/export_pn2021_c_candidate_coverage.py
output: /root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv
rows: 47 v3-super5 clean-eval checkpoints
current high-priority missing PN2021-C:
  latent_hull_super5_pilot/nin_M5_lambda025_ep20
  latent_hull_super5_pilot/extra_M5_lambda025_ep20
  latent_hull_super5_pilot/geo_M20_lambda025_ep20
optional:
  synth_anchored_super5_v13b_K400/nin_k400
```

## 实现规则

- 不把 PN2021-C 逻辑塞进 `eval_crosscenter.py` 主评测路径，避免污染 clean thesis metric。
- 先复用 clean v3 cache 或 mmap clean cache；不要重新从 raw 构建 corrupted cache。
- corrupted cache 只改变 signal，不改变 label。
- 每个 corrupted cache 固定 seed 并写 metadata。
- 不评估 `ptb-xl` / `ptbxl`。
- 默认主路径可以不保存 corrupted signals，只保存 corruption 配置、seed 和 metrics。
- 只有 smoke、可视化或复现实验需要时才物化 selected corruption cache。

## 自动执行顺序

1. 确保 clean v3 PN2021 cache 已存在。
2. 已新增并 smoke test `build_pn2021_corruptions.py`，先跑每中心小 limit。
3. 已新增并 smoke test `eval_pn2021_corruptions.py`。
4. 已改成 streaming/batch PN2021-C eval，避免全量物化 cache。
5. smoke 已通过：

```bash
/root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/eval_pn2021_corruptions.py \
  --mode stream \
  --model_dir /root/autodl-tmp/triple_labels/super5 \
  --centers ningbo \
  --corruptions powerline_noise \
  --severities 1 \
  --limit 16 \
  --min_pos 1 \
  --output_path /root/autodl-tmp/triple_labels/super5/eval_pn2021_c_stream_smoke.json
```

6. 跑 baseline checkpoint PN2021-C。
7. 后续对增强后 checkpoint 重复评测，比较 clean delta 和 corrupted delta。

## Current Baseline Result

已完成 baseline streaming PN2021-C：

```text
checkpoint: /root/autodl-tmp/triple_labels/super5/best_model.pt
output: /root/autodl-tmp/triple_labels/super5/eval_pn2021_c_stream_v1.json
mode: stream
centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
```

mean drop vs clean across the 4 centers:

| corruption | mean AUROC drop | mean AUPRC drop |
|---|---:|---:|
| powerline_noise | 0.0006 | 0.0010 |
| emg_noise | 0.0022 | 0.0036 |
| baseline_wander | 0.0154 | 0.0243 |
| baseline_shift | 0.0393 | 0.0605 |
| random_leads_masking | 0.0554 | 0.0811 |

Severity-5 mean metrics:

| corruption | AUROC | AUPRC |
|---|---:|---:|
| powerline_noise | 0.8370 | 0.5207 |
| emg_noise | 0.8336 | 0.5154 |
| baseline_wander | 0.8096 | 0.4780 |
| baseline_shift | 0.7652 | 0.4184 |
| random_leads_masking | 0.7337 | 0.3841 |

Interpretation: powerline and EMG are mild under current severity mapping; baseline
shift and lead masking are the strongest stressors and should be highlighted in
robustness tables.

## Current Enhanced Result

已完成当前 Latent-Hull AUPRC-best checkpoint 的 streaming PN2021-C：

```text
checkpoint: /root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20/best_model.pt
clean PN2021 v3: AUROC=0.8341, AUPRC=0.5560
output: /root/autodl-tmp/latent_hull_super5_pilot/nin_M10_lambda025_ep20/eval_pn2021_c_stream_v2_selfclean_all4.json
mode: stream
centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
```

Important correction:

```text
The older eval_pn2021_c_stream_v1.json used the baseline clean JSON for drop
calculation. The self-clean run above fixes this. Always use the CSV columns
mean_*_drop_vs_clean from export_pn2021_c_summaries.py, which are corrected
against each model's own eval_result_v3_super5_normsuppress.json.
```

Mean self-clean drop across the 4 centers:

| corruption | baseline drop AUROC | baseline drop AUPRC | Latent-Hull drop AUROC | Latent-Hull drop AUPRC | corrupted AUPRC baseline/LH |
|---|---:|---:|---:|---:|---:|
| powerline_noise | 0.0006 | 0.0010 | 0.0005 | 0.0009 | 0.5220 / 0.5269 |
| emg_noise | 0.0022 | 0.0036 | 0.0019 | 0.0036 | 0.5194 / 0.5242 |
| baseline_wander | 0.0154 | 0.0243 | 0.0153 | 0.0252 | 0.4987 / 0.5026 |
| baseline_shift | 0.0393 | 0.0605 | 0.0397 | 0.0626 | 0.4625 / 0.4652 |
| random_leads_masking | 0.0554 | 0.0811 | 0.0560 | 0.0833 | 0.4419 / 0.4445 |

Severity-5 mean metrics:

| corruption | baseline AUROC/AUPRC | Latent-Hull AUROC/AUPRC | Latent-Hull severity-5 drop |
|---|---:|---:|---:|
| powerline_noise | 0.8370 / 0.5207 | 0.8386 / 0.5256 | -0.0003 / -0.0026 |
| emg_noise | 0.8336 / 0.5154 | 0.8357 / 0.5199 | 0.0026 / 0.0031 |
| baseline_wander | 0.8096 / 0.4780 | 0.8110 / 0.4813 | 0.0273 / 0.0417 |
| baseline_shift | 0.7652 / 0.4184 | 0.7658 / 0.4204 | 0.0725 / 0.1026 |
| random_leads_masking | 0.7337 / 0.3841 | 0.7340 / 0.3861 | 0.1043 / 0.1369 |

Interpretation:

- The Latent-Hull checkpoint improves clean PN2021 AUPRC and has the highest
  absolute corrupted mean AUPRC among evaluated checkpoints (`0.492684`).
- It does not reduce mean self-clean corruption drop versus the baseline
  (`0.035135` vs `0.034103` AUPRC drop). The earlier drop-improvement claim came
  from comparing against the baseline clean JSON.
- The hard stressors remain `baseline_shift` and `random_leads_masking`; the
  method improves their absolute corrupted AUPRC but not their relative drop.

## Prompt-Token Pilot Result

已完成 `ningbo` prompt-token gated pool pilot 的 streaming PN2021-C：

```text
checkpoint: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20/best_model.pt
clean PN2021 v3: AUROC=0.8342, AUPRC=0.5555
output: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v1/ningbo_pool150_gated97_M10_ep20/eval_pn2021_c_stream_v1.json
mode: stream
centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
```

Mean drop vs clean across the 4 centers:

| corruption | baseline mean drop AUROC | baseline mean drop AUPRC | prompt-token mean drop AUROC | prompt-token mean drop AUPRC | drop delta |
|---|---:|---:|---:|---:|---:|
| powerline_noise | 0.0006 | 0.0010 | 0.0005 | 0.0009 | -0.0001 / -0.0000 |
| emg_noise | 0.0022 | 0.0036 | 0.0019 | 0.0037 | -0.0002 / +0.0001 |
| baseline_wander | 0.0154 | 0.0243 | 0.0154 | 0.0252 | -0.0000 / +0.0009 |
| baseline_shift | 0.0393 | 0.0605 | 0.0399 | 0.0624 | +0.0006 / +0.0018 |
| random_leads_masking | 0.0554 | 0.0811 | 0.0559 | 0.0830 | +0.0005 / +0.0019 |

Interpretation:

- Prompt-token gated pilot improves clean PN2021 AUPRC but does not yet reduce
  corruption drops consistently.
- The real-anchor Latent-Hull checkpoint remains stronger on absolute corrupted
  AUPRC, but it is not better on self-clean relative drop.
- Next prompt-token robustness attempt should first improve gated pool balance,
  especially STTC, before running heavier PN2021-C comparisons.

已完成当前最佳 prompt-token v4 balanced checkpoint 的 streaming PN2021-C：

```text
checkpoint: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20/best_model.pt
clean PN2021 v3: AUROC=0.8344, AUPRC=0.5558
output: /root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v2/ningbo_pool250_gated131_M10_ep20/eval_pn2021_c_stream_v2_all4.json
mode: stream
centers: ningbo, chapman_shaoxing, cpsc_2018, georgia
```

Mean drop vs clean across the 4 centers:

| corruption | baseline mean drop AUROC | baseline mean drop AUPRC | real-anchor mean drop AUROC | real-anchor mean drop AUPRC | prompt-v4 mean drop AUROC | prompt-v4 mean drop AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| powerline_noise | 0.0006 | 0.0010 | -0.0009 | -0.0039 | 0.0005 | 0.0009 |
| emg_noise | 0.0022 | 0.0036 | 0.0005 | -0.0012 | 0.0019 | 0.0037 |
| baseline_wander | 0.0154 | 0.0243 | 0.0139 | 0.0204 | 0.0156 | 0.0254 |
| baseline_shift | 0.0393 | 0.0605 | 0.0383 | 0.0578 | 0.0400 | 0.0627 |
| random_leads_masking | 0.0554 | 0.0811 | 0.0546 | 0.0785 | 0.0558 | 0.0831 |

Interpretation:

- Prompt-token v4 is useful for clean AUPRC, but it does not currently improve
  PN2021-C robustness.
- Real-anchor Latent-Hull remains stronger on absolute corrupted AUPRC, but not
  on self-clean relative drop.
- Do not spend more PN2021-C time on prompt-token checkpoints until a new
  generation pool beats v4 clean AUPRC or has a clear robustness mechanism.

## Current Full Comparison

After user request, PN2021-C was also run for the latest prompt-token v31 mixed
diverse checkpoint, the C2 Dirichlet coefficient control, the lambda=0.40
Latent-Hull control, the high-priority missing Latent-Hull candidates
`nin_M5`, `extra_M5`, and `geo_M20`, plus optional `synth_v13b_K400/nin_k400`
and prompt-token v14. The full-result CSV export includes 12 PN2021-C JSONs:

```text
script: scripts/triple_labels/export_pn2021_c_summaries.py
included: 12 full PN2021-C JSONs
run summary: /root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
```

Self-clean mean drop ranking, lower AUPRC drop is better:

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

Interpretation:

- Prompt-token v14 is now the best absolute corrupted-AUPRC checkpoint
  (`0.492808`), followed by synth-anchor K400 (`0.492707`) and real-anchor
  `nin_M10` (`0.492684`).
- None of the enhanced models reduces self-clean AUPRC drop below the baseline;
  report them as improving absolute corrupted performance, not reducing
  relative corruption sensitivity.
- `geo_M20_lambda025` has the lowest self-clean AUPRC drop among enhanced
  checkpoints, but its absolute corrupted AUPRC is lower than `nin_M10`, prompt
  v4, `extra_M5`, and `nin_M5`; treat it as a robustness diagnostic, not the
  main result.
- Prompt-token v31 has a slightly lower mean AUPRC drop than prompt-token v4,
  but its clean PN2021 AUPRC is lower (`0.5550` vs `0.5558`), so it is not a
  better center-token result overall.
- C2 Dirichlet and lambda=0.40 do not beat the optimized `lambda=0.25` main run.

## Completed 1-2h Fill-In Queue

为继续利用算力，author-repro partial run 清理后自动选择本 benchmark 中
最有价值且预计 1-2 小时内完成的缺口：补跑 clean AUPRC 接近主结果、但还缺
PN2021-C self-clean 结果的 Latent-Hull 候选。

脚本：

```bash
TMPDIR=/root/autodl-tmp/tmp XDG_CACHE_HOME=/root/autodl-tmp/cache \
bash scripts/triple_labels/run_selected_pn2021_c_queue.sh
```

候选：

| model dir | clean AUPRC | reason |
|---|---:|---|
| `/root/autodl-tmp/latent_hull_super5_pilot/nin_M5_lambda025_ep20` | 0.555731 | M=5 close to best clean utility; missing PN2021-C |
| `/root/autodl-tmp/latent_hull_super5_pilot/extra_M5_lambda025_ep20` | 0.555715 | extra-center anchor close to best clean utility; missing PN2021-C |
| `/root/autodl-tmp/latent_hull_super5_pilot/geo_M20_lambda025_ep20` | 0.555571 | larger M cross-center candidate; missing PN2021-C |

Actual outputs:

```text
/root/autodl-tmp/latent_hull_super5_pilot/nin_M5_lambda025_ep20/eval_pn2021_c_stream_v1_selfclean_all4.json
/root/autodl-tmp/latent_hull_super5_pilot/extra_M5_lambda025_ep20/eval_pn2021_c_stream_v1_selfclean_all4.json
/root/autodl-tmp/latent_hull_super5_pilot/geo_M20_lambda025_ep20/eval_pn2021_c_stream_v1_selfclean_all4.json
/root/autodl-tmp/synth_anchored_super5_v13b_K400/nin_k400/eval_pn2021_c_stream_v1_selfclean_all4.json
/root/autodl-tmp/ecgtwin_prompt_token_super5/online_at_pilot_v14/ningbo_v27_seed42_direct_mv4_balanced_q50_M10_ep20_seed03/eval_pn2021_c_stream_v1_selfclean_all4.json
```

完成后自动刷新：

```text
/root/autodl-tmp/triple_labels/pn2021_c_run_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_corruption_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_severity_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_center_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_per_class_summary.csv
/root/autodl-tmp/triple_labels/pn2021_c_candidate_coverage.csv
```
