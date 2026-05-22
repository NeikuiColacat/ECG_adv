# VAE-only v5 标签策略 all-zero 指标复核

- 生成时间：2026-05-22 13:04:00 UTC
- PN2021 Super5 mapping：`v5_super5_strict_voltage_pacing_suppress_20260522`，hash `1141e0a9f94b`
- 模型：PTB-XL full baseline vs VAE-only real-anchor LH-AT K=500。
- 评估：target center 指标；每个 target center 评估时排除对应 K=500 适配样本，避免测试污染。
- 输入协议：minimal_resample、per_sample_global、100Hz、1000 samples。

## 口径说明

- `all-zero kept`：Super5 全 0 样本作为五个类别的负样本参与 AUROC/AUPRC。
- `drop all-zero`：先删除 Super5 全 0 样本，只在至少一个 Super5 阳性的记录上计算指标。

## all-zero kept

| center | baseline | VAE-only | delta | eval n | all-zero n |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.8718 / 0.4222 | 0.8880 / 0.4393 | +1.62pp / +1.71pp | 34405 | 17632 |
| chapman_shaoxing | 0.8805 / 0.3458 | 0.8949 / 0.3720 | +1.44pp / +2.63pp | 9747 | 5102 |
| cpsc_2018 | 0.8115 / 0.5586 | 0.8545 / 0.5962 | +4.30pp / +3.76pp | 6377 | 2131 |
| georgia | 0.8168 / 0.5913 | 0.8266 / 0.6012 | +0.98pp / +0.99pp | 9844 | 1666 |
| **4-center avg** | **0.8452 / 0.4795** | **0.8660 / 0.5022** | **+2.08pp / +2.27pp** |  |  |

## drop all-zero

| center | baseline | VAE-only | delta | eval n | all-zero n |
|---|---:|---:|---:|---:|---:|
| ningbo | 0.9000 / 0.6072 | 0.9149 / 0.6234 | +1.49pp / +1.61pp | 16773 | 17632 |
| chapman_shaoxing | 0.9107 / 0.5366 | 0.9224 / 0.5536 | +1.17pp / +1.70pp | 4645 | 5102 |
| cpsc_2018 | 0.8627 / 0.6950 | 0.9083 / 0.7741 | +4.56pp / +7.91pp | 4246 | 2131 |
| georgia | 0.8254 / 0.6753 | 0.8351 / 0.6863 | +0.97pp / +1.11pp | 8178 | 1666 |
| **4-center avg** | **0.8747 / 0.6285** | **0.8952 / 0.6593** | **+2.05pp / +3.08pp** |  |  |

## 我的判断

主报告建议保留 `all-zero kept`。这些样本在 v5 中表示“没有足够证据归入 CD/HYP/MI/NORM/STTC，或属于 Super5 外异常”，真实部署时模型仍会遇到它们；把它们作为五类负样本可以测试模型是否会把 Super5 外记录误报成目标类。

`drop all-zero` 可以作为敏感性分析。它回答的是“只在 Super5 覆盖范围内，模型区分五类的能力如何”，但会排除大量 PN2021 真实外部样本，外部有效性更弱，不建议作为主指标。

## 产物

- CSV：`/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/vae_only_v5_allzero_eval.csv`
- JSON 目录：`/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals`
- chapman_shaoxing baseline: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/chapman_shaoxing_baseline_v5_allzero_views.json`
- chapman_shaoxing vae_only: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/chapman_shaoxing_vae_only_v5_allzero_views.json`
- cpsc_2018 baseline: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/cpsc_2018_baseline_v5_allzero_views.json`
- cpsc_2018 vae_only: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/cpsc_2018_vae_only_v5_allzero_views.json`
- georgia baseline: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/georgia_baseline_v5_allzero_views.json`
- georgia vae_only: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/georgia_vae_only_v5_allzero_views.json`
- ningbo baseline: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/ningbo_baseline_v5_allzero_views.json`
- ningbo vae_only: `/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522/evals/ningbo_vae_only_v5_allzero_views.json`
