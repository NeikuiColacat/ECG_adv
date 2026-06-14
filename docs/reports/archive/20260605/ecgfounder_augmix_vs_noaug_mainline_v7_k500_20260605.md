# ECGFounder VAE-LHAT AugMix vs NoAug, v7 K500

Run ID: `mainline_v7_k500_4gpu_20260605`

Scope: ECGFounder residual-adapter, four target centers, PN2021 v7 SJR/RGQ ref-excluded clean eval. Primary view is all-zero kept.

## Four-Center Mean

|View|Direct|VAE noAug|VAE AugMix|AugMix - noAug|AugMix - direct|
|---|---:|---:|---:|---:|---:|
|AUROC|0.897254|0.903117|0.903672|+0.055 pp|+0.642 pp|
|AUPRC|0.632539|0.646974|0.645512|-0.146 pp|+1.297 pp|

## Drop-All-Zero Mean

|View|VAE noAug|VAE AugMix|AugMix - noAug|
|---|---:|---:|---:|
|AUROC|0.921051|0.922884|+0.183 pp|
|AUPRC|0.756513|0.759114|+0.260 pp|

## Per Center

|Center|Direct|VAE noAug|VAE AugMix|AugMix - noAug|AugMix best epoch|
|---|---:|---:|---:|---:|---:|
|ningbo|0.917938/0.600789|0.917938/0.600789|0.917938/0.600789|+0.000/+0.000 pp|0|
|chapman_shaoxing|0.926087/0.584853|0.926087/0.584853|0.926087/0.584853|+0.000/+0.000 pp|0|
|cpsc_2018|0.862972/0.621943|0.887006/0.681460|0.888954/0.674982|+0.195/-0.648 pp|6|
|georgia|0.882018/0.722574|0.881438/0.720795|0.881708/0.721425|+0.027/+0.063 pp|1|

Interpretation: AugMix is not a replacement for the current ECGFounder noAug VAE-LHAT result under the primary all-zero-kept AUPRC view. It gives a tiny AUROC gain and a small drop-all-zero gain, but lowers all-zero-kept AUPRC because CPSC drops and Ningbo/Chapman select epoch 0/direct fallback.

CSV: `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/ECG_adv_Gen/docs/reports/archive/20260605/ecgfounder_augmix_vs_noaug_mainline_v7_k500_20260605.csv`
