# Mainline v7 K500 4-GPU Results Summary

Run ID: `mainline_v7_k500_4gpu_20260605`

Scope: target-center ref-excluded PN2021 clean evaluation on `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.

Mapping: `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.

Primary view below is all-zero kept, because it is available for every direct/VAE pair including ECGFounder.

## Four-Center Mean, All-Zero Kept

|Model|Direct AUROC|Direct AUPRC|VAE-LHAT AUROC|VAE-LHAT AUPRC|Delta AUROC pp|Delta AUPRC pp|
|---|---|---|---|---|---|---|
|EfficientNet1DV2|0.852149|0.524658|0.871208|0.554857|+1.906|+3.020|
|ECGFounder|0.897254|0.632539|0.903117|0.646974|+0.586|+1.443|
|ResNet1D-Wang|0.856688|0.533606|0.865479|0.553246|+0.879|+1.964|
|Inception1D|0.866703|0.557015|0.872853|0.569428|+0.615|+1.241|
|FCN-Wang|0.848012|0.518475|0.853406|0.529659|+0.539|+1.118|

## Drop-All-Zero View, When Available

ECGFounder direct did not record drop-all-zero target-view metrics in this run, so its drop-all-zero direct-vs-VAE delta is not reported here.

|Model|Direct n|Direct AUROC|Direct AUPRC|VAE n|VAE AUROC|VAE AUPRC|Delta AUROC pp|Delta AUPRC pp|
|---|---|---|---|---|---|---|---|---|
|EfficientNet1DV2|4/4|0.876274|0.666627|4/4|0.899645|0.717712|+2.337|+5.109|
|ECGFounder|0/4|NA|NA|4/4|0.921051|0.756513|NA|NA|
|ResNet1D-Wang|4/4|0.881341|0.687305|4/4|0.891040|0.711455|+0.970|+2.415|
|Inception1D|4/4|0.893365|0.709152|4/4|0.899435|0.727152|+0.607|+1.800|
|FCN-Wang|4/4|0.874814|0.670232|4/4|0.881002|0.688493|+0.619|+1.826|

## Strongest Per-Center VAE Gains By AUPRC

|Model|Center|Direct AUPRC|VAE AUPRC|Delta AUPRC pp|Delta AUROC pp|
|---|---|---|---|---|---|
|EfficientNet1DV2|cpsc_2018|0.559339|0.621378|+6.204|+5.511|
|ECGFounder|cpsc_2018|0.621943|0.681460|+5.952|+2.403|
|EfficientNet1DV2|chapman_shaoxing|0.445541|0.484273|+3.873|+1.194|
|Inception1D|cpsc_2018|0.598798|0.637056|+3.826|+2.263|
|FCN-Wang|cpsc_2018|0.556089|0.593491|+3.740|+2.199|
|ResNet1D-Wang|cpsc_2018|0.589417|0.623774|+3.436|+2.079|
|ResNet1D-Wang|georgia|0.589615|0.615198|+2.558|+0.974|
|ResNet1D-Wang|chapman_shaoxing|0.450121|0.468743|+1.862|+0.463|

## Largest Per-Center VAE Drops By AUPRC

|Model|Center|Direct AUPRC|VAE AUPRC|Delta AUPRC pp|Delta AUROC pp|
|---|---|---|---|---|---|
|ECGFounder|georgia|0.722574|0.720795|-0.178|-0.058|
|ECGFounder|ningbo|0.600789|0.600789|+0.000|+0.000|
|ECGFounder|chapman_shaoxing|0.584853|0.584853|+0.000|+0.000|
|ResNet1D-Wang|ningbo|0.505269|0.505269|+0.000|+0.000|
|Inception1D|ningbo|0.529308|0.529308|+0.000|+0.000|
|Inception1D|chapman_shaoxing|0.489719|0.489719|+0.000|+0.000|
|FCN-Wang|ningbo|0.497673|0.497673|+0.000|+0.000|
|FCN-Wang|chapman_shaoxing|0.428090|0.428090|+0.000|+0.000|

## Files

- Per-center metrics: `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/ECG_adv_Gen/docs/reports/archive/20260605/mainline_v7_k500_4gpu_per_center_metrics.csv`
- Direct-vs-VAE mean summary: `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/ECG_adv_Gen/docs/reports/archive/20260605/mainline_v7_k500_4gpu_direct_vs_vae_summary.csv`
- Per-center direct-vs-VAE deltas: `/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp/ECG_adv_Gen/docs/reports/archive/20260605/mainline_v7_k500_4gpu_per_center_direct_vs_vae_delta.csv`
